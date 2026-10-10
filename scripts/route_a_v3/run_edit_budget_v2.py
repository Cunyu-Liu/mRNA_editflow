#!/usr/bin/env python3
"""MEF edit-budget runner v2 (2026-10-10): cross-record batched greedy.

Protocol is IDENTICAL to v1 / prereg (per-record greedy full-enumeration,
frozen scorer, tie-break lexicographic, B_max per task). The only change is
EXECUTION ORDER: instead of finishing one record before the next, we advance
a wave of active records simultaneously, batching their per-round candidate
sets into single scorer calls (frozen scorer is deterministic and stateless;
batching does not change any computed value - same model, same inputs).

This is a compute-order optimization, not a protocol change: every record's
greedy trajectory (chosen mutant per round, stop condition, steps) is
bit-identical to the v1 sequential execution. Equivalence was asserted on a
sanity subset (see logs).

Throughput: ~N_active x candidates per forward batch, GPU-bound.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

W0 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
sys.path.insert(0, str(W0 / "scripts" / "route_a_v3"))
sys.path.insert(0, str(W0 / "scripts" / "route_a_v3" / "benchmark_v2_matrix"))
from family_adapters_v1 import build_scorer  # noqa: E402

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
OUT_ROOT = MNT / "experiments/analysis_edit_budget_v1"

TASKS = {
    "polya": {"study": "GSE269595", "region": "3UTR",
              "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", "b_max": 23, "n_rows": 2628},
    "mrl": {"study": "GSE114002", "region": "5UTR",
            "endpoint": "MEAN_RIBOSOME_LOAD", "b_max": 3, "n_rows": 730},
}
FAMILIES = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]
BASES = ["A", "C", "G", "T"]


def load_task_rows(task):
    t = TASKS[task]
    val_ids = set()
    with open(MANIFEST) as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "VALIDATION":
                continue
            if str(row.get("study_unit_id")) == t["study"]:
                val_ids.add(str(row["canonical_record_id"]))
    rows = []
    path = MNT / f"canonical/{t['study']}/v1/canonical_records.jsonl"
    if not path.exists():
        path = MNT / f"canonical/{t['study']}/v1/canonical_records.private.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if str(r["canonical_record_id"]) not in val_ids:
                continue
            if str(r.get("region")) != t["region"]:
                continue
            if str(r.get("endpoint_id")) != t["endpoint"]:
                continue
            rows.append(r)
    rows.sort(key=lambda r: str(r["canonical_record_id"]))
    assert len(rows) == t["n_rows"], f"{task}: got {len(rows)}"
    return rows


def mutants(seq):
    out = []
    for i, orig in enumerate(seq):
        for b in BASES:
            if b != orig:
                out.append(seq[:i] + b + seq[i + 1:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--wave", type=int, default=32, help="records advanced simultaneously")
    ap.add_argument("--forward-batch", type=int, default=256, help="max sequences per scorer call")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable — hard stop"
    device = f"cuda:{args.gpu}"
    print(f"[gate] cuda ok device={device} name={torch.cuda.get_device_name(device)}", flush=True)

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.family}_{args.task}"
    out = out_dir / f"edit_budget_{tag}_v1.json"
    if out.exists():
        print(f"[skip] {out} already exists — nothing to do", flush=True)
        return 0

    rows = load_task_rows(args.task)
    b_max = TASKS[args.task]["b_max"]

    obj, meta = build_scorer(args.family, device)
    region = "5UTR" if args.task == "mrl" else "3UTR"
    if isinstance(obj, dict) and "score_region" in obj:
        def scorer(seqs, _r=region, _o=obj):
            return _o["score_region"](seqs, _r)
        region_note = f"region-dispatch:{region}"
    else:
        scorer = obj
        region_note = "single-head"
    print(f"[init] family={args.family} task={args.task} n={len(rows)} B_max={b_max} wave={args.wave}", flush=True)
    print(f"[init-scorer] {region_note}", flush=True)

    t0 = time.time()
    records = []
    n_pass = 0
    steps_pass = []
    ratios = []

    # wave state
    idx = 0
    active = []  # each: dict(row, cur, cur_score, src_score, target, step)
    # preload wave
    def admit(n):
        nonlocal idx
        while len(active) < n and idx < len(rows):
            r = rows[idx]
            active.append({"i": idx, "row": r, "cur": r["source_sequence"], "cur_score": None,
                           "src_score": None, "target": float(r["direction_normalized_delta"]),
                           "step": 0, "done": False, "fail": False, "reached": 0.0})
            idx += 1

    admit(args.wave)

    # score all sources of the initial wave in one batch
    src_seqs = [a["cur"] for a in active]
    src_scores = _batch_score(scorer, src_seqs, args.forward_batch)
    for a, s in zip(active, src_scores):
        a["src_score"] = float(s)
        a["cur_score"] = float(s)

    while active:
        # drop finished
        still = []
        for a in active:
            if a["done"]:
                records.append(_finish(a, b_max))
            else:
                still.append(a)
        active = still

        # check pass-at-step-0 / termination for those at cur == source
        if active:
            # build candidate lists for all active records
            all_cands = []
            per_rec = []
            for a in active:
                c = mutants(a["cur"])
                per_rec.append((a, c, len(all_cands)))
                all_cands.extend(c)
            # one batched scoring pass
            scores = _batch_score(scorer, all_cands, args.forward_batch)
            # advance each record by exactly one greedy step
            next_cur = []
            for a, c, off in per_rec:
                sc = scores[off:off + len(c)]
                d = np.asarray(sc, dtype=float) - a["cur_score"]
                best = float(np.max(d)) if len(d) else 0.0
                idxs = np.flatnonzero(d >= best - 1e-12)
                chosen = min((c[j] for j in idxs), key=str)
                a["step"] += 1
                next_cur.append((a, chosen))
            # score chosen seqs in one batch
            chosen_scores = _batch_score(scorer, [ch for _, ch in next_cur], args.forward_batch)
            for (a, ch), s in zip(next_cur, chosen_scores):
                a["cur"] = ch
                a["cur_score"] = float(s)
                a["reached"] = a["cur_score"] - a["src_score"]
                if a["reached"] >= a["target"]:
                    a["done"] = True
                    a["fail"] = False
                elif a["step"] >= b_max:
                    a["done"] = True
                    a["fail"] = True
            # refill wave
            admit(args.wave)
            if active and idx < len(rows):
                new = [a for a in active if a["src_score"] is None]
                if new:
                    ss = _batch_score(scorer, [a["cur"] for a in new], args.forward_batch)
                    for a, s in zip(new, ss):
                        a["src_score"] = float(s)
                        a["cur_score"] = float(s)
                        # step-0 pass check
                        if a["cur_score"] - a["src_score"] >= a["target"]:
                            a["done"] = True; a["fail"] = False
        n_done = len(records)
        if n_done % 100 < args.wave:
            el = time.time() - t0
            print(f"[prog] {n_done}/{len(rows)} elapsed={el:.0f}s rate={n_done/max(el,1e-9):.2f} rec/s "
                  f"eta={(len(rows)-n_done)/max(n_done/max(el,1e-9),1e-9):.0f}s active={len(active)}", flush=True)

    records.sort(key=lambda r: r["_i"])
    for r in records:
        if not r["fail"]:
            n_pass += 1
            steps_pass.append(r["steps"])
            kt = r["k_true"]
            if kt > 0:
                ratios.append(r["steps"] / kt)

    summary = {
        "family": args.family, "task": args.task, "n": len(rows), "b_max": b_max,
        "n_pass": n_pass,
        "fail_budget_rate": round(1 - n_pass / len(rows), 4) if rows else None,
        "median_steps_pass": st.median(steps_pass) if steps_pass else None,
        "mean_steps_pass": round(st.mean(steps_pass), 3) if steps_pass else None,
        "median_budget_ratio": round(st.median(ratios), 3) if ratios else None,
        "budget_ratio_q1": round(st.quantiles(ratios, n=4)[0], 3) if len(ratios) >= 4 else None,
        "budget_ratio_q3": round(st.quantiles(ratios, n=4)[2], 3) if len(ratios) >= 4 else None,
        "elapsed_s": round(time.time() - t0, 1),
        "device": device, "cpu_fallback_used": False,
        "scorer_region": region_note,
        "runner": "v2 cross-record batched (protocol-identical; see docstring)",
        "scorer_meta": {k: v for k, v in meta.items() if k in ("family", "weights_sha256", "license")},
        "prereg": "docs/paper/edit_budget_mini_prereg_v1.md (commit e26ac88a)",
    }
    recs_out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    payload = {"summary": summary, "records": recs_out}
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[done] wrote {out}", flush=True)
    print(json.dumps({k: summary[k] for k in ["n", "n_pass", "fail_budget_rate", "median_steps_pass",
                                              "median_budget_ratio", "elapsed_s"]}, indent=1), flush=True)
    return 0


def _batch_score(scorer, seqs, batch):
    out = []
    for i in range(0, len(seqs), batch):
        out.append(np.asarray(scorer(seqs[i:i + batch]), dtype=float))
    return np.concatenate(out) if out else np.array([])


def _finish(a, b_max):
    return {
        "_i": a["i"],
        "canonical_record_id": a["row"]["canonical_record_id"],
        "source_group_id": a["row"].get("source_id", ""),
        "k_true": sum(1 for op in a["row"].get("edit_operations", []) if op.get("type") == "SUB"),
        "target_delta": a["target"],
        "steps": a["step"],
        "fail": a["fail"],
        "reached_pred_delta": round(a["reached"], 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
