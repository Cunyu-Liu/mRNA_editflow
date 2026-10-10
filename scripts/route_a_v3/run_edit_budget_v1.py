#!/usr/bin/env python3
"""MEF edit-budget runner v1 (2026-10-10). Frozen per edit_budget_mini_prereg_v1.md.

Greedy full-enumeration edit-budget test for the 5 ported matrix families
(scorers reused verbatim from benchmark_v2_matrix/family_adapters_v1.build_scorer).

Data caliber v2 (2026-10-10, before any full run): rows are read from the CANONICAL
records + development manifest VALIDATION split (identical to the matrix runner
load_task_records), so sequences use the canonical T alphabet — same surface the
frozen matrix_v2 numbers were computed on. The projections validation.jsonl (U
alphabet) is NOT used (alphabet mismatch with 3/5 adapters).

Protocol (frozen):
  - task rows: polyA (GSE269595, n=2628, B_max=23) and MRL (GSE114002, n=730, B_max=3)
  - for each (source, measured-candidate) VALIDATION record:
      target = record.direction_normalized_delta
      start from source; each step enumerate all 4L-1 single-nt mutants of the
      CURRENT sequence; pick argmax pred_delta (pred(x)=scorer(x)-scorer(source));
      stop when pred_delta(cur) >= target, or when steps > B_max (FAIL_budget).
  - tie-break: lexicographically smallest mutant sequence (frozen).
  - metrics per (family, task): median/mean steps (passing only), FAIL_budget rate,
    budget ratio k_model/k_true per record (median + IQR), plus per-record JSON.

Outputs (append-only, /mnt):
  <out>/edit_budget_results_v1.json      — full per-record + summary
  <out>/progress.json                     — resumable checkpoint (per family/task)
CUDA hard gate: cpu_fallback must stay false; torch.cuda required.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

W0 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
sys.path.insert(0, str(W0 / "scripts" / "route_a_v3"))
sys.path.insert(0, str(W0 / "scripts" / "route_a_v3" / "benchmark_v2_matrix"))
from family_adapters_v1 import build_scorer  # noqa: E402  (region-aware: GEMORNA score_region)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
OUT_ROOT = MNT / "experiments/analysis_edit_budget_v1"

TASKS = {
    "polya": {"study": "GSE269595", "region": "3UTR",
              "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", "b_max": 23, "n_rows": 2628},
    "mrl": {"study": "GSE114002", "region": "5UTR",
            "endpoint": "MEAN_RIBOSOME_LOAD", "b_max": 3, "n_rows": 730},
}
BASES_T = ["A", "C", "G", "T"]
FAMILIES = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]
BASES = BASES_T


def load_task_rows(task):
    """Canonical VALIDATION rows for the task — identical surface to the matrix runner."""
    t = TASKS[task]
    key = (t["study"], t["region"], t["endpoint"])
    val_ids = set()
    with open(MANIFEST) as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "VALIDATION":
                continue
            strat = row.get("stratum")
            study = str(row.get("study_unit_id", ""))
            if study == t["study"]:
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
    assert len(rows) == t["n_rows"], f"{task}: got {len(rows)} expected {t['n_rows']}"
    return rows


def mutants(seq):
    """All 4L-1 single-nt mutants, deterministic order (pos, base then lexicographic)."""
    out = []
    for i, orig in enumerate(seq):
        for b in BASES:
            if b != orig:
                out.append(seq[:i] + b + seq[i + 1:])
    return out


def greedy_budget(scorer, source, target, b_max, batch=256):
    """One (source, target) test. Returns dict with steps / fail / trajectory tail."""
    src_score = float(scorer([source])[0])
    cur = source
    cur_score = src_score
    # pass at step 0 (already at target)?
    if cur_score - src_score >= target:
        return {"steps": 0, "fail": False, "reached_pred_delta": 0.0}
    for step in range(1, b_max + 1):
        cands = mutants(cur)
        # dedupe not needed (4L-1 distinct); batch score
        scores = []
        for i in range(0, len(cands), batch):
            scores.append(scorer(cands[i:i + batch]))
        scores = np.concatenate(scores) if scores else np.array([])
        deltas = scores - cur_score
        best = float(np.max(deltas))
        # tie-break: lexicographically smallest among argmax
        idxs = np.flatnonzero(deltas >= best - 1e-12)
        chosen = min((cands[j] for j in idxs), key=str)
        # must strictly change sequence; if best <= 0 no improving mutant -> keep best anyway (frozen: argmax regardless)
        cur = chosen
        cur_score = float(scorer([chosen])[0])
        pred_delta = cur_score - src_score
        if pred_delta >= target:
            return {"steps": step, "fail": False, "reached_pred_delta": pred_delta}
    return {"steps": b_max, "fail": True, "reached_pred_delta": cur_score - src_score}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=0, help="0 = full; >0 = first N records (sanity)")
    ap.add_argument("--gpu", default="6")
    args = ap.parse_args()

    # CUDA hard gate
    assert torch.cuda.is_available(), "CUDA unavailable — hard stop (cpu_fallback forbidden)"
    device = f"cuda:{args.gpu}"
    print(f"[gate] cuda ok device={device} name={torch.cuda.get_device_name(device)}", flush=True)

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_task_rows(args.task)
    if args.limit:
        rows = rows[: args.limit]
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
    print(f"[init-scorer] {region_note}", flush=True)
    print(f"[init] family={args.family} task={args.task} n={len(rows)} B_max={b_max}", flush=True)
    print(f"[meta] {meta.get('family', args.family)} weights sha={meta.get('weights_sha256', 'n/a')[:16]}", flush=True)

    t0 = time.time()
    records = []
    n_pass = 0
    steps_pass = []
    ratios = []
    for i, r in enumerate(rows):
        source = r["source_sequence"]
        target = float(r["direction_normalized_delta"])
        k_true = sum(1 for op in r.get("edit_operations", []) if op.get("type") == "SUB")
        res = greedy_budget(scorer, source, target, b_max)
        rec = {
            "canonical_record_id": r["canonical_record_id"],
            "source_group_id": r.get("source_id", ""),
            "k_true": k_true,
            "target_delta": target,
            "steps": res["steps"],
            "fail": res["fail"],
            "reached_pred_delta": round(res["reached_pred_delta"], 6),
        }
        records.append(rec)
        if not res["fail"]:
            n_pass += 1
            steps_pass.append(res["steps"])
            if k_true > 0:
                ratios.append(res["steps"] / k_true)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[prog] {i+1}/{len(rows)} pass={n_pass} elapsed={el:.0f}s "
                  f"rate={(i+1)/el:.2f} rec/s eta={(len(rows)-i-1)/max((i+1)/el,1e-9):.0f}s", flush=True)

    summary = {
        "family": args.family,
        "task": args.task,
        "n": len(rows),
        "b_max": b_max,
        "n_pass": n_pass,
        "fail_budget_rate": round(1 - n_pass / len(rows), 4) if rows else None,
        "median_steps_pass": st.median(steps_pass) if steps_pass else None,
        "mean_steps_pass": round(st.mean(steps_pass), 3) if steps_pass else None,
        "median_budget_ratio": round(st.median(ratios), 3) if ratios else None,
        "budget_ratio_q1": round(st.quantiles(ratios, n=4)[0], 3) if len(ratios) >= 4 else None,
        "budget_ratio_q3": round(st.quantiles(ratios, n=4)[2], 3) if len(ratios) >= 4 else None,
        "elapsed_s": round(time.time() - t0, 1),
        "device": device,
        "cpu_fallback_used": False,
        "scorer_meta": {k: v for k, v in meta.items() if k in ("family", "weights_sha256", "license", "input_adaptation")},
        "prereg": "docs/paper/edit_budget_mini_prereg_v1.md (commit e26ac88a)",
        "scorer_region": region_note,
    }
    payload = {"summary": summary, "records": records}
    tag = f"{args.family}_{args.task}"
    if args.limit:
        tag += f"_sanity{args.limit}"
    out = out_dir / f"edit_budget_{tag}_v1.json"
    if out.exists():
        print(f"[skip] {out} already exists (append-only policy) — nothing to do", flush=True)
        return 0
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[done] wrote {out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
