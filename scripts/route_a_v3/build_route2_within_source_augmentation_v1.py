#!/usr/bin/env python3
"""D16-C within-source data augmentation builder (amendment v1, executor checklist step 2).

Protocol (frozen, route2_critic_d16c_within_source_amendment_v1.md sec.3):
- per TRAIN source: sample structure keys (n_edits x window-span block, D15-2 protocol)
  from that source's measured structure profile; random positions within window and
  random replacement bases; fill to 32 candidates/source (measured kept, synthetic ~27/src)
- H1 HARD CLAUSE: synthetic pairs carry NO regression label (no direction_normalized_delta,
  no any model-predicted value). They enter training only as structure-augmentation members
  with loss weight 0 (or InfoNCE structure channel per H2 - NOT implemented here, only H1).
- leakage audit: synthetic candidate sequences exact-matched against VALIDATION measured
  candidates; flagged > 0 -> that synthetic pair is dropped (D2 audit protocol reuse).
- determinism: seed 20260913 fixed; per-source generation trace in synthetic_trace.jsonl.
- output schema keeps all projection fields for measured rows; synthetic rows carry
  synthetic=true, structure_key, trace_seed, and NO label fields.

Only MRL P0 arm in this build (per amendment sec.2 priority table; HL/MPRAU/TE/REFALT arms
queued behind G1 gate). polyA untouched (reference group).
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ_TRAIN = MNT / "projections/xedit_v3/development_train_validation_v1/train.jsonl"
PROJ_VAL = MNT / "projections/xedit_v3/development_train_validation_v1/validation.jsonl"
OUT_DIR = MNT / "experiments/xeditcritic_d16c"
BASES = "ACGT"
SEED = 20260913
TARGET_PER_SOURCE = 32
TASK_PREFIXES = {
    "MRL": "MEAN_RIBOSOME_LOAD",
}
WINDOW_BLOCK = 8
SYNTH_CAP = 5


def struct_key_of_edits(positions):
    if not positions:
        return None
    return (len(positions), positions[0] // WINDOW_BLOCK, positions[-1] // WINDOW_BLOCK)


def load_task_rows(task_prefix):
    rows = []
    with open(PROJ_TRAIN) as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id", "").startswith(task_prefix):
                rows.append(d)
    return rows


def load_validation_measured(task_prefix):
    val = set()
    with open(PROJ_VAL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id", "").startswith(task_prefix):
                val.add(str(d["candidate_sequence"]).upper())
                val.add(str(d["source_sequence"]).upper())
    return val


def norm(s):
    return str(s).upper().replace("T", "U")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="MRL", choices=list(TASK_PREFIXES))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="audit-only: counts + trace, no jsonl write")
    args = ap.parse_args()
    task_prefix = TASK_PREFIXES[args.task]

    rows = load_task_rows(task_prefix)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(r)
    val_measured = load_validation_measured(task_prefix)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_path = out_dir / f"within_source_augmentation_{args.task.lower()}_v1.jsonl"
    trace_path = out_dir / f"synthetic_trace_{args.task.lower()}_v1.jsonl"
    audit_path = out_dir / f"leak_audit_{args.task.lower()}_v1.json"

    rng = random.Random(SEED)
    n_synth = 0
    n_flagged = 0
    n_sources = 0
    n_sources_with_5plus_keys = 0
    per_source_counts = []
    trace_lines = []
    jsonl = None if args.dry_run else open(synth_path, "w")

    for sid in sorted(by_src):
        rs = by_src[sid]
        n_sources += 1
        src_seq = str(rs[0]["source_sequence"]).upper()
        L = len(src_seq)
        if L < 4:
            continue
        measured_keys = set()
        for r in rs:
            es = sorted(e.get("position", 0) for e in (r.get("source_relative_edits") or []))
            k = struct_key_of_edits(es)
            if k:
                measured_keys.add(k)
        if not measured_keys:
            measured_keys = {(1, 0, (L - 1) // WINDOW_BLOCK)}
        if len(measured_keys) >= 5:
            n_sources_with_5plus_keys += 1
        have = len(rs)
        need = TARGET_PER_SOURCE - have
        if need <= 0:
            per_source_counts.append(have)
            continue
        keys_list = sorted(measured_keys)
        src_count = 0
        trace_entries = []
        for i in range(need):
            k = keys_list[i % len(keys_list)] if i < len(keys_list) * 8 else rng.choice(keys_list)
            n_edits = min(k[0], SYNTH_CAP, L)
            lo = k[1] * WINDOW_BLOCK
            hi = k[2] * WINDOW_BLOCK + WINDOW_BLOCK - 1
            lo = max(0, min(lo, L - 1))
            hi = max(lo, min(hi, L - 1))
            span = hi - lo + 1
            ne = min(n_edits, span)
            positions = sorted(rng.sample(range(lo, hi + 1), ne)) if span >= ne else list(range(lo, hi + 1))
            cand = list(src_seq)
            for p in positions:
                choices = [b for b in BASES if b != src_seq[p]]
                cand[p] = rng.choice(choices) if choices else src_seq[p]
            cand = "".join(cand)
            cand_u = norm(cand)
            if cand_u in val_measured or norm(src_seq) in val_measured:
                n_flagged += 1
                continue
            if cand == src_seq:
                continue
            trace_seed = rng.randrange(2**31)
            rec = {
                "source_id": sid,
                "task_id": rs[0]["task_id"],
                "assay_id": rs[0].get("assay_id"),
                "biological_context_id": rs[0].get("biological_context_id"),
                "source_sequence": src_seq,
                "candidate_sequence": cand,
                "structure_key": list(k),
                "synthetic": True,
                "h1_label": None,
                "loss_weight": 0.0,
                "trace_seed": trace_seed,
                "d16c_amendment": "v1",
            }
            if jsonl is not None:
                jsonl.write(json.dumps(rec) + "\n")
            trace_entries.append({"source_id": sid, "structure_key": list(k), "trace_seed": trace_seed, "positions": positions})
            src_count += 1
            n_synth += 1
        per_source_counts.append(have + src_count)
        if trace_entries:
            trace_lines.append(json.dumps({"source_id": sid, "seed": SEED, "n_measured": have, "n_synthetic": src_count, "entries": trace_entries[:32]}) + "\n")

    if jsonl is not None:
        jsonl.close()
    if not args.dry_run:
        with open(trace_path, "w") as f:
            f.writelines(trace_lines)

    avg = sum(per_source_counts) / len(per_source_counts) if per_source_counts else 0.0
    audit = {
        "schema_version": "route_a_v3_d16c_within_source_augmentation.v1",
        "task": args.task,
        "seed": SEED,
        "target_per_source": TARGET_PER_SOURCE,
        "n_sources": n_sources,
        "n_measured_rows": len(rows),
        "n_synthetic": n_synth,
        "n_flagged_dropped": n_flagged,
        "sources_with_5plus_structure_keys": n_sources_with_5plus_keys,
        "avg_cand_per_source_after": round(avg, 2),
        "h1_hard_clause": "synthetic rows carry NO regression label (h1_label=null, loss_weight=0.0)",
        "leak_audit": "exact-match vs VALIDATION measured candidates+sources; flagged dropped",
        "leak_audit_status": "PASS" if n_flagged == 0 else f"FLAGGED_DROP ({n_flagged})",
        "protected_reads": 0,
        "dry_run": bool(args.dry_run),
    }
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
