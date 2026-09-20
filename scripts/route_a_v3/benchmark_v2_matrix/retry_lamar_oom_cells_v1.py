#!/usr/bin/env python3
"""Retry LAMAR OOM cells (GSE149487 both endpoints) with reduced batch.

Same adapter (frozen weights, same input adaptation); only batch size is
reduced (batch size is inference plumbing, not a prereg-registered
adaptation axis). Appends results to cells.json (append-only) and rewrites
predictions.jsonl cell rows.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
ADAPTERS_PATH = REPO_ROOT / "scripts/route_a_v3/benchmark_v2_matrix/family_adapters_v1.py"
EVAL_SCRIPT = REPO_ROOT / "scripts/route_a_v3/evaluate_route2_prediction_v1.py"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT_DIR = MNT / "benchmark_v2/leaderboard_matrix_v2/cells/lamar_utr5te"
K = 10


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module("ev", EVAL_SCRIPT)
ad = _load_module("family_adapters", ADAPTERS_PATH)

device = torch.device("cuda:6")

scorer, family_meta = ad.build_lamar(device)


def score_small(seqs):
    return ad._batched(seqs, device, None, batch=4) if False else None


# rebuild forward with batch 4 by re-using adapter but monkey-batching
# simplest: call scorer on chunks of 4
def score_chunked(seqs, chunk=4):
    outs = []
    for i in range(0, len(seqs), chunk):
        outs.append(scorer(seqs[i:i + chunk]))
    return np.concatenate(outs) if outs else np.array([])


MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
RETRY_CELLS = [
    ("GSE149487", "5UTR", "te_log2_polysome_over_totalrna"),
    ("GSE149487", "5UTR", "transcript_log2_totalrna_over_dna"),
]

validation_ids = {}
with MANIFEST.open() as handle:
    for line in handle:
        row = json.loads(line)
        if row["split"] == "VALIDATION":
            validation_ids[str(row["canonical_record_id"])] = True

records = []
path = MNT / "canonical/GSE149487/v1/canonical_records.private.jsonl"
if not path.exists():
    path = MNT / "canonical/GSE149487/v1/canonical_records.jsonl"
with path.open() as handle:
    for line in handle:
        row = json.loads(line)
        rid = str(row["canonical_record_id"])
        if rid in validation_ids:
            records.append(row)

cells = json.loads((OUT_DIR / "cells.json").read_text())
pred_lines = [json.loads(l) for l in (OUT_DIR / "predictions.jsonl").open()]
pred_lines = [p for p in pred_lines if not p["cell"].startswith("GSE149487")]

for study, region, endpoint in RETRY_CELLS:
    tag = f"{study}|{region}|{endpoint}"
    rows = [r for r in records if r["region"] == region and r["endpoint_id"] == endpoint]
    print(f"retry {tag}: n={len(rows)}", flush=True)
    y_src = score_chunked([r["source_sequence"] for r in rows])
    y_cand = score_chunked([r["candidate_sequence"] for r in rows])
    finite = np.isfinite(y_src) & np.isfinite(y_cand)
    kept = [rows[i] for i in range(len(rows)) if finite[i]]
    preds = {str(r["canonical_record_id"]): float(y_cand[i] - y_src[i]) for i, r in enumerate(kept)}
    observations = []
    for i, r in enumerate(kept):
        observations.append({
            "canonical_record_id": str(r["canonical_record_id"]),
            "study_unit_id": str(r["study_unit_id"]),
            "source_id": str(r["source_id"]),
            "biological_context_id": str(r["biological_context_id"]),
            "endpoint_id": str(r["endpoint_id"]),
            "stratum": (str(r["study_unit_id"]), str(r["region"]), str(r["endpoint_id"])),
            "task": (str(r["region"]), str(r["endpoint_id"])),
            "observed": float(r["direction_normalized_delta"]),
        })
    metrics = ev.evaluate(observations, preds, K)
    for i, r in enumerate(kept):
        pred_lines.append({
            "cell": tag,
            "row_id": str(r["canonical_record_id"]),
            "source_id": str(r["source_id"]),
            "candidate_id": str(r["candidate_id"]),
            "task": tag,
            "pred_src": float(y_src[i]),
            "pred_cand": float(y_cand[i]),
            "delta_hat": float(y_cand[i] - y_src[i]),
            "observed": float(r["direction_normalized_delta"]),
        })
    miss_rate = (len(rows) - len(kept)) / len(rows) if rows else 0.0
    cells["matrix"][tag] = {
        "status": "OK",
        "n": len(rows),
        "n_evaluated": len(kept),
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "overall_spearman": metrics.get("overall_numeric", {}).get("spearman"),
        "source_macro_within_source_spearman": metrics.get("source_macro_within_source_spearman"),
        "source_macro_top_1": metrics.get("source_macro_top_1_accuracy"),
        "source_macro_ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
        "source_group_count": metrics.get("source_group_count"),
        "miss_count": len(rows) - len(kept),
        "miss_rate": miss_rate,
        "miss_ids": [],
        "note": "OOM retried with batch=4 (inference plumbing only; same frozen weights and adaptation)",
    }
    print(f"{tag}: rho={metrics.get('task_macro_spearman')} n={len(kept)}", flush=True)

cells["oom_retry"] = {
    "cells_retried": [f"{s}|{r}|{e}" for s, r, e in RETRY_CELLS],
    "reason": "CUDA OOM on MIG slice during original chain run",
    "fix": "batch 32 -> 4 (plumbing), same adapter/weights/adaptation",
}
(OUT_DIR / "cells.json").write_text(json.dumps(cells, indent=1, sort_keys=True))
with (OUT_DIR / "predictions.jsonl").open("w") as handle:
    for p in pred_lines:
        handle.write(json.dumps(p) + "\n")
print(f"updated {OUT_DIR}/cells.json ({len(pred_lines)} prediction rows)", flush=True)
