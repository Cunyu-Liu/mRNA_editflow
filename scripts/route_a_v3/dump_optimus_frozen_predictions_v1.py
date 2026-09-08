#!/usr/bin/env python3
"""Dump per-record frozen-Optimus predictions on GSE114002 VALIDATION (730).

Stage 0a (run_route2_frozen_delta_gse114002_v1.py) computed the frozen
zero-shot delta row but persisted only aggregate metrics. Task 12.1 Holm needs
per-record paired predictions; this script re-runs the identical frozen
protocol (official weights, source/candidate scoring, delta) and persists
predictions.jsonl so the Holm exact-bootstrap path can engage. Byte-identical
protocol to Stage 0a - only output persistence is added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import torch

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
HARNESS = REPO_ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"

import importlib.util

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

_h_spec = importlib.util.spec_from_file_location("harness", str(HARNESS))
harness = importlib.util.module_from_spec(_h_spec)
sys.modules["harness"] = harness
_h_spec.loader.exec_module(harness)

MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl"
)
WEIGHT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5"
)
OUT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_frozen_delta_gse114002_20260903/predictions.jsonl"
)


def main() -> int:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = torch.device(f"cuda:{gpu}")

    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE114002" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))

    observations = ev.load_observations([CANONICAL], validation_ids)
    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    if set(records) != validation_ids:
        raise SystemExit("coverage mismatch against manifest")

    ids = sorted(records)
    sources = [records[rid]["source_sequence"] for rid in ids]
    candidates = [records[rid]["candidate_sequence"] for rid in ids]

    model = harness.Optimus5Prime(WEIGHT).to(device).eval()
    with torch.no_grad():
        source_scores = model(harness.one_hot(sources, device)).double().cpu().numpy()
        candidate_scores = model(harness.one_hot(candidates, device)).double().cpu().numpy()
    delta = candidate_scores - source_scores

    with OUT.open("w") as handle:
        for i, rid in enumerate(ids):
            handle.write(json.dumps({
                "canonical_record_id": rid,
                "prediction": float(delta[i]),
                "source_score": float(source_scores[i]),
                "candidate_score": float(candidate_scores[i]),
            }) + "\n")

    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
    metrics = ev.evaluate(observations, predictions, 10)
    print(f"optimus5prime frozen-delta recheck: spearman {metrics['task_macro_spearman']:.4f} "
          f"top-1 {metrics['source_macro_top_1_accuracy']:.4f} "
          f"ndcg@10 {metrics['source_macro_ndcg_at_k']:.4f} (reference 0.3132)")
    print(f"wrote {len(ids)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
