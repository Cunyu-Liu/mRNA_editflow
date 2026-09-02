#!/usr/bin/env python3
"""Stage 0a: frozen zero-shot delta for Optimus/FramePool on GSE114002 (route A discriminator).

Measures how much the external-library prior alone (280K-pretrained weights,
NO task fine-tuning) transfers to source-relative delta ranking - the upper
bound of what route A Step 1 can buy. Frozen protocol: official weights, score
source and candidate, delta = candidate - source, evaluate against
direction_normalized_delta with the frozen Task-1 evaluator (K=10).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
HARNESS = REPO_ROOT / "scripts/route_a_v3/run_route2_external_prediction_baselines_v1.py"

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

_h_spec = importlib.util.spec_from_file_location(
    "harness", str(HARNESS)
)
harness = importlib.util.module_from_spec(_h_spec)
sys.modules["harness"] = harness
_h_spec.loader.exec_module(harness)

MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl"
)
WEIGHTS = {
    "optimus5prime": Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/optimus5prime/main_MRL_model.hdf5"
    ),
    "framepool": Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/framepool/Framepool_combined_residual.h5"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--models", nargs="+", default=["optimus5prime", "framepool"])
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_frozen_delta_gse114002_20260903"
    ))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")

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
    for rid in ids:
        if len(records[rid]["source_sequence"]) != 50 or len(records[rid]["candidate_sequence"]) != 50:
            raise SystemExit(f"non-50nt record {rid}")

    sources = [records[rid]["source_sequence"] for rid in ids]
    candidates = [records[rid]["candidate_sequence"] for rid in ids]

    results = {}
    for name in args.models:
        model_cls = harness.Optimus5Prime if name == "optimus5prime" else harness.FramePool
        model = model_cls(WEIGHTS[name]).to(device).eval()
        with torch.no_grad():
            source_scores = model(harness.one_hot(sources, device)).double().cpu().numpy()
            candidate_scores = model(harness.one_hot(candidates, device)).double().cpu().numpy()
        delta = candidate_scores - source_scores
        predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}
        metrics = ev.evaluate(observations, predictions, 10)
        results[name] = {
            "mode": "FROZEN_ZERO_SHOT_DELTA",
            "task_macro_spearman": metrics.get("task_macro_spearman"),
            "within_source": metrics.get("source_macro_within_source_spearman"),
            "top_1": metrics.get("source_macro_top_1_accuracy"),
            "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
            "source_score_std": float(source_scores.std()),
            "delta_std": float(delta.std()),
        }
        print(f"{name}: spearman {results[name]['task_macro_spearman']:.4f} | top-1 {results[name]['top_1']:.4f} | ndcg@10 {results[name]['ndcg_at_10']:.4f} | delta_std {results[name]['delta_std']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frozen_delta_results.json").open("w") as handle:
        json.dump({
            "schema_version": "route_a_v3_route2_frozen_delta_gse114002.v1",
            "record_count": len(ids),
            "reference": {
                "optimus_adapter_finetuned": 0.3132,
                "framepool_adapter_finetuned": 0.2956,
                "critic_v5": 0.1354,
                "w0_from_scratch": 0.1987,
            },
            "results": results,
        }, handle, indent=1, sort_keys=True)
    print("wrote", args.output_dir / "frozen_delta_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
