#!/usr/bin/env python3
"""Saluki frozen zero-shot delta evaluation on GSE217518 (Task 4, protocol v1).

Frozen mode per the baseline leaderboard protocol: official Zenodo Saluki
checkpoints, no fine-tuning. For each VALIDATION record of GSE217518, score
source and candidate sequences with the full official fold ensemble
(mean over all model{0,1}_best.h5 checkpoints), delta = candidate - source,
then evaluate delta against direction_normalized_delta with the frozen
Task-1 evaluator (K=10) for the two RNA_HALF_LIFE region tasks.

UTR-window caveat (documented, not hidden): Saluki consumes full spliced mRNA
transcripts (12,288 right-padded); GSE217518 supplies 115bp reconstructed UTR
windows. The frozen row therefore measures the official model applied to UTR
windows - the protocol's native-truncation clause - and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
sys.path.insert(0, EVAL_REPO + "/scripts/route_a_v3")

import importlib.util

_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)

from core.route2_saluki_port_v1 import SalukiGRUV1, encode_saluki_six_channel_v1  # noqa: E402

SALUKI_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/saluki/datasets/deeplearning/train_gru"
)
MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/manifests/route2_development_frozen_v1/development_manifest.jsonl"
)
CANONICAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE217518/v1/canonical_records.jsonl"
)
SALUKI_FULL_LENGTH = 12288
BATCH = 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--limit-models", type=int, default=0, help="0 = all checkpoints")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_saluki_frozen_gse217518_20260903"
    ))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")

    # Validation ids for GSE217518.
    validation_ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == "GSE217518" and row["split"] == "VALIDATION":
                validation_ids.add(str(row["canonical_record_id"]))

    records = {}
    with CANONICAL.open() as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row.get("canonical_record_id"))
            if rid in validation_ids:
                records[rid] = row
    if set(records) != validation_ids:
        raise SystemExit(
            f"coverage mismatch: {len(records)} loaded vs {len(validation_ids)} manifest ids"
        )

    ids = sorted(records)
    sources = [records[rid]["source_sequence"] for rid in ids]
    candidates = [records[rid]["candidate_sequence"] for rid in ids]
    encoded = np.stack(
        [encode_saluki_six_channel_v1(seq, SALUKI_FULL_LENGTH) for seq in sources + candidates]
    )
    tensor_all = torch.from_numpy(encoded)

    checkpoint_paths = sorted(SALUKI_ROOT.glob("f*_c*/train/model*_best.h5"))
    if args.limit_models:
        checkpoint_paths = checkpoint_paths[: args.limit_models]

    source_scores = np.zeros(len(ids), dtype=np.float64)
    candidate_scores = np.zeros(len(ids), dtype=np.float64)
    per_model = {}
    for index, checkpoint in enumerate(checkpoint_paths):
        model = SalukiGRUV1(checkpoint).to(device).eval()
        scores = []
        with torch.no_grad():
            for start in range(0, tensor_all.shape[0], BATCH):
                batch = tensor_all[start : start + BATCH].to(device)
                scores.append(model(batch).double().cpu().numpy())
        scores = np.concatenate(scores)
        model_source = scores[: len(ids)]
        model_candidate = scores[len(ids) :]
        source_scores += model_source
        candidate_scores += model_candidate
        per_model[str(checkpoint.relative_to(SALUKI_ROOT))] = {
            "source_mean": float(model_source.mean()),
            "candidate_mean": float(model_candidate.mean()),
            "delta_spearman_preview": None,  # filled below cheaply
        }
        del model
        torch.cuda.empty_cache()
        if (index + 1) % 10 == 0 or index + 1 == len(checkpoint_paths):
            print(f"[{index + 1}/{len(checkpoint_paths)}] checkpoints scored", flush=True)

    source_scores /= len(checkpoint_paths)
    candidate_scores /= len(checkpoint_paths)
    delta = candidate_scores - source_scores

    predictions = {rid: float(delta[i]) for i, rid in enumerate(ids)}

    # Frozen evaluator (same as Task 1 / Track B), K=10.
    observations = ev.load_observations([CANONICAL], validation_ids)
    metrics = ev.evaluate(observations, predictions, 10)

    region_counts = {}
    for rid in ids:
        region_counts[records[rid]["region"]] = region_counts.get(records[rid]["region"], 0) + 1

    report = {
        "schema_version": "route_a_v3_route2_saluki_frozen_delta_gse217518.v1",
        "mode": "FROZEN_ZERO_SHOT_DELTA",
        "device": torch.cuda.get_device_name(device),
        "checkpoint_count": len(checkpoint_paths),
        "record_count": len(ids),
        "region_counts": region_counts,
        "metrics": metrics,
        "note": (
            "Official Saluki fold-ensemble mean; delta = candidate - source; "
            "115bp UTR windows right-padded to 12288 (native-truncation clause)."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frozen_delta_results.json").open("w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for rid in ids:
            handle.write(json.dumps({
                "canonical_record_id": rid,
                "predicted_direction_normalized_delta": predictions[rid],
            }) + "\n")

    print(json.dumps({
        "checkpoint_count": len(checkpoint_paths),
        "record_count": len(ids),
        "region_counts": region_counts,
        "task_macro_spearman": metrics.get("task_macro_spearman"),
        "tasks": {k: {kk: vv for kk, vv in v.items() if kk in ("spearman", "record_count")}
                  for k, v in (metrics.get("tasks") or {}).items()},
        "top_1": metrics.get("source_macro_top_1_accuracy"),
        "ndcg_at_10": metrics.get("source_macro_ndcg_at_k"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
