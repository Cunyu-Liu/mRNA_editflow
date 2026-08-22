#!/usr/bin/env python3
"""Validate zero-LoRA online Critic V3 features against the frozen cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_edit_site_token_cache_v3 import load_edit_site_token_cache_v3
from core.route2_xeditcritic_cache_alignment_v3 import compare_cache_online_features_v3
from core.route2_xeditcritic_training_data_v3 import RNA_TOKEN, records_from_projection_rows
from scripts.route_a_v3.route2_mrnabert_lora_edit_site_encoder_v3 import TrainableMRNABERTEditSiteEncoderV3
from scripts.route_a_v3.train_route2_xeditcritic_v3 import EditSiteCacheViewV3, XEditCriticCollatorV3


class CacheOnlineAlignmentRunnerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheOnlineAlignmentRunnerError(message)


def _example(record, bundle):
    return {
        "record_id": record.record_id,
        "source_group": record.source_group,
        "task": record.task,
        "source": torch.tensor([RNA_TOKEN[base] for base in record.source]),
        "candidate": torch.tensor([RNA_TOKEN[base] for base in record.candidate]),
        "edits": record.edits,
        "target": 0.0,
        "scaled_target": 0.0,
        "target_scale": 1.0,
        "sample_weight": 1.0,
        "study": 0,
        "assay": 0,
        "context": 0,
        "quantity": 0,
        "measurement": 0,
        "numerator": 0,
        "denominator": 0,
        "region": record.region,
        "feature_bundle": bundle,
    }


def validate(config, physical_gpu_index: int):
    output = Path(config["output_path"])
    _require(not output.exists(), f"terminal cache/online alignment exists: {output}")
    _require(torch.cuda.is_available(), "CUDA is unavailable for cache/online alignment")
    device = torch.device(f"cuda:{physical_gpu_index}")
    torch.cuda.set_device(device)
    rows = load_projection_rows([Path(path) for path in config["projection_paths"]])
    records = {record.record_id: record for record in records_from_projection_rows(rows)}
    preset_ids = [str(value) for value in config["preset_record_ids"]]
    _require(set(preset_ids) <= set(records), "preset alignment record is absent")
    for record_id, geometry in zip(preset_ids, config["preset_geometries"], strict=True):
        record = records[record_id]
        _require(len(record.source) == int(geometry["sequence_length"]), "preset sequence length changed")
        _require(len(record.edits) == int(geometry["edit_count"]), "preset edit count changed")
    cache_payload = load_edit_site_token_cache_v3(Path(config["edit_site_cache_path"]))
    cache = EditSiteCacheViewV3(cache_payload, set(cache_payload["record_ids"]))
    encoder = TrainableMRNABERTEditSiteEncoderV3(
        Path(config["mrnabert_model_path"]),
        device,
        rank=int(config["lora_rank"]),
        alpha=float(config["lora_alpha"]),
        dropout=float(config["lora_dropout"]),
    ).eval()
    collator = XEditCriticCollatorV3(pretrained_width=768)
    record_rows = []
    all_differences = []
    for record_id in preset_ids:
        record = records[record_id]
        cached_batch = collator([_example(record, cache.bundle(record_id))])
        online_input = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in cached_batch.items()
        }
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            online_batch = encoder(online_input)
        cached_features = {
            key: cached_batch[key].float()
            for key in (
                "source_site", "candidate_site", "source_window_mean", "candidate_window_mean",
                "source_window_max", "candidate_window_max", "source_global", "candidate_global",
            )
        }
        online_features = {key: online_batch[key].float().cpu() for key in cached_features}
        comparison = compare_cache_online_features_v3(
            cached_features,
            online_features,
            active_edit_count=len(record.edits),
            maximum_absolute_tolerance=float(config["maximum_absolute_tolerance"]),
            mean_absolute_tolerance=float(config["mean_absolute_tolerance"]),
        )
        record_rows.append({
            "record_id": record_id,
            "sequence_length": len(record.source),
            "edit_count": len(record.edits),
            **comparison,
        })
        all_differences.append(comparison)
    maximum = max(row["maximum_absolute_difference"] for row in all_differences)
    weighted_mean = sum(
        sum(feature["mean_absolute_difference"] * feature["value_count"] for feature in row["feature_rows"].values())
        for row in all_differences
    ) / sum(
        sum(feature["value_count"] for feature in row["feature_rows"].values())
        for row in all_differences
    )
    passed = all(row["passed"] for row in all_differences)
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_cache_online_alignment.v3",
        "status": "XEDITCRITIC_V3_CACHE_ONLINE_ALIGNMENT_PASS" if passed else "XEDITCRITIC_V3_CACHE_ONLINE_ALIGNMENT_FAIL",
        "record_rows": record_rows,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": weighted_mean,
        "maximum_absolute_tolerance": float(config["maximum_absolute_tolerance"]),
        "mean_absolute_tolerance": float(config["mean_absolute_tolerance"]),
        "pretrained_parameter_count": sum(
            parameter.numel() for parameter in encoder.model.parameters()
            if not parameter.requires_grad
        ),
        "lora_trainable_parameter_count": encoder.trainable_parameter_count,
        "parameter_update_count": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _require(passed, "cache/online alignment exceeded the predeclared tolerance")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(validate(config, args.physical_gpu_index), sort_keys=True))


if __name__ == "__main__":
    main()
