#!/usr/bin/env python3
"""Summarize the matched mRNABERT loss comparison without reading predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_LOSSES = {
    "huber",
    "fixed_variance_gaussian_nll",
    "learned_variance_gaussian_nll",
}
MATCHED_CONFIG_KEYS = (
    "model_kind",
    "run_mode",
    "result_stage",
    "scientific_role",
    "candidate_control",
    "metadata_mode",
    "training_weighting_mode",
    "target_scaling_mode",
    "target_scale_floor",
    "target_scale_minimum_task_records",
    "hidden_dim",
    "depth",
    "max_length",
    "critic_position_features",
    "batch_size",
    "seed",
    "learning_rate",
    "weight_decay",
    "epochs",
    "optimizer_name",
    "checkpoint_selection",
    "checkpoint_metric",
    "development_manifest",
    "development_test_outcomes_accessed",
    "evaluation_outcomes_accessed",
    "pretrained_feature_cache_path",
    "pretrained_position_encoding",
    "encoder_attention_backend",
    "expected_frozen_pretrained_parameter_count",
    "frozen_capacity_profile_id",
    "canonical_paths",
    "training_precision",
    "optimizer_fused",
    "torch_compile",
    "num_workers",
    "pin_memory",
    "non_blocking_transfer",
    "huber_delta",
    "parameter_count_relative_tolerance",
)


class LossComparisonError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LossComparisonError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _matched_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in MATCHED_CONFIG_KEYS}


def summarize(summary_paths: list[Path]) -> dict[str, Any]:
    _require(len(summary_paths) == 3, "exactly three summaries are required")
    loaded = []
    for summary_path in summary_paths:
        _require(summary_path.name == "training_summary.json", "summary filename is unexpected")
        config_path = summary_path.parent / "training_config.json"
        _require(config_path.is_file(), f"training config is missing: {config_path}")
        summary = _load(summary_path)
        config = _load(config_path)
        _require(
            summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
            f"training is not complete: {summary_path}",
        )
        _require(
            summary.get("result_stage") == "HPO_VALIDATION_ONLY",
            "loss selection must use Development VALIDATION only",
        )
        _require(summary.get("evaluation_outcomes_read") == 0, "Evaluation entered loss selection")
        _require(summary.get("test_metrics") is None, "Development TEST entered loss selection")
        _require(summary.get("development_test_outcomes_evaluated") is False, "Development TEST was evaluated")
        _require(
            config.get("result_stage") == "HPO_VALIDATION_ONLY"
            and config.get("run_mode") == "FIXED_GROUPED_SPLIT"
            and config.get("candidate_control") == "NONE",
            "loss-selection config is not the primary fixed Development split",
        )
        _require(
            config.get("evaluation_outcomes_accessed") is False
            and config.get("development_test_outcomes_accessed") is False,
            "loss-selection config admits Evaluation or Development TEST outcomes",
        )
        _require(
            config.get("loss_kind") == summary.get("loss_kind"),
            "training config and summary loss kinds differ",
        )
        _require(
            config.get("training_precision") == summary.get("training_precision"),
            "training config and summary precision differ",
        )
        loaded.append((summary_path, config, summary))

    losses = {str(summary["loss_kind"]) for _, _, summary in loaded}
    _require(losses == EXPECTED_LOSSES, f"loss set differs from the frozen comparison: {sorted(losses)}")
    reference_signature = _matched_signature(loaded[0][1])
    for _path, config, _summary in loaded[1:]:
        _require(
            _matched_signature(config) == reference_signature,
            "loss runs differ in a non-loss training/configuration field",
        )

    rows = []
    for path, config, summary in loaded:
        metrics = summary.get("validation_metrics")
        _require(isinstance(metrics, Mapping), f"validation metrics are missing: {path}")
        task_macro = _finite(metrics.get("task_macro_spearman"), "task-macro Spearman")
        standardized_mae = _finite(
            metrics.get("task_macro_standardized_mae"),
            "task-macro standardized MAE",
        )
        global_spearman = _finite(metrics.get("spearman"), "global Spearman")
        prediction_spread = _finite(
            metrics.get("prediction_std_over_target_std"),
            "prediction std over target std",
        )
        uncertainty_used = bool(summary.get("uncertainty_head_used"))
        residual_scale = metrics.get("absolute_residual_scale_spearman")
        if residual_scale is not None:
            residual_scale = _finite(
                residual_scale, "uncertainty vs absolute-residual Spearman"
            )
        if summary["loss_kind"] == "learned_variance_gaussian_nll":
            _require(uncertainty_used, "learned-variance run lacks its uncertainty head")
        else:
            _require(not uncertainty_used, "mean/fixed-variance run unexpectedly trained uncertainty")
        rows.append({
            "loss_kind": summary["loss_kind"],
            "baseline_id": summary["baseline_id"],
            "summary_path": str(path),
            "seed": summary["seed"],
            "selected_epoch": summary["selected_epoch"],
            "optimizer_steps": summary["optimizer_steps"],
            "trainable_parameter_count": summary["trainable_parameter_count"],
            "training_precision": summary["training_precision"],
            "validation_global_spearman": global_spearman,
            "validation_task_macro_spearman": task_macro,
            "validation_task_macro_standardized_mae": standardized_mae,
            "validation_prediction_std": _finite(
                metrics.get("prediction_std"), "prediction std"
            ),
            "validation_prediction_std_over_target_std": prediction_spread,
            "uncertainty_head_used": uncertainty_used,
            "predicted_standard_deviation_mean": metrics.get(
                "predicted_standard_deviation_mean"
            ),
            "absolute_residual_scale_spearman": residual_scale,
        })

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["validation_task_macro_spearman"],
            row["validation_task_macro_standardized_mae"],
            -row["validation_global_spearman"],
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["validation_rank"] = index
    by_loss = {row["loss_kind"]: row for row in rows}
    learned = by_loss["learned_variance_gaussian_nll"]
    huber = by_loss["huber"]
    return {
        "schema_version": "route_a_v3_route2_mrnabert_loss_comparison.v1",
        "status": "THREE_MATCHED_DEVELOPMENT_VALIDATION_LOSSES_COMPLETE",
        "selection_rule": (
            "MAX_TASK_MACRO_SPEARMAN_THEN_MIN_TASK_MACRO_STANDARDIZED_MAE_"
            "THEN_MAX_GLOBAL_SPEARMAN"
        ),
        "selected_loss_for_controls": ranked[0]["loss_kind"],
        "matched_configuration": reference_signature,
        "rows": sorted(rows, key=lambda row: row["loss_kind"]),
        "learned_uncertainty_diagnostics": {
            "task_macro_spearman_difference_from_huber": (
                learned["validation_task_macro_spearman"]
                - huber["validation_task_macro_spearman"]
            ),
            "prediction_spread_ratio_difference_from_huber": (
                learned["validation_prediction_std_over_target_std"]
                - huber["validation_prediction_std_over_target_std"]
            ),
            "absolute_residual_scale_spearman": learned[
                "absolute_residual_scale_spearman"
            ],
            "interpretation": (
                "DIAGNOSTIC_ONLY_REVIEW_MEAN_PERFORMANCE_AND_SPREAD;_"
                "LOWER_NLL_ALONE_CANNOT_SELECT_THE_LOSS"
            ),
        },
        "next_required_controls": [
            "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
            "PARAMETER_MATCHED_PRETRAINED_SOURCE_ONLY",
        ],
        "development_test_opened": False,
        "evaluation_opened": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = summarize(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
