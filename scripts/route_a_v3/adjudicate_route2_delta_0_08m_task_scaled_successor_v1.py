#!/usr/bin/env python3
"""Adjudicate the pre-frozen single-factor 0.08M target-scaling repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class SuccessorAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorAdjudicationError(message)


def _finite(value: Any, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"nonfinite {label}")
    return number


def _successor_task_scores(metrics: Mapping[str, Any]) -> dict[str, float]:
    region = {"region=0": "5UTR", "region=1": "3UTR"}
    result = {}
    for key, row in metrics["task_metrics"].items():
        endpoint, separator, region_key = str(key).partition("::")
        _require(separator == "::" and region_key in region, f"unexpected successor task key: {key}")
        result[f"{region[region_key]}|{endpoint}"] = _finite(row["spearman"], f"successor task Spearman {key}")
    return result


def _reference_task_scores(reference: Mapping[str, Any]) -> dict[str, float]:
    metrics = reference["metrics"]
    return {
        str(key): _finite(row["spearman"], f"reference task Spearman {key}")
        for key, row in metrics["task_numeric"].items()
    }


def adjudicate(
    summary: Mapping[str, Any],
    reference: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        protocol["schema_version"]
        == "route_a_v3_route2_delta_0_08m_task_scaled_successor_protocol.v1",
        "unexpected protocol schema",
    )
    metrics = summary["validation_metrics"]
    successor_tasks = _successor_task_scores(metrics)
    reference_tasks = _reference_task_scores(reference)
    _require(set(successor_tasks) == set(reference_tasks), "successor/reference task support differs")
    improvements = {
        task: successor_tasks[task] - reference_tasks[task]
        for task in sorted(successor_tasks)
    }
    improved_count = sum(value > 0.0 for value in improvements.values())
    task_macro = _finite(metrics["task_macro_spearman"], "task-macro Spearman")
    standardized_mae = _finite(
        metrics["task_macro_standardized_mae"], "task-macro standardized MAE"
    )
    target_scaler = summary["target_scaler"]
    checks = {
        "terminal_training_summary": summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "successor_identity": summary.get("baseline_id") == protocol["successor_baseline_id"],
        "reference_identity": reference.get("schema_version") == "route_a_v3_route2_prediction_evaluation.v1",
        "hpo_validation_only": summary.get("result_stage") == "HPO_VALIDATION_ONLY",
        "fixed_grouped_split": summary.get("run_mode") == "FIXED_GROUPED_SPLIT",
        "model_kind": summary.get("model_kind") == protocol["required_model_kind"],
        "metadata_mode": summary.get("metadata_mode") == protocol["required_metadata_mode"],
        "training_weighting_mode": summary.get("training_weighting_mode") == protocol["required_training_weighting_mode"],
        "target_scaling_mode": target_scaler.get("mode") == protocol["required_target_scaling_mode"],
        "target_scaler_train_only": target_scaler.get("fit_scope") == "TRAIN_ONLY",
        "target_scaler_zero_preserving": target_scaler.get("center_subtracted") is False,
        "loss_kind": summary.get("loss_kind") == protocol["required_loss_kind"],
        "checkpoint_selection": summary.get("checkpoint_selection") == protocol["required_checkpoint_selection"],
        "final_epoch_selected": summary.get("selected_epoch") == protocol["required_final_epoch"],
        "seed": summary.get("seed") == protocol["required_seed"],
        "parameter_count": summary.get("parameter_count") == protocol["required_parameter_count"],
        "parameter_changed": summary.get("parameter_changed") is True,
        "optimizer_steps_positive": int(summary.get("optimizer_steps", 0)) > 0,
        "cuda_device": str(summary.get("device", "")).startswith("cuda:"),
        "cuda_training_tensors": summary.get("cuda_training_tensors_verified") is True,
        "no_cpu_fallback": summary.get("cpu_fallback_used") is False,
        "train_count": summary.get("record_counts", {}).get("TRAIN") == protocol["required_train_record_count"],
        "validation_count": summary.get("record_counts", {}).get("VALIDATION") == protocol["required_validation_record_count"],
        "test_withheld": summary.get("development_test_record_count_withheld") == protocol["required_withheld_test_record_count"],
        "development_test_not_evaluated": summary.get("development_test_outcomes_evaluated") is False,
        "evaluation_not_accessed": summary.get("evaluation_outcomes_read", 0) == 0,
        "all_task_spearman_defined": metrics.get("defined_task_spearman_count") == protocol["required_task_count"],
        "task_count": metrics.get("task_count") == protocol["required_task_count"],
        "task_macro_beats_reference": task_macro > protocol["minimum_task_macro_spearman_exclusive"],
        "standardized_mae_not_worse": standardized_mae <= protocol["maximum_task_macro_standardized_mae_inclusive"],
        "task_breadth": improved_count >= protocol["minimum_tasks_improved_over_reference"],
    }
    qualified = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_delta_0_08m_task_scaled_successor_adjudication.v1",
        "status": "QUALIFIED_FOR_FRESH_SEEDS_AND_LOSO" if qualified else "SINGLE_FACTOR_TARGET_SCALING_NO_GO",
        "qualified": qualified,
        "checks": checks,
        "task_macro_spearman": task_macro,
        "task_macro_standardized_mae": standardized_mae,
        "task_improvement_count": improved_count,
        "task_improvements_over_reference": improvements,
        "authorized_next_steps": protocol["pass_authorizes"] if qualified else [],
        "not_authorized": protocol["pass_does_not_authorize"],
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--reference-evaluation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output exists: {args.output}")
    result = adjudicate(
        json.loads(args.training_summary.read_text(encoding="utf-8")),
        json.loads(args.reference_evaluation.read_text(encoding="utf-8")),
        json.loads(args.protocol.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
