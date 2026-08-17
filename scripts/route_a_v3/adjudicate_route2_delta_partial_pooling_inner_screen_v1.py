#!/usr/bin/env python3
"""Adjudicate the frozen TRAIN-only partial-pooling screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


class PartialPoolingAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialPoolingAdjudicationError(message)


def load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_delta_partial_pooling_inner_screen_protocol.v1",
        "unexpected partial-pooling protocol schema",
    )
    _require(
        protocol.get("status") == "FROZEN_BEFORE_TRAIN_INNER_OUTCOMES",
        "partial-pooling protocol was not frozen before outcomes",
    )
    _require(
        protocol.get("single_changed_model_factor") == "endpoint_region_residual",
        "partial-pooling screen is not single-factor",
    )
    for key in (
        "parent_development_validation_outcomes_accessed",
        "parent_development_test_outcomes_accessed",
        "inner_test_outcomes_accessed",
        "evaluation_outcomes_accessed",
    ):
        _require(protocol.get(key) is False, f"forbidden protocol outcome access: {key}")


def validate_arm_config_pair(
    protocol: Mapping[str, Any],
    shared: Mapping[str, Any],
    residual: Mapping[str, Any],
) -> None:
    allowed_differences = {
        "baseline_id",
        "endpoint_region_residual",
        "frozen_capacity_profile_id",
        "frozen_expected_parameter_count",
        "output_directory",
    }
    differing = {
        key
        for key in set(shared) | set(residual)
        if shared.get(key) != residual.get(key)
    }
    _require(differing == allowed_differences, f"arm config differences changed: {sorted(differing)}")
    _require(shared.get("endpoint_region_residual") is False, "shared arm enables residual")
    _require(residual.get("endpoint_region_residual") is True, "residual arm disables residual")
    expected_parameters = protocol["expected_parameter_counts"]
    _require(
        shared.get("frozen_expected_parameter_count") == expected_parameters["shared"],
        "shared expected parameter count changed",
    )
    _require(
        residual.get("frozen_expected_parameter_count") == expected_parameters["residual"],
        "residual expected parameter count changed",
    )


def validate_summary(
    protocol: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    arm: str,
) -> Mapping[str, Any]:
    _require(
        summary.get("status") == "DELTA_PREDICTOR_TRAIN_INNER_GPU_RUN_COMPLETE",
        f"{arm} TRAIN-inner run is not complete",
    )
    _require(
        summary.get("scientific_role") == "TRAIN_ONLY_PARTIAL_POOLING_MODEL_SELECTION",
        f"{arm} scientific role changed",
    )
    _require(summary.get("result_stage") == protocol["expected_result_stage"], f"{arm} result stage changed")
    _require(summary.get("inner_split_id") == protocol["expected_inner_split_id"], f"{arm} inner split changed")
    _require(summary.get("record_counts") == protocol["expected_record_counts"], f"{arm} record counts changed")
    _require(
        summary.get("inner_test_record_count_withheld")
        == protocol["expected_inner_test_record_count_withheld"],
        f"{arm} inner TEST was not withheld",
    )
    _require(
        summary.get("parent_development_validation_record_count_excluded")
        == protocol["expected_parent_development_validation_record_count_excluded"],
        f"{arm} parent Development Validation exclusion changed",
    )
    _require(
        summary.get("parent_development_test_record_count_excluded")
        == protocol["expected_parent_development_test_record_count_excluded"],
        f"{arm} parent Development Test exclusion changed",
    )
    _require(summary.get("development_validation_outcomes_evaluated") is False, f"{arm} evaluated Development Validation")
    _require(summary.get("development_test_outcomes_evaluated") is False, f"{arm} evaluated Development Test")
    _require(summary.get("inner_validation_outcomes_evaluated") is True, f"{arm} lacks inner Validation")
    _require(summary.get("inner_test_outcomes_evaluated") is False, f"{arm} evaluated inner TEST")
    _require(summary.get("evaluation_outcomes_read") == 0, f"{arm} read Evaluation outcomes")
    _require(summary.get("cpu_fallback_used") is False, f"{arm} used CPU fallback")
    _require(summary.get("cuda_training_tensors_verified") is True, f"{arm} lacks CUDA training evidence")
    _require(summary.get("physical_gpu_index") == protocol["expected_physical_gpu_index"], f"{arm} GPU changed")
    _require(summary.get("seed") == protocol["expected_seed"], f"{arm} seed changed")
    _require(summary.get("final_training_epoch") == protocol["expected_final_epoch"], f"{arm} epoch budget changed")
    _require(summary.get("selected_epoch") == protocol["expected_final_epoch"], f"{arm} did not select final epoch")
    _require(
        summary.get("parameter_count") == protocol["expected_parameter_counts"][arm],
        f"{arm} parameter count changed",
    )
    _require(summary.get("endpoint_region_residual") is (arm == "residual"), f"{arm} residual state changed")
    metrics = summary.get("validation_metrics")
    _require(isinstance(metrics, dict), f"{arm} validation metrics are absent")
    _require(metrics.get("task_count") == protocol["expected_task_count"], f"{arm} task count changed")
    _require(
        metrics.get("defined_task_spearman_count") == protocol["expected_task_count"],
        f"{arm} has undefined task Spearman",
    )
    _require(isinstance(metrics.get("task_metrics"), dict), f"{arm} task metrics are absent")
    return metrics


def adjudicate(
    protocol: Mapping[str, Any],
    shared_summary: Mapping[str, Any],
    residual_summary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_protocol(protocol)
    shared_metrics = validate_summary(protocol, shared_summary, arm="shared")
    residual_metrics = validate_summary(protocol, residual_summary, arm="residual")
    _require(
        shared_summary.get("optimizer_steps") == residual_summary.get("optimizer_steps"),
        "arm optimizer-step budgets differ",
    )
    shared_tasks = shared_metrics["task_metrics"]
    residual_tasks = residual_metrics["task_metrics"]
    _require(set(shared_tasks) == set(residual_tasks), "arm task identities differ")
    improved_tasks = sorted(
        task
        for task in shared_tasks
        if residual_tasks[task]["spearman"] > shared_tasks[task]["spearman"]
    )
    macro_gain = (
        residual_metrics["task_macro_spearman"]
        - shared_metrics["task_macro_spearman"]
    )
    mae_ratio = (
        residual_metrics["task_macro_standardized_mae"]
        / shared_metrics["task_macro_standardized_mae"]
    )
    rule = protocol["material_gain_rule"]
    material_gain = (
        macro_gain >= rule["minimum_task_macro_spearman_gain_inclusive"]
        and len(improved_tasks) >= rule["minimum_tasks_with_strict_spearman_improvement"]
        and mae_ratio <= rule["maximum_task_macro_standardized_mae_ratio_inclusive"]
    )
    return {
        "schema_version": "route_a_v3_route2_delta_partial_pooling_inner_screen_adjudication.v1",
        "status": (
            "MATERIAL_TRAIN_INNER_GAIN_ESTABLISHED_FOR_SCREEN"
            if material_gain
            else "NO_MATERIAL_TRAIN_INNER_GAIN"
        ),
        "material_gain": material_gain,
        "shared_task_macro_spearman": shared_metrics["task_macro_spearman"],
        "residual_task_macro_spearman": residual_metrics["task_macro_spearman"],
        "task_macro_spearman_gain": macro_gain,
        "tasks_with_strict_spearman_improvement": improved_tasks,
        "tasks_with_strict_spearman_improvement_count": len(improved_tasks),
        "shared_task_macro_standardized_mae": shared_metrics["task_macro_standardized_mae"],
        "residual_task_macro_standardized_mae": residual_metrics["task_macro_standardized_mae"],
        "task_macro_standardized_mae_ratio": mae_ratio,
        "optimizer_steps_per_arm": shared_summary["optimizer_steps"],
        "parent_development_validation_outcomes_accessed": False,
        "parent_development_test_outcomes_accessed": False,
        "inner_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "authorizes": protocol["pass_authorizes"] if material_gain else [],
        "does_not_authorize": protocol["pass_does_not_authorize"],
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shared-summary", type=Path, required=True)
    parser.add_argument("--residual-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    validate_protocol(protocol)
    shared_config = load_json(REPO_ROOT / protocol["shared_arm_config"])
    residual_config = load_json(REPO_ROOT / protocol["residual_arm_config"])
    validate_arm_config_pair(protocol, shared_config, residual_config)
    result = adjudicate(
        protocol,
        load_json(args.shared_summary),
        load_json(args.residual_summary),
    )
    output = args.output or Path(protocol["output_adjudication"])
    _require(not output.exists(), f"adjudication output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
