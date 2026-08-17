#!/usr/bin/env python3
"""Adjudicate the frozen Development-only Route 2 method-repair screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


class MethodRepairScreenError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MethodRepairScreenError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    return float(value)


def _scale_for_task(target_scaler: Mapping[str, Any], task: str) -> float:
    task_scales = target_scaler.get("task_scales") or {}
    if task in task_scales:
        return _finite(task_scales[task], f"task scale for {task}")
    _require("::region=" in task, f"task key lacks a region: {task}")
    region = f"region={task.rsplit('::region=', 1)[1]}"
    region_scales = target_scaler.get("region_scales") or {}
    if region in region_scales:
        return _finite(region_scales[region], f"region scale for {task}")
    return _finite(target_scaler.get("global_scale"), f"global scale for {task}")


def validate_run(config: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    _require(summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE", "screen run is incomplete")
    _require(summary.get("result_stage") == "HPO_VALIDATION_ONLY", "screen run accessed the wrong result stage")
    _require(summary.get("development_test_outcomes_evaluated") is False, "Development TEST entered screen")
    _require(summary.get("evaluation_outcomes_read") == 0, "Evaluation entered screen")
    _require(summary.get("cpu_fallback_used") is False, "screen used CPU fallback")
    _require(summary.get("cuda_training_tensors_verified") is True, "screen lacks CUDA tensor proof")
    _require(summary.get("parameter_changed") is True, "screen lacks a learned parameter update")
    _require(summary.get("baseline_id") == config.get("baseline_id"), "baseline identity differs from config")
    _require(summary.get("model_kind") == config.get("model_kind"), "model kind differs from config")
    _require(summary.get("seed") == config.get("seed"), "screen seed differs from config")
    _require(summary.get("device") == config.get("device"), "screen device differs from config")
    _require(summary.get("physical_gpu_index") == config.get("physical_gpu_index"), "physical GPU differs from config")
    _require(summary.get("checkpoint_selection") == "BEST_VALIDATION", "screen did not use best validation checkpoint")
    _require(
        summary.get("checkpoint_metric") == "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "screen used the wrong checkpoint metric",
    )
    scaler = summary.get("target_scaler", {})
    _require(scaler.get("fit_scope") == "TRAIN_ONLY", "target scaler is not train-only")
    _require(scaler.get("center_subtracted") is False, "target scaler changed the Delta zero")
    validation = summary.get("validation_metrics") or {}
    task_macro_spearman = _finite(validation.get("task_macro_spearman"), "task-macro Spearman")
    task_macro_standardized_mae = _finite(
        validation.get("task_macro_standardized_mae"), "task-macro standardized MAE"
    )
    _require(validation.get("defined_task_spearman_count") == validation.get("task_count"), "some validation task Spearman is undefined")
    task_metrics = validation.get("task_metrics") or {}
    _require(len(task_metrics) == int(validation["task_count"]), "validation task metric closure differs")
    raw_task_mae_by_task = {
        str(task): _finite(row.get("mae"), f"raw task MAE for {task}")
        for task, row in task_metrics.items()
    }
    candidate_control = str(summary["candidate_control"])
    candidate_control_summary = summary.get("candidate_control_summary") or {}
    if config["scientific_role"] == "MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL":
        _require(
            candidate_control == "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
            "permutation control identity differs",
        )
        _require(
            candidate_control_summary.get("permutation_stratum")
            == "EXACT_SOURCE_SEQUENCE_ENDPOINT_REGION",
            "permutation control stratum differs",
        )
        _require(
            candidate_control_summary.get("candidate_pool_membership_preserved") is True,
            "permutation control left recipient candidate support",
        )
        _require(
            candidate_control_summary.get("edit_distance_multiset_preserved") is True,
            "permutation control changed edit-distance support",
        )
        _require(
            int(candidate_control_summary.get("changed_candidate_sequence_count", 0)) > 0,
            "permutation control changed no candidates",
        )
    return {
        "scientific_role": config["scientific_role"],
        "baseline_id": summary["baseline_id"],
        "model_kind": summary["model_kind"],
        "target_scaling_mode": scaler["mode"],
        "candidate_control": candidate_control,
        "candidate_control_summary": dict(candidate_control_summary),
        "parameter_count": int(summary["parameter_count"]),
        "selected_epoch": int(summary["selected_epoch"]),
        "task_macro_spearman": task_macro_spearman,
        "within_run_task_macro_standardized_mae": task_macro_standardized_mae,
        "raw_task_mae_by_task": raw_task_mae_by_task,
        "target_scaler": dict(scaler),
        "validation_task_count": int(validation["task_count"]),
        "physical_gpu_index": int(summary["physical_gpu_index"]),
        "cuda_device_uuid": str(summary["cuda_device_uuid"]),
        "cpu_fallback_used": False,
        "evaluation_outcomes_read": 0,
        "output_directory": config["output_directory"],
    }


def adjudicate_screen(protocol: Mapping[str, Any], runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(protocol.get("status") == "FROZEN_DEVELOPMENT_ONLY_EXPLORATORY_SCREEN", "method-repair protocol is not frozen")
    by_role = {str(run["scientific_role"]): dict(run) for run in runs}
    _require(len(by_role) == len(runs), "scientific role is duplicated")
    factorial_roles = (
        "FACTORIAL_GLOBAL_RAW",
        "FACTORIAL_GLOBAL_SCALED",
        "FACTORIAL_EDIT_CENTERED_RAW",
        "FACTORIAL_EDIT_CENTERED_SCALED",
    )
    control_roles = (
        "MATCHED_SOURCE_ONLY_CONTROL",
        "MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL",
    )
    _require(set(factorial_roles + control_roles) == set(by_role), "screen role coverage is incomplete")
    global_raw = by_role["FACTORIAL_GLOBAL_RAW"]
    global_scaled = by_role["FACTORIAL_GLOBAL_SCALED"]
    edit_raw = by_role["FACTORIAL_EDIT_CENTERED_RAW"]
    edit_scaled = by_role["FACTORIAL_EDIT_CENTERED_SCALED"]
    source_only = by_role["MATCHED_SOURCE_ONLY_CONTROL"]
    permutation = by_role["MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL"]
    _require(global_raw["model_kind"] == global_scaled["model_kind"], "global factorial model kinds differ")
    _require(edit_raw["model_kind"] == edit_scaled["model_kind"], "edit-centered factorial model kinds differ")
    _require(permutation["model_kind"] == edit_scaled["model_kind"], "permutation control model kind is not matched")
    _require(source_only["model_kind"] != edit_scaled["model_kind"], "source-only control is not a distinct model kind")
    _require(global_raw["target_scaling_mode"] == edit_raw["target_scaling_mode"] == "NONE", "raw factorial target scaling differs")
    _require(
        global_scaled["target_scaling_mode"]
        == edit_scaled["target_scaling_mode"]
        == source_only["target_scaling_mode"]
        == permutation["target_scaling_mode"]
        == "TRAIN_TASK_ROBUST",
        "scaled factorial/control target scaling differs",
    )
    _require(global_raw["parameter_count"] == global_scaled["parameter_count"], "global factorial parameter counts differ")
    _require(
        edit_scaled["parameter_count"] == source_only["parameter_count"] == permutation["parameter_count"],
        "edit-scaled control parameter counts differ",
    )
    _require(edit_raw["candidate_control"] == edit_scaled["candidate_control"] == source_only["candidate_control"] == "NONE", "unexpected candidate control in main/source-only arm")
    _require(
        permutation["candidate_control"] == "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
        "permutation control identity differs",
    )
    robust_scalers = [
        run["target_scaler"]
        for run in (global_scaled, edit_scaled, source_only, permutation)
    ]
    _require(
        all(scaler.get("mode") == "TRAIN_TASK_ROBUST" for scaler in robust_scalers),
        "common metric scaler is not train-task robust",
    )
    _require(
        all(scaler == robust_scalers[0] for scaler in robust_scalers[1:]),
        "train-only robust scalers differ across matched runs",
    )
    common_task_keys = set(global_raw["raw_task_mae_by_task"])
    _require(common_task_keys, "validation task metrics are empty")
    _require(
        all(set(run["raw_task_mae_by_task"]) == common_task_keys for run in by_role.values()),
        "validation task identities differ across screen arms",
    )
    common_scaler = robust_scalers[0]
    for run in by_role.values():
        run["common_train_robust_task_macro_standardized_mae"] = sum(
            run["raw_task_mae_by_task"][task] / _scale_for_task(common_scaler, task)
            for task in common_task_keys
        ) / len(common_task_keys)
    ranked = sorted(
        (by_role[role] for role in factorial_roles),
        key=lambda run: (
            -run["task_macro_spearman"],
            run["common_train_robust_task_macro_standardized_mae"],
            run["scientific_role"],
        ),
    )
    winner = ranked[0]
    factorial_effects = {
        "robust_scaling_with_global_pooling": global_scaled["task_macro_spearman"] - global_raw["task_macro_spearman"],
        "robust_scaling_with_edit_centering": edit_scaled["task_macro_spearman"] - edit_raw["task_macro_spearman"],
        "edit_centering_with_raw_target": edit_raw["task_macro_spearman"] - global_raw["task_macro_spearman"],
        "edit_centering_with_robust_scaling": edit_scaled["task_macro_spearman"] - global_scaled["task_macro_spearman"],
    }
    edit_control_margins = {
        "over_source_only": edit_scaled["task_macro_spearman"] - source_only["task_macro_spearman"],
        "over_train_candidate_permutation": edit_scaled["task_macro_spearman"] - permutation["task_macro_spearman"],
    }
    improvement_over_reference = winner["task_macro_spearman"] - global_raw["task_macro_spearman"]
    controls_matched_to_winner = winner["scientific_role"] == "FACTORIAL_EDIT_CENTERED_SCALED"
    edit_controls_positive = all(margin > 0.0 for margin in edit_control_margins.values())
    controls_support_winner = controls_matched_to_winner and edit_controls_positive
    supports_confirmation = improvement_over_reference > 0.0 and controls_support_winner
    if supports_confirmation:
        status = "EXPLORATORY_SCREEN_SUPPORTS_FRESH_SEED_CONFIRMATION"
    elif improvement_over_reference > 0.0 and winner["scientific_role"] == "FACTORIAL_EDIT_CENTERED_RAW":
        status = "EXPLORATORY_EDIT_RAW_REQUIRES_MATCHED_CONTROLS"
    elif improvement_over_reference > 0.0 and winner["scientific_role"].startswith("FACTORIAL_GLOBAL"):
        status = "EXPLORATORY_GLOBAL_REPAIR_REQUIRES_MATCHED_CONTROLS"
    else:
        status = "EXPLORATORY_SCREEN_DOES_NOT_SUPPORT_CONFIRMATION"
    return {
        "schema_version": "route_a_v3_route2_method_repair_screen_adjudication.v1",
        "status": status,
        "scientific_claim_status": "EXPLORATORY_DEVELOPMENT_ONLY_NOT_ESTABLISHED",
        "selected_role": winner["scientific_role"],
        "selected_baseline_id": winner["baseline_id"],
        "selected_task_macro_spearman": winner["task_macro_spearman"],
        "selected_common_train_robust_task_macro_standardized_mae": winner[
            "common_train_robust_task_macro_standardized_mae"
        ],
        "task_macro_spearman_improvement_over_global_raw": improvement_over_reference,
        "factorial_effects": factorial_effects,
        "edit_centered_control_margins": edit_control_margins,
        "matched_controls_support_edit_scaled": edit_controls_positive,
        "matched_control_target_role": "FACTORIAL_EDIT_CENTERED_SCALED",
        "matched_controls_are_for_selected_role": controls_matched_to_winner,
        "matched_controls_support_selected_edit_model": controls_support_winner,
        "fresh_confirmation_seeds": protocol["fresh_confirmation_seeds"] if supports_confirmation else [],
        "guided_generation_status": protocol["guided_generation_status"],
        "evaluation_used_for_selection": False,
        "development_test_used_for_selection": False,
        "new_external_confirmation_required": protocol["post_exposure_boundary"]["new_external_confirmation_required_for_new_method_claim"],
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"raw_task_mae_by_task", "target_scaler"}
            }
            for run in sorted(by_role.values(), key=lambda run: run["scientific_role"])
        ],
    }


def execute(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), f"output already exists: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config_paths = protocol["screen_arms"] + protocol["matched_controls"]
    runs = []
    for relative_path in config_paths:
        config_path = REPO_ROOT / relative_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary_path = Path(config["output_directory"]) / "final_summary.json"
        _require(summary_path.is_file(), f"screen summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(validate_run(config, summary))
    result = adjudicate_screen(protocol, runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/route_a_v3_route2_method_repair_protocol_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.protocol, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
