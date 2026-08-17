#!/usr/bin/env python3
"""Adjudicate the frozen Development-only task-gradient norm repair arm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping


class TaskGradientNormAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskGradientNormAdjudicationError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, name: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{name} is not finite")
    return number


def _validate_run(summary: Mapping[str, Any], role: str) -> None:
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        f"{role} is incomplete",
    )
    _require(summary.get("cpu_fallback_used") is False, f"{role} used CPU fallback")
    _require(
        summary.get("cuda_training_tensors_verified") is True,
        f"{role} lacks CUDA tensor proof",
    )
    physical_index = int(summary["physical_gpu_index"])
    _require(summary.get("device") == f"cuda:{physical_index}", f"{role} device differs")
    _require(bool(summary.get("cuda_device_uuid")), f"{role} CUDA UUID is absent")
    _require(summary.get("parameter_changed") is True, f"{role} did not update parameters")
    _require(int(summary.get("optimizer_steps", 0)) > 0, f"{role} has no optimizer steps")
    _require(
        summary.get("development_test_outcomes_evaluated") is False,
        f"{role} opened Development TEST",
    )
    _require(int(summary.get("evaluation_outcomes_read", -1)) == 0, f"{role} opened Evaluation")


def _task_spearman(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics = summary["validation_metrics"]
    _require(int(metrics["task_count"]) == 9, "validation task count differs")
    _require(int(metrics["defined_task_spearman_count"]) == 9, "task Spearman is undefined")
    values = {
        str(task): _finite(row["spearman"], f"Spearman for {task}")
        for task, row in metrics["task_metrics"].items()
    }
    _require(len(values) == 9, "task metric mapping differs")
    return values


def adjudicate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_task_gradient_norm_repair_protocol.v1",
        "protocol schema differs",
    )
    _require(
        protocol.get("status") == "FROZEN_DEVELOPMENT_ONLY_BEFORE_GRADNORM_ARM_OUTCOME",
        "protocol was not frozen before outcome",
    )
    candidate = _load(Path(protocol["candidate_output"]))
    references = protocol["frozen_references"]
    global_raw = _load(Path(references["global_raw_summary"]))
    global_scaled = _load(Path(references["global_scaled_summary"]))
    for role, summary in (
        ("candidate", candidate),
        ("global_raw", global_raw),
        ("global_scaled", global_scaled),
    ):
        _validate_run(summary, role)

    _require(
        candidate.get("training_update_mode")
        == "TRAIN_TASK_GRADIENT_NORM_CALIBRATED",
        "candidate did not use the frozen training update mode",
    )
    _require(
        candidate.get("model_kind") == "delta_edit_centered_antisymmetric",
        "candidate model kind differs",
    )
    _require(
        candidate.get("target_scaler") == global_scaled.get("target_scaler"),
        "candidate and reference target scalers differ",
    )
    calibration = candidate.get("task_gradient_calibration") or {}
    _require(
        calibration.get("fit_scope") == "TRAIN_ONLY_BEFORE_FIRST_OPTIMIZER_STEP",
        "calibration scope differs",
    )
    _require(calibration.get("cuda_losses_verified") is True, "calibration lacks CUDA proof")
    _require(int(calibration.get("task_count", -1)) == 7, "calibration task count differs")
    _require(int(calibration.get("optimizer_steps", -1)) == 0, "calibration optimized parameters")
    _require(int(calibration.get("parameter_updates", -1)) == 0, "calibration updated parameters")
    multipliers = calibration.get("loss_multipliers") or {}
    _require(len(multipliers) == 7, "calibration multiplier count differs")
    _require(
        all(math.isfinite(float(value)) and float(value) > 0.0 for value in multipliers.values()),
        "calibration multiplier is invalid",
    )

    candidate_tasks = _task_spearman(candidate)
    global_raw_tasks = _task_spearman(global_raw)
    global_scaled_tasks = _task_spearman(global_scaled)
    _require(
        set(candidate_tasks) == set(global_raw_tasks) == set(global_scaled_tasks),
        "task sets differ",
    )
    candidate_macro = _finite(
        candidate["validation_metrics"]["task_macro_spearman"], "candidate macro"
    )
    global_scaled_macro = _finite(
        global_scaled["validation_metrics"]["task_macro_spearman"],
        "global-scaled macro",
    )
    legacy_macro = _finite(
        references["legacy_best_task_macro_spearman"], "legacy macro"
    )
    _require(
        math.isclose(
            candidate_macro,
            sum(candidate_tasks.values()) / len(candidate_tasks),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "candidate macro does not match task metrics",
    )
    _require(
        math.isclose(
            global_scaled_macro,
            sum(global_scaled_tasks.values()) / len(global_scaled_tasks),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "global-scaled macro does not match task metrics",
    )
    candidate_mae = _finite(
        candidate["validation_metrics"]["task_macro_standardized_mae"],
        "candidate standardized MAE",
    )
    legacy_mae = _finite(
        references["legacy_best_common_train_robust_task_macro_standardized_mae"],
        "legacy standardized MAE",
    )
    task_margins = {
        task: candidate_tasks[task] - global_raw_tasks[task]
        for task in sorted(candidate_tasks)
    }
    task_wins = sum(value > 0.0 for value in task_margins.values())
    task_median = median(candidate_tasks.values())
    checks = {
        "beats_global_scaled_macro": candidate_macro > global_scaled_macro,
        "beats_legacy_macro": candidate_macro > legacy_macro,
        "positive_task_median": task_median > 0.0,
        "at_least_five_task_wins_over_global_raw": task_wins >= 5,
        "standardized_mae_not_worse_than_legacy": candidate_mae <= legacy_mae,
    }
    supports_controls = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_task_gradient_norm_repair_adjudication.v1",
        "status": (
            "EXPLORATORY_GRADNORM_SUPPORTS_MATCHED_CONTROLS"
            if supports_controls
            else "EXPLORATORY_GRADNORM_NO_GO"
        ),
        "scientific_claim_status": "DEVELOPMENT_ONLY_NOT_ESTABLISHED",
        "candidate_baseline_id": candidate["baseline_id"],
        "candidate_task_macro_spearman": candidate_macro,
        "global_scaled_task_macro_spearman": global_scaled_macro,
        "legacy_best_task_macro_spearman": legacy_macro,
        "candidate_task_macro_standardized_mae": candidate_mae,
        "legacy_best_task_macro_standardized_mae": legacy_mae,
        "candidate_task_median_spearman": task_median,
        "task_wins_over_global_raw": task_wins,
        "task_margins_over_global_raw": task_margins,
        "advance_checks": checks,
        "matched_controls_authorized": supports_controls,
        "fresh_confirmation_seeds": [],
        "guided_generation_status": "BLOCKED",
        "development_test_used_for_selection": False,
        "evaluation_used_for_selection": False,
        "training_update_mode": candidate["training_update_mode"],
        "task_gradient_calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = adjudicate(_load(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
