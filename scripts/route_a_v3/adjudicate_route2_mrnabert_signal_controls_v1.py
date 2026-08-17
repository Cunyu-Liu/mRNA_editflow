#!/usr/bin/env python3
"""Adjudicate whether matched mRNABERT controls support final-seed confirmation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
SOURCE_ONLY_KIND = "delta_pretrained_mrnabert_edit_centered_source_only_control"
PERMUTATION = "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"


class SignalControlError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SignalControlError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _metrics(summary: Mapping[str, Any]) -> tuple[dict[str, float], float, float]:
    validation = summary.get("validation_metrics")
    _require(isinstance(validation, Mapping), "validation metrics are missing")
    task_metrics = validation.get("task_metrics")
    _require(isinstance(task_metrics, Mapping), "task metrics are missing")
    by_task = {
        str(task): _finite(row.get("spearman"), f"task Spearman: {task}")
        for task, row in task_metrics.items()
    }
    macro = _finite(validation.get("task_macro_spearman"), "task-macro Spearman")
    _require(
        math.isclose(macro, statistics.fmean(by_task.values()), rel_tol=0.0, abs_tol=1e-12),
        "task-macro Spearman does not replay from task metrics",
    )
    standardized_mae = _finite(
        validation.get("task_macro_standardized_mae"),
        "task-macro standardized MAE",
    )
    return by_task, macro, standardized_mae


def _validate_run(summary: Mapping[str, Any]) -> None:
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "control input is not a completed GPU run",
    )
    _require(summary.get("result_stage") == "HPO_VALIDATION_ONLY", "run is not HPO validation")
    _require(summary.get("evaluation_outcomes_read") == 0, "Evaluation entered control adjudication")
    _require(summary.get("development_test_outcomes_evaluated") is False, "Development TEST was evaluated")
    _require(summary.get("test_metrics") is None, "Development TEST metrics entered adjudication")


def adjudicate(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    primary: Mapping[str, Any],
    permutation: Mapping[str, Any],
    source_only: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version") == "route_a_v3_route2_mrnabert_signal_control_gate.v1",
        "unexpected signal-control protocol",
    )
    _require(
        protocol.get("status") == "FROZEN_BEFORE_MRNABERT_TERMINAL_LOSS_AND_CONTROL_OUTCOMES",
        "signal-control protocol was not prospectively frozen",
    )
    _require(comparison.get("development_test_opened") is False, "comparison opened Development TEST")
    _require(comparison.get("evaluation_opened") is False, "comparison opened Evaluation")
    for summary in (primary, permutation, source_only):
        _validate_run(summary)

    selected_loss = comparison.get("selected_loss_for_controls")
    _require(primary.get("loss_kind") == selected_loss, "primary loss differs from selected loss")
    _require(permutation.get("loss_kind") == selected_loss, "permutation loss differs")
    _require(source_only.get("loss_kind") == selected_loss, "source-only loss differs")
    _require(primary.get("model_kind") == PRIMARY_KIND, "primary model kind differs")
    _require(permutation.get("model_kind") == PRIMARY_KIND, "permutation model kind differs")
    _require(source_only.get("model_kind") == SOURCE_ONLY_KIND, "source-only model kind differs")
    _require(primary.get("candidate_control") == "NONE", "primary is already a control")
    _require(permutation.get("candidate_control") == PERMUTATION, "permutation control differs")
    _require(source_only.get("candidate_control") == "NONE", "source-only candidate control differs")
    _require(
        primary.get("seed") == permutation.get("seed") == source_only.get("seed"),
        "control seeds differ",
    )
    _require(
        primary.get("trainable_parameter_count")
        == permutation.get("trainable_parameter_count")
        == source_only.get("trainable_parameter_count"),
        "controls are not parameter matched",
    )

    primary_tasks, primary_macro, primary_mae = _metrics(primary)
    permutation_tasks, permutation_macro, permutation_mae = _metrics(permutation)
    source_tasks, source_macro, source_mae = _metrics(source_only)
    required_task_count = int(protocol["required_task_count"])
    _require(len(primary_tasks) == required_task_count, "primary task count differs")
    _require(set(primary_tasks) == set(permutation_tasks) == set(source_tasks), "control tasks differ")

    strongest = protocol["strongest_same_information_baseline"]
    baseline_macro = _finite(strongest.get("task_macro_spearman"), "strongest baseline macro")
    baseline_mae = _finite(
        strongest.get("task_macro_standardized_mae"), "strongest baseline standardized MAE"
    )
    source_margins = {
        task: primary_tasks[task] - source_tasks[task] for task in sorted(primary_tasks)
    }
    source_win_count = sum(value > 0.0 for value in source_margins.values())
    source_required = int(protocol["source_only_control_requirements"]["minimum_tasks_won"])

    eligible_tasks = list(
        protocol["candidate_permutation_control_requirements"]["eligible_tasks"]
    )
    _require(len(eligible_tasks) == len(set(eligible_tasks)), "permutation tasks are duplicated")
    _require(set(eligible_tasks) <= set(primary_tasks), "permutation task is absent")
    permutation_margins = {
        task: primary_tasks[task] - permutation_tasks[task] for task in eligible_tasks
    }
    permutation_win_count = sum(value > 0.0 for value in permutation_margins.values())
    permutation_mean_margin = statistics.fmean(permutation_margins.values())
    permutation_required = int(
        protocol["candidate_permutation_control_requirements"]["minimum_eligible_task_wins"]
    )

    checks = {
        "primary_beats_strongest_same_information_baseline": primary_macro > baseline_macro,
        "primary_task_median_positive": statistics.median(primary_tasks.values()) > 0.0,
        "primary_beats_source_only_macro": primary_macro > source_macro,
        "primary_beats_source_only_on_required_task_breadth": source_win_count >= source_required,
        "primary_beats_permutation_on_all_required_tasks": permutation_win_count >= permutation_required,
        "primary_permutation_required_task_mean_margin_positive": permutation_mean_margin > 0.0,
    }
    supported = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_mrnabert_signal_control_adjudication.v1",
        "status": (
            "MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION"
            if supported
            else "MRNABERT_SIGNAL_CONTROLS_DO_NOT_SUPPORT_FINAL_SEED_CONFIRMATION"
        ),
        "selected_loss": selected_loss,
        "seed": primary["seed"],
        "checks": checks,
        "primary": {
            "task_macro_spearman": primary_macro,
            "task_macro_standardized_mae": primary_mae,
            "task_median_spearman": statistics.median(primary_tasks.values()),
            "margin_over_strongest_same_information_baseline": primary_macro - baseline_macro,
            "standardized_mae_margin_vs_strongest_baseline": baseline_mae - primary_mae,
        },
        "source_only_control": {
            "task_macro_spearman": source_macro,
            "task_macro_standardized_mae": source_mae,
            "macro_margin": primary_macro - source_macro,
            "task_win_count": source_win_count,
            "required_task_win_count": source_required,
            "task_margins": source_margins,
        },
        "candidate_permutation_control": {
            "task_macro_spearman": permutation_macro,
            "task_macro_standardized_mae": permutation_mae,
            "eligible_task_margins": permutation_margins,
            "eligible_task_mean_margin": permutation_mean_margin,
            "eligible_task_win_count": permutation_win_count,
            "required_task_win_count": permutation_required,
        },
        "supports_final_seed_confirmation": supported,
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--permutation-summary", type=Path, required=True)
    parser.add_argument("--source-only-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = adjudicate(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        json.loads(args.comparison.read_text(encoding="utf-8")),
        json.loads(args.primary_summary.read_text(encoding="utf-8")),
        json.loads(args.permutation_summary.read_text(encoding="utf-8")),
        json.loads(args.source_only_summary.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
