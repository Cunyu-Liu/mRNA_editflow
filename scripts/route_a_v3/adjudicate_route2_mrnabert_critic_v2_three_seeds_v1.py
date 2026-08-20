#!/usr/bin/env python3
"""Adjudicate the exact three Critic V2 seeds without opening TEST."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_SCHEMA = "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol.v1"
CONTROL_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1"
)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class CriticV2ThreeSeedAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2ThreeSeedAdjudicationError(message)


def _diagnostic_float(value: Any) -> tuple[float | None, bool]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None, True
    result = float(value)
    if not math.isfinite(result):
        return None, True
    return result, False


def _validate_seed(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    control_adjudication: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    policy = protocol["frozen_training_policy"]
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        f"seed {seed} is not a completed GPU run",
    )
    _require(summary.get("seed") == seed, f"seed {seed} identity differs")
    _require(summary.get("result_stage") == "FROZEN_DEVELOPMENT_VALIDATION", f"seed {seed} stage differs")
    _require(summary.get("run_mode") == "FIXED_GROUPED_SPLIT", f"seed {seed} split differs")
    _require(summary.get("model_kind") == PRIMARY_KIND, f"seed {seed} model kind differs")
    _require(summary.get("candidate_control") == "NONE", f"seed {seed} is a control")
    for key, expected in policy.items():
        if key == "target_scaling_mode":
            observed = summary.get("target_scaler", {}).get("mode")
        elif key == "epochs":
            observed = summary.get("final_training_epoch")
        else:
            observed = summary.get(key)
        _require(observed == expected, f"seed {seed} frozen policy differs: {key}")
    _require(summary.get("development_test_outcomes_evaluated") is False, f"seed {seed} opened TEST")
    _require(summary.get("development_test_record_count_withheld") == 18292, f"seed {seed} TEST count differs")
    _require(summary.get("test_metrics") is None, f"seed {seed} contains TEST metrics")
    _require(summary.get("evaluation_outcomes_read") == 0, f"seed {seed} read Evaluation")
    _require(summary.get("cuda_training_tensors_verified") is True, f"seed {seed} CUDA tensors unverified")
    _require(summary.get("cpu_fallback_used") is False, f"seed {seed} used CPU fallback")
    _require(summary.get("parameter_changed") is True, f"seed {seed} has no learned update")
    _require(int(summary.get("optimizer_steps", 0)) > 0, f"seed {seed} has no optimizer steps")
    _require(0 <= int(summary.get("physical_gpu_index", -1)) <= 5, f"seed {seed} GPU is outside 0-5")

    validation = summary.get("validation_metrics")
    _require(isinstance(validation, Mapping), f"seed {seed} Validation metrics are absent")
    task_metrics = validation.get("task_metrics")
    _require(isinstance(task_metrics, Mapping), f"seed {seed} task metrics are absent")
    _require(
        len(task_metrics) == int(protocol["required_task_count"]),
        f"seed {seed} task count differs",
    )

    nonfinite = False
    finite_task_values = []
    for task, task_row in task_metrics.items():
        _require(isinstance(task_row, Mapping), f"seed {seed} task metric {task} is malformed")
        value, bad = _diagnostic_float(task_row.get("spearman"))
        nonfinite = nonfinite or bad
        if value is not None:
            finite_task_values.append(value)
    macro, bad = _diagnostic_float(validation.get("task_macro_spearman"))
    nonfinite = nonfinite or bad
    standardized_mae, bad = _diagnostic_float(
        validation.get("task_macro_standardized_mae")
    )
    nonfinite = nonfinite or bad
    prediction_std, bad = _diagnostic_float(validation.get("prediction_std"))
    nonfinite = nonfinite or bad
    target_std, bad = _diagnostic_float(validation.get("target_std"))
    nonfinite = nonfinite or bad
    reported_ratio, bad = _diagnostic_float(
        validation.get("prediction_std_over_target_std")
    )
    nonfinite = nonfinite or bad

    task_macro_replays = (
        macro is not None
        and len(finite_task_values) == int(protocol["required_task_count"])
        and math.isclose(
            macro,
            statistics.fmean(finite_task_values),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    recomputed_ratio = None
    if prediction_std is not None and target_std is not None and target_std > 0.0:
        recomputed_ratio = prediction_std / target_std
    spread_ratio_replays = (
        reported_ratio is not None
        and recomputed_ratio is not None
        and math.isclose(reported_ratio, recomputed_ratio, rel_tol=1e-9, abs_tol=1e-12)
    )
    mean_collapse = (
        prediction_std is None
        or target_std is None
        or reported_ratio is None
        or prediction_std <= 0.0
        or target_std <= 0.0
        or reported_ratio <= 0.0
    )
    baseline_macro = float(
        protocol["strongest_same_information_baseline"]["task_macro_spearman"]
    )
    margin = None if macro is None else macro - baseline_macro
    control_gaps = {
        arm: None if macro is None else macro - float(control_adjudication["arms"][arm]["task_macro_spearman"])
        for arm in ("candidate_permutation", "source_only", "source_edit_metadata")
    }
    log10_ratio_distance = None
    if reported_ratio is not None and reported_ratio > 0.0:
        log10_ratio_distance = abs(math.log10(reported_ratio))

    return {
        "seed": seed,
        "baseline_id": summary["baseline_id"],
        "task_macro_spearman": macro,
        "task_median_spearman": (
            statistics.median(finite_task_values)
            if len(finite_task_values) == int(protocol["required_task_count"])
            else None
        ),
        "task_macro_standardized_mae": standardized_mae,
        "prediction_std": prediction_std,
        "target_std": target_std,
        "prediction_std_over_target_std": reported_ratio,
        "margin_over_strongest_same_information_baseline": margin,
        "positive_task_count": sum(value > 0.0 for value in finite_task_values),
        "task_macro_gaps_over_each_control": control_gaps,
        "nonfinite_metric_detected": nonfinite,
        "mean_collapse_detected": mean_collapse,
        "scale_diagnostics": {
            "recomputed_prediction_std_over_target_std": recomputed_ratio,
            "reported_ratio_replays": spread_ratio_replays,
            "absolute_log10_ratio_distance_from_one": log10_ratio_distance,
            "thresholded_abnormal_scale_verdict": "NOT_PREDECLARED_REPORT_ONLY",
        },
        "task_macro_replays": task_macro_replays,
        "trainable_parameter_count": int(summary["trainable_parameter_count"]),
        "frozen_pretrained_parameter_count": int(summary["frozen_pretrained_parameter_count"]),
        "total_effective_parameter_count": int(summary["total_effective_parameter_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "selected_epoch": int(summary["selected_epoch"]),
        "final_training_epoch": int(summary["final_training_epoch"]),
        "physical_gpu_index": int(summary["physical_gpu_index"]),
    }


def adjudicate(
    protocol: Mapping[str, Any],
    control_adjudication: Mapping[str, Any],
    summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "unexpected three-seed protocol")
    _require(
        protocol.get("status") == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 three-seed protocol was not frozen",
    )
    _require(protocol.get("development_test_outcomes_accessed") is False, "protocol opened TEST")
    _require(protocol.get("evaluation_outcomes_accessed") is False, "protocol opened Evaluation")
    _require(
        control_adjudication.get("schema_version") == CONTROL_ADJUDICATION_SCHEMA,
        "unexpected control adjudication",
    )
    controls_pass = (
        control_adjudication.get("status")
        == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS"
        and control_adjudication.get("supports_three_frozen_seeds") is True
    )
    _require(controls_pass, "Critic V2 controls did not authorize these seeds")
    _require(control_adjudication.get("development_test_opened") is False, "control adjudication opened TEST")
    _require(control_adjudication.get("evaluation_opened") is False, "control adjudication opened Evaluation")

    required_seeds = [int(seed) for seed in protocol["required_seeds"]]
    _require(len(required_seeds) == len(summaries) == 3, "exactly three seed summaries are required")
    by_seed = {int(summary.get("seed")): summary for summary in summaries}
    _require(len(by_seed) == 3, "confirmation seed is duplicated")
    _require(set(by_seed) == set(required_seeds), "confirmation seed set differs")
    _require(
        required_seeds == [int(seed) for seed in control_adjudication["frozen_confirmation_seeds"]],
        "control adjudication authorized a different seed set",
    )
    frozen_baseline = protocol["strongest_same_information_baseline"]
    observed_baseline = control_adjudication["strongest_same_information_baseline"]
    _require(frozen_baseline["baseline_id"] == observed_baseline["baseline_id"], "strongest baseline identity differs")
    _require(
        math.isclose(
            float(frozen_baseline["task_macro_spearman"]),
            float(observed_baseline["task_macro_spearman"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "strongest baseline value differs",
    )

    rows = [
        _validate_seed(by_seed[seed], protocol, control_adjudication, seed)
        for seed in required_seeds
    ]
    matched_shapes = {
        (
            row["trainable_parameter_count"],
            row["frozen_pretrained_parameter_count"],
            row["total_effective_parameter_count"],
            row["optimizer_steps"],
            row["final_training_epoch"],
        )
        for row in rows
    }
    _require(len(matched_shapes) == 1, "Critic V2 confirmation seeds are not budget matched")

    checks = {
        "control_adjudication_supports_three_frozen_seeds": controls_pass,
        "all_seed_metrics_finite": all(not row["nonfinite_metric_detected"] for row in rows),
        "all_seed_prediction_spreads_positive": all(not row["mean_collapse_detected"] for row in rows),
        "all_seed_task_macros_replay": all(row["task_macro_replays"] for row in rows),
        "all_seed_spread_ratios_replay": all(row["scale_diagnostics"]["reported_ratio_replays"] for row in rows),
        "all_three_seed_margins_over_strongest_baseline_positive": all(
            row["margin_over_strongest_same_information_baseline"] is not None
            and row["margin_over_strongest_same_information_baseline"] > 0.0
            for row in rows
        ),
    }
    supported = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1",
        "status": (
            "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST"
            if supported
            else "CRITIC_V2_THREE_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST"
        ),
        "checks": checks,
        "strongest_same_information_baseline": frozen_baseline,
        "seed_results": rows,
        "supports_single_frozen_development_test": supported,
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--control-adjudication", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"adjudication output already exists: {args.output}")
    result = adjudicate(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        json.loads(args.control_adjudication.read_text(encoding="utf-8")),
        [json.loads(path.read_text(encoding="utf-8")) for path in args.summary],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
