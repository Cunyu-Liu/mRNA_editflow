#!/usr/bin/env python3
"""Adjudicate the prospectively frozen Critic V2 four-arm control screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


class CriticV2AdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2AdjudicationError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _validate_summary(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    arm_spec = protocol["arms"][arm]
    policy = protocol["frozen_training_policy"]
    _require(
        summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        f"{arm} is not a completed GPU run",
    )
    _require(summary.get("result_stage") == "HPO_VALIDATION_ONLY", f"{arm} stage differs")
    _require(summary.get("run_mode") == "FIXED_GROUPED_SPLIT", f"{arm} split differs")
    _require(summary.get("seed") == protocol["screen_seed"], f"{arm} seed differs")
    _require(summary.get("model_kind") == arm_spec["model_kind"], f"{arm} model kind differs")
    _require(summary.get("candidate_control") == arm_spec["candidate_control"], f"{arm} candidate control differs")
    for key, expected in policy.items():
        if key in {
            "model_kind",
            "optimizer_name",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "loss_kind",
            "huber_delta",
            "checkpoint_selection",
            "checkpoint_metric",
            "training_precision",
            "training_weighting_mode",
            "training_sampling_mode",
            "loss_aggregation_mode",
            "training_update_mode",
            "target_scaling_mode",
        }:
            if key == "target_scaling_mode":
                observed = summary.get("target_scaler", {}).get("mode")
            elif key == "epochs":
                observed = summary.get("final_training_epoch")
            else:
                observed = summary.get(key)
            if key == "model_kind":
                continue
            _require(observed == expected, f"{arm} frozen policy differs: {key}")
    _require(summary.get("development_test_outcomes_evaluated") is False, f"{arm} opened TEST")
    _require(summary.get("development_test_record_count_withheld") == 18292, f"{arm} TEST count differs")
    _require(summary.get("test_metrics") is None, f"{arm} contains TEST metrics")
    _require(summary.get("evaluation_outcomes_read") == 0, f"{arm} read Evaluation")
    _require(summary.get("cuda_training_tensors_verified") is True, f"{arm} CUDA tensors unverified")
    _require(summary.get("cpu_fallback_used") is False, f"{arm} used CPU fallback")
    _require(summary.get("parameter_changed") is True, f"{arm} has no learned update")
    _require(int(summary.get("optimizer_steps", 0)) > 0, f"{arm} has no optimizer steps")
    _require(0 <= int(summary.get("physical_gpu_index", -1)) <= 5, f"{arm} GPU is outside 0-5")

    validation = summary.get("validation_metrics")
    _require(isinstance(validation, Mapping), f"{arm} Validation metrics are absent")
    task_metrics = validation.get("task_metrics")
    _require(isinstance(task_metrics, Mapping), f"{arm} task metrics are absent")
    task_values = {
        str(task): _finite(row.get("spearman"), f"{arm} task Spearman {task}")
        for task, row in task_metrics.items()
    }
    required_task_count = int(protocol["control_gate"]["required_task_count"])
    _require(len(task_values) == required_task_count, f"{arm} task count differs")
    macro = _finite(validation.get("task_macro_spearman"), f"{arm} task-macro Spearman")
    _require(
        math.isclose(macro, statistics.fmean(task_values.values()), rel_tol=0.0, abs_tol=1e-12),
        f"{arm} task-macro does not replay",
    )
    standardized_mae = _finite(
        validation.get("task_macro_standardized_mae"),
        f"{arm} task-macro standardized MAE",
    )
    prediction_std = _finite(validation.get("prediction_std"), f"{arm} prediction std")
    target_std = _finite(validation.get("target_std"), f"{arm} target std")
    spread_ratio = _finite(
        validation.get("prediction_std_over_target_std"),
        f"{arm} prediction spread ratio",
    )
    _require(prediction_std > 0.0 and target_std > 0.0 and spread_ratio > 0.0, f"{arm} mean collapsed")
    return {
        "baseline_id": summary["baseline_id"],
        "model_kind": summary["model_kind"],
        "task_macro_spearman": macro,
        "task_macro_standardized_mae": standardized_mae,
        "task_median_spearman": statistics.median(task_values.values()),
        "positive_task_count": sum(value > 0.0 for value in task_values.values()),
        "prediction_std": prediction_std,
        "target_std": target_std,
        "prediction_std_over_target_std": spread_ratio,
        "task_spearman": task_values,
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
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_protocol.v1",
        "unexpected Critic V2 protocol",
    )
    _require(
        protocol.get("status") == "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES",
        "Critic V2 protocol was not frozen",
    )
    _require(protocol.get("development_test_outcomes_accessed") is False, "protocol opened TEST")
    _require(protocol.get("evaluation_outcomes_accessed") is False, "protocol opened Evaluation")
    expected_arms = set(protocol["arms"])
    _require(set(summaries) == expected_arms, "Critic V2 summary arms differ")
    rows = {
        arm: _validate_summary(summary, protocol, arm)
        for arm, summary in summaries.items()
    }
    parameter_shapes = {
        (
            row["trainable_parameter_count"],
            row["frozen_pretrained_parameter_count"],
            row["total_effective_parameter_count"],
            row["optimizer_steps"],
            row["final_training_epoch"],
        )
        for row in rows.values()
    }
    _require(len(parameter_shapes) == 1, "Critic V2 controls are not parameter/budget matched")
    task_sets = {tuple(sorted(row["task_spearman"])) for row in rows.values()}
    _require(len(task_sets) == 1, "Critic V2 control task sets differ")

    primary = rows["full"]
    baseline = protocol["strongest_same_information_baseline"]
    baseline_macro = _finite(baseline["task_macro_spearman"], "strongest baseline macro")
    baseline_mae = _finite(
        baseline["task_macro_standardized_mae"],
        "strongest baseline standardized MAE",
    )

    def task_margins(control_arm: str) -> dict[str, float]:
        return {
            task: primary["task_spearman"][task] - rows[control_arm]["task_spearman"][task]
            for task in sorted(primary["task_spearman"])
        }

    source_margins = task_margins("source_only")
    anchor_margins = task_margins("source_edit_metadata")
    permutation_tasks = list(protocol["control_gate"]["permutation_eligible_tasks"])
    _require(set(permutation_tasks) <= set(primary["task_spearman"]), "permutation task is absent")
    permutation_margins = {
        task: primary["task_spearman"][task]
        - rows["candidate_permutation"]["task_spearman"][task]
        for task in permutation_tasks
    }
    source_wins = sum(value > 0.0 for value in source_margins.values())
    anchor_wins = sum(value > 0.0 for value in anchor_margins.values())
    permutation_wins = sum(value > 0.0 for value in permutation_margins.values())
    checks = {
        "full_beats_strongest_same_information_baseline": (
            primary["task_macro_spearman"] > baseline_macro
        ),
        "full_task_median_positive": primary["task_median_spearman"] > 0.0,
        "full_beats_source_only_macro": (
            primary["task_macro_spearman"] > rows["source_only"]["task_macro_spearman"]
        ),
        "full_beats_source_only_task_breadth": (
            source_wins
            >= int(protocol["control_gate"]["minimum_task_wins_over_source_only"])
        ),
        "full_beats_source_edit_metadata_macro": (
            primary["task_macro_spearman"]
            > rows["source_edit_metadata"]["task_macro_spearman"]
        ),
        "full_beats_source_edit_metadata_task_breadth": (
            anchor_wins
            >= int(
                protocol["control_gate"][
                    "minimum_task_wins_over_source_edit_metadata"
                ]
            )
        ),
        "full_beats_permutation_on_supported_tasks": (
            permutation_wins
            >= int(
                protocol["control_gate"][
                    "minimum_permutation_eligible_task_wins"
                ]
            )
        ),
        "full_permutation_supported_task_mean_margin_positive": (
            statistics.fmean(permutation_margins.values()) > 0.0
        ),
    }
    supported = all(checks.values())
    for row in rows.values():
        row.pop("task_spearman")
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "status": (
            "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS"
            if supported
            else "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
        ),
        "checks": checks,
        "strongest_same_information_baseline": {
            "baseline_id": baseline["baseline_id"],
            "task_macro_spearman": baseline_macro,
            "task_macro_standardized_mae": baseline_mae,
        },
        "arms": rows,
        "margins": {
            "full_over_strongest_baseline_task_macro_spearman": (
                primary["task_macro_spearman"] - baseline_macro
            ),
            "full_standardized_mae_margin_vs_strongest_baseline": (
                baseline_mae - primary["task_macro_standardized_mae"]
            ),
            "full_over_source_only_task_macro_spearman": (
                primary["task_macro_spearman"]
                - rows["source_only"]["task_macro_spearman"]
            ),
            "full_over_source_edit_metadata_task_macro_spearman": (
                primary["task_macro_spearman"]
                - rows["source_edit_metadata"]["task_macro_spearman"]
            ),
            "source_only_task_win_count": source_wins,
            "source_edit_metadata_task_win_count": anchor_wins,
            "candidate_permutation_eligible_task_margins": permutation_margins,
        },
        "frozen_confirmation_seeds": protocol["frozen_confirmation_seeds"],
        "supports_three_frozen_seeds": supported,
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--candidate-permutation-summary", type=Path, required=True)
    parser.add_argument("--source-only-summary", type=Path, required=True)
    parser.add_argument("--source-edit-metadata-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"adjudication output already exists: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summaries = {
        "full": json.loads(args.full_summary.read_text(encoding="utf-8")),
        "candidate_permutation": json.loads(
            args.candidate_permutation_summary.read_text(encoding="utf-8")
        ),
        "source_only": json.loads(args.source_only_summary.read_text(encoding="utf-8")),
        "source_edit_metadata": json.loads(
            args.source_edit_metadata_summary.read_text(encoding="utf-8")
        ),
    }
    result = adjudicate(protocol, summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
