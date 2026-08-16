#!/usr/bin/env python3
"""Freeze one Development-validation learning rate per Route 2 neural profile."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


class HpoSelectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HpoSelectionError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def select(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config["schema_version"] == "route_a_v3_route2_neural_hpo_selection_config.v1", "unexpected selection schema")
    _require(config["selection_pool"] == "DEVELOPMENT_VALIDATION", "selection is not Development validation")
    _require(config["evaluation_outcomes_accessed"] is False, "selection accessed Evaluation")
    expected_per_group = int(config["expected_trials_per_profile"])
    _require(expected_per_group == 2, "each profile must retain the frozen two-trial HPO budget")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trial_ids = set()
    for spec in config["trials"]:
        trial_id = str(spec["trial_id"])
        _require(trial_id not in trial_ids, f"trial is duplicated: {trial_id}")
        trial_ids.add(trial_id)
        summary_path = Path(spec["training_summary_path"])
        evaluation_path = Path(spec["validation_evaluation_path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _require(summary["status"] == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE", f"trial is incomplete: {trial_id}")
        _require(summary["result_stage"] == "HPO_VALIDATION_ONLY", f"trial is not HPO-only: {trial_id}")
        _require(summary["development_test_outcomes_evaluated"] is False, f"trial opened Development TEST: {trial_id}")
        _require(summary.get("test_metrics") is None, f"trial contains Development TEST metrics: {trial_id}")
        _require(summary["evaluation_outcomes_read"] == 0, f"trial read Evaluation: {trial_id}")
        _require(summary["cpu_fallback_used"] is False, f"trial used CPU fallback: {trial_id}")
        _require(summary["cuda_training_tensors_verified"] is True, f"trial lacks CUDA tensor proof: {trial_id}")
        _require(summary["parameter_changed"] is True and int(summary["optimizer_steps"]) > 0, f"trial lacks a learned update: {trial_id}")
        physical_index = int(summary["physical_gpu_index"])
        _require(summary["device"] == f"cuda:{physical_index}", f"trial device provenance differs: {trial_id}")
        _require(int(summary["cuda_device_index"]) == physical_index, f"trial CUDA index differs: {trial_id}")
        _require(bool(summary["cuda_device_uuid"]), f"trial CUDA UUID is absent: {trial_id}")
        _require(_finite(summary["cuda_total_memory_mb"], "CUDA memory") > 0.0, f"trial CUDA memory is absent: {trial_id}")
        _require(evaluation["split"] == "VALIDATION", f"evaluation is not validation-only: {trial_id}")
        _require(evaluation["evaluation_release_state"] == "CLOSED", f"Evaluation release opened: {trial_id}")
        metrics = evaluation["metrics"]
        task_count = int(metrics["task_count"])
        defined_task_count = int(metrics["task_spearman_defined_count"])
        _require(0 <= defined_task_count <= task_count and task_count > 0, f"task counts are invalid: {trial_id}")
        task_values = [
            _finite(row["spearman"], "task Spearman")
            for row in metrics["task_numeric"].values() if row.get("spearman") is not None
        ]
        _require(len(task_values) == defined_task_count, f"defined task count differs: {trial_id}")
        complete_task_macro = (
            _finite(metrics["task_macro_spearman"], "task-macro Spearman")
            if defined_task_count == task_count else None
        )
        grouped[str(spec["profile_id"])].append({
            "trial_id": trial_id,
            "baseline_id": str(summary["baseline_id"]),
            "model_kind": str(summary["model_kind"]),
            "parameter_count": int(summary["parameter_count"]),
            "task_count": task_count,
            "task_spearman_defined_count": defined_task_count,
            "task_macro_spearman": complete_task_macro,
            "defined_task_mean_spearman": None if not task_values else sum(task_values) / len(task_values),
            "source_macro_mae": _finite(metrics["source_macro_mae"], "source-macro MAE"),
            "training_summary_path": str(summary_path),
            "validation_evaluation_path": str(evaluation_path),
            "training_config_path": str(spec["training_config_path"]),
            "physical_gpu_index": physical_index,
            "cuda_device_uuid": str(summary["cuda_device_uuid"]),
        })
    _require(grouped, "no HPO trials were supplied")
    selections = {}
    for profile_id, trials in sorted(grouped.items()):
        _require(len(trials) == expected_per_group, f"profile does not have exactly two trials: {profile_id}")
        ranked = sorted(
            trials,
            key=lambda row: (
                row["task_macro_spearman"] is None,
                -row["task_spearman_defined_count"],
                -(
                    row["task_macro_spearman"]
                    if row["task_macro_spearman"] is not None
                    else row["defined_task_mean_spearman"]
                    if row["defined_task_mean_spearman"] is not None
                    else -math.inf
                ),
                row["parameter_count"],
                row["source_macro_mae"], row["trial_id"],
            ),
        )
        selections[profile_id] = {
            "selected_trial_id": ranked[0]["trial_id"],
            "selection_primary_metric": (
                "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN"
                if ranked[0]["task_macro_spearman"] is not None
                else "DEFINED_TASK_COUNT_THEN_DEFINED_TASK_MEAN_SPEARMAN_FALLBACK"
            ),
            "selected_training_config_path": ranked[0]["training_config_path"],
            "all_trials_ranked": ranked,
        }
    return {
        "schema_version": "route_a_v3_route2_neural_hpo_selection.v1",
        "status": "NEURAL_HPO_LEARNING_RATES_FROZEN_BY_PROFILE",
        "selection_pool": "DEVELOPMENT_VALIDATION",
        "profile_count": len(selections),
        "selections": selections,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = select(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
