#!/usr/bin/env python3
"""Adjudicate the pre-frozen Development-only generation evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class IndependentEvaluatorAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentEvaluatorAdjudicationError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def adjudicate(summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        summary["schema_version"] == "route_a_v3_route2_delta_predictor_training.v1",
        "unexpected training summary schema",
    )
    _require(
        protocol["schema_version"] == "route_a_v3_route2_generation_matched_compute_repair_protocol.v1",
        "unexpected protocol schema",
    )
    qualification = protocol["independent_evaluator_qualification"]
    metrics = summary["validation_metrics"]
    _require(isinstance(metrics, Mapping), "Development Validation metrics are absent")
    task_metrics = metrics["task_metrics"]
    _require(isinstance(task_metrics, Mapping) and task_metrics, "task metrics are absent")
    task_macro_spearman = _finite(metrics["task_macro_spearman"], "task-macro Spearman")
    positive_task_count = sum(
        _finite(row["spearman"], f"task Spearman {task}") > 0.0
        for task, row in task_metrics.items()
    )
    threshold = _finite(
        qualification["minimum_task_macro_spearman_exclusive"],
        "task-macro qualification threshold",
    )
    minimum_positive_tasks = int(qualification["minimum_positive_task_count"])
    checks = {
        "run_completed": summary["status"] == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "train_only_frozen_validation_stage": (
            summary["result_stage"] == "FROZEN_DEVELOPMENT_VALIDATION"
            and summary["development_validation_folded_into_training"] is False
        ),
        "development_test_withheld": (
            summary["development_test_outcomes_evaluated"] is False
            and int(summary["development_test_record_count_withheld"]) > 0
            and summary["test_metrics"] is None
        ),
        "evaluation_outcomes_closed": int(summary["evaluation_outcomes_read"]) == 0,
        "cuda_training_verified": (
            str(summary["device"]).startswith("cuda:")
            and summary["cuda_training_tensors_verified"] is True
            and summary["cpu_fallback_used"] is False
            and summary["parameter_changed"] is True
            and int(summary["optimizer_steps"]) > 0
        ),
        "architecture_distinct_from_guiding_critic": (
            protocol["guide_evaluator_architecture_distinct"] is True
            and summary["model_kind"] == protocol["independent_evaluator_model_kind"]
            and summary["model_kind"] != protocol["guiding_model_kind"]
        ),
        "all_expected_tasks_defined": (
            int(metrics["task_count"]) == len(task_metrics)
            and int(metrics["defined_task_spearman_count"]) == len(task_metrics)
        ),
        "task_macro_exceeds_exact_source_permutation": task_macro_spearman > threshold,
        "positive_task_breadth_reached": positive_task_count >= minimum_positive_tasks,
    }
    qualified = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_independent_generation_evaluator_adjudication.v1",
        "status": (
            "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
            if qualified
            else "INDEPENDENT_GENERATION_EVALUATOR_NO_GO"
        ),
        "candidate_rerun_authorized": qualified,
        "task_macro_spearman": task_macro_spearman,
        "task_macro_threshold_exclusive": threshold,
        "task_macro_margin": task_macro_spearman - threshold,
        "positive_task_count": positive_task_count,
        "minimum_positive_task_count": minimum_positive_tasks,
        "task_count": len(task_metrics),
        "parameter_count": int(summary["parameter_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "selected_epoch": int(summary["selected_epoch"]),
        "final_training_epoch": int(summary["final_training_epoch"]),
        "physical_gpu_index": int(summary["physical_gpu_index"]),
        "cuda_device_uuid": summary["cuda_device_uuid"],
        "checks": checks,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = adjudicate(
        json.loads(args.training_summary.read_text(encoding="utf-8")),
        json.loads(args.protocol.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
