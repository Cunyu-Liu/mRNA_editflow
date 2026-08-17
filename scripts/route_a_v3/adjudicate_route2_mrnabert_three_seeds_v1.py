#!/usr/bin/env python3
"""Adjudicate three frozen-validation seeds before opening Development TEST."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class ThreeSeedError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeSeedError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def adjudicate(
    protocol: Mapping[str, Any], summaries: list[Mapping[str, Any]]
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version") == "route_a_v3_route2_mrnabert_three_seed_gate.v1",
        "unexpected three-seed protocol",
    )
    _require(protocol.get("status") == "FROZEN_BEFORE_FINAL_SEED_OUTCOMES", "gate was not frozen")
    required_seeds = [int(value) for value in protocol["required_seeds"]]
    _require(len(summaries) == len(required_seeds) == 3, "exactly three seed summaries are required")
    by_seed = {int(row.get("seed")): row for row in summaries}
    _require(set(by_seed) == set(required_seeds), "final seed set differs")
    _require(len(by_seed) == len(summaries), "final seed is duplicated")

    losses = set()
    model_shapes = set()
    rows = []
    baseline = _finite(
        protocol.get("strongest_same_information_baseline_task_macro_spearman"),
        "strongest same-information baseline",
    )
    required_task_count = int(protocol["required_task_count"])
    for seed in required_seeds:
        summary = by_seed[seed]
        _require(
            summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
            f"seed {seed} is not complete",
        )
        _require(summary.get("result_stage") == "FROZEN_DEVELOPMENT_VALIDATION", "seed stage differs")
        _require(summary.get("run_mode") == "FIXED_GROUPED_SPLIT", "seed split mode differs")
        _require(summary.get("model_kind") == PRIMARY_KIND, "seed model kind differs")
        _require(summary.get("candidate_control") == "NONE", "seed is a control")
        _require(summary.get("evaluation_outcomes_read") == 0, "Evaluation entered final seeds")
        _require(summary.get("development_test_outcomes_evaluated") is False, "TEST entered final seeds")
        _require(summary.get("test_metrics") is None, "TEST metrics entered final seeds")
        validation = summary.get("validation_metrics")
        _require(isinstance(validation, Mapping), "seed validation metrics are missing")
        task_metrics = validation.get("task_metrics")
        _require(isinstance(task_metrics, Mapping), "seed task metrics are missing")
        task_values = [
            _finite(task.get("spearman"), f"seed {seed} task Spearman")
            for task in task_metrics.values()
        ]
        _require(len(task_values) == required_task_count, "seed task count differs")
        macro = _finite(validation.get("task_macro_spearman"), "seed task-macro Spearman")
        _require(
            math.isclose(macro, statistics.fmean(task_values), rel_tol=0.0, abs_tol=1e-12),
            "seed task-macro does not replay",
        )
        median = statistics.median(task_values)
        improvement = macro - baseline
        losses.add(str(summary.get("loss_kind")))
        model_shapes.add((
            int(summary.get("trainable_parameter_count")),
            int(summary.get("final_training_epoch")),
        ))
        rows.append({
            "seed": seed,
            "task_macro_spearman": macro,
            "task_median_spearman": median,
            "improvement_over_strongest_same_information_baseline": improvement,
            "directional_pass": improvement > 0.0 and median > 0.0,
        })
    _require(len(losses) == 1, "final seeds use different losses")
    _require(len(model_shapes) == 1, "final seeds use different model/budget shapes")
    directionally_consistent = all(row["directional_pass"] for row in rows)
    return {
        "schema_version": "route_a_v3_route2_mrnabert_three_seed_adjudication.v1",
        "status": (
            "THREE_FINAL_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST"
            if directionally_consistent
            else "THREE_FINAL_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST"
        ),
        "loss_kind": next(iter(losses)),
        "strongest_same_information_baseline_task_macro_spearman": baseline,
        "seed_results": rows,
        "all_seed_improvements_positive": all(
            row["improvement_over_strongest_same_information_baseline"] > 0.0
            for row in rows
        ),
        "all_seed_task_medians_positive": all(row["task_median_spearman"] > 0.0 for row in rows),
        "supports_single_frozen_development_test": directionally_consistent,
        "single_frozen_test_seed": int(protocol["single_frozen_test_seed"]),
        "development_test_opened": False,
        "evaluation_opened": False,
        "guided_generation_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = adjudicate(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        [json.loads(path.read_text(encoding="utf-8")) for path in args.summary],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
