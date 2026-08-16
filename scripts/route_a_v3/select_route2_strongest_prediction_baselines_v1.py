#!/usr/bin/env python3
"""Select the strongest Development prediction baseline separately by task."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


class BaselineSelectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineSelectionError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _bootstrap_index(payload: Mapping[str, Any], baseline_ids: set[str]) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
    result = {}
    for row in payload["paired_validation_bootstrap"]:
        _require(row["split"] == "VALIDATION", "baseline bootstrap comparison is not validation-only")
        left, right = str(row["left_baseline_id"]), str(row["right_baseline_id"])
        _require(left != right and {left, right} <= baseline_ids, "bootstrap baseline pair is invalid")
        ci = row["spearman_difference_ci_95"]
        _require(
            isinstance(ci, list) and len(ci) == 2
            and _finite(ci[0], "bootstrap lower bound") <= _finite(ci[1], "bootstrap upper bound"),
            "bootstrap confidence interval is invalid",
        )
        _require(int(row["defined_bootstrap_iterations"]) > 0, "bootstrap comparison has no defined iterations")
        key = (str(row["task"]), frozenset((left, right)))
        _require(key not in result, "bootstrap comparison is duplicated")
        result[key] = dict(row)
    return result


def complete_coverage_fallback(
    entries: list[Mapping[str, Any]], required_tasks: set[str]
) -> dict[str, Any]:
    _require(required_tasks, "fallback task set is empty")
    candidates = []
    for entry in entries:
        task_metrics = entry["evaluation"]["metrics"]["task_numeric"]
        if not required_tasks <= set(task_metrics):
            continue
        metrics = [task_metrics[task] for task in sorted(required_tasks)]
        spearman = [_optional_finite(row["spearman"], "fallback Spearman") for row in metrics]
        candidates.append({
            "baseline_id": str(entry["baseline_id"]),
            "baseline_family": str(entry["baseline_family"]),
            "parameter_count": int(entry["parameter_count"]),
            "task_count": len(required_tasks),
            "all_task_spearman_defined": all(value is not None for value in spearman),
            "task_macro_spearman": (
                None if any(value is None for value in spearman)
                else sum(float(value) for value in spearman) / len(spearman)
            ),
            "task_macro_mae": sum(_finite(row["mae"], "fallback MAE") for row in metrics) / len(metrics),
        })
    _require(candidates, "no baseline has complete coverage for unseen-endpoint fallback")
    finite = [row for row in candidates if row["all_task_spearman_defined"]]
    if finite:
        ranked = sorted(
            finite,
            key=lambda row: (
                -row["task_macro_spearman"], row["parameter_count"],
                row["task_macro_mae"], row["baseline_id"],
            ),
        )
        metric = "COMPLETE_COVERAGE_DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN"
    else:
        ranked = sorted(
            candidates,
            key=lambda row: (row["task_macro_mae"], row["parameter_count"], row["baseline_id"]),
        )
        metric = "COMPLETE_COVERAGE_DEVELOPMENT_VALIDATION_TASK_MACRO_MAE_ALL_SPEARMAN_UNDEFINED"
    return {
        "strongest_baseline_id": ranked[0]["baseline_id"],
        "selection_primary_metric": metric,
        "required_complete_task_count": len(required_tasks),
        "complete_coverage_candidate_count": len(candidates),
        "all_candidates_ranked": ranked,
    }


def select(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload["schema_version"] == "route_a_v3_route2_baseline_selection_input.v1", "unexpected selection schema")
    _require(payload["selection_pool"] == "DEVELOPMENT_VALIDATION", "selection is not Development validation")
    _require(payload["evaluation_outcomes_accessed"] is False, "baseline selection accessed Evaluation")
    entries = payload["baseline_evaluations"]
    _require(entries, "no baseline evaluations were provided")
    identifiers = [str(entry["baseline_id"]) for entry in entries]
    _require(len(identifiers) == len(set(identifiers)), "baseline evaluation is duplicated")
    bootstrap = _bootstrap_index(payload, set(identifiers))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        evaluation = entry["evaluation"]
        _require(evaluation["split"] == "VALIDATION", f"baseline is not validation-only: {entry['baseline_id']}")
        _require(evaluation.get("evaluation_release_state", "CLOSED") == "CLOSED", "Evaluation release state opened")
        parameter_count = int(entry["parameter_count"])
        _require(parameter_count >= 0, "baseline parameter count is negative")
        task_metrics = evaluation["metrics"]["task_numeric"]
        for task, metrics in task_metrics.items():
            by_task[str(task)].append({
                "baseline_id": str(entry["baseline_id"]),
                "baseline_family": str(entry["baseline_family"]),
                "parameter_count": parameter_count,
                "spearman": _optional_finite(metrics["spearman"], f"validation Spearman for {task}"),
                "mae": _finite(metrics["mae"], f"validation MAE for {task}"),
                "record_count": int(metrics["record_count"]),
            })
    _require(by_task, "no task-level validation metrics were provided")
    selected = {}
    for task, candidates in sorted(by_task.items()):
        finite_spearman = any(row["spearman"] is not None for row in candidates)
        # Higher finite Spearman is primary. Undefined constant-control correlations are retained
        # behind finite results. If all are undefined, lower MAE becomes the explicit fallback.
        if finite_spearman:
            ranked = sorted(
                candidates,
                key=lambda row: (
                    row["spearman"] is None,
                    0.0 if row["spearman"] is None else -row["spearman"],
                    row["parameter_count"], row["mae"], row["baseline_id"],
                ),
            )
            selection_metric = "DEVELOPMENT_VALIDATION_SPEARMAN"
            point_winner = ranked[0]
            uncertainty_equivalent = [point_winner]
            for candidate in ranked[1:]:
                if candidate["spearman"] is None:
                    continue
                key = (task, frozenset((point_winner["baseline_id"], candidate["baseline_id"])))
                _require(key in bootstrap, f"paired validation bootstrap is absent for {task}: {candidate['baseline_id']}")
                lower, upper = bootstrap[key]["spearman_difference_ci_95"]
                if float(lower) <= 0.0 <= float(upper):
                    uncertainty_equivalent.append(candidate)
            winner = min(
                uncertainty_equivalent,
                key=lambda row: (
                    row["parameter_count"],
                    -float(row["spearman"]),
                    row["mae"],
                    row["baseline_id"],
                ),
            )
        else:
            ranked = sorted(candidates, key=lambda row: (row["mae"], row["parameter_count"], row["baseline_id"]))
            selection_metric = "DEVELOPMENT_VALIDATION_MAE_ALL_SPEARMAN_UNDEFINED"
            winner = ranked[0]
            uncertainty_equivalent = [winner]
        selected[task] = {
            "strongest_baseline_id": winner["baseline_id"],
            "selection_primary_metric": selection_metric,
            "selected_spearman": winner["spearman"],
            "selected_mae": winner["mae"],
            "exact_spearman_tie_count": sum(row["spearman"] == winner["spearman"] for row in ranked),
            "bootstrap_uncertainty_equivalent_candidate_count": len(uncertainty_equivalent),
            "bootstrap_uncertainty_equivalent_baseline_ids": sorted(
                row["baseline_id"] for row in uncertainty_equivalent
            ),
            "finite_spearman_candidate_count": sum(row["spearman"] is not None for row in ranked),
            "candidate_count": len(ranked),
            "all_candidates_ranked": ranked,
        }
    tasks_by_region: dict[str, set[str]] = defaultdict(set)
    for task in by_task:
        region, separator, _endpoint = task.partition("|")
        _require(separator == "|" and region, f"task id does not encode region: {task}")
        tasks_by_region[region].add(task)
    region_fallbacks = {
        region: complete_coverage_fallback(entries, tasks)
        for region, tasks in sorted(tasks_by_region.items())
    }
    global_fallback = complete_coverage_fallback(entries, set(by_task))
    return {
        "schema_version": "route_a_v3_route2_strongest_prediction_baselines.v1",
        "status": "DEVELOPMENT_VALIDATION_STRONGEST_BASELINES_SELECTED_BY_TASK",
        "task_count": len(selected),
        "tasks": selected,
        "unseen_endpoint_fallbacks": {
            "policy": "EXACT_TASK_ELSE_COMPLETE_COVERAGE_REGION_ELSE_COMPLETE_COVERAGE_GLOBAL",
            "regions": region_fallbacks,
            "global": global_fallback,
            "selection_pool": "DEVELOPMENT_VALIDATION",
        },
        "evaluation_outcomes_accessed": False,
        "main_model_result_used_for_baseline_selection": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = select(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
