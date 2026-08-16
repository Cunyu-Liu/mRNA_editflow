#!/usr/bin/env python3
"""Build Development-validation task metrics and paired source-group bootstrap comparisons."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr


class BootstrapInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapInputError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows, f"input is empty: {path}")
    return rows


def load_validation_manifest(path: Path) -> set[str]:
    selected = set()
    for row in _read_jsonl(path):
        _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation record entered baseline bootstrap manifest")
        if row["split"] == "VALIDATION":
            record_id = str(row["canonical_record_id"])
            _require(record_id not in selected, "validation manifest id is duplicated")
            selected.add(record_id)
    _require(selected, "Development validation split is empty")
    return selected


def load_observations(canonical_paths: list[Path], selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in canonical_paths:
        for row in _read_jsonl(path):
            record_id = str(row["canonical_record_id"])
            if record_id not in selected_ids:
                continue
            _require(record_id not in result, f"canonical record is duplicated: {record_id}")
            _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation outcome entered baseline bootstrap")
            result[record_id] = {
                "record_id": record_id,
                "task": f"{row['region']}|{row['endpoint_id']}",
                "source_group": "::".join((
                    str(row["study_unit_id"]), str(row["source_id"]),
                    str(row["biological_context_id"]), str(row["endpoint_id"]),
                )),
                "observed": _finite(row["direction_normalized_delta"], "observed delta"),
            }
    _require(set(result) == selected_ids, "canonical observations do not exactly cover validation manifest")
    return result


def load_predictions(path: Path, validation_ids: set[str]) -> dict[str, float]:
    result = {}
    for row in _read_jsonl(path):
        record_id = str(row["canonical_record_id"])
        _require(record_id in validation_ids, "prediction is outside Development validation")
        _require(record_id not in result, f"prediction is duplicated: {record_id}")
        result[record_id] = _finite(row["predicted_direction_normalized_delta"], "predicted delta")
    return result


def _spearman(rows: list[dict[str, Any]], predictions: Mapping[str, float]) -> float | None:
    observed = np.asarray([row["observed"] for row in rows], dtype=float)
    predicted = np.asarray([predictions[row["record_id"]] for row in rows], dtype=float)
    if len(rows) < 3 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return None
    value = float(spearmanr(observed, predicted).statistic)
    return value if math.isfinite(value) else None


def _task_metrics(rows: list[dict[str, Any]], predictions: Mapping[str, float]) -> dict[str, Any]:
    observed = np.asarray([row["observed"] for row in rows], dtype=float)
    predicted = np.asarray([predictions[row["record_id"]] for row in rows], dtype=float)
    return {
        "record_count": len(rows),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "spearman": _spearman(rows, predictions),
    }


def paired_source_group_bootstrap(
    rows: list[dict[str, Any]],
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    keys = sorted(groups)
    _require(len(keys) >= 2, "paired source-group bootstrap requires at least two groups")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indices = rng.integers(0, len(keys), size=len(keys))
        sampled = [row for index in indices for row in groups[keys[int(index)]]]
        left_value, right_value = _spearman(sampled, left), _spearman(sampled, right)
        if left_value is not None and right_value is not None:
            differences.append(left_value - right_value)
    _require(differences, "paired source-group bootstrap produced no defined Spearman iterations")
    point_left, point_right = _spearman(rows, left), _spearman(rows, right)
    _require(point_left is not None and point_right is not None, "paired point Spearman is undefined")
    values = np.asarray(differences, dtype=float)
    return {
        "analysis_unit": "SOURCE_GROUP",
        "source_group_count": len(keys),
        "bootstrap_iterations": iterations,
        "defined_bootstrap_iterations": len(differences),
        "point_spearman_difference": point_left - point_right,
        "spearman_difference_ci_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def build(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config["schema_version"] == "route_a_v3_route2_prediction_baseline_bootstrap_config.v1", "unexpected config schema")
    _require(config["evaluation_outcomes_accessed"] is False, "baseline bootstrap accessed Evaluation")
    iterations = int(config["bootstrap_iterations"])
    _require(iterations >= 1000, "baseline bootstrap budget is below 1000 iterations")
    validation_ids = load_validation_manifest(Path(config["development_manifest_path"]))
    observations = load_observations([Path(path) for path in config["canonical_paths"]], validation_ids)
    task_ids: dict[str, set[str]] = defaultdict(set)
    for record_id, row in observations.items():
        task_ids[row["task"]].add(record_id)

    entries = []
    predictions_by_baseline = {}
    tasks_by_baseline = {}
    identifiers = [str(spec["baseline_id"]) for spec in config["baselines"]]
    _require(len(identifiers) == len(set(identifiers)) and identifiers, "baseline inventory is empty or duplicated")
    for spec in config["baselines"]:
        baseline_id = str(spec["baseline_id"])
        predictions = load_predictions(Path(spec["validation_predictions_path"]), validation_ids)
        covered_tasks = sorted({observations[record_id]["task"] for record_id in predictions})
        _require(covered_tasks, f"baseline covers no validation task: {baseline_id}")
        for task in covered_tasks:
            predicted_ids = {record_id for record_id in predictions if observations[record_id]["task"] == task}
            _require(predicted_ids == task_ids[task], f"baseline incompletely covers validation task: {baseline_id}/{task}")
        task_metrics = {
            task: _task_metrics([observations[record_id] for record_id in sorted(task_ids[task])], predictions)
            for task in covered_tasks
        }
        entries.append({
            "baseline_id": baseline_id,
            "baseline_family": str(spec["baseline_family"]),
            "parameter_count": int(spec["parameter_count"]),
            "evaluation": {
                "split": "VALIDATION",
                "evaluation_release_state": "CLOSED",
                "metrics": {"task_numeric": task_metrics},
            },
        })
        predictions_by_baseline[baseline_id] = predictions
        tasks_by_baseline[baseline_id] = set(covered_tasks)

    comparisons = []
    seed = int(config["seed"])
    for task in sorted(task_ids):
        baselines = sorted(baseline_id for baseline_id in identifiers if task in tasks_by_baseline[baseline_id])
        rows = [observations[record_id] for record_id in sorted(task_ids[task])]
        finite = [baseline_id for baseline_id in baselines if _spearman(rows, predictions_by_baseline[baseline_id]) is not None]
        for pair_index, (left_id, right_id) in enumerate(itertools.combinations(finite, 2)):
            result = paired_source_group_bootstrap(
                rows, predictions_by_baseline[left_id], predictions_by_baseline[right_id],
                iterations=iterations, seed=seed + pair_index,
            )
            comparisons.append({
                "task": task,
                "split": "VALIDATION",
                "left_baseline_id": left_id,
                "right_baseline_id": right_id,
                **result,
            })
    return {
        "schema_version": "route_a_v3_route2_baseline_selection_input.v1",
        "selection_pool": "DEVELOPMENT_VALIDATION",
        "evaluation_outcomes_accessed": False,
        "baseline_evaluations": entries,
        "paired_validation_bootstrap": comparisons,
        "bootstrap_analysis_unit": "SOURCE_GROUP",
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = build(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
