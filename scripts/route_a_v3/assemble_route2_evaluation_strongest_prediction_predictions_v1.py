#!/usr/bin/env python3
"""Assemble zero-shot Evaluation predictions using only frozen Development selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class EvaluationStrongestAssemblyError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationStrongestAssemblyError(message)


def _jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows or allow_empty, f"input is empty: {path}")
    return rows


def evaluation_manifest(path: Path) -> set[str]:
    result: set[str] = set()
    for row in _jsonl(path):
        _require(row["pool_assignment"] == "EVALUATION", "Development row entered Evaluation assembly")
        _require(row["split"] == "EVALUATION_ZERO_SHOT", "Evaluation row is not zero-shot")
        record_id = str(row["canonical_record_id"])
        _require(record_id not in result, "Evaluation manifest record is duplicated")
        result.add(record_id)
    return result


def evaluation_tasks(canonical_paths: list[Path], selected_ids: set[str]) -> dict[str, str]:
    tasks: dict[str, str] = {}
    for path in canonical_paths:
        for row in _jsonl(path, allow_empty=True):
            record_id = str(row["canonical_record_id"])
            if record_id not in selected_ids:
                continue
            _require(row["pool_assignment"] == "EVALUATION", "Development canonical row entered Evaluation assembly")
            _require(record_id not in tasks, f"Evaluation canonical record is duplicated: {record_id}")
            tasks[record_id] = f"{row['region']}|{row['endpoint_id']}"
    _require(set(tasks) == selected_ids, "Evaluation canonical inputs do not exactly cover manifest")
    return tasks


def baseline_for_task(selection: Mapping[str, Any], task: str) -> tuple[str, str]:
    if task in selection["tasks"]:
        return str(selection["tasks"][task]["strongest_baseline_id"]), "EXACT_TASK"
    region, separator, _endpoint = task.partition("|")
    _require(separator == "|" and region, f"Evaluation task id does not encode region: {task}")
    fallbacks = selection["unseen_endpoint_fallbacks"]
    _require(
        fallbacks["policy"] == "EXACT_TASK_ELSE_COMPLETE_COVERAGE_REGION_ELSE_COMPLETE_COVERAGE_GLOBAL",
        "unseen-endpoint fallback policy differs",
    )
    if region in fallbacks["regions"]:
        return str(fallbacks["regions"][region]["strongest_baseline_id"]), "COMPLETE_COVERAGE_REGION"
    return str(fallbacks["global"]["strongest_baseline_id"]), "COMPLETE_COVERAGE_GLOBAL"


def validate_prediction_summary(summary: Mapping[str, Any], baseline_id: str) -> None:
    _require(summary["status"] == "EVALUATION_ZERO_SHOT_PREDICTIONS_GENERATED", "prediction file is not frozen zero-shot output")
    _require(str(summary.get("baseline_id", summary.get("model_id", ""))) == baseline_id, "prediction summary baseline differs")
    _require(summary["evaluation_outcome_metrics_computed"] is False, "prediction summary computed Evaluation metrics")
    _require(summary["evaluation_outcomes_used_for_training_hpo_or_selection"] == 0, "Evaluation selected the predictor")
    _require(summary["cpu_fallback_used"] is False, "prediction summary used CPU fallback")
    _require(str(summary["device"]).startswith("cuda:"), "prediction summary lacks CUDA execution")
    _require(isinstance(summary.get("cuda_device_uuid"), str) and bool(summary["cuda_device_uuid"]), "prediction summary lacks GPU UUID")


def assemble(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        config["schema_version"] == "route_a_v3_route2_evaluation_strongest_prediction_assembly_config.v1",
        "unexpected config schema",
    )
    _require(config["evaluation_release_state"] == "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN", "Evaluation remains closed")
    _require(config["evaluation_outcomes_used_for_training_hpo_or_selection"] == 0, "Evaluation selected a baseline")
    selected_ids = evaluation_manifest(Path(config["evaluation_manifest_path"]))
    tasks = evaluation_tasks([Path(path) for path in config["evaluation_canonical_paths"]], selected_ids)
    selection = json.loads(Path(config["strongest_selection_path"]).read_text(encoding="utf-8"))
    _require(selection["status"] == "DEVELOPMENT_VALIDATION_STRONGEST_BASELINES_SELECTED_BY_TASK", "strongest selection is not frozen")
    _require(selection["evaluation_outcomes_accessed"] is False, "strongest selection accessed Evaluation")
    winner_by_task = {task: baseline_for_task(selection, task) for task in set(tasks.values())}

    specs = {str(spec["baseline_id"]): spec for spec in config["baseline_predictions"]}
    _require(len(specs) == len(config["baseline_predictions"]), "baseline prediction input is duplicated")
    required = {value[0] for value in winner_by_task.values()}
    _require(required <= set(specs), "frozen zero-shot prediction file is absent for a required baseline")
    predictions: dict[str, dict[str, float]] = {}
    for baseline_id in sorted(required):
        spec = specs[baseline_id]
        summary = json.loads(Path(spec["prediction_summary_path"]).read_text(encoding="utf-8"))
        validate_prediction_summary(summary, baseline_id)
        values: dict[str, float] = {}
        for row in _jsonl(Path(spec["prediction_path"])):
            record_id = str(row["canonical_record_id"])
            _require(record_id in selected_ids, f"prediction is outside Evaluation manifest: {baseline_id}/{record_id}")
            _require(record_id not in values, f"Evaluation prediction is duplicated: {baseline_id}/{record_id}")
            value = row["predicted_direction_normalized_delta"]
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                "Evaluation prediction is not finite",
            )
            values[record_id] = float(value)
        _require(set(values) == selected_ids, f"frozen baseline does not cover Evaluation manifest: {baseline_id}")
        predictions[baseline_id] = values

    output = []
    for record_id in sorted(selected_ids):
        task = tasks[record_id]
        baseline_id, resolution = winner_by_task[task]
        output.append({
            "canonical_record_id": record_id,
            "task": task,
            "baseline_id": baseline_id,
            "selection_resolution": resolution,
            "predicted_direction_normalized_delta": predictions[baseline_id][record_id],
        })
    summary = {
        "schema_version": "route_a_v3_route2_evaluation_strongest_prediction_assembly.v1",
        "status": "EVALUATION_ZERO_SHOT_STRONGEST_BASELINE_PREDICTIONS_ASSEMBLED",
        "record_count": len(output),
        "task_count": len(set(tasks.values())),
        "baseline_ids_used": sorted(required),
        "selection_resolution_counts": {
            resolution: sum(row["selection_resolution"] == resolution for row in output)
            for resolution in sorted({row["selection_resolution"] for row in output})
        },
        "selection_source": "FROZEN_DEVELOPMENT_VALIDATION",
        "evaluation_outcome_metrics_computed": False,
        "evaluation_outcomes_used_for_training_hpo_or_selection": 0,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    _require(not summary_path.exists(), f"summary output already exists: {summary_path}")
    rows, summary = assemble(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
