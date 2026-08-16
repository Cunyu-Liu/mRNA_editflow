#!/usr/bin/env python3
"""Assemble task-specific strongest Development baseline predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class StrongestPredictionAssemblyError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StrongestPredictionAssemblyError(message)


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows or allow_empty, f"input is empty: {path}")
    return rows


def selected_manifest_rows(path: Path, requested_split: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    _require(all(row["pool_assignment"] == "DEVELOPMENT" for row in rows), "non-Development row entered strongest-baseline assembly")
    if requested_split.startswith("LOSO::"):
        holdout = requested_split.removeprefix("LOSO::")
        _require(holdout, "LOSO holdout is empty")
        selected = {
            str(row["canonical_record_id"]): row
            for row in rows if str(row["study_unit_id"]) == holdout
        }
    else:
        _require(requested_split in {"VALIDATION", "TEST"}, f"unsupported split: {requested_split}")
        selected = {
            str(row["canonical_record_id"]): row
            for row in rows if row["split"] == requested_split
        }
    _require(selected, f"requested Development split is empty: {requested_split}")
    return selected


def load_tasks(canonical_paths: list[Path], selected_ids: set[str]) -> dict[str, str]:
    tasks: dict[str, str] = {}
    for path in canonical_paths:
        for row in _read_jsonl(path, allow_empty=True):
            record_id = str(row["canonical_record_id"])
            if record_id not in selected_ids:
                continue
            _require(row["pool_assignment"] == "DEVELOPMENT", "Evaluation row entered strongest-baseline task mapping")
            _require(record_id not in tasks, f"canonical record is duplicated: {record_id}")
            tasks[record_id] = f"{row['region']}|{row['endpoint_id']}"
    _require(set(tasks) == selected_ids, "canonical inputs do not exactly cover the requested Development records")
    return tasks


def assemble(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(config["schema_version"] == "route_a_v3_route2_strongest_prediction_assembly_config.v1", "unexpected config schema")
    _require(config["evaluation_outcomes_accessed"] is False, "strongest-baseline assembly accessed Evaluation")
    requested_split = str(config["requested_split"])
    manifest = selected_manifest_rows(Path(config["development_manifest_path"]), requested_split)
    tasks = load_tasks([Path(path) for path in config["canonical_paths"]], set(manifest))
    selection = json.loads(Path(config["strongest_selection_path"]).read_text(encoding="utf-8"))
    _require(selection["status"] == "DEVELOPMENT_VALIDATION_STRONGEST_BASELINES_SELECTED_BY_TASK", "strongest-baseline selection is not frozen")
    _require(selection["evaluation_outcomes_accessed"] is False, "strongest-baseline selection accessed Evaluation")
    winner_by_task = {
        str(task): str(payload["strongest_baseline_id"])
        for task, payload in selection["tasks"].items()
    }
    _require(set(tasks.values()) <= set(winner_by_task), "strongest baseline is absent for a requested task")

    prediction_paths = {
        str(spec["baseline_id"]): Path(spec["prediction_path"])
        for spec in config["baseline_predictions"]
    }
    _require(len(prediction_paths) == len(config["baseline_predictions"]), "baseline prediction input is duplicated")
    required_baselines = {winner_by_task[task] for task in tasks.values()}
    _require(required_baselines <= set(prediction_paths), "winning baseline prediction file is absent")
    predictions: dict[str, dict[str, float]] = {}
    for baseline_id in sorted(required_baselines):
        values: dict[str, float] = {}
        for row in _read_jsonl(prediction_paths[baseline_id]):
            record_id = str(row["canonical_record_id"])
            _require(record_id in manifest, f"baseline prediction is outside requested split: {baseline_id}/{record_id}")
            _require(record_id not in values, f"baseline prediction is duplicated: {baseline_id}/{record_id}")
            value = row["predicted_direction_normalized_delta"]
            _require(isinstance(value, (int, float)) and not isinstance(value, bool), "baseline prediction is not numeric")
            values[record_id] = float(value)
        predictions[baseline_id] = values

    output = []
    for record_id in sorted(manifest):
        task = tasks[record_id]
        baseline_id = winner_by_task[task]
        _require(record_id in predictions[baseline_id], f"winning baseline does not cover record: {baseline_id}/{record_id}")
        output.append({
            "canonical_record_id": record_id,
            "task": task,
            "baseline_id": baseline_id,
            "predicted_direction_normalized_delta": predictions[baseline_id][record_id],
        })
    summary = {
        "schema_version": "route_a_v3_route2_strongest_prediction_assembly.v1",
        "status": "TASK_SPECIFIC_STRONGEST_BASELINE_PREDICTIONS_ASSEMBLED",
        "requested_split": requested_split,
        "record_count": len(output),
        "task_count": len(set(tasks.values())),
        "baseline_ids_used": sorted({row["baseline_id"] for row in output}),
        "selection_source": "FROZEN_DEVELOPMENT_VALIDATION",
        "evaluation_outcomes_accessed": False,
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
