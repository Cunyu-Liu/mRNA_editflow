from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/assemble_route2_strongest_prediction_predictions_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("assemble_route2_strongest_prediction_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_task_specific_winners_are_assembled_without_outcomes(tmp_path: Path) -> None:
    module = _load()
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    selection = tmp_path / "selection.json"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _jsonl(manifest, [
        {"canonical_record_id": "r1", "study_unit_id": "S", "pool_assignment": "DEVELOPMENT", "split": "TEST"},
        {"canonical_record_id": "r2", "study_unit_id": "S", "pool_assignment": "DEVELOPMENT", "split": "TEST"},
    ])
    _jsonl(canonical, [
        {"canonical_record_id": "r1", "pool_assignment": "DEVELOPMENT", "region": "5UTR", "endpoint_id": "E1"},
        {"canonical_record_id": "r2", "pool_assignment": "DEVELOPMENT", "region": "3UTR", "endpoint_id": "E2"},
    ])
    selection.write_text(json.dumps({
        "status": "DEVELOPMENT_VALIDATION_STRONGEST_BASELINES_SELECTED_BY_TASK",
        "evaluation_outcomes_accessed": False,
        "tasks": {
            "5UTR|E1": {"strongest_baseline_id": "first"},
            "3UTR|E2": {"strongest_baseline_id": "second"},
        },
    }), encoding="utf-8")
    _jsonl(first, [{"canonical_record_id": "r1", "predicted_direction_normalized_delta": 0.1}])
    _jsonl(second, [{"canonical_record_id": "r2", "predicted_direction_normalized_delta": 0.2}])
    rows, summary = module.assemble({
        "schema_version": "route_a_v3_route2_strongest_prediction_assembly_config.v1",
        "evaluation_outcomes_accessed": False,
        "requested_split": "TEST",
        "development_manifest_path": str(manifest),
        "canonical_paths": [str(canonical)],
        "strongest_selection_path": str(selection),
        "baseline_predictions": [
            {"baseline_id": "first", "prediction_path": str(first)},
            {"baseline_id": "second", "prediction_path": str(second)},
        ],
    })
    assert [(row["canonical_record_id"], row["baseline_id"]) for row in rows] == [("r1", "first"), ("r2", "second")]
    assert summary["evaluation_outcomes_accessed"] is False


def test_assembly_rejects_evaluation_or_wrong_split_predictions(tmp_path: Path) -> None:
    module = _load()
    manifest = tmp_path / "manifest.jsonl"
    _jsonl(manifest, [{
        "canonical_record_id": "e", "study_unit_id": "E", "pool_assignment": "EVALUATION", "split": "TEST"
    }])
    with pytest.raises(module.StrongestPredictionAssemblyError, match="non-Development"):
        module.selected_manifest_rows(manifest, "TEST")


def test_empty_zero_record_canonical_study_is_allowed(tmp_path: Path) -> None:
    module = _load()
    empty = tmp_path / "zero.jsonl"
    empty.write_text("", encoding="utf-8")
    populated = tmp_path / "nonzero.jsonl"
    _jsonl(populated, [{
        "canonical_record_id": "r", "pool_assignment": "DEVELOPMENT", "region": "5UTR", "endpoint_id": "E"
    }])
    assert module.load_tasks([empty, populated], {"r"}) == {"r": "5UTR|E"}
