from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/route_a_v3/build_route2_aligned_validation_subset_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_route2_aligned_validation_subset_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_aligns_full_and_subset_prediction_inputs(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.jsonl"
    _write(manifest, [
        {"canonical_record_id": "a", "study_unit_id": "S1", "split": "VALIDATION", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "b", "study_unit_id": "S2", "split": "VALIDATION", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "c", "study_unit_id": "S1", "split": "TRAIN", "pool_assignment": "DEVELOPMENT"},
    ])
    full = tmp_path / "full.jsonl"
    subset = tmp_path / "subset.jsonl"
    _write(full, [
        {"canonical_record_id": "a", "predicted_direction_normalized_delta": 1.0},
        {"canonical_record_id": "b", "predicted_direction_normalized_delta": 2.0},
    ])
    _write(subset, [{"canonical_record_id": "a", "predicted_direction_normalized_delta": 3.0}])
    rows, predictions, summary = module.build({
        "schema_version": "route_a_v3_route2_aligned_validation_subset_config.v1",
        "development_manifest_path": str(manifest),
        "included_study_unit_ids": ["S1"],
        "prediction_inputs": [
            {"prediction_id": "main", "path": str(full)},
            {"prediction_id": "ablation", "path": str(subset)},
        ],
        "evaluation_outcomes_accessed": False,
    })
    assert [row["canonical_record_id"] for row in rows] == ["a"]
    assert {key: [row["canonical_record_id"] for row in value] for key, value in predictions.items()} == {
        "main": ["a"], "ablation": ["a"]
    }
    assert summary["record_count"] == 1
    assert summary["evaluation_outcomes_accessed"] is False


def test_build_rejects_prediction_outside_validation(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(manifest, [
        {"canonical_record_id": "a", "study_unit_id": "S1", "split": "VALIDATION", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "c", "study_unit_id": "S1", "split": "TEST", "pool_assignment": "DEVELOPMENT"},
    ])
    _write(predictions, [
        {"canonical_record_id": "a", "predicted_direction_normalized_delta": 1.0},
        {"canonical_record_id": "c", "predicted_direction_normalized_delta": 2.0},
    ])
    with pytest.raises(module.AlignedSubsetError, match="outside Development validation"):
        module.build({
            "schema_version": "route_a_v3_route2_aligned_validation_subset_config.v1",
            "development_manifest_path": str(manifest),
            "included_study_unit_ids": ["S1"],
            "prediction_inputs": [{"prediction_id": "bad", "path": str(predictions)}],
            "evaluation_outcomes_accessed": False,
        })


def test_build_filters_a_mixed_study_by_canonical_region(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(manifest, [
        {"canonical_record_id": "three", "study_unit_id": "MIXED", "split": "VALIDATION", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "five", "study_unit_id": "MIXED", "split": "VALIDATION", "pool_assignment": "DEVELOPMENT"},
    ])
    _write(canonical, [
        {"canonical_record_id": "three", "region": "3UTR"},
        {"canonical_record_id": "five", "region": "5UTR"},
    ])
    _write(predictions, [{"canonical_record_id": "five", "predicted_direction_normalized_delta": 1.0}])
    rows, filtered, summary = module.build({
        "schema_version": "route_a_v3_route2_aligned_validation_subset_config.v1",
        "development_manifest_path": str(manifest),
        "included_study_unit_ids": ["MIXED"],
        "included_regions": ["5′UTR"],
        "canonical_paths": [str(canonical)],
        "prediction_inputs": [{"prediction_id": "regional", "path": str(predictions)}],
        "evaluation_outcomes_accessed": False,
    })
    assert [row["canonical_record_id"] for row in rows] == ["five"]
    assert [row["canonical_record_id"] for row in filtered["regional"]] == ["five"]
    assert summary["included_regions"] == ["5UTR"]
