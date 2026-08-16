from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_prediction_baseline_bootstrap_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("baseline_bootstrap_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    canonical_rows = []
    manifest_rows = []
    left_rows = []
    right_rows = []
    for group in range(3):
        for member in range(3):
            record_id = f"R{group}{member}"
            observed = float(group * 3 + member)
            manifest_rows.append({
                "canonical_record_id": record_id, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION",
            })
            canonical_rows.append({
                "canonical_record_id": record_id, "pool_assignment": "DEVELOPMENT",
                "study_unit_id": "S", "source_id": f"SRC{group}", "biological_context_id": "C",
                "endpoint_id": "E", "region": "3UTR", "direction_normalized_delta": observed,
            })
            left_rows.append({"canonical_record_id": record_id, "predicted_direction_normalized_delta": observed})
            right_rows.append({"canonical_record_id": record_id, "predicted_direction_normalized_delta": -observed})
    for path, rows in ((manifest, manifest_rows), (canonical, canonical_rows), (left, left_rows), (right, right_rows)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest, canonical, left, right


def test_builder_uses_complete_development_tasks_and_source_group_bootstrap(tmp_path: Path) -> None:
    module = _load()
    manifest, canonical, left, right = _write_fixture(tmp_path)
    result = module.build({
        "schema_version": "route_a_v3_route2_prediction_baseline_bootstrap_config.v1",
        "development_manifest_path": str(manifest),
        "canonical_paths": [str(canonical)],
        "baselines": [
            {"baseline_id": "left", "baseline_family": "TEST", "parameter_count": 10, "validation_predictions_path": str(left)},
            {"baseline_id": "right", "baseline_family": "TEST", "parameter_count": 5, "validation_predictions_path": str(right)},
        ],
        "comparison_policy": "POINT_LEADER_VS_ALL_FINITE",
        "bootstrap_iterations": 1000,
        "seed": 7,
        "evaluation_outcomes_accessed": False,
    })
    assert len(result["paired_validation_bootstrap"]) == 1
    comparison = result["paired_validation_bootstrap"][0]
    assert comparison["task"] == "3UTR|E"
    assert comparison["source_group_count"] == 3
    assert comparison["defined_bootstrap_iterations"] > 0
    assert comparison["spearman_difference_ci_95"][0] > 0
    assert result["evaluation_outcomes_accessed"] is False


def test_incomplete_task_prediction_coverage_is_rejected(tmp_path: Path) -> None:
    module = _load()
    manifest, canonical, left, right = _write_fixture(tmp_path)
    rows = right.read_text(encoding="utf-8").splitlines()
    right.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(module.BootstrapInputError, match="incompletely covers"):
        module.build({
            "schema_version": "route_a_v3_route2_prediction_baseline_bootstrap_config.v1",
            "development_manifest_path": str(manifest), "canonical_paths": [str(canonical)],
            "baselines": [{"baseline_id": "right", "baseline_family": "TEST", "parameter_count": 5, "validation_predictions_path": str(right)}],
            "comparison_policy": "POINT_LEADER_VS_ALL_FINITE",
            "bootstrap_iterations": 1000, "seed": 7, "evaluation_outcomes_accessed": False,
        })


def test_point_leader_policy_avoids_unused_nonleader_pair(tmp_path: Path) -> None:
    module = _load()
    manifest, canonical, left, right = _write_fixture(tmp_path)
    middle = tmp_path / "middle.jsonl"
    rows = [json.loads(line) for line in left.read_text(encoding="utf-8").splitlines()]
    first = rows[0]["predicted_direction_normalized_delta"]
    rows[0]["predicted_direction_normalized_delta"] = rows[1]["predicted_direction_normalized_delta"]
    rows[1]["predicted_direction_normalized_delta"] = first
    middle.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = module.build({
        "schema_version": "route_a_v3_route2_prediction_baseline_bootstrap_config.v1",
        "development_manifest_path": str(manifest),
        "canonical_paths": [str(canonical)],
        "baselines": [
            {"baseline_id": "leader", "baseline_family": "TEST", "parameter_count": 10, "validation_predictions_path": str(left)},
            {"baseline_id": "middle", "baseline_family": "TEST", "parameter_count": 5, "validation_predictions_path": str(middle)},
            {"baseline_id": "last", "baseline_family": "TEST", "parameter_count": 1, "validation_predictions_path": str(right)},
        ],
        "comparison_policy": "POINT_LEADER_VS_ALL_FINITE",
        "bootstrap_iterations": 1000,
        "seed": 7,
        "evaluation_outcomes_accessed": False,
    })
    pairs = {
        frozenset((row["left_baseline_id"], row["right_baseline_id"]))
        for row in result["paired_validation_bootstrap"]
    }
    assert pairs == {
        frozenset(("leader", "middle")),
        frozenset(("leader", "last")),
    }
