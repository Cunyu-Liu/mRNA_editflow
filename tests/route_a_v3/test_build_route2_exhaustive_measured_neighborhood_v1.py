import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "build_route2_exhaustive_measured_neighborhood_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exhaustive_measured_neighborhood_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _source(key: str) -> dict:
    return {"source_key": key, "evaluation_outcomes_included": False}


def _measured(key: str, candidate: str, *, pool: str = "DEVELOPMENT") -> dict:
    return {
        "source_key": key,
        "candidate_sequence": candidate,
        "pool_assignment": pool,
        "split": "VALIDATION",
        "measured_direction_normalized_delta": 0.1,
    }


def test_build_filters_full_neighborhood_to_exact_source_subset() -> None:
    module = _load_module()
    rows, summary = module.build(
        [_source("s1"), _source("s2")],
        [_measured("s1", "AAAA"), _measured("s2", "CCCC"), _measured("s3", "GGGG")],
    )
    assert {row["source_key"] for row in rows} == {"s1", "s2"}
    assert summary["selected_source_count"] == 2
    assert summary["selected_measured_row_count"] == 2
    assert summary["excluded_measured_row_count"] == 1
    assert summary["evaluation_outcomes_accessed"] is False


def test_build_rejects_missing_measured_source() -> None:
    module = _load_module()
    with pytest.raises(module.ExhaustiveMeasuredNeighborhoodError):
        module.build([_source("s1"), _source("s2")], [_measured("s1", "AAAA")])


def test_build_rejects_non_development_selected_outcome() -> None:
    module = _load_module()
    with pytest.raises(module.ExhaustiveMeasuredNeighborhoodError):
        module.build([_source("s1")], [_measured("s1", "AAAA", pool="EVALUATION")])
