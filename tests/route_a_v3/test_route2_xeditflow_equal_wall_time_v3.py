from __future__ import annotations

import copy

import pytest

from core.route2_xeditflow_equal_wall_time_v3 import (
    EQUAL_WALL_TIME_SCOPE_V3,
    METHODS_V3,
    equal_wall_time_sensitivity_v3,
)


SOURCES = [f"source-{index:03d}" for index in range(891)]


def _inputs():
    times = {}
    closed = {}
    for method_index, method in enumerate(sorted(METHODS_V3)):
        times[method] = [
            {
                "source_key": source,
                "source_wall_time_seconds": 1.0 + method_index / 10.0,
                "wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
                "accelerator_name": "NVIDIA A100-SXM4-80GB",
                "peak_vram_mb": 1000.0 + method_index,
            }
            for source in SOURCES
        ]
        value = 0.8 if method == "full_soft_value_smc" else 0.6
        closed[method] = {
            "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
            "undefined_sources_are_not_filled_with_zero": True,
            "per_source": {
                source: {
                    "status": "DEFINED",
                    "ndcg": value,
                    "normalized_regret": 1.0 - value,
                    "top_1_recall": value,
                }
                for source in SOURCES
            },
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    return times, closed


def test_equal_wall_uses_one_common_completed_source_prefix() -> None:
    result = equal_wall_time_sensitivity_v3(
        *_inputs(), source_order=SOURCES, base_flow_training_seed=20260904
    )
    assert result["status"] == "XEDITFLOW_V3_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
    assert 2 <= result["common_source_prefix_count"] < 891
    assert len(result["common_source_keys"]) == result["common_source_prefix_count"]
    assert all(
        row["source_count"] == result["common_source_prefix_count"]
        for row in result["methods"].values()
    )
    assert all("A100" in row["accelerator_name"] for row in result["methods"].values())
    assert all(row["peak_vram_mb"] >= 1000.0 for row in result["methods"].values())
    assert result["direction_diagnostics_not_a_separate_gate"] == {
        "full_ndcg_above_unguided": True,
        "full_ndcg_above_strongest": True,
        "full_regret_below_unguided": True,
        "full_regret_below_strongest": True,
    }


def test_equal_wall_rejects_mismatched_source_order_or_missing_timing() -> None:
    times, closed = _inputs()
    times["unguided_setflow"][0]["source_key"] = "wrong"
    with pytest.raises(Exception, match="source order differs"):
        equal_wall_time_sensitivity_v3(
            times, closed, source_order=SOURCES, base_flow_training_seed=20260904
        )
    times, closed = _inputs()
    times["strongest_matched_baseline"][0]["source_wall_time_seconds"] = None
    with pytest.raises(Exception, match="not numeric"):
        equal_wall_time_sensitivity_v3(
            times, closed, source_order=SOURCES, base_flow_training_seed=20260904
        )
    times, closed = _inputs()
    times["strongest_matched_baseline"][0]["accelerator_name"] = "NVIDIA H100"
    with pytest.raises(Exception, match="not A100"):
        equal_wall_time_sensitivity_v3(
            times, closed, source_order=SOURCES, base_flow_training_seed=20260904
        )
    times, closed = _inputs()
    for row in times["strongest_matched_baseline"]:
        row["accelerator_name"] = "NVIDIA A100-PCIE-40GB"
    with pytest.raises(Exception, match="accelerator model differs"):
        equal_wall_time_sensitivity_v3(
            times, closed, source_order=SOURCES, base_flow_training_seed=20260904
        )


def test_equal_wall_preserves_undefined_closed_sources_without_zero_fill() -> None:
    times, closed = _inputs()
    for payload in closed.values():
        payload["per_source"][SOURCES[0]] = {
            "status": "UNDEFINED_TOO_FEW_CANDIDATES",
            "ndcg": None,
            "normalized_regret": None,
            "top_1_recall": None,
        }
    result = equal_wall_time_sensitivity_v3(
        times, closed, source_order=SOURCES, base_flow_training_seed=20260905
    )
    assert all(
        row["defined_source_count"] == row["source_count"] - 1
        for row in result["methods"].values()
    )
    invalid = copy.deepcopy(closed)
    invalid["full_soft_value_smc"]["per_source"][SOURCES[0]]["ndcg"] = 0.0
    with pytest.raises(Exception, match="undefined source carries"):
        equal_wall_time_sensitivity_v3(
            times, invalid, source_order=SOURCES, base_flow_training_seed=20260905
        )
