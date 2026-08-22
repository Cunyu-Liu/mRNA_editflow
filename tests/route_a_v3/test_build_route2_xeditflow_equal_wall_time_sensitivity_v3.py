from __future__ import annotations

import json

import pytest

from core.route2_xeditflow_equal_wall_time_v3 import (
    EQUAL_WALL_TIME_SCOPE_V3,
    METHODS_V3,
)
from scripts.route_a_v3.build_route2_xeditflow_equal_wall_time_sensitivity_v3 import (
    MATCHED_COMPUTE_JSONL,
    SEARCH_CANDIDATE_JSONL,
    build_equal_wall_time_sensitivity_v3,
)


SOURCES = [f"source-{index:03d}" for index in range(891)]


def _write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _config(tmp_path):
    manifest = tmp_path / "sources.jsonl"
    _write_jsonl(manifest, [{"source_key": key} for key in SOURCES])
    methods = {}
    for method_index, method in enumerate(sorted(METHODS_V3)):
        closed = tmp_path / f"{method}.closed.json"
        value = 0.8 if method == "full_soft_value_smc" else 0.6
        _write_json(
            closed,
            {
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
            },
        )
        timing = tmp_path / f"{method}.jsonl"
        if method == "strongest_matched_baseline":
            rows = []
            for source in SOURCES:
                rows.extend(
                    [
                        {
                            "source_key": source,
                            "source_equal_wall_time_seconds": 1.5,
                            "source_equal_wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
                            "source_equal_wall_peak_vram_mb": 1200.0,
                            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                        },
                        {
                            "source_key": source,
                            "source_equal_wall_time_seconds": 0.0,
                            "source_equal_wall_time_scope": "COUNTED_ON_SOURCE_FIRST_ROW",
                            "source_equal_wall_peak_vram_mb": 0.0,
                            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                        },
                    ]
                )
            timing_format = SEARCH_CANDIDATE_JSONL
        else:
            rows = [
                {
                    "source_key": source,
                    "source_equal_wall_time_seconds": 1.0 + method_index / 10.0,
                    "source_equal_wall_time_scope": EQUAL_WALL_TIME_SCOPE_V3,
                    "source_equal_wall_peak_vram_mb": 1000.0 + method_index,
                    "source_cuda_device_name": "NVIDIA A100-SXM4-80GB",
                }
                for source in SOURCES
            ]
            timing_format = MATCHED_COMPUTE_JSONL
        _write_jsonl(timing, rows)
        methods[method] = {
            "timing_path": str(timing),
            "timing_format": timing_format,
            "closed_summary_path": str(closed),
        }
    return {
        "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_config.v1",
        "base_flow_training_seed": 20260904,
        "source_manifest_path": str(manifest),
        "methods": methods,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_builder_normalizes_flow_and_search_timing_into_one_a100_scope(tmp_path) -> None:
    result = build_equal_wall_time_sensitivity_v3(_config(tmp_path))
    assert result["status"] == "XEDITFLOW_V3_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
    assert set(result["methods"]) == METHODS_V3
    assert result["methods"]["strongest_matched_baseline"]["peak_vram_mb"] == 1200.0
    assert result["direction_diagnostics_not_a_separate_gate"]["full_ndcg_above_strongest"] is True
    assert "POSTHOC_DIAGNOSTIC_SCORING_EXCLUDED" in result["timing_scope_definition"]


def test_builder_rejects_uninstrumented_historical_strongest_payload(tmp_path) -> None:
    config = _config(tmp_path)
    timing_path = config["methods"]["strongest_matched_baseline"]["timing_path"]
    rows = [json.loads(line) for line in open(timing_path, encoding="utf-8")]
    del rows[0]["source_equal_wall_time_seconds"]
    _write_jsonl(tmp_path / "strongest_matched_baseline.jsonl", rows)
    with pytest.raises(Exception, match="raw source time"):
        build_equal_wall_time_sensitivity_v3(config)


def test_builder_rejects_non_a100_timing(tmp_path) -> None:
    config = _config(tmp_path)
    timing_path = config["methods"]["unguided_setflow"]["timing_path"]
    rows = [json.loads(line) for line in open(timing_path, encoding="utf-8")]
    rows[0]["source_cuda_device_name"] = "NVIDIA H100"
    _write_jsonl(tmp_path / "unguided_setflow.jsonl", rows)
    with pytest.raises(Exception, match="not from A100"):
        build_equal_wall_time_sensitivity_v3(config)
