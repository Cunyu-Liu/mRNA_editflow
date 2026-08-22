from __future__ import annotations

import json

from scripts.route_a_v3.evaluate_route2_xeditflow_open_generation_v3 import (
    evaluate_open_generation_v3,
)


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_open_generation_reports_recovery_without_defining_closed_ndcg(tmp_path) -> None:
    source_path = tmp_path / "sources.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    measured_path = tmp_path / "measured.jsonl"
    _write_jsonl(source_path, [{
        "source_key": "s",
        "source_sequence": "AC",
        "edit_budget": 1,
        "candidate_budget": 2,
    }])
    _write_jsonl(candidate_path, [
        {"method_id": "guided", "source_key": "s", "candidate_sequence": "AA", "terminal_cause": "BUDGET_EXHAUSTED", "generation_score": 1.0},
        {"method_id": "guided", "source_key": "s", "candidate_sequence": "AG", "terminal_cause": "BUDGET_EXHAUSTED", "generation_score": 0.5},
    ])
    _write_jsonl(measured_path, [
        {"source_key": "s", "candidate_sequence": "AA", "measured_direction_normalized_delta": 1.0, "pool_assignment": "DEVELOPMENT"},
        {"source_key": "s", "candidate_sequence": "AU", "measured_direction_normalized_delta": 0.0, "pool_assignment": "DEVELOPMENT"},
    ])
    result = evaluate_open_generation_v3({
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_config.v1",
        "pool_assignment": "DEVELOPMENT",
        "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
        "source_eligibility_manifest": str(source_path),
        "candidate_path": str(candidate_path),
        "measured_neighborhood_path": str(measured_path),
        "measured_top_k": 10,
    })
    assert result["source_macro_candidate_recovery"] == 0.5
    assert result["closed_ndcg_defined_count"] == 0
    assert result["closed_ndcg_is_not_defined_on_open_support"] is True
    assert result["unknown_generated_candidates_are_zero_gain"] is False
