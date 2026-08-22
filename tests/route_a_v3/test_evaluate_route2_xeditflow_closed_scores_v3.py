from __future__ import annotations

from scripts.route_a_v3.evaluate_route2_xeditflow_closed_scores_v3 import (
    XEditFlowClosedScoresV3Error,
    evaluate_closed_method_scores_v3,
)
import pytest


def test_frozen_scores_use_common_measured_candidates_and_preserve_undefined() -> None:
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v1",
        "pool_assignment": "DEVELOPMENT", "split": "VALIDATION",
        "analysis_unit": "SOURCE", "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
        "method_id": "simple_rate_guidance", "base_flow_training_seed": 20260904,
    }
    measured = [
        {"source_key": "s", "candidate_sequence": "AA", "measured_direction_normalized_delta": 0.0, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION"},
        {"source_key": "s", "candidate_sequence": "AC", "measured_direction_normalized_delta": 1.0, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION"},
        {"source_key": "u", "candidate_sequence": "GG", "measured_direction_normalized_delta": 2.0, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION"},
    ]
    scores = [
        {"source_key": "s", "candidate_sequence": "AA", "frozen_method_score": -1000.0},
        {"source_key": "s", "candidate_sequence": "AC", "frozen_method_score": 1000.0},
        {"source_key": "u", "candidate_sequence": "GG", "frozen_method_score": 2.0},
    ]
    result = evaluate_closed_method_scores_v3(config, measured, scores)
    assert result["source_macro_ndcg"] == 1.0
    assert result["per_source"]["u"]["status"] == "UNDEFINED_FEWER_THAN_TWO_LEGAL_MEASURED_CANDIDATES"
    assert result["per_source"]["u"]["ndcg"] is None
    assert result["undefined_sources_are_not_filled_with_zero"] is True


def test_frozen_score_table_must_exactly_match_common_candidates() -> None:
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v1",
        "pool_assignment": "DEVELOPMENT", "split": "VALIDATION",
        "analysis_unit": "SOURCE", "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
        "method_id": "strongest_matched_baseline", "base_flow_training_seed": 20260904,
    }
    measured = [
        {"source_key": "s", "candidate_sequence": "AA", "measured_direction_normalized_delta": 0.0, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION"},
        {"source_key": "s", "candidate_sequence": "AC", "measured_direction_normalized_delta": 1.0, "pool_assignment": "DEVELOPMENT", "split": "VALIDATION"},
    ]
    scores = [
        {"source_key": "s", "candidate_sequence": "AA", "frozen_method_score": 0.0},
        {"source_key": "s", "candidate_sequence": "AC", "frozen_method_score": 1.0},
        {"source_key": "other", "candidate_sequence": "GG", "frozen_method_score": 2.0},
    ]
    with pytest.raises(XEditFlowClosedScoresV3Error, match="exactly match"):
        evaluate_closed_method_scores_v3(config, measured, scores)
