from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.evaluate_route2_xeditflow_closed_scores_v4 import (
    evaluate_closed_method_scores_v4,
)


def _measured():
    return [
        {
            "source_key": "s",
            "candidate_sequence": "AA",
            "measured_direction_normalized_delta": 0.0,
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
        },
        {
            "source_key": "s",
            "candidate_sequence": "AC",
            "measured_direction_normalized_delta": 1.0,
            "pool_assignment": "DEVELOPMENT",
            "split": "VALIDATION",
        },
    ]


def _config(method="simple_rate_guidance"):
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v4",
        "method_id": method,
        "base_flow_training_seed": 20260914,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "analysis_unit": "SOURCE",
        "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _scores(method="simple_rate_guidance"):
    return [
        {
            "source_key": "s",
            "candidate_sequence": "AA",
            "frozen_method_score": 0.0,
            "method_id": method,
            "base_flow_training_seed": 20260914,
            "kappa": 0.5,
            "temperature": 1.0,
            "beta_max": 2.0,
            "measured_outcome_used_to_construct_score": False,
            "independent_evaluator_used": False,
        },
        {
            "source_key": "s",
            "candidate_sequence": "AC",
            "frozen_method_score": 1.0,
            "method_id": method,
            "base_flow_training_seed": 20260914,
            "kappa": 0.5,
            "temperature": 1.0,
            "beta_max": 2.0,
            "measured_outcome_used_to_construct_score": False,
            "independent_evaluator_used": False,
        },
    ]


def _summary(method="simple_rate_guidance"):
    return {
        "status": "XEDITFLOW_V4_CLOSED_CONTROL_SCORES_COMPLETE",
        "method_id": method,
        "base_flow_training_seed": 20260914,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "measured_outcome_used_to_construct_score": False,
        "independent_evaluator_used": False,
    }


def test_v4_closed_control_scores_use_common_v4_metric_schema() -> None:
    result = evaluate_closed_method_scores_v4(
        _config(), _measured(), _scores(), score_summary=_summary()
    )
    assert result["status"] == "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE"
    assert result["source_macro_ndcg"] == 1.0
    assert result["base_flow_training_seed"] == 20260914
    assert result["undefined_sources_are_not_filled_with_zero"] is True


def test_v4_closed_control_scores_reject_contaminated_score_provenance() -> None:
    scores = _scores()
    scores[0] = {**scores[0], "measured_outcome_used_to_construct_score": True}
    with pytest.raises(Exception, match="score row provenance"):
        evaluate_closed_method_scores_v4(
            _config(), _measured(), scores, score_summary=_summary()
        )


def test_v4_strongest_closed_baseline_reuses_prefrozen_score_table() -> None:
    config = _config("strongest_matched_baseline")
    config.update(
        {
            "score_table_method_id": "strongest_matched_baseline",
            "strongest_baseline_frozen_before_v4_candidate_generation": True,
            "baseline_reselected_for_v4": False,
        }
    )
    scores = [
        {
            "source_key": row["source_key"],
            "candidate_sequence": row["candidate_sequence"],
            "frozen_method_score": float(index),
            "method_id": "strongest_matched_baseline",
        }
        for index, row in enumerate(_measured())
    ]
    result = evaluate_closed_method_scores_v4(config, _measured(), scores)
    assert result["method_id"] == "strongest_matched_baseline"
    assert result["status"] == "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE"
