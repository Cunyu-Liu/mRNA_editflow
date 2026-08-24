from __future__ import annotations

import pytest

from scripts.route_a_v3.evaluate_route2_xeditflow_open_generation_v4 import (
    validate_open_generation_config_v4,
)


def _config() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_config.v4",
        "pool_assignment": "DEVELOPMENT",
        "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "undefined_outcome_policy": "UNKNOWN_NOT_ZERO",
        "base_flow_training_seed": 20260912,
        "kappa": 0.5,
        "temperature": 1.0,
        "beta_max": 2.0,
        "method_id": "method",
        "measured_top_k": 10,
        "critic_self_score_used_for_ranking": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_open_metric_keeps_unknown_outcomes_and_generation_ranking() -> None:
    validate_open_generation_config_v4(_config())
    changed = _config()
    changed["undefined_outcome_policy"] = "ZERO_FILL"
    with pytest.raises(Exception, match="support policy"):
        validate_open_generation_config_v4(changed)
    changed = _config()
    changed["critic_self_score_used_for_ranking"] = True
    with pytest.raises(Exception, match="metric or protected-input"):
        validate_open_generation_config_v4(changed)
