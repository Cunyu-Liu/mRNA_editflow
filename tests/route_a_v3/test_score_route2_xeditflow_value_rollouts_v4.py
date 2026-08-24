from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.score_route2_xeditflow_value_rollouts_v4 import (
    projection_rows_from_terminal_rollouts_v4,
    validate_value_critic_score_config_v4,
)


def _config():
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_critic_score_config.v4"
        ),
        "critic_seeds": [20260908, 20260909, 20260910],
        "base_flow_training_seed": 20260912,
        "critic_refit_runtime_config_paths": {
            "20260908": "/mnt/refit-8.json",
            "20260909": "/mnt/refit-9.json",
            "20260910": "/mnt/refit-10.json",
        },
        "candidate_batch_size": 128,
        "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
        "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
        "trajectory_mode_used_as_critic_input": False,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "terminal_rollout_path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/rollouts.jsonl"
        ),
        "output_dir": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/critic-scores"
        ),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _terminal():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_terminal_rollout.v4",
        "state_id": "state",
        "rollout_index": 0,
        "base_flow_training_seed": 20260912,
        "trajectory_mode_id": 6,
        "source_sequence": "AA",
        "candidate_sequence": "CA",
        "source_relative_edits": [
            {"position": 0, "source_base": "A", "candidate_base": "C"}
        ],
        "task_id": "task",
        "source_group_id": "source",
        "endpoint_descriptor": {
            "quantity_family": "quantity",
            "measurement_form": "measurement",
            "numerator_family": None,
            "denominator_family": None,
        },
        "assay_category": "assay",
        "context_category": "context",
        "region_id": 0,
        "terminal_cause": "BUDGET_EXHAUSTED",
        "setflow_mode_is_fixed_trajectory_state": True,
    }


def test_v4_value_critic_score_config_freezes_seed_study_mode_and_paths() -> None:
    validate_value_critic_score_config_v4(_config())
    config = _config()
    config["base_flow_training_seed"] = 20260914
    validate_value_critic_score_config_v4(config)


def test_v4_value_critic_score_config_rejects_mode_or_protected_drift() -> None:
    config = _config()
    config["trajectory_mode_used_as_critic_input"] = True
    with pytest.raises(Exception, match="mode entered critic"):
        validate_value_critic_score_config_v4(config)
    config = _config()
    config["development_test_outcomes_accessed_after_atomic_test"] = True
    with pytest.raises(Exception, match="reopened Development TEST"):
        validate_value_critic_score_config_v4(config)
    config = _config()
    config["base_flow_training_seed"] = 20260915
    with pytest.raises(Exception, match="base-flow seed is undeclared"):
        validate_value_critic_score_config_v4(config)


def test_v4_terminal_rollout_conversion_uses_dummy_zero_only_for_inference() -> None:
    rows = projection_rows_from_terminal_rollouts_v4(
        [_terminal()], global_start=7, base_flow_training_seed=20260912
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["canonical_record_id"] == "generated-000000000007"
    assert row["study_unit_id"] == "__UNK__"
    assert row["direction_normalized_delta"] == 0.0
    assert row["dummy_target_for_inference_only"] is True
    assert "trajectory_mode_id" not in row

    terminal = copy.deepcopy(_terminal())
    terminal["source_relative_edits"][0]["position"] = 1
    with pytest.raises(Exception, match="candidate bundle differs"):
        projection_rows_from_terminal_rollouts_v4(
            [terminal], global_start=0, base_flow_training_seed=20260912
        )


def test_v4_terminal_rollout_conversion_rejects_mixed_base_flow_seed() -> None:
    terminal = _terminal()
    terminal["base_flow_training_seed"] = 20260913
    with pytest.raises(Exception, match="base-flow seed differs"):
        projection_rows_from_terminal_rollouts_v4(
            [terminal], global_start=0, base_flow_training_seed=20260912
        )
