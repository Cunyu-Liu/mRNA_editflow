from __future__ import annotations

import pytest

from scripts.route_a_v3.train_route2_xeditflow_value_v3 import validate_value_training_config_v3


def _target():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_value_targets.v3",
        "status": "XEDITFLOW_V3_VALUE_TARGETS_COMPLETE",
        "split": "TRAIN",
        "base_flow_training_seed": 20260904,
        "critic_seeds": [20260831, 20260901, 20260902],
        "kappa": 0.5,
        "temperature": 1.0,
        "rollouts_per_state": 8,
        "state_count": 1,
        "records": [{
            "state_id": "s",
            "source_sequence": "AC",
            "current_sequence": "UC",
            "cache_record_id": "r",
            "remaining_budget": 2,
            "assigned_budget": 3,
            "quantity_id": 2,
            "measurement_id": 3,
            "numerator_id": 0,
            "denominator_id": 1,
            "assay_id": 4,
            "context_id": 5,
            "region_id": 0,
            "soft_value_target": 0.2,
        }],
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _config():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_value_training_config.v1",
        "base_flow_training_seed": 20260904,
        "kappa": 0.5,
        "temperature": 1.0,
        "passes": 8,
        "batch_size": 32,
        "precision": "BF16",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "dropout": 0.1,
        "checkpoint_selection": "FINAL_PASS_NO_EPOCH_RESELECTION",
    }


def test_value_training_config_freezes_final_pass_and_vocab_geometry() -> None:
    sizes = validate_value_training_config_v3(_config(), _target())
    assert sizes == {
        "assay_count": 5,
        "context_count": 6,
        "quantity_count": 3,
        "measurement_count": 4,
        "numerator_count": 1,
        "denominator_count": 2,
    }


def test_value_training_config_rejects_epoch_or_grid_drift() -> None:
    config = _config()
    config["passes"] = 9
    with pytest.raises(Exception, match="passes changed"):
        validate_value_training_config_v3(config, _target())
    config = _config()
    config["kappa"] = 1.0
    with pytest.raises(Exception, match="kappa differs"):
        validate_value_training_config_v3(config, _target())
