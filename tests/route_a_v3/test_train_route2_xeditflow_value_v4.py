from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.train_route2_xeditflow_value_v4 import (
    validate_value_training_config_v4,
)


def _target():
    records = []
    for mode_id in range(8):
        records.append(
            {
                "state_id": f"s{mode_id}",
                "source_sequence": "AC",
                "current_sequence": "UC",
                "cache_record_id": "r",
                "remaining_budget": 2,
                "assigned_budget": 3,
                "trajectory_mode_id": mode_id,
                "quantity_id": 2,
                "measurement_id": 3,
                "numerator_id": 0,
                "denominator_id": 1,
                "assay_id": 4,
                "context_id": 5,
                "region_id": 0,
                "soft_value_target": 0.2,
            }
        )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_value_targets.v4",
        "status": "XEDITFLOW_V4_VALUE_TARGETS_COMPLETE",
        "split": "TRAIN",
        "base_flow_training_seed": 20260912,
        "critic_seeds": [20260908, 20260909, 20260910],
        "kappa": 0.5,
        "temperature": 1.0,
        "rollouts_per_state_mode": 8,
        "state_mode_count": 8,
        "records": records,
        "setflow_mode_is_fixed_trajectory_state": True,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _config():
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_training_config.v4"
        ),
        "base_flow_training_seed": 20260912,
        "kappa": 0.5,
        "temperature": 1.0,
        "passes": 8,
        "batch_size": 32,
        "precision": "BF16",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "dropout": 0.1,
        "checkpoint_selection": "FINAL_PASS_8_NO_EPOCH_RESELECTION",
        "physical_gpu_index": 4,
        "device": "cuda:4",
        "output_dir": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            "experiments/xeditflow_v4/value/seed20260912"
        ),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_value_training_config_freezes_budget_and_vocab_geometry() -> None:
    sizes = validate_value_training_config_v4(_config(), _target())
    assert sizes == {
        "assay_count": 5,
        "context_count": 6,
        "quantity_count": 3,
        "measurement_count": 4,
        "numerator_count": 1,
        "denominator_count": 2,
    }


def test_v4_value_training_config_rejects_pass_grid_or_mode_drift() -> None:
    config = _config()
    config["passes"] = 9
    with pytest.raises(Exception, match="passes changed"):
        validate_value_training_config_v4(config, _target())
    config = _config()
    config["kappa"] = 1.0
    with pytest.raises(Exception, match="kappa differs"):
        validate_value_training_config_v4(config, _target())
    target = copy.deepcopy(_target())
    target["records"] = target["records"][:-1]
    target["state_mode_count"] = 7
    with pytest.raises(Exception, match="all eight modes"):
        validate_value_training_config_v4(_config(), target)


def test_v4_value_training_config_rejects_gpu_or_protected_path_drift() -> None:
    config = _config()
    config["device"] = "cuda:3"
    with pytest.raises(Exception, match="device provenance"):
        validate_value_training_config_v4(config, _target())
    config = _config()
    config["development_test_outcomes_accessed_after_atomic_test"] = True
    with pytest.raises(Exception, match="reopened Development TEST"):
        validate_value_training_config_v4(config, _target())
