from __future__ import annotations

import copy

import pytest

from core.route2_legal_xeditflow import FlowState
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v4 import (
    _replay_identity,
    _selected_checkpoint_pass_v4,
    validate_value_rollout_config_v4,
)


def _config():
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_value_rollout_config.v4"
        ),
        "base_flow_training_seed": 20260912,
        "states_per_source": 4,
        "state_pass_index": 0,
        "rollouts_per_state_mode": 8,
        "sampling_state_batch_size": 32,
        "trajectory_forward_batch_size": 64,
        "fixed_seed_replay_check": True,
        "physical_gpu_index": 5,
        "device": "cuda:5",
        "output_dir": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            "experiments/xeditflow_v4/value_rollouts/seed_20260912"
        ),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _confirmation():
    return {
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": [20260912, 20260913, 20260914],
        "seed_results": {
            "20260912": {
                "selected_checkpoint_pass": 6,
                "passed": True,
            },
            "20260913": {
                "selected_checkpoint_pass": 8,
                "passed": True,
            },
            "20260914": {
                "selected_checkpoint_pass": 10,
                "passed": True,
            },
        },
    }


def test_v4_value_rollout_config_freezes_mode_replay_gpu_and_k8() -> None:
    validate_value_rollout_config_v4(_config())
    assert _selected_checkpoint_pass_v4(_confirmation(), seed=20260912) == 6


def test_v4_value_rollout_config_rejects_replay_or_protected_drift() -> None:
    config = _config()
    config["fixed_seed_replay_check"] = False
    with pytest.raises(Exception, match="replay check"):
        validate_value_rollout_config_v4(config)
    config = _config()
    config["development_test_outcomes_accessed_after_atomic_test"] = True
    with pytest.raises(Exception, match="reopened Development TEST"):
        validate_value_rollout_config_v4(config)
    confirmation = copy.deepcopy(_confirmation())
    confirmation["seed_results"]["20260912"]["passed"] = False
    with pytest.raises(Exception, match="no frozen selected checkpoint"):
        _selected_checkpoint_pass_v4(confirmation, seed=20260912)


def test_v4_rollout_replay_identity_includes_actions_terminal_and_forwards() -> None:
    state = FlowState(
        "AA",
        "CA",
        ((0, "C"),),
        0,
        "assay",
        "context",
        "BUDGET_EXHAUSTED",
    )
    first = [(state, ("A0C",), 1)]
    assert _replay_identity(first) == _replay_identity(first)
    changed = [(state, ("A0C",), 2)]
    assert _replay_identity(first) != _replay_identity(changed)
