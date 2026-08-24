from __future__ import annotations

import copy

import pytest

from scripts.route_a_v3.build_route2_xeditflow_final_value_target_v4 import (
    validate_final_value_target_config_v4,
)


def _gate():
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1",
        "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "base_flow_training_seed": 20260912,
        "combination_count": 18,
        "selected_kappa": 0.5,
        "selected_temperature": 1.0,
        "selected_beta_max": 2.0,
    }


def _config():
    root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_value_target_config.v4",
        "base_flow_training_seed": 20260913,
        "kappa": 0.5,
        "temperature": 1.0,
        "train_state_path": f"{root}/states.jsonl",
        "frozen_rollout_score_path": f"{root}/scores.jsonl",
        "rollout_summary_path": f"{root}/rollout.json",
        "critic_score_summary_path": f"{root}/critic.json",
        "critic_readiness_path": f"{root}/critic_ready.json",
        "setflow_confirmation_path": f"{root}/flow_ready.json",
        "guidance_screen_gate_path": f"{root}/guidance_gate.json",
        "output_dir": f"{root}/final_value_target",
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_v4_final_value_target_accepts_only_non_screen_seeds_and_selected_pair() -> None:
    validate_final_value_target_config_v4(_config(), _gate())
    config = _config()
    config["base_flow_training_seed"] = 20260914
    validate_final_value_target_config_v4(config, _gate())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_flow_training_seed", 20260912, "non-screen"),
        ("kappa", 1.0, "selected"),
        ("temperature", 0.5, "selected"),
    ],
)
def test_v4_final_value_target_rejects_reselection(
    field: str, value: object, message: str
) -> None:
    config = copy.deepcopy(_config())
    config[field] = value
    with pytest.raises(Exception, match=message):
        validate_final_value_target_config_v4(config, _gate())


def test_v4_final_value_target_rejects_beta_in_target() -> None:
    config = _config()
    config["beta_max"] = 2.0
    with pytest.raises(Exception, match="selected"):
        validate_final_value_target_config_v4(config, _gate())
