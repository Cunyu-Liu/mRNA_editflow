from __future__ import annotations

import pytest

from scripts.route_a_v3.score_route2_xeditflow_closed_frozen_methods_v3 import (
    _terminal_state_v3,
    score_critic_states_for_method_v3,
    validate_closed_frozen_score_config_v3,
)
from core.route2_legal_xeditflow import initial_state
from core.route2_xeditflow_matched_methods_v3 import CriticRewardBatchV3


def _config(method_id: str) -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_frozen_score_config.v1",
        "method_id": method_id,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "base_flow_training_seed": 20260904,
        "expected_source_count": 891,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "kappa": 0.5,
        "critic_online_microbatch_size": 4,
        "critic_refit_manifest_path": "/mnt/refit.json",
        "strongest_generation_baseline_path": "/mnt/strongest.json",
        "baseline_selection_input_path": "/mnt/selection.json",
    }


@pytest.mark.parametrize(
    "method_id",
    [
        "first_order_guidance",
        "simple_rate_guidance",
        "generate_then_rerank",
        "strongest_matched_baseline",
    ],
)
def test_closed_frozen_score_config_accepts_only_frozen_rerank_search_methods(method_id: str) -> None:
    validate_closed_frozen_score_config_v3(_config(method_id))


def test_closed_frozen_score_config_rejects_full_trajectory_method() -> None:
    with pytest.raises(Exception, match="method differs"):
        validate_closed_frozen_score_config_v3(_config("full_soft_value_smc"))


def test_closed_frozen_score_config_rejects_test_cohort() -> None:
    config = _config("generate_then_rerank")
    config["split"] = "TEST"
    with pytest.raises(Exception, match="cohort"):
        validate_closed_frozen_score_config_v3(config)


def test_closed_frozen_score_config_requires_strongest_selection_boundary() -> None:
    config = _config("strongest_matched_baseline")
    config["baseline_selection_input_path"] = ""
    with pytest.raises(Exception, match="selection input"):
        validate_closed_frozen_score_config_v3(config)


def test_closed_first_order_score_differs_from_terminal_reward_controls() -> None:
    root = initial_state("AA", budget=1, assay_id="a", context_id="c")
    terminal = _terminal_state_v3(root, "CA")

    def reward(states):
        return CriticRewardBatchV3(
            tuple(10.0 + state.edit_count for state in states),
            (1, 1, 1),
        )

    first_order = score_critic_states_for_method_v3(
        "first_order_guidance", root, [terminal], reward
    )
    simple = score_critic_states_for_method_v3(
        "simple_rate_guidance", root, [terminal], reward
    )
    rerank = score_critic_states_for_method_v3(
        "generate_then_rerank", root, [terminal], reward
    )
    assert first_order.values == (1.0,)
    assert simple.values == rerank.values == (11.0,)
