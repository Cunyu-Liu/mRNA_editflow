from __future__ import annotations

from core.route2_legal_xeditflow import apply_action, initial_state, legal_actions
from core.route2_xeditflow_guidance_v4 import SetFlowMixtureStateV4
from core.route2_xeditflow_matched_methods_v4 import (
    CriticRewardBatchV4,
    ExactCriticRewardPotentialV4,
    SourceAnchoredFirstOrderPotentialV4,
    ZeroCriticPotentialV4,
    run_mode_fixed_matched_control_smc_v4,
)
from core.route2_xeditflow_smc_runtime_v4 import BatchedModeRateRowV4
from core.route2_xeditflow_smc_runtime_v4 import merge_smc_rounds_v4


def _root():
    return initial_state("AA", budget=1, assay_id="a", context_id="c")


def _modes():
    return [index % 8 for index in range(32)]


def _seeds():
    return list(range(100, 132))


def _rates(states):
    return [
        BatchedModeRateRowV4(
            tuple(legal_actions(state.flow_state)),
            tuple(1.0 for _ in legal_actions(state.flow_state)),
            state.trajectory_mode_id,
        )
        for state in states
    ]


def test_v4_unguided_control_keeps_mode_and_charges_no_value_or_critic() -> None:
    result = run_mode_fixed_matched_control_smc_v4(
        _root(),
        _rates,
        ZeroCriticPotentialV4(),
        method_id="unguided_setflow",
        source_key="s",
        particle_mode_ids=_modes(),
        particle_seeds=_seeds(),
        resampling_seed=200,
        beta_max=1.0,
    )
    compute = result["matched_compute"]
    assert result["setflow_mode_is_fixed_trajectory_state"] is True
    assert result["potential_kind"] == "ZERO"
    assert compute["trunk_forwards"] == 1
    assert compute["mode_forwards"] == 8
    assert compute["value_forwards"] == 0
    assert compute["critic_forwards_by_member"] == [0, 0, 0]
    assert all(row["contributing_mode_ids"] for row in result["candidates"])


def test_v4_simple_rate_deduplicates_states_and_charges_critic_not_value() -> None:
    batches = []

    def reward(states):
        batches.append(tuple(state.current_sequence for state in states))
        return CriticRewardBatchV4(
            tuple(float(state.edit_count) for state in states),
            (1, 1, 1),
        )

    result = run_mode_fixed_matched_control_smc_v4(
        _root(),
        _rates,
        ExactCriticRewardPotentialV4(reward),
        method_id="simple_rate_guidance",
        source_key="s",
        particle_mode_ids=_modes(),
        particle_seeds=_seeds(),
        resampling_seed=200,
        beta_max=1.0,
    )
    compute = result["matched_compute"]
    assert len(batches) == 1
    assert len(batches[0]) < 64
    assert compute["critic_forwards_by_member"] == [1, 1, 1]
    assert compute["value_forwards"] == 0
    assert result["critic_forward_is_not_value_forward"] is True
    merged = merge_smc_rounds_v4(
        [result],
        source_key="s",
        prior_trunk_forwards=1,
        prior_mode_forwards=8,
        terminal_critic_forwards_by_member=(2, 2, 2),
    )
    assert merged["matched_compute"]["critic_forwards_by_member"] == [3, 3, 3]


def test_v4_first_order_is_shared_across_modes_and_cached() -> None:
    calls = []

    def reward(states):
        calls.append(tuple(state.current_sequence for state in states))
        return CriticRewardBatchV4(
            tuple(float(state.edit_count) for state in states),
            (1, 1, 1),
        )

    potential = SourceAnchoredFirstOrderPotentialV4(_root(), reward)
    state = SetFlowMixtureStateV4(_root(), 0)
    edited_flow = apply_action(
        _root(), next(action for action in legal_actions(_root()) if action.kind == "SUB")
    )
    first = potential([state, SetFlowMixtureStateV4(edited_flow, 1)])
    second = potential([SetFlowMixtureStateV4(edited_flow, 7)])
    assert first.values == (0.0, 1.0)
    assert second.values == (1.0,)
    assert first.forward_batches_by_member == (1, 1, 1)
    assert second.forward_batches_by_member == (0, 0, 0)
    assert len(calls) == 1
