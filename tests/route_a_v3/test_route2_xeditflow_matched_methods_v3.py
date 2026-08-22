from __future__ import annotations

import pytest

from core.route2_legal_xeditflow import LegalAction, apply_action, initial_state, legal_actions
from core.route2_xeditflow_matched_methods_v3 import (
    CriticRewardBatchV3,
    ExactCriticRewardPotentialV3,
    SourceAnchoredFirstOrderPotentialV3,
    ZeroPotentialV3,
    merge_matched_control_rounds_v3,
    rerank_terminal_candidates_v3,
    run_batched_critic_potential_smc_v3,
)
from core.route2_xeditflow_smc_runtime_v3 import BatchedRateRowV3


def _rates(states):
    return [
        BatchedRateRowV3(
            actions=tuple(legal_actions(state)),
            rates=tuple(1.0 for _ in legal_actions(state)),
        )
        for state in states
    ]


class InteractionReward:
    def __init__(self):
        self.calls = 0

    def __call__(self, states):
        self.calls += 1
        values = []
        for state in states:
            edits = len(state.source_relative_edits)
            value = float(edits)
            if state.current_sequence == "CC":
                value = 5.0
            values.append(value)
        return CriticRewardBatchV3(tuple(values), (1, 1, 1))


def test_first_order_is_source_anchored_while_exact_reward_has_interaction() -> None:
    root = initial_state("AA", budget=3, assay_id="a", context_id="c")
    first_reward = InteractionReward()
    first = SourceAnchoredFirstOrderPotentialV3(root, first_reward)
    exact_reward = InteractionReward()
    exact = ExactCriticRewardPotentialV3(exact_reward)
    state_c_a = apply_action(root, LegalAction("SUB", 0, "C"))
    state_c_c = apply_action(state_c_a, LegalAction("SUB", 1, "C"))
    first_values = first([root, state_c_a, state_c_c])
    exact_values = exact([root, state_c_a, state_c_c])
    assert first_values.values == pytest.approx((0.0, 1.0, 2.0))
    assert exact_values.values == pytest.approx((0.0, 1.0, 5.0))
    calls_after_fill = first_reward.calls
    cached = first([state_c_c, state_c_a])
    assert cached.values == pytest.approx((2.0, 1.0))
    assert cached.forward_batches_by_member == (0, 0, 0)
    assert first_reward.calls == calls_after_fill


def test_first_order_and_simple_rate_smc_are_replayable_and_separately_accounted() -> None:
    root = initial_state("AA", budget=1, assay_id="a", context_id="c")
    seeds = list(range(32))
    for method in ("first_order_guidance", "simple_rate_guidance"):
        def run_once():
            reward = InteractionReward()
            potential = (
                SourceAnchoredFirstOrderPotentialV3(root, reward)
                if method == "first_order_guidance"
                else ExactCriticRewardPotentialV3(reward)
            )
            return run_batched_critic_potential_smc_v3(
                root,
                _rates,
                potential,
                method_id=method,
                source_key="s",
                particle_seeds=seeds,
                resampling_seed=20260930,
                beta_max=1.0,
            )
        first = run_once()
        second = run_once()
        assert first["candidates"] == second["candidates"]
        assert first["resampling_events"] == second["resampling_events"]
        assert first["hard_legality_rate"] == 1.0
        assert first["candidate_budget_violation_count"] == 0
        assert first["matched_compute"]["critic_forwards_by_member"] == [1, 1, 1]
        assert first["matched_compute"]["total_forward_equivalents"] == 4
        assert first["matched_compute"]["total_forward_equivalents"] <= 320


def test_unguided_common_sampler_has_no_critic_or_value_charge() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")
    result = run_batched_critic_potential_smc_v3(
        root,
        _rates,
        ZeroPotentialV3(),
        method_id="unguided_setflow",
        source_key="s",
        particle_seeds=list(range(32)),
        resampling_seed=2,
        beta_max=1.0,
    )
    assert result["incremental_importance_weight"] == "UNITY"
    assert result["matched_compute"]["base_flow_forwards"] == 1
    assert result["matched_compute"]["value_forwards"] == 0
    assert result["matched_compute"]["critic_forwards_by_member"] == [0, 0, 0]
    merged = merge_matched_control_rounds_v3([result], source_key="s")
    assert merged["remaining_forward_equivalents_after_reservation"] == 295


def test_critic_potential_smc_rejects_incomplete_legal_bundle() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")
    reward = InteractionReward()
    potential = ExactCriticRewardPotentialV3(reward)
    with pytest.raises(Exception, match="hard legality"):
        run_batched_critic_potential_smc_v3(
            root,
            lambda states: [BatchedRateRowV3(actions=(), rates=()) for _ in states],
            potential,
            method_id="simple_rate_guidance",
            source_key="s",
            particle_seeds=list(range(32)),
            resampling_seed=1,
            beta_max=1.0,
        )


def test_terminal_rerank_changes_order_without_changing_support() -> None:
    candidates = [
        {"candidate_sequence": "AC", "trajectory_actions": ["STOP"]},
        {"candidate_sequence": "AG", "trajectory_actions": ["SUB:1:G"]},
    ]
    ranked = rerank_terminal_candidates_v3(candidates, [0.1, 0.9])
    assert [row["candidate_sequence"] for row in ranked] == ["AG", "AC"]
    assert {row["candidate_sequence"] for row in ranked} == {"AC", "AG"}
    assert ranked[0]["trajectory_actions"] == ["SUB:1:G"]
