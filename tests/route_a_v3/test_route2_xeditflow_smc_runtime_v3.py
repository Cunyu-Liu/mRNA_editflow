from __future__ import annotations

import math

import pytest

from core.route2_legal_xeditflow import LegalAction, initial_state, legal_actions
from core.route2_xeditflow_smc_runtime_v3 import (
    BatchedRateRowV3,
    merge_smc_rounds_v3,
    run_batched_potential_smc_v3,
    scalar_potential_rate_map_v3,
    sample_base_proposal_v3,
)


def test_batched_smc_is_replayable_hard_legal_and_compute_bounded() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")

    def rates(states):
        return [
            BatchedRateRowV3(
                actions=tuple(legal_actions(state)),
                rates=tuple(1.0 for _ in legal_actions(state)),
            )
            for state in states
        ]

    def values(states):
        return [2.0 if state.current_sequence == "C" else 0.0 for state in states]

    kwargs = {
        "source_key": "source",
        "particle_seeds": list(range(32)),
        "resampling_seed": 20260930,
        "beta_max": 2.0,
    }
    first = run_batched_potential_smc_v3(root, rates, values, **kwargs)
    second = run_batched_potential_smc_v3(root, rates, values, **kwargs)
    assert first["candidates"] == second["candidates"]
    assert first["resampling_events"] == second["resampling_events"]
    assert first["hard_legality_rate"] == 1.0
    assert first["edit_budget_violation_count"] == 0
    assert first["candidate_budget_violation_count"] == 0
    assert first["matched_compute"]["base_flow_forwards"] == 1
    assert first["matched_compute"]["value_forwards"] == 1
    assert first["matched_compute"]["total_forward_equivalents"] == 2


def test_batched_smc_rejects_incomplete_or_illegal_action_bundle() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")

    def bad_rates(states):
        return [BatchedRateRowV3(actions=(LegalAction("SUB", 0, "A"),), rates=(1.0,)) for _ in states]

    with pytest.raises(Exception, match="exactly hard-legal"):
        run_batched_potential_smc_v3(
            root,
            bad_rates,
            lambda states: [0.0] * len(states),
            source_key="source",
            particle_seeds=list(range(32)),
            resampling_seed=1,
            beta_max=1.0,
        )


def test_base_proposal_uses_positive_legal_rates_only() -> None:
    action = LegalAction("STOP")
    assert sample_base_proposal_v3(
        [BatchedRateRowV3(actions=(action,), rates=(2.0,))], uniforms=[0.99]
    ) == [0]
    with pytest.raises(Exception, match="invalid legal rate"):
        sample_base_proposal_v3(
            [BatchedRateRowV3(actions=(action,), rates=(0.0,))], uniforms=[0.5]
        )


def test_additional_smc_rounds_merge_mass_and_reserve_three_critic_calls() -> None:
    def result(sequence, base=2, value=2):
        return {
            "status": "XEDITFLOW_V3_SMC_COMPLETE",
            "source_key": "s",
            "candidates": [{
                "candidate_sequence": sequence,
                "merged_log_weight": 0.0,
                "particle_multiplicity": 32,
            }],
            "matched_compute": {
                "base_flow_forwards": base,
                "value_forwards": value,
                "wall_time_seconds": 1.0,
            },
        }

    merged = merge_smc_rounds_v3([result("A"), result("A"), result("B")], source_key="s")
    assert merged["sampling_round_count"] == 3
    assert merged["candidates"][0]["candidate_sequence"] == "A"
    assert merged["candidates"][0]["particle_multiplicity"] == 64
    assert merged["matched_compute"]["total_forward_equivalents"] == 12
    assert merged["remaining_forward_equivalents_after_reservation"] == 305


def test_exact_scalar_potential_rate_map_covers_all_legal_actions() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")
    actions = tuple(legal_actions(root))

    def rates(states):
        assert states == [root]
        return [BatchedRateRowV3(actions=actions, rates=tuple(2.0 for _ in actions))]

    def values(states):
        return [0.0] + [1.0 if state.current_sequence == "C" else 0.0 for state in states[1:]]

    guided = scalar_potential_rate_map_v3(
        root, actions, rates, values, beta_max=2.0
    )
    assert set(guided) == set(actions)
    assert guided[LegalAction("SUB", 0, "C")] == pytest.approx(2.0 * math.exp(0.5))
