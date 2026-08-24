from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from core.route2_legal_xeditflow import FlowState, LegalAction, legal_actions
from core.route2_xeditflow_gate_v4 import (
    GUIDANCE_GRID_V4,
    adjudicate_guided_three_seed_v4,
    adjudicate_guidance_screen_v4,
    authorize_xeditflow_guidance_v4,
)
from core.route2_xeditflow_guidance_v4 import (
    MatchedComputeRecordV4,
    PotentialTransitionSetV4,
    SetFlowMixtureStateV4,
    apply_mode_fixed_action_v4,
    potential_guided_rates_v4,
    run_mode_fixed_scalar_potential_smc_v4,
    validate_mode_fixed_transition_v4,
)
from core.route2_xeditflow_smc_runtime_v4 import (
    BatchedModeRateRowV4,
    combine_primary_and_replay_compute_v4,
    merge_smc_rounds_v4,
    run_batched_mode_fixed_potential_smc_v4,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"


def _critic_ready() -> dict[str, object]:
    return {
        "status": "CRITIC_V4_READY_FOR_GUIDANCE",
        "three_seed_passed": True,
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "guidance_authorized": True,
    }


def _setflow_ready() -> dict[str, object]:
    return {
        "status": "XEDITSETFLOW_V4_G0_READY",
        "required_seeds": [20260912, 20260913, 20260914],
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_v4_protocol_freezes_grid_mode_and_compute_boundaries() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["base_flow_screen_seed"] == 20260912
    assert protocol["guidance_grid"]["combination_count"] == 18
    assert protocol["setflow_mode_state"]["fixed_for_complete_trajectory"] is True
    assert protocol["value_to_go"]["output"] == "ONE_SCALAR_PER_STATE_MODE"
    assert protocol["value_to_go"]["rollouts_per_state_mode"] == 8
    assert protocol["value_to_go"]["study_identity_input"] is False
    assert protocol["potential"]["free_action_ratio_head_allowed"] is False
    assert protocol["smc"]["forward_equivalent_ceiling_per_source"] == 320
    assert protocol["protected_outcomes"]["new_final_evaluation_outcome_reads"] == 0


def test_v4_guidance_requires_exact_joint_readiness_and_does_not_reopen_test() -> None:
    result = authorize_xeditflow_guidance_v4(_critic_ready(), _setflow_ready())
    assert result["status"] == "XEDITFLOW_V4_GUIDANCE_AUTHORIZED"
    assert result["development_test_reopened"] is False
    assert result["new_final_evaluation_authorized"] is False

    critic = _critic_ready()
    critic["development_test_access_event_count"] = 2
    assert authorize_xeditflow_guidance_v4(critic, _setflow_ready())["guidance_authorized"] is False
    flow = _setflow_ready()
    flow["critic_used"] = True
    assert authorize_xeditflow_guidance_v4(_critic_ready(), flow)["guidance_authorized"] is False


def _screen_results() -> dict[tuple[float, float, float], dict[str, object]]:
    return {
        combination: {
            "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_COMBINATION_COMPLETE",
            "base_flow_training_seed": 20260912,
            "combination": list(combination),
            "closed_source_macro_ndcg": 0.5,
            "closed_source_macro_normalized_regret": 0.4,
            "independent_evaluator_paired_margin": 0.1,
            "open_source_macro_candidate_recovery": 0.4,
            "total_forward_equivalents": 300,
            "setflow_mode_is_fixed_trajectory_state": True,
            "free_action_ratio_head_used": False,
            "all_network_forwards_separately_charged": True,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
        for combination in GUIDANCE_GRID_V4
    }


def test_v4_guidance_screen_is_exact_grid_with_frozen_selection_order() -> None:
    rows = _screen_results()
    winner = (0.5, 1.0, 2.0)
    rows[winner]["closed_source_macro_ndcg"] = 0.6
    result = adjudicate_guidance_screen_v4(rows)
    assert result["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
    assert (
        result["selected_kappa"],
        result["selected_temperature"],
        result["selected_beta_max"],
    ) == winner
    failed = copy.deepcopy(rows)
    failed[winner]["all_network_forwards_separately_charged"] = False
    with pytest.raises(Exception, match="mechanism or compute"):
        adjudicate_guidance_screen_v4(failed)


def _root() -> FlowState:
    return FlowState("AA", "AA", (), 1, "assay", "context")


def test_v4_state_action_and_transition_keep_mode_fixed() -> None:
    state = SetFlowMixtureStateV4(_root(), 3)
    child = apply_mode_fixed_action_v4(state, LegalAction("SUB", 0, "C"))
    assert child.trajectory_mode_id == 3
    transition = PotentialTransitionSetV4(
        actions=("edit",),
        children=(child,),
        base_rates=(1.0,),
        current_potential=0.0,
        child_potentials=(1.0,),
        progress=0.0,
    )
    validate_mode_fixed_transition_v4(state, transition)
    changed = PotentialTransitionSetV4(
        actions=("edit",),
        children=(SetFlowMixtureStateV4(child.flow_state, 4),),
        base_rates=(1.0,),
        current_potential=0.0,
        child_potentials=(1.0,),
        progress=0.0,
    )
    with pytest.raises(Exception, match="changed latent mode"):
        validate_mode_fixed_transition_v4(state, changed)


def test_v4_guided_rate_is_exactly_single_scalar_potential_difference() -> None:
    result = potential_guided_rates_v4(
        torch.tensor([[2.0, 3.0, 0.0]]),
        torch.tensor([[True, True, False]]),
        torch.tensor([1.0]),
        torch.tensor([[2.0, 0.5, float("nan")]]),
        progress=torch.tensor([1.0]),
        beta_max=1.0,
    )
    assert result[0, 0].item() == pytest.approx(2.0 * math.exp(1.0))
    assert result[0, 1].item() == pytest.approx(3.0 * math.exp(-0.5))
    assert result[0, 2].item() == 0.0


def test_v4_matched_compute_charges_trunk_mode_value_and_each_critic() -> None:
    record = MatchedComputeRecordV4(
        "source",
        trunk_forwards=50,
        mode_forwards=50,
        value_forwards=50,
        candidate_count=32,
        trajectory_count=32,
    )
    for member in range(3):
        record.add_critic_forwards(member, 40)
    payload = record.to_dict()
    assert payload["critic_forwards_by_member"] == [40, 40, 40]
    assert payload["total_forward_equivalents"] == 270
    assert payload["all_network_forwards_separately_charged"] is True
    record.mode_forwards += 51
    with pytest.raises(Exception, match="ceiling"):
        record.to_dict()


def test_v4_additional_smc_rounds_count_trajectories_and_all_forwards() -> None:
    def round_result(sequence: str) -> dict:
        compute = MatchedComputeRecordV4(
            "source",
            trunk_forwards=2,
            mode_forwards=16,
            value_forwards=2,
            candidate_count=1,
            trajectory_count=32,
        ).to_dict()
        combined = combine_primary_and_replay_compute_v4(
            compute, compute, replay_ok=True
        )
        return {
            "status": "XEDITFLOW_V4_SMC_COMPLETE",
            "source_key": "source",
            "setflow_mode_is_fixed_trajectory_state": True,
            "free_action_ratio_head_used": False,
            "matched_compute": combined,
            "candidates": [
                {
                    "candidate_sequence": sequence,
                    "merged_log_weight": 0.0,
                    "particle_multiplicity": 32,
                    "contributing_mode_ids": list(range(8)),
                }
            ],
        }

    merged = merge_smc_rounds_v4(
        [round_result("A"), round_result("C")],
        source_key="source",
        prior_trunk_forwards=1,
        prior_mode_forwards=8,
    )
    assert merged["sampling_round_count"] == 2
    assert merged["particle_count_per_round"] == 32
    assert merged["trajectory_count"] == 64
    assert merged["matched_compute"]["trajectory_count"] == 64
    assert merged["matched_compute"]["sampling_round_count"] == 2
    assert merged["matched_compute"]["critic_forwards_by_member"] == [1, 1, 1]
    assert merged["matched_compute"]["total_forward_equivalents"] == 92
    assert merged["remaining_forward_equivalents"] == 228


def test_v4_smc_replays_and_rejects_within_trajectory_mode_change() -> None:
    def provider(state: SetFlowMixtureStateV4) -> PotentialTransitionSetV4:
        child = apply_mode_fixed_action_v4(state, LegalAction("SUB", 0, "C"))
        return PotentialTransitionSetV4(
            actions=("A0C",),
            children=(child,),
            base_rates=(1.0,),
            current_potential=0.0,
            child_potentials=(float(state.trajectory_mode_id) / 10.0,),
            progress=0.0,
        )

    kwargs = {
        "particle_mode_ids": tuple(range(8)) * 4,
        "particle_seeds": tuple(range(100, 132)),
        "resampling_seed": 20260920,
        "beta_max": 1.0,
        "candidate_sequence": lambda state: state.current_sequence,
    }
    first = run_mode_fixed_scalar_potential_smc_v4(_root(), provider, **kwargs)
    second = run_mode_fixed_scalar_potential_smc_v4(_root(), provider, **kwargs)
    assert first == second
    assert first["setflow_mode_is_fixed_trajectory_state"] is True
    assert first["free_action_ratio_head_used"] is False
    assert first["initial_particle_mode_ids"] == list(range(8)) * 4

    def changed_provider(state: SetFlowMixtureStateV4) -> PotentialTransitionSetV4:
        child = apply_mode_fixed_action_v4(state, LegalAction("SUB", 0, "C"))
        changed = SetFlowMixtureStateV4(child.flow_state, (state.trajectory_mode_id + 1) % 8)
        return PotentialTransitionSetV4(
            actions=("A0C",),
            children=(changed,),
            base_rates=(1.0,),
            current_potential=0.0,
            child_potentials=(0.0,),
            progress=0.0,
        )

    with pytest.raises(Exception, match="changed latent mode"):
        run_mode_fixed_scalar_potential_smc_v4(
            _root(), changed_provider, **kwargs
        )


def test_v4_batched_smc_keeps_mode_and_separately_charges_networks() -> None:
    def rates(states):
        rows = []
        for state in states:
            actions = tuple(legal_actions(state.flow_state))
            rows.append(
                BatchedModeRateRowV4(
                    actions=actions,
                    rates=(1.0,) * len(actions),
                    trajectory_mode_id=state.trajectory_mode_id,
                )
            )
        return rows

    def values(states):
        return [float(state.trajectory_mode_id) / 10.0 for state in states]

    kwargs = {
        "source_key": "source",
        "particle_mode_ids": tuple(range(8)) * 4,
        "particle_seeds": tuple(range(200, 232)),
        "resampling_seed": 20260921,
        "beta_max": 1.0,
    }
    first = run_batched_mode_fixed_potential_smc_v4(
        _root(), rates, values, **kwargs
    )
    second = run_batched_mode_fixed_potential_smc_v4(
        _root(), rates, values, **kwargs
    )
    assert first["candidates"] == second["candidates"]
    assert first["resampling_events"] == second["resampling_events"]
    assert first["setflow_mode_is_fixed_trajectory_state"] is True
    compute = first["matched_compute"]
    assert compute["trunk_forwards"] == 1
    assert compute["mode_forwards"] == 8
    assert compute["value_forwards"] == 1
    assert compute["trajectory_count"] == 32
    assert compute["all_network_forwards_separately_charged"] is True


def test_v4_batched_smc_rejects_rate_provider_mode_drift() -> None:
    def rates(states):
        rows = []
        for state in states:
            actions = tuple(legal_actions(state.flow_state))
            rows.append(
                BatchedModeRateRowV4(
                    actions=actions,
                    rates=(1.0,) * len(actions),
                    trajectory_mode_id=(state.trajectory_mode_id + 1) % 8,
                )
            )
        return rows

    with pytest.raises(Exception, match="changed the trajectory mode"):
        run_batched_mode_fixed_potential_smc_v4(
            _root(),
            rates,
            lambda states: [0.0] * len(states),
            source_key="source",
            particle_mode_ids=tuple(range(8)) * 4,
            particle_seeds=tuple(range(300, 332)),
            resampling_seed=20260922,
            beta_max=1.0,
        )


def _guided_method(ndcg: float, regret: float, top_1: float = 0.6):
    return {
        "closed_source_macro_ndcg": ndcg,
        "closed_source_macro_normalized_regret": regret,
        "closed_source_macro_top_1_recall": top_1,
    }


def _guided_v4_payloads():
    payloads = {}
    for seed in (20260912, 20260913, 20260914):
        full = {
            **_guided_method(0.72, 0.30),
            "open_source_macro_candidate_recovery": 0.36,
            "open_source_macro_top_k_recovery": 0.21,
            "open_source_macro_unique_candidate_rate": 0.92,
            "independent_evaluator_margin_over_strongest_baseline": 0.12,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0,
            "numerical_failure_count": 0,
            "maximum_forward_equivalents_per_source": 320,
        }
        payloads[seed] = {
            "methods": {
                "full_soft_value_smc": full,
                "unguided_setflow": _guided_method(0.64, 0.40),
                "first_order_guidance": _guided_method(0.62, 0.42),
                "simple_rate_guidance": _guided_method(0.63, 0.41),
                "generate_then_rerank": _guided_method(0.65, 0.39),
                "strongest_matched_baseline": _guided_method(0.64, 0.40),
            },
            "source_paired_ndcg_improvement_ci_95": {
                "over_unguided": [0.02, 0.14],
                "over_strongest_baseline": [0.02, 0.14],
            },
            "source_paired_independent_evaluator_margin_ci_95": [0.02, 0.20],
            "critic_self_score_increased": True,
            "all_methods_matched_compute_ceiling_met": True,
            "matched_compute_schema": "MatchedComputeRecordV4",
            "setflow_mode_is_fixed_trajectory_state": True,
            "free_action_ratio_head_used": False,
            "all_network_forwards_separately_charged": True,
            "independent_evaluator_in_gradient": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
    return payloads


def test_v4_three_seed_gate_requires_measured_and_independent_improvement() -> None:
    result = adjudicate_guided_three_seed_v4(_guided_v4_payloads())
    assert result["status"] == "XEDITFLOW_V4_PASS"
    assert result["new_final_evaluation_authorized"] is True
    assert result["submission_ready"] is False
    failed = _guided_v4_payloads()
    failed[20260913]["methods"]["full_soft_value_smc"][
        "closed_source_macro_ndcg"
    ] = 0.65
    result = adjudicate_guided_three_seed_v4(failed)
    assert result["status"] == "XEDITFLOW_V4_NO_GO"
    assert result["reward_exploitation"] is True
    assert result["new_final_evaluation_authorized"] is False


def test_v4_three_seed_gate_rejects_mode_compute_or_seed_drift() -> None:
    payloads = _guided_v4_payloads()
    payloads[20260912]["all_network_forwards_separately_charged"] = False
    with pytest.raises(Exception, match="mechanism or compute"):
        adjudicate_guided_three_seed_v4(payloads)
    payloads = _guided_v4_payloads()
    payloads[20260915] = payloads.pop(20260914)
    with pytest.raises(Exception, match="exactly the three"):
        adjudicate_guided_three_seed_v4(payloads)
