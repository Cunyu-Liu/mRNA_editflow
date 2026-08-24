from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from core.route2_legal_xeditflow import FlowState, LegalAction
from core.route2_xeditflow_gate_v4 import (
    GUIDANCE_GRID_V4,
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
