from __future__ import annotations

import copy

import pytest
import torch

from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, assemble_source_token_cache_v3
from core.route2_xeditflow_guidance_v3 import XEditValueV3
from core.route2_xeditflow_value_training_v3 import (
    assemble_value_targets_v3,
    collate_value_targets_v3,
    require_value_training_authorization_v3,
    value_distillation_loss_v3,
    value_target_records_v3,
)


def _readiness():
    return (
        {
            "status": "CRITIC_READY_FOR_GUIDANCE",
            "frozen_test_passed": True,
            "all_development_refit_complete": True,
            "loso_readiness_passed": True,
        },
        {
            "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS",
            "flow_status": "FLOW_G0_READY",
        },
    )


def _state():
    return {
        "state_id": "s0",
        "split": "TRAIN",
        "base_flow_training_seed": 20260904,
        "source_sequence": "ACGU",
        "current_sequence": "UCGU",
        "cache_record_id": "r0",
        "assigned_budget": 3,
        "remaining_budget": 2,
        "quantity_id": 1,
        "measurement_id": 2,
        "numerator_id": 0,
        "denominator_id": 0,
        "assay_id": 1,
        "context_id": 1,
        "region_id": 0,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _rollouts():
    return [
        {
            "state_id": "s0",
            "rollout_index": index,
            "base_flow_training_seed": 20260904,
            "critic_seeds": [20260831, 20260901, 20260902],
            "calibrated_seed_predictions": [float(index), float(index + 1), float(index + 2)],
            "study_neutral": True,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for index in range(8)
    ]


def test_value_targets_require_full_readiness_exact_k8_and_three_seed_reward() -> None:
    critic, flow = _readiness()
    payload = assemble_value_targets_v3(
        [_state()], _rollouts(),
        critic_readiness=critic,
        setflow_confirmation=flow,
        base_flow_training_seed=20260904,
        kappa=0.5,
        temperature=1.0,
    )
    assert payload["state_count"] == 1
    assert payload["records"][0]["rollout_count"] == 8
    assert payload["independent_evaluator_used"] is False
    broken = _rollouts()[:-1]
    with pytest.raises(Exception, match="exactly rollout indices"):
        assemble_value_targets_v3(
            [_state()], broken,
            critic_readiness=critic,
            setflow_confirmation=flow,
            base_flow_training_seed=20260904,
            kappa=0.5,
            temperature=1.0,
        )


def test_value_targets_reject_nontrain_and_block_before_readiness() -> None:
    critic, flow = _readiness()
    state = _state()
    state["split"] = "TEST"
    with pytest.raises(Exception, match="non-TRAIN"):
        assemble_value_targets_v3(
            [state], _rollouts(), critic_readiness=critic, setflow_confirmation=flow,
            base_flow_training_seed=20260904, kappa=0.0, temperature=0.5,
        )
    critic["loso_readiness_passed"] = False
    with pytest.raises(Exception, match="remain blocked"):
        require_value_training_authorization_v3(critic, flow)


def test_value_collation_and_scalar_huber_have_finite_gradient() -> None:
    critic, flow = _readiness()
    payload = assemble_value_targets_v3(
        [_state()], _rollouts(), critic_readiness=critic, setflow_confirmation=flow,
        base_flow_training_seed=20260904, kappa=0.0, temperature=0.5,
    )
    records = value_target_records_v3(payload)
    cache = SourceTokenCacheIndexV3(
        assemble_source_token_cache_v3(
            [{"canonical_record_id": "r0", "source_sequence": "ACGU"}],
            sequence_to_index={"ACGU": 0},
            encoded_tokens={0: torch.zeros(4, 768)},
            model_id="test",
            pretrained_parameter_count=1,
            attention_backend="test",
        )
    )
    batch = collate_value_targets_v3(records, source_cache=cache)
    model = XEditValueV3(
        assay_count=2, context_count=2, quantity_count=2,
        measurement_count=3, numerator_count=1, denominator_count=1,
        dropout=0.0,
    )
    loss = value_distillation_loss_v3(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.scalar_head[-1].weight.grad is not None
    assert model(batch).shape == (1,)


def test_value_artifact_rejects_independent_evaluator_contamination() -> None:
    critic, flow = _readiness()
    rollouts = copy.deepcopy(_rollouts())
    rollouts[0]["independent_evaluator_used"] = True
    with pytest.raises(Exception, match="independent evaluator"):
        assemble_value_targets_v3(
            [_state()], rollouts, critic_readiness=critic, setflow_confirmation=flow,
            base_flow_training_seed=20260904, kappa=1.0, temperature=1.0,
        )
