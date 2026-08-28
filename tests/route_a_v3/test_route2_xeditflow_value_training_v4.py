from __future__ import annotations

import copy

import pytest
import torch

from core.route2_source_token_cache_v3 import (
    SourceTokenCacheIndexV3,
    assemble_source_token_cache_v3,
)
from core.route2_xeditflow_guidance_v4 import XEditValueV4
from core.route2_xeditflow_value_training_v4 import (
    assemble_value_targets_v4,
    collate_value_targets_v4,
    load_value_checkpoint_v4,
    require_value_training_authorization_v4,
    value_distillation_loss_v4,
    value_target_records_v4,
)


def _readiness():
    return (
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_guidance_readiness.v1",
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
        },
        {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1",
            "status": "XEDITSETFLOW_V4_G0_READY",
            "required_seeds": [20260912, 20260913, 20260914],
            "seed_results": {
                str(seed): {"passed": True}
                for seed in (20260912, 20260913, 20260914)
            },
            "additional_seed_authorized": False,
            "development_test_authorized": False,
            "guidance_authorized": False,
            "critic_used": False,
            "independent_evaluator_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def _state(mode_id: int):
    return {
        "state_id": f"s{mode_id}",
        "split": "TRAIN",
        "base_flow_training_seed": 20260912,
        "trajectory_mode_id": mode_id,
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
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _rollouts(mode_id: int):
    return [
        {
            "state_id": f"s{mode_id}",
            "rollout_index": index,
            "base_flow_training_seed": 20260912,
            "trajectory_mode_id": mode_id,
            "critic_seeds": [20260908, 20260909, 20260910],
            "calibrated_seed_predictions": [
                float(index + mode_id),
                float(index + mode_id + 1),
                float(index + mode_id + 2),
            ],
            "study_neutral": True,
            "independent_evaluator_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for index in range(8)
    ]


def test_v4_value_targets_require_exact_k8_per_explicit_state_mode() -> None:
    critic, flow = _readiness()
    payload = assemble_value_targets_v4(
        [_state(0), _state(7)],
        _rollouts(0) + _rollouts(7),
        critic_readiness=critic,
        setflow_confirmation=flow,
        base_flow_training_seed=20260912,
        kappa=0.5,
        temperature=1.0,
    )
    assert payload["state_mode_count"] == 2
    assert {row["trajectory_mode_id"] for row in payload["records"]} == {0, 7}
    assert payload["setflow_mode_is_fixed_trajectory_state"] is True
    broken = _rollouts(0)[:-1]
    with pytest.raises(Exception, match="exactly rollout indices"):
        assemble_value_targets_v4(
            [_state(0)],
            broken,
            critic_readiness=critic,
            setflow_confirmation=flow,
            base_flow_training_seed=20260912,
            kappa=0.5,
            temperature=1.0,
        )


def test_v4_value_target_rejects_mode_drift_and_protected_contamination() -> None:
    critic, flow = _readiness()
    changed = _rollouts(0)
    changed[0]["trajectory_mode_id"] = 1
    with pytest.raises(Exception, match="changed the state trajectory mode"):
        assemble_value_targets_v4(
            [_state(0)],
            changed,
            critic_readiness=critic,
            setflow_confirmation=flow,
            base_flow_training_seed=20260912,
            kappa=0.0,
            temperature=0.5,
        )
    contaminated = copy.deepcopy(_rollouts(0))
    contaminated[0]["independent_evaluator_used"] = True
    with pytest.raises(Exception, match="independent evaluator"):
        assemble_value_targets_v4(
            [_state(0)],
            contaminated,
            critic_readiness=critic,
            setflow_confirmation=flow,
            base_flow_training_seed=20260912,
            kappa=0.0,
            temperature=0.5,
        )


def test_v4_value_targets_block_before_joint_readiness() -> None:
    critic, flow = _readiness()
    critic["loso_readiness_passed"] = False
    with pytest.raises(Exception, match="remain blocked"):
        require_value_training_authorization_v4(critic, flow)


def test_v4_value_collation_and_model_gradient_include_trajectory_mode() -> None:
    critic, flow = _readiness()
    payload = assemble_value_targets_v4(
        [_state(0), _state(7)],
        _rollouts(0) + _rollouts(7),
        critic_readiness=critic,
        setflow_confirmation=flow,
        base_flow_training_seed=20260912,
        kappa=0.0,
        temperature=0.5,
    )
    records = value_target_records_v4(payload)
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
    batch = collate_value_targets_v4(records, source_cache=cache)
    assert batch["trajectory_mode_ids"].tolist() == [0, 7]
    model = XEditValueV4(
        assay_count=2,
        context_count=2,
        quantity_count=2,
        measurement_count=3,
        numerator_count=1,
        denominator_count=1,
        dropout=0.0,
    )
    loss = value_distillation_loss_v4(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.scalar_head[-1].weight.grad is not None
    assert model.trajectory_mode.weight.grad is not None
    assert model.trajectory_mode.weight.grad[[0, 7]].abs().sum().item() > 0.0
    assert model(batch).shape == (2,)


def _checkpoint(path) -> dict:
    model_config = {
        "assay_count": 2,
        "context_count": 2,
        "quantity_count": 2,
        "measurement_count": 3,
        "numerator_count": 1,
        "denominator_count": 1,
        "dropout": 0.1,
    }
    model = XEditValueV4(**model_config)
    return {
        "schema_version": "route_a_v3_route2_xeditflow_value_checkpoint.v4",
        "model_state_dict": model.state_dict(),
        "model_config": {
            **model_config,
            "blocks": 6,
            "width": 384,
            "heads": 8,
            "ffn_width": 1536,
            "trajectory_mode_count": 8,
        },
        "base_flow_training_seed": 20260912,
        "critic_seeds": [20260908, 20260909, 20260910],
        "kappa": 0.5,
        "temperature": 1.0,
        "selected_pass": 8,
        "checkpoint_selection": "FINAL_PASS_8_NO_EPOCH_RESELECTION",
        "training_provenance": {
            "parameter_initialization_seed": 20260912,
            "parameter_initialization_seed_applied_before_model_construction": True,
            "optimizer_steps": 10,
            "parameter_changed": True,
            "cuda_available": True,
            "bf16_supported": True,
            "training_precision": "BF16",
            "cpu_fallback_used": False,
            "torch_device": "cuda:4",
            "physical_gpu_index": 4,
            "cuda_device_index": 4,
            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
            "cuda_device_uuid": "GPU-actual",
            "declared_physical_gpu_uuid": "actual",
            "cuda_parent_uuid_matches_declared_physical_index": True,
            "value_checkpoint_path": str(path),
        },
    }


def test_v4_value_checkpoint_loader_requires_frozen_identity(
    tmp_path,
) -> None:
    path = tmp_path / "value.pt"
    torch.save(_checkpoint(path), path)
    model, provenance = load_value_checkpoint_v4(
        path,
        base_flow_training_seed=20260912,
        kappa=0.5,
        temperature=1.0,
        device=torch.device("cpu"),
    )
    assert isinstance(model, XEditValueV4)
    assert model.training is False
    assert provenance["parameter_initialization_seed_applied_before_model_construction"]
    changed = _checkpoint(path)
    changed["selected_pass"] = 6
    torch.save(changed, path)
    with pytest.raises(Exception, match="frozen final pass"):
        load_value_checkpoint_v4(
            path,
            base_flow_training_seed=20260912,
            kappa=0.5,
            temperature=1.0,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "parameter_initialization_seed_applied_before_model_construction",
            False,
            "seed-before-model",
        ),
        ("bf16_supported", False, "BF16/no-CPU"),
        ("cuda_device_name", "NVIDIA H100", "actual A100"),
        ("cuda_parent_uuid_matches_declared_physical_index", False, "physical CUDA"),
        ("value_checkpoint_path", "/wrong/value.pt", "path provenance"),
    ),
)
def test_v4_value_checkpoint_loader_rejects_training_provenance_drift(
    tmp_path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "value.pt"
    checkpoint = _checkpoint(path)
    checkpoint["training_provenance"][field] = value
    torch.save(checkpoint, path)
    with pytest.raises(Exception, match=message):
        load_value_checkpoint_v4(
            path,
            base_flow_training_seed=20260912,
            kappa=0.5,
            temperature=1.0,
            device=torch.device("cpu"),
        )
