from __future__ import annotations

import pytest

from scripts.route_a_v3.prepare_route2_xeditflow_value_rollout_config_v3 import (
    ValueRolloutPrepareV3Error,
    build_value_rollout_config_v3,
)


def _protocol() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v3_guidance_protocol.v1",
        "status": "FROZEN_PROSPECTIVE_BEFORE_CRITIC_SETFLOW_OR_GUIDANCE_OUTCOME_READ",
        "critic_readiness_path": "/mnt/critic_ready.json",
        "setflow_confirmation_path": "/mnt/flow_ready.json",
        "critic_refit_manifest_path": "/mnt/refit.json",
        "mrnabert_model_path": "/mnt/model",
        "value_rollout_output_dir": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/value_rollout",
        "value_state_policy": {
            "split": "TRAIN",
            "base_flow_training_seed": 20260904,
            "state_pass_index": 0,
            "states_per_record": 2,
            "rollouts_per_state": 8,
        },
        "rollout_execution_policy": {
            "sampling_state_batch_size": 128,
            "trajectory_forward_batch_size": 64,
            "critic_batch_size": 256,
            "critic_online_microbatch_size": 4,
            "training_precision": "BF16",
            "allowed_physical_gpu_indices": [0, 1, 2, 3, 4, 5],
        },
        "critic_reward_policy": {
            "critic_seeds": [20260831, 20260901, 20260902],
            "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
            "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
            "independent_evaluator_used": False,
        },
        "guidance_grid": {
            "kappa": [0.0, 0.5, 1.0],
            "temperature": [0.5, 1.0],
            "beta_max": [0.5, 1.0, 2.0],
            "additional_combination_authorized": False,
        },
    }


def test_prepare_value_rollout_uses_seed_20260904_selected_checkpoint() -> None:
    critic = {
        "status": "CRITIC_READY_FOR_GUIDANCE",
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
    }
    flow = {
        "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS",
        "flow_status": "FLOW_G0_READY",
        "selected_arm": "f3",
    }
    refit = {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "completed_refit_count": 3,
    }
    runtime = {
        "seed": 20260904,
        "selected_arm": "f3",
        "output_root": "/mnt/run/f3/seed20260904",
        "train_projection_path": "/mnt/train.jsonl",
        "source_token_cache_path": "/mnt/cache.pt",
    }
    config = build_value_rollout_config_v3(
        _protocol(), critic, flow, refit, runtime, physical_gpu_index=5
    )
    assert config["setflow_checkpoint_path"] == "/mnt/run/f3/seed20260904/f3/best.pt"
    assert config["base_flow_training_seed"] == 20260904
    assert config["rollouts_per_state"] == 8
    assert config["device"] == "cuda:5"


def test_prepare_value_rollout_blocks_before_critic_readiness() -> None:
    critic = {
        "status": "CRITIC_NOT_READY_FOR_GUIDANCE",
        "frozen_test_passed": False,
        "all_development_refit_complete": False,
        "loso_readiness_passed": False,
    }
    flow = {
        "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS",
        "flow_status": "FLOW_G0_READY",
        "selected_arm": "f2",
    }
    with pytest.raises(ValueRolloutPrepareV3Error, match="blocked"):
        build_value_rollout_config_v3(
            _protocol(), critic, flow, {"status": "x"}, {}, physical_gpu_index=0
        )
