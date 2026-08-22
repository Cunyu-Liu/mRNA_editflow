#!/usr/bin/env python3
"""Prepare the single seed-20260904 value-rollout job after both readiness gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import authorize_xeditflow_guidance_v3
from scripts.route_a_v3.generate_route2_xeditflow_value_rollouts_v3 import (
    validate_value_rollout_config_v3,
)


class ValueRolloutPrepareV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueRolloutPrepareV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def build_value_rollout_config_v3(
    protocol: Mapping[str, Any],
    critic_readiness: Mapping[str, Any],
    setflow_confirmation: Mapping[str, Any],
    critic_refit_manifest: Mapping[str, Any],
    setflow_runtime: Mapping[str, Any],
    *,
    physical_gpu_index: int,
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditflow_v3_guidance_protocol.v1",
        "unexpected XEditFlow V3 guidance protocol",
    )
    _require(
        protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_CRITIC_SETFLOW_OR_GUIDANCE_OUTCOME_READ",
        "XEditFlow V3 guidance protocol is not prospective",
    )
    _require(
        protocol.get("value_state_policy")
        == {
            "split": "TRAIN",
            "base_flow_training_seed": 20260904,
            "state_pass_index": 0,
            "states_per_record": 2,
            "rollouts_per_state": 8,
        },
        "value state policy changed",
    )
    _require(
        protocol.get("rollout_execution_policy")
        == {
            "sampling_state_batch_size": 128,
            "trajectory_forward_batch_size": 64,
            "critic_batch_size": 256,
            "critic_online_microbatch_size": 4,
            "training_precision": "BF16",
            "allowed_physical_gpu_indices": [0, 1, 2, 3, 4, 5],
        },
        "value rollout execution policy changed",
    )
    _require(
        protocol.get("critic_reward_policy")
        == {
            "critic_seeds": [20260831, 20260901, 20260902],
            "study_policy": "UNKNOWN_STUDY_SCALE_FIXED_1",
            "prediction_scale": "TASK_ROBUST_STANDARDIZED_EFFECT",
            "independent_evaluator_used": False,
        },
        "value rollout Critic reward policy changed",
    )
    _require(
        protocol.get("guidance_grid")
        == {
            "kappa": [0.0, 0.5, 1.0],
            "temperature": [0.5, 1.0],
            "beta_max": [0.5, 1.0, 2.0],
            "additional_combination_authorized": False,
        },
        "value guidance grid changed",
    )
    authorization = authorize_xeditflow_guidance_v3(
        critic_readiness, setflow_confirmation
    )
    _require(authorization["guidance_authorized"] is True, "value rollout config remains blocked before readiness")
    _require(
        critic_refit_manifest.get("status")
        == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and int(critic_refit_manifest.get("completed_refit_count", -1)) == 3,
        "value rollout config requires three Critic refits",
    )
    selected = str(setflow_confirmation.get("selected_arm"))
    _require(selected in {"f2", "f3"}, "value rollout SetFlow selection differs")
    _require(
        int(setflow_runtime.get("seed", -1)) == 20260904
        and str(setflow_runtime.get("selected_arm")) == selected,
        "value rollout SetFlow runtime identity differs",
    )
    _require(int(physical_gpu_index) in set(range(6)), "value rollout GPU is outside 0-5")
    checkpoint_path = Path(str(setflow_runtime["output_root"])) / selected / "best.pt"
    config = {
        "schema_version": "route_a_v3_route2_xeditflow_value_rollout_config.v1",
        "status": "FROZEN_VALUE_ROLLOUT_CONFIG_NOT_STARTED",
        "critic_readiness_path": str(protocol["critic_readiness_path"]),
        "setflow_confirmation_path": str(protocol["setflow_confirmation_path"]),
        "critic_refit_manifest_path": str(protocol["critic_refit_manifest_path"]),
        "train_projection_path": str(setflow_runtime["train_projection_path"]),
        "source_token_cache_path": str(setflow_runtime["source_token_cache_path"]),
        "setflow_checkpoint_path": str(checkpoint_path),
        "setflow_arm": selected,
        "mrnabert_model_path": str(protocol["mrnabert_model_path"]),
        "expected_train_record_count": 68294,
        "base_flow_training_seed": 20260904,
        "states_per_record": 2,
        "state_pass_index": 0,
        "rollouts_per_state": 8,
        "sampling_state_batch_size": int(protocol["rollout_execution_policy"]["sampling_state_batch_size"]),
        "trajectory_forward_batch_size": int(protocol["rollout_execution_policy"]["trajectory_forward_batch_size"]),
        "critic_batch_size": int(protocol["rollout_execution_policy"]["critic_batch_size"]),
        "critic_online_microbatch_size": int(protocol["rollout_execution_policy"]["critic_online_microbatch_size"]),
        "physical_gpu_index": int(physical_gpu_index),
        "device": f"cuda:{int(physical_gpu_index)}",
        "output_dir": str(protocol["value_rollout_output_dir"]),
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    validate_value_rollout_config_v3(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"value rollout runtime config exists: {args.output}")
    protocol = _json(args.protocol)
    config = build_value_rollout_config_v3(
        protocol,
        _json(Path(protocol["critic_readiness_path"])),
        _json(Path(protocol["setflow_confirmation_path"])),
        _json(Path(protocol["critic_refit_manifest_path"])),
        _json(Path(protocol["setflow_confirmation_runtime_config_path"])),
        physical_gpu_index=args.physical_gpu_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(config, sort_keys=True))


if __name__ == "__main__":
    main()
