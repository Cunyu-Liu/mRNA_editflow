#!/usr/bin/env python3
"""Assemble critic/Flow readiness evidence after every prerequisite exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_LOSO_SEEDS = (20260822, 20260823, 20260824)


class ReadinessInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessInputError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _reward_policy_is_frozen(policy: Mapping[str, Any]) -> bool:
    transform = policy.get("potential_transform")
    return (
        policy.get("schema_version")
        == "route_a_v3_route2_mrnabert_guidance_reward_policy.v1"
        and policy.get("status")
        == "PROSPECTIVELY_FROZEN_BEFORE_GUIDED_GENERATION"
        and policy.get("critic_model_kind")
        == "delta_pretrained_mrnabert_edit_centered_antisymmetric"
        and policy.get("critic_checkpoint_role")
        == "FINAL_ALL_DEVELOPMENT_REFIT_FINAL_EPOCH"
        and policy.get("critic_parameter_update_during_generation") is False
        and policy.get("generator_gradient_into_critic") is False
        and policy.get("evaluation_model_gradient_into_generator") is False
        and policy.get("reward_signal") == "STANDARDIZED_PREDICTED_MEAN_DELTA"
        and policy.get("uncertainty_in_guidance")
        == "DISABLED_DIAGNOSTIC_ONLY"
        and policy.get("target_scale_source")
        == "FINAL_REFIT_CHECKPOINT_TRAIN_TASK_ROBUST_SCALER"
        and isinstance(transform, Mapping)
        and transform.get("kind") == "CLIPPED_IDENTITY"
        and transform.get("minimum") == -5.0
        and transform.get("maximum") == 5.0
        and policy.get("guidance_strength") == 1.0
        and policy.get("guidance_schedule") == "CONSTANT"
        and policy.get("transition_rule")
        == "BASE_TRANSITION_RATE_TIMES_EXP_POTENTIAL_DIFFERENCE"
        and policy.get("action_space") == "SUB_PLUS_STOP"
        and policy.get("generated_candidate_encoder")
        == "ONLINE_FROZEN_MRNABERT_WITH_SEQUENCE_MEMOIZATION"
        and policy.get("evaluation_records_used_for_training_hpo_threshold_or_reward")
        == 0
    )


def build_input(
    *,
    validation_training_summary: Mapping[str, Any],
    final_refit_summary: Mapping[str, Any],
    final_refit_checkpoint: Path,
    signal_adjudication: Mapping[str, Any],
    loso_results: list[Mapping[str, Any]],
    flow_training_summary: Mapping[str, Any],
    flow_validation_summary: Mapping[str, Any],
    reward_policy: Mapping[str, Any],
    online_encoder_validation: Mapping[str, Any],
) -> dict[str, Any]:
    _require(final_refit_checkpoint.is_file(), "final refit checkpoint is absent")
    _require(len(loso_results) == 3, "exactly three LOSO seed results are required")
    _require(
        tuple(sorted(int(row.get("seed", -1)) for row in loso_results))
        == EXPECTED_LOSO_SEEDS,
        "LOSO seed set differs",
    )
    policy_frozen = _reward_policy_is_frozen(reward_policy)
    online_encoder_ready = (
        online_encoder_validation.get("schema_version")
        == "route_a_v3_route2_mrnabert_online_encoder_validation.v1"
        and online_encoder_validation.get("status")
        == "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE"
        and online_encoder_validation.get("novel_candidate_encoding_supported") is True
        and online_encoder_validation.get("frozen_parameter_count", 0) > 100_000_000
        and online_encoder_validation.get("evaluation_records_read") == 0
        and online_encoder_validation.get("maximum_absolute_difference", float("inf"))
        <= online_encoder_validation.get("absolute_tolerance", -1.0)
    )
    return {
        "schema_version": "route_a_v3_route2_readiness_input.v1",
        "critic": {
            "validation_training_summary": dict(validation_training_summary),
            "final_refit_summary": dict(final_refit_summary),
            "final_refit_checkpoint": str(final_refit_checkpoint),
            "development_grouped_split_status":
            "ROUTE2_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED",
            "strongest_baseline_status": "COMPLETED_DEVELOPMENT_ONLY",
            "expected_loso_study_count": 7,
            "loso_seed_results": [dict(row) for row in loso_results],
            "signal_control_adjudication": dict(signal_adjudication),
            "critic_checkpoint_frozen": True,
            "input_schema_frozen": True,
            "context_policy_frozen": True,
            "reward_calibration_policy_frozen": policy_frozen,
            "generated_candidate_online_encoder_ready": online_encoder_ready,
            "online_encoder_validation": dict(online_encoder_validation),
            "reward_policy": dict(reward_policy),
            "evaluation_records_used_for_training_hpo_threshold_or_reward": 0,
        },
        "flow": {
            "training_summary": dict(flow_training_summary),
            "validation_summary": dict(flow_validation_summary),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-training-summary", type=Path, required=True)
    parser.add_argument("--final-refit-summary", type=Path, required=True)
    parser.add_argument("--final-refit-checkpoint", type=Path, required=True)
    parser.add_argument("--signal-adjudication", type=Path, required=True)
    parser.add_argument("--loso-result", type=Path, action="append", required=True)
    parser.add_argument("--flow-training-summary", type=Path, required=True)
    parser.add_argument("--flow-validation-summary", type=Path, required=True)
    parser.add_argument("--reward-policy", type=Path, required=True)
    parser.add_argument("--online-encoder-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"readiness input already exists: {args.output}")
    payload = build_input(
        validation_training_summary=_read_json(
            args.validation_training_summary, "validation training summary"
        ),
        final_refit_summary=_read_json(args.final_refit_summary, "final refit summary"),
        final_refit_checkpoint=args.final_refit_checkpoint,
        signal_adjudication=_read_json(args.signal_adjudication, "signal adjudication"),
        loso_results=[_read_json(path, "LOSO result") for path in args.loso_result],
        flow_training_summary=_read_json(
            args.flow_training_summary, "Flow training summary"
        ),
        flow_validation_summary=_read_json(
            args.flow_validation_summary, "Flow validation summary"
        ),
        reward_policy=_read_json(args.reward_policy, "reward policy"),
        online_encoder_validation=_read_json(
            args.online_encoder_validation, "online encoder validation"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "ROUTE2_GUIDANCE_READINESS_INPUT_BUILT",
        "output": str(args.output),
        "generated_candidate_online_encoder_ready": payload["critic"][
            "generated_candidate_online_encoder_ready"
        ],
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
