#!/usr/bin/env python3
"""Build the Critic V2/Flow readiness packet after every prerequisite exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMAS = {
    "readiness": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol.v1",
    "control": "route_a_v3_route2_mrnabert_critic_v2_protocol.v1",
    "three_seed": "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol.v1",
    "frozen_test": "route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol.v1",
    "refit": "route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol.v1",
    "primary_loso": "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol.v1",
    "baseline_loso": "route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol.v1",
}
FROZEN_STATUS = "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES"
REQUIRED_SEEDS = (20260822, 20260823, 20260824)


class CriticV2ReadinessInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2ReadinessInputError(message)


def _validate_protocols(protocols: Mapping[str, Mapping[str, Any]]) -> None:
    _require(set(protocols) == set(SCHEMAS), "readiness protocol set differs")
    for name, schema in SCHEMAS.items():
        protocol = protocols[name]
        _require(
            protocol.get("schema_version") == schema,
            f"unexpected {name} protocol schema",
        )
        expected_status = (
            "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES"
            if name == "control"
            else FROZEN_STATUS
        )
        _require(
            protocol.get("status") == expected_status,
            f"{name} protocol was not prospectively frozen",
        )
        _require(
            protocol.get("evaluation_outcomes_accessed") is False,
            f"Evaluation entered {name} protocol",
        )
        _require(
            protocol.get("guided_generation_authorized") is False,
            f"guided generation entered {name} protocol",
        )

    readiness = protocols["readiness"]
    expected_bindings = {
        "control_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json",
        "three_seed_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json",
        "frozen_test_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json",
        "all_development_refit_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json",
        "primary_loso_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json",
        "matched_baseline_loso_protocol": "configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json",
        "reward_policy": "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json",
    }
    for key, expected in expected_bindings.items():
        _require(readiness.get(key) == expected, f"readiness binding differs: {key}")
    _require(
        tuple(int(seed) for seed in readiness["required_seeds"])
        == REQUIRED_SEEDS,
        "readiness seed set differs",
    )
    _require(
        int(readiness["required_loso_studies"]) == 7
        and readiness.get("zero_record_development_studies") == ["GSE256185"],
        "readiness LOSO study inventory differs",
    )
    _require(
        int(readiness["single_frozen_test_seed"]) == 20260823,
        "readiness single TEST seed differs",
    )
    _require(
        readiness.get("single_test_metric_policy")
        == "REPORT_ONLY_NO_STRUCTURE_LOSS_SEED_EPOCH_THRESHOLD_OR_POLICY_SELECTION",
        "readiness single TEST metric policy differs",
    )
    _require(
        readiness.get("development_test_outcomes_accessed_at_protocol_freeze")
        is False,
        "Development TEST entered before readiness protocol freeze",
    )


def _validate_terminal_shapes(
    *,
    control_adjudication: Mapping[str, Any],
    three_seed_adjudication: Mapping[str, Any],
    frozen_test_config: Mapping[str, Any],
    frozen_test_summary: Mapping[str, Any],
    refit_config: Mapping[str, Any],
    refit_summary: Mapping[str, Any],
    loso_results: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        control_adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "unexpected Critic V2 control adjudication",
    )
    _require(
        control_adjudication.get("status")
        == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 control adjudication did not pass",
    )
    _require(
        three_seed_adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1",
        "unexpected Critic V2 three-seed adjudication",
    )
    _require(
        three_seed_adjudication.get("status")
        == "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "Critic V2 three-seed adjudication did not pass",
    )
    _require(
        frozen_test_config.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and int(frozen_test_config.get("seed", -1)) == 20260823,
        "frozen Development TEST config identity differs",
    )
    _require(
        frozen_test_summary.get("status")
        == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and frozen_test_summary.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and frozen_test_summary.get("development_test_outcomes_evaluated") is True
        and isinstance(frozen_test_summary.get("test_metrics"), Mapping),
        "frozen Development TEST summary is incomplete",
    )
    _require(
        refit_config.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT"
        and refit_config.get("development_record_scope") == "ALL_126165",
        "all-Development refit config identity differs",
    )
    _require(
        refit_summary.get("status")
        == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and refit_summary.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT"
        and refit_summary.get("record_counts") == {"TRAIN": 126165},
        "all-Development refit summary is incomplete",
    )
    _require(len(loso_results) == 3, "exactly three LOSO aggregate results are required")
    by_seed = {int(row.get("seed", -1)): row for row in loso_results}
    _require(
        len(by_seed) == 3 and set(by_seed) == set(REQUIRED_SEEDS),
        "LOSO aggregate seed set differs",
    )
    for seed in REQUIRED_SEEDS:
        row = by_seed[seed]
        _require(
            row.get("schema_version") == "route_a_v3_route2_loso_aggregation.v1"
            and row.get("status") == "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE"
            and row.get("study_count") == 7
            and row.get("aligned_study_count") == 7
            and row.get("undefined_study_count") == 0,
            f"LOSO seed {seed} aggregate is incomplete",
        )


def build_input(
    *,
    protocols: Mapping[str, Mapping[str, Any]],
    control_adjudication: Mapping[str, Any],
    three_seed_adjudication: Mapping[str, Any],
    frozen_test_config: Mapping[str, Any],
    frozen_test_summary: Mapping[str, Any],
    refit_config: Mapping[str, Any],
    refit_summary: Mapping[str, Any],
    refit_checkpoint: Path,
    loso_results: Sequence[Mapping[str, Any]],
    reward_policy: Mapping[str, Any],
    online_encoder_validation: Mapping[str, Any],
    flow_training_summary: Mapping[str, Any],
    flow_validation_summary: Mapping[str, Any],
    flow_checkpoint: Path,
) -> dict[str, Any]:
    _validate_protocols(protocols)
    _validate_terminal_shapes(
        control_adjudication=control_adjudication,
        three_seed_adjudication=three_seed_adjudication,
        frozen_test_config=frozen_test_config,
        frozen_test_summary=frozen_test_summary,
        refit_config=refit_config,
        refit_summary=refit_summary,
        loso_results=loso_results,
    )
    _require(refit_checkpoint.is_file(), "final Critic V2 refit checkpoint is absent")
    _require(flow_checkpoint.is_file(), "Base Flow checkpoint is absent")
    _require(
        reward_policy.get("schema_version")
        == "route_a_v3_route2_mrnabert_guidance_reward_policy.v1",
        "unexpected guidance reward policy",
    )
    _require(
        online_encoder_validation.get("schema_version")
        == "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
        "unexpected online encoder validation",
    )
    _require(isinstance(flow_training_summary, Mapping), "Flow training summary is malformed")
    _require(isinstance(flow_validation_summary, Mapping), "Flow validation summary is malformed")

    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_input.v1",
        "protocols": {name: dict(value) for name, value in protocols.items()},
        "critic": {
            "control_adjudication": dict(control_adjudication),
            "three_seed_adjudication": dict(three_seed_adjudication),
            "frozen_test_config": dict(frozen_test_config),
            "frozen_test_summary": dict(frozen_test_summary),
            "refit_config": dict(refit_config),
            "refit_summary": dict(refit_summary),
            "refit_checkpoint": str(refit_checkpoint),
            "loso_seed_results": [dict(row) for row in loso_results],
            "reward_policy": dict(reward_policy),
            "online_encoder_validation": dict(online_encoder_validation),
        },
        "flow": {
            "training_summary": dict(flow_training_summary),
            "validation_summary": dict(flow_validation_summary),
            "checkpoint": str(flow_checkpoint),
        },
        "guided_generation_executed": False,
        "evaluation_opened_by_readiness_builder": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def write_input_once(payload: Mapping[str, Any], output: Path) -> None:
    _require(not output.exists(), f"Critic V2 readiness input already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SCHEMAS:
        parser.add_argument(f"--{name.replace('_', '-')}-protocol", type=Path, required=True)
    parser.add_argument("--control-adjudication", type=Path, required=True)
    parser.add_argument("--three-seed-adjudication", type=Path, required=True)
    parser.add_argument("--frozen-test-config", type=Path, required=True)
    parser.add_argument("--frozen-test-summary", type=Path, required=True)
    parser.add_argument("--refit-config", type=Path, required=True)
    parser.add_argument("--refit-summary", type=Path, required=True)
    parser.add_argument("--refit-checkpoint", type=Path, required=True)
    parser.add_argument("--loso-result", type=Path, action="append", required=True)
    parser.add_argument("--reward-policy", type=Path, required=True)
    parser.add_argument("--online-encoder-validation", type=Path, required=True)
    parser.add_argument("--flow-training-summary", type=Path, required=True)
    parser.add_argument("--flow-validation-summary", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    protocols = {
        name: _read_json(getattr(args, f"{name}_protocol")) for name in SCHEMAS
    }
    payload = build_input(
        protocols=protocols,
        control_adjudication=_read_json(args.control_adjudication),
        three_seed_adjudication=_read_json(args.three_seed_adjudication),
        frozen_test_config=_read_json(args.frozen_test_config),
        frozen_test_summary=_read_json(args.frozen_test_summary),
        refit_config=_read_json(args.refit_config),
        refit_summary=_read_json(args.refit_summary),
        refit_checkpoint=args.refit_checkpoint,
        loso_results=[_read_json(path) for path in args.loso_result],
        reward_policy=_read_json(args.reward_policy),
        online_encoder_validation=_read_json(args.online_encoder_validation),
        flow_training_summary=_read_json(args.flow_training_summary),
        flow_validation_summary=_read_json(args.flow_validation_summary),
        flow_checkpoint=args.flow_checkpoint,
    )
    output = Path(str(protocols["readiness"]["readiness_input_output"]))
    write_input_once(payload, output)
    print(
        json.dumps(
            {
                "status": "CRITIC_V2_GUIDANCE_READINESS_INPUT_BUILT_NOT_ADJUDICATED",
                "output": str(output),
                "loso_seeds": list(REQUIRED_SEEDS),
                "guided_generation_executed": False,
                "evaluation_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
