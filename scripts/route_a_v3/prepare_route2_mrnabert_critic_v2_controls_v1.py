#!/usr/bin/env python3
"""Prepare one prospectively frozen Critic V2 control-screen arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class CriticV2PreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2PreparationError(message)


def prepare_arm(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    arm: str,
    gpu: int,
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_protocol.v1",
        "unexpected Critic V2 protocol",
    )
    _require(
        protocol.get("status") == "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES",
        "Critic V2 protocol was not prospectively frozen",
    )
    _require(protocol.get("development_test_outcomes_accessed") is False, "TEST entered protocol")
    _require(protocol.get("evaluation_outcomes_accessed") is False, "Evaluation entered protocol")
    _require(0 <= gpu <= 5, "Critic V2 must use physical GPU0-5")
    arms = protocol.get("arms")
    _require(isinstance(arms, Mapping) and arm in arms, "unknown Critic V2 arm")
    _require(
        base.get("model_kind")
        == "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "base config is not the frozen mRNABERT critic",
    )
    _require(base.get("loss_kind") == "huber", "base config is not the selected Huber arm")
    _require(base.get("result_stage") == "HPO_VALIDATION_ONLY", "base config is not Validation-only")
    _require(base.get("development_test_outcomes_accessed") is False, "base config accessed TEST")
    _require(base.get("evaluation_outcomes_accessed") is False, "base config accessed Evaluation")

    policy = dict(protocol["frozen_training_policy"])
    arm_spec = dict(arms[arm])
    seed = int(protocol["screen_seed"])
    run_root = Path(str(protocol["run_root"]))
    result = dict(base)
    result.update(policy)
    result.update(arm_spec)
    result.update({
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "seed": seed,
        "device": f"cuda:{gpu}",
        "physical_gpu_index": gpu,
        "baseline_id": f"mrnabert_critic_v2_{arm}_seed{seed}",
        "attempt_purpose": "MRNABERT_CRITIC_V2_TASK_STUDY_BALANCED_CONTROL_SCREEN",
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "output_directory": str(run_root / arm),
        "notes": (
            "Prospectively frozen Critic V2 Development TRAIN/VALIDATION arm; "
            "TEST and external Evaluation outcomes remain closed."
        ),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_config.exists(), f"runtime config already exists: {args.output_config}")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = prepare_arm(base, protocol, arm=args.arm, gpu=args.gpu)
    _require(
        not Path(result["output_directory"]).exists(),
        f"Critic V2 run already exists: {result['output_directory']}",
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "CRITIC_V2_ARM_PREPARED",
        "arm": args.arm,
        "gpu": args.gpu,
        "config": str(args.output_config),
        "output_directory": result["output_directory"],
        "development_test_opened": False,
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
