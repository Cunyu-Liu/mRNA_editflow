#!/usr/bin/env python3
"""Prepare the exact three Critic V2 confirmation configs after control PASS."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTROL_PROTOCOL_SCHEMA = "route_a_v3_route2_mrnabert_critic_v2_protocol.v1"
CONFIRMATION_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol.v1"
)
CONTROL_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1"
)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class CriticV2ThreeSeedPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2ThreeSeedPreparationError(message)


def build_configs(
    base: Mapping[str, Any],
    control_protocol: Mapping[str, Any],
    confirmation_protocol: Mapping[str, Any],
    control_adjudication: Mapping[str, Any],
    *,
    gpu_indices: Sequence[int],
) -> list[dict[str, Any]]:
    _require(
        control_protocol.get("schema_version") == CONTROL_PROTOCOL_SCHEMA,
        "unexpected Critic V2 control protocol",
    )
    _require(
        control_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES",
        "Critic V2 control protocol was not frozen",
    )
    _require(
        confirmation_protocol.get("schema_version")
        == CONFIRMATION_PROTOCOL_SCHEMA,
        "unexpected Critic V2 confirmation protocol",
    )
    _require(
        confirmation_protocol.get("status")
        == "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES",
        "Critic V2 confirmation protocol was not prospectively frozen",
    )
    for payload, label in (
        (control_protocol, "control protocol"),
        (confirmation_protocol, "confirmation protocol"),
    ):
        _require(
            payload.get("development_test_outcomes_accessed") is False,
            f"TEST entered {label}",
        )
        _require(
            payload.get("evaluation_outcomes_accessed") is False,
            f"Evaluation entered {label}",
        )

    _require(
        control_adjudication.get("schema_version") == CONTROL_ADJUDICATION_SCHEMA,
        "unexpected Critic V2 control adjudication",
    )
    _require(
        control_adjudication.get("status")
        == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 controls do not authorize confirmation seeds",
    )
    _require(
        control_adjudication.get("supports_three_frozen_seeds") is True,
        "Critic V2 control gate failed",
    )
    _require(
        control_adjudication.get("development_test_opened") is False,
        "Development TEST entered control adjudication",
    )
    _require(
        control_adjudication.get("evaluation_opened") is False,
        "Evaluation entered control adjudication",
    )

    required_seeds = [int(seed) for seed in confirmation_protocol["required_seeds"]]
    _require(len(required_seeds) == 3, "exactly three confirmation seeds are required")
    _require(
        required_seeds
        == [int(seed) for seed in control_protocol["frozen_confirmation_seeds"]]
        == [int(seed) for seed in control_adjudication["frozen_confirmation_seeds"]],
        "confirmation seed set differs from the frozen control decision",
    )
    _require(len(gpu_indices) == 3, "exactly three GPU indices are required")
    _require(len(set(gpu_indices)) == 3, "confirmation GPUs must be distinct")
    _require(all(0 <= int(gpu) <= 5 for gpu in gpu_indices), "Critic V2 must use physical GPU0-5")

    policy = dict(confirmation_protocol["frozen_training_policy"])
    _require(
        policy == control_protocol["frozen_training_policy"],
        "confirmation policy differs from the frozen control policy",
    )
    _require(policy.get("model_kind") == PRIMARY_KIND, "confirmation is not the full Critic V2 model")
    _require(base.get("model_kind") == PRIMARY_KIND, "base config is not the full mRNABERT critic")
    _require(base.get("loss_kind") == "huber", "base config is not the selected Huber arm")
    _require(base.get("result_stage") == "HPO_VALIDATION_ONLY", "base config is not Validation-only")
    _require(base.get("candidate_control") == "NONE", "base config is a control")
    _require(base.get("development_test_outcomes_accessed") is False, "base config accessed TEST")
    _require(base.get("evaluation_outcomes_accessed") is False, "base config accessed Evaluation")

    frozen_baseline = confirmation_protocol["strongest_same_information_baseline"]
    observed_baseline = control_adjudication["strongest_same_information_baseline"]
    _require(
        frozen_baseline["baseline_id"] == observed_baseline["baseline_id"],
        "strongest baseline identity differs",
    )
    _require(
        math.isclose(
            float(frozen_baseline["task_macro_spearman"]),
            float(observed_baseline["task_macro_spearman"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "strongest baseline value differs",
    )

    run_root = Path(str(confirmation_protocol["run_root"]))
    configs = []
    for seed, gpu in zip(required_seeds, gpu_indices):
        config = dict(base)
        config.update(policy)
        config.update({
            "scientific_role": "CRITIC_V2_THREE_SEED_FROZEN_DEVELOPMENT_VALIDATION_CONFIRMATION",
            "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_critic_v2_full_confirmation_seed{seed}",
            "attempt_purpose": "MRNABERT_CRITIC_V2_THREE_SEED_CONFIRMATION_BEFORE_TEST",
            "seed": seed,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_control": "NONE",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "output_directory": str(run_root / f"seed{seed}"),
            "notes": (
                "Exact prospectively frozen Critic V2 confirmation seed on Development "
                "TRAIN/VALIDATION; TEST and external Evaluation remain closed."
            ),
        })
        configs.append(config)
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--control-adjudication", type=Path, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(
        not args.output_config_dir.exists(),
        f"runtime config directory already exists: {args.output_config_dir}",
    )
    confirmation_protocol = json.loads(
        args.confirmation_protocol.read_text(encoding="utf-8")
    )
    configs = build_configs(
        json.loads(args.base_config.read_text(encoding="utf-8")),
        json.loads(args.control_protocol.read_text(encoding="utf-8")),
        confirmation_protocol,
        json.loads(args.control_adjudication.read_text(encoding="utf-8")),
        gpu_indices=args.gpu,
    )
    for config in configs:
        _require(
            not Path(config["output_directory"]).exists(),
            f"Critic V2 confirmation run already exists: {config['output_directory']}",
        )
    args.output_config_dir.mkdir(parents=True)
    paths = []
    for config in configs:
        path = args.output_config_dir / f"seed{config['seed']}.json"
        path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    print(json.dumps({
        "status": "CRITIC_V2_THREE_SEED_CONFIGS_PREPARED_TEST_UNOPENED",
        "config_paths": paths,
        "seeds": [config["seed"] for config in configs],
        "physical_gpu_indices": [config["physical_gpu_index"] for config in configs],
        "development_test_opened": False,
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
