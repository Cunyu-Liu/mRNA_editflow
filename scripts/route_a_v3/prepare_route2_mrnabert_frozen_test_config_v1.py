#!/usr/bin/env python3
"""Prepare the single frozen Development TEST run after three-seed confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class FrozenTestConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenTestConfigError(message)


def build_config(
    selected: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    gpu: int,
    output_directory: Path,
) -> dict[str, Any]:
    _require(
        adjudication.get("status") == "THREE_FINAL_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "three final seeds do not authorize Development TEST",
    )
    _require(
        adjudication.get("supports_single_frozen_development_test") is True,
        "single frozen TEST gate failed",
    )
    _require(adjudication.get("development_test_opened") is False, "TEST was already opened")
    _require(adjudication.get("evaluation_opened") is False, "Evaluation entered seed adjudication")
    _require(selected.get("model_kind") == PRIMARY_KIND, "selected config is not primary mRNABERT")
    _require(selected.get("loss_kind") == adjudication.get("loss_kind"), "selected loss differs")
    _require(selected.get("candidate_control") == "NONE", "selected config is a control")
    _require(selected.get("evaluation_outcomes_accessed") is False, "selected config accessed Evaluation")
    _require(0 <= gpu <= 5, "frozen TEST must use physical GPU 0-5")
    seed = int(adjudication["single_frozen_test_seed"])
    config = dict(selected)
    config.update({
        "scientific_role": "SINGLE_FROZEN_DEVELOPMENT_TEST_AFTER_THREE_SEED_CONFIRMATION",
        "result_stage": "FROZEN_DEVELOPMENT_TEST",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "baseline_id": f"mrnabert_edit_centered_{selected['loss_kind']}_frozen_test_seed{seed}",
        "attempt_purpose": "ONE_TIME_FROZEN_DEVELOPMENT_TEST_AFTER_THREE_SEED_CONFIRMATION",
        "seed": seed,
        "device": f"cuda:{gpu}",
        "physical_gpu_index": gpu,
        "candidate_control": "NONE",
        "checkpoint_selection": "FINAL_EPOCH",
        "development_test_outcomes_accessed": True,
        "evaluation_outcomes_accessed": False,
        "output_directory": str(output_directory),
    })
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--three-seed-adjudication", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_directory, args.output_config):
        _require(not path.exists(), f"frozen TEST target already exists: {path}")
    config = build_config(
        json.loads(args.selected_config.read_text(encoding="utf-8")),
        json.loads(args.three_seed_adjudication.read_text(encoding="utf-8")),
        gpu=args.gpu,
        output_directory=args.output_directory,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "SINGLE_FROZEN_DEVELOPMENT_TEST_CONFIG_PREPARED",
        "config": str(args.output_config),
        "seed": config["seed"],
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
