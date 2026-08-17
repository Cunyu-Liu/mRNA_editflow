#!/usr/bin/env python3
"""Prepare the final all-Development critic refit after the one frozen TEST run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"


class FinalRefitConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalRefitConfigError(message)


def build_config(
    frozen_test_config: Mapping[str, Any],
    frozen_test_summary: Mapping[str, Any],
    *,
    gpu: int,
    output_directory: Path,
) -> dict[str, Any]:
    _require(
        frozen_test_config.get("result_stage") == "FROZEN_DEVELOPMENT_TEST",
        "input config is not the frozen Development TEST run",
    )
    _require(
        frozen_test_summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "frozen Development TEST run is incomplete",
    )
    _require(
        frozen_test_summary.get("result_stage") == "FROZEN_DEVELOPMENT_TEST",
        "summary is not the frozen Development TEST result",
    )
    _require(
        frozen_test_summary.get("development_test_outcomes_evaluated") is True,
        "Development TEST was not evaluated",
    )
    _require(
        isinstance(frozen_test_summary.get("test_metrics"), Mapping),
        "frozen Development TEST metrics are missing",
    )
    _require(
        frozen_test_summary.get("evaluation_outcomes_read") == 0,
        "Evaluation pool entered the frozen Development TEST run",
    )
    _require(
        frozen_test_config.get("evaluation_outcomes_accessed") is False,
        "Evaluation pool entered the frozen Development TEST config",
    )
    _require(frozen_test_config.get("model_kind") == PRIMARY_KIND, "model kind differs")
    _require(frozen_test_summary.get("model_kind") == PRIMARY_KIND, "summary model kind differs")
    _require(
        frozen_test_config.get("loss_kind") == frozen_test_summary.get("loss_kind"),
        "frozen Development TEST loss differs",
    )
    _require(
        int(frozen_test_config.get("seed")) == int(frozen_test_summary.get("seed")),
        "frozen Development TEST seed differs",
    )
    _require(frozen_test_summary.get("candidate_control") == "NONE", "TEST run is a control")
    _require(
        frozen_test_summary.get("checkpoint_selection") == "FINAL_EPOCH",
        "TEST run selected a checkpoint after observing TEST",
    )
    _require(0 <= gpu <= 5, "final refit must use physical GPU 0-5")

    seed = int(frozen_test_summary["seed"])
    config = dict(frozen_test_config)
    config.update({
        "scientific_role": "FINAL_CRITIC_REFIT_ON_ALL_DEVELOPMENT_AFTER_SINGLE_FROZEN_TEST",
        "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "baseline_id": f"mrnabert_edit_centered_{config['loss_kind']}_all126165_seed{seed}",
        "attempt_purpose": "FINAL_ALL_126165_DEVELOPMENT_RECORD_REFIT_NO_MODEL_SELECTION",
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
    parser.add_argument("--frozen-test-config", type=Path, required=True)
    parser.add_argument("--frozen-test-summary", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_directory, args.output_config):
        _require(not path.exists(), f"final refit target already exists: {path}")
    config = build_config(
        json.loads(args.frozen_test_config.read_text(encoding="utf-8")),
        json.loads(args.frozen_test_summary.read_text(encoding="utf-8")),
        gpu=args.gpu,
        output_directory=args.output_directory,
    )
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "FINAL_ALL_DEVELOPMENT_REFIT_CONFIG_PREPARED",
        "config": str(args.output_config),
        "seed": config["seed"],
        "development_record_scope": "ALL_126165",
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
