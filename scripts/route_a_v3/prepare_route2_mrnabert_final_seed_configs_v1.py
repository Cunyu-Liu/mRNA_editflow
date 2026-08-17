#!/usr/bin/env python3
"""Prepare three frozen-validation mRNABERT final-seed confirmation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
SEED_GPU_PAIRS = ((20260822, 0), (20260823, 3), (20260824, 5))


class FinalSeedConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalSeedConfigError(message)


def build_configs(
    selected: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    _require(
        adjudication.get("status")
        == "MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION",
        "signal controls do not authorize final-seed confirmation",
    )
    _require(adjudication.get("supports_final_seed_confirmation") is True, "signal-control gate failed")
    _require(adjudication.get("development_test_opened") is False, "Development TEST entered adjudication")
    _require(adjudication.get("evaluation_opened") is False, "Evaluation entered adjudication")
    _require(selected.get("model_kind") == PRIMARY_KIND, "selected config is not the primary mRNABERT critic")
    _require(selected.get("result_stage") == "HPO_VALIDATION_ONLY", "selected config is not HPO validation")
    _require(selected.get("run_mode") == "FIXED_GROUPED_SPLIT", "selected config is not on the fixed split")
    _require(selected.get("candidate_control") == "NONE", "selected config is a control")
    _require(selected.get("evaluation_outcomes_accessed") is False, "selected config accessed Evaluation")
    _require(selected.get("development_test_outcomes_accessed") is False, "selected config accessed TEST")
    _require(selected.get("loss_kind") == adjudication.get("selected_loss"), "selected loss differs")

    configs = []
    for seed, gpu in SEED_GPU_PAIRS:
        config = dict(selected)
        loss = str(selected["loss_kind"])
        config.update({
            "scientific_role": "MRNABERT_FINAL_SEED_FROZEN_DEVELOPMENT_VALIDATION_CONFIRMATION",
            "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
            "run_mode": "FIXED_GROUPED_SPLIT",
            "baseline_id": f"mrnabert_edit_centered_{loss}_final_seed{seed}",
            "attempt_purpose": "MRNABERT_THREE_SEED_DIRECTIONAL_CONFIRMATION_BEFORE_FROZEN_TEST",
            "seed": seed,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "candidate_control": "NONE",
            "checkpoint_selection": "BEST_VALIDATION",
            "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "output_directory": str(
                run_root / f"seed{seed}_gpu{gpu}_{loss}_v1"
            ),
        })
        configs.append(config)
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--signal-adjudication", type=Path, required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_config_dir.exists(), f"config directory already exists: {args.output_config_dir}")
    configs = build_configs(
        json.loads(args.selected_config.read_text(encoding="utf-8")),
        json.loads(args.signal_adjudication.read_text(encoding="utf-8")),
        run_root=args.run_root,
    )
    for config in configs:
        _require(not Path(config["output_directory"]).exists(), f"run already exists: {config['output_directory']}")
    args.output_config_dir.mkdir(parents=True)
    paths = []
    for config in configs:
        path = args.output_config_dir / f"{config['baseline_id']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    print(json.dumps({
        "status": "THREE_FINAL_SEED_CONFIGS_PREPARED_TEST_UNOPENED",
        "config_paths": paths,
        "seeds": [config["seed"] for config in configs],
        "development_test_opened": False,
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
