#!/usr/bin/env python3
"""Prepare matched global-scaled baseline LOSO folds without Development TEST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_loso_schedule import HOLDOUT_STUDIES, loso_assignments

BASELINE_ID = "method_repair_global_scaled_seed20260821"
MODEL_KIND = "delta_anchored_position_aware_antisymmetric"


class GlobalScaledLosoConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GlobalScaledLosoConfigError(message)


def build_configs(
    base: Mapping[str, Any],
    three_seed_adjudication: Mapping[str, Any],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    _require(
        three_seed_adjudication.get("supports_single_frozen_development_test") is True,
        "primary three-seed gate failed",
    )
    _require(base.get("baseline_id") == BASELINE_ID, "strongest same-information baseline differs")
    _require(base.get("model_kind") == MODEL_KIND, "baseline model kind differs")
    _require(base.get("result_stage") == "HPO_VALIDATION_ONLY", "baseline is not HPO-only")
    _require(base.get("run_mode") == "FIXED_GROUPED_SPLIT", "baseline split differs")
    _require(base.get("candidate_control") == "NONE", "baseline is a control")
    _require(base.get("development_test_outcomes_accessed") is False, "TEST entered baseline config")
    _require(base.get("evaluation_outcomes_accessed") is False, "Evaluation entered baseline config")

    configs = []
    for study, seed, gpu in loso_assignments():
        study_label = study.lower().replace("-", "_")
        config = dict(base)
        config.update({
            "scientific_role": "STRONGEST_SAME_INFORMATION_BASELINE_TEST_PRESERVING_LOSO",
            "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
            "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
            "baseline_id": f"global_scaled_loso_{study_label}_seed{seed}",
            "attempt_purpose": "MATCHED_BASELINE_TEST_PRESERVING_LOSO_FOR_GUIDANCE_READINESS",
            "seed": seed,
            "device": f"cuda:{gpu}",
            "physical_gpu_index": gpu,
            "checkpoint_selection": "FINAL_EPOCH",
            "development_test_outcomes_accessed": False,
            "evaluation_outcomes_accessed": False,
            "loso_holdout_study_unit_id": study,
            "output_directory": str(
                run_root / study / f"seed{seed}_gpu{gpu}_global_scaled_v1"
            ),
        })
        configs.append(config)
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--three-seed-adjudication", type=Path, required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_config_dir.exists(), f"config directory already exists: {args.output_config_dir}")
    configs = build_configs(
        json.loads(args.base_config.read_text(encoding="utf-8")),
        json.loads(args.three_seed_adjudication.read_text(encoding="utf-8")),
        run_root=args.run_root,
    )
    _require(len(configs) == 21, "baseline LOSO config count differs")
    for config in configs:
        _require(not Path(config["output_directory"]).exists(), f"run already exists: {config['output_directory']}")
    args.output_config_dir.mkdir(parents=True)
    paths = []
    for config in configs:
        path = args.output_config_dir / f"{config['baseline_id']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    print(json.dumps({
        "status": "MATCHED_BASELINE_TEST_PRESERVING_LOSO_CONFIGS_PREPARED",
        "config_count": len(paths),
        "development_test_opened": False,
        "evaluation_opened": False,
        "paths": paths,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
