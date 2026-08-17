#!/usr/bin/env python3
"""Prepare three-seed, seven-study mRNABERT LOSO without opening Development TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
HOLDOUT_STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE269595",
)
SEED_GPU_PAIRS = ((20260822, 0), (20260823, 3), (20260824, 5))


class TestPreservingLosoConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TestPreservingLosoConfigError(message)


def build_configs(
    selected: Mapping[str, Any],
    three_seed_adjudication: Mapping[str, Any],
    *,
    run_root: Path,
) -> list[dict[str, Any]]:
    _require(
        three_seed_adjudication.get("status")
        == "THREE_FINAL_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "three final seeds did not pass",
    )
    _require(
        three_seed_adjudication.get("supports_single_frozen_development_test") is True,
        "three-seed directional gate failed",
    )
    _require(selected.get("model_kind") == PRIMARY_KIND, "selected model is not mRNABERT")
    _require(selected.get("result_stage") == "HPO_VALIDATION_ONLY", "selected config is not HPO-only")
    _require(selected.get("run_mode") == "FIXED_GROUPED_SPLIT", "selected config split differs")
    _require(selected.get("candidate_control") == "NONE", "selected config is a control")
    _require(selected.get("development_test_outcomes_accessed") is False, "TEST entered selected config")
    _require(selected.get("evaluation_outcomes_accessed") is False, "Evaluation entered selected config")
    _require(
        selected.get("loss_kind") == three_seed_adjudication.get("loss_kind"),
        "selected loss differs",
    )

    configs = []
    loss = str(selected["loss_kind"])
    for study in HOLDOUT_STUDIES:
        study_label = study.lower().replace("-", "_")
        for seed, gpu in SEED_GPU_PAIRS:
            config = dict(selected)
            config.update({
                "scientific_role": "MRNABERT_TEST_PRESERVING_CROSS_STUDY_TRANSFER",
                "result_stage": "LOSO_DEVELOPMENT_VALIDATION_ONLY_FROZEN_HYPERPARAMETERS",
                "run_mode": "LOSO_DEVELOPMENT_TRAIN_VALIDATION_ONLY",
                "baseline_id": f"mrnabert_{loss}_loso_{study_label}_seed{seed}",
                "attempt_purpose": "THREE_SEED_TEST_PRESERVING_LOSO_FOR_GUIDANCE_READINESS",
                "seed": seed,
                "device": f"cuda:{gpu}",
                "physical_gpu_index": gpu,
                "candidate_control": "NONE",
                "checkpoint_selection": "FINAL_EPOCH",
                "development_test_outcomes_accessed": False,
                "evaluation_outcomes_accessed": False,
                "loso_holdout_study_unit_id": study,
                "output_directory": str(
                    run_root / study / f"seed{seed}_gpu{gpu}_{loss}_v1"
                ),
            })
            configs.append(config)
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--three-seed-adjudication", type=Path, required=True)
    parser.add_argument("--output-config-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_config_dir.exists(), f"config directory already exists: {args.output_config_dir}")
    configs = build_configs(
        json.loads(args.selected_config.read_text(encoding="utf-8")),
        json.loads(args.three_seed_adjudication.read_text(encoding="utf-8")),
        run_root=args.run_root,
    )
    _require(len(configs) == 21, "LOSO config count differs")
    for config in configs:
        _require(not Path(config["output_directory"]).exists(), f"run already exists: {config['output_directory']}")
    args.output_config_dir.mkdir(parents=True)
    paths = []
    for config in configs:
        path = args.output_config_dir / f"{config['baseline_id']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    print(json.dumps({
        "status": "TEST_PRESERVING_LOSO_CONFIGS_PREPARED",
        "config_count": len(paths),
        "holdout_studies": list(HOLDOUT_STUDIES),
        "seeds": [seed for seed, _gpu in SEED_GPU_PAIRS],
        "development_test_opened": False,
        "evaluation_opened": False,
        "paths": paths,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
