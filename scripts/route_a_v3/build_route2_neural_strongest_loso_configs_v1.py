#!/usr/bin/env python3
"""Build the three frozen-hyperparameter GSE269595 siamese-CNN LOSO configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class NeuralLosoConfigError(RuntimeError):
    pass


SEEDS = (20260816, 20260817, 20260818)
HOLDOUT = "GSE269595"
GPU_INDEX = 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NeuralLosoConfigError(message)


def build_config(base: Mapping[str, Any], seed: int) -> dict[str, Any]:
    _require(seed in SEEDS, f"unexpected final seed: {seed}")
    _require(base["evaluation_outcomes_accessed"] is False, "base config accessed Evaluation")
    _require(base["model_kind"] == "siamese_cnn", "base config is not the selected siamese CNN")
    _require(float(base["learning_rate"]) == 0.0003, "selected siamese learning rate changed")
    config = dict(base)
    config.update({
        "baseline_id": "neural_main_siamese_cnn",
        "result_stage": "LOSO_FROZEN_HYPERPARAMETERS",
        "run_mode": "LOSO_FROZEN_HYPERPARAMETERS",
        "device": f"cuda:{GPU_INDEX}",
        "physical_gpu_index": GPU_INDEX,
        "seed": seed,
        "loso_holdout_study_unit_id": HOLDOUT,
        "output_directory": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/"
            f"development_loso_baselines/neural_main_siamese_gse269595_seed{seed}_gpu2_v1"
        ),
    })
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in SEEDS:
        path = args.output_dir / f"route_a_v3_route2_neural_main_siamese_loso_gse269595_seed{seed}_gpu2_v1.json"
        _require(not path.exists(), f"output config already exists: {path}")
        path.write_text(json.dumps(build_config(base, seed), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    print(json.dumps({"config_count": len(paths), "paths": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
