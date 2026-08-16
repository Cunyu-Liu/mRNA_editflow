#!/usr/bin/env python3
"""Build task-specific strongest-baseline assembly configs for all Route 2 LOSO folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class LosoAssemblyConfigError(RuntimeError):
    pass


SEEDS = (20260816, 20260817, 20260818)
STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE269595",
)
CLASSICAL_GPU = {
    "GSE200304": 0,
    "GSE149487": 2,
    "GSE217518": 3,
    "ENCSR854RUF": 4,
    "GSE186455": 5,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LosoAssemblyConfigError(message)


def prediction_specs(study: str, seed: int) -> list[dict[str, str]]:
    _require(study in STUDIES, f"unexpected LOSO study: {study}")
    _require(seed in SEEDS, f"unexpected LOSO seed: {seed}")
    root = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/development_loso_baselines"
    if study == "GSE114002":
        return [{
            "baseline_id": "external_framepool",
            "prediction_path": f"{root}/framepool_gse114002_gpu2_v1/loso_predictions.jsonl",
        }]
    if study == "GSE269595":
        return [{
            "baseline_id": "neural_main_siamese_cnn",
            "prediction_path": (
                f"{root}/neural_main_siamese_gse269595_seed{seed}_gpu2_v1/test_predictions.jsonl"
            ),
        }]
    gpu = CLASSICAL_GPU[study]
    label = study.lower()
    directory = f"{root}/classical_strongest_{label}_gpu{gpu}_v1"
    selected = {
        "GSE200304": ("classical_gc_mfe_motif_ridge",),
        "GSE149487": ("classical_edit_position_only_ridge", "classical_ref_alt_only_ridge"),
        "GSE217518": ("classical_context_only_mean",),
        "ENCSR854RUF": ("classical_context_only_mean",),
        "GSE186455": ("classical_context_only_mean",),
    }[study]
    return [
        {
            "baseline_id": baseline_id,
            "prediction_path": f"{directory}/{baseline_id}/loso_predictions.jsonl",
        }
        for baseline_id in selected
    ]


def build_config(base: Mapping[str, Any], study: str, seed: int) -> dict[str, Any]:
    _require(base["evaluation_outcomes_accessed"] is False, "validation assembly accessed Evaluation")
    _require(base["requested_split"] == "VALIDATION", "base assembly is not Development validation")
    return {
        "schema_version": "route_a_v3_route2_strongest_prediction_assembly_config.v1",
        "evaluation_outcomes_accessed": False,
        "requested_split": f"LOSO::{study}",
        "development_manifest_path": base["development_manifest_path"],
        "canonical_paths": list(base["canonical_paths"]),
        "strongest_selection_path": base["strongest_selection_path"],
        "baseline_predictions": prediction_specs(study, seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-assembly-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.validation_assembly_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in SEEDS:
        for study in STUDIES:
            path = args.output_dir / (
                f"route_a_v3_route2_strongest_loso_assembly_{study.lower()}_seed{seed}_v1.json"
            )
            _require(not path.exists(), f"output config already exists: {path}")
            path.write_text(
                json.dumps(build_config(base, study, seed), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths.append(str(path))
    print(json.dumps({"config_count": len(paths), "paths": paths}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
