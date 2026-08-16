#!/usr/bin/env python3
"""Build frozen-parameter classical LOSO configs for selected Route 2 task winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class ClassicalLosoConfigError(RuntimeError):
    pass


STUDY_GPU = {
    "GSE200304": 0,
    "GSE114002": 1,
    "GSE149487": 2,
    "GSE217518": 3,
    "ENCSR854RUF": 4,
    "GSE186455": 5,
    "GSE269595": 2,
}

FROZEN_PARAMETERS = {
    "context_only_mean": {},
    "gc_mfe_motif_ridge": {"alpha": 0.01},
    "edit_position_only_ridge": {"alpha": 1.0},
    "ref_alt_only_ridge": {"alpha": 1.0},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassicalLosoConfigError(message)


def build_config(base: Mapping[str, Any], study: str, gpu_index: int) -> dict[str, Any]:
    _require(study in STUDY_GPU, f"unexpected LOSO study: {study}")
    _require(gpu_index == STUDY_GPU[study], f"GPU mapping changed for {study}")
    _require(base["evaluation_outcomes_accessed"] is False, "base config accessed Evaluation")
    specs = {str(row["baseline_id"]): dict(row) for row in base["baselines"]}
    _require(set(FROZEN_PARAMETERS) <= set(specs), "selected classical baseline is absent")
    baselines = []
    for baseline_id, parameters in FROZEN_PARAMETERS.items():
        spec = specs[baseline_id]
        spec.pop("parameter_grid", None)
        spec["baseline_id"] = f"classical_{baseline_id}"
        spec["frozen_parameters"] = parameters
        baselines.append(spec)
    label = study.lower()
    return {
        "schema_version": "route_a_v3_route2_classical_prediction_hpo.v1",
        "result_stage": "LOSO_FROZEN_PARAMETERS",
        "run_mode": "LOSO_FROZEN_PARAMETERS",
        "evaluation_outcomes_accessed": False,
        "cpu_thread_cap": int(base["cpu_thread_cap"]),
        "device": f"cuda:{gpu_index}",
        "physical_gpu_index": gpu_index,
        "minimum_free_gpu_memory_bytes": int(base["minimum_free_gpu_memory_bytes"]),
        "development_manifest_path": base["development_manifest_path"],
        "canonical_paths": list(base["canonical_paths"]),
        "loso_holdout_study_unit_id": study,
        "seed": 20260816,
        "baselines": baselines,
        "output_directory": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/"
            f"development_loso_baselines/classical_strongest_{label}_gpu{gpu_index}_v1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for study, gpu_index in STUDY_GPU.items():
        path = args.output_dir / (
            f"route_a_v3_route2_classical_strongest_loso_{study.lower()}_gpu{gpu_index}_v1.json"
        )
        _require(not path.exists(), f"output config already exists: {path}")
        path.write_text(
            json.dumps(build_config(base, study, gpu_index), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(str(path))
    print(json.dumps({"config_count": len(written), "paths": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
