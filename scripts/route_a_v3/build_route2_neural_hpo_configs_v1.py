#!/usr/bin/env python3
"""Materialize the frozen Route 2 architecture-controlled neural HPO matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class NeuralHpoConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NeuralHpoConfigError(message)


def build_trials(
    matrix: Mapping[str, Any],
    capacity: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    _require(matrix["schema_version"] == "route_a_v3_route2_neural_hpo_matrix.v1", "unexpected matrix schema")
    _require(capacity["schema_version"] == "route_a_v3_route2_delta_capacity_profiles.v1", "unexpected capacity schema")
    _require(registry["schema_version"] == "route_a_v3_route2_experiment_registry.v1", "unexpected registry schema")
    frozen = registry["architecture_controlled_neural_baselines"]
    budget = registry["neural_common_budget"]
    profile_ids = list(frozen["capacity_profile_ids"])
    model_kinds = list(frozen["model_kinds"])
    learning_rates = [float(value) for value in budget["learning_rates"]]
    _require(profile_ids == ["MEDIUM_0_5M", "MAIN_2M"], "unexpected neural capacity profiles")
    _require(model_kinds == ["candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"], "unexpected neural model kinds")
    _require(learning_rates == [0.0003, 0.001], "unexpected neural learning rates")
    _require(int(budget["hpo_trial_count_per_architecture_profile"]) == len(learning_rates), "HPO count differs from learning-rate count")
    _require(budget["cuda_required"] is True and budget["cpu_fallback_allowed"] is False, "neural HPO is not fail-closed CUDA")
    gpu_indices = [int(value) for value in matrix["physical_gpu_indices"]]
    _require(gpu_indices and len(gpu_indices) == len(set(gpu_indices)), "GPU list is empty or duplicated")
    _require(all(0 <= value <= 5 for value in gpu_indices), "matrix includes a CUDA index without verified physical identity")
    by_profile = capacity["architecture_controlled_neural_baseline_profiles"]
    common = dict(matrix["common_training_config"])
    _require(common["result_stage"] == "HPO_VALIDATION_ONLY", "matrix is not validation-only")
    _require(common["evaluation_outcomes_accessed"] is False, "matrix accessed Evaluation")
    trials: list[tuple[str, dict[str, Any]]] = []
    index = 0
    for profile_id in profile_ids:
        rows = {str(row["model_kind"]): row for row in by_profile[profile_id]}
        _require(set(rows) == set(model_kinds), f"capacity profile does not cover all models: {profile_id}")
        profile_token = {"MEDIUM_0_5M": "medium_0_5m", "MAIN_2M": "main_2m"}[profile_id]
        for model_kind in model_kinds:
            row = rows[model_kind]
            for learning_rate in learning_rates:
                learning_rate_token = {0.0003: "lr3e4", 0.001: "lr1e3"}[learning_rate]
                gpu = gpu_indices[index % len(gpu_indices)]
                baseline_id = f"neural_{profile_token}_{model_kind}_{learning_rate_token}"
                filename = f"route_a_v3_route2_{baseline_id}_v1.json"
                output_directory = str(Path(matrix["output_root"]) / f"{baseline_id}_v1")
                config = {
                    **common,
                    "baseline_id": baseline_id,
                    "model_kind": model_kind,
                    "hidden_dim": int(row["hidden_dim"]),
                    "depth": int(row["depth"]),
                    "learning_rate": learning_rate,
                    "device": f"cuda:{gpu}",
                    "physical_gpu_index": gpu,
                    "output_directory": output_directory,
                    "frozen_capacity_profile_id": profile_id,
                    "frozen_target_parameter_count": int(row["target_parameter_count"]),
                    "frozen_expected_parameter_count": int(row["actual_parameter_count"]),
                }
                trials.append((filename, config))
                index += 1
    _require(len(trials) == 16, "architecture-controlled matrix is not exactly 16 trials")
    _require(len({name for name, _ in trials}) == len(trials), "generated config filename is duplicated")
    _require(len({trial["output_directory"] for _, trial in trials}) == len(trials), "generated output directory is duplicated")
    return trials


def execute(matrix_path: Path, output_directory: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    capacity = json.loads(Path(matrix["capacity_profile_path"]).read_text(encoding="utf-8"))
    registry = json.loads(Path(matrix["experiment_registry_path"]).read_text(encoding="utf-8"))
    trials = build_trials(matrix, capacity, registry)
    output_directory.mkdir(parents=True, exist_ok=True)
    existing = [output_directory / name for name, _ in trials if (output_directory / name).exists()]
    if existing:
        raise NeuralHpoConfigError(f"generated config already exists: {existing[0]}")
    for filename, config in trials:
        (output_directory / filename).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "ARCHITECTURE_CONTROLLED_NEURAL_HPO_CONFIGS_MATERIALIZED",
        "trial_count": len(trials),
        "capacity_profile_count": len({trial["frozen_capacity_profile_id"] for _, trial in trials}),
        "model_kind_count": len({trial["model_kind"] for _, trial in trials}),
        "learning_rate_count": len({trial["learning_rate"] for _, trial in trials}),
        "evaluation_outcomes_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.matrix, args.output_directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
