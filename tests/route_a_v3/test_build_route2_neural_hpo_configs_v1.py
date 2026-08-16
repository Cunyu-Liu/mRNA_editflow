from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_neural_hpo_configs_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("neural_hpo_builder_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_matrix_materializes_exact_architecture_budget() -> None:
    module = _load()
    matrix = {
        "schema_version": "route_a_v3_route2_neural_hpo_matrix.v1",
        "physical_gpu_indices": [0, 1, 2, 3, 4, 5],
        "output_root": "/runs",
        "common_training_config": {
            "result_stage": "HPO_VALIDATION_ONLY",
            "evaluation_outcomes_accessed": False,
        },
    }
    rows = {
        profile: [
            {"model_kind": model, "target_parameter_count": target, "hidden_dim": 20, "depth": 2, "actual_parameter_count": target}
            for model in ["candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"]
        ]
        for profile, target in [("MEDIUM_0_5M", 500000), ("MAIN_2M", 2000000)]
    }
    capacity = {
        "schema_version": "route_a_v3_route2_delta_capacity_profiles.v1",
        "architecture_controlled_neural_baseline_profiles": rows,
    }
    registry = {
        "schema_version": "route_a_v3_route2_experiment_registry.v1",
        "architecture_controlled_neural_baselines": {
            "capacity_profile_ids": ["MEDIUM_0_5M", "MAIN_2M"],
            "model_kinds": ["candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"],
        },
        "neural_common_budget": {
            "learning_rates": [0.0003, 0.001],
            "hpo_trial_count_per_architecture_profile": 2,
            "cuda_required": True,
            "cpu_fallback_allowed": False,
        },
    }
    trials = module.build_trials(matrix, capacity, registry)
    assert len(trials) == 16
    assert {trial["frozen_capacity_profile_id"] for _, trial in trials} == {"MEDIUM_0_5M", "MAIN_2M"}
    assert {trial["model_kind"] for _, trial in trials} == {"candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"}
    assert {trial["learning_rate"] for _, trial in trials} == {0.0003, 0.001}
    assert all(trial["result_stage"] == "HPO_VALIDATION_ONLY" for _, trial in trials)
    assert all(trial["evaluation_outcomes_accessed"] is False for _, trial in trials)
    assert all(trial["device"] == f"cuda:{trial['physical_gpu_index']}" for _, trial in trials)
    assert all(0 <= trial["physical_gpu_index"] <= 5 for _, trial in trials)


def test_execute_writes_all_trials_when_none_exist(tmp_path: Path) -> None:
    module = _load()
    capacity_path = tmp_path / "capacity.json"
    registry_path = tmp_path / "registry.json"
    matrix_path = tmp_path / "matrix.json"
    models = ["candidate_cnn", "siamese_cnn", "full_pair_cnn", "small_transformer"]
    capacity_path.write_text(json.dumps({
        "schema_version": "route_a_v3_route2_delta_capacity_profiles.v1",
        "architecture_controlled_neural_baseline_profiles": {
            profile: [
                {"model_kind": model, "target_parameter_count": target, "hidden_dim": 20, "depth": 2, "actual_parameter_count": target}
                for model in models
            ]
            for profile, target in [("MEDIUM_0_5M", 500000), ("MAIN_2M", 2000000)]
        },
    }), encoding="utf-8")
    registry_path.write_text(json.dumps({
        "schema_version": "route_a_v3_route2_experiment_registry.v1",
        "architecture_controlled_neural_baselines": {
            "capacity_profile_ids": ["MEDIUM_0_5M", "MAIN_2M"], "model_kinds": models,
        },
        "neural_common_budget": {
            "learning_rates": [0.0003, 0.001], "hpo_trial_count_per_architecture_profile": 2,
            "cuda_required": True, "cpu_fallback_allowed": False,
        },
    }), encoding="utf-8")
    matrix_path.write_text(json.dumps({
        "schema_version": "route_a_v3_route2_neural_hpo_matrix.v1",
        "capacity_profile_path": str(capacity_path), "experiment_registry_path": str(registry_path),
        "physical_gpu_indices": [0, 1], "output_root": "/runs",
        "common_training_config": {"result_stage": "HPO_VALIDATION_ONLY", "evaluation_outcomes_accessed": False},
    }), encoding="utf-8")
    output = tmp_path / "configs"
    summary = module.execute(matrix_path, output)
    assert summary["trial_count"] == 16
    assert len(list(output.glob("*.json"))) == 16
