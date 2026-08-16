from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/route_a_v3/build_route2_neural_strongest_loso_configs_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_neural_loso_config_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_changes_only_loso_identity_seed_device_and_output() -> None:
    module = _load()
    base = {
        "schema_version": "route_a_v3_route2_neural_baseline_hpo.v1",
        "baseline_id": "neural_main_2m_siamese_cnn_lr3e4",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "evaluation_outcomes_accessed": False,
        "device": "cuda:4",
        "physical_gpu_index": 4,
        "seed": 20260816,
        "model_kind": "siamese_cnn",
        "learning_rate": 0.0003,
        "hidden_dim": 173,
        "depth": 12,
        "output_directory": "/old",
    }
    config = module.build_config(base, 20260818)
    assert config["baseline_id"] == "neural_main_siamese_cnn"
    assert config["run_mode"] == config["result_stage"] == "LOSO_FROZEN_HYPERPARAMETERS"
    assert config["loso_holdout_study_unit_id"] == "GSE269595"
    assert config["device"] == "cuda:2"
    assert config["physical_gpu_index"] == 2
    assert config["seed"] == 20260818
    assert config["evaluation_outcomes_accessed"] is False
    assert config["model_kind"] == base["model_kind"]
    assert config["learning_rate"] == base["learning_rate"]
    assert config["hidden_dim"] == base["hidden_dim"]
    assert config["depth"] == base["depth"]
