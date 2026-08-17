from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_final_seed_configs_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_final_seed_configs_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected() -> dict:
    return {
        "scientific_role": "PRIMARY",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "loss_kind": "huber",
        "seed": 17,
        "device": "cuda:0",
        "physical_gpu_index": 0,
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "development_manifest": "/mnt/development.jsonl",
        "canonical_paths": ["/mnt/a.jsonl"],
        "pretrained_feature_cache_path": "/mnt/mrnabert.pt",
        "hidden_dim": 384,
        "depth": 10,
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    }


def _adjudication() -> dict:
    return {
        "status": "MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION",
        "supports_final_seed_confirmation": True,
        "selected_loss": "huber",
        "development_test_opened": False,
        "evaluation_opened": False,
    }


def test_builds_three_fixed_validation_seeds_without_test_or_evaluation() -> None:
    module = _load()
    configs = module.build_configs(
        _selected(), _adjudication(), run_root=Path("/mnt/final-seeds")
    )
    assert [row["seed"] for row in configs] == [20260822, 20260823, 20260824]
    assert [row["physical_gpu_index"] for row in configs] == [0, 3, 5]
    assert all(row["result_stage"] == "FROZEN_DEVELOPMENT_VALIDATION" for row in configs)
    assert all(row["run_mode"] == "FIXED_GROUPED_SPLIT" for row in configs)
    assert all(row["candidate_control"] == "NONE" for row in configs)
    assert all(row["checkpoint_selection"] == "BEST_VALIDATION" for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)


def test_failed_control_gate_or_contaminated_input_is_rejected() -> None:
    module = _load()
    for target, key in (("gate", "supports_final_seed_confirmation"), ("selected", "development_test_outcomes_accessed")):
        selected = _selected()
        adjudication = _adjudication()
        if target == "gate":
            adjudication[key] = False
        else:
            selected[key] = True
        with pytest.raises(module.FinalSeedConfigError):
            module.build_configs(selected, adjudication, run_root=Path("/mnt/final-seeds"))
