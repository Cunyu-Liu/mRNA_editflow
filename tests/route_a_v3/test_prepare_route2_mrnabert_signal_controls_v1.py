from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_signal_controls_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_controls_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected() -> dict:
    return {
        "baseline_id": "winner",
        "scientific_role": "PRIMARY",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "evaluation_outcomes_accessed": False,
        "development_test_outcomes_accessed": False,
        "device": "cuda:0",
        "physical_gpu_index": 0,
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "loss_kind": "huber",
        "seed": 17,
        "hidden_dim": 384,
        "depth": 10,
        "batch_size": 16,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "development_manifest": "/mnt/development.jsonl",
        "canonical_paths": ["/mnt/a.jsonl"],
        "pretrained_feature_cache_path": "/mnt/mrnabert.pt",
        "expected_trainable_parameter_count": 9_000_000,
    }


def test_controls_change_only_role_signal_and_execution_identity() -> None:
    module = _load()
    selected = _selected()
    permutation, source_only = module.prepare(
        selected,
        candidate_permutation_gpu=4,
        source_only_gpu=5,
        candidate_permutation_run_dir=Path("/mnt/permutation"),
        source_only_run_dir=Path("/mnt/source_only"),
    )
    matched_keys = (
        "loss_kind", "seed", "hidden_dim", "depth", "batch_size", "epochs",
        "learning_rate", "weight_decay", "development_manifest", "canonical_paths",
        "pretrained_feature_cache_path", "expected_trainable_parameter_count",
    )
    for key in matched_keys:
        assert permutation[key] == source_only[key] == selected[key]
    assert permutation["candidate_control"] == (
        "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"
    )
    assert permutation["model_kind"] == selected["model_kind"]
    assert source_only["model_kind"] == (
        "delta_pretrained_mrnabert_edit_centered_source_only_control"
    )
    assert source_only["candidate_control"] == "NONE"
    assert permutation["physical_gpu_index"] == 4
    assert source_only["physical_gpu_index"] == 5


def test_refuses_test_evaluation_or_nonprimary_input() -> None:
    module = _load()
    for key, value in (
        ("evaluation_outcomes_accessed", True),
        ("development_test_outcomes_accessed", True),
        ("candidate_control", "ALREADY_A_CONTROL"),
        ("model_kind", "candidate_cnn"),
    ):
        selected = _selected()
        selected[key] = value
        with pytest.raises(module.ControlPreparationError):
            module.prepare(
                selected,
                candidate_permutation_gpu=4,
                source_only_gpu=5,
                candidate_permutation_run_dir=Path("/mnt/permutation"),
                source_only_run_dir=Path("/mnt/source_only"),
            )
