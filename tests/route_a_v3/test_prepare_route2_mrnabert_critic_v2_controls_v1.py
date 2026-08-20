from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_controls_v1.py"
PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_prepare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base() -> dict:
    return {
        "baseline_id": "mrnabert-v1",
        "scientific_role": "MRNABERT_PRIMARY",
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
        "expected_trainable_parameter_count": 9_342_914,
        "expected_frozen_pretrained_parameter_count": 113_389_056,
    }


def test_prepares_four_matched_arms_with_distinct_information_boundaries() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    configs = {
        arm: module.prepare_arm(_base(), protocol, arm=arm, gpu=index + 2)
        for index, arm in enumerate(protocol["arms"])
    }
    matched_keys = (
        "seed",
        "hidden_dim",
        "depth",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "loss_kind",
        "training_weighting_mode",
        "training_sampling_mode",
        "loss_aggregation_mode",
        "target_scaling_mode",
        "pretrained_feature_cache_path",
        "expected_trainable_parameter_count",
    )
    for key in matched_keys:
        assert len({config[key] for config in configs.values()}) == 1
    assert configs["candidate_permutation"]["candidate_control"] == (
        "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"
    )
    assert configs["source_only"]["model_kind"].endswith("source_only_control")
    assert configs["source_edit_metadata"]["model_kind"].endswith(
        "source_edit_metadata_control"
    )
    assert all(config["development_test_outcomes_accessed"] is False for config in configs.values())
    assert all(config["evaluation_outcomes_accessed"] is False for config in configs.values())


def test_rejects_unfrozen_or_protected_outcome_input() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    protocol["status"] = "CHANGED_AFTER_OUTCOME"
    with pytest.raises(module.CriticV2PreparationError, match="not prospectively frozen"):
        module.prepare_arm(_base(), protocol, arm="full", gpu=2)
    protocol = json.loads(PROTOCOL.read_text())
    base = _base()
    base["development_test_outcomes_accessed"] = True
    with pytest.raises(module.CriticV2PreparationError, match="accessed TEST"):
        module.prepare_arm(base, protocol, arm="full", gpu=2)


def test_rejects_gpu_outside_user_authorized_zero_to_five() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    with pytest.raises(module.CriticV2PreparationError, match="GPU0-5"):
        module.prepare_arm(_base(), protocol, arm="full", gpu=6)
