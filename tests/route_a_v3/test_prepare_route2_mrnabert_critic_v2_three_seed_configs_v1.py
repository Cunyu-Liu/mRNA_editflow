from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_three_seed_configs_v1.py"
CONTROL_PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
CONFIRMATION_PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_seed_prepare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base() -> dict:
    return {
        "result_stage": "HPO_VALIDATION_ONLY",
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "candidate_control": "NONE",
        "loss_kind": "huber",
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "development_manifest": "/mnt/development.jsonl",
        "canonical_paths": ["/mnt/a.jsonl"],
        "pretrained_feature_cache_path": "/mnt/mrnabert.pt",
        "hidden_dim": 384,
        "depth": 10,
        "expected_trainable_parameter_count": 9_342_914,
        "expected_frozen_pretrained_parameter_count": 113_389_056,
    }


def _control_adjudication(control_protocol: dict) -> dict:
    baseline = control_protocol["strongest_same_information_baseline"]
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "status": "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "supports_three_frozen_seeds": True,
        "frozen_confirmation_seeds": [20260822, 20260823, 20260824],
        "strongest_same_information_baseline": {
            "baseline_id": baseline["baseline_id"],
            "task_macro_spearman": baseline["task_macro_spearman"],
        },
        "development_test_opened": False,
        "evaluation_opened": False,
    }


def test_builds_exact_three_policy_matched_confirmation_configs() -> None:
    module = _load()
    control = json.loads(CONTROL_PROTOCOL.read_text())
    confirmation = json.loads(CONFIRMATION_PROTOCOL.read_text())
    configs = module.build_configs(
        _base(),
        control,
        confirmation,
        _control_adjudication(control),
        gpu_indices=[2, 4, 5],
    )
    assert [row["seed"] for row in configs] == [20260822, 20260823, 20260824]
    assert [row["physical_gpu_index"] for row in configs] == [2, 4, 5]
    assert all(row["result_stage"] == "FROZEN_DEVELOPMENT_VALIDATION" for row in configs)
    assert all(row["candidate_control"] == "NONE" for row in configs)
    assert all(row["training_sampling_mode"] == "TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS" for row in configs)
    assert all(row["loss_aggregation_mode"] == "TASK_MACRO_MEAN" for row in configs)
    assert all(row["development_test_outcomes_accessed"] is False for row in configs)
    assert all(row["evaluation_outcomes_accessed"] is False for row in configs)


def test_rejects_failed_control_gate_or_changed_seed_set() -> None:
    module = _load()
    control = json.loads(CONTROL_PROTOCOL.read_text())
    confirmation = json.loads(CONFIRMATION_PROTOCOL.read_text())
    adjudication = _control_adjudication(control)
    adjudication["supports_three_frozen_seeds"] = False
    with pytest.raises(module.CriticV2ThreeSeedPreparationError, match="gate failed"):
        module.build_configs(
            _base(), control, confirmation, adjudication, gpu_indices=[2, 4, 5]
        )
    adjudication = _control_adjudication(control)
    confirmation["required_seeds"] = [20260822, 20260823, 20260826]
    with pytest.raises(module.CriticV2ThreeSeedPreparationError, match="seed set differs"):
        module.build_configs(
            _base(), control, confirmation, adjudication, gpu_indices=[2, 4, 5]
        )


def test_rejects_gpu_outside_zero_to_five() -> None:
    module = _load()
    control = json.loads(CONTROL_PROTOCOL.read_text())
    confirmation = json.loads(CONFIRMATION_PROTOCOL.read_text())
    with pytest.raises(module.CriticV2ThreeSeedPreparationError, match="GPU0-5"):
        module.build_configs(
            _base(),
            control,
            confirmation,
            _control_adjudication(control),
            gpu_indices=[2, 4, 6],
        )
