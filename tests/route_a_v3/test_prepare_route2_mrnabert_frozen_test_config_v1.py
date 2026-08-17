from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/prepare_route2_mrnabert_frozen_test_config_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_frozen_test_config_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected() -> dict:
    return {
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "loss_kind": "huber",
        "candidate_control": "NONE",
        "evaluation_outcomes_accessed": False,
        "seed": 17,
        "result_stage": "HPO_VALIDATION_ONLY",
        "checkpoint_selection": "BEST_VALIDATION",
    }


def _adjudication() -> dict:
    return {
        "status": "THREE_FINAL_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST",
        "supports_single_frozen_development_test": True,
        "development_test_opened": False,
        "evaluation_opened": False,
        "loss_kind": "huber",
        "single_frozen_test_seed": 20260823,
    }


def test_test_config_uses_prefrozen_seed_and_final_epoch() -> None:
    module = _load()
    config = module.build_config(
        _selected(), _adjudication(), gpu=0, output_directory=Path("/mnt/frozen-test")
    )
    assert config["seed"] == 20260823
    assert config["result_stage"] == "FROZEN_DEVELOPMENT_TEST"
    assert config["checkpoint_selection"] == "FINAL_EPOCH"
    assert config["development_test_outcomes_accessed"] is True
    assert config["evaluation_outcomes_accessed"] is False


def test_failed_three_seed_gate_is_rejected() -> None:
    module = _load()
    adjudication = _adjudication()
    adjudication["supports_single_frozen_development_test"] = False
    with pytest.raises(module.FrozenTestConfigError):
        module.build_config(
            _selected(), adjudication, gpu=0, output_directory=Path("/mnt/frozen-test")
        )
