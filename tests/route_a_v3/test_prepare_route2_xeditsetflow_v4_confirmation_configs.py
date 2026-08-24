from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.prepare_route2_xeditsetflow_v4_confirmation_configs import (
    build_confirmation_configs_v4,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads(
    (ROOT / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json").read_text(
        encoding="utf-8"
    )
)
PROTOCOL = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def _gate() -> dict:
    return {
        "status": "XEDITSETFLOW_V4_SCREEN_PASS",
        "confirmation_authorized": True,
        "selected_checkpoint_pass": 8,
    }


def test_prepares_exactly_three_full_only_confirmation_configs() -> None:
    configs = build_confirmation_configs_v4(BASE, PROTOCOL, _gate())
    assert [config["training_seed"] for config in configs] == [
        20260912,
        20260913,
        20260914,
    ]
    assert all(config["selected_model"] == "v4_full" for config in configs)
    assert all(config["run_stage"] == "CONFIRMATION" for config in configs)
    assert all(config["additional_seed_authorized"] is False for config in configs)


def test_confirmation_keeps_ten_pass_four_checkpoint_and_no_active_validation() -> None:
    configs = build_confirmation_configs_v4(BASE, PROTOCOL, _gate())
    for config in configs:
        assert config["training"]["pass_count"] == 10
        assert config["training"]["saved_checkpoint_passes"] == [4, 6, 8, 10]
        assert config["training"]["validation_generation_during_training"] is False


def test_confirmation_never_authorizes_test_guidance_or_extra_seed() -> None:
    configs = build_confirmation_configs_v4(BASE, PROTOCOL, _gate())
    assert PROTOCOL["development_test_authorized"] is False
    assert PROTOCOL["guidance_authorized"] is False
    assert PROTOCOL["additional_seed_authorized"] is False
    assert all(config["development_test_outcomes_accessed"] is False for config in configs)


def test_screen_no_go_or_missing_selected_checkpoint_hard_fails() -> None:
    gate = _gate()
    gate["status"] = "XEDITSETFLOW_V4_SCREEN_NO_GO"
    with pytest.raises(RuntimeError):
        build_confirmation_configs_v4(BASE, PROTOCOL, gate)
    gate = _gate()
    gate["selected_checkpoint_pass"] = None
    with pytest.raises(RuntimeError):
        build_confirmation_configs_v4(BASE, PROTOCOL, gate)
