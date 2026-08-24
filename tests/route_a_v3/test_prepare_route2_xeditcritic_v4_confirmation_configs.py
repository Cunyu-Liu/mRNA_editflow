from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_a_v3.prepare_route2_xeditcritic_v4_confirmation_configs import (
    build_critic_confirmation_configs_v4,
)


ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _screen_gate() -> dict:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
        "status": "XEDITCRITIC_V4_SCREEN_PASS",
        "passed": True,
        "confirmation_authorized": True,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_confirmation_configs_are_exact_matched_three_seed_package() -> None:
    configs = build_critic_confirmation_configs_v4(
        _json(ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"),
        _json(
            ROOT
            / "configs/route_a_v3_route2_xeditcritic_v4_confirmation_protocol_v1.json"
        ),
        _screen_gate(),
    )
    assert [config["training_seed"] for config in configs] == [
        20260908,
        20260909,
        20260910,
    ]
    assert all(
        config["required_confirmation_run_ids"] == ["v4_full", "c0_v4"]
        for config in configs
    )
    assert [config["bootstrap_seed"] for config in configs] == [
        2026090801,
        2026090901,
        2026091001,
    ]
    assert all(config["development_test_outcomes_accessed"] is False for config in configs)


def test_confirmation_configs_reject_no_go_or_test_authorization() -> None:
    base = _json(ROOT / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json")
    protocol = _json(
        ROOT / "configs/route_a_v3_route2_xeditcritic_v4_confirmation_protocol_v1.json"
    )
    gate = _screen_gate()
    gate["status"] = "XEDITCRITIC_V4_SCREEN_NO_GO"
    with pytest.raises(RuntimeError):
        build_critic_confirmation_configs_v4(base, protocol, gate)
    gate = _screen_gate()
    gate["development_test_authorized"] = True
    with pytest.raises(RuntimeError):
        build_critic_confirmation_configs_v4(base, protocol, gate)
