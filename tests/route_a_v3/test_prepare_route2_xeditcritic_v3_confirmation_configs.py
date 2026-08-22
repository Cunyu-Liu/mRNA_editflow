from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/route_a_v3/prepare_route2_xeditcritic_v3_confirmation_configs.py"
BASE = REPO_ROOT / "configs/route_a_v3_route2_xeditcritic_v3_screen_v1.json"
PROTOCOL = REPO_ROOT / "configs/route_a_v3_route2_xeditcritic_v3_confirmation_protocol_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_xeditcritic_v3_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(selected: str = "C2") -> dict:
    return {
        "status": "XEDITCRITIC_V3_SCREEN_PASS",
        "confirmation_authorized": True,
        "selected_arm": selected,
    }


def test_builds_exact_three_selected_arm_plus_c0_confirmation_configs() -> None:
    module = _module()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    configs = module.build_confirmation_configs(base, protocol, _gate("C3"))
    assert [config["seed"] for config in configs] == [20260831, 20260901, 20260902]
    assert {config["selected_arm"] for config in configs} == {"C3"}
    assert {config["run_stage"] for config in configs} == {"CONFIRMATION"}
    assert all(config["passes"] == 8 for config in configs)
    assert all(config["additional_seed_authorized"] is False for config in configs)
    assert all(config["development_test_outcomes_accessed"] is False for config in configs)
    assert configs[0]["output_root"].endswith("/c3/seed20260831")


def test_config_builder_fails_closed_on_screen_no_go_or_policy_drift() -> None:
    module = _module()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="does not authorize"):
        module.build_confirmation_configs(
            base, protocol,
            {"status": "XEDITCRITIC_V3_SCREEN_NO_GO", "confirmation_authorized": False},
        )
    drifted = dict(base)
    drifted["passes"] = 9
    with pytest.raises(RuntimeError, match="training policy differs"):
        module.build_confirmation_configs(drifted, protocol, _gate())
