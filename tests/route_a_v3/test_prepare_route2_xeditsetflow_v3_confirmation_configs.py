from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/route_a_v3/prepare_route2_xeditsetflow_v3_confirmation_configs.py"
BASE = REPO_ROOT / "configs/route_a_v3_route2_xeditsetflow_v3_screen_v1.json"
PROTOCOL = REPO_ROOT / "configs/route_a_v3_route2_xeditsetflow_v3_confirmation_protocol_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_xeditsetflow_v3_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(status="XEDITSETFLOW_V3_SCREEN_PASS", arm="f2"):
    return {
        "status": status,
        "confirmation_authorized": status == "XEDITSETFLOW_V3_SCREEN_PASS",
        "selected_arm": arm,
    }


def test_builds_exact_selected_arm_three_seed_setflow_configs() -> None:
    module = _module()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    configs = module.build_confirmation_configs(base, protocol, _gate(arm="f3"))
    assert [config["seed"] for config in configs] == [20260904, 20260905, 20260906]
    assert {config["selected_arm"] for config in configs} == {"f3"}
    assert {config["run_stage"] for config in configs} == {"CONFIRMATION"}
    assert all(config["common_validation_state_seed"] == 2026090301 for config in configs)
    assert all(config["development_test_outcomes_accessed"] is False for config in configs)
    assert configs[0]["validation_output_root"].endswith(
        "/f3/seed20260904/unguided_validation"
    )


def test_setflow_config_builder_rejects_no_go_and_policy_drift() -> None:
    module = _module()
    base = json.loads(BASE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="does not authorize"):
        module.build_confirmation_configs(base, protocol, _gate(status="XEDITSETFLOW_V3_SCREEN_NO_GO"))
    drifted = dict(base)
    drifted["maximum_passes"] = 13
    with pytest.raises(RuntimeError, match="policy differs"):
        module.build_confirmation_configs(drifted, protocol, _gate())
