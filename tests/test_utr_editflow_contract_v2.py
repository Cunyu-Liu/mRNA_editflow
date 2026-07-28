from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GOAL_SHA = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"


def _contract() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/utr_editflow_contract_v2.yaml").read_text(encoding="utf-8")
    )


def test_goal_snapshot_is_exact_and_is_highest_authority():
    goal = ROOT / "docs/contracts/mrna_latest_build_contract_v2.md"
    assert hashlib.sha256(goal.read_bytes()).hexdigest() == GOAL_SHA
    contract = _contract()
    assert contract["contract_id"] == "utr_editflow_goal_v2"
    assert contract["goal_document"]["sha256"] == GOAL_SHA
    assert contract["authority_order"][0] == goal.relative_to(ROOT).as_posix()


def test_editflow_is_mandatory_and_predictor_is_support_only():
    method = _contract()["method"]
    assert method["primary"] == "mrna_editflow"
    assert method["edit_flow_required"] is True
    assert method["continuous_time_required"] is True
    assert method["source_conditioning_required"] is True
    assert method["predictor_role"] == "support_only"
    assert method["predictor_only_fallback_allowed"] is False
    assert method["flow_optional"] is False
    assert method["action_types"] == ["INS", "SUB", "DEL", "STOP"]


def test_c0_d0_cannot_start_formal_training():
    contract = _contract()
    assert contract["current_scope"]["formal_training_in_c0_d0"] == "forbidden"
    assert contract["phases"]["C0"]["formal_training_allowed"] is False
    assert contract["phases"]["D0"]["final_label_selection_allowed"] is False
