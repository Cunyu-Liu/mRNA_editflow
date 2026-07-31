from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/utr_editflow_execution_policy.yaml").read_text(encoding="utf-8")
    )



def test_single_contract_and_derived_policy_binding():
    goal = ROOT / "docs/contracts/mrna_editflow_contract.md"
    goal_hash = hashlib.sha256(goal.read_bytes()).hexdigest()
    policy = _contract()
    assert policy["execution_policy_id"] == "utr_editflow_execution_policy"
    assert policy["non_authoritative_derivative"] is True
    assert policy["generated_from_contract_path"] == goal.relative_to(ROOT).as_posix()
    assert policy["generated_from_contract_sha256"] == goal_hash
    active_contracts = [path for path in (ROOT / "docs/contracts").glob("*.md") if path.name != "v2_contract_conflict_matrix.md"]
    assert active_contracts == [goal]

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
