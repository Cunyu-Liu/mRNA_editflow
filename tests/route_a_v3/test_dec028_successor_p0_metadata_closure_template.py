from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec028_successor_p0_metadata_closure_template_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec028_successor_p0_metadata_closure_template.py"
SPEC = importlib.util.spec_from_file_location("dec028_successor_p0_metadata_closure_template", SCRIPT_PATH)
assert SPEC and SPEC.loader
P0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P0)


def config() -> dict[str, Any]:
    return P0.load_config(CONFIG_PATH)


def test_disk_template_is_static_only_and_reports_current_three_pass_seven_fail_one_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    template = config()
    assert template["document_status"] == "SCHEMA_FIXTURE_AND_FAILURE_CODES_ONLY_NOT_A_PRODUCTION_P0_RECORD"
    assert template["production_p0_authorized"] is False
    assert template["runtime_output_allowed"] is False
    assert template["static_authority"]["current_qualified_counts"] == P0.FROZEN_COUNTS
    assert P0.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    statuses = {item["gate_id"]: item["status"] for item in report["gate_statuses"]}
    assert report["result_status"] == P0.FAILURE_RESULT
    assert {gate for gate, status in statuses.items() if status == "PASS"} == {"P0.3", "P0.6", "P0.11"}
    assert statuses["P0.2"] == "UNKNOWN_NOT_ASSERTED"
    assert sum(status.startswith("FAIL_CLOSED") for status in statuses.values()) == 7
    assert report["materialization_started"] is False
    assert report["g1_launched"] is False
    assert report["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert report["runtime_truth"]["project_rows_read"] == 0
    assert report["runtime_truth"]["cuda_probe_calls"] == 0


def test_any_nonpass_stops_before_data_cuda_or_model_even_when_other_gates_pass() -> None:
    template = config()
    groups = copy.deepcopy(template["initial_template_groups"])
    for group in groups.values():
        group["declared_status"] = "PASS"
    groups["P0.5"]["declared_status"] = "FAIL_CLOSED_COMPLETE_ROW_CONTRACT_NOT_BOUND"
    result = P0.evaluate_aggregate_groups(template, groups)
    assert result["result_status"] == P0.FAILURE_RESULT
    assert result["materialization_started"] is False
    assert result["g1_launched"] is False
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_synthetic_all_pass_is_only_eligible_to_request_materialization_not_g1() -> None:
    template = config()
    groups = copy.deepcopy(template["initial_template_groups"])
    for group in groups.values():
        group["declared_status"] = "PASS"
    result = P0.evaluate_aggregate_groups(template, groups)
    assert result["result_status"] == P0.SUCCESS_RESULT
    assert {item["status"] for item in result["gate_statuses"]} == {"PASS"}
    assert result["materialization_started"] is False
    assert result["g1_launched"] is False
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"


def test_missing_extra_partial_or_unrecognized_groups_fail_closed() -> None:
    template = config()
    groups = copy.deepcopy(template["initial_template_groups"])
    del groups["P0.4"]
    with pytest.raises(P0.ClosureError, match="missing or unexpected"):
        P0.evaluate_aggregate_groups(template, groups)
    groups = copy.deepcopy(template["initial_template_groups"])
    groups["P0.12"] = {"declared_status": "PASS", "failure_code": "EXTRA"}
    with pytest.raises(P0.ClosureError, match="missing or unexpected"):
        P0.evaluate_aggregate_groups(template, groups)
    groups = copy.deepcopy(template["initial_template_groups"])
    groups["P0.7"] = {"declared_status": "PASS"}
    with pytest.raises(P0.ClosureError, match="key closure"):
        P0.evaluate_aggregate_groups(template, groups)
    groups = copy.deepcopy(template["initial_template_groups"])
    groups["P0.8"]["declared_status"] = "PARTIAL"
    with pytest.raises(P0.ClosureError, match="outside vocabulary"):
        P0.evaluate_aggregate_groups(template, groups)


def test_config_rejects_active_authority_count_lock_and_runtime_truth_drift() -> None:
    template = config()
    for keys, value in (
        (("static_authority", "pending_successor_decision_id"), "V3-DEC-029"),
        (("static_authority", "current_qualified_counts", "a1"), 2),
        (("persistent_locks", "materialization_allowed"), True),
        (("runtime_truth", "cuda_probe_calls"), 1),
        (("production_p0_authorized",), True),
    ):
        altered = copy.deepcopy(template)
        cursor: dict[str, Any] = altered
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        with pytest.raises(P0.ClosureError):
            P0.validate_config(altered)


def test_closure_inputs_are_aggregate_owner_responsibilities_without_row_payloads() -> None:
    template = config()
    expected = {"P0.1", "P0.2", "P0.4", "P0.5", "P0.7"}
    assert set(template["owner_metadata_closure_inputs"]) == expected
    for gate_id, item in template["owner_metadata_closure_inputs"].items():
        assert item["row_or_member_payload_allowed"] is False, gate_id
        assert item["responsible_role"]
        assert item["minimum_aggregate_input"]


def test_source_imports_only_static_standard_library_dependencies() -> None:
    module = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for statement in module.body:
        if isinstance(statement, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            imported.add(statement.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "copy", "hashlib", "json", "sys", "pathlib", "typing"}
