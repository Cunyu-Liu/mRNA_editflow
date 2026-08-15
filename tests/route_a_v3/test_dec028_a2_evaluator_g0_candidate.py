from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec028_a2_evaluator_g0_candidate_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec028_a2_evaluator_g0_candidate.py"
SPEC = importlib.util.spec_from_file_location("dec028_a2_evaluator_g0_candidate", SCRIPT_PATH)
assert SPEC and SPEC.loader
A2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A2)


def config() -> dict[str, Any]:
    return A2.load_config(CONFIG_PATH)


def valid_components() -> dict[str, tuple[str, ...]]:
    return {
        "synthetic-component-a": ("opaque-a0", "opaque-a1"),
        "synthetic-component-b": ("opaque-b0",),
        "synthetic-component-c": ("opaque-c0", "opaque-c1"),
    }


def test_disk_candidate_is_non_authoritative_and_validate_only_reports_zero_runtime_touchpoints(capsys: pytest.CaptureFixture[str]) -> None:
    candidate = config()
    assert candidate["authority_status"] == "NON_AUTHORITATIVE_G0_PREPARATION_ONLY"
    assert candidate["activation_state"] == "INACTIVE_NO_REAL_MEMBERSHIP_OR_ASSIGNMENT"
    assert candidate["static_authority"]["current_qualified_counts"] == A2.FROZEN_COUNTS
    assert candidate["runtime_truth"]["project_rows_read"] == 0
    assert candidate["runtime_truth"]["cuda_probe_calls"] == 0
    assert A2.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "a2_pass_asserted": False,
        "a7_unlocked": False,
        "activation_state": "INACTIVE_NO_REAL_MEMBERSHIP_OR_ASSIGNMENT",
        "authority_status": "NON_AUTHORITATIVE_G0_PREPARATION_ONLY",
        "cuda_probe_calls": 0,
        "current_qualified_counts": {"a1": 1, "canonical_records": 6547, "ordinary": 1, "true_a2": 0},
        "gpu_runs": 0,
        "project_rows_read": 0,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "split_assignment_count": 0,
        "status": "G0_SYNTHETIC_EVALUATOR_CONTRACT_VALIDATED_NOT_A2_PASS",
        "training_runs": 0,
    }


def test_synthetic_component_partition_and_outcome_blind_recipe_are_valid_without_assignments() -> None:
    candidate = config()
    component_summary = A2.validate_synthetic_component_graph(valid_components())
    split_summary = A2.validate_outcome_blind_split_recipe(candidate["synthetic_split_recipe"], valid_components())
    assert component_summary == {"component_count": 3, "opaque_synthetic_node_count": 5, "split_assignment_count": 0}
    assert split_summary == {"outcome_blind": 1, "component_disjoint": 1, "split_assignment_count": 0}


def test_component_overlap_and_outcome_or_assignment_leakage_are_rejected() -> None:
    overlapping = valid_components()
    overlapping["synthetic-component-b"] = ("opaque-a1", "opaque-b0")
    with pytest.raises(A2.ContractError, match="disjoint"):
        A2.validate_synthetic_component_graph(overlapping)

    recipe = copy.deepcopy(config()["synthetic_split_recipe"])
    recipe["outcome_blind"] = False
    with pytest.raises(A2.ContractError, match="outcome_blind"):
        A2.validate_outcome_blind_split_recipe(recipe, valid_components())

    recipe = copy.deepcopy(config()["synthetic_split_recipe"])
    recipe["split_assignment_count"] = 1
    with pytest.raises(A2.ContractError, match="split_assignment_count"):
        A2.validate_outcome_blind_split_recipe(recipe, valid_components())


def test_direction_normalized_endpoint_and_biological_se_rules_reject_zero_imputation_and_nonfinite_values() -> None:
    candidate = config()
    assert A2.validate_direction_normalized_endpoint_schema(candidate["aggregate_evaluator_schema"]) == {
        "target_definition": "CANDIDATE_MINUS_SOURCE_DIRECTION_NORMALIZED",
        "real_endpoint_values_allowed": "false",
    }
    schema = copy.deepcopy(candidate["aggregate_evaluator_schema"])
    schema["direction_normalization_required"] = False
    with pytest.raises(A2.ContractError, match="direction_normalization_required"):
        A2.validate_direction_normalized_endpoint_schema(schema)
    schema = copy.deepcopy(candidate["aggregate_evaluator_schema"])
    schema["real_endpoint_values_allowed"] = True
    with pytest.raises(A2.ContractError, match="real_endpoint_values_allowed"):
        A2.validate_direction_normalized_endpoint_schema(schema)

    records = [
        {"status": "OBSERVED", "biological_se": 0.25, "imputed_as_zero": False},
        {"status": "MISSING_OR_CENSORED", "biological_se": None, "imputed_as_zero": False},
    ]
    assert A2.validate_synthetic_biological_se_records(records) == {
        "observed_with_finite_positive_se": 1,
        "missing_or_censored": 1,
        "zero_imputations": 0,
    }
    for invalid in (
        [{"status": "OBSERVED", "biological_se": 0.0, "imputed_as_zero": False}],
        [{"status": "OBSERVED", "biological_se": math.nan, "imputed_as_zero": False}],
        [{"status": "MISSING_OR_CENSORED", "biological_se": 0.0, "imputed_as_zero": False}],
        [{"status": "MISSING_OR_CENSORED", "biological_se": None, "imputed_as_zero": True}],
    ):
        with pytest.raises(A2.ContractError):
            A2.validate_synthetic_biological_se_records(invalid)


def test_same_information_baseline_rejects_guide_model_outcome_and_selection_inputs() -> None:
    contract = config()["same_information_direct_baseline_contract"]
    assert A2.validate_same_information_direct_baseline(contract) == {
        "guide_input_allowed": False,
        "model_input_allowed": False,
        "selection_allowed": False,
    }
    altered = copy.deepcopy(contract)
    altered["forbidden_input_roles"].remove("guide_output")
    with pytest.raises(A2.ContractError, match="guide"):
        A2.validate_same_information_direct_baseline(altered)
    altered = copy.deepcopy(contract)
    altered["usable_for_model_selection"] = True
    with pytest.raises(A2.ContractError, match="model_selection"):
        A2.validate_same_information_direct_baseline(altered)


def test_every_operational_request_fails_before_callback_or_output() -> None:
    invoked: list[str] = []

    def forbidden_callback() -> None:
        invoked.append("called")

    for operation in sorted(A2.FORBIDDEN_OPERATION_SET):
        with pytest.raises(A2.ContractError, match="rejected before callback"):
            A2.reject_operational_request(operation, forbidden_callback)
    assert invoked == []
    with pytest.raises(A2.ContractError, match="unknown operation"):
        A2.reject_operational_request("STATIC_REPORT", forbidden_callback)


def test_config_rejects_count_lock_or_active_authority_drift() -> None:
    candidate = config()
    for keys, value in (
        (("static_authority", "current_qualified_counts", "a1"), 2),
        (("static_authority", "effective_active_amendment_decision_ids"), list(A2.BEFORE_DECISION_IDS) + ["V3-DEC-028"]),
        (("runtime_truth", "a2_pass_asserted"), True),
    ):
        altered = copy.deepcopy(candidate)
        cursor: dict[str, Any] = altered
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        with pytest.raises(A2.ContractError):
            A2.validate_config(altered)
