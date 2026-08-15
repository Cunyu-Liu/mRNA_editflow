#!/usr/bin/env python3
"""Validate the non-authoritative DEC028 synthetic A2 evaluator candidate.

This module intentionally does not load project rows, create assignments,
evaluate predictions, or initialize a learned system.  Its public command is a
static JSON report; operational requests fail before their callback runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/route_a_v3_dec028_a2_evaluator_g0_candidate_v1.json"
HEX64 = "0123456789abcdef"
BEFORE_DECISION_IDS = (
    "V3-DEC-017",
    "V3-DEC-018",
    "V3-DEC-019",
    "V3-DEC-020",
    "V3-DEC-021",
    "V3-DEC-022",
    "V3-DEC-023",
    "V3-DEC-024",
    "V3-DEC-027",
)
FROZEN_COUNTS = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
FORBIDDEN_OPERATION_SET = {
    "PROJECT_ROW_OR_MEMBER_ID_READ",
    "REAL_MEMBERSHIP_OR_SPLIT_ASSIGNMENT",
    "REAL_EVALUATION_OR_BIOLOGICAL_A2_PASS",
    "GUIDE_TO_EVALUATOR_FEEDBACK",
    "MODEL_OR_CHECKPOINT_INPUT",
    "MODEL_SELECTION_OR_THRESHOLD_SELECTION",
    "CUDA_OR_GPU_TOUCH",
    "TRAINING_OR_PARAMETER_UPDATE",
    "RUNTIME_OUTPUT_FILE_WRITE",
    "QUALIFICATION_CREDIT_OR_CANONICAL_CHANGE",
    "A7_UNLOCK",
    "SEALED_ACCESS",
}


class ContractError(RuntimeError):
    """A static G0 rule or authority boundary has been violated."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError(f"{label} is not a unique-key JSON object") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ContractError(f"{label} differs from the frozen candidate")


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{label} key closure differs")
    return value


def _hex64(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in HEX64 for character in value):
        raise ContractError(f"{label} is not a lower-case SHA-256")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _expect(config.get("schema_version"), "route_a_v3_dec028_a2_evaluator_g0_candidate.v1", label="schema")
    _expect(config.get("candidate_id"), "ROUTE_A_V3_DEC028_A2_EVALUATOR_G0_CANDIDATE_V1", label="candidate id")
    _expect(config.get("document_status"), "DRAFT_FOR_DISTINCT_REVIEW_NOT_ACTIVE_PROTOCOL", label="document status")
    _expect(config.get("authority_status"), "NON_AUTHORITATIVE_G0_PREPARATION_ONLY", label="authority status")
    _expect(config.get("activation_state"), "INACTIVE_NO_REAL_MEMBERSHIP_OR_ASSIGNMENT", label="activation state")

    authority = _expect_keys(
        config.get("static_authority"),
        {
            "root_config_path",
            "root_config_sha256",
            "dec028_amendment_path",
            "dec028_amendment_sha256",
            "effective_active_amendment_decision_ids",
            "pending_successor_decision_id",
            "current_qualified_counts",
            "scientific_claim_status",
        },
        label="static authority",
    )
    _expect(authority["root_config_path"], "configs/route_a_v3.yaml", label="root config path")
    _hex64(authority["root_config_sha256"], label="root config hash")
    _expect(
        authority["dec028_amendment_path"],
        "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml",
        label="amendment path",
    )
    _hex64(authority["dec028_amendment_sha256"], label="amendment hash")
    _expect(tuple(authority["effective_active_amendment_decision_ids"]), BEFORE_DECISION_IDS, label="active decision ids")
    _expect(authority["pending_successor_decision_id"], "V3-DEC-028", label="pending decision")
    _expect(authority["current_qualified_counts"], FROZEN_COUNTS, label="current counts")
    _expect(authority["scientific_claim_status"], "NOT_ESTABLISHED", label="claim status")

    permitted = config.get("permitted_g0_operations")
    if not isinstance(permitted, list) or set(permitted) != {
        "VALIDATE_SYNTHETIC_CONNECTED_COMPONENTS",
        "VALIDATE_OUTCOME_BLIND_SPLIT_RECIPE_WITH_ZERO_ASSIGNMENTS",
        "VALIDATE_CANDIDATE_MINUS_SOURCE_DIRECTION_NORMALIZED_SCHEMA",
        "VALIDATE_SYNTHETIC_BIOLOGICAL_SE_AND_MISSINGNESS_POLICY",
        "VALIDATE_SAME_INFORMATION_DIRECT_BASELINE_CONTRACT",
        "VALIDATE_ONLY_JSON_REPORT_TO_STDOUT",
    }:
        raise ContractError("permitted G0 operation closure differs")
    forbidden = config.get("forbidden_operations")
    if not isinstance(forbidden, list) or set(forbidden) != FORBIDDEN_OPERATION_SET:
        raise ContractError("forbidden operation closure differs")

    recipe = _expect_keys(
        config.get("synthetic_split_recipe"),
        {"scope", "outcome_blind", "split_assignment_count", "grouping_keys", "salt", "forbidden_information"},
        label="synthetic split recipe",
    )
    _expect(recipe["scope"], "SYNTHETIC_ONLY_NO_PROJECT_MEMBERSHIP_OR_ASSIGNMENTS", label="recipe scope")
    _expect(recipe["outcome_blind"], True, label="recipe outcome blindness")
    _expect(recipe["split_assignment_count"], 0, label="recipe assignment count")
    _expect(recipe["grouping_keys"], ["synthetic_connected_component"], label="recipe grouping keys")
    _expect(recipe["salt"], "DEC028_A2_G0_SYNTHETIC_COMPONENT_RECIPE_V1", label="recipe salt")
    expected_forbidden_information = {
        "outcome",
        "effect",
        "significance",
        "standard_error",
        "prediction",
        "guide_output",
        "checkpoint",
        "model_selection_result",
    }
    if set(recipe["forbidden_information"]) != expected_forbidden_information:
        raise ContractError("recipe forbidden-information closure differs")

    schema = _expect_keys(
        config.get("aggregate_evaluator_schema"),
        {
            "target_definition",
            "source_required",
            "candidate_required",
            "direction_normalization_required",
            "biological_se_required_when_observed",
            "finite_positive_biological_se_required",
            "missing_or_censored_never_imputed_as_zero",
            "real_endpoint_values_allowed",
            "real_member_or_sequence_values_allowed",
        },
        label="aggregate evaluator schema",
    )
    _expect(schema["target_definition"], "CANDIDATE_MINUS_SOURCE_DIRECTION_NORMALIZED", label="target definition")
    for key in (
        "source_required",
        "candidate_required",
        "direction_normalization_required",
        "biological_se_required_when_observed",
        "finite_positive_biological_se_required",
        "missing_or_censored_never_imputed_as_zero",
    ):
        _expect(schema[key], True, label=f"schema {key}")
    _expect(schema["real_endpoint_values_allowed"], False, label="real endpoints")
    _expect(schema["real_member_or_sequence_values_allowed"], False, label="real members")

    baseline = _expect_keys(
        config.get("same_information_direct_baseline_contract"),
        {"baseline_id", "allowed_input_roles", "forbidden_input_roles", "usable_for_model_selection", "usable_for_checkpoint_selection", "usable_for_scientific_claim"},
        label="baseline contract",
    )
    _expect(baseline["baseline_id"], "DIRECT_CANDIDATE_MINUS_SOURCE_AGGREGATE_BASELINE_V1", label="baseline id")
    _expect(
        set(baseline["allowed_input_roles"]),
        {"candidate_metadata", "source_metadata", "direction_metadata", "aggregate_schema_metadata"},
        label="baseline allowed roles",
    )
    _expect(
        set(baseline["forbidden_input_roles"]),
        {"guide_output", "model_prediction", "checkpoint", "optimizer_state", "selection_metric", "real_outcome", "split_assignment"},
        label="baseline forbidden roles",
    )
    for key in ("usable_for_model_selection", "usable_for_checkpoint_selection", "usable_for_scientific_claim"):
        _expect(baseline[key], False, label=f"baseline {key}")

    truth = _expect_keys(
        config.get("runtime_truth"),
        {
            "project_rows_read", "member_ids_read", "split_assignments_created", "guide_outputs_read", "model_inputs_read",
            "cuda_probe_calls", "gpu_runs", "training_runs", "parameter_updates", "runtime_output_files_written",
            "qualification_changes", "credit_changes", "canonical_changes", "a2_pass_asserted", "a7_unlocked", "sealed_accessed",
        },
        label="runtime truth",
    )
    for key, value in truth.items():
        if key in {"a2_pass_asserted", "a7_unlocked", "sealed_accessed"}:
            _expect(value, False, label=f"runtime truth {key}")
        else:
            _expect(value, 0, label=f"runtime truth {key}")


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        config = load_json(path.read_bytes(), label="DEC028 A2 G0 candidate config")
    except OSError as exc:
        raise ContractError(f"cannot read candidate configuration: {path}") from exc
    validate_config(config)
    if path.resolve() == PRODUCTION_CONFIG_PATH.resolve():
        authority = config["static_authority"]
        for relative, expected in (
            (authority["root_config_path"], authority["root_config_sha256"]),
            (authority["dec028_amendment_path"], authority["dec028_amendment_sha256"]),
        ):
            try:
                actual = sha256((REPOSITORY_ROOT / relative).read_bytes())
            except OSError as exc:
                raise ContractError(f"cannot read bound static authority: {relative}") from exc
            if actual != expected:
                raise ContractError("bound current-authority byte identity differs")
    return config


def validate_synthetic_component_graph(components: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Validate an opaque synthetic component partition without assignments."""

    if not isinstance(components, Mapping) or len(components) < 2:
        raise ContractError("synthetic component graph requires at least two components")
    observed_nodes: set[str] = set()
    for component, nodes in components.items():
        if not isinstance(component, str) or not component or not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise ContractError("synthetic component graph shape is invalid")
        if not nodes:
            raise ContractError("synthetic component is empty")
        local_nodes: set[str] = set()
        for node in nodes:
            if not isinstance(node, str) or not node or node in local_nodes or node in observed_nodes:
                raise ContractError("synthetic component graph is not a disjoint partition")
            local_nodes.add(node)
            observed_nodes.add(node)
    return {"component_count": len(components), "opaque_synthetic_node_count": len(observed_nodes), "split_assignment_count": 0}


def validate_outcome_blind_split_recipe(recipe: Mapping[str, Any], components: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Validate a future split recipe; never generate real or synthetic assignments."""

    validate_synthetic_component_graph(components)
    expected = {
        "scope": "SYNTHETIC_ONLY_NO_PROJECT_MEMBERSHIP_OR_ASSIGNMENTS",
        "outcome_blind": True,
        "split_assignment_count": 0,
        "grouping_keys": ["synthetic_connected_component"],
        "salt": "DEC028_A2_G0_SYNTHETIC_COMPONENT_RECIPE_V1",
    }
    for key, value in expected.items():
        _expect(recipe.get(key), value, label=f"split recipe {key}")
    forbidden = recipe.get("forbidden_information")
    if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes)):
        raise ContractError("split recipe forbidden information is invalid")
    required = {"outcome", "effect", "significance", "standard_error", "prediction", "guide_output", "checkpoint", "model_selection_result"}
    if set(forbidden) != required:
        raise ContractError("split recipe permits outcome or selection information")
    return {"outcome_blind": 1, "component_disjoint": 1, "split_assignment_count": 0}


def validate_direction_normalized_endpoint_schema(schema: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "target_definition": "CANDIDATE_MINUS_SOURCE_DIRECTION_NORMALIZED",
        "source_required": True,
        "candidate_required": True,
        "direction_normalization_required": True,
        "biological_se_required_when_observed": True,
        "finite_positive_biological_se_required": True,
        "missing_or_censored_never_imputed_as_zero": True,
        "real_endpoint_values_allowed": False,
        "real_member_or_sequence_values_allowed": False,
    }
    for key, value in expected.items():
        _expect(schema.get(key), value, label=f"endpoint schema {key}")
    return {"target_definition": expected["target_definition"], "real_endpoint_values_allowed": "false"}


def validate_synthetic_biological_se_records(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Reject zero-imputation, nonfinite, and nonpositive SE in synthetic fixtures."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ContractError("synthetic SE fixture is absent")
    observed = 0
    missing = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"status", "biological_se", "imputed_as_zero"}:
            raise ContractError("synthetic SE record shape differs")
        if record["imputed_as_zero"] is not False:
            raise ContractError("missing or nonfinite biological SE may not be imputed as zero")
        if record["status"] == "OBSERVED":
            value = record["biological_se"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ContractError("observed biological SE must be finite and positive")
            observed += 1
        elif record["status"] == "MISSING_OR_CENSORED":
            if record["biological_se"] is not None:
                raise ContractError("missing biological SE must remain missing, not a numeric substitute")
            missing += 1
        else:
            raise ContractError("synthetic SE status is invalid")
    return {"observed_with_finite_positive_se": observed, "missing_or_censored": missing, "zero_imputations": 0}


def validate_same_information_direct_baseline(contract: Mapping[str, Any]) -> dict[str, Any]:
    expected_allowed = {"candidate_metadata", "source_metadata", "direction_metadata", "aggregate_schema_metadata"}
    expected_forbidden = {"guide_output", "model_prediction", "checkpoint", "optimizer_state", "selection_metric", "real_outcome", "split_assignment"}
    _expect(contract.get("baseline_id"), "DIRECT_CANDIDATE_MINUS_SOURCE_AGGREGATE_BASELINE_V1", label="baseline id")
    if set(contract.get("allowed_input_roles", [])) != expected_allowed:
        raise ContractError("baseline allowed inputs differ from same-information contract")
    if set(contract.get("forbidden_input_roles", [])) != expected_forbidden:
        raise ContractError("baseline accepts guide, model, outcome, or selection input")
    for key in ("usable_for_model_selection", "usable_for_checkpoint_selection", "usable_for_scientific_claim"):
        _expect(contract.get(key), False, label=f"baseline {key}")
    return {"guide_input_allowed": False, "model_input_allowed": False, "selection_allowed": False}


def reject_operational_request(operation: str, callback: Callable[[], Any]) -> None:
    """Fail before callback so static G0 never touches a forbidden operation."""

    if operation not in FORBIDDEN_OPERATION_SET:
        raise ContractError("unknown operation is not part of this G0 candidate")
    raise ContractError(f"{operation} rejected before callback/data/CUDA/model/runtime I/O")


def validate_only_report(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    validate_direction_normalized_endpoint_schema(config["aggregate_evaluator_schema"])
    validate_same_information_direct_baseline(config["same_information_direct_baseline_contract"])
    return {
        "status": "G0_SYNTHETIC_EVALUATOR_CONTRACT_VALIDATED_NOT_A2_PASS",
        "authority_status": config["authority_status"],
        "activation_state": config["activation_state"],
        "current_qualified_counts": copy.deepcopy(config["static_authority"]["current_qualified_counts"]),
        "scientific_claim_status": config["static_authority"]["scientific_claim_status"],
        "split_assignment_count": 0,
        "project_rows_read": 0,
        "cuda_probe_calls": 0,
        "gpu_runs": 0,
        "training_runs": 0,
        "a2_pass_asserted": False,
        "a7_unlocked": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="emit the static G0 report and exit")
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("only --validate-only is available for this non-authoritative G0 candidate")
    try:
        report = validate_only_report(load_config())
    except ContractError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
