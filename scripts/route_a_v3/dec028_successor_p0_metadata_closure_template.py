#!/usr/bin/env python3
"""Validate the static DEC028 successor-P0 closure template.

This is not a production P0 runner. It evaluates only the template's aggregate
gate vocabulary and has no data, CUDA, model, checkpoint, or runtime-output
surface. A synthetic all-PASS fixture can prove result semantics but remains
only eligible to request a later materialization authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG_PATH = REPOSITORY_ROOT / "configs/route_a_v3_dec028_successor_p0_metadata_closure_template_v1.json"
BEFORE_DECISION_IDS = (
    "V3-DEC-017", "V3-DEC-018", "V3-DEC-019", "V3-DEC-020", "V3-DEC-021",
    "V3-DEC-022", "V3-DEC-023", "V3-DEC-024", "V3-DEC-027",
)
GROUPS = (
    ("P0.1", "INPUT_MEMBERSHIP_AND_BINDING"),
    ("P0.2", "PRIOR_USE_ATTESTATION"),
    ("P0.3", "EXPOSURE_ROLE"),
    ("P0.4", "RIGHTS"),
    ("P0.5", "SCIENTIFIC_ROW_CONTRACT"),
    ("P0.6", "SCRATCH_ONLY_ROUTE"),
    ("P0.7", "PROSPECTIVE_SPLIT"),
    ("P0.8", "SINGLE_RUN_POLICY"),
    ("P0.9", "EXECUTABLE_SCIENTIFIC_GATES"),
    ("P0.10", "SUCCESSOR_LEARNED_RUN_IMPLEMENTATION"),
    ("P0.11", "STATE_LOCKS"),
)
FROZEN_COUNTS = {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547}
FAILURE_RESULT = "STOP_BEFORE_DATA_CUDA_MODEL"
SUCCESS_RESULT = "ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED"


class ClosureError(RuntimeError):
    """The static closure template violates its own fail-closed contract."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClosureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ClosureError) as exc:
        raise ClosureError(f"{label} is not a unique-key JSON object") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object")
    return value


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ClosureError(f"{label} differs from the static closure template")


def _expect_keys(value: Any, keys: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ClosureError(f"{label} key closure differs")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _group_ids(groups: list[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(group.get("gate_id") for group in groups if isinstance(group, Mapping))


def validate_config(config: Mapping[str, Any]) -> None:
    _expect(config.get("schema_version"), "route_a_v3_dec028_successor_p0_metadata_closure_template.v1", label="schema")
    _expect(config.get("template_id"), "ROUTE_A_V3_DEC028_SUCCESSOR_P0_METADATA_CLOSURE_TEMPLATE_V1", label="template id")
    _expect(config.get("document_status"), "SCHEMA_FIXTURE_AND_FAILURE_CODES_ONLY_NOT_A_PRODUCTION_P0_RECORD", label="document status")
    _expect(config.get("production_p0_authorized"), False, label="production authorization")
    _expect(config.get("runtime_output_allowed"), False, label="runtime output")

    authority = _expect_keys(
        config.get("static_authority"),
        {"root_config_path", "root_config_sha256", "dec028_amendment_path", "dec028_amendment_sha256", "effective_active_amendment_decision_ids", "pending_successor_decision_id", "current_qualified_counts", "scientific_claim_status"},
        label="static authority",
    )
    _expect(authority["root_config_path"], "configs/route_a_v3.yaml", label="root path")
    _expect(authority["dec028_amendment_path"], "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml", label="amendment path")
    if not _is_sha256(authority["root_config_sha256"]) or not _is_sha256(authority["dec028_amendment_sha256"]):
        raise ClosureError("static authority hash is invalid")
    _expect(tuple(authority["effective_active_amendment_decision_ids"]), BEFORE_DECISION_IDS, label="active decisions")
    _expect(authority["pending_successor_decision_id"], "V3-DEC-028", label="pending successor")
    _expect(authority["current_qualified_counts"], FROZEN_COUNTS, label="counts")
    _expect(authority["scientific_claim_status"], "NOT_ESTABLISHED", label="claim")

    p0_groups = config.get("p0_groups")
    if not isinstance(p0_groups, list) or [(item.get("gate_id"), item.get("gate_name")) for item in p0_groups if isinstance(item, dict)] != list(GROUPS):
        raise ClosureError("P0 group order or names differ")
    allowed = config.get("allowed_declared_statuses")
    if not isinstance(allowed, list) or "PASS" not in allowed or "UNKNOWN_NOT_ASSERTED" not in allowed or not all(item.startswith("FAIL_CLOSED") or item in {"PASS", "UNKNOWN_NOT_ASSERTED"} for item in allowed):
        raise ClosureError("declared-status vocabulary differs")

    initial = config.get("initial_template_groups")
    if not isinstance(initial, dict) or set(initial) != {gate_id for gate_id, _ in GROUPS}:
        raise ClosureError("initial template group closure differs")
    for gate_id, _ in GROUPS:
        group = _expect_keys(initial[gate_id], {"declared_status", "failure_code"}, label=f"initial {gate_id}")
        if group["declared_status"] not in allowed:
            raise ClosureError(f"initial {gate_id} status is outside vocabulary")
        if not isinstance(group["failure_code"], str) or not group["failure_code"]:
            raise ClosureError(f"initial {gate_id} failure code is invalid")
    expected_initial = {
        "P0.1": "FAIL_CLOSED_BINDING_ABSENT", "P0.2": "UNKNOWN_NOT_ASSERTED", "P0.3": "PASS",
        "P0.4": "FAIL_CLOSED_INTENDED_INTERNAL_TRAIN_RIGHTS_NOT_BOUND", "P0.5": "FAIL_CLOSED_COMPLETE_ROW_CONTRACT_NOT_BOUND",
        "P0.6": "PASS", "P0.7": "FAIL_CLOSED_PROSPECTIVE_SPLIT_NOT_FROZEN",
        "P0.8": "FAIL_CLOSED_SINGLE_RUN_POLICY_NOT_ACTIVE", "P0.9": "FAIL_CLOSED_EXECUTABLE_GATE_BUNDLE_NOT_BOUND",
        "P0.10": "FAIL_CLOSED_SUCCESSOR_NOT_BOUND", "P0.11": "PASS",
    }
    _expect({gate_id: initial[gate_id]["declared_status"] for gate_id, _ in GROUPS}, expected_initial, label="initial statuses")

    owner_inputs = config.get("owner_metadata_closure_inputs")
    required_owner_gates = {"P0.1", "P0.2", "P0.4", "P0.5", "P0.7"}
    if not isinstance(owner_inputs, dict) or set(owner_inputs) != required_owner_gates:
        raise ClosureError("owner metadata closure input set differs")
    for gate_id, entry in owner_inputs.items():
        entry = _expect_keys(entry, {"responsible_role", "minimum_aggregate_input", "row_or_member_payload_allowed"}, label=f"owner input {gate_id}")
        if not isinstance(entry["responsible_role"], str) or not entry["responsible_role"] or not isinstance(entry["minimum_aggregate_input"], str) or not entry["minimum_aggregate_input"]:
            raise ClosureError(f"owner input {gate_id} is incomplete")
        _expect(entry["row_or_member_payload_allowed"], False, label=f"owner input {gate_id} payload boundary")

    result_policy = _expect_keys(config.get("result_policy"), {"all_eleven_pass_result", "any_nonpass_result", "all_pass_starts_materialization", "all_pass_launches_g1", "all_pass_changes_scientific_claim"}, label="result policy")
    _expect(result_policy["all_eleven_pass_result"], SUCCESS_RESULT, label="all pass result")
    _expect(result_policy["any_nonpass_result"], FAILURE_RESULT, label="nonpass result")
    for key in ("all_pass_starts_materialization", "all_pass_launches_g1", "all_pass_changes_scientific_claim"):
        _expect(result_policy[key], False, label=f"result policy {key}")

    forbidden = config.get("forbidden_touchpoints")
    required_forbidden = {"DATA_ROWS", "MEMBER_IDS", "SEQUENCES", "ENDPOINT_VALUES", "STANDARD_ERRORS", "SPLIT_ASSIGNMENTS", "CUDA", "DEVICE", "MODEL", "OPTIMIZER", "CHECKPOINT", "PARAMETER_UPDATE", "TRAINING", "RUNTIME_OUTPUT"}
    if not isinstance(forbidden, list) or set(forbidden) != required_forbidden:
        raise ClosureError("forbidden touchpoint closure differs")
    locks = config.get("persistent_locks")
    if not isinstance(locks, dict) or set(locks) != {"materialization_allowed", "training_allowed", "gpu_work_allowed", "model_selection_allowed", "a7_allowed", "sealed_access_allowed", "g1_launched"} or any(value is not False for value in locks.values()):
        raise ClosureError("persistent locks differ")
    truth = config.get("runtime_truth")
    if not isinstance(truth, dict) or truth.get("g1_launched") is not False or any(value != 0 for key, value in truth.items() if key != "g1_launched"):
        raise ClosureError("runtime truth differs")


def load_config(path: Path = PRODUCTION_CONFIG_PATH) -> dict[str, Any]:
    try:
        config = load_json(path.read_bytes(), label="DEC028 successor P0 template")
    except OSError as exc:
        raise ClosureError(f"cannot read closure template: {path}") from exc
    validate_config(config)
    if path.resolve() == PRODUCTION_CONFIG_PATH.resolve():
        authority = config["static_authority"]
        for relative, expected in ((authority["root_config_path"], authority["root_config_sha256"]), (authority["dec028_amendment_path"], authority["dec028_amendment_sha256"])):
            try:
                actual = sha256((REPOSITORY_ROOT / relative).read_bytes())
            except OSError as exc:
                raise ClosureError(f"cannot read bound static authority: {relative}") from exc
            if actual != expected:
                raise ClosureError("bound current-authority byte identity differs")
    return config


def evaluate_aggregate_groups(config: Mapping[str, Any], groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate aggregate statuses only; no owner input or payload is read here."""

    validate_config(config)
    expected_ids = {gate_id for gate_id, _ in GROUPS}
    if set(groups) != expected_ids:
        raise ClosureError("aggregate package has missing or unexpected P0 groups")
    allowed = set(config["allowed_declared_statuses"])
    statuses: list[dict[str, str]] = []
    for gate_id, gate_name in GROUPS:
        group = _expect_keys(groups[gate_id], {"declared_status", "failure_code"}, label=f"aggregate {gate_id}")
        status = group["declared_status"]
        if status not in allowed:
            raise ClosureError(f"aggregate {gate_id} status is outside vocabulary")
        if not isinstance(group["failure_code"], str) or not group["failure_code"]:
            raise ClosureError(f"aggregate {gate_id} failure code is invalid")
        statuses.append({"gate_id": gate_id, "gate_name": gate_name, "status": status})
    all_pass = all(item["status"] == "PASS" for item in statuses)
    return {
        "result_status": SUCCESS_RESULT if all_pass else FAILURE_RESULT,
        "gate_statuses": statuses,
        "materialization_started": False,
        "g1_launched": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def initial_template_report(config: Mapping[str, Any]) -> dict[str, Any]:
    result = evaluate_aggregate_groups(config, config["initial_template_groups"])
    return {
        **result,
        "document_status": config["document_status"],
        "production_p0_authorized": False,
        "runtime_output_allowed": False,
        "current_qualified_counts": copy.deepcopy(config["static_authority"]["current_qualified_counts"]),
        "runtime_truth": copy.deepcopy(config["runtime_truth"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="validate the static template and print its aggregate report")
    args = parser.parse_args(argv)
    if not args.validate_only:
        parser.error("only --validate-only is available; production P0 is not authorized")
    try:
        report = initial_template_report(load_config())
    except ClosureError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
