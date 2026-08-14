#!/usr/bin/env python3
"""Active metadata-only DEC026 zero-step P0 evaluator.

The evaluator reads only its frozen repository configuration and Git metadata.
It evaluates the exact eleven DEC026 P0 groups, then writes exactly one small
aggregate record.  It has no data/materialization, CUDA/device, model,
optimizer, checkpoint, parameter-update, trainer, or G1-launch surface.

An all-PASS result means only that a separate orchestrator may launch the one
authorized G1 run.  Any non-PASS result stops before all operational surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec026_zero_step_p0_v1.json"

PROTOCOL_ID = "ROUTE_A_V3_DEC026_METADATA_ONLY_ZERO_STEP_P0_V1"
DOCUMENT_STATUS = "FROZEN_USER_AUTHORIZED_METADATA_ONLY_ZERO_STEP_P0"
AUTHORITY_STATUS = "GRANTED_METADATA_ONLY_ZERO_STEP_P0"
PACKAGE_SCHEMA = "route_a_v3_dec026_zero_step_p0_metadata_package.v1"
PACKAGE_ROLE = "STATIC_METADATA_AND_AUTHORITY_BINDINGS_ONLY"
FAILURE_STATUS = "ZERO_STEP_P0_FAILURE_STOP_BEFORE_DATA_CUDA_MODEL"
PASS_STATUS = "ZERO_STEP_P0_PASS_G1_ONE_RUN_ELIGIBLE_NOT_LAUNCHED"
REPORT_FILENAME = "ROUTE_A_V3_DEC026_ZERO_STEP_P0_RECORD_V1.json"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

P0_GROUPS: tuple[tuple[str, str], ...] = (
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

FORBIDDEN_TOUCHPOINTS: tuple[str, ...] = (
    "DATA_ROW_READ",
    "MATERIALIZATION_READ_OR_EXERCISE",
    "CUDA_PROBE",
    "DEVICE_PROBE",
    "MODEL_CONSTRUCTION",
    "OPTIMIZER_CONSTRUCTION",
    "CHECKPOINT_READ",
    "CHECKPOINT_WRITE",
    "PARAMETER_UPDATE",
    "TRAINING_OUTPUT_WRITE",
    "TRAINING_START",
    "GPU_RUN",
)


class ProtocolError(RuntimeError):
    """The frozen active zero-step protocol is invalid."""


CandidateContractError = ProtocolError


class ForbiddenTouchpointError(RuntimeError):
    """A forbidden operation was deliberately instrumented in a test."""


class PartialGroupError(ValueError):
    """A P0 group is missing or contains an incomplete/expanded field set."""


class SemanticMismatchError(ValueError):
    """A complete P0 group does not encode the frozen required state."""


@dataclass
class ForbiddenTouchpointSentinel:
    """Test seam that records any attempted forbidden operation.

    The validator only calls :meth:`assert_zero`; it never calls
    :meth:`touch`.  Focused tests inject this sentinel and verify every named
    touchpoint remains zero on success-shaped, partial, missing, and malformed
    metadata packages.
    """

    counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in FORBIDDEN_TOUCHPOINTS}
    )

    def touch(self, name: str) -> None:
        if name not in self.counts:
            raise ForbiddenTouchpointError(f"unknown touchpoint: {name}")
        self.counts[name] += 1
        raise ForbiddenTouchpointError(f"forbidden touchpoint attempted: {name}")

    def assert_zero(self) -> None:
        if set(self.counts) != set(FORBIDDEN_TOUCHPOINTS):
            raise ForbiddenTouchpointError("touchpoint instrumentation set differs")
        nonzero = {name: count for name, count in self.counts.items() if count != 0}
        if nonzero:
            raise ForbiddenTouchpointError(f"forbidden touchpoint count is nonzero: {nonzero}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise CandidateContractError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    """Read the candidate's own static JSON contract, never operational data."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CandidateContractError(f"cannot read static candidate config: {path}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateContractError(f"invalid static candidate config: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateContractError("candidate config root must be an object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise PartialGroupError(
            f"{label} field closure differs: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _expect_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise SemanticMismatchError(f"{label} differs")


def _expect_false(value: Any, label: str) -> None:
    if value is not False:
        raise SemanticMismatchError(f"{label} must be false")


def _expect_true(value: Any, label: str) -> None:
    if value is not True:
        raise SemanticMismatchError(f"{label} must be true")


def _expect_nonempty_binding(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticMismatchError(f"{label} is not a bound identifier")


def _expect_count(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SemanticMismatchError(f"{label} is not an integer count >= {minimum}")
    return value


def _expect_exact_count(value: Any, expected: int, label: str) -> None:
    observed = _expect_count(value, label)
    if observed != expected:
        raise SemanticMismatchError(f"{label} differs")


def _expect_positive_finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticMismatchError(f"{label} is not numeric")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise SemanticMismatchError(f"{label} must be positive and finite")


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_protocol(payload: Mapping[str, Any]) -> None:
    _expect_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "decision_id",
            "document_status",
            "authority",
            "implementation_binding",
            "evaluation_scope",
            "p0_groups",
            "current_submission",
            "gate_evidence_ledger",
            "allowed_declared_statuses",
            "forbidden_touchpoints",
            "retained_locks",
            "scientific_state",
            "output_contract",
        },
        "protocol",
    )
    exact_root = {
        "schema_version": "route_a_v3_dec026_zero_step_p0.v1",
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A6_G1_BRIDGE",
        "decision_id": "V3-DEC-026",
        "document_status": DOCUMENT_STATUS,
    }
    for key, expected in exact_root.items():
        if payload[key] != expected:
            raise ProtocolError(f"protocol {key} differs")

    authority = payload["authority"]
    expected_authority = {
        "decision_status": AUTHORITY_STATUS,
        "owner_authorization_record": (
            "ACTIVE_CODEX_THREAD_2026-08-14_DEC026_METADATA_ONLY_ZERO_STEP_P0_"
            "AND_CONDITIONAL_ONE_G1_RUN"
        ),
        "validator_activation_state": "ACTIVE_METADATA_ONLY_ZERO_STEP_P0",
        "validator_may_publish_exactly_one_aggregate_record": True,
        "validator_may_read_data_or_materialize": False,
        "validator_may_probe_cuda_or_device": False,
        "validator_may_construct_model_or_optimizer": False,
        "validator_may_read_or_write_checkpoint": False,
        "validator_may_update_parameters": False,
        "validator_may_launch_g1": False,
        "g1_launch_requires_all_eleven_pass": True,
        "g1_launch_is_separate_orchestrator_action": True,
    }
    if authority != expected_authority:
        raise ProtocolError("active zero-step authority boundary differs")

    binding = payload["implementation_binding"]
    _expect_exact_keys(
        binding,
        {
            "status",
            "implementation_expected_parent",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
            "implementation_exact_changed_paths",
            "binding_exact_changed_paths",
            "unknown_to_bound_scalar_paths",
        },
        "implementation binding",
    )
    if not _is_hex(binding["implementation_expected_parent"], 40):
        raise ProtocolError("implementation expected parent is not bound")
    expected_paths = [
        "configs/route_a_v3_dec026_zero_step_p0_v1.json",
        "scripts/route_a_v3/dec026_metadata_only_zero_step_p0.py",
        "tests/route_a_v3/test_dec026_metadata_only_zero_step_p0.py",
    ]
    if binding["implementation_exact_changed_paths"] != expected_paths:
        raise ProtocolError("implementation exact3 path closure differs")
    if binding["binding_exact_changed_paths"] != [expected_paths[0]]:
        raise ProtocolError("binding config-only path closure differs")
    if binding["implementation_script_path"] != expected_paths[1]:
        raise ProtocolError("implementation script path differs")
    if binding["implementation_test_path"] != expected_paths[2]:
        raise ProtocolError("implementation test path differs")
    dynamic = (
        binding["status"],
        binding["implementation_commit"],
        binding["implementation_script_sha256"],
        binding["implementation_test_sha256"],
    )
    if all(value == UNKNOWN for value in dynamic):
        pass
    elif (
        binding["status"] == BOUND
        and _is_hex(binding["implementation_commit"], 40)
        and _is_hex(binding["implementation_script_sha256"], 64)
        and _is_hex(binding["implementation_test_sha256"], 64)
    ):
        pass
    else:
        raise ProtocolError("implementation binding is partially bound")

    expected_scope = {
        "input_role": PACKAGE_ROLE,
        "missing_or_unknown_is_pass": False,
        "partial_group_policy": "FAIL_CLOSED",
        "unexpected_group_policy": "FAIL_CLOSED_WHOLE_SUBMISSION",
        "allowed_runtime_record": "EXACTLY_ONE_AGGREGATE_ZERO_STEP_P0_RECORD",
        "runtime_or_training_output_allowed": False,
        "all_eleven_pass_action": "RETURN_G1_ONE_RUN_ELIGIBLE_NOT_LAUNCHED",
        "any_nonpass_action": (
            "STOP_BEFORE_DATA_MATERIALIZATION_CUDA_DEVICE_MODEL_OPTIMIZER_"
            "CHECKPOINT_PARAMETER_UPDATE_OR_TRAINING"
        ),
    }
    if payload["evaluation_scope"] != expected_scope:
        raise ProtocolError("evaluation scope differs")

    expected_groups = [
        {"gate_id": gate_id, "gate_name": gate_name}
        for gate_id, gate_name in P0_GROUPS
    ]
    if payload["p0_groups"] != expected_groups:
        raise ProtocolError("the exact eleven P0 groups differ")
    if payload["forbidden_touchpoints"] != list(FORBIDDEN_TOUCHPOINTS):
        raise ProtocolError("forbidden-touchpoint closure differs")

    allowed = payload["allowed_declared_statuses"]
    if (
        not isinstance(allowed, list)
        or len(allowed) != len(set(allowed))
        or "PASS" not in allowed
        or UNKNOWN not in allowed
        or any(not isinstance(value, str) or not value for value in allowed)
    ):
        raise ProtocolError("declared-status vocabulary differs")

    submission = payload["current_submission"]
    _expect_exact_keys(
        submission,
        {"schema_version", "package_role", "groups"},
        "current submission",
    )
    if submission["schema_version"] != PACKAGE_SCHEMA:
        raise ProtocolError("current submission schema differs")
    if submission["package_role"] != PACKAGE_ROLE:
        raise ProtocolError("current submission role differs")
    groups = submission["groups"]
    expected_ids = {gate_id for gate_id, _ in P0_GROUPS}
    if not isinstance(groups, Mapping) or set(groups) != expected_ids:
        raise ProtocolError("current submission group closure differs")
    for gate_id in expected_ids:
        group = groups[gate_id]
        if not isinstance(group, Mapping):
            raise ProtocolError(f"{gate_id} current group is not an object")
        _expect_exact_keys(group, {"declared_status", "evidence"}, f"{gate_id} group")
        if group["declared_status"] not in allowed:
            raise ProtocolError(f"{gate_id} declared status is outside vocabulary")
        if not isinstance(group["evidence"], Mapping):
            raise ProtocolError(f"{gate_id} evidence is not an object")

    ledger = payload["gate_evidence_ledger"]
    if not isinstance(ledger, Mapping) or set(ledger) != expected_ids:
        raise ProtocolError("gate evidence ledger closure differs")
    for gate_id, item in ledger.items():
        if not isinstance(item, Mapping):
            raise ProtocolError(f"{gate_id} evidence ledger entry is not an object")
        _expect_exact_keys(
            item,
            {"fact_class", "reason_code", "evidence_refs"},
            f"{gate_id} evidence ledger",
        )
        if item["fact_class"] not in {"CONFIRMED_FACT", "UNKNOWN_NOT_ASSERTED"}:
            raise ProtocolError(f"{gate_id} fact class differs")
        _expect_nonempty_binding(item["reason_code"], f"{gate_id} reason code")
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or not refs or any(
            not isinstance(ref, str) or not ref for ref in refs
        ):
            raise ProtocolError(f"{gate_id} evidence refs are not closed")

    retained_locks = payload["retained_locks"]
    if set(retained_locks) != {
        "training_allowed",
        "gpu_work_allowed",
        "parameter_updates_allowed",
        "model_selection_allowed",
        "qualification_change_allowed",
        "credit_change_allowed",
        "canonical_mutation_allowed",
        "a6_pass_allowed",
        "l3_claim_allowed",
        "a7_unlock_allowed",
        "private_or_sealed_access_allowed",
        "scientific_claim_allowed",
    } or any(value is not False for value in retained_locks.values()):
        raise ProtocolError("retained state locks are not all false")

    expected_state = {
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "contribution_delta": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "a6_status": "IN_PROGRESS_UNCHANGED",
        "l3_claim_status": "NOT_ESTABLISHED_UNCHANGED",
        "a7_status": "NOT_RUN_NOT_UNLOCKED",
    }
    if payload["scientific_state"] != expected_state:
        raise ProtocolError("scientific no-change boundary differs")

    expected_output = {
        "run_root_prefix": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A6/"
            "ROUTE_A_V3_DEC026_ZERO_STEP_P0_B_"
        ),
        "report_filename": REPORT_FILENAME,
        "final_file_count": 1,
        "member_row_sequence_endpoint_split_model_optimizer_checkpoint_or_device_payload_allowed": False,
        "g1_launch_side_effect_allowed": False,
    }
    if payload["output_contract"] != expected_output:
        raise ProtocolError("output contract differs")


validate_candidate_config = validate_protocol


def load_protocol(path: Path = CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    try:
        validate_protocol(payload)
    except (PartialGroupError, SemanticMismatchError) as exc:
        raise ProtocolError(str(exc)) from exc
    return payload


load_candidate_config = load_protocol


def _validate_p01(evidence: Mapping[str, Any]) -> None:
    _expect_exact_keys(evidence, {"binding_route", "binding"}, "P0.1 evidence")
    route = evidence["binding_route"]
    binding = evidence["binding"]
    if not isinstance(binding, Mapping):
        raise PartialGroupError("P0.1 binding must be an object")
    if route == "BOUND_EXISTING_CANONICAL_ASSET":
        _expect_exact_keys(
            binding,
            {
                "authoritative_locator",
                "schema_binding_id",
                "row_count",
                "membership_rule_id",
                "producing_provenance_id",
                "materialization_authority_id",
                "materialization_destination",
                "materialization_exercised",
                "independent_review_status",
            },
            "P0.1 existing-asset binding",
        )
        for key in (
            "authoritative_locator",
            "schema_binding_id",
            "membership_rule_id",
            "producing_provenance_id",
        ):
            _expect_nonempty_binding(binding[key], f"P0.1 {key}")
        _expect_exact_count(binding["row_count"], 6547, "P0.1 row_count")
        _expect_equal(
            binding["materialization_authority_id"], None, "P0.1 materialization authority"
        )
        _expect_equal(
            binding["materialization_destination"], None, "P0.1 materialization destination"
        )
        _expect_false(binding["materialization_exercised"], "P0.1 materialization_exercised")
        _expect_equal(
            binding["independent_review_status"],
            "INDEPENDENT_REVIEW_PASS",
            "P0.1 independent review",
        )
        return
    if route == "BOUND_UNEXERCISED_MEMBERSHIP_PRESERVING_MATERIALIZATION_AUTHORITY":
        _expect_exact_keys(
            binding,
            {
                "authoritative_source_locator",
                "frozen_membership_rule_id",
                "destination_locator",
                "materialization_authority_id",
                "frozen_member_count",
                "materialization_exercised",
                "member_add_allowed",
                "member_drop_allowed",
                "member_relabel_allowed",
                "member_deduplicate_allowed",
                "member_resample_allowed",
                "observed_data_selection_allowed",
                "independent_review_status",
            },
            "P0.1 materialization-authority binding",
        )
        for key in (
            "authoritative_source_locator",
            "frozen_membership_rule_id",
            "destination_locator",
            "materialization_authority_id",
        ):
            _expect_nonempty_binding(binding[key], f"P0.1 {key}")
        _expect_exact_count(binding["frozen_member_count"], 6547, "P0.1 frozen_member_count")
        for key in (
            "materialization_exercised",
            "member_add_allowed",
            "member_drop_allowed",
            "member_relabel_allowed",
            "member_deduplicate_allowed",
            "member_resample_allowed",
            "observed_data_selection_allowed",
        ):
            _expect_false(binding[key], f"P0.1 {key}")
        _expect_equal(
            binding["independent_review_status"],
            "INDEPENDENT_REVIEW_PASS",
            "P0.1 independent review",
        )
        return
    raise SemanticMismatchError("P0.1 does not bind exactly one frozen alternative")


def _validate_p02(evidence: Mapping[str, Any]) -> None:
    keys = {
        "authority_binding_record_id",
        "attestation_status",
        "bound_input_count",
        "derived_asset_count",
        "attested_input_count",
        "attested_derived_asset_count",
        "unknown_history_count",
        "partial_history_count",
        "inferred_history_count",
        "dataset_level_only_history_count",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.2 evidence")
    _expect_nonempty_binding(evidence["authority_binding_record_id"], "P0.2 authority record")
    _expect_equal(
        evidence["attestation_status"],
        "FULL_PRIOR_ANALYTIC_USE_ATTESTATION_PASS",
        "P0.2 attestation status",
    )
    inputs = _expect_count(evidence["bound_input_count"], "P0.2 bound_input_count", minimum=1)
    derived = _expect_count(evidence["derived_asset_count"], "P0.2 derived_asset_count")
    _expect_exact_count(evidence["attested_input_count"], inputs, "P0.2 attested input coverage")
    _expect_exact_count(
        evidence["attested_derived_asset_count"], derived, "P0.2 attested derived coverage"
    )
    for key in (
        "unknown_history_count",
        "partial_history_count",
        "inferred_history_count",
        "dataset_level_only_history_count",
    ):
        _expect_exact_count(evidence[key], 0, f"P0.2 {key}")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.2 independent review",
    )


def _validate_p03(evidence: Mapping[str, Any]) -> None:
    keys = {
        "authority_binding_record_id",
        "participating_study_count",
        "study_level_binding_count",
        "gse200304_included",
        "all_study_role",
        "untouched_role_count",
        "sealed_role_count",
        "confirmatory_role_count",
        "later_confirmatory_reuse_eligible_count",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.3 evidence")
    _expect_nonempty_binding(evidence["authority_binding_record_id"], "P0.3 authority record")
    study_count = _expect_count(
        evidence["participating_study_count"], "P0.3 participating_study_count", minimum=1
    )
    _expect_exact_count(
        evidence["study_level_binding_count"], study_count, "P0.3 study-level binding coverage"
    )
    _expect_true(evidence["gse200304_included"], "P0.3 gse200304_included")
    _expect_equal(
        evidence["all_study_role"], "EXPOSED_DEVELOPMENT_ONLY", "P0.3 all-study role"
    )
    for key in (
        "untouched_role_count",
        "sealed_role_count",
        "confirmatory_role_count",
        "later_confirmatory_reuse_eligible_count",
    ):
        _expect_exact_count(evidence[key], 0, f"P0.3 {key}")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.3 independent review",
    )


def _validate_p04(evidence: Mapping[str, Any]) -> None:
    keys = {
        "rights_authority_record_id",
        "input_count",
        "internal_processing_authorized_count",
        "training_authorized_count",
        "evaluation_authorized_count",
        "public_access_only_count",
        "redistribution_only_count",
        "permission_basis",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.4 evidence")
    _expect_nonempty_binding(evidence["rights_authority_record_id"], "P0.4 rights authority")
    input_count = _expect_count(evidence["input_count"], "P0.4 input_count", minimum=1)
    for key in (
        "internal_processing_authorized_count",
        "training_authorized_count",
        "evaluation_authorized_count",
    ):
        _expect_exact_count(evidence[key], input_count, f"P0.4 {key}")
    _expect_exact_count(evidence["public_access_only_count"], 0, "P0.4 public_access_only_count")
    _expect_exact_count(evidence["redistribution_only_count"], 0, "P0.4 redistribution_only_count")
    _expect_equal(
        evidence["permission_basis"],
        "EXPLICIT_INTENDED_INTERNAL_PROCESS_TRAIN_EVALUATE",
        "P0.4 permission basis",
    )
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.4 independent review",
    )


def _validate_p05(evidence: Mapping[str, Any]) -> None:
    keys = {
        "row_contract_binding_id",
        "contract_member_count",
        "required_authoritative_fields",
        "missing_required_field_allowed",
        "inferred_identity_allowed",
        "enforcement_binding_id",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.5 evidence")
    _expect_nonempty_binding(evidence["row_contract_binding_id"], "P0.5 row contract")
    _expect_exact_count(evidence["contract_member_count"], 6547, "P0.5 contract_member_count")
    required_fields = {
        "source",
        "candidate",
        "endpoint_transform",
        "endpoint_direction",
        "biological_source_group",
        "context",
        "rights",
        "exposure",
        "membership",
    }
    observed = evidence["required_authoritative_fields"]
    if (
        not isinstance(observed, list)
        or not all(isinstance(item, str) for item in observed)
        or set(observed) != required_fields
        or len(observed) != len(required_fields)
    ):
        raise SemanticMismatchError("P0.5 authoritative field closure differs")
    _expect_false(
        evidence["missing_required_field_allowed"], "P0.5 missing_required_field_allowed"
    )
    _expect_false(evidence["inferred_identity_allowed"], "P0.5 inferred_identity_allowed")
    _expect_nonempty_binding(evidence["enforcement_binding_id"], "P0.5 enforcement binding")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.5 independent review",
    )


def _validate_p06(evidence: Mapping[str, Any]) -> None:
    keys = {
        "route_binding_id",
        "route",
        "initialization",
        "foundation_input_count",
        "warm_start_input_count",
        "resumed_checkpoint_count",
        "previously_failed_checkpoint_count",
        "external_learned_input_count",
        "checkpoint_reads_before_first_update",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.6 evidence")
    _expect_nonempty_binding(evidence["route_binding_id"], "P0.6 route binding")
    _expect_equal(evidence["route"], "SCRATCH_ONLY_NO_EXTERNAL_LEARNED_INPUTS", "P0.6 route")
    _expect_equal(evidence["initialization"], "RANDOM_INITIALIZATION", "P0.6 initialization")
    for key in (
        "foundation_input_count",
        "warm_start_input_count",
        "resumed_checkpoint_count",
        "previously_failed_checkpoint_count",
        "external_learned_input_count",
        "checkpoint_reads_before_first_update",
    ):
        _expect_exact_count(evidence[key], 0, f"P0.6 {key}")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.6 independent review",
    )


def _validate_p07(evidence: Mapping[str, Any]) -> None:
    keys = {
        "split_authority_record_id",
        "split_binding_id",
        "frozen_before_label_bearing_access",
        "outcome_blind",
        "assignment_unit",
        "source_group_disjoint",
        "known_duplicate_disjoint",
        "membership_adjustable_from_model_results",
        "membership_adjustable_from_endpoint_results",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.7 evidence")
    for key in ("split_authority_record_id", "split_binding_id"):
        _expect_nonempty_binding(evidence[key], f"P0.7 {key}")
    _expect_true(
        evidence["frozen_before_label_bearing_access"],
        "P0.7 frozen_before_label_bearing_access",
    )
    _expect_true(evidence["outcome_blind"], "P0.7 outcome_blind")
    _expect_equal(
        evidence["assignment_unit"], "SOURCE_GROUP_WITH_KNOWN_DUPLICATE_COMPONENTS", "P0.7 assignment unit"
    )
    _expect_true(evidence["source_group_disjoint"], "P0.7 source_group_disjoint")
    _expect_true(evidence["known_duplicate_disjoint"], "P0.7 known_duplicate_disjoint")
    _expect_false(
        evidence["membership_adjustable_from_model_results"],
        "P0.7 model-result adjustability",
    )
    _expect_false(
        evidence["membership_adjustable_from_endpoint_results"],
        "P0.7 endpoint-result adjustability",
    )
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.7 independent review",
    )


def _validate_p08(evidence: Mapping[str, Any]) -> None:
    keys = {
        "policy_binding_id",
        "run_count",
        "seed",
        "architecture_binding_id",
        "optimizer_binding_id",
        "learning_rate",
        "schedule_binding_id",
        "compute_budget_binding_id",
        "checkpoint_emission_retention_rule_id",
        "terminal_checkpoint_rule_id",
        "stop_rule_id",
        "cuda_device_ownership_rule_id",
        "aggregate_metric_set_id",
        "independent_exact_reference_binding_id",
        "alternative_count_after_results",
        "selection_after_results_allowed",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.8 evidence")
    for key in (
        "policy_binding_id",
        "architecture_binding_id",
        "optimizer_binding_id",
        "schedule_binding_id",
        "compute_budget_binding_id",
        "checkpoint_emission_retention_rule_id",
        "terminal_checkpoint_rule_id",
        "stop_rule_id",
        "cuda_device_ownership_rule_id",
        "aggregate_metric_set_id",
        "independent_exact_reference_binding_id",
    ):
        _expect_nonempty_binding(evidence[key], f"P0.8 {key}")
    _expect_exact_count(evidence["run_count"], 1, "P0.8 run_count")
    _expect_count(evidence["seed"], "P0.8 seed")
    _expect_positive_finite(evidence["learning_rate"], "P0.8 learning_rate")
    _expect_exact_count(evidence["alternative_count_after_results"], 0, "P0.8 alternatives")
    _expect_false(evidence["selection_after_results_allowed"], "P0.8 result selection")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.8 independent review",
    )


def _validate_p09(evidence: Mapping[str, Any]) -> None:
    keys = {
        "gate_bundle_binding_id",
        "checks",
        "deferred_check_count",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.9 evidence")
    _expect_nonempty_binding(evidence["gate_bundle_binding_id"], "P0.9 gate bundle")
    checks = evidence["checks"]
    if not isinstance(checks, Mapping):
        raise PartialGroupError("P0.9 checks must be an object")
    expected_checks = {
        "LEGAL_CTMC_PRODUCTION_INTERFACE",
        "STOP_SUPPORT_FLOOR_ALIAS_BUDGET",
        "BASE_RECOVERY",
        "LEARNED_POTENTIAL_APPROXIMATION_ERROR",
        "LEGALITY",
        "TRAJECTORY_REPLAY",
        "PROVENANCE",
        "FAILURE_BUNDLE",
    }
    _expect_exact_keys(checks, expected_checks, "P0.9 check bundle")
    for name in expected_checks:
        check = checks[name]
        if not isinstance(check, Mapping):
            raise PartialGroupError(f"P0.9 {name} must be an object")
        _expect_exact_keys(check, {"binding_id", "binding_status"}, f"P0.9 {name}")
        _expect_nonempty_binding(check["binding_id"], f"P0.9 {name} binding")
        _expect_equal(
            check["binding_status"],
            "BOUND_EXECUTABLE_NOT_RUN",
            f"P0.9 {name} binding status",
        )
    _expect_exact_count(evidence["deferred_check_count"], 0, "P0.9 deferred_check_count")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.9 independent review",
    )


def _validate_p10(evidence: Mapping[str, Any]) -> None:
    keys = {
        "parent_g0_candidate_id",
        "successor_implementation_id",
        "review_status",
        "successor_state",
        "future_activation_scope",
        "bindings",
        "draft_implementation",
        "partial_binding",
        "configurable_placeholder_count",
    }
    _expect_exact_keys(evidence, keys, "P0.10 evidence")
    _expect_equal(
        evidence["parent_g0_candidate_id"],
        "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_G0_IMPLEMENTATION_CANDIDATE_V1",
        "P0.10 parent G0 candidate",
    )
    _expect_nonempty_binding(
        evidence["successor_implementation_id"], "P0.10 successor implementation"
    )
    _expect_equal(evidence["review_status"], "INDEPENDENT_REVIEW_PASS", "P0.10 review")
    _expect_equal(
        evidence["successor_state"],
        "REVIEWED_READY_FOR_FUTURE_ACTIVE_SUCCESSOR",
        "P0.10 successor state",
    )
    _expect_equal(
        evidence["future_activation_scope"],
        "ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY",
        "P0.10 activation scope",
    )
    bindings = evidence["bindings"]
    if not isinstance(bindings, Mapping):
        raise PartialGroupError("P0.10 bindings must be an object")
    expected_bindings = {
        "executable",
        "configuration",
        "exact_reference",
        "environment",
        "input_branch_from_p0_1",
        "split",
        "seed",
        "optimizer",
        "budget",
        "device",
        "outputs",
        "stop_and_failure_destinations",
    }
    _expect_exact_keys(bindings, expected_bindings, "P0.10 successor bindings")
    for name, value in bindings.items():
        _expect_nonempty_binding(value, f"P0.10 {name} binding")
    _expect_false(evidence["draft_implementation"], "P0.10 draft_implementation")
    _expect_false(evidence["partial_binding"], "P0.10 partial_binding")
    _expect_exact_count(
        evidence["configurable_placeholder_count"],
        0,
        "P0.10 configurable_placeholder_count",
    )


def _validate_p11(evidence: Mapping[str, Any]) -> None:
    keys = {
        "lock_binding_id",
        "pre_p0_lock_state",
        "p0_01_to_p0_10_required_pass_count",
        "conditional_one_run_atomic_unlock",
        "conditional_unlock_scope",
        "persistent_locks",
        "independent_review_status",
    }
    _expect_exact_keys(evidence, keys, "P0.11 evidence")
    _expect_nonempty_binding(evidence["lock_binding_id"], "P0.11 lock binding")
    pre = evidence["pre_p0_lock_state"]
    if not isinstance(pre, Mapping):
        raise PartialGroupError("P0.11 pre-P0 locks must be an object")
    _expect_exact_keys(
        pre,
        {"training_allowed", "gpu_work_allowed", "parameter_updates_allowed"},
        "P0.11 pre-P0 locks",
    )
    if any(value is not False for value in pre.values()):
        raise SemanticMismatchError("P0.11 pre-P0 locks are not all false")
    _expect_exact_count(
        evidence["p0_01_to_p0_10_required_pass_count"],
        10,
        "P0.11 required preceding passes",
    )
    _expect_true(
        evidence["conditional_one_run_atomic_unlock"],
        "P0.11 conditional atomic unlock",
    )
    _expect_equal(
        evidence["conditional_unlock_scope"],
        "ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY",
        "P0.11 conditional scope",
    )
    persistent = evidence["persistent_locks"]
    if not isinstance(persistent, Mapping):
        raise PartialGroupError("P0.11 persistent locks must be an object")
    _expect_exact_keys(
        persistent,
        {
            "model_selection_allowed",
            "qualification_change_allowed",
            "credit_change_allowed",
            "canonical_mutation_allowed",
            "a6_pass_allowed",
            "l3_claim_allowed",
            "a7_unlock_allowed",
            "private_or_sealed_access_allowed",
            "scientific_claim_allowed",
        },
        "P0.11 persistent locks",
    )
    if any(value is not False for value in persistent.values()):
        raise SemanticMismatchError("P0.11 persistent locks are not all false")
    _expect_equal(
        evidence["independent_review_status"],
        "INDEPENDENT_REVIEW_PASS",
        "P0.11 independent review",
    )


GROUP_VALIDATORS: Mapping[str, Callable[[Mapping[str, Any]], None]] = {
    "P0.1": _validate_p01,
    "P0.2": _validate_p02,
    "P0.3": _validate_p03,
    "P0.4": _validate_p04,
    "P0.5": _validate_p05,
    "P0.6": _validate_p06,
    "P0.7": _validate_p07,
    "P0.8": _validate_p08,
    "P0.9": _validate_p09,
    "P0.10": _validate_p10,
    "P0.11": _validate_p11,
}


def _all_gate_statuses(status: str) -> list[dict[str, str]]:
    return [
        {"gate_id": gate_id, "gate_name": gate_name, "status": status}
        for gate_id, gate_name in P0_GROUPS
    ]


def _result(gate_statuses: list[dict[str, str]]) -> dict[str, Any]:
    all_pass = len(gate_statuses) == len(P0_GROUPS) and all(
        item["status"] == "PASS" for item in gate_statuses
    )
    return {
        "result_status": PASS_STATUS if all_pass else FAILURE_STATUS,
        "authority_status": AUTHORITY_STATUS,
        "gate_statuses": gate_statuses,
    }


def evaluate_zero_step_p0(
    package: Mapping[str, Any],
    *,
    candidate_config: Mapping[str, Any] | None = None,
    touchpoints: ForbiddenTouchpointSentinel | None = None,
) -> dict[str, Any]:
    """Evaluate the exact eleven metadata groups without operational access."""

    sentinel = touchpoints or ForbiddenTouchpointSentinel()
    sentinel.assert_zero()
    if candidate_config is None:
        candidate_config = load_protocol()
    else:
        try:
            validate_protocol(candidate_config)
        except (PartialGroupError, SemanticMismatchError) as exc:
            raise ProtocolError(str(exc)) from exc

    allowed_declared_statuses = set(candidate_config["allowed_declared_statuses"])

    if not isinstance(package, Mapping):
        result = _result(_all_gate_statuses("FAIL_CLOSED_SUBMISSION_SCOPE_MISMATCH"))
        sentinel.assert_zero()
        return result
    try:
        _expect_exact_keys(package, {"schema_version", "package_role", "groups"}, "package")
    except PartialGroupError:
        result = _result(_all_gate_statuses("FAIL_CLOSED_SUBMISSION_SCOPE_MISMATCH"))
        sentinel.assert_zero()
        return result
    if package["schema_version"] != PACKAGE_SCHEMA or package["package_role"] != PACKAGE_ROLE:
        result = _result(_all_gate_statuses("FAIL_CLOSED_SUBMISSION_SCOPE_MISMATCH"))
        sentinel.assert_zero()
        return result

    groups = package["groups"]
    if not isinstance(groups, Mapping):
        result = _result(_all_gate_statuses("FAIL_CLOSED_SUBMISSION_SCOPE_MISMATCH"))
        sentinel.assert_zero()
        return result
    expected_ids = {gate_id for gate_id, _ in P0_GROUPS}
    observed_ids = set(groups)
    if observed_ids - expected_ids:
        result = _result(_all_gate_statuses("FAIL_CLOSED_UNEXPECTED_GROUP_SCOPE"))
        sentinel.assert_zero()
        return result

    statuses: list[dict[str, str]] = []
    for gate_id, gate_name in P0_GROUPS:
        if gate_id not in groups:
            status = "FAIL_CLOSED_MISSING_GROUP"
        else:
            group = groups[gate_id]
            if not isinstance(group, Mapping):
                status = "FAIL_CLOSED_PARTIAL_GROUP"
            else:
                try:
                    _expect_exact_keys(
                        group,
                        {"declared_status", "evidence"},
                        f"{gate_id} group",
                    )
                    declared_status = group["declared_status"]
                    if declared_status not in allowed_declared_statuses:
                        status = "FAIL_CLOSED_DECLARED_STATUS_UNRECOGNIZED"
                    elif declared_status != "PASS":
                        status = declared_status
                    elif not isinstance(group["evidence"], Mapping):
                        status = "FAIL_CLOSED_PARTIAL_GROUP"
                    else:
                        GROUP_VALIDATORS[gate_id](group["evidence"])
                        status = "PASS"
                except PartialGroupError:
                    status = "FAIL_CLOSED_PARTIAL_GROUP"
                except SemanticMismatchError:
                    status = "FAIL_CLOSED_SEMANTIC_MISMATCH"
        statuses.append({"gate_id": gate_id, "gate_name": gate_name, "status": status})

    result = _result(statuses)
    sentinel.assert_zero()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ProtocolError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_repository_binding(
    protocol: Mapping[str, Any], repo_root: Path = REPO_ROOT
) -> dict[str, str]:
    """Require clean, pushed I/config-only-B exact lineage before publication."""

    binding = protocol["implementation_binding"]
    if binding["status"] != BOUND:
        raise ProtocolError("DEC026 implementation is not config-only bound")
    if _git(repo_root, "status", "--porcelain"):
        raise ProtocolError("repository worktree is not clean")
    head = _git(repo_root, "rev-parse", "HEAD")
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    if head != upstream:
        raise ProtocolError("DEC026 binding HEAD is not pushed upstream")
    implementation = binding["implementation_commit"]
    if _git(repo_root, "rev-parse", "HEAD^") != implementation:
        raise ProtocolError("binding HEAD is not the direct child of DEC026 I")
    if _git(repo_root, "rev-parse", f"{implementation}^") != binding[
        "implementation_expected_parent"
    ]:
        raise ProtocolError("DEC026 I parent differs from frozen predecessor")

    implementation_paths = _git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        implementation,
    ).splitlines()
    if sorted(implementation_paths) != sorted(
        binding["implementation_exact_changed_paths"]
    ):
        raise ProtocolError("DEC026 I is not exact3")
    binding_paths = _git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        head,
    ).splitlines()
    if binding_paths != binding["binding_exact_changed_paths"]:
        raise ProtocolError("DEC026 B is not config-only")

    script_path = repo_root / binding["implementation_script_path"]
    test_path = repo_root / binding["implementation_test_path"]
    if _sha256_file(script_path) != binding["implementation_script_sha256"]:
        raise ProtocolError("DEC026 script digest differs from binding")
    if _sha256_file(test_path) != binding["implementation_test_sha256"]:
        raise ProtocolError("DEC026 test digest differs from binding")
    return {"implementation_commit": implementation, "binding_commit": head}


def _gate_counts(result: Mapping[str, Any]) -> dict[str, int]:
    statuses = [item["status"] for item in result["gate_statuses"]]
    return {
        "pass": sum(status == "PASS" for status in statuses),
        "fail_closed": sum(status.startswith("FAIL_CLOSED") for status in statuses),
        "unknown_not_asserted": sum(status == UNKNOWN for status in statuses),
        "total": len(statuses),
    }


def build_aggregate_record(
    protocol: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact aggregate-only report payload."""

    all_pass = result["result_status"] == PASS_STATUS
    return {
        "schema_version": "route_a_v3_dec026_zero_step_p0_record.v1",
        "protocol_id": PROTOCOL_ID,
        "decision_id": "V3-DEC-026",
        "result_status": result["result_status"],
        "gate_statuses": result["gate_statuses"],
        "gate_counts": _gate_counts(result),
        "all_eleven_pass": all_pass,
        "g1_one_run_eligible": all_pass,
        "g1_launched": False,
        "retained_locks": protocol["retained_locks"],
        "scientific_state": protocol["scientific_state"],
        "forbidden_touchpoint_counts": {
            name: 0 for name in FORBIDDEN_TOUCHPOINTS
        },
    }


def write_aggregate_record(output_dir: Path, record: Mapping[str, Any]) -> Path:
    """Create one new run directory containing exactly the one permitted file."""

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / REPORT_FILENAME
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")
    final_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if final_files != [REPORT_FILENAME]:
        raise ProtocolError("zero-step output directory is not exact-one-file")
    return report_path


def run_zero_step(
    output_dir: Path,
    *,
    protocol_path: Path = CONFIG_PATH,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], Path]:
    protocol = load_protocol(protocol_path)
    verify_repository_binding(protocol, repo_root)
    result = evaluate_zero_step_p0(
        protocol["current_submission"],
        candidate_config=protocol,
    )
    record = build_aggregate_record(protocol, result)
    report_path = write_aggregate_record(output_dir, record)
    return record, report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CONFIG_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record, report_path = run_zero_step(
        args.output_dir,
        protocol_path=args.protocol,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "status": record["result_status"],
                "gate_counts": record["gate_counts"],
                "g1_one_run_eligible": record["g1_one_run_eligible"],
                "report_path": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
