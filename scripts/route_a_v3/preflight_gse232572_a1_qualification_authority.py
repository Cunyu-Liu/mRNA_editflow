#!/usr/bin/env python3
"""Aggregate-only GSE232572 A1 qualification-authority preflight.

This producer verifies the current Route-A V3 authority documents and one
registered public aggregate materialization report.  It emits one public JSON
report whose only valid decision is ``BLOCKED_MISSING_EXTERNAL_AUTHORITY``.

The preflight is deliberately not a qualifier.  It does not read row records,
sequence values, or effect values; it does not start split, leakage, power,
canonicalization, training, model-selection, or next-phase work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = (
    "route_a_v3_gse232572_a1_qualification_authority_preflight.v1"
)
PROTOCOL_ID = "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1"
PROTOCOL_BASENAME = (
    "route_a_v3_gse232572_a1_qualification_authority_preflight_v1.json"
)
REPORT_FILENAME = "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT.json"
DATASET_ID = "GSE232572"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)

EXPECTED_EXACT3 = (
    f"configs/{PROTOCOL_BASENAME}",
    "scripts/route_a_v3/preflight_gse232572_a1_qualification_authority.py",
    "tests/route_a_v3/test_preflight_gse232572_a1_qualification_authority.py",
)

EXPECTED_AUTHORITIES = (
    (
        "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
        "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
        "SOLE_CURRENT_CONTRACT",
    ),
    (
        "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec019.yaml",
        "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca",
        "DEC019_FROZEN_USER_AUTHORIZED_A1_QUALIFICATION_POLICY",
    ),
    (
        "configs/route_a_v3_a1_qualification.json",
        "fe3f7736c1f64b362ebda683ca571fc1a84e1fff36aed3a9ae67272665ba2343",
        "CURRENT_A1_QUALIFICATION_AUTHORITY",
    ),
    (
        "docs/execution/route_a_v3_data_role_registry.yaml",
        "4d14ebd1a6adc04a344165f775df8586ef9f8f0461fdcac08649d0644d9956f2",
        "CURRENT_DATA_ROLE_REGISTRY",
    ),
    (
        "docs/execution/route_a_v3_split_registry.yaml",
        "2764d471c09a27da889b690cac317ac582bf9f25b79b6a34ac491f2e0b434929",
        "CURRENT_SPLIT_REGISTRY",
    ),
)

EXPECTED_REPORT_PATH = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE232572/"
    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_20260812T234243P0800/"
    "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
)
EXPECTED_REPORT_BYTES = 3601
EXPECTED_REPORT_SHA256 = (
    "007a70b9b43b71cbc21b2614473126e53ab760df824ac9d95cb14304cc647ef3"
)

EXPECTED_PASS_IDS = (
    "AUTHORIZED_PRIMARY_MEASUREMENT_AND_PAPER_FAITHFUL_LNFC",
    "PUBLIC_RAW_LOCATOR_AND_LINEAGE",
    "SOURCE_CANDIDATE_CONTEXT_GROUP_RECONSTRUCTION",
)

EXPECTED_BLOCKERS = (
    (
        "DATASET_SPECIFIC_QUALIFICATION_AUTHORITY_AND_CONSUMER_USER_APPROVAL",
        "MISSING_EXTERNAL_AUTHORITY",
    ),
    ("OWNER_PROJECT_USE_AND_EXPOSURE_ATTESTATION", "MISSING_EXTERNAL_AUTHORITY"),
    ("CHECKPOINT_SPECIFIC_EXACT_AND_NEAR_EXPOSURE_AUDIT", UNKNOWN),
    ("ASSET_LEVEL_QUALIFICATION_USE_RIGHTS_AND_RELEASE_SCOPE", "UNKNOWN_BLOCKED"),
    ("EXECUTABLE_BIOLOGICAL_SOURCE_GROUP_AUTHORITY", "NOT_CLOSED"),
    ("REPLICATE_OR_VALID_STANDARD_ERROR", "NOT_CLOSED"),
    ("ELIGIBLE_MULTI_CANDIDATE_POOLS", UNKNOWN),
    (
        "A1_GROUP_SPLIT_NEAR_DUPLICATE_GRAPH_SALT_AND_ZERO_LEAKAGE",
        "NOT_RUN",
    ),
    ("PREFROZEN_GROUP_EFFECTIVE_N_POWER_AND_FULL_CI_WIDTH", "NOT_RUN"),
    ("REQUIRED_QUALIFICATION_REPORT_FIELDS", "INCOMPLETE"),
    ("V3_CANONICAL_ADMISSION", "BLOCKED"),
    ("FINAL_CANONICAL_QUALIFIER_AND_ADJUDICATION", "NOT_RUN"),
)

EXPECTED_EXTERNAL_REQUESTS = (
    "CHECKPOINT_SPECIFIC_EXPOSURE_AUTHORITY",
    "ASSET_LEVEL_QUALIFICATION_USE_RIGHTS",
    "PRIMARY_ENDPOINT_UNCERTAINTY_AUTHORITY",
    "USER_APPROVAL_FOR_DATASET_SPECIFIC_QUALIFICATION_AUTHORITY",
)


class PreflightError(RuntimeError):
    """Base class for a fail-closed preflight error."""


class ProtocolError(PreflightError):
    """The protocol or implementation binding is invalid."""


class BindingNotFrozen(ProtocolError):
    """The UNKNOWN-I to config-only-B lifecycle is incomplete."""


class AuthorityError(PreflightError):
    """A bound public authority input is absent or has drifted."""


class MaterializationReportError(PreflightError):
    """The registered public aggregate report is absent or has drifted."""


class OutputError(PreflightError):
    """The exclusive single-output contract cannot be satisfied."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreflightError("output is not finite JSON") from exc


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def _read_bytes(path: Path, *, label: str, error_type: type[PreflightError]) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise error_type(f"cannot read {label}: {path}") from exc


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be a list")
    return value


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _expect(mapping: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for key, expected_value in expected.items():
        if mapping.get(key) != expected_value:
            raise ProtocolError(f"{label}.{key} differs from the frozen value")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    _expect(
        protocol,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_id": DATASET_ID,
            "protocol_status": "PREFLIGHT_ONLY_AGGREGATE_ONLY_STOP_BEFORE_ROW_ACCESS",
        },
        label="protocol",
    )

    binding = _mapping(
        protocol.get("implementation_binding"), label="implementation_binding"
    )
    _expect(
        binding,
        {
            "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
            "base_commit": "13baa39e87406b5bc81b7e236cee637f694bfd0f",
            "implementation_script_path": EXPECTED_EXACT3[1],
            "implementation_test_path": EXPECTED_EXACT3[2],
            "unknown_to_bound_scalar_paths": [
                f"implementation_binding.{field}"
                for field in UNKNOWN_BINDING_SCALARS
            ],
            "implementation_commit_exact_changed_paths": list(EXPECTED_EXACT3),
            "binding_commit_exact_changed_paths": [EXPECTED_EXACT3[0]],
        },
        label="implementation_binding",
    )
    binding_values = [binding.get(field) for field in UNKNOWN_BINDING_SCALARS]
    if all(value == UNKNOWN for value in binding_values):
        pass
    elif (
        binding.get("status") == BOUND
        and _is_hex(binding.get("implementation_commit"), 40)
        and _is_hex(binding.get("implementation_script_sha256"), 64)
        and _is_hex(binding.get("implementation_test_sha256"), 64)
    ):
        pass
    else:
        raise ProtocolError("implementation binding is neither UNKNOWN-I nor BOUND-B")

    authorities = _list(protocol.get("authority_inputs"), label="authority_inputs")
    observed_authorities = []
    for item in authorities:
        item_mapping = _mapping(item, label="authority input")
        observed_authorities.append(
            (
                item_mapping.get("path"),
                item_mapping.get("sha256"),
                item_mapping.get("role"),
            )
        )
    if tuple(observed_authorities) != EXPECTED_AUTHORITIES:
        raise ProtocolError("authority_inputs differ from the exact public authority set")

    report_identity = _mapping(
        protocol.get("materialization_report_identity"),
        label="materialization_report_identity",
    )
    _expect(
        report_identity,
        {
            "absolute_path": EXPECTED_REPORT_PATH,
            "bytes": EXPECTED_REPORT_BYTES,
            "sha256": EXPECTED_REPORT_SHA256,
            "role": "PUBLIC_AGGREGATE_ONLY_DEVELOPMENT_MATERIALIZATION_REPORT",
        },
        label="materialization_report_identity",
    )

    ledger = _mapping(
        protocol.get("registered_public_ledger_facts"),
        label="registered_public_ledger_facts",
    )
    _expect(
        ledger,
        {
            "runtime_event_id": "A1-EVT-048",
            "runtime_event_name": (
                "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REGISTERED_"
                "QUALIFICATION_GATE_UNCHANGED"
            ),
            "publication_status": "PUBLISHED_VERIFIED",
            "runtime_output_count_after": 203,
            "outer_recovery_gate_summary": {"PASS": 7, UNKNOWN: 1},
            "outer_recovery_summary_is_dataset_qualification": False,
            "outer_recovery_unknown_is_the_only_qualification_blocker": False,
            "primary_measurement_route": "AUTHOR_PUBLISHED_PRIMARY_ENDPOINT",
            "paper_faithful_transform": "LN_FOLD_CHANGE",
            "public_raw_locator_and_lineage_status": "REGISTERED_PUBLIC_LINEAGE_EXACT",
            "development_reconstruction_status": "DEVELOPMENT_PASS_NOT_CANONICAL",
            "sequence_exposure": "SEQUENCE_EXPOSED",
            "label_exposure": "LABEL_EXPOSED",
            "untouched_confirmatory": False,
            "public_replicate_count": 3,
            "primary_label_standard_error": None,
        },
        label="registered_public_ledger_facts",
    )

    frozen = _mapping(
        protocol.get("frozen_materialization_facts"),
        label="frozen_materialization_facts",
    )
    _expect(
        frozen,
        {
            "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
            "scientific_disposition": (
                "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
            ),
            "published_universe_row_count": 11929,
            "accepted_pair_complete_raw_endpoint_count": 8068,
            "rejected_published_row_count": 3861,
            "schema_valid_development_record_count": 8068,
            "canonical_record_count": 0,
            "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
            "qualified": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_allowed": False,
            "license_boundary": {
                "private_derivative_use_fact": (
                    "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED"
                ),
                "public_redistribution_status": (
                    "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
                ),
                "redistribution_allowed": False,
                "row_license_status": "UNKNOWN_BLOCKED",
            },
        },
        label="frozen_materialization_facts",
    )

    passes = _list(
        protocol.get("registered_aggregate_passes"),
        label="registered_aggregate_passes",
    )
    if [item.get("gate_id") for item in passes if isinstance(item, dict)] != list(
        EXPECTED_PASS_IDS
    ) or len(passes) != len(EXPECTED_PASS_IDS):
        raise ProtocolError("registered aggregate PASS set must remain exact3")
    if any(
        not isinstance(item, dict)
        or item.get("status") != "PASS_FROM_REGISTERED_AGGREGATE"
        for item in passes
    ):
        raise ProtocolError("registered aggregate PASS statuses differ")
    if passes[2].get("scope") != "DEVELOPMENT_PASS_NOT_CANONICAL":
        raise ProtocolError("development reconstruction may not become canonical")

    blockers = _list(
        protocol.get("qualification_blockers"), label="qualification_blockers"
    )
    observed_blockers = [
        (item.get("gate_id"), item.get("status"))
        for item in blockers
        if isinstance(item, dict)
    ]
    if tuple(observed_blockers) != EXPECTED_BLOCKERS or len(blockers) != len(
        EXPECTED_BLOCKERS
    ):
        raise ProtocolError("qualification blocker closure differs")
    rights = blockers[3]
    if (
        rights.get("private_derivative_use_is_qualification_use_grant") is not False
        or rights.get("recommended_future_scope") != "PRIVATE_CANONICAL_ONLY"
        or rights.get("recommended_future_scope_approved") is not False
    ):
        raise ProtocolError("qualification-use rights boundary differs")
    replicate = blockers[5]
    if (
        replicate.get("public_replicate_count") != 3
        or replicate.get("primary_label_standard_error") is not None
    ):
        raise ProtocolError("replicate/primary-label uncertainty boundary differs")
    if blockers[6].get(
        "accepted_pair_count_establishes_at_least_three_candidate_pools"
    ) is not False:
        raise ProtocolError("pair count may not imply candidate-pool eligibility")

    requests = _list(
        protocol.get("minimum_external_authority_requests"),
        label="minimum_external_authority_requests",
    )
    if [item.get("request_id") for item in requests if isinstance(item, dict)] != list(
        EXPECTED_EXTERNAL_REQUESTS
    ) or len(requests) != len(EXPECTED_EXTERNAL_REQUESTS):
        raise ProtocolError("minimum external authority request set differs")

    contribution = _mapping(
        protocol.get("future_contribution_boundary"),
        label="future_contribution_boundary",
    )
    _expect(
        contribution,
        {
            "maximum_if_fully_qualified": {
                "ordinary": 1,
                "a1": 1,
                "true_a2": 0,
            },
            "authorization_status": "NOT_AUTHORIZED",
            "current_contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        },
        label="future_contribution_boundary",
    )

    inputs = _mapping(protocol.get("input_contract"), label="input_contract")
    if inputs.get("aggregate_only") is not True:
        raise ProtocolError("input must remain aggregate-only")
    for key in (
        "row_records_allowed",
        "sequence_values_allowed",
        "effect_values_allowed",
        "training_allowed",
        "model_selection_allowed",
        "split_execution_allowed",
        "power_execution_allowed",
        "canonical_materialization_allowed",
        "qualifier_execution_allowed",
    ):
        if inputs.get(key) is not False:
            raise ProtocolError(f"input_contract.{key} must remain false")

    output = _mapping(protocol.get("output_contract"), label="output_contract")
    _expect(
        output,
        {
            "filename": REPORT_FILENAME,
            "single_public_aggregate_output_only": True,
            "exclusive_new_output_directory_required": True,
            "overall_decision": "BLOCKED_MISSING_EXTERNAL_AUTHORITY",
            "terminal_status": (
                "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION"
            ),
            "qualified": False,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_allowed": False,
        },
        label="output_contract",
    )


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename differs")
    payload = _read_bytes(path, label="protocol", error_type=ProtocolError)
    protocol = _json_object(payload, label="protocol")
    _validate_protocol(protocol)
    return protocol, payload


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for binding audit") from exc
    if completed.returncode != 0:
        raise ProtocolError("git binding audit failed")
    return completed.stdout.strip()


def _normalise_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    binding = result["implementation_binding"]
    for field in UNKNOWN_BINDING_SCALARS:
        binding[field] = UNKNOWN
    return result


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_payload: bytes,
    repo_root: Path,
) -> dict[str, str]:
    binding = protocol["implementation_binding"]
    if any(binding.get(field) == UNKNOWN for field in UNKNOWN_BINDING_SCALARS):
        raise BindingNotFrozen(
            "implementation binding remains UNKNOWN; config-only-B is required"
        )
    if binding.get("status") != BOUND:
        raise BindingNotFrozen("implementation binding is not BOUND")

    expected_protocol_path = (repo_root / EXPECTED_EXACT3[0]).resolve()
    expected_script_path = (repo_root / EXPECTED_EXACT3[1]).resolve()
    if protocol_path.resolve() != expected_protocol_path:
        raise ProtocolError("executed protocol path is not the bound repository path")
    if Path(__file__).resolve() != expected_script_path:
        raise ProtocolError("executed script path is not the bound repository path")

    working_payloads: dict[str, bytes] = {}
    for path_key, sha_key in (
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        relative_path = str(binding[path_key])
        working_payload = _read_bytes(
            repo_root / relative_path,
            label=f"bound working file {relative_path}",
            error_type=ProtocolError,
        )
        if _sha256(working_payload) != binding[sha_key]:
            raise ProtocolError(f"working {relative_path} hash differs from I binding")
        working_payloads[relative_path] = working_payload

    binding_commit = _run_git(repo_root, "rev-parse", "HEAD")
    upstream_commit = _run_git(repo_root, "rev-parse", "@{upstream}")
    if upstream_commit != binding_commit:
        raise ProtocolError("HEAD differs from the configured upstream")
    tracked_status = _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status:
        raise ProtocolError("tracked worktree or index is not clean")

    implementation_commit = str(binding["implementation_commit"])
    base_commit = str(binding["base_commit"])
    if _run_git(repo_root, "rev-parse", f"{binding_commit}^") != implementation_commit:
        raise ProtocolError("B is not the direct child of I")
    if _run_git(repo_root, "rev-parse", f"{implementation_commit}^") != base_commit:
        raise ProtocolError("I is not the direct child of the frozen base")

    implementation_paths = tuple(
        line
        for line in _run_git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            implementation_commit,
        ).splitlines()
        if line
    )
    if implementation_paths != EXPECTED_EXACT3:
        raise ProtocolError("I changed paths other than exact3")
    binding_paths = tuple(
        line
        for line in _run_git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            binding_commit,
        ).splitlines()
        if line
    )
    if binding_paths != (EXPECTED_EXACT3[0],):
        raise ProtocolError("B is not config-only")

    head_protocol_payload = _run_git(
        repo_root, "show", f"{binding_commit}:{EXPECTED_EXACT3[0]}"
    ).encode("utf-8")
    if not head_protocol_payload.endswith(b"\n"):
        head_protocol_payload += b"\n"
    if head_protocol_payload != protocol_payload:
        raise ProtocolError("working protocol bytes differ from B")
    implementation_protocol = _json_object(
        _run_git(
            repo_root, "show", f"{implementation_commit}:{EXPECTED_EXACT3[0]}"
        ).encode("utf-8"),
        label="I protocol",
    )
    _validate_protocol(implementation_protocol)
    if _normalise_binding(implementation_protocol) != _normalise_binding(protocol):
        raise ProtocolError("B changed protocol semantics beyond the four scalars")

    for path_key, sha_key in (
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        relative_path = str(binding[path_key])
        implementation_payload = _run_git(
            repo_root, "show", f"{implementation_commit}:{relative_path}"
        ).encode("utf-8")
        if not implementation_payload.endswith(b"\n"):
            implementation_payload += b"\n"
        if _sha256(implementation_payload) != binding[sha_key]:
            raise ProtocolError(f"{relative_path} hash differs from I binding")
        head_payload = _run_git(
            repo_root, "show", f"{binding_commit}:{relative_path}"
        ).encode("utf-8")
        if not head_payload.endswith(b"\n"):
            head_payload += b"\n"
        if head_payload != implementation_payload:
            raise ProtocolError(f"{relative_path} changed in B")
        working_payload = working_payloads[relative_path]
        if working_payload != implementation_payload:
            raise ProtocolError(f"working {relative_path} bytes differ from I")

    return {
        "status": "BOUND_CONFIG_ONLY_LIFECYCLE_VERIFIED",
        "base_commit": base_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
    }


def _default_authority_auditor(
    protocol: Mapping[str, Any], repo_root: Path
) -> list[dict[str, str]]:
    verified = []
    for item in protocol["authority_inputs"]:
        relative_path = str(item["path"])
        payload = _read_bytes(
            repo_root / relative_path,
            label=f"authority {item['role']}",
            error_type=AuthorityError,
        )
        observed_sha256 = _sha256(payload)
        if observed_sha256 != item["sha256"]:
            raise AuthorityError(f"authority identity drifted: {relative_path}")
        verified.append(
            {
                "path": relative_path,
                "role": str(item["role"]),
                "sha256": observed_sha256,
            }
        )
    return verified


def _validate_materialization_report(report: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": "1.0.0",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "dataset_id": DATASET_ID,
        "study_id": DATASET_ID,
        "status": "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED",
        "scientific_disposition": (
            "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
        ),
        "published_universe_row_count": 11929,
        "accepted_pair_complete_raw_endpoint_count": 8068,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "rejected_published_row_count": 3861,
        "schema_valid_development_record_count": 8068,
        "canonical_record_count": 0,
        "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "license_boundary": {
            "private_derivative_use_fact": "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED",
            "public_redistribution_status": (
                "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
            ),
            "redistribution_allowed": False,
            "row_license_status": "UNKNOWN_BLOCKED",
            "verified_at_semantics": (
                "PUBLIC_RECOVERY_REPORT_RECORDED_AT_IS_STATUS_OBSERVATION_"
                "NOT_A_LICENSE_GRANT"
            ),
        },
        "rejection_reason_counts": {
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
            "NO_UNIQUE_SEQUENCE_PAIR": 3404,
        },
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise MaterializationReportError(
                f"materialization aggregate fact drifted: {key}"
            )
    if 8068 + 3861 != 11929:
        raise MaterializationReportError("frozen materialization arithmetic differs")


def _default_materialization_loader(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = protocol["materialization_report_identity"]
    path = Path(str(identity["absolute_path"]))
    payload = _read_bytes(
        path,
        label="registered aggregate materialization report",
        error_type=MaterializationReportError,
    )
    if len(payload) != identity["bytes"]:
        raise MaterializationReportError("materialization report byte count drifted")
    observed_sha256 = _sha256(payload)
    if observed_sha256 != identity["sha256"]:
        raise MaterializationReportError("materialization report identity drifted")
    try:
        report = _json_object(payload, label="materialization report")
    except PreflightError as exc:
        raise MaterializationReportError(str(exc)) from exc
    return report, {
        "path": str(path),
        "role": str(identity["role"]),
        "bytes": len(payload),
        "sha256": observed_sha256,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _build_report(
    protocol: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    verified_authorities: list[dict[str, str]],
    materialization_identity: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    ledger = protocol["registered_public_ledger_facts"]
    frozen = protocol["frozen_materialization_facts"]
    output = protocol["output_contract"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": protocol["contract_id"],
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "recorded_at": recorded_at,
        "overall_decision": output["overall_decision"],
        "terminal_status": output["terminal_status"],
        "implementation_binding": dict(binding),
        "verified_public_input_identities": {
            "repository_authorities": verified_authorities,
            "materialization_report": dict(materialization_identity),
            "registered_runtime_event": {
                "event_id": ledger["runtime_event_id"],
                "event_name": ledger["runtime_event_name"],
                "publication_status": ledger["publication_status"],
                "runtime_output_count_after": ledger["runtime_output_count_after"],
            },
        },
        "registered_aggregate_evidence": {
            "published_universe_row_count": frozen["published_universe_row_count"],
            "accepted_pair_complete_raw_endpoint_count": frozen[
                "accepted_pair_complete_raw_endpoint_count"
            ],
            "rejected_published_row_count": frozen["rejected_published_row_count"],
            "schema_valid_development_record_count": frozen[
                "schema_valid_development_record_count"
            ],
            "scientific_disposition": frozen["scientific_disposition"],
            "canonical_record_count": 0,
            "contribution": copy.deepcopy(frozen["contribution"]),
            "primary_measurement_route": ledger["primary_measurement_route"],
            "paper_faithful_transform": ledger["paper_faithful_transform"],
            "public_raw_locator_and_lineage_status": ledger[
                "public_raw_locator_and_lineage_status"
            ],
            "development_reconstruction_status": ledger[
                "development_reconstruction_status"
            ],
            "public_replicate_count": 3,
            "primary_label_standard_error": None,
        },
        "registered_aggregate_passes": copy.deepcopy(
            protocol["registered_aggregate_passes"]
        ),
        "registered_aggregate_pass_count": 3,
        "qualification_blockers": copy.deepcopy(protocol["qualification_blockers"]),
        "open_qualification_blocker_count": len(protocol["qualification_blockers"]),
        "minimum_external_authority_requests": copy.deepcopy(
            protocol["minimum_external_authority_requests"]
        ),
        "required_qualification_report_fields": copy.deepcopy(
            protocol["required_qualification_report_fields"]
        ),
        "exposure_boundary": {
            "sequence_exposure": ledger["sequence_exposure"],
            "label_exposure": ledger["label_exposure"],
            "untouched_confirmatory": False,
            "checkpoint_specific_exposure": UNKNOWN,
        },
        "rights_boundary": {
            **copy.deepcopy(frozen["license_boundary"]),
            "private_derivative_use_is_qualification_use_grant": False,
            "recommended_future_scope": "PRIVATE_CANONICAL_ONLY",
            "recommended_future_scope_approved": False,
        },
        "historical_outer_recovery_boundary": {
            "gate_summary": copy.deepcopy(ledger["outer_recovery_gate_summary"]),
            "is_dataset_qualification": False,
            "unknown_is_the_only_qualification_blocker": False,
        },
        "future_contribution_boundary": copy.deepcopy(
            protocol["future_contribution_boundary"]
        ),
        "scope_attestation": {
            "aggregate_only": True,
            "repository_public_authority_file_count": len(verified_authorities),
            "public_aggregate_report_count": 1,
            "private_row_artifact_read_count": 0,
            "row_record_read_count": 0,
            "sequence_value_read_count": 0,
            "effect_value_read_count": 0,
            "split_run_count": 0,
            "leakage_run_count": 0,
            "power_run_count": 0,
            "qualifier_run_count": 0,
            "canonical_materialization_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
        },
        "terminal_truth": {
            "qualified": False,
            "schema_valid_development_is_qualification": False,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
        "sole_next_action": (
            "OBTAIN_AND_USER_APPROVE_THE_GSE232572_SPECIFIC_EXTERNAL_AUTHORITY_"
            "BUNDLE_BEFORE_ROW_ACCESS_OR_CANONICAL_MATERIALIZATION"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_exclusive(output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise OutputError("exclusive output directory already exists")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        output_path = output_dir / REPORT_FILENAME
        with output_path.open("xb") as handle:
            handle.write(_json_bytes(report))
    except OSError as exc:
        raise OutputError("cannot create the single aggregate output") from exc
    return output_path


BindingAuditor = Callable[
    [Mapping[str, Any], Path, bytes, Path], Mapping[str, str]
]
AuthorityAuditor = Callable[
    [Mapping[str, Any], Path], list[dict[str, str]]
]
MaterializationLoader = Callable[
    [Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]
]


def execute(
    protocol_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    binding_auditor: BindingAuditor | None = None,
    authority_auditor: AuthorityAuditor | None = None,
    materialization_loader: MaterializationLoader | None = None,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_payload = load_protocol(protocol_path)
    effective_repo_root = repo_root or protocol_path.parent.parent

    binding = (binding_auditor or _default_binding_auditor)(
        protocol,
        protocol_path,
        protocol_payload,
        effective_repo_root,
    )
    verified_authorities = (authority_auditor or _default_authority_auditor)(
        protocol, effective_repo_root
    )
    if len(verified_authorities) != len(EXPECTED_AUTHORITIES):
        raise AuthorityError("authority auditor did not verify exact5")
    materialization_report, materialization_identity = (
        materialization_loader or _default_materialization_loader
    )(protocol)
    _validate_materialization_report(materialization_report)

    report = _build_report(
        protocol,
        binding=binding,
        verified_authorities=verified_authorities,
        materialization_identity=materialization_identity,
        recorded_at=recorded_at or _utc_now(),
    )
    _write_exclusive(output_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = execute(
        args.protocol,
        args.output_dir,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "overall_decision": report["overall_decision"],
                "terminal_status": report["terminal_status"],
                "output": str(args.output_dir / REPORT_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
