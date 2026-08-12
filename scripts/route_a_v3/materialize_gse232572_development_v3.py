#!/usr/bin/env python3
"""Materialize private, schema-valid GSE232572 DEVELOPMENT records.

This producer deliberately re-reads the five official public assets and reuses
the registered recovery producer's FASTA, pairing, matrix, and XLSX parsers.
It never consumes the prior private reconstruction JSONL.  Successful rows
implement CanonicalInterventionRecordV3 only as DEVELOPMENT records: they stay
unqualified, carry zero ordinary/A1/A2 contribution, and cannot unlock training,
model selection, or a next phase.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_ID = "ROUTE_A_V3_GSE232572_DEVELOPMENT_V3_MATERIALIZATION_V1"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE232572"
STUDY_ID = "GSE232572"
INDEPENDENT_STUDY_GROUP_ID = "GSE232573"
BASE_COMMIT = "aa396dbdeac083c9f88df62877ff7cbcb7e0d318"
PRODUCTION_AUTHORITY_COMMIT = "b69b9932a5e170f97d2e6fe1e3b9442bdf31f5db"
HEX40 = "0123456789abcdef"
HEX64 = "0123456789abcdef"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
SUCCESS_STATUS = "DEVELOPMENT_V3_MATERIALIZED_NOT_QUALIFIED"
STOP_STATUS = "STOP_BEFORE_DEVELOPMENT_V3_ROW_PRODUCTION"
PRIVATE_FILENAME = "development_v3_records.private.jsonl"
REPORT_FILENAME = "GSE232572_DEVELOPMENT_V3_MATERIALIZATION_REPORT.json"
EXPECTED_PUBLISHED = 11929
EXPECTED_ACCEPTED = 8068
EXPECTED_NO_UNIQUE = 3404
EXPECTED_AMBIGUOUS = 457
EXPECTED_COMPLETE_ENDPOINTS = 8068
EXPECTED_INCOMPLETE_ENDPOINTS = 0
EXPECTED_AUXILIARY_DEFINED = 8068
EXPECTED_AUXILIARY_ZERO_UNDEFINED = 0
CONFIG_RELATIVE_PATH = (
    "configs/route_a_v3_gse232572_development_v3_materialization_v1.json"
)
SCRIPT_RELATIVE_PATH = (
    "scripts/route_a_v3/materialize_gse232572_development_v3.py"
)
TEST_RELATIVE_PATH = (
    "tests/route_a_v3/test_materialize_gse232572_development_v3.py"
)
EXPECTED_I1_PATHS = sorted(
    [CONFIG_RELATIVE_PATH, SCRIPT_RELATIVE_PATH, TEST_RELATIVE_PATH]
)
EXPECTED_I2_PATHS = sorted([SCRIPT_RELATIVE_PATH, TEST_RELATIVE_PATH])
EXPECTED_B2_PATHS = [CONFIG_RELATIVE_PATH]
EXPECTED_BINDING_SCALARS = sorted(
    [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ]
)
EXPECTED_AUTHORITY_ROLES = {
    "RECOVERY_CONFIG": "configs/route_a_v3_gse232572_a1_recovery_v1.json",
    "RECOVERY_SCRIPT": "scripts/route_a_v3/recover_gse232572_a1.py",
    "GENERIC_FASTA_HELPER": (
        "d1_staging/scripts/d1/reconstruct_gse232572_sequences.py"
    ),
    "CANONICAL_V3_SCHEMA": (
        "schemas/route_a_v3/canonical_intervention_record.schema.json"
    ),
}


class MaterializationError(RuntimeError):
    """A public, aggregate-reportable STOP."""

    def __init__(self, gate: str, code: str):
        super().__init__(f"{gate}: {code}")
        self.gate = gate
        self.code = code


def _stop(gate: str, code: str) -> None:
    raise MaterializationError(gate, code)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_json_bytes(payload: bytes, gate: str, code: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        _stop(gate, code)
    if not isinstance(value, dict):
        _stop(gate, code)
    return value


def _read_json(path: Path, gate: str, code: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError:
        _stop(gate, code)
    return _read_json_bytes(payload, gate, code)


def _mapping(value: Any, gate: str, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _stop(gate, code)
    return value


def _is_hex(value: Any, length: int) -> bool:
    alphabet = HEX40 if length == 40 else HEX64
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in alphabet for character in value)
    )


def _parse_timestamp(value: str, gate: str, code: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _stop(gate, code)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _stop(gate, code)


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        _stop("REPOSITORY_AUTHORITY", "GIT_AUTHORITY_QUERY_FAILED")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeError:
        _stop("REPOSITORY_AUTHORITY", "GIT_AUTHORITY_OUTPUT_NOT_UTF8")


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        _stop("REPOSITORY_AUTHORITY", "FROZEN_GIT_BLOB_NOT_READABLE")
    return completed.stdout


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    rendered = _git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return sorted(line for line in rendered.splitlines() if line)


def _leaf_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_leaf_differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def _validate_config(config: Mapping[str, Any]) -> None:
    frozen_root = {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "CONFIG_ONLY_IMPLEMENTATION_BINDING_REQUIRED",
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "study_id": STUDY_ID,
    }
    for key, expected in frozen_root.items():
        if config.get(key) != expected:
            _stop("CONFIG", f"{key.upper()}_NOT_FROZEN")

    binding = _mapping(
        config.get("implementation_binding"),
        "CONFIG",
        "IMPLEMENTATION_BINDING_NOT_OBJECT",
    )
    if binding.get("binding_scheme") != "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1":
        _stop("CONFIG", "IMPLEMENTATION_BINDING_SCHEME_NOT_FROZEN")
    if binding.get("implementation_script_path") != SCRIPT_RELATIVE_PATH:
        _stop("CONFIG", "IMPLEMENTATION_SCRIPT_PATH_NOT_FROZEN")
    if binding.get("implementation_test_path") != TEST_RELATIVE_PATH:
        _stop("CONFIG", "IMPLEMENTATION_TEST_PATH_NOT_FROZEN")
    if sorted(binding.get("unknown_to_bound_scalar_paths", [])) != (
        EXPECTED_BINDING_SCALARS
    ):
        _stop("CONFIG", "IMPLEMENTATION_BINDING_SCALAR_SET_NOT_FROZEN")

    repository = _mapping(
        config.get("repository_authority"),
        "CONFIG",
        "REPOSITORY_AUTHORITY_NOT_OBJECT",
    )
    if repository.get("base_commit") != BASE_COMMIT:
        _stop("CONFIG", "BASE_COMMIT_NOT_FROZEN")
    if repository.get("implementation_exact_changed_paths") != EXPECTED_I1_PATHS:
        _stop("CONFIG", "IMPLEMENTATION_PATH_SET_NOT_FROZEN")
    if repository.get("binding_exact_changed_paths") != EXPECTED_B2_PATHS:
        _stop("CONFIG", "BINDING_PATH_SET_NOT_FROZEN")
    if not isinstance(repository.get("production_repo_root"), str) or not str(
        repository["production_repo_root"]
    ).startswith("/"):
        _stop("CONFIG", "PRODUCTION_REPO_ROOT_NOT_ABSOLUTE")
    if not isinstance(repository.get("branch"), str) or not repository["branch"]:
        _stop("CONFIG", "BRANCH_NOT_FROZEN")

    blobs = repository.get("frozen_authority_blobs")
    if not isinstance(blobs, list) or len(blobs) != 4:
        _stop("CONFIG", "FROZEN_AUTHORITY_BLOB_SET_NOT_CLOSED")
    by_role: dict[str, Mapping[str, Any]] = {}
    for value in blobs:
        item = _mapping(value, "CONFIG", "FROZEN_AUTHORITY_BLOB_NOT_OBJECT")
        role = item.get("role")
        if role in by_role or role not in EXPECTED_AUTHORITY_ROLES:
            _stop("CONFIG", "FROZEN_AUTHORITY_ROLE_SET_NOT_CLOSED")
        if item.get("path") != EXPECTED_AUTHORITY_ROLES[str(role)]:
            _stop("CONFIG", "FROZEN_AUTHORITY_PATH_NOT_FROZEN")
        if not _is_hex(item.get("git_blob_oid"), 40):
            _stop("CONFIG", "FROZEN_AUTHORITY_GIT_BLOB_INVALID")
        if not _is_hex(item.get("sha256"), 64):
            _stop("CONFIG", "FROZEN_AUTHORITY_SHA256_INVALID")
        by_role[str(role)] = item
    if set(by_role) != set(EXPECTED_AUTHORITY_ROLES):
        _stop("CONFIG", "FROZEN_AUTHORITY_ROLE_SET_NOT_CLOSED")

    recovery_report = _mapping(
        config.get("public_recovery_report"),
        "CONFIG",
        "PUBLIC_RECOVERY_REPORT_NOT_OBJECT",
    )
    if recovery_report.get("public_aggregate_only") is not True:
        _stop("CONFIG", "RECOVERY_REPORT_PUBLIC_AGGREGATE_BOUNDARY_NOT_FROZEN")
    if recovery_report.get("private_row_artifacts_consumed") is not False:
        _stop("CONFIG", "PRIVATE_RECOVERY_ROWS_MUST_NOT_BE_CONSUMED")
    if not isinstance(recovery_report.get("absolute_path"), str) or not str(
        recovery_report["absolute_path"]
    ).startswith("/"):
        _stop("CONFIG", "PUBLIC_RECOVERY_REPORT_PATH_NOT_ABSOLUTE")
    if not isinstance(recovery_report.get("bytes"), int) or int(
        recovery_report["bytes"]
    ) <= 0:
        _stop("CONFIG", "PUBLIC_RECOVERY_REPORT_BYTES_INVALID")
    if not _is_hex(recovery_report.get("sha256"), 64):
        _stop("CONFIG", "PUBLIC_RECOVERY_REPORT_SHA256_INVALID")

    contract = _mapping(
        config.get("materialization_contract"),
        "CONFIG",
        "MATERIALIZATION_CONTRACT_NOT_OBJECT",
    )
    frozen_counts = {
        "required_published_universe_row_count": EXPECTED_PUBLISHED,
        "required_development_record_count": EXPECTED_ACCEPTED,
        "required_rejection_reason_counts": {
            "NO_UNIQUE_SEQUENCE_PAIR": EXPECTED_NO_UNIQUE,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": EXPECTED_AMBIGUOUS,
        },
        "required_complete_raw_endpoint_pair_count": EXPECTED_COMPLETE_ENDPOINTS,
        "required_incomplete_raw_endpoint_pair_count": EXPECTED_INCOMPLETE_ENDPOINTS,
        "required_raw_auxiliary_defined_pair_count": EXPECTED_AUXILIARY_DEFINED,
        "required_raw_auxiliary_zero_undefined_pair_count": (
            EXPECTED_AUXILIARY_ZERO_UNDEFINED
        ),
    }
    for key, expected in frozen_counts.items():
        if contract.get(key) != expected:
            _stop("CONFIG", f"{key.upper()}_NOT_FROZEN")
    frozen_contract = {
        "published_sheet_name": "Sheet 5",
        "published_header_row_1_based": 4,
        "published_lnfc_column": "lnFC",
        "published_lnfc_scale": "NATURAL_LOG_RELATIVE_ACTIVITY_ALT_OVER_REF",
        "published_lnfc_is_primary_label": True,
        "published_row_number_must_be_retained": True,
        "canonical_schema_title": "CanonicalInterventionRecordV3",
        "canonical_schema_version": "3.0.0",
        "canonical_schema_draft": (
            "https://json-schema.org/draft/2020-12/schema"
        ),
        "required_validator_class": "Draft202012Validator",
        "private_derivative_use_fact_aggregate_only": (
            "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED"
        ),
        "public_redistribution_status": (
            "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
        ),
        "canonical_qualification_allowed": False,
        "canonical_study_contribution_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
    }
    for key, expected in frozen_contract.items():
        if contract.get(key) != expected:
            _stop("CONFIG", f"{key.upper()}_NOT_FROZEN")
    row_contract = _mapping(
        contract.get("row_contract"), "CONFIG", "ROW_CONTRACT_NOT_OBJECT"
    )
    expected_row_contract = {
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "data_role": "ORDINARY_DEVELOPMENT",
        "eligibility_status": "DEVELOPMENT_ONLY",
        "exposure_stratum": "DEVELOPMENT_ONLY",
        "sequence_exposed": True,
        "label_exposed": True,
        "exposure_audit_id": UNKNOWN,
        "split_partition": "DEVELOPMENT",
        "leakage_audit_status": "NOT_RUN",
        "replicate_count": 3,
        "standard_error": None,
        "study_independent_group_id": INDEPENDENT_STUDY_GROUP_ID,
        "cell_line": "HeLa",
        "license_status": "UNKNOWN_BLOCKED",
        "redistribution_allowed": False,
        "license_verified_at_source": (
            "PUBLIC_RECOVERY_REPORT_RECORDED_AT_STATUS_OBSERVATION_NOT_A_LICENSE_GRANT"
        ),
    }
    if dict(row_contract) != expected_row_contract:
        _stop("CONFIG", "DEVELOPMENT_ROW_BOUNDARY_NOT_FROZEN")

    outputs = _mapping(config.get("outputs"), "CONFIG", "OUTPUTS_NOT_OBJECT")
    expected_outputs = {
        "private_records_filename": PRIVATE_FILENAME,
        "public_report_filename": REPORT_FILENAME,
        "success_status": SUCCESS_STATUS,
        "stop_status": STOP_STATUS,
        "success_exact_output_count": 2,
        "failure_exact_output_count": 1,
    }
    if dict(outputs) != expected_outputs:
        _stop("CONFIG", "OUTPUT_CONTRACT_NOT_FROZEN")


def _audit_repository(
    repo_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    repository = _mapping(
        config["repository_authority"],
        "REPOSITORY_AUTHORITY",
        "REPOSITORY_AUTHORITY_NOT_OBJECT",
    )
    try:
        resolved_repo = repo_root.resolve(strict=True)
        resolved_config = config_path.resolve(strict=True)
    except OSError:
        _stop("REPOSITORY_AUTHORITY", "REPOSITORY_OR_CONFIG_NOT_RESOLVABLE")
    if resolved_repo != Path(str(repository["production_repo_root"])).resolve():
        _stop("REPOSITORY_AUTHORITY", "PRODUCTION_REPO_ROOT_MISMATCH")
    if resolved_config != resolved_repo / PurePosixPath(CONFIG_RELATIVE_PATH):
        _stop("REPOSITORY_AUTHORITY", "CONFIG_PATH_NOT_CANONICAL")
    if _git(resolved_repo, "status", "--porcelain"):
        _stop("REPOSITORY_AUTHORITY", "WORKTREE_NOT_CLEAN")
    if _git(resolved_repo, "symbolic-ref", "--short", "HEAD") != repository["branch"]:
        _stop("REPOSITORY_AUTHORITY", "BRANCH_MISMATCH")

    binding = _mapping(
        config["implementation_binding"],
        "REPOSITORY_AUTHORITY",
        "IMPLEMENTATION_BINDING_NOT_OBJECT",
    )
    if binding.get("status") != BOUND:
        _stop("REPOSITORY_AUTHORITY", "IMPLEMENTATION_BINDING_NOT_BOUND")
    implementation_commit = binding.get("implementation_commit")
    if not _is_hex(implementation_commit, 40):
        _stop("REPOSITORY_AUTHORITY", "IMPLEMENTATION_COMMIT_INVALID")
    if not _is_hex(binding.get("implementation_script_sha256"), 64):
        _stop("REPOSITORY_AUTHORITY", "IMPLEMENTATION_SCRIPT_SHA256_INVALID")
    if not _is_hex(binding.get("implementation_test_sha256"), 64):
        _stop("REPOSITORY_AUTHORITY", "IMPLEMENTATION_TEST_SHA256_INVALID")

    head = _git(resolved_repo, "rev-parse", "HEAD")
    if _git(resolved_repo, "rev-parse", f"{head}^") != implementation_commit:
        _stop("REPOSITORY_AUTHORITY", "B2_NOT_DIRECT_CHILD_OF_I2")
    if (
        _git(resolved_repo, "rev-parse", f"{implementation_commit}^")
        != PRODUCTION_AUTHORITY_COMMIT
    ):
        _stop("REPOSITORY_AUTHORITY", "I2_NOT_DIRECT_CHILD_OF_I1")
    if (
        _git(resolved_repo, "rev-parse", f"{PRODUCTION_AUTHORITY_COMMIT}^")
        != BASE_COMMIT
    ):
        _stop("REPOSITORY_AUTHORITY", "I1_NOT_DIRECT_CHILD_OF_BASE")
    if _changed_paths(resolved_repo, PRODUCTION_AUTHORITY_COMMIT) != EXPECTED_I1_PATHS:
        _stop("REPOSITORY_AUTHORITY", "I1_NOT_EXACT3")
    if _changed_paths(resolved_repo, str(implementation_commit)) != EXPECTED_I2_PATHS:
        _stop("REPOSITORY_AUTHORITY", "I2_NOT_EXACT2")
    if _changed_paths(resolved_repo, head) != EXPECTED_B2_PATHS:
        _stop("REPOSITORY_AUTHORITY", "BINDING_COMMIT_NOT_CONFIG_ONLY")

    config_i1_payload = _git_blob(
        resolved_repo, PRODUCTION_AUTHORITY_COMMIT, CONFIG_RELATIVE_PATH
    )
    config_i2_payload = _git_blob(
        resolved_repo, str(implementation_commit), CONFIG_RELATIVE_PATH
    )
    if config_i2_payload != config_i1_payload:
        _stop("REPOSITORY_AUTHORITY", "I2_CHANGED_I1_CONFIG")
    config_i = _read_json_bytes(
        config_i2_payload,
        "REPOSITORY_AUTHORITY",
        "IMPLEMENTATION_CONFIG_NOT_JSON",
    )
    differences = sorted(_leaf_differences(config_i, config))
    if differences != EXPECTED_BINDING_SCALARS:
        _stop("REPOSITORY_AUTHORITY", "BINDING_COMMIT_CHANGED_NON_BINDING_SCALAR")
    binding_i = _mapping(
        config_i.get("implementation_binding"),
        "REPOSITORY_AUTHORITY",
        "IMPLEMENTATION_CONFIG_BINDING_NOT_OBJECT",
    )
    for scalar in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        if binding_i.get(scalar) != UNKNOWN:
            _stop("REPOSITORY_AUTHORITY", "IMPLEMENTATION_CONFIG_NOT_UNBOUND")

    for path_key, sha_key in (
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        relative_path = str(binding[path_key])
        implementation_payload = _git_blob(
            resolved_repo, str(implementation_commit), relative_path
        )
        if _sha256(implementation_payload) != binding[sha_key]:
            _stop("REPOSITORY_AUTHORITY", "BOUND_IMPLEMENTATION_BLOB_SHA256_MISMATCH")
        try:
            working_payload = (resolved_repo / PurePosixPath(relative_path)).read_bytes()
        except OSError:
            _stop("REPOSITORY_AUTHORITY", "BOUND_IMPLEMENTATION_FILE_NOT_READABLE")
        if working_payload != implementation_payload:
            _stop("REPOSITORY_AUTHORITY", "WORKING_IMPLEMENTATION_BLOB_DRIFT")

    authorities: dict[str, Mapping[str, Any]] = {}
    for value in repository["frozen_authority_blobs"]:
        item = _mapping(
            value,
            "REPOSITORY_AUTHORITY",
            "FROZEN_AUTHORITY_BLOB_NOT_OBJECT",
        )
        role = str(item["role"])
        path = str(item["path"])
        if _git(
            resolved_repo,
            "rev-parse",
            f"{PRODUCTION_AUTHORITY_COMMIT}:{path}",
        ) != item["git_blob_oid"]:
            _stop("REPOSITORY_AUTHORITY", "I1_AUTHORITY_GIT_BLOB_DRIFT")
        if _git(resolved_repo, "rev-parse", f"HEAD:{path}") != item["git_blob_oid"]:
            _stop("REPOSITORY_AUTHORITY", "CURRENT_AUTHORITY_GIT_BLOB_DRIFT")
        payload = _git_blob(resolved_repo, PRODUCTION_AUTHORITY_COMMIT, path)
        if _sha256(payload) != item["sha256"]:
            _stop("REPOSITORY_AUTHORITY", "I1_AUTHORITY_SHA256_DRIFT")
        try:
            if (resolved_repo / PurePosixPath(path)).read_bytes() != payload:
                _stop("REPOSITORY_AUTHORITY", "WORKING_AUTHORITY_BLOB_DRIFT")
        except OSError:
            _stop("REPOSITORY_AUTHORITY", "WORKING_AUTHORITY_BLOB_NOT_READABLE")
        authorities[role] = item
    return authorities


def _load_module(path: Path, name: str, gate: str, code: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _stop(gate, code)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _stop(gate, code)
    return module


def _call_recovery(
    recovery: Any,
    operation: Callable[..., Any],
    *arguments: Any,
) -> Any:
    try:
        return operation(*arguments)
    except recovery.RecoveryError as error:
        _stop(str(error.gate), str(error.code))


def _load_schema_validator(schema_path: Path, contract: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    schema = _read_json(schema_path, "SCHEMA", "CANONICAL_SCHEMA_NOT_READABLE_JSON")
    if schema.get("$schema") != contract["canonical_schema_draft"]:
        _stop("SCHEMA", "CANONICAL_SCHEMA_DRAFT_MISMATCH")
    if schema.get("title") != contract["canonical_schema_title"]:
        _stop("SCHEMA", "CANONICAL_SCHEMA_TITLE_MISMATCH")
    if schema.get("schema_version") != contract["canonical_schema_version"]:
        _stop("SCHEMA", "CANONICAL_SCHEMA_VERSION_MISMATCH")
    if schema.get("additionalProperties") is not False:
        _stop("SCHEMA", "CANONICAL_SCHEMA_NOT_CLOSED")
    try:
        from jsonschema import FormatChecker, validators

        validator_class = validators.validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema, format_checker=FormatChecker())
    except Exception:
        _stop("SCHEMA", "DRAFT_2020_12_VALIDATOR_UNAVAILABLE")
    if validator_class.__name__ != contract["required_validator_class"]:
        _stop("SCHEMA", "CANONICAL_SCHEMA_VALIDATOR_CLASS_MISMATCH")
    return validator, schema


def _require_public_report(
    path: Path,
    report_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _stop("PUBLIC_RECOVERY_REPORT", "PUBLIC_RECOVERY_REPORT_MISSING")
    if resolved != Path(str(report_contract["absolute_path"])).resolve():
        _stop("PUBLIC_RECOVERY_REPORT", "PUBLIC_RECOVERY_REPORT_PATH_MISMATCH")
    try:
        payload = resolved.read_bytes()
    except OSError:
        _stop("PUBLIC_RECOVERY_REPORT", "PUBLIC_RECOVERY_REPORT_NOT_READABLE")
    if len(payload) != report_contract["bytes"]:
        _stop("PUBLIC_RECOVERY_REPORT", "PUBLIC_RECOVERY_REPORT_BYTES_MISMATCH")
    if _sha256(payload) != report_contract["sha256"]:
        _stop("PUBLIC_RECOVERY_REPORT", "PUBLIC_RECOVERY_REPORT_SHA256_MISMATCH")
    report = _read_json_bytes(
        payload,
        "PUBLIC_RECOVERY_REPORT",
        "PUBLIC_RECOVERY_REPORT_NOT_JSON",
    )
    expected = _mapping(
        report_contract.get("expected"),
        "CONFIG",
        "PUBLIC_RECOVERY_REPORT_EXPECTATION_NOT_OBJECT",
    )
    for key, value in expected.items():
        if report.get(key) != value:
            _stop("PUBLIC_RECOVERY_REPORT", f"PUBLIC_RECOVERY_REPORT_{key.upper()}_DRIFT")
    _parse_timestamp(
        str(report.get("recorded_at")),
        "PUBLIC_RECOVERY_REPORT",
        "PUBLIC_RECOVERY_REPORT_RECORDED_AT_INVALID",
    )
    return report


def _read_published_results_with_rows(
    recovery: Any,
    path: Path,
    contract: Mapping[str, Any],
) -> dict[tuple[str, ...], dict[str, Any]]:
    sheet_name = contract.get("sheet_name")
    header_row_number = contract.get("header_row_1_based")
    if not isinstance(sheet_name, str) or not isinstance(header_row_number, int):
        _stop("CONFIG", "PUBLISHED_SHEET_CONTRACT_INVALID")
    rows = _call_recovery(
        recovery, recovery._read_xlsx_sheet, path, sheet_name
    )
    if header_row_number not in rows:
        _stop("PUBLISHED_RESULTS", "PUBLISHED_HEADER_ROW_MISSING")
    headers: dict[str, int] = {}
    for index, value in rows[header_row_number].items():
        rendered = str(value).strip()
        if not rendered:
            continue
        if rendered in headers:
            _stop("PUBLISHED_RESULTS", "PUBLISHED_HEADER_DUPLICATED")
        headers[rendered] = index
    columns = _mapping(
        contract.get("columns"), "CONFIG", "PUBLISHED_COLUMNS_NOT_OBJECT"
    )
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row_number in sorted(rows):
        if row_number <= header_row_number:
            continue
        row = rows[row_number]
        if not row or not any(str(value).strip() for value in row.values()):
            continue
        key = _call_recovery(
            recovery,
            recovery._join_key,
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["chromosome_position"],
            ),
            _call_recovery(
                recovery, recovery._column_value, row, headers, columns["gene"]
            ),
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["gene_strand"],
            ),
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["reference_allele"],
            ),
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["alternate_allele"],
            ),
        )
        if key in result:
            _stop("PUBLISHED_RESULTS", "PUBLISHED_JOIN_KEY_DUPLICATED")
        published_lnfc = _call_recovery(
            recovery,
            recovery._as_finite_float,
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["published_ln_activity"],
            ),
            "PUBLISHED_LNFC_NOT_FINITE",
        )
        fdr = _call_recovery(
            recovery,
            recovery._as_finite_float,
            _call_recovery(
                recovery,
                recovery._column_value,
                row,
                headers,
                columns["mpranalyze_fdr"],
            ),
            "MPRANALYZE_FDR_NOT_FINITE",
        )
        fdr_range = contract.get("mpranalyze_fdr_range")
        if not (
            isinstance(fdr_range, list)
            and len(fdr_range) == 2
            and float(fdr_range[0]) <= fdr <= float(fdr_range[1])
        ):
            _stop("PUBLISHED_RESULTS", "MPRANALYZE_FDR_OUT_OF_RANGE")
        result[key] = {
            "published_lnfc": published_lnfc,
            "mpranalyze_fdr": fdr,
            "sheet_row_1_based": row_number,
        }
    if not result:
        _stop("PUBLISHED_RESULTS", "NO_RELEVANT_PUBLISHED_RESULT_ROWS")
    return result


def _raw_auxiliary_counts(
    pairs: Sequence[Mapping[str, Any]],
    matrices: Mapping[tuple[int, str, int], Mapping[str, float]],
) -> tuple[int, int]:
    defined = 0
    zero_undefined = 0
    for pair in pairs:
        reference = pair["ref"]
        alternate = pair["alt"]
        subpool = int(reference["subpool_number"])
        values = [
            matrices[(subpool, molecule, replicate)][str(record["header"])]
            for replicate in (1, 2, 3)
            for molecule in ("DNA", "RNA")
            for record in (reference, alternate)
        ]
        if any(value == 0 for value in values):
            zero_undefined += 1
        else:
            defined += 1
    return defined, zero_undefined


def _build_record(
    pair: Mapping[str, Any],
    published_row: Mapping[str, Any],
    config: Mapping[str, Any],
    recovery_recorded_at: str,
) -> dict[str, Any]:
    reference = pair["ref"]
    alternate = pair["alt"]
    row_number = int(published_row["sheet_row_1_based"])
    subpool = int(reference["subpool_number"])
    inputs = config["inputs"]
    fasta_contract = inputs["fasta_by_subpool"][str(subpool)]
    results_contract = inputs["published_results"]
    binding = config["implementation_binding"]
    source_group_id = "|".join(
        [
            INDEPENDENT_STUDY_GROUP_ID,
            "HeLa",
            str(reference["gene"]),
            str(reference["source"]),
            str(reference["chr_pos"]),
            str(reference["strand"]),
            str(reference["orientation"]),
            str(reference["header"]),
        ]
    )
    record_id = f"GSE232572|Sheet5|row_{row_number}"
    pair_payload = {
        "sheet_row_1_based": row_number,
        "reference_header": reference["header"],
        "alternate_header": alternate["header"],
        "source_sequence": reference["insert"],
        "candidate_sequence": alternate["insert"],
        "edit_position_zero_based": pair["edit_position"],
        "edit_ref": pair["edit_ref"],
        "edit_alt": pair["edit_alt"],
    }
    pair_sha256 = _sha256(_canonical_bytes(pair_payload))
    combined_sha256 = _sha256(
        _canonical_bytes(
            {
                "pair_sha256": pair_sha256,
                "published_key": list(pair["published_key"]),
                "published_lnfc": published_row["published_lnfc"],
                "mpranalyze_fdr": published_row["mpranalyze_fdr"],
            }
        )
    )
    raw_locator = (
        f"Sheet 5!row={row_number};"
        f"reference_fasta_header={reference['header']};"
        f"alternate_fasta_header={alternate['header']}"
    )
    return {
        "contract_id": CONTRACT_ID,
        "schema_version": "3.0.0",
        "record_id": record_id,
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "source": {
            "source_id": source_group_id,
            "sequence_id": str(reference["header"]),
            "gene_id": str(reference["gene"]),
            "locus_id": str(reference["chr_pos"]),
            "design_family_id": str(reference["source"]),
        },
        "candidate": {
            "candidate_id": f"{record_id}|alternate",
            "sequence_id": str(alternate["header"]),
            "design_id": f"GSE232572|{reference['subpool']}|Sheet5|row_{row_number}",
        },
        "source_sequence": str(reference["insert"]),
        "candidate_sequence": str(alternate["insert"]),
        "sequence_alphabet": "DNA",
        "edit_set": [
            {
                "edit_id": f"{record_id}|edit_1",
                "position": int(pair["edit_position"]),
                "coordinate_system": "ZERO_BASED_SOURCE",
                "ref_base": str(pair["edit_ref"]),
                "alt_base": str(pair["edit_alt"]),
                "region": "3UTR",
                "distance_from_region_start": int(pair["edit_position"]),
            }
        ],
        "region": "3UTR",
        "study": {
            "study_id": STUDY_ID,
            "accession": STUDY_ID,
            "independent_study_group_id": INDEPENDENT_STUDY_GROUP_ID,
            "publication_doi": "10.1038/s41467-024-46795-7",
        },
        "assay": {
            "assay_id": "GSE232572_MAPUTR_HELA",
            "assay_type": "MPRA_MAPUTR_RNA_DNA",
            "protocol_version": "Fu_et_al_2024",
        },
        "context": {
            "context_id": "GSE232572|HeLa",
            "observable_context": "HeLa cells",
            "cell_type": "HeLa",
            "condition": "MapUTR transfection assay",
        },
        "endpoint": {
            "endpoint_id": "GSE232572|ln_activity_ratio_alt_over_ref",
            "endpoint_name": "ln_activity_ratio_alt_over_ref",
            "beneficial_direction": "HIGHER_IS_BETTER",
        },
        "raw_measurement": {
            "value": float(published_row["published_lnfc"]),
            "unit": "natural_log_ratio",
            "scale": "ln",
            "source_column": f"Sheet 5!G{row_number} (lnFC)",
            "detection_limit": None,
        },
        "paper_faithful_transform": {
            "transform_id": "GSE232572_MOESM4_SHEET5_LNFC_IDENTITY_V1",
            "description": (
                "Use the official Sheet 5 lnFC natural-log relative activity "
                "for alternate versus reference as the primary delta."
            ),
            "version": "1.0.0",
            "direction_verified": True,
            "implementation_sha256": binding["implementation_script_sha256"],
        },
        "delta": float(published_row["published_lnfc"]),
        "replicate": {
            "replicate_id": f"GSE232572|{reference['subpool']}|HeLa|triplicate",
            "replicate_group_id": f"GSE232572|{reference['subpool']}|HeLa",
            "replicate_count": 3,
            "aggregation_rule": (
                "Official published Sheet 5 lnFC; no unpublished standard error reconstructed"
            ),
        },
        "standard_error": None,
        "biological_source_group_id": source_group_id,
        "gene_group_id": str(reference["gene"]),
        "data_role": "ORDINARY_DEVELOPMENT",
        "exposure": {
            "stratum": "DEVELOPMENT_ONLY",
            "label_exposed": True,
            "sequence_exposed": True,
            "audit_id": UNKNOWN,
        },
        "split": {
            "split_id": "GSE232572|DEVELOPMENT_ONLY|NOT_LEAKAGE_AUDITED",
            "partition": "DEVELOPMENT",
            "leakage_audit_status": "NOT_RUN",
        },
        "provenance": {
            "dataset_id": DATASET_ID,
            "asset_id": (
                f"MOESM4_Sheet5_plus_{fasta_contract['filename']}"
            ),
            "source_uri": str(results_contract["url"]),
            "source_file_sha256": str(results_contract["sha256"]),
            "raw_record_locator": raw_locator,
            "lineage": [
                {
                    "step_id": "map_exact_fasta_pair",
                    "operation": (
                        "Map the Sheet 5 key to exactly one physical, distinct "
                        "Hamming-distance-one FASTA reference/alternate pair"
                    ),
                    "input_sha256": str(fasta_contract["sha256"]),
                    "output_sha256": pair_sha256,
                },
                {
                    "step_id": "attach_official_published_lnfc",
                    "operation": (
                        "Attach the official Sheet 5 lnFC while retaining the "
                        "one-based worksheet row and exact FASTA headers"
                    ),
                    "input_sha256": str(results_contract["sha256"]),
                    "output_sha256": combined_sha256,
                },
            ],
        },
        "license": {
            "license_id": "GSE232572_PUBLIC_ROW_REDISTRIBUTION_UNRESOLVED",
            "license_name": None,
            "license_uri": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
            "status": "UNKNOWN_BLOCKED",
            "redistribution_allowed": False,
            "verified_at": recovery_recorded_at,
        },
        "eligibility": {
            "status": "DEVELOPMENT_ONLY",
            "reject_reason_code": "PUBLIC_ROW_REDISTRIBUTION_RIGHTS_UNKNOWN",
            "reject_reason_detail": (
                "Private research derivative use is recorded at aggregate level, "
                "but row-level public redistribution remains unknown and no "
                "canonical qualification is claimed."
            ),
        },
    }


def _base_report(recorded_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "study_id": STUDY_ID,
        "recorded_at": recorded_at,
        "status": STOP_STATUS,
        "scientific_disposition": "NOT_QUALIFIED",
        "published_universe_row_count": 0,
        "schema_valid_development_record_count": 0,
        "canonical_record_count": 0,
        "rejected_published_row_count": 0,
        "rejection_reason_counts": {
            "NO_UNIQUE_SEQUENCE_PAIR": 0,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 0,
        },
        "accepted_pair_complete_raw_endpoint_count": 0,
        "accepted_pair_incomplete_raw_endpoint_count": 0,
        "raw_auxiliary_defined_pair_count": 0,
        "raw_auxiliary_zero_undefined_pair_count": 0,
        "qualified": False,
        "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_allowed": False,
        "gates": [],
    }


def _gate(name: str, status: str, code: str) -> dict[str, str]:
    return {"gate": name, "status": status, "code": code}


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    repo_root: Path,
    config_path: Path,
    fasta_paths: Mapping[int, Path],
    raw_tar: Path,
    published_results: Path,
    public_recovery_report: Path,
    output_dir: Path,
    recorded_at: str,
) -> tuple[int, dict[str, Any]]:
    _parse_timestamp(recorded_at, "INVOCATION", "RECORDED_AT_INVALID")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RuntimeError("output directory already exists")
    report = _base_report(recorded_at)
    gates: list[dict[str, str]] = report["gates"]
    try:
        config = _read_json(config_path, "CONFIG", "CONFIG_NOT_READABLE_JSON")
        _validate_config(config)
        gates.append(_gate("CONFIG", "PASS", "DEVELOPMENT_V3_CONTRACT_FROZEN"))

        authorities = _audit_repository(repo_root, config_path, config)
        gates.append(
            _gate(
                "REPOSITORY_AUTHORITY",
                "PASS",
                "BASE_EXACT3_I_CONFIG_ONLY_B_AND_FROZEN_AUTHORITIES_BOUND",
            )
        )

        recovery_script_path = repo_root / PurePosixPath(
            str(authorities["RECOVERY_SCRIPT"]["path"])
        )
        recovery = _load_module(
            recovery_script_path,
            "route_a_v3_gse232572_registered_recovery",
            "RECOVERY_AUTHORITY",
            "REGISTERED_RECOVERY_PRODUCER_NOT_LOADABLE",
        )
        recovery_config = _read_json(
            repo_root / PurePosixPath(str(authorities["RECOVERY_CONFIG"]["path"])),
            "RECOVERY_AUTHORITY",
            "REGISTERED_RECOVERY_CONFIG_NOT_READABLE_JSON",
        )
        _call_recovery(recovery, recovery._validate_config, recovery_config)
        if recovery_config.get("inputs") != config.get("inputs"):
            _stop("RECOVERY_AUTHORITY", "MATERIALIZER_INPUTS_DIVERGE_FROM_RECOVERY_CONFIG")
        result_contract = _mapping(
            recovery_config.get("published_result_contract"),
            "RECOVERY_AUTHORITY",
            "RECOVERY_PUBLISHED_RESULT_CONTRACT_NOT_OBJECT",
        )
        matrix_contract = _mapping(
            recovery_config.get("matrix_contract"),
            "RECOVERY_AUTHORITY",
            "RECOVERY_MATRIX_CONTRACT_NOT_OBJECT",
        )
        gates.append(
            _gate(
                "RECOVERY_AUTHORITY",
                "PASS",
                "REGISTERED_RECOVERY_CONFIG_AND_PARSERS_REUSED",
            )
        )

        materialization_contract = _mapping(
            config["materialization_contract"],
            "CONFIG",
            "MATERIALIZATION_CONTRACT_NOT_OBJECT",
        )
        validator, _ = _load_schema_validator(
            repo_root / PurePosixPath(str(authorities["CANONICAL_V3_SCHEMA"]["path"])),
            materialization_contract,
        )
        gates.append(
            _gate("SCHEMA", "PASS", "DRAFT_2020_12_CANONICAL_V3_SCHEMA_LOADED")
        )

        report_contract = _mapping(
            config["public_recovery_report"],
            "CONFIG",
            "PUBLIC_RECOVERY_REPORT_NOT_OBJECT",
        )
        recovery_report = _require_public_report(
            public_recovery_report, report_contract
        )
        gates.append(
            _gate(
                "PUBLIC_RECOVERY_REPORT",
                "PASS",
                "REGISTERED_PUBLIC_AGGREGATE_STATE_EXACT",
            )
        )

        inputs = _mapping(config["inputs"], "CONFIG", "INPUTS_NOT_OBJECT")
        fasta_contracts = _mapping(
            inputs.get("fasta_by_subpool"), "CONFIG", "FASTA_INPUTS_NOT_OBJECT"
        )
        for subpool in (1, 2, 3):
            _call_recovery(
                recovery,
                recovery._require_input,
                fasta_paths[subpool],
                _mapping(
                    fasta_contracts.get(str(subpool)),
                    "CONFIG",
                    "FASTA_INPUT_NOT_OBJECT",
                ),
                f"fasta{subpool}",
            )
        _call_recovery(
            recovery,
            recovery._require_input,
            raw_tar,
            _mapping(inputs.get("raw_tar"), "CONFIG", "RAW_TAR_INPUT_NOT_OBJECT"),
            "raw_tar",
        )
        _call_recovery(
            recovery,
            recovery._require_input,
            published_results,
            _mapping(
                inputs.get("published_results"),
                "CONFIG",
                "PUBLISHED_RESULTS_INPUT_NOT_OBJECT",
            ),
            "published_results",
        )
        gates.append(
            _gate("OFFICIAL_INPUTS", "PASS", "EXACT_FIVE_OFFICIAL_ASSETS_REREAD")
        )

        helper = _call_recovery(
            recovery,
            recovery._load_generic_helper,
            repo_root,
            str(authorities["GENERIC_FASTA_HELPER"]["path"]),
        )
        fasta_records: list[dict[str, Any]] = []
        for subpool in (1, 2, 3):
            fasta_records.extend(
                _call_recovery(
                    recovery,
                    recovery._read_fasta_records,
                    fasta_paths[subpool],
                    subpool,
                    helper,
                )
            )
        if len({record["header"] for record in fasta_records}) != len(
            fasta_records
        ):
            _stop("FASTA", "FASTAS_SHARE_DUPLICATE_HEADERS")
        matrices = _call_recovery(
            recovery, recovery._read_matrices, raw_tar, matrix_contract
        )
        published = _read_published_results_with_rows(
            recovery, published_results, result_contract
        )
        report["published_universe_row_count"] = len(published)
        pairs, rejection_counts = _call_recovery(
            recovery,
            recovery._map_published_universe,
            fasta_records,
            published,
        )
        report["rejection_reason_counts"] = rejection_counts
        report["rejected_published_row_count"] = sum(rejection_counts.values())
        actual_partition = {
            "published": len(published),
            "accepted": len(pairs),
            "NO_UNIQUE_SEQUENCE_PAIR": rejection_counts[
                "NO_UNIQUE_SEQUENCE_PAIR"
            ],
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": rejection_counts[
                "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS"
            ],
        }
        expected_partition = {
            "published": EXPECTED_PUBLISHED,
            "accepted": EXPECTED_ACCEPTED,
            "NO_UNIQUE_SEQUENCE_PAIR": EXPECTED_NO_UNIQUE,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": EXPECTED_AMBIGUOUS,
        }
        if actual_partition != expected_partition:
            _stop("PAIRING", "DEVELOPMENT_MAPPING_COUNTS_MISMATCH")
        if len(published) != len(pairs) + sum(rejection_counts.values()):
            _stop("PAIRING", "PUBLISHED_UNIVERSE_PARTITION_NOT_CLOSED")
        gates.append(
            _gate(
                "PAIRING",
                "PASS",
                "11929_EQUALS_8068_PLUS_3404_PLUS_457",
            )
        )

        complete_pairs, incomplete_pair_count = _call_recovery(
            recovery,
            recovery._accepted_pairs_with_complete_raw_endpoints,
            pairs,
            matrices,
        )
        report["accepted_pair_complete_raw_endpoint_count"] = len(complete_pairs)
        report["accepted_pair_incomplete_raw_endpoint_count"] = incomplete_pair_count
        if (
            len(complete_pairs) != EXPECTED_COMPLETE_ENDPOINTS
            or incomplete_pair_count != EXPECTED_INCOMPLETE_ENDPOINTS
        ):
            _stop("MATRICES", "ACCEPTED_PAIR_RAW_ENDPOINT_COUNTS_MISMATCH")
        auxiliary_defined, auxiliary_zero_undefined = _raw_auxiliary_counts(
            complete_pairs, matrices
        )
        report["raw_auxiliary_defined_pair_count"] = auxiliary_defined
        report["raw_auxiliary_zero_undefined_pair_count"] = (
            auxiliary_zero_undefined
        )
        if (
            auxiliary_defined != EXPECTED_AUXILIARY_DEFINED
            or auxiliary_zero_undefined != EXPECTED_AUXILIARY_ZERO_UNDEFINED
        ):
            _stop("ENDPOINT", "RAW_AUXILIARY_DEFINED_ZERO_COUNTS_MISMATCH")
        gates.append(
            _gate(
                "MATRICES",
                "PASS",
                "8068_COMPLETE_AND_0_INCOMPLETE_RAW_ENDPOINT_PAIRS",
            )
        )

        row_numbers: set[int] = set()
        reference_headers: set[str] = set()
        alternate_headers: set[str] = set()
        records: list[dict[str, Any]] = []
        for pair in sorted(
            complete_pairs,
            key=lambda item: int(published[item["published_key"]]["sheet_row_1_based"]),
        ):
            published_row = published[pair["published_key"]]
            row_number = int(published_row["sheet_row_1_based"])
            reference_header = str(pair["ref"]["header"])
            alternate_header = str(pair["alt"]["header"])
            if row_number in row_numbers:
                _stop("LOCATOR", "SHEET_ROW_LOCATOR_DUPLICATED")
            if reference_header in reference_headers:
                _stop("LOCATOR", "REFERENCE_FASTA_HEADER_LOCATOR_DUPLICATED")
            if alternate_header in alternate_headers:
                _stop("LOCATOR", "ALTERNATE_FASTA_HEADER_LOCATOR_DUPLICATED")
            row_numbers.add(row_number)
            reference_headers.add(reference_header)
            alternate_headers.add(alternate_header)
            record = _build_record(
                pair,
                published_row,
                config,
                str(recovery_report["recorded_at"]),
            )
            errors = sorted(
                validator.iter_errors(record), key=lambda error: list(error.path)
            )
            if errors:
                _stop("SCHEMA", "DEVELOPMENT_RECORD_SCHEMA_VALIDATION_FAILED")
            records.append(record)
        if len(records) != EXPECTED_ACCEPTED:
            _stop("OUTPUT", "DEVELOPMENT_RECORD_COUNT_MISMATCH")
        if len({record["record_id"] for record in records}) != EXPECTED_ACCEPTED:
            _stop("OUTPUT", "DEVELOPMENT_RECORD_ID_NOT_UNIQUE")
        gates.append(
            _gate(
                "SCHEMA",
                "PASS",
                "ALL_8068_DEVELOPMENT_RECORDS_VALIDATE_DRAFT_2020_12",
            )
        )
        gates.append(
            _gate(
                "SCIENTIFIC_BOUNDARY",
                "PASS",
                "DEVELOPMENT_ONLY_ZERO_CREDIT_NO_TRAINING_UNLOCK",
            )
        )

        private_payload = b"".join(
            _canonical_bytes(record) + b"\n" for record in records
        )
        private_path = output_dir / PRIVATE_FILENAME
        private_path.write_bytes(private_payload)
        report.update(
            {
                "status": SUCCESS_STATUS,
                "scientific_disposition": (
                    "SCHEMA_VALID_DEVELOPMENT_ONLY_NOT_CANONICALLY_QUALIFIED"
                ),
                "schema_valid_development_record_count": len(records),
                "canonical_record_count": 0,
                "qualified": False,
                "contribution": {"ordinary": 0, "a1": 0, "true_a2": 0},
                "license_boundary": {
                    "private_derivative_use_fact": (
                        "VERIFIED_PRIVATE_DERIVATIVE_USE_ALLOWED"
                    ),
                    "public_redistribution_status": (
                        "UNKNOWN_NOT_ASSERTED_SUBMITTER_IP_CAVEAT"
                    ),
                    "row_license_status": "UNKNOWN_BLOCKED",
                    "redistribution_allowed": False,
                    "verified_at_semantics": (
                        "PUBLIC_RECOVERY_REPORT_RECORDED_AT_IS_STATUS_OBSERVATION_NOT_A_LICENSE_GRANT"
                    ),
                },
                "source_public_recovery_report": {
                    "absolute_path": str(public_recovery_report.resolve()),
                    "bytes": report_contract["bytes"],
                    "sha256": report_contract["sha256"],
                    "recorded_at": recovery_report["recorded_at"],
                    "private_row_artifacts_consumed": False,
                },
                "private_output": {
                    "absolute_path": str(private_path.resolve()),
                    "bytes": len(private_payload),
                    "sha256": _sha256(private_payload),
                    "record_count": len(records),
                    "purpose": "WHOLE_FILE_DIGEST_ALLOWS_LATER_IDENTITY_CHECK_WITHOUT_REREADING_ROWS",
                },
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_allowed": False,
            }
        )
        _write_report(output_dir / REPORT_FILENAME, report)
        if sorted(path.name for path in output_dir.iterdir()) != sorted(
            [PRIVATE_FILENAME, REPORT_FILENAME]
        ):
            raise RuntimeError("success output directory is not exact2")
        return 0, report
    except MaterializationError as error:
        gates.append(_gate(error.gate, "FAIL", error.code))
        _write_report(output_dir / REPORT_FILENAME, report)
        return 2, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fasta-subpool-1", required=True, type=Path)
    parser.add_argument("--fasta-subpool-2", required=True, type=Path)
    parser.add_argument("--fasta-subpool-3", required=True, type=Path)
    parser.add_argument("--raw-tar", required=True, type=Path)
    parser.add_argument("--published-results", required=True, type=Path)
    parser.add_argument("--public-recovery-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recorded-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, report = materialize(
        repo_root=args.repo_root,
        config_path=args.config,
        fasta_paths={
            1: args.fasta_subpool_1,
            2: args.fasta_subpool_2,
            3: args.fasta_subpool_3,
        },
        raw_tar=args.raw_tar,
        published_results=args.published_results,
        public_recovery_report=args.public_recovery_report,
        output_dir=args.output_dir,
        recorded_at=args.recorded_at,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "schema_valid_development_record_count": report[
                    "schema_valid_development_record_count"
                ],
                "canonical_record_count": report["canonical_record_count"],
                "rejected_published_row_count": report[
                    "rejected_published_row_count"
                ],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
