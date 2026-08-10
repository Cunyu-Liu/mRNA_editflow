#!/usr/bin/env python3
"""Fail-closed P0 preflight authority for the GSE200302 raw replay.

The production entry point has no acquisition, reference, candidate-document,
or publication callback interface.  While the implementation binding is
UNKNOWN it publishes one closed failure and returns.  Lower-level aggregate
validators and publication primitives exist for synthetic verification only;
there is permanently no aligner, SAM-to-count, R, or xTail executor here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse200304_raw_replay_preflight.v1"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_RAW_REPLAY_PREFLIGHT_V1"
PROTOCOL_BASENAME = "route_a_v3_gse200304_raw_replay.json"
PROTOCOL_CORE_SHA256 = (
    "381a65d3070eef00bd4b73a8936fd779a999c2a890c221802fdea772b48a24de"
)
CANONICALIZATION = "CANONICAL_SORTED_UTF8_V1"

BLOCKED_OUTCOME = "BLOCKED_PRE_EXECUTION_WITH_EVIDENCE"
FAILURE_OUTCOME = "FAIL_CLOSED"
PREFLIGHT_FILENAME = "PREFLIGHT.json"
FAILURE_FILENAME = "FAILURE.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
PUBLICATION_MARKER_FILENAME = "PUBLICATION_COMMIT.json"

EXPECTED_FILE_COUNT = 48
EXPECTED_RUN_COUNT = 24
EXPECTED_TOTAL_BYTES = 12_738_938_976
EXPECTED_REFERENCE_RECORDS = 13_836
EXPECTED_REFERENCE_UNIQUE = 13_832
EXPECTED_IDENTICAL_REFERENCE_GROUPS = 4
EXPECTED_REFERENCE_LENGTH = 250
EXPECTED_RUN_SET_SHA256 = (
    "c70266fc865a13fea4915fcbab66afbadad069fcb1e5439300e73a61dedd9ea5"
)
EXPECTED_ACQUIRER_SHA256 = (
    "1b0d1c5db7e32475fb835cadb5d1805415447a490a1a83840bcb6e8518fa6340"
)
EXPECTED_QUALIFIER_SHA256 = (
    "49950a460079924d5e5b98b7a49bf2dc378a1cf82cba633d19b2bff0b52c9944"
)
EXPECTED_ACQUISITION_MANIFEST_SHA256 = (
    "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b"
)
EXPECTED_ACQUISITION_SOURCE_MARKER_SHA256 = (
    "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b"
)
EXPECTED_ACQUISITION_CLAIM = (
    "This terminal commit establishes acquisition and transport integrity for "
    "the exact 48-file ENA manifest only. It does not establish count "
    "reconstruction, xTail replay, A1 qualification, training authorization, "
    "model performance, or a scientific conclusion."
)

EXPECTED_BUILD_ARGV_TEMPLATE = (
    "bowtie2-build",
    "{reference_fasta}",
    "{index_prefix}",
)
EXPECTED_ALIGNMENT_ARGV_TEMPLATE = (
    "bowtie2",
    "-x",
    "{index_prefix}",
    "-1",
    "{mate_1_fastq}",
    "-2",
    "{mate_2_fastq}",
    "-N",
    "0",
    "--no-sq",
    "--no-hd",
    "-S",
    "{output_sam}",
)

EXPECTED_HARD_BLOCKERS = (
    "PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN",
    "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
    "SAM_TO_COUNT_PAIRED_HANDLING_UNKNOWN",
    "SAM_TO_COUNT_MULTIMAP_POLICY_UNKNOWN",
    "SAM_TO_COUNT_FLAG_POLICY_UNKNOWN",
    "SAM_TO_COUNT_MAPQ_POLICY_UNKNOWN",
    "SAM_TO_COUNT_DUPLICATE_POLICY_UNKNOWN",
    "SAM_TO_COUNT_IDENTICAL_REFERENCE_TIE_POLICY_UNKNOWN",
    "XTAIL_6772_INCLUSION_POLICY_UNKNOWN",
    "PAPER_6892_VS_AUDIT_6885_DENOMINATOR_CONFLICT_UNRESOLVED",
    "EDGER_EXACT_VERSION_UNKNOWN",
    "DESEQ2_EXACT_VERSION_UNKNOWN",
    "XTAIL_DEPENDENCY_LOCK_UNKNOWN",
    "XTAIL_RNG_SEED_AND_STATE_UNKNOWN",
    "PRJNA824033_VS_GSE200304_PRJNA824026_IDENTITY_CONFLICT_UNKNOWN",
    "AUTHOR_CODE_REDISTRIBUTION_PERMISSION_UNKNOWN",
    "RAW_FASTQ_REDISTRIBUTION_PERMISSION_UNKNOWN",
)
NON_BINDING_HARD_BLOCKERS = EXPECTED_HARD_BLOCKERS[1:]

VALID_FAILURE_CODES = (
    "PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN",
    "PROTOCOL_TRUST_FAILED",
    "SCOPE_VIOLATION",
    "ADAPTER_SOURCE_DRIFT",
    "ACQUISITION_ATTESTATION_INVALID",
    "REFERENCE_AGGREGATE_INVALID",
    "SAMPLE_SHEET_INVALID",
    "COUNT_POLICY_INVALID",
    "PUBLICATION_FAILED",
)

EXPECTED_EXECUTION_POLICY = {
    "network_access_allowed": False,
    "subprocess_allowed": False,
    "shell_allowed": False,
    "fastq_body_read_allowed": False,
    "bowtie2_build_execution_allowed": False,
    "alignment_execution_allowed": False,
    "sam_to_count_execution_allowed": False,
    "r_execution_allowed": False,
    "xtail_execution_allowed": False,
    "qualification_allowed": False,
    "canonical_materialization_allowed": False,
    "training_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
}
EXPECTED_GATE_TRUTH = {
    "qualified": False,
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "canonical_record_count": 0,
    "training_started": False,
    "model_selection_started": False,
    "next_phase_authorized": False,
}
EXPECTED_DENOMINATOR_DISCREPANCY = {
    "status": "CONFLICT_UNRESOLVED",
    "paper_reported": {
        "designed_pair_count": 6892,
        "xtail_included_row_count": 6772,
        "attrition_count": 120,
        "evidence_status": "PAPER_REPORTED_NOT_RECONCILED_TO_CURRENT_DESIGN",
    },
    "current_mechanical_audit": {
        "design_pair_count": 6885,
        "processed_row_count": 6772,
        "attrition_count": 113,
        "evidence_status": "CURRENT_AUDITED_NOT_RECONCILED_TO_PAPER_DENOMINATOR",
    },
    "must_not_collapse_to_single_attrition_count": True,
}
EXPECTED_BINDING_KEYS = {
    "binding_scheme",
    "status",
    "production_implementation_commit",
    "production_script_sha256",
    "script_repo_path",
    "test_repo_path",
    "hard_blocker",
}
EXPECTED_SAMPLE_FAMILIES = ("80S_RNA", "High_Poly", "Low_Poly", "Total_RNA")
EXPECTED_REPLICATES = (1, 2, 3, 4, 5, 6)
EXPECTED_SAMPLE_ROW_KEYS = {
    "mate_1_filename",
    "mate_2_filename",
    "measurement_family",
    "replicate",
    "run_accession",
}
EXPECTED_COUNT_POLICY_KEYS = {
    "count_increment",
    "count_unit",
    "discordant_pair_handling",
    "duplicate_handling",
    "excluded_sam_flags",
    "identical_reference_tie_handling",
    "minimum_mapq",
    "multimapping_handling",
    "overlapping_mates_handling",
    "paired_read_handling",
    "required_sam_flags",
    "secondary_alignment_handling",
    "supplementary_alignment_handling",
    "unmapped_mate_handling",
}

EXPECTED_ACQUISITION_AUDIT = {
    "status": "FASTQ_ACQUISITION_COMMITTED",
    "target_subseries_accession": "GSE200302",
    "superseries_accession": "GSE200304",
    "committed": True,
    "accepted": True,
    "verified_file_count": EXPECTED_FILE_COUNT,
    "verified_run_count": EXPECTED_RUN_COUNT,
    "verified_total_bytes": EXPECTED_TOTAL_BYTES,
    "run_set_hash_match": True,
    "paired_mate_set_closed": True,
    "unexpected_member_class_count": 0,
    "fastq_body_read_count_by_preflight": 0,
}
EXPECTED_REFERENCE_AUDIT = {
    "status": "PASS_EXACT_REFERENCE_AGGREGATE",
    "record_count": EXPECTED_REFERENCE_RECORDS,
    "unique_sequence_count": EXPECTED_REFERENCE_UNIQUE,
    "identical_sequence_group_count": EXPECTED_IDENTICAL_REFERENCE_GROUPS,
    "sequence_length": EXPECTED_REFERENCE_LENGTH,
    "acgt_only": True,
    "normalized_uppercase": True,
    "identifier_output_count": 0,
    "sequence_output_count": 0,
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUN_RE = re.compile(r"SRR[0-9]{8}")
FASTQ_MEMBER_RE = re.compile(r"(SRR[0-9]{8})_([12])\.fastq\.gz")
TRANSFER_MEMBER_RE = re.compile(r"(SRR[0-9]{8})_([12])\.fastq\.gz\.transfer\.json")
ACGT_250_RE = re.compile(r"[ACGT]{250}")
SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
FORBIDDEN_PATH_TOKENS = (
    "gse246381",
    "access_log",
    "sealed_external",
    "sealed",
    "restricted",
)
MAX_PROTOCOL_BYTES = 2 * 1024 * 1024
MAX_CANDIDATE_JSON_BYTES = 4 * 1024 * 1024
MAX_MARKER_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_COMPRESSED_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_DECOMPRESSED_BYTES = 128 * 1024 * 1024


class PreflightError(RuntimeError):
    code = "PREFLIGHT_FAILED"


class ScopeViolation(PreflightError):
    code = "SCOPE_VIOLATION"


class ProtocolError(PreflightError):
    code = "PROTOCOL_TRUST_FAILED"


class AdapterSourceError(PreflightError):
    code = "ADAPTER_SOURCE_DRIFT"


class AcquisitionError(PreflightError):
    code = "ACQUISITION_ATTESTATION_INVALID"


class ReferenceAuditError(PreflightError):
    code = "REFERENCE_AGGREGATE_INVALID"


class SampleSheetError(PreflightError):
    code = "SAMPLE_SHEET_INVALID"


class CountPolicyError(PreflightError):
    code = "COUNT_POLICY_INVALID"


class PublicationError(PreflightError):
    code = "PUBLICATION_FAILED"


class PublicationContention(PublicationError):
    code = "OUTPUT_ALREADY_EXISTS"


class PartialPrecommitError(PublicationError):
    code = "PARTIAL_PRECOMMIT"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if set(actual) != set(expected):
            return False
        return all(_strict_equal(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    if type(expected) is float:
        return math.isfinite(actual) and actual == expected
    return actual == expected


def _require_strict(actual: Any, expected: Any, *, label: str) -> None:
    if not _strict_equal(actual, expected):
        raise ProtocolError(f"{label} differs in value, type, or closed structure")


def _assert_json_tree(value: Any, *, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise PublicationError(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_json_tree(item, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise PublicationError(f"{label} contains a non-string key")
            _assert_json_tree(item, label=f"{label}.{key}")
        return
    raise PublicationError(f"{label} contains a non-JSON value type")


def _json_bytes(value: Mapping[str, Any], *, pretty: bool = True) -> bytes:
    _assert_json_tree(value, label="JSON document")
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _strict_json_object(
    payload: bytes,
    *,
    label: str,
    error_type: type[PreflightError] = ProtocolError,
) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise ValueError("non-finite constant")

    def parse_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return value

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise error_type(f"{label} is not strict finite duplicate-free UTF-8 JSON") from exc
    if type(value) is not dict:
        raise error_type(f"{label} root must be an exact object")
    try:
        _assert_json_tree(value, label=label)
    except (PublicationError, RecursionError) as exc:
        raise error_type(f"{label} contains a non-JSON value") from exc
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: Any, *, error_type: type[PreflightError], label: str) -> str:
    if type(value) is not str:
        raise error_type(f"{label} must be a timezone-aware string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise error_type(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise error_type(f"{label} is timezone-naive")
    return value


def _require_platform_capabilities() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        if not hasattr(os, name):
            raise PublicationError(f"platform lacks required {name} capability")
    for function in (os.open, os.mkdir, os.stat, os.unlink, os.link):
        if function not in os.supports_dir_fd:
            raise PublicationError("platform lacks required dir_fd capability")
    if os.stat not in os.supports_follow_symlinks or os.link not in os.supports_follow_symlinks:
        raise PublicationError("platform lacks required no-follow metadata capability")


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    text = os.fspath(path).casefold()
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ScopeViolation(f"{label} rejected before read by the frozen scope guard")


def _absolute_without_resolving(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path.cwd() / value


def _safe_basename(name: str, *, label: str) -> str:
    if type(name) is not str or SAFE_BASENAME_RE.fullmatch(name) is None or Path(name).name != name:
        raise ScopeViolation(f"{label} is not an allowed basename")
    return name


def _close_once(descriptor: int) -> BaseException | None:
    """Close exactly once and return, rather than raise, any close failure."""

    try:
        os.close(descriptor)
    except BaseException as exc:
        return exc
    return None


def _open_directory_chain(path: Path, *, label: str) -> int:
    _require_platform_capabilities()
    if not path.is_absolute():
        raise ScopeViolation(f"{label} must be absolute")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor: int | None = None
    result: int | None = None
    primary_error: PreflightError | None = None
    close_error: BaseException | None = None
    try:
        descriptor = os.open("/", flags)
        for part in path.parts[1:]:
            if part in {"", ".", ".."}:
                raise ScopeViolation(f"{label} contains an unsafe component")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            previous_descriptor = descriptor
            descriptor = next_descriptor
            transition_error = _close_once(previous_descriptor)
            if transition_error is not None:
                raise ScopeViolation(f"{label} directory transition close failed") from transition_error
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ScopeViolation(f"{label} is not a directory")
        result = descriptor
        descriptor = None
    except ScopeViolation as exc:
        primary_error = exc
    except BaseException as exc:
        primary_error = ScopeViolation(f"{label} could not be opened without following links")
        primary_error.__cause__ = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
    if primary_error is not None:
        raise primary_error
    if close_error is not None:
        raise ScopeViolation(f"{label} directory close failed") from close_error
    if result is None:
        raise ScopeViolation(f"{label} directory open produced no descriptor")
    return result


def _read_regular_snapshot(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    _reject_forbidden_path(path, label=label)
    absolute = _absolute_without_resolving(path)
    _reject_forbidden_path(absolute, label=label)
    parent_fd: int | None = None
    descriptor: int | None = None
    result: tuple[bytes, str] | None = None
    primary_error: PreflightError | None = None
    close_errors: list[BaseException] = []
    try:
        parent_fd = _open_directory_chain(absolute.parent, label=f"{label} parent")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ScopeViolation(f"{label} is not a unique regular file")
        if type(maximum_bytes) is not int or maximum_bytes <= 0 or opened.st_size > maximum_bytes:
            raise ScopeViolation(f"{label} exceeds the closed byte bound")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise ScopeViolation(f"{label} exceeds the closed byte bound")
            digest.update(block)
            chunks.append(block)
        final = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        entry = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            opened_identity != final_identity
            or total != opened.st_size
            or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ScopeViolation(f"{label} changed during descriptor capture")
        result = (b"".join(chunks), digest.hexdigest())
    except ScopeViolation as exc:
        primary_error = exc
    except BaseException as exc:
        primary_error = ScopeViolation(f"{label} could not be read safely")
        primary_error.__cause__ = exc
    finally:
        if descriptor is not None:
            error = _close_once(descriptor)
            if error is not None:
                close_errors.append(error)
        if parent_fd is not None:
            error = _close_once(parent_fd)
            if error is not None:
                close_errors.append(error)
    if primary_error is not None:
        raise primary_error
    if close_errors:
        raise ScopeViolation(f"{label} descriptor close failed") from close_errors[0]
    if result is None:
        raise ScopeViolation(f"{label} descriptor capture produced no result")
    return result


def _validate_binding(binding: Any) -> str:
    if type(binding) is not dict or set(binding) != EXPECTED_BINDING_KEYS:
        raise ProtocolError("implementation binding field set is not exact")
    static_expected = {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "script_repo_path": "scripts/route_a_v3/preflight_gse200304_raw_replay.py",
        "test_repo_path": "tests/route_a_v3/test_preflight_gse200304_raw_replay.py",
    }
    for key, expected in static_expected.items():
        _require_strict(binding.get(key), expected, label=f"binding {key}")
    status = binding.get("status")
    if status == "UNKNOWN_NOT_ASSERTED":
        expected_dynamic = {
            "status": "UNKNOWN_NOT_ASSERTED",
            "production_implementation_commit": "UNKNOWN_NOT_ASSERTED",
            "production_script_sha256": "UNKNOWN_NOT_ASSERTED",
            "hard_blocker": "PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN",
        }
        for key, expected in expected_dynamic.items():
            _require_strict(binding.get(key), expected, label=f"UNKNOWN binding {key}")
        return status
    if status == "BOUND":
        if type(binding.get("production_implementation_commit")) is not str or COMMIT_RE.fullmatch(
            binding["production_implementation_commit"]
        ) is None:
            raise ProtocolError("BOUND implementation commit is not exact")
        if type(binding.get("production_script_sha256")) is not str or SHA256_RE.fullmatch(
            binding["production_script_sha256"]
        ) is None:
            raise ProtocolError("BOUND implementation script SHA-256 is not exact")
        if binding.get("hard_blocker") is not None:
            raise ProtocolError("BOUND implementation hard blocker must be null")
        return status
    raise ProtocolError("implementation binding status is outside the closed enum")


def _validate_protocol(protocol: Any) -> str:
    expected_top_level = {
        "schema_version",
        "protocol_id",
        "protocol_trust",
        "protocol_status",
        "terminal_status",
        "target_subseries_accession",
        "superseries_accession",
        "ordinary_public_data_only",
        "implementation_binding",
        "adapter_bindings",
        "acquisition_contract",
        "reference_contract",
        "sample_sheet_contract",
        "author_method_contract",
        "count_policy_contract",
        "denominator_discrepancy",
        "hard_unknown_blockers",
        "execution_policy",
        "gate_truth",
        "output_contract",
        "claim_boundary",
    }
    if type(protocol) is not dict or set(protocol) != expected_top_level:
        raise ProtocolError("protocol top-level field set is not exact")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("protocol_id", PROTOCOL_ID),
        ("protocol_status", "P0_PREFLIGHT_ONLY"),
        ("terminal_status", BLOCKED_OUTCOME),
        ("target_subseries_accession", "GSE200302"),
        ("superseries_accession", "GSE200304"),
        ("ordinary_public_data_only", True),
    ):
        _require_strict(protocol.get(key), expected, label=key)
    expected_trust = {
        "canonicalization": CANONICALIZATION,
        "normalized_binding_dynamic_fields": [
            "hard_blocker",
            "production_implementation_commit",
            "production_script_sha256",
            "status",
        ],
        "normalized_binding_dependent_fields": ["hard_unknown_blockers"],
        "purpose": "ALLOW_CONFIG_ONLY_UNKNOWN_TO_BOUND_BINDING_WITHOUT_SCRIPT_SELF_REFERENCE",
    }
    _require_strict(protocol.get("protocol_trust"), expected_trust, label="protocol trust")
    binding_status = _validate_binding(protocol["implementation_binding"])
    expected_blockers = (
        list(EXPECTED_HARD_BLOCKERS)
        if binding_status == "UNKNOWN_NOT_ASSERTED"
        else list(NON_BINDING_HARD_BLOCKERS)
    )
    _require_strict(
        protocol.get("hard_unknown_blockers"),
        expected_blockers,
        label=f"{binding_status} blocker list",
    )

    adapters = protocol.get("adapter_bindings")
    _require_strict(
        adapters,
        {
            "acquisition": {
                "repo_path": "scripts/route_a_v3/acquire_gse200304_fastq.py",
                "sha256": EXPECTED_ACQUIRER_SHA256,
                "load_mode": "HASH_PINNED_LAZY_SOURCE_VERIFICATION_ONLY",
                "allowed_operation": "READ_COMMITTED_TERMINAL_METADATA_ONLY",
                "fastq_body_read_allowed": False,
            },
            "reference": {
                "repo_path": "scripts/route_a_v3/qualify_gse200304_a1.py",
                "sha256": EXPECTED_QUALIFIER_SHA256,
                "load_mode": "HASH_PINNED_LAZY_SOURCE_VERIFICATION_ONLY",
                "allowed_operation": "READ_HASH_PINNED_DESIGN_AND_RETURN_AGGREGATES_ONLY",
                "row_identifiers_or_sequences_may_be_returned": False,
            },
        },
        label="adapter bindings",
    )

    acquisition = protocol.get("acquisition_contract")
    if type(acquisition) is not dict:
        raise ProtocolError("acquisition contract must be an object")
    for key, expected in (
        ("target_subseries_accession", "GSE200302"),
        ("superseries_accession", "GSE200304"),
        ("source_marker_dataset_accession_semantics", "SUPERSET_SUPERSERIES_ONLY"),
        ("bioproject_accession", "PRJNA824033"),
        ("layout", "PAIRED"),
        ("expected_run_count", EXPECTED_RUN_COUNT),
        ("expected_fastq_file_count", EXPECTED_FILE_COUNT),
        ("expected_total_fastq_bytes", EXPECTED_TOTAL_BYTES),
        ("expected_run_accession_set_sha256_lf_sorted", EXPECTED_RUN_SET_SHA256),
        ("committed_acquisition_required", True),
        ("fastq_body_read_count_by_preflight", 0),
        ("adapter_result_exact_keys", list(EXPECTED_ACQUISITION_AUDIT)),
    ):
        _require_strict(acquisition.get(key), expected, label=f"acquisition {key}")

    reference = protocol.get("reference_contract")
    if type(reference) is not dict:
        raise ProtocolError("reference contract must be an object")
    for key, expected in (
        ("expected_record_count", EXPECTED_REFERENCE_RECORDS),
        ("expected_unique_sequence_count", EXPECTED_REFERENCE_UNIQUE),
        ("expected_identical_sequence_group_count", EXPECTED_IDENTICAL_REFERENCE_GROUPS),
        ("expected_sequence_length", EXPECTED_REFERENCE_LENGTH),
        ("expected_sequence_alphabet", "ACGT"),
        ("identifier_output_allowed", False),
        ("sequence_output_allowed", False),
        ("identical_reference_multimap_tie_status", "UNKNOWN_NOT_ASSERTED"),
    ):
        _require_strict(reference.get(key), expected, label=f"reference {key}")

    method = protocol.get("author_method_contract")
    if type(method) is not dict:
        raise ProtocolError("author method contract must be an object")
    for key, expected in (
        ("bowtie2_version", "2.4.2"),
        ("bowtie2_build_argv_template", list(EXPECTED_BUILD_ARGV_TEMPLATE)),
        ("bowtie2_alignment_argv_template", list(EXPECTED_ALIGNMENT_ARGV_TEMPLATE)),
        ("argv_builders_are_pure_and_nonexecuting", True),
        ("r_version", "4.2.0"),
        ("xtail_version", "1.1.15"),
        ("xtail_bins", 1000),
        ("multiple_testing_adjustment", "BH"),
        ("biological_replicates_per_condition", 6),
    ):
        _require_strict(method.get(key), expected, label=f"method {key}")
    _require_strict(
        method.get("xtail_endpoints"),
        [
            {
                "endpoint": "COMBINED_POLY_VS_TOTAL_RNA",
                "numerator_selector": "poly",
                "denominator_selector": "Total",
            },
            {
                "endpoint": "HIGH_POLY_VS_TOTAL_RNA",
                "numerator_selector": "High",
                "denominator_selector": "Total",
            },
        ],
        label="xTail endpoint contract",
    )
    _require_strict(protocol.get("execution_policy"), EXPECTED_EXECUTION_POLICY, label="execution policy")
    _require_strict(protocol.get("gate_truth"), EXPECTED_GATE_TRUTH, label="gate truth")
    _require_strict(
        protocol.get("denominator_discrepancy"),
        EXPECTED_DENOMINATOR_DISCREPANCY,
        label="denominator discrepancy",
    )

    output = protocol.get("output_contract")
    if type(output) is not dict:
        raise ProtocolError("output contract must be an object")
    for key, expected in (
        ("aggregate_only", True),
        ("success_outcome", BLOCKED_OUTCOME),
        ("failure_outcome", FAILURE_OUTCOME),
        ("production_unbound_failure_code", "PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN"),
        ("valid_failure_codes", list(VALID_FAILURE_CODES)),
        ("success_files", [PREFLIGHT_FILENAME, SHA256SUMS_FILENAME, PUBLICATION_MARKER_FILENAME]),
        ("failure_files", [FAILURE_FILENAME, SHA256SUMS_FILENAME, PUBLICATION_MARKER_FILENAME]),
        ("terminal_marker", PUBLICATION_MARKER_FILENAME),
        ("terminal_marker_written_last", True),
        ("directory_and_parent_fsync_required", True),
        ("advisory_lock_required", False),
        ("no_overwrite", True),
        ("success_and_failure_mutually_exclusive", True),
        ("bundle_id_mode", "CONTENT_SHA256_PREFIX"),
        ("absolute_output_path_in_artifact_allowed", False),
        ("raw_payload_allowed", False),
        ("row_identifier_allowed", False),
        ("run_identifier_list_allowed", False),
        ("sequence_allowed", False),
    ):
        _require_strict(output.get(key), expected, label=f"output {key}")
    return binding_status


def _canonical_protocol_projection(protocol: Mapping[str, Any]) -> bytes:
    copied = json.loads(json.dumps(protocol, allow_nan=False))
    binding = copied["implementation_binding"]
    binding["status"] = "<BINDING_STATUS>"
    binding["production_implementation_commit"] = "<IMPLEMENTATION_COMMIT>"
    binding["production_script_sha256"] = "<IMPLEMENTATION_SCRIPT_SHA256>"
    binding["hard_blocker"] = "<BINDING_HARD_BLOCKER>"
    copied["hard_unknown_blockers"] = [
        "<IMPLEMENTATION_BINDING_BLOCKER_SLOT>",
        *NON_BINDING_HARD_BLOCKERS,
    ]
    return _json_bytes(copied, pretty=False)


def load_protocol(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute_without_resolving(path)
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename is outside the frozen allowlist")
    payload, full_sha256 = _read_regular_snapshot(
        path,
        label="raw replay protocol",
        maximum_bytes=MAX_PROTOCOL_BYTES,
    )
    protocol = _strict_json_object(payload, label="raw replay protocol")
    binding_status = _validate_protocol(protocol)
    projection = _canonical_protocol_projection(protocol)
    core_sha256 = _sha256_bytes(projection)
    if core_sha256 != PROTOCOL_CORE_SHA256:
        raise ProtocolError("protocol core projection differs from the compiled trust root")
    if binding_status == "BOUND":
        implementation_path = path.parent.parent / protocol["implementation_binding"]["script_repo_path"]
        try:
            _, observed_implementation_sha256 = _read_regular_snapshot(
                implementation_path,
                label="bound production implementation",
                maximum_bytes=8 * 1024 * 1024,
            )
        except PreflightError as exc:
            raise ProtocolError("BOUND production implementation could not be verified") from exc
        if observed_implementation_sha256 != protocol["implementation_binding"][
            "production_script_sha256"
        ]:
            raise ProtocolError("BOUND production implementation SHA-256 differs from source bytes")
    return protocol, {
        "full_file_sha256_observed": full_sha256,
        "core_projection_sha256": core_sha256,
        "canonicalization": CANONICALIZATION,
        "binding_status": binding_status,
    }


def audit_implementation_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    status = _validate_binding(protocol["implementation_binding"])
    if status == "UNKNOWN_NOT_ASSERTED":
        return {
            "status": "UNKNOWN_NOT_ASSERTED",
            "production_bound": False,
            "hard_blocker_present": True,
            "external_input_read_before_binding_audit": False,
        }
    return {
        "status": "BOUND",
        "production_bound": True,
        "hard_blocker_present": False,
        "external_input_read_before_binding_audit": False,
    }


@dataclass
class HashPinnedLazySource:
    source_path: Path
    expected_sha256: str
    logical_name: str
    _verified_sha256: str | None = field(default=None, init=False, repr=False)

    def verify(self) -> str:
        if self._verified_sha256 is not None:
            return self._verified_sha256
        _, observed = _read_regular_snapshot(
            self.source_path,
            label=f"{self.logical_name} adapter source",
            maximum_bytes=8 * 1024 * 1024,
        )
        if observed != self.expected_sha256:
            raise AdapterSourceError(f"{self.logical_name} adapter source hash drift")
        self._verified_sha256 = observed
        return observed


def _validate_implementation_evidence(value: Any) -> None:
    keys = {
        "status",
        "binding_mode",
        "implementation_commit",
        "implementation_script_sha256",
        "binding_commit",
        "protocol_sha256",
        "worktree_and_index_clean",
    }
    if type(value) is not dict or set(value) != keys:
        raise AcquisitionError("acquisition marker implementation evidence is not closed")
    if value.get("status") != "BOUND" or value.get("binding_mode") != "TWO_COMMIT_NON_SELF_REFERENTIAL":
        raise AcquisitionError("acquisition marker implementation evidence is not bound")
    for key in ("implementation_commit", "binding_commit"):
        if type(value.get(key)) is not str or COMMIT_RE.fullmatch(value[key]) is None:
            raise AcquisitionError("acquisition marker contains an invalid commit binding")
    for key in ("implementation_script_sha256", "protocol_sha256"):
        if type(value.get(key)) is not str or SHA256_RE.fullmatch(value[key]) is None:
            raise AcquisitionError("acquisition marker contains an invalid SHA-256 binding")
    if value.get("worktree_and_index_clean") is not True:
        raise AcquisitionError("acquisition marker worktree truth is not exact")


def _default_acquisition_audit(
    acquisition_directory: Path | str,
    source_binding: HashPinnedLazySource,
) -> dict[str, Any]:
    # Hash trust deliberately precedes every acquisition-target path operation.
    source_binding.verify()
    acquisition_directory = _absolute_without_resolving(acquisition_directory)
    try:
        payload, _ = _read_regular_snapshot(
            acquisition_directory / PUBLICATION_MARKER_FILENAME,
            label="acquisition terminal marker",
            maximum_bytes=MAX_MARKER_BYTES,
        )
    except PreflightError as exc:
        raise AcquisitionError("acquisition terminal marker could not be read safely") from exc
    marker = _strict_json_object(
        payload,
        label="acquisition terminal marker",
        error_type=AcquisitionError,
    )
    marker_keys = {
        "schema_version",
        "record_type",
        "dataset_accession",
        "bioproject_accession",
        "generated_at",
        "publication_status",
        "output_directory",
        "manifest_sha256",
        "source_terminal_marker_sha256",
        "implementation_binding",
        "member_set",
        "member_sha256",
        "verified_file_count",
        "verified_run_count",
        "verified_total_bytes",
        "repository_md5_verified_count",
        "local_sha256_recorded_count",
        "qualified_study_contribution",
        "training_allowed",
        "next_phase_authorized",
        "claim_boundary",
    }
    if set(marker) != marker_keys:
        raise AcquisitionError("acquisition marker field set is not exact")
    fixed = {
        "schema_version": "route_a_v3_gse200304_fastq_acquisition.v1",
        "record_type": "GSE200304_FASTQ_ACQUISITION_PUBLICATION_COMMIT",
        "dataset_accession": "GSE200304",
        "bioproject_accession": "PRJNA824033",
        "publication_status": "FASTQ_ACQUISITION_COMMITTED",
        "output_directory": os.fspath(acquisition_directory),
        "manifest_sha256": EXPECTED_ACQUISITION_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_ACQUISITION_SOURCE_MARKER_SHA256,
        "verified_file_count": EXPECTED_FILE_COUNT,
        "verified_run_count": EXPECTED_RUN_COUNT,
        "verified_total_bytes": EXPECTED_TOTAL_BYTES,
        "repository_md5_verified_count": EXPECTED_FILE_COUNT,
        "local_sha256_recorded_count": EXPECTED_FILE_COUNT,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
        "claim_boundary": EXPECTED_ACQUISITION_CLAIM,
    }
    for key, expected in fixed.items():
        if not _strict_equal(marker.get(key), expected):
            raise AcquisitionError("acquisition marker fixed metadata is not exact")
    _validate_timestamp(marker.get("generated_at"), error_type=AcquisitionError, label="acquisition timestamp")
    _validate_implementation_evidence(marker.get("implementation_binding"))

    member_set = marker.get("member_set")
    member_sha256 = marker.get("member_sha256")
    if type(member_set) is not list or any(type(name) is not str for name in member_set):
        raise AcquisitionError("acquisition marker member set is not a string list")
    if member_set != sorted(member_set) or len(member_set) != len(set(member_set)):
        raise AcquisitionError("acquisition marker member set is not unique and sorted")
    if type(member_sha256) is not dict or set(member_sha256) != set(member_set):
        raise AcquisitionError("acquisition marker member hash map is not exact")
    if any(type(value) is not str or SHA256_RE.fullmatch(value) is None for value in member_sha256.values()):
        raise AcquisitionError("acquisition marker member hash is invalid")

    metadata_members = {
        "ACQUISITION_BINDING.json",
        "ACQUISITION_STATUS.json",
        "FASTQ_INTEGRITY_MANIFEST.json",
        "SHA256SUMS",
    }
    observed_metadata: set[str] = set()
    fastq_members: set[str] = set()
    transfer_members: set[str] = set()
    run_mates: dict[str, set[int]] = {}
    unexpected = 0
    for name in member_set:
        if Path(name).name != name:
            raise AcquisitionError("acquisition marker contains a non-basename member")
        fastq_match = FASTQ_MEMBER_RE.fullmatch(name)
        transfer_match = TRANSFER_MEMBER_RE.fullmatch(name)
        if fastq_match is not None:
            run, mate_text = fastq_match.groups()
            fastq_members.add(name)
            run_mates.setdefault(run, set()).add(int(mate_text))
        elif transfer_match is not None:
            transfer_members.add(name)
        elif name in metadata_members:
            observed_metadata.add(name)
        else:
            unexpected += 1
    if unexpected != 0 or observed_metadata != metadata_members:
        raise AcquisitionError("acquisition marker contains an unexpected member class")
    if len(fastq_members) != EXPECTED_FILE_COUNT or len(transfer_members) != EXPECTED_FILE_COUNT:
        raise AcquisitionError("acquisition marker FASTQ/transfer member counts are not exact")
    if transfer_members != {f"{name}.transfer.json" for name in fastq_members}:
        raise AcquisitionError("acquisition marker transfer bindings do not match FASTQ members")
    if len(run_mates) != EXPECTED_RUN_COUNT or any(mates != {1, 2} for mates in run_mates.values()):
        raise AcquisitionError("acquisition marker does not contain exact paired mates")
    run_set_hash = _sha256_bytes(
        "".join(f"{run}\n" for run in sorted(run_mates)).encode("ascii")
    )
    if run_set_hash != EXPECTED_RUN_SET_SHA256:
        raise AcquisitionError("acquisition marker run set hash differs from frozen truth")
    # Identifiers and member names are dropped here and never enter an output audit.
    return dict(EXPECTED_ACQUISITION_AUDIT)


def audit_reference_records(records: Iterable[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    record_count = 0
    for raw_sequence in records:
        if type(raw_sequence) is not str:
            raise ReferenceAuditError("reference contains a non-string Full_Oligo")
        sequence = raw_sequence.upper()
        if ACGT_250_RE.fullmatch(sequence) is None:
            raise ReferenceAuditError("reference contains a non-250nt-ACGT Full_Oligo")
        counts[sequence] += 1
        record_count += 1
    audit = {
        "status": "PASS_EXACT_REFERENCE_AGGREGATE",
        "record_count": record_count,
        "unique_sequence_count": len(counts),
        "identical_sequence_group_count": sum(count > 1 for count in counts.values()),
        "sequence_length": EXPECTED_REFERENCE_LENGTH,
        "acgt_only": True,
        "normalized_uppercase": True,
        "identifier_output_count": 0,
        "sequence_output_count": 0,
    }
    validate_reference_audit(audit)
    return audit


def _default_reference_audit(
    reference_source: Path | str,
    source_binding: HashPinnedLazySource,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    # Hash trust deliberately precedes every reference-target path operation.
    source_binding.verify()
    contract = protocol["reference_contract"]
    try:
        payload, observed_sha = _read_regular_snapshot(
            _absolute_without_resolving(reference_source),
            label="reference design source",
            maximum_bytes=MAX_REFERENCE_COMPRESSED_BYTES,
        )
    except PreflightError as exc:
        raise ReferenceAuditError("reference design source could not be read safely") from exc
    if observed_sha != contract["source_asset_sha256"] or len(payload) != contract["source_asset_bytes"]:
        raise ReferenceAuditError("reference design source binding is not exact")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as handle:
            decompressed = handle.read(MAX_REFERENCE_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise ReferenceAuditError("reference design source is corrupt gzip data") from exc
    if len(decompressed) > MAX_REFERENCE_DECOMPRESSED_BYTES:
        raise ReferenceAuditError("reference design source exceeds the decompression bound")
    try:
        reader = csv.reader(io.StringIO(decompressed.decode("utf-8"), newline=""), delimiter="\t")
        header = next(reader)
        rows = list(reader)
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ReferenceAuditError("reference design source is not closed UTF-8 TSV") from exc
    if len(rows) != EXPECTED_REFERENCE_RECORDS or any(len(row) != len(header) for row in rows):
        raise ReferenceAuditError("reference design row geometry is not exact")
    try:
        sequence_index = header.index(contract["sequence_column"])
    except ValueError as exc:
        raise ReferenceAuditError("reference design lacks Full_Oligo") from exc
    return audit_reference_records(row[sequence_index] for row in rows)


def _default_source_bindings(
    protocol_path: Path | str,
    protocol: Mapping[str, Any],
) -> tuple[HashPinnedLazySource, HashPinnedLazySource]:
    root = _absolute_without_resolving(protocol_path).parent.parent
    adapters = protocol["adapter_bindings"]
    return (
        HashPinnedLazySource(
            root / adapters["acquisition"]["repo_path"],
            adapters["acquisition"]["sha256"],
            "gse200304_acquirer",
        ),
        HashPinnedLazySource(
            root / adapters["reference"]["repo_path"],
            adapters["reference"]["sha256"],
            "gse200304_qualifier",
        ),
    )


def validate_acquisition_audit(value: Any) -> dict[str, Any]:
    if not _strict_equal(value, EXPECTED_ACQUISITION_AUDIT):
        raise AcquisitionError("acquisition aggregate is not exact and type-strict")
    return dict(EXPECTED_ACQUISITION_AUDIT)


def validate_reference_audit(value: Any) -> dict[str, Any]:
    if not _strict_equal(value, EXPECTED_REFERENCE_AUDIT):
        raise ReferenceAuditError("reference aggregate is not exact and type-strict")
    return dict(EXPECTED_REFERENCE_AUDIT)


def validate_sample_sheet(document: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract = protocol["sample_sheet_contract"]
    if type(document) is not dict or set(document) != {"schema_version", "rows"}:
        raise SampleSheetError("sample sheet top-level structure is not exact")
    if type(document.get("schema_version")) is not str or document["schema_version"] != contract["schema_version"]:
        raise SampleSheetError("sample sheet schema version is not exact")
    rows = document.get("rows")
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise SampleSheetError("sample sheet rows are not an exact object list")
    expected_roles = {
        (family, replicate)
        for family in EXPECTED_SAMPLE_FAMILIES
        for replicate in EXPECTED_REPLICATES
    }
    roles: set[tuple[str, int]] = set()
    runs: set[str] = set()
    for row in rows:
        if set(row) != EXPECTED_SAMPLE_ROW_KEYS:
            raise SampleSheetError("sample sheet row field set is not exact")
        run = row.get("run_accession")
        family = row.get("measurement_family")
        replicate = row.get("replicate")
        if type(run) is not str or RUN_RE.fullmatch(run) is None:
            raise SampleSheetError("sample sheet run accession is invalid")
        if run in runs:
            raise SampleSheetError("sample sheet contains a duplicate run")
        runs.add(run)
        if type(family) is not str or type(replicate) is not int:
            raise SampleSheetError("sample sheet role types are invalid")
        role = (family, replicate)
        if role not in expected_roles or role in roles:
            raise SampleSheetError("sample sheet role grid is missing, duplicated, or invalid")
        roles.add(role)
        if type(row.get("mate_1_filename")) is not str or type(row.get("mate_2_filename")) is not str:
            raise SampleSheetError("sample sheet mate filename type is invalid")
        if row["mate_1_filename"] != f"{run}_1.fastq.gz" or row["mate_2_filename"] != f"{run}_2.fastq.gz":
            raise SampleSheetError("sample sheet mate filename binding is not exact")
    if roles != expected_roles or len(rows) != EXPECTED_RUN_COUNT:
        raise SampleSheetError("sample sheet role grid is incomplete")
    run_hash = _sha256_bytes("".join(f"{run}\n" for run in sorted(runs)).encode("ascii"))
    if run_hash != EXPECTED_RUN_SET_SHA256:
        raise SampleSheetError("sample sheet run set hash is not exact")
    return {
        "structural_status": "PASS_CLOSED_SCHEMA_UNTRUSTED_AUTHORITY",
        "row_count": 24,
        "unique_run_count": 24,
        "unique_role_slot_count": 24,
        "missing_role_count": 0,
        "duplicate_role_count": 0,
        "exact_run_set_hash_match": True,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _contains_unknown(value: Any) -> bool:
    if type(value) is str:
        return value.strip().upper() in {"UNKNOWN", "UNKNOWN_NOT_ASSERTED", "TBD", "NA"}
    if type(value) is list:
        return any(_contains_unknown(item) for item in value)
    if type(value) is dict:
        return any(_contains_unknown(item) for item in value.values())
    return False


def _validate_flag_list(value: Any) -> list[int]:
    if (
        type(value) is not list
        or any(type(item) is not int or not 0 <= item <= 65535 for item in value)
        or len(value) != len(set(value))
    ):
        raise CountPolicyError("SAM flag list is not exact and type-strict")
    return value


def validate_count_policy(document: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract = protocol["count_policy_contract"]
    if type(document) is not dict or set(document) != {"schema_version", "policy"}:
        raise CountPolicyError("count policy top-level structure is not exact")
    if type(document.get("schema_version")) is not str or document["schema_version"] != contract["schema_version"]:
        raise CountPolicyError("count policy schema version is not exact")
    policy = document.get("policy")
    if type(policy) is not dict or set(policy) != EXPECTED_COUNT_POLICY_KEYS:
        raise CountPolicyError("count policy field set is not exact")
    if _contains_unknown(policy):
        raise CountPolicyError("count policy contains an unresolved value")
    required_flags = _validate_flag_list(policy["required_sam_flags"])
    excluded_flags = _validate_flag_list(policy["excluded_sam_flags"])
    if set(required_flags) & set(excluded_flags):
        raise CountPolicyError("required and excluded SAM flags overlap")
    if type(policy["minimum_mapq"]) is not int or not 0 <= policy["minimum_mapq"] <= 255:
        raise CountPolicyError("minimum MAPQ is not an exact integer")
    increment = policy["count_increment"]
    if type(increment) not in {int, float} or increment <= 0 or (
        type(increment) is float and not math.isfinite(increment)
    ):
        raise CountPolicyError("count increment is not a positive finite number")
    categorical = EXPECTED_COUNT_POLICY_KEYS - {
        "required_sam_flags",
        "excluded_sam_flags",
        "minimum_mapq",
        "count_increment",
    }
    if any(type(policy[key]) is not str or not policy[key].strip() for key in categorical):
        raise CountPolicyError("count policy categorical type is not exact")
    return {
        "structural_status": "PASS_CLOSED_SCHEMA_UNTRUSTED_AUTHORITY",
        "policy_field_count": 14,
        "required_flag_count": len(required_flags),
        "excluded_flag_count": len(excluded_flags),
        "unknown_value_count": 0,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _argv_token(value: Path | str, *, label: str) -> str:
    token = os.fspath(value)
    if type(token) is not str or not token or "\x00" in token:
        raise ValueError(f"{label} must be a nonempty inert argv token")
    return token


def build_bowtie2_build_argv(reference_fasta: Path | str, index_prefix: Path | str) -> tuple[str, ...]:
    return (
        "bowtie2-build",
        _argv_token(reference_fasta, label="reference FASTA"),
        _argv_token(index_prefix, label="index prefix"),
    )


def build_bowtie2_alignment_argv(
    index_prefix: Path | str,
    mate_1_fastq: Path | str,
    mate_2_fastq: Path | str,
    output_sam: Path | str,
) -> tuple[str, ...]:
    return (
        "bowtie2",
        "-x",
        _argv_token(index_prefix, label="index prefix"),
        "-1",
        _argv_token(mate_1_fastq, label="mate 1 FASTQ"),
        "-2",
        _argv_token(mate_2_fastq, label="mate 2 FASTQ"),
        "-N",
        "0",
        "--no-sq",
        "--no-hd",
        "-S",
        _argv_token(output_sam, label="output SAM"),
    )


def _not_provided_sample_sheet_audit() -> dict[str, Any]:
    return {
        "structural_status": "NOT_PROVIDED_HARD_UNKNOWN",
        "row_count": 0,
        "unique_run_count": 0,
        "unique_role_slot_count": 0,
        "missing_role_count": 24,
        "duplicate_role_count": 0,
        "exact_run_set_hash_match": False,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _valid_sample_sheet_audit() -> dict[str, Any]:
    return {
        "structural_status": "PASS_CLOSED_SCHEMA_UNTRUSTED_AUTHORITY",
        "row_count": 24,
        "unique_run_count": 24,
        "unique_role_slot_count": 24,
        "missing_role_count": 0,
        "duplicate_role_count": 0,
        "exact_run_set_hash_match": True,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _not_provided_count_policy_audit() -> dict[str, Any]:
    return {
        "structural_status": "NOT_PROVIDED_HARD_UNKNOWN",
        "policy_field_count": 0,
        "required_flag_count": 0,
        "excluded_flag_count": 0,
        "unknown_value_count": 14,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _valid_count_policy_audit(required_flags: int = 2, excluded_flags: int = 6) -> dict[str, Any]:
    return {
        "structural_status": "PASS_CLOSED_SCHEMA_UNTRUSTED_AUTHORITY",
        "policy_field_count": 14,
        "required_flag_count": required_flags,
        "excluded_flag_count": excluded_flags,
        "unknown_value_count": 0,
        "authority_status": "UNKNOWN_NOT_ASSERTED",
        "closed_for_execution": False,
    }


def _confirmed_method() -> dict[str, Any]:
    return {
        "bowtie2_version": "2.4.2",
        "bowtie2_build_argv_template": list(EXPECTED_BUILD_ARGV_TEMPLATE),
        "bowtie2_alignment_argv_template": list(EXPECTED_ALIGNMENT_ARGV_TEMPLATE),
        "r_version": "4.2.0",
        "xtail_version": "1.1.15",
        "xtail_bins": 1000,
        "multiple_testing_adjustment": "BH",
        "biological_replicates_per_condition": 6,
        "xtail_endpoints": [
            {
                "endpoint": "COMBINED_POLY_VS_TOTAL_RNA",
                "numerator_selector": "poly",
                "denominator_selector": "Total",
            },
            {
                "endpoint": "HIGH_POLY_VS_TOTAL_RNA",
                "numerator_selector": "High",
                "denominator_selector": "Total",
            },
        ],
        "external_tool_execution_count": 0,
    }


def _validate_sample_audit(value: Any) -> None:
    if not (
        _strict_equal(value, _not_provided_sample_sheet_audit())
        or _strict_equal(value, _valid_sample_sheet_audit())
    ):
        raise PublicationError("sample-sheet aggregate schema is not exact")


def _validate_count_audit(value: Any) -> None:
    if not (
        _strict_equal(value, _not_provided_count_policy_audit())
        or _strict_equal(value, _valid_count_policy_audit())
    ):
        raise PublicationError("count-policy aggregate schema and values are not exact")


def _build_preflight_document(
    protocol: Mapping[str, Any],
    protocol_provenance: Mapping[str, Any],
    binding_audit: Mapping[str, Any],
    acquisition_audit: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
    sample_sheet_audit: Mapping[str, Any],
    count_policy_audit: Mapping[str, Any],
) -> dict[str, Any]:
    document = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "target_subseries_accession": "GSE200302",
        "superseries_accession": "GSE200304",
        "execution_outcome": BLOCKED_OUTCOME,
        "status": BLOCKED_OUTCOME,
        "protocol_core_sha256": protocol_provenance["core_projection_sha256"],
        "implementation_binding_audit": dict(binding_audit),
        "acquisition_audit": dict(acquisition_audit),
        "reference_audit": dict(reference_audit),
        "sample_sheet_audit": dict(sample_sheet_audit),
        "count_policy_audit": dict(count_policy_audit),
        "confirmed_method": _confirmed_method(),
        "denominator_discrepancy": json.loads(json.dumps(protocol["denominator_discrepancy"])),
        "hard_unknown_blockers": list(protocol["hard_unknown_blockers"]),
        "execution_policy": dict(EXPECTED_EXECUTION_POLICY),
        "gate_truth": dict(EXPECTED_GATE_TRUTH),
        "claim_boundary": protocol["claim_boundary"],
    }
    _validate_preflight_document(document)
    return document


def _validate_preflight_document(value: Any) -> None:
    keys = {
        "schema_version",
        "protocol_id",
        "target_subseries_accession",
        "superseries_accession",
        "execution_outcome",
        "status",
        "protocol_core_sha256",
        "implementation_binding_audit",
        "acquisition_audit",
        "reference_audit",
        "sample_sheet_audit",
        "count_policy_audit",
        "confirmed_method",
        "denominator_discrepancy",
        "hard_unknown_blockers",
        "execution_policy",
        "gate_truth",
        "claim_boundary",
    }
    if type(value) is not dict or set(value) != keys:
        raise PublicationError("preflight document top-level schema is not exact")
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "target_subseries_accession": "GSE200302",
        "superseries_accession": "GSE200304",
        "execution_outcome": BLOCKED_OUTCOME,
        "status": BLOCKED_OUTCOME,
        "protocol_core_sha256": PROTOCOL_CORE_SHA256,
        "implementation_binding_audit": {
            "status": "UNKNOWN_NOT_ASSERTED",
            "production_bound": False,
            "hard_blocker_present": True,
            "external_input_read_before_binding_audit": False,
        },
        "confirmed_method": _confirmed_method(),
        "hard_unknown_blockers": list(EXPECTED_HARD_BLOCKERS),
        "execution_policy": EXPECTED_EXECUTION_POLICY,
        "gate_truth": EXPECTED_GATE_TRUTH,
    }
    for key, expected in fixed.items():
        if not _strict_equal(value.get(key), expected):
            raise PublicationError(f"preflight {key} is not exact and type-strict")
    validate_acquisition_audit(value["acquisition_audit"])
    validate_reference_audit(value["reference_audit"])
    _validate_sample_audit(value["sample_sheet_audit"])
    _validate_count_audit(value["count_policy_audit"])
    discrepancy = value["denominator_discrepancy"]
    if type(discrepancy) is not dict or set(discrepancy) != {
        "status",
        "paper_reported",
        "current_mechanical_audit",
        "must_not_collapse_to_single_attrition_count",
    }:
        raise PublicationError("denominator discrepancy schema is not exact")
    expected_paper = {
        "designed_pair_count": 6892,
        "xtail_included_row_count": 6772,
        "attrition_count": 120,
        "evidence_status": "PAPER_REPORTED_NOT_RECONCILED_TO_CURRENT_DESIGN",
    }
    expected_current = {
        "design_pair_count": 6885,
        "processed_row_count": 6772,
        "attrition_count": 113,
        "evidence_status": "CURRENT_AUDITED_NOT_RECONCILED_TO_PAPER_DENOMINATOR",
    }
    if not _strict_equal(discrepancy.get("status"), "CONFLICT_UNRESOLVED"):
        raise PublicationError("denominator status is not exact")
    if not _strict_equal(discrepancy.get("paper_reported"), expected_paper):
        raise PublicationError("paper denominator evidence is not exact")
    if not _strict_equal(discrepancy.get("current_mechanical_audit"), expected_current):
        raise PublicationError("audited denominator evidence is not exact")
    if discrepancy.get("must_not_collapse_to_single_attrition_count") is not True:
        raise PublicationError("denominator separation truth is not exact")
    if type(value.get("claim_boundary")) is not str or value["claim_boundary"] != (
        "This aggregate-only P0 artifact records confirmed method facts and unresolved prerequisites. "
        "It permanently performs no alignment, SAM-to-count conversion, xTail analysis, qualification, "
        "canonicalization, training, model selection, or phase unlock."
    ):
        raise PublicationError("preflight claim boundary is not exact")
    _assert_json_tree(value, label="preflight document")


def _failure_payload(code: str) -> dict[str, Any]:
    if type(code) is not str or code not in VALID_FAILURE_CODES:
        raise PublicationError("failure code is outside the exact enum")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "execution_outcome": FAILURE_OUTCOME,
        "status": FAILURE_OUTCOME,
        "failure_code": code,
        "aggregate_only": True,
        "raw_payload_included": False,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "success_bundle_published": False,
    }


def _validate_failure_payload(value: Any) -> None:
    if type(value) is not dict or type(value.get("failure_code")) is not str:
        raise PublicationError("failure document structure or code type is invalid")
    expected = _failure_payload(value["failure_code"])
    if not _strict_equal(value, expected):
        raise PublicationError("failure document is not exactly equal to its closed factory")
    _assert_json_tree(value, label="failure document")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PublicationError("exclusive write made no progress")
        view = view[written:]


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
    _safe_basename(name, label="publication member")
    descriptor: int | None = None
    primary_error: PublicationError | None = None
    close_error: BaseException | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(payload):
            raise PublicationError("exclusive member identity is invalid")
    except FileExistsError as exc:
        primary_error = PublicationContention("exclusive member already exists")
        primary_error.__cause__ = exc
    except PublicationError as exc:
        primary_error = exc
    except BaseException as exc:
        primary_error = PublicationError("exclusive member write failed")
        primary_error.__cause__ = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
    if primary_error is not None:
        raise primary_error
    if close_error is not None:
        raise PublicationError("exclusive member close failed") from close_error


def _read_member_at(directory_fd: int, name: str, *, maximum_bytes: int) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise PublicationError("member byte bound is not an exact positive integer")
    descriptor: int | None = None
    result: bytes | None = None
    primary_error: PublicationError | None = None
    close_error: BaseException | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink < 1 or opened.st_size > maximum_bytes:
            raise PublicationError("member identity or size is invalid")
        payload = bytearray()
        while True:
            block = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise PublicationError("member exceeds the closed byte bound")
        final = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if (
            len(payload) != opened.st_size
            or opened_identity != final_identity
            or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PublicationError("member changed during validation")
        result = bytes(payload)
    except PublicationError as exc:
        primary_error = exc
    except BaseException as exc:
        primary_error = PublicationError("member read failed")
        primary_error.__cause__ = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
    if primary_error is not None:
        raise primary_error
    if close_error is not None:
        raise PublicationError("member close failed") from close_error
    if result is None:
        raise PublicationError("member capture produced no result")
    return result


def _bundle_identity(document_payload: bytes, sums_payload: bytes) -> tuple[str, str]:
    digest = _sha256_bytes(document_payload + b"\x00" + sums_payload)
    return f"route-a-v3-{digest[:24]}", digest


def _marker_payload(
    *,
    outcome: str,
    bundle_id: str,
    bundle_digest: str,
    member_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200302_TARGET_RAW_REPLAY_PREFLIGHT_PUBLICATION_COMMIT",
        "target_subseries_accession": "GSE200302",
        "superseries_accession": "GSE200304",
        "execution_outcome": outcome,
        "status": outcome,
        "generated_at": _utc_now(),
        "bundle_id": bundle_id,
        "bundle_digest": bundle_digest,
        "member_set": sorted(member_payloads),
        "member_sha256": {
            name: _sha256_bytes(member_payloads[name]) for name in sorted(member_payloads)
        },
        "aggregate_only": True,
        "raw_payload_included": False,
        "terminal_marker_written_last": True,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }


def _validate_marker_payload(
    value: Any,
    *,
    outcome: str,
    bundle_id: str,
    bundle_digest: str,
    member_payloads: Mapping[str, bytes],
) -> None:
    if type(value) is not dict:
        raise PublicationError("marker root must be an exact object")
    timestamp = _validate_timestamp(value.get("generated_at"), error_type=PublicationError, label="marker timestamp")
    expected = _marker_payload(
        outcome=outcome,
        bundle_id=bundle_id,
        bundle_digest=bundle_digest,
        member_payloads=member_payloads,
    )
    expected["generated_at"] = timestamp
    if not _strict_equal(value, expected):
        raise PublicationError("marker is not exact and type-strict")
    _assert_json_tree(value, label="publication marker")


def _trip_fault(faults: Mapping[str, BaseException] | None, phase: str) -> None:
    if faults is not None and phase in faults:
        raise faults[phase]


def _create_marker_stage(
    parent_fd: int,
    payload: bytes,
    *,
    faults: Mapping[str, BaseException] | None,
) -> str:
    for _ in range(16):
        name = f".route-a-v3-marker-{secrets.token_hex(16)}.stage"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        except BaseException as exc:
            raise PublicationError("marker staging inode could not be created") from exc
        operation_error: PublicationError | None = None
        try:
            if faults is not None and "marker_stage_partial_write" in faults:
                _write_all(descriptor, payload[: max(1, len(payload) // 2)])
                os.fsync(descriptor)
                raise faults["marker_stage_partial_write"]
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(payload):
                raise PublicationError("marker staging inode is invalid")
        except PublicationError as exc:
            operation_error = exc
        except BaseException as exc:
            operation_error = PublicationError("marker staging write failed")
            operation_error.__cause__ = exc
        close_error = _close_once(descriptor)
        if operation_error is None and close_error is None:
            return name
        cleanup_error: BaseException | None = None
        try:
            os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_error = exc
        if operation_error is not None:
            raise operation_error
        if close_error is not None:
            raise PublicationError("marker staging close failed") from close_error
        if cleanup_error is not None:
            raise PublicationError("failed to remove incomplete marker staging inode") from cleanup_error
    raise PublicationError("could not allocate a unique marker staging inode")


def _staged_marker_is_visible(parent_fd: int, output_fd: int, stage_name: str) -> bool:
    """Resolve only an in-progress hard-link commit ambiguity by inode identity."""

    try:
        staged = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        linked = os.stat(
            PUBLICATION_MARKER_FILENAME,
            dir_fd=output_fd,
            follow_symlinks=False,
        )
    except BaseException:
        return False
    return (staged.st_dev, staged.st_ino) == (linked.st_dev, linked.st_ino)


def _publish_closed_bundle(
    output_directory: Path | str,
    document: Mapping[str, Any],
    *,
    outcome: str,
    _faults: Mapping[str, BaseException] | None = None,
    _postcommit_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if outcome == BLOCKED_OUTCOME:
        _validate_preflight_document(document)
        json_name = PREFLIGHT_FILENAME
    elif outcome == FAILURE_OUTCOME:
        _validate_failure_payload(document)
        json_name = FAILURE_FILENAME
    else:
        raise PublicationError("outcome is outside the closed enum")
    document_payload = _json_bytes(document)
    sums_payload = f"{_sha256_bytes(document_payload)}  {json_name}\n".encode("ascii")
    member_payloads = {json_name: document_payload, SHA256SUMS_FILENAME: sums_payload}
    bundle_id, bundle_digest = _bundle_identity(document_payload, sums_payload)
    marker = _marker_payload(
        outcome=outcome,
        bundle_id=bundle_id,
        bundle_digest=bundle_digest,
        member_payloads=member_payloads,
    )
    _validate_marker_payload(
        marker,
        outcome=outcome,
        bundle_id=bundle_id,
        bundle_digest=bundle_digest,
        member_payloads=member_payloads,
    )
    marker_bytes = _json_bytes(marker)

    output_directory = _absolute_without_resolving(output_directory)
    _reject_forbidden_path(output_directory, label="output directory")
    _safe_basename(output_directory.name, label="output directory basename")
    parent_fd = _open_directory_chain(output_directory.parent, label="output parent")
    output_fd: int | None = None
    stage_name: str | None = None
    marker_visible = False
    marker_link_attempted = False
    write_trace: list[str] = []
    warnings: list[str] = []
    precommit_error: BaseException | None = None
    try:
        try:
            os.mkdir(output_directory.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise PublicationContention("exclusive output directory already exists") from exc
        output_fd = os.open(
            output_directory.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened = os.fstat(output_fd)
        if not stat.S_ISDIR(opened.st_mode) or os.listdir(output_fd):
            raise PublicationError("exclusive output namespace is not an empty directory")

        _write_exclusive_at(output_fd, json_name, document_payload)
        write_trace.append(json_name)
        _trip_fault(_faults, "after_document_write")
        _write_exclusive_at(output_fd, SHA256SUMS_FILENAME, sums_payload)
        write_trace.append(SHA256SUMS_FILENAME)

        # Every acceptance-critical check is completed before marker visibility.
        if set(os.listdir(output_fd)) != set(member_payloads):
            raise PublicationError("precommit member set is not exact")
        for name, expected_payload in member_payloads.items():
            if _read_member_at(output_fd, name, maximum_bytes=4 * 1024 * 1024) != expected_payload:
                raise PublicationError("precommit member bytes are not exact")
        current = os.stat(output_directory.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise PublicationError("precommit directory identity changed")
        _validate_marker_payload(
            marker,
            outcome=outcome,
            bundle_id=bundle_id,
            bundle_digest=bundle_digest,
            member_payloads=member_payloads,
        )
        _trip_fault(_faults, "precommit_output_fsync")
        os.fsync(output_fd)
        _trip_fault(_faults, "precommit_parent_fsync")
        os.fsync(parent_fd)

        try:
            stage_name = _create_marker_stage(parent_fd, marker_bytes, faults=_faults)
            if _read_member_at(parent_fd, stage_name, maximum_bytes=MAX_MARKER_BYTES) != marker_bytes:
                raise PublicationError("marker staging bytes are not exact")
            _trip_fault(_faults, "before_marker_link")
            try:
                marker_link_attempted = True
                os.link(
                    stage_name,
                    PUBLICATION_MARKER_FILENAME,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=output_fd,
                    follow_symlinks=False,
                )
                _trip_fault(_faults, "after_marker_link_before_visibility")
            except BaseException as link_error:
                # A remote filesystem may report an error after completing the
                # hard link.  Resolve that one commit-point ambiguity before
                # declaring the marker committed; after this branch no
                # acceptance-critical read/stat is permitted.
                if not _staged_marker_is_visible(parent_fd, output_fd, stage_name):
                    if isinstance(link_error, FileExistsError):
                        raise PublicationContention("terminal marker already exists") from link_error
                    raise link_error
                warnings.append("MARKER_LINK_RETURN_AMBIGUITY_RECOVERED")
            _trip_fault(_faults, "after_link_resolution_before_visibility")
            marker_visible = True
            write_trace.append(PUBLICATION_MARKER_FILENAME)
        except FileExistsError as exc:
            raise PublicationContention("terminal marker already exists") from exc

        # From this point onward no acceptance-critical read, list, or stat occurs.
        if stage_name is not None:
            try:
                os.unlink(stage_name, dir_fd=parent_fd)
                stage_name = None
            except BaseException:
                warnings.append("MARKER_STAGE_CLEANUP_WARNING")
        try:
            _trip_fault(_faults, "post_marker_output_fsync")
            os.fsync(output_fd)
        except BaseException:
            warnings.append("POST_MARKER_OUTPUT_FSYNC_WARNING")
        try:
            _trip_fault(_faults, "post_marker_parent_fsync")
            os.fsync(parent_fd)
        except BaseException:
            warnings.append("POST_MARKER_PARENT_FSYNC_WARNING")
    except BaseException as exc:
        if (
            not marker_visible
            and marker_link_attempted
            and stage_name is not None
            and output_fd is not None
            and _staged_marker_is_visible(parent_fd, output_fd, stage_name)
        ):
            marker_visible = True
            warnings.append("MARKER_VISIBILITY_RECOVERED_AFTER_EXCEPTION")
            if PUBLICATION_MARKER_FILENAME not in write_trace:
                write_trace.append(PUBLICATION_MARKER_FILENAME)
        if marker_visible:
            warnings.append("POST_MARKER_UNEXPECTED_WARNING")
        else:
            precommit_error = exc
    finally:
        if stage_name is not None:
            try:
                os.unlink(stage_name, dir_fd=parent_fd)
            except BaseException:
                if marker_visible:
                    warnings.append("MARKER_STAGE_CLEANUP_WARNING")
                elif precommit_error is None:
                    precommit_error = PublicationError("precommit marker staging cleanup failed")
        if output_fd is not None:
            try:
                os.close(output_fd)
                _trip_fault(_faults, "postcommit_close_output")
            except BaseException:
                if marker_visible:
                    warnings.append("POSTCOMMIT_OUTPUT_CLOSE_WARNING")
                elif precommit_error is None:
                    precommit_error = PublicationError("precommit output close failed")
        try:
            os.close(parent_fd)
            _trip_fault(_faults, "postcommit_close_parent")
        except BaseException:
            if marker_visible:
                warnings.append("POSTCOMMIT_PARENT_CLOSE_WARNING")
            elif precommit_error is None:
                precommit_error = PublicationError("precommit parent close failed")

    if not marker_visible:
        if isinstance(precommit_error, PublicationError):
            raise precommit_error
        raise PartialPrecommitError("publication stopped before atomic marker visibility") from precommit_error

    result = {
        "status": outcome,
        "execution_outcome": outcome,
        "published": True,
        "committed": True,
        "bundle_id": bundle_id,
        "bundle_digest": bundle_digest,
        "terminal_marker": PUBLICATION_MARKER_FILENAME,
        "terminal_marker_atomic_visibility": True,
        "write_trace": write_trace,
        "durability_warning": bool(warnings),
        "durability_warning_codes": sorted(set(warnings)),
        "no_overwrite": True,
        "aggregate_only": True,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }
    if _postcommit_observer is not None:
        try:
            _postcommit_observer(dict(result))
        except BaseException:
            result["durability_warning"] = True
            result["durability_warning_codes"] = sorted(
                set(result["durability_warning_codes"]) | {"POSTCOMMIT_OBSERVER_WARNING"}
            )
    return result


def validate_published_bundle(output_directory: Path | str) -> dict[str, Any]:
    output_directory = _absolute_without_resolving(output_directory)
    _reject_forbidden_path(output_directory, label="published bundle")
    _safe_basename(output_directory.name, label="published bundle basename")
    parent_fd = _open_directory_chain(output_directory.parent, label="published bundle parent")
    output_fd: int | None = None
    result: dict[str, Any] | None = None
    primary_error: PreflightError | None = None
    close_errors: list[BaseException] = []
    try:
        output_fd = os.open(
            output_directory.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened = os.fstat(output_fd)
        marker_bytes = _read_member_at(output_fd, PUBLICATION_MARKER_FILENAME, maximum_bytes=MAX_MARKER_BYTES)
        marker = _strict_json_object(
            marker_bytes,
            label="published marker",
            error_type=PublicationError,
        )
        outcome = marker.get("execution_outcome")
        if outcome == BLOCKED_OUTCOME:
            json_name = PREFLIGHT_FILENAME
        elif outcome == FAILURE_OUTCOME:
            json_name = FAILURE_FILENAME
        else:
            raise PublicationError("published outcome is outside the closed enum")
        expected_names = {json_name, SHA256SUMS_FILENAME, PUBLICATION_MARKER_FILENAME}
        if set(os.listdir(output_fd)) != expected_names:
            raise PublicationError("published success/failure member set is not exact")
        document_bytes = _read_member_at(output_fd, json_name, maximum_bytes=4 * 1024 * 1024)
        sums_bytes = _read_member_at(output_fd, SHA256SUMS_FILENAME, maximum_bytes=1024 * 1024)
        expected_sums = f"{_sha256_bytes(document_bytes)}  {json_name}\n".encode("ascii")
        if sums_bytes != expected_sums:
            raise PublicationError("published SHA256SUMS is not exact")
        document = _strict_json_object(
            document_bytes,
            label="published aggregate document",
            error_type=PublicationError,
        )
        if outcome == BLOCKED_OUTCOME:
            _validate_preflight_document(document)
        else:
            _validate_failure_payload(document)
        member_payloads = {json_name: document_bytes, SHA256SUMS_FILENAME: sums_bytes}
        bundle_id, bundle_digest = _bundle_identity(document_bytes, sums_bytes)
        _validate_marker_payload(
            marker,
            outcome=outcome,
            bundle_id=bundle_id,
            bundle_digest=bundle_digest,
            member_payloads=member_payloads,
        )
        current = os.stat(output_directory.name, dir_fd=parent_fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise PublicationError("published directory identity changed during consumer validation")
        result = {
            "status": outcome,
            "execution_outcome": outcome,
            "committed": True,
            "accepted": True,
            "bundle_id": bundle_id,
            "bundle_digest": bundle_digest,
            "aggregate_only": True,
            "qualified": False,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        }
    except PreflightError as exc:
        primary_error = exc
    except BaseException as exc:
        primary_error = PublicationError("published bundle could not be opened safely")
        primary_error.__cause__ = exc
    finally:
        if output_fd is not None:
            error = _close_once(output_fd)
            if error is not None:
                close_errors.append(error)
        error = _close_once(parent_fd)
        if error is not None:
            close_errors.append(error)
    if primary_error is not None:
        raise primary_error
    if close_errors:
        raise PublicationError("published bundle descriptor close failed") from close_errors[0]
    if result is None:
        raise PublicationError("published bundle validation produced no result")
    return result


def _production_paths_before_read(
    protocol_path: Path | str,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    for path, label in ((protocol_path, "protocol path"), (output_directory, "output directory")):
        _reject_forbidden_path(path, label=label)
    protocol = _absolute_without_resolving(protocol_path)
    output = _absolute_without_resolving(output_directory)
    _reject_forbidden_path(protocol, label="absolute protocol path")
    _reject_forbidden_path(output, label="absolute output directory")
    _safe_basename(output.name, label="output directory basename")
    if protocol == output:
        raise ScopeViolation("output path overlaps protocol authority")
    return protocol, output


def run_preflight(
    *,
    protocol_path: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    """Production entry: UNKNOWN binding immediately emits one closed failure."""

    protocol_path, output_directory = _production_paths_before_read(
        protocol_path,
        output_directory,
    )
    protocol, _ = load_protocol(protocol_path)
    binding = audit_implementation_binding(protocol)
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        return _publish_closed_bundle(
            output_directory,
            _failure_payload("PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN"),
            outcome=FAILURE_OUTCOME,
        )
    raise ProtocolError("BOUND production execution is not part of this P0-only scaffold")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_preflight(
            protocol_path=args.protocol,
            output_directory=args.output_directory,
        )
    except PreflightError as exc:
        print(json.dumps({"status": FAILURE_OUTCOME, "failure_code": exc.code}, sort_keys=True))
        return 2
    safe_result = {
        "status": result["status"],
        "execution_outcome": result["execution_outcome"],
        "committed": result["committed"],
        "bundle_id": result["bundle_id"],
        "bundle_digest": result["bundle_digest"],
        "durability_warning": result["durability_warning"],
        "durability_warning_codes": result["durability_warning_codes"],
    }
    print(json.dumps(safe_result, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
