#!/usr/bin/env python3
"""Reconcile the public GSE114002 endpoint and provisional pool geometry.

This v2 auditor is deliberately decision-neutral.  It can establish only that
the published ``rl`` column is mechanically reconstructed by the paper's
processed, normalized fraction columns and that a provisional pool rule has
the frozen aggregate geometry.  It never uses raw fraction counts as the
endpoint, never derives a standard error, never emits row-level material, and
never upgrades GSE114002 to an ordinary, A1, or true-A2 contribution.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
SCHEMA_VERSION = "3.0.0"
PROTOCOL_ID = "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2"
PROTOCOL_STATUS = "PREFROZEN_AGGREGATE_ONLY_FAIL_CLOSED_NOT_QUALIFIED"
DATASET_ID = "GSE114002"
DATA_ROLE = "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED"
MECHANICAL_STATUS = "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED"
FAILED_MECHANICAL_STATUS = "MECHANICAL_RECONCILIATION_FAILED_NOT_QUALIFIED"

PROTOCOL_BASENAME = "route_a_v3_gse114002_endpoint_geometry_reconciliation_v2.json"
SOURCE_BASENAME = "GSM3130443_designed_library.csv.gz"
EXPECTED_SOURCE_BYTES = 17_332_142
EXPECTED_SOURCE_SHA256 = (
    "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5"
)
EXPECTED_SOURCE_ROW_COUNT = 100_017
EXPECTED_HEADER: tuple[str, ...] = (
    "", "utr", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "total", "r0", "r1", "r2", "r3", "r4",
    "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13",
    "r_total", "rl", "id", "info1", "info2", "info3", "info4", "library",
    "mother", "designed", "match_score",
)
RAW_FRACTION_COLUMNS: tuple[str, ...] = tuple(str(index) for index in range(14))
NORMALIZED_FRACTION_COLUMNS: tuple[str, ...] = tuple(
    f"r{index}" for index in range(14)
)
FRACTION_WEIGHTS: tuple[float, ...] = (
    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.8, 5.8, 7.0, 8.2, 9.5, 10.5,
    12.0, 13.0,
)
LEGACY_INCORRECT_WEIGHTS: tuple[float, ...] = tuple(float(index) for index in range(14))
INCLUDED_LIBRARIES = frozenset({"human_utrs", "snv"})
EXPECTED_POOL_COUNT = 959
EXPECTED_CANDIDATE_COUNT = 3_899
EXPECTED_DISTANCE_COUNTS = {"1": 2_925, "2": 870, "3": 104}
EXPECTED_POOL_WITH_DISTANCE_3_COUNT = 91
EXPECTED_INCLUDED_DISTANCE_5_ROW_COUNT = 81
EXPECTED_ELIGIBLE_DISTANCE_5_UTR_HASH_COUNT = 4
EXPECTED_ELIGIBLE_POOL_WITH_DISTANCE_5_COUNT = 4
ENDPOINT_TOLERANCE = 5e-11
NORMALIZED_SUM_TOLERANCE = 5e-11
RAW_TOTAL_TOLERANCE = 1e-9

FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "gse246381", "access_log", "sealed", "restricted",
)
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "sequence", "sequences", "utr", "mother", "raw_row", "raw_rows",
        "id", "ids", "value", "values", "info1", "info2", "info3", "info4",
        "match_score",
    }
)
SENSITIVE_ARRAY_TOKENS = ("sequence", "utr", "mother", "raw_row", "id", "value")
DNA_RE = re.compile(r"^[ACGT]+$")
SENSITIVE_NUCLEOTIDE_RUN_RE = re.compile(r"[ACGTUacgtu]{26,}")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_KEY_DOMAIN = b"ROUTE_A_V3_GSE114002_PROVISIONAL_SOURCE_KEY_V2\0"

BASE_BLOCKERS: tuple[str, ...] = (
    "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
    "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED",
    "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN",
    "PREFROZEN_GROUP_POWER_NOT_RUN",
    "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
)
IMPLEMENTATION_BINDING_BLOCKER = "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED"
CONDITIONAL_BLOCKERS = frozenset(
    {
        "SOURCE_ROW_COUNT_RECONCILIATION_MISMATCH",
        "ENDPOINT_RECONCILIATION_MISMATCH",
        "NORMALIZED_FRACTION_SUM_RECONCILIATION_MISMATCH",
        "SUM_NORMALIZED_EQUIVALENT_RECONCILIATION_MISMATCH",
        "RAW_TOTAL_RECONCILIATION_MISMATCH",
        "TWO_STAGE_GLOBAL_NORMALIZATION_RECONCILIATION_MISMATCH",
        "MALFORMED_INCLUDED_ROWS_PRESENT",
        "PROVISIONAL_POOL_GEOMETRY_RECONCILIATION_MISMATCH",
        "HAMMING_DISTANCE_DISTRIBUTION_RECONCILIATION_MISMATCH",
        "K5_CLAIM_BOUNDARY_SCOPE_RECONCILIATION_MISMATCH",
    }
)
ALL_BLOCKERS = frozenset(BASE_BLOCKERS) | {
    IMPLEMENTATION_BINDING_BLOCKER,
} | CONDITIONAL_BLOCKERS

JSON_PAYLOAD_FILENAMES: tuple[str, ...] = (
    "INPUT_INTEGRITY_AUDIT.json",
    "ENDPOINT_RECONCILIATION_AUDIT.json",
    "POOL_GEOMETRY_RECONCILIATION_AUDIT.json",
    "QUALIFICATION_REPORT.json",
)
SHA256SUMS_FILENAME = "SHA256SUMS"
PUBLICATION_COMMIT_FILENAME = "PUBLICATION_COMMIT.json"
EXACT_BUNDLE_MEMBERS: tuple[str, ...] = (
    *JSON_PAYLOAD_FILENAMES,
    SHA256SUMS_FILENAME,
    PUBLICATION_COMMIT_FILENAME,
)
OUTPUT_ID = "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_BUNDLE"
PRIMARY_PUBLICATION_MODE = "KERNEL_ATOMIC_RENAME_NOREPLACE_V2"
FALLBACK_PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_MARKER_V2"
ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)

EXPECTED_AUTHORITY = {
    "contract_path": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
    "contract_sha256": "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
    "data_role_registry_path": "docs/execution/route_a_v3_data_role_registry.yaml",
    "data_role_registry_sha256": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
    "decision_log_path": "docs/execution/route_a_v3_decision_log.yaml",
    "decision_log_sha256": "a5b041fab24d9a4309603a085fa3fcab936d69a899285bfa752689a2ee5fd4fd",
    "active_authority_commit": "d328bf04c394d4960ac11058e079c063e09280af",
    "staging_parent_head": "998d030a51737bfa1e27580efe8b89e22ae39149",
}
IMPLEMENTATION_BINDING_UNKNOWN = {
    "status": "UNKNOWN_NOT_ASSERTED",
    "repository_root_rule": "PROTOCOL_PATH_PARENT_PARENT",
    "implementation_commit": "UNKNOWN_NOT_ASSERTED",
    "qualifier_path": "scripts/route_a_v3/audit_gse114002_endpoint_geometry_reconciliation_v2.py",
    "qualifier_blob_sha256": "UNKNOWN_NOT_ASSERTED",
    "test_path": "tests/route_a_v3/test_audit_gse114002_endpoint_geometry_reconciliation_v2.py",
    "test_blob_sha256": "UNKNOWN_NOT_ASSERTED",
    "implementation_commit_must_be_direct_child": True,
    "implementation_changed_paths": [
        "configs/route_a_v3_gse114002_endpoint_geometry_reconciliation_v2.json",
        "scripts/route_a_v3/audit_gse114002_endpoint_geometry_reconciliation_v2.py",
        "tests/route_a_v3/test_audit_gse114002_endpoint_geometry_reconciliation_v2.py",
    ],
    "post_implementation_allowed_changed_paths": [
        "configs/route_a_v3_gse114002_endpoint_geometry_reconciliation_v2.json"
    ],
    "binding_commit_must_be_direct_child": True,
    "current_head_must_strictly_descend": True,
    "active_authority_must_be_ancestor": True,
    "clean_worktree_required": True,
    "running_script_must_match_qualifier_blob": True,
}


class ReconciliationError(RuntimeError):
    """The evidence or execution state failed closed."""


class ScopeViolation(ReconciliationError):
    """A forbidden or unsafe path was rejected before payload access."""


class ProtocolError(ReconciliationError):
    """The prefrozen protocol or its Git binding is not exact."""


class BindingNotFrozen(ProtocolError):
    """Production binding is still UNKNOWN and source access is forbidden."""


class InputIntegrityError(ReconciliationError):
    """The ordinary-public asset differs from its frozen trust root."""


class BoundaryViolation(ReconciliationError):
    """A prohibited endpoint or uncertainty derivation was requested."""


class PublicationError(ReconciliationError):
    """The aggregate bundle could not be committed exactly."""


class PublicationContention(PublicationError):
    """A concurrent or prior writer already owns the final directory."""


class AtomicNoReplaceUnsupported(PublicationError):
    """The kernel no-replace rename primitive is unavailable."""

    def __init__(self, error_number: int) -> None:
        self.error_number = int(error_number)
        super().__init__(f"atomic no-replace unsupported: errno {error_number}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationError("output is not finite canonical JSON") from exc


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = child
        return value

    def reject_nonfinite(token: str) -> Any:
        raise ValueError(f"non-finite token {token}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"{label} is not strict finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} root is not an object")
    return value


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise ProtocolError(f"{label} differs from the frozen value")


def _require_closed_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ProtocolError(f"{label} keys differ from the closed schema")


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    folded = os.fspath(path).casefold()
    if any(token in folded for token in FORBIDDEN_PATH_TOKENS):
        raise ScopeViolation(f"{label} rejected before read by forbidden path policy")


def _absolute_without_resolving(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return Path(os.path.normpath(os.fspath(candidate)))
    return Path(os.path.abspath(os.fspath(candidate)))


def _preflight_paths_before_read(
    protocol_path: Path | str,
    source_path: Path | str,
    output_directory: Path | str,
) -> tuple[Path, Path, Path]:
    raw = (
        (Path(protocol_path), "protocol path"),
        (Path(source_path), "ordinary public source path"),
        (Path(output_directory), "output path"),
    )
    for path, label in raw:
        _reject_forbidden_path(path, label=label)
    protocol, source, output = tuple(
        _absolute_without_resolving(path) for path, _ in raw
    )
    for path, (_, label) in zip((protocol, source, output), raw):
        _reject_forbidden_path(path, label=label)
    if protocol.name != PROTOCOL_BASENAME:
        raise ScopeViolation("protocol basename is outside the frozen allowlist")
    if source.name != SOURCE_BASENAME:
        raise ScopeViolation("source basename is outside the frozen allowlist")
    if output in {protocol, protocol.parent, source, source.parent}:
        raise ScopeViolation("output overlaps an authority or input path")
    return protocol, source, output


def _open_regular_no_symlinks(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None:
        raise ScopeViolation(f"{label} requires O_NOFOLLOW and O_DIRECTORY")
    absolute = _absolute_without_resolving(path)
    flags_directory = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory_flag | nofollow
    )
    try:
        parent_fd = os.open(absolute.anchor, flags_directory)
    except OSError as exc:
        raise ScopeViolation(f"{label} root cannot be opened safely") from exc
    try:
        for component in absolute.parts[1:-1]:
            if component in {"", ".", ".."} or Path(component).name != component:
                raise ScopeViolation(f"{label} contains an unsafe component")
            try:
                next_fd = os.open(component, flags_directory, dir_fd=parent_fd)
            except OSError as exc:
                raise ScopeViolation(f"{label} parent contains a symlink") from exc
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                raise ScopeViolation(f"{label} parent is not a directory")
            os.close(parent_fd)
            parent_fd = next_fd
        flags_leaf = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow
        )
        try:
            descriptor = os.open(absolute.name, flags_leaf, dir_fd=parent_fd)
        except OSError as exc:
            raise ScopeViolation(f"{label} leaf cannot be opened safely") from exc
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            os.close(descriptor)
            raise ScopeViolation(f"{label} must be a single-link regular file")
        return descriptor, opened
    finally:
        os.close(parent_fd)


def _read_verified_snapshot(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    descriptor, opened = _open_regular_no_symlinks(path, label=label)
    identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
        opened.st_nlink,
    )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
        final.st_nlink,
    )
    if final_identity != identity:
        raise InputIntegrityError(f"{label} changed during same-descriptor capture")
    payload = b"".join(chunks)
    observed_sha256 = digest.hexdigest()
    if len(payload) != opened.st_size:
        raise InputIntegrityError(f"{label} byte count changed during capture")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise InputIntegrityError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise InputIntegrityError(f"{label} byte count mismatch")
    return payload, {
        "basename": path.name,
        "sha256": observed_sha256,
        "bytes": len(payload),
        "parser_input_mode": "SAME_DESCRIPTOR_VERIFIED_SNAPSHOT",
    }


def _read_source_snapshot(
    path: Path, *, expected_sha256: str, expected_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    return _read_verified_snapshot(
        path,
        label="ordinary public GSE114002 source",
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def _load_protocol(
    path: Path, launch_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if SHA256_RE.fullmatch(launch_sha256) is None:
        raise ProtocolError("launch protocol SHA-256 is invalid")
    payload, provenance = _read_verified_snapshot(
        path,
        label="GSE114002 v2 protocol",
        expected_sha256=launch_sha256,
    )
    protocol = _strict_json_object(payload, label="GSE114002 v2 protocol")
    _validate_protocol(protocol)
    return protocol, {**provenance, "launch_expected_sha256": launch_sha256}


def _validate_binding_document(binding: Mapping[str, Any]) -> None:
    if set(binding) != set(IMPLEMENTATION_BINDING_UNKNOWN):
        raise ProtocolError("implementation-binding keys differ from frozen schema")
    fixed_keys = set(IMPLEMENTATION_BINDING_UNKNOWN) - {
        "status", "implementation_commit", "qualifier_blob_sha256", "test_blob_sha256"
    }
    for key in fixed_keys:
        _require_exact(
            binding.get(key), IMPLEMENTATION_BINDING_UNKNOWN[key],
            label=f"implementation_binding.{key}",
        )
    if binding.get("status") == "UNKNOWN_NOT_ASSERTED":
        if dict(binding) != IMPLEMENTATION_BINDING_UNKNOWN:
            raise ProtocolError("UNKNOWN implementation binding is not exact")
        return
    if binding.get("status") != "BOUND":
        raise ProtocolError("implementation-binding status is outside the closed enum")
    if COMMIT_RE.fullmatch(str(binding.get("implementation_commit"))) is None:
        raise ProtocolError("bound implementation commit is invalid")
    for key in ("qualifier_blob_sha256", "test_blob_sha256"):
        if SHA256_RE.fullmatch(str(binding.get(key))) is None:
            raise ProtocolError(f"bound {key} is invalid")


def _expected_input_contract() -> dict[str, Any]:
    return {
        "ordinary_public_data_only": True,
        "asset_basename": SOURCE_BASENAME,
        "asset_accession": "GSM3130443",
        "compressed_bytes": EXPECTED_SOURCE_BYTES,
        "compressed_sha256": EXPECTED_SOURCE_SHA256,
        "compression": "gzip",
        "encoding": "utf-8",
        "delimiter": ",",
        "expected_source_row_count": EXPECTED_SOURCE_ROW_COUNT,
        "exact_header": list(EXPECTED_HEADER),
        "forbidden_path_tokens": ["GSE246381", "access_log", "sealed", "restricted"],
        "same_descriptor_verified_snapshot_required": True,
        "root_to_leaf_o_nofollow_required": True,
        "single_link_regular_file_required": True,
        "hash_and_size_verified_before_decompression": True,
    }


def _expected_endpoint_contract() -> dict[str, Any]:
    return {
        "paper_endpoint_column": "rl",
        "normalized_fraction_columns": list(NORMALIZED_FRACTION_COLUMNS),
        "fraction_weights": list(FRACTION_WEIGHTS),
        "paper_endpoint_formula": "SUM_RI_TIMES_FROZEN_WEIGHT_NO_DENOMINATOR",
        "normalized_fraction_sum_target": 1,
        "normalized_fraction_sum_tolerance": NORMALIZED_SUM_TOLERANCE,
        "expected_normalized_fraction_sum_match_count": EXPECTED_SOURCE_ROW_COUNT,
        "paper_endpoint_absolute_tolerance": ENDPOINT_TOLERANCE,
        "expected_paper_endpoint_match_count": EXPECTED_SOURCE_ROW_COUNT,
        "expected_paper_endpoint_max_abs_residual": 2.2500223906263273e-11,
        "paper_endpoint_max_abs_residual_cap": ENDPOINT_TOLERANCE,
        "expected_normalized_fraction_sum_max_abs_residual": 2.099875828776021e-12,
        "sum_normalized_equivalent_absolute_tolerance": ENDPOINT_TOLERANCE,
        "expected_sum_normalized_equivalent_match_count": EXPECTED_SOURCE_ROW_COUNT,
        "expected_sum_normalized_equivalent_max_abs_residual": (
            1.2160050744114415e-11
        ),
        "sum_normalized_equivalent_max_abs_residual_cap": ENDPOINT_TOLERANCE,
        "r_total_role": "PAPER_TWO_STAGE_GLOBAL_NORMALIZATION_ROW_SUM",
        "direct_second_division_of_stored_r_vector_allowed": False,
        "raw_fraction_count_columns": list(RAW_FRACTION_COLUMNS),
        "raw_total_column": "total",
        "raw_total_formula": "SUM_RAW_FRACTION_COUNTS",
        "raw_total_absolute_tolerance": RAW_TOTAL_TOLERANCE,
        "expected_raw_total_match_count": EXPECTED_SOURCE_ROW_COUNT,
        "expected_raw_total_max_abs_residual": 0,
        "raw_fraction_count_endpoint_role": (
            "RAW_RECONSTRUCTS_ONLY_VIA_PAPER_TWO_STAGE_GLOBAL_NORMALIZATION"
        ),
        "two_stage_global_normalization_required": True,
        "two_stage_global_fraction_totals_output_allowed": False,
        "two_stage_r_total_absolute_tolerance": 1e-12,
        "expected_two_stage_r_total_match_count": EXPECTED_SOURCE_ROW_COUNT,
        "two_stage_r_total_max_abs_residual_cap": 1e-12,
        "two_stage_stored_fraction_vector_absolute_tolerance": 1e-12,
        "expected_two_stage_stored_fraction_vector_match_count": (
            EXPECTED_SOURCE_ROW_COUNT
        ),
        "two_stage_stored_fraction_vector_max_abs_residual_cap": 1e-12,
        "naive_raw_row_weighted_endpoint_allowed": False,
        "prohibited_raw_weighted_endpoint_reference_not_recomputed": {
            "absolute_tolerance_match_counts": {
                "1e-08": 0,
                "1e-06": 1,
                "0.0001": 8,
            },
            "max_abs_residual": 1.1678941555455555,
            "runtime_recomputation_allowed": False,
            "role": "FROZEN_NEGATIVE_CONTROL_ONLY",
        },
        "technical_standard_error_derivation_allowed": False,
        "biological_replicate_status": "ABSENT_BY_DESIGN",
        "paper_standard_error": None,
        "paper_standard_error_status": "ABSENT_NOT_DERIVABLE",
        "biological_standard_error_derivation_allowed": False,
        "p_or_fdr_back_calculation_allowed": False,
        "owner_uncertainty_policy_status": "UNKNOWN_NOT_ASSERTED",
    }


def _expected_pool_contract() -> dict[str, Any]:
    return {
        "region": "5UTR",
        "included_libraries": ["human_utrs", "snv"],
        "fixed_assay_id": "SAMPLE_2019_POLYSOME_FRACTIONATION",
        "fixed_context_id": "SAMPLE_2019_DESIGNED_LIBRARY_CONTEXT_UNRESOLVED",
        "fixed_endpoint_id": "MEAN_RIBOSOME_LOAD",
        "source_key_fields": [
            "mother", "library", "FIXED_ASSAY_ID", "FIXED_CONTEXT_ID",
            "FIXED_ENDPOINT_ID",
        ],
        "source_key_status": (
            "PROVISIONAL_PENDING_FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY"
        ),
        "identity_rule": "EXACTLY_ONE_DESIGNED_TRUE_ROW_WITH_UTR_EQUAL_TO_MOTHER",
        "candidate_rule": "DESIGNED_FALSE_EQUAL_LENGTH_HAMMING_DISTANCE_1_TO_3",
        "minimum_distinct_candidate_count_per_pool": 3,
        "expected_included_library_record_count": 50_600,
        "expected_malformed_included_record_count": 0,
        "expected_valid_out_of_rule_included_record_count": 8_909,
        "expected_provisional_source_pool_count": 23_177,
        "expected_valid_rule_record_count_in_noneligible_pools": 36_833,
        "expected_eligible_rule_record_count": 4_858,
        "expected_eligible_identity_record_count": 959,
        "expected_eligible_provisional_pool_count": EXPECTED_POOL_COUNT,
        "expected_eligible_distinct_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "expected_hamming_distance_candidate_counts": EXPECTED_DISTANCE_COUNTS,
        "expected_pool_with_hamming_distance_candidate_counts": {
            "1": 952, "2": 574, "3": 91,
        },
        "expected_pool_with_hamming_distance_3_candidate_count": (
            EXPECTED_POOL_WITH_DISTANCE_3_COUNT
        ),
        "expected_included_valid_designed_false_hamming_distance_5_row_count": (
            EXPECTED_INCLUDED_DISTANCE_5_ROW_COUNT
        ),
        "expected_eligible_pool_distinct_hamming_distance_5_utr_hash_count": (
            EXPECTED_ELIGIBLE_DISTANCE_5_UTR_HASH_COUNT
        ),
        "expected_eligible_pool_with_hamming_distance_5_count": (
            EXPECTED_ELIGIBLE_POOL_WITH_DISTANCE_5_COUNT
        ),
        "k5_role": (
            "CLAIM_BOUNDARY_ONLY_AFTER_K1_TO_K3_ELIGIBILITY_NOT_ELIGIBILITY_"
            "QUALIFICATION_OR_TRUE_A2_EVIDENCE"
        ),
        "expected_counts_are_field_or_source_authority": False,
        "raw_id_collision_role": (
            "AGGREGATE_COLLISION_AUDIT_ONLY_NOT_AUTOMATIC_BLOCKER"
        ),
        "expected_global_raw_id_audit": {
            "blank_record_count": 53_029,
            "nonblank_distinct_token_count": 25_606,
            "duplicated_nonblank_token_count": 4_776,
            "record_count_in_duplicated_nonblank_tokens": 26_158,
            "maximum_nonblank_token_multiplicity": 355,
        },
        "expected_included_scope_raw_id_audit": {
            "blank_record_count": 27_048,
            "nonblank_distinct_token_count": 23_552,
            "duplicated_nonblank_token_count": 0,
            "record_count_in_duplicated_nonblank_tokens": 0,
            "maximum_nonblank_token_multiplicity": 1,
        },
        "expected_first_unnamed_index_audit": {
            "distinct_token_count": 53_029,
            "duplicate_excess_record_count": 46_988,
        },
        "canonical_identity_requirement": (
            "FUTURE_COMPOSITE_OR_HASH_IDENTITY_PENDING_AUTHORITY"
        ),
        "valid_out_of_rule_included_row_status": "OUT_OF_SCOPE_DISPOSITION_PENDING",
        "valid_out_of_rule_included_rows_are_malformed": False,
        "malformed_included_rows_block_mechanical_reconciliation": True,
        "near_duplicate_sequence_cluster_audit_status": "NOT_RUN",
        "split_role": "DEVELOPMENT_ONLY",
    }


def _expected_decision_boundary() -> dict[str, Any]:
    return {
        "status": MECHANICAL_STATUS,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "mechanical_reconciliation_may_upgrade_qualification": False,
    }


def _expected_publication_contract() -> dict[str, Any]:
    return {
        "output_id": OUTPUT_ID,
        "aggregate_only": True,
        "exact_bundle_members": list(EXACT_BUNDLE_MEMBERS),
        "sha256sums_member_names": sorted(JSON_PAYLOAD_FILENAMES),
        "forbidden_output_fields": [
            "sequence", "sequences", "utr", "mother", "raw_row", "raw_rows",
            "id", "ids", "value", "values", "info1", "info2", "info3", "info4",
            "match_score",
        ],
        "sequence_id_row_or_value_arrays_allowed": False,
        "all_string_scalar_nucleotide_scan_minimum_length": 26,
        "atomic_no_overwrite_required": True,
        "primary_publication_mode": PRIMARY_PUBLICATION_MODE,
        "fallback_publication_mode": FALLBACK_PUBLICATION_MODE,
        "commit_marker_filename": PUBLICATION_COMMIT_FILENAME,
        "commit_marker_written_last": True,
        "commit_marker_validation_required": True,
        "unmarked_directory_status": "PARTIAL_NOT_COMMITTED_NOT_ACCEPTED",
        "publication_result_states": [
            "PUBLISHED",
            "COMMITTED_WITH_POST_COMMIT_WARNING",
            "COMMITTED_NOT_ACCEPTED",
            "ALREADY_COMMITTED_EXACT",
            "PARTIAL_REQUIRES_MANUAL_ADJUDICATION",
            "PRECOMMIT_STAGING_REQUIRES_MANUAL_ADJUDICATION",
        ],
        "post_visible_validation_failure_status": "COMMITTED_NOT_ACCEPTED",
        "post_commit_durability_failure_status": (
            "COMMITTED_WITH_POST_COMMIT_WARNING"
        ),
        "fallback_pre_marker_failure_status": (
            "PARTIAL_REQUIRES_MANUAL_ADJUDICATION"
        ),
        "retry_on_visible_unmarked_final_allowed": False,
        "existing_exact_output_status": "ALREADY_COMMITTED_EXACT",
        "existing_output_requires_current_expected_member_bytes": True,
        "rename_error_requires_final_classification_before_staging_cleanup": True,
        "precommit_staging_is_accepted": False,
        "unsupported_fallback_requires_safe_primary_staging_cleanup": True,
        "commit_acceptance_requires_final_target_binding": True,
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    top_keys = {
        "contract_id", "schema_version", "protocol_id", "protocol_status",
        "dataset_id", "data_role", "authority", "implementation_binding",
        "decision_neutral_boundary", "input_contract",
        "endpoint_reconciliation_contract", "pool_geometry_contract",
        "unresolved_blockers", "publication_contract",
    }
    _require_closed_keys(protocol, top_keys, label="protocol")
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "dataset_id": DATASET_ID,
        "data_role": DATA_ROLE,
    }.items():
        _require_exact(protocol.get(key), expected, label=f"protocol.{key}")
    _require_exact(protocol.get("authority"), EXPECTED_AUTHORITY, label="authority")
    binding = protocol.get("implementation_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("implementation binding is not an object")
    _validate_binding_document(binding)
    _require_exact(
        protocol.get("decision_neutral_boundary"), _expected_decision_boundary(),
        label="decision-neutral boundary",
    )
    _require_exact(
        protocol.get("input_contract"), _expected_input_contract(),
        label="input contract",
    )
    _require_exact(
        protocol.get("endpoint_reconciliation_contract"),
        _expected_endpoint_contract(), label="endpoint reconciliation contract",
    )
    _require_exact(
        protocol.get("pool_geometry_contract"), _expected_pool_contract(),
        label="pool geometry contract",
    )
    expected_blockers = list(BASE_BLOCKERS)
    if binding.get("status") == "UNKNOWN_NOT_ASSERTED":
        expected_blockers.append(IMPLEMENTATION_BINDING_BLOCKER)
    _require_exact(
        protocol.get("unresolved_blockers"), expected_blockers,
        label="unresolved blockers",
    )
    _require_exact(
        protocol.get("publication_contract"), _expected_publication_contract(),
        label="publication contract",
    )


def _git_capture(repository_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("Git binding query failed closed") from exc
    if completed.returncode != 0:
        raise ProtocolError("Git binding query failed closed")
    return completed.stdout


def _git_is_ancestor(repository_root: Path, older: str, newer: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository_root), "merge-base", "--is-ancestor", older, newer],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("Git ancestry query failed closed") from exc
    if completed.returncode not in {0, 1}:
        raise ProtocolError("Git ancestry query failed closed")
    return completed.returncode == 0


def _validate_i_to_b_protocol_transition(
    implementation_protocol: Mapping[str, Any],
    binding_protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    if implementation_protocol.get("implementation_binding") != (
        IMPLEMENTATION_BINDING_UNKNOWN
    ):
        raise ProtocolError("implementation commit protocol is not exact UNKNOWN-I")
    if binding_protocol.get("implementation_binding") != dict(binding):
        raise ProtocolError("binding commit protocol does not contain running binding")
    if implementation_protocol.get("unresolved_blockers") != [
        *BASE_BLOCKERS, IMPLEMENTATION_BINDING_BLOCKER,
    ]:
        raise ProtocolError("implementation commit blocker set is not exact UNKNOWN-I")
    if binding_protocol.get("unresolved_blockers") != list(BASE_BLOCKERS):
        raise ProtocolError("binding commit scientific blocker set drifted")
    implementation_core = dict(implementation_protocol)
    binding_core = dict(binding_protocol)
    for document in (implementation_core, binding_core):
        document.pop("implementation_binding", None)
        document.pop("unresolved_blockers", None)
    if implementation_core != binding_core:
        raise ProtocolError("config-only B changed protocol core")


def _verify_implementation_binding(
    binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    repository_root: Path,
    *,
    running_script_path: Path | None = None,
) -> dict[str, Any]:
    _validate_binding_document(binding)
    if binding.get("status") == "UNKNOWN_NOT_ASSERTED":
        return {
            "status": "UNKNOWN_NOT_ASSERTED",
            "verified": False,
            "implementation_commit": "UNKNOWN_NOT_ASSERTED",
            "binding_commit": "UNKNOWN_NOT_ASSERTED",
            "clean_worktree": "UNKNOWN_NOT_ASSERTED",
            "implementation_direct_child_of_staging_parent": "UNKNOWN_NOT_ASSERTED",
            "implementation_changed_paths_exact": "UNKNOWN_NOT_ASSERTED",
            "config_only_direct_child": "UNKNOWN_NOT_ASSERTED",
            "active_authority_blobs_match": "UNKNOWN_NOT_ASSERTED",
            "head_authority_blobs_match": "UNKNOWN_NOT_ASSERTED",
            "implementation_blobs_match": "UNKNOWN_NOT_ASSERTED",
            "running_script_matches_bound_blob": "UNKNOWN_NOT_ASSERTED",
        }
    root = _absolute_without_resolving(repository_root)
    head = _git_capture(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if COMMIT_RE.fullmatch(head) is None:
        raise ProtocolError("current Git HEAD is invalid")
    if _git_capture(root, ["status", "--porcelain=v1", "--untracked-files=all"]):
        raise ProtocolError("bound execution requires a clean worktree")
    active = str(authority["active_authority_commit"])
    staging_parent = str(authority["staging_parent_head"])
    implementation = str(binding["implementation_commit"])
    if active == staging_parent or not _git_is_ancestor(root, active, staging_parent):
        raise ProtocolError("active authority is not a strict staging-parent ancestor")
    implementation_parents = _git_capture(
        root, ["rev-list", "--parents", "-n", "1", implementation]
    ).decode("ascii").split()
    if len(implementation_parents) != 2 or implementation_parents[1] != staging_parent:
        raise ProtocolError("implementation is not the direct staging-parent child")
    implementation_changed = _git_capture(
        root, ["diff", "--name-only", staging_parent, implementation, "--"]
    ).decode("utf-8").splitlines()
    if implementation_changed != list(binding["implementation_changed_paths"]):
        raise ProtocolError("implementation changed paths are not exactly the v2 trio")
    parents = _git_capture(root, ["rev-list", "--parents", "-n", "1", head]).decode(
        "ascii"
    ).split()
    if len(parents) != 2 or parents[1] != implementation:
        raise ProtocolError("binding commit is not the direct implementation child")
    changed = _git_capture(
        root, ["diff", "--name-only", implementation, head, "--"]
    ).decode("utf-8").splitlines()
    if changed != list(binding["post_implementation_allowed_changed_paths"]):
        raise ProtocolError("binding commit changes are not exactly config-only")
    config_relative = str(binding["post_implementation_allowed_changed_paths"][0])
    implementation_protocol = _strict_json_object(
        _git_capture(root, ["show", f"{implementation}:{config_relative}"]),
        label="implementation-commit protocol",
    )
    binding_protocol = _strict_json_object(
        _git_capture(root, ["show", f"{head}:{config_relative}"]),
        label="binding-commit protocol",
    )
    _validate_i_to_b_protocol_transition(
        implementation_protocol, binding_protocol, binding
    )
    for path_key, hash_key in (
        ("contract_path", "contract_sha256"),
        ("data_role_registry_path", "data_role_registry_sha256"),
        ("decision_log_path", "decision_log_sha256"),
    ):
        blob = _git_capture(root, ["show", f"{active}:{authority[path_key]}"])
        if _sha256_bytes(blob) != authority[hash_key]:
            raise ProtocolError("active-authority blob differs from frozen hash")
        head_blob = _git_capture(root, ["show", f"{head}:{authority[path_key]}"])
        if _sha256_bytes(head_blob) != authority[hash_key]:
            raise ProtocolError("current-HEAD authority blob differs from frozen hash")
    implementation_blobs: dict[str, bytes] = {}
    for path_key, hash_key in (
        ("qualifier_path", "qualifier_blob_sha256"),
        ("test_path", "test_blob_sha256"),
    ):
        blob = _git_capture(root, ["show", f"{implementation}:{binding[path_key]}"])
        if _sha256_bytes(blob) != binding[hash_key]:
            raise ProtocolError("implementation blob differs from frozen hash")
        implementation_blobs[path_key] = blob
    running = Path(__file__) if running_script_path is None else running_script_path
    running_bytes, _ = _read_verified_snapshot(
        running,
        label="running GSE114002 v2 auditor",
        expected_sha256=str(binding["qualifier_blob_sha256"]),
    )
    if running_bytes != implementation_blobs["qualifier_path"]:
        raise ProtocolError("running auditor differs from implementation commit")
    return {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "verified": True,
        "implementation_commit": implementation,
        "binding_commit": head,
        "clean_worktree": True,
        "implementation_direct_child_of_staging_parent": True,
        "implementation_changed_paths_exact": True,
        "config_only_direct_child": True,
        "active_authority_blobs_match": True,
        "head_authority_blobs_match": True,
        "implementation_blobs_match": True,
        "running_script_matches_bound_blob": True,
    }


def _parse_bool(token: str) -> bool | None:
    normalized = token.strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def hamming_distance(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(first != second for first, second in zip(left, right))


def reconstruct_paper_rl(
    normalized_fractions: Sequence[float],
    *,
    weights: Sequence[float] = FRACTION_WEIGHTS,
    denominator: float | None = None,
) -> float:
    """Apply the only accepted paper-endpoint formula.

    ``r0..r13`` are already normalized.  A caller cannot substitute the legacy
    integer weights or divide the weighted sum by ``r_total`` a second time.
    """
    if tuple(float(value) for value in weights) != FRACTION_WEIGHTS:
        raise BoundaryViolation("fraction weights differ from the frozen paper weights")
    if denominator is not None:
        raise BoundaryViolation("normalized fractions must not be divided a second time")
    if len(normalized_fractions) != len(FRACTION_WEIGHTS):
        raise BoundaryViolation("normalized fractions and weights are not aligned")
    values = tuple(float(value) for value in normalized_fractions)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise BoundaryViolation("normalized fractions must be finite and nonnegative")
    return math.fsum(value * weight for value, weight in zip(values, FRACTION_WEIGHTS))


def reconstruct_paper_rl_from_raw_counts(*_args: Any, **_kwargs: Any) -> float:
    raise BoundaryViolation(
        "naive raw-row endpoint is prohibited; raw counts require the paper "
        "two-stage global normalization"
    )


def derive_standard_error_from_fraction_counts(*_args: Any, **_kwargs: Any) -> float:
    raise BoundaryViolation("fraction counts cannot provide a technical or biological SE")


def infer_standard_error_from_p_or_fdr(*_args: Any, **_kwargs: Any) -> float:
    raise BoundaryViolation("p/FDR back-calculation of standard error is prohibited")


def _provisional_source_key(
    mother: str, library: str, assay_id: str, context_id: str, endpoint_id: str
) -> str:
    digest = hashlib.sha256(SOURCE_KEY_DOMAIN)
    for part in (mother, library, assay_id, context_id, endpoint_id):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _parse_finite_nonnegative(token: str, *, label: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise InputIntegrityError(f"{label} is not numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise InputIntegrityError(f"{label} is not finite and nonnegative")
    return value


def _header_sha256(header: Sequence[str]) -> str:
    return _sha256_bytes(_compact_json_bytes(list(header)))


def _compute_raw_fraction_global_totals(
    compressed: bytes,
) -> tuple[tuple[float, ...], int, int]:
    """First pass for paper-faithful global fraction normalization.

    The returned totals remain internal.  Neither the per-fraction totals nor
    any row-level values are eligible for aggregate output.
    """
    try:
        gzip_stream = gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb")
        text_stream = io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="")
        reader = csv.reader(text_stream, strict=True)
        header = next(reader)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise InputIntegrityError("source is not a nonempty UTF-8 gzip CSV") from exc
    if tuple(header) != EXPECTED_HEADER:
        text_stream.close()
        raise InputIntegrityError("source exact header mismatch")
    index = {name: position for position, name in enumerate(EXPECTED_HEADER)}
    totals = [0.0] * len(RAW_FRACTION_COLUMNS)
    record_count = 0
    invalid_count = 0
    try:
        for record in reader:
            record_count += 1
            if len(record) != len(EXPECTED_HEADER):
                invalid_count += 1
                continue
            try:
                raw_counts = tuple(
                    _parse_finite_nonnegative(
                        record[index[column]], label="raw fraction count"
                    )
                    for column in RAW_FRACTION_COLUMNS
                )
            except InputIntegrityError:
                invalid_count += 1
                continue
            for position, count in enumerate(raw_counts):
                totals[position] += count
    except (UnicodeDecodeError, csv.Error, OSError, EOFError) as exc:
        raise InputIntegrityError("source CSV first-pass stream is malformed") from exc
    finally:
        text_stream.close()
    return tuple(totals), record_count, invalid_count


def _analyze_verified_source(
    compressed: bytes, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    raw_fraction_global_totals, first_pass_record_count, first_pass_invalid_count = (
        _compute_raw_fraction_global_totals(compressed)
    )
    two_stage_global_totals_valid = (
        first_pass_invalid_count == 0
        and all(
            math.isfinite(total) and total > 0
            for total in raw_fraction_global_totals
        )
    )
    try:
        gzip_stream = gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb")
        text_stream = io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="")
        reader = csv.reader(text_stream, strict=True)
        header = next(reader)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise InputIntegrityError("source is not a nonempty UTF-8 gzip CSV") from exc
    if tuple(header) != EXPECTED_HEADER:
        raise InputIntegrityError("source exact header mismatch")

    input_contract = protocol["input_contract"]
    endpoint_contract = protocol["endpoint_reconciliation_contract"]
    pool_contract = protocol["pool_geometry_contract"]
    index = {name: position for position, name in enumerate(EXPECTED_HEADER)}
    blockers = set(str(value) for value in protocol["unresolved_blockers"])
    pools: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "identity_rows": 0,
            "candidates": {},
            "distance_5_utr_hashes": set(),
            "rule_record_count": 0,
            "library": "",
        }
    )
    raw_id_counts: Counter[str] = Counter()
    included_raw_id_counts: Counter[str] = Counter()
    first_index_counts: Counter[str] = Counter()

    source_row_count = 0
    wrong_width_count = 0
    endpoint_malformed_count = 0
    included_count = 0
    malformed_included_count = 0
    out_of_scope_count = 0
    raw_id_missing_count = 0
    included_raw_id_missing_count = 0
    included_valid_designed_false_distance_5_row_count = 0

    normalized_sum_match_count = 0
    endpoint_match_count = 0
    sum_normalized_equivalent_match_count = 0
    raw_total_match_count = 0
    two_stage_r_total_match_count = 0
    two_stage_stored_fraction_vector_match_count = 0
    max_normalized_sum_residual = 0.0
    max_endpoint_residual = 0.0
    max_sum_normalized_equivalent_residual = 0.0
    max_raw_total_residual = 0.0
    max_two_stage_r_total_residual = 0.0
    max_two_stage_stored_fraction_vector_residual = 0.0

    try:
        for row in reader:
            source_row_count += 1
            if len(row) != len(EXPECTED_HEADER):
                wrong_width_count += 1
                endpoint_malformed_count += 1
                continue

            first_index_counts[row[0]] += 1
            raw_id = row[index["id"]]
            if raw_id:
                raw_id_counts[raw_id] += 1
            else:
                raw_id_missing_count += 1

            try:
                normalized = tuple(
                    _parse_finite_nonnegative(
                        row[index[column]], label="normalized fraction"
                    )
                    for column in NORMALIZED_FRACTION_COLUMNS
                )
                raw_counts = tuple(
                    _parse_finite_nonnegative(row[index[column]], label="raw count")
                    for column in RAW_FRACTION_COLUMNS
                )
                published_rl = _parse_finite_nonnegative(
                    row[index["rl"]], label="paper endpoint"
                )
                raw_total = _parse_finite_nonnegative(
                    row[index["total"]], label="raw total"
                )
                stored_r_total = _parse_finite_nonnegative(
                    row[index["r_total"]], label="paper two-stage row total"
                )
            except InputIntegrityError:
                endpoint_malformed_count += 1
            else:
                normalized_sum = math.fsum(normalized)
                direct = reconstruct_paper_rl(normalized)
                sum_residual = abs(normalized_sum - 1.0)
                endpoint_residual = abs(published_rl - direct)
                equivalent_residual = (
                    abs(published_rl - direct / normalized_sum)
                    if normalized_sum > 0
                    else math.inf
                )
                raw_total_residual = abs(raw_total - math.fsum(raw_counts))
                if two_stage_global_totals_valid:
                    globally_normalized = tuple(
                        count / global_total
                        for count, global_total in zip(
                            raw_counts, raw_fraction_global_totals
                        )
                    )
                    reconstructed_r_total = math.fsum(globally_normalized)
                    if reconstructed_r_total > 0:
                        two_stage_r_total_residual = abs(
                            reconstructed_r_total - stored_r_total
                        )
                        reconstructed_stored_fractions = tuple(
                            value / reconstructed_r_total
                            for value in globally_normalized
                        )
                        two_stage_vector_residual = max(
                            abs(observed - reconstructed)
                            for observed, reconstructed in zip(
                                normalized, reconstructed_stored_fractions
                            )
                        )
                        max_two_stage_r_total_residual = max(
                            max_two_stage_r_total_residual,
                            two_stage_r_total_residual,
                        )
                        max_two_stage_stored_fraction_vector_residual = max(
                            max_two_stage_stored_fraction_vector_residual,
                            two_stage_vector_residual,
                        )
                        two_stage_r_total_match_count += (
                            two_stage_r_total_residual
                            <= float(
                                endpoint_contract[
                                    "two_stage_r_total_absolute_tolerance"
                                ]
                            )
                        )
                        two_stage_stored_fraction_vector_match_count += (
                            two_stage_vector_residual
                            <= float(
                                endpoint_contract[
                                    "two_stage_stored_fraction_vector_absolute_tolerance"
                                ]
                            )
                        )
                max_normalized_sum_residual = max(
                    max_normalized_sum_residual, sum_residual
                )
                max_endpoint_residual = max(max_endpoint_residual, endpoint_residual)
                max_sum_normalized_equivalent_residual = max(
                    max_sum_normalized_equivalent_residual, equivalent_residual
                )
                max_raw_total_residual = max(
                    max_raw_total_residual, raw_total_residual
                )
                normalized_sum_match_count += sum_residual <= float(
                    endpoint_contract["normalized_fraction_sum_tolerance"]
                )
                endpoint_match_count += endpoint_residual <= float(
                    endpoint_contract["paper_endpoint_absolute_tolerance"]
                )
                sum_normalized_equivalent_match_count += equivalent_residual <= float(
                    endpoint_contract["sum_normalized_equivalent_absolute_tolerance"]
                )
                raw_total_match_count += raw_total_residual <= float(
                    endpoint_contract["raw_total_absolute_tolerance"]
                )

            library = row[index["library"]]
            if library not in INCLUDED_LIBRARIES:
                continue
            included_count += 1
            if raw_id:
                included_raw_id_counts[raw_id] += 1
            else:
                included_raw_id_missing_count += 1
            utr = row[index["utr"]].strip().upper()
            mother = row[index["mother"]].strip().upper()
            designed = _parse_bool(row[index["designed"]])
            if (
                DNA_RE.fullmatch(utr) is None
                or DNA_RE.fullmatch(mother) is None
                or designed is None
            ):
                malformed_included_count += 1
                continue
            source_key = _provisional_source_key(
                mother,
                library,
                str(pool_contract["fixed_assay_id"]),
                str(pool_contract["fixed_context_id"]),
                str(pool_contract["fixed_endpoint_id"]),
            )
            pool = pools[source_key]
            pool["library"] = library
            distance = hamming_distance(utr, mother)
            if designed is True and utr == mother:
                pool["identity_rows"] += 1
                pool["rule_record_count"] += 1
            elif designed is False and distance is not None and 1 <= distance <= 3:
                candidate_hash = _sha256_bytes(utr.encode("ascii"))
                pool["candidates"][candidate_hash] = distance
                pool["rule_record_count"] += 1
            else:
                out_of_scope_count += 1
                if designed is False and distance == 5:
                    included_valid_designed_false_distance_5_row_count += 1
                    pool["distance_5_utr_hashes"].add(
                        _sha256_bytes(utr.encode("ascii"))
                    )
    except (UnicodeDecodeError, csv.Error, OSError, EOFError) as exc:
        raise InputIntegrityError("source CSV stream is malformed") from exc
    finally:
        try:
            text_stream.close()
        except Exception:
            pass

    raw_id_collision_value_count = sum(
        count > 1 for count in raw_id_counts.values()
    )
    raw_id_collision_record_count = sum(
        count for count in raw_id_counts.values() if count > 1
    )
    raw_id_maximum_multiplicity = max(raw_id_counts.values(), default=0)
    included_raw_id_collision_value_count = sum(
        count > 1 for count in included_raw_id_counts.values()
    )
    included_raw_id_collision_record_count = sum(
        count for count in included_raw_id_counts.values() if count > 1
    )
    included_raw_id_maximum_multiplicity = max(
        included_raw_id_counts.values(), default=0
    )
    first_index_distinct_count = len(first_index_counts)
    first_index_duplicate_excess_count = sum(first_index_counts.values()) - len(
        first_index_counts
    )
    eligible = {
        key: value
        for key, value in pools.items()
        if value["identity_rows"] == 1
        and len(value["candidates"])
        >= int(pool_contract["minimum_distinct_candidate_count_per_pool"])
    }
    eligible_candidate_count = sum(
        len(value["candidates"]) for value in eligible.values()
    )
    distance_counts = Counter(
        distance
        for value in eligible.values()
        for distance in value["candidates"].values()
    )
    distance_counts_closed = {
        str(distance): distance_counts[distance] for distance in (1, 2, 3)
    }
    pool_with_distance_3_count = sum(
        3 in value["candidates"].values() for value in eligible.values()
    )
    pool_with_distance_counts = {
        str(distance): sum(
            distance in value["candidates"].values() for value in eligible.values()
        )
        for distance in (1, 2, 3)
    }
    eligible_distance_5_utr_hash_count = sum(
        len(value["distance_5_utr_hashes"]) for value in eligible.values()
    )
    eligible_pool_with_distance_5_count = sum(
        bool(value["distance_5_utr_hashes"]) for value in eligible.values()
    )
    eligible_rule_record_count = sum(
        value["rule_record_count"] for value in eligible.values()
    )
    eligible_identity_record_count = sum(
        value["identity_rows"] for value in eligible.values()
    )
    noneligible_rule_record_count = sum(
        value["rule_record_count"]
        for key, value in pools.items()
        if key not in eligible
    )
    pool_with_bad_identity_count = sum(
        value["identity_rows"] != 1 for value in pools.values()
    )
    pool_with_too_few_candidates_count = sum(
        len(value["candidates"])
        < int(pool_contract["minimum_distinct_candidate_count_per_pool"])
        for value in pools.values()
    )

    row_count_match = source_row_count == int(input_contract["expected_source_row_count"])
    endpoint_match = (
        endpoint_malformed_count == 0
        and endpoint_match_count
        == int(endpoint_contract["expected_paper_endpoint_match_count"])
        and max_endpoint_residual
        <= float(endpoint_contract["paper_endpoint_max_abs_residual_cap"])
        and math.isclose(
            max_endpoint_residual,
            float(endpoint_contract["expected_paper_endpoint_max_abs_residual"]),
            rel_tol=0.0,
            abs_tol=5e-14,
        )
    )
    normalized_sum_match = (
        endpoint_malformed_count == 0
        and normalized_sum_match_count
        == int(endpoint_contract["expected_normalized_fraction_sum_match_count"])
        and math.isclose(
            max_normalized_sum_residual,
            float(
                endpoint_contract[
                    "expected_normalized_fraction_sum_max_abs_residual"
                ]
            ),
            rel_tol=0.0,
            abs_tol=5e-14,
        )
    )
    sum_normalized_equivalent_match = (
        endpoint_malformed_count == 0
        and sum_normalized_equivalent_match_count
        == int(endpoint_contract["expected_sum_normalized_equivalent_match_count"])
        and max_sum_normalized_equivalent_residual
        <= float(endpoint_contract["sum_normalized_equivalent_max_abs_residual_cap"])
        and math.isclose(
            max_sum_normalized_equivalent_residual,
            float(
                endpoint_contract[
                    "expected_sum_normalized_equivalent_max_abs_residual"
                ]
            ),
            rel_tol=0.0,
            abs_tol=5e-14,
        )
    )
    raw_total_match = (
        endpoint_malformed_count == 0
        and raw_total_match_count == int(endpoint_contract["expected_raw_total_match_count"])
        and math.isclose(
            max_raw_total_residual,
            float(endpoint_contract["expected_raw_total_max_abs_residual"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    two_stage_normalization_match = (
        first_pass_record_count == source_row_count
        and first_pass_invalid_count == 0
        and two_stage_global_totals_valid
        and endpoint_malformed_count == 0
        and two_stage_r_total_match_count
        == int(endpoint_contract["expected_two_stage_r_total_match_count"])
        and max_two_stage_r_total_residual
        <= float(endpoint_contract["two_stage_r_total_max_abs_residual_cap"])
        and two_stage_stored_fraction_vector_match_count
        == int(
            endpoint_contract[
                "expected_two_stage_stored_fraction_vector_match_count"
            ]
        )
        and max_two_stage_stored_fraction_vector_residual
        <= float(
            endpoint_contract[
                "two_stage_stored_fraction_vector_max_abs_residual_cap"
            ]
        )
    )
    geometry_count_match = (
        included_count == int(pool_contract["expected_included_library_record_count"])
        and malformed_included_count
        == int(pool_contract["expected_malformed_included_record_count"])
        and out_of_scope_count
        == int(pool_contract["expected_valid_out_of_rule_included_record_count"])
        and len(pools) == int(pool_contract["expected_provisional_source_pool_count"])
        and noneligible_rule_record_count
        == int(
            pool_contract["expected_valid_rule_record_count_in_noneligible_pools"]
        )
        and eligible_rule_record_count
        == int(pool_contract["expected_eligible_rule_record_count"])
        and eligible_identity_record_count
        == int(pool_contract["expected_eligible_identity_record_count"])
        and included_count
        == (
            malformed_included_count
            + out_of_scope_count
            + noneligible_rule_record_count
            + eligible_rule_record_count
        )
        and wrong_width_count == 0
        and len(raw_id_counts)
        == int(
            pool_contract["expected_global_raw_id_audit"][
                "nonblank_distinct_token_count"
            ]
        )
        and raw_id_missing_count
        == int(pool_contract["expected_global_raw_id_audit"]["blank_record_count"])
        and raw_id_collision_value_count
        == int(
            pool_contract["expected_global_raw_id_audit"][
                "duplicated_nonblank_token_count"
            ]
        )
        and raw_id_collision_record_count
        == int(
            pool_contract["expected_global_raw_id_audit"][
                "record_count_in_duplicated_nonblank_tokens"
            ]
        )
        and raw_id_maximum_multiplicity
        == int(
            pool_contract["expected_global_raw_id_audit"][
                "maximum_nonblank_token_multiplicity"
            ]
        )
        and included_raw_id_missing_count
        == int(
            pool_contract["expected_included_scope_raw_id_audit"][
                "blank_record_count"
            ]
        )
        and len(included_raw_id_counts)
        == int(
            pool_contract["expected_included_scope_raw_id_audit"][
                "nonblank_distinct_token_count"
            ]
        )
        and included_raw_id_collision_value_count
        == int(
            pool_contract["expected_included_scope_raw_id_audit"][
                "duplicated_nonblank_token_count"
            ]
        )
        and included_raw_id_collision_record_count
        == int(
            pool_contract["expected_included_scope_raw_id_audit"][
                "record_count_in_duplicated_nonblank_tokens"
            ]
        )
        and included_raw_id_maximum_multiplicity
        == int(
            pool_contract["expected_included_scope_raw_id_audit"][
                "maximum_nonblank_token_multiplicity"
            ]
        )
        and first_index_distinct_count
        == int(
            pool_contract["expected_first_unnamed_index_audit"][
                "distinct_token_count"
            ]
        )
        and first_index_duplicate_excess_count
        == int(
            pool_contract["expected_first_unnamed_index_audit"][
                "duplicate_excess_record_count"
            ]
        )
        and len(eligible)
        == int(pool_contract["expected_eligible_provisional_pool_count"])
        and eligible_candidate_count
        == int(pool_contract["expected_eligible_distinct_candidate_count"])
    )
    distance_match = (
        distance_counts_closed
        == dict(pool_contract["expected_hamming_distance_candidate_counts"])
        and pool_with_distance_counts
        == dict(pool_contract["expected_pool_with_hamming_distance_candidate_counts"])
        and pool_with_distance_3_count
        == int(pool_contract["expected_pool_with_hamming_distance_3_candidate_count"])
    )
    k5_scope_match = (
        included_valid_designed_false_distance_5_row_count
        == int(
            pool_contract[
                "expected_included_valid_designed_false_hamming_distance_5_row_count"
            ]
        )
        and eligible_distance_5_utr_hash_count
        == int(
            pool_contract[
                "expected_eligible_pool_distinct_hamming_distance_5_utr_hash_count"
            ]
        )
        and eligible_pool_with_distance_5_count
        == int(
            pool_contract[
                "expected_eligible_pool_with_hamming_distance_5_count"
            ]
        )
    )

    if not row_count_match:
        blockers.add("SOURCE_ROW_COUNT_RECONCILIATION_MISMATCH")
    if not endpoint_match:
        blockers.add("ENDPOINT_RECONCILIATION_MISMATCH")
    if not normalized_sum_match:
        blockers.add("NORMALIZED_FRACTION_SUM_RECONCILIATION_MISMATCH")
    if not sum_normalized_equivalent_match:
        blockers.add("SUM_NORMALIZED_EQUIVALENT_RECONCILIATION_MISMATCH")
    if not raw_total_match:
        blockers.add("RAW_TOTAL_RECONCILIATION_MISMATCH")
    if not two_stage_normalization_match:
        blockers.add("TWO_STAGE_GLOBAL_NORMALIZATION_RECONCILIATION_MISMATCH")
    if malformed_included_count or wrong_width_count:
        blockers.add("MALFORMED_INCLUDED_ROWS_PRESENT")
    if not geometry_count_match:
        blockers.add("PROVISIONAL_POOL_GEOMETRY_RECONCILIATION_MISMATCH")
    if not distance_match:
        blockers.add("HAMMING_DISTANCE_DISTRIBUTION_RECONCILIATION_MISMATCH")
    if not k5_scope_match:
        blockers.add("K5_CLAIM_BOUNDARY_SCOPE_RECONCILIATION_MISMATCH")

    input_audit = {
        "dataset_id": DATASET_ID,
        "status": "PASS_EXACT_PUBLIC_ASSET_AND_HEADER" if row_count_match else "FAIL_ROW_COUNT",
        "source_basename": SOURCE_BASENAME,
        "source_asset_sha256": str(input_contract["compressed_sha256"]),
        "source_asset_bytes": int(input_contract["compressed_bytes"]),
        "exact_header_column_count": len(EXPECTED_HEADER),
        "exact_header_sha256": _header_sha256(EXPECTED_HEADER),
        "source_record_count": source_row_count,
        "expected_source_record_count": int(input_contract["expected_source_row_count"]),
        "source_record_count_matches": row_count_match,
        "same_descriptor_verified_snapshot": True,
        "restricted_or_sealed_payload_opened": False,
        "gse246381_payload_opened": False,
    }
    endpoint_audit = {
        "dataset_id": DATASET_ID,
        "status": (
            "PASS_EXACT_PAPER_ENDPOINT_RECONCILIATION"
            if endpoint_match
            and normalized_sum_match
            and sum_normalized_equivalent_match
            and raw_total_match
            and two_stage_normalization_match
            else "FAIL_ENDPOINT_OR_TOTAL_RECONCILIATION"
        ),
        "source_record_count": source_row_count,
        "endpoint_parse_failure_count": endpoint_malformed_count,
        "fraction_weights": list(FRACTION_WEIGHTS),
        "paper_endpoint_formula": "SUM_RI_TIMES_FROZEN_WEIGHT_NO_DENOMINATOR",
        "paper_endpoint_match_count": endpoint_match_count,
        "paper_endpoint_max_abs_residual": max_endpoint_residual,
        "paper_endpoint_absolute_tolerance": float(
            endpoint_contract["paper_endpoint_absolute_tolerance"]
        ),
        "normalized_fraction_sum_match_count": normalized_sum_match_count,
        "normalized_fraction_sum_max_abs_residual": max_normalized_sum_residual,
        "normalized_fraction_sum_tolerance": float(
            endpoint_contract["normalized_fraction_sum_tolerance"]
        ),
        "sum_normalized_equivalent_match_count": (
            sum_normalized_equivalent_match_count
        ),
        "sum_normalized_equivalent_max_abs_residual": (
            max_sum_normalized_equivalent_residual
        ),
        "sum_normalized_equivalent_absolute_tolerance": float(
            endpoint_contract["sum_normalized_equivalent_absolute_tolerance"]
        ),
        "raw_total_match_count": raw_total_match_count,
        "raw_total_max_abs_residual": max_raw_total_residual,
        "raw_total_absolute_tolerance": float(
            endpoint_contract["raw_total_absolute_tolerance"]
        ),
        "raw_fraction_count_endpoint_role": (
            "RAW_RECONSTRUCTS_ONLY_VIA_PAPER_TWO_STAGE_GLOBAL_NORMALIZATION"
        ),
        "two_stage_r_total_match_count": two_stage_r_total_match_count,
        "two_stage_r_total_max_abs_residual": max_two_stage_r_total_residual,
        "two_stage_r_total_absolute_tolerance": float(
            endpoint_contract["two_stage_r_total_absolute_tolerance"]
        ),
        "two_stage_stored_fraction_vector_match_count": (
            two_stage_stored_fraction_vector_match_count
        ),
        "two_stage_stored_fraction_vector_max_abs_residual": (
            max_two_stage_stored_fraction_vector_residual
        ),
        "two_stage_stored_fraction_vector_absolute_tolerance": float(
            endpoint_contract[
                "two_stage_stored_fraction_vector_absolute_tolerance"
            ]
        ),
        "global_fraction_totals_emitted": False,
        "naive_raw_row_endpoint_reconstruction_allowed": False,
        "stored_normalized_fractions_second_division_by_r_total": False,
        "technical_standard_error_derived": False,
        "biological_replicate_status": "ABSENT_BY_DESIGN",
        "paper_standard_error": None,
        "paper_standard_error_status": "ABSENT_NOT_DERIVABLE",
        "biological_standard_error_derived": False,
        "p_or_fdr_used_to_back_calculate_standard_error": False,
    }
    geometry_audit = {
        "dataset_id": DATASET_ID,
        "status": (
            "PASS_PROVISIONAL_GEOMETRY_MATCH_NOT_AUTHORITY"
            if geometry_count_match
            and distance_match
            and k5_scope_match
            and malformed_included_count == 0
            else "FAIL_PROVISIONAL_GEOMETRY_OR_MALFORMED_ROWS"
        ),
        "included_library_record_count": included_count,
        "provisional_source_pool_count": len(pools),
        "eligible_provisional_pool_count": len(eligible),
        "eligible_provisional_distinct_candidate_count": eligible_candidate_count,
        "eligible_rule_record_count": eligible_rule_record_count,
        "eligible_identity_record_count": eligible_identity_record_count,
        "valid_rule_record_count_in_noneligible_pools": noneligible_rule_record_count,
        "pool_with_nonunique_or_missing_identity_count": pool_with_bad_identity_count,
        "pool_with_fewer_than_three_distinct_candidate_count": (
            pool_with_too_few_candidates_count
        ),
        "hamming_distance_candidate_counts": distance_counts_closed,
        "pool_with_hamming_distance_candidate_counts": pool_with_distance_counts,
        "pool_with_hamming_distance_3_candidate_count": pool_with_distance_3_count,
        "included_valid_designed_false_hamming_distance_5_row_count": (
            included_valid_designed_false_distance_5_row_count
        ),
        "eligible_pool_distinct_hamming_distance_5_utr_hash_count": (
            eligible_distance_5_utr_hash_count
        ),
        "eligible_pool_with_hamming_distance_5_count": (
            eligible_pool_with_distance_5_count
        ),
        "k5_role": (
            "CLAIM_BOUNDARY_ONLY_AFTER_K1_TO_K3_ELIGIBILITY_NOT_ELIGIBILITY_"
            "QUALIFICATION_OR_TRUE_A2_EVIDENCE"
        ),
        "raw_identifier_collision_distinct_token_count": raw_id_collision_value_count,
        "raw_identifier_collision_record_count": raw_id_collision_record_count,
        "raw_identifier_nonblank_distinct_token_count": len(raw_id_counts),
        "raw_identifier_maximum_nonblank_token_multiplicity": (
            raw_id_maximum_multiplicity
        ),
        "raw_identifier_missing_record_count": raw_id_missing_count,
        "included_scope_raw_identifier_collision_distinct_token_count": (
            included_raw_id_collision_value_count
        ),
        "included_scope_raw_identifier_collision_record_count": (
            included_raw_id_collision_record_count
        ),
        "included_scope_raw_identifier_nonblank_distinct_token_count": (
            len(included_raw_id_counts)
        ),
        "included_scope_raw_identifier_maximum_nonblank_token_multiplicity": (
            included_raw_id_maximum_multiplicity
        ),
        "included_scope_raw_identifier_missing_record_count": (
            included_raw_id_missing_count
        ),
        "first_unnamed_index_distinct_token_count": first_index_distinct_count,
        "first_unnamed_index_duplicate_excess_record_count": (
            first_index_duplicate_excess_count
        ),
        "raw_identifier_collision_role": (
            "AGGREGATE_COLLISION_AUDIT_ONLY_NOT_AUTOMATIC_BLOCKER"
        ),
        "canonical_identity_status": "FUTURE_COMPOSITE_OR_HASH_IDENTITY_PENDING_AUTHORITY",
        "valid_out_of_rule_included_record_count": out_of_scope_count,
        "valid_out_of_rule_included_record_status": "OUT_OF_SCOPE_DISPOSITION_PENDING",
        "malformed_included_record_count": malformed_included_count + wrong_width_count,
        "included_record_conservation_equation_holds": (
            included_count
            == malformed_included_count
            + out_of_scope_count
            + noneligible_rule_record_count
            + eligible_rule_record_count
        ),
        "field_and_biological_source_authority_status": "UNKNOWN_NOT_ASSERTED",
        "expected_counts_are_field_or_source_authority": False,
        "sequence_identifier_or_record_arrays_emitted": False,
        "true_a2_dense_pool_count": 0,
    }
    return input_audit, endpoint_audit, geometry_audit, sorted(blockers)


def _assert_aggregate_safe(value: Any, *, parent_key: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_OUTPUT_FIELDS:
                raise PublicationError(f"forbidden raw field in aggregate output: {key}")
            if isinstance(child, list) and any(
                token in normalized for token in SENSITIVE_ARRAY_TOKENS
            ):
                raise PublicationError(f"sensitive array is forbidden: {key}")
            _assert_aggregate_safe(child, parent_key=normalized)
    elif isinstance(value, list):
        if any(token in parent_key for token in SENSITIVE_ARRAY_TOKENS):
            raise PublicationError(f"sensitive array is forbidden: {parent_key}")
        for child in value:
            _assert_aggregate_safe(child, parent_key=parent_key)
    elif isinstance(value, str) and SENSITIVE_NUCLEOTIDE_RUN_RE.search(value):
        raise PublicationError(
            "aggregate output contains a prohibited 26-nt-or-longer nucleotide string"
        )


def _validate_closed_output_payloads(payloads: Mapping[str, Any]) -> None:
    if set(payloads) != set(JSON_PAYLOAD_FILENAMES):
        raise PublicationError("JSON payload member set differs from closed schema")
    for name in JSON_PAYLOAD_FILENAMES:
        if not isinstance(payloads[name], Mapping):
            raise PublicationError(f"{name} root is not an object")
    expected_keys = {
        "INPUT_INTEGRITY_AUDIT.json": {
            "dataset_id", "status", "source_basename", "source_asset_sha256",
            "source_asset_bytes", "exact_header_column_count", "exact_header_sha256",
            "source_record_count", "expected_source_record_count",
            "source_record_count_matches", "same_descriptor_verified_snapshot",
            "restricted_or_sealed_payload_opened", "gse246381_payload_opened",
        },
        "ENDPOINT_RECONCILIATION_AUDIT.json": {
            "dataset_id", "status", "source_record_count", "endpoint_parse_failure_count",
            "fraction_weights", "paper_endpoint_formula", "paper_endpoint_match_count",
            "paper_endpoint_max_abs_residual", "paper_endpoint_absolute_tolerance",
            "normalized_fraction_sum_match_count",
            "normalized_fraction_sum_max_abs_residual",
            "normalized_fraction_sum_tolerance",
            "sum_normalized_equivalent_match_count",
            "sum_normalized_equivalent_max_abs_residual",
            "sum_normalized_equivalent_absolute_tolerance", "raw_total_match_count",
            "raw_total_max_abs_residual", "raw_total_absolute_tolerance",
            "raw_fraction_count_endpoint_role", "two_stage_r_total_match_count",
            "two_stage_r_total_max_abs_residual",
            "two_stage_r_total_absolute_tolerance",
            "two_stage_stored_fraction_vector_match_count",
            "two_stage_stored_fraction_vector_max_abs_residual",
            "two_stage_stored_fraction_vector_absolute_tolerance",
            "global_fraction_totals_emitted",
            "naive_raw_row_endpoint_reconstruction_allowed",
            "stored_normalized_fractions_second_division_by_r_total",
            "technical_standard_error_derived", "biological_replicate_status",
            "paper_standard_error", "paper_standard_error_status",
            "biological_standard_error_derived",
            "p_or_fdr_used_to_back_calculate_standard_error",
        },
        "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": {
            "dataset_id", "status", "included_library_record_count",
            "provisional_source_pool_count", "eligible_provisional_pool_count",
            "eligible_provisional_distinct_candidate_count",
            "eligible_rule_record_count", "eligible_identity_record_count",
            "valid_rule_record_count_in_noneligible_pools",
            "pool_with_nonunique_or_missing_identity_count",
            "pool_with_fewer_than_three_distinct_candidate_count",
            "hamming_distance_candidate_counts",
            "pool_with_hamming_distance_candidate_counts",
            "pool_with_hamming_distance_3_candidate_count",
            "included_valid_designed_false_hamming_distance_5_row_count",
            "eligible_pool_distinct_hamming_distance_5_utr_hash_count",
            "eligible_pool_with_hamming_distance_5_count", "k5_role",
            "raw_identifier_collision_distinct_token_count",
            "raw_identifier_collision_record_count",
            "raw_identifier_nonblank_distinct_token_count",
            "raw_identifier_maximum_nonblank_token_multiplicity",
            "raw_identifier_missing_record_count", "raw_identifier_collision_role",
            "included_scope_raw_identifier_collision_distinct_token_count",
            "included_scope_raw_identifier_collision_record_count",
            "included_scope_raw_identifier_nonblank_distinct_token_count",
            "included_scope_raw_identifier_maximum_nonblank_token_multiplicity",
            "included_scope_raw_identifier_missing_record_count",
            "first_unnamed_index_distinct_token_count",
            "first_unnamed_index_duplicate_excess_record_count",
            "canonical_identity_status", "valid_out_of_rule_included_record_count",
            "valid_out_of_rule_included_record_status",
            "malformed_included_record_count",
            "included_record_conservation_equation_holds",
            "field_and_biological_source_authority_status",
            "expected_counts_are_field_or_source_authority",
            "sequence_identifier_or_record_arrays_emitted", "true_a2_dense_pool_count",
        },
        "QUALIFICATION_REPORT.json": {
            "contract_id", "protocol_id", "dataset_id", "status", "qualified",
            "data_role", "scientific_claim_status", "ordinary_study_contribution",
            "a1_intervention_study_contribution", "true_a2_dense_study_contribution",
            "canonical_record_count", "canonical_materialization_allowed",
            "training_allowed", "model_selection_allowed", "next_phase_authorized",
            "true_a2_claim_established", "aggregate_only", "blockers",
            "protocol_provenance", "source_provenance", "implementation_binding",
        },
    }
    for name, keys in expected_keys.items():
        if set(payloads[name]) != keys:
            raise PublicationError(f"{name} keys differ from closed schema")
    report = payloads["QUALIFICATION_REPORT.json"]
    blockers = report["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not set(blockers).issubset(ALL_BLOCKERS)
    ):
        raise PublicationError("qualification blocker list violates closed enum")
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "qualified": False,
        "data_role": DATA_ROLE,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
    }.items():
        if report.get(key) != expected or isinstance(report.get(key), bool) != isinstance(
            expected, bool
        ):
            raise PublicationError(f"qualification boundary drifted at {key}")
    if report.get("status") not in {MECHANICAL_STATUS, FAILED_MECHANICAL_STATUS}:
        raise PublicationError("qualification status is outside closed enum")
    input_audit = payloads["INPUT_INTEGRITY_AUDIT.json"]
    endpoint_audit = payloads["ENDPOINT_RECONCILIATION_AUDIT.json"]
    geometry_audit = payloads["POOL_GEOMETRY_RECONCILIATION_AUDIT.json"]
    if input_audit.get("dataset_id") != DATASET_ID or input_audit.get(
        "source_basename"
    ) != SOURCE_BASENAME:
        raise PublicationError("input audit identity differs")
    if endpoint_audit.get("fraction_weights") != list(FRACTION_WEIGHTS):
        raise PublicationError("endpoint audit weights differ")
    if endpoint_audit.get("raw_fraction_count_endpoint_role") != (
        "RAW_RECONSTRUCTS_ONLY_VIA_PAPER_TWO_STAGE_GLOBAL_NORMALIZATION"
    ):
        raise PublicationError("raw fraction endpoint role differs")
    for key in (
        "global_fraction_totals_emitted",
        "naive_raw_row_endpoint_reconstruction_allowed",
        "stored_normalized_fractions_second_division_by_r_total",
        "technical_standard_error_derived",
        "biological_standard_error_derived",
        "p_or_fdr_used_to_back_calculate_standard_error",
    ):
        if endpoint_audit.get(key) is not False:
            raise PublicationError(f"endpoint boundary drifted at {key}")
    if (
        endpoint_audit.get("biological_replicate_status") != "ABSENT_BY_DESIGN"
        or endpoint_audit.get("paper_standard_error") is not None
        or endpoint_audit.get("paper_standard_error_status")
        != "ABSENT_NOT_DERIVABLE"
    ):
        raise PublicationError("endpoint uncertainty boundary drifted")
    if geometry_audit.get("sequence_identifier_or_record_arrays_emitted") is not False:
        raise PublicationError("geometry output privacy boundary drifted")
    if geometry_audit.get("true_a2_dense_pool_count") != 0:
        raise PublicationError("geometry output cannot establish true A2")
    for key in (
        "included_valid_designed_false_hamming_distance_5_row_count",
        "eligible_pool_distinct_hamming_distance_5_utr_hash_count",
        "eligible_pool_with_hamming_distance_5_count",
    ):
        value = geometry_audit.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PublicationError(f"geometry K5 aggregate {key} is invalid")
    if geometry_audit.get("k5_role") != (
        "CLAIM_BOUNDARY_ONLY_AFTER_K1_TO_K3_ELIGIBILITY_NOT_ELIGIBILITY_"
        "QUALIFICATION_OR_TRUE_A2_EVIDENCE"
    ):
        raise PublicationError("geometry K5 claim-boundary role drifted")
    for key in (
        "hamming_distance_candidate_counts",
        "pool_with_hamming_distance_candidate_counts",
    ):
        child = geometry_audit.get(key)
        if not isinstance(child, Mapping) or set(child) != {"1", "2", "3"}:
            raise PublicationError(f"geometry aggregate {key} differs from closed schema")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in child.values()
        ):
            raise PublicationError(f"geometry aggregate {key} is invalid")
    protocol_provenance = report.get("protocol_provenance")
    source_provenance = report.get("source_provenance")
    implementation_binding = report.get("implementation_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol_provenance, source_provenance, implementation_binding)
    ):
        raise PublicationError("report provenance or binding root is invalid")
    if set(protocol_provenance) != {
        "basename", "sha256", "bytes", "parser_input_mode", "launch_expected_sha256"
    }:
        raise PublicationError("protocol provenance keys differ")
    if set(source_provenance) != {
        "basename", "sha256", "bytes", "parser_input_mode"
    }:
        raise PublicationError("source provenance keys differ")
    if set(implementation_binding) != {
        "status", "verified", "implementation_commit", "binding_commit",
        "clean_worktree", "implementation_direct_child_of_staging_parent",
        "implementation_changed_paths_exact", "config_only_direct_child",
        "active_authority_blobs_match", "head_authority_blobs_match",
        "implementation_blobs_match", "running_script_matches_bound_blob",
    }:
        raise PublicationError("implementation binding audit keys differ")
    if (
        protocol_provenance.get("basename") != PROTOCOL_BASENAME
        or source_provenance.get("basename") != SOURCE_BASENAME
        or protocol_provenance.get("parser_input_mode")
        != "SAME_DESCRIPTOR_VERIFIED_SNAPSHOT"
        or source_provenance.get("parser_input_mode")
        != "SAME_DESCRIPTOR_VERIFIED_SNAPSHOT"
        or SHA256_RE.fullmatch(str(protocol_provenance.get("sha256"))) is None
        or SHA256_RE.fullmatch(
            str(protocol_provenance.get("launch_expected_sha256"))
        )
        is None
        or SHA256_RE.fullmatch(str(source_provenance.get("sha256"))) is None
    ):
        raise PublicationError("report provenance values differ")
    if implementation_binding.get("status") != "PASS_BOUND_IMPLEMENTATION" or (
        implementation_binding.get("verified") is not True
    ):
        raise PublicationError("report implementation binding is not verified")
    for value in payloads.values():
        _assert_aggregate_safe(value)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o640)
    except OSError as exc:
        raise PublicationError(f"exclusive output creation failed: {path.name}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PublicationError("exclusive output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationError("publication directory cannot be opened safely") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PublicationError("publication fsync target is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    if source.parent != target.parent:
        raise PublicationError("atomic rename requires one parent")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise AtomicNoReplaceUnsupported(errno.ENOSYS)
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        if function is None:
            raise AtomicNoReplaceUnsupported(errno.ENOSYS)
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, target_bytes, 0x00000004)
    else:
        raise AtomicNoReplaceUnsupported(errno.ENOSYS)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PublicationContention("final output already exists")
        if error_number in ATOMIC_NOREPLACE_UNSUPPORTED_ERRNOS:
            raise AtomicNoReplaceUnsupported(error_number)
        raise PublicationError(f"atomic no-replace rename failed with errno {error_number}")


def _final_target_sha256(path: Path) -> str:
    return _sha256_bytes(
        b"ROUTE_A_V3_GSE114002_RECONCILIATION_V2_FINAL_TARGET\n"
        + os.fsencode(_absolute_without_resolving(path))
        + b"\n"
    )


def _publication_marker(
    complete_payloads: Mapping[str, bytes], *, mode: str, output: Path
) -> dict[str, Any]:
    if set(complete_payloads) != set(JSON_PAYLOAD_FILENAMES) | {SHA256SUMS_FILENAME}:
        raise PublicationError("commit-marker member set differs")
    if mode not in {PRIMARY_PUBLICATION_MODE, FALLBACK_PUBLICATION_MODE}:
        raise PublicationError("publication mode is outside the closed enum")
    try:
        scientific_status = json.loads(
            complete_payloads["QUALIFICATION_REPORT.json"].decode("utf-8")
        )["status"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PublicationError("qualification report status cannot bind marker") from exc
    if scientific_status not in {MECHANICAL_STATUS, FAILED_MECHANICAL_STATUS}:
        raise PublicationError("qualification report status is outside marker enum")
    return {
        "schema_version": "1.0.0",
        "record_type": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_COMMIT",
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": OUTPUT_ID,
        "scientific_status": scientific_status,
        "publication_mode": mode,
        "sha256sums_sha256": _sha256_bytes(complete_payloads[SHA256SUMS_FILENAME]),
        "bundle_file_count_excluding_commit_marker": len(complete_payloads),
        "bundle_member_names_excluding_commit_marker": sorted(complete_payloads),
        "final_output_directory_name_sha256": _sha256_bytes(output.name.encode("utf-8")),
        "final_output_target_sha256": _final_target_sha256(output),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }


def _render_complete_payloads(payloads: Mapping[str, Any]) -> dict[str, bytes]:
    _validate_closed_output_payloads(payloads)
    rendered = {name: _pretty_json_bytes(payloads[name]) for name in JSON_PAYLOAD_FILENAMES}
    sums = "".join(
        f"{_sha256_bytes(rendered[name])}  {name}\n" for name in sorted(rendered)
    ).encode("ascii")
    return {**rendered, SHA256SUMS_FILENAME: sums}


def _write_complete_directory(
    directory: Path,
    complete_payloads: Mapping[str, bytes],
    *,
    mode: str,
    final_output: Path,
) -> None:
    for name in sorted(JSON_PAYLOAD_FILENAMES):
        _write_exclusive(directory / name, complete_payloads[name])
    _write_exclusive(
        directory / SHA256SUMS_FILENAME, complete_payloads[SHA256SUMS_FILENAME]
    )
    marker = _publication_marker(complete_payloads, mode=mode, output=final_output)
    _write_exclusive(
        directory / PUBLICATION_COMMIT_FILENAME, _pretty_json_bytes(marker)
    )
    _fsync_directory(directory)


def _validate_publication_commit(
    directory: Path, *, expected_mode: str, final_output: Path | None = None
) -> dict[str, Any]:
    target = directory if final_output is None else final_output
    try:
        names = set(os.listdir(directory))
    except OSError as exc:
        raise PublicationError("publication directory cannot be listed") from exc
    if names != set(EXACT_BUNDLE_MEMBERS):
        raise PublicationError("publication member set is not exact")
    marker_payload, _ = _read_verified_snapshot(
        directory / PUBLICATION_COMMIT_FILENAME,
        label="publication commit marker",
    )
    marker = _strict_json_object(marker_payload, label="publication commit marker")
    marker_keys = {
        "schema_version", "record_type", "contract_id", "protocol_id", "dataset_id",
        "output_id", "scientific_status", "publication_mode", "sha256sums_sha256",
        "bundle_file_count_excluding_commit_marker",
        "bundle_member_names_excluding_commit_marker",
        "final_output_directory_name_sha256", "final_output_target_sha256",
        "committed", "commit_marker_written_last",
        "aggregate_acceptance_requires_exact_marker",
    }
    if set(marker) != marker_keys:
        raise PublicationError("publication marker keys differ")
    expected_scalars = {
        "schema_version": "1.0.0",
        "record_type": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_COMMIT",
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "output_id": OUTPUT_ID,
        "publication_mode": expected_mode,
        "bundle_file_count_excluding_commit_marker": 5,
        "bundle_member_names_excluding_commit_marker": sorted(
            set(JSON_PAYLOAD_FILENAMES) | {SHA256SUMS_FILENAME}
        ),
        "final_output_directory_name_sha256": _sha256_bytes(
            target.name.encode("utf-8")
        ),
        "final_output_target_sha256": _final_target_sha256(target),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }
    for key, expected in expected_scalars.items():
        if marker.get(key) != expected:
            raise PublicationError(f"publication marker differs at {key}")
    sums_payload, _ = _read_verified_snapshot(
        directory / SHA256SUMS_FILENAME,
        label="published SHA256SUMS",
    )
    if _sha256_bytes(sums_payload) != marker.get("sha256sums_sha256"):
        raise PublicationError("publication marker SHA256SUMS binding differs")
    try:
        lines = sums_payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicationError("SHA256SUMS is not ASCII") from exc
    declared: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise PublicationError("SHA256SUMS line is malformed")
        digest, name = line[:64], line[66:]
        if SHA256_RE.fullmatch(digest) is None or Path(name).name != name or name in declared:
            raise PublicationError("SHA256SUMS entry is unsafe")
        declared[name] = digest
    if set(declared) != set(JSON_PAYLOAD_FILENAMES):
        raise PublicationError("SHA256SUMS declaration set differs")
    for name, digest in declared.items():
        payload, _ = _read_verified_snapshot(
            directory / name, label="published aggregate member"
        )
        if _sha256_bytes(payload) != digest:
            raise PublicationError("published aggregate member hash differs")
    report_payload, _ = _read_verified_snapshot(
        directory / "QUALIFICATION_REPORT.json",
        label="published qualification report",
    )
    report = _strict_json_object(report_payload, label="published qualification report")
    if marker.get("scientific_status") != report.get("status") or report.get(
        "status"
    ) not in {MECHANICAL_STATUS, FAILED_MECHANICAL_STATUS}:
        raise PublicationError("publication marker scientific status binding differs")
    return marker


def _committed_result(
    *,
    kind: str,
    output: Path,
    mode: str,
    accepted: bool,
    atomic_no_replace: bool,
    warning_codes: Sequence[str] = (),
) -> dict[str, Any]:
    if kind not in {
        "PUBLISHED",
        "COMMITTED_WITH_POST_COMMIT_WARNING",
        "COMMITTED_NOT_ACCEPTED",
        "ALREADY_COMMITTED_EXACT",
    }:
        raise PublicationError("committed result kind is outside the closed enum")
    return {
        "kind": kind,
        "published": accepted,
        "committed": True,
        "accepted": accepted,
        "output": os.fspath(output),
        "file_count": len(EXACT_BUNDLE_MEMBERS) if accepted else 0,
        "publication_mode": mode,
        "atomic_no_replace": atomic_no_replace,
        "terminal_commit_marker_validated": accepted,
        "requires_manual_adjudication": not accepted,
        "post_commit_warning_codes": sorted(set(warning_codes)),
    }


def _partial_result(
    output: Path, *, mode: str, warning_codes: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        "kind": "PARTIAL_REQUIRES_MANUAL_ADJUDICATION",
        "published": False,
        "committed": False,
        "accepted": False,
        "output": os.fspath(output),
        "file_count": 0,
        "publication_mode": mode,
        "atomic_no_replace": False,
        "terminal_commit_marker_validated": False,
        "requires_manual_adjudication": True,
        "post_commit_warning_codes": sorted(set(warning_codes)),
    }


def _precommit_staging_manual_result(
    output: Path, staging: Path, *, warning_code: str
) -> dict[str, Any]:
    return {
        "kind": "PRECOMMIT_STAGING_REQUIRES_MANUAL_ADJUDICATION",
        "published": False,
        "committed": False,
        "accepted": False,
        "output": os.fspath(output),
        "staging_evidence_path": os.fspath(staging),
        "file_count": 0,
        "publication_mode": "PRECOMMIT_STAGING_ONLY",
        "atomic_no_replace": False,
        "terminal_commit_marker_validated": False,
        "requires_manual_adjudication": True,
        "post_commit_warning_codes": [warning_code],
    }


def _cleanup_owned_precommit_staging(staging: Path, output: Path) -> None:
    """Remove only the exact temporary directory created for this output.

    The marker is unlinked first so an interrupted cleanup cannot leave a
    marker-bearing precommit directory that resembles a second accepted truth.
    Any unexpected entry, symlink, hard link, or identity drift stops cleanup
    and lets the caller return the explicit manual-adjudication state.
    """
    expected_prefix = f".{output.name}.staging-"
    if staging.parent != output.parent or not staging.name.startswith(expected_prefix):
        raise PublicationError("precommit staging path is outside the owned namespace")
    opened = staging.lstat()
    if stat.S_ISLNK(opened.st_mode) or not stat.S_ISDIR(opened.st_mode):
        raise PublicationError("precommit staging identity is not a real directory")
    names = set(os.listdir(staging))
    if not names.issubset(set(EXACT_BUNDLE_MEMBERS)):
        raise PublicationError("precommit staging contains an unexpected entry")
    ordered = [PUBLICATION_COMMIT_FILENAME] + sorted(
        names - {PUBLICATION_COMMIT_FILENAME}
    )
    for name in ordered:
        path = staging / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            if name == PUBLICATION_COMMIT_FILENAME:
                continue
            raise PublicationError("precommit staging entry disappeared during cleanup")
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PublicationError("precommit staging entry is not an owned regular file")
        path.unlink()
    try:
        staging.rmdir()
    except OSError as exc:
        raise PublicationError("precommit staging directory cleanup failed") from exc


def _classify_existing_output(
    output: Path, *, expected_complete_payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    """Classify an occupied final name without changing or overwriting it."""
    if set(expected_complete_payloads) != set(JSON_PAYLOAD_FILENAMES) | {
        SHA256SUMS_FILENAME
    }:
        raise PublicationError("existing-output expected member set differs")
    try:
        opened = output.lstat()
    except FileNotFoundError:
        return _partial_result(
            output,
            mode="UNKNOWN_NOT_ASSERTED",
            warning_codes=["FINAL_NAME_DISAPPEARED_DURING_CONTENTION_CLASSIFICATION"],
        )
    if stat.S_ISLNK(opened.st_mode) or not stat.S_ISDIR(opened.st_mode):
        return _partial_result(
            output,
            mode="UNKNOWN_NOT_ASSERTED",
            warning_codes=["FINAL_NAME_OCCUPIED_BY_NON_DIRECTORY_OR_SYMLINK"],
        )
    marker_path = output / PUBLICATION_COMMIT_FILENAME
    try:
        marker_info = marker_path.lstat()
    except FileNotFoundError:
        return _partial_result(
            output,
            mode=FALLBACK_PUBLICATION_MODE,
            warning_codes=["VISIBLE_FINAL_DIRECTORY_HAS_NO_TERMINAL_MARKER"],
        )
    if stat.S_ISLNK(marker_info.st_mode) or not stat.S_ISREG(marker_info.st_mode):
        return _committed_result(
            kind="COMMITTED_NOT_ACCEPTED",
            output=output,
            mode="UNKNOWN_NOT_ASSERTED",
            accepted=False,
            atomic_no_replace=False,
            warning_codes=["TERMINAL_MARKER_ENTRY_IS_NOT_A_REGULAR_FILE"],
        )
    try:
        marker_payload, _ = _read_verified_snapshot(
            marker_path, label="existing publication commit marker"
        )
        marker = _strict_json_object(
            marker_payload, label="existing publication commit marker"
        )
        mode = str(marker.get("publication_mode"))
        if mode not in {PRIMARY_PUBLICATION_MODE, FALLBACK_PUBLICATION_MODE}:
            raise PublicationError("existing marker publication mode is invalid")
        _validate_publication_commit(output, expected_mode=mode)
    except Exception:
        return _committed_result(
            kind="COMMITTED_NOT_ACCEPTED",
            output=output,
            mode=(
                mode
                if "mode" in locals()
                and mode in {PRIMARY_PUBLICATION_MODE, FALLBACK_PUBLICATION_MODE}
                else "UNKNOWN_NOT_ASSERTED"
            ),
            accepted=False,
            atomic_no_replace=False,
            warning_codes=["OCCUPIED_FINAL_MARKER_OR_BUNDLE_VALIDATION_FAILED"],
        )
    for name, expected_payload in expected_complete_payloads.items():
        try:
            observed_payload, _ = _read_verified_snapshot(
                output / name,
                label="existing publication expected-byte comparison",
            )
        except Exception:
            return _committed_result(
                kind="COMMITTED_NOT_ACCEPTED",
                output=output,
                mode=mode,
                accepted=False,
                atomic_no_replace=(mode == PRIMARY_PUBLICATION_MODE),
                warning_codes=["EXISTING_BUNDLE_EXPECTED_BYTE_COMPARISON_FAILED"],
            )
        if observed_payload != expected_payload:
            return _committed_result(
                kind="COMMITTED_NOT_ACCEPTED",
                output=output,
                mode=mode,
                accepted=False,
                atomic_no_replace=(mode == PRIMARY_PUBLICATION_MODE),
                warning_codes=["EXISTING_BUNDLE_DIFFERS_FROM_CURRENT_EXPECTED_BYTES"],
            )
    return _committed_result(
        kind="ALREADY_COMMITTED_EXACT",
        output=output,
        mode=mode,
        accepted=True,
        atomic_no_replace=(mode == PRIMARY_PUBLICATION_MODE),
        warning_codes=["FINAL_OUTPUT_ALREADY_COMMITTED_EXACT_NO_WRITE_PERFORMED"],
    )


def _validate_visible_commit_without_raising(
    output: Path,
    *,
    mode: str,
    atomic_no_replace: bool,
    initial_warning_codes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return an explicit committed state after final visibility is established."""
    warnings = list(initial_warning_codes)
    validation_error: Exception | None = None
    for attempt in range(2):
        try:
            _validate_publication_commit(output, expected_mode=mode)
        except Exception as exc:
            validation_error = exc
            continue
        if attempt == 1:
            warnings.append("POST_COMMIT_VALIDATION_RETRY_REQUIRED")
        validation_error = None
        break
    if validation_error is not None:
        warnings.append("POST_COMMIT_EXACT_VALIDATION_FAILED")
        return _committed_result(
            kind="COMMITTED_NOT_ACCEPTED",
            output=output,
            mode=mode,
            accepted=False,
            atomic_no_replace=atomic_no_replace,
            warning_codes=warnings,
        )
    try:
        _fsync_directory(output.parent)
    except Exception:
        warnings.append("POST_COMMIT_PARENT_DIRECTORY_FSYNC_FAILED")
    return _committed_result(
        kind=("COMMITTED_WITH_POST_COMMIT_WARNING" if warnings else "PUBLISHED"),
        output=output,
        mode=mode,
        accepted=True,
        atomic_no_replace=atomic_no_replace,
        warning_codes=warnings,
    )


def _publish_bundle(output: Path, payloads: Mapping[str, Any]) -> dict[str, Any]:
    complete = _render_complete_payloads(payloads)
    parent = output.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise PublicationError("publication parent must be an existing real directory")
    # Avoid creating a second fully rendered staging candidate when a prior
    # committed or partial final name is already visible.  A race after this
    # check is still closed by the kernel no-replace operation or fallback
    # exclusive mkdir and then classified without overwrite.
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        return _classify_existing_output(
            output, expected_complete_payloads=complete
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        _write_complete_directory(
            staging, complete, mode=PRIMARY_PUBLICATION_MODE, final_output=output
        )
        _validate_publication_commit(
            staging, expected_mode=PRIMARY_PUBLICATION_MODE, final_output=output
        )
    except Exception as exc:
        try:
            _cleanup_owned_precommit_staging(staging, output)
        except Exception as cleanup_exc:
            raise PublicationError(
                "precommit failed and owned staging requires manual adjudication at "
                f"{staging}"
            ) from cleanup_exc
        raise exc
    rename_error: Exception | None = None
    try:
        _rename_directory_noreplace(staging, output)
    except Exception as exc:
        rename_error = exc
    if rename_error is None:
        # The final name now exists atomically.  Nothing below may report an
        # ordinary pre-commit failure, even if validation or durability fails.
        return _validate_visible_commit_without_raising(
            output,
            mode=PRIMARY_PUBLICATION_MODE,
            atomic_no_replace=True,
        )

    # A wrapper may raise after the kernel already completed the rename.  The
    # final namespace is therefore the first source of truth after *any*
    # reported rename error; staging cleanup is allowed only after final
    # absence has been proven by lstat.
    try:
        output.lstat()
    except FileNotFoundError:
        final_visible_after_error = False
    else:
        final_visible_after_error = True
    if final_visible_after_error:
        classified = _classify_existing_output(
            output, expected_complete_payloads=complete
        )
        try:
            staging.lstat()
        except FileNotFoundError:
            pass
        else:
            try:
                _cleanup_owned_precommit_staging(staging, output)
            except Exception:
                return _precommit_staging_manual_result(
                    output,
                    staging,
                    warning_code=(
                        "RENAME_ERROR_FINAL_VISIBLE_AND_STAGING_CLEANUP_FAILED"
                    ),
                )
        if classified.get("accepted") is True:
            return _validate_visible_commit_without_raising(
                output,
                mode=str(classified["publication_mode"]),
                atomic_no_replace=(
                    classified["publication_mode"] == PRIMARY_PUBLICATION_MODE
                ),
                initial_warning_codes=[
                    "RENAME_REPORTED_ERROR_AFTER_EXACT_FINAL_VISIBILITY"
                ],
            )
        classified["post_commit_warning_codes"] = sorted(
            set(classified.get("post_commit_warning_codes", []))
            | {"RENAME_COMPLETION_AMBIGUOUS_FINAL_NOT_ACCEPTED"}
        )
        return classified

    if isinstance(rename_error, AtomicNoReplaceUnsupported):
        try:
            _cleanup_owned_precommit_staging(staging, output)
        except Exception:
            return _precommit_staging_manual_result(
                output,
                staging,
                warning_code="UNSUPPORTED_FALLBACK_PRIMARY_STAGING_CLEANUP_FAILED",
            )
        try:
            os.mkdir(output, 0o700)
        except FileExistsError:
            return _classify_existing_output(
                output, expected_complete_payloads=complete
            )
        # From this point the fallback final directory is externally visible.
        # A missing terminal marker is a preserved partial requiring manual
        # adjudication, never an implied safe retry.
        try:
            _write_complete_directory(
                output,
                complete,
                mode=FALLBACK_PUBLICATION_MODE,
                final_output=output,
            )
        except Exception:
            try:
                marker_exists = (output / PUBLICATION_COMMIT_FILENAME).lstat()
            except FileNotFoundError:
                return _partial_result(
                    output,
                    mode=FALLBACK_PUBLICATION_MODE,
                    warning_codes=["FALLBACK_WRITE_FAILED_BEFORE_TERMINAL_MARKER"],
                )
            if stat.S_ISLNK(marker_exists.st_mode) or not stat.S_ISREG(
                marker_exists.st_mode
            ):
                return _committed_result(
                    kind="COMMITTED_NOT_ACCEPTED",
                    output=output,
                    mode=FALLBACK_PUBLICATION_MODE,
                    accepted=False,
                    atomic_no_replace=False,
                    warning_codes=["FALLBACK_TERMINAL_MARKER_ENTRY_INVALID"],
                )
            return _validate_visible_commit_without_raising(
                output,
                mode=FALLBACK_PUBLICATION_MODE,
                atomic_no_replace=False,
                initial_warning_codes=[
                    "FALLBACK_POST_MARKER_WRITE_OR_DIRECTORY_FSYNC_ERROR"
                ],
            )
        return _validate_visible_commit_without_raising(
            output,
            mode=FALLBACK_PUBLICATION_MODE,
            atomic_no_replace=False,
        )
    if isinstance(rename_error, PublicationContention):
        try:
            _cleanup_owned_precommit_staging(staging, output)
        except Exception:
            return _precommit_staging_manual_result(
                output,
                staging,
                warning_code="CONTENDED_PRIMARY_STAGING_CLEANUP_FAILED",
            )
        return _partial_result(
            output,
            mode=PRIMARY_PUBLICATION_MODE,
            warning_codes=["RENAME_CONTENTION_REPORTED_BUT_FINAL_NOW_ABSENT"],
        )
    try:
        _cleanup_owned_precommit_staging(staging, output)
    except Exception as cleanup_exc:
        raise PublicationError(
            "precommit publication failed and staging requires manual "
            f"adjudication at {staging}"
        ) from cleanup_exc
    raise rename_error


def audit_gse114002_endpoint_geometry_reconciliation_v2(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    source_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol_file, source_file, output = _preflight_paths_before_read(
        protocol_path, source_path, output_directory
    )
    protocol, protocol_provenance = _load_protocol(protocol_file, protocol_sha256)
    binding = _verify_implementation_binding(
        protocol["implementation_binding"],
        protocol["authority"],
        protocol_file.parents[1],
    )
    if binding.get("verified") is not True:
        raise BindingNotFrozen(
            "implementation binding is UNKNOWN_NOT_ASSERTED; source access stopped"
        )
    source_contract = protocol["input_contract"]
    compressed, source_provenance = _read_source_snapshot(
        source_file,
        expected_sha256=str(source_contract["compressed_sha256"]),
        expected_bytes=int(source_contract["compressed_bytes"]),
    )
    input_audit, endpoint_audit, geometry_audit, blockers = _analyze_verified_source(
        compressed, protocol
    )
    conditional_present = bool(set(blockers) & CONDITIONAL_BLOCKERS)
    status = FAILED_MECHANICAL_STATUS if conditional_present else MECHANICAL_STATUS
    report = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": status,
        "qualified": False,
        "data_role": DATA_ROLE,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
        "blockers": blockers,
        "protocol_provenance": protocol_provenance,
        "source_provenance": source_provenance,
        "implementation_binding": binding,
    }
    result = _publish_bundle(
        output,
        {
            "INPUT_INTEGRITY_AUDIT.json": input_audit,
            "ENDPOINT_RECONCILIATION_AUDIT.json": endpoint_audit,
            "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": geometry_audit,
            "QUALIFICATION_REPORT.json": report,
        },
    )
    result.update(
        {
            "status": status,
            "qualified": False,
            "canonical_record_count": 0,
            "ordinary_study_contribution": 0,
            "a1_intervention_study_contribution": 0,
            "true_a2_dense_study_contribution": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "blocker_count": len(blockers),
        }
    )
    return result


def _cli_exit_code(result: Mapping[str, Any]) -> int:
    kind = result.get("kind")
    if (
        kind
        in {
            "PUBLISHED",
            "COMMITTED_WITH_POST_COMMIT_WARNING",
            "ALREADY_COMMITTED_EXACT",
        }
        and result.get("accepted") is True
    ):
        return 0
    if kind == "COMMITTED_NOT_ACCEPTED":
        return 3
    if kind == "PARTIAL_REQUIRES_MANUAL_ADJUDICATION":
        return 4
    if kind == "PRECOMMIT_STAGING_REQUIRES_MANUAL_ADJUDICATION":
        return 5
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit_gse114002_endpoint_geometry_reconciliation_v2(
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        source_path=args.source,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return _cli_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
