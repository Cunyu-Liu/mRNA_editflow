#!/usr/bin/env python3
"""Fail-closed gap qualifier for the public GSE114002 designed library.

This module can mechanically audit the pre-frozen, provisional source-pool
rule.  It cannot qualify GSE114002 as true A2: the production protocol leaves
the field semantics, biological uncertainty, complete construct context,
license, checkpoint exposure, near-duplicate split, and power gates open.
Consequently it publishes only an aggregate blocked bundle and never emits a
sequence, raw row, source field, or canonical record.
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
from typing import Any, Callable, Iterable, Mapping, Sequence


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE114002"
PROTOCOL_ID = "ROUTE_A_V3_GSE114002_DESIGNED_TRUE_A2_GAP_QUALIFICATION_V1"
PROTOCOL_STATUS = "PREFROZEN_FAIL_CLOSED_GAP_QUALIFIER_NOT_QUALIFIED"
SOURCE_BASENAME = "GSM3130443_designed_library.csv.gz"
EXPECTED_SOURCE_SHA256 = (
    "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5"
)
EXPECTED_SOURCE_BYTES = 17_332_142
EXPECTED_POOL_COUNT = 959
EXPECTED_CANDIDATE_COUNT = 3_899
EXPECTED_HEADER: tuple[str, ...] = (
    "", "utr", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "total", "r0", "r1", "r2", "r3", "r4",
    "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13",
    "r_total", "rl", "id", "info1", "info2", "info3", "info4", "library",
    "mother", "designed", "match_score",
)
INCLUDED_LIBRARIES = frozenset({"human_utrs", "snv"})
FRACTION_COLUMNS: tuple[str, ...] = tuple(str(i) for i in range(14))
FRACTION_WEIGHTS: tuple[int, ...] = tuple(range(14))
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "gse246381", "restricted", "sealed_external", "access_log",
)
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "sequence", "utr", "mother", "raw_row", "info1", "info2", "info3",
        "info4", "match_score", "id",
    }
)
ALWAYS_OUTPUT_FILES: tuple[str, ...] = (
    "POOL_GEOMETRY_AUDIT.json",
    "MEASUREMENT_UNCERTAINTY_AUDIT.json",
    "QUALIFICATION_REPORT.json",
    "SHA256SUMS",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DNA_RE = re.compile(r"^[ACGT]+$")
SOURCE_KEY_DOMAIN = b"ROUTE_A_V3_GSE114002_PROVISIONAL_SOURCE_KEY_V1\0"
SPLIT_DOMAIN = b"ROUTE_A_V3_GSE114002_OUTCOME_INDEPENDENT_SPLIT_V1\0"
ACCEPTED_A0_BASE_COMMIT = "fd722d5fa3c2538fce742b8942b1fb48e782760b"
ACTIVE_AUTHORITY_COMMIT = "d078060c81114687db5068902a5aad5d9bedbee6"
ACTIVE_CONTRACT_SHA256 = "3ba224de6277edd67387913cf1c83a5e1344e0ad44ef196db07d0772b45c4d79"
ACTIVE_DATA_ROLE_REGISTRY_SHA256 = "03a805c6441f0778225f9a8ec10feeadba23f572cd9ea7b234903384e6a902bf"
A1_QUALIFICATION_SHA256 = "1d348671de50c0fe8b155f8cc114d14a74360fe1a87f9d9bac5207ae794806c4"
BASE_PROTOCOL_BLOCKERS = frozenset(
    {
        "MOTHER_INFO1_INFO2_INFO3_MATCH_SCORE_ID_AUTHORITY_UNKNOWN_NOT_ASSERTED",
        "FULL_25NT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
        "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
        "LICENSE_AND_REDISTRIBUTION_UNKNOWN_NOT_ASSERTED",
        "NEAR_DUPLICATE_SPLIT_NOT_RUN",
        "PREFROZEN_GROUP_POWER_NOT_RUN",
        "OWNER_TECHNICAL_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
    }
)
IMPLEMENTATION_BLOCKER = "IMPLEMENTATION_COMMIT_UNKNOWN_NOT_ASSERTED"
CONDITIONAL_BLOCKERS = frozenset(
    {
        "SOURCE_ID_NOT_GLOBALLY_UNIQUE",
        "INVALID_OR_OUT_OF_RULE_INCLUDED_ROWS_PRESENT",
        "PROVISIONAL_959_POOL_3899_CANDIDATE_RECONCILIATION_MISMATCH",
        "FRACTION_COUNT_MRL_RECONCILIATION_NOT_EXACT",
    }
)
ALL_BLOCKERS = BASE_PROTOCOL_BLOCKERS | {IMPLEMENTATION_BLOCKER} | CONDITIONAL_BLOCKERS

# Tests replace the original pathname here after the single verified snapshot
# has been captured.  Production execution leaves this unset.
_POST_VERIFIED_SNAPSHOT_HOOK: Callable[[], None] | None = None


class QualificationError(RuntimeError):
    """Evidence or execution integrity failed and qualification must stop."""


class ScopeViolation(QualificationError):
    """A forbidden path was rejected before any payload read."""


class PublicationContention(QualificationError):
    """Another writer won the atomic no-replace output publication race."""


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    text = os.fspath(path).casefold()
    hits = sorted(token for token in FORBIDDEN_PATH_TOKENS if token in text)
    if hits:
        raise ScopeViolation(
            f"{label} rejected before read; forbidden path token(s): {','.join(hits)}"
        )


def _preflight_paths_before_read(
    protocol_path: Path | str,
    source_path: Path | str,
    output_directory: Path | str,
) -> tuple[Path, Path, Path]:
    raw = (
        (Path(protocol_path).expanduser(), "protocol path"),
        (Path(source_path).expanduser(), "ordinary public source path"),
        (Path(output_directory).expanduser(), "output path"),
    )
    # This loop must remain before resolve/lstat/open: path text is the first gate.
    for path, label in raw:
        _reject_forbidden_path(path, label=label)
    if raw[0][0].name != "route_a_v3_gse114002_a2_qualification.json":
        raise ScopeViolation("protocol path is not on the ordinary-public allowlist")
    if raw[1][0].name != SOURCE_BASENAME:
        raise ScopeViolation("source basename is not on the ordinary-public allowlist")
    resolved = tuple(path.resolve(strict=False) for path, _ in raw)
    for path, (_, label) in zip(resolved, raw):
        _reject_forbidden_path(path, label=label)
    return resolved  # type: ignore[return-value]


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise QualificationError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise QualificationError(f"{label} must be a non-symlink regular file")
    return info


def _read_regular_verified_snapshot(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    initial = _require_regular_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"{label} could not be opened safely") from exc
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
        )
        if identity(initial) != identity(opened) or not stat.S_ISREG(opened.st_mode):
            raise QualificationError(f"{label} changed before descriptor capture")
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            payload.extend(block)
        final = os.fstat(descriptor)
        if identity(opened) != identity(final) or len(payload) != final.st_size:
            raise QualificationError(f"{label} changed during descriptor capture")
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise QualificationError(f"{label} SHA-256 mismatch")
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise QualificationError(f"{label} byte count mismatch")
    return bytes(payload), {
        "basename": path.name,
        "sha256": observed_sha256,
        "bytes": len(payload),
        "parser_input_mode": "SINGLE_OPEN_VERIFIED_COMPRESSED_BYTES_SNAPSHOT",
    }


def _load_json_from_verified_bytes(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, provenance = _read_regular_verified_snapshot(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} root must be an object")
    return value, provenance


def _load_protocol_with_launch_hash(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise QualificationError("explicit protocol launch SHA-256 is invalid")
    raw, provenance = _read_regular_verified_snapshot(
        path,
        label="GSE114002 qualification protocol",
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("GSE114002 qualification protocol is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise QualificationError("GSE114002 qualification protocol root must be an object")
    provenance["launch_expected_sha256"] = expected_sha256
    return value, provenance


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise QualificationError(f"{label} must equal {expected!r}")


def _require_closed_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise QualificationError(
            f"{label} keys are not closed; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _safe_relative_path(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise QualificationError(f"{label} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.name in {"", ".", ".."} or ".." in relative.parts:
        raise QualificationError(f"{label} is not a safe relative path")
    _reject_forbidden_path(relative, label=label)
    candidate = repo_root.joinpath(*relative.parts)
    if candidate.parent == candidate:
        raise QualificationError(f"{label} is unsafe")
    return candidate


def _validate_protocol(document: Mapping[str, Any]) -> None:
    _require_closed_keys(
        document,
        {
            "contract_id", "schema_version", "protocol_id", "protocol_status",
            "dataset_id", "study_group_id", "data_role", "authority",
            "ordinary_public_asset_allowlist", "scope", "provisional_pool_rule",
            "measurement", "field_authority", "construct_context",
            "foundation_exposure", "license_and_redistribution",
            "split_and_leakage", "claim_boundary", "power_prefreeze",
            "known_blockers", "output_contract",
            "model_results_may_change_this_protocol",
        },
        label="protocol root",
    )
    _require_exact(document.get("contract_id"), CONTRACT_ID, label="protocol contract_id")
    _require_exact(document.get("schema_version"), "3.0.0", label="protocol schema_version")
    _require_exact(document.get("protocol_id"), PROTOCOL_ID, label="protocol protocol_id")
    _require_exact(document.get("protocol_status"), PROTOCOL_STATUS, label="protocol status")
    _require_exact(document.get("dataset_id"), DATASET_ID, label="protocol dataset")
    _require_exact(
        document.get("study_group_id"), "GSE114002_SAMPLE_2019_DESIGNED_LIBRARY",
        label="protocol study group",
    )
    _require_exact(
        document.get("data_role"), "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        label="protocol data role",
    )
    allowlist = document.get("ordinary_public_asset_allowlist")
    if not isinstance(allowlist, list) or len(allowlist) != 1 or not isinstance(allowlist[0], Mapping):
        raise QualificationError("protocol must bind exactly one ordinary-public asset")
    source = allowlist[0]
    expected_source = {
        "accession": "GSM3130443",
        "basename": SOURCE_BASENAME,
        "compressed_sha256": EXPECTED_SOURCE_SHA256,
        "compressed_bytes": EXPECTED_SOURCE_BYTES,
        "compression": "gzip",
        "encoding": "utf-8",
        "delimiter": ",",
        "exact_header": list(EXPECTED_HEADER),
    }
    _require_exact(dict(source), expected_source, label="ordinary-public source trust root")
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        raise QualificationError("protocol scope must be an object")
    _require_closed_keys(
        scope,
        {
            "ordinary_public_data_only", "region", "included_libraries", "assay_id",
            "context_id", "endpoint_id", "info4_may_define_context",
            "training_allowed", "model_selection_allowed",
            "canonical_materialization_allowed",
        },
        label="protocol scope",
    )
    for key, expected in {
        "ordinary_public_data_only": True,
        "region": "5UTR",
        "included_libraries": ["human_utrs", "snv"],
        "assay_id": "SAMPLE_2019_POLYSOME_FRACTIONATION",
        "context_id": "SAMPLE_2019_DESIGNED_LIBRARY_CONTEXT_UNRESOLVED",
        "endpoint_id": "MEAN_RIBOSOME_LOAD",
        "info4_may_define_context": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_materialization_allowed": False,
    }.items():
        _require_exact(scope.get(key), expected, label=f"protocol scope.{key}")
    pool = document.get("provisional_pool_rule")
    if not isinstance(pool, Mapping):
        raise QualificationError("protocol provisional_pool_rule must be an object")
    _require_closed_keys(
        pool,
        {
            "status", "source_key_fields", "identity_rule", "edited_rule",
            "exact_identity_count_per_pool",
            "minimum_distinct_edited_candidates_per_pool",
            "id_must_be_globally_unique", "expected_pool_count",
            "expected_distinct_candidate_count", "expected_counts_are_authority",
            "reconciliation_action",
        },
        label="protocol provisional_pool_rule",
    )
    for key, expected in {
        "status": "PROVISIONAL_POOL_GEOMETRY_PENDING_FIELD_AUTHORITY",
        "source_key_fields": [
            "mother", "library", "FIXED_ASSAY_ID", "FIXED_CONTEXT_ID",
            "FIXED_ENDPOINT_ID",
        ],
        "identity_rule": "utr_equals_mother_and_designed_true",
        "edited_rule": "designed_false_and_equal_length_hamming_distance_1_to_3",
        "exact_identity_count_per_pool": 1,
        "minimum_distinct_edited_candidates_per_pool": 3,
        "id_must_be_globally_unique": True,
        "expected_pool_count": EXPECTED_POOL_COUNT,
        "expected_distinct_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "expected_counts_are_authority": False,
        "reconciliation_action": "MISMATCH_BLOCKS_AND_MATCH_REMAINS_PROVISIONAL",
    }.items():
        _require_exact(pool.get(key), expected, label=f"protocol provisional_pool_rule.{key}")
    measurement = document.get("measurement")
    if not isinstance(measurement, Mapping):
        raise QualificationError("protocol measurement must be an object")
    _require_closed_keys(
        measurement,
        {
            "paper_endpoint_column", "fraction_count_columns", "fraction_weights",
            "fraction_count_derived_value_role", "biological_replicate_status",
            "paper_standard_error_status", "technical_fraction_uncertainty_may_be_derived",
            "technical_fraction_uncertainty_is_biological_standard_error",
            "owner_uncertainty_policy_status",
        },
        label="protocol measurement",
    )
    for key, expected in {
        "paper_endpoint_column": "rl",
        "fraction_count_columns": list(FRACTION_COLUMNS),
        "fraction_weights": list(FRACTION_WEIGHTS),
        "fraction_count_derived_value_role": "TECHNICAL_RECONCILIATION_DIAGNOSTIC_ONLY",
        "biological_replicate_status": "ABSENT_BY_DESIGN",
        "paper_standard_error_status": "ABSENT",
        "technical_fraction_uncertainty_may_be_derived": True,
        "technical_fraction_uncertainty_is_biological_standard_error": False,
        "owner_uncertainty_policy_status": "UNKNOWN_NOT_ASSERTED",
    }.items():
        _require_exact(measurement.get(key), expected, label=f"protocol measurement.{key}")
    fields = document.get("field_authority")
    if not isinstance(fields, Mapping) or set(fields) != {
        "mother", "info1", "info2", "info3", "match_score", "id"
    } or any(value != "UNKNOWN_NOT_ASSERTED" for value in fields.values()):
        raise QualificationError("source field authority must remain explicitly unknown")
    for section_name, expected_values in {
        "construct_context": {
            "full_25nt_prefix_status": "UNKNOWN_NOT_ASSERTED",
            "reporter_identity_status": "UNKNOWN_NOT_ASSERTED",
            "rna_chemistry_status": "UNKNOWN_NOT_ASSERTED",
            "unknown_context_blocks_true_a2": True,
        },
        "foundation_exposure": {
            "sequence_exposed": True,
            "checkpoint_id": "UNKNOWN_NOT_ASSERTED",
            "checkpoint_sha256": "UNKNOWN_NOT_ASSERTED",
            "checkpoint_specific_audit_status": "UNKNOWN_NOT_ASSERTED",
            "unknown_checkpoint_blocks_true_a2": True,
        },
        "license_and_redistribution": {
            "audit_status": "UNKNOWN_NOT_ASSERTED",
            "license_id": "UNKNOWN_NOT_ASSERTED",
            "canonical_sequence_redistribution_allowed": False,
            "unknown_license_blocks_true_a2": True,
        },
        "power_prefreeze": {
            "analysis_unit": "SOURCE_POOL",
            "bootstrap_unit": "SOURCE_POOL",
            "status": "NOT_RUN",
            "model_results_may_change_rule": False,
        },
    }.items():
        section = document.get(section_name)
        if not isinstance(section, Mapping):
            raise QualificationError(f"protocol {section_name} must be an object")
        _require_closed_keys(
            section, set(expected_values), label=f"protocol {section_name}"
        )
        for key, expected in expected_values.items():
            _require_exact(
                section.get(key), expected, label=f"protocol {section_name}.{key}"
            )
    boundary = document.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise QualificationError("claim boundary must be an object")
    boundary_expected = {
        "k5_dense_pool_count_observed": 0,
        "k5_dense_pool_count_role": "CLAIM_BOUNDARY_ONLY_NOT_A1_HARD_BLOCKER",
        "true_a2_status": "NOT_QUALIFIED",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
    }
    _require_closed_keys(boundary, set(boundary_expected), label="protocol claim_boundary")
    for key, expected in boundary_expected.items():
        _require_exact(boundary.get(key), expected, label=f"claim boundary.{key}")
    split = document.get("split_and_leakage")
    if not isinstance(split, Mapping):
        raise QualificationError("protocol split_and_leakage must be an object")
    split_expected = {
        "assignment_inputs": [
            "source_key_sha256", "library", "fixed_assay_id", "fixed_context_id"
        ],
        "outcome_or_rl_used_for_assignment": False,
        "near_duplicate_sequence_cluster_audit": "NOT_RUN",
        "partition_role": "DEVELOPMENT_ONLY",
    }
    _require_closed_keys(split, set(split_expected), label="protocol split_and_leakage")
    for key, expected in split_expected.items():
        _require_exact(split.get(key), expected, label=f"protocol split_and_leakage.{key}")
    authority = document.get("authority")
    if not isinstance(authority, Mapping):
        raise QualificationError("protocol authority must be an object")
    _require_closed_keys(
        authority,
        {
            "contract_path", "contract_sha256", "accepted_a0_base_commit",
            "active_authority_commit", "active_amendment_decision_ids",
            "a1_qualification_path", "a1_qualification_sha256",
            "data_role_registry_path", "data_role_registry_sha256",
            "qualifier_path", "qualifier_sha256", "focused_test_path",
            "focused_test_sha256", "implementation_commit",
        },
        label="protocol authority",
    )
    _require_exact(
        authority.get("accepted_a0_base_commit"), ACCEPTED_A0_BASE_COMMIT,
        label="protocol accepted A0 base commit",
    )
    _require_exact(
        authority.get("active_authority_commit"), ACTIVE_AUTHORITY_COMMIT,
        label="protocol active authority commit",
    )
    _require_exact(
        authority.get("active_amendment_decision_ids"), ["V3-DEC-017"],
        label="protocol active amendment decisions",
    )
    for key in (
        "contract_path", "a1_qualification_path", "data_role_registry_path",
        "qualifier_path", "focused_test_path",
    ):
        _safe_relative_path(Path("."), authority.get(key), label=f"protocol authority.{key}")
    for key, expected in {
        "contract_path": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
        "contract_sha256": ACTIVE_CONTRACT_SHA256,
        "a1_qualification_path": "configs/route_a_v3_a1_qualification.json",
        "a1_qualification_sha256": A1_QUALIFICATION_SHA256,
        "data_role_registry_path": "docs/execution/route_a_v3_data_role_registry.yaml",
        "data_role_registry_sha256": ACTIVE_DATA_ROLE_REGISTRY_SHA256,
        "qualifier_path": "scripts/route_a_v3/qualify_gse114002_designed_a2.py",
        "focused_test_path": "tests/route_a_v3/test_qualify_gse114002_designed_a2.py",
    }.items():
        _require_exact(authority.get(key), expected, label=f"protocol authority.{key}")
    for key in ("contract_sha256", "a1_qualification_sha256", "data_role_registry_sha256"):
        if not SHA256_RE.fullmatch(str(authority.get(key, ""))):
            raise QualificationError(f"protocol authority.{key} is not a SHA-256")
    for key in ("qualifier_sha256", "focused_test_sha256"):
        value = authority.get(key)
        if value != "UNKNOWN_NOT_ASSERTED" and not SHA256_RE.fullmatch(str(value)):
            raise QualificationError(f"protocol authority.{key} is invalid")
    implementation = authority.get("implementation_commit")
    if implementation != "UNKNOWN_NOT_ASSERTED" and not COMMIT_RE.fullmatch(str(implementation)):
        raise QualificationError("protocol implementation_commit is invalid")
    blockers = document.get("known_blockers")
    if not isinstance(blockers, list) or not blockers or any(
        not isinstance(item, str) or not item for item in blockers
    ) or len(blockers) != len(set(blockers)):
        raise QualificationError("protocol known blockers must be a unique nonempty string list")
    expected_blockers = set(BASE_PROTOCOL_BLOCKERS)
    if implementation == "UNKNOWN_NOT_ASSERTED":
        expected_blockers.add(IMPLEMENTATION_BLOCKER)
    if set(blockers) != expected_blockers:
        raise QualificationError("protocol known blockers differ from the closed conditional set")
    output_contract = document.get("output_contract")
    if not isinstance(output_contract, Mapping):
        raise QualificationError("protocol output_contract must be an object")
    _require_closed_keys(
        output_contract,
        {
            "aggregate_only_blocked_bundle", "forbidden_output_fields",
            "always_materialized_files", "conditional_success_only_private_canonical_path",
            "atomic_no_overwrite_publish_required",
        },
        label="protocol output_contract",
    )
    _require_exact(
        output_contract.get("forbidden_output_fields"),
        ["sequence", "utr", "mother", "raw_row", "info1", "info2", "info3", "info4", "match_score", "id"],
        label="protocol forbidden output fields",
    )
    _require_exact(
        output_contract.get("always_materialized_files"), list(ALWAYS_OUTPUT_FILES),
        label="protocol output filenames",
    )
    for key, expected in {
        "aggregate_only_blocked_bundle": True,
        "conditional_success_only_private_canonical_path": "HARD_DISABLED_NOT_IMPLEMENTED",
        "atomic_no_overwrite_publish_required": True,
    }.items():
        _require_exact(
            output_contract.get(key), expected, label=f"protocol output_contract.{key}"
        )
    _require_exact(document.get("model_results_may_change_this_protocol"), False, label="model-result mutability")


def _verify_git_binding(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    authority = protocol["authority"]
    commit = authority["implementation_commit"]
    if commit == "UNKNOWN_NOT_ASSERTED":
        return {
            "status": "UNKNOWN_NOT_ASSERTED",
            "accepted_a0_base_commit": authority["accepted_a0_base_commit"],
            "active_authority_commit": authority["active_authority_commit"],
            "implementation_commit": commit,
            "observed_head": None,
            "accepted_a0_is_ancestor_of_active_authority": None,
            "active_authority_is_ancestor_of_implementation": None,
            "implementation_is_ancestor_of_head": None,
            "active_authority_file_hashes_match": None,
            "implementation_file_hashes_match": None,
            "worktree_clean": None,
        }

    accepted = authority["accepted_a0_base_commit"]
    active = authority["active_authority_commit"]
    authority_files = (
        (authority["contract_path"], authority["contract_sha256"]),
        (authority["data_role_registry_path"], authority["data_role_registry_sha256"]),
    )
    implementation_files = (
        (authority["qualifier_path"], authority["qualifier_sha256"]),
        (authority["focused_test_path"], authority["focused_test_sha256"]),
    )
    for relative, expected_hash in authority_files + implementation_files:
        _safe_relative_path(repo_root, relative, label="Git-bound relative path")
        if not SHA256_RE.fullmatch(str(expected_hash)):
            raise QualificationError(f"Git-bound file {relative} lacks a frozen SHA-256")

    def run_git(args: Sequence[str], *, binary: bool = False) -> bytes | str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=not binary,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise QualificationError("repository binding could not be verified") from exc
        return result.stdout

    def require_ancestor(older: str, newer: str, *, label: str) -> None:
        try:
            subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", older, newer],
                check=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise QualificationError(f"{label} ancestry is not satisfied") from exc

    def require_blobs(bound_commit: str, files: Sequence[tuple[str, str]], *, label: str) -> None:
        for relative, expected_hash in files:
            blob = run_git(["show", f"{bound_commit}:{relative}"], binary=True)
            assert isinstance(blob, bytes)
            if _sha256_bytes(blob) != expected_hash:
                raise QualificationError(f"{label} file hash differs at its bound commit")

    head = str(run_git(["rev-parse", "HEAD"])).strip()
    require_ancestor(accepted, active, label="accepted A0 to active authority")
    require_ancestor(active, commit, label="active authority to implementation")
    require_ancestor(commit, head, label="implementation to executing HEAD")
    require_blobs(active, authority_files, label="active authority")
    require_blobs(commit, implementation_files, label="implementation")
    porcelain = str(run_git(["status", "--porcelain", "--untracked-files=all"]))
    if porcelain:
        raise QualificationError("repository worktree is not clean during qualification")
    return {
        "status": "PASS",
        "accepted_a0_base_commit": accepted,
        "active_authority_commit": active,
        "implementation_commit": commit,
        "observed_head": head,
        "accepted_a0_is_ancestor_of_active_authority": True,
        "active_authority_is_ancestor_of_implementation": True,
        "implementation_is_ancestor_of_head": True,
        "active_authority_file_hashes_match": True,
        "implementation_file_hashes_match": True,
        "worktree_clean": True,
    }


def _verify_running_qualifier(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = protocol["authority"]["qualifier_sha256"]
    if expected == "UNKNOWN_NOT_ASSERTED":
        return {"status": "UNKNOWN_NOT_ASSERTED"}
    raw, provenance = _read_regular_verified_snapshot(
        Path(__file__).resolve(), label="running GSE114002 qualifier"
    )
    if _sha256_bytes(raw) != expected:
        raise QualificationError("running qualifier does not match its frozen SHA-256")
    return {"status": "PASS", "sha256": provenance["sha256"]}


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def hamming_distance(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def derive_fraction_count_mrl_and_technical_se(
    counts: Sequence[float], weights: Sequence[float] = FRACTION_WEIGHTS
) -> tuple[float, float]:
    """Return weighted MRL and multinomial technical SE, never biological SE."""
    if len(counts) != len(weights) or not counts:
        raise QualificationError("fraction counts and weights must be nonempty and aligned")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in counts):
        raise QualificationError("fraction counts must be finite and nonnegative")
    total = math.fsum(float(value) for value in counts)
    if total <= 0:
        raise QualificationError("fraction counts must have positive total")
    mean = math.fsum(float(count) * float(weight) for count, weight in zip(counts, weights)) / total
    variance = math.fsum(
        float(count) * (float(weight) - mean) ** 2
        for count, weight in zip(counts, weights)
    ) / total
    return mean, math.sqrt(max(0.0, variance) / total)


def provisional_source_key(
    mother: str, library: str, *, assay_id: str, context_id: str, endpoint_id: str
) -> str:
    parts = (mother, library, assay_id, context_id, endpoint_id)
    digest = hashlib.sha256(SOURCE_KEY_DOMAIN)
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def outcome_independent_partition(
    source_key_sha256: str, library: str, assay_id: str, context_id: str
) -> str:
    # No endpoint value, rl, delta, significance, or candidate effect enters this hash.
    payload = "\0".join((source_key_sha256, library, assay_id, context_id)).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(SPLIT_DOMAIN + payload).digest()[:8], "big") % 10
    return "DEVELOPMENT_A" if bucket < 8 else "DEVELOPMENT_B"


def reconcile_provisional_geometry(
    observed_pool_count: int,
    observed_candidate_count: int,
    *,
    expected_pool_count: int = EXPECTED_POOL_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
) -> dict[str, Any]:
    pool_match = observed_pool_count == expected_pool_count
    candidate_match = observed_candidate_count == expected_candidate_count
    return {
        "status": (
            "PROVISIONAL_COUNTS_MATCH_NOT_AUTHORITY"
            if pool_match and candidate_match
            else "PROVISIONAL_COUNTS_MISMATCH_BLOCKED"
        ),
        "observed_pool_count": observed_pool_count,
        "observed_distinct_candidate_count": observed_candidate_count,
        "expected_pool_count": expected_pool_count,
        "expected_distinct_candidate_count": expected_candidate_count,
        "pool_count_matches": pool_match,
        "candidate_count_matches": candidate_match,
        "expected_counts_are_authority": False,
    }


def _parse_verified_gzip_csv(compressed: bytes) -> tuple[list[str], Iterable[list[str]]]:
    try:
        uncompressed = gzip.decompress(compressed)
        text = uncompressed.decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise QualificationError("designed source is not a valid UTF-8 gzip CSV") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise QualificationError("designed source is empty") from exc
    if tuple(header) != EXPECTED_HEADER:
        raise QualificationError("designed source exact header mismatch")
    return header, reader


def _analyze_verified_source(compressed: bytes, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    _, rows = _parse_verified_gzip_csv(compressed)
    scope = protocol["scope"]
    assay_id = str(scope["assay_id"])
    context_id = str(scope["context_id"])
    endpoint_id = str(scope["endpoint_id"])
    index = {name: position for position, name in enumerate(EXPECTED_HEADER)}
    pools: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"identity": [], "edited": set(), "library": "", "partition": ""}
    )
    ids: Counter[str] = Counter()
    row_count = included_count = invalid_row_count = 0
    mrl_reconciliation_count = mrl_mismatch_count = technical_se_count = 0
    blockers = set(str(value) for value in protocol["known_blockers"])
    for row in rows:
        row_count += 1
        if len(row) != len(EXPECTED_HEADER):
            invalid_row_count += 1
            continue
        # The protocol says globally unique: count the id column across the
        # complete 42-column asset before any library-scope filtering.
        ids[row[index["id"]]] += 1
        library = row[index["library"]]
        if library not in INCLUDED_LIBRARIES:
            continue
        included_count += 1
        utr = row[index["utr"]].upper()
        mother = row[index["mother"]].upper()
        designed = _parse_bool(row[index["designed"]])
        if not DNA_RE.fullmatch(utr) or not DNA_RE.fullmatch(mother) or designed is None:
            invalid_row_count += 1
            continue
        key = provisional_source_key(
            mother, library, assay_id=assay_id, context_id=context_id,
            endpoint_id=endpoint_id,
        )
        pool = pools[key]
        pool["library"] = library
        pool["partition"] = outcome_independent_partition(key, library, assay_id, context_id)
        distance = hamming_distance(utr, mother)
        if designed is True and utr == mother:
            pool["identity"].append(_sha256_bytes(utr.encode("ascii")))
        elif designed is False and distance is not None and 1 <= distance <= 3:
            pool["edited"].add(_sha256_bytes(utr.encode("ascii")))
        else:
            invalid_row_count += 1
        try:
            counts = [float(row[index[column]]) for column in FRACTION_COLUMNS]
            derived_mrl, _technical_se = derive_fraction_count_mrl_and_technical_se(counts)
            published_rl = float(row[index["rl"]])
            if not math.isfinite(published_rl):
                raise ValueError
            mrl_reconciliation_count += 1
            technical_se_count += 1
            if not math.isclose(derived_mrl, published_rl, rel_tol=1e-6, abs_tol=1e-6):
                mrl_mismatch_count += 1
        except (ValueError, QualificationError):
            mrl_mismatch_count += 1
    duplicate_id_value_count = sum(1 for count in ids.values() if count > 1)
    if duplicate_id_value_count:
        blockers.add("SOURCE_ID_NOT_GLOBALLY_UNIQUE")
    if invalid_row_count:
        blockers.add("INVALID_OR_OUT_OF_RULE_INCLUDED_ROWS_PRESENT")
    eligible = {
        key: value
        for key, value in pools.items()
        if len(value["identity"]) == 1 and len(value["edited"]) >= 3
    }
    identity_bad_count = sum(len(value["identity"]) != 1 for value in pools.values())
    too_few_edited_count = sum(len(value["edited"]) < 3 for value in pools.values())
    # A Route-A candidate is an edited construct.  The unique identity anchor
    # defines the source and is not added to the candidate count.
    candidate_count = sum(len(value["edited"]) for value in eligible.values())
    geometry = reconcile_provisional_geometry(len(eligible), candidate_count)
    if geometry["status"] != "PROVISIONAL_COUNTS_MATCH_NOT_AUTHORITY":
        blockers.add("PROVISIONAL_959_POOL_3899_CANDIDATE_RECONCILIATION_MISMATCH")
    if mrl_mismatch_count:
        blockers.add("FRACTION_COUNT_MRL_RECONCILIATION_NOT_EXACT")
    pool_audit = {
        "dataset_id": DATASET_ID,
        "status": "PROVISIONAL_POOL_GEOMETRY_PENDING_FIELD_AUTHORITY",
        "source_asset_sha256": EXPECTED_SOURCE_SHA256,
        "source_row_count": row_count,
        "included_library_row_count": included_count,
        "provisional_source_pool_count": len(pools),
        "eligible_provisional_pool_count": len(eligible),
        "eligible_provisional_distinct_candidate_count": candidate_count,
        "pool_with_nonunique_or_missing_identity_count": identity_bad_count,
        "pool_with_fewer_than_three_distinct_edited_count": too_few_edited_count,
        "duplicate_id_value_count": duplicate_id_value_count,
        "invalid_or_out_of_rule_included_row_count": invalid_row_count,
        "outcome_independent_partition_counts": dict(
            sorted(Counter(value["partition"] for value in eligible.values()).items())
        ),
        "reconciliation": geometry,
        "field_authority_status": "UNKNOWN_NOT_ASSERTED",
        "info4_used_as_context": False,
        "sequence_or_source_fields_emitted": False,
    }
    uncertainty = {
        "dataset_id": DATASET_ID,
        "endpoint": "MEAN_RIBOSOME_LOAD",
        "fraction_count_mrl_reconciliation_row_count": mrl_reconciliation_count,
        "fraction_count_mrl_mismatch_count": mrl_mismatch_count,
        "technical_fraction_uncertainty_derived_row_count": technical_se_count,
        "technical_uncertainty_role": "TECHNICAL_RECONCILIATION_DIAGNOSTIC_ONLY",
        "biological_replicate_status": "ABSENT_BY_DESIGN",
        "paper_standard_error_status": "ABSENT",
        "biological_standard_error_derived": False,
        "owner_uncertainty_policy_status": "UNKNOWN_NOT_ASSERTED",
    }
    return pool_audit, uncertainty, sorted(blockers)


def _assert_aggregate_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_OUTPUT_KEYS:
                raise QualificationError(f"forbidden raw field in aggregate output: {key}")
            _assert_aggregate_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_aggregate_safe(child)


def _validate_closed_output_payloads(payloads: Mapping[str, Any]) -> None:
    if set(payloads) != {
        "POOL_GEOMETRY_AUDIT.json",
        "MEASUREMENT_UNCERTAINTY_AUDIT.json",
        "QUALIFICATION_REPORT.json",
    }:
        raise QualificationError("aggregate output payload filenames are not closed")
    pool = payloads["POOL_GEOMETRY_AUDIT.json"]
    uncertainty = payloads["MEASUREMENT_UNCERTAINTY_AUDIT.json"]
    report = payloads["QUALIFICATION_REPORT.json"]
    if not all(isinstance(value, Mapping) for value in (pool, uncertainty, report)):
        raise QualificationError("aggregate output roots must be objects")
    _require_closed_keys(
        pool,
        {
            "dataset_id", "status", "source_asset_sha256", "source_row_count",
            "included_library_row_count", "provisional_source_pool_count",
            "eligible_provisional_pool_count",
            "eligible_provisional_distinct_candidate_count",
            "pool_with_nonunique_or_missing_identity_count",
            "pool_with_fewer_than_three_distinct_edited_count",
            "duplicate_id_value_count", "invalid_or_out_of_rule_included_row_count",
            "outcome_independent_partition_counts", "reconciliation",
            "field_authority_status", "info4_used_as_context",
            "sequence_or_source_fields_emitted",
        },
        label="POOL_GEOMETRY_AUDIT",
    )
    reconciliation = pool["reconciliation"]
    if not isinstance(reconciliation, Mapping):
        raise QualificationError("pool reconciliation must be an object")
    _require_closed_keys(
        reconciliation,
        {
            "status", "observed_pool_count", "observed_distinct_candidate_count",
            "expected_pool_count", "expected_distinct_candidate_count",
            "pool_count_matches", "candidate_count_matches",
            "expected_counts_are_authority",
        },
        label="pool reconciliation",
    )
    _require_closed_keys(
        uncertainty,
        {
            "dataset_id", "endpoint", "fraction_count_mrl_reconciliation_row_count",
            "fraction_count_mrl_mismatch_count",
            "technical_fraction_uncertainty_derived_row_count",
            "technical_uncertainty_role", "biological_replicate_status",
            "paper_standard_error_status", "biological_standard_error_derived",
            "owner_uncertainty_policy_status",
        },
        label="MEASUREMENT_UNCERTAINTY_AUDIT",
    )
    _require_closed_keys(
        report,
        {
            "contract_id", "protocol_id", "dataset_id", "status", "data_role",
            "true_a2_status", "scientific_claim_status", "training_allowed",
            "model_selection_allowed", "canonical_materialization_allowed",
            "canonical_record_count", "k5_dense_pool_count_observed",
            "k5_dense_pool_count_role", "blockers", "protocol_provenance",
            "source_provenance", "git_binding", "running_qualifier_binding",
            "aggregate_only",
        },
        label="QUALIFICATION_REPORT",
    )
    blockers = report["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not set(blockers).issubset(ALL_BLOCKERS)
    ):
        raise QualificationError("qualification blockers violate the closed enum")
    for key, expected in {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "BLOCKED_NOT_QUALIFIED",
        "data_role": "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        "true_a2_status": "NOT_QUALIFIED",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_materialization_allowed": False,
        "canonical_record_count": 0,
        "k5_dense_pool_count_observed": 0,
        "k5_dense_pool_count_role": "CLAIM_BOUNDARY_ONLY_NOT_A1_HARD_BLOCKER",
        "aggregate_only": True,
    }.items():
        _require_exact(report.get(key), expected, label=f"qualification report.{key}")
    _require_exact(pool.get("dataset_id"), DATASET_ID, label="pool audit dataset")
    _require_exact(
        pool.get("status"), "PROVISIONAL_POOL_GEOMETRY_PENDING_FIELD_AUTHORITY",
        label="pool audit status",
    )
    _require_exact(
        pool.get("sequence_or_source_fields_emitted"), False,
        label="pool audit raw-field boundary",
    )
    partitions = pool.get("outcome_independent_partition_counts")
    if not isinstance(partitions, Mapping) or not set(partitions).issubset(
        {"DEVELOPMENT_A", "DEVELOPMENT_B"}
    ):
        raise QualificationError("pool audit partition counts violate the closed enum")
    protocol_provenance = report.get("protocol_provenance")
    source_provenance = report.get("source_provenance")
    git_binding = report.get("git_binding")
    running_binding = report.get("running_qualifier_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol_provenance, source_provenance, git_binding, running_binding)
    ):
        raise QualificationError("qualification report provenance roots must be objects")
    _require_closed_keys(
        protocol_provenance,
        {"basename", "sha256", "bytes", "parser_input_mode", "launch_expected_sha256"},
        label="protocol provenance",
    )
    _require_closed_keys(
        source_provenance,
        {"basename", "sha256", "bytes", "parser_input_mode"},
        label="source provenance",
    )
    _require_closed_keys(
        git_binding,
        {
            "status", "accepted_a0_base_commit", "active_authority_commit",
            "implementation_commit", "observed_head",
            "accepted_a0_is_ancestor_of_active_authority",
            "active_authority_is_ancestor_of_implementation",
            "implementation_is_ancestor_of_head",
            "active_authority_file_hashes_match", "implementation_file_hashes_match",
            "worktree_clean",
        },
        label="Git binding output",
    )
    running_expected = (
        {"status"}
        if running_binding.get("status") == "UNKNOWN_NOT_ASSERTED"
        else {"status", "sha256"}
    )
    _require_closed_keys(running_binding, running_expected, label="running qualifier binding")
    for name, value in payloads.items():
        _assert_aggregate_safe(value)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(
    source: Path, target: Path, parent_descriptor: int
) -> None:
    if source.parent != target.parent:
        raise QualificationError("atomic no-replace publication requires one parent")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source.name)
    target_bytes = os.fsencode(target.name)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            parent_descriptor, source_bytes, parent_descriptor, target_bytes, 1
        )
    elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        result = libc.renameatx_np(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            target_bytes,
            0x00000004,
        )
    else:
        raise QualificationError("atomic kernel no-replace directory publish is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PublicationContention(f"atomic output publication contended: {target}")
        raise QualificationError(f"atomic no-replace publish failed with errno {error}")


def _publish_blocked_bundle(
    output: Path,
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    parent = output.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError as exc:
        raise QualificationError("output parent directory is missing") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise QualificationError("output parent must be a non-symlink directory")
    _validate_closed_output_payloads(payloads)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    rendered: dict[str, bytes] = {}
    for name, value in payloads.items():
        rendered[name] = _pretty_json_bytes(value)
    for name, payload in rendered.items():
        _write_exclusive(staging / name, payload)
    sums = "".join(
        f"{_sha256_bytes(rendered[name])}  {name}\n" for name in sorted(rendered)
    ).encode("ascii")
    _write_exclusive(staging / "SHA256SUMS", sums)
    directory_descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    parent_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as exc:
        raise QualificationError("publication parent could not be opened before commit") from exc
    committed = False
    try:
        try:
            _rename_directory_noreplace(staging, output, parent_descriptor)
            committed = True
        except PublicationContention:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
            return {
                "kind": "CONTENDED",
                "published": False,
                "output": str(output),
                "file_count": 0,
                "atomic_no_replace": True,
                "contention_status": "ATOMIC_NO_REPLACE_CONTENDED",
                "durability_warning_codes": [],
            }
        except Exception:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
            raise
        durability_warnings: list[str] = []
        try:
            os.fsync(parent_descriptor)
        except Exception:
            durability_warnings.append("POST_COMMIT_PARENT_DIRECTORY_FSYNC_FAILED")
        try:
            os.close(parent_descriptor)
        except Exception:
            durability_warnings.append("POST_COMMIT_PARENT_DIRECTORY_CLOSE_FAILED")
    except Exception:
        if committed:
            # This branch is intentionally defensive: no post-commit exception
            # may be converted into a pre-commit failure or a zero-output claim.
            return {
                "kind": "PUBLISHED",
                "published": True,
                "output": str(output),
                "file_count": len(ALWAYS_OUTPUT_FILES),
                "atomic_no_replace": True,
                "contention_status": None,
                "durability_warning_codes": [
                    "POST_COMMIT_UNEXPECTED_FINALIZATION_ERROR"
                ],
            }
        raise
    return {
        "kind": "PUBLISHED",
        "published": True,
        "output": str(output),
        "file_count": len(ALWAYS_OUTPUT_FILES),
        "atomic_no_replace": True,
        "contention_status": None,
        "durability_warning_codes": sorted(set(durability_warnings)),
    }


def qualify_gse114002_designed_a2(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    source_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    protocol_file, source_file, output = _preflight_paths_before_read(
        protocol_path, source_path, output_directory
    )
    protocol, protocol_provenance = _load_protocol_with_launch_hash(
        protocol_file, protocol_sha256
    )
    _validate_protocol(protocol)
    running_qualifier = _verify_running_qualifier(protocol)
    # The authority/Git chain is a source-read gate, not merely a report field.
    git_binding = _verify_git_binding(protocol, protocol_file.parents[1])
    compressed, source_provenance = _read_regular_verified_snapshot(
        source_file,
        label="ordinary public GSE114002 designed source",
        expected_sha256=EXPECTED_SOURCE_SHA256,
        expected_bytes=EXPECTED_SOURCE_BYTES,
    )
    if _POST_VERIFIED_SNAPSHOT_HOOK is not None:
        _POST_VERIFIED_SNAPSHOT_HOOK()
    pool_audit, uncertainty, blockers = _analyze_verified_source(compressed, protocol)
    if git_binding["status"] != "PASS":
        blockers = sorted(set(blockers) | {IMPLEMENTATION_BLOCKER})
    report = {
        "contract_id": CONTRACT_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "BLOCKED_NOT_QUALIFIED",
        "data_role": "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED",
        "true_a2_status": "NOT_QUALIFIED",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_materialization_allowed": False,
        "canonical_record_count": 0,
        "k5_dense_pool_count_observed": 0,
        "k5_dense_pool_count_role": "CLAIM_BOUNDARY_ONLY_NOT_A1_HARD_BLOCKER",
        "blockers": blockers,
        "protocol_provenance": protocol_provenance,
        "source_provenance": source_provenance,
        "git_binding": git_binding,
        "running_qualifier_binding": running_qualifier,
        "aggregate_only": True,
    }
    result = _publish_blocked_bundle(
        output,
        {
            "POOL_GEOMETRY_AUDIT.json": pool_audit,
            "MEASUREMENT_UNCERTAINTY_AUDIT.json": uncertainty,
            "QUALIFICATION_REPORT.json": report,
        },
    )
    if result["kind"] == "CONTENDED":
        result.update({
            "status": "CONTENDED",
            "canonical_record_count": 0,
            "blocker_count": len(blockers),
        })
        return result
    result.update({
        "status": report["status"],
        "canonical_record_count": 0,
        "blocker_count": len(blockers),
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    result = qualify_gse114002_designed_a2(
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        source_path=args.source,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
