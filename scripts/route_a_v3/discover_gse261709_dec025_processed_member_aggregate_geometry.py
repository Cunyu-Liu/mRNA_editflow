#!/usr/bin/env python3
"""Aggregate-only GSE261709 DEC025 processed-member discovery successor.

The reviewed exact3 protocol freezes the owner decision while its own grouped
implementation identity remains unknown.  The production entrypoint therefore
stops before repository, archive, member, or output-path I/O until a direct
config-only child binds that identity; only that bound child can reach the
fixed built-in TAR/gzip/TSV parser below.  The append-only predecessor attempt
is retained as failed closed: its frozen SHA and seven-member directory were
correct, but its outer byte count was a transcription error.

The parser holds barcode tokens only in memory long enough to compute set
geometry.  Its report contains aggregate schema, dimension, missingness,
duplicate, and cross-member join counts.  It cannot invoke or emit input for
the existing qualifier, and every scientific gate remains STOP.
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
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = (
    "route_a_v3_gse261709_dec025_processed_member_aggregate_discovery.v1"
)
PROTOCOL_ID = "GSE261709_DEC025_PROCESSED_MEMBER_AGGREGATE_DISCOVERY_V1"
DATASET_ID = "GSE261709"
BIOPROJECT_ID = "PRJNA1088465"
DECISION_ID = "V3-DEC-025"
DRAFT_STATUS = "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
ACTIVE_DISCOVERY_STATUS = "AUTHORIZED_AGGREGATE_DISCOVERY_ONLY_NOT_QUALIFICATION"
NOT_GRANTED = "NOT_GRANTED"
BOUND = "BOUND"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
STOP_GATE = "STOP_NOT_EVALUATED_BY_DISCOVERY"
TERMINAL_STATUS = "DISCOVERY_ONLY_STOP_NOT_QUALIFIED"

OUTER_FILENAME = "GSE261709_RAW.tar"
OUTER_BYTE_COUNT = 706560
OUTER_SHA256 = "3024746ce25f4b795daa376ac6dbafd3d53f6d30be8aed9fb14db0f118c6f434"
OUTER_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261709/suppl/"
    "GSE261709_RAW.tar"
)
MEMBERS: tuple[tuple[str, int], ...] = (
    ("GSM8149344_S1_BARCODES.txt.gz", 98593),
    ("GSM8149345_S3_BARCODES.txt.gz", 99839),
    ("GSM8149346_S5_BARCODES.txt.gz", 96359),
    ("GSM8149347_S2_BARCODES.txt.gz", 98438),
    ("GSM8149348_S4_BARCODES.txt.gz", 100030),
    ("GSM8149349_S6_BARCODES.txt.gz", 97483),
    ("GSM8149350_S7_BARCODES.txt.gz", 99818),
)

GSE269_IMPLEMENTATION_COMMIT = "99112bedf8cf7c399a772def9f34e9db6c1d5310"
GSE269_BINDING_COMMIT = "da4174f05fd026bcbf8788e0182e5cc68ffb7d1e"
DEC025_I1_COMMIT = "b64768a7cf9c789bd4a6296211e897801a899804"
DEC025_B1_COMMIT = "bfb3cc084eebbe65404b60fe81f5b8296b9b3a1f"

CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse261709_dec025_processed_member_aggregate_"
    "discovery_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/"
    "discover_gse261709_dec025_processed_member_aggregate_geometry.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/"
    "test_discover_gse261709_dec025_processed_member_aggregate_geometry.py"
)
EXACT3 = (CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH)
OUTPUT_FILENAME = "GSE261709_DEC025_PROCESSED_MEMBER_AGGREGATE_DISCOVERY_V1.json"

OWNER_APPROVAL_TEXT = (
    "APPROVE V3-DEC-025 GSE261709 PROCESSED_MEMBER_AGGREGATE_ONLY_READ: "
    "authorize one aggregate-only inspection of the exact 667648-byte official "
    "GSE261709_RAW.tar with SHA-256 "
    "3024746ce25f4b795daa376ac6dbafd3d53f6d30be8aed9fb14db0f118c6f434 "
    "and its seven frozen *_BARCODES.txt.gz members. Internal parsing may use "
    "header, barcode-token and numeric-count fields only for schema, dimensions, "
    "missingness, duplicate and cross-sample join geometry. Do not output member "
    "or barcode values and do not infer mapping, qualification, credit, canonical, "
    "split, power, training, GPU, model selection, A7 or next-phase status. Any "
    "unclosed mapping or scientific gate must remain STOP."
)
OWNER_ACTIVATION_INSTRUCTION = (
    "先激活并运行 DEC025 discovery reader，快速判断 GSE261709 是否值得继续；"
)
IDENTITY_CORRECTION = {
    "status": "BOUND_SAME_ASSET_BYTE_COUNT_TRANSCRIPTION_CORRECTION_NO_SCOPE_CHANGE",
    "original_incorrect_outer_byte_count": 667648,
    "corrected_outer_byte_count": OUTER_BYTE_COUNT,
    "frozen_outer_sha256_matches_observed": True,
    "frozen_seven_member_directory_matches_observed": True,
    "compressed_member_byte_count_total": sum(size for _, size in MEMBERS),
    "member_payload_open_count_before_correction": 0,
    "report_publication_count_before_correction": 0,
    "asset_or_access_scope_changed": False,
}

BARCODE_ALIASES = (
    "barcode",
    "barcodes",
    "barcode_id",
    "barcode_seq",
    "barcode_sequence",
    "bc",
)
MISSING_TOKENS = ("", ".", "na", "n/a", "nan")
SCIENTIFIC_GATE_IDS = (
    "BARCODE_TO_ALLELE_OR_VARIANT_MAPPING_CLOSED",
    "SOURCE_TO_CANDIDATE_RELATION_AND_EDIT_REPLAY_CLOSED",
    "FAMILY_AND_CONTEXT_STRATIFICATION_CLOSED",
    "ENDPOINT_DIRECTION_AND_SCALE_CLOSED",
    "INDEPENDENT_BIOLOGICAL_REPLICATES_AND_STANDARD_ERROR_CLOSED",
    "RIGHTS_AND_EXPOSURE_CLOSED",
    "SOURCE_GROUP_SPLIT_AND_ZERO_LEAKAGE_CLOSED",
    "POST_DEDUP_EFFECTIVE_N_AND_PREFROZEN_POWER_CLOSED",
)
REPORT_SECTIONS = (
    "asset_identity",
    "aggregate_schema_geometry",
    "aggregate_dimensions",
    "aggregate_missingness",
    "aggregate_duplicates",
    "aggregate_cross_sample_join_geometry",
    "downstream_stop_state",
)
OWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DiscoveryError(RuntimeError):
    """Base error for this fail-closed discovery protocol."""


class ProtocolError(DiscoveryError):
    """The protocol does not match its frozen interface."""


class AuthorityNotBound(DiscoveryError):
    """Owner authority or implementation binding is not active."""


class RepositoryError(DiscoveryError):
    """The exact reviewed repository lineage is not closed."""


class AssetError(DiscoveryError):
    """The exact public archive or a fixed member schema differs."""


class PublicationError(DiscoveryError):
    """The aggregate report cannot be published without replacement."""


@dataclass(frozen=True)
class MemberProfile:
    """Private in-memory member state; ``tokens`` is never serialized."""

    header_signature: tuple[str, ...]
    row_count: int
    column_count: int
    numeric_column_count: int
    missing_barcode_row_count: int
    missing_numeric_cell_count: int
    invalid_numeric_cell_count: int
    unique_token_count: int
    duplicate_token_row_count: int
    tokens: frozenset[str]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    if set(value) != set(expected):
        raise ProtocolError(f"{label} fields differ from the frozen schema")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_roots = {
        "schema_version",
        "protocol_id",
        "contract_id",
        "phase_id",
        "dataset_id",
        "bioproject_id",
        "decision_id",
        "protocol_status",
        "owner_decision",
        "implementation_binding",
        "repository_authority",
        "ordinary_public_asset",
        "identity_correction",
        "fixed_parser_contract",
        "allowed_discovery_output",
        "downstream_stop_contract",
        "frozen_outer_truth",
    }
    _exact_keys(protocol, expected_roots, label="protocol")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
    }
    for field, expected in expected_scalars.items():
        if protocol.get(field) != expected:
            raise ProtocolError(f"protocol {field} differs")

    owner = _mapping(protocol["owner_decision"], label="owner_decision")
    _exact_keys(
        owner,
        {
            "status",
            "exact_approval_text",
            "activation_instruction_exact",
            "current_candidate_may_be_activated_in_place",
            "explicit_approval_and_new_reviewed_exact3_successor_required",
            "reviewed_successor_requirement_status",
        },
        label="owner_decision",
    )
    if owner["exact_approval_text"] != OWNER_APPROVAL_TEXT:
        raise ProtocolError("owner approval text differs")
    if owner["activation_instruction_exact"] != OWNER_ACTIVATION_INSTRUCTION:
        raise ProtocolError("owner activation instruction differs")
    if owner["current_candidate_may_be_activated_in_place"] is not False:
        raise ProtocolError("current candidate must not activate in place")
    if owner["explicit_approval_and_new_reviewed_exact3_successor_required"] is not True:
        raise ProtocolError("reviewed successor requirement differs")
    if owner["reviewed_successor_requirement_status"] != (
        "SATISFIED_BY_APPEND_ONLY_REVIEWED_SUCCESSOR_CHAIN"
    ):
        raise ProtocolError("reviewed successor status differs")

    binding = _mapping(protocol["implementation_binding"], label="implementation_binding")
    _exact_keys(
        binding,
        {
            "binding_scheme",
            "current_predecessor",
            "failed_outer_identity_attempt_group",
            "candidate_group",
            "activation_rule",
        },
        label="implementation_binding",
    )
    if binding["binding_scheme"] != (
        "GSE269595_B_THEN_DEC025_I1_B1_FAILED_OUTER_BYTE_COUNT_"
        "THEN_REVIEWED_I2_CONFIG_ONLY_B2_V1"
    ):
        raise ProtocolError("binding scheme differs")
    predecessor = _mapping(binding["current_predecessor"], label="current_predecessor")
    if predecessor != {
        "status": BOUND,
        "protocol_id": "GSE269595_REPLACEMENT_A1_OR_TRUE_A2_ROLE_ADJUDICATION_PREFLIGHT_V1",
        "implementation_commit": GSE269_IMPLEMENTATION_COMMIT,
        "binding_commit": GSE269_BINDING_COMMIT,
        "binding_expected_parent": GSE269_IMPLEMENTATION_COMMIT,
        "binding_changed_paths_exactly": [
            "configs/route_a_v3_gse269595_replacement_a1_true_a2_role_"
            "adjudication_preflight_v1.json"
        ],
    }:
        raise ProtocolError("GSE269595 predecessor lineage differs")

    failed_attempt = _mapping(
        binding["failed_outer_identity_attempt_group"],
        label="failed_outer_identity_attempt_group",
    )
    if failed_attempt != {
        "status": "FAILED_CLOSED_BEFORE_MEMBER_OR_OUTPUT_IO",
        "implementation_commit": DEC025_I1_COMMIT,
        "implementation_expected_parent": GSE269_BINDING_COMMIT,
        "binding_commit": DEC025_B1_COMMIT,
        "binding_expected_parent": DEC025_I1_COMMIT,
        "implementation_changed_paths_exactly": list(EXACT3),
        "binding_changed_paths_exactly": [CONFIG_REPO_PATH],
        "terminal_reason": "OUTER_ARCHIVE_BYTE_COUNT_DIFFERS",
        "member_payload_open_count": 0,
        "report_publication_count": 0,
    }:
        raise ProtocolError("failed DEC025 outer-identity attempt lineage differs")

    candidate = _mapping(binding["candidate_group"], label="candidate_group")
    _exact_keys(
        candidate,
        {
            *OWN_BINDING_FIELDS,
            "implementation_expected_parent",
            "implementation_changed_paths_exactly",
            "binding_changed_paths_exactly",
            "unknown_to_bound_scalar_paths",
        },
        label="candidate_group",
    )
    if candidate["implementation_expected_parent"] != DEC025_B1_COMMIT:
        raise ProtocolError("candidate expected parent differs")
    if tuple(candidate["implementation_changed_paths_exactly"]) != EXACT3:
        raise ProtocolError("candidate implementation is not exact3")
    if candidate["binding_changed_paths_exactly"] != [CONFIG_REPO_PATH]:
        raise ProtocolError("candidate binding is not config-only")
    if tuple(candidate["unknown_to_bound_scalar_paths"]) != tuple(
        f"implementation_binding.candidate_group.{field}" for field in OWN_BINDING_FIELDS
    ):
        raise ProtocolError("candidate binding scalar paths differ")

    own_values = [candidate[field] for field in OWN_BINDING_FIELDS]
    if owner["status"] == NOT_GRANTED:
        if protocol["protocol_status"] != DRAFT_STATUS:
            raise ProtocolError("NOT_GRANTED protocol must remain DRAFT")
        if own_values != [UNKNOWN] * 4:
            raise ProtocolError("NOT_GRANTED candidate binding must remain wholly UNKNOWN")
    elif owner["status"] == BOUND:
        if protocol["protocol_status"] != ACTIVE_DISCOVERY_STATUS:
            raise ProtocolError("BOUND owner decision requires discovery-only active status")
        if own_values == [UNKNOWN] * 4:
            pass
        elif UNKNOWN in own_values:
            raise ProtocolError("BOUND candidate binding must not be partially bound")
        else:
            if candidate["status"] != BOUND:
                raise ProtocolError("BOUND candidate status is invalid")
            if not HEX40.fullmatch(str(candidate["implementation_commit"])):
                raise ProtocolError("candidate implementation commit is invalid")
            for field in (
                "implementation_script_sha256",
                "implementation_test_sha256",
            ):
                if not HEX64.fullmatch(str(candidate[field])):
                    raise ProtocolError(f"candidate {field} is invalid")
    else:
        raise ProtocolError("owner decision must be NOT_GRANTED or BOUND")

    repo = _mapping(protocol["repository_authority"], label="repository_authority")
    if repo != {
        "production_repo_root": "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810",
        "branch": "routea-v3-a1-20260810",
        "upstream_ref": "origin/routea-v3-a1-20260810",
        "required_state_before_archive_or_output_io": (
            "CLEAN_HEAD_EQUALS_UPSTREAM_EQUALS_LIVE_ORIGIN_AT_DEC025_BINDING_COMMIT"
        ),
    }:
        raise ProtocolError("repository authority differs")

    asset = _mapping(protocol["ordinary_public_asset"], label="ordinary_public_asset")
    if asset["canonical_url"] != OUTER_URL or asset["filename"] != OUTER_FILENAME:
        raise ProtocolError("outer asset locator differs")
    if asset["byte_count"] != OUTER_BYTE_COUNT or asset["sha256"] != OUTER_SHA256:
        raise ProtocolError("outer asset byte identity differs")
    member_pairs = tuple(
        (item.get("filename"), item.get("gzip_byte_count"))
        for item in asset["tar_members_exactly"]
        if isinstance(item, dict)
    )
    if member_pairs != MEMBERS or len(asset["tar_members_exactly"]) != len(MEMBERS):
        raise ProtocolError("frozen seven-member directory differs")
    for field in (
        "ordinary_public_only",
        "raw_fastq_sra_or_private_asset_access_allowed",
        "archive_or_member_access_before_owner_and_binding_bound",
        "network_access_allowed",
    ):
        expected = field == "ordinary_public_only"
        if asset[field] is not expected:
            raise ProtocolError(f"ordinary_public_asset.{field} differs")

    if protocol["identity_correction"] != IDENTITY_CORRECTION:
        raise ProtocolError("outer byte-count identity correction differs")

    parser = _mapping(protocol["fixed_parser_contract"], label="fixed_parser_contract")
    expected_parser = {
        "execution_model": "BUILT_IN_IN_MEMORY_ONLY_NO_CALLER_PARSER_OR_CALLBACK",
        "member_container": "GZIP",
        "member_text_encoding": "UTF-8",
        "record_delimiter": "LF",
        "field_delimiter": "TAB",
        "header_mode": "FIRST_NONEMPTY_RECORD",
        "barcode_column_detection": "EXACTLY_ONE_NORMALIZED_HEADER_IN_FIXED_ALIAS_SET",
        "barcode_header_aliases": list(BARCODE_ALIASES),
        "remaining_columns_role": "NUMERIC_COUNT",
        "missing_tokens_case_insensitive": list(MISSING_TOKENS),
        "numeric_value_rule": "FINITE_NONNEGATIVE_DECIMAL",
        "member_payload_retention_after_report": False,
        "barcode_token_retention_after_report": False,
        "persistent_member_level_intermediate_allowed": False,
    }
    if parser != expected_parser:
        raise ProtocolError("fixed parser contract differs")

    output = _mapping(protocol["allowed_discovery_output"], label="allowed_discovery_output")
    if output["sole_report_filename"] != OUTPUT_FILENAME:
        raise ProtocolError("output filename differs")
    if tuple(output["sections_exactly"]) != REPORT_SECTIONS:
        raise ProtocolError("output sections differ")
    if output["atomic_no_replace_publication_required"] is not True:
        raise ProtocolError("atomic no-replace publication is required")
    for field in (
        "member_identifier_included",
        "barcode_value_included",
        "actual_header_name_included",
        "sequence_variant_transcript_included",
        "row_effect_or_standard_error_included",
        "split_assignment_included",
        "qualifier_input_emitted",
        "qualifier_invocation_allowed_in_same_execution",
    ):
        if output[field] is not False:
            raise ProtocolError(f"allowed_discovery_output.{field} must remain false")

    stop = _mapping(protocol["downstream_stop_contract"], label="downstream_stop_contract")
    if tuple(stop["scientific_gate_ids_exactly"]) != SCIENTIFIC_GATE_IDS:
        raise ProtocolError("scientific gate set differs")
    if stop["status_for_every_scientific_gate"] != STOP_GATE:
        raise ProtocolError("scientific gate STOP state differs")
    if stop["terminal_status"] != TERMINAL_STATUS:
        raise ProtocolError("terminal status differs")
    for field in (
        "qualification_allowed",
        "credit_or_canonical_change_allowed",
        "split_or_power_execution_allowed",
        "training_or_gpu_work_allowed",
        "model_selection_or_a7_allowed",
        "next_phase_allowed",
    ):
        if stop[field] is not False:
            raise ProtocolError(f"downstream_stop_contract.{field} must remain false")

    expected_truth = {
        "current_qualified_counts": {
            "ordinary": 1,
            "a1": 1,
            "true_a2": 0,
            "canonical_records": 6547,
        },
        "gse261709_contribution": {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        },
        "qualification_run_count": 0,
        "qualifier_input_emission_count": 0,
        "qualifier_invocation_count": 0,
        "split_run_count": 0,
        "power_run_count": 0,
        "training_run_count": 0,
        "gpu_run_count": 0,
        "model_selection_run_count": 0,
        "a7_unlocked": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    if protocol["frozen_outer_truth"] != expected_truth:
        raise ProtocolError("frozen outer truth differs")


def _assert_active(protocol: Mapping[str, Any]) -> None:
    """Stop before repository, archive, member, or output-path I/O."""

    owner = protocol["owner_decision"]
    candidate = protocol["implementation_binding"]["candidate_group"]
    if protocol["protocol_status"] != ACTIVE_DISCOVERY_STATUS:
        raise AuthorityNotBound("DEC025 protocol is DRAFT_NOT_ACTIVE")
    if owner["status"] != BOUND:
        raise AuthorityNotBound("V3-DEC-025 owner decision is NOT_GRANTED")
    if candidate["status"] != BOUND:
        raise AuthorityNotBound("DEC025 implementation binding is not BOUND")


def _run_git(repo_root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", os.fspath(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RepositoryError(f"git {' '.join(args)} failed: {message}")
    return process.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    output = _run_git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in output.decode("utf-8").splitlines() if line))


def _single_parent(repo_root: Path, commit: str) -> str:
    fields = _run_git(repo_root, "rev-list", "--parents", "-n", "1", commit).decode(
        "ascii"
    ).split()
    if len(fields) != 2:
        raise RepositoryError(f"commit {commit} is not single-parent")
    return fields[1]


def _audit_repository(protocol: Mapping[str, Any]) -> dict[str, str]:
    repo_contract = protocol["repository_authority"]
    candidate = protocol["implementation_binding"]["candidate_group"]
    repo_root = Path(repo_contract["production_repo_root"])
    if _run_git(repo_root, "status", "--porcelain").strip():
        raise RepositoryError("production worktree is not clean")
    branch = _run_git(repo_root, "branch", "--show-current").decode("utf-8").strip()
    if branch != repo_contract["branch"]:
        raise RepositoryError("production branch differs")
    head = _run_git(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}").decode("ascii").strip()
    if head != upstream:
        raise RepositoryError("HEAD differs from upstream tracking ref")
    remote_line = _run_git(
        repo_root,
        "ls-remote",
        "--heads",
        "origin",
        f"refs/heads/{repo_contract['branch']}",
    ).decode("ascii").strip()
    remote_fields = remote_line.split()
    if len(remote_fields) != 2 or remote_fields[0] != head:
        raise RepositoryError("HEAD differs from live origin branch")

    implementation = str(candidate["implementation_commit"])
    if _single_parent(repo_root, DEC025_I1_COMMIT) != GSE269_BINDING_COMMIT:
        raise RepositoryError("DEC025 I1 is not the direct child of GSE269595 B")
    if _changed_paths(repo_root, DEC025_I1_COMMIT) != tuple(sorted(EXACT3)):
        raise RepositoryError("DEC025 I1 is not exact3")
    if _single_parent(repo_root, DEC025_B1_COMMIT) != DEC025_I1_COMMIT:
        raise RepositoryError("DEC025 B1 is not the direct child of I1")
    if _changed_paths(repo_root, DEC025_B1_COMMIT) != (CONFIG_REPO_PATH,):
        raise RepositoryError("DEC025 B1 is not config-only")
    if _single_parent(repo_root, implementation) != DEC025_B1_COMMIT:
        raise RepositoryError("DEC025 corrected implementation is not the direct child of B1")
    if _changed_paths(repo_root, implementation) != tuple(sorted(EXACT3)):
        raise RepositoryError("DEC025 implementation commit is not exact3")
    if _single_parent(repo_root, head) != implementation:
        raise RepositoryError("DEC025 binding is not the direct child of implementation I")
    if _changed_paths(repo_root, head) != (CONFIG_REPO_PATH,):
        raise RepositoryError("DEC025 binding commit is not config-only")

    script_blob = _run_git(repo_root, "show", f"{implementation}:{SCRIPT_REPO_PATH}")
    test_blob = _run_git(repo_root, "show", f"{implementation}:{TEST_REPO_PATH}")
    if hashlib.sha256(script_blob).hexdigest() != candidate["implementation_script_sha256"]:
        raise RepositoryError("implementation script digest differs")
    if hashlib.sha256(test_blob).hexdigest() != candidate["implementation_test_sha256"]:
        raise RepositoryError("implementation test digest differs")
    if Path(__file__).read_bytes() != script_blob:
        raise RepositoryError("executing script differs from reviewed implementation blob")
    if (repo_root / CONFIG_REPO_PATH).read_bytes() != _run_git(
        repo_root, "show", f"{head}:{CONFIG_REPO_PATH}"
    ):
        raise RepositoryError("working config differs from binding blob")
    return {
        "status": "BOUND_CLEAN_HEAD_EQUALS_UPSTREAM_EQUALS_LIVE_ORIGIN",
        "implementation_commit": implementation,
        "binding_commit": head,
        "predecessor_binding_commit": GSE269_BINDING_COMMIT,
    }


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _parse_member_payload(payload: bytes) -> MemberProfile:
    """Parse one decompressed member using the frozen in-memory TSV grammar."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssetError("member is not UTF-8") from exc
    try:
        parsed_rows = [
            row
            for row in csv.reader(io.StringIO(text), delimiter="\t", strict=True)
            if row and any(cell.strip() for cell in row)
        ]
    except csv.Error as exc:
        raise AssetError("member is not valid tab-delimited text") from exc
    if not parsed_rows:
        raise AssetError("member has no nonempty header record")
    header = tuple(_normalise_header(cell) for cell in parsed_rows[0])
    if len(header) < 2 or any(not field for field in header):
        raise AssetError("member header must contain at least two named columns")
    if len(set(header)) != len(header):
        raise AssetError("member has duplicate normalized header names")
    barcode_columns = [index for index, field in enumerate(header) if field in BARCODE_ALIASES]
    if len(barcode_columns) != 1:
        raise AssetError("member must have exactly one fixed-alias barcode column")
    barcode_index = barcode_columns[0]

    token_counts: Counter[str] = Counter()
    missing_barcode = 0
    missing_numeric = 0
    invalid_numeric = 0
    data_rows = parsed_rows[1:]
    for row in data_rows:
        if len(row) != len(header):
            raise AssetError("member row width differs from its header")
        token = row[barcode_index].strip()
        if token.lower() in MISSING_TOKENS:
            missing_barcode += 1
        else:
            token_counts[token] += 1
        for index, raw_value in enumerate(row):
            if index == barcode_index:
                continue
            value = raw_value.strip()
            if value.lower() in MISSING_TOKENS:
                missing_numeric += 1
                continue
            try:
                numeric = float(value)
            except ValueError:
                invalid_numeric += 1
                continue
            if not math.isfinite(numeric) or numeric < 0:
                invalid_numeric += 1

    duplicate_rows = sum(count - 1 for count in token_counts.values() if count > 1)
    return MemberProfile(
        header_signature=header,
        row_count=len(data_rows),
        column_count=len(header),
        numeric_column_count=len(header) - 1,
        missing_barcode_row_count=missing_barcode,
        missing_numeric_cell_count=missing_numeric,
        invalid_numeric_cell_count=invalid_numeric,
        unique_token_count=len(token_counts),
        duplicate_token_row_count=duplicate_rows,
        tokens=frozenset(token_counts),
    )


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def _aggregate_profiles(profiles: Sequence[MemberProfile]) -> dict[str, Any]:
    if len(profiles) != len(MEMBERS):
        raise AssetError("aggregate discovery requires exactly seven member profiles")
    signatures = Counter(profile.header_signature for profile in profiles)
    schema_multiplicity = _histogram(signatures.values())

    membership = Counter(token for profile in profiles for token in profile.tokens)
    pairwise_sizes = [
        len(left.tokens & right.tokens) for left, right in combinations(profiles, 2)
    ]
    all_intersection = set(profiles[0].tokens)
    for profile in profiles[1:]:
        all_intersection.intersection_update(profile.tokens)

    return {
        "aggregate_schema_geometry": {
            "member_schema_observation_count": len(profiles),
            "distinct_normalized_header_schema_count": len(signatures),
            "members_per_normalized_header_schema_histogram": schema_multiplicity,
            "same_normalized_header_schema_across_all_members": len(signatures) == 1,
            "barcode_role_column_count_per_member_histogram": {"1": len(profiles)},
            "numeric_count_column_count_per_member_histogram": _histogram(
                profile.numeric_column_count for profile in profiles
            ),
            "actual_header_name_output_count": 0,
        },
        "aggregate_dimensions": {
            "member_count": len(profiles),
            "row_count_per_member_histogram": _histogram(
                profile.row_count for profile in profiles
            ),
            "column_count_per_member_histogram": _histogram(
                profile.column_count for profile in profiles
            ),
            "total_member_row_count": sum(profile.row_count for profile in profiles),
            "unique_token_count_per_member_histogram": _histogram(
                profile.unique_token_count for profile in profiles
            ),
        },
        "aggregate_missingness": {
            "total_missing_barcode_row_count": sum(
                profile.missing_barcode_row_count for profile in profiles
            ),
            "missing_barcode_row_count_per_member_histogram": _histogram(
                profile.missing_barcode_row_count for profile in profiles
            ),
            "total_missing_numeric_cell_count": sum(
                profile.missing_numeric_cell_count for profile in profiles
            ),
            "missing_numeric_cell_count_per_member_histogram": _histogram(
                profile.missing_numeric_cell_count for profile in profiles
            ),
            "total_invalid_numeric_cell_count": sum(
                profile.invalid_numeric_cell_count for profile in profiles
            ),
            "invalid_numeric_cell_count_per_member_histogram": _histogram(
                profile.invalid_numeric_cell_count for profile in profiles
            ),
        },
        "aggregate_duplicates": {
            "total_duplicate_token_row_count": sum(
                profile.duplicate_token_row_count for profile in profiles
            ),
            "duplicate_token_row_count_per_member_histogram": _histogram(
                profile.duplicate_token_row_count for profile in profiles
            ),
        },
        "aggregate_cross_sample_join_geometry": {
            "union_token_count": len(membership),
            "all_member_intersection_token_count": len(all_intersection),
            "member_presence_cardinality_histogram": _histogram(membership.values()),
            "pairwise_comparison_count": len(pairwise_sizes),
            "pairwise_intersection_size_histogram": _histogram(pairwise_sizes),
            "member_or_barcode_value_output_count": 0,
        },
    }


def _build_report(
    protocol: Mapping[str, Any],
    profiles: Sequence[MemberProfile],
    *,
    binding: Mapping[str, str],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    aggregate = _aggregate_profiles(profiles)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": TERMINAL_STATUS,
        "binding": dict(binding),
        "owner_authority_status": protocol["owner_decision"]["status"],
        "asset_identity": {
            "outer_asset_identity_verified": True,
            "outer_archive_byte_count": OUTER_BYTE_COUNT,
            "outer_archive_sha256": OUTER_SHA256,
            "identity_correction_status": IDENTITY_CORRECTION["status"],
            "frozen_member_count": len(MEMBERS),
            "frozen_compressed_member_byte_count_total": sum(size for _, size in MEMBERS),
            "member_identifier_output_count": 0,
        },
        **aggregate,
        "downstream_stop_state": {
            "scientific_gates": {
                gate_id: STOP_GATE for gate_id in SCIENTIFIC_GATE_IDS
            },
            "qualifier_input_emission_count": 0,
            "qualifier_invocation_count": 0,
            "qualification_run_count": 0,
            "split_run_count": 0,
            "power_run_count": 0,
            "training_run_count": 0,
            "gpu_run_count": 0,
            "model_selection_run_count": 0,
            "credit_or_canonical_delta": {
                "ordinary": 0,
                "a1": 0,
                "true_a2": 0,
                "canonical_records": 0,
            },
            "a7_unlocked": False,
            "next_phase_allowed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
    }
    if tuple(section for section in REPORT_SECTIONS if section not in report):
        raise ProtocolError("report is missing a frozen aggregate section")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_archive_profiles(path: Path) -> list[MemberProfile]:
    if path.name != OUTER_FILENAME:
        raise AssetError("outer archive filename differs")
    try:
        stat = path.stat()
    except OSError as exc:
        raise AssetError("outer archive is unavailable") from exc
    if stat.st_size != OUTER_BYTE_COUNT:
        raise AssetError("outer archive byte count differs")
    if _sha256_file(path) != OUTER_SHA256:
        raise AssetError("outer archive SHA-256 differs")

    expected = sorted(MEMBERS)
    profiles: list[MemberProfile] = []
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            observed = sorted((member.name, member.size) for member in members)
            if observed != expected or any(not member.isfile() for member in members):
                raise AssetError("TAR member directory differs from the frozen exact seven")
            by_name = {member.name: member for member in members}
            for filename, expected_size in MEMBERS:
                extracted = archive.extractfile(by_name[filename])
                if extracted is None:
                    raise AssetError("frozen TAR member cannot be opened")
                compressed = extracted.read()
                if len(compressed) != expected_size:
                    raise AssetError("frozen gzip member byte count differs")
                try:
                    decompressed = gzip.decompress(compressed)
                except (OSError, EOFError) as exc:
                    raise AssetError("frozen member is not valid gzip") from exc
                profiles.append(_parse_member_payload(decompressed))
                del compressed, decompressed
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, AssetError):
            raise
        raise AssetError("outer archive is not the frozen readable TAR") from exc
    return profiles


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
        raise PublicationError("report is not finite JSON") from exc


def _publish_atomic_no_replace(output_dir: Path, report: Mapping[str, Any]) -> Path:
    """Publish one complete report without replacing an existing artifact."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PublicationError("output directory cannot be created") from exc
    final_path = output_dir / OUTPUT_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{OUTPUT_FILENAME}.", dir=output_dir, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, final_path)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise PublicationError("aggregate report already exists; refusing replacement") from exc
    except OSError as exc:
        raise PublicationError("atomic no-replace publication failed") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return final_path


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("protocol cannot be read") from exc
    protocol = _strict_json_object(payload, label="protocol")
    _validate_protocol(protocol)
    return protocol


def execute(protocol_path: Path, archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the production discovery path after exact future authority binding."""

    protocol = _load_protocol(protocol_path)
    _assert_active(protocol)
    binding = _audit_repository(protocol)
    profiles = _read_exact_archive_profiles(archive_path)
    report = _build_report(protocol, profiles, binding=binding)
    _publish_atomic_no_replace(output_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        execute(args.protocol, args.archive, args.output_dir)
    except DiscoveryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
