#!/usr/bin/env python3
"""Fail-closed DEC024 GSE261709 aggregate A1 preflight producer.

The checked-in I candidate stops before processed-asset or output I/O because
its own implementation binding is not yet frozen, member-body access authority
is not granted, and the processed member schema/formula/evidence remain unknown.
After a separately authorized successor freezes those groups, the only
production path verifies the exact Git lineage, exact ordinary-public asset
identity, frozen TSV schema, and frozen endpoint formula before parsing.
Synthetic aggregation remains permanently implementation-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = (
    "route_a_v3_gse261709_dec024_aggregate_row_level_a1_qualification_preflight.v1"
)
PROTOCOL_ID = "GSE261709_DEC024_AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_V1"
DATASET_ID = "GSE261709"
BIOPROJECT_ID = "PRJNA1088465"
DECISION_ID = "V3-DEC-024"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
PASS = "PASS_PREFLIGHT_ONLY"
BLOCKED = "BLOCKED"
SYNTHETIC_EVIDENCE = "SYNTHETIC_FIXTURE_PREFLIGHT_ONLY"
PUBLIC_PROCESSED_EVIDENCE = "ORDINARY_PUBLIC_PROCESSED_ASSET_PREFLIGHT_ONLY"
PRODUCTION_BRANCH = "routea-v3-a1-20260810"
PRODUCTION_REPO_ROOT = "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
PRODUCTION_UPSTREAM = f"origin/{PRODUCTION_BRANCH}"

CONFIG_PATH = (
    "configs/route_a_v3_gse261709_dec024_aggregate_row_level_a1_"
    "qualification_preflight_v1.json"
)
SCRIPT_PATH = (
    "scripts/route_a_v3/preflight_gse261709_dec024_aggregate_row_level_"
    "a1_qualification.py"
)
TEST_PATH = (
    "tests/route_a_v3/test_preflight_gse261709_dec024_aggregate_row_level_"
    "a1_qualification.py"
)
EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
RUNTIME_CONFIG_PATH = "configs/route_a_v3_dec024_authority_runtime_sync_v1.json"
RUNTIME_SCRIPT_PATH = "scripts/route_a_v3/dec024_authority_runtime_sync.py"
RUNTIME_TEST_PATH = "tests/route_a_v3/test_dec024_authority_runtime_sync.py"
RUNTIME_EXACT3 = (RUNTIME_CONFIG_PATH, RUNTIME_SCRIPT_PATH, RUNTIME_TEST_PATH)
AUTHORITY_COMMIT = "0bb84dffb1389b9eced7e92e36ef80b8a97ed0be"
AUTHORITY_PARENT = "e5d089a43d194caf59369fd12c203c0694ba40c6"
A6_G0_ENGINEERING_COMMIT = "8fde46ca7daa765fa3a8ad8ce24a3da82ce1a8d0"
A6_G0_ENGINEERING_EXACT4 = (
    "configs/route_a_v3_a6_learned_base_value_g0_implementation_candidate_v1.json",
    "docs/plans/2026-08-14-route-a-v3-a6-learned-g0-implementation-candidate-v1.md",
    "scripts/route_a_v3/a6_learned_base_value_g0_candidate.py",
    "tests/route_a_v3/test_a6_learned_base_value_g0_candidate.py",
)
EVT058_PROJECTION_COMMIT = "6df392e61d0d55b836c5baf84ce67f4aa9e7d1fe"
EVT058_PROJECTION_EXACT4 = (
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
EVT058_PROJECTION_BLOBS = {
    "docs/execution/route_a_v3_a1_interim.yaml": (
        "06bfbcf468e28ee27f2f02210a0cad6719cb805cb441ea40142b7b837680b44b"
    ),
    "docs/execution/route_a_v3_registry_manifest.json": (
        "fde5f7150a6dbd8b3e1caa53c69beb5ddb8b7fb9f3242bd1ffe270b165a579b9"
    ),
    "scripts/route_a_v3/validate_a0_bundle.py": (
        "4d4188f7777a2651c73b19e691be22687a73ad0da2abe8fc39f2c0e297ddc3a0"
    ),
    "tests/route_a_v3/test_a0_integrity_guards.py": (
        "aa7e5773d4353f4c8fb9a9afb6d5b9a3f3fb1ba035c6c54e6ae0d4eabac99c6b"
    ),
}
AUTHORITY_EXACT12 = (
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec024.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_a6_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "docs/execution/route_a_v3_task_registry.yaml",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
OWN_BINDING_FIELDS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
MANIFEST_FIELDS = (
    "status",
    "official_download_url",
    "filename",
    "byte_count",
    "sha256",
    "tar_member_directory_exactly",
    "member_body_access_authority_status",
    "encoding",
    "container_format",
    "tar_member_filename",
    "tar_member_byte_count",
    "tar_member_sha256",
    "tar_member_uncompressed_byte_count",
    "tar_member_uncompressed_sha256",
    "delimiter",
    "header_names_exactly",
    "row_count_excluding_header",
    "column_count",
    "field_columns_exactly",
    "endpoint_formula_id",
    "endpoint_pseudocount",
    "endpoint_primary_documentation_locator",
    "primary_measurement_route_status",
    "endpoint_formula_and_primary_documentation_status",
    "biological_replicate_sample_role_provenance_status",
    "license_and_reuse_rights_status",
    "historical_analytic_or_checkpoint_exposure_status",
    "split_readiness",
)
MEMBER_SCHEMA_FIELDS = (
    "encoding",
    "tar_member_filename",
    "tar_member_byte_count",
    "tar_member_sha256",
    "tar_member_uncompressed_byte_count",
    "tar_member_uncompressed_sha256",
    "delimiter",
    "header_names_exactly",
    "row_count_excluding_header",
    "column_count",
    "field_columns_exactly",
    "endpoint_formula_id",
    "endpoint_pseudocount",
    "endpoint_primary_documentation_locator",
    "primary_measurement_route_status",
    "endpoint_formula_and_primary_documentation_status",
    "biological_replicate_sample_role_provenance_status",
    "license_and_reuse_rights_status",
    "historical_analytic_or_checkpoint_exposure_status",
    "split_readiness",
)
ARCHIVE_DIRECTORY_BOUND = (
    "ARCHIVE_DIRECTORY_BOUND_MEMBER_BODY_AUTHORITY_AND_SCHEMA_UNKNOWN_NOT_ASSERTED"
)
ARCHIVE_MEMBER_DIRECTORY = (
    ("GSM8149344_S1_BARCODES.txt.gz", 98593),
    ("GSM8149345_S3_BARCODES.txt.gz", 99839),
    ("GSM8149346_S5_BARCODES.txt.gz", 96359),
    ("GSM8149347_S2_BARCODES.txt.gz", 98438),
    ("GSM8149348_S4_BARCODES.txt.gz", 100030),
    ("GSM8149349_S6_BARCODES.txt.gz", 97483),
    ("GSM8149350_S7_BARCODES.txt.gz", 99818),
)
ASSET_FIELD_KEYS = (
    "barcode_token",
    "allele_token",
    "transcript_token",
    "source_join_token",
    "full_construct_token",
    "source_group_token",
    "candidate_token",
    "near_duplicate_component_token",
    "source_sequence",
    "candidate_sequence",
    "construct_context",
    "cell_context",
    "biological_1_role",
    "biological_1_sample_provenance",
    "biological_1_rna_count",
    "biological_1_dna_count",
    "biological_2_role",
    "biological_2_sample_provenance",
    "biological_2_rna_count",
    "biological_2_dna_count",
    "biological_3_role",
    "biological_3_sample_provenance",
    "biological_3_rna_count",
    "biological_3_dna_count",
    "reported_effect",
    "reported_standard_error",
    "endpoint_direction",
    "missing",
    "censored",
    "qc_pass",
)
SUPPORTED_ENDPOINT_FORMULA = (
    "LOG2_RNA_PLUS_PSEUDOCOUNT_OVER_DNA_PLUS_PSEUDOCOUNT"
)
BINDING_SCALAR_PATHS = tuple(
    f"implementation_binding.{field}" for field in OWN_BINDING_FIELDS
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

GATE_IDS = (
    "PUBLIC_PROCESSED_ASSET_IDENTITY_ROLE_PROVENANCE_AND_PRIMARY_MEASUREMENT_ROUTE_CLOSED",
    "BARCODE_ALLELE_TRANSCRIPT_SOURCE_AND_FULL_CONSTRUCT_JOIN_CLOSED",
    "SOURCE_CANDIDATE_IDENTITY_AND_DENSE_FAMILY_MINIMUM_THREE_CANDIDATES_CLOSED",
    "SOURCE_TO_CANDIDATE_LEGAL_EDIT_REPLAY_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_EFFECT_AND_STANDARD_ERROR_SEMANTICS_CLOSED",
    "THREE_INDEPENDENT_BIOLOGICAL_REPLICATE_RNA_DNA_COUNTS_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_CENSORING_QC_AND_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "HISTORICAL_ANALYTIC_OR_CHECKPOINT_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_READINESS_CLOSED",
    "POST_DEDUP_INDEPENDENT_SOURCE_GROUP_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
)

RECORD_FIELDS = (
    "source_group_token",
    "candidate_token",
    "near_duplicate_component_token",
    "source_sequence",
    "candidate_sequence",
    "construct_context",
    "cell_context",
    "barcode_to_allele_join_closed",
    "allele_to_transcript_join_closed",
    "transcript_to_source_join_closed",
    "full_construct_join_closed",
    "replicate_roles",
    "replicate_sample_provenance_tokens",
    "replicate_independence_closed",
    "rna_counts",
    "dna_counts",
    "replicate_effects",
    "reported_effect",
    "reported_standard_error",
    "endpoint_direction",
    "missing",
    "censored",
    "qc_pass",
)

EVIDENCE_FIELDS = (
    "processed_asset_identity_status",
    "processed_asset_role_and_primary_measurement_route_status",
    "processed_asset_schema_role_binding_status",
    "endpoint_formula_and_primary_documentation_status",
    "biological_replicate_sample_role_provenance_status",
    "license_and_reuse_rights_status",
    "historical_analytic_or_checkpoint_exposure_status",
    "split_readiness",
)

SPLIT_FIELDS = (
    "status",
    "outcome_blind",
    "components_indivisible",
    "split_executed",
    "assignment_output_count",
    "source_group_leakage_count",
    "exact_sequence_leakage_count",
    "near_duplicate_leakage_count",
    "reverse_edge_leakage_count",
    "candidate_leakage_count",
    "study_context_leakage_count",
)

OBSERVATION_SECTIONS = (
    "processed_asset_role_schema_and_join_coverage",
    "source_family_size_histogram",
    "edit_distance_histogram",
    "replicate_coverage_histogram",
    "rna_dna_count_validity_histogram",
    "effect_direction_histogram",
    "standard_error_validity_histogram",
    "missing_censoring_qc_histogram",
    "rights_exposure_split_gate_status",
    "post_dedup_effective_n_and_power_readiness",
)

ROOT_KEYS = (
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "dataset_id",
    "bioproject_id",
    "decision_id",
    "protocol_status",
    "implementation_binding",
    "repository_authority",
    "decision_authority",
    "processed_asset_contract",
    "canonical_internal_record_interface",
    "gate_contract",
    "aggregate_evidence_interface",
    "prefrozen_power_contract",
    "output_contract",
    "frozen_outer_truth",
)

ALLOWED_INPUT_CLASSES = (
    "PUBLIC_PROCESSED_ASSET_IDENTIFIER_AND_ROLE",
    "BARCODE_TO_ALLELE_TRANSCRIPT_SOURCE_MAPPING",
    "SOURCE_AND_CANDIDATE_SEQUENCE",
    "FULL_CONSTRUCT_AND_REPORTER_CONTEXT",
    "RNA_AND_DNA_THREE_BIOLOGICAL_REPLICATE_COUNTS",
    "ENDPOINT_EFFECT_AND_STANDARD_ERROR_FIELDS",
    "MISSINGNESS_CENSORING_AND_QC_STATUS",
    "LICENSE_AND_REUSE_NOTICE",
    "NECESSARY_CONTEXT",
)

ALLOWED_INTERNAL_USES = (
    "PROCESSED_ASSET_PRIMARY_MEASUREMENT_ROUTE_AUDIT",
    "BARCODE_ALLELE_TRANSCRIPT_SOURCE_FULL_CONSTRUCT_JOIN_AUDIT",
    "SOURCE_FAMILY_AND_LEGAL_EDIT_REPLAY_AUDIT",
    "ENDPOINT_DIRECTION_SCALE_EFFECT_AND_STANDARD_ERROR_AUDIT",
    "THREE_BIOLOGICAL_REPLICATE_RNA_DNA_COUNT_AUDIT",
    "RIGHTS_AND_EXPOSURE_AUDIT",
    "SOURCE_GROUP_SPLIT_LEAKAGE_EFFECTIVE_N_AND_POWER_READINESS_AUDIT",
)


class PreflightError(RuntimeError):
    """Base fail-closed error."""


class ProtocolError(PreflightError):
    """The candidate protocol or binding shape is invalid."""


class BindingNotFrozen(ProtocolError):
    """Execution authority is incomplete."""


class ObservationError(PreflightError):
    """A canonical processed observation is malformed."""


class OutputError(PreflightError):
    """A report violates the aggregate-only output contract."""


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"cannot read protocol: {path}") from exc
    protocol = _strict_json(payload, label=str(path))
    _validate_protocol(protocol)
    return protocol


def _expect_exact_keys(value: Mapping[str, Any], keys: Iterable[str], *, label: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        raise ProtocolError(
            f"{label} keys differ: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _validate_sha_map(value: Any, paths: Sequence[str], *, label: str) -> None:
    mapping = _mapping(value, label=label)
    if set(mapping) != set(paths):
        raise ProtocolError(f"{label} paths differ")
    if any(not isinstance(item, str) or not HEX64.fullmatch(item) for item in mapping.values()):
        raise ProtocolError(f"{label} contains an invalid SHA-256")


def _runtime_mode(runtime: Mapping[str, Any]) -> str:
    predecessor = _mapping(
        runtime.get("mandatory_non_authoritative_predecessor"),
        label="mandatory A6 G0 engineering predecessor",
    )
    if predecessor != {
        "status": "BOUND_ENGINEERING_ONLY_NO_AUTHORITY_OR_SCIENCE_CHANGE",
        "commit": A6_G0_ENGINEERING_COMMIT,
        "expected_parent": AUTHORITY_COMMIT,
        "exact_changed_paths": list(A6_G0_ENGINEERING_EXACT4),
        "changes_dec024_authority": False,
        "changes_scientific_state": False,
    }:
        raise ProtocolError("mandatory A6 G0 engineering predecessor differs")
    if tuple(runtime.get("paths", ())) != RUNTIME_EXACT3:
        raise ProtocolError("authority-runtime exact3 paths differ")
    if runtime.get("implementation_expected_parent") != A6_G0_ENGINEERING_COMMIT:
        raise ProtocolError("authority-runtime implementation parent differs")
    if tuple(runtime.get("implementation_exact_changed_paths", ())) != RUNTIME_EXACT3:
        raise ProtocolError("authority-runtime implementation changed paths differ")
    if runtime.get("binding_exact_changed_paths") != [RUNTIME_CONFIG_PATH]:
        raise ProtocolError("authority-runtime binding must be config-only")
    grouped = (
        runtime.get("implementation_commit"),
        runtime.get("implementation_blob_sha256_by_path"),
        runtime.get("binding_commit"),
        runtime.get("binding_expected_parent"),
        runtime.get("binding_blob_sha256_by_path"),
    )
    if runtime.get("status") == UNKNOWN:
        if grouped != (UNKNOWN,) * 5:
            raise ProtocolError("partial UNKNOWN authority-runtime group is forbidden")
        return UNKNOWN
    if runtime.get("status") != BOUND:
        raise ProtocolError("authority-runtime status must be grouped UNKNOWN or BOUND")
    implementation = runtime.get("implementation_commit")
    binding = runtime.get("binding_commit")
    if not isinstance(implementation, str) or not HEX40.fullmatch(implementation):
        raise ProtocolError("authority-runtime implementation commit is invalid")
    if not isinstance(binding, str) or not HEX40.fullmatch(binding):
        raise ProtocolError("authority-runtime binding commit is invalid")
    if runtime.get("binding_expected_parent") != implementation:
        raise ProtocolError("authority-runtime binding parent differs")
    _validate_sha_map(runtime.get("implementation_blob_sha256_by_path"), RUNTIME_EXACT3, label="runtime I blobs")
    _validate_sha_map(runtime.get("binding_blob_sha256_by_path"), RUNTIME_EXACT3, label="runtime B blobs")
    return BOUND


def _own_binding_mode(binding: Mapping[str, Any], *, runtime_mode: str) -> str:
    values = tuple(binding.get(field) for field in OWN_BINDING_FIELDS)
    if binding.get("status") == UNKNOWN:
        if values != (UNKNOWN,) * 4:
            raise ProtocolError("partial UNKNOWN exact3 implementation group is forbidden")
        return UNKNOWN
    if binding.get("status") != BOUND:
        raise ProtocolError("exact3 implementation status must be grouped UNKNOWN or BOUND")
    if runtime_mode != BOUND:
        raise ProtocolError("BOUND exact3 implementation requires BOUND authority-runtime")
    if not isinstance(binding.get("implementation_commit"), str) or not HEX40.fullmatch(
        str(binding["implementation_commit"])
    ):
        raise ProtocolError("exact3 implementation commit is invalid")
    for field in ("implementation_script_sha256", "implementation_test_sha256"):
        if not isinstance(binding.get(field), str) or not HEX64.fullmatch(str(binding[field])):
            raise ProtocolError(f"{field} is invalid")
    return BOUND


def _manifest_mode(manifest: Mapping[str, Any], *, forbidden_suffixes: Sequence[str]) -> str:
    _expect_exact_keys(manifest, MANIFEST_FIELDS, label="official processed-asset manifest")
    expected_directory = [
        {"filename": filename, "gzip_byte_count": byte_count}
        for filename, byte_count in ARCHIVE_MEMBER_DIRECTORY
    ]
    expected_outer = {
        "official_download_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261709/"
            "suppl/GSE261709_RAW.tar"
        ),
        "filename": "GSE261709_RAW.tar",
        "byte_count": 667648,
        "sha256": "3024746ce25f4b795daa376ac6dbafd3d53f6d30be8aed9fb14db0f118c6f434",
        "tar_member_directory_exactly": expected_directory,
        "container_format": "TAR_DIRECTORY_BOUND_SEVEN_GZIP_TXT_MEMBERS",
    }
    for field, expected in expected_outer.items():
        if manifest.get(field) != expected:
            raise ProtocolError(f"official processed-archive identity differs: {field}")

    if manifest.get("status") == ARCHIVE_DIRECTORY_BOUND:
        if manifest.get("member_body_access_authority_status") != (
            "EXPLICIT_USER_AUTHORITY_REQUIRED_NOT_GRANTED"
        ):
            raise ProtocolError("processed member-body authority must remain not granted")
        if tuple(manifest.get(field) for field in MEMBER_SCHEMA_FIELDS) != (
            UNKNOWN,
        ) * len(MEMBER_SCHEMA_FIELDS):
            raise ProtocolError("partial processed member schema is forbidden")
        return ARCHIVE_DIRECTORY_BOUND
    if manifest.get("status") != BOUND:
        raise ProtocolError("official processed-asset manifest status is invalid")
    if manifest.get("member_body_access_authority_status") != "EXPLICIT_USER_AUTHORITY_GRANTED":
        raise ProtocolError("BOUND member schema requires explicit member-body authority")

    if manifest.get("encoding") != "UTF-8":
        raise ProtocolError("processed asset encoding must be UTF-8")
    member_filename = manifest.get("tar_member_filename")
    directory = dict(ARCHIVE_MEMBER_DIRECTORY)
    if (
        not isinstance(member_filename, str)
        or not member_filename.lower().endswith(".txt.gz")
        or Path(member_filename).name != member_filename
        or member_filename not in directory
    ):
        raise ProtocolError("processed TAR member filename is invalid or raw/archive")
    if type(manifest.get("tar_member_byte_count")) is not int or manifest[
        "tar_member_byte_count"
    ] != directory[member_filename]:
        raise ProtocolError("processed TAR member byte count is invalid")
    if not isinstance(manifest.get("tar_member_sha256"), str) or not HEX64.fullmatch(
        manifest["tar_member_sha256"]
    ):
        raise ProtocolError("processed TAR member SHA-256 is invalid")
    if type(manifest.get("tar_member_uncompressed_byte_count")) is not int or manifest[
        "tar_member_uncompressed_byte_count"
    ] <= 0:
        raise ProtocolError("processed TXT member uncompressed byte count is invalid")
    if not isinstance(
        manifest.get("tar_member_uncompressed_sha256"), str
    ) or not HEX64.fullmatch(manifest["tar_member_uncompressed_sha256"]):
        raise ProtocolError("processed TXT member uncompressed SHA-256 is invalid")
    if manifest.get("delimiter") != "TAB":
        raise ProtocolError("processed asset delimiter must be TAB")

    headers = manifest.get("header_names_exactly")
    if (
        not isinstance(headers, list)
        or not headers
        or any(not isinstance(value, str) or not value for value in headers)
        or len(set(headers)) != len(headers)
    ):
        raise ProtocolError("official processed-asset header is invalid")
    if type(manifest.get("column_count")) is not int or manifest["column_count"] != len(headers):
        raise ProtocolError("official processed-asset column count differs from header")
    if type(manifest.get("row_count_excluding_header")) is not int or manifest[
        "row_count_excluding_header"
    ] <= 0:
        raise ProtocolError("official processed-asset row count is invalid")
    columns = _mapping(manifest.get("field_columns_exactly"), label="asset field columns")
    _expect_exact_keys(columns, ASSET_FIELD_KEYS, label="asset field columns")
    if (
        any(not isinstance(value, str) or value not in headers for value in columns.values())
        or len(set(columns.values())) != len(columns)
    ):
        raise ProtocolError("asset field columns are missing, aliased, or outside the header")

    if manifest.get("endpoint_formula_id") != SUPPORTED_ENDPOINT_FORMULA:
        raise ProtocolError("endpoint formula is not the frozen supported formula")
    pseudocount = manifest.get("endpoint_pseudocount")
    if not _finite_number(pseudocount) or float(pseudocount) <= 0.0:
        raise ProtocolError("endpoint pseudocount is invalid")
    locator = manifest.get("endpoint_primary_documentation_locator")
    if not isinstance(locator, str) or not locator:
        raise ProtocolError("endpoint primary-documentation locator is missing")
    for field in (
        "primary_measurement_route_status",
        "endpoint_formula_and_primary_documentation_status",
        "biological_replicate_sample_role_provenance_status",
        "license_and_reuse_rights_status",
        "historical_analytic_or_checkpoint_exposure_status",
    ):
        if manifest.get(field) not in {PASS, BLOCKED, UNKNOWN}:
            raise ProtocolError(f"manifest evidence status is invalid: {field}")
    split = _mapping(manifest.get("split_readiness"), label="manifest split readiness")
    _expect_exact_keys(split, SPLIT_FIELDS, label="manifest split readiness")
    if split.get("status") not in {PASS, BLOCKED, UNKNOWN}:
        raise ProtocolError("manifest split status is invalid")
    for field in ("outcome_blind", "components_indivisible", "split_executed"):
        if type(split.get(field)) is not bool:
            raise ProtocolError(f"manifest split boolean is invalid: {field}")
    for field in (
        "assignment_output_count",
        "source_group_leakage_count",
        "exact_sequence_leakage_count",
        "near_duplicate_leakage_count",
        "reverse_edge_leakage_count",
        "candidate_leakage_count",
        "study_context_leakage_count",
    ):
        if type(split.get(field)) is not int or split[field] < 0:
            raise ProtocolError(f"manifest split count is invalid: {field}")
    return BOUND


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if set(protocol) != set(ROOT_KEYS):
        raise ProtocolError("protocol root closure differs")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "protocol_status": "LOCAL_EXACT3_I_CANDIDATE_RUNTIME_BOUND_OWN_BINDING_AND_MEMBER_BODY_AUTHORITY_SCHEMA_PENDING_NOT_QUALIFICATION",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} differs")

    binding = _mapping(protocol.get("implementation_binding"), label="implementation binding")
    authority = _mapping(binding.get("authority_group"), label="authority group")
    if authority.get("status") != BOUND:
        raise ProtocolError("DEC024 authority A must be BOUND")
    if authority.get("authority_commit") != AUTHORITY_COMMIT:
        raise ProtocolError("DEC024 authority commit differs")
    if authority.get("authority_expected_parent") != AUTHORITY_PARENT:
        raise ProtocolError("DEC024 authority parent differs")
    if tuple(authority.get("authority_exact_changed_paths", ())) != AUTHORITY_EXACT12:
        raise ProtocolError("DEC024 authority exact12 differs")
    runtime = _mapping(authority.get("authority_runtime_lifecycle"), label="authority runtime")
    runtime_mode = _runtime_mode(runtime)
    projection = _mapping(
        binding.get("mandatory_non_science_current_projection_predecessor"),
        label="EVT058 current projection predecessor",
    )
    if projection != {
        "status": "BOUND_RUNTIME_PROJECTION_ONLY_NO_AUTHORITY_OR_SCIENCE_CHANGE",
        "commit": EVT058_PROJECTION_COMMIT,
        "expected_parent_source": (
            "implementation_binding.authority_group.authority_runtime_lifecycle.binding_commit"
        ),
        "exact_changed_paths": list(EVT058_PROJECTION_EXACT4),
        "blob_sha256_by_path": EVT058_PROJECTION_BLOBS,
        "changes_dec024_authority": False,
        "changes_scientific_state": False,
    }:
        raise ProtocolError("EVT058 current projection predecessor differs")
    own_mode = _own_binding_mode(binding, runtime_mode=runtime_mode)
    if binding.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("implementation script path differs")
    if binding.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("implementation test path differs")
    if tuple(binding.get("implementation_commit_exact_changed_paths", ())) != EXACT3:
        raise ProtocolError("implementation commit must be exact3")
    if binding.get("binding_commit_exact_changed_paths") != [CONFIG_PATH]:
        raise ProtocolError("implementation binding must be config-only")
    if tuple(binding.get("unknown_to_bound_scalar_paths", ())) != BINDING_SCALAR_PATHS:
        raise ProtocolError("implementation binding scalar paths differ")

    repository = _mapping(protocol.get("repository_authority"), label="repository authority")
    if repository != {
        "production_repo_root": PRODUCTION_REPO_ROOT,
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": PRODUCTION_UPSTREAM,
        "require_clean_worktree": True,
        "require_head_upstream_origin_match": True,
    }:
        raise ProtocolError("production repository authority differs")

    decision = _mapping(protocol.get("decision_authority"), label="decision authority")
    if decision.get("authorized_role") != "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY":
        raise ProtocolError("DEC024 authorized role differs")
    if decision.get("authority_surface") != "ORDINARY_PUBLIC_PROCESSED_ASSET_ONLY":
        raise ProtocolError("authority surface differs")
    if tuple(decision.get("allowed_internal_input_field_classes_exactly", ())) != ALLOWED_INPUT_CLASSES:
        raise ProtocolError("allowed internal input classes differ")
    if tuple(decision.get("allowed_internal_uses_exactly", ())) != ALLOWED_INTERNAL_USES:
        raise ProtocolError("allowed internal uses differ")
    if decision.get("allowed_output_class") != "AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_ONLY":
        raise ProtocolError("output class differs")
    if decision.get("all_required_gates_passing_automatically_qualifies_dataset") is not False:
        raise ProtocolError("preflight may not auto-qualify")
    if decision.get("separate_user_authority_required_for_qualification_or_counting") is not True:
        raise ProtocolError("separate qualification/counting authority must remain required")

    asset = _mapping(protocol.get("processed_asset_contract"), label="processed asset contract")
    for field in (
        "raw_fastq_or_sra_member_payload_read_allowed",
        "raw_archive_run_member_open_allowed",
        "network_download_allowed_by_this_candidate",
    ):
        if asset.get(field) is not False:
            raise ProtocolError(f"processed asset boundary differs: {field}")
    if asset.get("processed_public_asset_body_read_allowed_after_all_bindings") is not True:
        raise ProtocolError("processed body access boundary differs")
    if (
        asset.get(
            "official_manifest_bound_processed_tar_txt_member_read_allowed_after_all_bindings"
        )
        is not True
    ):
        raise ProtocolError("manifest-bound processed TAR/TXT access boundary differs")
    if asset.get("synthetic_fixture_execution_is_scientific_evidence") is not False:
        raise ProtocolError("synthetic fixtures may not become scientific evidence")
    if asset.get("unknown_group_action") != "STOP_BEFORE_PROCESSED_ASSET_BODY_OR_OUTPUT_IO":
        raise ProtocolError("unknown asset action differs")
    manifest = _mapping(
        asset.get("official_processed_asset_manifest"),
        label="official processed-asset manifest",
    )
    _manifest_mode(
        manifest,
        forbidden_suffixes=tuple(asset.get("forbidden_filename_suffixes", ())),
    )

    record = _mapping(protocol.get("canonical_internal_record_interface"), label="record interface")
    if tuple(record.get("fields_exactly", ())) != RECORD_FIELDS:
        raise ProtocolError("canonical record fields differ")
    if record.get("minimum_distinct_candidates_per_source_family") != 3:
        raise ProtocolError("minimum source-family size differs")
    if record.get("biological_replicate_count") != 3:
        raise ProtocolError("biological replicate count differs")
    if record.get("required_replicate_roles") != ["BIOLOGICAL_1", "BIOLOGICAL_2", "BIOLOGICAL_3"]:
        raise ProtocolError("biological replicate roles differ")
    if record.get("technical_units_may_substitute_for_biological_replicates") is not False:
        raise ProtocolError("technical units cannot substitute")
    if record.get("source_to_candidate_edit_relation_may_be_presumed") is not False:
        raise ProtocolError("edit relation may not be presumed")
    if record.get("barcode_to_source_join_may_be_inferred_from_row_order") is not False:
        raise ProtocolError("barcode join may not use row order")
    if record.get("replicate_effect_recompute") != SUPPORTED_ENDPOINT_FORMULA:
        raise ProtocolError("replicate endpoint formula differs")
    if record.get("reported_effect_recompute") != (
        "MEAN_OF_THREE_RECOMPUTED_PAIRED_BIOLOGICAL_REPLICATE_EFFECTS"
    ):
        raise ProtocolError("reported endpoint recompute differs")

    gates = _mapping(protocol.get("gate_contract"), label="gate contract")
    if tuple(gates.get("gate_ids_exactly", ())) != GATE_IDS:
        raise ProtocolError("DEC024 gate IDs differ")
    if gates.get("allowed_statuses") != [PASS, BLOCKED, UNKNOWN]:
        raise ProtocolError("gate statuses differ")
    if gates.get("unknown_or_not_run_gate_is_pass") is not False:
        raise ProtocolError("UNKNOWN may not pass")
    if (
        gates.get("synthetic_all_pass_action")
        != "SYNTHETIC_IMPLEMENTATION_EXERCISE_ONLY_NOT_DATA_PREFLIGHT"
    ):
        raise ProtocolError("synthetic all-pass action differs")
    evidence = _mapping(protocol.get("aggregate_evidence_interface"), label="evidence interface")
    if tuple(evidence.get("fields_exactly", ())) != EVIDENCE_FIELDS:
        raise ProtocolError("evidence fields differ")
    if tuple(evidence.get("split_readiness_fields_exactly", ())) != SPLIT_FIELDS:
        raise ProtocolError("split evidence fields differ")

    power = _mapping(protocol.get("prefrozen_power_contract"), label="power")
    if power.get("analysis_unit") != "POST_DEDUP_INDEPENDENT_SOURCE_GROUP":
        raise ProtocolError("power analysis unit differs")
    if power.get("alternative_spearman_rho") != 0.25:
        raise ProtocolError("power alternative differs")
    if power.get("target_power_minimum") != 0.8 or power.get("maximum_full_ci_width") != 0.3:
        raise ProtocolError("power threshold differs")
    if power.get("required_effective_n_for_power_and_ci_width") != 156:
        raise ProtocolError("prefrozen required effective N differs")
    if power.get("formal_qualification_power_gate_execution_allowed") is not False:
        raise ProtocolError("formal qualification power execution is forbidden")

    output = _mapping(protocol.get("output_contract"), label="output")
    if tuple(output.get("allowed_aggregate_sections_exactly", ())) != OBSERVATION_SECTIONS:
        raise ProtocolError("aggregate output sections differ")
    if output.get("single_aggregate_output_only") is not True:
        raise ProtocolError("single aggregate output must remain required")
    if output.get("atomic_no_replace_publication_required") is not True:
        raise ProtocolError("atomic no-replace publication must remain required")
    if output.get("filename") != (
        "GSE261709_DEC024_AGGREGATE_ROW_LEVEL_A1_QUALIFICATION_PREFLIGHT_V1.json"
    ):
        raise ProtocolError("aggregate output filename differs")
    for field, value in output.items():
        if field.endswith("_included") and value is not False:
            raise ProtocolError(f"forbidden output became allowed: {field}")
    outer = _mapping(protocol.get("frozen_outer_truth"), label="outer truth")
    if outer.get("current_qualified_counts") != {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }:
        raise ProtocolError("current qualified counts differ")
    if outer.get("gse261709_contribution") != {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }:
        raise ProtocolError("GSE261709 contribution differs")
    for field in (
        "gse261709_qualified",
        "gse261709_a1_credit_established",
        "training_allowed",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_unlocked",
        "next_phase_authorized",
    ):
        if outer.get(field) is not False:
            raise ProtocolError(f"outer truth drifted: {field}")
    for field in (
        "qualification_run_count",
        "canonical_materialization_run_count",
        "training_run_count",
        "gpu_work_run_count",
        "model_selection_run_count",
    ):
        if outer.get(field) != 0:
            raise ProtocolError(f"outer run count drifted: {field}")


def _binding_modes(protocol: Mapping[str, Any]) -> tuple[str, str, str]:
    binding = protocol["implementation_binding"]
    runtime_mode = _runtime_mode(binding["authority_group"]["authority_runtime_lifecycle"])
    own_mode = _own_binding_mode(binding, runtime_mode=runtime_mode)
    asset = protocol["processed_asset_contract"]
    manifest_mode = _manifest_mode(
        asset["official_processed_asset_manifest"],
        forbidden_suffixes=tuple(asset["forbidden_filename_suffixes"]),
    )
    return runtime_mode, own_mode, manifest_mode


def _ensure_ready_before_asset_or_output_io(protocol: Mapping[str, Any]) -> None:
    runtime_mode, own_mode, manifest_mode = _binding_modes(protocol)
    if runtime_mode != BOUND:
        raise BindingNotFrozen(
            "DEC024 authority-runtime binding group is UNKNOWN_NOT_ASSERTED; stopped before asset/output I/O"
        )
    if own_mode != BOUND:
        raise BindingNotFrozen(
            "GSE261709 exact3 implementation binding is UNKNOWN_NOT_ASSERTED; stopped before asset/output I/O"
        )
    if manifest_mode != BOUND:
        raise BindingNotFrozen(
            "official processed-asset manifest is UNKNOWN_NOT_ASSERTED; stopped before asset/output I/O"
        )


def validate_processed_asset_path(path: Path, protocol: Mapping[str, Any]) -> None:
    manifest = protocol["processed_asset_contract"]["official_processed_asset_manifest"]
    if manifest["status"] != BOUND:
        raise BindingNotFrozen("official processed-asset manifest is not BOUND")
    forbidden = tuple(protocol["processed_asset_contract"]["forbidden_filename_suffixes"])
    lowered = path.name.lower()
    if path.name != manifest["filename"]:
        raise ProtocolError("processed asset filename differs from the frozen official manifest")
    if any(lowered.endswith(suffix) for suffix in forbidden) and not (
        path.name == "GSE261709_RAW.tar"
        and manifest["container_format"] == "TAR_DIRECTORY_BOUND_SEVEN_GZIP_TXT_MEMBERS"
    ):
        raise ProtocolError("raw FASTQ/SRA/archive member input is forbidden")


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for repository binding audit") from exc
    if result.returncode != 0:
        raise ProtocolError("git repository binding audit failed")
    return result.stdout.strip()


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for blob binding audit") from exc
    if result.returncode != 0:
        raise ProtocolError(f"cannot read bound Git blob: {path}")
    return result.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    value = _run_git(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in value.splitlines() if line))


def _verify_frozen_commit(
    repo_root: Path,
    *,
    label: str,
    commit: str,
    expected_parent: str,
    expected_paths: Sequence[str],
    expected_blobs: Mapping[str, str] | None = None,
) -> None:
    if _run_git(repo_root, "rev-parse", f"{commit}^") != expected_parent:
        raise ProtocolError(f"{label} parent differs")
    if _changed_paths(repo_root, commit) != tuple(sorted(expected_paths)):
        raise ProtocolError(f"{label} changed-path closure differs")
    for path, expected_sha in (expected_blobs or {}).items():
        if hashlib.sha256(_git_blob(repo_root, commit, path)).hexdigest() != expected_sha:
            raise ProtocolError(f"{label} blob identity differs: {path}")


def _normalise_binding_and_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    normalised = copy.deepcopy(dict(protocol))
    binding = normalised["implementation_binding"]
    for field in OWN_BINDING_FIELDS:
        binding[field] = UNKNOWN
    return normalised


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> dict[str, str]:
    """Verify 0bb -> 8fde -> runtime I/B -> EVT058 P -> preflight I/B."""

    _ensure_ready_before_asset_or_output_io(protocol)
    repository = protocol["repository_authority"]
    if repo_root.resolve() != Path(repository["production_repo_root"]).resolve():
        raise ProtocolError("execution repository is not the frozen production root")
    head = _run_git(repo_root, "rev-parse", "HEAD")
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}")
    origin = _run_git(
        repo_root,
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{repository['branch']}",
    )
    if head != upstream or head != origin:
        raise ProtocolError("HEAD, upstream, and local origin ref do not match")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != repository["branch"]:
        raise ProtocolError("production branch differs")
    if _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}") != repository[
        "upstream_ref"
    ]:
        raise ProtocolError("production upstream differs")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProtocolError("production worktree or index is dirty")

    binding = protocol["implementation_binding"]
    authority = binding["authority_group"]
    runtime = authority["authority_runtime_lifecycle"]
    runtime_i = str(runtime["implementation_commit"])
    runtime_b = str(runtime["binding_commit"])
    implementation_i = str(binding["implementation_commit"])

    _verify_frozen_commit(
        repo_root,
        label="DEC024 authority A",
        commit=AUTHORITY_COMMIT,
        expected_parent=AUTHORITY_PARENT,
        expected_paths=AUTHORITY_EXACT12,
    )
    _verify_frozen_commit(
        repo_root,
        label="A6 G0 non-authoritative predecessor",
        commit=A6_G0_ENGINEERING_COMMIT,
        expected_parent=AUTHORITY_COMMIT,
        expected_paths=A6_G0_ENGINEERING_EXACT4,
    )
    _verify_frozen_commit(
        repo_root,
        label="DEC024 authority-runtime I",
        commit=runtime_i,
        expected_parent=A6_G0_ENGINEERING_COMMIT,
        expected_paths=RUNTIME_EXACT3,
        expected_blobs=runtime["implementation_blob_sha256_by_path"],
    )
    _verify_frozen_commit(
        repo_root,
        label="DEC024 authority-runtime B",
        commit=runtime_b,
        expected_parent=runtime_i,
        expected_paths=(RUNTIME_CONFIG_PATH,),
        expected_blobs=runtime["binding_blob_sha256_by_path"],
    )
    _verify_frozen_commit(
        repo_root,
        label="EVT058 non-science current projection P",
        commit=EVT058_PROJECTION_COMMIT,
        expected_parent=runtime_b,
        expected_paths=EVT058_PROJECTION_EXACT4,
        expected_blobs=EVT058_PROJECTION_BLOBS,
    )
    _verify_frozen_commit(
        repo_root,
        label="GSE261709 preflight I",
        commit=implementation_i,
        expected_parent=EVT058_PROJECTION_COMMIT,
        expected_paths=EXACT3,
    )
    _verify_frozen_commit(
        repo_root,
        label="GSE261709 preflight B",
        commit=head,
        expected_parent=implementation_i,
        expected_paths=(CONFIG_PATH,),
    )

    implementation_protocol = _strict_json(
        _git_blob(repo_root, implementation_i, CONFIG_PATH),
        label="preflight I protocol",
    )
    if _normalise_binding_and_manifest(protocol) != implementation_protocol:
        raise ProtocolError("preflight B changed fields outside the frozen four-scalar binding group")
    script_blob = _git_blob(repo_root, implementation_i, SCRIPT_PATH)
    test_blob = _git_blob(repo_root, implementation_i, TEST_PATH)
    if hashlib.sha256(script_blob).hexdigest() != binding["implementation_script_sha256"]:
        raise ProtocolError("bound implementation script identity differs")
    if hashlib.sha256(test_blob).hexdigest() != binding["implementation_test_sha256"]:
        raise ProtocolError("bound implementation test identity differs")
    if protocol_path.resolve() != (repo_root / CONFIG_PATH).resolve():
        raise ProtocolError("protocol path is outside the frozen repository location")
    if protocol_path.read_bytes() != _git_blob(repo_root, head, CONFIG_PATH):
        raise ProtocolError("working protocol differs from preflight B")
    executing_script = Path(__file__).resolve()
    if executing_script != (repo_root / SCRIPT_PATH).resolve():
        raise ProtocolError("executing producer is not the bound repository script")
    if executing_script.read_bytes() != script_blob:
        raise ProtocolError("executing producer differs from preflight I")
    if (repo_root / TEST_PATH).read_bytes() != test_blob:
        raise ProtocolError("working focused test differs from preflight I")
    return {
        "status": PASS,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_runtime_binding_commit": runtime_b,
        "current_projection_commit": EVT058_PROJECTION_COMMIT,
        "implementation_commit": implementation_i,
        "binding_commit": head,
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _parse_finite_number(value: str, *, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ObservationError(f"row {row_number} has invalid numeric field: {field}") from exc
    if not math.isfinite(result):
        raise ObservationError(f"row {row_number} has non-finite numeric field: {field}")
    return result


def _parse_boolean(value: str, *, field: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ObservationError(f"row {row_number} has invalid boolean field: {field}")


def _decoded_tsv_payload(payload: bytes, manifest: Mapping[str, Any]) -> str:
    """Read one approved exact GZIP/TXT member in memory; never extract paths."""

    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            observed_directory = [(member.name, member.size) for member in members]
            if observed_directory != list(ARCHIVE_MEMBER_DIRECTORY) or any(
                not member.isfile() for member in members
            ):
                raise ObservationError("official processed TAR directory differs from frozen seven-member listing")
            member = next(
                item for item in members if item.name == manifest["tar_member_filename"]
            )
            if (
                member.size != manifest["tar_member_byte_count"]
            ):
                raise ObservationError("processed TAR member identity or type differs from manifest")
            handle = archive.extractfile(member)
            if handle is None:
                raise ObservationError("cannot read the bound processed TXT member")
            compressed = handle.read()
    except (StopIteration, tarfile.TarError, OSError, EOFError) as exc:
        raise ObservationError("processed asset is not the frozen TAR/TXT container") from exc
    if len(compressed) != manifest["tar_member_byte_count"]:
        raise ObservationError("processed GZIP/TXT member byte count differs from manifest")
    if hashlib.sha256(compressed).hexdigest() != manifest["tar_member_sha256"]:
        raise ObservationError("processed GZIP/TXT member digest differs from manifest")
    try:
        decoded = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ObservationError("processed TAR member is not the frozen GZIP/TXT payload") from exc
    if len(decoded) != manifest["tar_member_uncompressed_byte_count"]:
        raise ObservationError("processed TXT member uncompressed byte count differs")
    if hashlib.sha256(decoded).hexdigest() != manifest["tar_member_uncompressed_sha256"]:
        raise ObservationError("processed TXT member uncompressed digest differs")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ObservationError("processed asset is not strict UTF-8") from exc


def read_bound_processed_asset(
    processed_asset_path: Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the one manifest-bound processed TSV; no network or archive path exists."""

    _ensure_ready_before_asset_or_output_io(protocol)
    validate_processed_asset_path(processed_asset_path, protocol)
    manifest = protocol["processed_asset_contract"]["official_processed_asset_manifest"]
    try:
        payload = processed_asset_path.read_bytes()
    except OSError as exc:
        raise ObservationError("cannot read the frozen ordinary-public processed asset") from exc
    if len(payload) != manifest["byte_count"]:
        raise ObservationError("processed asset byte count differs from frozen manifest")
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ObservationError("processed asset digest differs from frozen manifest")

    text = _decoded_tsv_payload(payload, manifest)
    return _parse_bound_tsv_text(text, manifest, protocol)


def _parse_bound_tsv_text(
    text: str,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse one already identity-verified member using its exact frozen schema."""

    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True))
    except csv.Error as exc:
        raise ObservationError("processed asset is not a valid frozen TSV") from exc
    if not rows:
        raise ObservationError("processed asset is empty")
    headers = rows[0]
    if headers != manifest["header_names_exactly"]:
        raise ObservationError("processed asset header differs from frozen manifest")
    if len(headers) != manifest["column_count"]:
        raise ObservationError("processed asset column dimension differs")
    body = rows[1:]
    if len(body) != manifest["row_count_excluding_header"]:
        raise ObservationError("processed asset row dimension differs")
    if any(len(row) != len(headers) for row in body):
        raise ObservationError("processed asset contains a ragged row")

    header_index = {name: index for index, name in enumerate(headers)}
    field_columns = manifest["field_columns_exactly"]
    pseudocount = float(manifest["endpoint_pseudocount"])
    required_roles = protocol["canonical_internal_record_interface"]["required_replicate_roles"]
    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(body, start=2):
        def value(field: str) -> str:
            return row[header_index[field_columns[field]]]

        roles = [value(f"biological_{index}_role") for index in range(1, 4)]
        provenance = [
            value(f"biological_{index}_sample_provenance") for index in range(1, 4)
        ]
        rna = [
            _parse_finite_number(
                value(f"biological_{index}_rna_count"),
                field=f"biological_{index}_rna_count",
                row_number=row_number,
            )
            for index in range(1, 4)
        ]
        dna = [
            _parse_finite_number(
                value(f"biological_{index}_dna_count"),
                field=f"biological_{index}_dna_count",
                row_number=row_number,
            )
            for index in range(1, 4)
        ]
        effects = [
            _replicate_effect(left, right, pseudocount)
            for left, right in zip(rna, dna)
        ]
        join_values = [
            value("barcode_token"),
            value("allele_token"),
            value("transcript_token"),
            value("source_join_token"),
            value("full_construct_token"),
        ]
        records.append(
            {
                "source_group_token": value("source_group_token"),
                "candidate_token": value("candidate_token"),
                "near_duplicate_component_token": value("near_duplicate_component_token"),
                "source_sequence": value("source_sequence"),
                "candidate_sequence": value("candidate_sequence"),
                "construct_context": value("construct_context"),
                "cell_context": value("cell_context"),
                "barcode_to_allele_join_closed": all(join_values[:2]),
                "allele_to_transcript_join_closed": all(join_values[1:3]),
                "transcript_to_source_join_closed": all(join_values[2:4]),
                "full_construct_join_closed": all(join_values[3:5]),
                "replicate_roles": roles,
                "replicate_sample_provenance_tokens": provenance,
                "replicate_independence_closed": (
                    roles == required_roles
                    and all(provenance)
                    and len(set(provenance)) == 3
                ),
                "rna_counts": rna,
                "dna_counts": dna,
                "replicate_effects": effects,
                "reported_effect": _parse_finite_number(
                    value("reported_effect"),
                    field="reported_effect",
                    row_number=row_number,
                ),
                "reported_standard_error": _parse_finite_number(
                    value("reported_standard_error"),
                    field="reported_standard_error",
                    row_number=row_number,
                ),
                "endpoint_direction": value("endpoint_direction"),
                "missing": _parse_boolean(
                    value("missing"), field="missing", row_number=row_number
                ),
                "censored": _parse_boolean(
                    value("censored"), field="censored", row_number=row_number
                ),
                "qc_pass": _parse_boolean(
                    value("qc_pass"), field="qc_pass", row_number=row_number
                ),
            }
        )

    evidence = {
        "processed_asset_identity_status": PASS,
        "processed_asset_role_and_primary_measurement_route_status": manifest[
            "primary_measurement_route_status"
        ],
        "processed_asset_schema_role_binding_status": PASS,
        "endpoint_formula_and_primary_documentation_status": manifest[
            "endpoint_formula_and_primary_documentation_status"
        ],
        "biological_replicate_sample_role_provenance_status": manifest[
            "biological_replicate_sample_role_provenance_status"
        ],
        "license_and_reuse_rights_status": manifest["license_and_reuse_rights_status"],
        "historical_analytic_or_checkpoint_exposure_status": manifest[
            "historical_analytic_or_checkpoint_exposure_status"
        ],
        "split_readiness": copy.deepcopy(manifest["split_readiness"]),
    }
    return records, evidence


def _evidence_status(values: Sequence[Any]) -> str:
    if any(value == BLOCKED for value in values):
        return BLOCKED
    if any(value != PASS for value in values):
        return UNKNOWN
    return PASS


def _gate(gate_id: str, status: str, reason: str) -> dict[str, str]:
    if gate_id not in GATE_IDS or status not in {PASS, BLOCKED, UNKNOWN}:
        raise ObservationError("invalid gate result")
    return {"gate_id": gate_id, "status": status, "reason": reason}


def _hamming_distance(source: str, candidate: str) -> int | None:
    if len(source) != len(candidate):
        return None
    return sum(left != right for left, right in zip(source, candidate))


def _se_bin(value: float) -> str:
    if value == 0.0:
        return "ZERO"
    if value <= 0.01:
        return "GT_0_LE_0.01"
    if value <= 0.05:
        return "GT_0.01_LE_0.05"
    if value <= 0.1:
        return "GT_0.05_LE_0.1"
    return "GT_0.1"


def _replicate_effect(rna_count: float, dna_count: float, pseudocount: float) -> float:
    return math.log2((rna_count + pseudocount) / (dna_count + pseudocount))


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    evidence_class: str,
    production_asset_verified: bool,
) -> dict[str, Any]:
    """Audit in-memory canonical records and return only aggregate geometry."""

    _validate_protocol(protocol)
    if set(evidence) != set(EVIDENCE_FIELDS):
        raise ObservationError("aggregate evidence fields differ")
    split = evidence.get("split_readiness")
    if not isinstance(split, dict) or set(split) != set(SPLIT_FIELDS):
        raise ObservationError("split-readiness fields differ")
    if evidence_class not in {SYNTHETIC_EVIDENCE, PUBLIC_PROCESSED_EVIDENCE}:
        raise ObservationError("evidence class differs")
    if evidence_class == PUBLIC_PROCESSED_EVIDENCE and production_asset_verified is not True:
        raise ObservationError("public readiness requires the bound production asset path")
    if evidence_class == SYNTHETIC_EVIDENCE and production_asset_verified is not False:
        raise ObservationError("synthetic aggregation cannot carry production attestation")

    interface = protocol["canonical_internal_record_interface"]
    alphabet = set(interface["source_and_candidate_alphabet"])
    required_roles = tuple(interface["required_replicate_roles"])
    tolerance = float(interface["effect_and_se_absolute_tolerance"])
    maximum_edits = int(interface["maximum_source_relative_substitutions"])
    manifest = protocol["processed_asset_contract"]["official_processed_asset_manifest"]
    if evidence_class == PUBLIC_PROCESSED_EVIDENCE:
        if manifest["status"] != BOUND:
            raise ObservationError("public readiness requires a BOUND official asset manifest")
        pseudocount = float(manifest["endpoint_pseudocount"])
    else:
        pseudocount = 1.0

    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    join_axes = Counter()
    edit_histogram: Counter[str] = Counter()
    replicate_histogram: Counter[str] = Counter()
    count_validity = Counter()
    direction_histogram = Counter()
    se_histogram = Counter()
    qc_histogram = Counter()
    endpoint_math_valid = True
    replicate_math_valid = True
    edit_replay_valid = True
    join_valid = True
    record_shape_valid = bool(records)
    component_mapping_valid = bool(records)
    components: set[str] = set()

    for record in records:
        if set(record) != set(RECORD_FIELDS):
            record_shape_valid = False
            continue
        source_token = record["source_group_token"]
        candidate_token = record["candidate_token"]
        component_token = record["near_duplicate_component_token"]
        if not all(isinstance(item, str) and item for item in (source_token, candidate_token, component_token)):
            record_shape_valid = False
            continue
        if not all(
            isinstance(item, str) and item
            for item in (record["construct_context"], record["cell_context"])
        ):
            record_shape_valid = False
            continue
        components.add(component_token)
        families[source_token].append(record)

        for field in (
            "barcode_to_allele_join_closed",
            "allele_to_transcript_join_closed",
            "transcript_to_source_join_closed",
            "full_construct_join_closed",
        ):
            status = record[field] is True
            join_axes[f"{field}:{'PASS' if status else 'BLOCKED'}"] += 1
            join_valid = join_valid and status

        source = record["source_sequence"]
        candidate = record["candidate_sequence"]
        if not (
            isinstance(source, str)
            and isinstance(candidate, str)
            and source
            and candidate
            and set(source).issubset(alphabet)
            and set(candidate).issubset(alphabet)
        ):
            edit_replay_valid = False
            edit_histogram["INVALID_SEQUENCE_INTERFACE"] += 1
        else:
            distance = _hamming_distance(source, candidate)
            if distance is None or not 1 <= distance <= maximum_edits:
                edit_replay_valid = False
                edit_histogram["ILLEGAL_OR_OUT_OF_BUDGET"] += 1
            else:
                edit_histogram[str(distance)] += 1

        roles = record["replicate_roles"]
        provenance_tokens = record["replicate_sample_provenance_tokens"]
        rna = record["rna_counts"]
        dna = record["dna_counts"]
        effects = record["replicate_effects"]
        lengths = tuple(
            len(value) if isinstance(value, list) else -1
            for value in (roles, provenance_tokens, rna, dna, effects)
        )
        replicate_histogram[str(lengths)] += 1
        triplicate_shape = lengths == (3, 3, 3, 3, 3) and tuple(roles) == required_roles
        provenance_ok = (
            triplicate_shape
            and all(isinstance(value, str) and value for value in provenance_tokens)
            and len(set(provenance_tokens)) == 3
            and record["replicate_independence_closed"] is True
        )
        counts_ok = triplicate_shape and all(
            _finite_number(value) and float(value) >= 0.0 for value in [*rna, *dna]
        )
        nonempty_pairs = counts_ok and all(float(left) > 0.0 or float(right) > 0.0 for left, right in zip(rna, dna))
        effects_ok = triplicate_shape and all(_finite_number(value) for value in effects)
        recomputed_replicate_effects = (
            [
                _replicate_effect(float(left), float(right), pseudocount)
                for left, right in zip(rna, dna)
            ]
            if counts_ok and nonempty_pairs
            else []
        )
        effects_replay_ok = effects_ok and len(recomputed_replicate_effects) == 3 and all(
            abs(float(observed) - recomputed) <= tolerance
            for observed, recomputed in zip(effects, recomputed_replicate_effects)
        )
        count_validity["VALID_THREE_RNA_DNA_PAIRS" if counts_ok and nonempty_pairs else "INVALID"] += 1
        replicate_math_valid = (
            replicate_math_valid
            and counts_ok
            and nonempty_pairs
            and effects_replay_ok
            and provenance_ok
        )

        reported_effect = record["reported_effect"]
        reported_se = record["reported_standard_error"]
        if effects_replay_ok and _finite_number(reported_effect) and _finite_number(reported_se):
            recomputed_effect = mean(recomputed_replicate_effects)
            recomputed_se = stdev(recomputed_replicate_effects) / math.sqrt(3.0)
            math_ok = (
                float(reported_se) >= 0.0
                and abs(float(reported_effect) - recomputed_effect) <= tolerance
                and abs(float(reported_se) - recomputed_se) <= tolerance
            )
            endpoint_math_valid = endpoint_math_valid and math_ok
            se_histogram[_se_bin(float(reported_se)) if math_ok else "RECOMPUTE_MISMATCH"] += 1
            if recomputed_effect > 0.0:
                direction_histogram["POSITIVE"] += 1
            elif recomputed_effect < 0.0:
                direction_histogram["NEGATIVE"] += 1
            else:
                direction_histogram["ZERO"] += 1
        else:
            endpoint_math_valid = False
            se_histogram["INVALID"] += 1

        endpoint_math_valid = endpoint_math_valid and (
            record["endpoint_direction"] == interface["endpoint_direction"]
        )
        qc_key = (
            "QC_PASS_COMPLETE"
            if record["missing"] is False and record["censored"] is False and record["qc_pass"] is True
            else "MISSING_CENSORED_OR_QC_FAIL"
        )
        qc_histogram[qc_key] += 1

    family_sizes: Counter[str] = Counter()
    family_valid = bool(families) and record_shape_valid
    for members in families.values():
        candidate_tokens = [str(item["candidate_token"]) for item in members]
        distinct_candidates = {str(item["candidate_sequence"]) for item in members}
        family_sizes[str(len(distinct_candidates))] += 1
        if len(distinct_candidates) < int(interface["minimum_distinct_candidates_per_source_family"]):
            family_valid = False
        if len(set(candidate_tokens)) != len(candidate_tokens):
            family_valid = False
        if len(distinct_candidates) != len(candidate_tokens):
            family_valid = False
        if len({str(item["source_sequence"]) for item in members}) != 1:
            family_valid = False
        construct_contexts = {item["construct_context"] for item in members}
        cell_contexts = {item["cell_context"] for item in members}
        if (
            len(construct_contexts) != 1
            or not all(isinstance(item, str) and item for item in construct_contexts)
            or len(cell_contexts) != 1
            or not all(isinstance(item, str) and item for item in cell_contexts)
        ):
            family_valid = False
        if len({str(item["near_duplicate_component_token"]) for item in members}) != 1:
            component_mapping_valid = False

    asset_gate_status = _evidence_status(
        [
            evidence["processed_asset_identity_status"],
            evidence["processed_asset_role_and_primary_measurement_route_status"],
            evidence["processed_asset_schema_role_binding_status"],
        ]
    )
    endpoint_evidence_status = evidence["endpoint_formula_and_primary_documentation_status"]
    endpoint_status = (
        BLOCKED
        if not endpoint_math_valid or not record_shape_valid
        else PASS
        if endpoint_evidence_status == PASS
        else endpoint_evidence_status
        if endpoint_evidence_status in {BLOCKED, UNKNOWN}
        else UNKNOWN
    )
    replicate_evidence_status = evidence[
        "biological_replicate_sample_role_provenance_status"
    ]
    replicate_status = (
        BLOCKED
        if not replicate_math_valid or not record_shape_valid
        else PASS
        if replicate_evidence_status == PASS
        else replicate_evidence_status
        if replicate_evidence_status in {BLOCKED, UNKNOWN}
        else UNKNOWN
    )
    qc_valid = (
        bool(records)
        and record_shape_valid
        and qc_histogram.get("MISSING_CENSORED_OR_QC_FAIL", 0) == 0
    )
    rights_status = evidence["license_and_reuse_rights_status"]
    if rights_status not in {PASS, BLOCKED, UNKNOWN}:
        rights_status = UNKNOWN
    exposure_status = evidence["historical_analytic_or_checkpoint_exposure_status"]
    if exposure_status not in {PASS, BLOCKED, UNKNOWN}:
        exposure_status = UNKNOWN

    leakage_fields = (
        "source_group_leakage_count",
        "exact_sequence_leakage_count",
        "near_duplicate_leakage_count",
        "reverse_edge_leakage_count",
        "candidate_leakage_count",
        "study_context_leakage_count",
    )
    split_closed = (
        split["status"] == PASS
        and split["outcome_blind"] is True
        and split["components_indivisible"] is True
        and split["split_executed"] is False
        and type(split["assignment_output_count"]) is int
        and split["assignment_output_count"] == 0
        and all(type(split[field]) is int and split[field] == 0 for field in leakage_fields)
        and component_mapping_valid
    )
    split_status = BLOCKED if split["status"] == BLOCKED or not split_closed and split["status"] == PASS else UNKNOWN
    if split_closed:
        split_status = PASS

    effective_n = len(components)
    effective_n_status = (
        PASS if record_shape_valid and component_mapping_valid and effective_n > 0 else BLOCKED
    )
    required_n = int(protocol["prefrozen_power_contract"]["required_effective_n_for_power_and_ci_width"])
    power_status = PASS if effective_n_status == PASS and effective_n >= required_n else BLOCKED

    gate_results = [
        _gate(GATE_IDS[0], asset_gate_status, "processed asset identity, role, provenance, schema, and primary route"),
        _gate(GATE_IDS[1], PASS if join_valid and record_shape_valid else BLOCKED, "explicit barcode-to-source and full-construct joins"),
        _gate(GATE_IDS[2], PASS if family_valid else BLOCKED, "unique source identity and at least three distinct candidates per family"),
        _gate(GATE_IDS[3], PASS if edit_replay_valid and record_shape_valid else BLOCKED, "source-relative substitution replay within frozen budget"),
        _gate(GATE_IDS[4], endpoint_status, "primary endpoint semantics plus effect/SE recomputation"),
        _gate(GATE_IDS[5], replicate_status, "three independently proven biological RNA/DNA replicate pairs, count-derived effects, and valid SE"),
        _gate(GATE_IDS[6], PASS if qc_valid else BLOCKED, "missingness, censoring, QC, and selection closure"),
        _gate(GATE_IDS[7], rights_status, "license and internal reuse rights"),
        _gate(GATE_IDS[8], exposure_status, "historical analytic and checkpoint exposure"),
        _gate(GATE_IDS[9], split_status, "outcome-blind component split readiness and zero leakage"),
        _gate(GATE_IDS[10], effective_n_status, "post-dedup independent source-group effective N"),
        _gate(GATE_IDS[11], power_status, "prefrozen rho=0.25 power and full-CI-width readiness"),
    ]
    status_counts = Counter(item["status"] for item in gate_results)
    all_gates_pass = status_counts.get(PASS, 0) == len(GATE_IDS)
    terminal_status = protocol["gate_contract"]["nonpass_action"]
    if all_gates_pass:
        terminal_status = (
            protocol["gate_contract"]["all_pass_action"]
            if evidence_class == PUBLIC_PROCESSED_EVIDENCE
            else protocol["gate_contract"]["synthetic_all_pass_action"]
        )

    observations = {
        "processed_asset_role_schema_and_join_coverage": {
            "asset_gate_status": asset_gate_status,
            "join_axis_status_counts": dict(sorted(join_axes.items())),
        },
        "source_family_size_histogram": dict(sorted(family_sizes.items(), key=lambda item: int(item[0]))),
        "edit_distance_histogram": dict(sorted(edit_histogram.items())),
        "replicate_coverage_histogram": dict(sorted(replicate_histogram.items())),
        "rna_dna_count_validity_histogram": dict(sorted(count_validity.items())),
        "effect_direction_histogram": dict(sorted(direction_histogram.items())),
        "standard_error_validity_histogram": dict(sorted(se_histogram.items())),
        "missing_censoring_qc_histogram": dict(sorted(qc_histogram.items())),
        "rights_exposure_split_gate_status": {
            "rights": rights_status,
            "exposure": exposure_status,
            "split_readiness": split_status,
        },
        "post_dedup_effective_n_and_power_readiness": {
            "analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
            "effective_n": effective_n,
            "required_effective_n": required_n,
            "power_and_ci_width_readiness": power_status,
            "formal_qualification_power_gate_executed": False,
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": terminal_status,
        "evidence_status": evidence_class,
        "gate_status_counts": {status: status_counts.get(status, 0) for status in (PASS, BLOCKED, UNKNOWN)},
        "gate_results": gate_results,
        "aggregate_observations": observations,
        "frozen_outer_truth": copy.deepcopy(protocol["frozen_outer_truth"]),
        "qualification_changed": False,
        "credit_changed": False,
        "canonical_changed": False,
    }
    validate_public_report(report, protocol)
    return report


def aggregate_canonical_records(
    records: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise the gate implementation; this public helper is always synthetic."""

    return _aggregate_records(
        records,
        evidence,
        protocol,
        evidence_class=SYNTHETIC_EVIDENCE,
        production_asset_verified=False,
    )


def validate_public_report(report: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected_root = {
        "schema_version",
        "protocol_id",
        "dataset_id",
        "status",
        "evidence_status",
        "gate_status_counts",
        "gate_results",
        "aggregate_observations",
        "frozen_outer_truth",
        "qualification_changed",
        "credit_changed",
        "canonical_changed",
    }
    if set(report) != expected_root:
        raise OutputError("public report root differs")
    observations = report.get("aggregate_observations")
    if not isinstance(observations, dict) or set(observations) != set(OBSERVATION_SECTIONS):
        raise OutputError("public aggregate sections differ")
    gates = report.get("gate_results")
    if not isinstance(gates, list) or [item.get("gate_id") for item in gates] != list(GATE_IDS):
        raise OutputError("public gate list differs")
    if any(item.get("status") not in {PASS, BLOCKED, UNKNOWN} for item in gates):
        raise OutputError("public gate status differs")
    evidence_class = report.get("evidence_status")
    if evidence_class not in {SYNTHETIC_EVIDENCE, PUBLIC_PROCESSED_EVIDENCE}:
        raise OutputError("public evidence class differs")
    observed_counts = Counter(item["status"] for item in gates)
    expected_counts = {
        status: observed_counts.get(status, 0) for status in (PASS, BLOCKED, UNKNOWN)
    }
    if report.get("gate_status_counts") != expected_counts:
        raise OutputError("public gate status counts differ")
    all_gates_pass = observed_counts.get(PASS, 0) == len(GATE_IDS)
    expected_status = protocol["gate_contract"]["nonpass_action"]
    if all_gates_pass:
        expected_status = (
            protocol["gate_contract"]["all_pass_action"]
            if evidence_class == PUBLIC_PROCESSED_EVIDENCE
            else protocol["gate_contract"]["synthetic_all_pass_action"]
        )
    if report.get("status") != expected_status:
        raise OutputError("public terminal status differs")
    if any(report.get(field) is not False for field in ("qualification_changed", "credit_changed", "canonical_changed")):
        raise OutputError("preflight changed scientific state")
    if report.get("frozen_outer_truth") != protocol["frozen_outer_truth"]:
        raise OutputError("outer truth differs")

    forbidden_exact_keys = {
        "member_id",
        "barcode",
        "barcode_id",
        "candidate_token",
        "source_group_token",
        "near_duplicate_component_token",
        "source_sequence",
        "candidate_sequence",
        "transcript_id",
        "replicate_identifier",
        "split_assignment",
        "row_count",
        "row_effect",
        "row_standard_error",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if forbidden_exact_keys.intersection(value):
                raise OutputError("member/row payload key escaped aggregate report")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise OutputError("non-finite value in public report")

    walk(report)


def json_bytes(report: Mapping[str, Any], protocol: Mapping[str, Any]) -> bytes:
    validate_public_report(report, protocol)
    try:
        return (
            json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("report is not finite aggregate JSON") from exc


def _write_exclusive_aggregate(
    output_directory: Path,
    payload: bytes,
    protocol: Mapping[str, Any],
) -> Path:
    """Atomically publish exactly one fixed JSON file without replacement."""

    filename = protocol["output_contract"]["filename"]
    temporary: Path | None = None
    output_created = False
    try:
        if output_directory.exists() and any(output_directory.iterdir()):
            raise OutputError("output directory is not empty")
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / filename
        if output_path.exists():
            raise OutputError("aggregate report already exists")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=output_directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output_path)
            output_created = True
        except FileExistsError as exc:
            raise OutputError("aggregate report appeared during no-replace publication") from exc
        temporary.unlink()
        temporary = None
        directory_descriptor = os.open(output_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if tuple(path.name for path in output_directory.iterdir()) != (filename,):
            raise OutputError("publication did not produce exactly one aggregate JSON")
        return output_path
    except OutputError:
        raise
    except OSError as exc:
        if output_created:
            try:
                (output_directory / filename).unlink()
            except OSError:
                pass
        raise OutputError("cannot atomically publish aggregate-only output") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def execute(
    protocol_path: Path,
    processed_asset_path: Path,
    output_directory: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """The sole path that can emit ordinary-public readiness."""

    protocol = load_protocol(protocol_path)
    _ensure_ready_before_asset_or_output_io(protocol)
    production_root = repo_root or Path(protocol["repository_authority"]["production_repo_root"])
    binding_result = _default_binding_auditor(
        protocol,
        protocol_path,
        production_root,
    )
    if binding_result.get("status") != PASS:
        raise BindingNotFrozen("repository ancestry and exact-path binding did not pass")
    records, evidence = read_bound_processed_asset(processed_asset_path, protocol)
    report = _aggregate_records(
        records,
        evidence,
        protocol,
        evidence_class=PUBLIC_PROCESSED_EVIDENCE,
        production_asset_verified=True,
    )
    _write_exclusive_aggregate(output_directory, json_bytes(report, protocol), protocol)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--processed-asset", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        execute(
            args.protocol,
            args.processed_asset,
            args.output_directory,
            repo_root=args.repo_root,
        )
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("OK: one aggregate-only report published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
