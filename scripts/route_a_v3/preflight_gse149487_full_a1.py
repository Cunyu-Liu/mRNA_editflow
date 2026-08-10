#!/usr/bin/env python3
"""Fail-closed, stop-before-data preflight for public GSE149487/PLUMAGE.

The production path verifies Git/config authority, environment prerequisites,
and an exact directory inventory using descriptor-bound, no-follow metadata
operations.  It deliberately never opens or hashes ``manifest.json`` or any of
the 21 data payloads and has no qualifier or scientific-processing executor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse149487_stop_before_data_preflight.v1"
PROTOCOL_SCHEMA_VERSION = "route_a_v3_gse149487_external_evidence_roots.v1"
PROTOCOL_ID = "ROUTE_A_V3_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_V1"
DATASET_ID = "GSE149487"
TERMINAL_OUTCOME = "NOT_READY_FOR_STUDY_QUALIFICATION"
HASH_REVERIFICATION = "NOT_RUN_STOP_BEFORE_DATA"
UNBOUND_TOKEN = "UNKNOWN_NOT_ASSERTED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_AUTHORITY_PATHS = {
    "contract": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
    "data_role_registry": "docs/execution/route_a_v3_data_role_registry.yaml",
    "decision_log": "docs/execution/route_a_v3_decision_log.yaml",
    "asset_manifest": "configs/route_a_v3_gse149487_asset_manifest_v2.json",
    "raw_asset_registry": "data/v3_1/registry/raw_asset_manifest.jsonl",
}
PRODUCTION_AUTHORITY_ROOT = {
    "accepted_a0_base_commit": "fd722d5fa3c2538fce742b8942b1fb48e782760b",
    "active_authority_commit": "d328bf04c394d4960ac11058e079c063e09280af",
    "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
    "expected_branch": "routea-v3-a1-20260810",
    "authority_blobs": {
        "contract": {
            "repo_path": "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md",
            "sha256": "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
        },
        "data_role_registry": {
            "repo_path": "docs/execution/route_a_v3_data_role_registry.yaml",
            "sha256": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
        },
        "decision_log": {
            "repo_path": "docs/execution/route_a_v3_decision_log.yaml",
            "sha256": "a5b041fab24d9a4309603a085fa3fcab936d69a899285bfa752689a2ee5fd4fd",
        },
    },
}
QUALIFIER_CONFIG_PATH = "configs/route_a_v3_gse149487_a1_qualification.json"
QUALIFIER_SCRIPT_PATH = "scripts/route_a_v3/qualify_gse149487_plumage.py"
QUALIFIER_TEST_PATH = "tests/route_a_v3/test_qualify_gse149487_plumage.py"
PREFLIGHT_CONFIG_PATH = "configs/route_a_v3_gse149487_external_evidence_roots_v1.json"
PREFLIGHT_SCRIPT_PATH = "scripts/route_a_v3/preflight_gse149487_full_a1.py"
PREFLIGHT_TEST_PATH = "tests/route_a_v3/test_preflight_gse149487_full_a1.py"
PREFLIGHT_BINDING_KEY = "stop_before_data_preflight_binding"
PREFLIGHT_BINDING_KEYS = {
    "binding_scheme",
    "status",
    "implementation_commit",
    "external_evidence_config_path",
    "external_evidence_config_sha256",
    "preflight_script_path",
    "preflight_script_sha256",
    "preflight_test_path",
    "preflight_test_sha256",
}
EXPECTED_R4_BLOCKERS = (
    "CHECKPOINT_SPECIFIC_FOUNDATION_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_UNKNOWN_NOT_ASSERTED",
    "OUTCOME_BLIND_LONG_READ_MAPPING_PROVENANCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_METHOD_NOT_REPRODUCED",
    "PAPER_NATIVE_METHOD_SOURCE_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_MULTIPLE_TESTING_FAMILY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_OR_CI_GATE_FAILED",
    "PUBLISHED_RESULT_CROSSCHECK_UNKNOWN_NOT_ASSERTED",
    "RAW_KEY_UNCLASSIFIED_OUTCOME_BLIND_RECONCILIATION_NOT_ZERO",
    "UNADJUDICATED_OR_AMBIGUOUS_MAPPING_ROWS_PRESENT",
    "UNADJUDICATED_SEQUENCE_UNIVERSE_CLASSES_PRESENT",
)
EXPECTED_FOUNDATION_FAMILIES = (
    "OPTIMUS_5_PRIME",
    "UTR_LM_MRL",
    "MRNABERT",
    "ORTHRUS",
)
EXPECTED_GATE_TRUTH = {
    "ready_for_study_qualification": False,
    "qualified": False,
    "training_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "canonical_record_count": 0,
}
EXPECTED_CLAIM_BOUNDARY = (
    "This stop-before-data preflight may verify repository authority, environment readiness, "
    "and descriptor-bound filenames and sizes only. It never opens or hashes the manifest or "
    "any of the 21 data payloads, never executes the qualifier, and never establishes study "
    "qualification, training permission, model-selection permission, checkpoint-specific "
    "foundation exposure, or a scientific claim."
)


class PreflightError(RuntimeError):
    """Base class for a closed preflight failure."""


class ProtocolError(PreflightError):
    pass


class BindingError(PreflightError):
    pass


class AuthorityError(PreflightError):
    pass


class ScopeError(PreflightError):
    pass


class InventoryError(PreflightError):
    pass


class EnvironmentError(PreflightError):
    pass


class PublicationError(PreflightError):
    pass


class CommittedPublicationNotAccepted(RuntimeError):
    """The final name exists, but its identity could not be accepted."""

    def __init__(self, path: Path, receipt: Mapping[str, Any]):
        super().__init__("publication reached its commit point but was not accepted")
        self.path = path
        self.receipt = dict(receipt)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ProtocolError(f"{label} keys mismatch: {actual!r}")
    return value


def _strict(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected):
        raise ProtocolError(f"{label} must equal the frozen value")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ProtocolError(f"{label} must equal the frozen value")
        for key in expected:
            _strict(actual[key], expected[key], label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ProtocolError(f"{label} must equal the frozen value")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _strict(actual_item, expected_item, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise ProtocolError(f"{label} must equal the frozen value")


def _safe_repo_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ProtocolError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ProtocolError(f"{label} is not canonical")
    lowered = {part.casefold() for part in path.parts}
    if {"restricted", "sealed_external"} & lowered or "gse246381" in lowered:
        raise ScopeError(f"{label} enters a forbidden scope")
    return value


def _lexical_absolute(value: Path | str, *, label: str) -> Path:
    try:
        absolute = Path(os.path.abspath(os.fspath(value)))
    except TypeError as exc:
        raise ScopeError(f"{label} is not path-like") from exc
    lowered = {part.casefold() for part in absolute.parts}
    if {"restricted", "sealed_external"} & lowered or "gse246381" in lowered:
        raise ScopeError(f"{label} enters a forbidden scope")
    return absolute


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        common = Path(os.path.commonpath((str(first), str(second))))
    except ValueError:
        return False
    return common == first or common == second


def _pre_open_lexical_scope(
    *,
    repo_root: Path | str,
    protocol_path: Path | str,
    data_root: Path | str,
    output_path: Path | str,
    failure_path: Path | str,
    claim_path: Path | str,
) -> dict[str, Path]:
    """Validate caller paths without opening or stating any caller-controlled path."""

    paths = {
        "repo_root": _lexical_absolute(repo_root, label="repo root"),
        "protocol_path": _lexical_absolute(protocol_path, label="protocol path"),
        "data_root": _lexical_absolute(data_root, label="data root"),
        "output_path": _lexical_absolute(output_path, label="output path"),
        "failure_path": _lexical_absolute(failure_path, label="failure path"),
        "claim_path": _lexical_absolute(claim_path, label="claim path"),
    }
    expected_protocol = paths["repo_root"] / PREFLIGHT_CONFIG_PATH
    if paths["protocol_path"] != expected_protocol:
        raise ScopeError("protocol path must be the exact repository external-evidence config path")
    if len({paths["output_path"], paths["failure_path"], paths["claim_path"]}) != 3:
        raise ScopeError("output, failure, and claim targets must be distinct")
    for first_label, second_label in (
        ("repo_root", "data_root"),
        ("repo_root", "output_path"),
        ("repo_root", "failure_path"),
        ("repo_root", "claim_path"),
        ("data_root", "output_path"),
        ("data_root", "failure_path"),
        ("data_root", "claim_path"),
    ):
        if _paths_overlap(paths[first_label], paths[second_label]):
            raise ScopeError(f"caller paths overlap: {first_label} and {second_label}")
    return paths


def _read_regular_nofollow(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuthorityError(f"cannot open {label} without following symlinks") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AuthorityError(f"{label} is not a single-link regular file")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise AuthorityError(f"{label} size is outside the allowed range")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != before.st_size or len(payload) > maximum_bytes:
            raise AuthorityError(f"{label} changed or exceeded the read limit")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise AuthorityError(f"{label} changed while read")
        return payload
    finally:
        os.close(descriptor)


def _parse_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    authority_root: Mapping[str, Any] = PRODUCTION_AUTHORITY_ROOT,
) -> None:
    _exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "dataset_id",
            "dataset_alias",
            "ordinary_public_data_only",
            "protocol_status",
            "terminal_outcome",
            "protocol_trust",
            "production_authority_root",
            "authority_bindings",
            "qualifier_config_contract",
            "external_evidence_roots",
            "historical_r4_closure",
            "scientific_evidence_boundaries",
            "foundation_checkpoint_exposure",
            "data_directory_contract",
            "environment_contract",
            "gate_truth",
            "claim_boundary",
        },
        label="protocol",
    )
    _strict(protocol["schema_version"], PROTOCOL_SCHEMA_VERSION, label="schema_version")
    _strict(protocol["protocol_id"], PROTOCOL_ID, label="protocol_id")
    _strict(protocol["dataset_id"], DATASET_ID, label="dataset_id")
    _strict(protocol["dataset_alias"], "PLUMAGE", label="dataset_alias")
    _strict(protocol["ordinary_public_data_only"], True, label="ordinary_public_data_only")
    _strict(protocol["protocol_status"], "PREFROZEN_FAIL_CLOSED_STOP_BEFORE_DATA", label="protocol_status")
    _strict(protocol["terminal_outcome"], TERMINAL_OUTCOME, label="terminal_outcome")

    trust = _exact_keys(
        protocol["protocol_trust"],
        {"canonicalization", "immutable_in_implementation_commit", "binding_owner", "purpose"},
        label="protocol_trust",
    )
    _strict(trust["canonicalization"], "RAW_FILE_BYTES_SHA256_V1", label="canonicalization")
    _strict(trust["immutable_in_implementation_commit"], True, label="immutable protocol")
    _strict(trust["binding_owner"], QUALIFIER_CONFIG_PATH, label="binding owner")
    _strict(
        trust["purpose"],
        "STATIC_SCIENTIFIC_EVIDENCE_ROOT_BOUND_BY_QUALIFIER_CONFIG_ONLY_REBIND",
        label="protocol purpose",
    )
    _strict(protocol["production_authority_root"], dict(authority_root), label="production authority root")

    root_blobs = authority_root.get("authority_blobs")
    if not isinstance(root_blobs, dict) or set(root_blobs) != {"contract", "data_role_registry", "decision_log"}:
        raise ProtocolError("immutable authority root has an invalid authority_blobs schema")
    for key in ("accepted_a0_base_commit", "active_authority_commit"):
        if not isinstance(authority_root.get(key), str) or COMMIT_RE.fullmatch(authority_root[key]) is None:
            raise ProtocolError(f"immutable authority root has an invalid {key}")
    _strict(
        authority_root.get("active_amendment_decision_ids"),
        ["V3-DEC-017", "V3-DEC-018"],
        label="immutable active amendment decision IDs",
    )
    expected_branch = authority_root.get("expected_branch")
    if not isinstance(expected_branch, str) or not re.fullmatch(r"[A-Za-z0-9._/-]+", expected_branch):
        raise ProtocolError("immutable authority root has an invalid expected branch")

    authorities = _exact_keys(protocol["authority_bindings"], set(EXPECTED_AUTHORITY_PATHS), label="authority_bindings")
    for key, expected_path in EXPECTED_AUTHORITY_PATHS.items():
        item = _exact_keys(authorities[key], {"repo_path", "sha256"}, label=f"authority.{key}")
        _strict(_safe_repo_path(item["repo_path"], label=f"authority.{key}.repo_path"), expected_path, label=f"authority.{key}.repo_path")
        if not isinstance(item["sha256"], str) or SHA256_RE.fullmatch(item["sha256"]) is None:
            raise ProtocolError(f"authority.{key}.sha256 is invalid")
    for key, expected in root_blobs.items():
        _strict(authorities[key], expected, label=f"production-root authority.{key}")

    qualifier = _exact_keys(
        protocol["qualifier_config_contract"],
        {
            "protocol_id",
            "config_repo_path",
            "qualifier_repo_path",
            "test_repo_path",
            "preflight_binding_object_key",
            "preflight_binding_scheme",
            "preflight_config_repo_path",
            "preflight_script_repo_path",
            "preflight_test_repo_path",
            "required_fail_closed_fields",
        },
        label="qualifier_config_contract",
    )
    _strict(qualifier["protocol_id"], "ROUTE_A_V3_GSE149487_PLUMAGE_FULL_A1_QUALIFICATION_V1", label="qualifier protocol")
    _strict(qualifier["config_repo_path"], QUALIFIER_CONFIG_PATH, label="qualifier config path")
    _strict(qualifier["qualifier_repo_path"], QUALIFIER_SCRIPT_PATH, label="qualifier script path")
    _strict(qualifier["test_repo_path"], QUALIFIER_TEST_PATH, label="qualifier test path")
    _strict(qualifier["preflight_binding_object_key"], PREFLIGHT_BINDING_KEY, label="preflight binding key")
    _strict(
        qualifier["preflight_binding_scheme"],
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="preflight binding scheme",
    )
    _strict(qualifier["preflight_config_repo_path"], PREFLIGHT_CONFIG_PATH, label="preflight config path")
    _strict(qualifier["preflight_script_repo_path"], PREFLIGHT_SCRIPT_PATH, label="preflight script path")
    _strict(qualifier["preflight_test_repo_path"], PREFLIGHT_TEST_PATH, label="preflight test path")
    required_fields = qualifier["required_fail_closed_fields"]
    _strict(
        required_fields,
        {
            "foundation_exposure.audit_status": "UNKNOWN_NOT_ASSERTED",
            "foundation_exposure.checkpoint_id": "UNKNOWN_NOT_ASSERTED",
            "foundation_exposure.checkpoint_sha256": "UNKNOWN_NOT_ASSERTED",
            "foundation_exposure.sequence_exposed": True,
            "foundation_exposure.label_exposed": True,
            "foundation_exposure.unknown_checkpoint_blocks_qualification": True,
            "scope.training_allowed": False,
            "scope.model_selection_allowed": False,
        },
        label="required_fail_closed_fields",
    )

    roots = _exact_keys(
        protocol["external_evidence_roots"],
        {"paper", "geo", "paper_supplements", "lim6c_mapping_workbook"},
        label="external_evidence_roots",
    )
    paper = _exact_keys(
        roots["paper"],
        {"immutable_id", "pmc_id", "official_url", "public_full_text_url", "method_surface_status", "exact_executable_method_source_status"},
        label="paper evidence root",
    )
    _strict(paper["immutable_id"], "DOI:10.1038/s41467-021-24445-6", label="paper DOI")
    _strict(paper["pmc_id"], "PMC8270899", label="paper PMC ID")
    _strict(paper["method_surface_status"], "VERIFIED_METHOD_SURFACE_ONLY", label="paper method surface")
    _strict(paper["exact_executable_method_source_status"], "BLOCKED_UNKNOWN_NOT_ASSERTED", label="paper executable source")
    geo = _exact_keys(
        roots["geo"],
        {"immutable_id", "official_url", "payload_policy", "license_status"},
        label="GEO evidence root",
    )
    _strict(geo["immutable_id"], "GEO:GSE149487", label="GEO immutable ID")
    _strict(geo["payload_policy"], "NONREDISTRIBUTABLE_LOCATOR_HASH_ONLY", label="GEO payload policy")
    _strict(geo["license_status"], "BLOCKED_NO_DATA_SPECIFIC_REDISTRIBUTION_GRANT", label="GEO license status")

    evidence = _exact_keys(
        protocol["scientific_evidence_boundaries"],
        {"long_read_identity_method", "paper_native_method", "published_result_crosscheck", "license_boundary"},
        label="scientific_evidence_boundaries",
    )
    _strict(evidence["long_read_identity_method"], {
        "status": "VERIFIED_METHOD_SURFACE_ONLY",
        "exact_public_description_to_barcode_map": "BLOCKED",
        "pre_outcome_mapping_timing": "BLOCKED",
        "qualification_effect": "BLOCK",
    }, label="long_read_identity_method")
    _strict(evidence["paper_native_method"], {
        "exact_executable_source": "BLOCKED",
        "mann_whitney_implementation": "BLOCKED",
        "fdr_family_definition": "BLOCKED",
        "qualification_effect": "BLOCK",
    }, label="paper_native_method")
    _strict(evidence["published_result_crosscheck"], {
        "published_endpoint_discovery_count": 190,
        "unique_construct_pair_count": 180,
        "interpretation": "ENDPOINT_DISCOVERIES_ARE_NOT_UNIQUE_PAIRS",
        "status": "AUTHOR_ADJUDICATION_REQUIRED",
        "qualification_effect": "BLOCK",
    }, label="published_result_crosscheck")
    _strict(evidence["license_boundary"], {
        "moesm3_and_moesm8": "CC_BY_4_0_ONLY",
        "geo_raw_18": "NONREDISTRIBUTABLE_LOCATOR_HASH_ONLY",
        "lim6c": "NO_EXPLICIT_LICENSE",
        "all_21_assets_license_status": "BLOCKED",
        "qualification_effect": "BLOCK",
    }, label="license_boundary")

    r4 = _exact_keys(
        protocol["historical_r4_closure"],
        {"bundle_path", "qualification_report", "sha256sums", "publication_commit", "reuse_policy", "rerun_is_qualification_path", "exact_blockers"},
        label="historical_r4_closure",
    )
    _strict(
        r4["bundle_path"],
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_GSE149487_PLUMAGE_FULL_QUAL_20260810T131156P0800_a859166_R4",
        label="R4 bundle path",
    )
    _strict(
        r4["qualification_report"],
        {
            "filename": "QUALIFICATION_REPORT.json",
            "bytes": 19987,
            "sha256": "19df844b55ef7b8dbf53ba3044a51132bdea1f0d1dfa6809a720a2a83a7030b3",
            "status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "qualified": False,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
        },
        label="R4 qualification report",
    )
    _strict(
        r4["sha256sums"],
        {"filename": "SHA256SUMS", "sha256": "c72c63c2090052657beaa797e3ba3196200f8cbc3e9c5a97cf1a4a04a4db3631"},
        label="R4 SHA256SUMS",
    )
    _strict(
        r4["publication_commit"],
        {"filename": "PUBLICATION_COMMIT.json", "sha256": "3149001644cf1b21db74021b12ca1e887977a9d0d13deff3b2f57b18e4b64ca4"},
        label="R4 publication commit",
    )
    _strict(tuple(r4["exact_blockers"]), EXPECTED_R4_BLOCKERS, label="historical exact blockers")
    _strict(r4["reuse_policy"], "REFERENCE_AGGREGATE_ONLY_DO_NOT_REOPEN_OR_REHASH", label="R4 reuse policy")
    _strict(r4["rerun_is_qualification_path"], False, label="R4 rerun boundary")

    foundation = protocol["foundation_checkpoint_exposure"]
    if not isinstance(foundation, list) or tuple(item.get("checkpoint_family") for item in foundation) != EXPECTED_FOUNDATION_FAMILIES:
        raise ProtocolError("foundation checkpoint family set/order changed")
    for item in foundation:
        _exact_keys(item, {"checkpoint_family", "binding_status", "model_card_source_level_status", "exact_sequence_overlap", "near_duplicate_overlap", "gse149487_label_exposure"}, label="foundation checkpoint")
        for key in ("exact_sequence_overlap", "near_duplicate_overlap", "gse149487_label_exposure"):
            _strict(item[key], "BLOCKED_UNKNOWN_NOT_ASSERTED", label=f"{item['checkpoint_family']}.{key}")

    directory = _exact_keys(
        protocol["data_directory_contract"],
        {
            "listing_mode", "inventory_source_authority_key", "inventory_derivation",
            "expected_entry_count", "expected_payload_asset_count",
            "expected_geo_raw_count", "expected_supplement_count", "expected_manifest_count",
            "manifest_filename", "manifest_bytes", "manifest_declared_sha256_reference_only",
            "manifest_open_allowed", "payload_open_allowed", "hash_reverification", "expected_entries",
        },
        label="data_directory_contract",
    )
    _strict(directory["listing_mode"], "DESCRIPTOR_BOUND_NOFOLLOW_NAMES_AND_SIZES_ONLY", label="listing mode")
    _strict(directory["inventory_source_authority_key"], "asset_manifest", label="inventory source")
    _strict(
        directory["inventory_derivation"],
        "PREFROZEN_FROM_BOUND_REPO_ASSET_MANIFEST_NO_DATA_ROOT_MANIFEST_OPEN",
        label="inventory derivation",
    )
    for key, expected in {
        "expected_entry_count": 22,
        "expected_payload_asset_count": 21,
        "expected_geo_raw_count": 18,
        "expected_supplement_count": 3,
        "expected_manifest_count": 1,
        "manifest_filename": "manifest.json",
        "manifest_bytes": 6670,
        "manifest_open_allowed": False,
        "payload_open_allowed": False,
        "hash_reverification": HASH_REVERIFICATION,
    }.items():
        _strict(directory[key], expected, label=f"data_directory_contract.{key}")
    entries = directory["expected_entries"]
    if not isinstance(entries, list) or len(entries) != 22:
        raise ProtocolError("expected_entries must contain exactly 22 members")
    names: set[str] = set()
    kind_counts: dict[str, int] = {}
    for entry in entries:
        _exact_keys(entry, {"name", "bytes", "kind"}, label="expected entry")
        name = entry["name"]
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ProtocolError("expected entry name is unsafe or duplicated")
        if not isinstance(entry["bytes"], int) or isinstance(entry["bytes"], bool) or entry["bytes"] < 1:
            raise ProtocolError("expected entry size is invalid")
        names.add(name)
        kind_counts[entry["kind"]] = kind_counts.get(entry["kind"], 0) + 1
    _strict(kind_counts, {"MANIFEST_METADATA_NOT_OPENED": 1, "GEO_RAW_COUNT": 18, "SUPPLEMENT_WORKBOOK": 3}, label="inventory kind counts")

    environment = _exact_keys(
        protocol["environment_contract"],
        {
            "minimum_python", "required_python_modules", "minimum_output_free_bytes",
            "network_allowed", "qualifier_execution_allowed", "scientific_processing_allowed",
        },
        label="environment_contract",
    )
    _strict(environment["minimum_python"], "3.10", label="minimum Python")
    _strict(environment["required_python_modules"], ["numpy", "openpyxl", "scipy"], label="required modules")
    _strict(environment["network_allowed"], False, label="network policy")
    _strict(environment["qualifier_execution_allowed"], False, label="qualifier execution policy")
    _strict(environment["scientific_processing_allowed"], False, label="scientific processing policy")
    _strict(environment["minimum_output_free_bytes"], 1073741824, label="minimum output free bytes")
    _strict(protocol["gate_truth"], EXPECTED_GATE_TRUTH, label="gate truth")
    _strict(protocol["claim_boundary"], EXPECTED_CLAIM_BOUNDARY, label="claim boundary")


def load_protocol(
    path: Path | str,
    *,
    test_only_authority_root: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = Path(os.path.abspath(os.fspath(path)))
    payload = _read_regular_nofollow(protocol_path, maximum_bytes=2_000_000, label="external-evidence protocol")
    protocol = _parse_json(payload, label="external-evidence protocol")
    authority_root = PRODUCTION_AUTHORITY_ROOT if test_only_authority_root is None else test_only_authority_root
    _validate_protocol(protocol, authority_root=authority_root)
    return protocol, {
        "path": str(protocol_path),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "binding_model": "IMMUTABLE_STATIC_BLOB_BOUND_FROM_QUALIFIER_CONFIG",
    }


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AuthorityError(f"git {' '.join(args)} failed")
    return result


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AuthorityError(f"git {' '.join(args)} failed")
    return result.stdout


def _repo_file(repo_root: Path, relative: str, expected_sha256: str, *, label: str) -> bytes:
    _safe_repo_path(relative, label=label)
    payload = _read_regular_nofollow(repo_root / relative, maximum_bytes=100_000_000, label=label)
    if _sha256_bytes(payload) != expected_sha256:
        raise AuthorityError(f"{label} SHA-256 drift")
    return payload


def _dotted(document: Mapping[str, Any], dotted: str) -> Any:
    value: Any = document
    for component in dotted.split("."):
        if not isinstance(value, dict) or component not in value:
            raise AuthorityError(f"qualifier config lacks {dotted}")
        value = value[component]
    return value


def _validate_preflight_binding(binding: Any, *, require_bound: bool) -> Mapping[str, Any]:
    if not isinstance(binding, dict) or set(binding) != PREFLIGHT_BINDING_KEYS:
        raise BindingError("qualifier preflight binding has a non-canonical schema")
    fixed = {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "external_evidence_config_path": PREFLIGHT_CONFIG_PATH,
        "preflight_script_path": PREFLIGHT_SCRIPT_PATH,
        "preflight_test_path": PREFLIGHT_TEST_PATH,
    }
    for key, expected in fixed.items():
        if binding.get(key) != expected:
            raise BindingError(f"qualifier preflight binding drift: {key}")
    for key in (
        "external_evidence_config_sha256",
        "preflight_script_sha256",
        "preflight_test_sha256",
    ):
        if not isinstance(binding.get(key), str) or SHA256_RE.fullmatch(binding[key]) is None:
            raise BindingError(f"qualifier preflight binding has an invalid {key}")
    status_value = binding.get("status")
    commit_value = binding.get("implementation_commit")
    if status_value == UNBOUND_TOKEN:
        if commit_value != UNBOUND_TOKEN:
            raise BindingError(f"{UNBOUND_TOKEN} binding must retain implementation_commit={UNBOUND_TOKEN}")
        if require_bound:
            raise BindingError(f"qualifier stop-before-data preflight binding is {UNBOUND_TOKEN}")
    elif status_value == "BOUND":
        if not isinstance(commit_value, str) or COMMIT_RE.fullmatch(commit_value) is None:
            raise BindingError("BOUND binding lacks a full implementation commit")
    else:
        raise BindingError("qualifier stop-before-data preflight binding status is invalid")
    return binding


def _load_bound_qualifier_before_data(
    repo_root: Path | str,
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], bytes, Mapping[str, Any]]:
    """Reject UNKNOWN_NOT_ASSERTED from qualifier config before data/output stat."""

    root = Path(os.path.abspath(os.fspath(repo_root)))
    qualifier_path = protocol["qualifier_config_contract"]["config_repo_path"]
    payload = _read_regular_nofollow(
        root / qualifier_path,
        maximum_bytes=5_000_000,
        label="qualifier config",
    )
    qualifier = _parse_json(payload, label="qualifier config")
    if qualifier.get("protocol_id") != protocol["qualifier_config_contract"]["protocol_id"]:
        raise BindingError("qualifier protocol ID mismatch")
    binding = _validate_preflight_binding(qualifier.get(PREFLIGHT_BINDING_KEY), require_bound=True)
    return root, qualifier, payload, binding


def _git_json(repo_root: Path, revision_and_path: str, *, label: str) -> dict[str, Any]:
    return _parse_json(_git_bytes(repo_root, "show", revision_and_path), label=label)


def audit_repo_authority(
    repo_root: Path | str,
    protocol: Mapping[str, Any],
    *,
    qualifier_config: Mapping[str, Any] | None = None,
    qualifier_payload: bytes | None = None,
    bound_binding: Mapping[str, Any] | None = None,
    protocol_sha256: str | None = None,
    authority_root: Mapping[str, Any] = PRODUCTION_AUTHORITY_ROOT,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(repo_root)))
    if qualifier_config is None or qualifier_payload is None or bound_binding is None:
        root, loaded_config, loaded_payload, loaded_binding = _load_bound_qualifier_before_data(root, protocol)
        qualifier_config = loaded_config
        qualifier_payload = loaded_payload
        bound_binding = loaded_binding
    else:
        _validate_preflight_binding(bound_binding, require_bound=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip())
    if top != root:
        raise AuthorityError("repo_root is not the Git toplevel")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise AuthorityError("repository is not clean")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if COMMIT_RE.fullmatch(head) is None:
        raise AuthorityError("HEAD is not a full commit")
    expected_branch = authority_root["expected_branch"]
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip()
    if branch != expected_branch:
        raise AuthorityError("current branch is not the immutable production branch")
    origin_ref = f"refs/remotes/origin/{expected_branch}"
    origin_head = _git(root, "rev-parse", "--verify", origin_ref).stdout.strip()
    if origin_head != head:
        raise AuthorityError("immutable production origin ref does not equal HEAD")

    implementation_commit = bound_binding["implementation_commit"]
    accepted_a0_commit = authority_root["accepted_a0_base_commit"]
    active_authority_commit = authority_root["active_authority_commit"]
    ancestry_pairs = (
        (accepted_a0_commit, active_authority_commit, "accepted A0 to active authority"),
        (active_authority_commit, implementation_commit, "active authority to implementation I"),
        (implementation_commit, head, "implementation I to HEAD"),
    )
    for ancestor, descendant, label in ancestry_pairs:
        if _git(root, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode != 0:
            raise AuthorityError(f"immutable Git ancestry failure: {label}")
    for key, item in authority_root["authority_blobs"].items():
        historical = _git_bytes(root, "show", f"{active_authority_commit}:{item['repo_path']}")
        if _sha256_bytes(historical) != item["sha256"]:
            raise AuthorityError(f"active authority commit blob drift: {key}")

    current_qualifier_payload = _read_regular_nofollow(
        root / QUALIFIER_CONFIG_PATH,
        maximum_bytes=5_000_000,
        label="qualifier config after Git clean check",
    )
    if current_qualifier_payload != qualifier_payload:
        raise AuthorityError("qualifier config changed during authority audit")

    observed: dict[str, str] = {}
    for key, item in protocol["authority_bindings"].items():
        _repo_file(root, item["repo_path"], item["sha256"], label=f"authority {key}")
        observed[key] = item["sha256"]

    current_preflight_hashes: dict[str, str] = {}
    preflight_blob_specs = (
        ("external_evidence_config", "external_evidence_config_path", "external_evidence_config_sha256"),
        ("preflight_script", "preflight_script_path", "preflight_script_sha256"),
        ("preflight_test", "preflight_test_path", "preflight_test_sha256"),
    )
    for label, path_key, hash_key in preflight_blob_specs:
        _repo_file(root, bound_binding[path_key], bound_binding[hash_key], label=label)
        current_preflight_hashes[label] = bound_binding[hash_key]
    if protocol_sha256 is not None and protocol_sha256 != bound_binding["external_evidence_config_sha256"]:
        raise AuthorityError("loaded external-evidence config is not the qualifier-bound blob")

    for dotted, expected in protocol["qualifier_config_contract"]["required_fail_closed_fields"].items():
        actual = _dotted(qualifier_config, dotted)
        if type(actual) is not type(expected) or actual != expected:
            raise AuthorityError(f"qualifier fail-closed field drift: {dotted}")
    qualifier_authority = qualifier_config.get("authority", {})
    if not isinstance(qualifier_authority, dict):
        raise AuthorityError("qualifier authority must be an object")
    qualifier_root_fields = {
        "accepted_a0_base_commit": authority_root["accepted_a0_base_commit"],
        "active_authority_commit": authority_root["active_authority_commit"],
        "active_amendment_decision_ids": authority_root["active_amendment_decision_ids"],
        "contract_path": authority_root["authority_blobs"]["contract"]["repo_path"],
        "contract_sha256": authority_root["authority_blobs"]["contract"]["sha256"],
        "data_role_registry_path": authority_root["authority_blobs"]["data_role_registry"]["repo_path"],
        "data_role_registry_sha256": authority_root["authority_blobs"]["data_role_registry"]["sha256"],
        "decision_log_path": authority_root["authority_blobs"]["decision_log"]["repo_path"],
        "decision_log_sha256": authority_root["authority_blobs"]["decision_log"]["sha256"],
    }
    for key, expected in qualifier_root_fields.items():
        actual = qualifier_authority.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise AuthorityError(f"qualifier production authority root drift: {key}")
    if qualifier_authority.get("asset_manifest_sha256") != observed["asset_manifest"]:
        raise AuthorityError("qualifier authority link drift: asset_manifest_sha256")
    qualifier_code_hashes: dict[str, str] = {}
    qualifier_code_specs = (
        ("qualifier_script", QUALIFIER_SCRIPT_PATH, "qualifier_sha256"),
        ("qualifier_test", QUALIFIER_TEST_PATH, "focused_test_sha256"),
    )
    for label, relative, hash_key in qualifier_code_specs:
        expected_hash = qualifier_authority.get(hash_key)
        if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
            raise AuthorityError(f"qualifier authority has an invalid {hash_key}")
        _repo_file(root, relative, expected_hash, label=label)
        qualifier_code_hashes[label] = expected_hash

    qualifier_implementation_commit = qualifier_authority.get("implementation_commit")
    if qualifier_implementation_commit != implementation_commit:
        raise AuthorityError("qualifier authority implementation commit is not the bound I commit")
    if _git(root, "merge-base", "--is-ancestor", implementation_commit, head, check=False).returncode != 0:
        raise AuthorityError("preflight implementation commit is not an ancestor of HEAD")
    for _label, path_key, hash_key in preflight_blob_specs:
        relative = bound_binding[path_key]
        historical = _git_bytes(root, "show", f"{implementation_commit}:{relative}")
        if _sha256_bytes(historical) != bound_binding[hash_key]:
            raise AuthorityError(f"implementation commit does not bind {relative}")

    implementation_qualifier = _git_json(
        root,
        f"{implementation_commit}:{QUALIFIER_CONFIG_PATH}",
        label="implementation-stage qualifier config",
    )
    expected_unknown_binding = dict(bound_binding)
    expected_unknown_binding["status"] = UNBOUND_TOKEN
    expected_unknown_binding["implementation_commit"] = UNBOUND_TOKEN
    unknown_binding = _validate_preflight_binding(
        implementation_qualifier.get(PREFLIGHT_BINDING_KEY),
        require_bound=False,
    )
    if dict(unknown_binding) != expected_unknown_binding:
        raise AuthorityError("implementation commit does not contain the exact UNKNOWN_NOT_ASSERTED-I binding")
    implementation_authority = implementation_qualifier.get("authority")
    if not isinstance(implementation_authority, dict):
        raise AuthorityError("implementation-stage qualifier authority must be an object")
    if implementation_authority.get("implementation_commit") != UNBOUND_TOKEN:
        raise AuthorityError(
            f"implementation-stage qualifier authority must retain implementation_commit={UNBOUND_TOKEN}"
        )
    for label, relative, hash_key in qualifier_code_specs:
        expected_hash = implementation_authority.get(hash_key)
        if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
            raise AuthorityError(f"implementation-stage qualifier authority has an invalid {hash_key}")
        if _sha256_bytes(_git_bytes(root, "show", f"{implementation_commit}:{relative}")) != expected_hash:
            raise AuthorityError(f"implementation commit does not bind {label}")

    descendants = _git(root, "rev-list", "--first-parent", "--reverse", f"{implementation_commit}..{head}").stdout.splitlines()
    if not descendants:
        raise AuthorityError("BOUND-B commit is absent after implementation commit")
    binding_commit = descendants[0]
    parent_line = _git(root, "rev-list", "--parents", "-n", "1", binding_commit).stdout.split()
    if len(parent_line) != 2 or parent_line[1] != implementation_commit:
        raise AuthorityError("BOUND-B must be a non-merge direct child of implementation commit")
    changed_paths = {
        line for line in _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", binding_commit).stdout.splitlines()
        if line
    }
    if changed_paths != {QUALIFIER_CONFIG_PATH}:
        raise AuthorityError("BOUND-B must modify only the qualifier config")
    bound_qualifier = _git_json(
        root,
        f"{binding_commit}:{QUALIFIER_CONFIG_PATH}",
        label="binding-stage qualifier config",
    )
    binding_at_b = _validate_preflight_binding(bound_qualifier.get(PREFLIGHT_BINDING_KEY), require_bound=True)
    if dict(binding_at_b) != dict(bound_binding):
        raise AuthorityError("BOUND-B binding differs from the current qualifier binding")
    expected_bound_qualifier = json.loads(json.dumps(implementation_qualifier))
    expected_bound_qualifier[PREFLIGHT_BINDING_KEY] = dict(bound_binding)
    expected_bound_qualifier["authority"]["implementation_commit"] = implementation_commit
    if bound_qualifier != expected_bound_qualifier:
        raise AuthorityError("BOUND-B changed qualifier content beyond the three allowed binding fields")
    return {
        "status": "PASS_CONFIG_ONLY_BINDING_VERIFIED",
        "git_head": head,
        "git_clean": True,
        "branch": branch,
        "origin_branch_head": origin_head,
        "accepted_a0_base_commit": accepted_a0_commit,
        "active_authority_commit": active_authority_commit,
        "active_amendment_decision_ids": list(authority_root["active_amendment_decision_ids"]),
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
        "binding_scheme": bound_binding["binding_scheme"],
        "qualifier_authority_implementation_commit": qualifier_implementation_commit,
        "qualifier_config_sha256": _sha256_bytes(qualifier_payload),
        "qualifier_code_sha256": dict(sorted(qualifier_code_hashes.items())),
        "preflight_blob_sha256": dict(sorted(current_preflight_hashes.items())),
        "authority_sha256": dict(sorted(observed.items())),
    }


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    components = absolute.parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(components[0], flags)
    try:
        for component in components[1:]:
            if component in {"", ".", ".."}:
                raise ScopeError(f"{label} contains a non-canonical component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def audit_data_directory(data_root: Path | str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(data_root)))
    lowered = {part.casefold() for part in root.parts}
    if {"restricted", "sealed_external"} & lowered or "gse246381" in lowered:
        raise ScopeError("data root enters a forbidden scope")
    descriptor = _open_directory_nofollow(root, label="data root")
    try:
        before = os.fstat(descriptor)
        names = os.listdir(descriptor)
        expected_entries = protocol["data_directory_contract"]["expected_entries"]
        expected = {item["name"]: item for item in expected_entries}
        observed_names = set(names)
        if len(names) != len(observed_names):
            raise InventoryError("directory listing contains duplicate names")
        if observed_names != set(expected):
            missing = sorted(set(expected) - observed_names)
            extra = sorted(observed_names - set(expected))
            raise InventoryError(f"directory inventory mismatch: missing={missing!r}, extra={extra!r}")
        total_bytes = 0
        kinds: dict[str, int] = {}
        size_bindings: list[str] = []
        metadata_bindings: dict[str, tuple[int, ...]] = {}
        for name in sorted(names):
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise InventoryError(f"cannot stat expected entry {name}") from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise InventoryError(f"expected entry is not a single-link regular file: {name}")
            if metadata.st_size != expected[name]["bytes"]:
                raise InventoryError(f"size mismatch for {name}")
            total_bytes += metadata.st_size
            kind = expected[name]["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
            size_bindings.append(f"{name}\t{metadata.st_size}")
            metadata_bindings[name] = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        for name in sorted(names):
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise InventoryError(f"cannot restat expected entry {name}") from exc
            observed_binding = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if observed_binding != metadata_bindings[name]:
                raise InventoryError(f"expected entry changed during metadata audit: {name}")
        after = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise InventoryError("data directory changed during metadata audit")
        filename_set_sha256 = _sha256_bytes(("\n".join(sorted(names)) + "\n").encode("utf-8"))
        size_binding_sha256 = _sha256_bytes(("\n".join(size_bindings) + "\n").encode("utf-8"))
        return {
            "status": "PASS_METADATA_ONLY_STOP_BEFORE_DATA",
            "entry_count": len(names),
            "payload_asset_count": kinds.get("GEO_RAW_COUNT", 0) + kinds.get("SUPPLEMENT_WORKBOOK", 0),
            "geo_raw_count": kinds.get("GEO_RAW_COUNT", 0),
            "supplement_count": kinds.get("SUPPLEMENT_WORKBOOK", 0),
            "manifest_count": kinds.get("MANIFEST_METADATA_NOT_OPENED", 0),
            "total_lstat_bytes": total_bytes,
            "filename_set_sha256": filename_set_sha256,
            "name_size_binding_sha256": size_binding_sha256,
            "payload_open_count": 0,
            "manifest_open_count": 0,
            "payload_hash_count": 0,
            "hash_reverification": HASH_REVERIFICATION,
            "scientific_processing_count": 0,
        }
    finally:
        os.close(descriptor)


def _require_absent(path: Path, *, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PublicationError(f"cannot inspect {label}") from exc
    raise PublicationError(f"{label} already exists")


def audit_environment(
    output_path: Path,
    failure_path: Path,
    claim_path: Path,
    protocol: Mapping[str, Any],
    *,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    python_version: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    for path, label in ((output_path, "output"), (failure_path, "failure"), (claim_path, "claim")):
        _require_absent(path, label=label)
    if len({output_path, failure_path, claim_path}) != 3:
        raise PublicationError("output, failure, and claim targets must be distinct")
    parent = output_path.parent
    parent_fd = _open_directory_nofollow(parent, label="output parent")
    os.close(parent_fd)
    free_bytes = int(disk_usage(parent).free)
    minimum = int(protocol["environment_contract"]["minimum_output_free_bytes"])
    if free_bytes < minimum:
        raise EnvironmentError("insufficient output disk space")
    minimum_python = tuple(int(part) for part in protocol["environment_contract"]["minimum_python"].split("."))
    observed_python = python_version or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    if observed_python[:2] < minimum_python:
        raise EnvironmentError("Python version is below the frozen minimum")
    modules: dict[str, bool] = {}
    for name in protocol["environment_contract"]["required_python_modules"]:
        modules[name] = module_finder(name) is not None
    if not all(modules.values()):
        raise EnvironmentError("required Python module is unavailable")
    return {
        "status": "PASS_PREFLIGHT_ENVIRONMENT_ONLY",
        "python": ".".join(str(part) for part in observed_python),
        "required_modules": modules,
        "output_free_bytes": free_bytes,
        "minimum_output_free_bytes": minimum,
        "output_absent": True,
        "failure_absent": True,
        "claim_absent": True,
    }


def _historical_summary(protocol: Mapping[str, Any]) -> dict[str, Any]:
    r4 = protocol["historical_r4_closure"]
    return {
        "status": r4["qualification_report"]["status"],
        "bundle_path": r4["bundle_path"],
        "qualification_report_sha256": r4["qualification_report"]["sha256"],
        "sha256sums_sha256": r4["sha256sums"]["sha256"],
        "publication_commit_sha256": r4["publication_commit"]["sha256"],
        "reference_only_not_reopened": True,
        "rerun_is_qualification_path": False,
        "exact_blockers": list(EXPECTED_R4_BLOCKERS),
    }


def _external_evidence_summary(protocol: Mapping[str, Any]) -> dict[str, Any]:
    boundaries = protocol["scientific_evidence_boundaries"]
    return {
        "long_read_identity_method_status": boundaries["long_read_identity_method"]["status"],
        "exact_public_description_to_barcode_map": "BLOCKED",
        "pre_outcome_mapping_timing": "BLOCKED",
        "paper_exact_executable_method_source": "BLOCKED",
        "mann_whitney_implementation": "BLOCKED",
        "fdr_family_definition": "BLOCKED",
        "published_190_vs_180_status": "AUTHOR_ADJUDICATION_REQUIRED",
        "all_21_assets_license_status": "BLOCKED",
        "foundation_checkpoint_family_count": 4,
        "foundation_exact_overlap_closed_count": 0,
        "foundation_near_duplicate_closed_count": 0,
        "foundation_label_exposure_closed_count": 0,
    }


def _validate_document(
    document: Mapping[str, Any],
    *,
    authority_root: Mapping[str, Any] = PRODUCTION_AUTHORITY_ROOT,
) -> None:
    _exact_keys(document, {
        "schema_version", "protocol_id", "dataset_id", "recorded_at_utc", "outcome",
        "ready_for_study_qualification", "protocol_provenance", "authority_audit",
        "environment_audit", "inventory_audit", "historical_r4_closure",
        "external_evidence_audit", "blockers", "gate_truth", "counters", "claim_boundary",
    }, label="preflight document")
    _strict(document["schema_version"], SCHEMA_VERSION, label="output schema")
    _strict(document["protocol_id"], PROTOCOL_ID, label="output protocol")
    _strict(document["dataset_id"], DATASET_ID, label="output dataset")
    _strict(document["outcome"], TERMINAL_OUTCOME, label="output outcome")
    _strict(document["ready_for_study_qualification"], False, label="output readiness")
    timestamp = document["recorded_at_utc"]
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ProtocolError("output timestamp is not explicit UTC")
    try:
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ProtocolError("output timestamp is invalid") from exc

    provenance = _exact_keys(
        document["protocol_provenance"],
        {"path", "sha256", "bytes", "binding_model"},
        label="output protocol provenance",
    )
    if not isinstance(provenance["path"], str) or not os.path.isabs(provenance["path"]):
        raise ProtocolError("output protocol path is not absolute")
    if not isinstance(provenance["sha256"], str) or SHA256_RE.fullmatch(provenance["sha256"]) is None:
        raise ProtocolError("output protocol SHA-256 is invalid")
    if not isinstance(provenance["bytes"], int) or isinstance(provenance["bytes"], bool) or provenance["bytes"] < 1:
        raise ProtocolError("output protocol byte count is invalid")
    _strict(
        provenance["binding_model"],
        "IMMUTABLE_STATIC_BLOB_BOUND_FROM_QUALIFIER_CONFIG",
        label="output binding model",
    )

    authority = _exact_keys(
        document["authority_audit"],
        {
            "status", "git_head", "git_clean", "branch", "origin_branch_head",
            "accepted_a0_base_commit", "active_authority_commit", "active_amendment_decision_ids",
            "implementation_commit", "binding_commit",
            "binding_scheme", "qualifier_authority_implementation_commit", "qualifier_config_sha256",
            "qualifier_code_sha256", "preflight_blob_sha256", "authority_sha256",
        },
        label="output authority audit",
    )
    _strict(authority["status"], "PASS_CONFIG_ONLY_BINDING_VERIFIED", label="output authority status")
    _strict(authority["git_clean"], True, label="output Git cleanliness")
    _strict(authority["binding_scheme"], "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1", label="output binding scheme")
    _strict(authority["branch"], authority_root["expected_branch"], label="output branch")
    _strict(authority["origin_branch_head"], authority["git_head"], label="output origin branch head")
    _strict(
        authority["active_amendment_decision_ids"],
        authority_root["active_amendment_decision_ids"],
        label="output active decisions",
    )
    for key in (
        "git_head", "accepted_a0_base_commit", "active_authority_commit", "implementation_commit",
        "binding_commit", "qualifier_authority_implementation_commit",
    ):
        if not isinstance(authority[key], str) or COMMIT_RE.fullmatch(authority[key]) is None:
            raise ProtocolError(f"output authority commit is invalid: {key}")
    _strict(
        authority["accepted_a0_base_commit"],
        authority_root["accepted_a0_base_commit"],
        label="output accepted A0 commit",
    )
    _strict(
        authority["active_authority_commit"],
        authority_root["active_authority_commit"],
        label="output active authority commit",
    )
    _strict(
        authority["qualifier_authority_implementation_commit"],
        authority["implementation_commit"],
        label="output qualifier authority implementation commit",
    )
    if not isinstance(authority["qualifier_config_sha256"], str) or SHA256_RE.fullmatch(authority["qualifier_config_sha256"]) is None:
        raise ProtocolError("output qualifier config SHA-256 is invalid")
    preflight_hashes = _exact_keys(
        authority["preflight_blob_sha256"],
        {"external_evidence_config", "preflight_script", "preflight_test"},
        label="output preflight blob hashes",
    )
    qualifier_code_hashes = _exact_keys(
        authority["qualifier_code_sha256"],
        {"qualifier_script", "qualifier_test"},
        label="output qualifier code hashes",
    )
    authority_hashes = _exact_keys(
        authority["authority_sha256"],
        set(EXPECTED_AUTHORITY_PATHS),
        label="output authority hashes",
    )
    for key, value in {**preflight_hashes, **qualifier_code_hashes, **authority_hashes}.items():
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ProtocolError(f"output authority SHA-256 is invalid: {key}")

    environment = _exact_keys(
        document["environment_audit"],
        {
            "status", "python", "required_modules", "output_free_bytes",
            "minimum_output_free_bytes", "output_absent", "failure_absent", "claim_absent",
        },
        label="output environment audit",
    )
    _strict(environment["status"], "PASS_PREFLIGHT_ENVIRONMENT_ONLY", label="output environment status")
    _strict(environment["required_modules"], {"numpy": True, "openpyxl": True, "scipy": True}, label="output modules")
    _strict(environment["minimum_output_free_bytes"], 1073741824, label="output disk minimum")
    for key in ("output_absent", "failure_absent", "claim_absent"):
        _strict(environment[key], True, label=f"output {key}")
    if not isinstance(environment["output_free_bytes"], int) or environment["output_free_bytes"] < environment["minimum_output_free_bytes"]:
        raise ProtocolError("output free-byte audit is invalid")

    _strict(tuple(document["blockers"]), EXPECTED_R4_BLOCKERS, label="output blockers")
    _strict(document["gate_truth"], EXPECTED_GATE_TRUTH, label="output gate truth")
    _strict(document["counters"], {
        "payload_open_count": 0,
        "manifest_open_count": 0,
        "payload_hash_count": 0,
        "scientific_processing_count": 0,
        "qualifier_execution_count": 0,
        "canonical_record_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
    }, label="output counters")
    _strict(document["claim_boundary"], EXPECTED_CLAIM_BOUNDARY, label="output claim boundary")
    inventory = _exact_keys(
        document["inventory_audit"],
        {
            "status", "entry_count", "payload_asset_count", "geo_raw_count", "supplement_count",
            "manifest_count", "total_lstat_bytes", "filename_set_sha256", "name_size_binding_sha256",
            "payload_open_count", "manifest_open_count", "payload_hash_count", "hash_reverification",
            "scientific_processing_count",
        },
        label="output inventory audit",
    )
    _strict(inventory["status"], "PASS_METADATA_ONLY_STOP_BEFORE_DATA", label="output inventory status")
    for key, expected in {
        "entry_count": 22,
        "payload_asset_count": 21,
        "geo_raw_count": 18,
        "supplement_count": 3,
        "manifest_count": 1,
        "payload_open_count": 0,
        "manifest_open_count": 0,
        "payload_hash_count": 0,
        "hash_reverification": HASH_REVERIFICATION,
        "scientific_processing_count": 0,
    }.items():
        if inventory.get(key) != expected:
            raise ProtocolError(f"output inventory field drift: {key}")
    if not isinstance(inventory["total_lstat_bytes"], int) or inventory["total_lstat_bytes"] < 1:
        raise ProtocolError("output inventory total bytes are invalid")
    for key in ("filename_set_sha256", "name_size_binding_sha256"):
        if not isinstance(inventory[key], str) or SHA256_RE.fullmatch(inventory[key]) is None:
            raise ProtocolError(f"output inventory digest is invalid: {key}")

    _strict(
        document["historical_r4_closure"],
        {
            "status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "bundle_path": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_GSE149487_PLUMAGE_FULL_QUAL_20260810T131156P0800_a859166_R4",
            "qualification_report_sha256": "19df844b55ef7b8dbf53ba3044a51132bdea1f0d1dfa6809a720a2a83a7030b3",
            "sha256sums_sha256": "c72c63c2090052657beaa797e3ba3196200f8cbc3e9c5a97cf1a4a04a4db3631",
            "publication_commit_sha256": "3149001644cf1b21db74021b12ca1e887977a9d0d13deff3b2f57b18e4b64ca4",
            "reference_only_not_reopened": True,
            "rerun_is_qualification_path": False,
            "exact_blockers": list(EXPECTED_R4_BLOCKERS),
        },
        label="output historical R4 closure",
    )
    _strict(
        document["external_evidence_audit"],
        {
            "long_read_identity_method_status": "VERIFIED_METHOD_SURFACE_ONLY",
            "exact_public_description_to_barcode_map": "BLOCKED",
            "pre_outcome_mapping_timing": "BLOCKED",
            "paper_exact_executable_method_source": "BLOCKED",
            "mann_whitney_implementation": "BLOCKED",
            "fdr_family_definition": "BLOCKED",
            "published_190_vs_180_status": "AUTHOR_ADJUDICATION_REQUIRED",
            "all_21_assets_license_status": "BLOCKED",
            "foundation_checkpoint_family_count": 4,
            "foundation_exact_overlap_closed_count": 0,
            "foundation_near_duplicate_closed_count": 0,
            "foundation_label_exposure_closed_count": 0,
        },
        label="output external evidence audit",
    )


def _publication_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
    )


def _publish_exclusive(
    path: Path,
    payload: bytes,
    *,
    link_fn: Callable[..., Any] = os.link,
    post_link_unlink_fn: Callable[..., Any] = os.unlink,
    post_link_fsync_fn: Callable[[int], Any] = os.fsync,
    post_link_close_fn: Callable[[int], Any] = os.close,
) -> dict[str, Any]:
    """Atomically commit a complete file and never report ordinary failure afterward."""

    parent_fd = _open_directory_nofollow(path.parent, label="output parent")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    temporary_name = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(12)}"
    temporary_created = False
    committed = False
    try:
        descriptor = os.open(temporary_name, flags, 0o640, dir_fd=parent_fd)
        temporary_created = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise PublicationError("short write while publishing preflight")
            offset += written
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed_digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed_digest.update(chunk)
            observed_bytes += len(chunk)
        if observed_bytes != len(payload) or observed_digest.hexdigest() != _sha256_bytes(payload):
            raise PublicationError("fsynced temporary output content verification failed")
        descriptor_identity = _publication_identity(os.fstat(descriptor))
        named_identity = _publication_identity(
            os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if descriptor_identity != named_identity:
            raise PublicationError("temporary output name no longer identifies the fsynced file")
        try:
            link_fn(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise PublicationError("output already exists") from exc
        committed = True
    except BaseException:
        if committed:
            raise AssertionError("post-commit exception escaped the commit-point boundary")
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass
        raise

    warnings: list[str] = []
    final_identity_verified = False
    try:
        final_identity = _publication_identity(
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        )
        final_identity_verified = final_identity == descriptor_identity
        if not final_identity_verified:
            warnings.append("FINAL_IDENTITY_MISMATCH_AFTER_COMMIT")
    except OSError:
        warnings.append("FINAL_IDENTITY_VERIFICATION_FAILED_AFTER_COMMIT")

    try:
        post_link_unlink_fn(temporary_name, dir_fd=parent_fd)
        temporary_created = False
    except OSError:
        warnings.append("TEMPORARY_NAME_UNLINK_FAILED_AFTER_COMMIT")

    def close_after_commit(fd: int, warning_code: str) -> None:
        try:
            post_link_close_fn(fd)
        except OSError:
            warnings.append(warning_code)
            try:
                os.close(fd)
            except OSError:
                pass

    if descriptor is not None:
        close_after_commit(descriptor, "TEMPORARY_FD_CLOSE_FAILED_AFTER_COMMIT")
        descriptor = None
    try:
        post_link_fsync_fn(parent_fd)
    except OSError:
        warnings.append("PARENT_DIRECTORY_FSYNC_FAILED_AFTER_COMMIT")
    close_after_commit(parent_fd, "PARENT_DIRECTORY_FD_CLOSE_FAILED_AFTER_COMMIT")

    if not final_identity_verified:
        status_value = "COMMITTED_NOT_ACCEPTED"
    elif warnings:
        status_value = "COMMITTED_WITH_POST_COMMIT_WARNING"
    else:
        status_value = "COMMITTED_AND_ACCEPTED"
    return {
        "status": status_value,
        "commit_point": "ATOMIC_HARD_LINK_AFTER_FSYNCED_FD_CONTENT_VERIFICATION",
        "final_identity_verified": final_identity_verified,
        "payload_sha256": _sha256_bytes(payload),
        "warnings": warnings,
    }


def run_preflight(
    *,
    repo_root: Path | str,
    protocol_path: Path | str,
    data_root: Path | str,
    output_path: Path | str,
    failure_path: Path | str,
    claim_path: Path | str,
    recorded_at_utc: str | None = None,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    python_version: tuple[int, int, int] | None = None,
    test_only_authority_root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    caller_paths = _pre_open_lexical_scope(
        repo_root=repo_root,
        protocol_path=protocol_path,
        data_root=data_root,
        output_path=output_path,
        failure_path=failure_path,
        claim_path=claim_path,
    )
    authority_root = PRODUCTION_AUTHORITY_ROOT if test_only_authority_root is None else test_only_authority_root
    protocol, protocol_provenance = load_protocol(
        caller_paths["protocol_path"],
        test_only_authority_root=test_only_authority_root,
    )
    # This config-only gate intentionally precedes every data-root/output operation.
    root, qualifier_config, qualifier_payload, bound_binding = _load_bound_qualifier_before_data(
        caller_paths["repo_root"],
        protocol,
    )
    authority_audit = audit_repo_authority(
        root,
        protocol,
        qualifier_config=qualifier_config,
        qualifier_payload=qualifier_payload,
        bound_binding=bound_binding,
        protocol_sha256=protocol_provenance["sha256"],
        authority_root=authority_root,
    )

    output = caller_paths["output_path"]
    failure = caller_paths["failure_path"]
    claim = caller_paths["claim_path"]
    environment_audit = audit_environment(
        output,
        failure,
        claim,
        protocol,
        module_finder=module_finder,
        disk_usage=disk_usage,
        python_version=python_version,
    )
    inventory_audit = audit_data_directory(caller_paths["data_root"], protocol)
    timestamp = recorded_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ProtocolError("recorded_at_utc must be an explicit UTC Z timestamp")
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "recorded_at_utc": timestamp,
        "outcome": TERMINAL_OUTCOME,
        "ready_for_study_qualification": False,
        "protocol_provenance": protocol_provenance,
        "authority_audit": authority_audit,
        "environment_audit": environment_audit,
        "inventory_audit": inventory_audit,
        "historical_r4_closure": _historical_summary(protocol),
        "external_evidence_audit": _external_evidence_summary(protocol),
        "blockers": list(EXPECTED_R4_BLOCKERS),
        "gate_truth": dict(EXPECTED_GATE_TRUTH),
        "counters": {
            "payload_open_count": 0,
            "manifest_open_count": 0,
            "payload_hash_count": 0,
            "scientific_processing_count": 0,
            "qualifier_execution_count": 0,
            "canonical_record_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
        },
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }
    _validate_document(document, authority_root=authority_root)
    publication_receipt = _publish_exclusive(output, _json_bytes(document))
    if publication_receipt["status"] == "COMMITTED_NOT_ACCEPTED":
        raise CommittedPublicationNotAccepted(output, publication_receipt)
    if publication_receipt["status"] == "COMMITTED_WITH_POST_COMMIT_WARNING":
        try:
            print(
                json.dumps(
                    {
                        "publication_status": publication_receipt["status"],
                        "output": str(output),
                        "warnings": publication_receipt["warnings"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except OSError:
            pass
    return document


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--recorded-at-utc")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        document = run_preflight(
            repo_root=args.repo_root,
            protocol_path=args.protocol,
            data_root=args.data_root,
            output_path=args.output,
            failure_path=args.failure,
            claim_path=args.claim,
            recorded_at_utc=args.recorded_at_utc,
        )
    except CommittedPublicationNotAccepted as exc:
        print(
            json.dumps(
                {
                    "outcome": "COMMITTED_NOT_ACCEPTED",
                    "output": str(exc.path),
                    "warnings": exc.receipt["warnings"],
                },
                sort_keys=True,
            )
        )
        return 3
    except PreflightError as exc:
        print(json.dumps({"outcome": "FAIL_CLOSED", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"outcome": document["outcome"], "output": os.path.abspath(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
