#!/usr/bin/env python3
"""Acquire the frozen GSE149487 exact-21 ordinary-public asset set.

This producer consumes metadata only: three repository JSON authorities and
the historical R4 ``ASSET_MANIFEST_EFFECTIVE.json`` aggregate.  It downloads
the exact 18 GEO raw-count objects plus MOESM3, MOESM8, and Lim6c without
parsing their contents.  Every new file must match the R4 byte count and
SHA-256 before it is renamed from ``.part`` to its final basename.

The sole metadata output is one aggregate acquisition report.  Even after all
21 files pass integrity verification, that report remains
``STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER``: it does not invoke the full
qualifier, materialize canonical records, download models, or authorize any
scientific or model work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "route_a_v3_gse149487_public_asset_acquisition.v1"
PROTOCOL_ID = "GSE149487_PUBLIC_ASSET_ACQUISITION_V1"
PROTOCOL_BASENAME = "route_a_v3_gse149487_public_asset_acquisition_v1.json"
REPORT_FILENAME = "GSE149487_PUBLIC_ASSET_ACQUISITION_V1.json"
DATASET_ID = "GSE149487"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
OUTPUT_BASENAME_RE = re.compile(
    r"^GSE149487_PUBLIC_ASSETS_[A-Za-z0-9][A-Za-z0-9._-]{7,95}$"
)

EXPECTED_BINDING_KEYS = frozenset(
    {
        "binding_scheme",
        "status",
        "base_commit",
        "implementation_commit",
        "implementation_script_path",
        "implementation_script_sha256",
        "implementation_test_path",
        "implementation_test_sha256",
        "implementation_changed_paths",
        "binding_commit_allowed_changed_paths",
        "unknown_to_bound_scalar_paths",
        "activation_rule",
    }
)
UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
EXPECTED_TERMINAL_TRUTH = {
    "status_after_exact21_acquisition": "STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER",
    "asset_integrity_may_be_confirmed": True,
    "ready_for_full_qualifier_input": False,
    "ready_for_study_qualification": False,
    "qualified": False,
    "canonical_record_count": 0,
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "training_allowed": False,
    "model_selection_allowed": False,
    "next_phase_authorized": False,
    "qualifier_invoked": False,
    "model_downloaded": False,
}
EXPECTED_RETAINED_BLOCKERS = (
    "CHECKPOINT_SPECIFIC_FOUNDATION_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "GEO_RAW_DATA_SPECIFIC_REDISTRIBUTION_GRANT_ABSENT_PRIVATE_LOCATOR_HASH_ONLY",
    "LIM6C_EXPLICIT_LICENSE_ABSENT",
    "OUTCOME_BLIND_LONG_READ_DESCRIPTION_TO_BARCODE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    "OUTCOME_BLIND_LONG_READ_MAPPING_PRE_OUTCOME_TIMING_UNKNOWN_NOT_ASSERTED",
    "PAPER_NATIVE_METHOD_NOT_REPRODUCED",
    "PAPER_NATIVE_EXACT_MULTIPLE_TESTING_FAMILY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_OR_CI_GATE_FAILED",
    "PUBLISHED_190_VS_UNIQUE_PAIR_180_ADJUDICATION_UNKNOWN_NOT_ASSERTED",
    "RAW_KEY_UNCLASSIFIED_OUTCOME_BLIND_RECONCILIATION_NOT_ZERO",
    "UNADJUDICATED_OR_AMBIGUOUS_MAPPING_ROWS_PRESENT",
    "UNADJUDICATED_SEQUENCE_UNIVERSE_CLASSES_PRESENT",
)


class AcquisitionError(RuntimeError):
    """Base class for a fail-closed acquisition error."""


class ProtocolError(AcquisitionError):
    """The protocol or implementation binding is not the frozen authority."""


class BindingNotFrozen(ProtocolError):
    """The normal UNKNOWN-I to config-only-B lifecycle is not complete."""


class AuthorityError(AcquisitionError):
    """A metadata authority differs from its action-changing frozen facts."""


class OutputScopeError(AcquisitionError):
    """The requested target is outside the exclusive public-asset root."""


class TransportError(AcquisitionError):
    """An exact frozen HTTPS object could not be downloaded."""


class IntegrityError(AcquisitionError):
    """A downloaded object differs in bytes or SHA-256 from R4."""


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    asset_kind: str
    filename: str
    expected_bytes: int
    expected_sha256: str
    source_uri: str
    context: str | None = None
    assay: str | None = None
    biological_replicate: int | None = None


class HTTPResponse(Protocol):
    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class DownloadTransport(Protocol):
    def open(self, url: str, *, timeout_seconds: int) -> HTTPResponse: ...


class UrllibTransport:
    """Read the exact HTTPS locator frozen in R4 using the standard library."""

    def open(self, url: str, *, timeout_seconds: int) -> HTTPResponse:
        request = urllib.request.Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "mRNA-XEditFlow-GSE149487-public-asset-acquirer/1",
            },
            method="GET",
        )
        try:
            return urllib.request.urlopen(  # noqa: S310 - URL is exact hash-bound R4 metadata
                request, timeout=timeout_seconds
            )  # type: ignore[return-value]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise TransportError(f"HTTPS acquisition failed for {url}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
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
        raise AcquisitionError("aggregate report is not finite JSON") from exc


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> Any:
        raise ValueError(f"non-finite token {token}")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise AuthorityError(f"{label} must be a JSON object")
    return result


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be read: {path}") from exc


def _read_bound_json(
    path: Path, *, label: str, expected_bytes: int, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_bytes(path, label=label)
    observed_sha256 = _sha256(payload)
    if len(payload) != expected_bytes:
        raise AuthorityError(f"{label} byte count mismatch")
    if observed_sha256 != expected_sha256:
        raise AuthorityError(f"{label} SHA-256 mismatch")
    return _strict_json_object(payload, label=label), {
        "path": str(path),
        "bytes": len(payload),
        "sha256": observed_sha256,
        "status": "EXACT_METADATA_AUTHORITY_VERIFIED",
    }


def _load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise OutputScopeError("protocol basename is outside the frozen allowlist")
    payload = _read_bytes(path, label="acquisition protocol")
    protocol = _strict_json_object(payload, label="acquisition protocol")
    _validate_protocol(protocol)
    return protocol, payload


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "ACQUISITION_ONLY_STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "dataset_alias": "PLUMAGE",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"{key} differs from the frozen protocol")

    binding = protocol.get("implementation_binding")
    if not isinstance(binding, dict) or set(binding) != EXPECTED_BINDING_KEYS:
        raise ProtocolError("implementation_binding schema differs from the frozen lifecycle")
    if binding.get("binding_scheme") != "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1":
        raise ProtocolError("implementation binding scheme is not config-only-B")
    if COMMIT_RE.fullmatch(str(binding.get("base_commit"))) is None:
        raise ProtocolError("base commit is invalid")
    if binding.get("status") not in {"UNKNOWN_NOT_ASSERTED", "BOUND"}:
        raise ProtocolError("implementation binding status is outside the closed enum")
    if binding.get("status") == "UNKNOWN_NOT_ASSERTED":
        for key in UNKNOWN_BINDING_SCALARS:
            if binding.get(key) != "UNKNOWN_NOT_ASSERTED":
                raise ProtocolError("UNKNOWN implementation binding is not exact")
    else:
        if COMMIT_RE.fullmatch(str(binding.get("implementation_commit"))) is None:
            raise ProtocolError("bound implementation commit is invalid")
        for key in ("implementation_script_sha256", "implementation_test_sha256"):
            if SHA256_RE.fullmatch(str(binding.get(key))) is None:
                raise ProtocolError(f"bound {key} is invalid")
    if binding.get("implementation_changed_paths") != [
        "configs/route_a_v3_gse149487_public_asset_acquisition_v1.json",
        "scripts/route_a_v3/acquire_gse149487_public_assets.py",
        "tests/route_a_v3/test_acquire_gse149487_public_assets.py",
    ]:
        raise ProtocolError("implementation changed paths differ from the three-file scope")
    if binding.get("binding_commit_allowed_changed_paths") != [
        "configs/route_a_v3_gse149487_public_asset_acquisition_v1.json"
    ]:
        raise ProtocolError("binding commit is not config-only")
    if binding.get("unknown_to_bound_scalar_paths") != [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ]:
        raise ProtocolError("unknown-to-bound scalar set differs from the lifecycle")

    asset_contract = protocol.get("asset_contract")
    if not isinstance(asset_contract, dict):
        raise ProtocolError("asset_contract is absent")
    required_asset_truth = {
        "asset_specification_source": "HISTORICAL_R4_ASSET_MANIFEST_EFFECTIVE_ONLY",
        "expected_asset_count": 21,
        "expected_geo_raw_count": 18,
        "expected_supplement_count": 3,
        "per_asset_exact_bytes_required": True,
        "per_asset_exact_sha256_required": True,
        "payload_parse_allowed": False,
        "row_sequence_effect_read_allowed": False,
        "qualifier_execution_allowed": False,
        "model_download_allowed": False,
        "scientific_processing_allowed": False,
    }
    for key, expected in required_asset_truth.items():
        if asset_contract.get(key) != expected:
            raise ProtocolError(f"asset_contract.{key} differs from frozen truth")
    if (
        not isinstance(asset_contract.get("expected_total_payload_bytes"), int)
        or asset_contract["expected_total_payload_bytes"] <= 0
    ):
        raise ProtocolError("asset_contract.expected_total_payload_bytes is invalid")

    output = protocol.get("output_contract")
    if not isinstance(output, dict):
        raise ProtocolError("output_contract is absent")
    if output.get("report_filename") != REPORT_FILENAME:
        raise ProtocolError("aggregate report filename differs from the protocol")
    for key, expected in {
        "first_execution_requires_nonexistent_subdirectory": True,
        "single_aggregate_report_only": True,
        "sha256sums_file_written": False,
        "terminal_marker_written": False,
    }.items():
        if output.get(key) != expected:
            raise ProtocolError(f"output_contract.{key} differs from frozen truth")

    if protocol.get("terminal_truth") != EXPECTED_TERMINAL_TRUTH:
        raise ProtocolError("terminal truth may not be upgraded by acquisition")
    if tuple(protocol.get("retained_blockers", ())) != EXPECTED_RETAINED_BLOCKERS:
        raise ProtocolError("retained public-evidence blockers differ from frozen truth")

    confirmed = protocol.get("confirmed_public_evidence")
    unknown = protocol.get("unknown_not_asserted")
    if not isinstance(confirmed, dict) or not isinstance(unknown, dict):
        raise ProtocolError("confirmed and unknown evidence partitions are required")
    method = confirmed.get("paper_jats_method_surface")
    if method != {
        "status": "CONFIRMED_METHOD_SURFACE_ONLY",
        "original_cpm_minimum_inclusive": 0.5,
        "test_sidedness": "TWO_SIDED",
        "test_family": "MANN_WHITNEY_U",
        "reported_r_call": "wilcox.test",
        "multiple_testing_adjustment": "FDR",
        "significance_rule": "FDR_LT_0.1",
    }:
        raise ProtocolError("paper JATS method-surface truth differs from the evidence")
    if confirmed.get("license_and_redistribution") != {
        "paper_and_moesm3_moesm8_license": (
            "CC_BY_4_0_CONFIRMED_FOR_PAPER_AND_PAPER_SUPPLEMENTS_ONLY"
        ),
        "geo_raw_18": (
            "PRIVATE_CANONICAL_LOCATOR_HASH_USE_ONLY_NO_DATA_SPECIFIC_"
            "REDISTRIBUTION_GRANT"
        ),
        "lim6c_github_blob": "NO_EXPLICIT_LICENSE_CONFIRMED",
    }:
        raise ProtocolError("license and redistribution truth differs from the evidence")
    if set(unknown.values()) != {"UNKNOWN_NOT_ASSERTED"}:
        raise ProtocolError("unclosed evidence must remain UNKNOWN_NOT_ASSERTED")


def _run_git(repo_root: Path, *args: str, check: bool = True) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for implementation binding") from exc
    if check and completed.returncode != 0:
        raise ProtocolError("git implementation-binding check failed")
    return completed.stdout


def _commit_parents(repo_root: Path, commit: str) -> list[str]:
    fields = _run_git(repo_root, "rev-list", "--parents", "-n", "1", commit).decode(
        "ascii"
    ).strip().split()
    if not fields or fields[0] != commit:
        raise ProtocolError("git did not resolve the expected commit")
    return fields[1:]


def _changed_paths(repo_root: Path, commit: str) -> list[str]:
    return sorted(
        line
        for line in _run_git(
            repo_root,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit,
        )
        .decode("utf-8")
        .splitlines()
        if line
    )


def _audit_implementation_binding(
    protocol: Mapping[str, Any], protocol_path: Path, protocol_payload: bytes
) -> dict[str, Any]:
    binding = protocol["implementation_binding"]
    if binding["status"] != "BOUND":
        raise BindingNotFrozen(
            "implementation binding is UNKNOWN_NOT_ASSERTED; finish config-only-B first"
        )

    repo_root = protocol_path.parent.parent
    relative_protocol = protocol_path.relative_to(repo_root).as_posix()
    implementation = str(binding["implementation_commit"])
    base_commit = str(binding["base_commit"])
    implementation_paths = sorted(binding["implementation_changed_paths"])
    binding_paths = sorted(binding["binding_commit_allowed_changed_paths"])

    if _commit_parents(repo_root, implementation) != [base_commit]:
        raise ProtocolError("implementation commit is not the direct child of base_commit")
    if _changed_paths(repo_root, implementation) != implementation_paths:
        raise ProtocolError("implementation commit does not have the exact three-file scope")

    commits_after_i = [
        line
        for line in _run_git(
            repo_root,
            "rev-list",
            "--ancestry-path",
            "--reverse",
            f"{implementation}..HEAD",
        )
        .decode("ascii")
        .splitlines()
        if line
    ]
    if not commits_after_i:
        raise BindingNotFrozen("config-only binding commit B is absent")
    binding_commit = commits_after_i[0]
    if _commit_parents(repo_root, binding_commit) != [implementation]:
        raise ProtocolError("binding commit is not the direct child of implementation")
    if _changed_paths(repo_root, binding_commit) != binding_paths:
        raise ProtocolError("binding commit changed more than the protocol config")

    worktree_status = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *implementation_paths,
    )
    if worktree_status:
        raise ProtocolError("acquisition config, producer, or focused test is dirty")

    implementation_config = _run_git(
        repo_root, "show", f"{implementation}:{relative_protocol}"
    )
    expected_unknown = copy.deepcopy(dict(protocol))
    for key in UNKNOWN_BINDING_SCALARS:
        expected_unknown["implementation_binding"][key] = "UNKNOWN_NOT_ASSERTED"
    if _strict_json_object(
        implementation_config, label="implementation-commit acquisition protocol"
    ) != expected_unknown:
        raise ProtocolError("binding commit changed fields outside the four scalar bindings")
    if _run_git(repo_root, "show", f"{binding_commit}:{relative_protocol}") != protocol_payload:
        raise ProtocolError("current protocol bytes differ from config-only binding commit B")

    for path_key, sha_key in (
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        relative = str(binding[path_key])
        expected_sha256 = str(binding[sha_key])
        committed_payload = _run_git(repo_root, "show", f"{implementation}:{relative}")
        if _sha256(committed_payload) != expected_sha256:
            raise ProtocolError(f"{relative} implementation-commit SHA-256 mismatch")
        current_payload = _read_bytes(repo_root / relative, label=relative)
        if _sha256(current_payload) != expected_sha256:
            raise ProtocolError(f"{relative} current SHA-256 mismatch")

    return {
        "status": "BOUND_CONFIG_ONLY_LIFECYCLE_VERIFIED",
        "base_commit": base_commit,
        "implementation_commit": implementation,
        "binding_commit": binding_commit,
        "current_head": _run_git(repo_root, "rev-parse", "HEAD")
        .decode("ascii")
        .strip(),
    }


def _authority_path(repo_root: Path, entry: Mapping[str, Any]) -> Path:
    raw = Path(str(entry["path"]))
    return raw if raw.is_absolute() else repo_root / raw


def _load_authorities(
    protocol: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    authority = protocol.get("metadata_authorities")
    if not isinstance(authority, dict):
        raise ProtocolError("metadata_authorities is absent")
    documents: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for key in (
        "asset_manifest",
        "external_evidence_roots",
        "a1_qualification",
        "historical_r4_effective_asset_manifest",
    ):
        entry = authority.get(key)
        if not isinstance(entry, dict):
            raise ProtocolError(f"metadata_authorities.{key} is absent")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise ProtocolError(f"metadata_authorities.{key}.bytes is invalid")
        if SHA256_RE.fullmatch(str(expected_sha256)) is None:
            raise ProtocolError(f"metadata_authorities.{key}.sha256 is invalid")
        documents[key], records[key] = _read_bound_json(
            _authority_path(repo_root, entry),
            label=key,
            expected_bytes=expected_bytes,
            expected_sha256=str(expected_sha256),
        )
    return documents, records


def _asset_spec(record: Mapping[str, Any]) -> AssetSpec:
    required = ("asset_id", "asset_kind", "filename", "bytes", "sha256", "source_uri")
    if any(key not in record for key in required):
        raise AuthorityError("R4 asset is missing an acquisition field")
    filename = str(record["filename"])
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise AuthorityError("R4 asset filename is not a basename")
    expected_bytes = record["bytes"]
    expected_sha256 = str(record["sha256"])
    source_uri = str(record["source_uri"])
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise AuthorityError(f"R4 byte count is invalid for {filename}")
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise AuthorityError(f"R4 SHA-256 is invalid for {filename}")
    if not source_uri.startswith("https://"):
        raise AuthorityError(f"R4 source is not HTTPS for {filename}")
    kind = str(record["asset_kind"])
    if kind not in {"GEO_RAW_COUNT", "SUPPLEMENT_WORKBOOK"}:
        raise AuthorityError(f"R4 asset kind is invalid for {filename}")
    replicate = record.get("biological_replicate")
    if replicate is not None and not isinstance(replicate, int):
        raise AuthorityError(f"R4 biological replicate is invalid for {filename}")
    return AssetSpec(
        asset_id=str(record["asset_id"]),
        asset_kind=kind,
        filename=filename,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        source_uri=source_uri,
        context=str(record["context"]) if "context" in record else None,
        assay=str(record["assay"]) if "assay" in record else None,
        biological_replicate=replicate,
    )


def _validate_authorities(
    protocol: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> list[AssetSpec]:
    asset_manifest = documents["asset_manifest"]
    external = documents["external_evidence_roots"]
    qualification = documents["a1_qualification"]
    r4 = documents["historical_r4_effective_asset_manifest"]
    contract = protocol["asset_contract"]

    if r4.get("dataset_id") != DATASET_ID or r4.get("all_input_hashes_verified") is not True:
        raise AuthorityError("R4 does not establish historical exact-input integrity")
    raw_assets = r4.get("assets")
    if not isinstance(raw_assets, list):
        raise AuthorityError("R4 assets must be a list")
    assets = [_asset_spec(record) for record in raw_assets if isinstance(record, dict)]
    if len(assets) != len(raw_assets):
        raise AuthorityError("R4 contains a non-object asset")
    if len(assets) != contract["expected_asset_count"] or r4.get("asset_count") != len(assets):
        raise AuthorityError("R4 exact asset count is not 21")
    if len({asset.asset_id for asset in assets}) != len(assets):
        raise AuthorityError("R4 asset IDs are not unique")
    if len({asset.filename for asset in assets}) != len(assets):
        raise AuthorityError("R4 filenames are not unique")
    if sum(asset.expected_bytes for asset in assets) != contract["expected_total_payload_bytes"]:
        raise AuthorityError("R4 total payload bytes differ from the frozen exact-21 total")

    geo = [asset for asset in assets if asset.asset_kind == "GEO_RAW_COUNT"]
    supplements = [
        asset for asset in assets if asset.asset_kind == "SUPPLEMENT_WORKBOOK"
    ]
    if len(geo) != contract["expected_geo_raw_count"]:
        raise AuthorityError("R4 GEO raw-count asset count is not 18")
    if len(supplements) != contract["expected_supplement_count"]:
        raise AuthorityError("R4 supplement asset count is not 3")
    expected_grid = {
        (context, assay, replicate)
        for context in ("PC3", "293T")
        for assay in ("DNA", "POLYSOME", "TOTALRNA")
        for replicate in (1, 2, 3)
    }
    observed_grid = {
        (asset.context, asset.assay, asset.biological_replicate) for asset in geo
    }
    if observed_grid != expected_grid:
        raise AuthorityError("R4 GEO context-assay-replicate grid is not exact 2x3x3")
    if {asset.asset_id for asset in supplements} != {
        "GSE149487_MOESM3",
        "GSE149487_MOESM8",
        "GSE149487_LIM6C_293T",
    }:
        raise AuthorityError("R4 supplement identity set is not exact")

    manifest_records = asset_manifest.get("assets")
    if (
        asset_manifest.get("dataset_id") != DATASET_ID
        or asset_manifest.get("expected_asset_count") != 21
        or not isinstance(manifest_records, list)
    ):
        raise AuthorityError("repository asset manifest is not the GSE149487 exact-21 authority")
    manifest_by_id = {
        str(record.get("asset_id")): record
        for record in manifest_records
        if isinstance(record, dict)
    }
    if set(manifest_by_id) != {asset.asset_id for asset in assets}:
        raise AuthorityError("repository asset manifest and R4 asset IDs differ")
    for asset in assets:
        record = manifest_by_id[asset.asset_id]
        if record.get("asset_kind") != asset.asset_kind:
            raise AuthorityError(f"asset kind differs for {asset.asset_id}")
        if asset.asset_kind == "GEO_RAW_COUNT":
            if (
                record.get("context"),
                record.get("assay"),
                record.get("biological_replicate"),
            ) != (asset.context, asset.assay, asset.biological_replicate):
                raise AuthorityError(f"GEO slot differs for {asset.asset_id}")
        else:
            for key, observed in (
                ("filename", asset.filename),
                ("bytes", asset.expected_bytes),
                ("sha256", asset.expected_sha256),
                ("source_uri", asset.source_uri),
            ):
                if record.get(key) != observed:
                    raise AuthorityError(f"supplement {key} differs for {asset.asset_id}")

    metadata = protocol["metadata_authorities"]
    if external.get("dataset_id") != DATASET_ID:
        raise AuthorityError("external evidence root dataset differs")
    external_binding = external.get("authority_bindings", {}).get("asset_manifest", {})
    if external_binding.get("sha256") != metadata["asset_manifest"]["sha256"]:
        raise AuthorityError("external evidence root does not bind the asset manifest")
    historical = external.get("historical_r4_closure", {})
    r4_path = str(Path(metadata["historical_r4_effective_asset_manifest"]["path"]).parent)
    if historical.get("bundle_path") != r4_path:
        raise AuthorityError("external evidence root points to a different R4 bundle")
    if historical.get("qualification_report", {}).get("qualified") is not False:
        raise AuthorityError("historical R4 qualification truth was upgraded")
    if historical.get("reuse_policy") != "REFERENCE_AGGREGATE_ONLY_DO_NOT_REOPEN_OR_REHASH":
        raise AuthorityError("historical R4 reuse policy differs")
    license_boundary = external.get("scientific_evidence_boundaries", {}).get(
        "license_boundary", {}
    )
    if license_boundary != {
        "moesm3_and_moesm8": "CC_BY_4_0_ONLY",
        "geo_raw_18": "NONREDISTRIBUTABLE_LOCATOR_HASH_ONLY",
        "lim6c": "NO_EXPLICIT_LICENSE",
        "all_21_assets_license_status": "BLOCKED",
        "qualification_effect": "BLOCK",
    }:
        raise AuthorityError("external license boundary differs from confirmed truth")

    qualification_authority = qualification.get("authority", {})
    if qualification_authority.get("asset_manifest_sha256") != metadata["asset_manifest"]["sha256"]:
        raise AuthorityError("A1 qualification does not bind the asset manifest")
    scope = qualification.get("scope", {})
    if scope.get("full_raw_geo_table_count") != 18 or scope.get("supplement_count") != 3:
        raise AuthorityError("A1 qualification exact-21 scope differs")
    gate = qualification.get("current_gate_contract", {})
    required_gate = {
        "qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "next_phase_authorized": False,
    }
    if gate != required_gate:
        raise AuthorityError("A1 qualification gate truth differs from the stopped authority")
    return assets


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_output_path(protocol: Mapping[str, Any], output_directory: Path) -> Path:
    output = _absolute(output_directory)
    base = _absolute(Path(str(protocol["output_contract"]["base_directory"])))
    if output.parent != base:
        raise OutputScopeError("output must be one direct child of the GSE149487 data root")
    regex = str(protocol["output_contract"]["subdirectory_basename_regex"])
    if re.fullmatch(regex, output.name) is None or OUTPUT_BASENAME_RE.fullmatch(output.name) is None:
        raise OutputScopeError("output basename does not follow GSE149487 public-asset convention")
    if output.exists():
        raise OutputScopeError("exclusive output directory already exists")
    return output


def _download_one(
    asset: AssetSpec,
    output_directory: Path,
    *,
    transport: DownloadTransport,
    timeout_seconds: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    final_path = output_directory / asset.filename
    partial_path = output_directory / f"{asset.filename}.part"
    if final_path.exists() or partial_path.exists():
        raise OutputScopeError(f"asset target already exists: {asset.filename}")

    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        response = transport.open(asset.source_uri, timeout_seconds=timeout_seconds)
        with closing(response), partial_path.open("xb") as output:
            while True:
                block = response.read(chunk_bytes)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                observed_bytes += len(block)
            output.flush()
            os.fsync(output.fileno())
    except AcquisitionError:
        raise
    except (
        OSError,
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
    ) as exc:
        raise TransportError(f"download failed for {asset.asset_id}") from exc

    observed_sha256 = digest.hexdigest()
    if observed_bytes != asset.expected_bytes:
        raise IntegrityError(
            f"byte count mismatch for {asset.asset_id}: "
            f"expected {asset.expected_bytes}, observed {observed_bytes}"
        )
    if observed_sha256 != asset.expected_sha256:
        raise IntegrityError(f"SHA-256 mismatch for {asset.asset_id}")
    try:
        partial_path.rename(final_path)
    except OSError as exc:
        raise AcquisitionError(f"verified asset rename failed for {asset.asset_id}") from exc
    return {
        "asset_id": asset.asset_id,
        "asset_kind": asset.asset_kind,
        "filename": asset.filename,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "source_uri": asset.source_uri,
        "integrity_status": "EXACT_BYTES_AND_SHA256_VERIFIED",
        "payload_parsed": False,
    }


def _license_class(asset_id: str, asset_kind: str) -> str:
    if asset_kind == "GEO_RAW_COUNT":
        return "PRIVATE_CANONICAL_LOCATOR_HASH_ONLY_NO_REDISTRIBUTION_GRANT"
    if asset_id in {"GSE149487_MOESM3", "GSE149487_MOESM8"}:
        return "CC_BY_4_0_PAPER_SUPPLEMENT"
    return "NO_EXPLICIT_LICENSE_PRIVATE_EVIDENCE_ONLY"


def _write_aggregate_report(output_directory: Path, report: Mapping[str, Any]) -> Path:
    final_path = output_directory / REPORT_FILENAME
    temporary = output_directory / f".{REPORT_FILENAME}.tmp"
    payload = _pretty_json_bytes(report)
    if final_path.exists() or temporary.exists():
        raise OutputScopeError("aggregate acquisition report target already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.rename(final_path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AcquisitionError("aggregate acquisition report could not be written") from exc
    return final_path


def execute(
    protocol_path: Path,
    output_directory: Path,
    *,
    transport: DownloadTransport | None = None,
    binding_auditor: Callable[
        [Mapping[str, Any], Path, bytes], Mapping[str, Any]
    ] = _audit_implementation_binding,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Execute exact-21 acquisition without invoking any scientific consumer."""

    protocol, protocol_payload = _load_protocol(protocol_path)
    output = _validate_output_path(protocol, output_directory)
    binding_record = dict(binding_auditor(protocol, protocol_path, protocol_payload))

    repo_root = protocol_path.parent.parent
    documents, authority_records = _load_authorities(protocol, repo_root)
    assets = _validate_authorities(protocol, documents)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)
    except OSError as exc:
        raise OutputScopeError("exclusive output directory could not be created") from exc

    download = protocol["download_policy"]
    selected_transport = transport or UrllibTransport()
    verified_assets = [
        _download_one(
            asset,
            output,
            transport=selected_transport,
            timeout_seconds=int(download["request_timeout_seconds"]),
            chunk_bytes=int(download["stream_chunk_bytes"]),
        )
        for asset in assets
    ]
    for record in verified_assets:
        record["license_and_redistribution_class"] = _license_class(
            str(record["asset_id"]), str(record["asset_kind"])
        )

    terminal = copy.deepcopy(protocol["terminal_truth"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "recorded_at_utc": recorded_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": terminal["status_after_exact21_acquisition"],
        "acquisition_status": "EXACT_21_ASSETS_ACQUIRED_AND_INTEGRITY_VERIFIED",
        "output_directory": str(output),
        "implementation_binding": binding_record,
        "metadata_authorities": authority_records,
        "asset_counts": {
            "asset_count": len(verified_assets),
            "geo_raw_count": sum(
                record["asset_kind"] == "GEO_RAW_COUNT" for record in verified_assets
            ),
            "supplement_count": sum(
                record["asset_kind"] == "SUPPLEMENT_WORKBOOK"
                for record in verified_assets
            ),
            "total_verified_bytes": sum(int(record["bytes"]) for record in verified_assets),
        },
        "assets": verified_assets,
        "confirmed_public_evidence": copy.deepcopy(protocol["confirmed_public_evidence"]),
        "unknown_not_asserted": copy.deepcopy(protocol["unknown_not_asserted"]),
        "retained_blockers": list(protocol["retained_blockers"]),
        "terminal_truth": terminal,
        "execution_counters": {
            "payload_files_opened_for_scientific_parsing": 0,
            "row_sequence_effect_records_read": 0,
            "qualifier_execution_count": 0,
            "model_download_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
            "canonical_record_count": 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    _write_aggregate_report(output, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = execute(args.protocol, args.output_directory)
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "acquisition_status": report["acquisition_status"],
                "asset_count": report["asset_counts"]["asset_count"],
                "ready_for_full_qualifier_input": report["terminal_truth"][
                    "ready_for_full_qualifier_input"
                ],
                "qualified": report["terminal_truth"]["qualified"],
                "report": str(Path(report["output_directory"]) / REPORT_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
