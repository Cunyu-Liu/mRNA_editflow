#!/usr/bin/env python3
"""GSE261709 ordinary-public metadata/schema aggregate preflight.

The live mode reads only the two frozen small official metadata pages.  It
does not download ``GSE261709_RAW.tar``, follow its link, enumerate or open an
archive member, or contact an SRA payload.  A second mode accepts a previously
prepared aggregate of ordinary-public metadata/schema facts.  Neither mode
accepts member rows or member values.

The sole output is an aggregate gate report.  Even three PASS gates only say
that the owner may be asked for separate row-level qualification authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


SCHEMA_VERSION = (
    "route_a_v3_gse261709_public_identifier_asset_schema_aggregate_geometry_"
    "preflight.v1"
)
OBSERVATION_SCHEMA_VERSION = (
    "route_a_v3_gse261709_public_metadata_schema_observation.v1"
)
PROTOCOL_ID = (
    "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_V1"
)
PROTOCOL_BASENAME = (
    "route_a_v3_gse261709_public_identifier_asset_schema_aggregate_geometry_"
    "preflight_v1.json"
)
REPORT_FILENAME = (
    "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_"
    "V1.json"
)
DATASET_ID = "GSE261709"
BIOPROJECT_ID = "PRJNA1088465"
PMID = "38773080"
PMCID = "PMC11109163"
DOI = "10.1038/s41467-024-48436-5"
DECISION_ID = "V3-DEC-023"

UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
PASS = "PASS"
BLOCKED = "BLOCKED"
GATE_STATUSES = frozenset({PASS, BLOCKED, UNKNOWN})

CONFIG_PATH = f"configs/{PROTOCOL_BASENAME}"
SCRIPT_PATH = (
    "scripts/route_a_v3/"
    "preflight_gse261709_public_identifier_asset_schema_aggregate_geometry.py"
)
TEST_PATH = (
    "tests/route_a_v3/"
    "test_preflight_gse261709_public_identifier_asset_schema_aggregate_geometry.py"
)
EXPECTED_EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
RUNTIME_PATHS = (
    "configs/route_a_v3_dec023_authority_runtime_sync_v1.json",
    "scripts/route_a_v3/dec023_authority_runtime_sync.py",
    "tests/route_a_v3/test_dec023_authority_runtime_sync.py",
)
AUTHORITY_COMMIT = "f7cfff896a1a30d25a3b73ea7f89957d70d95d39"
RUNTIME_I1_COMMIT = "b0afa92eea9718c15a5989cfa67bac57036617d9"
RUNTIME_I2_COMMIT = "d125bec7e0d9f28a679ff98c25b4feb70e198034"
RUNTIME_B2_COMMIT = "b225f73b72d73be3380e2d48325c7773a67c0d17"
RUNTIME_I1_BLOB_SHA256_BY_PATH = {
    RUNTIME_PATHS[0]: "330a5fceaa97a1c1f16fcb20f1c6e4e35329923a293bcb463f35eaa666cb4701",
    RUNTIME_PATHS[1]: "3082aa44b70356d0e512fa8d1c92daadd08084a97594ba20976d0b93ca4706bd",
    RUNTIME_PATHS[2]: "0ad48f947eee136e9104ba1dcdd5c921281ad0379ce2a17dcd4c411861115e65",
}
RUNTIME_I2_BLOB_SHA256_BY_PATH = {
    RUNTIME_PATHS[0]: "4baa8ff33938dfc0448fb7e94097d407d8fb60beb122192aafb15eecd4f81c20",
    RUNTIME_PATHS[1]: "3502227b48a0506a8c2e9c2ddec0a51017d2cb86e68c155fbdc4bccf72fa41be",
    RUNTIME_PATHS[2]: "e3147049d06536b95298089ef1cd03676d8e477848890b1149dbd66352c44cc7",
}
RUNTIME_B2_BLOB_SHA256_BY_PATH = {
    RUNTIME_PATHS[0]: "d1d39a659aa3ccde38d88fadb76c72d77d47585cfd7038e58493eec818b6b280",
    RUNTIME_PATHS[1]: RUNTIME_I2_BLOB_SHA256_BY_PATH[RUNTIME_PATHS[1]],
    RUNTIME_PATHS[2]: RUNTIME_I2_BLOB_SHA256_BY_PATH[RUNTIME_PATHS[2]],
}
UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261709"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11109163/"
PUBMED_SUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
    "db=pubmed&id=38773080&retmode=json"
)
LIVE_URLS = (GEO_URL, PUBMED_SUMMARY_URL)
MAX_METADATA_BYTES = 4 * 1024 * 1024

EXPECTED_KNOWN_GEOMETRY = {
    "organism": "Homo sapiens",
    "assay": "MASSIVELY_PARALLEL_REPORTER_ASSAY",
    "utr_region": "3UTR",
    "cell_context_count": 2,
    "cell_context_labels": ["AGS", "SNU719"],
    "replicate_count_per_cell_context": 3,
    "biological_assay_sample_count": 6,
    "input_pool_sample_count": 1,
    "total_sample_count": 7,
    "platform_count": 1,
    "supplementary_archive_listing_count": 1,
    "supplementary_archive_filename": "GSE261709_RAW.tar",
    "supplementary_archive_display_size": "690.0 Kb",
    "raw_data_sra_relation_listed": True,
}
EXPECTED_GATE_IDS = (
    "OFFICIAL_IDENTIFIER_AND_CONTEXT",
    "ASSET_SAMPLE_AND_RUN_ROLE_AGGREGATE_GEOMETRY",
    "HEADER_DIMENSION_AND_ASSET_LICENSE_NOTICE",
)
REQUIRED_HEADER_ROLE_CLASSES = (
    "PUBLIC_IDENTIFIER",
    "SOURCE_CANDIDATE_ROLE",
    "SEQUENCE",
    "ENDPOINT",
    "REPLICATE",
    "CONTEXT",
)
EXPECTED_OUTER_TRUTH = {
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
    "gse261709_qualified": False,
    "gse261709_a1_credit_established": False,
    "row_level_qualification_run_count": 0,
    "split_run_count": 0,
    "power_run_count": 0,
    "canonical_materialization_run_count": 0,
    "training_run_count": 0,
    "gpu_work_run_count": 0,
    "model_selection_run_count": 0,
    "training_allowed": False,
    "gpu_work_allowed": False,
    "model_selection_allowed": False,
    "a7_unlocked": False,
    "next_phase_authorized": False,
    "scientific_claim_status": "NOT_ESTABLISHED",
}


class PreflightError(RuntimeError):
    """Base class for a fail-closed preflight failure."""


class ProtocolError(PreflightError):
    """The protocol, lifecycle, or aggregate observation is invalid."""


class BindingNotFrozen(ProtocolError):
    """The authority/runtime/preflight I/B chain is not complete."""


class MetadataError(PreflightError):
    """The small official metadata surfaces cannot be evaluated."""


class OutputError(PreflightError):
    """The single aggregate output cannot be published atomically."""


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
        raise PreflightError("value is not finite JSON") from exc


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return parsed


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ProtocolError(f"{label} fields differ from the aggregate-only schema")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "protocol_status": (
            "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_"
            "PREFLIGHT_ONLY_NOT_QUALIFICATION"
        ),
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} differs from the frozen value")

    binding = _mapping(
        protocol.get("implementation_binding"), label="implementation_binding"
    )
    if binding.get("binding_scheme") != (
        "AUTHORITY_A_THEN_AUTHORITY_RUNTIME_I1_I2_B2_THEN_EXACT3_I_CONFIG_ONLY_"
        "B_V2"
    ):
        raise ProtocolError("implementation binding scheme differs")
    if binding.get("authority_commit") != AUTHORITY_COMMIT:
        raise ProtocolError("DEC023 authority commit differs")
    if binding.get("authority_runtime_binding_commit") != RUNTIME_B2_COMMIT:
        raise ProtocolError("DEC023 authority-runtime B2 commit differs")
    runtime = _mapping(
        binding.get("authority_runtime_lineage"),
        label="authority_runtime_lineage",
    )
    if set(runtime) != {
        "paths",
        "implementation_i1_commit",
        "implementation_i1_blob_sha256_by_path",
        "implementation_i2_commit",
        "implementation_i2_blob_sha256_by_path",
        "binding_b2_blob_sha256_by_path",
        "implementation_exact_changed_paths",
        "binding_exact_changed_paths",
    }:
        raise ProtocolError("authority-runtime lineage fields differ")
    if tuple(runtime.get("paths", ())) != RUNTIME_PATHS:
        raise ProtocolError("authority-runtime exact3 paths differ")
    if runtime.get("implementation_i1_commit") != RUNTIME_I1_COMMIT:
        raise ProtocolError("authority-runtime I1 commit differs")
    if runtime.get("implementation_i2_commit") != RUNTIME_I2_COMMIT:
        raise ProtocolError("authority-runtime I2 commit differs")
    for field, expected in (
        (
            "implementation_i1_blob_sha256_by_path",
            RUNTIME_I1_BLOB_SHA256_BY_PATH,
        ),
        (
            "implementation_i2_blob_sha256_by_path",
            RUNTIME_I2_BLOB_SHA256_BY_PATH,
        ),
        ("binding_b2_blob_sha256_by_path", RUNTIME_B2_BLOB_SHA256_BY_PATH),
    ):
        if runtime.get(field) != expected:
            raise ProtocolError(f"authority-runtime {field} differs")
    if tuple(runtime.get("implementation_exact_changed_paths", ())) != RUNTIME_PATHS:
        raise ProtocolError("authority-runtime implementation is not exact3")
    if runtime.get("binding_exact_changed_paths") != [RUNTIME_PATHS[0]]:
        raise ProtocolError("authority-runtime B2 must be config-only")
    if binding.get("pre_implementation_authority_scalar_paths") != [
        "implementation_binding.authority_commit",
        "implementation_binding.authority_runtime_binding_commit",
    ]:
        raise ProtocolError("pre-implementation authority scalar paths differ")
    if binding.get("unknown_to_bound_scalar_paths") != [
        f"implementation_binding.{field}" for field in UNKNOWN_BINDING_SCALARS
    ]:
        raise ProtocolError("normal UNKNOWN-to-bound scalar paths differ")
    if tuple(binding.get("implementation_commit_exact_changed_paths", ())) != (
        EXPECTED_EXACT3
    ):
        raise ProtocolError("implementation commit is not exact3")
    if binding.get("binding_commit_exact_changed_paths") != [CONFIG_PATH]:
        raise ProtocolError("preflight binding commit must be config-only")
    if binding.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("implementation script path differs")
    if binding.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("implementation test path differs")

    normal_values = [binding.get(field) for field in UNKNOWN_BINDING_SCALARS]
    if binding.get("status") == UNKNOWN:
        if normal_values != [UNKNOWN] * 4:
            raise ProtocolError("initial-I binding scalars must remain an UNKNOWN group")
    elif binding.get("status") == BOUND:
        if not (
            isinstance(binding.get("implementation_commit"), str)
            and HEX40_RE.fullmatch(str(binding["implementation_commit"]))
        ):
            raise ProtocolError("BOUND implementation commit is invalid")
        for field in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if not (
                isinstance(binding.get(field), str)
                and HEX64_RE.fullmatch(str(binding[field]))
            ):
                raise ProtocolError(f"BOUND {field} is invalid")
    else:
        raise ProtocolError("implementation binding status is invalid")

    authority = _mapping(protocol.get("decision_authority"), label="authority")
    if authority.get("authorized_role") != (
        "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_ONLY"
    ):
        raise ProtocolError("DEC023 role differs")
    if authority.get("allowed_output_class") != (
        "AGGREGATE_GEOMETRY_AND_GATE_STATUS_ONLY"
    ):
        raise ProtocolError("output authority differs")
    if authority.get("separate_row_level_authority_required_after_all_gates_pass") is not True:
        raise ProtocolError("separate row-level authority boundary was removed")

    sources = protocol.get("official_public_sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ProtocolError("exactly three official authority locators are required")
    by_id = {
        item.get("authority_id"): item for item in sources if isinstance(item, dict)
    }
    if set(by_id) != {
        "NCBI_GEO_GSE261709",
        "NCBI_BIOPROJECT_PRJNA1088465",
        "NCBI_PUBMED_38773080_AND_PMC_11109163",
    }:
        raise ProtocolError("official authority identities differ")
    if by_id["NCBI_GEO_GSE261709"].get("url") != GEO_URL:
        raise ProtocolError("GEO live metadata URL differs")
    article = by_id["NCBI_PUBMED_38773080_AND_PMC_11109163"]
    if (
        article.get("url") != PMC_URL
        or article.get("live_metadata_url") != PUBMED_SUMMARY_URL
        or article.get("pmid") != PMID
        or article.get("pmcid") != PMCID
        or article.get("doi") != DOI
    ):
        raise ProtocolError("article identity differs")
    live_urls = tuple(
        item.get("live_metadata_url", item["url"])
        for item in sources
        if item.get("live_metadata_fetch_allowed")
    )
    if live_urls != LIVE_URLS:
        raise ProtocolError("live fetch surface differs from the two small pages")

    if protocol.get("frozen_known_public_geometry") != EXPECTED_KNOWN_GEOMETRY:
        raise ProtocolError("known public aggregate geometry differs")
    gate_contract = _mapping(protocol.get("gate_contract"), label="gate_contract")
    if tuple(gate_contract.get("gate_ids_exactly", ())) != EXPECTED_GATE_IDS:
        raise ProtocolError("three-gate contract differs")
    if set(gate_contract.get("allowed_statuses", ())) != GATE_STATUSES:
        raise ProtocolError("gate status vocabulary differs")
    if tuple(gate_contract.get("required_header_role_classes_exactly", ())) != (
        REQUIRED_HEADER_ROLE_CLASSES
    ):
        raise ProtocolError("required header-role classes differ")
    if gate_contract.get("all_pass_terminal_action") != (
        "GO_REQUEST_SEPARATE_ROW_LEVEL_QUALIFICATION_AUTHORITY"
    ):
        raise ProtocolError("all-PASS terminal action was promoted")

    inputs = _mapping(protocol.get("input_contract"), label="input_contract")
    if inputs.get("existing_aggregate_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ProtocolError("existing aggregate schema differs")
    for key in (
        "archive_or_processed_asset_download_allowed",
        "archive_member_listing_or_open_allowed",
        "row_or_member_body_read_allowed",
        "barcode_value_allowed",
        "variant_value_allowed",
        "transcript_value_allowed",
        "sequence_value_allowed",
        "effect_value_allowed",
        "se_value_allowed",
        "private_or_sealed_allowed",
    ):
        if inputs.get(key) is not False:
            raise ProtocolError(f"input_contract.{key} must remain false")
    output = _mapping(protocol.get("output_contract"), label="output_contract")
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("sole output filename differs")
    if output.get("single_aggregate_output_only") is not True:
        raise ProtocolError("output must remain one aggregate")
    for key, value in output.items():
        if key.endswith("_included") and value is not False:
            raise ProtocolError(f"output_contract.{key} must remain false")
    if protocol.get("frozen_outer_truth") != EXPECTED_OUTER_TRUTH:
        raise ProtocolError("outer truth differs")


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename differs")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("cannot read protocol") from exc
    protocol = _strict_json_object(payload, label="protocol")
    _validate_protocol(protocol)
    return protocol, payload


def _run_git_text(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for lifecycle audit") from exc
    if completed.returncode != 0:
        raise ProtocolError("git lifecycle audit failed")
    return completed.stdout.strip()


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProtocolError("git is unavailable for bound blob audit") from exc
    if completed.returncode != 0:
        raise ProtocolError("cannot read a bound implementation blob")
    return completed.stdout


def _changed_paths(repo_root: Path, commit: str) -> tuple[str, ...]:
    text = _run_git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in text.splitlines() if line))


def _verify_blob_map(
    repo_root: Path,
    commit: str,
    expected_sha256_by_path: Mapping[str, str],
) -> None:
    for path, expected in expected_sha256_by_path.items():
        observed = hashlib.sha256(_git_blob(repo_root, commit, path)).hexdigest()
        if observed != expected:
            raise ProtocolError("authority-runtime frozen blob differs")


def _normalise_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    for field in UNKNOWN_BINDING_SCALARS:
        result["implementation_binding"][field] = UNKNOWN
    return result


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_payload: bytes,
    repo_root: Path,
) -> dict[str, str]:
    del protocol_payload
    binding = protocol["implementation_binding"]
    if binding.get("status") != BOUND:
        raise BindingNotFrozen("preflight exact3-I/config-only-B is not BOUND")
    if any(binding.get(field) == UNKNOWN for field in UNKNOWN_BINDING_SCALARS):
        raise BindingNotFrozen("preflight binding scalars remain UNKNOWN")
    if (
        binding.get("authority_commit") == UNKNOWN
        or binding.get("authority_runtime_binding_commit") == UNKNOWN
    ):
        raise BindingNotFrozen("DEC023 authority/runtime binding remains UNKNOWN")

    executing_script_path = Path(__file__).resolve()
    production_script_path = (repo_root / SCRIPT_PATH).resolve()
    if executing_script_path != production_script_path:
        raise ProtocolError("executing producer is not the bound production path")

    preflight_b = _run_git_text(repo_root, "rev-parse", "HEAD")
    preflight_i = str(binding["implementation_commit"])
    runtime = binding["authority_runtime_lineage"]
    runtime_i1 = str(runtime["implementation_i1_commit"])
    runtime_i2 = str(runtime["implementation_i2_commit"])
    runtime_b2 = str(binding["authority_runtime_binding_commit"])
    authority_a = str(binding["authority_commit"])
    if _run_git_text(repo_root, "rev-parse", f"{preflight_b}^") != preflight_i:
        raise ProtocolError("preflight B is not the direct child of preflight I")
    if _run_git_text(repo_root, "rev-parse", f"{preflight_i}^") != runtime_b2:
        raise ProtocolError("preflight I is not the direct child of runtime B2")
    if _run_git_text(repo_root, "rev-parse", f"{runtime_b2}^") != runtime_i2:
        raise ProtocolError("authority-runtime B2 is not the direct child of I2")
    if _run_git_text(repo_root, "rev-parse", f"{runtime_i2}^") != runtime_i1:
        raise ProtocolError("authority-runtime I2 is not the direct child of I1")
    if _run_git_text(repo_root, "rev-parse", f"{runtime_i1}^") != authority_a:
        raise ProtocolError("authority-runtime I1 is not based on DEC023 A")
    if _changed_paths(repo_root, runtime_i1) != tuple(sorted(RUNTIME_PATHS)):
        raise ProtocolError("authority-runtime I1 did not change exact3")
    if _changed_paths(repo_root, runtime_i2) != tuple(sorted(RUNTIME_PATHS)):
        raise ProtocolError("authority-runtime I2 did not change exact3")
    if _changed_paths(repo_root, runtime_b2) != (RUNTIME_PATHS[0],):
        raise ProtocolError("authority-runtime B2 did not change config-only")
    _verify_blob_map(
        repo_root,
        runtime_i1,
        runtime["implementation_i1_blob_sha256_by_path"],
    )
    _verify_blob_map(
        repo_root,
        runtime_i2,
        runtime["implementation_i2_blob_sha256_by_path"],
    )
    _verify_blob_map(
        repo_root,
        runtime_b2,
        runtime["binding_b2_blob_sha256_by_path"],
    )
    if _changed_paths(repo_root, preflight_i) != tuple(sorted(EXPECTED_EXACT3)):
        raise ProtocolError("preflight I did not change exact3")
    if _changed_paths(repo_root, preflight_b) != (CONFIG_PATH,):
        raise ProtocolError("preflight B did not change config-only")

    i_protocol = _strict_json_object(
        _git_blob(repo_root, preflight_i, CONFIG_PATH),
        label="implementation protocol",
    )
    if i_protocol != _normalise_binding(protocol):
        raise ProtocolError("preflight B changed more than four binding scalars")
    script_blob = _git_blob(repo_root, preflight_i, SCRIPT_PATH)
    test_blob = _git_blob(repo_root, preflight_i, TEST_PATH)
    if hashlib.sha256(script_blob).hexdigest() != binding.get(
        "implementation_script_sha256"
    ):
        raise ProtocolError("bound script digest differs")
    if hashlib.sha256(test_blob).hexdigest() != binding.get(
        "implementation_test_sha256"
    ):
        raise ProtocolError("bound test digest differs")
    if protocol_path.resolve() != (repo_root / CONFIG_PATH).resolve():
        raise ProtocolError("protocol path is outside the bound repository location")
    if protocol_path.read_bytes() != _git_blob(repo_root, preflight_b, CONFIG_PATH):
        raise ProtocolError("working protocol differs from bound B")
    if (repo_root / SCRIPT_PATH).read_bytes() != script_blob:
        raise ProtocolError("working script differs from preflight I")
    if executing_script_path.read_bytes() != script_blob:
        raise ProtocolError("executing producer bytes differ from preflight I")
    if (repo_root / TEST_PATH).read_bytes() != test_blob:
        raise ProtocolError("working test differs from preflight I")
    return {
        "status": "BOUND_AUTHORITY_RUNTIME_I1_I2_B2_AND_EXACT3_I_B_VERIFIED",
        "authority_commit": authority_a,
        "authority_runtime_implementation_i1_commit": runtime_i1,
        "authority_runtime_implementation_i2_commit": runtime_i2,
        "authority_runtime_binding_commit": runtime_b2,
        "implementation_commit": preflight_i,
        "binding_commit": preflight_b,
    }


class MetadataFetcher(Protocol):
    def fetch_text(self, url: str) -> str: ...


class OfficialSmallMetadataFetcher:
    """Fetch only the two exact metadata pages, with no payload URL method."""

    def fetch_text(self, url: str) -> str:
        if url not in LIVE_URLS:
            raise MetadataError("URL is outside the frozen small-metadata allowlist")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json,text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "User-Agent": "mRNA-XEditFlow-GSE261709-metadata-preflight/1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                payload = response.read(MAX_METADATA_BYTES + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise MetadataError("official small-metadata fetch failed") from exc
        if len(payload) > MAX_METADATA_BYTES:
            raise MetadataError("official metadata page exceeds the frozen size cap")
        return payload.decode("utf-8", errors="replace")


def _plain_text(page: str) -> str:
    without_markup = re.sub(r"<[^>]+>", " ", page)
    return " ".join(html.unescape(without_markup).split())


def build_live_observation(
    protocol: Mapping[str, Any], fetcher: MetadataFetcher
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    del protocol
    geo = _plain_text(fetcher.fetch_text(GEO_URL))
    article_metadata = _plain_text(fetcher.fetch_text(PUBMED_SUMMARY_URL))
    source_results = [
        {"authority_id": "NCBI_GEO_GSE261709", "status": PASS},
        {
            "authority_id": "NCBI_PUBMED_38773080_AND_PMC_11109163",
            "status": PASS,
        },
    ]
    expected_roles = (
        "AGS, rep1",
        "AGS, rep2",
        "AGS, rep3",
        "SNU719, rep1",
        "SNU719, rep2",
        "SNU719, rep3",
        "plasmid pool",
    )
    identity_verified = all(
        token in f"{geo} {article_metadata}"
        for token in (DATASET_ID, BIOPROJECT_ID, PMID, DOI)
    )
    context_verified = all(
        token in geo
        for token in (
            "Homo sapiens",
            "massively parallel reporter assay",
            "AGS",
            "SNU719",
            "triplicate",
        )
    )
    roles_complete = all(role in geo for role in expected_roles)
    listing_complete = all(
        token in geo
        for token in (
            "GSE261709_RAW.tar",
            "690.0 Kb",
            "Raw data are available in SRA",
        )
    )
    # PubMed ESummary is identifier metadata only.  The article-body license
    # text and, separately, an asset-applicable license must come from a
    # sanitized ordinary-public aggregate; this live path does not fetch them.
    article_license_visible = False
    return (
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "identity_context": {
                "dataset_id": DATASET_ID,
                "bioproject_id": BIOPROJECT_ID,
                "pmid": PMID,
                "pmcid": PMCID,
                "doi": DOI,
                "official_identity_verified": identity_verified,
                "context_verified": context_verified,
            },
            "aggregate_role_geometry": {
                **copy.deepcopy(EXPECTED_KNOWN_GEOMETRY),
                "sample_role_listing_complete": roles_complete,
                "archive_listing_complete": listing_complete,
                "run_count": UNKNOWN,
                "run_role_mapping_complete": False,
            },
            "schema_and_rights": {
                "header_name_count": 0,
                "header_role_class_presence": {
                    role_class: False
                    for role_class in REQUIRED_HEADER_ROLE_CLASSES
                },
                "dimension_measure_count": 0,
                "exact_dimensions_complete": False,
                "article_license_notice_visible": article_license_visible,
                "asset_license_notice_visible": False,
                "asset_license_notice_applies_to_row_level_research": False,
            },
            "scope": _zero_scope(),
        },
        source_results,
    )


def _zero_scope(*, page_reads: int = 2, listing_reads: int = 1) -> dict[str, Any]:
    return {
        "ordinary_public_only": True,
        "small_metadata_page_read_count": page_reads,
        "archive_listing_read_count": listing_reads,
        "archive_download_count": 0,
        "archive_member_listing_count": 0,
        "archive_member_open_count": 0,
        "row_or_member_body_read_count": 0,
        "barcode_value_read_count": 0,
        "variant_value_read_count": 0,
        "transcript_value_read_count": 0,
        "sequence_value_read_count": 0,
        "effect_value_read_count": 0,
        "se_value_read_count": 0,
        "private_or_sealed_read_count": 0,
    }


def _validate_observation(observation: Mapping[str, Any]) -> None:
    _exact_keys(
        observation,
        {
            "schema_version",
            "identity_context",
            "aggregate_role_geometry",
            "schema_and_rights",
            "scope",
        },
        label="observation",
    )
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ProtocolError("observation schema version differs")
    identity = _mapping(observation.get("identity_context"), label="identity_context")
    _exact_keys(
        identity,
        {
            "dataset_id",
            "bioproject_id",
            "pmid",
            "pmcid",
            "doi",
            "official_identity_verified",
            "context_verified",
        },
        label="identity_context",
    )
    if any(
        identity.get(key) != expected
        for key, expected in {
            "dataset_id": DATASET_ID,
            "bioproject_id": BIOPROJECT_ID,
            "pmid": PMID,
            "pmcid": PMCID,
            "doi": DOI,
        }.items()
    ):
        raise ProtocolError("observation public identity differs")
    if not all(
        isinstance(identity.get(key), bool)
        for key in ("official_identity_verified", "context_verified")
    ):
        raise ProtocolError("identity/context observations must be booleans")

    geometry = _mapping(
        observation.get("aggregate_role_geometry"), label="aggregate_role_geometry"
    )
    expected_geometry_keys = set(EXPECTED_KNOWN_GEOMETRY) | {
        "sample_role_listing_complete",
        "archive_listing_complete",
        "run_count",
        "run_role_mapping_complete",
    }
    _exact_keys(geometry, expected_geometry_keys, label="aggregate_role_geometry")
    for key, expected in EXPECTED_KNOWN_GEOMETRY.items():
        if geometry.get(key) != expected:
            raise ProtocolError(f"aggregate_role_geometry.{key} differs")
    if not all(
        isinstance(geometry.get(key), bool)
        for key in (
            "sample_role_listing_complete",
            "archive_listing_complete",
            "run_role_mapping_complete",
        )
    ):
        raise ProtocolError("role-listing completion fields must be booleans")
    run_count = geometry.get("run_count")
    if run_count != UNKNOWN and (not isinstance(run_count, int) or run_count < 1):
        raise ProtocolError("run_count must be positive or UNKNOWN_NOT_ASSERTED")

    schema = _mapping(
        observation.get("schema_and_rights"), label="schema_and_rights"
    )
    _exact_keys(
        schema,
        {
            "header_name_count",
            "header_role_class_presence",
            "dimension_measure_count",
            "exact_dimensions_complete",
            "article_license_notice_visible",
            "asset_license_notice_visible",
            "asset_license_notice_applies_to_row_level_research",
        },
        label="schema_and_rights",
    )
    for key in ("header_name_count", "dimension_measure_count"):
        if not isinstance(schema.get(key), int) or schema[key] < 0:
            raise ProtocolError(f"schema_and_rights.{key} must be nonnegative")
    header_roles = _mapping(
        schema.get("header_role_class_presence"),
        label="schema_and_rights.header_role_class_presence",
    )
    _exact_keys(
        header_roles,
        set(REQUIRED_HEADER_ROLE_CLASSES),
        label="schema_and_rights.header_role_class_presence",
    )
    if not all(isinstance(value, bool) for value in header_roles.values()):
        raise ProtocolError("header-role class presence values must be booleans")
    for key in (
        "exact_dimensions_complete",
        "article_license_notice_visible",
        "asset_license_notice_visible",
        "asset_license_notice_applies_to_row_level_research",
    ):
        if not isinstance(schema.get(key), bool):
            raise ProtocolError(f"schema_and_rights.{key} must be boolean")

    scope = _mapping(observation.get("scope"), label="scope")
    if dict(scope) not in (
        _zero_scope(),
        _zero_scope(page_reads=0, listing_reads=0),
    ):
        raise ProtocolError("observation scope exceeds the no-member boundary")


def _load_observation(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("cannot read aggregate metadata observation") from exc
    observation = _strict_json_object(payload, label="aggregate observation")
    _validate_observation(observation)
    return observation


def _gate(gate_id: str, status: str, reason: str) -> dict[str, str]:
    if gate_id not in EXPECTED_GATE_IDS or status not in GATE_STATUSES:
        raise ProtocolError("invalid gate result")
    return {"gate_id": gate_id, "status": status, "reason": reason}


def evaluate_observation(
    protocol: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    source_mode: str,
    source_results: list[dict[str, str]],
    recorded_at: str,
) -> dict[str, Any]:
    _validate_observation(observation)
    identity = observation["identity_context"]
    geometry = observation["aggregate_role_geometry"]
    schema = observation["schema_and_rights"]

    identity_pass = (
        identity["official_identity_verified"] and identity["context_verified"]
    )
    role_pass = (
        geometry["sample_role_listing_complete"]
        and geometry["archive_listing_complete"]
        and geometry["run_count"] != UNKNOWN
        and geometry["run_role_mapping_complete"]
    )
    schema_pass = (
        schema["header_name_count"] > 0
        and all(schema["header_role_class_presence"].values())
        and schema["dimension_measure_count"] > 0
        and schema["exact_dimensions_complete"]
        and schema["asset_license_notice_visible"]
        and schema["asset_license_notice_applies_to_row_level_research"]
    )
    gates = [
        _gate(
            EXPECTED_GATE_IDS[0],
            PASS if identity_pass else BLOCKED,
            "OFFICIAL_IDENTITIES_AND_CONTEXT_VISIBLE"
            if identity_pass
            else "OFFICIAL_IDENTITY_OR_CONTEXT_NOT_CLOSED",
        ),
        _gate(
            EXPECTED_GATE_IDS[1],
            PASS if role_pass else BLOCKED,
            "ASSET_SAMPLE_AND_RUN_ROLE_GEOMETRY_COMPLETE"
            if role_pass
            else "RUN_ROLE_OR_ARCHIVE_LISTING_GEOMETRY_NOT_CLOSED",
        ),
        _gate(
            EXPECTED_GATE_IDS[2],
            PASS if schema_pass else BLOCKED,
            "HEADER_DIMENSIONS_AND_ASSET_LICENSE_NOTICE_COMPLETE"
            if schema_pass
            else "HEADER_DIMENSIONS_OR_ASSET_LEVEL_LICENSE_NOTICE_NOT_CLOSED",
        ),
    ]
    counts = Counter(gate["status"] for gate in gates)
    all_pass = all(gate["status"] == PASS for gate in gates)
    status = (
        "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_PREFLIGHT_"
        "COMPLETE_ROW_LEVEL_AUTHORITY_REQUIRED"
        if all_pass
        else "STOP_PREFLIGHT_GATES_NOT_CLOSED"
    )
    next_action = (
        "GO_REQUEST_SEPARATE_ROW_LEVEL_QUALIFICATION_AUTHORITY"
        if all_pass
        else "STOP_PREFLIGHT_GATES_NOT_CLOSED"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": BIOPROJECT_ID,
        "decision_id": DECISION_ID,
        "recorded_at": recorded_at,
        "status": status,
        "preflight_complete": True,
        "implementation_binding": dict(binding),
        "source_mode": source_mode,
        "official_source_results": copy.deepcopy(source_results),
        "gates": gates,
        "gate_counts": {
            PASS: counts[PASS],
            BLOCKED: counts[BLOCKED],
            UNKNOWN: counts[UNKNOWN],
        },
        "aggregate_geometry": {
            "official_identifier_count": 5,
            "official_source_locator_count": 3,
            "live_metadata_source_count": len(source_results),
            "platform_count": geometry["platform_count"],
            "cell_context_count": geometry["cell_context_count"],
            "biological_assay_sample_count": geometry[
                "biological_assay_sample_count"
            ],
            "input_pool_sample_count": geometry["input_pool_sample_count"],
            "total_sample_count": geometry["total_sample_count"],
            "replicate_count_per_cell_context": geometry[
                "replicate_count_per_cell_context"
            ],
            "run_count_or_status": geometry["run_count"],
            "supplementary_archive_listing_count": geometry[
                "supplementary_archive_listing_count"
            ],
            "supplementary_archive_display_size": geometry[
                "supplementary_archive_display_size"
            ],
            "header_name_count": schema["header_name_count"],
            "required_header_role_class_count": len(REQUIRED_HEADER_ROLE_CLASSES),
            "observed_required_header_role_class_count": sum(
                schema["header_role_class_presence"].values()
            ),
            "dimension_measure_count": schema["dimension_measure_count"],
            "article_license_notice_count": int(
                schema["article_license_notice_visible"]
            ),
            "applicable_asset_license_notice_count": int(
                schema["asset_license_notice_visible"]
                and schema["asset_license_notice_applies_to_row_level_research"]
            ),
        },
        "scope_attestation": {
            "authority_role": (
                "PUBLIC_IDENTIFIER_ASSET_SCHEMA_AND_AGGREGATE_GEOMETRY_"
                "PREFLIGHT_ONLY"
            ),
            "ordinary_public_only": True,
            "single_aggregate_output_only": True,
            **copy.deepcopy(observation["scope"]),
            "header_name_output_count": 0,
            "member_identifier_output_count": 0,
            "sample_or_run_identifier_output_count": 0,
            "row_record_output_count": 0,
            "barcode_value_output_count": 0,
            "variant_value_output_count": 0,
            "transcript_value_output_count": 0,
            "sequence_value_output_count": 0,
            "effect_value_output_count": 0,
            "se_value_output_count": 0,
        },
        "terminal_truth": copy.deepcopy(dict(protocol["frozen_outer_truth"])),
        "interpretation_boundary": {
            "all_gates_pass_is_row_level_qualification": False,
            "all_gates_pass_is_a1_credit": False,
            "all_gates_pass_changes_canonical_count": False,
            "all_gates_pass_authorizes_training_or_gpu": False,
            "all_gates_pass_authorizes_model_selection_or_a7": False,
            "all_gates_pass_authorizes_only_a_separate_authority_request": True,
        },
        "sole_next_action": next_action,
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_temp_payload(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_exclusive(output_dir: Path, report: Mapping[str, Any]) -> Path:
    temporary_path: Path | None = None
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise OutputError("output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / REPORT_FILENAME
        if output_path.exists():
            raise OutputError("aggregate report already exists")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{REPORT_FILENAME}.", suffix=".tmp", dir=output_dir
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        _write_temp_payload(temporary_path, _json_bytes(report))
        if output_path.exists():
            raise OutputError("aggregate report appeared during publication")
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OutputError:
        raise
    except OSError as exc:
        raise OutputError("cannot publish aggregate-only report") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


BindingAuditor = Callable[
    [Mapping[str, Any], Path, bytes, Path], Mapping[str, str]
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execute(
    protocol_path: Path,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    observation_path: Path | None = None,
    fetcher: MetadataFetcher | None = None,
    binding_auditor: BindingAuditor = _default_binding_auditor,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_payload = load_protocol(protocol_path)
    binding = binding_auditor(
        protocol,
        protocol_path,
        protocol_payload,
        repo_root or protocol_path.parent.parent,
    )
    if observation_path is None:
        observation, source_results = build_live_observation(
            protocol, fetcher or OfficialSmallMetadataFetcher()
        )
        source_mode = "LIVE_OFFICIAL_SMALL_METADATA_AND_ARCHIVE_LISTING_ONLY"
    else:
        if fetcher is not None:
            raise ProtocolError("live fetcher and aggregate observation are exclusive")
        observation = _load_observation(observation_path)
        source_results = []
        source_mode = "EXISTING_ORDINARY_PUBLIC_METADATA_SCHEMA_AGGREGATE_ONLY"
    report = evaluate_observation(
        protocol,
        observation,
        binding=binding,
        source_mode=source_mode,
        source_results=source_results,
        recorded_at=recorded_at or _utc_now(),
    )
    _write_exclusive(output_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--observation", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = execute(
            args.protocol,
            args.output_dir,
            repo_root=args.repo_root,
            observation_path=args.observation,
        )
    except PreflightError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
