#!/usr/bin/env python3
"""Aggregate-only GSE217518 public-authority preflight.

The producer reads either two official metadata pages (NCBI GEO and the eLife
Version of Record) or one previously prepared aggregate of official public
metadata/header facts.  It never downloads or opens the four processed CSV
assets or the article supplement, and it never reads row, sequence, or effect
values.  Its only output is ``GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_V1.json``.

``ready_for_ordinary_public_row_level_producer`` means only that a later
ordinary-public producer may be written.  It is not dataset qualification and
does not authorize reconstruction, canonical materialization, a qualifier,
training, model selection, or a later phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


SCHEMA_VERSION = "route_a_v3_gse217518_public_authority_preflight.v1"
OBSERVATION_SCHEMA_VERSION = (
    "route_a_v3_gse217518_public_authority_observation.v1"
)
PROTOCOL_ID = "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_V1"
PROTOCOL_BASENAME = "route_a_v3_gse217518_public_authority_preflight_v1.json"
REPORT_FILENAME = "GSE217518_PUBLIC_AUTHORITY_PREFLIGHT_V1.json"
DATASET_ID = "GSE217518"

UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PASS = "PASS"
BLOCKED = "BLOCKED"
GATE_STATUSES = frozenset({PASS, BLOCKED, UNKNOWN})
UNKNOWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)

EXPECTED_ASSETS = (
    (
        "GSE217518_HEK_U3_RAW",
        "GSE217518_HEK_U3_Raw.csv.gz",
        "HEK293T",
        "3UTR",
    ),
    (
        "GSE217518_HEK_U5_RAW",
        "GSE217518_HEK_U5_Raw.csv.gz",
        "HEK293T",
        "5UTR",
    ),
    (
        "GSE217518_SH_U3_RAW",
        "GSE217518_SH_U3_Raw.csv.gz",
        "SH-SY5Y",
        "3UTR",
    ),
    (
        "GSE217518_SH_U5_RAW",
        "GSE217518_SH_U5_Raw.csv.gz",
        "SH-SY5Y",
        "5UTR",
    ),
)
EXPECTED_SAMPLE_NAMES = tuple(
    [
        f"SH-SY5Y_{region}_{time}_{replicate}"
        for region in ("5U", "3U")
        for time in (20, 40, 60)
        for replicate in (1, 2, 3)
    ]
    + [
        f"HEK293_{region}_{replicate}_{time}"
        for region in ("5U", "3U")
        for replicate in (1, 2, 3)
        for time in (30, 75, 120)
    ]
)


class PreflightError(RuntimeError):
    """Base class for a fail-closed preflight error."""


class ProtocolError(PreflightError):
    """The protocol or its implementation binding is not valid."""


class BindingNotFrozen(ProtocolError):
    """The normal UNKNOWN-I to config-only-B lifecycle is incomplete."""


class ObservationError(PreflightError):
    """An existing aggregate is outside the supported aggregate-only shape."""


class OutputError(PreflightError):
    """The sole aggregate output cannot be written without overwriting."""


class MetadataFetcher(Protocol):
    def fetch_text(self, url: str) -> str: ...


class UrllibMetadataFetcher:
    """Fetch only the two exact official HTML metadata pages."""

    def fetch_text(self, url: str) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "User-Agent": "mRNA-XEditFlow-GSE217518-public-preflight/1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise PreflightError(f"official metadata fetch failed for {url}") from exc


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
        raise PreflightError("report is not finite JSON") from exc


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
        raise ObservationError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ObservationError(f"{label} must be a JSON object")
    return parsed


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ObservationError(f"cannot read {label}: {path}") from exc
    return _strict_json_object(payload, label=label), payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "protocol_status": "PREFLIGHT_ONLY_STOP_BEFORE_PAYLOAD",
    }
    for key, expected in expected_scalars.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} differs from the supported value")

    binding = _require_mapping(
        protocol.get("implementation_binding"), label="implementation_binding"
    )
    if binding.get("binding_scheme") != "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1":
        raise ProtocolError("implementation binding is not config-only-B")
    if binding.get("status") not in {UNKNOWN, "BOUND"}:
        raise ProtocolError("implementation binding status is invalid")
    expected_paths = [
        f"implementation_binding.{field}" for field in UNKNOWN_BINDING_SCALARS
    ]
    if binding.get("unknown_to_bound_scalar_paths") != expected_paths:
        raise ProtocolError("the four UNKNOWN-to-bound scalar paths differ")
    if binding.get("implementation_commit_exact_changed_paths") != [
        f"configs/{PROTOCOL_BASENAME}",
        "scripts/route_a_v3/preflight_gse217518_public_authority.py",
        "tests/route_a_v3/test_preflight_gse217518_public_authority.py",
    ]:
        raise ProtocolError("implementation exact3 differs")
    if binding.get("binding_commit_exact_changed_paths") != [
        f"configs/{PROTOCOL_BASENAME}"
    ]:
        raise ProtocolError("binding commit must be config-only")

    assets = protocol.get("official_processed_assets")
    if not isinstance(assets, list) or len(assets) != 4:
        raise ProtocolError("exactly four official processed assets are required")
    observed_assets = [
        (
            item.get("asset_id"),
            item.get("filename"),
            item.get("cell_line"),
            item.get("region"),
        )
        for item in assets
        if isinstance(item, dict)
    ]
    if observed_assets != list(EXPECTED_ASSETS):
        raise ProtocolError("the U3/U5 by HEK/SH asset authority differs")
    for item in assets:
        if not str(item.get("locator", "")).startswith(
            "https://ftp.ncbi.nlm.nih.gov/geo/"
        ):
            raise ProtocolError("processed asset locator is not official NCBI HTTPS")

    endpoint = _require_mapping(
        protocol.get("endpoint_authority"), label="endpoint_authority"
    )
    endpoint_expected = {
        "status": PASS,
        "endpoint_identity": "RNA_HALF_LIFE_T1_2",
        "endpoint_unit": "MINUTES",
        "direction": "HIGHER_T1_2_IS_MORE_STABLE",
        "raw_measurement_scale": (
            "NONNEGATIVE_AMPLICON_READ_COUNT_AT_EACH_TIMEPOINT_AND_"
            "EXPERIMENTAL_REPLICATE"
        ),
    }
    for key, expected in endpoint_expected.items():
        if endpoint.get(key) != expected:
            raise ProtocolError(f"endpoint_authority.{key} differs")

    outlier = _require_mapping(
        protocol.get("author_defined_outlier_policy"),
        label="author_defined_outlier_policy",
    )
    if (
        outlier.get("status") != BLOCKED
        or outlier.get("executable_threshold_or_membership_rule") is not None
        or outlier.get("row_value_inference_allowed") is not False
    ):
        raise ProtocolError("current outlier-policy truth must remain blocked")

    current = _require_mapping(
        protocol.get("current_authority_assessment"),
        label="current_authority_assessment",
    )
    if current.get("ready_for_ordinary_public_row_level_producer") is not False:
        raise ProtocolError("current authority assessment may not claim GO")
    if current.get("terminal_status") != (
        "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
    ):
        raise ProtocolError("current terminal status must be STOP")

    inputs = _require_mapping(protocol.get("input_contract"), label="input_contract")
    for key in (
        "row_values_allowed",
        "sequence_values_allowed",
        "effect_values_allowed",
        "raw_processed_asset_download_allowed",
        "raw_processed_asset_open_allowed",
        "restricted_or_sealed_allowed",
        "GSE246381_allowed",
    ):
        if inputs.get(key) is not False:
            raise ProtocolError(f"input_contract.{key} must be false")
    output = _require_mapping(
        protocol.get("output_contract"), label="output_contract"
    )
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("sole report filename differs")
    if output.get("single_aggregate_output_only") is not True:
        raise ProtocolError("output must remain a single aggregate")


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename differs")
    protocol, payload = _read_json(path, label="protocol")
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
        raise ProtocolError("git is unavailable for implementation binding") from exc
    if completed.returncode != 0:
        raise ProtocolError("git implementation-binding check failed")
    return completed.stdout.strip()


def _normalise_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    binding = result["implementation_binding"]
    for field in UNKNOWN_BINDING_SCALARS:
        binding[field] = UNKNOWN
    return result


def _default_binding_auditor(
    protocol: Mapping[str, Any], protocol_path: Path, protocol_payload: bytes
) -> dict[str, str]:
    del protocol_payload
    binding = protocol["implementation_binding"]
    if any(binding.get(field) == UNKNOWN for field in UNKNOWN_BINDING_SCALARS):
        raise BindingNotFrozen(
            "implementation binding remains UNKNOWN; config-only-B is required"
        )
    if binding.get("status") != "BOUND":
        raise BindingNotFrozen("implementation binding is not BOUND")

    repo_root = protocol_path.parent.parent
    binding_commit = _run_git(repo_root, "rev-parse", "HEAD")
    implementation_commit = str(binding["implementation_commit"])
    base_commit = str(binding["base_commit"])
    if _run_git(repo_root, "rev-parse", f"{binding_commit}^") != implementation_commit:
        raise ProtocolError("binding commit is not the direct child of I")
    if _run_git(repo_root, "rev-parse", f"{implementation_commit}^") != base_commit:
        raise ProtocolError("implementation commit is not the direct child of base")

    binding_paths = _run_git(
        repo_root, "diff", "--name-only", implementation_commit, binding_commit
    ).splitlines()
    if binding_paths != binding["binding_commit_exact_changed_paths"]:
        raise ProtocolError("binding commit is not config-only")
    implementation_paths = _run_git(
        repo_root, "diff", "--name-only", base_commit, implementation_commit
    ).splitlines()
    if implementation_paths != binding["implementation_commit_exact_changed_paths"]:
        raise ProtocolError("implementation commit does not contain exact3")

    implementation_config = _strict_json_object(
        _run_git(
            repo_root,
            "show",
            f"{implementation_commit}:configs/{PROTOCOL_BASENAME}",
        ).encode("utf-8"),
        label="implementation config",
    )
    if _normalise_binding(implementation_config) != _normalise_binding(protocol):
        raise ProtocolError("B changed more than the four binding scalars")

    for path_key, sha_key in (
        ("implementation_script_path", "implementation_script_sha256"),
        ("implementation_test_path", "implementation_test_sha256"),
    ):
        payload = (repo_root / str(binding[path_key])).read_bytes()
        if _sha256(payload) != binding[sha_key]:
            raise ProtocolError(f"{path_key} differs from its bound implementation")
    if _run_git(repo_root, "status", "--porcelain"):
        raise ProtocolError("production worktree or index is not clean")
    return {
        "status": "BOUND_CONFIG_ONLY_VERIFIED",
        "base_commit": base_commit,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
    }


def _plain_text(document: str) -> str:
    without_markup = re.sub(r"<[^>]+>", " ", document)
    return " ".join(html.unescape(without_markup).split())


def _official_url_by_id(protocol: Mapping[str, Any], authority_id: str) -> str:
    for authority in protocol["official_public_authorities"]:
        if authority.get("authority_id") == authority_id:
            return str(authority["url"])
    raise ProtocolError(f"official authority {authority_id} is absent")


def build_live_observation(
    protocol: Mapping[str, Any], fetcher: MetadataFetcher
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    sources: list[dict[str, str]] = []
    documents: dict[str, str] = {}
    for authority_id in ("NCBI_GEO_GSE217518", "ELIFE_97682_VERSION_OF_RECORD"):
        url = _official_url_by_id(protocol, authority_id)
        try:
            documents[authority_id] = _plain_text(fetcher.fetch_text(url))
            sources.append(
                {"authority_id": authority_id, "url": url, "status": PASS}
            )
        except PreflightError:
            documents[authority_id] = ""
            sources.append(
                {"authority_id": authority_id, "url": url, "status": UNKNOWN}
            )

    geo = documents["NCBI_GEO_GSE217518"]
    article = documents["ELIFE_97682_VERSION_OF_RECORD"]
    asset_listing_pass = bool(geo) and all(
        filename in geo for _, filename, _, _ in EXPECTED_ASSETS
    )
    replicate_pass = bool(geo) and all(name in geo for name in EXPECTED_SAMPLE_NAMES)
    article_lower = article.lower()
    geo_lower = geo.lower()
    endpoint_pass = all(
        marker in geo_lower
        for marker in ("half-life", "weighted linear regression", "wild-type")
    ) and all(
        marker in article_lower
        for marker in ("half-life estimation", "mean squared error", "115 bp")
    )

    assets: list[dict[str, Any]] = []
    by_id = {
        item["asset_id"]: item for item in protocol["official_processed_assets"]
    }
    for asset_id, filename, cell_line, region in EXPECTED_ASSETS:
        item = by_id[asset_id]
        assets.append(
            {
                "asset_id": asset_id,
                "filename": filename,
                "locator": item["locator"],
                "cell_line": cell_line,
                "region": region,
                "listing_status": PASS if asset_listing_pass else UNKNOWN,
            }
        )

    endpoint = copy.deepcopy(protocol["endpoint_authority"])
    endpoint["status"] = PASS if endpoint_pass else UNKNOWN
    outlier = {
        "status": BLOCKED,
        "author_defined_rule": None,
        "applies_unambiguously_to_ordinary_row_level_effect_production": False,
        "requires_row_value_inference": True,
        "public_authority_ids": ["ELIFE_97682_VERSION_OF_RECORD"],
        "blocker": protocol["author_defined_outlier_policy"]["blocker"],
    }
    replicate = copy.deepcopy(protocol["replicate_authority"])
    replicate["status"] = PASS if replicate_pass else UNKNOWN
    grouping = copy.deepcopy(protocol["biological_grouping_authority"])
    context = copy.deepcopy(protocol["full_context_reconstruction_authority"])
    return (
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "record_type": "OFFICIAL_PUBLIC_AUTHORITY_AGGREGATE_ONLY",
            "dataset_id": DATASET_ID,
            "scope": {
                "ordinary_public_only": True,
                "aggregate_only": True,
                "row_values_read": False,
                "sequence_values_read": False,
                "effect_values_read": False,
                "raw_processed_asset_downloaded": False,
                "raw_processed_asset_opened": False,
                "restricted_or_sealed_required": False,
                "GSE246381_contact": False,
            },
            "official_assets": assets,
            "endpoint_authority": endpoint,
            "author_defined_outlier_policy": outlier,
            "replicate_authority": replicate,
            "biological_grouping_authority": grouping,
            "full_context_reconstruction_authority": context,
            "header_names": [],
        },
        sources,
    )


def _validate_status(value: Any, *, label: str) -> str:
    if value not in GATE_STATUSES:
        raise ObservationError(f"{label}.status is invalid")
    return str(value)


def validate_observation(
    protocol: Mapping[str, Any], observation: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "record_type",
        "dataset_id",
        "scope",
        "official_assets",
        "endpoint_authority",
        "author_defined_outlier_policy",
        "replicate_authority",
        "biological_grouping_authority",
        "full_context_reconstruction_authority",
        "header_names",
    }
    if set(observation) != expected_keys:
        raise ObservationError("existing aggregate top-level shape differs")
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ObservationError("existing aggregate schema version differs")
    if observation.get("record_type") != "OFFICIAL_PUBLIC_AUTHORITY_AGGREGATE_ONLY":
        raise ObservationError("existing input is not an official aggregate-only record")
    if observation.get("dataset_id") != DATASET_ID:
        raise ObservationError("existing aggregate is not GSE217518")

    scope = observation.get("scope")
    expected_scope_keys = {
        "ordinary_public_only",
        "aggregate_only",
        "row_values_read",
        "sequence_values_read",
        "effect_values_read",
        "raw_processed_asset_downloaded",
        "raw_processed_asset_opened",
        "restricted_or_sealed_required",
        "GSE246381_contact",
    }
    if not isinstance(scope, dict) or set(scope) != expected_scope_keys:
        raise ObservationError("scope attestation shape differs")
    for key in ("ordinary_public_only", "aggregate_only"):
        if not isinstance(scope[key], bool):
            raise ObservationError(f"scope.{key} must be boolean")
    for key in expected_scope_keys - {"ordinary_public_only", "aggregate_only"}:
        if not isinstance(scope[key], bool):
            raise ObservationError(f"scope.{key} must be boolean")

    assets = observation.get("official_assets")
    if not isinstance(assets, list) or len(assets) != 4:
        raise ObservationError("existing aggregate must describe exact four assets")
    protocol_assets = {
        item["asset_id"]: item for item in protocol["official_processed_assets"]
    }
    seen: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {
            "asset_id",
            "filename",
            "locator",
            "cell_line",
            "region",
            "listing_status",
        }:
            raise ObservationError("asset aggregate shape differs")
        asset_id = str(asset["asset_id"])
        if asset_id in seen or asset_id not in protocol_assets:
            raise ObservationError("asset identity is duplicate or unrecognized")
        seen.add(asset_id)
        expected = protocol_assets[asset_id]
        for key in ("filename", "locator", "cell_line", "region"):
            if asset[key] != expected[key]:
                raise ObservationError(f"{asset_id}.{key} differs from official authority")
        _validate_status(asset["listing_status"], label=asset_id)

    endpoint = observation.get("endpoint_authority")
    if not isinstance(endpoint, dict):
        raise ObservationError("endpoint authority is absent")
    if set(endpoint) != set(protocol["endpoint_authority"]):
        raise ObservationError("endpoint authority shape differs")
    _validate_status(endpoint.get("status"), label="endpoint_authority")
    for key in (
        "endpoint_identity",
        "endpoint_unit",
        "direction",
        "raw_measurement_scale",
    ):
        if endpoint.get(key) != protocol["endpoint_authority"][key]:
            raise ObservationError(f"endpoint_authority.{key} differs")

    nested_allowed_keys = {
        "author_defined_outlier_policy": {
            "status",
            "author_defined_rule",
            "applies_unambiguously_to_ordinary_row_level_effect_production",
            "requires_row_value_inference",
            "public_authority_ids",
            "blocker",
        },
        "replicate_authority": set(protocol["replicate_authority"]),
        "biological_grouping_authority": {
            "status",
            "sample_level_fields_confirmed",
            "required_source_level_fields",
            "authoritative_fields",
            "executable_source_group_definition",
            "public_authority_ids",
            "blocker",
        },
        "full_context_reconstruction_authority": {
            "status",
            "measured_construct_context",
            "required_public_assets",
            "verified_public_assets",
            "existing_reconstruction_observation",
            "existing_reconstruction_may_be_run_by_this_preflight",
            "reconstruction_rule",
            "source_candidate_crosswalk_rule",
            "produces_entire_refseq_utr_as_measured_context",
            "public_authority_ids",
            "blocker",
        },
    }
    for key, allowed_keys in nested_allowed_keys.items():
        value = observation.get(key)
        if not isinstance(value, dict):
            raise ObservationError(f"{key} is absent")
        if not set(value).issubset(allowed_keys):
            raise ObservationError(f"{key} contains a payload or unsupported field")
        _validate_status(value.get("status"), label=key)
    headers = observation.get("header_names")
    if not isinstance(headers, list) or not all(isinstance(x, str) for x in headers):
        raise ObservationError("header_names must contain names only")


def _gate(
    gate_id: str,
    status: str,
    finding: str,
    *,
    blocker: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "gate_id": gate_id,
        "status": status,
        "finding": finding,
    }
    if blocker is not None:
        record["blocker"] = blocker
    return record


def evaluate_observation(
    protocol: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    source_mode: str,
    source_results: list[dict[str, str]],
    recorded_at: str | None = None,
) -> dict[str, Any]:
    validate_observation(protocol, observation)
    scope = observation["scope"]
    scope_pass = (
        scope["ordinary_public_only"]
        and scope["aggregate_only"]
        and not any(
            scope[key]
            for key in (
                "row_values_read",
                "sequence_values_read",
                "effect_values_read",
                "raw_processed_asset_downloaded",
                "raw_processed_asset_opened",
                "restricted_or_sealed_required",
                "GSE246381_contact",
            )
        )
    )
    scope_status = PASS if scope_pass else BLOCKED

    assets = observation["official_assets"]
    asset_statuses = {asset["listing_status"] for asset in assets}
    if asset_statuses == {PASS}:
        asset_status = PASS
    elif BLOCKED in asset_statuses:
        asset_status = BLOCKED
    else:
        asset_status = UNKNOWN

    endpoint = observation["endpoint_authority"]
    endpoint_status = str(endpoint["status"])
    outlier = observation["author_defined_outlier_policy"]
    outlier_status = str(outlier["status"])
    if outlier_status == PASS:
        if (
            not isinstance(outlier.get("author_defined_rule"), str)
            or not outlier["author_defined_rule"].strip()
            or outlier.get("requires_row_value_inference") is not False
            or outlier.get(
                "applies_unambiguously_to_ordinary_row_level_effect_production"
            )
            is not True
            or not outlier.get("public_authority_ids")
        ):
            outlier_status = BLOCKED

    replicate = observation["replicate_authority"]
    replicate_status = str(replicate["status"])
    if replicate_status == PASS and (
        replicate.get("replicate_count_per_cell_region") != 3
        or set(replicate.get("required_row_mapping_fields", []))
        != set(protocol["replicate_authority"]["required_row_mapping_fields"])
    ):
        replicate_status = BLOCKED

    grouping = observation["biological_grouping_authority"]
    grouping_status = str(grouping["status"])
    if grouping_status == PASS:
        required = set(
            protocol["biological_grouping_authority"]["required_source_level_fields"]
        )
        if (
            not required.issubset(set(grouping.get("authoritative_fields", [])))
            or not isinstance(grouping.get("executable_source_group_definition"), str)
            or not grouping["executable_source_group_definition"].strip()
            or not grouping.get("public_authority_ids")
        ):
            grouping_status = BLOCKED

    context = observation["full_context_reconstruction_authority"]
    context_status = str(context["status"])
    if context_status == PASS:
        required_assets = set(
            protocol["full_context_reconstruction_authority"][
                "required_public_assets"
            ]
        )
        if (
            not required_assets.issubset(set(context.get("verified_public_assets", [])))
            or context.get("measured_construct_context")
            != protocol["full_context_reconstruction_authority"][
                "measured_construct_context"
            ]
            or not isinstance(context.get("reconstruction_rule"), str)
            or not context["reconstruction_rule"].strip()
            or not isinstance(context.get("source_candidate_crosswalk_rule"), str)
            or not context["source_candidate_crosswalk_rule"].strip()
            or context.get("produces_entire_refseq_utr_as_measured_context") is not False
            or not context.get("public_authority_ids")
        ):
            context_status = BLOCKED

    gates = [
        _gate(
            "ORDINARY_PUBLIC_AGGREGATE_SCOPE",
            scope_status,
            "Only ordinary-public metadata/listing/header aggregates were used."
            if scope_pass
            else "The observation needs or contacted a forbidden payload/source surface.",
            blocker=None if scope_pass else "NONPUBLIC_OR_PAYLOAD_SCOPE_NOT_ALLOWED",
        ),
        _gate(
            "OFFICIAL_U3_U5_BY_HEK_SH_ASSET_LISTING",
            asset_status,
            "All four official U3/U5 by HEK/SH processed asset locators are present."
            if asset_status == PASS
            else "The exact four-asset official listing is not fully established.",
            blocker=None
            if asset_status == PASS
            else "OFFICIAL_FOUR_ASSET_LISTING_NOT_PASS",
        ),
        _gate(
            "ENDPOINT_IDENTITY_DIRECTION_AND_RAW_SCALE",
            endpoint_status,
            "RNA half-life, direction, count scale, transform, and units are explicit."
            if endpoint_status == PASS
            else "Endpoint identity, direction, or raw scale is not fully explicit.",
            blocker=None
            if endpoint_status == PASS
            else "ENDPOINT_IDENTITY_DIRECTION_RAW_SCALE_NOT_PASS",
        ),
        _gate(
            "AUTHOR_DEFINED_OUTLIER_POLICY",
            outlier_status,
            "An executable author-defined ordinary-producer rule is explicit."
            if outlier_status == PASS
            else "No executable author-defined ordinary-producer outlier rule is available; it may not be inferred from row values.",
            blocker=None
            if outlier_status == PASS
            else "AUTHOR_DEFINED_OUTLIER_POLICY_NOT_EXECUTABLE",
        ),
        _gate(
            "REPLICATE_AUTHORITY",
            replicate_status,
            "Three independent experiments and their cell/region/time mappings are explicit."
            if replicate_status == PASS
            else "Replicate count or mapping authority is incomplete.",
            blocker=None
            if replicate_status == PASS
            else "REPLICATE_AUTHORITY_NOT_PASS",
        ),
        _gate(
            "BIOLOGICAL_GROUPING_FIELDS",
            grouping_status,
            "The source-level grouping fields and executable grouping rule are public-authority bound."
            if grouping_status == PASS
            else "Sample-level context exists, but the row-level source/biological-group crosswalk is not closed.",
            blocker=None
            if grouping_status == PASS
            else "BIOLOGICAL_GROUPING_FIELDS_NOT_PASS",
        ),
        _gate(
            "MEASURED_FULL_CONTEXT_RECONSTRUCTION",
            context_status,
            "The measured 115-bp WT/mutant reporter context can be reconstructed from public authority."
            if context_status == PASS
            else "The measured 115-bp reporter fragment/source context is not executable; the existing entire-RefSeq-UTR reconstruction is not equivalent.",
            blocker=None
            if context_status == PASS
            else "MEASURED_FULL_CONTEXT_RECONSTRUCTION_NOT_PASS",
        ),
    ]
    ready = all(gate["status"] == PASS for gate in gates)
    status = (
        "READY_FOR_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
        if ready
        else "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
    )
    blockers = [gate["blocker"] for gate in gates if "blocker" in gate]
    counts = {state: sum(g["status"] == state for g in gates) for state in GATE_STATUSES}
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": PROTOCOL_ID,
        "record_type": "PUBLIC_AUTHORITY_PREFLIGHT_AGGREGATE_ONLY",
        "dataset_id": DATASET_ID,
        "recorded_at": timestamp,
        "status": status,
        "ready_for_ordinary_public_row_level_producer": ready,
        "implementation_binding": dict(binding),
        "authority_observation": {
            "source_mode": source_mode,
            "source_results": source_results,
            "official_asset_count": len(assets),
            "header_names_observed": list(observation["header_names"]),
        },
        "gates": gates,
        "gate_counts": counts,
        "blockers": blockers,
        "official_assets": assets,
        "endpoint_authority": {
            key: endpoint[key]
            for key in (
                "endpoint_identity",
                "endpoint_unit",
                "direction",
                "raw_measurement_scale",
                "normalization",
                "decay_model",
                "derived_endpoint",
                "effect_orientation",
                "half_life_qc",
            )
            if key in endpoint
        },
        "scope_attestation": {
            "ordinary_public_only": scope["ordinary_public_only"],
            "aggregate_only": scope["aggregate_only"],
            "row_values_read": scope["row_values_read"],
            "sequence_values_read": scope["sequence_values_read"],
            "effect_values_read": scope["effect_values_read"],
            "processed_asset_download_count": 0,
            "processed_asset_open_count": 0,
            "supplement_payload_open_count": 0,
            "restricted_or_sealed_required": scope[
                "restricted_or_sealed_required"
            ],
            "restricted_or_sealed_contact": False,
            "GSE246381_contact": scope["GSE246381_contact"],
            "reconstruction_run_count": 0,
            "qualifier_run_count": 0,
            "canonical_materialization_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
        },
        "terminal_truth": {
            "qualified": False,
            "canonical_record_count": 0,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        },
        "sole_next_action": (
            "WRITE_THE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER_UNDER_A_SEPARATE_AUTHORIZED_TASK"
            if ready
            else "OBTAIN_AN_EXECUTABLE_PRIMARY_PUBLIC_OUTLIER_RULE_AND_AUTHORITATIVE_115_BP_SOURCE_GROUP_CROSSWALK"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def execute(
    protocol_path: Path,
    output_dir: Path,
    *,
    authority_aggregate_path: Path | None = None,
    fetcher: MetadataFetcher | None = None,
    binding_auditor: Callable[
        [Mapping[str, Any], Path, bytes], Mapping[str, str]
    ] = _default_binding_auditor,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    protocol, protocol_payload = load_protocol(protocol_path)
    binding = dict(binding_auditor(protocol, protocol_path, protocol_payload))

    if authority_aggregate_path is None:
        observation, source_results = build_live_observation(
            protocol, fetcher or UrllibMetadataFetcher()
        )
        source_mode = "LIVE_OFFICIAL_METADATA_AND_FILE_LISTING"
    else:
        observation, _ = _read_json(
            authority_aggregate_path, label="official public authority aggregate"
        )
        source_results = []
        source_mode = "EXISTING_OFFICIAL_PUBLIC_AUTHORITY_AGGREGATE_ONLY"

    report = evaluate_observation(
        protocol,
        observation,
        binding=binding,
        source_mode=source_mode,
        source_results=source_results,
        recorded_at=recorded_at,
    )
    output_path = output_dir / REPORT_FILENAME
    if output_path.exists():
        raise OutputError(f"refusing to overwrite existing report: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate-only GSE217518 public-authority preflight"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--authority-aggregate",
        type=Path,
        default=None,
        help="Existing official-public aggregate; omit for live GEO/eLife metadata",
    )
    parser.add_argument("--recorded-at", default=None)
    args = parser.parse_args(argv)
    try:
        report = execute(
            args.protocol,
            args.output_dir,
            authority_aggregate_path=args.authority_aggregate,
            recorded_at=args.recorded_at,
        )
    except PreflightError as exc:
        print(f"STOP: {exc}")
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "ready_for_ordinary_public_row_level_producer": report[
                    "ready_for_ordinary_public_row_level_producer"
                ],
                "report": str(args.output_dir / REPORT_FILENAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
