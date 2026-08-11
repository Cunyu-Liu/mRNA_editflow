#!/usr/bin/env python3
"""Produce the closed GSE200304 upstream source/viability audit bundle.

This producer is deliberately audit-only.  It verifies the frozen published-
endpoint aggregate authority, same-descriptor hashes the exact ordinary-public
source bundle, downloads three exact official upstream sources, and emits only
aggregate viability facts.  It never invokes a consumer, adjudicator,
qualifier, canonicalizer, raw replay, model, or training path.

The output is an NFS-safe six-member directory: three verbatim source files, a
single closed audit JSON, SHA256SUMS, and a terminal marker created last.  Any
pre-terminal failure preserves the partial directory for manual adjudication;
no automatic cleanup or overwrite path exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import posixpath
import re
import ssl
import stat
import subprocess
import sys
import zlib
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)
from xml.etree import ElementTree as ET


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

SCHEMA_VERSION = "route_a_v3_gse200304_upstream_authority_viability.v1"
PROTOCOL_ID = "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1"
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
PHASE_ID = "A1"
DECISION_ID = "V3-DEC-019"
MODE = "AUDIT_ONLY_NO_GATE_CHANGE"

CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_upstream_authority_viability_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/produce_gse200304_upstream_authority_viability.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_produce_gse200304_upstream_authority_viability.py"
)
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
PRODUCTION_SCRIPT_PATH = PRODUCTION_REPO_ROOT / SCRIPT_REPO_PATH
BRANCH = "routea-v3-a1-20260810"
IMPLEMENTATION_BASE_COMMIT = "0b95ac77a44644e57cc4d0bfb31a9154238fdca6"
EXPECTED_I_PATHS = [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]
EXPECTED_B_PATHS = [CONFIG_REPO_PATH]
I_TO_B_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
FROZEN_CONFIG_CORE_SHA256 = "d02928ce40f4a77f356cba47929b96d6b07cd8f4c0f9facb3a4156887719f15f"

JATS_CONFIG_KEY = "europe_pmc_jats"
SOFT_CONFIG_KEY = "gse200302_soft_gz"
MATRIX_CONFIG_KEY = "gse200302_log2_cpm_matrix_gz"
AUDIT_NAME = "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json"
CHECKSUMS_NAME = "SHA256SUMS"
MARKER_NAME = "PUBLICATION_COMMIT.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,191}$")
CELL_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
SOFT_RECORD = re.compile(r"^\^([A-Z]+) = (\S+)$")
MATRIX_HEADER = re.compile(
    r"^(80S_RNA|High_Poly|Low_Poly|pDNA|Total_RNA)_"
    r"([1-6])_(S[0-9]+)_(WT|Mutant)$"
)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_JATS_BYTES = 512 * 1024
MAX_SOFT_PLAIN_BYTES = 2 * 1024 * 1024
MAX_MATRIX_PLAIN_BYTES = 12 * 1024 * 1024
MAX_XLSX_MEMBER_BYTES = 8 * 1024 * 1024

CONFIG_TOP_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "dataset_id",
    "decision_id",
    "mode",
    "implementation_binding",
    "repository_authority",
    "scope",
    "predecessor_authority",
    "public_sources",
    "viability_contract",
    "decision_boundary",
    "output_contract",
}
DERIVED_FORBIDDEN_KEYS = {
    "gene",
    "genes",
    "row_id",
    "row_ids",
    "pair_id",
    "pair_ids",
    "sequence",
    "sequences",
    "effect_value",
    "effect_values",
    "raw_row",
    "raw_rows",
}


class AuditError(RuntimeError):
    """Base class for all fail-closed producer errors."""


class ConfigError(AuditError):
    """The frozen config or its I/B lifecycle is invalid."""


class BindingError(AuditError):
    """Git implementation authority is not exact."""


class InputIntegrityError(AuditError):
    """A frozen local or downloaded source differs."""


class SourceSemanticConflict(AuditError):
    """Exact source bytes do not support the frozen aggregate semantics."""


class PublicationError(AuditError):
    """Publication could not be proved exact and durable."""


class PartialPublicationError(PublicationError):
    """A newly created output is intentionally preserved for manual review."""


class ExistingOutputRequiresManualReview(PublicationError):
    """An existing output is not the exact idempotent bundle."""


class PostMarkerCommitOutcomeIndeterminate(PublicationError):
    """A terminal marker exists but canonical durable exact6 cannot be proved."""


@dataclass(frozen=True)
class S3SelectiveState:
    keys: frozenset[str]
    finite_totalpoly_keys: frozenset[str]
    row_count: int


@dataclass(frozen=True)
class PredecessorSummary:
    published_endpoint_config_sha256: str
    published_endpoint_trio_manifest_sha256: str
    source_exact7_manifest_sha256: str
    published_endpoint_bundle_manifest_sha256: str
    source_exact7_member_count: int
    published_endpoint_bundle_member_count: int
    s3: S3SelectiveState

    def aggregate_dict(self) -> dict[str, Any]:
        return {
            "published_endpoint_config_sha256": self.published_endpoint_config_sha256,
            "published_endpoint_trio_manifest_sha256": (
                self.published_endpoint_trio_manifest_sha256
            ),
            "source_exact7_manifest_sha256": self.source_exact7_manifest_sha256,
            "published_endpoint_bundle_manifest_sha256": (
                self.published_endpoint_bundle_manifest_sha256
            ),
            "source_exact7_member_count": self.source_exact7_member_count,
            "published_endpoint_bundle_member_count": (
                self.published_endpoint_bundle_member_count
            ),
            "table_s3_selective_pair_count": len(self.s3.keys),
            "table_s3_finite_totalpoly_pair_count": len(
                self.s3.finite_totalpoly_keys
            ),
            "table_s3_gene_column_selected_or_persisted": False,
            "table_s3_translation_significance_selected_or_persisted": False,
        }


@dataclass(frozen=True)
class MatrixState:
    keys: frozenset[str]
    row_count: int
    header_field_count: int


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    if len(payload) > MAX_JSON_BYTES:
        raise InputIntegrityError(f"{label} exceeds the JSON byte bound")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON token: {token}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InputIntegrityError(f"{label} is not strict finite UTF-8 JSON") from exc
    if type(value) is not dict:
        raise InputIntegrityError(f"{label} JSON root is not an object")
    return value


def config_core_sha256(config: Mapping[str, Any]) -> str:
    projection = copy.deepcopy(dict(config))
    binding = projection.get("implementation_binding")
    if type(binding) is not dict:
        raise ConfigError("implementation_binding is not an object")
    for path in I_TO_B_SCALAR_PATHS:
        binding[path.rsplit(".", 1)[1]] = UNKNOWN
    binding.pop("config_core_sha256", None)
    return sha256(canonical_json_bytes(projection))


def _semantic_diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "$"}
    if type(left) is dict:
        paths: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_semantic_diff_paths(left[key], right[key], child))
        return paths
    if type(left) is list:
        if len(left) != len(right):
            return {prefix or "$"}
        paths: set[str] = set()
        for index, (one, two) in enumerate(zip(left, right)):
            paths.update(_semantic_diff_paths(one, two, f"{prefix}[{index}]"))
        return paths
    return set() if left == right else {prefix or "$"}


def _manifest_sha256(members: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "name": item["name"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in sorted(members, key=lambda item: str(item["name"]))
    ]
    return sha256(canonical_json_bytes(normalized))


def validate_static_config(config: Mapping[str, Any]) -> None:
    if type(config) is not dict or set(config) != CONFIG_TOP_KEYS:
        raise ConfigError("config top-level keys differ from the closed schema")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "mode": MODE,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise ConfigError(f"config scalar drifted: {key}")

    binding = config["implementation_binding"]
    if type(binding) is not dict:
        raise ConfigError("implementation binding is not an object")
    if binding.get("unknown_to_bound_scalar_paths") != I_TO_B_SCALAR_PATHS:
        raise ConfigError("I/B scalar allowlist differs")
    if binding.get("implementation_script_path") != SCRIPT_REPO_PATH:
        raise ConfigError("implementation script path differs")
    if binding.get("implementation_test_path") != TEST_REPO_PATH:
        raise ConfigError("implementation test path differs")
    frozen_core = binding.get("config_core_sha256")
    if not isinstance(frozen_core, str) or HEX64.fullmatch(frozen_core) is None:
        raise ConfigError("config core hash is invalid")
    if frozen_core != FROZEN_CONFIG_CORE_SHA256:
        raise ConfigError("compiled config core hash differs")
    if config_core_sha256(config) != frozen_core:
        raise ConfigError("config core projection differs")

    four_values = {
        "status": binding.get("status"),
        "implementation_commit": binding.get("implementation_commit"),
        "implementation_script_sha256": binding.get(
            "implementation_script_sha256"
        ),
        "implementation_test_sha256": binding.get("implementation_test_sha256"),
    }
    if four_values["status"] == UNKNOWN:
        if set(four_values.values()) != {UNKNOWN}:
            raise ConfigError("UNKNOWN-I binding is not exact four-scalar UNKNOWN")
    elif four_values["status"] == BOUND:
        if HEX40.fullmatch(str(four_values["implementation_commit"])) is None:
            raise ConfigError("bound implementation commit is invalid")
        for key in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if HEX64.fullmatch(str(four_values[key])) is None:
                raise ConfigError(f"bound hash is invalid: {key}")
    else:
        raise ConfigError("implementation status is outside the closed enum")

    repository = config["repository_authority"]
    if repository.get("production_repo_root") != os.fspath(PRODUCTION_REPO_ROOT):
        raise ConfigError("production repository root differs")
    if repository.get("branch") != BRANCH:
        raise ConfigError("production branch differs")
    if repository.get("implementation_base_commit") != IMPLEMENTATION_BASE_COMMIT:
        raise ConfigError("implementation base differs")
    if repository.get("implementation_commit_exact_changed_paths") != EXPECTED_I_PATHS:
        raise ConfigError("implementation changed-path contract differs")
    if repository.get("binding_commit_exact_changed_paths") != EXPECTED_B_PATHS:
        raise ConfigError("binding changed-path contract differs")

    output = config["output_contract"]
    source_names = [
        config["public_sources"][key]["output_name"]
        for key in (JATS_CONFIG_KEY, SOFT_CONFIG_KEY, MATRIX_CONFIG_KEY)
    ]
    expected_output_names = [
        *source_names,
        AUDIT_NAME,
        CHECKSUMS_NAME,
        MARKER_NAME,
    ]
    if output.get("exact_member_names") != expected_output_names:
        raise ConfigError("output exact-six member order differs")
    if output.get("audit_name") != AUDIT_NAME:
        raise ConfigError("audit output name differs")
    if output.get("checksums_name") != CHECKSUMS_NAME:
        raise ConfigError("checksum output name differs")
    if output.get("terminal_marker_name") != MARKER_NAME:
        raise ConfigError("terminal output name differs")
    if output.get("terminal_marker_written_last") is not True:
        raise ConfigError("terminal-marker-last contract is disabled")
    if output.get("no_overwrite") is not True:
        raise ConfigError("no-overwrite contract is disabled")

    decision = config["decision_boundary"]
    zero_keys = (
        "ordinary_study_contribution",
        "a1_intervention_study_contribution",
        "true_a2_dense_study_contribution",
        "canonical_record_count",
        "gate_records_written",
    )
    if any(decision.get(key) != 0 for key in zero_keys):
        raise ConfigError("decision-neutral zero count drifted")
    false_keys = (
        "qualified",
        "canonical_materialization_allowed",
        "training_allowed",
        "model_selection_allowed",
        "next_phase_authorized",
        "consumer_run",
        "adjudicator_run",
        "qualifier_run",
        "raw_replay_run",
        "row_mapping_producer_run",
    )
    if any(decision.get(key) is not False for key in false_keys):
        raise ConfigError("decision-neutral false boundary drifted")

    viability = config["viability_contract"]
    endpoint = viability["canonical_reported_endpoint_semantics"]
    replicate = viability["row_replicate_or_valid_se"]
    rights = viability["license_rights"]
    group = viability["biological_group_authority"]
    if endpoint.get("status_if_all_source_checks_pass") != (
        "READY_FOR_PASS_RECORD_NOT_YET_BOUND"
    ) or endpoint.get("consumer_gate_pass") is not False:
        raise ConfigError("endpoint readiness boundary differs")
    if replicate.get("status_if_all_source_checks_pass") != (
        "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND"
    ) or replicate.get("consumer_gate_pass") is not False:
        raise ConfigError("replicate readiness boundary differs")
    if rights.get("status_if_all_source_checks_pass") != (
        "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND"
    ) or rights.get("consumer_gate_pass") is not False:
        raise ConfigError("rights readiness boundary differs")
    if group.get("status") != "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT":
        raise ConfigError("group blocker differs")
    if replicate.get("standard_error_status") != (
        "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED"
    ):
        raise ConfigError("standard-error boundary differs")


def validate_i_to_b_config_pair(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    implementation_commit: str,
    script_sha256: str,
    test_sha256: str,
) -> None:
    validate_static_config(i_config)
    validate_static_config(b_config)
    i_binding = i_config["implementation_binding"]
    b_binding = b_config["implementation_binding"]
    if any(
        i_binding.get(key) != UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    ):
        raise BindingError("implementation config is not exact UNKNOWN-I")
    expected_bound = {
        "status": BOUND,
        "implementation_commit": implementation_commit,
        "implementation_script_sha256": script_sha256,
        "implementation_test_sha256": test_sha256,
    }
    if any(b_binding.get(key) != value for key, value in expected_bound.items()):
        raise BindingError("bound config four-scalar identity differs")
    if _semantic_diff_paths(i_config, b_config) != set(I_TO_B_SCALAR_PATHS):
        raise BindingError("I-to-B semantic diff is not the exact four scalars")


def _validate_production_entrypoint_paths(
    *, config_path: Path, script_path: Path
) -> None:
    if not config_path.is_absolute() or config_path != PRODUCTION_CONFIG_PATH:
        raise BindingError("production config path is not the canonical authority path")
    if not script_path.is_absolute() or script_path != PRODUCTION_SCRIPT_PATH:
        raise BindingError("executed producer is not the canonical production script")


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path != PRODUCTION_CONFIG_PATH:
        raise ConfigError("config path is not the canonical production path")
    payload = _read_bounded_regular_relative(
        PRODUCTION_REPO_ROOT,
        CONFIG_REPO_PATH,
        maximum_bytes=MAX_JSON_BYTES,
        label="canonical production config",
    )
    config = strict_json(payload, label="producer config")
    validate_static_config(config)
    return config, payload


def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingError(f"git authority check failed: {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BindingError(
            f"git blob authority check failed: {' '.join(arguments)}"
        ) from exc
    return result.stdout


def _single_parent(repo: Path, commit: str, expected_parent: str, *, label: str) -> None:
    tokens = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    if tokens != [commit, expected_parent]:
        raise BindingError(f"{label} is not the required direct child")


def _changed_paths(repo: Path, commit: str) -> list[str]:
    output = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return sorted(line for line in output.splitlines() if line)


def _git_blob(repo: Path, commit: str, path: str, expected_sha256: str | None = None) -> bytes:
    listing = _git(repo, "ls-tree", commit, "--", path).split()
    if len(listing) < 4 or listing[0] != "100644" or listing[1] != "blob":
        raise BindingError(f"authority Git mode/type differs: {path}")
    payload = _git_bytes(repo, "show", f"{commit}:{path}")
    if expected_sha256 is not None and sha256(payload) != expected_sha256:
        raise BindingError(f"authority Git blob hash differs: {path}")
    return payload


def validate_production_authority(
    config: Mapping[str, Any],
    config_payload: bytes,
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    script_path: Path | None = None,
) -> dict[str, str]:
    """Validate the exact two-commit I/B lifecycle before any source/output."""

    executed_script = script_path or Path(os.path.abspath(__file__))
    _validate_production_entrypoint_paths(
        config_path=config_path,
        script_path=executed_script,
    )
    validate_static_config(config)
    binding = config["implementation_binding"]
    if binding["status"] != BOUND:
        raise BindingError(
            "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED; stopped before source/output"
        )
    repo = Path(config["repository_authority"]["production_repo_root"])
    if repo != PRODUCTION_REPO_ROOT:
        raise BindingError("production repository root differs")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        raise BindingError("production branch differs")
    if _git(repo, "status", "--porcelain"):
        raise BindingError("production worktree is not clean")
    if _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}") != f"origin/{BRANCH}":
        raise BindingError("production upstream name differs")
    if _git(repo, "rev-parse", "@{upstream}") != head:
        raise BindingError("production HEAD is not exactly pushed upstream")

    implementation = binding["implementation_commit"]
    _single_parent(
        repo,
        implementation,
        IMPLEMENTATION_BASE_COMMIT,
        label="producer implementation commit",
    )
    if _changed_paths(repo, implementation) != sorted(EXPECTED_I_PATHS):
        raise BindingError("producer implementation changed-path set differs")
    _single_parent(repo, head, implementation, label="producer binding commit")
    if _changed_paths(repo, head) != EXPECTED_B_PATHS:
        raise BindingError("producer binding commit is not exact config-only")

    i_payload = _git_blob(repo, implementation, CONFIG_REPO_PATH)
    b_payload = _git_blob(repo, head, CONFIG_REPO_PATH)
    if b_payload != config_payload:
        raise BindingError("running bound config bytes differ from HEAD")
    i_config = strict_json(i_payload, label="producer UNKNOWN-I config")
    b_config = strict_json(b_payload, label="producer B config")
    validate_i_to_b_config_pair(
        i_config,
        b_config,
        implementation_commit=implementation,
        script_sha256=binding["implementation_script_sha256"],
        test_sha256=binding["implementation_test_sha256"],
    )
    worktree_blobs: dict[str, bytes] = {CONFIG_REPO_PATH: b_payload}
    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        implementation_blob = _git_blob(repo, implementation, path, digest)
        current_blob = _git_blob(repo, head, path, digest)
        if implementation_blob != current_blob:
            raise BindingError(f"bound producer blob drifted: {path}")
        worktree_blobs[path] = current_blob

    for path, expected_blob in worktree_blobs.items():
        observed = _read_exact_relative(
            repo,
            path,
            expected_bytes=len(expected_blob),
            expected_sha256=sha256(expected_blob),
            collect=True,
            label=f"canonical worktree authority {path}",
        )
        if observed != expected_blob:
            raise BindingError(f"canonical worktree bytes differ from HEAD: {path}")
    return {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": implementation,
        "binding_commit": head,
        "implementation_script_sha256": binding["implementation_script_sha256"],
        "implementation_test_sha256": binding["implementation_test_sha256"],
        "config_core_sha256": binding["config_core_sha256"],
    }


def _safe_relative_parts(path: str, *, label: str) -> tuple[str, ...]:
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in path
    ):
        raise InputIntegrityError(f"{label} is not a safe relative path")
    return pure.parts


def _open_absolute_directory(path: Path, *, label: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not path.is_absolute() or path.parts[0] != os.sep:
        raise InputIntegrityError(f"{label} is not absolute")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise InputIntegrityError(f"{label} contains an unsafe component")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise InputIntegrityError(f"{label} is not a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _directory_dev_ino(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise InputIntegrityError("directory identity was requested for a non-directory")
    return value.st_dev, value.st_ino


def _read_bounded_regular_member_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if SAFE_BASENAME.fullmatch(name) is None or "/" in name:
        raise InputIntegrityError(f"{label} basename is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise InputIntegrityError(f"{label} cannot be opened") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > maximum_bytes
        ):
            raise InputIntegrityError(f"{label} type/link/size is unsafe")
        identity = _file_identity(opened)
        chunks: list[bytes] = []
        count = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            count += len(block)
            if count > opened.st_size or count > maximum_bytes:
                raise InputIntegrityError(f"{label} grew while read")
            chunks.append(block)
        final = os.fstat(descriptor)
        if _file_identity(final) != identity or count != opened.st_size:
            raise InputIntegrityError(f"{label} changed during same-FD read")
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(path_stat) != identity:
            raise InputIntegrityError(f"{label} path binding changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_exact_member_at(
    directory_fd: int,
    name: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    collect: bool,
    label: str,
) -> bytes | None:
    if SAFE_BASENAME.fullmatch(name) is None or "/" in name:
        raise InputIntegrityError(f"{label} basename is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise InputIntegrityError(f"{label} cannot be opened") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_bytes
        ):
            raise InputIntegrityError(f"{label} type/link/size differs")
        identity = _file_identity(opened)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        count = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            count += len(block)
            if count > expected_bytes:
                raise InputIntegrityError(f"{label} grew while read")
            digest.update(block)
            if collect:
                chunks.append(block)
        final = os.fstat(descriptor)
        if _file_identity(final) != identity:
            raise InputIntegrityError(f"{label} changed during same-FD read")
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _file_identity(path_stat) != identity:
            raise InputIntegrityError(f"{label} path binding changed")
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise InputIntegrityError(f"{label} byte/hash authority differs")
        return b"".join(chunks) if collect else None
    finally:
        os.close(descriptor)


def _read_exact_relative(
    root: Path,
    relative_path: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
    collect: bool,
    label: str,
) -> bytes | None:
    parts = _safe_relative_parts(relative_path, label=label)
    root_fd = _open_absolute_directory(root, label="repository root")
    parent_fd = root_fd
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_fd)
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = child
        return _read_exact_member_at(
            parent_fd,
            parts[-1],
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            collect=collect,
            label=label,
        )
    finally:
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _read_bounded_regular_relative(
    root: Path,
    relative_path: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    parts = _safe_relative_parts(relative_path, label=label)
    root_fd = _open_absolute_directory(root, label=f"{label} root")
    parent_fd = root_fd
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_fd)
            if parent_fd != root_fd:
                os.close(parent_fd)
            parent_fd = child
        return _read_bounded_regular_member_at(
            parent_fd,
            parts[-1],
            maximum_bytes=maximum_bytes,
            label=label,
        )
    finally:
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalized_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _xlsx_cell_token(cell: ET.Element) -> tuple[str, str]:
    cell_type = cell.attrib.get("t", "n")
    values = [child.text or "" for child in cell if _local_name(child.tag) == "v"]
    if len(values) != 1:
        raise SourceSemanticConflict("selected Table S3 cell lacks one cached value")
    value = values[0]
    if cell_type == "s":
        if not value.isdigit():
            raise SourceSemanticConflict("Table S3 shared-string index is invalid")
        return "shared", value
    if cell_type in {"n", ""}:
        return "numeric", value
    raise SourceSemanticConflict(
        f"selected Table S3 cell type is outside the closed subset: {cell_type}"
    )


def _select_shared_strings(payload: bytes, wanted: set[int]) -> dict[int, str]:
    if len(payload) > MAX_XLSX_MEMBER_BYTES:
        raise SourceSemanticConflict("Table S3 sharedStrings exceeds the byte bound")
    resolved: dict[int, str] = {}
    index = -1
    try:
        iterator = ET.iterparse(io.BytesIO(payload), events=("end",))
        for _event, element in iterator:
            if _local_name(element.tag) != "si":
                continue
            index += 1
            if index in wanted:
                value = "".join(
                    child.text or ""
                    for child in element.iter()
                    if _local_name(child.tag) == "t"
                )
                if "\x00" in value:
                    raise SourceSemanticConflict("Table S3 selected string contains NUL")
                resolved[index] = value
            element.clear()
    except ET.ParseError as exc:
        raise SourceSemanticConflict("Table S3 sharedStrings XML is invalid") from exc
    if set(resolved) != wanted:
        raise SourceSemanticConflict("Table S3 selected shared strings are incomplete")
    return resolved


def audit_table_s3_selective(
    payload: bytes,
    *,
    expected_pair_count: int = 6_772,
    expected_finite_totalpoly_count: int = 6_547,
) -> S3SelectiveState:
    """Read only key/comparison and D--F finite/NA state from frozen Table S3."""

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceSemanticConflict("Table S3 XLSX container is invalid") from exc
    try:
        required = {"xl/worksheets/sheet1.xml", "xl/sharedStrings.xml"}
        if not required <= set(archive.namelist()):
            raise SourceSemanticConflict("Table S3 XLSX required members are absent")
        for name in required:
            info = archive.getinfo(name)
            if info.file_size > MAX_XLSX_MEMBER_BYTES:
                raise SourceSemanticConflict("Table S3 XLSX member exceeds byte bound")
        worksheet = archive.read("xl/worksheets/sheet1.xml")
        shared_payload = archive.read("xl/sharedStrings.xml")
    finally:
        archive.close()

    selected_columns = {"A", "C", "D", "E", "F"}
    raw_rows: list[tuple[int, dict[str, tuple[str, str]]]] = []
    wanted_shared: set[int] = set()
    try:
        iterator = ET.iterparse(io.BytesIO(worksheet), events=("end",))
        for _event, element in iterator:
            if _local_name(element.tag) != "row":
                continue
            row_number_text = element.attrib.get("r", "")
            if not row_number_text.isdigit():
                raise SourceSemanticConflict("Table S3 row number is invalid")
            row_number = int(row_number_text)
            selected: dict[str, tuple[str, str]] = {}
            for cell in element:
                if _local_name(cell.tag) != "c":
                    continue
                match = CELL_REF.fullmatch(cell.attrib.get("r", ""))
                if match is None or int(match.group(2)) != row_number:
                    raise SourceSemanticConflict("Table S3 cell reference is invalid")
                column = match.group(1)
                if column not in selected_columns:
                    continue
                if column in selected:
                    raise SourceSemanticConflict("Table S3 selected cell is duplicated")
                token = _xlsx_cell_token(cell)
                selected[column] = token
                if token[0] == "shared":
                    wanted_shared.add(int(token[1]))
            if set(selected) != selected_columns:
                raise SourceSemanticConflict("Table S3 selected row width differs")
            raw_rows.append((row_number, selected))
            element.clear()
    except ET.ParseError as exc:
        raise SourceSemanticConflict("Table S3 worksheet XML is invalid") from exc

    expected_row_count = 1 + 2 * expected_pair_count
    if (
        len(raw_rows) != expected_row_count
        or raw_rows[0][0] != 1
        or raw_rows[-1][0] != expected_row_count
    ):
        raise SourceSemanticConflict("Table S3 selected row geometry differs")
    shared = _select_shared_strings(shared_payload, wanted_shared)

    def resolve(token: tuple[str, str]) -> str:
        return shared[int(token[1])] if token[0] == "shared" else token[1]

    expected_header = {
        "A": "barcode",
        "C": "Comparison",
        "D": "xtail_log2FC_TE",
        "E": "xtail_pvalue",
        "F": "xtail_FDR",
    }
    if {
        column: resolve(raw_rows[0][1][column]) for column in selected_columns
    } != expected_header:
        raise SourceSemanticConflict("Table S3 selected header differs")

    per_key: dict[str, dict[str, bool]] = defaultdict(dict)
    for _row_number, cells in raw_rows[1:]:
        key = resolve(cells["A"])
        comparison = resolve(cells["C"])
        if not key or "\x00" in key:
            raise SourceSemanticConflict("Table S3 key is empty or invalid")
        if comparison not in {"HighPoly:RNA", "TotalPoly:RNA"}:
            raise SourceSemanticConflict("Table S3 comparison is outside the closed set")
        states: list[str] = []
        for column in ("D", "E", "F"):
            token = cells[column]
            value = resolve(token)
            if token[0] == "shared":
                if value != "NA":
                    raise SourceSemanticConflict("Table S3 selected string is not exact NA")
                states.append("NA")
            else:
                try:
                    numeric = float(value)
                except ValueError as exc:
                    raise SourceSemanticConflict(
                        "Table S3 selected statistic is not numeric"
                    ) from exc
                if not math.isfinite(numeric):
                    raise SourceSemanticConflict(
                        "Table S3 selected statistic is not finite"
                    )
                states.append("FINITE")
        if len(set(states)) != 1:
            raise SourceSemanticConflict("Table S3 D--F state is mixed within a row")
        if comparison in per_key[key]:
            raise SourceSemanticConflict("Table S3 key/comparison row is duplicated")
        per_key[key][comparison] = states[0] == "FINITE"

    expected_comparisons = {"HighPoly:RNA", "TotalPoly:RNA"}
    if len(per_key) != expected_pair_count or any(
        set(value) != expected_comparisons for value in per_key.values()
    ):
        raise SourceSemanticConflict("Table S3 pair/comparison geometry differs")
    finite_total = frozenset(
        key for key, state in per_key.items() if state["TotalPoly:RNA"]
    )
    if len(finite_total) != expected_finite_totalpoly_count:
        raise SourceSemanticConflict("Table S3 finite TotalPoly membership differs")
    return S3SelectiveState(
        keys=frozenset(per_key),
        finite_totalpoly_keys=finite_total,
        row_count=len(raw_rows) - 1,
    )


def _validate_published_endpoint_config(document: Mapping[str, Any]) -> None:
    if document.get("protocol_id") != "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_V1":
        raise SourceSemanticConflict("published endpoint protocol identity differs")
    if document.get("dataset_id") != DATASET_ID:
        raise SourceSemanticConflict("published endpoint dataset identity differs")
    endpoint = document.get("endpoint_boundary")
    if type(endpoint) is not dict:
        raise SourceSemanticConflict("published endpoint boundary is absent")
    expected = {
        "primary_endpoint_id": "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
        "primary_comparison_value": "TotalPoly:RNA",
        "primary_membership_pair_count": 6_772,
        "primary_finite_effect_pair_count": 6_547,
        "primary_complete_distinct_wt_201nt_proxy_group_count": 6_544,
        "study_level_reported_biological_replicate_count": 6,
        "wt_201nt_grouping_authority": False,
        "wt_201nt_grouping_proxy_only": True,
        "biological_source_group_authority_closed": False,
    }
    if any(endpoint.get(key) != value for key, value in expected.items()):
        raise SourceSemanticConflict("published endpoint config aggregate differs")
    boundary = document.get("decision_neutral_boundary")
    if type(boundary) is not dict:
        raise SourceSemanticConflict("published endpoint decision boundary is absent")
    if any(
        boundary.get(key) != 0
        for key in (
            "ordinary_study_contribution",
            "a1_intervention_study_contribution",
            "true_a2_dense_study_contribution",
            "canonical_record_count",
        )
    ):
        raise SourceSemanticConflict("published endpoint nonzero contribution observed")


def validate_published_endpoint_aggregate_documents(
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    endpoint_document = documents.get("PUBLISHED_ENDPOINT_AUDIT.json")
    report = documents.get("QUALIFICATION_REPORT.json")
    integrity = documents.get("INPUT_INTEGRITY_AUDIT.json")
    marker = documents.get(MARKER_NAME)
    if any(type(value) is not dict for value in (endpoint_document, report, integrity, marker)):
        raise SourceSemanticConflict("published endpoint aggregate JSON set is incomplete")

    endpoint_boundary = endpoint_document.get("endpoint_boundary")
    table_s2 = endpoint_document.get("table_s2")
    table_s3 = endpoint_document.get("table_s3")
    if any(type(value) is not dict for value in (endpoint_boundary, table_s2, table_s3)):
        raise SourceSemanticConflict("published endpoint aggregate sections are absent")
    expected_endpoint = {
        "primary_membership_pair_count": 6_772,
        "primary_finite_effect_pair_count": 6_547,
        "primary_complete_distinct_wt_201nt_proxy_group_count": 6_544,
        "study_level_reported_biological_replicate_count": 6,
        "row_level_effective_replicate_count": None,
        "standard_error": None,
        "wt_201nt_grouping_authority": False,
        "wt_201nt_grouping_proxy_only": True,
        "biological_source_group_authority_closed": False,
    }
    if any(endpoint_boundary.get(key) != value for key, value in expected_endpoint.items()):
        raise SourceSemanticConflict("published endpoint boundary aggregate differs")
    if table_s2.get("deduplicated_pair_count") != 6_885:
        raise SourceSemanticConflict("published Table S2 pair count differs")
    if table_s2.get("distinct_wt_201nt_proxy_count") != 6_882:
        raise SourceSemanticConflict("published Table S2 proxy count differs")
    if table_s3.get("primary_pair_count") != 6_772:
        raise SourceSemanticConflict("published Table S3 pair count differs")
    finite = table_s3.get("finite_statistic_rows")
    if type(finite) is not dict or finite.get("TotalPoly:RNA") != 6_547:
        raise SourceSemanticConflict("published Table S3 finite count differs")

    if report.get("qualification_status") != "BLOCKED_NOT_QUALIFIED":
        raise SourceSemanticConflict("published qualification status differs")
    if any(
        report.get(key) != 0
        for key in (
            "ordinary_study_contribution",
            "a1_intervention_study_contribution",
            "true_a2_dense_study_contribution",
            "canonical_record_count",
        )
    ):
        raise SourceSemanticConflict("published aggregate contribution is nonzero")
    if any(
        report.get(key) is not False
        for key in (
            "qualified",
            "training_allowed",
            "model_selection_allowed",
            "next_phase_authorized",
        )
    ):
        raise SourceSemanticConflict("published aggregate decision boundary elevated")
    if report.get("raw_replay_status") != "NOT_RUN_NOT_IN_SCOPE":
        raise SourceSemanticConflict("published raw-replay status differs")
    if integrity.get("asset_count") != 7 or integrity.get("aggregate_only") is not True:
        raise SourceSemanticConflict("published input-integrity aggregate differs")
    if marker.get("terminal_marker_written_last") is not True or marker.get("committed") is not True:
        raise SourceSemanticConflict("published endpoint terminal marker is not closed")


def inspect_predecessor_authority(
    config: Mapping[str, Any], *, repo_root: Path | None = None
) -> PredecessorSummary:
    """Read/hash the exact predecessor trio, source7, and aggregate bundle."""

    validate_static_config(config)
    repo = repo_root or Path(config["repository_authority"]["production_repo_root"])
    trio = config["predecessor_authority"]["published_endpoint_trio"]
    trio_specs = [
        {
            "name": trio["config_path"],
            "bytes": trio["config_bytes"],
            "sha256": trio["config_sha256"],
            "collect": True,
        },
        {
            "name": trio["implementation_script_path"],
            "bytes": trio["implementation_script_bytes"],
            "sha256": trio["implementation_script_sha256"],
            "collect": False,
        },
        {
            "name": trio["implementation_test_path"],
            "bytes": trio["implementation_test_bytes"],
            "sha256": trio["implementation_test_sha256"],
            "collect": False,
        },
    ]
    published_config_payload: bytes | None = None
    for item in trio_specs:
        observed = _read_exact_relative(
            repo,
            item["name"],
            expected_bytes=item["bytes"],
            expected_sha256=item["sha256"],
            collect=item["collect"],
            label=f"published endpoint trio {item['name']}",
        )
        if item["collect"]:
            published_config_payload = observed
    assert published_config_payload is not None
    _validate_published_endpoint_config(
        strict_json(published_config_payload, label="published endpoint config")
    )

    source = config["predecessor_authority"]["source_exact7"]
    source_root = Path(source["absolute_root"])
    source_fd = _open_absolute_directory(source_root, label="source exact7 root")
    s3_payload: bytes | None = None
    try:
        if sorted(os.listdir(source_fd)) != sorted(source["exact_member_names"]):
            raise InputIntegrityError("source exact7 member set differs")
        for item in source["members"]:
            collect = item["name"] == source["selectively_parsed_member"]
            observed = _read_exact_member_at(
                source_fd,
                item["name"],
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
                collect=collect,
                label=f"source exact7 {item['name']}",
            )
            if collect:
                s3_payload = observed
    finally:
        os.close(source_fd)
    assert s3_payload is not None
    s3 = audit_table_s3_selective(s3_payload)

    bundle = config["predecessor_authority"]["published_endpoint_bundle"]
    bundle_fd = _open_absolute_directory(
        Path(bundle["absolute_root"]), label="published endpoint bundle root"
    )
    bundle_payloads: dict[str, bytes] = {}
    try:
        if sorted(os.listdir(bundle_fd)) != sorted(bundle["exact_member_names"]):
            raise InputIntegrityError("published endpoint bundle member set differs")
        for item in bundle["members"]:
            observed = _read_exact_member_at(
                bundle_fd,
                item["name"],
                expected_bytes=item["bytes"],
                expected_sha256=item["sha256"],
                collect=True,
                label=f"published endpoint bundle {item['name']}",
            )
            assert observed is not None
            bundle_payloads[item["name"]] = observed
    finally:
        os.close(bundle_fd)

    json_names = bundle["aggregate_json_members_allowed_to_decode"]
    documents = {
        name: strict_json(bundle_payloads[name], label=f"published bundle {name}")
        for name in json_names
    }
    validate_published_endpoint_aggregate_documents(documents)
    expected_sums = "".join(
        f"{sha256(bundle_payloads[name])}  {name}\n"
        for name in sorted(
            {
                "INPUT_INTEGRITY_AUDIT.json",
                "PUBLISHED_ENDPOINT_AUDIT.json",
                "QUALIFICATION_REPORT.json",
            }
        )
    ).encode("ascii")
    if bundle_payloads[CHECKSUMS_NAME] != expected_sums:
        raise InputIntegrityError("published endpoint SHA256SUMS differs")

    return PredecessorSummary(
        published_endpoint_config_sha256=trio["config_sha256"],
        published_endpoint_trio_manifest_sha256=_manifest_sha256(trio_specs),
        source_exact7_manifest_sha256=_manifest_sha256(source["members"]),
        published_endpoint_bundle_manifest_sha256=_manifest_sha256(
            bundle["members"]
        ),
        source_exact7_member_count=len(source["members"]),
        published_endpoint_bundle_member_count=len(bundle["members"]),
        s3=s3,
    )


def audit_jats(payload: bytes, spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise SourceSemanticConflict("Europe PMC JATS XML is invalid") from exc

    identifiers: dict[str, list[str]] = defaultdict(list)
    for element in root.iter():
        if _local_name(element.tag) == "article-id":
            identifiers[element.attrib.get("pub-id-type", "")].append(
                _normalized_text(element)
            )
    for key, expected in spec["identity"].items():
        if identifiers.get(key) != [expected]:
            raise SourceSemanticConflict(f"JATS {key} identity differs")

    license_refs = [
        _normalized_text(element)
        for element in root.iter()
        if _local_name(element.tag) == "license_ref"
    ]
    license_texts = [
        _normalized_text(element)
        for element in root.iter()
        if _local_name(element.tag) == "license-p"
    ]
    if license_refs != [spec["license"]["license_ref"]]:
        raise SourceSemanticConflict("JATS CC BY license_ref differs")
    if license_texts != [spec["license"]["license_text"]]:
        raise SourceSemanticConflict("JATS CC BY license text differs")

    supplements: dict[str, tuple[str, str]] = {}
    for element in root.iter():
        if _local_name(element.tag) != "supplementary-material":
            continue
        supplement_id = element.attrib.get("id", "")
        labels = [
            _normalized_text(child)
            for child in element
            if _local_name(child.tag) == "label"
        ]
        media = [
            child
            for child in element.iter()
            if _local_name(child.tag) == "media"
        ]
        if len(labels) == 1 and len(media) == 1:
            hrefs = {
                _local_name(key): value for key, value in media[0].attrib.items()
            }
            supplements[supplement_id] = (labels[0], hrefs.get("href", ""))
    xrefs: dict[str, list[str]] = defaultdict(list)
    for element in root.iter():
        if (
            _local_name(element.tag) == "xref"
            and element.attrib.get("ref-type") == "supplementary-material"
        ):
            xrefs[_normalized_text(element)].append(element.attrib.get("rid", ""))
    linkage_counts: dict[str, int] = {}
    for linkage in spec["supplement_linkages"]:
        supplement_id = linkage["supplement_id"]
        if supplements.get(supplement_id) != (
            linkage["supplement_label"],
            linkage["media_href"],
        ):
            raise SourceSemanticConflict(
                f"JATS supplement linkage differs: {supplement_id}"
            )
        observed_rids = xrefs.get(linkage["table_label"], [])
        if not observed_rids or set(observed_rids) != {supplement_id}:
            raise SourceSemanticConflict(
                f"JATS table cross-reference differs: {linkage['table_label']}"
            )
        linkage_counts[linkage["table_label"]] = len(observed_rids)

    paragraph_results: dict[str, dict[str, Any]] = {}
    paragraphs = [
        _normalized_text(element)
        for element in root.iter()
        if _local_name(element.tag) == "p"
    ]
    for name, paragraph_spec in spec["normalized_paragraphs"].items():
        matches = [
            paragraph
            for paragraph in paragraphs
            if paragraph_spec["required_anchor"] in paragraph
        ]
        if len(matches) != 1:
            raise SourceSemanticConflict(f"JATS paragraph anchor is not unique: {name}")
        encoded = matches[0].encode("utf-8")
        if (
            len(encoded) != paragraph_spec["utf8_bytes"]
            or sha256(encoded) != paragraph_spec["sha256"]
        ):
            raise SourceSemanticConflict(f"JATS paragraph digest differs: {name}")
        paragraph_results[name] = {
            "utf8_bytes": len(encoded),
            "sha256": sha256(encoded),
        }
    return {
        "status": "PASS_EXACT_JATS_IDENTITY_LICENSE_LINKAGE_AND_PARAGRAPHS",
        "identity": dict(spec["identity"]),
        "license_ref": spec["license"]["license_ref"],
        "license_text_verified": True,
        "supplement_table_cross_reference_counts": dict(sorted(linkage_counts.items())),
        "normalized_paragraphs": paragraph_results,
    }


def _decompress_one_gzip_member(
    payload: bytes,
    *,
    expected_bytes: int,
    expected_sha256: str,
    maximum_bytes: int,
    label: str,
) -> bytes:
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decoded = inflater.decompress(payload, maximum_bytes + 1)
        decoded += inflater.flush(maximum_bytes + 1 - len(decoded))
    except zlib.error as exc:
        raise SourceSemanticConflict(f"{label} gzip stream is invalid") from exc
    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or len(decoded) > maximum_bytes
    ):
        raise SourceSemanticConflict(f"{label} is not one bounded gzip member")
    if len(decoded) != expected_bytes or sha256(decoded) != expected_sha256:
        raise SourceSemanticConflict(f"{label} plain byte/hash authority differs")
    return decoded


def _parse_soft_records(text: str) -> list[tuple[str, str, dict[str, list[str]]]]:
    records: list[tuple[str, str, dict[str, list[str]]]] = []
    current_type: str | None = None
    current_name: str | None = None
    fields: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        record_match = SOFT_RECORD.fullmatch(line)
        if record_match is not None:
            if current_type is not None and current_name is not None:
                records.append((current_type, current_name, dict(fields)))
            current_type, current_name = record_match.groups()
            fields = defaultdict(list)
            continue
        if line.startswith("!"):
            if current_type is None or " = " not in line:
                raise SourceSemanticConflict("GEO SOFT field is outside a record")
            key, value = line[1:].split(" = ", 1)
            fields[key].append(value)
    if current_type is not None and current_name is not None:
        records.append((current_type, current_name, dict(fields)))
    return records


def audit_soft(payload: bytes, spec: Mapping[str, Any]) -> dict[str, Any]:
    plain = _decompress_one_gzip_member(
        payload,
        expected_bytes=spec["plain_bytes"],
        expected_sha256=spec["plain_sha256"],
        maximum_bytes=MAX_SOFT_PLAIN_BYTES,
        label="GSE200302 family SOFT",
    )
    try:
        text = plain.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceSemanticConflict("GEO SOFT is not strict UTF-8") from exc
    if "\x00" in text:
        raise SourceSemanticConflict("GEO SOFT contains NUL")
    records = _parse_soft_records(text)
    series_records = [record for record in records if record[0] == "SERIES"]
    sample_records = [record for record in records if record[0] == "SAMPLE"]
    if len(series_records) != 1 or series_records[0][1] != spec["series_accession"]:
        raise SourceSemanticConflict("GEO SOFT series identity differs")
    series_fields = series_records[0][2]
    if series_fields.get("Series_geo_accession") != [spec["series_accession"]]:
        raise SourceSemanticConflict("GEO SOFT series accession field differs")
    if series_fields.get("Series_relation", []).count(
        spec["required_subseries_relation"]
    ) != 1:
        raise SourceSemanticConflict("GEO SOFT SubSeries relation differs")
    if len(sample_records) != spec["sample_count"]:
        raise SourceSemanticConflict("GEO SOFT sample count differs")
    series_sample_ids = series_fields.get("Series_sample_id", [])
    if len(series_sample_ids) != spec["sample_count"]:
        raise SourceSemanticConflict("GEO SOFT series sample list count differs")
    if set(series_sample_ids) != {record[1] for record in sample_records}:
        raise SourceSemanticConflict("GEO SOFT series/sample record set differs")

    title_pattern = re.compile(spec["sample_title_regex"])
    role_counts: Counter[str] = Counter()
    replicates: dict[str, set[int]] = defaultdict(set)
    sample_none_count = 0
    for _record_type, record_name, fields in sample_records:
        if fields.get("Sample_geo_accession") != [record_name]:
            raise SourceSemanticConflict("GEO SOFT sample accession field differs")
        titles = fields.get("Sample_title", [])
        if len(titles) != 1:
            raise SourceSemanticConflict("GEO SOFT sample title is not singular")
        match = title_pattern.fullmatch(titles[0])
        if match is None:
            raise SourceSemanticConflict("GEO SOFT sample title role differs")
        role, replicate = match.group(1), int(match.group(2))
        role_counts[role] += 1
        if replicate in replicates[role]:
            raise SourceSemanticConflict("GEO SOFT role replicate is duplicated")
        replicates[role].add(replicate)
        if fields.get(spec["sample_supplementary_file_key"]) != [
            spec["sample_supplementary_file_value"]
        ]:
            raise SourceSemanticConflict("GEO SOFT sample supplementary file is not NONE")
        sample_none_count += 1
    if dict(role_counts) != spec["role_counts"]:
        raise SourceSemanticConflict("GEO SOFT role counts differ")
    expected_replicates = set(spec["replicate_numbers_per_role"])
    if any(values != expected_replicates for values in replicates.values()):
        raise SourceSemanticConflict("GEO SOFT role replicate grid differs")

    series_supplements = series_fields.get("Series_supplementary_file", [])
    if len(series_supplements) != spec["series_supplementary_file_count"]:
        raise SourceSemanticConflict("GEO SOFT series supplementary count differs")
    matrix_references = [
        value
        for value in series_supplements
        if value.endswith("GSE200302_log2_cpm_counts_all_samples.txt.gz")
    ]
    if len(matrix_references) != spec["series_processed_matrix_reference_count"]:
        raise SourceSemanticConflict("GEO SOFT processed matrix reference differs")
    restriction_fields = [
        key
        for key in series_fields
        if "restriction" in key.casefold() or "controlled" in key.casefold()
    ]
    if restriction_fields:
        raise SourceSemanticConflict(
            "GEO SOFT contains a dataset restriction/control field outside private-only readiness"
        )
    return {
        "status": "PASS_EXACT_GSE200302_SUBSERIES_AND_24_SAMPLE_ROLE_GRID",
        "series_accession": spec["series_accession"],
        "subseries_of_gse200304": True,
        "sample_count": len(sample_records),
        "role_counts": dict(sorted(role_counts.items())),
        "replicates_per_role": sorted(expected_replicates),
        "sample_supplementary_none_count": sample_none_count,
        "series_supplementary_file_count": len(series_supplements),
        "series_processed_matrix_reference_count": len(matrix_references),
        "processed_matrix_payload_embedded_in_soft": False,
        "geo_dataset_restriction_field_count": len(restriction_fields),
    }


def audit_matrix(
    payload: bytes,
    spec: Mapping[str, Any],
    *,
    s3: S3SelectiveState,
) -> tuple[dict[str, Any], MatrixState]:
    plain = _decompress_one_gzip_member(
        payload,
        expected_bytes=spec["plain_bytes"],
        expected_sha256=spec["plain_sha256"],
        maximum_bytes=MAX_MATRIX_PLAIN_BYTES,
        label="GSE200302 processed matrix",
    )
    try:
        text = plain.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceSemanticConflict("processed matrix is not strict UTF-8") from exc
    if "\x00" in text:
        raise SourceSemanticConflict("processed matrix contains NUL")
    lines = text.splitlines()
    if len(lines) != spec["row_count"] + 1:
        raise SourceSemanticConflict("processed matrix row count differs")
    header = lines[0].split("\t")
    if len(header) != spec["header_field_count"] or header[0] != spec["first_field"]:
        raise SourceSemanticConflict("processed matrix header width/first field differs")

    observed_geometry: dict[tuple[str, int, str], str] = {}
    for field in header[1:]:
        match = MATRIX_HEADER.fullmatch(field)
        if match is None:
            raise SourceSemanticConflict("processed matrix header role grammar differs")
        family, replicate_text, technical_suffix, arm = match.groups()
        key = (family, int(replicate_text), arm)
        if key in observed_geometry:
            raise SourceSemanticConflict("processed matrix header role is duplicated")
        observed_geometry[key] = technical_suffix
    expected_geometry = {
        (family, replicate, arm)
        for family in spec["header_role_families"]
        for replicate in spec["header_replicates"]
        for arm in spec["header_arms"]
    }
    if set(observed_geometry) != expected_geometry:
        raise SourceSemanticConflict("processed matrix closed 60-column role grid differs")
    for family in spec["header_role_families"]:
        for replicate in spec["header_replicates"]:
            if (
                observed_geometry[(family, replicate, "WT")]
                != observed_geometry[(family, replicate, "Mutant")]
            ):
                raise SourceSemanticConflict(
                    "processed matrix WT/Mutant technical suffix differs"
                )

    keys: set[str] = set()
    width_errors = 0
    duplicate_keys = 0
    missing_values = 0
    invalid_numeric = 0
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != spec["header_field_count"]:
            width_errors += 1
            continue
        key = fields[0]
        if not key:
            missing_values += 1
        elif key in keys:
            duplicate_keys += 1
        else:
            keys.add(key)
        for value in fields[1:]:
            if value == "":
                missing_values += 1
                continue
            try:
                numeric = float(value)
            except ValueError:
                invalid_numeric += 1
                continue
            if not math.isfinite(numeric):
                invalid_numeric += 1
    observed_errors = {
        "row_width_error_count": width_errors,
        "duplicate_key_count": duplicate_keys,
        "missing_value_count": missing_values,
        "invalid_numeric_count": invalid_numeric,
    }
    expected_errors = {
        key: spec[key]
        for key in (
            "row_width_error_count",
            "duplicate_key_count",
            "missing_value_count",
            "invalid_numeric_count",
        )
    }
    if observed_errors != expected_errors:
        raise SourceSemanticConflict("processed matrix completeness errors are nonzero")
    if len(keys) != spec["row_count"]:
        raise SourceSemanticConflict("processed matrix unique key count differs")
    matrix_keys = frozenset(keys)
    if len(s3.keys) != spec["s3_key_count"]:
        raise SourceSemanticConflict("Table S3 key count differs")
    if matrix_keys != s3.keys:
        raise SourceSemanticConflict("processed matrix key set differs from Table S3")
    if not s3.finite_totalpoly_keys <= matrix_keys:
        raise SourceSemanticConflict(
            "processed matrix does not cover every finite TotalPoly key"
        )
    if len(s3.finite_totalpoly_keys) != spec["s3_finite_totalpoly_key_count"]:
        raise SourceSemanticConflict("Table S3 finite coverage count differs")

    state = MatrixState(
        keys=matrix_keys,
        row_count=len(lines) - 1,
        header_field_count=len(header),
    )
    return (
        {
            "status": "PASS_EXACT_6772_BY_61_MATRIX_AND_S3_MEMBERSHIP_CROSSCHECK",
            "header_field_count": len(header),
            "value_field_count": len(header) - 1,
            "row_count": len(lines) - 1,
            **observed_errors,
            "closed_role_geometry_count": len(observed_geometry),
            "required_endpoint_families": list(
                spec["replicate_branch_required_families"]
            ),
            "required_arms": list(spec["header_arms"]),
            "required_replicates": list(spec["header_replicates"]),
            "endpoint_excluded_families": list(spec["endpoint_excluded_families"]),
            "matrix_key_set_equals_s3_key_set": True,
            "matrix_key_count": len(matrix_keys),
            "s3_key_count": len(s3.keys),
            "finite_totalpoly_key_count": len(s3.finite_totalpoly_keys),
            "matrix_covers_every_finite_totalpoly_key": True,
            "standard_error_status": "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED",
            "p_or_fdr_back_calculation_used": False,
        },
        state,
    )


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


def _default_https_open(url: str) -> Any:
    context = ssl.create_default_context()
    opener = build_opener(_RejectRedirects(), HTTPSHandler(context=context))
    request = Request(
        url,
        headers={
            "Accept": "application/xml,text/plain,application/gzip,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": "GSE200304-upstream-authority-audit/1.0",
        },
        method="GET",
    )
    try:
        return opener.open(request, timeout=120)
    except HTTPError as exc:
        raise InputIntegrityError(f"official source HTTP status differs: {url}") from exc


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PublicationError("exclusive file write made no progress")
        view = view[written:]


def _close_fd_after_validation(descriptor: int) -> None:
    """Best-effort descriptor cleanup that cannot change publication truth."""

    try:
        os.close(descriptor)
    except OSError:
        pass


def _download_source_member(
    directory_fd: int,
    spec: Mapping[str, Any],
    *,
    open_url: Callable[[str], Any],
) -> bytes:
    name = spec["output_name"]
    if SAFE_BASENAME.fullmatch(name) is None:
        raise PublicationError("download output basename is unsafe")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicationError(f"download member cannot be created: {name}") from exc
    response: Any | None = None
    try:
        response = open_url(spec["url"])
        status = getattr(response, "status", 200)
        final_url = response.geturl() if hasattr(response, "geturl") else spec["url"]
        if status != 200 or final_url != spec["url"]:
            raise InputIntegrityError("official source status/final URL differs")
        headers = getattr(response, "headers", {})
        content_length = headers.get("Content-Length") if hasattr(headers, "get") else None
        if content_length is not None:
            try:
                parsed_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise InputIntegrityError("official source Content-Length is invalid") from exc
            if parsed_length != spec["bytes"]:
                raise InputIntegrityError("official source Content-Length differs")
        content_encoding = headers.get("Content-Encoding") if hasattr(headers, "get") else None
        if content_encoding not in {None, "", "identity"}:
            raise InputIntegrityError("official source Content-Encoding is not identity")

        digest = hashlib.sha256()
        count = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            if not isinstance(block, (bytes, bytearray)):
                raise InputIntegrityError("official source returned a non-byte block")
            count += len(block)
            if count > spec["bytes"]:
                raise InputIntegrityError("official source exceeds frozen size")
            digest.update(block)
            _write_all(descriptor, bytes(block))
        if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise InputIntegrityError("official source byte/hash authority differs")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise PublicationError("download member type/link differs")
        if opened.st_size != spec["bytes"]:
            raise PublicationError("download member same-FD size differs")
        os.fsync(descriptor)

        # Rewind and materialize the exact bytes from the same descriptor that
        # was written and hashed.  Semantic parsing consumes this snapshot.
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        reread_digest = hashlib.sha256()
        reread_count = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            reread_count += len(block)
            reread_digest.update(block)
            chunks.append(block)
        final = os.fstat(descriptor)
        if _file_identity(final) != _file_identity(opened):
            raise PublicationError("download member changed during same-FD reread")
        if (
            reread_count != spec["bytes"]
            or reread_digest.hexdigest() != spec["sha256"]
        ):
            raise PublicationError("download member same-FD reread differs")
        try:
            path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PublicationError("download member path binding disappeared") from exc
        if _file_identity(path_stat) != _file_identity(final):
            raise PublicationError("download member path binding changed")
        return b"".join(chunks)
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        os.close(descriptor)


def _assert_derived_aggregate_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in DERIVED_FORBIDDEN_KEYS:
                raise PublicationError(f"derived audit contains prohibited key: {key}")
            _assert_derived_aggregate_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_derived_aggregate_safe(child)


def build_closed_audit(
    config: Mapping[str, Any],
    binding: Mapping[str, str],
    predecessor: PredecessorSummary,
    *,
    jats: Mapping[str, Any],
    soft: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    viability = config["viability_contract"]
    sources = config["public_sources"]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_UPSTREAM_SOURCE_AUTHORITY_VIABILITY_AUDIT_V1",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": PHASE_ID,
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE",
        "mode": MODE,
        "producer_binding": dict(binding),
        "predecessor_authority": predecessor.aggregate_dict(),
        "official_source_authority": {
            "status": "PASS_EXACT_THREE_OFFICIAL_SOURCE_SNAPSHOTS",
            "network_download_count": 3,
            "verbatim_source_member_count": 3,
            "sources": [
                {
                    "source_kind": sources[key]["source_kind"],
                    "url": sources[key]["url"],
                    "output_name": sources[key]["output_name"],
                    "bytes": sources[key]["bytes"],
                    "sha256": sources[key]["sha256"],
                    "same_fd_size_and_hash_verified": True,
                }
                for key in (JATS_CONFIG_KEY, SOFT_CONFIG_KEY, MATRIX_CONFIG_KEY)
            ],
        },
        "jats_authority": dict(jats),
        "geo_soft_authority": dict(soft),
        "processed_matrix_authority": dict(matrix),
        "endpoint_crosswalk": copy.deepcopy(
            viability["canonical_reported_endpoint_semantics"]
        ),
        "replicate_branch": copy.deepcopy(viability["row_replicate_or_valid_se"]),
        "private_only_rights": copy.deepcopy(viability["license_rights"]),
        "biological_group_authority": copy.deepcopy(
            viability["biological_group_authority"]
        ),
        "unchanged_gates": copy.deepcopy(viability["unchanged_gates"]),
        "decision_boundary": copy.deepcopy(config["decision_boundary"]),
        "execution_boundary": {
            "published_endpoint_aggregate_authority_read": True,
            "source_exact7_same_fd_hash_verified": True,
            "table_s2_row_payload_opened": False,
            "table_s3_selective_key_and_finite_state_replayed": True,
            "table_s3_gene_selected_or_persisted": False,
            "table_s3_translation_significance_selected_or_persisted": False,
            "consumer_run": False,
            "adjudicator_run": False,
            "qualifier_run": False,
            "canonicalizer_run": False,
            "raw_replay_run": False,
            "training_run": False,
            "model_selection_run": False,
            "row_mapping_producer_run": False,
            "gate_change_count": 0,
        },
        "privacy": {
            "derived_row_payload": False,
            "derived_sequence_payload": False,
            "derived_row_identifier_payload": False,
            "derived_effect_value_payload": False,
            "derived_gene_payload": False,
            "verbatim_raw_source_members_are_not_derived_payload": True,
        },
    }
    validate_closed_audit(audit, config)
    return audit


def validate_closed_audit(audit: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if audit.get("status") != (
        "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE"
    ):
        raise PublicationError("closed audit status differs")
    endpoint = audit.get("endpoint_crosswalk")
    replicate = audit.get("replicate_branch")
    rights = audit.get("private_only_rights")
    group = audit.get("biological_group_authority")
    if any(type(value) is not dict for value in (endpoint, replicate, rights, group)):
        raise PublicationError("closed audit viability components are absent")
    if endpoint.get("status_if_all_source_checks_pass") != (
        "READY_FOR_PASS_RECORD_NOT_YET_BOUND"
    ) or endpoint.get("consumer_gate_pass") is not False:
        raise PublicationError("closed audit endpoint readiness differs")
    if replicate.get("status_if_all_source_checks_pass") != (
        "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND"
    ) or replicate.get("consumer_gate_pass") is not False:
        raise PublicationError("closed audit replicate readiness differs")
    if rights.get("status_if_all_source_checks_pass") != (
        "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND"
    ) or rights.get("consumer_gate_pass") is not False:
        raise PublicationError("closed audit private-only readiness differs")
    if group.get("status") != "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT":
        raise PublicationError("closed audit group blocker differs")
    if audit.get("processed_matrix_authority", {}).get(
        "matrix_key_set_equals_s3_key_set"
    ) is not True:
        raise PublicationError("closed audit matrix/S3 equality is not PASS")
    if audit.get("processed_matrix_authority", {}).get(
        "matrix_covers_every_finite_totalpoly_key"
    ) is not True:
        raise PublicationError("closed audit finite coverage is not PASS")
    decision = audit.get("decision_boundary")
    if decision != config["decision_boundary"]:
        raise PublicationError("closed audit decision boundary differs")
    execution = audit.get("execution_boundary")
    if type(execution) is not dict or execution.get("gate_change_count") != 0:
        raise PublicationError("closed audit gate-change boundary differs")
    if any(
        execution.get(key) is not False
        for key in (
            "consumer_run",
            "adjudicator_run",
            "qualifier_run",
            "canonicalizer_run",
            "raw_replay_run",
            "training_run",
            "model_selection_run",
            "row_mapping_producer_run",
        )
    ):
        raise PublicationError("closed audit prohibited execution is true")
    _assert_derived_aggregate_safe(audit)


def _write_exclusive_member(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    on_created: Callable[[], None] | None = None,
) -> None:
    if SAFE_BASENAME.fullmatch(name) is None or "/" in name:
        raise PublicationError("publication member basename is unsafe")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicationError(f"publication member cannot be created: {name}") from exc
    try:
        if on_created is not None:
            on_created()
        _write_all(descriptor, payload)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(payload)
        ):
            raise PublicationError("publication member type/link/size differs")
        os.fsync(descriptor)
        try:
            path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PublicationError("publication member path binding disappeared") from exc
        if _file_identity(path_stat) != _file_identity(opened):
            raise PublicationError("publication member path binding changed")
    finally:
        os.close(descriptor)


def _bundle_payloads(
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    raw_sources: Mapping[str, bytes],
    *,
    output_directory: Path,
) -> dict[str, bytes]:
    if set(raw_sources) != {
        config["public_sources"][key]["output_name"]
        for key in (JATS_CONFIG_KEY, SOFT_CONFIG_KEY, MATRIX_CONFIG_KEY)
    }:
        raise PublicationError("raw source member set differs")
    audit_payload = json_bytes(audit)
    nonchecksum = {**dict(raw_sources), AUDIT_NAME: audit_payload}
    checksum_payload = "".join(
        f"{sha256(nonchecksum[name])}  {name}\n" for name in sorted(nonchecksum)
    ).encode("ascii")
    preterminal = {**nonchecksum, CHECKSUMS_NAME: checksum_payload}
    marker = {
        "schema_version": "1.0.0",
        "record_type": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_PUBLICATION_COMMIT_V1",
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "bundle_id": config["output_contract"]["bundle_id"],
        "preterminal_member_names": sorted(preterminal),
        "preterminal_member_count": len(preterminal),
        "exact_final_member_count": 6,
        "sha256sums_sha256": sha256(checksum_payload),
        "final_output_target_sha256": sha256(
            os.fspath(output_directory).encode("utf-8")
        ),
        "publication_mode": config["output_contract"]["publication_mode"],
        "committed": True,
        "terminal_marker_written_last": True,
        "no_overwrite": True,
        "partial_default": config["output_contract"]["partial_default"],
    }
    return {**preterminal, MARKER_NAME: json_bytes(marker)}


def _open_output_directory_at(parent_fd: int, name: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise PublicationError("output target is not a directory")
    try:
        path_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        os.close(descriptor)
        raise PublicationError("output target path binding disappeared") from exc
    if (
        not stat.S_ISDIR(path_stat.st_mode)
        or _directory_dev_ino(path_stat) != _directory_dev_ino(opened)
    ):
        os.close(descriptor)
        raise PublicationError("output target path binding changed")
    return descriptor


def _validate_exact_payloads_at(
    output_fd: int,
    expected_payloads: Mapping[str, bytes],
    expected_names: Sequence[str],
    *,
    label: str,
) -> None:
    expected_set = set(expected_names)
    if set(expected_payloads) != expected_set:
        raise PublicationError(f"{label} expected payload/member sets differ")
    directory_identity = _directory_dev_ino(os.fstat(output_fd))
    before = set(os.listdir(output_fd))
    if before != expected_set:
        raise PublicationError(f"{label} member set differs before validation")
    for name in sorted(expected_payloads):
        payload = expected_payloads[name]
        observed = _read_exact_member_at(
            output_fd,
            name,
            expected_bytes=len(payload),
            expected_sha256=sha256(payload),
            collect=False,
            label=f"{label} {name}",
        )
        if observed is not None:
            raise PublicationError(f"{label} unexpectedly collected member bytes")
    after = set(os.listdir(output_fd))
    if after != expected_set or after != before:
        raise PublicationError(f"{label} member set changed during validation")
    if _directory_dev_ino(os.fstat(output_fd)) != directory_identity:
        raise PublicationError(f"{label} directory identity changed")


def _open_fresh_canonical_output(
    target: Path,
    *,
    expected_parent_identity: tuple[int, int],
    expected_output_identity: tuple[int, int],
) -> tuple[int, int]:
    parent_fd = _open_absolute_directory(
        target.parent, label="fresh canonical output parent"
    )
    try:
        if _directory_dev_ino(os.fstat(parent_fd)) != expected_parent_identity:
            raise PublicationError("canonical output parent identity changed")
        output_fd = _open_output_directory_at(parent_fd, target.name)
        if _directory_dev_ino(os.fstat(output_fd)) != expected_output_identity:
            os.close(output_fd)
            raise PublicationError("canonical output directory identity changed")
        return parent_fd, output_fd
    except Exception:
        os.close(parent_fd)
        raise


def _fresh_canonical_exact_validation(
    target: Path,
    *,
    expected_parent_identity: tuple[int, int],
    expected_output_identity: tuple[int, int],
    expected_payloads: Mapping[str, bytes],
    expected_names: Sequence[str],
    label: str,
) -> None:
    parent_fd, output_fd = _open_fresh_canonical_output(
        target,
        expected_parent_identity=expected_parent_identity,
        expected_output_identity=expected_output_identity,
    )
    try:
        os.fsync(output_fd)
        os.fsync(parent_fd)
        _validate_exact_payloads_at(
            output_fd,
            expected_payloads,
            expected_names,
            label=label,
        )
        if _directory_dev_ino(os.fstat(parent_fd)) != expected_parent_identity:
            raise PublicationError("canonical output parent changed during validation")
        if _directory_dev_ino(os.fstat(output_fd)) != expected_output_identity:
            raise PublicationError("canonical output changed during validation")
    finally:
        _close_fd_after_validation(output_fd)
        _close_fd_after_validation(parent_fd)

    # Re-open once more from slash after the byte/hash pass so a path swap at
    # the final-validation boundary cannot be accepted through stale handles.
    confirm_parent_fd, confirm_output_fd = _open_fresh_canonical_output(
        target,
        expected_parent_identity=expected_parent_identity,
        expected_output_identity=expected_output_identity,
    )
    _close_fd_after_validation(confirm_output_fd)
    _close_fd_after_validation(confirm_parent_fd)


def _validate_existing_exact_bundle(
    config: Mapping[str, Any],
    binding: Mapping[str, str],
    predecessor: PredecessorSummary,
    output_directory: Path,
    parent_fd: int,
    parent_identity: tuple[int, int],
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    try:
        output_fd = _open_output_directory_at(parent_fd, output_directory.name)
    except (AuditError, OSError) as exc:
        raise ExistingOutputRequiresManualReview(
            "existing output namespace is not the canonical directory; "
            "preserve and review manually"
        ) from exc
    try:
        if _directory_dev_ino(os.fstat(parent_fd)) != parent_identity:
            raise ExistingOutputRequiresManualReview(
                "existing output parent identity changed"
            )
        output_identity = _directory_dev_ino(os.fstat(output_fd))
        if sorted(os.listdir(output_fd)) != sorted(
            config["output_contract"]["exact_member_names"]
        ):
            raise ExistingOutputRequiresManualReview(
                "existing output member set is not exact6; preserve and review manually"
            )
        raw_sources: dict[str, bytes] = {}
        for key in (JATS_CONFIG_KEY, SOFT_CONFIG_KEY, MATRIX_CONFIG_KEY):
            spec = config["public_sources"][key]
            payload = _read_exact_member_at(
                output_fd,
                spec["output_name"],
                expected_bytes=spec["bytes"],
                expected_sha256=spec["sha256"],
                collect=True,
                label=f"existing output {spec['output_name']}",
            )
            assert payload is not None
            raw_sources[spec["output_name"]] = payload
        jats = audit_jats(raw_sources[config["public_sources"][JATS_CONFIG_KEY]["output_name"]], config["public_sources"][JATS_CONFIG_KEY])
        soft = audit_soft(raw_sources[config["public_sources"][SOFT_CONFIG_KEY]["output_name"]], config["public_sources"][SOFT_CONFIG_KEY])
        matrix, _state = audit_matrix(
            raw_sources[config["public_sources"][MATRIX_CONFIG_KEY]["output_name"]],
            config["public_sources"][MATRIX_CONFIG_KEY],
            s3=predecessor.s3,
        )
        audit = build_closed_audit(
            config,
            binding,
            predecessor,
            jats=jats,
            soft=soft,
            matrix=matrix,
        )
        expected = _bundle_payloads(
            config, audit, raw_sources, output_directory=output_directory
        )
        _validate_exact_payloads_at(
            output_fd,
            expected,
            config["output_contract"]["exact_member_names"],
            label="existing exact6 initial validation",
        )
        strict_json(expected[AUDIT_NAME], label="existing closed audit")
        strict_json(expected[MARKER_NAME], label="existing terminal marker")
        try:
            if fault_hook is not None:
                fault_hook("before_existing_fresh_validation")
            _fresh_canonical_exact_validation(
                output_directory,
                expected_parent_identity=parent_identity,
                expected_output_identity=output_identity,
                expected_payloads=expected,
                expected_names=config["output_contract"]["exact_member_names"],
                label="existing exact6 canonical durability validation",
            )
            publication_state = (
                "IDEMPOTENT_EXACT_EXISTING_DURABILITY_RECONFIRMED"
            )
        except Exception as initial_fresh_error:
            try:
                _fresh_canonical_exact_validation(
                    output_directory,
                    expected_parent_identity=parent_identity,
                    expected_output_identity=output_identity,
                    expected_payloads=expected,
                    expected_names=config["output_contract"][
                        "exact_member_names"
                    ],
                    label="existing exact6 canonical durability recovery",
                )
                publication_state = (
                    "IDEMPOTENT_EXACT_EXISTING_AFTER_DURABILITY_RECOVERY"
                )
            except Exception as recovery_error:
                raise ExistingOutputRequiresManualReview(
                    "existing exact6 canonical identity/durability is indeterminate; "
                    "preserve and review manually"
                ) from recovery_error
        return {
            "publication_state": publication_state,
            "committed": True,
            "accepted": True,
            "member_count": 6,
            "terminal_marker": MARKER_NAME,
            "marker_created": True,
            "terminal_marker_written_last": True,
            "canonical_identity_reconfirmed": True,
            "durability_reconfirmed": True,
            "no_overwrite": True,
            "qualified": False,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        }
    except ExistingOutputRequiresManualReview:
        raise
    except AuditError as exc:
        raise ExistingOutputRequiresManualReview(
            "existing output is not exact idempotent; preserve and review manually"
        ) from exc
    finally:
        _close_fd_after_validation(output_fd)


def publish_bundle(
    config: Mapping[str, Any],
    binding: Mapping[str, str],
    predecessor: PredecessorSummary,
    output_directory: Path,
    *,
    open_url: Callable[[str], Any] = _default_https_open,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    validate_static_config(config)
    if config["implementation_binding"]["status"] != BOUND:
        raise BindingError(
            "IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED; stopped before source/output"
        )
    target = Path(os.path.abspath(os.fspath(output_directory)))
    if not target.is_absolute() or SAFE_BASENAME.fullmatch(target.name) is None:
        raise PublicationError("output target is not a safe absolute directory")
    forbidden_roots = {
        config["predecessor_authority"]["source_exact7"]["absolute_root"],
        config["predecessor_authority"]["published_endpoint_bundle"]["absolute_root"],
    }
    if any(
        os.path.commonpath([os.fspath(target), root]) == root for root in forbidden_roots
    ):
        raise PublicationError("output target overlaps a frozen input authority")

    parent_fd = _open_absolute_directory(target.parent, label="output parent")
    parent_identity = _directory_dev_ino(os.fstat(parent_fd))
    created = False
    marker_created = False
    output_fd: int | None = None
    output_identity: tuple[int, int] | None = None
    payloads: dict[str, bytes] | None = None
    try:
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        except FileExistsError:
            return _validate_existing_exact_bundle(
                config,
                binding,
                predecessor,
                target,
                parent_fd,
                parent_identity,
                fault_hook=fault_hook,
            )
        except OSError as exc:
            raise PublicationError("exclusive output directory cannot be created") from exc

        output_fd = _open_output_directory_at(parent_fd, target.name)
        output_identity = _directory_dev_ino(os.fstat(output_fd))
        raw_sources: dict[str, bytes] = {}
        for key in (JATS_CONFIG_KEY, SOFT_CONFIG_KEY, MATRIX_CONFIG_KEY):
            spec = config["public_sources"][key]
            raw_sources[spec["output_name"]] = _download_source_member(
                output_fd, spec, open_url=open_url
            )
        jats = audit_jats(
            raw_sources[config["public_sources"][JATS_CONFIG_KEY]["output_name"]],
            config["public_sources"][JATS_CONFIG_KEY],
        )
        soft = audit_soft(
            raw_sources[config["public_sources"][SOFT_CONFIG_KEY]["output_name"]],
            config["public_sources"][SOFT_CONFIG_KEY],
        )
        matrix, _state = audit_matrix(
            raw_sources[config["public_sources"][MATRIX_CONFIG_KEY]["output_name"]],
            config["public_sources"][MATRIX_CONFIG_KEY],
            s3=predecessor.s3,
        )
        audit = build_closed_audit(
            config,
            binding,
            predecessor,
            jats=jats,
            soft=soft,
            matrix=matrix,
        )
        payloads = _bundle_payloads(
            config, audit, raw_sources, output_directory=target
        )

        # Raw source members already exist.  Publish audit and checksums, make
        # them durable together with the directory and parent, then create the
        # terminal marker as the final namespace operation.
        _write_exclusive_member(output_fd, AUDIT_NAME, payloads[AUDIT_NAME])
        _write_exclusive_member(output_fd, CHECKSUMS_NAME, payloads[CHECKSUMS_NAME])
        if fault_hook is not None:
            fault_hook("after_preterminal_members_written")
        preterminal_names = [
            name
            for name in config["output_contract"]["exact_member_names"]
            if name != MARKER_NAME
        ]
        preterminal_payloads = {
            name: payloads[name] for name in preterminal_names
        }
        _validate_exact_payloads_at(
            output_fd,
            preterminal_payloads,
            preterminal_names,
            label="preterminal exact5 validation",
        )
        os.fsync(output_fd)
        os.fsync(parent_fd)

        def note_marker_created() -> None:
            nonlocal marker_created
            marker_created = True

        _write_exclusive_member(
            output_fd,
            MARKER_NAME,
            payloads[MARKER_NAME],
            on_created=note_marker_created,
        )
        if fault_hook is not None:
            fault_hook("after_marker_created_before_directory_fsync")
        os.fsync(output_fd)
        os.fsync(parent_fd)
        if fault_hook is not None:
            fault_hook("before_fresh_final_validation")
        assert output_identity is not None
        _fresh_canonical_exact_validation(
            target,
            expected_parent_identity=parent_identity,
            expected_output_identity=output_identity,
            expected_payloads=payloads,
            expected_names=config["output_contract"]["exact_member_names"],
            label="new exact6 canonical final validation",
        )
        return {
            "publication_state": "COMMITTED_ACCEPTED_AUDIT_ONLY",
            "committed": True,
            "accepted": True,
            "member_count": 6,
            "terminal_marker": MARKER_NAME,
            "marker_created": True,
            "terminal_marker_written_last": True,
            "canonical_identity_reconfirmed": True,
            "durability_reconfirmed": True,
            "no_overwrite": True,
            "qualified": False,
            "ordinary_study_contribution": 0,
            "a1_intervention_study_contribution": 0,
            "true_a2_dense_study_contribution": 0,
            "canonical_record_count": 0,
            "training_allowed": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        }
    except ExistingOutputRequiresManualReview:
        raise
    except Exception as exc:
        if marker_created:
            if output_identity is None or payloads is None:
                raise PostMarkerCommitOutcomeIndeterminate(
                    "POST_MARKER_COMMIT_OUTCOME_INDETERMINATE; preserve and review manually"
                ) from exc
            try:
                _fresh_canonical_exact_validation(
                    target,
                    expected_parent_identity=parent_identity,
                    expected_output_identity=output_identity,
                    expected_payloads=payloads,
                    expected_names=config["output_contract"]["exact_member_names"],
                    label="post-marker canonical exact6 recovery",
                )
            except Exception as recovery_error:
                raise PostMarkerCommitOutcomeIndeterminate(
                    "POST_MARKER_COMMIT_OUTCOME_INDETERMINATE; preserve and review manually"
                ) from recovery_error
            return {
                "publication_state": (
                    "COMMITTED_ACCEPTED_AFTER_POST_MARKER_RECOVERY"
                ),
                "committed": True,
                "accepted": True,
                "member_count": 6,
                "terminal_marker": MARKER_NAME,
                "marker_created": True,
                "terminal_marker_written_last": True,
                "canonical_identity_reconfirmed": True,
                "durability_reconfirmed": True,
                "recovered_from_post_marker_error": type(exc).__name__,
                "no_overwrite": True,
                "qualified": False,
                "ordinary_study_contribution": 0,
                "a1_intervention_study_contribution": 0,
                "true_a2_dense_study_contribution": 0,
                "canonical_record_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
            }
        if created:
            raise PartialPublicationError(
                f"partial output preserved at {target}; manual adjudication required"
            ) from exc
        if isinstance(exc, AuditError):
            raise
        raise PublicationError("publication failed before output creation") from exc
    finally:
        if output_fd is not None:
            _close_fd_after_validation(output_fd)
        _close_fd_after_validation(parent_fd)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PRODUCTION_CONFIG_PATH,
        help="Bound producer config (production default is fixed).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Absolute exact-six output directory; required outside inspect-only mode.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Read-only validation of Git and predecessor aggregate authorities.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config, config_payload = load_config(args.config)
        # Binding is checked before predecessor sources or the output namespace.
        binding = validate_production_authority(
            config,
            config_payload,
            config_path=args.config,
            script_path=Path(os.path.abspath(__file__)),
        )
        predecessor = inspect_predecessor_authority(config)
        if args.inspect_only:
            print(
                json.dumps(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "mode": "READ_ONLY_INSPECT",
                        "binding": binding,
                        "predecessor_authority": predecessor.aggregate_dict(),
                        "consumer_run": False,
                        "adjudicator_run": False,
                        "qualifier_run": False,
                        "training_run": False,
                        "gate_change_count": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output_directory is None:
            raise PublicationError("--output-directory is required")
        result = publish_bundle(
            config,
            binding,
            predecessor,
            args.output_directory,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except AuditError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
