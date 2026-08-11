#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed GSE114002 EVT-039 runtime sync.

The publisher registers two already-committed aggregate-only reconciliation
attempts without executing the qualifier or opening real row-level data.  It
replays the exact EVT-038 predecessor, preserves both attempt bundles, and
changes no scientific gate.  Production has no caller-selected repository,
runtime root, or attempt source.  The only caller-selected path is a prepared
artifact directory outside all run roots.  An UNKNOWN config stops before any
runtime or attempt path is opened.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


CONFIG_REPO_PATH = "configs/route_a_v3_gse114002_endpoint_geometry_reconciliation_v2_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/gse114002_endpoint_geometry_reconciliation_v2_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse114002_endpoint_geometry_reconciliation_v2_runtime_sync.py"
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
GIT_BINARY = "/usr/bin/git"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
PENDING_PREFIX = "PENDING_"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
BUNDLE_JSON_NAMES = (
    "INPUT_INTEGRITY_AUDIT.json",
    "ENDPOINT_RECONCILIATION_AUDIT.json",
    "POOL_GEOMETRY_RECONCILIATION_AUDIT.json",
    "QUALIFICATION_REPORT.json",
)
BUNDLE_NAMES = BUNDLE_JSON_NAMES + ("SHA256SUMS", "PUBLICATION_COMMIT.json")
PRIMARY_PUBLICATION_MODE = "KERNEL_ATOMIC_RENAME_NOREPLACE_V2"
FALLBACK_PUBLICATION_MODE = "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_MARKER_V2"
FAILED_MECHANICAL_STATUS = "MECHANICAL_RECONCILIATION_FAILED_NOT_QUALIFIED"
CURRENT_MECHANICAL_STATUS = "MECHANICAL_ENDPOINT_RECONCILED_NOT_QUALIFIED"
EXPECTED_BLOCKERS = [
    "CHECKPOINT_SPECIFIC_EXPOSURE_UNKNOWN_NOT_ASSERTED",
    "FIELD_AND_BIOLOGICAL_SOURCE_AUTHORITY_UNKNOWN_NOT_ASSERTED",
    "FULL_CONSTRUCT_PREFIX_REPORTER_RNA_CHEMISTRY_UNKNOWN_NOT_ASSERTED",
    "LICENSE_AND_REDISTRIBUTION_RIGHTS_UNKNOWN_NOT_ASSERTED",
    "NEAR_DUPLICATE_SPLIT_AND_LEAKAGE_AUDIT_NOT_RUN",
    "OWNER_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED",
    "PREFROZEN_GROUP_POWER_NOT_RUN",
]
EXPECTED_ATTEMPTS = [
    {
        "attempt_id": "ATTEMPT_1_FAILED_PRESERVED",
        "run_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_20260811T073353P0800_998d030",
        "status": FAILED_MECHANICAL_STATUS,
        "implementation_commit": "78b06434a9b94eaf5149dff7f6bb6b2d58e76ade",
        "binding_commit": "998d030a51737bfa1e27580efe8b89e22ae39149",
        "members": {
            "INPUT_INTEGRITY_AUDIT.json": {"bytes": 640, "sha256": "d01377c1e05bc85beafba4893a05255e7f590c9518a8ee069fca607131d1b80b"},
            "ENDPOINT_RECONCILIATION_AUDIT.json": {"bytes": 1907, "sha256": "bd5f89751fa69d61aa124e7424c04f78ce75ce3cfab16145c5f47ef7799b38b9"},
            "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": {"bytes": 2250, "sha256": "f4910ba655433076ddc44085770ef2d36df6169d2668030bcfbcb7372a46acc3"},
            "QUALIFICATION_REPORT.json": {"bytes": 2450, "sha256": "7f5269866be96deab3e41099181c53294f73e4e14c6a4138968a3e578b3c89d0"},
            "SHA256SUMS": {"bytes": 392, "sha256": "a7556e12c26c062698ffa0b03407ae521dc8238dfd5437820a5451674ee06d0e"},
            "PUBLICATION_COMMIT.json": {"bytes": 1174, "sha256": "07e17411b32c630b25e66f23212a65939c6d259a5fa554a260cc03505346bddf"},
        },
    },
    {
        "attempt_id": "ATTEMPT_2_CURRENT",
        "run_root": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_20260811T075711P0800_a148c73",
        "status": CURRENT_MECHANICAL_STATUS,
        "implementation_commit": "1543f09e74643a9a36b89742c7a1cc458b6b0d56",
        "binding_commit": "a148c737101ed8d0e24209233b86a55e2710633e",
        "members": {
            "INPUT_INTEGRITY_AUDIT.json": {"bytes": 640, "sha256": "d01377c1e05bc85beafba4893a05255e7f590c9518a8ee069fca607131d1b80b"},
            "ENDPOINT_RECONCILIATION_AUDIT.json": {"bytes": 1910, "sha256": "0d515639174ad0f1daa6dd9e46197984bd3fa43b769023a2d6d1cc2ea0d1e641"},
            "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": {"bytes": 2423, "sha256": "40a7cef042e4b4f8db4d6db4be469a64fefad94d55f43ff54fbca2a580b23d21"},
            "QUALIFICATION_REPORT.json": {"bytes": 2325, "sha256": "34c2cc0c861286f8e22bf1ba4026d5f754254ffb1bd10a4b989a9546e874a9c3"},
            "SHA256SUMS": {"bytes": 392, "sha256": "9fbd4970a3974786cd246715dfef029bd0ad87718b4eff48266d355b003ef9f0"},
            "PUBLICATION_COMMIT.json": {"bytes": 1172, "sha256": "522568111b68fc68508dfb2cc82b48121c3f890a7a479ef3510fada56e38663a"},
        },
    },
]


class RuntimeSyncError(RuntimeError):
    """Base class for a fail-closed runtime-sync error."""


class BindingError(RuntimeSyncError):
    """The config-only implementation binding is absent or inconsistent."""


class AuthorityError(RuntimeSyncError):
    """Repository authority is not the exact frozen I -> B chain."""


class PublicationError(RuntimeSyncError):
    """A runtime or prepared-artifact publication invariant failed."""


FaultInjector = Callable[[str], None]


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def compact_json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeSyncError(f"non-finite JSON constant in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeSyncError(f"JSON root is not an object: {label}")
    return value


def load_json_lines(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(
                line,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    RuntimeSyncError(f"non-finite JSON constant in {label}: {token}")
                ),
            )
            for line in payload.splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSyncError(f"invalid JSONL: {label}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeSyncError(f"JSONL contains a non-object line: {label}")
    return values


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def is_strict_descendant(path: Path, parent: Path) -> bool:
    path_text = os.fspath(lexical_absolute(path))
    parent_text = os.fspath(lexical_absolute(parent))
    try:
        common = os.path.commonpath((path_text, parent_text))
    except ValueError:
        return False
    return common == parent_text and path_text != parent_text


def open_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationError(f"cannot open non-symlink directory: {path}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise PublicationError(f"path is not a directory: {path}")
    return descriptor


@contextmanager
def locked_directory(
    path: Path,
    *,
    timeout_seconds: float = 10.0,
    cleanup_warnings: list[dict[str, str]] | None = None,
    fault_injector: FaultInjector | None = None,
) -> Iterator[int]:
    """Hold an exclusive directory lock; exit faults are reported, never masking body truth."""

    descriptor = open_directory(path)
    deadline = time.monotonic() + timeout_seconds
    warnings = cleanup_warnings if cleanup_warnings is not None else []
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise PublicationError(f"timed out locking runtime directory: {path}") from exc
                time.sleep(0.05)
        yield descriptor
    finally:
        try:
            if fault_injector is not None:
                fault_injector("lock_exit_unlock")
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except Exception as exc:
            warnings.append(
                {
                    "point": "lock_exit_unlock",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        try:
            if fault_injector is not None:
                fault_injector("lock_exit_close")
            os.close(descriptor)
            descriptor = -1
        except Exception as exc:
            warnings.append(
                {
                    "point": "lock_exit_close",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                warnings.append(
                    {
                        "point": "lock_exit_emergency_close",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )


def read_regular_at(directory_fd: int, name: str, *, require_single_link: bool = False) -> bytes:
    if not name or "/" in name or name in {".", ".."}:
        raise PublicationError(f"member name is not a single path component: {name!r}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicationError(f"cannot open regular member: {name}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PublicationError(f"member is not a regular file: {name}")
        if require_single_link and info.st_nlink != 1:
            raise PublicationError(f"member does not have exactly one hard link: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_optional_regular_at(directory_fd: int, name: str) -> bytes | None:
    try:
        return read_regular_at(directory_fd, name)
    except PublicationError as exc:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise exc


def read_regular_path(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeSyncError(f"path is not a regular non-symlink file: {path}")
    return path.read_bytes()


def write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise PublicationError("short write")
        remaining = remaining[written:]


def read_all_from_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, str):
        return value == UNKNOWN or value.startswith(PENDING_PREFIX)
    if isinstance(value, list):
        return any(_contains_unresolved(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unresolved(item) for item in value.values())
    return False


def _expect_hex(value: Any, pattern: re.Pattern[str], *, label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise BindingError(f"{label} is not a frozen lowercase hex identifier")
    return value


def _expect_exact(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise RuntimeSyncError(f"{label} drift")


def _expect_typed_exact(actual: Any, expected: Any, *, label: str) -> None:
    """Deep exact equality with closed mappings and bool/int separation."""

    if type(actual) is not type(expected):
        raise RuntimeSyncError(
            f"{label} type drift: {type(actual).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise RuntimeSyncError(f"{label} key closure drift")
        for key, value in expected.items():
            _expect_typed_exact(actual[key], value, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeSyncError(f"{label} list length drift")
        for index, value in enumerate(expected):
            _expect_typed_exact(actual[index], value, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise RuntimeSyncError(f"{label} value drift")


def compiled_core_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable config core used to compare I and B."""

    return {key: copy.deepcopy(value) for key, value in config.items() if key != "implementation_binding"}


def compiled_core_sha256(config: dict[str, Any]) -> str:
    return sha256(json.dumps(compiled_core_projection(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _expect_int(value: Any, expected: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise BindingError(f"{label} must be the exact integer {expected}")


def validate_bound_config(config: dict[str, Any]) -> None:
    """Validate the direct config-only B binding before any runtime access."""

    _expect_typed_exact(
        set(config),
        {
            "schema_version", "protocol_id", "contract_id", "phase_id", "dataset_id",
            "event_id", "event_name", "sync_type", "implementation_binding",
            "repository_authority", "runtime", "attempt_lineage_truth",
            "unresolved_blockers", "successor_invariants", "historical_runtime_authority",
            "access_and_materialization_boundary", "publication_policy",
        },
        label="config root",
    )
    _expect_typed_exact(
        {key: config.get(key) for key in (
            "schema_version", "protocol_id", "contract_id", "phase_id", "dataset_id",
            "event_id", "event_name", "sync_type",
        )},
        {
            "schema_version": "route_a_v3_gse114002_endpoint_geometry_reconciliation_v2_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "dataset_id": "GSE114002",
            "event_id": "A1-EVT-039",
            "event_name": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_LINEAGE_SYNCED_GATE_UNCHANGED",
            "sync_type": "APPEND_ONLY_ATTEMPT_LINEAGE_REGISTRATION_NO_GATE_CHANGE",
        },
        label="config identity",
    )
    binding = config.get("implementation_binding")
    if not isinstance(binding, dict) or binding.get("status") != "BOUND":
        raise BindingError("runtime-sync config is not BOUND")
    _expect_typed_exact(set(binding), {
        "binding_scheme", "status", "implementation_commit", "implementation_script_path",
        "implementation_script_sha256", "implementation_test_path", "implementation_test_sha256",
        "compiled_core_sha256", "unknown_to_bound_scalar_paths",
    }, label="implementation binding")
    _expect_exact(binding.get("binding_scheme"), "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1", label="binding scheme")
    implementation = _expect_hex(binding.get("implementation_commit"), HEX40, label="implementation commit")
    _expect_hex(binding.get("implementation_script_sha256"), HEX64, label="script SHA-256")
    _expect_hex(binding.get("implementation_test_sha256"), HEX64, label="test SHA-256")
    _expect_hex(binding.get("compiled_core_sha256"), HEX64, label="compiled core SHA-256")
    _expect_exact(binding.get("implementation_script_path"), SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding.get("implementation_test_path"), TEST_REPO_PATH, label="test path")
    _expect_exact(binding.get("unknown_to_bound_scalar_paths"), [
        "implementation_binding.status", "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256", "implementation_binding.implementation_test_sha256",
    ], label="binding scalar allowlist")
    _expect_exact(binding.get("compiled_core_sha256"), compiled_core_sha256(config), label="compiled core")

    authority = config.get("repository_authority")
    if not isinstance(authority, dict):
        raise BindingError("repository authority is absent")
    _expect_typed_exact(set(authority), {
        "production_repo_root", "branch", "base_commit", "base_commit_expected_parent",
        "implementation_commit_expected_parent",
        "binding_commit_expected_parent", "implementation_commit_exact_changed_paths",
        "binding_commit_exact_changed_paths", "gse114002_v2_attempt_commit_lineage",
    }, label="repository authority")
    _expect_exact(authority.get("production_repo_root"), str(PRODUCTION_REPO_ROOT), label="repo root")
    _expect_exact(authority.get("branch"), "routea-v3-a1-20260810", label="branch")
    base = _expect_hex(authority.get("base_commit"), HEX40, label="base commit")
    _expect_exact(base, "f42df04cd44bdfa8c17b08b88bc0e1082fc1bf4b", label="base commit")
    _expect_exact(authority.get("base_commit_expected_parent"), "a148c737101ed8d0e24209233b86a55e2710633e", label="base parent")
    _expect_exact(authority.get("implementation_commit_expected_parent"), base, label="I parent")
    _expect_exact(authority.get("binding_commit_expected_parent"), "IMPLEMENTATION_COMMIT_FROM_BINDING", label="B parent")
    if implementation == base:
        raise BindingError("implementation commit did not advance from base")
    _expect_exact(authority.get("implementation_commit_exact_changed_paths"), sorted([CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]), label="I changed paths")
    _expect_exact(authority.get("binding_commit_exact_changed_paths"), [CONFIG_REPO_PATH], label="B changed paths")
    _expect_typed_exact(
        authority.get("gse114002_v2_attempt_commit_lineage"),
        [
            {
                "attempt_id": "ATTEMPT_1_FAILED_PRESERVED",
                "implementation_commit": "78b06434a9b94eaf5149dff7f6bb6b2d58e76ade",
                "binding_commit": "998d030a51737bfa1e27580efe8b89e22ae39149",
            },
            {
                "attempt_id": "ATTEMPT_2_CURRENT",
                "implementation_commit": "1543f09e74643a9a36b89742c7a1cc458b6b0d56",
                "binding_commit": "a148c737101ed8d0e24209233b86a55e2710633e",
            },
        ],
        label="GSE114002 v2 attempt commit lineage",
    )

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise BindingError("runtime section is absent")
    _expect_typed_exact(set(runtime), {
        "run_root", "allowed_prepared_root", "predecessor_event_id", "predecessor_event_count",
        "predecessor_manifest_output_count", "successor_event_id", "successor_event_count",
        "successor_manifest_output_count", "mutable_publish_order", "allowed_mutable_states",
        "predecessor_mutables", "predecessor_tail_event", "attempts", "sync_name",
        "output_delta_count", "forbidden_path_tokens",
    }, label="runtime")
    _expect_exact(runtime.get("run_root"), "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", label="run root")
    _expect_exact(runtime.get("allowed_prepared_root"), "/mnt/cunyuliu/mrna_xeditflow_routea_v3/staging/runtime_sync/A1", label="allowed prepared root")
    for field, expected in (("predecessor_event_count", 38), ("successor_event_count", 39), ("predecessor_manifest_output_count", 106), ("successor_manifest_output_count", 122), ("output_delta_count", 16)):
        _expect_int(runtime.get(field), expected, label=field)
    _expect_exact(runtime.get("predecessor_event_id"), "A1-EVT-038", label="predecessor event")
    _expect_exact(runtime.get("successor_event_id"), "A1-EVT-039", label="successor event")
    _expect_exact(runtime.get("mutable_publish_order"), list(MUTABLE_NAMES), label="mutable order")
    _expect_exact(runtime.get("allowed_mutable_states"), [
        ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"], ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"], ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"],
    ], label="recovery states")
    _expect_typed_exact(runtime.get("predecessor_mutables"), {
        "STATUS.json": {"bytes": 20114, "sha256": "80f0b7a987afa27d6be500c7177e03405dd5602c350cf5ff99ed7ecbe0f9def4", "snapshot_name": "STATUS_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json"},
        "RUN_MANIFEST.json": {"bytes": 43342, "sha256": "389f6efeb10af0a6d9d9907736d668c016bc913682b879391f853761a514adfd", "snapshot_name": "RUN_MANIFEST_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json"},
        "EVENT_LOG.jsonl": {"bytes": 45298, "sha256": "fcfb4155191548f228fe823552641906a2abedc9321401da1822e9a1bdf9b496", "snapshot_name": "EVENT_LOG_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.jsonl"},
    }, label="predecessor mutables")
    _expect_typed_exact(runtime.get("predecessor_tail_event"), {
        "event_id": "A1-EVT-038",
        "sha256": "f532c7ade9f60898bd993739c779326d2c6990def17760c92395029c5684b423",
        "training_started_key_present": True,
        "training_started": False,
    }, label="EVT038 tail identity")
    _expect_typed_exact(runtime.get("attempts"), EXPECTED_ATTEMPTS, label="attempt bundle identities")
    _expect_exact(runtime.get("sync_name"), "A1_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json", label="runtime sync name")
    _expect_typed_exact(runtime.get("forbidden_path_tokens"), ["GSE246381", "/restricted/", "/sealed_external/", "FASTQ", "raw_replay"], label="forbidden path tokens")
    _expect_typed_exact(config.get("attempt_lineage_truth"), {
        "attempt_count": 2,
        "current_attempt_id": "ATTEMPT_2_CURRENT",
        "failed_attempt_preserved": True,
        "current_mechanical_status": CURRENT_MECHANICAL_STATUS,
        "qualification_changed": False,
        "gate_changed": False,
        "historical_attempt_rewritten": False,
    }, label="attempt lineage truth")
    _expect_typed_exact(config.get("unresolved_blockers"), EXPECTED_BLOCKERS, label="seven blockers")
    _expect_typed_exact(config.get("successor_invariants"), {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified_independent_ordinary_studies": 0, "qualified_a1_studies": 0, "qualified_a2_dense_studies": 0,
        "metadata_only_qualification_count": 0, "ordinary_study_contribution": 0, "a1_study_contribution": 0,
        "true_a2_study_contribution": 0, "canonical_intervention_record_count": 0, "qualified": False,
        "training_started": False, "training_allowed": False, "training_authorized": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
    }, label="gate truth")
    _expect_typed_exact(config.get("historical_runtime_authority"), {
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
    }, label="historical runtime authority")
    _expect_typed_exact(config.get("access_and_materialization_boundary"), {
        "real_row_level_data_opened": False, "raw_reads_or_alignments_opened": False, "raw_fastq_body_read_count": 0,
        "raw_replay_run_count": 0, "paper_native_xtail_replay_run_count": 0,
        "gpu_work_started": False, "aggregate_reconciliation_artifact_body_opened": True,
        "aggregate_reconciliation_artifact_registration_count": 12,
        "row_level_payload_included": False,
        "row_identifier_payload_included": False, "row_key_payload_included": False,
        "sequence_payload_included": False, "effect_value_payload_included": False,
        "gene_payload_included": False, "barcode_payload_included": False,
        "annotation_label_payload_included": False, "restricted_or_sealed_path_accessed": False,
        "restricted_or_sealed_payload_contact": False, "canonical_read_count": 0,
        "canonical_write_count": 0, "qualifier_execution_count": 0,
        "training_run_count": 0, "model_selection_run_count": 0,
    }, label="access and materialization boundary")
    _expect_typed_exact(config.get("publication_policy"), {
        "immutable_no_overwrite": True, "mutable_order_recoverable": True,
        "event_is_last_mutable_commit": True,
        "post_commit_error_truth": "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY",
        "immutable_temp_unlink_failure_truth": "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION",
        "sync_references_successor_hashes": False, "successors_reference_sync_hash": True,
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }, label="publication policy")
    if _contains_unresolved(config):
        raise BindingError("bound runtime-sync config contains UNKNOWN/PENDING tokens")


def load_bound_config(config_path: Path, *, production: bool) -> tuple[dict[str, Any], bytes]:
    lexical = lexical_absolute(config_path)
    if production and lexical != PRODUCTION_CONFIG_PATH:
        raise BindingError(f"production config path must be exactly {PRODUCTION_CONFIG_PATH}")
    payload = read_regular_path(lexical)
    config = load_json(payload, label=str(lexical))
    validate_bound_config(config)
    return config, payload


def expected_unknown_i_config(bound_config: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(bound_config)
    for key in ("status", "implementation_commit", "implementation_script_sha256", "implementation_test_sha256"):
        expected["implementation_binding"][key] = UNKNOWN
    return expected


def _run_git(repo_root: Path, *arguments: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run([GIT_BINARY, "-C", str(repo_root), *arguments], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuthorityError("read-only git command failed to start") from exc
    if result.returncode not in allowed_returncodes:
        raise AuthorityError(f"read-only git command failed: {arguments!r}")
    return result.stdout


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{relative}")


def _paths_changed_by_commit(repo_root: Path, commit: str) -> list[str]:
    return sorted(_run_git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines())


def audit_repo_authority(repo_root: Path, config: dict[str, Any], config_payload: bytes) -> dict[str, Any]:
    if lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT or repo_root.is_symlink() or not repo_root.is_dir():
        raise AuthorityError("production repository root drift")
    binding, authority = config["implementation_binding"], config["repository_authority"]
    implementation, base, branch = binding["implementation_commit"], authority["base_commit"], authority["branch"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    origin = _run_git(repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}").decode().strip()
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), branch, label="branch")
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip(), f"origin/{branch}", label="upstream")
    _expect_exact(_run_git(repo_root, "rev-parse", "@{upstream}").decode().strip(), head, label="upstream head")
    _expect_exact(origin, head, label="origin head")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{head}^").decode().strip(), implementation, label="B parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{implementation}^").decode().strip(), base, label="I parent")
    lineage = authority["gse114002_v2_attempt_commit_lineage"]
    attempt1_i, attempt1_b = lineage[0]["implementation_commit"], lineage[0]["binding_commit"]
    attempt2_i, attempt2_b = lineage[1]["implementation_commit"], lineage[1]["binding_commit"]
    _expect_exact(_run_git(repo_root, "rev-parse", f"{base}^").decode().strip(), attempt2_b, label="ledger parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{attempt2_b}^").decode().strip(), attempt2_i, label="attempt 2 B parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{attempt2_i}^").decode().strip(), attempt1_b, label="attempt 2 I parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{attempt1_b}^").decode().strip(), attempt1_i, label="attempt 1 B parent")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AuthorityError("worktree or index is dirty")
    _run_git(repo_root, "merge-base", "--is-ancestor", base, head)
    _expect_exact(_paths_changed_by_commit(repo_root, implementation), authority["implementation_commit_exact_changed_paths"], label="I paths")
    _expect_exact(_paths_changed_by_commit(repo_root, head), authority["binding_commit_exact_changed_paths"], label="B paths")
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config blob drift")
    i_config = load_json(_git_blob(repo_root, implementation, CONFIG_REPO_PATH), label="I config")
    if i_config != expected_unknown_i_config(config) or compiled_core_projection(i_config) != compiled_core_projection(config):
        raise AuthorityError("I/B config transition drift")
    for path, digest in ((SCRIPT_REPO_PATH, binding["implementation_script_sha256"]), (TEST_REPO_PATH, binding["implementation_test_sha256"])):
        if sha256(_git_blob(repo_root, implementation, path)) != digest or sha256(_git_blob(repo_root, head, path)) != digest or sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"runtime-sync implementation blob drift: {path}")
    return {
        "status": "PASS_EXACT_GSE114002_V2_LINEAGE_TO_LEDGER_TO_RUNTIME_SYNC_I_TO_CONFIG_ONLY_B",
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin,
        "config_sha256": sha256(config_payload),
        "base_commit": base,
        "base_commit_parent": attempt2_b,
        "implementation_commit": implementation,
        "attempt_commit_lineage": copy.deepcopy(lineage),
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value) is None:
        raise RuntimeSyncError("recorded_at must be RFC3339 +08:00")
    try:
        current, predecessor = datetime.fromisoformat(value), datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("invalid recorded_at") from exc
    if current.utcoffset() != timedelta(hours=8) or current <= predecessor:
        raise RuntimeSyncError("recorded_at must follow EVT-038")
    if current.astimezone(timezone.utc).date().isoformat() != "2026-08-11":
        raise RuntimeSyncError("recorded_at outside EVT-039 window")


def open_absolute_directory_nofollow(path: Path) -> int:
    """Open an absolute directory root-to-leaf without following symlinks."""

    raw = os.fspath(path)
    if not raw.startswith(os.sep):
        raise PublicationError(f"attempt root must be absolute: {raw}")
    components = raw.split(os.sep)[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise PublicationError(f"attempt root has unsafe components: {raw}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in components:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _final_target_sha256(path: Path) -> str:
    return sha256(
        b"ROUTE_A_V3_GSE114002_RECONCILIATION_V2_FINAL_TARGET\n"
        + os.fsencode(path)
        + b"\n"
    )


def _parse_sha256sums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise PublicationError("attempt SHA256SUMS is not ASCII") from exc
    declared: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise PublicationError("attempt SHA256SUMS line is malformed")
        digest, name = line[:64], line[66:]
        if HEX64.fullmatch(digest) is None or Path(name).name != name or name in declared:
            raise PublicationError("attempt SHA256SUMS entry is unsafe")
        declared[name] = digest
    if set(declared) != set(BUNDLE_JSON_NAMES):
        raise PublicationError("attempt SHA256SUMS declaration set differs")
    return declared


def _validate_attempt_bundle(attempt: dict[str, Any], payloads: dict[str, bytes]) -> dict[str, Any]:
    if set(payloads) != set(BUNDLE_NAMES):
        raise PublicationError("attempt bundle member set differs")
    declared = _parse_sha256sums(payloads["SHA256SUMS"])
    for name in BUNDLE_JSON_NAMES:
        if declared[name] != sha256(payloads[name]):
            raise PublicationError(f"attempt SHA256SUMS member binding differs: {name}")

    report = load_json(payloads["QUALIFICATION_REPORT.json"], label="attempt qualification report")
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="attempt publication marker")
    _expect_typed_exact(
        set(marker),
        {
            "schema_version", "record_type", "contract_id", "protocol_id", "dataset_id",
            "output_id", "scientific_status", "publication_mode", "sha256sums_sha256",
            "bundle_file_count_excluding_commit_marker",
            "bundle_member_names_excluding_commit_marker",
            "final_output_directory_name_sha256", "final_output_target_sha256",
            "committed", "commit_marker_written_last",
            "aggregate_acceptance_requires_exact_marker",
        },
        label="attempt publication marker",
    )
    root = Path(attempt["run_root"])
    expected_marker = {
        "schema_version": "1.0.0",
        "record_type": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_COMMIT",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "protocol_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2",
        "dataset_id": "GSE114002",
        "output_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_BUNDLE",
        "scientific_status": attempt["status"],
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]),
        "bundle_file_count_excluding_commit_marker": 5,
        "bundle_member_names_excluding_commit_marker": sorted(
            set(BUNDLE_JSON_NAMES) | {"SHA256SUMS"}
        ),
        "final_output_directory_name_sha256": sha256(root.name.encode("utf-8")),
        "final_output_target_sha256": _final_target_sha256(root),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }
    if marker.get("publication_mode") not in {
        PRIMARY_PUBLICATION_MODE,
        FALLBACK_PUBLICATION_MODE,
    }:
        raise PublicationError("attempt publication mode is outside the closed enum")
    for key, expected in expected_marker.items():
        _expect_typed_exact(marker.get(key), expected, label=f"attempt marker {key}")

    _expect_typed_exact(
        {key: report.get(key) for key in (
            "status", "qualified", "scientific_claim_status",
            "ordinary_study_contribution", "a1_intervention_study_contribution",
            "true_a2_dense_study_contribution", "canonical_record_count",
            "canonical_materialization_allowed", "training_allowed",
            "model_selection_allowed", "next_phase_authorized",
            "true_a2_claim_established", "aggregate_only",
        )},
        {
            "status": attempt["status"], "qualified": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
            "ordinary_study_contribution": 0,
            "a1_intervention_study_contribution": 0,
            "true_a2_dense_study_contribution": 0,
            "canonical_record_count": 0,
            "canonical_materialization_allowed": False,
            "training_allowed": False, "model_selection_allowed": False,
            "next_phase_authorized": False, "true_a2_claim_established": False,
            "aggregate_only": True,
        },
        label=f"{attempt['attempt_id']} qualification boundary",
    )
    implementation = report.get("implementation_binding")
    if not isinstance(implementation, dict):
        raise PublicationError("attempt report implementation binding is absent")
    _expect_typed_exact(
        {key: implementation.get(key) for key in (
            "status", "verified", "implementation_commit", "binding_commit",
        )},
        {
            "status": "PASS_BOUND_IMPLEMENTATION", "verified": True,
            "implementation_commit": attempt["implementation_commit"],
            "binding_commit": attempt["binding_commit"],
        },
        label=f"{attempt['attempt_id']} report commit binding",
    )
    blockers = report.get("blockers")
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        raise PublicationError("attempt blocker list is not closed and deduplicated")
    if attempt["attempt_id"] == "ATTEMPT_2_CURRENT":
        _expect_typed_exact(blockers, EXPECTED_BLOCKERS, label="current seven blockers")
    elif not set(EXPECTED_BLOCKERS).issubset(blockers):
        raise PublicationError("failed attempt does not preserve all current blockers")

    return {
        "attempt_id": attempt["attempt_id"],
        "run_root": attempt["run_root"],
        "status": attempt["status"],
        "implementation_commit": attempt["implementation_commit"],
        "binding_commit": attempt["binding_commit"],
        "member_count": 6,
        "members": [
            {
                "name": name,
                "bytes": attempt["members"][name]["bytes"],
                "sha256": attempt["members"][name]["sha256"],
            }
            for name in BUNDLE_NAMES
        ],
        "terminal_commit": {
            "marker_name": "PUBLICATION_COMMIT.json",
            "marker_sha256": attempt["members"]["PUBLICATION_COMMIT.json"]["sha256"],
            "committed": True,
            "commit_marker_written_last": True,
            "scientific_status": attempt["status"],
            "publication_mode": marker["publication_mode"],
        },
    }


def read_exact_attempt_lineage(config: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    forbidden = [token.lower() for token in config["runtime"]["forbidden_path_tokens"]]
    for attempt in config["runtime"]["attempts"]:
        root = Path(attempt["run_root"])
        lowered = os.fspath(root).lower()
        if any(token in lowered for token in forbidden):
            raise PublicationError("attempt root enters a forbidden scope")
        directory_fd = open_absolute_directory_nofollow(root)
        try:
            observed = set(os.listdir(directory_fd))
            if observed != set(BUNDLE_NAMES):
                raise PublicationError("attempt directory member set differs")
            payloads: dict[str, bytes] = {}
            for name in BUNDLE_NAMES:
                payload = read_regular_at(directory_fd, name, require_single_link=True)
                specification = attempt["members"][name]
                if len(payload) != specification["bytes"] or sha256(payload) != specification["sha256"]:
                    raise PublicationError(f"exact attempt member identity drift: {attempt['attempt_id']} {name}")
                payloads[name] = payload
        finally:
            os.close(directory_fd)
        summaries.append(_validate_attempt_bundle(attempt, payloads))
    return summaries


def _validate_predecessor_objects(status: dict[str, Any], manifest: dict[str, Any], events: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if len(events) != 38 or not events or events[-1].get("event_id") != "A1-EVT-038":
        raise PublicationError("predecessor event count or tail drift")
    if events[-1].get("training_started") is not False:
        raise PublicationError("EVT038 explicit training_started=false truth drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 106:
        raise PublicationError("predecessor outputs are not exactly 106")
    for key, expected in {"run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE", "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "claim_status": "NOT_ESTABLISHED", "training_started": False, "next_phase_authorized": False, "code_commit": config["historical_runtime_authority"]["code_commit"]}.items():
        _expect_exact(status.get(key), expected, label=f"predecessor STATUS {key}")
    for key, expected in {"run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE", "claim_status": "NOT_ESTABLISHED", "code_commit": config["historical_runtime_authority"]["code_commit"], "active_authority_commit": config["historical_runtime_authority"]["active_authority_commit"]}.items():
        _expect_exact(manifest.get(key), expected, label=f"predecessor manifest {key}")
    new_paths = {
        str(Path(attempt["run_root"]) / name)
        for attempt in config["runtime"]["attempts"]
        for name in BUNDLE_NAMES
    }
    new_paths.update(
        str(Path(config["runtime"]["run_root"]) / name)
        for name in immutable_names(config)
    )
    if any(isinstance(item, dict) and item.get("absolute_path") in new_paths for item in outputs):
        raise PublicationError("EVT039 output is already registered")


def read_exact_predecessor(run_fd: int, config: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payloads: dict[str, bytes] = {}
    for name in MUTABLE_NAMES:
        payload = read_regular_at(run_fd, name)
        spec = config["runtime"]["predecessor_mutables"][name]
        if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
            raise PublicationError(f"exact predecessor mutable drift: {name}")
        payloads[name] = payload
    status = load_json(payloads["STATUS.json"], label="predecessor status")
    manifest = load_json(payloads["RUN_MANIFEST.json"], label="predecessor manifest")
    events = load_json_lines(payloads["EVENT_LOG.jsonl"], label="predecessor events")
    tail_line = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    if sha256(tail_line) != config["runtime"]["predecessor_tail_event"]["sha256"]:
        raise PublicationError("EVT038 tail line identity drift")
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def output_record(artifact_type: str, absolute_path: Path, digest: str) -> dict[str, str]:
    return {"artifact_type": artifact_type, "absolute_path": str(absolute_path), "sha256": digest, "status": "COMPLETE"}


def snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"] for name in MUTABLE_NAMES}


def immutable_names(config: dict[str, Any]) -> tuple[str, ...]:
    snapshots = snapshot_names(config)
    return (snapshots["STATUS.json"], snapshots["RUN_MANIFEST.json"], snapshots["EVENT_LOG.jsonl"], config["runtime"]["sync_name"])


def expected_output_delta(config: dict[str, Any], sync_digest: str) -> list[dict[str, str]]:
    runtime, snapshots = config["runtime"], snapshot_names(config)
    root = Path(runtime["run_root"])
    records: list[dict[str, str]] = []
    for index, attempt in enumerate(runtime["attempts"], start=1):
        attempt_root = Path(attempt["run_root"])
        for name in BUNDLE_NAMES:
            token = name.upper().replace(".JSON", "").replace(".", "_")
            records.append(
                output_record(
                    f"A1_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_{index}_{token}",
                    attempt_root / name,
                    attempt["members"][name]["sha256"],
                )
            )
    records.extend([
        output_record("A1_STATUS_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_SNAPSHOT", root / snapshots["STATUS.json"], runtime["predecessor_mutables"]["STATUS.json"]["sha256"]),
        output_record("A1_RUN_MANIFEST_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_SNAPSHOT", root / snapshots["RUN_MANIFEST.json"], runtime["predecessor_mutables"]["RUN_MANIFEST.json"]["sha256"]),
        output_record("A1_EVENT_LOG_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_SNAPSHOT", root / snapshots["EVENT_LOG.jsonl"], runtime["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"]),
        output_record("A1_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1", root / runtime["sync_name"], sync_digest),
    ])
    return records


def evt039_event_document(
    config: dict[str, Any],
    *,
    recorded_at: str,
    sync_digest: str,
    attempt_lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the closed, self-contained EVT-039 lineage event."""

    event: dict[str, Any] = {
        "event_id": "A1-EVT-039",
        "at": recorded_at,
        "phase_id": "A1",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "sync_record": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_id": "A1-EVT-038",
        "predecessor_event_line_sha256": config["runtime"]["predecessor_tail_event"]["sha256"],
        "attempt_lineage": copy.deepcopy(attempt_lineage),
        "attempt_count": 2,
        "current_attempt_id": "ATTEMPT_2_CURRENT",
        "current_attempt_status": CURRENT_MECHANICAL_STATUS,
        "failed_attempt_preserved": True,
        "historical_attempt_rewritten": False,
        "qualification_changed": False,
        "gate_changed": False,
        "scientific_blocker_count": 7,
        "unresolved_blockers": list(config["unresolved_blockers"]),
        "pre_sync_snapshot_count": 3,
        "manifest_output_count_before": 106,
        "manifest_output_count_after": 122,
        "detail": "Append-only lineage sync: the failed first and mechanically reconciled second GSE114002 v2 aggregate bundles are registered with exact terminal-marker and commit bindings. No qualifier, row-level data, canonical materialization, training, model selection, GPU work, scientific qualification, or next-phase authorization is added.",
    }
    event.update(copy.deepcopy(config["successor_invariants"]))
    event.update(copy.deepcopy(config["access_and_materialization_boundary"]))
    return event


def build_successors(config: dict[str, Any], predecessor_payloads: dict[str, bytes], predecessor_status: dict[str, Any], predecessor_manifest: dict[str, Any], predecessor_events: list[dict[str, Any]], attempt_lineage: list[dict[str, Any]], authority_audit: dict[str, Any], recorded_at: str) -> dict[str, bytes]:
    runtime, binding = config["runtime"], config["implementation_binding"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    root, snapshots = Path(runtime["run_root"]), snapshot_names(config)
    sync = {
        "schema_version": "1.0.0", "record_type": "ROUTE_A_V3_A1_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC",
        "sync_type": config["sync_type"],
        "contract_id": config["contract_id"], "phase_id": "A1", "dataset_id": "GSE114002",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", "recorded_at": recorded_at,
        "predecessor_runtime": {"runtime_root": str(root), "last_event_id": "A1-EVT-038", "event_count": 38, "manifest_output_count": 106, "last_event_line_sha256": config["runtime"]["predecessor_tail_event"]["sha256"], "immutable_pre_sync_snapshots": [
            {"source_mutable_path": str(root / name), "snapshot_path": str(root / snapshots[name]), "bytes": len(predecessor_payloads[name]), "sha256": sha256(predecessor_payloads[name])} for name in MUTABLE_NAMES
        ]},
        "attempt_lineage": copy.deepcopy(attempt_lineage),
        "attempt_lineage_truth": copy.deepcopy(config["attempt_lineage_truth"]),
        "runtime_sync_publisher_authority": {"status": authority_audit["status"], "base_commit": config["repository_authority"]["base_commit"], "implementation_commit": binding["implementation_commit"], "binding_commit": authority_audit["binding_commit"], "config_sha256": authority_audit["config_sha256"], "script_sha256": binding["implementation_script_sha256"], "test_sha256": binding["implementation_test_sha256"]},
        "scientific_blockers": {"count": 7, "exact": config["unresolved_blockers"]},
        "a1_gate_snapshot": copy.deepcopy(config["successor_invariants"]),
        "access_and_materialization_boundary": copy.deepcopy(config["access_and_materialization_boundary"]),
        "hash_linkage": {"direction": "EVT038_PREDECESSOR_AND_TWO_COMMITTED_ATTEMPT_BUNDLES_TO_RUNTIME_SYNC_TO_SUCCESSORS", "sync_record_references_successor_hashes": False, "successors_reference_sync_hash": True},
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload, prefix = json_bytes(sync), "gse114002_endpoint_geometry_reconciliation_v2_runtime_sync_"
    sync_digest = sha256(sync_payload)
    status = dict(predecessor_status)
    status.update({"updated_at": recorded_at, prefix + "status": "SYNCED_EVT_039", prefix + "runtime_sync_record_sha256": sync_digest, prefix + "attempt_count": 2, prefix + "current_attempt_status": CURRENT_MECHANICAL_STATUS, prefix + "failed_attempt_preserved": True, prefix + "historical_attempt_rewritten": False, prefix + "qualification_changed": False, prefix + "gate_changed": False, prefix + "scientific_blocker_count": 7})
    event = evt039_event_document(
        config,
        recorded_at=recorded_at,
        sync_digest=sync_digest,
        attempt_lineage=attempt_lineage,
    )
    manifest = dict(predecessor_manifest)
    manifest[prefix + "runtime_sync_record_sha256"] = sync_digest
    manifest["outputs"] = list(predecessor_manifest["outputs"]) + expected_output_delta(config, sync_digest)
    artifacts = {snapshots["STATUS.json"]: predecessor_payloads["STATUS.json"], snapshots["RUN_MANIFEST.json"]: predecessor_payloads["RUN_MANIFEST.json"], snapshots["EVENT_LOG.jsonl"]: predecessor_payloads["EVENT_LOG.jsonl"], runtime["sync_name"]: sync_payload, "STATUS.json": json_bytes(status), "RUN_MANIFEST.json": json_bytes(manifest), "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(event)}
    validate_successors(config, artifacts, predecessor_payloads, predecessor_status, predecessor_manifest, sync_digest)
    return artifacts


def validate_successors(config: dict[str, Any], artifacts: dict[str, bytes], predecessor_payloads: dict[str, bytes], predecessor_status: dict[str, Any], predecessor_manifest: dict[str, Any], sync_digest: str) -> None:
    runtime, snapshots = config["runtime"], snapshot_names(config)
    if set(artifacts) != set(MUTABLE_NAMES) | set(snapshots.values()) | {runtime["sync_name"]}:
        raise RuntimeSyncError("prepared artifact schema drift")
    for mutable, snapshot in snapshots.items():
        if artifacts[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError("snapshot byte drift")
    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label="GSE114002 runtime sync")
    if sha256(sync_payload) != sync_digest:
        raise RuntimeSyncError("runtime sync identity drift")
    _expect_typed_exact(set(sync), {
        "schema_version", "record_type", "sync_type", "contract_id", "phase_id",
        "dataset_id", "run_id", "recorded_at", "predecessor_runtime", "attempt_lineage",
        "attempt_lineage_truth", "runtime_sync_publisher_authority", "scientific_blockers",
        "a1_gate_snapshot", "access_and_materialization_boundary", "hash_linkage", "self_hash",
    }, label="runtime sync root")
    _expect_typed_exact(sync.get("sync_type"), config["sync_type"], label="runtime sync type")
    _expect_typed_exact(sync.get("attempt_lineage_truth"), config["attempt_lineage_truth"], label="runtime sync lineage truth")
    _expect_typed_exact(sync.get("access_and_materialization_boundary"), config["access_and_materialization_boundary"], label="runtime sync access boundary")
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode() in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("successor hash leaked into runtime sync")
    for name in MUTABLE_NAMES:
        if sync_digest.encode() not in artifacts[name]:
            raise RuntimeSyncError(f"successor lacks sync binding: {name}")
    status = load_json(artifacts["STATUS.json"], label="successor status")
    prefix = "gse114002_endpoint_geometry_reconciliation_v2_runtime_sync_"
    if {key: value for key, value in status.items() if key != "updated_at" and not key.startswith(prefix)} != {key: value for key, value in predecessor_status.items() if key != "updated_at"}:
        raise RuntimeSyncError("predecessor STATUS field rewritten")
    _expect_exact(status.get("training_started"), False, label="successor STATUS training_started")
    _expect_exact(status.get(prefix + "current_attempt_status"), CURRENT_MECHANICAL_STATUS, label="runtime-sync STATUS current attempt")
    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor manifest")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError("predecessor manifest field rewritten")
    if set(manifest) - set(predecessor_manifest) != {prefix + "runtime_sync_record_sha256"}:
        raise RuntimeSyncError("manifest top-level runtime-sync delta drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 122 or outputs[:106] != predecessor_manifest["outputs"] or outputs[106:] != expected_output_delta(config, sync_digest):
        raise RuntimeSyncError("manifest ordered 106 -> 122 append drift")
    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor events")
    old_events = load_json_lines(predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor events")
    if len(events) != 39 or events[:-1] != old_events or not artifacts["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG is not an exact append")
    expected_event = evt039_event_document(
        config,
        recorded_at=sync["recorded_at"],
        sync_digest=sync_digest,
        attempt_lineage=sync["attempt_lineage"],
    )
    _expect_typed_exact(events[-1], expected_event, label="EVT-039 closed event")

def _safe_absolute_components(raw_path: str, *, label: str) -> tuple[str, ...]:
    if not raw_path.startswith(os.sep):
        raise PublicationError(f"{label} must be absolute")
    raw_components = raw_path.split(os.sep)[1:]
    if not raw_components or any(component in {"", ".", ".."} for component in raw_components):
        raise PublicationError(f"{label} contains an empty, dot, or dot-dot component")
    return tuple(raw_components)


def _validate_prepared_path(
    prepared_directory: Path | str, config: dict[str, Any], *, production: bool
) -> tuple[Path, Path, tuple[str, ...]]:
    raw_path = os.fspath(prepared_directory)
    _safe_absolute_components(raw_path, label="prepared directory")
    path = Path(raw_path)
    runtime = config["runtime"]
    allowed_raw = str(runtime["allowed_prepared_root"])
    allowed_components = _safe_absolute_components(allowed_raw, label="allowed prepared root")
    allowed_root = Path(allowed_raw)
    text = raw_path
    if not is_strict_descendant(path, allowed_root):
        raise PublicationError(
            f"production prepared directory must be below {runtime['allowed_prepared_root']}"
        )
    run_root = Path(runtime["run_root"])
    if path == run_root or is_strict_descendant(path, run_root) or is_strict_descendant(run_root, path):
        raise PublicationError("prepared directory overlaps the live run root")
    for attempt in runtime["attempts"]:
        attempt_root = Path(attempt["run_root"])
        if path == attempt_root or is_strict_descendant(path, attempt_root) or is_strict_descendant(attempt_root, path):
            raise PublicationError("prepared directory overlaps an attempt bundle root")
    for token in runtime["forbidden_path_tokens"]:
        if token.lower() in text.lower():
            raise PublicationError(f"prepared directory contains forbidden path token: {token}")
    path_components = _safe_absolute_components(raw_path, label="prepared directory")
    if path_components[: len(allowed_components)] != allowed_components:
        raise PublicationError("prepared directory does not share the exact allowed-root components")
    relative_components = path_components[len(allowed_components) :]
    if not relative_components:
        raise PublicationError("prepared directory must be below, not equal to, allowed root")
    return path, allowed_root, relative_components


def _open_child_directory(parent_fd: int, component: str, *, create: bool) -> int:
    if component in {"", ".", ".."} or "/" in component:
        raise PublicationError(f"unsafe prepared-directory component: {component!r}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(component, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise PublicationError(f"prepared-directory component is absent: {component}")
        try:
            os.mkdir(component, 0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(component, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise PublicationError(
                f"created prepared-directory component is not a nofollow directory: {component}"
            ) from exc
    except OSError as exc:
        raise PublicationError(
            f"prepared-directory component is not a nofollow directory: {component}"
        ) from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise PublicationError(f"prepared-directory component is not a directory: {component}")
    return descriptor


def open_trusted_prepared_directory(
    prepared_directory: Path | str,
    config: dict[str, Any],
    *,
    production: bool,
    create: bool,
) -> tuple[Path, int]:
    """Open/create prepared paths only below a nofollow-walked trusted allowed-root fd."""

    path, allowed_root, relative_components = _validate_prepared_path(
        prepared_directory, config, production=production
    )
    root_fd = open_directory(Path(os.sep))
    current_fd = root_fd
    try:
        for component in _safe_absolute_components(
            str(allowed_root), label="allowed prepared root"
        ):
            child_fd = _open_child_directory(current_fd, component, create=False)
            os.close(current_fd)
            current_fd = child_fd
        for component in relative_components:
            child_fd = _open_child_directory(current_fd, component, create=create)
            os.close(current_fd)
            current_fd = child_fd
        return path, current_fd
    except Exception:
        os.close(current_fd)
        raise


def _write_prepared_member(directory_fd: int, name: str, payload: bytes) -> str:
    existing = read_optional_regular_at(directory_fd, name)
    if existing is not None:
        if existing != payload:
            raise PublicationError(f"refusing to overwrite differing prepared member: {name}")
        return "EXISTING_EXACT_REUSED"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, 0o644, dir_fd=directory_fd)
        write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    if read_regular_at(directory_fd, name) != payload:
        raise PublicationError(f"prepared member post-write mismatch: {name}")
    return "CREATED_EXCLUSIVE"


def write_prepared_directory(
    prepared_directory: Path | str,
    artifacts: dict[str, bytes],
    config: dict[str, Any],
    *,
    production: bool,
) -> dict[str, Any]:
    prepared_path, directory_fd = open_trusted_prepared_directory(
        prepared_directory, config, production=production, create=True
    )
    try:
        observed = set(os.listdir(directory_fd))
        expected = set(artifacts)
        if not observed.issubset(expected):
            raise PublicationError(f"unexpected prepared members: {sorted(observed - expected)!r}")
        results = {
            name: _write_prepared_member(directory_fd, name, artifacts[name])
            for name in sorted(artifacts)
        }
        if set(os.listdir(directory_fd)) != expected:
            raise PublicationError("prepared member closure is incomplete")
    finally:
        os.close(directory_fd)
    return {
        "prepared_directory": str(prepared_path),
        "results": results,
        "bytes": {name: len(payload) for name, payload in artifacts.items()},
        "sha256": {name: sha256(payload) for name, payload in artifacts.items()},
    }


def read_prepared_directory(
    prepared_directory: Path | str,
    config: dict[str, Any],
    *,
    production: bool,
) -> tuple[Path, dict[str, bytes]]:
    expected_names = set(MUTABLE_NAMES) | set(immutable_names(config))
    prepared_path, directory_fd = open_trusted_prepared_directory(
        prepared_directory, config, production=production, create=False
    )
    try:
        observed = set(os.listdir(directory_fd))
        if observed != expected_names:
            raise PublicationError(
                f"prepared member set drift: observed={sorted(observed)!r}, expected={sorted(expected_names)!r}"
            )
        payloads = {name: read_regular_at(directory_fd, name) for name in sorted(expected_names)}
    finally:
        os.close(directory_fd)
    return prepared_path, payloads


def _load_runtime_context(
    config_path: Path,
    *,
    production: bool,
    config_override: dict[str, Any] | None,
    authority_override: dict[str, Any] | None,
    repo_root: Path | None,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if production and (config_override is not None or authority_override is not None or repo_root is not None):
        raise BindingError("test-only config/authority/repository overrides are forbidden in production")
    if config_override is None:
        config, config_payload = load_bound_config(config_path, production=production)
    else:
        config = copy.deepcopy(config_override)
        validate_bound_config(config)
        config_payload = json_bytes(config)
    if authority_override is None:
        actual_repo_root = repo_root or PRODUCTION_REPO_ROOT
        authority = audit_repo_authority(actual_repo_root, config, config_payload)
    else:
        authority = copy.deepcopy(authority_override)
        required = {
            "status",
            "binding_commit",
            "head_commit",
            "origin_branch_head_commit",
            "config_sha256",
        }
        if not required.issubset(authority):
            raise AuthorityError("test-only authority override is incomplete")
    return config, config_payload, authority


def prepare_runtime_sync(
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    prepared_directory: Path | str,
    recorded_at: str,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    authority_override: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config, _config_payload, authority = _load_runtime_context(
        config_path,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and run_root_override is not None:
        raise PublicationError("test-only run-root override is forbidden in production")
    prepared_path, _allowed_root, _relative = _validate_prepared_path(
        prepared_directory, config, production=production
    )
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    lock_cleanup_warnings: list[dict[str, str]] = []
    with locked_directory(run_root, cleanup_warnings=lock_cleanup_warnings) as run_fd:
        predecessor, status, manifest, events = read_exact_predecessor(run_fd, config)
        attempt_lineage = read_exact_attempt_lineage(config)
        artifacts = build_successors(
            config,
            predecessor,
            status,
            manifest,
            events,
            attempt_lineage,
            authority,
            recorded_at,
        )
    if lock_cleanup_warnings:
        raise PublicationError(
            f"run-lock cleanup failed before prepared publication: {lock_cleanup_warnings!r}"
        )
    written = write_prepared_directory(
        prepared_directory, artifacts, config, production=production
    )
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": config["event_id"],
        "prepared_directory": str(prepared_path),
        "runtime_artifact_count": len(artifacts),
        "manifest_output_transition": "106_TO_122",
        "attempt_lineage_state": "TWO_EXACT_COMMITTED_BUNDLES_REPLAYED",
        "authority": authority,
        **written,
    }


def _predecessor_from_prepared(
    prepared: dict[str, bytes], config: dict[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    runtime = config["runtime"]
    snapshots = snapshot_names(config)
    payloads: dict[str, bytes] = {}
    for mutable_name, snapshot_name in snapshots.items():
        payload = prepared[snapshot_name]
        spec = runtime["predecessor_mutables"][mutable_name]
        if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
            raise PublicationError(f"prepared predecessor snapshot drift: {snapshot_name}")
        payloads[mutable_name] = payload
    status = load_json(payloads["STATUS.json"], label="prepared predecessor STATUS")
    manifest = load_json(payloads["RUN_MANIFEST.json"], label="prepared predecessor manifest")
    events = load_json_lines(payloads["EVENT_LOG.jsonl"], label="prepared predecessor events")
    tail_line = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    if sha256(tail_line) != runtime["predecessor_tail_event"]["sha256"]:
        raise PublicationError("prepared EVT038 tail line identity drift")
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def validate_prepared_against_context(
    prepared: dict[str, bytes],
    config: dict[str, Any],
    authority: dict[str, Any],
    attempt_lineage: list[dict[str, Any]],
) -> None:
    predecessor, status, manifest, events = _predecessor_from_prepared(prepared, config)
    sync = load_json(prepared[config["runtime"]["sync_name"]], label="prepared sync record")
    recorded_at = sync.get("recorded_at")
    if not isinstance(recorded_at, str):
        raise PublicationError("prepared sync recorded_at is absent")
    expected = build_successors(
        config,
        predecessor,
        status,
        manifest,
        events,
        attempt_lineage,
        authority,
        recorded_at,
    )
    if prepared != expected:
        differing = sorted(name for name in expected if prepared.get(name) != expected[name])
        raise PublicationError(f"prepared runtime bytes do not match current authority: {differing!r}")


def classify_target(
    run_fd: int, prepared: dict[str, bytes], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Freshly classify every source/immutable/mutable under the held run lock."""

    runtime = config["runtime"]
    stale_temporaries = sorted(
        name
        for name in os.listdir(run_fd)
        if re.fullmatch(r"\.evt039\.[0-9]+\.[0-9a-f]{16}\.[^/]+\.tmp", name)
    )
    if stale_temporaries:
        raise PublicationError(
            "stale EVT-039 publisher temporary member makes the run namespace unclosed: "
            + repr(stale_temporaries)
        )
    predecessor, _status, _manifest, _events = _predecessor_from_prepared(prepared, config)
    attempt_lineage = read_exact_attempt_lineage(config)
    mutable_states: dict[str, str] = {}
    for name in MUTABLE_NAMES:
        current = read_regular_at(run_fd, name)
        if current == predecessor[name]:
            mutable_states[name] = "OLD_EXACT"
        elif current == prepared[name]:
            mutable_states[name] = "NEW_EXACT"
        else:
            raise PublicationError(f"mutable preflight drift: {name} sha256={sha256(current)}")
    state = [mutable_states[name] for name in MUTABLE_NAMES]
    if state not in runtime["allowed_mutable_states"]:
        raise PublicationError(f"mutable publication-order state is invalid: {state!r}")

    immutable_states: dict[str, str] = {}
    for name in immutable_names(config):
        existing = read_optional_regular_at(run_fd, name)
        if existing is None:
            immutable_states[name] = "ABSENT"
        elif existing == prepared[name]:
            immutable_states[name] = "EXISTING_EXACT"
        else:
            raise PublicationError(f"existing immutable artifact differs: {name}")
    return mutable_states, immutable_states, attempt_lineage


def _invoke_fault(fault_injector: FaultInjector | None, point: str) -> None:
    if fault_injector is not None:
        fault_injector(point)


def _warning(point: str, exc: BaseException) -> dict[str, str]:
    return {"point": point, "error_type": type(exc).__name__, "message": str(exc)}


def _reconfirm_existing_commit(
    directory_fd: int,
    *,
    state: str,
    fault_point: str,
    fault_injector: FaultInjector | None,
) -> dict[str, Any]:
    """Re-establish directory durability before accepting an existing commit."""

    warnings: list[dict[str, str]] = []
    try:
        _invoke_fault(fault_injector, fault_point)
        os.fsync(directory_fd)
    except Exception as exc:
        warnings.append(_warning(fault_point, exc))
    return {
        "state": state,
        "committed_by_this_call": False,
        "accepted": not warnings,
        "warnings": warnings,
    }


def _cleanup_precommit_temp(directory_fd: int, temporary_name: str, descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
    try:
        os.unlink(temporary_name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def publish_immutable_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    """No-overwrite publish with an explicit link-success commit point.

    After the hard-link commit succeeds, cleanup/fsync/close/verification errors
    become warnings and never escape as an ordinary pre-commit failure.
    """

    existing = read_optional_regular_at(directory_fd, name)
    if existing is not None:
        if existing != payload:
            raise PublicationError(f"existing immutable artifact differs: {name}")
        return _reconfirm_existing_commit(
            directory_fd,
            state="EXISTING_EXACT_REUSED",
            fault_point="immutable_existing_exact_directory_fsync",
            fault_injector=fault_injector,
        )

    temporary_name = f".evt039.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    committed = False
    try:
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=directory_fd)
        write_all(descriptor, payload)
        os.fsync(descriptor)
        held_info = os.fstat(descriptor)
        if not stat.S_ISREG(held_info.st_mode) or held_info.st_size != len(payload):
            raise PublicationError(f"temporary immutable identity drift: {name}")
        if read_all_from_fd(descriptor) != payload:
            raise PublicationError(f"temporary immutable bytes drift: {name}")
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            committed = True
        except FileExistsError:
            existing = read_regular_at(directory_fd, name)
            _cleanup_precommit_temp(directory_fd, temporary_name, descriptor)
            descriptor = None
            if existing != payload:
                raise PublicationError(f"immutable link race produced differing final: {name}")
            return _reconfirm_existing_commit(
                directory_fd,
                state="EXISTING_EXACT_REUSED",
                fault_point="immutable_existing_exact_directory_fsync",
                fault_injector=fault_injector,
            )
    except Exception:
        if not committed:
            _cleanup_precommit_temp(directory_fd, temporary_name, descriptor)
            descriptor = None
        raise

    warnings: list[dict[str, str]] = []

    def post_commit(point: str, operation: Callable[[], None]) -> None:
        try:
            _invoke_fault(fault_injector, point)
            operation()
        except Exception as exc:  # commit already exists; retain truthful committed state
            warnings.append(_warning(point, exc))

    def verify_identity() -> None:
        assert descriptor is not None
        held = os.fstat(descriptor)
        final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(final.st_mode):
            raise PublicationError(f"committed immutable is not regular: {name}")
        if (held.st_dev, held.st_ino, held.st_size) != (final.st_dev, final.st_ino, final.st_size):
            raise PublicationError(f"committed immutable identity differs from held temp: {name}")
        if read_regular_at(directory_fd, name) != payload:
            raise PublicationError(f"committed immutable bytes differ: {name}")

    post_commit("immutable_post_link_identity", verify_identity)
    post_commit(
        "immutable_post_link_unlink",
        lambda: os.unlink(temporary_name, dir_fd=directory_fd),
    )
    post_commit("immutable_post_link_directory_fsync", lambda: os.fsync(directory_fd))

    def close_held() -> None:
        nonlocal descriptor
        assert descriptor is not None
        os.close(descriptor)
        descriptor = None

    post_commit("immutable_post_link_close", close_held)
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError as exc:
            warnings.append(_warning("immutable_emergency_close", exc))
        descriptor = None
    return {
        "state": "CREATED_EXCLUSIVE",
        "committed_by_this_call": True,
        "accepted": not warnings,
        "warnings": warnings,
    }


def replace_mutable_at(
    directory_fd: int,
    name: str,
    old_payload: bytes,
    new_payload: bytes,
    *,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    current = read_regular_at(directory_fd, name)
    if current == new_payload:
        return _reconfirm_existing_commit(
            directory_fd,
            state="EXISTING_NEW_EXACT_REUSED",
            fault_point="mutable_existing_new_directory_fsync",
            fault_injector=fault_injector,
        )
    if current != old_payload:
        raise PublicationError(f"unexpected mutable predecessor: {name}")

    temporary_name = f".evt039.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    committed = False
    try:
        descriptor = os.open(temporary_name, flags, 0o644, dir_fd=directory_fd)
        write_all(descriptor, new_payload)
        os.fsync(descriptor)
        held_info = os.fstat(descriptor)
        if not stat.S_ISREG(held_info.st_mode) or held_info.st_size != len(new_payload):
            raise PublicationError(f"temporary mutable identity drift: {name}")
        if read_all_from_fd(descriptor) != new_payload:
            raise PublicationError(f"temporary mutable bytes drift: {name}")
        os.replace(temporary_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        committed = True
    except Exception:
        if not committed:
            _cleanup_precommit_temp(directory_fd, temporary_name, descriptor)
            descriptor = None
        raise

    warnings: list[dict[str, str]] = []

    def post_commit(point: str, operation: Callable[[], None]) -> None:
        try:
            _invoke_fault(fault_injector, point)
            operation()
        except Exception as exc:
            warnings.append(_warning(point, exc))

    def verify_identity() -> None:
        assert descriptor is not None
        held = os.fstat(descriptor)
        final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(final.st_mode):
            raise PublicationError(f"committed mutable is not regular: {name}")
        if (held.st_dev, held.st_ino, held.st_size) != (final.st_dev, final.st_ino, final.st_size):
            raise PublicationError(f"committed mutable identity differs from held temp: {name}")
        if read_regular_at(directory_fd, name) != new_payload:
            raise PublicationError(f"committed mutable bytes differ: {name}")

    post_commit("mutable_post_replace_identity", verify_identity)
    post_commit("mutable_post_replace_directory_fsync", lambda: os.fsync(directory_fd))

    def close_held() -> None:
        nonlocal descriptor
        assert descriptor is not None
        os.close(descriptor)
        descriptor = None

    post_commit("mutable_post_replace_close", close_held)
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError as exc:
            warnings.append(_warning("mutable_emergency_close", exc))
        descriptor = None
    return {
        "state": "REPLACED_OLD_EXACT",
        "committed_by_this_call": True,
        "accepted": not warnings,
        "warnings": warnings,
    }


def _load_prepared_context(
    *,
    config_path: Path,
    prepared_directory: Path | str,
    production: bool,
    config_override: dict[str, Any] | None,
    authority_override: dict[str, Any] | None,
    repo_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, bytes]]:
    config, _config_payload, authority = _load_runtime_context(
        config_path,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    prepared_path, prepared = read_prepared_directory(
        prepared_directory, config, production=production
    )
    return config, authority, prepared_path, prepared


def validate_target_only(
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    prepared_directory: Path | str,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    authority_override: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    config, authority, prepared_path, prepared = _load_prepared_context(
        config_path=config_path,
        prepared_directory=prepared_directory,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and run_root_override is not None:
        raise PublicationError("test-only run-root override is forbidden in production")
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    lock_cleanup_warnings: list[dict[str, str]] = []
    with locked_directory(run_root, cleanup_warnings=lock_cleanup_warnings) as run_fd:
        mutable, immutable, attempt_lineage = classify_target(run_fd, prepared, config)
        validate_prepared_against_context(prepared, config, authority, attempt_lineage)
    if lock_cleanup_warnings:
        raise PublicationError(f"run-lock cleanup failed after validation: {lock_cleanup_warnings!r}")
    return {
        "status": "VALIDATED_NOT_PUBLISHED",
        "run_root": str(run_root),
        "prepared_directory": str(prepared_path),
        "attempt_lineage_state": "TWO_EXACT_COMMITTED_BUNDLES_REPLAYED",
        "mutable_preflight": mutable,
        "immutable_preflight": immutable,
        "authority": authority,
    }


def publish_prepared(
    *,
    config_path: Path = PRODUCTION_CONFIG_PATH,
    prepared_directory: Path | str,
    production: bool = True,
    config_override: dict[str, Any] | None = None,
    authority_override: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    run_root_override: Path | None = None,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    config, authority, prepared_path, prepared = _load_prepared_context(
        config_path=config_path,
        prepared_directory=prepared_directory,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and (run_root_override is not None or fault_injector is not None):
        raise PublicationError("test-only run-root/fault overrides are forbidden in production")
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    snapshots = snapshot_names(config)
    old_payloads = {
        mutable: prepared[snapshot] for mutable, snapshot in snapshots.items()
    }
    results: dict[str, dict[str, Any]] = {
        "GSE114002_V2_ATTEMPT_LINEAGE": {
            "state": "TWO_EXACT_COMMITTED_BUNDLES_REPLAYED",
            "committed_by_this_call": False,
            "accepted": True,
            "warnings": [],
        }
    }
    committed_names: list[str] = []
    lock_cleanup_warnings: list[dict[str, str]] = []

    with locked_directory(
        run_root,
        cleanup_warnings=lock_cleanup_warnings,
        fault_injector=fault_injector,
    ) as run_fd:
        mutable_preflight, immutable_preflight, attempt_lineage = classify_target(
            run_fd, prepared, config
        )
        validate_prepared_against_context(prepared, config, authority, attempt_lineage)
        preexisting_partial = (
            [mutable_preflight[name] for name in MUTABLE_NAMES]
            != ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"]
            or any(value == "EXISTING_EXACT" for value in immutable_preflight.values())
        )

        try:
            for name in immutable_names(config):
                outcome = publish_immutable_at(
                    run_fd, name, prepared[name], fault_injector=fault_injector
                )
                results[name] = outcome
                if outcome["committed_by_this_call"]:
                    committed_names.append(name)
                if not outcome["accepted"]:
                    temp_unlink_unresolved = any(
                        warning["point"] == "immutable_post_link_unlink"
                        for warning in outcome["warnings"]
                    )
                    return {
                        "status": (
                            "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
                            if temp_unlink_unresolved
                            else "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
                        ),
                        "event_id": config["event_id"],
                        "run_root": str(run_root),
                        "prepared_directory": str(prepared_path),
                        "warning_member": name,
                        "committed_members": committed_names,
                        "mutable_preflight": mutable_preflight,
                        "immutable_preflight": immutable_preflight,
                        "results": results,
                        "lock_cleanup_warnings": lock_cleanup_warnings,
                    }

            for name in MUTABLE_NAMES:
                outcome = replace_mutable_at(
                    run_fd,
                    name,
                    old_payloads[name],
                    prepared[name],
                    fault_injector=fault_injector,
                )
                results[name] = outcome
                if outcome["committed_by_this_call"]:
                    committed_names.append(name)
                if not outcome["accepted"]:
                    return {
                        "status": "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY",
                        "event_id": config["event_id"],
                        "run_root": str(run_root),
                        "prepared_directory": str(prepared_path),
                        "warning_member": name,
                        "committed_members": committed_names,
                        "mutable_preflight": mutable_preflight,
                        "immutable_preflight": immutable_preflight,
                        "results": results,
                        "lock_cleanup_warnings": lock_cleanup_warnings,
                    }

            final_mutable, final_immutable, final_lineage = classify_target(run_fd, prepared, config)
            validate_prepared_against_context(prepared, config, authority, final_lineage)
            if [final_mutable[name] for name in MUTABLE_NAMES] != [
                "NEW_EXACT",
                "NEW_EXACT",
                "NEW_EXACT",
            ]:
                raise PublicationError("final mutable state is not NEW/NEW/NEW")
            if any(value != "EXISTING_EXACT" for value in final_immutable.values()):
                raise PublicationError("final immutable closure is incomplete")
        except Exception as exc:
            if committed_names or preexisting_partial:
                return {
                    "status": "PARTIAL_STATE_REQUIRES_IDEMPOTENT_RETRY",
                    "event_id": config["event_id"],
                    "run_root": str(run_root),
                    "prepared_directory": str(prepared_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "committed_members": committed_names,
                    "preexisting_partial_state": preexisting_partial,
                    "mutable_preflight": mutable_preflight,
                    "immutable_preflight": immutable_preflight,
                    "results": results,
                    "lock_cleanup_warnings": lock_cleanup_warnings,
                }
            raise

    publication_status = "PUBLISHED_VERIFIED"
    if lock_cleanup_warnings:
        publication_status = "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    return {
        "status": publication_status,
        "event_id": config["event_id"],
        "run_root": str(run_root),
        "prepared_directory": str(prepared_path),
        "manifest_output_transition": "106_TO_122",
        "attempt_lineage_state": "TWO_EXACT_COMMITTED_BUNDLES_REPLAYED",
        "mutable_preflight": mutable_preflight,
        "immutable_preflight": immutable_preflight,
        "results": results,
        "committed_members": committed_names,
        "lock_cleanup_warnings": lock_cleanup_warnings,
        "sha256": {name: sha256(payload) for name, payload in prepared.items()},
        "authority": authority,
    }


def _json_result(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--prepared-directory", required=True)
    prepare_parser.add_argument("--recorded-at", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--prepared-directory", required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--prepared-directory", required=True)
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "prepare":
            result = prepare_runtime_sync(
                prepared_directory=arguments.prepared_directory,
                recorded_at=arguments.recorded_at,
            )
        elif arguments.command == "validate":
            result = validate_target_only(prepared_directory=arguments.prepared_directory)
        else:
            result = publish_prepared(prepared_directory=arguments.prepared_directory)
    except Exception as exc:
        _json_result(
            {
                "status": "FAIL_CLOSED_NOT_PUBLISHED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return 2
    _json_result(result)
    if result.get("status") in {
        "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY",
        "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION",
        "PARTIAL_STATE_REQUIRES_IDEMPOTENT_RETRY",
    }:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
