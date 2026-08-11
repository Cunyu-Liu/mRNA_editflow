#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed GSE200304 EVT-041 runtime sync.

The publisher revalidates the historical negative-producer and adjudicator
lifecycles, the current D2 descendant, the predecessor ledger integration, and
the exact terminal negative-evidence/adjudication bundles.  It registers those
already-published aggregate artifacts in place and appends one runtime event;
it never runs either producer, opens row-level evidence, rewrites a source
bundle, materializes a canonical record, or changes a scientific gate.

Production has no caller-selected repository, runtime root, or evidence source.
The only caller-selected path is a prepared artifact directory outside every
run root.  Any unresolved repository or implementation binding stops before a
repository, runtime, or evidence bundle is opened.
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


CONFIG_REPO_PATH = (
    "configs/route_a_v3_gse200304_dec019_blocked_adjudication_runtime_sync_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/gse200304_dec019_blocked_adjudication_runtime_sync.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_gse200304_dec019_blocked_adjudication_runtime_sync.py"
)
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
GIT_BINARY = "/usr/bin/git"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
RUNTIME_RUN_ID = "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5"
NEGATIVE_PACK_KEY = "negative_gate_pack"
ADJUDICATION_KEY = "blocked_adjudication"
SOURCE_KEYS = (NEGATIVE_PACK_KEY, ADJUDICATION_KEY)
SOURCE_MEMBER_OUTPUT_ORDER = (
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_CHECKPOINT_SPECIFIC_EXPOSURE_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json"),
    (NEGATIVE_PACK_KEY, "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json"),
    (NEGATIVE_PACK_KEY, "PUBLICATION_COMMIT.json"),
    (ADJUDICATION_KEY, "ADJUDICATION_REPORT.json"),
    (ADJUDICATION_KEY, "INPUT_EVIDENCE_AUDIT.json"),
    (ADJUDICATION_KEY, "PUBLICATION_COMMIT.json"),
    (ADJUDICATION_KEY, "SHA256SUMS"),
)


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


def read_optional_regular_at(
    directory_fd: int, name: str, *, require_single_link: bool = False
) -> bytes | None:
    try:
        return read_regular_at(
            directory_fd, name, require_single_link=require_single_link
        )
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


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise BindingError(f"{label} key closure drift")
    return value


def _binding_values_are_unknown(binding: Mapping[str, Any]) -> bool:
    return all(
        binding.get(key) == UNKNOWN
        for key in (
            "status",
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        )
    )


def _ledger_binding_values(authority: Mapping[str, Any]) -> list[Any]:
    ledger = authority["predecessor_ledger"]
    return [
        ledger.get("status"),
        ledger.get("commit"),
        *(item.get("sha256") for item in ledger.get("frozen_blobs", [])),
    ]


def _ledger_values_are_unknown(authority: Mapping[str, Any]) -> bool:
    values = _ledger_binding_values(authority)
    return len(values) == 6 and all(value == UNKNOWN for value in values)


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate the staging I-form or fully-bound B-form without external I/O."""

    _expect_keys(
        config,
        {
            "schema_version", "protocol_id", "contract_id", "phase_id", "event_id",
            "event_name", "sync_type", "implementation_binding",
            "repository_authority", "runtime_authority", "registered_evidence",
            "runtime", "successor_invariants", "access_and_materialization_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect_typed_exact(
        {key: config.get(key) for key in (
            "schema_version", "protocol_id", "contract_id", "phase_id", "event_id",
            "event_name", "sync_type",
        )},
        {
            "schema_version": "route_a_v3_gse200304_dec019_blocked_adjudication_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "event_id": "A1-EVT-041",
            "event_name": "GSE200304_DEC019_NEGATIVE_EVIDENCE_AND_BLOCKED_ADJUDICATION_SYNCED_GATE_UNCHANGED",
            "sync_type": "APPEND_ONLY_NEGATIVE_EVIDENCE_AND_BLOCKED_ADJUDICATION_REGISTRATION_NO_GATE_CHANGE",
        },
        label="config identity",
    )

    binding = _expect_keys(
        config.get("implementation_binding"),
        {
            "binding_scheme", "status", "implementation_commit",
            "implementation_script_path", "implementation_script_sha256",
            "implementation_test_path", "implementation_test_sha256",
            "compiled_core_sha256", "unknown_to_bound_scalar_paths",
        },
        label="implementation binding",
    )
    _expect_exact(binding["binding_scheme"], "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1", label="binding scheme")
    _expect_exact(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect_exact(
        binding["unknown_to_bound_scalar_paths"],
        [
            "implementation_binding.status",
            "implementation_binding.implementation_commit",
            "implementation_binding.implementation_script_sha256",
            "implementation_binding.implementation_test_sha256",
        ],
        label="implementation scalar allowlist",
    )
    if not _binding_values_are_unknown(binding):
        _expect_exact(binding["status"], "BOUND", label="implementation binding status")
        _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
        _expect_hex(binding["implementation_script_sha256"], HEX64, label="implementation script SHA")
        _expect_hex(binding["implementation_test_sha256"], HEX64, label="implementation test SHA")
    _expect_hex(binding["compiled_core_sha256"], HEX64, label="compiled core SHA")
    _expect_exact(binding["compiled_core_sha256"], compiled_core_sha256(config), label="compiled core")

    authority = _expect_keys(
        config.get("repository_authority"),
        {
            "production_repo_root", "branch", "base_commit",
            "current_pre_runtime_sync_head", "base_commit_expected_parent",
            "implementation_commit_expected_parent", "binding_commit_expected_parent",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths", "predecessor_ledger",
            "historical_runtime_publisher_lifecycle",
            "negative_producer_lifecycle", "adjudicator_lifecycle",
        },
        label="repository authority",
    )
    _expect_exact(authority["production_repo_root"], str(PRODUCTION_REPO_ROOT), label="repository root")
    _expect_exact(authority["branch"], "routea-v3-a1-20260810", label="repository branch")
    _expect_exact(authority["base_commit"], "4b544c7b8e95efc658c3c9336898a8c1898c4c94", label="repair base/runtime B2")
    _expect_exact(authority["current_pre_runtime_sync_head"], authority["base_commit"], label="repair pre-I3 head/runtime B2")
    _expect_exact(authority["base_commit_expected_parent"], "83fe7027a30acc83063262ad2c69c5cf80417ca5", label="repair base parent/runtime I2")
    _expect_exact(authority["implementation_commit_expected_parent"], "REPAIR_RUNTIME_BINDING_COMMIT_FROM_LIFECYCLE", label="repair I3 parent rule")
    _expect_exact(authority["binding_commit_expected_parent"], "IMPLEMENTATION_COMMIT_FROM_BINDING", label="runtime B parent rule")
    _expect_exact(authority["implementation_commit_exact_changed_paths"], sorted([CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]), label="runtime I paths")
    _expect_exact(authority["binding_commit_exact_changed_paths"], [CONFIG_REPO_PATH], label="runtime B paths")

    ledger = _expect_keys(
        authority["predecessor_ledger"],
        {
            "status", "commit", "expected_parent", "integration_id",
            "manifest_status", "commit_exact_changed_paths", "frozen_blobs",
            "unknown_to_bound_scalar_paths",
        },
        label="predecessor ledger",
    )
    ledger_paths = [
        "docs/execution/route_a_v3_a1_interim.yaml",
        "docs/execution/route_a_v3_registry_manifest.json",
        "scripts/route_a_v3/validate_a0_bundle.py",
        "tests/route_a_v3/test_a0_integrity_guards.py",
    ]
    _expect_exact(ledger["expected_parent"], "c278f29a18b7858c85686fcec3857a992fd07d5f", label="ledger expected parent")
    _expect_exact(ledger["integration_id"], "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_BUNDLE_V3", label="ledger integration ID")
    _expect_exact(ledger["manifest_status"], "A1_GSE200304_DEC019_V3_POST_ADJUDICATION_LEDGER_INTEGRATED", label="ledger manifest status")
    _expect_exact(ledger["commit_exact_changed_paths"], ledger_paths, label="ledger paths")
    _expect_exact(
        ledger["unknown_to_bound_scalar_paths"],
        [
            "repository_authority.base_commit",
            "repository_authority.current_pre_runtime_sync_head",
            "repository_authority.predecessor_ledger.status",
            "repository_authority.predecessor_ledger.commit",
            "repository_authority.predecessor_ledger.frozen_blobs[0].sha256",
            "repository_authority.predecessor_ledger.frozen_blobs[1].sha256",
            "repository_authority.predecessor_ledger.frozen_blobs[2].sha256",
            "repository_authority.predecessor_ledger.frozen_blobs[3].sha256",
        ],
        label="ledger binding scalar allowlist",
    )
    blobs = ledger["frozen_blobs"]
    if not isinstance(blobs, list) or [item.get("path") for item in blobs] != ledger_paths:
        raise BindingError("ledger frozen blob path closure drift")
    ledger_values = _ledger_binding_values(authority)
    if any(value == UNKNOWN for value in ledger_values):
        raise BindingError("predecessor ledger authority is not fully BOUND")
    _expect_exact(ledger["status"], "BOUND", label="ledger status")
    ledger_commit = _expect_hex(ledger["commit"], HEX40, label="ledger commit")
    _expect_exact(ledger_commit, "f465dd03ae792b98c0604b1d225cd2df37d28f9e", label="frozen predecessor ledger commit")
    for item in blobs:
        _expect_hex(item["sha256"], HEX64, label=f"ledger blob {item['path']}")

    publisher = _expect_keys(
        authority["historical_runtime_publisher_lifecycle"],
        {
            "predecessor_ledger_commit", "initial_implementation_commit",
            "initial_binding_commit", "repair_implementation_commit",
            "repair_binding_commit", "config_path", "script_path", "test_path",
            "initial_implementation_blobs", "initial_binding_blobs",
            "repair_implementation_blobs", "repair_binding_blobs",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
        },
        label="historical runtime publisher lifecycle",
    )
    _expect_typed_exact(
        {
            key: publisher[key]
            for key in (
                "predecessor_ledger_commit", "initial_implementation_commit",
                "initial_binding_commit", "repair_implementation_commit",
                "repair_binding_commit", "config_path", "script_path", "test_path",
            )
        },
        {
            "predecessor_ledger_commit": "f465dd03ae792b98c0604b1d225cd2df37d28f9e",
            "initial_implementation_commit": "9acdefc4a410e03827532b359ec245d8a6cb76df",
            "initial_binding_commit": "1dc7da6300dfd192d69656fe63d582fe8f71da48",
            "repair_implementation_commit": "83fe7027a30acc83063262ad2c69c5cf80417ca5",
            "repair_binding_commit": "4b544c7b8e95efc658c3c9336898a8c1898c4c94",
            "config_path": CONFIG_REPO_PATH,
            "script_path": SCRIPT_REPO_PATH,
            "test_path": TEST_REPO_PATH,
        },
        label="historical runtime publisher identity",
    )
    for key, expected in {
        "initial_implementation_blobs": {
            "config_sha256": "7fd790fd4d841c771a9324dfb665555d5a93e8d2d66011d4751e3802621f0fd7",
            "script_sha256": "fcbe15c4ea0fd7ef5a049c71f388b242757fa59e17f15f9ea22f9d5def353530",
            "test_sha256": "93633d61cc9bd10d5003f2b29d4ab1d15f1a6b2ab8f3674a0c2617351c8bed64",
        },
        "initial_binding_blobs": {
            "config_sha256": "0dc02a9122686d08ca4eee071f4e966b5ae51c9afc6ad29d3bf530dd9ae74921",
            "script_sha256": "fcbe15c4ea0fd7ef5a049c71f388b242757fa59e17f15f9ea22f9d5def353530",
            "test_sha256": "93633d61cc9bd10d5003f2b29d4ab1d15f1a6b2ab8f3674a0c2617351c8bed64",
        },
        "repair_implementation_blobs": {
            "config_sha256": "9a4164d3e3e7a99970ceb237992d076fc648932f78497584dc15c1cfcd3fbc18",
            "script_sha256": "52e17db1f647738f5677c97e06c056d00e07316c786796c81fa9e77e35842392",
            "test_sha256": "91b85b8630c13a68c1dfa37987c8c3d643e4cad7fbc82c3e739caa2741d089d4",
        },
        "repair_binding_blobs": {
            "config_sha256": "895990bd015b64fb5ae65fe75ea67460d555a30e194c01600d960cd5270e8cda",
            "script_sha256": "52e17db1f647738f5677c97e06c056d00e07316c786796c81fa9e77e35842392",
            "test_sha256": "91b85b8630c13a68c1dfa37987c8c3d643e4cad7fbc82c3e739caa2741d089d4",
        },
    }.items():
        _expect_typed_exact(publisher[key], expected, label=f"historical runtime {key}")
    _expect_exact(
        publisher["implementation_commit_exact_changed_paths"],
        authority["implementation_commit_exact_changed_paths"],
        label="historical runtime I1 paths",
    )
    _expect_exact(
        publisher["binding_commit_exact_changed_paths"],
        authority["binding_commit_exact_changed_paths"],
        label="historical runtime B1 paths",
    )

    negative = _expect_keys(
        authority["negative_producer_lifecycle"],
        {
            "initial_implementation_commit", "initial_binding_commit",
            "nfs_implementation_commit", "nfs_binding_commit",
            "current_descriptor_commit", "config_path", "script_path", "test_path",
            "initial_implementation_blobs", "initial_binding_blobs",
            "nfs_implementation_blobs", "nfs_binding_and_current_blobs",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
        },
        label="negative lifecycle",
    )
    _expect_typed_exact(
        {key: negative[key] for key in (
            "initial_implementation_commit", "initial_binding_commit",
            "nfs_implementation_commit", "nfs_binding_commit", "current_descriptor_commit",
        )},
        {
            "initial_implementation_commit": "01f3f818937c97d9804b94413f54d3e654e35120",
            "initial_binding_commit": "a677454ec78ad5df4a5880444b0764d42676025a",
            "nfs_implementation_commit": "8da6abaeddfadeca5542b997e7c0ba6501c1f1f7",
            "nfs_binding_commit": "981f001d6290cbc1b9b48a55d8e51a963e45d785",
            "current_descriptor_commit": "c278f29a18b7858c85686fcec3857a992fd07d5f",
        },
        label="negative lifecycle commits",
    )
    expected_negative_blob_sets = {
        "initial_implementation_blobs": {
            "config_sha256": "cc0ee5e4111966e40cf2c2c8f1d1b7154407854cf505dd44edb82a8e9a047434",
            "script_sha256": "8559903eee8fd2d6a85b32db4a88fa39666ee8a7b315f059e08bfe28c8265e27",
            "test_sha256": "f4346a36ab5a388e366f4ee23e999dee6752909052a71da4ac0ea9038969c47f",
        },
        "initial_binding_blobs": {
            "config_sha256": "2693bea877e1b2eccb97d2b75ee0f3b7d40a47932526b12b9cd92067d636bc07",
            "script_sha256": "8559903eee8fd2d6a85b32db4a88fa39666ee8a7b315f059e08bfe28c8265e27",
            "test_sha256": "f4346a36ab5a388e366f4ee23e999dee6752909052a71da4ac0ea9038969c47f",
        },
        "nfs_implementation_blobs": {
            "config_sha256": "b8b8d0280444c7d2bfe7fd1a6ab4ac8c5e7f2fe129eaab3e39cb7d069363b34f",
            "script_sha256": "3716ccf6492b067c374fde38f58d7b46e878dec7c58192b94030e05161e33205",
            "test_sha256": "5523d3a1b5216963bd3793ba9ec3f8cf15d9a01867192ddf1eae32ac0e327948",
        },
        "nfs_binding_and_current_blobs": {
            "config_sha256": "fea1c56d21dc848b535c876b31799eb6ccca48ecf9c4d8a58a7dbc7f7187297e",
            "script_sha256": "3716ccf6492b067c374fde38f58d7b46e878dec7c58192b94030e05161e33205",
            "test_sha256": "5523d3a1b5216963bd3793ba9ec3f8cf15d9a01867192ddf1eae32ac0e327948",
        },
    }
    for key, expected in expected_negative_blob_sets.items():
        _expect_typed_exact(negative[key], expected, label=f"negative {key}")
    negative_paths = sorted([negative["config_path"], negative["script_path"], negative["test_path"]])
    _expect_exact(negative["implementation_commit_exact_changed_paths"], negative_paths, label="negative I paths")
    _expect_exact(negative["binding_commit_exact_changed_paths"], [negative["config_path"]], label="negative B paths")

    adjudicator = _expect_keys(
        authority["adjudicator_lifecycle"],
        {
            "implementation_commit", "binding_commit", "descriptor_commit",
            "config_path", "script_path", "test_path", "implementation_blobs",
            "binding_blobs", "descriptor_blobs", "implementation_commit_exact_changed_paths",
            "binding_and_descriptor_commit_exact_changed_paths",
        },
        label="adjudicator lifecycle",
    )
    _expect_typed_exact(
        {key: adjudicator[key] for key in ("implementation_commit", "binding_commit", "descriptor_commit")},
        {
            "implementation_commit": "6d103877bbfb8e1196bfc22890bb239dcb87c3c8",
            "binding_commit": "6c42d8e1d75f70906afb7cde5704669b2c8ab6f7",
            "descriptor_commit": "c278f29a18b7858c85686fcec3857a992fd07d5f",
        },
        label="adjudicator lifecycle commits",
    )
    expected_adjudicator_blobs = {
        "implementation_blobs": {
            "config_bytes": 20553,
            "config_sha256": "b1ebe555dc7e42c5c63cccc05ee58454da0bea95712e29a2eb4f607c13ecd57d",
            "script_sha256": "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe",
            "test_sha256": "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db",
        },
        "binding_blobs": {
            "config_bytes": 20646,
            "config_sha256": "675a48355d89f64f29abc5fa23df7bbd854816ffb867db24ed4554071c200091",
            "script_sha256": "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe",
            "test_sha256": "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db",
        },
        "descriptor_blobs": {
            "config_bytes": 22008,
            "config_sha256": "88fa21a08df60935f3d2d1bf44c6573889c22c110021146acf241fd92d6b5a13",
            "script_sha256": "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe",
            "test_sha256": "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db",
            "science_core_sha256": "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170",
            "descriptor_set_sha256": "14223d0193e4b3a4a3c1d98a5894849dd429e6eed021ff98e6697e73ac286a40",
        },
    }
    for key, expected in expected_adjudicator_blobs.items():
        _expect_typed_exact(adjudicator[key], expected, label=f"adjudicator {key}")
    adjudicator_paths = sorted([adjudicator["config_path"], adjudicator["script_path"], adjudicator["test_path"]])
    _expect_exact(adjudicator["implementation_commit_exact_changed_paths"], adjudicator_paths, label="adjudicator I paths")
    _expect_exact(adjudicator["binding_and_descriptor_commit_exact_changed_paths"], [adjudicator["config_path"]], label="adjudicator B/D2 paths")

    _expect_typed_exact(
        config.get("runtime_authority"),
        {"historical_outer_runtime_authority": {
            "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
            "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        }},
        label="outer runtime authority",
    )

    registered = _expect_keys(config.get("registered_evidence"), set(SOURCE_KEYS), label="registered evidence")
    _expect_typed_exact(
        list(registered), list(SOURCE_KEYS), label="registered evidence source order"
    )
    negative_pack = _expect_keys(
        registered[NEGATIVE_PACK_KEY],
        {"dataset_id", "absolute_directory", "protocol_id", "terminal_record_type", "publication_mode", "gate_payload_set_sha256", "final_output_target_sha256", "members"},
        label="negative pack",
    )
    _expect_exact(negative_pack["dataset_id"], "GSE200304", label="negative dataset")
    _expect_exact(negative_pack["absolute_directory"], f"{config['runtime']['run_root']}/GSE200304_DEC019_NEGATIVE_GATE_PACK_V1", label="negative directory")
    _expect_exact(negative_pack["terminal_record_type"], "GSE200304_DEC019_NEGATIVE_GATE_PACK_PUBLICATION_COMMIT_V1", label="negative marker type")
    _expect_exact(negative_pack["publication_mode"], "ATOMIC_MKDIR_TERMINAL_COMMIT_MARKER_V1", label="negative publication mode")
    _expect_exact(negative_pack["gate_payload_set_sha256"], "a760c37d313c047e791692efd1adfb1e5c76d63c2b5c85979da5e42ba04ce2db", label="negative payload set")
    _expect_exact(negative_pack["final_output_target_sha256"], "d25bbeea21f82db4494c6253bf3d4b8388d809e74141c6129f3118adf35be5df", label="negative target")
    expected_negative = {
        "GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_GATE.json": ("A1_GSE200304_DEC019_BIOLOGICAL_GROUP_AUTHORITY_NEGATIVE_GATE", 3921, "2db95ec41d5e76a77d17104076c5823f5cc1f8646260964e92166f0faf440950", "BLOCKED"),
        "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json": ("A1_GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NEGATIVE_GATE", 4022, "a2b4dd52f0fbe4cd31324d4b760b9fee104f719d67d4f6834b6c1ea3adffaf9e", "BLOCKED"),
        "GSE200304_DEC019_CHECKPOINT_SPECIFIC_EXPOSURE_GATE.json": ("A1_GSE200304_DEC019_CHECKPOINT_SPECIFIC_EXPOSURE_NEGATIVE_GATE", 3997, "33b659a86eb8058adad922649b3c89a78ad65a61014826cd3ab28f1c4a6214f9", UNKNOWN),
        "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json": ("A1_GSE200304_DEC019_LICENSE_RIGHTS_NEGATIVE_GATE", 3919, "7556db47e8b3ec9cdc0a7d795161cf4ded8679d0c04644005a475753c482aa28", UNKNOWN),
        "GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_GATE.json": ("A1_GSE200304_DEC019_OUTCOME_BLIND_SPLIT_LEAKAGE_NEGATIVE_GATE", 4004, "b3be463eb502b5b8a2b171ea763cffa8eba1449bf2c380aa3d16435733e35821", "NOT_RUN"),
        "GSE200304_DEC019_PREFROZEN_POWER_PRECISION_GATE.json": ("A1_GSE200304_DEC019_PREFROZEN_POWER_PRECISION_NEGATIVE_GATE", 3908, "817165a2e9ea2e01efae2374a606334375b9ec3afafae665fefacf0c6779fc95", "NOT_RUN"),
        "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json": ("A1_GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_NEGATIVE_GATE", 3977, "235683969801d375a59a0bde56af2a448afd2a56cd5224cbede1320c2c15026b", "BLOCKED"),
        "PUBLICATION_COMMIT.json": ("A1_GSE200304_DEC019_NEGATIVE_GATE_PACK_TERMINAL_COMMIT", 1256, "bf2cad9cfdc3b6dfc537bc0bf302ad79e332c5cf5d341e8dfd7fa64675b423c4", "COMMITTED"),
    }
    members = negative_pack["members"]
    if not isinstance(members, list) or len(members) != 8:
        raise BindingError("negative pack is not exact8")
    for item in members:
        _expect_keys(item, {"name", "artifact_type", "bytes", "sha256", "terminal_status"}, label="negative member")
    _expect_typed_exact(
        [item["name"] for item in members],
        list(expected_negative),
        label="negative exact8 member order",
    )
    observed_negative = {
        item["name"]: (
            item["artifact_type"], item["bytes"], item["sha256"],
            item["terminal_status"],
        )
        for item in members
    }
    _expect_typed_exact(observed_negative, expected_negative, label="negative exact8 map")

    blocked = _expect_keys(
        registered[ADJUDICATION_KEY],
        {
            "dataset_id", "absolute_directory", "output_id", "terminal_record_type", "scientific_status",
            "final_output_target_sha256", "members", "input_count", "unresolved_blockers",
            "ordinary_study_contribution", "a1_study_contribution", "true_a2_study_contribution",
            "canonical_record_count", "qualified", "canonical_materialization_allowed",
            "training_allowed", "model_selection_allowed", "next_phase_authorized",
        },
        label="blocked adjudication",
    )
    _expect_exact(blocked["absolute_directory"], "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3", label="adjudication directory")
    _expect_exact(blocked["output_id"], ledger["integration_id"], label="adjudication output ID")
    _expect_exact(blocked["scientific_status"], "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE", label="adjudication status")
    _expect_exact(blocked["final_output_target_sha256"], "62074e5013e65ef3892bcfb91e820f74d414e636c3b3f9a5cef3c93e6d7c28be", label="adjudication target")
    expected_adjudication = {
        "ADJUDICATION_REPORT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_REPORT", 2486, "62d2391bc61533f0374195605ba2a1e4ba3385f997b233087632f53901ae2de3"),
        "INPUT_EVIDENCE_AUDIT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_INPUT_AUDIT", 3005, "d84763040507e34f9c5913075ee306b4432a75389560bdeb85c9b6ba088809e6"),
        "PUBLICATION_COMMIT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_TERMINAL_COMMIT", 1055, "6fb7c07c493ace456d4c4918fdc270986796c32e73e23f6901e34352d4bdf310"),
        "SHA256SUMS": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_SHA256SUMS", 183, "f856f14508db876aec3438a069d2aa0aacc92989153a33a05d54b59b4d256477"),
    }
    adjudication_members = blocked["members"]
    if not isinstance(adjudication_members, list) or len(adjudication_members) != 4:
        raise BindingError("blocked adjudication is not exact4")
    for item in adjudication_members:
        _expect_keys(item, {"name", "artifact_type", "bytes", "sha256"}, label="adjudication member")
    _expect_typed_exact(
        [item["name"] for item in adjudication_members],
        list(expected_adjudication),
        label="adjudication exact4 member order",
    )
    observed_adjudication = {
        item["name"]: (item["artifact_type"], item["bytes"], item["sha256"])
        for item in adjudication_members
    }
    _expect_typed_exact(observed_adjudication, expected_adjudication, label="adjudication exact4 map")
    _expect_int(blocked["input_count"], 8, label="adjudication input count")
    _expect_exact(
        blocked["unresolved_blockers"],
        [
            "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS",
            "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_PASS",
            "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
            "LICENSE_RIGHTS_NOT_PASS",
            "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
            "PREFROZEN_POWER_PRECISION_NOT_PASS",
            "ROW_REPLICATE_OR_VALID_SE_NOT_PASS",
        ],
        label="adjudication exact seven blockers",
    )
    for key in ("ordinary_study_contribution", "a1_study_contribution", "true_a2_study_contribution", "canonical_record_count"):
        _expect_int(blocked[key], 0, label=f"adjudication {key}")
    for key in ("qualified", "canonical_materialization_allowed", "training_allowed", "model_selection_allowed", "next_phase_authorized"):
        _expect_typed_exact(blocked[key], False, label=f"adjudication {key}")

    runtime = _expect_keys(
        config.get("runtime"),
        {
            "run_root", "allowed_prepared_root", "predecessor_event_id", "predecessor_event_count",
            "predecessor_manifest_output_count", "successor_event_id", "successor_event_count",
            "successor_manifest_output_count", "mutable_publish_order", "allowed_mutable_states",
            "predecessor_mutables", "predecessor_tail_event", "sync_name", "output_delta_count",
            "forbidden_runtime_path_tokens",
        },
        label="runtime",
    )
    _expect_exact(runtime["run_root"], "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", label="run root")
    _expect_exact(runtime["allowed_prepared_root"], "/mnt/cunyuliu/mrna_xeditflow_routea_v3/staging/runtime_sync/A1", label="prepared root")
    for field, expected in (("predecessor_event_count", 40), ("successor_event_count", 41), ("predecessor_manifest_output_count", 127), ("successor_manifest_output_count", 143), ("output_delta_count", 16)):
        _expect_int(runtime[field], expected, label=field)
    _expect_exact(runtime["predecessor_event_id"], "A1-EVT-040", label="predecessor event")
    _expect_exact(runtime["successor_event_id"], "A1-EVT-041", label="successor event")
    _expect_exact(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable order")
    _expect_exact(runtime["allowed_mutable_states"], [["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"], ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"], ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"], ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"]], label="recovery states")
    _expect_typed_exact(
        runtime["predecessor_mutables"],
        {
            "STATUS.json": {"bytes": 21540, "sha256": "df0d353eb0fb30998df29024a802a1d89c202f75326ff6acec1402a2b0c5f408", "snapshot_name": "STATUS_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1.json"},
            "RUN_MANIFEST.json": {"bytes": 52144, "sha256": "d7e81b98d5708e006fbc345ff71161f42b30725d8fc2ff5a13f047edbc4fe8cd", "snapshot_name": "RUN_MANIFEST_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1.json"},
            "EVENT_LOG.jsonl": {"bytes": 54532, "sha256": "b8da2bbb22d14b0693fb8e8e9e3540cb4d5e72b5b4fa02fc5906679a8d4c8516", "snapshot_name": "EVENT_LOG_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1.jsonl"},
        },
        label="predecessor mutables",
    )
    _expect_typed_exact(runtime["predecessor_tail_event"], {"event_id": "A1-EVT-040", "bytes": 3130, "sha256": "90a6b6704114a21c1eec3cf8996557ade2c1f7e94122771b8750d77742a1347c", "training_started_key_present": True, "training_started": False}, label="EVT040 tail")
    _expect_exact(runtime["sync_name"], "A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1.json", label="sync name")
    _expect_exact(runtime["forbidden_runtime_path_tokens"], ["GSE246381", "/restricted/", "/sealed/", "/sealed_external/", "FASTQ", "raw_replay"], label="forbidden tokens")

    _expect_typed_exact(
        config.get("successor_invariants"),
        {
            "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "scientific_claim_status": "NOT_ESTABLISHED",
            "qualified_independent_ordinary_studies": 0, "qualified_a1_studies": 0,
            "qualified_a2_dense_studies": 0, "ordinary_study_contribution": 0,
            "a1_study_contribution": 0, "true_a2_study_contribution": 0,
            "canonical_intervention_record_count": 0, "qualified": False,
            "canonical_materialization_allowed": False, "training_started": False,
            "training_allowed": False, "training_authorized": False,
            "model_selection_allowed": False, "next_phase_authorized": False,
        },
        label="successor gate truth",
    )
    boundary = _expect_keys(
        config.get("access_and_materialization_boundary"),
        {
            "negative_gate_artifact_read_count", "blocked_adjudication_artifact_read_count",
            "adjudicator_execution_count", "adjudication_input_count", "adjudication_unresolved_blocker_count",
            "runtime_evidence_payload_read_count", "real_row_level_data_opened", "raw_reads_or_alignments_opened",
            "raw_fastq_body_read_count", "restricted_or_sealed_path_accessed", "restricted_or_sealed_payload_contact",
            "gse246381_contact", "runtime_mutable_predecessor_read_count", "predecessor_snapshot_count",
            "runtime_sync_record_count", "registered_in_place_artifact_count", "new_runtime_output_count",
            "canonical_read_count", "canonical_write_count", "qualifier_execution_count", "training_run_count",
            "model_selection_run_count", "gpu_work_started",
        },
        label="access boundary",
    )
    expected_counts = {
        "negative_gate_artifact_read_count": 8, "blocked_adjudication_artifact_read_count": 4,
        "adjudicator_execution_count": 0, "adjudication_input_count": 8,
        "adjudication_unresolved_blocker_count": 7, "runtime_evidence_payload_read_count": 0,
        "runtime_mutable_predecessor_read_count": 3, "predecessor_snapshot_count": 3,
        "runtime_sync_record_count": 1, "registered_in_place_artifact_count": 12,
        "new_runtime_output_count": 16, "canonical_read_count": 0, "canonical_write_count": 0,
        "qualifier_execution_count": 0, "training_run_count": 0, "model_selection_run_count": 0,
        "raw_fastq_body_read_count": 0,
    }
    for key, expected in expected_counts.items():
        _expect_int(boundary[key], expected, label=f"access {key}")
    for key in ("real_row_level_data_opened", "raw_reads_or_alignments_opened", "restricted_or_sealed_path_accessed", "restricted_or_sealed_payload_contact", "gse246381_contact", "gpu_work_started"):
        _expect_typed_exact(boundary[key], False, label=f"access {key}")
    _expect_typed_exact(
        config.get("publication_policy"),
        {
            "immutable_no_overwrite": True,
            "registered_sources_never_copied_or_rewritten": True,
            "registered_sources_revalidated_before_mutables": True,
            "mutable_order_recoverable": True,
            "event_is_last_mutable_commit": True,
            "post_commit_error_truth": "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY",
            "immutable_temp_unlink_failure_truth": "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION",
            "sync_references_successor_hashes": False,
            "successors_reference_sync_hash": True,
            "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
        },
        label="publication policy",
    )


def validate_bound_config(config: dict[str, Any]) -> None:
    """Require the frozen history and the repair runtime I3/config-only-B3."""

    validate_static_config(config)
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    if _binding_values_are_unknown(binding):
        raise BindingError("runtime-sync implementation is not BOUND")
    if _ledger_values_are_unknown(authority):
        raise BindingError("predecessor ledger authority is not BOUND")
    implementation = _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
    base = _expect_hex(authority["base_commit"], HEX40, label="base commit")
    if implementation == base:
        raise BindingError("runtime-sync repair implementation did not advance from B2 base")


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


def _expect_parent(repo_root: Path, child: str, parent: str, *, label: str) -> None:
    actual = _run_git(repo_root, "rev-parse", f"{child}^").decode().strip()
    _expect_exact(actual, parent, label=label)


def _expect_ancestor(repo_root: Path, ancestor: str, descendant: str, *, label: str) -> None:
    _run_git(repo_root, "merge-base", "--is-ancestor", ancestor, descendant)
    _expect_exact(_run_git(repo_root, "rev-parse", ancestor).decode().strip(), ancestor, label=f"{label} ancestor identity")
    _expect_exact(_run_git(repo_root, "rev-parse", descendant).decode().strip(), descendant, label=f"{label} descendant identity")


def _verify_three_blobs(
    repo_root: Path,
    *,
    commit: str,
    config_path: str,
    script_path: str,
    test_path: str,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for path, digest in (
        (config_path, expected["config_sha256"]),
        (script_path, expected["script_sha256"]),
        (test_path, expected["test_sha256"]),
    ):
        payload = _git_blob(repo_root, commit, path)
        if sha256(payload) != digest:
            raise AuthorityError(f"{label} blob drift: {path}")
        if path == config_path and "config_bytes" in expected and len(payload) != expected["config_bytes"]:
            raise AuthorityError(f"{label} config byte-count drift")


def audit_repo_authority(
    repo_root: Path, config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove historical producer/adjudicator lifecycles plus runtime I/B.

    This deliberately inspects frozen Git objects.  It never imports or invokes
    the current negative producer's ``validate_production_authority`` entrypoint,
    whose current-head assumptions no longer describe the post-D2 repository.
    """

    if lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT or repo_root.is_symlink() or not repo_root.is_dir():
        raise AuthorityError("production repository root drift")
    validate_bound_config(config)
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    ledger = authority["predecessor_ledger"]
    publisher = authority["historical_runtime_publisher_lifecycle"]
    negative = authority["negative_producer_lifecycle"]
    adjudicator = authority["adjudicator_lifecycle"]
    implementation = binding["implementation_commit"]
    base = authority["base_commit"]
    branch = authority["branch"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    origin = _run_git(repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}").decode().strip()
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), branch, label="branch")
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip(), f"origin/{branch}", label="upstream")
    _expect_exact(_run_git(repo_root, "rev-parse", "@{upstream}").decode().strip(), head, label="upstream head")
    _expect_exact(origin, head, label="origin head")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AuthorityError("worktree or index is dirty")

    # Runtime publisher history and repair:
    # ledger -> I1 -> B1 -> I2 -> B2/base -> I3 -> config-only B3/current.
    ledger_commit = ledger["commit"]
    initial_implementation = publisher["initial_implementation_commit"]
    initial_binding = publisher["initial_binding_commit"]
    repair_implementation = publisher["repair_implementation_commit"]
    repair_binding = publisher["repair_binding_commit"]
    _expect_exact(publisher["predecessor_ledger_commit"], ledger_commit, label="runtime I1 ledger")
    _expect_exact(base, repair_binding, label="repair base/runtime B2")
    _expect_exact(authority["current_pre_runtime_sync_head"], base, label="repair pre-I3 head")
    _expect_parent(repo_root, head, implementation, label="runtime B3 parent")
    _expect_parent(repo_root, implementation, base, label="runtime I3 parent")
    _expect_parent(repo_root, base, repair_implementation, label="runtime B2 parent")
    _expect_parent(repo_root, repair_implementation, initial_binding, label="runtime I2 parent")
    _expect_parent(repo_root, initial_binding, initial_implementation, label="runtime B1 parent")
    _expect_parent(repo_root, initial_implementation, ledger_commit, label="runtime I1 parent")
    _expect_parent(repo_root, ledger_commit, ledger["expected_parent"], label="ledger/D2 parent")
    _expect_exact(_paths_changed_by_commit(repo_root, ledger_commit), ledger["commit_exact_changed_paths"], label="ledger changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, initial_implementation), publisher["implementation_commit_exact_changed_paths"], label="runtime I1 changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, initial_binding), publisher["binding_commit_exact_changed_paths"], label="runtime B1 changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, repair_implementation), publisher["implementation_commit_exact_changed_paths"], label="runtime I2 changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, repair_binding), publisher["binding_commit_exact_changed_paths"], label="runtime B2 changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, implementation), authority["implementation_commit_exact_changed_paths"], label="runtime I3 changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, head), authority["binding_commit_exact_changed_paths"], label="runtime B3 changed paths")
    for commit, blob_key, label in (
        (initial_implementation, "initial_implementation_blobs", "runtime I1"),
        (initial_binding, "initial_binding_blobs", "runtime B1"),
        (repair_implementation, "repair_implementation_blobs", "runtime I2"),
        (repair_binding, "repair_binding_blobs", "runtime B2"),
    ):
        _verify_three_blobs(
            repo_root,
            commit=commit,
            config_path=publisher["config_path"],
            script_path=publisher["script_path"],
            test_path=publisher["test_path"],
            expected=publisher[blob_key],
            label=label,
        )
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("runtime B config blob drift")
    i_config = load_json(
        _git_blob(repo_root, implementation, CONFIG_REPO_PATH),
        label="runtime I config",
    )
    try:
        _expect_typed_exact(
            i_config,
            expected_unknown_i_config(config),
            label="runtime I/B config transition",
        )
        _expect_typed_exact(
            compiled_core_projection(i_config),
            compiled_core_projection(config),
            label="runtime I/B compiled core transition",
        )
    except RuntimeSyncError as exc:
        raise AuthorityError("runtime I/B config transition drift") from exc
    for path, digest in ((SCRIPT_REPO_PATH, binding["implementation_script_sha256"]), (TEST_REPO_PATH, binding["implementation_test_sha256"])):
        if sha256(_git_blob(repo_root, implementation, path)) != digest or sha256(_git_blob(repo_root, head, path)) != digest or sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"runtime implementation blob drift: {path}")
    for item in ledger["frozen_blobs"]:
        path, digest = item["path"], item["sha256"]
        if any(
            sha256(_git_blob(repo_root, commit, path)) != digest
            for commit in (
                ledger_commit, initial_implementation, initial_binding,
                repair_implementation, repair_binding, implementation, head,
            )
        ) or sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"predecessor ledger blob drift: {path}")

    # Historical negative producer: initial I/B -> NFS-safe I/B -> direct-child D2.
    initial_i = negative["initial_implementation_commit"]
    initial_b = negative["initial_binding_commit"]
    nfs_i = negative["nfs_implementation_commit"]
    nfs_b = negative["nfs_binding_commit"]
    d2 = negative["current_descriptor_commit"]
    _expect_parent(repo_root, initial_b, initial_i, label="negative initial B parent")
    _expect_parent(repo_root, nfs_i, initial_b, label="negative NFS I parent")
    _expect_parent(repo_root, nfs_b, nfs_i, label="negative NFS B parent")
    _expect_parent(repo_root, d2, nfs_b, label="negative D2 parent")
    _expect_ancestor(repo_root, d2, ledger_commit, label="D2 to ledger")
    _expect_ancestor(repo_root, ledger_commit, base, label="ledger to repair base")
    for commit, expected_paths, label in (
        (initial_i, negative["implementation_commit_exact_changed_paths"], "negative initial I paths"),
        (initial_b, negative["binding_commit_exact_changed_paths"], "negative initial B paths"),
        (nfs_i, negative["implementation_commit_exact_changed_paths"], "negative NFS I paths"),
        (nfs_b, negative["binding_commit_exact_changed_paths"], "negative NFS B paths"),
    ):
        _expect_exact(_paths_changed_by_commit(repo_root, commit), expected_paths, label=label)
    for commit, blob_key, label in (
        (initial_i, "initial_implementation_blobs", "negative initial I"),
        (initial_b, "initial_binding_blobs", "negative initial B"),
        (nfs_i, "nfs_implementation_blobs", "negative NFS I"),
        (nfs_b, "nfs_binding_and_current_blobs", "negative NFS B"),
        (d2, "nfs_binding_and_current_blobs", "negative current D2"),
        (head, "nfs_binding_and_current_blobs", "negative current descendant"),
    ):
        _verify_three_blobs(
            repo_root,
            commit=commit,
            config_path=negative["config_path"],
            script_path=negative["script_path"],
            test_path=negative["test_path"],
            expected=negative[blob_key],
            label=label,
        )

    # Historical/current adjudicator I4 -> B4, with B4 an ancestor of D2.
    adjudicator_i = adjudicator["implementation_commit"]
    adjudicator_b = adjudicator["binding_commit"]
    adjudicator_d2 = adjudicator["descriptor_commit"]
    _expect_parent(repo_root, adjudicator_b, adjudicator_i, label="adjudicator B4 parent")
    _expect_ancestor(repo_root, adjudicator_b, adjudicator_d2, label="adjudicator B4 to D2")
    _expect_exact(_paths_changed_by_commit(repo_root, adjudicator_i), adjudicator["implementation_commit_exact_changed_paths"], label="adjudicator I4 paths")
    _expect_exact(_paths_changed_by_commit(repo_root, adjudicator_b), adjudicator["binding_and_descriptor_commit_exact_changed_paths"], label="adjudicator B4 paths")
    _expect_exact(_paths_changed_by_commit(repo_root, adjudicator_d2), adjudicator["binding_and_descriptor_commit_exact_changed_paths"], label="adjudicator D2 paths")
    for commit, blob_key, label in (
        (adjudicator_i, "implementation_blobs", "adjudicator I4"),
        (adjudicator_b, "binding_blobs", "adjudicator B4"),
        (adjudicator_d2, "descriptor_blobs", "adjudicator D2"),
        (head, "descriptor_blobs", "adjudicator current descendant"),
    ):
        _verify_three_blobs(
            repo_root,
            commit=commit,
            config_path=adjudicator["config_path"],
            script_path=adjudicator["script_path"],
            test_path=adjudicator["test_path"],
            expected=adjudicator[blob_key],
            label=label,
        )

    return {
        "status": "PASS_STRICT_LINEAR_DAG_HISTORICAL_RUNTIME_I1_B1_I2_B2_AND_REPAIR_I3_TO_CONFIG_ONLY_B3",
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin,
        "config_sha256": sha256(config_payload),
        "base_commit": base,
        "implementation_commit": implementation,
        "predecessor_ledger_commit": ledger["commit"],
        "initial_runtime_implementation_commit": initial_implementation,
        "initial_runtime_binding_commit": initial_binding,
        "repair_runtime_implementation_commit": repair_implementation,
        "repair_runtime_binding_commit": repair_binding,
        "negative_nfs_binding_commit": nfs_b,
        "adjudicator_descriptor_commit": adjudicator_d2,
        "historical_negative_blob_check_count": 18,
        "adjudicator_blob_check_count": 12,
        "historical_runtime_blob_check_count": 12,
        "ledger_blob_check_count": 4,
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value) is None:
        raise RuntimeSyncError("recorded_at must be RFC3339 +08:00")
    try:
        current, predecessor = datetime.fromisoformat(value), datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("invalid recorded_at") from exc
    if current.utcoffset() != timedelta(hours=8) or current <= predecessor:
        raise RuntimeSyncError("recorded_at must follow EVT-040")
    if current.astimezone(timezone.utc).date().isoformat() != "2026-08-11":
        raise RuntimeSyncError("recorded_at outside EVT-041 window")


def open_absolute_directory_nofollow(path: Path) -> int:
    """Walk an absolute directory root-to-leaf without following symlinks."""

    if not path.is_absolute():
        raise PublicationError("registered bundle directory must be absolute")
    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for component in path.parts[1:]:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            child = os.open(component, flags, dir_fd=descriptor)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise PublicationError(f"registered bundle component is not a directory: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _expect_artifact_value(document: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if document.get(key) != expected:
        raise PublicationError(f"{label} {key} drift")


def _validate_negative_bundle(config: dict[str, Any], payloads: Mapping[str, bytes]) -> None:
    spec = config["registered_evidence"][NEGATIVE_PACK_KEY]
    for item in spec["members"]:
        if item["name"] == "PUBLICATION_COMMIT.json":
            continue
        record = load_json(payloads[item["name"]], label=f"negative gate {item['name']}")
        _expect_artifact_value(record, "status", item["terminal_status"], label=item["name"])
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="negative terminal marker")
    gate_names = sorted(item["name"] for item in spec["members"] if item["name"] != "PUBLICATION_COMMIT.json")
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": spec["terminal_record_type"],
        "contract_id": config["contract_id"],
        "protocol_id": spec["protocol_id"],
        "dataset_id": "GSE200304",
        "decision_id": "V3-DEC-019",
        "publication_mode": spec["publication_mode"],
        "gate_record_count": 7,
        "gate_record_names": gate_names,
        "gate_payload_set_sha256": spec["gate_payload_set_sha256"],
        "final_output_target_sha256": spec["final_output_target_sha256"],
        "committed": True,
        "commit_marker_written_last": True,
    }.items():
        _expect_artifact_value(marker, key, expected, label="negative terminal marker")


def _validate_blocked_adjudication_bundle(config: dict[str, Any], payloads: Mapping[str, bytes]) -> None:
    spec = config["registered_evidence"][ADJUDICATION_KEY]
    report = load_json(payloads["ADJUDICATION_REPORT.json"], label="blocked adjudication report")
    audit = load_json(payloads["INPUT_EVIDENCE_AUDIT.json"], label="blocked adjudication input audit")
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="blocked adjudication terminal marker")
    for key, expected in {
        "status": spec["scientific_status"],
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "aggregate_only": True,
        "blockers": spec["unresolved_blockers"],
    }.items():
        _expect_artifact_value(report, key, expected, label="blocked adjudication report")
    for key, expected in {
        "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED",
        "all_inputs_aggregate_only": True,
        "row_level_payload_read_count": 0,
        "sequence_read_count": 0,
        "opened_input_count": 8,
    }.items():
        _expect_artifact_value(audit, key, expected, label="blocked adjudication input audit")
    sums = payloads["SHA256SUMS"]
    expected_sums = "".join(
        f"{sha256(payloads[name])}  {name}\n"
        for name in sorted(("ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json"))
    ).encode("ascii")
    if sums != expected_sums:
        raise PublicationError("blocked adjudication SHA256SUMS closure drift")
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": spec["terminal_record_type"],
        "contract_id": config["contract_id"],
        "decision_id": "V3-DEC-019",
        "dataset_id": "GSE200304",
        "output_id": spec["output_id"],
        "scientific_status": spec["scientific_status"],
        "publication_mode": "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1",
        "sha256sums_sha256": sha256(sums),
        "bundle_member_names_excluding_commit_marker": [
            "ADJUDICATION_REPORT.json", "INPUT_EVIDENCE_AUDIT.json", "SHA256SUMS"
        ],
        "bundle_file_count_excluding_commit_marker": 3,
        "final_output_target_sha256": spec["final_output_target_sha256"],
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }.items():
        _expect_artifact_value(marker, key, expected, label="blocked adjudication terminal marker")


def validate_registered_bundles(
    config: dict[str, Any],
    *,
    payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
) -> dict[str, Any]:
    """Validate the exact8 and exact4 terminal bundles without rewriting them."""

    selected: dict[str, Any] = {}
    if payload_overrides is not None and set(payload_overrides) != set(SOURCE_KEYS):
        raise PublicationError("registered bundle override closure drift")
    for source_key in SOURCE_KEYS:
        spec = config["registered_evidence"][source_key]
        expected_names = [item["name"] for item in spec["members"]]
        if payload_overrides is None:
            descriptor = open_absolute_directory_nofollow(Path(spec["absolute_directory"]))
            try:
                if sorted(os.listdir(descriptor)) != sorted(expected_names):
                    raise PublicationError(f"{source_key} exact member closure drift")
                payloads = {
                    item["name"]: read_regular_at(descriptor, item["name"], require_single_link=True)
                    for item in spec["members"]
                }
            finally:
                os.close(descriptor)
        else:
            payloads = dict(payload_overrides[source_key])
            if set(payloads) != set(expected_names):
                raise PublicationError(f"{source_key} override member closure drift")
        for item in spec["members"]:
            payload = payloads[item["name"]]
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise PublicationError(f"{source_key} artifact member bytes or SHA-256 drift: {item['name']}")
        if source_key == NEGATIVE_PACK_KEY:
            _validate_negative_bundle(config, payloads)
        else:
            _validate_blocked_adjudication_bundle(config, payloads)
        selected[source_key] = {
            "absolute_directory": spec["absolute_directory"],
            "member_count": len(spec["members"]),
            "members": copy.deepcopy(spec["members"]),
            "terminal_marker_validated": True,
        }
    return selected


def _validate_predecessor_objects(
    status: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    runtime = config["runtime"]
    if len(events) != 40 or not events or events[-1].get("event_id") != "A1-EVT-040":
        raise PublicationError("predecessor event count or tail drift")
    if events[-1].get("training_started") is not False:
        raise PublicationError("EVT040 explicit training_started=false truth drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 127:
        raise PublicationError("predecessor outputs are not exactly 127")
    historical = config["runtime_authority"]["historical_outer_runtime_authority"]
    for key, expected in {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "training_started": False,
        "next_phase_authorized": False,
        "code_commit": historical["code_commit"],
    }.items():
        _expect_exact(status.get(key), expected, label=f"predecessor STATUS {key}")
    for key, expected in {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "code_commit": historical["code_commit"],
        "active_authority_commit": historical["active_authority_commit"],
    }.items():
        _expect_exact(manifest.get(key), expected, label=f"predecessor manifest {key}")
    new_paths = {
        str(Path(source["absolute_directory"]) / item["name"])
        for source in config["registered_evidence"].values()
        for item in source["members"]
    }
    new_paths.update(str(Path(runtime["run_root"]) / name) for name in immutable_names(config))
    if any(isinstance(item, dict) and item.get("absolute_path") in new_paths for item in outputs):
        raise PublicationError("EVT041 output is already registered")


def read_exact_predecessor(
    run_fd: int, config: dict[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
    tail = config["runtime"]["predecessor_tail_event"]
    if len(tail_line) != tail["bytes"] or sha256(tail_line) != tail["sha256"]:
        raise PublicationError("EVT040 tail line identity drift")
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def output_record(artifact_type: str, absolute_path: Path, digest: str) -> dict[str, str]:
    return {"artifact_type": artifact_type, "absolute_path": str(absolute_path), "sha256": digest, "status": "COMPLETE"}


def snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {
        name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"]
        for name in MUTABLE_NAMES
    }


def immutable_names(config: dict[str, Any]) -> tuple[str, ...]:
    snapshots = snapshot_names(config)
    return (
        snapshots["STATUS.json"],
        snapshots["RUN_MANIFEST.json"],
        snapshots["EVENT_LOG.jsonl"],
        config["runtime"]["sync_name"],
    )


def expected_output_delta(config: dict[str, Any], sync_digest: str) -> list[dict[str, str]]:
    runtime = config["runtime"]
    root = Path(runtime["run_root"])
    sources = config["registered_evidence"]
    member_maps = {
        source_key: {item["name"]: item for item in sources[source_key]["members"]}
        for source_key in SOURCE_KEYS
    }
    result = []
    for source_key, member_name in SOURCE_MEMBER_OUTPUT_ORDER:
        source = sources[source_key]
        item = member_maps[source_key][member_name]
        result.append(
            output_record(
                item["artifact_type"],
                Path(source["absolute_directory"]) / member_name,
                item["sha256"],
            )
        )
    snapshots = snapshot_names(config)
    result.extend(
        [
            output_record("A1_STATUS_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_SNAPSHOT", root / snapshots["STATUS.json"], runtime["predecessor_mutables"]["STATUS.json"]["sha256"]),
            output_record("A1_RUN_MANIFEST_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_SNAPSHOT", root / snapshots["RUN_MANIFEST.json"], runtime["predecessor_mutables"]["RUN_MANIFEST.json"]["sha256"]),
            output_record("A1_EVENT_LOG_PRE_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_SNAPSHOT", root / snapshots["EVENT_LOG.jsonl"], runtime["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"]),
            output_record("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC_V1", root / runtime["sync_name"], sync_digest),
        ]
    )
    return result


def evt041_event_document(config: dict[str, Any], *, recorded_at: str, sync_digest: str) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": "A1-EVT-041",
        "at": recorded_at,
        "phase_id": "A1",
        "run_id": RUNTIME_RUN_ID,
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "sync_record": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_id": "A1-EVT-040",
        "predecessor_event_line_sha256": config["runtime"]["predecessor_tail_event"]["sha256"],
        "decision_id": "V3-DEC-019",
        "negative_evidence_runtime_sync_status": "REGISTERED_IN_PLACE_EVT_041",
        "blocked_adjudication_runtime_sync_status": "REGISTERED_IN_PLACE_EVT_041",
        "negative_evidence_member_count": 8,
        "blocked_adjudication_member_count": 4,
        "adjudication_scientific_status": config["registered_evidence"][ADJUDICATION_KEY]["scientific_status"],
        "qualification_changed": False,
        "gate_changed": False,
        "pre_sync_snapshot_count": 3,
        "registered_in_place_artifact_count": 12,
        "runtime_sync_record_count": 1,
        "manifest_output_count_before": 127,
        "manifest_output_count_after": 143,
        "detail": "Registered the exact terminal GSE200304 DEC019 negative-gate pack (8 members) and blocked adjudication bundle (4 members) in place. Seven blockers, all zero study/canonical contributions, and every training/model-selection/next-phase flag remain unchanged; no producer, adjudicator, qualifier, row-level payload, restricted/sealed path, or GPU work was invoked by this sync.",
    }
    event.update(copy.deepcopy(config["successor_invariants"]))
    event.update(copy.deepcopy(config["access_and_materialization_boundary"]))
    return event


def build_successors(
    config: dict[str, Any],
    predecessor_payloads: dict[str, bytes],
    predecessor_status: dict[str, Any],
    predecessor_manifest: dict[str, Any],
    predecessor_events: list[dict[str, Any]],
    selected_sources: dict[str, Any],
    authority_audit: dict[str, Any],
    recorded_at: str,
) -> dict[str, bytes]:
    runtime = config["runtime"]
    binding = config["implementation_binding"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    sync = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_RUNTIME_SYNC",
        "sync_type": config["sync_type"],
        "contract_id": config["contract_id"],
        "phase_id": "A1",
        "dataset_id": "GSE200304",
        "run_id": RUNTIME_RUN_ID,
        "recorded_at": recorded_at,
        "predecessor_runtime": {
            "runtime_root": str(root),
            "last_event_id": "A1-EVT-040",
            "event_count": 40,
            "manifest_output_count": 127,
            "last_event_line_sha256": runtime["predecessor_tail_event"]["sha256"],
            "immutable_pre_sync_snapshots": [
                {
                    "source_mutable_path": str(root / name),
                    "snapshot_path": str(root / snapshots[name]),
                    "bytes": len(predecessor_payloads[name]),
                    "sha256": sha256(predecessor_payloads[name]),
                }
                for name in MUTABLE_NAMES
            ],
        },
        "registered_evidence": {
            key: {
                "absolute_directory": selected_sources[key]["absolute_directory"],
                "member_count": selected_sources[key]["member_count"],
                "members": selected_sources[key]["members"],
                "terminal_marker_validated": True,
                "bodies_embedded": False,
            }
            for key in SOURCE_KEYS
        },
        "repository_lifecycle_authority": {
            "status": authority_audit["status"],
            "predecessor_ledger_commit": authority_audit["predecessor_ledger_commit"],
            "negative_nfs_binding_commit": authority_audit["negative_nfs_binding_commit"],
            "adjudicator_descriptor_commit": authority_audit["adjudicator_descriptor_commit"],
        },
        "runtime_sync_publisher_authority": {
            "status": authority_audit["status"],
            "base_commit": authority_audit["base_commit"],
            "implementation_commit": binding["implementation_commit"],
            "binding_commit": authority_audit["binding_commit"],
            "config_sha256": authority_audit["config_sha256"],
            "script_sha256": binding["implementation_script_sha256"],
            "test_sha256": binding["implementation_test_sha256"],
        },
        "a1_gate_snapshot": copy.deepcopy(config["successor_invariants"]),
        "access_and_materialization_boundary": copy.deepcopy(config["access_and_materialization_boundary"]),
        "hash_linkage": {
            "direction": "EVT040_AND_EXACT8_NEGATIVE_AND_EXACT4_BLOCKED_ADJUDICATION_TO_SYNC_TO_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successors_reference_sync_hash": True,
        },
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload = json_bytes(sync)
    sync_digest = sha256(sync_payload)
    prefix = "gse200304_dec019_blocked_adjudication_"
    status = dict(predecessor_status)
    status.update(
        {
            "updated_at": recorded_at,
            prefix + "runtime_sync_status": "SYNCED_EVT_041",
            prefix + "runtime_sync_record_sha256": sync_digest,
            prefix + "negative_evidence_member_count": 8,
            prefix + "adjudication_member_count": 4,
            prefix + "adjudication_input_count": 8,
            prefix + "unresolved_blocker_count": 7,
            prefix + "scientific_status": config["registered_evidence"][ADJUDICATION_KEY]["scientific_status"],
            prefix + "qualification_changed": False,
            prefix + "gate_changed": False,
            prefix + "ordinary_study_contribution": 0,
            prefix + "a1_study_contribution": 0,
            prefix + "true_a2_study_contribution": 0,
            prefix + "canonical_record_count": 0,
            prefix + "qualified": False,
            prefix + "canonical_materialization_allowed": False,
            prefix + "training_allowed": False,
            prefix + "model_selection_allowed": False,
            prefix + "next_phase_authorized": False,
        }
    )
    manifest = dict(predecessor_manifest)
    manifest[prefix + "runtime_sync_record_sha256"] = sync_digest
    manifest["outputs"] = list(predecessor_manifest["outputs"]) + expected_output_delta(config, sync_digest)
    event = evt041_event_document(config, recorded_at=recorded_at, sync_digest=sync_digest)
    artifacts = {
        snapshots["STATUS.json"]: predecessor_payloads["STATUS.json"],
        snapshots["RUN_MANIFEST.json"]: predecessor_payloads["RUN_MANIFEST.json"],
        snapshots["EVENT_LOG.jsonl"]: predecessor_payloads["EVENT_LOG.jsonl"],
        runtime["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(status),
        "RUN_MANIFEST.json": json_bytes(manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(event),
    }
    validate_successors(config, artifacts, predecessor_payloads, predecessor_status, predecessor_manifest, sync_digest)
    return artifacts


def validate_successors(
    config: dict[str, Any],
    artifacts: dict[str, bytes],
    predecessor_payloads: dict[str, bytes],
    predecessor_status: dict[str, Any],
    predecessor_manifest: dict[str, Any],
    sync_digest: str,
) -> None:
    runtime = config["runtime"]
    snapshots = snapshot_names(config)
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {runtime["sync_name"]}
    if set(artifacts) != expected_names or len(artifacts) != 7:
        raise RuntimeSyncError("prepared artifact schema is not exact seven-member closure")
    for mutable, snapshot in snapshots.items():
        if artifacts[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError(f"snapshot is not byte-exact: {snapshot}")
    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label="EVT041 runtime sync")
    if sha256(sync_payload) != sync_digest or sync.get("self_hash") != "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST":
        raise RuntimeSyncError("runtime sync identity drift")
    _expect_typed_exact(
        set(sync),
        {
            "schema_version", "record_type", "sync_type", "contract_id", "phase_id", "dataset_id",
            "run_id", "recorded_at", "predecessor_runtime", "registered_evidence",
            "repository_lifecycle_authority", "runtime_sync_publisher_authority", "a1_gate_snapshot",
            "access_and_materialization_boundary", "hash_linkage", "self_hash",
        },
        label="runtime sync root",
    )
    for key, count in ((NEGATIVE_PACK_KEY, 8), (ADJUDICATION_KEY, 4)):
        _expect_exact(sync["registered_evidence"][key]["member_count"], count, label=f"sync {key} member count")
        _expect_exact(sync["registered_evidence"][key]["bodies_embedded"], False, label=f"sync {key} body boundary")
    _expect_typed_exact(
        sync["hash_linkage"],
        {
            "direction": "EVT040_AND_EXACT8_NEGATIVE_AND_EXACT4_BLOCKED_ADJUDICATION_TO_SYNC_TO_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successors_reference_sync_hash": True,
        },
        label="one-way sync hash linkage",
    )
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode("ascii") in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("successor hash leaked into runtime sync")
    for name in MUTABLE_NAMES:
        if sync_digest.encode("ascii") not in artifacts[name]:
            raise RuntimeSyncError(f"successor lacks sync binding: {name}")

    prefix = "gse200304_dec019_blocked_adjudication_"
    status = load_json(artifacts["STATUS.json"], label="successor status")
    if {key: value for key, value in status.items() if not key.startswith(prefix) and key != "updated_at"} != {key: value for key, value in predecessor_status.items() if key != "updated_at"}:
        raise RuntimeSyncError("predecessor STATUS field rewritten")
    for key, expected in {
        "runtime_sync_status": "SYNCED_EVT_041",
        "runtime_sync_record_sha256": sync_digest,
        "negative_evidence_member_count": 8,
        "adjudication_member_count": 4,
        "adjudication_input_count": 8,
        "unresolved_blocker_count": 7,
        "scientific_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
        "qualification_changed": False,
        "gate_changed": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "qualified": False,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }.items():
        _expect_exact(status.get(prefix + key), expected, label=f"successor STATUS {key}")

    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor manifest")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError(f"predecessor manifest field rewritten: {key}")
    if set(manifest) - set(predecessor_manifest) != {prefix + "runtime_sync_record_sha256"}:
        raise RuntimeSyncError("manifest top-level runtime-sync delta drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 143 or outputs[:127] != predecessor_manifest["outputs"] or outputs[127:] != expected_output_delta(config, sync_digest):
        raise RuntimeSyncError("manifest ordered 127 -> 143 append drift")
    new_paths = [item["absolute_path"] for item in outputs[127:]]
    if len(set(new_paths)) != 16:
        raise RuntimeSyncError("EVT041 output paths are not unique")

    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor events")
    old_events = load_json_lines(predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor events")
    if len(events) != 41 or events[:-1] != old_events or not artifacts["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG is not an exact one-event append")
    expected_event = evt041_event_document(config, recorded_at=sync["recorded_at"], sync_digest=sync_digest)
    _expect_typed_exact(events[-1], expected_event, label="EVT-041 closed event")


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
    for token in runtime["forbidden_runtime_path_tokens"]:
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
            "base_commit",
            "implementation_commit",
            "predecessor_ledger_commit",
            "initial_runtime_implementation_commit",
            "initial_runtime_binding_commit",
            "repair_runtime_implementation_commit",
            "repair_runtime_binding_commit",
            "negative_nfs_binding_commit",
            "adjudicator_descriptor_commit",
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
    source_payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
) -> dict[str, Any]:
    config, _config_payload, authority = _load_runtime_context(
        config_path,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and (run_root_override is not None or source_payload_overrides is not None):
        raise PublicationError(
            "test-only run-root/source overrides are forbidden in production"
        )
    prepared_path, _allowed_root, _relative = _validate_prepared_path(
        prepared_directory, config, production=production
    )
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    lock_cleanup_warnings: list[dict[str, str]] = []
    with locked_directory(run_root, cleanup_warnings=lock_cleanup_warnings) as run_fd:
        selected_sources = validate_registered_bundles(
            config, payload_overrides=source_payload_overrides
        )
        predecessor, status, manifest, events = read_exact_predecessor(run_fd, config)
        artifacts = build_successors(
            config,
            predecessor,
            status,
            manifest,
            events,
            selected_sources,
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
        "manifest_output_transition": "127_TO_143",
        "negative_evidence_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
        "blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
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
    tail = runtime["predecessor_tail_event"]
    if len(tail_line) != tail["bytes"] or sha256(tail_line) != tail["sha256"]:
        raise PublicationError("prepared EVT040 tail line identity drift")
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def validate_prepared_against_context(
    prepared: dict[str, bytes],
    config: dict[str, Any],
    authority: dict[str, Any],
) -> None:
    predecessor, status, manifest, events = _predecessor_from_prepared(prepared, config)
    sync = load_json(prepared[config["runtime"]["sync_name"]], label="prepared sync record")
    recorded_at = sync.get("recorded_at")
    if not isinstance(recorded_at, str):
        raise PublicationError("prepared sync recorded_at is absent")
    selected_sources = {
        key: {
            "absolute_directory": config["registered_evidence"][key]["absolute_directory"],
            "member_count": len(config["registered_evidence"][key]["members"]),
            "members": copy.deepcopy(config["registered_evidence"][key]["members"]),
            "terminal_marker_validated": True,
        }
        for key in SOURCE_KEYS
    }
    expected = build_successors(
        config,
        predecessor,
        status,
        manifest,
        events,
        selected_sources,
        authority,
        recorded_at,
    )
    if prepared != expected:
        differing = sorted(name for name in expected if prepared.get(name) != expected[name])
        raise PublicationError(f"prepared runtime bytes do not match current authority: {differing!r}")


EVT041_TEMP_PATTERN = re.compile(
    r"\.evt041\.[0-9]+\.[0-9a-f]{16}\.(?P<target>[^/]+)\.tmp"
)


def _stale_evt041_temporaries(run_fd: int) -> list[str]:
    return sorted(
        name for name in os.listdir(run_fd) if EVT041_TEMP_PATTERN.fullmatch(name)
    )


def _inspect_preexisting_target_state(
    run_fd: int, prepared: Mapping[str, bytes], config: dict[str, Any]
) -> dict[str, list[str]]:
    """Read-only evidence for truthful recovery when initial classification fails."""

    immutable = immutable_names(config)
    immutable_set = set(immutable)
    stale_temporaries = _stale_evt041_temporaries(run_fd)
    manual_targets: list[str] = []
    exact_members: list[str] = []

    for temporary_name in stale_temporaries:
        match = EVT041_TEMP_PATTERN.fullmatch(temporary_name)
        assert match is not None
        target = match.group("target")
        if target not in immutable_set:
            continue
        try:
            temporary_info = os.stat(
                temporary_name, dir_fd=run_fd, follow_symlinks=False
            )
            target_info = os.stat(target, dir_fd=run_fd, follow_symlinks=False)
            linked = (
                stat.S_ISREG(temporary_info.st_mode)
                and stat.S_ISREG(target_info.st_mode)
                and (temporary_info.st_dev, temporary_info.st_ino)
                == (target_info.st_dev, target_info.st_ino)
            )
            exact = (
                linked
                and read_regular_at(run_fd, temporary_name) == prepared[target]
                and read_regular_at(run_fd, target) == prepared[target]
            )
        except (OSError, RuntimeSyncError):
            continue
        if exact:
            manual_targets.append(target)

    for name in (*immutable, *MUTABLE_NAMES):
        try:
            payload = read_optional_regular_at(run_fd, name)
        except RuntimeSyncError:
            continue
        if payload is not None and payload == prepared[name]:
            exact_members.append(name)

    return {
        "stale_temporaries": stale_temporaries,
        "manual_targets": sorted(set(manual_targets)),
        "exact_members": sorted(set(exact_members)),
    }


def classify_target(
    run_fd: int, prepared: dict[str, bytes], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Freshly classify every immutable/mutable under the held run lock."""

    runtime = config["runtime"]
    stale_temporaries = _stale_evt041_temporaries(run_fd)
    if stale_temporaries:
        raise PublicationError(
            "stale EVT-041 publisher temporary member makes the run namespace unclosed: "
            + repr(stale_temporaries)
        )
    predecessor, _status, _manifest, _events = _predecessor_from_prepared(prepared, config)
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
        existing = read_optional_regular_at(
            run_fd, name, require_single_link=True
        )
        if existing is None:
            immutable_states[name] = "ABSENT"
        elif existing == prepared[name]:
            immutable_states[name] = "EXISTING_EXACT"
        else:
            raise PublicationError(f"existing immutable artifact differs: {name}")
    return mutable_states, immutable_states


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

    existing = read_optional_regular_at(
        directory_fd, name, require_single_link=True
    )
    if existing is not None:
        if existing != payload:
            raise PublicationError(f"existing immutable artifact differs: {name}")
        return _reconfirm_existing_commit(
            directory_fd,
            state="EXISTING_EXACT_REUSED",
            fault_point="immutable_existing_exact_directory_fsync",
            fault_injector=fault_injector,
        )

    temporary_name = f".evt041.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
            existing = read_regular_at(
                directory_fd, name, require_single_link=True
            )
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

    def verify_single_link_after_temp_unlink() -> None:
        final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
            raise PublicationError(
                f"committed immutable does not have exactly one hard link: {name}"
            )
        if (
            read_regular_at(directory_fd, name, require_single_link=True)
            != payload
        ):
            raise PublicationError(
                f"committed immutable bytes differ after temp unlink: {name}"
            )

    post_commit(
        "immutable_post_unlink_single_link_identity",
        verify_single_link_after_temp_unlink,
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

    temporary_name = f".evt041.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
    source_payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
) -> dict[str, Any]:
    config, authority, prepared_path, prepared = _load_prepared_context(
        config_path=config_path,
        prepared_directory=prepared_directory,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and (run_root_override is not None or source_payload_overrides is not None):
        raise PublicationError("test-only run-root/source override is forbidden in production")
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    lock_cleanup_warnings: list[dict[str, str]] = []
    with locked_directory(run_root, cleanup_warnings=lock_cleanup_warnings) as run_fd:
        validate_registered_bundles(config, payload_overrides=source_payload_overrides)
        mutable, immutable = classify_target(run_fd, prepared, config)
        validate_prepared_against_context(prepared, config, authority)
    if lock_cleanup_warnings:
        raise PublicationError(f"run-lock cleanup failed after validation: {lock_cleanup_warnings!r}")
    return {
        "status": "VALIDATED_NOT_PUBLISHED",
        "run_root": str(run_root),
        "prepared_directory": str(prepared_path),
        "negative_evidence_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
        "blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
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
    source_payload_overrides: Mapping[str, Mapping[str, bytes]] | None = None,
) -> dict[str, Any]:
    config, authority, prepared_path, prepared = _load_prepared_context(
        config_path=config_path,
        prepared_directory=prepared_directory,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and (run_root_override is not None or fault_injector is not None or source_payload_overrides is not None):
        raise PublicationError("test-only run-root/fault/source overrides are forbidden in production")
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    snapshots = snapshot_names(config)
    old_payloads = {
        mutable: prepared[snapshot] for mutable, snapshot in snapshots.items()
    }
    results: dict[str, dict[str, Any]] = {
        "GSE200304_DEC019_REGISTERED_EVIDENCE": {
            "state": "NOT_YET_VALIDATED",
            "committed_by_this_call": False,
            "accepted": False,
            "last_validation_phase": "NOT_RUN",
            "warnings": [],
        }
    }
    committed_names: list[str] = []
    lock_cleanup_warnings: list[dict[str, str]] = []

    def revalidate_registered_sources(phase: str) -> None:
        source_result = results["GSE200304_DEC019_REGISTERED_EVIDENCE"]
        source_result.update(
            {
                "state": f"{phase}_VALIDATION_IN_PROGRESS",
                "accepted": False,
                "last_validation_phase": phase,
                "warnings": [],
            }
        )
        try:
            validate_registered_bundles(
                config, payload_overrides=source_payload_overrides
            )
        except Exception as exc:
            source_result.update(
                {
                    "state": f"{phase}_VALIDATION_FAILED",
                    "accepted": False,
                    "warnings": [
                        {
                            "point": f"registered_sources_{phase.lower()}",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                }
            )
            raise
        source_result.update(
            {
                "state": "EXACT8_NEGATIVE_AND_EXACT4_BLOCKED_ADJUDICATION_TERMINAL_CLOSURE_HASH_BOUND",
                "accepted": True,
                "last_validation_phase": phase,
                "warnings": [],
            }
        )

    with locked_directory(
        run_root,
        cleanup_warnings=lock_cleanup_warnings,
        fault_injector=fault_injector,
    ) as run_fd:
        mutable_preflight: dict[str, str] = {}
        immutable_preflight: dict[str, str] = {}
        preexisting_partial = False
        try:
            # Keep classification inside the truthful-state handler: an earlier
            # post-link temp-unlink failure leaves both the committed immutable
            # and its hard-linked .evt041 temp in the namespace.
            mutable_preflight, immutable_preflight = classify_target(
                run_fd, prepared, config
            )
            validate_prepared_against_context(prepared, config, authority)
            preexisting_partial = (
                [mutable_preflight[name] for name in MUTABLE_NAMES]
                != ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"]
                or any(
                    value == "EXISTING_EXACT"
                    for value in immutable_preflight.values()
                )
            )

            # Classify first so a source failure on an already-partial retry is
            # reported truthfully as partial state, while still preceding writes.
            revalidate_registered_sources("PREWRITE")
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

            # P0: source terminal closure is re-read after immutable creation and
            # immediately before the first mutable commit.
            revalidate_registered_sources("BEFORE_MUTABLES")
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

            final_mutable, final_immutable = classify_target(run_fd, prepared, config)
            revalidate_registered_sources("FINAL")
            validate_prepared_against_context(prepared, config, authority)
            if [final_mutable[name] for name in MUTABLE_NAMES] != [
                "NEW_EXACT",
                "NEW_EXACT",
                "NEW_EXACT",
            ]:
                raise PublicationError("final mutable state is not NEW/NEW/NEW")
            if any(value != "EXISTING_EXACT" for value in final_immutable.values()):
                raise PublicationError("final immutable closure is incomplete")
        except Exception as exc:
            inspected = _inspect_preexisting_target_state(run_fd, prepared, config)
            manual_targets = inspected["manual_targets"]
            exact_preexisting = inspected["exact_members"]
            exact_before_this_call = sorted(
                set(exact_preexisting) - set(committed_names)
            )
            if manual_targets:
                for target in manual_targets:
                    results[target] = {
                        "state": "EXISTING_EXACT_WITH_STALE_TEMP_HARDLINK",
                        "committed_by_this_call": False,
                        "accepted": False,
                        "warnings": [
                            {
                                "point": "immutable_post_link_unlink",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ],
                    }
                return {
                    "status": "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION",
                    "event_id": config["event_id"],
                    "run_root": str(run_root),
                    "prepared_directory": str(prepared_path),
                    "warning_member": manual_targets[0],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "committed_members": sorted(
                        set(committed_names + exact_preexisting)
                    ),
                    "preexisting_partial_state": True,
                    "stale_temporary_members": inspected["stale_temporaries"],
                    "mutable_preflight": mutable_preflight,
                    "immutable_preflight": immutable_preflight,
                    "results": results,
                    "lock_cleanup_warnings": lock_cleanup_warnings,
                }
            if committed_names or preexisting_partial or exact_before_this_call:
                return {
                    "status": "PARTIAL_STATE_REQUIRES_IDEMPOTENT_RETRY",
                    "event_id": config["event_id"],
                    "run_root": str(run_root),
                    "prepared_directory": str(prepared_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "committed_members": sorted(
                        set(committed_names + exact_preexisting)
                    ),
                    "preexisting_partial_state": bool(
                        preexisting_partial or exact_before_this_call
                    ),
                    "stale_temporary_members": inspected["stale_temporaries"],
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
        "manifest_output_transition": "127_TO_143",
        "negative_evidence_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
        "blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_041",
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
