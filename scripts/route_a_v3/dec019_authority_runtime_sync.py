#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed DEC-019 EVT-040 runtime sync.

The publisher revalidates the exact DEC-019 repository authority, copies the
already-committed aggregate-only GSE114002 public-gap audit byte-for-byte, and
appends one runtime event without changing any scientific gate.  It does not
run an adjudicator or qualifier, open a data/evidence payload, materialize a
canonical record, or authorize training.  Production has no caller-selected
repository, runtime root, or audit source.  The only caller-selected path is a
prepared artifact directory outside every run root.  An UNKNOWN config stops
before any repository or runtime source is opened.
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


CONFIG_REPO_PATH = "configs/route_a_v3_dec019_authority_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/dec019_authority_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_dec019_authority_runtime_sync.py"
PRODUCTION_REPO_ROOT = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_CONFIG_PATH = PRODUCTION_REPO_ROOT / CONFIG_REPO_PATH
GIT_BINARY = "/usr/bin/git"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_NAMES = ("STATUS.json", "RUN_MANIFEST.json", "EVENT_LOG.jsonl")
EXPECTED_AUTHORITY_FILES = [
    ("AUTHORITATIVE_CONTRACT", "docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md", "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982"),
    ("STATIC_VALIDATOR", "scripts/route_a_v3/validate_a0_bundle.py", "d1313fd4bdf5de8f8f08db1337e5017089cab5448a68446b9aa27a18e4950386"),
    ("REGISTRY_MANIFEST", "docs/execution/route_a_v3_registry_manifest.json", "7cab782ca5c7658f97ece895ebb96c5dca7f243936887830f03f8195ed4b9273"),
    ("A1_INTERIM", "docs/execution/route_a_v3_a1_interim.yaml", "64a41d3168df9cc80181e8865f2512f14f2aa8877f8b4574c90ea87363bf5b02"),
    ("DEC019_AMENDMENT", "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec019.yaml", "8c82e564398f0735fe4976f875fe91f053937b05044e5232e237694a2b36e1ca"),
    ("DECISION_LOG", "docs/execution/route_a_v3_decision_log.yaml", "b537a2ce19e4bb8b099f05df4ba383b56b8957cbc7be0b5954c9c11d741eb23b"),
    ("DATA_ROLE_REGISTRY", "docs/execution/route_a_v3_data_role_registry.yaml", "4d14ebd1a6adc04a344165f775df8586ef9f8f0461fdcac08649d0644d9956f2"),
    ("SPLIT_REGISTRY", "docs/execution/route_a_v3_split_registry.yaml", "2764d471c09a27da889b690cac317ac582bf9f25b79b6a34ac491f2e0b434929"),
    ("TASK_REGISTRY", "docs/execution/route_a_v3_task_registry.yaml", "6c6659ef0e9ddbbbba002f77d39d388dbdacc7b383e98ebb30a1580d590d85b4"),
    ("TASK_SPLIT_MATRIX", "docs/execution/route_a_v3_task_split_matrix.yaml", "dd340bcfb291138b862c5858daa28910c44689299647b468aedcc48b3d90b534"),
    ("CLAIM_EVIDENCE_MATRIX", "docs/execution/route_a_v3_claim_evidence_matrix.yaml", "25b62c17320032c764f986892647d4548065cac3a6d42414f96737da3fb3cbad"),
    ("A1_QUALIFICATION_REGISTRY", "configs/route_a_v3_a1_qualification.json", "fe3f7736c1f64b362ebda683ca571fc1a84e1fff36aed3a9ae67272665ba2343"),
]
PUBLIC_GAP_AUDIT_REPO_PATH = "docs/execution/gse114002_public_authority_gap_audit_v1.json"
PUBLIC_GAP_AUDIT_RUNTIME_NAME = (
    "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1_REPOSITORY_BLOB_SNAPSHOT.json"
)
PUBLIC_GAP_AUDIT_RUNTIME_ROLE = (
    "IMMUTABLE_EXACT_PRE_DEC019_AGGREGATE_AUDIT_RUNTIME_REGISTRATION_NO_QUALIFICATION"
)
PUBLIC_GAP_AUDIT_BYTES = 24861
PUBLIC_GAP_AUDIT_SHA256 = "3be184767bd297f2b50deff2b056e30e2229b970e9bbf0a9c3e5656e3147821f"


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


def validate_bound_config(config: dict[str, Any]) -> None:
    """Validate the closed DEC-019 I -> config-only B authority."""

    _expect_typed_exact(
        set(config),
        {
            "schema_version",
            "protocol_id",
            "contract_id",
            "phase_id",
            "event_id",
            "event_name",
            "sync_type",
            "implementation_binding",
            "repository_authority",
            "dec019_authority",
            "runtime_authority",
            "legacy_gse114002_public_gap",
            "runtime",
            "successor_invariants",
            "access_and_materialization_boundary",
            "publication_policy",
        },
        label="config root",
    )
    _expect_typed_exact(
        {
            key: config.get(key)
            for key in (
                "schema_version",
                "protocol_id",
                "contract_id",
                "phase_id",
                "event_id",
                "event_name",
                "sync_type",
            )
        },
        {
            "schema_version": "route_a_v3_dec019_authority_runtime_sync.v1",
            "protocol_id": "ROUTE_A_V3_DEC019_AUTHORITY_RUNTIME_SYNC_V1",
            "contract_id": "mrna_xeditflow_route_a_v3",
            "phase_id": "A1",
            "event_id": "A1-EVT-040",
            "event_name": "DEC019_AUTHORITY_AND_GSE114002_PUBLIC_GAP_LINEAGE_SYNCED_GATE_UNCHANGED",
            "sync_type": "APPEND_ONLY_AUTHORITY_REGISTRATION_NO_GATE_CHANGE",
        },
        label="config identity",
    )

    binding = config.get("implementation_binding")
    if not isinstance(binding, dict) or binding.get("status") != "BOUND":
        raise BindingError("runtime-sync config is not BOUND")
    _expect_typed_exact(
        set(binding),
        {
            "binding_scheme",
            "status",
            "implementation_commit",
            "implementation_script_path",
            "implementation_script_sha256",
            "implementation_test_path",
            "implementation_test_sha256",
            "compiled_core_sha256",
            "unknown_to_bound_scalar_paths",
        },
        label="implementation binding",
    )
    _expect_exact(
        binding.get("binding_scheme"),
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="binding scheme",
    )
    implementation = _expect_hex(
        binding.get("implementation_commit"), HEX40, label="implementation commit"
    )
    _expect_hex(
        binding.get("implementation_script_sha256"), HEX64, label="script SHA-256"
    )
    _expect_hex(
        binding.get("implementation_test_sha256"), HEX64, label="test SHA-256"
    )
    _expect_hex(
        binding.get("compiled_core_sha256"), HEX64, label="compiled core SHA-256"
    )
    _expect_exact(
        binding.get("implementation_script_path"), SCRIPT_REPO_PATH, label="script path"
    )
    _expect_exact(
        binding.get("implementation_test_path"), TEST_REPO_PATH, label="test path"
    )
    _expect_exact(
        binding.get("unknown_to_bound_scalar_paths"),
        [
            "implementation_binding.status",
            "implementation_binding.implementation_commit",
            "implementation_binding.implementation_script_sha256",
            "implementation_binding.implementation_test_sha256",
        ],
        label="binding scalar allowlist",
    )
    _expect_exact(
        binding.get("compiled_core_sha256"),
        compiled_core_sha256(config),
        label="compiled core",
    )

    authority = config.get("repository_authority")
    if not isinstance(authority, dict):
        raise BindingError("repository authority is absent")
    _expect_typed_exact(
        set(authority),
        {
            "production_repo_root",
            "branch",
            "base_commit",
            "base_commit_expected_parent",
            "implementation_commit_expected_parent",
            "binding_commit_expected_parent",
            "implementation_commit_exact_changed_paths",
            "binding_commit_exact_changed_paths",
        },
        label="repository authority",
    )
    _expect_exact(
        authority.get("production_repo_root"), str(PRODUCTION_REPO_ROOT), label="repo root"
    )
    _expect_exact(authority.get("branch"), "routea-v3-a1-20260810", label="branch")
    base = _expect_hex(authority.get("base_commit"), HEX40, label="base commit")
    _expect_exact(base, "78827501c7efcef28550b04876c98206d94d4808", label="base commit")
    _expect_exact(
        authority.get("base_commit_expected_parent"),
        "d54de63605a2df51e91262c99218684a80cb6515",
        label="base parent",
    )
    _expect_exact(
        authority.get("implementation_commit_expected_parent"), base, label="I parent"
    )
    _expect_exact(
        authority.get("binding_commit_expected_parent"),
        "IMPLEMENTATION_COMMIT_FROM_BINDING",
        label="B parent",
    )
    if implementation == base:
        raise BindingError("implementation commit did not advance from base")
    _expect_exact(
        authority.get("implementation_commit_exact_changed_paths"),
        sorted([CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]),
        label="I changed paths",
    )
    _expect_exact(
        authority.get("binding_commit_exact_changed_paths"),
        [CONFIG_REPO_PATH],
        label="B changed paths",
    )

    expected_authority = {
        "decision_id": "V3-DEC-019",
        "approval_itself_qualifies_any_study": False,
        "qualified_independent_ordinary_studies": 0,
        "qualified_a1_studies": 0,
        "qualified_a2_dense_studies": 0,
        "canonical_intervention_record_count": 0,
        "authority_files": [
            {"role": role, "path": path, "sha256": digest}
            for role, path, digest in EXPECTED_AUTHORITY_FILES
        ],
    }
    _expect_typed_exact(
        config.get("dec019_authority"), expected_authority, label="DEC-019 authority"
    )
    _expect_typed_exact(
        config.get("runtime_authority"),
        {
            "historical_outer_runtime_authority": {
                "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
                "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
            },
            "current_contract_authority": {
                "implementation_commit": "d54de63605a2df51e91262c99218684a80cb6515",
                "binding_commit": "78827501c7efcef28550b04876c98206d94d4808",
                "scope": "DEC019_AUTHORITY_AND_SUCCESSOR_ADJUDICATOR_BINDING",
                "active_amendment_decision_ids": [
                    "V3-DEC-017",
                    "V3-DEC-018",
                    "V3-DEC-019",
                ],
            },
        },
        label="runtime authority",
    )
    _expect_typed_exact(
        config.get("legacy_gse114002_public_gap"),
        {
            "dataset_id": "GSE114002",
            "artifact_lineage_id": "gse114002_public_authority_gap_audit_v1",
            "source_role": "REPOSITORY_AGGREGATE_AUDIT_AUTHORITY_BLOB_PRE_DEC019",
            "audit_path": PUBLIC_GAP_AUDIT_REPO_PATH,
            "audit_bytes": PUBLIC_GAP_AUDIT_BYTES,
            "audit_sha256": PUBLIC_GAP_AUDIT_SHA256,
            "audit_record_id": "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1",
            "audit_status": "PUBLIC_AUTHORITY_GAPS_AUDITED_NOT_QUALIFIED",
            "audited_at": "2026-08-11T10:24:16+08:00",
            "introduced_commit": "ad1c57b9255c3066510b08e7a4cf0bd571006811",
            "runtime_snapshot_name": PUBLIC_GAP_AUDIT_RUNTIME_NAME,
            "runtime_snapshot_role": PUBLIC_GAP_AUDIT_RUNTIME_ROLE,
            "interim_path": "docs/execution/route_a_v3_a1_interim.yaml",
            "interim_sha256": "64a41d3168df9cc80181e8865f2512f14f2aa8877f8b4574c90ea87363bf5b02",
            "historical_failed_attempt_lineage_id": "gse114002_endpoint_geometry_reconciliation_v2_attempt_001_failure",
            "current_mechanical_closure_lineage_id": "gse114002_endpoint_geometry_reconciliation_v2_attempt_002_mechanical_closure",
            "predecessor_runtime_event_id": "A1-EVT-039",
            "source_runtime_sync_status": "PENDING_NO_EVT_040",
            "evt040_registration_status": "REGISTERED_HASH_BOUND_NO_SOURCE_REWRITE",
            "source_authority_file_rewritten": False,
            "scientific_state_changed": False,
            "new_science_blocker_closed_count": 0,
            "new_scientific_output_contribution_count": 0,
            "new_runtime_audit_snapshot_count": 1,
        },
        label="legacy GSE114002 public-gap binding",
    )

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise BindingError("runtime section is absent")
    _expect_typed_exact(
        set(runtime),
        {
            "run_root",
            "allowed_prepared_root",
            "predecessor_event_id",
            "predecessor_event_count",
            "predecessor_manifest_output_count",
            "successor_event_id",
            "successor_event_count",
            "successor_manifest_output_count",
            "mutable_publish_order",
            "allowed_mutable_states",
            "predecessor_mutables",
            "predecessor_tail_event",
            "sync_name",
            "output_delta_count",
            "forbidden_runtime_path_tokens",
        },
        label="runtime",
    )
    _expect_exact(
        runtime.get("run_root"),
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        label="run root",
    )
    _expect_exact(
        runtime.get("allowed_prepared_root"),
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/staging/runtime_sync/A1",
        label="allowed prepared root",
    )
    for field, expected in (
        ("predecessor_event_count", 39),
        ("successor_event_count", 40),
        ("predecessor_manifest_output_count", 122),
        ("successor_manifest_output_count", 127),
        ("output_delta_count", 5),
    ):
        _expect_int(runtime.get(field), expected, label=field)
    _expect_exact(runtime.get("predecessor_event_id"), "A1-EVT-039", label="predecessor event")
    _expect_exact(runtime.get("successor_event_id"), "A1-EVT-040", label="successor event")
    _expect_exact(runtime.get("mutable_publish_order"), list(MUTABLE_NAMES), label="mutable order")
    _expect_exact(
        runtime.get("allowed_mutable_states"),
        [
            ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"],
            ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"],
            ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"],
            ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"],
        ],
        label="recovery states",
    )
    _expect_typed_exact(
        runtime.get("predecessor_mutables"),
        {
            "STATUS.json": {
                "bytes": 21042,
                "sha256": "a94fedccd0b19801b2d82dff55063bf0b03740ecb351628bc26e3e1c440d8376",
                "snapshot_name": "STATUS_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
            },
            "RUN_MANIFEST.json": {
                "bytes": 50117,
                "sha256": "4669f52f2ac1946f91f6121a80a310025c8f6825162bffb0f19ee472e5394d86",
                "snapshot_name": "RUN_MANIFEST_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json",
            },
            "EVENT_LOG.jsonl": {
                "bytes": 51402,
                "sha256": "6e82f992616fbd503b1ea79c5fd200910fbbe79361fdabf5714e11e93c4241cb",
                "snapshot_name": "EVENT_LOG_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_V1.jsonl",
            },
        },
        label="predecessor mutables",
    )
    _expect_typed_exact(
        runtime.get("predecessor_tail_event"),
        {
            "event_id": "A1-EVT-039",
            "bytes": 6104,
            "sha256": "a878458c70d2a6b9dd08d3448f0ccb1c89372831238e198b9025ba94b0e32994",
            "training_started_key_present": True,
            "training_started": False,
        },
        label="EVT039 tail identity",
    )
    _expect_exact(runtime.get("sync_name"), "A1_DEC019_AUTHORITY_RUNTIME_SYNC_V1.json", label="sync name")
    _expect_typed_exact(
        runtime.get("forbidden_runtime_path_tokens"),
        ["GSE246381", "/restricted/", "/sealed_external/", "FASTQ", "raw_replay", "adjudicate_"],
        label="forbidden runtime path tokens",
    )

    _expect_typed_exact(
        config.get("successor_invariants"),
        {
            "run_status": "IN_PROGRESS",
            "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "scientific_claim_status": "NOT_ESTABLISHED",
            "qualified_independent_ordinary_studies": 0,
            "qualified_a1_studies": 0,
            "qualified_a2_dense_studies": 0,
            "metadata_only_qualification_count": 0,
            "ordinary_study_contribution": 0,
            "a1_study_contribution": 0,
            "true_a2_study_contribution": 0,
            "canonical_intervention_record_count": 0,
            "qualified": False,
            "dec019_approval_is_study_qualification": False,
            "training_started": False,
            "training_allowed": False,
            "training_authorized": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        },
        label="successor gate truth",
    )
    _expect_typed_exact(
        config.get("access_and_materialization_boundary"),
        {
            "adjudicator_execution_count": 0,
            "adjudicator_artifact_read_count": 0,
            "external_evidence_source_read_count": 0,
            "runtime_evidence_payload_read_count": 0,
            "real_row_level_data_opened": False,
            "raw_reads_or_alignments_opened": False,
            "raw_fastq_body_read_count": 0,
            "restricted_or_sealed_path_accessed": False,
            "restricted_or_sealed_payload_contact": False,
            "gse246381_contact": False,
            "repository_authority_blob_hash_revalidation_count": 13,
            "runtime_mutable_predecessor_read_count": 3,
            "predecessor_snapshot_count": 3,
            "runtime_sync_record_count": 1,
            "new_runtime_output_count": 5,
            "canonical_read_count": 0,
            "canonical_write_count": 0,
            "qualifier_execution_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
            "gpu_work_started": False,
        },
        label="access and materialization boundary",
    )
    _expect_typed_exact(
        config.get("publication_policy"),
        {
            "immutable_no_overwrite": True,
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


def audit_repo_authority(
    repo_root: Path, config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Freshly bind the pushed direct I -> config-only B chain and authority blobs."""

    if (
        lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT
        or repo_root.is_symlink()
        or not repo_root.is_dir()
    ):
        raise AuthorityError("production repository root drift")
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    implementation = binding["implementation_commit"]
    base = authority["base_commit"]
    branch = authority["branch"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    origin = _run_git(
        repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}"
    ).decode().strip()
    _expect_exact(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        branch,
        label="branch",
    )
    _expect_exact(
        _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}")
        .decode()
        .strip(),
        f"origin/{branch}",
        label="upstream",
    )
    _expect_exact(
        _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip(),
        head,
        label="upstream head",
    )
    _expect_exact(origin, head, label="origin head")
    _expect_exact(
        _run_git(repo_root, "rev-parse", f"{head}^").decode().strip(),
        implementation,
        label="B parent",
    )
    _expect_exact(
        _run_git(repo_root, "rev-parse", f"{implementation}^").decode().strip(),
        base,
        label="I parent",
    )
    _expect_exact(
        _run_git(repo_root, "rev-parse", f"{base}^").decode().strip(),
        authority["base_commit_expected_parent"],
        label="DEC019 B parent",
    )
    if _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    ) != b"":
        raise AuthorityError("worktree or index is dirty")
    _run_git(repo_root, "merge-base", "--is-ancestor", base, head)
    _run_git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        config["legacy_gse114002_public_gap"]["introduced_commit"],
        base,
    )
    _expect_exact(
        _paths_changed_by_commit(repo_root, implementation),
        authority["implementation_commit_exact_changed_paths"],
        label="I paths",
    )
    _expect_exact(
        _paths_changed_by_commit(repo_root, head),
        authority["binding_commit_exact_changed_paths"],
        label="B paths",
    )
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("B config blob drift")
    i_config = load_json(
        _git_blob(repo_root, implementation, CONFIG_REPO_PATH), label="I config"
    )
    if (
        i_config != expected_unknown_i_config(config)
        or compiled_core_projection(i_config) != compiled_core_projection(config)
    ):
        raise AuthorityError("I/B config transition drift")
    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        if (
            sha256(_git_blob(repo_root, implementation, path)) != digest
            or sha256(_git_blob(repo_root, head, path)) != digest
            or sha256(read_regular_path(repo_root / path)) != digest
        ):
            raise AuthorityError(f"runtime-sync implementation blob drift: {path}")

    frozen_authority = list(config["dec019_authority"]["authority_files"])
    frozen_authority.append(
        {
            "role": "GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT",
            "path": config["legacy_gse114002_public_gap"]["audit_path"],
            "sha256": config["legacy_gse114002_public_gap"]["audit_sha256"],
        }
    )
    for record in frozen_authority:
        path = record["path"]
        expected_digest = record["sha256"]
        for commit, label in ((base, "base"), (implementation, "I"), (head, "B")):
            if sha256(_git_blob(repo_root, commit, path)) != expected_digest:
                raise AuthorityError(
                    f"DEC019 authority blob drift at {label}: {path}"
                )
        if sha256(read_regular_path(repo_root / path)) != expected_digest:
            raise AuthorityError(f"DEC019 worktree authority drift: {path}")

    gap = config["legacy_gse114002_public_gap"]
    introduced_payload = _git_blob(
        repo_root, gap["introduced_commit"], gap["audit_path"]
    )
    if (
        len(introduced_payload) != gap["audit_bytes"]
        or sha256(introduced_payload) != gap["audit_sha256"]
    ):
        raise AuthorityError("public-gap introduced blob drift")
    parent_tree_entry = _run_git(
        repo_root,
        "ls-tree",
        "-z",
        "--full-tree",
        f"{gap['introduced_commit']}^",
        "--",
        gap["audit_path"],
    )
    if parent_tree_entry != b"":
        raise AuthorityError("public-gap audit predates its frozen introduced commit")

    return {
        "status": "PASS_EXACT_DEC019_AUTHORITY_TO_RUNTIME_SYNC_I_TO_CONFIG_ONLY_B",
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin,
        "config_sha256": sha256(config_payload),
        "base_commit": base,
        "base_commit_parent": authority["base_commit_expected_parent"],
        "implementation_commit": implementation,
        "dec019_authority_blob_count": len(frozen_authority),
        "public_gap_audit_sha256": gap["audit_sha256"],
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value) is None:
        raise RuntimeSyncError("recorded_at must be RFC3339 +08:00")
    try:
        current, predecessor = datetime.fromisoformat(value), datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("invalid recorded_at") from exc
    if current.utcoffset() != timedelta(hours=8) or current <= predecessor:
        raise RuntimeSyncError("recorded_at must follow EVT-039")
    if current.astimezone(timezone.utc).date().isoformat() != "2026-08-11":
        raise RuntimeSyncError("recorded_at outside EVT-040 window")


def output_record(artifact_type: str, absolute_path: Path, digest: str) -> dict[str, str]:
    return {"artifact_type": artifact_type, "absolute_path": str(absolute_path), "sha256": digest, "status": "COMPLETE"}


def _validate_predecessor_objects(
    status: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    runtime = config["runtime"]
    if (
        len(events) != runtime["predecessor_event_count"]
        or not events
        or events[-1].get("event_id") != runtime["predecessor_event_id"]
    ):
        raise PublicationError("predecessor event count or tail drift")
    if events[-1].get("training_started") is not False:
        raise PublicationError("EVT039 explicit training_started=false truth drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 122:
        raise PublicationError("predecessor outputs are not exactly 122")
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
    registered_paths = {
        str(Path(runtime["run_root"]) / name) for name in immutable_names(config)
    }
    if any(
        isinstance(item, dict) and item.get("absolute_path") in registered_paths
        for item in outputs
    ):
        raise PublicationError("EVT040 output is already registered")


def read_exact_predecessor(
    run_fd: int, config: dict[str, Any]
) -> tuple[
    dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
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
        raise PublicationError("EVT039 tail line identity drift")
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def read_exact_public_gap_source(repo_root: Path, config: dict[str, Any]) -> bytes:
    """Read only the frozen repository audit authority; never an evidence payload."""

    gap = config["legacy_gse114002_public_gap"]
    payload = read_regular_path(repo_root / gap["audit_path"])
    if len(payload) != gap["audit_bytes"] or sha256(payload) != gap["audit_sha256"]:
        raise AuthorityError("public-gap audit source identity drift")
    audit = load_json(payload, label="GSE114002 public-gap audit authority")
    _expect_exact(
        audit.get("schema_version"),
        "route_a_v3_gse114002_public_authority_gap_audit.v1",
        label="public-gap schema",
    )
    _expect_exact(audit.get("record_id"), gap["audit_record_id"], label="public-gap record")
    _expect_exact(audit.get("status"), gap["audit_status"], label="public-gap status")
    _expect_exact(audit.get("audited_at"), gap["audited_at"], label="public-gap audited_at")
    lineage = audit.get("lineage")
    if not isinstance(lineage, dict):
        raise AuthorityError("public-gap audit lineage is absent")
    _expect_typed_exact(
        lineage,
        {
            "historical_failed_attempt_lineage_id": gap[
                "historical_failed_attempt_lineage_id"
            ],
            "current_mechanical_closure_lineage_id": gap[
                "current_mechanical_closure_lineage_id"
            ],
            "predecessor_runtime_event_id": gap["predecessor_runtime_event_id"],
            "predecessor_runtime_event_name": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_ATTEMPT_LINEAGE_SYNCED_GATE_UNCHANGED",
            "runtime_sync_status": gap["source_runtime_sync_status"],
        },
        label="public-gap historical PENDING lineage",
    )
    return payload


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
        config["legacy_gse114002_public_gap"]["runtime_snapshot_name"],
        config["runtime"]["sync_name"],
    )


def expected_output_delta(
    config: dict[str, Any], sync_digest: str
) -> list[dict[str, str]]:
    runtime = config["runtime"]
    snapshots = snapshot_names(config)
    gap = config["legacy_gse114002_public_gap"]
    root = Path(runtime["run_root"])
    return [
        output_record(
            "A1_STATUS_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            root / snapshots["STATUS.json"],
            runtime["predecessor_mutables"]["STATUS.json"]["sha256"],
        ),
        output_record(
            "A1_RUN_MANIFEST_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            root / snapshots["RUN_MANIFEST.json"],
            runtime["predecessor_mutables"]["RUN_MANIFEST.json"]["sha256"],
        ),
        output_record(
            "A1_EVENT_LOG_PRE_DEC019_AUTHORITY_RUNTIME_SYNC_SNAPSHOT",
            root / snapshots["EVENT_LOG.jsonl"],
            runtime["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"],
        ),
        output_record(
            "A1_GSE114002_PUBLIC_AUTHORITY_GAP_AUDIT_V1_REPOSITORY_BLOB_SNAPSHOT",
            root / gap["runtime_snapshot_name"],
            gap["audit_sha256"],
        ),
        output_record(
            "A1_DEC019_AUTHORITY_RUNTIME_SYNC_V1",
            root / runtime["sync_name"],
            sync_digest,
        ),
    ]


def evt040_event_document(
    config: dict[str, Any], *, recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    gap = config["legacy_gse114002_public_gap"]
    event: dict[str, Any] = {
        "event_id": "A1-EVT-040",
        "at": recorded_at,
        "phase_id": "A1",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "sync_record": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_id": "A1-EVT-039",
        "predecessor_event_line_sha256": config["runtime"]["predecessor_tail_event"][
            "sha256"
        ],
        "decision_id": "V3-DEC-019",
        "dec019_authority_runtime_sync_status": "SYNCED_EVT_040",
        "public_gap_runtime_sync_status": "SYNCED_EVT_040",
        "public_gap_source_runtime_sync_status": gap["source_runtime_sync_status"],
        "public_gap_source_authority_file_rewritten": False,
        "public_gap_audit_snapshot": gap["runtime_snapshot_name"],
        "public_gap_audit_sha256": gap["audit_sha256"],
        "public_gap_interim_sha256": gap["interim_sha256"],
        "approval_itself_qualifies_any_study": False,
        "qualification_changed": False,
        "gate_changed": False,
        "pre_sync_snapshot_count": 3,
        "public_gap_audit_snapshot_count": 1,
        "runtime_sync_record_count": 1,
        "manifest_output_count_before": 122,
        "manifest_output_count_after": 127,
        "detail": "Append-only authority sync: the exact DEC-019 authority and unchanged historical GSE114002 public-gap audit/interim lineage are hash-bound. The source PENDING_NO_EVT_040 authority bytes remain historical and unchanged; live runtime registration is SYNCED_EVT_040. No adjudicator, evidence payload, qualifier, canonical materialization, training, model selection, GPU work, scientific qualification, or next-phase authorization is added.",
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
    public_gap_payload: bytes,
    authority_audit: dict[str, Any],
    recorded_at: str,
) -> dict[str, bytes]:
    runtime = config["runtime"]
    binding = config["implementation_binding"]
    gap = config["legacy_gse114002_public_gap"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    if len(public_gap_payload) != gap["audit_bytes"] or sha256(public_gap_payload) != gap["audit_sha256"]:
        raise RuntimeSyncError("public-gap audit snapshot identity drift")
    sync = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_DEC019_AUTHORITY_RUNTIME_SYNC",
        "sync_type": config["sync_type"],
        "contract_id": config["contract_id"],
        "phase_id": "A1",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        "recorded_at": recorded_at,
        "predecessor_runtime": {
            "runtime_root": str(root),
            "last_event_id": "A1-EVT-039",
            "event_count": 39,
            "manifest_output_count": 122,
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
        "dec019_authority": copy.deepcopy(config["dec019_authority"]),
        "runtime_authority": copy.deepcopy(config["runtime_authority"]),
        "legacy_gse114002_public_gap": {
            **copy.deepcopy(gap),
            "runtime_snapshot_path": str(root / gap["runtime_snapshot_name"]),
            "runtime_snapshot_sha256": sha256(public_gap_payload),
            "live_runtime_sync_status": "SYNCED_EVT_040",
        },
        "runtime_sync_publisher_authority": {
            "status": authority_audit["status"],
            "base_commit": config["repository_authority"]["base_commit"],
            "implementation_commit": binding["implementation_commit"],
            "binding_commit": authority_audit["binding_commit"],
            "config_sha256": authority_audit["config_sha256"],
            "script_sha256": binding["implementation_script_sha256"],
            "test_sha256": binding["implementation_test_sha256"],
            "dec019_authority_blob_count": authority_audit[
                "dec019_authority_blob_count"
            ],
        },
        "a1_gate_snapshot": copy.deepcopy(config["successor_invariants"]),
        "access_and_materialization_boundary": copy.deepcopy(
            config["access_and_materialization_boundary"]
        ),
        "hash_linkage": {
            "direction": "EVT039_PREDECESSOR_AND_DEC019_REPOSITORY_AUTHORITY_AND_EXACT_PUBLIC_GAP_AUDIT_TO_RUNTIME_SYNC_TO_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successors_reference_sync_hash": True,
        },
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload = json_bytes(sync)
    sync_digest = sha256(sync_payload)
    status = dict(predecessor_status)
    status.update(
        {
            "updated_at": recorded_at,
            "dec019_authority_runtime_sync_status": "SYNCED_EVT_040",
            "dec019_authority_runtime_sync_record_sha256": sync_digest,
            "gse114002_public_authority_gap_runtime_sync_status": "SYNCED_EVT_040",
            "gse114002_public_authority_gap_runtime_snapshot_sha256": gap[
                "audit_sha256"
            ],
            "dec019_authority_runtime_sync_qualification_changed": False,
            "dec019_authority_runtime_sync_gate_changed": False,
        }
    )
    event = evt040_event_document(
        config, recorded_at=recorded_at, sync_digest=sync_digest
    )
    manifest = dict(predecessor_manifest)
    manifest["dec019_authority_runtime_sync_record_sha256"] = sync_digest
    manifest["outputs"] = list(predecessor_manifest["outputs"]) + expected_output_delta(
        config, sync_digest
    )
    artifacts = {
        snapshots["STATUS.json"]: predecessor_payloads["STATUS.json"],
        snapshots["RUN_MANIFEST.json"]: predecessor_payloads["RUN_MANIFEST.json"],
        snapshots["EVENT_LOG.jsonl"]: predecessor_payloads["EVENT_LOG.jsonl"],
        gap["runtime_snapshot_name"]: public_gap_payload,
        runtime["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(status),
        "RUN_MANIFEST.json": json_bytes(manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"]
        + compact_json_line(event),
    }
    validate_successors(
        config,
        artifacts,
        predecessor_payloads,
        predecessor_status,
        predecessor_manifest,
        sync_digest,
    )
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
    gap = config["legacy_gse114002_public_gap"]
    expected_names = (
        set(MUTABLE_NAMES)
        | set(snapshots.values())
        | {gap["runtime_snapshot_name"], runtime["sync_name"]}
    )
    if set(artifacts) != expected_names or len(artifacts) != 8:
        raise RuntimeSyncError("prepared artifact schema is not exact eight-member closure")
    for mutable, snapshot in snapshots.items():
        if artifacts[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError("snapshot byte drift")
    if (
        len(artifacts[gap["runtime_snapshot_name"]]) != gap["audit_bytes"]
        or sha256(artifacts[gap["runtime_snapshot_name"]]) != gap["audit_sha256"]
    ):
        raise RuntimeSyncError("public-gap exact snapshot drift")
    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label="DEC019 runtime sync")
    if sha256(sync_payload) != sync_digest:
        raise RuntimeSyncError("runtime sync identity drift")
    _expect_typed_exact(
        set(sync),
        {
            "schema_version",
            "record_type",
            "sync_type",
            "contract_id",
            "phase_id",
            "run_id",
            "recorded_at",
            "predecessor_runtime",
            "dec019_authority",
            "runtime_authority",
            "legacy_gse114002_public_gap",
            "runtime_sync_publisher_authority",
            "a1_gate_snapshot",
            "access_and_materialization_boundary",
            "hash_linkage",
            "self_hash",
        },
        label="runtime sync root",
    )
    _expect_typed_exact(sync.get("dec019_authority"), config["dec019_authority"], label="sync DEC019 authority")
    _expect_typed_exact(sync.get("runtime_authority"), config["runtime_authority"], label="sync runtime authority")
    _expect_exact(
        sync["legacy_gse114002_public_gap"].get("source_runtime_sync_status"),
        "PENDING_NO_EVT_040",
        label="historical source PENDING status",
    )
    _expect_exact(
        sync["legacy_gse114002_public_gap"].get("live_runtime_sync_status"),
        "SYNCED_EVT_040",
        label="live public-gap runtime status",
    )
    _expect_typed_exact(
        sync.get("access_and_materialization_boundary"),
        config["access_and_materialization_boundary"],
        label="runtime sync access boundary",
    )
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode() in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("successor hash leaked into runtime sync")
    for name in MUTABLE_NAMES:
        if sync_digest.encode() not in artifacts[name]:
            raise RuntimeSyncError(f"successor lacks sync binding: {name}")

    status = load_json(artifacts["STATUS.json"], label="successor status")
    allowed_status_delta = {
        "dec019_authority_runtime_sync_status",
        "dec019_authority_runtime_sync_record_sha256",
        "gse114002_public_authority_gap_runtime_sync_status",
        "gse114002_public_authority_gap_runtime_snapshot_sha256",
        "dec019_authority_runtime_sync_qualification_changed",
        "dec019_authority_runtime_sync_gate_changed",
    }
    if {
        key: value
        for key, value in status.items()
        if key != "updated_at" and key not in allowed_status_delta
    } != {
        key: value for key, value in predecessor_status.items() if key != "updated_at"
    }:
        raise RuntimeSyncError("predecessor STATUS field rewritten")
    if set(status) - set(predecessor_status) != allowed_status_delta:
        raise RuntimeSyncError("STATUS delta closure drift")
    for key, expected in {
        "dec019_authority_runtime_sync_status": "SYNCED_EVT_040",
        "gse114002_public_authority_gap_runtime_sync_status": "SYNCED_EVT_040",
        "gse114002_public_authority_gap_runtime_snapshot_sha256": gap["audit_sha256"],
        "dec019_authority_runtime_sync_qualification_changed": False,
        "dec019_authority_runtime_sync_gate_changed": False,
        "training_started": False,
    }.items():
        _expect_exact(status.get(key), expected, label=f"successor STATUS {key}")

    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor manifest")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError("predecessor manifest field rewritten")
    if set(manifest) - set(predecessor_manifest) != {
        "dec019_authority_runtime_sync_record_sha256"
    }:
        raise RuntimeSyncError("manifest top-level runtime-sync delta drift")
    outputs = manifest.get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 127
        or outputs[:122] != predecessor_manifest["outputs"]
        or outputs[122:] != expected_output_delta(config, sync_digest)
    ):
        raise RuntimeSyncError("manifest ordered 122 -> 127 append drift")

    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor events")
    old_events = load_json_lines(
        predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor events"
    )
    if (
        len(events) != 40
        or events[:-1] != old_events
        or not artifacts["EVENT_LOG.jsonl"].startswith(
            predecessor_payloads["EVENT_LOG.jsonl"]
        )
    ):
        raise RuntimeSyncError("EVENT_LOG is not an exact append")
    expected_event = evt040_event_document(
        config, recorded_at=sync["recorded_at"], sync_digest=sync_digest
    )
    _expect_typed_exact(events[-1], expected_event, label="EVT-040 closed event")


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
            "dec019_authority_blob_count",
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
    public_gap_payload_override: bytes | None = None,
) -> dict[str, Any]:
    config, _config_payload, authority = _load_runtime_context(
        config_path,
        production=production,
        config_override=config_override,
        authority_override=authority_override,
        repo_root=repo_root,
    )
    if production and (
        run_root_override is not None or public_gap_payload_override is not None
    ):
        raise PublicationError(
            "test-only run-root/public-gap overrides are forbidden in production"
        )
    prepared_path, _allowed_root, _relative = _validate_prepared_path(
        prepared_directory, config, production=production
    )
    run_root = run_root_override or Path(config["runtime"]["run_root"])
    public_gap_payload = public_gap_payload_override
    if public_gap_payload is None:
        public_gap_payload = read_exact_public_gap_source(
            repo_root or PRODUCTION_REPO_ROOT, config
        )
    elif (
        len(public_gap_payload)
        != config["legacy_gse114002_public_gap"]["audit_bytes"]
        or sha256(public_gap_payload)
        != config["legacy_gse114002_public_gap"]["audit_sha256"]
    ):
        raise AuthorityError("test public-gap override identity drift")
    lock_cleanup_warnings: list[dict[str, str]] = []
    with locked_directory(run_root, cleanup_warnings=lock_cleanup_warnings) as run_fd:
        predecessor, status, manifest, events = read_exact_predecessor(run_fd, config)
        artifacts = build_successors(
            config,
            predecessor,
            status,
            manifest,
            events,
            public_gap_payload,
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
        "manifest_output_transition": "122_TO_127",
        "public_gap_runtime_status": "SYNCED_EVT_040",
        "dec019_authority_runtime_status": "SYNCED_EVT_040",
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
        raise PublicationError("prepared EVT039 tail line identity drift")
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
    expected = build_successors(
        config,
        predecessor,
        status,
        manifest,
        events,
        prepared[config["legacy_gse114002_public_gap"]["runtime_snapshot_name"]],
        authority,
        recorded_at,
    )
    if prepared != expected:
        differing = sorted(name for name in expected if prepared.get(name) != expected[name])
        raise PublicationError(f"prepared runtime bytes do not match current authority: {differing!r}")


def classify_target(
    run_fd: int, prepared: dict[str, bytes], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Freshly classify every source/immutable/mutable under the held run lock."""

    runtime = config["runtime"]
    stale_temporaries = sorted(
        name
        for name in os.listdir(run_fd)
        if re.fullmatch(r"\.evt040\.[0-9]+\.[0-9a-f]{16}\.[^/]+\.tmp", name)
    )
    if stale_temporaries:
        raise PublicationError(
            "stale EVT-040 publisher temporary member makes the run namespace unclosed: "
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

    temporary_name = f".evt040.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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

    temporary_name = f".evt040.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
        mutable, immutable = classify_target(run_fd, prepared, config)
        validate_prepared_against_context(prepared, config, authority)
    if lock_cleanup_warnings:
        raise PublicationError(f"run-lock cleanup failed after validation: {lock_cleanup_warnings!r}")
    return {
        "status": "VALIDATED_NOT_PUBLISHED",
        "run_root": str(run_root),
        "prepared_directory": str(prepared_path),
        "public_gap_runtime_status": "SYNCED_EVT_040",
        "dec019_authority_runtime_status": "SYNCED_EVT_040",
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
        "DEC019_AUTHORITY_AND_PUBLIC_GAP_LINEAGE": {
            "state": "EXACT_REPOSITORY_AUTHORITY_AND_PENDING_SOURCE_LINEAGE_HASH_BOUND",
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
        mutable_preflight, immutable_preflight = classify_target(
            run_fd, prepared, config
        )
        validate_prepared_against_context(prepared, config, authority)
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

            final_mutable, final_immutable = classify_target(run_fd, prepared, config)
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
        "manifest_output_transition": "122_TO_127",
        "public_gap_runtime_status": "SYNCED_EVT_040",
        "dec019_authority_runtime_status": "SYNCED_EVT_040",
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
