#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed GSE200304 EVT-042 runtime sync.

The publisher revalidates the current producer/adjudicator descendants, the
post-upstream-pass ledger integration, and three exact terminal bundles.  It
registers those 16 already-published artifacts in place and appends one runtime
event; it never reruns a producer or adjudicator, decodes row/sequence payload,
rewrites a source bundle, materializes a canonical record, or changes the
overall qualification gate.

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
    "configs/route_a_v3_gse200304_dec019_upstream_pass_runtime_sync_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/gse200304_dec019_upstream_pass_runtime_sync.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/test_gse200304_dec019_upstream_pass_runtime_sync.py"
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
UPSTREAM_AUTHORITY_KEY = "upstream_authority"
UPSTREAM_PASS_PACK_KEY = "upstream_pass_gate_pack"
ADJUDICATION_KEY = "updated_blocked_adjudication"
SOURCE_KEYS = (UPSTREAM_AUTHORITY_KEY, UPSTREAM_PASS_PACK_KEY, ADJUDICATION_KEY)
SOURCE_MEMBER_OUTPUT_ORDER = (
    (UPSTREAM_AUTHORITY_KEY, "PMC10540565_EUROPE_PMC_FULLTEXT.xml"),
    (UPSTREAM_AUTHORITY_KEY, "GSE200302_family.soft.gz"),
    (UPSTREAM_AUTHORITY_KEY, "GSE200302_log2_cpm_counts_all_samples.txt.gz"),
    (UPSTREAM_AUTHORITY_KEY, "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json"),
    (UPSTREAM_AUTHORITY_KEY, "SHA256SUMS"),
    (UPSTREAM_AUTHORITY_KEY, "PUBLICATION_COMMIT.json"),
    (UPSTREAM_PASS_PACK_KEY, "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json"),
    (UPSTREAM_PASS_PACK_KEY, "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json"),
    (UPSTREAM_PASS_PACK_KEY, "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json"),
    (UPSTREAM_PASS_PACK_KEY, "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json"),
    (UPSTREAM_PASS_PACK_KEY, "SHA256SUMS"),
    (UPSTREAM_PASS_PACK_KEY, "PUBLICATION_COMMIT.json"),
    (ADJUDICATION_KEY, "ADJUDICATION_REPORT.json"),
    (ADJUDICATION_KEY, "INPUT_EVIDENCE_AUDIT.json"),
    (ADJUDICATION_KEY, "SHA256SUMS"),
    (ADJUDICATION_KEY, "PUBLICATION_COMMIT.json"),
)
STATUS_PREFIX = "gse200304_dec019_upstream_pass_"
UNRESOLVED_BLOCKERS = (
    "BIOLOGICAL_GROUP_AUTHORITY_NOT_PASS",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS",
    "OUTCOME_BLIND_SPLIT_LEAKAGE_NOT_PASS",
    "PREFROZEN_POWER_PRECISION_NOT_PASS",
)
INPUT_STATUS_COUNTS = {
    "PASS": 4,
    "BLOCKED": 1,
    "UNKNOWN_NOT_ASSERTED": 1,
    "NOT_RUN": 2,
}
PREDECESSOR_LEDGER_COMMIT = "ef2666e7a3e224f2043e7c647e10a4b8cadf01e8"
PREDECESSOR_LEDGER_PARENT = "8084a1e2b68eaf84bd4befb2f232759d7540b97c"
PREDECESSOR_LEDGER_INTEGRATION_ID = (
    "GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_UPSTREAM_PASS_GATE_PACK_V1"
)
PREDECESSOR_LEDGER_MANIFEST_STATUS = (
    "A1_DEC019_GSE200304_UPSTREAM_PASS_POST_ADJUDICATION_LEDGER_REGISTERED_PENDING_EVT042"
)
PREDECESSOR_LEDGER_LINEAGE_IDS = {
    UPSTREAM_AUTHORITY_KEY: "gse200304_upstream_authority_viability_v1",
    UPSTREAM_PASS_PACK_KEY: "gse200304_dec019_upstream_pass_gate_pack_v1",
    ADJUDICATION_KEY: (
        "gse200304_dec019_reported_endpoint_a1_adjudication_v3_upstream_pass_gate_pack_v1"
    ),
}
PREDECESSOR_LEDGER_BLOBS = (
    (
        "docs/execution/route_a_v3_a1_interim.yaml",
        "b1aceb1cf3d7dc2de4b77270045949659de64f70d7bc677e084b67c176a8beb1",
    ),
    (
        "docs/execution/route_a_v3_registry_manifest.json",
        "9ed5e415d96bbfe2c0fc6161fe4caa691b12ddae8fcc74c2f7dde0293123af8b",
    ),
    (
        "scripts/route_a_v3/validate_a0_bundle.py",
        "f087235353a53574e63b969c1e06c110572c07244063e551bb1e98dd7b612028",
    ),
    (
        "tests/route_a_v3/test_a0_integrity_guards.py",
        "b35f0e3e22eebb19a03f0a8bff42b658a6561aa54e5f56c8231ea0ef2d6a9920",
    ),
)
PREDECESSOR_LEDGER_UNKNOWN_TO_BOUND_PATHS = (
    "repository_authority.base_commit",
    "repository_authority.current_pre_runtime_sync_head",
    "repository_authority.predecessor_ledger.status",
    "repository_authority.predecessor_ledger.commit",
    "repository_authority.predecessor_ledger.integration_id",
    "repository_authority.predecessor_ledger.manifest_status",
    "repository_authority.predecessor_ledger.registered_lineage_ids.upstream_authority",
    "repository_authority.predecessor_ledger.registered_lineage_ids.upstream_pass_gate_pack",
    "repository_authority.predecessor_ledger.registered_lineage_ids.updated_blocked_adjudication",
    "repository_authority.predecessor_ledger.frozen_blobs[0].sha256",
    "repository_authority.predecessor_ledger.frozen_blobs[1].sha256",
    "repository_authority.predecessor_ledger.frozen_blobs[2].sha256",
    "repository_authority.predecessor_ledger.frozen_blobs[3].sha256",
)
HISTORICAL_RUNTIME_I1_COMMIT = "0bca29b958a6b5b7f65422812960ffaf53542d3a"
HISTORICAL_RUNTIME_I1_BLOBS = {
    CONFIG_REPO_PATH: "ec15f4f5e2da6da54511d54f42151bcfc9d04d03309f290c4f5a558a7a36bbd6",
    SCRIPT_REPO_PATH: "53189353aaa618c33384e24b568facc6c9840cc21f25552239ad12b44b8efb06",
    TEST_REPO_PATH: "b8d1f658639e24f6acf85863dc2976e583aa7b10ba5df83989e99c5353994148",
}
RUNTIME_I2_EXACT_CHANGED_PATHS = sorted([SCRIPT_REPO_PATH, TEST_REPO_PATH])
EXPECTED_SOURCE_MEMBERS = {
    UPSTREAM_AUTHORITY_KEY: {
        "PMC10540565_EUROPE_PMC_FULLTEXT.xml": ("A1_GSE200304_UPSTREAM_AUTHORITY_EUROPE_PMC_FULLTEXT", 298763, "4fe53c9ea58b5268b1014c0ef4b18cfbd7b5b3764f4c82542c065cb0aff5a7f0"),
        "GSE200302_family.soft.gz": ("A1_GSE200304_UPSTREAM_AUTHORITY_GEO_SOFT", 4699, "6df39a3406fe1bdf5a37345fee5605510ca1086fbce54d5aeeb934b562bb7d2e"),
        "GSE200302_log2_cpm_counts_all_samples.txt.gz": ("A1_GSE200304_UPSTREAM_AUTHORITY_PROCESSED_LOG2_CPM_MATRIX", 2843042, "ed93162f9540676138cfba05af2841c90619ac4335eb55ee3d956a3cd8aace3c"),
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json": ("A1_GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT", 10427, "997101dda5cbe3cf5a97bcfe9dda07150d11552decc159a2bc3cb96d9ebd0e45"),
        "SHA256SUMS": ("A1_GSE200304_UPSTREAM_AUTHORITY_SHA256SUMS", 420, "5f00d0d75ef8f12de5ed903a2c599498e5a6717f13a32b95d3f33765522ba371"),
        "PUBLICATION_COMMIT.json": ("A1_GSE200304_UPSTREAM_AUTHORITY_TERMINAL_COMMIT", 1031, "1dc26e885964bb15a2fad1ebb18e4ebf89fdf888e08e4b02058f396b2a4db664"),
    },
    UPSTREAM_PASS_PACK_KEY: {
        "GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_GATE.json": ("A1_GSE200304_DEC019_CANONICAL_REPORTED_ENDPOINT_SEMANTICS_PASS_GATE", 3988, "6603803960b747126a5b6dfb7d56bf124d36144fa87667098813ccae2fe41ba3", "PASS"),
        "GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_GATE.json": ("A1_GSE200304_DEC019_ROW_REPLICATE_OR_VALID_SE_PASS_GATE", 3944, "dc0a08a1a6b389fcd4c982a7e52ad34ebc9cf67563482c6adc84a7c2c51b3d0f", "PASS"),
        "GSE200304_DEC019_LICENSE_RIGHTS_GATE.json": ("A1_GSE200304_DEC019_LICENSE_RIGHTS_PASS_GATE", 3905, "08cb30aeac3b6e1e989e0d379b0b51c83a7fcbea6f4a3bb0501b4529d3a5192c", "PASS"),
        "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json": ("A1_GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT", 2171, "bdfc4d8c7cf941e28e545cf70b33ac12cf0ca7fae02914b95c15bef46fef7cf2"),
        "SHA256SUMS": ("A1_GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_SHA256SUMS", 476, "91cee112a8daa4fb562c76fe6a579146a9f1e7495785cbd24131c1032e6761c2"),
        "PUBLICATION_COMMIT.json": ("A1_GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_TERMINAL_COMMIT", 1362, "f22e074f049db71e20fac05b58dad17953232b4d651aee460c3a3c27b3a185a3"),
    },
    ADJUDICATION_KEY: {
        "ADJUDICATION_REPORT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_REPORT", 2359, "cc1423d84add812380641998c4e36e7096c10eaaaf74ed12c3b781b45fc4cece"),
        "INPUT_EVIDENCE_AUDIT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_INPUT_AUDIT", 2983, "72d836ecb373fd3841c9c3f91b6777172d979831bea7042c9a3da30f16040352"),
        "SHA256SUMS": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_SHA256SUMS", 183, "bff424c43fd392148a2d8417b171f325badb9adae1b2366010de4cfaca887dc6"),
        "PUBLICATION_COMMIT.json": ("A1_GSE200304_DEC019_BLOCKED_ADJUDICATION_TERMINAL_COMMIT", 1055, "b4094b0621d50a18fe5ab64d1662b4ee95cfdd93c572e25f111e9b53d2586b42"),
    },
}


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


def validate_static_config(config: dict[str, Any]) -> None:
    """Validate frozen ledger authority and the grouped-unbound/bound implementation."""

    _expect_keys(config, {
        "schema_version", "protocol_id", "contract_id", "phase_id", "event_id",
        "event_name", "sync_type", "implementation_binding",
        "repository_authority", "runtime_authority", "registered_evidence",
        "runtime", "successor_invariants", "access_and_materialization_boundary",
        "publication_policy",
    }, label="config root")
    _expect_typed_exact({key: config[key] for key in (
        "schema_version", "protocol_id", "contract_id", "phase_id", "event_id",
        "event_name", "sync_type",
    )}, {
        "schema_version": "route_a_v3_gse200304_dec019_upstream_pass_runtime_sync.v1",
        "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "event_id": "A1-EVT-042",
        "event_name": "GSE200304_DEC019_UPSTREAM_AUTHORITY_PASS_GATES_AND_UPDATED_BLOCKED_ADJUDICATION_SYNCED_QUALIFICATION_GATE_UNCHANGED",
        "sync_type": "APPEND_ONLY_UPSTREAM_AUTHORITY_PASS_GATES_AND_UPDATED_BLOCKED_ADJUDICATION_REGISTRATION_NO_QUALIFICATION_GATE_CHANGE",
    }, label="config identity")

    binding = _expect_keys(config["implementation_binding"], {
        "binding_scheme", "status", "implementation_commit",
        "implementation_script_path", "implementation_script_sha256",
        "implementation_test_path", "implementation_test_sha256",
        "compiled_core_sha256", "unknown_to_bound_scalar_paths",
    }, label="implementation binding")
    _expect_exact(binding["binding_scheme"], "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1", label="binding scheme")
    _expect_exact(binding["implementation_script_path"], SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding["implementation_test_path"], TEST_REPO_PATH, label="test path")
    _expect_exact(binding["unknown_to_bound_scalar_paths"], [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ], label="implementation binding transition")
    implementation_values = [binding[key] for key in (
        "status", "implementation_commit", "implementation_script_sha256",
        "implementation_test_sha256",
    )]
    if any(value == UNKNOWN for value in implementation_values):
        if not _binding_values_are_unknown(binding):
            raise BindingError("implementation binding is partially known")
    else:
        _expect_exact(binding["status"], "BOUND", label="implementation status")
        _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
        _expect_hex(binding["implementation_script_sha256"], HEX64, label="implementation script SHA")
        _expect_hex(binding["implementation_test_sha256"], HEX64, label="implementation test SHA")
    _expect_hex(binding["compiled_core_sha256"], HEX64, label="compiled core SHA")
    _expect_exact(binding["compiled_core_sha256"], compiled_core_sha256(config), label="compiled core")

    authority = _expect_keys(config["repository_authority"], {
        "production_repo_root", "branch", "base_commit",
        "current_pre_runtime_sync_head", "base_commit_expected_parent",
        "implementation_commit_expected_parent", "binding_commit_expected_parent",
        "implementation_commit_exact_changed_paths", "binding_commit_exact_changed_paths",
        "predecessor_ledger", "upstream_authority_producer_lifecycle",
        "upstream_pass_gate_producer_lifecycle", "adjudicator_lifecycle",
    }, label="repository authority")
    _expect_exact(authority["production_repo_root"], str(PRODUCTION_REPO_ROOT), label="repository root")
    _expect_exact(authority["branch"], "routea-v3-a1-20260810", label="repository branch")
    _expect_exact(authority["base_commit_expected_parent"], PREDECESSOR_LEDGER_PARENT, label="ledger parent/D3")
    _expect_exact(authority["implementation_commit_expected_parent"], "PREDECESSOR_LEDGER_COMMIT_FROM_CONFIG", label="historical runtime I1 parent rule")
    _expect_exact(authority["binding_commit_expected_parent"], "IMPLEMENTATION_COMMIT_FROM_BINDING", label="runtime B parent rule")
    _expect_exact(authority["implementation_commit_exact_changed_paths"], sorted([CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]), label="historical runtime I1 exact3 paths")
    _expect_exact(authority["binding_commit_exact_changed_paths"], [CONFIG_REPO_PATH], label="runtime B paths")

    ledger = authority["predecessor_ledger"]
    expected_ledger = {
        "status": "BOUND",
        "commit": PREDECESSOR_LEDGER_COMMIT,
        "expected_parent": PREDECESSOR_LEDGER_PARENT,
        "integration_id": PREDECESSOR_LEDGER_INTEGRATION_ID,
        "manifest_status": PREDECESSOR_LEDGER_MANIFEST_STATUS,
        "registered_lineage_ids": PREDECESSOR_LEDGER_LINEAGE_IDS,
        "commit_exact_changed_paths": [path for path, _digest in PREDECESSOR_LEDGER_BLOBS],
        "frozen_blobs": [
            {"path": path, "sha256": digest}
            for path, digest in PREDECESSOR_LEDGER_BLOBS
        ],
        "unknown_to_bound_scalar_paths": list(
            PREDECESSOR_LEDGER_UNKNOWN_TO_BOUND_PATHS
        ),
    }
    _expect_typed_exact(
        {
            "base_commit": authority["base_commit"],
            "current_pre_runtime_sync_head": authority["current_pre_runtime_sync_head"],
            "predecessor_ledger": ledger,
        },
        {
            "base_commit": PREDECESSOR_LEDGER_COMMIT,
            "current_pre_runtime_sync_head": PREDECESSOR_LEDGER_COMMIT,
            "predecessor_ledger": expected_ledger,
        },
        label="frozen predecessor ledger authority",
    )

    _expect_typed_exact(authority["upstream_authority_producer_lifecycle"], {
        "binding_commit": "9c313d2793880edd2a4355ec3781e045cae27252",
        "config_path": "configs/route_a_v3_gse200304_upstream_authority_viability_v1.json",
        "script_path": "scripts/route_a_v3/produce_gse200304_upstream_authority_viability.py",
        "test_path": "tests/route_a_v3/test_produce_gse200304_upstream_authority_viability.py",
        "bound_blobs": {
            "config_sha256": "c52688866026122488f1d8eef8d0bffebf864b99d78ddcc40c39a26221da76a1",
            "script_sha256": "525635da3d84183e325a17f00fc7cece1517acbd9ce317c2cc4e26a4ba38f03d",
            "test_sha256": "78bca039152874a09dd6a31a0789b712c72b45e60cb9e99e72391809a1bd7035",
        },
    }, label="upstream authority producer")
    _expect_typed_exact(authority["upstream_pass_gate_producer_lifecycle"], {
        "binding_commit": "ae4813a11b7e65e3aa118178f5d0e3d850cb73b8",
        "config_path": "configs/route_a_v3_gse200304_dec019_upstream_pass_gate_pack_v1.json",
        "script_path": "scripts/route_a_v3/produce_gse200304_dec019_upstream_pass_gate_pack.py",
        "test_path": "tests/route_a_v3/test_produce_gse200304_dec019_upstream_pass_gate_pack.py",
        "bound_blobs": {
            "config_sha256": "a241837bbd68a3c7321bfd96f1b3acf975cdc602762468b5ead161835778a7ae",
            "script_sha256": "b7ba11974f20472e77111a8f385d2595ad6600bd23d2fd5f969cfa8a91ed459a",
            "test_sha256": "9e36b125589a38c7e264d15762d1f55af4264b02feb10b7db5e904dd0151dcbe",
        },
    }, label="upstream PASS producer")
    _expect_typed_exact(authority["adjudicator_lifecycle"], {
        "descriptor_commit": "8084a1e2b68eaf84bd4befb2f232759d7540b97c",
        "config_path": "configs/route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json",
        "script_path": "scripts/route_a_v3/adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "test_path": "tests/route_a_v3/test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "descriptor_blobs": {
            "config_bytes": 22023,
            "config_sha256": "e7040fedd6e7217d402c36597c177f08fdf4921c55aced7379a1580c33c31891",
            "script_sha256": "9cd4411fcb02e1feed913b799296351e38ab9071b9506611318645e41b8dbbfe",
            "test_sha256": "8e7b188cfa2e5015fa307acad980f9ff2f45145943384fcadb50d67b1263e1db",
            "science_core_sha256": "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170",
            "descriptor_set_sha256": "97e2d5ca135f2e5668ef513de0247d5973481f6532f99beac9e9d8d9a828148b",
        },
    }, label="D3 adjudicator")

    _expect_typed_exact(config["runtime_authority"], {
        "historical_outer_runtime_authority": {
            "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
            "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        }
    }, label="outer runtime authority")

    registered = _expect_keys(config["registered_evidence"], set(SOURCE_KEYS), label="registered evidence")
    _expect_exact(list(registered), list(SOURCE_KEYS), label="registered evidence order")
    for source_key in SOURCE_KEYS:
        members = registered[source_key].get("members")
        expected = EXPECTED_SOURCE_MEMBERS[source_key]
        if not isinstance(members, list) or [item.get("name") for item in members] != list(expected):
            raise BindingError(f"{source_key} member order/closure drift")
        for item in members:
            expected_keys = {"name", "artifact_type", "bytes", "sha256"}
            expected_value = expected[item["name"]]
            if len(expected_value) == 4:
                expected_keys.add("terminal_status")
            _expect_keys(item, expected_keys, label=f"{source_key} member")
            actual = (item["artifact_type"], item["bytes"], item["sha256"])
            if len(expected_value) == 4:
                actual += (item["terminal_status"],)
            _expect_typed_exact(actual, expected_value, label=f"{source_key} {item['name']}")

    upstream = _expect_keys(registered[UPSTREAM_AUTHORITY_KEY], {
        "dataset_id", "absolute_directory", "protocol_id", "terminal_record_type",
        "publication_mode", "partial_default", "final_output_target_sha256", "members",
    }, label="upstream authority")
    _expect_typed_exact({key: upstream[key] for key in upstream if key != "members"}, {
        "dataset_id": "GSE200304",
        "absolute_directory": f"{config['runtime']['run_root']}/GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1",
        "protocol_id": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1",
        "terminal_record_type": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_PUBLICATION_COMMIT_V1",
        "publication_mode": "NFS_SAFE_EXCLUSIVE_MKDIR_O_EXCL_TERMINAL_MARKER_V1",
        "partial_default": "PRESERVE_AND_REQUIRE_MANUAL_ADJUDICATION",
        "final_output_target_sha256": "ad9b64166586813d86c99de49589fff565dbe24eb48d7d6aeb07808fb390dfaa",
    }, label="upstream authority identity")
    pass_pack = _expect_keys(registered[UPSTREAM_PASS_PACK_KEY], {
        "dataset_id", "absolute_directory", "protocol_id", "terminal_record_type",
        "publication_mode", "payload_set_sha256", "final_output_target_sha256", "members",
    }, label="upstream PASS pack")
    _expect_typed_exact({key: pass_pack[key] for key in pass_pack if key != "members"}, {
        "dataset_id": "GSE200304",
        "absolute_directory": f"{config['runtime']['run_root']}/GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1",
        "protocol_id": "ROUTE_A_V3_GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_V1",
        "terminal_record_type": "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PUBLICATION_COMMIT_V1",
        "publication_mode": "ATOMIC_MKDIR_TERMINAL_COMMIT_MARKER_V1",
        "payload_set_sha256": "f40f202577d2414390c6cd0fddb8126402ebbabb94408b9436bfa1fd4ae41277",
        "final_output_target_sha256": "af4569d71bdbe8286964795c190f048be100743a111baef0f12235424ef892e5",
    }, label="upstream PASS pack identity")
    blocked = _expect_keys(registered[ADJUDICATION_KEY], {
        "dataset_id", "absolute_directory", "output_id", "terminal_record_type",
        "publication_mode", "scientific_status", "final_output_target_sha256", "members",
        "input_count", "input_status_counts", "prior_unresolved_blocker_count",
        "resolved_input_blocker_count", "unresolved_blockers", "consumer_config_sha256",
        "evidence_descriptor_set_sha256", "science_core_sha256", "license_rights_scope",
        "ordinary_study_contribution", "a1_study_contribution", "true_a2_study_contribution",
        "canonical_record_count", "qualified", "canonical_materialization_allowed",
        "training_allowed", "model_selection_allowed", "next_phase_authorized",
    }, label="updated blocked adjudication")
    _expect_typed_exact({key: blocked[key] for key in (
        "dataset_id", "absolute_directory", "output_id", "terminal_record_type",
        "publication_mode", "scientific_status", "final_output_target_sha256",
        "consumer_config_sha256", "evidence_descriptor_set_sha256",
        "science_core_sha256", "license_rights_scope",
    )}, {
        "dataset_id": "GSE200304",
        "absolute_directory": "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_V3_UPSTREAM_PASS_GATE_PACK_V1",
        "output_id": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_BUNDLE_V3",
        "terminal_record_type": "ROUTE_A_V3_GSE200304_DEC019_REPORTED_ENDPOINT_A1_ADJUDICATION_COMMIT_V3",
        "publication_mode": "ATOMIC_EXCLUSIVE_DIRECTORY_TERMINAL_COMMIT_MARKER_V1",
        "scientific_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
        "final_output_target_sha256": "a18cbcbe6f4b578270a0beda868b2f28fd0f24bff33e61fa299599525dd7fc86",
        "consumer_config_sha256": "e7040fedd6e7217d402c36597c177f08fdf4921c55aced7379a1580c33c31891",
        "evidence_descriptor_set_sha256": "97e2d5ca135f2e5668ef513de0247d5973481f6532f99beac9e9d8d9a828148b",
        "science_core_sha256": "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170",
        "license_rights_scope": "PRIVATE_CANONICAL_ONLY",
    }, label="updated adjudication identity")
    _expect_int(blocked["input_count"], 8, label="adjudication input count")
    _expect_typed_exact(blocked["input_status_counts"], INPUT_STATUS_COUNTS, label="input status counts")
    _expect_int(blocked["prior_unresolved_blocker_count"], 7, label="prior blockers")
    _expect_int(blocked["resolved_input_blocker_count"], 3, label="resolved blockers")
    _expect_exact(blocked["unresolved_blockers"], list(UNRESOLVED_BLOCKERS), label="unresolved blockers")
    for key in ("ordinary_study_contribution", "a1_study_contribution", "true_a2_study_contribution", "canonical_record_count"):
        _expect_int(blocked[key], 0, label=f"adjudication {key}")
    for key in ("qualified", "canonical_materialization_allowed", "training_allowed", "model_selection_allowed", "next_phase_authorized"):
        _expect_typed_exact(blocked[key], False, label=f"adjudication {key}")

    runtime = _expect_keys(config["runtime"], {
        "run_root", "allowed_prepared_root", "predecessor_event_id", "predecessor_event_count",
        "predecessor_manifest_output_count", "successor_event_id", "successor_event_count",
        "successor_manifest_output_count", "mutable_publish_order", "allowed_mutable_states",
        "predecessor_mutables", "predecessor_tail_event", "sync_name", "output_delta_count",
        "forbidden_runtime_path_tokens",
    }, label="runtime")
    _expect_exact(runtime["run_root"], "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", label="run root")
    _expect_exact(runtime["allowed_prepared_root"], "/mnt/cunyuliu/mrna_xeditflow_routea_v3/staging/runtime_sync/A1", label="prepared root")
    for field, expected in (("predecessor_event_count", 41), ("successor_event_count", 42), ("predecessor_manifest_output_count", 143), ("successor_manifest_output_count", 163), ("output_delta_count", 20)):
        _expect_int(runtime[field], expected, label=field)
    _expect_exact(runtime["predecessor_event_id"], "A1-EVT-041", label="predecessor event")
    _expect_exact(runtime["successor_event_id"], "A1-EVT-042", label="successor event")
    _expect_exact(runtime["mutable_publish_order"], list(MUTABLE_NAMES), label="mutable order")
    _expect_exact(runtime["allowed_mutable_states"], [["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"], ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"], ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"], ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"]], label="recovery states")
    _expect_typed_exact(runtime["predecessor_mutables"], {
        "STATUS.json": {"bytes": 22946, "sha256": "cdf240e4c3903921fc543a8fd51810216acb9d4fdfa7b47b7db189c280f4c700", "snapshot_name": "STATUS_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1.json"},
        "RUN_MANIFEST.json": {"bytes": 58740, "sha256": "0591810dce67bced8899675aa7f0aa88789cd2d9660e25f3a74426c1770e7f07", "snapshot_name": "RUN_MANIFEST_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1.json"},
        "EVENT_LOG.jsonl": {"bytes": 57406, "sha256": "b8b82229003e3931e321ef8e740b18b8d4742c661768b51079cbdbca101c1469", "snapshot_name": "EVENT_LOG_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1.jsonl"},
    }, label="predecessor mutables")
    _expect_typed_exact(runtime["predecessor_tail_event"], {
        "event_id": "A1-EVT-041", "bytes": 2874,
        "sha256": "66eba75d44b0c62354c496f679d07933c0e0ccc8b28a54a20669582d8dd44969",
        "at": "2026-08-11T21:40:09+08:00", "training_started_key_present": True,
        "training_started": False,
    }, label="predecessor tail")
    _expect_exact(runtime["sync_name"], "A1_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1.json", label="sync name")
    _expect_exact(runtime["forbidden_runtime_path_tokens"], ["GSE246381", "/restricted/", "/sealed/", "/sealed_external/", "FASTQ", "raw_replay"], label="forbidden tokens")
    for source in registered.values():
        if any(token.casefold() in source["absolute_directory"].casefold() for token in runtime["forbidden_runtime_path_tokens"]):
            raise BindingError("registered source path crosses forbidden boundary")

    _expect_typed_exact(config["successor_invariants"], {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified_independent_ordinary_studies": 0, "qualified_a1_studies": 0,
        "qualified_a2_dense_studies": 0, "ordinary_study_contribution": 0,
        "a1_study_contribution": 0, "true_a2_study_contribution": 0,
        "canonical_intervention_record_count": 0, "qualified": False,
        "canonical_materialization_allowed": False, "training_started": False,
        "training_allowed": False, "training_authorized": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
    }, label="successor gate truth")
    _expect_typed_exact(config["access_and_materialization_boundary"], {
        "upstream_authority_artifact_read_count": 6,
        "upstream_pass_gate_artifact_read_count": 6,
        "blocked_adjudication_artifact_read_count": 4,
        "artifact_hash_validation_count": 16,
        "upstream_authority_producer_execution_count": 0,
        "upstream_pass_gate_producer_execution_count": 0,
        "adjudicator_execution_count": 0,
        "qualifier_execution_count": 0,
        "adjudication_input_count": 8,
        "adjudication_unresolved_blocker_count": 4,
        "runtime_mutable_predecessor_read_count": 3,
        "predecessor_snapshot_count": 3,
        "runtime_sync_record_count": 1,
        "registered_in_place_artifact_count": 16,
        "new_runtime_output_count": 20,
        "upstream_public_aggregate_artifact_body_opened_for_hash_validation": True,
        "upstream_public_aggregate_rows_decoded_or_parsed": False,
        "row_level_payload_decoded_or_parsed": False,
        "row_level_payload_read_count": 0,
        "sequence_payload_decoded_or_parsed": False,
        "sequence_read_count": 0,
        "raw_reads_or_alignments_opened": False,
        "raw_fastq_body_read_count": 0,
        "raw_replay_run_count": 0,
        "canonical_read_count": 0,
        "canonical_write_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
        "gpu_work_started": False,
        "restricted_or_sealed_path_accessed": False,
        "restricted_or_sealed_payload_contact": False,
        "gse246381_contact": False,
    }, label="access boundary")
    _expect_typed_exact(config["publication_policy"], {
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
    }, label="publication policy")


def validate_bound_config(config: dict[str, Any]) -> None:
    """Require the frozen ledger and the EVT042 runtime I/config-only-B binding."""

    validate_static_config(config)
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    if _binding_values_are_unknown(binding):
        raise BindingError("runtime-sync implementation is not BOUND")
    implementation = _expect_hex(binding["implementation_commit"], HEX40, label="implementation commit")
    base = _expect_hex(authority["base_commit"], HEX40, label="base commit")
    if implementation == base:
        raise BindingError("runtime-sync implementation did not advance from ledger base")


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
    """Prove D3 -> ledger -> frozen I1 -> script/test-only I2 -> config-only B2."""

    if lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT or repo_root.is_symlink() or not repo_root.is_dir():
        raise AuthorityError("production repository root drift")
    validate_bound_config(config)
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    ledger = authority["predecessor_ledger"]
    upstream = authority["upstream_authority_producer_lifecycle"]
    pass_pack = authority["upstream_pass_gate_producer_lifecycle"]
    adjudicator = authority["adjudicator_lifecycle"]
    implementation = binding["implementation_commit"]
    historical_i1 = HISTORICAL_RUNTIME_I1_COMMIT
    ledger_commit = ledger["commit"]
    d3 = adjudicator["descriptor_commit"]
    branch = authority["branch"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    origin = _run_git(repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}").decode().strip()
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), branch, label="branch")
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip(), f"origin/{branch}", label="upstream")
    _expect_exact(_run_git(repo_root, "rev-parse", "@{upstream}").decode().strip(), head, label="upstream head")
    _expect_exact(origin, head, label="origin head")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AuthorityError("worktree or index is dirty")

    _expect_exact(authority["base_commit"], ledger_commit, label="runtime base/ledger")
    _expect_exact(authority["current_pre_runtime_sync_head"], ledger_commit, label="pre-runtime head/ledger")
    _expect_parent(repo_root, head, implementation, label="runtime B2 parent/I2")
    _expect_parent(repo_root, implementation, historical_i1, label="runtime I2 parent/I1")
    _expect_parent(repo_root, historical_i1, ledger_commit, label="historical runtime I1 parent/ledger")
    _expect_parent(repo_root, ledger_commit, d3, label="ledger parent")
    _expect_parent(repo_root, d3, pass_pack["binding_commit"], label="D3/pass-pack B parent")
    _expect_ancestor(repo_root, upstream["binding_commit"], pass_pack["binding_commit"], label="upstream producer to PASS producer")
    _expect_exact(_paths_changed_by_commit(repo_root, ledger_commit), ledger["commit_exact_changed_paths"], label="ledger exact4 paths")
    _expect_exact(_paths_changed_by_commit(repo_root, historical_i1), authority["implementation_commit_exact_changed_paths"], label="historical runtime I1 exact3 paths")
    _expect_exact(_paths_changed_by_commit(repo_root, implementation), RUNTIME_I2_EXACT_CHANGED_PATHS, label="runtime I2 script/test-only paths")
    _expect_exact(_paths_changed_by_commit(repo_root, head), authority["binding_commit_exact_changed_paths"], label="runtime B2 config-only path")
    _expect_exact(_paths_changed_by_commit(repo_root, upstream["binding_commit"]), [upstream["config_path"]], label="upstream producer B path")
    _expect_exact(_paths_changed_by_commit(repo_root, pass_pack["binding_commit"]), [pass_pack["config_path"]], label="PASS producer B path")
    _expect_exact(_paths_changed_by_commit(repo_root, d3), [adjudicator["config_path"]], label="D3 path")

    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("runtime B2 config blob drift")
    historical_i1_payloads = {
        path: _git_blob(repo_root, historical_i1, path)
        for path in HISTORICAL_RUNTIME_I1_BLOBS
    }
    for path, digest in HISTORICAL_RUNTIME_I1_BLOBS.items():
        if sha256(historical_i1_payloads[path]) != digest:
            raise AuthorityError(f"historical runtime I1 blob drift: {path}")
    i2_config_payload = _git_blob(repo_root, implementation, CONFIG_REPO_PATH)
    if i2_config_payload != historical_i1_payloads[CONFIG_REPO_PATH]:
        raise AuthorityError("runtime I2 rewrote the frozen UNKNOWN I1 config")
    i_config = load_json(i2_config_payload, label="shared runtime I1/I2 UNKNOWN config")
    try:
        _expect_typed_exact(i_config, expected_unknown_i_config(config), label="runtime I1/I2/B2 transition")
        _expect_typed_exact(compiled_core_projection(i_config), compiled_core_projection(config), label="runtime I1/I2/B2 core")
    except RuntimeSyncError as exc:
        raise AuthorityError("runtime I1/I2/B2 config transition drift") from exc
    for path, digest in (
        (SCRIPT_REPO_PATH, binding["implementation_script_sha256"]),
        (TEST_REPO_PATH, binding["implementation_test_sha256"]),
    ):
        for commit in (implementation, head):
            if sha256(_git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"runtime I2/B2 implementation blob drift: {path}")
        if sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"runtime worktree implementation blob drift: {path}")

    for item in ledger["frozen_blobs"]:
        path, digest = item["path"], item["sha256"]
        for commit in (ledger_commit, historical_i1, implementation, head):
            if sha256(_git_blob(repo_root, commit, path)) != digest:
                raise AuthorityError(f"ledger blob drift: {path}")
        if sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"ledger worktree blob drift: {path}")

    for lifecycle, blob_key, label in (
        (upstream, "bound_blobs", "upstream authority producer"),
        (pass_pack, "bound_blobs", "upstream PASS producer"),
        (adjudicator, "descriptor_blobs", "D3 adjudicator"),
    ):
        expected = lifecycle[blob_key]
        current_commit = lifecycle.get("binding_commit", lifecycle.get("descriptor_commit"))
        for commit in (current_commit, head):
            _verify_three_blobs(
                repo_root,
                commit=commit,
                config_path=lifecycle["config_path"],
                script_path=lifecycle["script_path"],
                test_path=lifecycle["test_path"],
                expected=expected,
                label=label,
            )

    return {
        "status": "PASS_STRICT_LINEAR_DAG_D3_LEDGER_RUNTIME_I1_I2_CONFIG_ONLY_B2",
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin,
        "config_sha256": sha256(config_payload),
        "base_commit": ledger_commit,
        "implementation_commit": implementation,
        "historical_runtime_i1_commit": historical_i1,
        "runtime_i2_commit": implementation,
        "predecessor_ledger_commit": ledger_commit,
        "upstream_authority_binding_commit": upstream["binding_commit"],
        "upstream_pass_gate_binding_commit": pass_pack["binding_commit"],
        "adjudicator_descriptor_commit": d3,
        "ledger_blob_check_count": 4,
        "historical_runtime_i1_blob_check_count": 3,
        "runtime_i2_changed_path_count": 2,
        "producer_and_adjudicator_blob_check_count": 18,
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value) is None:
        raise RuntimeSyncError("recorded_at must be RFC3339 +08:00")
    try:
        current, predecessor = datetime.fromisoformat(value), datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("invalid recorded_at") from exc
    if current.utcoffset() != timedelta(hours=8) or current <= predecessor:
        raise RuntimeSyncError("recorded_at must follow EVT-041")


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


def _validate_upstream_authority_bundle(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> None:
    spec = config["registered_evidence"][UPSTREAM_AUTHORITY_KEY]
    members = {item["name"]: item for item in spec["members"]}
    content_names = (
        "PMC10540565_EUROPE_PMC_FULLTEXT.xml",
        "GSE200302_family.soft.gz",
        "GSE200302_log2_cpm_counts_all_samples.txt.gz",
        "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json",
    )
    expected_sums = "".join(
        f"{members[name]['sha256']}  {name}\n" for name in sorted(content_names)
    ).encode("ascii")
    if payloads["SHA256SUMS"] != expected_sums:
        raise PublicationError("upstream authority SHA256SUMS closure drift")
    audit = load_json(
        payloads["GSE200304_UPSTREAM_AUTHORITY_VIABILITY_AUDIT.json"],
        label="upstream authority closed audit",
    )
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": "GSE200304_UPSTREAM_SOURCE_AUTHORITY_VIABILITY_AUDIT_V1",
        "protocol_id": spec["protocol_id"],
        "contract_id": config["contract_id"],
        "phase_id": "A1",
        "dataset_id": "GSE200304",
        "decision_id": "V3-DEC-019",
        "status": "CLOSED_SOURCE_AUTHORITY_VIABILITY_READY_COMPONENTS_NO_GATE_CHANGE",
        "mode": "AUDIT_ONLY_NO_GATE_CHANGE",
    }.items():
        _expect_artifact_value(audit, key, expected, label="upstream authority audit")
    for section, status in (
        ("endpoint_crosswalk", "READY_FOR_PASS_RECORD_NOT_YET_BOUND"),
        ("replicate_branch", "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND"),
        ("private_only_rights", "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND"),
    ):
        _expect_artifact_value(audit[section], "status_if_all_source_checks_pass", status, label=section)
        _expect_artifact_value(audit[section], "consumer_gate_pass", False, label=section)
    _expect_artifact_value(
        audit["biological_group_authority"],
        "status",
        "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT",
        label="upstream group authority",
    )
    _expect_artifact_value(
        audit["processed_matrix_authority"],
        "matrix_key_set_equals_s3_key_set",
        True,
        label="upstream matrix authority",
    )
    _expect_artifact_value(
        audit["processed_matrix_authority"],
        "matrix_covers_every_finite_totalpoly_key",
        True,
        label="upstream matrix authority",
    )
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="upstream authority marker")
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": spec["terminal_record_type"],
        "protocol_id": spec["protocol_id"],
        "contract_id": config["contract_id"],
        "dataset_id": "GSE200304",
        "bundle_id": "GSE200304_UPSTREAM_AUTHORITY_VIABILITY_V1",
        "preterminal_member_names": sorted((*content_names, "SHA256SUMS")),
        "preterminal_member_count": 5,
        "exact_final_member_count": 6,
        "sha256sums_sha256": members["SHA256SUMS"]["sha256"],
        "final_output_target_sha256": spec["final_output_target_sha256"],
        "publication_mode": spec["publication_mode"],
        "committed": True,
        "terminal_marker_written_last": True,
        "no_overwrite": True,
        "partial_default": spec["partial_default"],
    }.items():
        _expect_artifact_value(marker, key, expected, label="upstream authority marker")


def _pass_pack_payload_set_sha256(payloads: Mapping[str, bytes]) -> str:
    preterminal = tuple(sorted(name for name in payloads if name != "PUBLICATION_COMMIT.json"))
    digest = hashlib.sha256()
    digest.update(b"GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_PAYLOAD_SET_V1\n")
    for name in preterminal:
        encoded = os.fsencode(name)
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(sha256(payloads[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_upstream_pass_pack(
    config: dict[str, Any], payloads: Mapping[str, bytes]
) -> None:
    spec = config["registered_evidence"][UPSTREAM_PASS_PACK_KEY]
    gate_names = tuple(
        item["name"] for item in spec["members"] if "terminal_status" in item
    )
    for name in gate_names:
        record = load_json(payloads[name], label=f"upstream PASS gate {name}")
        _expect_artifact_value(record, "status", "PASS", label=name)
    audit_name = "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT.json"
    audit = load_json(payloads[audit_name], label="upstream PASS pack audit")
    for key, expected in {
        "record_type": "GSE200304_DEC019_UPSTREAM_PASS_GATE_PACK_AUDIT_V1",
        "protocol_id": spec["protocol_id"],
        "contract_id": config["contract_id"],
        "dataset_id": "GSE200304",
        "decision_id": "V3-DEC-019",
        "status": "PASS_EXACT_THREE_CONSUMER_ACCEPTED_GATES_NO_ADJUDICATION",
        "upstream_exact6_verified": True,
        "decoded_raw_source_count": 0,
        "pass_gate_ids": [
            "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
            "LICENSE_RIGHTS",
            "ROW_REPLICATE_OR_VALID_SE",
        ],
        "consumer_validate_gate_record_pass_count": 3,
        "consumer_slot_gate_pass_exact_true_count": 3,
        "ordinary_study_contribution_delta": 0,
        "a1_study_contribution_delta": 0,
        "true_a2_study_contribution_delta": 0,
        "canonical_record_count_delta": 0,
        "aggregate_only": True,
    }.items():
        _expect_artifact_value(audit, key, expected, label="upstream PASS pack audit")
    content_names = tuple(sorted((*gate_names, audit_name)))
    expected_sums = "".join(
        f"{sha256(payloads[name])}  {name}\n" for name in content_names
    ).encode("ascii")
    if payloads["SHA256SUMS"] != expected_sums:
        raise PublicationError("upstream PASS pack SHA256SUMS closure drift")
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="upstream PASS marker")
    for key, expected in {
        "schema_version": "1.0.0",
        "record_type": spec["terminal_record_type"],
        "protocol_id": spec["protocol_id"],
        "contract_id": config["contract_id"],
        "dataset_id": "GSE200304",
        "decision_id": "V3-DEC-019",
        "publication_mode": spec["publication_mode"],
        "preterminal_member_names": sorted((*content_names, "SHA256SUMS")),
        "preterminal_member_count": 5,
        "exact_final_member_count": 6,
        "gate_record_names": sorted(gate_names),
        "descriptor_binding_scope": "THREE_GATE_JSON_FILES_ONLY",
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]),
        "payload_set_sha256": spec["payload_set_sha256"],
        "final_output_target_sha256": spec["final_output_target_sha256"],
        "committed": True,
        "commit_marker_written_last": True,
        "overwrite_allowed": False,
    }.items():
        _expect_artifact_value(marker, key, expected, label="upstream PASS marker")
    if _pass_pack_payload_set_sha256(payloads) != spec["payload_set_sha256"]:
        raise PublicationError("upstream PASS payload-set identity drift")


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
        "config_core_sha256": spec["science_core_sha256"],
        "evidence_descriptor_set_sha256": spec["evidence_descriptor_set_sha256"],
    }.items():
        _expect_artifact_value(report, key, expected, label="blocked adjudication report")
    for key, expected in {
        "mode": "ALL_HASH_BOUND_AGGREGATES_VERIFIED",
        "all_inputs_aggregate_only": True,
        "row_level_payload_read_count": 0,
        "sequence_read_count": 0,
        "opened_input_count": 8,
        "evidence_descriptor_set_sha256": spec["evidence_descriptor_set_sha256"],
    }.items():
        _expect_artifact_value(audit, key, expected, label="blocked adjudication input audit")
    slots = audit.get("slots")
    expected_slot_statuses = [
        ("CANONICAL_ROW_LOCATOR_MULTI_ASSET_LINEAGE", "PASS"),
        ("CANONICAL_REPORTED_ENDPOINT_SEMANTICS", "PASS"),
        ("BIOLOGICAL_GROUP_AUTHORITY", "BLOCKED"),
        ("ROW_REPLICATE_OR_VALID_SE", "PASS"),
        ("CHECKPOINT_SPECIFIC_EXPOSURE", UNKNOWN),
        ("LICENSE_RIGHTS", "PASS"),
        ("OUTCOME_BLIND_SPLIT_LEAKAGE", "NOT_RUN"),
        ("PREFROZEN_POWER_PRECISION", "NOT_RUN"),
    ]
    if not isinstance(slots, list) or [
        (slot.get("slot_id"), slot.get("gate_status")) for slot in slots
    ] != expected_slot_statuses:
        raise PublicationError("blocked adjudication slot status/order drift")
    observed_counts = {key: 0 for key in INPUT_STATUS_COUNTS}
    for _slot_id, status in expected_slot_statuses:
        observed_counts[status] += 1
    _expect_typed_exact(observed_counts, INPUT_STATUS_COUNTS, label="adjudication input status counts")
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
        "publication_mode": spec["publication_mode"],
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
    """Validate exact6 + exact6 + exact4 without copying or rewriting them."""

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
        if source_key == UPSTREAM_AUTHORITY_KEY:
            _validate_upstream_authority_bundle(config, payloads)
        elif source_key == UPSTREAM_PASS_PACK_KEY:
            _validate_upstream_pass_pack(config, payloads)
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
    if len(events) != 41 or not events or events[-1].get("event_id") != "A1-EVT-041":
        raise PublicationError("predecessor event count or tail drift")
    if events[-1].get("training_started") is not False:
        raise PublicationError("EVT041 explicit training_started=false truth drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 143:
        raise PublicationError("predecessor outputs are not exactly 143")
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
        raise PublicationError("EVT042 output is already registered")


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
        raise PublicationError("EVT041 tail line identity drift")
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
            output_record("A1_STATUS_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_SNAPSHOT", root / snapshots["STATUS.json"], runtime["predecessor_mutables"]["STATUS.json"]["sha256"]),
            output_record("A1_RUN_MANIFEST_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_SNAPSHOT", root / snapshots["RUN_MANIFEST.json"], runtime["predecessor_mutables"]["RUN_MANIFEST.json"]["sha256"]),
            output_record("A1_EVENT_LOG_PRE_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_SNAPSHOT", root / snapshots["EVENT_LOG.jsonl"], runtime["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"]),
            output_record("A1_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC_V1", root / runtime["sync_name"], sync_digest),
        ]
    )
    return result


def evt042_event_document(
    config: dict[str, Any], *, recorded_at: str, sync_digest: str
) -> dict[str, Any]:
    blocked = config["registered_evidence"][ADJUDICATION_KEY]
    event: dict[str, Any] = {
        "event_id": "A1-EVT-042",
        "at": recorded_at,
        "phase_id": "A1",
        "run_id": RUNTIME_RUN_ID,
        "event": config["event_name"],
        "sync_type": config["sync_type"],
        "sync_record": config["runtime"]["sync_name"],
        "sync_record_sha256": sync_digest,
        "predecessor_event_id": "A1-EVT-041",
        "predecessor_event_line_sha256": config["runtime"]["predecessor_tail_event"]["sha256"],
        "decision_id": "V3-DEC-019",
        "upstream_authority_runtime_sync_status": "REGISTERED_IN_PLACE_EVT_042",
        "upstream_pass_gate_pack_runtime_sync_status": "REGISTERED_IN_PLACE_EVT_042",
        "updated_blocked_adjudication_runtime_sync_status": "REGISTERED_IN_PLACE_EVT_042",
        "upstream_authority_member_count": 6,
        "upstream_pass_gate_pack_member_count": 6,
        "blocked_adjudication_member_count": 4,
        "adjudication_input_status_counts": copy.deepcopy(INPUT_STATUS_COUNTS),
        "prior_unresolved_blocker_count": 7,
        "resolved_input_blocker_count": 3,
        "adjudication_unresolved_blocker_count": 4,
        "adjudication_unresolved_blockers": list(UNRESOLVED_BLOCKERS),
        "scientific_status": blocked["scientific_status"],
        "qualification_changed": False,
        "evidence_gate_statuses_changed_since_evt041": True,
        "overall_qualification_gate_changed": False,
        "pre_sync_snapshot_count": 3,
        "registered_in_place_artifact_count": 16,
        "runtime_sync_record_count": 1,
        "manifest_output_count_before": 143,
        "manifest_output_count_after": 163,
        "detail": (
            "Registered the exact terminal GSE200304 upstream-authority exact6, "
            "upstream PASS-gate exact6, and updated blocked-adjudication exact4 "
            "in place. Three adjudication inputs are now PASS relative to EVT041, "
            "but four blockers remain; qualification, study contributions, canonical "
            "materialization, training, model selection, next-phase authorization, "
            "and scientific claims remain unchanged."
        ),
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
    blocked = config["registered_evidence"][ADJUDICATION_KEY]
    ledger = config["repository_authority"]["predecessor_ledger"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    sync = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_GSE200304_DEC019_UPSTREAM_PASS_RUNTIME_SYNC",
        "sync_type": config["sync_type"],
        "contract_id": config["contract_id"],
        "phase_id": "A1",
        "dataset_id": "GSE200304",
        "run_id": RUNTIME_RUN_ID,
        "recorded_at": recorded_at,
        "predecessor_runtime": {
            "runtime_root": str(root),
            "last_event_id": "A1-EVT-041",
            "event_count": 41,
            "manifest_output_count": 143,
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
                "lineage_id": ledger["registered_lineage_ids"][key],
                "absolute_directory": selected_sources[key]["absolute_directory"],
                "member_count": selected_sources[key]["member_count"],
                "members": selected_sources[key]["members"],
                "terminal_marker_validated": True,
                "bodies_embedded": False,
            }
            for key in SOURCE_KEYS
        },
        "adjudication_transition": {
            "scientific_status": blocked["scientific_status"],
            "input_count": 8,
            "input_status_counts": copy.deepcopy(INPUT_STATUS_COUNTS),
            "prior_unresolved_blocker_count": 7,
            "resolved_input_blocker_count": 3,
            "unresolved_blocker_count": 4,
            "unresolved_blockers": list(UNRESOLVED_BLOCKERS),
            "consumer_config_sha256": blocked["consumer_config_sha256"],
            "evidence_descriptor_set_sha256": blocked["evidence_descriptor_set_sha256"],
            "science_core_sha256": blocked["science_core_sha256"],
            "license_rights_scope": blocked["license_rights_scope"],
            "qualification_changed": False,
            "evidence_gate_statuses_changed_since_evt041": True,
            "overall_qualification_gate_changed": False,
        },
        "repository_lifecycle_authority": {
            "status": authority_audit["status"],
            "predecessor_ledger_commit": authority_audit["predecessor_ledger_commit"],
            "upstream_authority_binding_commit": authority_audit["upstream_authority_binding_commit"],
            "upstream_pass_gate_binding_commit": authority_audit["upstream_pass_gate_binding_commit"],
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
            "direction": "EVT041_AND_EXACT16_REGISTERED_SOURCES_TO_SYNC_TO_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successors_reference_sync_hash": True,
        },
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload = json_bytes(sync)
    sync_digest = sha256(sync_payload)
    status = dict(predecessor_status)
    status.update({
        "updated_at": recorded_at,
        STATUS_PREFIX + "runtime_sync_status": "SYNCED_EVT_042",
        STATUS_PREFIX + "runtime_sync_record_sha256": sync_digest,
        STATUS_PREFIX + "upstream_authority_member_count": 6,
        STATUS_PREFIX + "upstream_pass_gate_pack_member_count": 6,
        STATUS_PREFIX + "adjudication_member_count": 4,
        STATUS_PREFIX + "registered_in_place_artifact_count": 16,
        STATUS_PREFIX + "adjudication_input_count": 8,
        STATUS_PREFIX + "adjudication_pass_input_count": 4,
        STATUS_PREFIX + "adjudication_blocked_input_count": 1,
        STATUS_PREFIX + "adjudication_unknown_not_asserted_input_count": 1,
        STATUS_PREFIX + "adjudication_not_run_input_count": 2,
        STATUS_PREFIX + "prior_unresolved_blocker_count": 7,
        STATUS_PREFIX + "resolved_input_blocker_count": 3,
        STATUS_PREFIX + "unresolved_blocker_count": 4,
        STATUS_PREFIX + "unresolved_blockers": list(UNRESOLVED_BLOCKERS),
        STATUS_PREFIX + "consumer_config_sha256": blocked["consumer_config_sha256"],
        STATUS_PREFIX + "evidence_descriptor_set_sha256": blocked["evidence_descriptor_set_sha256"],
        STATUS_PREFIX + "science_core_sha256": blocked["science_core_sha256"],
        STATUS_PREFIX + "license_rights_scope": blocked["license_rights_scope"],
        STATUS_PREFIX + "scientific_status": blocked["scientific_status"],
        STATUS_PREFIX + "evidence_gate_statuses_changed_since_evt041": True,
        STATUS_PREFIX + "overall_qualification_gate_changed": False,
        STATUS_PREFIX + "qualification_changed": False,
        STATUS_PREFIX + "ordinary_study_contribution": 0,
        STATUS_PREFIX + "a1_study_contribution": 0,
        STATUS_PREFIX + "true_a2_study_contribution": 0,
        STATUS_PREFIX + "canonical_record_count": 0,
        STATUS_PREFIX + "qualified": False,
        STATUS_PREFIX + "canonical_materialization_allowed": False,
        STATUS_PREFIX + "training_started": False,
        STATUS_PREFIX + "training_allowed": False,
        STATUS_PREFIX + "training_authorized": False,
        STATUS_PREFIX + "model_selection_allowed": False,
        STATUS_PREFIX + "next_phase_authorized": False,
    })
    manifest = dict(predecessor_manifest)
    manifest[STATUS_PREFIX + "runtime_sync_record_sha256"] = sync_digest
    manifest["outputs"] = list(predecessor_manifest["outputs"]) + expected_output_delta(config, sync_digest)
    event = evt042_event_document(config, recorded_at=recorded_at, sync_digest=sync_digest)
    artifacts = {
        snapshots["STATUS.json"]: predecessor_payloads["STATUS.json"],
        snapshots["RUN_MANIFEST.json"]: predecessor_payloads["RUN_MANIFEST.json"],
        snapshots["EVENT_LOG.jsonl"]: predecessor_payloads["EVENT_LOG.jsonl"],
        runtime["sync_name"]: sync_payload,
        "STATUS.json": json_bytes(status),
        "RUN_MANIFEST.json": json_bytes(manifest),
        "EVENT_LOG.jsonl": predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(event),
    }
    validate_successors(
        config, artifacts, predecessor_payloads, predecessor_status,
        predecessor_manifest, sync_digest,
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
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {runtime["sync_name"]}
    if set(artifacts) != expected_names or len(artifacts) != 7:
        raise RuntimeSyncError("prepared artifact schema is not exact seven-member closure")
    for mutable, snapshot in snapshots.items():
        if artifacts[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError(f"snapshot is not byte-exact: {snapshot}")
    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label="EVT042 runtime sync")
    if sha256(sync_payload) != sync_digest or sync.get("self_hash") != "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST":
        raise RuntimeSyncError("runtime sync identity drift")
    _expect_typed_exact(set(sync), {
        "schema_version", "record_type", "sync_type", "contract_id", "phase_id",
        "dataset_id", "run_id", "recorded_at", "predecessor_runtime",
        "registered_evidence", "adjudication_transition", "repository_lifecycle_authority",
        "runtime_sync_publisher_authority", "a1_gate_snapshot",
        "access_and_materialization_boundary", "hash_linkage", "self_hash",
    }, label="runtime sync root")
    for key, count in (
        (UPSTREAM_AUTHORITY_KEY, 6),
        (UPSTREAM_PASS_PACK_KEY, 6),
        (ADJUDICATION_KEY, 4),
    ):
        _expect_exact(sync["registered_evidence"][key]["member_count"], count, label=f"sync {key} count")
        _expect_exact(sync["registered_evidence"][key]["bodies_embedded"], False, label=f"sync {key} body boundary")
        _expect_exact(
            sync["registered_evidence"][key]["lineage_id"],
            config["repository_authority"]["predecessor_ledger"]["registered_lineage_ids"][key],
            label=f"sync {key} lineage",
        )
    transition = sync["adjudication_transition"]
    _expect_typed_exact(transition, {
        "scientific_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
        "input_count": 8,
        "input_status_counts": INPUT_STATUS_COUNTS,
        "prior_unresolved_blocker_count": 7,
        "resolved_input_blocker_count": 3,
        "unresolved_blocker_count": 4,
        "unresolved_blockers": list(UNRESOLVED_BLOCKERS),
        "consumer_config_sha256": "e7040fedd6e7217d402c36597c177f08fdf4921c55aced7379a1580c33c31891",
        "evidence_descriptor_set_sha256": "97e2d5ca135f2e5668ef513de0247d5973481f6532f99beac9e9d8d9a828148b",
        "science_core_sha256": "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170",
        "license_rights_scope": "PRIVATE_CANONICAL_ONLY",
        "qualification_changed": False,
        "evidence_gate_statuses_changed_since_evt041": True,
        "overall_qualification_gate_changed": False,
    }, label="adjudication transition")
    _expect_typed_exact(sync["hash_linkage"], {
        "direction": "EVT041_AND_EXACT16_REGISTERED_SOURCES_TO_SYNC_TO_SUCCESSORS",
        "sync_record_references_successor_hashes": False,
        "successors_reference_sync_hash": True,
    }, label="one-way sync hash linkage")
    _expect_typed_exact(
        sync["access_and_materialization_boundary"],
        config["access_and_materialization_boundary"],
        label="sync access truth",
    )
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode("ascii") in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("successor hash leaked into runtime sync")
    for name in MUTABLE_NAMES:
        if sync_digest.encode("ascii") not in artifacts[name]:
            raise RuntimeSyncError(f"successor lacks sync binding: {name}")

    status = load_json(artifacts["STATUS.json"], label="successor status")
    if {
        key: value for key, value in status.items()
        if not key.startswith(STATUS_PREFIX) and key != "updated_at"
    } != {
        key: value for key, value in predecessor_status.items() if key != "updated_at"
    }:
        raise RuntimeSyncError("predecessor STATUS field rewritten")
    expected_status = {
        "runtime_sync_status": "SYNCED_EVT_042",
        "runtime_sync_record_sha256": sync_digest,
        "upstream_authority_member_count": 6,
        "upstream_pass_gate_pack_member_count": 6,
        "adjudication_member_count": 4,
        "registered_in_place_artifact_count": 16,
        "adjudication_input_count": 8,
        "adjudication_pass_input_count": 4,
        "adjudication_blocked_input_count": 1,
        "adjudication_unknown_not_asserted_input_count": 1,
        "adjudication_not_run_input_count": 2,
        "prior_unresolved_blocker_count": 7,
        "resolved_input_blocker_count": 3,
        "unresolved_blocker_count": 4,
        "unresolved_blockers": list(UNRESOLVED_BLOCKERS),
        "consumer_config_sha256": "e7040fedd6e7217d402c36597c177f08fdf4921c55aced7379a1580c33c31891",
        "evidence_descriptor_set_sha256": "97e2d5ca135f2e5668ef513de0247d5973481f6532f99beac9e9d8d9a828148b",
        "science_core_sha256": "13394ac6a9b9ec6e6241d0d9b1048ecfa5c90874c7447991fc2a8248a574c170",
        "license_rights_scope": "PRIVATE_CANONICAL_ONLY",
        "scientific_status": "BLOCKED_DEC019_REPORTED_ENDPOINT_A1_EVIDENCE_INCOMPLETE",
        "evidence_gate_statuses_changed_since_evt041": True,
        "overall_qualification_gate_changed": False,
        "qualification_changed": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "qualified": False,
        "canonical_materialization_allowed": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }
    for key, expected in expected_status.items():
        _expect_typed_exact(status.get(STATUS_PREFIX + key), expected, label=f"successor STATUS {key}")

    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor manifest")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError(f"predecessor manifest field rewritten: {key}")
    if set(manifest) - set(predecessor_manifest) != {STATUS_PREFIX + "runtime_sync_record_sha256"}:
        raise RuntimeSyncError("manifest top-level runtime-sync delta drift")
    outputs = manifest.get("outputs")
    expected_delta = expected_output_delta(config, sync_digest)
    if not isinstance(outputs, list) or len(outputs) != 163 or outputs[:143] != predecessor_manifest["outputs"] or outputs[143:] != expected_delta:
        raise RuntimeSyncError("manifest ordered 143 -> 163 append drift")
    new_paths = [item["absolute_path"] for item in outputs[143:]]
    if len(new_paths) != 20 or len(set(new_paths)) != 20:
        raise RuntimeSyncError("EVT042 output paths are not exact20 unique")

    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor events")
    old_events = load_json_lines(predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor events")
    if len(events) != 42 or events[:-1] != old_events or not artifacts["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG is not an exact one-event append")
    expected_event = evt042_event_document(
        config, recorded_at=sync["recorded_at"], sync_digest=sync_digest
    )
    _expect_typed_exact(events[-1], expected_event, label="EVT-042 closed event")


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
            "upstream_authority_binding_commit",
            "upstream_pass_gate_binding_commit",
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
        "manifest_output_transition": "143_TO_163",
        "upstream_authority_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "upstream_pass_gate_pack_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "updated_blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
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
        raise PublicationError("prepared EVT041 tail line identity drift")
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


EVT042_TEMP_PATTERN = re.compile(
    r"\.evt042\.[0-9]+\.[0-9a-f]{16}\.(?P<target>[^/]+)\.tmp"
)


def _stale_evt042_temporaries(run_fd: int) -> list[str]:
    return sorted(
        name for name in os.listdir(run_fd) if EVT042_TEMP_PATTERN.fullmatch(name)
    )


def _inspect_preexisting_target_state(
    run_fd: int, prepared: Mapping[str, bytes], config: dict[str, Any]
) -> dict[str, list[str]]:
    """Read-only evidence for truthful recovery when initial classification fails."""

    immutable = immutable_names(config)
    immutable_set = set(immutable)
    stale_temporaries = _stale_evt042_temporaries(run_fd)
    manual_targets: list[str] = []
    exact_members: list[str] = []

    for temporary_name in stale_temporaries:
        match = EVT042_TEMP_PATTERN.fullmatch(temporary_name)
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
    stale_temporaries = _stale_evt042_temporaries(run_fd)
    if stale_temporaries:
        raise PublicationError(
            "stale EVT-042 publisher temporary member makes the run namespace unclosed: "
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

    temporary_name = f".evt042.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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

    temporary_name = f".evt042.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
        "upstream_authority_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "upstream_pass_gate_pack_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "updated_blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
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
                "state": "EXACT6_UPSTREAM_AUTHORITY_EXACT6_PASS_PACK_AND_EXACT4_BLOCKED_ADJUDICATION_HASH_BOUND",
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
            # and its hard-linked .evt042 temp in the namespace.
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
            complete_commit_order = [*immutable_names(config), *MUTABLE_NAMES]
            source_result = results["GSE200304_DEC019_REGISTERED_EVIDENCE"]
            final_source_warning_after_complete_commit = (
                set(exact_preexisting) == set(complete_commit_order)
                and source_result.get("state") == "FINAL_VALIDATION_FAILED"
                and source_result.get("last_validation_phase") == "FINAL"
            )
            if final_source_warning_after_complete_commit:
                return {
                    "status": "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY",
                    "event_id": config["event_id"],
                    "event_committed": True,
                    "run_root": str(run_root),
                    "prepared_directory": str(prepared_path),
                    "warning_member": "GSE200304_DEC019_REGISTERED_EVIDENCE",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "committed_members": complete_commit_order,
                    "preexisting_partial_state": preexisting_partial,
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
        "manifest_output_transition": "143_TO_163",
        "upstream_authority_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "upstream_pass_gate_pack_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
        "updated_blocked_adjudication_runtime_status": "REGISTERED_IN_PLACE_EVT_042",
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
