#!/usr/bin/env python3
"""Fail-closed guard and evidence writer for the production B0 driver.

This module freezes executable state, validates approved runtime/input topology,
evaluates a closed set of named B0 JSON gates, records attempt state, samples
only operational health, and seals checksums after every gate has passed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

try:
    from scripts.execution.run_audited_command import _capture_git_snapshot
except ModuleNotFoundError:  # direct execution from scripts/execution
    from run_audited_command import _capture_git_snapshot
try:
    from scripts.execution.acceptance_semantics import validate_phase_acceptance
except ModuleNotFoundError:  # direct execution from scripts/execution
    from acceptance_semantics import validate_phase_acceptance


SCHEMA_VERSION = "b0_driver_guard.v1"
WATCHDOG_INTERVAL_SECONDS = 300
EXPECTED_AUDIT_NODES = (
    "00_preflight",
    "01_canonical_validation",
    "02_split_5utr_source",
    "03_split_5utr_study",
    "04_split_3utr_source",
    "05_split_3utr_study",
    "06_split_cross_region",
    "07_leakage_5utr_source",
    "08_leakage_5utr_study",
    "09_leakage_3utr_source",
    "10_leakage_3utr_study",
    "11_leakage_cross_region",
    "12_evaluation_bundle",
    "13_final_acceptance",
)
CRITICAL_RELATIVE_PATHS = (
    "scripts/data/run_b0_production.sh",
    "scripts/execution/b0_driver_guard.py",
    "scripts/execution/run_audited_command.py",
    "scripts/execution/acceptance_semantics.py",
    "scripts/data/build_b0_splits.py",
    "scripts/data/audit_b0_leakage.py",
    "scripts/data/build_b0_evaluation_artifacts.py",
    "scripts/data/validate_b0_acceptance.py",
    "data/utr_benchmark_v2/split_graph.py",
    "data/utr_benchmark_v2/leakage.py",
    "data/utr_benchmark_v2/near_neighbors.py",
    "data/utr_benchmark_v2/track_loader.py",
    "schemas/utr_edit_record.schema.json",
    "schemas/edit_script.schema.json",
    "schemas/generation_task.schema.json",
)
CODE_ROOTS = (
    "scripts/",
    "data/utr_benchmark_v2/",
    "tests/",
    "schemas/",
    "configs/",
    "core/",
    "eval/",
)
REQUIRED_TOOLS = ("git", "df", "ps", "sha256sum")
SUCCESS_INDEX_EXCLUSIONS = frozenset(
    {
        "artifact_checksums.json",
        "driver_completion.json",
        "logs/events.jsonl",
        "status.json",
    }
)
SUCCESS_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "entry_count",
        "entries",
        "entries_sha256",
        "excluded_mutable_or_self_referential_paths",
    }
)
SUCCESS_INDEX_ENTRY_FIELDS = frozenset({"path", "bytes", "sha256"})
TERMINAL_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "updated_at_utc",
        "current_node",
        "wrapper_pid",
        "terminal",
        "driver_completion",
        "terminal_event",
    }
)
TERMINAL_COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "completed_at_utc",
        "attempt_root",
        "audit_node_order",
        "final_acceptance",
        "acceptance_binding",
        "bundle_manifest",
        "bundle_binding",
        "code_manifest",
        "frozen_input_manifest",
        "frozen_input_source_preflight",
        "accepted_result_binding",
        "bundle_result_binding",
        "attempt_manifest",
        "events_snapshot",
        "terminal_event",
        "named_gate_set_validation",
        "final_artifact_graph_validation",
        "final_acceptance_recheck",
        "artifact_checksum_index",
        "authoritative_only_when_terminal_status_ref_matches",
        "sealable_for_post_acceptance_release",
        "post_acceptance_git_release_chain_required",
        "stage_completion_claimed",
        "scientific_result_claimed",
    }
)
TERMINAL_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "at_utc",
        "event",
        "attempt_root",
        "final_acceptance",
        "failure_evidence_present",
        "authoritative_only_when_terminal_status_ref_matches",
        "stage_completion_claimed",
        "scientific_result_claimed",
    }
)
SEAL_READY_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "at_utc",
        "event",
        "stage_completion_claimed",
    }
)
FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "failed_at_utc",
        "exit_code",
        "reason",
        "current_node",
        "signal",
        "line",
        "command",
        "wrapper_pid",
        "evidence_preserved",
        "unrelated_processes_terminated",
        "failure_id",
    }
)
FAILURE_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "updated_at_utc",
        "current_node",
        "wrapper_pid",
        "terminal",
        "reason",
        "exit_code",
        "failure",
    }
)
FAILURE_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "at_utc",
        "event",
        "node",
        "reason",
        "exit_code",
        "signal",
        "failure_id",
    }
)
TERMINAL_COMPLETION_FILE_REFS = {
    "final_acceptance": "artifacts/acceptance.json",
    "acceptance_binding": "provenance/acceptance_binding.json",
    "bundle_manifest": "artifacts/bundle/build_manifest.json",
    "bundle_binding": "provenance/bundle_binding.json",
    "code_manifest": "provenance/code_manifest.json",
    "frozen_input_manifest": "provenance/input_manifest.json",
    "frozen_input_source_preflight": "artifacts/preflight.json",
    "attempt_manifest": "attempt_manifest.json",
    "events_snapshot": "provenance/events_at_terminal.jsonl",
    "terminal_event": "provenance/terminal_success_event.json",
    "named_gate_set_validation": "provenance/named_gate_set_validation.json",
    "final_artifact_graph_validation": (
        "provenance/final_artifact_graph_validation.json"
    ),
    "final_acceptance_recheck": ("provenance/final_acceptance_recheck_manifest.json"),
    "artifact_checksum_index": "artifact_checksums.json",
}
EXPECTED_INPUT_CHECK_LABELS = (
    "00_preflight.after",
    *(
        label
        for node in EXPECTED_AUDIT_NODES[1:]
        for label in (f"{node}.before", f"{node}.after")
    ),
    "13_final_seal.before",
)
EXPECTED_FINGERPRINT_LABELS = (
    *(
        label
        for node in EXPECTED_AUDIT_NODES
        for label in (f"{node}.before", f"{node}.after")
    ),
    "13_final_seal.before",
)
NODE_ENTRYPOINTS = {
    "00_preflight": "b0_driver_guard.py",
    "01_canonical_validation": "build_b0_splits.py",
    "02_split_5utr_source": "build_b0_splits.py",
    "03_split_5utr_study": "build_b0_splits.py",
    "04_split_3utr_source": "build_b0_splits.py",
    "05_split_3utr_study": "build_b0_splits.py",
    "06_split_cross_region": "build_b0_splits.py",
    "07_leakage_5utr_source": "audit_b0_leakage.py",
    "08_leakage_5utr_study": "audit_b0_leakage.py",
    "09_leakage_3utr_source": "audit_b0_leakage.py",
    "10_leakage_3utr_study": "audit_b0_leakage.py",
    "11_leakage_cross_region": "audit_b0_leakage.py",
    "12_evaluation_bundle": "build_b0_evaluation_artifacts.py",
    "13_final_acceptance": "validate_b0_acceptance.py",
}
NAMED_GATE_NAMES = (
    "audit-completion",
    "audit-git-binding",
    "preflight",
    "canonical-validation",
    "split-common",
    "split-5utr-source-role",
    "split-5utr-study-role",
    "split-3utr-source-role",
    "split-3utr-study-role",
    "split-cross-region-role",
    "leakage",
    "evaluation-bundle",
    "final-acceptance",
)
SPLIT_BASENAMES = (
    "5utr_source_disjoint.json",
    "5utr_study_disjoint.json",
    "3utr_source_disjoint.json",
    "3utr_study_disjoint.json",
    "cross_region_transfer.json",
)
MANDATORY_BUNDLE_OUTPUTS = frozenset(
    {
        "artifact_bindings.json",
        "docs/data/UTR_EditBench_v2_Data_Card.md",
        "evaluation/claims/allowed_unsupported_claims.yaml",
        "evaluation/tracks/closed_measured_pool.yaml",
        "evaluation/tracks/heldout_generative.yaml",
        "evaluation/tracks/open_legal_generation.yaml",
        "evaluation/tracks/track_role_matrix.yaml",
    }
)
TERMINAL_EVENTS = frozenset(
    {"B0_DRIVER_COMPLETED", "FAILED_WITH_EVIDENCE", "SAFE_PAUSED"}
)
LEAKAGE_ACCEPTANCE_GATE_NAMES = frozenset(
    {
        "unexplained_overlap_zero",
        "exact_source_overlap_zero",
        "exact_candidate_overlap_zero",
        "reverse_edge_leakage_zero",
        "path_leakage_zero",
        "near_neighbor_leakage_zero",
        "final_endpoint_as_train_intermediate_zero",
        "required_axis_overlap_zero",
        "foundation_overlap_gate",
    }
)
RUNTIME_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        "runtime_prefix",
        "python_bin",
        "python_version",
        "implementation",
        "base_prefix",
        "project_root",
        "task_resource",
        "non_neural",
        "cuda_required",
        "packages",
        "project_imports",
        "overlay_file",
    }
)
RUNTIME_PACKAGE_MODULES = {
    "jsonschema": "jsonschema",
    "PyYAML": "yaml",
    "numpy": "numpy",
}


class GuardError(RuntimeError):
    """A fail-closed driver guard rejection."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_lexical_directory_no_follow(path: Path, *, label: str) -> int:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    try:
        before = absolute_path.lstat()
        resolved = absolute_path.resolve(strict=True)
    except OSError as exc:
        raise GuardError(f"{label} is unavailable: {absolute_path}: {exc}") from exc
    if (
        resolved != absolute_path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
    ):
        raise GuardError(f"{label} is not a lexical directory: {absolute_path}")
    try:
        descriptor = os.open(
            absolute_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise GuardError(
            f"{label} cannot be opened safely: {absolute_path}: {exc}"
        ) from exc
    after = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(after.st_mode)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
    ):
        os.close(descriptor)
        raise GuardError(f"{label} changed during no-follow open: {absolute_path}")
    return descriptor


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    parent_descriptor = _open_lexical_directory_no_follow(
        absolute_path.parent,
        label="exclusive-write parent",
    )
    try:
        descriptor = os.open(
            absolute_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o640,
            dir_fd=parent_descriptor,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    _write_bytes_exclusive(path, _json_bytes(payload))


def _write_json_atomic(path: Path, payload: Any) -> None:
    if not path.is_absolute():
        raise GuardError(f"atomic write target must be absolute: {path}")
    try:
        parent_mode = path.parent.lstat().st_mode
        target_before = path.lstat()
    except OSError as exc:
        raise GuardError(f"atomic write target is unavailable: {path}: {exc}") from exc
    if (
        not stat.S_ISDIR(parent_mode)
        or path.parent.resolve(strict=True) != path.parent
        or stat.S_ISLNK(target_before.st_mode)
        or not stat.S_ISREG(target_before.st_mode)
        or target_before.st_nlink != 1
    ):
        raise GuardError(f"atomic write target is not a lexical regular file: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    _write_bytes_exclusive(temporary, _json_bytes(payload))
    parent_descriptor = _open_lexical_directory_no_follow(
        path.parent,
        label="atomic-write parent",
    )
    try:
        try:
            target_after = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise GuardError(
                f"atomic write target changed before replace: {path}: {exc}"
            ) from exc
        if (
            target_after.st_dev != target_before.st_dev
            or target_after.st_ino != target_before.st_ino
            or stat.S_ISLNK(target_after.st_mode)
            or not stat.S_ISREG(target_after.st_mode)
            or target_after.st_nlink != 1
        ):
            raise GuardError(f"atomic write target changed before replace: {path}")
        os.replace(
            temporary.name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _open_existing_regular_no_follow(
    path: Path,
    flags: int,
    *,
    label: str,
) -> int:
    if not path.is_absolute():
        raise GuardError(f"{label} path must be absolute: {path}")
    parent_descriptor = _open_lexical_directory_no_follow(
        path.parent,
        label=f"{label} parent",
    )
    try:
        try:
            before = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise GuardError(f"{label} is unavailable: {path}: {exc}") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise GuardError(
                f"{label} is not a lexical single-link regular file: {path}"
            )
        try:
            descriptor = os.open(
                path.name,
                flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise GuardError(f"{label} cannot be opened safely: {path}: {exc}") from exc
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
        ):
            os.close(descriptor)
            raise GuardError(f"{label} changed during no-follow open: {path}")
        return descriptor
    finally:
        os.close(parent_descriptor)


def _append_json_line(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = _open_existing_regular_no_follow(
        path,
        os.O_WRONLY | os.O_APPEND,
        label="JSONL append target",
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise GuardError(f"short append to {path}: {written}/{len(encoded)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _terminal_lock(attempt_root: Path) -> Iterable[None]:
    lock_path = attempt_root / "terminal.lock"
    descriptor = _open_existing_regular_no_follow(
        lock_path,
        os.O_RDWR,
        label="attempt terminal lock",
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_sha256_sidecar(path: Path, artifact: Path) -> str:
    digest = _sha256_file(artifact)
    _write_bytes_exclusive(
        path,
        f"{digest}  {artifact.name}\n".encode("ascii"),
    )
    return digest


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError(
            f"invalid JSON object {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GuardError(f"expected JSON object: {path}")
    return payload


def _get(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise GuardError(f"missing JSON field: {'.'.join(path)}")
        current = current[key]
    return current


def _require_bool(payload: Mapping[str, Any], path: Sequence[str], value: bool) -> None:
    observed = _get(payload, path)
    if observed is not value:
        raise GuardError(
            f"required {'.'.join(path)} == {value!r}, observed {observed!r}"
        )


def _require_json_equal(
    payload: Mapping[str, Any],
    path: Sequence[str],
    expected: Any,
) -> None:
    observed = _get(payload, path)
    matches = type(observed) is type(expected) and observed == expected
    if not matches:
        raise GuardError(
            f"required {'.'.join(path)} == {expected!r}, observed {observed!r}"
        )


def _require_json_array(payload: Mapping[str, Any], path: Sequence[str]) -> list[Any]:
    observed = _get(payload, path)
    if not isinstance(observed, list):
        raise GuardError(f"required JSON array: {'.'.join(path)}")
    return observed


def _require_strict_numeric_one(
    payload: Mapping[str, Any],
    path: Sequence[str],
) -> None:
    observed = _get(payload, path)
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or observed != 1
    ):
        raise GuardError(
            f"required strict numeric {'.'.join(path)} == 1, observed " f"{observed!r}"
        )


def _validate_d1_acceptance_payload(payload: Mapping[str, Any]) -> None:
    errors = validate_phase_acceptance("D1", payload, require_pass=True)
    if errors:
        raise GuardError(
            "D1 acceptance semantic validation failed: " + "; ".join(errors)
        )


def _runtime_probe(python_bin: Path, project_root: Path) -> dict[str, Any]:
    probe = r"""
import importlib
import importlib.metadata
import json
import pathlib
import platform
import sys

packages = {}
for distribution, module_name in (
    ("jsonschema", "jsonschema"),
    ("PyYAML", "yaml"),
    ("numpy", "numpy"),
):
    module = importlib.import_module(module_name)
    packages[distribution] = {
        "distribution": distribution,
        "version": importlib.metadata.version(distribution),
        "module": module_name,
        "module_file": str(pathlib.Path(module.__file__).resolve(strict=True)),
    }
data_module = importlib.import_module("data")
print(json.dumps({
    "python_bin": sys.executable,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "runtime_prefix": str(pathlib.Path(sys.prefix).resolve(strict=True)),
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    "packages": packages,
    "project_imports": {
        "data": {
            "module": "data",
            "module_file": str(
                pathlib.Path(data_module.__file__).resolve(strict=True)
            ),
        },
    },
    }, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python_bin), "-B", "-c", probe],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GuardError(
            "runtime live import probe failed: "
            f"exit={completed.returncode}, stderr={completed.stderr.strip()!r}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise GuardError("runtime live import probe must emit exactly one JSON line")
    try:
        observed = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise GuardError(
            f"runtime live import probe emitted invalid JSON: {exc}"
        ) from exc
    if not isinstance(observed, dict):
        raise GuardError("runtime live import probe did not emit an object")
    return observed


def _validate_runtime_manifest_file(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_python_bin: str,
    expected_runtime_prefix: Path,
    expected_project_root: Path,
) -> dict[str, Any]:
    expected_sha256 = _require_hex(
        expected_sha256,
        64,
        "expected runtime-manifest SHA256",
    )
    if _sha256_file(manifest_path) != expected_sha256:
        raise GuardError("runtime manifest differs from the approved SHA256")
    payload = _load_object(manifest_path)
    if set(payload) != RUNTIME_MANIFEST_FIELDS:
        raise GuardError(
            "runtime manifest top-level field set is not exact: "
            f"observed={sorted(payload)!r}"
        )
    if payload.get("schema_version") != "b0-runtime-manifest-v1":
        raise GuardError("runtime manifest schema_version is invalid")
    created_at = payload.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise GuardError("runtime manifest created_at_utc must be UTC Z time")
    try:
        parsed_created_at = datetime.fromisoformat(
            created_at.removesuffix("Z") + "+00:00"
        )
    except ValueError as exc:
        raise GuardError("runtime manifest created_at_utc is invalid") from exc
    if parsed_created_at.utcoffset() != timezone.utc.utcoffset(parsed_created_at):
        raise GuardError("runtime manifest created_at_utc is not UTC")
    if not (
        payload.get("implementation") == "CPython"
        and payload.get("task_resource") == "CPU_HEAVY"
        and payload.get("non_neural") is True
        and payload.get("cuda_required") is False
    ):
        raise GuardError("runtime manifest workload boundary is invalid")

    manifest_python = payload.get("python_bin")
    if (
        not isinstance(manifest_python, str)
        or not Path(manifest_python).is_absolute()
        or manifest_python != expected_python_bin
    ):
        raise GuardError("runtime manifest python_bin differs from the CLI launcher")
    python_bin = Path(manifest_python)
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise GuardError("runtime manifest python_bin is not executable")

    def exact_directory(field: str, expected: Path) -> Path:
        raw = payload.get(field)
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise GuardError(f"runtime manifest {field} must be absolute")
        observed = _absolute_directory(raw, kind=f"runtime manifest {field}")
        if observed != expected.resolve(strict=True):
            raise GuardError(f"runtime manifest {field} differs from approval")
        return observed

    runtime_prefix = exact_directory("runtime_prefix", expected_runtime_prefix)
    project_root = exact_directory("project_root", expected_project_root)
    base_prefix_raw = payload.get("base_prefix")
    if not isinstance(base_prefix_raw, str) or not Path(base_prefix_raw).is_absolute():
        raise GuardError("runtime manifest base_prefix must be absolute")
    _absolute_directory(base_prefix_raw, kind="runtime manifest base_prefix")
    python_version = payload.get("python_version")
    if not isinstance(python_version, str) or not python_version:
        raise GuardError("runtime manifest python_version must be a string")

    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(
        RUNTIME_PACKAGE_MODULES
    ):
        raise GuardError("runtime manifest package set is not exact")
    for distribution, module_name in RUNTIME_PACKAGE_MODULES.items():
        item = packages.get(distribution)
        if not isinstance(item, Mapping) or set(item) != {
            "distribution",
            "version",
            "module",
            "module_file",
        }:
            raise GuardError(
                f"runtime manifest package fields are not exact: {distribution}"
            )
        if (
            item.get("distribution") != distribution
            or item.get("module") != module_name
            or not isinstance(item.get("version"), str)
            or not item.get("version")
        ):
            raise GuardError(
                f"runtime manifest package identity is invalid: {distribution}"
            )
        module_file = item.get("module_file")
        if not isinstance(module_file, str) or not Path(module_file).is_absolute():
            raise GuardError(
                f"runtime manifest module_file is not absolute: {distribution}"
            )
        _absolute_file(
            module_file,
            kind=f"runtime manifest module_file {distribution}",
        )

    project_imports = payload.get("project_imports")
    if not isinstance(project_imports, Mapping) or set(project_imports) != {"data"}:
        raise GuardError("runtime manifest project_imports set is not exact")
    data_import = project_imports.get("data")
    if not isinstance(data_import, Mapping) or set(data_import) != {
        "module",
        "module_file",
    }:
        raise GuardError("runtime manifest data import fields are not exact")
    expected_data_module = (project_root / "data/__init__.py").resolve(strict=True)
    if data_import.get("module") != "data" or data_import.get("module_file") != str(
        expected_data_module
    ):
        raise GuardError("runtime manifest data import is not bound to project_root")

    overlay = payload.get("overlay_file")
    if not isinstance(overlay, Mapping) or set(overlay) != {"path", "sha256"}:
        raise GuardError("runtime manifest overlay_file fields are not exact")
    overlay_raw = overlay.get("path")
    if not isinstance(overlay_raw, str) or not Path(overlay_raw).is_absolute():
        raise GuardError("runtime manifest overlay path must be absolute")
    overlay_path = _absolute_file(overlay_raw, kind="runtime manifest overlay")
    overlay_sha = _require_hex(
        overlay.get("sha256"),
        64,
        "runtime manifest overlay SHA256",
    )
    if _sha256_file(overlay_path) != overlay_sha:
        raise GuardError("runtime manifest overlay file hash changed")

    observed = _runtime_probe(python_bin, project_root)
    expected_live = {
        "python_bin": manifest_python,
        "python_version": python_version,
        "implementation": "CPython",
        "runtime_prefix": str(runtime_prefix),
        "base_prefix": str(
            _absolute_directory(
                base_prefix_raw,
                kind="runtime manifest base_prefix",
            )
        ),
        "packages": dict(packages),
        "project_imports": dict(project_imports),
    }
    if observed != expected_live:
        raise GuardError("runtime live probe differs from the frozen runtime manifest")
    return {
        "schema_version": "b0_runtime_manifest_validation.v1",
        "manifest": _path_ref(manifest_path),
        "overlay_file": _path_ref(overlay_path),
        "live_probe": observed,
        "passed": True,
    }


def _require_hex(value: Any, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GuardError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _require_utc_z(value: Any, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise GuardError(f"{name} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise GuardError(f"{name} must be a UTC Z timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GuardError(f"{name} must be a UTC Z timestamp")
    return parsed


def _absolute_existing(path_text: str, *, kind: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise GuardError(f"{kind} must be an absolute path: {path_text!r}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GuardError(f"{kind} is unavailable: {path_text!r}: {exc}") from exc
    return resolved


def _absolute_file(path_text: str, *, kind: str) -> Path:
    path = _absolute_existing(path_text, kind=kind)
    if not path.is_file():
        raise GuardError(f"{kind} is not a regular file: {path}")
    return path


def _absolute_directory(path_text: str, *, kind: str) -> Path:
    path = _absolute_existing(path_text, kind=kind)
    if not path.is_dir():
        raise GuardError(f"{kind} is not a directory: {path}")
    return path


def _absolute_lexical_directory(path_text: str, *, kind: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise GuardError(f"{kind} must be an absolute path: {path_text!r}")
    try:
        mode = path.lstat().st_mode
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GuardError(f"{kind} is unavailable: {path_text!r}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise GuardError(f"{kind} must be a lexical non-symlink directory: {path}")
    if resolved != path:
        raise GuardError(
            f"{kind} path contains a symlink or non-canonical component: {path}"
        )
    return path


def _run_text(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip()
        raise GuardError(f"command failed ({completed.returncode}): {argv!r}: {error}")
    return completed.stdout.strip()


def _run_bytes(argv: Sequence[str], *, cwd: Path | None = None) -> bytes:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GuardError(f"command failed ({completed.returncode}): {argv!r}: {error}")
    return completed.stdout


def _artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True))
        rendered = relative.as_posix()
    except ValueError:
        rendered = str(resolved)
    return {
        "path": rendered,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _path_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _overlaps(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def _require_no_overlap(left: Path, right: Path, label: str) -> dict[str, Any]:
    overlap = _overlaps(left, right)
    if overlap:
        raise GuardError(f"forbidden path overlap ({label}): {left} <-> {right}")
    return {
        "label": label,
        "left": str(left),
        "right": str(right),
        "overlap": False,
    }


def _canonical_json_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _strict_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return bool(left == right)


def _fingerprint_payload(project_root: Path, driver_path: Path) -> dict[str, Any]:
    captured = _capture_git_snapshot(project_root)
    index_flags_raw = _run_bytes(
        ("git", "-C", str(project_root), "ls-files", "-v", "-z")
    )
    index_flag_entries: list[dict[str, str]] = []
    invalid_index_flags: list[str] = []
    for raw_entry in index_flags_raw.split(b"\0"):
        if not raw_entry:
            continue
        if len(raw_entry) < 3 or raw_entry[1:2] != b" ":
            raise GuardError("git ls-files -v returned an invalid entry")
        tag = raw_entry[:1].decode("ascii", errors="strict")
        relative = os.fsdecode(raw_entry[2:])
        index_flag_entries.append({"path": relative, "tag": tag})
        if tag != "H":
            invalid_index_flags.append(f"{tag} {relative}")
    tracked_critical = set(
        _run_text(
            (
                "git",
                "-C",
                str(project_root),
                "ls-files",
                "--full-name",
                "--",
                *CRITICAL_RELATIVE_PATHS,
            )
        ).splitlines()
    )
    missing_tracked = sorted(set(CRITICAL_RELATIVE_PATHS) - tracked_critical)
    critical_entries: list[dict[str, Any]] = []
    critical_integrity_passed = not missing_tracked
    for relative in CRITICAL_RELATIVE_PATHS:
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            critical_integrity_passed = False
            critical_entries.append(
                {
                    "path": relative,
                    "tracked": relative in tracked_critical,
                    "regular_non_symlink": False,
                    "matches_head_blob": False,
                }
            )
            continue
        head_bytes = _run_bytes(
            ("git", "-C", str(project_root), "show", f"HEAD:{relative}")
        )
        live_sha256 = _sha256_file(path)
        head_sha256 = _sha256_bytes(head_bytes)
        matches_head = live_sha256 == head_sha256 and path.stat().st_size == len(
            head_bytes
        )
        critical_integrity_passed = (
            critical_integrity_passed and relative in tracked_critical and matches_head
        )
        critical_entries.append(
            {
                "path": relative,
                "tracked": relative in tracked_critical,
                "regular_non_symlink": True,
                "matches_head_blob": matches_head,
                "bytes": path.stat().st_size,
                "sha256": live_sha256,
                "head_blob_bytes": len(head_bytes),
                "head_blob_sha256": head_sha256,
            }
        )
    untracked_code = [
        entry
        for entry in captured["untracked_manifest"]["entries"]
        if any(str(entry["path"]).startswith(prefix) for prefix in CODE_ROOTS)
    ]
    core = {
        "schema_version": "b0_code_freeze.v1",
        "project_root": str(project_root),
        "head": captured["head"],
        "clean": captured["clean"],
        "git_dirty_state_sha256": captured["dirty_state_sha256"],
        "git_index_flags": {
            "policy": "ALL_TRACKED_ENTRIES_TAG_H",
            "entry_count": len(index_flag_entries),
            "entries": index_flag_entries,
            "raw_sha256": _sha256_bytes(index_flags_raw),
            "invalid_entries": invalid_index_flags,
            "passed": not invalid_index_flags,
        },
        "tracked_diff": {
            "bytes": len(captured["diff_bytes"]),
            "sha256": captured["component_hashes"]["diff_head_binary"],
        },
        "untracked_all": {
            "entry_count": captured["untracked_manifest"]["entry_count"],
            "entries_sha256": captured["untracked_manifest"]["entries_sha256"],
            "manifest_sha256": captured["component_hashes"][
                "untracked_content_manifest"
            ],
        },
        "untracked_code": {
            "entry_count": len(untracked_code),
            "entries": untracked_code,
            "entries_sha256": _canonical_json_sha({"entries": untracked_code}),
        },
        "critical_entries": critical_entries,
        "critical_integrity": {
            "missing_tracked_paths": missing_tracked,
            "all_tracked_regular_and_equal_to_head": critical_integrity_passed,
            "passed": critical_integrity_passed,
        },
        "driver": _path_ref(driver_path),
    }
    return {**core, "fingerprint_sha256": _canonical_json_sha(core)}


def _fingerprint_command(args: argparse.Namespace) -> int:
    project_root = _absolute_directory(args.project_root, kind="project root")
    driver_path = _absolute_file(args.driver_path, kind="driver")
    output = Path(args.output)
    payload = _fingerprint_payload(project_root, driver_path)
    expected_commit = _require_hex(
        args.expected_commit,
        40,
        "expected commit",
    )
    expected_driver_sha256 = _require_hex(
        args.expected_driver_sha256,
        64,
        "expected driver SHA256",
    )
    expected_dirty_state_sha256 = _require_hex(
        args.expected_dirty_state_sha256,
        64,
        "expected dirty-state SHA256",
    )
    checks = {
        "head": payload["head"] == expected_commit,
        "driver_sha256": payload["driver"]["sha256"] == expected_driver_sha256,
        "dirty_state_sha256": (
            payload["git_dirty_state_sha256"] == expected_dirty_state_sha256
        ),
        "clean_committed_code": payload["clean"] is True,
        "standard_git_index_flags": (
            _get(payload, ("git_index_flags", "passed")) is True
        ),
        "critical_files_equal_head": (
            _get(payload, ("critical_integrity", "passed")) is True
        ),
    }
    payload["caller_approval"] = {
        "expected_commit": expected_commit,
        "expected_driver_sha256": expected_driver_sha256,
        "expected_dirty_state_sha256": expected_dirty_state_sha256,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json_exclusive(output, payload)
    digest = _write_sha256_sidecar(Path(args.sha256_output), output)
    print(digest)
    if not payload["caller_approval"]["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise GuardError(
            "code fingerprint failed caller approval checks: " + ", ".join(failed)
        )
    return 0


def _assert_fingerprint_command(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    expected_baseline_sha256 = _require_hex(
        args.expected_baseline_sha256,
        64,
        "expected code-manifest SHA256",
    )
    if _sha256_file(baseline_path) != expected_baseline_sha256:
        raise GuardError("code manifest changed after its in-memory approval")
    baseline = _load_object(baseline_path)
    project_root = _absolute_directory(args.project_root, kind="project root")
    driver_path = _absolute_file(args.driver_path, kind="driver")
    observed = _fingerprint_payload(project_root, driver_path)
    fingerprint_matches = observed.get("fingerprint_sha256") == baseline.get(
        "fingerprint_sha256"
    )
    integrity_passed = (
        _get(observed, ("git_index_flags", "passed")) is True
        and _get(observed, ("critical_integrity", "passed")) is True
    )
    observed["comparison"] = {
        "label": args.label,
        "baseline": _path_ref(baseline_path.resolve(strict=True)),
        "expected_baseline_sha256": expected_baseline_sha256,
        "fingerprint_matches": fingerprint_matches,
        "integrity_passed": integrity_passed,
        "matches": fingerprint_matches and integrity_passed,
    }
    _write_json_exclusive(Path(args.observed_output), observed)
    if not observed["comparison"]["matches"]:
        raise GuardError(
            "code freeze drift: "
            f"{args.label}: baseline={baseline.get('fingerprint_sha256')!r}, "
            f"observed={observed.get('fingerprint_sha256')!r}"
        )
    return 0


def _freeze_inputs_command(args: argparse.Namespace) -> int:
    preflight_path = _absolute_file(args.preflight, kind="B0 preflight")
    stdout_path = _absolute_file(args.node00_stdout, kind="node00 stdout")
    preflight = _load_object(preflight_path)
    if preflight.get("status") != "PASS":
        raise GuardError("B0 preflight is not PASS")
    stdout_lines = stdout_path.read_text(encoding="utf-8").splitlines()
    if len(stdout_lines) != 1:
        raise GuardError("node00 stdout must contain exactly one JSON line")
    try:
        stdout_payload = json.loads(stdout_lines[0])
    except json.JSONDecodeError as exc:
        raise GuardError(f"node00 stdout is not valid JSON: {exc}") from exc
    if stdout_payload != preflight:
        raise GuardError("node00 stdout does not exactly equal live preflight JSON")
    d1 = _get(preflight, ("d1",))
    if not isinstance(d1, Mapping):
        raise GuardError("B0 preflight d1 field must be an object")
    frozen_inputs: dict[str, Any] = {}
    for key in (
        "acceptance",
        "build_manifest",
        "canonical_label_store",
        "sealed_label_free_candidate_store",
        "exposure_ledger",
    ):
        reference = d1.get(key)
        if not isinstance(reference, Mapping):
            raise GuardError(f"B0 preflight omits frozen D1 reference: {key}")
        path = _absolute_file(
            str(reference.get("path", "")),
            kind=f"frozen D1 {key}",
        )
        frozen_inputs[key] = _validate_ref(
            path,
            reference,
            label=f"frozen D1 {key}",
        )
    runtime_reference = _get(preflight, ("runtime", "manifest"))
    if not isinstance(runtime_reference, Mapping):
        raise GuardError("B0 preflight omits frozen runtime-manifest reference")
    runtime_manifest_path = _absolute_file(
        str(runtime_reference.get("path", "")),
        kind="frozen runtime manifest",
    )
    frozen_inputs["runtime_manifest"] = _validate_ref(
        runtime_manifest_path,
        runtime_reference,
        label="frozen runtime manifest",
    )
    acceptance_path = _absolute_file(
        str(frozen_inputs["acceptance"]["path"]),
        kind="frozen D1 acceptance",
    )
    _validate_d1_acceptance_payload(_load_object(acceptance_path))
    d1_acceptance_sha256 = _require_hex(
        frozen_inputs["acceptance"]["sha256"],
        64,
        "frozen D1 acceptance SHA256",
    )
    payload = {
        "schema_version": "b0_frozen_input_manifest.v1",
        "created_at_utc": _utc_now(),
        "preflight": _path_ref(preflight_path),
        "node00_stdout": _path_ref(stdout_path),
        "d1_acceptance_sha256": d1_acceptance_sha256,
        "inputs": frozen_inputs,
    }
    output = Path(args.output)
    _write_json_exclusive(output, payload)
    digest = _write_sha256_sidecar(Path(args.sha256_output), output)
    print(digest)
    return 0


def _validated_input_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_sha256 = _require_hex(
        expected_sha256,
        64,
        "expected frozen-input-manifest SHA256",
    )
    if _sha256_file(path) != expected_sha256:
        raise GuardError("frozen input manifest changed after in-memory approval")
    manifest = _load_object(path)
    if set(manifest) != {
        "schema_version",
        "created_at_utc",
        "preflight",
        "node00_stdout",
        "d1_acceptance_sha256",
        "inputs",
    }:
        raise GuardError("frozen input manifest field set is not exact")
    if manifest.get("schema_version") != "b0_frozen_input_manifest.v1":
        raise GuardError("frozen input manifest schema is invalid")
    _require_utc_z(manifest.get("created_at_utc"), "frozen input manifest created_at")
    preflight_ref = manifest.get("preflight")
    stdout_ref = manifest.get("node00_stdout")
    if not (
        isinstance(preflight_ref, Mapping)
        and set(preflight_ref) == SUCCESS_INDEX_ENTRY_FIELDS
        and isinstance(stdout_ref, Mapping)
        and set(stdout_ref) == SUCCESS_INDEX_ENTRY_FIELDS
    ):
        raise GuardError("frozen input manifest source refs are invalid")
    preflight_path = _absolute_file(
        str(preflight_ref.get("path", "")),
        kind="bound B0 preflight",
    )
    stdout_path = _absolute_file(
        str(stdout_ref.get("path", "")),
        kind="bound node00 stdout",
    )
    _validate_ref(preflight_path, preflight_ref, label="bound B0 preflight")
    _validate_ref(stdout_path, stdout_ref, label="bound node00 stdout")
    preflight = _load_object(preflight_path)
    stdout_lines = stdout_path.read_text(encoding="utf-8").splitlines()
    if len(stdout_lines) != 1 or json.loads(stdout_lines[0]) != preflight:
        raise GuardError("bound node00 stdout no longer equals bound preflight")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise GuardError("frozen input manifest inputs field is invalid")
    expected_input_names = {
        "acceptance",
        "build_manifest",
        "canonical_label_store",
        "sealed_label_free_candidate_store",
        "exposure_ledger",
        "runtime_manifest",
    }
    if set(inputs) != expected_input_names:
        raise GuardError("frozen input manifest input set is not exact")
    for input_name, reference in inputs.items():
        if (
            not isinstance(reference, Mapping)
            or set(reference) != SUCCESS_INDEX_ENTRY_FIELDS
        ):
            raise GuardError(
                f"frozen input manifest ref field set is invalid: {input_name}"
            )
    acceptance_ref = inputs.get("acceptance")
    if not isinstance(acceptance_ref, Mapping):
        raise GuardError("frozen input manifest acceptance ref is invalid")
    declared_acceptance_sha = _require_hex(
        manifest.get("d1_acceptance_sha256"),
        64,
        "frozen D1 acceptance SHA256",
    )
    if acceptance_ref.get("sha256") != declared_acceptance_sha:
        raise GuardError("frozen input manifest D1 acceptance hashes are inconsistent")
    acceptance_path = _absolute_file(
        str(acceptance_ref.get("path", "")),
        kind="frozen D1 acceptance",
    )
    _validate_ref(
        acceptance_path,
        acceptance_ref,
        label="frozen D1 acceptance",
    )
    _validate_d1_acceptance_payload(_load_object(acceptance_path))
    runtime_ref = inputs.get("runtime_manifest")
    preflight_runtime_ref = _get(preflight, ("runtime", "manifest"))
    if (
        not isinstance(runtime_ref, Mapping)
        or not isinstance(preflight_runtime_ref, Mapping)
        or runtime_ref != preflight_runtime_ref
    ):
        raise GuardError("frozen runtime manifest differs from bound preflight")
    runtime_manifest_path = _absolute_file(
        str(runtime_ref.get("path", "")),
        kind="frozen runtime manifest",
    )
    _validate_ref(
        runtime_manifest_path,
        runtime_ref,
        label="frozen runtime manifest",
    )
    expected_python_bin = _get(
        preflight,
        ("runtime", "launcher_path_preserved_without_final_symlink_resolution"),
    )
    if not isinstance(expected_python_bin, str):
        raise GuardError("bound preflight runtime launcher is invalid")
    expected_runtime_prefix = _absolute_directory(
        str(_get(preflight, ("approved", "runtime_prefix"))),
        kind="bound preflight runtime prefix",
    )
    expected_project_root = _absolute_directory(
        str(_get(preflight, ("git", "top_level"))),
        kind="bound preflight project root",
    )
    _validate_runtime_manifest_file(
        runtime_manifest_path,
        expected_sha256=runtime_ref.get("sha256"),
        expected_python_bin=expected_python_bin,
        expected_runtime_prefix=expected_runtime_prefix,
        expected_project_root=expected_project_root,
    )
    return manifest, inputs


def _assert_inputs_command(args: argparse.Namespace) -> int:
    input_manifest_path = _absolute_file(
        args.input_manifest,
        kind="frozen input manifest",
    )
    input_manifest, frozen_inputs = _validated_input_manifest(
        input_manifest_path,
        args.expected_input_manifest_sha256,
    )
    observed: dict[str, Any] = {}
    for key in (
        "acceptance",
        "build_manifest",
        "canonical_label_store",
        "sealed_label_free_candidate_store",
        "exposure_ledger",
        "runtime_manifest",
    ):
        reference = frozen_inputs.get(key)
        if not isinstance(reference, Mapping):
            raise GuardError(f"frozen input manifest omits D1 reference: {key}")
        path = _absolute_file(str(reference.get("path", "")), kind=f"frozen D1 {key}")
        observed[key] = _validate_ref(path, reference, label=f"frozen D1 {key}")
    payload = {
        "schema_version": "b0_frozen_inputs_check.v1",
        "label": args.label,
        "checked_at_utc": _utc_now(),
        "input_manifest": _path_ref(input_manifest_path),
        "input_manifest_sha256": args.expected_input_manifest_sha256,
        "source_preflight": input_manifest["preflight"],
        "source_node00_stdout": input_manifest["node00_stdout"],
        "inputs": observed,
        "matches": True,
    }
    _write_json_exclusive(Path(args.output), payload)
    return 0


def _validate_b0_acceptance_hard_gate(payload: Mapping[str, Any]) -> None:
    required_checks = (
        payload.get("schema_version") == "utr_b0_acceptance.v2",
        payload.get("b0_gate_passed") is True,
        payload.get("failed_gates") == [],
        payload.get("allowed_claim") == "NONE",
        payload.get("requires_fm0_reaudit") is True,
        payload.get("re_audit_required_before_foundation_use") is True,
        payload.get("supplied_leakage_reports_match_recomputation") is True,
        _get(payload, ("exposure_ledger", "gate_passed")) is True,
        _get(payload, ("track_role_audit", "gate_passed")) is True,
        _get(payload, ("track_a_label_seal_audit", "gate_passed")) is True,
        _get(
            payload,
            ("track_a_label_seal_audit", "role_policy_exact_binding_passed"),
        )
        is True,
        _get(
            payload,
            ("track_a_label_seal_audit", "current_d1_chain_binding_passed"),
        )
        is True,
        _get(payload, ("required_artifact_audit", "gate_passed")) is True,
        _get(payload, ("d1_exposure_ledger_binding", "gate_passed")) is True,
    )
    if not all(required_checks):
        raise GuardError("B0 acceptance fails the production driver hard gates")
    _require_strict_numeric_one(payload, ("exposure_ledger", "coverage"))


def _validate_b0_acceptance_payload(payload: Mapping[str, Any]) -> None:
    _validate_b0_acceptance_hard_gate(payload)
    errors = validate_phase_acceptance("B0", payload, require_pass=True)
    if errors:
        raise GuardError(
            "B0 acceptance semantic validation failed: " + "; ".join(errors)
        )


def _validate_b0_bundle_payload(payload: Mapping[str, Any]) -> None:
    required_checks = (
        payload.get("schema_version") == "utr_b0_evaluation_artifact_build.v2",
        payload.get("status") == "PASS",
        _get(payload, ("acceptance_preview", "b0_gate_passed")) is True,
        _get(payload, ("acceptance_preview", "failed_gates")) == [],
        _get(
            payload,
            (
                "leakage_evidence_binding",
                "supplied_reports_exactly_match_recomputation",
            ),
        )
        is True,
        _get(payload, ("track_a_label_seal_audit", "gate_passed")) is True,
        _get(
            payload,
            ("track_a_label_seal_audit", "role_policy_exact_binding_passed"),
        )
        is True,
        _get(
            payload,
            ("track_a_label_seal_audit", "current_d1_chain_binding_passed"),
        )
        is True,
        _get(payload, ("required_artifact_audit", "gate_passed")) is True,
        _get(payload, ("d1_exposure_ledger_binding", "gate_passed")) is True,
        _get(payload, ("full_d1_binding", "passed")) is True,
        payload.get("scientific_result_claimed") is False,
        payload.get("foundation_status") == "UNKNOWN_PENDING_FM0",
    )
    if not all(required_checks):
        raise GuardError("B0 bundle manifest fails the production driver hard gates")


def _validate_named_gate(
    name: str,
    payload: Mapping[str, Any],
    *,
    expected_head: str | None,
    expected_dirty_state_sha256: str | None,
    expected_d1_acceptance: str | None,
    expected_canonical_validation: str | None,
) -> dict[str, str]:
    context: dict[str, str] = {}
    if name == "audit-completion":
        _require_json_equal(payload, ("state",), "COMMAND_COMPLETED")
        _require_json_equal(payload, ("observed_process_exit_code",), 0)
        _require_json_equal(payload, ("wrapper_exit_code",), 0)
        _require_json_equal(payload, ("stop_reason",), None)
    elif name == "audit-git-binding":
        if expected_head is None or expected_dirty_state_sha256 is None:
            raise GuardError(
                "audit-git-binding requires expected HEAD and dirty-state SHA256"
            )
        expected_head = _require_hex(expected_head, 40, "gate expected HEAD")
        expected_dirty_state_sha256 = _require_hex(
            expected_dirty_state_sha256,
            64,
            "gate expected dirty-state SHA256",
        )
        _require_json_equal(payload, ("state",), "COMMAND_COMPLETED")
        _require_json_equal(
            payload,
            ("git_prelaunch_snapshot", "head"),
            expected_head,
        )
        _require_json_equal(
            payload,
            ("git_prelaunch_snapshot", "dirty_state_sha256"),
            expected_dirty_state_sha256,
        )
        context = {
            "expected_head": expected_head,
            "expected_dirty_state_sha256": expected_dirty_state_sha256,
        }
    elif name == "preflight":
        _require_json_equal(payload, ("schema_version",), "b0_preflight.v1")
        _require_json_equal(payload, ("status",), "PASS")
        _require_json_equal(
            payload,
            ("workload_class",),
            "NON_NEURAL_DATA_BENCHMARK",
        )
        _require_json_equal(payload, ("formal_neural_activity",), False)
        _require_json_equal(
            payload,
            ("path_topology", "attempt_parent_is_exact_approved_parent"),
            True,
        )
        checks = _require_json_array(payload, ("path_topology", "checks"))
        if not all(
            isinstance(check, Mapping) and check.get("overlap") is False
            for check in checks
        ):
            raise GuardError("preflight path topology includes an overlap")
        _require_json_equal(payload, ("disk", "passed"), True)
        _require_json_equal(
            payload,
            ("d1", "exposure_ledger_path_is_absolute"),
            True,
        )
        _require_json_equal(
            payload,
            ("claim_boundary", "stage_completion_claimed"),
            False,
        )
    elif name == "canonical-validation":
        _require_json_equal(
            payload,
            ("schema_version",),
            "utr_b0_canonical_schema_validation.v2",
        )
        _require_json_equal(payload, ("status",), "PASS")
        _require_json_equal(payload, ("invalid_record_count",), 0)
        _require_json_equal(payload, ("d1_acceptance_bound",), True)
        _require_json_equal(payload, ("d1_binding", "passed"), True)
        _require_json_equal(payload, ("legacy_schema_only_validation",), False)
    elif name == "split-common":
        if expected_d1_acceptance is None or expected_canonical_validation is None:
            raise GuardError(
                "split-common requires expected D1 acceptance and canonical "
                "validation paths"
            )
        _require_json_equal(payload, ("status",), "READY")
        _require_json_equal(payload, ("d1_phase_gate_passed",), True)
        _require_json_equal(
            payload,
            ("d1_acceptance_path",),
            expected_d1_acceptance,
        )
        _require_json_equal(
            payload,
            ("canonical_validation_report_path",),
            expected_canonical_validation,
        )
        partitions = _require_json_array(payload, ("partitions",))
        if not partitions:
            raise GuardError("split-common requires a non-empty partitions array")
        if not all(
            isinstance(partition, Mapping) and partition.get("status") == "READY"
            for partition in partitions
        ):
            raise GuardError("split-common contains a partition that is not READY")
        partitions_sha256 = _get(payload, ("partitions_sha256",))
        if not isinstance(partitions_sha256, str):
            raise GuardError("split-common partitions_sha256 must be a string")
        _require_hex(
            partitions_sha256,
            64,
            "split-common partitions_sha256",
        )
        context = {
            "expected_d1_acceptance": expected_d1_acceptance,
            "expected_canonical_validation": expected_canonical_validation,
        }
    elif name == "split-5utr-source-role":
        _require_json_equal(payload, ("split_kind",), "source_disjoint")
        _require_json_equal(payload, ("region",), "five_utr")
    elif name == "split-5utr-study-role":
        _require_json_equal(payload, ("split_kind",), "study_disjoint")
        _require_json_equal(payload, ("region",), "five_utr")
    elif name == "split-3utr-source-role":
        _require_json_equal(payload, ("split_kind",), "source_disjoint")
        _require_json_equal(payload, ("region",), "three_utr")
    elif name == "split-3utr-study-role":
        _require_json_equal(payload, ("split_kind",), "study_disjoint")
        _require_json_equal(payload, ("region",), "three_utr")
    elif name == "split-cross-region-role":
        _require_json_equal(payload, ("split_kind",), "cross_region_transfer")
        _require_json_equal(payload, ("source_region",), "five_utr")
        _require_json_equal(payload, ("target_region",), "three_utr")
    elif name == "leakage":
        _require_json_equal(payload, ("gate_passed",), True)
        _require_json_equal(
            payload,
            ("recomputed_from_bound_structural_records",),
            True,
        )
        _require_json_equal(
            payload,
            ("canonical_manifest_exact_recomputation",),
            True,
        )
        _require_json_equal(
            payload,
            ("foundation_pretraining_overlap", "status"),
            "UNKNOWN_PENDING_FM0",
        )
        _require_json_equal(
            payload,
            ("foundation_pretraining_overlap", "foundation_selected"),
            False,
        )
        _require_json_equal(
            payload,
            ("foundation_pretraining_overlap", "allowed_claim"),
            "NONE",
        )
        _require_json_equal(
            payload,
            ("foundation_pretraining_overlap", "re_audit_required"),
            True,
        )
        acceptance_gates = _get(payload, ("acceptance_gates",))
        if (
            not isinstance(acceptance_gates, Mapping)
            or set(acceptance_gates) != LEAKAGE_ACCEPTANCE_GATE_NAMES
            or not all(value is True for value in acceptance_gates.values())
        ):
            raise GuardError(
                "leakage acceptance_gates must be the exact production "
                "gate-name to true mapping"
            )
        for field in (
            "unexplained_overlap_count",
            "exact_source_leakage_count",
            "exact_candidate_leakage_count",
            "reverse_edge_leakage_count",
            "path_leakage_count",
            "near_neighbor_leakage_count",
            "final_endpoint_as_train_intermediate_count",
            "required_axis_overlap_count",
        ):
            _require_json_equal(payload, ("counts", field), 0)
    elif name == "evaluation-bundle":
        _validate_b0_bundle_payload(payload)
    elif name == "final-acceptance":
        _validate_b0_acceptance_payload(payload)
    else:
        raise GuardError(f"unsupported named gate: {name}")
    return context


def _validate_gate_command(args: argparse.Namespace) -> int:
    artifact_path = Path(args.artifact)
    evidence_path = Path(args.evidence_output)
    evidence: dict[str, Any] = {
        "schema_version": "b0_named_gate_evidence.v1",
        "checked_at_utc": _utc_now(),
        "label": args.label,
        "gate": args.gate,
        "artifact_path": str(artifact_path),
        "passed": False,
    }
    try:
        artifact = _absolute_file(args.artifact, kind=f"{args.gate} gate artifact")
        payload = _load_object(artifact)
        context = _validate_named_gate(
            args.gate,
            payload,
            expected_head=args.expected_head,
            expected_dirty_state_sha256=args.expected_dirty_state_sha256,
            expected_d1_acceptance=args.expected_d1_acceptance,
            expected_canonical_validation=args.expected_canonical_validation,
        )
        evidence.update(
            {
                "artifact": _path_ref(artifact),
                "context": context,
                "passed": True,
            }
        )
    except (GuardError, OSError, ValueError, KeyError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(evidence_path, evidence)
        raise
    _write_json_exclusive(evidence_path, evidence)
    return 0


def _validate_result_payload(kind: str, payload: Mapping[str, Any]) -> None:
    if kind == "acceptance":
        _validate_b0_acceptance_payload(payload)
    elif kind == "bundle":
        _validate_b0_bundle_payload(payload)
    else:
        raise GuardError(f"unsupported B0 result binding kind: {kind}")


def _bind_result_command(args: argparse.Namespace) -> int:
    artifact = _absolute_file(args.artifact, kind=f"B0 {args.kind}")
    payload = _load_object(artifact)
    _validate_result_payload(args.kind, payload)
    binding = {
        "schema_version": "b0_result_binding.v1",
        "created_at_utc": _utc_now(),
        "kind": args.kind,
        "artifact": _path_ref(artifact),
        "semantic_validation": {
            "validator": (
                "acceptance_semantics.validate_phase_acceptance"
                if args.kind == "acceptance"
                else "b0_driver_guard.bundle_hard_gates"
            ),
            "require_pass": True,
            "passed": True,
        },
    }
    output = Path(args.output)
    _write_json_exclusive(output, binding)
    digest = _write_sha256_sidecar(Path(args.sha256_output), output)
    print(digest)
    return 0


def _validate_result_binding_document(
    binding: Mapping[str, Any],
    *,
    expected_kind: str,
    live_artifact: Path,
) -> dict[str, Any]:
    if set(binding) != {
        "schema_version",
        "created_at_utc",
        "kind",
        "artifact",
        "semantic_validation",
    }:
        raise GuardError(f"{expected_kind} binding field set is not exact")
    if (
        binding.get("schema_version") != "b0_result_binding.v1"
        or binding.get("kind") != expected_kind
    ):
        raise GuardError(f"{expected_kind} binding schema or kind is invalid")
    _require_utc_z(
        binding.get("created_at_utc"),
        f"{expected_kind} binding created_at_utc",
    )
    artifact_ref = binding.get("artifact")
    if (
        not isinstance(artifact_ref, Mapping)
        or set(artifact_ref) != SUCCESS_INDEX_ENTRY_FIELDS
        or artifact_ref.get("path") != str(live_artifact.resolve(strict=True))
    ):
        raise GuardError(f"{expected_kind} binding artifact ref is invalid")
    _validate_ref(
        live_artifact.resolve(strict=True),
        artifact_ref,
        label=f"bound B0 {expected_kind}",
    )
    semantic_validation = binding.get("semantic_validation")
    expected_validator = (
        "acceptance_semantics.validate_phase_acceptance"
        if expected_kind == "acceptance"
        else "b0_driver_guard.bundle_hard_gates"
    )
    if not (
        isinstance(semantic_validation, Mapping)
        and set(semantic_validation) == {"validator", "require_pass", "passed"}
        and semantic_validation.get("validator") == expected_validator
        and semantic_validation.get("require_pass") is True
        and semantic_validation.get("passed") is True
    ):
        raise GuardError(f"{expected_kind} binding semantic validation is not exact")
    _validate_result_payload(expected_kind, _load_object(live_artifact))
    return dict(binding)


def _validated_result_binding(
    *,
    binding_path: Path,
    expected_binding_sha256: str,
    expected_kind: str,
    live_artifact: Path,
) -> dict[str, Any]:
    expected_binding_sha256 = _require_hex(
        expected_binding_sha256,
        64,
        f"expected {expected_kind} binding SHA256",
    )
    if _sha256_file(binding_path) != expected_binding_sha256:
        raise GuardError(f"{expected_kind} binding changed after in-memory approval")
    binding = _load_object(binding_path)
    return _validate_result_binding_document(
        binding,
        expected_kind=expected_kind,
        live_artifact=live_artifact,
    )


def _validate_ref(
    path: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise GuardError(f"{label} is missing: {path}")
    expected_sha = _require_hex(expected.get("sha256"), 64, f"{label}.sha256")
    try:
        expected_bytes = expected["bytes"]
    except KeyError as exc:
        raise GuardError(f"{label}.bytes is invalid") from exc
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise GuardError(f"{label}.bytes is invalid")
    observed = _path_ref(path)
    if observed["sha256"] != expected_sha or observed["bytes"] != expected_bytes:
        raise GuardError(
            f"{label} binding mismatch: expected bytes={expected_bytes}, "
            f"sha256={expected_sha}; observed={observed}"
        )
    return observed


def _resolve_d1_store(d1_root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise GuardError(f"{label}.path must be a non-empty string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = d1_root / candidate
    return candidate.resolve(strict=True)


def _tool_evidence(tool: str) -> dict[str, Any]:
    resolved = shutil.which(tool)
    if resolved is None:
        raise GuardError(f"required tool is unavailable: {tool}")
    version = "VERSION_QUERY_UNSUPPORTED"
    for option in ("--version", "-V"):
        completed = subprocess.run(
            [resolved, option],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            version = completed.stdout.splitlines()[0].strip()
            break
    return {"name": tool, "path": resolved, "version": version}


def _preflight_command(args: argparse.Namespace) -> int:
    manifest_path = _absolute_file(args.manifest, kind="attempt manifest")
    expected_manifest_sha256 = _require_hex(
        args.expected_manifest_sha256,
        64,
        "expected attempt-manifest SHA256",
    )
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise GuardError("attempt manifest changed after in-memory approval")
    manifest = _load_object(manifest_path)
    paths = _get(manifest, ("paths",))
    expected = _get(manifest, ("expected",))
    limits = _get(manifest, ("limits",))
    if not isinstance(paths, Mapping) or not isinstance(expected, Mapping):
        raise GuardError("attempt manifest paths/expected must be objects")

    project_root = _absolute_directory(
        str(_get(paths, ("project_root",))),
        kind="isolated worktree",
    )
    approved_parent = _absolute_lexical_directory(
        str(_get(paths, ("approved_b0_parent",))),
        kind="approved B0 parent",
    )
    attempt_root = _absolute_lexical_directory(
        str(_get(paths, ("attempt_root",))),
        kind="B0 attempt root",
    )
    if attempt_root.parent != approved_parent:
        raise GuardError(
            "B0 attempt root must be a direct child of the approved B0 parent"
        )
    driver = _absolute_file(str(_get(paths, ("driver",))), kind="B0 driver")
    expected_driver = (project_root / "scripts/data/run_b0_production.sh").resolve(
        strict=True
    )
    if driver != expected_driver:
        raise GuardError(f"driver path mismatch: {driver} != {expected_driver}")
    guard = _absolute_file(str(_get(paths, ("guard",))), kind="B0 driver guard")
    expected_guard = (project_root / "scripts/execution/b0_driver_guard.py").resolve(
        strict=True
    )
    if guard != expected_guard:
        raise GuardError(f"guard path mismatch: {guard} != {expected_guard}")

    git_root = Path(
        _run_text(("git", "-C", str(project_root), "rev-parse", "--show-toplevel"))
    ).resolve(strict=True)
    if git_root != project_root:
        raise GuardError(f"project root is not exact Git top-level: {git_root}")
    head = _run_text(("git", "-C", str(project_root), "rev-parse", "--verify", "HEAD"))
    approved_head = _require_hex(
        _get(expected, ("commit",)),
        40,
        "expected commit",
    )
    if head != approved_head:
        raise GuardError(f"HEAD mismatch: expected {approved_head}, observed {head}")
    approved_driver_sha = _require_hex(
        _get(expected, ("driver_sha256",)),
        64,
        "expected driver SHA256",
    )
    actual_driver_sha = _sha256_file(driver)
    if actual_driver_sha != approved_driver_sha:
        raise GuardError(
            "driver hash mismatch: "
            f"expected {approved_driver_sha}, observed {actual_driver_sha}"
        )
    approved_dirty_state_sha = _require_hex(
        _get(expected, ("dirty_state_sha256",)),
        64,
        "expected dirty-state SHA256",
    )

    git_dir_raw = _run_text(("git", "-C", str(project_root), "rev-parse", "--git-dir"))
    git_dir = Path(git_dir_raw)
    if not git_dir.is_absolute():
        git_dir = project_root / git_dir
    git_dir = git_dir.resolve(strict=True)

    python_launcher_text = str(_get(paths, ("python_launcher",)))
    python_launcher = Path(python_launcher_text)
    if not python_launcher.is_absolute():
        raise GuardError("Python launcher must be absolute")
    if not python_launcher.is_file() or not os.access(python_launcher, os.X_OK):
        raise GuardError(f"Python launcher is unavailable: {python_launcher}")
    runtime_raw = _run_text(
        (
            str(python_launcher),
            "-c",
            (
                "import json,os,platform,sys;"
                "print(json.dumps({"
                "'sys_executable':sys.executable,"
                "'executable_realpath':os.path.realpath(sys.executable),"
                "'sys_prefix':sys.prefix,"
                "'sys_base_prefix':sys.base_prefix,"
                "'python_version':platform.python_version()"
                "},sort_keys=True))"
            ),
        )
    )
    try:
        runtime = json.loads(runtime_raw)
    except json.JSONDecodeError as exc:
        raise GuardError(f"Python runtime probe returned invalid JSON: {exc}") from exc
    expected_prefix = _absolute_directory(
        str(_get(expected, ("runtime_prefix",))),
        kind="expected Python runtime prefix",
    )
    observed_prefix = Path(str(runtime.get("sys_prefix", ""))).resolve(strict=True)
    if observed_prefix != expected_prefix:
        raise GuardError(
            f"Python runtime prefix mismatch: expected {expected_prefix}, "
            f"observed {observed_prefix}"
        )
    runtime_manifest = _absolute_file(
        str(_get(paths, ("runtime_manifest",))),
        kind="B0 runtime manifest",
    )
    expected_runtime_manifest_sha256 = _require_hex(
        _get(expected, ("runtime_manifest_sha256",)),
        64,
        "expected runtime-manifest SHA256",
    )
    runtime_manifest_validation = _validate_runtime_manifest_file(
        runtime_manifest,
        expected_sha256=expected_runtime_manifest_sha256,
        expected_python_bin=python_launcher_text,
        expected_runtime_prefix=expected_prefix,
        expected_project_root=project_root,
    )

    d1_acceptance = _absolute_file(
        str(_get(paths, ("d1_acceptance",))),
        kind="D1 acceptance",
    )
    acceptance = _load_object(d1_acceptance)
    _validate_d1_acceptance_payload(acceptance)

    d1_root_raw = acceptance.get("stage_d1_root")
    if not isinstance(d1_root_raw, str) or not Path(d1_root_raw).is_absolute():
        raise GuardError("D1 stage root must be an absolute path")
    d1_root = _absolute_directory(d1_root_raw, kind="D1 stage root")
    d1_build = _absolute_file(
        str(d1_root / "build_manifest.json"),
        kind="D1 build manifest",
    )
    build_ref = _get(
        acceptance,
        ("required_artifact_validation", "build_manifest"),
    )
    if not isinstance(build_ref, Mapping):
        raise GuardError("D1 accepted build manifest ref must be an object")
    build_ref_path_raw = build_ref.get("path")
    if (
        not isinstance(build_ref_path_raw, str)
        or not Path(build_ref_path_raw).is_absolute()
    ):
        raise GuardError("D1 accepted build manifest path must be absolute")
    if (
        _absolute_file(build_ref_path_raw, kind="accepted D1 build manifest")
        != d1_build
    ):
        raise GuardError("D1 accepted build manifest path does not bind current build")
    build_observed = _validate_ref(d1_build, build_ref, label="D1 build manifest")
    build = _load_object(d1_build)

    canonical_ref = _get(build, ("global_stores", "canonical_label_store"))
    structural_ref = _get(
        build,
        ("global_stores", "sealed_label_free_candidate_store"),
    )
    ledger_ref = _get(
        build,
        ("required_artifacts", "data/data_exposure_ledger.jsonl"),
    )
    if not all(
        isinstance(item, Mapping)
        for item in (canonical_ref, structural_ref, ledger_ref)
    ):
        raise GuardError("D1 store and ledger references must be objects")
    canonical = _resolve_d1_store(
        d1_root,
        canonical_ref.get("path"),
        label="canonical label store",
    )
    structural = _resolve_d1_store(
        d1_root,
        structural_ref.get("path"),
        label="sealed label-free candidate store",
    )
    ledger_raw = ledger_ref.get("path")
    if not isinstance(ledger_raw, str) or not Path(ledger_raw).is_absolute():
        raise GuardError("D1 build exposure ledger path must be absolute")
    ledger = _absolute_file(ledger_raw, kind="D1 exposure ledger")
    canonical_observed = _validate_ref(
        canonical,
        canonical_ref,
        label="canonical label store",
    )
    structural_observed = _validate_ref(
        structural,
        structural_ref,
        label="sealed label-free candidate store",
    )
    ledger_observed = _validate_ref(
        ledger,
        ledger_ref,
        label="D1 build exposure ledger",
    )
    accepted_ledger_ref = _get(
        acceptance,
        (
            "required_artifact_validation",
            "artifacts",
            "data/data_exposure_ledger.jsonl",
        ),
    )
    if not isinstance(accepted_ledger_ref, Mapping):
        raise GuardError("D1 accepted exposure ledger ref must be an object")
    accepted_ledger_raw = accepted_ledger_ref.get("path")
    if (
        not isinstance(accepted_ledger_raw, str)
        or not Path(accepted_ledger_raw).is_absolute()
    ):
        raise GuardError("D1 accepted exposure ledger path must be absolute")
    accepted_ledger = _absolute_file(
        accepted_ledger_raw,
        kind="accepted D1 exposure ledger",
    )
    if accepted_ledger != ledger:
        raise GuardError("D1 accepted exposure ledger does not bind current ledger")
    if (
        _validate_ref(
            accepted_ledger,
            accepted_ledger_ref,
            label="accepted D1 exposure ledger",
        )
        != ledger_observed
    ):
        raise GuardError("D1 build and acceptance ledger bindings differ")

    path_checks = [
        _require_no_overlap(approved_parent, project_root, "approved_parent_vs_repo"),
        _require_no_overlap(attempt_root, project_root, "attempt_vs_repo"),
        _require_no_overlap(attempt_root, git_dir, "attempt_vs_git_dir"),
        _require_no_overlap(
            attempt_root, project_root / "data", "attempt_vs_repo_data"
        ),
        _require_no_overlap(attempt_root, d1_root, "attempt_vs_d1_root"),
        _require_no_overlap(attempt_root, canonical, "attempt_vs_canonical"),
        _require_no_overlap(attempt_root, structural, "attempt_vs_structural"),
        _require_no_overlap(attempt_root, ledger, "attempt_vs_ledger"),
    ]

    minimum_free_bytes = _get(limits, ("minimum_free_bytes",))
    if type(minimum_free_bytes) is not int:
        raise GuardError("minimum_free_bytes must be an integer")
    if minimum_free_bytes <= 0:
        raise GuardError("minimum_free_bytes must be positive")
    disk = shutil.disk_usage(approved_parent)
    if disk.free < minimum_free_bytes:
        raise GuardError(
            f"insufficient B0 artifact disk: free={disk.free}, "
            f"minimum={minimum_free_bytes}"
        )

    tools = [_tool_evidence(tool) for tool in REQUIRED_TOOLS]
    optional_gpu_tool = shutil.which("nvidia-smi")
    gpu_tool = {
        "applicability": "OBSERVATIONAL_ONLY_B0_IS_NON_NEURAL",
        "nvidia_smi_path": optional_gpu_tool,
    }
    if optional_gpu_tool:
        gpu_tool["summary"] = _run_text(
            (
                optional_gpu_tool,
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            )
        ).splitlines()

    code_manifest = _load_object(
        _absolute_file(
            str(_get(paths, ("code_manifest",))),
            kind="B0 code manifest",
        )
    )
    if code_manifest.get("head") != head:
        raise GuardError("B0 code manifest HEAD differs from audited preflight HEAD")
    if code_manifest.get("driver", {}).get("sha256") != actual_driver_sha:
        raise GuardError("B0 code manifest driver hash differs from current driver")
    if code_manifest.get("git_dirty_state_sha256") != approved_dirty_state_sha:
        raise GuardError(
            "B0 code manifest dirty state differs from caller-approved dirty state"
        )
    if code_manifest.get("clean") is not True:
        raise GuardError("formal B0 requires a clean committed isolated worktree")
    if _get(code_manifest, ("git_index_flags", "passed")) is not True:
        raise GuardError("B0 code manifest contains nonstandard Git index flags")
    if _get(code_manifest, ("critical_integrity", "passed")) is not True:
        raise GuardError("B0 critical execution files do not equal committed HEAD")
    caller_approval = code_manifest.get("caller_approval")
    if (
        not isinstance(caller_approval, Mapping)
        or caller_approval.get("passed") is not True
    ):
        raise GuardError("B0 code manifest did not pass all caller approval checks")
    critical_by_path = {
        item.get("path"): item
        for item in code_manifest.get("critical_entries", [])
        if isinstance(item, Mapping)
    }
    for required_relative in (
        "scripts/data/run_b0_production.sh",
        "scripts/execution/b0_driver_guard.py",
    ):
        if required_relative not in critical_by_path:
            raise GuardError(f"code manifest omits {required_relative}")

    output = Path(args.output)
    resolved = {
        "schema_version": "b0_preflight.v1",
        "status": "PASS",
        "completed_at_utc": _utc_now(),
        "workload_class": "NON_NEURAL_DATA_BENCHMARK",
        "formal_neural_activity": False,
        "approved": {
            "isolated_worktree": str(project_root),
            "commit": head,
            "driver_sha256": actual_driver_sha,
            "dirty_state_sha256": approved_dirty_state_sha,
            "runtime_prefix": str(expected_prefix),
            "runtime_manifest_sha256": expected_runtime_manifest_sha256,
            "approved_b0_parent": str(approved_parent),
        },
        "runtime": {
            "launcher_path_preserved_without_final_symlink_resolution": python_launcher_text,
            **runtime,
            "manifest": _path_ref(runtime_manifest),
            "manifest_validation": runtime_manifest_validation,
        },
        "git": {
            "top_level": str(git_root),
            "git_dir": str(git_dir),
            "head": head,
            "code_manifest": _artifact_ref(
                Path(str(_get(paths, ("code_manifest",)))),
                attempt_root,
            ),
        },
        "d1": {
            "acceptance": _path_ref(d1_acceptance),
            "stage_root": str(d1_root),
            "build_manifest": build_observed,
            "canonical_label_store": canonical_observed,
            "sealed_label_free_candidate_store": structural_observed,
            "exposure_ledger": ledger_observed,
            "exposure_ledger_path_is_absolute": True,
        },
        "path_topology": {
            "attempt_root": str(attempt_root),
            "attempt_parent_is_exact_approved_parent": True,
            "checks": path_checks,
        },
        "disk": {
            "path": str(approved_parent),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "minimum_free_bytes": minimum_free_bytes,
            "passed": True,
        },
        "tools": tools,
        "gpu_observation": gpu_tool,
        "claim_boundary": {
            "b0_preflight_is_scientific_acceptance": False,
            "smoke_or_proxy_is_final_scientific_evidence": False,
            "stage_completion_claimed": False,
        },
    }
    _write_json_exclusive(output, resolved)
    print(
        json.dumps(
            resolved,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def _init_command(args: argparse.Namespace) -> int:
    attempt_root = Path(args.attempt_root)
    if not attempt_root.is_absolute():
        raise GuardError("attempt root must be absolute")
    approved_parent = _absolute_lexical_directory(
        args.approved_b0_parent,
        kind="approved B0 parent",
    )
    if attempt_root.parent != approved_parent:
        raise GuardError("attempt root is not a direct child of approved B0 parent")
    if os.path.lexists(attempt_root):
        raise GuardError(f"refusing existing attempt root: {attempt_root}")
    attempt_root.mkdir(mode=0o750, exist_ok=False)
    attempt_root = _absolute_lexical_directory(
        str(attempt_root),
        kind="new B0 attempt root",
    )
    try:
        for relative in (
            "artifacts",
            "audit",
            "failure",
            "logs",
            "provenance",
            "provenance/fingerprints",
            "provenance/gates",
            "provenance/input_checks",
        ):
            (attempt_root / relative).mkdir(mode=0o750, exist_ok=False)
        _write_bytes_exclusive(attempt_root / "terminal.lock", b"")
        _write_bytes_exclusive(attempt_root / "logs/events.jsonl", b"")
        _write_bytes_exclusive(attempt_root / "logs/system_metrics.jsonl", b"")
        _write_json_exclusive(
            attempt_root / "status.json",
            {
                "schema_version": "b0_driver_status.v1",
                "state": "REGISTERED",
                "updated_at_utc": _utc_now(),
                "current_node": None,
                "wrapper_pid": None,
                "terminal": False,
            },
        )
    except Exception as exc:
        try:
            _write_json_exclusive(
                attempt_root / "bootstrap_failure.json",
                {
                    "schema_version": "b0_bootstrap_failure.v1",
                    "failed_at_utc": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "partial_attempt_root_preserved": True,
                },
            )
        except Exception:
            pass
        raise
    manifest = {
        "schema_version": "b0_driver_attempt.v1",
        "attempt_id": attempt_root.name,
        "created_at_utc": _utc_now(),
        "driver_pid": os.getppid(),
        "paths": {
            "project_root": args.project_root,
            "d1_acceptance": args.d1_acceptance,
            "attempt_root": str(attempt_root),
            "approved_b0_parent": str(approved_parent),
            "python_launcher": args.python_launcher,
            "runtime_manifest": args.runtime_manifest,
            "driver": args.driver,
            "guard": args.guard,
            "code_manifest": str(attempt_root / "provenance/code_manifest.json"),
        },
        "expected": {
            "commit": args.expected_commit,
            "driver_sha256": args.expected_driver_sha256,
            "dirty_state_sha256": args.expected_dirty_state_sha256,
            "runtime_prefix": args.expected_runtime_prefix,
            "runtime_manifest_sha256": args.expected_runtime_manifest_sha256,
        },
        "limits": {
            "minimum_free_bytes": args.minimum_free_bytes,
            "watchdog_interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        },
        "execution": {
            "workload_class": "NON_NEURAL_DATA_BENCHMARK",
            "audit_nodes": list(EXPECTED_AUDIT_NODES),
            "business_gate_order_preserved": True,
            "gpu_training_or_validation_started": False,
        },
        "claim_boundary": {
            "driver_completion_is_stage_completion": False,
            "post_acceptance_git_release_chain_required": True,
            "scientific_result_claimed": False,
        },
    }
    manifest_path = attempt_root / "attempt_manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = _write_sha256_sidecar(
        attempt_root / "provenance/attempt_manifest.sha256",
        manifest_path,
    )
    _append_json_line(
        attempt_root / "logs/events.jsonl",
        {
            "schema_version": "b0_driver_event.v1",
            "at_utc": _utc_now(),
            "event": "ATTEMPT_REGISTERED",
            "attempt_id": attempt_root.name,
        },
    )
    print(manifest_sha256)
    return 0


def _event_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    with _terminal_lock(attempt_root):
        status_path = _require_lexical_regular_file(
            attempt_root,
            Path("status.json"),
            label="event status",
        )
        events_path = _require_lexical_regular_file(
            attempt_root,
            Path("logs/events.jsonl"),
            label="event log",
        )
        status = _load_object(status_path)
        if status.get("terminal") is True:
            raise GuardError("refusing event append after terminal status")
        if args.event in TERMINAL_EVENTS:
            raise GuardError("terminal events may only be written by terminal commands")
        payload: dict[str, Any] = {
            "schema_version": "b0_driver_event.v1",
            "at_utc": _utc_now(),
            "event": args.event,
        }
        if args.node is not None:
            payload["node"] = args.node
        if args.detail is not None:
            payload["detail"] = args.detail
        if args.exit_code is not None:
            payload["exit_code"] = args.exit_code
        _append_json_line(events_path, payload)
    return 0


def _status_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    with _terminal_lock(attempt_root):
        status_path = _require_lexical_regular_file(
            attempt_root,
            Path("status.json"),
            label="status transition target",
        )
        current = _load_object(status_path)
        if current.get("terminal") is True:
            raise GuardError(
                "refusing status transition from terminal state "
                f"{current.get('state')!r}"
            )
        if args.terminal:
            raise GuardError(
                "terminal status may only be written by seal/failure commands"
            )
        payload = {
            "schema_version": "b0_driver_status.v1",
            "state": args.state,
            "updated_at_utc": _utc_now(),
            "current_node": args.node,
            "wrapper_pid": args.wrapper_pid,
            "terminal": False,
            "previous_state": current.get("state"),
        }
        if args.reason is not None:
            payload["reason"] = args.reason
        _write_json_atomic(status_path, payload)
    return 0


def _validate_terminal_relative_ref(
    attempt_root: Path,
    reference: Any,
    relative_text: str,
    *,
    label: str,
    absolute_rendering: bool = False,
) -> Path:
    if (
        not isinstance(reference, Mapping)
        or set(reference) != SUCCESS_INDEX_ENTRY_FIELDS
    ):
        raise GuardError(f"{label} reference field set is not exact")
    relative = Path(relative_text)
    path = _require_lexical_regular_file(
        attempt_root.resolve(strict=True),
        relative,
        label=label,
    )
    expected_rendering = str(path) if absolute_rendering else relative.as_posix()
    if (
        type(reference.get("path")) is not str
        or reference.get("path") != expected_rendering
    ):
        raise GuardError(f"{label} reference path is not canonical")
    _validate_ref(path, reference, label=label)
    return path


def _validate_terminal_attempt_manifest(
    attempt_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    if set(manifest) != {
        "schema_version",
        "attempt_id",
        "created_at_utc",
        "driver_pid",
        "paths",
        "expected",
        "limits",
        "execution",
        "claim_boundary",
    }:
        raise GuardError("terminal attempt manifest field set is not exact")
    if (
        manifest.get("schema_version") != "b0_driver_attempt.v1"
        or manifest.get("attempt_id") != attempt_root.name
        or type(manifest.get("driver_pid")) is not int
        or manifest.get("driver_pid") <= 0
    ):
        raise GuardError("terminal attempt manifest identity is invalid")
    _require_utc_z(manifest.get("created_at_utc"), "attempt manifest created_at_utc")

    paths = manifest.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != {
        "project_root",
        "d1_acceptance",
        "attempt_root",
        "approved_b0_parent",
        "python_launcher",
        "runtime_manifest",
        "driver",
        "guard",
        "code_manifest",
    }:
        raise GuardError("terminal attempt manifest paths are not exact")
    if (
        paths.get("attempt_root") != str(attempt_root)
        or paths.get("approved_b0_parent") != str(attempt_root.parent)
        or paths.get("code_manifest")
        != str(attempt_root / "provenance/code_manifest.json")
    ):
        raise GuardError("terminal attempt manifest canonical paths are inconsistent")
    for field in (
        "project_root",
        "d1_acceptance",
        "approved_b0_parent",
        "python_launcher",
        "runtime_manifest",
        "driver",
        "guard",
        "code_manifest",
    ):
        value = paths.get(field)
        if type(value) is not str or not Path(value).is_absolute():
            raise GuardError(f"terminal attempt manifest path is invalid: {field}")

    expected = manifest.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {
        "commit",
        "driver_sha256",
        "dirty_state_sha256",
        "runtime_prefix",
        "runtime_manifest_sha256",
    }:
        raise GuardError("terminal attempt manifest expected fields are not exact")
    _require_hex(expected.get("commit"), 40, "terminal expected commit")
    _require_hex(
        expected.get("driver_sha256"),
        64,
        "terminal expected driver SHA256",
    )
    _require_hex(
        expected.get("dirty_state_sha256"),
        64,
        "terminal expected dirty-state SHA256",
    )
    _require_hex(
        expected.get("runtime_manifest_sha256"),
        64,
        "terminal expected runtime-manifest SHA256",
    )
    if (
        type(expected.get("runtime_prefix")) is not str
        or not Path(expected["runtime_prefix"]).is_absolute()
    ):
        raise GuardError("terminal expected runtime prefix is invalid")

    limits = manifest.get("limits")
    if not (
        isinstance(limits, Mapping)
        and set(limits) == {"minimum_free_bytes", "watchdog_interval_seconds"}
        and type(limits.get("minimum_free_bytes")) is int
        and limits.get("minimum_free_bytes") > 0
        and type(limits.get("watchdog_interval_seconds")) is int
        and limits.get("watchdog_interval_seconds") == WATCHDOG_INTERVAL_SECONDS
    ):
        raise GuardError("terminal attempt manifest limits are invalid")
    execution = manifest.get("execution")
    if not (
        isinstance(execution, Mapping)
        and set(execution)
        == {
            "workload_class",
            "audit_nodes",
            "business_gate_order_preserved",
            "gpu_training_or_validation_started",
        }
        and execution.get("workload_class") == "NON_NEURAL_DATA_BENCHMARK"
        and execution.get("audit_nodes") == list(EXPECTED_AUDIT_NODES)
        and execution.get("business_gate_order_preserved") is True
        and execution.get("gpu_training_or_validation_started") is False
    ):
        raise GuardError("terminal attempt execution boundary is invalid")
    claim_boundary = manifest.get("claim_boundary")
    if not (
        isinstance(claim_boundary, Mapping)
        and set(claim_boundary)
        == {
            "driver_completion_is_stage_completion",
            "post_acceptance_git_release_chain_required",
            "scientific_result_claimed",
        }
        and claim_boundary.get("driver_completion_is_stage_completion") is False
        and claim_boundary.get("post_acceptance_git_release_chain_required") is True
        and claim_boundary.get("scientific_result_claimed") is False
    ):
        raise GuardError("terminal attempt claim boundary is invalid")
    return manifest


def _validate_terminal_recheck_manifest(
    attempt_root: Path,
    manifest_path: Path,
    final_acceptance: Path,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    if set(manifest) != {
        "schema_version",
        "argv",
        "output",
        "stdout",
        "stderr",
        "matches_node13_acceptance",
        "passed",
    }:
        raise GuardError("final acceptance recheck field set is not exact")
    if not (
        manifest.get("schema_version") == "b0_final_acceptance_recheck.v1"
        and manifest.get("matches_node13_acceptance") is True
        and manifest.get("passed") is True
    ):
        raise GuardError("final acceptance recheck state is invalid")

    recheck_paths = {
        "output": attempt_root / "provenance/final_acceptance_recheck.json",
        "stdout": attempt_root / "provenance/final_acceptance_recheck.stdout.log",
        "stderr": attempt_root / "provenance/final_acceptance_recheck.stderr.log",
    }
    for field, path in recheck_paths.items():
        reference = manifest.get(field)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != SUCCESS_INDEX_ENTRY_FIELDS
            or reference.get("path") != str(path)
        ):
            raise GuardError(f"final acceptance recheck {field} ref is invalid")
        _validate_ref(path, reference, label=f"final acceptance recheck {field}")

    audit_manifest = _load_object(
        attempt_root / "audit/13_final_acceptance/audit_manifest.json"
    )
    audit_argv = audit_manifest.get("argv")
    if (
        type(audit_argv) is not list
        or not all(type(value) is str for value in audit_argv)
        or audit_argv.count("--output") != 1
    ):
        raise GuardError("node13 argv is invalid at terminal verification")
    output_index = audit_argv.index("--output") + 1
    if output_index >= len(audit_argv):
        raise GuardError("node13 output argv is incomplete at terminal verification")
    expected_argv = list(audit_argv)
    expected_argv[output_index] = str(recheck_paths["output"])
    if manifest.get("argv") != expected_argv:
        raise GuardError("final acceptance recheck argv differs from node13 provenance")

    rechecked_acceptance = _load_object(recheck_paths["output"])
    _validate_b0_acceptance_payload(rechecked_acceptance)
    if rechecked_acceptance != _load_object(final_acceptance):
        raise GuardError("final acceptance recheck no longer matches final acceptance")
    return manifest


def _validate_terminal_code_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    expected_fields = {
        "schema_version",
        "project_root",
        "head",
        "clean",
        "git_dirty_state_sha256",
        "git_index_flags",
        "tracked_diff",
        "untracked_all",
        "untracked_code",
        "critical_entries",
        "critical_integrity",
        "driver",
        "fingerprint_sha256",
        "caller_approval",
    }
    if set(manifest) != expected_fields:
        raise GuardError("terminal code manifest field set is not exact")
    caller_approval = manifest.get("caller_approval")
    checks = (
        caller_approval.get("checks") if isinstance(caller_approval, Mapping) else None
    )
    expected_checks = {
        "head",
        "driver_sha256",
        "dirty_state_sha256",
        "clean_committed_code",
        "standard_git_index_flags",
        "critical_files_equal_head",
    }
    if not (
        manifest.get("schema_version") == "b0_code_freeze.v1"
        and manifest.get("project_root") == str(project_root)
        and manifest.get("head") == expected.get("commit")
        and manifest.get("git_dirty_state_sha256") == expected.get("dirty_state_sha256")
        and manifest.get("clean") is True
        and _get(manifest, ("git_index_flags", "passed")) is True
        and _get(manifest, ("critical_integrity", "passed")) is True
        and isinstance(caller_approval, Mapping)
        and set(caller_approval)
        == {
            "expected_commit",
            "expected_driver_sha256",
            "expected_dirty_state_sha256",
            "checks",
            "passed",
        }
        and caller_approval.get("expected_commit") == expected.get("commit")
        and caller_approval.get("expected_driver_sha256")
        == expected.get("driver_sha256")
        and caller_approval.get("expected_dirty_state_sha256")
        == expected.get("dirty_state_sha256")
        and isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(checks.get(name) is True for name in expected_checks)
        and caller_approval.get("passed") is True
    ):
        raise GuardError("terminal code manifest approval is invalid")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint_sha256", "caller_approval"}
    }
    if manifest.get("fingerprint_sha256") != _canonical_json_sha(core):
        raise GuardError(
            "terminal code manifest fingerprint is internally inconsistent"
        )
    driver_ref = manifest.get("driver")
    driver_path = (project_root / "scripts/data/run_b0_production.sh").resolve(
        strict=True
    )
    if (
        not isinstance(driver_ref, Mapping)
        or set(driver_ref) != SUCCESS_INDEX_ENTRY_FIELDS
        or driver_ref.get("path") != str(driver_path)
        or driver_ref.get("sha256") != expected.get("driver_sha256")
    ):
        raise GuardError("terminal code manifest driver ref is invalid")
    _validate_ref(driver_path, driver_ref, label="terminal code-manifest driver")
    observed = _fingerprint_payload(project_root, driver_path)
    frozen_payload = {
        key: value for key, value in manifest.items() if key != "caller_approval"
    }
    if not _strict_json_equal(frozen_payload, observed):
        raise GuardError(
            "terminal live code fingerprint differs from the complete frozen manifest"
        )
    return manifest


def _parse_driver_event_log(events_path: Path, *, label: str) -> list[dict[str, Any]]:
    parsed_events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GuardError(f"{label} line {line_number} is invalid: {exc}") from exc
        if not isinstance(event, dict):
            raise GuardError(f"{label} line {line_number} is not an object")
        if (
            event.get("schema_version") != "b0_driver_event.v1"
            or type(event.get("event")) is not str
            or not event.get("event")
        ):
            raise GuardError(f"{label} line {line_number} identity is invalid")
        _require_utc_z(event.get("at_utc"), f"{label} line {line_number} at_utc")
        if event.get("event") not in TERMINAL_EVENTS:
            for claim_field in (
                "stage_completion_claimed",
                "scientific_result_claimed",
            ):
                if claim_field in event and event.get(claim_field) is not False:
                    raise GuardError(
                        f"{label} line {line_number} has invalid {claim_field}"
                    )
        parsed_events.append(event)
    return parsed_events


def _validate_terminal_success_state(attempt_root: Path) -> dict[str, Any]:
    attempt_root = _absolute_lexical_directory(
        str(attempt_root),
        kind="terminal success attempt root",
    )
    status_path = _require_lexical_regular_file(
        attempt_root,
        Path("status.json"),
        label="terminal status",
    )
    completion_path = _require_lexical_regular_file(
        attempt_root,
        Path("driver_completion.json"),
        label="terminal driver completion",
    )
    terminal_event_path = _require_lexical_regular_file(
        attempt_root,
        Path("provenance/terminal_success_event.json"),
        label="terminal success event",
    )
    failure_path = attempt_root / "failure/failure.json"
    if os.path.lexists(failure_path):
        raise GuardError("successful terminal state has failure evidence")

    status = _load_object(status_path)
    completion = _load_object(completion_path)
    terminal_event = _load_object(terminal_event_path)
    if set(status) != TERMINAL_STATUS_FIELDS:
        raise GuardError("successful terminal status field set is not exact")
    if set(completion) != TERMINAL_COMPLETION_FIELDS:
        raise GuardError("driver completion field set is not exact")
    if set(terminal_event) != TERMINAL_EVENT_FIELDS:
        raise GuardError("terminal success event field set is not exact")
    if not (
        status.get("schema_version") == "b0_driver_status.v1"
        and status.get("state") == "B0_DRIVER_COMPLETED"
        and status.get("current_node") is None
        and status.get("wrapper_pid") is None
        and status.get("terminal") is True
        and completion.get("schema_version") == "b0_driver_completion.v1"
        and completion.get("state") == "B0_DRIVER_COMPLETED"
        and completion.get("attempt_root") == str(attempt_root)
        and completion.get("audit_node_order") == list(EXPECTED_AUDIT_NODES)
        and completion.get("authoritative_only_when_terminal_status_ref_matches")
        is True
        and completion.get("sealable_for_post_acceptance_release") is True
        and completion.get("post_acceptance_git_release_chain_required") is True
        and completion.get("stage_completion_claimed") is False
        and completion.get("scientific_result_claimed") is False
        and terminal_event.get("schema_version") == "b0_driver_event.v1"
        and terminal_event.get("event") == "B0_DRIVER_COMPLETED"
        and terminal_event.get("attempt_root") == str(attempt_root)
        and terminal_event.get("failure_evidence_present") is False
        and terminal_event.get("authoritative_only_when_terminal_status_ref_matches")
        is True
        and terminal_event.get("stage_completion_claimed") is False
        and terminal_event.get("scientific_result_claimed") is False
    ):
        raise GuardError("successful terminal documents have inconsistent states")
    _require_utc_z(status.get("updated_at_utc"), "terminal status updated_at_utc")
    _require_utc_z(
        completion.get("completed_at_utc"),
        "driver completion completed_at_utc",
    )
    _require_utc_z(terminal_event.get("at_utc"), "terminal event at_utc")

    completion_ref = status.get("driver_completion")
    terminal_event_ref = status.get("terminal_event")
    _validate_terminal_relative_ref(
        attempt_root,
        completion_ref,
        "driver_completion.json",
        label="terminal driver completion",
    )
    _validate_terminal_relative_ref(
        attempt_root,
        terminal_event_ref,
        "provenance/terminal_success_event.json",
        label="terminal success event",
    )

    completion_paths: dict[str, Path] = {}
    for field, relative_text in TERMINAL_COMPLETION_FILE_REFS.items():
        completion_paths[field] = _validate_terminal_relative_ref(
            attempt_root,
            completion.get(field),
            relative_text,
            label=f"driver completion {field}",
            absolute_rendering=field == "frozen_input_source_preflight",
        )
    if completion.get("terminal_event") != terminal_event_ref:
        raise GuardError("status and completion terminal-event refs differ")
    if terminal_event.get("final_acceptance") != completion.get("final_acceptance"):
        raise GuardError("terminal final-acceptance refs are inconsistent")

    artifact_checksum_index = _validate_success_index(
        attempt_root,
        expected_reference=completion["artifact_checksum_index"],
    )
    checksum_index = _load_object(completion_paths["artifact_checksum_index"])
    indexed_paths = {
        entry["path"]
        for entry in checksum_index["entries"]
        if isinstance(entry, Mapping) and type(entry.get("path")) is str
    }
    required_indexed_paths = set(TERMINAL_COMPLETION_FILE_REFS.values()) - {
        "artifact_checksums.json"
    }
    if not required_indexed_paths.issubset(indexed_paths):
        raise GuardError("terminal completion provenance refs are not fully indexed")

    events_snapshot_path = completion_paths["events_snapshot"]
    events_path = _require_lexical_regular_file(
        attempt_root,
        Path("logs/events.jsonl"),
        label="terminal live event log",
    )
    if events_snapshot_path.read_bytes() != events_path.read_bytes():
        raise GuardError("terminal events snapshot differs from the live event log")
    parsed_events = _parse_driver_event_log(
        events_path,
        label="terminal event log",
    )
    terminal_events = [
        event for event in parsed_events if event.get("event") in TERMINAL_EVENTS
    ]
    if len(terminal_events) != 1 or not _strict_json_equal(
        terminal_events[0],
        terminal_event,
    ):
        raise GuardError("event log does not contain exactly one bound success event")
    seal_ready_events = [
        event for event in parsed_events if event.get("event") == "B0_DRIVER_SEAL_READY"
    ]
    if len(seal_ready_events) != 1:
        raise GuardError("event log must contain exactly one seal-ready event")
    seal_ready = seal_ready_events[0]
    if not (
        set(seal_ready) == SEAL_READY_EVENT_FIELDS
        and seal_ready.get("schema_version") == "b0_driver_event.v1"
        and seal_ready.get("event") == "B0_DRIVER_SEAL_READY"
        and seal_ready.get("stage_completion_claimed") is False
    ):
        raise GuardError("seal-ready event schema or claim boundary is invalid")
    _require_utc_z(seal_ready.get("at_utc"), "seal-ready event at_utc")
    if (
        len(parsed_events) < 2
        or not _strict_json_equal(parsed_events[-2], seal_ready)
        or not _strict_json_equal(parsed_events[-1], terminal_event)
    ):
        raise GuardError(
            "seal-ready must be the final nonterminal event adjacent to success"
        )

    attempt_manifest = _validate_terminal_attempt_manifest(
        attempt_root,
        completion_paths["attempt_manifest"],
    )
    attempt_paths = attempt_manifest["paths"]
    attempt_expected = attempt_manifest["expected"]
    project_root = _absolute_directory(
        attempt_paths["project_root"],
        kind="terminal project root",
    )
    python_launcher = attempt_paths["python_launcher"]
    baseline = _validate_terminal_code_manifest(
        completion_paths["code_manifest"],
        project_root=project_root,
        expected=attempt_expected,
    )
    if (
        baseline.get("head") != attempt_expected["commit"]
        or baseline.get("git_dirty_state_sha256")
        != attempt_expected["dirty_state_sha256"]
    ):
        raise GuardError("terminal code provenance differs from attempt approval")

    frozen_input_manifest = _validate_input_check_set(
        attempt_root=attempt_root,
        input_manifest=completion_paths["frozen_input_manifest"],
        expected_input_manifest_sha256=completion["frozen_input_manifest"]["sha256"],
    )
    if completion.get("frozen_input_source_preflight") != frozen_input_manifest.get(
        "preflight"
    ):
        raise GuardError("terminal frozen preflight differs from frozen input source")
    frozen_inputs = frozen_input_manifest["inputs"]
    d1_acceptance_ref = frozen_inputs["acceptance"]
    d1_acceptance = _absolute_file(
        d1_acceptance_ref["path"],
        kind="terminal frozen D1 acceptance",
    )

    acceptance_binding = _load_object(completion_paths["acceptance_binding"])
    bundle_binding = _load_object(completion_paths["bundle_binding"])
    if completion.get("accepted_result_binding") != acceptance_binding:
        raise GuardError("embedded acceptance binding differs from indexed provenance")
    if completion.get("bundle_result_binding") != bundle_binding:
        raise GuardError("embedded bundle binding differs from indexed provenance")
    _validate_result_binding_document(
        acceptance_binding,
        expected_kind="acceptance",
        live_artifact=completion_paths["final_acceptance"],
    )
    _validate_result_binding_document(
        bundle_binding,
        expected_kind="bundle",
        live_artifact=completion_paths["bundle_manifest"],
    )

    recomputed_named_gate_set = _validate_named_gate_evidence_set(
        attempt_root=attempt_root,
        expected_head=attempt_expected["commit"],
        expected_dirty_state_sha256=attempt_expected["dirty_state_sha256"],
        d1_acceptance=d1_acceptance,
        parsed_events=parsed_events,
    )
    if (
        _load_object(completion_paths["named_gate_set_validation"])
        != recomputed_named_gate_set
    ):
        raise GuardError("terminal named-gate provenance differs from recomputation")

    exposure_ledger_ref = frozen_inputs["exposure_ledger"]
    expected_exposure_ledger = _absolute_file(
        exposure_ledger_ref["path"],
        kind="terminal frozen D1 exposure ledger",
    )
    recomputed_artifact_graph = _validate_final_artifact_graph(
        attempt_root=attempt_root,
        bundle_manifest=completion_paths["bundle_manifest"],
        final_acceptance=completion_paths["final_acceptance"],
        expected_exposure_ledger=expected_exposure_ledger,
    )
    if (
        _load_object(completion_paths["final_artifact_graph_validation"])
        != recomputed_artifact_graph
    ):
        raise GuardError("terminal artifact graph differs from recomputation")
    _validate_terminal_recheck_manifest(
        attempt_root,
        completion_paths["final_acceptance_recheck"],
        completion_paths["final_acceptance"],
    )

    return {
        "status": _path_ref(status_path),
        "completion": _path_ref(completion_path),
        "terminal_event": _path_ref(terminal_event_path),
        "artifact_checksum_index": artifact_checksum_index,
    }


def _ensure_terminal_success_event_logged(attempt_root: Path) -> None:
    events_path = _require_lexical_regular_file(
        attempt_root.resolve(strict=True),
        Path("logs/events.jsonl"),
        label="terminal recovery event log",
    )
    terminal_event = _load_object(
        _require_lexical_regular_file(
            attempt_root.resolve(strict=True),
            Path("provenance/terminal_success_event.json"),
            label="terminal recovery success event",
        )
    )
    parsed_events = _parse_driver_event_log(
        events_path,
        label="terminal recovery event log",
    )
    observed_terminal_events = [
        event for event in parsed_events if event.get("event") in TERMINAL_EVENTS
    ]
    if not observed_terminal_events:
        _append_json_line(events_path, terminal_event)
    elif len(observed_terminal_events) != 1 or not _strict_json_equal(
        observed_terminal_events[0],
        terminal_event,
    ):
        raise GuardError("terminal recovery refuses conflicting terminal events")


def _terminal_success_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    with _terminal_lock(attempt_root):
        status_path = _require_lexical_regular_file(
            attempt_root,
            Path("status.json"),
            label="terminal success status",
        )
        status = _load_object(status_path)
        if (
            status.get("state") == "B0_DRIVER_COMPLETED"
            and status.get("terminal") is True
        ):
            _ensure_terminal_success_event_logged(attempt_root)
        payload = _validate_terminal_success_state(attempt_root)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


def _require_optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise GuardError(f"{name} must be null or a positive integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise GuardError(f"{name} must be a positive integer")
    return value


def _validate_failure_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != FAILURE_FIELDS:
        raise GuardError("failure evidence field set is not exact")
    state = payload.get("state")
    signal_name = payload.get("signal")
    if (
        payload.get("schema_version") != "b0_driver_failure.v1"
        or type(state) is not str
        or state not in {"FAILED_WITH_EVIDENCE", "SAFE_PAUSED"}
    ):
        raise GuardError("failure evidence schema or state is invalid")
    _require_utc_z(payload.get("failed_at_utc"), "failure evidence failed_at_utc")
    exit_code = payload.get("exit_code")
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise GuardError("failure evidence exit_code must be an integer in 0..255")
    reason = payload.get("reason")
    if type(reason) is not str or not reason:
        raise GuardError("failure evidence reason must be a non-empty string")
    current_node = payload.get("current_node")
    if current_node is not None and (
        type(current_node) is not str or current_node not in EXPECTED_AUDIT_NODES
    ):
        raise GuardError("failure evidence current_node is invalid")
    if signal_name is not None and (
        type(signal_name) is not str or signal_name not in {"INT", "TERM", "HUP"}
    ):
        raise GuardError("failure evidence signal is invalid")
    if (state == "SAFE_PAUSED") is not (signal_name is not None):
        raise GuardError("failure evidence state and signal are inconsistent")
    _require_optional_positive_int(payload.get("line"), "failure evidence line")
    command = payload.get("command")
    if command is not None and (type(command) is not str or not command):
        raise GuardError("failure evidence command must be null or a non-empty string")
    _require_optional_positive_int(
        payload.get("wrapper_pid"),
        "failure evidence wrapper_pid",
    )
    if (
        payload.get("evidence_preserved") is not True
        or type(payload.get("unrelated_processes_terminated")) is not int
        or payload.get("unrelated_processes_terminated") != 0
    ):
        raise GuardError("failure evidence preservation boundary is invalid")
    failure_id = _require_hex(
        payload.get("failure_id"),
        64,
        "failure evidence failure_id",
    )
    core = {key: value for key, value in payload.items() if key != "failure_id"}
    if failure_id != _canonical_json_sha(core):
        raise GuardError("failure evidence failure_id is not canonical")
    return dict(payload)


def _validate_recoverable_nonterminal_status(status: Mapping[str, Any]) -> None:
    base_fields = {
        "schema_version",
        "state",
        "updated_at_utc",
        "current_node",
        "wrapper_pid",
        "terminal",
    }
    state = status.get("state")
    if state == "REGISTERED":
        expected_fields = base_fields
    elif state in {"RUNNING", "PREFLIGHT_PASSED"}:
        expected_fields = base_fields | {"previous_state"}
        if "reason" in status:
            expected_fields.add("reason")
    else:
        raise GuardError("failure recovery status state is not recoverable")
    if set(status) != expected_fields:
        raise GuardError("failure recovery nonterminal status field set is not exact")
    if (
        status.get("schema_version") != "b0_driver_status.v1"
        or type(state) is not str
        or status.get("terminal") is not False
    ):
        raise GuardError("failure recovery nonterminal status is invalid")
    _require_utc_z(status.get("updated_at_utc"), "failure recovery status updated_at")
    current_node = status.get("current_node")
    wrapper_pid = status.get("wrapper_pid")
    if state == "RUNNING":
        if type(current_node) is not str or current_node not in EXPECTED_AUDIT_NODES:
            raise GuardError("failure recovery RUNNING node is invalid")
        _require_optional_positive_int(
            wrapper_pid,
            "failure recovery RUNNING wrapper_pid",
        )
    elif current_node is not None or wrapper_pid is not None:
        raise GuardError("failure recovery idle status has process fields")
    if "previous_state" in status:
        previous_state = status.get("previous_state")
        if type(previous_state) is not str or previous_state not in {
            "REGISTERED",
            "RUNNING",
            "PREFLIGHT_PASSED",
        }:
            raise GuardError("failure recovery previous_state is invalid")
    if "reason" in status and (
        type(status.get("reason")) is not str or not status.get("reason")
    ):
        raise GuardError("failure recovery status reason is invalid")


def _failure_event_payload(
    failure: Mapping[str, Any],
    *,
    at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": "b0_driver_event.v1",
        "at_utc": at_utc,
        "event": failure["state"],
        "node": failure["current_node"],
        "reason": failure["reason"],
        "exit_code": failure["exit_code"],
        "signal": failure["signal"],
        "failure_id": failure["failure_id"],
    }


def _validate_failure_event(
    event: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> None:
    if set(event) != FAILURE_EVENT_FIELDS:
        raise GuardError("terminal failure event field set is not exact")
    if event.get("schema_version") != "b0_driver_event.v1" or event.get(
        "event"
    ) != failure.get("state"):
        raise GuardError("terminal failure event identity is invalid")
    _require_utc_z(event.get("at_utc"), "terminal failure event at_utc")
    for event_field, failure_field in (
        ("node", "current_node"),
        ("reason", "reason"),
        ("exit_code", "exit_code"),
        ("signal", "signal"),
        ("failure_id", "failure_id"),
    ):
        if not _strict_json_equal(
            event.get(event_field),
            failure.get(failure_field),
        ):
            raise GuardError(
                f"terminal failure event {event_field} differs from failure evidence"
            )


def _validate_terminal_failure_state(attempt_root: Path) -> dict[str, Any]:
    attempt_root = _absolute_lexical_directory(
        str(attempt_root),
        kind="terminal failure attempt root",
    )
    status_path = _require_lexical_regular_file(
        attempt_root,
        Path("status.json"),
        label="terminal failure status",
    )
    failure_path = _require_lexical_regular_file(
        attempt_root,
        Path("failure/failure.json"),
        label="terminal failure evidence",
    )
    events_path = _require_lexical_regular_file(
        attempt_root,
        Path("logs/events.jsonl"),
        label="terminal failure event log",
    )
    failure = _validate_failure_payload(_load_object(failure_path))
    status = _load_object(status_path)
    if set(status) != FAILURE_STATUS_FIELDS:
        raise GuardError("terminal failure status field set is not exact")
    if not (
        status.get("schema_version") == "b0_driver_status.v1"
        and status.get("state") == failure["state"]
        and status.get("terminal") is True
    ):
        raise GuardError("terminal failure status identity is invalid")
    _require_utc_z(status.get("updated_at_utc"), "terminal failure status updated_at")
    for field in ("current_node", "wrapper_pid", "reason", "exit_code"):
        if not _strict_json_equal(status.get(field), failure.get(field)):
            raise GuardError(f"terminal failure status {field} is inconsistent")
    _validate_terminal_relative_ref(
        attempt_root,
        status.get("failure"),
        "failure/failure.json",
        label="terminal failure evidence",
    )

    parsed_events = _parse_driver_event_log(
        events_path,
        label="terminal failure event log",
    )
    terminal_events = [
        event for event in parsed_events if event.get("event") in TERMINAL_EVENTS
    ]
    if len(terminal_events) != 1:
        raise GuardError("terminal failure event must be unique")
    terminal_event = terminal_events[0]
    _validate_failure_event(terminal_event, failure)
    if not parsed_events or not _strict_json_equal(parsed_events[-1], terminal_event):
        raise GuardError("terminal failure event must be the final event")
    return {
        "status": _path_ref(status_path),
        "failure": _path_ref(failure_path),
        "terminal_event": terminal_event,
    }


def _failure_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    with _terminal_lock(attempt_root):
        status_path = _require_lexical_regular_file(
            attempt_root,
            Path("status.json"),
            label="failure recovery status",
        )
        status = _load_object(status_path)
        failure_dir = attempt_root / "failure"
        try:
            failure_dir_mode = failure_dir.lstat().st_mode
        except OSError as exc:
            raise GuardError(
                f"failure recovery directory is unavailable: {failure_dir}: {exc}"
            ) from exc
        if not stat.S_ISDIR(failure_dir_mode):
            raise GuardError(
                f"failure recovery directory is not a lexical directory: {failure_dir}"
            )
        failure_path = failure_dir / "failure.json"
        if status.get("terminal") is True:
            if (
                status.get("state") == "B0_DRIVER_COMPLETED"
                and (attempt_root / "driver_completion.json").is_file()
                and not os.path.lexists(failure_path)
            ):
                _ensure_terminal_success_event_logged(attempt_root)
                _validate_terminal_success_state(attempt_root)
                return 0
            if status.get("state") in {"FAILED_WITH_EVIDENCE", "SAFE_PAUSED"}:
                _validate_terminal_failure_state(attempt_root)
                return 0
            raise GuardError("attempt has an inconsistent pre-existing terminal state")

        _validate_recoverable_nonterminal_status(status)
        state = "SAFE_PAUSED" if args.signal is not None else "FAILED_WITH_EVIDENCE"
        if os.path.lexists(failure_path):
            failure_path = _require_lexical_regular_file(
                attempt_root,
                Path("failure/failure.json"),
                label="failure recovery evidence",
            )
            payload = _validate_failure_payload(_load_object(failure_path))
            state = payload["state"]
        else:
            core = {
                "schema_version": "b0_driver_failure.v1",
                "state": state,
                "failed_at_utc": _utc_now(),
                "exit_code": args.exit_code,
                "reason": args.reason,
                "current_node": args.node,
                "signal": args.signal,
                "line": args.line,
                "command": args.command,
                "wrapper_pid": args.wrapper_pid,
                "evidence_preserved": True,
                "unrelated_processes_terminated": 0,
            }
            payload = {
                **core,
                "failure_id": _canonical_json_sha(core),
            }
            _validate_failure_payload(payload)
            _write_json_exclusive(failure_path, payload)

        events_path = _require_lexical_regular_file(
            attempt_root,
            Path("logs/events.jsonl"),
            label="failure recovery event log",
        )
        parsed_events = _parse_driver_event_log(
            events_path,
            label="failure recovery event log",
        )
        terminal_events = [
            event for event in parsed_events if event.get("event") in TERMINAL_EVENTS
        ]
        if not terminal_events:
            terminal_event = _failure_event_payload(payload, at_utc=_utc_now())
            _append_json_line(events_path, terminal_event)
        elif len(terminal_events) == 1:
            terminal_event = terminal_events[0]
            _validate_failure_event(terminal_event, payload)
            if not _strict_json_equal(parsed_events[-1], terminal_event):
                raise GuardError("pre-existing terminal failure event is not final")
        else:
            raise GuardError("failure recovery refuses duplicate terminal events")
        _write_json_atomic(
            status_path,
            {
                "schema_version": "b0_driver_status.v1",
                "state": state,
                "updated_at_utc": _utc_now(),
                "current_node": payload.get("current_node"),
                "wrapper_pid": payload.get("wrapper_pid"),
                "terminal": True,
                "reason": payload.get("reason"),
                "exit_code": payload.get("exit_code"),
                "failure": _artifact_ref(failure_path, attempt_root),
            },
        )
        _validate_terminal_failure_state(attempt_root)
    return 0


def _process_sample(pid: int | None) -> dict[str, Any] | None:
    if pid is None or pid <= 0:
        return None
    completed = subprocess.run(
        ("ps", "-o", "pid=,ppid=,stat=,etime=,%cpu=,rss=,command=", "-p", str(pid)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "pid": pid,
        "present": completed.returncode == 0 and bool(completed.stdout.strip()),
        "ps": completed.stdout.strip() or None,
    }


def _memory_sample() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return {"source": "UNAVAILABLE_NON_LINUX", "fields_kib": {}}
    fields: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if separator and key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            fields[key] = int(raw.strip().split()[0])
    return {"source": "/proc/meminfo", "fields_kib": fields}


def _watchdog_sample(attempt_root: Path, d1_acceptance: Path) -> dict[str, Any]:
    try:
        status = _load_object(attempt_root / "status.json")
    except GuardError as exc:
        status = {"state": "STATUS_UNREADABLE", "error": str(exc)}
    wrapper_pid_raw = status.get("wrapper_pid")
    wrapper_pid = wrapper_pid_raw if isinstance(wrapper_pid_raw, int) else None
    child_pid: int | None = None
    node = status.get("current_node")
    if isinstance(node, str):
        process_path = attempt_root / "audit" / node / "process.json"
        if process_path.is_file():
            process = _load_object(process_path)
            raw_child = process.get("child_pid")
            child_pid = raw_child if isinstance(raw_child, int) else None
    attempt_disk = shutil.disk_usage(attempt_root)
    d1_disk = shutil.disk_usage(d1_acceptance)
    return {
        "schema_version": "b0_operational_watchdog.v1",
        "at_utc": _utc_now(),
        "interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        "scientific_logs_read": False,
        "scientific_gate_evaluated": False,
        "status": {
            "state": status.get("state"),
            "current_node": node,
        },
        "processes": {
            "wrapper": _process_sample(wrapper_pid),
            "child": _process_sample(child_pid),
        },
        "memory": _memory_sample(),
        "disk": {
            "attempt": {
                "path": str(attempt_root),
                "free_bytes": attempt_disk.free,
                "total_bytes": attempt_disk.total,
            },
            "d1": {
                "path": str(d1_acceptance),
                "free_bytes": d1_disk.free,
                "total_bytes": d1_disk.total,
            },
        },
    }


def _watchdog_once_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    d1_acceptance = _absolute_file(args.d1_acceptance, kind="D1 acceptance")
    metrics_path = _require_lexical_regular_file(
        attempt_root,
        Path("logs/system_metrics.jsonl"),
        label="watchdog metrics log",
    )
    _append_json_line(
        metrics_path,
        _watchdog_sample(attempt_root, d1_acceptance),
    )
    return 0


def _watchdog_parent_is_alive(parent_pid: int) -> bool:
    if os.getppid() != parent_pid:
        return False
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _watchdog_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    parent_pid = _require_positive_int(
        getattr(args, "parent_pid", None),
        "watchdog parent PID",
    )
    if not _watchdog_parent_is_alive(parent_pid):
        return 0
    d1_acceptance = _absolute_file(args.d1_acceptance, kind="D1 acceptance")
    metrics_path = _require_lexical_regular_file(
        attempt_root,
        Path("logs/system_metrics.jsonl"),
        label="watchdog metrics log",
    )
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        if not _watchdog_parent_is_alive(parent_pid):
            return 0
        _append_json_line(
            metrics_path,
            _watchdog_sample(attempt_root, d1_acceptance),
        )


def _require_lexical_regular_file(
    root: Path,
    relative: Path,
    *,
    label: str,
) -> Path:
    root = _absolute_lexical_directory(
        str(root),
        kind=f"{label} root",
    )
    if relative.is_absolute() or ".." in relative.parts:
        raise GuardError(f"{label} relative path is not contained: {relative}")
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise GuardError(f"{label} is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise GuardError(f"{label} contains a symbolic-link component: {current}")
        is_final = index == len(relative.parts) - 1
        if not is_final and not stat.S_ISDIR(mode):
            raise GuardError(f"{label} has a non-directory component: {current}")
        if is_final and not stat.S_ISREG(mode):
            raise GuardError(f"{label} is not a regular file: {current}")
    return current


def _require_lexical_directory(
    root: Path,
    relative: Path,
    *,
    label: str,
) -> Path:
    root = _absolute_lexical_directory(
        str(root),
        kind=f"{label} root",
    )
    if relative.is_absolute() or ".." in relative.parts:
        raise GuardError(f"{label} relative path is not contained: {relative}")
    current = root
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise GuardError(f"{label} is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise GuardError(f"{label} is not a lexical directory: {current}")
    return current


def _success_index(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.resolve(strict=True)
    regular_files: list[Path] = []
    pending_directories = [attempt_root]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise GuardError(
                f"success artifact inventory cannot read directory: {directory}: {exc}"
            ) from exc
        for child in children:
            path = Path(child.path)
            try:
                mode = child.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise GuardError(
                    f"success artifact inventory cannot inspect entry: {path}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise GuardError(
                    f"success artifact inventory forbids symbolic links: {path}"
                )
            if stat.S_ISDIR(mode):
                pending_directories.append(path)
                continue
            if not stat.S_ISREG(mode):
                raise GuardError(
                    "success artifact inventory forbids non-regular entries: " f"{path}"
                )
            regular_files.append(path)

    entries: list[dict[str, Any]] = []
    for path in sorted(regular_files):
        relative = path.relative_to(attempt_root).as_posix()
        if relative in SUCCESS_INDEX_EXCLUSIONS:
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    core = {
        "schema_version": "b0_success_artifact_checksums.v1",
        "entry_count": len(entries),
        "entries": entries,
        "excluded_mutable_or_self_referential_paths": sorted(SUCCESS_INDEX_EXCLUSIONS),
    }
    return {**core, "entries_sha256": _canonical_json_sha({"entries": entries})}


def _validate_success_index(
    attempt_root: Path,
    *,
    expected_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempt_root_resolved = attempt_root.resolve(strict=True)
    index_path = attempt_root / "artifact_checksums.json"
    _require_lexical_regular_file(
        attempt_root_resolved,
        Path("artifact_checksums.json"),
        label="artifact checksum index",
    )
    index = _load_object(index_path)
    if set(index) != SUCCESS_INDEX_FIELDS:
        raise GuardError(
            "artifact checksum index field set is not exact: "
            f"observed={sorted(index)!r}"
        )
    if index.get("schema_version") != "b0_success_artifact_checksums.v1":
        raise GuardError("artifact checksum index schema_version is invalid")

    exclusions = index.get("excluded_mutable_or_self_referential_paths")
    expected_exclusions = sorted(SUCCESS_INDEX_EXCLUSIONS)
    if type(exclusions) is not list or exclusions != expected_exclusions:
        raise GuardError("artifact checksum index exclusions are not exact")

    entries = index.get("entries")
    if type(entries) is not list:
        raise GuardError("artifact checksum index entries must be an array")
    entry_count = index.get("entry_count")
    if type(entry_count) is not int or entry_count < 0:
        raise GuardError("artifact checksum index entry_count must be an integer")
    if entry_count != len(entries):
        raise GuardError("artifact checksum index entry_count differs from entries")

    observed_paths: set[str] = set()
    for entry_number, entry in enumerate(entries):
        label = f"artifact checksum index entries[{entry_number}]"
        if not isinstance(entry, Mapping) or set(entry) != SUCCESS_INDEX_ENTRY_FIELDS:
            raise GuardError(f"{label} field set is not exact")
        raw_path = entry.get("path")
        if type(raw_path) is not str or not raw_path:
            raise GuardError(f"{label}.path is invalid")
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or raw_path != relative.as_posix()
            or raw_path in SUCCESS_INDEX_EXCLUSIONS
            or ".." in relative.parts
        ):
            raise GuardError(f"{label}.path is not a canonical indexed path")
        if raw_path in observed_paths:
            raise GuardError(f"artifact checksum index repeats path: {raw_path}")
        observed_paths.add(raw_path)
        lexical_path = _require_lexical_regular_file(
            attempt_root_resolved,
            relative,
            label=f"{label}.path",
        )
        try:
            indexed_path = lexical_path.resolve(strict=True)
            indexed_path.relative_to(attempt_root_resolved)
        except (OSError, ValueError) as exc:
            raise GuardError(f"{label}.path escapes or is unavailable") from exc
        if not indexed_path.is_file():
            raise GuardError(f"{label}.path is not a file")

        expected_bytes = entry.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise GuardError(f"{label}.bytes is invalid")
        expected_sha256 = entry.get("sha256")
        if type(expected_sha256) is not str:
            raise GuardError(f"{label}.sha256 is invalid")
        _require_hex(expected_sha256, 64, f"{label}.sha256")

    entries_sha256 = index.get("entries_sha256")
    if type(entries_sha256) is not str:
        raise GuardError("artifact checksum index entries_sha256 is invalid")
    _require_hex(entries_sha256, 64, "artifact checksum index entries_sha256")
    recomputed_entries_sha256 = _canonical_json_sha({"entries": entries})
    if entries_sha256 != recomputed_entries_sha256:
        raise GuardError("artifact checksum index entries_sha256 mismatch")

    live_index = _success_index(attempt_root)
    if index != live_index:
        raise GuardError(
            "live artifact inventory or checksums differ from artifact checksum index"
        )

    if expected_reference is not None:
        if (
            set(expected_reference) != SUCCESS_INDEX_ENTRY_FIELDS
            or expected_reference.get("path") != "artifact_checksums.json"
        ):
            raise GuardError(
                "artifact checksum index reference fields/path are invalid"
            )
        _validate_ref(
            index_path,
            expected_reference,
            label="terminal artifact checksum index",
        )

    return {
        "path": "artifact_checksums.json",
        "bytes": index_path.stat().st_size,
        "sha256": _sha256_file(index_path),
        "entry_count": entry_count,
        "entries_sha256": entries_sha256,
        "passed": True,
    }


def _validate_relative_artifact_ref(
    reference: Mapping[str, Any],
    root: Path,
    *,
    label: str,
) -> Path:
    raw_path = reference.get("path")
    if not isinstance(raw_path, str):
        raise GuardError(f"{label} path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise GuardError(f"{label} path escapes its audit root")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise GuardError(f"{label} path escapes its audit root") from exc
    _validate_ref(path, reference, label=label)
    return path


def _validate_audit_node(
    *,
    attempt_root: Path,
    node: str,
    project_root: Path,
    python_launcher: str,
    expected_head: str,
    expected_dirty_state_sha256: str,
) -> None:
    run_root = attempt_root / "audit" / node
    completion_path = _absolute_file(
        str(run_root / "completion.json"),
        kind=f"{node} completion",
    )
    manifest_path = _absolute_file(
        str(run_root / "audit_manifest.json"),
        kind=f"{node} audit manifest",
    )
    completion = _load_object(completion_path)
    manifest = _load_object(manifest_path)
    if not (
        completion.get("schema_version") == "audited_command_run.v1"
        and completion.get("state") == "COMMAND_COMPLETED"
        and type(completion.get("observed_process_exit_code")) is int
        and completion.get("observed_process_exit_code") == 0
        and type(completion.get("wrapper_exit_code")) is int
        and completion.get("wrapper_exit_code") == 0
        and completion.get("stop_reason") is None
    ):
        raise GuardError(f"audit node did not complete cleanly: {node}")
    if not (
        manifest.get("schema_version") == "audited_command_run.v1"
        and manifest.get("state") == "COMMAND_COMPLETED"
        and manifest.get("run_root") == str(run_root)
        and manifest.get("project_root") == str(project_root)
        and manifest.get("working_directory") == str(project_root)
        and manifest.get("workload_class") == "NON_NEURAL_DATA_BENCHMARK"
        and manifest.get("child_pid") == completion.get("child_pid")
        and type(manifest.get("observed_process_exit_code")) is int
        and manifest.get("observed_process_exit_code") == 0
        and type(manifest.get("wrapper_exit_code")) is int
        and manifest.get("wrapper_exit_code") == 0
        and manifest.get("stop_reason") is None
    ):
        raise GuardError(f"audit manifest/completion mismatch: {node}")
    argv = manifest.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or argv[0] != python_launcher
        or Path(str(argv[1])).name != NODE_ENTRYPOINTS[node]
    ):
        raise GuardError(f"audited node argv entrypoint mismatch: {node}")
    cuda = manifest.get("cuda")
    if not (
        isinstance(cuda, Mapping)
        and cuda.get("applicability") == "NOT_APPLICABLE_NON_NEURAL_WORKLOAD"
        and cuda.get("formal_neural_activity") is False
        and cuda.get("gpu_validation_started") is False
        and cuda.get("automatic_cpu_fallback") is False
    ):
        raise GuardError(f"audited node workload boundary mismatch: {node}")
    git_snapshot = manifest.get("git_prelaunch_snapshot")
    if not (
        isinstance(git_snapshot, Mapping)
        and git_snapshot.get("repository") == str(project_root)
        and git_snapshot.get("head") == expected_head
        and git_snapshot.get("dirty_state_sha256") == expected_dirty_state_sha256
        and git_snapshot.get("clean") is True
    ):
        raise GuardError(f"audited node Git binding mismatch: {node}")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, Mapping):
        raise GuardError(f"audited node evidence map is invalid: {node}")
    for required_key in ("invocation", "completion", "stdout", "stderr", "process"):
        reference = evidence.get(required_key)
        if not isinstance(reference, Mapping):
            raise GuardError(f"{node} omits audit evidence {required_key}")
        observed_path = _validate_relative_artifact_ref(
            reference,
            run_root,
            label=f"{node}.{required_key}",
        )
        if required_key == "completion" and observed_path != completion_path:
            raise GuardError(f"{node} completion evidence path mismatch")
    git_artifacts = git_snapshot.get("artifacts")
    if not isinstance(git_artifacts, Mapping):
        raise GuardError(f"{node} Git snapshot artifacts are invalid")
    for key, reference in git_artifacts.items():
        if not isinstance(reference, Mapping):
            raise GuardError(f"{node} Git snapshot ref is invalid: {key}")
        _validate_relative_artifact_ref(
            reference,
            run_root,
            label=f"{node}.git.{key}",
        )
    snapshot_manifest = git_snapshot.get("snapshot_manifest")
    if not isinstance(snapshot_manifest, Mapping):
        raise GuardError(f"{node} Git snapshot manifest ref is invalid")
    _validate_relative_artifact_ref(
        snapshot_manifest,
        run_root,
        label=f"{node}.git.snapshot_manifest",
    )


def _validate_fingerprint_set(
    *,
    attempt_root: Path,
    code_manifest: Path,
    expected_code_manifest_sha256: str,
) -> dict[str, Any]:
    expected_code_manifest_sha256 = _require_hex(
        expected_code_manifest_sha256,
        64,
        "expected code-manifest SHA256",
    )
    if _sha256_file(code_manifest) != expected_code_manifest_sha256:
        raise GuardError("code manifest changed after in-memory approval")
    baseline = _load_object(code_manifest)
    if (
        baseline.get("schema_version") != "b0_code_freeze.v1"
        or baseline.get("clean") is not True
        or _get(baseline, ("git_index_flags", "passed")) is not True
        or _get(baseline, ("critical_integrity", "passed")) is not True
        or _get(baseline, ("caller_approval", "passed")) is not True
    ):
        raise GuardError("code manifest is not an approved clean baseline")
    fingerprint_dir = attempt_root / "provenance/fingerprints"
    observed_labels = {
        path.name.removesuffix(".json") for path in fingerprint_dir.glob("*.json")
    }
    if observed_labels != set(EXPECTED_FINGERPRINT_LABELS):
        raise GuardError(
            "fingerprint evidence set mismatch: "
            f"expected={sorted(EXPECTED_FINGERPRINT_LABELS)!r}, "
            f"observed={sorted(observed_labels)!r}"
        )
    for label in EXPECTED_FINGERPRINT_LABELS:
        observed = _load_object(fingerprint_dir / f"{label}.json")
        comparison = observed.get("comparison")
        if not (
            isinstance(comparison, Mapping)
            and comparison.get("label") == label.replace(".", ":", 1)
            and comparison.get("matches") is True
            and comparison.get("expected_baseline_sha256")
            == expected_code_manifest_sha256
            and observed.get("fingerprint_sha256") == baseline.get("fingerprint_sha256")
            and observed.get("head") == baseline.get("head")
            and observed.get("git_dirty_state_sha256")
            == baseline.get("git_dirty_state_sha256")
            and observed.get("critical_entries") == baseline.get("critical_entries")
            and observed.get("git_index_flags") == baseline.get("git_index_flags")
            and observed.get("critical_integrity") == baseline.get("critical_integrity")
            and _get(observed, ("git_index_flags", "passed")) is True
            and _get(observed, ("critical_integrity", "passed")) is True
        ):
            raise GuardError(
                f"fingerprint evidence is not independently bound: {label}"
            )
        baseline_ref = comparison.get("baseline")
        if not isinstance(baseline_ref, Mapping):
            raise GuardError(f"fingerprint baseline ref is invalid: {label}")
        _validate_ref(code_manifest, baseline_ref, label=f"{label}.baseline")
    project_root = _absolute_directory(
        str(baseline.get("project_root", "")),
        kind="code-manifest project root",
    )
    driver_ref = baseline.get("driver")
    if not isinstance(driver_ref, Mapping):
        raise GuardError("code manifest driver ref is invalid")
    driver_path = _absolute_file(
        str(driver_ref.get("path", "")),
        kind="code-manifest driver",
    )
    _validate_ref(driver_path, driver_ref, label="code-manifest driver")
    final_observed = _fingerprint_payload(project_root, driver_path)
    if (
        final_observed.get("fingerprint_sha256") != baseline.get("fingerprint_sha256")
        or _get(final_observed, ("git_index_flags", "passed")) is not True
        or _get(final_observed, ("critical_integrity", "passed")) is not True
    ):
        raise GuardError("final seal code fingerprint differs from approved baseline")
    return baseline


def _validate_input_check_set(
    *,
    attempt_root: Path,
    input_manifest: Path,
    expected_input_manifest_sha256: str,
) -> dict[str, Any]:
    manifest, frozen_inputs = _validated_input_manifest(
        input_manifest,
        expected_input_manifest_sha256,
    )
    check_dir = attempt_root / "provenance/input_checks"
    observed_labels = {
        path.name.removesuffix(".json") for path in check_dir.glob("*.json")
    }
    if observed_labels != set(EXPECTED_INPUT_CHECK_LABELS):
        raise GuardError(
            "input-check evidence set mismatch: "
            f"expected={sorted(EXPECTED_INPUT_CHECK_LABELS)!r}, "
            f"observed={sorted(observed_labels)!r}"
        )
    input_manifest_ref = _path_ref(input_manifest)
    for label in EXPECTED_INPUT_CHECK_LABELS:
        check = _load_object(check_dir / f"{label}.json")
        if not (
            check.get("schema_version") == "b0_frozen_inputs_check.v1"
            and check.get("label") == label.replace(".", ":", 1)
            and check.get("matches") is True
            and check.get("input_manifest_sha256") == expected_input_manifest_sha256
            and check.get("input_manifest") == input_manifest_ref
            and check.get("inputs") == frozen_inputs
        ):
            raise GuardError(
                f"input-check evidence is not independently bound: {label}"
            )
    for key, reference in frozen_inputs.items():
        if not isinstance(reference, Mapping):
            raise GuardError(f"frozen input ref is invalid: {key}")
        path = _absolute_file(
            str(reference.get("path", "")),
            kind=f"seal-frozen D1 {key}",
        )
        _validate_ref(path, reference, label=f"seal-frozen D1 {key}")
    return manifest


def _expected_gate_specs(
    *,
    attempt_root: Path,
    expected_head: str,
    expected_dirty_state_sha256: str,
    d1_acceptance: Path,
) -> dict[str, dict[str, Any]]:
    canonical_validation = (
        attempt_root / "artifacts/canonical_validation.json"
    ).resolve()
    split_paths = {
        basename: (attempt_root / "artifacts/splits" / basename).resolve()
        for basename in SPLIT_BASENAMES
    }
    leakage_paths = {
        basename: (attempt_root / "artifacts/leakage" / basename).resolve()
        for basename in SPLIT_BASENAMES
    }
    specs: dict[str, dict[str, Any]] = {}

    def add(
        label: str,
        gate: str,
        artifact: Path,
        **context: Any,
    ) -> None:
        if label in specs:
            raise GuardError(f"duplicate expected named-gate label: {label}")
        specs[label] = {
            "gate": gate,
            "artifact": artifact,
            "expected_head": context.get("expected_head"),
            "expected_dirty_state_sha256": context.get("expected_dirty_state_sha256"),
            "expected_d1_acceptance": context.get("expected_d1_acceptance"),
            "expected_canonical_validation": context.get(
                "expected_canonical_validation"
            ),
        }

    for node in EXPECTED_AUDIT_NODES:
        add(
            f"audit_completion_{node}",
            "audit-completion",
            (attempt_root / "audit" / node / "completion.json").resolve(),
        )
        add(
            f"audit_git_binding_{node}",
            "audit-git-binding",
            (attempt_root / "audit" / node / "audit_manifest.json").resolve(),
            expected_head=expected_head,
            expected_dirty_state_sha256=expected_dirty_state_sha256,
        )
    add(
        "preflight",
        "preflight",
        (attempt_root / "artifacts/preflight.json").resolve(),
    )
    add(
        "canonical_validation",
        "canonical-validation",
        canonical_validation,
    )
    for basename, path in split_paths.items():
        add(
            f"split_common_{basename}",
            "split-common",
            path,
            expected_d1_acceptance=str(d1_acceptance),
            expected_canonical_validation=str(canonical_validation),
        )
    add(
        "split_5utr_source_role",
        "split-5utr-source-role",
        split_paths["5utr_source_disjoint.json"],
    )
    add(
        "split_5utr_study_role",
        "split-5utr-study-role",
        split_paths["5utr_study_disjoint.json"],
    )
    add(
        "split_3utr_source_role",
        "split-3utr-source-role",
        split_paths["3utr_source_disjoint.json"],
    )
    add(
        "split_3utr_study_role",
        "split-3utr-study-role",
        split_paths["3utr_study_disjoint.json"],
    )
    add(
        "split_cross_region_role",
        "split-cross-region-role",
        split_paths["cross_region_transfer.json"],
    )
    for basename, path in leakage_paths.items():
        add(f"leakage_{basename}", "leakage", path)
    add(
        "evaluation_bundle",
        "evaluation-bundle",
        (attempt_root / "artifacts/bundle/build_manifest.json").resolve(),
    )
    add(
        "final_acceptance",
        "final-acceptance",
        (attempt_root / "artifacts/acceptance.json").resolve(),
    )
    if len(specs) != 47:
        raise GuardError(
            f"internal expected named-gate inventory is not 47: {len(specs)}"
        )
    return specs


def _expected_business_event_sequence() -> list[tuple[str, str]]:
    expected: list[tuple[str, str]] = []

    def gate(label: str) -> None:
        expected.extend((("GATE_START", label), ("GATE_PASSED", label)))

    def node(name: str) -> None:
        expected.append(("NODE_START", name))
        gate(f"audit_completion_{name}")
        gate(f"audit_git_binding_{name}")
        expected.append(("NODE_COMPLETED", name))

    node("00_preflight")
    gate("preflight")
    node("01_canonical_validation")
    gate("canonical_validation")
    for name in EXPECTED_AUDIT_NODES[2:7]:
        node(name)
    for basename in SPLIT_BASENAMES:
        gate(f"split_common_{basename}")
    for label in (
        "split_5utr_source_role",
        "split_5utr_study_role",
        "split_3utr_source_role",
        "split_3utr_study_role",
        "split_cross_region_role",
    ):
        gate(label)
    for name in EXPECTED_AUDIT_NODES[7:12]:
        node(name)
    for basename in SPLIT_BASENAMES:
        gate(f"leakage_{basename}")
    node("12_evaluation_bundle")
    gate("evaluation_bundle")
    node("13_final_acceptance")
    gate("final_acceptance")
    return expected


def _validate_named_gate_evidence_set(
    *,
    attempt_root: Path,
    expected_head: str,
    expected_dirty_state_sha256: str,
    d1_acceptance: Path,
    parsed_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    specs = _expected_gate_specs(
        attempt_root=attempt_root,
        expected_head=expected_head,
        expected_dirty_state_sha256=expected_dirty_state_sha256,
        d1_acceptance=d1_acceptance,
    )
    gate_dir = attempt_root / "provenance/gates"
    observed_paths = list(gate_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in observed_paths):
        raise GuardError("named-gate evidence directory contains a non-regular file")
    observed_labels = {
        path.name.removesuffix(".json")
        for path in observed_paths
        if path.name.endswith(".json")
    }
    if len(observed_paths) != len(observed_labels) or observed_labels != set(specs):
        raise GuardError(
            "named-gate evidence set mismatch: "
            f"expected={sorted(specs)!r}, observed={sorted(observed_labels)!r}"
        )

    checked: list[dict[str, Any]] = []
    for label, spec in specs.items():
        evidence_path = gate_dir / f"{label}.json"
        evidence = _load_object(evidence_path)
        artifact = _absolute_file(
            str(spec["artifact"]),
            kind=f"{label} live gate artifact",
        )
        context = _validate_named_gate(
            str(spec["gate"]),
            _load_object(artifact),
            expected_head=spec["expected_head"],
            expected_dirty_state_sha256=spec["expected_dirty_state_sha256"],
            expected_d1_acceptance=spec["expected_d1_acceptance"],
            expected_canonical_validation=spec["expected_canonical_validation"],
        )
        checked_at = evidence.get("checked_at_utc")
        try:
            checked_at_value = (
                datetime.fromisoformat(checked_at.removesuffix("Z") + "+00:00")
                if isinstance(checked_at, str) and checked_at.endswith("Z")
                else None
            )
        except ValueError:
            checked_at_value = None
        if not (
            set(evidence)
            == {
                "schema_version",
                "checked_at_utc",
                "label",
                "gate",
                "artifact_path",
                "artifact",
                "context",
                "passed",
            }
            and checked_at_value is not None
            and checked_at_value.utcoffset() == timezone.utc.utcoffset(checked_at_value)
            and evidence.get("schema_version") == "b0_named_gate_evidence.v1"
            and evidence.get("label") == label
            and evidence.get("gate") == spec["gate"]
            and evidence.get("artifact_path") == str(artifact)
            and evidence.get("artifact") == _path_ref(artifact)
            and evidence.get("context") == context
            and evidence.get("passed") is True
            and "error" not in evidence
        ):
            raise GuardError(f"named-gate evidence is not exactly bound: {label}")
        checked.append(
            {
                "label": label,
                "gate": spec["gate"],
                "evidence": _path_ref(evidence_path),
                "artifact": _path_ref(artifact),
            }
        )

    observed_business_events: list[tuple[str, str]] = []
    for event in parsed_events:
        event_name = event.get("event")
        if event_name in {"NODE_START", "NODE_COMPLETED"}:
            observed_business_events.append((str(event_name), str(event.get("node"))))
        elif event_name in {"GATE_START", "GATE_PASSED"}:
            observed_business_events.append((str(event_name), str(event.get("detail"))))
    expected_business_events = _expected_business_event_sequence()
    if observed_business_events != expected_business_events:
        raise GuardError("business events do not prove strict node/gate interleaving")
    return {
        "schema_version": "b0_named_gate_set_validation.v1",
        "gate_count": len(checked),
        "gates": checked,
        "business_event_count": len(observed_business_events),
        "strict_order_passed": True,
    }


def _validated_manifest_ref(
    reference: Any,
    *,
    label: str,
    required_parent: Path | None = None,
) -> Path:
    if not isinstance(reference, Mapping):
        raise GuardError(f"{label} reference is not an object")
    path = _absolute_file(str(reference.get("path", "")), kind=label)
    if required_parent is not None:
        try:
            path.relative_to(required_parent.resolve(strict=True))
        except ValueError as exc:
            raise GuardError(f"{label} escapes its required parent") from exc
    _validate_ref(path, reference, label=label)
    return path


def _validate_final_artifact_graph(
    *,
    attempt_root: Path,
    bundle_manifest: Path,
    final_acceptance: Path,
    expected_exposure_ledger: Path,
) -> dict[str, Any]:
    bundle_root = (attempt_root / "artifacts/bundle").resolve(strict=True)
    if bundle_manifest != bundle_root / "build_manifest.json":
        raise GuardError("bundle manifest is not at the canonical bundle path")
    bundle = _load_object(bundle_manifest)
    acceptance = _load_object(final_acceptance)

    outputs = bundle.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise GuardError("bundle manifest outputs must be a non-empty list")
    output_paths: list[Path] = []
    for index, reference in enumerate(outputs):
        output_paths.append(
            _validated_manifest_ref(
                reference,
                label=f"bundle.outputs[{index}]",
                required_parent=bundle_root,
            )
        )
    if len(output_paths) != len(set(output_paths)):
        raise GuardError("bundle manifest contains duplicate output paths")
    declared_relatives = {
        path.relative_to(bundle_root).as_posix() for path in output_paths
    }
    if not MANDATORY_BUNDLE_OUTPUTS.issubset(declared_relatives):
        raise GuardError(
            "bundle manifest omits mandatory tracks, Data Card, or bindings"
        )
    actual_relatives = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    expected_relatives = declared_relatives | {"build_manifest.json"}
    if actual_relatives != expected_relatives:
        raise GuardError(
            "bundle live-file inventory differs from its exact output manifest"
        )

    inputs = bundle.get("inputs")
    if not isinstance(inputs, Mapping):
        raise GuardError("bundle manifest inputs must be an object")
    supplied_split_refs = inputs.get("split_manifests")
    supplied_leakage_refs = inputs.get("supplied_leakage_reports")
    if not isinstance(supplied_split_refs, list) or not isinstance(
        supplied_leakage_refs, list
    ):
        raise GuardError("bundle split/leakage input refs must be arrays")
    expected_split_paths = {
        (attempt_root / "artifacts/splits" / basename).resolve(strict=True)
        for basename in SPLIT_BASENAMES
    }
    expected_leakage_paths = {
        (attempt_root / "artifacts/leakage" / basename).resolve(strict=True)
        for basename in SPLIT_BASENAMES
    }
    observed_split_paths = {
        _validated_manifest_ref(
            reference,
            label=f"bundle split input {index}",
        )
        for index, reference in enumerate(supplied_split_refs)
    }
    observed_leakage_paths = {
        _validated_manifest_ref(
            reference,
            label=f"bundle leakage input {index}",
        )
        for index, reference in enumerate(supplied_leakage_refs)
    }
    if (
        len(supplied_split_refs) != 5
        or observed_split_paths != expected_split_paths
        or len(supplied_leakage_refs) != 5
        or observed_leakage_paths != expected_leakage_paths
    ):
        raise GuardError("bundle input graph is not the exact five split/leakage set")

    acceptance_leakage_refs = acceptance.get("supplied_leakage_report_files")
    if not isinstance(acceptance_leakage_refs, list):
        raise GuardError("final acceptance omits supplied leakage file bindings")
    acceptance_leakage_paths = {
        _validated_manifest_ref(
            reference,
            label=f"acceptance leakage input {index}",
        )
        for index, reference in enumerate(acceptance_leakage_refs)
    }
    if (
        len(acceptance_leakage_refs) != 5
        or acceptance_leakage_paths != expected_leakage_paths
    ):
        raise GuardError("final acceptance leakage file set is not exact")

    required_audit = acceptance.get("required_artifact_audit")
    if not isinstance(required_audit, Mapping):
        raise GuardError("final acceptance required-artifact audit is invalid")
    binding_path = _absolute_file(
        str(required_audit.get("binding_manifest_path", "")),
        kind="accepted artifact-binding manifest",
    )
    expected_binding_path = bundle_root / "artifact_bindings.json"
    if binding_path != expected_binding_path:
        raise GuardError("accepted artifact-binding path differs from bundle")
    binding_sha = _require_hex(
        required_audit.get("binding_manifest_sha256"),
        64,
        "accepted artifact-binding SHA256",
    )
    if _sha256_file(binding_path) != binding_sha:
        raise GuardError("accepted artifact-binding manifest hash changed")
    binding = _load_object(binding_path)
    if binding.get("schema_version") != "utr_b0_artifact_bindings.v2":
        raise GuardError("artifact-binding manifest schema is invalid")
    binding_artifacts = binding.get("artifacts")
    accepted_artifacts = required_audit.get("artifacts")
    expected_bound_names = {
        "exposure_ledger",
        "track_role_matrix",
        "data_card",
        "claims",
    }
    canonical_bound_paths = {
        "exposure_ledger": expected_exposure_ledger.resolve(strict=True),
        "track_role_matrix": (
            bundle_root / "evaluation/tracks/track_role_matrix.yaml"
        ).resolve(strict=True),
        "data_card": (bundle_root / "docs/data/UTR_EditBench_v2_Data_Card.md").resolve(
            strict=True
        ),
        "claims": (
            bundle_root / "evaluation/claims/allowed_unsupported_claims.yaml"
        ).resolve(strict=True),
    }
    if (
        not isinstance(binding_artifacts, Mapping)
        or set(binding_artifacts) != expected_bound_names
        or not isinstance(accepted_artifacts, Mapping)
        or set(accepted_artifacts) != expected_bound_names
    ):
        raise GuardError("required artifact binding set is not exact")
    for name in sorted(expected_bound_names):
        reference = binding_artifacts[name]
        if not isinstance(reference, Mapping):
            raise GuardError(f"artifact binding is invalid: {name}")
        raw_path = reference.get("path")
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = bundle_root / candidate
        path = _absolute_file(str(candidate), kind=f"bound artifact {name}")
        if path != canonical_bound_paths[name]:
            raise GuardError(f"bound artifact is not at its canonical path: {name}")
        _validate_ref(path, reference, label=f"bound artifact {name}")
        accepted = accepted_artifacts[name]
        if not (
            isinstance(accepted, Mapping)
            and accepted.get("exists") is True
            and accepted.get("schema_valid") is True
            and accepted.get("path") == str(path)
            and type(accepted.get("bytes")) is int
            and accepted.get("bytes") == path.stat().st_size
            and accepted.get("sha256") == _sha256_file(path)
        ):
            raise GuardError(
                f"accepted required artifact no longer matches live bytes: {name}"
            )

    return {
        "schema_version": "b0_final_artifact_graph_validation.v1",
        "bundle_output_count": len(output_paths),
        "bundle_output_paths": sorted(declared_relatives),
        "split_input_count": len(observed_split_paths),
        "leakage_input_count": len(observed_leakage_paths),
        "required_bound_artifacts": sorted(expected_bound_names),
        "passed": True,
    }


def _recompute_final_acceptance(
    *,
    attempt_root: Path,
    project_root: Path,
    python_launcher: str,
    expected_acceptance: Path,
) -> dict[str, Any]:
    audit_manifest = _load_object(
        attempt_root / "audit/13_final_acceptance/audit_manifest.json"
    )
    argv = audit_manifest.get("argv")
    expected_entrypoint = (
        project_root / "scripts/data/validate_b0_acceptance.py"
    ).resolve(strict=True)
    observed_entrypoint = (
        Path(str(argv[1])) if isinstance(argv, list) and len(argv) > 1 else Path("")
    )
    if not observed_entrypoint.is_absolute():
        observed_entrypoint = project_root / observed_entrypoint
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or argv[0] != python_launcher
        or observed_entrypoint.resolve(strict=True) != expected_entrypoint
        or argv.count("--output") != 1
    ):
        raise GuardError("node13 argv cannot support exact final revalidation")
    output_index = argv.index("--output") + 1
    if output_index >= len(argv):
        raise GuardError("node13 --output has no value")
    if Path(str(argv[output_index])).resolve(strict=True) != expected_acceptance:
        raise GuardError("node13 acceptance output path is not canonical")

    recheck_path = attempt_root / "provenance/final_acceptance_recheck.json"
    recheck_stdout = attempt_root / "provenance/final_acceptance_recheck.stdout.log"
    recheck_stderr = attempt_root / "provenance/final_acceptance_recheck.stderr.log"
    recheck_argv = [str(value) for value in argv]
    recheck_argv[output_index] = str(recheck_path)
    completed = subprocess.run(
        recheck_argv,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _write_bytes_exclusive(recheck_stdout, completed.stdout)
    _write_bytes_exclusive(recheck_stderr, completed.stderr)
    if completed.returncode != 0:
        raise GuardError(
            f"final acceptance live recomputation failed: {completed.returncode}"
        )
    rechecked = _load_object(
        _absolute_file(
            str(recheck_path),
            kind="recomputed final acceptance",
        )
    )
    expected = _load_object(expected_acceptance)
    _validate_b0_acceptance_payload(rechecked)
    if rechecked != expected:
        raise GuardError("recomputed final acceptance differs from node13 acceptance")
    return {
        "schema_version": "b0_final_acceptance_recheck.v1",
        "argv": recheck_argv,
        "output": _path_ref(recheck_path),
        "stdout": _path_ref(recheck_stdout),
        "stderr": _path_ref(recheck_stderr),
        "matches_node13_acceptance": True,
        "passed": True,
    }


def _seal_command(args: argparse.Namespace) -> int:
    attempt_root = _absolute_lexical_directory(
        args.attempt_root,
        kind="attempt root",
    )
    acceptance = _absolute_file(args.final_acceptance, kind="B0 final acceptance")
    bundle = _absolute_file(args.bundle_manifest, kind="B0 bundle manifest")
    code_manifest = _absolute_file(args.code_manifest, kind="B0 code manifest")
    input_manifest = _absolute_file(
        args.input_manifest,
        kind="B0 frozen input manifest",
    )
    acceptance_binding = _absolute_file(
        args.acceptance_binding,
        kind="B0 acceptance binding",
    )
    bundle_binding = _absolute_file(
        args.bundle_binding,
        kind="B0 bundle binding",
    )
    attempt_manifest = _require_lexical_regular_file(
        attempt_root,
        Path("attempt_manifest.json"),
        label="B0 attempt manifest",
    )
    events = _require_lexical_regular_file(
        attempt_root,
        Path("logs/events.jsonl"),
        label="B0 events log",
    )
    expected_attempt_manifest_sha256 = _require_hex(
        args.expected_attempt_manifest_sha256,
        64,
        "expected attempt-manifest SHA256",
    )
    if _sha256_file(attempt_manifest) != expected_attempt_manifest_sha256:
        raise GuardError("attempt manifest changed after in-memory approval")

    with _terminal_lock(attempt_root):
        status_path = _require_lexical_regular_file(
            attempt_root,
            Path("status.json"),
            label="seal status",
        )
        _require_lexical_directory(
            attempt_root,
            Path("failure"),
            label="seal failure directory",
        )
        _require_lexical_directory(
            attempt_root,
            Path("logs"),
            label="seal logs directory",
        )
        _require_lexical_directory(
            attempt_root,
            Path("provenance"),
            label="seal provenance directory",
        )
        status = _load_object(status_path)
        if status.get("terminal") is True:
            raise GuardError("seal refuses a pre-existing terminal status")
        if (attempt_root / "failure/failure.json").exists():
            raise GuardError("seal refuses an attempt with failure evidence")
        for forbidden_existing in (
            attempt_root / "artifact_checksums.json",
            attempt_root / "driver_completion.json",
            attempt_root / "provenance/events_at_terminal.jsonl",
            attempt_root / "provenance/terminal_success_event.json",
            attempt_root / "provenance/final_acceptance_recheck.json",
            attempt_root / "provenance/final_acceptance_recheck.stdout.log",
            attempt_root / "provenance/final_acceptance_recheck.stderr.log",
            attempt_root / "provenance/final_acceptance_recheck_manifest.json",
            attempt_root / "provenance/named_gate_set_validation.json",
            attempt_root / "provenance/final_artifact_graph_validation.json",
        ):
            if os.path.lexists(forbidden_existing):
                raise GuardError(
                    f"seal refuses pre-existing success artifact: {forbidden_existing}"
                )

        manifest = _load_object(attempt_manifest)
        paths = manifest.get("paths")
        expected = manifest.get("expected")
        if not isinstance(paths, Mapping) or not isinstance(expected, Mapping):
            raise GuardError("attempt manifest paths/expected fields are invalid")
        project_root = _absolute_directory(
            str(paths.get("project_root", "")),
            kind="sealed project root",
        )
        python_launcher = str(paths.get("python_launcher", ""))
        expected_head = _require_hex(
            expected.get("commit"),
            40,
            "sealed expected commit",
        )
        expected_dirty = _require_hex(
            expected.get("dirty_state_sha256"),
            64,
            "sealed expected dirty state",
        )
        baseline = _validate_fingerprint_set(
            attempt_root=attempt_root,
            code_manifest=code_manifest,
            expected_code_manifest_sha256=args.expected_code_manifest_sha256,
        )
        if (
            baseline.get("head") != expected_head
            or baseline.get("git_dirty_state_sha256") != expected_dirty
        ):
            raise GuardError("code manifest differs from attempt approval")

        audit_nodes = sorted(
            child.name for child in (attempt_root / "audit").iterdir() if child.is_dir()
        )
        if tuple(audit_nodes) != EXPECTED_AUDIT_NODES:
            raise GuardError(
                f"audit node set mismatch: expected={EXPECTED_AUDIT_NODES!r}, "
                f"observed={tuple(audit_nodes)!r}"
            )
        for node in EXPECTED_AUDIT_NODES:
            _validate_audit_node(
                attempt_root=attempt_root,
                node=node,
                project_root=project_root,
                python_launcher=python_launcher,
                expected_head=expected_head,
                expected_dirty_state_sha256=expected_dirty,
            )

        frozen_input_payload = _validate_input_check_set(
            attempt_root=attempt_root,
            input_manifest=input_manifest,
            expected_input_manifest_sha256=args.expected_input_manifest_sha256,
        )
        accepted_binding_payload = _validated_result_binding(
            binding_path=acceptance_binding,
            expected_binding_sha256=args.expected_acceptance_binding_sha256,
            expected_kind="acceptance",
            live_artifact=acceptance,
        )
        bundle_binding_payload = _validated_result_binding(
            binding_path=bundle_binding,
            expected_binding_sha256=args.expected_bundle_binding_sha256,
            expected_kind="bundle",
            live_artifact=bundle,
        )
        final_artifact_graph = _validate_final_artifact_graph(
            attempt_root=attempt_root,
            bundle_manifest=bundle,
            final_acceptance=acceptance,
            expected_exposure_ledger=_absolute_file(
                str(frozen_input_payload["inputs"]["exposure_ledger"]["path"]),
                kind="seal-frozen D1 exposure ledger",
            ),
        )
        final_acceptance_recheck = _recompute_final_acceptance(
            attempt_root=attempt_root,
            project_root=project_root,
            python_launcher=python_launcher,
            expected_acceptance=acceptance,
        )

        parsed_events: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            events.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                event_payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GuardError(
                    f"events JSONL line {line_number} is invalid: {exc}"
                ) from exc
            if not isinstance(event_payload, dict):
                raise GuardError(f"events JSONL line {line_number} is not an object")
            parsed_events.append(event_payload)
        if any(event.get("event") in TERMINAL_EVENTS for event in parsed_events):
            raise GuardError("seal refuses a pre-existing terminal event")
        d1_acceptance_ref = frozen_input_payload["inputs"]["acceptance"]
        if not isinstance(d1_acceptance_ref, Mapping):
            raise GuardError("frozen D1 acceptance ref is invalid at seal")
        frozen_d1_acceptance = _absolute_file(
            str(d1_acceptance_ref.get("path", "")),
            kind="seal-frozen D1 acceptance",
        )
        named_gate_set = _validate_named_gate_evidence_set(
            attempt_root=attempt_root,
            expected_head=expected_head,
            expected_dirty_state_sha256=expected_dirty,
            d1_acceptance=frozen_d1_acceptance,
            parsed_events=parsed_events,
        )
        _write_json_exclusive(
            attempt_root / "provenance/named_gate_set_validation.json",
            named_gate_set,
        )
        _write_json_exclusive(
            attempt_root / "provenance/final_artifact_graph_validation.json",
            final_artifact_graph,
        )
        _write_json_exclusive(
            attempt_root / "provenance/final_acceptance_recheck_manifest.json",
            final_acceptance_recheck,
        )

        _append_json_line(
            events,
            {
                "schema_version": "b0_driver_event.v1",
                "at_utc": _utc_now(),
                "event": "B0_DRIVER_SEAL_READY",
                "stage_completion_claimed": False,
            },
        )
        terminal_event = {
            "schema_version": "b0_driver_event.v1",
            "at_utc": _utc_now(),
            "event": "B0_DRIVER_COMPLETED",
            "attempt_root": str(attempt_root),
            "final_acceptance": _artifact_ref(acceptance, attempt_root),
            "failure_evidence_present": False,
            "authoritative_only_when_terminal_status_ref_matches": True,
            "stage_completion_claimed": False,
            "scientific_result_claimed": False,
        }
        terminal_event_path = attempt_root / "provenance/terminal_success_event.json"
        _write_json_exclusive(terminal_event_path, terminal_event)
        events_snapshot = attempt_root / "provenance/events_at_terminal.jsonl"
        terminal_event_line = (
            json.dumps(
                terminal_event,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        _write_bytes_exclusive(
            events_snapshot,
            events.read_bytes() + terminal_event_line,
        )

        index_path = attempt_root / "artifact_checksums.json"
        _write_json_exclusive(index_path, _success_index(attempt_root))
        completion_payload = {
            "schema_version": "b0_driver_completion.v1",
            "state": "B0_DRIVER_COMPLETED",
            "completed_at_utc": _utc_now(),
            "attempt_root": str(attempt_root),
            "audit_node_order": list(EXPECTED_AUDIT_NODES),
            "final_acceptance": _artifact_ref(acceptance, attempt_root),
            "acceptance_binding": _artifact_ref(
                acceptance_binding,
                attempt_root,
            ),
            "bundle_manifest": _artifact_ref(bundle, attempt_root),
            "bundle_binding": _artifact_ref(bundle_binding, attempt_root),
            "code_manifest": _artifact_ref(code_manifest, attempt_root),
            "frozen_input_manifest": _artifact_ref(
                input_manifest,
                attempt_root,
            ),
            "frozen_input_source_preflight": frozen_input_payload["preflight"],
            "accepted_result_binding": accepted_binding_payload,
            "bundle_result_binding": bundle_binding_payload,
            "attempt_manifest": _artifact_ref(attempt_manifest, attempt_root),
            "events_snapshot": _artifact_ref(events_snapshot, attempt_root),
            "terminal_event": _artifact_ref(
                terminal_event_path,
                attempt_root,
            ),
            "named_gate_set_validation": _artifact_ref(
                attempt_root / "provenance/named_gate_set_validation.json",
                attempt_root,
            ),
            "final_artifact_graph_validation": _artifact_ref(
                attempt_root / "provenance/final_artifact_graph_validation.json",
                attempt_root,
            ),
            "final_acceptance_recheck": _artifact_ref(
                attempt_root / "provenance/final_acceptance_recheck_manifest.json",
                attempt_root,
            ),
            "artifact_checksum_index": _artifact_ref(index_path, attempt_root),
            "authoritative_only_when_terminal_status_ref_matches": True,
            "sealable_for_post_acceptance_release": True,
            "post_acceptance_git_release_chain_required": True,
            "stage_completion_claimed": False,
            "scientific_result_claimed": False,
        }
        completion_path = attempt_root / "driver_completion.json"
        _write_json_exclusive(completion_path, completion_payload)
        status_payload = {
            "schema_version": "b0_driver_status.v1",
            "state": "B0_DRIVER_COMPLETED",
            "updated_at_utc": _utc_now(),
            "current_node": None,
            "wrapper_pid": None,
            "terminal": True,
            "driver_completion": _artifact_ref(
                completion_path,
                attempt_root,
            ),
            "terminal_event": _artifact_ref(
                terminal_event_path,
                attempt_root,
            ),
        }
        _validate_ref(
            completion_path,
            status_payload["driver_completion"],
            label="precommit driver completion",
        )
        _validate_ref(
            terminal_event_path,
            status_payload["terminal_event"],
            label="precommit terminal success event",
        )
        if (attempt_root / "failure/failure.json").exists():
            raise GuardError("seal observed failure evidence before terminal commit")
        _validate_success_index(
            attempt_root,
            expected_reference=completion_payload["artifact_checksum_index"],
        )
        _write_json_atomic(status_path, status_payload)
        _ensure_terminal_success_event_logged(attempt_root)
        _validate_terminal_success_state(attempt_root)
        print(
            json.dumps(
                {
                    "driver_completion": str(completion_path),
                    "event": "B0_DRIVER_COMPLETED",
                    "final_acceptance": completion_payload["final_acceptance"]["path"],
                    "stage_completion_claimed": False,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--project-root", required=True)
    fingerprint.add_argument("--driver-path", required=True)
    fingerprint.add_argument("--output", required=True)
    fingerprint.add_argument("--sha256-output", required=True)
    fingerprint.add_argument("--expected-commit", required=True)
    fingerprint.add_argument("--expected-driver-sha256", required=True)
    fingerprint.add_argument("--expected-dirty-state-sha256", required=True)
    fingerprint.set_defaults(handler=_fingerprint_command)

    assertion = subparsers.add_parser("assert-fingerprint")
    assertion.add_argument("--baseline", required=True)
    assertion.add_argument("--project-root", required=True)
    assertion.add_argument("--driver-path", required=True)
    assertion.add_argument("--expected-baseline-sha256", required=True)
    assertion.add_argument("--label", required=True)
    assertion.add_argument("--observed-output", required=True)
    assertion.set_defaults(handler=_assert_fingerprint_command)

    freeze_inputs = subparsers.add_parser("freeze-inputs")
    freeze_inputs.add_argument("--preflight", required=True)
    freeze_inputs.add_argument("--node00-stdout", required=True)
    freeze_inputs.add_argument("--output", required=True)
    freeze_inputs.add_argument("--sha256-output", required=True)
    freeze_inputs.set_defaults(handler=_freeze_inputs_command)

    input_assertion = subparsers.add_parser("assert-inputs")
    input_assertion.add_argument("--input-manifest", required=True)
    input_assertion.add_argument(
        "--expected-input-manifest-sha256",
        required=True,
    )
    input_assertion.add_argument("--label", required=True)
    input_assertion.add_argument("--output", required=True)
    input_assertion.set_defaults(handler=_assert_inputs_command)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--expected-manifest-sha256", required=True)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=_preflight_command)

    init = subparsers.add_parser("init")
    init.add_argument("--attempt-root", required=True)
    init.add_argument("--project-root", required=True)
    init.add_argument("--d1-acceptance", required=True)
    init.add_argument("--approved-b0-parent", required=True)
    init.add_argument("--python-launcher", required=True)
    init.add_argument("--runtime-manifest", required=True)
    init.add_argument("--driver", required=True)
    init.add_argument("--guard", required=True)
    init.add_argument("--expected-commit", required=True)
    init.add_argument("--expected-driver-sha256", required=True)
    init.add_argument("--expected-dirty-state-sha256", required=True)
    init.add_argument("--expected-runtime-prefix", required=True)
    init.add_argument("--expected-runtime-manifest-sha256", required=True)
    init.add_argument("--minimum-free-bytes", type=int, required=True)
    init.set_defaults(handler=_init_command)

    event = subparsers.add_parser("event")
    event.add_argument("--attempt-root", required=True)
    event.add_argument("--event", required=True)
    event.add_argument("--node")
    event.add_argument("--detail")
    event.add_argument("--exit-code", type=int)
    event.set_defaults(handler=_event_command)

    status = subparsers.add_parser("status")
    status.add_argument("--attempt-root", required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--node")
    status.add_argument("--wrapper-pid", type=int)
    status.add_argument("--reason")
    status.add_argument("--terminal", action="store_true")
    status.set_defaults(handler=_status_command)

    failure = subparsers.add_parser("failure")
    failure.add_argument("--attempt-root", required=True)
    failure.add_argument("--exit-code", type=int, required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--node")
    failure.add_argument("--signal")
    failure.add_argument("--line", type=int)
    failure.add_argument("--command")
    failure.add_argument("--wrapper-pid", type=int)
    failure.set_defaults(handler=_failure_command)

    terminal_success = subparsers.add_parser("terminal-success")
    terminal_success.add_argument("--attempt-root", required=True)
    terminal_success.set_defaults(handler=_terminal_success_command)

    watchdog = subparsers.add_parser("watchdog")
    watchdog.add_argument("--attempt-root", required=True)
    watchdog.add_argument("--d1-acceptance", required=True)
    watchdog.add_argument("--parent-pid", type=int, required=True)
    watchdog.set_defaults(handler=_watchdog_command)

    watchdog_once = subparsers.add_parser("watchdog-once")
    watchdog_once.add_argument("--attempt-root", required=True)
    watchdog_once.add_argument("--d1-acceptance", required=True)
    watchdog_once.set_defaults(handler=_watchdog_once_command)

    validate_gate = subparsers.add_parser("validate-gate")
    validate_gate.add_argument("--gate", choices=NAMED_GATE_NAMES, required=True)
    validate_gate.add_argument("--label", required=True)
    validate_gate.add_argument("--artifact", required=True)
    validate_gate.add_argument("--evidence-output", required=True)
    validate_gate.add_argument("--expected-head")
    validate_gate.add_argument("--expected-dirty-state-sha256")
    validate_gate.add_argument("--expected-d1-acceptance")
    validate_gate.add_argument("--expected-canonical-validation")
    validate_gate.set_defaults(handler=_validate_gate_command)

    bind_result = subparsers.add_parser("bind-result")
    bind_result.add_argument(
        "--kind",
        choices=("acceptance", "bundle"),
        required=True,
    )
    bind_result.add_argument("--artifact", required=True)
    bind_result.add_argument("--output", required=True)
    bind_result.add_argument("--sha256-output", required=True)
    bind_result.set_defaults(handler=_bind_result_command)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--attempt-root", required=True)
    seal.add_argument("--final-acceptance", required=True)
    seal.add_argument("--bundle-manifest", required=True)
    seal.add_argument("--code-manifest", required=True)
    seal.add_argument("--input-manifest", required=True)
    seal.add_argument("--acceptance-binding", required=True)
    seal.add_argument("--bundle-binding", required=True)
    seal.add_argument("--expected-attempt-manifest-sha256", required=True)
    seal.add_argument("--expected-code-manifest-sha256", required=True)
    seal.add_argument("--expected-input-manifest-sha256", required=True)
    seal.add_argument("--expected-acceptance-binding-sha256", required=True)
    seal.add_argument("--expected-bundle-binding-sha256", required=True)
    seal.set_defaults(handler=_seal_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (GuardError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state": "FAILED_WITH_EVIDENCE",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
