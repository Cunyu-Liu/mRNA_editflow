#!/usr/bin/env python3
"""Run one command in a fresh, fail-closed audit evidence root.

This wrapper records process, Git, Python, CUDA-applicability, stdout, stderr,
and exit-code evidence.  It never interprets a zero command exit as scientific
or phase-gate acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


NON_NEURAL = "NON_NEURAL_DATA_BENCHMARK"
NEURAL = "NEURAL_GPU"
WORKLOAD_CLASSES = (NON_NEURAL, NEURAL)
SCHEMA_VERSION = "audited_command_run.v1"

CUDA_HEALTH_FIELDS = (
    "torch_cuda_available",
    "model_parameters_on_cuda",
    "input_batch_on_cuda",
    "real_forward_on_cuda",
    "real_backward_on_cuda",
    "optimizer_update_completed",
    "max_memory_allocated_gt_zero",
    "cpu_fallback_count_zero",
)

CUDA_FAILURE_EXIT = 70
LAUNCH_FAILURE_EXIT = 71
AUDIT_FAILURE_EXIT = 72
EXISTING_ROOT_EXIT = 73


class AuditSnapshotError(RuntimeError):
    """Raised when an exact pre-launch audit snapshot cannot be completed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Create and fsync one evidence file without replacing anything."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Any) -> None:
    _write_bytes_exclusive(path, _json_bytes(payload))


def _artifact_ref(path: Path, run_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AuditSnapshotError(
            f"git {' '.join(args)} failed with exit {completed.returncode}: {error}"
        )
    return completed.stdout


def _stable_regular_file_hash(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AuditSnapshotError(
            f"cannot open untracked regular file {path}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AuditSnapshotError(
                f"untracked path changed type while hashing: {path}"
            )
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    observed_fields = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if stable_fields != observed_fields or total != after.st_size:
        raise AuditSnapshotError(f"untracked file changed while hashing: {path}")
    return total, digest.hexdigest()


def _untracked_content_manifest(
    git_root: Path,
    raw_paths: bytes,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    raw_entries = raw_paths.split(b"\0")
    if raw_entries and raw_entries[-1] == b"":
        raw_entries.pop()
    for raw_relative in raw_entries:
        relative = os.fsdecode(raw_relative)
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise AuditSnapshotError(
                f"git returned unsafe untracked path: {relative!r}"
            )
        path = git_root.joinpath(*pure.parts)
        try:
            before = path.lstat()
        except OSError as exc:
            raise AuditSnapshotError(
                f"cannot stat untracked path {relative!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if stat.S_ISREG(before.st_mode):
            byte_count, content_sha256 = _stable_regular_file_hash(path)
            kind = "regular_file"
        elif stat.S_ISLNK(before.st_mode):
            try:
                target = os.readlink(path)
                after = path.lstat()
            except OSError as exc:
                raise AuditSnapshotError(
                    f"cannot read untracked symlink {relative!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if (
                before.st_dev,
                before.st_ino,
                before.st_mtime_ns,
                before.st_mode,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
                after.st_mode,
            ):
                raise AuditSnapshotError(
                    f"untracked symlink changed while hashing: {relative!r}"
                )
            target_bytes = os.fsencode(target)
            byte_count = len(target_bytes)
            content_sha256 = _sha256_bytes(target_bytes)
            kind = "symbolic_link_target"
        else:
            raise AuditSnapshotError(
                f"unsupported untracked path type for {relative!r}"
            )
        entries.append(
            {
                "path": relative,
                "path_bytes_sha256": _sha256_bytes(raw_relative),
                "kind": kind,
                "mode": stat.S_IMODE(before.st_mode),
                "bytes": byte_count,
                "sha256": content_sha256,
            }
        )
    canonical_entries = json.dumps(
        entries,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "git_untracked_content.v1",
        "path_encoding": "os.fsdecode with JSON escaping",
        "entry_count": len(entries),
        "entries": entries,
        "entries_sha256": _sha256_bytes(canonical_entries),
    }


def _explicit_prelaunch_file_manifest(
    git_root: Path,
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Hash caller-selected repository files, including ignored files."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_relative in relative_paths:
        if not isinstance(raw_relative, str) or not raw_relative:
            raise AuditSnapshotError(
                "explicit prelaunch file path must be a non-empty string"
            )
        pure = PurePosixPath(raw_relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise AuditSnapshotError(
                f"unsafe explicit prelaunch file path: {raw_relative!r}"
            )
        relative = pure.as_posix()
        if relative in seen:
            raise AuditSnapshotError(
                f"duplicate explicit prelaunch file path: {relative!r}"
            )
        seen.add(relative)
        cursor = git_root
        for part in pure.parts:
            cursor /= part
            try:
                metadata = cursor.lstat()
            except OSError as exc:
                raise AuditSnapshotError(
                    f"cannot stat explicit prelaunch path {relative!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise AuditSnapshotError(
                    f"symlinked explicit prelaunch path is forbidden: {relative!r}"
                )
        if not cursor.is_file():
            raise AuditSnapshotError(
                f"explicit prelaunch path is not a regular file: {relative!r}"
            )
        byte_count, content_sha256 = _stable_regular_file_hash(cursor)
        entries.append(
            {
                "path": relative,
                "kind": "regular_file",
                "mode": stat.S_IMODE(cursor.stat().st_mode),
                "bytes": byte_count,
                "sha256": content_sha256,
            }
        )
    entries.sort(key=lambda item: item["path"])
    canonical_entries = json.dumps(
        entries,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "git_explicit_prelaunch_files.v1",
        "entry_count": len(entries),
        "entries": entries,
        "entries_sha256": _sha256_bytes(canonical_entries),
    }


def _index_flag_manifest(raw_entries: bytes) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    values = raw_entries.split(b"\0")
    if values and values[-1] == b"":
        values.pop()
    for raw in values:
        if len(raw) < 3 or raw[1:2] != b" ":
            raise AuditSnapshotError("git ls-files -v returned an invalid entry")
        tag = raw[:1].decode("ascii", errors="strict")
        raw_path = raw[2:]
        entries.append(
            {
                "tag": tag,
                "path": os.fsdecode(raw_path),
                "path_bytes_sha256": _sha256_bytes(raw_path),
                "safe_normal_index_entry": tag == "H",
            }
        )
    unsafe = [item for item in entries if item["safe_normal_index_entry"] is False]
    return {
        "schema_version": "git_index_flags.v1",
        "entry_count": len(entries),
        "entries": entries,
        "unsafe_entries": unsafe,
        "all_entries_normal": not unsafe,
    }


def _recheck_explicit_prelaunch_files(
    git_root: Path,
    captured_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash explicit controls immediately before the child may launch."""
    captured_entries = captured_manifest.get("entries")
    if not isinstance(captured_entries, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("path"), str)
        for item in captured_entries
    ):
        raise AuditSnapshotError("captured explicit prelaunch manifest is invalid")
    relative_paths = [str(item["path"]) for item in captured_entries]
    observed = _explicit_prelaunch_file_manifest(git_root, relative_paths)
    captured_sha256 = _sha256_bytes(_json_bytes(dict(captured_manifest)))
    observed_sha256 = _sha256_bytes(_json_bytes(observed))
    return {
        "schema_version": "git_explicit_prelaunch_recheck.v1",
        "checked_at_utc": _utc_now(),
        "checked_immediately_before_child_launch": True,
        "captured_manifest_sha256": captured_sha256,
        "observed_manifest_sha256": observed_sha256,
        "entry_count": len(relative_paths),
        "matches": observed == dict(captured_manifest),
    }


def _capture_git_snapshot(
    project_root: Path,
    *,
    prelaunch_bind_files: Sequence[str] = (),
) -> dict[str, Any]:
    captured_at_utc = _utc_now()
    git_root_raw = _git_bytes(project_root, "rev-parse", "--show-toplevel")
    git_root_text = git_root_raw.decode("utf-8", errors="surrogateescape").rstrip("\n")
    if not git_root_text:
        raise AuditSnapshotError("git returned an empty worktree root")
    git_root = Path(git_root_text).resolve(strict=True)
    head = (
        _git_bytes(git_root, "rev-parse", "--verify", "HEAD")
        .decode("ascii", errors="strict")
        .strip()
    )
    status_bytes = _git_bytes(
        git_root,
        "status",
        "--porcelain=v2",
        "--branch",
        "-z",
    )
    diff_bytes = _git_bytes(
        git_root,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    untracked_paths = _git_bytes(
        git_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_manifest = _untracked_content_manifest(git_root, untracked_paths)
    index_flags_raw = _git_bytes(git_root, "ls-files", "-v", "-z")
    index_flags = _index_flag_manifest(index_flags_raw)
    explicit_prelaunch_manifest = _explicit_prelaunch_file_manifest(
        git_root,
        prelaunch_bind_files,
    )
    component_hashes = {
        "head": _sha256_bytes((head + "\n").encode("ascii")),
        "status_porcelain_v2_z": _sha256_bytes(status_bytes),
        "diff_head_binary": _sha256_bytes(diff_bytes),
        "untracked_paths_z": _sha256_bytes(untracked_paths),
        "untracked_content_manifest": _sha256_bytes(_json_bytes(untracked_manifest)),
        "explicit_prelaunch_file_manifest": _sha256_bytes(
            _json_bytes(explicit_prelaunch_manifest)
        ),
        "index_flags": _sha256_bytes(_json_bytes(index_flags)),
    }
    dirty_state_sha256 = _sha256_bytes(_json_bytes(component_hashes))
    return {
        "captured_at_utc": captured_at_utc,
        "git_root": git_root,
        "head": head,
        "status_bytes": status_bytes,
        "diff_bytes": diff_bytes,
        "untracked_paths": untracked_paths,
        "untracked_manifest": untracked_manifest,
        "explicit_prelaunch_manifest": explicit_prelaunch_manifest,
        "index_flags": index_flags,
        "component_hashes": component_hashes,
        "dirty_state_sha256": dirty_state_sha256,
        "clean": (
            not bool(diff_bytes or untracked_paths)
            and index_flags["all_entries_normal"] is True
        ),
    }


def _write_git_snapshot(
    run_root: Path,
    captured: Mapping[str, Any],
) -> dict[str, Any]:
    git_dir = run_root / "git"
    head_path = git_dir / "head.txt"
    status_path = git_dir / "status.porcelain_v2.z"
    diff_path = git_dir / "diff.head.binary.patch"
    untracked_paths_path = git_dir / "untracked.paths.z"
    untracked_manifest_path = git_dir / "untracked_content_hashes.json"
    explicit_prelaunch_path = git_dir / "explicit_prelaunch_files.json"
    index_flags_path = git_dir / "index_flags.json"
    _write_bytes_exclusive(head_path, (str(captured["head"]) + "\n").encode("ascii"))
    _write_bytes_exclusive(status_path, bytes(captured["status_bytes"]))
    _write_bytes_exclusive(diff_path, bytes(captured["diff_bytes"]))
    _write_bytes_exclusive(
        untracked_paths_path,
        bytes(captured["untracked_paths"]),
    )
    _write_json_exclusive(
        untracked_manifest_path,
        captured["untracked_manifest"],
    )
    _write_json_exclusive(
        explicit_prelaunch_path,
        captured["explicit_prelaunch_manifest"],
    )
    _write_json_exclusive(index_flags_path, captured["index_flags"])
    summary = {
        "schema_version": "git_prelaunch_snapshot.v1",
        "captured_before_command": True,
        "captured_at_utc": captured["captured_at_utc"],
        "repository": str(captured["git_root"]),
        "head": captured["head"],
        "clean": captured["clean"],
        "dirty_state_sha256": captured["dirty_state_sha256"],
        "index_flags_safe": captured["index_flags"]["all_entries_normal"],
        "component_hashes": captured["component_hashes"],
        "artifacts": {
            "head": _artifact_ref(head_path, run_root),
            "status_porcelain_v2_z": _artifact_ref(status_path, run_root),
            "diff_head_binary": _artifact_ref(diff_path, run_root),
            "untracked_paths_z": _artifact_ref(
                untracked_paths_path,
                run_root,
            ),
            "untracked_content_hashes": _artifact_ref(
                untracked_manifest_path,
                run_root,
            ),
            "explicit_prelaunch_files": _artifact_ref(
                explicit_prelaunch_path,
                run_root,
            ),
            "index_flags": _artifact_ref(index_flags_path, run_root),
        },
    }
    snapshot_path = git_dir / "snapshot.json"
    _write_json_exclusive(snapshot_path, summary)
    return {
        **summary,
        "snapshot_manifest": _artifact_ref(snapshot_path, run_root),
    }


def _python_evidence(
    command: list[str], environment: Mapping[str, str]
) -> dict[str, Any]:
    resolved = shutil.which(command[0], path=environment.get("PATH"))
    return {
        "applicability": "APPLICABLE_TO_AUDIT_WRAPPER_RUNTIME",
        "wrapper_executable": sys.executable,
        "wrapper_version": platform.python_version(),
        "wrapper_implementation": platform.python_implementation(),
        "child_argv0": command[0],
        "child_argv0_resolved": resolved,
        "child_argv0_looks_like_python": Path(command[0]).name.startswith("python"),
        "child_python_version_not_inferred": True,
    }


def _default_cuda_probe() -> dict[str, Any]:
    try:
        from scripts.execution.launch_gpu_run import probe_cuda
    except ModuleNotFoundError:  # direct script execution from scripts/execution
        from launch_gpu_run import probe_cuda
    return probe_cuda()


def _non_neural_cuda_statement() -> dict[str, Any]:
    return {
        "applicability": "NOT_APPLICABLE_NON_NEURAL_WORKLOAD",
        "formal_neural_activity": False,
        "gpu_validation_started": False,
        "probe_executed": False,
        "cuda_visible_devices_for_child": "",
        "gpu_launched_by_wrapper": False,
        "automatic_cpu_fallback": False,
        "statement": (
            "This workload is non-neural. The wrapper intentionally did not "
            "probe or launch CUDA and masked CUDA_VISIBLE_DEVICES for the child."
        ),
    }


def _neural_cuda_preflight(
    probe: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        report = dict(probe())
    except Exception as exc:  # preserve probe failures as evidence
        report = {
            **{field: False for field in CUDA_HEALTH_FIELDS},
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    report["passed"] = all(report.get(field) is True for field in CUDA_HEALTH_FIELDS)
    return {
        "applicability": "REQUIRED_FORMAL_NEURAL_WORKLOAD",
        "formal_neural_activity": True,
        "gpu_validation_started": True,
        "probe_executed": True,
        "gpu_launched_by_wrapper": bool(report["passed"]),
        "automatic_cpu_fallback": False,
        "preflight": report,
    }


def _load_actual_cuda_health(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            **{field: False for field in CUDA_HEALTH_FIELDS},
            "passed": False,
            "error": "MISSING_ACTUAL_COMMAND_CUDA_HEALTH",
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **{field: False for field in CUDA_HEALTH_FIELDS},
            "passed": False,
            "error": f"INVALID_ACTUAL_COMMAND_CUDA_HEALTH: {type(exc).__name__}",
        }
    if not isinstance(payload, dict):
        return {
            **{field: False for field in CUDA_HEALTH_FIELDS},
            "passed": False,
            "error": "ACTUAL_COMMAND_CUDA_HEALTH_NOT_OBJECT",
        }
    payload["passed"] = all(payload.get(field) is True for field in CUDA_HEALTH_FIELDS)
    if not payload["passed"] and not payload.get("error"):
        payload["error"] = "ACTUAL_COMMAND_CUDA_HEALTH_FIELD_FALSE_OR_MISSING"
    return payload


def _child_wrapper_exit(returncode: int) -> int:
    if returncode == 0:
        return 0
    if returncode < 0:
        return min(255, 128 + abs(returncode))
    return min(255, returncode)


def _prepare_run_root(run_root: Path) -> Path:
    resolved = run_root.expanduser().resolve(strict=False)
    if not resolved.parent.is_dir():
        raise FileNotFoundError(
            f"audit run-root parent does not exist: {resolved.parent}"
        )
    resolved.mkdir(mode=0o750, exist_ok=False)
    for child in ("git", "logs"):
        (resolved / child).mkdir(mode=0o750, exist_ok=False)
    return resolved


def _finalize(
    *,
    run_root: Path,
    started_at_utc: str,
    started_monotonic: float,
    workload_class: str,
    command: list[str],
    project_root: Path,
    working_directory: Path,
    python_evidence: Mapping[str, Any],
    cuda_evidence: Mapping[str, Any],
    git_evidence: Mapping[str, Any] | None,
    wrapper_exit_code: int,
    state: str,
    stop_reason: str | None,
    child_pid: int | None,
    child_exit_code: int | None,
    interrupted_by_signal: int | None = None,
) -> int:
    ended_at_utc = _utc_now()
    completion_path = run_root / "completion.json"
    completion = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "stop_reason": stop_reason,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "wrapper_pid": os.getpid(),
        "child_pid": child_pid,
        "observed_process_exit_code": child_exit_code,
        "wrapper_exit_code": wrapper_exit_code,
        "interrupted_by_signal": interrupted_by_signal,
        "zero_exit_is_phase_acceptance": False,
    }
    _write_json_exclusive(completion_path, completion)

    evidence: dict[str, Any] = {
        "invocation": _artifact_ref(run_root / "invocation.json", run_root),
        "completion": _artifact_ref(completion_path, run_root),
        "stdout": _artifact_ref(run_root / "logs/stdout.log", run_root),
        "stderr": _artifact_ref(run_root / "logs/stderr.log", run_root),
    }
    process_path = run_root / "process.json"
    if process_path.is_file():
        evidence["process"] = _artifact_ref(process_path, run_root)
    cuda_preflight_path = run_root / "logs/cuda_preflight.json"
    if cuda_preflight_path.is_file():
        evidence["cuda_preflight"] = _artifact_ref(
            cuda_preflight_path,
            run_root,
        )
    actual_health_path = run_root / "logs/cuda_health.json"
    if actual_health_path.is_file():
        evidence["actual_cuda_health"] = _artifact_ref(
            actual_health_path,
            run_root,
        )
    git_error_path = run_root / "git/snapshot_error.json"
    if git_error_path.is_file():
        evidence["git_snapshot_error"] = _artifact_ref(
            git_error_path,
            run_root,
        )
    git_binding_failure_path = run_root / "git/prelaunch_binding_failure.json"
    if git_binding_failure_path.is_file():
        evidence["git_prelaunch_binding_failure"] = _artifact_ref(
            git_binding_failure_path,
            run_root,
        )
    explicit_recheck_path = run_root / "git/explicit_prelaunch_recheck.json"
    if explicit_recheck_path.is_file():
        evidence["explicit_prelaunch_recheck"] = _artifact_ref(
            explicit_recheck_path,
            run_root,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "run_root": str(run_root),
        "workload_class": workload_class,
        "state": state,
        "stop_reason": stop_reason,
        "argv": command,
        "shell_used": False,
        "project_root": str(project_root),
        "working_directory": str(working_directory),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "wrapper_pid": os.getpid(),
        "child_pid": child_pid,
        "observed_process_exit_code": child_exit_code,
        "wrapper_exit_code": wrapper_exit_code,
        "python": dict(python_evidence),
        "cuda": dict(cuda_evidence),
        "git_prelaunch_snapshot": (
            dict(git_evidence) if git_evidence is not None else None
        ),
        "evidence": evidence,
        "protection": {
            "exclusive_new_run_root": True,
            "existing_results_overwritten": 0,
            "unrelated_processes_terminated": 0,
            "only_exact_child_pid_may_receive_interrupt": True,
        },
        "claim_boundary": {
            "command_exit_zero_is_phase_gate": False,
            "smoke_or_proxy_is_final_scientific_evidence": False,
        },
    }
    _write_json_exclusive(run_root / "audit_manifest.json", manifest)
    return wrapper_exit_code


def run_audited_command(
    *,
    run_root: Path,
    project_root: Path,
    command: list[str],
    workload_class: str,
    working_directory: Path | None = None,
    cuda_probe: Callable[[], Mapping[str, Any]] | None = None,
    expected_git_head: str | None = None,
    expected_git_dirty_state_sha256: str | None = None,
    prelaunch_bind_files: Sequence[str] = (),
) -> int:
    """Run ``command`` once and preserve immutable audit evidence."""
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError(f"unsupported workload class: {workload_class}")
    if not command:
        raise ValueError("audited command cannot be empty")
    if (
        expected_git_head is not None
        and re.fullmatch(r"[0-9a-f]{40}", expected_git_head) is None
    ):
        raise ValueError("expected_git_head must be 40 lowercase hex characters")
    if (
        expected_git_dirty_state_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_git_dirty_state_sha256) is None
    ):
        raise ValueError(
            "expected_git_dirty_state_sha256 must be 64 lowercase hex characters"
        )
    project_root = project_root.expanduser().resolve(strict=True)
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)
    working_directory = (
        working_directory.expanduser().resolve(strict=True)
        if working_directory is not None
        else project_root
    )
    if not working_directory.is_dir():
        raise NotADirectoryError(working_directory)

    run_root = _prepare_run_root(run_root)
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    child: subprocess.Popen[bytes] | None = None
    child_exit_code: int | None = None
    interrupted_by_signal: int | None = None
    previous_signal_handlers: dict[int, Any] = {}
    signal_handlers_restored = False

    def forward_exact_child(signum: int, _frame: Any) -> None:
        nonlocal interrupted_by_signal
        if interrupted_by_signal is None:
            interrupted_by_signal = signum
        active_child = child
        if active_child is not None and active_child.poll() is None:
            try:
                active_child.send_signal(signum)
            except ProcessLookupError:
                pass

    def restore_signal_handlers() -> None:
        nonlocal signal_handlers_restored
        if signal_handlers_restored:
            return
        for handled_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(handled_signal, previous_handler)
        signal_handlers_restored = True

    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_signal_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, forward_exact_child)

    def finalize_with_signal_restore(**kwargs: Any) -> int:
        restore_signal_handlers()
        return _finalize(**kwargs)

    stdout_path = run_root / "logs/stdout.log"
    stderr_path = run_root / "logs/stderr.log"
    environment = dict(os.environ)
    environment["EDITFLOW_AUDIT_RUN_ROOT"] = str(run_root)
    environment["EDITFLOW_WORKLOAD_CLASS"] = workload_class

    try:
        captured_git = _capture_git_snapshot(
            project_root,
            prelaunch_bind_files=prelaunch_bind_files,
        )
        git_evidence = _write_git_snapshot(run_root, captured_git)
    except Exception as exc:
        _write_bytes_exclusive(stdout_path, b"")
        _write_bytes_exclusive(stderr_path, b"")
        _write_json_exclusive(
            run_root / "git/snapshot_error.json",
            {
                "captured_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "command_started": False,
            },
        )
        python_evidence = _python_evidence(command, environment)
        cuda_evidence = {
            "applicability": (
                "NOT_APPLICABLE_NON_NEURAL_WORKLOAD"
                if workload_class == NON_NEURAL
                else "REQUIRED_BUT_NOT_RUN_DUE_TO_GIT_AUDIT_FAILURE"
            ),
            "probe_executed": False,
            "gpu_launched_by_wrapper": False,
            "automatic_cpu_fallback": False,
        }
        _write_json_exclusive(
            run_root / "invocation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "started_at_utc": started_at_utc,
                "wrapper_pid": os.getpid(),
                "argv": command,
                "workload_class": workload_class,
                "command_started": False,
                "git_snapshot_error": f"{type(exc).__name__}: {exc}",
                "python": python_evidence,
                "cuda": cuda_evidence,
            },
        )
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=None,
            wrapper_exit_code=AUDIT_FAILURE_EXIT,
            state="FAILED_WITH_EVIDENCE",
            stop_reason=f"GIT_PRELAUNCH_SNAPSHOT_FAILED_{type(exc).__name__}",
            child_pid=None,
            child_exit_code=None,
        )

    git_binding_checks = {
        "index_flags_safe": {
            "applicable": True,
            "expected": True,
            "observed": git_evidence["index_flags_safe"],
            "passed": git_evidence["index_flags_safe"] is True,
        },
        "head": {
            "applicable": expected_git_head is not None,
            "expected": expected_git_head,
            "observed": git_evidence["head"],
            "passed": (
                expected_git_head is None or git_evidence["head"] == expected_git_head
            ),
        },
        "dirty_state_sha256": {
            "applicable": expected_git_dirty_state_sha256 is not None,
            "expected": expected_git_dirty_state_sha256,
            "observed": git_evidence["dirty_state_sha256"],
            "passed": (
                expected_git_dirty_state_sha256 is None
                or git_evidence["dirty_state_sha256"] == expected_git_dirty_state_sha256
            ),
        },
    }
    if not all(check["passed"] for check in git_binding_checks.values()):
        _write_bytes_exclusive(stdout_path, b"")
        _write_bytes_exclusive(stderr_path, b"")
        binding_failure = {
            "schema_version": "git_prelaunch_binding.v1",
            "captured_at_utc": _utc_now(),
            "command_started": False,
            "checks": git_binding_checks,
        }
        _write_json_exclusive(
            run_root / "git/prelaunch_binding_failure.json",
            binding_failure,
        )
        python_evidence = _python_evidence(command, environment)
        cuda_evidence = {
            "applicability": (
                "NOT_APPLICABLE_NON_NEURAL_WORKLOAD"
                if workload_class == NON_NEURAL
                else "REQUIRED_BUT_NOT_RUN_DUE_TO_GIT_BINDING_FAILURE"
            ),
            "probe_executed": False,
            "gpu_launched_by_wrapper": False,
            "automatic_cpu_fallback": False,
        }
        _write_json_exclusive(
            run_root / "invocation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "started_at_utc": started_at_utc,
                "wrapper_pid": os.getpid(),
                "argv": command,
                "workload_class": workload_class,
                "command_started": False,
                "git_prelaunch_snapshot": git_evidence,
                "git_prelaunch_binding": binding_failure,
                "python": python_evidence,
                "cuda": cuda_evidence,
            },
        )
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=AUDIT_FAILURE_EXIT,
            state="FAILED_WITH_EVIDENCE",
            stop_reason="GIT_PRELAUNCH_BINDING_MISMATCH",
            child_pid=None,
            child_exit_code=None,
        )

    if workload_class == NON_NEURAL:
        cuda_evidence = _non_neural_cuda_statement()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["EDITFLOW_REQUIRE_CUDA"] = "0"
    else:
        cuda_evidence = _neural_cuda_preflight(cuda_probe or _default_cuda_probe)
        environment["EDITFLOW_REQUIRE_CUDA"] = "1"
        environment["EDITFLOW_CUDA_HEALTH_FILE"] = str(
            run_root / "logs/cuda_health.json"
        )
        _write_json_exclusive(
            run_root / "logs/cuda_preflight.json",
            cuda_evidence["preflight"],
        )

    python_evidence = _python_evidence(command, environment)
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "run_root": str(run_root),
        "started_at_utc": started_at_utc,
        "wrapper_pid": os.getpid(),
        "argv": command,
        "shell_used": False,
        "project_root": str(project_root),
        "working_directory": str(working_directory),
        "workload_class": workload_class,
        "environment_overrides": {
            "EDITFLOW_AUDIT_RUN_ROOT": str(run_root),
            "EDITFLOW_WORKLOAD_CLASS": workload_class,
            "EDITFLOW_REQUIRE_CUDA": environment["EDITFLOW_REQUIRE_CUDA"],
            **(
                {"CUDA_VISIBLE_DEVICES": ""}
                if workload_class == NON_NEURAL
                else {
                    "EDITFLOW_CUDA_HEALTH_FILE": environment[
                        "EDITFLOW_CUDA_HEALTH_FILE"
                    ]
                }
            ),
        },
        "python": python_evidence,
        "cuda": cuda_evidence,
        "git_prelaunch_snapshot": git_evidence,
        "command_started": False,
        "zero_exit_is_phase_acceptance": False,
    }
    _write_json_exclusive(run_root / "invocation.json", invocation)

    if workload_class == NEURAL and not cuda_evidence["preflight"]["passed"]:
        _write_bytes_exclusive(stdout_path, b"")
        _write_bytes_exclusive(stderr_path, b"")
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=CUDA_FAILURE_EXIT,
            state="FAILED_WITH_EVIDENCE",
            stop_reason="CUDA_PREFLIGHT_FAILED",
            child_pid=None,
            child_exit_code=None,
        )

    try:
        explicit_prelaunch_recheck = _recheck_explicit_prelaunch_files(
            Path(str(git_evidence["repository"])),
            captured_git["explicit_prelaunch_manifest"],
        )
    except Exception as exc:
        explicit_prelaunch_recheck = {
            "schema_version": "git_explicit_prelaunch_recheck.v1",
            "checked_at_utc": _utc_now(),
            "checked_immediately_before_child_launch": True,
            "entry_count": None,
            "matches": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json_exclusive(
        run_root / "git/explicit_prelaunch_recheck.json",
        explicit_prelaunch_recheck,
    )
    if explicit_prelaunch_recheck["matches"] is not True:
        _write_bytes_exclusive(stdout_path, b"")
        _write_bytes_exclusive(stderr_path, b"")
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=AUDIT_FAILURE_EXIT,
            state="FAILED_WITH_EVIDENCE",
            stop_reason="EXPLICIT_PRELAUNCH_FILE_RECHECK_FAILED",
            child_pid=None,
            child_exit_code=None,
        )

    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            if interrupted_by_signal is None:
                child = subprocess.Popen(
                    command,
                    cwd=working_directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                )
                if interrupted_by_signal is not None and child.poll() is None:
                    child.send_signal(interrupted_by_signal)
                _write_json_exclusive(
                    run_root / "process.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "launched_at_utc": _utc_now(),
                        "wrapper_pid": os.getpid(),
                        "child_pid": child.pid,
                        "argv": command,
                    },
                )
                try:
                    child_exit_code = child.wait()
                except KeyboardInterrupt:
                    interrupted_by_signal = signal.SIGINT
                    child.send_signal(signal.SIGINT)
                    child_exit_code = child.wait()
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except OSError as exc:
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=LAUNCH_FAILURE_EXIT,
            state="FAILED_WITH_EVIDENCE",
            stop_reason=f"COMMAND_LAUNCH_FAILED_{type(exc).__name__}",
            child_pid=None,
            child_exit_code=None,
        )
    finally:
        restore_signal_handlers()

    if child is None:
        assert interrupted_by_signal is not None
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=min(255, 128 + interrupted_by_signal),
            state="FAILED_WITH_EVIDENCE",
            stop_reason=f"INTERRUPTED_BEFORE_COMMAND_SIGNAL_{interrupted_by_signal}",
            child_pid=None,
            child_exit_code=None,
            interrupted_by_signal=interrupted_by_signal,
        )

    assert child_exit_code is not None
    if child_exit_code != 0:
        wrapper_exit = _child_wrapper_exit(child_exit_code)
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=wrapper_exit,
            state="FAILED_WITH_EVIDENCE",
            stop_reason=(
                f"INTERRUPTED_BY_SIGNAL_{interrupted_by_signal}"
                if interrupted_by_signal is not None
                else f"COMMAND_EXIT_{child_exit_code}"
            ),
            child_pid=child.pid,
            child_exit_code=child_exit_code,
            interrupted_by_signal=interrupted_by_signal,
        )
    if interrupted_by_signal is not None:
        return finalize_with_signal_restore(
            run_root=run_root,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            workload_class=workload_class,
            command=command,
            project_root=project_root,
            working_directory=working_directory,
            python_evidence=python_evidence,
            cuda_evidence=cuda_evidence,
            git_evidence=git_evidence,
            wrapper_exit_code=min(255, 128 + interrupted_by_signal),
            state="FAILED_WITH_EVIDENCE",
            stop_reason=f"INTERRUPTED_BY_SIGNAL_{interrupted_by_signal}",
            child_pid=child.pid,
            child_exit_code=child_exit_code,
            interrupted_by_signal=interrupted_by_signal,
        )

    if workload_class == NEURAL:
        actual_health = _load_actual_cuda_health(run_root / "logs/cuda_health.json")
        cuda_evidence = {
            **cuda_evidence,
            "actual_command_health": actual_health,
        }
        if not actual_health["passed"]:
            return finalize_with_signal_restore(
                run_root=run_root,
                started_at_utc=started_at_utc,
                started_monotonic=started_monotonic,
                workload_class=workload_class,
                command=command,
                project_root=project_root,
                working_directory=working_directory,
                python_evidence=python_evidence,
                cuda_evidence=cuda_evidence,
                git_evidence=git_evidence,
                wrapper_exit_code=CUDA_FAILURE_EXIT,
                state="FAILED_WITH_EVIDENCE",
                stop_reason="ACTUAL_COMMAND_CUDA_HEALTH_FAILED",
                child_pid=child.pid,
                child_exit_code=child_exit_code,
            )

    return finalize_with_signal_restore(
        run_root=run_root,
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        workload_class=workload_class,
        command=command,
        project_root=project_root,
        working_directory=working_directory,
        python_evidence=python_evidence,
        cuda_evidence=cuda_evidence,
        git_evidence=git_evidence,
        wrapper_exit_code=0,
        state="COMMAND_COMPLETED",
        stop_reason=None,
        child_pid=child.pid,
        child_exit_code=child_exit_code,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="fresh audit-only directory; its existing parent must already exist",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument(
        "--workload-class",
        choices=WORKLOAD_CLASSES,
        required=True,
    )
    parser.add_argument("--expected-git-head")
    parser.add_argument("--expected-git-dirty-state-sha256")
    parser.add_argument(
        "--prelaunch-bind-file",
        action="append",
        default=[],
        help=(
            "repository-relative regular file to hash before child launch; "
            "repeat for ignored or otherwise explicitly bound control files"
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="exact argv after '--'; no shell interpretation is performed",
    )
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return run_audited_command(
            run_root=args.run_root,
            project_root=args.project_root,
            working_directory=args.working_directory,
            command=command,
            workload_class=args.workload_class,
            expected_git_head=args.expected_git_head,
            expected_git_dirty_state_sha256=(args.expected_git_dirty_state_sha256),
            prelaunch_bind_files=args.prelaunch_bind_file,
        )
    except FileExistsError as exc:
        print(
            json.dumps(
                {
                    "state": "REFUSED_EXISTING_RUN_ROOT",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXISTING_ROOT_EXIT
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "state": "REFUSED_INVALID_INVOCATION",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return AUDIT_FAILURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
