#!/usr/bin/env python3
"""Fail-closed, stdout-only, read-only preflight for a future R3X remote run.

The collector is intentionally narrower than a general remote-inspection tool:

* every data/artifact/contract path must be absolute, explicitly named, and
  confined to an explicitly allowed read root;
* directories are never walked and untracked Git files are never scanned;
* protected names, symlinks, hardlinks, special files, and mutable reads fail
  closed before their contents can be consumed;
* only a fixed allowlist of read-only Git, process, and NVIDIA queries can run;
* the only emitted artifact is one canonical JSON document on stdout.

This module does not connect to a server.  It is meant to be copied with the
reviewed code snapshot and invoked once on the server only after its exact
bytes have been verified.  It does not establish OS-level mount separation or
publication eligibility; those remain independent HOLD gates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Protocol, Sequence


SCHEMA_VERSION = "r3x_remote_readonly_preflight_v1"
COMPLETE_STATUS = "READ_ONLY_PREFLIGHT_COMPLETE_NO_EXECUTION_AUTHORIZATION"
HOLD_STATUS = "HOLD_READ_ONLY_PREFLIGHT"
REMOTE_EXECUTION_ENABLED = False
REMOTE_EXECUTION_HOLD_CODE = (
    "HOLD_REMOTE_PREFLIGHT_V1_DO_NOT_RUN_PENDING_CAPABILITY_SEPARATED_MANIFEST"
)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 20
DEFAULT_MAX_HASH_BYTES = 512 * 1024 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_STATUS_LINES = 20_000
MAX_JOB_RECORDS = 512
HASH_CHUNK_BYTES = 4 * 1024 * 1024

SAFE_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "PATH": "/usr/bin:/bin",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}

EXECUTABLE_CANDIDATES = {
    "git": (Path("/usr/bin/git"), Path("/bin/git")),
    "ps": (Path("/usr/bin/ps"), Path("/bin/ps")),
    "nvidia-smi": (Path("/usr/bin/nvidia-smi"), Path("/bin/nvidia-smi")),
}

GIT_READ_ONLY_PREFIX = (
    "--no-optional-locks",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.preloadIndex=false",
)

PROTECTED_SINGLE_TOKENS = (
    "sealed",
    "restricted",
    "gse246381",
)
PROTECTED_COMPOUND_TOKENS = (
    "rawoutcome",
    "outcomeraw",
    "internaltestoutcome",
    "outcomeinternaltest",
    "ordinaryoutcome",
    "outcomeauthority",
    "outcomerelease",
    "testlabel",
    "finallabel",
)

JOB_EXECUTABLE_RE = re.compile(
    r"^(?:python(?:[0-9.]*)?|torchrun|accelerate|deepspeed|srun|sbatch)$",
    re.IGNORECASE,
)
HEX_HEAD_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

BOUNDARIES = {
    "artifact_directory_recursion": False,
    "directory_creation": False,
    "gpu_allocation_or_workload_start": False,
    "network_access": False,
    "ordinary_internal_test_outcome_access": False,
    "protected_member_access": False,
    "sealed_access": False,
    "training_or_evaluation_start": False,
    "untracked_file_scan": False,
    "write_operations": False,
}


class PreflightError(RuntimeError):
    """Public, path-sanitized preflight failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


def _hold(
    code: str,
    message: str,
    **details: Any,
) -> NoReturn:
    raise PreflightError(code, message, details=details or None)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return one compact, key-sorted, strict-JSON stdout record."""

    try:
        return (
            json.dumps(
                dict(payload),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreflightError(
            "HOLD_NONCANONICAL_PREFLIGHT_PAYLOAD",
            "preflight result is not strict canonical JSON",
        ) from exc


def _normalized_security_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _protected_reason(value: str) -> str | None:
    squashed = _normalized_security_text(value)
    for token in PROTECTED_SINGLE_TOKENS:
        if token in squashed:
            return "PROTECTED_SINGLE_TOKEN"
    for token in PROTECTED_COMPOUND_TOKENS:
        if token in squashed:
            return "PROTECTED_OUTCOME_TOKEN"
    if "protected" in squashed and (
        "data" in squashed or "outcome" in squashed or "label" in squashed
    ):
        return "PROTECTED_DATA_TOKEN"
    return None


def _reject_protected_text(
    value: str,
    *,
    context: str,
    index: int | None = None,
) -> None:
    reason = _protected_reason(value)
    if reason is not None:
        details: dict[str, Any] = {"context": context, "reason": reason}
        if index is not None:
            details["index"] = index
        _hold(
            "HOLD_PROTECTED_PATH_OR_REFERENCE",
            "a protected path or reference was rejected before content access",
            **details,
        )


def _absolute_lexical_path(
    value: str | os.PathLike[str],
    *,
    context: str,
    index: int | None = None,
) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        _hold(
            "HOLD_INVALID_ABSOLUTE_PATH",
            "preflight paths must be non-empty absolute text paths",
            context=context,
            index=index,
        )
    _reject_protected_text(raw, context=context, index=index)
    path = Path(raw)
    if not path.is_absolute():
        _hold(
            "HOLD_INVALID_ABSOLUTE_PATH",
            "preflight paths must be absolute",
            context=context,
            index=index,
        )
    normalized = Path(os.path.normpath(raw))
    _reject_protected_text(str(normalized), context=context, index=index)
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _validate_no_symlink_ancestors(
    path: Path,
    *,
    context: str,
    index: int | None = None,
    allow_missing: bool,
) -> os.stat_result | None:
    """lstat every existing component without following any symlink."""

    anchor = Path(path.anchor)
    current = anchor
    final_stat: os.stat_result | None = None
    for component in path.parts[1:]:
        current = current / component
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return None
            _hold(
                "HOLD_REQUIRED_PATH_MISSING",
                "a required preflight path does not exist",
                context=context,
                index=index,
            )
        except OSError as exc:
            _hold(
                "HOLD_PATH_METADATA_UNREADABLE",
                "path metadata could not be inspected safely",
                context=context,
                index=index,
                errno=exc.errno,
            )
        if stat.S_ISLNK(current_stat.st_mode):
            _hold(
                "HOLD_SYMLINK_PATH_REJECTED",
                "symlink paths are not admitted by the read-only preflight",
                context=context,
                index=index,
            )
        final_stat = current_stat
    return final_stat


@dataclass(frozen=True)
class PreflightConfig:
    repo_root: Path
    read_roots: tuple[Path, ...]
    authority_contract_paths: tuple[Path, ...] = ()
    ordinary_data_paths: tuple[Path, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES


def _normalize_config(config: PreflightConfig) -> PreflightConfig:
    repo_root = _absolute_lexical_path(config.repo_root, context="repo_root")
    roots_input = config.read_roots or (repo_root,)
    roots = tuple(
        _absolute_lexical_path(value, context="read_root", index=index)
        for index, value in enumerate(roots_input)
    )
    if repo_root not in roots:
        roots = (repo_root,) + roots
    if len(set(roots)) != len(roots):
        _hold(
            "HOLD_DUPLICATE_READ_ROOT",
            "read roots must be unique after lexical normalization",
        )
    for index, root in enumerate(roots):
        root_stat = _validate_no_symlink_ancestors(
            root,
            context="read_root",
            index=index,
            allow_missing=False,
        )
        if root_stat is None or not stat.S_ISDIR(root_stat.st_mode):
            _hold(
                "HOLD_READ_ROOT_NOT_DIRECTORY",
                "every read root must be an existing real directory",
                index=index,
            )
    repo_stat = _validate_no_symlink_ancestors(
        repo_root,
        context="repo_root",
        allow_missing=False,
    )
    if repo_stat is None or not stat.S_ISDIR(repo_stat.st_mode):
        _hold(
            "HOLD_REPOSITORY_ROOT_NOT_DIRECTORY",
            "repository root must be an existing real directory",
        )
    if (
        isinstance(config.max_hash_bytes, bool)
        or not isinstance(config.max_hash_bytes, int)
        or config.max_hash_bytes <= 0
    ):
        _hold(
            "HOLD_INVALID_HASH_BOUND",
            "max_hash_bytes must be a positive integer",
        )

    def normalize_targets(values: Sequence[Path], kind: str) -> tuple[Path, ...]:
        targets = tuple(
            _absolute_lexical_path(value, context=kind, index=index)
            for index, value in enumerate(values)
        )
        if len(set(targets)) != len(targets):
            _hold(
                "HOLD_DUPLICATE_EXPLICIT_PATH",
                "explicit preflight paths must be unique within each category",
                context=kind,
            )
        for index, target in enumerate(targets):
            if not any(_is_within(target, root) for root in roots):
                _hold(
                    "HOLD_PATH_OUTSIDE_EXPLICIT_READ_ROOT",
                    "an explicit path is outside every allowed read root",
                    context=kind,
                    index=index,
                )
            _validate_no_symlink_ancestors(
                target,
                context=kind,
                index=index,
                allow_missing=True,
            )
        return targets

    return PreflightConfig(
        repo_root=repo_root,
        read_roots=roots,
        authority_contract_paths=normalize_targets(
            config.authority_contract_paths, "authority_contract"
        ),
        ordinary_data_paths=normalize_targets(
            config.ordinary_data_paths, "ordinary_data"
        ),
        artifact_paths=normalize_targets(config.artifact_paths, "artifact"),
        max_hash_bytes=config.max_hash_bytes,
    )


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes = b""


class CommandRunner(Protocol):
    def run(
        self,
        tool: str,
        arguments: Sequence[str],
        *,
        cwd: Path,
    ) -> CommandResult:
        ...


class SystemCommandRunner:
    """Execute only three fixed read-only query tools from absolute binaries."""

    def __init__(self, *, timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise ValueError("timeout_seconds must be an integer")
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        self.timeout_seconds = timeout_seconds
        self.executables = {
            tool: self._resolve_executable(tool)
            for tool in EXECUTABLE_CANDIDATES
        }

    @staticmethod
    def _resolve_executable(tool: str) -> Path | None:
        for candidate in EXECUTABLE_CANDIDATES[tool]:
            try:
                candidate_stat = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(candidate_stat.st_mode):
                continue
            if stat.S_ISREG(candidate_stat.st_mode) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def run(
        self,
        tool: str,
        arguments: Sequence[str],
        *,
        cwd: Path,
    ) -> CommandResult:
        if tool not in self.executables:
            _hold(
                "HOLD_COMMAND_NOT_ALLOWLISTED",
                "preflight attempted a command outside the fixed allowlist",
            )
        executable = self.executables[tool]
        if executable is None:
            return CommandResult(returncode=127, stdout=b"", stderr=b"")
        _validate_command_arguments(tool, arguments, cwd=cwd)
        try:
            completed = subprocess.run(
                [str(executable), *arguments],
                cwd=str(cwd),
                env=dict(SAFE_ENVIRONMENT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _hold(
                "HOLD_READ_ONLY_COMMAND_TIMEOUT",
                "a bounded read-only query timed out",
                tool=tool,
            )
        except OSError as exc:
            _hold(
                "HOLD_READ_ONLY_COMMAND_UNAVAILABLE",
                "a bounded read-only query could not be executed",
                tool=tool,
                errno=exc.errno,
            )
        if (
            len(completed.stdout) > MAX_COMMAND_OUTPUT_BYTES
            or len(completed.stderr) > MAX_COMMAND_OUTPUT_BYTES
        ):
            _hold(
                "HOLD_READ_ONLY_COMMAND_OUTPUT_BOUND",
                "a read-only query exceeded the fixed output bound",
                tool=tool,
            )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _validate_command_arguments(
    tool: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
) -> None:
    if not isinstance(arguments, (tuple, list)) or not all(
        isinstance(item, str) and "\x00" not in item for item in arguments
    ):
        _hold(
            "HOLD_COMMAND_ARGUMENT_SCHEMA",
            "read-only command arguments must be a finite text sequence",
            tool=tool,
        )
    args = tuple(arguments)
    if tool == "git":
        required_prefix = GIT_READ_ONLY_PREFIX + ("-C", str(cwd))
        if args[: len(required_prefix)] != required_prefix:
            _hold(
                "HOLD_GIT_COMMAND_NOT_READ_ONLY",
                "Git query is missing its fixed read-only prefix",
            )
        tail = args[len(required_prefix) :]
        allowed_tails = {
            ("rev-parse", "--show-toplevel"),
            ("rev-parse", "--absolute-git-dir"),
            ("rev-parse", "--verify", "HEAD"),
            ("symbolic-ref", "--short", "-q", "HEAD"),
            ("ls-files", "-z"),
            (
                "status",
                "--porcelain=v2",
                "--branch",
                "--untracked-files=no",
            ),
            ("worktree", "list", "--porcelain"),
        }
        if tail not in allowed_tails:
            _hold(
                "HOLD_GIT_COMMAND_NOT_READ_ONLY",
                "Git query is outside the exact read-only command set",
            )
    elif tool == "ps":
        if args != ("-eo", "pid=,ppid=,user=,stat=,etimes=,comm="):
            _hold(
                "HOLD_PROCESS_COMMAND_NOT_READ_ONLY",
                "process query differs from the exact metadata-only form",
            )
    elif tool == "nvidia-smi":
        allowed = {
            (
                "--query-gpu=index,uuid,name,memory.total,memory.used,"
                "memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ),
            (
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ),
        }
        if args not in allowed:
            _hold(
                "HOLD_GPU_COMMAND_NOT_QUERY_ONLY",
                "GPU command differs from the exact query-only forms",
            )
    else:
        _hold(
            "HOLD_COMMAND_NOT_ALLOWLISTED",
            "preflight attempted a command outside the fixed allowlist",
        )


def _command(
    runner: CommandRunner,
    tool: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    allow_unavailable: bool = False,
    allow_nonzero: bool = False,
) -> CommandResult:
    _validate_command_arguments(tool, arguments, cwd=cwd)
    result = runner.run(tool, arguments, cwd=cwd)
    if result.returncode == 127 and allow_unavailable:
        return result
    if result.returncode != 0 and not allow_nonzero:
        _hold(
            "HOLD_READ_ONLY_COMMAND_FAILED",
            "a required read-only query failed",
            tool=tool,
            returncode=result.returncode,
        )
    if len(result.stdout) > MAX_COMMAND_OUTPUT_BYTES:
        _hold(
            "HOLD_READ_ONLY_COMMAND_OUTPUT_BOUND",
            "a read-only query exceeded the fixed output bound",
            tool=tool,
        )
    return result


def _decode_strict(raw: bytes, *, context: str) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _hold(
            "HOLD_NON_UTF8_QUERY_OUTPUT",
            "read-only query output is not strict UTF-8",
            context=context,
        )


def _git_args(repo_root: Path, *tail: str) -> tuple[str, ...]:
    return GIT_READ_ONLY_PREFIX + ("-C", str(repo_root), *tail)


def _collect_git(repo_root: Path, runner: CommandRunner) -> dict[str, Any]:
    root_text = _decode_strict(
        _command(
            runner,
            "git",
            _git_args(repo_root, "rev-parse", "--show-toplevel"),
            cwd=repo_root,
        ).stdout,
        context="git_root",
    ).strip()
    _reject_protected_text(root_text, context="git_root")
    reported_root = _absolute_lexical_path(root_text, context="git_root")
    if reported_root != repo_root:
        _hold(
            "HOLD_GIT_ROOT_MISMATCH",
            "Git reported a different repository root",
        )

    git_dir_text = _decode_strict(
        _command(
            runner,
            "git",
            _git_args(repo_root, "rev-parse", "--absolute-git-dir"),
            cwd=repo_root,
        ).stdout,
        context="git_dir",
    ).strip()
    _reject_protected_text(git_dir_text, context="git_dir")
    git_dir = _absolute_lexical_path(git_dir_text, context="git_dir")

    head = _decode_strict(
        _command(
            runner,
            "git",
            _git_args(repo_root, "rev-parse", "--verify", "HEAD"),
            cwd=repo_root,
        ).stdout,
        context="git_head",
    ).strip()
    if not HEX_HEAD_RE.fullmatch(head):
        _hold("HOLD_INVALID_GIT_HEAD", "Git HEAD is not a canonical object id")

    branch_result = _command(
        runner,
        "git",
        _git_args(repo_root, "symbolic-ref", "--short", "-q", "HEAD"),
        cwd=repo_root,
        allow_nonzero=True,
    )
    if branch_result.returncode not in {0, 1}:
        _hold(
            "HOLD_GIT_BRANCH_QUERY_FAILED",
            "Git branch query returned an unexpected status",
            returncode=branch_result.returncode,
        )
    branch = None
    if branch_result.returncode == 0:
        branch = _decode_strict(branch_result.stdout, context="git_branch").strip()
        if not branch:
            _hold("HOLD_EMPTY_GIT_BRANCH", "Git branch query returned an empty name")
        _reject_protected_text(branch, context="git_branch")

    tracked_raw = _command(
        runner,
        "git",
        _git_args(repo_root, "ls-files", "-z"),
        cwd=repo_root,
    ).stdout
    tracked_entries = tracked_raw.split(b"\x00")
    if tracked_entries and tracked_entries[-1] == b"":
        tracked_entries.pop()
    for index, raw_entry in enumerate(tracked_entries):
        entry = _decode_strict(raw_entry, context="git_tracked_path")
        _reject_protected_text(entry, context="git_tracked_path", index=index)
        if Path(entry).is_absolute() or ".." in Path(entry).parts:
            _hold(
                "HOLD_INVALID_TRACKED_PATH",
                "Git index contains a non-relative or traversing path",
                index=index,
            )

    status_text = _decode_strict(
        _command(
            runner,
            "git",
            _git_args(
                repo_root,
                "status",
                "--porcelain=v2",
                "--branch",
                "--untracked-files=no",
            ),
            cwd=repo_root,
        ).stdout,
        context="git_status",
    )
    _reject_protected_text(status_text, context="git_status")
    status_lines = status_text.splitlines()
    if len(status_lines) > MAX_STATUS_LINES:
        _hold(
            "HOLD_GIT_STATUS_LINE_BOUND",
            "Git status exceeded the fixed line bound",
        )
    change_lines = [line for line in status_lines if not line.startswith("# ")]

    worktree_text = _decode_strict(
        _command(
            runner,
            "git",
            _git_args(repo_root, "worktree", "list", "--porcelain"),
            cwd=repo_root,
        ).stdout,
        context="git_worktrees",
    )
    _reject_protected_text(worktree_text, context="git_worktrees")
    worktrees = _parse_worktrees(worktree_text)
    return {
        "branch": branch,
        "detached": branch is None,
        "git_dir": str(git_dir),
        "head": head,
        "repo_root": str(repo_root),
        "status": {
            "change_count": len(change_lines),
            "clean_ignoring_untracked": not change_lines,
            "porcelain_v2_lines": status_lines,
            "untracked_files_scanned": False,
        },
        "tracked_path_count": len(tracked_entries),
        "worktrees": worktrees,
    }


def _parse_worktrees(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in [*text.splitlines(), ""]:
        if not line:
            if current:
                if "path" not in current or "head" not in current:
                    _hold(
                        "HOLD_GIT_WORKTREE_SCHEMA",
                        "Git worktree record is incomplete",
                    )
                records.append(current)
                current = {}
            continue
        key, separator, value = line.partition(" ")
        if key == "worktree" and separator:
            path = _absolute_lexical_path(value, context="git_worktree_path")
            current["path"] = str(path)
        elif key == "HEAD" and separator:
            if not HEX_HEAD_RE.fullmatch(value):
                _hold(
                    "HOLD_GIT_WORKTREE_SCHEMA",
                    "Git worktree HEAD is not a canonical object id",
                )
            current["head"] = value
        elif key == "branch" and separator:
            _reject_protected_text(value, context="git_worktree_branch")
            current["branch"] = value
        elif key in {"detached", "bare", "prunable"}:
            current[key] = True
            if value:
                current[f"{key}_reason_present"] = True
        elif key == "locked":
            current["locked"] = True
            current["locked_reason_present"] = bool(value)
        else:
            _hold(
                "HOLD_GIT_WORKTREE_SCHEMA",
                "Git worktree output contains an unexpected field",
            )
    return records


def _open_regular_readonly(
    path: Path,
    *,
    context: str,
    index: int,
) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    no_atime_flag = getattr(os, "O_NOATIME", 0)
    try:
        return os.open(path, flags | no_atime_flag)
    except OSError as exc:
        if no_atime_flag and exc.errno in {errno.EPERM, errno.EACCES, errno.EINVAL}:
            try:
                return os.open(path, flags)
            except OSError as fallback:
                exc = fallback
        _hold(
            "HOLD_FILE_OPEN_REJECTED",
            "an explicit file could not be opened with read-only no-follow flags",
            context=context,
            index=index,
            errno=exc.errno,
        )


def _hash_regular_file(
    path: Path,
    *,
    context: str,
    index: int,
    max_hash_bytes: int,
) -> dict[str, Any]:
    descriptor = _open_regular_readonly(path, context=context, index=index)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _hold(
                "HOLD_NONREGULAR_FILE_REJECTED",
                "explicit hash inputs must be regular files",
                context=context,
                index=index,
            )
        if before.st_nlink != 1:
            _hold(
                "HOLD_HARDLINK_ALIAS_REJECTED",
                "multi-link files are rejected because their other names are unbound",
                context=context,
                index=index,
                link_count=before.st_nlink,
            )
        if before.st_size > max_hash_bytes:
            _hold(
                "HOLD_FILE_EXCEEDS_EXPLICIT_HASH_BOUND",
                "explicit file exceeds the configured full-hash byte bound",
                context=context,
                index=index,
                size_bytes=before.st_size,
                max_hash_bytes=max_hash_bytes,
            )
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_hash_bytes:
                _hold(
                    "HOLD_FILE_EXCEEDS_EXPLICIT_HASH_BOUND",
                    "file grew beyond the configured full-hash byte bound",
                    context=context,
                    index=index,
                )
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        _hold(
            "HOLD_FILE_CHANGED_DURING_READ",
            "explicit file metadata changed during the exact read",
            context=context,
            index=index,
        )
    if bytes_read != before.st_size:
        _hold(
            "HOLD_FILE_SIZE_READ_MISMATCH",
            "exact read length differs from the stable file size",
            context=context,
            index=index,
        )
    return {
        "device": before.st_dev,
        "inode": before.st_ino,
        "link_count": before.st_nlink,
        "modified_time_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "size_bytes": before.st_size,
    }


def _collect_explicit_paths(
    paths: Sequence[Path],
    *,
    context: str,
    max_hash_bytes: int,
    allow_directory_metadata: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        path_stat = _validate_no_symlink_ancestors(
            path,
            context=context,
            index=index,
            allow_missing=True,
        )
        if path_stat is None:
            records.append(
                {
                    "entry_type": "MISSING",
                    "exists": False,
                    "index": index,
                    "path": str(path),
                    "sha256": None,
                    "size_bytes": None,
                }
            )
            continue
        if stat.S_ISDIR(path_stat.st_mode):
            if not allow_directory_metadata:
                _hold(
                    "HOLD_EXPECTED_REGULAR_FILE",
                    "authority contracts and ordinary data must be regular files",
                    context=context,
                    index=index,
                )
            records.append(
                {
                    "directory_traversed": False,
                    "entry_type": "DIRECTORY_METADATA_ONLY",
                    "exists": True,
                    "index": index,
                    "modified_time_ns": path_stat.st_mtime_ns,
                    "path": str(path),
                    "sha256": None,
                    "size_bytes": None,
                }
            )
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            _hold(
                "HOLD_NONREGULAR_FILE_REJECTED",
                "explicit preflight paths may not be devices, sockets, or FIFOs",
                context=context,
                index=index,
            )
        exact = _hash_regular_file(
            path,
            context=context,
            index=index,
            max_hash_bytes=max_hash_bytes,
        )
        records.append(
            {
                "entry_type": "REGULAR_FILE",
                "exists": True,
                "index": index,
                "path": str(path),
                **exact,
            }
        )
    return records


def _collect_jobs(repo_root: Path, runner: CommandRunner) -> dict[str, Any]:
    result = _command(
        runner,
        "ps",
        ("-eo", "pid=,ppid=,user=,stat=,etimes=,comm="),
        cwd=repo_root,
        allow_unavailable=True,
    )
    if result.returncode == 127:
        return {
            "records": [],
            "selection_policy": "EXECUTABLE_BASENAME_ONLY_NO_ARGUMENTS",
            "status": "PS_NOT_AVAILABLE",
            "truncated": False,
        }
    if result.returncode != 0:
        return {
            "records": [],
            "returncode": result.returncode,
            "selection_policy": "EXECUTABLE_BASENAME_ONLY_NO_ARGUMENTS",
            "status": "PS_QUERY_FAILED",
            "truncated": False,
        }
    text = _decode_strict(result.stdout, context="process_metadata")
    records: list[dict[str, Any]] = []
    matched_count = 0
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        pid_text, ppid_text, user, state, elapsed_text, executable = fields
        executable = os.path.basename(executable)
        if not JOB_EXECUTABLE_RE.fullmatch(executable):
            continue
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
            elapsed_seconds = int(elapsed_text)
        except ValueError:
            continue
        matched_count += 1
        if len(records) >= MAX_JOB_RECORDS:
            continue
        records.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "executable_basename": executable,
                "pid": pid,
                "ppid": ppid,
                "state": state,
                "user": user,
            }
        )
    return {
        "matched_count": matched_count,
        "records": records,
        "selection_policy": "EXECUTABLE_BASENAME_ONLY_NO_ARGUMENTS",
        "status": "QUERY_COMPLETE",
        "truncated": matched_count > len(records),
    }


def _csv_rows(raw: bytes, *, context: str) -> list[list[str]]:
    text = _decode_strict(raw, context=context)
    return [
        [field.strip() for field in row]
        for row in csv.reader(text.splitlines())
        if row
    ]


def _number_or_none(value: str, *, integer: bool) -> int | float | None:
    if value.casefold() in {"n/a", "na", "not supported", ""}:
        return None
    try:
        return int(value) if integer else float(value)
    except ValueError:
        return None


def _collect_gpu(repo_root: Path, runner: CommandRunner) -> dict[str, Any]:
    gpu_args = (
        "--query-gpu=index,uuid,name,memory.total,memory.used,"
        "memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    gpu_result = _command(
        runner,
        "nvidia-smi",
        gpu_args,
        cwd=repo_root,
        allow_unavailable=True,
        allow_nonzero=True,
    )
    if gpu_result.returncode == 127:
        return {
            "compute_processes": [],
            "devices": [],
            "query_only": True,
            "status": "NVIDIA_SMI_NOT_AVAILABLE",
        }
    if gpu_result.returncode != 0:
        return {
            "compute_processes": [],
            "devices": [],
            "query_only": True,
            "returncode": gpu_result.returncode,
            "status": "NVIDIA_SMI_QUERY_FAILED",
        }
    devices: list[dict[str, Any]] = []
    for row in _csv_rows(gpu_result.stdout, context="gpu_devices"):
        if len(row) != 8:
            _hold("HOLD_GPU_QUERY_SCHEMA", "GPU query returned an unexpected row")
        index, uuid, name, total, used, free, utilization, temperature = row
        _reject_protected_text(name, context="gpu_name")
        devices.append(
            {
                "index": _number_or_none(index, integer=True),
                "memory_free_mib": _number_or_none(free, integer=True),
                "memory_total_mib": _number_or_none(total, integer=True),
                "memory_used_mib": _number_or_none(used, integer=True),
                "name": name,
                "temperature_c": _number_or_none(temperature, integer=False),
                "utilization_percent": _number_or_none(utilization, integer=False),
                "uuid": uuid,
            }
        )

    process_args = (
        "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    )
    process_result = _command(
        runner,
        "nvidia-smi",
        process_args,
        cwd=repo_root,
        allow_nonzero=True,
    )
    processes: list[dict[str, Any]] = []
    process_status = "QUERY_COMPLETE"
    if process_result.returncode != 0:
        process_status = "COMPUTE_PROCESS_QUERY_FAILED"
    else:
        for row in _csv_rows(process_result.stdout, context="gpu_processes"):
            if len(row) != 4:
                _hold(
                    "HOLD_GPU_PROCESS_QUERY_SCHEMA",
                    "GPU process query returned an unexpected row",
                )
            pid, process_name, gpu_uuid, used_memory = row
            _reject_protected_text(process_name, context="gpu_process_name")
            processes.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": _number_or_none(pid, integer=True),
                    "process_basename": os.path.basename(process_name),
                    "used_memory_mib": _number_or_none(used_memory, integer=True),
                }
            )
    return {
        "compute_process_query_status": process_status,
        "compute_processes": processes,
        "devices": devices,
        "query_only": True,
        "status": "QUERY_COMPLETE",
    }


def _collect_disk(read_roots: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_devices: set[int] = set()
    for root in read_roots:
        root_stat = os.stat(root, follow_symlinks=False)
        if root_stat.st_dev in seen_devices:
            continue
        seen_devices.add(root_stat.st_dev)
        fs = os.statvfs(root)
        fragment_size = fs.f_frsize or fs.f_bsize
        records.append(
            {
                "available_bytes": fs.f_bavail * fragment_size,
                "device": root_stat.st_dev,
                "free_bytes": fs.f_bfree * fragment_size,
                "read_root": str(root),
                "total_bytes": fs.f_blocks * fragment_size,
            }
        )
    return records


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def collect_preflight(
    config: PreflightConfig,
    *,
    runner: CommandRunner,
    clock: Callable[[], str] = _utc_now,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Collect a bounded snapshot without creating or modifying filesystem state."""

    normalized = _normalize_config(config)
    started = clock()
    host = socket.gethostname() if hostname is None else hostname
    if not isinstance(host, str) or not host:
        _hold("HOLD_HOSTNAME_SCHEMA", "hostname must be non-empty text")
    git = _collect_git(normalized.repo_root, runner)
    authority_contracts = _collect_explicit_paths(
        normalized.authority_contract_paths,
        context="authority_contract",
        max_hash_bytes=normalized.max_hash_bytes,
        allow_directory_metadata=False,
    )
    ordinary_data = _collect_explicit_paths(
        normalized.ordinary_data_paths,
        context="ordinary_data",
        max_hash_bytes=normalized.max_hash_bytes,
        allow_directory_metadata=False,
    )
    artifacts = _collect_explicit_paths(
        normalized.artifact_paths,
        context="artifact",
        max_hash_bytes=normalized.max_hash_bytes,
        allow_directory_metadata=True,
    )
    jobs = _collect_jobs(normalized.repo_root, runner)
    gpu = _collect_gpu(normalized.repo_root, runner)
    disk = _collect_disk(normalized.read_roots)
    finished = clock()
    return {
        "artifacts": artifacts,
        "authority_contracts": authority_contracts,
        "boundaries": dict(BOUNDARIES),
        "disk": disk,
        "finished_utc": finished,
        "git": git,
        "gpu": gpu,
        "hostname": host,
        "jobs": jobs,
        "limitations": [
            "METADATA_SNAPSHOT_ONLY_NOT_EXECUTION_AUTHORIZATION",
            "NO_OS_MOUNT_CAPABILITY_ATTESTATION",
            "NO_BIND_MOUNT_CONTENT_ALIAS_PROOF",
            "NO_UNTRACKED_FILE_DISCOVERY",
            "NO_ARTIFACT_DIRECTORY_RECURSION",
            "NO_SCIENTIFIC_OR_PUBLICATION_CLAIM",
        ],
        "ordinary_data": ordinary_data,
        "read_roots": [str(root) for root in normalized.read_roots],
        "schema_version": SCHEMA_VERSION,
        "started_utc": started,
        "status": COMPLETE_STATUS,
    }


MUTATING_AUDIT_EVENTS = {
    "os.chmod",
    "os.chown",
    "os.link",
    "os.mkdir",
    "os.remove",
    "os.removexattr",
    "os.rename",
    "os.rmdir",
    "os.setxattr",
    "os.symlink",
    "os.truncate",
    "os.utime",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "tempfile.mkstemp",
    "tempfile.mkdtemp",
}


def _make_read_only_audit_hook(
    allowed_executables: set[str],
) -> Callable[[str, tuple[Any, ...]], None]:
    write_flags = (
        os.O_WRONLY
        | os.O_RDWR
        | os.O_APPEND
        | os.O_CREAT
        | os.O_EXCL
        | os.O_TRUNC
    )

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event in MUTATING_AUDIT_EVENTS:
            raise PermissionError(f"read-only preflight blocked audit event: {event}")
        if event == "open" and len(arguments) >= 3:
            mode = arguments[1]
            flags = arguments[2]
            if isinstance(mode, str) and any(char in mode for char in "wax+"):
                raise PermissionError("read-only preflight blocked writable file mode")
            if isinstance(flags, int) and flags & write_flags:
                raise PermissionError("read-only preflight blocked writable open flags")
        if event == "subprocess.Popen" and arguments:
            executable = os.fspath(arguments[0])
            if executable not in allowed_executables:
                raise PermissionError(
                    "read-only preflight blocked a non-allowlisted executable"
                )

    return audit


class FailClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PreflightError(
            "HOLD_CLI_ARGUMENTS",
            "command-line arguments do not match the exact preflight interface",
        )


def _parser() -> argparse.ArgumentParser:
    parser = FailClosedArgumentParser(
        description=(
            "Collect a canonical stdout-only R3X remote read-only preflight; "
            "this command never starts training or accesses protected data."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--read-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--authority-contract", action="append", type=Path, default=[]
    )
    parser.add_argument("--ordinary-data", action="append", type=Path, default=[])
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=DEFAULT_MAX_HASH_BYTES,
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    return parser


def _hold_payload(error: PreflightError) -> dict[str, Any]:
    return {
        "boundaries": dict(BOUNDARIES),
        "error": {
            "code": error.code,
            "details": error.details,
            "message": error.message,
        },
        "schema_version": SCHEMA_VERSION,
        "status": HOLD_STATUS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if not REMOTE_EXECUTION_ENABLED:
            _hold(
                REMOTE_EXECUTION_HOLD_CODE,
                "remote preflight v1 is disabled before all path, Git, process, GPU, "
                "and file access; a signed ordinary train/development manifest plus "
                "a capability-separated read-only root are required for a new version",
                remote_run_approval=False,
                sealed_zero_access_guarantee_established=False,
                write_state_zero_guarantee_established=False,
            )
        config = PreflightConfig(
            repo_root=arguments.repo_root,
            read_roots=tuple(arguments.read_root),
            authority_contract_paths=tuple(arguments.authority_contract),
            ordinary_data_paths=tuple(arguments.ordinary_data),
            artifact_paths=tuple(arguments.artifact),
            max_hash_bytes=arguments.max_hash_bytes,
        )
        # Reject unsafe path strings and path topology before resolving any
        # external executable or running a repository query.
        config = _normalize_config(config)
        runner = SystemCommandRunner(
            timeout_seconds=arguments.command_timeout_seconds
        )
        allowed_executables = {
            str(path)
            for path in runner.executables.values()
            if path is not None
        }
        sys.addaudithook(_make_read_only_audit_hook(allowed_executables))
        payload = collect_preflight(config, runner=runner)
        sys.stdout.buffer.write(canonical_json_bytes(payload))
        sys.stdout.buffer.flush()
        return 0
    except PreflightError as exc:
        sys.stdout.buffer.write(canonical_json_bytes(_hold_payload(exc)))
        sys.stdout.buffer.flush()
        return 2
    except (OSError, ValueError) as exc:
        error = PreflightError(
            "HOLD_UNEXPECTED_LOCAL_PREFLIGHT_ERROR",
            "preflight stopped on an unexpected local read/query error",
            details={"error_type": type(exc).__name__},
        )
        sys.stdout.buffer.write(canonical_json_bytes(_hold_payload(error)))
        sys.stdout.buffer.flush()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
