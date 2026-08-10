#!/usr/bin/env python3
"""Fail-closed, aggregate-only qualification audit for public GSE145046.

This formal entry point is deliberately narrow.  It accepts only the canonical
protocol embedded below, verifies that protocol, this script, and the focused
test are clean HEAD bytes in the declared repository, binds the exact public
manifest and thirty gzip payloads, and emits one aggregate-only JSON record.
It never materializes an endpoint, canonical intervention, or measured pool.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE145046"
PROTOCOL_ID = "ROUTE_A_V3_GSE145046_A2_QUALIFICATION_AUDIT_V1"
REPORT_TYPE = "GSE145046_A2_QUALIFICATION_AUDIT"
REPORT_SCHEMA_VERSION = "2.0.0"

# This is the formal trust root.  Tests may monkeypatch the in-memory value for
# an isolated synthetic repository; the CLI exposes no bypass.
CANONICAL_PROTOCOL_SHA256 = (
    "666c9ee86033a05a006171df963fa3d96b68430d9a9d4e817789e255b28b300d"
)
PROTOCOL_REPOSITORY_PATH = "configs/route_a_v3_gse145046_a2_audit.json"
SCRIPT_REPOSITORY_PATH = "scripts/route_a_v3/audit_gse145046_a2.py"
TEST_REPOSITORY_PATH = "tests/route_a_v3/test_audit_gse145046_a2.py"

FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (
    "gse246381",
    "restricted",
    "sealed_external",
    "access_log",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
COUNT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
VALID_10MER_RE = re.compile(r"^[ACGT]{10}$")
SUSPICIOUS_SEQUENCE_TOKEN_RE = re.compile(r"[ACGTNacgtn]{10,}")
BASE4 = {"A": 0, "C": 1, "G": 2, "T": 3}
KEY_SPACE_SIZE = 4**10
BITSET_SIZE = KEY_SPACE_SIZE // 8
MAX_DECOMPRESSED_LINE_BYTES = 4096
BYTE_POPCOUNT: tuple[int, ...] = tuple(bin(value).count("1") for value in range(256))

EXPECTED_SAMPLE_SPECS: tuple[tuple[str, str, str, str, int | None], ...] = (
    ("GSM4305122_1_read_count_Randomly_synthesized_oligos.txt.gz", "GSM4305122", "INPUT_LIBRARY", "RANDOMLY_SYNTHESIZED_OLIGOS", 1),
    ("GSM4546416_read_count_Randomly_synthesized_oligos_rep2.txt.gz", "GSM4546416", "INPUT_LIBRARY", "RANDOMLY_SYNTHESIZED_OLIGOS", 2),
    ("GSM4305123_2_read_count_In_vivo_Monosome_rep1.txt.gz", "GSM4305123", "IN_VIVO_RIBOSOME_FRACTION", "MONOSOME", 1),
    ("GSM4305124_3_read_count_In_vivo_Polysome_rep1.txt.gz", "GSM4305124", "IN_VIVO_RIBOSOME_FRACTION", "POLYSOME", 1),
    ("GSM4305125_4_read_count_In_vivo_Monosome_rep2.txt.gz", "GSM4305125", "IN_VIVO_RIBOSOME_FRACTION", "MONOSOME", 2),
    ("GSM4305126_5_read_count_In_vivo_Polysome_rep2.txt.gz", "GSM4305126", "IN_VIVO_RIBOSOME_FRACTION", "POLYSOME", 2),
    ("GSM4305127_6_read_count_In_vivo_Ribosome_free_rep1.txt.gz", "GSM4305127", "IN_VIVO_RIBOSOME_FREE", "RIBOSOME_FREE", 1),
    ("GSM4305128_7_read_count_In_vivo_Ribosome_free_rep2.txt.gz", "GSM4305128", "IN_VIVO_RIBOSOME_FREE", "RIBOSOME_FREE", 2),
    ("GSM4305129_8_read_count_Non_functional_cap_Ribosome_free.txt.gz", "GSM4305129", "NON_FUNCTIONAL_CAP_RIBOSOME_FRACTION", "RIBOSOME_FREE", None),
    ("GSM4305130_9_read_count_Non_functional_cap_Ribosome_bound.txt.gz", "GSM4305130", "NON_FUNCTIONAL_CAP_RIBOSOME_FRACTION", "RIBOSOME_BOUND", None),
    ("GSM4305131_10_read_count_In_vivo_Gating_25D_low_rep1.txt.gz", "GSM4305131", "IN_VIVO_FACS_GATE", "25D_LOW", 1),
    ("GSM4305132_11_read_count_In_vivo_Gating_25D_high_rep1.txt.gz", "GSM4305132", "IN_VIVO_FACS_GATE", "25D_HIGH", 1),
    ("GSM4305133_12_read_count_In_vivo_Gating_GFP_low_rep1.txt.gz", "GSM4305133", "IN_VIVO_FACS_GATE", "GFP_LOW", 1),
    ("GSM4305134_13_read_count_In_vivo_Gating_GFP_high_rep1.txt.gz", "GSM4305134", "IN_VIVO_FACS_GATE", "GFP_HIGH", 1),
    ("GSM4305135_14_read_count_In_vivo_Gating_25D_low_rep2.txt.gz", "GSM4305135", "IN_VIVO_FACS_GATE", "25D_LOW", 2),
    ("GSM4305136_15_read_count_In_vivo_Gating_25D_high_rep2.txt.gz", "GSM4305136", "IN_VIVO_FACS_GATE", "25D_HIGH", 2),
    ("GSM4305137_16_read_count_In_vivo_Gating_GFP_low_rep2.txt.gz", "GSM4305137", "IN_VIVO_FACS_GATE", "GFP_LOW", 2),
    ("GSM4305138_17_read_count_In_vivo_Gating_GFP_high_rep2.txt.gz", "GSM4305138", "IN_VIVO_FACS_GATE", "GFP_HIGH", 2),
    ("GSM4305139_18_read_count_In_vivo_Half_life_2h_rep1.txt.gz", "GSM4305139", "IN_VIVO_HALF_LIFE", "2H", 1),
    ("GSM4305140_19_read_count_In_vivo_Half_life_5h_rep1.txt.gz", "GSM4305140", "IN_VIVO_HALF_LIFE", "5H", 1),
    ("GSM4305141_20_read_count_In_vivo_Half_life_2h_rep2.txt.gz", "GSM4305141", "IN_VIVO_HALF_LIFE", "2H", 2),
    ("GSM4305142_21_read_count_In_vivo_Half_life_5h_rep2.txt.gz", "GSM4305142", "IN_VIVO_HALF_LIFE", "5H", 2),
    ("GSM4305143_22_read_count_In_vitro_Half_life_0min.txt.gz", "GSM4305143", "IN_VITRO_HALF_LIFE", "0MIN", None),
    ("GSM4305144_23_read_count_In_vitro_Half_life_10min.txt.gz", "GSM4305144", "IN_VITRO_HALF_LIFE", "10MIN", None),
    ("GSM4305145_24_read_count_In_vitro_Half_life_30min.txt.gz", "GSM4305145", "IN_VITRO_HALF_LIFE", "30MIN", None),
    ("GSM4305146_25_read_count_In_vitro_Half_life_60min.txt.gz", "GSM4305146", "IN_VITRO_HALF_LIFE", "60MIN", None),
    ("GSM4305147_26_read_count_Non_functional_cap_in_vivo_half_life_2h_rep1.txt.gz", "GSM4305147", "NON_FUNCTIONAL_CAP_IN_VIVO_HALF_LIFE", "2H", 1),
    ("GSM4305148_27_read_count_Non_functional_cap_in_vivo_half_life_5h_rep1.txt.gz", "GSM4305148", "NON_FUNCTIONAL_CAP_IN_VIVO_HALF_LIFE", "5H", 1),
    ("GSM4305149_28_read_count_Non_functional_cap_in_vivo_half_life_2h_rep2.txt.gz", "GSM4305149", "NON_FUNCTIONAL_CAP_IN_VIVO_HALF_LIFE", "2H", 2),
    ("GSM4305150_29_read_count_Non_functional_cap_in_vivo_half_life_5h_rep2.txt.gz", "GSM4305150", "NON_FUNCTIONAL_CAP_IN_VIVO_HALF_LIFE", "5H", 2),
)
EXPECTED_FILENAMES: tuple[str, ...] = tuple(item[0] for item in EXPECTED_SAMPLE_SPECS)
INPUT_FILENAMES = EXPECTED_FILENAMES[:2]

EXPECTED_BLOCKERS: tuple[str, ...] = (
    "FULL_REPORTER_SOURCE_ANCHOR_NOT_IDENTIFIABLE",
    "FACS_GATE_CONSTANTS_NOT_RECOVERED",
    "IN_VIVO_HALF_LIFE_BASELINE_AND_AGGREGATION_NOT_RECOVERED",
    "IN_VITRO_REPLICATE_AND_SE_NOT_IDENTIFIABLE",
    "LICENSE_AND_REDISTRIBUTION_NOT_BOUND",
    "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_CLOSED",
    "DENSE_SPLIT_AND_HAMMING_MOAT_NOT_FROZEN",
    "ABSOLUTE_OUTCOME_NOT_DIRECT_SOURCE_CANDIDATE_INTERVENTION",
    "TRUE_A2_NOT_QUALIFIED",
)


class A2AuditError(RuntimeError):
    """Raised when an audit precondition or closed-world invariant fails."""


class ScopeViolation(A2AuditError):
    """Raised before payload reads for a forbidden or escaping path."""


FileIdentity = tuple[int, int, int, int, int, int]
DirectoryCoreIdentity = tuple[int, int, int]


class DirectoryBinding:
    def __init__(
        self,
        *,
        path: Path,
        fd: int,
        full_identity: FileIdentity,
        core_identity: DirectoryCoreIdentity,
        label: str,
    ) -> None:
        self.path = path
        self.fd = fd
        self.full_identity = full_identity
        self.core_identity = core_identity
        self.label = label


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    text = os.fspath(path).casefold()
    matches = sorted(token for token in FORBIDDEN_PATH_TOKENS if token in text)
    if matches:
        raise ScopeViolation(
            f"{label} rejected before read; forbidden path token(s): "
            + ",".join(matches)
        )


def _absolute_without_resolving(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return Path(os.path.normpath(os.fspath(expanded)))
    return Path(os.path.abspath(os.fspath(expanded)))


def _file_identity(info: os.stat_result) -> FileIdentity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_core_identity(info: os.stat_result) -> DirectoryCoreIdentity:
    return (info.st_dev, info.st_ino, info.st_mode)


def _safe_basename(name: str, *, label: str) -> str:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ScopeViolation(f"{label} must be a safe basename")
    _reject_forbidden_path(name, label=label)
    return name


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _open_directory_no_symlinks(path: Path, *, label: str) -> DirectoryBinding:
    """Open every path component with openat + O_NOFOLLOW."""

    absolute = _absolute_without_resolving(path)
    _reject_forbidden_path(absolute, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            _safe_basename(component, label=label)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise A2AuditError(f"{label} must be a directory")
        path_info = os.stat(absolute, follow_symlinks=False)
        if _file_identity(path_info) != _file_identity(info):
            raise ScopeViolation(f"{label} path identity changed while opening")
        return DirectoryBinding(
            path=absolute,
            fd=descriptor,
            full_identity=_file_identity(info),
            core_identity=_directory_core_identity(info),
            label=label,
        )
    except Exception:
        os.close(descriptor)
        raise


def _assert_directory_binding(binding: DirectoryBinding, *, full: bool) -> None:
    current_fd = os.fstat(binding.fd)
    try:
        current_path = os.stat(binding.path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ScopeViolation(f"{binding.label} original path disappeared") from exc
    if not stat.S_ISDIR(current_path.st_mode):
        raise ScopeViolation(f"{binding.label} original path is no longer a directory")
    if full:
        if _file_identity(current_fd) != binding.full_identity:
            raise ScopeViolation(f"{binding.label} descriptor identity changed")
        if _file_identity(current_path) != binding.full_identity:
            raise ScopeViolation(f"{binding.label} original path identity changed")
    else:
        if _directory_core_identity(current_fd) != binding.core_identity:
            raise ScopeViolation(f"{binding.label} descriptor identity changed")
        if _directory_core_identity(current_path) != binding.core_identity:
            raise ScopeViolation(f"{binding.label} original path identity changed")


def _stat_regular_at(binding: DirectoryBinding, name: str, *, label: str) -> os.stat_result:
    _safe_basename(name, label=label)
    try:
        info = os.stat(name, dir_fd=binding.fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise A2AuditError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ScopeViolation(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise A2AuditError(f"{label} must be a regular file")
    return info


def _open_regular_at(
    binding: DirectoryBinding, name: str, *, label: str
) -> tuple[int, FileIdentity]:
    before = _stat_regular_at(binding, name, label=label)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(name, flags, dir_fd=binding.fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise A2AuditError(f"{label} must be a regular file")
        identity = _file_identity(opened)
        if identity != _file_identity(before):
            raise ScopeViolation(f"{label} identity changed while opening")
        after = _stat_regular_at(binding, name, label=label)
        if _file_identity(after) != identity:
            raise ScopeViolation(f"{label} leaf identity changed while opening")
        return descriptor, identity
    except Exception:
        os.close(descriptor)
        raise


def _assert_open_file_binding(
    descriptor: int,
    identity: FileIdentity,
    directory: DirectoryBinding,
    name: str,
    *,
    label: str,
) -> None:
    if _file_identity(os.fstat(descriptor)) != identity:
        raise ScopeViolation(f"{label} descriptor changed during audit phase")
    leaf = _stat_regular_at(directory, name, label=label)
    if _file_identity(leaf) != identity:
        raise ScopeViolation(f"{label} leaf identity changed during audit")


def _read_all_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def _read_regular_at(
    binding: DirectoryBinding,
    name: str,
    *,
    label: str,
) -> tuple[bytes, dict[str, Any], FileIdentity]:
    descriptor, identity = _open_regular_at(binding, name, label=label)
    try:
        _assert_open_file_binding(descriptor, identity, binding, name, label=label)
        raw = _read_all_fd(descriptor)
        _assert_open_file_binding(descriptor, identity, binding, name, label=label)
        if len(raw) != identity[3]:
            raise ScopeViolation(f"{label} byte count changed while reading")
        binding_record = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        return raw, binding_record, identity
    finally:
        os.close(descriptor)


def _read_path_bound_bytes(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, Any], FileIdentity]:
    absolute = _absolute_without_resolving(path)
    parent = _open_directory_no_symlinks(absolute.parent, label=f"{label} parent")
    try:
        result = _read_regular_at(parent, absolute.name, label=label)
        _assert_directory_binding(parent, full=True)
        return result
    finally:
        os.close(parent.fd)


def _assert_path_identity(path: Path, identity: FileIdentity, *, label: str) -> None:
    absolute = _absolute_without_resolving(path)
    parent = _open_directory_no_symlinks(absolute.parent, label=f"{label} parent")
    try:
        current = _stat_regular_at(parent, absolute.name, label=label)
        if _file_identity(current) != identity:
            raise ScopeViolation(f"{label} identity changed during audit")
    finally:
        os.close(parent.fd)


def _normalize_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise A2AuditError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise A2AuditError(f"{label} must be a non-empty safe identifier")
    if SUSPICIOUS_SEQUENCE_TOKEN_RE.search(value):
        raise A2AuditError(f"{label} must not contain a sequence-like token")
    return value


def _validate_run_metadata(
    *,
    run_id: Any,
    audit_execution_id: Any,
    code_commit: Any,
    recorded_at: Any,
    test_sha256: Any,
) -> dict[str, str]:
    run = _validate_identifier(run_id, label="run_id")
    execution = _validate_identifier(audit_execution_id, label="audit_execution_id")
    if run == execution:
        raise A2AuditError("audit_execution_id must differ from parent run_id")
    if not isinstance(code_commit, str) or COMMIT_RE.fullmatch(code_commit) is None:
        raise A2AuditError("code_commit must be a lowercase 40-hex commit")
    if not isinstance(recorded_at, str) or "T" not in recorded_at:
        raise A2AuditError("recorded_at must be an ISO timestamp with UTC or offset")
    parse_value = recorded_at[:-1] + "+00:00" if recorded_at.endswith("Z") else recorded_at
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise A2AuditError("recorded_at must be an ISO timestamp with UTC or offset") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise A2AuditError("recorded_at must include UTC or an explicit offset")
    return {
        "run_id": run,
        "audit_execution_id": execution,
        "code_commit": code_commit,
        "recorded_at": recorded_at,
        "test_sha256": _normalize_sha256(test_sha256, label="test SHA-256"),
    }


def _prepare_paths_before_read(
    *,
    contract_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    output_path: Path,
    repo_root: Path,
    test_path: Path,
    audit_execution_id: str,
) -> tuple[dict[str, Path], DirectoryBinding]:
    raw = {
        "contract": Path(contract_path),
        "protocol": Path(protocol_path),
        "manifest": Path(manifest_path),
        "output": Path(output_path),
        "repo_root": Path(repo_root),
        "test": Path(test_path),
    }
    for label, path in raw.items():
        _reject_forbidden_path(path, label=f"{label} path")
    paths = {key: _absolute_without_resolving(value) for key, value in raw.items()}
    for label, path in paths.items():
        _reject_forbidden_path(path, label=f"{label} path")

    expected_output_name = f"{audit_execution_id}.json"
    if paths["output"].name != expected_output_name:
        raise A2AuditError(
            f"formal output basename must be {expected_output_name}"
        )
    if len({paths["contract"], paths["protocol"], paths["manifest"], paths["output"]}) != 4:
        raise ScopeViolation("output and contract/protocol/manifest paths must be distinct")
    if _is_relative_to(paths["output"], paths["manifest"].parent):
        raise ScopeViolation("output path must not be inside the P0 input directory")

    output_binding = _open_directory_no_symlinks(
        paths["output"].parent, label="output parent"
    )
    try:
        try:
            os.stat(paths["output"].name, dir_fd=output_binding.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise A2AuditError("refusing to overwrite existing output")

        # Metadata-only checks follow the output no-overwrite check; no input
        # content is read here.
        for key in ("contract", "protocol", "manifest", "test"):
            absolute = paths[key]
            parent = _open_directory_no_symlinks(
                absolute.parent, label=f"{key} parent"
            )
            try:
                _stat_regular_at(parent, absolute.name, label=key)
            finally:
                os.close(parent.fd)
        repo_probe = _open_directory_no_symlinks(paths["repo_root"], label="repo root")
        os.close(repo_probe.fd)
        return paths, output_binding
    except Exception:
        os.close(output_binding.fd)
        raise


def _run_git(repo_root: Path, arguments: Sequence[str], *, text: bool) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo_root), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        shell=False,
    )
    if completed.returncode != 0:
        raise A2AuditError("Git repository binding command failed")
    return completed.stdout


def _verify_repo_binding(
    *,
    repo_root: Path,
    protocol_path: Path,
    test_path: Path,
    expected_protocol_sha256: str,
    code_commit: str,
    test_sha256: str,
) -> tuple[bytes, dict[str, Any], list[tuple[Path, FileIdentity, str]], DirectoryBinding]:
    repo_binding = _open_directory_no_symlinks(repo_root, label="repo root")
    try:
        expected_paths = {
            "protocol": (protocol_path, PROTOCOL_REPOSITORY_PATH),
            "script": (_absolute_without_resolving(Path(__file__)), SCRIPT_REPOSITORY_PATH),
            "test": (test_path, TEST_REPOSITORY_PATH),
        }
        for label, (path, expected_relative) in expected_paths.items():
            if not _is_relative_to(path, repo_root):
                raise ScopeViolation(f"{label} path must be inside repo root")
            relative = path.relative_to(repo_root).as_posix()
            if relative != expected_relative:
                raise ScopeViolation(f"{label} repository-relative path mismatch")

        top = str(_run_git(repo_root, ["rev-parse", "--show-toplevel"], text=True)).strip()
        if _absolute_without_resolving(Path(top)) != repo_root:
            raise ScopeViolation("declared repo root is not Git top level")
        head = str(_run_git(repo_root, ["rev-parse", "HEAD"], text=True)).strip()
        if head != code_commit:
            raise A2AuditError("Git HEAD does not match code_commit")
        status_output = str(
            _run_git(
                repo_root,
                ["status", "--porcelain=v1", "--untracked-files=all"],
                text=True,
            )
        )
        if status_output:
            raise A2AuditError("Git worktree must be clean")

        records: dict[str, dict[str, Any]] = {}
        raw_by_label: dict[str, bytes] = {}
        tracked_state: list[tuple[Path, FileIdentity, str]] = []
        for label, (path, relative) in expected_paths.items():
            raw, binding, identity = _read_path_bound_bytes(path, label=label)
            head_blob = _run_git(repo_root, ["cat-file", "blob", f"HEAD:{relative}"], text=False)
            if not isinstance(head_blob, bytes) or raw != head_blob:
                raise A2AuditError(f"{label} bytes do not match Git HEAD")
            records[label] = {
                "repository_relative_path": relative,
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
                "head_blob_match": True,
            }
            raw_by_label[label] = raw
            tracked_state.append((path, identity, label))

        actual_protocol_sha = records["protocol"]["sha256"]
        caller_protocol_sha = _normalize_sha256(
            expected_protocol_sha256, label="caller protocol SHA-256"
        )
        if not (
            caller_protocol_sha
            == CANONICAL_PROTOCOL_SHA256
            == actual_protocol_sha
        ):
            raise A2AuditError(
                "protocol SHA-256 must equal caller declaration, hardcoded canonical trust root, and actual bytes"
            )
        if records["test"]["sha256"] != test_sha256:
            raise A2AuditError("actual focused test SHA-256 does not match test_sha256")
        _assert_directory_binding(repo_binding, full=True)
        code_binding = {
            "code_commit": code_commit,
            "git_head": head,
            "git_worktree_clean": True,
            "verification": "VERIFIED_HEAD_CLEAN_AND_EXACT_BYTES",
            "protocol": records["protocol"],
            "script": records["script"],
            "test": records["test"],
        }
        return raw_by_label["protocol"], code_binding, tracked_state, repo_binding
    except Exception:
        os.close(repo_binding.fd)
        raise


def _assert_repo_binding_stable(
    repo_binding: DirectoryBinding,
    tracked_state: Sequence[tuple[Path, FileIdentity, str]],
    *,
    code_commit: str,
) -> None:
    _assert_directory_binding(repo_binding, full=True)
    for path, identity, label in tracked_state:
        _assert_path_identity(path, identity, label=label)
    head = str(_run_git(repo_binding.path, ["rev-parse", "HEAD"], text=True)).strip()
    if head != code_commit:
        raise A2AuditError("Git HEAD changed during audit")
    status_output = str(
        _run_git(
            repo_binding.path,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        )
    )
    if status_output:
        raise A2AuditError("Git worktree changed during audit")
    _assert_directory_binding(repo_binding, full=True)


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise A2AuditError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise A2AuditError(f"{label} root must be an object")
    return document


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise A2AuditError(f"{label} exact key set mismatch")
    return value


def _json_exact_equal(value: Any, expected: Any) -> bool:
    """JSON equality with type identity (so true != 1 and 1.0 != 1)."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _json_exact_equal(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(value) == len(expected) and all(
            _json_exact_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(value, expected)
        )
    return bool(value == expected)


def _require_exact_value(value: Any, expected: Any, *, label: str) -> None:
    if not _json_exact_equal(value, expected):
        raise A2AuditError(f"{label} mismatch")


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise A2AuditError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    number = _nonnegative_integer(value, label=label)
    if number == 0:
        raise A2AuditError(f"{label} must be positive")
    return number


def _nonnegative_finite(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise A2AuditError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise A2AuditError(f"{label} must be a finite nonnegative number")
    return number


def _load_and_validate_protocol(raw: bytes) -> dict[str, Any]:
    document = _decode_json_object(raw, label="protocol")
    _require_exact_keys(
        document,
        {
            "contract_id", "schema_version", "protocol_id", "protocol_status",
            "dataset_id", "authority", "scope", "science_authority", "inputs",
            "source_anchor", "endpoint_semantics", "qualification_boundary",
            "unresolved_blockers", "aggregate_preflight_evidence",
            "execution_binding_policy", "output_policy",
            "model_results_may_change_this_protocol",
        },
        label="protocol",
    )
    static_top = {
        "contract_id": CONTRACT_ID,
        "schema_version": "3.0.0",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "FROZEN_AFTER_AGGREGATE_PREFLIGHT_BEFORE_FORMAL_VERSIONED_AUDIT_AND_MODEL_RESULTS",
        "dataset_id": DATASET_ID,
        "model_results_may_change_this_protocol": False,
    }
    for key, expected in static_top.items():
        _require_exact_value(document[key], expected, label=f"protocol.{key}")

    authority = _require_exact_keys(
        document["authority"], {"contract_sha256", "base_commit"}, label="protocol.authority"
    )
    _normalize_sha256(authority["contract_sha256"], label="protocol contract SHA-256")
    _require_exact_value(
        authority["base_commit"], "fd722d5fa3c2538fce742b8942b1fb48e782760b",
        label="protocol.authority.base_commit",
    )
    _require_exact_value(
        document["scope"],
        {
            "ordinary_public_data_only": True,
            "forbidden_path_tokens": list(FORBIDDEN_PATH_TOKENS),
            "training_allowed": False,
            "model_selection_allowed": False,
            "qualification_allowed": False,
            "canonical_intervention_records_allowed": False,
            "measured_candidate_pools_allowed": False,
            "network_access_allowed": False,
        },
        label="protocol.scope",
    )
    _require_exact_value(
        document["science_authority"],
        {
            "version_of_record": {
                "doi": "10.1038/s41594-020-0465-x",
                "url": "https://doi.org/10.1038/s41594-020-0465-x",
            },
            "preprint": {
                "doi": "10.1101/2020.03.13.990887",
                "url": "https://doi.org/10.1101/2020.03.13.990887",
            },
            "geo": {
                "accession": DATASET_ID,
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145046",
            },
            "supplementary_table_1": {
                "url": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41594-020-0465-x/MediaObjects/41594_2020_465_MOESM3_ESM.xlsx",
                "bytes": 14024,
                "sha256": "4e33a6e351c3096dd708296807b6def84a347dba891e0401207ce4b0912de9ba",
                "local_p0_input": False,
                "authority_scope": "PRIMER_LEVEL_ONLY",
            },
            "geo_family_soft": {
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145046/soft/GSE145046_family.soft.gz",
                "bytes": 5308,
                "sha256": "f0e158e6899cfa7fb8731a4e290c7c08c64f13fbde79e9ac4eab1494d601ba75",
                "local_p0_input": False,
                "authority_scope": "PUBLIC_METADATA_ONLY",
            },
        },
        label="protocol.science_authority",
    )

    inputs = _require_exact_keys(
        document["inputs"], {"p0_manifest", "row_schema", "rpm_definition", "samples"},
        label="protocol.inputs",
    )
    manifest_spec = _require_exact_keys(
        inputs["p0_manifest"],
        {"filename", "sha256", "required_top_level_keys", "required_file_entry_keys", "file_count"},
        label="protocol.inputs.p0_manifest",
    )
    _require_exact_value(manifest_spec["filename"], "manifest.json", label="protocol manifest filename")
    _normalize_sha256(manifest_spec["sha256"], label="protocol manifest SHA-256")
    _require_exact_value(
        manifest_spec["required_top_level_keys"],
        ["accession", "files", "provider", "retrieved_at_utc", "skipped", "source_url"],
        label="protocol manifest top-level keys",
    )
    _require_exact_value(
        manifest_spec["required_file_entry_keys"],
        ["bytes", "downloaded", "expected_bytes", "name", "sha256", "url"],
        label="protocol manifest entry keys",
    )
    _require_exact_value(manifest_spec["file_count"], 30, label="protocol manifest file count")
    _require_exact_value(
        inputs["row_schema"],
        {
            "compression": "GZIP", "encoding": "UTF-8", "delimiter": "TAB",
            "header_present": False, "exact_column_count": 3,
            "columns": [
                {"position": 1, "name": "random_n10", "valid_pattern": "^[ACGT]{10}$"},
                {"position": 2, "name": "count", "constraint": "NONNEGATIVE_INTEGER"},
                {"position": 3, "name": "rpm", "constraint": "FINITE_NONNEGATIVE_NUMBER"},
            ],
        },
        label="protocol.inputs.row_schema",
    )
    rpm_definition = _require_exact_keys(
        inputs["rpm_definition"],
        {"formula", "source", "absolute_tolerance", "relative_tolerance", "mismatch_is_reported_not_imputed"},
        label="protocol.inputs.rpm_definition",
    )
    _require_exact_value(rpm_definition["formula"], "count / total_reads_in_file * 1000000", label="protocol RPM formula")
    _require_exact_value(rpm_definition["source"], "GEO_SAMPLE_METADATA", label="protocol RPM source")
    _require_exact_value(rpm_definition["mismatch_is_reported_not_imputed"], True, label="protocol RPM mismatch policy")
    _require_exact_value(
        rpm_definition["absolute_tolerance"],
        1e-6,
        label="protocol RPM absolute tolerance",
    )
    _require_exact_value(
        rpm_definition["relative_tolerance"],
        1e-9,
        label="protocol RPM relative tolerance",
    )

    samples = inputs["samples"]
    if not isinstance(samples, list) or len(samples) != len(EXPECTED_SAMPLE_SPECS):
        raise A2AuditError("protocol must declare exactly 30 samples")
    for order, (sample, expected) in enumerate(zip(samples, EXPECTED_SAMPLE_SPECS), start=1):
        entry = _require_exact_keys(
            sample,
            {"order", "gsm_accession", "filename", "sample_group", "condition", "replicate"},
            label=f"protocol sample {order}",
        )
        observed_tuple = (
            entry["filename"], entry["gsm_accession"], entry["sample_group"],
            entry["condition"], entry["replicate"],
        )
        _require_exact_value(
            entry["order"], order, label="protocol exact sample order"
        )
        _require_exact_value(
            observed_tuple,
            expected,
            label="protocol exact 30-sample role tuple universe",
        )

    _require_exact_value(
        document["source_anchor"],
        {
            "n10_locus_status": "CLOSED_AT_PRIMER_LEVEL",
            "full_reporter_anchor_status": "NOT_CLOSED",
            "canonical_source_hash": None,
            "interpretation": "PUBLIC_PRIMER_COORDINATES_CLOSE_THE_RANDOM_N10_INSERTION_LOCUS_BUT_NOT_THE_COMPLETE_REPORTER_OR_SOURCE_SEQUENCE",
        },
        label="protocol.source_anchor",
    )
    _require_exact_value(
        document["endpoint_semantics"],
        {
            "raw_columns": {
                "column_2": "COUNT",
                "column_3": "RPM_COUNT_DIVIDED_BY_TOTAL_READS_TIMES_ONE_MILLION",
                "status": "POSITIONAL_SEMANTICS_CLOSED_BY_GEO",
            },
            "monosome_polysome": {
                "paper_formula": "log2(RPM_MONOSOME / RPM_POLYSOME)",
                "aggregation_scope": "REPLICATE_LOCAL",
                "execution_status": "KNOWN_FORMULA_NOT_EXECUTED",
                "zero_and_pseudocount_rule": "UNKNOWN_NOT_ASSERTED",
            },
            "functional_cap_ribosome_free_bound": {
                "bound_construction_status": "NOT_RECOVERED",
                "endpoint_status": "BLOCKED_NOT_MATERIALIZED",
            },
            "non_functional_cap_apppg_ribosome_free_bound": {
                "available_pair_count": 1,
                "paper_use": "PRIMARILY_AGGREGATED_MOTIF_ANALYSIS",
                "per_variant_endpoint_status": "BLOCKED_NOT_MATERIALIZED",
            },
            "facs": {
                "weighted_gate_mean_constants": ["fH", "fNEG"],
                "constants_status": "NOT_RECOVERED",
                "endpoint_status": "BLOCKED_NOT_MATERIALIZED",
            },
            "in_vivo_half_life": {
                "baseline_status": "NOT_RECOVERED",
                "replicate_aggregation_status": "NOT_RECOVERED",
                "endpoint_status": "BLOCKED_NOT_MATERIALIZED",
            },
            "in_vitro_half_life": {
                "public_method_status": "DESCRIBED",
                "replicate_and_standard_error_status": "NOT_IDENTIFIABLE",
                "endpoint_status": "BLOCKED_NOT_MATERIALIZED",
            },
            "any_endpoint_materialized_by_this_audit": False,
            "absent_key_semantics": "MISSING_NOT_ZERO",
        },
        label="protocol.endpoint_semantics",
    )
    _require_exact_value(
        document["qualification_boundary"],
        {
            "classification": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
            "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
            "qualified": False,
            "training_allowed": False,
            "model_selection_allowed": False,
            "canonical_intervention_records_materialized": False,
            "measured_candidate_pools_materialized": False,
            "endpoint_values_materialized": False,
            "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
            "license": {"status": "UNKNOWN_BLOCKED", "redistribution_allowed": False},
            "foundation_exposure": {
                "checkpoint_specific_status": "UNKNOWN_NOT_ASSERTED",
                "project_engineering_consumption": "CONFIRMED",
                "independent_holdout_eligibility": False,
            },
        },
        label="protocol.qualification_boundary",
    )
    _require_exact_value(document["unresolved_blockers"], list(EXPECTED_BLOCKERS), label="protocol blockers")

    preflight = _require_exact_keys(
        document["aggregate_preflight_evidence"],
        {"evidence_status", "formal_versioned_audit_status", "constitutes_formal_acceptance", "formal_reconciliation_required", "observed_aggregates"},
        label="protocol.aggregate_preflight_evidence",
    )
    _require_exact_value(preflight["evidence_status"], "READ_ONLY_AGGREGATE_PREFLIGHT_COMPLETED", label="protocol preflight evidence status")
    _require_exact_value(preflight["formal_versioned_audit_status"], "NOT_RUN", label="protocol formal audit status")
    _require_exact_value(preflight["constitutes_formal_acceptance"], False, label="protocol preflight acceptance")
    _require_exact_value(preflight["formal_reconciliation_required"], True, label="protocol reconciliation requirement")
    aggregates = _require_exact_keys(
        preflight["observed_aggregates"],
        {"total_rows", "valid_key_rows", "invalid_key_rows", "input_union", "input_intersection", "all_30_union", "all_30_intersection"},
        label="protocol preflight aggregates",
    )
    for key, value in aggregates.items():
        _nonnegative_integer(value, label=f"protocol preflight {key}")
    if aggregates["total_rows"] != aggregates["valid_key_rows"] + aggregates["invalid_key_rows"]:
        raise A2AuditError("protocol preflight row counts do not reconcile")
    if not (aggregates["input_intersection"] <= aggregates["input_union"] <= KEY_SPACE_SIZE):
        raise A2AuditError("protocol preflight input set counts are impossible")
    if not (aggregates["all_30_intersection"] <= aggregates["all_30_union"] <= KEY_SPACE_SIZE):
        raise A2AuditError("protocol preflight all-file set counts are impossible")
    _require_exact_value(
        document["execution_binding_policy"],
        {
            "repo_root_required": True,
            "git_head_must_equal_code_commit": True,
            "git_worktree_must_be_clean": True,
            "protocol_script_test_must_match_head_bytes": True,
            "test_sha256_must_match_actual_bytes": True,
            "canonical_protocol_sha256_hardcoded_in_auditor": True,
        },
        label="protocol.execution_binding_policy",
    )
    _require_exact_value(
        document["output_policy"],
        {
            "aggregate_only": True,
            "raw_sequence_output_allowed": False,
            "raw_row_output_allowed": False,
            "raw_label_output_allowed": False,
            "determinism": "DETERMINISTIC_GIVEN_FIXED_EXPLICIT_PARAMETERS",
            "single_json_file": True,
            "hidden_sibling_staging": True,
            "fsync_before_publish": True,
            "no_replace_publish": True,
            "formal_output_name": "AUDIT_EXECUTION_ID_DOT_JSON",
            "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST_OR_LEDGER",
        },
        label="protocol.output_policy",
    )
    return document


def _load_and_validate_manifest(
    raw: bytes, *, protocol: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    manifest = _decode_json_object(raw, label="P0 manifest")
    required_top = set(protocol["inputs"]["p0_manifest"]["required_top_level_keys"])
    _require_exact_keys(manifest, required_top, label="P0 manifest")
    if manifest["accession"] != DATASET_ID:
        raise A2AuditError(f"P0 manifest accession must be {DATASET_ID}")
    for key in ("provider", "retrieved_at_utc", "source_url"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise A2AuditError(f"P0 manifest {key} must be non-empty")
    if not isinstance(manifest["skipped"], (list, bool)):
        raise A2AuditError("P0 manifest skipped must be a list or boolean")
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != 30:
        raise A2AuditError("P0 manifest must contain exactly 30 file entries")
    entry_keys = set(protocol["inputs"]["p0_manifest"]["required_file_entry_keys"])
    by_name: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        item = _require_exact_keys(entry, entry_keys, label="P0 manifest file entry")
        name = _safe_basename(item["name"], label="P0 manifest filename")
        if name in by_name:
            raise A2AuditError("P0 manifest contains a duplicate filename")
        if item["downloaded"] is not True:
            raise A2AuditError("all P0 manifest entries must be downloaded")
        byte_count = _positive_integer(item["bytes"], label=f"manifest bytes for {name}")
        expected_bytes = _positive_integer(item["expected_bytes"], label=f"manifest expected bytes for {name}")
        if byte_count != expected_bytes:
            raise A2AuditError(f"manifest byte declarations disagree for {name}")
        _normalize_sha256(item["sha256"], label=f"manifest SHA-256 for {name}")
        if not isinstance(item["url"], str) or not item["url"]:
            raise A2AuditError(f"manifest URL is missing for {name}")
        by_name[name] = item
    if set(by_name) != set(EXPECTED_FILENAMES):
        raise A2AuditError("P0 manifest exact sample universe mismatch")
    return by_name


def _inventory_exact_payloads(data: DirectoryBinding, *, manifest_filename: str) -> None:
    _assert_directory_binding(data, full=True)
    observed = set(os.listdir(data.fd))
    for name in observed:
        _safe_basename(name, label="P0 directory entry")
    expected = set(EXPECTED_FILENAMES) | {manifest_filename}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise A2AuditError(
            f"P0 directory exact sample universe mismatch; missing={missing}; unexpected={unexpected}"
        )
    for name in sorted(observed):
        _stat_regular_at(data, name, label="P0 directory entry")
    _assert_directory_binding(data, full=True)


def _parse_count_field(text: str, *, filename: str, line_number: int) -> int:
    if COUNT_RE.fullmatch(text) is None:
        raise A2AuditError(f"strict schema violation in {filename} line {line_number}: invalid count")
    return int(text, 10)


def _parse_rpm_field(text: str, *, filename: str, line_number: int) -> float:
    if not text or text != text.strip():
        raise A2AuditError(f"strict schema violation in {filename} line {line_number}: invalid RPM")
    try:
        value = float(text)
    except ValueError as exc:
        raise A2AuditError(f"strict schema violation in {filename} line {line_number}: invalid RPM") from exc
    if not math.isfinite(value) or value < 0:
        raise A2AuditError(f"strict schema violation in {filename} line {line_number}: invalid RPM")
    return value


def _iter_strict_rows_fd(descriptor: int, filename: str) -> Iterator[tuple[str, int, float]]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    raw_handle = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
    try:
        with gzip.GzipFile(fileobj=raw_handle, mode="rb") as handle:
            line_number = 0
            while True:
                raw_line = handle.readline(MAX_DECOMPRESSED_LINE_BYTES + 1)
                if not raw_line:
                    break
                line_number += 1
                if len(raw_line) > MAX_DECOMPRESSED_LINE_BYTES:
                    raise A2AuditError(
                        f"strict schema violation in {filename} line {line_number}: row too long"
                    )
                if raw_line.endswith(b"\n"):
                    raw_line = raw_line[:-1]
                    if raw_line.endswith(b"\r"):
                        raw_line = raw_line[:-1]
                if b"\r" in raw_line or b"\n" in raw_line:
                    raise A2AuditError(
                        f"strict schema violation in {filename} line {line_number}: newline"
                    )
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise A2AuditError(
                        f"strict schema violation in {filename} line {line_number}: UTF-8"
                    ) from exc
                fields = line.split("\t")
                if len(fields) != 3:
                    raise A2AuditError(
                        f"strict schema violation in {filename} line {line_number}: expected 3 columns"
                    )
                yield (
                    fields[0],
                    _parse_count_field(fields[1], filename=filename, line_number=line_number),
                    _parse_rpm_field(fields[2], filename=filename, line_number=line_number),
                )
    except A2AuditError:
        raise
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as exc:
        raise A2AuditError(f"gzip integrity failed for {filename}") from exc
    finally:
        raw_handle.close()


def _encode_10mer(sequence: str) -> tuple[int | None, str | None]:
    if len(sequence) != 10:
        return None, "INVALID_LENGTH"
    if VALID_10MER_RE.fullmatch(sequence) is None:
        return None, "INVALID_ALPHABET"
    value = 0
    for base in sequence:
        value = (value << 2) | BASE4[base]
    return value, None


def _bit_is_set(bitset: bytearray, value: int) -> bool:
    return bool(bitset[value >> 3] & (1 << (value & 7)))


def _set_bit(bitset: bytearray, value: int) -> None:
    bitset[value >> 3] |= 1 << (value & 7)


def _bit_count(bitset: bytes | bytearray) -> int:
    return sum(BYTE_POPCOUNT[byte] for byte in bitset)


def _bit_union(bitsets: Sequence[bytearray]) -> bytearray:
    result = bytearray(BITSET_SIZE)
    for bitset in bitsets:
        for index, value in enumerate(bitset):
            result[index] |= value
    return result


def _bit_intersection(bitsets: Sequence[bytearray]) -> bytearray:
    if not bitsets:
        return bytearray(BITSET_SIZE)
    result = bytearray(bitsets[0])
    for bitset in bitsets[1:]:
        for index, value in enumerate(bitset):
            result[index] &= value
    return result


def _intersection_count(left: bytearray, right: bytearray) -> int:
    return sum(BYTE_POPCOUNT[a & b] for a, b in zip(left, right))


def _left_only_count(left: bytearray, right: bytearray) -> int:
    return sum(BYTE_POPCOUNT[a & (~b & 0xFF)] for a, b in zip(left, right))


class _CompensatedSum:
    def __init__(self) -> None:
        self.total = 0.0
        self.compensation = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.compensation
        updated = self.total + adjusted
        self.compensation = (updated - self.total) - adjusted
        self.total = updated


def _stable_number(value: int | float) -> int | float:
    if isinstance(value, int):
        return value
    if value == 0:
        return 0
    if value.is_integer() and abs(value) <= 2**53:
        return int(value)
    return float(format(value, ".15g"))


def _scan_presence_fd(descriptor: int, filename: str) -> dict[str, Any]:
    bitset = bytearray(BITSET_SIZE)
    invalid_reasons: Counter[str] = Counter()
    rows = valid_rows = zero_count_rows = zero_rpm_rows = 0
    count_sum = 0
    rpm_sum = _CompensatedSum()
    for line_number, (sequence, count, rpm) in enumerate(
        _iter_strict_rows_fd(descriptor, filename), start=1
    ):
        rows += 1
        count_sum += count
        rpm_sum.add(rpm)
        zero_count_rows += int(count == 0)
        zero_rpm_rows += int(rpm == 0)
        encoded, reason = _encode_10mer(sequence)
        if encoded is None:
            invalid_reasons[str(reason)] += 1
            continue
        valid_rows += 1
        if _bit_is_set(bitset, encoded):
            raise A2AuditError(
                f"duplicate valid 10-mer key in {filename} line {line_number}"
            )
        _set_bit(bitset, encoded)
    if rows == 0:
        raise A2AuditError(f"strict schema violation in {filename}: empty file")
    return {
        "bitset": bitset,
        "rows": rows,
        "valid_key_rows": valid_rows,
        "invalid_key_rows": rows - valid_rows,
        "invalid_key_reason_counts": dict(sorted(invalid_reasons.items())),
        "distinct_valid_keys": _bit_count(bitset),
        "duplicate_valid_key_rows": 0,
        "zero_count_rows": zero_count_rows,
        "zero_rpm_rows": zero_rpm_rows,
        "count_sum": count_sum,
        "rpm_sum": rpm_sum.total,
    }


def _scan_rpm_consistency_fd(
    descriptor: int,
    filename: str,
    *,
    total_reads: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    rows_checked = rows_undefined = mismatch_rows = nonzero_undefined = 0
    max_absolute_error = 0.0
    max_relative_error = 0.0
    infinite_relative_error = False
    for _sequence, count, rpm in _iter_strict_rows_fd(descriptor, filename):
        if total_reads == 0:
            rows_undefined += 1
            nonzero_undefined += int(rpm != 0)
            continue
        rows_checked += 1
        expected = count / total_reads * 1_000_000.0
        absolute_error = abs(rpm - expected)
        relative_error = absolute_error / abs(expected) if expected else (0.0 if rpm == 0 else math.inf)
        max_absolute_error = max(max_absolute_error, absolute_error)
        if math.isinf(relative_error):
            infinite_relative_error = True
        else:
            max_relative_error = max(max_relative_error, relative_error)
        if not math.isclose(rpm, expected, rel_tol=relative_tolerance, abs_tol=absolute_tolerance):
            mismatch_rows += 1
    if total_reads == 0:
        status_value = "UNDEFINED_ZERO_DENOMINATOR"
        max_abs: int | float | None = None
        max_rel: int | float | None = None
    else:
        status_value = "PASS" if mismatch_rows == 0 else "FAIL_MISMATCH"
        max_abs = _stable_number(max_absolute_error)
        max_rel = None if infinite_relative_error else _stable_number(max_relative_error)
    return {
        "definition": "COUNT_DIVIDED_BY_TOTAL_READS_TIMES_ONE_MILLION",
        "total_reads_denominator": total_reads,
        "rows_checked": rows_checked,
        "rows_not_checkable_zero_denominator": rows_undefined,
        "nonzero_deposited_rpm_rows_with_zero_denominator": nonzero_undefined,
        "mismatch_rows": mismatch_rows,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "status": status_value,
    }


def _hash_open_fd(descriptor: int, filename: str) -> tuple[str, int]:
    del filename
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            break
        digest.update(block)
        byte_count += len(block)
    return digest.hexdigest(), byte_count


def _open_payload_at(data: DirectoryBinding, name: str) -> tuple[int, FileIdentity]:
    return _open_regular_at(data, name, label=f"payload {name}")


def _run_payload_phase(
    *,
    descriptor: int,
    identity: FileIdentity,
    data: DirectoryBinding,
    filename: str,
    operation: Any,
) -> Any:
    _assert_open_file_binding(
        descriptor, identity, data, filename, label=f"payload {filename}"
    )
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = operation()
    _assert_open_file_binding(
        descriptor, identity, data, filename, label=f"payload {filename}"
    )
    return result


def _inspect_file_at(
    data: DirectoryBinding,
    filename: str,
    *,
    entry: Mapping[str, Any],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    descriptor, identity = _open_payload_at(data, filename)
    try:
        observed_sha, observed_bytes = _run_payload_phase(
            descriptor=descriptor,
            identity=identity,
            data=data,
            filename=filename,
            operation=lambda: _hash_open_fd(descriptor, filename),
        )
        if observed_sha != entry["sha256"]:
            raise A2AuditError(f"payload SHA-256 mismatch for {filename}")
        if observed_bytes != entry["bytes"] or observed_bytes != entry["expected_bytes"]:
            raise A2AuditError(f"payload byte-size mismatch for {filename}")
        presence = _run_payload_phase(
            descriptor=descriptor,
            identity=identity,
            data=data,
            filename=filename,
            operation=lambda: _scan_presence_fd(descriptor, filename),
        )
        rpm = _run_payload_phase(
            descriptor=descriptor,
            identity=identity,
            data=data,
            filename=filename,
            operation=lambda: _scan_rpm_consistency_fd(
                descriptor,
                filename,
                total_reads=int(presence["count_sum"]),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ),
        )
        if rpm["rows_checked"] + rpm["rows_not_checkable_zero_denominator"] != presence["rows"]:
            raise A2AuditError(f"payload row count changed during audit: {filename}")
        _assert_open_file_binding(
            descriptor, identity, data, filename, label=f"payload {filename}"
        )
        presence.update(
            {
                "compressed_sha256": observed_sha,
                "compressed_bytes": observed_bytes,
                "gzip_integrity": "PASS",
                "strict_three_column_schema": "PASS",
                "rpm_consistency": rpm,
            }
        )
        return presence
    finally:
        os.close(descriptor)


def _aggregate_presence(
    inspections: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bitsets = [item["bitset"] for item in inspections]
    input_union = _bit_union(bitsets[:2])
    input_intersection = _bit_intersection(bitsets[:2])
    all_union = _bit_union(bitsets)
    all_intersection = _bit_intersection(bitsets)
    endpoint_presence: list[dict[str, Any]] = []
    for sample, inspection in zip(samples[2:], inspections[2:]):
        bitset = inspection["bitset"]
        endpoint_presence.append(
            {
                "order": sample["order"],
                "filename": sample["filename"],
                "sample_group": sample["sample_group"],
                "condition": sample["condition"],
                "replicate": sample["replicate"],
                "shared_with_input_union_keys": _intersection_count(bitset, input_union),
                "endpoint_only_vs_input_union_keys": _left_only_count(bitset, input_union),
                "missing_from_endpoint_vs_input_union_keys": _left_only_count(input_union, bitset),
            }
        )
    reasons: Counter[str] = Counter()
    for inspection in inspections:
        reasons.update(inspection["invalid_key_reason_counts"])
    summary = {
        "key_space": "DNA_10MER_BASE4_0_TO_1048575",
        "bitset_used": True,
        "key_space_size": KEY_SPACE_SIZE,
        "total_rows": sum(int(item["rows"]) for item in inspections),
        "valid_key_rows": sum(int(item["valid_key_rows"]) for item in inspections),
        "invalid_key_rows": sum(int(item["invalid_key_rows"]) for item in inspections),
        "invalid_key_reason_counts": dict(sorted(reasons.items())),
        "duplicate_valid_key_rows": 0,
        "input_two_file_union_keys": _bit_count(input_union),
        "input_two_file_intersection_keys": _bit_count(input_intersection),
        "all_30_file_union_keys": _bit_count(all_union),
        "all_30_file_intersection_keys": _bit_count(all_intersection),
    }
    return summary, endpoint_presence


def _aggregate_rpm_validation(
    inspections: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mismatch_files = [
        sample["filename"]
        for inspection, sample in zip(inspections, samples)
        if inspection["rpm_consistency"]["status"] == "FAIL_MISMATCH"
    ]
    zero_files = [
        sample["filename"]
        for inspection, sample in zip(inspections, samples)
        if inspection["rpm_consistency"]["status"] == "UNDEFINED_ZERO_DENOMINATOR"
    ]
    nonzero_zero_files = [
        sample["filename"]
        for inspection, sample in zip(inspections, samples)
        if int(inspection["rpm_consistency"]["nonzero_deposited_rpm_rows_with_zero_denominator"]) > 0
    ]
    if mismatch_files and zero_files:
        status_value = "FAIL_MISMATCH_AND_UNDEFINED_ZERO_DENOMINATOR"
    elif mismatch_files:
        status_value = "FAIL_MISMATCH"
    elif zero_files:
        status_value = "UNDEFINED_ZERO_DENOMINATOR"
    else:
        status_value = "PASS"
    return {
        "status": status_value,
        "all_pass": status_value == "PASS",
        "total_mismatch_rows": sum(int(item["rpm_consistency"]["mismatch_rows"]) for item in inspections),
        "files_with_mismatches": mismatch_files,
        "zero_denominator_files": zero_files,
        "total_nonzero_deposited_rpm_rows_with_zero_denominator": sum(
            int(item["rpm_consistency"]["nonzero_deposited_rpm_rows_with_zero_denominator"])
            for item in inspections
        ),
        "files_with_nonzero_deposited_rpm_zero_denominator": nonzero_zero_files,
    }


def _reconcile_aggregate_preflight(
    presence: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = protocol["aggregate_preflight_evidence"]["observed_aggregates"]
    expected = {
        "total_rows": frozen["total_rows"],
        "valid_key_rows": frozen["valid_key_rows"],
        "invalid_key_rows": frozen["invalid_key_rows"],
        "input_union": frozen["input_union"],
        "input_intersection": frozen["input_intersection"],
        "all_30_union": frozen["all_30_union"],
        "all_30_intersection": frozen["all_30_intersection"],
    }
    observed = {
        "total_rows": presence["total_rows"],
        "valid_key_rows": presence["valid_key_rows"],
        "invalid_key_rows": presence["invalid_key_rows"],
        "input_union": presence["input_two_file_union_keys"],
        "input_intersection": presence["input_two_file_intersection_keys"],
        "all_30_union": presence["all_30_file_union_keys"],
        "all_30_intersection": presence["all_30_file_intersection_keys"],
    }
    if observed != expected:
        raise A2AuditError("formal aggregate reconciliation mismatch against protocol preflight evidence")
    return {
        "status": "MATCH",
        "source": "PROTOCOL_AGGREGATE_PREFLIGHT_EVIDENCE",
        "expected": expected,
        "observed": observed,
    }


def _public_file_summary(
    inspection: Mapping[str, Any], sample: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "order": sample["order"],
        "gsm_accession": sample["gsm_accession"],
        "filename": sample["filename"],
        "sample_group": sample["sample_group"],
        "condition": sample["condition"],
        "replicate": sample["replicate"],
        "compressed_sha256": inspection["compressed_sha256"],
        "compressed_bytes": inspection["compressed_bytes"],
        "gzip_integrity": inspection["gzip_integrity"],
        "strict_three_column_schema": inspection["strict_three_column_schema"],
        "rows": inspection["rows"],
        "valid_key_rows": inspection["valid_key_rows"],
        "invalid_key_rows": inspection["invalid_key_rows"],
        "invalid_key_reason_counts": inspection["invalid_key_reason_counts"],
        "distinct_valid_keys": inspection["distinct_valid_keys"],
        "duplicate_valid_key_rows": inspection["duplicate_valid_key_rows"],
        "zero_count_rows": inspection["zero_count_rows"],
        "zero_rpm_rows": inspection["zero_rpm_rows"],
        "count_sum": int(inspection["count_sum"]),
        "rpm_sum": _stable_number(float(inspection["rpm_sum"])),
        "rpm_consistency": inspection["rpm_consistency"],
    }


def _validate_reason_counts(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or not set(value).issubset({"INVALID_LENGTH", "INVALID_ALPHABET"}):
        raise A2AuditError(f"{label} invalid reason schema mismatch")
    for reason, count in value.items():
        _nonnegative_integer(count, label=f"{label}.{reason}")


def _validate_rpm_record(value: Any, *, label: str) -> None:
    record = _require_exact_keys(
        value,
        {
            "definition", "total_reads_denominator", "rows_checked",
            "rows_not_checkable_zero_denominator",
            "nonzero_deposited_rpm_rows_with_zero_denominator", "mismatch_rows",
            "max_absolute_error", "max_relative_error", "status",
        },
        label=label,
    )
    _require_exact_value(record["definition"], "COUNT_DIVIDED_BY_TOTAL_READS_TIMES_ONE_MILLION", label=f"{label}.definition")
    for key in (
        "total_reads_denominator", "rows_checked", "rows_not_checkable_zero_denominator",
        "nonzero_deposited_rpm_rows_with_zero_denominator", "mismatch_rows",
    ):
        _nonnegative_integer(record[key], label=f"{label}.{key}")
    if record["status"] not in {"PASS", "FAIL_MISMATCH", "UNDEFINED_ZERO_DENOMINATOR"}:
        raise A2AuditError(f"{label}.status mismatch")
    for key in ("max_absolute_error", "max_relative_error"):
        if record[key] is not None:
            _nonnegative_finite(record[key], label=f"{label}.{key}")
    if record["status"] == "UNDEFINED_ZERO_DENOMINATOR":
        if (
            record["total_reads_denominator"] != 0
            or record["rows_checked"] != 0
            or record["mismatch_rows"] != 0
            or record["max_absolute_error"] is not None
            or record["max_relative_error"] is not None
        ):
            raise A2AuditError(f"{label} zero denominator invariants mismatch")
    else:
        if (
            record["total_reads_denominator"] == 0
            or record["rows_not_checkable_zero_denominator"] != 0
            or record["nonzero_deposited_rpm_rows_with_zero_denominator"] != 0
        ):
            raise A2AuditError(f"{label} checked denominator invariants mismatch")
        if record["status"] == "PASS" and record["mismatch_rows"] != 0:
            raise A2AuditError(f"{label} PASS mismatch count is nonzero")
        if record["status"] == "FAIL_MISMATCH" and record["mismatch_rows"] == 0:
            raise A2AuditError(f"{label} FAIL_MISMATCH has no mismatch rows")


def _validate_closed_report_schema(report: Mapping[str, Any]) -> None:
    """Exact allowlist and invariant validation; the primary no-raw boundary."""

    top = _require_exact_keys(
        report,
        {
            "contract_id", "schema_version", "report_type", "record_id", "phase_id",
            "run_id", "audit_execution_id", "recorded_at", "execution_scope",
            "audit_execution_status", "payload_integrity_status", "rpm_validation_status",
            "aggregate_reconciliation_status", "dataset_qualification_status",
            "training_authorization", "recoverability_status", "evidence_status",
            "scientific_claim_status", "paper_method_reproduced", "dataset_id",
            "bindings", "structural_validation", "row_semantics", "files",
            "presence_aggregates", "rpm_validation_aggregate", "aggregate_reconciliation",
            "endpoint_presence_relative_to_input_union", "source_anchor",
            "endpoint_materialization", "qualification", "unresolved_blockers",
            "record_materialization", "output_policy",
        },
        label="report",
    )
    fixed = {
        "contract_id": CONTRACT_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "phase_id": "A1",
        "execution_scope": "ORDINARY_PUBLIC_CPU_DATA_AUDIT",
        "audit_execution_status": "COMPLETED",
        "payload_integrity_status": "PASS",
        "aggregate_reconciliation_status": "MATCH",
        "dataset_qualification_status": "NOT_QUALIFIED",
        "training_authorization": "DENIED",
        "recoverability_status": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "paper_method_reproduced": False,
        "dataset_id": DATASET_ID,
    }
    for key, expected in fixed.items():
        _require_exact_value(top[key], expected, label=f"report.{key}")
    run_id = _validate_identifier(top["run_id"], label="report.run_id")
    execution_id = _validate_identifier(top["audit_execution_id"], label="report.audit_execution_id")
    _require_exact_value(top["record_id"], f"GSE145046_A2_AUDIT_{execution_id}", label="report.record_id")
    _validate_run_metadata(
        run_id=run_id,
        audit_execution_id=execution_id,
        code_commit=top["bindings"]["code"]["code_commit"] if isinstance(top["bindings"], Mapping) and isinstance(top["bindings"].get("code"), Mapping) else None,
        recorded_at=top["recorded_at"],
        test_sha256=top["bindings"]["code"]["test"]["sha256"] if isinstance(top["bindings"], Mapping) and isinstance(top["bindings"].get("code"), Mapping) and isinstance(top["bindings"]["code"].get("test"), Mapping) else None,
    )

    bindings = _require_exact_keys(
        top["bindings"], {"contract", "protocol", "p0_manifest", "payloads_bound_by_manifest_sha256", "run", "code"}, label="report.bindings"
    )
    for label in ("contract", "p0_manifest"):
        item = _require_exact_keys(bindings[label], {"sha256", "bytes"}, label=f"report.bindings.{label}")
        _normalize_sha256(item["sha256"], label=f"report.bindings.{label}.sha256")
        _positive_integer(item["bytes"], label=f"report.bindings.{label}.bytes")
    protocol_binding = _require_exact_keys(bindings["protocol"], {"sha256", "bytes", "verification"}, label="report.bindings.protocol")
    _normalize_sha256(protocol_binding["sha256"], label="report.bindings.protocol.sha256")
    _positive_integer(protocol_binding["bytes"], label="report.bindings.protocol.bytes")
    _require_exact_value(protocol_binding["verification"], "VERIFIED_CANONICAL_CALLER_ACTUAL_AND_HEAD", label="report.bindings.protocol.verification")
    _require_exact_value(
        protocol_binding["sha256"],
        CANONICAL_PROTOCOL_SHA256,
        label="report canonical protocol SHA-256",
    )
    _require_exact_value(bindings["payloads_bound_by_manifest_sha256"], True, label="report.bindings.payloads")
    _require_exact_value(
        bindings["run"],
        {"parent_a1_run_id": run_id, "audit_execution_id": execution_id, "recorded_at": top["recorded_at"]},
        label="report.bindings.run",
    )
    code = _require_exact_keys(
        bindings["code"], {"code_commit", "git_head", "git_worktree_clean", "verification", "protocol", "script", "test"}, label="report.bindings.code"
    )
    if code["code_commit"] != code["git_head"] or COMMIT_RE.fullmatch(code["code_commit"]) is None:
        raise A2AuditError("report code commit binding mismatch")
    _require_exact_value(code["git_worktree_clean"], True, label="report code clean")
    _require_exact_value(code["verification"], "VERIFIED_HEAD_CLEAN_AND_EXACT_BYTES", label="report code verification")
    for label, expected_path in (
        ("protocol", PROTOCOL_REPOSITORY_PATH), ("script", SCRIPT_REPOSITORY_PATH), ("test", TEST_REPOSITORY_PATH)
    ):
        item = _require_exact_keys(code[label], {"repository_relative_path", "sha256", "bytes", "head_blob_match"}, label=f"report code {label}")
        _require_exact_value(item["repository_relative_path"], expected_path, label=f"report code {label} path")
        _normalize_sha256(item["sha256"], label=f"report code {label} sha256")
        _positive_integer(item["bytes"], label=f"report code {label} bytes")
        _require_exact_value(item["head_blob_match"], True, label=f"report code {label} head match")
    if protocol_binding["sha256"] != code["protocol"]["sha256"] or protocol_binding["bytes"] != code["protocol"]["bytes"]:
        raise A2AuditError("report protocol binding duplication mismatch")

    _require_exact_value(
        top["structural_validation"],
        {
            "status": "PASS", "ordinary_public_only": True,
            "exact_sample_universe": True, "manifest_file_count": 30,
            "payload_file_count": 30, "all_payload_hashes_match": True,
            "all_gzip_streams_complete": True, "all_rows_strict_three_columns": True,
            "all_counts_nonnegative_integers": True,
            "all_files_free_of_duplicate_valid_keys": True,
            "network_access_used": False,
        },
        label="report.structural_validation",
    )
    _require_exact_value(
        top["row_semantics"],
        {
            "column_1": "RANDOM_N10_KEY", "column_2": "COUNT", "column_3": "RPM",
            "rpm_definition": "COUNT_DIVIDED_BY_TOTAL_READS_TIMES_ONE_MILLION",
            "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
            "absent_key_is_missing": True, "absent_key_imputed_as_zero": False,
            "endpoint_values_materialized": False,
        },
        label="report.row_semantics",
    )

    files = top["files"]
    if not isinstance(files, list) or len(files) != 30:
        raise A2AuditError("report.files must contain exactly 30 entries")
    file_keys = {
        "order", "gsm_accession", "filename", "sample_group", "condition", "replicate",
        "compressed_sha256", "compressed_bytes", "gzip_integrity",
        "strict_three_column_schema", "rows", "valid_key_rows", "invalid_key_rows",
        "invalid_key_reason_counts", "distinct_valid_keys", "duplicate_valid_key_rows",
        "zero_count_rows", "zero_rpm_rows", "count_sum", "rpm_sum", "rpm_consistency",
    }
    calculated_total_rows = 0
    calculated_valid_rows = 0
    calculated_invalid_rows = 0
    calculated_reasons: Counter[str] = Counter()
    calculated_mismatch_rows = 0
    calculated_nonzero_zero_denominator_rows = 0
    calculated_mismatch_files: list[str] = []
    calculated_zero_denominator_files: list[str] = []
    calculated_nonzero_zero_denominator_files: list[str] = []
    for order, (item, expected) in enumerate(zip(files, EXPECTED_SAMPLE_SPECS), start=1):
        entry = _require_exact_keys(item, file_keys, label=f"report.files[{order}]")
        observed = (entry["filename"], entry["gsm_accession"], entry["sample_group"], entry["condition"], entry["replicate"])
        _require_exact_value(entry["order"], order, label="report file order")
        _require_exact_value(observed, expected, label="report file role tuple")
        _normalize_sha256(entry["compressed_sha256"], label="report payload SHA-256")
        _positive_integer(entry["compressed_bytes"], label="report payload bytes")
        _require_exact_value(entry["gzip_integrity"], "PASS", label="report gzip status")
        _require_exact_value(entry["strict_three_column_schema"], "PASS", label="report row schema status")
        for key in ("rows", "valid_key_rows", "invalid_key_rows", "distinct_valid_keys", "duplicate_valid_key_rows", "zero_count_rows", "zero_rpm_rows", "count_sum"):
            _nonnegative_integer(entry[key], label=f"report file {key}")
        if entry["rows"] != entry["valid_key_rows"] + entry["invalid_key_rows"]:
            raise A2AuditError("report file row counts do not reconcile")
        if entry["duplicate_valid_key_rows"] != 0:
            raise A2AuditError("report cannot publish duplicate valid key evidence as structural PASS")
        _validate_reason_counts(entry["invalid_key_reason_counts"], label="report file invalid reasons")
        if (
            sum(entry["invalid_key_reason_counts"].values())
            != entry["invalid_key_rows"]
            or entry["distinct_valid_keys"] != entry["valid_key_rows"]
            or entry["valid_key_rows"] > KEY_SPACE_SIZE
            or entry["zero_count_rows"] > entry["rows"]
            or entry["zero_rpm_rows"] > entry["rows"]
        ):
            raise A2AuditError("report file key/count aggregates do not reconcile")
        _nonnegative_finite(entry["rpm_sum"], label="report file RPM sum")
        _validate_rpm_record(entry["rpm_consistency"], label="report file RPM")
        if (
            entry["rpm_consistency"]["total_reads_denominator"]
            != entry["count_sum"]
            or entry["rpm_consistency"]["rows_checked"]
            + entry["rpm_consistency"]["rows_not_checkable_zero_denominator"]
            != entry["rows"]
            or entry["rpm_consistency"][
                "nonzero_deposited_rpm_rows_with_zero_denominator"
            ]
            > entry["rpm_consistency"]["rows_not_checkable_zero_denominator"]
        ):
            raise A2AuditError("report file and RPM row evidence do not reconcile")
        calculated_total_rows += entry["rows"]
        calculated_valid_rows += entry["valid_key_rows"]
        calculated_invalid_rows += entry["invalid_key_rows"]
        calculated_reasons.update(entry["invalid_key_reason_counts"])
        calculated_mismatch_rows += entry["rpm_consistency"]["mismatch_rows"]
        calculated_nonzero_zero_denominator_rows += entry["rpm_consistency"][
            "nonzero_deposited_rpm_rows_with_zero_denominator"
        ]
        if entry["rpm_consistency"]["status"] == "FAIL_MISMATCH":
            calculated_mismatch_files.append(entry["filename"])
        if entry["rpm_consistency"]["status"] == "UNDEFINED_ZERO_DENOMINATOR":
            calculated_zero_denominator_files.append(entry["filename"])
        if entry["rpm_consistency"]["nonzero_deposited_rpm_rows_with_zero_denominator"] > 0:
            calculated_nonzero_zero_denominator_files.append(entry["filename"])

    presence = _require_exact_keys(
        top["presence_aggregates"],
        {"key_space", "bitset_used", "key_space_size", "total_rows", "valid_key_rows", "invalid_key_rows", "invalid_key_reason_counts", "duplicate_valid_key_rows", "input_two_file_union_keys", "input_two_file_intersection_keys", "all_30_file_union_keys", "all_30_file_intersection_keys"},
        label="report.presence_aggregates",
    )
    _require_exact_value(presence["key_space"], "DNA_10MER_BASE4_0_TO_1048575", label="report key space")
    _require_exact_value(presence["bitset_used"], True, label="report bitset flag")
    _require_exact_value(presence["key_space_size"], KEY_SPACE_SIZE, label="report key space size")
    for key in ("total_rows", "valid_key_rows", "invalid_key_rows", "duplicate_valid_key_rows", "input_two_file_union_keys", "input_two_file_intersection_keys", "all_30_file_union_keys", "all_30_file_intersection_keys"):
        _nonnegative_integer(presence[key], label=f"report presence {key}")
    if presence["duplicate_valid_key_rows"] != 0 or presence["total_rows"] != presence["valid_key_rows"] + presence["invalid_key_rows"]:
        raise A2AuditError("report presence row invariants mismatch")
    _validate_reason_counts(presence["invalid_key_reason_counts"], label="report aggregate invalid reasons")
    if (
        presence["total_rows"] != calculated_total_rows
        or presence["valid_key_rows"] != calculated_valid_rows
        or presence["invalid_key_rows"] != calculated_invalid_rows
        or presence["invalid_key_reason_counts"] != dict(sorted(calculated_reasons.items()))
        or sum(presence["invalid_key_reason_counts"].values())
        != presence["invalid_key_rows"]
        or presence["input_two_file_intersection_keys"]
        > presence["input_two_file_union_keys"]
        or presence["all_30_file_intersection_keys"]
        > presence["all_30_file_union_keys"]
        or presence["input_two_file_union_keys"] > KEY_SPACE_SIZE
        or presence["all_30_file_union_keys"] > KEY_SPACE_SIZE
    ):
        raise A2AuditError("report file and aggregate row evidence mismatch")

    rpm_aggregate = _require_exact_keys(
        top["rpm_validation_aggregate"],
        {"status", "all_pass", "total_mismatch_rows", "files_with_mismatches", "zero_denominator_files", "total_nonzero_deposited_rpm_rows_with_zero_denominator", "files_with_nonzero_deposited_rpm_zero_denominator"},
        label="report.rpm_validation_aggregate",
    )
    allowed_rpm_statuses = {"PASS", "FAIL_MISMATCH", "UNDEFINED_ZERO_DENOMINATOR", "FAIL_MISMATCH_AND_UNDEFINED_ZERO_DENOMINATOR"}
    if rpm_aggregate["status"] not in allowed_rpm_statuses or top["rpm_validation_status"] != rpm_aggregate["status"]:
        raise A2AuditError("report aggregate RPM status mismatch")
    if rpm_aggregate["all_pass"] is not (rpm_aggregate["status"] == "PASS"):
        raise A2AuditError("report aggregate RPM all_pass mismatch")
    for key in ("total_mismatch_rows", "total_nonzero_deposited_rpm_rows_with_zero_denominator"):
        _nonnegative_integer(rpm_aggregate[key], label=f"report RPM aggregate {key}")
    for key in ("files_with_mismatches", "zero_denominator_files", "files_with_nonzero_deposited_rpm_zero_denominator"):
        if not isinstance(rpm_aggregate[key], list) or any(name not in EXPECTED_FILENAMES for name in rpm_aggregate[key]):
            raise A2AuditError(f"report RPM aggregate {key} mismatch")
    if calculated_mismatch_files and calculated_zero_denominator_files:
        calculated_rpm_status = "FAIL_MISMATCH_AND_UNDEFINED_ZERO_DENOMINATOR"
    elif calculated_mismatch_files:
        calculated_rpm_status = "FAIL_MISMATCH"
    elif calculated_zero_denominator_files:
        calculated_rpm_status = "UNDEFINED_ZERO_DENOMINATOR"
    else:
        calculated_rpm_status = "PASS"
    if (
        rpm_aggregate["status"] != calculated_rpm_status
        or rpm_aggregate["total_mismatch_rows"] != calculated_mismatch_rows
        or rpm_aggregate["files_with_mismatches"] != calculated_mismatch_files
        or rpm_aggregate["zero_denominator_files"] != calculated_zero_denominator_files
        or rpm_aggregate["total_nonzero_deposited_rpm_rows_with_zero_denominator"]
        != calculated_nonzero_zero_denominator_rows
        or rpm_aggregate["files_with_nonzero_deposited_rpm_zero_denominator"]
        != calculated_nonzero_zero_denominator_files
    ):
        raise A2AuditError("report per-file and aggregate RPM evidence mismatch")

    reconciliation = _require_exact_keys(top["aggregate_reconciliation"], {"status", "source", "expected", "observed"}, label="report.aggregate_reconciliation")
    _require_exact_value(reconciliation["status"], "MATCH", label="report reconciliation status")
    _require_exact_value(reconciliation["source"], "PROTOCOL_AGGREGATE_PREFLIGHT_EVIDENCE", label="report reconciliation source")
    aggregate_keys = {"total_rows", "valid_key_rows", "invalid_key_rows", "input_union", "input_intersection", "all_30_union", "all_30_intersection"}
    expected_aggregate = _require_exact_keys(reconciliation["expected"], aggregate_keys, label="report reconciliation expected")
    observed_aggregate = _require_exact_keys(reconciliation["observed"], aggregate_keys, label="report reconciliation observed")
    for side, values in (
        ("expected", expected_aggregate),
        ("observed", observed_aggregate),
    ):
        for value in values.values():
            _nonnegative_integer(
                value, label=f"report reconciled {side} aggregate"
            )
    _require_exact_value(
        expected_aggregate,
        observed_aggregate,
        label="report reconciliation values",
    )
    if observed_aggregate != {
        "total_rows": presence["total_rows"],
        "valid_key_rows": presence["valid_key_rows"],
        "invalid_key_rows": presence["invalid_key_rows"],
        "input_union": presence["input_two_file_union_keys"],
        "input_intersection": presence["input_two_file_intersection_keys"],
        "all_30_union": presence["all_30_file_union_keys"],
        "all_30_intersection": presence["all_30_file_intersection_keys"],
    }:
        raise A2AuditError("report reconciliation and presence aggregates mismatch")

    endpoints = top["endpoint_presence_relative_to_input_union"]
    endpoint_keys = {"order", "filename", "sample_group", "condition", "replicate", "shared_with_input_union_keys", "endpoint_only_vs_input_union_keys", "missing_from_endpoint_vs_input_union_keys"}
    if not isinstance(endpoints, list) or len(endpoints) != 28:
        raise A2AuditError("report endpoint presence entry count mismatch")
    for item, expected_order in zip(endpoints, range(3, 31)):
        entry = _require_exact_keys(item, endpoint_keys, label="report endpoint presence")
        expected = EXPECTED_SAMPLE_SPECS[expected_order - 1]
        observed = (entry["filename"], EXPECTED_SAMPLE_SPECS[expected_order - 1][1], entry["sample_group"], entry["condition"], entry["replicate"])
        _require_exact_value(
            entry["order"], expected_order, label="report endpoint presence order"
        )
        _require_exact_value(
            observed, expected, label="report endpoint presence role tuple"
        )
        for key in ("shared_with_input_union_keys", "endpoint_only_vs_input_union_keys", "missing_from_endpoint_vs_input_union_keys"):
            _nonnegative_integer(entry[key], label=f"report endpoint presence {key}")

    _require_exact_value(
        top["source_anchor"],
        {"n10_locus_status": "CLOSED_AT_PRIMER_LEVEL", "full_reporter_anchor_status": "NOT_CLOSED", "canonical_source_hash": None},
        label="report.source_anchor",
    )
    _require_exact_value(
        top["endpoint_materialization"],
        {
            "monosome_polysome_formula": "KNOWN_REPLICATE_LOCAL_NOT_EXECUTED",
            "monosome_polysome_zero_or_pseudocount_rule": "UNKNOWN_NOT_ASSERTED",
            "functional_cap_free_bound": {"bound_construction_status": "NOT_RECOVERED", "endpoint_status": "BLOCKED_NOT_MATERIALIZED"},
            "non_functional_cap_apppg_ribosome_free_bound": {"available_pair_count": 1, "paper_use": "PRIMARILY_AGGREGATED_MOTIF_ANALYSIS", "per_variant_endpoint_status": "BLOCKED_NOT_MATERIALIZED"},
            "facs": "BLOCKED_NOT_MATERIALIZED",
            "in_vivo_half_life": "BLOCKED_NOT_MATERIALIZED",
            "in_vitro_half_life": "BLOCKED_NOT_MATERIALIZED",
            "any_endpoint_materialized": False,
        },
        label="report.endpoint_materialization",
    )
    _require_exact_value(
        top["qualification"],
        {
            "classification": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
            "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
            "qualified": False, "training_allowed": False, "model_selection_allowed": False,
            "canonical_intervention_records_materialized": False,
            "measured_candidate_pools_materialized": False,
            "endpoint_values_materialized": False,
            "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
            "license": {"status": "UNKNOWN_BLOCKED", "redistribution_allowed": False},
            "foundation_exposure": {"checkpoint_specific_status": "UNKNOWN_NOT_ASSERTED", "project_engineering_consumption": "CONFIRMED", "independent_holdout_eligibility": False},
        },
        label="report.qualification",
    )
    _require_exact_value(top["unresolved_blockers"], list(EXPECTED_BLOCKERS), label="report blockers")
    _require_exact_value(
        top["record_materialization"],
        {"canonical_intervention_record_count": 0, "measured_candidate_pool_count": 0, "canonical_or_pool_artifacts_written": False},
        label="report.record_materialization",
    )
    _require_exact_value(
        top["output_policy"],
        {
            "aggregate_only": True,
            "determinism": "DETERMINISTIC_GIVEN_FIXED_EXPLICIT_PARAMETERS",
            "explicit_parameters_bound": [
                "run_id", "audit_execution_id", "code_commit", "recorded_at",
                "test_sha256", "contract_sha256", "protocol_sha256", "p0_manifest_sha256",
            ],
            "raw_sequence_output": False, "raw_row_output": False,
            "raw_label_output": False, "hidden_sibling_staging": True,
            "fsync_before_publish": True, "atomic_no_replace_publish": True,
            "formal_output_name": "AUDIT_EXECUTION_ID_DOT_JSON",
            "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST_OR_LEDGER",
        },
        label="report.output_policy",
    )


def _build_report(
    *,
    paths: Mapping[str, Path],
    output_binding: DirectoryBinding,
    expected_protocol_sha256: str,
    run_metadata: Mapping[str, str],
) -> dict[str, Any]:
    protocol_raw, code_binding, tracked_state, repo_binding = _verify_repo_binding(
        repo_root=paths["repo_root"],
        protocol_path=paths["protocol"],
        test_path=paths["test"],
        expected_protocol_sha256=expected_protocol_sha256,
        code_commit=run_metadata["code_commit"],
        test_sha256=run_metadata["test_sha256"],
    )
    try:
        protocol = _load_and_validate_protocol(protocol_raw)
        contract_raw, contract_binding, contract_identity = _read_path_bound_bytes(
            paths["contract"], label="contract"
        )
        del contract_raw
        if contract_binding["sha256"] != protocol["authority"]["contract_sha256"]:
            raise A2AuditError("contract SHA-256 mismatch")

        data_binding = _open_directory_no_symlinks(
            paths["manifest"].parent, label="P0 data directory"
        )
        try:
            manifest_name = protocol["inputs"]["p0_manifest"]["filename"]
            if paths["manifest"].name != manifest_name:
                raise A2AuditError("P0 manifest filename mismatch")
            manifest_raw, manifest_binding, manifest_identity = _read_regular_at(
                data_binding, manifest_name, label="P0 manifest"
            )
            if manifest_binding["sha256"] != protocol["inputs"]["p0_manifest"]["sha256"]:
                raise A2AuditError("P0 manifest SHA-256 mismatch")
            entries = _load_and_validate_manifest(manifest_raw, protocol=protocol)
            _inventory_exact_payloads(data_binding, manifest_filename=manifest_name)
            absolute_tolerance = _nonnegative_finite(
                protocol["inputs"]["rpm_definition"]["absolute_tolerance"],
                label="RPM absolute tolerance",
            )
            relative_tolerance = _nonnegative_finite(
                protocol["inputs"]["rpm_definition"]["relative_tolerance"],
                label="RPM relative tolerance",
            )
            inspections = [
                _inspect_file_at(
                    data_binding,
                    name,
                    entry=entries[name],
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                for name in EXPECTED_FILENAMES
            ]
            samples = protocol["inputs"]["samples"]
            presence, endpoint_presence = _aggregate_presence(inspections, samples)
            rpm_aggregate = _aggregate_rpm_validation(inspections, samples)
            reconciliation = _reconcile_aggregate_preflight(presence, protocol)
            current_manifest = _stat_regular_at(data_binding, manifest_name, label="P0 manifest")
            if _file_identity(current_manifest) != manifest_identity:
                raise ScopeViolation("P0 manifest identity changed during audit")
            _assert_directory_binding(data_binding, full=True)
        finally:
            os.close(data_binding.fd)

        _assert_path_identity(paths["contract"], contract_identity, label="contract")
        _assert_repo_binding_stable(
            repo_binding, tracked_state, code_commit=run_metadata["code_commit"]
        )
        _assert_directory_binding(output_binding, full=False)

        protocol_binding = {
            "sha256": code_binding["protocol"]["sha256"],
            "bytes": code_binding["protocol"]["bytes"],
            "verification": "VERIFIED_CANONICAL_CALLER_ACTUAL_AND_HEAD",
        }
        report: dict[str, Any] = {
            "contract_id": CONTRACT_ID,
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": REPORT_TYPE,
            "record_id": f"GSE145046_A2_AUDIT_{run_metadata['audit_execution_id']}",
            "phase_id": "A1",
            "run_id": run_metadata["run_id"],
            "audit_execution_id": run_metadata["audit_execution_id"],
            "recorded_at": run_metadata["recorded_at"],
            "execution_scope": "ORDINARY_PUBLIC_CPU_DATA_AUDIT",
            "audit_execution_status": "COMPLETED",
            "payload_integrity_status": "PASS",
            "rpm_validation_status": rpm_aggregate["status"],
            "aggregate_reconciliation_status": "MATCH",
            "dataset_qualification_status": "NOT_QUALIFIED",
            "training_authorization": "DENIED",
            "recoverability_status": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
            "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
            "scientific_claim_status": "NOT_ESTABLISHED",
            "paper_method_reproduced": False,
            "dataset_id": DATASET_ID,
            "bindings": {
                "contract": contract_binding,
                "protocol": protocol_binding,
                "p0_manifest": manifest_binding,
                "payloads_bound_by_manifest_sha256": True,
                "run": {
                    "parent_a1_run_id": run_metadata["run_id"],
                    "audit_execution_id": run_metadata["audit_execution_id"],
                    "recorded_at": run_metadata["recorded_at"],
                },
                "code": code_binding,
            },
            "structural_validation": {
                "status": "PASS", "ordinary_public_only": True,
                "exact_sample_universe": True, "manifest_file_count": 30,
                "payload_file_count": 30, "all_payload_hashes_match": True,
                "all_gzip_streams_complete": True, "all_rows_strict_three_columns": True,
                "all_counts_nonnegative_integers": True,
                "all_files_free_of_duplicate_valid_keys": True,
                "network_access_used": False,
            },
            "row_semantics": {
                "column_1": "RANDOM_N10_KEY", "column_2": "COUNT", "column_3": "RPM",
                "rpm_definition": "COUNT_DIVIDED_BY_TOTAL_READS_TIMES_ONE_MILLION",
                "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
                "absent_key_is_missing": True, "absent_key_imputed_as_zero": False,
                "endpoint_values_materialized": False,
            },
            "files": [_public_file_summary(item, sample) for item, sample in zip(inspections, samples)],
            "presence_aggregates": presence,
            "rpm_validation_aggregate": rpm_aggregate,
            "aggregate_reconciliation": reconciliation,
            "endpoint_presence_relative_to_input_union": endpoint_presence,
            "source_anchor": {
                "n10_locus_status": "CLOSED_AT_PRIMER_LEVEL",
                "full_reporter_anchor_status": "NOT_CLOSED",
                "canonical_source_hash": None,
            },
            "endpoint_materialization": {
                "monosome_polysome_formula": "KNOWN_REPLICATE_LOCAL_NOT_EXECUTED",
                "monosome_polysome_zero_or_pseudocount_rule": "UNKNOWN_NOT_ASSERTED",
                "functional_cap_free_bound": {"bound_construction_status": "NOT_RECOVERED", "endpoint_status": "BLOCKED_NOT_MATERIALIZED"},
                "non_functional_cap_apppg_ribosome_free_bound": {"available_pair_count": 1, "paper_use": "PRIMARILY_AGGREGATED_MOTIF_ANALYSIS", "per_variant_endpoint_status": "BLOCKED_NOT_MATERIALIZED"},
                "facs": "BLOCKED_NOT_MATERIALIZED",
                "in_vivo_half_life": "BLOCKED_NOT_MATERIALIZED",
                "in_vitro_half_life": "BLOCKED_NOT_MATERIALIZED",
                "any_endpoint_materialized": False,
            },
            # Explicit reconstruction from validated constants; no protocol
            # subtree is copied into the report.
            "qualification": {
                "classification": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
                "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
                "qualified": False, "training_allowed": False,
                "model_selection_allowed": False,
                "canonical_intervention_records_materialized": False,
                "measured_candidate_pools_materialized": False,
                "endpoint_values_materialized": False,
                "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
                "license": {"status": "UNKNOWN_BLOCKED", "redistribution_allowed": False},
                "foundation_exposure": {"checkpoint_specific_status": "UNKNOWN_NOT_ASSERTED", "project_engineering_consumption": "CONFIRMED", "independent_holdout_eligibility": False},
            },
            "unresolved_blockers": list(EXPECTED_BLOCKERS),
            "record_materialization": {
                "canonical_intervention_record_count": 0,
                "measured_candidate_pool_count": 0,
                "canonical_or_pool_artifacts_written": False,
            },
            "output_policy": {
                "aggregate_only": True,
                "determinism": "DETERMINISTIC_GIVEN_FIXED_EXPLICIT_PARAMETERS",
                "explicit_parameters_bound": [
                    "run_id", "audit_execution_id", "code_commit", "recorded_at",
                    "test_sha256", "contract_sha256", "protocol_sha256", "p0_manifest_sha256",
                ],
                "raw_sequence_output": False, "raw_row_output": False,
                "raw_label_output": False, "hidden_sibling_staging": True,
                "fsync_before_publish": True, "atomic_no_replace_publish": True,
                "formal_output_name": "AUDIT_EXECUTION_ID_DOT_JSON",
                "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST_OR_LEDGER",
            },
        }
        _validate_closed_report_schema(report)
        return report
    finally:
        os.close(repo_binding.fd)


def _json_bytes(report: Mapping[str, Any]) -> bytes:
    _validate_closed_report_schema(report)
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _assert_output_absent(binding: DirectoryBinding, basename: str) -> None:
    try:
        os.stat(basename, dir_fd=binding.fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise A2AuditError("refusing to overwrite existing output")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise A2AuditError("short write to hidden staging file")
        written += count


def _publish_single_file_at(
    binding: DirectoryBinding, basename: str, payload: bytes
) -> None:
    _safe_basename(basename, label="output basename")
    _assert_directory_binding(binding, full=False)
    _assert_output_absent(binding, basename)
    staging = f".{basename}.partial-staging-{secrets.token_hex(12)}"
    _safe_basename(staging, label="hidden staging basename")
    # The staging inode remains open through publication.  A read/write fd is
    # used only so its bytes can be re-hashed after link; mode 0400 prevents a
    # new path-based writer from opening it during the publication window.
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(staging, flags, 0o400, dir_fd=binding.fd)
    except FileExistsError as exc:
        raise A2AuditError("hidden staging collision") from exc
    expected_payload_sha = hashlib.sha256(payload).hexdigest()
    try:
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            staging_identity_before_link = _file_identity(os.fstat(descriptor))
            staging_leaf_before_link = _stat_regular_at(
                binding, staging, label="hidden staging file"
            )
            if _file_identity(staging_leaf_before_link) != staging_identity_before_link:
                raise ScopeViolation("hidden staging leaf identity changed before publish")
            observed_sha, observed_bytes = _hash_open_fd(descriptor, staging)
            if observed_sha != expected_payload_sha or observed_bytes != len(payload):
                raise ScopeViolation("hidden staging bytes changed before publish")
        except Exception:
            # A partial or suspicious hidden name is retained as evidence.
            raise

        os.fsync(binding.fd)
        _assert_directory_binding(binding, full=False)
        _assert_output_absent(binding, basename)
        current_staging_leaf = _stat_regular_at(
            binding, staging, label="hidden staging file"
        )
        if _file_identity(current_staging_leaf) != staging_identity_before_link:
            raise ScopeViolation("hidden staging leaf identity changed before link")
        try:
            os.link(
                staging,
                basename,
                src_dir_fd=binding.fd,
                dst_dir_fd=binding.fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise A2AuditError("refusing to overwrite concurrently created output") from exc
        except OSError as exc:
            raise A2AuditError("atomic no-replace publication failed; staging retained") from exc

        # A same-directory writer could replace the staging name between the
        # last stat and link.  Compare the final name to the still-open inode
        # and re-hash that inode.  On any mismatch, remove only our reserved
        # final basename before failing closed.
        staging_identity_after_link = _file_identity(os.fstat(descriptor))
        final_matches_open_inode = False
        try:
            final_info = os.stat(
                basename, dir_fd=binding.fd, follow_symlinks=False
            )
            final_matches_open_inode = (
                stat.S_ISREG(final_info.st_mode)
                and _file_identity(final_info) == staging_identity_after_link
                and staging_identity_before_link[:-1]
                == staging_identity_after_link[:-1]
            )
            observed_sha, observed_bytes = _hash_open_fd(descriptor, staging)
            final_matches_open_inode = final_matches_open_inode and (
                observed_sha == expected_payload_sha
                and observed_bytes == len(payload)
            )
        except FileNotFoundError:
            final_matches_open_inode = False
        if not final_matches_open_inode:
            try:
                os.unlink(basename, dir_fd=binding.fd)
            except FileNotFoundError:
                pass
            os.fsync(binding.fd)
            raise ScopeViolation(
                "published basename did not bind the open staging inode"
            )

        os.fsync(binding.fd)
        try:
            os.unlink(staging, dir_fd=binding.fd)
        except OSError as exc:
            raise A2AuditError(
                "published output but hidden staging cleanup failed"
            ) from exc
        os.fsync(binding.fd)
        final_after_cleanup = os.stat(
            basename, dir_fd=binding.fd, follow_symlinks=False
        )
        if _file_identity(final_after_cleanup) != _file_identity(os.fstat(descriptor)):
            raise ScopeViolation("published output inode changed during cleanup")
    finally:
        os.close(descriptor)


def audit_gse145046_a2(
    *,
    contract_path: Path,
    protocol_path: Path,
    expected_protocol_sha256: str,
    manifest_path: Path,
    output_path: Path,
    repo_root: Path,
    test_path: Path,
    run_id: str,
    audit_execution_id: str,
    code_commit: str,
    recorded_at: str,
    test_sha256: str,
) -> dict[str, Any]:
    run_metadata = _validate_run_metadata(
        run_id=run_id,
        audit_execution_id=audit_execution_id,
        code_commit=code_commit,
        recorded_at=recorded_at,
        test_sha256=test_sha256,
    )
    paths, output_binding = _prepare_paths_before_read(
        contract_path=Path(contract_path),
        protocol_path=Path(protocol_path),
        manifest_path=Path(manifest_path),
        output_path=Path(output_path),
        repo_root=Path(repo_root),
        test_path=Path(test_path),
        audit_execution_id=run_metadata["audit_execution_id"],
    )
    try:
        report = _build_report(
            paths=paths,
            output_binding=output_binding,
            expected_protocol_sha256=expected_protocol_sha256,
            run_metadata=run_metadata,
        )
        payload = _json_bytes(report)
        _publish_single_file_at(output_binding, paths["output"].name, payload)
        return report
    finally:
        os.close(output_binding.fd)


def audit(**kwargs: Any) -> dict[str, Any]:
    return audit_gse145046_a2(**kwargs)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--p0-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--test-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-execution-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--recorded-at", required=True)
    parser.add_argument("--test-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_gse145046_a2(
        contract_path=args.contract,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        manifest_path=args.p0_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
        test_path=args.test_path,
        run_id=args.run_id,
        audit_execution_id=args.audit_execution_id,
        code_commit=args.code_commit,
        recorded_at=args.recorded_at,
        test_sha256=args.test_sha256,
    )
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "report_type": REPORT_TYPE,
                "record_id": report["record_id"],
                "run_id": report["run_id"],
                "audit_execution_id": report["audit_execution_id"],
                "audit_execution_status": report["audit_execution_status"],
                "dataset_qualification_status": report["dataset_qualification_status"],
                "training_authorization": report["training_authorization"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
