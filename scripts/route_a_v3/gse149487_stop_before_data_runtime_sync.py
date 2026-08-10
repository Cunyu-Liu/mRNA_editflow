#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed GSE149487 EVT-036 runtime sync.

Production has no caller-selected repository, run root, or source artifact.  The
only caller-selected path is a prepared-artifact directory outside the run and
data roots.  An UNKNOWN/PENDING config stops before the run root is opened.
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


CONFIG_REPO_PATH = "configs/route_a_v3_gse149487_stop_before_data_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/gse149487_stop_before_data_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse149487_stop_before_data_runtime_sync.py"
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


def validate_bound_config(config: dict[str, Any]) -> None:
    """Validate all authority/binding bytes before any run-root access."""

    _expect_exact(
        config.get("schema_version"),
        "route_a_v3_gse149487_stop_before_data_runtime_sync.v1",
        label="runtime-sync schema",
    )
    _expect_exact(
        config.get("protocol_id"),
        "ROUTE_A_V3_GSE149487_STOP_BEFORE_DATA_RUNTIME_SYNC_V1",
        label="runtime-sync protocol",
    )
    _expect_exact(config.get("event_id"), "A1-EVT-036", label="event id")
    binding = config.get("implementation_binding")
    if not isinstance(binding, dict):
        raise BindingError("implementation_binding is absent")
    if binding.get("status") != "BOUND":
        raise BindingError("runtime-sync config is not BOUND")
    implementation_commit = _expect_hex(
        binding.get("implementation_commit"), HEX40, label="implementation commit"
    )
    base_commit = _expect_hex(binding.get("base_commit"), HEX40, label="base commit")
    if base_commit != "aeecf0f043a94f2e5a738807c6d13d92f16e129f":
        raise BindingError("runtime-sync base authority drift")
    if implementation_commit == base_commit:
        raise BindingError("implementation commit did not advance from the fixed base")
    _expect_hex(
        binding.get("implementation_script_sha256"),
        HEX64,
        label="implementation script SHA-256",
    )
    _expect_hex(
        binding.get("implementation_test_sha256"),
        HEX64,
        label="implementation test SHA-256",
    )
    _expect_exact(
        binding.get("implementation_script_path"), SCRIPT_REPO_PATH, label="script path"
    )
    _expect_exact(binding.get("implementation_test_path"), TEST_REPO_PATH, label="test path")
    _expect_exact(
        binding.get("unknown_to_bound_scalar_paths"),
        ["implementation_binding.status", "implementation_binding.implementation_commit"],
        label="config-only binding scalar allowlist",
    )

    authority = config.get("repository_authority")
    if not isinstance(authority, dict):
        raise BindingError("repository_authority is absent")
    _expect_exact(
        authority.get("production_repo_root"),
        str(PRODUCTION_REPO_ROOT),
        label="production repository root",
    )
    _expect_exact(authority.get("branch"), "routea-v3-a1-20260810", label="branch")
    _expect_exact(
        authority.get("implementation_commit_expected_parent"),
        base_commit,
        label="implementation parent",
    )
    _expect_exact(
        authority.get("implementation_commit_exact_changed_paths"),
        sorted(
            [
                CONFIG_REPO_PATH,
                "docs/execution/route_a_v3_a1_interim.yaml",
                "docs/execution/route_a_v3_registry_manifest.json",
                SCRIPT_REPO_PATH,
                "scripts/route_a_v3/validate_a0_bundle.py",
                "tests/route_a_v3/test_a0_integrity_guards.py",
                TEST_REPO_PATH,
            ]
        ),
        label="implementation commit path closure",
    )
    _expect_exact(
        authority.get("binding_commit_exact_changed_paths"),
        [CONFIG_REPO_PATH],
        label="binding commit path closure",
    )
    _expect_hex(
        authority.get("accepted_a0_base_commit"), HEX40, label="accepted A0 base commit"
    )
    _expect_hex(
        authority.get("active_authority_commit"), HEX40, label="active authority commit"
    )
    _expect_exact(
        authority.get("active_amendment_decision_ids"),
        ["V3-DEC-017", "V3-DEC-018"],
        label="active decision set",
    )
    for group_name in ("fixed_authority_files", "implementation_ledger_files"):
        group = authority.get(group_name)
        if not isinstance(group, list) or not group:
            raise BindingError(f"{group_name} is absent")
        for item in group:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise BindingError(f"invalid {group_name} entry")
            _expect_hex(item.get("sha256"), HEX64, label=f"{group_name}:{item.get('path')}")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise BindingError("runtime section is absent")
    run_root = Path(str(runtime.get("run_root", "")))
    allowed_prepared_root = Path(str(runtime.get("allowed_prepared_root", "")))
    if not run_root.is_absolute() or not allowed_prepared_root.is_absolute():
        raise BindingError("runtime paths must be absolute")
    if "/runs/A1/" not in str(run_root):
        raise BindingError("runtime root is not the frozen A1 run")
    source = runtime.get("source_artifact")
    if not isinstance(source, dict) or "/" in str(source.get("name", "")):
        raise BindingError("source artifact must be one exact run-root member")
    _expect_hex(source.get("sha256"), HEX64, label="source artifact SHA-256")
    for mutable_name in MUTABLE_NAMES:
        spec = runtime.get("predecessor_mutables", {}).get(mutable_name)
        if not isinstance(spec, dict):
            raise BindingError(f"predecessor mutable spec missing: {mutable_name}")
        _expect_hex(spec.get("sha256"), HEX64, label=f"predecessor {mutable_name} SHA-256")
        snapshot_name = spec.get("snapshot_name")
        if not isinstance(snapshot_name, str) or "/" in snapshot_name:
            raise BindingError(f"invalid snapshot name: {mutable_name}")
    _expect_exact(
        runtime.get("mutable_publish_order"), list(MUTABLE_NAMES), label="mutable order"
    )
    _expect_exact(runtime.get("predecessor_manifest_output_count"), 88, label="old outputs")
    _expect_exact(runtime.get("successor_manifest_output_count"), 93, label="new outputs")

    if _contains_unresolved(config):
        raise BindingError("runtime-sync config still contains UNKNOWN/PENDING tokens")


def load_bound_config(config_path: Path, *, production: bool) -> tuple[dict[str, Any], bytes]:
    lexical_path = lexical_absolute(config_path)
    if production and lexical_path != PRODUCTION_CONFIG_PATH:
        raise BindingError(f"production config path must be exactly {PRODUCTION_CONFIG_PATH}")
    payload = read_regular_path(lexical_path)
    config = load_json(payload, label=str(lexical_path))
    validate_bound_config(config)
    return config, payload


def expected_unknown_i_config(bound_config: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(bound_config)
    expected["implementation_binding"]["status"] = UNKNOWN
    expected["implementation_binding"]["implementation_commit"] = UNKNOWN
    return expected


def _run_git(repo_root: Path, *arguments: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run(
            [GIT_BINARY, "-C", str(repo_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuthorityError(f"read-only git command failed to start: {arguments!r}") from exc
    if result.returncode not in allowed_returncodes:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuthorityError(
            f"read-only git command failed: {arguments!r} rc={result.returncode} stderr={message!r}"
        )
    return result.stdout


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{relative}")


def _paths_changed_by_commit(repo_root: Path, commit: str) -> list[str]:
    return sorted(
        _run_git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        .decode("utf-8")
        .splitlines()
    )


def audit_repo_authority(
    repo_root: Path, config: dict[str, Any], config_payload: bytes
) -> dict[str, Any]:
    """Prove exact aeecf0f -> I -> config-only B/HEAD/origin authority."""

    if lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT:
        raise AuthorityError(f"repository root must be exactly {PRODUCTION_REPO_ROOT}")
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise AuthorityError("production repository root is not a regular directory")
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    implementation = binding["implementation_commit"]
    base = binding["base_commit"]
    branch = authority["branch"]

    observed_branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    upstream_name = _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip()
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    origin_ref = f"refs/remotes/origin/{branch}"
    origin_head = _run_git(repo_root, "rev-parse", "--verify", origin_ref).decode().strip()
    worktree = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    binding_parent = _run_git(repo_root, "rev-parse", f"{head}^").decode().strip()
    implementation_parent = _run_git(repo_root, "rev-parse", f"{implementation}^").decode().strip()

    _expect_exact(observed_branch, branch, label="authority branch")
    _expect_exact(upstream_name, f"origin/{branch}", label="upstream name")
    _expect_exact(upstream, head, label="upstream HEAD")
    _expect_exact(origin_head, head, label="explicit origin branch HEAD")
    _expect_exact(binding_parent, implementation, label="B direct parent")
    _expect_exact(implementation_parent, base, label="I direct parent")
    if worktree != b"":
        raise AuthorityError("production repository worktree or index is not clean")

    for ancestor, descendant, label in (
        (authority["accepted_a0_base_commit"], authority["active_authority_commit"], "A0->active"),
        (authority["active_authority_commit"], base, "active->base"),
        (base, implementation, "base->I"),
        (implementation, head, "I->B"),
    ):
        _run_git(repo_root, "merge-base", "--is-ancestor", ancestor, descendant)

    _expect_exact(
        _paths_changed_by_commit(repo_root, implementation),
        authority["implementation_commit_exact_changed_paths"],
        label="I exact changed paths",
    )
    _expect_exact(
        _paths_changed_by_commit(repo_root, head),
        authority["binding_commit_exact_changed_paths"],
        label="B exact changed paths",
    )

    bound_blob = _git_blob(repo_root, head, CONFIG_REPO_PATH)
    if bound_blob != config_payload:
        raise AuthorityError("working config bytes do not equal the B commit blob")
    unknown_blob = _git_blob(repo_root, implementation, CONFIG_REPO_PATH)
    unknown_config = load_json(unknown_blob, label="I runtime-sync config blob")
    if unknown_config != expected_unknown_i_config(config):
        raise AuthorityError("I config is not the exact UNKNOWN form of B")

    expected_hashes = {
        binding["implementation_script_path"]: binding["implementation_script_sha256"],
        binding["implementation_test_path"]: binding["implementation_test_sha256"],
    }
    expected_hashes.update(
        {item["path"]: item["sha256"] for item in authority["implementation_ledger_files"]}
    )
    for relative, digest in expected_hashes.items():
        if sha256(_git_blob(repo_root, implementation, relative)) != digest:
            raise AuthorityError(f"I implementation blob drift: {relative}")
        if sha256(_git_blob(repo_root, head, relative)) != digest:
            raise AuthorityError(f"B unexpectedly changes implementation blob: {relative}")

    for item in authority["fixed_authority_files"]:
        relative = item["path"]
        digest = item["sha256"]
        if sha256(_git_blob(repo_root, head, relative)) != digest:
            raise AuthorityError(f"fixed authority blob drift: {relative}")
        if sha256(read_regular_path(repo_root / relative)) != digest:
            raise AuthorityError(f"working fixed authority file drift: {relative}")

    return {
        "status": "PASS_EXACT_AEECF0F_TO_I_TO_CONFIG_ONLY_B",
        "branch": branch,
        "base_commit": base,
        "implementation_commit": implementation,
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin_head,
        "head_equals_origin_branch_head": True,
        "implementation_parent_is_fixed_base": True,
        "binding_parent_is_implementation": True,
        "binding_commit_is_config_only": True,
        "worktree_and_index_clean": True,
        "config_sha256": sha256(config_payload),
        "implementation_changed_paths": authority["implementation_commit_exact_changed_paths"],
        "binding_changed_paths": authority["binding_commit_exact_changed_paths"],
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value
    ) is None:
        raise RuntimeSyncError("recorded_at must be second-resolution RFC3339 with +08:00")
    try:
        current = datetime.fromisoformat(value)
        predecessor = datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("recorded_at or predecessor timestamp is invalid") from exc
    if current.utcoffset() != timedelta(hours=8):
        raise RuntimeSyncError("recorded_at offset must be +08:00")
    if current <= predecessor:
        raise RuntimeSyncError("recorded_at must be strictly after A1-EVT-035")
    # Guard accidental workstation-clock dates far outside this frozen successor window.
    if current.astimezone(timezone.utc).date().isoformat() not in {"2026-08-10", "2026-08-11"}:
        raise RuntimeSyncError("recorded_at is outside the frozen EVT-036 publication window")


def validate_source_artifact(
    payload: bytes, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = config["runtime"]
    source_spec = runtime["source_artifact"]
    if len(payload) != source_spec["bytes"] or sha256(payload) != source_spec["sha256"]:
        raise PublicationError("source aggregate artifact bytes or SHA-256 drift")
    source = load_json(payload, label=source_spec["name"])
    truth = config["source_truth"]
    for key in (
        "schema_version",
        "protocol_id",
        "dataset_id",
        "recorded_at_utc",
        "outcome",
        "ready_for_study_qualification",
    ):
        _expect_exact(source.get(key), truth[key], label=f"source artifact {key}")
    _expect_exact(source.get("blockers"), truth["blockers"], label="source blockers")
    _expect_exact(source.get("counters"), truth["zero_counters"], label="source counters")
    _expect_exact(source.get("gate_truth"), truth["gate_truth"], label="source gate truth")

    authority_audit = source.get("authority_audit", {})
    _expect_exact(
        authority_audit.get("status"), truth["authority_status"], label="source authority status"
    )
    binding = config["implementation_binding"]
    repo_authority = config["repository_authority"]
    expected_source_authority = {
        "accepted_a0_base_commit": repo_authority["accepted_a0_base_commit"],
        "active_authority_commit": repo_authority["active_authority_commit"],
        "active_amendment_decision_ids": repo_authority["active_amendment_decision_ids"],
        "implementation_commit": "d10a42a564ecac2af048b39c05cbc863ebdacd02",
        "binding_commit": binding["base_commit"],
    }
    for key, value in expected_source_authority.items():
        _expect_exact(authority_audit.get(key), value, label=f"source authority {key}")

    environment = source.get("environment_audit", {})
    _expect_exact(
        environment.get("status"), truth["environment_status"], label="source environment status"
    )
    expected_environment = truth["environment_truth"]
    _expect_exact(
        environment.get("claim_absent"), expected_environment["claim_absent"], label="claim absent"
    )
    _expect_exact(
        environment.get("failure_absent"),
        expected_environment["failure_absent"],
        label="failure absent",
    )
    _expect_exact(
        environment.get("output_absent"),
        expected_environment["output_absent_before_publication"],
        label="prepublication output absent",
    )

    inventory = source.get("inventory_audit", {})
    _expect_exact(
        inventory.get("status"), truth["inventory_status"], label="source inventory status"
    )
    for key, value in truth["inventory_counts"].items():
        _expect_exact(inventory.get(key), value, label=f"source inventory {key}")
    _expect_exact(inventory.get("payload_open_count"), 0, label="source payload open count")
    _expect_exact(inventory.get("manifest_open_count"), 0, label="source manifest open count")
    _expect_exact(inventory.get("payload_hash_count"), 0, label="source payload hash count")
    _expect_exact(
        inventory.get("scientific_processing_count"), 0, label="source scientific processing count"
    )
    _expect_exact(
        inventory.get("hash_reverification"),
        "NOT_RUN_STOP_BEFORE_DATA",
        label="source hash-reverification boundary",
    )

    external = source.get("external_evidence_audit", {})
    for key, value in truth["external_evidence_truth"].items():
        _expect_exact(external.get(key), value, label=f"external evidence {key}")
    historical = source.get("historical_r4_closure", {})
    _expect_exact(
        historical.get("reference_only_not_reopened"), True, label="historical R4 reference-only"
    )
    _expect_exact(
        historical.get("rerun_is_qualification_path"), False, label="historical R4 rerun boundary"
    )

    selected = {
        "schema_version": source["schema_version"],
        "protocol_id": source["protocol_id"],
        "recorded_at_utc": source["recorded_at_utc"],
        "outcome": source["outcome"],
        "ready_for_study_qualification": source["ready_for_study_qualification"],
        "authority_status": authority_audit["status"],
        "environment_status": environment["status"],
        "inventory_status": inventory["status"],
        "inventory_counts": copy.deepcopy(truth["inventory_counts"]),
        "blockers": list(source["blockers"]),
        "zero_counters": copy.deepcopy(source["counters"]),
        "gate_truth": copy.deepcopy(source["gate_truth"]),
        "external_evidence_truth": copy.deepcopy(truth["external_evidence_truth"]),
        "prepublication_claim_absent": environment["claim_absent"],
        "prepublication_failure_absent": environment["failure_absent"],
        "prepublication_output_absent": environment["output_absent"],
        "artifact_producer": {
            "implementation_commit": authority_audit["implementation_commit"],
            "binding_commit": authority_audit["binding_commit"],
        },
    }
    return source, selected


def _validate_predecessor_objects(
    status: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    runtime = config["runtime"]
    if len(events) != runtime["predecessor_event_count"]:
        raise PublicationError("predecessor event count drift")
    if not events or events[-1].get("event_id") != runtime["predecessor_event_id"]:
        raise PublicationError("predecessor tail is not exactly A1-EVT-035")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != runtime["predecessor_manifest_output_count"]:
        raise PublicationError("predecessor manifest output count is not exactly 88")
    for key, expected in config["successor_invariants"].items():
        _expect_exact(status.get(key), expected, label=f"predecessor STATUS {key}")
    for key in ("run_status", "evidence_status", "claim_status"):
        _expect_exact(
            manifest.get(key), config["successor_invariants"][key], label=f"predecessor manifest {key}"
        )
    _expect_exact(
        manifest.get("v3_contract_sha256"),
        "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
        label="predecessor contract hash",
    )
    _expect_exact(
        manifest.get("active_amendment_decision_ids"),
        ["V3-DEC-017", "V3-DEC-018"],
        label="predecessor decisions",
    )
    source_path = str(Path(runtime["run_root"]) / runtime["source_artifact"]["name"])
    sync_path = str(Path(runtime["run_root"]) / runtime["sync_name"])
    if any(
        isinstance(record, dict) and record.get("absolute_path") in {source_path, sync_path}
        for record in outputs
    ):
        raise PublicationError("EVT-036 source or sync is already registered in predecessor outputs")


def read_exact_predecessor(
    run_fd: int, config: dict[str, Any]
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    runtime = config["runtime"]
    payloads: dict[str, bytes] = {}
    for name in MUTABLE_NAMES:
        payload = read_regular_at(run_fd, name)
        spec = runtime["predecessor_mutables"][name]
        if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
            raise PublicationError(f"exact predecessor mutable drift: {name}")
        payloads[name] = payload
    status = load_json(payloads["STATUS.json"], label="predecessor STATUS.json")
    manifest = load_json(payloads["RUN_MANIFEST.json"], label="predecessor RUN_MANIFEST.json")
    events = load_json_lines(payloads["EVENT_LOG.jsonl"], label="predecessor EVENT_LOG.jsonl")
    _validate_predecessor_objects(status, manifest, events, config)

    source_name = runtime["source_artifact"]["name"]
    source_payload = read_regular_at(run_fd, source_name, require_single_link=True)
    _source, selected_source = validate_source_artifact(source_payload, config)
    return payloads, status, manifest, events, selected_source


def output_record(artifact_type: str, absolute_path: Path, digest: str) -> dict[str, str]:
    return {
        "artifact_type": artifact_type,
        "absolute_path": str(absolute_path),
        "sha256": digest,
        "status": "COMPLETE",
    }


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
    run_root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    predecessor = runtime["predecessor_mutables"]
    source = runtime["source_artifact"]
    return [
        output_record(
            "GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RECORD",
            run_root / source["name"],
            source["sha256"],
        ),
        output_record(
            "A1_STATUS_PRE_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots["STATUS.json"],
            predecessor["STATUS.json"]["sha256"],
        ),
        output_record(
            "A1_RUN_MANIFEST_PRE_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots["RUN_MANIFEST.json"],
            predecessor["RUN_MANIFEST.json"]["sha256"],
        ),
        output_record(
            "A1_EVENT_LOG_PRE_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC_SNAPSHOT",
            run_root / snapshots["EVENT_LOG.jsonl"],
            predecessor["EVENT_LOG.jsonl"]["sha256"],
        ),
        output_record(
            "A1_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC_V1",
            run_root / runtime["sync_name"],
            sync_digest,
        ),
    ]


def build_successors(
    config: dict[str, Any],
    predecessor_payloads: dict[str, bytes],
    predecessor_status: dict[str, Any],
    predecessor_manifest: dict[str, Any],
    predecessor_events: list[dict[str, Any]],
    selected_source: dict[str, Any],
    authority_audit: dict[str, Any],
    recorded_at: str,
) -> dict[str, bytes]:
    runtime = config["runtime"]
    binding = config["implementation_binding"]
    repo_authority = config["repository_authority"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    run_root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    snapshot_lineage = [
        {
            "source_mutable_path": str(run_root / name),
            "snapshot_path": str(run_root / snapshots[name]),
            "bytes": len(predecessor_payloads[name]),
            "sha256": sha256(predecessor_payloads[name]),
        }
        for name in MUTABLE_NAMES
    ]

    sync_record = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC",
        "contract_id": config["contract_id"],
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        "phase_id": config["phase_id"],
        "dataset_id": config["dataset_id"],
        "recorded_at": recorded_at,
        "predecessor_runtime": {
            "runtime_root": str(run_root),
            "last_event_id": runtime["predecessor_event_id"],
            "manifest_output_count": runtime["predecessor_manifest_output_count"],
            "immutable_pre_sync_snapshots": snapshot_lineage,
        },
        "outer_runtime_historical_semantics": {
            "code_commit": predecessor_manifest.get("code_commit"),
            "active_authority_commit": predecessor_manifest.get("active_authority_commit"),
            "code_commit_rewritten": False,
            "active_authority_commit_rewritten": False,
            "meaning": "HISTORICAL_EXECUTION_AND_PRIOR_RUNTIME_AUTHORITY_IDENTITY",
        },
        "current_contract_authority": {
            "accepted_a0_base_commit": repo_authority["accepted_a0_base_commit"],
            "active_authority_commit": repo_authority["active_authority_commit"],
            "active_amendment_decision_ids": repo_authority["active_amendment_decision_ids"],
            "contract_sha256": "cbac4c3dcba8f1b8df95d8edad52d19e3c126d1c865d0cc423537c754cc90982",
            "data_role_registry_sha256": "746439ef5d88d8167176d19e9c675746fdc78984a66f6f123f77f6ec49523030",
            "decision_log_sha256": "a5b041fab24d9a4309603a085fa3fcab936d69a899285bfa752689a2ee5fd4fd",
        },
        "runtime_sync_publisher_authority": {
            "status": authority_audit["status"],
            "branch": repo_authority["branch"],
            "base_commit": binding["base_commit"],
            "implementation_commit": binding["implementation_commit"],
            "binding_commit": authority_audit["binding_commit"],
            "head_commit": authority_audit["head_commit"],
            "origin_branch_head_commit": authority_audit["origin_branch_head_commit"],
            "config_repo_path": CONFIG_REPO_PATH,
            "config_sha256": authority_audit["config_sha256"],
            "script_repo_path": binding["implementation_script_path"],
            "script_sha256": binding["implementation_script_sha256"],
            "test_repo_path": binding["implementation_test_path"],
            "test_sha256": binding["implementation_test_sha256"],
            "implementation_ledger_file_sha256": {
                item["path"]: item["sha256"]
                for item in repo_authority["implementation_ledger_files"]
            },
            "implementation_parent_is_fixed_base": True,
            "binding_parent_is_implementation": True,
            "binding_commit_is_config_only": True,
            "head_equals_origin_branch_head": True,
            "worktree_and_index_clean": True,
        },
        "source_preflight_artifact": {
            "path": str(run_root / runtime["source_artifact"]["name"]),
            "bytes": runtime["source_artifact"]["bytes"],
            "sha256": runtime["source_artifact"]["sha256"],
            "recorded_at_utc": selected_source["recorded_at_utc"],
            "outcome": selected_source["outcome"],
            "ready_for_study_qualification": False,
            "authority_status": selected_source["authority_status"],
            "environment_status": selected_source["environment_status"],
            "inventory_status": selected_source["inventory_status"],
            "artifact_producer": selected_source["artifact_producer"],
            "prepublication_claim_absent": selected_source["prepublication_claim_absent"],
            "prepublication_failure_absent": selected_source["prepublication_failure_absent"],
            "prepublication_output_absent": selected_source["prepublication_output_absent"],
            "body_embedded": False,
        },
        "scientific_blockers": {
            "count": len(selected_source["blockers"]),
            "exact": selected_source["blockers"],
        },
        "external_evidence_boundary": selected_source["external_evidence_truth"],
        "runtime_sync_scope": {
            "metadata_only_aggregate": True,
            "aggregate_preflight_record_open_required": True,
            "aggregate_preflight_record_body_embedded": False,
            "data_payload_open_count": 0,
            "data_manifest_open_count": 0,
            "data_payload_hash_count": 0,
            "scientific_processing_count": 0,
            "qualifier_execution_count": 0,
            "training_run_count": 0,
            "model_selection_run_count": 0,
            "sequence_embedded": False,
            "barcode_embedded": False,
            "label_value_embedded": False,
            "gpu_used": False,
            "restricted_or_sealed_path_accessed": False,
            "restricted_or_sealed_payload_contact": False,
        },
        "a1_gate_snapshot": {
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
            "training_started": False,
            "training_authorized": False,
            "model_selection_allowed": False,
            "next_phase_authorized": False,
        },
        "claim_boundary": {
            "established": [
                "EXACT_AGGREGATE_PREFLIGHT_ARTIFACT_IDENTITY",
                "REPOSITORY_AUTHORITY_PREFLIGHT_PASS",
                "METADATA_INVENTORY_PREFLIGHT_PASS",
                "ELEVEN_FAIL_CLOSED_SCIENTIFIC_BLOCKERS_RECORDED",
            ],
            "not_established": [
                "DATA_PAYLOAD_INTEGRITY_REVERIFICATION",
                "PAPER_NATIVE_METHOD_REPRODUCTION",
                "STUDY_QUALIFICATION",
                "CANONICAL_INTERVENTION_RECORDS",
                "TRAINING_AUTHORIZATION",
                "MODEL_SELECTION_AUTHORIZATION",
                "NEXT_PHASE_AUTHORIZATION",
                "MODEL_PERFORMANCE",
                "SCIENTIFIC_CLAIM",
            ],
        },
        "hash_linkage": {
            "direction": "PREDECESSOR_SNAPSHOTS_TO_SYNC_RECORD_TO_MUTABLE_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successor_status_event_and_manifest_reference_sync_record_sha256": True,
        },
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload = json_bytes(sync_record)
    sync_digest = sha256(sync_payload)

    status = dict(predecessor_status)
    status.update(
        {
            "updated_at": recorded_at,
            "gse149487_stop_before_data_preflight_runtime_sync_status": "SYNCED_EVT_036",
            "gse149487_stop_before_data_preflight_runtime_sync_record_sha256": sync_digest,
            "gse149487_stop_before_data_preflight_artifact_path": str(
                run_root / runtime["source_artifact"]["name"]
            ),
            "gse149487_stop_before_data_preflight_artifact_bytes": runtime["source_artifact"]["bytes"],
            "gse149487_stop_before_data_preflight_artifact_sha256": runtime["source_artifact"]["sha256"],
            "gse149487_stop_before_data_preflight_outcome": selected_source["outcome"],
            "gse149487_stop_before_data_preflight_authority_status": selected_source[
                "authority_status"
            ],
            "gse149487_stop_before_data_preflight_environment_status": selected_source[
                "environment_status"
            ],
            "gse149487_stop_before_data_preflight_inventory_status": selected_source[
                "inventory_status"
            ],
            "gse149487_stop_before_data_preflight_scientific_blocker_count": len(
                selected_source["blockers"]
            ),
            "gse149487_stop_before_data_preflight_payload_asset_count": selected_source[
                "inventory_counts"
            ]["payload_asset_count"],
            "gse149487_stop_before_data_preflight_payload_open_count": 0,
            "gse149487_stop_before_data_preflight_manifest_open_count": 0,
            "gse149487_stop_before_data_preflight_payload_hash_count": 0,
            "gse149487_stop_before_data_preflight_scientific_processing_count": 0,
            "gse149487_stop_before_data_preflight_qualifier_execution_count": 0,
            "gse149487_stop_before_data_preflight_training_run_count": 0,
            "gse149487_stop_before_data_preflight_model_selection_run_count": 0,
            "gse149487_stop_before_data_preflight_ordinary_study_contribution": 0,
            "gse149487_stop_before_data_preflight_a1_study_contribution": 0,
            "gse149487_stop_before_data_preflight_true_a2_study_contribution": 0,
            "gse149487_stop_before_data_preflight_canonical_record_count": 0,
            "gse149487_stop_before_data_preflight_ready_for_study_qualification": False,
            "gse149487_stop_before_data_preflight_qualified": False,
            "gse149487_stop_before_data_preflight_training_allowed": False,
            "gse149487_stop_before_data_preflight_model_selection_allowed": False,
            "gse149487_stop_before_data_preflight_next_phase_authorized": False,
            "gse149487_stop_before_data_preflight_artifact_producer_implementation_commit": selected_source[
                "artifact_producer"
            ]["implementation_commit"],
            "gse149487_stop_before_data_preflight_artifact_producer_binding_commit": selected_source[
                "artifact_producer"
            ]["binding_commit"],
            "gse149487_stop_before_data_preflight_runtime_sync_repo_implementation_commit": binding[
                "implementation_commit"
            ],
            "gse149487_stop_before_data_preflight_runtime_sync_repo_binding_commit": authority_audit[
                "binding_commit"
            ],
            "gse149487_stop_before_data_preflight_metadata_only": True,
            "gse149487_stop_before_data_preflight_gpu_used": False,
            "gse149487_stop_before_data_preflight_restricted_or_sealed_path_accessed": False,
        }
    )
    status_payload = json_bytes(status)

    event = {
        "event_id": config["event_id"],
        "at": recorded_at,
        "phase_id": config["phase_id"],
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5",
        "event": config["event_name"],
        "sync_record": runtime["sync_name"],
        "sync_record_sha256": sync_digest,
        "artifact_path": str(run_root / runtime["source_artifact"]["name"]),
        "artifact_bytes": runtime["source_artifact"]["bytes"],
        "artifact_sha256": runtime["source_artifact"]["sha256"],
        "artifact_outcome": selected_source["outcome"],
        "authority_status": selected_source["authority_status"],
        "environment_status": selected_source["environment_status"],
        "inventory_status": selected_source["inventory_status"],
        "scientific_blocker_count": len(selected_source["blockers"]),
        "artifact_producer_implementation_commit": selected_source["artifact_producer"][
            "implementation_commit"
        ],
        "artifact_producer_binding_commit": selected_source["artifact_producer"]["binding_commit"],
        "runtime_sync_implementation_commit": binding["implementation_commit"],
        "runtime_sync_binding_commit": authority_audit["binding_commit"],
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "qualified_independent_ordinary_studies": 0,
        "qualified_a1_studies": 0,
        "qualified_a2_dense_studies": 0,
        "metadata_only_qualification_count": 0,
        "canonical_intervention_record_count": 0,
        "qualified": False,
        "training_started": False,
        "training_authorized": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "payload_open_count": 0,
        "manifest_open_count": 0,
        "payload_hash_count": 0,
        "scientific_processing_count": 0,
        "qualifier_execution_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
        "pre_sync_snapshot_count": 3,
        "manifest_output_count_before": runtime["predecessor_manifest_output_count"],
        "manifest_output_count_after": runtime["successor_manifest_output_count"],
        "aggregate_preflight_record_open_required": True,
        "artifact_body_copied_into_event": False,
        "artifact_body_copied_into_runtime_sync": False,
        "data_payload_body_copied_into_event": False,
        "data_payload_body_copied_into_runtime_sync": False,
        "data_manifest_body_copied_into_event": False,
        "data_manifest_body_copied_into_runtime_sync": False,
        "sequence_copied_into_event": False,
        "barcode_copied_into_event": False,
        "label_value_copied_into_event": False,
        "gpu_used": False,
        "sealed_evaluation_count": 0,
        "restricted_or_sealed_path_accessed": False,
        "restricted_or_sealed_payload_contact": False,
        "detail": (
            "Registered the exact metadata-only GSE149487 stop-before-data preflight aggregate. "
            "Authority, environment, and descriptor-bound inventory checks passed, but eleven frozen "
            "scientific blockers remain open. Contributions remain 0/0/0, metadata-only qualification "
            "and canonical records remain zero, and no payload/manifest opening, payload hashing, "
            "scientific processing, qualifier execution, training, model selection, next phase, GPU, "
            "restricted/sealed access, model-performance claim, or scientific claim is authorized."
        ),
    }
    event_payload = predecessor_payloads["EVENT_LOG.jsonl"] + compact_json_line(event)

    manifest = dict(predecessor_manifest)
    manifest["gse149487_stop_before_data_preflight_runtime_sync_record_sha256"] = sync_digest
    outputs = list(predecessor_manifest["outputs"])
    outputs.extend(expected_output_delta(config, sync_digest))
    manifest["outputs"] = outputs
    manifest_payload = json_bytes(manifest)

    artifacts = {
        snapshots["STATUS.json"]: predecessor_payloads["STATUS.json"],
        snapshots["RUN_MANIFEST.json"]: predecessor_payloads["RUN_MANIFEST.json"],
        snapshots["EVENT_LOG.jsonl"]: predecessor_payloads["EVENT_LOG.jsonl"],
        runtime["sync_name"]: sync_payload,
        "STATUS.json": status_payload,
        "RUN_MANIFEST.json": manifest_payload,
        "EVENT_LOG.jsonl": event_payload,
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
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {runtime["sync_name"]}
    if set(artifacts) != expected_names:
        raise RuntimeSyncError("prepared runtime member set is not the exact seven-file closure")
    for mutable_name, snapshot_name in snapshots.items():
        if artifacts[snapshot_name] != predecessor_payloads[mutable_name]:
            raise RuntimeSyncError(f"predecessor snapshot is not byte-exact: {snapshot_name}")

    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label=runtime["sync_name"])
    _expect_exact(
        sync.get("record_type"),
        "ROUTE_A_V3_A1_GSE149487_STOP_BEFORE_DATA_PREFLIGHT_RUNTIME_SYNC",
        label="sync record type",
    )
    _expect_exact(
        sync.get("self_hash"),
        "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
        label="sync self-hash policy",
    )
    _expect_exact(
        sync.get("hash_linkage"),
        {
            "direction": "PREDECESSOR_SNAPSHOTS_TO_SYNC_RECORD_TO_MUTABLE_SUCCESSORS",
            "sync_record_references_successor_hashes": False,
            "successor_status_event_and_manifest_reference_sync_record_sha256": True,
        },
        label="one-way sync hash linkage",
    )
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode("ascii") in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("mutable successor digest leaked into the sync record")
    for name in MUTABLE_NAMES:
        if sync_digest.encode("ascii") not in artifacts[name]:
            raise RuntimeSyncError(f"successor does not bind the sync record: {name}")

    status = load_json(artifacts["STATUS.json"], label="successor STATUS.json")
    for key, value in predecessor_status.items():
        if key != "updated_at" and status.get(key) != value:
            raise RuntimeSyncError(f"predecessor STATUS field was rewritten: {key}")
    allowed_new_status = {
        key for key in status if key.startswith("gse149487_stop_before_data_preflight_")
    }
    unexpected_new_status = set(status) - set(predecessor_status) - allowed_new_status
    if unexpected_new_status:
        raise RuntimeSyncError(f"unexpected successor STATUS keys: {sorted(unexpected_new_status)!r}")
    if status.get("updated_at") != sync.get("recorded_at"):
        raise RuntimeSyncError("successor STATUS timestamp does not match sync record")
    for key, expected in config["successor_invariants"].items():
        _expect_exact(status.get(key), expected, label=f"successor STATUS {key}")
    _expect_exact(
        status.get("gse149487_stop_before_data_preflight_runtime_sync_record_sha256"),
        sync_digest,
        label="STATUS sync binding",
    )
    zero_status_fields = (
        "payload_open_count",
        "manifest_open_count",
        "payload_hash_count",
        "scientific_processing_count",
        "qualifier_execution_count",
        "training_run_count",
        "model_selection_run_count",
        "ordinary_study_contribution",
        "a1_study_contribution",
        "true_a2_study_contribution",
        "canonical_record_count",
    )
    for suffix in zero_status_fields:
        _expect_exact(
            status.get(f"gse149487_stop_before_data_preflight_{suffix}"),
            0,
            label=f"STATUS zero invariant {suffix}",
        )
    for suffix in (
        "ready_for_study_qualification",
        "qualified",
        "training_allowed",
        "model_selection_allowed",
        "next_phase_authorized",
        "gpu_used",
        "restricted_or_sealed_path_accessed",
    ):
        _expect_exact(
            status.get(f"gse149487_stop_before_data_preflight_{suffix}"),
            False,
            label=f"STATUS false invariant {suffix}",
        )

    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor RUN_MANIFEST.json")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError(f"predecessor RUN_MANIFEST field was rewritten: {key}")
    allowed_manifest_key = "gse149487_stop_before_data_preflight_runtime_sync_record_sha256"
    if set(manifest) - set(predecessor_manifest) != {allowed_manifest_key}:
        raise RuntimeSyncError("RUN_MANIFEST top-level delta is not the one-key sync binding")
    _expect_exact(manifest.get(allowed_manifest_key), sync_digest, label="manifest sync binding")
    old_outputs = predecessor_manifest.get("outputs")
    outputs = manifest.get("outputs")
    if not isinstance(old_outputs, list) or not isinstance(outputs, list):
        raise RuntimeSyncError("RUN_MANIFEST outputs are not arrays")
    if len(old_outputs) != 88 or len(outputs) != 93:
        raise RuntimeSyncError("RUN_MANIFEST output transition is not exactly 88 -> 93")
    if outputs[:88] != old_outputs or outputs[88:] != expected_output_delta(config, sync_digest):
        raise RuntimeSyncError("RUN_MANIFEST exact five-record output closure drift")
    if len({record.get("absolute_path") for record in outputs if isinstance(record, dict)}) != len(
        outputs
    ):
        raise RuntimeSyncError("RUN_MANIFEST contains duplicate output paths")

    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor EVENT_LOG.jsonl")
    old_events = load_json_lines(
        predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor EVENT_LOG.jsonl"
    )
    if len(events) != 36 or events[:-1] != old_events:
        raise RuntimeSyncError("EVENT_LOG is not an exact one-event append")
    if not artifacts["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("historical EVENT_LOG bytes changed")
    event = events[-1]
    _expect_exact(event.get("event_id"), "A1-EVT-036", label="successor event id")
    _expect_exact(event.get("event"), config["event_name"], label="successor event name")
    _expect_exact(event.get("sync_record_sha256"), sync_digest, label="event sync binding")
    exact_event_truth = {
        "artifact_outcome": "NOT_READY_FOR_STUDY_QUALIFICATION",
        "scientific_blocker_count": 11,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "qualified_independent_ordinary_studies": 0,
        "qualified_a1_studies": 0,
        "qualified_a2_dense_studies": 0,
        "metadata_only_qualification_count": 0,
        "canonical_intervention_record_count": 0,
        "qualified": False,
        "training_started": False,
        "training_authorized": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "payload_open_count": 0,
        "manifest_open_count": 0,
        "payload_hash_count": 0,
        "scientific_processing_count": 0,
        "qualifier_execution_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
        "pre_sync_snapshot_count": 3,
        "manifest_output_count_before": 88,
        "manifest_output_count_after": 93,
        "artifact_body_copied_into_event": False,
        "artifact_body_copied_into_runtime_sync": False,
        "data_payload_body_copied_into_event": False,
        "data_payload_body_copied_into_runtime_sync": False,
        "data_manifest_body_copied_into_event": False,
        "data_manifest_body_copied_into_runtime_sync": False,
        "sequence_copied_into_event": False,
        "barcode_copied_into_event": False,
        "label_value_copied_into_event": False,
        "gpu_used": False,
        "sealed_evaluation_count": 0,
        "restricted_or_sealed_path_accessed": False,
        "restricted_or_sealed_payload_contact": False,
    }
    for key, expected in exact_event_truth.items():
        _expect_exact(event.get(key), expected, label=f"EVT-036 {key}")

    generated_scope = sync_payload + compact_json_line(event)
    for token in runtime["forbidden_path_tokens"]:
        if token.encode("utf-8") in generated_scope:
            raise RuntimeSyncError(f"forbidden data/restricted token entered runtime aggregate: {token}")


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
    for token in runtime["forbidden_path_tokens"]:
        if token in text:
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
        predecessor, status, manifest, events, selected_source = read_exact_predecessor(
            run_fd, config
        )
        artifacts = build_successors(
            config,
            predecessor,
            status,
            manifest,
            events,
            selected_source,
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
        "manifest_output_transition": "88_TO_93",
        "source_artifact_state": "EXISTING_SOURCE_EXACT_REUSED",
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
    _validate_predecessor_objects(status, manifest, events, config)
    return payloads, status, manifest, events


def validate_prepared_against_context(
    prepared: dict[str, bytes],
    config: dict[str, Any],
    authority: dict[str, Any],
    selected_source: dict[str, Any],
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
        selected_source,
        authority,
        recorded_at,
    )
    if prepared != expected:
        differing = sorted(name for name in expected if prepared.get(name) != expected[name])
        raise PublicationError(f"prepared runtime bytes do not match current authority: {differing!r}")


def classify_target(
    run_fd: int, prepared: dict[str, bytes], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Freshly classify every source/immutable/mutable under the held run lock."""

    runtime = config["runtime"]
    source_spec = runtime["source_artifact"]
    source_payload = read_regular_at(run_fd, source_spec["name"], require_single_link=True)
    _source, selected_source = validate_source_artifact(source_payload, config)

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
        existing = read_optional_regular_at(run_fd, name)
        if existing is None:
            immutable_states[name] = "ABSENT"
        elif existing == prepared[name]:
            immutable_states[name] = "EXISTING_EXACT"
        else:
            raise PublicationError(f"existing immutable artifact differs: {name}")
    return mutable_states, immutable_states, selected_source


def _invoke_fault(fault_injector: FaultInjector | None, point: str) -> None:
    if fault_injector is not None:
        fault_injector(point)


def _warning(point: str, exc: BaseException) -> dict[str, str]:
    return {"point": point, "error_type": type(exc).__name__, "message": str(exc)}


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
        return {
            "state": "EXISTING_EXACT_REUSED",
            "committed_by_this_call": False,
            "accepted": True,
            "warnings": [],
        }

    temporary_name = f".evt036.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
            return {
                "state": "EXISTING_EXACT_REUSED",
                "committed_by_this_call": False,
                "accepted": True,
                "warnings": [],
            }
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
        return {
            "state": "EXISTING_NEW_EXACT_REUSED",
            "committed_by_this_call": False,
            "accepted": True,
            "warnings": [],
        }
    if current != old_payload:
        raise PublicationError(f"unexpected mutable predecessor: {name}")

    temporary_name = f".evt036.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
        mutable, immutable, selected_source = classify_target(run_fd, prepared, config)
        validate_prepared_against_context(prepared, config, authority, selected_source)
    if lock_cleanup_warnings:
        raise PublicationError(f"run-lock cleanup failed after validation: {lock_cleanup_warnings!r}")
    return {
        "status": "VALIDATED_NOT_PUBLISHED",
        "run_root": str(run_root),
        "prepared_directory": str(prepared_path),
        "source_artifact_state": "EXISTING_SOURCE_EXACT_REUSED",
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
        config["runtime"]["source_artifact"]["name"]: {
            "state": "EXISTING_SOURCE_EXACT_REUSED",
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
        mutable_preflight, immutable_preflight, selected_source = classify_target(
            run_fd, prepared, config
        )
        validate_prepared_against_context(prepared, config, authority, selected_source)
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

            final_mutable, final_immutable, final_source = classify_target(run_fd, prepared, config)
            validate_prepared_against_context(prepared, config, authority, final_source)
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
        "manifest_output_transition": "88_TO_93",
        "source_artifact_state": "EXISTING_SOURCE_EXACT_REUSED",
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
        "PARTIAL_STATE_REQUIRES_IDEMPOTENT_RETRY",
    }:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
