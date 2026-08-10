#!/usr/bin/env python3
"""Prepare, validate, and publish the fail-closed GSE200304 EVT-037 runtime sync.

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


CONFIG_REPO_PATH = "configs/route_a_v3_gse200304_published_endpoint_runtime_sync_v1.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/gse200304_published_endpoint_runtime_sync.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse200304_published_endpoint_runtime_sync.py"
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


def compiled_core_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable config core used to compare I and B."""

    return {key: copy.deepcopy(value) for key, value in config.items() if key != "implementation_binding"}


def compiled_core_sha256(config: dict[str, Any]) -> str:
    return sha256(json.dumps(compiled_core_projection(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _expect_int(value: Any, expected: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise BindingError(f"{label} must be the exact integer {expected}")


def validate_bound_config(config: dict[str, Any]) -> None:
    """Reject unresolved or semantically drifting config before external access."""

    _expect_exact(config.get("schema_version"), "route_a_v3_gse200304_published_endpoint_runtime_sync.v1", label="runtime-sync schema")
    _expect_exact(config.get("protocol_id"), "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1", label="runtime-sync protocol")
    _expect_exact(config.get("event_id"), "A1-EVT-037", label="event id")
    _expect_exact(config.get("event_name"), "GSE200304_PUBLISHED_ENDPOINT_COMMITTED_ACCEPTED_SYNCED_GATE_UNCHANGED", label="event name")
    binding = config.get("implementation_binding")
    if not isinstance(binding, dict):
        raise BindingError("implementation_binding is absent")
    if binding.get("status") != "BOUND":
        raise BindingError("runtime-sync config is not BOUND")
    implementation = _expect_hex(binding.get("implementation_commit"), HEX40, label="implementation commit")
    _expect_hex(binding.get("implementation_script_sha256"), HEX64, label="implementation script SHA-256")
    _expect_hex(binding.get("implementation_test_sha256"), HEX64, label="implementation test SHA-256")
    _expect_exact(binding.get("implementation_script_path"), SCRIPT_REPO_PATH, label="script path")
    _expect_exact(binding.get("implementation_test_path"), TEST_REPO_PATH, label="test path")
    dynamic = [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ]
    _expect_exact(binding.get("unknown_to_bound_scalar_paths"), dynamic, label="binding scalar allowlist")
    _expect_exact(binding.get("compiled_core_sha256"), compiled_core_sha256(config), label="compiled core projection")

    authority = config.get("repository_authority")
    if not isinstance(authority, dict):
        raise BindingError("repository_authority is absent")
    _expect_exact(authority.get("production_repo_root"), str(PRODUCTION_REPO_ROOT), label="repository root")
    _expect_exact(authority.get("branch"), "routea-v3-a1-20260810", label="branch")
    ledger = _expect_hex(authority.get("ledger_commit"), HEX40, label="ledger commit")
    _expect_exact(ledger, "bdd30d50c04a565b68a1b33b5cb164e4eda3fa9f", label="ledger commit")
    _expect_exact(authority.get("ledger_commit_expected_parent"), "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189", label="ledger parent")
    _expect_exact(authority.get("implementation_commit_expected_parent"), ledger, label="implementation parent")
    if implementation == ledger:
        raise BindingError("implementation commit did not advance from ledger")
    _expect_exact(authority.get("implementation_commit_exact_changed_paths"), sorted([CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH]), label="I changed paths")
    _expect_exact(authority.get("binding_commit_exact_changed_paths"), [CONFIG_REPO_PATH], label="B changed paths")
    ledger_files = authority.get("ledger_files")
    if not isinstance(ledger_files, list) or [item.get("path") for item in ledger_files if isinstance(item, dict)] != [
        "docs/execution/route_a_v3_a1_interim.yaml",
        "docs/execution/route_a_v3_registry_manifest.json",
        "scripts/route_a_v3/validate_a0_bundle.py",
        "tests/route_a_v3/test_a0_integrity_guards.py",
    ]:
        raise BindingError("ledger file closure drift")
    for item in ledger_files:
        _expect_hex(item.get("sha256"), HEX64, label=f"ledger file {item.get('path')}")
    _expect_exact(authority.get("ledger_semantics"), {
        "node_id": "gse200304_published_endpoint_a1_v1",
        "runtime_sync_status": "PENDING_NO_EVT_037",
        "publication_state": "COMMITTED_ACCEPTED",
        "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED",
        "qualification_status": "BLOCKED_NOT_QUALIFIED",
    }, label="ledger semantics")
    qualifier = authority.get("external_qualifier_binding", {})
    _expect_exact(qualifier.get("implementation_commit"), "84fc6932de32fe0de8e5ddf540e14dee62a2b723", label="qualifier I")
    _expect_exact(qualifier.get("binding_commit"), "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189", label="qualifier B")
    for field in ("config_sha256", "script_sha256", "test_sha256"):
        _expect_hex(qualifier.get(field), HEX64, label=f"qualifier {field}")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise BindingError("runtime section is absent")
    run_root = Path(str(runtime.get("run_root", "")))
    artifact_root = Path(str(runtime.get("artifact_root", "")))
    prepared_root = Path(str(runtime.get("allowed_prepared_root", "")))
    if not all(path.is_absolute() for path in (run_root, artifact_root, prepared_root)):
        raise BindingError("all runtime roots must be absolute")
    if not is_strict_descendant(artifact_root, run_root):
        raise BindingError("artifact root must be a strict run-root descendant")
    _expect_int(runtime.get("predecessor_event_count"), 36, label="predecessor event count")
    _expect_int(runtime.get("successor_event_count"), 37, label="successor event count")
    _expect_int(runtime.get("predecessor_manifest_output_count"), 93, label="predecessor output count")
    _expect_int(runtime.get("successor_manifest_output_count"), 102, label="successor output count")
    _expect_int(runtime.get("output_delta_count"), 9, label="output delta")
    _expect_exact(runtime.get("mutable_publish_order"), list(MUTABLE_NAMES), label="mutable order")
    _expect_exact(runtime.get("allowed_mutable_states"), [
        ["OLD_EXACT", "OLD_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "OLD_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "NEW_EXACT", "OLD_EXACT"],
        ["NEW_EXACT", "NEW_EXACT", "NEW_EXACT"],
    ], label="recoverable mutable states")
    for name in MUTABLE_NAMES:
        spec = runtime.get("predecessor_mutables", {}).get(name)
        if not isinstance(spec, dict):
            raise BindingError(f"predecessor mutable spec missing: {name}")
        if isinstance(spec.get("bytes"), bool) or not isinstance(spec.get("bytes"), int) or spec["bytes"] <= 0:
            raise BindingError(f"{name} byte count is invalid")
        _expect_hex(spec.get("sha256"), HEX64, label=f"{name} SHA-256")
        if not isinstance(spec.get("snapshot_name"), str) or "/" in spec["snapshot_name"]:
            raise BindingError(f"invalid snapshot name: {name}")
    members = runtime.get("artifact_members")
    if not isinstance(members, list) or len(members) != 5:
        raise BindingError("artifact member closure must contain five records")
    expected_names = ["INPUT_INTEGRITY_AUDIT.json", "PUBLISHED_ENDPOINT_AUDIT.json", "QUALIFICATION_REPORT.json", "SHA256SUMS", "PUBLICATION_COMMIT.json"]
    _expect_exact([item.get("name") for item in members], expected_names, label="artifact member order")
    for observed in members:
        if isinstance(observed.get("bytes"), bool) or not isinstance(observed.get("bytes"), int) or observed["bytes"] <= 0:
            raise BindingError("artifact member byte count is invalid")
        _expect_hex(observed.get("sha256"), HEX64, label=f"artifact member {observed.get('name')}")

    exact_blockers = [
        "OWNER_POLICY_FOR_PUBLISHED_ENDPOINT_USE_NOT_FROZEN",
        "CHECKPOINT_SPECIFIC_ENDPOINT_USE_NOT_CLEARED",
        "BIOLOGICAL_SOURCE_GROUP_AUTHORITY_NOT_CLOSED",
        "CURRENT_AUTHORITY_80S_BLOCKER_SCOPE_NOT_ROUTED_FOR_PUBLISHED_ENDPOINT_REUSE",
        "OUTCOME_BLIND_SPLIT_AND_LEAKAGE_POLICY_NOT_FROZEN",
        "POWER_AND_CONFIDENCE_INTERVAL_ADEQUACY_NOT_ESTABLISHED",
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS_NOT_ADJUDICATED",
        "ROW_LEVEL_REPLICATE_AND_STANDARD_ERROR_ADJUDICATION_NOT_CLOSED",
    ]
    _expect_exact(config.get("unresolved_blockers"), exact_blockers, label="ordered blocker closure")
    aggregate = config.get("aggregate_truth", {})
    _expect_exact(aggregate.get("table_s2"), {
        "raw_row_count": 13850, "unique_content_row_count": 13836,
        "exact_duplicate_excess_row_count": 14, "duplicated_pair_id_count": 7,
        "deduplicated_pair_count": 6885, "deduplicated_control_count": 66,
        "design_orientation_counts": {"forward": 3497, "reverse_complement": 3388, "unresolved": 0},
    }, label="S2 aggregate truth")
    _expect_exact(aggregate.get("table_s3"), {
        "primary_data_row_count": 13544, "primary_pair_key_count": 6772,
        "total_poly_complete_pair_count": 6547, "total_poly_na_pair_count": 225,
        "high_poly_complete_pair_count": 6538, "high_poly_na_pair_count": 234,
        "table_s2_absent_from_table_s3_pair_count": 113, "post_dedup_primary_attrition_count": 338,
        "both_comparisons_complete_pair_count": 6538, "primary_only_complete_pair_count": 9,
        "secondary_only_complete_pair_count": 0, "neither_comparison_complete_pair_count": 225,
        "joined_orientation_counts": {"forward": 3451, "reverse_complement": 3321, "unresolved": 0},
        "control_sheet_data_cell_read_count": 0, "translation_formula_cell_count": 13544,
        "translation_cached_string_cell_count": 13544,
        "translation_cached_values_role": "DESCRIPTIVE_ONLY_NOT_MEMBERSHIP_OR_GATE",
    }, label="S3 aggregate truth")
    _expect_exact(aggregate.get("endpoint_boundary"), {
        "primary_complete_distinct_wt_201nt_proxy_group_count": 6544,
        "singleton_proxy_group_count": 6541, "two_candidate_proxy_group_count": 3,
        "biological_source_group_authority_closed": False,
        "study_level_reported_biological_replicate_count": 6,
        "row_level_effective_replicate_count": None, "standard_error": None,
        "power_effective_n": None, "true_a2_dense_pool_count": 0,
        "true_a2_dense_candidate_count": 0,
    }, label="endpoint aggregate truth")
    _expect_exact(config.get("successor_invariants"), {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified_independent_ordinary_studies": 0, "qualified_a1_studies": 0,
        "qualified_a2_dense_studies": 0, "metadata_only_qualification_count": 0,
        "ordinary_study_contribution": 0, "a1_study_contribution": 0,
        "true_a2_study_contribution": 0, "canonical_intervention_record_count": 0,
        "qualified": False, "training_started": False, "training_authorized": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
    }, label="unchanged A1 gate truth")
    unresolved_projection = copy.deepcopy(config)
    unresolved_projection["repository_authority"]["ledger_semantics"][
        "runtime_sync_status"
    ] = "FROZEN_LEDGER_STATUS_CHECKED_SEPARATELY"
    if _contains_unresolved(unresolved_projection):
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
    binding = expected["implementation_binding"]
    for key in ("status", "implementation_commit", "implementation_script_sha256", "implementation_test_sha256"):
        binding[key] = UNKNOWN
    return expected


def _run_git(repo_root: Path, *arguments: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run([GIT_BINARY, "-C", str(repo_root), *arguments], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuthorityError(f"read-only git command failed to start: {arguments!r}") from exc
    if result.returncode not in allowed_returncodes:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise AuthorityError(f"read-only git command failed: {arguments!r} rc={result.returncode} stderr={message!r}")
    return result.stdout


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{relative}")


def _paths_changed_by_commit(repo_root: Path, commit: str) -> list[str]:
    return sorted(_run_git(repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines())


def _validate_ledger_interim(payload: bytes, config: dict[str, Any]) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthorityError("ledger interim is not UTF-8") from exc
    semantics = config["repository_authority"]["ledger_semantics"]
    required = [
        f"  {semantics['node_id']}:",
        f"    publication_state: {semantics['publication_state']}",
        f"    execution_outcome: {semantics['execution_outcome']}",
        f"    qualification_status: {semantics['qualification_status']}",
        f"    runtime_sync_status: {semantics['runtime_sync_status']}",
    ]
    if any(token not in text for token in required):
        raise AuthorityError("ledger GSE200304 node semantics drift")


def audit_repo_authority(repo_root: Path, config: dict[str, Any], config_payload: bytes) -> dict[str, Any]:
    """Prove frozen L -> I -> config-only B/HEAD/origin and exact blobs."""

    if lexical_absolute(repo_root) != PRODUCTION_REPO_ROOT or repo_root.is_symlink() or not repo_root.is_dir():
        raise AuthorityError(f"repository root must be exactly {PRODUCTION_REPO_ROOT}")
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    implementation = binding["implementation_commit"]
    ledger = authority["ledger_commit"]
    branch = authority["branch"]
    head = _run_git(repo_root, "rev-parse", "HEAD").decode().strip()
    origin = _run_git(repo_root, "rev-parse", "--verify", f"refs/remotes/origin/{branch}").decode().strip()
    upstream_name = _run_git(repo_root, "rev-parse", "--abbrev-ref", "@{upstream}").decode().strip()
    upstream = _run_git(repo_root, "rev-parse", "@{upstream}").decode().strip()
    _expect_exact(_run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(), branch, label="branch")
    _expect_exact(upstream_name, f"origin/{branch}", label="upstream")
    _expect_exact(upstream, head, label="upstream HEAD")
    _expect_exact(origin, head, label="origin HEAD")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{head}^").decode().strip(), implementation, label="B direct parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{implementation}^").decode().strip(), ledger, label="I direct parent")
    _expect_exact(_run_git(repo_root, "rev-parse", f"{ledger}^").decode().strip(), authority["ledger_commit_expected_parent"], label="L direct parent")
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AuthorityError("production repository worktree or index is not clean")
    _run_git(repo_root, "merge-base", "--is-ancestor", ledger, head)
    _expect_exact(_paths_changed_by_commit(repo_root, implementation), authority["implementation_commit_exact_changed_paths"], label="I exact changed paths")
    _expect_exact(_paths_changed_by_commit(repo_root, head), authority["binding_commit_exact_changed_paths"], label="B exact changed paths")
    if _git_blob(repo_root, head, CONFIG_REPO_PATH) != config_payload:
        raise AuthorityError("working config bytes do not equal B blob")
    i_config = load_json(_git_blob(repo_root, implementation, CONFIG_REPO_PATH), label="I config")
    if i_config != expected_unknown_i_config(config) or compiled_core_projection(i_config) != compiled_core_projection(config):
        raise AuthorityError("I/B config compiled core or dynamic binding transition drift")
    for path, digest in ((SCRIPT_REPO_PATH, binding["implementation_script_sha256"]), (TEST_REPO_PATH, binding["implementation_test_sha256"])):
        if sha256(_git_blob(repo_root, implementation, path)) != digest or sha256(_git_blob(repo_root, head, path)) != digest:
            raise AuthorityError(f"implementation blob drift: {path}")
        if sha256(read_regular_path(repo_root / path)) != digest:
            raise AuthorityError(f"working implementation file drift: {path}")
    for item in authority["ledger_files"]:
        payload = _git_blob(repo_root, ledger, item["path"])
        if sha256(payload) != item["sha256"] or sha256(_git_blob(repo_root, head, item["path"])) != item["sha256"]:
            raise AuthorityError(f"ledger blob drift: {item['path']}")
        if sha256(read_regular_path(repo_root / item["path"])) != item["sha256"]:
            raise AuthorityError(f"working ledger file drift: {item['path']}")
        if item["path"].endswith("route_a_v3_a1_interim.yaml"):
            _validate_ledger_interim(payload, config)
    qualifier = authority["external_qualifier_binding"]
    for path_field, hash_field in (("config_path", "config_sha256"), ("script_path", "script_sha256"), ("test_path", "test_sha256")):
        if sha256(_git_blob(repo_root, qualifier["binding_commit"], qualifier[path_field])) != qualifier[hash_field]:
            raise AuthorityError(f"external qualifier blob drift: {qualifier[path_field]}")
    return {
        "status": "PASS_EXACT_LEDGER_L_TO_I_TO_CONFIG_ONLY_B",
        "branch": branch,
        "ledger_commit": ledger,
        "implementation_commit": implementation,
        "binding_commit": head,
        "head_commit": head,
        "origin_branch_head_commit": origin,
        "config_sha256": sha256(config_payload),
        "head_equals_origin_branch_head": True,
        "ledger_is_ancestor": True,
        "implementation_parent_is_ledger": True,
        "binding_parent_is_implementation": True,
        "binding_commit_is_config_only": True,
        "worktree_and_index_clean": True,
    }


def validate_recorded_at(value: str, predecessor_at: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", value) is None:
        raise RuntimeSyncError("recorded_at must be second-resolution RFC3339 with +08:00")
    try:
        current, predecessor = datetime.fromisoformat(value), datetime.fromisoformat(predecessor_at)
    except ValueError as exc:
        raise RuntimeSyncError("invalid recorded_at") from exc
    if current.utcoffset() != timedelta(hours=8) or current <= predecessor:
        raise RuntimeSyncError("recorded_at must be +08:00 and strictly after A1-EVT-036")
    if current.astimezone(timezone.utc).date().isoformat() not in {"2026-08-10", "2026-08-11"}:
        raise RuntimeSyncError("recorded_at is outside the frozen EVT-037 window")


def open_absolute_directory_nofollow(path: Path) -> int:
    """Root-to-leaf descriptor walk; no path component may be a symlink."""

    if not path.is_absolute():
        raise PublicationError("directory path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        for component in path.parts[1:]:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            child = os.open(component, flags, dir_fd=descriptor)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise PublicationError(f"not a nofollow directory: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _expect_artifact_value(document: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if document.get(key) != expected:
        raise PublicationError(f"{label} {key} drift")


def validate_artifact_bundle(config: dict[str, Any]) -> dict[str, Any]:
    """Read only the frozen five aggregate artifacts and validate their closure."""

    runtime = config["runtime"]
    root = Path(runtime["artifact_root"])
    descriptor = open_absolute_directory_nofollow(root)
    try:
        expected_names = [item["name"] for item in runtime["artifact_members"]]
        if sorted(os.listdir(descriptor)) != sorted(expected_names):
            raise PublicationError("artifact bundle exact member set drift")
        payloads: dict[str, bytes] = {}
        for item in runtime["artifact_members"]:
            payload = read_regular_at(descriptor, item["name"], require_single_link=True)
            if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
                raise PublicationError(f"artifact member bytes or SHA-256 drift: {item['name']}")
            payloads[item["name"]] = payload
    finally:
        os.close(descriptor)
    checksummed = sorted(["INPUT_INTEGRITY_AUDIT.json", "PUBLISHED_ENDPOINT_AUDIT.json", "QUALIFICATION_REPORT.json"])
    expected_sums = "".join(f"{sha256(payloads[name])}  {name}\n" for name in checksummed).encode("ascii")
    if payloads["SHA256SUMS"] != expected_sums:
        raise PublicationError("artifact SHA256SUMS closure drift")
    integrity = load_json(payloads["INPUT_INTEGRITY_AUDIT.json"], label="artifact input audit")
    endpoint = load_json(payloads["PUBLISHED_ENDPOINT_AUDIT.json"], label="artifact endpoint audit")
    report = load_json(payloads["QUALIFICATION_REPORT.json"], label="artifact qualification report")
    marker = load_json(payloads["PUBLICATION_COMMIT.json"], label="artifact terminal marker")
    truth = config["artifact_truth"]
    for key, expected in {
        "execution_outcome": truth["execution_outcome"], "qualification_status": truth["qualification_status"],
        "qualified": False, "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0, "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0, "training_allowed": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }.items():
        _expect_artifact_value(report, key, expected, label="qualification report")
    _expect_artifact_value(report, "unresolved_blockers", config["unresolved_blockers"], label="qualification report")
    qualifier_binding = report.get("implementation_binding", {})
    _expect_artifact_value(qualifier_binding, "implementation_commit", truth["qualifier_implementation_commit"], label="qualifier binding")
    _expect_artifact_value(qualifier_binding, "binding_commit", truth["qualifier_binding_commit"], label="qualifier binding")
    for key, expected in {"network_accessed": False, "raw_fastq_or_alignment_input_count": 0, "external_code_executed": False, "aggregate_only": True}.items():
        _expect_artifact_value(integrity, key, expected, label="input audit")
    _expect_artifact_value(endpoint, "aggregate_only", True, label="endpoint audit")
    _expect_artifact_value(endpoint, "published_endpoint_is_not_raw_replay", True, label="endpoint audit")
    _expect_artifact_value(endpoint, "published_endpoint_is_not_canonical_materialization", True, label="endpoint audit")
    s2, s3, boundary = endpoint.get("table_s2", {}), endpoint.get("table_s3", {}), endpoint.get("endpoint_boundary", {})
    aggregate = config["aggregate_truth"]
    s2_map = {"raw_row_count": "raw_row_count", "unique_content_row_count": "unique_content_row_count", "duplicate_extra_row_count": "exact_duplicate_excess_row_count", "duplicated_pair_count": "duplicated_pair_id_count", "deduplicated_pair_count": "deduplicated_pair_count", "deduplicated_control_count": "deduplicated_control_count"}
    for actual, frozen in s2_map.items():
        _expect_artifact_value(s2, actual, aggregate["table_s2"][frozen], label="S2 audit")
    _expect_artifact_value(s2, "all_pair_orientation_counts", {"FORWARD": 3497, "REVERSE_COMPLEMENT": 3388, "UNRESOLVED": 0}, label="S2 audit")
    for key, expected in {
        "primary_data_row_count": 13544, "primary_pair_count": 6772,
        "finite_statistic_rows": {"HighPoly:RNA": 6538, "TotalPoly:RNA": 6547},
        "na_statistic_rows": {"HighPoly:RNA": 234, "TotalPoly:RNA": 225},
        "both_comparisons_finite_pair_count": 6538, "primary_only_finite_pair_count": 9,
        "secondary_only_finite_pair_count": 0, "neither_comparison_finite_pair_count": 225,
        "translation_formula_count": 13544, "cached_translation_string_count": 13544,
        "opaque_control_data_cell_read_count": 0,
        "cached_translation_counts_role": "DESCRIPTIVE_ONLY_NOT_MEMBERSHIP_OR_GATE",
    }.items():
        _expect_artifact_value(s3, key, expected, label="S3 audit")
    for key, expected in {
        "joined_pair_count": 6772, "table_s2_absent_from_table_s3_count": 113,
        "joined_pair_orientation_counts": {"FORWARD": 3451, "REVERSE_COMPLEMENT": 3321, "UNRESOLVED": 0},
        "primary_finite_effect_pair_count": 6547, "primary_na_pair_count": 225,
        "primary_total_attrition_count": 338,
        "primary_complete_distinct_wt_201nt_proxy_group_count": 6544,
        "primary_complete_wt_201nt_proxy_pool_size_counts": {"1": 6541, "2": 3},
        "biological_source_group_authority_closed": False,
        "study_level_reported_biological_replicate_count": 6,
        "row_level_effective_replicate_count": None, "standard_error": None,
        "power_effective_n": None, "true_a2_dense_candidate_count": 0,
    }.items():
        _expect_artifact_value(boundary, key, expected, label="endpoint boundary")
    for key, expected in {
        "record_type": truth["terminal_record_type"], "execution_outcome": truth["execution_outcome"],
        "bundle_member_names": truth["terminal_declared_member_names"], "bundle_member_count": 4,
        "sha256sums_sha256": sha256(payloads["SHA256SUMS"]), "committed": True,
        "terminal_marker_written_last": True,
        "terminal_publication_operation": truth["terminal_publication_operation"],
    }.items():
        _expect_artifact_value(marker, key, expected, label="terminal marker")
    return {
        "artifact_root": str(root),
        "member_count": 5,
        "publication_state": truth["publication_state"],
        "execution_outcome": truth["execution_outcome"],
        "qualification_status": truth["qualification_status"],
        "member_records": copy.deepcopy(runtime["artifact_members"]),
        "blockers": list(config["unresolved_blockers"]),
        "aggregate_truth": copy.deepcopy(config["aggregate_truth"]),
    }


def _validate_predecessor_objects(status: dict[str, Any], manifest: dict[str, Any], events: list[dict[str, Any]], config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    if len(events) != 36 or not events or events[-1].get("event_id") != "A1-EVT-036":
        raise PublicationError("predecessor event count/tail drift")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 93:
        raise PublicationError("predecessor manifest output count is not exactly 93")
    for key, expected in {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "claim_status": "NOT_ESTABLISHED",
        "qualified_a1_studies": 0, "qualified_a2_dense_studies": 0,
        "metadata_only_qualification_count": 0, "training_started": False,
        "next_phase_authorized": False,
        "code_commit": config["historical_runtime_authority"]["code_commit"],
    }.items():
        _expect_exact(status.get(key), expected, label=f"predecessor STATUS {key}")
    for key, expected in {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED", "code_commit": config["historical_runtime_authority"]["code_commit"],
        "active_authority_commit": config["historical_runtime_authority"]["active_authority_commit"],
    }.items():
        _expect_exact(manifest.get(key), expected, label=f"predecessor manifest {key}")
    new_paths = {str(Path(runtime["artifact_root"]) / item["name"]) for item in runtime["artifact_members"]}
    new_paths.update(str(Path(runtime["run_root"]) / name) for name in immutable_names(config))
    if any(isinstance(item, dict) and item.get("absolute_path") in new_paths for item in outputs):
        raise PublicationError("EVT-037 output is already registered")


def read_exact_predecessor(run_fd: int, config: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, bytes] = {}
    for name in MUTABLE_NAMES:
        payload = read_regular_at(run_fd, name)
        spec = config["runtime"]["predecessor_mutables"][name]
        if len(payload) != spec["bytes"] or sha256(payload) != spec["sha256"]:
            raise PublicationError(f"exact predecessor mutable drift: {name}")
        payloads[name] = payload
    status = load_json(payloads["STATUS.json"], label="predecessor STATUS")
    manifest = load_json(payloads["RUN_MANIFEST.json"], label="predecessor manifest")
    events = load_json_lines(payloads["EVENT_LOG.jsonl"], label="predecessor events")
    _validate_predecessor_objects(status, manifest, events, config)
    selected = validate_artifact_bundle(config)
    return payloads, status, manifest, events, selected


def output_record(artifact_type: str, absolute_path: Path, digest: str) -> dict[str, str]:
    return {"artifact_type": artifact_type, "absolute_path": str(absolute_path), "sha256": digest, "status": "COMPLETE"}


def snapshot_names(config: dict[str, Any]) -> dict[str, str]:
    return {name: config["runtime"]["predecessor_mutables"][name]["snapshot_name"] for name in MUTABLE_NAMES}


def immutable_names(config: dict[str, Any]) -> tuple[str, ...]:
    snapshots = snapshot_names(config)
    return (snapshots["STATUS.json"], snapshots["RUN_MANIFEST.json"], snapshots["EVENT_LOG.jsonl"], config["runtime"]["sync_name"])


def expected_output_delta(config: dict[str, Any], sync_digest: str) -> list[dict[str, str]]:
    runtime = config["runtime"]
    run_root, artifact_root = Path(runtime["run_root"]), Path(runtime["artifact_root"])
    result = [output_record(item["artifact_type"], artifact_root / item["name"], item["sha256"]) for item in runtime["artifact_members"]]
    snapshots = snapshot_names(config)
    result.extend([
        output_record("A1_STATUS_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_SNAPSHOT", run_root / snapshots["STATUS.json"], runtime["predecessor_mutables"]["STATUS.json"]["sha256"]),
        output_record("A1_RUN_MANIFEST_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_SNAPSHOT", run_root / snapshots["RUN_MANIFEST.json"], runtime["predecessor_mutables"]["RUN_MANIFEST.json"]["sha256"]),
        output_record("A1_EVENT_LOG_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_SNAPSHOT", run_root / snapshots["EVENT_LOG.jsonl"], runtime["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"]),
        output_record("A1_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1", run_root / runtime["sync_name"], sync_digest),
    ])
    return result


def build_successors(config: dict[str, Any], predecessor_payloads: dict[str, bytes], predecessor_status: dict[str, Any], predecessor_manifest: dict[str, Any], predecessor_events: list[dict[str, Any]], selected_source: dict[str, Any], authority_audit: dict[str, Any], recorded_at: str) -> dict[str, bytes]:
    runtime, binding = config["runtime"], config["implementation_binding"]
    validate_recorded_at(recorded_at, predecessor_events[-1]["at"])
    run_root = Path(runtime["run_root"])
    snapshots = snapshot_names(config)
    sync = {
        "schema_version": "1.0.0",
        "record_type": "ROUTE_A_V3_A1_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC",
        "contract_id": config["contract_id"], "phase_id": "A1", "dataset_id": "GSE200304",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", "recorded_at": recorded_at,
        "predecessor_runtime": {
            "runtime_root": str(run_root), "last_event_id": "A1-EVT-036", "event_count": 36,
            "manifest_output_count": 93,
            "immutable_pre_sync_snapshots": [
                {"source_mutable_path": str(run_root / name), "snapshot_path": str(run_root / snapshots[name]), "bytes": len(predecessor_payloads[name]), "sha256": sha256(predecessor_payloads[name])}
                for name in MUTABLE_NAMES
            ],
        },
        "published_endpoint_artifact": {
            "root": selected_source["artifact_root"], "member_count": 5,
            "members": selected_source["member_records"],
            "publication_state": selected_source["publication_state"],
            "execution_outcome": selected_source["execution_outcome"],
            "qualification_status": selected_source["qualification_status"],
            "producer": {"implementation_commit": config["artifact_truth"]["qualifier_implementation_commit"], "binding_commit": config["artifact_truth"]["qualifier_binding_commit"]},
            "body_embedded": False,
        },
        "repository_ledger": {
            "ledger_commit": config["repository_authority"]["ledger_commit"],
            "ledger_files": config["repository_authority"]["ledger_files"],
            "ledger_semantics": config["repository_authority"]["ledger_semantics"],
        },
        "runtime_sync_publisher_authority": {
            "status": authority_audit["status"], "branch": config["repository_authority"]["branch"],
            "implementation_commit": binding["implementation_commit"], "binding_commit": authority_audit["binding_commit"],
            "head_commit": authority_audit["head_commit"], "origin_branch_head_commit": authority_audit["origin_branch_head_commit"],
            "config_path": CONFIG_REPO_PATH, "config_sha256": authority_audit["config_sha256"],
            "script_path": SCRIPT_REPO_PATH, "script_sha256": binding["implementation_script_sha256"],
            "test_path": TEST_REPO_PATH, "test_sha256": binding["implementation_test_sha256"],
        },
        "mechanical_aggregates": selected_source["aggregate_truth"],
        "scientific_blockers": {"count": 8, "exact": selected_source["blockers"]},
        "a1_gate_snapshot": copy.deepcopy(config["successor_invariants"]),
        "access_and_materialization_boundary": copy.deepcopy(config["privacy_boundary"]),
        "hash_linkage": {"direction": "PREDECESSOR_AND_ARTIFACT_AND_REPOSITORY_LEDGER_TO_SYNC_TO_SUCCESSORS", "sync_record_references_successor_hashes": False, "successors_reference_sync_hash": True},
        "self_hash": "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST",
    }
    sync_payload = json_bytes(sync)
    sync_digest = sha256(sync_payload)
    prefix = "gse200304_published_endpoint_"
    status = dict(predecessor_status)
    status.update({
        "updated_at": recorded_at,
        prefix + "runtime_sync_status": "SYNCED_EVT_037",
        prefix + "runtime_sync_record_sha256": sync_digest,
        prefix + "artifact_root": runtime["artifact_root"],
        prefix + "artifact_member_count": 5,
        prefix + "publication_state": selected_source["publication_state"],
        prefix + "execution_outcome": selected_source["execution_outcome"],
        prefix + "qualification_status": selected_source["qualification_status"],
        prefix + "scientific_blocker_count": 8,
        prefix + "qualifier_implementation_commit": config["artifact_truth"]["qualifier_implementation_commit"],
        prefix + "qualifier_binding_commit": config["artifact_truth"]["qualifier_binding_commit"],
        prefix + "runtime_sync_implementation_commit": binding["implementation_commit"],
        prefix + "runtime_sync_binding_commit": authority_audit["binding_commit"],
        prefix + "ordinary_study_contribution": 0, prefix + "a1_study_contribution": 0,
        prefix + "true_a2_study_contribution": 0, prefix + "canonical_record_count": 0,
        prefix + "qualified": False, prefix + "training_allowed": False,
        prefix + "model_selection_allowed": False, prefix + "next_phase_authorized": False,
        prefix + "scientific_claim_status": "NOT_ESTABLISHED",
        prefix + "raw_or_row_payload_access_count": 0,
        prefix + "restricted_or_sealed_path_accessed": False,
    })
    event = {
        "event_id": "A1-EVT-037", "at": recorded_at, "phase_id": "A1",
        "run_id": "A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5", "event": config["event_name"],
        "sync_record": runtime["sync_name"], "sync_record_sha256": sync_digest,
        "artifact_root": runtime["artifact_root"], "artifact_member_count": 5,
        "publication_state": selected_source["publication_state"], "execution_outcome": selected_source["execution_outcome"],
        "qualification_status": selected_source["qualification_status"], "scientific_blocker_count": 8,
        "ordinary_study_contribution": 0, "a1_study_contribution": 0, "true_a2_study_contribution": 0,
        "canonical_intervention_record_count": 0, "qualified": False,
        "training_authorized": False, "model_selection_allowed": False, "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED", "pre_sync_snapshot_count": 3,
        "manifest_output_count_before": 93, "manifest_output_count_after": 102,
        "raw_reads_or_alignments_opened": False, "raw_fastq_body_read_count": 0,
        "row_level_payload_included": False, "sequence_payload_included": False,
        "barcode_payload_included": False, "annotation_label_payload_included": False,
        "gpu_work_started": False, "restricted_or_sealed_path_accessed": False,
        "restricted_or_sealed_payload_contact": False,
        "detail": "Registered the exact committed GSE200304 published-endpoint aggregate bundle. Engineering publication remains immutably blocked from qualification; all study contributions, canonical records, training, model selection, next-phase authorization, and scientific claims remain zero or false.",
    }
    manifest = dict(predecessor_manifest)
    manifest[prefix + "runtime_sync_record_sha256"] = sync_digest
    manifest["outputs"] = list(predecessor_manifest["outputs"]) + expected_output_delta(config, sync_digest)
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


def validate_successors(config: dict[str, Any], artifacts: dict[str, bytes], predecessor_payloads: dict[str, bytes], predecessor_status: dict[str, Any], predecessor_manifest: dict[str, Any], sync_digest: str) -> None:
    runtime, snapshots = config["runtime"], snapshot_names(config)
    expected_names = set(MUTABLE_NAMES) | set(snapshots.values()) | {runtime["sync_name"]}
    if set(artifacts) != expected_names:
        raise RuntimeSyncError("prepared artifact schema drift")
    for mutable, snapshot in snapshots.items():
        if artifacts[snapshot] != predecessor_payloads[mutable]:
            raise RuntimeSyncError(f"snapshot is not byte-exact: {snapshot}")
    sync_payload = artifacts[runtime["sync_name"]]
    sync = load_json(sync_payload, label="sync record")
    if sha256(sync_payload) != sync_digest or sync.get("self_hash") != "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST":
        raise RuntimeSyncError("sync identity drift")
    if sync.get("hash_linkage", {}).get("sync_record_references_successor_hashes") is not False:
        raise RuntimeSyncError("sync successor-hash policy drift")
    successor_digests = [sha256(artifacts[name]) for name in MUTABLE_NAMES]
    if any(digest.encode("ascii") in sync_payload for digest in successor_digests):
        raise RuntimeSyncError("successor hash leaked into sync")
    for name in MUTABLE_NAMES:
        if sync_digest.encode("ascii") not in artifacts[name]:
            raise RuntimeSyncError(f"successor lacks sync binding: {name}")
    status = load_json(artifacts["STATUS.json"], label="successor status")
    if {key: value for key, value in status.items() if not key.startswith("gse200304_published_endpoint_") and key != "updated_at"} != {key: value for key, value in predecessor_status.items() if key != "updated_at"}:
        raise RuntimeSyncError("predecessor STATUS field rewritten")
    if status.get("code_commit") != config["historical_runtime_authority"]["code_commit"]:
        raise RuntimeSyncError("historical STATUS code commit rewritten")
    prefix = "gse200304_published_endpoint_"
    for key, expected in {
        "runtime_sync_status": "SYNCED_EVT_037", "runtime_sync_record_sha256": sync_digest,
        "artifact_member_count": 5, "publication_state": "COMMITTED_ACCEPTED",
        "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED", "qualification_status": "BLOCKED_NOT_QUALIFIED",
        "scientific_blocker_count": 8, "ordinary_study_contribution": 0,
        "a1_study_contribution": 0, "true_a2_study_contribution": 0,
        "canonical_record_count": 0, "qualified": False, "training_allowed": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED", "raw_or_row_payload_access_count": 0,
        "restricted_or_sealed_path_accessed": False,
    }.items():
        _expect_exact(status.get(prefix + key), expected, label=f"successor STATUS {key}")
    manifest = load_json(artifacts["RUN_MANIFEST.json"], label="successor manifest")
    for key, value in predecessor_manifest.items():
        if key != "outputs" and manifest.get(key) != value:
            raise RuntimeSyncError(f"predecessor manifest field rewritten: {key}")
    if set(manifest) - set(predecessor_manifest) != {prefix + "runtime_sync_record_sha256"}:
        raise RuntimeSyncError("manifest top-level delta drift")
    old_outputs, outputs = predecessor_manifest["outputs"], manifest.get("outputs")
    if not isinstance(outputs, list) or len(old_outputs) != 93 or len(outputs) != 102:
        raise RuntimeSyncError("manifest output transition is not 93 -> 102")
    if outputs[:93] != old_outputs or outputs[93:] != expected_output_delta(config, sync_digest):
        raise RuntimeSyncError("manifest ordered nine-record append drift")
    if len({item.get("absolute_path") for item in outputs if isinstance(item, dict)}) != len(outputs):
        raise RuntimeSyncError("manifest output paths are not unique")
    events = load_json_lines(artifacts["EVENT_LOG.jsonl"], label="successor events")
    old_events = load_json_lines(predecessor_payloads["EVENT_LOG.jsonl"], label="predecessor events")
    if len(events) != 37 or events[:-1] != old_events or not artifacts["EVENT_LOG.jsonl"].startswith(predecessor_payloads["EVENT_LOG.jsonl"]):
        raise RuntimeSyncError("EVENT_LOG is not an exact one-event append")
    event = events[-1]
    _expect_exact(event.get("event_id"), "A1-EVT-037", label="event id")
    _expect_exact(event.get("event"), config["event_name"], label="event name")
    _expect_exact(event.get("sync_record_sha256"), sync_digest, label="event sync binding")
    for key, expected in {"manifest_output_count_before": 93, "manifest_output_count_after": 102, "scientific_blocker_count": 8, "qualified": False, "training_authorized": False, "next_phase_authorized": False, "scientific_claim_status": "NOT_ESTABLISHED", "raw_reads_or_alignments_opened": False, "raw_fastq_body_read_count": 0, "row_level_payload_included": False, "sequence_payload_included": False, "barcode_payload_included": False, "gpu_work_started": False}.items():
        _expect_exact(event.get(key), expected, label=f"EVT-037 {key}")

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
        "manifest_output_transition": "93_TO_102",
        "published_endpoint_artifact_state": "EXISTING_FIVE_MEMBER_BUNDLE_EXACT_REUSED",
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
    stale_temporaries = sorted(
        name
        for name in os.listdir(run_fd)
        if re.fullmatch(r"\.evt037\.[0-9]+\.[0-9a-f]{16}\.[^/]+\.tmp", name)
    )
    if stale_temporaries:
        raise PublicationError(
            "stale EVT-037 publisher temporary member makes the run namespace unclosed: "
            + repr(stale_temporaries)
        )
    selected_source = validate_artifact_bundle(config)

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

    temporary_name = f".evt037.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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

    temporary_name = f".evt037.{os.getpid()}.{secrets.token_hex(8)}.{name}.tmp"
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
        "published_endpoint_artifact_state": "EXISTING_FIVE_MEMBER_BUNDLE_EXACT_REUSED",
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
        "PUBLISHED_ENDPOINT_ARTIFACT_BUNDLE": {
            "state": "EXISTING_FIVE_MEMBER_BUNDLE_EXACT_REUSED",
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
        "manifest_output_transition": "93_TO_102",
        "published_endpoint_artifact_state": "EXISTING_FIVE_MEMBER_BUNDLE_EXACT_REUSED",
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
