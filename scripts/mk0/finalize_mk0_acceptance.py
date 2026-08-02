#!/usr/bin/env python3
"""Fail-closed merge of CPU, CUDA and upstream evidence for MK0-v1.

The finalizer never manufactures a passing gate.  It validates runner-emitted
runtime bindings against the frozen configuration, re-hashes every support
artifact, verifies the clean source tree, and re-verifies D1/B0/FM0 prerequisite
bytes before it writes terminal acceptance records.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from types import ModuleType
import traceback
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
_bootstrap_path = REPO_ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
_bootstrap_module = ModuleType("_mk0_strict_worktree_import")
_bootstrap_module.__file__ = str(_bootstrap_path)
_bootstrap_module.__cached__ = None
exec(
    compile(
        _bootstrap_path.read_bytes(),
        str(_bootstrap_path),
        "exec",
        dont_inherit=True,
        optimize=0,
    ),
    _bootstrap_module.__dict__,
)

import yaml

with _bootstrap_module.strict_worktree_package_import(REPO_ROOT):
    from mrna_editflow.core.mk0.acceptance import (
        GateResult,
        aggregate_acceptance,
        canonical_json_bytes,
        gate_result_from_runtime_binding,
        sha256_file,
        verify_bound_file,
    )
    from mrna_editflow.core.mk0.run_contract import (
        EVIDENCE_LEVEL,
        LOG_FILENAMES,
        RUN_DIRECTORIES,
        append_event,
        append_jsonl,
        append_text,
        immutable_file_inventory,
        resume_failure_closure_if_present,
        update_status,
        validate_terminal_chain,
        write_bytes_exclusive_atomic,
        write_failed_sentinel,
        write_json_exclusive_atomic,
        write_whole_run_checksum_ledger,
    )

GPU_GATE_IDS = {"M05", "M31", "M32", "M35"}
CPU_RESULTS = "mk0_cpu_gate_results.json"
GPU_RESULTS = "mk0_gpu_gate_results.json"
CPU_SUMMARY = "cpu_acceptance_summary.json"
GPU_SUMMARY = "gpu_acceptance_summary.json"
FM0_HASH_LICENSE_RELATIVE = Path("evaluation/hash_license_manifest.json")
FM0_READY_MARKER_TEXT = (
    "FM0 formal acceptance bound and verified; no downstream phase was started."
)
CPU_SUPPORT = {
    "coupling_manifest.json",
    "transition_aggregation_oracle.json",
    "loss_oracle_report.json",
    "hazard_audit.json",
    "sampler_convergence.json",
    "stop_audit.json",
    "stop_survival_oracle.json",
    "critic_role_audit.json",
    "mk0_schema_action_audit.json",
    "mk0_text_contract_audit.json",
}
GPU_SUPPORT = {
    "foundation_fusion_audit.json",
    "target_alignment_leakage_audit.json",
}
MANDATORY_SUPPORT = CPU_SUPPORT | GPU_SUPPORT | {CPU_RESULTS, GPU_RESULTS}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
LEDGER_LINE = re.compile(r"([0-9a-f]{64})  (?:\./)?(.+)")
FORMAL_RUN_ID = re.compile(
    r"^MK0_(?P<model>[A-Za-z0-9]+)_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<split>[A-Za-z0-9]+)_(?P<utc>[0-9]{8}T[0-9]{6}Z)_"
    r"(?P<short_sha>[0-9a-f]{7,12})_s(?P<seed>[0-9]+)$"
)
CANONICAL_RUN_PARENT = Path(
    "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/artifacts/runs"
)

GPU_ROLE_PHASE_SPECS = (
    (
        "generator_rate_official_frozen_arm",
        "generator_rate",
        "_run_forced_action_arm",
        ("generator_interface", "rate_interface"),
    ),
    (
        "generator_rate_from_scratch_control_arm",
        "generator_rate",
        "_run_forced_action_arm",
        ("generator_interface", "rate_interface"),
    ),
    (
        "sampler_paper_official_foundation",
        "sampler_rate",
        "_run_official_paper_sampler_route",
        ("sampler_interface", "rate_interface"),
    ),
    (
        "sampler_primary_official_foundation",
        "sampler_rate",
        "_run_primary_gpu_sampler_integration",
        ("sampler_interface", "rate_interface"),
    ),
    (
        "rate_target_alignment_leakage_audit",
        "rate_audit",
        "_audit_target_alignment_leakage",
        ("rate_interface",),
    ),
    (
        "rate_dynamic_current_encoding_audit",
        "rate_audit",
        "_audit_dynamic_current_encoding",
        ("rate_interface",),
    ),
)
GPU_ROLE_INTERFACE_LABELS = (
    "gpu_runner.forced_action_arm",
    "gpu_runner.paper_sampler_route",
    "gpu_runner.primary_sampler_integration",
    "gpu_runner.target_alignment_leakage_audit",
    "gpu_runner.dynamic_current_encoding_audit",
    "foundation_fusion.rate_field_forward",
    "foundation_fusion.official_paper_adapter",
    "samplers.constrained_primary",
    "samplers.paper_parallel",
    "samplers.replay_constrained",
    "samplers.replay_paper",
)
GPU_ROLE_CALL_CATEGORIES = (
    "generator_interface",
    "rate_interface",
    "sampler_interface",
    "critic_query",
    "guidance_query",
    "final_evaluator_query",
)
GPU_ROLE_QUERY_CATEGORIES = (
    "critic_query",
    "guidance_query",
    "final_evaluator_query",
)

FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES = (
    "torch",
    "transformers",
    "multimolecule",
    "tokenizers",
    "numpy",
    "safetensors",
    "huggingface_hub",
    "packaging",
)

TYPING_COMPATIBILITY_SHIM_DISTRIBUTION = "typing_extensions"
TYPING_COMPATIBILITY_SHIM_MODULE = "typing_extensions"
TYPING_COMPATIBILITY_SHIM_VERSION = "4.16.0"
TYPING_COMPATIBILITY_SHIM_SOURCE_SHA256 = (
    "4040ca1a1ecbee00d1385c12a93084d1c5bd46f0b774f07e5ae7e91c4f55e696"
)
TYPING_COMPATIBILITY_SHIM_CALLS = (
    ("_collect_type_vars", 3316),
    ("_has_generic_or_protocol_as_origin", 3281),
    ("_is_unpacked_typevartuple", 3303),
    ("_should_collect_from_parameters", 281),
)
TYPING_COMPATIBILITY_SHIM_CLASSIFICATION = (
    "bound_typing_compatibility_shim_exact_callable"
)
ENVIRONMENT_LOCK_DRIFT_MARKER = "ENVIRONMENT_LOCK_DRIFT_RECORDED_NOT_SILENTLY_MUTATED"
REQUIREMENTS_LOCK_PATH = REPO_ROOT / "requirements-lock.txt"

ROLE_PROHIBITED_TOKENS = (
    "critic",
    "guidance",
    "evaluator",
    "final_evaluator",
    "reward",
    "rerank",
    "selector",
)

STDLIB_ROOT = Path(sysconfig.get_path("stdlib")).resolve()
SITE_PACKAGE_ROOTS = tuple(
    sorted(
        {
            Path(value).resolve()
            for key in ("purelib", "platlib")
            if (value := sysconfig.get_path(key))
        },
        key=str,
    )
)

EXACT_GPU_ROLE_INTERFACES = {
    ("scripts/mk0/run_mk0_gpu_smoke.py", "_run_forced_action_arm"): (
        "generator_interface",
    ),
    (
        "core/mk0/foundation_fusion.py",
        "FoundationFusionRateField.forward",
    ): ("rate_interface",),
    (
        "core/mk0/foundation_fusion.py",
        "OfficialPaperRateAdapter.__call__",
    ): ("rate_interface",),
    ("core/mk0/samplers.py", "constrained_single_event_first_order"): (
        "sampler_interface",
    ),
    ("core/mk0/samplers.py", "paper_first_order_parallel"): ("sampler_interface",),
    ("core/mk0/samplers.py", "replay_constrained_result"): ("sampler_interface",),
    ("core/mk0/samplers.py", "replay_paper_result"): ("sampler_interface",),
}


class FinalizeFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _standard_failure_reason(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizeFailure(message)


def verify_fm0_ready_marker(marker_path: Path) -> None:
    """Bind the terminal marker to the exact text emitted by formal FM0 closure."""

    require(
        marker_path.read_text(encoding="utf-8").strip() == FM0_READY_MARKER_TEXT,
        "FM0 marker content drift",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _ordinary_unlinked_file_snapshot(
    path: Path,
    *,
    label: str,
    relative_to: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Read one no-follow FD and bind identity, bytes, size and digest together."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FinalizeFailure(
            f"{label} is not an ordinary unlinked file or cannot be opened without following"
        ) from error
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{label} is not an ordinary unlinked file",
        )
        require(before.st_size > 0, f"{label} is empty")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    require(
        all(
            getattr(before, field) == getattr(after, field) for field in identity_fields
        ),
        f"{label} changed while it was read",
    )
    data = b"".join(chunks)
    require(len(data) == before.st_size, f"{label} size changed while it was read")
    try:
        live = path.lstat()
    except FileNotFoundError as error:
        raise FinalizeFailure(f"{label} disappeared after it was read") from error
    require(
        not path.is_symlink()
        and all(
            getattr(before, field) == getattr(live, field) for field in identity_fields
        ),
        f"{label} pathname identity changed while it was read",
    )
    resolved = path.resolve(strict=True)
    path_value = (
        resolved.relative_to(relative_to).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return (
        {
            "path": path_value,
            "size_bytes": before.st_size,
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        data,
    )


def _ordinary_unlinked_file_record(
    path: Path,
    *,
    label: str,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    record, _data = _ordinary_unlinked_file_snapshot(
        path,
        label=label,
        relative_to=relative_to,
    )
    return record


def _parent_failure_path_snapshot(parent_root: Path) -> tuple[str, list[Path]]:
    """Enumerate the exact canonical failure-evidence path set."""

    sentinel_paths = [parent_root / name for name in ("DONE", "FAILED")]
    sentinels = [
        path.name for path in sentinel_paths if path.exists() or path.is_symlink()
    ]
    require(len(sentinels) <= 1, "parent run has contradictory terminal sentinels")
    require("DONE" not in sentinels, "repair parent is terminal DONE")

    evidence_paths: set[Path] = set()
    if "FAILED" in sentinels:
        evidence_paths.add(parent_root / "FAILED")

    failure_root = parent_root / "failure"
    if failure_root.exists() or failure_root.is_symlink():
        failure_metadata = failure_root.lstat()
        require(
            stat.S_ISDIR(failure_metadata.st_mode) and not failure_root.is_symlink(),
            "parent failure evidence root is not an ordinary directory",
        )
        for candidate in failure_root.rglob("*"):
            candidate_metadata = candidate.lstat()
            require(
                not candidate.is_symlink(),
                "parent failure evidence contains a symlink",
            )
            if stat.S_ISDIR(candidate_metadata.st_mode):
                continue
            require(
                stat.S_ISREG(candidate_metadata.st_mode),
                "parent failure evidence contains a special file",
            )
            evidence_paths.add(candidate)

    artifacts_root = parent_root / "artifacts"
    if artifacts_root.exists() or artifacts_root.is_symlink():
        artifacts_metadata = artifacts_root.lstat()
        require(
            stat.S_ISDIR(artifacts_metadata.st_mode)
            and not artifacts_root.is_symlink(),
            "parent artifacts root is not an ordinary directory",
        )
        for candidate in artifacts_root.rglob("*failure*.json"):
            candidate_metadata = candidate.lstat()
            require(
                not candidate.is_symlink(),
                "parent failure evidence contains a symlink",
            )
            require(
                stat.S_ISREG(candidate_metadata.st_mode),
                "parent failure evidence contains a special file",
            )
            evidence_paths.add(candidate)

    classification = "FAILED" if "FAILED" in sentinels else "UNSEALED_FAILED_EVIDENCE"
    return classification, sorted(
        evidence_paths,
        key=lambda candidate: candidate.relative_to(parent_root).as_posix(),
    )


def _live_parent_failure_evidence(parent_root: Path) -> tuple[str, dict[str, Any]]:
    """Rebuild the canonical parent-failure inventory from live bytes."""

    classification, evidence_paths = _parent_failure_path_snapshot(parent_root)
    records = [
        _ordinary_unlinked_file_record(
            path,
            label="parent failure evidence file",
            relative_to=parent_root,
        )
        for path in evidence_paths
    ]
    require(records, "repair parent has no failure evidence")
    classification_after, evidence_paths_after = _parent_failure_path_snapshot(
        parent_root
    )
    require(
        classification_after == classification
        and [path.relative_to(parent_root) for path in evidence_paths_after]
        == [path.relative_to(parent_root) for path in evidence_paths],
        "parent failure evidence inventory changed while it was verified",
    )
    return classification, {
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
        "files_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def verify_parent_run_binding(
    parent_binding: Mapping[str, Any],
    *,
    parent_run_id: str,
    goal_sha256: str,
    canonical_parent: Path = CANONICAL_RUN_PARENT,
) -> dict[str, Any]:
    """Independently revalidate a repair parent's manifest and failure bytes."""

    require(isinstance(parent_binding, Mapping), "MK0 parent binding is absent")
    canonical_root = canonical_parent.resolve(strict=True)
    parent_root = canonical_root / parent_run_id
    try:
        root_metadata = parent_root.lstat()
    except FileNotFoundError as error:
        raise FinalizeFailure("parent run root is absent") from error
    require(
        stat.S_ISDIR(root_metadata.st_mode) and not parent_root.is_symlink(),
        "parent run root is not an ordinary directory",
    )
    require(
        parent_root.resolve(strict=True) == parent_root,
        "parent run root is not canonical",
    )

    manifest_path = parent_root / "run_manifest.json"
    manifest_record, manifest_bytes = _ordinary_unlinked_file_snapshot(
        manifest_path,
        label="parent run registration manifest",
    )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizeFailure(
            "parent run registration manifest is invalid JSON"
        ) from error
    require(
        isinstance(manifest, dict),
        "parent run registration manifest is not a JSON object",
    )
    require(
        manifest.get("schema_version") == "mk0_run_manifest_v3",
        "parent run manifest schema drift",
    )
    require(manifest.get("run_id") == parent_run_id, "parent run manifest ID drift")
    require(manifest.get("task_id") == "MK0-01", "parent run manifest task drift")
    require(manifest.get("phase") == "MK0", "parent run manifest phase drift")
    declared_root = Path(str(manifest.get("run_root", "")))
    require(
        declared_root.is_absolute()
        and declared_root == parent_root
        and declared_root.resolve(strict=True) == parent_root,
        "parent run manifest root drift",
    )
    require(
        manifest.get("goal_sha256") == goal_sha256
        and isinstance(manifest.get("contract"), Mapping)
        and manifest["contract"].get("sha256") == goal_sha256,
        "parent run manifest Goal drift",
    )
    parent_commit = manifest.get("implementation_commit")
    require(
        isinstance(parent_commit, str) and HEX40.fullmatch(parent_commit) is not None,
        "parent run implementation commit is invalid",
    )
    require(
        isinstance(manifest.get("code"), Mapping)
        and manifest["code"].get("commit") == parent_commit,
        "parent run code binding drift",
    )
    require(
        isinstance(manifest.get("source_binding"), Mapping)
        and manifest["source_binding"].get("git_commit") == parent_commit,
        "parent run source binding drift",
    )
    parent_match = FORMAL_RUN_ID.fullmatch(parent_run_id)
    require(parent_match is not None, "parent run ID is not formal")
    require(
        parent_commit.startswith(parent_match.group("short_sha")),
        "parent run ID short SHA differs from implementation commit",
    )

    classification, failure_evidence = _live_parent_failure_evidence(parent_root)
    root_after = parent_root.lstat()
    require(
        not parent_root.is_symlink()
        and all(
            getattr(root_metadata, field) == getattr(root_after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ),
        "parent run root identity changed while lineage was verified",
    )
    live_binding = {
        "schema_version": "mk0_parent_run_binding_v1",
        "run_id": parent_run_id,
        "run_root": str(parent_root),
        "registration_manifest": manifest_record,
        "observed_classification": classification,
        "failure_evidence": failure_evidence,
    }
    require(
        canonical_json_bytes(dict(parent_binding))
        == canonical_json_bytes(live_binding),
        "MK0 parent binding differs from live parent manifest/failure bytes",
    )
    return live_binding


def write_new(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _git_text(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_current_source_binding(
    expected_commit: str, reported: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-hash the exact tracked-source inventory emitted by the CPU runner."""

    require(
        HEX40.fullmatch(expected_commit) is not None, "invalid implementation commit"
    )
    require(isinstance(reported, Mapping), "source binding is absent")
    head = _git_text(["rev-parse", "HEAD"])
    require(head == expected_commit, "finalizer HEAD/implementation commit mismatch")
    status = _git_text(["status", "--porcelain=v1"])
    require(not status, "finalization requires a clean implementation worktree")
    tree = _git_text(["rev-parse", "HEAD^{tree}"])
    require(
        reported.get("repo_root") == str(REPO_ROOT), "source repo-root substitution"
    )
    require(reported.get("git_commit") == head, "source commit drift")
    require(reported.get("git_tree") == tree, "source tree drift")
    require(reported.get("git_status_porcelain") == "", "runner source was not clean")
    files = reported.get("tracked_source_files")
    require(isinstance(files, dict) and files, "source file-hash inventory is absent")
    observed: dict[str, str] = {}
    for relative, expected_sha in sorted(files.items()):
        require(isinstance(relative, str) and relative, "invalid source inventory path")
        path = (REPO_ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as error:
            raise FinalizeFailure(
                f"source inventory path escaped repository: {relative}"
            ) from error
        require(path.is_file(), f"source inventory entry is not a file: {relative}")
        require(HEX64.fullmatch(str(expected_sha)) is not None, "invalid source digest")
        digest = sha256_file(path)
        require(digest == expected_sha, f"source byte drift: {relative}")
        observed[relative] = digest
    inventory_sha = hashlib.sha256(canonical_json_bytes(observed)).hexdigest()
    require(
        reported.get("tracked_source_file_count") == len(observed),
        "source file-count drift",
    )
    require(
        reported.get("tracked_source_files_sha256") == inventory_sha,
        "source inventory digest drift",
    )
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": head,
        "git_tree": tree,
        "git_status_porcelain": "",
        "tracked_source_file_count": len(observed),
        "tracked_source_files_sha256": inventory_sha,
        "tracked_source_files": observed,
    }


def _ast_string_constant(
    path: Path,
    name: str,
    *,
    source_bytes: bytes | None = None,
) -> str:
    """Read a literal module string without importing or executing the module."""

    try:
        tree = ast.parse(
            path.read_bytes() if source_bytes is None else source_bytes,
            filename=str(path),
        )
    except (OSError, SyntaxError) as error:
        raise FinalizeFailure(f"cannot parse source constant: {path}:{name}") from error
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
        and target.id == name
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    require(len(values) == 1, f"source constant is not unique: {path}:{name}")
    return values[0]


def _pytest_junit_tree(data: bytes) -> ET.Element:
    """Parse the exact already-hashed JUnit bytes."""

    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise FinalizeFailure("pytest JUnit evidence is invalid") from error


def _pytest_junit_totals(data: bytes) -> dict[str, int]:
    """Independently derive pytest totals from leaf JUnit test suites."""

    root = _pytest_junit_tree(data)

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    suites = [
        element
        for element in root.iter()
        if local_name(element) == "testsuite"
        and not any(local_name(child) == "testsuite" for child in element)
    ]
    require(bool(suites), "pytest JUnit contains no leaf test suite")
    totals = {name: 0 for name in ("tests", "errors", "failures", "skipped")}
    for suite in suites:
        for name in totals:
            raw = suite.get(name)
            try:
                value = int(raw) if raw is not None else None
            except ValueError as error:
                raise FinalizeFailure(f"pytest JUnit {name} is invalid") from error
            require(
                value is not None and value >= 0,
                f"pytest JUnit {name} is invalid",
            )
            totals[name] += value
    passed = totals["tests"] - totals["errors"] - totals["failures"] - totals["skipped"]
    require(passed >= 0, "pytest JUnit totals are inconsistent")
    return {**totals, "passed": passed}


def _pytest_junit_identities(data: bytes) -> list[tuple[str, str]]:
    """Extract canonical (classname, test name) identities from JUnit bytes."""

    root = _pytest_junit_tree(data)

    def local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    identities: list[tuple[str, str]] = []
    for element in root.iter():
        if local_name(element) != "testcase":
            continue
        classname = element.get("classname")
        name = element.get("name")
        require(
            isinstance(classname, str)
            and bool(classname)
            and isinstance(name, str)
            and bool(name),
            "pytest JUnit testcase identity is invalid",
        )
        identities.append((classname, name))
    require(bool(identities), "pytest JUnit contains no testcase identity")
    return sorted(identities)


def _nodeid_junit_identity(nodeid: str) -> tuple[str, str]:
    """Map a standard pytest Python nodeid to its JUnit identity."""

    parts = nodeid.split("::")
    require(len(parts) >= 2, f"pytest nodeid has no test selector: {nodeid}")
    relative = parts[0]
    require(
        relative.startswith("tests/mk0/")
        and relative.endswith(".py")
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"pytest nodeid escaped the complete MK0 domain: {nodeid}",
    )
    source = (REPO_ROOT / relative).resolve(strict=True)
    try:
        source.relative_to(REPO_ROOT / "tests" / "mk0")
    except ValueError as error:
        raise FinalizeFailure(f"pytest nodeid escaped MK0 tests: {nodeid}") from error
    require(source.is_file(), f"pytest nodeid source is absent: {nodeid}")
    classname = relative[:-3].replace("/", ".")
    if len(parts) > 2:
        classname += "." + ".".join(parts[1:-1])
    return classname, parts[-1]


def _json_markers(stdout: str, marker: str) -> list[dict[str, Any]]:
    raw_records = []
    for line in stdout.splitlines():
        index = line.find(marker)
        if index >= 0:
            raw_records.append(line[index + len(marker) :])
    records: list[dict[str, Any]] = []
    for raw in raw_records:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise FinalizeFailure(
                f"pytest evidence emitted invalid {marker} JSON"
            ) from error
        require(isinstance(value, dict), f"pytest {marker} is not an object")
        records.append(value)
    return records


def _unique_json_marker(stdout: str, marker: str) -> dict[str, Any]:
    records = _json_markers(stdout, marker)
    require(len(records) == 1, f"independent pytest collection lacks unique {marker}")
    return records[0]


def _independent_pytest_collection(
    *,
    run_root: Path,
    python_executable: str,
    bootstrap: str,
    expected_nodeids: list[str],
    expected_pytest_version: str,
    helper_path: Path,
    helper_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Re-collect current source-bound tests outside the formal run tree."""

    command = [
        python_executable,
        "-c",
        bootstrap,
        "-q",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "tests/mk0",
    ]
    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    require(
        temporary_parent != run_root and run_root not in temporary_parent.parents,
        "independent pytest temporary root entered the formal run tree",
    )
    external_removed = False
    with tempfile.TemporaryDirectory(
        prefix="mk0-finalizer-collect-",
        dir=temporary_parent,
    ) as temporary_text:
        import_root = Path(temporary_text).resolve(strict=True)
        binding = import_root / "mrna_editflow"
        os.symlink(REPO_ROOT, binding, target_is_directory=True)
        require(
            binding.is_symlink() and binding.resolve(strict=True) == REPO_ROOT,
            "independent pytest import binding is invalid",
        )
        environment = os.environ.copy()
        for key in tuple(environment):
            if key.startswith("PYTEST_") or key.startswith("PYTHON"):
                environment.pop(key, None)
        environment.update(
            {
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONPATH": str(import_root),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "MK0_EXPECTED_PACKAGE_INIT": str(REPO_ROOT / "__init__.py"),
                "MK0_EXPECTED_PACKAGE_ROOT": str(REPO_ROOT),
                "MK0_PYTEST_MODE": "collect",
            }
        )
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise FinalizeFailure(
                "independent pytest collection failed to run"
            ) from error
    external_removed = not Path(temporary_text).exists()
    log_bytes = (
        "=== FINALIZER INDEPENDENT PYTEST COLLECT STDOUT ===\n"
        + completed.stdout
        + "\n=== FINALIZER INDEPENDENT PYTEST COLLECT STDERR ===\n"
        + completed.stderr
    ).encode("utf-8")
    require(completed.returncode == 0, "independent pytest collection returned nonzero")
    origin = _unique_json_marker(completed.stdout, "__MK0_BOUND_MODULE_ORIGIN__=")
    audit = _unique_json_marker(completed.stdout, "__MK0_PYTEST_AUDIT__=")
    require(
        origin.get("matches_current_worktree") is True
        and origin.get("resolved_init") == str(REPO_ROOT / "__init__.py")
        and origin.get("resolved_search_locations") == [str(REPO_ROOT)]
        and origin.get("strict_importer_path") == str(helper_path)
        and origin.get("strict_importer_sha256") == helper_sha256
        and origin.get("strict_importer_loaded_from_source_bytes") is True,
        "independent pytest source-byte origin drift",
    )
    nodeids = audit.get("nodeids")
    require(
        audit.get("schema_version") == "mk0_pytest_audit_v1"
        and audit.get("mode") == "collect"
        and audit.get("exitstatus") == 0
        and audit.get("pytest_version") == expected_pytest_version
        and isinstance(nodeids, list)
        and nodeids == sorted(nodeids)
        and len(nodeids) == len(set(nodeids))
        and audit.get("collected_count") == len(nodeids)
        and audit.get("deselected_count") == 0
        and audit.get("xfailed_count") == 0
        and audit.get("xpassed_count") == 0,
        "independent pytest collection audit drift",
    )
    require(
        nodeids == expected_nodeids,
        "independent pytest collection differs from CPU collect/execute inventory",
    )
    nodeids_sha256 = hashlib.sha256(
        "".join(f"{nodeid}\n" for nodeid in nodeids).encode("utf-8")
    ).hexdigest()
    require(external_removed, "independent pytest import root was not removed")
    return (
        {
            "schema_version": "mk0_finalizer_pytest_collection_v1",
            "status": "PASS",
            "collection_method": "fresh_source_bound_sanitized_collect_only",
            "python_executable": python_executable,
            "command": command,
            "pytest_version": expected_pytest_version,
            "collected_count": len(nodeids),
            "nodeids_sha256": nodeids_sha256,
            "cpu_inventory_equal": True,
            "deselected_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
            "external_import_root_removed": True,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
            "source_origin": origin,
        },
        log_bytes,
    )


def verify_cpu_pytest_evidence(
    cpu: Mapping[str, Any],
    run_root: Path,
    *,
    source_binding: Mapping[str, Any],
    expected_python_executable: str,
) -> dict[str, Any]:
    """Independently revalidate the CPU launcher's full pytest-v2 evidence."""

    report = cpu.get("pytest")
    require(isinstance(report, Mapping), "CPU pytest evidence is absent")
    report = dict(report)
    require(
        report.get("schema_version") == "mk0_bound_pytest_report_v2",
        "CPU pytest report schema drift",
    )
    require(
        report.get("status") == "PASS"
        and report.get("returncode") == 0
        and report.get("collection_returncode") == 0
        and report.get("pytest_returncode") == 0
        and report.get("execution_started") is True,
        "CPU pytest did not complete both formal phases",
    )
    require(report.get("pytest_args") == ["tests/mk0"], "CPU pytest domain drift")
    require(
        report.get("repo_root") == str(REPO_ROOT)
        and report.get("formal_output_root") == str(run_root),
        "CPU pytest repository/output binding drift",
    )
    python_executable = Path(str(report.get("python_executable", ""))).resolve(
        strict=True
    )
    require(
        python_executable == Path(expected_python_executable).resolve(strict=True),
        "CPU pytest Python executable drift",
    )
    require(
        isinstance(report.get("pytest_version"), str)
        and bool(report["pytest_version"]),
        "CPU pytest version binding is absent",
    )

    persisted_path = run_root / "provenance" / "pytest_import_binding.json"
    persisted_record, persisted_bytes = _ordinary_unlinked_file_snapshot(
        persisted_path,
        label="CPU pytest persisted report",
    )
    try:
        persisted = json.loads(persisted_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizeFailure("CPU pytest persisted report is invalid JSON") from error
    require(
        isinstance(persisted, dict),
        "CPU pytest persisted report is not a JSON object",
    )
    extension_keys = {
        "launcher_path",
        "launcher_sha256",
        "report_path",
        "report_sha256",
        "log_path",
        "log_sha256",
        "junit_path",
        "junit_sha256",
    }
    require(
        extension_keys.isdisjoint(persisted),
        "CPU pytest persisted report contains runner-only fields",
    )
    require(
        {key: value for key, value in report.items() if key not in extension_keys}
        == persisted,
        "CPU pytest embedded/persisted report drift",
    )
    require(
        report.get("report_path") == str(persisted_path)
        and report.get("report_sha256") == persisted_record["sha256"],
        "CPU pytest persisted-report byte binding drift",
    )

    tracked_files = source_binding.get("tracked_source_files")
    require(isinstance(tracked_files, Mapping), "tracked source inventory is absent")
    launcher_relative = "scripts/mk0/run_bound_pytest.py"
    helper_relative = "scripts/mk0/strict_worktree_import.py"
    launcher_path = REPO_ROOT / launcher_relative
    helper_path = REPO_ROOT / helper_relative
    launcher_record, launcher_bytes = _ordinary_unlinked_file_snapshot(
        launcher_path,
        label="CPU pytest launcher source",
    )
    helper_record, _helper_bytes = _ordinary_unlinked_file_snapshot(
        helper_path,
        label="CPU pytest strict importer source",
    )
    require(
        tracked_files.get(launcher_relative) == launcher_record["sha256"]
        and tracked_files.get(helper_relative) == helper_record["sha256"],
        "CPU pytest launcher/importer is absent from source binding",
    )
    require(
        report.get("launcher_path") == str(launcher_path)
        and report.get("launcher_sha256") == launcher_record["sha256"],
        "CPU pytest launcher byte binding drift",
    )

    collection_nodeids = report.get("collection_nodeids")
    execution_nodeids = report.get("execution_nodeids")
    require(
        isinstance(collection_nodeids, list)
        and collection_nodeids
        and all(isinstance(nodeid, str) and nodeid for nodeid in collection_nodeids)
        and collection_nodeids == sorted(collection_nodeids)
        and len(collection_nodeids) == len(set(collection_nodeids))
        and execution_nodeids == collection_nodeids,
        "CPU pytest collect/execute nodeid inventory drift",
    )
    nodeids_sha256 = hashlib.sha256(
        "".join(f"{nodeid}\n" for nodeid in collection_nodeids).encode("utf-8")
    ).hexdigest()
    require(
        report.get("collection_nodeids_sha256") == nodeids_sha256
        and report.get("execution_nodeids_sha256") == nodeids_sha256,
        "CPU pytest nodeid inventory digest drift",
    )
    collected = report.get("collected_count")
    require(
        isinstance(collected, int)
        and not isinstance(collected, bool)
        and collected > 0
        and collected == len(collection_nodeids)
        and report.get("executed_count") == collected
        and report.get("passed_count") == collected,
        "CPU pytest requires collected == executed == passed > 0",
    )
    for field in (
        "failed_count",
        "error_count",
        "skipped_count",
        "deselected_count",
        "xfailed_count",
        "xpassed_count",
    ):
        require(report.get(field) == 0, f"CPU pytest formal {field} is nonzero")
    require(report.get("contract_violations") == [], "CPU pytest contract violation")
    expected_junit_identities = sorted(
        _nodeid_junit_identity(nodeid) for nodeid in collection_nodeids
    )
    for relative in sorted({nodeid.split("::", 1)[0] for nodeid in collection_nodeids}):
        test_record = _ordinary_unlinked_file_record(
            REPO_ROOT / relative,
            label=f"CPU pytest collected source: {relative}",
        )
        require(
            tracked_files.get(relative) == test_record["sha256"],
            f"CPU pytest collected source is absent from source binding: {relative}",
        )

    environment = report.get("environment_contract")
    require(
        isinstance(environment, Mapping)
        and environment.get("pytest_plugin_autoload_disabled") is True
        and environment.get("pythonpath_replaced_with_external_binding") is True,
        "CPU pytest environment was not sanitized",
    )
    require(
        set(environment.get("controlled_environment_keys", []))
        == {
            "MK0_EXPECTED_PACKAGE_INIT",
            "MK0_EXPECTED_PACKAGE_ROOT",
            "MK0_PYTEST_MODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
        },
        "CPU pytest controlled environment inventory drift",
    )
    for field, prefix in (
        ("sanitized_pytest_environment_keys", "PYTEST_"),
        ("sanitized_python_environment_keys", "PYTHON"),
    ):
        keys = environment.get(field)
        require(
            isinstance(keys, list)
            and keys == sorted(set(keys))
            and all(isinstance(key, str) and key.startswith(prefix) for key in keys),
            f"CPU pytest sanitized environment inventory drift: {field}",
        )

    for field in (
        "module_origin",
        "collection_module_origin",
        "execution_module_origin",
    ):
        origin = report.get(field)
        require(
            isinstance(origin, Mapping)
            and origin.get("matches_current_worktree") is True
            and origin.get("resolved_init") == str(REPO_ROOT / "__init__.py")
            and origin.get("resolved_search_locations") == [str(REPO_ROOT)]
            and origin.get("expected_init") == str(REPO_ROOT / "__init__.py")
            and origin.get("expected_root") == str(REPO_ROOT)
            and origin.get("strict_importer_path") == str(helper_path)
            and origin.get("strict_importer_sha256") == helper_record["sha256"]
            and origin.get("strict_importer_loaded_from_source_bytes") is True,
            f"CPU pytest source-byte import binding drift: {field}",
        )
    isolation = report.get("import_isolation")
    require(
        isinstance(isolation, Mapping)
        and isolation.get("method") == "external_ephemeral_symlink"
        and isolation.get("resolved_target") == str(REPO_ROOT)
        and isolation.get("inside_formal_output_tree") is False
        and isolation.get("external_import_root_removed") is True
        and isolation.get("ambient_pythonpath_replaced") is True,
        "CPU pytest import isolation drift",
    )

    junit_path = run_root / "evaluation" / "pytest_mk0.junit.xml"
    log_path = run_root / "logs" / "pytest_mk0.log"
    junit_record, junit_bytes = _ordinary_unlinked_file_snapshot(
        junit_path,
        label="CPU pytest JUnit evidence",
    )
    log_record, log_bytes = _ordinary_unlinked_file_snapshot(
        log_path,
        label="CPU pytest log evidence",
    )
    junit = report.get("junit")
    log = report.get("log")
    expected_totals = {
        "tests": collected,
        "errors": 0,
        "failures": 0,
        "skipped": 0,
        "passed": collected,
    }
    require(
        isinstance(junit, Mapping)
        and junit.get("path") == str(junit_path)
        and junit.get("exists") is True
        and junit.get("sha256") == junit_record["sha256"]
        and junit.get("totals") == expected_totals
        and _pytest_junit_totals(junit_bytes) == expected_totals
        and _pytest_junit_identities(junit_bytes) == expected_junit_identities,
        "CPU pytest JUnit binding drift",
    )
    require(
        isinstance(log, Mapping)
        and log.get("path") == str(log_path)
        and log.get("sha256") == log_record["sha256"],
        "CPU pytest log binding drift",
    )
    require(
        report.get("junit_path") == str(junit_path)
        and report.get("junit_sha256") == junit_record["sha256"]
        and report.get("log_path") == str(log_path)
        and report.get("log_sha256") == log_record["sha256"],
        "CPU pytest runner evidence aliases drift",
    )
    try:
        log_text = log_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FinalizeFailure("CPU pytest log is not UTF-8") from error
    log_origins = _json_markers(log_text, "__MK0_BOUND_MODULE_ORIGIN__=")
    log_audits = _json_markers(log_text, "__MK0_PYTEST_AUDIT__=")
    require(
        log_origins
        == [report["collection_module_origin"], report["execution_module_origin"]],
        "CPU pytest log origin markers drift",
    )
    expected_log_audits = [
        {
            "schema_version": "mk0_pytest_audit_v1",
            "mode": "collect",
            "pytest_version": report["pytest_version"],
            "exitstatus": 0,
            "nodeids": collection_nodeids,
            "collected_count": collected,
            "deselected_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
        },
        {
            "schema_version": "mk0_pytest_audit_v1",
            "mode": "execute",
            "pytest_version": report["pytest_version"],
            "exitstatus": 0,
            "nodeids": collection_nodeids,
            "collected_count": collected,
            "deselected_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
        },
    ]
    require(log_audits == expected_log_audits, "CPU pytest log audit markers drift")
    require(
        report.get("formal_output_tree_regular_only") is True,
        "CPU pytest regular-tree certificate is absent",
    )

    bootstrap = _ast_string_constant(
        launcher_path,
        "BOOTSTRAP",
        source_bytes=launcher_bytes,
    )
    expected_collect_command = [
        expected_python_executable,
        "-c",
        bootstrap,
        "-q",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "tests/mk0",
    ]
    expected_command = [
        expected_python_executable,
        "-c",
        bootstrap,
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/mk0",
        f"--junitxml={junit_path}",
    ]
    require(
        report.get("collect_command") == expected_collect_command
        and report.get("command") == expected_command,
        "CPU pytest exact subprocess command drift",
    )
    independent_collection, independent_log_bytes = _independent_pytest_collection(
        run_root=run_root,
        python_executable=expected_python_executable,
        bootstrap=bootstrap,
        expected_nodeids=collection_nodeids,
        expected_pytest_version=report["pytest_version"],
        helper_path=helper_path,
        helper_sha256=helper_record["sha256"],
    )
    independent_log_path = run_root / "logs" / "finalizer_pytest_collect.log"
    independent_log_sha256 = write_bytes_exclusive_atomic(
        independent_log_path,
        independent_log_bytes,
    )
    independent_collection["log"] = {
        "path": str(independent_log_path),
        "sha256": independent_log_sha256,
    }
    independent_report_path = (
        run_root / "provenance" / "finalizer_pytest_collection.json"
    )
    independent_report_sha256 = write_json_exclusive_atomic(
        independent_report_path,
        independent_collection,
    )
    return {
        "schema_version": report["schema_version"],
        "pytest_version": report["pytest_version"],
        "collected_count": collected,
        "executed_count": collected,
        "passed_count": collected,
        "nodeids_sha256": nodeids_sha256,
        "persisted_report_sha256": persisted_record["sha256"],
        "junit_sha256": junit_record["sha256"],
        "log_sha256": log_record["sha256"],
        "launcher_sha256": launcher_record["sha256"],
        "strict_importer_sha256": helper_record["sha256"],
        "independent_collection": {
            "path": str(independent_report_path),
            "sha256": independent_report_sha256,
            "log_path": str(independent_log_path),
            "log_sha256": independent_log_sha256,
            "collected_count": independent_collection["collected_count"],
            "nodeids_sha256": independent_collection["nodeids_sha256"],
            "cpu_inventory_equal": True,
        },
        "all_forbidden_outcome_counts_zero": True,
        "collect_execute_inventory_equal": True,
    }


def verify_checksum_ledger(root: Path, ledger_path: Path) -> dict[str, Any]:
    """Verify every ledger entry, rejecting duplicates and path escapes."""

    root = root.resolve(strict=True)
    ledger = ledger_path.resolve(strict=True)
    require(ledger.is_file(), f"checksum ledger is missing: {ledger}")
    try:
        ledger.relative_to(root)
    except ValueError as error:
        raise FinalizeFailure("checksum ledger escaped its declared root") from error
    seen: set[str] = set()
    seen_targets: set[Path] = set()
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(), start=1
    ):
        require(bool(line), f"blank checksum-ledger line {line_number}")
        match = LEDGER_LINE.fullmatch(line)
        require(match is not None, f"malformed checksum-ledger line {line_number}")
        expected_sha, relative = match.groups()
        require(relative not in seen, f"duplicate checksum-ledger path: {relative}")
        seen.add(relative)
        target = (root / relative).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise FinalizeFailure(
                f"checksum-ledger path escaped root: {relative}"
            ) from error
        require(
            target not in seen_targets, f"duplicate checksum-ledger target: {relative}"
        )
        seen_targets.add(target)
        require(target.is_file(), f"checksum-ledger target is not a file: {relative}")
        require(sha256_file(target) == expected_sha, f"checksum mismatch: {relative}")
    require(seen, "checksum ledger is empty")
    return {
        "path": str(ledger),
        "sha256": sha256_file(ledger),
        "verified_entry_count": len(seen),
    }


def _verify_declared_file(binding: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(binding, Mapping), "declared file binding is absent")
    return verify_bound_file(
        binding["path"],
        expected_path=binding["path"],
        expected_sha256=binding["sha256"],
        expected_size_bytes=int(binding["size_bytes"]),
    )


def verify_fm0_b0_d1(
    fm0_root: Path,
    *,
    goal_sha256: str,
    implementation_commit: str,
    d1_data: Path,
    d1_ledger: Path,
) -> dict[str, Any]:
    """Re-verify terminal FM0, its current B0 gate and the actual D1 bytes."""

    root = fm0_root.resolve(strict=True)
    require(root.is_dir(), "FM0 closure root is missing")
    ledger = verify_checksum_ledger(root, root / "artifact_checksums.sha256")
    required = {
        "run_manifest.json",
        "status.json",
        "summary.json",
        "evaluation/fm0_acceptance.json",
        FM0_HASH_LICENSE_RELATIVE.as_posix(),
        "provenance/b0_current_gate.json",
        "provenance/data_binding.json",
        "PASS_READY_FOR_MK0",
    }
    ledger_paths = {
        match.group(2)
        for line in (root / "artifact_checksums.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
        if (match := LEDGER_LINE.fullmatch(line)) is not None
    }
    require(
        required <= ledger_paths,
        "FM0 ledger omits a required terminal/prerequisite file",
    )
    manifest = read_json(root / "run_manifest.json")
    status = read_json(root / "status.json")
    summary = read_json(root / "summary.json")
    acceptance = read_json(root / "evaluation" / "fm0_acceptance.json")
    b0 = read_json(root / "provenance" / "b0_current_gate.json")
    data_binding = read_json(root / "provenance" / "data_binding.json")
    run_id = manifest.get("run_id")
    require(bool(run_id), "FM0 run ID is absent")
    for name, report in (
        ("status", status),
        ("summary", summary),
        ("acceptance", acceptance),
    ):
        require(report.get("run_id") == run_id, f"FM0 {name} run ID drift")
    require(
        manifest.get("phase_id") == "FM0" and manifest.get("task_id") == "FM0-01",
        "FM0 phase identity drift",
    )
    require(
        manifest.get("formal_fm0_pass") is True, "FM0 manifest is not a formal pass"
    )
    require(manifest.get("ready_for_mk0") is True, "FM0 manifest is not ready for MK0")
    require(manifest.get("state") == "PASS_READY_FOR_MK0", "FM0 terminal state drift")
    require(manifest.get("exit_code") == 0, "FM0 exit code is nonzero")
    require(
        manifest.get("cpu_fallback_count") == 0
        and manifest.get("gpu_execution") is True,
        "FM0 GPU binding failed",
    )
    require(manifest.get("formal_stop_reasons") == [], "FM0 has formal stop reasons")
    require(
        manifest.get("goal_contract", {}).get("sha256") == goal_sha256,
        "FM0 Goal hash drift",
    )
    safety = manifest.get("safety", {})
    require(
        safety.get("downstream_phases_started") == [], "FM0 started a downstream phase"
    )
    require(
        safety.get("unrelated_processes_terminated") is False,
        "FM0 terminated unrelated work",
    )
    require(
        safety.get("existing_results_overwritten") is False, "FM0 overwrote results"
    )
    require(
        status.get("state") == "PASS_READY_FOR_MK0", "FM0 status is not terminal pass"
    )
    require(
        status.get("formal_fm0_pass") is True and status.get("ready_for_mk0") is True,
        "FM0 status pass fields failed",
    )
    require(
        status.get("validation_exit_code") == 0
        and status.get("cpu_fallback_count") == 0,
        "FM0 status validation/GPU failed",
    )
    require(
        status.get("downstream_phases_started") == [],
        "FM0 status records downstream work",
    )
    require(
        summary.get("formal_fm0_pass") is True and summary.get("ready_for_mk0") is True,
        "FM0 summary failed",
    )
    require(
        summary.get("next_phase") == "MK0"
        and summary.get("next_phase_started") is False,
        "FM0 summary phase boundary drift",
    )
    require(
        acceptance.get("formal_fm0_pass") is True
        and acceptance.get("ready_for_mk0") is True,
        "FM0 acceptance failed",
    )
    require(
        acceptance.get("formal_phase_state") == "PASS_READY_FOR_MK0",
        "FM0 acceptance state drift",
    )
    require(
        acceptance.get("formal_stop_reasons") == [], "FM0 acceptance has stop reasons"
    )
    require(
        acceptance.get("contract", {}).get("sha256") == goal_sha256,
        "FM0 acceptance contract drift",
    )
    require(
        acceptance.get("b0_bound_gate", {}).get("strict_unlock") is True,
        "FM0 did not strictly unlock B0",
    )
    verify_fm0_ready_marker(root / "PASS_READY_FOR_MK0")

    require(
        b0.get("schema_version") == "b0-current-gate/v4", "B0 current gate schema drift"
    )
    require(
        b0.get("current_contract_unlock") is True, "B0 current-contract unlock failed"
    )
    require(
        b0.get("contract", {}).get("sha256") == goal_sha256, "B0 contract hash drift"
    )
    hard = b0.get("hard_gate_status")
    require(
        isinstance(hard, dict)
        and hard
        and all(value is True for value in hard.values()),
        "B0 hard gate failed",
    )
    for phase in ("b0_02", "b0_03", "b0_04", "b0_05"):
        criteria = b0.get(phase, {}).get("acceptance")
        require(
            isinstance(criteria, dict)
            and criteria
            and all(value is True for value in criteria.values()),
            f"{phase} acceptance failed",
        )
        _verify_declared_file(b0[phase]["report"])
    declared_splits = data_binding.get("split_manifests")
    require(isinstance(declared_splits, dict), "FM0 split-manifest bindings are absent")
    for split in b0["b0_02"]["splits"].values():
        require(split.get("pass") is True, "B0-02 split gate failed")
        declared = declared_splits.get(Path(split["manifest_path"]).name)
        require(
            isinstance(declared, dict), "B0 split lacks an absolute FM0 data binding"
        )
        require(
            declared.get("sha256") == split["manifest_sha256"],
            "B0 split hash binding drift",
        )
        _verify_declared_file(declared)
    for split in b0["b0_03"]["splits"].values():
        require(
            split.get("hard_gate_pass") is True and split.get("split_pass") is True,
            "B0-03 split gate failed",
        )
        require(split.get("unclassified") == 0, "B0-03 has unclassified test records")
    foundation_manifest = b0["b0_03"]["foundation_manifest_binding"]
    verify_bound_file(
        foundation_manifest["path"],
        expected_path=foundation_manifest["path"],
        expected_sha256=foundation_manifest["sha256"],
    )

    checkpoint_manifest_path = root / FM0_HASH_LICENSE_RELATIVE
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    require(
        checkpoint_manifest.get("task_id") == "FM0-01"
        and checkpoint_manifest.get("manifest_kind")
        == "foundation_checkpoint_hash_license",
        "FM0 checkpoint hash/license manifest identity drift",
    )
    expected_files = checkpoint_manifest.get("expected_files")
    checkpoint_files = checkpoint_manifest.get("files")
    require(
        isinstance(expected_files, list) and expected_files,
        "FM0 checkpoint expected-file inventory is empty",
    )
    require(
        isinstance(checkpoint_files, list) and checkpoint_files,
        "FM0 checkpoint file bindings are empty",
    )
    normalized_checkpoint_files: list[dict[str, Any]] = []
    checkpoint_names: set[str] = set()
    for record in checkpoint_files:
        require(isinstance(record, Mapping), "FM0 checkpoint file record is invalid")
        filename = record.get("filename")
        size_bytes = record.get("size_bytes")
        sha256 = record.get("sha256")
        require(
            isinstance(filename, str)
            and bool(filename)
            and Path(filename).name == filename
            and filename not in checkpoint_names,
            "FM0 checkpoint filename is invalid or duplicated",
        )
        require(_is_nonnegative_int(size_bytes), "FM0 checkpoint file size is invalid")
        require(
            HEX64.fullmatch(str(sha256)) is not None,
            "FM0 checkpoint file hash is invalid",
        )
        checkpoint_names.add(filename)
        normalized_checkpoint_files.append(
            {"path": filename, "size_bytes": size_bytes, "sha256": sha256}
        )
    normalized_checkpoint_files.sort(key=lambda record: record["path"])
    require(
        set(expected_files) == checkpoint_names
        and len(expected_files) == len(checkpoint_names),
        "FM0 checkpoint expected-file list differs from its hash bindings",
    )
    require(
        checkpoint_manifest.get("missing_files") == [],
        "FM0 checkpoint manifest records missing files",
    )
    checkpoint_license = checkpoint_manifest.get("license")
    require(
        isinstance(checkpoint_license, Mapping),
        "FM0 checkpoint license binding is missing",
    )
    require(
        str(checkpoint_license.get("type", "")).lower() == "agpl-3.0",
        "FM0 checkpoint license type drift",
    )
    license_file = next(
        (
            record
            for record in normalized_checkpoint_files
            if record["path"] == "license.md"
        ),
        None,
    )
    require(license_file is not None, "FM0 checkpoint inventory omits license.md")
    require(
        checkpoint_license.get("license_md_sha256") == license_file["sha256"]
        and checkpoint_license.get("license_md_size") == license_file["size_bytes"],
        "FM0 license semantics differ from license.md bytes",
    )

    items = data_binding.get("items")
    require(isinstance(items, dict), "FM0 data binding items are absent")
    canonical = items["canonical_records"]
    exposure = items["exposure_ledger"]
    d1_data_evidence = verify_bound_file(
        d1_data,
        expected_path=canonical["path"],
        expected_sha256=canonical["sha256"],
        expected_size_bytes=int(canonical["size_bytes"]),
    )
    d1_ledger_evidence = verify_bound_file(
        d1_ledger,
        expected_path=exposure["path"],
        expected_sha256=exposure["sha256"],
        expected_size_bytes=int(exposure["size_bytes"]),
    )
    fm0_commit = acceptance.get("final_repository_commit")
    require(HEX40.fullmatch(str(fm0_commit)) is not None, "FM0 final commit is invalid")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", fm0_commit, implementation_commit],
        cwd=REPO_ROOT,
    )
    require(
        ancestor.returncode == 0, "MK0 implementation is not descended from FM0 closure"
    )
    return {
        "root": str(root),
        "run_id": run_id,
        "checksum_ledger": ledger,
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "acceptance_sha256": sha256_file(root / "evaluation" / "fm0_acceptance.json"),
        "b0_current_gate_sha256": sha256_file(
            root / "provenance" / "b0_current_gate.json"
        ),
        "data_binding_sha256": sha256_file(root / "provenance" / "data_binding.json"),
        "final_repository_commit": fm0_commit,
        "d1_canonical_records": d1_data_evidence,
        "d1_exposure_ledger": d1_ledger_evidence,
        "foundation_checkpoint_manifest": {
            "path": str(checkpoint_manifest_path),
            "sha256": sha256_file(checkpoint_manifest_path),
            "model_id": checkpoint_manifest.get("model_id"),
            "revision": checkpoint_manifest.get("revision"),
            "snapshot_dir": checkpoint_manifest.get("snapshot_dir"),
            "file_count": len(normalized_checkpoint_files),
            "files": normalized_checkpoint_files,
            "files_sha256": _canonical_sha256(normalized_checkpoint_files),
            "license": {
                "type": "agpl-3.0",
                "license_md_sha256": license_file["sha256"],
                "license_md_size": license_file["size_bytes"],
            },
        },
    }


def verify_preflight(
    path: Path,
    *,
    run_id: str,
    parent_run_id: str | None,
    goal_sha256: str,
    implementation_commit: str,
    fm0: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    report = read_json(resolved)
    require(
        report.get("schema_version") == "mk0_preflight_v1", "preflight schema drift"
    )
    require(report.get("run_id") == run_id, "preflight run ID drift")
    require(
        report.get("parent_run_id") == parent_run_id,
        "preflight parent run ID drift",
    )
    require(report.get("goal_sha256") == goal_sha256, "preflight Goal hash drift")
    require(
        report.get("mode") == "read_only_metadata_and_hashes", "preflight mode drift"
    )
    require(
        Path(report["worktree"]["path"]).resolve(strict=True) == REPO_ROOT,
        "preflight worktree substitution",
    )
    preflight_head = report["worktree"]["head"]
    require(
        preflight_head == implementation_commit,
        "formal preflight HEAD differs from implementation commit",
    )
    require(
        report["worktree"].get("status_porcelain") == "",
        "formal preflight worktree was not clean",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", preflight_head, implementation_commit],
        cwd=REPO_ROOT,
    )
    require(
        ancestor.returncode == 0, "preflight head is not an implementation ancestor"
    )
    common_dir_text = _git_text(["rev-parse", "--git-common-dir"])
    common_dir = Path(common_dir_text)
    if not common_dir.is_absolute():
        common_dir = REPO_ROOT / common_dir
    expected_main_repo = common_dir.resolve(strict=True).parent
    _validate_preflight_observation_fields(
        report,
        expected_worktree=REPO_ROOT,
        expected_main_repo=expected_main_repo,
        expected_mnt_root=Path("/mnt/cunyuliu"),
        expected_fm0_root=Path(fm0["root"]),
    )
    _validate_live_preflight_authenticity(
        report,
        expected_worktree=REPO_ROOT,
        expected_main_repo=expected_main_repo,
        expected_mnt_root=Path("/mnt/cunyuliu"),
    )
    safety = report.get("safety", {})
    require(
        safety.get("unrelated_processes_killed") == 0, "preflight killed unrelated work"
    )
    require(
        safety.get("existing_results_overwritten") == 0,
        "preflight overwrote existing results",
    )
    require(safety.get("final_labels_read") is False, "preflight accessed final labels")
    require(
        safety.get("neural_forward_executed") is False, "preflight executed neural work"
    )
    require(
        safety.get("downstream_stage_started") is False, "preflight started downstream"
    )
    upstream = report.get("upstream", {})
    require(
        Path(upstream["fm0_closure_root"]).resolve(strict=True) == Path(fm0["root"]),
        "preflight FM0 substitution",
    )
    require(
        upstream["fm0_checksum_ledger_sha256"] == fm0["checksum_ledger"]["sha256"],
        "preflight FM0 ledger drift",
    )
    for prefix, bound in (
        ("d1_canonical_records", fm0["d1_canonical_records"]),
        ("d1_exposure_ledger", fm0["d1_exposure_ledger"]),
    ):
        require(
            Path(upstream[f"{prefix}_path"]).resolve(strict=True)
            == Path(bound["path"]),
            f"preflight {prefix} path drift",
        )
        require(
            upstream[f"{prefix}_sha256"] == bound["sha256"],
            f"preflight {prefix} hash drift",
        )
        require(
            upstream[f"{prefix}_size_bytes"] == bound["size_bytes"],
            f"preflight {prefix} size drift",
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "observed_at_utc": report["observed_at_utc"],
        "preflight_worktree_head": preflight_head,
        "parent_run_id": parent_run_id,
        "safety": safety,
    }


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_inventory_block(
    value: Any,
    *,
    expected_root: Path,
    require_nonempty: bool,
    label: str,
    required_name_prefix: str | None = None,
) -> None:
    require(isinstance(value, Mapping), f"preflight {label} inventory is missing")
    root = Path(str(value.get("root", ""))).resolve(strict=True)
    require(root == expected_root.resolve(strict=True), f"preflight {label} root drift")
    require(value.get("recursive") is False, f"preflight {label} was recursive")
    require(value.get("metadata_only") is True, f"preflight {label} read data bytes")
    entries = value.get("entries")
    require(isinstance(entries, list), f"preflight {label} entries are missing")
    require(
        value.get("entry_count") == len(entries),
        f"preflight {label} entry count drift",
    )
    if require_nonempty:
        require(entries, f"preflight {label} inventory is empty")
    names: list[str] = []
    for record in entries:
        require(isinstance(record, Mapping), f"preflight {label} record is invalid")
        name = record.get("name")
        require(isinstance(name, str) and name, f"preflight {label} name is invalid")
        require(name not in names, f"preflight {label} duplicate entry")
        names.append(name)
        if required_name_prefix is not None:
            require(
                name.startswith(required_name_prefix),
                f"preflight {label} entry prefix drift",
            )
        child = Path(str(record.get("path", "")))
        require(
            child.parent == root and child.name == name,
            f"preflight {label} entry path drift",
        )
        require(
            record.get("kind") in {"file", "directory", "symlink", "other"},
            f"preflight {label} entry kind drift",
        )
        require(
            _is_nonnegative_int(record.get("size_bytes")),
            f"preflight {label} entry size drift",
        )
        require(
            _is_nonnegative_int(record.get("mtime_ns")),
            f"preflight {label} entry timestamp drift",
        )
        try:
            live_stat = child.lstat()
        except OSError as error:
            raise FinalizeFailure(
                f"preflight {label} entry no longer exists"
            ) from error
        live_kind = (
            "symlink"
            if child.is_symlink()
            else (
                "directory"
                if child.is_dir()
                else "file" if child.is_file() else "other"
            )
        )
        require(live_kind == record["kind"], f"preflight {label} live kind drift")
        if live_kind != "directory":
            require(
                live_stat.st_size == record["size_bytes"]
                and live_stat.st_mtime_ns == record["mtime_ns"],
                f"preflight {label} live metadata drift",
            )
    require(names == sorted(names), f"preflight {label} entries are not sorted")
    require(
        HEX64.fullmatch(str(value.get("inventory_sha256"))) is not None,
        f"preflight {label} inventory digest is invalid",
    )
    require(
        value["inventory_sha256"] == _canonical_sha256(entries),
        f"preflight {label} inventory digest drift",
    )


def _validate_preflight_observation_fields(
    report: Mapping[str, Any],
    *,
    expected_worktree: Path,
    expected_main_repo: Path,
    expected_mnt_root: Path,
    expected_fm0_root: Path,
) -> None:
    """Require evidence for every user-requested read-only preflight domain."""

    collector = report.get("collector")
    require(isinstance(collector, Mapping), "preflight collector binding is missing")
    expected_collector = REPO_ROOT / "scripts" / "mk0" / "record_mk0_preflight.py"
    require(
        Path(str(collector.get("path", ""))).resolve(strict=True)
        == expected_collector.resolve(strict=True),
        "preflight collector path drift",
    )
    require(
        collector.get("sha256") == sha256_file(expected_collector),
        "preflight collector source hash drift",
    )
    require(
        isinstance(collector.get("pid"), int)
        and not isinstance(collector["pid"], bool)
        and collector["pid"] > 0,
        "preflight collector PID is invalid",
    )

    worktree = report.get("worktree")
    require(isinstance(worktree, Mapping), "preflight worktree record is missing")
    require(
        HEX40.fullmatch(str(worktree.get("head"))) is not None,
        "preflight worktree HEAD is invalid",
    )
    require(
        isinstance(worktree.get("branch"), str) and bool(worktree["branch"]),
        "preflight worktree branch is missing",
    )
    require(
        isinstance(worktree.get("status_porcelain"), str),
        "preflight worktree status is missing",
    )

    main = report.get("protected_main_repo")
    require(isinstance(main, Mapping), "preflight protected main repo is missing")
    require(
        Path(str(main.get("path", ""))).resolve(strict=True)
        == expected_main_repo.resolve(strict=True),
        "preflight protected main repo substitution",
    )
    require(
        HEX40.fullmatch(str(main.get("head"))) is not None,
        "preflight protected main repo HEAD is invalid",
    )
    require(
        isinstance(main.get("branch"), str) and bool(main["branch"]),
        "preflight protected main repo branch is missing",
    )
    require(
        isinstance(main.get("status_porcelain"), str),
        "preflight protected main repo status is missing",
    )
    require(
        HEX64.fullmatch(str(main.get("worktree_list_sha256"))) is not None,
        "preflight worktree inventory digest is invalid",
    )
    worktree_list = main.get("worktree_list_porcelain")
    require(
        isinstance(worktree_list, str) and bool(worktree_list),
        "preflight worktree inventory is missing",
    )
    require(
        hashlib.sha256(worktree_list.encode("utf-8")).hexdigest()
        == main["worktree_list_sha256"],
        "preflight worktree inventory digest drift",
    )

    resources = report.get("resources")
    require(isinstance(resources, Mapping), "preflight resources are missing")
    for key, expected_path in (
        ("home_filesystem", expected_worktree),
        ("mnt_filesystem", expected_mnt_root),
    ):
        filesystem = resources.get(key)
        require(isinstance(filesystem, Mapping), f"preflight {key} is missing")
        require(
            Path(str(filesystem.get("path", ""))).resolve(strict=True)
            == expected_path.resolve(strict=True),
            f"preflight {key} path drift",
        )
        for field in ("total_bytes", "used_bytes", "free_bytes", "reserved_bytes"):
            require(
                _is_nonnegative_int(filesystem.get(field)),
                f"preflight {key} {field} is invalid",
            )
        require(filesystem["total_bytes"] > 0, f"preflight {key} total is zero")
        require(filesystem["free_bytes"] > 0, f"preflight {key} has no free space")
        require(
            filesystem["used_bytes"]
            + filesystem["free_bytes"]
            + filesystem["reserved_bytes"]
            == filesystem["total_bytes"],
            f"preflight {key} accounting drift",
        )

    cpu_ram = resources.get("cpu_memory")
    require(isinstance(cpu_ram, Mapping), "preflight CPU RAM inventory is missing")
    for field in (
        "total_bytes",
        "available_bytes",
        "free_bytes",
        "swap_total_bytes",
        "swap_free_bytes",
    ):
        require(
            _is_nonnegative_int(cpu_ram.get(field)),
            f"preflight CPU RAM {field} is invalid",
        )
    require(cpu_ram["total_bytes"] > 0, "preflight CPU RAM total is zero")
    require(
        cpu_ram["available_bytes"] <= cpu_ram["total_bytes"]
        and cpu_ram["free_bytes"] <= cpu_ram["total_bytes"]
        and cpu_ram["swap_free_bytes"] <= cpu_ram["swap_total_bytes"],
        "preflight CPU RAM accounting is invalid",
    )

    framework = resources.get("framework")
    require(isinstance(framework, Mapping), "preflight framework inventory is missing")
    for field in (
        "python_version",
        "python_implementation",
        "python_executable",
        "torch_version",
        "torch_cuda_build_version",
    ):
        require(
            isinstance(framework.get(field), str) and bool(framework[field]),
            f"preflight framework {field} is missing",
        )
    require(
        Path(framework["python_executable"]).is_file(),
        "preflight Python executable is absent",
    )
    require(
        framework.get("torch_cuda_is_available") is True,
        "preflight torch CUDA availability is false",
    )
    require(
        _is_nonnegative_int(framework.get("torch_cuda_device_count"))
        and framework["torch_cuda_device_count"] > 0,
        "preflight torch CUDA device count is invalid",
    )
    cudnn_version = framework.get("torch_cudnn_version")
    require(
        cudnn_version is None or _is_nonnegative_int(cudnn_version),
        "preflight cuDNN version is invalid",
    )
    require(
        resources.get("nvidia_smi_exit_code") == 0,
        "preflight nvidia-smi header query did not succeed",
    )
    require(
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)*",
            str(resources.get("driver_supported_cuda_version", "")),
        )
        is not None,
        "preflight driver-supported CUDA version is invalid",
    )
    for field in ("nvidia_smi_stdout_sha256", "nvidia_smi_stderr_sha256"):
        require(
            HEX64.fullmatch(str(resources.get(field))) is not None,
            f"preflight {field} is invalid",
        )
    require(
        resources.get("nvidia_smi_l_exit_code") == 0,
        "preflight nvidia-smi topology query did not succeed",
    )
    for field in (
        "nvidia_smi_l_stdout_sha256",
        "nvidia_smi_l_stderr_sha256",
        "mig_instances_sha256",
    ):
        require(
            HEX64.fullmatch(str(resources.get(field))) is not None,
            f"preflight {field} is invalid",
        )

    require(
        resources.get("gpu_query_exit_code") == 0,
        "preflight GPU query did not succeed",
    )
    require(
        HEX64.fullmatch(str(resources.get("gpu_query_stderr_sha256"))) is not None,
        "preflight GPU stderr digest is invalid",
    )
    gpus = resources.get("gpus")
    require(isinstance(gpus, list) and gpus, "preflight GPU inventory is empty")
    gpu_indices: set[int] = set()
    gpu_uuids: set[str] = set()
    for gpu in gpus:
        require(isinstance(gpu, Mapping), "preflight GPU record is invalid")
        index = gpu.get("index")
        uuid = gpu.get("uuid")
        require(_is_nonnegative_int(index), "preflight GPU index is invalid")
        require(index not in gpu_indices, "preflight duplicate GPU index")
        gpu_indices.add(index)
        require(
            isinstance(uuid, str)
            and re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", uuid) is not None
            and uuid not in gpu_uuids,
            "preflight GPU UUID is invalid or duplicated",
        )
        gpu_uuids.add(uuid)
        require(
            isinstance(gpu.get("name"), str) and bool(gpu["name"]),
            "preflight GPU name is missing",
        )
        require(
            isinstance(gpu.get("driver_version"), str)
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", gpu["driver_version"]) is not None,
            "preflight GPU driver version is invalid",
        )
        for field in ("memory_total_mib", "memory_used_mib", "memory_free_mib"):
            require(
                _is_nonnegative_int(gpu.get(field)),
                f"preflight GPU {field} is invalid",
            )
        require(gpu["memory_total_mib"] > 0, "preflight GPU total memory is zero")
        require(
            gpu["memory_used_mib"] <= gpu["memory_total_mib"]
            and gpu["memory_free_mib"] <= gpu["memory_total_mib"],
            "preflight GPU memory accounting is invalid",
        )
        utilization = gpu.get("utilization_gpu_percent")
        require(
            utilization is None
            or (_is_nonnegative_int(utilization) and utilization <= 100),
            "preflight GPU utilization is invalid",
        )
    require(
        framework["torch_cuda_device_count"] == len(gpus),
        "preflight PyTorch/nvidia-smi GPU inventory count drift",
    )
    mig_instances = resources.get("mig_instances")
    require(
        isinstance(mig_instances, list)
        and resources.get("mig_instance_count") == len(mig_instances),
        "preflight MIG inventory count drift",
    )
    mig_uuids: set[str] = set()
    for instance in mig_instances:
        require(isinstance(instance, Mapping), "preflight MIG record is invalid")
        mig_uuid = instance.get("mig_uuid")
        require(
            isinstance(mig_uuid, str)
            and re.fullmatch(r"MIG-[0-9A-Fa-f-]{36}", mig_uuid) is not None
            and mig_uuid not in mig_uuids,
            "preflight MIG UUID is invalid or duplicated",
        )
        mig_uuids.add(mig_uuid)
        require(
            instance.get("parent_gpu_uuid") in gpu_uuids,
            "preflight MIG parent is absent from physical GPU inventory",
        )
        require(
            isinstance(instance.get("profile"), str) and bool(instance["profile"]),
            "preflight MIG profile is missing",
        )
        require(
            _is_nonnegative_int(instance.get("device_index_within_parent")),
            "preflight MIG device index is invalid",
        )
    require(
        resources["mig_instances_sha256"] == _canonical_sha256(mig_instances),
        "preflight MIG inventory digest drift",
    )
    require(
        resources.get("gpu_compute_process_query_exit_code") == 0,
        "preflight GPU process query did not succeed",
    )
    require(
        _is_nonnegative_int(resources.get("gpu_compute_process_count")),
        "preflight GPU process count is invalid",
    )
    gpu_processes = resources.get("gpu_compute_processes")
    require(
        isinstance(gpu_processes, list)
        and len(gpu_processes) == resources["gpu_compute_process_count"],
        "preflight GPU process records/count drift",
    )
    gpu_process_keys: set[tuple[int, str]] = set()
    for process in gpu_processes:
        require(isinstance(process, Mapping), "preflight GPU process record is invalid")
        pid = process.get("pid")
        process_key = (pid, str(process.get("gpu_uuid")))
        require(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and process_key not in gpu_process_keys,
            "preflight GPU process identity is invalid or duplicated",
        )
        gpu_process_keys.add(process_key)
        require(
            process.get("gpu_uuid") in gpu_uuids,
            "preflight GPU process UUID is absent from GPU inventory",
        )
        require(
            _is_nonnegative_int(process.get("used_memory_mib")),
            "preflight GPU process memory is invalid",
        )
        require(
            isinstance(process.get("process_name"), str)
            and bool(process["process_name"]),
            "preflight GPU process name is missing",
        )
        owner_resolution = process.get("owner_resolution")
        require(
            owner_resolution
            in {"RESOLVED_FROM_PROC_STATUS", "PROCESS_EXITED_DURING_PREFLIGHT"},
            "preflight GPU process owner resolution failed",
        )
        if owner_resolution == "RESOLVED_FROM_PROC_STATUS":
            require(
                isinstance(process.get("owner"), str)
                and bool(process["owner"])
                and _is_nonnegative_int(process.get("owner_uid")),
                "preflight GPU process owner is invalid",
            )
        else:
            require(
                process.get("owner") is None and process.get("owner_uid") is None,
                "preflight exited GPU process has fabricated owner metadata",
            )
    for field in (
        "gpu_compute_process_query_stderr_sha256",
        "gpu_compute_process_metadata_sha256",
    ):
        require(
            HEX64.fullmatch(str(resources.get(field))) is not None,
            f"preflight {field} is invalid",
        )
    require(
        resources["gpu_compute_process_metadata_sha256"]
        == _canonical_sha256(gpu_processes),
        "preflight GPU process record digest drift",
    )
    require(
        resources.get("current_user_process_query_exit_code") == 0,
        "preflight process query did not succeed",
    )
    require(
        _is_nonnegative_int(resources.get("current_user_process_count"))
        and resources["current_user_process_count"] > 0,
        "preflight process inventory is empty",
    )
    user_processes = resources.get("current_user_processes")
    require(
        isinstance(user_processes, list)
        and len(user_processes) == resources["current_user_process_count"],
        "preflight current-user process records/count drift",
    )
    user_process_pids: set[int] = set()
    for process in user_processes:
        require(
            isinstance(process, Mapping),
            "preflight current-user process record is invalid",
        )
        pid = process.get("pid")
        require(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and pid not in user_process_pids,
            "preflight current-user process PID is invalid or duplicated",
        )
        user_process_pids.add(pid)
        for field in ("stat", "elapsed", "command"):
            require(
                isinstance(process.get(field), str) and bool(process[field]),
                f"preflight current-user process {field} is invalid",
            )
    for field in (
        "current_user_process_query_stderr_sha256",
        "current_user_process_metadata_sha256",
    ):
        require(
            HEX64.fullmatch(str(resources.get(field))) is not None,
            f"preflight {field} is invalid",
        )
    require(
        resources["current_user_process_metadata_sha256"]
        == _canonical_sha256(user_processes),
        "preflight current-user process record digest drift",
    )
    require(
        resources.get("gpu_process_policy")
        == "no process killed; any card with sufficient free memory may be used per user authorization",
        "preflight GPU/process protection policy drift",
    )

    inventory = report.get("inventory")
    require(
        isinstance(inventory, Mapping), "preflight data/artifact inventory is missing"
    )
    project_data = inventory.get("project_data")
    artifacts = inventory.get("existing_artifacts")
    require(
        isinstance(project_data, Mapping), "preflight project data inventory is missing"
    )
    require(isinstance(artifacts, Mapping), "preflight artifact inventory is missing")
    _validate_inventory_block(
        project_data.get("data"),
        expected_root=expected_main_repo / "data",
        require_nonempty=True,
        label="project data",
    )
    _validate_inventory_block(
        project_data.get("data_registry"),
        expected_root=expected_main_repo / "data_registry",
        require_nonempty=True,
        label="data registry",
    )
    _validate_inventory_block(
        artifacts.get("main_repo_artifacts"),
        expected_root=expected_main_repo / "artifacts",
        require_nonempty=True,
        label="main repo artifacts",
    )
    _validate_inventory_block(
        artifacts.get("fm0_closure"),
        expected_root=expected_fm0_root,
        require_nonempty=True,
        label="FM0 closure artifacts",
    )
    _validate_inventory_block(
        artifacts.get("mnt_mrna_editflow_roots"),
        expected_root=expected_mnt_root,
        require_nonempty=True,
        label="mnt mRNA EditFlow roots",
        required_name_prefix="mrna_editflow",
    )


def _run_read_only_text(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as error:
        raise FinalizeFailure(f"live preflight probe failed: {command[0]}") from error
    return completed.stdout.strip()


def _read_linux_cpu_memory_total() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        parts = value.split()
        if key == "MemTotal" and separator and parts and parts[0].isdigit():
            multiplier = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
            return int(parts[0]) * multiplier
    raise FinalizeFailure("live /proc/meminfo lacks MemTotal")


def _validate_live_preflight_authenticity(
    report: Mapping[str, Any],
    *,
    expected_worktree: Path,
    expected_main_repo: Path,
    expected_mnt_root: Path,
) -> None:
    """Re-probe stable identities so a shape-only preflight cannot self-certify."""

    worktree = report["worktree"]
    main = report["protected_main_repo"]
    for label, path, recorded in (
        ("worktree", expected_worktree, worktree),
        ("protected main repo", expected_main_repo, main),
    ):
        branch = _run_read_only_text(["git", "branch", "--show-current"], cwd=path)
        head = _run_read_only_text(["git", "rev-parse", "HEAD"], cwd=path)
        status = _run_read_only_text(["git", "status", "--porcelain=v1"], cwd=path)
        require(branch == recorded["branch"], f"live {label} branch drift")
        require(head == recorded["head"], f"live {label} HEAD drift")
        require(status == recorded["status_porcelain"], f"live {label} status drift")
    worktrees = _run_read_only_text(
        ["git", "worktree", "list", "--porcelain"], cwd=expected_main_repo
    )
    require(
        worktrees == main["worktree_list_porcelain"],
        "live Git worktree inventory drift",
    )
    require(
        hashlib.sha256(worktrees.encode("utf-8")).hexdigest()
        == main["worktree_list_sha256"],
        "live Git worktree inventory digest drift",
    )

    resources = report["resources"]
    for key, path in (
        ("home_filesystem", expected_worktree),
        ("mnt_filesystem", expected_mnt_root),
    ):
        live = shutil.disk_usage(path)
        recorded = resources[key]
        require(
            live.total == recorded["total_bytes"],
            f"live {key} total-capacity drift",
        )
        require(live.free > 0, f"live {key} has no available space")

    require(
        _read_linux_cpu_memory_total() == resources["cpu_memory"]["total_bytes"],
        "live CPU RAM total-capacity drift",
    )

    gpu_csv = _run_read_only_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    live_gpus: list[dict[str, Any]] = []
    for line in gpu_csv.splitlines():
        parts = [part.strip() for part in line.split(",")]
        require(len(parts) == 8, "live GPU inventory is malformed")
        live_gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
            }
        )
    stable_recorded = [
        {
            "index": gpu["index"],
            "name": gpu["name"],
            "uuid": gpu["uuid"],
            "driver_version": gpu["driver_version"],
            "memory_total_mib": gpu["memory_total_mib"],
        }
        for gpu in resources["gpus"]
    ]
    require(
        live_gpus == stable_recorded,
        "live GPU identity differs from preflight inventory",
    )
    nvidia_smi_header = _run_read_only_text(["nvidia-smi"])
    cuda_version_match = re.search(
        r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", nvidia_smi_header
    )
    require(cuda_version_match is not None, "live nvidia-smi CUDA version is missing")
    require(
        cuda_version_match.group(1) == resources["driver_supported_cuda_version"],
        "live driver-supported CUDA version drift",
    )
    live_topology = _run_read_only_text(["nvidia-smi", "-L"])
    require(
        hashlib.sha256(live_topology.encode("utf-8")).hexdigest()
        == resources["nvidia_smi_l_stdout_sha256"],
        "live GPU/MIG topology drift",
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_foundation_snapshot_provenance(
    foundation: Mapping[str, Any], fm0: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently re-hash the GPU snapshot against the FM0 closure."""

    binding = foundation.get("snapshot_binding")
    expected = fm0.get("foundation_checkpoint_manifest")
    require(isinstance(binding, Mapping), "GPU snapshot binding is missing")
    require(isinstance(expected, Mapping), "FM0 checkpoint manifest binding is missing")
    snapshot_dir = Path(str(expected.get("snapshot_dir", ""))).resolve(strict=True)
    require(snapshot_dir.is_dir(), "FM0-bound snapshot directory is missing")
    require(
        Path(str(binding.get("snapshot_dir", ""))).resolve(strict=True) == snapshot_dir,
        "GPU snapshot path differs from FM0 closure",
    )
    require(
        binding.get("model_id") == expected.get("model_id"),
        "GPU snapshot model ID differs from FM0 closure",
    )
    require(
        binding.get("observed_revision") == expected.get("revision")
        and binding.get("expected_revision") == expected.get("revision"),
        "GPU snapshot revision differs from FM0 closure",
    )
    require(
        Path(str(binding.get("fm0_hash_license_manifest_path", ""))).resolve(
            strict=True
        )
        == Path(str(expected.get("path", ""))).resolve(strict=True),
        "GPU checkpoint-manifest path differs from FM0 closure",
    )
    require(
        binding.get("fm0_hash_license_manifest_sha256") == expected.get("sha256"),
        "GPU checkpoint-manifest hash differs from FM0 closure",
    )
    require(
        binding.get("fm0_checksum_ledger_sha256")
        == fm0.get("checksum_ledger", {}).get("sha256"),
        "GPU FM0 checksum-ledger hash drift",
    )

    observed_files = [
        {
            "path": str(path.relative_to(snapshot_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(snapshot_dir.rglob("*"))
        if path.is_file()
    ]
    expected_files = expected.get("files")
    require(
        isinstance(expected_files, list) and expected_files,
        "FM0 checkpoint file inventory is empty",
    )
    require(
        expected.get("file_count") == len(expected_files)
        and expected.get("files_sha256") == _canonical_sha256(expected_files),
        "FM0 checkpoint file-inventory binding drift",
    )
    require(
        observed_files == expected_files,
        "live foundation snapshot bytes differ from FM0 closure",
    )
    require(
        binding.get("files") == observed_files
        and binding.get("file_count") == len(observed_files),
        "GPU-observed snapshot inventory differs from finalizer re-hash",
    )
    observed_manifest_sha256 = _canonical_sha256(observed_files)
    require(
        binding.get("snapshot_manifest_sha256") == observed_manifest_sha256
        and binding.get("fm0_expected_snapshot_manifest_sha256")
        == observed_manifest_sha256,
        "GPU snapshot manifest digest differs from FM0 closure",
    )
    require(
        binding.get("fm0_expected_snapshot_file_count") == len(observed_files)
        and binding.get("snapshot_bytes_match_fm0_closure") is True,
        "GPU snapshot FM0 byte-match certificate failed",
    )
    require(
        binding.get("post_model_load_rehash_match") is True,
        "GPU snapshot lacks the post-model-load re-hash certificate",
    )
    license_binding = expected.get("license")
    require(
        isinstance(license_binding, Mapping)
        and license_binding.get("type") == "agpl-3.0",
        "FM0 checkpoint license binding is invalid",
    )
    require(
        binding.get("license_binding") == license_binding,
        "GPU checkpoint license binding differs from FM0 closure",
    )
    return {
        "model_id": expected["model_id"],
        "revision": expected["revision"],
        "snapshot_dir": str(snapshot_dir),
        "file_count": len(observed_files),
        "snapshot_manifest_sha256": observed_manifest_sha256,
        "fm0_hash_license_manifest_path": expected["path"],
        "fm0_hash_license_manifest_sha256": expected["sha256"],
        "fm0_checksum_ledger_sha256": fm0["checksum_ledger"]["sha256"],
        "license_type": license_binding["type"],
        "license_md_sha256": license_binding["license_md_sha256"],
        "license_md_size": license_binding["license_md_size"],
        "runner_and_finalizer_independent_rehash_match": True,
    }


def verify_preflight_gpu_execution_identity(
    preflight_path: Path,
    foundation: Mapping[str, Any],
    *,
    expected_preflight_sha256: str,
) -> dict[str, Any]:
    resolved = preflight_path.resolve(strict=True)
    require(
        sha256_file(resolved) == expected_preflight_sha256,
        "preflight changed before CUDA identity cross-binding",
    )
    report = read_json(resolved)
    gpus = report.get("resources", {}).get("gpus")
    cuda = foundation.get("cuda")
    require(isinstance(gpus, list) and gpus, "preflight GPU inventory is absent")
    require(isinstance(cuda, Mapping), "formal CUDA identity is absent")
    device_uuid = cuda.get("device_uuid")
    matched = next((gpu for gpu in gpus if gpu.get("uuid") == device_uuid), None)
    require(
        isinstance(matched, Mapping),
        "formal CUDA UUID was absent from the preflight GPU inventory",
    )
    require(
        matched.get("name") == cuda.get("device_name"),
        "formal CUDA device name differs from preflight",
    )
    framework = report.get("resources", {}).get("framework", {})
    require(
        framework.get("torch_version") == cuda.get("torch_version"),
        "formal torch version differs from preflight",
    )
    require(
        framework.get("torch_cuda_build_version") == cuda.get("torch_cuda_version"),
        "formal torch CUDA build differs from preflight",
    )
    require(
        framework.get("python_version") == cuda.get("python_version"),
        "formal Python version differs from preflight",
    )
    return {
        "device_uuid": device_uuid,
        "device_name": cuda["device_name"],
        "preflight_physical_index": matched["index"],
        "formal_logical_index": cuda["logical_device_index"],
        "cuda_visible_devices": cuda.get("cuda_visible_devices"),
        "preflight_driver_version": matched["driver_version"],
        "preflight_driver_supported_cuda_version": report["resources"][
            "driver_supported_cuda_version"
        ],
        "python_version": framework["python_version"],
        "torch_version": framework["torch_version"],
        "torch_cuda_build_version": framework["torch_cuda_build_version"],
        "preflight_gpu_inventory_sha256": _canonical_sha256(gpus),
        "formal_cuda_device_was_present_at_preflight": True,
    }


def _module_matches_prefix(module_name: str, prefix: str) -> bool:
    return module_name == prefix or module_name.startswith(f"{prefix}.")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _requirements_lock_typing_extensions_version() -> str:
    matches = re.findall(
        r"(?m)^typing_extensions==([^\s#]+)\s*$",
        REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise FinalizeFailure("requirements lock has no unique typing_extensions pin")
    return matches[0]


@lru_cache(maxsize=1)
def _typing_compatibility_shim_binding() -> dict[str, Any]:
    """Independently bind the exact Python 3.10 typing shim."""

    distribution = importlib_metadata.distribution(
        TYPING_COMPATIBILITY_SHIM_DISTRIBUTION
    )
    if distribution.version != TYPING_COMPATIBILITY_SHIM_VERSION:
        raise FinalizeFailure("typing compatibility shim version drift")
    owned_files = [
        item
        for item in distribution.files or ()
        if str(item).replace("\\", "/") == "typing_extensions.py"
    ]
    if len(owned_files) != 1:
        raise FinalizeFailure("typing compatibility shim distribution ownership drift")
    distribution_source = Path(distribution.locate_file(owned_files[0])).resolve(
        strict=True
    )
    distribution_root = Path(distribution.locate_file(".")).resolve(strict=True)
    module = importlib.import_module(TYPING_COMPATIBILITY_SHIM_MODULE)
    module_file = getattr(module, "__file__", None)
    module_spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(module_file, str) or not isinstance(module_spec_origin, str):
        raise FinalizeFailure("typing compatibility shim lacks an import origin")
    source_path = Path(module_file).resolve(strict=True)
    if (
        source_path != distribution_source
        or Path(module_spec_origin).resolve(strict=True) != source_path
        or not _path_is_within(source_path, distribution_root)
    ):
        raise FinalizeFailure("typing compatibility shim source origin drift")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != TYPING_COMPATIBILITY_SHIM_SOURCE_SHA256:
        raise FinalizeFailure("typing compatibility shim source hash drift")

    allowed_calls: list[dict[str, Any]] = []
    for qualname, first_lineno in TYPING_COMPATIBILITY_SHIM_CALLS:
        function = getattr(module, qualname, None)
        code = getattr(function, "__code__", None)
        if (
            code is None
            or getattr(function, "__module__", None) != TYPING_COMPATIBILITY_SHIM_MODULE
            or getattr(function, "__qualname__", None) != qualname
            or int(code.co_firstlineno) != first_lineno
            or Path(code.co_filename).resolve(strict=True) != source_path
        ):
            raise FinalizeFailure(
                f"typing compatibility shim callable identity drift: {qualname}"
            )
        allowed_calls.append(
            {"function_qualname": qualname, "first_lineno": first_lineno}
        )

    locked_version = _requirements_lock_typing_extensions_version()
    binding: dict[str, Any] = {
        "schema_version": "mk0_typing_compatibility_shim_binding_v1",
        "classification": TYPING_COMPATIBILITY_SHIM_CLASSIFICATION,
        "distribution_name": TYPING_COMPATIBILITY_SHIM_DISTRIBUTION,
        "distribution_version": distribution.version,
        "module_name": TYPING_COMPATIBILITY_SHIM_MODULE,
        "module_match": "exact_only_no_submodules",
        "source_file": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": source_sha256,
        "distribution_owns_source": True,
        "distribution_root": str(distribution_root),
        "distribution_root_verified": True,
        "current_sysconfig_site_package_root_match": any(
            _path_is_within(source_path, root) for root in SITE_PACKAGE_ROOTS
        ),
        "allowed_calls": allowed_calls,
        "requirements_lock": {
            "path": str(REQUIREMENTS_LOCK_PATH.resolve(strict=True)),
            "sha256": sha256_file(REQUIREMENTS_LOCK_PATH),
            "expected_version": locked_version,
            "observed_version": distribution.version,
            "reproduced": distribution.version == locked_version,
            "drift_disclosure_marker": ENVIRONMENT_LOCK_DRIFT_MARKER,
        },
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return binding


def _external_call_policy_payload() -> dict[str, Any]:
    return {
        "unknown_external_calls": "FAIL_CLOSED",
        "stdlib_root": str(STDLIB_ROOT),
        "site_package_roots_excluded_from_stdlib": [
            str(path) for path in SITE_PACKAGE_ROOTS
        ],
        "frozen_foundation_module_prefixes": list(
            FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES
        ),
        "exact_runtime_dependency_bindings": [_typing_compatibility_shim_binding()],
    }


def _external_prohibited_categories(
    module_name: str,
    source_file: str,
    qualname: str,
) -> tuple[str, ...]:
    lower = f"{module_name} {source_file} {qualname}".lower()
    categories: set[str] = set()
    if "critic" in lower:
        categories.add("critic_query")
    if "guidance" in lower:
        categories.add("guidance_query")
    if "evaluator" in lower:
        categories.add("final_evaluator_query")
    if any(token in lower for token in ("reward", "rerank", "selector")):
        categories.add("final_evaluator_query")
    return tuple(sorted(categories))


def _external_call_classification(
    module_name: str,
    source_file: str,
    qualname: str,
    first_lineno: int,
) -> tuple[str, tuple[str, ...]]:
    categories = _external_prohibited_categories(
        module_name,
        source_file,
        qualname,
    )
    if categories:
        return "prohibited_role", categories
    if module_name == TYPING_COMPATIBILITY_SHIM_MODULE:
        binding = _typing_compatibility_shim_binding()
        allowed = {
            (record["function_qualname"], record["first_lineno"])
            for record in binding["allowed_calls"]
        }
        if (
            source_file == binding["source_file"]
            and (qualname, first_lineno) in allowed
        ):
            return TYPING_COMPATIBILITY_SHIM_CLASSIFICATION, ()
        return "unknown_external", ()
    if any(
        _module_matches_prefix(module_name, prefix)
        for prefix in FROZEN_FOUNDATION_EXTERNAL_MODULE_PREFIXES
    ):
        return "frozen_foundation_stack_allowlist", ()
    root_module = module_name.partition(".")[0]
    if root_module in sys.stdlib_module_names:
        if source_file.startswith(("<built-in", "<frozen")):
            return "stdlib_allowlist", ()
        try:
            source_path = Path(source_file).resolve(strict=True)
        except OSError:
            return "unknown_external", ()
        in_site_packages = any(
            _path_is_within(source_path, root) for root in SITE_PACKAGE_ROOTS
        )
        if _path_is_within(source_path, STDLIB_ROOT) and not in_site_packages:
            return "stdlib_allowlist", ()
    return "unknown_external", ()


def _ast_function_lines(relative: str) -> dict[str, int]:
    """Map full class/function qualnames to lines from bound source bytes."""

    path = (REPO_ROOT / relative).resolve(strict=True)
    tree = ast.parse(path.read_bytes(), filename=str(path))
    observed: dict[str, int] = {}

    def visit_body(body: list[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit_body(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join((*parents, node.name))
                observed[qualname] = int(node.lineno)

    visit_body(tree.body, ())
    return observed


def _bound_gpu_role_identities(
    source_binding: Mapping[str, Any] | None = None,
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    """Independently reconstruct exact positive roles from bound source ASTs."""

    source_files = (
        source_binding.get("tracked_source_files", {})
        if source_binding is not None
        else {}
    )
    lines_by_source: dict[str, dict[str, int]] = {}
    identities: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for (relative, qualname), categories in EXACT_GPU_ROLE_INTERFACES.items():
        path = (REPO_ROOT / relative).resolve(strict=True)
        if source_binding is not None:
            require(
                source_files.get(relative) == sha256_file(path),
                f"exact GPU role source hash drift: {relative}",
            )
        lines = lines_by_source.setdefault(relative, _ast_function_lines(relative))
        require(
            qualname in lines,
            f"exact GPU role callable is absent from bound source: {qualname}",
        )
        identity = (relative, qualname, lines[qualname])
        require(identity not in identities, "duplicate exact GPU role identity")
        identities[identity] = tuple(sorted(categories))
    return identities


def _gpu_role_categories(
    source_file: str,
    function_qualname: str,
    first_lineno: int | None = None,
    exact_identities: Mapping[tuple[str, str, int], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Recompute roles from exact AST identity plus conservative prohibitions."""

    identities = (
        dict(exact_identities)
        if exact_identities is not None
        else _bound_gpu_role_identities()
    )
    categories: set[str] = set()
    if first_lineno is None:
        for (relative, qualname, _line), positive in identities.items():
            if (source_file, function_qualname) == (relative, qualname):
                categories.update(positive)
    else:
        categories.update(
            identities.get((source_file, function_qualname, first_lineno), ())
        )
    lower = f"{source_file}:{function_qualname}".lower()
    if "critic" in lower:
        categories.add("critic_query")
    if "guidance" in lower:
        categories.add("guidance_query")
    if "evaluator" in lower or any(
        token in lower for token in ("reward", "rerank", "selector")
    ):
        categories.add("final_evaluator_query")
    return tuple(sorted(categories))


def verify_model_loading_thread_policy(policy: Any) -> dict[str, Any]:
    """Independently verify the scoped, three-class model-loading policy."""

    expected_keys = {
        "schema_version",
        "status",
        "scope",
        "prevention_mechanism",
        "loads_covered",
        "load_completion",
        "control_apis",
        "package_versions",
        "class_records",
        "distinct_class_count",
        "restoration_order",
        "hf_hub_env_present_before",
        "hf_hub_env_value_before",
        "hf_hub_env_value_during_load",
        "hf_hub_env_present_after_restore",
        "hf_hub_env_value_after_restore",
        "hf_hub_env_state_restored",
        "transformers_progress_bar_enabled_before",
        "transformers_progress_bar_enabled_during_load",
        "transformers_progress_bar_enabled_after_restore",
        "transformers_progress_state_restored",
        "huggingface_hub_progress_registry_before",
        "huggingface_hub_progress_registry_during_load",
        "huggingface_hub_progress_registry_after_restore",
        "huggingface_hub_progress_registry_restored",
        "huggingface_hub_progress_registry_object_identity_preserved",
        "huggingface_hub_progress_registry_restoration_method",
        "noncurrent_python_threads_before_model_loading",
        "noncurrent_python_threads_after_policy_configuration",
        "noncurrent_python_threads_after_model_loading",
        "noncurrent_python_threads_after_policy_restoration",
        "thread_name_allowlist",
        "thread_name_filtering_used",
        "thread_cleanup_attempted",
        "tqdm_monitor_object_touched",
        "all_monitor_states_restored",
        "restoration_errors",
        "failure_stage",
        "failure_reason",
    }
    require(
        isinstance(policy, Mapping) and set(policy) == expected_keys,
        "GPU model-loading thread policy schema is missing or malformed",
    )
    expected_roles = [
        "official_frozen",
        "same_architecture_from_scratch_control",
    ]
    require(
        policy.get("schema_version") == "mk0_model_loading_thread_policy_v1"
        and policy.get("status") == "PASS"
        and policy.get("scope") == "two_load_official_utrlm_calls_only"
        and policy.get("prevention_mechanism") == "scoped_tqdm_monitor_interval_zero"
        and policy.get("loads_covered") == expected_roles,
        "GPU model-loading thread policy scope or load coverage drift",
    )
    load_completion = policy.get("load_completion")
    require(
        type(load_completion) is dict
        and list(load_completion) == expected_roles
        and all(load_completion[role] is True for role in expected_roles),
        "GPU model-loading completion binding drift",
    )
    require(
        policy.get("control_apis")
        == [
            "transformers.utils.logging.disable_progress_bar",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
            "tqdm.monitor_interval=0",
        ],
        "GPU model-loading progress-control API drift",
    )

    package_versions = policy.get("package_versions")
    require(
        isinstance(package_versions, Mapping)
        and set(package_versions) == {"transformers", "tqdm", "huggingface_hub"}
        and all(
            isinstance(package_versions.get(name), str)
            and bool(package_versions.get(name))
            for name in ("transformers", "tqdm", "huggingface_hub")
        ),
        "GPU model-loading package-version binding is malformed",
    )
    for distribution in ("transformers", "tqdm", "huggingface_hub"):
        try:
            current_version = importlib_metadata.version(distribution)
        except Exception as error:
            raise FinalizeFailure(
                f"cannot resolve current model-loading package version: {distribution}"
            ) from error
        require(
            package_versions[distribution] == current_version,
            f"GPU model-loading package-version drift: {distribution}",
        )

    require(
        type(policy.get("distinct_class_count")) is int
        and policy.get("distinct_class_count") == 3,
        "GPU model-loading tqdm class identity coverage drift",
    )
    class_records = policy.get("class_records")
    require(
        isinstance(class_records, list) and len(class_records) == 3,
        "GPU model-loading tqdm class coverage is incomplete",
    )
    expected_class_roles = [
        "tqdm_standard",
        "tqdm_auto",
        "huggingface_hub_tqdm",
    ]
    record_keys = {
        "role",
        "class_path",
        "original_monitor_interval_owned_by_class",
        "original_monitor_interval",
        "original_monitor_interval_type",
        "monitor_interval_owned_during_load",
        "monitor_interval_during_load",
        "monitor_interval_during_load_type",
        "monitor_interval_owned_after_restore",
        "monitor_interval_after_restore",
        "monitor_interval_after_restore_type",
        "state_restored_exactly",
    }
    require(
        all(
            isinstance(record, Mapping) and set(record) == record_keys
            for record in class_records
        ),
        "GPU model-loading tqdm class record schema drift",
    )
    require(
        [record.get("role") for record in class_records] == expected_class_roles,
        "GPU model-loading tqdm class role drift",
    )
    class_paths = [record.get("class_path") for record in class_records]
    try:
        actual_standard_tqdm = importlib.import_module("tqdm.std").tqdm
        actual_auto_tqdm = importlib.import_module("tqdm.auto").tqdm
        actual_hub_tqdm = importlib.import_module("huggingface_hub.utils.tqdm").tqdm
    except Exception as error:
        raise FinalizeFailure(
            "cannot independently resolve the model-loading tqdm classes"
        ) from error
    actual_classes = (
        actual_standard_tqdm,
        actual_auto_tqdm,
        actual_hub_tqdm,
    )
    actual_class_paths = [
        f"{cls.__module__}.{cls.__qualname__}" for cls in actual_classes
    ]
    require(
        len({id(cls) for cls in actual_classes}) == 3
        and class_paths == actual_class_paths,
        "GPU model-loading tqdm runtime class identity drift",
    )
    for record in class_records:
        original = record.get("original_monitor_interval")
        restored = record.get("monitor_interval_after_restore")
        require(
            type(record.get("original_monitor_interval_owned_by_class")) is bool
            and type(original) is int
            and original >= 0
            and record.get("original_monitor_interval_type") == "int"
            and record.get("monitor_interval_owned_during_load") is True
            and type(record.get("monitor_interval_during_load")) is int
            and record.get("monitor_interval_during_load") == 0
            and record.get("monitor_interval_during_load_type") == "int",
            f"GPU model-loading tqdm monitor suppression drift: {record.get('role')}",
        )
        require(
            type(record.get("monitor_interval_owned_after_restore")) is bool
            and record.get("monitor_interval_owned_after_restore")
            is record.get("original_monitor_interval_owned_by_class")
            and type(restored) is type(original)
            and restored == original
            and record.get("monitor_interval_after_restore_type") == "int"
            and record.get("state_restored_exactly") is True,
            f"GPU model-loading tqdm state restoration drift: {record.get('role')}",
        )
    require(
        policy.get("restoration_order")
        == ["huggingface_hub_tqdm", "tqdm_auto", "tqdm_standard"]
        and policy.get("all_monitor_states_restored") is True
        and policy.get("restoration_errors") == [],
        "GPU model-loading tqdm restoration audit drift",
    )

    progress_before = policy.get("transformers_progress_bar_enabled_before")
    progress_during = policy.get("transformers_progress_bar_enabled_during_load")
    progress_after = policy.get("transformers_progress_bar_enabled_after_restore")
    require(
        type(progress_before) is bool
        and progress_during is False
        and type(progress_after) is bool
        and progress_after is progress_before
        and policy.get("transformers_progress_state_restored") is True,
        "GPU model-loading Transformers progress state restoration drift",
    )
    hub_registry_before = policy.get("huggingface_hub_progress_registry_before")
    hub_registry_during = policy.get("huggingface_hub_progress_registry_during_load")
    hub_registry_after = policy.get("huggingface_hub_progress_registry_after_restore")
    require(
        type(hub_registry_before) is dict
        and type(hub_registry_during) is dict
        and type(hub_registry_after) is dict
        and all(
            isinstance(key, str) and type(value) is bool
            for registry in (
                hub_registry_before,
                hub_registry_during,
                hub_registry_after,
            )
            for key, value in registry.items()
        )
        and hub_registry_during == {"_global": False},
        "GPU model-loading Hugging Face Hub progress registry is malformed",
    )
    require(
        hub_registry_after == hub_registry_before
        and policy.get("huggingface_hub_progress_registry_restored") is True
        and policy.get("huggingface_hub_progress_registry_object_identity_preserved")
        is True
        and policy.get("huggingface_hub_progress_registry_restoration_method")
        == "in_place_clear_and_update_after_official_api_restore",
        "GPU model-loading Hugging Face Hub progress registry restoration drift",
    )
    environment_before_present = policy.get("hf_hub_env_present_before")
    environment_after_present = policy.get("hf_hub_env_present_after_restore")
    environment_before_value = policy.get("hf_hub_env_value_before")
    environment_after_value = policy.get("hf_hub_env_value_after_restore")
    require(
        type(environment_before_present) is bool
        and type(environment_after_present) is bool
        and (
            (environment_before_present and isinstance(environment_before_value, str))
            or (not environment_before_present and environment_before_value is None)
        )
        and policy.get("hf_hub_env_value_during_load") == "1"
        and environment_after_present is environment_before_present
        and type(environment_after_value) is type(environment_before_value)
        and environment_after_value == environment_before_value
        and policy.get("hf_hub_env_state_restored") is True,
        "GPU model-loading HF Hub environment restoration drift",
    )

    thread_fields = (
        "noncurrent_python_threads_before_model_loading",
        "noncurrent_python_threads_after_policy_configuration",
        "noncurrent_python_threads_after_model_loading",
        "noncurrent_python_threads_after_policy_restoration",
    )
    require(
        all(policy.get(field) == [] for field in thread_fields),
        "GPU model loading created or inherited a noncurrent Python thread",
    )
    require(
        policy.get("thread_name_allowlist") == []
        and policy.get("thread_name_filtering_used") is False
        and policy.get("thread_cleanup_attempted") is False
        and policy.get("tqdm_monitor_object_touched") is False,
        "GPU model-loading thread escape-hatch policy drift",
    )
    require(
        policy.get("failure_stage") is None and policy.get("failure_reason") is None,
        "GPU model-loading PASS policy contains failure residue",
    )
    return dict(policy)


def verify_gpu_post_role_query_audit(
    foundation: Mapping[str, Any],
    gpu: Mapping[str, Any],
    *,
    run_id: str,
    goal_sha256: str,
    implementation_commit: str,
    run_manifest_path: Path,
    run_manifest_sha256: str,
    preflight: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    foundation_sha256: str,
) -> dict[str, Any]:
    """Verify post-GPU M34 role/query evidence before gate aggregation."""

    progress_policy = verify_model_loading_thread_policy(
        foundation.get("model_loading_progress_policy")
    )
    audit = foundation.get("post_gpu_role_query_audit")
    require(isinstance(audit, Mapping), "GPU post-role/query audit is missing")
    require(
        audit.get("schema_version") == "mk0_gpu_post_role_query_audit_v2",
        "GPU post-role/query audit schema drift",
    )
    require(audit.get("status") == "PASS", "GPU post-role/query audit did not pass")
    require(
        audit.get("placement")
        == "after_all_formal_gpu_generator_rate_sampler_phases_before_support_publication",
        "GPU post-role/query audit placement drift",
    )
    require(
        audit.get("runtime_instrumentation") == "sys_setprofile_python_calls",
        "GPU role/query runtime instrumentation drift",
    )
    require(
        audit.get("thread_scope")
        == (
            "formal_gpu_main_and_new_threading_threads_with_"
            "preexisting_noncurrent_threads_forbidden"
        ),
        "GPU role/query thread scope drift",
    )
    require(
        audit.get("external_call_policy") == _external_call_policy_payload(),
        "GPU external-call policy drift",
    )
    run_manifest = read_json(run_manifest_path)
    typing_binding = _typing_compatibility_shim_binding()
    if typing_binding["requirements_lock"]["reproduced"] is False:
        require(
            ENVIRONMENT_LOCK_DRIFT_MARKER in run_manifest.get("known_deviations", []),
            "GPU typing compatibility lock drift is not disclosed",
        )
    require(audit.get("run_id") == run_id, "GPU role/query run ID drift")
    require(audit.get("goal_sha256") == goal_sha256, "GPU role/query Goal drift")
    require(
        audit.get("implementation_commit") == implementation_commit,
        "GPU role/query implementation drift",
    )
    manifest = audit.get("run_manifest", {})
    require(
        manifest.get("sha256") == run_manifest_sha256,
        "GPU role/query run-manifest hash drift",
    )
    require(
        Path(manifest.get("path", "")).resolve(strict=True)
        == run_manifest_path.resolve(strict=True),
        "GPU role/query run-manifest path drift",
    )
    require(audit.get("preflight") == preflight, "GPU role/query preflight drift")
    expected_source_sha = _canonical_sha256(source_binding)
    require(
        audit.get("source_binding_sha256") == expected_source_sha,
        "GPU role/query source-binding digest drift",
    )
    require(
        audit.get("tracked_source_files_sha256")
        == source_binding.get("tracked_source_files_sha256"),
        "GPU role/query tracked-source digest drift",
    )
    source_files = source_binding.get("tracked_source_files")
    require(
        isinstance(source_files, Mapping) and source_files,
        "GPU role/query source inventory is absent",
    )
    exact_role_identities = _bound_gpu_role_identities(source_binding)

    expected_phase_ids = [item[0] for item in GPU_ROLE_PHASE_SPECS]
    require(
        audit.get("required_phase_ids") == expected_phase_ids,
        "GPU role/query required-phase inventory drift",
    )
    phases = audit.get("phase_records")
    require(
        isinstance(phases, list) and len(phases) == len(GPU_ROLE_PHASE_SPECS),
        "GPU role/query phase coverage is incomplete",
    )
    totals = {f"{category}_call_count": 0 for category in GPU_ROLE_CALL_CATEGORIES}
    totals.update(
        {
            "repository_python_call_count": 0,
            "external_python_call_count": 0,
            "unknown_external_call_count": 0,
            "total_python_call_count": 0,
        }
    )
    stream_material: list[dict[str, Any]] = []
    for phase, expected in zip(phases, GPU_ROLE_PHASE_SPECS):
        phase_id, phase_kind, entrypoint, required_categories = expected
        require(isinstance(phase, Mapping), "GPU role/query phase record is invalid")
        require(
            phase.get("phase_id") == phase_id, "GPU role/query phase order/ID drift"
        )
        require(
            phase.get("phase_kind") == phase_kind,
            f"GPU role/query phase-kind drift: {phase_id}",
        )
        require(
            phase.get("entrypoint") == entrypoint,
            f"GPU role/query entrypoint drift: {phase_id}",
        )
        require(
            phase.get("required_call_categories") == list(required_categories),
            f"GPU role/query required-call categories drift: {phase_id}",
        )
        require(
            phase.get("completed") is True,
            f"GPU formal phase did not complete: {phase_id}",
        )
        require(
            phase.get("phase_status") == "PASS" and phase.get("failure_reason") is None,
            f"GPU formal phase status failed: {phase_id}",
        )
        inventory = phase.get("call_inventory")
        require(
            isinstance(inventory, list) and inventory,
            f"GPU role/query call inventory is empty: {phase_id}",
        )
        observed_counts = {category: 0 for category in GPU_ROLE_CALL_CATEGORIES}
        repository_call_count = 0
        previous_sort_key: tuple[Any, ...] | None = None
        for record in inventory:
            require(
                isinstance(record, Mapping), "GPU role/query call record is invalid"
            )
            require(
                set(record)
                == {
                    "source_file",
                    "function_qualname",
                    "first_lineno",
                    "categories",
                    "call_count",
                },
                "GPU role/query call-record shape drift",
            )
            source_file = record["source_file"]
            qualname = record["function_qualname"]
            first_lineno = record["first_lineno"]
            call_count = record["call_count"]
            require(
                isinstance(source_file, str) and source_file in source_files,
                "GPU role/query call source is outside the bound inventory",
            )
            require(
                isinstance(qualname, str) and qualname,
                "GPU role/query call qualname is invalid",
            )
            require(
                isinstance(first_lineno, int)
                and not isinstance(first_lineno, bool)
                and first_lineno > 0,
                "GPU role/query call line is invalid",
            )
            require(
                isinstance(call_count, int)
                and not isinstance(call_count, bool)
                and call_count > 0,
                "GPU role/query call count is invalid",
            )
            expected_categories = _gpu_role_categories(
                source_file,
                qualname,
                first_lineno,
                exact_role_identities,
            )
            require(
                record["categories"] == list(expected_categories),
                "GPU role/query call category substitution",
            )
            sort_key = (source_file, qualname, first_lineno, tuple(expected_categories))
            require(
                previous_sort_key is None or sort_key > previous_sort_key,
                "GPU role/query call inventory is not canonical or is duplicated",
            )
            previous_sort_key = sort_key
            repository_call_count += call_count
            for category in expected_categories:
                observed_counts[category] += call_count
        require(
            phase.get("record_stream_sha256") == _canonical_sha256(inventory),
            f"GPU role/query phase stream digest drift: {phase_id}",
        )
        require(
            phase.get("repository_python_call_count") == repository_call_count,
            f"GPU role/query repository-call count drift: {phase_id}",
        )
        require(
            repository_call_count > 0, f"GPU role/query phase is unobserved: {phase_id}"
        )
        for category, observed in observed_counts.items():
            key = f"{category}_call_count"
            require(
                phase.get(key) == observed,
                f"GPU role/query category count drift: {phase_id}/{category}",
            )
            totals[key] += observed
        for category in required_categories:
            require(
                observed_counts[category] > 0,
                f"GPU formal phase lacks {category}: {phase_id}",
            )
        for category in GPU_ROLE_QUERY_CATEGORIES:
            require(
                observed_counts[category] == 0,
                f"GPU formal phase made a prohibited {category}: {phase_id}",
            )
        external_inventory = phase.get("external_call_inventory")
        require(
            isinstance(external_inventory, list),
            f"GPU external-call inventory is invalid: {phase_id}",
        )
        external_call_count = 0
        unknown_external_call_count = 0
        previous_external_key: tuple[Any, ...] | None = None
        for record in external_inventory:
            require(
                isinstance(record, Mapping)
                and set(record)
                == {
                    "module_name",
                    "source_file",
                    "function_qualname",
                    "first_lineno",
                    "classification",
                    "categories",
                    "call_count",
                },
                "GPU external-call record shape drift",
            )
            module_name = record["module_name"]
            external_source = record["source_file"]
            external_qualname = record["function_qualname"]
            external_line = record["first_lineno"]
            external_count = record["call_count"]
            require(
                all(
                    isinstance(value, str) and value
                    for value in (module_name, external_source, external_qualname)
                ),
                "GPU external-call identity is invalid",
            )
            require(
                isinstance(external_line, int)
                and not isinstance(external_line, bool)
                and external_line > 0,
                "GPU external-call line is invalid",
            )
            require(
                isinstance(external_count, int)
                and not isinstance(external_count, bool)
                and external_count > 0,
                "GPU external-call count is invalid",
            )
            classification, external_categories = _external_call_classification(
                module_name,
                external_source,
                external_qualname,
                external_line,
            )
            require(
                record["classification"] == classification
                and record["categories"] == list(external_categories),
                "GPU external-call classification substitution",
            )
            external_key = (
                module_name,
                external_source,
                external_qualname,
                external_line,
                classification,
                external_categories,
            )
            require(
                previous_external_key is None or external_key > previous_external_key,
                "GPU external-call inventory is not canonical or is duplicated",
            )
            previous_external_key = external_key
            external_call_count += external_count
            unknown_external_call_count += int(classification == "unknown_external") * (
                external_count
            )
            for category in external_categories:
                observed_counts[category] += external_count
        require(
            phase.get("external_record_stream_sha256")
            == _canonical_sha256(external_inventory),
            f"GPU external-call stream digest drift: {phase_id}",
        )
        require(
            phase.get("external_python_call_count") == external_call_count,
            f"GPU external-call count drift: {phase_id}",
        )
        require(
            phase.get("unknown_external_call_count")
            == unknown_external_call_count
            == 0,
            f"GPU phase contains an unknown external call: {phase_id}",
        )
        require(
            phase.get("total_python_call_count")
            == repository_call_count + external_call_count,
            f"GPU total Python-call count drift: {phase_id}",
        )
        for category in GPU_ROLE_QUERY_CATEGORIES:
            require(
                observed_counts[category] == 0,
                f"GPU external call made a prohibited {category}: {phase_id}",
            )
        thread_inventory = phase.get("thread_inventory")
        require(
            isinstance(thread_inventory, list) and thread_inventory,
            f"GPU thread inventory is absent: {phase_id}",
        )
        thread_repository_calls = 0
        thread_external_calls = 0
        previous_thread_key: tuple[int, str] | None = None
        for thread_record in thread_inventory:
            require(
                isinstance(thread_record, Mapping)
                and set(thread_record)
                == {
                    "thread_id",
                    "thread_name",
                    "repository_call_count",
                    "external_call_count",
                    "total_python_call_count",
                },
                "GPU thread record shape drift",
            )
            thread_id = thread_record["thread_id"]
            thread_name = thread_record["thread_name"]
            require(
                isinstance(thread_id, int)
                and not isinstance(thread_id, bool)
                and thread_id > 0
                and isinstance(thread_name, str)
                and thread_name,
                "GPU thread identity is invalid",
            )
            thread_key = (thread_id, thread_name)
            require(
                previous_thread_key is None or thread_key > previous_thread_key,
                "GPU thread inventory is not canonical or is duplicated",
            )
            previous_thread_key = thread_key
            repository_calls = thread_record["repository_call_count"]
            external_calls = thread_record["external_call_count"]
            require(
                isinstance(repository_calls, int)
                and not isinstance(repository_calls, bool)
                and repository_calls >= 0
                and isinstance(external_calls, int)
                and not isinstance(external_calls, bool)
                and external_calls >= 0
                and thread_record["total_python_call_count"]
                == repository_calls + external_calls
                > 0,
                "GPU thread call counts are invalid",
            )
            thread_repository_calls += repository_calls
            thread_external_calls += external_calls
        require(
            thread_repository_calls == repository_call_count
            and thread_external_calls == external_call_count,
            f"GPU thread/call inventory count drift: {phase_id}",
        )
        require(
            phase.get("python_thread_count") == len(thread_inventory),
            f"GPU thread-count drift: {phase_id}",
        )
        require(
            phase.get("thread_record_stream_sha256")
            == _canonical_sha256(thread_inventory),
            f"GPU thread stream digest drift: {phase_id}",
        )
        require(
            phase.get("unjoined_new_thread_ids") == [],
            f"GPU phase left a new thread running: {phase_id}",
        )
        require(
            phase.get("preexisting_noncurrent_python_thread_count") == 0
            and phase.get("preexisting_noncurrent_python_threads") == [],
            f"GPU phase started with a preexisting noncurrent thread: {phase_id}",
        )
        totals["repository_python_call_count"] += repository_call_count
        totals["external_python_call_count"] += external_call_count
        totals["unknown_external_call_count"] += unknown_external_call_count
        totals["total_python_call_count"] += repository_call_count + external_call_count
        stream_material.append(
            {
                "phase_id": phase_id,
                "call_inventory": inventory,
                "external_call_inventory": external_inventory,
                "thread_inventory": thread_inventory,
                "preexisting_noncurrent_python_threads": [],
            }
        )

    require(
        audit.get("formal_gpu_phase_count") == len(GPU_ROLE_PHASE_SPECS),
        "GPU role/query formal-phase count drift",
    )
    require(
        audit.get("completed_phase_count") == len(GPU_ROLE_PHASE_SPECS),
        "GPU role/query completed-phase count drift",
    )
    require(
        audit.get("record_stream_sha256") == _canonical_sha256(stream_material),
        "GPU role/query global record-stream digest drift",
    )
    for key, observed in totals.items():
        require(audit.get(key) == observed, f"GPU role/query total count drift: {key}")
    require(
        all(
            totals[f"{category}_call_count"] == 0
            for category in GPU_ROLE_QUERY_CATEGORIES
        ),
        "GPU role/query audit contains a prohibited query",
    )
    require(
        audit.get("all_role_query_counts_zero") is True,
        "GPU role/query zero-count certificate failed",
    )
    require(
        audit.get("formal_gpu_computation_complete") is True,
        "GPU role/query formal-computation completion is absent",
    )
    require(
        audit.get("all_external_calls_allowlisted") is True
        and totals["unknown_external_call_count"] == 0,
        "GPU role/query external-call allowlist failed",
    )
    require(
        audit.get("all_new_threads_joined") is True,
        "GPU role/query new-thread closure failed",
    )
    require(
        audit.get("all_preexisting_noncurrent_threads_absent") is True,
        "GPU role/query preexisting-thread exclusion failed",
    )

    interfaces = audit.get("audited_interfaces")
    require(
        isinstance(interfaces, list)
        and [record.get("interface") for record in interfaces]
        == list(GPU_ROLE_INTERFACE_LABELS),
        "GPU role/query interface inventory drift",
    )
    expected_interface_sources = {
        **{
            label: "scripts/mk0/run_mk0_gpu_smoke.py"
            for label in GPU_ROLE_INTERFACE_LABELS
            if label.startswith("gpu_runner.")
        },
        **{
            label: "core/mk0/foundation_fusion.py"
            for label in GPU_ROLE_INTERFACE_LABELS
            if label.startswith("foundation_fusion.")
        },
        **{
            label: "core/mk0/samplers.py"
            for label in GPU_ROLE_INTERFACE_LABELS
            if label.startswith("samplers.")
        },
    }
    expected_interface_qualnames = {
        "gpu_runner.forced_action_arm": "_run_forced_action_arm",
        "gpu_runner.paper_sampler_route": "_run_official_paper_sampler_route",
        "gpu_runner.primary_sampler_integration": "_run_primary_gpu_sampler_integration",
        "gpu_runner.target_alignment_leakage_audit": "_audit_target_alignment_leakage",
        "gpu_runner.dynamic_current_encoding_audit": "_audit_dynamic_current_encoding",
        "foundation_fusion.rate_field_forward": "FoundationFusionRateField.forward",
        "foundation_fusion.official_paper_adapter": "OfficialPaperRateAdapter.__call__",
        "samplers.constrained_primary": "constrained_single_event_first_order",
        "samplers.paper_parallel": "paper_first_order_parallel",
        "samplers.replay_constrained": "replay_constrained_result",
        "samplers.replay_paper": "replay_paper_result",
    }
    for record in interfaces:
        require(
            isinstance(record, Mapping), "GPU role/query interface record is invalid"
        )
        label = record["interface"]
        source_file = record.get("source_file")
        require(
            source_file == expected_interface_sources[label],
            f"GPU role/query interface source drift: {label}",
        )
        require(
            record.get("source_file_sha256") == source_files.get(source_file),
            f"GPU role/query interface source hash drift: {label}",
        )
        qualname = record.get("function_qualname")
        require(
            qualname == expected_interface_qualnames[label],
            f"GPU role/query interface qualname drift: {label}",
        )
        require(
            isinstance(record.get("first_lineno"), int)
            and not isinstance(record["first_lineno"], bool)
            and record["first_lineno"] > 0,
            f"GPU role/query interface line drift: {label}",
        )
        require(
            record.get("role_categories")
            == list(
                _gpu_role_categories(
                    source_file,
                    qualname,
                    record["first_lineno"],
                    exact_role_identities,
                )
            ),
            f"GPU role/query interface category drift: {label}",
        )
        parameters = record.get("parameters")
        require(
            isinstance(parameters, list)
            and all(isinstance(value, str) and value for value in parameters),
            f"GPU role/query interface parameter inventory invalid: {label}",
        )
        require(
            record.get("prohibited_parameters") == [],
            f"GPU role/query interface exposes a prohibited role: {label}",
        )
    require(
        audit.get("audited_interface_count") == len(GPU_ROLE_INTERFACE_LABELS),
        "GPU role/query audited-interface count drift",
    )
    require(
        audit.get("interface_failure_count") == 0,
        "GPU role/query interface audit failed",
    )

    sidecar = gpu.get("post_gpu_role_query_audit")
    require(isinstance(sidecar, Mapping), "GPU sidecar lacks post-role/query binding")
    require(
        sidecar.get("schema_version") == "mk0_gpu_post_role_query_binding_v1",
        "GPU role/query sidecar schema drift",
    )
    require(
        sidecar.get("qualifies_cpu_gate_id") == "M34",
        "GPU role/query sidecar is not bound to CPU M34",
    )
    require(
        sidecar.get("support_artifact") == "foundation_fusion_audit.json",
        "GPU role/query support artifact drift",
    )
    require(
        sidecar.get("support_artifact_sha256") == foundation_sha256,
        "GPU role/query support hash drift",
    )
    require(
        sidecar.get("audit_sha256") == _canonical_sha256(audit),
        "GPU role/query audit digest drift",
    )
    require(
        sidecar.get("record_stream_sha256") == audit["record_stream_sha256"],
        "GPU role/query sidecar stream digest drift",
    )
    require(
        sidecar.get("formal_gpu_phase_count") == len(GPU_ROLE_PHASE_SPECS),
        "GPU role/query sidecar phase-count drift",
    )
    require(
        sidecar.get("all_role_query_counts_zero") is True,
        "GPU role/query sidecar zero-count certificate failed",
    )
    return {
        "qualifies_cpu_gate_id": "M34",
        "support_artifact": "foundation_fusion_audit.json",
        "support_artifact_sha256": foundation_sha256,
        "audit_sha256": _canonical_sha256(audit),
        "record_stream_sha256": audit["record_stream_sha256"],
        "formal_gpu_phase_count": len(GPU_ROLE_PHASE_SPECS),
        "repository_python_call_count": totals["repository_python_call_count"],
        "generator_interface_call_count": totals["generator_interface_call_count"],
        "rate_interface_call_count": totals["rate_interface_call_count"],
        "sampler_interface_call_count": totals["sampler_interface_call_count"],
        "critic_query_call_count": 0,
        "guidance_query_call_count": 0,
        "final_evaluator_query_call_count": 0,
    }


def verify_runner_summaries(
    run_root: Path,
    artifact_dir: Path,
    *,
    run_id: str,
    goal_sha256: str,
    implementation_commit: str,
    run_manifest_sha256: str,
    preflight: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify the post-run terminal links which bind each results sidecar."""

    summary_dir = run_root / "summary"
    cpu_path = summary_dir / CPU_SUMMARY
    gpu_path = summary_dir / GPU_SUMMARY
    require(cpu_path.is_file(), "CPU acceptance summary is missing")
    require(gpu_path.is_file(), "GPU acceptance summary is missing")
    cpu_summary = read_json(cpu_path)
    gpu_summary = read_json(gpu_path)
    expected_source_sha = hashlib.sha256(
        canonical_json_bytes(source_binding)
    ).hexdigest()
    common = (
        (
            "CPU",
            cpu_summary,
            "mk0_cpu_acceptance_summary_v2",
            cpu.get("status"),
        ),
        (
            "GPU",
            gpu_summary,
            "mk0_gpu_acceptance_summary_v1",
            gpu.get("status"),
        ),
    )
    for runner, summary, schema, status in common:
        require(
            summary.get("schema_version") == schema, f"{runner} summary schema drift"
        )
        require(summary.get("run_id") == run_id, f"{runner} summary run ID drift")
        require(summary.get("status") == status, f"{runner} summary status drift")
        require(
            summary.get("evidence_level") == "E0_MATH_ENGINEERING_ONLY",
            f"{runner} summary evidence-level drift",
        )
        require(
            summary.get("goal_sha256") == goal_sha256, f"{runner} summary Goal drift"
        )
        require(
            summary.get("implementation_commit") == implementation_commit,
            f"{runner} summary commit drift",
        )
        require(
            Path(summary.get("run_root", "")).resolve(strict=True) == run_root,
            f"{runner} summary run-root substitution",
        )
        require(
            summary.get("preflight") == preflight, f"{runner} summary preflight drift"
        )
        require(
            summary.get("source_binding") == source_binding,
            f"{runner} summary source drift",
        )
        require(
            summary.get("source_binding_sha256") == expected_source_sha,
            f"{runner} summary source digest drift",
        )
        manifest = summary.get("run_manifest", {})
        require(
            Path(manifest.get("path", "")).resolve(strict=True)
            == run_root / "run_manifest.json",
            f"{runner} summary run-manifest path drift",
        )
        require(
            manifest.get("sha256") == run_manifest_sha256,
            f"{runner} summary run-manifest hash drift",
        )

    cpu_results_sha = sha256_file(artifact_dir / CPU_RESULTS)
    gpu_results_sha = sha256_file(artifact_dir / GPU_RESULTS)
    expected_cpu_hashes = {**cpu["artifact_hashes"], CPU_RESULTS: cpu_results_sha}
    expected_gpu_hashes = {**gpu["artifact_hashes"], GPU_RESULTS: gpu_results_sha}
    require(
        cpu_summary.get("artifact_hashes") == expected_cpu_hashes,
        "CPU summary artifact inventory/hash drift",
    )
    require(
        gpu_summary.get("artifact_hashes") == expected_gpu_hashes,
        "GPU summary artifact inventory/hash drift",
    )
    require(
        cpu_summary.get("artifact_count") == len(expected_cpu_hashes),
        "CPU summary artifact-count drift",
    )
    require(
        gpu_summary.get("artifact_count") == len(expected_gpu_hashes),
        "GPU summary artifact-count drift",
    )
    cpu_result_binding = cpu_summary.get("cpu_gate_results", {})
    require(
        Path(cpu_result_binding.get("path", "")).resolve(strict=True)
        == artifact_dir / CPU_RESULTS,
        "CPU summary results path drift",
    )
    require(
        cpu_result_binding.get("sha256") == cpu_results_sha,
        "CPU summary results hash drift",
    )
    require(
        cpu_summary.get("cpu_gate_results_sha256") == cpu_results_sha,
        "CPU summary duplicate results hash drift",
    )
    gpu_result_binding = gpu_summary.get("gpu_gate_results", {})
    require(
        Path(gpu_result_binding.get("path", "")).resolve(strict=True)
        == artifact_dir / GPU_RESULTS,
        "GPU summary results path drift",
    )
    require(
        gpu_result_binding.get("sha256") == gpu_results_sha,
        "GPU summary results hash drift",
    )
    require(
        cpu_summary.get("failed_gate_ids") == cpu.get("failed_gate_ids"),
        "CPU summary failed-gate drift",
    )
    require(
        cpu_summary.get("pending_gpu_gate_ids") == cpu.get("pending_gpu_gate_ids"),
        "CPU summary pending-gate drift",
    )
    require(
        gpu_summary.get("failed_gate_ids") == gpu.get("failed_gate_ids"),
        "GPU summary failed-gate drift",
    )
    return {
        "cpu": {"path": str(cpu_path), "sha256": sha256_file(cpu_path)},
        "gpu": {"path": str(gpu_path), "sha256": sha256_file(gpu_path)},
    }


def verify_runner_results(
    artifact_dir: Path,
    run_root: Path,
    *,
    run_id: str,
    goal_sha256: str,
    implementation_commit: str,
    preflight: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    cpu = read_json(artifact_dir / CPU_RESULTS)
    gpu = read_json(artifact_dir / GPU_RESULTS)
    require(
        cpu.get("schema_version") == "mk0_cpu_gate_results_v2",
        "CPU gate-results schema drift",
    )
    require(
        gpu.get("schema_version") == "mk0_gpu_gate_results_v1",
        "GPU gate-results schema drift",
    )
    for runner, report in (("CPU", cpu), ("GPU", gpu)):
        require(report.get("run_id") == run_id, f"{runner} run ID drift")
        require(report.get("goal_sha256") == goal_sha256, f"{runner} Goal hash drift")
        require(
            report.get("implementation_commit") == implementation_commit,
            f"{runner} commit drift",
        )
        require(
            report.get("source_binding") == source_binding,
            f"{runner} source binding drift",
        )
        runner_preflight = report.get("preflight", {})
        require(
            runner_preflight.get("path") == preflight["path"],
            f"{runner} preflight path drift",
        )
        require(
            runner_preflight.get("sha256") == preflight["sha256"],
            f"{runner} preflight hash drift",
        )
    run_manifest_path = run_root / "run_manifest.json"
    require(run_manifest_path.is_file(), "MK0 run manifest is missing")
    run_manifest_sha = sha256_file(run_manifest_path)
    require(
        cpu.get("run_manifest_path") == "run_manifest.json",
        "CPU run-manifest path drift",
    )
    require(
        cpu.get("run_manifest_sha256") == run_manifest_sha,
        "CPU run-manifest hash drift",
    )
    require(
        gpu.get("run_manifest_sha256") == run_manifest_sha,
        "GPU run-manifest hash drift",
    )
    manifest = read_json(run_manifest_path)
    require(
        manifest.get("schema_version") == "mk0_run_manifest_v3",
        "MK0 run-manifest schema drift",
    )
    require(manifest.get("run_id") == run_id, "MK0 run-manifest ID drift")
    require(manifest.get("goal_sha256") == goal_sha256, "MK0 run-manifest Goal drift")
    require(
        manifest.get("implementation_commit") == implementation_commit,
        "MK0 run-manifest commit drift",
    )
    require(
        Path(manifest.get("run_root", "")).resolve(strict=True) == run_root,
        "MK0 run-root substitution",
    )
    require(
        manifest.get("source_binding") == source_binding,
        "MK0 run-manifest source drift",
    )
    require(
        manifest.get("preflight", {}).get("sha256") == preflight["sha256"],
        "MK0 run-manifest preflight drift",
    )
    require(
        manifest.get("final_labels_accessed") is False,
        "MK0 run manifest records final-label access",
    )
    require(
        manifest.get("downstream_stage_started") is False,
        "MK0 run manifest records downstream work",
    )
    cpu_command_argv = (
        manifest.get("exact_commands", {}).get("cpu_acceptance", {}).get("argv")
    )
    require(
        isinstance(cpu_command_argv, list)
        and cpu_command_argv
        and isinstance(cpu_command_argv[0], str)
        and bool(cpu_command_argv[0]),
        "MK0 registered CPU command is absent",
    )
    pytest_evidence = verify_cpu_pytest_evidence(
        cpu,
        run_root,
        source_binding=source_binding,
        expected_python_executable=cpu_command_argv[0],
    )

    cpu_hashes = cpu.get("artifact_hashes")
    gpu_hashes = gpu.get("artifact_hashes")
    require(
        set(cpu_hashes or {}) == CPU_SUPPORT, "CPU support-artifact inventory drift"
    )
    require(
        set(gpu_hashes or {}) == GPU_SUPPORT, "GPU support-artifact inventory drift"
    )
    for name, expected in {**cpu_hashes, **gpu_hashes}.items():
        require(
            sha256_file(artifact_dir / name) == expected,
            f"runner-recorded support hash drift: {name}",
        )
    foundation = read_json(artifact_dir / "foundation_fusion_audit.json")
    leakage = read_json(artifact_dir / "target_alignment_leakage_audit.json")
    require(
        foundation.get("run_id") == run_id and leakage.get("run_id") == run_id,
        "GPU support run ID drift",
    )
    require(
        foundation.get("status") == "PASS" and leakage.get("status") == "PASS",
        "GPU support status failed",
    )
    support_bindings = {
        "M05": leakage.get("gate_binding"),
        **{
            gate_id: foundation.get("gate_bindings", {}).get(gate_id)
            for gate_id in ("M31", "M32", "M35")
        },
    }
    for gate_id, inner in support_bindings.items():
        require(
            isinstance(inner, dict) and inner,
            f"GPU support binding is absent: {gate_id}",
        )
        recorded = (
            gpu.get("gate_bindings", {})
            .get(gate_id, {})
            .get("metrics", {})
            .get("support_gate_binding_sha256")
        )
        require(
            recorded == hashlib.sha256(canonical_json_bytes(inner)).hexdigest(),
            f"GPU sidecar/support gate-binding drift: {gate_id}",
        )
    cuda = foundation.get("cuda", {})
    require(
        cuda.get("cpu_fallback_observed") is False
        and cuda.get("cpu_fallback_allowed") is False,
        "CPU neural fallback observed/allowed",
    )
    require(cuda.get("cuda_tensor_evidence") is True, "CUDA tensor evidence is absent")
    require(
        int(cuda.get("max_memory_allocated_bytes", 0)) > 0,
        "CUDA memory evidence is absent",
    )
    cpu_m34 = cpu.get("gate_bindings", {}).get("M34")
    require(
        isinstance(cpu_m34, Mapping),
        "CPU M34 binding is absent before GPU post-role audit",
    )
    require(
        cpu_m34.get("gate_id") == "M34"
        and cpu_m34.get("artifact_path") == "artifacts/mk0/critic_role_audit.json",
        "CPU M34 binding identity drift before GPU post-role audit",
    )
    gpu_post_role_query = verify_gpu_post_role_query_audit(
        foundation,
        gpu,
        run_id=run_id,
        goal_sha256=goal_sha256,
        implementation_commit=implementation_commit,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha,
        preflight=preflight,
        source_binding=source_binding,
        foundation_sha256=sha256_file(artifact_dir / "foundation_fusion_audit.json"),
    )
    summaries = verify_runner_summaries(
        run_root,
        artifact_dir,
        run_id=run_id,
        goal_sha256=goal_sha256,
        implementation_commit=implementation_commit,
        run_manifest_sha256=run_manifest_sha,
        preflight=preflight,
        source_binding=source_binding,
        cpu=cpu,
        gpu=gpu,
    )
    summaries["gpu_post_role_query_audit"] = gpu_post_role_query
    summaries["cpu_pytest_evidence"] = pytest_evidence
    return cpu, gpu, foundation, leakage, summaries


def verify_run_contract_registration(
    run_root: Path,
    *,
    run_id: str,
    parent_run_id: str | None,
    goal_sha256: str,
    implementation_commit: str,
    preflight: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    actual_finalizer_argv: list[str],
) -> dict[str, Any]:
    """Verify the immutable section-19 registration and preterminal tree."""

    match = FORMAL_RUN_ID.fullmatch(run_id)
    require(match is not None, "formal run ID violates section 19.2")
    require(
        match.group("model") == "utrlm"
        and match.group("dataset") == "mathkernel"
        and match.group("split") == "tiny",
        "formal run ID model/dataset/split semantics drift",
    )
    require(
        match.group("short_sha")
        == implementation_commit[: len(match.group("short_sha"))],
        "formal run ID short SHA differs from implementation commit",
    )
    require(int(match.group("seed")) == 20260802, "formal run ID seed drift")
    try:
        run_id_time = datetime.strptime(match.group("utc"), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise FinalizeFailure("formal run ID UTC is not a calendar time") from error
    require(
        run_id_time.strftime("%Y%m%dT%H%M%SZ") == match.group("utc"),
        "formal run ID UTC is not canonical",
    )
    parent_time: datetime | None = None
    if parent_run_id is not None:
        parent_match = FORMAL_RUN_ID.fullmatch(parent_run_id)
        require(parent_match is not None, "parent run ID is not formal")
        try:
            parent_time = datetime.strptime(
                parent_match.group("utc"), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError as error:
            raise FinalizeFailure("parent run ID UTC is not a calendar time") from error
        require(
            parent_time.strftime("%Y%m%dT%H%M%SZ") == parent_match.group("utc"),
            "parent run ID UTC is not canonical",
        )
        require(parent_time < run_id_time, "parent run ID must precede child run ID")
    require(run_root.name == run_id, "formal run root basename drift")
    require(
        run_root.parent == CANONICAL_RUN_PARENT,
        "formal run root violates section 19.1 canonical parent",
    )
    for directory in RUN_DIRECTORIES:
        require(
            (run_root / directory).is_dir(),
            f"run-contract directory missing: {directory}",
        )
    for name in LOG_FILENAMES:
        require(
            (run_root / "logs" / name).is_file(), f"run-contract log missing: {name}"
        )
    for relative in (
        "run_manifest.json",
        "resolved_config.yaml",
        "command.txt",
        "status.json",
        "provenance/goal_contract.sha256",
        "provenance/data_manifest.json",
        "provenance/split_manifest.json",
        "provenance/foundation_manifest.json",
        "provenance/exposure_ledger.json",
        "provenance/code_manifest.json",
        "provenance/cpu_command.json",
        "provenance/gpu_command.json",
        "provenance/pytest_import_binding.json",
        "git/commit.txt",
        "git/diff.patch",
        "git/diff.sha256",
        "checkpoints/NOT_APPLICABLE.json",
        "checkpoints/checksums.sha256",
        "summary/cpu_acceptance_summary.json",
        "summary/gpu_acceptance_summary.json",
        "evaluation/pytest_mk0.junit.xml",
    ):
        require(
            (run_root / relative).is_file(),
            f"run-contract artifact missing: {relative}",
        )
    require(not (run_root / "DONE").exists(), "DONE existed before final verification")
    require(
        not (run_root / "FAILED").exists(), "FAILED run cannot be finalized as PASS"
    )
    require(
        not any((run_root / "failure").iterdir()),
        "run contains failure evidence and cannot be finalized as PASS",
    )
    require(
        not list((run_root / "artifacts" / "mk0").glob("*failure*.json")),
        "GPU/CPU support directory contains failure evidence",
    )

    manifest_path = run_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version") == "mk0_run_manifest_v3",
        "MK0 run-manifest schema drift",
    )
    require(manifest.get("run_id") == run_id, "MK0 run-manifest ID drift")
    require(manifest.get("task_id") == "MK0-01", "MK0 run-manifest task drift")
    require(
        manifest.get("parent_run_id") == parent_run_id,
        "MK0 run-manifest parent drift",
    )
    parent_binding = manifest.get("parent_run_binding")
    if parent_run_id is None:
        require(parent_binding is None, "unexpected MK0 parent binding")
    else:
        parent_binding = verify_parent_run_binding(
            parent_binding,
            parent_run_id=parent_run_id,
            goal_sha256=goal_sha256,
        )
    require(
        preflight.get("parent_run_id") == parent_run_id,
        "MK0 preflight/registration parent drift",
    )
    require(manifest.get("phase") == "MK0", "MK0 run-manifest phase drift")
    require(
        manifest.get("hypotheses")
        == {
            key: "NOT_TESTED_AT_MK0_E0"
            for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
        },
        "MK0 H1-H8 boundary drift",
    )
    require(
        manifest.get("evidence_level") == EVIDENCE_LEVEL,
        "MK0 run-manifest evidence-level drift",
    )
    require(
        manifest.get("contract", {}).get("sha256") == goal_sha256
        and manifest.get("contract", {}).get("scope") == "MK0_ONLY",
        "MK0 run-manifest contract drift",
    )
    require(
        manifest.get("implementation_commit") == implementation_commit
        and manifest.get("code", {}).get("commit") == implementation_commit,
        "MK0 run-manifest commit drift",
    )
    require(
        manifest.get("source_binding") == source_binding,
        "MK0 run-manifest source drift",
    )
    require(
        manifest.get("preflight", {}).get("sha256") == preflight["sha256"],
        "MK0 run-manifest preflight drift",
    )
    require(
        manifest.get("seed") == 20260802,
        "MK0 run-manifest seed drift",
    )
    require(
        manifest.get("gpu_uuid")
        in {
            gpu.get("uuid")
            for gpu in read_json(Path(preflight["path"]))["resources"]["gpus"]
        },
        "MK0 run-manifest GPU UUID was absent from preflight",
    )
    timing = manifest.get("timing", {})
    require(
        isinstance(timing.get("start_utc"), str) and timing.get("end_utc") is None,
        "MK0 immutable registration timing drift",
    )
    try:
        registration_time = datetime.fromisoformat(
            timing["start_utc"].replace("Z", "+00:00")
        )
        preflight_time = datetime.fromisoformat(
            read_json(Path(preflight["path"]))["observed_at_utc"].replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FinalizeFailure("MK0 registration/preflight UTC is invalid") from error
    require(
        run_id_time <= preflight_time <= registration_time,
        "MK0 run-ID/preflight/registration chronology drift",
    )
    process = manifest.get("process_identity", {})
    require(
        isinstance(process.get("cpu_pid"), int)
        and process["cpu_pid"] > 0
        and process.get("gpu_pid") is None
        and process.get("finalizer_pid") is None,
        "MK0 immutable registration process identity drift",
    )
    require(
        manifest.get("exit_code") is None
        and manifest.get("stop_reason") == "RUNNING_PENDING_GPU_AND_FINALIZER",
        "MK0 immutable registration terminal placeholders drift",
    )
    require(
        manifest.get("paper_eligibility", {}).get("eligible") is False,
        "MK0 E0 run cannot be paper-eligible",
    )
    require(
        isinstance(manifest.get("known_deviations"), list)
        and manifest["known_deviations"],
        "MK0 known deviations are absent",
    )
    require(
        manifest.get("final_labels_accessed") is False
        and manifest.get("downstream_stage_started") is False,
        "MK0 run-manifest scope boundary drift",
    )

    resolved = manifest.get("resolved_config", {})
    require(
        Path(str(resolved.get("path", ""))).resolve(strict=True)
        == run_root / "resolved_config.yaml"
        and resolved.get("sha256") == sha256_file(run_root / "resolved_config.yaml"),
        "MK0 resolved-config binding drift",
    )
    provenance = manifest.get("provenance", {})
    expected_provenance = {
        "goal_contract.sha256",
        "data_manifest.json",
        "split_manifest.json",
        "foundation_manifest.json",
        "exposure_ledger.json",
        "code_manifest.json",
    }
    require(set(provenance) == expected_provenance, "MK0 provenance inventory drift")
    for name in expected_provenance:
        bound = provenance[name]
        path = run_root / "provenance" / name
        require(
            Path(bound["path"]).resolve(strict=True) == path
            and bound["sha256"] == sha256_file(path),
            f"MK0 provenance binding drift: {name}",
        )
    goal_sidecar = (run_root / "provenance" / "goal_contract.sha256").read_text(
        encoding="utf-8"
    )
    require(
        goal_sidecar
        == (
            f"{goal_sha256}  USER_DECLARED_SOLE_CONTRACT:"
            "mrna_latest_build_contract_first.md\n"
        ),
        "MK0 Goal-contract checksum sidecar drift",
    )
    require(
        (run_root / "git" / "commit.txt").read_text(encoding="utf-8").strip()
        == implementation_commit,
        "MK0 Git commit artifact drift",
    )
    require(
        (run_root / "git" / "diff.patch").read_bytes() == b""
        and (run_root / "git" / "diff.sha256").read_text(encoding="utf-8").strip()
        == hashlib.sha256(b"").hexdigest(),
        "MK0 clean-diff artifact drift",
    )
    verify_checksum_ledger(
        run_root / "checkpoints", run_root / "checkpoints" / "checksums.sha256"
    )
    checkpoint_na = read_json(run_root / "checkpoints" / "NOT_APPLICABLE.json")
    require(
        checkpoint_na.get("last_healthy_checkpoint") == "MK0_NOT_APPLICABLE"
        and checkpoint_na.get("best_primary_checkpoint") == "MK0_NOT_APPLICABLE",
        "MK0 checkpoint N/A semantics drift",
    )

    commands = manifest.get("exact_commands", {})
    require(
        set(commands) == {"cpu_acceptance", "gpu_smoke", "finalizer"},
        "MK0 exact-command inventory drift",
    )
    require(
        commands["finalizer"].get("argv") == actual_finalizer_argv
        and commands["finalizer"].get("environment") == {},
        "actual finalizer command differs from registration",
    )
    cpu_command = read_json(run_root / "provenance" / "cpu_command.json")
    gpu_command = read_json(run_root / "provenance" / "gpu_command.json")
    require(
        cpu_command.get("schema_version") == "mk0_cpu_command_v1"
        and cpu_command.get("run_id") == run_id
        and cpu_command.get("argv") == commands["cpu_acceptance"]["argv"]
        and cpu_command.get("environment") == commands["cpu_acceptance"]["environment"]
        and cpu_command.get("pid") == process["cpu_pid"]
        and cpu_command.get("cwd") == str(REPO_ROOT)
        and cpu_command.get("python_executable")
        == commands["cpu_acceptance"]["argv"][0]
        and isinstance(cpu_command.get("python_version"), str)
        and bool(cpu_command["python_version"])
        and cpu_command.get("neural_forward_allowed") is False,
        "actual CPU command/process identity differs from registration",
    )
    require(
        gpu_command.get("schema_version") == "mk0_gpu_command_v1"
        and gpu_command.get("run_id") == run_id
        and gpu_command.get("argv") == commands["gpu_smoke"]["argv"]
        and gpu_command.get("environment") == commands["gpu_smoke"]["environment"]
        and isinstance(gpu_command.get("pid"), int)
        and gpu_command["pid"] > 0
        and gpu_command.get("cwd") == str(REPO_ROOT),
        "actual GPU command/process differs from registration",
    )
    command_lines = (run_root / "command.txt").read_text(encoding="utf-8").splitlines()
    require(
        command_lines
        == [
            f"{name}: {commands[name]['shell_escaped']}"
            for name in ("cpu_acceptance", "gpu_smoke", "finalizer")
        ],
        "MK0 command.txt differs from registration",
    )
    status = read_json(run_root / "status.json")
    require(
        status.get("state") == "GPU_VERIFIED_PENDING_FINALIZER"
        and status.get("terminal") is False,
        "MK0 finalizer requires GPU-verified nonterminal status",
    )
    for log_name in ("metrics.jsonl", "system_metrics.jsonl", "events.jsonl"):
        lines = (run_root / "logs" / log_name).read_text(encoding="utf-8").splitlines()
        require(lines, f"MK0 structured log is empty: {log_name}")
        for line in lines:
            require(
                isinstance(json.loads(line), dict),
                f"MK0 JSONL record invalid: {log_name}",
            )
    return {
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "registered_cpu_pid": process["cpu_pid"],
        "observed_gpu_pid": gpu_command["pid"],
        "registered_gpu_uuid": manifest["gpu_uuid"],
        "parent_run_id": parent_run_id,
        "parent_run_binding": parent_binding,
        "section_19_tree_verified": True,
        "local_structured_logs_verified": True,
        "checkpoint_semantics": "MK0_NOT_APPLICABLE",
        "paper_eligibility": False,
    }


def validated_gate_results(
    config: Mapping[str, Any],
    run_root: Path,
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> list[GateResult]:
    cpu_bindings = cpu.get("gate_bindings")
    gpu_bindings = gpu.get("gate_bindings")
    require(isinstance(cpu_bindings, dict), "CPU gate bindings are absent")
    require(isinstance(gpu_bindings, dict), "GPU gate bindings are absent")
    configured_ids = [gate["id"] for gate in config["acceptance"]["gates"]]
    require(
        configured_ids == [f"M{i:02d}" for i in range(1, 36)], "frozen gate order drift"
    )
    require(set(gpu_bindings) == GPU_GATE_IDS, "GPU gate binding coverage drift")
    require(
        set(cpu_bindings) == set(configured_ids) - GPU_GATE_IDS,
        "CPU gate binding coverage drift",
    )
    records: list[GateResult] = []
    for gate_config in config["acceptance"]["gates"]:
        gate_id = gate_config["id"]
        binding = (
            gpu_bindings[gate_id] if gate_id in GPU_GATE_IDS else cpu_bindings[gate_id]
        )
        relative = Path(gate_config["artifact_path"])
        support = (run_root / relative).resolve(strict=True)
        try:
            support.relative_to(run_root)
        except ValueError as error:
            raise FinalizeFailure(
                f"gate {gate_id} support path escaped run root"
            ) from error
        require(support.is_file(), f"gate {gate_id} support artifact is missing")
        try:
            record = gate_result_from_runtime_binding(
                binding,
                gate_config,
                actual_artifact_sha256=sha256_file(support),
            )
        except ValueError as error:
            raise FinalizeFailure(str(error)) from error
        records.append(record)
    derived_cpu_failed = [
        item.gate_id
        for item in records
        if item.gate_id not in GPU_GATE_IDS and not item.passed
    ]
    derived_gpu_failed = [
        item.gate_id
        for item in records
        if item.gate_id in GPU_GATE_IDS and not item.passed
    ]
    require(
        cpu.get("failed_gate_ids") == derived_cpu_failed,
        "CPU failed-gate summary drift",
    )
    require(
        gpu.get("failed_gate_ids") == derived_gpu_failed,
        "GPU failed-gate summary drift",
    )
    require(
        cpu.get("status")
        == (
            "PASS_CPU_GATES_PENDING_GPU"
            if not derived_cpu_failed
            else "FAILED_WITH_EVIDENCE"
        ),
        "CPU status is not derived from bindings",
    )
    require(
        gpu.get("status")
        == ("PASS_GPU_GATES" if not derived_gpu_failed else "FAILED_WITH_EVIDENCE"),
        "GPU status is not derived from bindings",
    )
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--goal-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--fm0-closure-root", type=Path, required=True)
    parser.add_argument("--d1-data", type=Path, required=True)
    parser.add_argument("--d1-ledger", type=Path, required=True)
    parser.add_argument("--preflight-record", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    actual_finalizer_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *raw_argv,
    ]
    run_root = args.run_root.resolve()
    artifact_dir = run_root / "artifacts" / "mk0"
    bound_run_id: str | None = None
    manifest_path = run_root / "run_manifest.json"
    if Path.cwd().resolve(strict=True) != REPO_ROOT:
        print(
            "MK0 finalization failed: finalizer must be launched from the worktree",
            file=sys.stderr,
        )
        return 1
    if manifest_path.is_file():
        try:
            bound_run_id = read_json(manifest_path).get("run_id")
        except BaseException:
            bound_run_id = None
    if isinstance(bound_run_id, str):
        terminal_before = (run_root / "FAILED").exists() or (run_root / "DONE").exists()
        terminal = resume_failure_closure_if_present(run_root, run_id=bound_run_id)
        if terminal is not None:
            requested_matches = args.run_id == bound_run_id
            try:
                print(
                    json.dumps(
                        {
                            "status": (
                                f"ALREADY_{terminal}"
                                if requested_matches
                                else "TERMINAL_BOUND_TO_DIFFERENT_REQUEST"
                            ),
                            "run_id": bound_run_id,
                            "requested_run_id": args.run_id,
                            "tree_mutated": not terminal_before,
                            "tree_mutated_only_to_finish_failure_closure": (
                                not terminal_before and terminal == "FAILED"
                            ),
                        },
                        sort_keys=True,
                    )
                )
            except BaseException:
                pass
            if not requested_matches:
                return 2
            return 0 if terminal == "DONE" else 1
    try:
        require(FORMAL_RUN_ID.fullmatch(args.run_id) is not None, "formal run ID drift")
        require(HEX64.fullmatch(args.goal_sha256) is not None, "invalid Goal hash")
        require(
            HEX40.fullmatch(args.implementation_commit) is not None,
            "invalid implementation commit",
        )
        require(
            run_root.is_dir() and artifact_dir.is_dir(), "formal run root is incomplete"
        )
        require(
            not (artifact_dir / "mk0_acceptance.json").exists(),
            "acceptance already exists",
        )
        require(
            not (artifact_dir / "mk0_freeze_manifest.json").exists(),
            "freeze manifest already exists",
        )
        for name in MANDATORY_SUPPORT:
            require(
                (artifact_dir / name).is_file(),
                f"missing mandatory support artifact: {name}",
            )

        config_path = REPO_ROOT / "configs" / "math" / "math_kernel_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(
            config["contract"]["sha256"] == args.goal_sha256, "config Goal hash drift"
        )
        source_binding = verify_current_source_binding(
            args.implementation_commit,
            read_json(artifact_dir / CPU_RESULTS).get("source_binding", {}),
        )
        fm0 = verify_fm0_b0_d1(
            args.fm0_closure_root,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            d1_data=args.d1_data,
            d1_ledger=args.d1_ledger,
        )
        preflight = verify_preflight(
            args.preflight_record,
            run_id=args.run_id,
            parent_run_id=args.parent_run_id,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            fm0=fm0,
        )
        run_contract_registration = verify_run_contract_registration(
            run_root,
            run_id=args.run_id,
            parent_run_id=args.parent_run_id,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            preflight=preflight,
            source_binding=source_binding,
            actual_finalizer_argv=actual_finalizer_argv,
        )
        write_new(
            run_root / "provenance" / "finalizer_command.json",
            {
                "schema_version": "mk0_finalizer_command_v1",
                "run_id": args.run_id,
                "created_at_utc": utc_now(),
                "argv": actual_finalizer_argv,
                "environment": {},
                "pid": os.getpid(),
                "cwd": str(REPO_ROOT),
            },
        )
        append_event(
            run_root,
            "FINALIZER_STARTED",
            run_id=args.run_id,
            pid=os.getpid(),
        )
        cpu, gpu, foundation, leakage, runner_summaries = verify_runner_results(
            artifact_dir,
            run_root,
            run_id=args.run_id,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            preflight=preflight,
            source_binding=source_binding,
        )
        runner_summaries["run_contract_registration"] = run_contract_registration
        snapshot_provenance = verify_foundation_snapshot_provenance(foundation, fm0)
        runner_summaries["foundation_snapshot_provenance"] = snapshot_provenance
        preflight_gpu_identity = verify_preflight_gpu_execution_identity(
            args.preflight_record,
            foundation,
            expected_preflight_sha256=preflight["sha256"],
        )
        runner_summaries["preflight_gpu_execution_identity"] = preflight_gpu_identity
        gates = validated_gate_results(config, run_root, cpu, gpu)
        post_gpu_m34 = runner_summaries.get("gpu_post_role_query_audit")
        require(
            isinstance(post_gpu_m34, Mapping)
            and post_gpu_m34.get("qualifies_cpu_gate_id") == "M34",
            "CPU M34 is preliminary until the post-GPU role/query binding passes",
        )
        m34_preliminary = next(record for record in gates if record.gate_id == "M34")
        m34_qualification = {
            "semantics": "CPU_M34_PRELIMINARY_PLUS_REQUIRED_POST_GPU_ROLE_QUERY_AUDIT",
            "cpu_preliminary_pass": m34_preliminary.passed,
            "cpu_binding_sha256": _canonical_sha256(cpu["gate_bindings"]["M34"]),
            "post_gpu_required": True,
            "post_gpu_verified": True,
            "post_gpu_audit_sha256": post_gpu_m34["audit_sha256"],
            "post_gpu_support_sha256": post_gpu_m34["support_artifact_sha256"],
            "post_gpu_record_stream_sha256": post_gpu_m34["record_stream_sha256"],
            "final_m34_accepted": m34_preliminary.passed,
        }
        gates = [
            (
                replace(
                    record,
                    metrics={
                        **record.metrics,
                        "two_stage_qualification": m34_qualification,
                    },
                )
                if record.gate_id == "M34"
                else record
            )
            for record in gates
        ]
        acceptance = aggregate_acceptance(
            gates, run_id=args.run_id, goal_sha256=args.goal_sha256
        )
        acceptance["m34_two_stage_qualification"] = m34_qualification
        if acceptance["pass"] is not True:
            write_new(
                run_root / "failure" / "mk0_gate_acceptance_failure.json",
                {
                    **acceptance,
                    "created_at_utc": utc_now(),
                    "implementation_commit": args.implementation_commit,
                    "source_binding_sha256": source_binding[
                        "tracked_source_files_sha256"
                    ],
                    "run_manifest_sha256": cpu["run_manifest_sha256"],
                    "preflight_sha256": preflight["sha256"],
                    "cpu_gate_results_sha256": sha256_file(artifact_dir / CPU_RESULTS),
                    "gpu_gate_results_sha256": sha256_file(artifact_dir / GPU_RESULTS),
                    "runner_summaries": runner_summaries,
                    "upstream": fm0,
                },
            )
            raise FinalizeFailure(f"MK0 gates failed: {acceptance['failed_gate_ids']}")
        acceptance.update(
            {
                "created_at_utc": utc_now(),
                "implementation_commit": args.implementation_commit,
                "source_binding_sha256": source_binding["tracked_source_files_sha256"],
                "run_manifest_sha256": cpu["run_manifest_sha256"],
                "preflight_sha256": preflight["sha256"],
                "cpu_gate_results_sha256": sha256_file(artifact_dir / CPU_RESULTS),
                "gpu_gate_results_sha256": sha256_file(artifact_dir / GPU_RESULTS),
                "runner_summaries": runner_summaries,
                "run_root": str(run_root),
                "cpu_neural_forward_count": 0,
                "gpu_neural_forward_required": True,
                "gpu_neural_forward_observed": True,
                "gpu_device_uuid": foundation["cuda"]["device_uuid"],
                "upstream": fm0,
                "failure_records": [],
            }
        )
        acceptance_sha = write_new(artifact_dir / "mk0_acceptance.json", acceptance)

        support_hashes = {
            name: sha256_file(artifact_dir / name) for name in sorted(MANDATORY_SUPPORT)
        }
        support_hashes["mk0_acceptance.json"] = acceptance_sha
        freeze_manifest = {
            "schema_version": "mk0_freeze_manifest_v2",
            "run_id": args.run_id,
            "status": acceptance["status"],
            "created_at_utc": utc_now(),
            "goal": {
                "contract_path_declared_by_user": "/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 最新构建合同-先做.md",
                "sha256": args.goal_sha256,
                "section": "31A",
            },
            "code": {
                "repo_root": str(REPO_ROOT),
                "implementation_commit": args.implementation_commit,
                "git_tree": source_binding["git_tree"],
                "tracked_source_file_count": source_binding[
                    "tracked_source_file_count"
                ],
                "tracked_source_files_sha256": source_binding[
                    "tracked_source_files_sha256"
                ],
                "dirty_diff_allowed_during_formal_run": False,
                "implementation_commit_role": "pre_evidence_code_docs_config_schemas_tests",
                "closure_commit_role": "post_run_transport_metadata_recorded_after_manifest_copy",
                "self_commit_note": "A manifest cannot contain the Git commit that contains its own bytes; a subsequent focused closure commit records transport metadata.",
            },
            "formula": {
                "version": "mk0_bregman_transition_aggregate_v1",
                "time_direction": "source_at_0_to_target_at_1",
                "coupling": "unit_cost_levenshtein_optimal_canonical_with_independent_switch_clock_product",
                "schedule_primary": "cubic",
                "schedule_sensitivity": "linear",
                "time_eps": 1.0e-4,
                "sampler_primary": "constrained_single_event_first_order",
                "sampler_reference": "paper_first_order_parallel",
                "exact_gillespie": False,
                "stop_primary": "post_completion_exponential_dwell_survival",
                "gamma_ref": 16.0,
            },
            "foundation": {
                "model_id": foundation["snapshot_binding"]["model_id"],
                "revision": foundation["snapshot_binding"]["observed_revision"],
                "snapshot_manifest_sha256": foundation["snapshot_binding"][
                    "snapshot_manifest_sha256"
                ],
                "fm0_checkpoint_provenance": snapshot_provenance,
                "preflight_gpu_execution_identity": preflight_gpu_identity,
                "foundation_class": foundation["arms"]["official_frozen"][
                    "foundation_class"
                ],
                "license": snapshot_provenance["license_type"],
                "dynamic_current_full_reencode": True,
                "incremental_update_enabled": False,
                "placeholder_foundation_forward_calls": foundation["placeholder_audit"][
                    "placeholder_foundation_forward_calls"
                ],
            },
            "upstream": fm0,
            "preflight": preflight,
            "runner_summaries": runner_summaries,
            "environment": {
                "python": platform.python_version(),
                "torch": foundation["cuda"]["torch_version"],
                "cuda": foundation["cuda"]["torch_cuda_version"],
                "gpu_name": foundation["cuda"]["device_name"],
                "gpu_uuid": foundation["cuda"]["device_uuid"],
                "cpu_fallback_allowed": False,
                "cpu_fallback_observed": False,
                "repository_lock_reproduced": False,
                "environment_drift_disclosed": True,
            },
            "tests": {
                "pytest_passed": cpu["pytest"]["passed_count"],
                "pytest_failed": cpu["pytest"]["failed_count"],
                "pytest_skipped": cpu["pytest"]["skipped_count"],
                "acceptance_gate_count": len(gates),
                "failed_gate_ids": acceptance["failed_gate_ids"],
                "gpu_forced_actions": ["INS", "SUB", "DEL", "STOP"],
            },
            "artifacts": support_hashes,
            "run_manifest_sha256": cpu["run_manifest_sha256"],
            "run_root": str(run_root),
            "failure_records": [],
            "evidence_boundary": {
                "level": "E0_MATH_ENGINEERING_ONLY",
                "functional_improvement": False,
                "matched_budget_superiority": False,
                "paper_success": False,
                "observed_biological_trajectory": False,
            },
            "original_method_provenance": {
                "paper": "Edit Flows: Variable Length Discrete Flow Matching with Sequence-Level Edit Operations",
                "arxiv": "2506.09018v3",
                "publication": "NeurIPS 2025",
                "official_public_implementation_status": "NOT_VERIFIED_PUBLIC",
            },
        }
        freeze_sha = write_new(
            artifact_dir / "mk0_freeze_manifest.json", freeze_manifest
        )
        sidecar = artifact_dir / "mk0_freeze_manifest.sha256"
        with sidecar.open("xb") as handle:
            handle.write(f"{freeze_sha}  mk0_freeze_manifest.json\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())

        checksum_lines = []
        for path in sorted(artifact_dir.iterdir()):
            if path.is_file() and path.name != "artifact_checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.name}")
        checksum_path = artifact_dir / "artifact_checksums.sha256"
        with checksum_path.open("xb") as handle:
            handle.write(("\n".join(checksum_lines) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())

        resolved_config = json.loads(json.dumps(config))
        resolved_config["provenance"]["project_reference_implementation"][
            "git_commit"
        ] = args.implementation_commit
        resolved_config["provenance"]["project_reference_implementation"][
            "dirty_diff_sha256"
        ] = hashlib.sha256(b"").hexdigest()
        provenance_dir = run_root / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = provenance_dir / "resolved_math_kernel_v1.yaml"
        with resolved_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)

        summary = {
            "schema_version": "mk0_final_summary_v2",
            "run_id": args.run_id,
            "status": acceptance["status"],
            "evidence_level": "E0_MATH_ENGINEERING_ONLY",
            "goal_sha256": args.goal_sha256,
            "implementation_commit": args.implementation_commit,
            "acceptance_sha256": acceptance_sha,
            "freeze_manifest_sha256": freeze_sha,
            "artifact_checksum_ledger_sha256": sha256_file(checksum_path),
            "gpu_uuid": foundation["cuda"]["device_uuid"],
            "failed_gate_ids": acceptance["failed_gate_ids"],
            "downstream_started": False,
        }
        write_new(run_root / "summary" / "mk0_final_summary.json", summary)
        terminal_parent_binding: dict[str, Any] | None = None
        if args.parent_run_id is not None:
            terminal_parent_binding = verify_parent_run_binding(
                run_contract_registration["parent_run_binding"],
                parent_run_id=args.parent_run_id,
                goal_sha256=args.goal_sha256,
            )
        terminal_at = utc_now()
        root_summary = {
            **summary,
            "schema_version": "mk0_root_summary_v1",
            "status": "DONE",
            "task_id": "MK0-01",
            "terminal_at_utc": terminal_at,
            "paper_eligibility": False,
            "scientific_claim_boundary": "E0_MATH_ENGINEERING_ONLY",
            "whole_run_checksum_ledger": {
                "path": str(run_root / "artifact_checksums.sha256"),
                "state": "TERMINAL_FILES_COVERED_THEN_VERIFIED_BEFORE_DONE",
                "publication_protocol": "all_terminal_files_then_verified_ledger_then_DONE",
                "self_reference_exception": "ledger and final DONE sentinel are not ledger entries; DONE binds the ledger hash",
            },
        }
        write_new(run_root / "summary.json", root_summary)
        append_jsonl(
            run_root / "logs" / "metrics.jsonl",
            {
                "created_at_utc": terminal_at,
                "stage": "FINAL_ACCEPTANCE",
                "status": acceptance["status"],
                "passed_gate_count": len(gates),
                "failed_gate_ids": [],
                "gpu_uuid": foundation["cuda"]["device_uuid"],
                "cpu_fallback_count": 0,
            },
        )
        append_jsonl(
            run_root / "logs" / "system_metrics.jsonl",
            {
                "created_at_utc": terminal_at,
                "event": "FINAL_ACCEPTANCE_COMPLETED",
                "gpu_uuid": foundation["cuda"]["device_uuid"],
                "run_root": str(run_root),
                "finalizer_pid": os.getpid(),
            },
        )
        append_text(
            run_root / "logs" / "stdout.log",
            f"{terminal_at} MK0 M01-M35 accepted at E0; terminal closure starting\n",
        )
        update_status(
            run_root,
            run_id=args.run_id,
            state="CLOSED_ACCEPTED",
            terminal=True,
            stop_reason="ALL_M01_M35_PASSED",
            exit_code=0,
        )

        completion_path = run_root / "summary" / "run_completion_manifest.json"
        whole_run_ledger_path = run_root / "artifact_checksums.sha256"
        registration_manifest = read_json(run_root / "run_manifest.json")
        completion = {
            "schema_version": "mk0_run_completion_manifest_v1",
            "run_id": args.run_id,
            "task_id": "MK0-01",
            "parent_run_id": registration_manifest["parent_run_id"],
            "parent_run_binding": terminal_parent_binding,
            "status": "DONE",
            "evidence_level": EVIDENCE_LEVEL,
            "hypotheses": registration_manifest["hypotheses"],
            "contract": registration_manifest["contract"],
            "code": registration_manifest["code"],
            "exact_commands": registration_manifest["exact_commands"],
            "resolved_config": registration_manifest["resolved_config"],
            "data": registration_manifest["data"],
            "split": registration_manifest["split"],
            "foundation": registration_manifest["foundation"],
            "exposure_ledger": registration_manifest["exposure_ledger"],
            "seed": registration_manifest["seed"],
            "gpu_uuid": foundation["cuda"]["device_uuid"],
            "timing": {
                "start_utc": registration_manifest["timing"]["start_utc"],
                "end_utc": terminal_at,
            },
            "process_identity": {
                "cpu_pid": run_contract_registration["registered_cpu_pid"],
                "gpu_pid": run_contract_registration["observed_gpu_pid"],
                "finalizer_pid": os.getpid(),
                "tmux_pane": registration_manifest["process_identity"]["tmux_pane"],
                "job_id": registration_manifest["process_identity"]["job_id"],
            },
            "exit_code": 0,
            "stop_reason": "ALL_M01_M35_PASSED",
            "artifact_checksums": {
                "whole_run_ledger_path": str(whole_run_ledger_path),
                "whole_run_ledger_excludes_itself": True,
                "whole_run_ledger_excludes_final_DONE_sentinel": True,
                "final_DONE_sentinel_binds_ledger_sha256": True,
            },
            "paper_eligibility": registration_manifest["paper_eligibility"],
            "known_deviations": registration_manifest["known_deviations"],
            "final_labels_accessed": False,
            "downstream_stage_started": False,
            "acceptance": {
                "path": str(artifact_dir / "mk0_acceptance.json"),
                "sha256": acceptance_sha,
                "gate_count": len(gates),
                "failed_gate_ids": [],
            },
            "freeze_manifest": {
                "path": str(artifact_dir / "mk0_freeze_manifest.json"),
                "sha256": freeze_sha,
            },
        }
        completion_sha256 = write_json_exclusive_atomic(completion_path, completion)
        terminal_status_sha256 = sha256_file(run_root / "status.json")
        mk0_status_path = run_root / "mk0_status.json"
        mk0_status_sha256 = write_json_exclusive_atomic(
            mk0_status_path,
            {
                "schema_version": "mk0_task_status_v1",
                "run_id": args.run_id,
                "task_id": "MK0-01",
                "status": "DONE",
                "all_m01_m35_passed": True,
                "evidence_level": EVIDENCE_LEVEL,
                "completion_manifest_sha256": completion_sha256,
                "whole_run_checksum_ledger_path": str(whole_run_ledger_path),
                "whole_run_ledger_and_DONE_self_reference_exception": True,
                "final_labels_accessed": False,
                "downstream_stage_started": False,
                "updated_at_utc": terminal_at,
            },
        )
        whole_run_ledger = write_whole_run_checksum_ledger(run_root)
        verified_whole_run_ledger = verify_checksum_ledger(
            run_root, whole_run_ledger_path
        )
        require(
            verified_whole_run_ledger["verified_entry_count"]
            == whole_run_ledger["entry_count"],
            "whole-run checksum ledger entry-count drift",
        )
        if args.parent_run_id is not None:
            require(
                canonical_json_bytes(
                    verify_parent_run_binding(
                        terminal_parent_binding,
                        parent_run_id=args.parent_run_id,
                        goal_sha256=args.goal_sha256,
                    )
                )
                == canonical_json_bytes(terminal_parent_binding),
                "MK0 parent binding drifted before DONE publication",
            )
        done_path = run_root / "DONE"
        write_bytes_exclusive_atomic(
            done_path,
            (
                f"{args.run_id}\n{freeze_sha}\n{completion_sha256}\n"
                f"{whole_run_ledger['sha256']}\n{terminal_status_sha256}\n"
                f"{mk0_status_sha256}\n"
            ).encode("ascii"),
        )
        require(
            validate_terminal_chain(run_root, run_id=args.run_id) == "DONE",
            "published DONE chain failed exact terminal validation",
        )
        console_summary = {
            **summary,
            "completion_manifest_sha256": completion_sha256,
            "whole_run_checksum_ledger_sha256": whole_run_ledger["sha256"],
            "whole_run_checksum_entry_count": whole_run_ledger["entry_count"],
            "mk0_status_sha256": mk0_status_sha256,
            "terminal_status_sha256": terminal_status_sha256,
        }
        print(json.dumps(console_summary, sort_keys=True))
        return 0
    except BaseException as error:
        failure_reason = _standard_failure_reason(error)
        if isinstance(bound_run_id, str) and (
            (run_root / "DONE").exists() or (run_root / "FAILED").exists()
        ):
            try:
                terminal = validate_terminal_chain(run_root, run_id=bound_run_id)
            except BaseException as terminal_error:
                print(
                    f"MK0 terminal chain validation failed: {terminal_error}",
                    file=sys.stderr,
                )
                return 1
            return 0 if terminal == "DONE" else 1
        failure_dir = run_root / "failure"
        try:
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "mk0_finalize_failure_v2",
                "run_id": args.run_id,
                "status": "FAILED_WITH_EVIDENCE",
                "created_at_utc": utc_now(),
                "exception_type": type(error).__name__,
                "exception_message": failure_reason,
                "traceback": traceback.format_exc(),
            }
            path = failure_dir / "finalize_failure.json"
            if not path.exists():
                write_new(path, failure)
            done_path = run_root / "DONE"
            if done_path.exists():
                revoked = failure_dir / "DONE_REVOKED_AFTER_FINALIZER_ERROR"
                if not revoked.exists():
                    os.replace(done_path, revoked)
            if (run_root / "logs" / "stderr.log").is_file():
                append_text(
                    run_root / "logs" / "stderr.log",
                    f"{utc_now()} MK0 finalization failed: {failure_reason}\n",
                )
            if (run_root / "logs" / "events.jsonl").is_file():
                write_failed_sentinel(
                    run_root,
                    run_id=args.run_id,
                    stage="FINALIZER",
                    reason=failure_reason,
                    exit_code=1,
                )
        except BaseException:
            pass
        print(f"MK0 finalization failed: {failure_reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
