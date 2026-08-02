#!/usr/bin/env python3
"""Formal CPU mathematical/oracle acceptance for MK0-v1.

Only symbolic, exhaustive, numerical-oracle and non-neural tests run here.
Neural forward/backward is deliberately delegated to the fail-closed CUDA
runner.  This script writes the small canonical bundle under the contract's
``/home/.../artifacts/runs`` root; large bound data and weights remain external.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_bootstrap_path = (REPO_ROOT / "scripts" / "mk0" / "strict_worktree_import.py").resolve(
    strict=True
)
_bootstrap_source = _bootstrap_path.read_bytes()
_bootstrap_namespace = {
    "__file__": str(_bootstrap_path),
    "__name__": "_mk0_strict_worktree_import",
}
exec(
    compile(
        _bootstrap_source,
        str(_bootstrap_path),
        "exec",
        dont_inherit=True,
        optimize=0,
    ),
    _bootstrap_namespace,
)
_strict_worktree_package_import = _bootstrap_namespace["strict_worktree_package_import"]

import numpy as np
import yaml

with _strict_worktree_package_import(REPO_ROOT):
    from mrna_editflow.core.mk0.acceptance import (
        canonical_json_bytes,
        gate_result_from_runtime_binding,
        sha256_file,
    )
    from mrna_editflow.core.mk0.alignment_coupling import (
        BLANK,
        alignment_actions,
        build_alignment,
        changed_indices,
        joint_path_probability,
        reconstruct_alignment,
        sample_optimal_alignment,
        sample_switch_clocks,
    )
    from mrna_editflow.core.mk0.bregman import (
        brute_force_bregman_loss,
        edit_flow_loss,
    )
    from mrna_editflow.core.mk0.critic_boundary import (
        base_generation_without_critic,
        reject_final_evaluator_as_guidance,
    )
    from mrna_editflow.core.mk0.foundation_fusion import (
        FoundationFusionRateField,
        OfficialPaperRateAdapter,
    )
    from mrna_editflow.core.mk0.rate_kernel import (
        FactorizedRates,
        aggregate_transition_rates,
        conditioned_event_distribution,
        enumerate_action_rates,
        generator,
        total_hazard,
    )
    from mrna_editflow.core.mk0.run_contract import (
        EVIDENCE_LEVEL,
        append_event,
        append_jsonl,
        append_text,
        create_contract_tree,
        resume_failure_closure_if_present,
        update_status,
        write_failed_sentinel,
        write_json_exclusive_atomic,
    )
    from mrna_editflow.core.mk0.samplers import (
        certify_remaining_integrated_hazard,
        constrained_single_event_first_order,
        paper_first_order_parallel,
        replay_constrained_result,
        replay_paper_result,
        sampler_result_to_schema_record,
    )
    from mrna_editflow.core.mk0.schedule import (
        cubic_schedule,
        evaluate_schedule,
        linear_schedule,
        rho,
    )
    from mrna_editflow.core.mk0.state_action import (
        action_to_schema_record,
        apply_action,
        enumerate_legal_actions,
        force_terminate,
        replay_actions,
        state_to_schema_record,
        termination_to_schema_record,
        validate_schema_facing_record,
    )
    from mrna_editflow.core.mk0.stop import (
        StopTarget,
        constant_hazard_stop_loss,
        sample_stop_target,
        stop_event_censor_oracle,
        survival_stop_loss,
    )
    from mrna_editflow.core.mk0.target_kernel import (
        TargetKernelRejected,
        build_target_transition_oracle,
    )
    from mrna_editflow.core.mk0.types import (
        ALPHABET,
        ActionType,
        AtomicAction,
        EditState,
        Phase,
        TerminationReason,
    )

SEED = 20260802
ATOL = 1.0e-10
RTOL = 1.0e-8
SCHEMA_FILES = (
    "edit_state_v1.schema.json",
    "edit_action_v1.schema.json",
    "edit_trajectory_v1.schema.json",
    "termination_event_v1.schema.json",
    "coupling_manifest_v1.schema.json",
)
FORMAL_RUN_ID = re.compile(
    r"^MK0_(?P<model>[A-Za-z0-9]+)_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<split>[A-Za-z0-9]+)_(?P<utc>[0-9]{8}T[0-9]{6}Z)_"
    r"(?P<short_sha>[0-9a-f]{7,12})_s(?P<seed>[0-9]+)$"
)
CANONICAL_RUN_PARENT = Path(
    "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/artifacts/runs"
)


class AcceptanceFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_new(path: Path, value: Any) -> str:
    data = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def sha_record(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def failure_reason(error: BaseException) -> str:
    """Return a non-empty, stable reason for terminal failure closure."""

    exception_type = type(error).__name__
    message = str(error).strip()
    return f"{exception_type}: {message}" if message else exception_type


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _read_ordinary_unlinked_bytes(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read one no-follow FD and prove its identity stayed stable."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcceptanceFailure(
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
        raise AcceptanceFailure(f"{label} disappeared after it was read") from error
    require(
        not path.is_symlink()
        and all(
            getattr(before, field) == getattr(live, field) for field in identity_fields
        ),
        f"{label} pathname identity changed while it was read",
    )
    return data, before


def _formal_run_time(run_id: str, *, label: str) -> datetime:
    match = FORMAL_RUN_ID.fullmatch(run_id)
    require(match is not None, f"{label} is not a formal MK0 run ID")
    try:
        observed = datetime.strptime(match.group("utc"), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise AcceptanceFailure(f"{label} UTC is not a calendar time") from error
    require(
        observed.strftime("%Y%m%dT%H%M%SZ") == match.group("utc"),
        f"{label} UTC is not canonical",
    )
    return observed


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
        failure_root_stat = failure_root.lstat()
        require(
            stat.S_ISDIR(failure_root_stat.st_mode) and not failure_root.is_symlink(),
            "parent failure evidence root is not an ordinary directory",
        )
        for candidate in failure_root.rglob("*"):
            candidate_stat = candidate.lstat()
            require(
                not candidate.is_symlink(),
                "parent failure evidence contains a symlink",
            )
            if stat.S_ISDIR(candidate_stat.st_mode):
                continue
            require(
                stat.S_ISREG(candidate_stat.st_mode),
                "parent failure evidence contains a special file",
            )
            evidence_paths.add(candidate)

    artifacts_root = parent_root / "artifacts"
    if artifacts_root.exists() or artifacts_root.is_symlink():
        artifacts_root_stat = artifacts_root.lstat()
        require(
            stat.S_ISDIR(artifacts_root_stat.st_mode)
            and not artifacts_root.is_symlink(),
            "parent artifacts root is not an ordinary directory",
        )
        for candidate in artifacts_root.rglob("*failure*.json"):
            candidate_stat = candidate.lstat()
            require(
                not candidate.is_symlink(),
                "parent failure evidence contains a symlink",
            )
            require(
                stat.S_ISREG(candidate_stat.st_mode),
                "parent failure evidence contains a special file",
            )
            evidence_paths.add(candidate)
    classification = "FAILED" if "FAILED" in sentinels else "UNSEALED_FAILED_EVIDENCE"
    return classification, sorted(
        evidence_paths,
        key=lambda path: path.relative_to(parent_root).as_posix(),
    )


def validate_parent_run_lineage(
    child_run_id: str,
    parent_run_id: str | None,
    *,
    goal_sha256: str | None = None,
    canonical_parent: Path = CANONICAL_RUN_PARENT,
) -> dict[str, Any] | None:
    """Bind a repair child to a valid failed parent and its failure evidence."""

    if parent_run_id is None:
        return None
    parent_time = _formal_run_time(parent_run_id, label="parent run ID")
    child_time = _formal_run_time(child_run_id, label="child run ID")
    require(parent_time < child_time, "parent run ID UTC must precede child run ID UTC")
    require(
        isinstance(goal_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", goal_sha256) is not None,
        "current Goal hash is invalid",
    )
    canonical_root = canonical_parent.resolve(strict=True)
    parent_root = canonical_root / parent_run_id
    try:
        parent_root_stat = parent_root.lstat()
    except FileNotFoundError as error:
        raise AcceptanceFailure("parent run root is absent") from error
    require(
        stat.S_ISDIR(parent_root_stat.st_mode) and not parent_root.is_symlink(),
        "parent run root is not an ordinary directory",
    )
    require(
        parent_root.resolve(strict=True) == parent_root,
        "parent run root is not canonical",
    )
    manifest_path = parent_root / "run_manifest.json"
    manifest_bytes, manifest_stat = _read_ordinary_unlinked_bytes(
        manifest_path,
        label="parent run registration manifest",
    )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceFailure(
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
    implementation_commit = manifest.get("implementation_commit")
    require(
        isinstance(implementation_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is not None,
        "parent run implementation commit is invalid",
    )
    code = manifest.get("code")
    source_binding = manifest.get("source_binding")
    require(
        isinstance(code, Mapping) and code.get("commit") == implementation_commit,
        "parent run code binding drift",
    )
    require(
        isinstance(source_binding, Mapping)
        and source_binding.get("git_commit") == implementation_commit,
        "parent run source binding drift",
    )
    parent_match = FORMAL_RUN_ID.fullmatch(parent_run_id)
    require(parent_match is not None, "parent run ID is not formal")
    require(
        implementation_commit.startswith(parent_match.group("short_sha")),
        "parent run ID short SHA differs from implementation commit",
    )

    classification, evidence_paths = _parent_failure_path_snapshot(parent_root)

    evidence_records: list[dict[str, Any]] = []
    for evidence_path in evidence_paths:
        evidence_bytes, evidence_stat = _read_ordinary_unlinked_bytes(
            evidence_path,
            label="parent failure evidence file",
        )
        evidence_records.append(
            {
                "path": evidence_path.relative_to(parent_root).as_posix(),
                "size_bytes": evidence_stat.st_size,
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            }
        )
    require(evidence_records, "repair parent has no failure evidence")
    classification_after, evidence_paths_after = _parent_failure_path_snapshot(
        parent_root
    )
    require(
        classification_after == classification
        and [path.relative_to(parent_root) for path in evidence_paths_after]
        == [path.relative_to(parent_root) for path in evidence_paths],
        "parent failure evidence inventory changed while it was bound",
    )
    parent_root_after = parent_root.lstat()
    require(
        not parent_root.is_symlink()
        and all(
            getattr(parent_root_stat, field) == getattr(parent_root_after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ),
        "parent run root identity changed while lineage was bound",
    )
    return {
        "schema_version": "mk0_parent_run_binding_v1",
        "run_id": parent_run_id,
        "run_root": str(parent_root),
        "registration_manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_stat.st_size,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "observed_classification": classification,
        "failure_evidence": {
            "file_count": len(evidence_records),
            "total_size_bytes": sum(
                record["size_bytes"] for record in evidence_records
            ),
            "files": evidence_records,
            "files_sha256": sha_record(evidence_records),
        },
    }


def git_source_binding(expected_commit: str) -> dict[str, Any]:
    """Bind the clean tracked MK0 source bytes used by this runner."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == expected_commit, "implementation commit mismatch")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(
        not status.strip(), "formal CPU run requires a clean implementation worktree"
    )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "core/mk0",
            "scripts/mk0",
            "configs/math",
            "schemas",
            "docs/math",
            "tests/mk0",
            "pyproject.toml",
            "__init__.py",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths = [item for item in listed.decode("utf-8").split("\0") if item]
    require(paths, "no tracked MK0 source files were bound")
    files: dict[str, str] = {}
    for relative in paths:
        path = REPO_ROOT / relative
        require(path.is_file(), f"tracked MK0 source is missing: {relative}")
        files[relative] = sha256_file(path)
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": head,
        "git_tree": tree,
        "git_status_porcelain": status,
        "tracked_source_file_count": len(files),
        "tracked_source_files_sha256": hashlib.sha256(
            canonical_json_bytes(files)
        ).hexdigest(),
        "tracked_source_files": files,
    }


def validate_preflight_binding(
    path: Path,
    *,
    run_id: str,
    parent_run_id: str | None,
    goal_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    report = read_json_object(resolved)
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
        report.get("mode") == "read_only_metadata_and_hashes",
        "preflight was not read-only",
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
        ancestor.returncode == 0,
        "preflight commit is not an ancestor of implementation",
    )
    safety = report.get("safety", {})
    require(safety.get("unrelated_processes_killed") == 0, "preflight killed a process")
    require(
        safety.get("existing_results_overwritten") == 0, "preflight overwrote a result"
    )
    require(safety.get("final_labels_read") is False, "preflight accessed final labels")
    require(
        safety.get("neural_forward_executed") is False, "preflight executed neural work"
    )
    require(
        safety.get("downstream_stage_started") is False, "preflight started downstream"
    )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "observed_at_utc": report["observed_at_utc"],
        "preflight_worktree_head": preflight_head,
        "parent_run_id": parent_run_id,
        "safety": safety,
    }


def tiny_sequences(alphabet: str = "AC") -> tuple[str, ...]:
    return tuple(
        "".join(tokens)
        for length in (1, 2, 3)
        for tokens in itertools.product(alphabet, repeat=length)
    )


def tiny_active_rate_states() -> tuple[EditState, ...]:
    """Construct the preregistered 1,176 valid tiny extended ACTIVE states.

    The 196 source/current pairs are reached by replaying an optimal edit path,
    never by fabricating a current string or mapping.  Each pair is crossed
    with three remaining-budget values and both declared UTR regions.
    """

    states: list[EditState] = []
    for source, current in itertools.product(tiny_sequences(), repeat=2):
        alignment = build_alignment(source, current)
        actions = alignment_actions(alignment)
        for remaining_budget in (0, 1, 2):
            for region in ("5UTR", "3UTR"):
                initial = EditState.initial(
                    source,
                    budget=alignment.cost + remaining_budget,
                    region=region,
                )
                state, _ = replay_actions(initial, actions, min_length=1, max_length=4)
                require(state.current == current, "tiny rate-state replay drift")
                require(
                    state.remaining_budget == remaining_budget,
                    "tiny rate-state remaining-budget drift",
                )
                states.append(state)
    require(len(states) == 1176, "tiny active rate-state count drift")
    return tuple(states)


def tiny_active_halted_rate_states() -> (
    tuple[tuple[EditState, ...], tuple[EditState, ...]]
):
    """Resolve the config's 1,176-state ACTIVE/HALTED extended domain.

    There are 196 valid replayed source/current pairs crossed with three
    remaining-budget values: 588 ACTIVE states and their 588 HALTED twins.
    Region is balanced across the domain but is not used to inflate its count.
    """

    region_paired = tiny_active_rate_states()
    active = tuple(
        region_paired[index + ((index // 2) % 2)]
        for index in range(0, len(region_paired), 2)
    )
    halted = tuple(
        force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
        for state in active
    )
    require(len(active) == len(halted) == 588, "tiny run-state stratum drift")
    require(
        sum(state.region == "5UTR" for state in active) == 294
        and sum(state.region == "3UTR" for state in active) == 294,
        "tiny run-state region balance drift",
    )
    return active, halted


def coupling_record(alignment, source_id: str, target_id: str) -> dict[str, Any]:
    return {
        "schema_version": "coupling_manifest_v1",
        "source_id": source_id,
        "target_id": target_id,
        "coupling_type": alignment.coupling_type,
        "alignment_algorithm_version": alignment.algorithm_version,
        "alignment_cost": alignment.cost,
        "tie_break_rule": alignment.tie_break_rule,
        "alignment_hash": alignment.alignment_hash,
        "path_is_observed": False,
        "path_semantics": "latent_algorithmic",
        "z_src": [
            "EPSILON" if column.source_token == BLANK else column.source_token
            for column in alignment.columns
        ],
        "z_tar": [
            "EPSILON" if column.target_token == BLANK else column.target_token
            for column in alignment.columns
        ],
        "source_reconstruction": alignment.source,
        "target_reconstruction": alignment.target,
        "joint_path": "independent_switch_clock_product",
        "schedule": {
            "name": "cubic",
            "kappa": "t**3",
            "kappa_derivative": "3*t**2",
            "rho": "kappa_derivative/(1-kappa)",
            "time_eps": 1.0e-4,
        },
        "target_auxiliary_isolation": {
            "z_aux_training_only": True,
            "rate_network_receives_z_aux": False,
            "leakage_test_status": "PASS",
        },
        "rejected_path_ledger": [],
    }


def run_schema_action_audit(rng: random.Random) -> tuple[dict[str, Any], int]:
    schemas: dict[str, Any] = {}
    for name in SCHEMA_FILES:
        path = REPO_ROOT / "schemas" / name
        schema = json.loads(path.read_text(encoding="utf-8"))
        require(schema["$schema"].endswith("2020-12/schema"), f"draft drift: {name}")
        require(
            schema.get("additionalProperties") is False, f"schema not strict: {name}"
        )
        schemas[name] = {"sha256": sha256_file(path), "title": schema.get("title")}

    state_count = 0
    halted_count = 0
    action_count = 0
    exact_failures = 0
    record_digest = hashlib.sha256()
    for sequence in tiny_sequences():
        for budget in (0, 1, 2):
            for mask_bits in itertools.product((False, True), repeat=len(sequence)):
                protected = tuple(i for i, value in enumerate(mask_bits) if value)
                for region in ("5UTR", "3UTR"):
                    state = EditState.initial(
                        sequence,
                        budget=budget,
                        protected_indices=protected,
                        region=region,
                        context={
                            "assay": "mk0_tiny",
                            "cell_or_tissue": "synthetic",
                            "endpoint": "schema",
                        },
                    )
                    state_record = state_to_schema_record(
                        state, source_id=f"tiny-{state_count}", external_time=0.0
                    )
                    validate_schema_facing_record(state_record, "state")
                    record_digest.update(canonical_json_bytes(state_record))
                    state_count += 1

                    halted = force_terminate(
                        state, TerminationReason.FORCED_TIME_HORIZON
                    )
                    termination = termination_to_schema_record(
                        reason=TerminationReason.FORCED_TIME_HORIZON,
                        external_time=1.0,
                        state_hash_before=state.state_hash,
                        state_hash_after=halted.state_hash,
                    )
                    validate_schema_facing_record(termination, "termination")
                    halted_record = state_to_schema_record(
                        halted,
                        source_id=f"tiny-{state_count}",
                        external_time=1.0,
                        parent_state_hash=state.state_hash,
                        termination=termination,
                    )
                    validate_schema_facing_record(halted_record, "state")
                    record_digest.update(canonical_json_bytes(halted_record))
                    halted_count += 1

                    for action in enumerate_legal_actions(
                        state, min_length=1, max_length=4
                    ):
                        transition = apply_action(
                            state, action, min_length=1, max_length=4
                        )
                        record = action_to_schema_record(
                            transition,
                            action_id=f"a-{action_count}",
                            external_time=0.5,
                        )
                        validate_schema_facing_record(record, "action")
                        before = state.current
                        if action.kind == ActionType.INS:
                            expected = (
                                before[: int(action.position)]
                                + str(action.token)
                                + before[int(action.position) :]
                            )
                        elif action.kind == ActionType.SUB:
                            expected = (
                                before[: int(action.position)]
                                + str(action.token)
                                + before[int(action.position) + 1 :]
                            )
                        elif action.kind == ActionType.DEL:
                            expected = (
                                before[: int(action.position)]
                                + before[int(action.position) + 1 :]
                            )
                        else:
                            expected = before
                        exact_failures += int(transition.after.current != expected)
                        record_digest.update(canonical_json_bytes(record))
                        action_count += 1
    require(state_count == 504, f"tiny active state count drift: {state_count}")
    require(halted_count == 504, f"tiny halted state count drift: {halted_count}")
    require(action_count == 6736, f"tiny action count drift: {action_count}")
    require(exact_failures == 0, "tiny action exactness failure")

    real_failures = 0
    lengths = (16, 32, 64, 128, 256)
    for index in range(1024):
        length = lengths[index % len(lengths)]
        sequence = "".join(rng.choice(ALPHABET) for _ in range(length))
        state = EditState.initial(sequence, budget=1)
        legal = enumerate_legal_actions(
            state, min_length=1, max_length=300, include_stop=False
        )
        action = rng.choice(legal)
        transition = apply_action(state, action, min_length=1, max_length=300)
        replayed, _ = replay_actions(state, (action,), min_length=1, max_length=300)
        real_failures += int(replayed != transition.after)
    require(real_failures == 0, "real-length action property failure")

    example_alignment = build_alignment("AA", "A")
    coupling = coupling_record(example_alignment, "schema-source", "schema-target")
    validate_schema_facing_record(coupling, "coupling")
    record_digest.update(canonical_json_bytes(coupling))
    initial = EditState.initial(
        "AC",
        budget=1,
        context={"assay": "schema", "cell_or_tissue": "synthetic", "endpoint": "stop"},
    )

    def schema_stop_rates(
        _state: EditState, _time: float
    ) -> Mapping[AtomicAction, float]:
        return {AtomicAction(ActionType.STOP): 64.0}

    schema_result = constrained_single_event_first_order(
        initial,
        schema_stop_rates,
        step_size=0.25,
        stability_hazard=0.05,
        min_length=1,
        max_length=4,
        seed=SEED,
    )
    trajectory = sampler_result_to_schema_record(
        schema_result,
        schema_stop_rates,
        trajectory_id="schema-trajectory",
        source_id="schema-source",
    )
    require(trajectory["replay"]["status"] == "PASS", "real trajectory replay failed")
    validate_schema_facing_record(trajectory, "trajectory")
    record_digest.update(canonical_json_bytes(trajectory))
    sample_count = 5 + state_count + halted_count + action_count + halted_count + 2
    require(sample_count == 8255, f"M01 sample count drift: {sample_count}")
    return {
        "schema_version": "mk0_schema_action_audit_v1",
        "status": "PASS",
        "seed": SEED,
        "validator": "jsonschema_Draft202012_plus_cross_field_invariants_and_adversarial_pytest",
        "schema_documents": schemas,
        "active_state_record_count": state_count,
        "halted_state_record_count": halted_count,
        "termination_record_count": halted_count,
        "action_record_count": action_count,
        "representative_coupling_record_count": 1,
        "representative_trajectory_record_count": 1,
        "representative_trajectory_time_direction": trajectory["time_direction"],
        "representative_trajectory_sha256": sha_record(trajectory),
        "runtime_validation_count": sample_count,
        "schema_validation_failure_count": 0,
        "runtime_record_stream_sha256": record_digest.hexdigest(),
        "tiny_exactness_failure_count": exact_failures,
        "real_length_property_count": 1024,
        "real_length_property_failure_count": real_failures,
        "M01_pass": True,
        "M02_pass": True,
    }, action_count + 1024


def run_coupling_audit(rng: random.Random) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures = 0
    sequences = tiny_sequences()
    for index, (source, target) in enumerate(itertools.product(sequences, repeat=2)):
        alignment = build_alignment(source, target)
        record = coupling_record(alignment, f"tiny-s-{index}", f"tiny-t-{index}")
        validate_schema_facing_record(record, "coupling")
        reconstructed = reconstruct_alignment(alignment)
        state = EditState.initial(source, budget=alignment.cost)
        replayed, _ = replay_actions(
            state, alignment_actions(alignment), min_length=0, max_length=6
        )
        failures += int(reconstructed != (source, target) or replayed.current != target)
        records.append(record)
    lengths = (16, 32, 64, 128, 256)
    for index in range(1024):
        length = lengths[index % len(lengths)]
        source = "".join(rng.choice(ALPHABET) for _ in range(length))
        target_chars = list(source)
        for _ in range(1 + index % 3):
            pos = rng.randrange(length)
            target_chars[pos] = rng.choice(
                tuple(token for token in ALPHABET if token != target_chars[pos])
            )
        target = "".join(target_chars)
        alignment = build_alignment(source, target)
        record = coupling_record(alignment, f"real-s-{index}", f"real-t-{index}")
        validate_schema_facing_record(record, "coupling")
        failures += int(reconstruct_alignment(alignment) != (source, target))
        records.append(record)
    require(len(records) == 1220 and failures == 0, "alignment reconstruction failure")

    sensitivity: list[dict[str, Any]] = []
    for source, target in (
        ("AA", "A"),
        ("A", "AA"),
        ("AAA", "AA"),
        ("AA", "AAA"),
        ("ACA", "AAC"),
    ):
        hashes = {
            sample_optimal_alignment(source, target, rng=rng).alignment_hash
            for _ in range(512)
        }
        sensitivity.append(
            {
                "source": source,
                "target": target,
                "unique_optimal_alignment_hashes": len(hashes),
            }
        )
    # M06 is an exact product-law gate, not a Monte Carlo non-rejection test.
    # Enumerate all 2^3 switch vectors in 128 deterministic tiny-alignment,
    # schedule and time fixtures: 128 * 8 = 1,024 path-probability checks.
    three_switch_alignments = [
        build_alignment(source, target)
        for source, target in itertools.product(sequences, repeat=2)
        if len(changed_indices(build_alignment(source, target))) == 3
    ]
    require(three_switch_alignments, "M06 lacks three-switch tiny alignments")
    path_probability_failures = 0
    clock_sampler_binding_failures = 0
    group_consistency_failures = 0
    maximum_path_probability_error = 0.0
    maximum_normalization_error = 0.0
    marginal_error = 0.0
    covariance_error = 0.0
    kappa_values: list[float] = []
    switch_vectors_checked = 0
    for fixture_index in range(128):
        alignment = three_switch_alignments[
            fixture_index % len(three_switch_alignments)
        ]
        changed = changed_indices(alignment)
        schedule_name = "cubic" if fixture_index % 2 == 0 else "linear"
        time = (1 + fixture_index % 15) / 16.0
        kappa = (
            cubic_schedule(time)[0]
            if schedule_name == "cubic"
            else linear_schedule(time)[0]
        )
        kappa_values.append(kappa)
        enumerated: list[tuple[tuple[int, ...], float]] = []
        for vector_index, switch_bits in enumerate(
            itertools.product((0, 1), repeat=len(changed))
        ):
            switched_by_column = dict(zip(changed, switch_bits))
            z = tuple(
                (
                    column.target_token
                    if switched_by_column.get(column_index, 0)
                    else column.source_token
                )
                for column_index, column in enumerate(alignment.columns)
            )
            observed_probability = joint_path_probability(
                alignment, z, time, schedule=schedule_name
            )
            switched_count = sum(switch_bits)
            expected_probability = kappa**switched_count * (1.0 - kappa) ** (
                len(changed) - switched_count
            )
            error = abs(observed_probability - expected_probability)
            maximum_path_probability_error = max(maximum_path_probability_error, error)
            path_probability_failures += int(
                not math.isclose(
                    observed_probability,
                    expected_probability,
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
            )
            # Bind the actual clock sampler to fresh, independent RNG draws and
            # the registered inverse CDF; a correct product evaluator cannot
            # mask a correlated or stale-draw clock implementation.
            clock_seed = SEED + fixture_index * 8 + vector_index
            oracle_rng = random.Random(clock_seed)
            uniforms = [oracle_rng.random() for _ in changed]
            expected_clocks = {
                column_index: (
                    uniform ** (1.0 / 3.0) if schedule_name == "cubic" else uniform
                )
                for column_index, uniform in zip(changed, uniforms)
            }
            observed_clocks = sample_switch_clocks(
                alignment,
                rng=random.Random(clock_seed),
                schedule=schedule_name,
            )
            clock_sampler_binding_failures += int(
                set(observed_clocks) != set(expected_clocks)
                or any(
                    not math.isclose(
                        observed_clocks[column_index],
                        expected_clocks[column_index],
                        abs_tol=1.0e-15,
                        rel_tol=1.0e-15,
                    )
                    for column_index in expected_clocks
                )
            )
            enumerated.append((switch_bits, observed_probability))
            switch_vectors_checked += 1

        normalization_error = abs(
            math.fsum(probability for _bits, probability in enumerated) - 1.0
        )
        maximum_normalization_error = max(
            maximum_normalization_error, normalization_error
        )
        fixture_failed = normalization_error > ATOL
        for coordinate in range(len(changed)):
            marginal = math.fsum(
                probability for bits, probability in enumerated if bits[coordinate]
            )
            marginal_error = max(marginal_error, abs(marginal - kappa))
            fixture_failed = fixture_failed or not math.isclose(
                marginal, kappa, abs_tol=ATOL, rel_tol=RTOL
            )
        for left in range(len(changed)):
            for right in range(left + 1, len(changed)):
                joint = math.fsum(
                    probability
                    for bits, probability in enumerated
                    if bits[left] and bits[right]
                )
                covariance = joint - kappa * kappa
                covariance_error = max(covariance_error, abs(covariance))
                fixture_failed = fixture_failed or abs(covariance) > ATOL
        group_consistency_failures += int(fixture_failed)
    require(switch_vectors_checked == 1024, "M06 sample-count drift")
    require(
        path_probability_failures
        == clock_sampler_binding_failures
        == group_consistency_failures
        == 0,
        "joint switch-clock product law failed",
    )
    return {
        "schema_version": "mk0_coupling_audit_v1",
        "status": "PASS",
        "seed": SEED,
        "record_count": len(records),
        "failure_count": failures,
        "path_is_observed_true_count": sum(
            record["path_is_observed"] for record in records
        ),
        "records": records,
        "sampled_optimal_sensitivity": sensitivity,
        "joint_product_switch_clock_oracle": {
            "sample_count": switch_vectors_checked,
            "fixture_count": 128,
            "changed_coordinate_count": 3,
            "expected_switch_probability": "schedule_specific_kappa(t)",
            "minimum_kappa": min(kappa_values),
            "maximum_kappa": max(kappa_values),
            "maximum_path_probability_error": maximum_path_probability_error,
            "maximum_normalization_error": maximum_normalization_error,
            "max_marginal_error": marginal_error,
            "max_off_diagonal_covariance": covariance_error,
            "path_probability_failure_count": path_probability_failures,
            "clock_sampler_binding_failure_count": clock_sampler_binding_failures,
            "clock_sampler_binding": "fresh_seeded_draws_plus_registered_inverse_cdf",
            "group_consistency_failure_count": group_consistency_failures,
            "failure_count": path_probability_failures
            + clock_sampler_binding_failures
            + group_consistency_failures,
            "pass": True,
        },
    }


def _m09_schedule_endpoint_derivative_audit() -> dict[str, Any]:
    """Run exactly the 2*1001 FD grid plus four endpoint fixtures in M09."""

    finite_difference_step = 5.0e-6
    origin_forward_step = 1.0e-4
    derivative_failures = 0
    maximum_derivative_error = 0.0
    grid_stream = hashlib.sha256()
    for name, fn in (("cubic", cubic_schedule), ("linear", linear_schedule)):
        for index in range(1001):
            t = index * (1.0 - 1.0e-4) / 1000
            _kappa, analytic = fn(t)
            if index == 0:
                h = origin_forward_step
                numerical = (
                    -25.0 * fn(t)[0]
                    + 48.0 * fn(t + h)[0]
                    - 36.0 * fn(t + 2.0 * h)[0]
                    + 16.0 * fn(t + 3.0 * h)[0]
                    - 3.0 * fn(t + 4.0 * h)[0]
                ) / (12.0 * h)
                stencil = "five_point_forward"
            else:
                h = finite_difference_step
                numerical = (fn(t + h)[0] - fn(t - h)[0]) / (2.0 * h)
                stencil = "central"
            error = abs(numerical - analytic)
            maximum_derivative_error = max(maximum_derivative_error, error)
            derivative_failures += int(
                not math.isclose(
                    numerical,
                    analytic,
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
            )
            grid_stream.update(
                canonical_json_bytes(
                    {
                        "schedule": name,
                        "grid_index": index,
                        "t": t,
                        "stencil": stencil,
                        "h": h,
                        "analytic_derivative": analytic,
                        "finite_difference_derivative": numerical,
                    }
                )
            )

    endpoint_specs = (
        ("cubic", cubic_schedule, 0.0, 0.0, 0.0, False),
        ("cubic", cubic_schedule, 1.0, 1.0, 3.0, True),
        ("linear", linear_schedule, 0.0, 0.0, 1.0, False),
        ("linear", linear_schedule, 1.0, 1.0, 1.0, True),
    )
    endpoint_failures = 0
    endpoint_nonfinite_count = 0
    endpoint_clip_hit_count = 0
    endpoint_records: list[dict[str, Any]] = []
    for (
        name,
        fn,
        t,
        expected_kappa,
        expected_derivative,
        expected_clip,
    ) in endpoint_specs:
        kappa, derivative = fn(t)
        evaluated = evaluate_schedule(t, name=name)
        nonfinite = not math.isfinite(evaluated.rho)
        endpoint_nonfinite_count += int(nonfinite)
        endpoint_clip_hit_count += int(evaluated.endpoint_clipped)
        failed = (
            not math.isclose(kappa, expected_kappa, abs_tol=ATOL, rel_tol=RTOL)
            or not math.isclose(
                derivative,
                expected_derivative,
                abs_tol=ATOL,
                rel_tol=RTOL,
            )
            or nonfinite
            or evaluated.endpoint_clipped != expected_clip
        )
        endpoint_failures += int(failed)
        endpoint_records.append(
            {
                "schedule": name,
                "t": t,
                "kappa": kappa,
                "derivative": derivative,
                "rho_finite": not nonfinite,
                "endpoint_clipped": evaluated.endpoint_clipped,
                "failed": failed,
            }
        )

    require(derivative_failures == 0, "M09 finite-difference derivative oracle failed")
    require(endpoint_failures == 0, "M09 schedule endpoint oracle failed")
    return {
        "schedule_derivative_grid_checks": 2002,
        "schedule_endpoint_checks": 4,
        "schedule_derivative_failure_count": derivative_failures,
        "schedule_endpoint_failure_count": endpoint_failures,
        "schedule_endpoint_nonfinite_count": endpoint_nonfinite_count,
        "endpoint_clip_hit_count": endpoint_clip_hit_count,
        "maximum_derivative_absolute_error": maximum_derivative_error,
        "finite_difference_stencil": "central_except_five_point_forward_at_t0",
        "finite_difference_step": finite_difference_step,
        "origin_forward_step": origin_forward_step,
        "actual_atol": ATOL,
        "actual_rtol": RTOL,
        "grid_stream_sha256": grid_stream.hexdigest(),
        "endpoint_records": endpoint_records,
    }


def _m14_conditioned_distribution_audit(
    states: list[EditState],
) -> dict[str, Any]:
    """Check every returned action probability against rate/fsum(rate)."""

    require(len(states) == 1176, "M14 positive-hazard state domain drift")
    state_failures = 0
    action_probability_checks = 0
    maximum_action_probability_error = 0.0
    fixture_stream = hashlib.sha256()
    for state in states:
        actions = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        denominator = math.fsum(actions.values())
        require(denominator > 0.0, "M14 fixture has non-positive total hazard")
        observed = conditioned_event_distribution(actions)
        expected = {action: rate / denominator for action, rate in actions.items()}
        key_failure = set(observed) != set(expected)
        normalization_failure = not math.isclose(
            math.fsum(observed.values()), 1.0, abs_tol=1.0e-12, rel_tol=0.0
        )
        probability_failure = False
        for action in sorted(expected, key=lambda item: item.key):
            action_probability_checks += 1
            value = observed.get(action, math.nan)
            error = abs(value - expected[action]) if math.isfinite(value) else math.inf
            maximum_action_probability_error = max(
                maximum_action_probability_error, error
            )
            probability_failure |= not math.isclose(
                value,
                expected[action],
                abs_tol=ATOL,
                rel_tol=RTOL,
            )
            fixture_stream.update(
                canonical_json_bytes(
                    {
                        "state_hash": state.state_hash,
                        "action_key": action.key,
                        "rate": actions[action],
                        "total_hazard": denominator,
                        "observed_probability": value,
                        "expected_probability": expected[action],
                    }
                )
            )
        state_failures += int(
            key_failure or normalization_failure or probability_failure
        )
    require(state_failures == 0, "M14 conditioned action law differs from rate/total")
    return {
        "conditioned_distribution_checks": len(states),
        "conditioned_distribution_failure_count": state_failures,
        "conditioned_action_probability_checks": action_probability_checks,
        "conditioned_action_probability_maximum_absolute_error": (
            maximum_action_probability_error
        ),
        "conditioned_action_probability_actual_atol": ATOL,
        "conditioned_action_probability_actual_rtol": RTOL,
        "conditioned_action_probability_fixture_stream_sha256": (
            fixture_stream.hexdigest()
        ),
    }


def _operational_state_payload(state: EditState) -> dict[str, Any]:
    payload = state.inference_dict()
    payload.pop("context")
    return payload


def _operational_fixture_id(
    state: EditState,
    *,
    case: str,
    attempted_action_keys: tuple[str, ...],
    min_length: int,
    max_length: int,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    return sha_record(
        {
            "case": case,
            "operational_state": _operational_state_payload(state),
            "attempted_action_keys": sorted(attempted_action_keys),
            "min_length": min_length,
            "max_length": max_length,
            "parameters": dict(parameters or {}),
        }
    )


def _invalid_action_payload_id(payload: Mapping[str, Any]) -> str:
    """Hash only the kwargs actually passed to ``AtomicAction``.

    In particular, no state or decorative context can make a repeated
    constructor attempt appear to be a new member of the M11 denominator.
    """

    return sha_record(
        {
            "constructor": "AtomicAction",
            "attempted_payload": dict(payload),
        }
    )


def _m11_invalid_action_payloads() -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return 588 distinct kwargs payloads that must all be rejected.

    The 14 blocks below exercise invalid enum kinds, null/wrong/negative
    positions, alphabet violations, field conflicts, missing required fields,
    and unexpected fields.  Every payload differs in kwargs that are actually
    passed to the constructor; category labels are not used for uniqueness.
    """

    attempts: list[tuple[str, dict[str, Any]]] = []
    for slot in range(42):
        attempts.extend(
            (
                (
                    "invalid_kind_string",
                    {
                        "kind": f"INVALID_KIND_{slot:02d}",
                        "position": slot,
                        "token": None,
                    },
                ),
                (
                    "invalid_kind_null",
                    {"kind": None, "position": slot, "token": None},
                ),
                (
                    "invalid_INS_alphabet_token",
                    {
                        "kind": ActionType.INS,
                        "position": slot,
                        "token": f"T_INS_{slot:02d}",
                    },
                ),
                (
                    "invalid_SUB_alphabet_token",
                    {
                        "kind": ActionType.SUB,
                        "position": slot,
                        "token": f"N_SUB_{slot:02d}",
                    },
                ),
                (
                    "negative_position",
                    {
                        "kind": ActionType.INS,
                        "position": -(slot + 1),
                        "token": ALPHABET[slot % len(ALPHABET)],
                    },
                ),
                (
                    "string_position",
                    {
                        "kind": ActionType.SUB,
                        "position": f"position-{slot:02d}",
                        "token": ALPHABET[(slot + 1) % len(ALPHABET)],
                    },
                ),
                (
                    "float_position",
                    {
                        "kind": ActionType.DEL,
                        "position": slot + 0.5,
                        "token": None,
                    },
                ),
                (
                    "list_position",
                    {
                        "kind": ActionType.DEL,
                        "position": [slot],
                        "token": None,
                    },
                ),
                (
                    "mapping_position",
                    {
                        "kind": ActionType.DEL,
                        "position": {"index": slot},
                        "token": None,
                    },
                ),
                (
                    "null_position",
                    {
                        "kind": ActionType.DEL,
                        "position": None,
                        "token": f"NULL_POSITION_{slot:02d}",
                    },
                ),
                (
                    "DEL_token_conflict",
                    {
                        "kind": ActionType.DEL,
                        "position": slot,
                        "token": f"DEL_TOKEN_{slot:02d}",
                    },
                ),
                (
                    "STOP_position_conflict",
                    {
                        "kind": ActionType.STOP,
                        "position": slot,
                        "token": None,
                    },
                ),
                (
                    "STOP_token_conflict",
                    {
                        "kind": ActionType.STOP,
                        "position": None,
                        "token": f"STOP_TOKEN_{slot:02d}",
                    },
                ),
                (
                    "missing_kind" if slot % 2 == 0 else "unexpected_field",
                    (
                        {"position": slot, "token": None}
                        if slot % 2 == 0
                        else {
                            "kind": ActionType.STOP,
                            "position": None,
                            "token": None,
                            f"unexpected_{slot:02d}": slot,
                        }
                    ),
                ),
            )
        )
    require(len(attempts) == 588, "M11 invalid-action payload count drift")
    return tuple(attempts)


def _replayed_operational_state(
    source: str,
    current: str,
    *,
    remaining_budget: int,
    region: str,
    target_condition: str,
) -> EditState:
    alignment = build_alignment(source, current)
    initial = EditState.initial(
        source,
        budget=alignment.cost + remaining_budget,
        region=region,
        target_condition=target_condition,
    )
    state, _ = replay_actions(
        initial,
        alignment_actions(alignment),
        min_length=1,
        max_length=6,
    )
    require(state.current == current, "operational fixture replay drift")
    require(
        state.remaining_budget == remaining_budget,
        "operational fixture budget drift",
    )
    return state


def _m11_mask_audit(base_states: tuple[EditState, ...]) -> dict[str, Any]:
    """Build 588 context-independent operational checks for each mask class."""

    target_count = 588
    case_ids: dict[str, set[str]] = defaultdict(set)
    case_counts: dict[str, int] = defaultdict(int)
    case_failures: dict[str, int] = defaultdict(int)
    histograms: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    fixture_stream = hashlib.sha256()
    invalid_payload_stream = hashlib.sha256()
    invalid_payload_records: list[dict[str, Any]] = []
    invalid_payload_category_counts: dict[str, int] = defaultdict(int)
    invalid_payload_exception_counts: dict[str, int] = defaultdict(int)

    def add_fixture(
        case: str,
        state: EditState,
        attempted_actions: tuple[AtomicAction, ...],
        *,
        min_length: int,
        max_length: int,
        failed: bool,
        attempted_action_keys: tuple[str, ...] | None = None,
    ) -> None:
        if case_counts[case] >= target_count:
            return
        keys = attempted_action_keys or tuple(
            action.key for action in attempted_actions
        )
        operational_id = _operational_fixture_id(
            state,
            case=case,
            attempted_action_keys=keys,
            min_length=min_length,
            max_length=max_length,
        )
        if operational_id in case_ids[case]:
            return
        case_ids[case].add(operational_id)
        case_counts[case] += 1
        case_failures[case] += int(failed)
        values = {
            "source_length": len(state.source),
            "current_length": len(state.current),
            "history_executed": state.history.executed,
            "remaining_budget": state.remaining_budget,
            "region": state.region,
            "protected_token_count": sum(
                token.protected for token in state.mapping.tokens
            ),
            "phase": state.phase.value,
        }
        for name, value in values.items():
            histograms[case][name][str(value)] += 1
        fixture_stream.update(
            canonical_json_bytes(
                {
                    "case": case,
                    "operational_fixture_id": operational_id,
                    "failed": failed,
                }
            )
        )

    require(len(base_states) == target_count, "M11 base-domain drift")
    for state in base_states:
        token = state.current[0]
        action = AtomicAction(ActionType.SUB, 0, token)
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        add_fixture(
            "identity_SUB",
            state,
            (action,),
            min_length=1,
            max_length=4,
            failed=action in rates,
        )

    operational_sequences = tiny_sequences("ACGU")
    targets = ("increase", "decrease")
    for source, token, remaining, region, target in itertools.product(
        operational_sequences,
        ALPHABET,
        (1, 2, 3),
        ("5UTR", "3UTR"),
        targets,
    ):
        state = _replayed_operational_state(
            source,
            token,
            remaining_budget=remaining,
            region=region,
            target_condition=target,
        )
        action = AtomicAction(ActionType.DEL, 0)
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=6,
        )
        add_fixture(
            "minimum_length_DEL",
            state,
            (action,),
            min_length=1,
            max_length=6,
            failed=action in rates,
        )

    length_three = tuple(
        sequence for sequence in operational_sequences if len(sequence) == 3
    )
    for source, current, remaining, region, target in itertools.product(
        operational_sequences,
        length_three,
        (1, 2),
        ("5UTR", "3UTR"),
        targets,
    ):
        state = _replayed_operational_state(
            source,
            current,
            remaining_budget=remaining,
            region=region,
            target_condition=target,
        )
        gap = (len(source) + remaining) % (len(current) + 1)
        action = AtomicAction(ActionType.INS, gap, ALPHABET[(remaining + gap) % 4])
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=3,
        )
        add_fixture(
            "maximum_length_INS",
            state,
            (action,),
            min_length=1,
            max_length=3,
            failed=action in rates,
        )

    for source, remaining, region, target, position in itertools.product(
        operational_sequences,
        (1, 2, 3),
        ("5UTR", "3UTR"),
        targets,
        (0, 1, 2),
    ):
        if position >= len(source):
            continue
        state = EditState.initial(
            source,
            budget=remaining,
            region=region,
            target_condition=target,
            protected_indices=(position,),
        )
        replacement = next(
            token for token in ALPHABET if token != state.current[position]
        )
        actions = (
            AtomicAction(ActionType.SUB, position, replacement),
            AtomicAction(ActionType.DEL, position),
        )
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        add_fixture(
            "protected_token_SUB_DEL",
            state,
            actions,
            min_length=1,
            max_length=4,
            failed=any(action in rates for action in actions),
        )

    length_two_or_three = tuple(
        sequence for sequence in operational_sequences if len(sequence) >= 2
    )
    for source, remaining, region, target, gap in itertools.product(
        length_two_or_three,
        (1, 2, 3),
        ("5UTR", "3UTR"),
        targets,
        (1, 2),
    ):
        if gap >= len(source):
            continue
        state = EditState.initial(
            source,
            budget=remaining,
            region=region,
            target_condition=target,
            protected_indices=(gap - 1, gap),
        )
        action = AtomicAction(ActionType.INS, gap, ALPHABET[(gap + remaining) % 4])
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        add_fixture(
            "protected_internal_gap_INS",
            state,
            (action,),
            min_length=1,
            max_length=4,
            failed=action in rates,
        )

    short_sequences = tuple(
        sequence for sequence in operational_sequences if len(sequence) <= 2
    )
    for source, current, region, target in itertools.product(
        short_sequences,
        short_sequences,
        ("5UTR", "3UTR"),
        targets,
    ):
        state = _replayed_operational_state(
            source,
            current,
            remaining_budget=0,
            region=region,
            target_condition=target,
        )
        attempted = AtomicAction(
            ActionType.SUB,
            0,
            next(token for token in ALPHABET if token != state.current[0]),
        )
        rates = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        add_fixture(
            "zero_budget_edits",
            state,
            (attempted,),
            min_length=1,
            max_length=4,
            failed=any(action.kind != ActionType.STOP for action in rates),
        )

    for state in base_states:
        halted = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
        rates = enumerate_action_rates(
            halted,
            FactorizedRates.constant(halted),
            min_length=1,
            max_length=4,
        )
        add_fixture(
            "halted_all_actions",
            halted,
            (),
            min_length=1,
            max_length=4,
            failed=bool(rates),
            attempted_action_keys=("ALL_ACTIONS",),
        )

    invalid_payloads = _m11_invalid_action_payloads()
    for attempt_index, (state, payload_spec) in enumerate(
        zip(base_states, invalid_payloads, strict=True)
    ):
        category, payload = payload_spec
        payload_id = _invalid_action_payload_id(payload)
        rejected = False
        exception_type: str | None = None
        exception_message: str | None = None
        try:
            AtomicAction(**payload)
        except (TypeError, ValueError) as exc:
            rejected = True
            exception_type = type(exc).__name__
            exception_message = str(exc)
        case = "invalid_alphabet_action"
        failed = not rejected
        case_counts[case] += 1
        case_ids[case].add(payload_id)
        case_failures[case] += int(failed)
        invalid_payload_category_counts[category] += 1
        invalid_payload_exception_counts[exception_type or "NOT_REJECTED"] += 1
        bound_state_id = sha_record(
            {"operational_state": _operational_state_payload(state)}
        )
        values = {
            "source_length": len(state.source),
            "current_length": len(state.current),
            "history_executed": state.history.executed,
            "remaining_budget": state.remaining_budget,
            "region": state.region,
            "protected_token_count": sum(
                token.protected for token in state.mapping.tokens
            ),
            "phase": state.phase.value,
            "payload_category": category,
            "payload_key_count": len(payload),
            "kind_runtime_type": (
                type(payload.get("kind")).__name__ if "kind" in payload else "MISSING"
            ),
            "position_runtime_type": (
                type(payload.get("position")).__name__
                if "position" in payload
                else "MISSING"
            ),
            "token_runtime_type": (
                type(payload.get("token")).__name__ if "token" in payload else "MISSING"
            ),
            "exception_type": exception_type or "NOT_REJECTED",
        }
        for name, value in values.items():
            histograms[case][name][str(value)] += 1
        record = {
            "attempt_index": attempt_index,
            "category": category,
            "attempted_payload": payload,
            "attempted_payload_sha256": payload_id,
            "bound_operational_state_sha256": bound_state_id,
            "rejected": rejected,
            "exception_type": exception_type,
            "exception_message": exception_message,
        }
        invalid_payload_records.append(record)
        invalid_payload_stream.update(canonical_json_bytes(record))
        fixture_stream.update(
            canonical_json_bytes(
                {
                    "case": case,
                    "operational_fixture_id": payload_id,
                    "bound_operational_state_sha256": bound_state_id,
                    "failed": failed,
                }
            )
        )

    expected_cases = {
        "identity_SUB",
        "minimum_length_DEL",
        "maximum_length_INS",
        "protected_token_SUB_DEL",
        "protected_internal_gap_INS",
        "zero_budget_edits",
        "halted_all_actions",
        "invalid_alphabet_action",
    }
    require(
        set(case_counts) == expected_cases
        and all(count == target_count for count in case_counts.values())
        and all(len(case_ids[case]) == target_count for case in expected_cases),
        "M11 operational fixture domain drift",
    )
    total_failures = sum(case_failures.values())
    require(total_failures == 0, "M11 hard-masked action retained a rate")
    return {
        "masked_illegal_rate_checks": target_count * len(expected_cases),
        "masked_illegal_rate_failure_count": total_failures,
        "masked_illegal_case_counts": dict(case_counts),
        "masked_illegal_case_failure_counts": dict(case_failures),
        "masked_illegal_case_unique_operational_fixture_counts": {
            case: len(case_ids[case]) for case in sorted(expected_cases)
        },
        "masked_illegal_operational_histograms": {
            case: {name: dict(values) for name, values in histograms[case].items()}
            for case in sorted(expected_cases)
        },
        "masked_illegal_operational_fixture_stream_sha256": (
            fixture_stream.hexdigest()
        ),
        "masked_illegal_operational_id_excludes_context": True,
        "masked_illegal_invalid_action_payload_attempt_count": len(
            invalid_payload_records
        ),
        "masked_illegal_invalid_action_payload_rejection_count": sum(
            record["rejected"] for record in invalid_payload_records
        ),
        "masked_illegal_invalid_action_payload_unique_count": len(
            case_ids["invalid_alphabet_action"]
        ),
        "masked_illegal_invalid_action_payload_category_counts": dict(
            sorted(invalid_payload_category_counts.items())
        ),
        "masked_illegal_invalid_action_payload_exception_counts": dict(
            sorted(invalid_payload_exception_counts.items())
        ),
        "masked_illegal_invalid_action_payload_stream_sha256": (
            invalid_payload_stream.hexdigest()
        ),
        "masked_illegal_invalid_action_payload_uniqueness_excludes_state": True,
        "masked_illegal_invalid_action_payload_records": invalid_payload_records,
    }


def _m12_zero_instantaneous_hazard_audit() -> dict[str, Any]:
    """Exercise one cubic-t0 fixture plus 63 distinct constructed states."""

    fixtures: list[tuple[str, EditState, Any]] = []
    cubic_state = EditState.initial(
        "AC",
        budget=2,
        context={
            "assay": "mk0_m12",
            "cell_or_tissue": "synthetic",
            "endpoint": "cubic_t0",
            "batch": "cubic-t0",
        },
    )

    def cubic_target_rate(
        state: EditState, time: float
    ) -> Mapping[AtomicAction, float]:
        replacement = next(token for token in ALPHABET if token != state.current[0])
        return {AtomicAction(ActionType.SUB, 0, replacement): rho(time, name="cubic")}

    fixtures.append(("cubic_t0", cubic_state, cubic_target_rate))
    sequences = [
        "".join(tokens)
        for length in (1, 2, 3)
        for tokens in itertools.product(ALPHABET, repeat=length)
        if "".join(tokens) != "AC"
    ][:63]
    require(len(sequences) == 63, "M12 constructed sequence fixture drift")
    for index, sequence in enumerate(sequences):
        state = EditState.initial(
            sequence,
            budget=1 + index % 3,
            region="5UTR" if index % 2 == 0 else "3UTR",
            target_condition="increase" if index % 2 == 0 else "decrease",
            context={
                "assay": "mk0_m12",
                "cell_or_tissue": "synthetic",
                "endpoint": "constructed_zero_hazard",
                "batch": f"constructed-{index:02d}",
            },
        )
        scale = 0.25 + (index + 1) / 21.0
        power = 1 + index % 3

        def constructed_rate(
            current: EditState,
            time: float,
            *,
            scale_value: float = scale,
            power_value: int = power,
        ) -> Mapping[AtomicAction, float]:
            replacement = next(
                token for token in ALPHABET if token != current.current[0]
            )
            return {
                AtomicAction(ActionType.SUB, 0, replacement): (
                    scale_value * time**power_value
                )
            }

        fixtures.append(("constructed_zero_hazard", state, constructed_rate))

    fixture_ids = {state.state_hash for _kind, state, _rate_fn in fixtures}
    require(
        len(fixtures) == len(fixture_ids) == 64, "M12 state fixtures are not distinct"
    )
    failures = 0
    fixture_stream = hashlib.sha256()
    minimum_time_advance = math.inf
    maximum_initial_hazard = 0.0
    kind_counts: dict[str, int] = defaultdict(int)
    for offset, (kind, state, rate_fn) in enumerate(fixtures):
        initial_hazard = math.fsum(rate_fn(state, 0.0).values())
        result = constrained_single_event_first_order(
            state,
            rate_fn,
            step_size=0.25,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            seed=SEED + offset,
        )
        first_step = result.steps[0] if result.steps else None
        time_advance = first_step.t_end - first_step.t_start if first_step else 0.0
        minimum_time_advance = min(minimum_time_advance, time_advance)
        maximum_initial_hazard = max(maximum_initial_hazard, initial_hazard)
        failed = (
            initial_hazard != 0.0
            or first_step is None
            or first_step.total_hazard != 0.0
            or first_step.event_probability != 0.0
            or first_step.event_draw is not None
            or first_step.action_draw is not None
            or first_step.selected_action is not None
            or first_step.outcome != "NO_EVENT"
            or time_advance <= 0.0
        )
        failures += int(failed)
        kind_counts[kind] += 1
        fixture_stream.update(
            canonical_json_bytes(
                {
                    "kind": kind,
                    "state_hash": state.state_hash,
                    "initial_total_hazard": initial_hazard,
                    "first_step_time_advance": time_advance,
                    "first_step_outcome": (
                        first_step.outcome if first_step is not None else None
                    ),
                    "failed": failed,
                }
            )
        )
    require(failures == 0, "M12 zero-instantaneous-hazard semantics failed")
    return {
        "zero_instantaneous_hazard_checks": len(fixtures),
        "zero_instantaneous_hazard_failure_count": failures,
        "zero_instantaneous_hazard_fixture_kind_counts": dict(kind_counts),
        "zero_instantaneous_hazard_unique_state_fixture_count": len(fixture_ids),
        "zero_instantaneous_hazard_minimum_time_advance": minimum_time_advance,
        "zero_instantaneous_hazard_maximum_initial_hazard": maximum_initial_hazard,
        "zero_instantaneous_hazard_fixture_stream_sha256": fixture_stream.hexdigest(),
    }


def _m13_fixture_state(stratum: str, index: int, sequence: str) -> EditState:
    return EditState.initial(
        sequence,
        budget=1 + index % 3,
        region="5UTR" if index % 2 == 0 else "3UTR",
        target_condition="increase" if index % 2 == 0 else "decrease",
        context={
            "assay": "mk0_m13",
            "cell_or_tissue": "synthetic",
            "endpoint": stratum,
            "batch": f"{stratum}-{index:02d}",
        },
    )


def _m13_future_rate_factory(*, scale: float, power: int, onset: float) -> Any:
    def future_rate(state: EditState, time: float) -> Mapping[AtomicAction, float]:
        replacement = next(token for token in ALPHABET if token != state.current[0])
        return {
            AtomicAction(ActionType.SUB, 0, replacement): (
                scale * max(0.0, time - onset) ** power
            )
        }

    return future_rate


def _m13_remaining_integrated_hazard_audit() -> dict[str, Any]:
    """Run 32 distinct zero-integral and 32 distinct future-positive fixtures."""

    sequences = [
        "".join(tokens)
        for length in (1, 2, 3)
        for tokens in itertools.product(ALPHABET, repeat=length)
    ][:64]
    require(len(sequences) == 64, "M13 sequence fixture domain drift")
    failures = 0
    zero_certificates: list[dict[str, Any]] = []
    future_certificates: list[dict[str, Any]] = []
    state_ids: dict[str, set[str]] = defaultdict(set)
    fixture_ids: dict[str, set[str]] = defaultdict(set)
    fixture_stream = hashlib.sha256()
    zero_rates = lambda _state, _time: {}

    for index in range(32):
        stratum = "zero_integral"
        state = _m13_fixture_state(stratum, index, sequences[index])
        horizon = 0.55 + 0.40 * (index + 1) / 33.0
        fixture = {
            "stratum": stratum,
            "index": index,
            "state_hash": state.state_hash,
            "scale": 0.0,
            "power": 1 + index % 4,
            "rate_start_time": 0.01 + 0.20 * (index + 1) / 33.0,
            "integration_start_time": 0.0,
            "horizon": horizon,
        }
        fixture_hash = sha_record(fixture)
        certificate = certify_remaining_integrated_hazard(
            state,
            0.0,
            zero_rates,
            horizon=horizon,
            min_length=1,
            max_length=6,
            lower_order=64,
            higher_order=128,
            zero_atol=1.0e-10,
            convergence_atol=1.0e-10,
        )
        result = constrained_single_event_first_order(
            state,
            zero_rates,
            step_size=0.25,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            horizon=horizon,
            seed=SEED + index,
            remaining_hazard_verifier=lambda current, time, h=horizon: certify_remaining_integrated_hazard(
                current,
                time,
                zero_rates,
                horizon=h,
                min_length=1,
                max_length=6,
                lower_order=64,
                higher_order=128,
                zero_atol=1.0e-10,
                convergence_atol=1.0e-10,
            ),
        )
        failed = (
            not certificate.verified_zero
            or certificate.integral != 0.0
            or result.final_state.termination_reason
            != TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
            or bool(result.steps)
            or result.remaining_hazard_certificate != certificate
        )
        failures += int(failed)
        state_ids[stratum].add(state.state_hash)
        fixture_ids[stratum].add(fixture_hash)
        zero_certificates.append(
            {
                **fixture,
                "fixture_hash": fixture_hash,
                **asdict(certificate),
                "failed": failed,
            }
        )
        fixture_stream.update(canonical_json_bytes(zero_certificates[-1]))

    for index in range(32):
        stratum = "future_positive"
        state = _m13_fixture_state(stratum, index, sequences[32 + index])
        scale = 0.5 + (index + 1) / 16.0
        power = 1 + index % 4
        onset = 0.01 + 0.20 * (index + 1) / 33.0
        horizon = 0.70 + 0.25 * (index + 1) / 33.0
        rate_fn = _m13_future_rate_factory(scale=scale, power=power, onset=onset)
        fixture = {
            "stratum": stratum,
            "index": index,
            "state_hash": state.state_hash,
            "scale": scale,
            "power": power,
            "rate_start_time": onset,
            "integration_start_time": 0.0,
            "horizon": horizon,
        }
        fixture_hash = sha_record(fixture)
        certificate = certify_remaining_integrated_hazard(
            state,
            0.0,
            rate_fn,
            horizon=horizon,
            min_length=1,
            max_length=6,
            lower_order=64,
            higher_order=128,
            zero_atol=1.0e-10,
            convergence_atol=1.0e-10,
        )
        result = constrained_single_event_first_order(
            state,
            rate_fn,
            step_size=0.25,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            horizon=horizon,
            seed=SEED + 32 + index,
            remaining_hazard_verifier=lambda current, time, fn=rate_fn, h=horizon: certify_remaining_integrated_hazard(
                current,
                time,
                fn,
                horizon=h,
                min_length=1,
                max_length=6,
                lower_order=64,
                higher_order=128,
                zero_atol=1.0e-10,
                convergence_atol=1.0e-10,
            ),
        )
        first_step = result.steps[0] if result.steps else None
        failed = (
            certificate.verified_zero
            or certificate.integral <= 0.0
            or result.final_state.termination_reason
            == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
            or first_step is None
            or first_step.total_hazard != 0.0
            or first_step.t_end <= first_step.t_start
        )
        failures += int(failed)
        state_ids[stratum].add(state.state_hash)
        fixture_ids[stratum].add(fixture_hash)
        future_certificates.append(
            {
                **fixture,
                "fixture_hash": fixture_hash,
                **asdict(certificate),
                "failed": failed,
            }
        )
        fixture_stream.update(canonical_json_bytes(future_certificates[-1]))

    require(
        set(state_ids) == {"zero_integral", "future_positive"}
        and all(len(values) == 32 for values in state_ids.values())
        and all(len(values) == 32 for values in fixture_ids.values()),
        "M13 distinct state/fixture domain drift",
    )
    require(failures == 0, "M13 remaining-integrated-hazard oracle failed")
    return {
        "zero_remaining_integrated_hazard_checks": 64,
        "zero_remaining_integrated_hazard_failure_count": failures,
        "zero_remaining_integrated_hazard_zero_case_count": 32,
        "zero_remaining_integrated_hazard_future_positive_case_count": 32,
        "zero_remaining_integrated_hazard_unique_state_counts": {
            key: len(values) for key, values in state_ids.items()
        },
        "zero_remaining_integrated_hazard_unique_fixture_counts": {
            key: len(values) for key, values in fixture_ids.items()
        },
        "zero_remaining_integrated_hazard_max_zero_integral": max(
            item["integral"] for item in zero_certificates
        ),
        "zero_remaining_integrated_hazard_min_future_positive_integral": min(
            item["integral"] for item in future_certificates
        ),
        "zero_remaining_integrated_hazard_max_quadrature_disagreement": max(
            item["disagreement"] for item in zero_certificates + future_certificates
        ),
        "zero_remaining_integrated_hazard_fixture_stream_sha256": (
            fixture_stream.hexdigest()
        ),
        "zero_remaining_integrated_hazard_fixture_records": (
            zero_certificates + future_certificates
        ),
    }


def run_hazard_audit(rng: random.Random) -> dict[str, Any]:
    schedule_audit = _m09_schedule_endpoint_derivative_audit()
    rho_failures = 0
    for name in ("cubic", "linear"):
        for index in range(1001):
            t = index * (1.0 - 1.0e-4) / 1000
            kappa, derivative = (
                cubic_schedule(t) if name == "cubic" else linear_schedule(t)
            )
            rho_failures += int(
                not math.isclose(
                    rho(t, name=name),
                    derivative / (1.0 - kappa),
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
            )
    require(rho_failures == 0, "schedule/rho oracle failed")

    negative_rate_count = 0
    evaluated_action_rate_count = 0
    factorization_failures = 0
    generator_failures = 0
    positive_hazard_states = list(tiny_active_rate_states())
    conditioned_distribution_audit = _m14_conditioned_distribution_audit(
        positive_hazard_states
    )
    extended_active_states, extended_halted_states = tiny_active_halted_rate_states()
    preregistered_states = list(extended_active_states + extended_halted_states)
    real_lengths = (16, 32, 64, 128, 256)
    random_states = [
        EditState.initial(
            "".join(
                rng.choice(ALPHABET)
                for _ in range(real_lengths[index % len(real_lengths)])
            ),
            budget=index % 3,
            region="5UTR" if index % 2 == 0 else "3UTR",
        )
        for index in range(1024)
    ]
    rate_states = preregistered_states + random_states
    require(len(rate_states) == 2200, "M10/M15 rate-state count drift")
    for index, state in enumerate(rate_states):
        maximum = 4 if index < 1176 else 300
        ins = 0.1 + rng.random()
        sub = 0.1 + rng.random()
        delete = 0.1 + rng.random()
        stop = 0.1 + rng.random()
        rates = FactorizedRates.constant(
            state, ins=ins, sub=sub, delete=delete, stop=stop
        )
        actions = enumerate_action_rates(state, rates, min_length=1, max_length=maximum)
        evaluated_action_rate_count += len(actions)
        negative_rate_count += sum(
            value < 0.0 or not math.isfinite(value) for value in actions.values()
        )
        expected = 0.0
        if state.phase == Phase.ACTIVE:
            expected = stop
        if state.phase == Phase.ACTIVE and state.remaining_budget > 0:
            if len(state.current) < maximum:
                expected += math.fsum(rates.ins_operation)
            expected += math.fsum(rates.sub_operation)
            if len(state.current) > 1:
                expected += math.fsum(rates.delete)
        factorization_failures += int(
            not math.isclose(
                total_hazard(actions), expected, abs_tol=ATOL, rel_tol=RTOL
            )
        )
    for state in extended_active_states + extended_halted_states:
        actions = enumerate_action_rates(
            state,
            FactorizedRates.constant(state),
            min_length=1,
            max_length=4,
        )
        row = generator(state, actions, min_length=1, max_length=4)
        generator_failures += int(
            not math.isclose(row.row_sum, 0.0, abs_tol=ATOL, rel_tol=RTOL)
        )
    require(
        negative_rate_count == factorization_failures == generator_failures == 0,
        "rate/generator oracle failure",
    )

    m11_mask_audit = _m11_mask_audit(extended_active_states)

    zero_instantaneous_audit = _m12_zero_instantaneous_hazard_audit()
    zero_remaining_audit = _m13_remaining_integrated_hazard_audit()
    require(
        zero_instantaneous_audit["zero_instantaneous_hazard_failure_count"]
        == zero_remaining_audit["zero_remaining_integrated_hazard_failure_count"]
        == 0,
        "zero hazard semantics failed",
    )
    return {
        "schema_version": "mk0_hazard_audit_v1",
        "status": "PASS",
        "seed": SEED,
        "dtype": "float64",
        "rho_checks": 2002,
        "rho_failure_count": rho_failures,
        **schedule_audit,
        "negative_rate_checks": 2200,
        "evaluated_action_rate_count": evaluated_action_rate_count,
        "tiny_source_current_budget_run_state_count": len(preregistered_states),
        "tiny_active_run_state_count": len(extended_active_states),
        "tiny_halted_run_state_count": len(extended_halted_states),
        "tiny_positive_hazard_region_paired_state_count": len(positive_hazard_states),
        "random_real_length_state_count": len(random_states),
        "random_real_lengths": list(real_lengths),
        "negative_rate_count": negative_rate_count,
        **m11_mask_audit,
        **conditioned_distribution_audit,
        "factorization_checks": 2200,
        "factorization_failure_count": factorization_failures,
        "generator_checks": 1176,
        "generator_failure_count": generator_failures,
        **zero_instantaneous_audit,
        **zero_remaining_audit,
    }


def _softplus_float(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _central_finite_difference_gradient(
    raw_values: list[float],
    loss_fn: Any,
    *,
    epsilon: float,
) -> list[float]:
    gradient: list[float] = []
    for index in range(len(raw_values)):
        lower = list(raw_values)
        upper = list(raw_values)
        lower[index] -= epsilon
        upper[index] += epsilon
        gradient.append((loss_fn(upper) - loss_fn(lower)) / (2.0 * epsilon))
    return gradient


def _m19_finite_difference_gradient_audit(
    rng: random.Random,
) -> dict[str, Any]:
    """Compare 64 production-loss autograd vectors to central differences."""

    import torch

    finite_difference_atol = 1.0e-6
    finite_difference_rtol = 1.0e-5
    finite_difference_epsilon = 1.0e-6
    state = EditState.initial("A", budget=1)
    alignment = build_alignment("A", "C")
    oracle = build_target_transition_oracle(
        state,
        alignment,
        {index: 0.75 for index in changed_indices(alignment)},
        0.5,
        min_length=1,
        max_length=1,
    )
    actions = enumerate_legal_actions(
        state,
        min_length=1,
        max_length=1,
        include_stop=False,
    )
    require(bool(actions), "M19 finite-difference action domain is empty")
    finite_gradient_failure_count = 0
    analytic_gradient_failure_count = 0
    maximum_finite_difference_gradient_error = 0.0
    maximum_analytic_gradient_error = 0.0
    finite_difference_coordinate_count = 0
    gradient_stream = hashlib.sha256()
    hash_by_action = {
        action: apply_action(state, action, min_length=1, max_length=1).after.state_hash
        for action in actions
    }
    for case_index in range(64):
        raw_values = [-1.0 + 2.0 * rng.random() for _ in actions]
        raw = torch.tensor(raw_values, dtype=torch.float64, requires_grad=True)
        rates_tensor = torch.nn.functional.softplus(raw)
        rates = dict(zip(actions, rates_tensor))
        loss_value = edit_flow_loss(
            state,
            rates,
            oracle,
            min_length=1,
            max_length=1,
        )
        loss_value.backward()
        autograd_values = [float(value) for value in raw.grad.detach().tolist()]

        def scalar_loss(values: list[float]) -> float:
            scalar_rates = {
                action: _softplus_float(value) for action, value in zip(actions, values)
            }
            return float(
                edit_flow_loss(
                    state,
                    scalar_rates,
                    oracle,
                    min_length=1,
                    max_length=1,
                )
            )

        finite_difference_values = _central_finite_difference_gradient(
            raw_values,
            scalar_loss,
            epsilon=finite_difference_epsilon,
        )
        finite_difference_coordinate_count += len(actions)
        finite_difference_errors = [
            abs(observed - expected)
            for observed, expected in zip(autograd_values, finite_difference_values)
        ]
        maximum_finite_difference_gradient_error = max(
            maximum_finite_difference_gradient_error,
            max(finite_difference_errors, default=0.0),
        )
        finite_failure = (
            not bool(torch.isfinite(loss_value))
            or raw.grad is None
            or not bool(torch.all(torch.isfinite(raw.grad)))
            or any(not math.isfinite(value) for value in finite_difference_values)
            or any(
                not math.isclose(
                    observed,
                    expected,
                    abs_tol=finite_difference_atol,
                    rel_tol=finite_difference_rtol,
                )
                for observed, expected in zip(autograd_values, finite_difference_values)
            )
        )
        finite_gradient_failure_count += int(finite_failure)

        total_by_hash: dict[str, float] = defaultdict(float)
        for action, rate in rates.items():
            total_by_hash[hash_by_action[action]] += float(rate.detach())
        analytic_values = []
        for action_index, action in enumerate(actions):
            key = hash_by_action[action]
            weight = oracle.target_transition_weights.get(key, 0.0)
            analytic_values.append(
                float(torch.sigmoid(raw.detach()[action_index]))
                * (1.0 - weight / total_by_hash[key])
            )
        analytic_errors = [
            abs(observed - expected)
            for observed, expected in zip(autograd_values, analytic_values)
        ]
        maximum_analytic_gradient_error = max(
            maximum_analytic_gradient_error,
            max(analytic_errors, default=0.0),
        )
        analytic_gradient_failure_count += int(
            any(
                not math.isclose(
                    observed,
                    expected,
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
                for observed, expected in zip(autograd_values, analytic_values)
            )
        )
        gradient_stream.update(
            canonical_json_bytes(
                {
                    "case_index": case_index,
                    "raw": raw_values,
                    "autograd": autograd_values,
                    "finite_difference": finite_difference_values,
                    "analytic": analytic_values,
                    "finite_difference_failed": finite_failure,
                }
            )
        )
    require(
        finite_gradient_failure_count == analytic_gradient_failure_count == 0,
        "M19 finite-difference/analytic gradient oracle failed",
    )
    return {
        "gradient_case_count": 64,
        "finite_gradient_count": 64,
        "finite_gradient_failure_count": finite_gradient_failure_count,
        "finite_difference_gradient_case_count": 64,
        "finite_difference_coordinate_count": finite_difference_coordinate_count,
        "finite_difference_epsilon": finite_difference_epsilon,
        "finite_difference_atol": finite_difference_atol,
        "finite_difference_rtol": finite_difference_rtol,
        "maximum_finite_difference_gradient_error": (
            maximum_finite_difference_gradient_error
        ),
        "analytic_gradient_failure_count": analytic_gradient_failure_count,
        "maximum_analytic_gradient_error": maximum_analytic_gradient_error,
        "gradient_fixture_stream_sha256": gradient_stream.hexdigest(),
    }


def _m17_transition_parameter_audit(
    repeated_cases: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Bind M17 to five named cases crossed with 64 real parameter vectors."""

    require(len(repeated_cases) == 5, "M17 named-case domain drift")
    case_counts: dict[str, int] = defaultdict(int)
    parameter_failures = 0
    target_aggregation_failures = 0
    model_aggregation_failures = 0
    transition_replay_failures = 0
    maximum_target_aggregation_error = 0.0
    maximum_model_aggregation_error = 0.0
    parameter_stream = hashlib.sha256()
    parameter_records: list[dict[str, Any]] = []
    for case_index, (source, target) in enumerate(repeated_cases):
        case = f"{source}_to_{target}"
        alignment = build_alignment(source, target)
        state = EditState.initial(source, budget=max(4, alignment.cost))
        legal_actions = enumerate_legal_actions(
            state,
            min_length=0,
            max_length=8,
            include_stop=False,
        )
        require(bool(legal_actions), f"M17 {case} has no legal edit scripts")
        for vector_index in range(64):
            time = 0.05 + 0.90 * (vector_index + 0.5) / 64.0
            clocks = {index: 0.975 for index in changed_indices(alignment)}
            oracle = build_target_transition_oracle(
                state,
                alignment,
                clocks,
                time,
                min_length=0,
                max_length=8,
            )
            target_expected_lists: dict[str, list[float]] = defaultdict(list)
            replay_failure = False
            for transition in oracle.transitions:
                target_expected_lists[transition.next_state_hash].append(
                    transition.weight
                )
                replayed = apply_action(
                    state,
                    transition.action,
                    min_length=0,
                    max_length=8,
                ).after
                replay_failure |= replayed.state_hash != transition.next_state_hash
            target_expected = {
                key: math.fsum(values) for key, values in target_expected_lists.items()
            }
            target_observed = oracle.target_transition_weights
            target_key_failure = set(target_observed) != set(target_expected)
            target_error = max(
                (
                    abs(target_observed.get(key, math.nan) - value)
                    for key, value in target_expected.items()
                ),
                default=0.0,
            )
            target_failure = target_key_failure or any(
                not math.isclose(
                    target_observed.get(key, math.nan),
                    value,
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
                for key, value in target_expected.items()
            )

            rates = {
                action: (
                    0.05
                    + (
                        ((case_index + 1) * (vector_index + 3) * (action_index + 5))
                        % 211
                    )
                    / 97.0
                )
                for action_index, action in enumerate(legal_actions)
            }
            model_observed = aggregate_transition_rates(
                state,
                rates,
                min_length=0,
                max_length=8,
            )
            model_expected_lists: dict[str, list[float]] = defaultdict(list)
            for action, rate in rates.items():
                next_state = apply_action(
                    state,
                    action,
                    min_length=0,
                    max_length=8,
                ).after
                model_expected_lists[next_state.state_hash].append(rate)
            model_expected = {
                key: math.fsum(values) for key, values in model_expected_lists.items()
            }
            model_key_failure = set(model_observed) != set(model_expected)
            model_error = max(
                (
                    abs(model_observed.get(key, math.nan) - value)
                    for key, value in model_expected.items()
                ),
                default=0.0,
            )
            model_failure = model_key_failure or any(
                not math.isclose(
                    model_observed.get(key, math.nan),
                    value,
                    abs_tol=ATOL,
                    rel_tol=RTOL,
                )
                for key, value in model_expected.items()
            )
            failed = target_failure or model_failure or replay_failure
            target_aggregation_failures += int(target_failure)
            model_aggregation_failures += int(model_failure)
            transition_replay_failures += int(replay_failure)
            parameter_failures += int(failed)
            maximum_target_aggregation_error = max(
                maximum_target_aggregation_error, target_error
            )
            maximum_model_aggregation_error = max(
                maximum_model_aggregation_error, model_error
            )
            case_counts[case] += 1
            parameter_record = {
                "case": case,
                "parameter_vector_index": vector_index,
                "time": time,
                "clock": 0.975,
                "legal_edit_script_count": len(legal_actions),
                "target_transition_count": len(oracle.transitions),
                "target_aggregate_count": len(target_observed),
                "model_aggregate_count": len(model_observed),
                "failed": failed,
            }
            parameter_records.append(parameter_record)
            parameter_stream.update(
                canonical_json_bytes(
                    {
                        **parameter_record,
                        "state_hash": state.state_hash,
                        "alignment_hash": alignment.alignment_hash,
                        "rates": [
                            [action.key, rates[action]] for action in legal_actions
                        ],
                        "target_aggregates": sorted(target_observed.items()),
                        "model_aggregates": sorted(model_observed.items()),
                    }
                )
            )
    require(
        len(parameter_records) == 320
        and set(case_counts)
        == {f"{source}_to_{target}" for source, target in repeated_cases}
        and all(count == 64 for count in case_counts.values()),
        "M17 case-by-parameter domain drift",
    )
    require(parameter_failures == 0, "M17 transition parameter aggregation failed")
    return {
        "sample_count": len(parameter_records),
        "failure_count": parameter_failures,
        "named_case_parameter_vector_counts": dict(case_counts),
        "parameter_vector_count": len(parameter_records),
        "all_legal_edit_scripts_per_vector": True,
        "target_aggregation_failure_count": target_aggregation_failures,
        "model_aggregation_failure_count": model_aggregation_failures,
        "transition_replay_failure_count": transition_replay_failures,
        "maximum_target_aggregation_error": maximum_target_aggregation_error,
        "maximum_model_aggregation_error": maximum_model_aggregation_error,
        "actual_atol": ATOL,
        "actual_rtol": RTOL,
        "parameter_vector_stream_sha256": parameter_stream.hexdigest(),
        "parameter_vectors": parameter_records,
    }


def run_transition_loss_audits(
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repeated_cases = (
        ("AA", "A"),
        ("A", "AA"),
        ("AAA", "AA"),
        ("AA", "AAA"),
        ("ACA", "AAC"),
    )
    m17_parameter_audit = _m17_transition_parameter_audit(repeated_cases)
    min_length, max_length, t = 0, 8, 0.5
    canonical_records: list[dict[str, Any]] = []
    canonical_remaining_switches = 0
    full_state_hash_replay_failures = 0
    non_sha_target_key_count = 0
    for source, target in repeated_cases:
        alignment = build_alignment(source, target)
        state = EditState.initial(source, budget=max(4, alignment.cost))
        clocks = {index: 0.75 for index in changed_indices(alignment)}
        oracle = build_target_transition_oracle(
            state,
            alignment,
            clocks,
            t,
            min_length=min_length,
            max_length=max_length,
        )
        canonical_remaining_switches += len(oracle.transitions)
        for transition in oracle.transitions:
            replayed = apply_action(
                state,
                transition.action,
                min_length=min_length,
                max_length=max_length,
            ).after
            full_state_hash_replay_failures += int(
                replayed.state_hash != transition.next_state_hash
            )
        non_sha_target_key_count += sum(
            re.fullmatch(r"[0-9a-f]{64}", key) is None
            for key in oracle.target_transition_weights
        )
        canonical_records.append(
            {"case": f"{source}_to_{target}", "oracle": oracle.to_record()}
        )

    sampled_optimal_path_count = 0
    sampled_remaining_switch_count = 0
    sampled_replay_failures = 0
    for source, target in repeated_cases:
        state = EditState.initial(source, budget=4)
        for _ in range(128):
            alignment = sample_optimal_alignment(source, target, rng=rng)
            oracle = build_target_transition_oracle(
                state,
                alignment,
                {index: 0.75 for index in changed_indices(alignment)},
                t,
                min_length=min_length,
                max_length=max_length,
            )
            sampled_optimal_path_count += 1
            sampled_remaining_switch_count += len(oracle.transitions)
            for transition in oracle.transitions:
                replayed = apply_action(
                    state,
                    transition.action,
                    min_length=min_length,
                    max_length=max_length,
                ).after
                sampled_replay_failures += int(
                    replayed.state_hash != transition.next_state_hash
                )

    collision_state = EditState.initial("AA", budget=2)
    collision_alignment = build_alignment("AA", "")
    collision_oracle = build_target_transition_oracle(
        collision_state,
        collision_alignment,
        {index: 0.75 for index in changed_indices(collision_alignment)},
        t,
        min_length=min_length,
        max_length=max_length,
    )
    observable_collision_cases = int(
        len({item.observable_next for item in collision_oracle.transitions}) == 1
        and len({item.next_state_hash for item in collision_oracle.transitions}) == 2
    )
    observable_collisions_conflated = int(
        len(collision_oracle.target_transition_weights) != 2
    )

    multiplicity_state = EditState.initial("", budget=2)
    multiplicity_alignment = build_alignment("", "AA")
    multiplicity_oracle = build_target_transition_oracle(
        multiplicity_state,
        multiplicity_alignment,
        {index: 0.75 for index in changed_indices(multiplicity_alignment)},
        t,
        min_length=min_length,
        max_length=max_length,
    )
    multiplicity_expected = 2.0 * rho(t)
    multiplicity_observed = math.fsum(
        multiplicity_oracle.target_transition_weights.values()
    )
    target_multiplicity_weight_failures = int(
        len(multiplicity_oracle.target_transition_weights) != 1
        or not math.isclose(
            multiplicity_observed,
            multiplicity_expected,
            abs_tol=ATOL,
            rel_tol=RTOL,
        )
    )

    equivalent_order_failures = 0
    equivalent_alignment = build_alignment("AC", "GU")
    equivalent_clocks = {index: 0.25 for index in changed_indices(equivalent_alignment)}
    initial = EditState.initial("AC", budget=2)
    for actions in (
        (
            AtomicAction(ActionType.SUB, 0, "G"),
            AtomicAction(ActionType.SUB, 1, "U"),
        ),
        (
            AtomicAction(ActionType.SUB, 1, "U"),
            AtomicAction(ActionType.SUB, 0, "G"),
        ),
    ):
        completed, _ = replay_actions(
            initial, actions, min_length=min_length, max_length=max_length
        )
        oracle = build_target_transition_oracle(
            completed,
            equivalent_alignment,
            equivalent_clocks,
            t,
            min_length=min_length,
            max_length=max_length,
        )
        equivalent_order_failures += int(bool(oracle.transitions))

    rejected_records: list[dict[str, Any]] = []
    try:
        protected_alignment = build_alignment("A", "C")
        build_target_transition_oracle(
            EditState.initial("A", budget=1, protected_indices=(0,)),
            protected_alignment,
            {index: 0.75 for index in changed_indices(protected_alignment)},
            t,
            min_length=1,
            max_length=1,
        )
    except TargetKernelRejected as error:
        rejected_records.append(error.to_record())
    explicit_rejection_count = len(rejected_records)
    silent_repair_count = sum(
        record["repair_applied_count"] for record in rejected_records
    )

    loss_failure_count = 0
    finite_loss_failure_count = 0
    maximum_absolute_loss_error = 0.0
    production_loss_count = 0
    for source, target in repeated_cases:
        alignment = build_alignment(source, target)
        state = EditState.initial(source, budget=max(4, alignment.cost))
        oracle = build_target_transition_oracle(
            state,
            alignment,
            {index: 0.75 for index in changed_indices(alignment)},
            t,
            min_length=min_length,
            max_length=max_length,
        )
        legal = enumerate_legal_actions(
            state,
            min_length=min_length,
            max_length=max_length,
            include_stop=False,
        )
        for _ in range(64):
            rates = {action: 0.05 + rng.random() for action in legal}
            observed = float(
                edit_flow_loss(
                    state,
                    rates,
                    oracle,
                    min_length=min_length,
                    max_length=max_length,
                )
            )
            model_pairs = [
                (
                    apply_action(
                        state,
                        action,
                        min_length=min_length,
                        max_length=max_length,
                    ).after.state_hash,
                    rate,
                )
                for action, rate in rates.items()
            ]
            target_pairs = [
                (transition.next_state_hash, transition.weight)
                for transition in oracle.transitions
            ]
            expected = brute_force_bregman_loss(model_pairs, target_pairs)
            error = abs(observed - expected)
            maximum_absolute_loss_error = max(maximum_absolute_loss_error, error)
            finite_loss_failure_count += int(not math.isfinite(observed))
            loss_failure_count += int(
                not math.isclose(observed, expected, abs_tol=ATOL, rel_tol=RTOL)
            )
            production_loss_count += 1

    m19_gradient_audit = _m19_finite_difference_gradient_audit(rng)
    gradient_state = EditState.initial("A", budget=1)
    gradient_alignment = build_alignment("A", "C")
    gradient_oracle = build_target_transition_oracle(
        gradient_state,
        gradient_alignment,
        {index: 0.75 for index in changed_indices(gradient_alignment)},
        t,
        min_length=1,
        max_length=1,
    )
    gradient_actions = enumerate_legal_actions(
        gradient_state, min_length=1, max_length=1, include_stop=False
    )

    halted_exact_zero_count = 0
    halted_exact_zero_failure_count = 0
    for _ in range(64):
        active = EditState.initial("A", budget=1)
        halted = apply_action(
            active,
            AtomicAction(ActionType.STOP),
            min_length=1,
            max_length=1,
        ).after
        value = edit_flow_loss(halted, {}, None, min_length=1, max_length=1)
        halted_exact_zero_count += 1
        halted_exact_zero_failure_count += int(value != 0.0)

    incomplete_neighbourhood_rejections = 0
    complete_rates = {action: 1.0 for action in gradient_actions}
    incomplete_rates = dict(complete_rates)
    incomplete_rates.pop(next(iter(incomplete_rates)))
    try:
        edit_flow_loss(
            gradient_state,
            incomplete_rates,
            gradient_oracle,
            min_length=1,
            max_length=1,
        )
    except ValueError:
        incomplete_neighbourhood_rejections += 1
    oracle_state_binding_rejections = 0
    different_state = EditState.initial("A", budget=1, context={"assay": "different"})
    different_rates = {
        action: 1.0
        for action in enumerate_legal_actions(
            different_state, min_length=1, max_length=1, include_stop=False
        )
    }
    try:
        edit_flow_loss(
            different_state,
            different_rates,
            gradient_oracle,
            min_length=1,
            max_length=1,
        )
    except ValueError:
        oracle_state_binding_rejections += 1

    transition_failure_count = (
        m17_parameter_audit["failure_count"]
        + full_state_hash_replay_failures
        + sampled_replay_failures
        + non_sha_target_key_count
        + observable_collisions_conflated
        + target_multiplicity_weight_failures
        + equivalent_order_failures
    )
    combined_loss_failure_count = (
        loss_failure_count
        + finite_loss_failure_count
        + m19_gradient_audit["finite_gradient_failure_count"]
        + m19_gradient_audit["analytic_gradient_failure_count"]
        + halted_exact_zero_failure_count
        + int(incomplete_neighbourhood_rejections != 1)
        + int(oracle_state_binding_rejections != 1)
    )
    require(
        transition_failure_count == 0,
        "full-state target transition aggregation oracle failed",
    )
    require(
        explicit_rejection_count == 1 and silent_repair_count == 0,
        "target-kernel rejection/repair ledger failed",
    )
    require(combined_loss_failure_count == 0, "production Bregman/loss oracle failed")
    return (
        {
            "schema_version": "mk0_transition_aggregation_oracle_v2",
            "status": "PASS",
            "seed": SEED,
            "sample_count": m17_parameter_audit["sample_count"],
            "failure_count": transition_failure_count,
            **{
                key: value
                for key, value in m17_parameter_audit.items()
                if key not in {"sample_count", "failure_count"}
            },
            "required_named_cases": [
                f"{source}_to_{target}" for source, target in repeated_cases
            ],
            "canonical_path_count": len(canonical_records),
            "canonical_remaining_switch_count": canonical_remaining_switches,
            "sampled_optimal_path_count": sampled_optimal_path_count,
            "sampled_remaining_switch_count": sampled_remaining_switch_count,
            "full_state_hash_replay_failure_count": full_state_hash_replay_failures
            + sampled_replay_failures,
            "non_sha_target_key_count": non_sha_target_key_count,
            "observable_collision_case_count": observable_collision_cases,
            "observable_collisions_conflated": observable_collisions_conflated,
            "target_multiplicity_weight_failure_count": target_multiplicity_weight_failures,
            "equivalent_edit_order_check_count": 2,
            "equivalent_edit_order_failure_count": equivalent_order_failures,
            "canonical_oracles": canonical_records,
            "multiplicity_oracle": multiplicity_oracle.to_record(),
            "collision_oracle": collision_oracle.to_record(),
        },
        {
            "schema_version": "mk0_loss_oracle_report_v2",
            "status": "PASS",
            "seed": SEED,
            "dtype": "float64",
            "loss_oracle_count": production_loss_count,
            "loss_failure_count": loss_failure_count,
            "finite_loss_count": production_loss_count,
            "finite_loss_failure_count": finite_loss_failure_count,
            **m19_gradient_audit,
            "finite_loss_and_gradient_sample_count": production_loss_count
            + m19_gradient_audit["gradient_case_count"],
            "finite_loss_and_gradient_failure_count": finite_loss_failure_count
            + m19_gradient_audit["finite_gradient_failure_count"],
            "maximum_absolute_loss_error": maximum_absolute_loss_error,
            "maximum_gradient_error": m19_gradient_audit[
                "maximum_finite_difference_gradient_error"
            ],
            "valid_target_switch_count": canonical_remaining_switches
            + sampled_remaining_switch_count,
            "explicit_rejection_count": explicit_rejection_count,
            "silent_repair_count": silent_repair_count,
            "rejected_path_records": rejected_records,
            "halted_exact_zero_count": halted_exact_zero_count,
            "halted_exact_zero_failure_count": halted_exact_zero_failure_count,
            "incomplete_neighbourhood_rejections": incomplete_neighbourhood_rejections,
            "oracle_state_binding_rejections": oracle_state_binding_rejections,
            "atol": ATOL,
            "rtol": RTOL,
        },
    )


def _m21_dwell_independence_audit(rng: random.Random) -> dict[str, Any]:
    """Prove the dwell draw does not read completion, with corr only diagnostic."""

    gamma_values = (8.0, 16.0, 32.0)
    samples_per_gamma = 8192
    analytic_atol = 1.0e-8
    analytic_rtol = 1.0e-7
    clock_variants = (
        {0: 0.05, 1: 0.10, 2: 0.15},
        {0: 0.35, 1: 0.45, 2: 0.55},
        {0: 0.75, 1: 0.85, 2: 0.95},
    )
    gamma_reports: dict[str, dict[str, Any]] = {}
    total_nonpositive_count = 0
    total_strictly_positive_count = 0
    dwell_oracle_failure_count = 0
    structural_independence_failure_count = 0
    analytic_inverse_failure_count = 0
    sample_stream = hashlib.sha256()
    for gamma_index, gamma in enumerate(gamma_values):
        completions = np.empty(samples_per_gamma, dtype=np.float64)
        dwells = np.empty(samples_per_gamma, dtype=np.float64)
        events = np.empty(samples_per_gamma, dtype=np.float64)
        gamma_structural_failures = 0
        gamma_analytic_failures = 0
        gamma_dwell_oracle_failures = 0
        for index in range(samples_per_gamma):
            completion_uniform = rng.random()
            targets = [
                sample_stop_target(
                    clocks,
                    gamma_ref=gamma,
                    rng=random.Random((SEED + gamma_index * samples_per_gamma + index)),
                )
                for clocks in clock_variants
            ]
            # Replay the exact uniform consumed by sample_stop_target.  The
            # independent oracle is constructed from that draw, not completion.
            replay_rng = random.Random(SEED + gamma_index * samples_per_gamma + index)
            replay_uniform = max(replay_rng.random(), math.nextafter(0.0, 1.0))
            replay_expected_dwell = max(
                -math.log1p(-replay_uniform) / gamma,
                math.nextafter(0.0, 1.0),
            )
            representative = targets[index % len(targets)]
            bitwise_equal = len({target.dwell.hex() for target in targets}) == 1
            analytic_equal = all(
                target.dwell.hex() == replay_expected_dwell.hex()
                and math.isclose(
                    target.dwell,
                    replay_expected_dwell,
                    abs_tol=analytic_atol,
                    rel_tol=analytic_rtol,
                )
                for target in targets
            )
            # `uniform` is used only to choose a diagnostic completion clock;
            # the dwell comes from the separately seeded RNG above.
            diagnostic_completion = min(0.999, completion_uniform)
            diagnostic_target = sample_stop_target(
                {0: diagnostic_completion},
                gamma_ref=gamma,
                rng=random.Random(SEED + gamma_index * samples_per_gamma + index),
            )
            structural_failed = (
                not bitwise_equal
                or diagnostic_target.dwell.hex() != targets[0].dwell.hex()
            )
            analytic_failed = not analytic_equal
            sample_failed = (
                structural_failed
                or analytic_failed
                or not diagnostic_target.dwell > 0.0
            )
            gamma_structural_failures += int(structural_failed)
            gamma_analytic_failures += int(analytic_failed)
            gamma_dwell_oracle_failures += int(sample_failed)
            completions[index] = diagnostic_target.completion_time
            dwells[index] = diagnostic_target.dwell
            events[index] = float(diagnostic_target.event_observed)
            sample_stream.update(
                canonical_json_bytes(
                    {
                        "gamma_ref": gamma,
                        "sample_index": index,
                        "uniform_for_completion_diagnostic": completion_uniform,
                        "independent_rng_uniform": replay_uniform,
                        "independent_rng_expected_dwell": replay_expected_dwell,
                        "observed_dwell_hex": [
                            target.dwell.hex() for target in targets
                        ],
                        "diagnostic_completion": diagnostic_target.completion_time,
                        "event_observed": diagnostic_target.event_observed,
                        "structural_failed": structural_failed,
                        "analytic_failed": analytic_failed,
                        "sample_failed": sample_failed,
                    }
                )
            )
        nonpositive_count = int(np.sum(dwells <= 0.0))
        strictly_positive_count = int(np.sum(dwells > 0.0))
        total_nonpositive_count += nonpositive_count
        total_strictly_positive_count += strictly_positive_count
        structural_independence_failure_count += gamma_structural_failures
        analytic_inverse_failure_count += gamma_analytic_failures
        dwell_oracle_failure_count += gamma_dwell_oracle_failures
        correlation = float(np.corrcoef(completions, dwells)[0, 1])
        bounded_correlation = min(max(correlation, -1.0 + 1.0e-15), 1.0 - 1.0e-15)
        fisher_z_statistic = math.atanh(bounded_correlation) * math.sqrt(
            samples_per_gamma - 3
        )
        two_sided_p_value = math.erfc(abs(fisher_z_statistic) / math.sqrt(2.0))
        expected_event_fraction = float(
            np.mean(1.0 - np.exp(-gamma * (1.0 - completions)))
        )
        gamma_reports[str(gamma)] = {
            "gamma_ref": gamma,
            "sample_count": samples_per_gamma,
            "strictly_positive_dwell_count": strictly_positive_count,
            "nonpositive_dwell_count": nonpositive_count,
            "structural_independence_failure_count": gamma_structural_failures,
            "analytic_inverse_failure_count": gamma_analytic_failures,
            "dwell_oracle_failure_count": gamma_dwell_oracle_failures,
            "completion_clock_variant_count": len(clock_variants),
            "completion_dwell_pearson_correlation_diagnostic": correlation,
            "fisher_z_statistic_diagnostic": fisher_z_statistic,
            "two_sided_normal_approximation_p_value_diagnostic": two_sided_p_value,
            "empirical_diagnostic_used_for_gate": False,
            "observed_mean_dwell": float(dwells.mean()),
            "expected_mean_dwell": 1.0 / gamma,
            "observed_event_fraction": float(events.mean()),
            "expected_event_fraction": expected_event_fraction,
            "event_fraction_absolute_error": abs(
                float(events.mean()) - expected_event_fraction
            ),
        }
    require(
        total_strictly_positive_count == samples_per_gamma * len(gamma_values)
        and total_nonpositive_count == 0,
        "M21 STOP dwell was not strictly positive in the frozen sample",
    )
    require(
        structural_independence_failure_count == analytic_inverse_failure_count == 0,
        "M21 structural dwell-independence/inverse-transform oracle failed",
    )
    return {
        "dwell_sample_count": samples_per_gamma * len(gamma_values),
        "dwell_samples_per_gamma": samples_per_gamma,
        "dwell_gamma_values": list(gamma_values),
        "strictly_positive_dwell_count": total_strictly_positive_count,
        "nonpositive_dwell_count": total_nonpositive_count,
        "dwell_oracle_failure_count": dwell_oracle_failure_count,
        "structural_independence_check_count": samples_per_gamma * len(gamma_values),
        "structural_independence_failure_count": structural_independence_failure_count,
        "analytic_inverse_check_count": samples_per_gamma * len(gamma_values),
        "analytic_inverse_failure_count": analytic_inverse_failure_count,
        "completion_clock_variant_count": len(clock_variants),
        "dwell_analytic_atol": analytic_atol,
        "dwell_analytic_rtol": analytic_rtol,
        "independence_oracle": "shared_rng_draw_multiple_completion_clocks_bitwise",
        "empirical_pearson_fisher_role": "DIAGNOSTIC_ONLY_NOT_GATE",
        "independence_claim_boundary": (
            "STRUCTURAL_RNG_INDEPENDENCE_ORACLE_PLUS_EMPIRICAL_DIAGNOSTIC"
        ),
        "gamma_dwell_independence": gamma_reports,
        "dwell_sample_stream_sha256": sample_stream.hexdigest(),
    }


def run_stop_audits(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute_failures = 0
    primary_errors: list[float] = []
    reference_errors: list[float] = []
    quadrature_disagreements: list[float] = []
    for index in range(32):
        event = bool(index % 2)
        observed = 0.25 + 0.7 * (index / 31)
        latent = observed if event else 1.1 + 0.01 * index
        target = StopTarget(
            0.1,
            latent - 0.1,
            latent,
            observed if event else 1.0,
            event,
        )
        hazard = 0.125 + 16.0 * (index + 1) / 32
        numerical = survival_stop_loss(
            target, lambda _t, h=hazard: h, quadrature_points=64
        )
        reference = survival_stop_loss(
            target, lambda _t, h=hazard: h, quadrature_points=128
        )
        analytic = constant_hazard_stop_loss(target, hazard)
        primary_errors.append(abs(numerical - analytic))
        reference_errors.append(abs(reference - analytic))
        quadrature_disagreements.append(abs(numerical - reference))
        absolute_failures += int(
            not math.isclose(numerical, analytic, abs_tol=1.0e-8, rel_tol=1.0e-7)
            or not math.isclose(reference, analytic, abs_tol=1.0e-8, rel_tol=1.0e-7)
            or not math.isclose(numerical, reference, abs_tol=1.0e-8, rel_tol=1.0e-7)
        )

    for index in range(32):
        event = bool(index % 2)
        observed = 0.55 + 0.4 * (index / 31)
        latent = observed if event else 1.1 + 0.01 * index
        target = StopTarget(
            0.1,
            latent - 0.1,
            latent,
            observed if event else 1.0,
            event,
        )
        cut = 0.15 + 0.3 * ((index % 8) / 7)
        left_hazard = 0.25 + 0.125 * index
        right_hazard = 1.0 + 0.25 * index

        def piecewise_rate(
            time: float,
            *,
            cut_value: float = cut,
            left: float = left_hazard,
            right: float = right_hazard,
        ) -> float:
            return left if time < cut_value else right

        numerical = survival_stop_loss(
            target,
            piecewise_rate,
            quadrature_points=64,
            breakpoints=(cut,),
        )
        reference = survival_stop_loss(
            target,
            piecewise_rate,
            quadrature_points=128,
            breakpoints=(cut,),
        )
        analytic = (
            left_hazard * cut
            + right_hazard * (target.observed_time - cut)
            - int(event) * math.log(right_hazard)
        )
        primary_errors.append(abs(numerical - analytic))
        reference_errors.append(abs(reference - analytic))
        quadrature_disagreements.append(abs(numerical - reference))
        absolute_failures += int(
            not math.isclose(numerical, analytic, abs_tol=1.0e-8, rel_tol=1.0e-7)
            or not math.isclose(reference, analytic, abs_tol=1.0e-8, rel_tol=1.0e-7)
            or not math.isclose(numerical, reference, abs_tol=1.0e-8, rel_tol=1.0e-7)
        )
    require(len(primary_errors) == 64, "M20 fixture-count drift")
    require(absolute_failures == 0, "STOP absolute-hazard oracle failure")

    m21_dwell_audit = _m21_dwell_independence_audit(rng)
    gamma16_dwell_report = m21_dwell_audit["gamma_dwell_independence"]["16.0"]
    correlation = gamma16_dwell_report[
        "completion_dwell_pearson_correlation_diagnostic"
    ]
    expected_event_fraction = gamma16_dwell_report["expected_event_fraction"]
    observed_event_fraction = gamma16_dwell_report["observed_event_fraction"]
    event_fraction_tolerance = 0.02
    gamma_event_fraction = {
        gamma: {
            "sample_count": report["sample_count"],
            "observed_event_fraction": report["observed_event_fraction"],
            "expected_event_fraction": report["expected_event_fraction"],
            "absolute_error": report["event_fraction_absolute_error"],
            "failed": report["event_fraction_absolute_error"]
            >= event_fraction_tolerance,
        }
        for gamma, report in m21_dwell_audit["gamma_dwell_independence"].items()
    }
    event_fraction_failure_count = sum(
        report["failed"] for report in gamma_event_fraction.values()
    )
    require(
        event_fraction_failure_count == 0,
        "M22 gamma 8/16/32 event/censor fraction oracle failed",
    )
    gamma_sensitivity = {
        str(gamma): stop_event_censor_oracle(
            gamma_ref=gamma,
            completion_times=(0.0, 0.25, 0.5, 0.75, 0.9),
            samples_per_completion=5000,
            seed=SEED,
        )
        for gamma in (8.0, 16.0, 32.0)
    }
    gamma_sensitivity_failure_count = sum(
        report["absolute_fraction_error"] >= 0.01
        for report in gamma_sensitivity.values()
    )

    halted_failures = 0
    halted_rate_rejection_count = 0
    _tiny_active_domain, tiny_halted_domain = tiny_active_halted_rate_states()
    for index, halted in enumerate(tiny_halted_domain):
        halted_failures += int(
            edit_flow_loss(halted, {}, None, min_length=1, max_length=4) != 0.0
        )
        if index == 0:
            try:
                edit_flow_loss(
                    halted,
                    {AtomicAction(ActionType.SUB, 0, "C"): 1.0},
                    None,
                    min_length=1,
                    max_length=4,
                )
            except ValueError:
                halted_rate_rejection_count += 1
    # Exercise the real sampler state-machine priority for five terminal
    # outcomes and the real fail-closed numerical exception path for the sixth.
    # Merely serializing an injected enum would not test termination semantics.
    reason_failures = 0
    reasons = list(TerminationReason)
    reason_counts: dict[str, int] = defaultdict(int)
    actual_sampler_reason_counts: dict[str, int] = defaultdict(int)
    termination_operational_fixture_ids: dict[str, set[str]] = defaultdict(set)
    termination_stream = hashlib.sha256()
    termination_sequences = tiny_sequences("ACGU")[:64]

    def termination_state(reason: TerminationReason, offset: int) -> EditState:
        source = termination_sequences[offset]
        region = "5UTR" if offset % 2 == 0 else "3UTR"
        target_condition = "increase" if offset % 2 == 0 else "decrease"
        if reason == TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION:
            return EditState.initial(
                source,
                budget=1 + offset % 3,
                region=region,
                target_condition=target_condition,
                protected_indices=tuple(range(len(source))),
            )
        current = termination_sequences[
            (offset * 17 + reasons.index(reason) * 11) % len(termination_sequences)
        ]
        remaining_budget = (
            0 if reason == TerminationReason.FORCED_BUDGET else 1 + offset % 2
        )
        return _replayed_operational_state(
            source,
            current,
            remaining_budget=remaining_budget,
            region=region,
            target_condition=target_condition,
        )

    for offset in range(64):
        learned_hazard = 1.0e6 + 1000.0 * offset
        zero_horizon = 0.75 + 0.25 * (offset + 1) / 65.0
        time_horizon = 0.50 + 0.50 * (offset + 1) / 65.0

        def zero_certificate(
            state: EditState,
            time: float,
            *,
            horizon: float = zero_horizon,
        ):
            return certify_remaining_integrated_hazard(
                state,
                time,
                lambda _state, _time: {},
                horizon=horizon,
                min_length=1,
                max_length=4,
                lower_order=64,
                higher_order=128,
                zero_atol=1.0e-10,
                convergence_atol=1.0e-10,
            )

        no_legal_state = termination_state(
            TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION, offset
        )
        sampler_cases = (
            (
                TerminationReason.LEARNED_STOP,
                termination_state(TerminationReason.LEARNED_STOP, offset),
                stop_only(learned_hazard),
                {},
                {"rate_fixture": "stop_only", "stop_hazard": learned_hazard},
            ),
            (
                TerminationReason.FORCED_BUDGET,
                termination_state(TerminationReason.FORCED_BUDGET, offset),
                stop_only(1.0),
                {},
                {"rate_fixture": "stop_only", "stop_hazard": 1.0},
            ),
            (
                TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION,
                no_legal_state,
                lambda _state, _time: {},
                {
                    "min_length": len(no_legal_state.current),
                    "max_length": len(no_legal_state.current),
                },
                {
                    "rate_fixture": "zero_rates_all_tokens_protected_fixed_length",
                    "protected_count": len(no_legal_state.current),
                },
            ),
            (
                TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD,
                termination_state(
                    TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD,
                    offset,
                ),
                lambda _state, _time: {},
                {
                    "horizon": zero_horizon,
                    "remaining_hazard_verifier": zero_certificate,
                },
                {
                    "rate_fixture": "verified_zero_remaining_hazard",
                    "horizon": zero_horizon,
                },
            ),
            (
                TerminationReason.FORCED_TIME_HORIZON,
                termination_state(TerminationReason.FORCED_TIME_HORIZON, offset),
                lambda _state, _time: {},
                {"horizon": time_horizon},
                {"rate_fixture": "zero_rates_no_verifier", "horizon": time_horizon},
            ),
        )
        for case_index, (
            expected_reason,
            initial,
            rate_fn,
            overrides,
            fixture_parameters,
        ) in enumerate(sampler_cases):
            options = {
                "step_size": 1.0,
                "stability_hazard": 0.05,
                "min_length": 1,
                "max_length": 4,
                "seed": SEED + 1000 * case_index + offset,
            }
            options.update(overrides)
            operational_id = _operational_fixture_id(
                initial,
                case=expected_reason.value,
                attempted_action_keys=(fixture_parameters["rate_fixture"],),
                min_length=options["min_length"],
                max_length=options["max_length"],
                parameters={
                    **fixture_parameters,
                    "step_size": options["step_size"],
                    "stability_hazard": options["stability_hazard"],
                    "horizon": options.get("horizon", 1.0),
                },
            )
            result = constrained_single_event_first_order(initial, rate_fn, **options)
            observed_reason = result.final_state.termination_reason
            replay_ok = replay_constrained_result(
                result,
                rate_fn,
                min_length=options["min_length"],
                max_length=options["max_length"],
                remaining_hazard_verifier=options.get("remaining_hazard_verifier"),
            )
            failure = int(observed_reason != expected_reason or not replay_ok)
            reason_failures += failure
            actual_sampler_reason_counts[expected_reason.value] += 1
            record = termination_to_schema_record(
                reason=expected_reason,
                external_time=result.termination_time,
                state_hash_before=result.termination_before_hash,
                state_hash_after=result.final_state.state_hash,
                remaining_integrated_total_hazard=(
                    0.0
                    if expected_reason
                    == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
                    else None
                ),
            )
            validate_schema_facing_record(record, "termination")
            separation_failure = int(
                record["learned_stop"] == record["forced_termination"]
                and (record["learned_stop"] or record["forced_termination"])
            )
            reason_failures += separation_failure
            reason_counts[expected_reason.value] += 1
            termination_operational_fixture_ids[expected_reason.value].add(
                operational_id
            )
            termination_stream.update(
                canonical_json_bytes(
                    {
                        "termination_record": record,
                        "operational_fixture_id": operational_id,
                        "fixture_parameters": fixture_parameters,
                    }
                )
            )

        numerical_state = termination_state(TerminationReason.FAILED_NUMERICAL, offset)
        invalid_stop_rate = -1.0 - offset / 64.0

        def invalid_numerical_rate(
            _state: EditState, _time: float
        ) -> Mapping[AtomicAction, float]:
            return {AtomicAction(ActionType.STOP): invalid_stop_rate}

        numerical_operational_id = _operational_fixture_id(
            numerical_state,
            case=TerminationReason.FAILED_NUMERICAL.value,
            attempted_action_keys=(AtomicAction(ActionType.STOP).key,),
            min_length=1,
            max_length=4,
            parameters={
                "rate_fixture": "negative_stop_rate",
                "invalid_stop_rate": invalid_stop_rate,
                "step_size": 1.0,
                "stability_hazard": 0.05,
                "horizon": 1.0,
            },
        )

        numerical_rejected = False
        diagnostic = ""
        try:
            constrained_single_event_first_order(
                numerical_state,
                invalid_numerical_rate,
                step_size=1.0,
                stability_hazard=0.05,
                min_length=1,
                max_length=4,
                seed=SEED + 5000 + offset,
            )
        except (FloatingPointError, ValueError) as error:
            numerical_rejected = True
            diagnostic = f"{type(error).__name__}: {error}"
        reason_failures += int(not numerical_rejected)
        numerical_record = termination_to_schema_record(
            reason=TerminationReason.FAILED_NUMERICAL,
            external_time=0.0,
            state_hash_before=numerical_state.state_hash,
            # FAILED_NUMERICAL is an execution status outside the CTMC.  It
            # must not manufacture a new biological/CTMC state transition.
            state_hash_after=numerical_state.state_hash,
            diagnostic=diagnostic or "numerical failure was not rejected",
        )
        validate_schema_facing_record(numerical_record, "termination")
        reason_counts[TerminationReason.FAILED_NUMERICAL.value] += 1
        termination_operational_fixture_ids[
            TerminationReason.FAILED_NUMERICAL.value
        ].add(numerical_operational_id)
        termination_stream.update(
            canonical_json_bytes(
                {
                    "termination_record": numerical_record,
                    "operational_fixture_id": numerical_operational_id,
                    "fixture_parameters": {
                        "rate_fixture": "negative_stop_rate",
                        "invalid_stop_rate": invalid_stop_rate,
                    },
                }
            )
        )
    require(sum(reason_counts.values()) == 384, "M24 sample-count drift")
    require(
        set(reason_counts) == {reason.value for reason in TerminationReason}
        and all(count == 64 for count in reason_counts.values()),
        "M24 termination-domain drift",
    )
    require(
        set(termination_operational_fixture_ids) == set(reason_counts)
        and all(
            len(fixture_ids) == 64
            for fixture_ids in termination_operational_fixture_ids.values()
        ),
        "M24 unique operational-fixture drift",
    )
    require(
        halted_failures == reason_failures == 0 and halted_rate_rejection_count == 1,
        "STOP termination separation failed",
    )
    return (
        {
            "schema_version": "mk0_stop_audit_v1",
            "status": "PASS",
            "halted_edit_flow_checks": 588,
            "halted_edit_flow_failure_count": halted_failures,
            "halted_production_loss_api": "edit_flow_loss",
            "halted_nonempty_rate_rejection_count": halted_rate_rejection_count,
            "termination_separation_checks": 384,
            "termination_separation_failure_count": reason_failures,
            "termination_reasons": [reason.value for reason in reasons],
            "termination_reason_counts": dict(reason_counts),
            "termination_reason_unique_operational_fixture_counts": {
                reason: len(fixture_ids)
                for reason, fixture_ids in termination_operational_fixture_ids.items()
            },
            "termination_operational_id_excludes_context": True,
            "actual_sampler_reason_counts": dict(actual_sampler_reason_counts),
            "failed_numerical_exception_case_count": reason_counts[
                TerminationReason.FAILED_NUMERICAL.value
            ],
            "failed_numerical_mutates_ctmc_state": False,
            "termination_record_stream_sha256": termination_stream.hexdigest(),
        },
        {
            "schema_version": "mk0_stop_survival_oracle_v1",
            "status": "PASS",
            "absolute_hazard_checks": 64,
            "absolute_hazard_failure_count": absolute_failures,
            "constant_hazard_fixture_count": 32,
            "piecewise_hazard_fixture_count": 32,
            "maximum_primary_vs_analytic_error": max(primary_errors),
            "maximum_reference_vs_analytic_error": max(reference_errors),
            "maximum_64_vs_128_quadrature_disagreement": max(quadrature_disagreements),
            **m21_dwell_audit,
            "completion_dwell_correlation": correlation,
            "observed_event_fraction": observed_event_fraction,
            "expected_event_fraction": expected_event_fraction,
            "event_fraction_absolute_error": abs(
                observed_event_fraction - expected_event_fraction
            ),
            "event_fraction_check_count": len(gamma_event_fraction),
            "event_fraction_failure_count": event_fraction_failure_count,
            "event_fraction_tolerance": event_fraction_tolerance,
            "gamma_event_fraction": gamma_event_fraction,
            "observed_mean_dwell": gamma16_dwell_report["observed_mean_dwell"],
            "expected_mean_dwell": 1.0 / 16.0,
            "gamma_sensitivity": gamma_sensitivity,
            "gamma_sensitivity_failure_count": gamma_sensitivity_failure_count,
            "gamma_sensitivity_role": "DIAGNOSTIC_ONLY_NOT_M22_GATE",
            "quadrature": {
                "primary_nodes_per_smooth_interval": 64,
                "reference_nodes_per_smooth_interval": 128,
                "piecewise_breakpoints_split_explicitly": True,
            },
        },
    )


def factorized_rate_fn(state: EditState, _time: float) -> Mapping[AtomicAction, float]:
    """Sparse nonzero oracle rates cycling through INS/SUB/DEL then budget."""

    if state.remaining_budget <= 0:
        return {AtomicAction(ActionType.STOP): 0.1}
    phase = state.history.executed % 3
    if phase == 0:
        edit = AtomicAction(ActionType.INS, len(state.current), "A")
    elif phase == 1:
        token = "C" if state.current[0] != "C" else "G"
        edit = AtomicAction(ActionType.SUB, 0, token)
    else:
        edit = AtomicAction(ActionType.DEL, len(state.current) - 1)
    return {edit: 4.0, AtomicAction(ActionType.STOP): 0.1}


def stop_only(hazard: float):
    def rate_fn(state: EditState, _time: float) -> Mapping[AtomicAction, float]:
        return (
            {AtomicAction(ActionType.STOP): hazard}
            if state.phase == Phase.ACTIVE
            else {}
        )

    return rate_fn


def convergence_rate_fn(state: EditState, time: float) -> Mapping[AtomicAction, float]:
    """Frozen time-varying and post-edit-state-dependent convergence oracle."""

    if state.phase != Phase.ACTIVE:
        return {}
    rates: dict[AtomicAction, float] = {
        AtomicAction(ActionType.STOP): 0.15 + 0.2 * time
    }
    if state.remaining_budget > 0:
        if state.history.executed % 2 == 0:
            token = "C" if state.current[0] != "C" else "G"
            action = AtomicAction(ActionType.SUB, 0, token)
        else:
            action = AtomicAction(ActionType.INS, len(state.current), "A")
        rates[action] = 0.3 + 0.5 * time + 0.1 * state.history.executed
    return rates


def _terminal_signature(state: EditState) -> tuple[int, str, str]:
    require(state.phase == Phase.HALTED, "terminal distribution contains ACTIVE state")
    return (
        state.history.executed,
        str(state.termination_reason.value),
        state.current,
    )


def _frozen_grid_distribution(step_size: float) -> dict[tuple[int, str, str], float]:
    """Exact probability propagation for the frozen-rate numerical scheme."""

    steps = round(1.0 / step_size)
    require(
        math.isclose(steps * step_size, 1.0, abs_tol=1.0e-15),
        "unfrozen convergence grid",
    )
    active: dict[EditState, float] = {EditState.initial("AC", budget=2): 1.0}
    terminal: dict[tuple[int, str, str], float] = defaultdict(float)
    for step in range(steps):
        time = step * step_size
        next_active: dict[EditState, float] = defaultdict(float)
        for state, mass in active.items():
            rates = dict(convergence_rate_fn(state, time))
            hazard = total_hazard(rates)
            require(hazard > 0.0, "convergence oracle unexpectedly has zero hazard")
            event_probability = -math.expm1(-step_size * hazard)
            next_active[state] += mass * (1.0 - event_probability)
            for action, rate in rates.items():
                event_mass = mass * event_probability * rate / hazard
                after = apply_action(state, action, min_length=1, max_length=6).after
                if after.phase == Phase.HALTED:
                    terminal[_terminal_signature(after)] += event_mass
                elif after.remaining_budget == 0:
                    after = force_terminate(after, TerminationReason.FORCED_BUDGET)
                    terminal[_terminal_signature(after)] += event_mass
                else:
                    next_active[after] += event_mass
        active = dict(next_active)
    for state, mass in active.items():
        halted = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
        terminal[_terminal_signature(halted)] += mass
    return dict(terminal)


def _rk4_integrated_hazard_reference() -> dict[tuple[int, str, str], float]:
    """Independent RK4 solution of the two-active-state forward equations."""

    # Vector: active_before_edit, active_after_SUB, stop_before, stop_after,
    # forced_budget_after_SUB_then_INS.
    values = [1.0, 0.0, 0.0, 0.0, 0.0]
    step = 1.0 / 16384

    def derivative(time: float, vector: list[float]) -> list[float]:
        p0, p1, _stop0, _stop1, _budget = vector
        stop_hazard = 0.15 + 0.2 * time
        first_edit_hazard = 0.3 + 0.5 * time
        second_edit_hazard = 0.4 + 0.5 * time
        return [
            -(first_edit_hazard + stop_hazard) * p0,
            first_edit_hazard * p0 - (second_edit_hazard + stop_hazard) * p1,
            stop_hazard * p0,
            stop_hazard * p1,
            second_edit_hazard * p1,
        ]

    for index in range(16384):
        time = index * step
        k1 = derivative(time, values)
        k2 = derivative(
            time + step / 2,
            [value + step * delta / 2 for value, delta in zip(values, k1)],
        )
        k3 = derivative(
            time + step / 2,
            [value + step * delta / 2 for value, delta in zip(values, k2)],
        )
        k4 = derivative(
            time + step,
            [value + step * delta for value, delta in zip(values, k3)],
        )
        values = [
            value + step * (a + 2 * b + 2 * c + d) / 6
            for value, a, b, c, d in zip(values, k1, k2, k3, k4)
        ]
    p0, p1, stop0, stop1, budget = values
    return {
        (0, TerminationReason.LEARNED_STOP.value, "AC"): stop0,
        (1, TerminationReason.LEARNED_STOP.value, "CC"): stop1,
        (2, TerminationReason.FORCED_BUDGET.value, "CCA"): budget,
        (0, TerminationReason.FORCED_TIME_HORIZON.value, "AC"): p0,
        (1, TerminationReason.FORCED_TIME_HORIZON.value, "CC"): p1,
    }


def _total_variation(
    left: Mapping[tuple[int, str, str], float],
    right: Mapping[tuple[int, str, str], float],
) -> float:
    return 0.5 * math.fsum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right)
    )


def _mean_edit_count(distribution: Mapping[tuple[int, str, str], float]) -> float:
    return math.fsum(key[0] * probability for key, probability in distribution.items())


def _termination_fractions(
    distribution: Mapping[tuple[int, str, str], float]
) -> dict[str, float]:
    fractions: dict[str, float] = defaultdict(float)
    for (_edits, reason, _current), probability in distribution.items():
        fractions[reason] += probability
    return dict(fractions)


def _distribution_record(
    distribution: Mapping[tuple[int, str, str], float]
) -> list[dict[str, Any]]:
    return [
        {
            "edit_count": key[0],
            "termination_reason": key[1],
            "current": key[2],
            "probability": probability,
        }
        for key, probability in sorted(distribution.items())
    ]


M29_STEP_SIZES = (1 / 32, 1 / 64, 1 / 128, 1 / 256)
M29_TRAJECTORIES_PER_STRATUM = 512
M29_NO_EVENT = "NO_EVENT"


def _m29_independent_action_from_draw(
    rates: Mapping[AtomicAction, float], draw: float
) -> AtomicAction:
    """Invert the rate CDF without calling the production action selector."""

    require(0.0 <= draw < 1.0, "M29 action uniform is outside [0,1)")
    ordered = sorted(rates.items(), key=lambda item: item[0].key)
    require(bool(ordered), "M29 action oracle received no candidate rates")
    hazard = math.fsum(rate for _action, rate in ordered)
    require(hazard > 0.0, "M29 action oracle received zero total hazard")
    target = draw * hazard
    cumulative = 0.0
    for action, rate in ordered:
        cumulative += rate
        if target < cumulative:
            return action
    return ordered[-1][0]


def _m29_independent_first_action_law(step_size: float) -> dict[str, float]:
    """Propagate first-event survival independently of the sampler/draw path."""

    state = EditState.initial("AC", budget=2)
    survival = 1.0
    time = 0.0
    law: dict[str, float] = defaultdict(float)
    while time < 1.0:
        proposed_h = min(step_size, 1.0 - time)
        rates = dict(convergence_rate_fn(state, time))
        hazard = math.fsum(rates.values())
        require(hazard > 0.0, "M29 first-action oracle has zero hazard")
        pieces = max(1, math.ceil(proposed_h * hazard / 0.05))
        h = proposed_h / pieces
        event_probability = -math.expm1(-h * hazard)
        event_mass = survival * event_probability
        for action, rate in rates.items():
            law[action.key] += event_mass * rate / hazard
        survival *= 1.0 - event_probability
        time = min(1.0, time + h)
    law[M29_NO_EVENT] += survival
    require(
        abs(math.fsum(law.values()) - 1.0) <= 1.0e-12,
        "M29 first-action oracle lost probability mass",
    )
    return dict(law)


def _m29_law_comparison(
    counts: Mapping[Any, int],
    expected: Mapping[Any, float],
    sample_count: int,
) -> dict[str, Any]:
    keys = set(counts) | set(expected)
    observed = {key: counts.get(key, 0) / sample_count for key in keys}
    absolute_errors = {key: abs(observed[key] - expected.get(key, 0.0)) for key in keys}
    zero_expected_observation_count = sum(
        counts.get(key, 0) for key in keys if expected.get(key, 0.0) == 0.0
    )
    return {
        "max_absolute_error": max(absolute_errors.values(), default=0.0),
        "total_variation": 0.5 * math.fsum(absolute_errors.values()),
        "zero_expected_observation_count": zero_expected_observation_count,
    }


def _m29_terminal_count_records(
    counts: Mapping[tuple[int, str, str], int], sample_count: int
) -> list[dict[str, Any]]:
    return [
        {
            "edit_count": key[0],
            "termination_reason": key[1],
            "current": key[2],
            "count": count,
            "fraction": count / sample_count,
        }
        for key, count in sorted(counts.items())
    ]


def _m29_action_count_records(
    counts: Mapping[str, int], sample_count: int
) -> list[dict[str, Any]]:
    return [
        {"action_key": key, "count": count, "fraction": count / sample_count}
        for key, count in sorted(counts.items())
    ]


def _run_m29_actual_sampler_law_audit() -> dict[str, Any]:
    """Bind M29 to actual logs with independent step and action oracles."""

    initial = EditState.initial("AC", budget=2)
    input_stream = hashlib.sha256()
    terminal_stream = hashlib.sha256()
    first_action_stream = hashlib.sha256()
    action_oracle_stream = hashlib.sha256()
    expected_law_stream = hashlib.sha256()
    pooled_terminal_counts: dict[tuple[int, str, str], int] = defaultdict(int)
    pooled_action_counts: dict[str, int] = defaultdict(int)
    pooled_expected_terminal: dict[tuple[int, str, str], float] = defaultdict(float)
    pooled_expected_action: dict[str, float] = defaultdict(float)
    sampled_validity_failures = 0
    step_kernel_oracle_stream = hashlib.sha256()
    step_kernel_check_count = 0
    step_kernel_mismatch_count = 0
    action_oracle_check_count = 0
    action_oracle_mismatch_count = 0
    trajectory_kernel_replay_mismatch_count = 0
    stratum_records: list[dict[str, Any]] = []

    for stratum_index, step_size in enumerate(M29_STEP_SIZES):
        expected_terminal = _frozen_grid_distribution(step_size)
        expected_action = _m29_independent_first_action_law(step_size)
        expected_law_stream.update(
            canonical_json_bytes(
                {
                    "stratum_index": stratum_index,
                    "step_size_hex": float(step_size).hex(),
                    "terminal": _distribution_record(expected_terminal),
                    "first_action": [
                        {"action_key": key, "probability": probability}
                        for key, probability in sorted(expected_action.items())
                    ],
                }
            )
        )
        for key, probability in expected_terminal.items():
            pooled_expected_terminal[key] += probability / len(M29_STEP_SIZES)
        for key, probability in expected_action.items():
            pooled_expected_action[key] += probability / len(M29_STEP_SIZES)

        terminal_counts: dict[tuple[int, str, str], int] = defaultdict(int)
        first_action_counts: dict[str, int] = defaultdict(int)
        stratum_step_checks = 0
        stratum_step_mismatches = 0
        stratum_oracle_checks = 0
        stratum_oracle_mismatches = 0
        stratum_trajectory_replay_mismatches = 0
        adaptive_subdivision_count = 0
        for trajectory_index in range(M29_TRAJECTORIES_PER_STRATUM):
            seed = SEED + trajectory_index
            input_stream.update(
                canonical_json_bytes(
                    {
                        "stratum_index": stratum_index,
                        "step_size_hex": float(step_size).hex(),
                        "trajectory_index": trajectory_index,
                        "seed": seed,
                        "initial_state_hash": initial.state_hash,
                    }
                )
            )
            result = constrained_single_event_first_order(
                initial,
                convergence_rate_fn,
                step_size=step_size,
                stability_hazard=0.05,
                min_length=1,
                max_length=6,
                seed=seed,
            )
            sampled_validity_failures += int(
                result.final_state.phase != Phase.HALTED
                or result.final_state.history.executed > 2
                or result.edit_events != result.final_state.history.executed
                or result.final_state.source != initial.source
            )
            terminal_key = (
                result.edit_events,
                result.final_state.termination_reason.value,
                result.final_state.current,
            )
            terminal_counts[terminal_key] += 1
            pooled_terminal_counts[terminal_key] += 1
            terminal_stream.update(
                canonical_json_bytes(
                    {
                        "stratum_index": stratum_index,
                        "trajectory_index": trajectory_index,
                        "terminal_signature": list(terminal_key),
                    }
                )
            )

            replay_state = initial
            replay_time = 0.0
            first_action_key = M29_NO_EVENT
            first_action_step: int | None = None
            selected_event_count = 0
            for expected_step_index, step in enumerate(result.steps):
                adaptive_subdivision_count += step.adaptive_subdivision_count
                rates = dict(convergence_rate_fn(replay_state, replay_time))
                ordered_rates = sorted(rates.items(), key=lambda item: item[0].key)
                hazard = math.fsum(rates.values())
                proposed_h = min(step_size, 1.0 - replay_time)
                pieces = max(1, math.ceil(proposed_h * hazard / 0.05))
                expected_h = proposed_h / pieces
                expected_end = min(1.0, replay_time + expected_h)
                event_probability = -math.expm1(-expected_h * hazard)
                expected_actions_hash = hashlib.sha256(
                    json.dumps(
                        [action.key for action, _rate in ordered_rates],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                expected_rates_hash = hashlib.sha256(
                    json.dumps(
                        [
                            [action.key, float(rate).hex()]
                            for action, rate in ordered_rates
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                event_draw_valid = (
                    step.event_draw is not None and 0.0 <= step.event_draw < 1.0
                )
                expected_event = bool(
                    event_draw_valid and step.event_draw < event_probability
                )
                actual_event = step.selected_action is not None
                event_kernel_matches = all(
                    (
                        step.step == expected_step_index,
                        step.t_start == replay_time,
                        step.t_end == expected_end,
                        step.h == expected_h,
                        step.total_hazard == hazard,
                        step.event_probability == event_probability,
                        step.adaptive_subdivision_count == pieces - 1,
                        step.before_hash == replay_state.state_hash,
                        step.candidate_actions_hash == expected_actions_hash,
                        step.candidate_rates_hash == expected_rates_hash,
                        step.rate_recomputed_after_step is True,
                        event_draw_valid,
                        actual_event == expected_event,
                        (step.action_draw is not None) == expected_event,
                        step.outcome
                        == (
                            step.selected_action.kind.value
                            if actual_event
                            else M29_NO_EVENT
                        ),
                    )
                )
                step_kernel_check_count += 1
                stratum_step_checks += 1

                expected_selected: AtomicAction | None = None
                action_matches = True
                if expected_event:
                    expected_selected = (
                        None
                        if step.action_draw is None
                        else _m29_independent_action_from_draw(rates, step.action_draw)
                    )
                    action_matches = expected_selected == step.selected_action
                    action_oracle_check_count += 1
                    stratum_oracle_checks += 1
                    action_oracle_mismatch_count += int(not action_matches)
                    stratum_oracle_mismatches += int(not action_matches)
                    if step.selected_action is not None:
                        selected_event_count += 1
                        if first_action_step is None:
                            first_action_step = step.step
                            first_action_key = step.selected_action.key

                before_apply_state = replay_state
                if step.selected_action is not None:
                    replay_state = apply_action(
                        replay_state,
                        step.selected_action,
                        min_length=1,
                        max_length=6,
                    ).after
                after_hash_matches = step.after_hash == replay_state.state_hash
                step_kernel_matches = event_kernel_matches and after_hash_matches
                step_kernel_mismatch_count += int(not step_kernel_matches)
                stratum_step_mismatches += int(not step_kernel_matches)
                step_kernel_oracle_stream.update(
                    canonical_json_bytes(
                        {
                            "stratum_index": stratum_index,
                            "trajectory_index": trajectory_index,
                            "step": step.step,
                            "before_state_hash": before_apply_state.state_hash,
                            "expected_t_start_hex": float(replay_time).hex(),
                            "expected_t_end_hex": float(expected_end).hex(),
                            "expected_h_hex": float(expected_h).hex(),
                            "expected_hazard_hex": float(hazard).hex(),
                            "expected_event_probability_hex": float(
                                event_probability
                            ).hex(),
                            "event_draw_hex": (
                                None
                                if step.event_draw is None
                                else float(step.event_draw).hex()
                            ),
                            "expected_event": expected_event,
                            "actual_event": actual_event,
                            "event_kernel_matches": event_kernel_matches,
                            "after_hash_matches": after_hash_matches,
                            "step_kernel_matches": step_kernel_matches,
                            "candidate_actions_hash": expected_actions_hash,
                            "candidate_rates_hash": expected_rates_hash,
                        }
                    )
                )
                if expected_event:
                    action_oracle_stream.update(
                        canonical_json_bytes(
                            {
                                "stratum_index": stratum_index,
                                "trajectory_index": trajectory_index,
                                "step": step.step,
                                "state_hash": before_apply_state.state_hash,
                                "action_draw_hex": (
                                    None
                                    if step.action_draw is None
                                    else float(step.action_draw).hex()
                                ),
                                "ordered_rates": [
                                    [action.key, float(rate).hex()]
                                    for action, rate in ordered_rates
                                ],
                                "expected_action": (
                                    None
                                    if expected_selected is None
                                    else expected_selected.key
                                ),
                                "actual_action": (
                                    None
                                    if step.selected_action is None
                                    else step.selected_action.key
                                ),
                                "matches": action_matches,
                            }
                        )
                    )
                replay_time = expected_end

            if replay_state.phase == Phase.ACTIVE:
                if replay_state.remaining_budget == 0:
                    replay_state = force_terminate(
                        replay_state, TerminationReason.FORCED_BUDGET
                    )
                else:
                    replay_state = force_terminate(
                        replay_state, TerminationReason.FORCED_TIME_HORIZON
                    )
            trajectory_replay_matches = (
                replay_state.state_hash == result.final_state.state_hash
            )
            trajectory_kernel_replay_mismatch_count += int(
                not trajectory_replay_matches
            )
            stratum_trajectory_replay_mismatches += int(not trajectory_replay_matches)
            first_action_counts[first_action_key] += 1
            pooled_action_counts[first_action_key] += 1
            first_action_stream.update(
                canonical_json_bytes(
                    {
                        "stratum_index": stratum_index,
                        "trajectory_index": trajectory_index,
                        "first_action_step": first_action_step,
                        "first_action_key": first_action_key,
                        "selected_event_count": selected_event_count,
                    }
                )
            )

        terminal_comparison = _m29_law_comparison(
            terminal_counts, expected_terminal, M29_TRAJECTORIES_PER_STRATUM
        )
        action_comparison = _m29_law_comparison(
            first_action_counts, expected_action, M29_TRAJECTORIES_PER_STRATUM
        )
        stratum_records.append(
            {
                "stratum_index": stratum_index,
                "step_size": step_size,
                "sample_unit": "one_actual_sampler_trajectory",
                "trajectory_denominator": M29_TRAJECTORIES_PER_STRATUM,
                "seed_first": SEED,
                "seed_last": SEED + M29_TRAJECTORIES_PER_STRATUM - 1,
                "terminal_expected_distribution": _distribution_record(
                    expected_terminal
                ),
                "terminal_observed_distribution": _m29_terminal_count_records(
                    terminal_counts, M29_TRAJECTORIES_PER_STRATUM
                ),
                "terminal_law_comparison": terminal_comparison,
                "first_action_expected_distribution": [
                    {"action_key": key, "probability": probability}
                    for key, probability in sorted(expected_action.items())
                ],
                "first_action_observed_distribution": _m29_action_count_records(
                    first_action_counts, M29_TRAJECTORIES_PER_STRATUM
                ),
                "first_action_law_comparison": action_comparison,
                "empirical_distribution_role": "diagnostic_only_not_a_gate_condition",
                "step_kernel_check_denominator": stratum_step_checks,
                "step_kernel_mismatch_count": stratum_step_mismatches,
                "action_oracle_check_denominator": stratum_oracle_checks,
                "action_oracle_mismatch_count": stratum_oracle_mismatches,
                "trajectory_kernel_replay_denominator": M29_TRAJECTORIES_PER_STRATUM,
                "trajectory_kernel_replay_mismatch_count": stratum_trajectory_replay_mismatches,
                "adaptive_subdivision_count": adaptive_subdivision_count,
            }
        )

    trajectory_count = len(M29_STEP_SIZES) * M29_TRAJECTORIES_PER_STRATUM
    pooled_terminal_comparison = _m29_law_comparison(
        pooled_terminal_counts, pooled_expected_terminal, trajectory_count
    )
    pooled_action_comparison = _m29_law_comparison(
        pooled_action_counts, pooled_expected_action, trajectory_count
    )
    zero_step_kernel_failure_count = int(step_kernel_check_count == 0)
    zero_action_oracle_failure_count = int(action_oracle_check_count == 0)
    failure_count = (
        sampled_validity_failures
        + step_kernel_mismatch_count
        + action_oracle_mismatch_count
        + trajectory_kernel_replay_mismatch_count
        + zero_step_kernel_failure_count
        + zero_action_oracle_failure_count
    )
    failure_denominator = (
        2 * trajectory_count + step_kernel_check_count + action_oracle_check_count + 2
    )
    expected_stream_sha256 = expected_law_stream.hexdigest()
    audit_hash_payload = {
        "input_stream_sha256": input_stream.hexdigest(),
        "terminal_stream_sha256": terminal_stream.hexdigest(),
        "first_action_stream_sha256": first_action_stream.hexdigest(),
        "step_kernel_oracle_stream_sha256": step_kernel_oracle_stream.hexdigest(),
        "action_oracle_stream_sha256": action_oracle_stream.hexdigest(),
        "expected_law_stream_sha256": expected_stream_sha256,
        "trajectory_count": trajectory_count,
        "step_kernel_check_count": step_kernel_check_count,
        "action_oracle_check_count": action_oracle_check_count,
        "failure_count": failure_count,
    }
    return {
        "schema_version": "mk0_m29_actual_sampler_kernel_binding_v2",
        "status": "PASS" if failure_count == 0 else "FAIL",
        "sample_unit": "one_actual_sampler_trajectory_at_one_frozen_step_size",
        "trajectory_denominator": trajectory_count,
        "stratum_count": len(M29_STEP_SIZES),
        "trajectories_per_stratum": M29_TRAJECTORIES_PER_STRATUM,
        "paired_seed_design": True,
        "base_seed": SEED,
        "seed_rule": "seed=20260802+trajectory_index; identical seed paired across all four step sizes",
        "step_sizes": list(M29_STEP_SIZES),
        "horizon": 1.0,
        "stability_hazard": 0.05,
        "initial_state_hash": initial.state_hash,
        "rate_domain": "time_varying_and_post_edit_state_dependent",
        "independent_expected_terminal_method": "exact_frozen_grid_probability_propagation_without_sampler_trajectories",
        "independent_expected_first_action_method": "survival_mass_propagation_without_sampler_or_production_draw_action",
        "step_kernel_oracle_method": "independent_per_logged_step_recomputation_of_h_hazard_event_probability_event_decision_candidate_ledgers_and_state_transition",
        "step_kernel_required_agreement_fraction": 1.0,
        "step_kernel_check_denominator": step_kernel_check_count,
        "step_kernel_mismatch_count": step_kernel_mismatch_count,
        "step_kernel_agreement_fraction": (
            0.0
            if step_kernel_check_count == 0
            else 1.0 - step_kernel_mismatch_count / step_kernel_check_count
        ),
        "zero_step_kernel_failure_count": zero_step_kernel_failure_count,
        "action_oracle_method": "independent_sorted_rate_CDF_inversion_from_each_logged_action_uniform",
        "action_oracle_required_agreement_fraction": 1.0,
        "action_oracle_check_denominator": action_oracle_check_count,
        "action_oracle_mismatch_count": action_oracle_mismatch_count,
        "action_oracle_agreement_fraction": (
            0.0
            if action_oracle_check_count == 0
            else 1.0 - action_oracle_mismatch_count / action_oracle_check_count
        ),
        "zero_action_oracle_failure_count": zero_action_oracle_failure_count,
        "trajectory_kernel_replay_denominator": trajectory_count,
        "trajectory_kernel_replay_mismatch_count": trajectory_kernel_replay_mismatch_count,
        "sampled_validity_failure_count": sampled_validity_failures,
        "empirical_distribution_role": "diagnostic_only_not_a_gate_condition",
        "empirical_distribution_can_grant_pass": False,
        "registered_exact_grid_gate_thresholds": {
            "finest_pair_total_variation": 0.03,
            "finest_pair_mean_edit_count_difference": 0.05,
            "finest_pair_max_termination_fraction_difference": 0.03,
        },
        "pooled_terminal_expected_distribution": _distribution_record(
            pooled_expected_terminal
        ),
        "pooled_terminal_observed_distribution": _m29_terminal_count_records(
            pooled_terminal_counts, trajectory_count
        ),
        "pooled_terminal_law_comparison": pooled_terminal_comparison,
        "pooled_first_action_expected_distribution": [
            {"action_key": key, "probability": probability}
            for key, probability in sorted(pooled_expected_action.items())
        ],
        "pooled_first_action_observed_distribution": _m29_action_count_records(
            pooled_action_counts, trajectory_count
        ),
        "pooled_first_action_law_comparison": pooled_action_comparison,
        "strata": stratum_records,
        "input_stream_sha256": input_stream.hexdigest(),
        "actual_terminal_stream_sha256": terminal_stream.hexdigest(),
        "actual_first_action_stream_sha256": first_action_stream.hexdigest(),
        "step_kernel_oracle_stream_sha256": step_kernel_oracle_stream.hexdigest(),
        "action_oracle_stream_sha256": action_oracle_stream.hexdigest(),
        "expected_law_stream_sha256": expected_stream_sha256,
        "audit_binding_sha256": hashlib.sha256(
            canonical_json_bytes(audit_hash_payload)
        ).hexdigest(),
        "failure_count": failure_count,
        "failure_denominator": failure_denominator,
    }


def run_sampler_audit() -> dict[str, Any]:
    validity_failures = 0
    budget_violations = 0
    replay_failures = 0
    termination_counts: dict[str, int] = {}
    primary_input_stream = hashlib.sha256()
    primary_tiny_count = 0
    primary_real_length_count = 0
    primary_real_length_histogram: dict[str, int] = defaultdict(int)
    input_rng = random.Random(SEED + 25)
    real_lengths = (16, 32, 64, 128, 256)
    for offset in range(4096):
        if offset < 2048:
            length = 1 + offset % 3
            sequence = "".join(input_rng.choice(ALPHABET) for _ in range(length))
            maximum = 6
            primary_tiny_count += 1
        else:
            length = real_lengths[(offset - 2048) % len(real_lengths)]
            sequence = "".join(input_rng.choice(ALPHABET) for _ in range(length))
            maximum = 300
            primary_real_length_count += 1
            primary_real_length_histogram[str(length)] += 1
        initial = EditState.initial(
            sequence,
            budget=1 + offset % 3,
            region="5UTR" if offset % 2 == 0 else "3UTR",
            target_condition=("increase", "decrease", "maintain", "interval")[
                offset % 4
            ],
        )
        primary_input_stream.update(canonical_json_bytes(initial.inference_dict()))
        result = constrained_single_event_first_order(
            initial,
            factorized_rate_fn,
            step_size=0.05,
            stability_hazard=0.05,
            min_length=1,
            max_length=maximum,
            seed=SEED + offset,
        )
        valid = (
            1 <= len(result.final_state.current) <= maximum
            and result.final_state.history.executed <= initial.initial_budget
            and result.final_state.remaining_budget
            == initial.initial_budget - result.final_state.history.executed
            and result.edit_events == result.final_state.history.executed
            and result.final_state.phase == Phase.HALTED
            and result.final_state.source == initial.source
        )
        validity_failures += int(not valid)
        budget_violations += int(
            result.final_state.history.executed > initial.initial_budget
        )
        if offset < 1024:
            replay_failures += int(
                not replay_constrained_result(
                    result,
                    factorized_rate_fn,
                    min_length=1,
                    max_length=maximum,
                )
            )
        reason = result.final_state.termination_reason.value
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    require(
        primary_tiny_count == primary_real_length_count == 2048,
        "M25 tiny/real stratum drift",
    )
    require(
        set(primary_real_length_histogram) == {str(value) for value in real_lengths}
        and sum(primary_real_length_histogram.values()) == 2048,
        "M25 real-length coverage drift",
    )
    require(
        validity_failures == budget_violations == replay_failures == 0,
        "primary sampler validity/replay failed",
    )

    def high_rates(state: EditState, _time: float) -> Mapping[AtomicAction, float]:
        return {
            action: 100.0
            for action in enumerate_legal_actions(state, min_length=1, max_length=6)
        }

    paper_invalid = 0
    paper_validity_failures = 0
    paper_replay_failures = 0
    paper_input_stream = hashlib.sha256()
    paper_input_rng = random.Random(SEED + 26)
    paper_length_histogram: dict[str, int] = defaultdict(int)
    for offset in range(4096):
        length = 1 + offset % 3
        sequence = "".join(paper_input_rng.choice(ALPHABET) for _ in range(length))
        paper_initial = EditState.initial(
            sequence,
            budget=1 + offset % 3,
            region="5UTR" if offset % 2 == 0 else "3UTR",
        )
        paper_input_stream.update(canonical_json_bytes(paper_initial.inference_dict()))
        paper_length_histogram[str(length)] += 1
        result = paper_first_order_parallel(
            paper_initial,
            high_rates,
            step_size=0.1,
            min_length=1,
            max_length=6,
            seed=SEED + offset,
            horizon=0.1,
        )
        paper_invalid += int(result.invalid_joint_proposals > 0)
        paper_validity_failures += int(
            result.final_state.phase != Phase.HALTED
            or not 1 <= len(result.final_state.current) <= 6
            or result.final_state.history.executed > paper_initial.initial_budget
            or result.edit_events != result.final_state.history.executed
            or result.final_state.source != paper_initial.source
        )
        if offset < 256:
            paper_replay_failures += int(
                not replay_paper_result(result, high_rates, min_length=1, max_length=6)
            )
    require(
        paper_validity_failures == paper_replay_failures == 0,
        "paper sampler validity/replay failed",
    )

    step_sizes = M29_STEP_SIZES
    reference_distribution = _rk4_integrated_hazard_reference()
    reference_mass_error = abs(math.fsum(reference_distribution.values()) - 1.0)
    actual_sampler_law = _run_m29_actual_sampler_law_audit()
    grid_distributions: list[dict[tuple[int, str, str], float]] = []
    grid_records: list[dict[str, Any]] = []
    sampled_validity_failures = int(
        actual_sampler_law["sampled_validity_failure_count"]
    )
    reference_errors: list[float] = []
    for stratum_index, step_size in enumerate(step_sizes):
        distribution = _frozen_grid_distribution(step_size)
        grid_distributions.append(distribution)
        error = _total_variation(distribution, reference_distribution)
        reference_errors.append(error)
        actual_stratum = actual_sampler_law["strata"][stratum_index]
        require(
            actual_stratum["step_size"] == step_size,
            "M29 actual sampler stratum order drift",
        )
        grid_records.append(
            {
                "step_size": step_size,
                "exact_frozen_grid_distribution": _distribution_record(distribution),
                "probability_mass_error": abs(math.fsum(distribution.values()) - 1.0),
                "total_variation_to_rk4_reference": error,
                "mean_edit_count": _mean_edit_count(distribution),
                "termination_fractions": _termination_fractions(distribution),
                "empirical_trajectory_count": M29_TRAJECTORIES_PER_STRATUM,
                "empirical_distribution": actual_stratum[
                    "terminal_observed_distribution"
                ],
                "actual_terminal_law_comparison": actual_stratum[
                    "terminal_law_comparison"
                ],
                "actual_first_action_law_comparison": actual_stratum[
                    "first_action_law_comparison"
                ],
                "action_oracle_check_denominator": actual_stratum[
                    "action_oracle_check_denominator"
                ],
                "action_oracle_mismatch_count": actual_stratum[
                    "action_oracle_mismatch_count"
                ],
            }
        )
    finest_left = grid_distributions[-2]
    finest_right = grid_distributions[-1]
    finest_pair_total_variation = _total_variation(finest_left, finest_right)
    finest_pair_mean_edit_count_difference = abs(
        _mean_edit_count(finest_left) - _mean_edit_count(finest_right)
    )
    left_termination = _termination_fractions(finest_left)
    right_termination = _termination_fractions(finest_right)
    finest_pair_max_termination_fraction_difference = max(
        abs(left_termination.get(key, 0.0) - right_termination.get(key, 0.0))
        for key in set(left_termination) | set(right_termination)
    )
    reference_nonincrease_failures = sum(
        later > earlier + 1.0e-12
        for earlier, later in zip(reference_errors, reference_errors[1:])
    )
    grid_mass_failures = sum(
        record["probability_mass_error"] > 1.0e-12 for record in grid_records
    ) + int(reference_mass_error > 1.0e-10)
    convergence_failure_count = (
        int(actual_sampler_law["failure_count"])
        + reference_nonincrease_failures
        + grid_mass_failures
        + int(finest_pair_total_variation > 0.03)
        + int(finest_pair_mean_edit_count_difference > 0.05)
        + int(finest_pair_max_termination_fraction_difference > 0.03)
    )
    convergence_statistic_count = int(actual_sampler_law["failure_denominator"]) + 11
    require(
        convergence_failure_count == 0,
        "time/state-dependent step-halving convergence failed",
    )
    return {
        "schema_version": "mk0_sampler_convergence_v2",
        "status": "PASS",
        "primary_sampler": "constrained_single_event_first_order",
        "primary_exact_gillespie": False,
        "primary_trajectory_count": 4096,
        "primary_randomized_tiny_trajectory_count": primary_tiny_count,
        "primary_randomized_real_length_trajectory_count": primary_real_length_count,
        "primary_real_length_histogram": dict(primary_real_length_histogram),
        "primary_input_state_stream_sha256": primary_input_stream.hexdigest(),
        "primary_validity_failure_count": validity_failures,
        "budget_violation_count": budget_violations,
        "primary_replay_count": 1024,
        "primary_replay_failure_count": replay_failures,
        "primary_termination_distribution": termination_counts,
        "paper_sampler": "paper_first_order_parallel",
        "paper_exact_gillespie": False,
        "paper_trajectory_count": 4096,
        "paper_randomized_tiny_length_histogram": dict(paper_length_histogram),
        "paper_input_state_stream_sha256": paper_input_stream.hexdigest(),
        "paper_hard_validity_failure_count": paper_validity_failures,
        "paper_invalid_joint_proposal_trajectory_count": paper_invalid,
        "paper_replay_count": 256,
        "paper_replay_failure_count": paper_replay_failures,
        "paper_validity_reported_separately": True,
        "step_halving_trajectory_count": int(
            actual_sampler_law["trajectory_denominator"]
        ),
        "step_halving_rate_domain": "time_varying_and_post_edit_state_dependent",
        "step_halving": grid_records,
        "step_halving_actual_sampler_kernel_binding": actual_sampler_law,
        "reference_method": "independent_RK4_forward_equation_integrated_hazard_reference",
        "reference_distribution": _distribution_record(reference_distribution),
        "reference_probability_mass_error": reference_mass_error,
        "reference_total_variation_errors": reference_errors,
        "reference_error_nonincrease_failure_count": reference_nonincrease_failures,
        "sampled_convergence_validity_failure_count": sampled_validity_failures,
        "finest_pair_total_variation": finest_pair_total_variation,
        "finest_pair_mean_edit_count_difference": finest_pair_mean_edit_count_difference,
        "finest_pair_max_termination_fraction_difference": finest_pair_max_termination_fraction_difference,
        "convergence_statistic_count": convergence_statistic_count,
        "convergence_failure_count": convergence_failure_count,
    }


def run_critic_audit() -> dict[str, Any]:
    failures = 0
    query_log_stream = hashlib.sha256()
    audited_rate_fn = stop_only(8.0)
    closure = inspect.getclosurevars(audited_rate_fn)
    closure_forbidden_names = sorted(
        name
        for name in set(closure.nonlocals) | set(closure.globals)
        if any(
            token in name.lower()
            for token in (
                "critic",
                "evaluator",
                "guidance",
                "reward",
                "rerank",
                "selector",
                "score",
            )
        )
    )
    require(
        not closure_forbidden_names,
        "formal no-critic rate function closes over a prohibited role",
    )
    for offset in range(256):
        _result, audit = base_generation_without_critic(
            EditState.initial("AC", budget=2),
            audited_rate_fn,
            step_size=0.05,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            seed=SEED + offset,
        )
        failures += int(
            audit.critic_present
            or audit.guidance_queries
            or audit.final_evaluator_queries
        )
        query_log_stream.update(
            canonical_json_bytes(
                {
                    "trajectory_index": offset,
                    "critic_present": audit.critic_present,
                    "guidance_queries": audit.guidance_queries,
                    "final_evaluator_queries": audit.final_evaluator_queries,
                }
            )
        )

    interfaces = {
        "constrained_single_event_first_order": constrained_single_event_first_order,
        "paper_first_order_parallel": paper_first_order_parallel,
        "base_generation_without_critic": base_generation_without_critic,
        "FoundationFusionRateField.forward": FoundationFusionRateField.forward,
        "OfficialPaperRateAdapter.__call__": OfficialPaperRateAdapter.__call__,
        "enumerate_action_rates": enumerate_action_rates,
        "edit_flow_loss": edit_flow_loss,
        "build_target_transition_oracle": build_target_transition_oracle,
    }
    forbidden_parameter_tokens = (
        "critic",
        "evaluator",
        "guidance",
        "reward",
        "rerank",
        "selector",
        "score",
    )
    interface_records: list[dict[str, Any]] = []
    interface_failures = 0
    for name, interface in interfaces.items():
        signature = inspect.signature(interface)
        prohibited_parameters = sorted(
            parameter
            for parameter in signature.parameters
            if any(token in parameter.lower() for token in forbidden_parameter_tokens)
        )
        source = inspect.getsource(interface)
        interface_failures += int(bool(prohibited_parameters))
        interface_records.append(
            {
                "interface": name,
                "parameters": list(signature.parameters),
                "prohibited_parameters": prohibited_parameters,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )

    guidance_rejections = 0
    for _ in range(3):
        try:
            reject_final_evaluator_as_guidance(lambda _x: 1.0, as_guidance=True)
        except PermissionError:
            guidance_rejections += 1
    forbidden_keyword_rejections = 0
    forbidden_keywords = (
        "critic",
        "evaluator",
        "final_evaluator",
        "guidance",
        "reward",
        "reranker",
        "selector",
        "score_fn",
    )
    for forbidden_keyword in forbidden_keywords:
        try:
            base_generation_without_critic(
                EditState.initial("AC", budget=1),
                audited_rate_fn,
                step_size=0.05,
                stability_hazard=0.05,
                min_length=1,
                max_length=6,
                seed=SEED,
                **{forbidden_keyword: object()},
            )
        except PermissionError:
            forbidden_keyword_rejections += 1
    require(
        failures == interface_failures == 0
        and guidance_rejections == 3
        and forbidden_keyword_rejections == len(forbidden_keywords),
        "critic/final evaluator boundary failed",
    )
    m34_failure_count = (
        failures
        + interface_failures
        + len(closure_forbidden_names)
        + (3 - guidance_rejections)
        + (len(forbidden_keywords) - forbidden_keyword_rejections)
    )
    m34_sample_count = 256 + len(interface_records) + 3 + len(forbidden_keywords)
    return {
        "schema_version": "mk0_critic_role_audit_v1",
        "status": "PASS",
        "base_generation_count": 256,
        "base_generation_failure_count": failures,
        "critic_present": False,
        "guidance_query_count": 0,
        "final_evaluator_query_count": 0,
        "final_evaluator_guidance_injection_count": 3,
        "final_evaluator_guidance_rejection_count": guidance_rejections,
        "forbidden_keyword_injection_count": len(forbidden_keywords),
        "forbidden_keyword_rejection_count": forbidden_keyword_rejections,
        "audited_query_log_count": 256,
        "query_log_stream_sha256": query_log_stream.hexdigest(),
        "audited_generator_rate_interface_count": len(interface_records),
        "generator_rate_interfaces": interface_records,
        "interface_failure_count": interface_failures,
        "formal_rate_fn_closure_nonlocals": sorted(closure.nonlocals),
        "formal_rate_fn_closure_forbidden_names": closure_forbidden_names,
        "m34_sample_count": m34_sample_count,
        "m34_failure_count": m34_failure_count,
    }


def _validate_bound_pytest_report(
    report: Mapping[str, Any],
    *,
    output_dir: Path,
    launcher_path: Path,
) -> None:
    """Re-verify the launcher's fail-closed test-domain evidence."""

    require(
        report.get("schema_version") == "mk0_bound_pytest_report_v2",
        "bound pytest report schema drift",
    )
    require(
        report.get("status") == "PASS"
        and report.get("returncode") == 0
        and report.get("collection_returncode") == 0
        and report.get("pytest_returncode") == 0
        and report.get("execution_started") is True,
        "bound pytest did not complete both formal phases",
    )
    require(report.get("pytest_args") == ["tests/mk0"], "pytest domain drift")
    require(
        report.get("repo_root") == str(REPO_ROOT)
        and report.get("formal_output_root") == str(output_dir),
        "pytest repository/output binding drift",
    )
    require(
        isinstance(report.get("pytest_version"), str)
        and bool(report.get("pytest_version")),
        "pytest version binding is absent",
    )

    collection_nodeids = report.get("collection_nodeids")
    execution_nodeids = report.get("execution_nodeids")
    require(
        isinstance(collection_nodeids, list)
        and collection_nodeids
        and collection_nodeids == sorted(collection_nodeids)
        and len(collection_nodeids) == len(set(collection_nodeids))
        and execution_nodeids == collection_nodeids,
        "pytest collect/execute nodeid inventory drift",
    )
    nodeid_sha256 = hashlib.sha256(
        "".join(f"{nodeid}\n" for nodeid in collection_nodeids).encode("utf-8")
    ).hexdigest()
    require(
        report.get("collection_nodeids_sha256") == nodeid_sha256
        and report.get("execution_nodeids_sha256") == nodeid_sha256,
        "pytest nodeid inventory digest drift",
    )
    collected = report.get("collected_count")
    require(
        isinstance(collected, int)
        and not isinstance(collected, bool)
        and collected == len(collection_nodeids)
        and report.get("executed_count") == collected
        and report.get("passed_count") == collected,
        "pytest requires collected == executed == passed > 0",
    )
    for field in (
        "failed_count",
        "error_count",
        "skipped_count",
        "deselected_count",
        "xfailed_count",
        "xpassed_count",
    ):
        require(report.get(field) == 0, f"pytest formal {field} is nonzero")
    require(report.get("contract_violations") == [], "pytest contract violation")

    environment = report.get("environment_contract")
    require(isinstance(environment, Mapping), "pytest environment contract is absent")
    require(
        environment.get("pytest_plugin_autoload_disabled") is True
        and environment.get("pythonpath_replaced_with_external_binding") is True,
        "pytest environment was not sanitized",
    )
    controlled = environment.get("controlled_environment_keys")
    require(
        isinstance(controlled, list)
        and {
            "MK0_EXPECTED_PACKAGE_INIT",
            "MK0_EXPECTED_PACKAGE_ROOT",
            "MK0_PYTEST_MODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
        }
        == set(controlled),
        "pytest controlled environment inventory drift",
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
            f"pytest sanitized environment inventory drift: {field}",
        )

    helper_path = REPO_ROOT / "scripts" / "mk0" / "strict_worktree_import.py"
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
            and origin.get("strict_importer_path") == str(helper_path)
            and origin.get("strict_importer_sha256") == sha256_file(helper_path)
            and origin.get("strict_importer_loaded_from_source_bytes") is True,
            f"pytest source-byte import binding drift: {field}",
        )
    isolation = report.get("import_isolation")
    require(
        isinstance(isolation, Mapping)
        and isolation.get("inside_formal_output_tree") is False
        and isolation.get("external_import_root_removed") is True
        and isolation.get("ambient_pythonpath_replaced") is True,
        "pytest import isolation contaminated the formal run tree",
    )

    junit = report.get("junit")
    log = report.get("log")
    require(
        isinstance(junit, Mapping) and isinstance(log, Mapping),
        "pytest evidence absent",
    )
    junit_path = output_dir / "evaluation" / "pytest_mk0.junit.xml"
    log_path = output_dir / "logs" / "pytest_mk0.log"
    require(
        Path(str(junit.get("path", ""))).resolve(strict=True) == junit_path
        and junit.get("exists") is True
        and junit.get("sha256") == sha256_file(junit_path),
        "pytest JUnit binding drift",
    )
    require(
        junit.get("totals")
        == {
            "tests": collected,
            "errors": 0,
            "failures": 0,
            "skipped": 0,
            "passed": collected,
        },
        "pytest JUnit totals drift",
    )
    require(
        Path(str(log.get("path", ""))).resolve(strict=True) == log_path
        and log.get("sha256") == sha256_file(log_path),
        "pytest log binding drift",
    )
    require(
        report.get("formal_output_tree_regular_only") is True,
        "pytest regular-tree certificate is absent",
    )
    require(
        launcher_path.is_file() and not launcher_path.is_symlink(),
        "pytest launcher is not a regular source file",
    )


def run_pytest(output_dir: Path) -> dict[str, Any]:
    launcher_path = REPO_ROOT / "scripts" / "mk0" / "run_bound_pytest.py"
    launcher_source = launcher_path.read_bytes()
    launcher_namespace = {
        "__file__": str(launcher_path),
        "__name__": "mk0_bound_pytest_launcher",
    }
    exec(
        compile(
            launcher_source,
            str(launcher_path),
            "exec",
            dont_inherit=True,
            optimize=0,
        ),
        launcher_namespace,
    )
    report = launcher_namespace["run_bound_pytest"](
        repo_root=REPO_ROOT,
        formal_output_root=output_dir,
        pytest_args=["tests/mk0"],
        junit_path=Path("evaluation/pytest_mk0.junit.xml"),
        log_path=Path("logs/pytest_mk0.log"),
        report_path=Path("provenance/pytest_import_binding.json"),
        python_executable=sys.executable,
    )
    _validate_bound_pytest_report(
        report,
        output_dir=output_dir,
        launcher_path=launcher_path,
    )
    persisted_report = read_json_object(
        output_dir / "provenance" / "pytest_import_binding.json"
    )
    require(persisted_report == report, "persisted pytest report drift")
    return {
        **report,
        "launcher_path": str(launcher_path),
        "launcher_sha256": hashlib.sha256(launcher_source).hexdigest(),
        "report_path": str(output_dir / "provenance" / "pytest_import_binding.json"),
        "report_sha256": sha256_file(
            output_dir / "provenance" / "pytest_import_binding.json"
        ),
        "log_path": report["log"]["path"],
        "log_sha256": report["log"]["sha256"],
        "junit_path": report["junit"]["path"],
        "junit_sha256": report["junit"]["sha256"],
    }


def run_text_audit(trajectory_direction: str, trajectory_sha256: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "mk0_exact_claim_audit", REPO_ROOT / "scripts" / "mk0" / "audit_exact_claims.py"
    )
    if spec is None or spec.loader is None:
        raise AcceptanceFailure("cannot load exact-claim auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exact_report = module.audit(REPO_ROOT)
    require(exact_report["pass"], "unsupported affirmative exact-sampling claim")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "math" / "math_kernel_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_direction = "source_at_0_to_target_at_1"
    require(
        re.fullmatch(r"[0-9a-f]{64}", trajectory_sha256) is not None,
        "M08 representative trajectory hash is invalid",
    )
    try:
        tracked_output = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-files",
                "-z",
                "--",
                "configs/math",
                "schemas",
                "docs/math",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        relevant_paths = [
            REPO_ROOT / item.decode("utf-8")
            for item in tracked_output.split(b"\0")
            if item
        ]
        discovery = "git_ls_files_scoped_to_configs_math_schemas_docs_math"
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        relevant_paths = sorted(
            path
            for root in (
                REPO_ROOT / "configs" / "math",
                REPO_ROOT / "schemas",
                REPO_ROOT / "docs" / "math",
            )
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
        )
        discovery = "recursive_fallback_no_git_metadata"
    relevant_paths = sorted(set(relevant_paths))
    require(relevant_paths, "M08 direction-audit file universe is empty")

    reverse_patterns = (
        re.compile(r"target_at_0_to_source_at_1", re.I),
        re.compile(
            r"target.{0,48}(?:t\s*=\s*0|at[_ -]?0).{0,96}"
            r"source.{0,48}(?:t\s*=\s*1|at[_ -]?1)",
            re.I | re.S,
        ),
    )
    reverse_hits: list[dict[str, Any]] = []
    structured_bindings: list[dict[str, Any]] = []
    canonical_occurrence_count = 0
    inventory: list[dict[str, Any]] = []

    def walk_direction_bindings(value: Any, path: str = "$") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}"
                if key == "time_direction":
                    observed = item.get("const") if isinstance(item, dict) else item
                    structured_bindings.append(
                        {
                            "json_path": child,
                            "observed": observed,
                            "pass": observed == expected_direction,
                        }
                    )
                walk_direction_bindings(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk_direction_bindings(item, f"{path}[{index}]")

    for path in relevant_paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        raw = path.read_bytes()
        require(
            b"\0" not in raw[:8192], f"binary file entered M08 universe: {relative}"
        )
        text = raw.decode("utf-8")
        inventory.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        canonical_occurrence_count += text.count(expected_direction)
        for pattern in reverse_patterns:
            for match in pattern.finditer(text):
                reverse_hits.append(
                    {
                        "path": relative,
                        "offset": match.start(),
                        "match": match.group(0)[:160],
                    }
                )
        if path.suffix in {".json", ".yaml", ".yml"}:
            structured = (
                json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
            )
            before = len(structured_bindings)
            walk_direction_bindings(structured)
            for binding in structured_bindings[before:]:
                binding["path"] = relative

    inventory_sha256 = hashlib.sha256(canonical_json_bytes(inventory)).hexdigest()
    endpoint_checks = {
        "cubic_source_endpoint": cubic_schedule(0.0)[0] == 0.0,
        "cubic_target_endpoint": cubic_schedule(1.0)[0] == 1.0,
        "linear_source_endpoint": linear_schedule(0.0)[0] == 0.0,
        "linear_target_endpoint": linear_schedule(1.0)[0] == 1.0,
    }
    structured_binding_failures = sum(
        not binding["pass"] for binding in structured_bindings
    )
    missing_required_binding_count = max(0, 3 - len(structured_bindings))
    missing_canonical_occurrence_count = int(canonical_occurrence_count < 3)
    trajectory_failure_count = int(trajectory_direction != expected_direction)
    endpoint_failure_count = sum(not value for value in endpoint_checks.values())
    direction_error_count = (
        len(reverse_hits)
        + structured_binding_failures
        + missing_required_binding_count
        + missing_canonical_occurrence_count
        + trajectory_failure_count
        + endpoint_failure_count
    )
    direction_check_count = (
        2 * len(relevant_paths)
        + len(structured_bindings)
        + canonical_occurrence_count
        + 1
        + len(endpoint_checks)
    )
    direction_checks = {
        "config": config["time_direction"] == expected_direction,
        "trajectory_fixture": trajectory_failure_count == 0,
        "no_reverse_direction_text": not reverse_hits,
        "all_structured_bindings_canonical": structured_binding_failures == 0,
        "required_structured_bindings_present": missing_required_binding_count == 0,
        "required_canonical_occurrences_present": missing_canonical_occurrence_count
        == 0,
        **endpoint_checks,
    }
    require(direction_check_count > 0, "M08 direction check count is zero")
    require(direction_error_count == 0, "time direction audit failed")
    return {
        "schema_version": "mk0_text_contract_audit_v1",
        "status": "PASS",
        "time_direction_checks": direction_checks,
        "time_direction_check_count": direction_check_count,
        "time_direction_error_count": direction_error_count,
        "file_universe_discovery": discovery,
        "audited_file_count": len(relevant_paths),
        "audited_file_inventory_sha256": inventory_sha256,
        "audited_file_inventory": inventory,
        "canonical_direction_occurrence_count": canonical_occurrence_count,
        "structured_time_direction_bindings": structured_bindings,
        "structured_binding_failure_count": structured_binding_failures,
        "reverse_direction_hits": reverse_hits,
        "trajectory_time_direction": trajectory_direction,
        "representative_trajectory_sha256": trajectory_sha256,
        "exact_sampling_claim_audit": exact_report,
    }


def _runtime_gate_binding(
    gate_config: Mapping[str, Any],
    *,
    sample_count: int,
    failure_count: int,
    failure_denominator: int,
    artifact_sha256: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Create and immediately self-validate one observed CPU gate binding."""

    binding = {
        "gate_id": gate_config["id"],
        "name": gate_config["name"],
        "passed": failure_count == 0,
        "test_domain": gate_config["domain"],
        "exhaustive_or_sampled": gate_config["coverage"],
        "sample_count": sample_count,
        "dtype": str(gate_config["dtype"]),
        "atol": gate_config["atol"],
        "rtol": gate_config["rtol"],
        "seed": int(gate_config["seed"]),
        "failure_count": failure_count,
        "failure_denominator": failure_denominator,
        "artifact_path": gate_config["artifact_path"],
        "artifact_sha256": artifact_sha256,
        "metrics": dict(metrics),
    }
    gate_result_from_runtime_binding(
        binding, gate_config, actual_artifact_sha256=artifact_sha256
    )
    return binding


def build_cpu_gate_bindings(
    config: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Bind CPU gates to observed counters in their immutable support files."""

    schema = reports["mk0_schema_action_audit.json"]
    coupling = reports["coupling_manifest.json"]
    hazard = reports["hazard_audit.json"]
    transition = reports["transition_aggregation_oracle.json"]
    loss = reports["loss_oracle_report.json"]
    stop = reports["stop_audit.json"]
    survival = reports["stop_survival_oracle.json"]
    sampler = reports["sampler_convergence.json"]
    critic = reports["critic_role_audit.json"]
    text = reports["mk0_text_contract_audit.json"]
    exact = text["exact_sampling_claim_audit"]
    joint = coupling["joint_product_switch_clock_oracle"]

    specs: dict[str, dict[str, Any]] = {
        "M01": {
            "sample_count": int(schema["runtime_validation_count"]),
            "failure_count": int(schema["schema_validation_failure_count"]),
            "failure_denominator": int(schema["runtime_validation_count"]),
            "metrics": {
                "schema_document_count": len(schema["schema_documents"]),
                "active_state_record_count": schema["active_state_record_count"],
                "halted_state_record_count": schema["halted_state_record_count"],
                "action_record_count": schema["action_record_count"],
                "termination_record_count": schema["termination_record_count"],
                "runtime_record_stream_sha256": schema["runtime_record_stream_sha256"],
            },
        },
        "M02": {
            "sample_count": int(
                schema["action_record_count"] + schema["real_length_property_count"]
            ),
            "failure_count": int(
                schema["tiny_exactness_failure_count"]
                + schema["real_length_property_failure_count"]
            ),
            "failure_denominator": int(
                schema["action_record_count"] + schema["real_length_property_count"]
            ),
            "metrics": {
                "tiny_action_failure_count": schema["tiny_exactness_failure_count"],
                "real_length_failure_count": schema[
                    "real_length_property_failure_count"
                ],
            },
        },
        "M03": {
            "sample_count": int(coupling["record_count"]),
            "failure_count": int(coupling["failure_count"]),
            "failure_denominator": int(coupling["record_count"]),
            "metrics": {
                "alignment_reconstruction_failure_count": coupling["failure_count"]
            },
        },
        "M04": {
            "sample_count": int(coupling["record_count"]),
            "failure_count": int(coupling["path_is_observed_true_count"]),
            "failure_denominator": int(coupling["record_count"]),
            "metrics": {
                "path_is_observed_true_count": coupling["path_is_observed_true_count"]
            },
        },
        "M06": {
            "sample_count": int(joint["sample_count"]),
            "failure_count": int(joint["failure_count"]),
            "failure_denominator": int(joint["sample_count"]),
            "metrics": {
                "changed_coordinate_count": joint["changed_coordinate_count"],
                "expected_switch_probability": joint["expected_switch_probability"],
                "maximum_path_probability_error": joint[
                    "maximum_path_probability_error"
                ],
                "maximum_normalization_error": joint["maximum_normalization_error"],
                "max_marginal_error": joint["max_marginal_error"],
                "max_off_diagonal_covariance": joint["max_off_diagonal_covariance"],
                "clock_sampler_binding_failure_count": joint[
                    "clock_sampler_binding_failure_count"
                ],
                "group_consistency_failure_count": joint[
                    "group_consistency_failure_count"
                ],
            },
        },
        "M07": {
            "sample_count": int(hazard["rho_checks"]),
            "failure_count": int(hazard["rho_failure_count"]),
            "failure_denominator": int(hazard["rho_checks"]),
            "metrics": {"rho_failure_count": hazard["rho_failure_count"]},
        },
        "M08": {
            "sample_count": int(text["time_direction_check_count"]),
            "failure_count": int(text["time_direction_error_count"]),
            "failure_denominator": int(text["time_direction_check_count"]),
            "metrics": {
                "time_direction_checks": text["time_direction_checks"],
                "audited_file_count": text["audited_file_count"],
                "audited_file_inventory_sha256": text["audited_file_inventory_sha256"],
                "canonical_direction_occurrence_count": text[
                    "canonical_direction_occurrence_count"
                ],
                "structured_binding_failure_count": text[
                    "structured_binding_failure_count"
                ],
                "reverse_direction_hit_count": len(text["reverse_direction_hits"]),
            },
        },
        "M09": {
            "sample_count": int(
                hazard["schedule_derivative_grid_checks"]
                + hazard["schedule_endpoint_checks"]
            ),
            "failure_count": int(
                hazard["schedule_derivative_failure_count"]
                + hazard["schedule_endpoint_failure_count"]
            ),
            "failure_denominator": int(
                hazard["schedule_derivative_grid_checks"]
                + hazard["schedule_endpoint_checks"]
            ),
            "metrics": {
                "derivative_failure_count": hazard["schedule_derivative_failure_count"],
                "endpoint_failure_count": hazard["schedule_endpoint_failure_count"],
                "endpoint_nonfinite_count": hazard["schedule_endpoint_nonfinite_count"],
                "endpoint_clip_hit_count": hazard["endpoint_clip_hit_count"],
                "maximum_derivative_absolute_error": hazard[
                    "maximum_derivative_absolute_error"
                ],
                "finite_difference_stencil": hazard["finite_difference_stencil"],
                "finite_difference_step": hazard["finite_difference_step"],
                "origin_forward_step": hazard["origin_forward_step"],
                "actual_atol": hazard["actual_atol"],
                "actual_rtol": hazard["actual_rtol"],
                "grid_stream_sha256": hazard["grid_stream_sha256"],
            },
        },
        "M10": {
            "sample_count": int(hazard["negative_rate_checks"]),
            "failure_count": int(hazard["negative_rate_count"]),
            "failure_denominator": int(hazard["evaluated_action_rate_count"]),
            "metrics": {
                "negative_rate_count": hazard["negative_rate_count"],
                "evaluated_action_rate_count": hazard["evaluated_action_rate_count"],
                "tiny_extended_run_state_count": hazard[
                    "tiny_source_current_budget_run_state_count"
                ],
                "tiny_active_run_state_count": hazard["tiny_active_run_state_count"],
                "tiny_halted_run_state_count": hazard["tiny_halted_run_state_count"],
                "random_real_length_state_count": hazard[
                    "random_real_length_state_count"
                ],
                "random_real_lengths": hazard["random_real_lengths"],
            },
        },
        "M11": {
            "sample_count": int(hazard["masked_illegal_rate_checks"]),
            "failure_count": int(hazard["masked_illegal_rate_failure_count"]),
            "failure_denominator": int(hazard["masked_illegal_rate_checks"]),
            "metrics": {
                key: hazard[key]
                for key in sorted(hazard)
                if key.startswith("masked_illegal_")
            },
        },
        "M12": {
            "sample_count": int(hazard["zero_instantaneous_hazard_checks"]),
            "failure_count": int(hazard["zero_instantaneous_hazard_failure_count"]),
            "failure_denominator": int(hazard["zero_instantaneous_hazard_checks"]),
            "metrics": {
                key: hazard[key]
                for key in sorted(hazard)
                if key.startswith("zero_instantaneous_hazard_")
            },
        },
        "M13": {
            "sample_count": int(hazard["zero_remaining_integrated_hazard_checks"]),
            "failure_count": int(
                hazard["zero_remaining_integrated_hazard_failure_count"]
            ),
            "failure_denominator": int(
                hazard["zero_remaining_integrated_hazard_checks"]
            ),
            "metrics": {
                key: hazard[key]
                for key in sorted(hazard)
                if key.startswith("zero_remaining_integrated_hazard_")
                and key != "zero_remaining_integrated_hazard_fixture_records"
            },
        },
        "M14": {
            "sample_count": int(hazard["conditioned_distribution_checks"]),
            "failure_count": int(hazard["conditioned_distribution_failure_count"]),
            "failure_denominator": int(hazard["conditioned_distribution_checks"]),
            "metrics": {
                key: hazard[key]
                for key in sorted(hazard)
                if key.startswith("conditioned_")
            },
        },
        "M15": {
            "sample_count": int(hazard["factorization_checks"]),
            "failure_count": int(hazard["factorization_failure_count"]),
            "failure_denominator": int(hazard["factorization_checks"]),
            "metrics": {
                "factorization_failure_count": hazard["factorization_failure_count"]
            },
        },
        "M16": {
            "sample_count": int(hazard["generator_checks"]),
            "failure_count": int(hazard["generator_failure_count"]),
            "failure_denominator": int(hazard["generator_checks"]),
            "metrics": {
                "generator_row_sum_failure_count": hazard["generator_failure_count"]
            },
        },
        "M17": {
            "sample_count": int(transition["sample_count"]),
            "failure_count": int(transition["failure_count"]),
            "failure_denominator": int(transition["sample_count"]),
            "metrics": {
                key: value
                for key, value in transition.items()
                if key
                not in {
                    "records",
                    "examples",
                    "schema_version",
                    "status",
                    "parameter_vectors",
                    "canonical_oracles",
                    "multiplicity_oracle",
                    "collision_oracle",
                }
            },
        },
        "M18": {
            "sample_count": int(loss["loss_oracle_count"]),
            "failure_count": int(loss["loss_failure_count"]),
            "failure_denominator": int(loss["loss_oracle_count"]),
            "metrics": {
                "loss_failure_count": loss["loss_failure_count"],
                "maximum_absolute_loss_error": loss.get(
                    "maximum_absolute_loss_error", 0.0
                ),
            },
        },
        "M19": {
            "sample_count": int(loss["finite_loss_and_gradient_sample_count"]),
            "failure_count": int(loss["finite_loss_and_gradient_failure_count"]),
            "failure_denominator": int(loss["finite_loss_and_gradient_sample_count"]),
            "metrics": {
                "finite_loss_failure_count": loss["finite_loss_failure_count"],
                "finite_gradient_failure_count": loss["finite_gradient_failure_count"],
                "finite_difference_gradient_case_count": loss[
                    "finite_difference_gradient_case_count"
                ],
                "finite_difference_coordinate_count": loss[
                    "finite_difference_coordinate_count"
                ],
                "finite_difference_epsilon": loss["finite_difference_epsilon"],
                "finite_difference_atol": loss["finite_difference_atol"],
                "finite_difference_rtol": loss["finite_difference_rtol"],
                "maximum_finite_difference_gradient_error": loss[
                    "maximum_finite_difference_gradient_error"
                ],
                "analytic_gradient_failure_count": loss[
                    "analytic_gradient_failure_count"
                ],
                "gradient_fixture_stream_sha256": loss[
                    "gradient_fixture_stream_sha256"
                ],
                "halted_exact_zero_count": loss["halted_exact_zero_count"],
                "halted_exact_zero_failure_count": loss[
                    "halted_exact_zero_failure_count"
                ],
                "incomplete_neighbourhood_rejections": loss[
                    "incomplete_neighbourhood_rejections"
                ],
                "oracle_state_binding_rejections": loss[
                    "oracle_state_binding_rejections"
                ],
            },
        },
        "M20": {
            "sample_count": int(survival["absolute_hazard_checks"]),
            "failure_count": int(survival["absolute_hazard_failure_count"]),
            "failure_denominator": int(survival["absolute_hazard_checks"]),
            "metrics": {
                "absolute_hazard_failure_count": survival[
                    "absolute_hazard_failure_count"
                ]
            },
        },
        "M21": {
            "sample_count": int(survival["dwell_sample_count"]),
            "failure_count": int(survival["dwell_oracle_failure_count"]),
            "failure_denominator": int(survival["dwell_sample_count"]),
            "metrics": {
                "dwell_oracle_failure_count": survival["dwell_oracle_failure_count"],
                "nonpositive_dwell_count": survival["nonpositive_dwell_count"],
                "strictly_positive_dwell_count": survival[
                    "strictly_positive_dwell_count"
                ],
                "structural_independence_check_count": survival[
                    "structural_independence_check_count"
                ],
                "structural_independence_failure_count": survival[
                    "structural_independence_failure_count"
                ],
                "analytic_inverse_check_count": survival[
                    "analytic_inverse_check_count"
                ],
                "analytic_inverse_failure_count": survival[
                    "analytic_inverse_failure_count"
                ],
                "dwell_gamma_values": survival["dwell_gamma_values"],
                "dwell_samples_per_gamma": survival["dwell_samples_per_gamma"],
                "completion_clock_variant_count": survival[
                    "completion_clock_variant_count"
                ],
                "dwell_analytic_atol": survival["dwell_analytic_atol"],
                "dwell_analytic_rtol": survival["dwell_analytic_rtol"],
                "independence_oracle": survival["independence_oracle"],
                "empirical_pearson_fisher_role": survival[
                    "empirical_pearson_fisher_role"
                ],
                "independence_claim_boundary": survival["independence_claim_boundary"],
                "gamma_dwell_independence": survival["gamma_dwell_independence"],
                "dwell_sample_stream_sha256": survival["dwell_sample_stream_sha256"],
            },
        },
        "M22": {
            "sample_count": int(survival["dwell_sample_count"]),
            "failure_count": int(survival["event_fraction_failure_count"]),
            "failure_denominator": int(survival["event_fraction_check_count"]),
            "metrics": {
                "event_fraction_check_count": survival["event_fraction_check_count"],
                "event_fraction_tolerance": survival["event_fraction_tolerance"],
                "gamma_event_fraction": survival["gamma_event_fraction"],
                "gamma_sensitivity": survival["gamma_sensitivity"],
                "gamma_sensitivity_role": survival["gamma_sensitivity_role"],
                "event_fraction_failure_count": survival[
                    "event_fraction_failure_count"
                ],
                "gamma_sensitivity_failure_count": survival[
                    "gamma_sensitivity_failure_count"
                ],
            },
        },
        "M23": {
            "sample_count": int(stop["halted_edit_flow_checks"]),
            "failure_count": int(stop["halted_edit_flow_failure_count"]),
            "failure_denominator": int(stop["halted_edit_flow_checks"]),
            "metrics": {
                "halted_edit_flow_failure_count": stop["halted_edit_flow_failure_count"]
            },
        },
        "M24": {
            "sample_count": int(stop["termination_separation_checks"]),
            "failure_count": int(stop["termination_separation_failure_count"]),
            "failure_denominator": int(stop["termination_separation_checks"]),
            "metrics": {
                "termination_separation_failure_count": stop[
                    "termination_separation_failure_count"
                ],
                "termination_reasons": stop["termination_reasons"],
                "termination_reason_counts": stop["termination_reason_counts"],
                "termination_reason_unique_operational_fixture_counts": stop[
                    "termination_reason_unique_operational_fixture_counts"
                ],
                "termination_operational_id_excludes_context": stop[
                    "termination_operational_id_excludes_context"
                ],
                "termination_record_stream_sha256": stop[
                    "termination_record_stream_sha256"
                ],
            },
        },
        "M25": {
            "sample_count": int(sampler["primary_trajectory_count"]),
            "failure_count": int(sampler["primary_validity_failure_count"]),
            "failure_denominator": int(sampler["primary_trajectory_count"]),
            "metrics": {
                "primary_validity_failure_count": sampler[
                    "primary_validity_failure_count"
                ],
                "primary_termination_distribution": sampler[
                    "primary_termination_distribution"
                ],
            },
        },
        "M26": {
            "sample_count": int(sampler["paper_trajectory_count"]),
            "failure_count": int(not sampler["paper_validity_reported_separately"]),
            "failure_denominator": int(sampler["paper_trajectory_count"]),
            "metrics": {
                "paper_invalid_joint_proposal_trajectory_count": sampler[
                    "paper_invalid_joint_proposal_trajectory_count"
                ],
                "paper_validity_reported_separately": sampler[
                    "paper_validity_reported_separately"
                ],
            },
        },
        "M27": {
            "sample_count": int(sampler["primary_trajectory_count"]),
            "failure_count": int(sampler["budget_violation_count"]),
            "failure_denominator": int(sampler["primary_trajectory_count"]),
            "metrics": {"budget_violation_count": sampler["budget_violation_count"]},
        },
        "M28": {
            "sample_count": int(sampler["primary_replay_count"]),
            "failure_count": int(sampler["primary_replay_failure_count"]),
            "failure_denominator": int(sampler["primary_replay_count"]),
            "metrics": {
                key: value
                for key, value in sampler.items()
                if key.startswith("primary_replay_")
            },
        },
        "M29": {
            "sample_count": int(sampler["step_halving_trajectory_count"]),
            "failure_count": int(sampler["convergence_failure_count"]),
            "failure_denominator": int(sampler["convergence_statistic_count"]),
            "metrics": {
                key: value
                for key, value in sampler.items()
                if key.startswith("step_halving")
                or key.startswith("finest_pair")
                or key.startswith("reference_")
                or key == "convergence_failure_count"
            },
        },
        "M30": {
            "sample_count": int(exact["files_audited"]),
            "failure_count": int(
                exact["unsupported_affirmative_claim_count"]
                + len(exact["structural_false_binding_failures"])
            ),
            "failure_denominator": int(exact["files_audited"]),
            "metrics": {
                "claim_hits_reviewed": exact["claim_hits_reviewed"],
                "unsupported_affirmative_claim_count": exact[
                    "unsupported_affirmative_claim_count"
                ],
                "structural_false_binding_failure_count": len(
                    exact["structural_false_binding_failures"]
                ),
            },
        },
        "M33": {
            "sample_count": int(critic["base_generation_count"]),
            "failure_count": int(critic["base_generation_failure_count"]),
            "failure_denominator": int(critic["base_generation_count"]),
            "metrics": {
                "base_generation_failure_count": critic[
                    "base_generation_failure_count"
                ],
                "critic_present": critic["critic_present"],
                "guidance_query_count": critic["guidance_query_count"],
                "final_evaluator_query_count": critic["final_evaluator_query_count"],
            },
        },
        "M34": {
            "sample_count": int(critic["m34_sample_count"]),
            "failure_count": int(critic["m34_failure_count"]),
            "failure_denominator": int(critic["m34_sample_count"]),
            "metrics": {
                "guidance_query_count": critic["guidance_query_count"],
                "final_evaluator_query_count": critic["final_evaluator_query_count"],
                "injection_count": critic["final_evaluator_guidance_injection_count"],
                "rejection_count": critic["final_evaluator_guidance_rejection_count"],
                "forbidden_keyword_injection_count": critic[
                    "forbidden_keyword_injection_count"
                ],
                "forbidden_keyword_rejection_count": critic[
                    "forbidden_keyword_rejection_count"
                ],
                "audited_query_log_count": critic["audited_query_log_count"],
                "query_log_stream_sha256": critic["query_log_stream_sha256"],
                "audited_generator_rate_interface_count": critic[
                    "audited_generator_rate_interface_count"
                ],
                "interface_failure_count": critic["interface_failure_count"],
                "formal_rate_fn_closure_forbidden_names": critic[
                    "formal_rate_fn_closure_forbidden_names"
                ],
            },
        },
    }
    gpu_ids = {"M05", "M31", "M32", "M35"}
    configured = {gate["id"]: gate for gate in config["acceptance"]["gates"]}
    require(set(specs) == set(configured) - gpu_ids, "CPU gate binding coverage drift")
    bindings: dict[str, dict[str, Any]] = {}
    for gate_id in sorted(specs):
        spec = specs[gate_id]
        relative = Path(configured[gate_id]["artifact_path"])
        name = relative.name
        require(name in artifact_hashes, f"missing CPU support digest for {gate_id}")
        bindings[gate_id] = _runtime_gate_binding(
            configured[gate_id],
            sample_count=spec["sample_count"],
            failure_count=spec["failure_count"],
            failure_denominator=spec["failure_denominator"],
            artifact_sha256=artifact_hashes[name],
            metrics=spec["metrics"],
        )
    return bindings


def _write_bytes_new(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _command_record(
    argv: list[str], *, environment: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "argv": argv,
        "shell_escaped": " ".join(
            [*(f"{key}={shlex.quote(value)}" for key, value in environment.items())]
            + [shlex.join(argv)]
        ),
        "environment": dict(environment),
        "cwd": str(REPO_ROOT),
    }


def initialize_formal_run(
    *,
    output_dir: Path,
    run_id: str,
    parent_run_binding: Mapping[str, Any] | None,
    goal_sha256: str,
    implementation_commit: str,
    preflight_path: Path,
    preflight_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    config_path: Path,
    fm0_root: Path,
    d1_data: Path,
    d1_ledger: Path,
    foundation_snapshot: Path,
    gpu_uuid: str,
) -> tuple[str, dict[str, Any]]:
    """Create the immutable registration manifest and the section-19 tree."""

    output_dir.mkdir(parents=True)
    write_json_exclusive_atomic(
        output_dir / ".mk0_run_owner.json",
        {
            "schema_version": "mk0_run_owner_v1",
            "run_id": run_id,
            "creator_pid": os.getpid(),
            "created_at_utc": utc_now(),
        },
    )
    create_contract_tree(output_dir)
    update_status(
        output_dir,
        run_id=run_id,
        state="PREFLIGHT_PASSED",
        terminal=False,
        stop_reason="FORMAL_CPU_ACCEPTANCE_REGISTERED",
    )
    preflight_report = read_json_object(preflight_path)
    upstream = preflight_report["upstream"]
    require(
        Path(upstream["fm0_closure_root"]).resolve(strict=True)
        == fm0_root.resolve(strict=True),
        "preflight/FM0 root drift during run registration",
    )
    for prefix, path in (
        ("d1_canonical_records", d1_data),
        ("d1_exposure_ledger", d1_ledger),
    ):
        require(
            Path(upstream[f"{prefix}_path"]).resolve(strict=True)
            == path.resolve(strict=True),
            f"preflight {prefix} path drift during run registration",
        )
        require(
            upstream[f"{prefix}_sha256"] == sha256_file(path),
            f"preflight {prefix} hash drift during run registration",
        )

    data_binding_path = fm0_root / "provenance" / "data_binding.json"
    checkpoint_manifest_path = fm0_root / "evaluation" / "hash_license_manifest.json"
    data_binding = read_json_object(data_binding_path)
    checkpoint_manifest = read_json_object(checkpoint_manifest_path)
    require(
        Path(str(checkpoint_manifest.get("snapshot_dir", ""))).resolve(strict=True)
        == foundation_snapshot.resolve(strict=True),
        "foundation snapshot differs from FM0 manifest during registration",
    )
    require(
        checkpoint_manifest.get("license", {}).get("type", "").lower() == "agpl-3.0",
        "foundation license is not FM0-bound AGPL-3.0",
    )

    provenance_payloads = {
        "data_manifest.json": {
            "schema_version": "mk0_data_manifest_v1",
            "d1_canonical_records": {
                "path": str(d1_data.resolve(strict=True)),
                "size_bytes": d1_data.stat().st_size,
                "sha256": sha256_file(d1_data),
            },
            "fm0_data_binding": {
                "path": str(data_binding_path.resolve(strict=True)),
                "sha256": sha256_file(data_binding_path),
            },
            "final_labels_accessed": False,
        },
        "split_manifest.json": {
            "schema_version": "mk0_split_manifest_binding_v1",
            "mk0_split_semantics": "NO_TRAIN_VALIDATION_TEST_SPLIT_MK0_E0_ONLY",
            "fm0_split_manifests": data_binding.get("split_manifests", {}),
            "final_labels_accessed": False,
        },
        "foundation_manifest.json": {
            "schema_version": "mk0_foundation_manifest_binding_v1",
            "fm0_hash_license_manifest": {
                "path": str(checkpoint_manifest_path.resolve(strict=True)),
                "sha256": sha256_file(checkpoint_manifest_path),
            },
            "model_id": checkpoint_manifest.get("model_id"),
            "revision": checkpoint_manifest.get("revision"),
            "snapshot_dir": str(foundation_snapshot.resolve(strict=True)),
            "files": checkpoint_manifest.get("files"),
            "license": checkpoint_manifest.get("license"),
        },
        "exposure_ledger.json": {
            "schema_version": "mk0_exposure_ledger_binding_v1",
            "path": str(d1_ledger.resolve(strict=True)),
            "size_bytes": d1_ledger.stat().st_size,
            "sha256": sha256_file(d1_ledger),
            "exact_foundation_sequence_overlap": "NOT_AVAILABLE_NOT_ASSERTED",
            "final_labels_accessed": False,
        },
        "code_manifest.json": {
            "schema_version": "mk0_code_manifest_v1",
            "implementation_commit": implementation_commit,
            "source_binding": source_binding,
            "formal_worktree_clean": True,
        },
    }
    provenance_bindings: dict[str, dict[str, Any]] = {}
    for name, payload in provenance_payloads.items():
        path = output_dir / "provenance" / name
        digest = write_new(path, payload)
        provenance_bindings[name] = {"path": str(path), "sha256": digest}
    goal_contract_sidecar = output_dir / "provenance" / "goal_contract.sha256"
    goal_contract_sidecar_sha256 = _write_bytes_new(
        goal_contract_sidecar,
        (
            f"{goal_sha256}  USER_DECLARED_SOLE_CONTRACT:"
            "mrna_latest_build_contract_first.md\n"
        ).encode("utf-8"),
    )
    provenance_bindings["goal_contract.sha256"] = {
        "path": str(goal_contract_sidecar),
        "sha256": goal_contract_sidecar_sha256,
    }

    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_sha256 = _write_bytes_new(
        resolved_config_path, config_path.read_bytes()
    )
    diff_sha256 = hashlib.sha256(b"").hexdigest()
    _write_bytes_new(
        output_dir / "git" / "commit.txt", f"{implementation_commit}\n".encode()
    )
    _write_bytes_new(output_dir / "git" / "diff.patch", b"")
    _write_bytes_new(output_dir / "git" / "diff.sha256", f"{diff_sha256}\n".encode())

    checkpoint_na = output_dir / "checkpoints" / "NOT_APPLICABLE.json"
    checkpoint_na_sha256 = write_new(
        checkpoint_na,
        {
            "schema_version": "mk0_checkpoint_not_applicable_v1",
            "reason": "MK0 performs mathematical acceptance plus a tiny GPU optimizer update, not model training.",
            "last_healthy_checkpoint": "MK0_NOT_APPLICABLE",
            "best_primary_checkpoint": "MK0_NOT_APPLICABLE",
            "resume_semantics": "MK0_NOT_APPLICABLE",
        },
    )
    _write_bytes_new(
        output_dir / "checkpoints" / "checksums.sha256",
        f"{checkpoint_na_sha256}  NOT_APPLICABLE.json\n".encode(),
    )

    cpu_command = _command_record([sys.executable, *sys.argv], environment={})
    gpu_argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "mk0" / "run_mk0_gpu_smoke.py"),
        "--output-dir",
        str(output_dir / "artifacts" / "mk0"),
        "--run-id",
        run_id,
        "--goal-sha256",
        goal_sha256,
        "--implementation-commit",
        implementation_commit,
        "--run-manifest",
        str(output_dir / "run_manifest.json"),
        "--preflight-record",
        str(preflight_path.resolve(strict=True)),
        "--snapshot-dir",
        str(foundation_snapshot.resolve(strict=True)),
        "--device",
        "cuda:0",
    ]
    gpu_command = _command_record(
        gpu_argv, environment={"CUDA_VISIBLE_DEVICES": gpu_uuid}
    )
    finalizer_argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "mk0" / "finalize_mk0_acceptance.py"),
        "--run-root",
        str(output_dir),
        "--run-id",
        run_id,
        "--goal-sha256",
        goal_sha256,
        "--implementation-commit",
        implementation_commit,
        "--fm0-closure-root",
        str(fm0_root.resolve(strict=True)),
        "--d1-data",
        str(d1_data.resolve(strict=True)),
        "--d1-ledger",
        str(d1_ledger.resolve(strict=True)),
        "--preflight-record",
        str(preflight_path.resolve(strict=True)),
    ]
    if parent_run_binding is not None:
        finalizer_argv.extend(["--parent-run-id", str(parent_run_binding["run_id"])])
    finalizer_command = _command_record(finalizer_argv, environment={})
    commands = {
        "cpu_acceptance": cpu_command,
        "gpu_smoke": gpu_command,
        "finalizer": finalizer_command,
    }
    _write_bytes_new(
        output_dir / "command.txt",
        "\n".join(
            f"{name}: {record['shell_escaped']}" for name, record in commands.items()
        ).encode("utf-8")
        + b"\n",
    )

    start_time = utc_now()
    manifest = {
        "schema_version": "mk0_run_manifest_v3",
        "run_id": run_id,
        "task_id": "MK0-01",
        "parent_run_id": (
            None if parent_run_binding is None else parent_run_binding["run_id"]
        ),
        "parent_run_binding": (
            None if parent_run_binding is None else dict(parent_run_binding)
        ),
        "phase": "MK0",
        "hypotheses": {
            hypothesis: "NOT_TESTED_AT_MK0_E0"
            for hypothesis in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
        },
        "evidence_level": EVIDENCE_LEVEL,
        "contract": {
            "sha256": goal_sha256,
            "section": "31A",
            "scope": "MK0_ONLY",
        },
        "code": {
            "commit": implementation_commit,
            "diff_sha256": diff_sha256,
            "clean_worktree_required": True,
        },
        "exact_commands": commands,
        "resolved_config": {
            "path": str(resolved_config_path),
            "sha256": resolved_config_sha256,
        },
        "provenance": provenance_bindings,
        "data": provenance_payloads["data_manifest.json"],
        "split": provenance_payloads["split_manifest.json"],
        "foundation": provenance_payloads["foundation_manifest.json"],
        "exposure_ledger": provenance_payloads["exposure_ledger.json"],
        "seed": SEED,
        "gpu_uuid": gpu_uuid,
        "timing": {"start_utc": start_time, "end_utc": None},
        "process_identity": {
            "cpu_pid": os.getpid(),
            "gpu_pid": None,
            "finalizer_pid": None,
            "tmux_pane": os.environ.get("TMUX_PANE"),
            "job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID"),
        },
        "exit_code": None,
        "stop_reason": "RUNNING_PENDING_GPU_AND_FINALIZER",
        "artifact_checksums": {
            "state": "PENDING_UNTIL_TERMINAL_COMPLETION",
            "completion_manifest_path": str(
                output_dir / "summary" / "run_completion_manifest.json"
            ),
            "whole_run_ledger_path": str(output_dir / "artifact_checksums.sha256"),
        },
        "paper_eligibility": {
            "eligible": False,
            "reason": "MK0 E0 mathematical and engineering evidence only",
        },
        "known_deviations": [
            "NO_MODEL_CHECKPOINT_MK0_NOT_A_TRAINING_RUN",
            "NO_SCIENTIFIC_H1_H8_RESULT_AT_E0",
            "ENVIRONMENT_LOCK_DRIFT_RECORDED_NOT_SILENTLY_MUTATED",
        ],
        "goal_sha256": goal_sha256,
        "implementation_commit": implementation_commit,
        "source_binding": source_binding,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preflight": dict(preflight_binding),
        "run_root": str(output_dir),
        "large_artifact_policy": "small canonical MK0 run bundle under contract /home path; large upstream data and foundation snapshot remain externally bound and are not copied",
        "cpu_scope": "symbolic_exhaustive_numerical_oracle_non_neural_only",
        "gpu_scope": "separate fail_closed_real_utrlm_runner",
        "final_labels_accessed": False,
        "downstream_stage_started": False,
    }
    manifest_sha256 = write_new(output_dir / "run_manifest.json", manifest)
    append_text(
        output_dir / "logs" / "stdout.log",
        f"{start_time} MK0 formal run registered; CPU acceptance starting\n",
    )
    append_jsonl(
        output_dir / "logs" / "system_metrics.jsonl",
        {
            "created_at_utc": start_time,
            "event": "PREFLIGHT_BINDING",
            "preflight": dict(preflight_binding),
            "gpu_uuid_reserved_for_formal_smoke": gpu_uuid,
        },
    )
    append_event(
        output_dir,
        "CPU_ACCEPTANCE_STARTED",
        run_id=run_id,
        pid=os.getpid(),
        run_manifest_sha256=manifest_sha256,
    )
    return manifest_sha256, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--goal-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--preflight-record", type=Path, required=True)
    parser.add_argument("--fm0-closure-root", type=Path, required=True)
    parser.add_argument("--d1-data", type=Path, required=True)
    parser.add_argument("--d1-ledger", type=Path, required=True)
    parser.add_argument("--foundation-snapshot", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_preexisting = output_dir.exists()
    try:
        require(
            re.fullmatch(r"[0-9a-f]{64}", args.goal_sha256) is not None,
            "invalid Goal hash",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit) is not None,
            "invalid implementation commit",
        )
        run_id_match = FORMAL_RUN_ID.fullmatch(args.run_id)
        require(run_id_match is not None, "formal run ID violates section 19.2")
        require(
            run_id_match.group("model") == "utrlm"
            and run_id_match.group("dataset") == "mathkernel"
            and run_id_match.group("split") == "tiny",
            "formal run ID model/dataset/split semantics drift",
        )
        require(
            run_id_match.group("short_sha")
            == args.implementation_commit[: len(run_id_match.group("short_sha"))],
            "formal run ID short SHA differs from implementation commit",
        )
        require(
            int(run_id_match.group("seed")) == SEED,
            "formal run ID seed differs from frozen seed",
        )
        try:
            run_id_time = datetime.strptime(
                run_id_match.group("utc"), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError as error:
            raise AcceptanceFailure(
                "formal run ID UTC is not a calendar time"
            ) from error
        require(
            run_id_time.strftime("%Y%m%dT%H%M%SZ") == run_id_match.group("utc"),
            "formal run ID UTC is not canonical",
        )
        parent_run_binding = validate_parent_run_lineage(
            args.run_id,
            args.parent_run_id,
            goal_sha256=args.goal_sha256,
        )
        require(output_dir.name == args.run_id, "formal run root basename drift")
        require(
            output_dir.parent == CANONICAL_RUN_PARENT,
            "formal run root violates section 19.1 canonical parent",
        )
        require(
            re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", args.gpu_uuid) is not None,
            "planned GPU UUID is invalid",
        )
        require(
            Path.cwd().resolve(strict=True) == REPO_ROOT,
            "formal CPU acceptance must be launched from the implementation worktree",
        )
        if output_dir.exists():
            manifest_path = output_dir / "run_manifest.json"
            if manifest_path.is_file():
                existing_manifest = read_json_object(manifest_path)
                bound_run_id = existing_manifest.get("run_id")
                bound_root = Path(str(existing_manifest.get("run_root", ""))).resolve(
                    strict=True
                )
                require(
                    isinstance(bound_run_id, str) and bound_root == output_dir,
                    "pre-existing formal root is not self-bound by its manifest",
                )
                terminal_before = (output_dir / "FAILED").exists() or (
                    output_dir / "DONE"
                ).exists()
                terminal = resume_failure_closure_if_present(
                    output_dir, run_id=bound_run_id
                )
                if terminal is not None:
                    requested_matches = args.run_id == bound_run_id
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
                                "failure_closure_resumed": not terminal_before,
                                "tree_mutated_only_to_finish_failure_closure": (
                                    not terminal_before and terminal == "FAILED"
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                    if not requested_matches:
                        return 2
                    return 0 if terminal == "DONE" else 1
        require(
            not output_dir.exists(),
            "output directory already exists; refusing overwrite",
        )
        config_path = REPO_ROOT / "configs" / "math" / "math_kernel_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(
            config["contract"]["sha256"] == args.goal_sha256,
            "Goal/config hash mismatch",
        )
        source_binding = git_source_binding(args.implementation_commit)
        preflight_binding = validate_preflight_binding(
            args.preflight_record,
            run_id=args.run_id,
            parent_run_id=args.parent_run_id,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
        )
        preflight_report = read_json_object(args.preflight_record.resolve(strict=True))
        try:
            preflight_time = datetime.fromisoformat(
                str(preflight_report["observed_at_utc"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError) as error:
            raise AcceptanceFailure("preflight UTC is invalid") from error
        require(
            preflight_time.tzinfo is not None and run_id_time <= preflight_time,
            "formal run ID UTC must not follow its preflight",
        )
        require(
            any(
                gpu.get("uuid") == args.gpu_uuid
                for gpu in preflight_report.get("resources", {}).get("gpus", [])
            ),
            "planned GPU UUID was absent from the formal preflight",
        )
        run_manifest_sha256, _ = initialize_formal_run(
            output_dir=output_dir,
            run_id=args.run_id,
            parent_run_binding=parent_run_binding,
            goal_sha256=args.goal_sha256,
            implementation_commit=args.implementation_commit,
            preflight_path=args.preflight_record,
            preflight_binding=preflight_binding,
            source_binding=source_binding,
            config_path=config_path,
            fm0_root=args.fm0_closure_root.resolve(strict=True),
            d1_data=args.d1_data.resolve(strict=True),
            d1_ledger=args.d1_ledger.resolve(strict=True),
            foundation_snapshot=args.foundation_snapshot.resolve(strict=True),
            gpu_uuid=args.gpu_uuid,
        )
        artifact_dir = output_dir / "artifacts" / "mk0"
        artifact_dir.mkdir(parents=True)
        provenance_dir = output_dir / "provenance"
        write_new(
            provenance_dir / "cpu_command.json",
            {
                "schema_version": "mk0_cpu_command_v1",
                "run_id": args.run_id,
                "created_at_utc": utc_now(),
                "argv": [sys.executable, *sys.argv],
                "environment": {},
                "pid": os.getpid(),
                "cwd": str(REPO_ROOT),
                "python_executable": sys.executable,
                "python_version": sys.version,
                "neural_forward_allowed": False,
            },
        )
        write_new(
            output_dir / "logs" / "cpu_event_start.json",
            {
                "run_id": args.run_id,
                "event": "CPU_ACCEPTANCE_STARTED",
                "created_at_utc": utc_now(),
            },
        )
        rng = random.Random(SEED)
        np.random.seed(SEED)
        pytest_report = run_pytest(output_dir)
        schema_report, exact_action_count = run_schema_action_audit(rng)
        require(exact_action_count == 7760, "M02 sample count mismatch")
        coupling_report = run_coupling_audit(rng)
        hazard_report = run_hazard_audit(rng)
        transition_report, loss_report = run_transition_loss_audits(rng)
        stop_report, stop_survival_report = run_stop_audits(rng)
        sampler_report = run_sampler_audit()
        critic_report = run_critic_audit()
        text_report = run_text_audit(
            str(schema_report["representative_trajectory_time_direction"]),
            str(schema_report["representative_trajectory_sha256"]),
        )

        reports = {
            "mk0_schema_action_audit.json": schema_report,
            "coupling_manifest.json": coupling_report,
            "hazard_audit.json": hazard_report,
            "transition_aggregation_oracle.json": transition_report,
            "loss_oracle_report.json": loss_report,
            "stop_audit.json": stop_report,
            "stop_survival_oracle.json": stop_survival_report,
            "sampler_convergence.json": sampler_report,
            "critic_role_audit.json": critic_report,
            "mk0_text_contract_audit.json": text_report,
        }
        artifact_hashes = {
            name: write_new(artifact_dir / name, report)
            for name, report in reports.items()
        }
        cpu_gate_bindings = build_cpu_gate_bindings(config, reports, artifact_hashes)
        source_binding_after = git_source_binding(args.implementation_commit)
        require(
            source_binding_after == source_binding,
            "tracked source changed during CPU acceptance",
        )
        pending_gpu_gates = ["M05", "M31", "M32", "M35"]
        failed_gate_ids = [
            gate_id
            for gate_id, binding in cpu_gate_bindings.items()
            if not binding["passed"]
        ]
        cpu_status = (
            "PASS_CPU_GATES_PENDING_GPU"
            if not failed_gate_ids
            else "FAILED_WITH_EVIDENCE"
        )
        cpu_gate_report = {
            "schema_version": "mk0_cpu_gate_results_v2",
            "run_id": args.run_id,
            "status": cpu_status,
            "evidence_level": "E0_MATH_ENGINEERING_ONLY",
            "created_at_utc": utc_now(),
            "goal_sha256": args.goal_sha256,
            "implementation_commit": args.implementation_commit,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "run_manifest_path": "run_manifest.json",
            "run_manifest_sha256": run_manifest_sha256,
            "preflight": preflight_binding,
            "source_binding": source_binding_after,
            "python": sys.version,
            "numpy": np.__version__,
            "pytest": pytest_report,
            "gate_bindings": cpu_gate_bindings,
            "pending_gpu_gate_ids": pending_gpu_gates,
            "failed_gate_ids": failed_gate_ids,
            "artifact_hashes": artifact_hashes,
            "scientific_claims": {
                "functional_improvement": False,
                "matched_budget_superiority": False,
                "paper_success": False,
            },
        }
        cpu_gate_hash = write_new(
            artifact_dir / "mk0_cpu_gate_results.json", cpu_gate_report
        )
        summary = {
            "schema_version": "mk0_cpu_acceptance_summary_v2",
            "run_id": args.run_id,
            "status": cpu_status,
            "evidence_level": "E0_MATH_ENGINEERING_ONLY",
            "created_at_utc": utc_now(),
            "goal_sha256": args.goal_sha256,
            "implementation_commit": args.implementation_commit,
            "run_root": str(output_dir),
            "run_manifest": {
                "path": str(output_dir / "run_manifest.json"),
                "sha256": run_manifest_sha256,
            },
            "preflight": preflight_binding,
            "source_binding": source_binding_after,
            "source_binding_sha256": hashlib.sha256(
                canonical_json_bytes(source_binding_after)
            ).hexdigest(),
            "cpu_gate_results": {
                "path": str(artifact_dir / "mk0_cpu_gate_results.json"),
                "sha256": cpu_gate_hash,
            },
            "cpu_gate_results_sha256": cpu_gate_hash,
            "artifact_count": len(artifact_hashes) + 1,
            "artifact_hashes": {
                **artifact_hashes,
                "mk0_cpu_gate_results.json": cpu_gate_hash,
            },
            "failed_gate_ids": failed_gate_ids,
            "pending_gpu_gate_ids": pending_gpu_gates,
        }
        write_new(output_dir / "summary" / "cpu_acceptance_summary.json", summary)
        append_jsonl(
            output_dir / "logs" / "metrics.jsonl",
            {
                "created_at_utc": utc_now(),
                "stage": "CPU_ACCEPTANCE",
                "status": cpu_status,
                "passed_gate_count": len(cpu_gate_bindings) - len(failed_gate_ids),
                "failed_gate_ids": failed_gate_ids,
                "pending_gpu_gate_ids": pending_gpu_gates,
                "pytest_passed": pytest_report["passed_count"],
            },
        )
        require(not failed_gate_ids, f"CPU gates failed: {failed_gate_ids}")
        update_status(
            output_dir,
            run_id=args.run_id,
            state="CPU_VERIFIED_PENDING_GPU",
            terminal=False,
            stop_reason="CPU_GATES_PASSED_GPU_NOT_STARTED",
        )
        append_text(
            output_dir / "logs" / "stdout.log",
            f"{utc_now()} CPU acceptance passed; GPU smoke pending\n",
        )
        return 0
    except BaseException as error:
        owned_output = False
        owner_path = output_dir / ".mk0_run_owner.json"
        if not output_preexisting and owner_path.is_file():
            try:
                owner = read_json_object(owner_path)
                owned_output = (
                    owner.get("run_id") == args.run_id
                    and owner.get("creator_pid") == os.getpid()
                )
            except BaseException:
                owned_output = False
        if owned_output:
            failure_dir = output_dir / "failure"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "mk0_cpu_acceptance_failure_v1",
                "run_id": args.run_id,
                "status": "FAILED_WITH_EVIDENCE",
                "created_at_utc": utc_now(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            }
            target = failure_dir / "cpu_acceptance_failure.json"
            if not target.exists():
                write_new(target, failure)
            if (output_dir / "logs" / "stderr.log").is_file():
                append_text(
                    output_dir / "logs" / "stderr.log",
                    f"{utc_now()} CPU acceptance failed: {error}\n",
                )
            if (output_dir / "logs" / "events.jsonl").is_file():
                try:
                    write_failed_sentinel(
                        output_dir,
                        run_id=args.run_id,
                        stage="CPU_ACCEPTANCE",
                        reason=failure_reason(error),
                        exit_code=1,
                    )
                except BaseException as closure_error:
                    print(
                        f"MK0 CPU failure closure also failed: {closure_error}",
                        file=sys.stderr,
                    )
        print(f"MK0 CPU acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
