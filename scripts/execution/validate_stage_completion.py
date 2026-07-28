#!/usr/bin/env python3
"""Validate a D1/B0 completion manifest against evidence and remote Git state."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
import yaml

try:
    from scripts.execution.acceptance_semantics import validate_phase_acceptance
    from scripts.execution.validate_registry import (
        CANONICAL_GITHUB_URL,
        _canonical_remote_identity,
        _remote_ref_oid,
        validate as validate_registry,
    )
    from scripts.execution.validate_stage_manifest import (
        validate as validate_stage_manifest,
    )
except ModuleNotFoundError:  # direct script execution from scripts/execution
    from acceptance_semantics import validate_phase_acceptance
    from validate_registry import (
        CANONICAL_GITHUB_URL,
        _canonical_remote_identity,
        _remote_ref_oid,
        validate as validate_registry,
    )
    from validate_stage_manifest import validate as validate_stage_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = SOURCE_ROOT / "schemas" / "stage_completion_manifest.schema.json"
GOAL_SHA256 = "c3dc5875868d847b8519fee40b14c43b65e4c5948dc5c3b98101ca61a5671dd5"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
STAGE_RE = re.compile(
    r"^D1_B0_[0-9]{8}T[0-9]{6}Z_" r"(?P<base_short_sha>[0-9a-f]{7})(?:_A[0-9]+)?$"
)
MINIMUM_INDEPENDENT_GATE_ID = 26
REQUIRED_ARTIFACT_ROLES = {
    "code_manifest": "code",
    "d1_acceptance": "d1_freeze",
    "b0_acceptance": "evidence",
    "preflight_manifest": "d1_freeze",
    "protected_state": "d1_freeze",
    "independent_gate_review": "evidence",
    "task_registry": "registry",
    "decision_log": "registry",
    "protection_recheck": "registry",
}
STAGE_SCOPED_ROLES = {
    "code_manifest",
    "d1_acceptance",
    "b0_acceptance",
    "preflight_manifest",
    "protected_state",
    "protection_recheck",
}
STAGE_ROLE_SUFFIXES = {
    "code_manifest": "release/code_manifest.json",
    "d1_acceptance": "D1/acceptance.json",
    "b0_acceptance": "B0/acceptance.json",
    "preflight_manifest": "preflight_manifest.json",
    "protected_state": "protected_state.json",
    "protection_recheck": "release/protection_recheck.json",
}
FIXED_ROLE_PATHS = {
    "task_registry": "docs/execution/task_registry_v2.yaml",
    "decision_log": "docs/decision_log.md",
    "independent_gate_review": (
        "docs/audits/2026-07-29-d1-b0-independent-gate-review.md"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(
    repo: Path, *args: str, check: bool = False
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(repo), *args]
    try:
        return subprocess.run(
            command,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=b"",
            stderr=str(exc).encode("utf-8", errors="replace"),
        )


def _git_text(repo: Path, *args: str) -> str | None:
    result = _git(repo, *args)
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None


def _git_object_exists(repo: Path, commit: str) -> bool:
    return (
        bool(COMMIT_RE.fullmatch(commit))
        and _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0
    )


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        _git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0
    )


def _blob_bytes(repo: Path, commit: str, relative_path: str) -> bytes | None:
    result = _git(repo, "show", f"{commit}:{relative_path}")
    return result.stdout if result.returncode == 0 else None


def _blob_sha256(repo: Path, commit: str, relative_path: str) -> str | None:
    value = _blob_bytes(repo, commit, relative_path)
    return hashlib.sha256(value).hexdigest() if value is not None else None


def _live_dirty_diff_sha256(repo: Path) -> str | None:
    """Hash the exact tracked worktree/index delta against HEAD as raw bytes."""
    result = _git(repo, "diff", "--binary", "HEAD", "--")
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _valid_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _resolve_repo_path(repo: Path, value: Any) -> tuple[Path, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("path is empty")
    raw = Path(text)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("path must be repository-relative without parent traversal")
    root = repo.resolve(strict=True)
    lexical = root / raw
    cursor = root
    for part in raw.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlinked artifact paths are forbidden")
    candidate = lexical.resolve(strict=True)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes git.repository") from exc
    return candidate, relative


def _resolve_manifest_path(repo: Path, value: Path) -> tuple[Path, str]:
    """Resolve a CLI manifest path while preserving the repository boundary."""
    root = repo.resolve(strict=True)
    raw = Path(value)
    if raw.is_absolute():
        try:
            relative = raw.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes git.repository") from exc
    else:
        relative = raw
    return _resolve_repo_path(root, relative.as_posix())


def _load_json(path: Path, label: str, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"{label}: invalid UTF-8 JSON")
        return None


def _validate_schema(manifest: Mapping[str, Any], errors: list[str]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        format_checker = jsonschema.FormatChecker()
        # ``date-time`` support is otherwise optional in jsonschema and can
        # silently disappear when ``rfc3339-validator`` is not installed.
        # Register the contract's timezone-aware parser on this checker so
        # production validation has identical fail-closed semantics in every
        # supported runtime.
        format_checker.checks("date-time")(lambda value: _valid_time(value) is not None)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=format_checker
        )
    except Exception as exc:  # missing/old jsonschema is a hard release failure
        errors.append(f"completion schema unavailable or invalid: {exc}")
        return
    for error in sorted(
        validator.iter_errors(manifest),
        key=lambda item: tuple(str(part) for part in item.path),
    ):
        location = ".".join(str(part) for part in error.path) or "<root>"
        errors.append(f"schema:{location}: {error.message}")


def _validate_file_reference(
    repo: Path,
    reference: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> tuple[Path, str] | None:
    try:
        path, relative = _resolve_repo_path(repo, reference.get("path"))
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
        return None
    if not path.is_file():
        errors.append(f"{label}: artifact is not a regular file")
        return None
    if path.stat().st_size != reference.get("bytes"):
        errors.append(f"{label}: byte count mismatch")
    if _sha256(path) != reference.get("sha256"):
        errors.append(f"{label}: sha256 mismatch")
    return path, relative


def _validate_artifacts(
    manifest: Mapping[str, Any],
    repo: Path,
    commits: Mapping[str, str],
    errors: list[str],
) -> dict[str, tuple[Mapping[str, Any], Path, str]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
        return {}
    by_role: dict[str, tuple[Mapping[str, Any], Path, str]] = {}
    for index, reference in enumerate(artifacts):
        if not isinstance(reference, Mapping):
            errors.append(f"artifacts[{index}] must be an object")
            continue
        role = str(reference.get("role") or "")
        if not role or role in by_role:
            errors.append(f"artifacts[{index}]: role must be unique and non-empty")
            continue
        expected_commit_role = REQUIRED_ARTIFACT_ROLES.get(role)
        if expected_commit_role is None:
            errors.append(f"artifacts[{index}]: unexpected role {role}")
            continue
        if reference.get("commit_role") != expected_commit_role:
            errors.append(
                f"artifacts[{index}]: {role} must bind to {expected_commit_role} commit"
            )
        resolved = _validate_file_reference(
            repo, reference, f"artifacts[{index}]", errors
        )
        if resolved is None:
            continue
        path, relative = resolved
        if role in FIXED_ROLE_PATHS and relative != FIXED_ROLE_PATHS[role]:
            errors.append(f"artifacts[{index}]: {role} path is not canonical")
        commit = commits.get(expected_commit_role or "", "")
        if _git_object_exists(repo, commit):
            observed = _blob_sha256(repo, commit, relative)
            if observed != reference.get("sha256"):
                errors.append(
                    f"artifacts[{index}]: {role} is not the exact committed blob"
                )
        by_role[role] = (reference, path, relative)
    missing = set(REQUIRED_ARTIFACT_ROLES) - set(by_role)
    if missing:
        errors.append(f"artifacts missing required roles: {sorted(missing)}")

    try:
        stage_root, stage_relative = _resolve_repo_path(
            repo, manifest.get("stage_root")
        )
    except (OSError, ValueError) as exc:
        errors.append(f"stage_root: {exc}")
        return by_role
    if not stage_root.is_dir():
        errors.append("stage_root must be an existing directory")
        return by_role
    expected_stage_relative = f"artifacts/stages/{str(manifest.get('stage_id') or '')}"
    if stage_relative != expected_stage_relative:
        errors.append("stage_root must equal artifacts/stages/<stage_id>")
    for role in STAGE_SCOPED_ROLES & set(by_role):
        path, relative = by_role[role][1], by_role[role][2]
        try:
            path.resolve().relative_to(stage_root.resolve())
        except ValueError:
            errors.append(f"artifact role {role} escapes stage_root")
            continue
        expected_relative = f"{stage_relative}/{STAGE_ROLE_SUFFIXES[role]}"
        if relative != expected_relative:
            errors.append(f"artifact role {role} path is not canonical")
    return by_role


def _validate_phase_acceptances(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Mapping[str, Any], Path, str]],
    errors: list[str],
) -> dict[str, bool]:
    expected = {
        "D1": ("d1_acceptance_v2", "phase_gate_passed", "d1_acceptance"),
        "B0": ("utr_b0_acceptance.v2", "b0_gate_passed", "b0_acceptance"),
    }
    phase_acceptance = manifest.get("phase_acceptance")
    if not isinstance(phase_acceptance, Mapping) or set(phase_acceptance) != set(
        expected
    ):
        errors.append("phase_acceptance must contain exactly D1 and B0")
        return {}
    observed_gates: dict[str, bool] = {}
    for phase, (schema_version, gate_field, role) in expected.items():
        reference = phase_acceptance.get(phase)
        if not isinstance(reference, Mapping):
            errors.append(f"phase_acceptance.{phase} must be an object")
            continue
        role_value = artifacts.get(role)
        if role_value is None:
            continue
        artifact_reference, path, _ = role_value
        for field in ("path", "sha256", "bytes"):
            if reference.get(field) != artifact_reference.get(field):
                errors.append(
                    f"phase_acceptance.{phase} does not match artifact role {role}"
                )
        payload = _load_json(path, f"phase_acceptance.{phase}", errors)
        if not isinstance(payload, Mapping):
            continue
        if (
            reference.get("schema_version") != schema_version
            or payload.get("schema_version") != schema_version
        ):
            errors.append(f"phase_acceptance.{phase}: schema mismatch")
        if phase == "D1":
            declared_root = Path(str(payload.get("stage_d1_root") or ""))
            d1_gate_claimed = payload.get("phase_gate_passed") is True
            if not declared_root.is_absolute():
                errors.append("phase_acceptance.D1: stage_d1_root must be absolute")
            else:
                build_manifest = payload.get("required_artifact_validation", {}).get(
                    "build_manifest", {}
                )
                expected_build_manifest = (
                    declared_root / "build_manifest.json"
                ).resolve()
                declared_build_manifest = Path(str(build_manifest.get("path") or ""))
                if d1_gate_claimed and (
                    not expected_build_manifest.is_file()
                    or not declared_build_manifest.is_absolute()
                    or declared_build_manifest.resolve() != expected_build_manifest
                    or build_manifest.get("bytes")
                    != expected_build_manifest.stat().st_size
                    or build_manifest.get("sha256") != _sha256(expected_build_manifest)
                ):
                    errors.append(
                        "phase_acceptance.D1: external stage_d1_root "
                        "build_manifest path/bytes/sha256 binding is invalid"
                    )
                builder_audit = payload.get("builder_audit_validation", {})
                if d1_gate_claimed and (
                    not isinstance(builder_audit, Mapping)
                    or builder_audit.get("passed") is not True
                ):
                    errors.append(
                        "phase_acceptance.D1: audited production builder "
                        "causal-chain validation is not PASS"
                    )
        payload_gate = payload.get(gate_field)
        if (
            reference.get("gate_field") != gate_field
            or not isinstance(reference.get("gate_passed"), bool)
            or reference.get("gate_passed") is not payload_gate
        ):
            errors.append(f"phase_acceptance.{phase}: gate reference mismatch")
            continue
        observed_gates[phase] = bool(payload_gate)
        for semantic_error in validate_phase_acceptance(
            phase, payload, require_pass=bool(payload_gate)
        ):
            errors.append(f"phase_acceptance.{phase}: {semantic_error}")
    return observed_gates


def _validate_preflight_manifest(
    manifest: Mapping[str, Any],
    repo: Path,
    role: tuple[Mapping[str, Any], Path, str] | None,
    errors: list[str],
) -> Mapping[str, Any]:
    if role is None:
        return {}
    _, path, _ = role
    payload = _load_json(path, "preflight_manifest", errors)
    if not isinstance(payload, dict):
        return {}
    for error in validate_stage_manifest(payload):
        errors.append(f"preflight_manifest: {error}")
    if payload.get("stage_id") != manifest.get("stage_id"):
        errors.append("preflight_manifest stage_id mismatch")
    git_state = payload.get("git")
    isolated = git_state.get("isolated", {}) if isinstance(git_state, Mapping) else {}
    isolated_head = str(isolated.get("head") if isinstance(isolated, Mapping) else "")
    if not _git_object_exists(repo, isolated_head):
        errors.append("preflight_manifest git.isolated.head is not an existing commit")
    return payload


def _validate_code_manifest(
    repo: Path,
    code_commit: str,
    role: tuple[Mapping[str, Any], Path, str] | None,
    *,
    expected_base: str,
    stage_id: str,
    errors: list[str],
) -> None:
    if role is None:
        return
    _, path, self_relative = role
    payload = _load_json(path, "code_manifest", errors)
    if not isinstance(payload, Mapping):
        return
    if set(payload) != {
        "schema_version",
        "base_commit_sha",
        "files",
        "deleted_paths",
    }:
        errors.append("code_manifest keys are not sealed")
        return
    if payload.get("schema_version") != "d1_b0_code_manifest.v1":
        errors.append("code_manifest schema_version is invalid")
    base = str(payload.get("base_commit_sha") or "")
    if not _git_object_exists(repo, base) or not _is_ancestor(repo, base, code_commit):
        errors.append("code_manifest base_commit_sha is not an ancestor of code")
    if base != expected_base:
        errors.append(
            "code_manifest base_commit_sha does not equal "
            "preflight_manifest git.isolated.head"
        )
    stage_match = STAGE_RE.fullmatch(stage_id)
    if stage_match is None or not base.startswith(stage_match.group("base_short_sha")):
        errors.append("code_manifest base_commit_sha stage_id short SHA does not match")
    files = payload.get("files")
    deleted = payload.get("deleted_paths")
    if not isinstance(files, list) or not files:
        errors.append("code_manifest files must be a non-empty list")
        files = []
    if not isinstance(deleted, list) or not all(
        isinstance(item, str) and item for item in (deleted or [])
    ):
        errors.append("code_manifest deleted_paths must be a list of paths")
        deleted = []
    listed: set[str] = set()
    for index, reference in enumerate(files):
        if not isinstance(reference, Mapping) or set(reference) != {
            "path",
            "sha256",
            "bytes",
        }:
            errors.append(f"code_manifest.files[{index}] is invalid")
            continue
        resolved = _validate_file_reference(
            repo, reference, f"code_manifest.files[{index}]", errors
        )
        if resolved is None:
            continue
        _, relative = resolved
        if relative == self_relative or relative in listed:
            errors.append(f"code_manifest.files[{index}] duplicates a path")
        listed.add(relative)
        if _blob_sha256(repo, code_commit, relative) != reference.get("sha256"):
            errors.append(f"code_manifest.files[{index}] is not the code commit blob")
    deleted_set: set[str] = set()
    for raw in deleted:
        try:
            candidate = Path(raw)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError
            relative = candidate.as_posix()
        except ValueError:
            errors.append("code_manifest deleted path is unsafe")
            continue
        deleted_set.add(relative)
        if _blob_bytes(repo, code_commit, relative) is not None:
            errors.append(f"code_manifest deleted path still exists: {relative}")
    diff = _git(repo, "diff", "--name-only", "-z", base, code_commit)
    if diff.returncode == 0:
        try:
            changed = {
                item
                for item in diff.stdout.decode("utf-8", errors="strict").split("\0")
                if item
            }
        except UnicodeDecodeError:
            changed = set()
            errors.append("code_manifest Git diff paths are not UTF-8")
        expected = listed | deleted_set | {self_relative}
        if changed != expected:
            errors.append("code_manifest does not cover exact base..code changed paths")
    else:
        errors.append("code_manifest could not compute base..code diff")


def _validate_registry_artifact(
    repo: Path,
    role: tuple[Mapping[str, Any], Path, str] | None,
    artifacts: Mapping[str, tuple[Mapping[str, Any], Path, str]],
    commits: Mapping[str, str],
    remote_name: str,
    expected_remote_url: str,
    stage_status: str,
    observed_gates: Mapping[str, bool],
    errors: list[str],
) -> None:
    if role is None:
        return
    _, path, _ = role
    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        errors.append("task_registry is not valid UTF-8 YAML")
        return
    if not isinstance(registry, dict):
        errors.append("task_registry must be an object")
        return
    registry_errors = validate_registry(
        registry,
        repo_root=repo,
        release_profile="D1_B0",
        remote_name=remote_name,
        expected_remote_url=expected_remote_url,
        allow_unfrozen_b0_inventory=(
            stage_status != "FROZEN" and observed_gates.get("D1") is False
        ),
    )
    for error in registry_errors:
        errors.append(f"task_registry: {error}")
    tasks = {
        task.get("task_id"): task
        for task in registry.get("tasks", [])
        if isinstance(task, Mapping)
    }
    for phase, anchor in (("D1", "D1-08"), ("B0", "B0-05")):
        gate = observed_gates.get(phase)
        anchor_frozen = tasks.get(anchor, {}).get("status") == "FROZEN"
        if gate is True and not anchor_frozen:
            errors.append(
                f"task_registry: {anchor} must be FROZEN " f"for a passing {phase} gate"
            )
        if gate is False and anchor_frozen:
            errors.append(
                f"task_registry: {anchor} cannot be FROZEN "
                f"for a failing {phase} gate"
            )
    phase_bindings = {
        "D1": ("d1_acceptance", "d1_freeze"),
        "B0": ("b0_acceptance", "evidence"),
    }
    for phase, (artifact_role, commit_role) in phase_bindings.items():
        artifact = artifacts.get(artifact_role)
        if artifact is None:
            continue
        reference = artifact[0]
        expected_commit = commits.get(commit_role)
        for task_id, task in tasks.items():
            if (
                not isinstance(task_id, str)
                or not task_id.startswith(f"{phase}-")
                or task.get("status") not in {"VERIFIED", "FROZEN"}
            ):
                continue
            if task.get("commit_sha") != expected_commit:
                errors.append(
                    f"task_registry: {task_id}: commit_sha must equal "
                    f"git.{commit_role}_commit_sha"
                )
            if task.get("acceptance_artifact") != reference.get("path") or task.get(
                "acceptance_sha256"
            ) != reference.get("sha256"):
                errors.append(
                    f"task_registry: {task_id}: acceptance must equal the "
                    f"{artifact_role} completion reference"
                )


def _validate_governance_documents(
    manifest: Mapping[str, Any],
    repo: Path,
    artifacts: Mapping[str, tuple[Mapping[str, Any], Path, str]],
    commits: Mapping[str, str],
    errors: list[str],
) -> None:
    decision = artifacts.get("decision_log")
    if decision is not None:
        _, decision_path, decision_relative = decision
        try:
            decision_text = decision_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            errors.append("decision_log is not valid UTF-8")
        else:
            required_markers = {
                str(manifest.get("stage_id") or ""),
                "docs/contracts/mrna_latest_build_contract_v2.md",
                "D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-01",
                "D-2026-07-29-GOVERNANCE-METADATA-CORRECTION-02",
                "record_type: append_only_metadata_correction",
                "historical_records_modified: false",
            }
            missing = sorted(
                marker for marker in required_markers if marker not in decision_text
            )
            if missing:
                errors.append(
                    f"decision_log missing current governance markers: {missing}"
                )
        registry_blob = _blob_bytes(
            repo,
            commits.get("registry", ""),
            decision_relative,
        )
        for role in ("code", "d1_freeze", "evidence"):
            prior_blob = _blob_bytes(
                repo,
                commits.get(role, ""),
                decision_relative,
            )
            if (
                prior_blob is None
                or registry_blob is None
                or not registry_blob.startswith(prior_blob)
            ):
                errors.append(f"decision_log is not append-only from {role}")

    review = artifacts.get("independent_gate_review")
    if review is None:
        return
    _, review_path, _ = review
    try:
        review_text = review_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        errors.append("independent_gate_review is not valid UTF-8")
        return
    if str(manifest.get("stage_id") or "") not in review_text:
        errors.append("independent_gate_review does not identify the current stage")
    if GOAL_SHA256 not in review_text:
        errors.append("independent_gate_review does not identify the frozen contract")
    gate_ids = [int(value) for value in re.findall(r"\bGATE-([0-9]+)\b", review_text)]
    unique_gate_ids = set(gate_ids)
    latest = max(unique_gate_ids, default=0)
    if (
        len(gate_ids) != len(unique_gate_ids)
        or latest < MINIMUM_INDEPENDENT_GATE_ID
        or unique_gate_ids != set(range(1, latest + 1))
        or "blocking" not in review_text.lower()
    ):
        errors.append(
            "independent_gate_review must contain contiguous GATE-01 through "
            "the latest blocking record"
        )


def _status_has_unhashed_content(status_lines: list[str]) -> bool:
    """Return true when porcelain status names bytes the evidence did not hash."""
    for line in status_lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "?":
            return True
        if (
            fields[0] in {"1", "2"}
            and len(fields) >= 3
            and len(fields[2]) == 4
            and fields[2].startswith("S")
            and fields[2].endswith("U")
        ):
            return True
    return False


def _validate_protection(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Mapping[str, Any], Path, str]],
    errors: list[str],
) -> None:
    protection = manifest.get("protection")
    if not isinstance(protection, Mapping):
        errors.append("protection must be an object")
        return
    role_fields = {
        "preflight_role": "preflight_manifest",
        "protected_state_role": "protected_state",
        "recheck_role": "protection_recheck",
    }
    for field, expected in role_fields.items():
        if protection.get(field) != expected:
            errors.append(f"protection.{field} must be {expected}")
    payloads: dict[str, Mapping[str, Any]] = {}
    for role in role_fields.values():
        role_value = artifacts.get(role)
        if role_value is None:
            continue
        payload = _load_json(role_value[1], role, errors)
        if isinstance(payload, Mapping):
            payloads[role] = payload
    preflight = payloads.get("preflight_manifest", {})
    protected = payloads.get("protected_state", {})
    recheck = payloads.get("protection_recheck", {})
    if preflight.get("stage_id") != manifest.get("stage_id"):
        errors.append("preflight_manifest stage_id mismatch")
    if protected.get("stage_id") != manifest.get("stage_id"):
        errors.append("protected_state stage_id mismatch")
    if recheck.get("stage_id") != manifest.get("stage_id"):
        errors.append("protection_recheck stage_id mismatch")
    original_preflight = (
        preflight.get("git", {}).get("original", {})
        if isinstance(preflight.get("git"), Mapping)
        else {}
    )
    original_protected = (
        protected.get("original_worktree", {})
        if isinstance(protected.get("original_worktree"), Mapping)
        else {}
    )
    original_recheck = (
        recheck.get("original_worktree", {})
        if isinstance(recheck.get("original_worktree"), Mapping)
        else {}
    )
    path_text = str(protection.get("original_worktree_path") or "")
    initial_head = str(protection.get("initial_head") or "")
    final_head = str(protection.get("final_head") or "")
    initial_dirty = str(protection.get("initial_dirty_diff_sha256") or "")
    final_dirty = str(protection.get("final_dirty_diff_sha256") or "")
    if not (
        path_text
        and path_text == original_preflight.get("path")
        and path_text == original_protected.get("path")
        and path_text == original_recheck.get("path")
    ):
        errors.append("protection original worktree path is not evidence-bound")
    if not (
        initial_head
        == final_head
        == original_preflight.get("head")
        == original_protected.get("head")
        == original_recheck.get("head")
    ):
        errors.append("protection original worktree HEAD changed or is unbound")
    if not (
        initial_dirty
        == final_dirty
        == original_preflight.get("dirty_diff_sha256")
        == original_protected.get("dirty_diff_sha256")
        == original_recheck.get("dirty_diff_sha256")
    ):
        errors.append("protection dirty-state hash changed or is unbound")
    initial_status = original_protected.get("status_porcelain_v2")
    recheck_status = original_recheck.get("status_porcelain_v2")
    if not isinstance(initial_status, list) or not all(
        isinstance(item, str) for item in (initial_status or [])
    ):
        errors.append("protected_state status_porcelain_v2 is invalid")
        initial_status = []
    if _status_has_unhashed_content(initial_status):
        errors.append(
            "protection evidence has unhashed untracked content; "
            "original_worktree_unchanged cannot be proven"
        )
    if recheck_status != initial_status:
        errors.append("protection recheck status differs from preflight")
    live_repo = Path(path_text)
    if (
        not live_repo.is_absolute()
        or _git(live_repo, "rev-parse", "--is-inside-work-tree").returncode != 0
    ):
        errors.append("protection original_worktree_path is not a live Git worktree")
    else:
        live_head = _git_text(live_repo, "rev-parse", "HEAD")
        if live_head != final_head:
            errors.append("protection live original worktree HEAD differs")
        status = _git(live_repo, "status", "--porcelain=v2", "--untracked-files=all")
        try:
            live_status = status.stdout.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError:
            live_status = []
            errors.append("protection live Git status is not UTF-8")
        if status.returncode != 0 or live_status != recheck_status:
            errors.append("protection live original worktree status differs")
        live_dirty = _live_dirty_diff_sha256(live_repo)
        if live_dirty is None or live_dirty != final_dirty:
            errors.append("protection live original worktree dirty diff differs")
    protected_pids = sorted(
        int(item.get("pid"))
        for item in protected.get("processes", [])
        if isinstance(item, Mapping) and isinstance(item.get("pid"), int)
    )
    if sorted(protection.get("protected_pids", [])) != protected_pids:
        errors.append("protection protected_pids are not preflight-bound")
    actions = recheck.get("actions")
    if not isinstance(actions, Mapping):
        errors.append("protection_recheck actions must be an object")
        actions = {}
    zero_fields = {
        "processes_terminated",
        "existing_processes_modified",
        "original_worktree_mutations",
        "existing_results_overwritten",
        "raw_inputs_modified",
    }
    if any(actions.get(field) != 0 for field in zero_fields):
        errors.append("protection_recheck records a prohibited action")
    if (
        protection.get("original_worktree_unchanged") is not True
        or protection.get("terminated_protected_pids") != []
        or protection.get("processes_terminated") != 0
        or protection.get("existing_results_overwritten") != 0
        or recheck.get("terminated_protected_pids") != []
    ):
        errors.append("protection recheck is not clean")


def validate(
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
    expected_remote_url: str = CANONICAL_GITHUB_URL,
) -> list[str]:
    """Return semantic errors; empty means a remotely bound completion record."""
    errors: list[str] = []
    _validate_schema(manifest, errors)
    if manifest.get("artifact_type") != "stage_completion_manifest":
        errors.append("artifact_type must be stage_completion_manifest")
    if manifest.get("schema_version") != "utr_stage_completion.v2":
        errors.append("schema_version must be utr_stage_completion.v2")
    if not STAGE_RE.fullmatch(str(manifest.get("stage_id") or "")):
        errors.append("stage_id is invalid")
    if manifest.get("phase_ids") != ["D1", "B0"]:
        errors.append("phase_ids must be exactly ['D1', 'B0']")
    if manifest.get("workload_class") != "NON_NEURAL_DATA_BENCHMARK":
        errors.append("workload_class must be NON_NEURAL_DATA_BENCHMARK")
    started = _valid_time(manifest.get("started_at_utc"))
    ended = _valid_time(manifest.get("ended_at_utc"))
    if started is None or ended is None:
        errors.append("start/end timestamps must be timezone-aware ISO-8601")
    elif ended < started:
        errors.append("ended_at_utc precedes started_at_utc")
    contract = manifest.get("goal_contract")
    if contract != {
        "id": "utr_editflow_goal_v2",
        "sha256": GOAL_SHA256,
        "repository_snapshot": "docs/contracts/mrna_latest_build_contract_v2.md",
    }:
        errors.append("goal_contract is not the frozen contract")

    git_state = manifest.get("git")
    if not isinstance(git_state, Mapping):
        errors.append("git must be an object")
        return errors
    repository = Path(str(git_state.get("repository") or ""))
    if not repository.is_absolute():
        repository = (repo_root / repository).resolve()
    if _git(repository, "rev-parse", "--is-inside-work-tree").returncode != 0:
        errors.append("git.repository is not a Git worktree")
        return errors
    commits = {
        role: str(git_state.get(f"{role}_commit_sha") or "")
        for role in ("code", "d1_freeze", "evidence", "registry")
    }
    for role, commit in commits.items():
        if not _git_object_exists(repository, commit):
            errors.append(f"git.{role}_commit_sha is not an existing commit")
    if all(_git_object_exists(repository, value) for value in commits.values()):
        if len(set(commits.values())) != len(commits):
            errors.append(
                "git code/d1_freeze/evidence/registry commits must be distinct stages"
            )
        if not _is_ancestor(repository, commits["code"], commits["d1_freeze"]):
            errors.append("git code commit is not an ancestor of d1_freeze")
        if not _is_ancestor(
            repository,
            commits["d1_freeze"],
            commits["evidence"],
        ):
            errors.append("git d1_freeze commit is not an ancestor of evidence")
        if not _is_ancestor(repository, commits["evidence"], commits["registry"]):
            errors.append("git evidence commit is not an ancestor of registry")

    remote_name = str(git_state.get("remote_name") or "")
    manifest_remote_url = str(git_state.get("canonical_remote_url") or "")
    actual_remote_url = _git_text(repository, "remote", "get-url", remote_name)
    if (
        actual_remote_url is None
        or _canonical_remote_identity(actual_remote_url)
        != _canonical_remote_identity(expected_remote_url)
        or _canonical_remote_identity(manifest_remote_url)
        != _canonical_remote_identity(expected_remote_url)
    ):
        errors.append("git remote identity is not the expected canonical repository")
    remote_ref = str(git_state.get("published_remote_ref") or "")
    if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
        errors.append("git.published_remote_ref must be a real heads/tags ref")
        release_commit = None
    else:
        release_commit = _remote_ref_oid(repository, remote_name, remote_ref)
        if release_commit is None:
            errors.append("git.published_remote_ref is absent from the real remote")
        elif not _git_object_exists(repository, release_commit):
            errors.append("remote release commit is not available locally")
        elif _git_object_exists(repository, commits["registry"]) and not _is_ancestor(
            repository, commits["registry"], release_commit
        ):
            errors.append("git registry commit is not an ancestor of remote release")
        elif release_commit in set(commits.values()):
            errors.append("remote release commit must follow the registry commit")
        else:
            parent_line = _git_text(
                repository,
                "rev-list",
                "--parents",
                "-n",
                "1",
                release_commit,
            )
            parents = parent_line.split()[1:] if parent_line is not None else []
            if parents != [commits["registry"]]:
                errors.append(
                    "remote release must be the direct single-parent child of registry"
                )
            expected_manifest_relative = (
                f"{manifest.get('stage_root')}/release/completion_manifest.json"
            )
            release_diff = _git(
                repository,
                "diff",
                "--name-only",
                "-z",
                commits["registry"],
                release_commit,
            )
            try:
                changed_paths = (
                    {
                        item
                        for item in release_diff.stdout.decode(
                            "utf-8",
                            errors="strict",
                        ).split("\0")
                        if item
                    }
                    if release_diff.returncode == 0
                    else set()
                )
            except UnicodeDecodeError:
                changed_paths = set()
            if changed_paths != {expected_manifest_relative}:
                errors.append(
                    "remote release delta must contain only the completion manifest"
                )
            if (
                _blob_bytes(
                    repository,
                    commits["registry"],
                    expected_manifest_relative,
                )
                is not None
                or _blob_bytes(
                    repository,
                    release_commit,
                    expected_manifest_relative,
                )
                is None
            ):
                errors.append(
                    "remote release must newly add the canonical completion manifest"
                )

    artifacts = _validate_artifacts(manifest, repository, commits, errors)
    preflight = _validate_preflight_manifest(
        manifest,
        repository,
        artifacts.get("preflight_manifest"),
        errors,
    )
    observed_gates = _validate_phase_acceptances(manifest, artifacts, errors)
    if observed_gates.get("B0") is True and observed_gates.get("D1") is not True:
        errors.append("B0 acceptance cannot pass before D1 acceptance")
    status = manifest.get("status")
    if status == "FROZEN":
        if observed_gates != {"D1": True, "B0": True}:
            errors.append("FROZEN requires passing D1 and B0 acceptance")
        if manifest.get("stop_reason") is not None:
            errors.append("FROZEN completion must have null stop_reason")
        if manifest.get("known_deviations") != []:
            errors.append("FROZEN completion cannot carry known deviations")
    elif status in {"SAFE_PAUSED", "FAILED_WITH_EVIDENCE"}:
        if not str(manifest.get("stop_reason") or "").strip():
            errors.append("non-frozen completion requires a stop_reason")
        if observed_gates and all(observed_gates.values()):
            errors.append("non-frozen status requires at least one non-passing phase")
    else:
        errors.append("status is invalid")

    preflight_git = preflight.get("git")
    preflight_isolated = (
        preflight_git.get("isolated", {}) if isinstance(preflight_git, Mapping) else {}
    )
    expected_base = str(
        preflight_isolated.get("head")
        if isinstance(preflight_isolated, Mapping)
        else ""
    )
    _validate_code_manifest(
        repository,
        commits["code"],
        artifacts.get("code_manifest"),
        expected_base=expected_base,
        stage_id=str(manifest.get("stage_id") or ""),
        errors=errors,
    )
    _validate_registry_artifact(
        repository,
        artifacts.get("task_registry"),
        artifacts,
        commits,
        remote_name,
        expected_remote_url,
        str(status or ""),
        observed_gates,
        errors,
    )
    _validate_governance_documents(
        manifest,
        repository,
        artifacts,
        commits,
        errors,
    )
    _validate_protection(manifest, artifacts, errors)

    boundary = manifest.get("execution_boundary")
    if boundary != {
        "formal_neural_activity_started": False,
        "gpu_validation_started": False,
        "cuda_fallback_events": 0,
        "gpu_requirement_status": "NOT_APPLICABLE_NO_NEURAL_WORK",
        "smoke_or_proxy_is_final_evidence": False,
    }:
        errors.append("execution_boundary violates the non-neural stage contract")
    claims = manifest.get("claim_boundary")
    if claims != {
        "scientific_result_claimed": False,
        "efficacy_claimed": False,
        "sota_claimed": False,
        "foundation_status": "UNKNOWN_PENDING_FM0",
        "allowed_claim": "NONE",
    }:
        errors.append("claim_boundary exceeds D1/B0 evidence")
    if not isinstance(manifest.get("known_deviations"), list) or not all(
        isinstance(item, str) for item in manifest.get("known_deviations", [])
    ):
        errors.append("known_deviations must be a list of strings")

    if manifest_path is None:
        errors.append("manifest_path is required to bind the completion manifest")
    elif release_commit is not None and _git_object_exists(repository, release_commit):
        try:
            current_path, relative = _resolve_manifest_path(repository, manifest_path)
        except (OSError, ValueError) as exc:
            errors.append(f"completion manifest path is invalid: {exc}")
        else:
            expected_manifest_relative = (
                f"{manifest.get('stage_root')}/release/completion_manifest.json"
            )
            if relative != expected_manifest_relative:
                errors.append("completion manifest path is not canonical")
            disk_manifest = _load_json(current_path, "completion manifest", errors)
            if disk_manifest != manifest:
                errors.append("completion manifest payload differs from manifest_path")
            if _sha256(current_path) != _blob_sha256(
                repository, release_commit, relative
            ):
                errors.append(
                    "completion manifest is not the exact remote release commit blob"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "result": "FAIL",
                    "errors": [f"manifest is not valid UTF-8 JSON: {exc}"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    root = (args.repo_root or Path.cwd()).resolve()
    errors = validate(
        manifest,
        root,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "result": "PASS" if not errors else "FAIL",
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
