#!/usr/bin/env python3
"""Fail-closed semantic validation for a D1+B0 stage manifest."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import jsonschema

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = SOURCE_ROOT / "schemas" / "stage_manifest.schema.json"
GOAL_SHA256 = "ff5a440910c9c8ef47e460b3f2d6a291c7fce12e687e090f2200dc79796a89c3"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGE_RE = re.compile(
    r"^D1_B0_(?P<timestamp>[0-9]{8}T[0-9]{6}Z)_" r"[0-9a-f]{7}(?:_A[0-9]+)?$"
)


def _valid_time(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_schema(manifest: dict, errors: list[str]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        format_checker = jsonschema.FormatChecker()
        format_checker.checks("date-time")(_valid_time)
        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=format_checker,
        )
    except Exception as exc:
        errors.append(f"stage manifest schema unavailable or invalid: {exc}")
        return
    for error in sorted(
        validator.iter_errors(manifest),
        key=lambda item: tuple(str(part) for part in item.path),
    ):
        location = ".".join(str(part) for part in error.path) or "<root>"
        errors.append(f"schema:{location}: {error.message}")


def validate(manifest: dict) -> list[str]:
    """Return semantic errors; an empty list means the manifest is accepted."""
    errors: list[str] = []
    _validate_schema(manifest, errors)
    required = {
        "artifact_type",
        "schema_version",
        "stage_id",
        "phase_ids",
        "captured_at_utc",
        "workload_class",
        "goal_contract",
        "remote",
        "git",
        "protection",
        "resources",
        "data_state",
        "execution_boundary",
    }
    missing = sorted(required - set(manifest))
    if missing:
        return [f"missing required keys: {missing}"]

    if manifest["artifact_type"] != "stage_manifest":
        errors.append("artifact_type must be stage_manifest")
    if manifest["schema_version"] != "utr_stage_manifest.v1":
        errors.append("schema_version must be utr_stage_manifest.v1")
    stage_match = STAGE_RE.fullmatch(str(manifest["stage_id"]))
    if stage_match is None:
        errors.append("stage_id is not a D1_B0 stage identifier")
    else:
        try:
            datetime.strptime(stage_match.group("timestamp"), "%Y%m%dT%H%M%SZ")
        except ValueError:
            errors.append("stage_id contains an invalid UTC calendar timestamp")
    if manifest["phase_ids"] != ["D1", "B0"]:
        errors.append("phase_ids must be exactly ['D1', 'B0']")
    if manifest["workload_class"] != "NON_NEURAL_DATA_BENCHMARK":
        errors.append("D1+B0 workload_class must be NON_NEURAL_DATA_BENCHMARK")

    contract = manifest.get("goal_contract", {})
    if contract.get("id") != "mrna_editflow_single_active_contract":
        errors.append("goal contract id mismatch")
    if contract.get("sha256") != GOAL_SHA256:
        errors.append("goal contract sha256 mismatch")
    if (
        contract.get("repository_snapshot")
        != "docs/contracts/mrna_editflow_contract.md"
    ):
        errors.append("repository contract snapshot mismatch")

    git_state = manifest.get("git", {})
    for role in ("original", "isolated"):
        snapshot = git_state.get(role)
        if not isinstance(snapshot, dict):
            errors.append(f"git.{role} must be an object")
            continue
        if not COMMIT_RE.fullmatch(str(snapshot.get("head", ""))):
            errors.append(f"git.{role}.head must be a full commit SHA")
        dirty_sha = snapshot.get("dirty_diff_sha256")
        if dirty_sha is not None and not SHA256_RE.fullmatch(str(dirty_sha)):
            errors.append(f"git.{role}.dirty_diff_sha256 must be SHA-256 or null")
        if snapshot.get("clean") is True and dirty_sha is not None:
            errors.append(f"git.{role} cannot be clean with a dirty diff hash")
        if snapshot.get("clean") is False and dirty_sha is None:
            errors.append(f"git.{role} dirty state requires a diff hash")

    protection = manifest.get("protection", {})
    for counter in (
        "processes_terminated",
        "original_worktree_mutations",
        "existing_results_overwritten",
    ):
        if protection.get(counter) != 0:
            errors.append(f"protection.{counter} must remain zero")
    for process in protection.get("protected_processes", []):
        if process.get("action") != "observe_only":
            errors.append("every protected process must be observe_only")

    resources = manifest.get("resources", {})
    gpu = resources.get("gpu", {})
    if gpu.get("formal_neural_work_planned") is not False:
        errors.append("D1+B0 cannot declare formal neural work")

    data_state = manifest.get("data_state", {})
    encode = data_state.get("encode_reconstruction", {})
    if encode.get("role") != "OBSERVATIONAL_ONLY":
        errors.append("ENCODE reconstruction must remain OBSERVATIONAL_ONLY")
    if encode.get("verified_files", 0) > encode.get("expected_files", -1):
        errors.append("ENCODE verified_files cannot exceed expected_files")

    boundary = manifest.get("execution_boundary", {})
    expected_boundary = {
        "formal_neural_activity_started": False,
        "gpu_validation_started": False,
        "cuda_fallback_events": 0,
        "gpu_requirement_status": "NOT_APPLICABLE_NO_NEURAL_WORK",
        "smoke_or_proxy_is_final_evidence": False,
        "d1_required_before_b0": True,
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            errors.append(f"execution_boundary.{key} must equal {expected!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate(manifest)
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
