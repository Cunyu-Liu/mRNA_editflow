#!/usr/bin/env python3
"""Freeze the terminal 930 S1 family as non-authoritative execution evidence."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping


WORKTREE = Path(__file__).resolve().parents[2]
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OLD_HEAD = "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
SCREEN_SEED = 20260911
RUN_IDS = ("v4_s1_full", "v4_s1_single_mode")
CHECKPOINT_PASSES = (4, 6, 8, 10)
OBJECTIVE_IDENTITY = "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY"
OBJECTIVE_WEIGHT = 0.05
DEFECT_IDENTITY = "PARAMETER_INITIALIZATION_SEED_APPLIED_AFTER_MODEL_CONSTRUCTION"
RUNTIME_SCHEMA = "route_a_v3_route2_xeditsetflow_v4_s1_screen_runtime.v1"
RUNNING_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_RUNNING"
SCIENTIFIC_TERMINAL_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_AND_GATE_TERMINAL"
TECHNICAL_TERMINAL_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_TECHNICAL_FAILURE"
RUNNING_OBSERVED_STATUS = "XEDITSETFLOW_V4_S1_930_SCREEN_STILL_RUNNING"
SCIENTIFIC_INVALIDATION_STATUS = (
    "XEDITSETFLOW_V4_S1_930_SCIENTIFIC_TERMINAL_INVALIDATED"
)
TECHNICAL_INVALIDATION_STATUS = (
    "XEDITSETFLOW_V4_S1_930_TECHNICAL_TERMINAL_INVALIDATED"
)
RECEIPT_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_930_terminal_invalidation.v1"
)
OLD_FAMILY_ROOT = (
    ROOT
    / "experiments/xeditsetflow_v4/"
    f"s1_screen_seed_{SCREEN_SEED}_runner_{OLD_HEAD}"
)
OLD_RUNTIME = OLD_FAMILY_ROOT / "runtime.json"
CANONICAL_RECEIPT = (
    ROOT
    / "audits/xeditsetflow_v4/"
    f"s1_screen_seed_{SCREEN_SEED}_runner_{OLD_HEAD}_terminal_invalidation.json"
)
REPAIR_AUDIT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair_v1.json"
)


class S1TerminalInvalidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S1TerminalInvalidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _partial(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".partial")


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"terminal invalidation receipt already exists: {path}")
    partial = _partial(path)
    require(
        not partial.exists(),
        f"partial terminal invalidation receipt already exists: {partial}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def scheduler_process_is_alive(pid: int) -> bool:
    """Return whether the recorded scheduler PID still names a live process."""

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        text=True,
        capture_output=True,
        check=False,
    )
    command_line = result.stdout.strip()
    error = result.stderr.strip()
    if result.returncode == 1 and not command_line and not error:
        return False
    require(
        result.returncode == 0,
        f"scheduler process inspection failed: {error or result.returncode}",
    )
    return (
        "run_route2_xeditsetflow_s1_screen_scheduler.py" in command_line
        and OLD_HEAD in command_line
    )


def _protected_reads_zero(payload: Mapping[str, Any], label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a protected outcome read",
    )


def _validate_repair_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    affected = payload.get("affected_family")
    defect = payload.get("defect")
    claim = payload.get("claim_boundary")
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair.v1"
        and payload.get("status")
        == "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY"
        and isinstance(affected, Mapping)
        and affected.get("runner_git_head") == OLD_HEAD
        and int(affected.get("screen_seed", -1)) == SCREEN_SEED
        and affected.get("run_ids") == list(RUN_IDS)
        and affected.get("runtime_path") == str(OLD_RUNTIME)
        and affected.get("launcher_consumed_once") is True
        and affected.get("artifacts_immutable") is True
        and affected.get("same_family_retry_authorized") is False
        and isinstance(defect, Mapping)
        and defect.get("identity") == DEFECT_IDENTITY
        and defect.get("model_construction_consumes_cpu_rng") is True
        and defect.get("cpu_manual_seed_was_after_model_construction") is True
        and defect.get("cuda_manual_seed_all_was_after_model_construction") is True
        and defect.get("full_and_single_mode_used_independent_processes") is True
        and defect.get("nominal_seed_controlled_parameter_initialization") is False
        and defect.get("matched_full_single_initialization_established") is False
        and defect.get("exact_seed_reproducibility_established") is False
        and defect.get("affected_family_can_authorize_successor") is False
        and isinstance(claim, Mapping)
        and claim.get("affected_nominal_gate_is_scientific_successor_authority")
        is False,
        "tracked 930 S1 seed-after-model defect audit changed",
    )
    _protected_reads_zero(payload, "tracked 930 S1 repair audit")
    return {
        "identity": DEFECT_IDENTITY,
        "model_construction_consumes_cpu_rng": True,
        "nominal_seed_controlled_parameter_initialization": False,
        "matched_full_single_initialization_established": False,
        "affected_family_can_authorize_successor": False,
        "repair_audit_path": str(REPAIR_AUDIT),
    }


def _runtime_identity(runtime: Mapping[str, Any]) -> None:
    require(runtime.get("schema_version") == RUNTIME_SCHEMA, "old S1 runtime schema changed")
    require(runtime.get("git_head") == OLD_HEAD, "old S1 runtime Git HEAD changed")
    require(
        runtime.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(runtime.get("cross_state_candidate_mode_responsibility_weight", -1.0))
        == OBJECTIVE_WEIGHT,
        "old S1 runtime objective changed",
    )
    require(runtime.get("free_memory_gate_applied") is False, "old S1 used a free-memory gate")
    require(
        runtime.get("active_performance_output_read") is False,
        "old S1 runtime reports an active performance-output read",
    )
    _protected_reads_zero(runtime, "old S1 runtime")


def _expected_job_keys() -> tuple[set[str], set[str]]:
    training = {f"training:{run_id}" for run_id in RUN_IDS}
    validation = {
        f"validation:{run_id}:pass_{checkpoint_pass}"
        for run_id in RUN_IDS
        for checkpoint_pass in CHECKPOINT_PASSES
    }
    return training, validation


def _validate_terminal_payload(
    payload: Mapping[str, Any],
    *,
    stage: str,
    run_id: str,
    checkpoint_pass: int | None,
    terminal_kind: str,
) -> None:
    _protected_reads_zero(payload, f"old S1 {stage.lower()} {terminal_kind.lower()}")
    require(payload.get("run_id") == run_id, f"old S1 {stage} run id changed")
    if "seed" in payload:
        require(int(payload.get("seed", -1)) == SCREEN_SEED, f"old S1 {stage} seed changed")
    if "run_stage" in payload:
        require(payload.get("run_stage") == "SCREEN", f"old S1 {stage} stage changed")
    if checkpoint_pass is not None and "checkpoint_pass" in payload:
        require(
            int(payload.get("checkpoint_pass", -1)) == checkpoint_pass,
            f"old S1 {stage} checkpoint pass changed",
        )
    if terminal_kind == "SUMMARY" and stage == "TRAINING":
        require(
            payload.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_s1_training_summary.v1"
            and payload.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION"
            and payload.get("run_stage") == "SCREEN"
            and int(payload.get("seed", -1)) == SCREEN_SEED,
            "old S1 training summary identity changed",
        )
    if terminal_kind == "SUMMARY" and stage == "VALIDATION":
        require(
            payload.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1"
            and payload.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE"
            and payload.get("run_stage") == "SCREEN"
            and int(payload.get("seed", -1)) == SCREEN_SEED
            and int(payload.get("checkpoint_pass", -1)) == checkpoint_pass,
            "old S1 Validation summary identity changed",
        )


def _inspect_job(
    job_key: str,
    row: Mapping[str, Any],
    *,
    stage: str,
    scientific_terminal: bool,
) -> dict[str, Any]:
    run_id = str(row.get("run_id"))
    require(run_id in RUN_IDS and job_key.startswith(f"{stage.lower()}:{run_id}"), f"old S1 {stage} job identity changed: {job_key}")
    checkpoint_pass = None
    if stage == "VALIDATION":
        checkpoint_pass = int(row.get("checkpoint_pass", -1))
        require(
            checkpoint_pass in CHECKPOINT_PASSES
            and job_key == f"validation:{run_id}:pass_{checkpoint_pass}",
            f"old S1 Validation job identity changed: {job_key}",
        )
    summary = Path(str(row.get("terminal_summary", "")))
    failure = Path(str(row.get("terminal_failure", "")))
    require(summary.is_absolute() and failure.is_absolute(), f"old S1 terminal path is not absolute: {job_key}")
    require(
        summary.is_relative_to(OLD_RUNTIME.parent)
        and failure.is_relative_to(OLD_RUNTIME.parent),
        f"old S1 terminal path escaped the canonical family: {job_key}",
    )
    require(not _partial(summary).exists() and not _partial(failure).exists(), f"old S1 terminal partial exists: {job_key}")
    status = row.get("status")
    require(status not in {"PENDING", "RUNNING"}, f"old S1 job is not terminal: {job_key}")
    if status == "NOT_RUN_AFTER_TERMINAL_FAILURE":
        require(not scientific_terminal, f"scientific old S1 job was not run: {job_key}")
        require(
            not summary.exists()
            and not failure.exists()
            and row.get("terminal_artifact_kind") is None,
            f"not-run old S1 job has a terminal artifact: {job_key}",
        )
        return {"job_key": job_key, "status": status, "terminal_artifact_kind": None}
    require(status == "TERMINAL_COMPLETE", f"old S1 job has ambiguous terminal status: {job_key}")
    terminal_count = int(summary.is_file()) + int(failure.is_file())
    require(terminal_count == 1, f"old S1 job lacks a unique terminal artifact: {job_key}")
    terminal_kind = "SUMMARY" if summary.is_file() else "FAILURE"
    require(row.get("terminal_artifact_kind") == terminal_kind, f"old S1 runtime/artifact terminal kind differs: {job_key}")
    if scientific_terminal:
        require(
            terminal_kind == "SUMMARY" and int(row.get("return_code", -1)) == 0,
            f"scientific old S1 job is not an exact success: {job_key}",
        )
    payload_path = summary if terminal_kind == "SUMMARY" else failure
    payload = read_json(payload_path)
    _validate_terminal_payload(
        payload,
        stage=stage,
        run_id=run_id,
        checkpoint_pass=checkpoint_pass,
        terminal_kind=terminal_kind,
    )
    return {
        "job_key": job_key,
        "status": status,
        "terminal_artifact_kind": terminal_kind,
        "terminal_artifact": str(payload_path),
        "return_code": row.get("return_code"),
    }


def _inspect_jobs(runtime: Mapping[str, Any], *, scientific_terminal: bool) -> list[dict[str, Any]]:
    training = runtime.get("training_jobs")
    validation = runtime.get("validation_jobs")
    require(isinstance(training, Mapping) and isinstance(validation, Mapping), "old S1 runtime job tables are absent")
    expected_training, expected_validation = _expected_job_keys()
    require(set(training) == expected_training, "old S1 training job inventory changed")
    require(set(validation) == expected_validation, "old S1 Validation job inventory changed")
    rows: list[dict[str, Any]] = []
    for key, row in training.items():
        require(isinstance(row, Mapping), f"old S1 training row is not an object: {key}")
        rows.append(_inspect_job(str(key), row, stage="TRAINING", scientific_terminal=scientific_terminal))
    for key, row in validation.items():
        require(isinstance(row, Mapping), f"old S1 Validation row is not an object: {key}")
        rows.append(_inspect_job(str(key), row, stage="VALIDATION", scientific_terminal=scientific_terminal))
    return rows


def _inspect_scientific_adjudication(runtime: Mapping[str, Any]) -> dict[str, Any]:
    adjudication = runtime.get("adjudication")
    require(isinstance(adjudication, Mapping), "old S1 adjudication row is absent")
    gate = Path(str(adjudication.get("gate_path", "")))
    failure = Path(str(adjudication.get("failure_path", "")))
    require(gate.is_absolute() and failure.is_absolute(), "old S1 adjudication paths are not absolute")
    require(
        gate.is_relative_to(OLD_RUNTIME.parent)
        and failure.is_relative_to(OLD_RUNTIME.parent),
        "old S1 adjudication path escaped the canonical family",
    )
    require(not _partial(gate).exists() and not _partial(failure).exists(), "old S1 adjudication has a partial terminal")
    require(
        adjudication.get("status") == "TERMINAL_COMPLETE"
        and adjudication.get("terminal_artifact_kind") == "GATE"
        and int(adjudication.get("return_code", -1)) == 0
        and gate.is_file()
        and not failure.exists(),
        "old S1 scientific adjudication is not uniquely terminal",
    )
    payload = read_json(gate)
    status = payload.get("status")
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1"
        and status in {"XEDITSETFLOW_V4_S1_SCREEN_PASS", "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"}
        and int(payload.get("screen_seed", -1)) == SCREEN_SEED
        and payload.get("legacy_v4_confirmation_authorized") is False
        and payload.get("confirmation_authorized") is False
        and payload.get("additional_seed_authorized") is False
        and payload.get("development_test_authorized") is False
        and payload.get("guidance_authorized") is False
        and payload.get("successor_protocol_required")
        == (status == "XEDITSETFLOW_V4_S1_SCREEN_PASS")
        and payload.get("s1_mechanics_screen_passed")
        == (status == "XEDITSETFLOW_V4_S1_SCREEN_PASS"),
        "old S1 nominal scientific gate identity or authorization changed",
    )
    _protected_reads_zero(payload, "old S1 nominal scientific gate")
    require(runtime.get("first_terminal_failure") is None, "scientific old S1 runtime records a technical failure")
    return {"terminal_artifact": str(gate), "nominal_gate_status": status}


def _inspect_technical_adjudication(runtime: Mapping[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    first_failure = runtime.get("first_terminal_failure")
    require(isinstance(first_failure, Mapping) and bool(first_failure), "technical old S1 runtime lacks one first terminal failure")
    job_key = str(first_failure.get("job_key"))
    bound_jobs = [row for row in jobs if row["job_key"] == job_key]
    require(
        job_key == "adjudication" or len(bound_jobs) == 1,
        "old S1 first terminal failure is not uniquely bound",
    )
    if bound_jobs:
        bound = bound_jobs[0]
        require(
            first_failure.get("terminal_artifact_kind")
            == bound["terminal_artifact_kind"]
            and first_failure.get("return_code") == bound["return_code"]
            and (
                bound["terminal_artifact_kind"] == "FAILURE"
                or int(bound["return_code"] or 0) != 0
            ),
            "old S1 first terminal failure points to a successful job",
        )
    adjudication = runtime.get("adjudication")
    require(isinstance(adjudication, Mapping), "old S1 adjudication row is absent")
    gate = Path(str(adjudication.get("gate_path", "")))
    failure = Path(str(adjudication.get("failure_path", "")))
    require(gate.is_absolute() and failure.is_absolute(), "old S1 adjudication paths are not absolute")
    require(
        gate.is_relative_to(OLD_RUNTIME.parent)
        and failure.is_relative_to(OLD_RUNTIME.parent),
        "old S1 adjudication path escaped the canonical family",
    )
    require(not _partial(gate).exists() and not _partial(failure).exists(), "old S1 adjudication has a partial terminal")
    status = adjudication.get("status")
    if status == "NOT_RUN_AFTER_TERMINAL_FAILURE":
        require(job_key != "adjudication" and not gate.exists() and not failure.exists(), "old S1 not-run adjudication has terminal evidence")
        terminal = None
    else:
        require(
            status == "TECHNICAL_FAILURE"
            and job_key == "adjudication"
            and first_failure.get("terminal_artifact_kind") == "FAILURE"
            and not gate.exists()
            and failure.is_file()
            and adjudication.get("terminal_artifact_kind") == "FAILURE",
            "old S1 technical adjudication is not uniquely terminal",
        )
        payload = read_json(failure)
        _protected_reads_zero(payload, "old S1 adjudication failure")
        terminal = str(failure)
    not_run_count = sum(row["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE" for row in jobs)
    return {
        "status": status,
        "terminal_artifact": terminal,
        "first_terminal_failure": dict(first_failure),
        "not_run_after_terminal_failure_count": not_run_count,
    }


def run(
    *,
    runtime_path: Path = OLD_RUNTIME,
    receipt_path: Path = CANONICAL_RECEIPT,
    repair_audit_path: Path = REPAIR_AUDIT,
    process_is_alive: Callable[[int], bool] = scheduler_process_is_alive,
) -> dict[str, Any]:
    require(runtime_path == OLD_RUNTIME, "old S1 runtime path is not canonical")
    require(receipt_path == CANONICAL_RECEIPT, "old S1 invalidation receipt path is not canonical")
    require(repair_audit_path == REPAIR_AUDIT, "old S1 repair audit path is not canonical")
    require(not receipt_path.exists(), f"terminal invalidation receipt already exists: {receipt_path}")
    require(not _partial(receipt_path).exists(), f"partial terminal invalidation receipt already exists: {_partial(receipt_path)}")

    # This is the one and only old-runtime read in the transition.
    runtime = read_json(runtime_path)
    _runtime_identity(runtime)
    runtime_status = runtime.get("status")
    if runtime_status == RUNNING_STATUS:
        return {
            "status": RUNNING_OBSERVED_STATUS,
            "runtime_status": RUNNING_STATUS,
            "receipt_written": False,
        }
    require(
        runtime_status in {SCIENTIFIC_TERMINAL_STATUS, TECHNICAL_TERMINAL_STATUS},
        "old S1 runtime is neither running nor an exact terminal class",
    )
    require(not _partial(runtime_path).exists(), "terminal old S1 runtime has a partial sibling")
    scheduler_pid = runtime.get("scheduler_pid")
    require(
        isinstance(scheduler_pid, int)
        and not isinstance(scheduler_pid, bool)
        and scheduler_pid > 0,
        "old S1 scheduler PID is absent",
    )
    require(not process_is_alive(scheduler_pid), "old S1 scheduler process is still alive")
    defect = _validate_repair_audit(read_json(repair_audit_path))
    scientific_terminal = runtime_status == SCIENTIFIC_TERMINAL_STATUS
    jobs = _inspect_jobs(runtime, scientific_terminal=scientific_terminal)
    if scientific_terminal:
        terminal = _inspect_scientific_adjudication(runtime)
        receipt_status = SCIENTIFIC_INVALIDATION_STATUS
        terminal_class = "SCIENTIFIC_GATE_TERMINAL"
    else:
        terminal = _inspect_technical_adjudication(runtime, jobs)
        receipt_status = TECHNICAL_INVALIDATION_STATUS
        terminal_class = "TECHNICAL_FAILURE_TERMINAL"
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": receipt_status,
        "terminal_class": terminal_class,
        "old_runner_git_head": OLD_HEAD,
        "old_runtime_path": str(OLD_RUNTIME),
        "old_runtime_status": runtime_status,
        "screen_seed": SCREEN_SEED,
        "run_ids": list(RUN_IDS),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "scheduler_pid": scheduler_pid,
        "scheduler_process_gone": True,
        "terminal_jobs": jobs,
        "terminal_adjudication": terminal,
        "known_defect": defect,
        "nominal_terminal_retained_as_execution_evidence": True,
        "nominal_terminal_rewritten": False,
        "scientific_successor_authorized": False,
        "successor_authorized": False,
        "same_family_retry_authorized": False,
        "old_family_artifacts_read_only": True,
        "old_runtime_read_count_this_transition": 1,
        "gpu_inventory_or_probe_executed": False,
        "gpu_or_model_execution_started": False,
        "protected_outcome_payload_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    print(json.dumps(run(), sort_keys=True))


if __name__ == "__main__":
    main()
