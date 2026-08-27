#!/usr/bin/env python3
"""Run the isolated SetFlow V4 S1 screen as one fail-closed package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping


RUN_IDS = ("v4_s1_full", "v4_s1_single_mode")
CHECKPOINT_PASSES = (4, 6, 8, 10)
RUNTIME_SCHEMA = "route_a_v3_route2_xeditsetflow_v4_s1_screen_runtime.v1"
RUNNING_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_RUNNING"
TERMINAL_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_AND_GATE_TERMINAL"
FAILURE_STATUS = "XEDITSETFLOW_V4_S1_SCREEN_TECHNICAL_FAILURE"


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_new_terminal(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"terminal artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"partial terminal artifact already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def exact_terminal(job: Mapping[str, Any]) -> str | None:
    summary = Path(str(job["terminal_summary"]))
    failure = Path(str(job["terminal_failure"]))
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def partial_terminal_artifacts(job: Mapping[str, Any]) -> list[str]:
    paths = [
        Path(str(job["terminal_summary"])),
        Path(str(job["terminal_failure"])),
    ]
    return [
        str(partial)
        for path in paths
        for partial in (path.with_suffix(path.suffix + ".partial"),)
        if partial.exists()
    ]


def inspect_worktree_identity(
    worktree: Path, *, expected_head: str
) -> dict[str, Any] | None:
    try:
        observed_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as error:
        return {
            "reason": "WORKTREE_IDENTITY_INSPECTION_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if observed_head != expected_head:
        return {
            "reason": "WORKTREE_HEAD_MISMATCH",
            "expected_git_head": expected_head,
            "observed_git_head": observed_head,
        }
    if porcelain.strip():
        return {
            "reason": "WORKTREE_NOT_CLEAN",
            "expected_git_head": expected_head,
            "observed_git_head": observed_head,
        }
    return None


def _failure_payload(
    job: Mapping[str, Any],
    *,
    stage: str,
    reason: str,
    return_code: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_scheduler_failure.v1"
        ),
        "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
        "failure_stage": stage,
        "reason": reason,
        "job_key": str(job["job_key"]),
        "run_id": str(job["run_id"]),
        "physical_gpu_index": int(job["physical_gpu_index"]),
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    if "checkpoint_pass" in job:
        payload["checkpoint_pass"] = int(job["checkpoint_pass"])
    if return_code is not None:
        payload["return_code"] = int(return_code)
    if details is not None:
        payload["details"] = dict(details)
    return payload


def publish_missing_failure(
    job: Mapping[str, Any],
    *,
    stage: str,
    reason: str,
    return_code: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    summary = Path(str(job["terminal_summary"]))
    failure = Path(str(job["terminal_failure"]))
    if summary.exists() or failure.exists() or partial_terminal_artifacts(job):
        return
    write_new_terminal(
        failure,
        _failure_payload(
            job,
            stage=stage,
            reason=reason,
            return_code=return_code,
            details=details,
        ),
    )


def adjudication_terminal_kind(adjudication: Mapping[str, Any]) -> str:
    gate = Path(str(adjudication["gate_path"]))
    failure = Path(str(adjudication["failure_path"]))
    if gate.exists() and failure.exists():
        return "DOUBLE_TERMINAL"
    if gate.exists():
        return "GATE"
    if failure.exists():
        return "FAILURE"
    return "NONE"


def publish_adjudication_failure(
    adjudication: Mapping[str, Any],
    *,
    reason: str,
    return_code: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    failure = Path(str(adjudication["failure_path"]))
    if failure.exists() or failure.with_suffix(failure.suffix + ".partial").exists():
        return
    payload: dict[str, Any] = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_adjudication_failure.v1"
        ),
        "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
        "failure_stage": "ADJUDICATION",
        "reason": reason,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    if return_code is not None:
        payload["return_code"] = int(return_code)
    if details is not None:
        payload["details"] = dict(details)
    write_new_terminal(failure, payload)


def validate_schedule_inventory(schedule: Mapping[str, Any]) -> None:
    training = [
        job for queue in schedule["training_queues"] for job in queue["jobs"]
    ]
    validation = [
        job for queue in schedule["validation_queues"] for job in queue["jobs"]
    ]
    if len(training) != 2 or {str(job["run_id"]) for job in training} != set(
        RUN_IDS
    ):
        raise ValueError("S1 schedule must contain exactly two isolated trainings")
    expected_validations = {
        (run_id, checkpoint_pass)
        for run_id in RUN_IDS
        for checkpoint_pass in CHECKPOINT_PASSES
    }
    observed_validations = {
        (str(job["run_id"]), int(job["checkpoint_pass"])) for job in validation
    }
    if len(validation) != 8 or observed_validations != expected_validations:
        raise ValueError("S1 schedule must contain exactly eight checkpoint Validations")
    all_jobs = training + validation
    keys = [str(job["job_key"]) for job in all_jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("S1 schedule contains duplicate job keys")
    if any(int(job["physical_gpu_index"]) not in range(6) for job in all_jobs):
        raise ValueError("S1 schedule contains a GPU outside 0-5")


def run(schedule: dict[str, Any]) -> None:
    validate_schedule_inventory(schedule)
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    expected_head = str(schedule["git_head"])
    lock = threading.Lock()
    stop = threading.Event()
    first_terminal_failure: dict[str, Any] = {}

    def state_rows(queues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(job["job_key"]): {
                "run_id": str(job["run_id"]),
                **(
                    {"checkpoint_pass": int(job["checkpoint_pass"])}
                    if "checkpoint_pass" in job
                    else {}
                ),
                "physical_gpu_index": int(job["physical_gpu_index"]),
                "status": "PENDING",
                "terminal_summary": str(job["terminal_summary"]),
                "terminal_failure": str(job["terminal_failure"]),
                "log_path": str(job["log_path"]),
            }
            for queue in queues
            for job in queue["jobs"]
        }

    training_states = state_rows(schedule["training_queues"])
    validation_states = state_rows(schedule["validation_queues"])
    adjudication_state: dict[str, Any] = {
        "status": "PENDING",
        "gate_path": schedule["adjudication"]["gate_path"],
        "failure_path": schedule["adjudication"]["failure_path"],
        "log_path": schedule["adjudication"]["log_path"],
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": RUNTIME_SCHEMA,
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": expected_head,
                "objective_identity": schedule["objective_identity"],
                "cross_state_candidate_mode_responsibility_weight": schedule[
                    "cross_state_candidate_mode_responsibility_weight"
                ],
                "training_jobs": training_states,
                "validation_jobs": validation_states,
                "adjudication": adjudication_state,
                "first_terminal_failure": first_terminal_failure or None,
                "free_memory_gate_applied": False,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def record_failure(
        job: Mapping[str, Any],
        *,
        stage: str,
        reason: str,
        terminal: str | None,
        return_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "stage": stage,
                    "job_key": str(job["job_key"]),
                    "run_id": str(job["run_id"]),
                    "reason": reason,
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    **(
                        {"checkpoint_pass": int(job["checkpoint_pass"])}
                        if "checkpoint_pass" in job
                        else {}
                    ),
                    **({"details": dict(details)} if details else {}),
                }
            )
        stop.set()

    def mark_pending_not_run(states: dict[str, dict[str, Any]], reason: str) -> None:
        for row in states.values():
            if row["status"] == "PENDING":
                row.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": reason,
                    }
                )

    def record_adjudication_failure(
        *,
        reason: str,
        terminal: str,
        return_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "stage": "ADJUDICATION",
                    "job_key": "adjudication",
                    "reason": reason,
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    **({"details": dict(details)} if details else {}),
                }
            )
        stop.set()

    def run_phase(
        queues: list[dict[str, Any]],
        states: dict[str, dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        def run_queue(queue: dict[str, Any]) -> None:
            for job in list(queue["jobs"]):
                key = str(job["job_key"])
                process: subprocess.Popen[str] | None = None
                stream: Any = None
                with lock:
                    if stop.is_set():
                        mark_pending_not_run(states, "EARLIER_PACKAGE_TECHNICAL_FAILURE")
                        publish(RUNNING_STATUS)
                        return
                    inspection = inspect_worktree_identity(
                        worktree, expected_head=expected_head
                    )
                    if inspection is not None:
                        publish_missing_failure(
                            job,
                            stage=f"{stage}_SCHEDULER_PRELAUNCH",
                            reason=str(inspection["reason"]),
                            details=inspection,
                        )
                        terminal = exact_terminal(job)
                        states[key].update(
                            {
                                "status": "TERMINAL_COMPLETE",
                                "terminal_artifact_kind": terminal,
                                "finished_unix_seconds": time.time(),
                                "worktree_inspection": inspection,
                            }
                        )
                        record_failure(
                            job,
                            stage=stage,
                            reason=str(inspection["reason"]),
                            terminal=terminal,
                            details=inspection,
                        )
                        publish(RUNNING_STATUS)
                        return
                    log = Path(job["log_path"])
                    log.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        stream = log.open("w", encoding="utf-8")
                        process = subprocess.Popen(
                            list(job["command"]),
                            cwd=worktree,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                    except Exception as error:
                        if stream is not None:
                            stream.close()
                        details = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        publish_missing_failure(
                            job,
                            stage=f"{stage}_PROCESS_LAUNCH",
                            reason="JOB_PROCESS_LAUNCH_FAILED",
                            details=details,
                        )
                        terminal = exact_terminal(job)
                        states[key].update(
                            {
                                "status": "TERMINAL_COMPLETE",
                                "terminal_artifact_kind": terminal,
                                "finished_unix_seconds": time.time(),
                            }
                        )
                        record_failure(
                            job,
                            stage=stage,
                            reason="JOB_PROCESS_LAUNCH_FAILED",
                            terminal=terminal,
                            details=details,
                        )
                        publish(RUNNING_STATUS)
                        return
                    states[key].update(
                        {
                            "status": "RUNNING",
                            "process_pid": process.pid,
                            "started_unix_seconds": time.time(),
                        }
                    )
                    publish(RUNNING_STATUS)
                assert process is not None and stream is not None
                try:
                    return_code = process.wait()
                finally:
                    stream.close()
                terminal = exact_terminal(job)
                if terminal is None:
                    publish_missing_failure(
                        job,
                        stage=f"{stage}_WRAPPER",
                        reason="JOB_NO_EXACT_TERMINAL_ARTIFACT",
                        return_code=return_code,
                    )
                    terminal = exact_terminal(job)
                with lock:
                    successful = terminal == "SUMMARY" and return_code == 0
                    states[key].update(
                        {
                            "status": (
                                "TERMINAL_COMPLETE"
                                if terminal is not None
                                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                            ),
                            "return_code": return_code,
                            "terminal_artifact_kind": terminal,
                            "finished_unix_seconds": time.time(),
                        }
                    )
                    if not successful:
                        partials = partial_terminal_artifacts(job)
                        record_failure(
                            job,
                            stage=stage,
                            reason=(
                                "JOB_TERMINAL_FAILURE_ARTIFACT"
                                if terminal == "FAILURE"
                                else "JOB_NO_EXACT_SUCCESS_TERMINAL"
                            ),
                            terminal=terminal,
                            return_code=return_code,
                            details=(
                                {"partial_terminal_artifacts": partials}
                                if partials
                                else None
                            ),
                        )
                    publish(RUNNING_STATUS)

        threads = [
            threading.Thread(
                target=run_queue,
                args=(queue,),
                name=f"{stage.lower()}-gpu-{queue['physical_gpu_index']}",
            )
            for queue in queues
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    publish(RUNNING_STATUS)
    run_phase(schedule["training_queues"], training_states, stage="TRAINING")
    if stop.is_set():
        mark_pending_not_run(training_states, "TRAINING_TECHNICAL_FAILURE")
        mark_pending_not_run(validation_states, "TRAINING_TECHNICAL_FAILURE")
        adjudication_state["status"] = "NOT_RUN_AFTER_TERMINAL_FAILURE"
        publish(FAILURE_STATUS)
        return

    run_phase(schedule["validation_queues"], validation_states, stage="VALIDATION")
    if stop.is_set():
        mark_pending_not_run(validation_states, "VALIDATION_TECHNICAL_FAILURE")
        adjudication_state["status"] = "NOT_RUN_AFTER_TERMINAL_FAILURE"
        publish(FAILURE_STATUS)
        return

    adjudication = schedule["adjudication"]
    inspection = inspect_worktree_identity(worktree, expected_head=expected_head)
    if inspection is not None:
        publish_adjudication_failure(
            adjudication,
            reason=str(inspection["reason"]),
            details=inspection,
        )
        terminal = adjudication_terminal_kind(adjudication)
        adjudication_state.update(
            {
                "status": "TECHNICAL_FAILURE",
                "reason": str(inspection["reason"]),
                "worktree_inspection": inspection,
                "terminal_artifact_kind": terminal,
            }
        )
        record_adjudication_failure(
            reason=str(inspection["reason"]),
            terminal=terminal,
            details=inspection,
        )
        publish(FAILURE_STATUS)
        return
    log = Path(adjudication["log_path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                list(adjudication["command"]),
                cwd=worktree,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            adjudication_state.update(
                {"status": "RUNNING", "process_pid": process.pid}
            )
            publish(RUNNING_STATUS)
            return_code = process.wait()
    except Exception as error:
        details = {"error_type": type(error).__name__, "error": str(error)}
        publish_adjudication_failure(
            adjudication,
            reason="ADJUDICATION_PROCESS_LAUNCH_FAILED",
            details=details,
        )
        terminal = adjudication_terminal_kind(adjudication)
        adjudication_state.update(
            {
                "status": "TECHNICAL_FAILURE",
                "reason": "ADJUDICATION_PROCESS_LAUNCH_FAILED",
                "terminal_artifact_kind": terminal,
            }
        )
        record_adjudication_failure(
            reason="ADJUDICATION_PROCESS_LAUNCH_FAILED",
            terminal=terminal,
            details=details,
        )
        publish(FAILURE_STATUS)
        return
    gate = Path(adjudication["gate_path"])
    failure = Path(adjudication["failure_path"])
    gate_exact = gate.is_file() and not failure.exists() and return_code == 0
    if not gate_exact:
        if return_code != 0:
            reason = "ADJUDICATION_NONZERO_RETURN_CODE"
        elif gate.exists() and failure.exists():
            reason = "ADJUDICATION_DOUBLE_TERMINAL"
        elif failure.exists():
            reason = "ADJUDICATION_FAILURE_ARTIFACT"
        else:
            reason = "ADJUDICATION_NO_EXACT_TERMINAL_ARTIFACT"
        publish_adjudication_failure(
            adjudication,
            reason=reason,
            return_code=return_code,
        )
        terminal = adjudication_terminal_kind(adjudication)
        record_adjudication_failure(
            reason=reason,
            terminal=terminal,
            return_code=return_code,
        )
    else:
        terminal = "GATE"
    adjudication_state.update(
        {
            "status": "TERMINAL_COMPLETE" if gate_exact else "TECHNICAL_FAILURE",
            "return_code": return_code,
            "gate_present": gate.is_file(),
            "failure_present": failure.exists(),
            "terminal_artifact_kind": terminal,
        }
    )
    publish(TERMINAL_STATUS if gate_exact else FAILURE_STATUS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
