#!/usr/bin/env python3
"""Validate terminal SetFlow checkpoints and adjudicate V4 confirmations once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def exact_terminal(summary_path: str, failure_path: str) -> str | None:
    summary = Path(summary_path)
    failure = Path(failure_path)
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


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


def start_logged(
    command: list[str], *, cwd: Path, log: Path
) -> tuple[subprocess.Popen[str], Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        stream.close()
        raise
    return process, stream


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    process, stream = start_logged(command, cwd=cwd, log=log)
    try:
        return process.wait()
    finally:
        stream.close()


def validation_run_id(job: dict[str, Any]) -> str:
    run_id = job.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    command = [str(value) for value in job["command"]]
    try:
        index = command.index("--run-id")
        return command[index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError("validation schedule job has no run_id") from error


def publish_missing_validation_failure(
    job: dict[str, Any],
    *,
    return_code: int | None,
    failure_stage: str = "CHECKPOINT_VALIDATION_WRAPPER",
    inspection: dict[str, Any] | None = None,
) -> None:
    summary = Path(job["terminal_summary"])
    failure = Path(job["terminal_failure"])
    if summary.exists() or failure.exists():
        return
    payload: dict[str, Any] = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_checkpoint_validation_failure.v1"
        ),
        "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
        "stage": failure_stage,
        "run_id": validation_run_id(job),
        "run_stage": "CONFIRMATION",
        "seed": int(job["training_seed"]),
        "checkpoint_pass": int(job["checkpoint_pass"]),
        "physical_gpu_index": int(job["physical_gpu_index"]),
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    if return_code is not None:
        payload["return_code"] = int(return_code)
    if inspection is not None:
        payload["worktree_inspection"] = inspection
    write_atomic(failure, payload)


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    validation_states = {
        job["job_key"]: {
            "run_id": validation_run_id(job),
            "training_seed": int(job["training_seed"]),
            "checkpoint_pass": int(job["checkpoint_pass"]),
            "physical_gpu_index": int(queue["physical_gpu_index"]),
            "status": "PENDING",
            "terminal_summary": job["terminal_summary"],
            "terminal_failure": job["terminal_failure"],
            "log_path": job["log_path"],
        }
        for queue in schedule["validation_queues"]
        for job in queue["jobs"]
    }
    adjudication_states = {
        component: {
            "status": "PENDING",
            "gate_path": spec["gate_path"],
            "failure_path": spec["failure_path"],
            "log_path": spec["log_path"],
        }
        for component, spec in schedule["adjudications"].items()
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1"
                ),
                "status": status,
                "coordinator_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "eligible_components": schedule["eligible_components"],
                "validation_jobs": validation_states,
                "adjudications": adjudication_states,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")

    def record_terminal_failure(
        *,
        stage: str,
        key: str,
        reason: str,
        terminal: str | None,
        return_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "stage": stage,
                    "job_key": key,
                    "reason": reason,
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    **(details or {}),
                }
            )
        terminal_failure.set()

    def mark_validation_not_run(
        jobs: list[dict[str, Any]], *, reason: str
    ) -> None:
        for skipped in jobs:
            validation_states[str(skipped["job_key"])].update(
                {
                    "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                    "terminal_artifact_kind": None,
                    "stop_reason": reason,
                }
            )

    def mark_adjudications_not_run(
        components: list[str], *, reason: str
    ) -> None:
        for component in components:
            adjudication_states[component].update(
                {
                    "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                    "terminal_artifact_kind": None,
                    "stop_reason": reason,
                }
            )

    def run_queue(queue: dict[str, Any]) -> None:
        jobs = list(queue["jobs"])
        for index, job in enumerate(jobs):
            key = str(job["job_key"])
            process = None
            stream = None
            with lock:
                if terminal_failure.is_set():
                    mark_validation_not_run(
                        jobs[index:],
                        reason="EARLIER_VALIDATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
                    return
                inspection = inspect_worktree_identity(
                    worktree, expected_head=str(schedule["git_head"])
                )
                if inspection is not None:
                    publish_missing_validation_failure(
                        job,
                        return_code=None,
                        failure_stage="CHECKPOINT_VALIDATION_SCHEDULER_PRELAUNCH",
                        inspection=inspection,
                    )
                    terminal = exact_terminal(
                        job["terminal_summary"], job["terminal_failure"]
                    )
                    validation_states[key].update(
                        {
                            "status": (
                                "TERMINAL_COMPLETE"
                                if terminal is not None
                                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                            ),
                            "terminal_artifact_kind": terminal,
                            "finished_unix_seconds": time.time(),
                            "worktree_inspection": inspection,
                        }
                    )
                    record_terminal_failure(
                        stage="VALIDATION",
                        key=key,
                        reason=str(inspection["reason"]),
                        terminal=terminal,
                        details={
                            "training_seed": int(job["training_seed"]),
                            "checkpoint_pass": int(job["checkpoint_pass"]),
                            "worktree_inspection": inspection,
                        },
                    )
                    mark_validation_not_run(
                        jobs[index + 1 :],
                        reason="EARLIER_VALIDATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
                    return
                try:
                    process, stream = start_logged(
                        list(job["command"]),
                        cwd=worktree,
                        log=Path(job["log_path"]),
                    )
                except Exception as error:
                    inspection = {
                        "reason": "JOB_PROCESS_LAUNCH_FAILED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    publish_missing_validation_failure(
                        job,
                        return_code=None,
                        failure_stage="CHECKPOINT_VALIDATION_SCHEDULER_PRELAUNCH",
                        inspection=inspection,
                    )
                    terminal = exact_terminal(
                        job["terminal_summary"], job["terminal_failure"]
                    )
                    validation_states[key].update(
                        {
                            "status": (
                                "TERMINAL_COMPLETE"
                                if terminal is not None
                                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                            ),
                            "terminal_artifact_kind": terminal,
                            "finished_unix_seconds": time.time(),
                            "worktree_inspection": inspection,
                        }
                    )
                    record_terminal_failure(
                        stage="VALIDATION",
                        key=key,
                        reason="JOB_PROCESS_LAUNCH_FAILED",
                        terminal=terminal,
                        details={
                            "training_seed": int(job["training_seed"]),
                            "checkpoint_pass": int(job["checkpoint_pass"]),
                            "worktree_inspection": inspection,
                        },
                    )
                    mark_validation_not_run(
                        jobs[index + 1 :],
                        reason="EARLIER_VALIDATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
                    return
                validation_states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
            assert process is not None and stream is not None
            try:
                return_code = process.wait()
            finally:
                stream.close()
            terminal = exact_terminal(
                job["terminal_summary"], job["terminal_failure"]
            )
            if terminal is None:
                publish_missing_validation_failure(job, return_code=return_code)
                terminal = exact_terminal(
                    job["terminal_summary"], job["terminal_failure"]
                )
            with lock:
                successful = terminal == "SUMMARY" and return_code == 0
                validation_states[key].update(
                    {
                        "status": (
                            "TECHNICAL_FAILURE_NONZERO_RETURN_CODE"
                            if terminal == "SUMMARY" and return_code != 0
                            else (
                                "TERMINAL_COMPLETE"
                                if terminal is not None
                                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                            )
                        ),
                        "return_code": return_code,
                        "terminal_artifact_kind": terminal,
                        "finished_unix_seconds": time.time(),
                    }
                )
                if not successful:
                    record_terminal_failure(
                        stage="VALIDATION",
                        key=key,
                        reason=(
                            "VALIDATION_NONZERO_RETURN_CODE"
                            if terminal == "SUMMARY" and return_code != 0
                            else (
                                "VALIDATION_TERMINAL_FAILURE_ARTIFACT"
                                if terminal == "FAILURE"
                                else "VALIDATION_NO_EXACT_TERMINAL_ARTIFACT"
                            )
                        ),
                        terminal=terminal,
                        return_code=int(return_code),
                        details={
                            "training_seed": int(job["training_seed"]),
                            "checkpoint_pass": int(job["checkpoint_pass"]),
                        },
                    )
                publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")

    threads = [
        threading.Thread(
            target=run_queue,
            args=(queue,),
            name=f"gpu-{queue['physical_gpu_index']}",
        )
        for queue in schedule["validation_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if terminal_failure.is_set():
        mark_validation_not_run(
            [
                {"job_key": key}
                for key, state in validation_states.items()
                if state.get("status") == "PENDING"
            ],
            reason="EARLIER_VALIDATION_JOB_TECHNICAL_FAILURE",
        )
        mark_adjudications_not_run(
            list(schedule["eligible_components"]),
            reason="VALIDATION_TECHNICAL_FAILURE",
        )
        publish("V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE")
        return

    components = list(schedule["eligible_components"])
    for index, component in enumerate(components):
        spec = schedule["adjudications"][component]
        inspection = inspect_worktree_identity(
            worktree, expected_head=str(schedule["git_head"])
        )
        if inspection is not None:
            gate = Path(spec["gate_path"])
            failure = Path(spec["failure_path"])
            if not gate.exists() and not failure.exists():
                write_atomic(
                    failure,
                    {
                        "schema_version": (
                            "route_a_v3_route2_xedit_v4_confirmation_adjudication_failure.v1"
                        ),
                        "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                        "failure_stage": "CONFIRMATION_ADJUDICATION_SCHEDULER_PRELAUNCH",
                        "component": component,
                        "worktree_inspection": inspection,
                        "development_test_outcome_reads": 0,
                        "new_final_evaluation_outcome_reads": 0,
                    },
                )
            terminal = exact_terminal(str(gate), str(failure))
            adjudication_states[component].update(
                {
                    "status": (
                        "TERMINAL_COMPLETE"
                        if terminal is not None
                        else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                    ),
                    "terminal_artifact_kind": terminal,
                    "finished_unix_seconds": time.time(),
                    "worktree_inspection": inspection,
                }
            )
            record_terminal_failure(
                stage="ADJUDICATION",
                key=component,
                reason=str(inspection["reason"]),
                terminal=terminal,
                details={
                    "component": component,
                    "worktree_inspection": inspection,
                },
            )
            mark_adjudications_not_run(
                components[index + 1 :],
                reason="EARLIER_ADJUDICATION_TECHNICAL_FAILURE",
            )
            publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
            break
        adjudication_states[component]["status"] = "RUNNING"
        adjudication_states[component]["started_unix_seconds"] = time.time()
        publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
        launch_error: Exception | None = None
        try:
            return_code: int | None = run_logged(
                list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
            )
        except Exception as error:
            return_code = None
            launch_error = error
        gate = Path(spec["gate_path"])
        failure = Path(spec["failure_path"])
        if not gate.exists() and not failure.exists():
            failure_payload: dict[str, Any] = {
                "schema_version": (
                    "route_a_v3_route2_xedit_v4_confirmation_adjudication_failure.v1"
                ),
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "component": component,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
            if return_code is not None:
                failure_payload["return_code"] = int(return_code)
            if launch_error is not None:
                failure_payload.update(
                    {
                        "failure_stage": "CONFIRMATION_ADJUDICATION_SCHEDULER_PRELAUNCH",
                        "error_type": type(launch_error).__name__,
                        "error": str(launch_error),
                    }
                )
            write_atomic(failure, failure_payload)
        terminal = exact_terminal(str(gate), str(failure))
        successful = terminal == "SUMMARY" and return_code == 0
        adjudication_states[component].update(
            {
                "status": (
                    "TECHNICAL_FAILURE_NONZERO_RETURN_CODE"
                    if terminal == "SUMMARY" and return_code != 0
                    else (
                        "TERMINAL_COMPLETE"
                        if terminal is not None
                        else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                    )
                ),
                "return_code": return_code,
                "terminal_artifact_kind": terminal,
                "finished_unix_seconds": time.time(),
            }
        )
        if not successful:
            record_terminal_failure(
                stage="ADJUDICATION",
                key=component,
                reason=(
                    "ADJUDICATION_NONZERO_RETURN_CODE"
                    if terminal == "SUMMARY" and return_code is not None
                    else (
                        "ADJUDICATION_PROCESS_ERROR"
                        if terminal == "SUMMARY"
                        else (
                            "ADJUDICATION_TERMINAL_FAILURE_ARTIFACT"
                            if terminal == "FAILURE"
                            else "ADJUDICATION_NO_EXACT_TERMINAL_ARTIFACT"
                        )
                    )
                ),
                terminal=terminal,
                return_code=return_code,
                details={"component": component},
            )
            mark_adjudications_not_run(
                components[index + 1 :],
                reason="EARLIER_ADJUDICATION_TECHNICAL_FAILURE",
            )
        publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
        if terminal_failure.is_set():
            break

    exact_validations = all(
        row.get("terminal_artifact_kind") == "SUMMARY"
        and row.get("return_code") == 0
        for row in validation_states.values()
    )
    exact_adjudications = bool(adjudication_states) and all(
        row.get("terminal_artifact_kind") == "SUMMARY"
        and row.get("return_code") == 0
        for row in adjudication_states.values()
    )
    publish(
        "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
        if exact_validations and exact_adjudications
        else "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
