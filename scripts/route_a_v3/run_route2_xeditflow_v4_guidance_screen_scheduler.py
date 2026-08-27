#!/usr/bin/env python3
"""Run the authorized V4 value pipeline and exact 18-combination screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def start_logged(
    command: list[str], *, cwd: Path, log: Path
) -> tuple[subprocess.Popen[str], TextIO]:
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
    except OSError:
        stream.close()
        raise
    return process, stream


def wait_logged(process: subprocess.Popen[str], stream: TextIO) -> int:
    try:
        return process.wait()
    finally:
        stream.close()


def inspect_worktree(worktree: Path, expected_head: str) -> dict[str, Any]:
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        return {
            "exact_clean_head": False,
            "expected_git_head": expected_head,
            "observed_git_head": None,
            "git_status_porcelain": None,
            "git_head_return_code": None,
            "git_status_return_code": None,
            "inspection_error": f"{type(error).__name__}: {error}",
        }
    observed_head = head_result.stdout.strip() if head_result.returncode == 0 else None
    porcelain = status_result.stdout
    return {
        "exact_clean_head": (
            observed_head == expected_head
            and status_result.returncode == 0
            and not porcelain.strip()
        ),
        "expected_git_head": expected_head,
        "observed_git_head": observed_head,
        "git_status_porcelain": porcelain,
        "git_head_return_code": int(head_result.returncode),
        "git_status_return_code": int(status_result.returncode),
        "git_head_stderr": head_result.stderr,
        "git_status_stderr": status_result.stderr,
        "inspection_error": None,
    }


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    for job in schedule["serial_value_prerequisites"]:
        states[job["job_key"]] = {
            "status": "PENDING",
            "success_path": job["success_path"],
            "failure_path": job["failure_path"],
            "log_path": job["log_path"],
        }
    for queue in schedule["value_training_queues"]:
        for job in queue["jobs"]:
            states[job["job_key"]] = {
                "status": "PENDING",
                "physical_gpu_index": int(queue["physical_gpu_index"]),
                "success_path": job["success_path"],
                "failure_path": job["failure_path"],
                "log_path": job["log_path"],
            }
    for queue in schedule["guidance_queues"]:
        for chain in queue["chains"]:
            for job in chain["jobs"]:
                states[job["job_key"]] = {
                    "status": "PENDING",
                    "combination_id": chain["combination_id"],
                    "physical_gpu_index": int(queue["physical_gpu_index"]),
                    "success_path": job["success_path"],
                    "failure_path": job["failure_path"],
                    "log_path": job["log_path"],
                }
    adjudication = {
        "status": "PENDING",
        "success_path": schedule["adjudication"]["success_path"],
        "failure_path": schedule["adjudication"]["failure_path"],
        "log_path": schedule["adjudication"]["log_path"],
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v4_guidance_screen_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "experiment_head": schedule["experiment_head"],
                "jobs": states,
                "adjudication": adjudication,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_reopened": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def close_missing(job: dict[str, Any], return_code: int) -> str:
        success = Path(job["success_path"])
        failure = Path(job["failure_path"])
        if success.exists() and not failure.exists():
            return "SUCCESS"
        if failure.exists() and not success.exists():
            return "FAILURE"
        if not success.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_v4_guidance_screen_job_failure.v1"
                    ),
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "job_key": job["job_key"],
                    "return_code": int(return_code),
                    "development_test_outcomes_accessed_after_atomic_test": False,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
            return "FAILURE"
        return "DOUBLE_TERMINAL_FAILURE"

    def record_first_failure(
        job: dict[str, Any],
        *,
        return_code: int | None,
        terminal: str,
        failure_reason: str,
        worktree_evidence: dict[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "job_key": str(job["job_key"]),
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    "failure_reason": failure_reason,
                    "success_path": job["success_path"],
                    "failure_path": job["failure_path"],
                    "log_path": job["log_path"],
                    "worktree_evidence": worktree_evidence,
                }
            )

    def write_worktree_failure(
        job: dict[str, Any], evidence: dict[str, Any]
    ) -> str:
        failure = Path(job["failure_path"])
        if not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_v4_guidance_screen_job_failure.v1"
                    ),
                    "status": "TERMINAL_WORKTREE_STATE_FAILURE",
                    "job_key": job["job_key"],
                    "failure_stage": "PRE_POPEN_EXACT_CLEAN_HEAD",
                    "process_started": False,
                    **evidence,
                    "development_test_outcomes_accessed_after_atomic_test": False,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        success = Path(job["success_path"])
        if success.exists() and failure.exists():
            return "DOUBLE_TERMINAL_FAILURE"
        return "FAILURE" if failure.exists() else "NON_EXACT_TERMINAL_FAILURE"

    def write_process_start_failure(
        job: dict[str, Any], error: OSError
    ) -> str:
        failure = Path(job["failure_path"])
        if not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_v4_guidance_screen_job_failure.v1"
                    ),
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "job_key": job["job_key"],
                    "failure_stage": "PRE_POPEN_PROCESS_START",
                    "process_started": False,
                    "error": f"{type(error).__name__}: {error}",
                    "development_test_outcomes_accessed_after_atomic_test": False,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        success = Path(job["success_path"])
        if success.exists() and failure.exists():
            return "DOUBLE_TERMINAL_FAILURE"
        return "FAILURE" if failure.exists() else "NON_EXACT_TERMINAL_FAILURE"

    def execute(job: dict[str, Any], state: dict[str, Any]) -> bool:
        with lock:
            if terminal_failure.is_set():
                state.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
                    }
                )
                publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
                return False
            worktree_evidence = inspect_worktree(
                worktree, str(schedule["git_head"])
            )
            if not worktree_evidence["exact_clean_head"]:
                terminal = write_worktree_failure(job, worktree_evidence)
                state.update(
                    {
                        "status": "TERMINAL_FAILURE",
                        "return_code": None,
                        "terminal_artifact_kind": terminal,
                        "process_started": False,
                        "finished_unix_seconds": time.time(),
                    }
                )
                terminal_failure.set()
                record_first_failure(
                    job,
                    return_code=None,
                    terminal=terminal,
                    failure_reason="WORKTREE_NOT_AT_EXACT_CLEAN_SCHEDULE_HEAD",
                    worktree_evidence=worktree_evidence,
                )
                publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
                return False
            try:
                process, stream = start_logged(
                    list(job["command"]),
                    cwd=worktree,
                    log=Path(job["log_path"]),
                )
            except OSError as error:
                terminal = write_process_start_failure(job, error)
                state.update(
                    {
                        "status": "TERMINAL_FAILURE",
                        "return_code": None,
                        "terminal_artifact_kind": terminal,
                        "process_started": False,
                        "finished_unix_seconds": time.time(),
                    }
                )
                terminal_failure.set()
                record_first_failure(
                    job,
                    return_code=None,
                    terminal=terminal,
                    failure_reason="PROCESS_START_FAILURE",
                )
                publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
                return False
            state.update(
                {
                    "status": "RUNNING",
                    "process_started": True,
                    "started_unix_seconds": time.time(),
                }
            )
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
        return_code = wait_logged(process, stream)
        with lock:
            terminal = close_missing(job, return_code)
            if terminal != "SUCCESS":
                terminal_failure.set()
            state.update(
                {
                    "status": (
                        "TERMINAL_COMPLETE"
                        if terminal == "SUCCESS"
                        else "TERMINAL_FAILURE"
                        if terminal == "FAILURE"
                        else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                    ),
                    "return_code": int(return_code),
                    "terminal_artifact_kind": terminal,
                    "process_started": True,
                    "finished_unix_seconds": time.time(),
                }
            )
            if terminal != "SUCCESS":
                record_first_failure(
                    job,
                    return_code=int(return_code),
                    terminal=terminal,
                    failure_reason=(
                        "JOB_FAILURE_ARTIFACT"
                        if terminal == "FAILURE"
                        else "NON_EXACT_TERMINAL_ARTIFACTS"
                    ),
                )
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
        return terminal == "SUCCESS"

    def stop_pending(status: str, stop_reason: str) -> None:
        for row in states.values():
            if row["status"] == "PENDING":
                row.update(
                    {
                        "status": status,
                        "terminal_artifact_kind": None,
                        "stop_reason": stop_reason,
                    }
                )

    def stop_adjudication(status: str, stop_reason: str) -> None:
        if adjudication["status"] == "PENDING":
            adjudication.update(
                {
                    "status": status,
                    "terminal_artifact_kind": None,
                    "stop_reason": stop_reason,
                }
            )

    publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
    for job in schedule["serial_value_prerequisites"]:
        if not execute(job, states[str(job["job_key"])]):
            stop_pending(
                "NOT_RUN_AFTER_TERMINAL_FAILURE",
                "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
            )
            stop_adjudication(
                "NOT_RUN_AFTER_TERMINAL_FAILURE",
                "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
            )
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
            return

    value_results: dict[str, bool] = {}

    def run_value_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            result = execute(job, states[str(job["job_key"])])
            with lock:
                value_results[job["job_key"]] = result
            if not result:
                return

    value_threads = [
        threading.Thread(target=run_value_queue, args=(queue,))
        for queue in schedule["value_training_queues"]
    ]
    for thread in value_threads:
        thread.start()
    for thread in value_threads:
        thread.join()
    if terminal_failure.is_set() or not value_results or not all(value_results.values()):
        stop_pending(
            "NOT_RUN_AFTER_TERMINAL_FAILURE",
            "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
        )
        stop_adjudication(
            "NOT_RUN_AFTER_TERMINAL_FAILURE",
            "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
        )
        publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
        return

    chain_results: dict[str, bool] = {}

    def run_guidance_queue(queue: dict[str, Any]) -> None:
        for chain in queue["chains"]:
            complete = True
            for job in chain["jobs"]:
                if complete:
                    complete = execute(job, states[str(job["job_key"])])
                if not complete:
                    return
            chain_results[chain["combination_id"]] = complete

    guidance_threads = [
        threading.Thread(target=run_guidance_queue, args=(queue,))
        for queue in schedule["guidance_queues"]
    ]
    for thread in guidance_threads:
        thread.start()
    for thread in guidance_threads:
        thread.join()
    if (
        terminal_failure.is_set()
        or len(chain_results) != 18
        or not all(chain_results.values())
    ):
        stop_pending(
            "NOT_RUN_AFTER_TERMINAL_FAILURE",
            "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
        )
        stop_adjudication(
            "NOT_RUN_AFTER_TERMINAL_FAILURE",
            "EARLIER_GUIDANCE_SCREEN_JOB_TECHNICAL_FAILURE",
        )
        publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
        return

    spec = schedule["adjudication"]
    succeeded = execute(spec, adjudication)
    publish(
        "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        if succeeded
        else "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
