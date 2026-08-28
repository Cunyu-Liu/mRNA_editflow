#!/usr/bin/env python3
"""Run the frozen V4 confirmation training queues terminal-only."""

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


def terminal_kind(job: dict[str, Any]) -> str | None:
    output = Path(job["output_directory"])
    summary_name = (
        "run_summary.json" if job["component"] == "critic" else "training_summary.json"
    )
    summary = output / summary_name
    failure = output / "failure.json"
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


def publish_scheduler_prelaunch_failure(
    job: dict[str, Any], *, inspection: dict[str, Any]
) -> str | None:
    output = Path(job["output_directory"])
    if output.exists():
        return terminal_kind(job)
    write_atomic(
        output / "failure.json",
        {
            "schema_version": (
                "route_a_v3_route2_xedit_v4_confirmation_training_scheduler_failure.v1"
            ),
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "failure_stage": "CONFIRMATION_TRAINING_SCHEDULER_PRELAUNCH",
            "component": job["component"],
            "run_id": job["run_id"],
            "seed": int(job["training_seed"]),
            "cpu_fallback_used": False,
            "worktree_inspection": inspection,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return terminal_kind(job)


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    for queue in schedule["gpu_queues"]:
        for job in queue["jobs"]:
            states[job["job_key"]] = {
                "component": job["component"],
                "run_id": job["run_id"],
                "training_seed": int(job["training_seed"]),
                "physical_gpu_index": int(queue["physical_gpu_index"]),
                "status": "PENDING",
                "output_directory": job["output_directory"],
                "log_path": job["log_path"],
            }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_confirmation_training_runtime.v1",
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "experiment_head": schedule["experiment_head"],
                "eligible_components": schedule["eligible_components"],
                "jobs": states,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")

    def record_terminal_failure(
        *,
        key: str,
        job: dict[str, Any],
        reason: str,
        terminal: str | None,
        return_code: int | None = None,
        inspection: dict[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "job_key": key,
                    "component": job["component"],
                    "run_id": job["run_id"],
                    "training_seed": int(job["training_seed"]),
                    "reason": reason,
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    "output_directory": job["output_directory"],
                    "log_path": job["log_path"],
                    "worktree_inspection": inspection,
                }
            )
        terminal_failure.set()

    def mark_not_run(jobs: list[dict[str, Any]], *, reason: str) -> None:
        for skipped in jobs:
            states[str(skipped["job_key"])].update(
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
            output = Path(job["output_directory"])
            log = Path(job["log_path"])
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            stream = None
            with lock:
                if terminal_failure.is_set():
                    mark_not_run(
                        jobs[index:], reason="EARLIER_CONFIRMATION_JOB_TECHNICAL_FAILURE"
                    )
                    publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")
                    return
                if output.exists():
                    terminal = terminal_kind(job)
                    states[key]["status"] = "TECHNICAL_FAILURE_PREEXISTING_OUTPUT"
                    states[key]["terminal_artifact_kind"] = terminal
                    states[key]["finished_unix_seconds"] = time.time()
                    record_terminal_failure(
                        key=key,
                        job=job,
                        reason="PREEXISTING_OUTPUT_DIRECTORY",
                        terminal=terminal,
                    )
                    mark_not_run(
                        jobs[index + 1 :],
                        reason="EARLIER_CONFIRMATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")
                    return
                inspection = inspect_worktree_identity(
                    worktree, expected_head=str(schedule["git_head"])
                )
                if inspection is not None:
                    terminal = publish_scheduler_prelaunch_failure(
                        job, inspection=inspection
                    )
                    states[key].update(
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
                        key=key,
                        job=job,
                        reason=str(inspection["reason"]),
                        terminal=terminal,
                        inspection=inspection,
                    )
                    mark_not_run(
                        jobs[index + 1 :],
                        reason="EARLIER_CONFIRMATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")
                    return
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
                    inspection = {
                        "reason": "JOB_PROCESS_LAUNCH_FAILED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    terminal = publish_scheduler_prelaunch_failure(
                        job, inspection=inspection
                    )
                    states[key].update(
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
                        key=key,
                        job=job,
                        reason="JOB_PROCESS_LAUNCH_FAILED",
                        terminal=terminal,
                        inspection=inspection,
                    )
                    mark_not_run(
                        jobs[index + 1 :],
                        reason="EARLIER_CONFIRMATION_JOB_TECHNICAL_FAILURE",
                    )
                    publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")
                    return
                states[key].update(
                    {
                        "status": "RUNNING",
                        "training_pid": process.pid,
                        "started_unix_seconds": started,
                    }
                )
                publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")
            try:
                return_code = process.wait()
            finally:
                assert stream is not None
                stream.close()
            kind = terminal_kind(job)
            with lock:
                successful = kind == "SUMMARY" and return_code == 0
                states[key].update(
                    {
                        "status": (
                            "TECHNICAL_FAILURE_NONZERO_RETURN_CODE"
                            if kind == "SUMMARY" and return_code != 0
                            else (
                                "TERMINAL_COMPLETE"
                                if kind is not None
                                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                            )
                        ),
                        "return_code": return_code,
                        "terminal_artifact_kind": kind,
                        "finished_unix_seconds": time.time(),
                    }
                )
                if not successful:
                    record_terminal_failure(
                        key=key,
                        job=job,
                        reason=(
                            "JOB_NONZERO_RETURN_CODE"
                            if kind == "SUMMARY" and return_code != 0
                            else (
                                "JOB_TERMINAL_FAILURE_ARTIFACT"
                                if kind == "FAILURE"
                                else "JOB_NO_EXACT_TERMINAL_ARTIFACT"
                            )
                        ),
                        terminal=kind,
                        return_code=int(return_code),
                    )
                publish("V4_CONFIRMATION_TRAINING_SCHEDULER_RUNNING")

    threads = [
        threading.Thread(
            target=run_queue,
            args=(queue,),
            name=f"gpu-{queue['physical_gpu_index']}",
        )
        for queue in schedule["gpu_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if terminal_failure.is_set():
        with lock:
            mark_not_run(
                [
                    {"job_key": key}
                    for key, state in states.items()
                    if state.get("status") == "PENDING"
                ],
                reason="EARLIER_CONFIRMATION_JOB_TECHNICAL_FAILURE",
            )
        publish("V4_CONFIRMATION_TRAINING_TECHNICAL_FAILURE")
        return

    exact_terminal = bool(states) and all(
        row.get("terminal_artifact_kind") == "SUMMARY"
        and row.get("return_code") == 0
        for row in states.values()
    )
    publish(
        "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
        if exact_terminal
        else "V4_CONFIRMATION_TRAINING_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
