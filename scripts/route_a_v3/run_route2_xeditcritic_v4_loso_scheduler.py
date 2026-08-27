#!/usr/bin/env python3
"""Run the frozen 42-job Critic V4 LOSO package and compose readiness."""

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


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.wait()


def close_missing_job(job: dict[str, Any], *, return_code: int) -> str | None:
    terminal = exact_terminal(job["summary_path"], job["failure_path"])
    if terminal is not None:
        return terminal
    summary = Path(job["summary_path"])
    failure = Path(job["failure_path"])
    if not summary.exists() and not failure.exists():
        write_atomic(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_run_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_stage": "LOSO",
                "run_id": job["run_id"],
                "seed": int(job["seed"]),
                "held_out_study": job["held_out_study"],
                "return_code": int(return_code),
                "failure_stage": "FORMAL_LOSO_SCHEDULER",
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    return exact_terminal(job["summary_path"], job["failure_path"])


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states = {
        job["job_key"]: {
            "seed": int(job["seed"]),
            "held_out_study": job["held_out_study"],
            "run_id": job["run_id"],
            "physical_gpu_index": int(queue["physical_gpu_index"]),
            "status": "PENDING",
            "summary_path": job["summary_path"],
            "failure_path": job["failure_path"],
            "log_path": job["log_path"],
        }
        for queue in schedule["gpu_queues"]
        for job in queue["jobs"]
    }
    adjudication: dict[str, Any] = {
        "status": "PENDING",
        "summary_path": schedule["loso_adjudication"]["summary_path"],
        "failure_path": schedule["loso_adjudication"]["failure_path"],
    }
    readiness: dict[str, Any] = {
        "status": "PENDING",
        "summary_path": schedule["readiness"]["summary_path"],
        "failure_path": schedule["readiness"]["failure_path"],
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1",
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "jobs": states,
                "loso_adjudication": adjudication,
                "readiness": readiness,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_access_event_count_before_loso": 1,
                "development_test_outcome_reads_during_loso": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        jobs = list(queue["jobs"])
        for index, job in enumerate(jobs):
            key = str(job["job_key"])
            with lock:
                if terminal_failure.is_set():
                    for skipped in jobs[index:]:
                        states[str(skipped["job_key"])].update(
                            {
                                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                                "terminal_artifact_kind": None,
                                "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                            }
                        )
                    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
                    return
                states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
            return_code = run_logged(
                list(job["command"]), cwd=worktree, log=Path(job["log_path"])
            )
            terminal = close_missing_job(job, return_code=return_code)
            with lock:
                status = (
                    "TERMINAL_COMPLETE"
                    if terminal is not None
                    else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                )
                states[key].update(
                    {
                        "status": status,
                        "return_code": int(return_code),
                        "terminal_artifact_kind": terminal,
                        "finished_unix_seconds": time.time(),
                    }
                )
                if terminal != "SUMMARY":
                    if not first_terminal_failure:
                        first_terminal_failure.update(
                            {
                                "job_key": key,
                                "seed": int(job["seed"]),
                                "held_out_study": job["held_out_study"],
                                "run_id": job["run_id"],
                                "return_code": int(return_code),
                                "terminal_artifact_kind": terminal,
                                "summary_path": job["summary_path"],
                                "failure_path": job["failure_path"],
                                "log_path": job["log_path"],
                            }
                        )
                    terminal_failure.set()
                publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

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
        for state in states.values():
            if state.get("status") == "PENDING":
                state.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                    }
                )
        adjudication.update(
            {
                "status": "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
            }
        )
        readiness.update(
            {
                "status": "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )
        publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
        return

    spec = schedule["loso_adjudication"]
    adjudication["status"] = "RUNNING"
    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
    return_code = run_logged(
        list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
    )
    summary = Path(spec["summary_path"])
    failure = Path(spec["failure_path"])
    if not summary.exists() and not failure.exists():
        write_atomic(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_adjudication_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "return_code": int(return_code),
                "development_test_outcome_reads_during_loso": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    terminal = exact_terminal(str(summary), str(failure))
    adjudication.update(
        {
            "status": "TERMINAL_COMPLETE" if terminal is not None else "TECHNICAL_FAILURE",
            "return_code": int(return_code),
            "terminal_artifact_kind": terminal,
            "finished_unix_seconds": time.time(),
        }
    )
    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

    if terminal == "SUMMARY":
        spec = schedule["readiness"]
        readiness["status"] = "RUNNING"
        publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
        return_code = run_logged(
            list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
        )
        summary = Path(spec["summary_path"])
        failure = Path(spec["failure_path"])
        if not summary.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": "route_a_v3_route2_xeditcritic_v4_readiness_failure.v1",
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "return_code": int(return_code),
                    "development_test_outcome_reads_after_atomic_test": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        readiness_terminal = exact_terminal(str(summary), str(failure))
        readiness_status = None
        guidance_authorized = False
        if readiness_terminal == "SUMMARY":
            payload = json.loads(summary.read_text(encoding="utf-8"))
            readiness_status = payload.get("status")
            guidance_authorized = payload.get("guidance_authorized") is True
        readiness.update(
            {
                "status": (
                    "TERMINAL_COMPLETE"
                    if readiness_terminal is not None
                    else "TECHNICAL_FAILURE"
                ),
                "return_code": int(return_code),
                "terminal_artifact_kind": readiness_terminal,
                "readiness_status": readiness_status,
                "guidance_authorized": guidance_authorized,
                "finished_unix_seconds": time.time(),
            }
        )
    else:
        readiness.update(
            {
                "status": "NOT_RUN_LOSO_ADJUDICATION_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )

    exact_jobs = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    publish(
        "CRITIC_V4_READY_FOR_GUIDANCE"
        if exact_jobs and readiness.get("guidance_authorized") is True
        else "CRITIC_V4_NOT_READY_FOR_GUIDANCE"
        if exact_jobs
        and readiness.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        else "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
