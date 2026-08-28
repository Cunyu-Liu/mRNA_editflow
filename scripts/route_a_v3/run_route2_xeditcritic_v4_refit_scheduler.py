#!/usr/bin/env python3
"""Run three Critic V4 refits and publish their single terminal adjudication."""

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


def inspect_worktree_identity(worktree: Path, expected_head: str) -> dict[str, Any] | None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    observed_head = head.stdout.strip() if head.returncode == 0 else None
    dirty = porcelain.stdout.splitlines() if porcelain.returncode == 0 else None
    if (
        head.returncode == 0
        and porcelain.returncode == 0
        and observed_head == expected_head
        and dirty == []
    ):
        return None
    return {
        "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
        "expected_git_head": expected_head,
        "observed_git_head": observed_head,
        "head_return_code": int(head.returncode),
        "porcelain_return_code": int(porcelain.returncode),
        "porcelain_lines": dirty,
    }


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
                "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_run_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_stage": "REFIT",
                "run_id": "v4_full",
                "seed": int(job["seed"]),
                "return_code": int(return_code),
                "failure_stage": "FORMAL_REFIT_SCHEDULER",
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
            "run_id": "v4_full",
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
        "manifest_path": schedule["adjudication"]["manifest_path"],
        "failure_path": schedule["adjudication"]["failure_path"],
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_runtime.v1",
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "jobs": states,
                "adjudication": adjudication,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_access_event_count_before_refit": 1,
                "development_test_outcome_reads_during_refit": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")

    def record_failure(job: dict[str, Any], *, reason: str, **details: Any) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "job_key": str(job["job_key"]),
                    "seed": int(job["seed"]),
                    "run_id": "v4_full",
                    "failure_reason": reason,
                    "summary_path": job["summary_path"],
                    "failure_path": job["failure_path"],
                    "log_path": job["log_path"],
                    **details,
                }
            )
        terminal_failure.set()

    def mark_not_run(jobs: list[dict[str, Any]], start: int) -> None:
        for skipped in jobs[start:]:
            state = states[str(skipped["job_key"])]
            if state.get("status") == "PENDING":
                state.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_REFIT_JOB_TECHNICAL_FAILURE",
                    }
                )

    def run_queue(queue: dict[str, Any]) -> None:
        jobs = list(queue["jobs"])
        for index, job in enumerate(jobs):
            key = str(job["job_key"])
            with lock:
                if terminal_failure.is_set():
                    mark_not_run(jobs, index)
                    publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")
                    return
                identity_failure = inspect_worktree_identity(
                    worktree, str(schedule["git_head"])
                )
                if identity_failure is not None:
                    states[key].update(
                        {
                            "status": "TECHNICAL_FAILURE",
                            "return_code": None,
                            "terminal_artifact_kind": exact_terminal(
                                job["summary_path"], job["failure_path"]
                            ),
                            "finished_unix_seconds": time.time(),
                            **identity_failure,
                        }
                    )
                    record_failure(
                        job,
                        reason="SCHEDULE_WORKTREE_IDENTITY_DRIFT",
                        **identity_failure,
                    )
                    mark_not_run(jobs, index + 1)
                    publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")
                    return
                states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")
            return_code = run_logged(
                list(job["command"]), cwd=worktree, log=Path(job["log_path"])
            )
            terminal = close_missing_job(job, return_code=return_code)
            with lock:
                succeeded = terminal == "SUMMARY" and return_code == 0
                states[key].update(
                    {
                        "status": "TERMINAL_COMPLETE" if succeeded else "TECHNICAL_FAILURE",
                        "return_code": int(return_code),
                        "terminal_artifact_kind": terminal,
                        "finished_unix_seconds": time.time(),
                    }
                )
                if not succeeded:
                    record_failure(
                        job,
                        reason=(
                            "NONZERO_RETURN_CODE_WITH_SUMMARY"
                            if terminal == "SUMMARY" and return_code != 0
                            else "NO_UNIQUE_SUCCESS_SUMMARY"
                        ),
                        return_code=int(return_code),
                        terminal_artifact_kind=terminal,
                    )
                publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")

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
                [job for queue in schedule["gpu_queues"] for job in queue["jobs"]],
                0,
            )
            adjudication.update(
                {
                    "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                    "terminal_artifact_kind": None,
                }
            )
            publish("XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE")
        return

    spec = schedule["adjudication"]
    adjudication["status"] = "RUNNING"
    publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")
    return_code = run_logged(
        list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
    )
    manifest = Path(spec["manifest_path"])
    failure = Path(spec["failure_path"])
    if not manifest.exists() and not failure.exists():
        write_atomic(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_adjudication_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "return_code": int(return_code),
                "development_test_outcome_reads_during_refit": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    terminal = exact_terminal(str(manifest), str(failure))
    refit_status = None
    loso_authorized = False
    adjudication_succeeded = terminal == "SUMMARY" and return_code == 0
    if adjudication_succeeded:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        refit_status = payload.get("status")
        loso_authorized = payload.get("loso_authorized") is True
    adjudication.update(
        {
            "status": (
                "TERMINAL_COMPLETE" if adjudication_succeeded else "TECHNICAL_FAILURE"
            ),
            "return_code": int(return_code),
            "terminal_artifact_kind": terminal,
            "refit_status": refit_status,
            "loso_authorized": loso_authorized,
            "finished_unix_seconds": time.time(),
        }
    )
    exact_jobs = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    if not adjudication_succeeded:
        first_terminal_failure.update(
            {
                "job_key": "REFIT_ADJUDICATION",
                "failure_reason": (
                    "NONZERO_RETURN_CODE_WITH_SUMMARY"
                    if terminal == "SUMMARY" and return_code != 0
                    else "NO_UNIQUE_SUCCESS_SUMMARY"
                ),
                "return_code": int(return_code),
                "terminal_artifact_kind": terminal,
                "summary_path": str(manifest),
                "failure_path": str(failure),
                "log_path": spec["log_path"],
            }
        )
        publish("XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE")
        return
    publish(
        "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED"
        if exact_jobs and loso_authorized
        else "XEDITCRITIC_V4_REFIT_TERMINAL_NO_GO"
        if exact_jobs
        else "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
