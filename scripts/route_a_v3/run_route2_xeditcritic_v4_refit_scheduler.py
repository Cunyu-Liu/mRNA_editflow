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


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
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
                "active_performance_output_read": False,
                "development_test_access_event_count_before_refit": 1,
                "development_test_outcome_reads_during_refit": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            key = str(job["job_key"])
            with lock:
                states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("XEDITCRITIC_V4_REFIT_SCHEDULER_RUNNING")
            return_code = run_logged(
                list(job["command"]), cwd=worktree, log=Path(job["log_path"])
            )
            terminal = exact_terminal(job["summary_path"], job["failure_path"])
            if terminal is None:
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
                    terminal = exact_terminal(
                        job["summary_path"], job["failure_path"]
                    )
            with lock:
                states[key].update(
                    {
                        "status": (
                            "TERMINAL_COMPLETE"
                            if terminal is not None
                            else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                        ),
                        "return_code": int(return_code),
                        "terminal_artifact_kind": terminal,
                        "finished_unix_seconds": time.time(),
                    }
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
    if terminal == "SUMMARY":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        refit_status = payload.get("status")
        loso_authorized = payload.get("loso_authorized") is True
    adjudication.update(
        {
            "status": (
                "TERMINAL_COMPLETE"
                if terminal is not None
                else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
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
    publish(
        "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED"
        if exact_jobs and terminal == "SUMMARY" and loso_authorized
        else "XEDITCRITIC_V4_REFIT_TERMINAL_NO_GO"
        if exact_jobs and terminal is not None
        else "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
