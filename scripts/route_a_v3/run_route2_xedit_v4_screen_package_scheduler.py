#!/usr/bin/env python3
"""Run the frozen V4 screen queues without reading active performance output."""

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


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    lock = threading.Lock()
    states: dict[str, dict[str, Any]] = {}
    for queue in schedule["gpu_queues"]:
        for job in queue["jobs"]:
            states[job["job_key"]] = {
                "component": job["component"],
                "run_id": job["run_id"],
                "physical_gpu_index": int(queue["physical_gpu_index"]),
                "status": "PENDING",
                "output_directory": job["output_directory"],
                "log_path": job["log_path"],
            }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_screen_package_runtime.v1",
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "jobs": states,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("V4_SCREEN_PACKAGE_SCHEDULER_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            key = str(job["job_key"])
            output = Path(job["output_directory"])
            if output.exists():
                with lock:
                    states[key]["status"] = "TECHNICAL_FAILURE_PREEXISTING_OUTPUT"
                    states[key]["finished_unix_seconds"] = time.time()
                    publish("V4_SCREEN_PACKAGE_SCHEDULER_RUNNING")
                continue
            log = Path(job["log_path"])
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    list(job["command"]),
                    cwd=Path(schedule["worktree"]),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with lock:
                    states[key].update(
                        {
                            "status": "RUNNING",
                            "training_pid": process.pid,
                            "started_unix_seconds": started,
                        }
                    )
                    publish("V4_SCREEN_PACKAGE_SCHEDULER_RUNNING")
                return_code = process.wait()
            kind = terminal_kind(job)
            with lock:
                states[key].update(
                    {
                        "status": (
                            "TERMINAL_COMPLETE"
                            if kind is not None
                            else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                        ),
                        "return_code": return_code,
                        "terminal_artifact_kind": kind,
                        "finished_unix_seconds": time.time(),
                    }
                )
                publish("V4_SCREEN_PACKAGE_SCHEDULER_RUNNING")

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

    exact_terminal = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    publish(
        "V4_SCREEN_PACKAGE_ALL_JOBS_TERMINAL"
        if exact_terminal
        else "V4_SCREEN_PACKAGE_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
