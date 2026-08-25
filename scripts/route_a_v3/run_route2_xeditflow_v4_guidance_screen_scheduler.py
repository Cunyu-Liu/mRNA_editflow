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
from typing import Any


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


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

    def execute(job: dict[str, Any]) -> bool:
        key = str(job["job_key"])
        with lock:
            states[key].update(
                {"status": "RUNNING", "started_unix_seconds": time.time()}
            )
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
        return_code = run_logged(
            list(job["command"]), cwd=worktree, log=Path(job["log_path"])
        )
        terminal = close_missing(job, return_code)
        with lock:
            states[key].update(
                {
                    "status": (
                        "TERMINAL_COMPLETE"
                        if terminal == "SUCCESS"
                        else "TERMINAL_FAILURE"
                    ),
                    "return_code": int(return_code),
                    "terminal_artifact_kind": terminal,
                    "finished_unix_seconds": time.time(),
                }
            )
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
        return terminal == "SUCCESS"

    def stop_pending(status: str) -> None:
        for row in states.values():
            if row["status"] == "PENDING":
                row["status"] = status

    publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
    for job in schedule["serial_value_prerequisites"]:
        if not execute(job):
            stop_pending("NOT_RUN_PREREQUISITE_FAILURE")
            publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
            return

    value_results: dict[str, bool] = {}

    def run_value_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            value_results[job["job_key"]] = execute(job)

    value_threads = [
        threading.Thread(target=run_value_queue, args=(queue,))
        for queue in schedule["value_training_queues"]
    ]
    for thread in value_threads:
        thread.start()
    for thread in value_threads:
        thread.join()
    if not value_results or not all(value_results.values()):
        stop_pending("NOT_RUN_VALUE_TRAINING_FAILURE")
        publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
        return

    chain_results: dict[str, bool] = {}

    def run_guidance_queue(queue: dict[str, Any]) -> None:
        for chain in queue["chains"]:
            complete = True
            for job in chain["jobs"]:
                if complete:
                    complete = execute(job)
                else:
                    with lock:
                        states[job["job_key"]]["status"] = (
                            "NOT_RUN_CHAIN_PREDECESSOR_FAILURE"
                        )
                        publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
            chain_results[chain["combination_id"]] = complete

    guidance_threads = [
        threading.Thread(target=run_guidance_queue, args=(queue,))
        for queue in schedule["guidance_queues"]
    ]
    for thread in guidance_threads:
        thread.start()
    for thread in guidance_threads:
        thread.join()
    if len(chain_results) != 18 or not all(chain_results.values()):
        publish("XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE")
        return

    spec = schedule["adjudication"]
    adjudication.update(
        {"status": "RUNNING", "started_unix_seconds": time.time()}
    )
    publish("XEDITFLOW_V4_GUIDANCE_SCREEN_RUNNING")
    return_code = run_logged(
        list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
    )
    terminal = close_missing(spec, return_code)
    adjudication.update(
        {
            "status": (
                "TERMINAL_COMPLETE"
                if terminal == "SUCCESS"
                else "TERMINAL_FAILURE"
            ),
            "return_code": int(return_code),
            "terminal_artifact_kind": terminal,
            "finished_unix_seconds": time.time(),
        }
    )
    publish(
        "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        if terminal == "SUCCESS"
        else "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
