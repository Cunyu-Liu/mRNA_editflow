#!/usr/bin/env python3
"""Run the frozen V4 three-seed matched-compute comparison package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _all_jobs(schedule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jobs: list[Mapping[str, Any]] = []
    for queue in schedule["prerequisite_queues"]:
        jobs.extend(queue["jobs"])
    for chain in schedule["seed_chains"]:
        jobs.extend(chain["jobs"])
    jobs.extend(schedule["finalization_jobs"])
    return jobs


def run(schedule: Mapping[str, Any]) -> None:
    runtime_path = Path(str(schedule["runtime_manifest"]))
    worktree = Path(str(schedule["worktree"]))
    lock = threading.Lock()
    states = {
        str(job["job_key"]): {
            "status": "PENDING",
            "physical_gpu_indices": list(job.get("physical_gpu_indices", ())),
            "success_path": str(job["success_path"]),
            "failure_path": str(job["failure_path"]),
            "log_path": str(job["log_path"]),
        }
        for job in _all_jobs(schedule)
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v4_final_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "experiment_head": schedule["experiment_head"],
                "guidance_runner_head": schedule["guidance_runner_head"],
                "jobs": states,
                "active_performance_output_read": False,
                "development_test_reopened": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def close_missing(job: Mapping[str, Any], return_code: int) -> str:
        success = Path(str(job["success_path"]))
        failure = Path(str(job["failure_path"]))
        if success.exists() and not failure.exists():
            return "SUCCESS"
        if failure.exists() and not success.exists():
            return "FAILURE"
        if not success.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
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

    def execute(job: Mapping[str, Any]) -> bool:
        key = str(job["job_key"])
        with lock:
            states[key].update(
                {"status": "RUNNING", "started_unix_seconds": time.time()}
            )
            publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
        return_code = run_logged(
            list(job["command"]), cwd=worktree, log=Path(str(job["log_path"]))
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
            publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
        return terminal == "SUCCESS"

    def stop_pending(status: str) -> None:
        for row in states.values():
            if row["status"] == "PENDING":
                row["status"] = status

    def run_queue(queue: Mapping[str, Any], results: dict[str, bool]) -> None:
        complete = True
        for job in queue["jobs"]:
            if complete:
                complete = execute(job)
            else:
                with lock:
                    states[str(job["job_key"])]["status"] = (
                        "NOT_RUN_QUEUE_PREDECESSOR_FAILURE"
                    )
                    publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
        results[str(queue["queue_key"])] = complete

    publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
    prerequisite_results: dict[str, bool] = {}
    threads = [
        threading.Thread(target=run_queue, args=(queue, prerequisite_results))
        for queue in schedule["prerequisite_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not prerequisite_results or not all(prerequisite_results.values()):
        stop_pending("NOT_RUN_PREREQUISITE_FAILURE")
        publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
        return

    seed_results: dict[str, bool] = {}
    threads = [
        threading.Thread(target=run_queue, args=(chain, seed_results))
        for chain in schedule["seed_chains"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if set(seed_results) != {"seed_20260912", "seed_20260913", "seed_20260914"} or not all(
        seed_results.values()
    ):
        stop_pending("NOT_RUN_SEED_CHAIN_FAILURE")
        publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
        return

    for job in schedule["finalization_jobs"]:
        if not execute(job):
            stop_pending("NOT_RUN_FINALIZATION_FAILURE")
            publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
            return
    publish("XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
