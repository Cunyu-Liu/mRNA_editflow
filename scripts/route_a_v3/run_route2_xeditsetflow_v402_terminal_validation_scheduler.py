#!/usr/bin/env python3
"""Run the V4.0.2 SetFlow-only terminal validation and screen gate."""

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


def exact_terminal(job: dict[str, Any]) -> str | None:
    summary = Path(job["terminal_summary"])
    failure = Path(job["terminal_failure"])
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
    states = {
        job["job_key"]: {
            "run_id": job["run_id"],
            "checkpoint_pass": job["checkpoint_pass"],
            "physical_gpu_index": queue["physical_gpu_index"],
            "status": "PENDING",
            "terminal_summary": job["terminal_summary"],
            "terminal_failure": job["terminal_failure"],
            "log_path": job["log_path"],
        }
        for queue in schedule["validation_queues"]
        for job in queue["jobs"]
    }
    lock = threading.Lock()
    adjudication: dict[str, Any] = {"status": "PENDING"}

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v402_terminal_validation_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "source_screen_head": schedule["source_screen_head"],
                "experiment_head": schedule["experiment_head"],
                "setflow_adjudication": adjudication,
                "validation_jobs": states,
                "critic_failure_payload_reads": 0,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITSETFLOW_V402_TERMINAL_VALIDATION_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            key = str(job["job_key"])
            with lock:
                states[key]["status"] = "RUNNING"
                states[key]["started_unix_seconds"] = time.time()
                publish("XEDITSETFLOW_V402_TERMINAL_VALIDATION_RUNNING")
            return_code = run_logged(
                list(job["command"]), cwd=worktree, log=Path(job["log_path"])
            )
            terminal = exact_terminal(job)
            with lock:
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
                publish("XEDITSETFLOW_V402_TERMINAL_VALIDATION_RUNNING")

    threads = [
        threading.Thread(target=run_queue, args=(queue,))
        for queue in schedule["validation_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    gate = schedule["setflow_adjudication"]
    gate_return = run_logged(
        list(gate["command"]), cwd=worktree, log=Path(gate["log_path"])
    )
    gate_present = Path(gate["gate_path"]).is_file()
    adjudication.update(
        {
            "status": "TERMINAL_COMPLETE" if gate_present else "TECHNICAL_FAILURE",
            "return_code": gate_return,
            "gate_path": gate["gate_path"],
            "gate_present": gate_present,
            "log_path": gate["log_path"],
        }
    )
    exact_validations = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    publish(
        "XEDITSETFLOW_V402_VALIDATION_AND_GATE_TERMINAL"
        if exact_validations and gate_present
        else "XEDITSETFLOW_V402_VALIDATION_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
