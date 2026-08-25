#!/usr/bin/env python3
"""Run terminal-only V4 screen adjudication and SetFlow validation."""

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


def gate_terminal(*, gate_present: bool) -> str:
    """An atomically published PASS/NO-GO gate is the unique terminal."""

    return "TERMINAL_COMPLETE" if gate_present else "TECHNICAL_FAILURE"


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
    adjudication: dict[str, Any] = {
        "critic": {"status": "PENDING"},
        "setflow": {"status": "PENDING"},
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_postscreen_runtime.v1",
                "status": status,
                "coordinator_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "experiment_head": schedule["experiment_head"],
                "critic_adjudication": adjudication["critic"],
                "setflow_adjudication": adjudication["setflow"],
                "validation_jobs": states,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("V4_POSTSCREEN_COORDINATOR_RUNNING")
    critic = schedule["critic_adjudication"]
    critic_return = run_logged(
        list(critic["command"]), cwd=worktree, log=Path(critic["log_path"])
    )
    critic_gate_present = Path(critic["gate_path"]).is_file()
    adjudication["critic"] = {
        "status": gate_terminal(gate_present=critic_gate_present),
        "return_code": critic_return,
        "gate_path": critic["gate_path"],
        "gate_present": critic_gate_present,
        "log_path": critic["log_path"],
    }
    publish("V4_POSTSCREEN_COORDINATOR_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            key = str(job["job_key"])
            with lock:
                states[key]["status"] = "RUNNING"
                states[key]["started_unix_seconds"] = time.time()
                publish("V4_POSTSCREEN_COORDINATOR_RUNNING")
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
                publish("V4_POSTSCREEN_COORDINATOR_RUNNING")

    threads = [
        threading.Thread(target=run_queue, args=(queue,))
        for queue in schedule["validation_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    setflow = schedule["setflow_adjudication"]
    setflow_return = run_logged(
        list(setflow["command"]), cwd=worktree, log=Path(setflow["log_path"])
    )
    setflow_gate_present = Path(setflow["gate_path"]).is_file()
    adjudication["setflow"] = {
        "status": gate_terminal(gate_present=setflow_gate_present),
        "return_code": setflow_return,
        "gate_path": setflow["gate_path"],
        "gate_present": setflow_gate_present,
        "log_path": setflow["log_path"],
    }
    exact_validations = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    exact_gates = all(
        row["status"] == "TERMINAL_COMPLETE" and row["gate_present"]
        for row in adjudication.values()
    )
    publish(
        "V4_POSTSCREEN_ALL_TERMINAL"
        if exact_validations and exact_gates
        else "V4_POSTSCREEN_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
