#!/usr/bin/env python3
"""Validate terminal SetFlow checkpoints and adjudicate V4 confirmations once."""

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


def publish_missing_validation_failure(
    job: dict[str, Any], *, return_code: int
) -> None:
    summary = Path(job["terminal_summary"])
    failure = Path(job["terminal_failure"])
    if summary.exists() or failure.exists():
        return
    write_atomic(
        failure,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_checkpoint_validation_failure.v1"
            ),
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "stage": "CHECKPOINT_VALIDATION_WRAPPER",
            "run_id": "v4_full",
            "run_stage": "CONFIRMATION",
            "seed": int(job["training_seed"]),
            "checkpoint_pass": int(job["checkpoint_pass"]),
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "return_code": int(return_code),
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    validation_states = {
        job["job_key"]: {
            "training_seed": int(job["training_seed"]),
            "checkpoint_pass": int(job["checkpoint_pass"]),
            "physical_gpu_index": int(queue["physical_gpu_index"]),
            "status": "PENDING",
            "terminal_summary": job["terminal_summary"],
            "terminal_failure": job["terminal_failure"],
            "log_path": job["log_path"],
        }
        for queue in schedule["validation_queues"]
        for job in queue["jobs"]
    }
    adjudication_states = {
        component: {
            "status": "PENDING",
            "gate_path": spec["gate_path"],
            "failure_path": spec["failure_path"],
            "log_path": spec["log_path"],
        }
        for component, spec in schedule["adjudications"].items()
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xedit_v4_confirmation_posttraining_runtime.v1"
                ),
                "status": status,
                "coordinator_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "eligible_components": schedule["eligible_components"],
                "validation_jobs": validation_states,
                "adjudications": adjudication_states,
                "active_performance_output_read": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        for job in queue["jobs"]:
            key = str(job["job_key"])
            with lock:
                validation_states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
            return_code = run_logged(
                list(job["command"]), cwd=worktree, log=Path(job["log_path"])
            )
            terminal = exact_terminal(
                job["terminal_summary"], job["terminal_failure"]
            )
            if terminal is None:
                publish_missing_validation_failure(job, return_code=return_code)
                terminal = exact_terminal(
                    job["terminal_summary"], job["terminal_failure"]
                )
            with lock:
                validation_states[key].update(
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
                publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")

    threads = [
        threading.Thread(
            target=run_queue,
            args=(queue,),
            name=f"gpu-{queue['physical_gpu_index']}",
        )
        for queue in schedule["validation_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for component in schedule["eligible_components"]:
        spec = schedule["adjudications"][component]
        adjudication_states[component]["status"] = "RUNNING"
        adjudication_states[component]["started_unix_seconds"] = time.time()
        publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")
        return_code = run_logged(
            list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
        )
        gate = Path(spec["gate_path"])
        failure = Path(spec["failure_path"])
        if not gate.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xedit_v4_confirmation_adjudication_failure.v1"
                    ),
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "component": component,
                    "return_code": return_code,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        terminal = exact_terminal(str(gate), str(failure))
        adjudication_states[component].update(
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
        publish("V4_CONFIRMATION_POSTTRAINING_RUNNING")

    exact_validations = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in validation_states.values()
    )
    exact_adjudications = bool(adjudication_states) and all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in adjudication_states.values()
    )
    publish(
        "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
        if exact_validations and exact_adjudications
        else "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
