#!/usr/bin/env python3
"""Run the single authorized V4.0.2 Critic recovery queue on physical GPU5."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


RUN_IDS = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)


class XEditCriticV402RecoverySchedulerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV402RecoverySchedulerError(message)


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def terminal_kind(job: Mapping[str, Any]) -> str | None:
    output = Path(str(job["output_directory"]))
    summary = output / "run_summary.json"
    failure = output / "failure.json"
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v402_recovery_schedule.v1"
        and schedule.get("status") == "FROZEN_V402_RECOVERY_SCHEDULE",
        "V4.0.2 recovery schedule identity is invalid",
    )
    require(
        int(schedule.get("physical_gpu_index", -1)) == 5,
        "V4.0.2 recovery is not bound to physical GPU5",
    )
    jobs = schedule.get("jobs")
    require(isinstance(jobs, list), "V4.0.2 recovery jobs are absent")
    require(
        [str(job.get("run_id")) for job in jobs] == list(RUN_IDS),
        "V4.0.2 recovery is not the exact ordered eight-arm package",
    )
    require(
        len({str(job.get("output_directory")) for job in jobs}) == len(RUN_IDS),
        "V4.0.2 recovery output directories are not unique",
    )
    require(
        schedule.get("setflow_jobs_stopped_modified_or_restarted") is False
        and schedule.get("active_performance_output_read") is False
        and int(schedule.get("development_test_outcome_reads", -1)) == 0
        and int(schedule.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4.0.2 recovery schedule violates an isolation boundary",
    )


def run(schedule: Mapping[str, Any]) -> None:
    validate_schedule(schedule)
    runtime_path = Path(str(schedule["runtime_manifest"]))
    states: dict[str, dict[str, Any]] = {
        str(job["run_id"]): {
            "run_id": str(job["run_id"]),
            "physical_gpu_index": 5,
            "status": "PENDING",
            "output_directory": str(job["output_directory"]),
            "log_path": str(job["log_path"]),
        }
        for job in schedule["jobs"]
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditcritic_v402_recovery_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "physical_gpu_index": 5,
                "jobs": states,
                "terminal_artifact_payloads_read": 0,
                "active_performance_output_read": False,
                "setflow_jobs_stopped_modified_or_restarted": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V402_RECOVERY_SCHEDULER_RUNNING")
    for job in schedule["jobs"]:
        run_id = str(job["run_id"])
        output = Path(str(job["output_directory"]))
        if output.exists():
            states[run_id].update(
                {
                    "status": "TECHNICAL_FAILURE_PREEXISTING_OUTPUT",
                    "finished_unix_seconds": time.time(),
                }
            )
            publish("XEDITCRITIC_V402_RECOVERY_SCHEDULER_RUNNING")
            continue
        log = Path(str(job["log_path"]))
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                list(job["command"]),
                cwd=Path(str(schedule["worktree"])),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            states[run_id].update(
                {
                    "status": "RUNNING",
                    "training_pid": process.pid,
                    "started_unix_seconds": started,
                }
            )
            publish("XEDITCRITIC_V402_RECOVERY_SCHEDULER_RUNNING")
            return_code = process.wait()
        kind = terminal_kind(job)
        states[run_id].update(
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
        publish("XEDITCRITIC_V402_RECOVERY_SCHEDULER_RUNNING")

    exact_terminal = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    publish(
        "XEDITCRITIC_V402_RECOVERY_ALL_EIGHT_TERMINAL"
        if exact_terminal
        else "XEDITCRITIC_V402_RECOVERY_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
