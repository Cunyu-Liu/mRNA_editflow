#!/usr/bin/env python3
"""Run the six repaired Critic V4 controls concurrently on physical GPUs 0-5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


TRAINING_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
TRAINING_WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_v403_critic_rng_replay_20260827"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
TRAINER = (
    TRAINING_WORKTREE
    / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
)
CONTROL_RUN_IDS = (
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
PHYSICAL_GPU_INDICES = (0, 1, 2, 3, 4, 5)


class XEditCriticV403ControlRecoverySchedulerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403ControlRecoverySchedulerError(message)


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def terminal_kind(output_directory: Path) -> str | None:
    summary = output_directory / "run_summary.json"
    failure = output_directory / "failure.json"
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_control_recovery_schedule.v1"
        and schedule.get("status")
        == "XEDITCRITIC_V403_CONTROL_RECOVERY_SCHEDULED",
        "V4.0.3 control recovery schedule identity is invalid",
    )
    require(
        schedule.get("training_code_git_head") == TRAINING_GIT_HEAD
        and Path(str(schedule.get("training_worktree"))) == TRAINING_WORKTREE,
        "V4.0.3 controls are not bound to the repaired f34 training source",
    )
    jobs = schedule.get("jobs")
    require(isinstance(jobs, list), "V4.0.3 control jobs are absent")
    require(
        tuple(str(job.get("run_id")) for job in jobs) == CONTROL_RUN_IDS,
        "V4.0.3 recovery is not the exact six-control package",
    )
    require(
        tuple(int(job.get("physical_gpu_index", -1)) for job in jobs)
        == PHYSICAL_GPU_INDICES,
        "V4.0.3 control GPU inventory or assignment changed",
    )
    require(
        len({str(job.get("output_directory")) for job in jobs})
        == len(CONTROL_RUN_IDS),
        "V4.0.3 control output directories are not unique",
    )
    require(
        len({str(job.get("training_attempt_id")) for job in jobs})
        == len(CONTROL_RUN_IDS),
        "V4.0.3 control training attempt ids are not unique",
    )
    require(
        all(
            list(job.get("command", []))[:2] == [str(PYTHON), str(TRAINER)]
            and "--run-id" in job.get("command", [])
            and str(job["run_id"]) in job.get("command", [])
            for job in jobs
        ),
        "V4.0.3 control command is not the repaired f34 trainer",
    )
    require(
        not ({"c0_v4", "v4_full"} & {str(job["run_id"]) for job in jobs}),
        "V4.0.3 control schedule would retrain C0 or full",
    )
    inventory = schedule.get("cuda_bf16_inventory")
    require(
        isinstance(inventory, list)
        and [int(row.get("physical_gpu_index", -1)) for row in inventory]
        == list(PHYSICAL_GPU_INDICES)
        and all(row.get("bf16_supported") is True for row in inventory)
        and all(row.get("bf16_tensor_probe") is True for row in inventory),
        "V4.0.3 CUDA/BF16 inventory is not exact",
    )
    require(
        schedule.get("full_retrained") is False
        and schedule.get("c0_retrained") is False
        and schedule.get("old_v402_stopped_process_resumed") is False
        and schedule.get("free_memory_gate_applied") is False
        and int(schedule.get("terminal_artifact_payloads_read_by_scheduler", -1))
        == 0
        and int(schedule.get("development_test_outcome_reads", -1)) == 0
        and int(schedule.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4.0.3 control schedule violates a recovery isolation boundary",
    )


def run(schedule: Mapping[str, Any]) -> None:
    validate_schedule(schedule)
    require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    require(
        all(not Path(str(job["output_directory"])).exists() for job in schedule["jobs"]),
        "one or more V4.0.3 control output directories already exist",
    )
    runtime_path = Path(str(schedule["runtime_manifest"]))
    states: dict[str, dict[str, Any]] = {
        str(job["run_id"]): {
            "run_id": str(job["run_id"]),
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "status": "PENDING",
            "output_directory": str(job["output_directory"]),
            "log_path": str(job["log_path"]),
            "training_attempt_id": str(job["training_attempt_id"]),
        }
        for job in schedule["jobs"]
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditcritic_v403_"
                    "control_recovery_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "orchestration_git_head": schedule["orchestration_git_head"],
                "training_code_git_head": TRAINING_GIT_HEAD,
                "training_worktree": str(TRAINING_WORKTREE),
                "ordered_control_run_ids": list(CONTROL_RUN_IDS),
                "jobs": states,
                "full_retrained": False,
                "c0_retrained": False,
                "old_v402_stopped_process_resumed": False,
                "free_memory_gate_applied": False,
                "terminal_artifact_payloads_read_by_scheduler": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V403_CONTROL_RECOVERY_STARTING")
    processes: dict[str, subprocess.Popen[str]] = {}
    spawn_failed = False
    for job in schedule["jobs"]:
        run_id = str(job["run_id"])
        if spawn_failed:
            states[run_id]["status"] = "NOT_STARTED_AFTER_SPAWN_FAILURE"
            continue
        log = Path(str(job["log_path"]))
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                list(job["command"]),
                cwd=TRAINING_WORKTREE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception as exc:
            spawn_failed = True
            states[run_id].update(
                {
                    "status": "TECHNICAL_FAILURE_TO_SPAWN",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finished_unix_seconds": time.time(),
                }
            )
        else:
            processes[run_id] = process
            states[run_id].update(
                {
                    "status": "RUNNING",
                    "training_pid": process.pid,
                    "started_unix_seconds": time.time(),
                }
            )
        finally:
            stream.close()
    publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")

    for job in schedule["jobs"]:
        run_id = str(job["run_id"])
        process = processes.get(run_id)
        if process is None:
            continue
        return_code = process.wait()
        kind = terminal_kind(Path(str(job["output_directory"])))
        exact_summary = kind == "SUMMARY" and return_code == 0
        states[run_id].update(
            {
                "status": (
                    "TERMINAL_SUMMARY"
                    if exact_summary
                    else (
                        "TERMINAL_FAILURE"
                        if kind == "FAILURE"
                        else "TECHNICAL_FAILURE_NO_EXACT_TERMINAL_ARTIFACT"
                    )
                ),
                "return_code": return_code,
                "terminal_artifact_kind": kind,
                "finished_unix_seconds": time.time(),
            }
        )
        publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")

    exact_summaries = all(
        states[run_id].get("status") == "TERMINAL_SUMMARY"
        and states[run_id].get("terminal_artifact_kind") == "SUMMARY"
        and int(states[run_id].get("return_code", -1)) == 0
        for run_id in CONTROL_RUN_IDS
    )
    publish(
        "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
        if exact_summaries
        else "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
