#!/usr/bin/env python3
"""Launch terminal-only V4 screen adjudication at one exact A100 HEAD."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
POSTSCREEN_COORDINATOR = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_postscreen_adjudication_scheduler.py"
)


class XEditV4PostscreenLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4PostscreenLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments, cwd=WORKTREE, text=True, capture_output=True, check=True
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def gpu_free_memory_mib() -> dict[int, int]:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free = (part.strip() for part in line.split(",", maxsplit=1))
        values[int(index)] = int(free)
    return values


def expected_screen_jobs() -> set[str]:
    return {
        "critic:c0_v4",
        "critic:v4_full",
        "critic:v4_source_only",
        "critic:v4_edit_metadata_only",
        "critic:v4_no_candidate_sequence",
        "critic:v4_candidate_bundle_permutation",
        "critic:v4_no_cross",
        "critic:v4_no_moe",
        "setflow:v4_full",
        "setflow:v4_single_mode",
    }


def run(current_head: str, experiment_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", experiment_head) is not None,
        "expected cache experiment HEAD is invalid",
    )
    require(PYTHON.is_file(), "formal Python is absent")
    require(
        POSTSCREEN_COORDINATOR.is_file(),
        "current-HEAD post-screen coordinator is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )

    screen_root = (
        ROOT
        / "experiments/xedit_v4"
        / f"screen_package_{experiment_head}_runner_{current_head}"
    )
    screen_runtime = read_json(screen_root / "runtime.json")
    require(
        screen_runtime.get("status") == "V4_SCREEN_PACKAGE_ALL_JOBS_TERMINAL",
        "V4 screen package is not fully terminal",
    )
    require(
        str(screen_runtime.get("git_head")) == current_head
        and str(screen_runtime.get("experiment_head")) == experiment_head,
        "V4 screen runtime HEAD changed",
    )
    require(
        screen_runtime.get("active_performance_output_read") is False
        and int(screen_runtime.get("development_test_outcome_reads", -1)) == 0
        and int(screen_runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 screen runtime reports an active-performance or protected-outcome read",
    )
    require(
        set(screen_runtime.get("jobs", {})) == expected_screen_jobs(),
        "V4 screen runtime job set changed",
    )
    require(
        all(
            row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
            for row in screen_runtime["jobs"].values()
        ),
        "V4 screen package lacks an exact terminal artifact",
    )

    critic_config = (
        WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
    )
    setflow_config = (
        WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
    )
    critic_payload = read_json(critic_config)
    setflow_payload = read_json(setflow_config)
    critic_gate = Path(critic_payload["screen_gate_output"])
    setflow_gate = Path(setflow_payload["screen_gate_output_path"])
    for gate in (critic_gate, setflow_gate):
        require(not gate.exists(), f"screen gate already exists: {gate}")
        require(
            not gate.with_suffix(gate.suffix + ".partial").exists(),
            f"partial screen gate exists: {gate}",
        )

    training_kinds: dict[str, str] = {}
    for run_id in ("v4_full", "v4_single_mode"):
        directory = Path(setflow_payload["output_root"]) / run_id
        summary = directory / "training_summary.json"
        failure = directory / "failure.json"
        require(
            summary.exists() != failure.exists(),
            f"SetFlow training is not exactly terminal: {run_id}",
        )
        training_kinds[run_id] = "SUMMARY" if summary.exists() else "FAILURE"

    validation_root = Path(setflow_payload["validation_output_root"])
    validation_jobs_exist = any(
        (validation_root / run_id / f"pass_{checkpoint_pass}").exists()
        or (validation_root / run_id / f"pass_{checkpoint_pass}.failed.json").exists()
        or (
            validation_root
            / run_id
            / f"pass_{checkpoint_pass}.failed.json.partial"
        ).exists()
        for run_id in ("v4_full", "v4_single_mode")
        for checkpoint_pass in (4, 6, 8, 10)
    )
    require(
        not validation_jobs_exist,
        "SetFlow checkpoint validation output already exists",
    )

    runtime_root = ROOT / f"experiments/xedit_v4/postscreen_{current_head}"
    log_root = ROOT / f"logs/xedit_v4/postscreen_{current_head}"
    require(not runtime_root.exists(), "V4 post-screen runtime already exists")
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    critic_adjudicator = (
        WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_screen.py"
    )
    setflow_adjudicator = (
        WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v4_screen.py"
    )
    validator = (
        WORKTREE / "scripts/route_a_v3/validate_route2_xeditsetflow_v4_checkpoint.py"
    )
    authorization = (
        ROOT
        / "authorizations/xedit_v4"
        / f"screen_{experiment_head}_runner_{current_head}/setflow.json"
    )
    require(authorization.is_file(), "SetFlow screen authorization is absent")

    validation_queues: list[dict[str, Any]] = []
    free_memory: dict[int, int] = {}
    required_free = 0
    if set(training_kinds.values()) == {"SUMMARY"}:
        preflight = read_json(Path(setflow_payload["preflight_output_path"]))
        required_free = math.ceil(
            (float(preflight["peak_memory_allocated_gib"]) + 2.0) * 1024
        )
        free_memory = gpu_free_memory_mib()
        require(
            set(free_memory).issuperset(range(6)),
            "physical GPU inventory 0–5 is incomplete",
        )
        for gpu in range(6):
            require(
                free_memory[gpu] >= required_free,
                f"GPU {gpu} lacks SetFlow validation memory",
            )
        assignments = {
            0: (("v4_full", 4), ("v4_single_mode", 8)),
            1: (("v4_full", 6), ("v4_single_mode", 10)),
            2: (("v4_full", 8),),
            3: (("v4_full", 10),),
            4: (("v4_single_mode", 4),),
            5: (("v4_single_mode", 6),),
        }
        for gpu, rows in assignments.items():
            jobs = []
            for run_id, checkpoint_pass in rows:
                output = validation_root / run_id / f"pass_{checkpoint_pass}"
                failure = output.with_name(output.name + ".failed.json")
                jobs.append(
                    {
                        "job_key": f"{run_id}:pass_{checkpoint_pass}",
                        "run_id": run_id,
                        "checkpoint_pass": checkpoint_pass,
                        "terminal_summary": str(output / "validation_summary.json"),
                        "terminal_failure": str(failure),
                        "log_path": str(
                            log_root
                            / f"validate_{run_id}_pass_{checkpoint_pass}.log"
                        ),
                        "command": [
                            str(PYTHON),
                            str(validator),
                            "--config",
                            str(setflow_config),
                            "--run-id",
                            run_id,
                            "--checkpoint-pass",
                            str(checkpoint_pass),
                            "--authorization",
                            str(authorization),
                            "--physical-gpu-index",
                            str(gpu),
                        ],
                    }
                )
            validation_queues.append({"physical_gpu_index": gpu, "jobs": jobs})
    else:
        require(
            "FAILURE" in training_kinds.values(),
            "unexpected SetFlow training terminal package",
        )

    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xedit_v4_postscreen_schedule.v1",
        "status": "FROZEN_POSTSCREEN_SCHEDULE",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "screen_runtime": str(screen_root / "runtime.json"),
        "setflow_training_terminal_kinds": training_kinds,
        "gpu_free_memory_mib_before_launch": free_memory,
        "setflow_required_free_memory_mib": required_free,
        "critic_adjudication": {
            "gate_path": str(critic_gate),
            "log_path": str(log_root / "critic_adjudication.log"),
            "command": [
                str(PYTHON),
                str(critic_adjudicator),
                "--config",
                str(critic_config),
            ],
        },
        "validation_queues": validation_queues,
        "setflow_adjudication": {
            "gate_path": str(setflow_gate),
            "log_path": str(log_root / "setflow_adjudication.log"),
            "command": [
                str(PYTHON),
                str(setflow_adjudicator),
                "--config",
                str(setflow_config),
            ],
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    coordinator_log = log_root / "coordinator.log"
    stream = coordinator_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(PYTHON),
            str(POSTSCREEN_COORDINATOR),
            "--schedule",
            str(schedule_path),
        ],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xedit_v4_postscreen_launch.v1",
        "status": "V4_POSTSCREEN_COORDINATOR_LAUNCHED",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "coordinator_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "coordinator_log": str(coordinator_log),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--experiment-head", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.expected_head, arguments.experiment_head),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
