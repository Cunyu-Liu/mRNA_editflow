#!/usr/bin/env python3
"""Launch SetFlow-only V4.0.2 validation after both training arms terminate."""

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
SOURCE_SCREEN_HEAD = "edad89392077a0cf56e84dfcf94335606dd2b05a"
EXPERIMENT_HEAD = "a7ef72fac23cd5b25dcc6c8d560236b97fa8b09d"
SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/"
    "run_route2_xeditsetflow_v402_terminal_validation_scheduler.py"
)
CONFIG = WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"


class XEditSetFlowV402LaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV402LaunchError(message)


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


def validation_assignments() -> dict[int, tuple[tuple[str, int], ...]]:
    return {
        0: (("v4_full", 4), ("v4_single_mode", 10)),
        1: (("v4_full", 6),),
        2: (("v4_full", 8), ("v4_single_mode", 4)),
        3: (("v4_full", 10), ("v4_single_mode", 6)),
        4: (("v4_single_mode", 8),),
    }


def run(current_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    require(PYTHON.is_file(), "formal Python is absent")
    require(SCHEDULER.is_file(), "SetFlow V4.0.2 scheduler is absent")
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
        / f"screen_package_{EXPERIMENT_HEAD}_runner_{SOURCE_SCREEN_HEAD}"
    )
    screen_runtime = read_json(screen_root / "runtime.json")
    require(
        screen_runtime.get("status") == "V4_SCREEN_PACKAGE_ALL_JOBS_TERMINAL",
        "source V4 screen package is not fully terminal",
    )
    require(
        screen_runtime.get("active_performance_output_read") is False
        and int(screen_runtime.get("development_test_outcome_reads", -1)) == 0
        and int(screen_runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "source V4 screen runtime reports a prohibited read",
    )

    config = read_json(CONFIG)
    output_root = Path(config["output_root"])
    for run_id in ("v4_full", "v4_single_mode"):
        directory = output_root / run_id
        summary = directory / "training_summary.json"
        failure = directory / "failure.json"
        require(
            summary.is_file() and not failure.exists(),
            f"SetFlow training is not summary-terminal: {run_id}",
        )

    gate_path = Path(config["screen_gate_output_path"])
    require(not gate_path.exists(), "SetFlow V4 screen gate already exists")
    require(
        not gate_path.with_suffix(gate_path.suffix + ".partial").exists(),
        "partial SetFlow V4 screen gate already exists",
    )
    validation_root = Path(config["validation_output_root"])
    require(
        not validation_root.exists(),
        "SetFlow checkpoint validation output already exists",
    )

    authorization = (
        ROOT
        / "authorizations/xedit_v4"
        / f"screen_{EXPERIMENT_HEAD}_runner_{SOURCE_SCREEN_HEAD}/setflow.json"
    )
    require(authorization.is_file(), "SetFlow screen authorization is absent")
    preflight = read_json(Path(config["preflight_output_path"]))
    required_free = math.ceil(
        (float(preflight["peak_memory_allocated_gib"]) + 2.0) * 1024
    )
    free_memory = gpu_free_memory_mib()
    assignments = validation_assignments()
    for gpu in assignments:
        require(
            free_memory.get(gpu, -1) >= required_free,
            f"GPU {gpu} lacks SetFlow validation memory",
        )

    runtime_root = (
        ROOT
        / "experiments/xeditsetflow_v4"
        / f"v402_terminal_validation_runner_{current_head}"
    )
    log_root = (
        ROOT
        / "logs/xeditsetflow_v4"
        / f"v402_terminal_validation_runner_{current_head}"
    )
    launch_marker = (
        ROOT
        / "authorizations/xeditsetflow_v4"
        / "v402_terminal_validation_launch_consumed.json"
    )
    require(not runtime_root.exists(), "SetFlow V4.0.2 runtime already exists")
    require(not launch_marker.exists(), "SetFlow V4.0.2 launch already consumed")
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)

    validator = (
        WORKTREE
        / "scripts/route_a_v3/validate_route2_xeditsetflow_v4_checkpoint.py"
    )
    adjudicator = (
        WORKTREE
        / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v4_screen.py"
    )
    queues: list[dict[str, Any]] = []
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
                        log_root / f"validate_{run_id}_pass_{checkpoint_pass}.log"
                    ),
                    "command": [
                        str(PYTHON),
                        str(validator),
                        "--config",
                        str(CONFIG),
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
        queues.append({"physical_gpu_index": gpu, "jobs": jobs})

    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v402_terminal_validation_schedule.v1"
        ),
        "status": "FROZEN_SETFLOW_ONLY_TERMINAL_VALIDATION_SCHEDULE",
        "git_head": current_head,
        "source_screen_head": SOURCE_SCREEN_HEAD,
        "experiment_head": EXPERIMENT_HEAD,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "source_screen_runtime": str(screen_root / "runtime.json"),
        "gpu_free_memory_mib_before_launch": free_memory,
        "required_free_memory_mib": required_free,
        "validation_queues": queues,
        "setflow_adjudication": {
            "gate_path": str(gate_path),
            "log_path": str(log_root / "setflow_adjudication.log"),
            "command": [str(PYTHON), str(adjudicator), "--config", str(CONFIG)],
        },
        "critic_failure_payload_reads": 0,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    write_atomic(
        launch_marker,
        {
            "status": "CONSUMED",
            "git_head": current_head,
            "source_screen_head": SOURCE_SCREEN_HEAD,
            "schedule_path": str(schedule_path),
        },
    )
    coordinator_log = log_root / "scheduler.log"
    stream = coordinator_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v402_terminal_validation_launch.v1"
        ),
        "status": "XEDITSETFLOW_V402_TERMINAL_VALIDATION_LAUNCHED",
        "git_head": current_head,
        "source_screen_head": SOURCE_SCREEN_HEAD,
        "experiment_head": EXPERIMENT_HEAD,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(coordinator_log),
        "critic_failure_payload_reads": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
