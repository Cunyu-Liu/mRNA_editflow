#!/usr/bin/env python3
"""Recover SetFlow V4 checkpoint validation without retraining either model."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_v403_setflow_validation_recovery_20260827"
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
BASE_CONFIG = WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
VALIDATION_JOBS = tuple(
    (run_id, checkpoint_pass)
    for run_id in ("v4_full", "v4_single_mode")
    for checkpoint_pass in (4, 6, 8, 10)
)


class XEditSetFlowV403LaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV403LaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments, cwd=WORKTREE, text=True, capture_output=True, check=True
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_new_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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


def validation_assignments(
    physical_gpus: Sequence[int],
) -> dict[int, tuple[tuple[str, int], ...]]:
    gpus = tuple(dict.fromkeys(int(gpu) for gpu in physical_gpus))
    require(bool(gpus), "no physical GPU is configured for SetFlow validation")
    require(
        all(gpu >= 0 for gpu in gpus),
        "SetFlow validation recovery received a negative physical GPU index",
    )
    assignments: dict[int, list[tuple[str, int]]] = {gpu: [] for gpu in gpus}
    for index, job in enumerate(VALIDATION_JOBS):
        assignments[gpus[index % len(gpus)]].append(job)
    return {gpu: tuple(jobs) for gpu, jobs in assignments.items() if jobs}


def run(current_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    require(PYTHON.is_file(), "formal Python is absent")
    require(SCHEDULER.is_file(), "SetFlow validation scheduler is absent")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 recovery worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 recovery worktree is dirty",
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

    base_config = read_json(BASE_CONFIG)
    output_root = Path(base_config["output_root"])
    for run_id in ("v4_full", "v4_single_mode"):
        directory = output_root / run_id
        require(
            (directory / "training_summary.json").is_file()
            and not (directory / "failure.json").exists(),
            f"SetFlow training is not summary-terminal: {run_id}",
        )
        training_config = read_json(directory / "training_config.json")
        training_attempt = read_json(directory / "training_attempt.json")
        require(
            training_config.get("authorized_git_head") == SOURCE_SCREEN_HEAD
            and training_attempt.get("code_commit") == SOURCE_SCREEN_HEAD,
            f"SetFlow training provenance differs from source HEAD: {run_id}",
        )

    original_gate_path = Path(base_config["screen_gate_output_path"])
    original_gate = read_json(original_gate_path)
    require(
        original_gate.get("status") == "XEDITSETFLOW_V4_SCREEN_NO_GO"
        and original_gate.get("reason")
        == "ONE_OR_MORE_FROZEN_TRAINING_OR_CHECKPOINT_VALIDATION_RUNS_FAILED_TECHNICALLY",
        "original SetFlow gate is not the expected technical NO-GO",
    )

    authorization = (
        ROOT
        / "authorizations/xedit_v4"
        / f"screen_{EXPERIMENT_HEAD}_runner_{SOURCE_SCREEN_HEAD}/setflow.json"
    )
    require(authorization.is_file(), "SetFlow screen authorization is absent")
    free_memory = gpu_free_memory_mib()
    configured_gpus = tuple(
        int(gpu) for gpu in base_config["gpu_policy"]["physical_gpu_scope"]
    )
    require(
        all(gpu in free_memory for gpu in configured_gpus),
        "a configured SetFlow physical GPU is absent",
    )
    assignments = validation_assignments(configured_gpus)

    run_name = f"v403_validation_recovery_{current_head}"
    recovery_root = output_root / run_name
    validation_root = recovery_root / "outcome_free_validation_generation"
    gate_path = recovery_root / "screen_gate.json"
    runtime_config_path = (
        ROOT / "runtime_configs/xeditsetflow_v4" / f"{run_name}.json"
    )
    runtime_root = ROOT / "experiments/xeditsetflow_v4" / run_name
    log_root = ROOT / "logs/xeditsetflow_v4" / run_name
    launch_marker = (
        ROOT
        / "authorizations/xeditsetflow_v4"
        / f"v403_validation_recovery_launch_{current_head}.json"
    )
    for path, message in (
        (recovery_root, "SetFlow V4.0.3 recovery output already exists"),
        (runtime_root, "SetFlow V4.0.3 runtime already exists"),
        (runtime_config_path, "SetFlow V4.0.3 runtime config already exists"),
        (launch_marker, "SetFlow V4.0.3 launch already consumed"),
    ):
        require(not path.exists(), message)

    recovery_config = {
        **base_config,
        "status": "VALIDATION_ONLY_RECOVERY_FROM_TERMINAL_V4_CHECKPOINTS",
        "validation_output_root": str(validation_root),
        "screen_gate_output_path": str(gate_path),
        "validation_recovery": {
            "training_git_head": SOURCE_SCREEN_HEAD,
            "validation_git_head": current_head,
            "original_technical_gate": str(original_gate_path),
            "parameter_updates": 0,
            "training_reused": True,
            "scientific_thresholds_changed": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    }
    write_new_atomic(runtime_config_path, recovery_config)
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
                        str(runtime_config_path),
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
            "route_a_v3_route2_xeditsetflow_v403_validation_recovery_schedule.v1"
        ),
        "status": "SETFLOW_VALIDATION_ONLY_RECOVERY_SCHEDULED",
        "git_head": current_head,
        "source_screen_head": SOURCE_SCREEN_HEAD,
        "experiment_head": EXPERIMENT_HEAD,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "runtime_config": str(runtime_config_path),
        "source_screen_runtime": str(screen_root / "runtime.json"),
        "original_technical_gate": str(original_gate_path),
        "gpu_free_memory_mib_before_launch": free_memory,
        "free_memory_gate_applied": False,
        "configured_physical_gpu_scope": list(configured_gpus),
        "validation_queues": queues,
        "runtime_identity": {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v403_validation_recovery_runtime.v1"
            ),
            "running_status": "XEDITSETFLOW_V403_VALIDATION_RECOVERY_RUNNING",
            "terminal_status": (
                "XEDITSETFLOW_V403_VALIDATION_RECOVERY_AND_GATE_TERMINAL"
            ),
            "failure_status": "XEDITSETFLOW_V403_VALIDATION_RECOVERY_FAILED",
        },
        "setflow_adjudication": {
            "gate_path": str(gate_path),
            "log_path": str(log_root / "setflow_adjudication.log"),
            "command": [
                str(PYTHON),
                str(adjudicator),
                "--config",
                str(runtime_config_path),
            ],
        },
        "training_reused": True,
        "parameter_update_count": 0,
        "critic_failure_payload_reads": 0,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(schedule_path, schedule)
    write_new_atomic(
        launch_marker,
        {
            "status": "CONSUMED",
            "git_head": current_head,
            "source_screen_head": SOURCE_SCREEN_HEAD,
            "schedule_path": str(schedule_path),
            "training_reused": True,
            "parameter_update_count": 0,
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
            "route_a_v3_route2_xeditsetflow_v403_validation_recovery_launch.v1"
        ),
        "status": "XEDITSETFLOW_V403_VALIDATION_RECOVERY_LAUNCHED",
        "git_head": current_head,
        "training_git_head": SOURCE_SCREEN_HEAD,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "runtime_config": str(runtime_config_path),
        "scheduler_log": str(coordinator_log),
        "eligible_gpus": sorted(assignments),
        "training_reused": True,
        "parameter_update_count": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
