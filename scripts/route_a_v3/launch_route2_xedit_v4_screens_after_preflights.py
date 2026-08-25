#!/usr/bin/env python3
"""Authorize and detach the frozen Critic/SetFlow V4 screen package."""

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
C3_REFERENCE = (
    ROOT
    / "experiments/xeditcritic_v3/screen_seed_20260830/"
    "c3_v4_reference_read_once.json"
)
SCREEN_PACKAGE_SCHEDULER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_screen_package_scheduler.py"
)


class XEditV4ScreenLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4ScreenLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
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


def expected_authorization_status(component: str) -> str:
    require(component in {"critic", "setflow"}, "unknown V4 screen component")
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return f"{prefix}_V4_SCREEN_LAUNCH_AUTHORIZED"


def screen_run_ids() -> dict[str, list[str]]:
    return {
        "critic": [
            "c0_v4",
            "v4_full",
            "v4_source_only",
            "v4_edit_metadata_only",
            "v4_no_candidate_sequence",
            "v4_candidate_bundle_permutation",
            "v4_no_cross",
            "v4_no_moe",
        ],
        "setflow": ["v4_full", "v4_single_mode"],
    }


def assign_screen_jobs_to_gpu_queues(
    free_memory_mib: dict[int, int],
    *,
    critic_required_mib: int,
    setflow_required_mib: int,
) -> dict[int, list[tuple[str, str]]]:
    """Assign the frozen screen arms to any sufficient physical GPU 0–5."""

    require(
        set(free_memory_mib).issuperset(range(6)),
        "physical GPU inventory 0–5 is incomplete",
    )
    critic_gpus = [
        gpu for gpu in range(6) if free_memory_mib[gpu] >= critic_required_mib
    ]
    setflow_gpus = [
        gpu for gpu in range(6) if free_memory_mib[gpu] >= setflow_required_mib
    ]
    require(bool(critic_gpus), "no GPU 0–5 has enough measured memory for Critic V4")
    require(bool(setflow_gpus), "no GPU 0–5 has enough measured memory for SetFlow V4")

    queues: dict[int, list[tuple[str, str]]] = {}
    for run_id in (
        "v4_full",
        "v4_source_only",
        "v4_edit_metadata_only",
        "v4_no_candidate_sequence",
        "c0_v4",
        "v4_candidate_bundle_permutation",
        "v4_no_cross",
        "v4_no_moe",
    ):
        gpu = min(
            critic_gpus,
            key=lambda candidate: (
                sum(
                    component == "critic"
                    for component, _ in queues.get(candidate, [])
                ),
                len(queues.get(candidate, [])),
                candidate,
            ),
        )
        queues.setdefault(gpu, []).append(("critic", run_id))

    setflow_only_gpus = [gpu for gpu in setflow_gpus if gpu not in critic_gpus]
    for run_id in ("v4_full", "v4_single_mode"):
        candidates = setflow_only_gpus or setflow_gpus
        gpu = min(
            candidates,
            key=lambda candidate: (len(queues.get(candidate, [])), candidate),
        )
        queues.setdefault(gpu, []).append(("setflow", run_id))
    return {gpu: queues[gpu] for gpu in sorted(queues)}


def validate_screen_authorization(
    path: Path, *, component: str, head: str, experiment_head: str
) -> None:
    require(path.is_file(), f"{component} screen authorization is absent")
    payload = read_json(path)
    require(
        payload.get("status") == expected_authorization_status(component)
        and payload.get("authorized_git_head") == head
        and payload.get("cache_experiment_head") == experiment_head
        and set(payload.get("authorized_run_ids", [])) == set(screen_run_ids()[component]),
        f"{component} screen authorization content is invalid",
    )


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
        SCREEN_PACKAGE_SCHEDULER.is_file(),
        "current-HEAD screen package scheduler is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    require(C3_REFERENCE.is_file(), "C3 read-once reference is absent")

    a100_audit = ROOT / f"audits/a100_current_head_v4/sync_tests_{current_head}.json"
    require(a100_audit.is_file(), "exact current-HEAD A100 test audit is absent")
    critic_config = (
        WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
    )
    setflow_config = (
        WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
    )
    critic_preflight_path = (
        ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_5/preflight.json"
    )
    setflow_preflight_path = (
        ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_5/preflight.json"
    )
    source_audit_path = (
        ROOT
        / "experiments/xeditsetflow_v4/screen_seed_20260911/"
        "preflight_attempt_5/source_level_data_audit.json"
    )
    critic_preflight = read_json(critic_preflight_path)
    setflow_preflight = read_json(setflow_preflight_path)
    require(
        critic_preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and critic_preflight.get("passed") is True
        and str(critic_preflight.get("git_head")) == current_head,
        "Critic V4 preflight did not pass at expected HEAD",
    )
    require(
        setflow_preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and setflow_preflight.get("passed") is True
        and str(setflow_preflight.get("git_head")) == current_head,
        "SetFlow V4 preflight did not pass at expected HEAD",
    )
    source_audit = read_json(source_audit_path)
    for payload, label in (
        (critic_preflight, "critic preflight"),
        (setflow_preflight, "setflow preflight"),
        (source_audit, "setflow source audit"),
    ):
        require(
            int(payload.get("development_test_outcome_reads", -1)) == 0
            and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"{label} reports a protected outcome read",
        )

    free_memory = gpu_free_memory_mib()
    require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
    critic_required = math.ceil(
        (float(critic_preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
    )
    setflow_required = math.ceil(
        (float(setflow_preflight["peak_memory_allocated_gib"]) + 2.0) * 1024
    )
    screen_assignments = assign_screen_jobs_to_gpu_queues(
        free_memory,
        critic_required_mib=critic_required,
        setflow_required_mib=setflow_required,
    )
    for run_id in screen_run_ids()["critic"]:
        output = ROOT / f"experiments/xeditcritic_v4/screen_seed_20260907/{run_id}"
        require(not output.exists(), f"Critic output already exists: {run_id}")
    for run_id in screen_run_ids()["setflow"]:
        output = ROOT / f"experiments/xeditsetflow_v4/screen_seed_20260911/{run_id}"
        require(not output.exists(), f"SetFlow output already exists: {run_id}")

    authorization_root = (
        ROOT
        / "authorizations/xedit_v4"
        / f"screen_{experiment_head}_runner_{current_head}"
    )
    authorization_staging = authorization_root.with_name(
        authorization_root.name + ".partial"
    )
    runtime_root = (
        ROOT
        / "experiments/xedit_v4"
        / f"screen_package_{experiment_head}_runner_{current_head}"
    )
    log_root = (
        ROOT
        / "logs/xedit_v4"
        / f"screen_package_{experiment_head}_runner_{current_head}"
    )
    require(not authorization_root.exists(), "screen authorization package already exists")
    require(
        not authorization_staging.exists(),
        "partial screen authorization package already exists",
    )
    require(not runtime_root.exists(), "screen package runtime already exists")

    authorizer = (
        WORKTREE / "scripts/route_a_v3/authorize_route2_xedit_v4_screen_stages.py"
    )
    require(authorizer.is_file(), "current-HEAD screen authorizer is absent")
    components: dict[str, dict[str, Path | None]] = {
        "critic": {
            "config": critic_config,
            "cache_summary": ROOT
            / "pretrained_features/xeditcritic_v4/"
            "frozen_bottom_six_chunk_cache_v1.summary.json",
            "preflight": critic_preflight_path,
            "source_audit": None,
        },
        "setflow": {
            "config": setflow_config,
            "cache_summary": ROOT
            / "pretrained_features/xeditsetflow_v4/"
            "source_token_cache_v3_adoption_receipt_v1.json",
            "preflight": setflow_preflight_path,
            "source_audit": source_audit_path,
        },
    }
    for component, paths in components.items():
        authorization = authorization_staging / f"{component}.json"
        command_arguments = [
            str(PYTHON),
            str(authorizer),
            "--component",
            component,
            "--stage",
            "screen",
            "--screen-config",
            str(paths["config"]),
            "--c3-reference",
            str(C3_REFERENCE),
            "--a100-audit",
            str(a100_audit),
            "--cache-summary",
            str(paths["cache_summary"]),
            "--cache-experiment-head",
            experiment_head,
            "--preflight",
            str(paths["preflight"]),
            "--output",
            str(authorization),
        ]
        if paths["source_audit"] is not None:
            command_arguments.extend(
                ["--source-data-audit", str(paths["source_audit"])]
            )
        command(command_arguments)
        validate_screen_authorization(
            authorization,
            component=component,
            head=current_head,
            experiment_head=experiment_head,
        )
    os.replace(authorization_staging, authorization_root)

    gpu_queues: list[dict[str, Any]] = []
    critic_trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    setflow_trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py"
    for gpu, assigned_jobs in screen_assignments.items():
        jobs = []
        for component, run_id in assigned_jobs:
            if component == "critic":
                output = ROOT / f"experiments/xeditcritic_v4/screen_seed_20260907/{run_id}"
                trainer = critic_trainer
                config = critic_config
                authorization_flag = "--launch-authorization"
            else:
                output = ROOT / f"experiments/xeditsetflow_v4/screen_seed_20260911/{run_id}"
                trainer = setflow_trainer
                config = setflow_config
                authorization_flag = "--authorization"
            jobs.append(
                {
                    "job_key": f"{component}:{run_id}",
                    "component": component,
                    "run_id": run_id,
                    "output_directory": str(output),
                    "log_path": str(log_root / f"{component}_{run_id}.log"),
                    "command": [
                        str(PYTHON),
                        str(trainer),
                        "--config",
                        str(config),
                        "--run-id",
                        run_id,
                        "--physical-gpu-index",
                        str(gpu),
                        authorization_flag,
                        str(authorization_root / f"{component}.json"),
                    ],
                }
            )
        gpu_queues.append({"physical_gpu_index": gpu, "jobs": jobs})

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    schedule_path = runtime_root / "schedule.json"
    runtime_manifest = runtime_root / "runtime.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xedit_v4_screen_package_schedule.v1",
        "status": "FROZEN_SCREEN_PACKAGE_SCHEDULE",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "gpu_free_memory_mib_before_launch": free_memory,
        "critic_required_free_memory_mib": critic_required,
        "setflow_required_free_memory_mib": setflow_required,
        "gpu_assignment_policy": (
            "ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB"
        ),
        "gpu_queues": gpu_queues,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCREEN_PACKAGE_SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xedit_v4_screen_package_launch.v1",
        "status": "V4_SCREEN_PACKAGE_SCHEDULER_LAUNCHED",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
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
