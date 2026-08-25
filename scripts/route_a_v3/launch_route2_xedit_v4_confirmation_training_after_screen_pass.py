#!/usr/bin/env python3
"""Prepare, authorize, and launch only screen-passing V4 confirmations."""

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
CONFIRMATION_SCHEDULER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
)
CRITIC_SEEDS = (20260908, 20260909, 20260910)
SETFLOW_SEEDS = (20260912, 20260913, 20260914)


class XEditV4ConfirmationLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4ConfirmationLaunchError(message)


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


def gate_passed(component: str, gate: dict[str, Any]) -> bool:
    require(component in {"critic", "setflow"}, "unknown V4 confirmation component")
    expected = (
        "XEDITCRITIC_V4_SCREEN_PASS"
        if component == "critic"
        else "XEDITSETFLOW_V4_SCREEN_PASS"
    )
    passed = gate.get("status") == expected
    require(
        bool(gate.get("confirmation_authorized")) is passed,
        f"{component} screen gate confirmation authorization is inconsistent",
    )
    require(
        int(gate.get("development_test_outcome_reads", -1)) == 0
        and int(gate.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{component} screen gate reports a protected outcome read",
    )
    return passed


def validate_config_manifest(component: str, payload: dict[str, Any]) -> None:
    if component == "critic":
        require(
            payload.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_confirmation_config_manifest.v1"
            and payload.get("status")
            == "THREE_MATCHED_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
            and payload.get("required_seeds") == list(CRITIC_SEEDS)
            and payload.get("required_run_ids") == ["v4_full", "c0_v4"]
            and len(payload.get("config_paths", [])) == 3,
            "Critic V4 confirmation config manifest is invalid",
        )
    else:
        require(
            payload.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_confirmation_config_manifest.v1"
            and payload.get("status")
            == "THREE_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED"
            and payload.get("required_seeds") == list(SETFLOW_SEEDS)
            and payload.get("selected_model") == "v4_full"
            and len(payload.get("config_paths", [])) == 3,
            "SetFlow V4 confirmation config manifest is invalid",
        )
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{component} confirmation config manifest reports a protected read",
    )


def validate_authorization(
    component: str, payload: dict[str, Any], *, head: str
) -> None:
    if component == "critic":
        valid = (
            payload.get("status")
            == "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
            and payload.get("authorized_seeds") == list(CRITIC_SEEDS)
            and payload.get("authorized_run_ids") == ["v4_full", "c0_v4"]
        )
    else:
        valid = (
            payload.get("status")
            == "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
            and payload.get("authorized_seeds") == list(SETFLOW_SEEDS)
            and payload.get("authorized_run_id") == "v4_full"
        )
    require(
        valid
        and payload.get("authorized_git_head") == head
        and payload.get("additional_seed_authorized") is False
        and payload.get("development_test_authorized") is False
        and int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{component} confirmation authorization is invalid",
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
        CONFIRMATION_SCHEDULER.is_file(),
        "current-HEAD confirmation scheduler is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    postscreen_runtime_path = (
        ROOT / f"experiments/xedit_v4/postscreen_{current_head}/runtime.json"
    )
    postscreen = read_json(postscreen_runtime_path)
    require(
        postscreen.get("status") == "V4_POSTSCREEN_ALL_TERMINAL"
        and postscreen.get("git_head") == current_head
        and postscreen.get("experiment_head") == experiment_head
        and postscreen.get("active_performance_output_read") is False
        and int(postscreen.get("development_test_outcome_reads", -1)) == 0
        and int(postscreen.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 post-screen package is not an isolated exact-HEAD terminal",
    )

    critic_config = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
    setflow_config = WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
    critic_protocol = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_confirmation_protocol_v1.json"
    setflow_protocol = WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
    critic_gate = ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/screen_gate.json"
    setflow_gate = ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/screen_gate.json"
    gates = {"critic": read_json(critic_gate), "setflow": read_json(setflow_gate)}
    eligible = [
        component
        for component in ("critic", "setflow")
        if gate_passed(component, gates[component])
    ]
    require(bool(eligible), "neither V4 screen authorizes confirmation")

    component_data: dict[str, dict[str, Any]] = {
        "critic": {
            "base_config": critic_config,
            "protocol": critic_protocol,
            "screen_gate": critic_gate,
            "screen_authorization": ROOT
            / "authorizations/xedit_v4"
            / f"screen_{experiment_head}_runner_{current_head}/critic.json",
            "preflight": ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_5/preflight.json",
            "prepare": WORKTREE / "scripts/route_a_v3/prepare_route2_xeditcritic_v4_confirmation_configs.py",
            "authorize": WORKTREE / "scripts/route_a_v3/authorize_route2_xeditcritic_v4_confirmation.py",
            "config_root": ROOT / "runtime_configs/xeditcritic_v4/confirmation_v1",
        },
        "setflow": {
            "base_config": setflow_config,
            "protocol": setflow_protocol,
            "screen_gate": setflow_gate,
            "screen_authorization": ROOT
            / "authorizations/xedit_v4"
            / f"screen_{experiment_head}_runner_{current_head}/setflow.json",
            "preflight": ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_5/preflight.json",
            "source_data_audit": ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_5/source_level_data_audit.json",
            "prepare": WORKTREE / "scripts/route_a_v3/prepare_route2_xeditsetflow_v4_confirmation_configs.py",
            "authorize": WORKTREE / "scripts/route_a_v3/authorize_route2_xeditsetflow_v4_confirmation.py",
            "config_root": ROOT / "runtime_configs/xeditsetflow_v4/confirmation_v1",
        },
    }
    authorization_root = ROOT / f"authorizations/xedit_v4/confirmation_{current_head}"
    authorization_staging = authorization_root.with_name(
        authorization_root.name + ".partial"
    )
    runtime_root = ROOT / f"experiments/xedit_v4/confirmation_training_{current_head}"
    log_root = ROOT / f"logs/xedit_v4/confirmation_training_{current_head}"
    require(not authorization_root.exists(), "confirmation authorization package exists")
    require(
        not authorization_staging.exists(),
        "partial confirmation authorization package exists",
    )
    require(not runtime_root.exists(), "confirmation training runtime exists")

    manifests: dict[str, dict[str, Any]] = {}
    for component in eligible:
        data = component_data[component]
        for key in ("base_config", "protocol", "screen_gate", "prepare", "authorize"):
            require(Path(data[key]).is_file(), f"{component} confirmation {key} is absent")
        command(
            [
                str(PYTHON),
                str(data["prepare"]),
                "--base-config",
                str(data["base_config"]),
                "--protocol",
                str(data["protocol"]),
                "--screen-gate",
                str(data["screen_gate"]),
            ]
        )
        manifest = read_json(Path(data["config_root"]) / "manifest.json")
        validate_config_manifest(component, manifest)
        manifests[component] = manifest

        authorization = authorization_staging / f"{component}.json"
        arguments = [
            str(PYTHON),
            str(data["authorize"]),
            "--screen-config",
            str(data["base_config"]),
            "--screen-authorization",
            str(data["screen_authorization"]),
            "--preflight",
            str(data["preflight"]),
        ]
        if component == "setflow":
            arguments.extend(
                ["--source-data-audit", str(data["source_data_audit"])]
            )
        arguments.extend(
            [
                "--screen-gate",
                str(data["screen_gate"]),
                "--output",
                str(authorization),
            ]
        )
        command(arguments)
        validate_authorization(
            component, read_json(authorization), head=current_head
        )
    os.replace(authorization_staging, authorization_root)

    free_memory = gpu_free_memory_mib()
    require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in range(6)}
    required_by_gpu = {gpu: 0 for gpu in range(6)}
    critic_trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    setflow_trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py"
    if "critic" in eligible:
        preflight = read_json(Path(component_data["critic"]["preflight"]))
        required = math.ceil(
            (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
        )
        configs = {
            int(read_json(Path(path))["training_seed"]): Path(path)
            for path in manifests["critic"]["config_paths"]
        }
        assignments = [
            (0, CRITIC_SEEDS[0], "v4_full"),
            (1, CRITIC_SEEDS[0], "c0_v4"),
            (2, CRITIC_SEEDS[1], "v4_full"),
            (3, CRITIC_SEEDS[1], "c0_v4"),
            (4, CRITIC_SEEDS[2], "v4_full"),
            (5, CRITIC_SEEDS[2], "c0_v4"),
        ]
        for gpu, seed, run_id in assignments:
            config = configs[seed]
            output = ROOT / f"experiments/xeditcritic_v4/confirmation_v1/seed_{seed}/{run_id}"
            queues[gpu].append(
                {
                    "job_key": f"critic:{seed}:{run_id}",
                    "component": "critic",
                    "training_seed": seed,
                    "run_id": run_id,
                    "output_directory": str(output),
                    "log_path": str(log_root / f"critic_{seed}_{run_id}.log"),
                    "command": [
                        str(PYTHON), str(critic_trainer), "--config", str(config),
                        "--run-id", run_id, "--physical-gpu-index", str(gpu),
                        "--launch-authorization", str(authorization_root / "critic.json"),
                    ],
                }
            )
            required_by_gpu[gpu] = max(required_by_gpu[gpu], required)
    if "setflow" in eligible:
        preflight = read_json(Path(component_data["setflow"]["preflight"]))
        required = math.ceil(
            (float(preflight["peak_memory_allocated_gib"]) + 2.0) * 1024
        )
        configs = {
            int(read_json(Path(path))["training_seed"]): Path(path)
            for path in manifests["setflow"]["config_paths"]
        }
        for gpu, seed in zip((0, 1, 2), SETFLOW_SEEDS, strict=True):
            config = configs[seed]
            output = ROOT / f"experiments/xeditsetflow_v4/confirmation_v1/seed_{seed}/v4_full"
            queues[gpu].append(
                {
                    "job_key": f"setflow:{seed}:v4_full",
                    "component": "setflow",
                    "training_seed": seed,
                    "run_id": "v4_full",
                    "output_directory": str(output),
                    "log_path": str(log_root / f"setflow_{seed}_v4_full.log"),
                    "command": [
                        str(PYTHON), str(setflow_trainer), "--config", str(config),
                        "--run-id", "v4_full", "--physical-gpu-index", str(gpu),
                        "--authorization", str(authorization_root / "setflow.json"),
                    ],
                }
            )
            required_by_gpu[gpu] = max(required_by_gpu[gpu], required)
    for gpu, required in required_by_gpu.items():
        if queues[gpu]:
            require(
                free_memory[gpu] >= required,
                f"GPU {gpu} lacks V4 confirmation training memory",
            )

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xedit_v4_confirmation_training_schedule.v1",
        "status": "FROZEN_CONFIRMATION_TRAINING_SCHEDULE",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "postscreen_runtime": str(postscreen_runtime_path),
        "eligible_components": eligible,
        "skipped_no_go_components": [
            value for value in ("critic", "setflow") if value not in eligible
        ],
        "gpu_free_memory_mib_before_launch": free_memory,
        "required_free_memory_mib_by_gpu": required_by_gpu,
        "gpu_queues": [
            {"physical_gpu_index": gpu, "jobs": jobs}
            for gpu, jobs in queues.items()
            if jobs
        ],
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(CONFIRMATION_SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xedit_v4_confirmation_training_launch.v1",
        "status": "V4_CONFIRMATION_TRAINING_SCHEDULER_LAUNCHED",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "scheduler_pid": process.pid,
        "eligible_components": eligible,
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
