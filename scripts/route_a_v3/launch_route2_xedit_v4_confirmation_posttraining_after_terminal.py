#!/usr/bin/env python3
"""Launch exact V4 confirmation validation and gates after terminal training."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
POSTTRAINING_SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
)
CRITIC_SEEDS = (20260908, 20260909, 20260910)
SETFLOW_SEEDS = (20260912, 20260913, 20260914)
CHECKPOINT_PASSES = (4, 6, 8, 10)


class XEditV4ConfirmationPosttrainingLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4ConfirmationPosttrainingLaunchError(message)


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


def expected_training_job_keys(eligible_components: tuple[str, ...]) -> set[str]:
    require(
        bool(eligible_components)
        and set(eligible_components).issubset({"critic", "setflow"}),
        "confirmation eligible component set changed",
    )
    jobs: set[str] = set()
    if "critic" in eligible_components:
        jobs.update(
            f"critic:{seed}:{run_id}"
            for seed in CRITIC_SEEDS
            for run_id in ("v4_full", "c0_v4")
        )
    if "setflow" in eligible_components:
        jobs.update(f"setflow:{seed}:v4_full" for seed in SETFLOW_SEEDS)
    return jobs


def validation_assignments(
    successful_seeds: tuple[int, ...],
) -> dict[int, list[tuple[int, int]]]:
    require(
        set(successful_seeds).issubset(set(SETFLOW_SEEDS)),
        "SetFlow confirmation validation seed changed",
    )
    assignments = {gpu: [] for gpu in range(6)}
    jobs = [
        (seed, checkpoint_pass)
        for seed in SETFLOW_SEEDS
        if seed in successful_seeds
        for checkpoint_pass in CHECKPOINT_PASSES
    ]
    for index, job in enumerate(jobs):
        assignments[index % 6].append(job)
    return {gpu: rows for gpu, rows in assignments.items() if rows}


def validate_confirmation_runtime(
    runtime: Mapping[str, Any], *, head: str
) -> tuple[str, ...]:
    require(
        runtime.get("status") == "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
        and runtime.get("git_head") == head,
        "V4 confirmation training package is not an exact-HEAD terminal",
    )
    eligible = tuple(runtime.get("eligible_components", []))
    require(
        eligible in {
            ("critic",),
            ("setflow",),
            ("critic", "setflow"),
        }
        and set(runtime.get("jobs", {})) == expected_training_job_keys(eligible),
        "V4 confirmation training job set changed",
    )
    require(
        all(
            row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
            for row in runtime["jobs"].values()
        ),
        "V4 confirmation training package lacks an exact terminal artifact",
    )
    require(
        runtime.get("active_performance_output_read") is False
        and int(runtime.get("development_test_outcome_reads", -1)) == 0
        and int(runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 confirmation runtime reports an active-performance or protected read",
    )
    return eligible


def run(head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(PYTHON.is_file(), "formal Python is absent")
    require(POSTTRAINING_SCHEDULER.is_file(), "current-HEAD posttraining scheduler is absent")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )

    training_root = ROOT / f"experiments/xedit_v4/confirmation_training_{head}"
    training_runtime_path = training_root / "runtime.json"
    eligible = validate_confirmation_runtime(
        read_json(training_runtime_path), head=head
    )
    runtime_root = ROOT / f"experiments/xedit_v4/confirmation_posttraining_{head}"
    log_root = ROOT / f"logs/xedit_v4/confirmation_posttraining_{head}"
    require(not runtime_root.exists(), "V4 confirmation posttraining runtime exists")

    authorization_root = ROOT / f"authorizations/xedit_v4/confirmation_{head}"
    adjudications: dict[str, dict[str, Any]] = {}
    validation_queues: list[dict[str, Any]] = []
    free_memory: dict[int, int] = {}
    required_free_memory_mib = 0

    if "critic" in eligible:
        protocol = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_confirmation_protocol_v1.json"
        manifest = ROOT / "runtime_configs/xeditcritic_v4/confirmation_v1/manifest.json"
        preflight = ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_2/preflight.json"
        gate = Path(read_json(protocol)["confirmation_gate_output"])
        failure = runtime_root / "critic_adjudication.failed.json"
        require(
            not gate.exists()
            and not gate.with_suffix(gate.suffix + ".partial").exists()
            and not failure.exists(),
            "Critic confirmation adjudication exists",
        )
        authorization = authorization_root / "critic.json"
        for path in (manifest, preflight, authorization):
            require(path.is_file(), f"Critic confirmation prerequisite is absent: {path}")
        authorization_payload = read_json(authorization)
        require(
            authorization_payload.get("status")
            == "XEDITCRITIC_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
            and authorization_payload.get("authorized_git_head") == head
            and authorization_payload.get("authorized_seeds") == list(CRITIC_SEEDS)
            and authorization_payload.get("authorized_run_ids")
            == ["v4_full", "c0_v4"]
            and authorization_payload.get("additional_seed_authorized") is False
            and authorization_payload.get("development_test_authorized") is False
            and int(authorization_payload.get("development_test_outcome_reads", -1)) == 0
            and int(authorization_payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "Critic confirmation authorization changed",
        )
        adjudications["critic"] = {
            "gate_path": str(gate),
            "failure_path": str(failure),
            "log_path": str(log_root / "critic_adjudication.log"),
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_confirmation.py"),
                "--protocol", str(protocol),
                "--config-manifest", str(manifest),
                "--preflight", str(preflight),
            ],
        }

    if "setflow" in eligible:
        protocol = WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
        manifest_path = ROOT / "runtime_configs/xeditsetflow_v4/confirmation_v1/manifest.json"
        authorization = authorization_root / "setflow.json"
        preflight_path = ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_2/preflight.json"
        for path in (manifest_path, authorization, preflight_path):
            require(path.is_file(), f"SetFlow confirmation prerequisite is absent: {path}")
        authorization_payload = read_json(authorization)
        require(
            authorization_payload.get("status")
            == "XEDITSETFLOW_V4_CONFIRMATION_LAUNCH_AUTHORIZED"
            and authorization_payload.get("authorized_git_head") == head
            and authorization_payload.get("authorized_seeds") == list(SETFLOW_SEEDS),
            "SetFlow confirmation authorization changed",
        )
        require(
            authorization_payload.get("additional_seed_authorized") is False
            and authorization_payload.get("development_test_authorized") is False
            and int(authorization_payload.get("development_test_outcome_reads", -1)) == 0
            and int(authorization_payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "SetFlow confirmation authorization protected boundary changed",
        )
        manifest = read_json(manifest_path)
        require(
            manifest.get("required_seeds") == list(SETFLOW_SEEDS)
            and len(manifest.get("config_paths", [])) == 3,
            "SetFlow confirmation config manifest changed",
        )
        configs = {
            int(payload["training_seed"]): (Path(path), payload)
            for path in manifest["config_paths"]
            for payload in (read_json(Path(path)),)
        }
        require(set(configs) == set(SETFLOW_SEEDS), "SetFlow confirmation configs changed")
        successful: list[int] = []
        for seed in SETFLOW_SEEDS:
            config_path, config = configs[seed]
            training = Path(config["output_root"]) / "v4_full"
            summary = training / "training_summary.json"
            failure = training / "failure.json"
            require(
                summary.exists() != failure.exists(),
                f"SetFlow confirmation training is not exactly terminal: {seed}",
            )
            if summary.exists():
                successful.append(seed)
                for checkpoint_pass in CHECKPOINT_PASSES:
                    output = (
                        Path(config["validation_output_root"])
                        / "v4_full"
                        / f"pass_{checkpoint_pass}"
                    )
                    validation_failure = output.with_name(output.name + ".failed.json")
                    require(
                        not output.exists()
                        and not validation_failure.exists()
                        and not validation_failure.with_suffix(
                            validation_failure.suffix + ".partial"
                        ).exists(),
                        "SetFlow confirmation validation output exists",
                    )
        assignments = validation_assignments(tuple(successful))
        if assignments:
            preflight = read_json(preflight_path)
            required_free_memory_mib = math.ceil(
                (float(preflight["peak_memory_allocated_gib"]) + 2.0) * 1024
            )
            free_memory = gpu_free_memory_mib()
            require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
            validator = WORKTREE / "scripts/route_a_v3/validate_route2_xeditsetflow_v4_checkpoint.py"
            for gpu, rows in assignments.items():
                require(
                    free_memory[gpu] >= required_free_memory_mib,
                    f"GPU {gpu} lacks SetFlow confirmation validation memory",
                )
                jobs: list[dict[str, Any]] = []
                for seed, checkpoint_pass in rows:
                    config_path, config = configs[seed]
                    output = (
                        Path(config["validation_output_root"])
                        / "v4_full"
                        / f"pass_{checkpoint_pass}"
                    )
                    jobs.append(
                        {
                            "job_key": f"setflow:{seed}:pass_{checkpoint_pass}",
                            "training_seed": seed,
                            "checkpoint_pass": checkpoint_pass,
                            "physical_gpu_index": gpu,
                            "terminal_summary": str(output / "validation_summary.json"),
                            "terminal_failure": str(output.with_name(output.name + ".failed.json")),
                            "log_path": str(log_root / f"setflow_{seed}_pass_{checkpoint_pass}.log"),
                            "command": [
                                str(PYTHON), str(validator),
                                "--config", str(config_path),
                                "--run-id", "v4_full",
                                "--checkpoint-pass", str(checkpoint_pass),
                                "--authorization", str(authorization),
                                "--physical-gpu-index", str(gpu),
                            ],
                        }
                    )
                validation_queues.append({"physical_gpu_index": gpu, "jobs": jobs})
        gate = Path(read_json(protocol)["confirmation_gate_output"])
        gate_failure = runtime_root / "setflow_adjudication.failed.json"
        require(
            not gate.exists()
            and not gate.with_suffix(gate.suffix + ".partial").exists()
            and not gate_failure.exists(),
            "SetFlow confirmation adjudication exists",
        )
        adjudications["setflow"] = {
            "gate_path": str(gate),
            "failure_path": str(gate_failure),
            "log_path": str(log_root / "setflow_adjudication.log"),
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v4_confirmation.py"),
                "--protocol", str(protocol),
                "--config-manifest", str(manifest_path),
            ],
        }

    require(set(adjudications) == set(eligible), "confirmation adjudication scope changed")
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xedit_v4_confirmation_posttraining_schedule.v1",
        "status": "FROZEN_CONFIRMATION_POSTTRAINING_SCHEDULE",
        "git_head": head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "confirmation_training_runtime": str(training_runtime_path),
        "eligible_components": list(eligible),
        "validation_queues": validation_queues,
        "adjudications": adjudications,
        "gpu_free_memory_mib_before_launch": free_memory,
        "setflow_required_free_memory_mib": required_free_memory_mib,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    coordinator_log = log_root / "coordinator.log"
    stream = coordinator_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(POSTTRAINING_SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xedit_v4_confirmation_posttraining_launch.v1",
        "status": "V4_CONFIRMATION_POSTTRAINING_COORDINATOR_LAUNCHED",
        "git_head": head,
        "coordinator_pid": process.pid,
        "eligible_components": list(eligible),
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "coordinator_log": str(coordinator_log),
        "development_test_authorized": False,
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
