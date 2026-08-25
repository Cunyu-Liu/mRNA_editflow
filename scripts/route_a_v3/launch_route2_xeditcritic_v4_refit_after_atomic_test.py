#!/usr/bin/env python3
"""Prepare, authorize, and launch exactly three V4 all-Development refits."""

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
REFIT_SCHEDULER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py"
)
REFIT_SEEDS = (20260908, 20260909, 20260910)


class XEditCriticV4RefitLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV4RefitLaunchError(message)


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


def select_refit_gpus(
    free_memory: Mapping[int, int], *, required_mib: int
) -> tuple[int, int, int]:
    candidates = sorted(
        (
            gpu
            for gpu in range(6)
            if int(free_memory.get(gpu, -1)) >= required_mib
        ),
        key=lambda gpu: (-int(free_memory[gpu]), gpu),
    )
    require(len(candidates) >= 3, "fewer than three GPU 0–5 satisfy refit memory")
    return candidates[0], candidates[1], candidates[2]


def refit_decision(
    atomic_runtime: Mapping[str, Any], receipt: Mapping[str, Any] | None
) -> str:
    if atomic_runtime.get("terminal_artifact_kind") != "RESULT":
        return "REFIT_NOT_AUTHORIZED_ATOMIC_TEST_TECHNICAL_FAILURE"
    if receipt is None or receipt.get("status") != "XEDITCRITIC_V4_POSTTEST_AUTHORIZED":
        return "REFIT_NOT_AUTHORIZED_FROZEN_TEST_NO_GO"
    require(
        receipt.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1"
        and receipt.get("required_seeds") == list(REFIT_SEEDS)
        and receipt.get("frozen_test_gate_status")
        == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        and receipt.get("all_development_refit_authorized") is True
        and int(receipt.get("development_test_access_event_count", -1)) == 1
        and receipt.get("general_test_projection_persisted") is False
        and receipt.get("test_bottom_six_cache_persisted") is False
        and receipt.get("development_test_metrics_in_receipt") is False
        and receipt.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 outcome-free posttest receipt changed",
    )
    return "LAUNCH_EXACT_THREE_REFITS"


def validate_refit_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_refit_job_manifest.v1"
        and payload.get("status")
        == "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED"
        and payload.get("required_seeds") == list(REFIT_SEEDS)
        and int(payload.get("refit_pass_count", -1)) == 8
        and int(payload.get("job_count", -1)) == 3,
        "Critic V4 refit manifest changed",
    )
    jobs = payload.get("jobs", [])
    require(
        len(jobs) == 3
        and {(int(job["seed"]), job["run_id"]) for job in jobs}
        == {(seed, "v4_full") for seed in REFIT_SEEDS},
        "Critic V4 refit job set changed",
    )
    return list(jobs)


def run(head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(PYTHON.is_file() and REFIT_SCHEDULER.is_file(), "formal refit runtime is absent")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    atomic_runtime_path = ROOT / f"experiments/xedit_v4/atomic_test_launch_{head}/runtime.json"
    atomic_runtime = read_json(atomic_runtime_path)
    require(
        atomic_runtime.get("status") == "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TERMINAL"
        and atomic_runtime.get("git_head") == head
        and atomic_runtime.get("active_performance_output_read") is False
        and atomic_runtime.get("terminal_payload_content_read_by_wrapper") is False
        and int(atomic_runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 atomic TEST runtime is not an isolated exact-HEAD terminal",
    )
    output_directory = Path(atomic_runtime["output_directory"])
    receipt_path = output_directory / "posttest_authorization_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.is_file() else None
    decision = refit_decision(atomic_runtime, receipt)
    runtime_root = ROOT / f"experiments/xedit_v4/refit_execution_{head}"
    require(not runtime_root.exists(), "Critic V4 refit execution runtime exists")
    if decision != "LAUNCH_EXACT_THREE_REFITS":
        runtime_root.mkdir(parents=True)
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_launch.v1",
            "status": decision,
            "git_head": head,
            "refit_jobs_launched": 0,
            "loso_authorized": False,
            "development_test_access_event_count": int(
                atomic_runtime.get("terminal_artifact_kind") == "RESULT"
            ),
            "development_test_outcome_reads_during_refit": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        write_atomic(runtime_root / "launch.json", result)
        return result

    protocol = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
    protocol_payload = read_json(protocol)
    config_root = Path(protocol_payload["all_development_refit"]["runtime_config_root"])
    run_root = Path(protocol_payload["all_development_refit"]["run_root"])
    manifest_output = Path(protocol_payload["all_development_refit"]["terminal_manifest_output"])
    for path in (config_root, run_root, manifest_output):
        require(not path.exists(), f"Critic V4 refit artifact already exists: {path}")
    preflight = read_json(Path(protocol_payload["formal_preflight_path"]))
    required_mib = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
    )
    free_memory = gpu_free_memory_mib()
    require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
    selected_gpus = select_refit_gpus(free_memory, required_mib=required_mib)
    prepare = WORKTREE / "scripts/route_a_v3/prepare_route2_xeditcritic_v4_posttest_configs.py"
    authorize = WORKTREE / "scripts/route_a_v3/authorize_route2_xeditcritic_v4_posttest.py"
    adjudicator = WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_posttest.py"
    trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    command([str(PYTHON), str(prepare), "--protocol", str(protocol), "--mode", "REFIT"])
    manifest_path = config_root / "manifest.json"
    jobs = validate_refit_manifest(read_json(manifest_path))
    command([str(PYTHON), str(authorize), "--protocol", str(protocol), "--stage", "REFIT"])
    authorization = Path(protocol_payload["all_development_refit"]["authorization_output"])
    authorized = read_json(authorization)
    require(
        authorized.get("status") == "XEDITCRITIC_V4_REFIT_LAUNCH_AUTHORIZED"
        and authorized.get("authorized_git_head") == head
        and authorized.get("authorized_seeds") == list(REFIT_SEEDS)
        and authorized.get("authorized_run_ids") == ["v4_full"]
        and authorized.get("all_three_refits_complete") is False
        and int(authorized.get("development_test_access_event_count_before_posttest", -1)) == 1
        and int(authorized.get("development_test_outcome_reads_during_posttest", -1)) == 0
        and int(authorized.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 refit launch authorization changed",
    )
    queues: dict[int, list[dict[str, Any]]] = {}
    log_root = ROOT / f"logs/xedit_v4/refit_execution_{head}"
    for job, gpu in zip(
        sorted(jobs, key=lambda row: int(row["seed"])), selected_gpus, strict=True
    ):
        queues.setdefault(gpu, []).append(
            {
                "job_key": f"refit:{job['seed']}:v4_full",
                "seed": int(job["seed"]),
                "summary_path": job["summary_path"],
                "failure_path": job["failure_path"],
                "log_path": str(log_root / f"seed_{job['seed']}_v4_full.log"),
                "command": [
                    str(PYTHON), str(trainer),
                    "--config", job["config_path"],
                    "--run-id", "v4_full",
                    "--physical-gpu-index", str(gpu),
                    "--launch-authorization", str(authorization),
                ],
            }
        )
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_schedule.v1",
        "status": "FROZEN_EXACT_THREE_REFIT_SCHEDULE",
        "git_head": head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "gpu_free_memory_mib_before_launch": free_memory,
        "required_free_memory_mib": required_mib,
        "selected_physical_gpus": list(selected_gpus),
        "gpu_queues": [
            {"physical_gpu_index": gpu, "jobs": rows}
            for gpu, rows in sorted(queues.items())
        ],
        "adjudication": {
            "manifest_path": str(manifest_output),
            "failure_path": str(runtime_root / "refit_adjudication.failed.json"),
            "log_path": str(log_root / "refit_adjudication.log"),
            "command": [
                str(PYTHON), str(adjudicator),
                "--manifest", str(manifest_path),
                "--mode", "REFIT",
                "--output", str(manifest_output),
            ],
        },
        "development_test_access_event_count_before_refit": 1,
        "development_test_outcome_reads_during_refit": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    schedule_path = runtime_root / "schedule.json"
    write_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(REFIT_SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_launch.v1",
        "status": "XEDITCRITIC_V4_EXACT_THREE_REFITS_LAUNCHED",
        "git_head": head,
        "scheduler_pid": process.pid,
        "refit_jobs_launched": 3,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
        "development_test_access_event_count_before_refit": 1,
        "development_test_outcome_reads_during_refit": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
