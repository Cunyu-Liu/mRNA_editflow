#!/usr/bin/env python3
"""Launch the one authorized Critic V4 frozen TEST after confirmation PASS."""

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
JOB_RUNNER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test_job.py"
)
ATOMIC_TEST_RUNNER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test.py"
)


class XEditCriticV4AtomicTestLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV4AtomicTestLaunchError(message)


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


def atomic_test_decision(
    posttraining: Mapping[str, Any], gate: Mapping[str, Any] | None
) -> str:
    eligible = posttraining.get("eligible_components", [])
    if "critic" not in eligible:
        return "NOT_AUTHORIZED_CRITIC_SCREEN_NO_GO"
    adjudication = posttraining.get("adjudications", {}).get("critic", {})
    if adjudication.get("terminal_artifact_kind") != "SUMMARY" or gate is None:
        return "NOT_AUTHORIZED_CRITIC_CONFIRMATION_TECHNICAL_FAILURE"
    if gate.get("status") == "XEDITCRITIC_V4_THREE_SEED_NO_GO":
        return "NOT_AUTHORIZED_CRITIC_THREE_SEED_NO_GO"
    require(
        gate.get("status") == "XEDITCRITIC_V4_THREE_SEED_PASS"
        and gate.get("required_seeds") == [20260908, 20260909, 20260910]
        and gate.get("development_test_authorized") is True
        and gate.get("atomic_development_test_only") is True
        and gate.get("additional_seed_authorized") is False
        and gate.get("guidance_authorized") is False
        and int(gate.get("development_test_outcome_reads", -1)) == 0
        and int(gate.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 three-seed gate TEST authorization changed",
    )
    return "LAUNCH_EXACT_ATOMIC_TEST"


def select_gpu(free_memory: Mapping[int, int], *, required_mib: int) -> int:
    candidates = [
        gpu
        for gpu in range(6)
        if int(free_memory.get(gpu, -1)) >= required_mib
    ]
    require(bool(candidates), "no GPU 0–5 has enough memory for atomic TEST")
    return min(candidates, key=lambda gpu: (-int(free_memory[gpu]), gpu))


def run(head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(
        PYTHON.is_file() and JOB_RUNNER.is_file() and ATOMIC_TEST_RUNNER.is_file(),
        "formal atomic TEST runtime is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    posttraining_path = ROOT / f"experiments/xedit_v4/confirmation_posttraining_{head}/runtime.json"
    posttraining = read_json(posttraining_path)
    require(
        posttraining.get("status") == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
        and posttraining.get("git_head") == head
        and posttraining.get("active_performance_output_read") is False
        and int(posttraining.get("development_test_outcome_reads", -1)) == 0
        and int(posttraining.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 confirmation posttraining package is not an isolated exact-HEAD terminal",
    )
    protocol_path = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_frozen_test_protocol_v1.json"
    frozen_protocol = read_json(protocol_path)
    gate_path = Path(frozen_protocol["three_seed_gate_path"])
    gate = read_json(gate_path) if gate_path.is_file() else None
    decision = atomic_test_decision(posttraining, gate)
    runtime_root = ROOT / f"experiments/xedit_v4/atomic_test_launch_{head}"
    require(not runtime_root.exists(), "Critic V4 atomic TEST launch runtime exists")
    if decision != "LAUNCH_EXACT_ATOMIC_TEST":
        runtime_root.mkdir(parents=True)
        decision_path = runtime_root / "decision.json"
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_launch.v1",
            "status": decision,
            "git_head": head,
            "atomic_test_launched": False,
            "development_test_access_event_count": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        write_atomic(decision_path, result)
        return result

    output_directory = Path(frozen_protocol["output_directory"])
    require(not output_directory.exists(), "Critic V4 atomic TEST was already consumed")
    preflight = read_json(
        ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_3/preflight.json"
    )
    required_mib = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
    )
    free_memory = gpu_free_memory_mib()
    require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
    physical_gpu_index = select_gpu(free_memory, required_mib=required_mib)
    runtime_root.mkdir(parents=True)
    decision_path = runtime_root / "decision.json"
    runtime_protocol = {
        **frozen_protocol,
        "physical_gpu_index": physical_gpu_index,
        "runtime_device_selection": {
            "policy": "MAXIMUM_FREE_MEMORY_AMONG_GPU_0_TO_5_MEETING_PREFLIGHT_PLUS_2_GIB",
            "required_free_memory_mib": required_mib,
            "selected_gpu": physical_gpu_index,
            "scientific_configuration_changed": False,
        },
    }
    runtime_protocol_path = runtime_root / "runtime_protocol.json"
    write_atomic(runtime_protocol_path, runtime_protocol)
    log_root = ROOT / f"logs/xedit_v4/atomic_test_launch_{head}"
    log_root.mkdir(parents=True, exist_ok=True)
    job_runtime = runtime_root / "runtime.json"
    job = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_job.v1",
        "status": "FROZEN_ATOMIC_TEST_JOB",
        "git_head": head,
        "worktree": str(WORKTREE),
        "physical_gpu_index": physical_gpu_index,
        "output_directory": str(output_directory),
        "runtime_manifest": str(job_runtime),
        "log_path": str(log_root / "atomic_test.log"),
        "command": [
            str(PYTHON),
            str(ATOMIC_TEST_RUNNER),
            "--protocol",
            str(runtime_protocol_path),
        ],
    }
    job_path = runtime_root / "job.json"
    write_atomic(job_path, job)
    wrapper_log = log_root / "job_wrapper.log"
    stream = wrapper_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(JOB_RUNNER), "--job", str(job_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_launch.v1",
        "status": "XEDITCRITIC_V4_EXACT_ATOMIC_TEST_LAUNCHED",
        "git_head": head,
        "job_pid": process.pid,
        "physical_gpu_index": physical_gpu_index,
        "required_free_memory_mib": required_mib,
        "gpu_free_memory_mib_before_launch": free_memory,
        "runtime_protocol": str(runtime_protocol_path),
        "job_runtime": str(job_runtime),
        "wrapper_log": str(wrapper_log),
        "atomic_test_launched": True,
        "development_test_access_event_count_before_job": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(decision_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
