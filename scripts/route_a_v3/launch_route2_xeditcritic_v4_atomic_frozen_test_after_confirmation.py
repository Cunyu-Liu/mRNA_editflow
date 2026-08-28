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
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
JOB_RUNNER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test_job.py"
)
ATOMIC_TEST_RUNNER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test.py"
)
GPU_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
)


class XEditCriticV4AtomicTestLaunchError(RuntimeError):
    pass


class XEditCriticV4AtomicTestGpuInventoryError(
    XEditCriticV4AtomicTestLaunchError
):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        return_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        missing_physical_gpus: tuple[int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.command_line = GPU_INVENTORY_COMMAND
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.missing_physical_gpus = missing_physical_gpus


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


def write_wrapper_launch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    command_line: Sequence[str],
    job_path: Path,
    runtime_path: Path,
    created_artifacts: Mapping[str, Path],
    error: Exception,
) -> None:
    require(
        not path.exists()
        and not path.with_suffix(path.suffix + ".partial").exists(),
        "Critic atomic TEST wrapper launch failure evidence already exists; "
        "use a new retry family",
    )
    write_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v4_atomic_test_"
                "wrapper_launch_failure.v1"
            ),
            "status": (
                "XEDITCRITIC_V4_ATOMIC_TEST_WRAPPER_LAUNCH_TECHNICAL_FAILURE"
            ),
            "failure_stage": "JOB_WRAPPER_PROCESS_LAUNCH",
            "expected_git_head": expected_head,
            "worktree": str(WORKTREE),
            "wrapper_command": list(command_line),
            "job_path": str(job_path),
            "intended_runtime_manifest": str(runtime_path),
            "created_artifact_paths": {
                key: str(value) for key, value in created_artifacts.items()
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "wrapper_started": False,
            "gpu_job_started": False,
            "development_test_access_started": False,
            "development_test_access_event_count": 0,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def spawn_wrapper_with_failure_evidence(
    *,
    failure_path: Path,
    expected_head: str,
    command_line: Sequence[str],
    job_path: Path,
    runtime_path: Path,
    wrapper_log: Path,
    created_artifacts: Mapping[str, Path],
) -> subprocess.Popen[str]:
    stream = wrapper_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command_line),
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as error:
        stream.close()
        write_wrapper_launch_failure_evidence(
            failure_path,
            expected_head=expected_head,
            command_line=command_line,
            job_path=job_path,
            runtime_path=runtime_path,
            created_artifacts=created_artifacts,
            error=error,
        )
        raise XEditCriticV4AtomicTestLaunchError(
            "Critic atomic TEST wrapper process could not start; durable "
            f"technical failure evidence: {failure_path}"
        ) from error
    stream.close()
    return process


def sibling_failure_path(runtime_root: Path) -> Path:
    return runtime_root.with_name(runtime_root.name + ".failed.json")


def require_fresh_prelaunch_family(runtime_root: Path) -> Path:
    failure = sibling_failure_path(runtime_root)
    partial = failure.with_suffix(failure.suffix + ".partial")
    require(
        not failure.exists() and not partial.exists(),
        "Critic prelaunch failure evidence already exists; use a new retry family",
    )
    return failure


def gpu_free_memory_mib(
    required_physical_gpus: Sequence[int],
) -> dict[int, int]:
    try:
        result = subprocess.run(
            list(GPU_INVENTORY_COMMAND),
            cwd=WORKTREE,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise XEditCriticV4AtomicTestGpuInventoryError(
            f"nvidia-smi could not be executed: {error}",
            reason="COMMAND_EXECUTION_FAILED",
        ) from error
    if result.returncode != 0:
        raise XEditCriticV4AtomicTestGpuInventoryError(
            f"nvidia-smi exited with return code {result.returncode}",
            reason="NONZERO_RETURN_CODE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    values: dict[int, int] = {}
    try:
        for line in result.stdout.splitlines():
            index, free = (part.strip() for part in line.split(",", maxsplit=1))
            values[int(index)] = int(free)
    except (TypeError, ValueError) as error:
        raise XEditCriticV4AtomicTestGpuInventoryError(
            f"nvidia-smi inventory could not be parsed: {error}",
            reason="OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ) from error
    required = tuple(int(gpu) for gpu in required_physical_gpus)
    missing = tuple(sorted(set(required) - set(values)))
    if missing:
        raise XEditCriticV4AtomicTestGpuInventoryError(
            f"configured physical GPU inventory is incomplete; missing {list(missing)}",
            reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            missing_physical_gpus=missing,
        )
    return values


def write_prelaunch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    runtime_root: Path,
    error: XEditCriticV4AtomicTestGpuInventoryError,
) -> None:
    require(
        not runtime_root.exists(),
        "Critic atomic TEST runtime root exists before prelaunch failure evidence",
    )
    require(
        not path.exists()
        and not path.with_suffix(path.suffix + ".partial").exists(),
        "Critic prelaunch failure evidence already exists; use a new retry family",
    )
    write_atomic(
        path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_prelaunch_failure.v1",
            "status": "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE",
            "launcher": "atomic",
            "failure_stage": "INVENTORY",
            "expected_head": expected_head,
            "command": list(error.command_line),
            "return_code": error.return_code,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "reason": error.reason,
            "missing_physical_gpus": list(error.missing_physical_gpus),
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_runtime_root": str(runtime_root),
            "runtime_root_created": False,
            "jobs_started": 0,
            "cpu_fallback_used": False,
            "free_memory_gate_applied": False,
            "automatic_retry_attempted": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


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


def select_gpu(
    physical_gpu_index: int, inventory: Mapping[int, int]
) -> int:
    require(0 <= physical_gpu_index <= 5, "atomic TEST GPU must be 0–5")
    require(physical_gpu_index in inventory, "atomic TEST GPU is absent")
    return physical_gpu_index


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
    prelaunch_failure_path = require_fresh_prelaunch_family(runtime_root)
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
        ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_5/preflight.json"
    )
    required_mib = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
    )
    configured_gpu = int(frozen_protocol["physical_gpu_index"])
    try:
        free_memory = gpu_free_memory_mib((configured_gpu,))
    except XEditCriticV4AtomicTestGpuInventoryError as error:
        write_prelaunch_failure_evidence(
            prelaunch_failure_path,
            expected_head=head,
            runtime_root=runtime_root,
            error=error,
        )
        raise
    physical_gpu_index = select_gpu(configured_gpu, free_memory)
    runtime_root.mkdir(parents=True)
    decision_path = runtime_root / "decision.json"
    runtime_protocol = {
        **frozen_protocol,
        "physical_gpu_index": physical_gpu_index,
        "runtime_device_selection": {
            "policy": "FROZEN_PROTOCOL_PHYSICAL_GPU_WITHOUT_FREE_MEMORY_GATE",
            "diagnostic_peak_plus_two_gib_mib": required_mib,
            "free_memory_mib_before_launch": free_memory[physical_gpu_index],
            "free_memory_gate_applied": False,
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
    wrapper_command = [str(PYTHON), str(JOB_RUNNER), "--job", str(job_path)]
    process = spawn_wrapper_with_failure_evidence(
        failure_path=runtime_root / "scheduler_launch.failed.json",
        expected_head=head,
        command_line=wrapper_command,
        job_path=job_path,
        runtime_path=job_runtime,
        wrapper_log=wrapper_log,
        created_artifacts={
            "runtime_protocol": runtime_protocol_path,
            "job": job_path,
            "wrapper_log": wrapper_log,
        },
    )
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_launch.v1",
        "status": "XEDITCRITIC_V4_EXACT_ATOMIC_TEST_LAUNCHED",
        "git_head": head,
        "job_pid": process.pid,
        "physical_gpu_index": physical_gpu_index,
        "diagnostic_peak_plus_two_gib_mib": required_mib,
        "gpu_free_memory_mib_before_launch": free_memory,
        "free_memory_gate_applied": False,
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
