#!/usr/bin/env python3
"""Launch the six repaired Critic V4 controls after repaired full is terminal."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


TRAINING_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
TRAINING_WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_v403_critic_rng_replay_20260827"
)
ORCHESTRATION_WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BASE_CONFIG = (
    TRAINING_WORKTREE
    / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
)
TRAINER = (
    TRAINING_WORKTREE
    / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
)
SCHEDULER = (
    ORCHESTRATION_WORKTREE
    / "scripts/route_a_v3/"
    "run_route2_xeditcritic_v403_control_recovery_scheduler.py"
)
PREFLIGHT = (
    ROOT
    / "experiments/xeditcritic_v4/screen_seed_20260907/"
    "preflight_attempt_5/preflight.json"
)
SMOKE = (
    ROOT
    / "audits/xeditcritic_v4/"
    f"v403_rng_replay_smoke_{TRAINING_GIT_HEAD}.json"
)
CURRENT_FULL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_rng_replay_fix_{TRAINING_GIT_HEAD}"
)
CURRENT_FULL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_rng_replay_fix_runner_{TRAINING_GIT_HEAD}/runtime.json"
)
HISTORICAL_C0_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v402_recovery_runner_{C0_GIT_HEAD}"
)
CONTROL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_control_recovery_{TRAINING_GIT_HEAD}"
)
RUNTIME_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_control_recovery_runner_{TRAINING_GIT_HEAD}"
)
AUTHORIZATION_ROOT = (
    ROOT
    / "authorizations/xeditcritic_v4/"
    f"v403_control_recovery_{TRAINING_GIT_HEAD}"
)
LOG_ROOT = (
    ROOT
    / "logs/xeditcritic_v4/"
    f"v403_control_recovery_{TRAINING_GIT_HEAD}"
)
TRANSITION_GATE = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_cross_root_{TRAINING_GIT_HEAD}/screen_gate.json"
)

ALL_RUN_IDS = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
CONTROL_RUN_IDS = ALL_RUN_IDS[2:]
PHYSICAL_GPU_INDICES = (0, 1, 2, 3, 4, 5)
GPU_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index",
    "--format=csv,noheader,nounits",
)
CUDA_BF16_PROBE_SOURCE = """
import json
import sys
import torch

indices = [int(value) for value in sys.argv[1:]]
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
if torch.cuda.device_count() <= max(indices):
    raise RuntimeError("physical CUDA inventory is incomplete")
rows = []
for index in indices:
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if "A100" not in name:
        raise RuntimeError(f"physical GPU {index} is not A100: {name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"physical GPU {index} lacks BF16 support")
    value = (torch.ones(1, device=device, dtype=torch.bfloat16) * 2).item()
    if value != 2.0:
        raise RuntimeError(f"physical GPU {index} BF16 tensor probe failed")
    rows.append({"physical_gpu_index": index, "device_name": name,
                 "bf16_supported": True, "bf16_tensor_probe": True})
print(json.dumps(rows, sort_keys=True))
"""
FROZEN_FULL_SUMMARY_IDENTITY = {
    "seed": 20260907,
    "pass_count": 8,
    "selected_pass": 8,
    "update_count": 22416,
    "selection_policy": "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION",
    "train_record_count": 89580,
    "validation_record_count": 18293,
    "effective_batch_size": 32,
    "physical_batch_size": 32,
}


class XEditCriticV403ControlRecoveryLaunchError(RuntimeError):
    pass


class XEditCriticV403GpuInventoryError(
    XEditCriticV403ControlRecoveryLaunchError
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


class XEditCriticV403CudaBf16ProbeError(
    XEditCriticV403ControlRecoveryLaunchError
):
    def __init__(
        self, message: str, *, cpu_fallback_used: bool = False
    ) -> None:
        super().__init__(message)
        self.cpu_fallback_used = cpu_fallback_used


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403ControlRecoveryLaunchError(message)


def validated_physical_gpu_index(value: Any, label: str) -> int:
    require(
        type(value) is int and value in PHYSICAL_GPU_INDICES,
        f"{label} physical GPU is outside the frozen 0-5 scope",
    )
    return value


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(
    arguments: Sequence[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


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


def physical_gpu_inventory(
    required_physical_gpus: Sequence[int] = PHYSICAL_GPU_INDICES,
) -> tuple[int, ...]:
    try:
        result = subprocess.run(
            list(GPU_INVENTORY_COMMAND),
            cwd=TRAINING_WORKTREE,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise XEditCriticV403GpuInventoryError(
            f"nvidia-smi could not be executed: {error}",
            reason="COMMAND_EXECUTION_FAILED",
        ) from error
    if result.returncode != 0:
        raise XEditCriticV403GpuInventoryError(
            f"nvidia-smi exited with return code {result.returncode}",
            reason="NONZERO_RETURN_CODE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    try:
        values = tuple(int(line.strip()) for line in result.stdout.splitlines())
    except (TypeError, ValueError) as error:
        raise XEditCriticV403GpuInventoryError(
            f"nvidia-smi inventory could not be parsed: {error}",
            reason="OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ) from error
    required = tuple(int(gpu) for gpu in required_physical_gpus)
    missing = tuple(sorted(set(required) - set(values)))
    if missing:
        raise XEditCriticV403GpuInventoryError(
            f"configured physical GPU inventory is incomplete; missing {list(missing)}",
            reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            missing_physical_gpus=missing,
        )
    return values


def failure_process_details(
    error: Exception, *, command_line: Sequence[str]
) -> dict[str, Any]:
    if isinstance(error, XEditCriticV403GpuInventoryError):
        return {
            "command": list(error.command_line),
            "return_code": error.return_code,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "reason": error.reason,
            "missing_physical_gpus": list(error.missing_physical_gpus),
        }
    if isinstance(error, subprocess.CalledProcessError):
        raw_command = error.cmd
        recorded_command = (
            list(raw_command)
            if isinstance(raw_command, (list, tuple))
            else [str(raw_command)]
        )
        return {
            "command": recorded_command,
            "return_code": error.returncode,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "reason": "CUDA_BF16_PROBE_CHILD_FAILED",
            "missing_physical_gpus": [],
        }
    return {
        "command": list(command_line),
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "reason": (
            "COMMAND_EXECUTION_FAILED"
            if isinstance(error, OSError)
            else "CUDA_BF16_PROBE_OUTPUT_INVALID"
        ),
        "missing_physical_gpus": [],
    }


def write_prelaunch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    failure_stage: str,
    runtime_root: Path,
    command_line: Sequence[str],
    error: Exception,
) -> None:
    require(
        failure_stage in {"INVENTORY", "CUDA_BF16_PROBE"},
        "unknown Critic controls prelaunch failure stage",
    )
    require(
        not runtime_root.exists(),
        "Critic controls runtime root exists before prelaunch failure evidence",
    )
    require(
        not path.exists()
        and not path.with_suffix(path.suffix + ".partial").exists(),
        "Critic prelaunch failure evidence already exists; use a new retry family",
    )
    details = failure_process_details(error, command_line=command_line)
    cpu_fallback_used = getattr(error, "cpu_fallback_used", False)
    require(
        type(cpu_fallback_used) is bool,
        "CUDA/BF16 probe CPU-fallback observation is not boolean",
    )
    write_atomic(
        path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_prelaunch_failure.v1",
            "status": "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE",
            "launcher": "controls",
            "failure_stage": failure_stage,
            "expected_head": expected_head,
            "training_git_head": TRAINING_GIT_HEAD,
            **details,
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_runtime_root": str(runtime_root),
            "runtime_root_created": False,
            "jobs_started": 0,
            "cpu_fallback_used": cpu_fallback_used,
            "free_memory_gate_applied": False,
            "automatic_retry_attempted": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def require_zero_protected_reads(payload: Mapping[str, Any], label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0,
        f"{label} reports a Development TEST read",
    )
    require(
        int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a new Evaluation read",
    )


def exact_terminal_kind(output_directory: Path) -> str | None:
    summary = output_directory / "run_summary.json"
    failure = output_directory / "failure.json"
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def validate_current_full_terminal(
    output_root: Path | None = None,
    runtime_path: Path | None = None,
) -> dict[str, Any]:
    """Read full only after its directory is exactly one terminal SUMMARY."""

    output_root = CURRENT_FULL_OUTPUT_ROOT if output_root is None else output_root
    runtime_path = CURRENT_FULL_RUNTIME if runtime_path is None else runtime_path
    output = output_root / "v4_full"
    kind = exact_terminal_kind(output)
    require(
        kind == "SUMMARY",
        "current V4.0.3 full is not exact terminal SUMMARY",
    )
    runtime = read_json(runtime_path)
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1"
        and runtime.get("status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and runtime.get("terminal_artifact_kind") == "SUMMARY"
        and int(runtime.get("return_code", -1)) == 0
        and runtime.get("run_id") == "v4_full"
        and runtime.get("git_head") == TRAINING_GIT_HEAD,
        "current V4.0.3 full runtime is not exact successful terminal",
    )
    require_zero_protected_reads(runtime, "current full runtime")
    runtime_gpu_index = validated_physical_gpu_index(
        runtime.get("physical_gpu_index"), "current full runtime"
    )
    summary = read_json(output / "run_summary.json")
    require(
        summary.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
        and summary.get("status")
        == "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE"
        and summary.get("run_id") == "v4_full"
        and summary.get("model_kind") == "V4-FULL"
        and summary.get("precision")
        == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and summary.get("cpu_fallback_used") is False
        and summary.get("parameter_changed") is True,
        "current V4.0.3 full summary identity is invalid",
    )
    require(
        all(
            summary.get(key) == expected
            for key, expected in FROZEN_FULL_SUMMARY_IDENTITY.items()
        ),
        "current V4.0.3 full summary does not match frozen training identity",
    )
    summary_gpu_index = validated_physical_gpu_index(
        summary.get("physical_gpu_index"), "current full summary"
    )
    require(
        summary_gpu_index == runtime_gpu_index,
        "current full runtime and summary physical GPUs disagree",
    )
    require_zero_protected_reads(summary, "current full summary")
    authorization_path = Path(str(summary.get("launch_authorization_path", "")))
    require(authorization_path.is_file(), "current full authorization is absent")
    authorization = read_json(authorization_path)
    authorized_run_ids = authorization.get("authorized_run_ids")
    require(
        authorization.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        and authorization.get("status")
        == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
        and authorization.get("authorized_git_head") == TRAINING_GIT_HEAD
        and isinstance(authorized_run_ids, list)
        and "v4_full" in authorized_run_ids,
        "current full launch authorization identity is invalid",
    )
    require_zero_protected_reads(authorization, "current full authorization")
    recovery = authorization.get("v403_rng_replay_recovery")
    require(
        isinstance(recovery, Mapping) and recovery.get("run_id") == "v4_full",
        "current full launch authorization lacks its V4.0.3 recovery identity",
    )
    authorization_gpu_index = validated_physical_gpu_index(
        recovery.get("physical_gpu_index"), "current full authorization"
    )
    require(
        authorization_gpu_index == runtime_gpu_index == summary_gpu_index,
        "current full runtime, summary, and authorization physical GPUs disagree",
    )
    return {
        "summary": summary,
        "runtime": runtime,
        "authorization_path": str(authorization_path),
        "physical_gpu_index": summary_gpu_index,
    }


def validate_historical_c0_terminal(
    output_root: Path | None = None,
) -> dict[str, Any]:
    output_root = HISTORICAL_C0_OUTPUT_ROOT if output_root is None else output_root
    output = output_root / "c0_v4"
    require(
        exact_terminal_kind(output) == "SUMMARY",
        "historical matched C0 is not exact terminal SUMMARY",
    )
    summary = read_json(output / "run_summary.json")
    require(
        summary.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
        and summary.get("status")
        == "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE"
        and summary.get("run_id") == "c0_v4"
        and summary.get("model_kind") == "C0-V4"
        and summary.get("precision")
        == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and summary.get("cpu_fallback_used") is False,
        "historical matched C0 summary identity is invalid",
    )
    require_zero_protected_reads(summary, "historical matched C0 summary")
    return summary


def validate_training_source(expected_orchestration_head: str) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_orchestration_head) is not None,
        "expected orchestration HEAD is invalid",
    )
    require(
        PYTHON.is_file()
        and BASE_CONFIG.is_file()
        and TRAINER.is_file()
        and SCHEDULER.is_file(),
        "V4.0.3 training or orchestration source is incomplete",
    )
    require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden",
    )
    training_head = command(
        ["git", "rev-parse", "HEAD"], cwd=TRAINING_WORKTREE
    ).stdout.strip()
    require(training_head == TRAINING_GIT_HEAD, "training worktree HEAD changed")
    require(
        not command(
            ["git", "status", "--porcelain"], cwd=TRAINING_WORKTREE
        ).stdout.strip(),
        "training worktree is dirty",
    )
    orchestration_head = command(
        ["git", "rev-parse", "HEAD"], cwd=ORCHESTRATION_WORKTREE
    ).stdout.strip()
    require(
        orchestration_head == expected_orchestration_head,
        "orchestration worktree is at another HEAD",
    )
    require(
        not command(
            ["git", "status", "--porcelain"], cwd=ORCHESTRATION_WORKTREE
        ).stdout.strip(),
        "orchestration worktree is dirty",
    )
    preflight = read_json(PREFLIGHT)
    require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True,
        "frozen Critic preflight is not PASS",
    )
    require_zero_protected_reads(preflight, "frozen Critic preflight")
    smoke = read_json(SMOKE)
    require(
        smoke.get("status")
        == "XEDITCRITIC_V403_FULL_MODEL_RNG_REPLAY_SMOKE_PASS"
        and smoke.get("git_head") == TRAINING_GIT_HEAD
        and smoke.get("strict_replay_prediction_equal") is True
        and smoke.get("retained_graph_prediction_equal_to_replay") is True
        and smoke.get("retained_graph_parameter_gradients_equal_to_replay") is True
        and smoke.get("retained_graph_rng_terminal_state_equal_to_replay") is True,
        "exact f34 RNG replay smoke is absent or invalid",
    )
    require_zero_protected_reads(smoke, "exact f34 RNG replay smoke")
    return {"preflight": preflight, "smoke": smoke}


def cuda_bf16_probe_command(
    physical_gpu_indices: Sequence[int] = PHYSICAL_GPU_INDICES,
) -> list[str]:
    indices = [int(index) for index in physical_gpu_indices]
    require(
        indices == list(PHYSICAL_GPU_INDICES),
        "control recovery requires the frozen physical GPU inventory 0-5",
    )
    return [
        str(PYTHON),
        "-c",
        CUDA_BF16_PROBE_SOURCE,
        *[str(index) for index in indices],
    ]


def probe_cuda_bf16(
    physical_gpu_indices: Sequence[int] = PHYSICAL_GPU_INDICES,
) -> list[dict[str, Any]]:
    indices = [int(index) for index in physical_gpu_indices]
    result = command(
        cuda_bf16_probe_command(indices),
        cwd=TRAINING_WORKTREE,
    )
    rows = json.loads(result.stdout)
    if isinstance(rows, list) and any(
        isinstance(row, Mapping) and row.get("cpu_fallback_used") is True
        for row in rows
    ):
        raise XEditCriticV403CudaBf16ProbeError(
            "CUDA/BF16 probe observed CPU fallback",
            cpu_fallback_used=True,
        )
    require(
        isinstance(rows, list)
        and [row.get("physical_gpu_index") for row in rows] == indices
        and all(row.get("bf16_supported") is True for row in rows)
        and all(row.get("bf16_tensor_probe") is True for row in rows)
        and all(row.get("cpu_fallback_used", False) is False for row in rows),
        "CUDA/BF16 inventory probe is incomplete",
    )
    return rows


def build_recovery_config(
    base: Mapping[str, Any],
    output_root: Path = CONTROL_OUTPUT_ROOT,
    screen_gate_output: Path = TRANSITION_GATE,
) -> dict[str, Any]:
    run_ids = tuple(str(row["run_id"]) for row in base["required_screen_runs"])
    require(run_ids == ALL_RUN_IDS, "frozen Critic eight-arm order changed")
    recovery = copy.deepcopy(dict(base))
    recovery["output_root"] = str(output_root)
    recovery["screen_gate_output"] = str(screen_gate_output)
    allowed = {"output_root", "screen_gate_output"}
    require(
        {key: value for key, value in base.items() if key not in allowed}
        == {key: value for key, value in recovery.items() if key not in allowed},
        "control recovery changed a scientific config field",
    )
    return recovery


def build_launch_authorization(
    preflight: Mapping[str, Any],
    *,
    expected_orchestration_head: str,
) -> dict[str, Any]:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": TRAINING_GIT_HEAD,
        "preflight_runner_git_head": str(preflight["git_head"]),
        "authorized_run_ids": list(ALL_RUN_IDS),
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cache_online_equivalence_passed": True,
        },
        "v403_control_recovery": {
            "training_code_git_head": TRAINING_GIT_HEAD,
            "training_worktree": str(TRAINING_WORKTREE),
            "orchestration_git_head": expected_orchestration_head,
            "required_control_run_ids": list(CONTROL_RUN_IDS),
            "current_full_exact_terminal_summary": True,
            "historical_c0_exact_terminal_summary_reused": True,
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "scientific_config_changed": False,
            "free_memory_gate_applied": False,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def build_control_schedule(
    *,
    expected_orchestration_head: str,
    config_path: Path,
    authorization_path: Path,
    cuda_bf16_inventory: Sequence[Mapping[str, Any]],
    output_root: Path = CONTROL_OUTPUT_ROOT,
    runtime_manifest: Path = RUNTIME_ROOT / "runtime.json",
    log_root: Path = LOG_ROOT,
) -> dict[str, Any]:
    gpu_indices = [
        int(row["physical_gpu_index"]) for row in cuda_bf16_inventory
    ]
    require(
        gpu_indices == list(PHYSICAL_GPU_INDICES),
        "CUDA/BF16 inventory does not cover physical GPUs 0-5",
    )
    jobs = []
    for run_id, physical_gpu_index in zip(
        CONTROL_RUN_IDS, PHYSICAL_GPU_INDICES, strict=True
    ):
        output_directory = output_root / run_id
        attempt_id = (
            "xeditcritic_v4_screen_seed20260907::"
            f"{run_id}::v403_control_recovery_{TRAINING_GIT_HEAD}"
        )
        jobs.append(
            {
                "run_id": run_id,
                "physical_gpu_index": physical_gpu_index,
                "output_directory": str(output_directory),
                "log_path": str(log_root / f"{run_id}.log"),
                "training_attempt_id": attempt_id,
                "command": [
                    str(PYTHON),
                    str(TRAINER),
                    "--config",
                    str(config_path),
                    "--run-id",
                    run_id,
                    "--physical-gpu-index",
                    str(physical_gpu_index),
                    "--launch-authorization",
                    str(authorization_path),
                    "--training-attempt-id",
                    attempt_id,
                ],
            }
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_control_recovery_schedule.v1"
        ),
        "status": "XEDITCRITIC_V403_CONTROL_RECOVERY_SCHEDULED",
        "orchestration_git_head": expected_orchestration_head,
        "training_code_git_head": TRAINING_GIT_HEAD,
        "training_worktree": str(TRAINING_WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "screen_config": str(config_path),
        "launch_authorization": str(authorization_path),
        "current_full_summary": str(
            CURRENT_FULL_OUTPUT_ROOT / "v4_full/run_summary.json"
        ),
        "historical_c0_summary": str(
            HISTORICAL_C0_OUTPUT_ROOT / "c0_v4/run_summary.json"
        ),
        "cuda_bf16_inventory": [dict(row) for row in cuda_bf16_inventory],
        "jobs": jobs,
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "free_memory_gate_applied": False,
        "terminal_artifact_payloads_read_by_scheduler": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def launch(expected_orchestration_head: str) -> dict[str, Any]:
    # This is intentionally first: a running/failed full causes no writes, probe,
    # worker creation, or control process creation.
    full = validate_current_full_terminal()
    c0 = validate_historical_c0_terminal()
    for path, label in (
        (CONTROL_OUTPUT_ROOT, "control output root"),
        (RUNTIME_ROOT, "control runtime root"),
        (AUTHORIZATION_ROOT, "control authorization root"),
        (TRANSITION_GATE.parent, "cross-root gate root"),
    ):
        require(not path.exists(), f"{label} already exists")
    prelaunch_failure_path = require_fresh_prelaunch_family(RUNTIME_ROOT)
    source = validate_training_source(expected_orchestration_head)
    try:
        physical_gpu_inventory()
    except XEditCriticV403GpuInventoryError as error:
        write_prelaunch_failure_evidence(
            prelaunch_failure_path,
            expected_head=expected_orchestration_head,
            failure_stage="INVENTORY",
            runtime_root=RUNTIME_ROOT,
            command_line=GPU_INVENTORY_COMMAND,
            error=error,
        )
        raise
    probe_command = cuda_bf16_probe_command()
    try:
        inventory = probe_cuda_bf16()
    except Exception as error:
        write_prelaunch_failure_evidence(
            prelaunch_failure_path,
            expected_head=expected_orchestration_head,
            failure_stage="CUDA_BF16_PROBE",
            runtime_root=RUNTIME_ROOT,
            command_line=probe_command,
            error=error,
        )
        raise
    config = build_recovery_config(read_json(BASE_CONFIG))
    authorization = build_launch_authorization(
        source["preflight"],
        expected_orchestration_head=expected_orchestration_head,
    )

    AUTHORIZATION_ROOT.mkdir(parents=True)
    RUNTIME_ROOT.mkdir(parents=True)
    LOG_ROOT.mkdir(parents=True)
    config_path = AUTHORIZATION_ROOT / "screen_config.json"
    authorization_path = AUTHORIZATION_ROOT / "launch_authorization.json"
    schedule_path = RUNTIME_ROOT / "schedule.json"
    runtime_path = RUNTIME_ROOT / "runtime.json"
    write_atomic(config_path, config)
    write_atomic(authorization_path, authorization)
    schedule = build_control_schedule(
        expected_orchestration_head=expected_orchestration_head,
        config_path=config_path,
        authorization_path=authorization_path,
        cuda_bf16_inventory=inventory,
        runtime_manifest=runtime_path,
    )
    write_atomic(schedule_path, schedule)

    worker_log = (LOG_ROOT / "scheduler.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCHEDULER), "--schedule", str(schedule_path)],
        cwd=ORCHESTRATION_WORKTREE,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    worker_log.close()
    result = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_control_recovery_launch.v1"
        ),
        "status": "XEDITCRITIC_V403_CONTROL_RECOVERY_LAUNCHED",
        "orchestration_git_head": expected_orchestration_head,
        "training_code_git_head": TRAINING_GIT_HEAD,
        "scheduler_pid": process.pid,
        "required_control_run_ids": list(CONTROL_RUN_IDS),
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_path),
        "current_full_terminal_status": full["summary"]["status"],
        "historical_c0_terminal_status": c0["status"],
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "free_memory_gate_applied": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(RUNTIME_ROOT / "launch.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-orchestration-head", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            launch(arguments.expected_orchestration_head),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
