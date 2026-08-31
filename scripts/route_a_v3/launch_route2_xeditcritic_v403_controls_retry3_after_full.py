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


HISTORICAL_FULL_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
HISTORICAL_C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
PRIOR_FAILED_CONTROL_GIT_HEAD = "a21ae2a47b3275519611ad834660813534b38c41"
RETRY3_LICENSED_HEAD = "a21ae2a47b3275519611ad834660813534b38c41"
EXPECTED_RETRY2_FIRST_FAILURE_RUN_ID = "v4_candidate_bundle_permutation"
CONTROL_RETRY_ORDINAL = 3
CONTROL_RETRY_IDENTITY = "v403_control_recovery_retry3"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
ORCHESTRATION_WORKTREE = Path(__file__).resolve().parents[2]
TRAINING_WORKTREE = ORCHESTRATION_WORKTREE
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BASE_CONFIG = (
    TRAINING_WORKTREE
    / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
)
FULL_TERMINAL_AUDIT = (
    ORCHESTRATION_WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json"
)
TRAINER = (
    TRAINING_WORKTREE
    / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
)
SCHEDULER = (
    ORCHESTRATION_WORKTREE
    / "scripts/route_a_v3/"
    "run_route2_xeditcritic_v403_control_recovery_retry3_scheduler.py"
)
PREFLIGHT_RUNNER_GIT_HEAD = "107fa43d9990e4f72f989ca0cf417260bfb10de8"
CURRENT_FULL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_rng_replay_fix_{HISTORICAL_FULL_GIT_HEAD}"
)
CURRENT_FULL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_rng_replay_fix_runner_{HISTORICAL_FULL_GIT_HEAD}/runtime.json"
)
PRIOR_FAILED_CONTROL_RUNTIME = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_control_recovery_retry2_runner_{PRIOR_FAILED_CONTROL_GIT_HEAD}/runtime.json"
)
PRIOR_CONTROL_OOM_TERMINAL_RECEIPT = (
    ROOT
    / "audits/xeditcritic_v4/"
    f"v403_control_recovery_retry2_runner_{PRIOR_FAILED_CONTROL_GIT_HEAD}_terminal.json"
)
PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_control_recovery_"
    "retry2_terminal_receipt.v1"
)
PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_STATUS = (
    "XEDITCRITIC_V403_CONTROL_RECOVERY_RETRY2_TERMINAL_RECORDED"
)
HISTORICAL_C0_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v402_recovery_runner_{HISTORICAL_C0_GIT_HEAD}"
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
PHYSICAL_GPU_INDICES = (2, 3, 5, 2, 3, 5)
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
RUNNER_VERIFICATION_RECEIPT_SCHEMA = (
    "route_a_v3_route2_xedit_v403_successor_runner_verification_receipt.v1"
)
RUNNER_VERIFICATION_RECEIPT_PASS = (
    "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_PASS"
)
FOCUSED_PROCESS_GROUP_COUNT = 8
MIN_FOCUSED_TESTS = 203
FOCUSED_GROUP_REQUIRED_TEST_MARKERS = (
    (
        "test_score_route2_xeditflow_closed_frozen_methods_v3.py",
        "test_launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen.py",
    ),
    (
        "test_transition_adjudicate_route2_xeditcritic_v403_cross_root_screen.py",
        "test_route2_xeditcritic_v4_confirmation_runtime.py",
    ),
    (
        "test_route2_xeditsetflow_s1.py",
        "test_launch_route2_xeditsetflow_s1_confirmation_after_screen_pass.py",
    ),
    (
        "test_run_route2_xeditflow_v4_guidance_screen_scheduler.py",
        "test_adjudicate_route2_xeditflow_guidance_screen_v4.py",
    ),
    (
        "test_train_route2_xeditcritic_v4.py",
        "test_run_route2_xeditcritic_v4_loso_scheduler.py",
    ),
    (
        "test_run_route2_xedit_v4_confirmation_training_scheduler.py",
        "test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
    ),
    (
        "test_launch_route2_xeditcritic_v403_controls_after_full.py",
        "test_launch_route2_xeditcritic_v4_refit_after_atomic_test.py",
        "test_launch_route2_xeditcritic_v4_loso_after_refits.py",
    ),
    (
        "test_xeditflow_v4_final_evidence_chain.py",
        "test_export_route2_xeditflow_v4_terminal_training_ledger.py",
    ),
)
V332_TEST_GLOB_MARKER = "*v332*.py"


def control_family_paths(current_head: str) -> dict[str, Path]:
    """Return the independent retry2 roots bound to the new licensed HEAD."""

    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None
        and current_head == RETRY3_LICENSED_HEAD
        and current_head
        not in {
            HISTORICAL_FULL_GIT_HEAD,
            HISTORICAL_C0_GIT_HEAD,
        },
        "control retry runner HEAD must be the pinned Retry2 baseline",
    )
    return {
        "output_root": (
            ROOT
            / "experiments/xeditcritic_v4/"
            f"screen_seed_20260907_{CONTROL_RETRY_IDENTITY}_{current_head}"
        ),
        "runtime_root": (
            ROOT
            / "experiments/xeditcritic_v4/"
            f"{CONTROL_RETRY_IDENTITY}_runner_{current_head}"
        ),
        "authorization_root": (
            ROOT
            / "authorizations/xeditcritic_v4/"
            f"{CONTROL_RETRY_IDENTITY}_{current_head}"
        ),
        "log_root": (
            ROOT
            / "logs/xeditcritic_v4/"
            f"{CONTROL_RETRY_IDENTITY}_{current_head}"
        ),
        "transition_gate": (
            ROOT
            / "experiments/xeditcritic_v4/"
            "screen_seed_20260907_v403_cross_root_controls_"
            f"retry3_{current_head}/"
            "screen_gate.json"
        ),
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


def validate_prior_control_oom_terminal_receipt(
    receipt: Mapping[str, Any], *, receipt_path: Path
) -> None:
    """Accept only the canonical closed old family as retry2 provenance."""

    require(
        receipt_path == PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
        "prior Critic controls OOM terminal receipt path is not canonical",
    )
    first_failure = receipt.get("first_terminal_failure")
    require(
        receipt.get("schema_version")
        == PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_SCHEMA
        and receipt.get("status")
        == PRIOR_CONTROL_OOM_TERMINAL_RECEIPT_STATUS
        and receipt.get("terminal_class") == "TECHNICAL_FAILURE_TERMINAL"
        and receipt.get("old_runtime_path")
        == str(PRIOR_FAILED_CONTROL_RUNTIME)
        and receipt.get("old_runtime_status")
        == "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
        and receipt.get("old_current_git_head")
        == PRIOR_FAILED_CONTROL_GIT_HEAD
        and receipt.get("old_runner_git_head")
        == PRIOR_FAILED_CONTROL_GIT_HEAD
        and receipt.get("old_orchestration_git_head")
        == PRIOR_FAILED_CONTROL_GIT_HEAD
        and receipt.get("old_training_code_git_head")
        == PRIOR_FAILED_CONTROL_GIT_HEAD,
        "prior Critic controls OOM receipt identity is invalid",
    )
    technical_failure_run_ids = receipt.get("technical_failure_run_ids")
    terminal_summary_run_ids = receipt.get("terminal_summary_run_ids")
    not_run_run_ids = receipt.get("not_run_after_terminal_failure_run_ids")
    require(
        receipt.get("ordered_control_run_ids") == list(CONTROL_RUN_IDS)
        and isinstance(receipt.get("terminal_jobs"), list)
        and len(receipt["terminal_jobs"]) == len(CONTROL_RUN_IDS)
        and isinstance(technical_failure_run_ids, list)
        and bool(technical_failure_run_ids)
        and isinstance(terminal_summary_run_ids, list)
        and isinstance(not_run_run_ids, list)
        and not (
            set(technical_failure_run_ids)
            & set(terminal_summary_run_ids)
            & set(not_run_run_ids)
        )
        and set(technical_failure_run_ids)
        | set(terminal_summary_run_ids)
        | set(not_run_run_ids)
        == set(CONTROL_RUN_IDS)
        and isinstance(first_failure, Mapping)
        and first_failure.get("run_id") == EXPECTED_RETRY2_FIRST_FAILURE_RUN_ID
        and first_failure.get("run_id") in technical_failure_run_ids,
        "prior Critic controls retry2 terminal inventory is invalid",
    )
    require(
        receipt.get("scheduler_process_gone") is True
        and receipt.get("cross_root_adjudication_run") is False
        and receipt.get("cross_root_gate_absent") is True
        and receipt.get("free_memory_gate_applied") is False
        and int(receipt.get("terminal_artifact_payloads_read_by_transition", -1))
        == 0
        and int(
            receipt.get(
                "historical_terminal_payloads_read_before_cross_root", -1
            )
        )
        == 0
        and receipt.get("successor_authorized") is False
        and receipt.get("same_family_retry_authorized") is False
        and receipt.get("new_independent_retry_eligible") is True
        and receipt.get("old_family_artifacts_read_only") is True
        and int(receipt.get("old_runtime_read_count_this_transition", -1))
        == 1
        and receipt.get("gpu_inventory_or_probe_executed") is False
        and receipt.get("gpu_or_model_execution_started") is False
        and receipt.get("protected_outcome_payload_read") is False,
        "prior Critic controls OOM receipt does not authorize an independent retry",
    )
    require_zero_protected_reads(receipt, "prior Critic controls OOM receipt")


def runner_verification_receipt_path(current_head: str) -> Path:
    return (
        ROOT
        / "audits/xedit_v4/"
        f"v403_successor_runner_verification_{current_head}.json"
    )


def validate_runner_verification_receipt(
    receipt: Mapping[str, Any], *, current_head: str, receipt_path: Path
) -> None:
    require(
        receipt_path == runner_verification_receipt_path(current_head),
        "control runner verification receipt path is not canonical",
    )
    require(
        receipt.get("schema_version") == RUNNER_VERIFICATION_RECEIPT_SCHEMA
        and receipt.get("status") == RUNNER_VERIFICATION_RECEIPT_PASS
        and receipt.get("runner_git_head") == current_head
        and receipt.get("worktree_clean") is True,
        "control runner verification is not exact-HEAD clean PASS",
    )
    focused = receipt.get("focused_tests")
    require(isinstance(focused, Mapping), "runner receipt lacks focused tests")
    commands = focused.get("command")
    group_counts = focused.get("group_passed_counts")
    passed_count = focused.get("passed_count")
    failed_count = focused.get("failed_count")
    require(
        focused.get("isolated_process_groups") is True
        and isinstance(commands, list)
        and len(commands) == FOCUSED_PROCESS_GROUP_COUNT
        and all(isinstance(value, str) and value for value in commands)
        and isinstance(group_counts, list)
        and len(group_counts) == FOCUSED_PROCESS_GROUP_COUNT
        and all(type(value) is int and value > 0 for value in group_counts)
        and focused.get("passed") is True
        and type(passed_count) is int
        and passed_count >= MIN_FOCUSED_TESTS
        and sum(group_counts) == passed_count
        and type(failed_count) is int
        and failed_count == 0,
        "runner receipt focused cohort is failed or incomplete",
    )
    for group_index, required_markers in enumerate(
        FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    ):
        require(
            all(
                marker in commands[group_index]
                for marker in required_markers
            ),
            "runner receipt focused group lacks required module coverage",
        )
    v332 = receipt.get("v332_tests")
    require(isinstance(v332, Mapping), "runner receipt lacks V3.3.2 tests")
    v332_command = v332.get("command")
    require(
        isinstance(v332_command, list)
        and v332_command
        and all(isinstance(value, str) and value for value in v332_command)
        and any(V332_TEST_GLOB_MARKER in value for value in v332_command)
        and v332.get("passed") is True
        and type(v332.get("passed_count")) is int
        and v332.get("passed_count") == 96
        and type(v332.get("failed_count")) is int
        and v332.get("failed_count") == 0,
        "runner receipt V3.3.2 cohort is failed or incomplete",
    )
    require_zero_protected_reads(receipt, "control runner verification receipt")


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_scheduler_launch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    created_artifacts: Mapping[str, Path],
    error: Exception,
) -> None:
    """Persist the one-shot family when its scheduler process cannot start."""

    require(
        not path.exists()
        and not path.with_suffix(path.suffix + ".partial").exists(),
        "Critic control scheduler launch failure evidence already exists; "
        "use a new retry family",
    )
    write_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_control_"
                "scheduler_launch_failure.v1"
            ),
            "status": (
                "XEDITCRITIC_V403_CONTROL_SCHEDULER_LAUNCH_TECHNICAL_FAILURE"
            ),
            "failure_stage": "SCHEDULER_PROCESS_LAUNCH",
            "expected_git_head": expected_head,
            "worktree": str(ORCHESTRATION_WORKTREE),
            "scheduler_command": list(command_line),
            "schedule_path": str(schedule_path),
            "intended_runtime_manifest": str(runtime_path),
            "created_artifact_paths": {
                key: str(value) for key, value in created_artifacts.items()
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "scheduler_started": False,
            "gpu_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def spawn_scheduler_with_failure_evidence(
    *,
    failure_path: Path,
    expected_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    worker_log_path: Path,
    created_artifacts: Mapping[str, Path],
) -> subprocess.Popen[str]:
    stream = worker_log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command_line),
            cwd=ORCHESTRATION_WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception as error:
        stream.close()
        write_scheduler_launch_failure_evidence(
            failure_path,
            expected_head=expected_head,
            command_line=command_line,
            schedule_path=schedule_path,
            runtime_path=runtime_path,
            created_artifacts=created_artifacts,
            error=error,
        )
        raise XEditCriticV403ControlRecoveryLaunchError(
            "Critic control scheduler process could not start; durable "
            f"technical failure evidence: {failure_path}"
        ) from error
    stream.close()
    return process


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
            "training_git_head": expected_head,
            "historical_full_git_head": HISTORICAL_FULL_GIT_HEAD,
            "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
            **details,
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_runtime_root": str(runtime_root),
            "runtime_root_created": False,
            "jobs_started": 0,
            "cpu_fallback_used": cpu_fallback_used,
            "free_memory_gate_applied": False,
            "historical_terminal_payloads_read_before_cross_root": 0,
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


def validate_historical_full_terminal_audit(
    path: Path = FULL_TERMINAL_AUDIT,
) -> dict[str, Any]:
    audit = read_json(path)
    facts = audit.get("terminal_facts")
    decision = audit.get("successor_decision")
    boundary = audit.get("claim_boundary")
    require(
        audit.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_terminal.v1"
        and audit.get("status")
        == "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"
        and audit.get("evidence_scope")
        == "TERMINAL_FACTS_ALREADY_CONSUMED_BY_THE_LOW_FREQUENCY_HEARTBEAT_ONLY"
        and audit.get("runtime_path") == str(CURRENT_FULL_RUNTIME)
        and audit.get("output_root")
        == str(CURRENT_FULL_OUTPUT_ROOT / "v4_full")
        and audit.get("terminal_summary_path")
        == str(CURRENT_FULL_OUTPUT_ROOT / "v4_full/run_summary.json"),
        "tracked historical full terminal audit identity is invalid",
    )
    require(
        isinstance(facts, Mapping)
        and facts.get("runtime_status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and facts.get("terminal_artifact_kind") == "SUMMARY"
        and facts.get("run_id") == "v4_full"
        and facts.get("seed") == FROZEN_FULL_SUMMARY_IDENTITY["seed"]
        and facts.get("completed_passes")
        == FROZEN_FULL_SUMMARY_IDENTITY["pass_count"]
        and facts.get("selected_pass")
        == FROZEN_FULL_SUMMARY_IDENTITY["selected_pass"]
        and facts.get("optimizer_update_count")
        == FROZEN_FULL_SUMMARY_IDENTITY["update_count"]
        and facts.get("physical_batch_size")
        == FROZEN_FULL_SUMMARY_IDENTITY["physical_batch_size"]
        and facts.get("effective_batch_size")
        == FROZEN_FULL_SUMMARY_IDENTITY["effective_batch_size"]
        and facts.get("training_precision")
        == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and type(facts.get("physical_gpu_index")) is int
        and facts.get("physical_gpu_index") == 5
        and facts.get("device_class") == "A100"
        and facts.get("cuda_used") is True
        and facts.get("cpu_fallback_used") is False
        and facts.get("authorization_git_head") == HISTORICAL_FULL_GIT_HEAD
        and int(facts.get("development_test_outcome_reads", -1)) == 0
        and int(facts.get("new_final_evaluation_outcome_reads", -1)) == 0
        and int(facts.get("protected_outcome_reads", -1)) == 0,
        "tracked historical full terminal facts are invalid",
    )
    require(
        isinstance(decision, Mapping)
        and decision.get("controls_started") is False
        and decision.get("controls_status") == "PAUSED_NOT_LAUNCHED"
        and decision.get("reason")
        == "CRITIC_CONTROLS_CANNOT_RESTORE_THE_ALREADY_FAILED_SETFLOW_DUAL_READINESS"
        and decision.get("confirmation_or_posttest_authorized") is False
        and decision.get("critic_terminal_summary_reinterpreted_as_final_science")
        is False
        and isinstance(boundary, Mapping)
        and boundary.get("single_arm_terminal_summary_is_not_a_screen_pass")
        is True
        and boundary.get("single_arm_terminal_summary_is_not_final_scientific_evidence")
        is True
        and boundary.get("model_advantage_established") is False
        and boundary.get("submission_ready") is False,
        "tracked historical full claim boundary is invalid",
    )
    return audit


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
        and runtime.get("git_head") == HISTORICAL_FULL_GIT_HEAD,
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
        and authorization.get("authorized_git_head") == HISTORICAL_FULL_GIT_HEAD
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
        re.fullmatch(r"[0-9a-f]{40}", expected_orchestration_head) is not None
        and expected_orchestration_head == RETRY3_LICENSED_HEAD
        and expected_orchestration_head
        not in {
            HISTORICAL_FULL_GIT_HEAD,
            HISTORICAL_C0_GIT_HEAD,
        },
        "expected control retry runner HEAD is not the pinned Retry2 baseline",
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
    require(
        training_head == expected_orchestration_head,
        "current control training worktree HEAD changed",
    )
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
    receipt_path = runner_verification_receipt_path(
        expected_orchestration_head
    )
    receipt = read_json(receipt_path)
    validate_runner_verification_receipt(
        receipt,
        current_head=expected_orchestration_head,
        receipt_path=receipt_path,
    )
    return {
        "preflight": {"git_head": PREFLIGHT_RUNNER_GIT_HEAD},
        "runner_verification_receipt": receipt,
        "runner_verification_receipt_path": receipt_path,
    }


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
    *,
    current_head: str,
    output_root: Path,
    screen_gate_output: Path,
) -> dict[str, Any]:
    run_ids = tuple(str(row["run_id"]) for row in base["required_screen_runs"])
    require(run_ids == ALL_RUN_IDS, "frozen Critic eight-arm order changed")
    recovery = copy.deepcopy(dict(base))
    recovery["runner_git_head"] = current_head
    recovery["output_root"] = str(output_root)
    recovery["screen_gate_output"] = str(screen_gate_output)
    allowed = {"runner_git_head", "output_root", "screen_gate_output"}
    require(
        {key: value for key, value in base.items() if key not in allowed}
        == {key: value for key, value in recovery.items() if key not in allowed},
        "control recovery changed a scientific config field",
    )
    return recovery


def build_launch_authorization(
    preflight: Mapping[str, Any],
    *,
    current_head: str,
    prior_control_oom_terminal_receipt: Mapping[str, Any],
    prior_control_oom_terminal_receipt_path: Path,
    historical_full_terminal_audit: Mapping[str, Any],
    historical_full_terminal_audit_path: Path,
    runner_verification_receipt: Mapping[str, Any],
    runner_verification_receipt_path: Path,
) -> dict[str, Any]:
    validate_prior_control_oom_terminal_receipt(
        prior_control_oom_terminal_receipt,
        receipt_path=prior_control_oom_terminal_receipt_path,
    )
    validate_runner_verification_receipt(
        runner_verification_receipt,
        current_head=current_head,
        receipt_path=runner_verification_receipt_path,
    )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_head,
        "preflight_runner_git_head": str(preflight["git_head"]),
        "historical_full_terminal_audit": {
            **dict(historical_full_terminal_audit),
            "path": str(historical_full_terminal_audit_path),
        },
        "runner_verification_receipt": {
            **dict(runner_verification_receipt),
            "path": str(runner_verification_receipt_path),
        },
        "prior_control_oom_terminal_receipt": {
            **dict(prior_control_oom_terminal_receipt),
            "path": str(prior_control_oom_terminal_receipt_path),
        },
        "authorized_run_ids": list(ALL_RUN_IDS),
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": (
                runner_verification_receipt["focused_tests"]["passed"]
            ),
            "a100_current_head_v332_tests_passed": (
                runner_verification_receipt["v332_tests"]["passed"]
            ),
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cache_online_equivalence_passed": True,
            "prior_oom_family_exact_technical_terminal": True,
            "same_family_retry_forbidden": True,
            "new_independent_retry_eligible": True,
        },
        "v403_control_recovery": {
            "historical_full_git_head": HISTORICAL_FULL_GIT_HEAD,
            "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
            "current_git_head": current_head,
            "runner_git_head": current_head,
            "training_code_git_head": current_head,
            "training_worktree": str(TRAINING_WORKTREE),
            "orchestration_git_head": current_head,
            "required_control_run_ids": list(CONTROL_RUN_IDS),
            "historical_repaired_full_exact_terminal_summary": True,
            "historical_c0_exact_terminal_summary_reused": True,
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "scientific_config_changed": False,
            "retry_ordinal": CONTROL_RETRY_ORDINAL,
            "retry_identity": CONTROL_RETRY_IDENTITY,
            "prior_failed_control_git_head": PRIOR_FAILED_CONTROL_GIT_HEAD,
            "prior_failed_control_runtime": str(
                PRIOR_FAILED_CONTROL_RUNTIME
            ),
            "prior_family_reused": False,
            "all_six_controls_retrained": True,
            "wave_order": [
                list(CONTROL_RUN_IDS[:3]),
                list(CONTROL_RUN_IDS[3:]),
            ],
            "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
            "free_memory_gate_applied": False,
            "historical_terminal_payloads_read_before_cross_root": 0,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def build_control_schedule(
    *,
    current_head: str,
    config_path: Path,
    authorization_path: Path,
    cuda_bf16_inventory: Sequence[Mapping[str, Any]],
    output_root: Path,
    runtime_manifest: Path,
    log_root: Path,
    transition_gate: Path,
    prior_control_oom_terminal_receipt_path: Path,
    historical_full_terminal_audit_path: Path = FULL_TERMINAL_AUDIT,
) -> dict[str, Any]:
    gpu_indices = [
        int(row["physical_gpu_index"]) for row in cuda_bf16_inventory
    ]
    require(
        gpu_indices == list(PHYSICAL_GPU_INDICES),
        "CUDA/BF16 inventory does not cover physical GPUs 0-5",
    )
    jobs = []
    for job_index, (run_id, physical_gpu_index) in enumerate(zip(
        CONTROL_RUN_IDS, PHYSICAL_GPU_INDICES, strict=True
    )):
        output_directory = output_root / run_id
        attempt_id = (
            "xeditcritic_v4_screen_seed20260907::"
            f"{run_id}::{CONTROL_RETRY_IDENTITY}_{current_head}"
        )
        jobs.append(
            {
                "run_id": run_id,
                "physical_gpu_index": physical_gpu_index,
                "output_directory": str(output_directory),
                "log_path": str(log_root / f"{run_id}.log"),
                "training_attempt_id": attempt_id,
                "training_git_head": current_head,
                "wave_index": job_index // 3,
                "process_environment": {
                    "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF,
                },
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
        "historical_full_git_head": HISTORICAL_FULL_GIT_HEAD,
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "current_git_head": current_head,
        "runner_git_head": current_head,
        "orchestration_git_head": current_head,
        "training_code_git_head": current_head,
        "training_worktree": str(TRAINING_WORKTREE),
        "retry_ordinal": CONTROL_RETRY_ORDINAL,
        "retry_identity": CONTROL_RETRY_IDENTITY,
        "prior_failed_control_git_head": PRIOR_FAILED_CONTROL_GIT_HEAD,
        "prior_failed_control_runtime": str(PRIOR_FAILED_CONTROL_RUNTIME),
        "prior_control_oom_terminal_receipt": str(
            prior_control_oom_terminal_receipt_path
        ),
        "control_waves": [
            list(CONTROL_RUN_IDS[:3]),
            list(CONTROL_RUN_IDS[3:]),
        ],
        "wave1_requires_wave0_all_summaries": True,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "runtime_manifest": str(runtime_manifest),
        "screen_config": str(config_path),
        "launch_authorization": str(authorization_path),
        "current_full_summary": str(
            CURRENT_FULL_OUTPUT_ROOT / "v4_full/run_summary.json"
        ),
        "historical_c0_summary": str(
            HISTORICAL_C0_OUTPUT_ROOT / "c0_v4/run_summary.json"
        ),
        "historical_full_terminal_audit": str(
            historical_full_terminal_audit_path
        ),
        "cross_root_gate": str(transition_gate),
        "cuda_bf16_inventory": [dict(row) for row in cuda_bf16_inventory],
        "jobs": jobs,
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "prior_family_reused": False,
        "all_six_controls_retrained": True,
        "free_memory_gate_applied": False,
        "terminal_artifact_payloads_read_by_scheduler": 0,
        "historical_terminal_payloads_read_before_cross_root": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def launch(
    expected_orchestration_head: str,
    *,
    prior_terminal_receipt_path: Path = PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
) -> dict[str, Any]:
    # The canonical closure of the failed package is the first old-family
    # admission evidence.  No historical producer audit, current-HEAD check,
    # GPU probe, or family path is consumed before it passes.
    require(
        expected_orchestration_head == RETRY3_LICENSED_HEAD,
        "retry3 control retry HEAD is not the pinned Retry2 baseline a21ae2a4",
    )
    require(
        prior_terminal_receipt_path == PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
        "prior Critic controls OOM terminal receipt path is not canonical",
    )
    prior_terminal_receipt = read_json(prior_terminal_receipt_path)
    validate_prior_control_oom_terminal_receipt(
        prior_terminal_receipt,
        receipt_path=prior_terminal_receipt_path,
    )
    # Consume only the tracked heartbeat audit. Historical runtime and summary
    # payloads remain closed until the cross-root transition.
    full_terminal_audit = validate_historical_full_terminal_audit()
    paths = control_family_paths(expected_orchestration_head)
    output_root = paths["output_root"]
    runtime_root = paths["runtime_root"]
    authorization_root = paths["authorization_root"]
    log_root = paths["log_root"]
    transition_gate = paths["transition_gate"]
    for path, label in (
        (output_root, "control output root"),
        (runtime_root, "control runtime root"),
        (authorization_root, "control authorization root"),
        (transition_gate.parent, "cross-root gate root"),
    ):
        require(not path.exists(), f"{label} already exists")
    prelaunch_failure_path = require_fresh_prelaunch_family(runtime_root)
    source = validate_training_source(expected_orchestration_head)
    try:
        physical_gpu_inventory()
    except XEditCriticV403GpuInventoryError as error:
        write_prelaunch_failure_evidence(
            prelaunch_failure_path,
            expected_head=expected_orchestration_head,
            failure_stage="INVENTORY",
            runtime_root=runtime_root,
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
            runtime_root=runtime_root,
            command_line=probe_command,
            error=error,
        )
        raise
    config = build_recovery_config(
        read_json(BASE_CONFIG),
        current_head=expected_orchestration_head,
        output_root=output_root,
        screen_gate_output=transition_gate,
    )
    authorization = build_launch_authorization(
        source["preflight"],
        current_head=expected_orchestration_head,
        prior_control_oom_terminal_receipt=prior_terminal_receipt,
        prior_control_oom_terminal_receipt_path=prior_terminal_receipt_path,
        historical_full_terminal_audit=full_terminal_audit,
        historical_full_terminal_audit_path=FULL_TERMINAL_AUDIT,
        runner_verification_receipt=source["runner_verification_receipt"],
        runner_verification_receipt_path=source[
            "runner_verification_receipt_path"
        ],
    )

    authorization_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    config_path = authorization_root / "screen_config.json"
    authorization_path = authorization_root / "launch_authorization.json"
    schedule_path = runtime_root / "schedule.json"
    runtime_path = runtime_root / "runtime.json"
    write_atomic(config_path, config)
    write_atomic(authorization_path, authorization)
    schedule = build_control_schedule(
        current_head=expected_orchestration_head,
        config_path=config_path,
        authorization_path=authorization_path,
        cuda_bf16_inventory=inventory,
        output_root=output_root,
        runtime_manifest=runtime_path,
        log_root=log_root,
        transition_gate=transition_gate,
        prior_control_oom_terminal_receipt_path=prior_terminal_receipt_path,
        historical_full_terminal_audit_path=FULL_TERMINAL_AUDIT,
    )
    write_atomic(schedule_path, schedule)

    scheduler_command = [
        str(PYTHON),
        str(SCHEDULER),
        "--schedule",
        str(schedule_path),
    ]
    scheduler_log = log_root / "scheduler.log"
    process = spawn_scheduler_with_failure_evidence(
        failure_path=runtime_root / "scheduler_launch.failed.json",
        expected_head=expected_orchestration_head,
        command_line=scheduler_command,
        schedule_path=schedule_path,
        runtime_path=runtime_path,
        worker_log_path=scheduler_log,
        created_artifacts={
            "screen_config": config_path,
            "launch_authorization": authorization_path,
            "schedule": schedule_path,
            "scheduler_log": scheduler_log,
        },
    )
    result = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_control_recovery_launch.v1"
        ),
        "status": "XEDITCRITIC_V403_CONTROL_RECOVERY_LAUNCHED",
        "historical_full_git_head": HISTORICAL_FULL_GIT_HEAD,
        "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
        "current_git_head": expected_orchestration_head,
        "runner_git_head": expected_orchestration_head,
        "orchestration_git_head": expected_orchestration_head,
        "training_code_git_head": expected_orchestration_head,
        "retry_ordinal": CONTROL_RETRY_ORDINAL,
        "retry_identity": CONTROL_RETRY_IDENTITY,
        "prior_failed_control_git_head": PRIOR_FAILED_CONTROL_GIT_HEAD,
        "prior_failed_control_runtime": str(PRIOR_FAILED_CONTROL_RUNTIME),
        "prior_control_oom_terminal_receipt": str(
            prior_terminal_receipt_path
        ),
        "prior_control_oom_terminal_receipt_status": (
            prior_terminal_receipt["status"]
        ),
        "prior_family_reused": False,
        "all_six_controls_retrained": True,
        "control_waves": [
            list(CONTROL_RUN_IDS[:3]),
            list(CONTROL_RUN_IDS[3:]),
        ],
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "scheduler_pid": process.pid,
        "required_control_run_ids": list(CONTROL_RUN_IDS),
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_path),
        "historical_full_terminal_audit_path": str(FULL_TERMINAL_AUDIT),
        "historical_full_terminal_audit_status": full_terminal_audit["status"],
        "historical_c0_terminal_payload_read": False,
        "historical_terminal_payloads_read_before_cross_root": 0,
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "free_memory_gate_applied": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-orchestration-head", required=True)
    parser.add_argument(
        "--prior-terminal-receipt",
        type=Path,
        default=PRIOR_CONTROL_OOM_TERMINAL_RECEIPT,
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            launch(
                arguments.expected_orchestration_head,
                prior_terminal_receipt_path=arguments.prior_terminal_receipt,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
