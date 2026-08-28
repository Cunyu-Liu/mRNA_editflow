#!/usr/bin/env python3
"""Launch the isolated SetFlow V4 S1 screen from tracked terminal facts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    require_runner_verification_receipt_v403,
)
from scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen import (
    validate_runner_verification_receipt as validate_shared_runner_verification_receipt,
)
from scripts.route_a_v3.transition_record_route2_xeditsetflow_s1_930_terminal_invalidation import (
    CANONICAL_RECEIPT as OLD_S1_TERMINAL_INVALIDATION_RECEIPT,
    DEFECT_IDENTITY as OLD_S1_DEFECT_IDENTITY,
    OLD_HEAD as INVALIDATED_S1_RUNNER_HEAD,
    OLD_RUNTIME as INVALIDATED_S1_RUNTIME,
    RECEIPT_SCHEMA as OLD_S1_TERMINAL_INVALIDATION_SCHEMA,
    RUN_IDS as INVALIDATED_S1_RUN_IDS,
    SCIENTIFIC_INVALIDATION_STATUS as OLD_S1_SCIENTIFIC_INVALIDATION_STATUS,
    SCIENTIFIC_TERMINAL_STATUS as OLD_S1_SCIENTIFIC_TERMINAL_STATUS,
    SCREEN_SEED as INVALIDATED_S1_SCREEN_SEED,
    TECHNICAL_INVALIDATION_STATUS as OLD_S1_TECHNICAL_INVALIDATION_STATUS,
    TECHNICAL_TERMINAL_STATUS as OLD_S1_TECHNICAL_TERMINAL_STATUS,
)


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BRANCH = "route-a-v3-v403-no-vram-gate-20260827"
CONFIG = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json"
)
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_s1.py"
VALIDATOR = (
    WORKTREE / "scripts/route_a_v3/validate_route2_xeditsetflow_s1_checkpoint.py"
)
ADJUDICATOR = (
    WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_s1_screen.py"
)
SCHEDULER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xeditsetflow_s1_screen_scheduler.py"
)
RUN_IDS = ("v4_s1_full", "v4_s1_single_mode")
CHECKPOINT_PASSES = (4, 6, 8, 10)
OBJECTIVE_IDENTITY = "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY"
OBJECTIVE_WEIGHT = 0.05
CONFIG_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_config.v1"
)
GPU_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.free,memory.total",
    "--format=csv,noheader,nounits",
)


class XEditSetFlowS1LaunchError(RuntimeError):
    pass


class XEditSetFlowS1GpuError(XEditSetFlowS1LaunchError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        return_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        missing_physical_gpus: Sequence[int] = (),
        failed_physical_gpu_index: int | None = None,
        probe_command: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.missing_physical_gpus = tuple(int(v) for v in missing_physical_gpus)
        self.failed_physical_gpu_index = failed_physical_gpu_index
        self.probe_command = tuple(str(value) for value in probe_command)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowS1LaunchError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def _protected_reads_zero(payload: Mapping[str, Any], label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a protected outcome read",
    )


def validate_config(config: Mapping[str, Any]) -> None:
    require(
        config.get("schema_version") == CONFIG_SCHEMA,
        "SetFlow V4 S1 config schema changed",
    )
    require(
        int(config.get("training", {}).get("screen_seed", -1)) == 20260911,
        "SetFlow V4 S1 screen seed changed",
    )
    objective = config.get("objective")
    require(isinstance(objective, Mapping), "S1 objective config is absent")
    require(
        objective.get("identity") == OBJECTIVE_IDENTITY
        and float(
            objective.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT
        and objective.get("cross_state_candidate_mode_responsibility_weight_sweep")
        is False,
        "SetFlow V4 S1 objective identity, weight, or no-sweep policy changed",
    )
    expected_runs = {
        ("v4_s1_full", 8, 0.05, 0.05, True),
        ("v4_s1_single_mode", 1, 0.0, 0.05, False),
    }
    observed_runs = {
        (
            str(row.get("run_id")),
            int(row.get("mode_count", -1)),
            float(row.get("mode_information_weight", -1.0)),
            float(
                row.get(
                    "cross_state_candidate_mode_responsibility_weight", -1.0
                )
            ),
            bool(row.get("selectable")),
        )
        for row in config.get("required_screen_runs", [])
    }
    require(observed_runs == expected_runs, "SetFlow V4 S1 run roles changed")
    require(
        config.get("gpu_policy", {}).get("physical_gpu_scope")
        == [0, 1, 2, 3, 4, 5]
        and config.get("gpu_policy", {}).get("cuda_bf16_only") is True
        and config.get("gpu_policy", {}).get("cpu_fallback") is False,
        "SetFlow V4 S1 GPU policy changed",
    )
    gate = config.get("screen_gate")
    require(
        isinstance(gate, Mapping)
        and gate.get("success_status") == "XEDITSETFLOW_V4_S1_SCREEN_PASS"
        and gate.get("failure_status") == "XEDITSETFLOW_V4_S1_SCREEN_NO_GO",
        "SetFlow V4 S1 gate status identity changed",
    )
    family_paths = config.get("family_paths")
    require(
        isinstance(family_paths, Mapping)
        and set(family_paths)
        == {
            "runtime_root_template",
            "schedule_template",
            "runtime_template",
            "training_output_template",
            "validation_output_template",
            "screen_gate_template",
            "log_root_template",
            "authorization_template",
            "prelaunch_failure_template",
        },
        "SetFlow V4 S1 family path inventory changed",
    )
    require(
        config.get("development_test_outcomes_accessed") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "SetFlow V4 S1 config authorizes a protected read",
    )


def _tracked_audit_path(value: Any) -> Path:
    path = WORKTREE / str(value)
    resolved = path.resolve()
    require(
        resolved.is_relative_to(WORKTREE.resolve()),
        f"repo fact audit is outside the worktree: {path}",
    )
    require(resolved.is_file(), f"repo fact audit is absent: {resolved}")
    return resolved


def validate_repo_fact_audits(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.get("repo_fact_audits")
    require(
        isinstance(paths, Mapping)
        and set(paths)
        == {
            "v403_terminal_no_go",
            "critic_v403_full_terminal",
            "s1_mechanism_authorization",
            "s1_seed_initialization_repair",
        },
        "S1 repo fact audit inventory changed",
    )
    audits = {key: read_json(_tracked_audit_path(value)) for key, value in paths.items()}

    setflow = audits["v403_terminal_no_go"]
    facts = setflow.get("terminal_facts")
    successor = setflow.get("successor_state")
    immutability = setflow.get("immutability")
    require(
        setflow.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v403_recovered_screen_terminal_nogo.v1"
        and setflow.get("status")
        == "XEDITSETFLOW_V403_RECOVERED_SCREEN_TERMINAL_NO_GO_RECORDED"
        and isinstance(facts, Mapping)
        and facts.get("runtime_status")
        == "XEDITSETFLOW_V403_VALIDATION_RECOVERY_AND_GATE_TERMINAL"
        and int(facts.get("validation_job_count", -1)) == 8
        and int(facts.get("unique_summary_count", -1)) == 8
        and int(facts.get("failure_count", -1)) == 0
        and int(facts.get("double_terminal_count", -1)) == 0
        and facts.get("gate_status") == "XEDITSETFLOW_V4_SCREEN_NO_GO"
        and facts.get("confirmation_authorized") is False
        and isinstance(successor, Mapping)
        and successor.get("legacy_successor_schedule_or_runtime_exists") is False
        and isinstance(immutability, Mapping)
        and immutability.get("runtime_and_gate_are_read_only") is True,
        "tracked SetFlow V4.0.3 terminal NO-GO facts changed",
    )
    _protected_reads_zero(facts, "tracked SetFlow V4.0.3 terminal facts")
    require(
        int(facts.get("protected_outcome_reads", -1)) == 0,
        "tracked SetFlow V4.0.3 protected read count changed",
    )

    critic = audits["critic_v403_full_terminal"]
    facts = critic.get("terminal_facts")
    successor = critic.get("successor_decision")
    require(
        critic.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_terminal.v1"
        and critic.get("status")
        == "XEDITCRITIC_V403_FULL_TERMINAL_SUMMARY_RECORDED"
        and isinstance(facts, Mapping)
        and facts.get("runtime_status")
        == "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
        and facts.get("terminal_artifact_kind") == "SUMMARY"
        and facts.get("run_id") == "v4_full"
        and int(facts.get("seed", -1)) == 20260907
        and int(facts.get("completed_passes", -1)) == 8
        and int(facts.get("selected_pass", -1)) == 8
        and int(facts.get("optimizer_update_count", -1)) == 22416
        and int(facts.get("physical_batch_size", -1)) == 32
        and int(facts.get("effective_batch_size", -1)) == 32
        and facts.get("training_precision")
        == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and int(facts.get("physical_gpu_index", -1)) == 5
        and facts.get("device_class") == "A100"
        and facts.get("cuda_used") is True
        and facts.get("cpu_fallback_used") is False
        and facts.get("authorization_git_head")
        == "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
        and isinstance(successor, Mapping)
        and successor.get("controls_started") is False,
        "tracked Critic V4.0.3 terminal facts changed",
    )
    _protected_reads_zero(facts, "tracked Critic V4.0.3 terminal facts")
    require(
        int(facts.get("protected_outcome_reads", -1)) == 0,
        "tracked Critic V4.0.3 protected read count changed",
    )

    authority = audits["s1_mechanism_authorization"]
    authority_row = authority.get("authority")
    family = authority.get("new_family")
    mechanism = authority.get("mechanism_delta")
    execution = authority.get("execution_state")
    require(
        authority.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_freeze_and_runner.v1"
        and authority.get("status")
        == "XEDITSETFLOW_V4_S1_PROTOCOL_AND_RUNNER_FROZEN_NO_ATTEMPT"
        and isinstance(authority_row, Mapping)
        and authority_row.get("s1_config")
        == "configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json"
        and authority_row.get("subordinate_amendment") is True
        and authority_row.get("legacy_v403_gate_overwritten_or_reinterpreted")
        is False
        and isinstance(family, Mapping)
        and int(family.get("screen_seed", -1)) == 20260911
        and family.get("run_ids") == list(RUN_IDS)
        and family.get("independent_family") is True
        and family.get("legacy_v403_retry") is False
        and family.get("additional_screen_seed") is False
        and isinstance(mechanism, Mapping)
        and family.get("identity") == OBJECTIVE_IDENTITY
        and float(mechanism.get("weight", -1.0)) == OBJECTIVE_WEIGHT
        and mechanism.get("weight_sweep_authorized") is False
        and isinstance(execution, Mapping)
        and execution.get("optimizer_attempt_started") is False
        and execution.get("gpu_validation_started") is False,
        "tracked SetFlow V4 S1 frozen authority changed",
    )
    _protected_reads_zero(execution, "tracked S1 execution state")

    repair = audits["s1_seed_initialization_repair"]
    affected = repair.get("affected_family")
    defect = repair.get("defect")
    repair_contract = repair.get("repair_contract")
    claim_boundary = repair.get("claim_boundary")
    require(
        repair.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair.v1"
        and repair.get("status")
        == "XEDITSETFLOW_V4_S1_SEED_INITIALIZATION_REPAIR_FROZEN_BEFORE_INDEPENDENT_RETRY"
        and isinstance(affected, Mapping)
        and affected.get("runner_git_head")
        == "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
        and int(affected.get("screen_seed", -1)) == 20260911
        and affected.get("run_ids") == list(RUN_IDS)
        and affected.get("launcher_consumed_once") is True
        and affected.get("runtime_or_outcome_read_by_this_static_audit") is False
        and affected.get("artifacts_immutable") is True
        and affected.get("same_family_retry_authorized") is False
        and isinstance(defect, Mapping)
        and defect.get("identity")
        == "PARAMETER_INITIALIZATION_SEED_APPLIED_AFTER_MODEL_CONSTRUCTION"
        and defect.get("model_construction_consumes_cpu_rng") is True
        and defect.get("nominal_seed_controlled_parameter_initialization") is False
        and defect.get("matched_full_single_initialization_established") is False
        and defect.get("exact_seed_reproducibility_established") is False
        and defect.get("affected_family_can_authorize_successor") is False,
        "tracked S1 seed-initialization defect facts changed",
    )
    require(
        isinstance(repair_contract, Mapping)
        and repair_contract.get("new_clean_pushed_git_head_required") is True
        and repair_contract.get("new_independent_family_required") is True
        and int(repair_contract.get("same_screen_seed", -1)) == 20260911
        and repair_contract.get("same_run_ids") == list(RUN_IDS)
        and repair_contract.get("same_objective_identity") == OBJECTIVE_IDENTITY
        and float(repair_contract.get("same_objective_weight", -1.0))
        == OBJECTIVE_WEIGHT
        and repair_contract.get("weight_sweep_authorized") is False
        and repair_contract.get("additional_screen_seed_authorized") is False
        and repair_contract.get("threshold_reduction_authorized") is False
        and repair_contract.get(
            "cpu_and_cuda_seed_before_any_model_construction_required"
        )
        is True
        and repair_contract.get("parameter_initialization_seed_evidence_required")
        is True
        and repair_contract.get(
            "complete_isolated_focused_and_v332_receipts_required"
        )
        is True
        and repair_contract.get("technical_retry_not_scientific_threshold_change")
        is True
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("affected_nominal_gate_is_scientific_successor_authority")
        is False
        and claim_boundary.get("affected_nominal_pass_or_no_go_is_rewritten") is False,
        "tracked S1 seed-initialization repair contract changed",
    )
    _protected_reads_zero(repair, "tracked S1 seed-initialization repair")
    return audits


def expected_receipt_paths(head: str) -> tuple[Path, Path]:
    return (
        ROOT / "audits/xedit_v4" / f"v403_successor_runner_verification_{head}.json",
        ROOT
        / "audits/xeditsetflow_v4"
        / f"confirmation_v403_recovered_runner_verification_{head}.json",
    )


def consume_receipts(
    head: str, shared_path: Path, setflow_path: Path
) -> dict[str, Any]:
    expected_shared, expected_setflow = expected_receipt_paths(head)
    require(shared_path == expected_shared, "shared receipt path is not canonical")
    require(setflow_path == expected_setflow, "SetFlow receipt path is not canonical")
    shared = read_json(shared_path)
    setflow = read_json(setflow_path)
    validate_shared_runner_verification_receipt(
        shared, runner_head=head, receipt_path=shared_path
    )
    require_runner_verification_receipt_v403(
        setflow, current_runner_head=head
    )
    return {
        "shared": {"path": str(shared_path), "status": shared["status"]},
        "setflow": {"path": str(setflow_path), "status": setflow["status"]},
    }


def consume_old_s1_terminal_invalidation_receipt(path: Path) -> dict[str, Any]:
    require(
        path == OLD_S1_TERMINAL_INVALIDATION_RECEIPT,
        "old S1 terminal invalidation receipt path is not canonical",
    )
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"old S1 terminal invalidation receipt is partial: {partial}")
    require(path.is_file(), f"old S1 terminal invalidation receipt is absent: {path}")
    payload = read_json(path)
    status = payload.get("status")
    terminal_class = payload.get("terminal_class")
    runtime_status = payload.get("old_runtime_status")
    require(
        payload.get("schema_version") == OLD_S1_TERMINAL_INVALIDATION_SCHEMA
        and status
        in {
            OLD_S1_SCIENTIFIC_INVALIDATION_STATUS,
            OLD_S1_TECHNICAL_INVALIDATION_STATUS,
        }
        and (
            (
                status == OLD_S1_SCIENTIFIC_INVALIDATION_STATUS
                and terminal_class == "SCIENTIFIC_GATE_TERMINAL"
                and runtime_status == OLD_S1_SCIENTIFIC_TERMINAL_STATUS
            )
            or (
                status == OLD_S1_TECHNICAL_INVALIDATION_STATUS
                and terminal_class == "TECHNICAL_FAILURE_TERMINAL"
                and runtime_status == OLD_S1_TECHNICAL_TERMINAL_STATUS
            )
        ),
        "old S1 terminal invalidation class is not exact",
    )
    defect = payload.get("known_defect")
    require(
        payload.get("old_runner_git_head") == INVALIDATED_S1_RUNNER_HEAD
        and payload.get("old_runtime_path") == str(INVALIDATED_S1_RUNTIME)
        and int(payload.get("screen_seed", -1)) == INVALIDATED_S1_SCREEN_SEED
        and payload.get("run_ids") == list(INVALIDATED_S1_RUN_IDS)
        and payload.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            payload.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT
        and payload.get("scheduler_process_gone") is True
        and isinstance(payload.get("terminal_jobs"), list)
        and len(payload["terminal_jobs"]) == 10
        and isinstance(payload.get("terminal_adjudication"), Mapping)
        and isinstance(defect, Mapping)
        and defect.get("identity") == OLD_S1_DEFECT_IDENTITY
        and defect.get("model_construction_consumes_cpu_rng") is True
        and defect.get("nominal_seed_controlled_parameter_initialization") is False
        and defect.get("matched_full_single_initialization_established") is False
        and defect.get("affected_family_can_authorize_successor") is False,
        "old S1 terminal invalidation lineage or defect changed",
    )
    require(
        payload.get("nominal_terminal_retained_as_execution_evidence") is True
        and payload.get("nominal_terminal_rewritten") is False
        and payload.get("scientific_successor_authorized") is False
        and payload.get("successor_authorized") is False
        and payload.get("same_family_retry_authorized") is False
        and payload.get("old_family_artifacts_read_only") is True
        and int(payload.get("old_runtime_read_count_this_transition", -1)) == 1
        and payload.get("gpu_inventory_or_probe_executed") is False
        and payload.get("gpu_or_model_execution_started") is False
        and payload.get("protected_outcome_payload_read") is False,
        "old S1 invalidation authorization or read-only boundary changed",
    )
    _protected_reads_zero(payload, "old S1 terminal invalidation receipt")
    return {
        "path": str(path),
        "status": status,
        "terminal_class": terminal_class,
        "old_runner_git_head": INVALIDATED_S1_RUNNER_HEAD,
        "old_runtime_status": runtime_status,
        "successor_authorized": False,
        "same_family_retry_authorized": False,
    }


def gpu_diagnostics(
    required_physical_gpus: Sequence[int],
) -> dict[int, dict[str, Any]]:
    try:
        result = subprocess.run(
            list(GPU_INVENTORY_COMMAND),
            cwd=WORKTREE,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise XEditSetFlowS1GpuError(
            f"nvidia-smi could not be executed: {error}",
            reason="COMMAND_EXECUTION_FAILED",
        ) from error
    if result.returncode != 0:
        raise XEditSetFlowS1GpuError(
            f"nvidia-smi exited with return code {result.returncode}",
            reason="NONZERO_RETURN_CODE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    values: dict[int, dict[str, Any]] = {}
    try:
        for line in result.stdout.splitlines():
            index, name, free, total = (
                part.strip() for part in line.split(",", maxsplit=3)
            )
            values[int(index)] = {
                "name": name,
                "free_memory_mib": int(free),
                "total_memory_mib": int(total),
            }
    except (TypeError, ValueError) as error:
        raise XEditSetFlowS1GpuError(
            f"nvidia-smi inventory could not be parsed: {error}",
            reason="OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ) from error
    required = tuple(int(gpu) for gpu in required_physical_gpus)
    missing = tuple(sorted(set(required) - set(values)))
    if missing:
        raise XEditSetFlowS1GpuError(
            f"configured physical GPUs are absent: {list(missing)}",
            reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            missing_physical_gpus=missing,
        )
    return values


def cuda_bf16_probe(physical_gpu_index: int) -> dict[str, Any]:
    source = """
import json
import sys
import torch

index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA_UNAVAILABLE_CPU_FALLBACK_FORBIDDEN")
if index < 0 or index >= torch.cuda.device_count():
    raise RuntimeError("PHYSICAL_GPU_INDEX_UNAVAILABLE")
torch.cuda.set_device(index)
name = torch.cuda.get_device_name(index)
if "A100" not in name:
    raise RuntimeError("NON_A100_DEVICE_FORBIDDEN")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16_UNAVAILABLE_ON_SELECTED_GPU")
tensor = torch.ones((8,), device=f"cuda:{index}", dtype=torch.bfloat16)
if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
    raise RuntimeError("CUDA_BF16_PROBE_SILENT_CPU_FALLBACK")
print(json.dumps({
    "physical_gpu_index": index,
    "device_name": name,
    "device_class": "A100",
    "device_type": tensor.device.type,
    "dtype": "BF16",
    "cuda_available": True,
    "bf16_supported": True,
    "cpu_fallback_used": False,
}))
"""
    probe_command = [str(PYTHON), "-c", source, str(physical_gpu_index)]
    try:
        result = command(probe_command)
    except subprocess.CalledProcessError as error:
        raise XEditSetFlowS1GpuError(
            f"GPU {physical_gpu_index} CUDA/BF16 probe exited nonzero",
            reason="CUDA_BF16_PROBE_NONZERO_RETURN_CODE",
            return_code=error.returncode,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
            failed_physical_gpu_index=physical_gpu_index,
            probe_command=probe_command,
        ) from error
    except OSError as error:
        raise XEditSetFlowS1GpuError(
            f"GPU {physical_gpu_index} CUDA/BF16 probe could not execute: {error}",
            reason="CUDA_BF16_PROBE_COMMAND_EXECUTION_FAILED",
            failed_physical_gpu_index=physical_gpu_index,
            probe_command=probe_command,
        ) from error
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise XEditSetFlowS1GpuError(
            f"GPU {physical_gpu_index} CUDA/BF16 probe output is invalid",
            reason="CUDA_BF16_PROBE_OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            failed_physical_gpu_index=physical_gpu_index,
            probe_command=probe_command,
        ) from error
    if not isinstance(payload, Mapping):
        raise XEditSetFlowS1GpuError(
            f"GPU {physical_gpu_index} CUDA/BF16 probe payload is not an object",
            reason="CUDA_BF16_PROBE_OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            failed_physical_gpu_index=physical_gpu_index,
            probe_command=probe_command,
        )
    valid = (
        payload.get("physical_gpu_index") == physical_gpu_index
        and payload.get("device_class") == "A100"
        and payload.get("device_type") == "cuda"
        and payload.get("dtype") == "BF16"
        and payload.get("cuda_available") is True
        and payload.get("bf16_supported") is True
        and payload.get("cpu_fallback_used") is False
    )
    if not valid:
        raise XEditSetFlowS1GpuError(
            f"GPU {physical_gpu_index} failed the A100 CUDA/BF16 probe",
            reason="CUDA_BF16_PROBE_IDENTITY_MISMATCH",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            failed_physical_gpu_index=physical_gpu_index,
            probe_command=probe_command,
        )
    return payload


def _formatted_path(config: Mapping[str, Any], key: str, head: str, **values: Any) -> Path:
    template = str(config["family_paths"][key])
    return Path(template.format(runner_git_head=head, **values))


def write_prelaunch_failure(
    path: Path,
    *,
    head: str,
    family_root: Path,
    error: Exception,
    diagnostics: Mapping[int, Mapping[str, Any]],
    completed_probes: Mapping[int, Mapping[str, Any]],
) -> None:
    details: dict[str, Any] = {}
    if isinstance(error, XEditSetFlowS1GpuError):
        details = {
            "gpu_prelaunch_failure_reason": error.reason,
            "return_code": error.return_code,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "missing_physical_gpus": list(error.missing_physical_gpus),
            "failed_physical_gpu_index": error.failed_physical_gpu_index,
            "probe_command": list(error.probe_command),
        }
    if isinstance(error, subprocess.CalledProcessError):
        details = {
            "return_code": error.returncode,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_prelaunch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_STOPPED_BEFORE_FAMILY_MATERIALIZATION",
            "runner_git_head": head,
            "failure_stage": (
                "A100_CUDA_BF16_PROBE"
                if isinstance(error, XEditSetFlowS1GpuError)
                and error.failed_physical_gpu_index is not None
                else "GPU0_5_INVENTORY"
            ),
            "inventory_command": list(GPU_INVENTORY_COMMAND),
            "gpu_diagnostics": {
                str(gpu): dict(row) for gpu, row in diagnostics.items()
            },
            "completed_cuda_bf16_probes": {
                str(gpu): dict(row) for gpu, row in completed_probes.items()
            },
            "error_type": type(error).__name__,
            "error": str(error),
            **details,
            "intended_family_root": str(family_root),
            "family_root_created": family_root.exists(),
            "scheduler_started": False,
            "gpu_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "parameter_update_count": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def write_scheduler_launch_failure(
    path: Path,
    *,
    head: str,
    schedule_path: Path,
    runtime_path: Path,
    scheduler_command: Sequence[str],
    error: Exception,
) -> None:
    """Retain the exact failure when the frozen scheduler process cannot start."""

    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_scheduler_launch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_SCHEDULER_LAUNCH_TECHNICAL_FAILURE",
            "runner_git_head": head,
            "failure_stage": "SCHEDULER_PROCESS_LAUNCH",
            "scheduler_command": list(scheduler_command),
            "schedule_path": str(schedule_path),
            "runtime_manifest": str(runtime_path),
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


def build_schedule(
    config: Mapping[str, Any],
    *,
    head: str,
    runtime_config_path: Path,
    authorization_path: Path,
    diagnostics: Mapping[int, Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    training_queues = []
    for gpu, run_id in zip((0, 1), RUN_IDS, strict=True):
        output = _formatted_path(
            config, "training_output_template", head, run_id=run_id
        )
        log_root = _formatted_path(config, "log_root_template", head)
        training_queues.append(
            {
                "physical_gpu_index": gpu,
                "jobs": [
                    {
                        "job_key": f"training:{run_id}",
                        "run_id": run_id,
                        "physical_gpu_index": gpu,
                        "output_directory": str(output),
                        "terminal_summary": str(output / "training_summary.json"),
                        "terminal_failure": str(output / "failure.json"),
                        "log_path": str(log_root / f"training_{run_id}.log"),
                        "command": [
                            str(PYTHON),
                            str(TRAINER),
                            "--config",
                            str(runtime_config_path),
                            "--run-id",
                            run_id,
                            "--authorization",
                            str(authorization_path),
                            "--physical-gpu-index",
                            str(gpu),
                            "--output-dir",
                            str(output),
                        ],
                    }
                ],
            }
        )
    validation_jobs: list[dict[str, Any]] = []
    for job_index, (run_id, checkpoint_pass) in enumerate(
        (pair for run_id in RUN_IDS for pair in ((run_id, p) for p in CHECKPOINT_PASSES))
    ):
        gpu = job_index % 6
        output_root = _formatted_path(
            config,
            "validation_output_template",
            head,
            run_id=run_id,
        )
        output = output_root / f"pass_{checkpoint_pass}"
        log_root = _formatted_path(config, "log_root_template", head)
        validation_jobs.append(
            {
                "job_key": f"validation:{run_id}:pass_{checkpoint_pass}",
                "run_id": run_id,
                "checkpoint_pass": checkpoint_pass,
                "physical_gpu_index": gpu,
                "terminal_summary": str(output / "validation_summary.json"),
                "terminal_failure": str(
                    output.with_name(output.name + ".failed.json")
                ),
                "log_path": str(
                    log_root / f"validation_{run_id}_pass_{checkpoint_pass}.log"
                ),
                "command": [
                    str(PYTHON),
                    str(VALIDATOR),
                    "--config",
                    str(runtime_config_path),
                    "--run-id",
                    run_id,
                    "--checkpoint-pass",
                    str(checkpoint_pass),
                    "--authorization",
                    str(authorization_path),
                    "--physical-gpu-index",
                    str(gpu),
                    "--output-dir",
                    str(output),
                ],
            }
        )
    validation_queues = [
        {
            "physical_gpu_index": gpu,
            "jobs": [job for job in validation_jobs if job["physical_gpu_index"] == gpu],
        }
        for gpu in range(6)
    ]
    gate = _formatted_path(config, "screen_gate_template", head)
    log_root = _formatted_path(config, "log_root_template", head)
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_schedule.v1",
        "status": "FROZEN_XEDITSETFLOW_V4_S1_SCREEN_SCHEDULE",
        "git_head": head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(
            _formatted_path(config, "runtime_template", head)
        ),
        "runtime_config": str(runtime_config_path),
        "authorization": str(authorization_path),
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "training_queues": training_queues,
        "validation_queues": validation_queues,
        "adjudication": {
            "gate_path": str(gate),
            "failure_path": str(gate.with_name(gate.name + ".failed.json")),
            "log_path": str(log_root / "adjudication.log"),
            "command": [
                str(PYTHON),
                str(ADJUDICATOR),
                "--config",
                str(runtime_config_path),
            ],
        },
        "gpu_diagnostics_before_launch": {
            str(gpu): dict(row) for gpu, row in diagnostics.items()
        },
        "cuda_bf16_probes": {
            str(gpu): dict(row) for gpu, row in probes.items()
        },
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(
    expected_head: str,
    *,
    old_s1_terminal_invalidation_receipt_path: Path | None = None,
    shared_receipt_path: Path | None = None,
    setflow_receipt_path: Path | None = None,
) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected Git HEAD is invalid",
    )
    require(
        expected_head != INVALIDATED_S1_RUNNER_HEAD,
        "corrected S1 retry cannot use the invalidated 930 runner HEAD",
    )
    for path, label in (
        (PYTHON, "formal Python"),
        (CONFIG, "S1 config"),
        (TRAINER, "S1 trainer"),
        (VALIDATOR, "S1 validator"),
        (ADJUDICATOR, "S1 adjudicator"),
        (SCHEDULER, "S1 scheduler"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == expected_head,
        "S1 worktree is not at the expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "S1 worktree is dirty",
    )
    require(
        command(["git", "branch", "--show-current"]).stdout.strip() == BRANCH,
        "S1 worktree is on the wrong branch",
    )
    require(
        command(["git", "rev-parse", f"origin/{BRANCH}"]).stdout.strip()
        == expected_head,
        "S1 exact HEAD has not been pushed to the authorized GitHub branch",
    )

    config = read_json(CONFIG)
    validate_config(config)
    audits = validate_repo_fact_audits(config)
    old_s1_terminal_invalidation = consume_old_s1_terminal_invalidation_receipt(
        old_s1_terminal_invalidation_receipt_path
        or OLD_S1_TERMINAL_INVALIDATION_RECEIPT
    )
    canonical_shared, canonical_setflow = expected_receipt_paths(expected_head)
    receipts = consume_receipts(
        expected_head,
        shared_receipt_path or canonical_shared,
        setflow_receipt_path or canonical_setflow,
    )

    family_root = _formatted_path(config, "runtime_root_template", expected_head)
    schedule_path = _formatted_path(config, "schedule_template", expected_head)
    runtime_path = _formatted_path(config, "runtime_template", expected_head)
    authorization_path = _formatted_path(
        config, "authorization_template", expected_head
    )
    prelaunch_failure_path = _formatted_path(
        config, "prelaunch_failure_template", expected_head
    )
    gate_path = _formatted_path(config, "screen_gate_template", expected_head)
    for path, label in (
        (family_root, "S1 family root"),
        (schedule_path, "S1 schedule"),
        (runtime_path, "S1 runtime"),
        (authorization_path, "S1 authorization"),
        (prelaunch_failure_path, "S1 prelaunch failure"),
        (gate_path, "S1 gate"),
    ):
        require(not path.exists(), f"{label} already exists: {path}")
        require(
            not path.with_suffix(path.suffix + ".partial").exists(),
            f"partial {label} already exists: {path}",
        )

    diagnostics: dict[int, dict[str, Any]] = {}
    probes: dict[int, dict[str, Any]] = {}
    try:
        diagnostics = gpu_diagnostics(range(6))
        for gpu in range(6):
            probes[gpu] = cuda_bf16_probe(gpu)
    except Exception as error:
        write_prelaunch_failure(
            prelaunch_failure_path,
            head=expected_head,
            family_root=family_root,
            error=error,
            diagnostics=diagnostics,
            completed_probes=probes,
        )
        raise

    family_root.mkdir(parents=True)
    runtime_config_path = family_root / "runtime_config.json"
    runtime_config = copy.deepcopy(config)
    runtime_config.update(
        {
            "runner_git_head": expected_head,
            "run_stage": "SCREEN",
            "output_root": str(family_root),
            "validation_output_root": str(
                family_root / "outcome_free_validation_generation"
            ),
            "screen_gate_output_path": str(gate_path),
            "s1_repo_fact_audits_consumed": {
                key: str(_tracked_audit_path(config["repo_fact_audits"][key]))
                for key in audits
            },
            "invalidated_930_s1_terminal_receipt": old_s1_terminal_invalidation,
        }
    )
    write_new_atomic(runtime_config_path, runtime_config)
    authorization = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_screen_launch_authorization.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": expected_head,
        "authorized_run_ids": list(RUN_IDS),
        "screen_seed": 20260911,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "weight_sweep_authorized": False,
        "repo_fact_audits": runtime_config["s1_repo_fact_audits_consumed"],
        "invalidated_930_s1_terminal_receipt": old_s1_terminal_invalidation,
        "runner_verification_receipts": receipts,
        "gpu_diagnostics": {
            str(gpu): dict(row) for gpu, row in diagnostics.items()
        },
        "cuda_bf16_probes": {
            str(gpu): dict(row) for gpu, row in probes.items()
        },
        "free_memory_gate_applied": False,
        "legacy_v403_confirmation_authorized": False,
        "legacy_guidance_authorized": False,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(authorization_path, authorization)
    schedule = build_schedule(
        config,
        head=expected_head,
        runtime_config_path=runtime_config_path,
        authorization_path=authorization_path,
        diagnostics=diagnostics,
        probes=probes,
    )
    require(
        Path(schedule["runtime_manifest"]) == runtime_path,
        "S1 schedule runtime path differs from frozen family template",
    )
    write_new_atomic(schedule_path, schedule)
    log_root = _formatted_path(config, "log_root_template", expected_head)
    log_root.mkdir(parents=True, exist_ok=True)
    scheduler_log = log_root / "scheduler.log"
    scheduler_command = [
        str(PYTHON),
        str(SCHEDULER),
        "--schedule",
        str(schedule_path),
    ]
    stream = scheduler_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            scheduler_command,
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as error:
        stream.close()
        write_scheduler_launch_failure(
            family_root / "scheduler_launch.failed.json",
            head=expected_head,
            schedule_path=schedule_path,
            runtime_path=runtime_path,
            scheduler_command=scheduler_command,
            error=error,
        )
        raise
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_launch.v1",
        "status": "XEDITSETFLOW_V4_S1_SCREEN_SCHEDULER_LAUNCHED",
        "runner_git_head": expected_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_path),
        "runtime_config": str(runtime_config_path),
        "authorization": str(authorization_path),
        "screen_gate": str(gate_path),
        "scheduler_log": str(scheduler_log),
        "training_job_count": 2,
        "checkpoint_validation_job_count": 8,
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "free_memory_gate_applied": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    launch_path = family_root / "launch.json"
    write_new_atomic(launch_path, launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--old-s1-terminal-invalidation-receipt", type=Path)
    parser.add_argument("--runner-verification-receipt", type=Path)
    parser.add_argument("--setflow-runner-verification-receipt", type=Path)
    arguments = parser.parse_args()
    result = run(
        arguments.expected_head,
        old_s1_terminal_invalidation_receipt_path=(
            arguments.old_s1_terminal_invalidation_receipt
        ),
        shared_receipt_path=arguments.runner_verification_receipt,
        setflow_receipt_path=arguments.setflow_runner_verification_receipt,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
