#!/usr/bin/env python3
"""Launch the exact twelve-job S1 confirmation Validation package."""

from __future__ import annotations

import argparse
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

from core.route2_xeditsetflow_confirmation_s1 import (
    CHECKPOINT_PASSES,
    CONFIRMATION_RUN_ID,
    CONFIRMATION_SEEDS,
    OBJECTIVE_IDENTITY,
    OBJECTIVE_WEIGHT,
    SCREEN_HEAD,
)
from scripts.route_a_v3.launch_route2_xeditsetflow_s1_confirmation_after_screen_pass import (
    BRANCH,
    GPU_INVENTORY_COMMAND,
    PROTOCOL,
    PYTHON,
    SCHEDULER as TRAINING_SCHEDULER,
    XEditSetFlowS1GpuError,
    _gpu_error_details,
    command,
    format_output_path,
    gpu_diagnostics,
    cuda_bf16_probe,
    protected_reads_zero,
    read_json,
    require_exact_pushed_clean_head,
    validate_authorization_s1,
    validate_manifest_s1,
    write_new_atomic,
)


VALIDATOR = (
    WORKTREE / "scripts/route_a_v3/validate_route2_xeditsetflow_s1_checkpoint.py"
)
ADJUDICATOR = (
    WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_s1_confirmation.py"
)
POSTTRAINING_SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
)


class XEditSetFlowS1ConfirmationPosttrainingLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowS1ConfirmationPosttrainingLaunchError(message)


def validate_training_schedule_s1(
    schedule: Mapping[str, Any], *, expected_head: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_training_schedule.v1"
        and schedule.get("status") == "FROZEN_S1_CONFIRMATION_TRAINING_SCHEDULE"
        and schedule.get("git_head") == expected_head
        and schedule.get("experiment_head") == SCREEN_HEAD
        and schedule.get("eligible_components") == ["setflow"]
        and schedule.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and schedule.get("selected_model") == CONFIRMATION_RUN_ID
        and int(schedule.get("training_job_count", -1)) == 3
        and int(schedule.get("single_mode_training_job_count", -1)) == 0
        and schedule.get("free_memory_gate_applied") is False
        and schedule.get("cpu_fallback_used") is False
        and schedule.get("active_performance_output_read") is False,
        "S1 confirmation training schedule identity changed",
    )
    require(
        Path(str(schedule.get("worktree"))).resolve() == WORKTREE.resolve(),
        "S1 confirmation training schedule worktree changed",
    )
    require(
        Path(str(schedule.get("confirmation_protocol"))).resolve()
        == PROTOCOL.resolve(),
        "S1 confirmation training schedule protocol changed",
    )
    protected_reads_zero(schedule, label="training schedule")
    bindings = schedule.get("posttraining_bindings")
    require(isinstance(bindings, Mapping), "S1 posttraining bindings are absent")
    require(
        bindings.get("runner_git_head") == expected_head
        and Path(str(bindings.get("protocol_path"))).resolve() == PROTOCOL.resolve()
        and bindings.get("config_manifest_path") == schedule.get("config_manifest")
        and bindings.get("confirmation_authorization_path")
        == schedule.get("confirmation_authorization")
        and bindings.get("training_runtime_path") == schedule.get("runtime_manifest"),
        "S1 posttraining bindings differ from the training schedule",
    )
    jobs: dict[int, dict[str, Any]] = {}
    queues = schedule.get("gpu_queues")
    require(isinstance(queues, list), "S1 training GPU queues are absent")
    for queue in queues:
        require(isinstance(queue, Mapping), "S1 training queue is invalid")
        gpu = int(queue.get("physical_gpu_index", -1))
        rows = queue.get("jobs")
        require(isinstance(rows, list), "S1 training job list is absent")
        for raw in rows:
            require(isinstance(raw, Mapping), "S1 training job row is invalid")
            job = dict(raw)
            seed = int(job.get("training_seed", -1))
            require(
                seed in CONFIRMATION_SEEDS
                and seed not in jobs
                and job.get("component") == "setflow"
                and job.get("run_id") == CONFIRMATION_RUN_ID
                and job.get("job_key")
                == f"setflow:{seed}:{CONFIRMATION_RUN_ID}"
                and int(job.get("physical_gpu_index", -1)) == gpu,
                "S1 confirmation training job inventory changed",
            )
            job["physical_gpu_index"] = gpu
            jobs[seed] = job
    require(
        tuple(jobs) == CONFIRMATION_SEEDS and len(jobs) == 3,
        "S1 confirmation training is not exactly three full-only jobs",
    )
    return dict(bindings), jobs


def validate_successful_training_package_s1(
    runtime: Mapping[str, Any],
    schedule_jobs: Mapping[int, Mapping[str, Any]],
    configs: Mapping[int, Path],
    *,
    expected_head: str,
) -> dict[int, dict[str, Any]]:
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xedit_v4_confirmation_training_runtime.v1"
        and runtime.get("status") == "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
        and runtime.get("git_head") == expected_head
        and runtime.get("experiment_head") == SCREEN_HEAD
        and runtime.get("eligible_components") == ["setflow"]
        and runtime.get("first_terminal_failure") is None
        and runtime.get("active_performance_output_read") is False,
        "S1 confirmation training runtime is not exact successful terminal",
    )
    protected_reads_zero(runtime, label="training runtime")
    runtime_jobs = runtime.get("jobs")
    expected_keys = {
        f"setflow:{seed}:{CONFIRMATION_RUN_ID}" for seed in CONFIRMATION_SEEDS
    }
    require(
        isinstance(runtime_jobs, Mapping) and set(runtime_jobs) == expected_keys,
        "S1 confirmation runtime job inventory changed",
    )
    packages: dict[int, dict[str, Any]] = {}
    for seed in CONFIRMATION_SEEDS:
        job = schedule_jobs[seed]
        state = runtime_jobs[f"setflow:{seed}:{CONFIRMATION_RUN_ID}"]
        require(isinstance(state, Mapping), f"S1 runtime job is invalid: {seed}")
        output = Path(str(job["output_directory"]))
        config_path = configs[seed]
        config = read_json(config_path)
        gpu = int(job["physical_gpu_index"])
        require(
            output == Path(str(config["output_root"])) / CONFIRMATION_RUN_ID
            and Path(str(job.get("config_path"))) == config_path,
            f"S1 seed {seed} training path differs from its config",
        )
        require(
            state.get("status") == "TERMINAL_COMPLETE"
            and state.get("terminal_artifact_kind") == "SUMMARY"
            and int(state.get("return_code", -1)) == 0
            and state.get("run_id") == CONFIRMATION_RUN_ID
            and int(state.get("training_seed", -1)) == seed
            and int(state.get("physical_gpu_index", -1)) == gpu
            and Path(str(state.get("output_directory"))) == output,
            f"S1 seed {seed} lacks a zero-exit terminal SUMMARY",
        )
        summary_path = output / "training_summary.json"
        failure_path = output / "failure.json"
        require(
            summary_path.is_file()
            and not failure_path.exists()
            and not summary_path.with_suffix(".json.partial").exists()
            and not failure_path.with_suffix(".json.partial").exists(),
            f"S1 seed {seed} training is not uniquely summary-terminal",
        )
        summary = read_json(summary_path)
        require(
            summary.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_s1_training_summary.v1"
            and summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION"
            and summary.get("run_stage") == "CONFIRMATION"
            and summary.get("run_id") == CONFIRMATION_RUN_ID
            and summary.get("selected_model") == CONFIRMATION_RUN_ID
            and int(summary.get("seed", -1)) == seed
            and summary.get("objective_identity") == OBJECTIVE_IDENTITY
            and float(
                summary.get(
                    "cross_state_candidate_mode_responsibility_weight", -1.0
                )
            )
            == OBJECTIVE_WEIGHT
            and int(summary.get("active_responsibility_constraint_count", 0)) > 0
            and int(summary.get("completed_passes", -1)) == 10
            and summary.get("checkpoint_selection_status")
            == "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION"
            and summary.get("validation_generation_during_training") is False
            and summary.get("parameter_changed") is True
            and int(summary.get("optimizer_update_count", 0)) > 0,
            f"S1 seed {seed} training summary identity changed",
        )
        require(
            summary.get("physical_gpu_index") == gpu
            and summary.get("torch_device") == f"cuda:{gpu}"
            and "A100" in str(summary.get("device_name", ""))
            and summary.get("training_precision") == "BF16"
            and summary.get("cpu_fallback_used") is False,
            f"S1 seed {seed} was not trained on its bound A100 CUDA/BF16 GPU",
        )
        protected_reads_zero(summary, label=f"seed {seed} training summary")
        checkpoints = {
            checkpoint_pass: output / f"pass_{checkpoint_pass}.pt"
            for checkpoint_pass in CHECKPOINT_PASSES
        }
        require(
            summary.get("saved_checkpoint_paths")
            == {
                str(checkpoint_pass): str(path)
                for checkpoint_pass, path in checkpoints.items()
            }
            and all(path.is_file() for path in checkpoints.values()),
            f"S1 seed {seed} checkpoint package changed",
        )
        training_config_path = output / "training_config.json"
        attempt_path = output / "training_attempt.json"
        training_config = read_json(training_config_path)
        attempt = read_json(attempt_path)
        require(
            training_config.get("authorized_git_head") == expected_head
            and training_config.get("confirmation_runner_git_head") == expected_head
            and training_config.get("run_stage") == "CONFIRMATION"
            and training_config.get("run_id") == CONFIRMATION_RUN_ID
            and int(training_config.get("training_seed", -1)) == seed
            and training_config.get("selected_model") == CONFIRMATION_RUN_ID
            and training_config.get("screen_provenance")
            == config.get("screen_provenance")
            and training_config.get("device") == f"cuda:{gpu}"
            and "A100" in str(training_config.get("device_name", "")),
            f"S1 seed {seed} training config provenance changed",
        )
        attempt_id = str(attempt.get("attempt_id", ""))
        require(
            attempt.get("code_commit") == expected_head
            and attempt.get("status") == "COMPLETED"
            and attempt_id.endswith(f"_runner_{expected_head}"),
            f"S1 seed {seed} training attempt is not unique and HEAD-bound",
        )
        protected_reads_zero(training_config, label=f"seed {seed} training config")
        protected_reads_zero(attempt, label=f"seed {seed} training attempt")
        packages[seed] = {
            "training_summary_path": str(summary_path),
            "training_config_path": str(training_config_path),
            "training_attempt_path": str(attempt_path),
            "checkpoint_paths": {
                checkpoint_pass: str(path)
                for checkpoint_pass, path in checkpoints.items()
            },
            "training_physical_gpu_index": gpu,
        }
    return packages


def build_validation_inventory_s1(
    configs: Mapping[int, Path],
    packages: Mapping[int, Mapping[str, Any]],
    physical_gpus: Sequence[int],
    *,
    authorization_path: Path,
    log_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpus = tuple(int(gpu) for gpu in physical_gpus)
    require(
        gpus == tuple(range(6)),
        "S1 confirmation Validation GPU order changed from configured 0-5",
    )
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    inventory: list[dict[str, Any]] = []
    pairs = [
        (seed, checkpoint_pass)
        for seed in CONFIRMATION_SEEDS
        for checkpoint_pass in CHECKPOINT_PASSES
    ]
    for index, (seed, checkpoint_pass) in enumerate(pairs):
        gpu = gpus[index % len(gpus)]
        config_path = configs[seed]
        config = read_json(config_path)
        output = (
            Path(str(config["validation_output_root"]))
            / CONFIRMATION_RUN_ID
            / f"pass_{checkpoint_pass}"
        )
        checkpoint = Path(
            str(packages[seed]["checkpoint_paths"][checkpoint_pass])
        )
        job = {
            "job_key": f"setflow:{seed}:pass_{checkpoint_pass}",
            "run_id": CONFIRMATION_RUN_ID,
            "training_seed": seed,
            "checkpoint_pass": checkpoint_pass,
            "physical_gpu_index": gpu,
            "config_path": str(config_path),
            "training_summary_path": packages[seed]["training_summary_path"],
            "checkpoint_path": str(checkpoint),
            "confirmation_authorization_path": str(authorization_path),
            "terminal_summary": str(output / "validation_summary.json"),
            "terminal_failure": str(output.with_name(output.name + ".failed.json")),
            "log_path": str(log_root / f"setflow_{seed}_pass_{checkpoint_pass}.log"),
            "command": [
                str(PYTHON),
                str(VALIDATOR),
                "--config",
                str(config_path),
                "--run-id",
                CONFIRMATION_RUN_ID,
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
        inventory.append(job)
        queues[gpu].append(job)
    require(
        len(inventory) == 12
        and len({job["job_key"] for job in inventory}) == 12
        and all(len(rows) == 2 for rows in queues.values())
        and {job["run_id"] for job in inventory} == {CONFIRMATION_RUN_ID},
        "S1 confirmation Validation is not exact 12-job full-only round-robin",
    )
    return inventory, [
        {"physical_gpu_index": gpu, "jobs": queues[gpu]} for gpu in gpus
    ]


def require_fresh_posttraining_targets_s1(
    *,
    runtime_root: Path,
    log_root: Path,
    gate: Path,
    prelaunch_failure: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    fixed = (
        runtime_root,
        runtime_root.with_name(runtime_root.name + ".partial"),
        log_root,
        log_root.with_name(log_root.name + ".partial"),
        gate,
        gate.with_suffix(gate.suffix + ".partial"),
        gate.with_name(gate.name + ".failed.json"),
        gate.with_name(gate.name + ".failed.json.partial"),
        prelaunch_failure,
        prelaunch_failure.with_suffix(prelaunch_failure.suffix + ".partial"),
    )
    require(
        all(not path.exists() for path in fixed),
        "S1 confirmation posttraining family, final, failure, or partial exists",
    )
    for job in inventory:
        summary = Path(str(job["terminal_summary"]))
        failure = Path(str(job["terminal_failure"]))
        require(
            not summary.parent.exists()
            and not summary.exists()
            and not summary.with_suffix(summary.suffix + ".partial").exists()
            and not failure.exists()
            and not failure.with_suffix(failure.suffix + ".partial").exists(),
            f"S1 Validation target already exists: {job['job_key']}",
        )


def write_posttraining_prelaunch_failure_s1(
    path: Path,
    *,
    runner_head: str,
    runtime_root: Path,
    configured_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
    error: Exception,
) -> None:
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_posttraining_prelaunch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_POSTTRAINING_STOPPED_BEFORE_FAMILY_MATERIALIZATION",
            "runner_git_head": runner_head,
            "failure_stage": (
                "A100_CUDA_BF16_PROBE_BEFORE_POSTTRAINING"
                if isinstance(error, XEditSetFlowS1GpuError)
                and error.failed_physical_gpu_index is not None
                else "GPU0_5_INVENTORY_BEFORE_POSTTRAINING"
            ),
            "inventory_command": list(GPU_INVENTORY_COMMAND),
            "configured_physical_gpus": [int(gpu) for gpu in configured_gpus],
            "gpu_diagnostics": {
                str(gpu): dict(row) for gpu, row in diagnostics.items()
            },
            "completed_cuda_bf16_probes": {
                str(gpu): dict(row) for gpu, row in probes.items()
            },
            **_gpu_error_details(error),
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_runtime_root": str(runtime_root),
            "runtime_root_created": runtime_root.exists(),
            "scheduler_started": False,
            "validation_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def perform_posttraining_gpu_preflight_s1(
    *,
    runner_head: str,
    configured_gpus: Sequence[int],
    failure_path: Path,
    runtime_root: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    diagnostics: dict[int, dict[str, Any]] = {}
    probes: dict[int, dict[str, Any]] = {}
    try:
        diagnostics = gpu_diagnostics(configured_gpus)
        for gpu in configured_gpus:
            probes[int(gpu)] = cuda_bf16_probe(int(gpu))
    except Exception as error:
        write_posttraining_prelaunch_failure_s1(
            failure_path,
            runner_head=runner_head,
            runtime_root=runtime_root,
            configured_gpus=configured_gpus,
            diagnostics=diagnostics,
            probes=probes,
            error=error,
        )
        raise
    return diagnostics, probes


def build_posttraining_schedule_s1(
    training_schedule_path: Path,
    training_schedule: Mapping[str, Any],
    bindings: Mapping[str, Any],
    configs: Mapping[int, Path],
    packages: Mapping[int, Mapping[str, Any]],
    physical_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    probes: Mapping[int, Mapping[str, Any]],
    *,
    expected_head: str,
) -> dict[str, Any]:
    runtime_root = Path(str(bindings["posttraining_runtime_root"]))
    log_root = Path(str(bindings["posttraining_log_root"]))
    authorization_path = Path(str(bindings["confirmation_authorization_path"]))
    inventory, queues = build_validation_inventory_s1(
        configs,
        packages,
        physical_gpus,
        authorization_path=authorization_path,
        log_root=log_root,
    )
    gpus = tuple(int(gpu) for gpu in physical_gpus)
    require(
        all(gpu in diagnostics and gpu in probes for gpu in gpus),
        "S1 posttraining GPU evidence is incomplete",
    )
    gate = Path(str(bindings["confirmation_gate_output"]))
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_posttraining_schedule.v1"
        ),
        "status": "FROZEN_S1_CONFIRMATION_POSTTRAINING_SCHEDULE",
        "git_head": expected_head,
        "training_runner_git_head": expected_head,
        "screen_experiment_head": SCREEN_HEAD,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_root / "runtime.json"),
        "eligible_components": ["setflow"],
        "confirmation_training_schedule": str(training_schedule_path),
        "confirmation_training_runtime": str(bindings["training_runtime_path"]),
        "posttraining_bindings": dict(bindings),
        "validation_inventory": inventory,
        "validation_queues": queues,
        "adjudications": {
            "setflow": {
                "gate_path": str(gate),
                "failure_path": str(gate.with_name(gate.name + ".failed.json")),
                "log_path": str(log_root / "setflow_adjudication.log"),
                "command": [
                    str(PYTHON),
                    str(ADJUDICATOR),
                    "--protocol",
                    str(bindings["protocol_path"]),
                    "--config-manifest",
                    str(bindings["config_manifest_path"]),
                ],
            }
        },
        "gpu_diagnostics_before_launch": {
            str(gpu): dict(diagnostics[gpu]) for gpu in gpus
        },
        "cuda_bf16_probes": {
            str(gpu): dict(probes[gpu]) for gpu in gpus
        },
        "gpu_selection_policy": "CONFIG_ORDER_GPU0_5_ROUND_ROBIN_TWO_EACH_WITHOUT_MEMORY_SORTING_OR_GATE",
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require(
        training_schedule.get("posttraining_bindings") == bindings,
        "S1 posttraining did not consume frozen training bindings exactly",
    )
    return schedule


def write_scheduler_launch_failure_s1(
    path: Path,
    *,
    runner_head: str,
    command_line: Sequence[str],
    schedule_path: Path,
    runtime_path: Path,
    error: Exception,
) -> None:
    write_new_atomic(
        path,
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_posttraining_scheduler_launch_failure.v1"
            ),
            "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_POSTTRAINING_SCHEDULER_LAUNCH_TECHNICAL_FAILURE",
            "runner_git_head": runner_head,
            "failure_stage": "CONFIRMATION_POSTTRAINING_SCHEDULER_PROCESS_LAUNCH",
            "scheduler_command": list(command_line),
            "schedule_path": str(schedule_path),
            "runtime_manifest": str(runtime_path),
            "error_type": type(error).__name__,
            "error": str(error),
            "scheduler_started": False,
            "validation_job_started": False,
            "automatic_retry_attempted": False,
            "free_memory_gate_applied": False,
            "cpu_fallback_used": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def run(expected_head: str, training_schedule_path: Path) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected Git HEAD is invalid",
    )
    for path, label in (
        (PYTHON, "formal Python"),
        (PROTOCOL, "S1 confirmation protocol"),
        (VALIDATOR, "S1 checkpoint validator"),
        (ADJUDICATOR, "S1 confirmation adjudicator"),
        (POSTTRAINING_SCHEDULER, "generic confirmation posttraining scheduler"),
        (TRAINING_SCHEDULER, "generic confirmation training scheduler"),
        (training_schedule_path, "S1 confirmation training schedule"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require_exact_pushed_clean_head(expected_head)

    protocol = read_json(PROTOCOL)
    schedule = read_json(training_schedule_path)
    bindings, schedule_jobs = validate_training_schedule_s1(
        schedule, expected_head=expected_head
    )
    manifest_path = Path(str(bindings["config_manifest_path"]))
    authorization_path = Path(str(bindings["confirmation_authorization_path"]))
    training_runtime_path = Path(str(bindings["training_runtime_path"]))
    for path, label in (
        (manifest_path, "config manifest"),
        (authorization_path, "confirmation authorization"),
        (training_runtime_path, "training runtime"),
    ):
        require(path.is_file(), f"S1 {label} is absent: {path}")
    manifest = read_json(manifest_path)
    configs = validate_manifest_s1(manifest, protocol, runner_head=expected_head)
    authorization = read_json(authorization_path)
    validate_authorization_s1(
        authorization,
        read_json(configs[CONFIRMATION_SEEDS[0]]),
        runner_head=expected_head,
    )
    training_runtime = read_json(training_runtime_path)
    packages = validate_successful_training_package_s1(
        training_runtime,
        schedule_jobs,
        configs,
        expected_head=expected_head,
    )

    runtime_root = Path(str(bindings["posttraining_runtime_root"]))
    log_root = Path(str(bindings["posttraining_log_root"]))
    gate = Path(str(bindings["confirmation_gate_output"]))
    require(
        runtime_root
        == format_output_path(
            protocol, "posttraining_runtime_root_template", expected_head
        )
        and log_root
        == format_output_path(protocol, "posttraining_log_root_template", expected_head)
        and gate
        == format_output_path(
            protocol, "confirmation_gate_output_template", expected_head
        ),
        "S1 posttraining output family differs from frozen protocol",
    )
    inventory, _ = build_validation_inventory_s1(
        configs,
        packages,
        tuple(range(6)),
        authorization_path=authorization_path,
        log_root=log_root,
    )
    prelaunch_failure = runtime_root.with_name(runtime_root.name + ".failed.json")
    require_fresh_posttraining_targets_s1(
        runtime_root=runtime_root,
        log_root=log_root,
        gate=gate,
        prelaunch_failure=prelaunch_failure,
        inventory=inventory,
    )
    gpu_policy = protocol.get("gpu_policy")
    require(
        isinstance(gpu_policy, Mapping)
        and gpu_policy.get("physical_gpu_scope") == list(range(6))
        and gpu_policy.get("cuda_bf16_only") is True
        and gpu_policy.get("cpu_fallback") is False
        and gpu_policy.get("free_or_estimated_memory_gate") is False
        and gpu_policy.get("free_or_estimated_memory_sorting") is False,
        "S1 posttraining GPU policy changed",
    )
    physical_gpus = tuple(int(gpu) for gpu in gpu_policy["physical_gpu_scope"])
    diagnostics, probes = perform_posttraining_gpu_preflight_s1(
        runner_head=expected_head,
        configured_gpus=physical_gpus,
        failure_path=prelaunch_failure,
        runtime_root=runtime_root,
    )
    posttraining_schedule = build_posttraining_schedule_s1(
        training_schedule_path,
        schedule,
        bindings,
        configs,
        packages,
        physical_gpus,
        diagnostics,
        probes,
        expected_head=expected_head,
    )
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    schedule_path = runtime_root / "schedule.json"
    runtime_path = runtime_root / "runtime.json"
    require(
        Path(str(posttraining_schedule["runtime_manifest"])) == runtime_path,
        "S1 posttraining runtime path differs from frozen family",
    )
    write_new_atomic(schedule_path, posttraining_schedule)
    coordinator_log = log_root / "coordinator.log"
    scheduler_command = [
        str(PYTHON),
        str(POSTTRAINING_SCHEDULER),
        "--schedule",
        str(schedule_path),
    ]
    stream = coordinator_log.open("w", encoding="utf-8")
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
        write_scheduler_launch_failure_s1(
            runtime_root / "scheduler_launch.failed.json",
            runner_head=expected_head,
            command_line=scheduler_command,
            schedule_path=schedule_path,
            runtime_path=runtime_path,
            error=error,
        )
        raise
    stream.close()
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_posttraining_launch.v1"
        ),
        "status": "XEDITSETFLOW_V4_S1_CONFIRMATION_POSTTRAINING_SCHEDULER_LAUNCHED",
        "runner_git_head": expected_head,
        "coordinator_pid": process.pid,
        "training_schedule_path": str(training_schedule_path),
        "training_runtime_path": str(training_runtime_path),
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_path),
        "coordinator_log": str(coordinator_log),
        "confirmation_gate_output": str(gate),
        "validation_job_count": 12,
        "selected_physical_gpus": list(physical_gpus),
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "development_test_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--training-schedule", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.expected_head, arguments.training_schedule),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
