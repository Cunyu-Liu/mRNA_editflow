#!/usr/bin/env python3
"""Launch recovered SetFlow V4.0.3 confirmation Validation after exact training."""

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

from scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation import (
    CONFIRMATION_SEEDS,
    SCREEN_EXPERIMENT_HEAD,
    TRAINING_HEAD,
    VALIDATION_HEAD,
    require_recovery_config_derivation_v403,
    require_recovery_terminal_v403,
    require_runner_verification_receipt_v403,
    require_science_protocol_unchanged_v403,
)
from scripts.route_a_v3.launch_route2_xeditsetflow_v403_recovered_confirmation import (
    cuda_bf16_probe,
    gpu_diagnostics,
    validate_authorization_v403,
    validate_manifest_v403,
)


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_protocol_v1.json"
)
BASE_CONFIRMATION_PROTOCOL = (
    WORKTREE
    / "configs/route_a_v3_route2_xeditsetflow_v4_confirmation_protocol_v1.json"
)
SCREEN_CONFIG = (
    WORKTREE / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json"
)
VALIDATOR = (
    WORKTREE
    / "scripts/route_a_v3/validate_route2_xeditsetflow_v4_checkpoint.py"
)
ADJUDICATOR = (
    WORKTREE
    / "scripts/route_a_v3/adjudicate_route2_xeditsetflow_v4_confirmation.py"
)
POSTTRAINING_SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
)
CHECKPOINT_PASSES = (4, 6, 8, 10)


class XEditSetFlowV403ConfirmationPosttrainingLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowV403ConfirmationPosttrainingLaunchError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"artifact already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    require(not partial.exists(), f"partial artifact already exists: {partial}")
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


def require_zero_protected_reads(
    payload: Mapping[str, Any], *, label: str
) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"SetFlow V4.0.3 {label} reports a protected outcome read",
    )


def validate_training_schedule_v403(
    schedule: Mapping[str, Any], *, training_runner_head: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_training_schedule.v1"
        and schedule.get("status")
        == "FROZEN_RECOVERY_DERIVED_CONFIRMATION_TRAINING_SCHEDULE",
        "SetFlow V4.0.3 recovered confirmation training schedule changed",
    )
    require(
        schedule.get("git_head") == training_runner_head
        and schedule.get("training_git_head") == TRAINING_HEAD
        and schedule.get("validation_git_head") == VALIDATION_HEAD
        and schedule.get("experiment_head") == SCREEN_EXPERIMENT_HEAD,
        "SetFlow V4.0.3 training schedule Git provenance changed",
    )
    require(
        schedule.get("eligible_components") == ["setflow"]
        and schedule.get("required_seeds") == list(CONFIRMATION_SEEDS)
        and schedule.get("training_reused_from_screen") is False
        and schedule.get("screen_training_reused_by_recovery") is True
        and int(schedule.get("recovery_parameter_update_count", -1)) == 0
        and schedule.get("free_memory_gate_applied") is False,
        "SetFlow V4.0.3 confirmation training scope changed",
    )
    require(
        schedule.get("active_performance_output_read") is False,
        "SetFlow V4.0.3 training schedule reports an active performance read",
    )
    require_zero_protected_reads(schedule, label="training schedule")

    bindings = schedule.get("posttraining_bindings")
    require(isinstance(bindings, Mapping), "SetFlow V4.0.3 posttraining bindings are absent")
    require(
        bindings.get("training_git_head") == TRAINING_HEAD
        and bindings.get("validation_git_head") == VALIDATION_HEAD
        and bindings.get("runner_git_head") == training_runner_head
        and bindings.get("training_runtime_path") == schedule.get("runtime_manifest")
        and bindings.get("protocol_path") == schedule.get("confirmation_protocol")
        and bindings.get("recovery_config_path") == schedule.get("recovery_config")
        and bindings.get("recovered_screen_gate_path")
        == schedule.get("recovered_screen_gate")
        and bindings.get("confirmation_authorization_path")
        == schedule.get("confirmation_authorization")
        and bindings.get("config_manifest_path") == schedule.get("config_manifest"),
        "SetFlow V4.0.3 posttraining bindings do not match the training schedule",
    )

    jobs: dict[int, dict[str, Any]] = {}
    for queue in schedule.get("gpu_queues", []):
        gpu = int(queue.get("physical_gpu_index", -1))
        for raw_job in queue.get("jobs", []):
            job = dict(raw_job)
            seed = int(job.get("training_seed", -1))
            require(
                job.get("job_key") == f"setflow:{seed}:v4_full"
                and job.get("component") == "setflow"
                and job.get("run_id") == "v4_full"
                and seed in CONFIRMATION_SEEDS
                and seed not in jobs,
                "SetFlow V4.0.3 confirmation training job inventory changed",
            )
            require(
                str(gpu) in schedule.get("gpu_diagnostics_before_launch", {})
                and str(gpu) in schedule.get("cuda_bf16_probes", {}),
                f"SetFlow V4.0.3 training GPU evidence is absent: {gpu}",
            )
            job["physical_gpu_index"] = gpu
            jobs[seed] = job
    require(
        set(jobs) == set(CONFIRMATION_SEEDS) and len(jobs) == 3,
        "SetFlow V4.0.3 confirmation training job cohort changed",
    )
    return dict(bindings), jobs


def validate_posttraining_bindings_v403(
    schedule: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    training_runner_head: str,
    expected_protocol_path: Path,
) -> None:
    provenance = protocol["validation_recovery_provenance"]
    posttraining = protocol["posttraining_binding"]
    outputs = protocol["runner_outputs"]
    require(
        Path(str(bindings.get("protocol_path"))).resolve()
        == expected_protocol_path.resolve(),
        "SetFlow V4.0.3 posttraining protocol is not the recovered protocol",
    )
    require(
        bindings.get("training_git_head") == provenance.get("training_git_head")
        == TRAINING_HEAD
        and bindings.get("validation_git_head")
        == provenance.get("validation_git_head")
        == VALIDATION_HEAD
        and bindings.get("runner_git_head") == training_runner_head,
        "SetFlow V4.0.3 posttraining dual/triple-HEAD binding changed",
    )
    require(
        bindings.get("recovery_config_path")
        == provenance.get("recovery_config_path")
        == posttraining.get("recovery_config_path")
        and bindings.get("recovered_screen_gate_path")
        == provenance.get("recovered_screen_gate_path")
        == posttraining.get("recovered_screen_gate_path")
        and bindings.get("config_manifest_path")
        == posttraining.get("config_manifest_path")
        and bindings.get("confirmation_gate_output")
        == protocol.get("confirmation_gate_output")
        == posttraining.get("confirmation_gate_output"),
        "SetFlow V4.0.3 recovered artifact paths changed",
    )
    require(
        bindings.get("posttraining_runtime_root")
        == outputs["posttraining_runtime_root_template"].format(
            runner_git_head=training_runner_head
        )
        and bindings.get("posttraining_log_root")
        == outputs["posttraining_log_root_template"].format(
            runner_git_head=training_runner_head
        ),
        "SetFlow V4.0.3 posttraining output family changed",
    )
    require(
        bindings.get("confirmation_authorization_path")
        == outputs["authorization_output_template"].format(
            runner_git_head=training_runner_head
        )
        == schedule.get("confirmation_authorization"),
        "SetFlow V4.0.3 recovered confirmation authorization path changed",
    )
    legacy_root = ROOT / f"experiments/xedit_v4/confirmation_posttraining_{training_runner_head}"
    require(
        Path(str(bindings["posttraining_runtime_root"])) != legacy_root,
        "SetFlow V4.0.3 posttraining would overwrite the legacy confirmation family",
    )


def validate_successful_training_package_v403(
    runtime: Mapping[str, Any],
    schedule_jobs: Mapping[int, Mapping[str, Any]],
    configs: Mapping[int, Path],
    *,
    training_runner_head: str,
    recovered_screen_gate_path: str,
) -> dict[int, dict[str, Any]]:
    require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xedit_v4_confirmation_training_runtime.v1"
        and runtime.get("status")
        == "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
        and runtime.get("git_head") == training_runner_head
        and runtime.get("experiment_head") == SCREEN_EXPERIMENT_HEAD
        and runtime.get("eligible_components") == ["setflow"],
        "SetFlow V4.0.3 confirmation training runtime is not exact terminal",
    )
    require(
        runtime.get("active_performance_output_read") is False,
        "SetFlow V4.0.3 training runtime reports an active performance read",
    )
    require_zero_protected_reads(runtime, label="training runtime")
    expected_keys = {
        f"setflow:{seed}:v4_full" for seed in CONFIRMATION_SEEDS
    }
    runtime_jobs = runtime.get("jobs", {})
    require(
        isinstance(runtime_jobs, Mapping) and set(runtime_jobs) == expected_keys,
        "SetFlow V4.0.3 training runtime job inventory changed",
    )

    packages: dict[int, dict[str, Any]] = {}
    for seed in CONFIRMATION_SEEDS:
        schedule_job = schedule_jobs[seed]
        runtime_job = runtime_jobs[f"setflow:{seed}:v4_full"]
        config = read_json(configs[seed])
        output = Path(str(schedule_job["output_directory"]))
        require(
            output == Path(str(config["output_root"])) / "v4_full",
            f"SetFlow V4.0.3 seed {seed} training output does not match recovered config",
        )
        require(
            runtime_job.get("terminal_artifact_kind") == "SUMMARY"
            and runtime_job.get("status") == "TERMINAL_COMPLETE"
            and int(runtime_job.get("training_seed", -1)) == seed
            and runtime_job.get("run_id") == "v4_full"
            and Path(str(runtime_job.get("output_directory"))) == output
            and int(runtime_job.get("physical_gpu_index", -1))
            == int(schedule_job["physical_gpu_index"]),
            f"SetFlow V4.0.3 seed {seed} lacks its exact terminal SUMMARY",
        )
        summary_path = output / "training_summary.json"
        failure_path = output / "failure.json"
        require(
            summary_path.is_file() and not failure_path.exists(),
            f"SetFlow V4.0.3 seed {seed} is not exactly summary-terminal",
        )
        summary = read_json(summary_path)
        require(
            summary.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_training_summary.v1"
            and summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"
            and summary.get("run_stage") == "CONFIRMATION"
            and summary.get("run_id") == "v4_full"
            and int(summary.get("seed", -1)) == seed
            and int(summary.get("completed_passes", -1)) == 10
            and summary.get("checkpoint_selection_status")
            == "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION"
            and summary.get("validation_generation_during_training") is False
            and summary.get("parameter_changed") is True
            and int(summary.get("optimizer_update_count", 0)) > 0,
            f"SetFlow V4.0.3 seed {seed} training summary identity changed",
        )
        require(
            str(summary.get("torch_device", "")).startswith("cuda:")
            and summary.get("training_precision") == "BF16"
            and summary.get("cpu_fallback_used") is False
            and int(summary.get("physical_gpu_index", -1))
            == int(schedule_job["physical_gpu_index"]),
            f"SetFlow V4.0.3 seed {seed} was not trained on the bound CUDA GPU",
        )
        require_zero_protected_reads(summary, label=f"seed {seed} training summary")

        checkpoint_paths = {
            checkpoint_pass: output / f"pass_{checkpoint_pass}.pt"
            for checkpoint_pass in CHECKPOINT_PASSES
        }
        require(
            summary.get("saved_checkpoint_paths")
            == {
                str(checkpoint_pass): str(path)
                for checkpoint_pass, path in checkpoint_paths.items()
            }
            and all(path.is_file() for path in checkpoint_paths.values()),
            f"SetFlow V4.0.3 seed {seed} checkpoint package changed",
        )

        training_config = read_json(output / "training_config.json")
        training_attempt = read_json(output / "training_attempt.json")
        require(
            training_config.get("authorized_git_head") == training_runner_head
            and training_attempt.get("code_commit") == training_runner_head
            and training_config.get("run_stage") == "CONFIRMATION"
            and training_config.get("run_id") == "v4_full"
            and int(training_config.get("training_seed", -1)) == seed
            and training_config.get("screen_gate_path")
            == config.get("screen_gate_path")
            == recovered_screen_gate_path,
            f"SetFlow V4.0.3 seed {seed} training provenance changed",
        )
        derivation = training_config.get("validation_recovery", {})
        require(
            derivation.get("training_git_head") == TRAINING_HEAD
            and derivation.get("validation_git_head") == VALIDATION_HEAD
            and int(derivation.get("parameter_updates", -1)) == 0
            and derivation.get("scientific_thresholds_changed") is False
            and str(training_config.get("device", "")).startswith("cuda:")
            and training_config.get("development_test_outcomes_accessed") is False
            and training_config.get("new_final_evaluation_outcomes_accessed") is False,
            f"SetFlow V4.0.3 seed {seed} recovery derivation changed",
        )
        packages[seed] = {
            "training_summary_path": str(summary_path),
            "training_config_path": str(output / "training_config.json"),
            "training_attempt_path": str(output / "training_attempt.json"),
            "checkpoint_paths": {
                checkpoint_pass: str(path)
                for checkpoint_pass, path in checkpoint_paths.items()
            },
            "training_physical_gpu_index": int(schedule_job["physical_gpu_index"]),
        }
    return packages


def build_validation_inventory_v403(
    configs: Mapping[int, Path],
    packages: Mapping[int, Mapping[str, Any]],
    physical_gpus: Sequence[int],
    *,
    authorization_path: Path,
    recovered_screen_gate_path: str,
    log_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpus = tuple(int(gpu) for gpu in physical_gpus)
    require(
        len(gpus) == 6 and len(set(gpus)) == 6,
        "SetFlow V4.0.3 confirmation Validation requires six distinct physical GPUs",
    )
    queues = {gpu: [] for gpu in gpus}
    inventory: list[dict[str, Any]] = []
    for index, (seed, checkpoint_pass) in enumerate(
        (seed, checkpoint_pass)
        for seed in CONFIRMATION_SEEDS
        for checkpoint_pass in CHECKPOINT_PASSES
    ):
        gpu = gpus[index % len(gpus)]
        config_path = configs[seed]
        config = read_json(config_path)
        output = (
            Path(str(config["validation_output_root"]))
            / "v4_full"
            / f"pass_{checkpoint_pass}"
        )
        checkpoint_path = Path(
            str(packages[seed]["checkpoint_paths"][checkpoint_pass])
        )
        job = {
            "job_key": f"setflow:{seed}:pass_{checkpoint_pass}",
            "training_seed": seed,
            "checkpoint_pass": checkpoint_pass,
            "physical_gpu_index": gpu,
            "config_path": str(config_path),
            "training_summary_path": packages[seed]["training_summary_path"],
            "checkpoint_path": str(checkpoint_path),
            "recovered_screen_gate_path": recovered_screen_gate_path,
            "confirmation_authorization_path": str(authorization_path),
            "terminal_summary": str(output / "validation_summary.json"),
            "terminal_failure": str(
                output.with_name(output.name + ".failed.json")
            ),
            "log_path": str(
                log_root / f"setflow_{seed}_pass_{checkpoint_pass}.log"
            ),
            "command": [
                str(PYTHON),
                str(VALIDATOR),
                "--config",
                str(config_path),
                "--run-id",
                "v4_full",
                "--checkpoint-pass",
                str(checkpoint_pass),
                "--authorization",
                str(authorization_path),
                "--physical-gpu-index",
                str(gpu),
            ],
        }
        inventory.append(job)
        queues[gpu].append(job)
    require(
        len(inventory) == 12
        and {job["job_key"] for job in inventory}
        == {
            f"setflow:{seed}:pass_{checkpoint_pass}"
            for seed in CONFIRMATION_SEEDS
            for checkpoint_pass in CHECKPOINT_PASSES
        }
        and all(len(rows) == 2 for rows in queues.values()),
        "SetFlow V4.0.3 confirmation Validation inventory changed",
    )
    return inventory, [
        {"physical_gpu_index": gpu, "jobs": queues[gpu]} for gpu in gpus
    ]


def require_fresh_posttraining_targets_v403(
    *, runtime_root: Path, log_root: Path, confirmation_gate: Path
) -> None:
    targets = (
        runtime_root,
        log_root,
        confirmation_gate,
        confirmation_gate.with_suffix(confirmation_gate.suffix + ".partial"),
    )
    require(
        all(not path.exists() for path in targets),
        "SetFlow V4.0.3 recovered confirmation posttraining attempt already exists",
    )


def require_validation_targets_absent_v403(
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    for job in inventory:
        summary = Path(str(job["terminal_summary"]))
        failure = Path(str(job["terminal_failure"]))
        require(
            not summary.parent.exists()
            and not failure.exists()
            and not failure.with_suffix(failure.suffix + ".partial").exists(),
            f"SetFlow V4.0.3 confirmation Validation target exists: {job['job_key']}",
        )


def build_posttraining_schedule_v403(
    training_schedule_path: Path,
    training_schedule: Mapping[str, Any],
    bindings: Mapping[str, Any],
    configs: Mapping[int, Path],
    packages: Mapping[int, Mapping[str, Any]],
    physical_gpus: Sequence[int],
    diagnostics: Mapping[int, Mapping[str, Any]],
    cuda_probes: Mapping[int, Mapping[str, Any]],
    *,
    orchestration_head: str,
    training_runner_head: str,
) -> dict[str, Any]:
    runtime_root = Path(str(bindings["posttraining_runtime_root"]))
    log_root = Path(str(bindings["posttraining_log_root"]))
    authorization_path = Path(str(bindings["confirmation_authorization_path"]))
    inventory, queues = build_validation_inventory_v403(
        configs,
        packages,
        physical_gpus,
        authorization_path=authorization_path,
        recovered_screen_gate_path=str(bindings["recovered_screen_gate_path"]),
        log_root=log_root,
    )
    gpus = tuple(int(gpu) for gpu in physical_gpus)
    require(
        all(gpu in diagnostics and gpu in cuda_probes for gpu in gpus),
        "SetFlow V4.0.3 confirmation Validation GPU evidence is incomplete",
    )
    gate = Path(str(bindings["confirmation_gate_output"]))
    adjudication_failure = runtime_root / "setflow_adjudication.failed.json"
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_posttraining_schedule.v1"
        ),
        "status": "FROZEN_RECOVERED_CONFIRMATION_POSTTRAINING_SCHEDULE",
        "git_head": orchestration_head,
        "orchestration_git_head": orchestration_head,
        "training_runner_git_head": training_runner_head,
        "training_git_head": TRAINING_HEAD,
        "recovery_validation_git_head": VALIDATION_HEAD,
        "screen_experiment_head": SCREEN_EXPERIMENT_HEAD,
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
                "failure_path": str(adjudication_failure),
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
            str(gpu): dict(cuda_probes[gpu]) for gpu in gpus
        },
        "gpu_selection_policy": (
            "RECOVERY_CONFIG_ORDER_ROUND_ROBIN_WITHOUT_MEMORY_SORTING_OR_GATE"
        ),
        "free_memory_gate_applied": False,
        "cpu_fallback_used": False,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    require(
        training_schedule.get("posttraining_bindings") == bindings,
        "SetFlow V4.0.3 posttraining schedule did not consume source bindings exactly",
    )
    return schedule


def run(
    orchestration_head: str,
    training_runner_head: str,
    training_schedule_path: Path,
) -> dict[str, Any]:
    for head, label in (
        (orchestration_head, "orchestration"),
        (training_runner_head, "training runner"),
    ):
        require(
            re.fullmatch(r"[0-9a-f]{40}", head) is not None,
            f"expected {label} Git HEAD is invalid",
        )
    for path, label in (
        (PYTHON, "formal Python"),
        (PROTOCOL, "recovered confirmation protocol"),
        (BASE_CONFIRMATION_PROTOCOL, "base confirmation protocol"),
        (SCREEN_CONFIG, "screen config"),
        (VALIDATOR, "SetFlow checkpoint validator"),
        (ADJUDICATOR, "SetFlow confirmation adjudicator"),
        (POSTTRAINING_SCHEDULER, "generic confirmation posttraining scheduler"),
        (training_schedule_path, "recovered confirmation training schedule"),
    ):
        require(path.is_file(), f"{label} is absent: {path}")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip()
        == orchestration_head,
        "A100 worktree is not at expected orchestration HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 orchestration worktree is dirty",
    )

    schedule = read_json(training_schedule_path)
    bindings, schedule_jobs = validate_training_schedule_v403(
        schedule, training_runner_head=training_runner_head
    )
    protocol_path = Path(str(bindings["protocol_path"]))
    require(
        protocol_path.resolve() == PROTOCOL.resolve(),
        "training schedule does not bind the recovered confirmation protocol",
    )
    protocol = read_json(protocol_path)
    base_protocol = read_json(BASE_CONFIRMATION_PROTOCOL)
    screen_config = read_json(SCREEN_CONFIG)
    recovery_config_path = Path(str(bindings["recovery_config_path"]))
    recovery_runtime_path = Path(str(schedule["recovery_runtime"]))
    recovered_gate_path = Path(str(bindings["recovered_screen_gate_path"]))
    authorization_path = Path(str(bindings["confirmation_authorization_path"]))
    manifest_path = Path(str(bindings["config_manifest_path"]))
    training_runtime_path = Path(str(bindings["training_runtime_path"]))
    receipt_path = Path(str(schedule["runner_verification_receipt"]))
    for path, label in (
        (recovery_config_path, "recovery config"),
        (recovery_runtime_path, "recovery runtime"),
        (recovered_gate_path, "recovered screen gate"),
        (authorization_path, "recovered confirmation authorization"),
        (manifest_path, "recovered confirmation config manifest"),
        (training_runtime_path, "recovered confirmation training runtime"),
        (receipt_path, "training runner verification receipt"),
    ):
        require(path.is_file(), f"SetFlow V4.0.3 {label} is absent: {path}")

    validate_posttraining_bindings_v403(
        schedule,
        protocol,
        bindings,
        training_runner_head=training_runner_head,
        expected_protocol_path=PROTOCOL,
    )
    require_science_protocol_unchanged_v403(base_protocol, protocol)
    recovery_config = read_json(recovery_config_path)
    recovery_runtime = read_json(recovery_runtime_path)
    recovered_gate = read_json(recovered_gate_path)
    require_recovery_config_derivation_v403(
        screen_config, recovery_config, protocol
    )
    require_recovery_terminal_v403(protocol, recovery_runtime, recovered_gate)
    receipt = read_json(receipt_path)
    require_runner_verification_receipt_v403(
        receipt, current_runner_head=training_runner_head
    )
    authorization = read_json(authorization_path)
    validate_authorization_v403(
        authorization, runner_head=training_runner_head
    )
    require(
        authorization.get("authorization_derivation")
        == "V403_RECOVERED_SCREEN_PASS_WITH_DISTINCT_TRAINING_VALIDATION_AND_RUNNER_HEADS"
        and authorization.get("recovery_config_path")
        == str(recovery_config_path)
        and authorization.get("recovered_screen_gate_path")
        == str(recovered_gate_path)
        and authorization.get("runner_current_head_verification", {}).get(
            "runner_git_head"
        )
        == training_runner_head,
        "SetFlow V4.0.3 recovered confirmation authorization derivation changed",
    )
    require_zero_protected_reads(protocol, label="recovered protocol")
    require_zero_protected_reads(authorization, label="confirmation authorization")

    manifest = read_json(manifest_path)
    require_zero_protected_reads(manifest, label="confirmation config manifest")
    configs = validate_manifest_v403(manifest, protocol)
    training_runtime = read_json(training_runtime_path)
    packages = validate_successful_training_package_v403(
        training_runtime,
        schedule_jobs,
        configs,
        training_runner_head=training_runner_head,
        recovered_screen_gate_path=str(recovered_gate_path),
    )

    runtime_root = Path(str(bindings["posttraining_runtime_root"]))
    log_root = Path(str(bindings["posttraining_log_root"]))
    confirmation_gate = Path(str(bindings["confirmation_gate_output"]))
    require_fresh_posttraining_targets_v403(
        runtime_root=runtime_root,
        log_root=log_root,
        confirmation_gate=confirmation_gate,
    )
    physical_gpus = tuple(
        int(gpu) for gpu in recovery_config["gpu_policy"]["physical_gpu_scope"]
    )
    require(
        physical_gpus == (0, 1, 2, 3, 4, 5)
        and recovery_config["gpu_policy"].get("cuda_bf16_only") is True
        and recovery_config["gpu_policy"].get("cpu_fallback") is False,
        "SetFlow V4.0.3 recovered Validation GPU authorization changed",
    )
    diagnostics = gpu_diagnostics()
    require(
        all(gpu in diagnostics for gpu in physical_gpus),
        "SetFlow V4.0.3 physical GPU inventory 0-5 is incomplete",
    )
    cuda_failure_path = runtime_root.with_name(
        runtime_root.name + "_cuda_failure.json"
    )
    require(
        not cuda_failure_path.exists(),
        "SetFlow V4.0.3 posttraining CUDA failure evidence already exists",
    )
    try:
        probes = {gpu: cuda_bf16_probe(gpu) for gpu in physical_gpus}
    except Exception as error:
        write_new_atomic(
            cuda_failure_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_posttraining_cuda_failure.v1"
                ),
                "status": "STOPPED_BEFORE_CONFIRMATION_VALIDATION_LAUNCH",
                "orchestration_git_head": orchestration_head,
                "training_runner_git_head": training_runner_head,
                "training_git_head": TRAINING_HEAD,
                "recovery_validation_git_head": VALIDATION_HEAD,
                "selected_physical_gpus": list(physical_gpus),
                "gpu_diagnostics": {
                    str(gpu): dict(diagnostics[gpu]) for gpu in physical_gpus
                },
                "error_type": type(error).__name__,
                "error": str(error),
                "validation_jobs_launched": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        raise

    posttraining_schedule = build_posttraining_schedule_v403(
        training_schedule_path,
        schedule,
        bindings,
        configs,
        packages,
        physical_gpus,
        diagnostics,
        probes,
        orchestration_head=orchestration_head,
        training_runner_head=training_runner_head,
    )
    require_validation_targets_absent_v403(
        posttraining_schedule["validation_inventory"]
    )
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    schedule_path = runtime_root / "schedule.json"
    write_new_atomic(schedule_path, posttraining_schedule)
    coordinator_log = log_root / "coordinator.log"
    stream = coordinator_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(PYTHON),
            str(POSTTRAINING_SCHEDULER),
            "--schedule",
            str(schedule_path),
        ],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_posttraining_launch.v1"
        ),
        "status": (
            "XEDITSETFLOW_V403_RECOVERED_CONFIRMATION_POSTTRAINING_COORDINATOR_LAUNCHED"
        ),
        "orchestration_git_head": orchestration_head,
        "training_runner_git_head": training_runner_head,
        "training_git_head": TRAINING_HEAD,
        "recovery_validation_git_head": VALIDATION_HEAD,
        "coordinator_pid": process.pid,
        "training_schedule_path": str(training_schedule_path),
        "schedule_path": str(schedule_path),
        "runtime_manifest": posttraining_schedule["runtime_manifest"],
        "coordinator_log": str(coordinator_log),
        "confirmation_gate_output": str(confirmation_gate),
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
    parser.add_argument("--expected-orchestration-head", required=True)
    parser.add_argument("--expected-training-runner-head", required=True)
    parser.add_argument("--training-schedule", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.expected_orchestration_head,
                arguments.expected_training_runner_head,
                arguments.training_schedule,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
