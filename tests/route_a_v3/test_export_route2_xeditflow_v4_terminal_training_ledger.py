from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "scripts/route_a_v3/export_route2_xeditflow_v4_terminal_training_ledger.py"
)
HISTORICAL_C0_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
HISTORICAL_FULL_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"


def _load_module():
    spec = importlib.util.spec_from_file_location("terminal_training_ledger", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _touch(path: Path, text: str = "payload must not be read\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mutate(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = _read_json(path)
    change(payload)
    _write_json(path, payload)


def _training_artifact(
    root: Path,
    *,
    family: str,
    index: int,
    gpu: int,
    head: str,
    job_key: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    critic_evidence_version: str | None = None,
    held_out_study: str | None = None,
) -> dict[str, Any]:
    seed = seed if seed is not None else 100_000 + index
    job_key = job_key or f"{family}:{index:02d}"
    safe = job_key.replace(":", "_")
    output = root / "training" / family / safe
    output.mkdir(parents=True)
    config_path = root / "configs" / family / f"{safe}.json"
    log_path = root / "logs" / family / f"{safe}.log"
    failure_path = output / "failure.json"
    run_id = run_id or f"run_{index:02d}"
    if family.startswith("critic_"):
        trainer = "train_route2_xeditcritic_v4.py"
        stage = family.removeprefix("critic_").upper()
        critic_evidence_version = critic_evidence_version or "v2"
        assert critic_evidence_version in {"v1", "v2"}
        summary_path = output / "run_summary.json"
        checkpoint_path = output / "final_pass_8_checkpoint.pt"
        summary = {
            "schema_version": (
                f"route_a_v3_route2_xeditcritic_v4_{stage.lower()}_run."
                f"{critic_evidence_version}"
            ),
            "status": f"TERMINAL_XEDITCRITIC_V4_{stage}_RUN_COMPLETE",
            "run_stage": stage,
            "run_id": run_id,
            "seed": seed,
            "physical_gpu_index": gpu,
            "cuda_device_name": "NVIDIA A100-SXM4-40GB",
            "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
            "parameter_changed": True,
            "cpu_fallback_used": False,
            "checkpoint_path": str(checkpoint_path),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        config = (
            {"training": {"screen_seed": seed}}
            if family == "critic_screen"
            else {"training_seed": seed}
        )
        if critic_evidence_version == "v2":
            update_budget = 22_416
            runner_head_key = {
                "SCREEN": "runner_git_head",
                "CONFIRMATION": "confirmation_runner_git_head",
                "REFIT": "posttest_runner_git_head",
                "LOSO": "posttest_runner_git_head",
            }[stage]
            config[runner_head_key] = head
            config["data_geometry"] = {
                "total_optimizer_updates": update_budget
            }
            initialization_scope = (
                "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
                if run_id == "c0_v4"
                else (
                    "NOT_CLAIMED_PARAMETER_MATCHED_DIFFERENT_MODULE"
                    if run_id == "v4_no_cross"
                    else "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
                )
            )
            summary.update(
                {
                    "parameter_initialization_seed": seed,
                    "parameter_initialization_seed_applied_before_model_construction": True,
                    "parameter_initialization_tensor_identity_scope": initialization_scope,
                    "cuda_available": True,
                    "cuda_device": f"cuda:{gpu}",
                    "a100_device_verified": True,
                    "bf16_supported": True,
                    "training_git_head": head,
                    "output_directory": str(output),
                    "training_summary_path": str(summary_path),
                    "training_attempt_path": str(
                        output / "training_attempt.json"
                    ),
                    "update_count": update_budget,
                }
            )
        if held_out_study is not None:
            config["held_out_study"] = held_out_study
            summary["held_out_study"] = held_out_study
        attempt_seed: int | str = seed
        checkpoint_paths = [checkpoint_path]
    elif family.startswith("setflow_"):
        trainer = "train_route2_xeditsetflow_v4.py"
        stage = family.removeprefix("setflow_").upper()
        summary_path = output / "training_summary.json"
        saved = {str(value): str(output / f"pass_{value}.pt") for value in (4, 6, 8, 10)}
        summary = {
            "schema_version": "route_a_v3_route2_xeditsetflow_v4_training_summary.v1",
            "status": "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION",
            "run_stage": stage,
            "run_id": run_id,
            "seed": seed,
            "physical_gpu_index": gpu,
            "torch_device": f"cuda:{gpu}",
            "training_precision": "BF16",
            "parameter_changed": True,
            "cpu_fallback_used": False,
            "saved_checkpoint_paths": saved,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        config = (
            {"training": {"screen_seed": seed}}
            if family == "setflow_screen"
            else {"training_seed": seed}
        )
        attempt_seed = seed
        checkpoint_paths = [Path(path) for path in saved.values()]
    else:
        trainer = "train_route2_xeditflow_value_v4.py"
        summary_path = output / "run_summary.json"
        checkpoint_path = output / "value_checkpoint.pt"
        summary = {
            "schema_version": "route_a_v3_route2_xeditflow_value_training.v4",
            "status": "XEDITFLOW_V4_VALUE_TRAINING_COMPLETE",
            "base_flow_training_seed": seed,
            "training_precision": "BF16",
            "parameter_changed": True,
            "cpu_fallback_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        config = {"base_flow_training_seed": seed}
        # The current value trainer records base_flow_training_seed in its config
        # and summary, while the generic per-run ledger seed column is blank.
        attempt_seed = ""
        checkpoint_paths = [checkpoint_path]
    config.update({"physical_gpu_index": gpu, "device": f"cuda:{gpu}"})
    attempt = {
        "attempt_id": attempt_id or f"attempt::{family}::{index:02d}",
        "status": "COMPLETED",
        "started_at": "2026-08-27T10:00:00+0800",
        "completed_at": "2026-08-27T10:10:00+0800",
        "updated_at": "2026-08-27T10:10:00+0800",
        "code_commit": head,
        "seed": attempt_seed,
        "physical_gpu_index": gpu,
        "device": f"cuda:{gpu}",
        "training_precision": "BF16",
        "output_directory": str(output),
        "evaluation_record_count": 0,
    }
    if family.startswith("critic_") and critic_evidence_version == "v2":
        initialization_scope = summary[
            "parameter_initialization_tensor_identity_scope"
        ]
        attempt.update(
            {
                "baseline_id": f"xeditcritic_v4_{run_id}_seed{seed}",
                "parameter_initialization_seed": seed,
                "parameter_initialization_seed_applied_before_model_construction": True,
                "parameter_initialization_tensor_identity_scope": initialization_scope,
                "cuda_available": True,
                "cuda_device_name": "NVIDIA A100-SXM4-40GB",
                "a100_device_verified": True,
                "bf16_supported": True,
                "cpu_fallback_used": False,
                "training_git_head": head,
                "training_summary_path": str(summary_path),
                "checkpoint_path": str(checkpoint_path),
                "training_attempt_path": str(output / "training_attempt.json"),
                "optimizer_steps": update_budget,
            }
        )
        summary["checkpoint_path"] = str(checkpoint_path)
    _write_json(config_path, config)
    _write_json(output / "training_attempt.json", attempt)
    _write_json(summary_path, summary)
    for path in checkpoint_paths:
        _touch(path)
    _touch(log_path)
    command = [
        "/python",
        f"/repo/scripts/route_a_v3/{trainer}",
        "--config",
        str(config_path),
    ]
    if family.startswith(("critic_", "setflow_")):
        command.extend(
            ["--run-id", run_id, "--physical-gpu-index", str(gpu)]
        )
        if family.startswith("setflow_"):
            command.extend(["--output-dir", str(output)])
    else:
        command.extend(["--output-dir", str(output)])
    return {
        "family": family,
        "job_key": job_key,
        "seed": seed,
        "gpu": gpu,
        "run_id": run_id,
        "output": output,
        "config": config_path,
        "attempt": output / "training_attempt.json",
        "summary": summary_path,
        "checkpoint": checkpoint_paths[-1],
        "log": log_path,
        "failure": failure_path,
        "command": command,
        "attempt_id": attempt["attempt_id"],
        "critic_evidence_version": critic_evidence_version,
        "held_out_study": held_out_study,
    }


def _queued(rows: list[dict[str, Any]], builder: Callable[[dict[str, Any]], dict[str, Any]]):
    queues: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        queues.setdefault(row["gpu"], []).append(builder(row))
    return [
        {"physical_gpu_index": gpu, "jobs": jobs}
        for gpu, jobs in sorted(queues.items())
    ]


def _final_nontraining_job(root: Path, key: str, *, gpu: int | None = None) -> dict[str, Any]:
    safe = key.replace(":", "_")
    success = root / "final" / "success" / f"{safe}.json"
    failure = root / "final" / "failures" / f"{safe}.failed.json"
    log = root / "final" / "logs" / f"{safe}.log"
    _write_json(success, {"status": "TERMINAL_SYNTHETIC_DEPENDENCY"})
    _touch(log)
    return {
        "job_key": key,
        "command": ["/python", f"/repo/{safe}.py"],
        "physical_gpu_indices": [] if gpu is None else [gpu],
        "success_path": str(success),
        "failure_path": str(failure),
        "log_path": str(log),
    }


def _make_package(tmp_path: Path, *, gate_status: str = "XEDITFLOW_V4_PASS") -> dict[str, Any]:
    head = "a" * 40
    c0_head = HISTORICAL_C0_HEAD
    critic_training_head = HISTORICAL_FULL_HEAD
    paths = {
        "critic_screen_v402": tmp_path / "schedules" / "critic_v402.json",
        "critic_screen_full": tmp_path / "schedules" / "critic_v403_full.json",
        "critic_screen_controls": (
            tmp_path / "schedules" / "critic_v403_controls.json"
        ),
        "setflow_screen": tmp_path / "schedules" / "screen.json",
        "critic_confirmation": tmp_path / "schedules" / "confirmation.json",
        "setflow_confirmation": (
            tmp_path / "schedules" / "setflow_recovered_confirmation.json"
        ),
        "refit": tmp_path / "schedules" / "refit.json",
        "loso": tmp_path / "schedules" / "loso.json",
        "guidance": tmp_path / "schedules" / "guidance.json",
        "final": tmp_path / "schedules" / "final.json",
    }
    family_counts = {
        "critic_screen": 8,
        "critic_confirmation": 6,
        "critic_refit": 3,
        "critic_loso": 42,
        "setflow_screen": 2,
        "setflow_confirmation": 3,
        "guidance_value": 6,
        "final_value": 2,
    }
    artifacts: dict[str, list[dict[str, Any]]] = {}
    ordinal = 0

    critic_run_ids = (
        "c0_v4",
        "v4_full",
        "v4_source_only",
        "v4_edit_metadata_only",
        "v4_no_candidate_sequence",
        "v4_candidate_bundle_permutation",
        "v4_no_cross",
        "v4_no_moe",
    )
    critic_screen = []
    for index, run_id in enumerate(critic_run_ids):
        ordinal += 1
        row_head = (
            c0_head if index == 0 else critic_training_head if index == 1 else head
        )
        gpu = 5 if index == 0 else (3 if index == 1 else index - 2)
        attempt_id = (
            None
            if index == 0
            else f"xeditcritic_v4_screen::{run_id}::{row_head}"
        )
        row = _training_artifact(
            tmp_path,
            family="critic_screen",
            index=ordinal,
            gpu=gpu,
            head=row_head,
            job_key=f"critic:{run_id}",
            seed=20260907,
            run_id=run_id,
            attempt_id=attempt_id,
            critic_evidence_version="v1" if index < 2 else "v2",
        )
        if attempt_id is not None:
            row["command"].extend(["--training-attempt-id", attempt_id])
        critic_screen.append(row)
    artifacts["critic_screen"] = critic_screen

    setflow_screen_specs = (
        ("setflow:v4_full", "v4_full"),
        ("setflow:v4_single_mode", "v4_single_mode"),
    )
    artifacts["setflow_screen"] = []
    for index, (job_key, run_id) in enumerate(setflow_screen_specs):
        ordinal += 1
        row = _training_artifact(
            tmp_path,
            family="setflow_screen",
            index=ordinal,
            gpu=index,
            head=head,
            job_key=job_key,
            seed=20260911,
            run_id=run_id,
        )
        row["command"] = row["command"][:-2] + [
            "--authorization",
            str(tmp_path / "authorizations" / "setflow.json"),
        ]
        artifacts["setflow_screen"].append(row)

    artifacts["critic_confirmation"] = []
    confirmation_specs = [
        (seed, run_id)
        for seed in (20260908, 20260909, 20260910)
        for run_id in ("v4_full", "c0_v4")
    ]
    for index, (seed, run_id) in enumerate(confirmation_specs):
        ordinal += 1
        artifacts["critic_confirmation"].append(
            _training_artifact(
                tmp_path,
                family="critic_confirmation",
                index=ordinal,
                gpu=index,
                head=head,
                job_key=f"critic:{seed}:{run_id}",
                seed=seed,
                run_id=run_id,
            )
        )

    artifacts["setflow_confirmation"] = []
    for index, seed in enumerate((20260921, 20260922, 20260923)):
        ordinal += 1
        artifacts["setflow_confirmation"].append(
            _training_artifact(
                tmp_path,
                family="setflow_confirmation",
                index=ordinal,
                gpu=index,
                head=head,
                job_key=f"setflow:{seed}:v4_full",
                seed=seed,
                run_id="v4_full",
            )
        )

    artifacts["critic_refit"] = []
    for index, seed in enumerate((20260908, 20260909, 20260910)):
        ordinal += 1
        artifacts["critic_refit"].append(
            _training_artifact(
                tmp_path,
                family="critic_refit",
                index=ordinal,
                gpu=index,
                head=head,
                job_key=f"critic_refit:{seed}:v4_full",
                seed=seed,
                run_id="v4_full",
            )
        )

    loso_studies = (
        "GSE200304",
        "GSE114002",
        "GSE149487",
        "GSE217518",
        "GSE186455",
        "GSE256185",
        "GSE269595",
    )
    artifacts["critic_loso"] = []
    loso_specs = [
        (seed, study, run_id)
        for seed in (20260908, 20260909, 20260910)
        for study in loso_studies
        for run_id in ("v4_full", "c0_v4")
    ]
    for index, (seed, study, run_id) in enumerate(loso_specs):
        ordinal += 1
        artifacts["critic_loso"].append(
            _training_artifact(
                tmp_path,
                family="critic_loso",
                index=ordinal,
                gpu=index % 6,
                head=head,
                job_key=f"critic_loso:{seed}:{study}:{run_id}",
                seed=seed,
                run_id=run_id,
                held_out_study=study,
            )
        )

    for family in ("guidance_value", "final_value"):
        count = family_counts[family]
        rows = []
        for index in range(count):
            ordinal += 1
            key = None
            seed = None
            if family == "final_value":
                seed = (20260913, 20260914)[index]
                key = f"seed_{seed}:value_training"
            rows.append(
                _training_artifact(
                    tmp_path,
                    family=family,
                    index=ordinal,
                    gpu=index % 6,
                    head=head,
                    job_key=key,
                    seed=seed,
                )
            )
        artifacts[family] = rows

    v402_jobs = []
    for index, run_id in enumerate(critic_run_ids):
        if index == 0:
            row = artifacts["critic_screen"][0]
            output = row["output"]
            log = row["log"]
            command = row["command"]
        else:
            output = tmp_path / "historical_v402_planned" / run_id
            log = tmp_path / "historical_v402_logs" / f"{run_id}.log"
            command = [
                "/python",
                "/repo/scripts/route_a_v3/train_route2_xeditcritic_v4.py",
                "--config",
                str(artifacts["critic_screen"][0]["config"]),
                "--run-id",
                run_id,
                "--physical-gpu-index",
                "5",
            ]
        v402_jobs.append(
            {
                "job_key": f"critic:{run_id}",
                "run_id": run_id,
                "output_directory": str(output),
                "log_path": str(log),
                "command": command,
            }
        )
    _write_json(
        paths["critic_screen_v402"],
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v402_recovery_schedule.v1"
            ),
            "status": "FROZEN_V402_RECOVERY_SCHEDULE",
            "git_head": c0_head,
            "physical_gpu_index": 5,
            "gpu5_free_memory_mib_before_launch": 40_000,
            "required_free_memory_bytes": 30 * 1024**3,
            "required_free_memory_rule": "TRAIN_ONLY_SMOKE_PEAK_PLUS_2_GIB",
            "jobs": v402_jobs,
            "terminal_artifact_payloads_read": 0,
            "active_performance_output_read": False,
            "setflow_jobs_stopped_modified_or_restarted": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    full = artifacts["critic_screen"][1]
    _write_json(
        paths["critic_screen_full"],
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_full_recovery_schedule.v1"
            ),
            "status": "XEDITCRITIC_V403_FULL_RECOVERY_SCHEDULED",
            "git_head": critic_training_head,
            "run_id": "v4_full",
            "physical_gpu_index": full["gpu"],
            "output_directory": str(full["output"]),
            "runtime_manifest": str(tmp_path / "runtime" / "critic_full.json"),
            "log_path": str(full["log"]),
            "screen_config": str(full["config"]),
            "free_memory_gate_applied": False,
            "command": full["command"],
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    controls = artifacts["critic_screen"][2:]
    control_config = tmp_path / "configs" / "critic_controls_screen.json"
    _write_json(
        control_config,
        {
            "training": {"screen_seed": 20260907},
            "runner_git_head": head,
            "data_geometry": {"total_optimizer_updates": 22_416},
        },
    )
    _write_json(
        paths["critic_screen_controls"],
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_control_recovery_schedule.v1"
            ),
            "status": "XEDITCRITIC_V403_CONTROL_RECOVERY_SCHEDULED",
            "orchestration_git_head": head,
            "current_git_head": head,
            "runner_git_head": head,
            "training_code_git_head": head,
            "historical_full_git_head": critic_training_head,
            "historical_c0_git_head": c0_head,
            "training_worktree": "/home/synthetic/current_training_worktree",
            "screen_config": str(control_config),
            "cuda_bf16_inventory": [
                {
                    "physical_gpu_index": gpu,
                    "bf16_supported": True,
                    "bf16_tensor_probe": True,
                    "cpu_fallback_used": False,
                }
                for gpu in range(6)
            ],
            "jobs": [
                {
                    "run_id": row["run_id"],
                    "physical_gpu_index": row["gpu"],
                    "output_directory": str(row["output"]),
                    "log_path": str(row["log"]),
                    "training_attempt_id": row["attempt_id"],
                    "command": (
                        row["command"][:2]
                        + ["--config", str(control_config)]
                        + row["command"][4:]
                    ),
                }
                for row in controls
            ],
            "full_retrained": False,
            "c0_retrained": False,
            "old_v402_stopped_process_resumed": False,
            "free_memory_gate_applied": False,
            "terminal_artifact_payloads_read_by_scheduler": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    screen_dummies = [
        {
            "gpu": index % 6,
            "job_key": f"critic:{run_id}",
            "component": "critic",
            "run_id": run_id,
            "output": tmp_path / "historical_original_screen" / "critic" / run_id,
            "log": tmp_path / "historical_original_screen_logs" / f"critic_{run_id}.log",
            "command": [
                "/python",
                "/repo/scripts/route_a_v3/train_route2_xeditcritic_v4.py",
                "--config",
                str(artifacts["critic_screen"][0]["config"]),
                "--run-id",
                run_id,
                "--physical-gpu-index",
                str(index % 6),
                "--launch-authorization",
                str(tmp_path / "authorizations" / "critic.json"),
            ],
        }
        for index, run_id in enumerate(critic_run_ids)
    ]
    setflow_screen_rows = [
        {**row, "component": "setflow"} for row in artifacts["setflow_screen"]
    ]
    _write_json(
        paths["setflow_screen"],
        {
            "schema_version": "route_a_v3_route2_xedit_v4_screen_package_schedule.v1",
            "status": "FROZEN_SCREEN_PACKAGE_SCHEDULE",
            "git_head": head,
            "experiment_head": "c" * 40,
            "preflight_runner_git_head": "d" * 40,
            "worktree": "/home/synthetic/original_screen_worktree",
            "runtime_manifest": str(
                tmp_path / "historical_original_screen" / "runtime.json"
            ),
            "gpu_free_memory_mib_before_launch": {
                str(gpu): 40_000 for gpu in range(6)
            },
            "critic_required_free_memory_mib": 35_000,
            "setflow_required_free_memory_mib": 20_000,
            "gpu_assignment_policy": (
                "ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB"
            ),
            "active_performance_output_read": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
            "gpu_queues": _queued(
                screen_dummies + setflow_screen_rows,
                lambda row: (
                    {
                        "job_key": row["job_key"],
                        "component": "critic",
                        "run_id": row["run_id"],
                        "output_directory": str(row["output"]),
                        "log_path": str(row["log"]),
                        "command": row["command"],
                    }
                    if row["component"] == "critic"
                    else {
                        "job_key": row["job_key"],
                        "component": "setflow",
                        "run_id": row["run_id"],
                        "output_directory": str(row["output"]),
                        "log_path": str(row["log"]),
                        "command": row["command"],
                    }
                ),
            ),
        },
    )

    _write_json(
        paths["critic_confirmation"],
        {
            "schema_version": "route_a_v3_route2_xedit_v4_confirmation_training_schedule.v1",
            "status": "FROZEN_CONFIRMATION_TRAINING_SCHEDULE",
            "git_head": head,
            "eligible_components": ["critic"],
            "free_memory_gate_applied": False,
            "active_performance_output_read": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
            "gpu_queues": _queued(
                artifacts["critic_confirmation"],
                lambda row: {
                    "job_key": row["job_key"],
                    "component": "critic",
                    "training_seed": row["seed"],
                    "run_id": row["run_id"],
                    "output_directory": str(row["output"]),
                    "log_path": str(row["log"]),
                    "command": row["command"],
                },
            ),
        },
    )

    setflow_confirmation = artifacts["setflow_confirmation"]
    _write_json(
        paths["setflow_confirmation"],
        {
            "schema_version": (
                "route_a_v3_route2_xeditsetflow_v403_"
                "recovered_confirmation_training_schedule.v1"
            ),
            "status": "FROZEN_RECOVERY_DERIVED_CONFIRMATION_TRAINING_SCHEDULE",
            "git_head": head,
            "experiment_head": "c" * 40,
            "training_git_head": "d" * 40,
            "validation_git_head": "e" * 40,
            "eligible_components": ["setflow"],
            "required_seeds": [row["seed"] for row in setflow_confirmation],
            "free_memory_gate_applied": False,
            "cuda_bf16_probes": {
                str(row["gpu"]): {
                    "physical_gpu_index": row["gpu"],
                    "device_type": "cuda",
                    "cuda_available": True,
                    "bf16_supported": True,
                    "cpu_fallback_used": False,
                }
                for row in setflow_confirmation
            },
            "gpu_queues": _queued(
                setflow_confirmation,
                lambda row: {
                    "job_key": row["job_key"],
                    "component": "setflow",
                    "training_seed": row["seed"],
                    "run_id": "v4_full",
                    "output_directory": str(row["output"]),
                    "log_path": str(row["log"]),
                    "command": row["command"],
                },
            ),
            "training_reused_from_screen": False,
            "screen_training_reused_by_recovery": True,
            "recovery_parameter_update_count": 0,
            "active_performance_output_read": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    for family, schema, status, access_key, path_key in (
        (
            "critic_refit",
            "route_a_v3_route2_xeditcritic_v4_refit_schedule.v1",
            "FROZEN_EXACT_THREE_REFIT_SCHEDULE",
            "development_test_access_event_count_before_refit",
            "refit",
        ),
        (
            "critic_loso",
            "route_a_v3_route2_xeditcritic_v4_loso_schedule.v1",
            "FROZEN_EXACT_42_JOB_LOSO_SCHEDULE",
            "development_test_access_event_count_before_loso",
            "loso",
        ),
    ):
        payload = {
            "schema_version": schema,
            "status": status,
            "git_head": head,
            "free_memory_gate_applied": False,
            "active_performance_output_read": False,
            access_key: 1,
            (
                "development_test_outcome_reads_during_refit"
                if family == "critic_refit"
                else "development_test_outcome_reads_during_loso"
            ): 0,
            "new_final_evaluation_outcome_reads": 0,
            "gpu_queues": _queued(
                artifacts[family],
                lambda row: {
                    "job_key": row["job_key"],
                    "seed": row["seed"],
                    "run_id": row["run_id"],
                    **(
                        {"held_out_study": row["held_out_study"]}
                        if family == "critic_loso"
                        else {}
                    ),
                    "summary_path": str(row["summary"]),
                    "failure_path": str(row["failure"]),
                    "log_path": str(row["log"]),
                    "command": row["command"],
                },
            ),
        }
        _write_json(paths[path_key], payload)

    _write_json(
        paths["guidance"],
        {
            "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_schedule.v1",
            "status": "FROZEN_VALUE_AND_EXACT_18_COMBINATION_SCHEDULE",
            "git_head": head,
            "free_memory_gate_applied": False,
            "active_performance_output_read": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
            "value_training_queues": _queued(
                artifacts["guidance_value"],
                lambda row: {
                    "job_key": row["job_key"],
                    "success_path": str(row["summary"]),
                    "failure_path": str(row["failure"]),
                    "log_path": str(row["log"]),
                    "command": row["command"],
                },
            ),
        },
    )

    final_training = artifacts["final_value"]
    first_prerequisite = [
        _final_nontraining_job(tmp_path, "seed_20260913:value_rollout", gpu=0),
        _final_nontraining_job(tmp_path, "seed_20260913:value_critic_score", gpu=4),
        _final_nontraining_job(tmp_path, "seed_20260913:value_target"),
    ]
    second_prerequisite = [
        _final_nontraining_job(tmp_path, "seed_20260914:value_rollout", gpu=1),
        _final_nontraining_job(tmp_path, "seed_20260914:value_critic_score", gpu=5),
        _final_nontraining_job(tmp_path, "seed_20260914:value_target"),
    ]

    def final_training_job(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_key": row["job_key"],
            "command": row["command"],
            "physical_gpu_indices": [row["gpu"]],
            "success_path": str(row["summary"]),
            "failure_path": str(row["failure"]),
            "log_path": str(row["log"]),
        }

    first_prerequisite.append(final_training_job(final_training[0]))
    second_prerequisite.append(final_training_job(final_training[1]))
    strongest_timing = _final_nontraining_job(tmp_path, "strongest_timing", gpu=2)
    seed_chains = []
    for seed in (20260912, 20260913, 20260914):
        jobs = []
        labels = (
            ["strongest_adapter"]
            + [f"generation:method_{index}" for index in range(5)]
            + [f"terminal_critic:method_{index}" for index in range(5)]
            + [f"open:method_{index}" for index in range(5)]
            + [f"closed_exact:method_{index}" for index in range(2)]
            + [f"closed_score:method_{index}" for index in range(3)]
            + [f"closed_metric:method_{index}" for index in range(4)]
            + ["independent_evaluator", "independent_evaluator_comparison", "equal_wall_time", "final_evidence"]
        )
        assert len(labels) == 29
        for index, label in enumerate(labels):
            jobs.append(
                _final_nontraining_job(
                    tmp_path,
                    f"seed_{seed}:{label}",
                    gpu=index % 6 if any(value in label for value in ("generation", "critic", "closed_score", "evaluator")) else None,
                )
            )
        seed_chains.append({"queue_key": f"seed_{seed}", "jobs": jobs})
    compose = _final_nontraining_job(tmp_path, "compose_final_comparison")
    adjudication_path = tmp_path / "final" / "final_adjudication.json"
    adjudication_log = tmp_path / "final" / "logs" / "adjudicate_final_comparison.log"
    adjudication_failure = tmp_path / "final" / "failures" / "adjudicate_final_comparison.failed.json"
    _touch(adjudication_log)
    authorized = gate_status == "XEDITFLOW_V4_PASS"
    adjudication = {
        "schema_version": "route_a_v3_route2_xeditflow_final_adjudication.v4",
        "status": "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL",
        "gate": {
            "status": gate_status,
            "new_final_evaluation_authorized": authorized,
        },
        "new_final_evaluation_authorized": authorized,
        "additional_training_seed_authorized": False,
        "submission_ready": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    _write_json(adjudication_path, adjudication)
    adjudicate_job = {
        "job_key": "adjudicate_final_comparison",
        "command": ["/python", "/repo/adjudicate.py"],
        "physical_gpu_indices": [],
        "success_path": str(adjudication_path),
        "failure_path": str(adjudication_failure),
        "log_path": str(adjudication_log),
    }
    runtime_path = tmp_path / "final" / "runtime.json"
    final_schedule = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_final_schedule.v1",
        "status": "FROZEN_THREE_SEED_MATCHED_COMPUTE_SCHEDULE",
        "git_head": head,
        "runtime_manifest": str(runtime_path),
        "free_memory_gate_applied": False,
        "active_performance_output_read": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
        "prerequisite_queues": [
            {"queue_key": "value_seed_20260913", "jobs": first_prerequisite},
            {"queue_key": "value_seed_20260914", "jobs": second_prerequisite},
            {"queue_key": "strongest_timing", "jobs": [strongest_timing]},
        ],
        "seed_chains": seed_chains,
        "finalization_jobs": [compose, adjudicate_job],
    }
    _write_json(paths["final"], final_schedule)
    launch_path = tmp_path / "final" / "launch.json"
    _write_json(
        launch_path,
        {
            "schema_version": "route_a_v3_route2_xeditflow_v4_final_launch.v1",
            "status": "XEDITFLOW_V4_FINAL_SCHEDULER_LAUNCHED",
            "git_head": head,
            "schedule_path": str(paths["final"]),
            "runtime_manifest": str(runtime_path),
            "final_adjudication_path": str(adjudication_path),
            "development_test_reopened": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    all_final_jobs = [
        job
        for queue in final_schedule["prerequisite_queues"]
        for job in queue["jobs"]
    ] + [job for chain in seed_chains for job in chain["jobs"]] + final_schedule["finalization_jobs"]
    assert len(all_final_jobs) == 98
    runtime = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_final_runtime.v1",
        "status": "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL",
        "git_head": head,
        "jobs": {
            job["job_key"]: {
                "status": "TERMINAL_COMPLETE",
                "terminal_artifact_kind": "SUCCESS",
                "return_code": 0,
                "success_path": job["success_path"],
                "failure_path": job["failure_path"],
                "log_path": job["log_path"],
            }
            for job in all_final_jobs
        },
        "first_terminal_failure": None,
        "active_performance_output_read": False,
        "development_test_reopened": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    _write_json(runtime_path, runtime)

    private_atomic_test = tmp_path / "private" / "atomic_test_outcomes.json"
    _write_json(private_atomic_test, {"private_metric": 0.999})
    receipt_path = tmp_path / "atomic" / "posttest_authorization_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
            "status": "XEDITCRITIC_V4_POSTTEST_AUTHORIZED",
            "frozen_test_gate_status": "XEDITCRITIC_V4_FROZEN_TEST_PASS",
            "all_development_refit_authorized": True,
            "development_test_access_event_count": 1,
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
            "development_test_metrics_in_receipt": False,
            "new_final_evaluation_outcomes_accessed": False,
            "private_atomic_test_path": str(private_atomic_test),
        },
    )
    family_schedule = {
        "setflow_screen": paths["setflow_screen"],
        "critic_confirmation": paths["critic_confirmation"],
        "setflow_confirmation": paths["setflow_confirmation"],
        "critic_refit": paths["refit"],
        "critic_loso": paths["loso"],
        "guidance_value": paths["guidance"],
        "final_value": paths["final"],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_rows = []
    for family in family_counts:
        for index, row in enumerate(artifacts[family]):
            if family == "critic_screen":
                schedule_path = (
                    paths["critic_screen_v402"]
                    if index == 0
                    else (
                        paths["critic_screen_full"]
                        if index == 1
                        else paths["critic_screen_controls"]
                    )
                )
            else:
                schedule_path = family_schedule[family]
            inventory_rows.append(
                {
                    "family": family,
                    "schedule_path": str(schedule_path),
                    "job_key": row["job_key"],
                }
            )
    _write_json(
        inventory_path,
        {
            "schema_version": "route_a_v3_route2_xeditflow_v4_terminal_training_inventory.v1",
            "status": "EXPLICIT_COMPLETE_72_PARAMETER_UPDATING_ATTEMPTS",
            "attempts": inventory_rows,
        },
    )
    return {
        "head": head,
        "paths": paths,
        "artifacts": artifacts,
        "runtime": runtime_path,
        "launch": launch_path,
        "adjudication": adjudication_path,
        "receipt": receipt_path,
        "private_atomic_test": private_atomic_test,
        "inventory": inventory_path,
        "output": tmp_path / "terminal_training_ledger.json",
    }


def _export(module, package: dict[str, Any]):
    return module.export_terminal_training_ledger(
        final_launch_path=package["launch"],
        final_schedule_path=package["paths"]["final"],
        final_runtime_path=package["runtime"],
        final_adjudication_path=package["adjudication"],
        training_inventory_path=package["inventory"],
        posttest_authorization_receipt_path=package["receipt"],
        output_path=package["output"],
    )


@pytest.mark.parametrize(
    ("gate_status", "excellent"),
    (("XEDITFLOW_V4_PASS", True), ("XEDITFLOW_V4_NO_GO", False)),
)
def test_exports_exact_72_attempt_closure_for_pass_and_no_go(
    tmp_path: Path, gate_status: str, excellent: bool
) -> None:
    module = _load_module()
    package = _make_package(tmp_path, gate_status=gate_status)
    result = _export(module, package)
    assert result["parameter_updating_attempt_count"] == 72
    assert result["family_counts"] == {
        "critic_screen": 8,
        "critic_confirmation": 6,
        "critic_refit": 3,
        "critic_loso": 42,
        "setflow_screen": 2,
        "setflow_confirmation": 3,
        "guidance_value": 6,
        "final_value": 2,
    }
    assert len(result["training_attempts"]) == 72
    assert result["excellent_development_result"] is excellent
    assert result["submission_ready"] is False
    assert result["stage_barriers"]["atomic_frozen_development_test"]["development_test_access_event_count"] == 1
    assert result["stage_barriers"]["validation"]["training_attempt_row_count"] == 0
    assert result["frozen_dependencies"]["final_schedule_non_parameter_updating_job_count"] == 96
    assert _read_json(package["output"]) == result


def test_exports_real_recovered_multischedule_lineage(tmp_path: Path) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    result = _export(module, package)

    critic_screen = [
        row for row in result["training_attempts"] if row["family"] == "critic_screen"
    ]
    assert [
        Path(row["schedule_path"]).name for row in critic_screen
    ] == [
        "critic_v402.json",
        "critic_v403_full.json",
        "critic_v403_controls.json",
        "critic_v403_controls.json",
        "critic_v403_controls.json",
        "critic_v403_controls.json",
        "critic_v403_controls.json",
        "critic_v403_controls.json",
    ]
    assert [row["code_commit"] for row in critic_screen] == [
        HISTORICAL_C0_HEAD,
        HISTORICAL_FULL_HEAD,
        *("a" * 40 for _ in range(6)),
    ]
    assert [
        row["terminal_evidence_schema_version"] for row in critic_screen
    ] == [
        "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
        "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
        *(
            "route_a_v3_route2_xeditcritic_v4_screen_run.v2"
            for _ in range(6)
        ),
    ]
    assert sum(
        row["historical_free_memory_gate_applied"] for row in critic_screen
    ) == 1
    assert critic_screen[0]["historical_free_memory_gate_policy"] == (
        "TRAIN_ONLY_SMOKE_PEAK_PLUS_2_GIB"
    )

    original_screen_schedule = _read_json(package["paths"]["setflow_screen"])
    assert "free_memory_gate_applied" not in original_screen_schedule
    assert original_screen_schedule["gpu_assignment_policy"] == (
        "ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB"
    )
    setflow_screen = [
        row for row in result["training_attempts"] if row["family"] == "setflow_screen"
    ]
    assert [row["schedule_job_key"] for row in setflow_screen] == [
        "setflow:v4_full",
        "setflow:v4_single_mode",
    ]
    assert all(row["historical_free_memory_gate_applied"] for row in setflow_screen)
    assert {
        row["historical_free_memory_gate_policy"] for row in setflow_screen
    } == {"ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB"}

    critic_confirmation = [
        row
        for row in result["training_attempts"]
        if row["family"] == "critic_confirmation"
    ]
    setflow_confirmation = [
        row
        for row in result["training_attempts"]
        if row["family"] == "setflow_confirmation"
    ]
    assert {Path(row["schedule_path"]).name for row in critic_confirmation} == {
        "confirmation.json"
    }
    assert {Path(row["schedule_path"]).name for row in setflow_confirmation} == {
        "setflow_recovered_confirmation.json"
    }
    assert {row["code_commit"] for row in setflow_confirmation} == {"a" * 40}
    for family in ("critic_confirmation", "critic_refit", "critic_loso"):
        rows = [
            row
            for row in result["training_attempts"]
            if row["family"] == family
        ]
        stage = family.removeprefix("critic_")
        assert {
            row["terminal_evidence_schema_version"] for row in rows
        } == {f"route_a_v3_route2_xeditcritic_v4_{stage}_run.v2"}


def test_running_final_runtime_reads_no_inventory_summary_receipt_or_adjudication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    _mutate(
        package["runtime"],
        lambda value: value.update(
            status="XEDITFLOW_V4_FINAL_COMPARISON_RUNNING"
        ),
    )
    original = module._read_json
    reads: list[Path] = []
    forbidden = {
        package["inventory"],
        package["receipt"],
        package["adjudication"],
    }
    forbidden.update(
        row[field]
        for rows in package["artifacts"].values()
        for row in rows
        for field in ("config", "attempt", "summary")
    )

    def guarded(path: Path, label: str):
        reads.append(path)
        if path in forbidden:
            raise AssertionError(f"premature historical read: {path}")
        return original(path, label)

    monkeypatch.setattr(module, "_read_json", guarded)
    with pytest.raises(
        module.XEditFlowV4TerminalTrainingLedgerError,
        match="final runtime is not exact terminal",
    ):
        _export(module, package)
    assert reads == [
        package["paths"]["final"],
        package["launch"],
        package["runtime"],
    ]
    assert not package["output"].exists()


@pytest.mark.parametrize("lineage", ("controls", "recovered_setflow"))
def test_rejects_training_commit_from_wrong_lineage_head(
    tmp_path: Path, lineage: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    if lineage == "controls":
        target = package["artifacts"]["critic_screen"][2]["attempt"]
        _mutate(target, lambda value: value.update(code_commit="b" * 40))
    else:
        target = package["artifacts"]["setflow_confirmation"][0]["attempt"]
        _mutate(target, lambda value: value.update(code_commit="d" * 40))
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert not package["output"].exists()


def test_rejects_v1_for_not_yet_launched_critic_controls(
    tmp_path: Path,
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    control = package["artifacts"]["critic_screen"][2]
    _mutate(
        control["summary"],
        lambda value: value.update(
            schema_version=(
                "route_a_v3_route2_xeditcritic_v4_screen_run.v1"
            )
        ),
    )
    with pytest.raises(
        module.XEditFlowV4TerminalTrainingLedgerError,
        match="terminal summary differs",
    ):
        _export(module, package)
    assert not package["output"].exists()


@pytest.mark.parametrize(
    "drift",
    (
        "summary_initialization_seed",
        "attempt_before_model",
        "summary_initialization_scope",
        "summary_cuda_device",
        "summary_a100",
        "attempt_bf16",
        "summary_cpu_fallback",
        "summary_training_head",
        "config_runner_head",
        "summary_run_id",
        "summary_output_path",
        "summary_summary_path",
        "summary_checkpoint_path",
        "summary_attempt_path",
        "attempt_output_path",
        "attempt_summary_path",
        "attempt_checkpoint_path",
        "attempt_attempt_path",
        "summary_update_count",
        "attempt_optimizer_steps",
        "config_update_budget_missing",
    ),
)
def test_rejects_critic_v2_terminal_evidence_drift(
    tmp_path: Path, drift: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    target = package["artifacts"]["critic_confirmation"][0]
    wrong_path = str(tmp_path / "wrong" / "artifact")
    if drift == "summary_initialization_seed":
        _mutate(
            target["summary"],
            lambda value: value.update(
                parameter_initialization_seed=20260910
            ),
        )
    elif drift == "attempt_before_model":
        _mutate(
            target["attempt"],
            lambda value: value.update(
                parameter_initialization_seed_applied_before_model_construction=False
            ),
        )
    elif drift == "summary_initialization_scope":
        _mutate(
            target["summary"],
            lambda value: value.update(
                parameter_initialization_tensor_identity_scope="DRIFT"
            ),
        )
    elif drift == "summary_cuda_device":
        _mutate(
            target["summary"],
            lambda value: value.update(cuda_device="cpu"),
        )
    elif drift == "summary_a100":
        _mutate(
            target["summary"],
            lambda value: value.update(a100_device_verified=False),
        )
    elif drift == "attempt_bf16":
        _mutate(
            target["attempt"],
            lambda value: value.update(bf16_supported=False),
        )
    elif drift == "summary_cpu_fallback":
        _mutate(
            target["summary"],
            lambda value: value.update(cpu_fallback_used=True),
        )
    elif drift == "summary_training_head":
        _mutate(
            target["summary"],
            lambda value: value.update(training_git_head="e" * 40),
        )
    elif drift == "config_runner_head":
        _mutate(
            target["config"],
            lambda value: value.update(confirmation_runner_git_head="e" * 40),
        )
    elif drift == "summary_run_id":
        _mutate(
            target["summary"],
            lambda value: value.update(run_id="c0_v4"),
        )
    elif drift == "summary_update_count":
        _mutate(
            target["summary"],
            lambda value: value.update(update_count=22_415),
        )
    elif drift == "attempt_optimizer_steps":
        _mutate(
            target["attempt"],
            lambda value: value.update(optimizer_steps=22_415),
        )
    elif drift == "config_update_budget_missing":
        _mutate(
            target["config"],
            lambda value: value.pop("data_geometry"),
        )
    else:
        payload_name, field = {
            "summary_output_path": ("summary", "output_directory"),
            "summary_summary_path": ("summary", "training_summary_path"),
            "summary_checkpoint_path": ("summary", "checkpoint_path"),
            "summary_attempt_path": ("summary", "training_attempt_path"),
            "attempt_output_path": ("attempt", "output_directory"),
            "attempt_summary_path": ("attempt", "training_summary_path"),
            "attempt_checkpoint_path": ("attempt", "checkpoint_path"),
            "attempt_attempt_path": ("attempt", "training_attempt_path"),
        }[drift]
        _mutate(
            target[payload_name],
            lambda value: value.update({field: wrong_path}),
        )
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert not package["output"].exists()


@pytest.mark.parametrize("failure", ("runtime_status", "missing_runtime_job"))
def test_rejects_nonterminal_or_incomplete_final_runtime(
    tmp_path: Path, failure: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    if failure == "runtime_status":
        _mutate(package["runtime"], lambda value: value.update(status="XEDITFLOW_V4_FINAL_COMPARISON_RUNNING"))
    else:
        _mutate(package["runtime"], lambda value: value["jobs"].pop(next(iter(value["jobs"]))))
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert not package["output"].exists()


@pytest.mark.parametrize("failure", ("duplicate", "missing", "nonterminal"))
def test_rejects_duplicate_missing_or_nonterminal_training_attempt(
    tmp_path: Path, failure: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    if failure in {"duplicate", "missing"}:
        def change(value: dict[str, Any]) -> None:
            if failure == "duplicate":
                value["attempts"][-1] = dict(value["attempts"][0])
            else:
                value["attempts"].pop()
        _mutate(package["inventory"], change)
    else:
        target = package["artifacts"]["critic_screen"][0]["attempt"]
        _mutate(target, lambda value: value.update(status="RUNNING"))
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert not package["output"].exists()


@pytest.mark.parametrize(
    "failure",
    ("cuda_device", "cpu_fallback", "protected_read", "missing_checkpoint"),
)
def test_rejects_cuda_cpu_protected_or_required_path_mismatch(
    tmp_path: Path, failure: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    target = package["artifacts"]["critic_screen"][0]
    if failure == "cuda_device":
        _mutate(target["attempt"], lambda value: value.update(device="cpu"))
    elif failure == "cpu_fallback":
        _mutate(target["summary"], lambda value: value.update(cpu_fallback_used=True))
    elif failure == "protected_read":
        _mutate(
            package["runtime"],
            lambda value: value.update(
                development_test_outcomes_accessed_after_atomic_test=True
            ),
        )
    else:
        target["checkpoint"].unlink()
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert not package["output"].exists()


def test_output_contains_no_hash_checksum_or_fingerprint_fields(tmp_path: Path) -> None:
    module = _load_module()
    result = _export(module, _make_package(tmp_path))

    def keys(value: Any):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    forbidden = ("hash", "checksum", "fingerprint")
    assert not [key for key in keys(result) if any(word in key.lower() for word in forbidden)]


def test_does_not_read_private_log_or_checkpoint_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    original = Path.read_text
    forbidden = {package["private_atomic_test"]}
    forbidden.update(
        row["log"]
        for rows in package["artifacts"].values()
        for row in rows
    )
    forbidden.update(
        row["checkpoint"]
        for rows in package["artifacts"].values()
        for row in rows
    )

    def guarded(path: Path, *args, **kwargs):
        if path in forbidden:
            raise AssertionError(f"forbidden payload read: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    result = _export(module, package)
    assert result["exporter_protected_reads"] == {
        "active_performance_output_reads": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "private_or_outcome_payload_reads": 0,
        "log_payload_reads": 0,
        "checkpoint_payload_reads": 0,
    }


@pytest.mark.parametrize("preexisting", ("output", "partial"))
def test_rejects_existing_output_or_partial_without_overwrite(
    tmp_path: Path, preexisting: str
) -> None:
    module = _load_module()
    package = _make_package(tmp_path)
    path = (
        package["output"]
        if preexisting == "output"
        else package["output"].with_suffix(package["output"].suffix + ".partial")
    )
    _touch(path, "preserve me\n")
    with pytest.raises(module.XEditFlowV4TerminalTrainingLedgerError):
        _export(module, package)
    assert path.read_text(encoding="utf-8") == "preserve me\n"
