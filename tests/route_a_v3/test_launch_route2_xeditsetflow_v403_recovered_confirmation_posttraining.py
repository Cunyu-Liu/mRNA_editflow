from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_v403_recovered_confirmation_posttraining as launcher


TRAINING_RUNNER_HEAD = "a" * 40
ORCHESTRATION_HEAD = "b" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _bindings(tmp_path: Path) -> dict[str, str]:
    return {
        "protocol_path": str(launcher.PROTOCOL),
        "training_git_head": launcher.TRAINING_HEAD,
        "validation_git_head": launcher.VALIDATION_HEAD,
        "runner_git_head": TRAINING_RUNNER_HEAD,
        "recovery_config_path": str(tmp_path / "recovery_config.json"),
        "recovered_screen_gate_path": str(tmp_path / "recovered_gate.json"),
        "confirmation_authorization_path": str(tmp_path / "authorization.json"),
        "config_manifest_path": str(tmp_path / "manifest.json"),
        "training_runtime_path": str(tmp_path / "training_runtime.json"),
        "posttraining_runtime_root": str(tmp_path / "posttraining_runtime"),
        "posttraining_log_root": str(tmp_path / "posttraining_logs"),
        "confirmation_gate_output": str(tmp_path / "confirmation_gate.json"),
    }


def _training_schedule(
    tmp_path: Path,
) -> tuple[dict, dict[int, dict]]:
    bindings = _bindings(tmp_path)
    jobs: dict[int, dict] = {}
    queues = []
    diagnostics = {}
    probes = {}
    for gpu, seed in enumerate(launcher.CONFIRMATION_SEEDS):
        job = {
            "job_key": f"setflow:{seed}:v4_full",
            "component": "setflow",
            "training_seed": seed,
            "run_id": "v4_full",
            "output_directory": str(tmp_path / f"seed_{seed}" / "v4_full"),
            "log_path": str(tmp_path / f"seed_{seed}.log"),
            "command": ["trainer", str(seed)],
        }
        jobs[seed] = {**job, "physical_gpu_index": gpu}
        queues.append({"physical_gpu_index": gpu, "jobs": [job]})
        diagnostics[str(gpu)] = {
            "name": f"gpu-{gpu}",
            "free_memory_mib": gpu + 1,
            "total_memory_mib": 40960,
        }
        probes[str(gpu)] = {
            "physical_gpu_index": gpu,
            "device_type": "cuda",
            "dtype": "BFLOAT16",
            "cuda_available": True,
            "bf16_supported": True,
            "cpu_fallback_used": False,
        }
    schedule = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_training_schedule.v1"
        ),
        "status": "FROZEN_RECOVERY_DERIVED_CONFIRMATION_TRAINING_SCHEDULE",
        "git_head": TRAINING_RUNNER_HEAD,
        "experiment_head": launcher.SCREEN_EXPERIMENT_HEAD,
        "training_git_head": launcher.TRAINING_HEAD,
        "validation_git_head": launcher.VALIDATION_HEAD,
        "eligible_components": ["setflow"],
        "confirmation_protocol": bindings["protocol_path"],
        "recovery_config": bindings["recovery_config_path"],
        "recovery_runtime": str(tmp_path / "recovery_runtime.json"),
        "recovered_screen_gate": bindings["recovered_screen_gate_path"],
        "confirmation_authorization": bindings[
            "confirmation_authorization_path"
        ],
        "runner_verification_receipt": str(tmp_path / "receipt.json"),
        "config_manifest": bindings["config_manifest_path"],
        "runtime_manifest": bindings["training_runtime_path"],
        "required_seeds": list(launcher.CONFIRMATION_SEEDS),
        "gpu_diagnostics_before_launch": diagnostics,
        "cuda_bf16_probes": probes,
        "free_memory_gate_applied": False,
        "gpu_queues": queues,
        "posttraining_bindings": bindings,
        "training_reused_from_screen": False,
        "screen_training_reused_by_recovery": True,
        "recovery_parameter_update_count": 0,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return schedule, jobs


def _successful_training_package(
    tmp_path: Path, jobs: dict[int, dict]
) -> tuple[dict, dict[int, Path]]:
    configs: dict[int, Path] = {}
    runtime_jobs = {}
    recovered_gate = str(tmp_path / "recovered_gate.json")
    for seed, job in jobs.items():
        gpu = int(seed - launcher.CONFIRMATION_SEEDS[0])
        output = Path(job["output_directory"])
        output.mkdir(parents=True)
        saved = {}
        for checkpoint_pass in launcher.CHECKPOINT_PASSES:
            checkpoint = output / f"pass_{checkpoint_pass}.pt"
            checkpoint.touch()
            saved[str(checkpoint_pass)] = str(checkpoint)
        _write_json(
            output / "training_summary.json",
            {
                "schema_version": (
                    "route_a_v3_route2_xeditsetflow_v4_training_summary.v1"
                ),
                "status": (
                    "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"
                ),
                "run_stage": "CONFIRMATION",
                "run_id": "v4_full",
                "seed": seed,
                "completed_passes": 10,
                "checkpoint_selection_status": (
                    "PENDING_TERMINAL_OUTCOME_FREE_VALIDATION_GENERATION"
                ),
                "validation_generation_during_training": False,
                "parameter_changed": True,
                "optimizer_update_count": 123,
                "torch_device": f"cuda:{gpu}",
                "training_precision": "BF16",
                "cpu_fallback_used": False,
                "physical_gpu_index": gpu,
                "saved_checkpoint_paths": saved,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        _write_json(
            output / "training_config.json",
            {
                "authorized_git_head": TRAINING_RUNNER_HEAD,
                "run_stage": "CONFIRMATION",
                "run_id": "v4_full",
                "training_seed": seed,
                "screen_gate_path": recovered_gate,
                "device": f"cuda:{gpu}",
                "validation_recovery": {
                    "training_git_head": launcher.TRAINING_HEAD,
                    "validation_git_head": launcher.VALIDATION_HEAD,
                    "parameter_updates": 0,
                    "scientific_thresholds_changed": False,
                },
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            },
        )
        _write_json(
            output / "training_attempt.json",
            {"code_commit": TRAINING_RUNNER_HEAD},
        )
        config_path = tmp_path / "configs" / f"seed_{seed}.json"
        _write_json(
            config_path,
            {
                "training_seed": seed,
                "output_root": str(output.parent),
                "validation_output_root": str(
                    tmp_path / f"validation_{seed}"
                ),
                "screen_gate_path": recovered_gate,
            },
        )
        configs[seed] = config_path
        runtime_jobs[f"setflow:{seed}:v4_full"] = {
            "terminal_artifact_kind": "SUMMARY",
            "status": "TERMINAL_COMPLETE",
            "training_seed": seed,
            "run_id": "v4_full",
            "output_directory": str(output),
            "physical_gpu_index": gpu,
        }
    runtime = {
        "schema_version": (
            "route_a_v3_route2_xedit_v4_confirmation_training_runtime.v1"
        ),
        "status": "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL",
        "git_head": TRAINING_RUNNER_HEAD,
        "experiment_head": launcher.SCREEN_EXPERIMENT_HEAD,
        "eligible_components": ["setflow"],
        "jobs": runtime_jobs,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    return runtime, configs


def test_recovered_training_schedule_requires_exact_runner_head_and_jobs(
    tmp_path: Path,
) -> None:
    schedule, expected_jobs = _training_schedule(tmp_path)
    bindings, jobs = launcher.validate_training_schedule_v403(
        schedule, training_runner_head=TRAINING_RUNNER_HEAD
    )
    assert bindings == schedule["posttraining_bindings"]
    assert jobs == expected_jobs

    schedule["git_head"] = "c" * 40
    with pytest.raises(Exception, match="Git provenance"):
        launcher.validate_training_schedule_v403(
            schedule, training_runner_head=TRAINING_RUNNER_HEAD
        )


def test_exact_three_seed_summary_package_binds_cuda_and_training_provenance(
    tmp_path: Path,
) -> None:
    schedule, jobs = _training_schedule(tmp_path)
    runtime, configs = _successful_training_package(tmp_path, jobs)
    packages = launcher.validate_successful_training_package_v403(
        runtime,
        jobs,
        configs,
        training_runner_head=TRAINING_RUNNER_HEAD,
        recovered_screen_gate_path=schedule["recovered_screen_gate"],
    )
    assert set(packages) == set(launcher.CONFIRMATION_SEEDS)
    assert all(
        set(package["checkpoint_paths"]) == set(launcher.CHECKPOINT_PASSES)
        for package in packages.values()
    )

    wrong_config_path = configs[launcher.CONFIRMATION_SEEDS[0]]
    config = json.loads(wrong_config_path.read_text(encoding="utf-8"))
    config["output_root"] = str(tmp_path / "wrong_training_root")
    _write_json(wrong_config_path, config)
    with pytest.raises(Exception, match="does not match recovered config"):
        launcher.validate_successful_training_package_v403(
            runtime,
            jobs,
            configs,
            training_runner_head=TRAINING_RUNNER_HEAD,
            recovered_screen_gate_path=schedule["recovered_screen_gate"],
        )
    config["output_root"] = str(
        Path(jobs[launcher.CONFIRMATION_SEEDS[0]]["output_directory"]).parent
    )
    _write_json(wrong_config_path, config)

    runtime["jobs"]["setflow:20260914:v4_full"][
        "terminal_artifact_kind"
    ] = "FAILURE"
    with pytest.raises(Exception, match="exact terminal SUMMARY"):
        launcher.validate_successful_training_package_v403(
            runtime,
            jobs,
            configs,
            training_runner_head=TRAINING_RUNNER_HEAD,
            recovered_screen_gate_path=schedule["recovered_screen_gate"],
        )


def test_posttraining_schedule_has_fixed_recovered_twelve_job_inventory_without_memory_sorting(
    tmp_path: Path,
) -> None:
    training_schedule, jobs = _training_schedule(tmp_path)
    runtime, configs = _successful_training_package(tmp_path, jobs)
    packages = launcher.validate_successful_training_package_v403(
        runtime,
        jobs,
        configs,
        training_runner_head=TRAINING_RUNNER_HEAD,
        recovered_screen_gate_path=training_schedule["recovered_screen_gate"],
    )
    gpus = (3, 1, 4, 0, 5, 2)
    diagnostics = {
        gpu: {
            "name": f"gpu-{gpu}",
            "free_memory_mib": 1 if gpu == 3 else 40000,
            "total_memory_mib": 40960,
        }
        for gpu in gpus
    }
    probes = {
        gpu: {
            "physical_gpu_index": gpu,
            "device_type": "cuda",
            "dtype": "BFLOAT16",
            "cuda_available": True,
            "bf16_supported": True,
            "cpu_fallback_used": False,
        }
        for gpu in gpus
    }
    schedule = launcher.build_posttraining_schedule_v403(
        tmp_path / "training_schedule.json",
        training_schedule,
        training_schedule["posttraining_bindings"],
        configs,
        packages,
        gpus,
        diagnostics,
        probes,
        orchestration_head=ORCHESTRATION_HEAD,
        training_runner_head=TRAINING_RUNNER_HEAD,
    )
    inventory = schedule["validation_inventory"]
    assert len(inventory) == 12
    assert {
        (job["training_seed"], job["checkpoint_pass"])
        for job in inventory
    } == {
        (seed, checkpoint_pass)
        for seed in launcher.CONFIRMATION_SEEDS
        for checkpoint_pass in launcher.CHECKPOINT_PASSES
    }
    assert inventory[0]["physical_gpu_index"] == 3
    assert inventory[0]["checkpoint_path"].endswith("pass_4.pt")
    assert all(
        job["recovered_screen_gate_path"]
        == training_schedule["recovered_screen_gate"]
        for job in inventory
    )
    assert all(len(queue["jobs"]) == 2 for queue in schedule["validation_queues"])
    assert schedule["posttraining_bindings"] == training_schedule[
        "posttraining_bindings"
    ]
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["cpu_fallback_used"] is False


def test_recovered_protocol_bindings_keep_separate_artifact_family() -> None:
    protocol = json.loads(launcher.PROTOCOL.read_text(encoding="utf-8"))
    outputs = protocol["runner_outputs"]
    posttraining = protocol["posttraining_binding"]
    bindings = {
        "protocol_path": str(launcher.PROTOCOL),
        "training_git_head": launcher.TRAINING_HEAD,
        "validation_git_head": launcher.VALIDATION_HEAD,
        "runner_git_head": TRAINING_RUNNER_HEAD,
        "recovery_config_path": posttraining["recovery_config_path"],
        "recovered_screen_gate_path": posttraining[
            "recovered_screen_gate_path"
        ],
        "confirmation_authorization_path": outputs[
            "authorization_output_template"
        ].format(runner_git_head=TRAINING_RUNNER_HEAD),
        "config_manifest_path": posttraining["config_manifest_path"],
        "training_runtime_path": "/tmp/training_runtime.json",
        "posttraining_runtime_root": outputs[
            "posttraining_runtime_root_template"
        ].format(runner_git_head=TRAINING_RUNNER_HEAD),
        "posttraining_log_root": outputs[
            "posttraining_log_root_template"
        ].format(runner_git_head=TRAINING_RUNNER_HEAD),
        "confirmation_gate_output": posttraining["confirmation_gate_output"],
    }
    schedule = {"confirmation_authorization": bindings["confirmation_authorization_path"]}
    launcher.validate_posttraining_bindings_v403(
        schedule,
        protocol,
        bindings,
        training_runner_head=TRAINING_RUNNER_HEAD,
        expected_protocol_path=launcher.PROTOCOL,
    )
    assert "confirmation_v403_recovered_posttraining" in bindings[
        "posttraining_runtime_root"
    ]
    assert "/experiments/xedit_v4/confirmation_posttraining_" not in bindings[
        "posttraining_runtime_root"
    ]


def test_posttraining_one_shot_rejects_existing_runtime_log_or_gate(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    logs = tmp_path / "logs"
    gate = tmp_path / "gate.json"
    launcher.require_fresh_posttraining_targets_v403(
        runtime_root=runtime, log_root=logs, confirmation_gate=gate
    )
    runtime.mkdir()
    with pytest.raises(Exception, match="already exists"):
        launcher.require_fresh_posttraining_targets_v403(
            runtime_root=runtime, log_root=logs, confirmation_gate=gate
        )


def test_successor_reuses_generic_scheduler_and_fails_closed_on_cuda() -> None:
    assert launcher.POSTTRAINING_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
    )
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "cuda_bf16_probe(gpu)" in source
    assert "STOPPED_BEFORE_CONFIRMATION_VALIDATION_LAUNCH" in source
    assert '"free_memory_gate_applied": False' in source
    assert "sorted(diagnostics" not in source
    assert "free_memory_mib] >=" not in source
    assert "require_science_protocol_unchanged_v403(" in source
    assert "validate_authorization_v403(" in source
    assert "require_runner_verification_receipt_v403(" in source
    assert "subprocess.Popen(" in source
