from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_s1_confirmation_posttraining as launcher


HEAD = "b" * 40


def _config_paths(tmp_path: Path) -> dict[int, Path]:
    result = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "output_root": str(tmp_path / f"training/seed_{seed}"),
                    "validation_output_root": str(
                        tmp_path / f"validation/seed_{seed}"
                    ),
                }
            )
        )
        result[seed] = path
    return result


def _packages(tmp_path: Path) -> dict[int, dict]:
    result = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        training = tmp_path / f"training/seed_{seed}/v4_s1_full"
        training.mkdir(parents=True)
        checkpoints = {}
        for checkpoint_pass in launcher.CHECKPOINT_PASSES:
            checkpoint = training / f"pass_{checkpoint_pass}.pt"
            checkpoint.write_text("checkpoint")
            checkpoints[checkpoint_pass] = str(checkpoint)
        summary = training / "training_summary.json"
        summary.write_text("{}")
        result[seed] = {
            "training_summary_path": str(summary),
            "checkpoint_paths": checkpoints,
        }
    return result


def _bindings(tmp_path: Path) -> dict:
    return {
        "protocol_path": str(launcher.PROTOCOL),
        "runner_git_head": HEAD,
        "config_manifest_path": str(tmp_path / "manifest.json"),
        "confirmation_authorization_path": str(tmp_path / "authorization.json"),
        "training_runtime_path": str(tmp_path / "training_runtime.json"),
        "posttraining_runtime_root": str(tmp_path / "posttraining"),
        "posttraining_log_root": str(tmp_path / "posttraining_logs"),
        "confirmation_gate_output": str(tmp_path / "posttraining/confirmation_gate.json"),
    }


def test_validation_inventory_is_exact_twelve_full_only_two_per_gpu(
    tmp_path: Path,
) -> None:
    inventory, queues = launcher.build_validation_inventory_s1(
        _config_paths(tmp_path),
        _packages(tmp_path),
        tuple(range(6)),
        authorization_path=tmp_path / "authorization.json",
        log_root=tmp_path / "logs",
    )
    assert len(inventory) == 12
    assert {job["run_id"] for job in inventory} == {"v4_s1_full"}
    assert {
        (job["training_seed"], job["checkpoint_pass"]) for job in inventory
    } == {
        (seed, checkpoint_pass)
        for seed in launcher.CONFIRMATION_SEEDS
        for checkpoint_pass in launcher.CHECKPOINT_PASSES
    }
    assert [len(queue["jobs"]) for queue in queues] == [2] * 6
    assert [queue["physical_gpu_index"] for queue in queues] == list(range(6))


def test_low_free_memory_is_diagnostic_only_and_never_changes_twelve_job_schedule(
    tmp_path: Path,
) -> None:
    configs = _config_paths(tmp_path)
    packages = _packages(tmp_path)
    bindings = _bindings(tmp_path)
    training_schedule = {"posttraining_bindings": bindings}
    diagnostics = {
        gpu: {"name": "A100", "free_memory_mib": 1 if gpu == 5 else 80_000}
        for gpu in range(6)
    }
    probes = {gpu: {"device_class": "A100"} for gpu in range(6)}
    schedule = launcher.build_posttraining_schedule_s1(
        tmp_path / "training_schedule.json",
        training_schedule,
        bindings,
        configs,
        packages,
        tuple(range(6)),
        diagnostics,
        probes,
        expected_head=HEAD,
    )
    assert len(schedule["validation_inventory"]) == 12
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["gpu_diagnostics_before_launch"]["5"]["free_memory_mib"] == 1
    assert schedule["adjudications"]["setflow"]["command"][1] == str(
        launcher.ADJUDICATOR
    )


def test_incomplete_or_protected_training_runtime_cannot_enter_posttraining() -> None:
    incomplete = {
        "schema_version": (
            "route_a_v3_route2_xedit_v4_confirmation_training_runtime.v1"
        ),
        "status": "V4_CONFIRMATION_TRAINING_TECHNICAL_FAILURE",
        "git_head": HEAD,
        "experiment_head": launcher.SCREEN_HEAD,
        "eligible_components": ["setflow"],
        "first_terminal_failure": {"job_key": "failed"},
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "jobs": {},
    }
    with pytest.raises(
        launcher.XEditSetFlowS1ConfirmationPosttrainingLaunchError,
        match="not exact successful terminal",
    ):
        launcher.validate_successful_training_package_s1(
            incomplete, {}, {}, expected_head=HEAD
        )
    incomplete["status"] = "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
    incomplete["first_terminal_failure"] = None
    incomplete["development_test_outcome_reads"] = 1
    with pytest.raises(RuntimeError, match="protected outcome read"):
        launcher.validate_successful_training_package_s1(
            incomplete, {}, {}, expected_head=HEAD
        )


def test_existing_final_or_partial_blocks_same_posttraining_family(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "posttraining"
    log_root = tmp_path / "logs"
    gate = runtime_root / "confirmation_gate.json"
    failure = tmp_path / "posttraining.failed.json"
    gate.parent.mkdir()
    gate.write_text("{}")
    with pytest.raises(
        launcher.XEditSetFlowS1ConfirmationPosttrainingLaunchError,
        match="family, final, failure, or partial",
    ):
        launcher.require_fresh_posttraining_targets_s1(
            runtime_root=runtime_root,
            log_root=log_root,
            gate=gate,
            prelaunch_failure=failure,
            inventory=[],
        )


def test_posttraining_probe_failure_is_sibling_before_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "posttraining"
    failure = tmp_path / "posttraining.failed.json"

    def inventory(_gpus):
        return {gpu: {"name": "A100", "free_memory_mib": 1} for gpu in range(6)}

    def probe(gpu):
        if gpu == 2:
            raise launcher.XEditSetFlowS1GpuError(
                "probe failed",
                reason="CUDA_BF16_PROBE_NONZERO_RETURN_CODE",
                failed_physical_gpu_index=2,
                probe_command=("python", "probe", "2"),
            )
        return {"device_class": "A100", "device_type": "cuda"}

    monkeypatch.setattr(launcher, "gpu_diagnostics", inventory)
    monkeypatch.setattr(launcher, "cuda_bf16_probe", probe)
    with pytest.raises(launcher.XEditSetFlowS1GpuError):
        launcher.perform_posttraining_gpu_preflight_s1(
            runner_head=HEAD,
            configured_gpus=tuple(range(6)),
            failure_path=failure,
            runtime_root=runtime_root,
        )
    payload = json.loads(failure.read_text())
    assert payload["failure_stage"] == "A100_CUDA_BF16_PROBE_BEFORE_POSTTRAINING"
    assert payload["failed_physical_gpu_index"] == 2
    assert set(payload["completed_cuda_bf16_probes"]) == {"0", "1"}
    assert payload["runtime_root_created"] is False
    assert not runtime_root.exists()


def test_posttraining_scheduler_launch_failure_retains_exact_command_and_stage(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "scheduler_launch.failed.json"
    launcher.write_scheduler_launch_failure_s1(
        evidence,
        runner_head=HEAD,
        command_line=["python", "post_scheduler.py", "--schedule", "schedule.json"],
        schedule_path=tmp_path / "schedule.json",
        runtime_path=tmp_path / "runtime.json",
        error=OSError("Popen failed"),
    )
    payload = json.loads(evidence.read_text())
    assert payload["failure_stage"] == (
        "CONFIRMATION_POSTTRAINING_SCHEDULER_PROCESS_LAUNCH"
    )
    assert payload["scheduler_command"][1] == "post_scheduler.py"
    assert payload["scheduler_started"] is False
