from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_posttraining_after_terminal as launcher


def test_posttraining_scheduler_popen_failure_is_durable_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "posttraining_family"
    family.mkdir()
    schedule = family / "schedule.json"
    schedule.write_text("{}\n", encoding="utf-8")
    failure = family / "scheduler_launch.failed.json"

    def fail(*args, **kwargs):
        raise OSError("posttraining scheduler spawn failed")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    arguments = {
        "failure_path": failure,
        "expected_head": "a" * 40,
        "command_line": ["python", "scheduler.py", "--schedule", str(schedule)],
        "schedule_path": schedule,
        "runtime_path": family / "runtime.json",
        "scheduler_log": family / "scheduler.log",
        "created_artifacts": {"schedule": schedule},
    }
    with pytest.raises(
        launcher.XEditV4ConfirmationPosttrainingLaunchError,
        match="durable technical failure",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)

    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "V4_CONFIRMATION_POSTTRAINING_SCHEDULER_LAUNCH_TECHNICAL_FAILURE"
    )
    assert payload["scheduler_started"] is False
    assert payload["gpu_job_started"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert "coordinator_pid" not in payload
    assert not (family / "launch.json").exists()

    with pytest.raises(
        launcher.XEditV4ConfirmationPosttrainingLaunchError,
        match="already exists",
    ):
        launcher.spawn_scheduler_with_failure_evidence(**arguments)


def test_confirmation_posttraining_launcher_uses_current_head_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.POSTTRAINING_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
    )


def test_confirmation_training_job_set_is_component_exact() -> None:
    critic = launcher.expected_training_job_keys(("critic",))
    setflow = launcher.expected_training_job_keys(("setflow",))
    assert critic == {
        f"critic:{seed}:{run_id}"
        for seed in (20260908, 20260909, 20260910)
        for run_id in ("v4_full", "c0_v4")
    }
    assert setflow == {
        f"setflow:{seed}:v4_full" for seed in (20260912, 20260913, 20260914)
    }
    assert launcher.expected_training_job_keys(("critic", "setflow")) == (
        critic | setflow
    )


def test_confirmation_setflow_validation_assignment_is_exact_and_balanced() -> None:
    assignments = launcher.validation_assignments(
        (20260912, 20260913, 20260914)
    )
    jobs = [job for rows in assignments.values() for job in rows]
    assert set(jobs) == {
        (seed, checkpoint_pass)
        for seed in (20260912, 20260913, 20260914)
        for checkpoint_pass in (4, 6, 8, 10)
    }
    assert len(jobs) == 12
    assert set(assignments) == set(range(6))
    assert all(len(rows) == 2 for rows in assignments.values())


def test_confirmation_setflow_validation_rejects_additional_seed() -> None:
    with pytest.raises(Exception, match="seed changed"):
        launcher.validation_assignments((20260912, 20260915))


def test_confirmation_runtime_rejects_nonterminal_job() -> None:
    runtime = {
        "status": "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL",
        "git_head": "a" * 40,
        "eligible_components": ["setflow"],
        "jobs": {
            f"setflow:{seed}:v4_full": {
                "terminal_artifact_kind": "SUMMARY"
            }
            for seed in (20260912, 20260913, 20260914)
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    runtime["jobs"]["setflow:20260914:v4_full"]["terminal_artifact_kind"] = None
    with pytest.raises(Exception, match="lacks an exact terminal"):
        launcher.validate_confirmation_runtime(runtime, head="a" * 40)


def test_confirmation_validation_records_memory_without_gating() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"setflow_diagnostic_peak_plus_two_gib_mib"' in source
    assert "free_memory[gpu] >=" not in source


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 3, stdout="partial", stderr="driver"
            ),
            "NONZERO_RETURN_CODE",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 0, stdout="broken\n", stderr=""
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0, 1\n1, 1\n2, 1\n3, 1\n4, 1\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_posttraining_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(
        launcher.XEditV4ConfirmationPosttrainingGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_posttraining_inventory_command_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(
        launcher.XEditV4ConfirmationPosttrainingGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None


def test_posttraining_inventory_failure_is_persisted_once_for_scheduler(
    tmp_path: Path,
) -> None:
    path = tmp_path / "posttraining.failed.json"
    error = launcher.XEditV4ConfirmationPosttrainingGpuInventoryError(
        "malformed inventory",
        reason="OUTPUT_PARSE_FAILED",
        return_code=0,
        stdout="broken\n",
    )
    kwargs = {
        "head": "a" * 40,
        "eligible_components": ("setflow",),
        "training_runtime_path": tmp_path / "training" / "runtime.json",
        "runtime_root": tmp_path / "posttraining",
        "log_root": tmp_path / "logs",
        "error": error,
    }
    launcher.write_gpu_inventory_failure_evidence(path, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["inventory_failure_reason"] == "OUTPUT_PARSE_FAILED"
    assert payload["command"] == list(launcher.GPU_INVENTORY_COMMAND)
    assert payload["scheduler_started"] is False
    assert payload["validation_job_started"] is False
    assert payload["free_memory_gate_applied"] is False
    with pytest.raises(Exception, match="already exists"):
        launcher.write_gpu_inventory_failure_evidence(path, **kwargs)


def test_posttraining_inventory_evidence_guard_precedes_inventory() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(head: str)") :]
    failure_guard = run_source.index("not prelaunch_failure_path.exists()")
    inventory = run_source.index("free_memory = gpu_free_memory_mib()")
    runtime_creation = run_source.index("runtime_root.mkdir(parents=True)")
    assert failure_guard < inventory < runtime_creation
