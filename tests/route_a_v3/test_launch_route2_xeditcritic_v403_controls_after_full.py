from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v403_controls_after_full as launcher
import scripts.route_a_v3.run_route2_xeditcritic_v403_control_recovery_scheduler as scheduler


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _full_summary(
    authorization_path: Path,
    *,
    physical_gpu_index: int = 3,
    protected_reads: int = 0,
) -> dict:
    summary = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
        "status": "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE",
        "run_id": "v4_full",
        "model_kind": "V4-FULL",
        "precision": "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE",
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "physical_gpu_index": physical_gpu_index,
        "launch_authorization_path": str(authorization_path),
        "development_test_outcome_reads": protected_reads,
        "new_final_evaluation_outcome_reads": 0,
    }
    summary.update(launcher.FROZEN_FULL_SUMMARY_IDENTITY)
    return summary


def _full_runtime(*, physical_gpu_index: int = 3) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1"
        ),
        "status": "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL",
        "terminal_artifact_kind": "SUMMARY",
        "return_code": 0,
        "run_id": "v4_full",
        "git_head": launcher.TRAINING_GIT_HEAD,
        "physical_gpu_index": physical_gpu_index,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _full_authorization(
    *, physical_gpu_index: int = 3, protected_reads: int = 0
) -> dict:
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
        ),
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": launcher.TRAINING_GIT_HEAD,
        "authorized_run_ids": list(launcher.ALL_RUN_IDS),
        "v403_rng_replay_recovery": {
            "run_id": "v4_full",
            "physical_gpu_index": physical_gpu_index,
        },
        "development_test_outcome_reads": protected_reads,
        "new_final_evaluation_outcome_reads": 0,
    }


def _inventory() -> list[dict]:
    return [
        {
            "physical_gpu_index": index,
            "device_name": "NVIDIA A100-PCIE-40GB",
            "bf16_supported": True,
            "bf16_tensor_probe": True,
        }
        for index in launcher.PHYSICAL_GPU_INDICES
    ]


def test_nonterminal_full_stops_before_probe_spawn_or_artifact_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "CURRENT_FULL_OUTPUT_ROOT", tmp_path / "full")
    monkeypatch.setattr(launcher, "CURRENT_FULL_RUNTIME", tmp_path / "runtime.json")
    monkeypatch.setattr(launcher, "CONTROL_OUTPUT_ROOT", tmp_path / "controls")
    monkeypatch.setattr(launcher, "RUNTIME_ROOT", tmp_path / "runner")
    monkeypatch.setattr(launcher, "AUTHORIZATION_ROOT", tmp_path / "auth")
    monkeypatch.setattr(launcher, "TRANSITION_GATE", tmp_path / "gate/screen_gate.json")

    calls = {"probe": 0, "spawn": 0}

    def forbidden_probe() -> list[dict]:
        calls["probe"] += 1
        raise AssertionError("CUDA probe must not run before full terminal")

    def forbidden_spawn(*args, **kwargs):
        calls["spawn"] += 1
        raise AssertionError("worker must not spawn before full terminal")

    monkeypatch.setattr(launcher, "probe_cuda_bf16", forbidden_probe)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_spawn)

    with pytest.raises(Exception, match="not exact terminal SUMMARY"):
        launcher.launch("a" * 40)

    assert calls == {"probe": 0, "spawn": 0}
    assert not (tmp_path / "controls").exists()
    assert not (tmp_path / "runner").exists()
    assert not (tmp_path / "auth").exists()
    assert not (tmp_path / "gate").exists()


def test_full_terminal_requires_zero_protected_reads(tmp_path: Path) -> None:
    output_root = tmp_path / "full"
    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        _full_authorization(),
    )
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization, protected_reads=1),
    )
    runtime = tmp_path / "runtime.json"
    _write_json(runtime, _full_runtime())

    with pytest.raises(Exception, match="Development TEST read"):
        launcher.validate_current_full_terminal(output_root, runtime)


def test_full_terminal_accepts_consistent_non_gpu5_frozen_identity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        _full_authorization(physical_gpu_index=3),
    )
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization_path, physical_gpu_index=3),
    )
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime(physical_gpu_index=3))

    result = launcher.validate_current_full_terminal(output_root, runtime_path)

    assert result["physical_gpu_index"] == 3


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("seed", 20260908),
        ("pass_count", 7),
        ("selected_pass", 7),
        ("update_count", 22415),
        ("selection_policy", "VALIDATION_PEAK_RESELECTION"),
        ("train_record_count", 89579),
        ("validation_record_count", 18292),
        ("effective_batch_size", 16),
        ("physical_batch_size", 16),
    ],
)
def test_full_terminal_rejects_non_frozen_training_identity(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, _full_authorization())
    summary = _full_summary(authorization_path)
    summary[field] = invalid_value
    _write_json(output_root / "v4_full/run_summary.json", summary)
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime())

    with pytest.raises(Exception, match="frozen training identity"):
        launcher.validate_current_full_terminal(output_root, runtime_path)


def test_full_terminal_rejects_gpu_disagreement_or_unauthorized_run(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "full"
    authorization_path = tmp_path / "authorization.json"
    authorization = _full_authorization(physical_gpu_index=4)
    _write_json(authorization_path, authorization)
    _write_json(
        output_root / "v4_full/run_summary.json",
        _full_summary(authorization_path, physical_gpu_index=3),
    )
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, _full_runtime(physical_gpu_index=3))
    with pytest.raises(Exception, match="physical GPUs disagree"):
        launcher.validate_current_full_terminal(output_root, runtime_path)

    authorization["v403_rng_replay_recovery"]["physical_gpu_index"] = 3
    authorization["authorized_run_ids"].remove("v4_full")
    _write_json(authorization_path, authorization)
    with pytest.raises(Exception, match="launch authorization identity"):
        launcher.validate_current_full_terminal(output_root, runtime_path)


def test_schedule_is_exactly_six_fresh_controls_from_f34(tmp_path: Path) -> None:
    schedule = launcher.build_control_schedule(
        expected_orchestration_head="a" * 40,
        config_path=tmp_path / "screen_config.json",
        authorization_path=tmp_path / "authorization.json",
        cuda_bf16_inventory=_inventory(),
        output_root=tmp_path / "fresh_controls",
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
    )

    assert [job["run_id"] for job in schedule["jobs"]] == list(
        launcher.CONTROL_RUN_IDS
    )
    assert not {"c0_v4", "v4_full"} & {
        job["run_id"] for job in schedule["jobs"]
    }
    assert [job["physical_gpu_index"] for job in schedule["jobs"]] == list(
        range(6)
    )
    assert all(
        job["command"][:2] == [str(launcher.PYTHON), str(launcher.TRAINER)]
        for job in schedule["jobs"]
    )
    assert all(
        str(tmp_path / "fresh_controls")
        in str(job["output_directory"])
        for job in schedule["jobs"]
    )
    assert schedule["training_code_git_head"] == launcher.TRAINING_GIT_HEAD
    assert schedule["full_retrained"] is False
    assert schedule["c0_retrained"] is False
    assert schedule["old_v402_stopped_process_resumed"] is False
    assert schedule["development_test_outcome_reads"] == 0
    assert schedule["new_final_evaluation_outcome_reads"] == 0
    scheduler.validate_schedule(schedule)


def test_schedule_rejects_missing_control_or_full_injection(tmp_path: Path) -> None:
    schedule = launcher.build_control_schedule(
        expected_orchestration_head="a" * 40,
        config_path=tmp_path / "screen_config.json",
        authorization_path=tmp_path / "authorization.json",
        cuda_bf16_inventory=_inventory(),
        output_root=tmp_path / "fresh_controls",
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
    )
    schedule["jobs"] = schedule["jobs"][:-1]
    with pytest.raises(Exception, match="exact six-control package"):
        scheduler.validate_schedule(schedule)

    schedule = launcher.build_control_schedule(
        expected_orchestration_head="a" * 40,
        config_path=tmp_path / "screen_config.json",
        authorization_path=tmp_path / "authorization.json",
        cuda_bf16_inventory=_inventory(),
        output_root=tmp_path / "fresh_controls",
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
    )
    schedule["jobs"][0]["run_id"] = "v4_full"
    with pytest.raises(Exception, match="six-control package|retrain"):
        scheduler.validate_schedule(schedule)


def test_launcher_and_scheduler_have_no_free_memory_launch_gate() -> None:
    sources = [
        Path(launcher.__file__).read_text(encoding="utf-8"),
        Path(scheduler.__file__).read_text(encoding="utf-8"),
    ]
    forbidden = (
        "memory.free",
        "required_free_memory",
        "launch_required_free_memory",
        "peak plus 2 GiB",
    )
    for source in sources:
        assert '"free_memory_gate_applied": False' in source
        assert not any(text in source for text in forbidden)
    assert "SIGCONT" not in sources[0] + sources[1]
    assert "os.kill" not in sources[0] + sources[1]


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                6,
                stdout="partial",
                stderr="driver",
            ),
            "NONZERO_RETURN_CODE",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="not-an-index\n",
                stderr="",
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0\n1\n2\n3\n4\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_controls_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(launcher.XEditCriticV403GpuInventoryError) as captured:
        launcher.physical_gpu_inventory()
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_controls_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(launcher.XEditCriticV403GpuInventoryError) as captured:
        launcher.physical_gpu_inventory()
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None
    assert captured.value.command_line == launcher.GPU_INVENTORY_COMMAND


def _patch_controls_prelaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "CONTROL_OUTPUT_ROOT", tmp_path / "controls")
    monkeypatch.setattr(launcher, "RUNTIME_ROOT", tmp_path / "runner")
    monkeypatch.setattr(launcher, "AUTHORIZATION_ROOT", tmp_path / "authorization")
    monkeypatch.setattr(launcher, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(launcher, "TRANSITION_GATE", tmp_path / "gate/screen_gate.json")
    monkeypatch.setattr(
        launcher,
        "validate_current_full_terminal",
        lambda: {"summary": {"status": "TERMINAL"}},
    )
    monkeypatch.setattr(
        launcher,
        "validate_historical_c0_terminal",
        lambda: {"status": "TERMINAL"},
    )
    monkeypatch.setattr(
        launcher,
        "validate_training_source",
        lambda expected_head: {"preflight": {}},
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run after prelaunch failure")
        ),
    )


def test_controls_inventory_failure_stops_before_probe_runtime_or_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_controls_prelaunch(tmp_path, monkeypatch)
    error = launcher.XEditCriticV403GpuInventoryError(
        "missing configured GPU",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0\n1\n2\n3\n4\n",
        missing_physical_gpus=(5,),
    )
    monkeypatch.setattr(
        launcher,
        "physical_gpu_inventory",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        launcher,
        "probe_cuda_bf16",
        lambda: (_ for _ in ()).throw(
            AssertionError("CUDA/BF16 probe must not run after inventory failure")
        ),
    )

    with pytest.raises(launcher.XEditCriticV403GpuInventoryError):
        launcher.launch("d" * 40)

    runtime_root = tmp_path / "runner"
    evidence_path = launcher.sibling_failure_path(runtime_root)
    assert not runtime_root.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditcritic_prelaunch_failure.v1"
    )
    assert evidence["status"] == "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE"
    assert evidence["launcher"] == "controls"
    assert evidence["failure_stage"] == "INVENTORY"
    assert evidence["expected_head"] == "d" * 40
    assert evidence["runtime_root_created"] is False
    assert evidence["jobs_started"] == 0
    assert evidence["cpu_fallback_used"] is False
    assert evidence["free_memory_gate_applied"] is False
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


def test_controls_cuda_probe_failure_preserves_observed_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_controls_prelaunch(tmp_path, monkeypatch)
    monkeypatch.setattr(
        launcher,
        "physical_gpu_inventory",
        lambda: launcher.PHYSICAL_GPU_INDICES,
    )
    error = launcher.XEditCriticV403CudaBf16ProbeError(
        "CPU fallback observed", cpu_fallback_used=True
    )
    monkeypatch.setattr(
        launcher,
        "probe_cuda_bf16",
        lambda: (_ for _ in ()).throw(error),
    )

    with pytest.raises(launcher.XEditCriticV403CudaBf16ProbeError):
        launcher.launch("e" * 40)

    runtime_root = tmp_path / "runner"
    evidence = json.loads(
        launcher.sibling_failure_path(runtime_root).read_text(encoding="utf-8")
    )
    assert not runtime_root.exists()
    assert evidence["failure_stage"] == "CUDA_BF16_PROBE"
    assert evidence["reason"] == "CUDA_BF16_PROBE_OUTPUT_INVALID"
    assert evidence["cpu_fallback_used"] is True
    assert evidence["jobs_started"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_controls_existing_failure_evidence_requires_new_family(
    tmp_path: Path, partial: bool
) -> None:
    runtime_root = tmp_path / "controls_runtime"
    failure = launcher.sibling_failure_path(runtime_root)
    path = failure.with_suffix(failure.suffix + ".partial") if partial else failure
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="new retry family"):
        launcher.require_fresh_prelaunch_family(runtime_root)
