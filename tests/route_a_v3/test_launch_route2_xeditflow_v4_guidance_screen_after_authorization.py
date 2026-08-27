from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditflow_v4_guidance_screen_after_authorization as launcher


def test_guidance_screen_assignment_uses_all_six_gpus_three_times() -> None:
    assignment = launcher.fixed_guidance_gpu_assignment()
    assert assignment == tuple(index % 6 for index in range(18))
    assert all(assignment.count(gpu) == 3 for gpu in range(6))


def test_guidance_screen_manifest_requires_exact_chain_and_final_paths(
    tmp_path: Path,
) -> None:
    fields = (
        "value_training_config_paths",
        "guidance_smc_config_paths",
        "guidance_critic_config_paths",
        "guidance_closed_config_paths",
        "guidance_open_metric_config_paths",
        "guidance_independent_evaluator_config_paths",
        "guidance_independent_evaluator_comparison_config_paths",
    )
    lengths = (6, 18, 18, 18, 18, 18, 18)
    manifest: dict[str, object] = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1",
        "status": "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED",
        "base_flow_training_seed": 20260912,
        "rollout_job_count": 1,
        "critic_score_job_count": 1,
        "value_target_package_count": 6,
        "value_training_job_count": 6,
        "later_guidance_combination_count": 18,
        "guidance_gpu_assignment": list(
            launcher.fixed_guidance_gpu_assignment()
        ),
        "guidance_result_paths": [{} for _ in range(18)],
        "beta_max_used_in_value_target_or_training": False,
        "independent_evaluator_used": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    for field, length in zip(fields, lengths, strict=True):
        paths = []
        for index in range(length):
            path = tmp_path / f"{field}_{index}.json"
            path.write_text("{}\n", encoding="utf-8")
            paths.append(str(path))
        manifest[field] = paths
    launcher.validate_manifest(manifest, config_root=tmp_path)
    manifest["guidance_gpu_assignment"] = [0] * 18
    with pytest.raises(Exception, match="manifest changed"):
        launcher.validate_manifest(manifest, config_root=tmp_path)


def test_guidance_screen_launcher_uses_formal_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditflow_v4_guidance_screen_scheduler.py"
    )


def test_guidance_screen_records_memory_without_filtering_or_sorting() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"diagnostic_peak_plus_two_gib_mib"' in source
    assert "all(free_memory[gpu]" not in source
    assert "key=lambda gpu" not in source


def test_guidance_screen_can_bind_derived_protocol_and_distinct_preflight_heads() -> None:
    parameters = inspect.signature(launcher.run).parameters
    assert {
        "protocol_path",
        "authorization_decision_path",
        "critic_preflight_path",
        "critic_preflight_head",
        "setflow_preflight_path",
        "setflow_preflight_head",
        "execution_runtime_root",
        "execution_log_root",
    } <= set(parameters)
    assert parameters["protocol_path"].default == launcher.PROTOCOL
    assert parameters["critic_preflight_head"].default is None
    assert parameters["setflow_preflight_head"].default is None


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 9, stdout="partial", stderr="driver"
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
def test_guidance_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(
        launcher.XEditFlowV4GuidanceGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_guidance_inventory_command_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(
        launcher.XEditFlowV4GuidanceGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None


def test_guidance_inventory_failure_is_bridge_locatable_and_non_overwriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screen_execution.failed.json"
    error = launcher.XEditFlowV4GuidanceGpuInventoryError(
        "missing GPU 5",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, 100\n",
        missing_physical_gpus=(5,),
    )
    kwargs = {
        "current_head": "a" * 40,
        "experiment_head": "b" * 40,
        "protocol_path": tmp_path / "guidance_protocol.json",
        "authorization_decision_path": tmp_path / "decision.json",
        "critic_preflight_path": tmp_path / "critic.json",
        "critic_preflight_head": "c" * 40,
        "setflow_preflight_path": tmp_path / "setflow.json",
        "setflow_preflight_head": "d" * 40,
        "config_root": tmp_path / "configs",
        "output_root": tmp_path / "outputs",
        "runtime_root": tmp_path / "screen_execution",
        "log_root": tmp_path / "logs",
        "error": error,
    }
    launcher.write_gpu_inventory_failure_evidence(path, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["intended_runtime_root"] == str(tmp_path / "screen_execution")
    assert payload["guidance_protocol_path"] == str(
        tmp_path / "guidance_protocol.json"
    )
    assert payload["missing_physical_gpus"] == [5]
    assert payload["scheduler_started"] is False
    assert payload["free_memory_gate_applied"] is False
    with pytest.raises(Exception, match="already exists"):
        launcher.write_gpu_inventory_failure_evidence(path, **kwargs)


def test_guidance_inventory_failure_path_is_fixed_before_prepare() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(") :]
    failure_guard = run_source.index("not prelaunch_failure_path.exists()")
    inventory = run_source.index("free_memory = gpu_free_memory_mib()")
    prepare = run_source.index("str(PREPARER)")
    assert failure_guard < inventory < prepare
