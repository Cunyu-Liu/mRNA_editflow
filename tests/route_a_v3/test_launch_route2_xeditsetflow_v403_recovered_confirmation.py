from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditsetflow_v403_recovered_confirmation as launcher


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads(
    (
        ROOT
        / "configs/route_a_v3_route2_xeditsetflow_v403_recovered_confirmation_protocol_v1.json"
    ).read_text(encoding="utf-8")
)


def test_component_launcher_reuses_existing_prepare_train_and_scheduler_entries() -> None:
    assert launcher.PREPARE == (
        launcher.WORKTREE
        / "scripts/route_a_v3/prepare_route2_xeditsetflow_v4_confirmation_configs.py"
    )
    assert launcher.TRAINER == (
        launcher.WORKTREE / "scripts/route_a_v3/train_route2_xeditsetflow_v4.py"
    )
    assert launcher.SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
    )


def test_component_launcher_has_no_free_memory_gate_and_keeps_cuda_fail_closed() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert "required_free_memory" not in source
    assert "lacks SetFlow" not in source
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "CUDA_BF16_PROBE_SILENT_CPU_FALLBACK" in source
    assert "cuda_failure_evidence_template" in source


def test_nonterminal_recovery_guard_precedes_any_prepare_or_launch() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    terminal_guard = run_source.index("require_recovery_terminal_v403(")
    prepare_invocation = run_source.index(
        'str(PREPARE),\n            "--base-config"'
    )
    scheduler_launch = run_source.index("process = subprocess.Popen(")
    assert terminal_guard < prepare_invocation < scheduler_launch


def test_exact_runner_receipt_barrier_precedes_config_materialization() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    receipt_read = run_source.index(
        "runner_verification_receipt = read_json("
    )
    receipt_validation = run_source.index(
        "expected_authorization = build_recovered_confirmation_authorization_v403("
    )
    prepare_invocation = run_source.index(
        'str(PREPARE),\n            "--base-config"'
    )
    scheduler_launch = run_source.index("process = subprocess.Popen(")
    assert receipt_read < receipt_validation < prepare_invocation < scheduler_launch


def test_schedule_binds_dual_head_recovered_inputs_and_posttraining(
    tmp_path: Path,
) -> None:
    configs = {}
    for seed in launcher.CONFIRMATION_SEEDS:
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(
            json.dumps({"output_root": str(tmp_path / f"run_{seed}")}) + "\n",
            encoding="utf-8",
        )
        configs[seed] = path
    diagnostics = {
        gpu: {
            "name": f"gpu-{gpu}",
            "free_memory_mib": 1,
            "total_memory_mib": 40960,
        }
        for gpu in (0, 1, 2)
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
        for gpu in (0, 1, 2)
    }
    schedule = launcher.build_schedule_v403(
        PROTOCOL,
        {"gpu_policy": {"physical_gpu_scope": [0, 1, 2]}},
        {"required_seeds": list(launcher.CONFIRMATION_SEEDS)},
        tmp_path / "authorization.json",
        configs,
        (0, 1, 2),
        diagnostics,
        probes,
        runner_head="c" * 40,
        runtime_manifest=tmp_path / "runtime.json",
        log_root=tmp_path / "logs",
    )
    provenance = PROTOCOL["validation_recovery_provenance"]
    assert schedule["git_head"] == "c" * 40
    assert schedule["training_git_head"] == launcher.TRAINING_HEAD
    assert schedule["validation_git_head"] == launcher.VALIDATION_HEAD
    assert schedule["recovery_config"] == provenance["recovery_config_path"]
    assert (
        schedule["recovered_screen_gate"]
        == provenance["recovered_screen_gate_path"]
    )
    assert schedule["confirmation_authorization"].endswith(
        "authorization.json"
    )
    assert schedule["runner_verification_receipt"] == PROTOCOL[
        "runner_outputs"
    ]["runner_verification_receipt_template"].format(runner_git_head="c" * 40)
    assert schedule["posttraining_bindings"]["recovered_screen_gate_path"] == (
        provenance["recovered_screen_gate_path"]
    )
    assert schedule["posttraining_bindings"]["runner_git_head"] == "c" * 40
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["development_test_outcome_reads"] == 0
    assert schedule["new_final_evaluation_outcome_reads"] == 0
    assert {
        job["training_seed"]
        for queue in schedule["gpu_queues"]
        for job in queue["jobs"]
    } == set(launcher.CONFIRMATION_SEEDS)


def test_schedule_requires_three_distinct_gpus_without_memory_threshold(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception, match="three distinct physical GPUs"):
        launcher.build_schedule_v403(
            PROTOCOL,
            {},
            {},
            tmp_path / "authorization.json",
            {},
            (0, 0, 1),
            {},
            {},
            runner_head="c" * 40,
            runtime_manifest=tmp_path / "runtime.json",
            log_root=tmp_path / "logs",
        )


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 5, stdout="partial", stderr="driver"
            ),
            "NONZERO_RETURN_CODE",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0, A100, malformed\n",
                stderr="",
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout=(
                    "0, A100, 1, 40960\n"
                    "1, A100, 1, 40960\n"
                    "2, A100, 1, 40960\n"
                    "3, A100, 1, 40960\n"
                    "4, A100, 1, 40960\n"
                ),
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_recovered_confirmation_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(launcher.XEditSetFlowV403GpuInventoryError) as captured:
        launcher.gpu_diagnostics((0, 1, 2, 3, 4, 5))
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_recovered_confirmation_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(launcher.XEditSetFlowV403GpuInventoryError) as captured:
        launcher.gpu_diagnostics((0, 1, 2))
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None


def test_recovered_confirmation_inventory_failure_is_prelaunch_and_non_overwriting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cuda_failure.json"
    error = launcher.XEditSetFlowV403GpuInventoryError(
        "missing physical GPUs",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, A100, 1, 40960\n",
        missing_physical_gpus=(1, 2),
    )
    kwargs = {
        "current_head": "c" * 40,
        "configured_gpus": (0, 1, 2),
        "selected_gpus": (0, 1, 2),
        "diagnostics": {},
        "authorization_path": tmp_path / "authorization.json",
        "config_root": tmp_path / "configs",
        "run_root": tmp_path / "runs",
        "runtime_root": tmp_path / "runtime",
        "error": error,
    }
    launcher.write_prelaunch_cuda_failure_evidence_v403(path, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["failure_stage"] == (
        "GPU_INVENTORY_BEFORE_CONFIRMATION_FAMILY_MATERIALIZATION"
    )
    assert payload["missing_physical_gpus"] == [1, 2]
    assert payload["authorization_materialized"] is False
    assert payload["scheduler_started"] is False
    assert payload["cpu_fallback_used"] is False
    assert payload["free_memory_gate_applied"] is False
    with pytest.raises(Exception, match="artifact already exists"):
        launcher.write_prelaunch_cuda_failure_evidence_v403(path, **kwargs)


def test_recovered_confirmation_inventory_precedes_config_authorization_and_runtime() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str)") :]
    failure_guard = run_source.index(
        '(cuda_failure_path, "confirmation CUDA failure evidence exists")'
    )
    inventory = run_source.index("diagnostics = gpu_diagnostics(configured_gpus)")
    prepare = run_source.index('str(PREPARE),\n            "--base-config"')
    runtime_creation = run_source.index("runtime_root.mkdir(parents=True)")
    assert failure_guard < inventory < prepare < runtime_creation
