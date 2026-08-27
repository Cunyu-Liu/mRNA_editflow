from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_training_after_screen_pass as launcher


def _gate(component: str, *, passed: bool) -> dict[str, object]:
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return {
        "status": f"{prefix}_V4_SCREEN_{'PASS' if passed else 'NO_GO'}",
        "confirmation_authorized": passed,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_confirmation_launcher_uses_current_head_formal_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.CONFIRMATION_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
    )


def test_confirmation_gate_selection_is_component_independent() -> None:
    assert launcher.gate_passed("critic", _gate("critic", passed=True)) is True
    assert launcher.gate_passed("setflow", _gate("setflow", passed=False)) is False


def test_confirmation_gate_rejects_inconsistent_or_protected_authorization() -> None:
    inconsistent = _gate("critic", passed=True)
    inconsistent["confirmation_authorized"] = False
    with pytest.raises(Exception, match="inconsistent"):
        launcher.gate_passed("critic", inconsistent)

    protected = _gate("setflow", passed=True)
    protected["development_test_outcome_reads"] = 1
    with pytest.raises(Exception, match="protected outcome"):
        launcher.gate_passed("setflow", protected)


def test_confirmation_seed_sets_are_exact_and_additional_seed_is_absent() -> None:
    assert launcher.CRITIC_SEEDS == (20260908, 20260909, 20260910)
    assert launcher.SETFLOW_SEEDS == (20260912, 20260913, 20260914)


def test_confirmation_launcher_consumes_dual_head_screen_authorizations() -> None:
    source = open(launcher.__file__, encoding="utf-8").read()
    assert "def run(current_head: str, experiment_head: str)" in source
    assert "screen_{experiment_head}_runner_{current_head}/critic.json" in source
    assert "screen_{experiment_head}_runner_{current_head}/setflow.json" in source


def test_confirmation_training_records_memory_without_using_it_as_a_gate() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"diagnostic_peak_plus_two_gib_mib_by_gpu"' in source
    assert "free_memory[gpu] >=" not in source


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 7, stdout="partial\n", stderr="driver"
            ),
            "NONZERO_RETURN_CODE",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND, 0, stdout="not,csv,enough\n", stderr=""
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0, 100\n1, 100\n2, 100\n3, 100\n4, 100\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_confirmation_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(
        launcher.XEditV4ConfirmationGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_confirmation_inventory_command_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(
        launcher.XEditV4ConfirmationGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib()
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None
    assert captured.value.command_line == launcher.GPU_INVENTORY_COMMAND


def test_confirmation_inventory_failure_is_persisted_once_before_family_materialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "confirmation.failed.json"
    error = launcher.XEditV4ConfirmationGpuInventoryError(
        "missing GPU 5",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, 100\n",
        missing_physical_gpus=(5,),
    )
    kwargs = {
        "current_head": "a" * 40,
        "experiment_head": "b" * 40,
        "eligible_components": ["setflow"],
        "authorization_root": tmp_path / "authorization",
        "authorization_staging": tmp_path / "authorization.partial",
        "runtime_root": tmp_path / "runtime",
        "log_root": tmp_path / "logs",
        "error": error,
    }
    launcher.write_gpu_inventory_failure_evidence(path, **kwargs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["inventory_failure_reason"] == (
        "PHYSICAL_GPU_INVENTORY_INCOMPLETE"
    )
    assert payload["missing_physical_gpus"] == [5]
    assert payload["scheduler_started"] is False
    assert payload["gpu_job_started"] is False
    assert payload["free_memory_gate_applied"] is False
    with pytest.raises(Exception, match="already exists"):
        launcher.write_gpu_inventory_failure_evidence(path, **kwargs)


def test_confirmation_inventory_precedes_prepare_and_authorization_publish() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    run_source = source[source.index("def run(current_head: str, experiment_head: str)") :]
    failure_guard = run_source.index("not prelaunch_failure_path.exists()")
    inventory = run_source.index("free_memory = gpu_free_memory_mib()")
    prepare = run_source.index("manifests: dict[str, dict[str, Any]] = {}")
    authorization_publish = run_source.index(
        "os.replace(authorization_staging, authorization_root)"
    )
    assert failure_guard < inventory < prepare < authorization_publish
