from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation as launcher


def _posttraining(*, eligible: bool, terminal: str = "SUMMARY") -> dict[str, object]:
    return {
        "eligible_components": ["critic"] if eligible else ["setflow"],
        "adjudications": {
            "critic": {"terminal_artifact_kind": terminal}
        },
    }


def _gate(status: str) -> dict[str, object]:
    passed = status == "XEDITCRITIC_V4_THREE_SEED_PASS"
    return {
        "status": status,
        "required_seeds": [20260908, 20260909, 20260910],
        "development_test_authorized": passed,
        "atomic_development_test_only": passed,
        "additional_seed_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_atomic_test_decision_launches_only_exact_three_seed_pass() -> None:
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True), _gate("XEDITCRITIC_V4_THREE_SEED_PASS")
    ) == "LAUNCH_EXACT_ATOMIC_TEST"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True), _gate("XEDITCRITIC_V4_THREE_SEED_NO_GO")
    ) == "NOT_AUTHORIZED_CRITIC_THREE_SEED_NO_GO"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=False), None
    ) == "NOT_AUTHORIZED_CRITIC_SCREEN_NO_GO"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True, terminal="FAILURE"), None
    ) == "NOT_AUTHORIZED_CRITIC_CONFIRMATION_TECHNICAL_FAILURE"


def test_atomic_test_decision_rejects_relaxed_authorization() -> None:
    gate = _gate("XEDITCRITIC_V4_THREE_SEED_PASS")
    gate["additional_seed_authorized"] = True
    with pytest.raises(Exception, match="authorization changed"):
        launcher.atomic_test_decision(_posttraining(eligible=True), gate)


def test_atomic_test_gpu_selection_uses_frozen_protocol_without_memory_gate() -> None:
    inventory = {gpu: 1 for gpu in range(6)}
    assert launcher.select_gpu(4, inventory) == 4
    with pytest.raises(Exception, match="absent"):
        launcher.select_gpu(4, {0: 100_000})


def test_atomic_test_launcher_uses_formal_current_head_job_runner() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.JOB_RUNNER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test_job.py"
    )


def test_atomic_test_records_memory_without_filtering_or_sorting() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"diagnostic_peak_plus_two_gib_mib"' in source
    assert "key=lambda gpu" not in source


@pytest.mark.parametrize(
    ("result", "reason", "missing"),
    [
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                9,
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
                stdout="broken\n",
                stderr="",
            ),
            "OUTPUT_PARSE_FAILED",
            (),
        ),
        (
            subprocess.CompletedProcess(
                launcher.GPU_INVENTORY_COMMAND,
                0,
                stdout="0, 100\n1, 100\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (4,),
        ),
    ],
)
def test_atomic_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(
        launcher.XEditCriticV4AtomicTestGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib((4,))
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_atomic_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(
        launcher.XEditCriticV4AtomicTestGpuInventoryError
    ) as captured:
        launcher.gpu_free_memory_mib((4,))
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None
    assert captured.value.command_line == launcher.GPU_INVENTORY_COMMAND


def test_atomic_inventory_failure_stops_before_runtime_or_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "a" * 40
    worktree = tmp_path / "worktree"
    root = tmp_path / "root"
    python = tmp_path / "python"
    job_runner = tmp_path / "job_runner.py"
    atomic_runner = tmp_path / "atomic_runner.py"
    for path in (python, job_runner, atomic_runner):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "WORKTREE", worktree)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "PYTHON", python)
    monkeypatch.setattr(launcher, "JOB_RUNNER", job_runner)
    monkeypatch.setattr(launcher, "ATOMIC_TEST_RUNNER", atomic_runner)
    monkeypatch.setattr(
        launcher,
        "command",
        lambda arguments: subprocess.CompletedProcess(
            arguments,
            0,
            stdout=head + "\n" if arguments[-2:] == ["rev-parse", "HEAD"] else "",
            stderr="",
        ),
    )
    posttraining = {
        "status": "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL",
        "git_head": head,
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "eligible_components": ["critic"],
        "adjudications": {"critic": {"terminal_artifact_kind": "SUMMARY"}},
    }
    posttraining_path = (
        root / f"experiments/xedit_v4/confirmation_posttraining_{head}/runtime.json"
    )
    posttraining_path.parent.mkdir(parents=True, exist_ok=True)
    posttraining_path.write_text(json.dumps(posttraining), encoding="utf-8")
    gate_path = tmp_path / "three_seed_gate.json"
    gate_path.write_text(
        json.dumps(_gate("XEDITCRITIC_V4_THREE_SEED_PASS")),
        encoding="utf-8",
    )
    protocol_path = (
        worktree
        / "configs/route_a_v3_route2_xeditcritic_v4_frozen_test_protocol_v1.json"
    )
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(
            {
                "three_seed_gate_path": str(gate_path),
                "output_directory": str(tmp_path / "atomic_output"),
                "physical_gpu_index": 4,
            }
        ),
        encoding="utf-8",
    )
    preflight = (
        root
        / "experiments/xeditcritic_v4/screen_seed_20260907/"
        "preflight_attempt_5/preflight.json"
    )
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(
        json.dumps({"selected_peak_allocated_gib": 1.0}), encoding="utf-8"
    )
    inventory_error = launcher.XEditCriticV4AtomicTestGpuInventoryError(
        "missing configured GPU",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, 100\n",
        missing_physical_gpus=(4,),
    )
    monkeypatch.setattr(
        launcher,
        "gpu_free_memory_mib",
        lambda required: (_ for _ in ()).throw(inventory_error),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run after inventory failure")
        ),
    )

    with pytest.raises(
        launcher.XEditCriticV4AtomicTestGpuInventoryError
    ):
        launcher.run(head)

    runtime_root = root / f"experiments/xedit_v4/atomic_test_launch_{head}"
    evidence_path = launcher.sibling_failure_path(runtime_root)
    assert not runtime_root.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditcritic_prelaunch_failure.v1"
    )
    assert evidence["status"] == "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE"
    assert evidence["launcher"] == "atomic"
    assert evidence["failure_stage"] == "INVENTORY"
    assert evidence["runtime_root_created"] is False
    assert evidence["jobs_started"] == 0
    assert evidence["cpu_fallback_used"] is False
    assert evidence["free_memory_gate_applied"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_atomic_existing_failure_evidence_requires_new_family(
    tmp_path: Path, partial: bool
) -> None:
    runtime_root = tmp_path / "atomic_runtime"
    failure = launcher.sibling_failure_path(runtime_root)
    path = failure.with_suffix(failure.suffix + ".partial") if partial else failure
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="new retry family"):
        launcher.require_fresh_prelaunch_family(runtime_root)
