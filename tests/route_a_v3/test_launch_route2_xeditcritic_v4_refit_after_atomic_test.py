from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_refit_after_atomic_test as launcher


def _receipt(status: str) -> dict[str, object]:
    passed = status == "XEDITCRITIC_V4_POSTTEST_AUTHORIZED"
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
        "status": status,
        "required_seeds": [20260908, 20260909, 20260910],
        "frozen_test_gate_status": (
            "XEDITCRITIC_V4_FROZEN_TEST_PASS"
            if passed
            else "XEDITCRITIC_V4_FROZEN_TEST_NO_GO"
        ),
        "all_development_refit_authorized": passed,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "test_bottom_six_cache_persisted": False,
        "development_test_metrics_in_receipt": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_refit_decision_requires_atomic_result_and_passing_receipt() -> None:
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "RESULT"},
        _receipt("XEDITCRITIC_V4_POSTTEST_AUTHORIZED"),
    ) == "LAUNCH_EXACT_THREE_REFITS"
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "RESULT"},
        _receipt("XEDITCRITIC_V4_POSTTEST_NOT_AUTHORIZED"),
    ) == "REFIT_NOT_AUTHORIZED_FROZEN_TEST_NO_GO"
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "FAILURE"}, None
    ) == "REFIT_NOT_AUTHORIZED_ATOMIC_TEST_TECHNICAL_FAILURE"


def test_refit_decision_rejects_receipt_with_test_metrics() -> None:
    receipt = _receipt("XEDITCRITIC_V4_POSTTEST_AUTHORIZED")
    receipt["development_test_metrics_in_receipt"] = True
    with pytest.raises(Exception, match="receipt changed"):
        launcher.refit_decision({"terminal_artifact_kind": "RESULT"}, receipt)


def test_refit_manifest_is_exact_three_seed_full_only() -> None:
    payload = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_job_manifest.v1",
        "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": [20260908, 20260909, 20260910],
        "refit_pass_count": 8,
        "job_count": 3,
        "jobs": [
            {"seed": seed, "run_id": "v4_full"}
            for seed in (20260908, 20260909, 20260910)
        ],
    }
    assert len(launcher.validate_refit_manifest(payload)) == 3
    payload["jobs"].append({"seed": 20260911, "run_id": "v4_full"})
    with pytest.raises(Exception, match="job set changed"):
        launcher.validate_refit_manifest(payload)


def test_refit_launcher_uses_formal_current_head_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.REFIT_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py"
    )


def test_refit_gpu_selection_uses_frozen_protocol_order_without_memory_gate() -> None:
    inventory = {gpu: 1 for gpu in range(6)}
    assert launcher.select_refit_gpus((5, 2, 0, 1, 3, 4), inventory) == (5, 2, 0)
    with pytest.raises(Exception, match="fewer than three"):
        launcher.select_refit_gpus((0, 1), inventory)


def test_refit_records_memory_without_filtering_or_sorting() -> None:
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
                3,
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
            (2,),
        ),
    ],
)
def test_refit_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(launcher.XEditCriticV4RefitGpuInventoryError) as captured:
        launcher.gpu_free_memory_mib((0, 1, 2))
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_refit_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(launcher.XEditCriticV4RefitGpuInventoryError) as captured:
        launcher.gpu_free_memory_mib((0, 1, 2))
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None


def test_refit_inventory_failure_stops_before_runtime_or_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "b" * 40
    worktree = tmp_path / "worktree"
    root = tmp_path / "root"
    python = tmp_path / "python"
    scheduler = tmp_path / "refit_scheduler.py"
    for path in (python, scheduler):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "WORKTREE", worktree)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "PYTHON", python)
    monkeypatch.setattr(launcher, "REFIT_SCHEDULER", scheduler)
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
    output_directory = tmp_path / "atomic_output"
    atomic_runtime = {
        "status": "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TERMINAL",
        "git_head": head,
        "active_performance_output_read": False,
        "terminal_payload_content_read_by_wrapper": False,
        "new_final_evaluation_outcome_reads": 0,
        "terminal_artifact_kind": "RESULT",
        "output_directory": str(output_directory),
    }
    atomic_runtime_path = (
        root / f"experiments/xedit_v4/atomic_test_launch_{head}/runtime.json"
    )
    atomic_runtime_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_runtime_path.write_text(json.dumps(atomic_runtime), encoding="utf-8")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "posttest_authorization_receipt.json").write_text(
        json.dumps(_receipt("XEDITCRITIC_V4_POSTTEST_AUTHORIZED")),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps({"selected_peak_allocated_gib": 1.0}), encoding="utf-8"
    )
    protocol = (
        worktree
        / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
    )
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(
        json.dumps(
            {
                "physical_gpu_indices": [0, 1, 2, 3, 4, 5],
                "formal_preflight_path": str(preflight),
                "all_development_refit": {
                    "runtime_config_root": str(tmp_path / "refit_configs"),
                    "run_root": str(tmp_path / "refit_runs"),
                    "terminal_manifest_output": str(tmp_path / "refit_manifest.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    inventory_error = launcher.XEditCriticV4RefitGpuInventoryError(
        "missing configured GPU",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, 100\n1, 100\n",
        missing_physical_gpus=(2,),
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

    with pytest.raises(launcher.XEditCriticV4RefitGpuInventoryError):
        launcher.run(head)

    runtime_root = root / f"experiments/xedit_v4/refit_execution_{head}"
    evidence_path = launcher.sibling_failure_path(runtime_root)
    assert not runtime_root.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditcritic_prelaunch_failure.v1"
    )
    assert evidence["status"] == "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE"
    assert evidence["launcher"] == "refit"
    assert evidence["failure_stage"] == "INVENTORY"
    assert evidence["runtime_root_created"] is False
    assert evidence["jobs_started"] == 0
    assert evidence["cpu_fallback_used"] is False
    assert evidence["free_memory_gate_applied"] is False
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_refit_existing_failure_evidence_requires_new_family(
    tmp_path: Path, partial: bool
) -> None:
    runtime_root = tmp_path / "refit_runtime"
    failure = launcher.sibling_failure_path(runtime_root)
    path = failure.with_suffix(failure.suffix + ".partial") if partial else failure
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="new retry family"):
        launcher.require_fresh_prelaunch_family(runtime_root)
