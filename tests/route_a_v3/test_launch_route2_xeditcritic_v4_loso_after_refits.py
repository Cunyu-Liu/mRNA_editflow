from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_loso_after_refits as launcher


RUNNER_HEAD = "e" * 40


def _manifest() -> dict[str, object]:
    jobs = [
        {"seed": seed, "held_out_study": study, "run_id": run_id}
        for seed in launcher.SEEDS
        for study in launcher.STUDIES
        for run_id in ("v4_full", "c0_v4")
    ]
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_job_manifest.v1",
        "status": "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(launcher.SEEDS),
        "held_out_studies": list(launcher.STUDIES),
        "runner_git_head": RUNNER_HEAD,
        "job_count": 42,
        "jobs": jobs,
    }


def test_loso_manifest_is_exact_42_paired_jobs() -> None:
    assert len(
        launcher.validate_loso_manifest(_manifest(), expected_head=RUNNER_HEAD)
    ) == 42
    payload = _manifest()
    payload["jobs"] = payload["jobs"][:-1]
    with pytest.raises(Exception, match="job set changed"):
        launcher.validate_loso_manifest(payload, expected_head=RUNNER_HEAD)


def test_loso_manifest_rejects_runner_head_drift() -> None:
    payload = _manifest()
    payload["runner_git_head"] = "f" * 40
    with pytest.raises(Exception, match="runner Git HEAD"):
        launcher.validate_loso_manifest(payload, expected_head=RUNNER_HEAD)


def test_loso_gpu_selection_uses_frozen_protocol_order_without_memory_gate() -> None:
    inventory = {gpu: 1 for gpu in range(6)}
    assert launcher.eligible_loso_gpus((5, 2, 0, 1, 3, 4), inventory) == (
        5, 2, 0, 1, 3, 4
    )
    with pytest.raises(Exception, match="absent"):
        launcher.eligible_loso_gpus((0, 5), {0: 100_000})


def test_loso_launcher_uses_formal_current_head_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.LOSO_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py"
    )


def test_loso_records_memory_without_filtering_or_sorting() -> None:
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
                8,
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
                stdout="0, 100\n1, 100\n2, 100\n3, 100\n4, 100\n",
                stderr="",
            ),
            "PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            (5,),
        ),
    ],
)
def test_loso_inventory_failures_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str],
    reason: str,
    missing: tuple[int, ...],
) -> None:
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(launcher.XEditCriticV4LosoGpuInventoryError) as captured:
        launcher.gpu_free_memory_mib((0, 1, 2, 3, 4, 5))
    assert captured.value.reason == reason
    assert captured.value.return_code == result.returncode
    assert captured.value.stdout == result.stdout
    assert captured.value.stderr == result.stderr
    assert captured.value.missing_physical_gpus == missing


def test_loso_inventory_execution_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("nvidia-smi absent")

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(launcher.XEditCriticV4LosoGpuInventoryError) as captured:
        launcher.gpu_free_memory_mib((0, 1, 2, 3, 4, 5))
    assert captured.value.reason == "COMMAND_EXECUTION_FAILED"
    assert captured.value.return_code is None


def test_loso_inventory_failure_stops_before_runtime_or_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "c" * 40
    worktree = tmp_path / "worktree"
    root = tmp_path / "root"
    python = tmp_path / "python"
    scheduler = tmp_path / "loso_scheduler.py"
    for path in (python, scheduler):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "WORKTREE", worktree)
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "PYTHON", python)
    monkeypatch.setattr(launcher, "LOSO_SCHEDULER", scheduler)
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
    refit_runtime = {
        "status": "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED",
        "git_head": head,
        "adjudication": {"loso_authorized": True},
        "active_performance_output_read": False,
        "development_test_access_event_count_before_refit": 1,
        "development_test_outcome_reads_during_refit": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    refit_runtime_path = root / f"experiments/xedit_v4/refit_execution_{head}/runtime.json"
    refit_runtime_path.parent.mkdir(parents=True, exist_ok=True)
    refit_runtime_path.write_text(json.dumps(refit_runtime), encoding="utf-8")
    refit_manifest = tmp_path / "refit_manifest.json"
    refit_manifest.write_text(
        json.dumps(
            {
                "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
                "required_seeds": list(launcher.SEEDS),
                "completed_refit_count": 3,
                "refit_pass_count": 8,
                "loso_authorized": True,
                "development_test_outcomes_accessed_during_refit": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        ),
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
                    "terminal_manifest_output": str(refit_manifest),
                },
                "test_preserving_loso": {
                    "runtime_config_root": str(tmp_path / "loso_configs"),
                    "run_root": str(tmp_path / "loso_runs"),
                    "adjudication_output": str(tmp_path / "loso_adjudication.json"),
                },
                "readiness_output": str(tmp_path / "readiness.json"),
            }
        ),
        encoding="utf-8",
    )
    inventory_error = launcher.XEditCriticV4LosoGpuInventoryError(
        "missing configured GPU",
        reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
        return_code=0,
        stdout="0, 100\n",
        missing_physical_gpus=(5,),
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

    with pytest.raises(launcher.XEditCriticV4LosoGpuInventoryError):
        launcher.run(head)

    runtime_root = root / f"experiments/xedit_v4/loso_execution_{head}"
    evidence_path = launcher.sibling_failure_path(runtime_root)
    assert not runtime_root.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == (
        "route_a_v3_route2_xeditcritic_prelaunch_failure.v1"
    )
    assert evidence["status"] == "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE"
    assert evidence["launcher"] == "loso"
    assert evidence["failure_stage"] == "INVENTORY"
    assert evidence["runtime_root_created"] is False
    assert evidence["jobs_started"] == 0
    assert evidence["cpu_fallback_used"] is False
    assert evidence["free_memory_gate_applied"] is False
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


@pytest.mark.parametrize("partial", [False, True])
def test_loso_existing_failure_evidence_requires_new_family(
    tmp_path: Path, partial: bool
) -> None:
    runtime_root = tmp_path / "loso_runtime"
    failure = launcher.sibling_failure_path(runtime_root)
    path = failure.with_suffix(failure.suffix + ".partial") if partial else failure
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="new retry family"):
        launcher.require_fresh_prelaunch_family(runtime_root)
