from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from scripts.execution import launch_gpu_run


ROOT = Path(__file__).resolve().parents[1]


def _failed_health() -> dict:
    return {
        "checked_at_utc": "2026-07-28T00:00:00+00:00",
        **{field: False for field in launch_gpu_run.HEALTH_FIELDS},
        "device": None,
        "framework_version": "test",
        "max_memory_allocated_bytes": 0,
        "error": "RuntimeError: CUDA_UNAVAILABLE",
        "passed": False,
    }


def _passed_health() -> dict:
    return {
        "checked_at_utc": "2026-07-28T00:00:00+00:00",
        **{field: True for field in launch_gpu_run.HEALTH_FIELDS},
        "device": "cuda",
        "framework_version": "test",
        "max_memory_allocated_bytes": 1,
        "error": None,
        "passed": True,
    }


def test_contract_forbids_cpu_fallback():
    contract = yaml.safe_load(
        (ROOT / "configs/utr_editflow_execution_policy.yaml").read_text(encoding="utf-8")
    )
    assert contract["training"]["formal_neural_device"] == "cuda"
    assert contract["training"]["cpu_fallback_allowed"] is False
    assert contract["training"]["failure_state_on_cuda_violation"] == (
        "FAILED_WITH_EVIDENCE"
    )


def test_failed_cuda_probe_never_executes_command_and_preserves_evidence(
    tmp_path, monkeypatch
):
    marker = tmp_path / "command-was-executed"
    monkeypatch.setattr(launch_gpu_run, "probe_cuda", lambda torch_module=None: _failed_health())
    rc = launch_gpu_run.launch(
        tmp_path,
        ["python", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
    )
    assert rc != 0
    assert not marker.exists()
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (tmp_path / "failure/failure.json").read_text(encoding="utf-8")
    )
    health = json.loads(
        (tmp_path / "logs/cuda_health.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "FAILED_WITH_EVIDENCE"
    assert status["automatic_cpu_fallback"] is False
    assert failure["status"]["stop_reason"] == "CUDA_HEALTH_CHECK_FAILED"
    assert health["passed"] is False


def test_zero_exit_without_real_training_cuda_health_fails_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launch_gpu_run, "probe_cuda", lambda torch_module=None: _passed_health())
    rc = launch_gpu_run.launch(
        tmp_path / "run",
        [sys.executable, "-c", "print('command completed without health evidence')"],
        project_root=tmp_path,
    )
    assert rc != 0
    status = json.loads(
        (tmp_path / "run/status.json").read_text(encoding="utf-8")
    )
    failure = json.loads(
        (tmp_path / "run/failure/failure.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "FAILED_WITH_EVIDENCE"
    assert status["stop_reason"] == "REAL_TRAINING_CUDA_HEALTH_FAILED"
    assert failure["cuda_health"]["error"] == "MISSING_REAL_TRAINING_CUDA_HEALTH"
