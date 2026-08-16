from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import route2_gpu_failure_evidence as gpu_evidence
from core.route2_gpu_failure_evidence import write_gpu_failure_evidence


def test_declared_physical_index_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_evidence.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_evidence.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        gpu_evidence.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="MIG", uuid="PARENT-A"),
    )
    monkeypatch.setattr(gpu_evidence.torch.cuda, "mem_get_info", lambda _index: (100, 200))
    monkeypatch.setattr(gpu_evidence, "_physical_gpu_uuid", lambda _index: "PARENT-B")
    with pytest.raises(RuntimeError, match="does not belong to physical GPU 7"):
        gpu_evidence.cuda_device_observation(7, require_physical_index_match=True)


def test_matching_cuda_parent_and_physical_index_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_evidence.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_evidence.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        gpu_evidence.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="GPU", uuid="PARENT-A"),
    )
    monkeypatch.setattr(gpu_evidence.torch.cuda, "mem_get_info", lambda _index: (100, 200))
    monkeypatch.setattr(gpu_evidence, "_physical_gpu_uuid", lambda _index: "PARENT-A")
    observed = gpu_evidence.cuda_device_observation(2, require_physical_index_match=True)
    assert observed["cuda_parent_uuid_matches_declared_physical_index"] is True
    assert observed["declared_physical_gpu_uuid"] == "PARENT-A"


def test_gpu_failure_evidence_records_stop_without_cpu_fallback(tmp_path: Path) -> None:
    path = tmp_path / "run.failed.json"
    write_gpu_failure_evidence(
        path,
        {"device": "cuda:6", "physical_gpu_index": 6},
        RuntimeError("CUDA allocation failed"),
        entrypoint="test",
        evaluation_outcomes_accessed=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "STOPPED_WITH_EVIDENCE"
    assert payload["requested_device"] == "cuda:6"
    assert payload["physical_gpu_index"] == 6
    assert payload["cpu_fallback_used"] is False
    assert payload["evaluation_outcomes_accessed"] is False
    assert payload["error_type"] == "RuntimeError"
    observation = payload["requested_cuda_observation"]
    assert observation["cuda_device_index"] == 6
    assert "cuda_total_memory_mb" in observation
    assert "cuda_device_uuid" in observation


def test_failure_evidence_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "run.failed.json"
    path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_gpu_failure_evidence(
            path, {}, RuntimeError("new"), entrypoint="test", evaluation_outcomes_accessed=False
        )
    assert path.read_text(encoding="utf-8") == "existing\n"


def test_all_cuda_entrypoints_write_failure_evidence_and_check_physical_identity() -> None:
    root = Path(__file__).resolve().parents[2]
    entrypoints = (
        "train_route2_delta_predictor_v1.py",
        "train_route2_base_flow_g0_v1.py",
        "run_route2_base_flow_g0_validation_v1.py",
        "run_route2_external_prediction_baselines_v1.py",
        "run_route2_utrlm_baseline_v1.py",
        "run_route2_aparent_baseline_v1.py",
        "run_route2_search_generation_baselines_v1.py",
        "score_route2_generation_independent_evaluator_v1.py",
        "run_route2_classical_prediction_baselines_v1.py",
        "predict_route2_frozen_classical_v1.py",
        "predict_route2_frozen_neural_v1.py",
    )
    for filename in entrypoints:
        source = (root / "scripts/route_a_v3" / filename).read_text(encoding="utf-8")
        assert "write_gpu_failure_evidence(" in source, filename
        assert "require_physical_index_match=True" in source, filename
