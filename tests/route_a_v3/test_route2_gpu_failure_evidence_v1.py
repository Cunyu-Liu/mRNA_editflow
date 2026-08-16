from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.route2_gpu_failure_evidence import write_gpu_failure_evidence


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


def test_all_nonclassical_cuda_entrypoints_write_failure_evidence() -> None:
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
    )
    for filename in entrypoints:
        source = (root / "scripts/route_a_v3" / filename).read_text(encoding="utf-8")
        assert "write_gpu_failure_evidence(" in source, filename
