from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/predict_route2_frozen_classical_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("predict_route2_frozen_classical_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_loader_never_materializes_outcomes(tmp_path: Path) -> None:
    module = _load()
    canonical = tmp_path / "evaluation.jsonl"
    canonical.write_text(json.dumps({
        "canonical_record_id": "e", "pool_assignment": "EVALUATION", "training_eligible": False,
        "source_sequence": "AAAU", "candidate_sequence": "CAAU", "study_unit_id": "E",
        "assay_id": "A", "biological_context_id": "C", "endpoint_id": "END", "region": "3UTR",
        "source_id": "S", "edit_operations": [{"type": "SUB", "position_zero_based": 0, "ref": "A", "alt": "C"}],
        "direction_normalized_delta": 999.0, "source_endpoint_value": 123.0,
        "candidate_endpoint_value": 456.0,
    }) + "\n", encoding="utf-8")
    rows = module.load_prediction_records([canonical], {"e"})
    assert len(rows) == 1
    assert rows[0]["source_sequence"] == "AAAT"
    assert rows[0]["candidate_sequence"] == "CAAT"
    assert "target" not in rows[0]
    assert rows[0]["source_endpoint_value"] is None
    assert rows[0]["candidate_endpoint_value"] is None


def test_frozen_result_requires_cuda_fit_and_no_evaluation_access() -> None:
    module = _load()
    result = {
        "status": "COMPLETED_FROZEN_DEVELOPMENT_TEST",
        "evaluation_outcomes_accessed": False,
        "execution_provenance": {
            "parameter_fit_execution": "CUDA", "prediction_execution": "CUDA",
            "cpu_fallback_used": False, "physical_gpu_index": 2, "device": "cuda:2",
            "cuda_device_index": 2, "cuda_device_uuid": "GPU-frozen", "cuda_total_memory_mb": 40960.0,
        },
    }
    module.validate_frozen_result(result)
    invalid = dict(result, status="COMPLETED_DEVELOPMENT_VALIDATION_ONLY")
    with pytest.raises(module.FrozenClassicalPredictionError, match="frozen CUDA fit"):
        module.validate_frozen_result(invalid)


def test_manifest_rejects_development_rows(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({
        "canonical_record_id": "e", "pool_assignment": "DEVELOPMENT", "split": "TEST"
    }) + "\n", encoding="utf-8")
    with pytest.raises(module.FrozenClassicalPredictionError, match="Development row"):
        module.load_evaluation_manifest(path)


def test_frozen_artifact_identity_must_match_selected_baseline() -> None:
    module = _load()
    artifact = {"artifact_baseline_id": "ridge", "artifact_baseline_kind": "ridge"}
    module.validate_artifact_identity(artifact, "ridge", "ridge")
    with pytest.raises(module.FrozenClassicalPredictionError, match="artifact identity"):
        module.validate_artifact_identity(artifact, "xgboost", "xgboost")
