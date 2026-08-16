from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/predict_route2_frozen_neural_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("predict_route2_frozen_neural_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_loader_uses_sequences_and_context_without_target_field(tmp_path: Path) -> None:
    module = _load()
    canonical = tmp_path / "evaluation.jsonl"
    canonical.write_text(json.dumps({
        "canonical_record_id": "e", "pool_assignment": "EVALUATION", "training_eligible": False,
        "source_sequence": "AAAA", "candidate_sequence": "CAAA", "study_unit_id": "E",
        "assay_id": "A", "biological_context_id": "C", "endpoint_id": "END", "region": "3UTR",
        "direction_normalized_delta": 999.0,
    }) + "\n", encoding="utf-8")
    rows = module.load_prediction_records([canonical], {"e"})
    assert rows == [module.PredictionRecord("e", "AAAA", "CAAA", "E", "A", "C", "END", 1)]
    assert "target" not in rows[0].__dict__


def test_manifest_requires_evaluation_zero_shot(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({
        "canonical_record_id": "e", "pool_assignment": "EVALUATION", "split": "EVALUATION_ZERO_SHOT"
    }) + "\n", encoding="utf-8")
    assert module.load_evaluation_manifest(path) == {"e"}
    path.write_text(json.dumps({
        "canonical_record_id": "e", "pool_assignment": "DEVELOPMENT", "split": "TEST"
    }) + "\n", encoding="utf-8")
    with pytest.raises(module.FrozenNeuralPredictionError, match="Development row"):
        module.load_evaluation_manifest(path)


def test_checkpoint_must_be_frozen_with_observed_cuda_provenance() -> None:
    module = _load()
    provenance = {
        "result_stage": "FROZEN_DEVELOPMENT_TEST", "optimizer_steps": 10,
        "parameter_changed": True, "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False, "physical_gpu_index": 2, "device": "cuda:2",
        "cuda_device_index": 2, "cuda_device_uuid": "GPU-frozen", "cuda_total_memory_mb": 40960.0,
    }
    module.validate_frozen_checkpoint_provenance(provenance)
    invalid = dict(provenance, result_stage="HPO_VALIDATION_ONLY")
    with pytest.raises(module.FrozenNeuralPredictionError, match="frozen learned GPU"):
        module.validate_frozen_checkpoint_provenance(invalid)


def test_prediction_rows_reserve_explicit_baseline_binding() -> None:
    module = _load()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for frozen neural prediction binding test")
    device = torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")

    class Stub(torch.nn.Module):
        def forward(self, source, candidate, padding, *categories):
            return {"mean": torch.zeros(len(source), device=source.device)}

    record = module.PredictionRecord("e", "AAAA", "CAAA", "S", "A", "C", "E", 1)
    rows = module.predict_records(
        Stub(), [record], {field: {"__UNK__": 0} for field in ("study", "assay", "context", "endpoint")},
        device, 1, "frozen_neural",
    )
    assert rows[0]["baseline_id"] == "frozen_neural"


def test_frozen_neural_checkpoint_identity_must_match_selection() -> None:
    module = _load()
    module.validate_checkpoint_identity({"baseline_id": "delta_main"}, "delta_main")
    with pytest.raises(module.FrozenNeuralPredictionError, match="checkpoint identity"):
        module.validate_checkpoint_identity({"baseline_id": "delta_main"}, "full_pair_2m")
