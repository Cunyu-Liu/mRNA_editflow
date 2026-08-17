from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/score_route2_generation_independent_evaluator_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("score_route2_independent_evaluator_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_independent_scoring_caches_source_and_duplicate_candidates() -> None:
    module = _load()
    sources = {
        "S": {
            "source_key": "S", "source_sequence": "AAAA", "region": "5UTR",
            "study_unit_id": "STUDY", "assay_id": "A", "biological_context_id": "C", "endpoint_id": "E",
        }
    }
    candidates = [
        {"method_id": "beam", "source_key": "S", "candidate_sequence": "CAAA", "independent_evaluator_forwards": 0},
        {"method_id": "beam", "source_key": "S", "candidate_sequence": "CAAA", "independent_evaluator_forwards": 0},
        {"method_id": "beam", "source_key": "S", "candidate_sequence": "AAAA", "independent_evaluator_forwards": 0},
    ]
    calls = []

    def score(_source, candidate):
        calls.append(candidate)
        value = float(candidate.count("C"))
        return module.EvaluatorScore(
            standardized=value,
            raw=value * 2.0,
            target_scale=2.0,
            target_scale_source="TASK",
        )

    rows, forwards = module.augment_candidates(sources, candidates, score)
    assert calls == ["AAAA", "CAAA"]
    assert forwards == {"S": 2}
    assert sum(row["independent_evaluator_forwards"] for row in rows) == 2
    assert [row["independent_evaluator_score"] for row in rows] == [1.0, 1.0, 0.0]
    assert all(row["source_independent_evaluator_score"] == 0.0 for row in rows)
    assert [row["independent_evaluator_raw_score"] for row in rows] == [2.0, 2.0, 0.0]
    assert all(row["independent_evaluator_target_scale"] == 2.0 for row in rows)


def test_independent_evaluator_requires_frozen_observed_cuda_provenance() -> None:
    module = _load()
    provenance = {
        "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "cuda_device_index": 2,
        "cuda_device_uuid": "GPU-independent",
        "cuda_total_memory_mb": 40960.0,
    }
    module.validate_frozen_evaluator_provenance(provenance)
    for field, value in (
        ("result_stage", "HPO_VALIDATION_ONLY"),
        ("result_stage", "FROZEN_DEVELOPMENT_TEST"),
        ("cuda_device_index", 1),
    ):
        invalid = dict(provenance)
        invalid[field] = value
        with pytest.raises(module.IndependentEvaluatorError, match="TRAIN-only frozen"):
            module.validate_frozen_evaluator_provenance(invalid)
