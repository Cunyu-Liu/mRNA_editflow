from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/aggregate_route2_loso_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("aggregate_route2_loso_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(study: str, value):
    return {
        "split": f"LOSO::{study}",
        "metrics": {
            "overall_numeric": {"spearman": value},
            "task_count": 1,
            "task_spearman_defined_count": 0 if value is None else 1,
            "task_macro_spearman": value,
        },
    }


def _training(study: str, seed: int = 1):
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "run_mode": "LOSO_FROZEN_HYPERPARAMETERS",
        "result_stage": "LOSO_FROZEN_HYPERPARAMETERS",
        "loso_holdout_study_unit_id": study,
        "seed": seed,
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cuda_training_tensors_verified": True,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "cpu_fallback_used": False,
        "cuda_device_index": 2,
        "cuda_device_uuid": "GPU-fold",
        "cuda_total_memory_mb": 40960.0,
        "evaluation_outcomes_read": 0,
    }


def _payload():
    studies = [f"S{index}" for index in range(7)]
    return {
        "schema_version": "route_a_v3_route2_loso_aggregation_input.v1",
        "seed": 1,
        "development_inventory_studies": studies + ["S_ZERO"],
        "expected_loso_studies": studies,
        "zero_record_development_studies": ["S_ZERO"],
        "model_results": [
            {"study_unit_id": study, "training_summary": _training(study), "evaluation": _evaluation(study, 0.2)}
            for study in studies
        ],
        "baseline_results": [{"study_unit_id": study, "evaluation": _evaluation(study, 0.1)} for study in studies],
    }


def test_loso_macro_keeps_all_nonempty_studies_and_records_zero_study() -> None:
    module = _load()
    result = module.aggregate(_payload())
    assert result["study_count"] == 7
    assert result["development_inventory_study_count"] == 8
    assert result["zero_record_development_studies"] == ["S_ZERO"]
    assert result["model_macro_spearman"] == pytest.approx(0.2)
    assert result["baseline_macro_spearman"] == pytest.approx(0.1)
    assert result["macro_improvement"] == pytest.approx(0.1)
    assert result["all_model_training_gpu_provenance_verified"] is True


def test_undefined_study_spearman_is_not_silently_dropped() -> None:
    module = _load()
    payload = deepcopy(_payload())
    payload["model_results"][3]["evaluation"]["metrics"]["task_macro_spearman"] = None
    payload["model_results"][3]["evaluation"]["metrics"]["task_spearman_defined_count"] = 0
    result = module.aggregate(payload)
    assert result["status"] == "LOSO_MODEL_BASELINE_ALIGNMENT_NOT_ESTABLISHED"
    assert result["model_macro_spearman"] is None
    assert result["baseline_macro_spearman"] is None
    assert result["macro_improvement"] is None
    assert result["aligned_study_count"] == 6
    assert result["undefined_study_count"] == 1
    assert result["per_study"][3]["model_task_macro_spearman"] is None
    assert result["per_study"][3]["baseline_task_macro_spearman"] == pytest.approx(0.1)
    assert result["per_study"][3]["improvement"] is None
    assert result["per_study"][3]["failure_reasons"][0].startswith("MODEL:")


def test_undefined_baseline_spearman_preserves_model_and_failure_role() -> None:
    module = _load()
    payload = deepcopy(_payload())
    payload["baseline_results"][4]["evaluation"] = _evaluation("S4", None)
    result = module.aggregate(payload)
    assert result["status"] == "LOSO_MODEL_BASELINE_ALIGNMENT_NOT_ESTABLISHED"
    assert result["per_study"][4]["model_task_macro_spearman"] == pytest.approx(0.2)
    assert result["per_study"][4]["baseline_task_macro_spearman"] is None
    assert result["per_study"][4]["failure_reasons"][0].startswith("BASELINE:")


def test_zero_record_study_cannot_be_fabricated_as_loso_fold() -> None:
    module = _load()
    payload = _payload()
    payload["expected_loso_studies"].append("S_ZERO")
    payload["zero_record_development_studies"] = []
    with pytest.raises(module.LosoAggregationError, match="study sets differ"):
        module.aggregate(payload)


def test_loso_fold_requires_matching_gpu_training_provenance() -> None:
    module = _load()
    for field, value in (
        ("result_stage", "HPO_VALIDATION_ONLY"),
        ("loso_holdout_study_unit_id", "OTHER"),
        ("cpu_fallback_used", True),
        ("cuda_device_uuid", None),
    ):
        payload = deepcopy(_payload())
        payload["model_results"][0]["training_summary"][field] = value
        with pytest.raises(module.LosoAggregationError, match="training provenance"):
            module.aggregate(payload)
