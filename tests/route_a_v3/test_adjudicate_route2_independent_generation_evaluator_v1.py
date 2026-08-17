from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_independent_generation_evaluator_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("independent_generation_evaluator_adjudication_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict:
    return {
        "schema_version": "route_a_v3_route2_generation_matched_compute_repair_protocol.v1",
        "guiding_model_kind": "delta_anchored_position_aware_antisymmetric",
        "independent_evaluator_model_kind": "siamese_cnn",
        "guide_evaluator_architecture_distinct": True,
        "independent_evaluator_qualification": {
            "minimum_task_macro_spearman_exclusive": 0.1012475745988908,
            "minimum_positive_task_count": 5,
            "target_scaling_mode": "TRAIN_TASK_ROBUST",
        },
    }


def _summary() -> dict:
    task_metrics = {
        f"task-{index}": {"spearman": 0.2 if index < 6 else -0.1}
        for index in range(9)
    }
    return {
        "schema_version": "route_a_v3_route2_delta_predictor_training.v1",
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "seed": 20260816,
        "baseline_id": "independent_evaluator_neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation",
        "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
        "development_validation_folded_into_training": False,
        "development_test_outcomes_evaluated": False,
        "development_test_record_count_withheld": 18292,
        "test_metrics": None,
        "evaluation_outcomes_read": 0,
        "device": "cuda:6",
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "optimizer_steps": 22120,
        "model_kind": "siamese_cnn",
        "metadata_mode": "FULL_CONTEXT",
        "candidate_control": "NONE",
        "checkpoint_selection": "FINAL_EPOCH",
        "frozen_pretrained_parameter_count": 0,
        "pretrained_model_id": None,
        "target_scaler": {
            "mode": "TRAIN_TASK_ROBUST",
            "fit_scope": "TRAIN_ONLY",
            "center_subtracted": False,
        },
        "parameter_count": 509905,
        "selected_epoch": 8,
        "final_training_epoch": 8,
        "physical_gpu_index": 6,
        "cuda_device_uuid": "GPU-independent",
        "validation_metrics": {
            "task_macro_spearman": 0.13,
            "task_macro_standardized_mae": 1.7,
            "task_count": 9,
            "defined_task_spearman_count": 9,
            "task_metrics": task_metrics,
        },
    }


def test_qualified_evaluator_authorizes_candidate_rerun() -> None:
    module = _load()
    result = module.adjudicate(_summary(), _protocol())
    assert result["status"] == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
    assert result["candidate_rerun_authorized"] is True
    assert result["positive_task_count"] == 6
    assert all(result["checks"].values())


def test_metric_or_exposure_failure_is_no_go() -> None:
    module = _load()
    for mutation in ("metric", "test_exposure", "cpu_fallback"):
        summary = deepcopy(_summary())
        if mutation == "metric":
            summary["validation_metrics"]["task_macro_spearman"] = 0.09
        elif mutation == "test_exposure":
            summary["result_stage"] = "FROZEN_DEVELOPMENT_TEST"
            summary["development_validation_folded_into_training"] = True
        else:
            summary["cpu_fallback_used"] = True
        result = module.adjudicate(summary, _protocol())
        assert result["status"] == "INDEPENDENT_GENERATION_EVALUATOR_NO_GO"
        assert result["candidate_rerun_authorized"] is False


def test_mrnabert_generation_evaluator_protocol_binds_distinct_model_identity() -> None:
    module = _load()
    protocol = _protocol()
    protocol.update({
        "schema_version": "route_a_v3_route2_mrnabert_independent_evaluator_qualification.v1",
        "guiding_model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "independent_evaluator_baseline_id": "independent_evaluator_neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation",
        "seed": 20260816,
    })
    result = module.adjudicate(_summary(), protocol)
    assert result["status"] == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
    assert result["checks"]["exact_frozen_evaluator_identity"] is True
    summary = _summary()
    summary["pretrained_model_id"] = "YYLY66/mRNABERT"
    summary["frozen_pretrained_parameter_count"] = 113_389_056
    result = module.adjudicate(summary, protocol)
    assert result["status"] == "INDEPENDENT_GENERATION_EVALUATOR_NO_GO"
