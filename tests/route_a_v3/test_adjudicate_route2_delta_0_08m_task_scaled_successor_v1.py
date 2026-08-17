from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/adjudicate_route2_delta_0_08m_task_scaled_successor_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("adjudicate_route2_delta_0_08m_task_scaled_successor_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    protocol = {
        "schema_version": "route_a_v3_route2_delta_0_08m_task_scaled_successor_protocol.v1",
        "successor_baseline_id": "successor",
        "required_model_kind": "model",
        "required_metadata_mode": "FULL_CONTEXT",
        "required_training_weighting_mode": "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "required_target_scaling_mode": "TRAIN_TASK_ROBUST",
        "required_loss_kind": "huber",
        "required_checkpoint_selection": "FINAL_EPOCH",
        "required_final_epoch": 8,
        "required_seed": 1,
        "required_parameter_count": 10,
        "required_train_record_count": 20,
        "required_validation_record_count": 9,
        "required_withheld_test_record_count": 9,
        "required_task_count": 2,
        "minimum_task_macro_spearman_exclusive": 0.2,
        "maximum_task_macro_standardized_mae_inclusive": 1.0,
        "minimum_tasks_improved_over_reference": 2,
        "pass_authorizes": ["SEEDS"],
        "pass_does_not_authorize": ["TEST"],
    }
    reference = {
        "schema_version": "route_a_v3_route2_prediction_evaluation.v1",
        "metrics": {"task_numeric": {
            "5UTR|A": {"spearman": 0.1},
            "3UTR|B": {"spearman": 0.2},
        }},
    }
    summary = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "baseline_id": "successor",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "model_kind": "model",
        "metadata_mode": "FULL_CONTEXT",
        "training_weighting_mode": "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "target_scaler": {"mode": "TRAIN_TASK_ROBUST", "fit_scope": "TRAIN_ONLY", "center_subtracted": False},
        "loss_kind": "huber",
        "checkpoint_selection": "FINAL_EPOCH",
        "selected_epoch": 8,
        "seed": 1,
        "parameter_count": 10,
        "parameter_changed": True,
        "optimizer_steps": 8,
        "device": "cuda:0",
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "record_counts": {"TRAIN": 20, "VALIDATION": 9},
        "development_test_record_count_withheld": 9,
        "development_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "validation_metrics": {
            "task_count": 2,
            "defined_task_spearman_count": 2,
            "task_macro_spearman": 0.3,
            "task_macro_standardized_mae": 0.9,
            "task_metrics": {
                "A::region=0": {"spearman": 0.2},
                "B::region=1": {"spearman": 0.4},
            },
        },
    }
    return summary, reference, protocol


def test_qualifies_only_when_all_pre_frozen_checks_pass() -> None:
    module = _module()
    summary, reference, protocol = _fixture()
    result = module.adjudicate(summary, reference, protocol)
    assert result["qualified"] is True
    assert result["authorized_next_steps"] == ["SEEDS"]
    summary["validation_metrics"]["task_macro_spearman"] = 0.2
    result = module.adjudicate(summary, reference, protocol)
    assert result["qualified"] is False
    assert result["status"] == "SINGLE_FACTOR_TARGET_SCALING_NO_GO"
    assert result["authorized_next_steps"] == []
