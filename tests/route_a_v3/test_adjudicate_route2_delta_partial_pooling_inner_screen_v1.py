from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_delta_partial_pooling_inner_screen_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("partial_pooling_adjudication_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict:
    return {
        "schema_version": "route_a_v3_route2_delta_partial_pooling_inner_screen_protocol.v1",
        "status": "FROZEN_BEFORE_TRAIN_INNER_OUTCOMES",
        "single_changed_model_factor": "endpoint_region_residual",
        "expected_result_stage": "TRAIN_INNER_VALIDATION_ONLY",
        "expected_inner_split_id": "INNER_V1",
        "expected_record_counts": {"TRAIN": 70, "VALIDATION": 15},
        "expected_inner_test_record_count_withheld": 15,
        "expected_parent_development_validation_record_count_excluded": 20,
        "expected_parent_development_test_record_count_excluded": 20,
        "expected_task_count": 7,
        "expected_seed": 17,
        "expected_final_epoch": 8,
        "expected_optimizer_steps_per_arm": 24,
        "expected_physical_gpu_index": 6,
        "expected_parameter_counts": {"shared": 100, "residual": 114},
        "material_gain_rule": {
            "minimum_task_macro_spearman_gain_inclusive": 0.01,
            "minimum_tasks_with_strict_spearman_improvement": 4,
            "maximum_task_macro_standardized_mae_ratio_inclusive": 1.02,
        },
        "parent_development_validation_outcomes_accessed": False,
        "parent_development_test_outcomes_accessed": False,
        "inner_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "conditional_confirmation_plan": {
            "status": "PREFROZEN_BEFORE_TRAIN_INNER_VALIDATION_OUTCOMES",
            "activation_requires": "MATERIAL_TRAIN_INNER_GAIN_ESTABLISHED_FOR_SCREEN",
            "result_stage": "TRAIN_INNER_TEST_CONFIRMATION",
            "paired_seeds": [20260817, 20260818, 20260819],
            "arm_count": 2,
            "run_count": 6,
            "training_record_count_after_folding_inner_validation": 76458,
            "inner_test_record_count": 13122,
            "batch_size": 32,
            "epochs": 8,
            "expected_optimizer_steps_per_arm_per_seed": 19120,
            "minimum_seed_count_with_positive_paired_task_macro_spearman_gain": 3,
            "required_seed_count": 3,
            "minimum_mean_paired_task_macro_spearman_gain_inclusive": 0.01,
            "minimum_tasks_with_positive_median_paired_spearman_gain": 4,
            "maximum_mean_task_macro_standardized_mae_ratio_inclusive": 1.02,
            "parent_development_validation_outcomes_accessed": False,
            "parent_development_test_outcomes_accessed": False,
            "inner_test_outcomes_accessed": True,
            "evaluation_outcomes_accessed": False,
            "does_not_authorize": [
                "FORMAL_DEVELOPMENT_VALIDATION",
                "FORMAL_DEVELOPMENT_TEST",
                "EVALUATION",
                "CRITIC_READY_FOR_GUIDANCE",
                "GUIDED_XEDITFLOW",
                "PUBLICATION_SUCCESS_CLAIM",
            ],
        },
        "pass_authorizes": [
            "MATERIALIZE_AND_RUN_PREFROZEN_THREE_SEED_TRAIN_INNER_TEST_CONFIRMATION"
        ],
        "pass_does_not_authorize": ["FORMAL_DEVELOPMENT_VALIDATION", "EVALUATION"],
    }


def _summary(*, residual: bool, macro: float, mae: float, task_values: list[float]) -> dict:
    return {
        "status": "DELTA_PREDICTOR_TRAIN_INNER_GPU_RUN_COMPLETE",
        "scientific_role": "TRAIN_ONLY_PARTIAL_POOLING_MODEL_SELECTION",
        "result_stage": "TRAIN_INNER_VALIDATION_ONLY",
        "inner_split_id": "INNER_V1",
        "record_counts": {"TRAIN": 70, "VALIDATION": 15},
        "inner_test_record_count_withheld": 15,
        "parent_development_validation_record_count_excluded": 20,
        "parent_development_test_record_count_excluded": 20,
        "development_validation_outcomes_evaluated": False,
        "development_test_outcomes_evaluated": False,
        "inner_validation_outcomes_evaluated": True,
        "inner_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "physical_gpu_index": 6,
        "seed": 17,
        "final_training_epoch": 8,
        "selected_epoch": 8,
        "parameter_count": 114 if residual else 100,
        "endpoint_region_residual": residual,
        "optimizer_steps": 24,
        "validation_metrics": {
            "task_count": 7,
            "defined_task_spearman_count": 7,
            "task_macro_spearman": macro,
            "task_macro_standardized_mae": mae,
            "task_metrics": {
                f"task-{index}": {"spearman": value}
                for index, value in enumerate(task_values)
            },
        },
    }


def test_adjudication_requires_material_breadth_and_preserves_claim_boundary() -> None:
    module = _load_module()
    protocol = _protocol()
    shared = _summary(
        residual=False,
        macro=0.10,
        mae=1.0,
        task_values=[0.10] * 7,
    )
    residual = _summary(
        residual=True,
        macro=0.12,
        mae=1.01,
        task_values=[0.12, 0.12, 0.12, 0.12, 0.09, 0.09, 0.09],
    )
    result = module.adjudicate(protocol, shared, residual)
    assert result["material_gain"] is True
    assert result["tasks_with_strict_spearman_improvement_count"] == 4
    assert result["authorizes"] == [
        "MATERIALIZE_AND_RUN_PREFROZEN_THREE_SEED_TRAIN_INNER_TEST_CONFIRMATION"
    ]
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert "FORMAL_DEVELOPMENT_VALIDATION" in result["does_not_authorize"]

    residual["validation_metrics"]["task_metrics"]["task-3"]["spearman"] = 0.09
    rejected = module.adjudicate(protocol, shared, residual)
    assert rejected["material_gain"] is False
    assert rejected["authorizes"] == []


def test_frozen_arm_configs_differ_only_by_residual_factor_and_run_identity() -> None:
    module = _load_module()
    protocol_path = (
        ROOT
        / "configs/route_a_v3_route2_delta_partial_pooling_inner_screen_protocol_v1.json"
    )
    protocol = module.load_json(protocol_path)
    shared = module.load_json(ROOT / protocol["shared_arm_config"])
    residual = module.load_json(ROOT / protocol["residual_arm_config"])
    module.validate_protocol(protocol)
    module.validate_arm_config_pair(protocol, shared, residual)


def test_prefrozen_confirmation_rejects_success_rule_drift() -> None:
    module = _load_module()
    protocol = _protocol()
    protocol["conditional_confirmation_plan"][
        "minimum_mean_paired_task_macro_spearman_gain_inclusive"
    ] = 0.0
    with pytest.raises(
        module.PartialPoolingAdjudicationError,
        match="confirmation success rule changed",
    ):
        module.validate_protocol(protocol)
