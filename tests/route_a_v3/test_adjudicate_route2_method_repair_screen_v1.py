import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_method_repair_screen_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_method_repair_adjudicator_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(role, spearman, mae=1.0):
    is_global = role.startswith("FACTORIAL_GLOBAL")
    is_source_only = role == "MATCHED_SOURCE_ONLY_CONTROL"
    is_permutation = role == "MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL"
    is_scaled = role.endswith("SCALED") or is_source_only or is_permutation
    return {
        "scientific_role": role,
        "baseline_id": role.lower(),
        "model_kind": "global" if is_global else ("source_only" if is_source_only else "edit"),
        "target_scaling_mode": "TRAIN_TASK_ROBUST" if is_scaled else "NONE",
        "candidate_control": "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION" if is_permutation else "NONE",
        "parameter_count": 500_000,
        "selected_epoch": 1,
        "task_macro_spearman": spearman,
        "within_run_task_macro_standardized_mae": mae,
        "raw_task_mae_by_task": {"endpoint::region=0": mae * 2.0},
        "target_scaler": {
            "mode": "TRAIN_TASK_ROBUST" if is_scaled else "NONE",
            "task_scales": {"endpoint::region=0": 2.0} if is_scaled else {},
            "region_scales": {"region=0": 2.0} if is_scaled else {},
            "global_scale": 2.0 if is_scaled else 1.0,
        },
        "validation_task_count": 9,
        "physical_gpu_index": 0,
        "cuda_device_uuid": "uuid",
        "cpu_fallback_used": False,
        "evaluation_outcomes_read": 0,
        "output_directory": "/mnt/run",
    }


def _protocol():
    return {
        "status": "FROZEN_DEVELOPMENT_ONLY_EXPLORATORY_SCREEN",
        "fresh_confirmation_seeds": [2, 3, 4],
        "guided_generation_status": "BLOCKED_UNTIL_CRITIC_READY_AND_INDEPENDENT_EVALUATOR_FIXED",
        "post_exposure_boundary": {"new_external_confirmation_required_for_new_method_claim": True},
    }


def test_screen_supports_confirmation_only_when_selected_edit_model_beats_controls() -> None:
    module = _load()
    runs = [
        _run("FACTORIAL_GLOBAL_RAW", 0.10),
        _run("FACTORIAL_GLOBAL_SCALED", 0.12),
        _run("FACTORIAL_EDIT_CENTERED_RAW", 0.13),
        _run("FACTORIAL_EDIT_CENTERED_SCALED", 0.20),
        _run("MATCHED_SOURCE_ONLY_CONTROL", 0.08),
        _run("MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL", 0.09),
    ]
    result = module.adjudicate_screen(_protocol(), runs)
    assert result["status"] == "EXPLORATORY_SCREEN_SUPPORTS_FRESH_SEED_CONFIRMATION"
    assert result["selected_role"] == "FACTORIAL_EDIT_CENTERED_SCALED"
    assert result["task_macro_spearman_improvement_over_global_raw"] == pytest.approx(0.10)
    assert result["selected_common_train_robust_task_macro_standardized_mae"] == pytest.approx(1.0)
    assert result["edit_centered_control_margins"]["over_source_only"] == pytest.approx(0.12)
    assert result["fresh_confirmation_seeds"] == [2, 3, 4]
    assert result["evaluation_used_for_selection"] is False


def test_screen_stops_when_candidate_control_matches_selected_edit_model() -> None:
    module = _load()
    runs = [
        _run("FACTORIAL_GLOBAL_RAW", 0.10),
        _run("FACTORIAL_GLOBAL_SCALED", 0.12),
        _run("FACTORIAL_EDIT_CENTERED_RAW", 0.13),
        _run("FACTORIAL_EDIT_CENTERED_SCALED", 0.20),
        _run("MATCHED_SOURCE_ONLY_CONTROL", 0.08),
        _run("MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL", 0.21),
    ]
    result = module.adjudicate_screen(_protocol(), runs)
    assert result["status"] == "EXPLORATORY_SCREEN_DOES_NOT_SUPPORT_CONFIRMATION"
    assert result["fresh_confirmation_seeds"] == []


def test_global_winner_requires_architecture_matched_controls_before_confirmation() -> None:
    module = _load()
    runs = [
        _run("FACTORIAL_GLOBAL_RAW", 0.10),
        _run("FACTORIAL_GLOBAL_SCALED", 0.22),
        _run("FACTORIAL_EDIT_CENTERED_RAW", 0.13),
        _run("FACTORIAL_EDIT_CENTERED_SCALED", 0.20),
        _run("MATCHED_SOURCE_ONLY_CONTROL", 0.08),
        _run("MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL", 0.09),
    ]
    result = module.adjudicate_screen(_protocol(), runs)
    assert result["status"] == "EXPLORATORY_GLOBAL_REPAIR_REQUIRES_MATCHED_CONTROLS"
    assert result["matched_controls_support_selected_edit_model"] is False
    assert result["fresh_confirmation_seeds"] == []


def test_edit_raw_winner_requires_raw_target_controls_before_confirmation() -> None:
    module = _load()
    runs = [
        _run("FACTORIAL_GLOBAL_RAW", 0.10),
        _run("FACTORIAL_GLOBAL_SCALED", 0.12),
        _run("FACTORIAL_EDIT_CENTERED_RAW", 0.23),
        _run("FACTORIAL_EDIT_CENTERED_SCALED", 0.20),
        _run("MATCHED_SOURCE_ONLY_CONTROL", 0.08),
        _run("MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL", 0.09),
    ]
    result = module.adjudicate_screen(_protocol(), runs)
    assert result["status"] == "EXPLORATORY_EDIT_RAW_REQUIRES_MATCHED_CONTROLS"
    assert result["matched_controls_are_for_selected_role"] is False
    assert result["matched_controls_support_selected_edit_model"] is False
    assert result["fresh_confirmation_seeds"] == []


def test_screen_rejects_parameter_mismatched_control() -> None:
    module = _load()
    runs = [
        _run("FACTORIAL_GLOBAL_RAW", 0.10),
        _run("FACTORIAL_GLOBAL_SCALED", 0.12),
        _run("FACTORIAL_EDIT_CENTERED_RAW", 0.13),
        _run("FACTORIAL_EDIT_CENTERED_SCALED", 0.20),
        _run("MATCHED_SOURCE_ONLY_CONTROL", 0.08),
        _run("MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL", 0.09),
    ]
    runs[-1]["parameter_count"] += 1
    with pytest.raises(module.MethodRepairScreenError, match="parameter counts differ"):
        module.adjudicate_screen(_protocol(), runs)


def test_validate_run_rejects_undefined_task_or_cpu_fallback() -> None:
    module = _load()
    config = {
        "baseline_id": "x", "model_kind": "m", "seed": 1,
        "device": "cuda:0", "physical_gpu_index": 0,
        "scientific_role": "FACTORIAL_GLOBAL_RAW", "output_directory": "/mnt/x",
    }
    summary = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "HPO_VALIDATION_ONLY",
        "development_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "cpu_fallback_used": True,
        "cuda_training_tensors_verified": True,
        "parameter_changed": True,
        "baseline_id": "x", "model_kind": "m", "seed": 1,
        "device": "cuda:0", "physical_gpu_index": 0,
        "checkpoint_selection": "BEST_VALIDATION",
        "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "target_scaler": {"fit_scope": "TRAIN_ONLY", "center_subtracted": False, "mode": "NONE"},
        "candidate_control": "NONE", "parameter_count": 1, "selected_epoch": 1,
        "candidate_control_summary": {
            "permutation_stratum": "NONE",
            "candidate_pool_membership_preserved": True,
            "edit_distance_multiset_preserved": True,
            "changed_candidate_sequence_count": 0,
        },
        "cuda_device_uuid": "uuid",
        "validation_metrics": {
            "task_macro_spearman": 0.1,
            "task_macro_standardized_mae": 1.0,
            "defined_task_spearman_count": 8,
            "task_count": 9,
            "task_metrics": {
                f"endpoint_{index}::region=0": {"mae": 1.0}
                for index in range(9)
            },
        },
    }
    with pytest.raises(module.MethodRepairScreenError, match="CPU fallback"):
        module.validate_run(config, summary)
    summary["cpu_fallback_used"] = False
    with pytest.raises(module.MethodRepairScreenError, match="undefined"):
        module.validate_run(config, summary)


def test_validate_run_rejects_permutation_that_leaves_exact_source_support() -> None:
    module = _load()
    config = {
        "baseline_id": "permutation", "model_kind": "edit", "seed": 1,
        "device": "cuda:2", "physical_gpu_index": 2,
        "scientific_role": "MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL",
        "output_directory": "/mnt/permutation",
    }
    summary = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "HPO_VALIDATION_ONLY",
        "development_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "parameter_changed": True,
        "baseline_id": "permutation", "model_kind": "edit", "seed": 1,
        "device": "cuda:2", "physical_gpu_index": 2,
        "checkpoint_selection": "BEST_VALIDATION",
        "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "target_scaler": {
            "fit_scope": "TRAIN_ONLY", "center_subtracted": False,
            "mode": "TRAIN_TASK_ROBUST",
        },
        "candidate_control": "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION",
        "candidate_control_summary": {
            "permutation_stratum": "EXACT_SOURCE_SEQUENCE_ENDPOINT_REGION",
            "candidate_pool_membership_preserved": False,
            "edit_distance_multiset_preserved": True,
            "changed_candidate_sequence_count": 10,
        },
        "parameter_count": 1, "selected_epoch": 1,
        "cuda_device_uuid": "uuid",
        "validation_metrics": {
            "task_macro_spearman": 0.1,
            "task_macro_standardized_mae": 1.0,
            "defined_task_spearman_count": 1,
            "task_count": 1,
            "task_metrics": {"endpoint::region=0": {"mae": 1.0}},
        },
    }
    with pytest.raises(module.MethodRepairScreenError, match="left recipient candidate support"):
        module.validate_run(config, summary)
