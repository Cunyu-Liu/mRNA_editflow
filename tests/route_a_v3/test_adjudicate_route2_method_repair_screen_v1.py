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
    return {
        "scientific_role": role,
        "baseline_id": role.lower(),
        "model_kind": "model",
        "target_scaling_mode": "NONE",
        "candidate_control": "NONE",
        "parameter_count": 500_000,
        "selected_epoch": 1,
        "task_macro_spearman": spearman,
        "task_macro_standardized_mae": mae,
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
        "cuda_device_uuid": "uuid",
        "validation_metrics": {
            "task_macro_spearman": 0.1,
            "task_macro_standardized_mae": 1.0,
            "defined_task_spearman_count": 8,
            "task_count": 9,
        },
    }
    with pytest.raises(module.MethodRepairScreenError, match="CPU fallback"):
        module.validate_run(config, summary)
    summary["cpu_fallback_used"] = False
    with pytest.raises(module.MethodRepairScreenError, match="undefined"):
        module.validate_run(config, summary)
