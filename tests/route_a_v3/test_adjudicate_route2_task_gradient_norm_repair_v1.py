from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_task_gradient_norm_repair_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_gradnorm_adjudicator_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(value: float, *, candidate: bool = False, mae: float = 1.8):
    tasks = {f"task-{index}": {"spearman": value} for index in range(9)}
    summary = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "baseline_id": "candidate" if candidate else "reference",
        "model_kind": "delta_edit_centered_antisymmetric",
        "device": "cuda:2",
        "physical_gpu_index": 2,
        "cuda_device_uuid": "GPU-test",
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "optimizer_steps": 10,
        "development_test_outcomes_evaluated": False,
        "evaluation_outcomes_read": 0,
        "target_scaler": {"mode": "TRAIN_TASK_ROBUST"},
        "validation_metrics": {
            "task_count": 9,
            "defined_task_spearman_count": 9,
            "task_macro_spearman": value,
            "task_macro_standardized_mae": mae,
            "task_metrics": tasks,
        },
    }
    if candidate:
        summary["training_update_mode"] = "TRAIN_TASK_GRADIENT_NORM_CALIBRATED"
        summary["task_gradient_calibration"] = {
            "fit_scope": "TRAIN_ONLY_BEFORE_FIRST_OPTIMIZER_STEP",
            "cuda_losses_verified": True,
            "task_count": 7,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "loss_multipliers": {f"train-task-{index}": 1.0 for index in range(7)},
        }
    return summary


def _protocol(tmp_path: Path, candidate_value: float, *, mae: float = 1.8):
    candidate_path = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    scaled_path = tmp_path / "scaled.json"
    candidate_path.write_text(json.dumps(_summary(candidate_value, candidate=True, mae=mae)))
    raw_path.write_text(json.dumps(_summary(0.1)))
    scaled_path.write_text(json.dumps(_summary(0.12)))
    return {
        "schema_version": "route_a_v3_route2_task_gradient_norm_repair_protocol.v1",
        "status": "FROZEN_DEVELOPMENT_ONLY_BEFORE_GRADNORM_ARM_OUTCOME",
        "candidate_output": str(candidate_path),
        "frozen_references": {
            "global_raw_summary": str(raw_path),
            "global_scaled_summary": str(scaled_path),
            "legacy_best_task_macro_spearman": 0.15328703375929223,
            "legacy_best_common_train_robust_task_macro_standardized_mae": 1.822072723037273,
        },
    }


def test_adjudicator_authorizes_controls_but_not_fresh_seeds(tmp_path: Path) -> None:
    module = _load()
    result = module.adjudicate(_protocol(tmp_path, 0.2))
    assert result["status"] == "EXPLORATORY_GRADNORM_SUPPORTS_MATCHED_CONTROLS"
    assert result["matched_controls_authorized"] is True
    assert result["fresh_confirmation_seeds"] == []
    assert result["task_wins_over_global_raw"] == 9


def test_adjudicator_stops_when_legacy_macro_is_not_beaten(tmp_path: Path) -> None:
    module = _load()
    result = module.adjudicate(_protocol(tmp_path, 0.13))
    assert result["status"] == "EXPLORATORY_GRADNORM_NO_GO"
    assert result["advance_checks"]["beats_global_scaled_macro"] is True
    assert result["advance_checks"]["beats_legacy_macro"] is False
    assert result["matched_controls_authorized"] is False


def test_adjudicator_rejects_non_cuda_calibration(tmp_path: Path) -> None:
    module = _load()
    protocol = _protocol(tmp_path, 0.2)
    candidate_path = Path(protocol["candidate_output"])
    candidate = json.loads(candidate_path.read_text())
    candidate["task_gradient_calibration"]["cuda_losses_verified"] = False
    candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(module.TaskGradientNormAdjudicationError, match="CUDA"):
        module.adjudicate(protocol)

