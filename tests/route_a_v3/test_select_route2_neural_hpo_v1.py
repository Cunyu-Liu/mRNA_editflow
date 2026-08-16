from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/select_route2_neural_hpo_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("neural_hpo_selector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trial(tmp_path: Path, trial_id: str, spearman: float | None, *, cpu_fallback: bool = False) -> dict:
    summary_path = tmp_path / f"{trial_id}.summary.json"
    evaluation_path = tmp_path / f"{trial_id}.evaluation.json"
    config_path = tmp_path / f"{trial_id}.config.json"
    summary_path.write_text(json.dumps({
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "HPO_VALIDATION_ONLY",
        "development_test_outcomes_evaluated": False,
        "test_metrics": None,
        "evaluation_outcomes_read": 0,
        "cpu_fallback_used": cpu_fallback,
        "cuda_training_tensors_verified": True,
        "parameter_changed": True,
        "optimizer_steps": 10,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "cuda_device_index": 2,
        "cuda_device_uuid": "GPU-2",
        "cuda_total_memory_mb": 40960.0,
        "baseline_id": trial_id,
        "model_kind": "candidate_cnn",
        "parameter_count": 500000,
    }), encoding="utf-8")
    evaluation_path.write_text(json.dumps({
        "split": "VALIDATION",
        "evaluation_release_state": "CLOSED",
        "metrics": {
            "task_count": 2,
            "task_spearman_defined_count": 2 if spearman is not None else 1,
            "task_macro_spearman": spearman,
            "task_numeric": {
                "A": {"spearman": spearman},
                "B": {"spearman": spearman if spearman is not None else 0.0},
            },
            "source_macro_mae": 0.2,
        },
    }), encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")
    return {
        "trial_id": trial_id,
        "profile_id": "P",
        "training_summary_path": str(summary_path),
        "validation_evaluation_path": str(evaluation_path),
        "training_config_path": str(config_path),
    }


def test_selects_higher_task_macro_spearman_without_test_or_evaluation(tmp_path: Path) -> None:
    module = _load()
    result = module.select({
        "schema_version": "route_a_v3_route2_neural_hpo_selection_config.v1",
        "selection_pool": "DEVELOPMENT_VALIDATION",
        "evaluation_outcomes_accessed": False,
        "expected_trials_per_profile": 2,
        "trials": [_trial(tmp_path, "low", 0.1), _trial(tmp_path, "high", 0.2)],
    })
    assert result["selections"]["P"]["selected_trial_id"] == "high"
    assert result["development_test_outcomes_accessed"] is False
    assert result["evaluation_outcomes_accessed"] is False


def test_cpu_fallback_trial_is_rejected(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(module.HpoSelectionError, match="CPU fallback"):
        module.select({
            "schema_version": "route_a_v3_route2_neural_hpo_selection_config.v1",
            "selection_pool": "DEVELOPMENT_VALIDATION",
            "evaluation_outcomes_accessed": False,
            "expected_trials_per_profile": 2,
            "trials": [_trial(tmp_path, "invalid", 0.2, cpu_fallback=True), _trial(tmp_path, "valid", 0.1)],
        })


def test_undefined_task_spearman_is_preserved_but_ranked_behind_complete_trial(tmp_path: Path) -> None:
    module = _load()
    result = module.select({
        "schema_version": "route_a_v3_route2_neural_hpo_selection_config.v1",
        "selection_pool": "DEVELOPMENT_VALIDATION",
        "evaluation_outcomes_accessed": False,
        "expected_trials_per_profile": 2,
        "trials": [_trial(tmp_path, "collapsed", None), _trial(tmp_path, "complete", -0.1)],
    })
    assert result["selections"]["P"]["selected_trial_id"] == "complete"
    ranked = result["selections"]["P"]["all_trials_ranked"]
    assert ranked[1]["trial_id"] == "collapsed"
    assert ranked[1]["task_macro_spearman"] is None
