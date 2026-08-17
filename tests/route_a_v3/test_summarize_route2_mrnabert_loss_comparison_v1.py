from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/summarize_route2_mrnabert_loss_comparison_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_loss_summary_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, loss: str, macro: float, scaled_mae: float, spread: float) -> Path:
    run = root / loss
    run.mkdir()
    config = {
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "result_stage": "HPO_VALIDATION_ONLY",
        "scientific_role": "DEVELOPMENT_VALIDATION_LOSS_COMPARISON",
        "candidate_control": "NONE",
        "metadata_mode": "FULL_CONTEXT",
        "training_weighting_mode": "STUDY_THEN_SOURCE_CONTEXT_ENDPOINT_GROUP",
        "target_scaling_mode": "TRAIN_TASK_ROBUST",
        "target_scale_floor": 0.001,
        "target_scale_minimum_task_records": 20,
        "hidden_dim": 384,
        "depth": 10,
        "max_length": 164,
        "critic_position_features": "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED",
        "batch_size": 16,
        "seed": 17,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "optimizer_name": "AdamW",
        "checkpoint_selection": "BEST_VALIDATION",
        "checkpoint_metric": "TASK_MACRO_SPEARMAN_THEN_STANDARDIZED_MAE",
        "development_manifest": "/mnt/development.jsonl",
        "development_test_outcomes_accessed": False,
        "evaluation_outcomes_accessed": False,
        "pretrained_feature_cache_path": "/mnt/mrnabert.pt",
        "pretrained_position_encoding": "ALIBI_RELATIVE_BIAS",
        "encoder_attention_backend": "OFFICIAL_PYTORCH_FALLBACK",
        "expected_frozen_pretrained_parameter_count": 113_389_056,
        "frozen_capacity_profile_id": "mrnabert_frozen_113m",
        "canonical_paths": ["/mnt/a.jsonl"],
        "training_precision": "BF16",
        "optimizer_fused": True,
        "torch_compile": False,
        "num_workers": 0,
        "pin_memory": True,
        "non_blocking_transfer": True,
        "huber_delta": 1.0,
        "parameter_count_relative_tolerance": 0.01,
        "loss_kind": loss,
    }
    uncertainty = loss == "learned_variance_gaussian_nll"
    summary = {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "result_stage": "HPO_VALIDATION_ONLY",
        "evaluation_outcomes_read": 0,
        "test_metrics": None,
        "development_test_outcomes_evaluated": False,
        "loss_kind": loss,
        "baseline_id": loss,
        "seed": 17,
        "selected_epoch": 8,
        "optimizer_steps": 800,
        "trainable_parameter_count": 9_000_000 + int(uncertainty),
        "training_precision": "BF16",
        "uncertainty_head_used": uncertainty,
        "validation_metrics": {
            "spearman": macro + 0.1,
            "task_macro_spearman": macro,
            "task_macro_standardized_mae": scaled_mae,
            "prediction_std": 1.0,
            "prediction_std_over_target_std": spread,
            "predicted_standard_deviation_mean": 0.8 if uncertainty else None,
            "absolute_residual_scale_spearman": 0.3 if uncertainty else None,
        },
    }
    (run / "training_config.json").write_text(json.dumps(config))
    summary_path = run / "training_summary.json"
    summary_path.write_text(json.dumps(summary))
    return summary_path


def test_selects_by_mean_performance_and_exposes_uncertainty_diagnostics(tmp_path: Path) -> None:
    module = _load()
    paths = [
        _write_run(tmp_path, "huber", 0.20, 0.9, 0.2),
        _write_run(tmp_path, "fixed_variance_gaussian_nll", 0.25, 1.0, 0.25),
        _write_run(tmp_path, "learned_variance_gaussian_nll", 0.22, 0.8, 0.1),
    ]
    result = module.summarize(paths)
    assert result["selected_loss_for_controls"] == "fixed_variance_gaussian_nll"
    diagnostic = result["learned_uncertainty_diagnostics"]
    assert diagnostic["task_macro_spearman_difference_from_huber"] == pytest.approx(0.02)
    assert diagnostic["prediction_spread_ratio_difference_from_huber"] == pytest.approx(-0.1)
    assert diagnostic["absolute_residual_scale_spearman"] == pytest.approx(0.3)
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False


def test_refuses_nonmatched_configuration_or_incomplete_loss_set(tmp_path: Path) -> None:
    module = _load()
    paths = [
        _write_run(tmp_path, "huber", 0.20, 0.9, 0.2),
        _write_run(tmp_path, "fixed_variance_gaussian_nll", 0.25, 1.0, 0.25),
        _write_run(tmp_path, "learned_variance_gaussian_nll", 0.22, 0.8, 0.1),
    ]
    config_path = paths[-1].parent / "training_config.json"
    config = json.loads(config_path.read_text())
    config["batch_size"] = 32
    config_path.write_text(json.dumps(config))
    with pytest.raises(module.LossComparisonError, match="non-loss"):
        module.summarize(paths)
    with pytest.raises(module.LossComparisonError, match="exactly three"):
        module.summarize(paths[:2])


def test_refuses_hidden_architecture_or_split_scope_drift(tmp_path: Path) -> None:
    module = _load()
    paths = [
        _write_run(tmp_path, "huber", 0.20, 0.9, 0.2),
        _write_run(tmp_path, "fixed_variance_gaussian_nll", 0.25, 1.0, 0.25),
        _write_run(tmp_path, "learned_variance_gaussian_nll", 0.22, 0.8, 0.1),
    ]
    config_path = paths[-1].parent / "training_config.json"
    config = json.loads(config_path.read_text())
    config["critic_position_features"] = "DIFFERENT_POSITION_FEATURES"
    config_path.write_text(json.dumps(config))
    with pytest.raises(module.LossComparisonError, match="non-loss"):
        module.summarize(paths)

    config["critic_position_features"] = "NORMALIZED_ABSOLUTE_PLUS_EDIT_GATED"
    config["development_test_outcomes_accessed"] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(module.LossComparisonError, match="TEST outcomes"):
        module.summarize(paths)
