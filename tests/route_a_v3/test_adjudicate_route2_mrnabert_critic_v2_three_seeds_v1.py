from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_three_seeds_v1.py"
PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json"
TASKS = [f"task-{index}" for index in range(9)]


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_seed_adjudicate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _control_adjudication(protocol: dict) -> dict:
    baseline = protocol["strongest_same_information_baseline"]
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1",
        "status": "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS",
        "supports_three_frozen_seeds": True,
        "frozen_confirmation_seeds": protocol["required_seeds"],
        "strongest_same_information_baseline": baseline,
        "arms": {
            "candidate_permutation": {"task_macro_spearman": 0.10},
            "source_only": {"task_macro_spearman": 0.09},
            "source_edit_metadata": {"task_macro_spearman": 0.11},
        },
        "development_test_opened": False,
        "evaluation_opened": False,
    }


def _summary(protocol: dict, seed: int, task_value: float = 0.20) -> dict:
    policy = protocol["frozen_training_policy"]
    task_metrics = {
        task: {"spearman": task_value, "standardized_mae": 1.0}
        for task in TASKS
    }
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "baseline_id": f"critic-v2-seed-{seed}",
        "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "seed": seed,
        "model_kind": policy["model_kind"],
        "candidate_control": "NONE",
        "loss_kind": policy["loss_kind"],
        "huber_delta": policy["huber_delta"],
        "batch_size": policy["batch_size"],
        "learning_rate": policy["learning_rate"],
        "weight_decay": policy["weight_decay"],
        "optimizer_name": policy["optimizer_name"],
        "training_precision": policy["training_precision"],
        "training_weighting_mode": policy["training_weighting_mode"],
        "training_sampling_mode": policy["training_sampling_mode"],
        "loss_aggregation_mode": policy["loss_aggregation_mode"],
        "training_update_mode": policy["training_update_mode"],
        "checkpoint_selection": policy["checkpoint_selection"],
        "checkpoint_metric": policy["checkpoint_metric"],
        "target_scaler": {"mode": policy["target_scaling_mode"]},
        "final_training_epoch": policy["epochs"],
        "selected_epoch": 40,
        "development_test_outcomes_evaluated": False,
        "development_test_record_count_withheld": 18292,
        "test_metrics": None,
        "evaluation_outcomes_read": 0,
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "optimizer_steps": 559900,
        "physical_gpu_index": 2,
        "trainable_parameter_count": 9342914,
        "frozen_pretrained_parameter_count": 113389056,
        "total_effective_parameter_count": 122731970,
        "validation_metrics": {
            "task_macro_spearman": statistics.fmean(
                row["spearman"] for row in task_metrics.values()
            ),
            "task_macro_standardized_mae": 1.7,
            "prediction_std": 0.12,
            "target_std": 0.20,
            "prediction_std_over_target_std": 0.60,
            "task_metrics": task_metrics,
        },
    }


def test_three_positive_seeds_report_required_diagnostics_and_authorize_test() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    controls = _control_adjudication(protocol)
    summaries = [_summary(protocol, seed) for seed in protocol["required_seeds"]]
    result = module.adjudicate(protocol, controls, summaries)
    assert result["status"] == "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST"
    assert result["supports_single_frozen_development_test"] is True
    assert all(result["checks"].values())
    assert all(row["positive_task_count"] == 9 for row in result["seed_results"])
    assert all(row["task_macro_standardized_mae"] == 1.7 for row in result["seed_results"])
    assert all(row["prediction_std_over_target_std"] == 0.6 for row in result["seed_results"])
    assert all(set(row["task_macro_gaps_over_each_control"]) == {"candidate_permutation", "source_only", "source_edit_metadata"} for row in result["seed_results"])
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False


def test_one_nonpositive_baseline_margin_is_terminal_no_go() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    controls = _control_adjudication(protocol)
    summaries = [_summary(protocol, seed) for seed in protocol["required_seeds"]]
    summaries[1] = _summary(protocol, protocol["required_seeds"][1], task_value=0.12)
    result = module.adjudicate(protocol, controls, summaries)
    assert result["status"] == "CRITIC_V2_THREE_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST"
    assert result["supports_single_frozen_development_test"] is False
    assert result["checks"]["all_three_seed_margins_over_strongest_baseline_positive"] is False


def test_nonfinite_or_collapsed_spread_is_reported_as_no_go() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    controls = _control_adjudication(protocol)
    summaries = [_summary(protocol, seed) for seed in protocol["required_seeds"]]
    summaries[0]["validation_metrics"]["task_macro_standardized_mae"] = float("nan")
    summaries[1]["validation_metrics"]["prediction_std"] = 0.0
    summaries[1]["validation_metrics"]["prediction_std_over_target_std"] = 0.0
    result = module.adjudicate(protocol, controls, summaries)
    assert result["supports_single_frozen_development_test"] is False
    assert result["seed_results"][0]["nonfinite_metric_detected"] is True
    assert result["seed_results"][0]["task_macro_standardized_mae"] is None
    assert result["seed_results"][1]["mean_collapse_detected"] is True


def test_rejects_test_metrics() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    controls = _control_adjudication(protocol)
    summaries = [_summary(protocol, seed) for seed in protocol["required_seeds"]]
    summaries[0]["test_metrics"] = {"spearman": 0.9}
    with pytest.raises(module.CriticV2ThreeSeedAdjudicationError, match="TEST metrics"):
        module.adjudicate(protocol, controls, summaries)
