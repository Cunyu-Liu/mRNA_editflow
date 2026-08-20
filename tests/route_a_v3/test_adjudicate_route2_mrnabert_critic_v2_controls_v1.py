from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_controls_v1.py"
PROTOCOL = ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
TASKS = [
    "MEAN_RIBOSOME_LOAD::region=0",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "E3::region=0",
    "E4::region=0",
    "E5::region=0",
    "E6::region=0",
    "E7::region=1",
    "E8::region=1",
    "E9::region=1",
]


def _load():
    spec = importlib.util.spec_from_file_location("critic_v2_adjudicate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(protocol: dict, arm: str, task_value: float) -> dict:
    spec = protocol["arms"][arm]
    policy = protocol["frozen_training_policy"]
    task_metrics = {
        task: {"spearman": task_value, "standardized_mae": 1.0}
        for task in TASKS
    }
    return {
        "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
        "baseline_id": f"critic-v2-{arm}",
        "result_stage": "HPO_VALIDATION_ONLY",
        "run_mode": "FIXED_GROUPED_SPLIT",
        "seed": protocol["screen_seed"],
        "model_kind": spec["model_kind"],
        "candidate_control": spec["candidate_control"],
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
            "task_count": 9,
            "defined_task_spearman_count": 9,
            "task_macro_spearman": statistics.fmean(
                row["spearman"] for row in task_metrics.values()
            ),
            "task_macro_standardized_mae": 1.7,
            "prediction_std": 0.12,
            "target_std": 0.2,
            "prediction_std_over_target_std": 0.6,
            "task_metrics": task_metrics,
        },
    }


def _passing_summaries(protocol: dict) -> dict:
    return {
        "full": _summary(protocol, "full", 0.20),
        "candidate_permutation": _summary(protocol, "candidate_permutation", 0.10),
        "source_only": _summary(protocol, "source_only", 0.09),
        "source_edit_metadata": _summary(protocol, "source_edit_metadata", 0.12),
    }


def test_pass_requires_full_candidate_specific_signal_and_authorizes_only_three_frozen_seeds() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    result = module.adjudicate(protocol, _passing_summaries(protocol))
    assert result["status"] == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS"
    assert result["supports_three_frozen_seeds"] is True
    assert result["frozen_confirmation_seeds"] == [20260822, 20260823, 20260824]
    assert all(result["checks"].values())
    assert result["development_test_opened"] is False
    assert result["evaluation_opened"] is False
    assert result["guided_generation_authorized"] is False


def test_anchor_only_explaining_full_signal_is_terminal_no_go() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    summaries = _passing_summaries(protocol)
    summaries["source_edit_metadata"] = _summary(
        protocol, "source_edit_metadata", 0.21
    )
    result = module.adjudicate(protocol, summaries)
    assert result["status"] == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
    assert result["supports_three_frozen_seeds"] is False
    assert result["checks"]["full_beats_source_edit_metadata_macro"] is False


def test_rejects_test_metrics_or_unmatched_budget() -> None:
    module = _load()
    protocol = json.loads(PROTOCOL.read_text())
    summaries = _passing_summaries(protocol)
    summaries["full"]["test_metrics"] = {"spearman": 0.9}
    with pytest.raises(module.CriticV2AdjudicationError, match="TEST metrics"):
        module.adjudicate(protocol, summaries)
    summaries = _passing_summaries(protocol)
    summaries["source_only"]["optimizer_steps"] -= 1
    with pytest.raises(module.CriticV2AdjudicationError, match="not parameter/budget matched"):
        module.adjudicate(protocol, summaries)
