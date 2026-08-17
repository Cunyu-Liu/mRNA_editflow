from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/adjudicate_route2_readiness_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("adjudicate_route2_readiness_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_input():
    return {
        "schema_version": "route_a_v3_route2_readiness_input.v1",
        "critic": {
            "validation_training_summary": {
                "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
                "optimizer_steps": 10, "parameter_changed": True,
                "device": "cuda:3", "physical_gpu_index": 3,
                "cpu_fallback_used": False, "cuda_training_tensors_verified": True,
                "cuda_device_index": 3, "cuda_device_uuid": "GPU-critic",
                "cuda_total_memory_mb": 40960.0,
                "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
                "development_test_outcomes_evaluated": False,
                "test_metrics": None,
                "candidate_control": "NONE",
                "evaluation_outcomes_read": 0,
            },
            "final_refit_summary": {
                "status": "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE",
                "optimizer_steps": 20, "parameter_changed": True,
                "device": "cuda:0", "physical_gpu_index": 0,
                "cpu_fallback_used": False, "cuda_training_tensors_verified": True,
                "cuda_device_index": 0, "cuda_device_uuid": "GPU-final-critic",
                "cuda_total_memory_mb": 40960.0,
                "result_stage": "FINAL_ALL_DEVELOPMENT_REFIT",
                "checkpoint_selection": "FINAL_EPOCH",
                "candidate_control": "NONE",
                "validation_metrics": None,
                "test_metrics": None,
                "evaluation_outcomes_read": 0,
            },
            "development_grouped_split_status": "ROUTE2_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED",
            "strongest_baseline_status": "COMPLETED_DEVELOPMENT_ONLY",
            "expected_loso_study_count": 7,
            "loso_seed_results": [
                {
                    "seed": seed,
                    "status": "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE",
                    "study_count": 7,
                    "all_model_training_gpu_provenance_verified": True,
                    "development_test_preserved": True,
                    "evaluation_studies_included": 0,
                    "model_macro_spearman": 0.2,
                    "baseline_macro_spearman": 0.1,
                }
                for seed in (1, 2, 3)
            ],
            "signal_control_adjudication": {
                "schema_version": "route_a_v3_route2_mrnabert_signal_control_adjudication.v1",
                "status": "MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION",
                "supports_final_seed_confirmation": True,
                "development_test_opened": False,
                "evaluation_opened": False,
                "guided_generation_authorized": False,
                "checks": {
                    "primary_beats_strongest_same_information_baseline": True,
                    "primary_task_median_positive": True,
                    "primary_beats_source_only_macro": True,
                    "primary_beats_source_only_on_required_task_breadth": True,
                    "primary_beats_permutation_on_all_required_tasks": True,
                    "primary_permutation_required_task_mean_margin_positive": True,
                },
            },
            "critic_checkpoint_frozen": True,
            "input_schema_frozen": True,
            "context_policy_frozen": True,
            "reward_calibration_policy_frozen": True,
            "evaluation_records_used_for_training_hpo_threshold_or_reward": 0,
        },
        "flow": {
            "training_summary": {
                "status": "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE", "optimizer_steps": 10,
                "seed": 11,
                "parameter_changed": True, "torch_device": "cuda:2", "physical_gpu_index": 2,
                "cpu_fallback_used": False, "cuda_training_tensors_verified": True,
                "cuda_device_index": 2, "cuda_device_uuid": "GPU-flow-train",
                "cuda_total_memory_mb": 40960.0,
                "evaluation_records_read": 0,
                "biological_optimization_established": False,
            },
            "validation_summary": {
                "hard_legality_rate": 1.0, "edit_budget_violation_count": 0,
                "trajectory_replay_failure_count": 0,
                "distinguishable_terminal_causes": [
                    "EXPLICIT_STOP", "BUDGET_EXHAUSTED", "NO_LEGAL_ACTION", "NUMERICAL_FAILURE"
                ],
                "small_graph_reference": {"status": "PASS", "total_variation": 0.0, "tolerance": 1e-12},
                "device": "cuda:3", "physical_gpu_index": 3, "cpu_fallback_used": False,
                "cuda_device_index": 3, "cuda_device_uuid": "GPU-flow-validation",
                "cuda_total_memory_mb": 40960.0,
                "trajectory_sampling_device": "cuda:3",
                "checkpoint_gpu_parameter_update_provenance_verified": True,
                "checkpoint_training_device": "cuda:2", "checkpoint_training_physical_gpu_index": 2,
                "checkpoint_cpu_fallback_used": False,
                "checkpoint_training_seed": 11, "checkpoint_training_optimizer_steps": 10,
                "checkpoint_parameter_changed": True,
                "checkpoint_cuda_training_tensors_verified": True,
                "checkpoint_training_cuda_device_index": 2,
                "checkpoint_training_cuda_device_uuid": "GPU-flow-train",
                "checkpoint_training_cuda_total_memory_mb": 40960.0,
                "evaluation_outcomes_read": 0, "biological_optimization_established": False,
            },
        },
    }


def test_both_gates_are_required_for_guided_unlock() -> None:
    module = _load()
    result = module.adjudicate(_passing_input())
    assert result["critic_status"] == "CRITIC_READY_FOR_GUIDANCE"
    assert result["flow_status"] == "FLOW_G0_READY"
    assert result["guided_unlocked"] is True


def test_one_failed_seed_preserves_failure_and_keeps_guided_closed() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["loso_seed_results"][1]["model_macro_spearman"] = 0.0
    result = module.adjudicate(payload)
    assert result["critic_checks"]["all_loso_seed_improvements_positive"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["flow_status"] == "FLOW_G0_READY"
    assert result["guided_unlocked"] is False


def test_static_flow_without_gpu_update_cannot_unlock() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["flow"]["training_summary"]["optimizer_steps"] = 0
    result = module.adjudicate(payload)
    assert result["flow_checks"]["learned_gpu_parameter_update"] is False
    assert result["flow_status"] == "FLOW_G0_NOT_READY"
    assert result["guided_unlocked"] is False


def test_cpu_fallback_or_device_mismatch_keeps_guided_closed() -> None:
    module = _load()
    for field, value in (("cpu_fallback_used", True), ("torch_device", "cuda:1")):
        payload = deepcopy(_passing_input())
        payload["flow"]["training_summary"][field] = value
        result = module.adjudicate(payload)
        assert result["flow_checks"]["learned_gpu_parameter_update"] is False
        assert result["guided_unlocked"] is False


def test_validation_requires_verified_gpu_checkpoint_provenance() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["flow"]["validation_summary"]["checkpoint_gpu_parameter_update_provenance_verified"] = False
    result = module.adjudicate(payload)
    assert result["flow_checks"]["validation_gpu_and_checkpoint_provenance"] is False
    assert result["guided_unlocked"] is False


def test_validation_checkpoint_must_match_flow_training_summary() -> None:
    module = _load()
    for field, value in (
        ("checkpoint_training_seed", 12),
        ("checkpoint_training_cuda_device_uuid", "GPU-other"),
    ):
        payload = deepcopy(_passing_input())
        payload["flow"]["validation_summary"][field] = value
        result = module.adjudicate(payload)
        assert result["flow_checks"]["validation_gpu_and_checkpoint_provenance"] is False
        assert result["guided_unlocked"] is False


def test_validation_accepts_earlier_best_checkpoint_from_same_training_run() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["flow"]["validation_summary"]["checkpoint_training_optimizer_steps"] = 9
    result = module.adjudicate(payload)
    assert result["flow_checks"]["validation_gpu_and_checkpoint_provenance"] is True
    assert result["flow_status"] == "FLOW_G0_READY"


def test_validation_rejects_zero_or_future_checkpoint_steps() -> None:
    module = _load()
    for optimizer_steps in (0, 11):
        payload = deepcopy(_passing_input())
        payload["flow"]["validation_summary"]["checkpoint_training_optimizer_steps"] = optimizer_steps
        result = module.adjudicate(payload)
        assert result["flow_checks"]["validation_gpu_and_checkpoint_provenance"] is False
        assert result["guided_unlocked"] is False


def test_incomplete_loso_or_cpu_sampling_provenance_keeps_guided_closed() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["loso_seed_results"][0]["study_count"] = 6
    result = module.adjudicate(payload)
    assert result["critic_checks"]["three_complete_final_seed_loso_runs_present"] is False
    assert result["guided_unlocked"] is False

    payload = deepcopy(_passing_input())
    payload["flow"]["validation_summary"]["trajectory_sampling_device"] = "cpu"
    result = module.adjudicate(payload)
    assert result["flow_checks"]["validation_gpu_and_checkpoint_provenance"] is False
    assert result["guided_unlocked"] is False


def test_structured_undefined_loso_result_keeps_critic_closed_without_error() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["loso_seed_results"][1].update({
        "status": "LOSO_MODEL_BASELINE_ALIGNMENT_NOT_ESTABLISHED",
        "model_macro_spearman": None,
        "baseline_macro_spearman": None,
        "macro_improvement": None,
        "undefined_study_count": 1,
    })
    result = module.adjudicate(payload)
    assert result["critic_checks"]["three_complete_final_seed_loso_runs_present"] is False
    assert result["critic_checks"]["all_loso_seed_improvements_positive"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guided_unlocked"] is False


def test_nonpassing_signal_control_adjudication_keeps_critic_closed() -> None:
    module = _load()
    for field, value in (
        ("supports_final_seed_confirmation", False),
        ("development_test_opened", True),
        ("status", "MRNABERT_SIGNAL_CONTROLS_DO_NOT_SUPPORT_FINAL_SEED_CONFIRMATION"),
    ):
        payload = deepcopy(_passing_input())
        payload["critic"]["signal_control_adjudication"][field] = value
        result = module.adjudicate(payload)
        assert result["critic_checks"]["prospectively_frozen_signal_control_gate_pass"] is False
        assert result["critic_checks"]["candidate_permutation_control_worse"] is False
        assert result["critic_checks"]["source_only_control_worse"] is False
        assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
        assert result["guided_unlocked"] is False


def test_critic_requires_real_gpu_training_provenance() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["final_refit_summary"]["cpu_fallback_used"] = True
    result = module.adjudicate(payload)
    assert result["critic_checks"]["final_all_development_refit_gpu_parameter_update"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guided_unlocked"] is False


def test_missing_cuda_tensor_execution_proof_keeps_both_gates_closed() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["validation_training_summary"]["cuda_training_tensors_verified"] = False
    payload["flow"]["training_summary"]["cuda_training_tensors_verified"] = False
    result = module.adjudicate(payload)
    assert result["critic_checks"]["validation_evidence_gpu_parameter_update"] is False
    assert result["flow_checks"]["learned_gpu_parameter_update"] is False
    assert result["guided_unlocked"] is False


def test_missing_observed_cuda_identity_keeps_guided_closed() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["final_refit_summary"]["cuda_device_uuid"] = None
    result = module.adjudicate(payload)
    assert result["critic_checks"]["final_all_development_refit_gpu_parameter_update"] is False
    assert result["guided_unlocked"] is False


def test_hpo_checkpoint_or_unverified_loso_training_keeps_guided_closed() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["validation_training_summary"]["result_stage"] = "HPO_VALIDATION_ONLY"
    result = module.adjudicate(payload)
    assert result["critic_checks"]["validation_evidence_gpu_parameter_update"] is False
    assert result["guided_unlocked"] is False

    payload = deepcopy(_passing_input())
    payload["critic"]["loso_seed_results"][0]["all_model_training_gpu_provenance_verified"] = False
    result = module.adjudicate(payload)
    assert result["critic_checks"]["three_complete_final_seed_loso_runs_present"] is False
    assert result["guided_unlocked"] is False


def test_development_test_refit_cannot_guide_development_validation_generation() -> None:
    module = _load()
    payload = deepcopy(_passing_input())
    payload["critic"]["final_refit_summary"]["result_stage"] = "FROZEN_DEVELOPMENT_TEST"
    result = module.adjudicate(payload)
    assert result["critic_checks"]["final_all_development_refit_gpu_parameter_update"] is False
    assert result["critic_status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert result["guided_unlocked"] is False
