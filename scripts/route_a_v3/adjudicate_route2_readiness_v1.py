#!/usr/bin/env python3
"""Adjudicate Route 2 critic/Flow readiness without running guided generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


class ReadinessError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _cuda_provenance(
    payload: Mapping[str, Any],
    device_key: str,
    physical_index_key: str,
    cpu_fallback_key: str,
    *,
    require_observed_device: bool = True,
) -> bool:
    physical_index = payload.get(physical_index_key)
    basic = (
        isinstance(physical_index, int)
        and not isinstance(physical_index, bool)
        and physical_index >= 0
        and payload.get(device_key) == f"cuda:{physical_index}"
        and payload.get(cpu_fallback_key) is False
    )
    if not basic or not require_observed_device:
        return basic
    observed_index = payload.get("cuda_device_index")
    observed_uuid = payload.get("cuda_device_uuid")
    observed_total = payload.get("cuda_total_memory_mb")
    return (
        observed_index == physical_index
        and isinstance(observed_uuid, str)
        and bool(observed_uuid)
        and isinstance(observed_total, (int, float))
        and not isinstance(observed_total, bool)
        and math.isfinite(float(observed_total))
        and float(observed_total) > 0.0
    )


def adjudicate(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload["schema_version"] == "route_a_v3_route2_readiness_input.v1", "unexpected readiness input schema")
    critic = payload["critic"]
    flow = payload["flow"]
    critic_validation_training = critic["validation_training_summary"]
    critic_final_refit = critic["final_refit_summary"]
    loso_rows = critic["loso_seed_results"]
    complete_loso = (
        len(loso_rows) == 3
        and len({row["seed"] for row in loso_rows}) == 3
        and all(
            row.get("status") == "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE"
            and row.get("study_count") == critic["expected_loso_study_count"]
            and row.get("all_model_training_gpu_provenance_verified") is True
            and row.get("development_test_preserved") is True
            and row.get("evaluation_studies_included") == 0
            for row in loso_rows
        )
    )
    positive_loso = complete_loso and all(
        _finite(row["model_macro_spearman"], "model macro Spearman")
        - _finite(row["baseline_macro_spearman"], "baseline macro Spearman") > 0.0
        for row in loso_rows
    )
    signal = critic.get("signal_control_adjudication")
    signal_checks = signal.get("checks", {}) if isinstance(signal, Mapping) else {}
    signal_gate_pass = (
        isinstance(signal, Mapping)
        and signal.get("schema_version")
        == "route_a_v3_route2_mrnabert_signal_control_adjudication.v1"
        and signal.get("status")
        == "MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION"
        and signal.get("supports_final_seed_confirmation") is True
        and signal.get("development_test_opened") is False
        and signal.get("evaluation_opened") is False
        and signal.get("guided_generation_authorized") is False
        and signal_checks
        and all(value is True for value in signal_checks.values())
    )
    critic_checks = {
        "validation_evidence_gpu_parameter_update": (
            critic_validation_training["status"]
            == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
            and critic_validation_training.get("result_stage")
            == "FROZEN_DEVELOPMENT_VALIDATION"
            and critic_validation_training.get("development_test_outcomes_evaluated") is False
            and critic_validation_training.get("test_metrics") is None
            and critic_validation_training.get("candidate_control") == "NONE"
            and critic_validation_training["optimizer_steps"] > 0
            and critic_validation_training["parameter_changed"] is True
            and critic_validation_training.get("cuda_training_tensors_verified") is True
            and _cuda_provenance(
                critic_validation_training,
                "device",
                "physical_gpu_index",
                "cpu_fallback_used",
            )
        ),
        "final_all_development_refit_gpu_parameter_update": (
            critic_final_refit["status"]
            == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
            and critic_final_refit.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT"
            and critic_final_refit.get("checkpoint_selection") == "FINAL_EPOCH"
            and critic_final_refit.get("candidate_control") == "NONE"
            and critic_final_refit.get("validation_metrics") is None
            and critic_final_refit.get("test_metrics") is None
            and critic_final_refit["optimizer_steps"] > 0
            and critic_final_refit["parameter_changed"] is True
            and critic_final_refit.get("cuda_training_tensors_verified") is True
            and _cuda_provenance(
                critic_final_refit,
                "device",
                "physical_gpu_index",
                "cpu_fallback_used",
            )
        ),
        "development_grouped_split_frozen": critic["development_grouped_split_status"] == "ROUTE2_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED",
        "strongest_same_information_baseline_run": critic["strongest_baseline_status"] == "COMPLETED_DEVELOPMENT_ONLY",
        "three_complete_final_seed_loso_runs_present": complete_loso,
        "all_loso_seed_improvements_positive": positive_loso,
        "prospectively_frozen_signal_control_gate_pass": signal_gate_pass,
        "candidate_permutation_control_worse": signal_gate_pass and (
            signal_checks.get("primary_beats_permutation_on_all_required_tasks") is True
            and signal_checks.get("primary_permutation_required_task_mean_margin_positive") is True
        ),
        "source_only_control_worse": signal_gate_pass and (
            signal_checks.get("primary_beats_source_only_macro") is True
            and signal_checks.get("primary_beats_source_only_on_required_task_breadth") is True
        ),
        "critic_checkpoint_frozen": critic["critic_checkpoint_frozen"] is True,
        "input_schema_context_reward_frozen": all(
            critic[key] is True for key in ("input_schema_frozen", "context_policy_frozen", "reward_calibration_policy_frozen")
        ),
        "generated_candidate_online_encoder_ready": (
            critic.get("generated_candidate_online_encoder_ready") is True
        ),
        "evaluation_not_used": (
            critic["evaluation_records_used_for_training_hpo_threshold_or_reward"] == 0
            and critic_validation_training["evaluation_outcomes_read"] == 0
            and critic_final_refit["evaluation_outcomes_read"] == 0
        ),
    }
    flow_training = flow["training_summary"]
    flow_validation = flow["validation_summary"]
    flow_checks = {
        "learned_gpu_parameter_update": (
            flow_training["status"] == "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE"
            and flow_training["optimizer_steps"] > 0
            and flow_training["parameter_changed"] is True
            and flow_training.get("cuda_training_tensors_verified") is True
            and _cuda_provenance(
                flow_training, "torch_device", "physical_gpu_index", "cpu_fallback_used"
            )
        ),
        "validation_gpu_and_checkpoint_provenance": (
            _cuda_provenance(
                flow_validation, "device", "physical_gpu_index", "cpu_fallback_used"
            )
            and flow_validation.get("trajectory_sampling_device") == flow_validation.get("device")
            and flow_validation.get("checkpoint_gpu_parameter_update_provenance_verified") is True
            and _cuda_provenance(
                flow_validation,
                "checkpoint_training_device",
                "checkpoint_training_physical_gpu_index",
                "checkpoint_cpu_fallback_used",
                require_observed_device=False,
            )
            and flow_validation.get("checkpoint_training_seed") == flow_training.get("seed")
            and 0 < flow_validation.get("checkpoint_training_optimizer_steps", 0) <= flow_training.get("optimizer_steps", 0)
            and flow_validation.get("checkpoint_parameter_changed") is flow_training.get("parameter_changed") is True
            and flow_validation.get("checkpoint_cuda_training_tensors_verified") is flow_training.get("cuda_training_tensors_verified") is True
            and flow_validation.get("checkpoint_training_device") == flow_training.get("torch_device")
            and flow_validation.get("checkpoint_training_physical_gpu_index") == flow_training.get("physical_gpu_index")
            and flow_validation.get("checkpoint_cpu_fallback_used") is flow_training.get("cpu_fallback_used") is False
            and flow_validation.get("checkpoint_training_cuda_device_index") == flow_training.get("cuda_device_index")
            and flow_validation.get("checkpoint_training_cuda_device_uuid") == flow_training.get("cuda_device_uuid")
            and flow_validation.get("checkpoint_training_cuda_total_memory_mb") == flow_training.get("cuda_total_memory_mb")
        ),
        "hard_legality_100_percent": flow_validation["hard_legality_rate"] == 1.0,
        "edit_budget_violation_zero": flow_validation["edit_budget_violation_count"] == 0,
        "trajectory_replay_failure_zero": flow_validation["trajectory_replay_failure_count"] == 0,
        "terminal_causes_distinguishable": set(flow_validation["distinguishable_terminal_causes"]) == {
            "EXPLICIT_STOP", "BUDGET_EXHAUSTED", "NO_LEGAL_ACTION", "NUMERICAL_FAILURE"
        },
        "learned_small_graph_reference_pass": (
            flow_validation["small_graph_reference"]["status"] == "PASS"
            and flow_validation["small_graph_reference"]["total_variation"]
            <= flow_validation["small_graph_reference"]["tolerance"]
        ),
        "evaluation_not_used": flow_training["evaluation_records_read"] == 0 and flow_validation["evaluation_outcomes_read"] == 0,
        "g0_not_biological_success": (
            flow_training["biological_optimization_established"] is False
            and flow_validation["biological_optimization_established"] is False
        ),
    }
    critic_ready = all(critic_checks.values())
    flow_ready = all(flow_checks.values())
    guided_unlocked = critic_ready and flow_ready
    return {
        "schema_version": "route_a_v3_route2_readiness_adjudication.v1",
        "critic_status": "CRITIC_READY_FOR_GUIDANCE" if critic_ready else "CRITIC_NOT_READY_FOR_GUIDANCE",
        "flow_status": "FLOW_G0_READY" if flow_ready else "FLOW_G0_NOT_READY",
        "guided_generation_status": "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED" if guided_unlocked else "NOT_STARTED_DEPENDENCY_NOT_MET",
        "critic_checks": critic_checks,
        "signal_control_status": signal.get("status") if isinstance(signal, Mapping) else None,
        "flow_checks": flow_checks,
        "guided_unlocked": guided_unlocked,
        "evaluation_opened": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    result = adjudicate(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
