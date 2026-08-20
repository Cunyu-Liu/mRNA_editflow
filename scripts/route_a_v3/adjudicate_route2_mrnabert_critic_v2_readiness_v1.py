#!/usr/bin/env python3
"""Adjudicate Critic V2 and Flow G0 readiness without running guidance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


REQUIRED_SEEDS = (20260822, 20260823, 20260824)
PRIMARY_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
EXPECTED_PROTOCOL_SCHEMAS = {
    "readiness": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol.v1",
    "control": "route_a_v3_route2_mrnabert_critic_v2_protocol.v1",
    "three_seed": "route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol.v1",
    "frozen_test": "route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol.v1",
    "refit": "route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol.v1",
    "primary_loso": "route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol.v1",
    "baseline_loso": "route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol.v1",
}


class CriticV2ReadinessError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticV2ReadinessError(message)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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
        and 0 <= physical_index <= 5
        and payload.get(device_key) == f"cuda:{physical_index}"
        and payload.get(cpu_fallback_key) is False
    )
    if not basic or not require_observed_device:
        return basic
    observed_total = _finite_float(payload.get("cuda_total_memory_mb"))
    return (
        payload.get("cuda_device_index") == physical_index
        and isinstance(payload.get("cuda_device_uuid"), str)
        and bool(payload.get("cuda_device_uuid"))
        and observed_total is not None
        and observed_total > 0.0
    )


def _protocol_chain_frozen(protocols: Mapping[str, Any]) -> bool:
    if set(protocols) != set(EXPECTED_PROTOCOL_SCHEMAS):
        return False
    for name, schema in EXPECTED_PROTOCOL_SCHEMAS.items():
        protocol = protocols.get(name)
        expected_status = (
            "FROZEN_BEFORE_CRITIC_V2_TRAINING_OUTCOMES"
            if name == "control"
            else "FROZEN_BEFORE_CRITIC_V2_THREE_SEED_OUTCOMES"
        )
        if not (
            isinstance(protocol, Mapping)
            and protocol.get("schema_version") == schema
            and protocol.get("status") == expected_status
            and protocol.get("evaluation_outcomes_accessed") is False
            and protocol.get("guided_generation_authorized") is False
        ):
            return False
    control = protocols["control"]
    three = protocols["three_seed"]
    frozen_test = protocols["frozen_test"]
    refit = protocols["refit"]
    primary_loso = protocols["primary_loso"]
    baseline_loso = protocols["baseline_loso"]
    readiness = protocols["readiness"]
    seed_sets = (
        control.get("frozen_confirmation_seeds"),
        three.get("required_seeds"),
        frozen_test.get("required_confirmation_seeds"),
        primary_loso.get("required_seeds"),
        baseline_loso.get("required_seeds"),
        readiness.get("required_seeds"),
    )
    if any(tuple(int(seed) for seed in seeds) != REQUIRED_SEEDS for seeds in seed_sets):
        return False
    if not (
        int(frozen_test.get("single_frozen_test_seed", -1)) == 20260823
        and int(refit.get("single_frozen_test_seed", -1)) == 20260823
        and int(readiness.get("single_frozen_test_seed", -1)) == 20260823
        and control.get("frozen_training_policy")
        == three.get("frozen_training_policy")
        == frozen_test.get("frozen_training_policy")
    ):
        return False
    baseline_ids = (
        control.get("strongest_same_information_baseline", {}).get("baseline_id"),
        three.get("strongest_same_information_baseline", {}).get("baseline_id"),
        frozen_test.get("strongest_same_information_baseline", {}).get("baseline_id"),
        baseline_loso.get("strongest_same_information_baseline", {}).get("baseline_id"),
    )
    return len(set(baseline_ids)) == 1 and baseline_ids[0] == "method_repair_global_scaled_seed20260821"


def _reward_policy_frozen(policy: Mapping[str, Any]) -> bool:
    transform = policy.get("potential_transform")
    return (
        policy.get("schema_version")
        == "route_a_v3_route2_mrnabert_guidance_reward_policy.v1"
        and policy.get("status")
        == "PROSPECTIVELY_FROZEN_BEFORE_GUIDED_GENERATION"
        and policy.get("critic_model_kind") == PRIMARY_KIND
        and policy.get("critic_checkpoint_role")
        == "FINAL_ALL_DEVELOPMENT_REFIT_FINAL_EPOCH"
        and policy.get("critic_parameter_update_during_generation") is False
        and policy.get("generator_gradient_into_critic") is False
        and policy.get("evaluation_model_gradient_into_generator") is False
        and policy.get("reward_signal") == "STANDARDIZED_PREDICTED_MEAN_DELTA"
        and policy.get("uncertainty_in_guidance") == "DISABLED_DIAGNOSTIC_ONLY"
        and policy.get("target_scale_source")
        == "FINAL_REFIT_CHECKPOINT_TRAIN_TASK_ROBUST_SCALER"
        and isinstance(transform, Mapping)
        and transform.get("kind") == "CLIPPED_IDENTITY"
        and transform.get("minimum") == -5.0
        and transform.get("maximum") == 5.0
        and policy.get("guidance_strength") == 1.0
        and policy.get("guidance_schedule") == "CONSTANT"
        and policy.get("transition_rule")
        == "BASE_TRANSITION_RATE_TIMES_EXP_POTENTIAL_DIFFERENCE"
        and policy.get("action_space") == "SUB_PLUS_STOP"
        and policy.get("generated_candidate_encoder")
        == "ONLINE_FROZEN_MRNABERT_WITH_SEQUENCE_MEMOIZATION"
        and policy.get("development_records_used_for_reward_calibration") == 126165
        and policy.get("evaluation_records_used_for_training_hpo_threshold_or_reward")
        == 0
        and policy.get("generated_candidates_add_to_canonical_records") is False
    )


def _online_encoder_ready(summary: Mapping[str, Any]) -> bool:
    difference = _finite_float(summary.get("maximum_absolute_difference"))
    tolerance = _finite_float(summary.get("absolute_tolerance"))
    return (
        summary.get("schema_version")
        == "route_a_v3_route2_mrnabert_online_encoder_validation.v1"
        and summary.get("status")
        == "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE"
        and summary.get("novel_candidate_encoding_supported") is True
        and int(summary.get("frozen_parameter_count", 0)) > 100_000_000
        and summary.get("evaluation_records_read") == 0
        and difference is not None
        and tolerance is not None
        and difference <= tolerance
    )


def _control_pass(adjudication: Mapping[str, Any]) -> bool:
    checks = adjudication.get("checks")
    return (
        adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_control_adjudication.v1"
        and adjudication.get("status")
        == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS"
        and adjudication.get("supports_three_frozen_seeds") is True
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
        and tuple(adjudication.get("frozen_confirmation_seeds", ())) == REQUIRED_SEEDS
        and adjudication.get("development_test_opened") is False
        and adjudication.get("evaluation_opened") is False
        and adjudication.get("guided_generation_authorized") is False
    )


def _three_seed_pass(adjudication: Mapping[str, Any]) -> bool:
    checks = adjudication.get("checks")
    rows = adjudication.get("seed_results")
    if not (
        adjudication.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_three_seed_adjudication.v1"
        and adjudication.get("status")
        == "CRITIC_V2_THREE_SEEDS_SUPPORT_ONE_FROZEN_DEVELOPMENT_TEST"
        and adjudication.get("supports_single_frozen_development_test") is True
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
        and isinstance(rows, list)
        and len(rows) == 3
        and adjudication.get("development_test_opened") is False
        and adjudication.get("evaluation_opened") is False
        and adjudication.get("guided_generation_authorized") is False
    ):
        return False
    by_seed = {int(row.get("seed", -1)): row for row in rows}
    return set(by_seed) == set(REQUIRED_SEEDS) and all(
        (_finite_float(by_seed[seed].get("margin_over_strongest_same_information_baseline")) or 0.0)
        > 0.0
        and by_seed[seed].get("nonfinite_metric_detected") is False
        and by_seed[seed].get("mean_collapse_detected") is False
        for seed in REQUIRED_SEEDS
    )


def _single_test_complete(config: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    return (
        config.get("scientific_role") == "CRITIC_V2_SINGLE_FROZEN_DEVELOPMENT_TEST"
        and config.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and int(config.get("seed", -1)) == 20260823
        and config.get("validation_checkpoint_selection_before_test")
        == "BEST_VALIDATION"
        and config.get("checkpoint_selection") == "FINAL_EPOCH"
        and config.get("epochs") == 100
        and config.get("test_used_for_checkpoint_selection") is False
        and config.get("test_used_for_model_or_policy_selection") is False
        and config.get("evaluation_outcomes_accessed") is False
        and summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and summary.get("result_stage") == "FROZEN_DEVELOPMENT_TEST"
        and int(summary.get("seed", -1)) == 20260823
        and summary.get("model_kind") == PRIMARY_KIND
        and summary.get("candidate_control") == "NONE"
        and summary.get("checkpoint_selection") == "FINAL_EPOCH"
        and summary.get("selected_epoch") == summary.get("final_training_epoch") == 100
        and summary.get("development_test_outcomes_evaluated") is True
        and isinstance(summary.get("test_metrics"), Mapping)
        and summary.get("development_validation_folded_into_training") is True
        and summary.get("record_counts") == {"TRAIN": 107873, "TEST": 18292}
        and summary.get("evaluation_outcomes_read") == 0
        and summary.get("optimizer_steps", 0) > 0
        and summary.get("parameter_changed") is True
        and summary.get("cuda_training_tensors_verified") is True
        and _cuda_provenance(summary, "device", "physical_gpu_index", "cpu_fallback_used")
    )


def _refit_complete(
    config: Mapping[str, Any], summary: Mapping[str, Any], checkpoint: Path
) -> bool:
    return (
        checkpoint.is_file()
        and config.get("scientific_role") == "CRITIC_V2_FINAL_ALL_DEVELOPMENT_REFIT"
        and config.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT"
        and config.get("development_record_scope") == "ALL_126165"
        and config.get("checkpoint_selection") == "FINAL_EPOCH"
        and config.get("epochs") == 100
        and config.get("refit_model_selection_performed") is False
        and config.get("test_metrics_used_for_refit_selection") is False
        and config.get("evaluation_outcomes_accessed") is False
        and summary.get("status") == "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE"
        and summary.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT"
        and summary.get("model_kind") == PRIMARY_KIND
        and summary.get("candidate_control") == "NONE"
        and summary.get("checkpoint_selection") == "FINAL_EPOCH"
        and summary.get("selected_epoch") == summary.get("final_training_epoch") == 100
        and summary.get("record_counts") == {"TRAIN": 126165}
        and summary.get("development_validation_folded_into_training") is True
        and summary.get("development_test_record_count_withheld") == 0
        and summary.get("test_metrics") is None
        and summary.get("evaluation_outcomes_read") == 0
        and summary.get("optimizer_steps", 0) > 0
        and summary.get("parameter_changed") is True
        and summary.get("cuda_training_tensors_verified") is True
        and _cuda_provenance(summary, "device", "physical_gpu_index", "cpu_fallback_used")
    )


def _loso_checks(rows: Any) -> tuple[bool, bool]:
    if not isinstance(rows, list) or len(rows) != 3:
        return False, False
    by_seed = {int(row.get("seed", -1)): row for row in rows}
    if len(by_seed) != 3 or set(by_seed) != set(REQUIRED_SEEDS):
        return False, False
    complete = True
    positive = True
    for seed in REQUIRED_SEEDS:
        row = by_seed[seed]
        model = _finite_float(row.get("model_macro_spearman"))
        baseline = _finite_float(row.get("baseline_macro_spearman"))
        improvement = _finite_float(row.get("macro_improvement"))
        row_complete = (
            row.get("schema_version") == "route_a_v3_route2_loso_aggregation.v1"
            and row.get("status") == "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE"
            and row.get("study_count") == 7
            and row.get("aligned_study_count") == 7
            and row.get("undefined_study_count") == 0
            and row.get("development_inventory_study_count") == 8
            and row.get("zero_record_development_studies") == ["GSE256185"]
            and row.get("all_model_training_gpu_provenance_verified") is True
            and row.get("development_test_preserved") is True
            and row.get("evaluation_studies_included") == 0
            and row.get("failure_reasons") == []
            and isinstance(row.get("per_study"), list)
            and len(row.get("per_study")) == 7
            and model is not None
            and baseline is not None
            and improvement is not None
            and math.isclose(improvement, model - baseline, rel_tol=0.0, abs_tol=1e-12)
        )
        complete = complete and row_complete
        positive = positive and row_complete and improvement is not None and improvement > 0.0
    return complete, positive


def _flow_checks(flow: Mapping[str, Any]) -> dict[str, bool]:
    training = flow["training_summary"]
    validation = flow["validation_summary"]
    checkpoint = Path(str(flow["checkpoint"]))
    return {
        "flow_checkpoint_present": checkpoint.is_file(),
        "learned_gpu_parameter_update": (
            training.get("status") == "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE"
            and training.get("optimizer_steps", 0) > 0
            and training.get("parameter_changed") is True
            and training.get("cuda_training_tensors_verified") is True
            and _cuda_provenance(
                training, "torch_device", "physical_gpu_index", "cpu_fallback_used"
            )
        ),
        "validation_gpu_and_checkpoint_provenance": (
            _cuda_provenance(
                validation, "device", "physical_gpu_index", "cpu_fallback_used"
            )
            and validation.get("trajectory_sampling_device") == validation.get("device")
            and validation.get("checkpoint_gpu_parameter_update_provenance_verified")
            is True
            and _cuda_provenance(
                validation,
                "checkpoint_training_device",
                "checkpoint_training_physical_gpu_index",
                "checkpoint_cpu_fallback_used",
                require_observed_device=False,
            )
            and validation.get("checkpoint_training_seed") == training.get("seed")
            and 0 < validation.get("checkpoint_training_optimizer_steps", 0)
            <= training.get("optimizer_steps", 0)
            and validation.get("checkpoint_parameter_changed")
            is training.get("parameter_changed")
            is True
            and validation.get("checkpoint_cuda_training_tensors_verified")
            is training.get("cuda_training_tensors_verified")
            is True
            and validation.get("checkpoint_training_device")
            == training.get("torch_device")
            and validation.get("checkpoint_training_physical_gpu_index")
            == training.get("physical_gpu_index")
            and validation.get("checkpoint_cpu_fallback_used")
            is training.get("cpu_fallback_used")
            is False
            and validation.get("checkpoint_training_cuda_device_index")
            == training.get("cuda_device_index")
            and validation.get("checkpoint_training_cuda_device_uuid")
            == training.get("cuda_device_uuid")
            and validation.get("checkpoint_training_cuda_total_memory_mb")
            == training.get("cuda_total_memory_mb")
        ),
        "hard_legality_100_percent": validation.get("hard_legality_rate") == 1.0,
        "edit_budget_violation_zero": validation.get("edit_budget_violation_count") == 0,
        "trajectory_replay_failure_zero": validation.get("trajectory_replay_failure_count") == 0,
        "terminal_causes_distinguishable": set(
            validation.get("distinguishable_terminal_causes", [])
        )
        == {"EXPLICIT_STOP", "BUDGET_EXHAUSTED", "NO_LEGAL_ACTION", "NUMERICAL_FAILURE"},
        "learned_small_graph_reference_pass": (
            validation.get("small_graph_reference", {}).get("status") == "PASS"
            and _finite_float(
                validation.get("small_graph_reference", {}).get("total_variation")
            )
            is not None
            and _finite_float(
                validation.get("small_graph_reference", {}).get("tolerance")
            )
            is not None
            and float(validation["small_graph_reference"]["total_variation"])
            <= float(validation["small_graph_reference"]["tolerance"])
        ),
        "evaluation_not_used": (
            training.get("evaluation_records_read") == 0
            and validation.get("evaluation_outcomes_read") == 0
        ),
        "g0_not_biological_success": (
            training.get("biological_optimization_established") is False
            and validation.get("biological_optimization_established") is False
        ),
    }


def adjudicate(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version")
        == "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_input.v1",
        "unexpected Critic V2 readiness input schema",
    )
    protocols = payload["protocols"]
    critic = payload["critic"]
    flow = payload["flow"]
    loso_complete, loso_positive = _loso_checks(critic.get("loso_seed_results"))
    critic_checks = {
        "prospective_protocol_chain_frozen": _protocol_chain_frozen(protocols),
        "critic_v2_control_gate_pass": _control_pass(
            critic["control_adjudication"]
        ),
        "critic_v2_three_seed_gate_pass": _three_seed_pass(
            critic["three_seed_adjudication"]
        ),
        "single_frozen_test_complete_without_test_selection": _single_test_complete(
            critic["frozen_test_config"], critic["frozen_test_summary"]
        ),
        "all_126165_refit_complete": _refit_complete(
            critic["refit_config"],
            critic["refit_summary"],
            Path(str(critic["refit_checkpoint"])),
        ),
        "three_complete_matched_loso_aggregations": loso_complete,
        "all_loso_seed_improvements_positive": loso_positive,
        "guidance_reward_policy_frozen": _reward_policy_frozen(
            critic["reward_policy"]
        ),
        "generated_candidate_online_encoder_ready": _online_encoder_ready(
            critic["online_encoder_validation"]
        ),
        "evaluation_not_used": (
            critic["control_adjudication"].get("evaluation_opened") is False
            and critic["three_seed_adjudication"].get("evaluation_opened") is False
            and critic["frozen_test_summary"].get("evaluation_outcomes_read") == 0
            and critic["refit_summary"].get("evaluation_outcomes_read") == 0
            and critic["reward_policy"].get(
                "evaluation_records_used_for_training_hpo_threshold_or_reward"
            )
            == 0
            and critic["online_encoder_validation"].get("evaluation_records_read")
            == 0
            and all(
                row.get("evaluation_studies_included") == 0
                for row in critic["loso_seed_results"]
            )
        ),
    }
    flow_checks = _flow_checks(flow)
    critic_ready = all(critic_checks.values())
    flow_ready = all(flow_checks.values())
    guided_unlocked = critic_ready and flow_ready
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_adjudication.v1",
        "critic_status": (
            "CRITIC_READY_FOR_GUIDANCE"
            if critic_ready
            else "CRITIC_NOT_READY_FOR_GUIDANCE"
        ),
        "flow_status": "FLOW_G0_READY" if flow_ready else "FLOW_G0_NOT_READY",
        "guided_generation_status": (
            "GUIDED_XEDITFLOW_DEVELOPMENT_ALLOWED"
            if guided_unlocked
            else "NOT_STARTED_DEPENDENCY_NOT_MET"
        ),
        "critic_checks": critic_checks,
        "flow_checks": flow_checks,
        "guided_unlocked": guided_unlocked,
        "guided_generation_executed": False,
        "evaluation_opened": False,
        "biological_optimization_established": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _require(
        protocol.get("schema_version")
        == EXPECTED_PROTOCOL_SCHEMAS["readiness"],
        "unexpected Critic V2 readiness protocol",
    )
    output = Path(str(protocol["readiness_adjudication_output"]))
    _require(not output.exists(), f"Critic V2 readiness adjudication already exists: {output}")
    result = adjudicate(json.loads(args.input.read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
