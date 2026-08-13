#!/usr/bin/env python3
"""Static validator for the non-active Route A V3 A6 GPU protocol draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(
    "configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json"
)

EXPECTED_ROOT_KEYS = {
    "schema_version",
    "protocol_id",
    "contract_id",
    "phase_id",
    "task_id",
    "document_status",
    "authority_status",
    "activation_state",
    "evidence_status",
    "claim_status",
    "implementation_scope",
    "authority_boundary",
    "forbidden_state_changes",
    "design_access_attestation",
    "activation_preconditions",
    "formal_production_interface",
    "ordinary_public_data_contract",
    "split_contract",
    "exposure_contract",
    "rights_contract",
    "base_architecture",
    "value_architecture",
    "terminal_tilt_contract",
    "training_objectives",
    "independent_exact_reference",
    "learned_potential_approximation_gate",
    "optimizer_compute_checkpoint_policy",
    "cuda_fail_closed_contract",
    "required_gates",
    "future_provenance_and_manifest_outputs",
    "terminal_truth",
}

EXPECTED_STATUS = {
    "schema_version": "route_a_v3_a6_learned_base_value_gpu_protocol_draft.v1",
    "protocol_id": "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_GPU_PROTOCOL_DRAFT_V1",
    "contract_id": "mrna_xeditflow_route_a_v3",
    "phase_id": "A6",
    "task_id": "FLOW_BASE_LEGAL_CTMC",
    "document_status": "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL",
    "authority_status": "NON_AUTHORITATIVE",
    "activation_state": "INACTIVE_REVIEW_CANDIDATE",
    "evidence_status": "NOT_RUN",
    "claim_status": "NOT_ESTABLISHED",
    "implementation_scope": "PROTOCOL_SCHEMA_STATIC_VALIDATOR_AND_FOCUSED_TEST_ONLY",
}

EXPECTED_FORBIDDEN_STATE_CHANGES = {
    "training_started": False,
    "training_allowed": False,
    "training_authorized": False,
    "parameter_updates_allowed": False,
    "gpu_work_allowed": False,
    "gpu_run_allowed": False,
    "model_execution_allowed": False,
    "model_selection_allowed": False,
    "checkpoint_selection_allowed": False,
    "a6_pass_asserted": False,
    "l3_claim_established": False,
    "a7_unlocked": False,
    "a7_authorized": False,
    "private_data_access_allowed": False,
    "sealed_data_access_allowed": False,
    "canonical_materialization_allowed": False,
    "qualification_change_allowed": False,
    "ordinary_study_credit_delta": 0,
    "a1_study_credit_delta": 0,
    "true_a2_study_credit_delta": 0,
    "canonical_record_count_delta": 0,
}

EXPECTED_ZERO_ACCESS = {
    "ordinary_row_reads": 0,
    "private_row_reads": 0,
    "sealed_row_reads": 0,
    "sequence_reads": 0,
    "effect_reads": 0,
    "model_executions": 0,
    "optimizer_steps": 0,
    "gpu_runs": 0,
    "checkpoints_created": 0,
}


def _issue(issues: list[dict[str, str]], code: str, path: str, detail: str) -> None:
    issues.append({"code": code, "path": path, "detail": detail})


def _expect(
    payload: dict[str, Any],
    dotted_path: str,
    expected: Any,
    issues: list[dict[str, str]],
) -> None:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            _issue(issues, "A6_DRAFT_MISSING_FIELD", dotted_path, "field is absent")
            return
        value = value[part]
    if value != expected:
        _issue(
            issues,
            "A6_DRAFT_VALUE_DRIFT",
            dotted_path,
            f"expected {expected!r}, observed {value!r}",
        )


def _expect_mapping(
    payload: dict[str, Any],
    dotted_path: str,
    expected: dict[str, Any],
    issues: list[dict[str, str]],
) -> None:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            _issue(issues, "A6_DRAFT_MISSING_FIELD", dotted_path, "mapping is absent")
            return
        value = value[part]
    if not isinstance(value, dict):
        _issue(issues, "A6_DRAFT_TYPE_DRIFT", dotted_path, "expected mapping")
        return
    if value != expected:
        _issue(
            issues,
            "A6_DRAFT_MAPPING_DRIFT",
            dotted_path,
            "mapping is not the frozen closed form",
        )


def validate_config(payload: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    if set(payload) != EXPECTED_ROOT_KEYS:
        missing = sorted(EXPECTED_ROOT_KEYS - set(payload))
        extra = sorted(set(payload) - EXPECTED_ROOT_KEYS)
        _issue(
            issues,
            "A6_DRAFT_ROOT_CLOSURE_DRIFT",
            "$",
            f"missing={missing}, extra={extra}",
        )

    for key, value in EXPECTED_STATUS.items():
        _expect(payload, key, value, issues)

    _expect_mapping(
        payload,
        "forbidden_state_changes",
        EXPECTED_FORBIDDEN_STATE_CHANGES,
        issues,
    )
    _expect_mapping(
        payload,
        "design_access_attestation",
        EXPECTED_ZERO_ACCESS,
        issues,
    )

    exact_expectations: dict[str, Any] = {
        "authority_boundary.registered_in_active_authority": False,
        "authority_boundary.registered_in_static_manifest": False,
        "authority_boundary.runtime_event_required_for_this_draft": False,
        "authority_boundary.activation_requires_new_explicit_owner_authorization": True,
        "authority_boundary.parameter_updates_require_later_explicit_owner_authorization": True,
        "authority_boundary.current_task_registry_status_must_remain": "NOT_RUN",
        "authority_boundary.current_a6_phase_status_must_remain": "IN_PROGRESS",
        "authority_boundary.current_l3_claim_status_must_remain": "NOT_ESTABLISHED",
        "activation_preconditions.explicit_owner_execution_authorization": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.active_protocol_registration": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.a2_frozen_development_split": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.qualified_ordinary_public_training_records": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.frozen_calibrated_lcb_terminal_score_manifest": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.rights_and_exposure_manifest": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.cuda_device_assignment": "REQUIRED_NOT_PRESENT",
        "activation_preconditions.all_must_be_present_before_any_parameter_update": True,
        "activation_preconditions.missing_precondition_action": "STOP_BEFORE_DATA_MODEL_CUDA_AND_OUTPUT_IO",
        "formal_production_interface.state_type": "SOURCE_ANCHORED_ACYCLIC_SPARSE_EDIT_DAG",
        "formal_production_interface.action_types": ["SOURCE_BASE_TO_ALT_BASE", "STOP"],
        "formal_production_interface.candidate_budget_assignment_rule": "USE_SMALLEST_PRIMARY_BUDGET_GREATER_THAN_OR_EQUAL_TO_NET_EDIT_COUNT",
        "formal_production_interface.primary_edit_budgets": [1, 3, 5],
        "formal_production_interface.rate_time_dependence": "NONE_FOR_THIS_PROTOCOL",
        "formal_production_interface.terminal_tilt_time_dependence": "NONE_FOR_THIS_PROTOCOL",
        "formal_production_interface.general_time_inhomogeneous_exactness": "NOT_RUN",
        "formal_production_interface.source_sequence_immutable": True,
        "formal_production_interface.edited_position_may_recur": False,
        "formal_production_interface.revert_to_source_allowed": False,
        "formal_production_interface.remaining_budget_decrements_once_per_edit": True,
        "formal_production_interface.event_count_definition": "NET_SOURCE_RELATIVE_EDIT_COUNT",
        "formal_production_interface.hard_legality_before_any_rate_or_model_evaluation": True,
        "formal_production_interface.raw_alias_aggregation_key": "FULL_NEXT_EXTENDED_STATE",
        "formal_production_interface.alias_aggregation_before_normalization": True,
        "formal_production_interface.stop_is_positive_rate_competing_transition": True,
        "formal_production_interface.canonical_support_floor": 1e-08,
        "formal_production_interface.base_total_exit_hazard": 1.0,
        "formal_production_interface.free_action_ratio_head_allowed": False,
        "ordinary_public_data_contract.data_scope": "QUALIFIED_ORDINARY_PUBLIC_ONLY",
        "ordinary_public_data_contract.allowed_record_role": "A2_FROZEN_DEVELOPMENT_ONLY",
        "ordinary_public_data_contract.forbidden_roles": [
            "OUTER_TEST",
            "CONFIRMATORY_TEST",
            "SEALED",
            "PRIVATE_CANONICAL",
            "REJECTED_OR_EXCLUDED",
            "UNQUALIFIED_STUDY",
        ],
        "ordinary_public_data_contract.row_identifier_as_model_feature_allowed": False,
        "ordinary_public_data_contract.study_identifier_as_random_embedding_allowed": False,
        "ordinary_public_data_contract.significance_or_p_value_selection_allowed": False,
        "ordinary_public_data_contract.unpublished_private_input_allowed": False,
        "ordinary_public_data_contract.sealed_input_allowed": False,
        "ordinary_public_data_contract.public_member_level_output_allowed": False,
        "ordinary_public_data_contract.public_sequence_output_allowed": False,
        "ordinary_public_data_contract.public_effect_output_allowed": False,
        "ordinary_public_data_contract.public_prediction_output_allowed": False,
        "split_contract.parent_split_authority": "A2_FROZEN_OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_COMPONENT_SPLIT",
        "split_contract.components_indivisible": True,
        "split_contract.development_subroles": [
            "A6_PARAMETER_TRAIN",
            "A6_INDEPENDENT_EXACT_REFERENCE",
        ],
        "split_contract.reference_fraction": 0.2,
        "split_contract.assignment_salt": "ROUTE_A_V3_A6_LEARNED_BASE_VALUE_GPU_DRAFT_V1",
        "split_contract.label_blind_assignment": True,
        "split_contract.split_retry_after_labels_or_results_allowed": False,
        "split_contract.outer_test_label_access_allowed": False,
        "exposure_contract.model_input_route": "SCRATCH_ONLY_RANDOM_INITIALIZATION",
        "exposure_contract.pretrained_foundation_checkpoints": [],
        "exposure_contract.pretrained_weights": [],
        "exposure_contract.warm_start_checkpoints": [],
        "exposure_contract.external_learned_embeddings": [],
        "exposure_contract.external_pretraining_corpora": [],
        "exposure_contract.checkpoint_loading_before_first_optimizer_step_allowed": False,
        "exposure_contract.scratch_route_is_checkpoint_exposure_waiver": False,
        "rights_contract.aggregate_public_reporting_only": True,
        "rights_contract.raw_or_member_level_redistribution_allowed": False,
        "base_architecture.parameter_initialization": "RANDOM_FROM_FIXED_SEED_ONLY",
        "base_architecture.pretrained_components_allowed": False,
        "base_architecture.per_position_feature_width": 28,
        "base_architecture.encoder.residual_dilations": [1, 2, 4, 8],
        "base_architecture.encoder.dropout": 0.0,
        "base_architecture.encoder.global_sequence_width": 128,
        "base_architecture.observable_context_encoder.embedding_width": 16,
        "base_architecture.observable_context_encoder.context_dropout": 0.1,
        "base_architecture.observable_context_encoder.study_id_embedding_allowed": False,
        "base_architecture.state_vector_width": 154,
        "base_architecture.action_vector_width": 225,
        "base_architecture.rate_head_hidden_widths": [128, 64],
        "base_architecture.illegal_action_mask_value": "NOT_EVALUATED",
        "base_architecture.canonical_transition_only_scoring": True,
        "value_architecture.base_encoder_parameters_shared": False,
        "value_architecture.base_checkpoint_frozen_before_value_training": True,
        "value_architecture.value_head_hidden_widths": [128, 64],
        "value_architecture.positive_h_parameterization": "1e-6+softplus(raw_value)",
        "value_architecture.free_action_ratio_head_allowed": False,
        "terminal_tilt_contract.input_score": "FROZEN_CALIBRATED_LOWER_CONFIDENCE_BOUND_FROM_ORDINARY_DEVELOPMENT_ROLE",
        "terminal_tilt_contract.bare_critic_mean_allowed": False,
        "terminal_tilt_contract.mad_floor": 1e-06,
        "terminal_tilt_contract.standardized_score_clip": [-4.0, 4.0],
        "terminal_tilt_contract.beta": 1.0,
        "terminal_tilt_contract.strictly_positive": True,
        "terminal_tilt_contract.numerical_failure_weight_defined": False,
        "terminal_tilt_contract.terminal_tilt_frozen_before_value_parameter_updates": True,
        "training_objectives.base_objective.name": "EXACT_OBSERVED_TERMINAL_NEGATIVE_LOG_LIKELIHOOD",
        "training_objectives.base_objective.eligible_edit_count_max": 5,
        "training_objectives.base_objective.rate_scale_identification": "FIXED_TOTAL_EXIT_HAZARD_ONE",
        "training_objectives.value_objective.name": "EXACT_HARMONIC_EXTENSION_LOG_H_HUBER",
        "training_objectives.value_objective.base_parameters_updated": False,
        "training_objectives.value_objective.editable_positions_per_graph": 5,
        "training_objectives.value_objective.budgets": [1, 3, 5],
        "training_objectives.value_objective.huber_delta": 0.1,
        "training_objectives.joint_training_allowed": False,
        "training_objectives.base_then_value_order_required": True,
        "independent_exact_reference.implementation_role": "CPU_STDLIB_OR_NUMPY_NO_TORCH_NO_LEARNER_IMPORT",
        "independent_exact_reference.graph_count": 96,
        "independent_exact_reference.graphs_per_budget": 32,
        "independent_exact_reference.graph_unit": "ONE_FROZEN_SOURCE_GROUP_CONTEXT_WITH_FIVE_LABEL_BLIND_EDITABLE_POSITIONS",
        "independent_exact_reference.graph_selection_rule": "SORT_ELIGIBLE_FROZEN_COMPONENT_IDS_BY_SHA256_OF_ASSIGNMENT_SALT_AND_COMPONENT_ID_THEN_TAKE_FIRST_32_PER_BUDGET",
        "independent_exact_reference.insufficient_graph_action": "FAIL_CURRENT_PROTOCOL_KEEP_A6_IN_PROGRESS",
        "independent_exact_reference.budgets": [1, 3, 5],
        "independent_exact_reference.dp_and_exhaustive_path_enumeration_must_both_run": True,
        "independent_exact_reference.dp_vs_enumeration_terminal_tv_max": 1e-12,
        "independent_exact_reference.reference_graphs_may_train_parameters": False,
        "independent_exact_reference.reference_results_may_select_checkpoint": False,
        "learned_potential_approximation_gate.primary_threshold_max": 0.1,
        "learned_potential_approximation_gate.secondary_threshold_max": 0.02,
        "learned_potential_approximation_gate.all_reference_graphs_required": True,
        "learned_potential_approximation_gate.abstention_or_graph_deletion_allowed": False,
        "learned_potential_approximation_gate.failed_gate_action": "FAIL_CURRENT_PROTOCOL_KEEP_A6_IN_PROGRESS",
        "optimizer_compute_checkpoint_policy.seed": 2026081401,
        "optimizer_compute_checkpoint_policy.seed_count": 1,
        "optimizer_compute_checkpoint_policy.optimizer": "ADAMW",
        "optimizer_compute_checkpoint_policy.learning_rate": 0.0003,
        "optimizer_compute_checkpoint_policy.precision": "FLOAT32",
        "optimizer_compute_checkpoint_policy.source_group_batch_size": 32,
        "optimizer_compute_checkpoint_policy.base_optimizer_steps": 40000,
        "optimizer_compute_checkpoint_policy.value_optimizer_steps": 20000,
        "optimizer_compute_checkpoint_policy.total_optimizer_steps": 60000,
        "optimizer_compute_checkpoint_policy.early_stopping_allowed": False,
        "optimizer_compute_checkpoint_policy.hyperparameter_search_allowed": False,
        "optimizer_compute_checkpoint_policy.best_checkpoint_selection_allowed": False,
        "optimizer_compute_checkpoint_policy.recovery_checkpoint_role": "CRASH_RESUME_ONLY_NOT_SELECTION",
        "optimizer_compute_checkpoint_policy.scientific_checkpoint_roles": [
            "BASE_FINAL_STEP_40000",
            "VALUE_FINAL_STEP_20000",
        ],
        "optimizer_compute_checkpoint_policy.matched_compute_claim_allowed_in_a6": False,
        "cuda_fail_closed_contract.parameter_updates_require_cuda": True,
        "cuda_fail_closed_contract.cpu_parameter_updates_allowed": False,
        "cuda_fail_closed_contract.silent_cpu_fallback_allowed": False,
        "cuda_fail_closed_contract.single_gpu_required": True,
        "cuda_fail_closed_contract.failed_check_action": "FAIL_CLOSED_ZERO_PARAMETER_UPDATE_ZERO_OUTPUT",
        "required_gates.base_recovery.unit_terminal_tilt_recovers_base": True,
        "required_gates.base_recovery.operator_fixture": "SET_TERMINAL_WEIGHT_W_TO_ONE_AND_USE_EXACT_CONSTANT_V_ZERO_NOT_LEARNED_NETWORK_OUTPUT",
        "required_gates.base_recovery.max_rate_relative_error": 1e-05,
        "required_gates.legality_and_support.hard_legality_fraction": 1.0,
        "required_gates.legality_and_support.budget_violations_max": 0,
        "required_gates.legality_and_support.support_coverage_fraction": 1.0,
        "required_gates.trajectory.deterministic_replay_match_fraction": 1.0,
        "required_gates.trajectory.trajectory_legality_fraction": 1.0,
        "required_gates.trajectory.trajectory_budget_violations_max": 0,
        "required_gates.trajectory.statistical_trajectory_count": 20000,
        "required_gates.trajectory.statistical_seed": 2026081402,
        "required_gates.trajectory.holding_time_mean_relative_error_max": 0.02,
        "required_gates.trajectory.sampled_terminal_distribution_tv_max": 0.02,
        "required_gates.failure_cannot_be_rescued_by_checkpoint_seed_or_threshold_selection": True,
        "future_provenance_and_manifest_outputs.this_draft_creates_runtime_outputs": False,
        "future_provenance_and_manifest_outputs.public_outputs_aggregate_only": True,
        "future_provenance_and_manifest_outputs.member_ids_sequences_effects_predictions_split_assignments_public": False,
        "terminal_truth.status": "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL",
        "terminal_truth.training": "NOT_AUTHORIZED_NOT_RUN",
        "terminal_truth.gpu": "NOT_AUTHORIZED_NOT_RUN",
        "terminal_truth.parameter_updates": 0,
        "terminal_truth.model_selection": "NOT_AUTHORIZED_NOT_RUN",
        "terminal_truth.a6": "IN_PROGRESS_UNCHANGED",
        "terminal_truth.flow_base_legal_ctmc_formal_task": "NOT_RUN_UNCHANGED",
        "terminal_truth.l3_claim": "NOT_ESTABLISHED_UNCHANGED",
        "terminal_truth.a7": "NOT_RUN_NOT_UNLOCKED",
        "terminal_truth.a1_scientific_counts": "UNCHANGED",
        "terminal_truth.canonical_state": "UNCHANGED",
    }
    for path, expected in exact_expectations.items():
        _expect(payload, path, expected, issues)

    optimizer = payload.get("optimizer_compute_checkpoint_policy", {})
    if isinstance(optimizer, dict):
        base_steps = optimizer.get("base_optimizer_steps")
        value_steps = optimizer.get("value_optimizer_steps")
        total_steps = optimizer.get("total_optimizer_steps")
        if not isinstance(base_steps, int) or not isinstance(value_steps, int):
            _issue(issues, "A6_DRAFT_COMPUTE_INVALID", "optimizer_compute_checkpoint_policy", "step counts must be integers")
        elif total_steps != base_steps + value_steps:
            _issue(issues, "A6_DRAFT_COMPUTE_INVALID", "optimizer_compute_checkpoint_policy.total_optimizer_steps", "total must equal base plus value steps")

    reference = payload.get("independent_exact_reference", {})
    if isinstance(reference, dict):
        if reference.get("graph_count") != reference.get("graphs_per_budget") * len(reference.get("budgets", [])):
            _issue(issues, "A6_DRAFT_REFERENCE_COUNT_INVALID", "independent_exact_reference", "graph count must close across budgets")

    gates = payload.get("learned_potential_approximation_gate", {})
    if isinstance(gates, dict):
        for field in ("primary_threshold_max", "secondary_threshold_max"):
            value = gates.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                _issue(issues, "A6_DRAFT_THRESHOLD_INVALID", f"learned_potential_approximation_gate.{field}", "threshold must be finite and positive")

    return issues


def validate_file(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [{"code": "A6_DRAFT_CONFIG_UNREADABLE", "path": str(path), "detail": str(exc)}]
    if not isinstance(payload, dict):
        return [{"code": "A6_DRAFT_ROOT_TYPE", "path": "$", "detail": "root must be a mapping"}]
    return validate_config(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    config_path = args.repo_root.resolve() / CONFIG_PATH
    issues = validate_file(config_path)
    result = {
        "protocol_id": EXPECTED_STATUS["protocol_id"],
        "document_status": EXPECTED_STATUS["document_status"],
        "validator_mode": "STATIC_READ_ONLY_NO_EXECUTION",
        "issue_count": len(issues),
        "issues": issues,
        "training_authorized": False,
        "gpu_run_authorized": False,
        "parameter_updates": 0,
        "claim_status": "NOT_ESTABLISHED",
    }
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{len(issues)} issue(s)")
        for issue in issues:
            print(f"{issue['code']}: {issue['path']}: {issue['detail']}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
