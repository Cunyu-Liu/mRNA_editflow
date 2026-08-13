from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_a6_learned_base_value_gpu_protocol_draft_v1.json"
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/validate_a6_learned_base_value_gpu_protocol_draft.py"
DOC_PATH = REPO_ROOT / "docs/plans/2026-08-14-route-a-v3-a6-learned-base-value-gpu-protocol-draft-v1.md"


def _load_validator():
    spec = importlib.util.spec_from_file_location("a6_gpu_draft_validator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _set(payload: dict, path: str, value) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def test_draft_is_closed_nonactive_and_static_only() -> None:
    validator = _load_validator()
    payload = _payload()
    assert validator.validate_config(payload) == []
    assert payload["document_status"] == "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
    assert payload["authority_status"] == "NON_AUTHORITATIVE"
    assert payload["activation_state"] == "INACTIVE_REVIEW_CANDIDATE"
    assert payload["authority_boundary"]["registered_in_active_authority"] is False
    assert payload["authority_boundary"]["registered_in_static_manifest"] is False
    assert payload["future_provenance_and_manifest_outputs"]["this_draft_creates_runtime_outputs"] is False
    assert payload["terminal_truth"]["flow_base_legal_ctmc_formal_task"] == "NOT_RUN_UNCHANGED"
    assert payload["terminal_truth"]["a6"] == "IN_PROGRESS_UNCHANGED"
    assert payload["terminal_truth"]["l3_claim"] == "NOT_ESTABLISHED_UNCHANGED"
    assert payload["terminal_truth"]["a7"] == "NOT_RUN_NOT_UNLOCKED"
    assert "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL" in DOC_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("forbidden_state_changes.training_started", True),
        ("forbidden_state_changes.training_allowed", True),
        ("forbidden_state_changes.training_authorized", True),
        ("forbidden_state_changes.parameter_updates_allowed", True),
        ("forbidden_state_changes.gpu_work_allowed", True),
        ("forbidden_state_changes.model_selection_allowed", True),
        ("forbidden_state_changes.a6_pass_asserted", True),
        ("forbidden_state_changes.l3_claim_established", True),
        ("forbidden_state_changes.a7_unlocked", True),
        ("forbidden_state_changes.private_data_access_allowed", True),
        ("forbidden_state_changes.sealed_data_access_allowed", True),
        ("forbidden_state_changes.ordinary_study_credit_delta", 1),
        ("forbidden_state_changes.a1_study_credit_delta", 1),
        ("forbidden_state_changes.true_a2_study_credit_delta", 1),
        ("forbidden_state_changes.canonical_record_count_delta", 1),
    ],
)
def test_activation_or_scientific_promotion_is_rejected(path: str, value) -> None:
    validator = _load_validator()
    payload = _payload()
    _set(payload, path, value)
    assert validator.validate_config(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("formal_production_interface.edited_position_may_recur", True),
        ("formal_production_interface.revert_to_source_allowed", True),
        ("formal_production_interface.candidate_budget_assignment_rule", "EXACT_BUDGET_ONLY"),
        ("formal_production_interface.hard_legality_before_any_rate_or_model_evaluation", False),
        ("formal_production_interface.alias_aggregation_before_normalization", False),
        ("formal_production_interface.raw_alias_aggregation_key", "CURRENT_SEQUENCE_ONLY"),
        ("formal_production_interface.stop_is_positive_rate_competing_transition", False),
        ("formal_production_interface.canonical_support_floor", 0.0),
        ("formal_production_interface.free_action_ratio_head_allowed", True),
        ("formal_production_interface.rate_time_dependence", "LEARNED_CONTINUOUS_TIME"),
    ],
)
def test_legal_ctmc_semantic_drift_is_rejected(path: str, value) -> None:
    validator = _load_validator()
    payload = _payload()
    _set(payload, path, value)
    assert validator.validate_config(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("ordinary_public_data_contract.allowed_record_role", "OUTER_TEST"),
        ("ordinary_public_data_contract.study_identifier_as_random_embedding_allowed", True),
        ("ordinary_public_data_contract.public_sequence_output_allowed", True),
        ("split_contract.components_indivisible", False),
        ("split_contract.split_retry_after_labels_or_results_allowed", True),
        ("split_contract.outer_test_label_access_allowed", True),
        ("exposure_contract.pretrained_weights", ["some-checkpoint"]),
        ("exposure_contract.checkpoint_loading_before_first_optimizer_step_allowed", True),
        ("exposure_contract.scratch_route_is_checkpoint_exposure_waiver", True),
        ("rights_contract.raw_or_member_level_redistribution_allowed", True),
    ],
)
def test_data_split_exposure_and_rights_drift_is_rejected(path: str, value) -> None:
    validator = _load_validator()
    payload = _payload()
    _set(payload, path, value)
    assert validator.validate_config(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("base_architecture.pretrained_components_allowed", True),
        ("base_architecture.encoder.residual_dilations", [1, 2, 4]),
        ("base_architecture.observable_context_encoder.study_id_embedding_allowed", True),
        ("base_architecture.illegal_action_mask_value", "NEGATIVE_INFINITY_AFTER_EVALUATION"),
        ("value_architecture.base_encoder_parameters_shared", True),
        ("value_architecture.base_checkpoint_frozen_before_value_training", False),
        ("value_architecture.free_action_ratio_head_allowed", True),
        ("terminal_tilt_contract.bare_critic_mean_allowed", True),
        ("terminal_tilt_contract.terminal_tilt_frozen_before_value_parameter_updates", False),
        ("training_objectives.joint_training_allowed", True),
        ("training_objectives.value_objective.base_parameters_updated", True),
        ("independent_exact_reference.reference_graphs_may_train_parameters", True),
        ("independent_exact_reference.reference_results_may_select_checkpoint", True),
        ("independent_exact_reference.graph_unit", "ONE_CANDIDATE_ROW"),
        ("independent_exact_reference.insufficient_graph_action", "USE_FEWER_GRAPHS"),
        ("learned_potential_approximation_gate.primary_threshold_max", 0.2),
        ("learned_potential_approximation_gate.abstention_or_graph_deletion_allowed", True),
    ],
)
def test_architecture_objective_reference_and_gate_drift_is_rejected(path: str, value) -> None:
    validator = _load_validator()
    payload = _payload()
    _set(payload, path, value)
    assert validator.validate_config(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("optimizer_compute_checkpoint_policy.seed_count", 5),
        ("optimizer_compute_checkpoint_policy.optimizer", "SGD"),
        ("optimizer_compute_checkpoint_policy.base_optimizer_steps", 50000),
        ("optimizer_compute_checkpoint_policy.early_stopping_allowed", True),
        ("optimizer_compute_checkpoint_policy.hyperparameter_search_allowed", True),
        ("optimizer_compute_checkpoint_policy.best_checkpoint_selection_allowed", True),
        ("optimizer_compute_checkpoint_policy.recovery_checkpoint_role", "BEST_VALIDATION"),
        ("cuda_fail_closed_contract.parameter_updates_require_cuda", False),
        ("cuda_fail_closed_contract.cpu_parameter_updates_allowed", True),
        ("cuda_fail_closed_contract.silent_cpu_fallback_allowed", True),
        ("required_gates.base_recovery.operator_fixture", "LEARNED_NETWORK_OUTPUT"),
        ("required_gates.trajectory.statistical_trajectory_count", 1000),
    ],
)
def test_compute_checkpoint_and_cuda_drift_is_rejected(path: str, value) -> None:
    validator = _load_validator()
    payload = _payload()
    _set(payload, path, value)
    assert validator.validate_config(payload)


def test_zero_access_attestation_and_reference_count_are_fail_closed() -> None:
    validator = _load_validator()
    payload = _payload()
    payload["design_access_attestation"]["ordinary_row_reads"] = 1
    assert validator.validate_config(payload)

    payload = _payload()
    payload["independent_exact_reference"]["graph_count"] = 95
    issues = validator.validate_config(payload)
    assert any(issue["code"] == "A6_DRAFT_REFERENCE_COUNT_INVALID" for issue in issues)


def test_static_validator_imports_no_training_or_execution_runtime() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {"__future__", "argparse", "json", "pathlib", "typing"}
