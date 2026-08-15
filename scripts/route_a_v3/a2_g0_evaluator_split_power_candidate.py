#!/usr/bin/env python3
"""Synthetic-only Route A V3 A2 evaluator/split/power G0 candidate.

This module is deliberately non-authoritative.  Its command-line surface only
validates the frozen draft config and reproduces the pre-declared planning
calculation.  It does not read project or dataset rows, generate a real split,
evaluate a model, write an artifact, touch CUDA, or change scientific state.

Imported helpers accept only an explicit ``SYNTHETIC_TEST_FIXTURE_ONLY`` scope
and return aggregate summaries.  A later reviewed successor and explicit owner
authority are required before any real membership, assignment, evaluation, or
qualification work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_a2_g0_evaluator_candidate_v1.json"

SCHEMA_VERSION = "route_a_v3_a2_g0_evaluator_candidate.v1"
CANDIDATE_ID = "ROUTE_A_V3_A2_G0_EVALUATOR_SPLIT_POWER_CANDIDATE_V1"
DOCUMENT_STATUS = "DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL"
SYNTHETIC_SCOPE = "SYNTHETIC_TEST_FIXTURE_ONLY"
CONFIG_PASS = "PASS_DRAFT_INTERFACE_ONLY_NOT_ACTIVE_NOT_SCIENTIFIC_GATE"
SYNTHETIC_PASS = "PASS_SYNTHETIC_INTERFACE_ONLY_NOT_QUALIFICATION"
INVALID_CONTRACT = "STOP_FAIL_CLOSED_CONTRACT_INVALID"
UNDEFINED_METRIC = "STOP_FAIL_CLOSED_METRIC_UNDEFINED"

POWER_METHOD = "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN"
CI_METHOD = "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE"
WORKING_ASSUMPTION = (
    "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO"
)


class CandidateError(RuntimeError):
    """Base exception for this inactive candidate."""


class ContractError(CandidateError):
    """Raised when a draft or synthetic interface is not closed."""


class MetricUndefinedError(CandidateError):
    """Raised when a requested synthetic metric is mathematically undefined."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read candidate config: {path}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid candidate JSON: {path}") from exc
    if type(value) is not dict:
        raise ContractError("candidate config root must be an object")
    return value


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ContractError(f"{label} keys differ: missing={missing}, extra={extra}")


def _is_plain_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def bonett_wright_fisher_z_plan(
    effective_n: int,
    *,
    alternative_spearman_rho: float,
    two_sided_alpha: float,
    confidence_level: float,
    target_power: float,
    maximum_full_ci_width: float,
) -> dict[str, Any]:
    """Return the frozen asymptotic planning calculation for one effective N."""

    if isinstance(effective_n, bool) or not isinstance(effective_n, int) or effective_n <= 3:
        raise ContractError("effective N must be an integer greater than 3")
    numbers = {
        "alternative_spearman_rho": alternative_spearman_rho,
        "two_sided_alpha": two_sided_alpha,
        "confidence_level": confidence_level,
        "target_power": target_power,
        "maximum_full_ci_width": maximum_full_ci_width,
    }
    if any(not _is_plain_number(value) or not math.isfinite(float(value)) for value in numbers.values()):
        raise ContractError("power/precision inputs must be finite numbers")
    rho = float(alternative_spearman_rho)
    alpha = float(two_sided_alpha)
    confidence = float(confidence_level)
    power_target = float(target_power)
    width_target = float(maximum_full_ci_width)
    if not -1.0 < rho < 1.0 or rho == 0.0:
        raise ContractError("alternative Spearman rho must be nonzero and inside (-1, 1)")
    if not 0.0 < alpha < 1.0 or not 0.0 < confidence < 1.0:
        raise ContractError("alpha and confidence level must be inside (0, 1)")
    if not 0.0 < power_target < 1.0 or not 0.0 < width_target < 2.0:
        raise ContractError("power and full-CI-width thresholds are invalid")

    normal = NormalDist()
    null_standard_error = 1.0 / math.sqrt(effective_n - 3)
    alternative_z = math.atanh(rho)
    alternative_standard_error = (
        math.sqrt(1.0 + rho**2 / 2.0) * null_standard_error
    )
    rejection_boundary = normal.inv_cdf(1.0 - alpha / 2.0) * null_standard_error
    estimated_power = (
        1.0
        - normal.cdf((rejection_boundary - alternative_z) / alternative_standard_error)
        + normal.cdf((-rejection_boundary - alternative_z) / alternative_standard_error)
    )
    confidence_critical = normal.inv_cdf(0.5 + confidence / 2.0)
    lower = math.tanh(alternative_z - confidence_critical * alternative_standard_error)
    upper = math.tanh(alternative_z + confidence_critical * alternative_standard_error)
    full_width = upper - lower
    power_pass = estimated_power >= power_target
    precision_pass = full_width <= width_target
    return {
        "effective_n": effective_n,
        "analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
        "alternative_spearman_rho": rho,
        "two_sided_alpha": alpha,
        "target_power": power_target,
        "confidence_level": confidence,
        "maximum_full_confidence_interval_width": width_target,
        "power_method": POWER_METHOD,
        "confidence_interval_method": CI_METHOD,
        "null_fisher_z_standard_error": null_standard_error,
        "alternative_fisher_z_standard_error": alternative_standard_error,
        "estimated_design_power": estimated_power,
        "planned_confidence_interval_lower": lower,
        "planned_confidence_interval_upper": upper,
        "planned_full_confidence_interval_width": full_width,
        "power_pass": power_pass,
        "precision_pass": precision_pass,
        "both_thresholds_pass": power_pass and precision_pass,
        "observed_model_power_claimed": False,
        "actual_model_confidence_interval_claimed": False,
        "formal_qualification_gate_executed": False,
    }


def minimum_effective_n_for_power_and_precision(
    *,
    alternative_spearman_rho: float,
    two_sided_alpha: float,
    confidence_level: float,
    target_power: float,
    maximum_full_ci_width: float,
) -> int:
    """Find the first independent-source-group N satisfying both frozen rules."""

    for effective_n in range(4, 100_001):
        plan = bonett_wright_fisher_z_plan(
            effective_n,
            alternative_spearman_rho=alternative_spearman_rho,
            two_sided_alpha=two_sided_alpha,
            confidence_level=confidence_level,
            target_power=target_power,
            maximum_full_ci_width=maximum_full_ci_width,
        )
        if plan["both_thresholds_pass"]:
            return effective_n
    raise ContractError("no effective N up to 100000 satisfies the frozen thresholds")


def validate_candidate_config(config: Mapping[str, Any]) -> None:
    root_keys = {
        "schema_version",
        "candidate_id",
        "document_status",
        "authority_status",
        "activation_state",
        "phase_role",
        "governance_boundary",
        "permitted_g0_operations",
        "forbidden_operations",
        "synthetic_scope_contract",
        "structural_graph_contract",
        "split_plan_contract",
        "endpoint_effect_se_contract",
        "power_precision_contract",
        "evaluator_metric_schema",
        "fail_closed_status_contract",
        "validate_only_truth",
    }
    _expect_exact_keys(config, root_keys, "candidate config")
    exact = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "document_status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "activation_state": "INACTIVE_G0_IMPLEMENTATION_CANDIDATE",
        "phase_role": "A2_SCHEMA_EVALUATOR_SPLIT_AND_POWER_DRAFTING_ONLY",
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ContractError(f"candidate {key} differs")

    governance = config["governance_boundary"]
    _expect_exact_keys(
        governance,
        {
            "design_inputs",
            "current_qualified_counts",
            "changes_current_qualified_counts",
            "final_a2_membership_frozen",
            "final_split_assignments_frozen",
            "promotion_requires_new_explicit_owner_authority",
        },
        "governance boundary",
    )
    if governance["design_inputs"] != [
        "outputs/route_a_v3_execution_snapshot_20260814.md",
        "outputs/route_a_v3_a1_a2_gate_bridge_proposal_20260814.md",
    ]:
        raise ContractError("design-input references differ")
    if governance["current_qualified_counts"] != {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }:
        raise ContractError("design-time scientific-state projection differs")
    for key in (
        "changes_current_qualified_counts",
        "final_a2_membership_frozen",
        "final_split_assignments_frozen",
    ):
        if governance[key] is not False:
            raise ContractError(f"governance boundary is not fail-closed: {key}")
    if governance["promotion_requires_new_explicit_owner_authority"] is not True:
        raise ContractError("candidate could be promoted without new owner authority")

    expected_permitted = [
        "VALIDATE_THIS_DRAFT_CONFIG",
        "VALIDATE_SYNTHETIC_ENDPOINT_EFFECT_SE_MANIFESTS",
        "BUILD_SYNTHETIC_SOURCE_GROUP_KNOWN_DUPLICATE_COMPONENTS",
        "GENERATE_SYNTHETIC_OUTCOME_BLIND_AGGREGATE_SPLIT_PLANS",
        "EVALUATE_SYNTHETIC_AGGREGATE_METRICS",
        "CALCULATE_PREFROZEN_BONETT_WRIGHT_FISHER_Z_PLANNING_VALUES",
        "VALIDATE_ONLY_TO_STANDARD_OUTPUT",
    ]
    if config["permitted_g0_operations"] != expected_permitted:
        raise ContractError("permitted G0 operation set differs")
    required_forbidden = {
        "READ_PROJECT_OR_DATASET_ROWS",
        "READ_MEMBER_IDENTIFIERS_OR_SEQUENCES",
        "READ_PRIVATE_OR_SEALED_DATA",
        "WRITE_RUNTIME_ARTIFACTS",
        "PUBLISH_MEMBER_OR_SPLIT_ASSIGNMENTS",
        "FREEZE_FINAL_A2_MEMBERSHIP",
        "EXECUTE_FORMAL_QUALIFICATION_OR_POWER_GATE",
        "CHANGE_QUALIFICATION_CREDIT_OR_CANONICAL_STATE",
        "TRAIN_OR_UPDATE_PARAMETERS",
        "TOUCH_CUDA_OR_GPU",
        "SELECT_MODEL_OR_CHECKPOINT",
        "UNLOCK_A7_OR_NEXT_PHASE",
    }
    if set(config["forbidden_operations"]) != required_forbidden or len(
        config["forbidden_operations"]
    ) != len(required_forbidden):
        raise ContractError("forbidden-operation closure is incomplete")

    scope = config["synthetic_scope_contract"]
    if scope != {
        "required_scope_token": SYNTHETIC_SCOPE,
        "synthetic_identifier_prefix": "synthetic_",
        "project_or_dataset_rows_allowed": False,
        "member_or_sequence_payload_allowed": False,
        "synthetic_record_keys_may_be_returned": False,
        "split_assignments_may_be_returned": False,
    }:
        raise ContractError("synthetic-only scope differs")

    graph = config["structural_graph_contract"]
    _expect_exact_keys(
        graph,
        {
            "record_schema_version",
            "required_record_keys_exactly",
            "forbidden_outcome_field_tokens",
            "component_edges",
            "connected_components_are_indivisible",
            "unknown_duplicate_reference_status",
            "duplicate_record_key_status",
        },
        "structural graph contract",
    )
    if graph["record_schema_version"] != "route_a_v3_a2_g0_synthetic_structural_record.v1":
        raise ContractError("structural-record schema differs")
    if graph["required_record_keys_exactly"] != [
        "record_key",
        "source_group",
        "known_duplicate_record_keys",
    ]:
        raise ContractError("structural-record key closure differs")
    if graph["component_edges"] != [
        "SAME_SOURCE_GROUP",
        "KNOWN_DUPLICATE_RECORD_REFERENCE",
    ] or graph["connected_components_are_indivisible"] is not True:
        raise ContractError("connected-component semantics differ")
    required_outcome_tokens = {
        "endpoint",
        "effect",
        "label",
        "outcome",
        "prediction",
        "standard_error",
        "se",
    }
    if set(graph["forbidden_outcome_field_tokens"]) != required_outcome_tokens:
        raise ContractError("outcome-blind forbidden-field closure differs")

    split = config["split_plan_contract"]
    _expect_exact_keys(
        split,
        {
            "plan_id",
            "input_scope",
            "outcome_blind",
            "assignment_unit",
            "fold_ids",
            "allocation_rule",
            "synthetic_validation_salt",
            "synthetic_salt_is_final_a2_salt",
            "future_final_salt_status",
            "all_folds_must_be_nonempty",
            "source_group_cross_fold_count_max",
            "known_duplicate_cross_fold_count_max",
            "component_cross_fold_count_max",
            "output_class",
            "output_includes_record_or_component_keys",
            "output_includes_split_assignments",
        },
        "split plan contract",
    )
    expected_folds = [f"OUTER_FOLD_{index}" for index in range(5)]
    if split["input_scope"] != SYNTHETIC_SCOPE or split["outcome_blind"] is not True:
        raise ContractError("split plan is not synthetic and outcome-blind")
    if split["assignment_unit"] != "SOURCE_GROUP_AND_KNOWN_DUPLICATE_CONNECTED_COMPONENT":
        raise ContractError("split assignment unit differs")
    if split["allocation_rule"] != (
        "DESCENDING_COMPONENT_SIZE_THEN_SYNTHETIC_SALTED_ORDER_TO_MINIMUM_RECORD_COUNT_FOLD"
    ):
        raise ContractError("split allocation rule differs")
    if split["fold_ids"] != expected_folds or len(set(split["fold_ids"])) != 5:
        raise ContractError("five-fold identity closure differs")
    if split["synthetic_salt_is_final_a2_salt"] is not False:
        raise ContractError("synthetic salt cannot be the final A2 salt")
    if split["future_final_salt_status"] != "NOT_FROZEN":
        raise ContractError("final A2 split salt was prematurely frozen")
    for key in (
        "source_group_cross_fold_count_max",
        "known_duplicate_cross_fold_count_max",
        "component_cross_fold_count_max",
    ):
        if split[key] != 0:
            raise ContractError(f"split leakage maximum differs: {key}")
    if split["output_includes_record_or_component_keys"] is not False:
        raise ContractError("split output would expose record/component keys")
    if split["output_includes_split_assignments"] is not False:
        raise ContractError("split output would expose assignments")

    endpoint = config["endpoint_effect_se_contract"]
    _expect_exact_keys(
        endpoint,
        {
            "manifest_schema_version",
            "required_manifest_keys_exactly",
            "allowed_endpoint_transforms",
            "direction_multiplier_by_endpoint_direction",
            "required_effect_definition",
            "required_effect_analysis_unit",
            "allowed_standard_error_estimators",
            "required_standard_error_analysis_unit",
            "required_independent_replicate_unit",
            "minimum_independent_biological_replicates",
            "technical_replicates_may_count_as_biological",
            "required_missing_policy",
            "required_nonfinite_policy",
            "censoring_or_selection_rule_must_be_prefrozen",
        },
        "endpoint/effect/SE contract",
    )
    required_endpoint_keys = {
        "schema_version",
        "input_scope",
        "endpoint_name",
        "endpoint_scale",
        "endpoint_transform",
        "endpoint_direction",
        "direction_multiplier",
        "effect_definition",
        "effect_analysis_unit",
        "standard_error_estimator",
        "standard_error_analysis_unit",
        "independent_replicate_unit",
        "minimum_independent_biological_replicates",
        "technical_replicates_may_count_as_biological",
        "missing_policy",
        "nonfinite_policy",
        "censoring_or_selection_rule_prefrozen",
    }
    if set(endpoint["required_manifest_keys_exactly"]) != required_endpoint_keys:
        raise ContractError("endpoint/effect/SE manifest closure differs")
    if endpoint["direction_multiplier_by_endpoint_direction"] != {
        "HIGHER_IS_BETTER": 1,
        "LOWER_IS_BETTER": -1,
    }:
        raise ContractError("endpoint direction normalization differs")
    if endpoint["minimum_independent_biological_replicates"] != 3:
        raise ContractError("minimum independent biological replicates differs")
    if endpoint["technical_replicates_may_count_as_biological"] is not False:
        raise ContractError("technical replicates could count as biological")
    if endpoint["censoring_or_selection_rule_must_be_prefrozen"] is not True:
        raise ContractError("censoring/selection rule need not be prefrozen")

    power = config["power_precision_contract"]
    expected_power = {
        "analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
        "target_metric": "WITHIN_STUDY_SPEARMAN",
        "alternative_spearman_rho": 0.25,
        "two_sided_alpha": 0.05,
        "target_power": 0.8,
        "confidence_level": 0.95,
        "maximum_full_confidence_interval_width": 0.3,
        "power_method": POWER_METHOD,
        "confidence_interval_method": CI_METHOD,
        "null_standard_error_formula": "1/sqrt(n-3)",
        "alternative_standard_error_formula": "sqrt(1+rho^2/2)/sqrt(n-3)",
        "working_distribution_assumption": WORKING_ASSUMPTION,
        "required_effective_n_for_both_power_and_ci_width": 156,
        "n_is_post_dedup_independent_source_groups_not_rows": True,
        "formal_gate_execution_allowed": False,
    }
    if power != expected_power:
        raise ContractError("power/precision contract differs")
    minimum_n = minimum_effective_n_for_power_and_precision(
        alternative_spearman_rho=power["alternative_spearman_rho"],
        two_sided_alpha=power["two_sided_alpha"],
        confidence_level=power["confidence_level"],
        target_power=power["target_power"],
        maximum_full_ci_width=power["maximum_full_confidence_interval_width"],
    )
    if minimum_n != power["required_effective_n_for_both_power_and_ci_width"]:
        raise ContractError("declared required effective N is not the first passing N")

    metric = config["evaluator_metric_schema"]
    _expect_exact_keys(
        metric,
        {
            "schema_version",
            "input_scope",
            "analysis_unit",
            "primary_metric",
            "primary_score_direction",
            "rank_tie_method",
            "required_source_group_field",
            "required_prediction_field",
            "required_observation_field",
            "required_standard_error_field",
            "standard_error_requirement",
            "standard_error_usage",
            "secondary_diagnostic",
            "one_source_group_one_vote",
            "minimum_finite_independent_units",
            "constant_rank_input_status",
            "missing_or_nonfinite_input_status",
            "cross_study_summary_status",
            "output_class",
            "output_includes_member_or_source_group_keys",
            "metric_may_select_model_or_checkpoint",
        },
        "evaluator metric schema",
    )
    expected_metric_scalars = {
        "input_scope": SYNTHETIC_SCOPE,
        "analysis_unit": "POST_DEDUP_INDEPENDENT_SOURCE_GROUP",
        "primary_metric": "WITHIN_STUDY_SPEARMAN",
        "primary_score_direction": "HIGHER_IS_BETTER",
        "rank_tie_method": "AVERAGE_RANK",
        "required_source_group_field": "source_group_key",
        "required_prediction_field": "predicted_direction_normalized_effect",
        "required_observation_field": "observed_direction_normalized_effect",
        "required_standard_error_field": "observed_standard_error",
        "minimum_finite_independent_units": 4,
        "constant_rank_input_status": UNDEFINED_METRIC,
        "missing_or_nonfinite_input_status": INVALID_CONTRACT,
        "cross_study_summary_status": "NOT_FROZEN_PENDING_FINAL_A2_MEMBERSHIP",
        "output_class": "AGGREGATE_METRICS_ONLY",
        "output_includes_member_or_source_group_keys": False,
        "metric_may_select_model_or_checkpoint": False,
    }
    for key, expected in expected_metric_scalars.items():
        if metric[key] != expected:
            raise ContractError(f"evaluator metric schema differs: {key}")
    if metric["one_source_group_one_vote"] is not True:
        raise ContractError("evaluator does not give one source group one vote")
    if metric["standard_error_requirement"] != (
        "FINITE_STRICTLY_POSITIVE_BIOLOGICAL_STANDARD_ERROR"
    ) or metric["standard_error_usage"] != (
        "VALIDITY_GATE_AND_AGGREGATE_DIAGNOSTIC_ONLY_NOT_PRIMARY_METRIC_WEIGHTING"
    ):
        raise ContractError("evaluator standard-error semantics differ")
    if metric["secondary_diagnostic"] != "MEAN_ABSOLUTE_EFFECT_ERROR":
        raise ContractError("evaluator secondary diagnostic differs")

    statuses = config["fail_closed_status_contract"]
    if statuses != {
        "config_only_pass": CONFIG_PASS,
        "synthetic_interface_pass": SYNTHETIC_PASS,
        "invalid_contract": INVALID_CONTRACT,
        "undefined_metric": UNDEFINED_METRIC,
        "unknown_is_pass": False,
        "not_run_is_pass": False,
        "final_a2_gate_status": "NOT_RUN",
    }:
        raise ContractError("fail-closed status semantics differ")
    truth = config["validate_only_truth"]
    expected_truth_keys = {
        "project_rows_read",
        "dataset_rows_read",
        "member_identifiers_read",
        "sequences_read",
        "synthetic_fixture_rows_read",
        "split_assignments_generated",
        "metric_values_computed",
        "runtime_artifacts_written",
        "training_runs",
        "parameter_updates",
        "cuda_or_gpu_touches",
        "model_or_checkpoint_selections",
        "qualification_changes",
        "credit_changes",
        "canonical_changes",
        "a7_unlocks",
    }
    _expect_exact_keys(truth, expected_truth_keys, "validate-only truth")
    if any(isinstance(value, bool) or value != 0 for value in truth.values()):
        raise ContractError("validate-only truth contains a nonzero counter")


def load_candidate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_json_object(path)
    validate_candidate_config(config)
    return config


def _require_synthetic_scope(scope_token: str, config: Mapping[str, Any]) -> None:
    required = config["synthetic_scope_contract"]["required_scope_token"]
    if scope_token != required or required != SYNTHETIC_SCOPE:
        raise ContractError("only the explicit synthetic-test-fixture scope is allowed")


def _structural_components(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[tuple[str, ...]], dict[str, Mapping[str, Any]], int]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ContractError("synthetic structural records must be a sequence")
    required_keys = set(
        config["structural_graph_contract"]["required_record_keys_exactly"]
    )
    synthetic_prefix = config["synthetic_scope_contract"]["synthetic_identifier_prefix"]
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ContractError(f"synthetic structural record {index} is not an object")
        _expect_exact_keys(record, required_keys, f"synthetic structural record {index}")
        record_key = record["record_key"]
        source_group = record["source_group"]
        duplicate_keys = record["known_duplicate_record_keys"]
        if not isinstance(record_key, str) or not record_key:
            raise ContractError("synthetic record key must be a nonempty string")
        if not record_key.startswith(synthetic_prefix):
            raise ContractError("synthetic record key lacks the frozen synthetic prefix")
        if record_key in by_key:
            raise ContractError("duplicate synthetic record key")
        if not isinstance(source_group, str) or not source_group:
            raise ContractError("synthetic source group must be a nonempty string")
        if not source_group.startswith(synthetic_prefix):
            raise ContractError("synthetic source group lacks the frozen synthetic prefix")
        if type(duplicate_keys) is not list or any(
            not isinstance(item, str) or not item for item in duplicate_keys
        ):
            raise ContractError("known duplicate references must be a string list")
        if len(duplicate_keys) != len(set(duplicate_keys)):
            raise ContractError("known duplicate references contain duplicates")
        if record_key in duplicate_keys:
            raise ContractError("a record cannot be its own known duplicate")
        by_key[record_key] = record
    if not by_key:
        raise ContractError("at least one synthetic structural record is required")
    for record in by_key.values():
        for duplicate_key in record["known_duplicate_record_keys"]:
            if duplicate_key not in by_key:
                raise ContractError("known duplicate reference is not present")

    parent = {key: key for key in by_key}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    first_by_source: dict[str, str] = {}
    for key, record in by_key.items():
        source = record["source_group"]
        if source in first_by_source:
            union(key, first_by_source[source])
        else:
            first_by_source[source] = key

    duplicate_edges: set[tuple[str, str]] = set()
    for key, record in by_key.items():
        for duplicate_key in record["known_duplicate_record_keys"]:
            edge = tuple(sorted((key, duplicate_key)))
            duplicate_edges.add(edge)
            union(key, duplicate_key)

    groups: dict[str, list[str]] = defaultdict(list)
    for key in by_key:
        groups[find(key)].append(key)
    components = [tuple(sorted(items)) for items in groups.values()]
    components.sort()
    return components, by_key, len(duplicate_edges)


def build_structural_graph_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    scope_token: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only aggregate geometry for a synthetic structural graph."""

    _require_synthetic_scope(scope_token, config)
    components, by_key, duplicate_edge_count = _structural_components(records, config)
    histogram = Counter(len(component) for component in components)
    return {
        "status": SYNTHETIC_PASS,
        "record_count": len(by_key),
        "source_group_count": len({record["source_group"] for record in by_key.values()}),
        "known_duplicate_edge_count": duplicate_edge_count,
        "connected_component_count": len(components),
        "component_size_histogram": {
            str(size): count for size, count in sorted(histogram.items())
        },
        "outcome_fields_read": 0,
        "record_or_component_keys_included": False,
        "split_assignments_included": False,
        "scientific_gate_status": "NOT_RUN",
    }


def generate_outcome_blind_split_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    scope_token: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate a synthetic component split and return aggregate counts only."""

    _require_synthetic_scope(scope_token, config)
    components, by_key, duplicate_edge_count = _structural_components(records, config)
    split = config["split_plan_contract"]
    fold_ids = split["fold_ids"]
    if split["all_folds_must_be_nonempty"] and len(components) < len(fold_ids):
        raise ContractError("fewer connected components than required nonempty folds")

    salt = split["synthetic_validation_salt"]

    def salted_component_order(component: tuple[str, ...]) -> str:
        # This digest materially determines the synthetic assignment order; it
        # is not a provenance checksum and is never emitted.
        material = f"{salt}\0" + "\0".join(component)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    ordered = sorted(
        components,
        key=lambda component: (-len(component), salted_component_order(component)),
    )
    fold_state = [
        {"fold_id": fold_id, "record_count": 0, "component_count": 0}
        for fold_id in fold_ids
    ]
    record_to_fold: dict[str, int] = {}
    for component in ordered:
        fold_index = min(
            range(len(fold_state)),
            key=lambda index: (
                fold_state[index]["record_count"],
                fold_state[index]["component_count"],
                index,
            ),
        )
        fold_state[fold_index]["record_count"] += len(component)
        fold_state[fold_index]["component_count"] += 1
        for record_key in component:
            record_to_fold[record_key] = fold_index

    source_folds: dict[str, set[int]] = defaultdict(set)
    duplicate_cross_fold_count = 0
    duplicate_edges: set[tuple[str, str]] = set()
    for record_key, record in by_key.items():
        source_folds[record["source_group"]].add(record_to_fold[record_key])
        for duplicate_key in record["known_duplicate_record_keys"]:
            edge = tuple(sorted((record_key, duplicate_key)))
            if edge not in duplicate_edges:
                duplicate_edges.add(edge)
                if record_to_fold[record_key] != record_to_fold[duplicate_key]:
                    duplicate_cross_fold_count += 1
    source_cross_fold_count = sum(len(folds) > 1 for folds in source_folds.values())
    component_cross_fold_count = sum(
        len({record_to_fold[record_key] for record_key in component}) > 1
        for component in components
    )
    if (
        source_cross_fold_count != split["source_group_cross_fold_count_max"]
        or duplicate_cross_fold_count
        != split["known_duplicate_cross_fold_count_max"]
        or component_cross_fold_count != split["component_cross_fold_count_max"]
    ):
        raise ContractError("synthetic split violated a zero-leakage boundary")
    if split["all_folds_must_be_nonempty"] and any(
        item["component_count"] == 0 for item in fold_state
    ):
        raise ContractError("synthetic split produced an empty required fold")

    histogram = Counter(len(component) for component in components)
    return {
        "schema_version": "route_a_v3_a2_g0_aggregate_split_plan.v1",
        "status": SYNTHETIC_PASS,
        "plan_id": split["plan_id"],
        "assignment_unit": split["assignment_unit"],
        "outcome_blind": True,
        "record_count": len(by_key),
        "source_group_count": len(source_folds),
        "known_duplicate_edge_count": duplicate_edge_count,
        "connected_component_count": len(components),
        "component_size_histogram": {
            str(size): count for size, count in sorted(histogram.items())
        },
        "fold_aggregate_counts": fold_state,
        "source_group_cross_fold_count": source_cross_fold_count,
        "known_duplicate_cross_fold_count": duplicate_cross_fold_count,
        "component_cross_fold_count": component_cross_fold_count,
        "outcome_fields_read": 0,
        "record_or_component_keys_included": False,
        "split_assignments_included": False,
        "synthetic_salt_is_final_a2_salt": False,
        "final_a2_membership_status": "NOT_FROZEN",
        "scientific_gate_status": "NOT_RUN",
    }


def validate_endpoint_effect_se_manifest(
    manifest: Mapping[str, Any],
    *,
    scope_token: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate synthetic endpoint/effect/SE metadata without reading values."""

    _require_synthetic_scope(scope_token, config)
    contract = config["endpoint_effect_se_contract"]
    required_keys = set(contract["required_manifest_keys_exactly"])
    _expect_exact_keys(manifest, required_keys, "endpoint/effect/SE manifest")
    if manifest["schema_version"] != contract["manifest_schema_version"]:
        raise ContractError("endpoint/effect/SE manifest schema differs")
    if manifest["input_scope"] != SYNTHETIC_SCOPE:
        raise ContractError("endpoint/effect/SE manifest is not synthetic-only")
    for key in ("endpoint_name", "endpoint_scale"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ContractError(f"{key} must be a nonempty string")
        if not manifest[key].startswith(
            config["synthetic_scope_contract"]["synthetic_identifier_prefix"]
        ):
            raise ContractError(f"{key} lacks the frozen synthetic prefix")
    if manifest["endpoint_transform"] not in contract["allowed_endpoint_transforms"]:
        raise ContractError("endpoint transform is not frozen as allowed")
    direction_map = contract["direction_multiplier_by_endpoint_direction"]
    direction = manifest["endpoint_direction"]
    if direction not in direction_map:
        raise ContractError("endpoint direction is not declared")
    multiplier = manifest["direction_multiplier"]
    if isinstance(multiplier, bool) or multiplier != direction_map[direction]:
        raise ContractError("endpoint direction multiplier is inconsistent")
    if manifest["effect_definition"] != contract["required_effect_definition"]:
        raise ContractError("effect definition differs")
    if manifest["effect_analysis_unit"] != contract["required_effect_analysis_unit"]:
        raise ContractError("effect analysis unit differs")
    if manifest["standard_error_estimator"] not in contract["allowed_standard_error_estimators"]:
        raise ContractError("standard-error estimator is not allowed")
    if (
        manifest["standard_error_analysis_unit"]
        != contract["required_standard_error_analysis_unit"]
    ):
        raise ContractError("standard-error analysis unit differs")
    if manifest["independent_replicate_unit"] != contract["required_independent_replicate_unit"]:
        raise ContractError("replicate unit is not biological")
    minimum = manifest["minimum_independent_biological_replicates"]
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise ContractError("minimum biological replicate count must be an integer")
    if minimum < contract["minimum_independent_biological_replicates"]:
        raise ContractError("fewer than three independent biological replicates")
    if manifest["technical_replicates_may_count_as_biological"] is not False:
        raise ContractError("technical replicates may not count as biological")
    if manifest["missing_policy"] != contract["required_missing_policy"]:
        raise ContractError("missing-value policy differs")
    if manifest["nonfinite_policy"] != contract["required_nonfinite_policy"]:
        raise ContractError("nonfinite-value policy differs")
    if manifest["censoring_or_selection_rule_prefrozen"] is not True:
        raise ContractError("censoring/selection rule is not prefrozen")
    return {
        "status": SYNTHETIC_PASS,
        "endpoint_direction_and_transform_closed": True,
        "effect_definition_closed": True,
        "biological_standard_error_contract_closed": True,
        "missing_nonfinite_and_censoring_policy_closed": True,
        "endpoint_or_effect_values_read": 0,
        "member_identifiers_included": False,
        "scientific_gate_status": "NOT_RUN",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_left)
        * math.fsum(value * value for value in centered_right)
    )
    if denominator == 0.0:
        raise MetricUndefinedError("Spearman is undefined for a constant rank input")
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(centered_left, centered_right)
    ) / denominator


def evaluate_synthetic_effects(
    rows: Sequence[Mapping[str, Any]],
    endpoint_manifest: Mapping[str, Any],
    *,
    scope_token: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise the frozen evaluator on synthetic values and emit aggregates."""

    _require_synthetic_scope(scope_token, config)
    validate_endpoint_effect_se_manifest(
        endpoint_manifest, scope_token=scope_token, config=config
    )
    metric = config["evaluator_metric_schema"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ContractError("synthetic evaluator rows must be a sequence")
    required_keys = {
        metric["required_source_group_field"],
        metric["required_prediction_field"],
        metric["required_observation_field"],
        metric["required_standard_error_field"],
    }
    if len(rows) < metric["minimum_finite_independent_units"]:
        raise ContractError("too few finite independent synthetic source groups")

    seen_groups: set[str] = set()
    synthetic_prefix = config["synthetic_scope_contract"]["synthetic_identifier_prefix"]
    predicted: list[float] = []
    observed: list[float] = []
    standard_errors: list[float] = []
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise ContractError(f"synthetic evaluator row {index} is not an object")
        _expect_exact_keys(row, required_keys, f"synthetic evaluator row {index}")
        group = row[metric["required_source_group_field"]]
        if not isinstance(group, str) or not group:
            raise ContractError("synthetic source-group key must be a nonempty string")
        if not group.startswith(synthetic_prefix):
            raise ContractError("synthetic source-group key lacks the frozen synthetic prefix")
        if group in seen_groups:
            raise ContractError("synthetic evaluator source group is duplicated")
        seen_groups.add(group)
        values = [
            row[metric["required_prediction_field"]],
            row[metric["required_observation_field"]],
            row[metric["required_standard_error_field"]],
        ]
        if any(not _is_plain_number(value) or not math.isfinite(float(value)) for value in values):
            raise ContractError("synthetic evaluator value is missing or nonfinite")
        if float(values[2]) <= 0.0:
            raise ContractError("synthetic biological standard error is not strictly positive")
        predicted.append(float(values[0]))
        observed.append(float(values[1]))
        standard_errors.append(float(values[2]))

    rho = _pearson(_average_ranks(predicted), _average_ranks(observed))
    mean_absolute_error = math.fsum(
        abs(predicted_value - observed_value)
        for predicted_value, observed_value in zip(predicted, observed)
    ) / len(rows)
    return {
        "schema_version": metric["schema_version"],
        "status": SYNTHETIC_PASS,
        "analysis_unit": metric["analysis_unit"],
        "independent_source_group_count": len(rows),
        "primary_metric": metric["primary_metric"],
        "within_study_spearman": rho,
        "secondary_diagnostic": metric["secondary_diagnostic"],
        "mean_absolute_effect_error": mean_absolute_error,
        "mean_observed_standard_error": math.fsum(standard_errors) / len(rows),
        "one_source_group_one_vote": True,
        "member_or_source_group_keys_included": False,
        "row_level_effects_or_standard_errors_included": False,
        "model_or_checkpoint_selection_allowed": False,
        "formal_qualification_or_power_gate_executed": False,
        "final_a2_membership_status": "NOT_FROZEN",
        "scientific_gate_status": "NOT_RUN",
    }


def build_validate_only_report(config: Mapping[str, Any]) -> dict[str, Any]:
    power = config["power_precision_contract"]
    required_n = power["required_effective_n_for_both_power_and_ci_width"]
    required_plan = bonett_wright_fisher_z_plan(
        required_n,
        alternative_spearman_rho=power["alternative_spearman_rho"],
        two_sided_alpha=power["two_sided_alpha"],
        confidence_level=power["confidence_level"],
        target_power=power["target_power"],
        maximum_full_ci_width=power["maximum_full_confidence_interval_width"],
    )
    prior_plan = bonett_wright_fisher_z_plan(
        required_n - 1,
        alternative_spearman_rho=power["alternative_spearman_rho"],
        two_sided_alpha=power["two_sided_alpha"],
        confidence_level=power["confidence_level"],
        target_power=power["target_power"],
        maximum_full_ci_width=power["maximum_full_confidence_interval_width"],
    )
    if not required_plan["both_thresholds_pass"] or prior_plan["both_thresholds_pass"]:
        raise ContractError("required N=156 boundary is not reproduced")
    return {
        "status": CONFIG_PASS,
        "mode": "VALIDATE_ONLY_ZERO_PROJECT_ROW_IO_NO_RUNTIME_ARTIFACTS",
        "candidate_id": CANDIDATE_ID,
        "document_status": DOCUMENT_STATUS,
        "authority_status": "NON_AUTHORITATIVE",
        "activation_state": "INACTIVE_G0_IMPLEMENTATION_CANDIDATE",
        "config_validated": True,
        "power_precision_planning": {
            "required_effective_n": required_n,
            "analysis_unit": power["analysis_unit"],
            "n_155_both_thresholds_pass": prior_plan["both_thresholds_pass"],
            "n_156_estimated_design_power": required_plan["estimated_design_power"],
            "n_156_planned_full_confidence_interval_width": required_plan[
                "planned_full_confidence_interval_width"
            ],
            "n_156_both_thresholds_pass": required_plan["both_thresholds_pass"],
            "formal_gate_executed": False,
        },
        "validate_only_truth": dict(config["validate_only_truth"]),
        "final_a2_membership_status": "NOT_FROZEN",
        "final_split_assignment_status": "NOT_FROZEN",
        "current_qualified_counts": dict(
            config["governance_boundary"]["current_qualified_counts"]
        ),
        "scientific_gate_status": "NOT_RUN",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the fixed draft config and print a zero-I/O aggregate report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.validate_only:
        raise ContractError("this inactive G0 candidate supports --validate-only only")
    config = load_candidate_config()
    print(json.dumps(build_validate_only_report(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
