#!/usr/bin/env python3
"""Build the V3.3.2 Prediction/Generation matched-budget reporting matrix.

The matrix separates exact within-screen matching from broader frozen hurdles
and from prospective generation comparisons that were closed by Critic V2
NO-GO.  Missing compute fields remain blank with an explicit status; this
builder never reads Development TEST, final Evaluation outcomes, or generated
candidate payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    ROOT
    / "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json"
)
DEFAULT_CRITIC_AUDIT = (
    ROOT / "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json"
)
DEFAULT_CRITIC_PROTOCOL = (
    ROOT / "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
)
DEFAULT_GENERATION_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_baseline_table_v1.csv"
)
DEFAULT_GEOMETRY_AUDIT = (
    ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
)
DEFAULT_BASELINE_MATRIX = ROOT / "docs/paper/route2_v332_baseline_matrix_v1.csv"
DEFAULT_FLOW_CONFIG = (
    ROOT / "configs/route_a_v3_route2_base_flow_g0_position_progress_gpu_v2.json"
)
DEFAULT_OUTPUT_TABLE = (
    ROOT / "docs/paper/route2_v332_matched_budget_baseline_matrix_v1.csv"
)
DEFAULT_OUTPUT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_matched_budget_baseline_matrix_v1.json"
)

PREDICTION_ARM_ORDER = (
    "full",
    "candidate_permutation",
    "source_only",
    "source_edit_metadata",
)
GENERATION_METHOD_ORDER = (
    "random_legal",
    "greedy",
    "beam",
    "genetic",
    "local_search",
    "generate_then_rerank",
    "unguided_learned_base_flow_g0",
)
GUIDED_METHOD_ORDER = (
    "first_order_rate_guidance",
    "frozen_critic_xeditflow",
)

FIELDS = (
    "matrix_row_id",
    "track",
    "comparison_group",
    "method_id",
    "scientific_role",
    "task_scope",
    "result_status",
    "primary_metric_name",
    "primary_metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "train_record_count",
    "validation_record_count",
    "source_count",
    "training_seed",
    "physical_gpu_index",
    "gpu_cohort_status",
    "trainable_parameter_count",
    "frozen_parameter_count",
    "total_effective_parameter_count",
    "training_epochs",
    "optimizer_steps",
    "selected_epoch",
    "action_space",
    "edit_budget_set",
    "candidate_count",
    "candidate_cap_per_source",
    "critic_forward_cap_per_source",
    "total_forward_equivalent_cap_per_source",
    "observed_mean_total_forward_equivalents_per_source",
    "peak_vram_mb",
    "wall_time_seconds",
    "wall_time_status",
    "independent_scoring_wall_time_seconds",
    "source_pool_match_status",
    "split_match_status",
    "input_information_match_status",
    "training_or_hpo_budget_match_status",
    "generation_budget_match_status",
    "headline_comparison_eligible",
    "numeric_result_available",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_required",
    "guided_executed",
    "claim_boundary",
    "evidence_locator",
)


class MatchedBudgetInputError(RuntimeError):
    """Frozen matched-budget inputs do not support the declared reporting row."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedBudgetInputError(message)


def _false() -> str:
    return "false"


def _true() -> str:
    return "true"


def _blank_row() -> dict[str, Any]:
    return {field: "" for field in FIELDS}


def _prediction_rows(
    snapshot: Mapping[str, Any],
    critic_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    screen = snapshot["critic_v2_screen"]
    audit_screen = critic_audit["control_screen"]
    _require(
        critic_audit["scientific_claim_status"] == "NOT_ESTABLISHED",
        "Critic V2 scientific claim status changed",
    )
    _require(
        audit_screen["status"]
        == "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS",
        "Critic V2 terminal NO-GO changed",
    )
    _require(screen["screen_seed"] == protocol["screen_seed"] == 20260825,
             "Critic V2 screen seed changed")
    _require(screen["shared_epochs"] == protocol["frozen_training_policy"]["epochs"] == 100,
             "Critic V2 epoch budget changed")
    _require(screen["shared_optimizer_steps"] == 559900,
             "Critic V2 optimizer-step budget changed")
    _require(screen["shared_train_record_count"] == 89580,
             "Critic V2 TRAIN count changed")
    _require(screen["shared_validation_record_count"] == 18293,
             "Critic V2 Validation count changed")
    _require(screen["budget_match_status"] == "EXACT_MATCH_WITHIN_CRITIC_V2_SCREEN",
             "Critic V2 exact-match status changed")

    input_information = {
        "full": "SOURCE_AND_CANDIDATE_SEQUENCE_EDIT_CONTEXT",
        "candidate_permutation": "SOURCE_PLUS_WITHIN_SOURCE_TASK_PERMUTED_TRAIN_CANDIDATE",
        "source_only": "SOURCE_ONLY_PARAMETER_MATCHED_NO_CANDIDATE",
        "source_edit_metadata": "SOURCE_PLUS_EDIT_METADATA_NO_CANDIDATE_SEQUENCE_PARAMETER_MATCHED",
    }
    rows: list[dict[str, Any]] = []
    for index, arm_id in enumerate(PREDICTION_ARM_ORDER, start=1):
        arm = screen["arms"][arm_id]
        audit_arm = audit_screen["arms"][arm_id]
        protocol_arm = protocol["arms"][arm_id]
        _require(arm["status"] == audit_arm["ledger_status"] == "COMPLETED",
                 f"Critic V2 {arm_id} is not terminal complete")
        _require(arm["scientific_role"] == protocol_arm["scientific_role"],
                 f"Critic V2 {arm_id} scientific role changed")
        _require(arm["model_kind"] == protocol_arm["model_kind"],
                 f"Critic V2 {arm_id} model kind changed")
        for metric in ("task_macro_spearman", "task_macro_standardized_mae"):
            _require(arm[metric] == audit_arm[metric],
                     f"Critic V2 {arm_id} {metric} changed")
        _require(arm["physical_gpu_index"] == audit_arm["physical_gpu_index"],
                 f"Critic V2 {arm_id} GPU changed")
        _require(arm["selected_epoch"] == audit_arm["selected_epoch"],
                 f"Critic V2 {arm_id} selected epoch changed")
        _require(audit_arm["optimizer_steps"] == screen["shared_optimizer_steps"],
                 f"Critic V2 {arm_id} update budget is not matched")

        row = _blank_row()
        row.update(
            {
                "matrix_row_id": f"P-MB-{index:02d}",
                "track": "PREDICTION",
                "comparison_group": "CRITIC_V2_SAME_BUDGET_CONTROL_SCREEN",
                "method_id": f"critic_v2_{arm_id}",
                "scientific_role": arm["scientific_role"],
                "task_scope": "NINE_TASK_TRANSFERABLE_CONTEXT",
                "result_status": "EXECUTED_TERMINAL_CRITIC_V2_CONTROL",
                "primary_metric_name": "task_macro_spearman",
                "primary_metric_value": arm["task_macro_spearman"],
                "secondary_metric_name": "task_macro_standardized_mae",
                "secondary_metric_value": arm["task_macro_standardized_mae"],
                "train_record_count": screen["shared_train_record_count"],
                "validation_record_count": screen["shared_validation_record_count"],
                "training_seed": screen["screen_seed"],
                "physical_gpu_index": arm["physical_gpu_index"],
                "gpu_cohort_status": "A100_80GB_PHYSICAL_GPU_RECORDED_WITHIN_GPU0_5",
                "trainable_parameter_count": screen["shared_trainable_parameter_count"],
                "frozen_parameter_count": screen["shared_frozen_parameter_count"],
                "total_effective_parameter_count": screen[
                    "shared_total_effective_parameter_count"
                ],
                "training_epochs": screen["shared_epochs"],
                "optimizer_steps": screen["shared_optimizer_steps"],
                "selected_epoch": arm["selected_epoch"],
                "peak_vram_mb": arm["peak_vram_mb"],
                "wall_time_seconds": arm["wall_time_seconds"],
                "wall_time_status": "RECORDED_IN_CENTRAL_TERMINAL_LEDGER",
                "source_pool_match_status": "COMMON_FROZEN_DEVELOPMENT_MANIFEST",
                "split_match_status": "EXACT_FIXED_GROUPED_TRAIN_VALIDATION_SPLIT",
                "input_information_match_status": input_information[arm_id],
                "training_or_hpo_budget_match_status": screen["budget_match_status"],
                "generation_budget_match_status": "NOT_APPLICABLE_PREDICTION",
                "headline_comparison_eligible": _true() if arm_id == "full" else _false(),
                "numeric_result_available": _true(),
                "development_test_read": _false(),
                "new_final_evaluation_read": _false(),
                "guided_required": _false(),
                "guided_executed": _false(),
                "claim_boundary": (
                    "Exact-matched Critic V2 Development control-screen arm; only the full arm "
                    "enters the frozen hurdle comparison, and no row is external confirmation."
                ),
                "evidence_locator": (
                    "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json;"
                    "audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json"
                ),
            }
        )
        rows.append(row)

    hurdle = snapshot["strongest_same_information_hurdle"]
    audit_hurdle = audit_screen["strongest_same_information_baseline"]
    protocol_hurdle = protocol["strongest_same_information_baseline"]
    _require(hurdle["baseline_id"] == audit_hurdle["baseline_id"] == protocol_hurdle["baseline_id"],
             "strongest same-information baseline identity changed")
    for metric in ("task_macro_spearman", "task_macro_standardized_mae"):
        _require(hurdle[metric] == audit_hurdle[metric] == protocol_hurdle[metric],
                 f"strongest same-information {metric} changed")
    _require(
        hurdle["budget_match_status"]
        == "SAME_INFORMATION_HURDLE_NOT_UPDATE_BUDGET_MATCHED_TO_CRITIC_V2",
        "strongest hurdle compute-match boundary changed",
    )
    _require(hurdle["optimizer_steps"] < screen["shared_optimizer_steps"],
             "strongest hurdle is no longer lower-update than Critic V2")

    row = _blank_row()
    row.update(
        {
            "matrix_row_id": "P-MB-05",
            "track": "PREDICTION",
            "comparison_group": "FROZEN_STRONGEST_SAME_INFORMATION_HURDLE",
            "method_id": hurdle["baseline_id"],
            "scientific_role": "FROZEN_STRONGEST_SAME_INFORMATION_BASELINE",
            "task_scope": "NINE_TASK_TRANSFERABLE_CONTEXT",
            "result_status": "EXECUTED_FROZEN_DEVELOPMENT_HURDLE",
            "primary_metric_name": "task_macro_spearman",
            "primary_metric_value": hurdle["task_macro_spearman"],
            "secondary_metric_name": "task_macro_standardized_mae",
            "secondary_metric_value": hurdle["task_macro_standardized_mae"],
            "validation_record_count": hurdle["validation_record_count"],
            "training_seed": hurdle["seed"],
            "physical_gpu_index": hurdle["physical_gpu_index"],
            "gpu_cohort_status": "A100_80GB_PHYSICAL_GPU_RECORDED_WITHIN_GPU0_5",
            "optimizer_steps": hurdle["optimizer_steps"],
            "selected_epoch": hurdle["selected_epoch"],
            "peak_vram_mb": hurdle["peak_vram_mb"],
            "wall_time_seconds": hurdle["wall_time_seconds"],
            "wall_time_status": "RECORDED_IN_FROZEN_TERMINAL_SUMMARY",
            "source_pool_match_status": "FROZEN_SAME_INFORMATION_COMMON_VALIDATION_SCOPE",
            "split_match_status": "FROZEN_SAME_INFORMATION_HURDLE_SPLIT",
            "input_information_match_status": hurdle["information_match_status"],
            "training_or_hpo_budget_match_status": hurdle["budget_match_status"],
            "generation_budget_match_status": "NOT_APPLICABLE_PREDICTION",
            "headline_comparison_eligible": _true(),
            "numeric_result_available": _true(),
            "development_test_read": _false(),
            "new_final_evaluation_read": _false(),
            "guided_required": _false(),
            "guided_executed": _false(),
            "claim_boundary": (
                "Frozen same-information Development hurdle, but not exact update-budget matched: "
                "22120 versus 559900 optimizer steps; parameter counts and training epochs are not "
                "co-located in the frozen summary snapshot."
            ),
            "evidence_locator": (
                "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json;"
                "configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
            ),
        }
    )
    rows.append(row)
    return rows


def _generation_rows(
    generation_table: Sequence[Mapping[str, str]],
    geometry: Mapping[str, Any],
    baseline_matrix: Sequence[Mapping[str, str]],
    flow_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_method = {row["method_id"]: row for row in generation_table}
    _require(set(by_method) == set(GENERATION_METHOD_ORDER),
             "generation table no longer has the exact seven terminal methods")
    baseline_by_method = {
        row["implementation_id"]: row
        for row in baseline_matrix
        if row["track"] == "GENERATION"
    }
    boundary = geometry["protocol_boundary"]
    _require(boundary["source_count"] == 891, "generation source cohort changed")
    _require(boundary["candidate_cap_per_source"] == 32,
             "generation candidate cap changed")
    _require(boundary["critic_forward_budget_per_source"] == 256,
             "generation critic-forward cap changed")
    _require(boundary["forward_equivalent_budget_per_source"] == 320,
             "generation total-forward cap changed")
    _require(boundary["action_types_in_scope"] == ["SUB", "STOP"],
             "generation legal action set changed")
    _require(flow_config["allowed_edit_budgets"] == [1, 3, 5],
             "frozen source edit-budget set changed")
    _require(geometry["cross_method_geometry"]["all_method_hard_legality_rate"] == 1.0,
             "generation hard-legality result changed")

    rows: list[dict[str, Any]] = []
    for index, method_id in enumerate(GENERATION_METHOD_ORDER, start=1):
        source = by_method[method_id]
        inventory = baseline_by_method[method_id]
        _require(inventory["execution_status_v332"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT",
                 f"generation {method_id} terminal inventory status changed")
        _require(int(source["source_count"]) == boundary["source_count"],
                 f"generation {method_id} source count changed")
        _require(int(source["candidate_cap_per_source"]) == boundary["candidate_cap_per_source"],
                 f"generation {method_id} candidate cap changed")
        _require(float(source["hard_legality_rate"]) == 1.0,
                 f"generation {method_id} hard legality changed")
        _require(int(source["edit_budget_violation_count"]) == 0,
                 f"generation {method_id} edit budget violation appeared")
        _require(int(source["candidate_budget_violation_count"]) == 0,
                 f"generation {method_id} candidate budget violation appeared")

        wall_raw = source["generation_wall_time_seconds"]
        if wall_raw == "NOT_RECORDED":
            wall_time: Any = ""
            wall_status = "NOT_RECORDED_NO_TERMINAL_RERUN"
        else:
            wall_time = float(wall_raw)
            wall_status = "RECORDED_TERMINAL_GENERATION"
        generation_budget_status = (
            "MATCHED_SOURCE_ACTION_EDIT_CANDIDATE_AND_FORWARD_CAPS_WITH_NO_CRITIC_CALLS"
            if method_id == "unguided_learned_base_flow_g0"
            else "MATCHED_SOURCE_ACTION_EDIT_CANDIDATE_CRITIC_AND_TOTAL_FORWARD_CAPS"
        )

        row = _blank_row()
        row.update(
            {
                "matrix_row_id": f"G-MB-{index:02d}",
                "track": "GENERATION",
                "comparison_group": "GENERATION_MATCHED_DEVELOPMENT_SUB_STOP",
                "method_id": method_id,
                "scientific_role": inventory["contract_requirement"],
                "task_scope": "891_SOURCE_OPEN_GENERATED_SUPPORT",
                "result_status": inventory["execution_status_v332"],
                "primary_metric_name": "source_macro_independent_evaluator_max_uplift_over_source",
                "primary_metric_value": float(
                    source["source_macro_independent_evaluator_max_uplift_over_source"]
                ),
                "secondary_metric_name": "source_macro_candidate_recovery_rate",
                "secondary_metric_value": float(source["source_macro_candidate_recovery_rate"]),
                "source_count": int(source["source_count"]),
                "gpu_cohort_status": (
                    "A100_80GB_TERMINAL_COHORT_PHYSICAL_INDEX_NOT_RETAINED_IN_RESULT_TABLE"
                ),
                "action_space": "SUB_PLUS_STOP",
                "edit_budget_set": "1|3|5",
                "candidate_count": int(source["candidate_count"]),
                "candidate_cap_per_source": int(source["candidate_cap_per_source"]),
                "critic_forward_cap_per_source": boundary[
                    "critic_forward_budget_per_source"
                ],
                "total_forward_equivalent_cap_per_source": boundary[
                    "forward_equivalent_budget_per_source"
                ],
                "observed_mean_total_forward_equivalents_per_source": float(
                    source["mean_total_forward_equivalents_per_source"]
                ),
                "peak_vram_mb": float(source["generation_peak_vram_mb"]),
                "wall_time_seconds": wall_time,
                "wall_time_status": wall_status,
                "independent_scoring_wall_time_seconds": float(
                    source["independent_scoring_wall_time_seconds"]
                ),
                "source_pool_match_status": "EXACT_FROZEN_891_SOURCE_COHORT",
                "split_match_status": "NOT_APPLICABLE_COMMON_GENERATION_COHORT",
                "input_information_match_status": (
                    "MATCHED_AVAILABLE_SOURCE_ACTION_AND_FROZEN_EVALUATOR"
                ),
                "training_or_hpo_budget_match_status": (
                    "ALGORITHM_SPECIFIC_TRAINING_OR_HPO_NOT_COMMONLY_PARAMETERIZED"
                ),
                "generation_budget_match_status": generation_budget_status,
                "headline_comparison_eligible": _false(),
                "numeric_result_available": _true(),
                "development_test_read": _false(),
                "new_final_evaluation_read": _false(),
                "guided_required": _false(),
                "guided_executed": _false(),
                "claim_boundary": (
                    "Terminal Development generation method under matched source/action/edit/candidate/"
                    "forward caps. Algorithm-specific training/HPO is not a common numeric budget; "
                    "independent-evaluator uplift and sparse measured recovery are not biological outcomes."
                ),
                "evidence_locator": (
                    "docs/paper/route2_v332_generation_baseline_table_v1.csv;"
                    "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
                ),
            }
        )
        rows.append(row)

    for offset, method_id in enumerate(GUIDED_METHOD_ORDER, start=8):
        inventory = baseline_by_method[method_id]
        _require(inventory["execution_status_v332"] == "NOT_RUN_CRITIC_V2_NO_GO",
                 f"guided {method_id} is no longer terminal NO-GO")
        _require(inventory["guided_required"] == "true" and inventory["guided_executed"] == "false",
                 f"guided {method_id} execution boundary changed")
        row = _blank_row()
        row.update(
            {
                "matrix_row_id": f"G-MB-{offset:02d}",
                "track": "GENERATION",
                "comparison_group": "GENERATION_GUIDANCE_CLOSED",
                "method_id": method_id,
                "scientific_role": inventory["contract_requirement"],
                "task_scope": "891_SOURCE_OPEN_GENERATED_SUPPORT_PROSPECTIVE",
                "result_status": inventory["execution_status_v332"],
                "source_count": boundary["source_count"],
                "gpu_cohort_status": "PROSPECTIVE_A100_GPU0_5_NOT_RUN",
                "action_space": "SUB_PLUS_STOP",
                "edit_budget_set": "1|3|5",
                "candidate_cap_per_source": boundary["candidate_cap_per_source"],
                "critic_forward_cap_per_source": boundary[
                    "critic_forward_budget_per_source"
                ],
                "total_forward_equivalent_cap_per_source": boundary[
                    "forward_equivalent_budget_per_source"
                ],
                "wall_time_status": "NOT_RUN_CRITIC_V2_NO_GO",
                "source_pool_match_status": "PROSPECTIVE_FROZEN_891_SOURCE_PROTOCOL_NOT_EXECUTED",
                "split_match_status": "NOT_APPLICABLE_PROSPECTIVE_GENERATION",
                "input_information_match_status": "PROSPECTIVE_FROZEN_CRITIC_FEEDBACK",
                "training_or_hpo_budget_match_status": "NOT_EXECUTED_CRITIC_V2_NO_GO",
                "generation_budget_match_status": "PROSPECTIVE_MATCHED_PROTOCOL_NOT_EXECUTED",
                "headline_comparison_eligible": _false(),
                "numeric_result_available": _false(),
                "development_test_read": _false(),
                "new_final_evaluation_read": _false(),
                "guided_required": _true(),
                "guided_executed": _false(),
                "claim_boundary": (
                    "Critic V2 terminal NO-GO prohibits this guided method for the current cohort; "
                    "prospective budget fields are protocol caps, not executed results."
                ),
                "evidence_locator": inventory["evidence_locator"],
            }
        )
        rows.append(row)
    return rows


def derive_rows_and_audit(
    snapshot: Mapping[str, Any],
    critic_audit: Mapping[str, Any],
    critic_protocol: Mapping[str, Any],
    generation_table: Sequence[Mapping[str, str]],
    geometry_audit: Mapping[str, Any],
    baseline_matrix: Sequence[Mapping[str, str]],
    flow_config: Mapping[str, Any],
    *,
    source_paths: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        snapshot["status"]
        == "FROZEN_TERMINAL_COMPUTE_FIELDS_CAPTURED_FOR_MATCHED_BUDGET_MATRIX",
        "matched-budget terminal snapshot status changed",
    )
    _require(
        snapshot["protected_outcomes"]
        == {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "matched-budget snapshot opened a protected outcome",
    )
    prediction = _prediction_rows(snapshot, critic_audit, critic_protocol)
    generation = _generation_rows(
        generation_table, geometry_audit, baseline_matrix, flow_config
    )
    rows = prediction + generation
    _require(len(rows) == 14, "matched-budget matrix row count changed")
    _require(len({row["matrix_row_id"] for row in rows}) == len(rows),
             "matched-budget matrix row IDs are not unique")

    exact_screen = [
        row
        for row in prediction
        if row["training_or_hpo_budget_match_status"]
        == "EXACT_MATCH_WITHIN_CRITIC_V2_SCREEN"
    ]
    terminal_generation = [
        row
        for row in generation
        if row["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    ]
    guided_closed = [
        row for row in generation if row["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"
    ]
    recorded_wall = [row for row in rows if row["wall_time_seconds"] != ""]
    audit = {
        "schema_version": "route_a_v3_route2_v332_matched_budget_baseline_matrix.v1",
        "status": "MATCHED_BUDGET_REPORTING_MATRIX_RENDERED_CONTRACT_COMPLETE_MATCH_NOT_ESTABLISHED",
        "row_count": len(rows),
        "track_counts": {"PREDICTION": len(prediction), "GENERATION": len(generation)},
        "numeric_result_row_count": sum(
            row["numeric_result_available"] == "true" for row in rows
        ),
        "prediction": {
            "critic_v2_exact_within_screen_budget_rows": len(exact_screen),
            "strongest_same_information_hurdle_rows": 1,
            "strongest_hurdle_update_budget_matched": False,
            "critic_v2_optimizer_steps": snapshot["critic_v2_screen"][
                "shared_optimizer_steps"
            ],
            "strongest_hurdle_optimizer_steps": snapshot[
                "strongest_same_information_hurdle"
            ]["optimizer_steps"],
            "full_minus_hurdle_task_macro_spearman": critic_audit["control_screen"][
                "full_over_strongest_baseline_task_macro_spearman"
            ],
        },
        "generation": {
            "terminal_matched_method_rows": len(terminal_generation),
            "guided_dependency_no_go_rows": len(guided_closed),
            "full_candidate_cap_method_count": 6,
            "local_search_candidate_count": geometry_audit["cross_method_geometry"][
                "local_search_candidate_count"
            ],
            "search_method_wall_time_not_recorded_rows": sum(
                row["wall_time_status"] == "NOT_RECORDED_NO_TERMINAL_RERUN"
                for row in generation
            ),
            "terminal_generation_wall_time_recorded_rows": sum(
                row["wall_time_status"] == "RECORDED_TERMINAL_GENERATION"
                for row in generation
            ),
            "all_terminal_hard_legality_rate": geometry_audit[
                "cross_method_geometry"
            ]["all_method_hard_legality_rate"],
        },
        "compute_reporting": {
            "wall_time_recorded_row_count": len(recorded_wall),
            "peak_vram_recorded_row_count": sum(
                row["peak_vram_mb"] != "" for row in rows
            ),
            "fully_contract_matched_headline_comparison_row_count": 0,
            "reason": (
                "Prediction hurdle is not update-budget matched; generation training/HPO is "
                "algorithm-specific, six search wall times are not recorded, and both guided "
                "methods are closed by Critic V2 NO-GO."
            ),
        },
        "reporting_matrix_complete": True,
        "matched_budget_benchmark_execution_complete": False,
        "cross_track_numeric_ranking_allowed": False,
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "source_data": dict(source_paths),
        "new_training_attempt_created": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    return rows, audit


def build_matrix(
    *,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    critic_audit_path: Path = DEFAULT_CRITIC_AUDIT,
    critic_protocol_path: Path = DEFAULT_CRITIC_PROTOCOL,
    generation_table_path: Path = DEFAULT_GENERATION_TABLE,
    geometry_audit_path: Path = DEFAULT_GEOMETRY_AUDIT,
    baseline_matrix_path: Path = DEFAULT_BASELINE_MATRIX,
    flow_config_path: Path = DEFAULT_FLOW_CONFIG,
    output_table_path: Path = DEFAULT_OUTPUT_TABLE,
    output_audit_path: Path = DEFAULT_OUTPUT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    for output in (output_table_path, output_audit_path):
        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
    source_paths = {
        "terminal_snapshot": str(snapshot_path),
        "critic_terminal_audit": str(critic_audit_path),
        "critic_protocol": str(critic_protocol_path),
        "generation_table": str(generation_table_path),
        "generation_geometry_audit": str(geometry_audit_path),
        "baseline_inventory_matrix": str(baseline_matrix_path),
        "base_flow_config": str(flow_config_path),
    }
    rows, audit = derive_rows_and_audit(
        _read_json(snapshot_path),
        _read_json(critic_audit_path),
        _read_json(critic_protocol_path),
        _read_csv(generation_table_path),
        _read_json(geometry_audit_path),
        _read_csv(baseline_matrix_path),
        _read_json(flow_config_path),
        source_paths=source_paths,
    )
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit["table_path"] = str(output_table_path)
    output_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--critic-audit", type=Path, default=DEFAULT_CRITIC_AUDIT)
    parser.add_argument("--critic-protocol", type=Path, default=DEFAULT_CRITIC_PROTOCOL)
    parser.add_argument("--generation-table", type=Path, default=DEFAULT_GENERATION_TABLE)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--baseline-matrix", type=Path, default=DEFAULT_BASELINE_MATRIX)
    parser.add_argument("--flow-config", type=Path, default=DEFAULT_FLOW_CONFIG)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = build_matrix(
        snapshot_path=args.snapshot,
        critic_audit_path=args.critic_audit,
        critic_protocol_path=args.critic_protocol,
        generation_table_path=args.generation_table,
        geometry_audit_path=args.geometry_audit,
        baseline_matrix_path=args.baseline_matrix,
        flow_config_path=args.flow_config,
        output_table_path=args.output_table,
        output_audit_path=args.output_audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
