#!/usr/bin/env python3
"""Build the V3.3.2 A1/true-A2 task-results separation table.

The table reports terminal A1 Development Validation metrics task by task.  It
does not promote open generated-support recovery into true-A2 measured-candidate
ranking.  The true-A2 rows retain the materialized Development listwise data,
implemented evaluator, non-terminal listwise model, undefined closed measured
NDCG under open support, and absent independent Evaluation study as explicit
status boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A1_TASK_TABLE = ROOT / "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv"
DEFAULT_DATASET_TABLE = ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
DEFAULT_BASELINE_MATRIX = ROOT / "docs/paper/route2_v332_baseline_matrix_v1.csv"
DEFAULT_PACKAGE_TABLE = ROOT / "docs/paper/route2_v332_minimum_benchmark_package_table_v1.csv"
DEFAULT_GEOMETRY_AUDIT = ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
DEFAULT_FRESHNESS_AUDIT = ROOT / "audits/route_a_v3_route2_v332_freshness_and_critic_v2_freeze_v1.json"
DEFAULT_OUTPUT_TABLE = ROOT / "docs/paper/route2_v332_a1_true_a2_task_results_table_v1.csv"
DEFAULT_OUTPUT_AUDIT = ROOT / "audits/route_a_v3_route2_v332_a1_true_a2_task_results_table_v1.json"

TASK_IDS = (
    "MEAN_RIBOSOME_LOAD::region=0",
    "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1",
    "RNA_HALF_LIFE_MINUTES::region=0",
    "RNA_HALF_LIFE_MINUTES::region=1",
    "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",
    "te_log2_polysome_over_totalrna::region=0",
    "transcript_log2_totalrna_over_dna::region=0",
)

FIELDS = (
    "result_row_id",
    "estimand",
    "task_or_boundary_id",
    "region",
    "study_or_scope",
    "evidence_stage",
    "record_count",
    "primary_metric",
    "primary_value",
    "secondary_metric",
    "secondary_value",
    "tertiary_metric",
    "tertiary_value",
    "terminal_result_status",
    "numeric_performance_result_available",
    "qualified_true_a2_study_credit",
    "external_confirmation_eligible",
    "development_test_read",
    "new_final_evaluation_read",
    "guided_xeditflow_run",
    "claim_boundary",
    "evidence_locator",
)


class TaskResultsInputError(RuntimeError):
    """A frozen input changed or would blur the A1/true-A2 boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskResultsInputError(message)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(rows: Sequence[Mapping[str, str]], key: str, value: str, label: str) -> Mapping[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    _require(len(matches) == 1, f"{label} is not uniquely present")
    return matches[0]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _base_row() -> dict[str, str]:
    return {field: "" for field in FIELDS}


def derive_rows(
    *,
    a1_tasks: Sequence[Mapping[str, str]],
    dataset_rows: Sequence[Mapping[str, str]],
    baseline_rows: Sequence[Mapping[str, str]],
    package_rows: Sequence[Mapping[str, str]],
    geometry: Mapping[str, Any],
    freshness: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    _require(len(a1_tasks) == 9, "A1 evaluator task table must contain nine tasks")
    by_task = {row["task_id"]: row for row in a1_tasks}
    _require(set(by_task) == set(TASK_IDS), "A1 evaluator task set changed")
    _require(sum(int(row["record_count"]) for row in a1_tasks) == 18293,
             "A1 task record counts no longer sum to Development Validation")
    for row in a1_tasks:
        for field in ("spearman", "mae", "standardized_mae"):
            _require(math.isfinite(float(row[field])), f"A1 {row['task_id']} {field} is non-finite")

    evaluator = freshness.get("independent_evaluator", {})
    _require(evaluator.get("adjudication_status") == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
             "independent evaluator qualification changed")
    _require(evaluator.get("task_count") == 9, "independent evaluator task count changed")
    _require(evaluator.get("development_test_outcomes_accessed") is False,
             "independent evaluator accessed Development TEST")
    _require(evaluator.get("evaluation_outcomes_accessed") is False,
             "independent evaluator accessed Evaluation")

    gse269595 = _one(dataset_rows, "study_unit_id", "GSE269595", "GSE269595 dataset row")
    _require(gse269595["qualification_stratum"] == "DEVELOPMENT_LISTWISE_NOT_QUALIFIED",
             "GSE269595 qualification stratum changed")
    _require(gse269595["terminal_conversion_status"] == "MATERIALIZED_DEVELOPMENT_LISTWISE",
             "GSE269595 terminal conversion status changed")
    _require(int(gse269595["development_canonical_records"]) == 30966,
             "GSE269595 Development listwise record count changed")
    _require(int(gse269595["true_a2_study_credit"]) == 0,
             "GSE269595 now grants qualified true-A2 credit")
    _require(sum(int(row["final_evaluation_unexposed_canonical_records"]) for row in dataset_rows) == 0,
             "a new final Evaluation record set is now registered")

    listwise = _one(baseline_rows, "matrix_row_id", "P-NN-07", "listwise baseline row")
    _require(listwise["contract_requirement"] == "listwise ranker",
             "P-NN-07 is no longer the listwise ranker")
    _require(listwise["execution_status_v332"] == "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE",
             "listwise baseline terminal status changed")
    _require(listwise["headline_eligible_now"] == "false",
             "non-terminal listwise baseline became headline eligible")

    evaluator_implementation = _one(
        package_rows, "requirement_id", "MBP-05", "A1/true-A2 evaluator package row"
    )
    _require(evaluator_implementation["status"] == "COMPLETE",
             "A1/true-A2 evaluator implementation package status changed")
    _require("No new independent true-A2 Evaluation study exists" in evaluator_implementation["remaining_gap"],
             "true-A2 independent-study gap changed")

    protocol = geometry.get("protocol_boundary", {})
    cross_method = geometry.get("cross_method_geometry", {})
    _require(protocol.get("analysis_stage") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD_AND_FROZEN_INDEPENDENT_EVALUATOR",
             "generation analysis stage changed")
    _require(protocol.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT",
             "generation candidate support is no longer open")
    _require(protocol.get("unknown_generated_candidates_are_zero_gain") is False,
             "unknown generated candidates are now assigned zero gain")
    _require(protocol.get("generated_candidates_grant_canonical_credit") is False,
             "generated candidates now grant canonical credit")
    _require(protocol.get("development_test_outcomes_read") == 0,
             "generation geometry accessed Development TEST")
    _require(protocol.get("new_final_evaluation_outcomes_read") == 0,
             "generation geometry accessed new Evaluation")
    _require(cross_method.get("closed_measured_ndcg_defined_source_count_all_methods") == 0,
             "closed measured NDCG availability changed")

    rows: list[dict[str, str]] = []
    for index, task_id in enumerate(TASK_IDS, start=1):
        source = by_task[task_id]
        task_name, region_text = task_id.rsplit("::region=", maxsplit=1)
        row = _base_row()
        row.update({
            "result_row_id": f"A1-{index:02d}",
            "estimand": "A1_SOURCE_RELATIVE_NUMERIC_EFFECT",
            "task_or_boundary_id": task_name,
            "region": region_text,
            "study_or_scope": "DEVELOPMENT_VALIDATION_NINE_TASKS",
            "evidence_stage": "TERMINAL_DEVELOPMENT_VALIDATION",
            "record_count": source["record_count"],
            "primary_metric": "spearman",
            "primary_value": source["spearman"],
            "secondary_metric": "mae",
            "secondary_value": source["mae"],
            "tertiary_metric": "standardized_mae",
            "tertiary_value": source["standardized_mae"],
            "terminal_result_status": "TERMINAL_DEVELOPMENT_VALIDATION_NUMERIC",
            "numeric_performance_result_available": _bool(True),
            "external_confirmation_eligible": _bool(False),
            "development_test_read": _bool(False),
            "new_final_evaluation_read": _bool(False),
            "guided_xeditflow_run": _bool(False),
            "claim_boundary": "Qualified independent evaluator task metric for Development generation-method selection; not final external confirmation or biological validation.",
            "evidence_locator": "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv",
        })
        rows.append(row)

    boundary_rows = [
        {
            "result_row_id": "A2-01",
            "task_or_boundary_id": "GSE269595_DEVELOPMENT_LISTWISE_DATA",
            "study_or_scope": "GSE269595",
            "evidence_stage": "DEVELOPMENT_EXPOSED_MATERIALIZATION",
            "record_count": "30966",
            "primary_metric": "development_listwise_records",
            "primary_value": "30966",
            "terminal_result_status": "DATA_MATERIALIZED_PERFORMANCE_RESULT_NOT_TERMINAL",
            "qualified_true_a2_study_credit": "0",
            "claim_boundary": "Development listwise materialization is available but grants zero qualified true-A2 study credit and is not an unseen Evaluation study.",
            "evidence_locator": "docs/paper/route2_v332_dataset_qualification_table_v1.csv",
        },
        {
            "result_row_id": "A2-02",
            "task_or_boundary_id": "TRUE_A2_ESTIMAND_EVALUATOR",
            "study_or_scope": "ROUTE2_EVALUATION_IMPLEMENTATION",
            "evidence_stage": "IMPLEMENTATION_COMPLETE",
            "terminal_result_status": "EVALUATOR_IMPLEMENTED_NO_INDEPENDENT_TERMINAL_TRUE_A2_RESULT",
            "claim_boundary": "A1 and true-A2 evaluator code exists; implementation completeness is not a numeric true-A2 benchmark result.",
            "evidence_locator": "docs/paper/route2_v332_minimum_benchmark_package_table_v1.csv::MBP-05",
        },
        {
            "result_row_id": "A2-03",
            "task_or_boundary_id": "LISTWISE_RANKER",
            "study_or_scope": "COMMON_SOURCE_RELATIVE_TASK",
            "evidence_stage": "CONFIGURATION_ONLY",
            "terminal_result_status": "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE",
            "claim_boundary": "A runnable listwise loss configuration exists, but no independently terminal selected listwise baseline result is available.",
            "evidence_locator": listwise["evidence_locator"],
        },
        {
            "result_row_id": "A2-04",
            "task_or_boundary_id": "CLOSED_MEASURED_NDCG_UNDER_OPEN_GENERATED_SUPPORT",
            "study_or_scope": "SEVEN_METHOD_891_SOURCE_GENERATION_SUITE",
            "evidence_stage": "TERMINAL_DEVELOPMENT_GENERATION",
            "record_count": str(protocol["source_count"]),
            "primary_metric": "closed_measured_ndcg_defined_source_count_all_methods",
            "primary_value": "0",
            "terminal_result_status": "UNDEFINED_BY_OPEN_GENERATED_SUPPORT",
            "claim_boundary": "Sparse recovery of measured candidates is not closed-pool true-A2 ranking; unknown generated candidates are not assigned zero gain.",
            "evidence_locator": "audits/route_a_v3_route2_generation_action_space_geometry_v1.json",
        },
        {
            "result_row_id": "A2-05",
            "task_or_boundary_id": "NEW_INDEPENDENT_TRUE_A2_EVALUATION",
            "study_or_scope": "FINAL_EVALUATION",
            "evidence_stage": "NOT_AVAILABLE",
            "record_count": "0",
            "primary_metric": "final_evaluation_unexposed_canonical_records",
            "primary_value": "0",
            "terminal_result_status": "NOT_AVAILABLE_NO_OUTCOME_UNEXPOSED_MEASURED_NEIGHBORHOOD_STUDY",
            "qualified_true_a2_study_credit": "0",
            "claim_boundary": "No new convertible outcome-unexposed measured-neighborhood Evaluation study exists; no external true-A2 confirmation is claimed.",
            "evidence_locator": "docs/paper/route2_v332_dataset_qualification_table_v1.csv",
        },
    ]
    for values in boundary_rows:
        row = _base_row()
        row.update({
            "estimand": "TRUE_A2_WITHIN_SOURCE_MEASURED_CANDIDATE_RANKING",
            "numeric_performance_result_available": _bool(False),
            "external_confirmation_eligible": _bool(False),
            "development_test_read": _bool(False),
            "new_final_evaluation_read": _bool(False),
            "guided_xeditflow_run": _bool(False),
            **values,
        })
        rows.append(row)

    spearman_values = [float(row["spearman"]) for row in a1_tasks]
    audit = {
        "schema_version": "route_a_v3_route2_v332_a1_true_a2_task_results_table.v1",
        "status": "A1_TRUE_A2_TASK_RESULTS_SEPARATED_TRUE_A2_NUMERIC_NOT_TERMINAL",
        "row_count": len(rows),
        "a1_numeric_task_row_count": 9,
        "true_a2_boundary_row_count": 5,
        "true_a2_terminal_numeric_performance_row_count": 0,
        "a1": {
            "estimand": "SOURCE_RELATIVE_NUMERIC_EFFECT",
            "development_validation_record_count": 18293,
            "task_count": 9,
            "positive_task_spearman_count": sum(value > 0 for value in spearman_values),
            "task_spearman_min": min(spearman_values),
            "task_spearman_max": max(spearman_values),
            "role": "DEVELOPMENT_GENERATION_METHOD_SELECTION_NOT_FINAL_EXTERNAL_CONFIRMATION",
        },
        "true_a2": {
            "estimand": "WITHIN_SOURCE_MEASURED_CANDIDATE_RANKING",
            "development_listwise_study_unit_id": "GSE269595",
            "development_listwise_record_count": 30966,
            "qualified_true_a2_study_credit": 0,
            "evaluator_implementation_status": "COMPLETE",
            "listwise_baseline_status": "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE",
            "closed_measured_ndcg_defined_source_count_all_methods": 0,
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "new_independent_evaluation_unexposed_record_count": 0,
            "benchmark_execution_complete": False,
        },
        "reporting_table_complete": True,
        "cross_estimand_numeric_ranking_allowed": False,
        "open_support_recovery_substitutes_for_true_a2_ranking": False,
        "protected_outcomes": {
            "development_test_read": False,
            "new_final_evaluation_read": False,
            "guided_xeditflow_run": False,
        },
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    return rows, audit


def build_table(
    *,
    a1_task_table: Path = DEFAULT_A1_TASK_TABLE,
    dataset_table: Path = DEFAULT_DATASET_TABLE,
    baseline_matrix: Path = DEFAULT_BASELINE_MATRIX,
    package_table: Path = DEFAULT_PACKAGE_TABLE,
    geometry_audit: Path = DEFAULT_GEOMETRY_AUDIT,
    freshness_audit: Path = DEFAULT_FRESHNESS_AUDIT,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
    output_audit: Path = DEFAULT_OUTPUT_AUDIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = [a1_task_table, dataset_table, baseline_matrix, package_table,
             geometry_audit, freshness_audit, output_table, output_audit]
    (a1_task_table, dataset_table, baseline_matrix, package_table,
     geometry_audit, freshness_audit, output_table, output_audit) = [path.resolve() for path in paths]
    for path in (output_table, output_audit):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing A1/true-A2 task-results artifact: {path}")

    rows, audit = derive_rows(
        a1_tasks=_csv_rows(a1_task_table),
        dataset_rows=_csv_rows(dataset_table),
        baseline_rows=_csv_rows(baseline_matrix),
        package_rows=_csv_rows(package_table),
        geometry=_json(geometry_audit),
        freshness=_json(freshness_audit),
    )
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    persisted_audit = {
        **audit,
        "source_data": {
            "a1_task_table": str(a1_task_table),
            "dataset_table": str(dataset_table),
            "baseline_matrix": str(baseline_matrix),
            "minimum_package_table": str(package_table),
            "generation_geometry_audit": str(geometry_audit),
            "freshness_audit": str(freshness_audit),
        },
        "table_path": str(output_table),
        "new_training_attempt_created": False,
    }
    output_audit.write_text(json.dumps(persisted_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return persisted_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-task-table", type=Path, default=DEFAULT_A1_TASK_TABLE)
    parser.add_argument("--dataset-table", type=Path, default=DEFAULT_DATASET_TABLE)
    parser.add_argument("--baseline-matrix", type=Path, default=DEFAULT_BASELINE_MATRIX)
    parser.add_argument("--package-table", type=Path, default=DEFAULT_PACKAGE_TABLE)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--freshness-audit", type=Path, default=DEFAULT_FRESHNESS_AUDIT)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_table(
        a1_task_table=args.a1_task_table,
        dataset_table=args.dataset_table,
        baseline_matrix=args.baseline_matrix,
        package_table=args.package_table,
        geometry_audit=args.geometry_audit,
        freshness_audit=args.freshness_audit,
        output_table=args.output_table,
        output_audit=args.output_audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
