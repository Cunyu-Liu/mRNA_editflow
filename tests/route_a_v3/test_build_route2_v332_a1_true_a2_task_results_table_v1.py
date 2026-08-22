import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_a1_true_a2_task_results_table_v1 import (
    DEFAULT_A1_TASK_TABLE,
    DEFAULT_BASELINE_MATRIX,
    DEFAULT_DATASET_TABLE,
    DEFAULT_FRESHNESS_AUDIT,
    DEFAULT_GEOMETRY_AUDIT,
    DEFAULT_PACKAGE_TABLE,
    TaskResultsInputError,
    build_table,
)


def _build(tmp_path: Path, *, a1_task_table: Path = DEFAULT_A1_TASK_TABLE) -> tuple[dict, list[dict[str, str]]]:
    table = tmp_path / "results.csv"
    audit_path = tmp_path / "audit.json"
    audit = build_table(
        a1_task_table=a1_task_table,
        dataset_table=DEFAULT_DATASET_TABLE,
        baseline_matrix=DEFAULT_BASELINE_MATRIX,
        package_table=DEFAULT_PACKAGE_TABLE,
        geometry_audit=DEFAULT_GEOMETRY_AUDIT,
        freshness_audit=DEFAULT_FRESHNESS_AUDIT,
        output_table=table,
        output_audit=audit_path,
    )
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    return audit, rows


def test_builder_separates_a1_numeric_results_from_true_a2_boundaries(tmp_path: Path) -> None:
    audit, rows = _build(tmp_path)
    assert audit["status"] == "A1_TRUE_A2_TASK_RESULTS_SEPARATED_TRUE_A2_NUMERIC_NOT_TERMINAL"
    assert len(rows) == audit["row_count"] == 14
    a1 = [row for row in rows if row["estimand"] == "A1_SOURCE_RELATIVE_NUMERIC_EFFECT"]
    a2 = [row for row in rows if row["estimand"] == "TRUE_A2_WITHIN_SOURCE_MEASURED_CANDIDATE_RANKING"]
    assert len(a1) == audit["a1_numeric_task_row_count"] == 9
    assert len(a2) == audit["true_a2_boundary_row_count"] == 5
    assert sum(int(row["record_count"]) for row in a1) == 18293
    assert all(row["numeric_performance_result_available"] == "true" for row in a1)
    assert all(row["primary_metric"] == "spearman" for row in a1)
    assert all(row["secondary_metric"] == "mae" for row in a1)
    assert all(row["tertiary_metric"] == "standardized_mae" for row in a1)
    assert all(row["external_confirmation_eligible"] == "false" for row in a1)

    by_id = {row["result_row_id"]: row for row in a2}
    assert set(by_id) == {"A2-01", "A2-02", "A2-03", "A2-04", "A2-05"}
    assert by_id["A2-01"]["record_count"] == "30966"
    assert by_id["A2-01"]["qualified_true_a2_study_credit"] == "0"
    assert by_id["A2-02"]["terminal_result_status"] == (
        "EVALUATOR_IMPLEMENTED_NO_INDEPENDENT_TERMINAL_TRUE_A2_RESULT"
    )
    assert by_id["A2-03"]["terminal_result_status"] == (
        "CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE"
    )
    assert by_id["A2-04"]["primary_metric"] == (
        "closed_measured_ndcg_defined_source_count_all_methods"
    )
    assert by_id["A2-04"]["primary_value"] == "0"
    assert "not closed-pool true-A2 ranking" in by_id["A2-04"]["claim_boundary"]
    assert by_id["A2-05"]["record_count"] == "0"
    assert all(row["numeric_performance_result_available"] == "false" for row in a2)
    assert all(row["external_confirmation_eligible"] == "false" for row in a2)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)
    assert audit["true_a2_terminal_numeric_performance_row_count"] == 0
    assert audit["true_a2"]["benchmark_execution_complete"] is False
    assert audit["open_support_recovery_substitutes_for_true_a2_ranking"] is False
    assert audit["reporting_table_complete"] is True
    assert audit["scientific_claim_status"] == "NOT_ESTABLISHED"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_table(
            output_table=tmp_path / "results.csv",
            output_audit=tmp_path / "audit.json",
        )


def test_builder_rejects_incomplete_a1_task_table(tmp_path: Path) -> None:
    with DEFAULT_A1_TASK_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = rows[0].keys()
    shortened = tmp_path / "a1_short.csv"
    with shortened.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(TaskResultsInputError, match="must contain nine tasks"):
        build_table(
            a1_task_table=shortened,
            output_table=tmp_path / "results.csv",
            output_audit=tmp_path / "audit.json",
        )
