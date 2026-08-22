import csv
import json
from pathlib import Path

import pytest

from scripts.route_a_v3.build_route2_v332_matched_budget_baseline_matrix_v1 import (
    DEFAULT_SNAPSHOT,
    MatchedBudgetInputError,
    build_matrix,
)


def _build(tmp_path: Path, *, snapshot_path: Path = DEFAULT_SNAPSHOT):
    table = tmp_path / "matched_budget.csv"
    audit_path = tmp_path / "matched_budget.json"
    audit = build_matrix(
        snapshot_path=snapshot_path,
        output_table_path=table,
        output_audit_path=audit_path,
    )
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return audit, rows, table, audit_path


def test_builder_separates_exact_screen_matching_from_incomplete_contract_matching(
    tmp_path: Path,
) -> None:
    audit, rows, table, audit_path = _build(tmp_path)

    assert audit["status"] == (
        "MATCHED_BUDGET_REPORTING_MATRIX_RENDERED_CONTRACT_COMPLETE_MATCH_NOT_ESTABLISHED"
    )
    assert len(rows) == audit["row_count"] == 14
    assert audit["track_counts"] == {"PREDICTION": 5, "GENERATION": 9}
    assert audit["numeric_result_row_count"] == 12
    assert audit["reporting_matrix_complete"] is True
    assert audit["matched_budget_benchmark_execution_complete"] is False
    assert audit["cross_track_numeric_ranking_allowed"] is False
    assert audit["compute_reporting"]["fully_contract_matched_headline_comparison_row_count"] == 0

    prediction = [row for row in rows if row["track"] == "PREDICTION"]
    exact_screen = [
        row
        for row in prediction
        if row["training_or_hpo_budget_match_status"]
        == "EXACT_MATCH_WITHIN_CRITIC_V2_SCREEN"
    ]
    assert len(exact_screen) == audit["prediction"][
        "critic_v2_exact_within_screen_budget_rows"
    ] == 4
    assert {int(row["physical_gpu_index"]) for row in exact_screen} == {2, 3, 4, 5}
    assert {int(row["training_seed"]) for row in exact_screen} == {20260825}
    assert {int(row["training_epochs"]) for row in exact_screen} == {100}
    assert {int(row["optimizer_steps"]) for row in exact_screen} == {559900}
    assert {int(row["trainable_parameter_count"]) for row in exact_screen} == {9342914}
    assert {int(row["frozen_parameter_count"]) for row in exact_screen} == {113389056}
    assert all(row["wall_time_status"] == "RECORDED_IN_CENTRAL_TERMINAL_LEDGER" for row in exact_screen)

    hurdle = next(
        row for row in prediction if row["method_id"] == "method_repair_global_scaled_seed20260821"
    )
    assert hurdle["training_or_hpo_budget_match_status"] == (
        "SAME_INFORMATION_HURDLE_NOT_UPDATE_BUDGET_MATCHED_TO_CRITIC_V2"
    )
    assert int(hurdle["optimizer_steps"]) == audit["prediction"][
        "strongest_hurdle_optimizer_steps"
    ] == 22120
    assert audit["prediction"]["critic_v2_optimizer_steps"] == 559900
    assert hurdle["training_epochs"] == ""
    assert hurdle["trainable_parameter_count"] == ""
    assert hurdle["headline_comparison_eligible"] == "true"
    assert float(hurdle["primary_metric_value"]) == 0.13171439492559175

    generation = [row for row in rows if row["track"] == "GENERATION"]
    terminal = [
        row
        for row in generation
        if row["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    ]
    guided = [
        row for row in generation if row["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"
    ]
    assert len(terminal) == audit["generation"]["terminal_matched_method_rows"] == 7
    assert len(guided) == audit["generation"]["guided_dependency_no_go_rows"] == 2
    assert {row["action_space"] for row in terminal} == {"SUB_PLUS_STOP"}
    assert {row["edit_budget_set"] for row in terminal} == {"1|3|5"}
    assert {int(row["source_count"]) for row in terminal} == {891}
    assert {int(row["candidate_cap_per_source"]) for row in terminal} == {32}
    assert {int(row["critic_forward_cap_per_source"]) for row in terminal} == {256}
    assert {int(row["total_forward_equivalent_cap_per_source"]) for row in terminal} == {320}
    assert sum(int(row["candidate_count"]) == 28512 for row in terminal) == 6
    local_search = next(row for row in terminal if row["method_id"] == "local_search")
    assert int(local_search["candidate_count"]) == audit["generation"][
        "local_search_candidate_count"
    ] == 21027
    assert sum(row["wall_time_status"] == "NOT_RECORDED_NO_TERMINAL_RERUN" for row in terminal) == 6
    flow = next(
        row for row in terminal if row["method_id"] == "unguided_learned_base_flow_g0"
    )
    assert float(flow["wall_time_seconds"]) == 341.55688762664795
    assert flow["generation_budget_match_status"].endswith("WITH_NO_CRITIC_CALLS")
    assert all(row["numeric_result_available"] == "false" for row in guided)
    assert all(row["primary_metric_value"] == row["candidate_count"] == "" for row in guided)
    assert all(row["guided_required"] == "true" for row in guided)
    assert all(row["guided_executed"] == "false" for row in guided)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_executed"] == "false" for row in rows)

    with pytest.raises(FileExistsError):
        build_matrix(output_table_path=table, output_audit_path=audit_path)


def test_builder_rejects_false_exact_compute_match_for_frozen_hurdle(
    tmp_path: Path,
) -> None:
    snapshot = json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["strongest_same_information_hurdle"]["optimizer_steps"] = 559900
    bad_snapshot = tmp_path / "bad_snapshot.json"
    bad_snapshot.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(
        MatchedBudgetInputError,
        match="strongest hurdle is no longer lower-update than Critic V2",
    ):
        _build(tmp_path / "bad", snapshot_path=bad_snapshot)
