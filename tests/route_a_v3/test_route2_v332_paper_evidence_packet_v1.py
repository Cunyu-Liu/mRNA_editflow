import json
import re
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
EVIDENCE = ROOT / "docs/paper/route2_v332_evidence_manifest_v1.json"
CONSISTENCY = ROOT / "docs/paper/route2_v332_consistency_manifest_v1.json"
BOOTSTRAP_TABLE = ROOT / "docs/paper/route2_v332_generation_bootstrap_table_v1.csv"
ACTION_SPACE_GEOMETRY_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv"
)
ACTION_SPACE_GEOMETRY_AUDIT = (
    ROOT / "audits/route_a_v3_route2_generation_action_space_geometry_v1.json"
)
MINIMUM_PACKAGE_TABLE = (
    ROOT / "docs/paper/route2_v332_minimum_benchmark_package_table_v1.csv"
)
MINIMUM_PACKAGE_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_minimum_benchmark_package_v1.json"
)
BASELINE_MATRIX_TABLE = ROOT / "docs/paper/route2_v332_baseline_matrix_v1.csv"
BASELINE_MATRIX_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_baseline_matrix_v1.json"
)
THREE_TRACK_TABLE = ROOT / "docs/paper/route2_v332_three_track_results_table_v1.csv"
THREE_TRACK_AUDIT = ROOT / "audits/route_a_v3_route2_v332_three_track_results_table_v1.json"
THREE_TRACK_SNAPSHOT = ROOT / "audits/route_a_v3_route2_v332_three_track_terminal_input_snapshot_v1.json"
A1_TRUE_A2_TABLE = (
    ROOT / "docs/paper/route2_v332_a1_true_a2_task_results_table_v1.csv"
)
A1_TRUE_A2_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_a1_true_a2_task_results_table_v1.json"
)
MATCHED_BUDGET_TABLE = (
    ROOT / "docs/paper/route2_v332_matched_budget_baseline_matrix_v1.csv"
)
MATCHED_BUDGET_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_matched_budget_baseline_matrix_v1.json"
)
MATCHED_BUDGET_SNAPSHOT = (
    ROOT / "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json"
)
GENERATION_THREE_LAYER_TABLE = (
    ROOT / "docs/paper/route2_v332_generation_three_layer_results_table_v1.csv"
)
GENERATION_THREE_LAYER_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_generation_three_layer_results_table_v1.json"
)
GENERATION_THREE_LAYER_SNAPSHOT = (
    ROOT / "audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json"
)
DATASET_QUALIFICATION_TABLE = (
    ROOT / "docs/paper/route2_v332_dataset_qualification_table_v1.csv"
)
DATASET_QUALIFICATION_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_dataset_qualification_table_v1.json"
)
GSE232572_HISTORICAL = (
    ROOT / "audits/route_a_v3_route2_gse232572_zero_shot_summary_v1.json"
)
EVALUATOR_TASK_TABLE = (
    ROOT / "docs/paper/route2_v332_independent_evaluator_task_table_v1.csv"
)
EVALUATOR_CHECK_TABLE = (
    ROOT
    / "docs/paper/route2_v332_independent_evaluator_qualification_checks_v1.csv"
)
CRITIC_V2_DIAGNOSTIC_TABLE = (
    ROOT / "docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv"
)
ERROR_DOMAIN_SHIFT_TABLE = (
    ROOT / "docs/paper/route2_v332_error_domain_shift_analysis_table_v1.csv"
)
ERROR_DOMAIN_SHIFT_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_error_domain_shift_analysis_table_v1.json"
)
PAPER_OUTCOME_ADJUDICATION = (
    ROOT / "audits/route_a_v3_route2_v332_paper_outcome_adjudication_v1.json"
)
SELECTED_OUTCOME_CLAIM_EVIDENCE_TABLE = (
    ROOT / "docs/paper/route2_v332_selected_outcome_claim_evidence_table_v1.csv"
)
SELECTED_OUTCOME_CLAIM_EVIDENCE_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_selected_outcome_claim_evidence_table_v1.json"
)
DATA_RIGHTS_EXPOSURE_TABLE = (
    ROOT / "docs/paper/route2_v332_data_rights_exposure_limitations_table_v1.csv"
)
DATA_RIGHTS_EXPOSURE_AUDIT = (
    ROOT / "audits/route_a_v3_route2_v332_data_rights_exposure_limitations_table_v1.json"
)
READINESS = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_critic_v2_guidance_readiness_protocol_v1.json"
)
EVALUATOR_PROTOCOL = (
    ROOT
    / "configs/route_a_v3_route2_mrnabert_independent_evaluator_qualification_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_claim_and_consistency_evidence_references_are_closed() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    evidence = _load(EVIDENCE)
    consistency = _load(CONSISTENCY)

    evidence_ids = [row["evidence_id"] for row in evidence["sources"]]
    assert len(evidence_ids) == len(set(evidence_ids)) == 55

    claims = re.findall(r"\[claim:([^\]]+)\]", draft)
    assert len(claims) == len(set(claims)) == 22

    cited = set()
    for group in re.findall(r"\[evidence:([^\]]+)\]", draft):
        cited.update(item.strip() for item in group.split(","))
    for section in (consistency["methods"], consistency["results"]):
        for row in section:
            cited.update(row["evidence_ids"])
    assert cited <= set(evidence_ids)
    assert "E-R2-CRITIC-V2-READINESS" in cited


def test_evidence_source_paths_are_closed_without_overstating_verification() -> None:
    evidence = _load(EVIDENCE)
    preflight = evidence["source_path_preflight"]

    assert preflight["status"] == "PASS"
    assert preflight["source_locations_checked"] == len(evidence["sources"]) == 55
    assert (
        preflight["local_or_contract_locations_checked"]
        + preflight["a100_mnt_locations_checked"]
        == preflight["source_locations_checked"]
    )
    assert preflight["missing_locations"] == 0
    assert preflight["local_or_contract_locations_checked"] == 41
    assert preflight["a100_mnt_locations_checked"] == 14
    assert preflight["check_scope"] == "FILE_EXISTENCE_ONLY_NO_EVIDENCE_CONTENT_OPENED"
    assert preflight["human_content_verification_completed"] is False
    assert preflight["submission_readiness_changed"] is False
    assert evidence["human_verification_required"] is True
    assert evidence["submission_ready"] is False
    assert evidence["external_transfer_performed"] is True
    assert evidence["external_transfer_role"] == (
        "HISTORICALLY_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
    )
    assert evidence["independent_final_evaluation_performed"] is False

    by_id = {row["evidence_id"]: row for row in evidence["sources"]}
    assert by_id["E-R2-FIGURE-BUILDER"]["publisher_compliance_claimed"] is False
    assert by_id["E-R2-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_v1/route2_v332_figure_manifest_v1.json"
    )
    assert by_id["E-R2-DATA-TABLE"]["location"] == (
        "audits/route_a_v3_route2_v332_dataset_qualification_table_v1.json"
    )
    assert by_id["E-R2-CONVERSION-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_canonical_conversion_flow_figure_v1_manifest.json"
    )
    assert by_id["E-R2-DEV-EVAL-ARCH-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_development_evaluation_architecture_figure_v1_manifest.json"
    )
    assert by_id["E-R2-PXE-ARCH-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_predictor_xeditflow_evaluator_architecture_figure_v1_manifest.json"
    )
    assert by_id["E-R2-LEARNING-CURVES-FIGURE-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_development_learning_curves_figure_v1.py"
    )
    assert by_id["E-R2-LEARNING-CURVES-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_development_learning_curves_figure_v1_manifest.json"
    )
    assert by_id["E-R2-LEARNING-CURVES-FIGURE-MANIFEST"][
        "publisher_compliance_claimed"
    ] is False
    assert by_id["E-R2-BASELINE-MATRIX-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_baseline_matrix_v1.py"
    )
    assert by_id["E-R2-BASELINE-MATRIX-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_baseline_matrix_v1.json"
    )
    assert by_id["E-R2-THREE-TRACK-SNAPSHOT"]["location"] == (
        "audits/route_a_v3_route2_v332_three_track_terminal_input_snapshot_v1.json"
    )
    assert by_id["E-R2-THREE-TRACK-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_three_track_results_table_v1.py"
    )
    assert by_id["E-R2-THREE-TRACK-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_three_track_results_table_v1.json"
    )
    assert by_id["E-R2-A1-TRUE-A2-TABLE-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_a1_true_a2_task_results_table_v1.py"
    )
    assert by_id["E-R2-A1-TRUE-A2-TABLE-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_a1_true_a2_task_results_table_v1.json"
    )
    assert by_id["E-R2-MATCHED-BUDGET-SNAPSHOT"]["location"] == (
        "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json"
    )
    assert by_id["E-R2-MATCHED-BUDGET-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_matched_budget_baseline_matrix_v1.py"
    )
    assert by_id["E-R2-MATCHED-BUDGET-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_matched_budget_baseline_matrix_v1.json"
    )
    assert by_id["E-R2-GEN-THREE-LAYER-SNAPSHOT"]["location"] == (
        "audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json"
    )
    assert by_id["E-R2-GEN-THREE-LAYER-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_generation_three_layer_results_table_v1.py"
    )
    assert by_id["E-R2-GEN-THREE-LAYER-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_generation_three_layer_results_table_v1.json"
    )
    assert by_id["E-R2-GEN-QUALITY-COST-FIGURE-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_generation_quality_cost_diversity_failure_figure_v1.py"
    )
    assert by_id["E-R2-GEN-QUALITY-COST-FIGURE-MANIFEST"]["location"].endswith(
        "/route2_v332_generation_quality_cost_diversity_failure_figure_v1_manifest.json"
    )
    assert by_id["E-R2-GEN-QUALITY-COST-FIGURE-MANIFEST"][
        "publisher_compliance_claimed"
    ] is False
    assert by_id["E-R2-ERROR-DOMAIN-SHIFT-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_error_domain_shift_analysis_table_v1.py"
    )
    assert by_id["E-R2-ERROR-DOMAIN-SHIFT-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_error_domain_shift_analysis_table_v1.json"
    )
    assert by_id["E-R2-PAPER-OUTCOME-ADJUDICATION"]["location"] == (
        "audits/route_a_v3_route2_v332_paper_outcome_adjudication_v1.json"
    )
    assert by_id["E-R2-CLAIM-EVIDENCE-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_selected_outcome_claim_evidence_table_v1.py"
    )
    assert by_id["E-R2-CLAIM-EVIDENCE-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_selected_outcome_claim_evidence_table_v1.json"
    )


def test_dataset_qualification_table_closes_v332_role_and_credit_boundaries() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(DATASET_QUALIFICATION_AUDIT)
    with DATASET_QUALIFICATION_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["study_unit_id"]: row for row in rows}
    method = next(
        row for row in consistency["methods"] if row["method_id"] == "M-R2-DATA"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-DATA-QUALIFICATION"
    )

    assert len(rows) == len(by_id) == audit["study_count"] == result["study_count"] == 14
    assert sum(int(row["development_canonical_records"]) for row in rows) == (
        result["development_canonical_records"]
    ) == 126165
    assert sum(int(row["historical_transfer_canonical_records"]) for row in rows) == (
        result["historical_transfer_canonical_records"]
    ) == 8068
    assert sum(
        int(row["final_evaluation_unexposed_canonical_records"]) for row in rows
    ) == result["final_evaluation_unexposed_canonical_records"] == 0
    assert sum(int(row["qualified_canonical_credit_records"]) for row in rows) == (
        result["qualified_canonical_credit_records"]
    ) == 6547
    assert sum(int(row["canonical_records"]) == 0 for row in rows) == (
        result["zero_canonical_record_study_count"]
    ) == 6
    assert method["qualified_credit"] == {
        "ordinary_studies": 1,
        "a1_studies": 1,
        "true_a2_studies": 0,
        "canonical_records": 6547,
        "only_credited_study_unit_id": "GSE200304",
    }
    assert by_id["GSE232572"]["current_analysis_role_v332"] == (
        result["gse232572_current_role"]
    ) == "HISTORICAL_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
    assert result["new_final_evaluation_read"] is False
    assert result["sealed_gse246381_read"] is False
    assert "Generated candidates add zero canonical credit" in draft


def test_a1_numeric_tasks_and_true_a2_result_boundaries_remain_separate() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(A1_TRUE_A2_AUDIT)
    with A1_TRUE_A2_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-A1-TRUE-A2-TASK-RESULTS"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-A1-TRUE-A2-TASK-RESULTS"
    )
    a1_rows = [row for row in rows if row["result_row_id"].startswith("A1-")]
    true_a2_rows = [row for row in rows if row["result_row_id"].startswith("A2-")]

    assert len(rows) == audit["row_count"] == method["row_count"] == result["row_count"] == 14
    assert len(a1_rows) == audit["a1_numeric_task_row_count"] == result[
        "a1_numeric_task_rows"
    ] == 9
    assert sum(int(row["record_count"]) for row in a1_rows) == audit["a1"][
        "development_validation_record_count"
    ] == result["a1_development_validation_records"] == 18293
    assert all(row["numeric_performance_result_available"] == "true" for row in a1_rows)
    assert sum(float(row["primary_value"]) > 0.0 for row in a1_rows) == result[
        "a1_positive_task_spearman_count"
    ] == 5
    assert [
        min(float(row["primary_value"]) for row in a1_rows),
        max(float(row["primary_value"]) for row in a1_rows),
    ] == result["a1_task_spearman_range"]

    assert len(true_a2_rows) == audit["true_a2_boundary_row_count"] == result[
        "true_a2_boundary_rows"
    ] == 5
    assert all(
        row["numeric_performance_result_available"] == "false"
        for row in true_a2_rows
    )
    assert audit["true_a2"]["development_listwise_record_count"] == result[
        "true_a2_development_listwise_records"
    ] == 30966
    assert audit["true_a2"]["qualified_true_a2_study_credit"] == result[
        "qualified_true_a2_study_credit"
    ] == 0
    assert audit["true_a2_terminal_numeric_performance_row_count"] == result[
        "true_a2_terminal_numeric_performance_rows"
    ] == 0
    assert audit["true_a2"]["closed_measured_ndcg_defined_source_count_all_methods"] == (
        result["true_a2_closed_measured_ndcg_defined_source_count_all_methods"]
    ) == 0
    assert audit["true_a2"]["new_independent_evaluation_unexposed_record_count"] == (
        result["true_a2_new_independent_evaluation_unexposed_records"]
    ) == 0
    assert audit["cross_estimand_numeric_ranking_allowed"] is result[
        "cross_estimand_numeric_ranking_allowed"
    ] is False
    assert audit["open_support_recovery_substitutes_for_true_a2_ranking"] is result[
        "open_support_recovery_substitutes_for_true_a2_ranking"
    ] is False
    assert all(row["external_confirmation_eligible"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)
    assert "does not encode unavailable true-A2 results as zero performance" in draft
    assert "A1 and true-A2 are not placed in a cross-estimand numeric ranking" in draft


def test_matched_budget_matrix_reports_exact_and_incomplete_matching_separately() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(MATCHED_BUDGET_AUDIT)
    snapshot = _load(MATCHED_BUDGET_SNAPSHOT)
    with MATCHED_BUDGET_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-MATCHED-BUDGET-BASELINE-MATRIX"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-MATCHED-BUDGET-BASELINE-MATRIX"
    )
    baseline_result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-BASELINE-MATRIX"
    )

    assert len(rows) == audit["row_count"] == method["row_count"] == result["row_count"] == 14
    assert audit["track_counts"] == method["track_counts"] == {
        "PREDICTION": 5,
        "GENERATION": 9,
    }
    assert result["prediction_rows"] == 5
    assert result["generation_rows"] == 9
    assert audit["numeric_result_row_count"] == result["numeric_result_rows"] == 12

    prediction = [row for row in rows if row["track"] == "PREDICTION"]
    exact_screen = [
        row
        for row in prediction
        if row["training_or_hpo_budget_match_status"]
        == "EXACT_MATCH_WITHIN_CRITIC_V2_SCREEN"
    ]
    assert len(exact_screen) == audit["prediction"][
        "critic_v2_exact_within_screen_budget_rows"
    ] == result["critic_v2_exact_within_screen_budget_rows"] == 4
    assert {int(row["optimizer_steps"]) for row in exact_screen} == {
        result["critic_v2_optimizer_steps"]
    } == {559900}
    hurdle = next(
        row
        for row in prediction
        if row["method_id"] == "method_repair_global_scaled_seed20260821"
    )
    assert int(hurdle["optimizer_steps"]) == audit["prediction"][
        "strongest_hurdle_optimizer_steps"
    ] == result["strongest_same_information_hurdle_optimizer_steps"] == 22120
    assert hurdle["training_or_hpo_budget_match_status"] == (
        "SAME_INFORMATION_HURDLE_NOT_UPDATE_BUDGET_MATCHED_TO_CRITIC_V2"
    )
    assert audit["prediction"]["strongest_hurdle_update_budget_matched"] is result[
        "strongest_same_information_hurdle_update_budget_matched"
    ] is False
    assert result["full_minus_hurdle_task_macro_spearman"] == -0.015343731738697977

    generation = [row for row in rows if row["track"] == "GENERATION"]
    terminal = [
        row
        for row in generation
        if row["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    ]
    guided = [
        row for row in generation if row["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"
    ]
    assert len(terminal) == audit["generation"]["terminal_matched_method_rows"] == result[
        "terminal_generation_rows"
    ] == 7
    assert len(guided) == audit["generation"]["guided_dependency_no_go_rows"] == result[
        "guided_dependency_no_go_rows"
    ] == 2
    assert audit["generation"]["search_method_wall_time_not_recorded_rows"] == result[
        "search_method_wall_time_not_recorded_rows"
    ] == 6
    assert all(row["numeric_result_available"] == "false" for row in guided)
    assert all(row["guided_executed"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)

    assert audit["compute_reporting"][
        "fully_contract_matched_headline_comparison_row_count"
    ] == result["fully_contract_matched_headline_comparison_rows"] == 0
    assert audit["reporting_matrix_complete"] is method[
        "reporting_matrix_complete"
    ] is result["reporting_matrix_complete"] is True
    assert audit["matched_budget_benchmark_execution_complete"] is method[
        "matched_budget_benchmark_execution_complete"
    ] is result["matched_budget_benchmark_execution_complete"] is False
    assert snapshot["new_training_attempt_created"] is False
    assert baseline_result["native_common_arch_three_track_results_table_built"] is True
    assert baseline_result[
        "prediction_generation_matched_budget_numeric_matrix_built"
    ] is True
    assert baseline_result["generation_three_layer_result_table_built"] is True
    assert "not update-budget matched" in draft
    assert "zero fully contract-matched headline comparison rows" in draft


def test_generation_three_layer_table_preserves_self_score_and_measured_boundaries() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(GENERATION_THREE_LAYER_AUDIT)
    snapshot = _load(GENERATION_THREE_LAYER_SNAPSHOT)
    with GENERATION_THREE_LAYER_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-GENERATION-THREE-LAYER-RESULTS"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-GENERATION-THREE-LAYER-RESULTS"
    )
    assert len(rows) == audit["row_count"] == method["row_count"] == result["row_count"] == 9
    assert audit["executed_terminal_method_rows"] == result[
        "executed_terminal_method_rows"
    ] == 7
    assert audit["guided_no_go_boundary_rows"] == result[
        "guided_no_go_boundary_rows"
    ] == 2
    assert audit["numeric_coverage"] == {
        "critic_self_score_rows": result["critic_self_score_numeric_rows"],
        "independent_evaluator_rows": result["independent_evaluator_numeric_rows"],
        "measured_candidate_recovery_rows": result[
            "measured_candidate_recovery_numeric_rows"
        ],
        "closed_measured_ndcg_rows": result["closed_measured_ndcg_numeric_rows"],
        "biological_improvement_claim_rows": 0,
    } == {
        "critic_self_score_rows": 6,
        "independent_evaluator_rows": 7,
        "measured_candidate_recovery_rows": 7,
        "closed_measured_ndcg_rows": 0,
        "biological_improvement_claim_rows": 0,
    }
    executed = [
        row
        for row in rows
        if row["result_status"] == "EXECUTED_TERMINAL_MATCHED_DEVELOPMENT"
    ]
    guided = [row for row in rows if row["result_status"] == "NOT_RUN_CRITIC_V2_NO_GO"]
    assert len(executed) == 7 and len(guided) == 2
    assert all(row["source_macro_closed_measured_ndcg_at_k"] == "" for row in rows)
    assert all(row["cross_layer_numeric_ranking_allowed"] == "false" for row in rows)
    assert all(row["biological_improvement_claim_allowed"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_executed"] == "false" for row in rows)
    assert all(row["candidate_count"] == "" for row in guided)
    flow = next(row for row in executed if row["method_id"] == "unguided_learned_base_flow_g0")
    assert flow["critic_layer_status"] == "NOT_APPLICABLE_NO_CRITIC_CALLS"
    assert flow["source_macro_critic_max_uplift_over_source"] == ""
    assert result["critic_self_score_leader_method_id"] == "genetic"
    assert result["independent_evaluator_leader_method_id"] == "genetic"
    assert result["measured_candidate_recovery_leader_method_id"] == (
        "unguided_learned_base_flow_g0"
    )
    assert result["critic_or_independent_substitutes_for_measured_outcome"] is False
    assert result["headline_generation_improvement_established"] is False
    assert snapshot["aggregation_policy"]["candidate_payload_opened"] is False
    assert snapshot["aggregation_policy"]["missing_numeric_value_substituted_with_zero"] is False
    assert "does not substitute for measured outcome" in draft
    assert "zero sources with defined closed measured NDCG" in draft


def test_provisional_figure_method_preserves_protected_outcome_boundary() -> None:
    consistency = _load(CONSISTENCY)
    method = next(
        row for row in consistency["methods"] if row["method_id"] == "M-R2-FIGURES"
    )

    assert method["status"] == "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED"
    assert method["figure_count"] == 7
    assert method["formats"] == ["png", "pdf", "svg"]
    assert method["raster_dpi"] == 300
    assert method["target_journal"] == "PENDING_SELECTION"
    assert method["publisher_compliance_claimed"] is False
    assert method["protected_outcomes"] == {
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "emtab10902_outcome_read": False,
        "sealed_gse246381_read": False,
        "guided_xeditflow_run": False,
        "historical_outcome_exposed_gse232572_read": True,
    }
    assert method["evidence_ids"] == [
        "E-R2-FIGURE-BUILDER",
        "E-R2-FIGURE-MANIFEST",
        "E-R2-CONVERSION-FIGURE-BUILDER",
        "E-R2-CONVERSION-FIGURE-MANIFEST",
        "E-R2-DEV-EVAL-ARCH-FIGURE-BUILDER",
        "E-R2-DEV-EVAL-ARCH-FIGURE-MANIFEST",
        "E-R2-PXE-ARCH-FIGURE-BUILDER",
        "E-R2-PXE-ARCH-FIGURE-MANIFEST",
        "E-R2-LEARNING-CURVES-FIGURE-BUILDER",
        "E-R2-LEARNING-CURVES-FIGURE-MANIFEST",
        "E-R2-GEN-QUALITY-COST-FIGURE-BUILDER",
        "E-R2-GEN-QUALITY-COST-FIGURE-MANIFEST",
    ]

    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-CANONICAL-CONVERSION-FLOW"
    )
    assert result["registered_study_count"] == 14
    assert result["development_canonical_records"] == 126165
    assert result["historical_transfer_canonical_records"] == 8068
    assert result["split_record_counts"] == {
        "TRAIN": 89580,
        "VALIDATION": 18293,
        "TEST_WITHHELD": 18292,
    }
    assert result["workflow_arrow_widths_encode_magnitude"] is False
    assert result["new_final_evaluation_unexposed_records"] == 0

    architecture = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-DEVELOPMENT-EVALUATION-ARCHITECTURE"
    )
    assert architecture["development_canonical_records"] == 126165
    assert architecture["split_record_counts"] == {
        "TRAIN": 89580,
        "VALIDATION": 18293,
        "TEST_WITHHELD": 18292,
    }
    assert architecture["critic_ready_for_guidance"] is False
    assert architecture["historical_gse232572_final_confirmation_eligible"] is False
    assert architecture["emtab10902_outcome_read"] is False
    assert architecture["gse246381_outcome_read"] is False
    assert architecture["replacement_evaluation_registered"] is False
    assert architecture["replacement_evaluation_unexposed_records"] == 0
    assert architecture["replacement_evaluation_opened"] is False
    assert architecture["final_evaluation_order"] == [
        "FREEZE_PREDICTOR_GENERATOR_BASELINES_METRICS_AND_ADAPTATION_POLICY",
        "RUN_AND_PERMANENTLY_RECORD_ONE_NEW_STUDY_ZERO_SHOT",
        "ONLY_THEN_ALLOW_CALIBRATION_OR_FEW_SHOT_ADAPTATION",
        "ZERO_SHOT_REMAINS_HEADLINE",
    ]

    system_architecture = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-PREDICTOR-XEDITFLOW-EVALUATOR-ARCHITECTURE"
    )
    assert system_architecture["delta_critic"] == {
        "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
        "frozen_encoder_parameter_count": 113389056,
        "trainable_head_parameter_count": 9342914,
        "total_effective_parameter_count": 122731970,
        "critic_ready_for_guidance": False,
    }
    assert system_architecture["legal_xeditflow"]["engineering_status"] == "FLOW_G0_READY"
    assert system_architecture["legal_xeditflow"]["guided_xeditflow_run"] is False
    assert system_architecture["legal_xeditflow"]["action_types_in_scope"] == ["SUB", "STOP"]
    assert system_architecture["legal_xeditflow"]["action_types_out_of_scope"] == ["INS", "DEL"]
    assert system_architecture["independent_evaluator"]["terminal_actual_trainable_parameter_count"] == 509845
    assert system_architecture["independent_evaluator"]["architecture_distinct_from_guide"] is True
    assert system_architecture["frozen_feedback_boundaries"] == {
        "critic_parameter_update_during_generation": False,
        "generator_gradient_into_critic": False,
        "evaluation_model_gradient_into_generator": False,
        "evaluation_records_used_for_reward": 0,
    }
    assert system_architecture["development_test_read"] is False
    assert system_architecture["new_final_evaluation_read"] is False

    learning_curves = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-DEVELOPMENT-LEARNING-CURVES"
    )
    assert learning_curves["status"] == (
        "PROVISIONAL_DEVELOPMENT_LEARNING_CURVES_FIGURE_RENDERED"
    )
    assert learning_curves["raw_unsmoothed_histories"] is True
    assert learning_curves["cross_panel_metric_comparison_allowed"] is False
    assert learning_curves["predictor_profile_count"] == 6
    assert learning_curves["predictor_epoch_count"] == 8
    assert learning_curves["predictor_curve_metric"] == (
        "POOLED_DEVELOPMENT_VALIDATION_SPEARMAN"
    )
    assert learning_curves["predictor_selection_metric"] == (
        "DEVELOPMENT_VALIDATION_TASK_MACRO_SPEARMAN"
    )
    assert learning_curves["critic_arm_count"] == 4
    assert learning_curves["critic_epoch_count"] == 100
    assert learning_curves["critic_full_selected_epoch"] == 98
    assert learning_curves["critic_full_selected_task_macro_spearman"] == 0.11637066318689378
    assert learning_curves["critic_strongest_same_information_hurdle"] == 0.13171439492559175
    assert learning_curves["independent_evaluator_selected_task_macro_spearman"] == 0.10256553571558498
    assert learning_curves["independent_evaluator_exclusive_threshold"] == 0.1012475745988908
    assert learning_curves["base_flow_selected_epoch"] == 1
    assert learning_curves["base_flow_epoch1_validation_nll"] == 5.512483521877043
    assert learning_curves["base_flow_epoch30_validation_nll"] == 9.939703254814608
    assert learning_curves["base_flow_overfitting_pattern_visible"] is True
    assert learning_curves["base_flow_guided_critic_used"] is False
    assert learning_curves["base_flow_biological_optimization_established"] is False
    assert learning_curves["development_test_read"] is False
    assert learning_curves["new_final_evaluation_read"] is False
    assert learning_curves["guided_xeditflow_run"] is False
    assert learning_curves["publisher_compliance_claimed"] is False


def test_paper_packet_matches_frozen_critic_v2_readiness_boundary() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    readiness = _load(READINESS)
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-CRITIC-V2-GATES"
    )

    assert method["protocol_status"] == readiness["status"]
    assert method["status"] == "NOT_STARTED_CONTROL_GATE_NO_GO"
    assert method["required_seeds"] == readiness["required_seeds"]
    assert method["single_frozen_test_seed"] == readiness["single_frozen_test_seed"]
    assert method["single_test_metric_policy"] == readiness["single_test_metric_policy"]
    assert method["required_loso_studies"] == readiness["required_loso_studies"]
    assert method["guided_generation_requires"] == readiness["guided_generation_requires"]
    assert "TEST metrics are report-only" in draft
    assert "guided XEditFlow remains unauthorized" in draft
    assert consistency["protected_outcomes"] == {
        "development_test_opened": False,
        "new_final_evaluation_opened": False,
        "guided_xeditflow_authorized": False,
        "historical_outcome_exposed_gse232572_included": True,
        "historical_outcome_exposed_gse232572_final_confirmation_eligible": False,
    }


def test_paper_packet_reports_terminal_critic_v2_control_no_go_exactly() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    normalized_draft = " ".join(draft.split())
    consistency = _load(CONSISTENCY)
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-CRITIC-V2"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-CRITIC-V2"
    )

    assert method["status"] == result["status"] == (
        "CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS"
    )
    assert result["all_arms_completed"] is True
    assert result["arm_count"] == 4
    assert result["epochs_per_arm"] == 100
    assert result["optimizer_steps_per_arm"] == 559900
    assert result["full_task_macro_spearman"] == 0.11637066318689378
    assert result["strongest_same_information_baseline_task_macro_spearman"] == (
        0.13171439492559175
    )
    assert result["full_over_strongest_baseline_task_macro_spearman"] == (
        result["full_task_macro_spearman"]
        - result["strongest_same_information_baseline_task_macro_spearman"]
    )
    assert result["control_checks_passed"] == 7
    assert result["control_checks_total"] == 8
    assert result["supports_three_frozen_seeds"] is False
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert "No confirmation seed" in normalized_draft
    assert "failed its frozen control gate" in normalized_draft

    with CRITIC_V2_DIAGNOSTIC_TABLE.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    diagnostic = result["task_diagnostic"]
    spearman_margins = [
        float(row["full_minus_strongest_baseline_spearman"]) for row in rows
    ]
    mae_margins = [
        float(row["full_minus_strongest_baseline_standardized_mae"])
        for row in rows
    ]
    small_task_margins = [
        margin
        for row, margin in zip(rows, spearman_margins)
        if int(row["record_count"]) == 48
    ]
    remaining_task_margins = [
        margin
        for row, margin in zip(rows, spearman_margins)
        if int(row["record_count"]) != 48
    ]

    assert len(rows) == diagnostic["task_count"] == 9
    assert sum(int(row["record_count"]) for row in rows) == diagnostic[
        "validation_record_count_sum"
    ] == 18293
    assert [
        min(int(row["record_count"]) for row in rows),
        max(int(row["record_count"]) for row in rows),
    ] == diagnostic["task_record_count_range"] == [48, 12048]
    assert sum(margin > 0.0 for margin in spearman_margins) == diagnostic[
        "full_spearman_win_count_vs_strongest_baseline"
    ] == 4
    assert sum(margin < 0.0 for margin in spearman_margins) == diagnostic[
        "full_spearman_loss_count_vs_strongest_baseline"
    ] == 5
    assert sum(margin < 0.0 for margin in mae_margins) == diagnostic[
        "full_standardized_mae_better_task_count_vs_strongest_baseline"
    ] == 0
    assert sum(margin > 0.0 for margin in mae_margins) == diagnostic[
        "full_standardized_mae_worse_task_count_vs_strongest_baseline"
    ] == 9
    assert math.isclose(
        sum(mae_margins) / len(mae_margins),
        diagnostic["nine_task_mean_standardized_mae_margin_vs_strongest_baseline"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        sum(small_task_margins),
        diagnostic["two_n48_task_sum_spearman_margin_vs_strongest_baseline"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        sum(remaining_task_margins) / len(remaining_task_margins),
        diagnostic[
            "remaining_seven_task_post_hoc_mean_spearman_margin_vs_strongest_baseline"
        ],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert diagnostic["post_hoc_diagnostic_replaces_frozen_gate"] is False
    assert "localized candidate-specific rank signal" in normalized_draft


def test_generation_bootstrap_reporting_is_exact_and_source_paired() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-GENERATION"
    )
    with BOOTSTRAP_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert {row["point_leader_method_id"] for row in rows} == {"genetic"}
    assert {row["analysis_unit"] for row in rows} == {"SOURCE"}
    assert {int(row["source_count"]) for row in rows} == {891}
    assert {int(row["bootstrap_seed"]) for row in rows} == {20260816}
    assert {int(row["bootstrap_iterations"]) for row in rows} == {10000}
    assert {int(row["defined_bootstrap_iterations"]) for row in rows} == {10000}
    assert all(float(row["leader_advantage_ci_95_lower"]) > 0.0 for row in rows)

    nearest = min(rows, key=lambda row: float(row["point_leader_advantage"]))
    assert nearest["candidate_method_id"] == "generate_then_rerank"
    assert float(nearest["point_leader_advantage"]) == result["nearest_competitor"][
        "point_leader_advantage"
    ]
    assert [
        float(nearest["leader_advantage_ci_95_lower"]),
        float(nearest["leader_advantage_ci_95_upper"]),
    ] == result["nearest_competitor"]["leader_advantage_ci_95"]
    assert result["bootstrap_analysis_unit"] == "SOURCE"
    assert result["bootstrap_seed"] == 20260816
    assert result["paired_comparison_count"] == 6
    assert result["all_leader_advantage_ci_95_lower_bounds_positive"] is True
    assert "Development independent-evaluator separation only" in draft


def test_generation_action_space_geometry_is_conserved_and_claim_bounded() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    normalized_draft = " ".join(draft.split())
    consistency = _load(CONSISTENCY)
    audit = _load(ACTION_SPACE_GEOMETRY_AUDIT)
    with ACTION_SPACE_GEOMETRY_TABLE.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-GENERATION"
    )
    geometry = result["action_space_geometry"]

    assert len(rows) == audit["conservation_checks"]["method_count"] == 7
    assert {int(row["source_count"]) for row in rows} == {891}
    assert {int(row["candidate_cap_per_source"]) for row in rows} == {32}
    assert {row["candidate_support_mode"] for row in rows} == {
        "OPEN_GENERATED_SUPPORT"
    }
    assert {
        int(row["closed_measured_ndcg_defined_source_count"]) for row in rows
    } == {0}

    edit_distance_columns = [
        f"edit_distance_{distance}_count" for distance in range(6)
    ]
    for row in rows:
        candidate_count = int(row["candidate_count"])
        assert sum(int(row[column]) for column in edit_distance_columns) == (
            candidate_count
        )
        assert (
            int(row["explicit_stop_count"])
            + int(row["budget_exhausted_count"])
            + int(row["no_legal_action_count"])
            + int(row["numerical_failure_count"])
        ) == candidate_count
        assert (
            int(row["unique_candidate_count"])
            + int(row["duplicate_candidate_count"])
        ) == candidate_count
        assert float(row["hard_legality_rate"]) == 1.0
        assert int(row["edit_budget_violation_count"]) == 0
        assert int(row["candidate_budget_violation_count"]) == 0

    local_search = next(row for row in rows if row["method_id"] == "local_search")
    assert int(local_search["candidate_count"]) == geometry[
        "local_search_candidate_count"
    ] == 21027
    assert [
        int(local_search["source_candidate_count_min"]),
        int(local_search["source_candidate_count_max"]),
    ] == geometry["local_search_source_candidate_count_range"] == [3, 32]
    assert float(local_search["source_candidate_count_mean"]) == geometry[
        "local_search_mean_candidates_per_source"
    ]

    flow = next(
        row
        for row in rows
        if row["method_id"] == "unguided_learned_base_flow_g0"
    )
    assert int(flow["unique_candidate_count"]) == geometry[
        "flow_unique_candidate_count"
    ] == 25173
    assert int(flow["duplicate_candidate_count"]) == geometry[
        "flow_duplicate_candidate_count"
    ] == 3339
    assert float(flow["budget_exhausted_rate"]) == geometry[
        "highest_budget_exhausted_rate"
    ] == 0.8702651515151515

    assert audit["protocol_boundary"]["action_types_in_scope"] == ["SUB", "STOP"]
    assert audit["protocol_boundary"]["action_types_out_of_scope"] == ["INS", "DEL"]
    assert audit["protocol_boundary"]["development_test_outcomes_read"] == 0
    assert audit["protocol_boundary"]["new_final_evaluation_outcomes_read"] == 0
    assert audit["protocol_boundary"]["generated_candidates_grant_canonical_credit"] is False
    assert audit["protocol_boundary"]["unknown_generated_candidates_are_zero_gain"] is False
    assert geometry["total_no_legal_action_terminal_count"] == 0
    assert geometry["total_numerical_failure_terminal_count"] == 0
    assert "INS/DEL are outside this first-stage benchmark" in normalized_draft
    assert "computational action-space properties" in normalized_draft

    stop_gap = next(
        row
        for row in consistency["known_reporting_gaps"]
        if row["gap_id"] == "GAP-R2-GEN-STOP-TIME"
    )
    assert stop_gap["status"] == (
        "NOT_RETAINED_IN_TERMINAL_SELECTION_INPUT_NO_TERMINAL_RERUN"
    )


def test_generation_quality_cost_diversity_failure_figure_is_claim_bounded() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"]
        == "R-R2-GENERATION-QUALITY-COST-DIVERSITY-FAILURE-FIGURE"
    )
    with ACTION_SPACE_GEOMETRY_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def pareto(metric: str) -> set[str]:
        front = set()
        for row in rows:
            cost = float(row["mean_total_forward_equivalents_per_source"])
            quality = float(row[metric])
            dominated = any(
                float(other["mean_total_forward_equivalents_per_source"]) <= cost
                and float(other[metric]) >= quality
                and (
                    float(other["mean_total_forward_equivalents_per_source"]) < cost
                    or float(other[metric]) > quality
                )
                for other in rows
                if other["method_id"] != row["method_id"]
            )
            if not dominated:
                front.add(row["method_id"])
        return front

    assert len(rows) == result["terminal_method_count"] == 7
    assert pareto("source_macro_independent_evaluator_max_uplift_over_source") == set(
        result["independent_evaluator_point_estimate_pareto_method_ids"]
    ) == {"random_legal", "unguided_learned_base_flow_g0", "genetic"}
    assert pareto("source_macro_candidate_recovery_rate") == set(
        result["measured_candidate_recovery_point_estimate_pareto_method_ids"]
    ) == {"random_legal", "unguided_learned_base_flow_g0"}
    assert result["lowest_cost_mean_forward_equivalents_per_source"] == 64.00448933782268
    assert result["highest_independent_evaluator_max_uplift"] == 1.0978248587628674
    assert result["highest_measured_candidate_recovery_rate"] == 0.20286195286195285
    assert result["highest_source_macro_pairwise_hamming_diversity"] == 0.0765737532452552
    assert result["flow_duplicate_candidate_fraction"] == 3339 / 28512
    assert result["local_search_candidate_cap_shortfall_fraction"] == (
        28512 - 21027
    ) / 28512
    assert result["all_method_hard_legality_rate"] == 1.0
    assert result["total_edit_budget_violations"] == 0
    assert result["total_candidate_budget_violations"] == 0
    assert result["total_no_legal_action_terminals"] == 0
    assert result["total_numerical_failure_terminals"] == 0
    assert result["per_method_uncertainty_available"] is False
    assert result["generation_wall_time_complete"] is False
    assert result["closed_measured_ndcg_numeric_rows"] == 0
    assert result["guided_methods_plotted_as_executed"] is False
    assert result["publisher_compliance_claimed"] is False
    assert result["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert "descriptive independent-evaluator quality–cost frontier" in draft
    assert "without per-method intervals" in draft
    assert "Closed measured NDCG and complete wall time remain unavailable" in draft


def test_error_domain_shift_table_separates_development_and_historical_layers() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(ERROR_DOMAIN_SHIFT_AUDIT)
    with ERROR_DOMAIN_SHIFT_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-ERROR-DOMAIN-SHIFT-ANALYSIS"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-ERROR-DOMAIN-SHIFT-ANALYSIS"
    )
    development = [
        row
        for row in rows
        if row["analysis_layer"] == "DEVELOPMENT_VALIDATION_TASK_ERROR_DIAGNOSTIC"
    ]
    historical = [
        row
        for row in rows
        if row["analysis_layer"]
        == "HISTORICAL_OUTCOME_EXPOSED_ZERO_SHOT_DOMAIN_SHIFT"
    ]

    assert len(rows) == audit["row_count"] == method["row_count"] == result["row_count"] == 12
    assert len(development) == audit["development_task_rows"] == result[
        "development_task_rows"
    ] == 9
    assert len(historical) == audit["historical_seed_rows"] == result[
        "historical_seed_rows"
    ] == 3
    assert sum(int(row["record_count"]) for row in development) == result[
        "development_validation_record_count_sum"
    ] == 18293
    assert {int(row["record_count"]) for row in historical} == {8068}
    assert len({row["study_id"] for row in development}) == method[
        "development_study_count"
    ] == 7
    assert len({row["assay_id"] for row in development}) == method[
        "development_assay_count"
    ] == 7
    assert all(row["historical_seed"] == "" for row in development)
    assert all(row["critic_v2_full_spearman"] == "" for row in historical)
    assert all(row["independent_evaluator_spearman"] == "" for row in historical)
    assert all(row["external_confirmation_eligible"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)

    geometry = audit["development_failure_geometry"]
    assert geometry["critic_v2_spearman_win_count_vs_strongest_baseline"] == result[
        "critic_v2_spearman_win_count_vs_strongest_baseline"
    ] == 4
    assert geometry["critic_v2_spearman_loss_count_vs_strongest_baseline"] == result[
        "critic_v2_spearman_loss_count_vs_strongest_baseline"
    ] == 5
    assert geometry[
        "critic_v2_standardized_mae_worse_task_count_vs_strongest_baseline"
    ] == result[
        "critic_v2_standardized_mae_worse_task_count_vs_strongest_baseline"
    ] == 9
    assert geometry["minimum_task_record_count"] == 48
    assert geometry["maximum_task_record_count"] == 12048
    assert geometry["maximum_to_minimum_task_record_count_ratio"] == 251.0
    assert geometry["n48_exclusion_is_replacement_endpoint"] is False
    assert geometry["causal_mechanism_established"] is result[
        "causal_mechanism_established"
    ] is False

    assert audit["descriptive_region_summaries"]["5UTR"]["task_count"] == 4
    assert audit["descriptive_region_summaries"]["3UTR"]["task_count"] == 5
    assert result["region_summaries_are_post_hoc_and_confounded"] is True
    assert audit["assay_context_resolution"][
        "within_assay_context_specific_error_metrics_available"
    ] is method["within_assay_context_specific_error_metrics_available"] is result[
        "within_assay_context_specific_error_metrics_available"
    ] is False
    assert audit["historical_domain_shift"][
        "rank_improvement_ci_excludes_zero_seed_count"
    ] == result["historical_rank_improvement_ci_excludes_zero_seed_count"] == 2
    assert audit["historical_domain_shift"][
        "baseline_mae_minus_model_mae_negative_seed_count"
    ] == result["historical_baseline_mae_minus_model_mae_negative_seed_count"] == 3
    assert audit["historical_domain_shift"]["preregistered_pass"] is result[
        "historical_preregistered_pass"
    ] is False
    assert audit["external_transfer_established"] is method[
        "external_transfer_established"
    ] is False
    assert audit["metric_layer_separation"]["cross_layer_numeric_pooling_allowed"] is method[
        "cross_layer_numeric_pooling_allowed"
    ] is result["cross_layer_numeric_pooling_allowed"] is False
    assert "251-fold range" in draft
    assert "no region or context effect is claimed" in draft
    assert "outcome-exposed status precludes final confirmation" in draft


def test_paper_outcome_route_is_frozen_without_overcalling_submission_eligibility() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    package = _load(MINIMUM_PACKAGE_AUDIT)
    adjudication = _load(PAPER_OUTCOME_ADJUDICATION)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-PAPER-OUTCOME-ADJUDICATION"
    )

    selected = "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"
    assert adjudication["selected_final_paper_outcome"] == result[
        "selected_final_paper_outcome"
    ] == package["summary"]["selected_final_paper_outcome"] == selected
    assert adjudication["final_paper_outcome_frozen"] is result[
        "final_paper_outcome_frozen"
    ] is package["summary"]["current_final_paper_outcome_frozen"] is True
    assert adjudication["route_selection_status"] == result[
        "route_selection_status"
    ] == "FROZEN_FORWARD_MANUSCRIPT_ROUTE_PACKAGE_INCOMPLETE_NOT_SUBMISSION_READY"
    assert adjudication["outcome_trigger_fully_satisfied"] is result[
        "outcome_trigger_fully_satisfied"
    ] is package["summary"]["outcome_trigger_fully_satisfied"] is False
    assert adjudication["submission_level_outcome_eligibility"] is result[
        "submission_level_outcome_eligibility"
    ] is package["summary"]["submission_level_outcome_eligibility"] is False
    assert adjudication["submission_ready"] is result["submission_ready"] is False

    outcomes = {row["outcome"]: row for row in adjudication["outcome_adjudication"]}
    assert outcomes["XEDITFLOW_PLUS_DELTA_PLUS_BENCHMARK_PAPER"]["eligible"] is False
    assert outcomes["DELTA_MODEL_PLUS_BENCHMARK_PAPER"]["eligible"] is False
    outcome_c = outcomes[selected]
    assert outcome_c["selected"] is True
    assert outcome_c["eligible"] is False
    assert outcome_c["condition_status"] == {
        "minimum_benchmark_package_complete": False,
        "predictor_external_transfer_not_established_or_development_only": True,
        "generator_measured_or_independent_improvement_not_established": True,
        "negative_controls_data_geometry_and_error_analysis_support_not_single_implementation_failure": True,
        "next_generation_dataset_requirements_declared": True,
    }

    requirement_ids = [
        row["requirement_id"]
        for row in adjudication["next_generation_dataset_requirements"]
    ]
    assert requirement_ids == result["next_generation_dataset_requirement_ids"] == [
        "NEXT-DATA-SOURCE-CANDIDATE",
        "NEXT-DATA-DENSE-POOL",
        "NEXT-DATA-REPLICATE-SE",
        "NEXT-DATA-CONTEXT",
        "NEXT-DATA-EXTERNAL-EXPOSURE",
    ]
    assert adjudication["terminal_evidence"]["minimum_package_complete"] is False
    assert adjudication["terminal_evidence"]["critic_ready_for_guidance"] is False
    assert adjudication["terminal_evidence"]["flow_g0_ready"] is True
    assert adjudication["terminal_evidence"]["guided_xeditflow_run"] is False
    assert adjudication["terminal_evidence"][
        "historical_transfer_final_confirmation_eligible"
    ] is False
    assert adjudication["terminal_evidence"][
        "new_outcome_unexposed_evaluation_record_count"
    ] == 0
    assert adjudication["terminal_evidence"][
        "generation_measured_or_independent_improvement_established"
    ] is False
    assert all(value is False for value in adjudication["protected_outcomes"].values())
    assert result["model_or_biological_success_established"] is False
    assert "freezes the forward manuscript route" in draft
    assert "This selection freezes manuscript direction, not eligibility" in draft
    assert "sufficiently dense, closed measured candidate pool" in draft
    assert "biological replicate-level values and finite, positive uncertainty" in draft
    assert "outcomes unexposed until predictor, generator, baselines" in draft


def test_selected_outcome_claim_evidence_closes_markers_and_excludes_unsupported_claims() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(SELECTED_OUTCOME_CLAIM_EVIDENCE_AUDIT)
    with SELECTED_OUTCOME_CLAIM_EVIDENCE_TABLE.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-SELECTED-OUTCOME-CLAIM-EVIDENCE"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-SELECTED-OUTCOME-CLAIM-EVIDENCE"
    )

    assert len(rows) == audit["row_count"] == method["row_count"] == result[
        "row_count"
    ] == 35
    assert audit["draft_claim_marker_count"] == method[
        "draft_claim_marker_count"
    ] == result["draft_claim_marker_count"] == 22
    assert audit["supported_with_declared_boundary_row_count"] == result[
        "supported_with_declared_boundary_row_count"
    ] == 22
    assert audit["unsupported_claim_row_count"] == method[
        "unsupported_claim_row_count"
    ] == result["unsupported_claim_row_count"] == 13
    assert audit["unmapped_draft_claim_marker_count"] == result[
        "unmapped_draft_claim_marker_count"
    ] == 0
    assert audit["unknown_evidence_id_reference_count"] == result[
        "unknown_evidence_id_reference_count"
    ] == 0
    assert audit["unsupported_claims_allowed_in_manuscript_count"] == method[
        "unsupported_claims_allowed_in_manuscript_count"
    ] == result["unsupported_claims_allowed_in_manuscript_count"] == 0
    assert audit["claim_evidence_table_complete"] is result[
        "claim_evidence_table_complete"
    ] is True

    supported = [row for row in rows if row["claim_status"].startswith("SUPPORTED")]
    unsupported = [row for row in rows if row["claim_status"] == "UNSUPPORTED"]
    assert {row["claim_id"] for row in supported} == {
        f"C-R2-{index:03d}" for index in range(1, 23)
    }
    assert {row["claim_id"] for row in unsupported} == {
        f"U-R2-{index:03d}" for index in range(1, 14)
    }
    assert all(row["evidence_ids"] for row in rows)
    assert all(row["allowed_in_selected_outcome_manuscript"] == "true" for row in supported)
    assert all(row["allowed_in_selected_outcome_manuscript"] == "false" for row in unsupported)
    assert all(row["minimum_package_complete"] == "false" for row in rows)
    assert all(row["outcome_trigger_fully_satisfied"] == "false" for row in rows)
    assert all(row["submission_ready"] == "false" for row in rows)
    assert all(row["development_test_read"] == "false" for row in rows)
    assert all(row["new_final_evaluation_read"] == "false" for row in rows)
    assert all(row["guided_xeditflow_run"] == "false" for row in rows)
    assert result["selected_final_paper_outcome"] == (
        "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"
    )
    assert result["model_or_biological_success_established"] is False
    assert "closes all 22 claim markers" in draft
    assert "13 unsupported statements" in draft
    assert "No unsupported row is allowed" in draft


def test_data_rights_exposure_limitations_are_study_bound_and_not_overclaimed() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    evidence = _load(EVIDENCE)
    consistency = _load(CONSISTENCY)
    package = _load(MINIMUM_PACKAGE_AUDIT)
    audit = _load(DATA_RIGHTS_EXPOSURE_AUDIT)
    with DATA_RIGHTS_EXPOSURE_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-DATA-RIGHTS-EXPOSURE-LIMITATIONS"
    )
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-DATA-RIGHTS-EXPOSURE-LIMITATIONS"
    )
    package_section = package["data_rights_exposure_limitations_table"]

    assert len(rows) == audit["row_count"] == method["row_count"] == result[
        "row_count"
    ] == package_section["row_count"] == 14
    assert sum(row["declared_public_redistribution_allowed"] == "true" for row in rows) == 1
    assert sum(row["declared_public_redistribution_allowed"] == "false" for row in rows) == 8
    assert sum(row["declared_public_redistribution_allowed"] == "" for row in rows) == 5
    assert all(row["public_release_authorized"] == "false" for row in rows)
    assert all(row["license_verification_status"].startswith("HUMAN_REVIEW_PENDING") for row in rows)
    assert audit["converter_or_preflight_declaration_is_license_verification"] is False
    assert audit["study_bound_human_verified_license_registry_present"] is False
    assert audit["license_human_review_pending_count"] == method[
        "license_human_review_pending_count"
    ] == result["license_human_review_pending_count"] == 14
    assert audit["public_release_authorized_count"] == method[
        "public_release_authorized_count"
    ] == result["public_release_authorized_count"] == package_section[
        "public_release_authorized_count"
    ] == 0
    assert audit["data_rights_exposure_limitations_complete"] is result[
        "data_rights_exposure_limitations_complete"
    ] is package_section["data_rights_exposure_limitations_complete"] is True
    assert result["minimum_package_complete"] is package_section[
        "minimum_package_complete"
    ] is False
    assert result["submission_ready"] is package_section["submission_ready"] is False
    assert all(value is False for value in audit["protected_outcomes"].values())

    by_id = {row["evidence_id"]: row for row in evidence["sources"]}
    assert by_id["E-R2-DATA-RIGHTS-EXPOSURE-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_data_rights_exposure_limitations_table_v1.py"
    )
    assert by_id["E-R2-DATA-RIGHTS-EXPOSURE-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_data_rights_exposure_limitations_table_v1.json"
    )
    gaps = {row["gap_id"]: row for row in consistency["known_reporting_gaps"]}
    assert gaps["GAP-R2-DATA-RIGHTS-HUMAN-VERIFICATION"]["status"] == (
        "HUMAN_REVIEW_PENDING_PUBLIC_RELEASE_NOT_AUTHORIZED"
    )
    assert "Public accessibility, internal analysis permission and public redistribution authority were treated as distinct" in draft
    assert "public study-payload release is authorized for zero rows" in draft
    assert "must not claim that the Route 2 study payloads form an open-data package" in draft


def test_historical_gse232572_transfer_remains_negative_and_nonconfirmatory() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    historical = _load(GSE232572_HISTORICAL)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-GSE232-HISTORICAL-TRANSFER"
    )

    assert historical["evaluation_record_count"] == result["record_count"] == 8068
    assert historical["strongest_baseline_id"] == result[
        "strongest_baseline_id"
    ] == "neural_medium_siamese_cnn"
    paired = historical["paired_results"]
    assert [
        row["task_macro_spearman_improvement"] for row in paired
    ] == result["model_minus_baseline_task_macro_spearman"]
    assert [
        row["task_macro_spearman_improvement_ci_95"][0] for row in paired
    ] == result["spearman_improvement_ci_95_lower"]
    assert [row["baseline_mae_minus_model_mae"] for row in paired] == result[
        "baseline_mae_minus_model_mae"
    ]
    assert sum(
        row["task_macro_spearman_improvement_ci_95"][0] > 0.0 for row in paired
    ) == 2
    assert all(row["baseline_mae_minus_model_mae"] < 0.0 for row in paired)
    assert historical["preregistered_pass"] is result["preregistered_pass"] is False
    assert result["role"] == (
        "HISTORICALLY_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION"
    )
    assert "negative historical transfer evidence and not as final Evaluation" in draft
    assert "GSE232572 provides an independent final confirmation" in draft


def test_minimum_benchmark_package_is_itemized_and_not_overcalled() -> None:
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())
    consistency = _load(CONSISTENCY)
    audit = _load(MINIMUM_PACKAGE_AUDIT)
    baseline_audit = _load(BASELINE_MATRIX_AUDIT)
    three_track_audit = _load(THREE_TRACK_AUDIT)
    three_track_snapshot = _load(THREE_TRACK_SNAPSHOT)
    with MINIMUM_PACKAGE_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with BASELINE_MATRIX_TABLE.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    with THREE_TRACK_TABLE.open(newline="", encoding="utf-8") as handle:
        three_track_rows = list(csv.DictReader(handle))
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-MINIMUM-BENCHMARK-PACKAGE"
    )
    baseline_method = next(
        row
        for row in consistency["methods"]
        if row["method_id"] == "M-R2-BASELINE-MATRIX"
    )
    baseline_result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-BASELINE-MATRIX"
    )
    three_track_method = next(
        row for row in consistency["methods"]
        if row["method_id"] == "M-R2-THREE-TRACK-RESULTS"
    )
    three_track_result = next(
        row for row in consistency["results"]
        if row["result_id"] == "R-R2-THREE-TRACK-RESULTS"
    )

    assert [row["requirement_id"] for row in rows] == [
        f"MBP-{index:02d}" for index in range(1, 19)
    ]
    assert sum(row["status"].startswith("COMPLETE") for row in rows) == 14
    assert sum(row["status"].startswith("PARTIAL") for row in rows) == 3
    assert sum(row["status"].startswith("NOT_AVAILABLE") for row in rows) == 1
    assert audit["summary"]["requirement_count"] == result["requirement_count"] == 18
    assert audit["summary"]["itemwise_adjudication_complete"] is result[
        "itemwise_adjudication_complete"
    ] is True
    assert audit["itemwise_closure"] == {
        "status": "ITEMWISE_ADJUDICATION_COMPLETE_PACKAGE_INCOMPLETE",
        "adjudicated_requirement_count": 18,
        "unadjudicated_requirement_count": 0,
        "requirement_ids_complete_and_unique": True,
        "unfinished_requirements_written_as_pass": False,
        "blocking_requirement_ids": ["MBP-10", "MBP-13", "MBP-14", "MBP-15"],
        "authorized_current_cohort_action": "NO_RERUN_NO_GUIDED_NO_PROTECTED_OUTCOME_READ",
        "minimum_package_complete": False,
        "submission_ready": False,
    }
    assert result["unadjudicated_requirement_count"] == 0
    assert result["unfinished_requirements_written_as_pass"] is False
    assert result["itemwise_closure_status"] == (
        "ITEMWISE_ADJUDICATION_COMPLETE_PACKAGE_INCOMPLETE"
    )
    assert audit["summary"]["complete_or_complete_with_declared_limits_count"] == (
        result["complete_or_complete_with_declared_limits_count"]
    ) == 14
    assert audit["summary"]["partial_count"] == result["partial_count"] == 3
    assert audit["summary"]["not_available_count"] == result[
        "not_available_count"
    ] == 1
    expected_blockers = ["MBP-10", "MBP-13", "MBP-14", "MBP-15"]
    assert audit["blocking_requirement_ids"] == result[
        "blocking_requirement_ids"
    ] == expected_blockers
    assert audit["summary"]["minimum_package_complete"] is False
    assert result["minimum_package_complete"] is False
    assert audit["summary"]["submission_ready"] is result["submission_ready"] is False
    assert audit["summary"]["current_final_paper_outcome_frozen"] is True
    assert result["final_paper_outcome_frozen"] is True
    assert audit["summary"]["selected_final_paper_outcome"] == result[
        "selected_final_paper_outcome"
    ] == "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER"
    assert audit["summary"]["outcome_trigger_fully_satisfied"] is result[
        "outcome_trigger_fully_satisfied"
    ] is False
    assert audit["summary"]["submission_level_outcome_eligibility"] is result[
        "submission_level_outcome_eligibility"
    ] is False

    by_id = {row["requirement_id"]: row for row in rows}
    assert by_id["MBP-05"]["status"] == "COMPLETE"
    assert "route2_v332_a1_true_a2_task_results_table_v1.csv" in by_id["MBP-05"][
        "evidence_or_basis"
    ]
    assert by_id["MBP-10"]["status"] == "PARTIAL_GUIDED_NOT_AUTHORIZED"
    assert "three-layer generation table" in by_id["MBP-10"]["evidence_or_basis"]
    assert by_id["MBP-11"]["status"] == "COMPLETE_HISTORICAL_NEGATIVE"
    assert by_id["MBP-13"]["status"] == "NOT_AVAILABLE_DOWNGRADE_REQUIRED"
    assert "9-row critic/independent/measured" in by_id["MBP-16"][
        "evidence_or_basis"
    ]
    assert by_id["MBP-17"]["status"] == (
        "COMPLETE_WITH_PROVISIONAL_GENERAL_FIGURES"
    )
    assert "six reproducible figure builders" in by_id["MBP-17"]["evidence_or_basis"]
    assert "seven figures" in by_id["MBP-17"]["evidence_or_basis"]
    assert "12-row error/domain-shift" in by_id["MBP-17"]["evidence_or_basis"]
    assert "35-row selected-outcome claim-evidence" in by_id["MBP-17"][
        "evidence_or_basis"
    ]
    assert audit["manuscript_figures"]["status"] == (
        "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED"
    )
    assert audit["manuscript_figures"]["figure_count"] == 7
    assert len(audit["manuscript_figures"]["builders"]) == 6
    assert len(audit["manuscript_figures"]["focused_tests"]) == 6
    assert audit["manuscript_figures"]["publisher_compliance_claimed"] is False
    assert audit["manuscript_figures"]["new_final_evaluation_read"] is False
    assert audit["baseline_matrix"] == {
        "status": "BASELINE_INVENTORY_MATRIX_RENDERED_DEVELOPMENT_ONLY",
        "row_count": 45,
        "prediction_rows": 34,
        "generation_rows": 11,
        "builder": "scripts/route_a_v3/build_route2_v332_baseline_matrix_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_baseline_matrix_v1.py",
        "table": "docs/paper/route2_v332_baseline_matrix_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_baseline_matrix_v1.json",
        "matrix_is_result_table": False,
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert len(baseline_rows) == baseline_audit["row_count"] == 45
    assert baseline_method["row_count"] == baseline_result["row_count"] == 45
    assert baseline_method["track_counts"] == {"PREDICTION": 34, "GENERATION": 11}
    assert baseline_result["prediction_rows"] == 34
    assert baseline_result["generation_rows"] == 11
    assert baseline_result["prediction_neural_terminal_independent_rows"] == 5
    assert baseline_result["prediction_neural_configured_not_terminal_independent_rows"] == 2
    assert baseline_result["generation_terminal_matched_rows"] == 7
    assert baseline_result["generation_guided_not_authorized_rows"] == 2
    assert baseline_method["matrix_is_result_table"] is baseline_result[
        "matrix_is_result_table"
    ] is baseline_audit["matrix_is_result_table"] is False
    assert baseline_result["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert baseline_result["native_common_arch_three_track_results_table_built"] is True
    assert baseline_result["prediction_generation_matched_budget_numeric_matrix_built"] is True
    assert baseline_result["generation_three_layer_result_table_built"] is True
    assert all(row["development_test_accessed"] == "false" for row in baseline_rows)
    assert all(row["new_final_evaluation_accessed"] == "false" for row in baseline_rows)
    assert all(row["guided_executed"] == "false" for row in baseline_rows)
    assert {
        "spearman", "mae", "ndcg", "uplift", "recovery", "wall_time_seconds",
        "generation_peak_vram_mb", "candidate_count", "nfe",
    }.isdisjoint(baseline_rows[0])
    assert audit["three_track_results_table"]["row_count"] == 52
    assert audit["three_track_results_table"]["reporting_table_complete"] is True
    assert audit["three_track_results_table"]["three_track_benchmark_execution_complete"] is False
    assert len(three_track_rows) == three_track_audit["row_count"] == three_track_method["row_count"] == three_track_result["row_count"] == 52
    assert three_track_method["track_counts"] == {
        "NATIVE_REPRODUCTION": 10,
        "COMMON_SOURCE_RELATIVE_TASK": 12,
        "ARCH_CONTROLLED": 30,
    }
    assert three_track_result["native_numeric_rows"] == 0
    assert three_track_result["common_task_numeric_rows"] == 9
    assert three_track_result["architecture_controlled_numeric_rows"] == 26
    assert three_track_result["headline_horizontal_comparison_eligible_rows"] == 8
    assert three_track_result["reporting_table_complete"] is True
    assert three_track_result["three_track_benchmark_execution_complete"] is False
    assert three_track_result["native_results_enter_current_headline"] is False
    assert three_track_result["common_task_results_compared_only_within_same_task_scope"] is True
    assert three_track_snapshot["aligned_a1_comparison"]["direct_numeric_comparison_allowed"] is False
    native_rows = [row for row in three_track_rows if row["track"] == "NATIVE_REPRODUCTION"]
    assert len(native_rows) == 10
    assert all(row["primary_metric_value"] == row["secondary_metric_value"] == "" for row in native_rows)
    assert all(row["development_test_accessed"] == "false" for row in three_track_rows)
    assert all(row["new_final_evaluation_accessed"] == "false" for row in three_track_rows)
    assert all(row["guided_executed"] == "false" for row in three_track_rows)
    assert audit["a1_true_a2_task_results_table"] == {
        "status": "A1_TRUE_A2_TASK_RESULTS_SEPARATED_TRUE_A2_NUMERIC_NOT_TERMINAL",
        "row_count": 14,
        "a1_numeric_task_rows": 9,
        "a1_development_validation_records": 18293,
        "true_a2_boundary_rows": 5,
        "true_a2_development_listwise_records": 30966,
        "qualified_true_a2_study_credit": 0,
        "true_a2_terminal_numeric_performance_rows": 0,
        "reporting_table_complete": True,
        "cross_estimand_numeric_ranking_allowed": False,
        "builder": "scripts/route_a_v3/build_route2_v332_a1_true_a2_task_results_table_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_a1_true_a2_task_results_table_v1.py",
        "table": "docs/paper/route2_v332_a1_true_a2_task_results_table_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_a1_true_a2_task_results_table_v1.json",
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert audit["matched_budget_baseline_matrix"] == {
        "status": "MATCHED_BUDGET_REPORTING_MATRIX_RENDERED_CONTRACT_COMPLETE_MATCH_NOT_ESTABLISHED",
        "row_count": 14,
        "prediction_rows": 5,
        "generation_rows": 9,
        "critic_v2_exact_within_screen_budget_rows": 4,
        "strongest_same_information_hurdle_update_budget_matched": False,
        "terminal_generation_rows": 7,
        "guided_dependency_no_go_rows": 2,
        "fully_contract_matched_headline_comparison_rows": 0,
        "reporting_matrix_complete": True,
        "matched_budget_benchmark_execution_complete": False,
        "builder": "scripts/route_a_v3/build_route2_v332_matched_budget_baseline_matrix_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_matched_budget_baseline_matrix_v1.py",
        "table": "docs/paper/route2_v332_matched_budget_baseline_matrix_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_matched_budget_baseline_matrix_v1.json",
        "terminal_input_snapshot": "audits/route_a_v3_route2_v332_matched_budget_terminal_input_snapshot_v1.json",
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert audit["generation_three_layer_results_table"] == {
        "status": "GENERATION_EVIDENCE_LAYERS_SEPARATED_NO_GUIDED_OR_BIOLOGICAL_CLAIM",
        "row_count": 9,
        "executed_terminal_method_rows": 7,
        "guided_no_go_boundary_rows": 2,
        "critic_self_score_numeric_rows": 6,
        "independent_evaluator_numeric_rows": 7,
        "measured_candidate_recovery_numeric_rows": 7,
        "closed_measured_ndcg_numeric_rows": 0,
        "three_layer_reporting_complete": True,
        "guided_generation_comparison_complete": False,
        "cross_layer_numeric_ranking_allowed": False,
        "headline_generation_improvement_established": False,
        "builder": "scripts/route_a_v3/build_route2_v332_generation_three_layer_results_table_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_generation_three_layer_results_table_v1.py",
        "table": "docs/paper/route2_v332_generation_three_layer_results_table_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_generation_three_layer_results_table_v1.json",
        "terminal_input_snapshot": "audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json",
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert audit["error_domain_shift_analysis_table"] == {
        "status": "ERROR_AND_DOMAIN_SHIFT_ANALYSIS_REPORTED_NO_CAUSAL_OR_FINAL_CONFIRMATION_CLAIM",
        "row_count": 12,
        "development_task_rows": 9,
        "historical_seed_rows": 3,
        "development_study_count": 7,
        "development_assay_count": 7,
        "within_assay_context_specific_error_metrics_available": False,
        "cross_layer_numeric_pooling_allowed": False,
        "external_transfer_established": False,
        "reporting_table_complete": True,
        "builder": "scripts/route_a_v3/build_route2_v332_error_domain_shift_analysis_table_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_error_domain_shift_analysis_table_v1.py",
        "table": "docs/paper/route2_v332_error_domain_shift_analysis_table_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_error_domain_shift_analysis_table_v1.json",
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert audit["selected_outcome_claim_evidence_table"] == {
        "status": "SELECTED_OUTCOME_CLAIM_EVIDENCE_CLOSED_UNSUPPORTED_CLAIMS_EXPLICIT",
        "selected_final_paper_outcome": "BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER",
        "row_count": 35,
        "draft_claim_marker_count": 22,
        "supported_with_declared_boundary_row_count": 22,
        "unsupported_claim_row_count": 13,
        "unmapped_draft_claim_marker_count": 0,
        "unknown_evidence_id_reference_count": 0,
        "unsupported_claims_allowed_in_manuscript_count": 0,
        "claim_evidence_table_complete": True,
        "minimum_package_complete": False,
        "outcome_trigger_fully_satisfied": False,
        "submission_ready": False,
        "builder": "scripts/route_a_v3/build_route2_v332_selected_outcome_claim_evidence_table_v1.py",
        "focused_test": "tests/route_a_v3/test_build_route2_v332_selected_outcome_claim_evidence_table_v1.py",
        "table": "docs/paper/route2_v332_selected_outcome_claim_evidence_table_v1.csv",
        "audit": "audits/route_a_v3_route2_v332_selected_outcome_claim_evidence_table_v1.json",
        "development_test_read": False,
        "new_final_evaluation_read": False,
        "guided_xeditflow_run": False,
    }
    assert audit["external_evaluation"]["replacement_study_registered"] is False
    assert audit["external_evaluation"]["new_final_evaluation_opened"] is False
    assert audit["guided_generation"]["frozen_critic_xeditflow_run"] is False
    assert audit["guided_generation"]["authorized_action"] == (
        "DO_NOT_RUN_GUIDED_WITH_CURRENT_COHORT"
    )
    assert len(audit["stale_snapshot_findings"]) == 4
    assert "frozen forward manuscript route, not an achieved submission-ready outcome" in draft
    assert "itemwise closure is complete even though the package itself is not" in draft
    assert "minimum benchmark package or submission-ready paper is complete" in draft

    gap = next(
        row
        for row in consistency["known_reporting_gaps"]
        if row["gap_id"] == "GAP-R2-MINIMUM-BENCHMARK-PACKAGE"
    )
    assert gap["status"] == "MINIMUM_BENCHMARK_PACKAGE_NOT_COMPLETE"


def test_independent_evaluator_task_reporting_preserves_heterogeneity() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-EVALUATOR"
    )
    with EVALUATOR_TASK_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == result["task_count"] == 9
    assert sum(int(row["record_count"]) for row in rows) == result[
        "task_record_count_sum"
    ] == 18293
    spearman = [float(row["spearman"]) for row in rows]
    assert sum(value > 0.0 for value in spearman) == result["positive_task_count"] == 5
    assert sum(value <= 0.0 for value in spearman) == result[
        "nonpositive_task_count"
    ] == 4
    assert [min(spearman), max(spearman)] == result["task_spearman_range"]
    worst = max(rows, key=lambda row: float(row["standardized_mae"]))
    assert worst["task_id"] == result["maximum_task_standardized_mae"]["task_id"]
    assert float(worst["standardized_mae"]) == result[
        "maximum_task_standardized_mae"
    ]["value"]
    assert "does not imply uniformly reliable task-level evaluation" in draft


def test_evaluator_global_spread_is_reported_without_post_hoc_gate() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    protocol = _load(EVALUATOR_PROTOCOL)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-EVALUATOR"
    )
    qualification = protocol["independent_evaluator_qualification"]

    assert not any("prediction_std" in key for key in qualification)
    assert result["prediction_spread_is_qualification_threshold"] is False
    assert result["global_prediction_std"] == 1.6834847809909463
    assert result["global_target_std"] == 764.2945302862793
    assert result["global_prediction_std_over_target_std"] == (
        result["global_prediction_std"] / result["global_target_std"]
    )
    gap = next(
        row
        for row in consistency["known_reporting_gaps"]
        if row["gap_id"] == "GAP-R2-EVAL-TASK-SPREAD"
    )
    assert gap["status"] == "PER_TASK_SPREAD_NOT_RECORDED_NO_TERMINAL_RERUN"
    assert "cannot establish that every task underwent mean collapse" in draft


def test_evaluator_qualification_table_locks_exact_production_checks() -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    consistency = _load(CONSISTENCY)
    result = next(
        row
        for row in consistency["results"]
        if row["result_id"] == "R-R2-EVALUATOR"
    )
    with EVALUATOR_CHECK_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {
        "all_expected_tasks_defined",
        "architecture_distinct_from_guiding_critic",
        "cuda_training_verified",
        "development_test_withheld",
        "evaluation_outcomes_closed",
        "exact_frozen_evaluator_identity",
        "full_context_without_pretrained_guide_features",
        "positive_task_breadth_reached",
        "run_completed",
        "task_macro_exceeds_exact_source_permutation",
        "train_only_frozen_validation_stage",
        "train_only_task_robust_target_scaling",
    }
    assert {row["qualification_check_id"] for row in rows} == expected
    assert {row["passed"] for row in rows} == {"true"}
    assert len(rows) == result["qualification_check_count"] == 12
    assert result["all_qualification_checks_passed"] is True
    assert result["candidate_rerun_authorized"] is True
    assert "scientific_claim_status=NOT_ESTABLISHED" in draft
