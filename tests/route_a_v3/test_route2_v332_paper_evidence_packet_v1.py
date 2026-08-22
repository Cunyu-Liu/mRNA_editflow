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
    assert len(evidence_ids) == len(set(evidence_ids)) == 31

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
    assert preflight["source_locations_checked"] == len(evidence["sources"]) == 31
    assert (
        preflight["local_or_contract_locations_checked"]
        + preflight["a100_mnt_locations_checked"]
        == preflight["source_locations_checked"]
    )
    assert preflight["missing_locations"] == 0
    assert preflight["local_or_contract_locations_checked"] == 19
    assert preflight["a100_mnt_locations_checked"] == 12
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
    assert by_id["E-R2-BASELINE-MATRIX-BUILDER"]["location"] == (
        "scripts/route_a_v3/build_route2_v332_baseline_matrix_v1.py"
    )
    assert by_id["E-R2-BASELINE-MATRIX-AUDIT"]["location"] == (
        "audits/route_a_v3_route2_v332_baseline_matrix_v1.json"
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


def test_provisional_figure_method_preserves_protected_outcome_boundary() -> None:
    consistency = _load(CONSISTENCY)
    method = next(
        row for row in consistency["methods"] if row["method_id"] == "M-R2-FIGURES"
    )

    assert method["status"] == "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED"
    assert method["figure_count"] == 5
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
    with MINIMUM_PACKAGE_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with BASELINE_MATRIX_TABLE.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
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

    assert [row["requirement_id"] for row in rows] == [
        f"MBP-{index:02d}" for index in range(1, 19)
    ]
    assert sum(row["status"].startswith("COMPLETE") for row in rows) == 14
    assert sum(row["status"].startswith("PARTIAL") for row in rows) == 3
    assert sum(row["status"].startswith("NOT_AVAILABLE") for row in rows) == 1
    assert audit["summary"]["requirement_count"] == result["requirement_count"] == 18
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
    assert audit["summary"]["current_final_paper_outcome_frozen"] is False
    assert result["final_paper_outcome_frozen"] is False

    by_id = {row["requirement_id"]: row for row in rows}
    assert by_id["MBP-10"]["status"] == "PARTIAL_GUIDED_NOT_AUTHORIZED"
    assert by_id["MBP-11"]["status"] == "COMPLETE_HISTORICAL_NEGATIVE"
    assert by_id["MBP-13"]["status"] == "NOT_AVAILABLE_DOWNGRADE_REQUIRED"
    assert by_id["MBP-17"]["status"] == (
        "COMPLETE_WITH_PROVISIONAL_GENERAL_FIGURES"
    )
    assert audit["manuscript_figures"]["status"] == (
        "PROVISIONAL_GENERAL_MANUSCRIPT_FIGURES_RENDERED"
    )
    assert audit["manuscript_figures"]["figure_count"] == 5
    assert len(audit["manuscript_figures"]["builders"]) == 4
    assert len(audit["manuscript_figures"]["focused_tests"]) == 4
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
    assert baseline_result["native_common_arch_three_track_results_table_built"] is False
    assert baseline_result["prediction_generation_matched_budget_numeric_matrix_built"] is False
    assert all(row["development_test_accessed"] == "false" for row in baseline_rows)
    assert all(row["new_final_evaluation_accessed"] == "false" for row in baseline_rows)
    assert all(row["guided_executed"] == "false" for row in baseline_rows)
    assert {
        "spearman", "mae", "ndcg", "uplift", "recovery", "wall_time_seconds",
        "generation_peak_vram_mb", "candidate_count", "nfe",
    }.isdisjoint(baseline_rows[0])
    assert audit["external_evaluation"]["replacement_study_registered"] is False
    assert audit["external_evaluation"]["new_final_evaluation_opened"] is False
    assert audit["guided_generation"]["frozen_critic_xeditflow_run"] is False
    assert audit["guided_generation"]["authorized_action"] == (
        "DO_NOT_RUN_GUIDED_WITH_CURRENT_COHORT"
    )
    assert len(audit["stale_snapshot_findings"]) == 4
    assert "conditional target route, not a frozen submission-ready outcome" in draft
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
