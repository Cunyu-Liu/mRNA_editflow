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
    assert len(evidence_ids) == len(set(evidence_ids)) == 17

    claims = re.findall(r"\[claim:([^\]]+)\]", draft)
    assert len(claims) == len(set(claims)) == 20

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
    assert preflight["source_locations_checked"] == len(evidence["sources"]) == 17
    assert (
        preflight["local_or_contract_locations_checked"]
        + preflight["a100_mnt_locations_checked"]
        == preflight["source_locations_checked"]
    )
    assert preflight["missing_locations"] == 0
    assert preflight["check_scope"] == "FILE_EXISTENCE_ONLY_NO_EVIDENCE_CONTENT_OPENED"
    assert preflight["human_content_verification_completed"] is False
    assert preflight["submission_readiness_changed"] is False
    assert evidence["human_verification_required"] is True
    assert evidence["submission_ready"] is False


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
