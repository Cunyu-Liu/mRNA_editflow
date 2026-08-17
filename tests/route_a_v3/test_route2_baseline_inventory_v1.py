from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "configs/route_a_v3_route2_baseline_inventory_v1.json"


def test_inventory_has_at_least_three_implemented_common_task_external_adapters() -> None:
    inventory = json.loads(PATH.read_text(encoding="utf-8"))
    adapters = inventory["prediction_common_task_adapters"]
    assert len(adapters) >= 3
    assert all("IMPLEMENTED" in row["common_task_status"] for row in adapters)
    assert all(row["independent_external_transfer_claim_allowed"] is False for row in adapters)
    assert inventory["evaluation_outcomes_accessed"] is False


def test_unexecuted_external_and_guided_methods_are_not_presented_as_results() -> None:
    inventory = json.loads(PATH.read_text(encoding="utf-8"))
    literature = inventory["prediction_literature_only_or_not_executed"]
    assert {row["model_id"] for row in literature} == {
        "RiNALMo", "mRNABERT", "Orthrus", "APARENT-Perturb"
    }
    assert all(row["status"].startswith("LITERATURE_ONLY") for row in literature)
    generation = {row["method_id"]: row["status"] for row in inventory["generation_methods"]}
    assert generation["frozen_critic_xeditflow"] == "NOT_EXECUTED_REQUIRES_CRITIC_READY_AND_FLOW_G0_READY"
    assert generation["masked_discrete_flow_or_diffusion"] == "LITERATURE_ONLY_TASK_MISMATCH"


def test_generation_inventory_exposes_current_matched_suite_and_retired_selection() -> None:
    inventory = json.loads(PATH.read_text(encoding="utf-8"))
    state = inventory["generation_benchmark_current_state"]
    assert state["status"] == "MATCHED_FORWARD_QUALITY_SUITE_RUNNING"
    assert state["previous_strongest_selection_status"] == (
        "RETIRED_OPEN_SUPPORT_AND_COMPUTE_MISMATCH"
    )
    assert state["matched_quality_runtime_comparison_valid"] is False
    assert state["independent_evaluator_status"] == (
        "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
    )
    assert state["exhaustive_190_cohort_status"] == (
        "CONFIGURED_NOT_EXECUTED_REQUIRES_MATCHED_QUALITY_SUITE_SUCCESS"
    )
    assert state["runtime_valid_successor_status"] == "CONFIGURED_NOT_EXECUTED"
    assert state["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert (ROOT / state["runtime_valid_protocol_path"]).is_file()

    exhaustive = next(
        row for row in inventory["generation_methods"]
        if row["method_id"] == "exhaustive"
    )
    assert exhaustive["status"] == (
        "OUTCOME_BLIND_190_COHORT_SUITE_READY_NOT_EXECUTED_MATCHED_SUITE_PENDING"
    )
    assert exhaustive["source_cohort_count"] == 190
    assert exhaustive["legal_space_size_per_source"] == 151
    assert exhaustive["independent_evaluator_status"] == (
        "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
    )
    assert exhaustive["full_cohort_strongest_selector_eligible"] is False
    assert (ROOT / exhaustive["suite_config_path"]).is_file()
    assert (ROOT / exhaustive["scoring_config_path"]).is_file()


def test_unguided_flow_inventory_records_terminal_g0_evidence_without_biological_claim() -> None:
    inventory = json.loads(PATH.read_text(encoding="utf-8"))
    flow = next(
        row for row in inventory["generation_methods"]
        if row["method_id"] == "unguided_learned_base_flow_g0"
    )
    assert flow["status"] == "COMPLETED_DEVELOPMENT_GPU_FLOW_G0_READY"
    assert flow["training_result_path"].endswith("/base_flow_g0/development_v1/final_summary.json")
    assert flow["validation_result_path"].endswith(
        "/base_flow_g0/development_validation_replay_v1/final_summary.json"
    )
    assert flow["biological_optimization_established"] is False
