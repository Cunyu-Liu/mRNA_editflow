from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "audits" / name).read_text(encoding="utf-8"))


def test_generation_readiness_does_not_reinstate_retired_v1_selection() -> None:
    summary = _load("route_a_v3_route2_generation_readiness_summary_v1.json")
    selection = summary["generation_baseline_selection"]
    assert summary["g0_validation"]["status"] == "FLOW_G0_READY"
    assert selection["status"] == "NOT_FROZEN"
    assert selection["strongest_generation_baseline_id"] is None
    assert selection["old_v1_selection_artifact_state"] == "RETIRED_NOT_AUTHORITATIVE_FOR_SELECTION"
    assert selection["matched_forward_equivalent_budget"] is False
    assert selection["independent_evaluator_established"] is False
    assert summary["readiness"]["guided_unlocked"] is False


def test_open_support_replay_covers_all_required_methods_without_closed_ndcg() -> None:
    audit = _load("route_a_v3_route2_generation_evaluator_repair_audit_v1.json")
    methods = audit["methods"]
    assert len(methods) == 7
    assert {row["method_id"] for row in methods} == {
        "random_legal",
        "greedy",
        "beam",
        "genetic",
        "local_search",
        "generate_then_rerank",
        "unguided_learned_base_flow_g0",
    }
    assert all(row["source_closed_measured_ndcg_defined_count"] == 0 for row in methods)
    assert audit["strongest_generation_baseline_status"] == "NOT_FROZEN_IN_OPEN_GENERATED_SUPPORT"


def test_compute_audit_records_frozen_budget_mismatch() -> None:
    audit = _load("route_a_v3_route2_generation_compute_matching_audit_v1.json")
    assert audit["status"] == "EXISTING_GENERATION_BASELINE_COMPARISON_NOT_MATCHED_COMPUTE"
    assert audit["frozen_registry_critic_forward_equivalent_budget_per_source"] == 256
    assert audit["observed_search_critic_forward_budget_per_source"] == 32
    assert audit["selector_repair"]["matched_forward_equivalent_budget_is_required"] is True
    assert audit["scientific_claim_status"] == "NOT_ESTABLISHED"
