from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/select_route2_strongest_generation_baseline_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("select_route2_generation_baseline_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation(method_id: str, ndcg: float, regret: float, total_forwards: float):
    per_source_generation = {}
    per_source_measured = {}
    for index in range(4):
        source_key = f"S{index}"
        per_source_generation[source_key] = {
            "candidate_budget": 8,
            "compute": {
                "critic_forward_budget": 64,
                "generator_nfe": 0,
                "critic_forwards": total_forwards,
                "independent_evaluator_forwards": 0,
                "total_forward_equivalents": total_forwards,
            },
        }
        per_source_measured[source_key] = {
            "closed_measured_ndcg_at_k": ndcg,
            "normalized_regret": regret,
        }
    return {
        "schema_version": "route_a_v3_route2_generation_evaluation.v2",
        "evaluation_release_state": "CLOSED",
        "measured_neighborhood_pool": "DEVELOPMENT",
        "generation": {
            "method_id": method_id,
            "source_count": 4,
            "candidate_count": 32,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "generated_candidates_grant_canonical_credit": False,
            "per_source": per_source_generation,
        },
        "measured_neighborhood": {
            "candidate_support_mode": "CLOSED_MEASURED_SUPPORT",
            "unknown_generated_candidates_are_zero_gain": False,
            "source_count": 4,
            "source_closed_measured_ndcg_defined_count": 4,
            "source_normalized_regret_defined_count": 4,
            "source_macro_candidate_recovery_rate": 0.5,
            "source_macro_measured_top_k_recovery_at_k": 0.5,
            "source_macro_closed_measured_ndcg_at_k": ndcg,
            "source_macro_normalized_regret": regret,
            "per_source": per_source_measured,
        },
    }


def _payload():
    return {
        "schema_version": "route_a_v3_route2_generation_baseline_selection_input.v2",
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "selection_evidence_mode": "CLOSED_MEASURED_SUPPORT",
        "evaluation_release_state": "CLOSED",
        "bootstrap_iterations": 1000,
        "bootstrap_seed": 17,
        "forward_equivalent_budget_per_source": 64,
        "required_method_ids": ["random_legal", "beam"],
        "baseline_evaluations": [
            {"method_id": "random_legal", "evaluation": _evaluation("random_legal", 0.5, 0.5, 10.0)},
            {"method_id": "beam", "evaluation": _evaluation("beam", 0.8, 0.2, 20.0)},
        ],
    }


def test_measured_ndcg_freezes_strongest_matched_baseline() -> None:
    module = _load()
    result = module.select(_payload())
    assert result["strongest_generation_baseline_id"] == "beam"
    assert result["selection_evidence_mode"] == "CLOSED_MEASURED_SUPPORT"
    assert result["matched_source_and_candidate_budget"] is True
    assert result["matched_forward_equivalent_budget"] is True
    assert result["forward_equivalent_budget_per_source"] == 64
    assert result["critic_budget_matched_within_critic_using_methods"] is True
    assert result["evaluation_outcomes_accessed"] is False


def test_budget_mismatch_refuses_unfair_selection() -> None:
    module = _load()
    payload = deepcopy(_payload())
    payload["baseline_evaluations"][1]["evaluation"]["generation"]["per_source"]["S0"]["compute"]["critic_forward_budget"] = 32
    with pytest.raises(module.GenerationBaselineSelectionError, match="critic budget"):
        module.select(payload)


def test_missing_matched_forward_budget_refuses_selection() -> None:
    module = _load()
    payload = _payload()
    del payload["forward_equivalent_budget_per_source"]
    with pytest.raises(module.GenerationBaselineSelectionError, match="budget is missing"):
        module.select(payload)


def test_forward_budget_overrun_refuses_selection() -> None:
    module = _load()
    payload = _payload()
    compute = payload["baseline_evaluations"][1]["evaluation"]["generation"]["per_source"]["S0"]["compute"]
    compute["critic_forwards"] = 65
    compute["total_forward_equivalents"] = 65
    with pytest.raises(module.GenerationBaselineSelectionError, match="budget exceeded"):
        module.select(payload)


def test_forward_accounting_must_close() -> None:
    module = _load()
    payload = _payload()
    payload["baseline_evaluations"][1]["evaluation"]["generation"]["per_source"]["S0"]["compute"][
        "total_forward_equivalents"
    ] += 1
    with pytest.raises(module.GenerationBaselineSelectionError, match="accounting does not close"):
        module.select(payload)


def test_unguided_method_is_allowed_only_with_zero_critic_calls() -> None:
    module = _load()
    payload = _payload()
    payload["required_method_ids"].append("unguided_learned_base_flow_g0")
    evaluation = _evaluation("unguided_learned_base_flow_g0", 0.6, 0.4, 8.0)
    for row in evaluation["generation"]["per_source"].values():
        row["compute"]["critic_forward_budget"] = None
        row["compute"]["critic_forwards"] = 0
        row["compute"]["generator_nfe"] = row["compute"]["total_forward_equivalents"]
    payload["baseline_evaluations"].append({
        "method_id": "unguided_learned_base_flow_g0",
        "evaluation": evaluation,
    })
    result = module.select(payload)
    rows = {row["method_id"]: row for row in result["all_candidates_ranked"]}
    assert rows["unguided_learned_base_flow_g0"]["critic_budget_class"] == "NO_CRITIC_CALLS"
    evaluation["generation"]["per_source"]["S0"]["compute"]["critic_forwards"] = 1
    with pytest.raises(module.GenerationBaselineSelectionError, match="unbudgeted critic"):
        module.select(payload)


def test_illegal_method_cannot_be_selected() -> None:
    module = _load()
    payload = deepcopy(_payload())
    payload["baseline_evaluations"][1]["evaluation"]["generation"]["hard_legality_rate"] = 0.99
    with pytest.raises(module.GenerationBaselineSelectionError, match="illegal candidates"):
        module.select(payload)


def test_uncertainty_equivalent_point_leader_yields_to_faster_method() -> None:
    module = _load()
    payload = _payload()
    beam = payload["baseline_evaluations"][1]["evaluation"]
    values = [1.0, 1.0, 0.1, 0.1]
    for source_key, value in zip(sorted(beam["measured_neighborhood"]["per_source"]), values):
        beam["measured_neighborhood"]["per_source"][source_key]["closed_measured_ndcg_at_k"] = value
    beam["measured_neighborhood"]["source_macro_closed_measured_ndcg_at_k"] = sum(values) / len(values)
    result = module.select(payload)
    assert result["point_leader_method_id"] == "beam"
    assert result["strongest_generation_baseline_id"] == "random_legal"
    assert result["bootstrap_uncertainty_equivalent_method_ids"] == ["beam", "random_legal"]


def test_generation_selector_requires_real_bootstrap_budget() -> None:
    module = _load()
    payload = _payload()
    payload["bootstrap_iterations"] = 999
    with pytest.raises(module.GenerationBaselineSelectionError, match="below 1000"):
        module.select(payload)


def test_open_generated_support_requires_independent_evaluator() -> None:
    module = _load()
    payload = _payload()
    measured = payload["baseline_evaluations"][1]["evaluation"]["measured_neighborhood"]
    measured["candidate_support_mode"] = "OPEN_GENERATED_SUPPORT"
    measured["source_closed_measured_ndcg_defined_count"] = 0
    measured["source_macro_closed_measured_ndcg_at_k"] = None
    for row in measured["per_source"].values():
        row["closed_measured_ndcg_at_k"] = None
    with pytest.raises(module.GenerationBaselineSelectionError, match="independent evaluator required"):
        module.select(payload)


def _make_independent_open_support(payload) -> None:
    payload["selection_evidence_mode"] = "INDEPENDENT_EVALUATOR_OPEN_SUPPORT"
    for entry in payload["baseline_evaluations"]:
        evaluation = entry["evaluation"]
        method_id = entry["method_id"]
        uplift = 0.2 if method_id == "beam" else 0.1
        measured = evaluation["measured_neighborhood"]
        measured["candidate_support_mode"] = "OPEN_GENERATED_SUPPORT"
        measured["source_closed_measured_ndcg_defined_count"] = 0
        measured["source_macro_closed_measured_ndcg_at_k"] = None
        for row in measured["per_source"].values():
            row["closed_measured_ndcg_at_k"] = None
        for row in evaluation["generation"]["per_source"].values():
            row["compute"]["critic_forwards"] -= 1
            row["compute"]["independent_evaluator_forwards"] = 1
            row["independent_evaluator_score"] = {
                "count": 8,
                "max_uplift_over_source": uplift,
            }
        entry["independent_evaluator_summary"] = {
            "schema_version": "route_a_v3_route2_independent_generation_evaluator.v1",
            "status": "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE",
            "source_count": 4,
            "candidate_row_count": 32,
            "independent_evaluator_forward_count": 36,
            "evaluator_checkpoint_path": "/mnt/evaluator.pt",
            "guiding_checkpoint_path": "/mnt/guide.pt",
            "evaluator_result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
            "evaluator_frozen_before_candidate_generation": True,
            "guiding_checkpoint_distinct": True,
            "evaluation_outcomes_used_to_select_evaluator": 0,
            "device": "cuda:6",
            "cpu_fallback_used": False,
        }


def test_open_support_can_freeze_an_independent_evaluator_only_baseline() -> None:
    module = _load()
    payload = _payload()
    _make_independent_open_support(payload)
    result = module.select(payload)
    assert result["strongest_generation_baseline_id"] == "beam"
    assert result["selection_evidence_mode"] == "INDEPENDENT_EVALUATOR_OPEN_SUPPORT"
    assert result["independent_evaluator_checkpoint_path"] == "/mnt/evaluator.pt"
    assert result["scientific_claim_status"] == "INDEPENDENT_EVALUATOR_ONLY_MEASURED_OUTCOME_NOT_ESTABLISHED"


def test_open_support_rejects_development_test_evaluator_exposure() -> None:
    module = _load()
    payload = _payload()
    _make_independent_open_support(payload)
    payload["baseline_evaluations"][0]["independent_evaluator_summary"][
        "evaluator_result_stage"
    ] = "FROZEN_DEVELOPMENT_TEST"
    with pytest.raises(module.GenerationBaselineSelectionError, match="training exposure"):
        module.select(payload)
