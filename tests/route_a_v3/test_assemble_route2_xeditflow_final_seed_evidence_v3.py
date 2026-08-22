from __future__ import annotations

from scripts.route_a_v3.assemble_route2_xeditflow_final_seed_evidence_v3 import (
    METHODS,
    assemble_final_seed_evidence_v3,
)


def _closed(method, value):
    return {
        "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
        "method_id": method,
        "source_macro_ndcg": value,
        "source_macro_normalized_regret": 0.4,
        "source_macro_top_1_recall": 0.5,
        "undefined_sources_are_not_filled_with_zero": True,
        "per_source": {
            "a": {"status": "DEFINED", "ndcg": value},
            "b": {"status": "DEFINED", "ndcg": value + 0.01},
            "u": {"status": "UNDEFINED_ZERO_MEASURED_GAIN", "ndcg": None},
        },
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _open(method):
    return {
        "status": "XEDITFLOW_V3_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": method,
        "source_macro_candidate_recovery": 0.3,
        "source_macro_top_k_recovery": 0.2,
        "source_macro_unique_candidate_rate": 0.95,
        "hard_legality_rate": 1.0,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def _generation(method):
    return {
        "status": (
            "XEDITFLOW_V3_SMC_GENERATION_COMPLETE"
            if method == "full_soft_value_smc"
            else "XEDITFLOW_V3_STRONGEST_BASELINE_ADAPTER_COMPLETE"
            if method == "strongest_matched_baseline"
            else "XEDITFLOW_V3_MATCHED_CONTROL_GENERATION_COMPLETE"
        ),
        "base_flow_training_seed": 20260904,
        "maximum_forward_equivalents_per_source": 100,
        "edit_budget_violation_count": 0,
        "candidate_budget_violation_count": 0,
        "trajectory_replay_failure_count": 0,
        "numerical_failure_count": 0,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_final_seed_evidence_pairs_sources_without_zero_filling() -> None:
    values = {method: 0.6 for method in METHODS}
    values["full_soft_value_smc"] = 0.8
    values["strongest_matched_baseline"] = 0.65
    evidence = {
        method: {
            "closed": _closed(method, values[method]),
            "open": _open(method),
            "generation": _generation(method),
        }
        for method in METHODS
    }
    evaluator = {
        "status": "XEDITFLOW_V3_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE",
        "analysis_unit": "SOURCE",
        "paired_margin_over_strongest_baseline": 0.2,
        "per_source_paired_margin": {"a": 0.1, "b": 0.3},
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    payload = assemble_final_seed_evidence_v3(
        evidence,
        base_flow_training_seed=20260904,
        full_independent_evaluator=evaluator,
        full_candidate_rows=[
            {"source_key": "a", "critic_self_score": 2.0},
            {"source_key": "b", "critic_self_score": 2.0},
        ],
        unguided_candidate_rows=[
            {"source_key": "a", "critic_self_score": 1.0},
            {"source_key": "b", "critic_self_score": 1.0},
        ],
        bootstrap_iterations=10_000,
        bootstrap_seed=20261001,
    )
    bootstrap = payload["paired_bootstrap"]
    assert bootstrap["source_paired_ndcg_improvement_ci_95"]["over_unguided"][0] > 0
    assert bootstrap["source_paired_ndcg_improvement_ci_95"]["over_strongest_baseline"][0] > 0
    assert bootstrap["source_paired_independent_evaluator_margin_ci_95"][0] > 0
    assert bootstrap["critic_self_score_increased"] is True
    assert bootstrap["undefined_closed_sources_filled_with_zero"] is False
    assert set(payload["method_results"]) == METHODS
