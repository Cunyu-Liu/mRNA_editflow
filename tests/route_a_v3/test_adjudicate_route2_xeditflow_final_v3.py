from __future__ import annotations

import json

from scripts.route_a_v3.adjudicate_route2_xeditflow_final_v3 import (
    METHODS,
    adjudicate_final_manifest_v3,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metric(method):
    basic = {
        "closed_source_macro_ndcg": 0.72 if method == "full_soft_value_smc" else 0.64,
        "closed_source_macro_normalized_regret": 0.30 if method == "full_soft_value_smc" else 0.40,
        "closed_source_macro_top_1_recall": 0.60 if method == "full_soft_value_smc" else 0.50,
    }
    if method == "full_soft_value_smc":
        basic.update({
            "open_source_macro_candidate_recovery": 0.26,
            "open_source_macro_top_k_recovery": 0.16,
            "open_source_macro_unique_candidate_rate": 0.91,
            "independent_evaluator_margin_over_strongest_baseline": 0.12,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "trajectory_replay_failure_count": 0,
            "numerical_failure_count": 0,
            "maximum_forward_equivalents_per_source": 320,
        })
    return basic


def test_final_adjudicator_requires_exact_three_seed_six_method_evidence(tmp_path) -> None:
    seeds = []
    for seed in (20260904, 20260905, 20260906):
        methods = {}
        for method in METHODS:
            path = tmp_path / f"{seed}-{method}.json"
            _write(path, {
                "status": "XEDITFLOW_V3_MATCHED_METHOD_METRICS_COMPLETE",
                "method_role": method,
                "base_flow_training_seed": seed,
                "metrics": _metric(method),
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
            })
            methods[method] = str(path)
        bootstrap = tmp_path / f"{seed}-bootstrap.json"
        _write(bootstrap, {
            "status": "XEDITFLOW_V3_SOURCE_PAIRED_BOOTSTRAP_COMPLETE",
            "analysis_unit": "SOURCE",
            "base_flow_training_seed": seed,
            "bootstrap_iterations": 10_000,
            "closed_source_count": 3,
            "defined_closed_source_count": 2,
            "closed_method_source_support_exactly_matched": True,
            "undefined_closed_sources_filled_with_zero": False,
            "source_paired_ndcg_improvement_ci_95": {
                "over_unguided": [0.02, 0.14],
                "over_strongest_baseline": [0.02, 0.14],
            },
            "source_paired_independent_evaluator_margin_ci_95": [0.02, 0.20],
            "critic_self_score_increased": True,
            "all_methods_matched_compute_ceiling_met": True,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        })
        seeds.append({
            "base_flow_training_seed": seed,
            "methods": methods,
            "paired_bootstrap_path": str(bootstrap),
        })
    result = adjudicate_final_manifest_v3({
        "schema_version": "route_a_v3_route2_xeditflow_final_comparison_manifest.v1",
        "status": "XEDITFLOW_V3_FINAL_COMPARISON_RESULTS_COMPLETE",
        "guidance_screen_status": "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN",
        "seeds": seeds,
    })
    assert result["gate"]["status"] == "XEDITFLOW_V3_PASS"
    assert result["new_final_evaluation_authorized"] is True
    assert result["submission_ready"] is False
