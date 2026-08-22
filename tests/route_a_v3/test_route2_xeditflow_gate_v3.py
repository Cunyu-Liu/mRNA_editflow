from __future__ import annotations

import pytest

from core.route2_xeditflow_gate_v3 import (
    GUIDANCE_GRID_V3,
    adjudicate_guidance_screen_v3,
    adjudicate_guided_three_seed_v3,
    authorize_xeditflow_guidance_v3,
)


def test_guidance_requires_both_full_critic_readiness_and_setflow_confirmation() -> None:
    critic = {
        "status": "CRITIC_READY_FOR_GUIDANCE",
        "frozen_test_passed": True,
        "all_development_refit_complete": True,
        "loso_readiness_passed": True,
    }
    flow = {
        "status": "XEDITSETFLOW_V3_CONFIRMATION_PASS",
        "flow_status": "FLOW_G0_READY",
    }
    assert authorize_xeditflow_guidance_v3(critic, flow)["guidance_authorized"] is True
    critic["loso_readiness_passed"] = False
    result = authorize_xeditflow_guidance_v3(critic, flow)
    assert result["guidance_authorized"] is False
    assert result["new_final_evaluation_authorized"] is False


def _screen_results():
    return {
        combination: {
            "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_COMBINATION_COMPLETE",
            "base_flow_training_seed": 20260904,
            "combination": list(combination),
            "closed_source_macro_ndcg": 0.5,
            "closed_source_macro_normalized_regret": 0.4,
            "independent_evaluator_paired_margin": 0.1,
            "open_source_macro_candidate_recovery": 0.3,
            "total_forward_equivalents": 300,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for combination in GUIDANCE_GRID_V3
    }


def test_guidance_screen_is_exact_18_grid_and_uses_frozen_lexicographic_order() -> None:
    results = _screen_results()
    winner = (0.5, 1.0, 2.0)
    results[winner]["closed_source_macro_ndcg"] = 0.6
    result = adjudicate_guidance_screen_v3(results)
    assert result["combination_count"] == 18
    assert (result["selected_kappa"], result["selected_temperature"], result["selected_beta_max"]) == winner
    results.pop(winner)
    with pytest.raises(Exception, match="exactly the 18"):
        adjudicate_guidance_screen_v3(results)


def _method(ndcg, regret, top1=0.5):
    return {
        "closed_source_macro_ndcg": ndcg,
        "closed_source_macro_normalized_regret": regret,
        "closed_source_macro_top_1_recall": top1,
    }


def _guided_payloads():
    payloads = {}
    for seed in (20260904, 20260905, 20260906):
        full = {
            **_method(0.72, 0.30, 0.6),
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
        }
        payloads[seed] = {
            "methods": {
                "full_soft_value_smc": full,
                "unguided_setflow": _method(0.64, 0.40),
                "first_order_guidance": _method(0.62, 0.42),
                "simple_rate_guidance": _method(0.63, 0.41),
                "generate_then_rerank": _method(0.65, 0.39),
                "strongest_matched_baseline": _method(0.64, 0.40),
            },
            "source_paired_ndcg_improvement_ci_95": {
                "over_unguided": [0.02, 0.14],
                "over_strongest_baseline": [0.02, 0.14],
            },
            "source_paired_independent_evaluator_margin_ci_95": [0.02, 0.20],
            "critic_self_score_increased": True,
            "all_methods_matched_compute_ceiling_met": True,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    return payloads


def test_guided_three_seed_gate_requires_model_and_nonselfscore_evidence() -> None:
    result = adjudicate_guided_three_seed_v3(_guided_payloads())
    assert result["status"] == "XEDITFLOW_V3_PASS"
    assert result["new_final_evaluation_authorized"] is True
    assert result["submission_ready"] is False
    failed = _guided_payloads()
    failed[20260905]["methods"]["full_soft_value_smc"]["closed_source_macro_ndcg"] = 0.65
    result = adjudicate_guided_three_seed_v3(failed)
    assert result["status"] == "XEDITFLOW_V3_NO_GO"
    assert result["reward_exploitation"] is True
    assert result["new_final_evaluation_authorized"] is False
