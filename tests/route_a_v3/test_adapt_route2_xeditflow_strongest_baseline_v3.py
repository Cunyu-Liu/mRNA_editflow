from __future__ import annotations

from scripts.route_a_v3.adapt_route2_xeditflow_strongest_baseline_v3 import (
    adapt_strongest_baseline_v3,
)


def test_strongest_adapter_does_not_reuse_historical_open_ndcg_as_closed() -> None:
    strongest = {
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
        "forward_equivalent_budget_per_source": 320,
        "guiding_checkpoint_path": "/critic.pt",
        "independent_evaluator_checkpoint_path": "/evaluator.pt",
    }
    selection = {
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "evaluation_release_state": "CLOSED",
        "baseline_evaluations": [{
            "method_id": "genetic",
            "evaluation": {
                "generation": {
                    "method_id": "genetic", "source_count": 2,
                    "source_macro_unique_candidate_rate": 0.9, "hard_legality_rate": 1.0,
                    "edit_budget_violation_count": 0, "candidate_budget_violation_count": 0,
                },
                "measured_neighborhood": {
                    "source_macro_candidate_recovery_rate": 0.3,
                    "source_macro_measured_top_k_recovery_at_k": 0.2,
                    "source_macro_closed_measured_ndcg_at_k": None,
                },
            },
        }],
    }
    result = adapt_strongest_baseline_v3(strongest, selection, base_flow_training_seed=20260905)
    assert result["generation"]["method_id"] == "strongest_matched_baseline"
    assert result["generation"]["base_flow_training_seed"] == 20260905
    assert result["open"]["historical_open_ndcg_used_as_new_closed_ndcg"] is False
    assert "closed_source_macro_ndcg" not in result["open"]
    assert result["guiding_checkpoint_path"] == "/critic.pt"
