from __future__ import annotations

import pytest

from core.route2_closed_neighborhood_v3 import (
    closed_neighborhood_metrics_v1,
    exact_order_invariant_terminal_probability_v3,
)
from core.route2_legal_xeditflow import exact_terminal_distribution, initial_state


def _unit_rates(_state, actions):
    return {action: 1.0 for action in actions}


def test_exact_terminal_probability_sums_all_edit_permutations() -> None:
    result = exact_order_invariant_terminal_probability_v3(
        "AC", "GU", edit_budget=3, assay_id="a", context_id="c",
        rate_function=_unit_rates,
    )
    # Two direct orders: (1/7) * (1/4); after both source positions are edited,
    # NO_LEGAL_ACTION is structural and no synthetic STOP is added.
    assert result["permutation_path_count"] == 2
    assert result["terminal_probability"] == pytest.approx(1.0 / 14.0)
    assert result["terminal_causes"] == ["NO_LEGAL_ACTION"]
    root = initial_state("AC", budget=3, assay_id="a", context_id="c")
    distribution = exact_terminal_distribution(root, _unit_rates)
    exact_mass = sum(
        probability
        for state, probability in distribution.items()
        if state.current_sequence == "GU"
    )
    assert result["terminal_probability"] == pytest.approx(exact_mass)


def test_closed_probability_distinguishes_stop_and_budget_exhaustion() -> None:
    identity = exact_order_invariant_terminal_probability_v3(
        "AC", "AC", edit_budget=1, assay_id="a", context_id="c",
        rate_function=_unit_rates,
    )
    assert identity["permutation_path_count"] == 1
    assert identity["terminal_causes"] == ["EXPLICIT_STOP"]
    assert identity["terminal_probability"] == pytest.approx(1.0 / 7.0)
    exhausted = exact_order_invariant_terminal_probability_v3(
        "AC", "GC", edit_budget=1, assay_id="a", context_id="c",
        rate_function=_unit_rates,
    )
    assert exhausted["terminal_causes"] == ["BUDGET_EXHAUSTED"]
    assert exhausted["terminal_probability"] == pytest.approx(1.0 / 7.0)


def test_closed_probability_rejects_more_than_five_edits() -> None:
    with pytest.raises(Exception, match="five-edit"):
        exact_order_invariant_terminal_probability_v3(
            "AAAAAA", "CCCCCC", edit_budget=5, assay_id="a", context_id="c",
            rate_function=_unit_rates,
        )


def test_closed_metrics_are_source_macro_and_keep_undefined_sources() -> None:
    rows = [
        {"source_key": "a", "candidate_sequence": "A", "terminal_probability": 0.8, "measured_direction_normalized_delta": 2.0},
        {"source_key": "a", "candidate_sequence": "C", "terminal_probability": 0.2, "measured_direction_normalized_delta": 0.0},
        {"source_key": "b", "candidate_sequence": "G", "terminal_probability": 0.5, "measured_direction_normalized_delta": 1.0},
    ]
    result = closed_neighborhood_metrics_v1(rows)
    assert result["source_count"] == 2
    assert result["defined_source_count"] == 1
    assert result["undefined_source_count"] == 1
    assert result["source_macro_ndcg"] == 1.0
    assert result["source_macro_normalized_regret"] == 0.0
    assert result["source_macro_top_1_recall"] == 1.0
    assert result["per_source"]["b"]["ndcg"] is None
    assert result["undefined_sources_are_not_filled_with_zero"] is True
