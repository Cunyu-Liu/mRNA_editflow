from __future__ import annotations

import math

import pytest

from core.route2_xeditsetflow_edit_coupling_v5 import (
    BASES_V5,
    DEFAULT_STOP_PROBABILITY_V5,
    empirical_edit_prior_v5,
    edit_prior_log_rates_v5,
    prior_scaled_substitution_rates_v5,
    prior_stop_rate_v5,
)


def _pairs() -> list[tuple[str, str]]:
    # Length-4 pairs with edits concentrated at positions 0 and 2.
    return [
        ("ACGU", "GCGU"),   # edits at 0 (A->G)
        ("ACGU", "ACGA"),   # edit at 3 (U->A)
        ("ACGU", "CCGC"),   # edits at 0 (A->C) and 2 (G->C)
    ]


def test_prior_counts_positions_and_bases():
    prior = empirical_edit_prior_v5(_pairs())
    assert prior.sequence_length == 4
    assert prior.total_pairs == 3.0
    assert prior.total_edits == 4.0
    counts = prior.substitution_counts
    # position 0: one A->G, one A->C (+1 smoothing each base)
    assert counts[0][BASES_V5.index("G")] == 2.0
    assert counts[0][BASES_V5.index("C")] == 2.0
    assert counts[0][BASES_V5.index("A")] == 1.0
    assert counts[0][BASES_V5.index("U")] == 1.0
    # position 1: never edited, all smoothing
    assert counts[1] == (1.0, 1.0, 1.0, 1.0)
    # position 3: one U->A
    assert counts[3][BASES_V5.index("A")] == 2.0


def test_scaled_rates_normalize_to_one_minus_stop():
    prior = empirical_edit_prior_v5(_pairs())
    rows = prior_scaled_substitution_rates_v5(prior)
    expected_mass = 1.0 - prior.stop_probability
    for row in rows:
        assert math.isclose(sum(row), expected_mass, abs_tol=1e-9)
        assert all(value > 0.0 for value in row)
    # Edited positions must dominate unedited ones at the same base choice.
    assert rows[0][BASES_V5.index("G")] > rows[1][BASES_V5.index("G")]


def test_log_rates_finite_and_monotone_with_counts():
    prior = empirical_edit_prior_v5(_pairs())
    logs = edit_prior_log_rates_v5(prior)
    assert all(math.isfinite(value) for row in logs for value in row)
    # More counts -> larger log-rate at the same position.
    assert logs[0][BASES_V5.index("G")] > logs[0][BASES_V5.index("A")]


def test_stop_rate_is_frozen_value():
    prior = empirical_edit_prior_v5(_pairs())
    assert prior_stop_rate_v5(prior) == DEFAULT_STOP_PROBABILITY_V5
    custom = empirical_edit_prior_v5(_pairs(), stop_probability=0.25)
    assert prior_stop_rate_v5(custom) == 0.25


def test_invalid_inputs_rejected():
    with pytest.raises(Exception, match="empty"):
        empirical_edit_prior_v5([])
    with pytest.raises(Exception, match="single frozen sequence length"):
        empirical_edit_prior_v5([("ACGU", "ACGU"), ("AC", "AC")])
    with pytest.raises(Exception, match="equal length"):
        empirical_edit_prior_v5([("ACGU", "ACGUA")])
    with pytest.raises(Exception, match=r"inside \(0, 1\)"):
        empirical_edit_prior_v5(_pairs(), stop_probability=1.5)
    with pytest.raises(Exception, match="non-ACGU"):
        empirical_edit_prior_v5([("ACGN", "ACGA")])
