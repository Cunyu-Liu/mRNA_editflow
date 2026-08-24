from __future__ import annotations

import pytest

from core.route2_xeditsetflow_sampling_v4 import (
    largest_remainder_mode_counts_v4,
    stratified_trajectory_mode_ids_v4,
)


def test_eight_mode_equal_prior_allocates_exactly_four_each() -> None:
    counts = largest_remainder_mode_counts_v4([1 / 8] * 8)
    assert counts == (4,) * 8
    assert sum(counts) == 32


def test_smoothed_skew_prior_keeps_every_mode_and_exact_budget() -> None:
    prior = [0.5625] + [0.0625] * 7
    counts = largest_remainder_mode_counts_v4(prior)
    assert counts[0] > max(counts[1:])
    assert min(counts) >= 1
    assert sum(counts) == 32


def test_mode_ids_put_each_mode_first_then_use_largest_remainder_counts() -> None:
    prior = [0.5625] + [0.0625] * 7
    mode_ids = stratified_trajectory_mode_ids_v4(prior)
    assert mode_ids[:8] == tuple(range(8))
    assert len(mode_ids) == 32
    counts = tuple(mode_ids.count(index) for index in range(8))
    assert counts == largest_remainder_mode_counts_v4(prior)


def test_single_mode_control_assigns_all_32_to_the_only_mode() -> None:
    assert largest_remainder_mode_counts_v4([1.0]) == (32,)
    assert stratified_trajectory_mode_ids_v4([1.0]) == (0,) * 32


def test_invalid_prior_or_insufficient_trajectory_budget_hard_fails() -> None:
    with pytest.raises(Exception):
        largest_remainder_mode_counts_v4([0.4, 0.4])
    with pytest.raises(Exception):
        largest_remainder_mode_counts_v4([0.25] * 4, trajectory_count=3)
