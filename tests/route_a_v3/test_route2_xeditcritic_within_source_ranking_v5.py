from __future__ import annotations

import pytest
import torch

from core.route2_xeditcritic_within_source_ranking_v5 import (
    same_source_group_pair_indices,
    within_source_ranking_loss_v5,
)


def _bundle() -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    targets = torch.tensor([1.0, 2.0, 2.0, 0.5, 3.0, 1.5])
    predictions = torch.tensor([0.9, 1.8, 2.1, 0.4, 2.5, 1.4])
    source_groups = ["s1", "s1", "s1", "s2", "s2", "s3"]
    task_ids = ["TASK"] * 6
    return targets, predictions, source_groups, task_ids


def test_same_source_pairs_enumerate_only_within_group_and_skip_equal_targets():
    targets, _, groups, tasks = _bundle()
    pairs = same_source_group_pair_indices(targets, groups, tasks)
    # s1: (0,1) [1.0 vs 2.0], (0,2) [1.0 vs 2.0], (1,2) equal target skipped
    # s2: (3,4) [0.5 vs 3.0]; s3 has a single member (no pair)
    assert pairs == [(0, 1), (0, 2), (3, 4)]


def test_same_source_pairs_reject_cross_group_and_misaligned_input():
    targets, _, groups, tasks = _bundle()
    for left, right in same_source_group_pair_indices(targets, groups, tasks):
        assert groups[left] == groups[right]
    with pytest.raises(Exception, match="misaligned"):
        same_source_group_pair_indices(targets, groups, tasks[:-1])
    with pytest.raises(Exception, match="task homogeneous"):
        same_source_group_pair_indices(
            targets, groups, ["A", "A", "A", "A", "A", "B"]
        )


def test_perfect_ranking_gives_small_loss_and_reversed_gives_large():
    targets, predictions, groups, tasks = _bundle()
    good = within_source_ranking_loss_v5(predictions, targets, groups, tasks)
    reversed_predictions = -predictions
    bad = within_source_ranking_loss_v5(
        reversed_predictions, targets, groups, tasks
    )
    assert bad.total_loss > good.total_loss
    assert good.pair_count == 3
    assert good.group_count == 2


def test_zero_loss_when_predictions_order_matches_targets_within_groups():
    targets = torch.tensor([1.0, 2.0, 0.5, 3.0])
    predictions = torch.tensor([0.0, 10.0, 0.0, 10.0])
    groups = ["s1", "s1", "s2", "s2"]
    tasks = ["T"] * 4
    loss = within_source_ranking_loss_v5(predictions, targets, groups, tasks)
    assert loss.total_loss <= 1e-4


def test_no_pairs_returns_zero_loss():
    targets = torch.tensor([1.0, 2.0])
    predictions = torch.tensor([0.0, 0.0])
    loss = within_source_ranking_loss_v5(
        predictions, targets, ["s1", "s2"], ["T", "T"]
    )
    assert loss.total_loss == 0.0
    assert loss.pair_count == 0


def test_target_weighted_mode_still_finite_and_pair_count_preserved():
    targets, predictions, groups, tasks = _bundle()
    plain = within_source_ranking_loss_v5(predictions, targets, groups, tasks)
    weighted = within_source_ranking_loss_v5(
        predictions, targets, groups, tasks, target_weighted=True
    )
    assert weighted.pair_count == plain.pair_count
    assert weighted.total_loss >= 0.0
