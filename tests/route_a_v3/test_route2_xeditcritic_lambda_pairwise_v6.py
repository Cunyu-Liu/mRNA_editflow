from __future__ import annotations

import pytest
import torch

from core.route2_xeditcritic_training_v4 import (
    effective_prediction_objective_v4,
    lambda_rankic_pairwise_term_v4,
    pairwise_sigmoid_soft_ranks_v4,
    target_midranks_v4,
)


def _effective_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    generator = torch.Generator().manual_seed(7)
    predictions = torch.randn(32, generator=generator)
    targets = torch.randn(32, generator=generator)
    sample_weights = torch.ones(32)
    source_groups = [f"SG{i:03d}" for i in range(32)]
    task_ids = ["TASK_A::region=0"] * 32
    return predictions, targets, sample_weights, source_groups, task_ids


def test_lambda_term_defaults_to_bit_identical_v5_behavior():
    predictions, targets, weights, groups, tasks = _effective_batch()
    reference = effective_prediction_objective_v4(
        predictions, targets, weights, groups, tasks, pass_number=5
    )
    disabled = effective_prediction_objective_v4(
        predictions,
        targets,
        weights,
        groups,
        tasks,
        pass_number=5,
        lambda_pairwise_weight=0.0,
    )
    assert disabled.total_loss == reference.total_loss
    assert torch.equal(disabled.prediction_gradient, reference.prediction_gradient)
    assert disabled.lambda_pairwise_loss == 0.0
    assert disabled.lambda_pairwise_pair_count == 0


def test_lambda_term_enters_total_loss_at_exactly_its_weight():
    predictions, targets, weights, groups, tasks = _effective_batch()
    reference = effective_prediction_objective_v4(
        predictions, targets, weights, groups, tasks, pass_number=5
    )
    enabled = effective_prediction_objective_v4(
        predictions,
        targets,
        weights,
        groups,
        tasks,
        pass_number=5,
        lambda_pairwise_weight=0.7,
    )
    assert enabled.lambda_pairwise_pair_count == 16  # greedy disjoint pairing of 32 rows
    extra = enabled.total_loss - reference.total_loss
    assert extra == pytest.approx(0.7 * enabled.lambda_pairwise_loss, abs=1e-5)


def test_lambda_weight_scales_with_rank_displacement():
    # Greedy disjoint pairing yields (0,1) and (2,3).  Both pairs are misordered.
    # Pair (0,1) has a large predicted-rank gap (~3) while (2,3) has a small one (~1),
    # so the Delta-RankIC swap importance of (0,1) must dominate.
    predictions = torch.tensor([5.0, -5.0, 0.6, 0.4])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = ["A", "B", "C", "D"]
    tasks = ["T::region=0"] * 4
    term = lambda_rankic_pairwise_term_v4(predictions, targets, groups, tasks)
    assert term["pair_count"] == 2
    soft_ranks = pairwise_sigmoid_soft_ranks_v4(predictions, temperature=0.2)
    mid_ranks = target_midranks_v4(targets)
    big = abs(float(soft_ranks[0] - soft_ranks[1])) * abs(float(mid_ranks[0] - mid_ranks[1]))
    small = abs(float(soft_ranks[2] - soft_ranks[3])) * abs(float(mid_ranks[2] - mid_ranks[3]))
    assert big > 2.0 * small


def test_lambda_gradients_push_misordered_pairs_back_with_displacement_scaling():
    predictions = torch.tensor([5.0, -5.0, 0.6, 0.4])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = ["A", "B", "C", "D"]
    tasks = ["T::region=0"] * 4
    values = predictions.clone().requires_grad_(True)
    term = lambda_rankic_pairwise_term_v4(values, targets, groups, tasks)
    term["loss"].backward()
    gradient = values.grad
    # Both pairs are misordered (left row truly smaller yet scored higher): gradient
    # descent must lower the left prediction and raise the right one.
    assert gradient[0] > 0 and gradient[1] < 0
    assert gradient[2] > 0 and gradient[3] < 0
    # The large-displacement misordering receives a far stronger correction.
    assert abs(float(gradient[0])) > 4.0 * abs(float(gradient[2]))


def test_lambda_term_stays_finite_under_extreme_scores():
    generator = torch.Generator().manual_seed(11)
    predictions = torch.randn(32, generator=generator) * 1.0e3
    targets = torch.randn(32, generator=generator)
    weights = torch.ones(32)
    groups = [f"S{i:02d}" for i in range(32)]
    tasks = ["T::region=1"] * 32
    result = effective_prediction_objective_v4(
        predictions,
        targets,
        weights,
        groups,
        tasks,
        pass_number=5,
        lambda_pairwise_weight=0.7,
    )
    assert torch.isfinite(result.prediction_gradient).all()
