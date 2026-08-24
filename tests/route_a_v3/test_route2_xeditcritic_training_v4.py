from __future__ import annotations

from collections import Counter

import pytest
import torch

from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3
from core.route2_xeditcritic_training_v4 import (
    FixedEffectiveTaskBatchSamplerV4,
    XEditCriticTrainingV4Error,
    critic_v4_loss_weights,
    effective_prediction_objective_v4,
    pairwise_sigmoid_soft_ranks_v4,
    physical_microbatch_partitions_v4,
    select_physical_batch_from_memory_v4,
    soft_spearman_loss_v4,
    target_midranks_v4,
)


def _record(index: int, task: str) -> XEditCriticRecordV3:
    source = "AAAA"
    candidate = "ACAA"
    return XEditCriticRecordV3(
        record_id=f"record-{index:03d}",
        split="TRAIN",
        source=source,
        candidate=candidate,
        edits=((1, "A", "C"),),
        target=float(index % 13),
        task=task,
        study=f"study-{index % 3}",
        source_group=f"group-{index % 11}",
        assay="assay",
        context="context",
        region=index % 2,
        quantity="quantity",
        measurement="measurement",
        numerator="none",
        denominator="none",
    )


def test_fixed_sampler_emits_task_homogeneous_batches_of_exactly_32_with_cap_four() -> None:
    records = [_record(index, "task-a" if index < 70 else "task-b") for index in range(100)]
    sampler = FixedEffectiveTaskBatchSamplerV4(records, seed=20260907)
    sampler.set_pass(3)
    batches = sampler.batches_for_pass()
    assert batches and all(len(batch) == 32 for batch in batches)
    assert all(len({records[index].task for index in batch}) == 1 for batch in batches)
    counts = Counter(index for batch in batches for index in batch)
    assert max(counts.values()) <= 4


def test_physical_partitions_never_create_singleton_or_sub_four_forwards() -> None:
    for physical in (4, 8, 16, 32):
        partitions = physical_microbatch_partitions_v4(
            effective_batch_size=32,
            physical_batch_size=physical,
        )
        assert min(map(len, partitions)) >= 4
        assert [index for part in partitions for index in part] == list(range(32))
    with pytest.raises(XEditCriticTrainingV4Error, match="outside"):
        physical_microbatch_partitions_v4(
            effective_batch_size=32,
            physical_batch_size=1,
        )


def test_target_ties_receive_exact_mid_ranks() -> None:
    targets = torch.tensor([3.0, 1.0, 1.0, 5.0])
    assert target_midranks_v4(targets).tolist() == [3.0, 1.5, 1.5, 4.0]


def test_soft_rank_and_soft_spearman_have_correct_direction_and_finite_gradient() -> None:
    targets = torch.arange(32, dtype=torch.float32)
    predictions = targets.clone().requires_grad_(True)
    ranks = pairwise_sigmoid_soft_ranks_v4(predictions, temperature=0.2)
    assert torch.all(ranks[1:] > ranks[:-1])
    aligned = soft_spearman_loss_v4(predictions, targets, temperature=0.2)
    reversed_loss = soft_spearman_loss_v4(-predictions, targets, temperature=0.2)
    assert aligned.item() < 0.01
    assert reversed_loss.item() > 1.99
    aligned.backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_loss_schedule_is_exactly_frozen_across_all_eight_passes() -> None:
    assert all(
        critic_v4_loss_weights(pass_number)
        == {"huber": 1.0, "pairwise": 0.25, "soft_spearman": 0.0, "router_balance": 0.0}
        for pass_number in (1, 2)
    )
    assert all(
        critic_v4_loss_weights(pass_number)
        == {"huber": 1.0, "pairwise": 0.5, "soft_spearman": 0.25, "router_balance": 0.01}
        for pass_number in range(3, 9)
    )


def test_effective_objective_uses_cross_group_pairs_and_returns_exact_32_gradient() -> None:
    predictions = torch.linspace(-1, 1, 32)
    targets = torch.linspace(-2, 2, 32)
    groups = [f"group-{index % 8}" for index in range(32)]
    objective = effective_prediction_objective_v4(
        predictions,
        targets,
        torch.ones(32),
        groups,
        ["one-task"] * 32,
        pass_number=3,
    )
    assert objective.pair_count > 0
    assert objective.prediction_gradient.shape == (32,)
    assert torch.isfinite(objective.prediction_gradient).all()
    assert objective.soft_spearman_loss < 0.01


def test_effective_objective_rejects_same_group_or_mixed_task_batches() -> None:
    values = torch.arange(32, dtype=torch.float32)
    with pytest.raises(XEditCriticTrainingV4Error, match="no legal"):
        effective_prediction_objective_v4(
            values,
            values,
            torch.ones(32),
            ["same-group"] * 32,
            ["task"] * 32,
            pass_number=1,
        )
    with pytest.raises(XEditCriticTrainingV4Error, match="not task homogeneous"):
        effective_prediction_objective_v4(
            values,
            values,
            torch.ones(32),
            [f"group-{index}" for index in range(32)],
            ["task-a"] * 16 + ["task-b"] * 16,
            pass_number=1,
        )


def test_memory_selection_chooses_largest_under_35_and_requires_20_to_35_target() -> None:
    selected = select_physical_batch_from_memory_v4(
        {4: 22.0, 8: 29.0, 16: 34.5, 32: None}
    )
    assert selected["selected_physical_batch"] == 16
    assert selected["selected_peak_allocated_gib"] == 34.5
    with pytest.raises(XEditCriticTrainingV4Error, match="below 20"):
        select_physical_batch_from_memory_v4(
            {4: 8.0, 8: 10.0, 16: 12.0, 32: 15.0}
        )
    with pytest.raises(XEditCriticTrainingV4Error, match="batch four exceeds"):
        select_physical_batch_from_memory_v4(
            {4: 36.0, 8: None, 16: None, 32: None}
        )
