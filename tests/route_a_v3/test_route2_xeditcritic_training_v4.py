from __future__ import annotations

from collections import Counter

import pytest
import torch

from core.route2_xeditcritic_training_data_v3 import XEditCriticRecordV3
from core.route2_xeditcritic_training_v4 import (
    FixedEffectiveTaskBatchSamplerV4,
    XEditCriticTrainingV4Error,
    backward_replayed_prediction_gradient_v4,
    collect_replayable_predictions_v4,
    critic_v4_learning_rate_factor,
    critic_v4_loss_weights,
    critic_v4_optimizer_parameter_groups,
    effective_prediction_objective_v4,
    pairwise_sigmoid_soft_ranks_v4,
    physical_microbatch_partitions_v4,
    quantized_sqrt_task_batch_allocations_v4,
    require_physical_gpu_scope_v4,
    select_physical_batch_from_memory_v4,
    soft_spearman_loss_v4,
    target_midranks_v4,
)


def _gpu_config() -> dict:
    return {
        "gpu_policy": {
            "physical_gpu_scope": [0, 1, 2, 3, 4, 5],
            "cuda_bf16_only": True,
            "cpu_fallback": False,
        }
    }


def test_gpu_scope_is_exactly_physical_zero_through_five() -> None:
    config = _gpu_config()
    for physical_gpu_index in range(6):
        require_physical_gpu_scope_v4(config, physical_gpu_index)

    with pytest.raises(XEditCriticTrainingV4Error, match="outside 0–5"):
        require_physical_gpu_scope_v4(config, 6)
    with pytest.raises(XEditCriticTrainingV4Error, match="outside 0–5"):
        require_physical_gpu_scope_v4(config, True)


def test_gpu_scope_rejects_policy_drift_or_cpu_fallback() -> None:
    expanded = _gpu_config()
    expanded["gpu_policy"]["physical_gpu_scope"].append(6)
    with pytest.raises(XEditCriticTrainingV4Error, match="scope changed"):
        require_physical_gpu_scope_v4(expanded, 0)

    non_bf16 = _gpu_config()
    non_bf16["gpu_policy"]["cuda_bf16_only"] = False
    with pytest.raises(XEditCriticTrainingV4Error, match="CUDA/BF16"):
        require_physical_gpu_scope_v4(non_bf16, 0)

    cpu_fallback = _gpu_config()
    cpu_fallback["gpu_policy"]["cpu_fallback"] = True
    with pytest.raises(XEditCriticTrainingV4Error, match="CPU-fallback"):
        require_physical_gpu_scope_v4(cpu_fallback, 0)


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


def test_fixed_sampler_quantizes_saturated_task_before_drawing_rows() -> None:
    records = [_record(index, "small" if index < 9 else "large") for index in range(202)]
    sampler = FixedEffectiveTaskBatchSamplerV4(records, seed=20260907)
    sampler.set_pass(0)
    batches = sampler.batches_for_pass()
    assert len(batches) == 8
    assert all(len(batch) == 32 for batch in batches)
    assert all(len({records[index].task for index in batch}) == 1 for batch in batches)
    counts = Counter(index for batch in batches for index in batch)
    assert max(counts.values()) <= 4
    assert sampler.task_batch_allocations["small"] == 1


def test_real_geometry_batch_allocation_respects_saturated_task_caps() -> None:
    sizes = {
        "MEAN_RIBOSOME_LOAD::region=0": 2443,
        "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1": 55704,
        "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1": 25710,
        "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1": 204,
        "RNA_HALF_LIFE_MINUTES::region=0": 893,
        "RNA_HALF_LIFE_MINUTES::region=1": 1308,
        "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1": 3318,
    }
    allocations = quantized_sqrt_task_batch_allocations_v4(
        sizes, batch_count=2802
    )
    assert sum(allocations.values()) == 2802
    assert all(allocations[task] * 32 <= 4 * size for task, size in sizes.items())
    assert allocations["PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE::region=1"] == 25
    assert allocations["RNA_HALF_LIFE_MINUTES::region=0"] == 111
    assert allocations["RNA_HALF_LIFE_MINUTES::region=1"] == 163


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


def test_memory_selection_chooses_largest_under_35_without_lower_occupancy_gate() -> None:
    selected = select_physical_batch_from_memory_v4(
        {4: 22.0, 8: 29.0, 16: 34.5, 32: None}
    )
    assert selected["selected_physical_batch"] == 16
    assert selected["selected_peak_allocated_gib"] == 34.5
    low_peak = select_physical_batch_from_memory_v4(
        {4: 8.0, 8: 10.0, 16: 12.0, 32: 15.0}
    )
    assert low_peak["selected_physical_batch"] == 32
    assert low_peak["selected_peak_allocated_gib"] == 15.0
    assert low_peak["minimum_peak_allocated_gib"] is None
    with pytest.raises(XEditCriticTrainingV4Error, match="must remain disabled"):
        select_physical_batch_from_memory_v4(
            {4: 8.0, 8: 10.0, 16: 12.0, 32: 15.0},
            minimum_peak_gib=20.0,
        )
    with pytest.raises(XEditCriticTrainingV4Error, match="batch four exceeds"):
        select_physical_batch_from_memory_v4(
            {4: 36.0, 8: None, 16: None, 32: None}
        )
    with pytest.raises(XEditCriticTrainingV4Error, match="nonpositive or nonfinite"):
        select_physical_batch_from_memory_v4(
            {4: 8.0, 8: 10.0, 16: 12.0, 32: float("nan")}
        )


def test_rng_replay_reproduces_dropout_predictions_and_backpropagates_full_gradient() -> None:
    class DropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, batch):
            hidden = torch.nn.functional.dropout(
                batch["values"], p=0.3, training=True
            )
            shared = self.linear(hidden)
            mean = shared.squeeze(-1)
            return {
                "mean": mean,
                # The formal model's prediction and balance term share the
                # router forward graph.  This catches an invalid sequential
                # backward that would free the graph after prediction.backward.
                "router_balance_loss": shared.square().mean(),
            }

    torch.manual_seed(91)
    model = DropoutModel()
    batches = [
        {
            "source_tokens": torch.zeros((8, 1), dtype=torch.long),
            "values": torch.randn(8, 3),
        }
        for _ in range(4)
    ]
    predictions, states, first = collect_replayable_predictions_v4(
        batches,
        device=torch.device("cpu"),
        forward=model,
    )
    assert predictions.shape == (32,)
    gradient = torch.linspace(-0.5, 0.5, 32)
    replayed = backward_replayed_prediction_gradient_v4(
        batches,
        states,
        first,
        gradient,
        device=torch.device("cpu"),
        forward=model,
        router_balance_weight=0.01,
    )
    assert torch.equal(torch.cat(replayed), predictions)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_rng_replay_uses_the_same_grad_enabled_full_model_path_twice() -> None:
    class GradModeSensitiveDropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.grad_modes: list[bool] = []

        def forward(self, batch):
            grad_enabled = torch.is_grad_enabled()
            self.grad_modes.append(grad_enabled)
            hidden = torch.nn.functional.dropout(
                batch["values"], p=0.25, training=True
            )
            # XEditCriticV4 activates checkpointed attention only on the
            # grad-enabled path.  This offset makes a path mismatch visible in
            # a small unit test without constructing the 170M model.
            path_offset = 0.0 if grad_enabled else 1.0
            mean = hidden.sum(dim=1) * self.scale + path_offset
            return {
                "mean": mean,
                "router_balance_loss": mean.square().mean(),
            }

    torch.manual_seed(137)
    model = GradModeSensitiveDropoutModel()
    batches = [
        {
            "source_tokens": torch.zeros((8, 1), dtype=torch.long),
            "values": torch.randn(8, 3),
        }
        for _ in range(4)
    ]
    predictions, states, first = collect_replayable_predictions_v4(
        batches,
        device=torch.device("cpu"),
        forward=model,
    )
    assert all(model.grad_modes)
    replayed = backward_replayed_prediction_gradient_v4(
        batches,
        states,
        first,
        torch.linspace(-0.5, 0.5, 32),
        device=torch.device("cpu"),
        forward=model,
        router_balance_weight=0.01,
    )

    assert torch.equal(torch.cat(replayed), predictions)
    assert all(model.grad_modes)
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


def test_effective_objective_predictions_are_promoted_from_half_to_float32() -> None:
    class HalfOutputModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float16))

        def forward(self, batch):
            mean = torch.nn.functional.dropout(
                batch["values"], p=0.2, training=True
            ).sum(dim=1) * self.scale
            return {
                "mean": mean,
                "router_balance_loss": mean.sum() * 0.0,
            }

    torch.manual_seed(17)
    batches = [
        {
            "source_tokens": torch.zeros((8, 1), dtype=torch.long),
            "values": torch.randn(8, 3, dtype=torch.float16),
        }
        for _ in range(4)
    ]
    model = HalfOutputModel()
    predictions, states, first = collect_replayable_predictions_v4(
        batches,
        device=torch.device("cpu"),
        forward=model,
    )
    assert predictions.dtype == torch.float32
    assert all(value.dtype == torch.float16 for value in first)
    replayed = backward_replayed_prediction_gradient_v4(
        batches,
        states,
        first,
        torch.ones(32),
        device=torch.device("cpu"),
        forward=model,
        router_balance_weight=0.0,
    )
    assert all(value.dtype == torch.float16 for value in replayed)


def test_optimizer_groups_cover_each_parameter_once_at_the_three_frozen_rates() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upper_encoder = torch.nn.Linear(3, 3)
            self.router = torch.nn.Linear(3, 2)
            block = torch.nn.Module()
            block.experts = torch.nn.Linear(3, 3)
            self.blocks = torch.nn.ModuleList([block])
            self.head = torch.nn.Linear(3, 1)

    model = Model()
    groups = critic_v4_optimizer_parameter_groups(model)
    assert [group["name"] for group in groups] == [
        "HEAD_AND_V4_TRUNK",
        "SEMANTIC_EXPERTS_AND_ROUTER",
        "MRNABERT_TOP_SIX",
    ]
    assert [group["lr"] for group in groups] == [2e-4, 1e-4, 1e-5]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}


def test_scheduler_uses_ceil_five_percent_warmup_and_ends_at_ten_percent() -> None:
    warmup_updates = 1121
    assert critic_v4_learning_rate_factor(0) == pytest.approx(1 / warmup_updates)
    assert critic_v4_learning_rate_factor(warmup_updates - 1) == pytest.approx(1.0)
    assert critic_v4_learning_rate_factor(warmup_updates) == pytest.approx(1.0)
    assert critic_v4_learning_rate_factor(22416) == pytest.approx(0.1)
    backward_replayed_prediction_gradient_v4,
    collect_replayable_predictions_v4,
