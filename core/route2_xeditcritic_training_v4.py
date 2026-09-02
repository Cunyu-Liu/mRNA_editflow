"""Effective-batch sampler, ranking objectives, and memory policy for Critic V4."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Callable, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch.utils.data import Sampler

from core.route2_xeditcritic_training_data_v3 import (
    XEditCriticRecordV3,
    _StudyCycle,
    capped_sqrt_task_allocations,
    different_source_group_pair_indices,
)
from core.route2_xeditcritic_within_source_ranking_v5 import (
    same_source_group_pair_indices,
)
from core.route2_xeditcritic_cell_offset_v6 import cell_offset_loss_v6


EFFECTIVE_BATCH_V4 = 32
PHYSICAL_BATCH_CANDIDATES_V4 = (4, 8, 16, 32)
PHYSICAL_GPU_SCOPE_V4 = (0, 1, 2, 3, 4, 5)
# V6 family extends the frozen A100 scope to all eight physical cards.  The
# 0–5 scope remains valid so every historical V4/V5 run stays reproducible.
PHYSICAL_GPU_SCOPE_V6 = (0, 1, 2, 3, 4, 5, 6, 7)


class XEditCriticTrainingV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticTrainingV4Error(message)


def require_physical_gpu_scope_v4(
    config: Mapping[str, object], physical_gpu_index: int
) -> None:
    """Bind every Critic V4/V6 CUDA entry point to the frozen A100 GPU scope.

    The historical families are pinned to physical GPUs 0–5; the V6 family is
    admitted across 0–7 so that unused cards can be filled with parallel arms.
    """

    gpu_policy = config.get("gpu_policy")
    _require(isinstance(gpu_policy, Mapping), "Critic V4 GPU policy is absent")
    scope = gpu_policy.get("physical_gpu_scope")
    if scope == list(PHYSICAL_GPU_SCOPE_V6):
        _require(
            physical_gpu_index in PHYSICAL_GPU_SCOPE_V6,
            "Critic V6 GPU is outside the frozen 0–7 scope",
        )
    else:
        _require(
            scope == list(PHYSICAL_GPU_SCOPE_V4),
            "Critic V4 physical GPU scope changed from 0–5",
        )
        _require(
            isinstance(physical_gpu_index, int)
            and not isinstance(physical_gpu_index, bool)
            and physical_gpu_index in PHYSICAL_GPU_SCOPE_V4,
            "Critic V4 GPU is outside 0–5",
        )
    _require(
        gpu_policy.get("cuda_bf16_only") is True
        and gpu_policy.get("cpu_fallback") is False,
        "Critic V4 CUDA/BF16 or CPU-fallback policy changed",
    )


def quantized_sqrt_task_batch_allocations_v4(
    task_sizes: Mapping[str, int],
    *,
    batch_count: int,
    repeat_cap: int = 4,
    effective_batch: int = EFFECTIVE_BATCH_V4,
) -> dict[str, int]:
    """Allocate complete task batches in sqrt proportions within row caps."""

    checked = {str(task): int(size) for task, size in task_sizes.items()}
    _require(bool(checked) and min(checked.values()) > 0, "Critic V4 task sizes are invalid")
    _require(batch_count > 0, "Critic V4 batch count is not positive")
    _require(repeat_cap == 4, "Critic V4 repeat cap drifted from four")
    _require(effective_batch == EFFECTIVE_BATCH_V4, "Critic V4 effective batch drifted from 32")
    capacities = {
        task: (repeat_cap * size) // effective_batch
        for task, size in checked.items()
    }
    _require(min(capacities.values()) >= 1, "Critic V4 task cannot form one capped effective batch")
    _require(sum(capacities.values()) >= batch_count, "Critic V4 complete-batch repeat capacity is insufficient")
    allocations = {task: 0 for task in checked}
    weights = {task: math.sqrt(size) for task, size in checked.items()}
    for _ in range(batch_count):
        eligible = [
            task for task in checked if allocations[task] < capacities[task]
        ]
        _require(bool(eligible), "Critic V4 complete-batch allocation exhausted")
        selected = min(
            eligible,
            key=lambda task: (
                ((allocations[task] + 1) * effective_batch) / weights[task],
                task,
            ),
        )
        allocations[selected] += 1
    _require(sum(allocations.values()) == batch_count, "Critic V4 batch allocation count changed")
    _require(min(allocations.values()) >= 1, "Critic V4 sqrt allocation omitted a task")
    _require(
        all(allocations[task] <= capacities[task] for task in allocations),
        "Critic V4 batch allocation exceeded repeat capacity",
    )
    return allocations


class FixedEffectiveTaskBatchSamplerV4(Sampler[list[int]]):
    """Sqrt-task/source-balanced sampler with exactly 32 rows per update."""

    def __init__(
        self,
        records: Sequence[XEditCriticRecordV3],
        *,
        seed: int,
        repeat_cap: int = 4,
        effective_batch: int = EFFECTIVE_BATCH_V4,
    ) -> None:
        _require(bool(records), "Critic V4 sampler records are empty")
        _require(effective_batch == EFFECTIVE_BATCH_V4, "Critic V4 effective batch drifted from 32")
        _require(repeat_cap == 4, "Critic V4 repeat cap drifted from four")
        self.records = list(records)
        self.seed = int(seed)
        self.repeat_cap = int(repeat_cap)
        self.effective_batch = int(effective_batch)
        self.pass_index = 0
        self.task_sizes = Counter(record.task for record in self.records)
        base_row_allocations = capped_sqrt_task_allocations(
            self.task_sizes,
            draw_count=len(self.records),
            repeat_cap=self.repeat_cap,
        )
        self.update_count = sum(
            math.ceil(allocation / self.effective_batch)
            for allocation in base_row_allocations.values()
        )
        self.task_batch_allocations = quantized_sqrt_task_batch_allocations_v4(
            self.task_sizes,
            batch_count=self.update_count,
            repeat_cap=self.repeat_cap,
            effective_batch=self.effective_batch,
        )

    def set_pass(self, pass_index: int) -> None:
        _require(pass_index >= 0, "Critic V4 pass index is negative")
        self.pass_index = int(pass_index)

    def batches_for_pass(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.pass_index)
        hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
        for index, record in enumerate(self.records):
            hierarchy.setdefault(record.task, {}).setdefault(record.study, {}).setdefault(
                record.source_group, []
            ).append(index)
        completed: list[list[int]] = []
        for task in sorted(hierarchy):
            cycle = _StudyCycle(hierarchy[task], rng, self.repeat_cap)
            draw_count = self.task_batch_allocations[task] * self.effective_batch
            draws = [cycle.draw() for _ in range(draw_count)]
            task_batches = [
                draws[start : start + self.effective_batch]
                for start in range(0, draw_count, self.effective_batch)
            ]
            for batch in task_batches:
                rng.shuffle(batch)
            completed.extend(task_batches)
        rng.shuffle(completed)
        counts = Counter(index for batch in completed for index in batch)
        _require(len(completed) == self.update_count, "Critic V4 updates/pass changed")
        _require(sum(counts.values()) == self.update_count * self.effective_batch, "Critic V4 pass draw count changed")
        _require(max(counts.values()) <= self.repeat_cap, "Critic V4 record repeat cap was exceeded")
        _require(
            all(
                len({self.records[index].task for index in batch}) == 1
                for batch in completed
            ),
            "Critic V4 completed batch mixes tasks",
        )
        return completed

    def __iter__(self):
        yield from self.batches_for_pass()

    def __len__(self) -> int:
        return self.update_count


def physical_microbatch_partitions_v4(
    *,
    effective_batch_size: int,
    physical_batch_size: int,
) -> list[list[int]]:
    """Partition one effective task batch without singleton forwards."""

    _require(effective_batch_size == EFFECTIVE_BATCH_V4, "Critic V4 effective batch must equal 32")
    _require(physical_batch_size in PHYSICAL_BATCH_CANDIDATES_V4, "Critic V4 physical batch is outside {4,8,16,32}")
    _require(effective_batch_size % physical_batch_size == 0, "physical batch does not divide effective batch")
    partitions = [
        list(range(start, start + physical_batch_size))
        for start in range(0, effective_batch_size, physical_batch_size)
    ]
    _require(min(map(len, partitions)) >= 4, "Critic V4 created a singleton or sub-four forward")
    _require([index for part in partitions for index in part] == list(range(EFFECTIVE_BATCH_V4)), "physical partitions drop or repeat a record")
    return partitions


def target_midranks_v4(targets: torch.Tensor) -> torch.Tensor:
    """Exact ascending mid-ranks with tied targets sharing their mean rank."""

    _require(targets.ndim == 1 and targets.numel() >= 2, "mid-rank targets must be a nontrivial vector")
    left = targets[:, None]
    lower = (targets[None, :] < left).sum(dim=1).to(targets.dtype)
    equal = (targets[None, :] == left).sum(dim=1).to(targets.dtype)
    return lower + 0.5 * (equal + 1.0)


def pairwise_sigmoid_soft_ranks_v4(
    predictions: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Differentiable ascending ranks with fixed pairwise-sigmoid temperature."""

    _require(predictions.ndim == 1 and predictions.numel() >= 2, "soft-rank predictions must be a nontrivial vector")
    _require(temperature > 0, "soft-rank temperature must be positive")
    pairwise = (predictions[:, None] - predictions[None, :]) / float(temperature)
    # The diagonal contributes 0.5, so adding 0.5 yields rank one for a value
    # below every other value in the low-temperature limit.
    return 0.5 + torch.sigmoid(pairwise).sum(dim=1)


def lambda_rankic_pairwise_term_v4(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
    *,
    soft_rank_temperature: float = 0.2,
) -> dict[str, torch.Tensor | int]:
    """Rank-displacement weighted pairwise ranking term (LambdaRankIC-style).

    Plain pairwise softplus treats every misordered pair as equally costly,
    which aligns with Kendall tau rather than Spearman rho.  Following the
    Delta-RankIC identity, the damage of swapping two items scales with the
    product of their predicted-rank gap and target-rank gap, so this term
    weights each pair by |soft_rank_i - soft_rank_j| * |midrank_i - midrank_j|.
    The weight is detached: gradients flow only through the softplus margin,
    keeping the LambdaRank interpretation of a swap-importance weight.
    """

    pairs = different_source_group_pair_indices(
        targets,
        source_groups,
        task_ids,
    )
    if not pairs:
        return {"loss": predictions.new_zeros(()), "pair_count": 0, "weight_sum": 0.0}
    left = torch.tensor([pair[0] for pair in pairs], device=predictions.device)
    right = torch.tensor([pair[1] for pair in pairs], device=predictions.device)
    with torch.no_grad():
        soft_ranks = pairwise_sigmoid_soft_ranks_v4(
            predictions.detach(),
            temperature=soft_rank_temperature,
        )
    mid_ranks = target_midranks_v4(targets)
    rank_gap = (soft_ranks[left] - soft_ranks[right]).abs()
    target_gap = (mid_ranks[left] - mid_ranks[right]).abs()
    swap_importance = (rank_gap * target_gap).clamp_min(1.0e-6).detach()
    target_delta = targets[left] - targets[right]
    prediction_delta = predictions[left] - predictions[right]
    per_pair = F.softplus(-target_delta.sign() * prediction_delta)
    loss = (swap_importance * per_pair).sum() / swap_importance.sum()
    _require(bool(torch.isfinite(loss).item()), "lambda pairwise term is nonfinite")
    return {
        "loss": loss,
        "pair_count": len(pairs),
        "weight_sum": float(swap_importance.sum().detach().cpu()),
    }


def soft_spearman_loss_v4(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """One minus Pearson correlation of soft prediction ranks and target mid-ranks."""

    _require(predictions.shape == targets.shape, "soft-Spearman prediction/target geometry differs")
    soft = pairwise_sigmoid_soft_ranks_v4(predictions, temperature=temperature)
    target = target_midranks_v4(targets)
    soft_centered = soft - soft.mean()
    target_centered = target - target.mean()
    denominator = torch.linalg.vector_norm(soft_centered) * torch.linalg.vector_norm(target_centered)
    _require(bool((denominator > 0).item()), "soft-Spearman is undefined for a constant effective batch")
    correlation = (soft_centered * target_centered).sum() / denominator
    return 1.0 - correlation


def critic_v4_loss_weights(pass_number: int) -> dict[str, float]:
    """Return the immutable pass-1..8 V4 loss schedule."""

    _require(1 <= pass_number <= 8, "Critic V4 pass number is outside 1..8")
    if pass_number <= 2:
        return {
            "huber": 1.0,
            "pairwise": 0.25,
            "soft_spearman": 0.0,
            "router_balance": 0.0,
        }
    return {
        "huber": 1.0,
        "pairwise": 0.5,
        "soft_spearman": 0.25,
        "router_balance": 0.01,
    }


@dataclass(frozen=True)
class EffectivePredictionObjectiveV4:
    total_loss: float
    huber_loss: float
    pairwise_loss: float
    soft_spearman_loss: float
    within_source_loss: float
    within_source_pair_count: int
    pair_count: int
    prediction_gradient: torch.Tensor
    lambda_pairwise_loss: float = 0.0
    lambda_pairwise_pair_count: int = 0
    cell_offset_loss: float = 0.0
    cell_offset_gradient: torch.Tensor | None = None


@dataclass(frozen=True)
class ReplayRNGStateV4:
    cpu_state: torch.Tensor
    cuda_state: torch.Tensor | None


@dataclass(frozen=True)
class RetainedEffectiveBatchGraphV4:
    objective_predictions: torch.Tensor
    prediction: torch.Tensor
    router_balance_loss: torch.Tensor
    cell_offset: torch.Tensor | None = None


def capture_replay_rng_state_v4(device: torch.device) -> ReplayRNGStateV4:
    """Capture the stochastic state immediately before one physical forward."""

    cuda_state = None
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA RNG capture requested without CUDA")
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    return ReplayRNGStateV4(
        cpu_state=torch.random.get_rng_state().cpu(),
        cuda_state=cuda_state,
    )


def restore_replay_rng_state_v4(
    state: ReplayRNGStateV4, device: torch.device
) -> None:
    """Restore the exact dropout state for a physical-batch replay."""

    torch.random.set_rng_state(state.cpu_state)
    if device.type == "cuda":
        _require(state.cuda_state is not None, "CUDA replay state is absent")
        torch.cuda.set_rng_state(state.cuda_state, device)
    else:
        _require(state.cuda_state is None, "CPU replay unexpectedly carries CUDA state")


def collect_replayable_predictions_v4(
    physical_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    forward: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, list[ReplayRNGStateV4], list[torch.Tensor]]:
    """Collect detached predictions through the same path used by replay.

    The formal V4 model enables activation-checkpointed blocks only while
    gradients are enabled.  Its collection and replay forwards must therefore
    both run with gradients enabled; otherwise they execute different model
    paths and need not produce bit-identical predictions.  Each collection
    graph is released as soon as its detached prediction has been saved.
    """

    _require(bool(physical_batches), "Critic V4 has no physical batches to replay")
    predictions: list[torch.Tensor] = []
    states: list[ReplayRNGStateV4] = []
    first_pass_predictions: list[torch.Tensor] = []
    for batch in physical_batches:
        batch_size = int(batch["source_tokens"].shape[0]) if "source_tokens" in batch else int(next(iter(batch.values())).shape[0])
        _require(batch_size >= 4, "Critic V4 replay path received a sub-four physical batch")
        states.append(capture_replay_rng_state_v4(device))
        with torch.enable_grad():
            output = forward(batch)
            prediction = output["mean"]
            _require(prediction.shape == (batch_size,), "Critic V4 replay prediction geometry changed")
            _require(torch.isfinite(prediction).all().item(), "Critic V4 replay prediction is nonfinite")
            detached = prediction.detach()
        # Cross-record Huber/ranking/soft-rank arithmetic is deliberately
        # FP32 even though each formal forward runs under BF16 autocast.
        predictions.append(detached.float())
        first_pass_predictions.append(detached.clone())
        del output, prediction
    combined = torch.cat(predictions)
    _require(combined.shape == (EFFECTIVE_BATCH_V4,), "Critic V4 replay did not collect 32 predictions")
    return combined, states, first_pass_predictions


def forward_retained_effective_batch_v4(
    physical_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    forward: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
) -> RetainedEffectiveBatchGraphV4:
    """Keep the single formal batch graph for its later 32-vector VJP."""

    _require(
        len(physical_batches) == 1,
        "Critic V4 retained-graph path requires one physical batch",
    )
    batch = physical_batches[0]
    batch_size = (
        int(batch["source_tokens"].shape[0])
        if "source_tokens" in batch
        else int(next(iter(batch.values())).shape[0])
    )
    _require(
        batch_size == EFFECTIVE_BATCH_V4,
        "Critic V4 retained-graph batch is not the effective batch of 32",
    )
    with torch.enable_grad():
        output = forward(batch)
        prediction = output["mean"]
        router_balance_loss = output["router_balance_loss"]
        cell_offset = output.get("cell_offset")
        _require(
            prediction.shape == (EFFECTIVE_BATCH_V4,),
            "Critic V4 retained prediction geometry changed",
        )
        _require(
            torch.isfinite(prediction).all().item(),
            "Critic V4 retained prediction is nonfinite",
        )
        _require(
            router_balance_loss.numel() == 1
            and torch.isfinite(router_balance_loss).all().item(),
            "Critic V4 retained router balance is invalid",
        )
        if cell_offset is not None:
            _require(
                cell_offset.shape == (EFFECTIVE_BATCH_V4,)
                and torch.isfinite(cell_offset).all().item(),
                "Critic V6 retained cell offset is invalid",
            )
    return RetainedEffectiveBatchGraphV4(
        objective_predictions=prediction.detach().float(),
        prediction=prediction,
        router_balance_loss=router_balance_loss,
        cell_offset=cell_offset,
    )


def backward_retained_effective_batch_v4(
    graph: RetainedEffectiveBatchGraphV4,
    prediction_gradient: torch.Tensor,
    *,
    router_balance_weight: float,
    cell_offset_gradient: torch.Tensor | None = None,
) -> None:
    """Backpropagate the objective through the retained batch-32 graph."""

    _require(
        prediction_gradient.shape == (EFFECTIVE_BATCH_V4,),
        "Critic V4 retained gradient is not length 32",
    )
    _require(
        router_balance_weight >= 0,
        "Critic V4 retained router-balance weight is negative",
    )
    if cell_offset_gradient is not None:
        _require(
            graph.cell_offset is not None
            and cell_offset_gradient.shape == (EFFECTIVE_BATCH_V4,),
            "Critic V6 retained cell-offset gradient is invalid",
        )
        # W1' frozen-backbone mode: auxiliary leaves without a grad path
        # (frozen router / frozen cell-offset head) contribute zero gradient
        # and must be excluded from the multi-tensor backward call.
        leaf_pairs = [
            (leaf, grad)
            for leaf, grad in (
                (graph.cell_offset, cell_offset_gradient),
                (graph.router_balance_loss, graph.prediction.new_tensor(router_balance_weight)),
            )
            if leaf is not None and leaf.requires_grad
        ]
        torch.autograd.backward(
            (graph.prediction, *(leaf for leaf, _ in leaf_pairs)),
            (prediction_gradient, *(grad for _, grad in leaf_pairs)),
        )
    elif router_balance_weight > 0 and (
        graph.router_balance_loss is not None
        and graph.router_balance_loss.requires_grad
    ):
        torch.autograd.backward(
            (graph.prediction, graph.router_balance_loss),
            (
                prediction_gradient,
                graph.prediction.new_tensor(router_balance_weight),
            ),
        )
    else:
        graph.prediction.backward(prediction_gradient)


def backward_replayed_prediction_gradient_v4(
    physical_batches: Sequence[Mapping[str, torch.Tensor]],
    states: Sequence[ReplayRNGStateV4],
    first_pass_predictions: Sequence[torch.Tensor],
    prediction_gradient: torch.Tensor,
    *,
    device: torch.device,
    forward: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    router_balance_weight: float,
) -> list[torch.Tensor]:
    """Replay each >=4 batch and backpropagate the exact 32-vector gradient."""

    _require(len(physical_batches) == len(states) == len(first_pass_predictions), "Critic V4 replay bundles are misaligned")
    _require(prediction_gradient.shape == (EFFECTIVE_BATCH_V4,), "Critic V4 replay gradient is not length 32")
    _require(router_balance_weight >= 0, "Critic V4 router-balance weight is negative")
    cursor = 0
    replayed_predictions: list[torch.Tensor] = []
    for batch, state, expected in zip(
        physical_batches,
        states,
        first_pass_predictions,
        strict=True,
    ):
        restore_replay_rng_state_v4(state, device)
        output = forward(batch)
        prediction = output["mean"]
        _require(torch.equal(prediction.detach(), expected), "Critic V4 RNG replay changed a stochastic prediction")
        end = cursor + prediction.numel()
        if router_balance_weight > 0:
            router_gradient = prediction.new_tensor(
                router_balance_weight / len(physical_batches)
            )
            # Prediction and router balance share the endpoint-router graph in
            # the formal model.  A single multi-output backward preserves that
            # graph and applies both VJPs without retain_graph or a second
            # forward.
            torch.autograd.backward(
                (prediction, output["router_balance_loss"]),
                (prediction_gradient[cursor:end], router_gradient),
            )
        else:
            prediction.backward(prediction_gradient[cursor:end])
        replayed_predictions.append(prediction.detach())
        cursor = end
    _require(cursor == EFFECTIVE_BATCH_V4, "Critic V4 replay gradient did not cover 32 predictions")
    return replayed_predictions


def within_source_ranking_term_v4(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
) -> dict[str, torch.Tensor | int]:
    """Same-source-group softplus ranking term (Direction A, V5).

    Complements the V4 cross-source pairwise term: the evaluated task-macro
    Spearman also contains same-source candidate pairs, and V2 evidence
    showed the candidate-content channel contributes little.  Restricts to
    pairs within one source group with differing targets.
    """

    _require(
        predictions.shape == targets.shape and predictions.ndim == 1,
        "within-source objective vectors are misaligned",
    )
    pairs = same_source_group_pair_indices(targets, source_groups, task_ids)
    if not pairs:
        return {"loss": predictions.new_zeros(()), "pair_count": 0}
    left = torch.tensor([pair[0] for pair in pairs], device=predictions.device)
    right = torch.tensor([pair[1] for pair in pairs], device=predictions.device)
    target_delta = targets[left] - targets[right]
    prediction_delta = predictions[left] - predictions[right]
    loss = F.softplus(-target_delta.sign() * prediction_delta).mean()
    _require(
        bool(torch.isfinite(loss).item()),
        "within-source ranking term is nonfinite",
    )
    return {"loss": loss, "pair_count": len(pairs)}


def effective_prediction_objective_v4(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
    *,
    pass_number: int,
    huber_delta: float = 1.0,
    soft_rank_temperature: float = 0.2,
    within_source_ranking_weight: float = 0.0,
    lambda_pairwise_weight: float = 0.0,
    cell_offset_predictions: torch.Tensor | None = None,
    cell_offset_targets: torch.Tensor | None = None,
    cell_offset_weight: float = 0.0,
) -> EffectivePredictionObjectiveV4:
    """Compute the full task-batch objective and its exact prediction gradient.

    Formal training can first obtain replayable detached predictions for all 32
    rows, compute this 32-vector gradient, then replay physical batches of at
    least four with the recorded RNG states.  This retains the full effective
    pairwise/soft-rank objective without holding 32 records' activations or
    reverting to singleton ranking forwards.

    When the D1 cell-offset auxiliary head is enabled, an extra weighted Huber
    term supervises the per-cell offset (scaled target - scaled pair mean)
    whose detached 32-vector gradient is returned alongside the main one.
    """

    _require(predictions.shape == targets.shape == sample_weights.shape == (EFFECTIVE_BATCH_V4,), "Critic V4 objective requires exactly 32 aligned values")
    _require(len(source_groups) == len(task_ids) == EFFECTIVE_BATCH_V4, "Critic V4 objective metadata is misaligned")
    _require(len(set(task_ids)) == 1, "Critic V4 objective batch is not task homogeneous")
    _require(huber_delta > 0 and bool(torch.all(sample_weights >= 0).item()), "Critic V4 Huber geometry or weights are invalid")
    _require(bool((sample_weights.sum() > 0).item()), "Critic V4 sample weights sum to zero")
    values = predictions.detach().clone().requires_grad_(True)
    weights = critic_v4_loss_weights(pass_number)
    per_record = F.huber_loss(
        values,
        targets,
        reduction="none",
        delta=float(huber_delta),
    )
    huber = (per_record * sample_weights).sum() / sample_weights.sum()
    pairs = different_source_group_pair_indices(
        targets,
        source_groups,
        task_ids,
    )
    _require(bool(pairs), "Critic V4 effective task batch has no legal cross-source-group ranking pair")
    left = torch.tensor([pair[0] for pair in pairs], device=values.device)
    right = torch.tensor([pair[1] for pair in pairs], device=values.device)
    target_delta = targets[left] - targets[right]
    prediction_delta = values[left] - values[right]
    pairwise = F.softplus(-target_delta.sign() * prediction_delta).mean()
    soft = (
        values.new_zeros(())
        if weights["soft_spearman"] == 0.0
        else soft_spearman_loss_v4(
            values,
            targets,
            temperature=soft_rank_temperature,
        )
    )
    within_source = (
        {"loss": values.new_zeros(()), "pair_count": 0}
        if within_source_ranking_weight == 0.0
        else within_source_ranking_term_v4(
            values,
            targets,
            source_groups,
            task_ids,
        )
    )
    lambda_pairwise = (
        {"loss": values.new_zeros(()), "pair_count": 0, "weight_sum": 0.0}
        if lambda_pairwise_weight == 0.0
        else lambda_rankic_pairwise_term_v4(
            values,
            targets,
            source_groups,
            task_ids,
            soft_rank_temperature=soft_rank_temperature,
        )
    )
    cell_offset_gradient: torch.Tensor | None = None
    cell_offset_loss_value = values.new_zeros(())
    if cell_offset_weight > 0.0:
        _require(
            cell_offset_predictions is not None
            and cell_offset_targets is not None
            and cell_offset_predictions.shape == cell_offset_targets.shape == (EFFECTIVE_BATCH_V4,),
            "Critic V6 cell-offset bundle is misaligned",
        )
        offset_result = cell_offset_loss_v6(
            cell_offset_predictions,
            cell_offset_targets,
            sample_weights=sample_weights.detach(),
            delta=float(huber_delta),
        )
        cell_offset_loss_value = float(cell_offset_weight) * offset_result["loss"]
        cell_offset_gradient = float(cell_offset_weight) * offset_result["gradient"]
    total = (
        weights["huber"] * huber
        + weights["pairwise"] * pairwise
        + weights["soft_spearman"] * soft
        + within_source_ranking_weight * within_source["loss"]
        + lambda_pairwise_weight * lambda_pairwise["loss"]
        + cell_offset_loss_value
    )
    _require(torch.isfinite(total).item(), "Critic V4 effective prediction loss is nonfinite")
    gradient = torch.autograd.grad(total, values, create_graph=False)[0]
    _require(torch.isfinite(gradient).all().item(), "Critic V4 prediction gradient is nonfinite")
    return EffectivePredictionObjectiveV4(
        total_loss=float(total.detach().cpu()),
        huber_loss=float(huber.detach().cpu()),
        pairwise_loss=float(pairwise.detach().cpu()),
        soft_spearman_loss=float(soft.detach().cpu()),
        within_source_loss=float(within_source["loss"].detach().cpu()),
        within_source_pair_count=int(within_source["pair_count"]),
        pair_count=len(pairs),
        lambda_pairwise_loss=float(lambda_pairwise["loss"].detach().cpu()),
        lambda_pairwise_pair_count=int(lambda_pairwise["pair_count"]),
        cell_offset_loss=float(cell_offset_loss_value.detach().cpu()),
        cell_offset_gradient=cell_offset_gradient,
        prediction_gradient=gradient.detach(),
    )


def select_physical_batch_from_memory_v4(
    peak_allocated_gib: Mapping[int, float | None],
    *,
    minimum_peak_gib: float | None = None,
    maximum_peak_gib: float = 35.0,
) -> dict[str, float | int | bool | None]:
    """Choose the largest measured batch below the frozen 35-GiB ceiling."""

    _require(set(peak_allocated_gib) == set(PHYSICAL_BATCH_CANDIDATES_V4), "memory preflight did not cover {4,8,16,32}")
    measured = {
        batch: float(value)
        for batch, value in peak_allocated_gib.items()
        if value is not None
    }
    _require(bool(measured), "every Critic V4 physical batch preflight failed")
    _require(
        all(math.isfinite(peak) and peak > 0.0 for peak in measured.values()),
        "Critic V4 memory preflight contains a nonpositive or nonfinite peak",
    )
    _require(4 in measured, "physical batch four failed; model must pause rather than shrink")
    _require(measured[4] <= maximum_peak_gib, "physical batch four exceeds 35 GiB")
    eligible = [batch for batch, peak in measured.items() if peak <= maximum_peak_gib]
    _require(bool(eligible), "no Critic V4 physical batch fits the 35-GiB ceiling")
    selected = max(eligible)
    selected_peak = measured[selected]
    _require(
        minimum_peak_gib is None,
        "Critic V4 lower memory-occupancy gate must remain disabled",
    )
    return {
        "selected_physical_batch": selected,
        "selected_peak_allocated_gib": selected_peak,
        "minimum_peak_allocated_gib": minimum_peak_gib,
        "maximum_peak_allocated_gib": maximum_peak_gib,
        "measurement": "TORCH_CUDA_MAX_MEMORY_ALLOCATED",
        "passed": True,
    }


def critic_v4_optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    head_learning_rate: float = 2e-4,
    semantic_learning_rate: float = 1e-4,
    upper_six_learning_rate: float = 1e-5,
) -> list[dict[str, object]]:
    """Partition every trainable parameter exactly once into the frozen LRs."""

    _require(min(head_learning_rate, semantic_learning_rate, upper_six_learning_rate) > 0, "Critic V4 learning rate is nonpositive")
    _require(hasattr(model, "upper_encoder") and hasattr(model, "router") and hasattr(model, "blocks"), "Critic V4 model lacks optimizer group modules")
    upper = {
        id(parameter): parameter
        for parameter in model.upper_encoder.parameters()
        if parameter.requires_grad
    }
    semantic_modules = [model.router]
    semantic_modules.extend(block.experts for block in model.blocks)
    semantic = {
        id(parameter): parameter
        for module in semantic_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    _require(not (set(upper) & set(semantic)), "upper-six and semantic optimizer groups overlap")
    all_parameters = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    head_ids = set(all_parameters) - set(upper) - set(semantic)
    _require(bool(upper) and bool(semantic) and bool(head_ids), "Critic V4 optimizer group is empty")
    _require(set(all_parameters) == set(upper) | set(semantic) | head_ids, "Critic V4 optimizer groups do not cover every trainable parameter")
    groups = [
        {
            "name": "HEAD_AND_V4_TRUNK",
            "params": [all_parameters[index] for index in sorted(head_ids)],
            "lr": float(head_learning_rate),
        },
        {
            "name": "SEMANTIC_EXPERTS_AND_ROUTER",
            "params": [semantic[index] for index in sorted(semantic)],
            "lr": float(semantic_learning_rate),
        },
        {
            "name": "MRNABERT_TOP_SIX",
            "params": [upper[index] for index in sorted(upper)],
            "lr": float(upper_six_learning_rate),
        },
    ]
    _require(
        sum(parameter.numel() for group in groups for parameter in group["params"])
        == sum(parameter.numel() for parameter in all_parameters.values()),
        "Critic V4 optimizer parameter accounting changed",
    )
    return groups


def critic_v4_learning_rate_factor(
    completed_update_count: int,
    *,
    total_updates: int = 22416,
    warmup_fraction: float = 0.05,
    final_fraction: float = 0.10,
) -> float:
    """Linear 5% warmup followed by cosine decay to 10% of initial LR."""

    _require(total_updates > 0 and 0.0 < warmup_fraction < 1.0, "Critic V4 scheduler geometry is invalid")
    _require(0.0 < final_fraction <= 1.0, "Critic V4 final learning-rate fraction is invalid")
    _require(0 <= completed_update_count <= total_updates, "Critic V4 scheduler update is out of range")
    warmup_updates = math.ceil(total_updates * warmup_fraction)
    if completed_update_count < warmup_updates:
        return (completed_update_count + 1) / warmup_updates
    progress = (completed_update_count - warmup_updates) / max(1, total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_fraction + (1.0 - final_fraction) * cosine
