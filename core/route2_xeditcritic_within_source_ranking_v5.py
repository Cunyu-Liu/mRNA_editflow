"""Within-source-group ranking objective for the Critic V5 hypothesis.

Direction A of the 2026-08-29 architecture review: the V4 effective objective
ranks only *different* source-group pairs (``different_source_group_pair_indices``),
so the critic is never trained to discriminate candidates of the *same* source.
The evaluated task-macro Spearman is a pooled within-task ranking that also
contains same-source pairs, and V2 evidence showed the candidate-content
channel contributes little beyond edit metadata.  This module adds the missing
within-source-group pairwise ranking term as a pure, CPU-testable function.
It changes no frozen V4 artifact; the V5 family wires it into a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


class XEditCriticWithinSourceRankingV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticWithinSourceRankingV5Error(message)


@dataclass(frozen=True)
class WithinSourceRankingLossV5:
    """Mirror of the V4 objective fields for the within-source term."""

    total_loss: float
    pair_count: int
    group_count: int


def same_source_group_pair_indices(
    targets: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
) -> list[tuple[int, int]]:
    """Deterministically enumerate legal ranking pairs inside one source group.

    Complement of ``different_source_group_pair_indices``: every pair
    ``(i, j)`` with ``i < j``, the same source group, and differing targets.
    Pairs are ordered first by source group insertion order, then by index.
    """

    _require(targets.ndim == 1, "within-source ranking targets must be a vector")
    _require(
        len(targets) == len(source_groups) == len(task_ids),
        "within-source ranking bundle is misaligned",
    )
    _require(len(set(task_ids)) == 1, "ranking batch is not task homogeneous")
    group_order: dict[str, list[int]] = {}
    for index, group in enumerate(source_groups):
        group_order.setdefault(group, []).append(index)
    pairs: list[tuple[int, int]] = []
    for group in sorted(group_order):
        members = group_order[group]
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                if bool((targets[left] != targets[right]).item()):
                    pairs.append((left, right))
    _require(
        all(source_groups[left] == source_groups[right] for left, right in pairs),
        "within-source ranking pair crossed source groups",
    )
    return pairs


def within_source_ranking_loss_v5(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    source_groups: Sequence[str],
    task_ids: Sequence[str],
    *,
    target_weighted: bool = False,
) -> WithinSourceRankingLossV5:
    """Softplus pairwise ranking loss restricted to same-source-group pairs.

    Matches the V4 cross-source formulation (``softplus(-sign(dy) * dp)``) so
    the two terms share a scale; ``target_weighted=True`` weights each pair by
    the normalized absolute target gap, emphasising larger within-source
    effect differences.
    """

    _require(
        predictions.shape == targets.shape and predictions.ndim == 1,
        "within-source ranking predictions/targets must be aligned vectors",
    )
    pairs = same_source_group_pair_indices(targets, source_groups, task_ids)
    if not pairs:
        return WithinSourceRankingLossV5(
            total_loss=0.0,
            pair_count=0,
            group_count=0,
        )
    left = torch.tensor([pair[0] for pair in pairs], device=predictions.device)
    right = torch.tensor([pair[1] for pair in pairs], device=predictions.device)
    target_delta = targets[left] - targets[right]
    prediction_delta = predictions[left] - predictions[right]
    per_pair = F.softplus(-target_delta.sign() * prediction_delta)
    if target_weighted:
        gap = target_delta.abs()
        weights = gap / gap.sum().clamp_min(1e-12)
        loss = (per_pair * weights).sum()
    else:
        loss = per_pair.mean()
    _require(bool(torch.isfinite(loss).item()), "within-source ranking loss is nonfinite")
    groups_with_pairs = {
        source_groups[left_index] for left_index, _ in pairs
    }
    return WithinSourceRankingLossV5(
        total_loss=float(loss.detach().cpu()),
        pair_count=len(pairs),
        group_count=len(groups_with_pairs),
    )
