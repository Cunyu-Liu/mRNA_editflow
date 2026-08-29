"""Hard antisymmetric siamese readout for the Critic V5 hypothesis.

Direction C of the 2026-08-29 architecture review: the delta estimand
satisfies ``dy(s, c) = -dy(c, s)`` and ``dy(s, s) = 0`` by definition, but
the V4 critic only carries a *soft* raw-antisymmetric branch (one of six
readout branches).  DDMut-style siamese networks enforce antisymmetry as a
hard parameterization and were reported to stabilize both directions of
mutation-effect prediction.  This module provides the V5 head:

``delta = g(s, c) - g(c, s)``

for any score module ``g`` that consumes a paired batch, guaranteeing exact
antisymmetry and exact identity-zero (up to float rounding) without changing
the trunk.  CPU-testable with a tiny linear ``g``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import nn


class XEditCriticAntisymmetricReadoutV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticAntisymmetricReadoutV5Error(message)


def _swap_paired_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Swap the source/candidate roles of a paired critic batch.

    Tensors whose name starts with ``source_`` exchange with the matching
    ``candidate_`` tensor and vice versa; other tensors pass through
    unchanged.  Batch size and geometry are preserved.
    """

    swapped: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            swapped[key] = value
            continue
        if key.startswith("source_"):
            counterpart = "candidate_" + key[len("source_") :]
            _require(
                counterpart in batch,
                f"antisymmetric swap lacks counterpart for {key}",
            )
            swapped[key] = batch[counterpart]
        elif key.startswith("candidate_"):
            counterpart = "source_" + key[len("candidate_") :]
            _require(
                counterpart in batch,
                f"antisymmetric swap lacks counterpart for {key}",
            )
            swapped[key] = batch[counterpart]
        else:
            swapped[key] = value
    return swapped


class AntisymmetricSiameseHeadV5(nn.Module):
    """Wraps a score module ``g`` into an exactly antisymmetric delta head."""

    def __init__(self, score_module: nn.Module) -> None:
        super().__init__()
        self.score_module = score_module

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        _require("source_tokens" in batch and "candidate_tokens" in batch, "paired batch lacks source/candidate tokens")
        forward_score = self.score_module(batch)
        _require(
            forward_score.shape[0] == batch["source_tokens"].shape[0],
            "score module batch size disagrees with the paired batch",
        )
        reverse_score = self.score_module(_swap_paired_batch(batch))
        delta = forward_score - reverse_score
        _require(
            bool(torch.isfinite(delta).all().item()),
            "antisymmetric delta is nonfinite",
        )
        return delta


def antisymmetry_residual_v5(
    head: AntisymmetricSiameseHeadV5, batch: Mapping[str, torch.Tensor]
) -> float:
    """Max |f(s,c) + f(c,s)| over the batch; 0.0 means exact antisymmetry."""

    forward_delta = head(batch)
    reverse_delta = head(_swap_paired_batch(batch))
    residual = (forward_delta + reverse_delta).abs().max()
    return float(residual.detach().cpu())


def identity_zero_residual_v5(
    head: AntisymmetricSiameseHeadV5, batch: Mapping[str, torch.Tensor]
) -> float:
    """Max |f(s,s)| for a self-paired batch; 0.0 means exact identity-zero."""

    self_batch = {
        key: (batch["source_" + key[len("candidate_") :]] if key.startswith("candidate_") else value)
        for key, value in batch.items()
    }
    delta = head(self_batch)
    return float(delta.abs().max().detach().cpu())
