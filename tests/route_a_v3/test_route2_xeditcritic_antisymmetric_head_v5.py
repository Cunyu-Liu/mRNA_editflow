from __future__ import annotations

import torch
from torch import nn

from core.route2_xeditcritic_antisymmetric_head_v5 import (
    AntisymmetricSiameseHeadV5,
    _swap_paired_batch,
    antisymmetry_residual_v5,
    identity_zero_residual_v5,
)


class TinyScoreModule(nn.Module):
    """Positional score over one-hot-ish token sums (deterministic)."""

    def forward(self, batch):
        source = batch["source_tokens"].float()
        candidate = batch["candidate_tokens"].float()
        weight = torch.linspace(1.0, 2.0, source.shape[1], device=source.device)
        return (candidate * weight).sum(dim=1) - (source * weight * 0.5).sum(dim=1)


def _batch(batch_size: int = 4, length: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260829)
    source = torch.randint(0, 4, (batch_size, length), generator=generator)
    candidate = torch.randint(0, 4, (batch_size, length), generator=generator)
    return {
        "source_tokens": source,
        "candidate_tokens": candidate,
        "region_ids": torch.zeros(batch_size, dtype=torch.long),
    }


def test_swap_exchanges_source_and_candidate_and_keeps_others():
    batch = _batch()
    swapped = _swap_paired_batch(batch)
    assert torch.equal(swapped["source_tokens"], batch["candidate_tokens"])
    assert torch.equal(swapped["candidate_tokens"], batch["source_tokens"])
    assert torch.equal(swapped["region_ids"], batch["region_ids"])


def test_head_is_exactly_antisymmetric():
    head = AntisymmetricSiameseHeadV5(TinyScoreModule())
    batch = _batch()
    assert antisymmetry_residual_v5(head, batch) == 0.0


def test_head_is_identity_zero_on_self_pairs():
    head = AntisymmetricSiameseHeadV5(TinyScoreModule())
    batch = _batch()
    assert identity_zero_residual_v5(head, batch) == 0.0


def test_head_delta_changes_when_roles_flip():
    head = AntisymmetricSiameseHeadV5(TinyScoreModule())
    batch = _batch()
    forward_delta = head(batch)
    reverse_delta = head(_swap_paired_batch(batch))
    assert torch.allclose(forward_delta, -reverse_delta, atol=1e-6)
    assert not torch.allclose(forward_delta, torch.zeros_like(forward_delta))


def test_swap_rejects_missing_counterpart():
    with __import__("pytest").raises(Exception, match="counterpart"):
        _swap_paired_batch({"source_tokens": torch.zeros(1, 2), "region_ids": torch.zeros(1)})


def test_head_rejects_unpaired_batch():
    head = AntisymmetricSiameseHeadV5(TinyScoreModule())
    with __import__("pytest").raises(Exception, match="source/candidate"):
        head({"region_ids": torch.zeros(1)})
