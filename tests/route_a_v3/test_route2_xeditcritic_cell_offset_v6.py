"""Focused unit tests for Critic V6 W1-b cell-offset auxiliary head."""

from __future__ import annotations

import pytest
import torch

from core.route2_xeditcritic_cell_offset_v6 import (
    CellOffsetHeadV6,
    HEK293FT_CONTEXT,
    cell_offset_discrimination_v6,
    cell_offset_loss_v6,
)


def test_cell_offset_forward_shapes():
    head = CellOffsetHeadV6(condition_width=8, hidden_width=16)
    condition = torch.zeros((6, 8))
    offset = head(condition)
    assert offset.shape == (6,)
    assert torch.isfinite(offset).all()


def test_cell_offset_forward_uses_condition():
    head = CellOffsetHeadV6(condition_width=8, hidden_width=16)
    with torch.no_grad():
        condition_a = torch.zeros((4, 8))
        condition_b = torch.full((4, 8), 7.0)
        offset_a = head(condition_a)
        offset_b = head(condition_b)
    assert not torch.equal(offset_a, offset_b)


def test_cell_offset_loss_returns_gradient_and_loss():
    predictions = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.0, 0.2])
    targets = torch.tensor([0.2, -0.1, 0.4, 0.0, 0.1, 0.3])
    result = cell_offset_loss_v6(predictions, targets)
    assert result["record_count"] == 6
    assert result["gradient"].shape == (6,)
    assert torch.isfinite(result["loss"]).all()
    # predictions close to targets → small loss
    assert result["loss"].item() < 1.0


def test_cell_offset_loss_rejects_misaligned():
    with pytest.raises(Exception):
        cell_offset_loss_v6(torch.zeros(4), torch.zeros(6))


def test_cell_offset_loss_gradient_direction():
    # A positive gradient should push predictions up toward a higher target.
    predictions = torch.zeros(1)
    targets = torch.ones(1)
    result = cell_offset_loss_v6(predictions, targets)
    assert result["gradient"][0].item() < 0  # loss decreases as pred increases


def _assembly(size: int) -> CellOffsetHeadV6:
    return CellOffsetHeadV6(condition_width=8, hidden_width=16)


def test_discrimination_auroc_high_when_separated():
    # HEK293FT has strong positive offsets; everyone else negative.
    offsets = [1.0] * 6 + [-1.0] * 30
    contexts = [HEK293FT_CONTEXT] * 6 + ["GM12878", "HEPG2", "HMEC", "K562", "SKNSH"] * 6
    result = cell_offset_discrimination_v6(offsets, contexts)
    assert result["auroc"] > 0.9
    assert result["target_context"] == HEK293FT_CONTEXT


def test_discrimination_auroc_near_half_when_random():
    # Offsets independent of cell identity → AUC ~ 0.5.
    import random

    random.seed(7)
    offsets = [random.random() for _ in range(36)]
    contexts = [HEK293FT_CONTEXT] * 6 + ["GM12878", "HEPG2", "HMEC", "K562", "SKNSH"] * 6
    result = cell_offset_discrimination_v6(offsets, contexts)
    assert 0.2 <= result["auroc"] <= 0.8


def test_discrimination_rejects_missing_target_context():
    with pytest.raises(Exception):
        cell_offset_discrimination_v6([0.1, 0.2], ["GM12878", "HEPG2"])


def test_head_trainable_parameters():
    head = _assembly(2)
    count = sum(p.numel() for p in head.parameters())
    assert count > 0