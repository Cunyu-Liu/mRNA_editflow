from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (
    MRNABERTUpperSixEncoderV4Error,
    additive_attention_mask_v4,
    forward_upper_six_layers_v4,
)


class _Layer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.last_attention_mask: torch.Tensor | None = None

    def forward(self, hidden, *, attention_mask, **_):
        self.last_attention_mask = attention_mask
        return (hidden + self.scale,)


def test_upper_six_runs_exactly_six_trainable_layers_and_propagates_gradients() -> None:
    layers = nn.ModuleList([_Layer(float(index + 1)) for index in range(6)])
    hidden = torch.zeros((2, 5, 3), requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    output = forward_upper_six_layers_v4(layers, hidden, mask)
    assert torch.equal(output, torch.full_like(output, 21.0))
    output.sum().backward()
    assert hidden.grad is not None
    assert all(layer.scale.grad is not None for layer in layers)
    assert all(layer.last_attention_mask is not None for layer in layers)


def test_upper_six_activation_checkpointing_preserves_gradients() -> None:
    layers = nn.ModuleList([_Layer(float(index + 1)) for index in range(6)])
    hidden = torch.zeros((2, 5, 3), requires_grad=True)
    mask = torch.ones((2, 5), dtype=torch.bool)
    output = forward_upper_six_layers_v4(
        layers,
        hidden,
        mask,
        activation_checkpointing=True,
    )
    output.square().mean().backward()
    assert hidden.grad is not None
    assert all(layer.scale.grad is not None for layer in layers)


def test_upper_six_rejects_a_fifth_or_seventh_layer_scope() -> None:
    hidden = torch.zeros((1, 3, 2))
    mask = torch.ones((1, 3), dtype=torch.bool)
    for count in (5, 7):
        with pytest.raises(MRNABERTUpperSixEncoderV4Error, match="exactly"):
            forward_upper_six_layers_v4(
                nn.ModuleList([_Layer(1.0) for _ in range(count)]),
                hidden,
                mask,
            )


def test_additive_attention_mask_keeps_active_tokens_and_blocks_padding() -> None:
    mask = torch.tensor([[True, True, False]])
    additive = additive_attention_mask_v4(mask, dtype=torch.float32)
    assert additive.shape == (1, 1, 1, 3)
    assert additive[0, 0, 0, 0].item() == 0.0
    assert additive[0, 0, 0, 1].item() == 0.0
    assert additive[0, 0, 0, 2].item() == torch.finfo(torch.float32).min
