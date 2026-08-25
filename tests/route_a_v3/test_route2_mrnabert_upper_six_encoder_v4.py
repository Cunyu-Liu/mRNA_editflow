from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (
    MRNABERTUpperSixEncoderV4Error,
    forward_upper_six_layers_v4,
)


def unpad_input(hidden: torch.Tensor, active: torch.Tensor):
    indices = torch.nonzero(active.flatten(), as_tuple=False).flatten()
    lengths = active.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(
        active.shape[0] + 1,
        dtype=torch.int32,
        device=hidden.device,
    )
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    values = hidden.flatten(0, 1).index_select(0, indices)
    return values, indices, cu_seqlens, int(lengths.max().item())


def pad_input(
    values: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    padded = values.new_zeros((batch * seqlen, values.shape[-1]))
    padded.index_copy_(0, indices, values)
    return padded.reshape(batch, seqlen, values.shape[-1])


def _runtime(seqlen: int) -> dict[str, object]:
    return {
        "alibi": torch.zeros((1, 1, seqlen, seqlen)),
        "unpad_input": unpad_input,
        "pad_input": pad_input,
    }


class _Layer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.last_attention_mask: torch.Tensor | None = None

    def forward(
        self,
        hidden,
        cu_seqlens,
        seqlen,
        subset_idx,
        indices,
        *,
        attn_mask,
        bias,
    ):
        assert cu_seqlens.ndim == indices.ndim == 1
        assert subset_idx is None
        assert bias.shape[-2:] == (seqlen, seqlen)
        self.last_attention_mask = attn_mask
        return hidden + self.scale


def test_upper_six_runs_exactly_six_trainable_layers_and_propagates_gradients() -> None:
    layers = nn.ModuleList([_Layer(float(index + 1)) for index in range(6)])
    hidden = torch.zeros((2, 5, 3), requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    output = forward_upper_six_layers_v4(
        layers,
        hidden,
        mask,
        **_runtime(5),
    )
    expected = torch.full_like(output, 21.0)
    expected[1, 3:] = 0
    assert torch.equal(output, expected)
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
        **_runtime(5),
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
                **_runtime(3),
            )
