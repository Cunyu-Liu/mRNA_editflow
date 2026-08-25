from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
)
from core.route2_mrnabert_edit_site_features_v3 import ChunkSpan
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    MRNABERTBottomSixEncoderV4Error,
    _ChunkRequestV4,
    compare_bottom_encoded_sequences_v4,
    forward_bottom_six_hidden_v4,
    request_batches_v4,
)


class _Embeddings(nn.Module):
    def forward(self, *, input_ids, token_type_ids):
        return input_ids.float().unsqueeze(-1).repeat(1, 1, 3) + token_type_ids.unsqueeze(-1)


class _Layer(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden, **_):
        return hidden + self.increment


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _Embeddings()
        self.encoder = SimpleNamespace(layer=nn.ModuleList([_Layer(float(index + 1)) for index in range(12)]))
        self.extended_attention_mask_dtype = None

    def get_extended_attention_mask(self, attention_mask, input_shape, *, dtype):
        assert input_shape == attention_mask.shape
        self.extended_attention_mask_dtype = dtype
        return attention_mask[:, None, None, :]

def _sequence(value: float = 0.0) -> BottomEncodedSequenceV4:
    hidden = torch.tensor(
        [[value, value + 1], [value + 2, value + 3], [value + 4, value + 5], [value + 6, value + 7]],
        dtype=torch.float32,
    )
    return BottomEncodedSequenceV4(
        chunks=(
            BottomEncodedChunkV4(
                span=ChunkSpan(0, 2),
                hidden=hidden,
                attention_mask=torch.ones(4, dtype=torch.bool),
            ),
        ),
        global_residual=torch.tensor([value, value + 1]),
    )


def test_shared_bottom_six_forward_runs_exactly_blocks_zero_through_five() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    model = _Model()
    output = forward_bottom_six_hidden_v4(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=torch.zeros_like(input_ids),
    )
    # Sum 1..6 = 21.  Blocks 6..11 would add another 57 and must not run.
    expected = input_ids.float().unsqueeze(-1).repeat(1, 1, 3) + 21
    assert torch.equal(output, expected)
    assert model.extended_attention_mask_dtype == torch.float32


def test_bottom_six_forward_rejects_encoder_depth_drift() -> None:
    model = _Model()
    model.encoder.layer = nn.ModuleList(list(model.encoder.layer[:11]))
    with pytest.raises(MRNABERTBottomSixEncoderV4Error, match="depth"):
        forward_bottom_six_hidden_v4(
            model,
            input_ids=torch.ones((1, 3), dtype=torch.long),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            token_type_ids=torch.zeros((1, 3), dtype=torch.long),
        )


def test_cache_and_online_comparison_uses_both_max_and_mean_tolerances() -> None:
    cached = {0: _sequence(0.0)}
    passing = compare_bottom_encoded_sequences_v4(
        cached,
        {0: _sequence(0.004)},
        maximum_absolute_tolerance=0.02,
        mean_absolute_tolerance=0.005,
    )
    assert passing["passed"] is True
    failing = compare_bottom_encoded_sequences_v4(
        cached,
        {0: _sequence(0.006)},
        maximum_absolute_tolerance=0.02,
        mean_absolute_tolerance=0.005,
    )
    assert failing["passed"] is False


def test_cache_online_comparison_rejects_token_offset_or_mask_drift() -> None:
    cached = {0: _sequence()}
    changed = _sequence()
    changed = BottomEncodedSequenceV4(
        chunks=(
            BottomEncodedChunkV4(
                span=changed.chunks[0].span,
                hidden=changed.chunks[0].hidden,
                attention_mask=torch.tensor([True, True, False, True]),
            ),
        ),
        global_residual=changed.global_residual,
    )
    with pytest.raises(MRNABERTBottomSixEncoderV4Error, match="attention mask"):
        compare_bottom_encoded_sequences_v4(
            cached,
            {0: changed},
            maximum_absolute_tolerance=0.02,
            mean_absolute_tolerance=0.005,
        )


def test_chunk_request_batching_is_deterministic_and_respects_both_limits() -> None:
    requests = [
        _ChunkRequestV4(3, ChunkSpan(0, 1000)),
        _ChunkRequestV4(1, ChunkSpan(0, 100)),
        _ChunkRequestV4(2, ChunkSpan(0, 100)),
        _ChunkRequestV4(0, ChunkSpan(0, 100)),
    ]
    batches = request_batches_v4(requests, maximum_sequences=2, token_budget=1200)
    assert [[request.sequence_index for request in batch] for batch in batches] == [[0, 1], [2], [3]]
    assert all(len(batch) <= 2 for batch in batches)
