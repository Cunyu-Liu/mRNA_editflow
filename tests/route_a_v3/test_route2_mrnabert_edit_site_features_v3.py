from __future__ import annotations

import pytest
import torch

from core.route2_mrnabert_edit_site_features_v3 import (
    ChunkSpan,
    EditSiteFeatureError,
    extract_requested_position_features,
    legacy_global_chunk_spans,
    legacy_global_residual,
    overlapping_chunk_spans,
    select_most_centered_chunk,
    validate_token_layout,
)


def _fake_chunk(span: ChunkSpan, width: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    token_positions = torch.arange(span.length + 2, dtype=torch.float32)
    hidden = torch.stack((token_positions, -token_positions), dim=1)
    mask = torch.ones(span.length + 2, dtype=torch.long)
    return hidden, mask


def test_overlap_spans_anchor_the_final_full_chunk() -> None:
    assert overlapping_chunk_spans(1000) == [ChunkSpan(0, 1000)]
    assert overlapping_chunk_spans(1500) == [ChunkSpan(0, 1000), ChunkSpan(500, 1500)]
    assert overlapping_chunk_spans(2500) == [
        ChunkSpan(0, 1000),
        ChunkSpan(936, 1936),
        ChunkSpan(1500, 2500),
    ]
    assert legacy_global_chunk_spans(2500) == [
        ChunkSpan(0, 1000),
        ChunkSpan(1000, 2000),
        ChunkSpan(2000, 2500),
    ]


def test_most_centered_chunk_and_radius_window_are_deterministic() -> None:
    spans = overlapping_chunk_spans(1500)
    assert select_most_centered_chunk(520, spans, sequence_length=1500) == spans[0]
    assert select_most_centered_chunk(900, spans, sequence_length=1500) == spans[1]
    assert select_most_centered_chunk(0, spans, sequence_length=1500) == spans[0]


def test_special_tokens_are_excluded_from_site_and_window_features() -> None:
    span = ChunkSpan(0, 40)
    hidden, mask = _fake_chunk(span)
    features = extract_requested_position_features(
        sequence_length=40,
        positions=[0, 20, 39],
        hidden_by_span={span: (hidden, mask)},
    )
    assert features[20].site.tolist() == [21.0, -21.0]
    assert features[20].window_mean.tolist() == pytest.approx([21.0, -21.0])
    assert features[0].window_mean.tolist() == pytest.approx([9.0, -9.0])
    assert features[39].window_max.tolist() == pytest.approx([40.0, -24.0])


def test_legacy_global_residual_includes_specials_and_weights_by_nucleotides() -> None:
    spans = legacy_global_chunk_spans(1500)
    hidden_by_span = {span: _fake_chunk(span) for span in spans}
    residual = legacy_global_residual(
        sequence_length=1500,
        hidden_by_span=hidden_by_span,
    )
    first_mean = (1001.0 / 2.0)
    second_mean = (501.0 / 2.0)
    expected = (1000 * first_mean + 500 * second_mean) / 1500
    assert residual.tolist() == pytest.approx([expected, -expected])


def test_layout_validation_rejects_non_one_to_one_tokenization() -> None:
    with pytest.raises(EditSiteFeatureError, match="one token per nucleotide"):
        validate_token_layout(torch.ones(41, dtype=torch.long), chunk_length=40)


def test_multi_edit_requests_are_not_truncated() -> None:
    span = ChunkSpan(0, 64)
    hidden, mask = _fake_chunk(span)
    positions = list(range(38))
    features = extract_requested_position_features(
        sequence_length=64,
        positions=positions,
        hidden_by_span={span: (hidden, mask)},
    )
    assert list(features) == positions
    assert len(features) == 38
