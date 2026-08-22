"""Shared mRNABERT chunk and edit-site feature policy for XEditCritic V3.

The cache builder and the online encoder both call this module.  Keeping the
indexing policy here prevents a cached edit from silently referring to a
different token than the same edit encoded online.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


CHUNK_NUCLEOTIDES = 1000
CHUNK_OVERLAP = 64
LOCAL_RADIUS = 16
LEADING_SPECIAL_TOKENS = 1
TRAILING_SPECIAL_TOKENS = 1
FEATURE_SCHEMA_VERSION = "route_a_v3_route2_edit_site_token_features.v3"


class EditSiteFeatureError(RuntimeError):
    """The V3 token/chunk contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EditSiteFeatureError(message)


@dataclass(frozen=True, order=True)
class ChunkSpan:
    """A half-open nucleotide span in the un-tokenized sequence."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PositionFeature:
    """One edit position's source- or candidate-side local representation."""

    site: torch.Tensor
    window_mean: torch.Tensor
    window_max: torch.Tensor
    chunk: ChunkSpan


@dataclass(frozen=True)
class EncodedSequenceFeaturesV3:
    """All cached features requested for one unique sequence."""

    global_residual: torch.Tensor
    positions: dict[int, PositionFeature]
    local_chunk_spans: tuple[ChunkSpan, ...]


def normalize_rna(sequence: str) -> str:
    value = str(sequence).upper().replace("T", "U")
    _require(bool(value) and set(value) <= set("ACGUN"), "unsupported RNA sequence")
    return value


def format_utr_chunk(chunk: str) -> str:
    """Apply mRNABERT's official single-nucleotide, DNA-letter tokenizer input."""

    return " ".join(normalize_rna(chunk).replace("U", "T"))


def overlapping_chunk_spans(
    sequence_length: int,
    *,
    chunk_nucleotides: int = CHUNK_NUCLEOTIDES,
    overlap: int = CHUNK_OVERLAP,
) -> list[ChunkSpan]:
    """Return full-length chunks with at least the requested adjacent overlap."""

    _require(sequence_length > 0, "sequence length must be positive")
    _require(chunk_nucleotides > 0, "chunk length must be positive")
    _require(0 <= overlap < chunk_nucleotides, "chunk overlap is invalid")
    if sequence_length <= chunk_nucleotides:
        return [ChunkSpan(0, sequence_length)]
    last_start = sequence_length - chunk_nucleotides
    starts = list(range(0, last_start + 1, chunk_nucleotides - overlap))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [ChunkSpan(start, start + chunk_nucleotides) for start in starts]


def legacy_global_chunk_spans(
    sequence_length: int, *, chunk_nucleotides: int = CHUNK_NUCLEOTIDES
) -> list[ChunkSpan]:
    """Preserve V2's non-overlapping chunks for the global residual only."""

    _require(sequence_length > 0, "sequence length must be positive")
    _require(chunk_nucleotides > 0, "chunk length must be positive")
    return [
        ChunkSpan(start, min(sequence_length, start + chunk_nucleotides))
        for start in range(0, sequence_length, chunk_nucleotides)
    ]


def select_most_centered_chunk(
    position: int,
    spans: Sequence[ChunkSpan],
    *,
    sequence_length: int,
    local_radius: int = LOCAL_RADIUS,
) -> ChunkSpan:
    """Choose the containing chunk in which an edit is closest to the center.

    The chosen chunk must also contain the sequence-clipped radius-16 window.
    The 64-nt overlap guarantees this for the frozen geometry.  A smaller start
    is the deterministic tie break.
    """

    _require(0 <= position < sequence_length, "edit position is outside the sequence")
    _require(local_radius >= 0, "local radius must be nonnegative")
    window_start = max(0, position - local_radius)
    window_end = min(sequence_length, position + local_radius + 1)
    eligible = [
        span
        for span in spans
        if span.start <= window_start and window_end <= span.end
    ]
    _require(bool(eligible), "no chunk contains the complete clipped local window")
    return min(
        eligible,
        key=lambda span: (
            abs((position - span.start) - (span.length - 1) / 2.0),
            span.start,
        ),
    )


def validate_token_layout(
    attention_mask: torch.Tensor,
    *,
    chunk_length: int,
    leading_special_tokens: int = LEADING_SPECIAL_TOKENS,
    trailing_special_tokens: int = TRAILING_SPECIAL_TOKENS,
) -> None:
    """Verify the one-token-per-nucleotide offset assumed by feature extraction."""

    _require(attention_mask.ndim == 1, "attention mask must describe one chunk")
    expected = chunk_length + leading_special_tokens + trailing_special_tokens
    observed = int(attention_mask.to(torch.int64).sum().item())
    _require(observed == expected, "mRNABERT token layout is not one token per nucleotide plus two specials")
    _require(
        bool(torch.all(attention_mask[:expected] == 1).item()),
        "active mRNABERT tokens are not contiguous",
    )


def official_masked_chunk_mean(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """V2-compatible attention-masked mean, including special tokens."""

    _require(last_hidden_state.ndim == 2, "chunk hidden state must be token x hidden")
    _require(attention_mask.ndim == 1, "chunk attention mask must be one-dimensional")
    _require(last_hidden_state.shape[0] == attention_mask.shape[0], "hidden state and mask lengths differ")
    weights = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * weights).sum(dim=0) / weights.sum().clamp_min(1)


def extract_position_feature(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    span: ChunkSpan,
    sequence_length: int,
    position: int,
    local_radius: int = LOCAL_RADIUS,
    leading_special_tokens: int = LEADING_SPECIAL_TOKENS,
) -> PositionFeature:
    """Extract a site token plus radius-window mean and max from one chunk."""

    validate_token_layout(attention_mask, chunk_length=span.length)
    _require(0 <= position < sequence_length, "edit position is outside the sequence")
    _require(span.start <= position < span.end, "edit is outside the selected chunk")
    local_start = max(0, position - local_radius)
    local_end = min(sequence_length, position + local_radius + 1)
    _require(span.start <= local_start and local_end <= span.end, "local window crosses the selected chunk")
    token_position = leading_special_tokens + position - span.start
    token_start = leading_special_tokens + local_start - span.start
    token_end = leading_special_tokens + local_end - span.start
    window = last_hidden_state[token_start:token_end]
    _require(window.shape[0] == local_end - local_start, "local token window length changed")
    return PositionFeature(
        site=last_hidden_state[token_position],
        window_mean=window.mean(dim=0),
        window_max=window.max(dim=0).values,
        chunk=span,
    )


def extract_requested_position_features(
    *,
    sequence_length: int,
    positions: Iterable[int],
    hidden_by_span: Mapping[ChunkSpan, tuple[torch.Tensor, torch.Tensor]],
    local_radius: int = LOCAL_RADIUS,
) -> dict[int, PositionFeature]:
    """Shared cache/online extraction from already encoded overlapping chunks."""

    spans = overlapping_chunk_spans(sequence_length)
    _require(set(spans) <= set(hidden_by_span), "encoded local chunks are incomplete")
    result: dict[int, PositionFeature] = {}
    for position in sorted(set(int(value) for value in positions)):
        span = select_most_centered_chunk(
            position,
            spans,
            sequence_length=sequence_length,
            local_radius=local_radius,
        )
        hidden, mask = hidden_by_span[span]
        result[position] = extract_position_feature(
            hidden,
            mask,
            span=span,
            sequence_length=sequence_length,
            position=position,
            local_radius=local_radius,
        )
    return result


def legacy_global_residual(
    *,
    sequence_length: int,
    hidden_by_span: Mapping[ChunkSpan, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Reproduce V2's length-weighted mean of non-overlapping chunk means."""

    spans = legacy_global_chunk_spans(sequence_length)
    _require(set(spans) <= set(hidden_by_span), "encoded global-residual chunks are incomplete")
    weighted: torch.Tensor | None = None
    total = 0
    for span in spans:
        hidden, mask = hidden_by_span[span]
        validate_token_layout(mask, chunk_length=span.length)
        pooled = official_masked_chunk_mean(hidden, mask)
        weighted = pooled * span.length if weighted is None else weighted + pooled * span.length
        total += span.length
    _require(weighted is not None and total == sequence_length, "global residual coverage changed")
    return weighted / total
