"""Outcome-free bottom-six mRNABERT token cache for XEditCritic V4.

The cache retains token tensors and indexing metadata only.  Raw sequences stay
in the authorized TRAIN/VALIDATION projection and are never copied into this
artifact.  Record-level edit references are ragged and point to the
most-centered radius-32 source and candidate chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.route2_mrnabert_edit_site_features_v3 import (
    CHUNK_NUCLEOTIDES,
    CHUNK_OVERLAP,
    LEADING_SPECIAL_TOKENS,
    TRAILING_SPECIAL_TOKENS,
    ChunkSpan,
    overlapping_chunk_spans,
    select_most_centered_chunk,
    validate_token_layout,
)


FROZEN_BOTTOM_ENCODER_CHUNK_CACHE_V4_SCHEMA = (
    "route_a_v3_route2_frozen_bottom_encoder_chunk_cache.v4"
)
V4_LOCAL_RADIUS = 32
V4_ALLOWED_SPLITS = frozenset({"TRAIN", "VALIDATION"})


class FrozenBottomEncoderChunkCacheV4Error(RuntimeError):
    """The V4 outcome, chunk, token, or ragged-record boundary was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenBottomEncoderChunkCacheV4Error(message)


@dataclass(frozen=True)
class BottomEncodedChunkV4:
    """Bottom-six hidden states for one tokenized sequence chunk."""

    span: ChunkSpan
    hidden: torch.Tensor
    attention_mask: torch.Tensor
    special_token_offset: int = LEADING_SPECIAL_TOKENS


@dataclass(frozen=True)
class BottomEncodedSequenceV4:
    """All overlapping chunks and one outcome-free global residual."""

    chunks: tuple[BottomEncodedChunkV4, ...]
    global_residual: torch.Tensor


def _validate_encoded_sequence(
    encoded: BottomEncodedSequenceV4,
    *,
    sequence_length: int,
    width: int | None,
) -> int:
    expected_spans = tuple(
        overlapping_chunk_spans(
            sequence_length,
            chunk_nucleotides=CHUNK_NUCLEOTIDES,
            overlap=CHUNK_OVERLAP,
        )
    )
    _require(tuple(chunk.span for chunk in encoded.chunks) == expected_spans, "bottom-six chunks differ from the frozen overlap policy")
    _require(encoded.global_residual.ndim == 1, "global residual is not a vector")
    observed_width = int(encoded.global_residual.numel())
    _require(observed_width > 0, "bottom-six hidden width is empty")
    if width is not None:
        _require(observed_width == width, "bottom-six hidden width changed between sequences")
    _require(torch.isfinite(encoded.global_residual).all().item(), "global residual is nonfinite")
    for chunk in encoded.chunks:
        _require(chunk.hidden.ndim == 2, "chunk hidden must be token x hidden")
        _require(chunk.attention_mask.ndim == 1, "chunk attention mask must be one-dimensional")
        _require(chunk.hidden.shape == (chunk.attention_mask.numel(), observed_width), "chunk hidden and attention geometry differ")
        _require(chunk.special_token_offset == LEADING_SPECIAL_TOKENS, "special-token offset differs from the frozen tokenizer")
        validate_token_layout(
            chunk.attention_mask,
            chunk_length=chunk.span.length,
            leading_special_tokens=LEADING_SPECIAL_TOKENS,
            trailing_special_tokens=TRAILING_SPECIAL_TOKENS,
        )
        _require(torch.isfinite(chunk.hidden).all().item(), "chunk hidden is nonfinite")
    return observed_width


def assemble_frozen_bottom_encoder_chunk_cache_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence_to_index: Mapping[str, int],
    encoded: Mapping[int, BottomEncodedSequenceV4],
    model_id: str,
    pretrained_parameter_count: int,
    attention_backend: str,
) -> dict[str, Any]:
    """Assemble a raw-sequence-free token cache from authorized projection rows."""

    _require(bool(rows), "no projection rows were supplied")
    _require(
        all(str(row.get("split")) in V4_ALLOWED_SPLITS for row in rows),
        "V4 cache accepts TRAIN/VALIDATION projection rows only",
    )
    _require(
        not any(
            "evaluation" in str(value).lower()
            for row in rows
            for key, value in row.items()
            if key in {"split", "pool_assignment"}
        ),
        "Evaluation row entered the V4 cache input",
    )
    sequence_count = len(sequence_to_index)
    _require(set(sequence_to_index.values()) == set(range(sequence_count)), "sequence indices are not dense")
    _require(set(encoded) == set(range(sequence_count)), "bottom-six encodings do not cover every unique sequence")
    sequence_lengths = torch.zeros(sequence_count, dtype=torch.int32)
    for sequence, index in sequence_to_index.items():
        _require(bool(sequence), "empty sequence entered the V4 cache")
        sequence_lengths[index] = len(sequence)

    width: int | None = None
    sequence_chunk_offsets = [0]
    chunk_sequence_indices: list[int] = []
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    chunk_special_token_offsets: list[int] = []
    chunk_token_offsets = [0]
    token_hidden: list[torch.Tensor] = []
    token_attention_mask: list[torch.Tensor] = []
    sequence_global_residuals: list[torch.Tensor] = []
    chunk_index: dict[tuple[int, ChunkSpan], int] = {}
    for sequence_index in range(sequence_count):
        length = int(sequence_lengths[sequence_index].item())
        representation = encoded[sequence_index]
        width = _validate_encoded_sequence(
            representation,
            sequence_length=length,
            width=width,
        )
        sequence_global_residuals.append(representation.global_residual)
        for chunk in representation.chunks:
            cache_chunk_index = len(chunk_starts)
            chunk_index[(sequence_index, chunk.span)] = cache_chunk_index
            chunk_sequence_indices.append(sequence_index)
            chunk_starts.append(chunk.span.start)
            chunk_ends.append(chunk.span.end)
            chunk_special_token_offsets.append(chunk.special_token_offset)
            token_hidden.append(chunk.hidden)
            token_attention_mask.append(chunk.attention_mask.to(torch.bool))
            chunk_token_offsets.append(chunk_token_offsets[-1] + chunk.hidden.shape[0])
        sequence_chunk_offsets.append(len(chunk_starts))
    _require(width is not None, "bottom-six cache has no hidden width")

    ordered_rows = sorted(rows, key=lambda row: str(row["canonical_record_id"]))
    record_ids: list[str] = []
    record_splits: list[str] = []
    record_source_sequence_indices: list[int] = []
    record_candidate_sequence_indices: list[int] = []
    record_edit_offsets = [0]
    edit_positions: list[int] = []
    edit_source_chunk_indices: list[int] = []
    edit_candidate_chunk_indices: list[int] = []
    edit_source_token_centers: list[int] = []
    edit_candidate_token_centers: list[int] = []
    edit_source_window_starts: list[int] = []
    edit_source_window_ends: list[int] = []
    edit_candidate_window_starts: list[int] = []
    edit_candidate_window_ends: list[int] = []
    seen_records: set[str] = set()
    for row in ordered_rows:
        record_id = str(row["canonical_record_id"])
        _require(record_id and record_id not in seen_records, "projection record id is empty or duplicated")
        seen_records.add(record_id)
        split = str(row["split"])
        source = str(row["source_sequence"])
        candidate = str(row["candidate_sequence"])
        _require(source in sequence_to_index and candidate in sequence_to_index, "record sequence is absent from the unique sequence registry")
        _require(len(source) == len(candidate), "length-changing record entered the V4 SUB cache")
        source_index = int(sequence_to_index[source])
        candidate_index = int(sequence_to_index[candidate])
        source_spans = tuple(chunk.span for chunk in encoded[source_index].chunks)
        candidate_spans = tuple(chunk.span for chunk in encoded[candidate_index].chunks)
        record_ids.append(record_id)
        record_splits.append(split)
        record_source_sequence_indices.append(source_index)
        record_candidate_sequence_indices.append(candidate_index)
        for edit in row["source_relative_edits"]:
            position = int(edit["position"])
            source_span = select_most_centered_chunk(
                position,
                source_spans,
                sequence_length=len(source),
                local_radius=V4_LOCAL_RADIUS,
            )
            candidate_span = select_most_centered_chunk(
                position,
                candidate_spans,
                sequence_length=len(candidate),
                local_radius=V4_LOCAL_RADIUS,
            )
            source_chunk = encoded[source_index].chunks[source_spans.index(source_span)]
            candidate_chunk = encoded[candidate_index].chunks[candidate_spans.index(candidate_span)]
            source_window_start = max(0, position - V4_LOCAL_RADIUS)
            source_window_end = min(len(source), position + V4_LOCAL_RADIUS + 1)
            candidate_window_start = max(0, position - V4_LOCAL_RADIUS)
            candidate_window_end = min(len(candidate), position + V4_LOCAL_RADIUS + 1)
            edit_positions.append(position)
            edit_source_chunk_indices.append(chunk_index[(source_index, source_span)])
            edit_candidate_chunk_indices.append(chunk_index[(candidate_index, candidate_span)])
            edit_source_token_centers.append(
                source_chunk.special_token_offset + position - source_span.start
            )
            edit_candidate_token_centers.append(
                candidate_chunk.special_token_offset + position - candidate_span.start
            )
            edit_source_window_starts.append(
                source_chunk.special_token_offset + source_window_start - source_span.start
            )
            edit_source_window_ends.append(
                source_chunk.special_token_offset + source_window_end - source_span.start
            )
            edit_candidate_window_starts.append(
                candidate_chunk.special_token_offset + candidate_window_start - candidate_span.start
            )
            edit_candidate_window_ends.append(
                candidate_chunk.special_token_offset + candidate_window_end - candidate_span.start
            )
        record_edit_offsets.append(len(edit_positions))

    payload: dict[str, Any] = {
        "schema_version": FROZEN_BOTTOM_ENCODER_CHUNK_CACHE_V4_SCHEMA,
        "encoder_family": "mRNABERT",
        "model_id": str(model_id),
        "pretrained_parameter_count": int(pretrained_parameter_count),
        "embedding_width": int(width),
        "frozen_encoder_blocks": [0, 1, 2, 3, 4, 5],
        "trainable_encoder_blocks": [6, 7, 8, 9, 10, 11],
        "chunk_length": CHUNK_NUCLEOTIDES,
        "chunk_overlap": CHUNK_OVERLAP,
        "local_context_radius": V4_LOCAL_RADIUS,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "attention_backend": str(attention_backend),
        "record_ids": record_ids,
        "record_splits": record_splits,
        "record_source_sequence_indices": torch.tensor(record_source_sequence_indices, dtype=torch.int64),
        "record_candidate_sequence_indices": torch.tensor(record_candidate_sequence_indices, dtype=torch.int64),
        "record_edit_offsets": torch.tensor(record_edit_offsets, dtype=torch.int64),
        "edit_positions": torch.tensor(edit_positions, dtype=torch.int32),
        "edit_source_chunk_indices": torch.tensor(edit_source_chunk_indices, dtype=torch.int64),
        "edit_candidate_chunk_indices": torch.tensor(edit_candidate_chunk_indices, dtype=torch.int64),
        "edit_source_token_centers": torch.tensor(edit_source_token_centers, dtype=torch.int32),
        "edit_candidate_token_centers": torch.tensor(edit_candidate_token_centers, dtype=torch.int32),
        "edit_source_window_starts": torch.tensor(edit_source_window_starts, dtype=torch.int32),
        "edit_source_window_ends": torch.tensor(edit_source_window_ends, dtype=torch.int32),
        "edit_candidate_window_starts": torch.tensor(edit_candidate_window_starts, dtype=torch.int32),
        "edit_candidate_window_ends": torch.tensor(edit_candidate_window_ends, dtype=torch.int32),
        "sequence_lengths": sequence_lengths,
        "sequence_chunk_offsets": torch.tensor(sequence_chunk_offsets, dtype=torch.int64),
        "chunk_sequence_indices": torch.tensor(chunk_sequence_indices, dtype=torch.int64),
        "chunk_starts": torch.tensor(chunk_starts, dtype=torch.int32),
        "chunk_ends": torch.tensor(chunk_ends, dtype=torch.int32),
        "chunk_special_token_offsets": torch.tensor(chunk_special_token_offsets, dtype=torch.int16),
        "chunk_token_offsets": torch.tensor(chunk_token_offsets, dtype=torch.int64),
        "token_hidden": torch.cat(token_hidden, dim=0).to(torch.float16),
        "token_attention_mask": torch.cat(token_attention_mask, dim=0).to(torch.bool),
        "sequence_global_residuals": torch.stack(sequence_global_residuals).to(torch.float16),
        "raw_sequence_payload_written": 0,
        "label_or_outcome_payload_written": 0,
        "development_test_record_count": 0,
        "evaluation_record_count": 0,
    }
    validate_frozen_bottom_encoder_chunk_cache_v4(payload)
    return payload


def validate_frozen_bottom_encoder_chunk_cache_v4(payload: Mapping[str, Any]) -> None:
    """Validate the protected-data boundary and every ragged cache reference."""

    _require(payload.get("schema_version") == FROZEN_BOTTOM_ENCODER_CHUNK_CACHE_V4_SCHEMA, "unexpected V4 cache schema")
    _require(payload.get("raw_sequence_payload_written") == 0, "raw sequence entered the V4 cache")
    _require(payload.get("label_or_outcome_payload_written") == 0, "label or outcome entered the V4 cache")
    _require(payload.get("development_test_record_count") == 0, "Development TEST entered the V4 cache")
    _require(payload.get("evaluation_record_count") == 0, "Evaluation entered the V4 cache")
    _require(set(payload["record_splits"]) <= V4_ALLOWED_SPLITS, "unauthorized split entered the V4 cache")
    record_count = len(payload["record_ids"])
    _require(len(set(payload["record_ids"])) == record_count, "record ids are duplicated")
    for key in ("record_source_sequence_indices", "record_candidate_sequence_indices"):
        _require(tuple(payload[key].shape) == (record_count,), f"{key} geometry changed")
    edit_offsets = payload["record_edit_offsets"]
    _require(tuple(edit_offsets.shape) == (record_count + 1,), "record edit offsets geometry changed")
    _require(int(edit_offsets[0].item()) == 0, "record edit offsets do not start at zero")
    _require(bool(torch.all(edit_offsets[1:] >= edit_offsets[:-1]).item()), "record edit offsets are not monotone")
    edit_count = int(edit_offsets[-1].item())
    edit_keys = (
        "edit_positions",
        "edit_source_chunk_indices",
        "edit_candidate_chunk_indices",
        "edit_source_token_centers",
        "edit_candidate_token_centers",
        "edit_source_window_starts",
        "edit_source_window_ends",
        "edit_candidate_window_starts",
        "edit_candidate_window_ends",
    )
    for key in edit_keys:
        _require(int(payload[key].numel()) == edit_count, f"{key} does not follow ragged edit offsets")
    sequence_count = int(payload["sequence_lengths"].numel())
    sequence_chunk_offsets = payload["sequence_chunk_offsets"]
    _require(tuple(sequence_chunk_offsets.shape) == (sequence_count + 1,), "sequence chunk offsets geometry changed")
    chunk_count = int(sequence_chunk_offsets[-1].item())
    for key in ("chunk_sequence_indices", "chunk_starts", "chunk_ends", "chunk_special_token_offsets"):
        _require(int(payload[key].numel()) == chunk_count, f"{key} chunk count changed")
    chunk_token_offsets = payload["chunk_token_offsets"]
    _require(tuple(chunk_token_offsets.shape) == (chunk_count + 1,), "chunk token offsets geometry changed")
    token_count = int(chunk_token_offsets[-1].item())
    width = int(payload["embedding_width"])
    _require(tuple(payload["token_hidden"].shape) == (token_count, width), "flat token hidden geometry changed")
    _require(payload["token_hidden"].dtype == torch.float16, "bottom-six token cache is not float16")
    _require(tuple(payload["token_attention_mask"].shape) == (token_count,), "flat attention mask geometry changed")
    _require(payload["token_attention_mask"].dtype == torch.bool, "flat attention mask is not boolean")
    _require(tuple(payload["sequence_global_residuals"].shape) == (sequence_count, width), "global residual geometry changed")
    _require(payload["sequence_global_residuals"].dtype == torch.float16, "global residual cache is not float16")
    _require(torch.isfinite(payload["token_hidden"]).all().item(), "bottom-six token cache is nonfinite")
    _require(torch.isfinite(payload["sequence_global_residuals"]).all().item(), "global residual cache is nonfinite")
    if edit_count:
        for side in ("source", "candidate"):
            chunk_indices = payload[f"edit_{side}_chunk_indices"]
            centers = payload[f"edit_{side}_token_centers"]
            starts = payload[f"edit_{side}_window_starts"]
            ends = payload[f"edit_{side}_window_ends"]
            _require(int(chunk_indices.min().item()) >= 0 and int(chunk_indices.max().item()) < chunk_count, f"{side} edit chunk reference is out of range")
            for edit_index in range(edit_count):
                chunk_index = int(chunk_indices[edit_index].item())
                token_length = int(chunk_token_offsets[chunk_index + 1] - chunk_token_offsets[chunk_index])
                center = int(centers[edit_index].item())
                start = int(starts[edit_index].item())
                end = int(ends[edit_index].item())
                _require(0 <= start <= center < end <= token_length, f"{side} local token window is invalid")
                _require(end - start <= 2 * V4_LOCAL_RADIUS + 1, f"{side} local token window exceeds radius 32")


def require_frozen_bottom_encoder_chunk_cache_identity_v4(
    payload: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_record_count: int,
    expected_unique_sequence_count: int,
    expected_embedding_width: int = 768,
    validate_payload: bool = True,
) -> dict[str, Any]:
    """Bind a valid tensor cache to the frozen V4 encoder and chunk policy."""

    if validate_payload:
        validate_frozen_bottom_encoder_chunk_cache_v4(payload)
    _require(str(payload.get("model_id")) == str(expected_model_id), "bottom-six cache mRNABERT revision changed")
    _require(len(payload["record_ids"]) == int(expected_record_count), "bottom-six cache record count changed")
    _require(
        int(payload["sequence_lengths"].numel())
        == int(expected_unique_sequence_count),
        "bottom-six cache unique sequence count changed",
    )
    _require(int(payload.get("embedding_width", -1)) == int(expected_embedding_width), "bottom-six cache embedding width changed")
    _require(payload.get("frozen_encoder_blocks") == [0, 1, 2, 3, 4, 5], "bottom-six cache frozen block scope changed")
    _require(payload.get("trainable_encoder_blocks") == [6, 7, 8, 9, 10, 11], "bottom-six cache upper block scope changed")
    _require(int(payload.get("chunk_length", -1)) == CHUNK_NUCLEOTIDES, "bottom-six cache chunk length changed")
    _require(int(payload.get("chunk_overlap", -1)) == CHUNK_OVERLAP, "bottom-six cache chunk overlap changed")
    _require(int(payload.get("local_context_radius", -1)) == V4_LOCAL_RADIUS, "bottom-six cache local radius changed")
    _require(
        payload.get("tokenization_policy")
        == "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "bottom-six cache tokenization policy changed",
    )
    offsets = payload["chunk_special_token_offsets"]
    _require(
        bool(torch.all(offsets == LEADING_SPECIAL_TOKENS).item()),
        "bottom-six cache special-token offset changed",
    )
    receipt = {
        "model_id": str(payload["model_id"]),
        "record_count": len(payload["record_ids"]),
        "unique_sequence_count": int(payload["sequence_lengths"].numel()),
        "embedding_width": int(payload["embedding_width"]),
        "frozen_encoder_blocks": list(payload["frozen_encoder_blocks"]),
        "trainable_encoder_blocks": list(payload["trainable_encoder_blocks"]),
        "chunk_length": int(payload["chunk_length"]),
        "chunk_overlap": int(payload["chunk_overlap"]),
        "local_context_radius": int(payload["local_context_radius"]),
        "special_token_offset": LEADING_SPECIAL_TOKENS,
    }
    require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4(
        receipt,
        expected_model_id=expected_model_id,
        expected_record_count=expected_record_count,
        expected_unique_sequence_count=expected_unique_sequence_count,
        expected_embedding_width=expected_embedding_width,
    )
    return receipt


def require_frozen_bottom_encoder_chunk_cache_identity_receipt_v4(
    receipt: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_record_count: int,
    expected_unique_sequence_count: int,
    expected_embedding_width: int = 768,
) -> None:
    """Require the preflight receipt produced from the frozen tensor payload."""

    _require(isinstance(receipt, Mapping), "bottom-six cache identity receipt is absent")
    _require(str(receipt.get("model_id")) == str(expected_model_id), "bottom-six cache identity receipt revision changed")
    _require(int(receipt.get("record_count", -1)) == int(expected_record_count), "bottom-six cache identity receipt record count changed")
    _require(int(receipt.get("unique_sequence_count", -1)) == int(expected_unique_sequence_count), "bottom-six cache identity receipt sequence count changed")
    _require(int(receipt.get("embedding_width", -1)) == int(expected_embedding_width), "bottom-six cache identity receipt width changed")
    _require(receipt.get("frozen_encoder_blocks") == [0, 1, 2, 3, 4, 5], "bottom-six cache identity receipt frozen blocks changed")
    _require(receipt.get("trainable_encoder_blocks") == [6, 7, 8, 9, 10, 11], "bottom-six cache identity receipt upper blocks changed")
    _require(int(receipt.get("chunk_length", -1)) == CHUNK_NUCLEOTIDES, "bottom-six cache identity receipt chunk length changed")
    _require(int(receipt.get("chunk_overlap", -1)) == CHUNK_OVERLAP, "bottom-six cache identity receipt overlap changed")
    _require(int(receipt.get("local_context_radius", -1)) == V4_LOCAL_RADIUS, "bottom-six cache identity receipt radius changed")
    _require(int(receipt.get("special_token_offset", -1)) == LEADING_SPECIAL_TOKENS, "bottom-six cache identity receipt special-token offset changed")


def materialize_bottom_chunk_batch_v4(
    payload: Mapping[str, Any],
    record_indices: Sequence[int],
    *,
    validate_payload: bool = True,
) -> dict[str, torch.Tensor]:
    """Materialize each required source/candidate chunk exactly once per batch."""

    if validate_payload:
        validate_frozen_bottom_encoder_chunk_cache_v4(payload)
    _require(bool(record_indices), "physical batch is empty")
    record_count = len(payload["record_ids"])
    ordered_record_indices = [int(index) for index in record_indices]
    _require(min(ordered_record_indices) >= 0 and max(ordered_record_indices) < record_count, "physical batch record index is out of range")
    global_edit_indices: list[int] = []
    local_record_edit_offsets = [0]
    for record_index in ordered_record_indices:
        start = int(payload["record_edit_offsets"][record_index].item())
        end = int(payload["record_edit_offsets"][record_index + 1].item())
        global_edit_indices.extend(range(start, end))
        local_record_edit_offsets.append(len(global_edit_indices))
    edit_index_tensor = torch.tensor(global_edit_indices, dtype=torch.long)
    needed_chunks = sorted(
        {
            int(payload[f"edit_{side}_chunk_indices"][index].item())
            for side in ("source", "candidate")
            for index in global_edit_indices
        }
    )
    # Identity rows can contain no edits.  They still need one source chunk so
    # that upper-six batching and global residual paths retain a nonempty shape.
    if not needed_chunks:
        for record_index in ordered_record_indices:
            sequence_index = int(payload["record_source_sequence_indices"][record_index].item())
            needed_chunks.append(int(payload["sequence_chunk_offsets"][sequence_index].item()))
        needed_chunks = sorted(set(needed_chunks))
    local_chunk_index = {global_index: local_index for local_index, global_index in enumerate(needed_chunks)}
    token_offsets = payload["chunk_token_offsets"]
    lengths = [int(token_offsets[index + 1] - token_offsets[index]) for index in needed_chunks]
    maximum_length = max(lengths)
    width = int(payload["embedding_width"])
    chunk_hidden = torch.zeros((len(needed_chunks), maximum_length, width), dtype=torch.float16)
    chunk_attention_mask = torch.zeros((len(needed_chunks), maximum_length), dtype=torch.bool)
    for local_index, (global_index, length) in enumerate(zip(needed_chunks, lengths, strict=True)):
        start = int(token_offsets[global_index].item())
        end = int(token_offsets[global_index + 1].item())
        chunk_hidden[local_index, :length] = payload["token_hidden"][start:end]
        chunk_attention_mask[local_index, :length] = payload["token_attention_mask"][start:end]
    result: dict[str, torch.Tensor] = {
        "cache_record_indices": torch.tensor(ordered_record_indices, dtype=torch.int64),
        "cache_chunk_indices": torch.tensor(needed_chunks, dtype=torch.int64),
        "chunk_hidden": chunk_hidden,
        "chunk_attention_mask": chunk_attention_mask,
        "record_edit_offsets": torch.tensor(local_record_edit_offsets, dtype=torch.int64),
        "record_source_global": payload["sequence_global_residuals"][payload["record_source_sequence_indices"][ordered_record_indices]],
        "record_candidate_global": payload["sequence_global_residuals"][payload["record_candidate_sequence_indices"][ordered_record_indices]],
    }
    for side in ("source", "candidate"):
        result[f"edit_{side}_chunk_indices"] = torch.tensor(
            [
                local_chunk_index[int(payload[f"edit_{side}_chunk_indices"][index].item())]
                for index in global_edit_indices
            ],
            dtype=torch.int64,
        )
        for suffix in ("token_centers", "window_starts", "window_ends"):
            result[f"edit_{side}_{suffix}"] = payload[f"edit_{side}_{suffix}"][edit_index_tensor].to(torch.int64)
    result["edit_positions"] = payload["edit_positions"][edit_index_tensor].to(torch.int64)
    return result


def load_frozen_bottom_encoder_chunk_cache_v4(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "V4 cache payload is not a mapping")
    validate_frozen_bottom_encoder_chunk_cache_v4(payload)
    return payload
