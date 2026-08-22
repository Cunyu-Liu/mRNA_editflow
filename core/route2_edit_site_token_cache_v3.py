"""Record-aligned, raw-sequence-free edit-site token cache for Critic V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from core.route2_mrnabert_edit_site_features_v3 import (
    FEATURE_SCHEMA_VERSION,
    EncodedSequenceFeaturesV3,
)


class EditSiteTokenCacheError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EditSiteTokenCacheError(message)


def assemble_edit_site_token_cache_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence_to_index: Mapping[str, int],
    encoded: Mapping[int, EncodedSequenceFeaturesV3],
    model_id: str,
    pretrained_parameter_count: int,
    attention_backend: str,
) -> dict[str, Any]:
    """Assemble shared sequence/position features plus ragged record references."""

    ordered_rows = sorted(rows, key=lambda row: str(row["canonical_record_id"]))
    sequence_count = len(sequence_to_index)
    _require(set(sequence_to_index.values()) == set(range(sequence_count)), "sequence indices are not dense")
    _require(set(encoded) == set(range(sequence_count)), "encoded sequence keys are incomplete")
    global_residuals = torch.stack(
        [encoded[index].global_residual for index in range(sequence_count)]
    ).to(torch.float16)
    sequence_lengths = torch.zeros(sequence_count, dtype=torch.int32)
    chunk_offsets = [0]
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    position_offsets = [0]
    position_indices: list[int] = []
    sites: list[torch.Tensor] = []
    window_means: list[torch.Tensor] = []
    window_maxes: list[torch.Tensor] = []
    feature_index: dict[tuple[int, int], int] = {}
    for sequence, index in sequence_to_index.items():
        sequence_lengths[index] = len(sequence)
    for index in range(sequence_count):
        for span in encoded[index].local_chunk_spans:
            chunk_starts.append(span.start)
            chunk_ends.append(span.end)
        chunk_offsets.append(len(chunk_starts))
        for position, feature in sorted(encoded[index].positions.items()):
            feature_index[(index, position)] = len(position_indices)
            position_indices.append(position)
            sites.append(feature.site)
            window_means.append(feature.window_mean)
            window_maxes.append(feature.window_max)
        position_offsets.append(len(position_indices))

    record_edit_offsets = [0]
    edit_positions: list[int] = []
    edit_source_feature_indices: list[int] = []
    edit_candidate_feature_indices: list[int] = []
    source_sequence_indices: list[int] = []
    candidate_sequence_indices: list[int] = []
    for row in ordered_rows:
        source_index = sequence_to_index[str(row["source_sequence"])]
        candidate_index = sequence_to_index[str(row["candidate_sequence"])]
        source_sequence_indices.append(source_index)
        candidate_sequence_indices.append(candidate_index)
        for edit in row["source_relative_edits"]:
            position = int(edit["position"])
            edit_positions.append(position)
            edit_source_feature_indices.append(feature_index[(source_index, position)])
            edit_candidate_feature_indices.append(feature_index[(candidate_index, position)])
        record_edit_offsets.append(len(edit_positions))

    width = int(global_residuals.shape[1])
    empty = torch.empty((0, width), dtype=torch.float16)
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "encoder_family": "mRNABERT",
        "model_id": str(model_id),
        "pretrained_parameter_count": int(pretrained_parameter_count),
        "embedding_width": width,
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "local_chunk_policy": "1000_NT_OVERLAP_64_MOST_CENTERED_RADIUS_16",
        "global_residual_policy": "V2_OFFICIAL_MASKED_MEAN_NONOVERLAP_LENGTH_WEIGHTED",
        "attention_backend": str(attention_backend),
        "record_ids": [str(row["canonical_record_id"]) for row in ordered_rows],
        "record_edit_offsets": torch.tensor(record_edit_offsets, dtype=torch.int64),
        "edit_positions": torch.tensor(edit_positions, dtype=torch.int32),
        "edit_source_feature_indices": torch.tensor(edit_source_feature_indices, dtype=torch.int64),
        "edit_candidate_feature_indices": torch.tensor(edit_candidate_feature_indices, dtype=torch.int64),
        "record_source_sequence_indices": torch.tensor(source_sequence_indices, dtype=torch.int64),
        "record_candidate_sequence_indices": torch.tensor(candidate_sequence_indices, dtype=torch.int64),
        "sequence_lengths": sequence_lengths,
        "sequence_chunk_offsets": torch.tensor(chunk_offsets, dtype=torch.int64),
        "chunk_starts": torch.tensor(chunk_starts, dtype=torch.int32),
        "chunk_ends": torch.tensor(chunk_ends, dtype=torch.int32),
        "sequence_position_offsets": torch.tensor(position_offsets, dtype=torch.int64),
        "position_indices": torch.tensor(position_indices, dtype=torch.int32),
        "global_residuals": global_residuals,
        "position_site_hidden": torch.stack(sites).to(torch.float16) if sites else empty,
        "position_window_mean": torch.stack(window_means).to(torch.float16) if window_means else empty,
        "position_window_max": torch.stack(window_maxes).to(torch.float16) if window_maxes else empty,
        "raw_sequence_payload_written": 0,
        "development_test_record_count": 0,
        "evaluation_record_count": 0,
    }
    validate_edit_site_token_cache_v3(payload)
    return payload


def validate_edit_site_token_cache_v3(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == FEATURE_SCHEMA_VERSION, "unexpected cache schema")
    _require(payload.get("raw_sequence_payload_written") == 0, "raw sequence entered the cache")
    _require(payload.get("development_test_record_count") == 0, "Development TEST entered the cache")
    _require(payload.get("evaluation_record_count") == 0, "Evaluation entered the cache")
    record_count = len(payload["record_ids"])
    edit_offsets = payload["record_edit_offsets"]
    _require(edit_offsets.shape == (record_count + 1,), "record ragged offsets changed")
    edit_count = int(edit_offsets[-1].item())
    _require(edit_count == int(payload["edit_positions"].numel()), "ragged edit count changed")
    _require(edit_count == int(payload["edit_source_feature_indices"].numel()), "source edit references changed")
    _require(edit_count == int(payload["edit_candidate_feature_indices"].numel()), "candidate edit references changed")
    width = int(payload["embedding_width"])
    position_count = int(payload["position_indices"].numel())
    for key in ("position_site_hidden", "position_window_mean", "position_window_max"):
        _require(tuple(payload[key].shape) == (position_count, width), f"{key} geometry changed")
        _require(payload[key].dtype == torch.float16, f"{key} is not float16")
        _require(torch.isfinite(payload[key]).all().item(), f"{key} is nonfinite")
    _require(payload["global_residuals"].dtype == torch.float16, "global residual is not float16")
    if edit_count:
        _require(int(payload["edit_source_feature_indices"].min().item()) >= 0, "source feature index is negative")
        _require(int(payload["edit_candidate_feature_indices"].min().item()) >= 0, "candidate feature index is negative")
        _require(int(payload["edit_source_feature_indices"].max().item()) < position_count, "source feature index is out of range")
        _require(int(payload["edit_candidate_feature_indices"].max().item()) < position_count, "candidate feature index is out of range")


def load_edit_site_token_cache_v3(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "cache payload is not a mapping")
    validate_edit_site_token_cache_v3(payload)
    return payload
