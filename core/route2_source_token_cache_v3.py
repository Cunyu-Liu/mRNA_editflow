"""Raw-sequence-free source-side mRNABERT token cache for SetFlow V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SOURCE_TOKEN_CACHE_SCHEMA_V3 = "route_a_v3_route2_setflow_source_token_cache.v3"


class SourceTokenCacheV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceTokenCacheV3Error(message)


def assemble_source_token_cache_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    sequence_to_index: Mapping[str, int],
    encoded_tokens: Mapping[int, torch.Tensor],
    model_id: str,
    pretrained_parameter_count: int,
    attention_backend: str,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["canonical_record_id"]))
    sequence_count = len(sequence_to_index)
    _require(bool(ordered), "source-token cache rows are empty")
    _require(
        set(sequence_to_index.values()) == set(range(sequence_count)),
        "source sequence indices are not dense",
    )
    _require(set(encoded_tokens) == set(range(sequence_count)), "encoded source keys are incomplete")
    widths = {int(value.shape[1]) for value in encoded_tokens.values()}
    _require(len(widths) == 1, "source token widths differ")
    offsets = [0]
    token_rows: list[torch.Tensor] = []
    sequence_lengths: list[int] = []
    for index in range(sequence_count):
        hidden = encoded_tokens[index]
        _require(hidden.ndim == 2 and hidden.shape[0] > 0, "source token matrix is invalid")
        _require(torch.isfinite(hidden).all().item(), "source token matrix is nonfinite")
        sequence_lengths.append(int(hidden.shape[0]))
        token_rows.append(hidden.to(torch.float16))
        offsets.append(offsets[-1] + int(hidden.shape[0]))
    record_indices = [
        sequence_to_index[str(row["source_sequence"])] for row in ordered
    ]
    payload = {
        "schema_version": SOURCE_TOKEN_CACHE_SCHEMA_V3,
        "encoder_family": "mRNABERT",
        "model_id": str(model_id),
        "pretrained_parameter_count": int(pretrained_parameter_count),
        "embedding_width": widths.pop(),
        "attention_backend": str(attention_backend),
        "tokenization_policy": "UTR_SINGLE_NUCLEOTIDE_SPACE_SEPARATED_DNA_ALPHABET_ONE_LEADING_SPECIAL",
        "chunk_policy": "ONE_COMPLETE_CHUNK_MAXIMUM_1000_NUCLEOTIDES",
        "record_ids": [str(row["canonical_record_id"]) for row in ordered],
        "record_source_sequence_indices": torch.tensor(record_indices, dtype=torch.int64),
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.int32),
        "sequence_token_offsets": torch.tensor(offsets, dtype=torch.int64),
        "source_token_hidden": torch.cat(token_rows, dim=0),
        "raw_sequence_payload_written": 0,
        "development_test_record_count": 0,
        "evaluation_record_count": 0,
    }
    validate_source_token_cache_v3(payload)
    return payload


def validate_source_token_cache_v3(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == SOURCE_TOKEN_CACHE_SCHEMA_V3, "unexpected source-token cache schema")
    _require(payload.get("raw_sequence_payload_written") == 0, "raw source sequence entered the cache")
    _require(payload.get("development_test_record_count") == 0, "Development TEST entered the cache")
    _require(payload.get("evaluation_record_count") == 0, "Evaluation entered the cache")
    record_count = len(payload["record_ids"])
    _require(len(set(payload["record_ids"])) == record_count, "source-token cache record ids are duplicated")
    record_indices = payload["record_source_sequence_indices"]
    lengths = payload["sequence_lengths"]
    offsets = payload["sequence_token_offsets"]
    hidden = payload["source_token_hidden"]
    _require(record_indices.shape == (record_count,), "record source references changed")
    _require(offsets.shape == (lengths.numel() + 1,), "source token offsets changed")
    _require(int(offsets[0].item()) == 0, "source token offsets do not start at zero")
    _require(bool(torch.all(offsets[1:] >= offsets[:-1]).item()), "source token offsets are not monotonic")
    _require(
        torch.equal(offsets[1:] - offsets[:-1], lengths.to(torch.int64)),
        "source token lengths and offsets differ",
    )
    _require(hidden.dtype == torch.float16, "source token cache is not float16")
    _require(hidden.ndim == 2 and hidden.shape[1] == int(payload["embedding_width"]), "source token geometry changed")
    _require(int(offsets[-1].item()) == hidden.shape[0], "source token payload length changed")
    _require(torch.isfinite(hidden).all().item(), "source token cache is nonfinite")
    if record_count:
        _require(int(record_indices.min().item()) >= 0, "record source index is negative")
        _require(int(record_indices.max().item()) < lengths.numel(), "record source index is out of range")


class SourceTokenCacheIndexV3:
    """Record-aligned access without reconstructing or storing source strings."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        validate_source_token_cache_v3(payload)
        self.payload = payload
        self.record_to_row = {
            str(record_id): index
            for index, record_id in enumerate(payload["record_ids"])
        }

    def tokens_for_record(self, record_id: str) -> torch.Tensor:
        _require(record_id in self.record_to_row, "record is absent from source-token cache")
        row = self.record_to_row[record_id]
        sequence_index = int(
            self.payload["record_source_sequence_indices"][row].item()
        )
        offsets = self.payload["sequence_token_offsets"]
        start = int(offsets[sequence_index].item())
        end = int(offsets[sequence_index + 1].item())
        return self.payload["source_token_hidden"][start:end]


def load_source_token_cache_v3(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "source-token cache payload is not a mapping")
    validate_source_token_cache_v3(payload)
    return payload
