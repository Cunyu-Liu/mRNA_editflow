"""Frozen mRNABERT encoder shared by the V3 cache and online feature paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F

from core.route2_mrnabert_edit_site_features_v3 import (
    CHUNK_NUCLEOTIDES,
    CHUNK_OVERLAP,
    LOCAL_RADIUS,
    ChunkSpan,
    EncodedSequenceFeaturesV3,
    PositionFeature,
    extract_position_feature,
    extract_nucleotide_token_hidden,
    format_utr_chunk,
    legacy_global_chunk_spans,
    official_masked_chunk_mean,
    overlapping_chunk_spans,
    select_most_centered_chunk,
    validate_token_layout,
)


ATTENTION_BACKENDS = {"OFFICIAL_PYTORCH_FALLBACK", "PYTORCH_SDPA_AUTO"}


class EditSiteEncoderError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EditSiteEncoderError(message)


def pytorch_sdpa_qkvpacked(qkv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _require(
        qkv.ndim == 5 and qkv.shape[2] == 3,
        "packed mRNABERT QKV must have shape batch x length x 3 x heads x dim",
    )
    query = qkv[:, :, 0].permute(0, 2, 1, 3)
    key = qkv[:, :, 1].permute(0, 2, 1, 3)
    value = qkv[:, :, 2].permute(0, 2, 1, 3)
    attention = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=bias.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
    )
    return attention.permute(0, 2, 1, 3)


@dataclass(frozen=True, order=True)
class _ChunkRequest:
    sequence_key: int
    span: ChunkSpan


def _request_batches(
    requests: Iterable[_ChunkRequest],
    *,
    maximum_sequences: int,
    token_budget: int,
) -> Iterable[list[_ChunkRequest]]:
    _require(maximum_sequences > 0 and token_budget > 0, "batch limits must be positive")
    ordered = sorted(requests, key=lambda item: (item.span.length, item.sequence_key, item.span.start))
    current: list[_ChunkRequest] = []
    longest = 0
    for item in ordered:
        proposed_longest = max(longest, item.span.length + 2)
        if current and (
            len(current) >= maximum_sequences
            or proposed_longest * (len(current) + 1) > token_budget
        ):
            yield current
            current = []
            longest = 0
        current.append(item)
        longest = max(longest, item.span.length + 2)
    if current:
        yield current


class FrozenMRNABERTEditSiteEncoderV3:
    """Encode requested edit sites without retaining raw sequence in artifacts."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        chunk_nucleotides: int = CHUNK_NUCLEOTIDES,
        chunk_overlap: int = CHUNK_OVERLAP,
        local_radius: int = LOCAL_RADIUS,
        maximum_sequences_per_batch: int = 8,
        batch_token_budget: int = 4096,
        attention_backend: str = "OFFICIAL_PYTORCH_FALLBACK",
    ) -> None:
        _require(device.type == "cuda", "mRNABERT edit-site encoding requires CUDA")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(model_path.is_dir(), "mRNABERT model directory is absent")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EditSiteEncoderError("Transformers is unavailable for mRNABERT") from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = AutoModel.from_config(
            model_config, trust_remote_code=True, add_pooling_layer=False
        )
        backend = str(attention_backend)
        _require(backend in ATTENTION_BACKENDS, "unknown mRNABERT attention backend")
        modeling_module = sys.modules[model.__class__.__module__]
        modeling_module.flash_attn_qkvpacked_func = (
            None if backend == "OFFICIAL_PYTORCH_FALLBACK" else pytorch_sdpa_qkvpacked
        )
        checkpoint = torch.load(
            model_path / "pytorch_model.bin", map_location="cpu", weights_only=False
        )
        base_state = {
            key.removeprefix("bert."): value
            for key, value in checkpoint.items()
            if key.startswith("bert.")
        }
        _require(bool(base_state), "mRNABERT base encoder weights are absent")
        model.load_state_dict(base_state, strict=True)
        del checkpoint, base_state
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.attention_backend = backend
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.embedding_width = int(self.model.config.hidden_size)
        _require(self.parameter_count > 100_000_000, "mRNABERT parameter geometry changed")
        _require(self.embedding_width == 768, "mRNABERT hidden width changed")
        self.chunk_nucleotides = int(chunk_nucleotides)
        self.chunk_overlap = int(chunk_overlap)
        self.local_radius = int(local_radius)
        _require(
            self.chunk_nucleotides + 2 <= int(self.tokenizer.model_max_length),
            "mRNABERT chunk exceeds tokenizer maximum length",
        )
        self.maximum_sequences_per_batch = int(maximum_sequences_per_batch)
        self.batch_token_budget = int(batch_token_budget)

    def encode_full_nucleotide_tokens(
        self,
        sequences: Mapping[int, str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Encode one source-side hidden state per nucleotide.

        The frozen SetFlow V3 cohort has maximum source length 837, so its
        source-token cache deliberately uses one complete chunk per source.
        """

        _require(bool(sequences), "no sequences were supplied")
        _require(
            max(map(len, sequences.values())) <= self.chunk_nucleotides,
            "SetFlow source exceeds the prospectively frozen one-chunk boundary",
        )
        requests = [
            _ChunkRequest(key, ChunkSpan(0, len(sequence)))
            for key, sequence in sequences.items()
        ]
        batches = list(
            _request_batches(
                requests,
                maximum_sequences=self.maximum_sequences_per_batch,
                token_budget=self.batch_token_budget,
            )
        )
        result: dict[int, torch.Tensor] = {}
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches, start=1):
                chunks = [sequences[item.sequence_key] for item in batch]
                tokenized = self.tokenizer(
                    [format_utr_chunk(chunk) for chunk in chunks],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(self.device) for key, value in tokenized.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    last_hidden_state = self.model(**tokenized)[0]
                for row_index, item in enumerate(batch):
                    hidden = extract_nucleotide_token_hidden(
                        last_hidden_state[row_index],
                        tokenized["attention_mask"][row_index],
                        chunk_length=item.span.length,
                    )
                    result[item.sequence_key] = hidden.float().cpu()
                if progress_callback is not None:
                    progress_callback(batch_index, len(batches))
        _require(set(result) == set(sequences), "source-token encoding is incomplete")
        return result

    def encode_requested_features(
        self,
        sequences: Mapping[int, str],
        positions_by_sequence: Mapping[int, Iterable[int]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, EncodedSequenceFeaturesV3]:
        """Encode unique sequences and only the local positions actually requested."""

        _require(bool(sequences), "no sequences were supplied")
        _require(set(sequences) == set(positions_by_sequence), "sequence and position keys differ")
        local_spans: dict[int, list[ChunkSpan]] = {}
        global_spans: dict[int, list[ChunkSpan]] = {}
        selected_positions: dict[tuple[int, ChunkSpan], list[int]] = {}
        requests: set[_ChunkRequest] = set()
        for key, sequence in sequences.items():
            length = len(sequence)
            local_spans[key] = overlapping_chunk_spans(
                length,
                chunk_nucleotides=self.chunk_nucleotides,
                overlap=self.chunk_overlap,
            )
            global_spans[key] = legacy_global_chunk_spans(
                length, chunk_nucleotides=self.chunk_nucleotides
            )
            for span in {*local_spans[key], *global_spans[key]}:
                requests.add(_ChunkRequest(key, span))
            for position in sorted(set(int(value) for value in positions_by_sequence[key])):
                span = select_most_centered_chunk(
                    position,
                    local_spans[key],
                    sequence_length=length,
                    local_radius=self.local_radius,
                )
                selected_positions.setdefault((key, span), []).append(position)

        position_features: dict[int, dict[int, PositionFeature]] = {
            key: {} for key in sequences
        }
        global_sums: dict[int, torch.Tensor] = {}
        global_lengths: dict[int, int] = {}
        global_span_sets = {key: set(spans) for key, spans in global_spans.items()}
        batches = list(
            _request_batches(
                requests,
                maximum_sequences=self.maximum_sequences_per_batch,
                token_budget=self.batch_token_budget,
            )
        )
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches, start=1):
                chunks = [
                    sequences[item.sequence_key][item.span.start : item.span.end]
                    for item in batch
                ]
                tokenized = self.tokenizer(
                    [format_utr_chunk(chunk) for chunk in chunks],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    last_hidden_state = self.model(**tokenized)[0]
                for row_index, item in enumerate(batch):
                    mask = tokenized["attention_mask"][row_index]
                    hidden = last_hidden_state[row_index]
                    validate_token_layout(mask, chunk_length=item.span.length)
                    key = item.sequence_key
                    if item.span in global_span_sets[key]:
                        pooled = official_masked_chunk_mean(hidden, mask).float().cpu()
                        weight = item.span.length
                        global_sums[key] = (
                            pooled * weight
                            if key not in global_sums
                            else global_sums[key] + pooled * weight
                        )
                        global_lengths[key] = global_lengths.get(key, 0) + weight
                    for position in selected_positions.get((key, item.span), []):
                        feature = extract_position_feature(
                            hidden,
                            mask,
                            span=item.span,
                            sequence_length=len(sequences[key]),
                            position=position,
                            local_radius=self.local_radius,
                        )
                        position_features[key][position] = PositionFeature(
                            site=feature.site.float().cpu(),
                            window_mean=feature.window_mean.float().cpu(),
                            window_max=feature.window_max.float().cpu(),
                            chunk=feature.chunk,
                        )
                if progress_callback is not None:
                    progress_callback(batch_index, len(batches))

        result: dict[int, EncodedSequenceFeaturesV3] = {}
        for key in sorted(sequences):
            requested = set(int(value) for value in positions_by_sequence[key])
            _require(set(position_features[key]) == requested, "an edit position was not encoded")
            _require(global_lengths.get(key) == len(sequences[key]), "global residual coverage changed")
            global_residual = global_sums[key] / global_lengths[key]
            _require(torch.isfinite(global_residual).all().item(), "global residual is nonfinite")
            result[key] = EncodedSequenceFeaturesV3(
                global_residual=global_residual,
                positions=position_features[key],
                local_chunk_spans=tuple(local_spans[key]),
            )
        return result
