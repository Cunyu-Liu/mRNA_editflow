"""Shared frozen bottom-six mRNABERT encoder for V4 cache and online paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch import nn

from core.route2_bottom_encoder_chunk_cache_v4 import (
    BottomEncodedChunkV4,
    BottomEncodedSequenceV4,
)
from core.route2_mrnabert_edit_site_features_v3 import (
    CHUNK_NUCLEOTIDES,
    CHUNK_OVERLAP,
    ChunkSpan,
    format_utr_chunk,
    legacy_global_chunk_spans,
    official_masked_chunk_mean,
    overlapping_chunk_spans,
    validate_token_layout,
)
from scripts.route_a_v3.route2_mrnabert_edit_site_encoder_v3 import (
    ATTENTION_BACKENDS,
    pytorch_sdpa_qkvpacked,
)


class MRNABERTBottomSixEncoderV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MRNABERTBottomSixEncoderV4Error(message)


def forward_bottom_six_hidden_v4(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> torch.Tensor:
    """Run embeddings and exactly encoder blocks 0–5.

    This helper is the sole bottom-six forward used by both cache construction
    and future online encoding.  It deliberately does not execute blocks 6–11.
    """

    _require(input_ids.ndim == attention_mask.ndim == token_type_ids.ndim == 2, "bottom-six model inputs must be matrices")
    _require(input_ids.shape == attention_mask.shape == token_type_ids.shape, "bottom-six model input geometry differs")
    _require(hasattr(model, "embeddings") and hasattr(model, "encoder"), "mRNABERT lacks BERT embeddings or encoder")
    layers = model.encoder.layer
    _require(len(layers) == 12, "mRNABERT encoder depth changed")
    hidden = model.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
    )
    extended_attention_mask = model.get_extended_attention_mask(
        attention_mask,
        input_ids.shape,
        dtype=hidden.dtype,
    )
    head_mask = model.get_head_mask(None, len(layers))
    for layer_index, layer in enumerate(layers[:6]):
        output = layer(
            hidden,
            attention_mask=extended_attention_mask,
            head_mask=head_mask[layer_index],
            output_attentions=False,
        )
        hidden = output[0]
    _require(hidden.shape[:2] == input_ids.shape, "bottom-six hidden token geometry changed")
    _require(torch.isfinite(hidden).all().item(), "bottom-six hidden is nonfinite")
    return hidden


@dataclass(frozen=True, order=True)
class _ChunkRequestV4:
    sequence_index: int
    span: ChunkSpan


def request_batches_v4(
    requests: Iterable[_ChunkRequestV4],
    *,
    maximum_sequences: int,
    token_budget: int,
) -> list[list[_ChunkRequestV4]]:
    """Deterministically pack cache/online chunks under a token budget."""

    _require(maximum_sequences > 0 and token_budget > 0, "bottom-six batch limits must be positive")
    ordered = sorted(requests, key=lambda item: (item.span.length, item.sequence_index, item.span.start))
    batches: list[list[_ChunkRequestV4]] = []
    current: list[_ChunkRequestV4] = []
    longest = 0
    for request in ordered:
        proposed_longest = max(longest, request.span.length + 2)
        if current and (
            len(current) >= maximum_sequences
            or proposed_longest * (len(current) + 1) > token_budget
        ):
            batches.append(current)
            current = []
            longest = 0
        current.append(request)
        longest = max(longest, request.span.length + 2)
    if current:
        batches.append(current)
    return batches


def compare_bottom_encoded_sequences_v4(
    cached: Mapping[int, BottomEncodedSequenceV4],
    online: Mapping[int, BottomEncodedSequenceV4],
    *,
    maximum_absolute_tolerance: float,
    mean_absolute_tolerance: float,
) -> dict[str, float | int | bool]:
    """Compare cache and online bottom-six outputs under frozen tolerances."""

    _require(set(cached) == set(online), "cache/online sequence ids differ")
    _require(maximum_absolute_tolerance >= 0 and mean_absolute_tolerance >= 0, "cache/online tolerances must be nonnegative")
    absolute_differences: list[torch.Tensor] = []
    for sequence_index in sorted(cached):
        left = cached[sequence_index]
        right = online[sequence_index]
        _require(tuple(chunk.span for chunk in left.chunks) == tuple(chunk.span for chunk in right.chunks), "cache/online chunk spans differ")
        _require(len(left.chunks) == len(right.chunks), "cache/online chunk count differs")
        absolute_differences.append((left.global_residual.float() - right.global_residual.float()).abs().reshape(-1))
        for cached_chunk, online_chunk in zip(left.chunks, right.chunks, strict=True):
            _require(cached_chunk.special_token_offset == online_chunk.special_token_offset, "cache/online special-token offset differs")
            _require(torch.equal(cached_chunk.attention_mask, online_chunk.attention_mask), "cache/online attention mask differs")
            _require(cached_chunk.hidden.shape == online_chunk.hidden.shape, "cache/online hidden geometry differs")
            absolute_differences.append((cached_chunk.hidden.float() - online_chunk.hidden.float()).abs().reshape(-1))
    values = torch.cat(absolute_differences) if absolute_differences else torch.zeros(1)
    maximum = float(values.max().item())
    mean = float(values.mean().item())
    return {
        "sequence_count": len(cached),
        "compared_value_count": int(values.numel()),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": mean,
        "maximum_absolute_tolerance": float(maximum_absolute_tolerance),
        "mean_absolute_tolerance": float(mean_absolute_tolerance),
        "passed": maximum <= maximum_absolute_tolerance and mean <= mean_absolute_tolerance,
    }


class FrozenMRNABERTBottomSixEncoderV4:
    """CUDA-only bottom-six encoder with one cache/online implementation."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        maximum_sequences_per_batch: int = 8,
        batch_token_budget: int = 4096,
        attention_backend: str = "PYTORCH_SDPA_AUTO",
    ) -> None:
        _require(device.type == "cuda", "bottom-six mRNABERT encoding requires CUDA")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(model_path.is_dir(), "mRNABERT model directory is absent")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise MRNABERTBottomSixEncoderV4Error("Transformers is unavailable") from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        configuration = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModel.from_config(
            configuration,
            trust_remote_code=True,
            add_pooling_layer=False,
        )
        backend = str(attention_backend)
        _require(backend in ATTENTION_BACKENDS, "unknown mRNABERT attention backend")
        modeling_module = sys.modules[model.__class__.__module__]
        modeling_module.flash_attn_qkvpacked_func = (
            None if backend == "OFFICIAL_PYTORCH_FALLBACK" else pytorch_sdpa_qkvpacked
        )
        checkpoint = torch.load(
            model_path / "pytorch_model.bin",
            map_location="cpu",
            weights_only=False,
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
        self.tokenizer = tokenizer
        self.device = device
        self.attention_backend = backend
        self.maximum_sequences_per_batch = int(maximum_sequences_per_batch)
        self.batch_token_budget = int(batch_token_budget)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.embedding_width = int(self.model.config.hidden_size)
        _require(self.parameter_count > 100_000_000, "mRNABERT parameter geometry changed")
        _require(self.embedding_width == 768, "mRNABERT hidden width changed")
        _require(CHUNK_NUCLEOTIDES + 2 <= int(tokenizer.model_max_length), "V4 chunk exceeds tokenizer maximum length")

    def encode_sequences(
        self,
        sequences: Mapping[int, str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, BottomEncodedSequenceV4]:
        """Encode overlapping local chunks and a legacy-compatible global mean."""

        _require(bool(sequences), "no sequences were supplied")
        local_spans = {
            index: tuple(
                overlapping_chunk_spans(
                    len(sequence),
                    chunk_nucleotides=CHUNK_NUCLEOTIDES,
                    overlap=CHUNK_OVERLAP,
                )
            )
            for index, sequence in sequences.items()
        }
        global_spans = {
            index: tuple(
                legacy_global_chunk_spans(
                    len(sequence),
                    chunk_nucleotides=CHUNK_NUCLEOTIDES,
                )
            )
            for index, sequence in sequences.items()
        }
        requests = {
            _ChunkRequestV4(index, span)
            for index in sequences
            for span in {*local_spans[index], *global_spans[index]}
        }
        batches = request_batches_v4(
            requests,
            maximum_sequences=self.maximum_sequences_per_batch,
            token_budget=self.batch_token_budget,
        )
        local_hidden: dict[tuple[int, ChunkSpan], tuple[torch.Tensor, torch.Tensor]] = {}
        global_sums: dict[int, torch.Tensor] = {}
        global_lengths: dict[int, int] = {}
        global_span_sets = {index: set(spans) for index, spans in global_spans.items()}
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches, start=1):
                chunks = [
                    sequences[request.sequence_index][request.span.start : request.span.end]
                    for request in batch
                ]
                tokenized = self.tokenizer(
                    [format_utr_chunk(chunk) for chunk in chunks],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
                token_type_ids = tokenized.get("token_type_ids")
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(tokenized["input_ids"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = forward_bottom_six_hidden_v4(
                        self.model,
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        token_type_ids=token_type_ids,
                    )
                for row_index, request in enumerate(batch):
                    mask = tokenized["attention_mask"][row_index]
                    validate_token_layout(mask, chunk_length=request.span.length)
                    active_token_count = int(mask.sum().item())
                    active_hidden = hidden[row_index, :active_token_count].float().cpu()
                    active_mask = mask[:active_token_count].to(torch.bool).cpu()
                    key = request.sequence_index
                    if request.span in set(local_spans[key]):
                        local_hidden[(key, request.span)] = (active_hidden, active_mask)
                    if request.span in global_span_sets[key]:
                        pooled = official_masked_chunk_mean(active_hidden, active_mask)
                        weight = request.span.length
                        global_sums[key] = pooled * weight if key not in global_sums else global_sums[key] + pooled * weight
                        global_lengths[key] = global_lengths.get(key, 0) + weight
                if progress_callback is not None:
                    progress_callback(batch_index, len(batches))
        result: dict[int, BottomEncodedSequenceV4] = {}
        for index in sorted(sequences):
            _require(global_lengths.get(index) == len(sequences[index]), "global bottom-six residual coverage changed")
            chunks = tuple(
                BottomEncodedChunkV4(
                    span=span,
                    hidden=local_hidden[(index, span)][0],
                    attention_mask=local_hidden[(index, span)][1],
                )
                for span in local_spans[index]
            )
            result[index] = BottomEncodedSequenceV4(
                chunks=chunks,
                global_residual=global_sums[index] / global_lengths[index],
            )
        _require(set(result) == set(sequences), "bottom-six sequence encoding is incomplete")
        return result

    def encode_online(
        self,
        sequences: Mapping[int, str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, BottomEncodedSequenceV4]:
        """Future-sequence path; intentionally identical to cache encoding."""

        return self.encode_sequences(sequences, progress_callback=progress_callback)
