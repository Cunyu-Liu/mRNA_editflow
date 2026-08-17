"""Frozen mRNABERT sequence encoder for candidates not present in a feature cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


ATTENTION_BACKENDS = {
    "OFFICIAL_PYTORCH_FALLBACK",
    "PYTORCH_SDPA_AUTO",
}


class OnlineEncoderError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OnlineEncoderError(message)


def normalize_rna(sequence: str) -> str:
    value = str(sequence).upper().replace("T", "U")
    _require(bool(value) and set(value) <= set("ACGUN"), "unsupported RNA sequence")
    return value


def sequence_chunks(sequence: str, maximum_nucleotides: int) -> list[str]:
    _require(maximum_nucleotides > 0, "maximum chunk length must be positive")
    return [
        sequence[start : start + maximum_nucleotides]
        for start in range(0, len(sequence), maximum_nucleotides)
    ]


def format_utr_chunk(chunk: str) -> str:
    dna = normalize_rna(chunk).replace("U", "T")
    return " ".join(dna)


def pool_last_hidden(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    weights = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def pytorch_sdpa_qkvpacked(
    qkv: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Match mRNABERT's packed-QKV attention interface with PyTorch SDPA."""

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


def chunk_batches(
    chunks: list[tuple[str, str]], maximum_sequences: int, token_budget: int
) -> Iterable[list[tuple[str, str]]]:
    _require(maximum_sequences > 0 and token_budget > 0, "batch limits must be positive")
    ordered = sorted(chunks, key=lambda value: (len(value[1]), value[0]))
    current: list[tuple[str, str]] = []
    longest = 0
    for item in ordered:
        proposed_longest = max(longest, len(item[1]) + 2)
        if current and (
            len(current) >= maximum_sequences
            or proposed_longest * (len(current) + 1) > token_budget
        ):
            yield current
            current = []
            longest = 0
        current.append(item)
        longest = max(longest, len(item[1]) + 2)
    if current:
        yield current


class FrozenMRNABERTOnlineEncoder:
    """Encode arbitrary RNA sequences with frozen official mRNABERT weights."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        maximum_chunk_nucleotides: int = 1000,
        maximum_sequences_per_batch: int = 8,
        batch_token_budget: int = 4096,
        attention_backend: str = "OFFICIAL_PYTORCH_FALLBACK",
    ) -> None:
        _require(device.type == "cuda", "online mRNABERT encoding requires CUDA")
        _require(model_path.is_dir(), "mRNABERT model directory is absent")
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise OnlineEncoderError("Transformers is unavailable for mRNABERT") from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        model_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        model = AutoModel.from_config(
            model_config, trust_remote_code=True, add_pooling_layer=False
        )
        modeling_module = sys.modules[model.__class__.__module__]
        backend = str(attention_backend)
        _require(backend in ATTENTION_BACKENDS, "unknown mRNABERT attention backend")
        modeling_module.flash_attn_qkvpacked_func = (
            None
            if backend == "OFFICIAL_PYTORCH_FALLBACK"
            else pytorch_sdpa_qkvpacked
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
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.attention_backend = backend
        self.parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.embedding_width = int(self.model.config.hidden_size)
        _require(self.parameter_count > 100_000_000, "mRNABERT parameter geometry changed")
        _require(self.embedding_width == 768, "mRNABERT hidden width changed")
        self.maximum_chunk_nucleotides = int(maximum_chunk_nucleotides)
        _require(
            self.maximum_chunk_nucleotides + 2
            <= int(self.tokenizer.model_max_length),
            "mRNABERT chunk exceeds tokenizer maximum length",
        )
        self.maximum_sequences_per_batch = int(maximum_sequences_per_batch)
        self.batch_token_budget = int(batch_token_budget)
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def cached_sequence_count(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def encode_sequences(self, sequences: Iterable[str]) -> torch.Tensor:
        ordered = [normalize_rna(sequence) for sequence in sequences]
        _require(bool(ordered), "no sequences were supplied")
        missing = sorted(set(ordered) - set(self._cache))
        chunks = [
            (sequence, chunk)
            for sequence in missing
            for chunk in sequence_chunks(sequence, self.maximum_chunk_nucleotides)
        ]
        sums: dict[str, torch.Tensor] = {}
        lengths: dict[str, int] = {}
        with torch.inference_mode():
            for batch in chunk_batches(
                chunks,
                self.maximum_sequences_per_batch,
                self.batch_token_budget,
            ):
                tokenized = self.tokenizer(
                    [format_utr_chunk(chunk) for _sequence, chunk in batch],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                tokenized = {
                    key: value.to(self.device) for key, value in tokenized.items()
                }
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    last_hidden_state = self.model(**tokenized)[0]
                    pooled = pool_last_hidden(
                        last_hidden_state, tokenized["attention_mask"]
                    )
                pooled = pooled.float().cpu()
                _require(torch.isfinite(pooled).all().item(), "online embedding is nonfinite")
                for (sequence, chunk), embedding in zip(batch, pooled):
                    weight = len(chunk)
                    sums[sequence] = (
                        sums.get(sequence, torch.zeros_like(embedding))
                        + weight * embedding
                    )
                    lengths[sequence] = lengths.get(sequence, 0) + weight
        for sequence in missing:
            self._cache[sequence] = sums[sequence] / lengths[sequence]
        return torch.stack([self._cache[sequence] for sequence in ordered])
