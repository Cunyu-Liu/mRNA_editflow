"""Source/cargo/cell/assay context features for the paired local-delta model.

Categorical IDs are hash-stable so a final-test vocabulary can never be fit by
accident. Optional continuous vectors are accepted explicitly; the benchmark
derives deterministic provenance-preserving vectors when no measured embedding
is available. These are a fallback representation, not biological evidence.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import torch
from torch import nn


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def stable_vector(value: str, dim: int = 16) -> list[float]:
    """Return a deterministic identifier feature, not a biological embedding."""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 127.5 - 1.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]


class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, buckets: int = 4096, feature_dim: int = 16):
        super().__init__()
        self.buckets = buckets
        self.feature_dim = feature_dim
        self.embeddings = nn.ModuleList([nn.Embedding(buckets, hidden_dim) for _ in range(3)])
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.protein_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.cell_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.assay_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        context_ids: torch.Tensor,
        protein_embedding: Optional[torch.Tensor] = None,
        cell_embedding: Optional[torch.Tensor] = None,
        assay_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_ids.ndim != 2 or context_ids.shape[1] != 3:
            raise ValueError("context_ids must have shape [B, 3] for cargo/cell/assay")
        parts = [emb(context_ids[:, i].long().remainder(self.buckets)) for i, emb in enumerate(self.embeddings)]
        out = self.proj(torch.cat(parts, dim=-1))
        if protein_embedding is not None:
            out = out + self.protein_proj(protein_embedding.float())
        if cell_embedding is not None:
            out = out + self.cell_proj(cell_embedding.float())
        if assay_embedding is not None:
            out = out + self.assay_proj(assay_embedding.float())
        return out


def context_tensor(rows: Iterable[dict], device: torch.device) -> torch.Tensor:
    values = [[
        stable_bucket(row.get("cargo_id", ""), 4096),
        stable_bucket(row.get("cell_context", ""), 4096),
        stable_bucket(row.get("assay_type", ""), 4096),
    ] for row in rows]
    return torch.tensor(values, dtype=torch.long, device=device)


def context_feature_tensors(
    rows: Iterable[dict], device: torch.device, feature_dim: int = 16
) -> dict[str, torch.Tensor]:
    """Build categorical IDs and explicit cargo/cell/assay feature tensors."""
    rows = list(rows)

    def vector(row: dict, key: str, fallback: str) -> list[float]:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == feature_dim:
            return [float(x) for x in value]
        return stable_vector(str(row.get(fallback, "")), feature_dim)

    return {
        "context_ids": context_tensor(rows, device),
        "protein_embedding": torch.tensor(
            [vector(row, "protein_embedding", "protein_family_id") for row in rows],
            dtype=torch.float32, device=device,
        ),
        "cell_embedding": torch.tensor(
            [vector(row, "cell_embedding", "cell_context") for row in rows],
            dtype=torch.float32, device=device,
        ),
        "assay_embedding": torch.tensor(
            [vector(row, "assay_embedding", "assay_type") for row in rows],
            dtype=torch.float32, device=device,
        ),
    }


__all__ = ["ContextEncoder", "context_tensor", "context_feature_tensors", "stable_bucket", "stable_vector"]
