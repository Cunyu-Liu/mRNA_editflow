"""Hash-stable categorical context encoder with no test-set vocabulary fitting."""
from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, buckets: int = 4096):
        super().__init__()
        self.buckets = buckets
        self.embeddings = nn.ModuleList([nn.Embedding(buckets, hidden_dim) for _ in range(3)])
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim != 2 or context_ids.shape[1] != 3:
            raise ValueError("context_ids must have shape [B, 3] for cargo/cell/assay")
        parts = [emb(context_ids[:, i].long().remainder(self.buckets)) for i, emb in enumerate(self.embeddings)]
        return self.proj(torch.cat(parts, dim=-1))


def context_tensor(rows: Iterable[dict], device: torch.device) -> torch.Tensor:
    values = [[
        stable_bucket(row.get("cargo_id", ""), 4096),
        stable_bucket(row.get("cell_context", ""), 4096),
        stable_bucket(row.get("assay_type", ""), 4096),
    ] for row in rows]
    return torch.tensor(values, dtype=torch.long, device=device)
