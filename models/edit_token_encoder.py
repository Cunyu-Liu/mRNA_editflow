"""Explicit edit-token and relative-to-CDS-position encoder."""
from __future__ import annotations

import torch
from torch import nn


class EditTokenEncoder(nn.Module):
    """Encode ``(region, absolute position, ref, alt, relative position)``.

    Four-column input remains supported for backwards-compatible smoke tests;
    in that case relative positions are zero and the model records that no
    explicit CDS-relative feature was supplied by the caller.
    """

    def __init__(self, hidden_dim: int = 128, max_position: int = 4096, max_relative_position: int = 4096):
        super().__init__()
        self.region = nn.Embedding(8, hidden_dim)
        self.position = nn.Embedding(max_position, hidden_dim)
        self.relative_position = nn.Embedding(max_relative_position * 2 + 1, hidden_dim)
        self.nucleotide = nn.Embedding(5, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, edit_tokens: torch.Tensor) -> torch.Tensor:
        if edit_tokens.ndim != 3 or edit_tokens.shape[-1] not in (4, 5):
            raise ValueError("edit_tokens must have shape [B, E, 4] or [B, E, 5]")
        x = edit_tokens.long()
        relative = x[..., 4] if x.shape[-1] == 5 else torch.zeros_like(x[..., 1])
        rel_limit = (self.relative_position.num_embeddings - 1) // 2
        region = self.region(x[..., 0].clamp_min(0).clamp_max(self.region.num_embeddings - 1))
        pos = self.position(x[..., 1].clamp_min(0).clamp_max(self.position.num_embeddings - 1))
        rel = self.relative_position((relative.clamp(-rel_limit, rel_limit) + rel_limit).long())
        ref = self.nucleotide(x[..., 2].clamp_min(0).clamp_max(4))
        alt = self.nucleotide(x[..., 3].clamp_min(0).clamp_max(4))
        token = self.proj(torch.cat([region, pos, rel, ref, alt], dim=-1))
        valid = (x[..., 0] >= 0).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1)
        return (token * valid).sum(dim=1) / denom


__all__ = ["EditTokenEncoder"]
