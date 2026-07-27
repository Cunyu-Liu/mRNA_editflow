"""Explicit edit-token encoder for source-relative local-effect prediction."""
from __future__ import annotations

import torch
from torch import nn


class EditTokenEncoder(nn.Module):
    """Encode ``(region, position, ref, alt)`` tokens and pool per pair."""

    def __init__(self, hidden_dim: int = 128, max_position: int = 4096):
        super().__init__()
        self.region = nn.Embedding(8, hidden_dim)
        self.position = nn.Embedding(max_position, hidden_dim)
        self.nucleotide = nn.Embedding(5, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, edit_tokens: torch.Tensor) -> torch.Tensor:
        # edit_tokens: [B, E, 4], with -1 used for padding.
        x = edit_tokens.long().clamp_min(-1)
        region = self.region(x[..., 0].clamp_min(0))
        pos = self.position(x[..., 1].clamp_min(0).clamp_max(self.position.num_embeddings - 1))
        ref = self.nucleotide(x[..., 2].clamp_min(0).clamp_max(4))
        alt = self.nucleotide(x[..., 3].clamp_min(0).clamp_max(4))
        token = self.proj(torch.cat([region, pos, ref, alt], dim=-1))
        valid = (x[..., 0] >= 0).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1)
        return (token * valid).sum(dim=1) / denom
