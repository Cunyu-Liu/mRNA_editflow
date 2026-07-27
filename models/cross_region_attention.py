"""Shared trunk for nucleotide and codon regions."""
from __future__ import annotations

import torch
from torch import nn


class CrossRegionAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + self.ffn(attended))
