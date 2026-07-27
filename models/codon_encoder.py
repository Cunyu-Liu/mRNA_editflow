"""64-way codon encoder for the CDS resolution."""
from __future__ import annotations

import torch
from torch import nn

from mrna_editflow.core.constants import codon_to_index


class CodonEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, codon_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(codon_ids.long().clamp(0, 63)))

    @staticmethod
    def encode(codons: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor([codon_to_index(c) for c in codons], dtype=torch.long, device=device)
