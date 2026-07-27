"""PairedDeltaFormer: source/candidate difference model for local effects."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mrna_editflow.models.context_encoder import ContextEncoder
from mrna_editflow.models.edit_token_encoder import EditTokenEncoder
from mrna_editflow.models.uncertainty_head import UncertaintyHead


class _SequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int = 5, hidden_dim: int = 128, layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=4)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding = None if mask is None else ~mask.bool()
        x = self.embedding(tokens.long())
        x = self.encoder(x, src_key_padding_mask=key_padding)
        if mask is None:
            return self.norm(x.mean(dim=1))
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.norm((x * mask.unsqueeze(-1)).sum(dim=1) / denom)


class PairedDeltaFormer(nn.Module):
    """Predict relative delta plus uncertainty and ranking signals.

    The model intentionally receives both source and candidate sequences and
    explicit edit tokens.  It cannot silently degrade to an absolute-property
    predictor when the source changes.
    """

    def __init__(self, hidden_dim: int = 128, layers: int = 2):
        super().__init__()
        self.source_encoder = _SequenceEncoder(hidden_dim=hidden_dim, layers=layers)
        self.candidate_encoder = _SequenceEncoder(hidden_dim=hidden_dim, layers=layers)
        self.edit_encoder = EditTokenEncoder(hidden_dim=hidden_dim)
        self.context_encoder = ContextEncoder(hidden_dim=hidden_dim)
        self.source_value_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.head = UncertaintyHead(hidden_dim)

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        edit_tokens: torch.Tensor,
        context_ids: torch.Tensor,
        source_value: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        src = self.source_encoder(source_tokens, source_mask)
        cand = self.candidate_encoder(candidate_tokens, candidate_mask)
        edit = self.edit_encoder(edit_tokens)
        context = self.context_encoder(context_ids)
        value = self.source_value_proj(source_value.float().reshape(-1, 1))
        fused = self.fusion(torch.cat([src, cand, cand - src, edit, context + value], dim=-1))
        return self.head(fused)


__all__ = ["PairedDeltaFormer"]
