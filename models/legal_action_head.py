"""Single-normalization legal action head including STOP."""
from __future__ import annotations

import torch
from torch import nn

from mrna_editflow.core.mixed_resolution_state import MixedAction


class LegalActionHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, max_position: int = 4096):
        super().__init__()
        self.kind = nn.Embedding(8, hidden_dim)
        self.position = nn.Embedding(max_position + 1, hidden_dim)
        self.value = nn.Embedding(68, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.kind_ids = {"STOP": 0, "UTR_SUB": 1, "UTR3_SUB": 2, "CDS_SYN_SWAP": 3}

    def forward(self, state_repr: torch.Tensor, actions: list[MixedAction]) -> torch.Tensor:
        if state_repr.ndim == 1:
            state_repr = state_repr.unsqueeze(0)
        k = torch.tensor([self.kind_ids.get(a.kind, 7) for a in actions], device=state_repr.device)
        p = torch.tensor([max(0, a.index + 1) for a in actions], device=state_repr.device)
        v = torch.tensor([0 if a.is_stop() else sum((ord(c) for c in a.value)) % 68 for a in actions], device=state_repr.device)
        action_repr = torch.cat([
            self.kind(k), self.position(p.clamp_max(self.position.num_embeddings - 1)), self.value(v),
            torch.ones((len(actions), self.kind.embedding_dim), device=state_repr.device),
        ], dim=-1)
        logits = self.proj(torch.cat([state_repr.expand(len(actions), -1), action_repr[:, : self.kind.embedding_dim * 3]], dim=-1))
        return logits.squeeze(-1)
