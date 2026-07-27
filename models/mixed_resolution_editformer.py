"""Mixed-resolution mRNA Edit Flow policy backbone."""
from __future__ import annotations

import torch
from torch import nn

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState, legal_actions
from mrna_editflow.models.codon_encoder import CodonEncoder
from mrna_editflow.models.cross_region_attention import CrossRegionAttention
from mrna_editflow.models.legal_action_head import LegalActionHead


class MixedResolutionEditFormer(nn.Module):
    """Encode UTRs at nucleotide resolution and CDS at codon resolution.

    `log_probs` receives an explicit enumerated legal action list and applies
    one softmax over STOP + all legal edits.  This is the auditable policy
    contract; no position/target factorization is used for probability claims.
    """

    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.nuc_embedding = nn.Embedding(4, hidden_dim)
        self.codon_encoder = CodonEncoder(hidden_dim)
        self.trunk = CrossRegionAttention(hidden_dim, heads)
        self.global_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.action_head = LegalActionHead(hidden_dim)

    def encode_state(self, state: MixedResolutionState) -> torch.Tensor:
        device = next(self.parameters()).device
        nuc = torch.tensor([["ACGU".index(c) for c in state.five_utr + state.three_utr]], device=device)
        utr = self.nuc_embedding(nuc)
        cds = self.codon_encoder(CodonEncoder.encode(list(state.codons), device)).unsqueeze(0)
        tokens = torch.cat([utr, cds], dim=1)
        shared = self.trunk(tokens)
        return self.global_proj(shared.mean(dim=1)).squeeze(0)

    def logits(self, state: MixedResolutionState, actions: list[MixedAction] | None = None) -> tuple[torch.Tensor, list[MixedAction]]:
        acts = legal_actions(state) if actions is None else list(actions)
        return self.action_head(self.encode_state(state), acts), acts

    def log_probs(self, state: MixedResolutionState, actions: list[MixedAction] | None = None) -> tuple[torch.Tensor, list[MixedAction]]:
        logits, acts = self.logits(state, actions)
        return torch.log_softmax(logits, dim=0), acts

    def forward(self, state: MixedResolutionState, actions: list[MixedAction] | None = None) -> dict:
        logits, acts = self.logits(state, actions)
        return {"logits": logits, "log_probs": torch.log_softmax(logits, dim=0), "actions": acts}


__all__ = ["MixedResolutionEditFormer"]
