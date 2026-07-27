"""Policy over the complete enumerated legal action set."""
from __future__ import annotations

import torch
from torch import nn

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState
from mrna_editflow.models.mixed_resolution_editformer import MixedResolutionEditFormer


class LegalActionPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.backbone = MixedResolutionEditFormer(hidden_dim=hidden_dim)

    def log_probs(self, state: MixedResolutionState, actions: list[MixedAction] | None = None):
        return self.backbone.log_probs(state, actions)

    def forward(self, state: MixedResolutionState, actions: list[MixedAction] | None = None):
        return self.backbone(state, actions)

    @torch.no_grad()
    def sample(self, state: MixedResolutionState, actions: list[MixedAction] | None = None, greedy: bool = False):
        logp, acts = self.log_probs(state, actions)
        idx = int(logp.argmax()) if greedy else int(torch.distributions.Categorical(logits=logp).sample())
        return acts[idx], logp[idx], acts
