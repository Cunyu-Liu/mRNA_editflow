"""Mean, heteroscedastic variance, benefit probability and ranking heads."""
from __future__ import annotations

import torch
from torch import nn


class UncertaintyHead(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, 1)
        self.logvar = nn.Linear(hidden_dim, 1)
        self.beneficial = nn.Linear(hidden_dim, 1)
        self.rank = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = self.mean(x).squeeze(-1)
        logvar = self.logvar(x).squeeze(-1).clamp(-8.0, 6.0)
        beneficial_logit = self.beneficial(x).squeeze(-1)
        return {
            "mean": mean,
            "logvar": logvar,
            "variance": logvar.exp(),
            "beneficial_logit": beneficial_logit,
            "beneficial_probability": torch.sigmoid(beneficial_logit),
            "rank": self.rank(x).squeeze(-1),
        }


__all__ = ["UncertaintyHead"]
