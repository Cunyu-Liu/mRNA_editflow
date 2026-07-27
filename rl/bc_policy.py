"""Behavior-cloning baseline on the same legal-action distribution."""
from __future__ import annotations

import torch


def behavior_cloning_loss(log_probs: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be [batch, n_legal_actions]")
    return -log_probs.gather(1, target_indices.long().unsqueeze(1)).mean()
