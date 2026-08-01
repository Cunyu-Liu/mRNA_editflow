"""Frozen-reference KL helpers and adaptive safety controller."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def categorical_kl(current_log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
    """Exact ``KL(current || reference)`` over a common masked action set."""
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("KL distributions must share shape")
    current = torch.exp(current_log_probs)
    valid = torch.isfinite(current_log_probs) & torch.isfinite(reference_log_probs)
    return torch.where(valid, current * (current_log_probs - reference_log_probs), torch.zeros_like(current)).sum()


@dataclass
class AdaptiveKLController:
    coefficient: float = 0.01
    target_kl: float = 0.05
    max_kl: float = 0.25
    multiplier: float = 1.5
    min_coefficient: float = 1e-6
    max_coefficient: float = 100.0

    def update(self, observed_kl: float) -> tuple[float, bool]:
        """Adapt coefficient and return ``(coefficient, skip_update)``."""
        if not torch.isfinite(torch.tensor(observed_kl)):
            return self.coefficient, True
        if observed_kl > self.target_kl:
            self.coefficient = min(self.max_coefficient, self.coefficient * self.multiplier)
        elif observed_kl < self.target_kl / max(self.multiplier, 1e-8):
            self.coefficient = max(self.min_coefficient, self.coefficient / self.multiplier)
        return self.coefficient, bool(observed_kl > self.max_kl)


__all__ = ["categorical_kl", "AdaptiveKLController"]
