"""Differentiable GRPO objective with explicit full-action log probabilities."""
from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    out = torch.zeros_like(rewards)
    for group in torch.unique(group_ids):
        mask = group_ids == group
        values = rewards[mask]
        out[mask] = (values - values.mean()) / (values.std(unbiased=False) + eps)
    return out


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    entropy_coef: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Return scalar loss and differentiable diagnostics.

    `logp`, `old_logp`, and `ref_logp` are gathered from the same complete
    legal-action distribution.  KL and entropy stay tensors all the way into
    the objective; converting them to Python floats would silently remove the
    regularization gradient.
    """
    adv = group_advantages(rewards, group_ids).detach()
    ratio = torch.exp(logp - old_logp.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy = -torch.minimum(ratio * adv, clipped * adv).mean()
    # Bernoulli-style action-sample KL estimate, with gradient through logp.
    kl = (logp - ref_logp.detach()).mean()
    entropy = -logp.mean()
    total = policy + kl_coef * kl - entropy_coef * entropy
    return {"loss": total, "policy_loss": policy, "kl": kl, "entropy": entropy, "advantages": adv, "ratio": ratio}
