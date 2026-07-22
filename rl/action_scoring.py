"""Shared action-score semantics for offline proposal ranking and decoding.

All non-STOP scores are log CTMC intensities.  STOP currently has no learned
head, so it is an explicit configurable baseline rather than a claimed policy
prediction.  A later online stage may replace that baseline with a learned
head without changing the action interface.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from mrna_editflow.core.constants import NUC_TO_ID


ACTION_OPERATIONS = ("sub", "ins", "del", "stop")


def _reference_tensor(model_output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    rates = model_output.get("rates")
    if not isinstance(rates, torch.Tensor):
        raise TypeError("model_output['rates'] must be a torch.Tensor")
    if rates.ndim != 3 or rates.shape[0] < 1 or rates.shape[-1] < 3:
        raise ValueError("rates must have shape [batch, length, >=3]")
    return rates


def operation_log_score(
    model_output: Mapping[str, torch.Tensor],
    op: str,
    pos: Optional[int],
    nt: Optional[str],
    *,
    eps: float = 1e-20,
    stop_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Return the shared differentiable action log-score.

    ``sub`` and ``ins`` score ``log(lambda_op * p_nt)``; ``del`` scores
    ``log(lambda_del)``.  ``stop`` is a configurable, non-learned baseline
    score with the same device/dtype as ``rates``.  It intentionally has no
    position or nucleotide and consumes no edit budget in the decoder.
    """
    rates = _reference_tensor(model_output)
    op_l = str(op).lower()
    floor = max(float(eps), torch.finfo(rates.dtype).tiny)
    if op_l == "stop":
        return rates.sum() * 0.0 + rates.new_tensor(float(stop_logit_bias))
    if pos is None:
        raise ValueError(f"{op_l} requires a position")
    pos_i = int(pos)
    if not 0 <= pos_i < rates.shape[1]:
        raise IndexError(f"position {pos_i} is outside model output length {rates.shape[1]}")
    if op_l == "del":
        return torch.log(rates[0, pos_i, 2].clamp_min(floor))
    if op_l not in {"sub", "ins"}:
        raise ValueError(f"unsupported action op {op!r}; expected one of {ACTION_OPERATIONS}")
    if nt is None or str(nt) not in NUC_TO_ID:
        raise ValueError(f"{op_l} requires an A/C/G/U nucleotide")
    probs_name = "sub_probs" if op_l == "sub" else "ins_probs"
    probs = model_output.get(probs_name)
    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"model_output[{probs_name!r}] must be a torch.Tensor")
    rate_channel = 1 if op_l == "sub" else 0
    nt_idx = NUC_TO_ID[str(nt)]
    intensity = rates[0, pos_i, rate_channel].clamp_min(floor)
    probability = probs[0, pos_i, nt_idx].clamp_min(floor)
    return torch.log(intensity * probability)


def action_log_score_float(*args, **kwargs) -> float:
    """Detach :func:`operation_log_score` for deterministic decoder ranking."""
    return float(operation_log_score(*args, **kwargs).detach().float().cpu())


def softmax_from_log_scores(
    log_scores: Sequence[float] | torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return ``softmax(log_score / temperature)`` with stable validation."""
    scores = torch.as_tensor(log_scores, dtype=torch.float64)
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("log_scores must be a non-empty one-dimensional sequence")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive for stochastic softmax")
    return torch.softmax(scores / float(temperature), dim=0)


__all__ = [
    "ACTION_OPERATIONS",
    "operation_log_score",
    "action_log_score_float",
    "softmax_from_log_scores",
]
