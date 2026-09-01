"""Online EMA task-gradient-norm scaling for the Critic V7 loss.

V7 is the architecture-side loss-mechanism candidate taken after the V6 screen
main criterion did not pass (v6_full MPRAU pair-mean rho 0.051 < V5 0.103).
The diagnostic is task-gradient imbalance on a small, long-tailed dataset.
This module ingests, per task-homogeneous effective batch, the L2 norm of the
detached prediction gradient (dL/dv_hat, a 32-vector), keeps a per-task EMA,
and returns a multiplier centered around the geometric mean of all EMAs so
that tasks with large gradient norms are down-weighted and small ones are
up-weighted (gradient balancing, cf. 2025 IEEE analysis on optimization
imbalance <-> task gradient norm).
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


class XEditCriticV7GradientNormScaleError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV7GradientNormScaleError(message)


class GradientNormScalerV7:
    """Per-task EMA gradient-norm scaler, disabled-by-default semantics.

    When the scaler is not enabled the training loop must not call it, so the
    V6 loss path is untouched (switch-off == V6 bit-identical).
    """

    def __init__(self, ema_alpha: float = 0.05, floor: float = 1e-6) -> None:
        _require(
            0.0 < float(ema_alpha) < 1.0,
            "EMA alpha must be strictly inside (0, 1)",
        )
        _require(float(floor) > 0.0, "gradient-norm floor must be positive")
        self.ema_alpha = float(ema_alpha)
        self.floor = float(floor)
        self._ema: dict[str, float] = {}
        self._reference: float | None = None

    def ema_norms(self) -> dict[str, float]:
        return dict(self._ema)

    def reference(self) -> float | None:
        return self._reference

    def reset(self) -> None:
        self._ema = {}
        self._reference = None

    def ingest(self, task_id: str, gradient_norm: float) -> float:
        """Update the EMA for this task and return the centered multiplier."""
        number = float(gradient_norm)
        _require(
            math.isfinite(number) and number > 0.0,
            f"task gradient norm is nonfinite or nonpositive: {task_id}",
        )
        previous = self._ema.get(task_id)
        if previous is None:
            self._ema[task_id] = number
        else:
            self._ema[task_id] = self.ema_alpha * number + (
                1.0 - self.ema_alpha
            ) * previous
        _require(
            all(v > 0.0 and math.isfinite(v) for v in self._ema.values()),
            "a task EMA gradient norm became nonpositive or nonfinite",
        )
        self._reference = math.exp(
            sum(math.log(value) for value in self._ema.values())
            / len(self._ema)
        )
        ema = self._ema[task_id]
        return self._reference / max(ema, self.floor)

    def scale(
        self, task_id: str, prediction_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        """Scale a detached prediction-gradient vector; returns (scaled, mult, norm)."""
        _require(
            isinstance(prediction_gradient, torch.Tensor)
            and prediction_gradient.dim() == 1
            and prediction_gradient.numel() > 0,
            "prediction gradient must be a nonempty 1-D tensor",
        )
        norm = float(prediction_gradient.detach().norm().item())
        multiplier = self.ingest(task_id, norm)
        return prediction_gradient * multiplier, multiplier, norm

    def frozen_multipliers(self, ema_norms: Mapping[str, float]) -> dict[str, float]:
        """Turn externally supplied EMA norms into centered multipliers.

        Used by calibration/tests to verify geometric-mean centering without
        mutating the live scaler state.
        """
        checked = {}
        for task, value in ema_norms.items():
            number = float(value)
            _require(
                math.isfinite(number) and number > 0.0,
                f"invalid EMA gradient norm for {task}",
            )
            checked[str(task)] = number
        _require(len(checked) >= 2, "at least two tasks are required for centering")
        reference = math.exp(
            sum(math.log(value) for value in checked.values()) / len(checked)
        )
        return {task: reference / max(value, self.floor) for task, value in checked.items()}
