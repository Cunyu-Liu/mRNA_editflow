"""Frozen source-to-target schedules and conditional switch hazards."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScheduleValue:
    t_requested: float
    t_evaluated: float
    kappa: float
    derivative: float
    rho: float
    endpoint_clipped: bool


def _validate_time(t: float) -> None:
    if not math.isfinite(t) or not 0.0 <= t <= 1.0:
        raise ValueError("time must be finite and within [0, 1]")


def cubic_schedule(t: float) -> tuple[float, float]:
    _validate_time(t)
    return t**3, 3.0 * t**2


def linear_schedule(t: float) -> tuple[float, float]:
    _validate_time(t)
    return t, 1.0


def evaluate_schedule(
    t: float,
    *,
    name: str = "cubic",
    time_eps: float = 1.0e-4,
) -> ScheduleValue:
    _validate_time(t)
    if not 0.0 < time_eps < 1.0:
        raise ValueError("time_eps must lie strictly between zero and one")
    evaluated = min(t, 1.0 - time_eps)
    if name == "cubic":
        kappa, derivative = cubic_schedule(evaluated)
    elif name == "linear":
        kappa, derivative = linear_schedule(evaluated)
    else:
        raise ValueError(f"unknown schedule: {name}")
    denominator = 1.0 - kappa
    if denominator <= 0.0:
        raise FloatingPointError("schedule evaluated at singular target endpoint")
    return ScheduleValue(
        t_requested=t,
        t_evaluated=evaluated,
        kappa=kappa,
        derivative=derivative,
        rho=derivative / denominator,
        endpoint_clipped=evaluated != t,
    )


def rho(t: float, *, name: str = "cubic", time_eps: float = 1.0e-4) -> float:
    return evaluate_schedule(t, name=name, time_eps=time_eps).rho
