"""MK0 structural STOP target process and survival-hazard objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class StopTarget:
    completion_time: float
    dwell: float
    latent_stop_time: float
    observed_time: float
    event_observed: bool

    def __post_init__(self) -> None:
        values = (
            self.completion_time,
            self.dwell,
            self.latent_stop_time,
            self.observed_time,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("STOP target times must be finite")
        if not 0.0 <= self.completion_time <= 1.0:
            raise ValueError("STOP completion time must lie in [0,1]")
        if self.dwell <= 0.0:
            raise ValueError("STOP dwell must be strictly positive")
        if not math.isclose(
            self.latent_stop_time,
            self.completion_time + self.dwell,
            abs_tol=1.0e-12,
            rel_tol=1.0e-12,
        ):
            raise ValueError("STOP latent time must equal completion plus dwell")
        expected_observed = min(self.latent_stop_time, 1.0)
        if not math.isclose(
            self.observed_time,
            expected_observed,
            abs_tol=1.0e-12,
            rel_tol=1.0e-12,
        ):
            raise ValueError("STOP observed time must equal min(latent,1)")
        if self.event_observed != (self.latent_stop_time < 1.0):
            raise ValueError("STOP event indicator disagrees with latent time")


def sample_stop_target(
    switch_clocks: Mapping[int, float],
    *,
    gamma_ref: float,
    rng: random.Random,
) -> StopTarget:
    if not math.isfinite(gamma_ref) or gamma_ref <= 0.0:
        raise ValueError("gamma_ref must be finite and positive")
    if any(not 0.0 <= value <= 1.0 for value in switch_clocks.values()):
        raise ValueError("switch clocks must lie in [0,1]")
    completion = max(switch_clocks.values(), default=0.0)
    # random.Random.expovariate uses -log(1-U)/gamma and is strictly positive
    # because random() is in [0,1), with U=0 handled by nextafter.
    uniform = max(rng.random(), math.nextafter(0.0, 1.0))
    dwell = max(
        -math.log1p(-uniform) / gamma_ref,
        math.nextafter(0.0, 1.0),
    )
    if not math.isfinite(dwell) or dwell <= 0.0:
        raise FloatingPointError("STOP dwell must be finite and strictly positive")
    latent = completion + dwell
    event = latent < 1.0
    return StopTarget(
        completion_time=completion,
        dwell=dwell,
        latent_stop_time=latent,
        observed_time=min(latent, 1.0),
        event_observed=event,
    )


def survival_stop_loss(
    target: StopTarget,
    rate_fn: Callable[[float], float],
    *,
    quadrature_points: int = 64,
    breakpoints: Sequence[float] = (),
) -> float:
    """Evaluate -delta log lambda(t-) + integral_0^t lambda(s) ds.

    Known hazard discontinuities may be supplied as ``breakpoints``.  The
    Gauss--Legendre rule is then applied independently on every open-smooth
    interval, so a piecewise-constant or piecewise-smooth absolute hazard is
    not silently integrated as though it were globally smooth.
    """

    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least two")
    import numpy as np

    upper = target.observed_time
    if not math.isfinite(upper) or not 0.0 <= upper <= 1.0:
        raise ValueError("observed STOP time must lie in [0,1]")
    normalized_breakpoints = tuple(float(value) for value in breakpoints)
    if any(not math.isfinite(value) for value in normalized_breakpoints):
        raise ValueError("STOP quadrature breakpoints must be finite")
    if tuple(sorted(set(normalized_breakpoints))) != normalized_breakpoints:
        raise ValueError("STOP quadrature breakpoints must be strictly increasing")
    if any(not 0.0 < value < upper for value in normalized_breakpoints):
        raise ValueError("STOP quadrature breakpoints must lie inside (0,t_obs)")

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    integrated = 0.0
    boundaries = (0.0,) + normalized_breakpoints + (upper,)
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        times = midpoint + half_width * nodes
        rates = np.asarray([rate_fn(float(time)) for time in times], dtype=np.float64)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
            raise FloatingPointError("STOP rate is negative, NaN or Inf")
        integrated += float(half_width * np.dot(weights, rates))
    event_term = 0.0
    if target.event_observed:
        event_rate = float(rate_fn(math.nextafter(target.observed_time, 0.0)))
        if not math.isfinite(event_rate) or event_rate <= 0.0:
            raise FloatingPointError(
                "observed STOP requires positive finite event rate"
            )
        event_term = -math.log(event_rate)
    loss = event_term + integrated
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite STOP survival loss")
    return loss


def constant_hazard_stop_loss(target: StopTarget, hazard: float) -> float:
    if not math.isfinite(hazard) or hazard <= 0.0:
        raise ValueError("constant hazard must be finite and positive")
    return hazard * target.observed_time - int(target.event_observed) * math.log(hazard)


def stop_event_censor_oracle(
    *,
    gamma_ref: float,
    completion_times: Sequence[float],
    samples_per_completion: int,
    seed: int,
) -> dict[str, float | int]:
    """Monte Carlo event/censor oracle against the analytic exponential law."""

    if not completion_times or samples_per_completion <= 0:
        raise ValueError("oracle requires completions and positive sample count")
    rng = random.Random(seed)
    observed_events = 0
    total = 0
    expected_events = 0.0
    dwell_sum = 0.0
    for completion in completion_times:
        if not 0.0 <= completion <= 1.0:
            raise ValueError("completion time must lie in [0,1]")
        expected_events += samples_per_completion * (
            1.0 - math.exp(-gamma_ref * (1.0 - completion))
        )
        for _ in range(samples_per_completion):
            target = sample_stop_target({0: completion}, gamma_ref=gamma_ref, rng=rng)
            observed_events += int(target.event_observed)
            dwell_sum += target.dwell
            total += 1
    observed_fraction = observed_events / total
    expected_fraction = expected_events / total
    return {
        "sample_count": total,
        "observed_event_fraction": observed_fraction,
        "expected_event_fraction": expected_fraction,
        "absolute_fraction_error": abs(observed_fraction - expected_fraction),
        "observed_mean_dwell": dwell_sum / total,
        "expected_mean_dwell": 1.0 / gamma_ref,
    }


def halted_edit_flow_weight(*, halted: bool, target_weight: float) -> float:
    """The at-risk guard: HALTED auxiliary states contribute no EF target term."""

    return 0.0 if halted else target_weight
