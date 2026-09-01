"""Inference-time temperature control for XEditSetFlow (V5 hypothesis, Direction E).

Direction E of the 2026-08-29 architecture review: unique-candidate rate and
STOP calibration can be tuned at inference time without retraining, following
FlexFlow's Dirichlet-prior temperature scaling.  Two knobs:

1. ``mode prior temperature`` — sharpen (T < 1) or flatten (T > 1) the
   trajectory-fixed mode mixture at the root, trading mode concentration
   against mode diversity;
2. ``STOP rate scale`` — multiply the explicit STOP rate to stop earlier
   (scale > 1, fewer edits, higher precision) or explore longer
   (scale < 1, more edits, higher coverage).

Pure functions over already-computed tensors/rate maps so they can be
validated on CPU against any terminal checkpoint without touching the frozen
V4/V4-S1 training code.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from core.route2_legal_xeditflow import STOP, LegalAction


class XEditSetFlowTemperatureControlV5Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowTemperatureControlV5Error(message)


# Frozen prospective sweep grid: identity (1.0, 1.0) is always included so a
# sweep is self-controlled against the unmodified sampler.
MODE_PRIOR_TEMPERATURES_V5 = (0.5, 0.75, 1.0, 1.25, 1.5)
STOP_RATE_SCALES_V5 = (0.25, 0.5, 1.0, 2.0, 4.0)


def mode_prior_entropy_v5(prior: Sequence[float]) -> float:
    """Shannon entropy (nats) of a discrete prior, for sweep diagnostics."""

    _require(bool(prior), "mode prior is empty")
    entropy = 0.0
    for value in prior:
        probability = float(value)
        _require(
            math.isfinite(probability) and probability >= 0.0,
            "mode prior contains an invalid probability",
        )
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def temper_mode_prior_v5(
    prior: Sequence[float], *, temperature: float
) -> tuple[float, ...]:
    """Return ``prior_i ** (1/T)`` renormalized to sum to one.

    ``temperature == 1`` is the identity (up to float rounding).  ``T < 1``
    sharpens the mixture toward its dominant mode; ``T > 1`` flattens it.
    """

    _require(bool(prior), "mode prior is empty")
    _require(
        math.isfinite(temperature) and temperature > 0.0,
        "mode prior temperature must be finite and positive",
    )
    tempered: list[float] = []
    for value in prior:
        probability = float(value)
        _require(
            math.isfinite(probability) and probability >= 0.0,
            "mode prior contains an invalid probability",
        )
        tempered.append(math.pow(probability, 1.0 / temperature))
    total = sum(tempered)
    _require(
        math.isfinite(total) and total > 0.0,
        "tempered mode prior mass degenerated to zero",
    )
    result = tuple(value / total for value in tempered)
    _require(
        abs(sum(result) - 1.0) <= 1e-9,
        "tempered mode prior is not normalized",
    )
    return result


def scale_stop_rate_v5(
    rate_map: Mapping[LegalAction, float],
    *,
    stop_scale: float,
) -> dict[LegalAction, float]:
    """Return a new rate map with only the STOP action's rate rescaled."""

    _require(bool(rate_map), "rate map is empty")
    _require(
        math.isfinite(stop_scale) and stop_scale > 0.0,
        "STOP rate scale must be finite and positive",
    )
    stop_seen = 0
    result: dict[LegalAction, float] = {}
    for action, rate in rate_map.items():
        value = float(rate)
        _require(
            math.isfinite(value) and value > 0.0,
            "rate map contains an invalid rate",
        )
        if action.kind == STOP:
            stop_seen += 1
            scaled = value * stop_scale
            _require(
                math.isfinite(scaled) and scaled > 0.0,
                "scaled STOP rate degenerated",
            )
            result[action] = scaled
        else:
            result[action] = value
    _require(stop_seen == 1, "rate map must contain exactly one STOP action")
    return result


def frozen_temperature_sweep_v5() -> tuple[tuple[float, float], ...]:
    """Cartesian sweep grid over mode-prior temperature and STOP scale."""

    return tuple(
        (temperature, stop_scale)
        for temperature in MODE_PRIOR_TEMPERATURES_V5
        for stop_scale in STOP_RATE_SCALES_V5
    )
