from __future__ import annotations

import math

import pytest

from core.route2_legal_xeditflow import LegalAction, STOP
from core.route2_xeditsetflow_temperature_control_v5 import (
    MODE_PRIOR_TEMPERATURES_V5,
    STOP_RATE_SCALES_V5,
    frozen_temperature_sweep_v5,
    mode_prior_entropy_v5,
    scale_stop_rate_v5,
    temper_mode_prior_v5,
)


def _stop_action() -> LegalAction:
    return LegalAction(STOP)


def _sub_action(position: int = 3) -> LegalAction:
    return LegalAction("SUB", position, "G")


def test_identity_temperature_is_identity():
    prior = (0.4, 0.3, 0.2, 0.1)
    tempered = temper_mode_prior_v5(prior, temperature=1.0)
    assert all(abs(a - b) <= 1e-12 for a, b in zip(tempered, prior))


def test_sharpening_and_flattening_move_entropy_opposite_ways():
    prior = (0.4, 0.3, 0.2, 0.1)
    base = mode_prior_entropy_v5(prior)
    sharp = mode_prior_entropy_v5(temper_mode_prior_v5(prior, temperature=0.5))
    flat = mode_prior_entropy_v5(temper_mode_prior_v5(prior, temperature=1.5))
    assert sharp < base < flat
    for temperature in (0.5, 0.75, 1.5):
        tempered = temper_mode_prior_v5(prior, temperature=temperature)
        assert abs(sum(tempered) - 1.0) <= 1e-9


def test_extreme_temperature_limits():
    prior = (0.7, 0.2, 0.1)
    dominant = temper_mode_prior_v5(prior, temperature=1e-3)
    assert dominant[0] > 0.999
    uniform = temper_mode_prior_v5(prior, temperature=1e3)
    assert max(uniform) - min(uniform) < 1e-3


def test_invalid_temperature_or_prior_rejected():
    with pytest.raises(Exception, match="positive"):
        temper_mode_prior_v5((0.5, 0.5), temperature=0.0)
    with pytest.raises(Exception, match="invalid probability"):
        temper_mode_prior_v5((0.5, -0.5), temperature=1.0)


def test_stop_rate_scale_multiplies_only_stop():
    stop = _stop_action()
    sub = _sub_action()
    rate_map = {sub: 2.0, stop: 1.0}
    scaled = scale_stop_rate_v5(rate_map, stop_scale=3.0)
    assert scaled[stop] == pytest.approx(3.0)
    assert scaled[sub] == pytest.approx(2.0)
    identity = scale_stop_rate_v5(rate_map, stop_scale=1.0)
    assert identity[stop] == pytest.approx(1.0)


def test_stop_rate_scale_rejects_missing_or_extra_stop():
    sub = _sub_action()
    with pytest.raises(Exception, match="exactly one STOP"):
        scale_stop_rate_v5({sub: 1.0}, stop_scale=2.0)
    with pytest.raises(Exception, match="positive"):
        scale_stop_rate_v5({_stop_action(): 1.0}, stop_scale=-1.0)


def test_frozen_sweep_grid_contains_identity_and_is_complete():
    grid = frozen_temperature_sweep_v5()
    assert (1.0, 1.0) in grid
    assert len(grid) == len(MODE_PRIOR_TEMPERATURES_V5) * len(STOP_RATE_SCALES_V5)
    temperatures = {temperature for temperature, _ in grid}
    stop_scales = {stop_scale for _, stop_scale in grid}
    assert temperatures == set(MODE_PRIOR_TEMPERATURES_V5)
    assert stop_scales == set(STOP_RATE_SCALES_V5)
    assert all(math.isfinite(t) and t > 0 for t, s in grid)
    assert all(math.isfinite(s) and s > 0 for t, s in grid)
