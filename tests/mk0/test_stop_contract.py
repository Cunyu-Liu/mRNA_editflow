"""MK0-06 structural STOP, survival loss and censoring oracles."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from mrna_editflow.core.mk0.stop import (
    StopTarget,
    constant_hazard_stop_loss,
    halted_edit_flow_weight,
    sample_stop_target,
    stop_event_censor_oracle,
    survival_stop_loss,
)

from .conftest import FLOAT64_ATOL, FLOAT64_RTOL, SEED


class _ZeroDrawRandom(random.Random):
    def random(self) -> float:
        return 0.0


def test_zero_edit_pair_has_positive_dwell_and_no_step0_stop_atom() -> None:
    rng = random.Random(SEED)
    targets = [sample_stop_target({}, gamma_ref=16.0, rng=rng) for _ in range(10_000)]
    assert all(target.completion_time == 0.0 for target in targets)
    assert all(target.dwell > 0.0 for target in targets)
    assert all(target.latent_stop_time > target.completion_time for target in targets)
    assert all(target.observed_time > 0.0 for target in targets)


def test_zero_rng_endpoint_is_guarded_to_a_strictly_positive_dwell() -> None:
    # random.Random.random is defined on [0, 1), so an implementation must not
    # turn its representable zero endpoint into a simultaneous edit/STOP jump.
    target = sample_stop_target({}, gamma_ref=16.0, rng=_ZeroDrawRandom())
    assert target.dwell > 0.0
    assert target.latent_stop_time > target.completion_time


def test_completion_is_max_switch_clock_and_dwell_is_empirically_independent() -> None:
    rng = random.Random(SEED)
    completions = np.empty(20_000, dtype=np.float64)
    dwells = np.empty(20_000, dtype=np.float64)
    for index in range(len(completions)):
        clocks = {0: rng.random(), 1: rng.random(), 2: rng.random()}
        target = sample_stop_target(clocks, gamma_ref=16.0, rng=rng)
        completions[index] = target.completion_time
        dwells[index] = target.dwell
        assert target.completion_time == max(clocks.values())
        assert target.latent_stop_time == target.completion_time + target.dwell
        assert target.observed_time == min(target.latent_stop_time, 1.0)
        assert target.event_observed == (target.latent_stop_time < 1.0)
    correlation = float(np.corrcoef(completions, dwells)[0, 1])
    assert abs(correlation) < 0.025
    assert abs(float(dwells.mean()) - 1.0 / 16.0) < 0.002


@pytest.mark.parametrize("event_observed", (False, True))
def test_survival_loss_matches_closed_form_time_varying_hazard(
    event_observed: bool,
) -> None:
    observed_time = 0.65 if event_observed else 1.0
    latent_time = observed_time if event_observed else 1.2
    target = StopTarget(
        completion_time=0.4,
        dwell=latent_time - 0.4,
        latent_stop_time=latent_time,
        observed_time=observed_time,
        event_observed=event_observed,
    )
    rate_fn = lambda time: 2.0 + 3.0 * time
    observed = survival_stop_loss(target, rate_fn, quadrature_points=64)
    expected = 2.0 * observed_time + 1.5 * observed_time**2
    if event_observed:
        expected -= math.log(rate_fn(observed_time))
    assert math.isclose(
        observed,
        expected,
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )


@pytest.mark.parametrize("hazard", (0.125, 1.0, 8.0, 16.0))
@pytest.mark.parametrize("event_observed", (False, True))
def test_constant_absolute_hazard_loss_matches_quadrature(
    hazard: float, event_observed: bool
) -> None:
    target = StopTarget(
        completion_time=0.25,
        dwell=0.5 if event_observed else 1.0,
        latent_stop_time=0.75 if event_observed else 1.25,
        observed_time=0.75 if event_observed else 1.0,
        event_observed=event_observed,
    )
    observed = survival_stop_loss(target, lambda _: hazard, quadrature_points=64)
    expected = constant_hazard_stop_loss(target, hazard)
    assert math.isclose(
        observed,
        expected,
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )


@pytest.mark.parametrize("event_observed", (False, True))
def test_piecewise_absolute_hazard_uses_declared_breakpoint(
    event_observed: bool,
) -> None:
    observed_time = 0.8 if event_observed else 1.0
    breakpoint = 0.35
    left_hazard = 0.75
    right_hazard = 3.25
    latent_time = observed_time if event_observed else 1.2
    target = StopTarget(
        completion_time=0.2,
        dwell=latent_time - 0.2,
        latent_stop_time=latent_time,
        observed_time=observed_time,
        event_observed=event_observed,
    )

    def piecewise(time: float) -> float:
        return left_hazard if time < breakpoint else right_hazard

    observed = survival_stop_loss(
        target,
        piecewise,
        quadrature_points=64,
        breakpoints=(breakpoint,),
    )
    expected = left_hazard * breakpoint + right_hazard * (observed_time - breakpoint)
    if event_observed:
        expected -= math.log(right_hazard)
    assert math.isclose(
        observed,
        expected,
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )


@pytest.mark.parametrize(
    "breakpoints",
    ((0.0,), (0.5, 0.5), (0.75, 0.25), (math.nan,)),
)
def test_invalid_survival_breakpoints_fail_closed(
    breakpoints: tuple[float, ...],
) -> None:
    target = StopTarget(0.0, 0.5, 0.5, 0.5, True)
    with pytest.raises(ValueError):
        survival_stop_loss(target, lambda _: 1.0, breakpoints=breakpoints)


@pytest.mark.parametrize(
    "values",
    (
        (0.2, 0.0, 0.2, 0.2, True),
        (0.2, 0.3, 0.6, 0.6, True),
        (0.2, 0.3, 0.5, 1.0, True),
        (0.2, 1.0, 1.2, 1.0, True),
    ),
)
def test_internally_inconsistent_stop_target_fails_closed(
    values: tuple[float, float, float, float, bool],
) -> None:
    with pytest.raises(ValueError):
        StopTarget(*values)


def test_stop_primary_identifies_absolute_scale_not_only_event_type_ratio() -> None:
    target = StopTarget(0.2, 0.3, 0.5, 0.5, True)
    # Scaling STOP and edit rates together preserves this event-type ratio.
    ratio_1 = 1.0 / (1.0 + 3.0)
    ratio_2 = 2.0 / (2.0 + 6.0)
    assert ratio_1 == ratio_2
    # The survival objective still identifies the absolute STOP intensity.
    assert constant_hazard_stop_loss(target, 1.0) != constant_hazard_stop_loss(
        target, 2.0
    )


@pytest.mark.parametrize("gamma_ref", (8.0, 16.0, 32.0))
def test_stop_event_and_administrative_censor_fraction_oracle(gamma_ref: float) -> None:
    report = stop_event_censor_oracle(
        gamma_ref=gamma_ref,
        completion_times=(0.0, 0.25, 0.5, 0.75, 0.9),
        samples_per_completion=5_000,
        seed=SEED,
    )
    assert report["sample_count"] == 25_000
    assert report["absolute_fraction_error"] < 0.01
    assert abs(report["observed_mean_dwell"] - report["expected_mean_dwell"]) < 0.002
    assert 0.0 < report["observed_event_fraction"] < 1.0


def test_halted_auxiliary_state_contributes_no_edit_flow_weight() -> None:
    assert halted_edit_flow_weight(halted=True, target_weight=123.0) == 0.0
    assert halted_edit_flow_weight(halted=False, target_weight=123.0) == 123.0


@pytest.mark.parametrize("gamma_ref", (0.0, -1.0, math.nan, math.inf))
def test_invalid_stop_reference_rate_fails_closed(gamma_ref: float) -> None:
    with pytest.raises(ValueError):
        sample_stop_target({}, gamma_ref=gamma_ref, rng=random.Random(SEED))


@pytest.mark.parametrize("bad_rate", (-1.0, math.nan, math.inf))
def test_invalid_survival_rate_fails_closed(bad_rate: float) -> None:
    target = StopTarget(0.0, 0.5, 0.5, 0.5, True)
    with pytest.raises(FloatingPointError):
        survival_stop_loss(target, lambda _: bad_rate)
