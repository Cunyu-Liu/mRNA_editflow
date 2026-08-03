"""Exact-event-time CTMC samplers and fail-closed sampling gates for EF0.

The frozen MK0 ``constrained_single_event_first_order`` sampler remains the
contract-compatible engineering path.  This module adds two explicitly
separated routes:

* ``sample_exact_gillespie`` is exact only for a time-homogeneous generator.
  It probes the supplied rate function at multiple external times and refuses
  to run when the state-conditioned rates vary with time.
* ``sample_nonhomogeneous_ctmc`` samples a time-dependent generator by
  inverting a numerically certified integrated hazard.  Its trajectory
  likelihood is explicitly numerical and is never labelled exact Gillespie.

Neither route provides biological or benchmark evidence by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import sys
from typing import Callable, Mapping, Optional

from ..mk0.samplers import certify_remaining_integrated_hazard
from ..mk0.state_action import (
    apply_action,
    enumerate_legal_actions,
    force_terminate,
    is_legal,
)
from ..mk0.types import ActionType, AtomicAction, EditState, Phase, TerminationReason

RateFunction = Callable[[EditState, float], Mapping[AtomicAction, float]]


class TimeInhomogeneousRateError(ValueError):
    """Raised when a homogeneous exact-Gillespie gate cannot be admitted."""


class HazardConvergenceError(RuntimeError):
    """Raised when numerical integrated-hazard evidence is not converged."""


@dataclass(frozen=True)
class ExactCTMCSamplerConfig:
    """Numerical controls for event-time CTMC sampling."""

    min_length: int = 1
    max_length: int = 256
    horizon: float = 1.0
    integration_lower_order: int = 32
    integration_higher_order: int = 64
    integration_convergence_atol: float = 1.0e-8
    root_atol: float = 1.0e-8
    max_root_iterations: int = 48
    time_homogeneity_atol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError("invalid exact CTMC length bounds")
        if not 0.0 < self.horizon <= 1.0:
            raise ValueError("exact CTMC horizon must lie in (0,1]")
        if not (
            2 <= self.integration_lower_order < self.integration_higher_order <= 4096
        ):
            raise ValueError("invalid integrated-hazard quadrature orders")
        if self.integration_convergence_atol < 0.0 or self.root_atol < 0.0:
            raise ValueError("exact CTMC tolerances must be non-negative")
        if self.max_root_iterations < 8:
            raise ValueError("exact CTMC root iteration budget is too small")
        if self.time_homogeneity_atol < 0.0:
            raise ValueError("time homogeneity tolerance must be non-negative")


@dataclass(frozen=True)
class ExactCTMCStep:
    event_index: int
    t_start: float
    t_end: float
    waiting_time: float
    total_hazard_at_event: float
    integrated_hazard: float
    event_uniform: float
    action_uniform: Optional[float]
    selected_action: Optional[AtomicAction]
    outcome: str
    before_hash: str
    after_hash: str
    candidate_actions_hash: str
    candidate_rates_hash: str
    integration_disagreement: float
    root_residual: float
    log_likelihood_increment: float


@dataclass(frozen=True)
class ExactCTMCResult:
    sampler: str
    exact_gillespie: bool
    time_homogeneous: bool
    initial_state: EditState
    final_state: EditState
    steps: tuple[ExactCTMCStep, ...]
    seed: int
    min_length: int
    max_length: int
    horizon: float
    termination_time: float
    termination_before_hash: str
    trajectory_log_likelihood: float
    likelihood_semantics: str
    max_integration_disagreement: float
    max_root_residual: float
    max_time_homogeneity_delta: float

    @property
    def edit_events(self) -> int:
        return sum(
            int(
                step.selected_action is not None
                and step.selected_action.kind != ActionType.STOP
            )
            for step in self.steps
        )


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _candidate_actions_hash(
    ordered: tuple[tuple[AtomicAction, float], ...],
) -> str:
    return _hash_payload([action.key for action, _ in ordered])


def _candidate_rates_hash(
    ordered: tuple[tuple[AtomicAction, float], ...],
) -> str:
    return _hash_payload([[action.key, float(rate).hex()] for action, rate in ordered])


def _draw_action(
    ordered: tuple[tuple[AtomicAction, float], ...],
    draw: float,
    total: float,
) -> AtomicAction:
    if not ordered or total <= 0.0:
        raise ValueError("cannot draw an action from zero hazard")
    cumulative = 0.0
    for action, rate in ordered:
        cumulative += rate / total
        if draw < cumulative:
            return action
    return ordered[-1][0]


def _validated_rates(
    state: EditState,
    time: float,
    rate_fn: RateFunction,
    *,
    min_length: int,
    max_length: int,
) -> dict[AtomicAction, float]:
    if not 0.0 <= time < 1.0:
        raise ValueError("rate-function time must be in [0,1)")
    raw_rates = dict(rate_fn(state, time))
    filtered: dict[AtomicAction, float] = {}
    for action, raw_rate in raw_rates.items():
        if not isinstance(action, AtomicAction):
            raise TypeError("rate mapping keys must be AtomicAction instances")
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
            raise TypeError(f"rate for {action.key} must be a real scalar")
        rate = float(raw_rate)
        if not math.isfinite(rate) or rate < 0.0:
            raise FloatingPointError(f"invalid rate for {action.key}")
        legal = is_legal(state, action, min_length=min_length, max_length=max_length)
        if not legal and rate != 0.0:
            raise ValueError(f"nonzero rate assigned to hard-masked action {action.key}")
        if legal:
            filtered[action] = rate
    return filtered


def _ordered_rates(
    rates: Mapping[AtomicAction, float],
) -> tuple[tuple[AtomicAction, float], ...]:
    return tuple(sorted(rates.items(), key=lambda item: item[0].key))


def _total_hazard(rates: Mapping[AtomicAction, float]) -> float:
    total = math.fsum(rates.values())
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("invalid total CTMC hazard")
    return total


def _draw_exponential(rng: random.Random) -> tuple[float, float]:
    uniform = rng.random()
    safe = max(uniform, sys.float_info.min)
    return uniform, -math.log(safe)


def _validate_initial_state(
    initial_state: EditState,
    config: ExactCTMCSamplerConfig,
    seed: int,
) -> None:
    if initial_state.phase is not Phase.ACTIVE:
        raise ValueError("exact CTMC sampler initial state must be ACTIVE")
    if not config.min_length <= len(initial_state.current) <= config.max_length:
        raise ValueError("initial state violates exact CTMC length bounds")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def time_homogeneity_audit(
    state: EditState,
    rate_fn: RateFunction,
    *,
    time: float,
    horizon: float,
    min_length: int,
    max_length: int,
    atol: float,
) -> dict[str, object]:
    """Check whether this state has a time-homogeneous action generator."""

    if not 0.0 <= time < 1.0:
        raise ValueError("time must be in [0,1)")
    upper = min(math.nextafter(1.0, 0.0), max(time, horizon))
    probes = tuple(
        sorted(
            {
                time,
                min(upper, time + 0.25 * max(0.0, upper - time)),
                min(upper, time + 0.50 * max(0.0, upper - time)),
                min(upper, time + 0.75 * max(0.0, upper - time)),
            }
        )
    )
    baseline = _validated_rates(
        state,
        probes[0],
        rate_fn,
        min_length=min_length,
        max_length=max_length,
    )
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    key_mismatch = False
    for probe in probes[1:]:
        candidate = _validated_rates(
            state,
            probe,
            rate_fn,
            min_length=min_length,
            max_length=max_length,
        )
        if set(candidate) != set(baseline):
            key_mismatch = True
            continue
        for action, base_rate in baseline.items():
            delta = abs(candidate[action] - base_rate)
            scale = max(1.0, abs(base_rate), abs(candidate[action]))
            max_abs_delta = max(max_abs_delta, delta)
            max_rel_delta = max(max_rel_delta, delta / scale)
    return {
        "verified": not key_mismatch and max_abs_delta <= atol,
        "probe_times": list(probes),
        "max_abs_delta": max_abs_delta,
        "max_rel_delta": max_rel_delta,
        "key_mismatch": key_mismatch,
        "atol": atol,
    }


def _terminal_result(
    *,
    sampler: str,
    exact_gillespie: bool,
    time_homogeneous: bool,
    initial_state: EditState,
    state: EditState,
    steps: list[ExactCTMCStep],
    seed: int,
    config: ExactCTMCSamplerConfig,
    termination_time: float,
    termination_before_hash: str,
    log_likelihood: float,
    likelihood_semantics: str,
    max_integration_disagreement: float,
    max_root_residual: float,
    max_time_homogeneity_delta: float,
) -> ExactCTMCResult:
    if not math.isfinite(log_likelihood):
        raise FloatingPointError("trajectory log likelihood is not finite")
    return ExactCTMCResult(
        sampler=sampler,
        exact_gillespie=exact_gillespie,
        time_homogeneous=time_homogeneous,
        initial_state=initial_state,
        final_state=state,
        steps=tuple(steps),
        seed=seed,
        min_length=config.min_length,
        max_length=config.max_length,
        horizon=config.horizon,
        termination_time=termination_time,
        termination_before_hash=termination_before_hash,
        trajectory_log_likelihood=log_likelihood,
        likelihood_semantics=likelihood_semantics,
        max_integration_disagreement=max_integration_disagreement,
        max_root_residual=max_root_residual,
        max_time_homogeneity_delta=max_time_homogeneity_delta,
    )


def sample_exact_gillespie(
    initial_state: EditState,
    rate_fn: RateFunction,
    *,
    config: ExactCTMCSamplerConfig,
    seed: int,
) -> ExactCTMCResult:
    """Sample an exact event-time trajectory for a homogeneous generator.

    The function fails closed when rates vary with external time.  This is the
    gate that protects an exact-Gillespie or exact homogeneous-CTMC claim.
    """

    _validate_initial_state(initial_state, config, seed)
    rng = random.Random(seed)
    state = initial_state
    t = 0.0
    steps: list[ExactCTMCStep] = []
    log_likelihood = 0.0
    max_delta = 0.0

    while t < config.horizon and state.phase is Phase.ACTIVE:
        if state.remaining_budget == 0:
            before = state.state_hash
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            return _terminal_result(
                sampler="exact_gillespie_homogeneous",
                exact_gillespie=True,
                time_homogeneous=True,
                initial_state=initial_state,
                state=state,
                steps=steps,
                seed=seed,
                config=config,
                termination_time=t,
                termination_before_hash=before,
                log_likelihood=log_likelihood,
                likelihood_semantics="exact_homogeneous_ctmc",
                max_integration_disagreement=0.0,
                max_root_residual=0.0,
                max_time_homogeneity_delta=max_delta,
            )

        audit = time_homogeneity_audit(
            state,
            rate_fn,
            time=t,
            horizon=config.horizon,
            min_length=config.min_length,
            max_length=config.max_length,
            atol=config.time_homogeneity_atol,
        )
        max_delta = max(max_delta, float(audit["max_abs_delta"]))
        if not bool(audit["verified"]):
            raise TimeInhomogeneousRateError(
                "exact Gillespie gate failed: state-conditioned rates vary with time"
            )
        rates = _validated_rates(
            state,
            t,
            rate_fn,
            min_length=config.min_length,
            max_length=config.max_length,
        )
        ordered = _ordered_rates(rates)
        hazard = _total_hazard(rates)
        before_hash = state.state_hash
        event_uniform, exponential = _draw_exponential(rng)

        if hazard == 0.0:
            edit_legal = enumerate_legal_actions(
                state,
                min_length=config.min_length,
                max_length=config.max_length,
                include_stop=False,
            )
            reason = (
                TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION
                if not edit_legal
                else TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
            )
            state = force_terminate(state, reason)
            return _terminal_result(
                sampler="exact_gillespie_homogeneous",
                exact_gillespie=True,
                time_homogeneous=True,
                initial_state=initial_state,
                state=state,
                steps=steps,
                seed=seed,
                config=config,
                termination_time=t,
                termination_before_hash=before_hash,
                log_likelihood=log_likelihood,
                likelihood_semantics="exact_homogeneous_ctmc",
                max_integration_disagreement=0.0,
                max_root_residual=0.0,
                max_time_homogeneity_delta=max_delta,
            )

        waiting = exponential / hazard
        event_time = t + waiting
        if event_time >= config.horizon:
            survival = -hazard * (config.horizon - t)
            log_likelihood += survival
            steps.append(
                ExactCTMCStep(
                    event_index=len(steps),
                    t_start=t,
                    t_end=config.horizon,
                    waiting_time=config.horizon - t,
                    total_hazard_at_event=hazard,
                    integrated_hazard=hazard * (config.horizon - t),
                    event_uniform=event_uniform,
                    action_uniform=None,
                    selected_action=None,
                    outcome="NO_EVENT_HORIZON",
                    before_hash=before_hash,
                    after_hash=before_hash,
                    candidate_actions_hash=_candidate_actions_hash(ordered),
                    candidate_rates_hash=_candidate_rates_hash(ordered),
                    integration_disagreement=0.0,
                    root_residual=0.0,
                    log_likelihood_increment=survival,
                )
            )
            state = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
            return _terminal_result(
                sampler="exact_gillespie_homogeneous",
                exact_gillespie=True,
                time_homogeneous=True,
                initial_state=initial_state,
                state=state,
                steps=steps,
                seed=seed,
                config=config,
                termination_time=config.horizon,
                termination_before_hash=before_hash,
                log_likelihood=log_likelihood,
                likelihood_semantics="exact_homogeneous_ctmc",
                max_integration_disagreement=0.0,
                max_root_residual=0.0,
                max_time_homogeneity_delta=max_delta,
            )

        action_uniform = rng.random()
        selected = _draw_action(ordered, action_uniform, hazard)
        selected_rate = dict(ordered)[selected]
        transition = apply_action(
            state,
            selected,
            min_length=config.min_length,
            max_length=config.max_length,
        )
        state = transition.after
        increment = -hazard * waiting + math.log(selected_rate)
        log_likelihood += increment
        steps.append(
            ExactCTMCStep(
                event_index=len(steps),
                t_start=t,
                t_end=event_time,
                waiting_time=waiting,
                total_hazard_at_event=hazard,
                integrated_hazard=hazard * waiting,
                event_uniform=event_uniform,
                action_uniform=action_uniform,
                selected_action=selected,
                outcome=selected.kind.value,
                before_hash=before_hash,
                after_hash=state.state_hash,
                candidate_actions_hash=_candidate_actions_hash(ordered),
                candidate_rates_hash=_candidate_rates_hash(ordered),
                integration_disagreement=0.0,
                root_residual=0.0,
                log_likelihood_increment=increment,
            )
        )
        t = event_time
        if state.phase is Phase.HALTED:
            return _terminal_result(
                sampler="exact_gillespie_homogeneous",
                exact_gillespie=True,
                time_homogeneous=True,
                initial_state=initial_state,
                state=state,
                steps=steps,
                seed=seed,
                config=config,
                termination_time=t,
                termination_before_hash=before_hash,
                log_likelihood=log_likelihood,
                likelihood_semantics="exact_homogeneous_ctmc",
                max_integration_disagreement=0.0,
                max_root_residual=0.0,
                max_time_homogeneity_delta=max_delta,
            )
        if state.remaining_budget == 0:
            budget_before = state.state_hash
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            return _terminal_result(
                sampler="exact_gillespie_homogeneous",
                exact_gillespie=True,
                time_homogeneous=True,
                initial_state=initial_state,
                state=state,
                steps=steps,
                seed=seed,
                config=config,
                termination_time=t,
                termination_before_hash=budget_before,
                log_likelihood=log_likelihood,
                likelihood_semantics="exact_homogeneous_ctmc",
                max_integration_disagreement=0.0,
                max_root_residual=0.0,
                max_time_homogeneity_delta=max_delta,
            )

    if state.phase is Phase.ACTIVE:
        before = state.state_hash
        state = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
        return _terminal_result(
            sampler="exact_gillespie_homogeneous",
            exact_gillespie=True,
            time_homogeneous=True,
            initial_state=initial_state,
            state=state,
            steps=steps,
            seed=seed,
            config=config,
            termination_time=t,
            termination_before_hash=before,
            log_likelihood=log_likelihood,
            likelihood_semantics="exact_homogeneous_ctmc",
            max_integration_disagreement=0.0,
            max_root_residual=0.0,
            max_time_homogeneity_delta=max_delta,
        )
    raise AssertionError("exact Gillespie loop exited without a terminal state")


def _integrated_hazard(
    state: EditState,
    start: float,
    end: float,
    rate_fn: RateFunction,
    *,
    config: ExactCTMCSamplerConfig,
):
    if end <= start:
        raise ValueError("integrated-hazard end must be after start")
    certificate = certify_remaining_integrated_hazard(
        state,
        start,
        rate_fn,
        horizon=end,
        min_length=config.min_length,
        max_length=config.max_length,
        lower_order=config.integration_lower_order,
        higher_order=config.integration_higher_order,
        zero_atol=config.integration_convergence_atol,
        convergence_atol=config.integration_convergence_atol,
    )
    if certificate.disagreement > config.integration_convergence_atol:
        raise HazardConvergenceError(
            "nonhomogeneous CTMC integrated hazard did not converge"
        )
    return certificate


def sample_nonhomogeneous_ctmc(
    initial_state: EditState,
    rate_fn: RateFunction,
    *,
    config: ExactCTMCSamplerConfig,
    seed: int,
) -> ExactCTMCResult:
    """Sample a time-dependent CTMC by certified integrated-hazard inversion."""

    _validate_initial_state(initial_state, config, seed)
    rng = random.Random(seed)
    state = initial_state
    t = 0.0
    steps: list[ExactCTMCStep] = []
    log_likelihood = 0.0
    max_disagreement = 0.0
    max_root_residual = 0.0
    termination_before_hash = state.state_hash

    while t < config.horizon and state.phase is Phase.ACTIVE:
        if state.remaining_budget == 0:
            before = state.state_hash
            termination_before_hash = before
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            break

        whole = _integrated_hazard(
            state,
            t,
            config.horizon,
            rate_fn,
            config=config,
        )
        max_disagreement = max(max_disagreement, whole.disagreement)
        event_uniform, exponential = _draw_exponential(rng)
        before_hash = state.state_hash
        if exponential >= whole.integral:
            increment = -whole.integral
            log_likelihood += increment
            termination_before_hash = before_hash
            rates_at_start = _validated_rates(
                state,
                t,
                rate_fn,
                min_length=config.min_length,
                max_length=config.max_length,
            )
            ordered = _ordered_rates(rates_at_start)
            steps.append(
                ExactCTMCStep(
                    event_index=len(steps),
                    t_start=t,
                    t_end=config.horizon,
                    waiting_time=config.horizon - t,
                    total_hazard_at_event=_total_hazard(rates_at_start),
                    integrated_hazard=whole.integral,
                    event_uniform=event_uniform,
                    action_uniform=None,
                    selected_action=None,
                    outcome="NO_EVENT_HORIZON",
                    before_hash=before_hash,
                    after_hash=before_hash,
                    candidate_actions_hash=_candidate_actions_hash(ordered),
                    candidate_rates_hash=_candidate_rates_hash(ordered),
                    integration_disagreement=whole.disagreement,
                    root_residual=0.0,
                    log_likelihood_increment=increment,
                )
            )
            state = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
            t = config.horizon
            break

        low = t
        high = config.horizon
        best = None
        for _ in range(config.max_root_iterations):
            mid = 0.5 * (low + high)
            certificate = _integrated_hazard(
                state,
                t,
                mid,
                rate_fn,
                config=config,
            )
            max_disagreement = max(max_disagreement, certificate.disagreement)
            residual = certificate.integral - exponential
            best = (mid, certificate, abs(residual))
            if abs(residual) <= config.root_atol:
                break
            if residual < 0.0:
                low = mid
            else:
                high = mid
        if best is None or best[2] > config.root_atol:
            raise HazardConvergenceError(
                "nonhomogeneous CTMC event-time inversion did not converge"
            )
        event_time, event_certificate, residual = best
        max_root_residual = max(max_root_residual, residual)
        rates_at_event = _validated_rates(
            state,
            event_time,
            rate_fn,
            min_length=config.min_length,
            max_length=config.max_length,
        )
        ordered = _ordered_rates(rates_at_event)
        hazard = _total_hazard(rates_at_event)
        if hazard <= 0.0:
            raise HazardConvergenceError("event inversion reached a zero-hazard time")
        action_uniform = rng.random()
        selected = _draw_action(ordered, action_uniform, hazard)
        selected_rate = dict(ordered)[selected]
        state = apply_action(
            state,
            selected,
            min_length=config.min_length,
            max_length=config.max_length,
        ).after
        increment = -event_certificate.integral + math.log(selected_rate)
        log_likelihood += increment
        steps.append(
            ExactCTMCStep(
                event_index=len(steps),
                t_start=t,
                t_end=event_time,
                waiting_time=event_time - t,
                total_hazard_at_event=hazard,
                integrated_hazard=event_certificate.integral,
                event_uniform=event_uniform,
                action_uniform=action_uniform,
                selected_action=selected,
                outcome=selected.kind.value,
                before_hash=before_hash,
                after_hash=state.state_hash,
                candidate_actions_hash=_candidate_actions_hash(ordered),
                candidate_rates_hash=_candidate_rates_hash(ordered),
                integration_disagreement=event_certificate.disagreement,
                root_residual=residual,
                log_likelihood_increment=increment,
            )
        )
        t = event_time
        if state.phase is Phase.HALTED:
            termination_before_hash = before_hash
            break
        if state.remaining_budget == 0:
            before = state.state_hash
            termination_before_hash = before
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            break

    if state.phase is Phase.ACTIVE:
        before = state.state_hash
        termination_before_hash = before
        if t >= config.horizon:
            reason = TerminationReason.FORCED_TIME_HORIZON
        else:
            reason = TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION
        state = force_terminate(state, reason)
        if t < config.horizon:
            t = config.horizon

    return _terminal_result(
        sampler="nonhomogeneous_ctmc_integrated_hazard",
        exact_gillespie=False,
        time_homogeneous=False,
        initial_state=initial_state,
        state=state,
        steps=steps,
        seed=seed,
        config=config,
        termination_time=t,
        termination_before_hash=termination_before_hash,
        log_likelihood=log_likelihood,
        likelihood_semantics="numerically_converged_nonhomogeneous_ctmc",
        max_integration_disagreement=max_disagreement,
        max_root_residual=max_root_residual,
        max_time_homogeneity_delta=0.0,
    )


def replay_exact_ctmc_result(
    result: ExactCTMCResult,
    rate_fn: RateFunction,
    *,
    config: ExactCTMCSamplerConfig,
) -> bool:
    """Re-run the exact-event route and require an identical stochastic ledger."""

    try:
        if result.sampler == "exact_gillespie_homogeneous":
            replayed = sample_exact_gillespie(
                result.initial_state, rate_fn, config=config, seed=result.seed
            )
        elif result.sampler == "nonhomogeneous_ctmc_integrated_hazard":
            replayed = sample_nonhomogeneous_ctmc(
                result.initial_state, rate_fn, config=config, seed=result.seed
            )
        else:
            return False
        return (
            replayed.steps == result.steps
            and replayed.final_state == result.final_state
            and replayed.termination_time == result.termination_time
            and replayed.trajectory_log_likelihood == result.trajectory_log_likelihood
        )
    except (AssertionError, FloatingPointError, HazardConvergenceError, TypeError, ValueError):
        return False
