"""Frozen-rate first-order samplers required by MK0.

Both algorithms are numerical approximations.  Neither is exact Gillespie.
Every stochastic decision is recorded so that replay can recompute rates,
hazards, adaptive substeps and pseudo-random draws from the frozen inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from typing import Callable, Mapping, Optional

from .rate_kernel import conditioned_event_distribution, total_hazard
from .state_action import (
    action_to_schema_record,
    apply_action,
    enumerate_legal_actions,
    force_terminate,
    is_legal,
    state_to_schema_record,
    termination_to_schema_record,
)
from .types import (
    ActionType,
    AtomicAction,
    EditState,
    Phase,
    RuntimeMapping,
    TerminationReason,
    TokenOrigin,
    TokenRef,
)

RateFunction = Callable[[EditState, float], Mapping[AtomicAction, float]]


@dataclass(frozen=True)
class RemainingHazardCertificate:
    """Numerical, fail-closed certificate for a remaining hazard integral.

    ``integral`` is the higher-order Gauss--Legendre estimate and
    ``disagreement`` is its absolute difference from the lower-order estimate.
    This is numerical evidence under the supplied frozen rate function, not an
    analytic proof for an arbitrary discontinuous function.
    """

    integral: float
    lower_order_integral: float
    disagreement: float
    lower_order: int
    higher_order: int
    zero_atol: float
    convergence_atol: float

    def __post_init__(self) -> None:
        numeric = (
            self.integral,
            self.lower_order_integral,
            self.disagreement,
            self.zero_atol,
            self.convergence_atol,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError(
                "remaining-hazard certificate must be finite and non-negative"
            )
        if self.lower_order < 2 or self.higher_order <= self.lower_order:
            raise ValueError("remaining-hazard quadrature orders are invalid")

    @property
    def verified_zero(self) -> bool:
        return (
            self.integral <= self.zero_atol
            and self.lower_order_integral <= self.zero_atol
            and self.disagreement <= self.convergence_atol
        )


RemainingHazardVerifier = Callable[
    [EditState, float], RemainingHazardCertificate | float
]


@dataclass(frozen=True)
class StepLog:
    step: int
    t_start: float
    t_end: float
    h: float
    total_hazard: float
    event_probability: float
    event_draw: Optional[float]
    action_draw: Optional[float]
    selected_action: Optional[AtomicAction]
    outcome: str
    before_hash: str
    after_hash: str
    candidate_actions_hash: str
    candidate_rates_hash: str
    adaptive_subdivision_count: int
    rate_recomputed_after_step: bool = True
    parallel_draws: tuple[tuple[str, float], ...] = ()
    parallel_actions: tuple[AtomicAction, ...] = ()


@dataclass(frozen=True)
class SamplerResult:
    sampler: str
    initial_state: EditState
    final_state: EditState
    steps: tuple[StepLog, ...]
    exact_gillespie: bool
    seed: int
    step_size: float
    stability_hazard: Optional[float]
    min_length: int
    max_length: int
    horizon: float
    termination_time: float
    termination_before_hash: str
    remaining_hazard_certificate: Optional[RemainingHazardCertificate] = None
    invalid_joint_proposals: int = 0

    @property
    def edit_events(self) -> int:
        count = 0
        for step in self.steps:
            if step.parallel_actions and step.outcome == "PARALLEL_EVENTS_APPLIED":
                count += sum(
                    action.kind != ActionType.STOP for action in step.parallel_actions
                )
            elif (
                step.selected_action is not None
                and step.selected_action.kind != ActionType.STOP
            ):
                count += 1
        return count


def _validated_rates(
    state: EditState,
    time: float,
    rate_fn: RateFunction,
    *,
    min_length: int,
    max_length: int,
) -> dict[AtomicAction, float]:
    rates = dict(rate_fn(state, time))
    filtered: dict[AtomicAction, float] = {}
    for action, raw_rate in rates.items():
        if not isinstance(action, AtomicAction):
            raise TypeError("rate mapping keys must be AtomicAction instances")
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
            raise TypeError(f"rate for {action.key} must be a real scalar")
        rate = float(raw_rate)
        legal = is_legal(state, action, min_length=min_length, max_length=max_length)
        if not math.isfinite(rate) or rate < 0.0:
            raise FloatingPointError(f"invalid rate for {action.key}")
        if not legal and rate != 0.0:
            raise ValueError(
                f"nonzero rate assigned to hard-masked action {action.key}"
            )
        if legal:
            filtered[action] = rate
    return filtered


def _ordered_rates(
    rates: Mapping[AtomicAction, float],
) -> tuple[tuple[AtomicAction, float], ...]:
    return tuple(sorted(rates.items(), key=lambda item: item[0].key))


def _ledger_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_actions_hash(
    ordered: tuple[tuple[AtomicAction, float], ...],
) -> str:
    return _ledger_hash([action.key for action, _ in ordered])


def _candidate_rates_hash(
    ordered: tuple[tuple[AtomicAction, float], ...],
) -> str:
    # float.hex preserves the exact binary64 value used by the sampler.
    return _ledger_hash([[action.key, float(rate).hex()] for action, rate in ordered])


def _draw_action(
    distribution: Mapping[AtomicAction, float], draw: float
) -> AtomicAction:
    if not distribution:
        raise ValueError("cannot draw from an empty action distribution")
    cumulative = 0.0
    ordered = sorted(distribution.items(), key=lambda item: item[0].key)
    for action, probability in ordered:
        cumulative += probability
        if draw < cumulative:
            return action
    return ordered[-1][0]


def _adaptive_substep(
    proposed_h: float, hazard: float, stability_hazard: float
) -> tuple[float, int]:
    if hazard <= 0.0 or proposed_h * hazard <= stability_hazard:
        return proposed_h, 0
    required_pieces = proposed_h * hazard / stability_hazard
    if not math.isfinite(required_pieces):
        raise FloatingPointError("adaptive subdivision count overflowed")
    pieces = int(math.ceil(required_pieces))
    h = proposed_h / pieces
    if not math.isfinite(h) or h <= 0.0:
        raise FloatingPointError("adaptive substep underflowed")
    return h, pieces - 1


def certify_remaining_integrated_hazard(
    state: EditState,
    start_time: float,
    rate_fn: RateFunction,
    *,
    horizon: float,
    min_length: int,
    max_length: int,
    lower_order: int = 64,
    higher_order: int = 128,
    zero_atol: float = 1.0e-10,
    convergence_atol: float = 1.0e-10,
) -> RemainingHazardCertificate:
    """Estimate the frozen-state remaining total hazard with two GL rules."""

    if not 0.0 <= start_time <= horizon <= 1.0:
        raise ValueError("remaining-hazard interval must lie in [0,1]")
    if (
        not isinstance(lower_order, int)
        or isinstance(lower_order, bool)
        or not isinstance(higher_order, int)
        or isinstance(higher_order, bool)
        or not 2 <= lower_order < higher_order <= 4_096
    ):
        raise ValueError("remaining-hazard quadrature orders are invalid")
    if zero_atol < 0.0 or convergence_atol < 0.0:
        raise ValueError("remaining-hazard tolerances must be non-negative")

    if start_time == horizon:
        return RemainingHazardCertificate(
            0.0,
            0.0,
            0.0,
            lower_order,
            higher_order,
            zero_atol,
            convergence_atol,
        )

    # Numpy is already a frozen project dependency.  Importing here keeps the
    # non-certifying sampler path free of an eager numerical-library import.
    import numpy as np

    def integrate(order: int) -> float:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        midpoint = 0.5 * (start_time + horizon)
        half_width = 0.5 * (horizon - start_time)
        value = 0.0
        for node, weight in zip(nodes, weights):
            external_time = midpoint + half_width * float(node)
            rates = _validated_rates(
                state,
                external_time,
                rate_fn,
                min_length=min_length,
                max_length=max_length,
            )
            value += float(weight) * total_hazard(rates)
        estimate = half_width * value
        if not math.isfinite(estimate) or estimate < 0.0:
            raise FloatingPointError("remaining integrated hazard is invalid")
        return estimate

    lower = integrate(lower_order)
    higher = integrate(higher_order)
    return RemainingHazardCertificate(
        integral=higher,
        lower_order_integral=lower,
        disagreement=abs(higher - lower),
        lower_order=lower_order,
        higher_order=higher_order,
        zero_atol=zero_atol,
        convergence_atol=convergence_atol,
    )


def _coerce_remaining_certificate(
    value: RemainingHazardCertificate | float,
    *,
    zero_atol: float,
    convergence_atol: float,
) -> RemainingHazardCertificate:
    # bool is an int subclass.  Reject it explicitly so the legacy
    # ``lambda: True`` fake certificate fails closed.
    if isinstance(value, bool):
        raise TypeError("boolean remaining-hazard predicates are not evidence")
    if isinstance(value, RemainingHazardCertificate):
        return value
    if not isinstance(value, (int, float)):
        raise TypeError("remaining-hazard verifier must return a number or certificate")
    integral = float(value)
    return RemainingHazardCertificate(
        integral=integral,
        lower_order_integral=integral,
        disagreement=0.0,
        lower_order=2,
        higher_order=3,
        zero_atol=zero_atol,
        convergence_atol=convergence_atol,
    )


def constrained_single_event_first_order(
    initial_state: EditState,
    rate_fn: RateFunction,
    *,
    step_size: float,
    stability_hazard: float,
    min_length: int,
    max_length: int,
    seed: int,
    horizon: float = 1.0,
    remaining_hazard_verifier: Optional[RemainingHazardVerifier] = None,
    remaining_hazard_zero_atol: float = 1.0e-10,
    remaining_hazard_convergence_atol: float = 1.0e-10,
    remaining_integrated_hazard_is_zero: Optional[RemainingHazardVerifier] = None,
) -> SamplerResult:
    """Endpoint single-event frozen-rate approximation.

    A zero instantaneous hazard advances time without normalizing an action
    distribution.  Early termination requires a finite non-negative numerical
    remaining-integral estimate.  The deprecated predicate-named argument is a
    fail-closed compatibility route: a boolean return is rejected.
    """

    if step_size <= 0.0 or stability_hazard <= 0.0:
        raise ValueError("step_size and stability threshold must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < horizon <= 1.0:
        raise ValueError("horizon must lie in (0,1]")
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid sampler length bounds")
    if initial_state.phase != Phase.ACTIVE:
        raise ValueError("sampler initial state must be ACTIVE")
    if not min_length <= len(initial_state.current) <= max_length:
        raise ValueError("sampler initial state violates length bounds")
    if (
        remaining_hazard_verifier is not None
        and remaining_integrated_hazard_is_zero is not None
    ):
        raise ValueError("supply only one remaining-hazard verifier")
    verifier = remaining_hazard_verifier or remaining_integrated_hazard_is_zero

    rng = random.Random(seed)
    state = initial_state
    t = 0.0
    step_index = 0
    logs: list[StepLog] = []
    termination_time: Optional[float] = None
    termination_before_hash: Optional[str] = None
    remaining_certificate: Optional[RemainingHazardCertificate] = None

    while t < horizon and state.phase == Phase.ACTIVE:
        if state.remaining_budget == 0:
            termination_time = t
            termination_before_hash = state.state_hash
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            break
        proposed_h = min(step_size, horizon - t)
        rates = _validated_rates(
            state,
            t,
            rate_fn,
            min_length=min_length,
            max_length=max_length,
        )
        ordered = _ordered_rates(rates)
        hazard = total_hazard(rates)
        h, subdivisions = _adaptive_substep(proposed_h, hazard, stability_hazard)
        t_end = min(horizon, t + h)
        if not t_end > t:
            raise FloatingPointError("adaptive substep made no time progress")
        before_hash = state.state_hash

        if hazard == 0.0:
            if verifier is not None:
                candidate = _coerce_remaining_certificate(
                    verifier(state, t),
                    zero_atol=remaining_hazard_zero_atol,
                    convergence_atol=remaining_hazard_convergence_atol,
                )
                if candidate.verified_zero:
                    remaining_certificate = candidate
                    termination_time = t
                    termination_before_hash = state.state_hash
                    state = force_terminate(
                        state,
                        TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD,
                    )
                    break
            logs.append(
                StepLog(
                    step=step_index,
                    t_start=t,
                    t_end=t_end,
                    h=h,
                    total_hazard=0.0,
                    event_probability=0.0,
                    event_draw=None,
                    action_draw=None,
                    selected_action=None,
                    outcome="NO_EVENT",
                    before_hash=before_hash,
                    after_hash=before_hash,
                    candidate_actions_hash=_candidate_actions_hash(ordered),
                    candidate_rates_hash=_candidate_rates_hash(ordered),
                    adaptive_subdivision_count=subdivisions,
                )
            )
            t = t_end
            step_index += 1
            continue

        event_probability = -math.expm1(-h * hazard)
        if not 0.0 <= event_probability <= 1.0:
            raise FloatingPointError("invalid event probability")
        event_draw = rng.random()
        selected: Optional[AtomicAction] = None
        action_draw: Optional[float] = None
        outcome = "NO_EVENT"
        if event_draw < event_probability:
            distribution = conditioned_event_distribution(rates)
            action_draw = rng.random()
            selected = _draw_action(distribution, action_draw)
            state = apply_action(
                state,
                selected,
                min_length=min_length,
                max_length=max_length,
            ).after
            outcome = selected.kind.value

        # Step hashes describe only the numerical CTMC substep.  Any forced
        # termination is a separate, auditable event after this log entry.
        after_event_hash = state.state_hash
        logs.append(
            StepLog(
                step=step_index,
                t_start=t,
                t_end=t_end,
                h=h,
                total_hazard=hazard,
                event_probability=event_probability,
                event_draw=event_draw,
                action_draw=action_draw,
                selected_action=selected,
                outcome=outcome,
                before_hash=before_hash,
                after_hash=after_event_hash,
                candidate_actions_hash=_candidate_actions_hash(ordered),
                candidate_rates_hash=_candidate_rates_hash(ordered),
                adaptive_subdivision_count=subdivisions,
            )
        )
        t = t_end
        step_index += 1

        if state.phase == Phase.HALTED:
            termination_time = t_end
            termination_before_hash = before_hash
            break
        if selected is not None and state.remaining_budget == 0:
            termination_time = t_end
            termination_before_hash = state.state_hash
            state = force_terminate(state, TerminationReason.FORCED_BUDGET)
            break

    if state.phase == Phase.ACTIVE:
        edit_legal = enumerate_legal_actions(
            state, min_length=min_length, max_length=max_length, include_stop=False
        )
        reason = (
            TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION
            if not edit_legal
            else TerminationReason.FORCED_TIME_HORIZON
        )
        termination_time = t
        termination_before_hash = state.state_hash
        state = force_terminate(state, reason)

    assert termination_time is not None and termination_before_hash is not None
    return SamplerResult(
        sampler="constrained_single_event_first_order",
        initial_state=initial_state,
        final_state=state,
        steps=tuple(logs),
        exact_gillespie=False,
        seed=seed,
        step_size=step_size,
        stability_hazard=stability_hazard,
        min_length=min_length,
        max_length=max_length,
        horizon=horizon,
        termination_time=termination_time,
        termination_before_hash=termination_before_hash,
        remaining_hazard_certificate=remaining_certificate,
    )


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-13, abs_tol=1.0e-15)


def replay_constrained_result(
    result: SamplerResult,
    rate_fn: RateFunction,
    *,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    remaining_hazard_verifier: Optional[RemainingHazardVerifier] = None,
) -> bool:
    """Recompute and replay the complete constrained stochastic ledger."""

    try:
        if result.sampler != "constrained_single_event_first_order":
            return False
        if result.exact_gillespie is not False or result.stability_hazard is None:
            return False
        resolved_min = result.min_length if min_length is None else min_length
        resolved_max = result.max_length if max_length is None else max_length
        if resolved_min != result.min_length or resolved_max != result.max_length:
            return False
        rng = random.Random(result.seed)
        state = result.initial_state
        t = 0.0
        for index, log in enumerate(result.steps):
            if (
                state.phase != Phase.ACTIVE
                or log.step != index
                or not t < result.horizon
            ):
                return False
            if state.state_hash != log.before_hash or not _float_equal(log.t_start, t):
                return False
            proposed_h = min(result.step_size, result.horizon - t)
            rates = _validated_rates(
                state,
                t,
                rate_fn,
                min_length=resolved_min,
                max_length=resolved_max,
            )
            ordered = _ordered_rates(rates)
            hazard = total_hazard(rates)
            h, subdivisions = _adaptive_substep(
                proposed_h, hazard, result.stability_hazard
            )
            expected_end = min(result.horizon, t + h)
            if not expected_end > t:
                return False
            if not all(
                (
                    _float_equal(log.h, h),
                    _float_equal(log.t_end, expected_end),
                    _float_equal(log.total_hazard, hazard),
                    log.adaptive_subdivision_count == subdivisions,
                    log.candidate_actions_hash == _candidate_actions_hash(ordered),
                    log.candidate_rates_hash == _candidate_rates_hash(ordered),
                    log.rate_recomputed_after_step is True,
                )
            ):
                return False

            event_probability = 0.0 if hazard == 0.0 else -math.expm1(-h * hazard)
            if not _float_equal(log.event_probability, event_probability):
                return False
            selected: Optional[AtomicAction] = None
            if hazard == 0.0:
                if (
                    log.event_draw is not None
                    or log.action_draw is not None
                    or log.selected_action is not None
                    or log.outcome != "NO_EVENT"
                ):
                    return False
            else:
                expected_event_draw = rng.random()
                if log.event_draw is None or log.event_draw != expected_event_draw:
                    return False
                if expected_event_draw < event_probability:
                    expected_action_draw = rng.random()
                    distribution = conditioned_event_distribution(rates)
                    selected = _draw_action(distribution, expected_action_draw)
                    if (
                        log.action_draw != expected_action_draw
                        or log.selected_action != selected
                        or log.outcome != selected.kind.value
                    ):
                        return False
                    state = apply_action(
                        state,
                        selected,
                        min_length=resolved_min,
                        max_length=resolved_max,
                    ).after
                elif (
                    log.action_draw is not None
                    or log.selected_action is not None
                    or log.outcome != "NO_EVENT"
                ):
                    return False
            if state.state_hash != log.after_hash:
                return False
            t = expected_end

        reason = result.final_state.termination_reason
        if reason is None:
            return False
        if not _float_equal(t, result.termination_time):
            return False
        if reason == TerminationReason.LEARNED_STOP:
            if (
                state.phase != Phase.HALTED
                or not result.steps
                or result.steps[-1].selected_action != AtomicAction(ActionType.STOP)
                or result.steps[-1].before_hash != result.termination_before_hash
            ):
                return False
        else:
            if (
                state.phase != Phase.ACTIVE
                or state.state_hash != result.termination_before_hash
            ):
                return False
            if reason == TerminationReason.FORCED_BUDGET:
                if state.remaining_budget != 0 or (
                    result.steps
                    and (
                        result.steps[-1].selected_action is None
                        or result.steps[-1].selected_action.kind == ActionType.STOP
                    )
                ):
                    return False
            elif reason == TerminationReason.FORCED_NO_LEGAL_EDIT_ACTION:
                if state.remaining_budget == 0 or enumerate_legal_actions(
                    state,
                    min_length=resolved_min,
                    max_length=resolved_max,
                    include_stop=False,
                ):
                    return False
            elif reason == TerminationReason.FORCED_TIME_HORIZON:
                if not _float_equal(t, result.horizon):
                    return False
            elif reason == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD:
                if remaining_hazard_verifier is None:
                    return False
                rates = _validated_rates(
                    state,
                    t,
                    rate_fn,
                    min_length=resolved_min,
                    max_length=resolved_max,
                )
                if total_hazard(rates) != 0.0:
                    return False
                certificate = _coerce_remaining_certificate(
                    remaining_hazard_verifier(state, t),
                    zero_atol=(
                        result.remaining_hazard_certificate.zero_atol
                        if result.remaining_hazard_certificate
                        else 1.0e-10
                    ),
                    convergence_atol=(
                        result.remaining_hazard_certificate.convergence_atol
                        if result.remaining_hazard_certificate
                        else 1.0e-10
                    ),
                )
                if (
                    not certificate.verified_zero
                    or certificate != result.remaining_hazard_certificate
                ):
                    return False
            else:
                return False
            state = force_terminate(state, reason)
        return state == result.final_state
    except (AssertionError, FloatingPointError, OverflowError, TypeError, ValueError):
        return False


def _joint_conflict_free(actions: list[AtomicAction]) -> bool:
    edits = [action for action in actions if action.kind != ActionType.STOP]
    if any(action.kind == ActionType.STOP for action in actions) and edits:
        return False
    occupied = set()
    for action in edits:
        key = (
            ("token", action.position)
            if action.kind in (ActionType.SUB, ActionType.DEL)
            else ("gap", action.position)
        )
        if key in occupied:
            return False
        occupied.add(key)
    return True


def _canonical_parallel_order(actions: list[AtomicAction]) -> None:
    order = {
        ActionType.SUB: 0,
        ActionType.DEL: 1,
        ActionType.INS: 2,
        ActionType.STOP: 3,
    }
    actions.sort(
        key=lambda action: (
            order[action.kind],
            int(action.position or 0),
            action.token or "",
        )
    )


def _apply_parallel_pre_step_actions(
    state: EditState,
    actions: tuple[AtomicAction, ...],
    *,
    min_length: int,
    max_length: int,
) -> EditState:
    """Apply conflict-free events defined in the same pre-step coordinates.

    SUB is applied before length changes, DEL from right to left, and each INS
    gap is mapped through surviving original tokens.  This is an exact
    coordinate transform for the proposed simultaneous edit set; it is not a
    projection or repair of an incompatible joint proposal.
    """

    if not actions:
        return state
    proposed = list(actions)
    if not _joint_conflict_free(proposed):
        raise ValueError("parallel actions conflict in pre-step coordinates")
    if any(
        not is_legal(state, action, min_length=min_length, max_length=max_length)
        for action in proposed
    ):
        raise ValueError("parallel action is illegal under the pre-step mask")
    edits = [action for action in proposed if action.kind != ActionType.STOP]
    if not edits:
        return apply_action(
            state,
            AtomicAction(ActionType.STOP),
            min_length=min_length,
            max_length=max_length,
        ).after
    if len(edits) > state.remaining_budget:
        raise ValueError("parallel proposal exceeds remaining budget")
    final_length = (
        len(state.current)
        + sum(action.kind == ActionType.INS for action in edits)
        - sum(action.kind == ActionType.DEL for action in edits)
    )
    if not min_length <= final_length <= max_length:
        raise ValueError("parallel proposal violates final length bounds")

    substitutions = {
        int(action.position): str(action.token)
        for action in edits
        if action.kind == ActionType.SUB
    }
    deletions = {
        int(action.position) for action in edits if action.kind == ActionType.DEL
    }

    # Assign inserted stable IDs in the same canonical event order recorded by
    # the history, then construct sequence and mapping atomically.  This avoids
    # illegal transient lengths for a jointly valid DEL+INS proposal.
    canonical = list(edits)
    _canonical_parallel_order(canonical)
    history = state.history
    insertion_refs: dict[int, tuple[str, TokenRef]] = {}
    for action in canonical:
        if action.kind == ActionType.INS:
            next_event_id = history.executed + 1
            insertion_refs[int(action.position)] = (
                str(action.token),
                TokenRef(
                    TokenOrigin.INSERTED,
                    f"ins:{next_event_id}",
                    source_index=None,
                    protected=False,
                ),
            )
        history = history.append(action)

    sequence: list[str] = []
    refs: list[TokenRef] = []

    def append_gap(gap: int) -> None:
        inserted = insertion_refs.get(gap)
        if inserted is not None:
            token, ref = inserted
            sequence.append(token)
            refs.append(ref)

    append_gap(0)
    for index, (token, ref) in enumerate(zip(state.current, state.mapping.tokens)):
        if index not in deletions:
            sequence.append(substitutions.get(index, token))
            refs.append(ref)
        append_gap(index + 1)
    return replace(
        state,
        current="".join(sequence),
        mapping=RuntimeMapping.rebuild(tuple(refs)),
        remaining_budget=state.remaining_budget - len(edits),
        history=history,
    )


def paper_first_order_parallel(
    initial_state: EditState,
    rate_fn: RateFunction,
    *,
    step_size: float,
    min_length: int,
    max_length: int,
    seed: int,
    horizon: float = 1.0,
) -> SamplerResult:
    """Paper-style independent parallel first-order numerical reference.

    Incompatible joint proposals are reported, not repaired or presented as
    valid UTR samples.  This reference never serves as the strict-budget gate.
    """

    if step_size <= 0.0 or not 0.0 < horizon <= 1.0:
        raise ValueError("paper sampler step size/horizon are invalid")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid sampler length bounds")
    if initial_state.phase != Phase.ACTIVE:
        raise ValueError("sampler initial state must be ACTIVE")
    if not min_length <= len(initial_state.current) <= max_length:
        raise ValueError("sampler initial state violates length bounds")
    rng = random.Random(seed)
    state = initial_state
    logs: list[StepLog] = []
    invalid_joint = 0
    t = 0.0
    step_index = 0
    termination_time: Optional[float] = None
    termination_before_hash: Optional[str] = None
    while t < horizon and state.phase == Phase.ACTIVE:
        h = min(step_size, horizon - t)
        if not t + h > t:
            raise FloatingPointError("paper substep made no time progress")
        rates = _validated_rates(
            state, t, rate_fn, min_length=min_length, max_length=max_length
        )
        ordered = _ordered_rates(rates)
        draws = tuple((action.key, rng.random()) for action, _ in ordered)
        proposed_list = [
            action
            for (action, rate), (_, draw) in zip(ordered, draws)
            if draw < -math.expm1(-h * rate)
        ]
        _canonical_parallel_order(proposed_list)
        proposed = tuple(proposed_list)
        before = state.state_hash
        outcome = "NO_EVENT"
        if proposed:
            try:
                state = _apply_parallel_pre_step_actions(
                    state,
                    proposed,
                    min_length=min_length,
                    max_length=max_length,
                )
                outcome = "PARALLEL_EVENTS_APPLIED"
            except ValueError:
                invalid_joint += 1
                outcome = "INVALID_JOINT_PROPOSAL_REPORTED"
        hazard = total_hazard(rates)
        logs.append(
            StepLog(
                step=step_index,
                t_start=t,
                t_end=t + h,
                h=h,
                total_hazard=hazard,
                event_probability=-math.expm1(-h * hazard),
                event_draw=None,
                action_draw=None,
                selected_action=None,
                outcome=outcome,
                before_hash=before,
                after_hash=state.state_hash,
                candidate_actions_hash=_candidate_actions_hash(ordered),
                candidate_rates_hash=_candidate_rates_hash(ordered),
                adaptive_subdivision_count=0,
                parallel_draws=draws,
                parallel_actions=proposed,
            )
        )
        t += h
        step_index += 1
        if state.phase == Phase.HALTED:
            termination_time = t
            termination_before_hash = before
            break
    if state.phase == Phase.ACTIVE:
        termination_time = t
        termination_before_hash = state.state_hash
        state = force_terminate(state, TerminationReason.FORCED_TIME_HORIZON)
    assert termination_time is not None and termination_before_hash is not None
    return SamplerResult(
        sampler="paper_first_order_parallel",
        initial_state=initial_state,
        final_state=state,
        steps=tuple(logs),
        exact_gillespie=False,
        seed=seed,
        step_size=step_size,
        stability_hazard=None,
        min_length=min_length,
        max_length=max_length,
        horizon=horizon,
        termination_time=termination_time,
        termination_before_hash=termination_before_hash,
        invalid_joint_proposals=invalid_joint,
    )


def replay_paper_result(
    result: SamplerResult,
    rate_fn: RateFunction,
    *,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> bool:
    """Recompute and replay all paper-reference Bernoulli trials."""

    try:
        if result.sampler != "paper_first_order_parallel":
            return False
        resolved_min = result.min_length if min_length is None else min_length
        resolved_max = result.max_length if max_length is None else max_length
        if resolved_min != result.min_length or resolved_max != result.max_length:
            return False
        rng = random.Random(result.seed)
        state = result.initial_state
        t = 0.0
        invalid_joint = 0
        for index, log in enumerate(result.steps):
            if (
                state.phase != Phase.ACTIVE
                or log.step != index
                or not t < result.horizon
            ):
                return False
            h = min(result.step_size, result.horizon - t)
            if not t + h > t:
                return False
            if (
                state.state_hash != log.before_hash
                or not _float_equal(log.t_start, t)
                or not _float_equal(log.h, h)
                or not _float_equal(log.t_end, t + h)
                or log.adaptive_subdivision_count != 0
                or log.event_draw is not None
                or log.action_draw is not None
                or log.selected_action is not None
            ):
                return False
            rates = _validated_rates(
                state,
                t,
                rate_fn,
                min_length=resolved_min,
                max_length=resolved_max,
            )
            ordered = _ordered_rates(rates)
            hazard = total_hazard(rates)
            expected_probability = -math.expm1(-h * hazard)
            if (
                not _float_equal(log.total_hazard, hazard)
                or not _float_equal(log.event_probability, expected_probability)
                or log.candidate_actions_hash != _candidate_actions_hash(ordered)
                or log.candidate_rates_hash != _candidate_rates_hash(ordered)
                or log.rate_recomputed_after_step is not True
            ):
                return False
            expected_draws = tuple((action.key, rng.random()) for action, _ in ordered)
            if expected_draws != log.parallel_draws:
                return False
            proposed_list = [
                action
                for (action, rate), (_, draw) in zip(ordered, expected_draws)
                if draw < -math.expm1(-h * rate)
            ]
            _canonical_parallel_order(proposed_list)
            proposed = tuple(proposed_list)
            if proposed != log.parallel_actions:
                return False
            expected_outcome = "NO_EVENT"
            if proposed:
                try:
                    state = _apply_parallel_pre_step_actions(
                        state,
                        proposed,
                        min_length=resolved_min,
                        max_length=resolved_max,
                    )
                    expected_outcome = "PARALLEL_EVENTS_APPLIED"
                except ValueError:
                    invalid_joint += 1
                    expected_outcome = "INVALID_JOINT_PROPOSAL_REPORTED"
            if expected_outcome != log.outcome or state.state_hash != log.after_hash:
                return False
            t += h
        if invalid_joint != result.invalid_joint_proposals:
            return False
        reason = result.final_state.termination_reason
        if reason is None or not _float_equal(t, result.termination_time):
            return False
        if reason == TerminationReason.LEARNED_STOP:
            if state.phase != Phase.HALTED:
                return False
            if (
                not result.steps
                or AtomicAction(ActionType.STOP)
                not in result.steps[-1].parallel_actions
                or result.steps[-1].before_hash != result.termination_before_hash
            ):
                return False
        elif reason == TerminationReason.FORCED_TIME_HORIZON:
            if (
                state.phase != Phase.ACTIVE
                or state.state_hash != result.termination_before_hash
            ):
                return False
            if not _float_equal(t, result.horizon):
                return False
            state = force_terminate(state, reason)
        else:
            return False
        return state == result.final_state
    except (AssertionError, FloatingPointError, OverflowError, TypeError, ValueError):
        return False


def _simple_parallel_action_record(action: AtomicAction) -> dict[str, object]:
    return {
        "action_type": action.kind.value,
        "position": action.position,
        "nucleotide": action.token,
        "coordinate_system": "pre_step_current_state_zero_based",
    }


def sampler_result_to_schema_record(
    result: SamplerResult,
    rate_fn: RateFunction,
    *,
    trajectory_id: str,
    source_id: str,
    remaining_hazard_verifier: Optional[RemainingHazardVerifier] = None,
) -> dict[str, object]:
    """Serialize an actual result to ``edit_trajectory_v1``.

    The serializer independently invokes full replay.  It never writes a PASS
    replay claim based only on stored state hashes.
    """

    if result.sampler == "constrained_single_event_first_order":
        replay_ok = replay_constrained_result(
            result,
            rate_fn,
            remaining_hazard_verifier=remaining_hazard_verifier,
        )
        semantics = "endpoint_single_event_frozen_rate_approximation"
    elif result.sampler == "paper_first_order_parallel":
        replay_ok = replay_paper_result(result, rate_fn)
        semantics = "fixed_grid_parallel_first_order_approximation"
    else:
        raise ValueError("unsupported sampler result")

    state = result.initial_state
    step_records: list[dict[str, object]] = []
    matched = 0
    for log in result.steps:
        selected_record: Optional[dict[str, object]] = None
        parallel_trials: list[dict[str, object]] = []
        if result.sampler == "paper_first_order_parallel":
            ordered = _ordered_rates(
                _validated_rates(
                    state,
                    log.t_start,
                    rate_fn,
                    min_length=result.min_length,
                    max_length=result.max_length,
                )
            )
            if tuple(action.key for action, _ in ordered) != tuple(
                key for key, _ in log.parallel_draws
            ):
                raise ValueError(
                    "paper trajectory candidate ledger does not match rates"
                )
            parallel_trials = [
                {
                    "action_key": action.key,
                    "rate": rate,
                    "uniform": draw,
                    "event_probability": -math.expm1(-log.h * rate),
                    "proposed": action in log.parallel_actions,
                }
                for (action, rate), (_, draw) in zip(ordered, log.parallel_draws)
            ]
        if log.selected_action is not None:
            transition = apply_action(
                state,
                log.selected_action,
                min_length=result.min_length,
                max_length=result.max_length,
            )
            selected_record = action_to_schema_record(
                transition,
                action_id=f"{trajectory_id}:step:{log.step}",
                external_time=log.t_end,
            )
            state = transition.after
        elif log.parallel_actions and log.outcome == "PARALLEL_EVENTS_APPLIED":
            state = _apply_parallel_pre_step_actions(
                state,
                log.parallel_actions,
                min_length=result.min_length,
                max_length=result.max_length,
            )
        matched += int(state.state_hash == log.after_hash)
        step_records.append(
            {
                "step_index": log.step,
                "t_start": log.t_start,
                "t_end": log.t_end,
                "substep_h": log.h,
                "total_hazard": log.total_hazard,
                "event_probability": log.event_probability,
                "event_uniform": log.event_draw,
                "action_uniform": log.action_draw,
                "outcome": log.outcome,
                "candidate_actions_hash": log.candidate_actions_hash,
                "candidate_rates_hash": log.candidate_rates_hash,
                "selected_action": selected_record,
                "parallel_trials": parallel_trials,
                "parallel_actions": [
                    _simple_parallel_action_record(action)
                    for action in log.parallel_actions
                ],
                "state_hash_before": log.before_hash,
                "state_hash_after": log.after_hash,
                "rate_recomputed_after_step": log.rate_recomputed_after_step,
                "adaptive_subdivision_count": log.adaptive_subdivision_count,
            }
        )

    reason = result.final_state.termination_reason
    if reason is None:
        raise ValueError("sampler result lacks terminal reason")
    termination = termination_to_schema_record(
        reason=reason,
        external_time=result.termination_time,
        state_hash_before=result.termination_before_hash,
        state_hash_after=result.final_state.state_hash,
        remaining_integrated_total_hazard=(
            0.0
            if reason == TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
            else None
        ),
    )
    final_state = state_to_schema_record(
        result.final_state,
        source_id=source_id,
        external_time=result.termination_time,
        parent_state_hash=result.termination_before_hash,
        termination=termination,
    )
    certificate_record = None
    if result.remaining_hazard_certificate is not None:
        certificate_record = {
            "integral": result.remaining_hazard_certificate.integral,
            "lower_order_integral": result.remaining_hazard_certificate.lower_order_integral,
            "disagreement": result.remaining_hazard_certificate.disagreement,
            "lower_order": result.remaining_hazard_certificate.lower_order,
            "higher_order": result.remaining_hazard_certificate.higher_order,
            "zero_atol": result.remaining_hazard_certificate.zero_atol,
            "convergence_atol": result.remaining_hazard_certificate.convergence_atol,
        }
    return {
        "schema_version": "edit_trajectory_v1",
        "trajectory_id": trajectory_id,
        "source_id": source_id,
        "seed": result.seed,
        "sampler": result.sampler,
        "sampler_semantics": semantics,
        "exact_gillespie": False,
        "time_direction": "source_at_0_to_target_at_1",
        "sampler_config": {
            "step_size": result.step_size,
            "stability_hazard": result.stability_hazard,
            "min_length": result.min_length,
            "max_length": result.max_length,
            "horizon": result.horizon,
            "rate_evaluation": "frozen_at_substep_start",
        },
        "remaining_hazard_certificate": certificate_record,
        "initial_state": state_to_schema_record(
            result.initial_state, source_id=source_id, external_time=0.0
        ),
        "steps": step_records,
        "final_state": final_state,
        "run_status": "COMPLETED",
        "termination": termination,
        "replay": {
            "status": "PASS" if replay_ok else "FAIL",
            "replayed_step_count": len(result.steps) if replay_ok else 0,
            "state_hash_match_fraction": (
                matched / len(result.steps) if result.steps else 1.0
            ),
        },
    }
