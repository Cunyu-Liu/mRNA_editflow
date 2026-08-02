"""Non-negative factorized action hazards and extended-state generator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Optional, TypeVar

from .state_action import apply_action, enumerate_legal_actions
from .types import ALPHABET, ActionType, AtomicAction, EditState

Scalar = TypeVar("Scalar")


@dataclass(frozen=True)
class FactorizedRates:
    """External factorization of absolute CTMC hazards.

    Operation intensities are absolute non-negative rates.  Token maps are
    conditional distributions, not additional intensities.
    """

    ins_operation: tuple[float, ...]
    ins_token_probs: tuple[Mapping[str, float], ...]
    sub_operation: tuple[float, ...]
    sub_token_probs: tuple[Mapping[str, float], ...]
    delete: tuple[float, ...]
    stop: float

    @classmethod
    def constant(
        cls,
        state: EditState,
        *,
        ins: float = 0.5,
        sub: float = 0.7,
        delete: float = 0.3,
        stop: float = 0.2,
    ) -> "FactorizedRates":
        ins_q = {token: 0.25 for token in ALPHABET}
        sub_q = tuple(
            {token: 1.0 / 3.0 for token in ALPHABET if token != old}
            for old in state.current
        )
        return cls(
            ins_operation=(ins,) * (len(state.current) + 1),
            ins_token_probs=(ins_q,) * (len(state.current) + 1),
            sub_operation=(sub,) * len(state.current),
            sub_token_probs=sub_q,
            delete=(delete,) * len(state.current),
            stop=stop,
        )


def _finite_nonnegative(name: str, values: tuple[float, ...]) -> None:
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise FloatingPointError(f"{name} contains negative, NaN or Inf rate")


def validate_factorization(
    state: EditState, rates: FactorizedRates, *, atol: float = 1.0e-10
) -> None:
    length = len(state.current)
    if (
        len(rates.ins_operation) != length + 1
        or len(rates.ins_token_probs) != length + 1
    ):
        raise ValueError("INS factorization shape mismatch")
    if not (
        len(rates.sub_operation)
        == len(rates.sub_token_probs)
        == len(rates.delete)
        == length
    ):
        raise ValueError("SUB/DEL factorization shape mismatch")
    _finite_nonnegative("INS", rates.ins_operation)
    _finite_nonnegative("SUB", rates.sub_operation)
    _finite_nonnegative("DEL", rates.delete)
    _finite_nonnegative("STOP", (rates.stop,))
    for gap, distribution in enumerate(rates.ins_token_probs):
        if set(distribution) != set(ALPHABET):
            raise ValueError(f"INS Q[{gap}] must contain exactly A,C,G,U")
        _finite_nonnegative(f"INS Q[{gap}]", tuple(distribution.values()))
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=atol, rel_tol=0.0):
            raise ValueError(f"INS Q[{gap}] is not normalized")
    for pos, (old, distribution) in enumerate(
        zip(state.current, rates.sub_token_probs)
    ):
        if set(distribution) != set(ALPHABET) - {old}:
            raise ValueError(
                f"SUB Q[{pos}] must contain exactly the three legal tokens"
            )
        _finite_nonnegative(f"SUB Q[{pos}]", tuple(distribution.values()))
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=atol, rel_tol=0.0):
            raise ValueError(f"SUB Q[{pos}] is not normalized")


def enumerate_action_rates(
    state: EditState,
    rates: FactorizedRates,
    *,
    min_length: int,
    max_length: int,
) -> dict[AtomicAction, float]:
    """Apply the hard mask before exposing any event distribution."""

    validate_factorization(state, rates)
    legal = enumerate_legal_actions(
        state, min_length=min_length, max_length=max_length, include_stop=True
    )
    result: dict[AtomicAction, float] = {}
    for action in legal:
        if action.kind == ActionType.INS:
            value = (
                rates.ins_operation[int(action.position)]
                * rates.ins_token_probs[int(action.position)][str(action.token)]
            )
        elif action.kind == ActionType.SUB:
            value = (
                rates.sub_operation[int(action.position)]
                * rates.sub_token_probs[int(action.position)][str(action.token)]
            )
        elif action.kind == ActionType.DEL:
            value = rates.delete[int(action.position)]
        else:
            value = rates.stop
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError(f"invalid action rate for {action.key}")
        result[action] = value
    return result


def total_hazard(action_rates: Mapping[AtomicAction, float]) -> float:
    if any(not math.isfinite(value) or value < 0.0 for value in action_rates.values()):
        raise FloatingPointError("individual action hazard is negative, NaN or Inf")
    total = math.fsum(action_rates.values())
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("invalid total hazard")
    return total


def conditioned_event_distribution(
    action_rates: Mapping[AtomicAction, float],
) -> dict[AtomicAction, float]:
    total = total_hazard(action_rates)
    if total <= 0.0:
        raise ZeroDivisionError("event distribution is undefined at zero total hazard")
    probabilities = {action: rate / total for action, rate in action_rates.items()}
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1.0e-12):
        raise FloatingPointError("conditioned event distribution is not normalized")
    return probabilities


def aggregate_transition_rates(
    state: EditState,
    action_rates: Mapping[AtomicAction, Scalar],
    *,
    min_length: int,
    max_length: int,
    key_fn: Optional[Callable[[EditState], str]] = None,
) -> dict[str, Scalar]:
    """Sum rates for all actions reaching the same next-state key."""

    key_fn = key_fn or (lambda next_state: next_state.state_hash)
    aggregated: dict[str, Scalar] = {}
    for action, rate in action_rates.items():
        next_state = apply_action(
            state, action, min_length=min_length, max_length=max_length
        ).after
        key = key_fn(next_state)
        aggregated[key] = aggregated[key] + rate if key in aggregated else rate
    return aggregated


@dataclass(frozen=True)
class GeneratorRow:
    state_hash: str
    off_diagonal: Mapping[str, float]
    diagonal: float

    @property
    def row_sum(self) -> float:
        return self.diagonal + math.fsum(self.off_diagonal.values())


def generator(
    state: EditState,
    action_rates: Mapping[AtomicAction, float],
    *,
    min_length: int,
    max_length: int,
) -> GeneratorRow:
    off_diagonal = aggregate_transition_rates(
        state,
        action_rates,
        min_length=min_length,
        max_length=max_length,
    )
    return GeneratorRow(
        state_hash=state.state_hash,
        off_diagonal=off_diagonal,
        diagonal=-total_hazard(action_rates),
    )
