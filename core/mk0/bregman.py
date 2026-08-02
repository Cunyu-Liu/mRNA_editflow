"""Transition-aggregated Edit Flow/Bregman objective and toy oracle."""

from __future__ import annotations

import math
from typing import Mapping, TypeVar

from .rate_kernel import aggregate_transition_rates
from .state_action import apply_action, enumerate_legal_actions
from .target_kernel import TargetTransitionOracle
from .types import ActionType, AtomicAction, EditState, Phase

Scalar = TypeVar("Scalar")


def aggregate_target_weights(
    weighted_transition_keys: list[tuple[str, float]],
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for key, weight in weighted_transition_keys:
        if not math.isfinite(weight) or weight < 0.0:
            raise FloatingPointError("target weight must be finite and non-negative")
        aggregated[key] = aggregated.get(key, 0.0) + weight
    return aggregated


def _scalar_log(value: Scalar) -> Scalar:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return torch.log(value)
    except ImportError:  # pragma: no cover - torch is a project dependency
        pass
    return math.log(float(value))  # type: ignore[return-value]


def bregman_loss(
    model_transition_rates: Mapping[str, Scalar],
    target_transition_weights: Mapping[str, float],
) -> Scalar:
    """Compute sum U - sum W log U after transition aggregation.

    STOP is not passed to this edit-only objective.  A target transition absent
    from the legal model neighbourhood fails closed.
    """

    if not model_transition_rates:
        raise ValueError("edit neighbourhood cannot be empty for EF loss")
    try:
        import torch

        for value in model_transition_rates.values():
            if isinstance(value, torch.Tensor):
                if (
                    value.numel() != 1
                    or not bool(torch.isfinite(value))
                    or bool(value < 0)
                ):
                    raise FloatingPointError(
                        "model transition rate is negative, NaN or Inf"
                    )
            elif not math.isfinite(float(value)) or float(value) < 0.0:
                raise FloatingPointError(
                    "model transition rate is negative, NaN or Inf"
                )
    except ImportError:
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in model_transition_rates.values()
        ):
            raise FloatingPointError("model transition rate is negative, NaN or Inf")
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in target_transition_weights.values()
    ):
        raise FloatingPointError("target transition weight is negative, NaN or Inf")
    iterator = iter(model_transition_rates.values())
    loss = next(iterator)
    for value in iterator:
        loss = loss + value
    for key, weight in target_transition_weights.items():
        if weight <= 0.0:
            continue
        if key not in model_transition_rates:
            raise ValueError(f"illegal or missing target transition: {key}")
        rate = model_transition_rates[key]
        try:
            nonpositive = bool(rate <= 0)  # float or scalar tensor
        except TypeError:  # pragma: no cover
            nonpositive = False
        if nonpositive:
            raise FloatingPointError(
                "positive target weight requires positive model rate"
            )
        loss = loss - weight * _scalar_log(rate)
    return loss


def brute_force_bregman_loss(
    action_transition_rate_pairs: list[tuple[str, float]],
    target_transition_weight_pairs: list[tuple[str, float]],
) -> float:
    """Independent explicit-loop oracle used only on exhaustive toy states."""

    keys = sorted(
        {key for key, _ in action_transition_rate_pairs}
        | {key for key, _ in target_transition_weight_pairs}
    )
    total = 0.0
    for key in keys:
        model = math.fsum(
            rate for candidate, rate in action_transition_rate_pairs if candidate == key
        )
        target = math.fsum(
            weight
            for candidate, weight in target_transition_weight_pairs
            if candidate == key
        )
        total += model
        if target > 0.0:
            if model <= 0.0:
                raise ValueError(f"target transition {key} is not model-legal")
            total -= target * math.log(model)
    return total


def edit_flow_loss(
    state: EditState,
    model_action_rates: Mapping[AtomicAction, Scalar],
    target_oracle: TargetTransitionOracle | None,
    *,
    min_length: int,
    max_length: int,
) -> Scalar | float:
    """Production state-aware edit-only Bregman objective.

    The caller must provide one rate for every legal INS/SUB/DEL action and no
    STOP rate.  Model rates and target weights are independently aggregated by
    the full next extended-state SHA-256 before evaluating :func:`bregman_loss`.
    HALTED states are absorbing and contribute the exact scalar zero, but only
    when no edit rates or target auxiliary are supplied.
    """

    if state.phase == Phase.HALTED:
        if model_action_rates:
            raise ValueError("HALTED state cannot expose model edit rates")
        if target_oracle is not None:
            raise ValueError("HALTED state cannot receive a target edit oracle")
        return 0.0

    if target_oracle is None:
        raise ValueError("ACTIVE state requires a state-bound target oracle")
    if target_oracle.source_state_hash != state.state_hash:
        raise ValueError("target oracle is bound to a different extended state")
    if any(entry.status != "ACCEPTED" for entry in target_oracle.ledger):
        raise ValueError("target oracle contains a rejected coupling path")

    legal_edits = set(
        enumerate_legal_actions(
            state,
            min_length=min_length,
            max_length=max_length,
            include_stop=False,
        )
    )
    supplied_actions = set(model_action_rates)
    if supplied_actions != legal_edits:
        missing = sorted(action.key for action in legal_edits - supplied_actions)
        unexpected = sorted(action.key for action in supplied_actions - legal_edits)
        raise ValueError(
            "model edit-rate domain differs from the complete legal neighbourhood; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(action.kind == ActionType.STOP for action in model_action_rates):
        raise ValueError("STOP rate cannot enter the edit-flow loss")

    recomputed_target: dict[str, list[float]] = {}
    for transition in target_oracle.transitions:
        if transition.action.kind == ActionType.STOP:
            raise ValueError("target oracle contains STOP in the edit objective")
        next_state = apply_action(
            state,
            transition.action,
            min_length=min_length,
            max_length=max_length,
        ).after
        if next_state.state_hash != transition.next_state_hash:
            raise ValueError("target transition next-state hash failed replay")
        recomputed_target.setdefault(transition.next_state_hash, []).append(
            transition.weight
        )
    expected_target = {
        key: math.fsum(values) for key, values in recomputed_target.items()
    }
    if expected_target != target_oracle.target_transition_weights:
        raise ValueError("target weights are not full-state transition aggregates")

    if not legal_edits:
        if expected_target:
            raise ValueError("target edit exists outside an empty legal neighbourhood")
        return 0.0

    model_transition_rates = aggregate_transition_rates(
        state,
        model_action_rates,
        min_length=min_length,
        max_length=max_length,
    )
    return bregman_loss(model_transition_rates, expected_target)
