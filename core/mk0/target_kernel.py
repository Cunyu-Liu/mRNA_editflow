"""Training-only target-transition kernel for the MK0 Edit Flow objective.

This module is the only bridge from the latent alignment/switch-clock
construction to observable edit actions.  It deliberately returns transition
weights keyed by the *full next extended-state hash*, never by the observable
RNA sequence or by an action label.

The bridge is fail closed.  A target switch which conflicts with the runtime
state, edit budget, length bounds, or another hard constraint is recorded in a
rejected/repair ledger and raises :class:`TargetKernelRejected`; it is never
silently removed from the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Optional

from .alignment_coupling import (
    BLANK,
    CouplingAlignment,
    build_alignment,
    changed_indices,
    reconstruct_alignment,
    switched_alignment_state,
)
from .schedule import evaluate_schedule
from .state_action import apply_action, is_legal
from .types import ActionType, AtomicAction, EditState, Phase, TokenOrigin


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class TargetPathLedgerEntry:
    """Auditable disposition of one latent target switch (or global check)."""

    alignment_index: Optional[int]
    clock: Optional[float]
    source_token: Optional[str]
    current_token: Optional[str]
    target_token: Optional[str]
    action_key: Optional[str]
    next_state_hash: Optional[str]
    target_weight: float
    status: str
    reason: str
    repair_status: str
    repair_action_key: Optional[str] = None

    def to_record(self) -> dict[str, Any]:
        return {
            "alignment_index": self.alignment_index,
            "clock": self.clock,
            "source_token": self.source_token,
            "current_token": self.current_token,
            "target_token": self.target_token,
            "action_key": self.action_key,
            "next_state_hash": self.next_state_hash,
            "target_weight": self.target_weight,
            "status": self.status,
            "reason": self.reason,
            "repair_status": self.repair_status,
            "repair_action_key": self.repair_action_key,
        }


class TargetKernelRejected(ValueError):
    """A coupling/runtime conflict carrying its complete audit ledger."""

    def __init__(self, message: str, ledger: tuple[TargetPathLedgerEntry, ...]) -> None:
        super().__init__(message)
        self.ledger = ledger

    def to_record(self) -> dict[str, Any]:
        return {
            "status": "REJECTED_FAIL_CLOSED",
            "message": str(self),
            "ledger": [entry.to_record() for entry in self.ledger],
            "repair_applied_count": sum(
                entry.repair_status == "APPLIED" for entry in self.ledger
            ),
        }


@dataclass(frozen=True)
class TargetTransition:
    """One remaining latent switch mapped to one legal runtime transition."""

    alignment_index: int
    action: AtomicAction
    next_state_hash: str
    observable_next: str
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.alignment_index, int) or self.alignment_index < 0:
            raise ValueError("alignment_index must be a non-negative integer")
        if self.action.kind == ActionType.STOP:
            raise ValueError("STOP cannot be a target edit-flow transition")
        if not _SHA256_RE.fullmatch(self.next_state_hash):
            raise ValueError("next_state_hash must be a lowercase SHA-256")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise FloatingPointError("target transition weight is invalid")


@dataclass(frozen=True)
class TargetTransitionOracle:
    """Immutable target kernel result bound to one current extended state."""

    source_state_hash: str
    alignment_hash: str
    external_time: float
    schedule: str
    schedule_rho: float
    remaining_alignment_indices: tuple[int, ...]
    transitions: tuple[TargetTransition, ...]
    aggregated_weights: tuple[tuple[str, float], ...]
    ledger: tuple[TargetPathLedgerEntry, ...]

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.source_state_hash):
            raise ValueError("source_state_hash must be a lowercase SHA-256")
        if not _SHA256_RE.fullmatch(self.alignment_hash):
            raise ValueError("alignment_hash must be a lowercase SHA-256")
        if not math.isfinite(self.schedule_rho) or self.schedule_rho < 0.0:
            raise FloatingPointError("schedule rho is invalid")
        if (
            not math.isfinite(self.external_time)
            or not 0.0 <= self.external_time <= 1.0
        ):
            raise ValueError("external time must lie in [0, 1]")
        if any(entry.status != "ACCEPTED" for entry in self.ledger):
            raise ValueError("a returned target oracle cannot contain rejected paths")
        if tuple(t.alignment_index for t in self.transitions) != (
            self.remaining_alignment_indices
        ):
            raise ValueError("target transitions do not cover every remaining switch")
        ledger_projection = tuple(
            (
                entry.alignment_index,
                entry.action_key,
                entry.next_state_hash,
                entry.target_weight,
            )
            for entry in self.ledger
        )
        transition_projection = tuple(
            (
                transition.alignment_index,
                transition.action.key,
                transition.next_state_hash,
                transition.weight,
            )
            for transition in self.transitions
        )
        if ledger_projection != transition_projection:
            raise ValueError("accepted target ledger does not match transitions")
        recomputed: dict[str, list[float]] = {}
        for transition in self.transitions:
            recomputed.setdefault(transition.next_state_hash, []).append(
                transition.weight
            )
        expected = tuple(
            sorted((key, math.fsum(values)) for key, values in recomputed.items())
        )
        if expected != self.aggregated_weights:
            raise ValueError("aggregated target weights are not transition-derived")

    @property
    def target_transition_weights(self) -> dict[str, float]:
        """Return a fresh mapping keyed only by full next-state hashes."""

        return dict(self.aggregated_weights)

    def to_record(self) -> dict[str, Any]:
        """JSON-ready audit record with the full accepted-path ledger."""

        return {
            "source_state_hash": self.source_state_hash,
            "alignment_hash": self.alignment_hash,
            "external_time": self.external_time,
            "schedule": self.schedule,
            "rho": self.schedule_rho,
            "remaining_alignment_indices": list(self.remaining_alignment_indices),
            "transitions": [
                {
                    "alignment_index": transition.alignment_index,
                    "action": transition.action.to_dict(),
                    "action_key": transition.action.key,
                    "next_state_hash": transition.next_state_hash,
                    "observable_next": transition.observable_next,
                    "weight": transition.weight,
                }
                for transition in self.transitions
            ],
            "aggregated_weights": dict(self.aggregated_weights),
            "ledger": [entry.to_record() for entry in self.ledger],
            "rejected_path_count": 0,
            "repair_applied_count": 0,
            "aggregation_key": "full_next_extended_state_sha256",
        }


def _global_rejection(reason: str) -> TargetKernelRejected:
    entry = TargetPathLedgerEntry(
        alignment_index=None,
        clock=None,
        source_token=None,
        current_token=None,
        target_token=None,
        action_key=None,
        next_state_hash=None,
        target_weight=0.0,
        status="REJECTED_FAIL_CLOSED",
        reason=reason,
        repair_status="NOT_ATTEMPTED_FAIL_CLOSED",
    )
    return TargetKernelRejected(reason, (entry,))


def _validate_alignment(alignment: CouplingAlignment) -> None:
    if alignment.path_is_observed or alignment.path_semantics != "latent_algorithmic":
        raise _global_rejection(
            "alignment must remain a latent algorithmic training auxiliary"
        )
    if reconstruct_alignment(alignment) != (alignment.source, alignment.target):
        raise _global_rejection("alignment columns do not reconstruct source/target")
    if any(token not in "ACGU" for token in alignment.source + alignment.target):
        raise _global_rejection("alignment source/target uses a non-RNA token")
    if alignment.coupling_type not in {
        "canonical_optimal",
        "sampled_optimal_sensitivity",
    }:
        raise _global_rejection("unfrozen coupling type is not valid for MK0-v1")
    canonical_cost = build_alignment(alignment.source, alignment.target).cost
    if (
        alignment.cost != len(changed_indices(alignment))
        or alignment.cost != canonical_cost
    ):
        raise _global_rejection("alignment does not have a verified optimal edit cost")
    source_cursor = 0
    target_cursor = 0
    for column in alignment.columns:
        if column.source_token == BLANK and column.target_token == BLANK:
            raise _global_rejection("alignment contains a blank-to-blank column")
        if column.source_token == BLANK:
            if column.source_index is not None:
                raise _global_rejection("blank source column carries a source index")
        else:
            if (
                source_cursor >= len(alignment.source)
                or column.source_index != source_cursor
                or column.source_token != alignment.source[source_cursor]
            ):
                raise _global_rejection(
                    "alignment source indices/tokens are inconsistent"
                )
            source_cursor += 1
        if column.target_token == BLANK:
            if column.target_index is not None:
                raise _global_rejection("blank target column carries a target index")
        else:
            if (
                target_cursor >= len(alignment.target)
                or column.target_index != target_cursor
                or column.target_token != alignment.target[target_cursor]
            ):
                raise _global_rejection(
                    "alignment target indices/tokens are inconsistent"
                )
            target_cursor += 1


def _validate_runtime_path_state(
    state: EditState,
    alignment: CouplingAlignment,
    augmented_current: tuple[str, ...],
    completed: tuple[int, ...],
) -> None:
    completed_set = set(completed)
    expected_ins = sum(
        alignment.columns[index].source_token == BLANK for index in completed
    )
    expected_del = sum(
        alignment.columns[index].target_token == BLANK for index in completed
    )
    expected_sub = len(completed) - expected_ins - expected_del
    if (
        state.history.ins,
        state.history.sub,
        state.history.delete,
    ) != (expected_ins, expected_sub, expected_del):
        raise _global_rejection(
            "runtime edit-kind history does not match completed auxiliary switches"
        )

    expected_mapping: list[tuple[TokenOrigin, Optional[int]]] = []
    for index, (column, token) in enumerate(zip(alignment.columns, augmented_current)):
        if token == BLANK:
            continue
        if column.source_token == BLANK:
            if index not in completed_set:
                raise _global_rejection(
                    "unswitched insertion unexpectedly appears in runtime current"
                )
            expected_mapping.append((TokenOrigin.INSERTED, None))
        else:
            expected_mapping.append((TokenOrigin.SOURCE, column.source_index))
    observed_mapping = [
        (token.origin, token.source_index) for token in state.mapping.tokens
    ]
    if observed_mapping != expected_mapping:
        raise _global_rejection(
            "runtime source/current mapping is inconsistent with auxiliary switches"
        )


def _normalise_clocks(
    alignment: CouplingAlignment, clocks: Mapping[int, float]
) -> dict[int, float]:
    changed = set(changed_indices(alignment))
    if set(clocks) != changed:
        raise _global_rejection(
            "switch-clock keys must equal the changed alignment coordinates"
        )
    normalised: dict[int, float] = {}
    for index, raw_clock in clocks.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise _global_rejection("switch-clock keys must be integer coordinates")
        if isinstance(raw_clock, bool):
            raise _global_rejection("switch clocks must be finite real values")
        try:
            value = float(raw_clock)
        except (TypeError, ValueError) as error:
            raise _global_rejection(
                "switch clocks must be finite real values"
            ) from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise _global_rejection("switch clocks must lie in [0, 1]")
        normalised[index] = value
    return normalised


def _action_for_remaining_switch(
    alignment: CouplingAlignment,
    augmented_current: tuple[str, ...],
    alignment_index: int,
) -> AtomicAction:
    column = alignment.columns[alignment_index]
    current_token = augmented_current[alignment_index]
    if current_token != column.source_token:
        raise _global_rejection(
            "a remaining switch is not in its source-side augmented state"
        )
    position = sum(token != BLANK for token in augmented_current[:alignment_index])
    if column.source_token == BLANK and column.target_token != BLANK:
        return AtomicAction(ActionType.INS, position, column.target_token)
    if column.target_token == BLANK and column.source_token != BLANK:
        return AtomicAction(ActionType.DEL, position)
    if (
        column.source_token != BLANK
        and column.target_token != BLANK
        and column.source_token != column.target_token
    ):
        return AtomicAction(ActionType.SUB, position, column.target_token)
    raise _global_rejection("remaining coordinate does not define an atomic edit")


def build_target_transition_oracle(
    state: EditState,
    alignment: CouplingAlignment,
    clocks: Mapping[int, float],
    t: float,
    *,
    min_length: int,
    max_length: int,
    schedule: str = "cubic",
    time_eps: float = 1.0e-4,
) -> TargetTransitionOracle:
    """Map remaining auxiliary switches to legal next extended states.

    ``state`` is inference-visible and contains no target data.  ``alignment``
    and ``clocks`` are used only inside this training target constructor.  The
    returned weights are sums of ``rho(t)`` over latent switches that reach the
    same full next extended state.
    """

    if (
        isinstance(min_length, bool)
        or isinstance(max_length, bool)
        or not isinstance(min_length, int)
        or not isinstance(max_length, int)
    ):
        raise _global_rejection("length bounds must be integers")
    if min_length < 0 or max_length < min_length:
        raise _global_rejection("invalid length bounds")
    if state.phase != Phase.ACTIVE:
        raise _global_rejection("target transition kernel requires an ACTIVE state")
    _validate_alignment(alignment)
    if state.source != alignment.source:
        raise _global_rejection("runtime source does not match coupling source")
    if not min_length <= len(state.current) <= max_length:
        raise _global_rejection("runtime current length is outside frozen bounds")
    if not min_length <= len(alignment.target) <= max_length:
        raise _global_rejection("coupling target length is outside frozen bounds")
    normalised_clocks = _normalise_clocks(alignment, clocks)
    if isinstance(t, bool):
        raise _global_rejection("external time must be a finite real value")
    try:
        external_time = float(t)
    except (TypeError, ValueError) as error:
        raise _global_rejection("external time must be a finite real value") from error
    try:
        schedule_value = evaluate_schedule(
            external_time, name=schedule, time_eps=time_eps
        )
    except (TypeError, ValueError, FloatingPointError) as error:
        raise _global_rejection(f"invalid schedule evaluation: {error}") from error

    changed = changed_indices(alignment)
    if len(changed) > state.initial_budget:
        raise _global_rejection(
            "coupling edit cost exceeds the runtime state's initial budget"
        )
    completed = tuple(
        index for index in changed if normalised_clocks[index] <= external_time
    )
    remaining = tuple(
        index for index in changed if normalised_clocks[index] > external_time
    )
    if state.history.executed != len(completed):
        raise _global_rejection(
            "runtime history count does not match completed auxiliary switches"
        )

    augmented_current = switched_alignment_state(
        alignment, normalised_clocks, external_time
    )
    observable_current = "".join(token for token in augmented_current if token != BLANK)
    if observable_current != state.current:
        raise _global_rejection(
            "runtime current sequence does not match the auxiliary switched state"
        )
    _validate_runtime_path_state(state, alignment, augmented_current, completed)

    transitions: list[TargetTransition] = []
    ledger: list[TargetPathLedgerEntry] = []
    for index in remaining:
        column = alignment.columns[index]
        try:
            action = _action_for_remaining_switch(alignment, augmented_current, index)
        except TargetKernelRejected as error:
            ledger.append(
                TargetPathLedgerEntry(
                    alignment_index=index,
                    clock=normalised_clocks[index],
                    source_token=column.source_token,
                    current_token=augmented_current[index],
                    target_token=column.target_token,
                    action_key=None,
                    next_state_hash=None,
                    target_weight=schedule_value.rho,
                    status="REJECTED_FAIL_CLOSED",
                    reason=str(error),
                    repair_status="NOT_ATTEMPTED_FAIL_CLOSED",
                )
            )
            continue
        if not is_legal(state, action, min_length=min_length, max_length=max_length):
            ledger.append(
                TargetPathLedgerEntry(
                    alignment_index=index,
                    clock=normalised_clocks[index],
                    source_token=column.source_token,
                    current_token=augmented_current[index],
                    target_token=column.target_token,
                    action_key=action.key,
                    next_state_hash=None,
                    target_weight=schedule_value.rho,
                    status="REJECTED_FAIL_CLOSED",
                    reason="target action is forbidden by the runtime hard mask",
                    repair_status="NOT_ATTEMPTED_FAIL_CLOSED",
                )
            )
            continue
        next_state = apply_action(
            state, action, min_length=min_length, max_length=max_length
        ).after
        transition = TargetTransition(
            alignment_index=index,
            action=action,
            next_state_hash=next_state.state_hash,
            observable_next=next_state.current,
            weight=schedule_value.rho,
        )
        transitions.append(transition)
        ledger.append(
            TargetPathLedgerEntry(
                alignment_index=index,
                clock=normalised_clocks[index],
                source_token=column.source_token,
                current_token=augmented_current[index],
                target_token=column.target_token,
                action_key=action.key,
                next_state_hash=next_state.state_hash,
                target_weight=schedule_value.rho,
                status="ACCEPTED",
                reason="LEGAL_TARGET_SWITCH",
                repair_status="NOT_NEEDED",
            )
        )

    rejected = tuple(entry for entry in ledger if entry.status != "ACCEPTED")
    if rejected:
        coordinates = ",".join(str(entry.alignment_index) for entry in rejected)
        raise TargetKernelRejected(
            f"target switches rejected at alignment coordinates: {coordinates}",
            tuple(ledger),
        )

    grouped: dict[str, list[float]] = {}
    for transition in transitions:
        grouped.setdefault(transition.next_state_hash, []).append(transition.weight)
    aggregated = tuple(
        sorted((key, math.fsum(values)) for key, values in grouped.items())
    )
    return TargetTransitionOracle(
        source_state_hash=state.state_hash,
        alignment_hash=alignment.alignment_hash,
        external_time=external_time,
        schedule=schedule,
        schedule_rho=schedule_value.rho,
        remaining_alignment_indices=remaining,
        transitions=tuple(transitions),
        aggregated_weights=aggregated,
        ledger=tuple(ledger),
    )


__all__ = [
    "TargetKernelRejected",
    "TargetPathLedgerEntry",
    "TargetTransition",
    "TargetTransitionOracle",
    "build_target_transition_oracle",
]
