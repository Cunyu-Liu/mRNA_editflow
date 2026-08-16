"""Route 2 source-anchored legal CTMC kernel.

The kernel intentionally implements only the frozen V1 action space: one
source-relative nucleotide substitution at a previously untouched position,
or STOP.  It contains no data access, model architecture, optimizer, guided
potential, or Evaluation logic.  A learned model supplies positive rates only
after the complete legal action set has been enumerated.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence


ALPHABET = ("A", "C", "G", "U")
SUB = "SUB"
STOP = "STOP"
TERMINAL_CAUSES = (
    "EXPLICIT_STOP",
    "BUDGET_EXHAUSTED",
    "NO_LEGAL_ACTION",
    "NUMERICAL_FAILURE",
)


class LegalFlowError(RuntimeError):
    """A state, action, or rate violates the Route 2 legal-flow contract."""


@dataclass(frozen=True)
class FlowState:
    source_sequence: str
    current_sequence: str
    source_relative_edits: tuple[tuple[int, str], ...]
    remaining_budget: int
    assay_id: str
    context_id: str
    terminal_cause: str | None = None

    @property
    def edit_count(self) -> int:
        return len(self.source_relative_edits)


@dataclass(frozen=True, order=True)
class LegalAction:
    kind: str
    position: int | None = None
    alt_base: str | None = None

    @property
    def action_id(self) -> str:
        if self.kind == STOP:
            return STOP
        return f"SUB:{self.position}:{self.alt_base}"


RateFunction = Callable[[FlowState, Sequence[LegalAction]], Mapping[LegalAction, float]]


def replay_source_relative(
    source_sequence: str, edits: Sequence[tuple[int, str]]
) -> str:
    """Replay a sorted source-relative SUB set without sequential drift."""

    current = list(source_sequence)
    seen: set[int] = set()
    for position, alt_base in edits:
        if position in seen:
            raise LegalFlowError("a source position cannot be edited twice")
        if position < 0 or position >= len(source_sequence):
            raise LegalFlowError("source-relative edit position is out of range")
        if alt_base not in ALPHABET or alt_base == source_sequence[position]:
            raise LegalFlowError("SUB must change the source base to a distinct RNA base")
        seen.add(position)
        current[position] = alt_base
    return "".join(current)


def validate_state(state: FlowState) -> None:
    if not state.source_sequence:
        raise LegalFlowError("source sequence must be non-empty")
    if any(base not in ALPHABET for base in state.source_sequence):
        raise LegalFlowError("source sequence is outside the RNA alphabet")
    if state.source_relative_edits != tuple(sorted(state.source_relative_edits)):
        raise LegalFlowError("source-relative edits must be sorted")
    replayed = replay_source_relative(state.source_sequence, state.source_relative_edits)
    if replayed != state.current_sequence:
        raise LegalFlowError("current sequence does not replay exactly from the source")
    if isinstance(state.remaining_budget, bool) or state.remaining_budget < 0:
        raise LegalFlowError("remaining budget must be a nonnegative integer")
    if not isinstance(state.remaining_budget, int):
        raise LegalFlowError("remaining budget must be a nonnegative integer")
    if state.terminal_cause is not None and state.terminal_cause not in TERMINAL_CAUSES:
        raise LegalFlowError("unknown terminal cause")


def _unedited_positions(state: FlowState) -> tuple[int, ...]:
    edited = {position for position, _ in state.source_relative_edits}
    return tuple(position for position in range(len(state.source_sequence)) if position not in edited)


def _structural_terminal_cause(state: FlowState) -> str | None:
    if state.remaining_budget == 0:
        return "BUDGET_EXHAUSTED"
    if not _unedited_positions(state):
        return "NO_LEGAL_ACTION"
    return None


def initial_state(
    source_sequence: str,
    *,
    budget: int,
    assay_id: str,
    context_id: str,
) -> FlowState:
    state = FlowState(
        source_sequence=source_sequence,
        current_sequence=source_sequence,
        source_relative_edits=(),
        remaining_budget=budget,
        assay_id=assay_id,
        context_id=context_id,
    )
    validate_state(state)
    return replace(state, terminal_cause=_structural_terminal_cause(state))


def numerical_failure_state(state: FlowState) -> FlowState:
    """Record a distinct absorbing state after a detected numerical failure."""

    validate_state(state)
    if state.terminal_cause is not None:
        raise LegalFlowError("cannot replace an existing terminal cause")
    return replace(state, terminal_cause="NUMERICAL_FAILURE")


def legal_actions(state: FlowState) -> tuple[LegalAction, ...]:
    """Enumerate the complete hard-legal action set before model scoring."""

    validate_state(state)
    if state.terminal_cause is not None:
        return ()
    actions = [LegalAction(STOP)]
    for position in _unedited_positions(state):
        source_base = state.source_sequence[position]
        actions.extend(
            LegalAction(SUB, position, alt_base)
            for alt_base in ALPHABET
            if alt_base != source_base
        )
    return tuple(actions)


def apply_action(state: FlowState, action: LegalAction) -> FlowState:
    actions = legal_actions(state)
    if action not in actions:
        raise LegalFlowError(f"illegal action: {action.action_id}")
    if action.kind == STOP:
        return replace(state, terminal_cause="EXPLICIT_STOP")
    assert action.position is not None and action.alt_base is not None
    edits = tuple(sorted((*state.source_relative_edits, (action.position, action.alt_base))))
    child = FlowState(
        source_sequence=state.source_sequence,
        current_sequence=replay_source_relative(state.source_sequence, edits),
        source_relative_edits=edits,
        remaining_budget=state.remaining_budget - 1,
        assay_id=state.assay_id,
        context_id=state.context_id,
    )
    child = replace(child, terminal_cause=_structural_terminal_cause(child))
    validate_state(child)
    return child


def positive_rates(
    state: FlowState,
    rate_function: RateFunction,
    *,
    support_floor: float = 1e-8,
) -> tuple[tuple[LegalAction, float], ...]:
    """Score exactly the legal set and add a strictly positive support floor."""

    if not math.isfinite(support_floor) or support_floor <= 0.0:
        raise LegalFlowError("support floor must be finite and strictly positive")
    actions = legal_actions(state)
    if not actions:
        return ()
    supplied = rate_function(state, actions)
    if set(supplied) != set(actions):
        raise LegalFlowError("rate function must return exactly the enumerated legal actions")
    result: list[tuple[LegalAction, float]] = []
    for action in actions:
        raw_rate = float(supplied[action])
        if not math.isfinite(raw_rate) or raw_rate < 0.0:
            raise LegalFlowError(f"invalid learned base rate for {action.action_id}")
        result.append((action, raw_rate + support_floor))
    return tuple(result)


def jump_distribution(
    state: FlowState,
    rate_function: RateFunction,
    *,
    support_floor: float = 1e-8,
) -> tuple[tuple[LegalAction, FlowState, float], ...]:
    rates = positive_rates(state, rate_function, support_floor=support_floor)
    total = math.fsum(rate for _, rate in rates)
    if rates and (not math.isfinite(total) or total <= 0.0):
        raise LegalFlowError("total exit rate is invalid")
    return tuple(
        (action, apply_action(state, action), rate / total)
        for action, rate in rates
    )


def reachable_graph(
    root: FlowState,
    rate_function: RateFunction,
    *,
    support_floor: float = 1e-8,
) -> dict[FlowState, tuple[tuple[LegalAction, FlowState, float], ...]]:
    """Materialize the finite absorbing DAG for exact small-case validation."""

    validate_state(root)
    queue = deque([root])
    seen: set[FlowState] = set()
    graph: dict[FlowState, tuple[tuple[LegalAction, FlowState, float], ...]] = {}
    while queue:
        state = queue.popleft()
        if state in seen:
            continue
        seen.add(state)
        if state.terminal_cause is not None:
            graph[state] = ()
            continue
        edges = jump_distribution(state, rate_function, support_floor=support_floor)
        graph[state] = edges
        queue.extend(child for _, child, _ in edges)
    return graph


def exact_terminal_distribution(
    root: FlowState,
    rate_function: RateFunction,
    *,
    support_floor: float = 1e-8,
) -> dict[FlowState, float]:
    """Propagate embedded-jump mass exactly over the source-anchored DAG."""

    graph = reachable_graph(root, rate_function, support_floor=support_floor)
    mass: dict[FlowState, float] = defaultdict(float)
    mass[root] = 1.0
    for state in sorted(graph, key=lambda item: (item.edit_count, item.terminal_cause is not None)):
        for _, child, probability in graph[state]:
            mass[child] += mass[state] * probability
    terminal = {
        state: mass[state]
        for state in graph
        if state.terminal_cause is not None and mass[state] > 0.0
    }
    if not math.isclose(math.fsum(terminal.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise LegalFlowError("terminal probability mass does not sum to one")
    return terminal


def sample_trajectory(
    root: FlowState,
    rate_function: RateFunction,
    *,
    seed: int,
    support_floor: float = 1e-8,
) -> tuple[FlowState, ...]:
    """Sample embedded CTMC jumps; holding times are outside this G0 kernel."""

    rng = random.Random(seed)
    trajectory = [root]
    state = root
    while state.terminal_cause is None:
        edges = jump_distribution(state, rate_function, support_floor=support_floor)
        draw = rng.random()
        cumulative = 0.0
        child = edges[-1][1]
        for _, candidate, probability in edges:
            cumulative += probability
            if draw <= cumulative:
                child = candidate
                break
        trajectory.append(child)
        state = child
    return tuple(trajectory)
