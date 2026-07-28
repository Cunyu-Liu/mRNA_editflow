"""Exact, auditable state closure for every shortest edit execution order.

D1 counts minimum-cost *character alignments*.  That count is retained here,
but an alignment-prefix DAG is not sufficient for B0: independent edits can
be executed in a different order and expose additional intermediate sequence
states.  For example, ``AA -> CC`` has both ``CA`` and ``AC`` as shortest-path
intermediates.

B0 therefore traverses the sequence-state graph.  From a state whose exact
distance to the candidate is ``r``, it keeps every primitive dynamic edit
whose result has distance ``r - 1``.  This is both necessary and sufficient
for membership in a shortest path.  Substitutions and one-base insertions or
deletions use positions in the *current* sequence, so indel coordinates are
dynamic.  Coordinate-distinct operations that produce the same sequence in a
repeat are deliberately collapsed because leakage is defined over sequence
state identity.

The closure can be exponential because the scientific object itself can be
exponential.  Explicit state, transition-candidate, and dynamic-programming
cell limits therefore fail closed with retained error evidence; no
single-traceback or sampled approximation is substituted.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


RNA_ALPHABET = frozenset("ACGU")
MINIMUM_ALIGNMENT_COUNT_SCOPE = "minimum_cost_character_alignments"
STATE_CLOSURE_SCOPE = (
    "all_shortest_primitive_dynamic_edit_execution_orders_sequence_identity"
)
STATE_PATH_COUNT_SCOPE = (
    "minimum_primitive_edit_state_paths_coordinate_equivalent_transitions_collapsed"
)
ALGORITHM_ID = "all_shortest_dynamic_edit_state_closure_v2"


class PathStateError(ValueError):
    """Raised when endpoint sequences cannot be audited safely."""


@dataclass(frozen=True)
class MinimumAlignmentStateClosure:
    """Complete, deterministic shortest-path state closure for one pair."""

    source_sequence: str
    candidate_sequence: str
    minimum_edit_count: int
    minimum_alignment_count: int
    minimum_state_path_count: int
    reachable_node_count: int
    reachable_transition_count: int
    evaluated_primitive_action_count: int
    evaluated_state_dp_cell_count: int
    reachable_states: Tuple[str, ...]
    constructed_intermediate_states: Tuple[str, ...]
    reachable_states_sha256: str
    count_scope: str = MINIMUM_ALIGNMENT_COUNT_SCOPE
    state_closure_scope: str = STATE_CLOSURE_SCOPE
    state_path_count_scope: str = STATE_PATH_COUNT_SCOPE
    algorithm: str = ALGORITHM_ID
    constructed_not_observed: bool = True


def _validate_rna(sequence: str, name: str) -> None:
    if not isinstance(sequence, str) or not sequence:
        raise PathStateError(f"{name} must be a non-empty RNA string")
    if sequence != sequence.upper() or any(
        base not in RNA_ALPHABET for base in sequence
    ):
        raise PathStateError(
            f"{name} must use the uppercase canonical RNA alphabet A/C/G/U"
        )


def _banded_tables(
    source: str,
    candidate: str,
    band: int,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    """Return forward cost/count and backward cost inside an exact band."""

    n, m = len(source), len(candidate)
    forward_cost: dict[tuple[int, int], int] = {}
    forward_count: dict[tuple[int, int], int] = {}
    for i in range(n + 1):
        for j in range(max(0, i - band), min(m, i + band) + 1):
            node = (i, j)
            if node == (0, 0):
                forward_cost[node] = 0
                forward_count[node] = 1
                continue
            choices: list[tuple[int, int]] = []
            diagonal = (i - 1, j - 1)
            if i > 0 and j > 0 and diagonal in forward_cost:
                step = 0 if source[i - 1] == candidate[j - 1] else 1
                choices.append(
                    (
                        forward_cost[diagonal] + step,
                        forward_count[diagonal],
                    )
                )
            deletion = (i - 1, j)
            if i > 0 and deletion in forward_cost:
                choices.append(
                    (
                        forward_cost[deletion] + 1,
                        forward_count[deletion],
                    )
                )
            insertion = (i, j - 1)
            if j > 0 and insertion in forward_cost:
                choices.append(
                    (
                        forward_cost[insertion] + 1,
                        forward_count[insertion],
                    )
                )
            if choices:
                best = min(cost for cost, _ in choices)
                forward_cost[node] = best
                forward_count[node] = sum(
                    count for cost, count in choices if cost == best
                )

    backward_cost: dict[tuple[int, int], int] = {}
    for i in range(n, -1, -1):
        for j in range(min(m, i + band), max(0, i - band) - 1, -1):
            node = (i, j)
            if node == (n, m):
                backward_cost[node] = 0
                continue
            choices: list[int] = []
            diagonal = (i + 1, j + 1)
            if i < n and j < m and diagonal in backward_cost:
                step = 0 if source[i] == candidate[j] else 1
                choices.append(backward_cost[diagonal] + step)
            deletion = (i + 1, j)
            if i < n and deletion in backward_cost:
                choices.append(backward_cost[deletion] + 1)
            insertion = (i, j + 1)
            if j < m and insertion in backward_cost:
                choices.append(backward_cost[insertion] + 1)
            if choices:
                backward_cost[node] = min(choices)
    return forward_cost, forward_count, backward_cost


def _band_cell_count(source_length: int, candidate_length: int, band: int) -> int:
    return sum(
        min(candidate_length, i + band) - max(0, i - band) + 1
        for i in range(source_length + 1)
    )


def _banded_cost_tables(
    source: str,
    candidate: str,
    band: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """Return exact prefix and suffix costs needed for local edit queries."""

    n, m = len(source), len(candidate)
    forward_cost: dict[tuple[int, int], int] = {}
    for i in range(n + 1):
        for j in range(max(0, i - band), min(m, i + band) + 1):
            node = (i, j)
            if node == (0, 0):
                forward_cost[node] = 0
                continue
            choices: list[int] = []
            diagonal = (i - 1, j - 1)
            if i > 0 and j > 0 and diagonal in forward_cost:
                choices.append(
                    forward_cost[diagonal]
                    + (source[i - 1] != candidate[j - 1])
                )
            deletion = (i - 1, j)
            if i > 0 and deletion in forward_cost:
                choices.append(forward_cost[deletion] + 1)
            insertion = (i, j - 1)
            if j > 0 and insertion in forward_cost:
                choices.append(forward_cost[insertion] + 1)
            if choices:
                forward_cost[node] = min(choices)

    backward_cost: dict[tuple[int, int], int] = {}
    for i in range(n, -1, -1):
        for j in range(min(m, i + band), max(0, i - band) - 1, -1):
            node = (i, j)
            if node == (n, m):
                backward_cost[node] = 0
                continue
            choices = []
            diagonal = (i + 1, j + 1)
            if i < n and j < m and diagonal in backward_cost:
                choices.append(
                    backward_cost[diagonal]
                    + (source[i] != candidate[j])
                )
            deletion = (i + 1, j)
            if i < n and deletion in backward_cost:
                choices.append(backward_cost[deletion] + 1)
            insertion = (i, j + 1)
            if j < m and insertion in backward_cost:
                choices.append(backward_cost[insertion] + 1)
            if choices:
                backward_cost[node] = min(choices)
    return forward_cost, backward_cost


def _distance_reducing_neighbors(
    sequence: str,
    candidate: str,
    remaining_distance: int,
) -> Tuple[str, ...]:
    """Return every one-edit neighbor exactly one step closer to candidate.

    Prefix/suffix Levenshtein tables make the distance after each possible
    local edit exact without recomputing one full alignment per neighbor.
    Missing band cells cannot hide a qualifying neighbor: any alignment of
    cost ``remaining_distance - 1`` stays inside this band.
    """

    if remaining_distance < 1:
        raise AssertionError("distance-reducing neighbors require distance > 0")
    forward, backward = _banded_cost_tables(
        sequence, candidate, remaining_distance
    )
    terminal = (len(sequence), len(candidate))
    if forward.get(terminal) != remaining_distance:
        raise AssertionError(
            "state does not have the declared remaining edit distance"
        )

    n, m = len(sequence), len(candidate)
    target_distance = remaining_distance - 1
    unreachable = remaining_distance + max(n, m) + 2
    neighbors: set[str] = set()

    for i, original in enumerate(sequence):
        j_start = max(0, i - remaining_distance)
        j_stop = min(m, i + remaining_distance)

        deletion_distance = unreachable
        for j in range(j_start, j_stop + 1):
            prefix = forward.get((i, j))
            suffix = backward.get((i + 1, j))
            if prefix is not None and suffix is not None:
                deletion_distance = min(
                    deletion_distance, prefix + suffix
                )
        if deletion_distance == target_distance:
            neighbors.add(sequence[:i] + sequence[i + 1 :])

        for alternative in RNA_ALPHABET:
            if alternative == original:
                continue
            substitution_distance = unreachable
            for j in range(j_start, j_stop + 1):
                prefix = forward.get((i, j))
                if prefix is None:
                    continue
                deleted_suffix = backward.get((i + 1, j))
                if deleted_suffix is not None:
                    substitution_distance = min(
                        substitution_distance,
                        prefix + 1 + deleted_suffix,
                    )
                if j < m:
                    diagonal_suffix = backward.get((i + 1, j + 1))
                    if diagonal_suffix is not None:
                        substitution_distance = min(
                            substitution_distance,
                            prefix
                            + (alternative != candidate[j])
                            + diagonal_suffix,
                        )
            if substitution_distance == target_distance:
                neighbors.add(
                    sequence[:i] + alternative + sequence[i + 1 :]
                )

    for i in range(n + 1):
        j_start = max(0, i - remaining_distance)
        j_stop = min(m, i + remaining_distance)
        for base in RNA_ALPHABET:
            insertion_distance = unreachable
            for j in range(j_start, j_stop + 1):
                prefix = forward.get((i, j))
                if prefix is None:
                    continue
                deleted_suffix = backward.get((i, j))
                if deleted_suffix is not None:
                    insertion_distance = min(
                        insertion_distance,
                        prefix + 1 + deleted_suffix,
                    )
                if j < m:
                    diagonal_suffix = backward.get((i, j + 1))
                    if diagonal_suffix is not None:
                        insertion_distance = min(
                            insertion_distance,
                            prefix
                            + (base != candidate[j])
                            + diagonal_suffix,
                        )
            if insertion_distance == target_distance:
                neighbors.add(sequence[:i] + base + sequence[i:])

    return tuple(sorted(neighbors))


def _one_primitive_edit_apart(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    longer, shorter = (
        (left, right) if len(left) > len(right) else (right, left)
    )
    shorter_index = 0
    skipped = False
    for longer_index, base in enumerate(longer):
        if (
            shorter_index < len(shorter)
            and base == shorter[shorter_index]
        ):
            shorter_index += 1
        elif skipped:
            return False
        else:
            skipped = True
        if longer_index - shorter_index > 1:
            return False
    return True


def _states_digest(states: Tuple[str, ...]) -> str:
    payload = ("\n".join(states) + ("\n" if states else "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=8_192)
def minimum_alignment_state_closure(
    source: str,
    candidate: str,
    *,
    known_minimum_edit_count: int | None = None,
    max_dag_cells: int = 1_000_000,
    max_reachable_states: int = 50_000,
    max_neighbor_expansions: int = 5_000_000,
    max_state_dp_cells: int = 50_000_000,
) -> MinimumAlignmentStateClosure:
    """Compute every sequence state on every shortest primitive edit path.

    All resource limits are fail-closed.  Crossing one raises
    :class:`PathStateError`; the function never returns a partial closure.
    """

    _validate_rna(source, "source")
    _validate_rna(candidate, "candidate")
    for limit_name, limit_value in (
        ("max_dag_cells", max_dag_cells),
        ("max_reachable_states", max_reachable_states),
        ("max_neighbor_expansions", max_neighbor_expansions),
        ("max_state_dp_cells", max_state_dp_cells),
    ):
        if (
            isinstance(limit_value, bool)
            or not isinstance(limit_value, int)
            or limit_value < 1
        ):
            raise PathStateError(f"{limit_name} must be a positive integer")
    if (
        known_minimum_edit_count is not None
        and (
            isinstance(known_minimum_edit_count, bool)
            or not isinstance(known_minimum_edit_count, int)
            or known_minimum_edit_count < 0
        )
    ):
        raise PathStateError(
            "known_minimum_edit_count must be a non-negative integer"
        )
    band = (
        max(len(source), len(candidate))
        if known_minimum_edit_count is None
        else known_minimum_edit_count
    )
    if abs(len(source) - len(candidate)) > band:
        raise PathStateError(
            "known minimum edit count is smaller than the length difference"
        )
    cells = _band_cell_count(len(source), len(candidate), band)
    if cells > max_dag_cells:
        raise PathStateError(
            f"minimum-alignment DAG has {cells} cells, exceeding "
            f"the audited limit {max_dag_cells}"
        )

    forward_cost, forward_count, backward_cost = _banded_tables(
        source, candidate, band
    )
    terminal = (len(source), len(candidate))
    if terminal not in forward_cost:
        raise PathStateError(
            "minimum-alignment band cannot reach the endpoint"
        )
    minimum_edit_count = forward_cost[terminal]
    if (
        known_minimum_edit_count is not None
        and minimum_edit_count != known_minimum_edit_count
    ):
        raise PathStateError(
            "known minimum edit count disagrees with recomputed alignment"
        )

    del backward_cost
    current_layer = (source,)
    all_states = {source}
    state_path_counts = {source: 1}
    reachable_transition_count = 0
    evaluated_primitive_action_count = 0
    evaluated_state_dp_cell_count = 0

    for depth in range(1, minimum_edit_count):
        remaining_before_edit = minimum_edit_count - depth + 1
        next_layer_counts: dict[str, int] = {}
        for state in current_layer:
            primitive_action_count = 8 * len(state) + 4
            evaluated_primitive_action_count += primitive_action_count
            if evaluated_primitive_action_count > max_neighbor_expansions:
                raise PathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                    f"closure exceeded {max_neighbor_expansions} evaluated "
                    "primitive actions; no approximation was emitted"
                )
            state_cells = 2 * _band_cell_count(
                len(state), len(candidate), remaining_before_edit
            )
            evaluated_state_dp_cell_count += state_cells
            if evaluated_state_dp_cell_count > max_state_dp_cells:
                raise PathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                    f"closure exceeded {max_state_dp_cells} state DP cells; "
                    "no approximation was emitted"
                )
            neighbors = _distance_reducing_neighbors(
                state,
                candidate,
                remaining_before_edit,
            )
            for neighbor in neighbors:
                reachable_transition_count += 1
                if (
                    neighbor not in next_layer_counts
                    and len(all_states) + len(next_layer_counts) + 1
                    > max_reachable_states
                ):
                    raise PathStateError(
                        "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact "
                        f"shortest-action closure exceeded "
                        f"{max_reachable_states} reachable states; "
                        "no approximation was emitted"
                    )
                next_layer_counts[neighbor] = (
                    next_layer_counts.get(neighbor, 0)
                    + state_path_counts[state]
                )

        if not next_layer_counts:
            raise AssertionError(
                "shortest-action closure lost every path before the endpoint"
            )
        repeated_states = all_states & set(next_layer_counts)
        if repeated_states:
            raise AssertionError(
                "a shortest edit path revisited a sequence state"
            )
        if len(all_states) + len(next_layer_counts) > max_reachable_states:
            raise PathStateError(
                "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                f"closure exceeded {max_reachable_states} reachable states; "
                "no approximation was emitted"
            )
        all_states.update(next_layer_counts)
        state_path_counts.update(next_layer_counts)
        current_layer = tuple(sorted(next_layer_counts))

    if minimum_edit_count == 0:
        minimum_state_path_count = 1
    else:
        if candidate in all_states:
            raise AssertionError(
                "candidate appeared before the final shortest-path layer"
            )
        if len(all_states) + 1 > max_reachable_states:
            raise PathStateError(
                "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                f"closure exceeded {max_reachable_states} reachable states; "
                "no approximation was emitted"
            )
        for state in current_layer:
            if not _one_primitive_edit_apart(state, candidate):
                raise AssertionError(
                    "penultimate state is not one edit from the candidate"
                )
        reachable_transition_count += len(current_layer)
        minimum_state_path_count = sum(
            state_path_counts[state] for state in current_layer
        )
        all_states.add(candidate)

    ordered_states = tuple(sorted(all_states))
    intermediates = tuple(
        state
        for state in ordered_states
        if state not in {source, candidate}
    )
    return MinimumAlignmentStateClosure(
        source_sequence=source,
        candidate_sequence=candidate,
        minimum_edit_count=minimum_edit_count,
        minimum_alignment_count=forward_count[terminal],
        minimum_state_path_count=minimum_state_path_count,
        reachable_node_count=len(ordered_states),
        reachable_transition_count=reachable_transition_count,
        evaluated_primitive_action_count=evaluated_primitive_action_count,
        evaluated_state_dp_cell_count=evaluated_state_dp_cell_count,
        reachable_states=ordered_states,
        constructed_intermediate_states=intermediates,
        reachable_states_sha256=_states_digest(ordered_states),
    )


__all__ = [
    "ALGORITHM_ID",
    "MINIMUM_ALIGNMENT_COUNT_SCOPE",
    "STATE_CLOSURE_SCOPE",
    "STATE_PATH_COUNT_SCOPE",
    "MinimumAlignmentStateClosure",
    "PathStateError",
    "minimum_alignment_state_closure",
]
