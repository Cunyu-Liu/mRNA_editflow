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
ALGORITHM_ID = "all_shortest_dynamic_edit_state_closure_v3"
MINIMUM_ALIGNMENT_ALGORITHM_ID = "banded_levenshtein_alignment_count_v1"
PRIMITIVE_ACTION_EVALUATION_SCOPE = (
    "distinct_dynamic_geodesic_actions_before_sequence_identity_collapse"
)
# This operational guard remains intentionally fail-closed. The frozen D1
# universe contains a 95,217-state witness, so the default must stop rather
# than silently truncate, sample, or approximate that closure.
DEFAULT_MAX_REACHABLE_STATES = 50_000


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
    primitive_action_evaluation_scope: str = PRIMITIVE_ACTION_EVALUATION_SCOPE
    constructed_not_observed: bool = True


@dataclass(frozen=True)
class MinimumAlignmentStatistics:
    """Exact minimum character-alignment distance and multiplicity."""

    source_sequence: str
    candidate_sequence: str
    minimum_edit_count: int
    minimum_alignment_count: int
    evaluated_dag_cell_count: int
    count_scope: str = MINIMUM_ALIGNMENT_COUNT_SCOPE
    algorithm: str = MINIMUM_ALIGNMENT_ALGORITHM_ID


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


@lru_cache(maxsize=8_192)
def minimum_alignment_statistics(
    source: str,
    candidate: str,
    *,
    known_minimum_edit_count: int | None = None,
    max_dag_cells: int = 1_000_000,
) -> MinimumAlignmentStatistics:
    """Return exact alignment distance/count without expanding path states."""

    _validate_rna(source, "source")
    _validate_rna(candidate, "candidate")
    if (
        isinstance(max_dag_cells, bool)
        or not isinstance(max_dag_cells, int)
        or max_dag_cells < 1
    ):
        raise PathStateError("max_dag_cells must be a positive integer")
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
    forward_cost, forward_count, _ = _banded_tables(source, candidate, band)
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
    return MinimumAlignmentStatistics(
        source_sequence=source,
        candidate_sequence=candidate,
        minimum_edit_count=minimum_edit_count,
        minimum_alignment_count=forward_count[terminal],
        evaluated_dag_cell_count=cells,
    )


def _subsequence_embedding_bounds(
    shorter: str,
    longer: str,
) -> tuple[list[int], list[int]]:
    """Return embedding bounds for every cut in the shorter sequence."""

    alphabet = tuple(sorted(RNA_ALPHABET))
    alphabet_index = {base: index for index, base in enumerate(alphabet)}
    m = len(longer)
    next_position = [[m] * len(alphabet) for _ in range(m + 1)]
    for position in range(m - 1, -1, -1):
        next_position[position] = next_position[position + 1].copy()
        next_position[position][alphabet_index[longer[position]]] = position
    previous_position = [[-1] * len(alphabet) for _ in range(m + 1)]
    for end in range(1, m + 1):
        previous_position[end] = previous_position[end - 1].copy()
        previous_position[end][alphabet_index[longer[end - 1]]] = end - 1

    prefix_ends = [0]
    cursor = 0
    for base in shorter:
        position = next_position[cursor][alphabet_index[base]]
        if position == m:
            raise AssertionError("declared subsequence has no leftmost embedding")
        cursor = position + 1
        prefix_ends.append(cursor)

    suffix_starts = [m] * (len(shorter) + 1)
    cursor = m
    for index in range(len(shorter) - 1, -1, -1):
        position = previous_position[cursor][alphabet_index[shorter[index]]]
        if position < 0:
            raise AssertionError("declared subsequence has no rightmost embedding")
        cursor = position
        suffix_starts[index] = cursor
    return prefix_ends, suffix_starts


def _pure_indel_distance_reducing_neighbors(
    sequence: str,
    candidate: str,
    remaining_distance: int,
) -> tuple[Tuple[str, ...], int] | None:
    """Return exact geodesic neighbors when only one indel direction is legal."""

    if len(sequence) - len(candidate) == remaining_distance:
        prefix_matches = [0] * (len(sequence) + 1)
        matched = 0
        for position, base in enumerate(sequence):
            if matched < len(candidate) and base == candidate[matched]:
                matched += 1
            prefix_matches[position + 1] = matched
        if matched != len(candidate):
            raise AssertionError(
                "pure-deletion distance requires candidate to be a subsequence"
            )
        suffix_starts = [len(candidate)] * (len(sequence) + 1)
        matched = len(candidate)
        for position in range(len(sequence) - 1, -1, -1):
            if matched > 0 and sequence[position] == candidate[matched - 1]:
                matched -= 1
            suffix_starts[position] = matched
        if matched != 0:
            raise AssertionError(
                "pure-deletion distance requires candidate to be a subsequence"
            )
        actions = {
            ("DEL", position, "")
            for position in range(len(sequence))
            if prefix_matches[position] >= suffix_starts[position + 1]
        }
    elif len(candidate) - len(sequence) == remaining_distance:
        prefix_ends, suffix_starts = _subsequence_embedding_bounds(
            sequence, candidate
        )
        actions = set()
        for position in range(len(sequence) + 1):
            left = prefix_ends[position]
            right = suffix_starts[position]
            for base in set(candidate[left:right]):
                actions.add(("INS", position, base))
    else:
        return None

    neighbors: set[str] = set()
    for operation, position, base in actions:
        if operation == "DEL":
            neighbors.add(sequence[:position] + sequence[position + 1 :])
        else:
            neighbors.add(sequence[:position] + base + sequence[position:])
    return tuple(sorted(neighbors)), len(actions)


def _distance_reducing_neighbors(
    sequence: str,
    candidate: str,
    remaining_distance: int,
) -> tuple[Tuple[str, ...], int, int]:
    """Return every one-edit neighbor exactly one step closer to candidate.

    A primitive edit is geodesic exactly when it lies on at least one optimal
    prefix/edge/suffix alignment. Enumerating those certified edit edges is
    exhaustive but avoids trying every impossible base at every position.
    Coordinate-distinct edits are counted before their resulting sequence
    identities are collapsed.
    """

    if remaining_distance < 1:
        raise AssertionError("distance-reducing neighbors require distance > 0")
    pure_indel = _pure_indel_distance_reducing_neighbors(
        sequence, candidate, remaining_distance
    )
    if pure_indel is not None:
        neighbors, action_count = pure_indel
        return neighbors, action_count, 0
    forward, backward = _banded_cost_tables(
        sequence, candidate, remaining_distance
    )
    terminal = (len(sequence), len(candidate))
    if forward.get(terminal) != remaining_distance:
        raise AssertionError(
            "state does not have the declared remaining edit distance"
        )

    n, m = len(sequence), len(candidate)
    certified_actions: set[tuple[str, int, str]] = set()
    neighbors: set[str] = set()

    for (i, j), prefix_cost in forward.items():
        if i < n:
            deletion_suffix = backward.get((i + 1, j))
            if (
                deletion_suffix is not None
                and prefix_cost + 1 + deletion_suffix == remaining_distance
            ):
                certified_actions.add(("DEL", i, ""))
        if i < n and j < m and sequence[i] != candidate[j]:
            substitution_suffix = backward.get((i + 1, j + 1))
            if (
                substitution_suffix is not None
                and prefix_cost + 1 + substitution_suffix
                == remaining_distance
            ):
                certified_actions.add(("SUB", i, candidate[j]))
        if j < m:
            insertion_suffix = backward.get((i, j + 1))
            if (
                insertion_suffix is not None
                and prefix_cost + 1 + insertion_suffix == remaining_distance
            ):
                certified_actions.add(("INS", i, candidate[j]))

    for operation, position, base in certified_actions:
        if operation == "DEL":
            neighbors.add(sequence[:position] + sequence[position + 1 :])
        elif operation == "SUB":
            neighbors.add(
                sequence[:position] + base + sequence[position + 1 :]
            )
        else:
            neighbors.add(sequence[:position] + base + sequence[position:])

    evaluated_dp_cells = 2 * _band_cell_count(
        len(sequence), len(candidate), remaining_distance
    )
    return tuple(sorted(neighbors)), len(certified_actions), evaluated_dp_cells


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
    max_reachable_states: int = DEFAULT_MAX_REACHABLE_STATES,
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
            (
                neighbors,
                primitive_action_count,
                state_cells,
            ) = _distance_reducing_neighbors(
                state,
                candidate,
                remaining_before_edit,
            )
            evaluated_state_dp_cell_count += state_cells
            if evaluated_state_dp_cell_count > max_state_dp_cells:
                raise PathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                    f"closure exceeded {max_state_dp_cells} state DP cells; "
                    "no approximation was emitted"
                )
            evaluated_primitive_action_count += primitive_action_count
            if evaluated_primitive_action_count > max_neighbor_expansions:
                raise PathStateError(
                    "STOP_RULE_B0_PATH_STATE_COMPLEXITY: exact shortest-action "
                    f"closure exceeded {max_neighbor_expansions} evaluated "
                    "primitive actions; no approximation was emitted"
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
    "DEFAULT_MAX_REACHABLE_STATES",
    "MINIMUM_ALIGNMENT_ALGORITHM_ID",
    "MINIMUM_ALIGNMENT_COUNT_SCOPE",
    "PRIMITIVE_ACTION_EVALUATION_SCOPE",
    "STATE_CLOSURE_SCOPE",
    "STATE_PATH_COUNT_SCOPE",
    "MinimumAlignmentStatistics",
    "MinimumAlignmentStateClosure",
    "PathStateError",
    "minimum_alignment_statistics",
    "minimum_alignment_state_closure",
]
