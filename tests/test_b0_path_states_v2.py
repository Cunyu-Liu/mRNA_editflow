from __future__ import annotations

from itertools import product

import pytest

from data.utr_benchmark_v2.path_states import DEFAULT_MAX_REACHABLE_STATES
from data.utr_benchmark_v2.path_states import MINIMUM_ALIGNMENT_COUNT_SCOPE
from data.utr_benchmark_v2.path_states import PathStateError
from data.utr_benchmark_v2.path_states import (
    PRIMITIVE_ACTION_EVALUATION_SCOPE,
)
from data.utr_benchmark_v2.path_states import STATE_CLOSURE_SCOPE
from data.utr_benchmark_v2.path_states import STATE_PATH_COUNT_SCOPE
from data.utr_benchmark_v2.path_states import _distance_reducing_neighbors
from data.utr_benchmark_v2.path_states import minimum_alignment_statistics
from data.utr_benchmark_v2.path_states import minimum_alignment_state_closure


def test_all_shortest_action_state_closure_is_exact_and_not_observed() -> None:
    closure = minimum_alignment_state_closure("AC", "CA")
    assert closure.minimum_edit_count == 2
    assert closure.minimum_alignment_count == 3
    assert closure.count_scope == MINIMUM_ALIGNMENT_COUNT_SCOPE
    assert closure.state_closure_scope == STATE_CLOSURE_SCOPE
    assert closure.state_path_count_scope == STATE_PATH_COUNT_SCOPE
    assert closure.reachable_states == (
        "A",
        "AA",
        "AC",
        "ACA",
        "C",
        "CA",
        "CAC",
        "CC",
    )
    assert closure.constructed_intermediate_states == (
        "A",
        "AA",
        "ACA",
        "C",
        "CAC",
        "CC",
    )
    assert closure.minimum_state_path_count == 6
    assert closure.reachable_node_count == 8
    assert closure.reachable_transition_count == 12
    assert closure.constructed_not_observed is True
    assert (
        closure.primitive_action_evaluation_scope
        == PRIMITIVE_ACTION_EVALUATION_SCOPE
    )
    assert len(closure.reachable_states_sha256) == 64


def test_closure_includes_every_shortest_action_execution_order() -> None:
    closure = minimum_alignment_state_closure("AA", "CC")

    # A left-to-right alignment-prefix closure contains CA but misses AC.
    # Both are reachable by valid two-SUB shortest scripts, so omitting AC
    # creates a false-zero path-leakage result.
    assert closure.minimum_edit_count == 2
    assert closure.reachable_states == ("AA", "AC", "CA", "CC")
    assert closure.constructed_intermediate_states == ("AC", "CA")


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_base in enumerate(left, start=1):
        current = [i]
        for j, right_base in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_base != right_base),
                )
            )
        previous = current
    return previous[-1]


def _primitive_neighbors(sequence: str) -> set[str]:
    neighbors: set[str] = set()
    for position, original in enumerate(sequence):
        neighbors.add(sequence[:position] + sequence[position + 1 :])
        for base in "ACGU":
            if base != original:
                neighbors.add(
                    sequence[:position] + base + sequence[position + 1 :]
                )
    for position in range(len(sequence) + 1):
        for base in "ACGU":
            neighbors.add(sequence[:position] + base + sequence[position:])
    return neighbors


def _primitive_geodesic_actions(
    sequence: str,
    candidate: str,
    distance: int,
) -> set[tuple[str, int, str, str]]:
    actions: set[tuple[str, int, str, str]] = set()
    for position, original in enumerate(sequence):
        deletion = sequence[:position] + sequence[position + 1 :]
        if deletion and _edit_distance(deletion, candidate) == distance - 1:
            actions.add(("DEL", position, "", deletion))
        for base in "ACGU":
            if base == original:
                continue
            substitution = (
                sequence[:position] + base + sequence[position + 1 :]
            )
            if _edit_distance(substitution, candidate) == distance - 1:
                actions.add(("SUB", position, base, substitution))
    for position in range(len(sequence) + 1):
        for base in "ACGU":
            insertion = sequence[:position] + base + sequence[position:]
            if _edit_distance(insertion, candidate) == distance - 1:
                actions.add(("INS", position, base, insertion))
    return actions


def test_certified_neighbors_match_exhaustive_primitive_actions() -> None:
    endpoints = [
        "".join(bases)
        for length in (1, 2, 3)
        for bases in product("ACGU", repeat=length)
    ]
    for source in endpoints:
        for candidate in endpoints:
            distance = _edit_distance(source, candidate)
            if distance == 0:
                continue
            expected_actions = _primitive_geodesic_actions(
                source,
                candidate,
                distance,
            )
            expected = tuple(
                sorted({action[3] for action in expected_actions})
            )
            observed, evaluated_count, evaluated_dp_cells = (
                _distance_reducing_neighbors(
                    source, candidate, distance
                )
            )
            assert observed == expected
            assert evaluated_count == len(expected_actions)
            if abs(len(source) - len(candidate)) == distance:
                assert evaluated_dp_cells == 0
            else:
                assert evaluated_dp_cells > 0


def test_alignment_statistics_match_full_closure_without_state_expansion() -> None:
    for source, candidate in (("AC", "CA"), ("AA", "CC"), ("ACGU", "AGU")):
        statistics = minimum_alignment_statistics(source, candidate)
        closure = minimum_alignment_state_closure(source, candidate)
        assert statistics.minimum_edit_count == closure.minimum_edit_count
        assert (
            statistics.minimum_alignment_count
            == closure.minimum_alignment_count
        )


def test_alignment_statistics_limits_and_known_distance_fail_closed() -> None:
    with pytest.raises(PathStateError, match="exceeding"):
        minimum_alignment_statistics("ACGU", "UGCA", max_dag_cells=1)
    with pytest.raises(PathStateError, match="disagrees"):
        minimum_alignment_statistics(
            "AC",
            "CA",
            known_minimum_edit_count=1,
        )


def test_small_repeated_and_indel_cases_match_exhaustive_geodesic_set() -> None:
    endpoints = [
        "".join(bases)
        for length in (1, 2)
        for bases in product("AC", repeat=length)
    ]
    for source in endpoints:
        for candidate in endpoints:
            distance = _edit_distance(source, candidate)
            universe = (
                "".join(bases)
                for length in range(max(len(source), len(candidate)) + distance + 1)
                for bases in product("ACGU", repeat=length)
            )
            expected = tuple(
                sorted(
                    state
                    for state in universe
                    if _edit_distance(source, state)
                    + _edit_distance(state, candidate)
                    == distance
                )
            )
            closure = minimum_alignment_state_closure(source, candidate)
            assert closure.reachable_states == expected

            expected_set = set(expected)
            path_counts = {source: 1}
            transition_count = 0
            for remaining in range(distance, 0, -1):
                layer = sorted(
                    state
                    for state in expected
                    if _edit_distance(state, candidate) == remaining
                )
                for state in layer:
                    neighbors = {
                        neighbor
                        for neighbor in _primitive_neighbors(state)
                        if neighbor in expected_set
                        and _edit_distance(neighbor, candidate) == remaining - 1
                    }
                    transition_count += len(neighbors)
                    for neighbor in neighbors:
                        path_counts[neighbor] = (
                            path_counts.get(neighbor, 0) + path_counts[state]
                        )
            assert closure.reachable_transition_count == transition_count
            assert closure.minimum_state_path_count == path_counts[candidate]


def test_coordinate_equivalent_pure_indels_collapse_to_sequence_edges() -> None:
    for source, candidate in (("AAA", "A"), ("A", "AAA")):
        closure = minimum_alignment_state_closure(source, candidate)
        assert closure.reachable_states == ("A", "AA", "AAA")
        assert closure.reachable_transition_count == 2
        assert closure.minimum_state_path_count == 1
        assert closure.evaluated_primitive_action_count == (
            3 if source == "AAA" else 2
        )
        assert closure.evaluated_state_dp_cell_count == 0


def test_frozen_complexity_witness_is_exact_under_audited_default_guard() -> None:
    source = (
        "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCUGAC"
        "UUUCAGCGGGCGGAAAAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAA"
        "AAAUGUC"
    )
    candidate = (
        "CUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCGGAA"
        "AAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAAAAAUGUC"
    )
    assert DEFAULT_MAX_REACHABLE_STATES == 50_000
    with pytest.raises(
        PathStateError, match="STOP_RULE_B0_PATH_STATE_COMPLEXITY"
    ):
        minimum_alignment_state_closure(
            source,
            candidate,
            known_minimum_edit_count=15,
        )
    closure = minimum_alignment_state_closure(
        source,
        candidate,
        known_minimum_edit_count=15,
        max_reachable_states=100_000,
    )
    assert closure.minimum_alignment_count == 2_340
    assert closure.reachable_node_count == 95_217
    assert closure.reachable_transition_count == 751_771
    assert closure.minimum_state_path_count == 3_934_510_691_993
    assert closure.evaluated_primitive_action_count == 1_205_477
    assert closure.evaluated_state_dp_cell_count == 0
    assert (
        closure.reachable_states_sha256
        == "900076096ad75979a1b592b6d14fd7647dfe54c39b4cee80a053937de9411332"
    )


def test_dag_resource_guard_and_rna_validation_fail_closed() -> None:
    with pytest.raises(PathStateError, match="exceeding"):
        minimum_alignment_state_closure(
            "ACGU", "UGCA", max_dag_cells=1
        )
    with pytest.raises(PathStateError, match="RNA alphabet"):
        minimum_alignment_state_closure("ACGT", "ACGU")
    with pytest.raises(
        PathStateError, match="STOP_RULE_B0_PATH_STATE_COMPLEXITY"
    ):
        minimum_alignment_state_closure(
            "AA", "CC", max_reachable_states=3
        )
    with pytest.raises(
        PathStateError, match="STOP_RULE_B0_PATH_STATE_COMPLEXITY"
    ):
        minimum_alignment_state_closure(
            "AAA", "A", max_neighbor_expansions=2
        )
    with pytest.raises(
        PathStateError, match="STOP_RULE_B0_PATH_STATE_COMPLEXITY"
    ):
        minimum_alignment_state_closure(
            "AC", "CA", max_state_dp_cells=1
        )
