from __future__ import annotations

from itertools import product

import pytest

from data.utr_benchmark_v2.path_states import MINIMUM_ALIGNMENT_COUNT_SCOPE
from data.utr_benchmark_v2.path_states import PathStateError
from data.utr_benchmark_v2.path_states import STATE_CLOSURE_SCOPE
from data.utr_benchmark_v2.path_states import STATE_PATH_COUNT_SCOPE
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
