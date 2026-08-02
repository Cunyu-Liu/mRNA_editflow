"""Target alignment/switch-clock to next-extended-state kernel oracles."""

from __future__ import annotations

from collections import Counter
import math
import random

import pytest

from mrna_editflow.core.mk0.alignment_coupling import (
    build_alignment,
    changed_indices,
    sample_optimal_alignment,
)
from mrna_editflow.core.mk0.schedule import rho
from mrna_editflow.core.mk0.state_action import apply_action
from mrna_editflow.core.mk0.target_kernel import (
    TargetKernelRejected,
    build_target_transition_oracle,
)
from mrna_editflow.core.mk0.types import ActionType, AtomicAction, EditState

from .conftest import FLOAT64_ATOL, FLOAT64_RTOL, SEED


MIN_LENGTH = 0
MAX_LENGTH = 8
REPEATED_SYMBOL_CASES = (
    ("AA", "A"),
    ("A", "AA"),
    ("AAA", "AA"),
    ("AA", "AAA"),
    ("ACA", "AAC"),
)


def _all_remaining_clocks(alignment, value: float = 0.75) -> dict[int, float]:
    return {index: value for index in changed_indices(alignment)}


def _assert_oracle_replays_exactly(state: EditState, oracle) -> None:
    expected_weights: dict[str, list[float]] = {}
    for transition in oracle.transitions:
        replayed = apply_action(
            state,
            transition.action,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        ).after
        assert replayed.state_hash == transition.next_state_hash
        assert replayed.current == transition.observable_next
        expected_weights.setdefault(replayed.state_hash, []).append(transition.weight)
    assert oracle.target_transition_weights == {
        key: math.fsum(weights) for key, weights in expected_weights.items()
    }


@pytest.mark.parametrize(("source", "target"), REPEATED_SYMBOL_CASES)
def test_required_repeated_symbol_canonical_paths_map_to_full_state_hashes(
    source: str, target: str
) -> None:
    alignment = build_alignment(source, target)
    state = EditState.initial(source, budget=max(4, alignment.cost))
    oracle = build_target_transition_oracle(
        state,
        alignment,
        _all_remaining_clocks(alignment),
        0.5,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )

    assert oracle.source_state_hash == state.state_hash
    assert oracle.alignment_hash == alignment.alignment_hash
    assert len(oracle.transitions) == alignment.cost
    assert tuple(t.alignment_index for t in oracle.transitions) == changed_indices(
        alignment
    )
    assert all(t.action.kind != ActionType.STOP for t in oracle.transitions)
    assert all(entry.status == "ACCEPTED" for entry in oracle.ledger)
    assert all(entry.repair_status == "NOT_NEEDED" for entry in oracle.ledger)
    assert math.isclose(
        math.fsum(oracle.target_transition_weights.values()),
        alignment.cost * rho(0.5),
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )
    _assert_oracle_replays_exactly(state, oracle)
    record = oracle.to_record()
    assert record["aggregation_key"] == "full_next_extended_state_sha256"
    assert record["rejected_path_count"] == record["repair_applied_count"] == 0


@pytest.mark.parametrize(("source", "target"), REPEATED_SYMBOL_CASES)
def test_sampled_optimal_path_sensitivity_uses_the_same_legal_bridge(
    source: str, target: str
) -> None:
    rng = random.Random(SEED)
    state = EditState.initial(source, budget=4)
    observed_alignment_hashes: set[str] = set()
    for _ in range(128):
        alignment = sample_optimal_alignment(source, target, rng=rng)
        observed_alignment_hashes.add(alignment.alignment_hash)
        oracle = build_target_transition_oracle(
            state,
            alignment,
            _all_remaining_clocks(alignment),
            0.5,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )
        assert len(oracle.transitions) == alignment.cost
        _assert_oracle_replays_exactly(state, oracle)

    if source != "ACA":
        assert len(observed_alignment_hashes) > 1


def test_target_multiplicity_sums_rho_only_after_full_state_mapping() -> None:
    # Both latent insertion coordinates induce exactly INS(0,A) at the empty
    # source state and therefore reach the same full next extended state.
    alignment = build_alignment("", "AA")
    state = EditState.initial("", budget=2)
    oracle = build_target_transition_oracle(
        state,
        alignment,
        _all_remaining_clocks(alignment),
        0.5,
        min_length=0,
        max_length=2,
    )
    assert [transition.action.key for transition in oracle.transitions] == [
        "INS:0:A",
        "INS:0:A",
    ]
    assert len({transition.next_state_hash for transition in oracle.transitions}) == 1
    assert len(oracle.target_transition_weights) == 1
    assert next(iter(oracle.target_transition_weights.values())) == 2.0 * rho(0.5)


def test_observable_collision_never_conflates_distinct_extended_states() -> None:
    # Deleting either repeated A yields the same observable sequence, but the
    # source-token mapping and action history distinguish the extended states.
    alignment = build_alignment("AA", "")
    state = EditState.initial("AA", budget=2)
    oracle = build_target_transition_oracle(
        state,
        alignment,
        _all_remaining_clocks(alignment),
        0.5,
        min_length=0,
        max_length=2,
    )
    assert [transition.observable_next for transition in oracle.transitions] == [
        "A",
        "A",
    ]
    assert {transition.action.key for transition in oracle.transitions} == {
        "DEL:0",
        "DEL:1",
    }
    assert len({transition.next_state_hash for transition in oracle.transitions}) == 2
    assert len(oracle.target_transition_weights) == 2
    assert Counter(oracle.target_transition_weights.values()) == Counter(
        (rho(0.5), rho(0.5))
    )


def test_intermediate_switched_state_maps_remaining_coordinate_in_current_coordinates() -> (
    None
):
    alignment = build_alignment("AC", "GU")
    changed = changed_indices(alignment)
    assert len(changed) == 2
    clocks = {changed[0]: 0.25, changed[1]: 0.75}
    initial = EditState.initial("AC", budget=2)
    current = apply_action(
        initial,
        AtomicAction(ActionType.SUB, 0, "G"),
        min_length=0,
        max_length=4,
    ).after
    oracle = build_target_transition_oracle(
        current,
        alignment,
        clocks,
        0.5,
        min_length=0,
        max_length=4,
    )
    assert [transition.action.key for transition in oracle.transitions] == ["SUB:1:U"]
    assert oracle.transitions[0].observable_next == "GU"
    _assert_oracle_replays_exactly(current, oracle)


def test_equivalent_completed_edit_orders_bind_to_their_own_extended_state() -> None:
    alignment = build_alignment("AC", "GU")
    clocks = {index: 0.25 for index in changed_indices(alignment)}
    initial = EditState.initial("AC", budget=2)
    first_order = apply_action(
        initial,
        AtomicAction(ActionType.SUB, 0, "G"),
        min_length=0,
        max_length=4,
    ).after
    first_order = apply_action(
        first_order,
        AtomicAction(ActionType.SUB, 1, "U"),
        min_length=0,
        max_length=4,
    ).after
    second_order = apply_action(
        initial,
        AtomicAction(ActionType.SUB, 1, "U"),
        min_length=0,
        max_length=4,
    ).after
    second_order = apply_action(
        second_order,
        AtomicAction(ActionType.SUB, 0, "G"),
        min_length=0,
        max_length=4,
    ).after
    assert first_order.current == second_order.current == "GU"
    assert first_order.state_hash != second_order.state_hash

    first_oracle = build_target_transition_oracle(
        first_order,
        alignment,
        clocks,
        0.5,
        min_length=0,
        max_length=4,
    )
    second_oracle = build_target_transition_oracle(
        second_order,
        alignment,
        clocks,
        0.5,
        min_length=0,
        max_length=4,
    )
    assert first_oracle.source_state_hash == first_order.state_hash
    assert second_oracle.source_state_hash == second_order.state_hash
    assert first_oracle.transitions == second_oracle.transitions == ()


def test_illegal_target_switch_records_rejected_and_unattempted_repair_then_raises() -> (
    None
):
    alignment = build_alignment("A", "C")
    protected = EditState.initial("A", budget=1, protected_indices=(0,))
    with pytest.raises(TargetKernelRejected) as caught:
        build_target_transition_oracle(
            protected,
            alignment,
            _all_remaining_clocks(alignment),
            0.5,
            min_length=1,
            max_length=1,
        )
    assert len(caught.value.ledger) == 1
    entry = caught.value.ledger[0]
    assert entry.action_key == "SUB:0:C"
    assert entry.status == "REJECTED_FAIL_CLOSED"
    assert entry.reason == "target action is forbidden by the runtime hard mask"
    assert entry.repair_status == "NOT_ATTEMPTED_FAIL_CLOSED"
    assert entry.repair_action_key is None
    assert entry.next_state_hash is None
    rejection_record = caught.value.to_record()
    assert rejection_record["status"] == "REJECTED_FAIL_CLOSED"
    assert rejection_record["repair_applied_count"] == 0
    assert rejection_record["ledger"] == [entry.to_record()]


@pytest.mark.parametrize(
    "mutation",
    ("missing_clock", "bad_clock", "history_mismatch", "current_mismatch", "budget"),
)
def test_global_auxiliary_runtime_conflicts_fail_closed_with_ledger(
    mutation: str,
) -> None:
    alignment = build_alignment("A", "C")
    clocks = _all_remaining_clocks(alignment)
    state = EditState.initial("A", budget=1)
    if mutation == "missing_clock":
        clocks = {}
    elif mutation == "bad_clock":
        clocks = {changed_indices(alignment)[0]: math.nan}
    elif mutation == "history_mismatch":
        clocks = {changed_indices(alignment)[0]: 0.25}
    elif mutation == "current_mismatch":
        state = EditState.initial("A", budget=1)
        alignment = build_alignment("A", "G")
        clocks = {changed_indices(alignment)[0]: 0.25}
    elif mutation == "budget":
        state = EditState.initial("A", budget=0)

    with pytest.raises(TargetKernelRejected) as caught:
        build_target_transition_oracle(
            state,
            alignment,
            clocks,
            0.5,
            min_length=0,
            max_length=4,
        )
    assert caught.value.ledger
    assert all(entry.status == "REJECTED_FAIL_CLOSED" for entry in caught.value.ledger)
    assert all(
        entry.repair_status == "NOT_ATTEMPTED_FAIL_CLOSED"
        for entry in caught.value.ledger
    )
