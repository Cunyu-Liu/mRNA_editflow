"""MK0-02 state/action exactness, legality, inverse and replay tests."""

from __future__ import annotations

import random

import pytest

from mrna_editflow.core.mk0.state_action import (
    IllegalAction,
    action_mask,
    apply_action,
    enumerate_legal_actions,
    force_terminate,
    is_legal,
    replay_actions,
    undo_transition,
)
from mrna_editflow.core.mk0.types import (
    ALPHABET,
    ActionType,
    AtomicAction,
    EditState,
    Phase,
    TerminationReason,
    TokenOrigin,
)

from .conftest import SEED, rna_sequences


MIN_LENGTH = 1
MAX_LENGTH = 5


def _expected_apply(sequence: str, action: AtomicAction) -> str:
    if action.kind == ActionType.STOP:
        return sequence
    position = int(action.position)
    if action.kind == ActionType.INS:
        return sequence[:position] + str(action.token) + sequence[position:]
    if action.kind == ActionType.SUB:
        return sequence[:position] + str(action.token) + sequence[position + 1 :]
    if action.kind == ActionType.DEL:
        return sequence[:position] + sequence[position + 1 :]
    raise AssertionError(action)


def _unprotected_initial(sequence: str, *, budget: int = 8) -> EditState:
    return EditState.initial(sequence, budget=budget, region="5UTR")


def test_atomic_action_schema_rejects_malformed_actions() -> None:
    with pytest.raises(ValueError):
        AtomicAction(ActionType.STOP, 0)
    with pytest.raises(ValueError):
        AtomicAction(ActionType.INS, -1, "A")
    with pytest.raises(ValueError):
        AtomicAction(ActionType.SUB, 0, "T")
    with pytest.raises(ValueError):
        AtomicAction(ActionType.DEL, 0, "A")


def test_state_schema_rejects_non_rna_and_inconsistent_budget() -> None:
    with pytest.raises(ValueError):
        EditState.initial("ACT", budget=2)

    valid = EditState.initial("AC", budget=2)
    with pytest.raises(ValueError):
        EditState(
            source=valid.source,
            current=valid.current,
            mapping=valid.mapping,
            region=valid.region,
            context=valid.context,
            target_condition=valid.target_condition,
            initial_budget=2,
            remaining_budget=1,
        )


@pytest.mark.parametrize("sequence", tuple(rna_sequences(1, 3)))
def test_exhaustive_legal_action_cardinality(sequence: str) -> None:
    state = _unprotected_initial(sequence)
    actions = enumerate_legal_actions(
        state, min_length=MIN_LENGTH, max_length=MAX_LENGTH
    )
    length = len(sequence)
    expected = 4 * (length + 1) + 3 * length + int(length > MIN_LENGTH) * length + 1
    assert len(actions) == expected
    assert len({action.key for action in actions}) == expected
    assert actions[-1] == AtomicAction(ActionType.STOP)
    assert all(
        is_legal(state, action, min_length=MIN_LENGTH, max_length=MAX_LENGTH)
        for action in actions
    )


@pytest.mark.parametrize("sequence", tuple(rna_sequences(1, 3)))
def test_apply_is_exact_for_every_tiny_legal_action(sequence: str) -> None:
    state = _unprotected_initial(sequence)
    for action in enumerate_legal_actions(
        state, min_length=MIN_LENGTH, max_length=MAX_LENGTH
    ):
        transition = apply_action(
            state, action, min_length=MIN_LENGTH, max_length=MAX_LENGTH
        )
        assert transition.before is state
        assert transition.after.current == _expected_apply(sequence, action)
        assert undo_transition(transition) == state
        if action.kind == ActionType.STOP:
            assert transition.after.phase == Phase.HALTED
            assert transition.after.termination_reason == TerminationReason.LEARNED_STOP
            assert transition.after.remaining_budget == state.remaining_budget
            assert transition.after.history == state.history
        else:
            assert transition.after.phase == Phase.ACTIVE
            assert transition.after.remaining_budget == state.remaining_budget - 1
            assert transition.after.history.executed == 1
            assert len(transition.after.mapping.tokens) == len(transition.after.current)
            assert (
                len(transition.after.mapping.gap_ids)
                == len(transition.after.current) + 1
            )


def test_inserted_identity_and_source_identity_survive_coordinate_updates() -> None:
    state = EditState.initial("ACG", budget=4, protected_indices=(1,))
    inserted = apply_action(
        state,
        AtomicAction(ActionType.INS, 1, "U"),
        min_length=1,
        max_length=8,
    ).after
    assert inserted.current == "AUCG"
    assert inserted.mapping.tokens[1].origin == TokenOrigin.INSERTED
    assert inserted.mapping.tokens[1].stable_id == "ins:1"
    assert inserted.mapping.tokens[2].stable_id == "src:1"
    assert inserted.mapping.tokens[2].protected is True

    substituted = apply_action(
        inserted,
        AtomicAction(ActionType.SUB, 1, "G"),
        min_length=1,
        max_length=8,
    ).after
    assert substituted.mapping.tokens[1].stable_id == "ins:1"

    deleted = apply_action(
        substituted,
        AtomicAction(ActionType.DEL, 1),
        min_length=1,
        max_length=8,
    ).after
    assert deleted.current == "ACG"
    assert tuple(ref.stable_id for ref in deleted.mapping.tokens) == (
        "src:0",
        "src:1",
        "src:2",
    )
    assert deleted.remaining_budget == 1
    assert deleted.history.executed == 3


def test_protected_anchor_and_adjacent_internal_gap_are_hard_masked() -> None:
    state = EditState.initial("ACG", budget=4, protected_indices=(0, 1))
    protected_edits = (
        AtomicAction(ActionType.SUB, 0, "C"),
        AtomicAction(ActionType.DEL, 1),
        AtomicAction(ActionType.INS, 1, "U"),
    )
    assert action_mask(
        state,
        protected_edits,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    ) == (False, False, False)
    for action in protected_edits:
        with pytest.raises(IllegalAction):
            apply_action(state, action, min_length=MIN_LENGTH, max_length=MAX_LENGTH)


def test_identity_substitution_bounds_and_budget_are_hard_masked() -> None:
    state = EditState.initial("A", budget=1)
    illegal = (
        AtomicAction(ActionType.SUB, 0, "A"),
        AtomicAction(ActionType.DEL, 0),
        AtomicAction(ActionType.INS, 2, "C"),
    )
    assert not any(
        is_legal(state, action, min_length=1, max_length=1) for action in illegal
    )
    exhausted = apply_action(
        state,
        AtomicAction(ActionType.SUB, 0, "C"),
        min_length=1,
        max_length=3,
    ).after
    assert exhausted.remaining_budget == 0
    assert enumerate_legal_actions(exhausted, min_length=1, max_length=3) == (
        AtomicAction(ActionType.STOP),
    )


def test_reverse_edits_and_cycles_still_consume_budget() -> None:
    initial = EditState.initial("A", budget=2)
    actions = (
        AtomicAction(ActionType.SUB, 0, "C"),
        AtomicAction(ActionType.SUB, 0, "A"),
    )
    final, records = replay_actions(initial, actions, min_length=1, max_length=3)
    assert final.current == initial.current
    assert final.remaining_budget == 0
    assert final.history.executed == 2
    assert tuple(record.action for record in records) == actions
    assert final.state_hash != initial.state_hash


def test_deterministic_replay_is_bit_exact_for_random_legal_trajectories() -> None:
    rng = random.Random(SEED)
    for _ in range(256):
        initial = EditState.initial(
            "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 4))),
            budget=6,
            region=rng.choice(("5UTR", "3UTR")),
            context={"assay": "toy", "batch": str(rng.randint(1, 3))},
        )
        current = initial
        actions: list[AtomicAction] = []
        for _step in range(6):
            legal = enumerate_legal_actions(
                current,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                include_stop=False,
            )
            if not legal:
                break
            action = rng.choice(legal)
            actions.append(action)
            current = apply_action(
                current,
                action,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            ).after
        replayed, records = replay_actions(
            initial,
            actions,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )
        assert replayed == current
        assert replayed.state_hash == current.state_hash
        assert all(undo_transition(record) == record.before for record in records)


def test_halted_is_absorbing_and_learned_stop_is_not_forced() -> None:
    active = EditState.initial("AC", budget=2)
    halted = apply_action(
        active,
        AtomicAction(ActionType.STOP),
        min_length=1,
        max_length=4,
    ).after
    assert enumerate_legal_actions(halted, min_length=1, max_length=4) == ()
    with pytest.raises(IllegalAction):
        apply_action(
            halted,
            AtomicAction(ActionType.SUB, 0, "G"),
            min_length=1,
            max_length=4,
        )
    with pytest.raises(ValueError):
        force_terminate(active, TerminationReason.LEARNED_STOP)

    forced = force_terminate(active, TerminationReason.FORCED_BUDGET)
    assert forced.phase == Phase.HALTED
    assert forced.termination_reason == TerminationReason.FORCED_BUDGET
    assert forced.termination_reason != halted.termination_reason


def test_inference_payload_has_no_target_alignment_or_remaining_target_edits() -> None:
    payload = EditState.initial(
        "AC", context={"assay": "toy"}, target_condition="increase"
    ).inference_dict()
    serialized_keys = repr(payload).lower()
    assert "alignment" not in serialized_keys
    assert "target_sequence" not in serialized_keys
    assert "remaining_target" not in serialized_keys
    assert payload["target_condition"] == "increase"


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "target_sequence",
        "target_alignment",
        "remaining_target_edits",
        "z_aux",
        "target_derived_embedding",
        "replicate",
    ),
)
def test_state_constructor_rejects_unknown_or_training_only_context_keys(
    forbidden_key: str,
) -> None:
    with pytest.raises(ValueError, match="non-inference-visible"):
        EditState.initial("AC", context={"assay": "toy", forbidden_key: "secret"})


def test_context_is_canonical_and_serializer_cannot_hide_internal_keys() -> None:
    state = EditState.initial("AC", context={"assay": "toy"})
    assert dict(state.context) == {
        "assay": "toy",
        "cell_or_tissue": "unspecified",
        "endpoint": "unspecified",
        "batch": None,
    }
    with pytest.raises(ValueError, match="non-inference-visible"):
        EditState(
            source=state.source,
            current=state.current,
            mapping=state.mapping,
            region=state.region,
            context=state.context + (("target_sequence", "UU"),),
            target_condition=state.target_condition,
            initial_budget=state.initial_budget,
            remaining_budget=state.remaining_budget,
        )
