from __future__ import annotations

import pytest

from core.ef0.model import EF0ModelConfig, TrueUTREditFlowRateField
from core.mk0.state_action import IllegalAction, apply_action, enumerate_legal_actions
from core.mk0.types import ActionType, AtomicAction, EditState, Phase


def _state() -> EditState:
    return EditState.initial(
        "ACGU",
        region="5UTR",
        context={
            "assay": "ef0-test",
            "cell_or_tissue": "cell",
            "endpoint": "MRL",
            "batch": "b0",
        },
        target_condition="increase",
        budget=4,
        protected_indices=(1,),
    )


def test_dynamic_mapping_and_budget_are_exact() -> None:
    state = _state()
    insertion = AtomicAction(ActionType.INS, 0, "A")
    after_insert = apply_action(state, insertion, min_length=1, max_length=8).after
    assert after_insert.current == "AACGU"
    assert after_insert.remaining_budget == 3
    assert after_insert.mapping.tokens[0].origin.value == "INSERTED"
    assert after_insert.mapping.tokens[1].source_index == 0

    deletion = AtomicAction(ActionType.DEL, 0)
    after_delete = apply_action(
        after_insert, deletion, min_length=1, max_length=8
    ).after
    assert after_delete.current == "ACGU"
    assert after_delete.remaining_budget == 2
    assert after_delete.history.executed == 2


def test_protected_position_is_constructively_masked() -> None:
    state = _state()
    illegal = AtomicAction(ActionType.SUB, 1, "A")
    assert illegal not in enumerate_legal_actions(
        state, min_length=1, max_length=8, include_stop=True
    )
    with pytest.raises(IllegalAction):
        apply_action(state, illegal, min_length=1, max_length=8)


def test_stop_is_absorbing_and_costs_no_budget() -> None:
    state = _state()
    stopped = apply_action(
        state,
        AtomicAction(ActionType.STOP),
        min_length=1,
        max_length=8,
    ).after
    assert stopped.phase is Phase.HALTED
    assert stopped.remaining_budget == state.remaining_budget
    assert stopped.termination_reason.value == "LEARNED_STOP"
    assert enumerate_legal_actions(
        stopped, min_length=1, max_length=8, include_stop=True
    ) == ()


def test_ef0_has_explicit_operation_heads_and_inference_only_signature() -> None:
    # This is a structural test; neural construction and forward/backward are
    # deliberately reserved for the GPU acceptance runner.
    assert EF0ModelConfig().max_length == 256
    assert set(ActionType) == {
        ActionType.INS,
        ActionType.SUB,
        ActionType.DEL,
        ActionType.STOP,
    }
    assert set(TrueUTREditFlowRateField.inference_signature_fields) == {
        "source",
        "current",
        "M_run",
        "region",
        "context",
        "target_condition",
        "time",
        "remaining_budget",
        "h_run",
    }
