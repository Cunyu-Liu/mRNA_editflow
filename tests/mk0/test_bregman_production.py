"""Production state-aware Edit Flow/Bregman loss tests."""

from __future__ import annotations

import math

import pytest

from mrna_editflow.core.mk0.alignment_coupling import (
    build_alignment,
    changed_indices,
)
from mrna_editflow.core.mk0.bregman import edit_flow_loss
from mrna_editflow.core.mk0.rate_kernel import aggregate_transition_rates
from mrna_editflow.core.mk0.schedule import rho
from mrna_editflow.core.mk0.state_action import apply_action, enumerate_legal_actions
from mrna_editflow.core.mk0.target_kernel import build_target_transition_oracle
from mrna_editflow.core.mk0.types import ActionType, AtomicAction, EditState


def _oracle(
    state: EditState,
    target: str,
    *,
    t: float = 0.5,
    min_length: int,
    max_length: int,
):
    alignment = build_alignment(state.source, target)
    clocks = {index: 0.75 for index in changed_indices(alignment)}
    return build_target_transition_oracle(
        state,
        alignment,
        clocks,
        t,
        min_length=min_length,
        max_length=max_length,
    )


def _complete_rates(
    state: EditState,
    *,
    min_length: int,
    max_length: int,
    overrides: dict[str, object] | None = None,
) -> dict[AtomicAction, object]:
    overrides = overrides or {}
    return {
        action: overrides.get(action.key, 0.0)
        for action in enumerate_legal_actions(
            state,
            min_length=min_length,
            max_length=max_length,
            include_stop=False,
        )
    }


def test_production_loss_uses_distinct_full_states_for_repeated_deletions() -> None:
    state = EditState.initial("AA", budget=2)
    oracle = _oracle(state, "", min_length=0, max_length=2)
    action_rates = _complete_rates(
        state,
        min_length=0,
        max_length=2,
        overrides={"DEL:0": 0.4, "DEL:1": 0.6},
    )
    observed = float(
        edit_flow_loss(
            state,
            action_rates,
            oracle,
            min_length=0,
            max_length=2,
        )
    )
    expected = 1.0 - rho(0.5) * (math.log(0.4) + math.log(0.6))
    assert math.isclose(observed, expected, abs_tol=1.0e-12, rel_tol=1.0e-10)

    aggregated = aggregate_transition_rates(
        state,
        action_rates,
        min_length=0,
        max_length=2,
    )
    assert (
        len(
            {
                transition.next_state_hash
                for transition in oracle.transitions
                if transition.observable_next == "A"
            }
        )
        == 2
    )
    assert all(key in aggregated for key in oracle.target_transition_weights)


def test_production_loss_aggregates_target_multiplicity_before_log() -> None:
    state = EditState.initial("", budget=2)
    oracle = _oracle(state, "AA", min_length=0, max_length=2)
    action_rates = _complete_rates(
        state,
        min_length=0,
        max_length=2,
        overrides={"INS:0:A": 2.0},
    )
    assert len(oracle.transitions) == 2
    assert len(oracle.target_transition_weights) == 1
    observed = float(
        edit_flow_loss(
            state,
            action_rates,
            oracle,
            min_length=0,
            max_length=2,
        )
    )
    expected = 2.0 - 2.0 * rho(0.5) * math.log(2.0)
    assert math.isclose(observed, expected, abs_tol=1.0e-12, rel_tol=1.0e-10)


def test_production_loss_gradient_matches_full_transition_oracle() -> None:
    import torch

    state = EditState.initial("A", budget=1)
    oracle = _oracle(state, "C", min_length=1, max_length=1)
    legal = enumerate_legal_actions(
        state, min_length=1, max_length=1, include_stop=False
    )
    raw = torch.tensor([-0.4, 0.2, 1.1], dtype=torch.float64, requires_grad=True)
    rates = torch.nn.functional.softplus(raw)
    action_rates = dict(zip(legal, rates))
    loss = edit_flow_loss(
        state,
        action_rates,
        oracle,
        min_length=1,
        max_length=1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad is not None and torch.all(torch.isfinite(raw.grad))

    expected = torch.sigmoid(raw.detach())
    target_index = next(
        index for index, action in enumerate(legal) if action.key == "SUB:0:C"
    )
    expected[target_index] *= 1.0 - rho(0.5) / rates.detach()[target_index]
    assert torch.allclose(raw.grad, expected, atol=1.0e-12, rtol=1.0e-10)


def test_halted_state_contributes_exact_zero_and_rejects_any_edit_term() -> None:
    active = EditState.initial("A", budget=1)
    halted = apply_action(
        active,
        AtomicAction(ActionType.STOP),
        min_length=1,
        max_length=1,
    ).after
    assert edit_flow_loss(halted, {}, None, min_length=1, max_length=1) == 0.0

    with pytest.raises(ValueError, match="cannot expose model edit rates"):
        edit_flow_loss(
            halted,
            {AtomicAction(ActionType.SUB, 0, "C"): 1.0},
            None,
            min_length=1,
            max_length=1,
        )
    active_oracle = _oracle(active, "C", min_length=1, max_length=1)
    with pytest.raises(ValueError, match="cannot receive a target edit oracle"):
        edit_flow_loss(
            halted,
            {},
            active_oracle,
            min_length=1,
            max_length=1,
        )


def test_active_state_without_legal_edits_has_an_exact_zero_edit_term() -> None:
    state = EditState.initial("A", budget=0)
    oracle = _oracle(state, "A", min_length=1, max_length=1)
    assert edit_flow_loss(state, {}, oracle, min_length=1, max_length=1) == 0.0


def test_production_loss_rejects_incomplete_or_stop_contaminated_rate_domain() -> None:
    state = EditState.initial("A", budget=1)
    oracle = _oracle(state, "C", min_length=1, max_length=1)
    full_rates = _complete_rates(state, min_length=1, max_length=1)
    missing = dict(full_rates)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="complete legal neighbourhood"):
        edit_flow_loss(state, missing, oracle, min_length=1, max_length=1)

    contaminated = dict(full_rates)
    contaminated[AtomicAction(ActionType.STOP)] = 1.0
    with pytest.raises(ValueError, match="complete legal neighbourhood"):
        edit_flow_loss(state, contaminated, oracle, min_length=1, max_length=1)


def test_production_loss_rejects_oracle_bound_to_another_extended_state() -> None:
    first = EditState.initial("A", budget=1)
    second = EditState.initial(
        "A", budget=1, context={"assay": "different-runtime-context"}
    )
    oracle = _oracle(first, "C", min_length=1, max_length=1)
    rates = _complete_rates(second, min_length=1, max_length=1)
    with pytest.raises(ValueError, match="different extended state"):
        edit_flow_loss(second, rates, oracle, min_length=1, max_length=1)
