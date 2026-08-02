"""MK0-05 factorized hazard, generator and Bregman numerical oracles."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from mrna_editflow.core.mk0.bregman import (
    aggregate_target_weights,
    bregman_loss,
    brute_force_bregman_loss,
)
from mrna_editflow.core.mk0.rate_kernel import (
    FactorizedRates,
    aggregate_transition_rates,
    conditioned_event_distribution,
    enumerate_action_rates,
    generator,
    total_hazard,
    validate_factorization,
)
from mrna_editflow.core.mk0.state_action import apply_action
from mrna_editflow.core.mk0.types import ActionType, AtomicAction, EditState

from .conftest import FLOAT64_ATOL, FLOAT64_RTOL, SEED


MIN_LENGTH = 1
MAX_LENGTH = 6


def _random_factorization(state: EditState, rng: random.Random) -> FactorizedRates:
    ins_operation = tuple(rng.uniform(0.01, 3.0) for _ in range(len(state.current) + 1))
    ins_token_probs = []
    for _ in range(len(state.current) + 1):
        weights = [rng.uniform(0.01, 1.0) for _ in "ACGU"]
        normalizer = math.fsum(weights)
        ins_token_probs.append(
            dict(zip("ACGU", (value / normalizer for value in weights)))
        )
    sub_operation = tuple(rng.uniform(0.01, 3.0) for _ in state.current)
    sub_token_probs = []
    for old in state.current:
        tokens = [token for token in "ACGU" if token != old]
        weights = [rng.uniform(0.01, 1.0) for _ in tokens]
        normalizer = math.fsum(weights)
        sub_token_probs.append(
            dict(zip(tokens, (value / normalizer for value in weights)))
        )
    return FactorizedRates(
        ins_operation=ins_operation,
        ins_token_probs=tuple(ins_token_probs),
        sub_operation=sub_operation,
        sub_token_probs=tuple(sub_token_probs),
        delete=tuple(rng.uniform(0.01, 3.0) for _ in state.current),
        stop=rng.uniform(0.01, 3.0),
    )


def test_factorized_hazard_equals_explicit_action_enumeration() -> None:
    rng = random.Random(SEED)
    sequences = (
        "".join(tokens)
        for length in (1, 2, 3)
        for tokens in itertools.product("AC", repeat=length)
    )
    for sequence in sequences:
        state = EditState.initial(sequence, budget=4)
        for _ in range(16):
            rates = _random_factorization(state, rng)
            action_rates = enumerate_action_rates(
                state,
                rates,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
            manual = (
                rates.stop
                + math.fsum(rates.ins_operation)
                + math.fsum(rates.sub_operation)
            )
            if len(sequence) > MIN_LENGTH:
                manual += math.fsum(rates.delete)
            assert math.isclose(
                total_hazard(action_rates),
                manual,
                abs_tol=FLOAT64_ATOL,
                rel_tol=FLOAT64_RTOL,
            )
            assert all(
                math.isfinite(value) and value >= 0.0 for value in action_rates.values()
            )


def test_hard_mask_precedes_normalization_and_illegal_rate_is_absent() -> None:
    state = EditState.initial("AC", budget=2, protected_indices=(0,))
    action_rates = enumerate_action_rates(
        state,
        FactorizedRates.constant(state),
        min_length=2,
        max_length=2,
    )
    assert AtomicAction(ActionType.SUB, 0, "G") not in action_rates
    assert AtomicAction(ActionType.DEL, 0) not in action_rates
    assert AtomicAction(ActionType.DEL, 1) not in action_rates
    assert not any(action.kind == ActionType.INS for action in action_rates)
    assert AtomicAction(ActionType.STOP) in action_rates
    probabilities = conditioned_event_distribution(action_rates)
    assert set(probabilities) == set(action_rates)
    assert math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1.0e-12)


def test_conditioned_distribution_is_undefined_at_zero_total_hazard() -> None:
    assert total_hazard({}) == 0.0
    with pytest.raises(ZeroDivisionError):
        conditioned_event_distribution({})
    zero = {AtomicAction(ActionType.STOP): 0.0}
    with pytest.raises(ZeroDivisionError):
        conditioned_event_distribution(zero)


def test_conditioned_distribution_rejects_negative_individual_rate_even_if_sum_positive() -> (
    None
):
    rates = {
        AtomicAction(ActionType.STOP): 2.0,
        AtomicAction(ActionType.INS, 0, "A"): -1.0,
    }
    with pytest.raises(FloatingPointError):
        total_hazard(rates)
    with pytest.raises(FloatingPointError):
        conditioned_event_distribution(rates)


@pytest.mark.parametrize("bad", (-1.0, math.nan, math.inf))
def test_negative_or_nonfinite_operation_rates_fail_closed(bad: float) -> None:
    state = EditState.initial("AC", budget=2)
    valid = FactorizedRates.constant(state)
    broken = FactorizedRates(
        ins_operation=(bad,) + valid.ins_operation[1:],
        ins_token_probs=valid.ins_token_probs,
        sub_operation=valid.sub_operation,
        sub_token_probs=valid.sub_token_probs,
        delete=valid.delete,
        stop=valid.stop,
    )
    with pytest.raises(FloatingPointError):
        validate_factorization(state, broken)


def test_token_distributions_normalize_only_over_legal_tokens() -> None:
    state = EditState.initial("A", budget=2)
    valid = FactorizedRates.constant(state)
    assert set(valid.ins_token_probs[0]) == set("ACGU")
    assert set(valid.sub_token_probs[0]) == set("CGU")
    assert math.isclose(math.fsum(valid.sub_token_probs[0].values()), 1.0)

    identity_included = FactorizedRates(
        ins_operation=valid.ins_operation,
        ins_token_probs=valid.ins_token_probs,
        sub_operation=valid.sub_operation,
        sub_token_probs=({"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25},),
        delete=valid.delete,
        stop=valid.stop,
    )
    with pytest.raises(ValueError):
        validate_factorization(state, identity_included)


def test_generator_rows_sum_to_zero_on_exhaustive_tiny_states() -> None:
    for length in (1, 2, 3):
        for tokens in itertools.product("AC", repeat=length):
            state = EditState.initial("".join(tokens), budget=3)
            action_rates = enumerate_action_rates(
                state,
                FactorizedRates.constant(state),
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
            row = generator(
                state,
                action_rates,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
            assert row.diagonal <= 0.0
            assert all(value >= 0.0 for value in row.off_diagonal.values())
            assert math.isclose(
                row.row_sum,
                0.0,
                abs_tol=FLOAT64_ATOL,
                rel_tol=FLOAT64_RTOL,
            )


def test_repeated_symbol_transitions_are_aggregated_before_log() -> None:
    state = EditState.initial("AA", budget=2)
    left = AtomicAction(ActionType.DEL, 0)
    right = AtomicAction(ActionType.DEL, 1)
    action_rates = {left: 0.4, right: 0.6}
    extended = aggregate_transition_rates(
        state,
        action_rates,
        min_length=1,
        max_length=4,
    )
    # Runtime mapping/history make these two next extended states distinct.
    assert len(extended) == 2
    observable = aggregate_transition_rates(
        state,
        action_rates,
        min_length=1,
        max_length=4,
        key_fn=lambda next_state: next_state.current,
    )
    assert observable == {"A": 1.0}
    target = aggregate_target_weights([("A", 0.7), ("A", 0.3)])
    assert target == {"A": 1.0}
    assert bregman_loss(observable, target) == 1.0


def test_bregman_transition_loss_matches_independent_brute_force_oracle() -> None:
    action_pairs = [("next-a", 0.3), ("next-a", 0.7), ("next-b", 2.5)]
    target_pairs = [("next-a", 1.1), ("next-a", 0.4), ("next-b", 0.2)]
    model = {"next-a": 1.0, "next-b": 2.5}
    target = aggregate_target_weights(target_pairs)
    observed = float(bregman_loss(model, target))
    expected = brute_force_bregman_loss(action_pairs, target_pairs)
    assert math.isclose(
        observed,
        expected,
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )
    # Action-level logs would give a different value for duplicated next-a.
    incorrect = math.fsum(rate for _, rate in action_pairs) - math.fsum(
        weight * math.log(rate)
        for (key, weight), (_, rate) in zip(target_pairs, action_pairs)
        if key == "next-a"
    )
    assert not math.isclose(observed, incorrect, abs_tol=1.0e-6)


def test_bregman_loss_and_gradients_are_finite_and_match_analytic_oracle() -> None:
    import torch

    raw = torch.tensor([0.2, -0.4, 1.1], dtype=torch.float64, requires_grad=True)
    rates = torch.nn.functional.softplus(raw)
    weights = {"a": 0.3, "b": 1.7, "c": 0.9}
    loss = bregman_loss(dict(zip(weights, rates)), weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert raw.grad is not None and torch.all(torch.isfinite(raw.grad))

    sigmoid = torch.sigmoid(raw.detach())
    analytic = torch.stack(
        [
            sigmoid[i] * (1.0 - weights[key] / rates.detach()[i])
            for i, key in enumerate(weights)
        ]
    )
    assert torch.allclose(raw.grad, analytic, atol=1.0e-10, rtol=1.0e-8)


def test_bregman_fails_closed_for_illegal_or_zero_rate_target() -> None:
    with pytest.raises(ValueError):
        bregman_loss({"legal": 1.0}, {"illegal": 1.0})
    with pytest.raises(FloatingPointError):
        bregman_loss({"legal": 0.0}, {"legal": 1.0})


@pytest.mark.parametrize("bad_rate", (-0.1, math.nan, math.inf))
def test_bregman_fails_closed_for_invalid_model_rate_without_target_weight(
    bad_rate: float,
) -> None:
    with pytest.raises(FloatingPointError):
        bregman_loss({"bad": bad_rate, "legal": 1.0}, {})


def test_bregman_fails_closed_for_negative_direct_target_weight() -> None:
    with pytest.raises(FloatingPointError):
        bregman_loss({"legal": 1.0}, {"legal": -0.1})


def test_halted_state_has_no_edit_flow_neighbourhood() -> None:
    active = EditState.initial("AC", budget=2)
    halted = apply_action(
        active,
        AtomicAction(ActionType.STOP),
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    ).after
    action_rates = enumerate_action_rates(
        halted,
        FactorizedRates.constant(halted),
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )
    assert action_rates == {}
    assert total_hazard(action_rates) == 0.0
    with pytest.raises(ValueError):
        bregman_loss({}, {})
