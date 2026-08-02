"""MK0-07 zero-hazard, validity, replay and convergence tests."""

from __future__ import annotations

from dataclasses import replace
import itertools
import math

import pytest

from mrna_editflow.core.mk0.rate_kernel import FactorizedRates, enumerate_action_rates
from mrna_editflow.core.mk0.samplers import (
    certify_remaining_integrated_hazard,
    constrained_single_event_first_order,
    paper_first_order_parallel,
    replay_constrained_result,
    replay_paper_result,
    sampler_result_to_schema_record,
)
from mrna_editflow.core.mk0.state_action import (
    enumerate_legal_actions,
    validate_schema_facing_record,
)
from mrna_editflow.core.mk0.types import (
    ActionType,
    AtomicAction,
    EditState,
    Phase,
    TerminationReason,
)

from .conftest import FLOAT64_ATOL, FLOAT64_RTOL, SEED


MIN_LENGTH = 1
MAX_LENGTH = 6


def _remaining_certificate(rate_fn, state: EditState, time: float):
    return certify_remaining_integrated_hazard(
        state,
        time,
        rate_fn,
        horizon=1.0,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )


def _factorized_rate_fn(state: EditState, _time: float):
    return enumerate_action_rates(
        state,
        FactorizedRates.constant(state, ins=0.25, sub=0.35, delete=0.15, stop=0.10),
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )


def _stop_only_rate(hazard: float):
    def rate_fn(state: EditState, _time: float):
        return (
            {AtomicAction(ActionType.STOP): hazard}
            if state.phase == Phase.ACTIVE
            else {}
        )

    return rate_fn


def test_zero_instantaneous_hazard_advances_without_division_or_termination() -> None:
    def delayed_stop(state: EditState, time: float):
        return {AtomicAction(ActionType.STOP): 0.0 if time == 0.0 else 2.0}

    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        delayed_stop,
        step_size=0.25,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    first = result.steps[0]
    assert first.outcome == "NO_EVENT"
    assert first.t_start == 0.0 and first.t_end == 0.25
    assert first.total_hazard == 0.0
    assert first.event_probability == 0.0
    assert first.action_draw is None and first.selected_action is None
    assert result.final_state.termination_reason != (
        TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
    )


def test_zero_remaining_integrated_hazard_requires_separate_positive_verification() -> (
    None
):
    zero_rates = lambda _state, _time: {}
    verifier = lambda state, time: _remaining_certificate(zero_rates, state, time)
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        zero_rates,
        step_size=0.25,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        remaining_hazard_verifier=verifier,
    )
    assert result.steps == ()
    assert result.final_state.termination_reason == (
        TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
    )
    assert result.remaining_hazard_certificate is not None
    assert result.remaining_hazard_certificate.verified_zero
    assert replay_constrained_result(
        result,
        zero_rates,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        remaining_hazard_verifier=verifier,
    )


def test_boolean_zero_hazard_predicate_fails_closed() -> None:
    with pytest.raises(TypeError, match="boolean"):
        constrained_single_event_first_order(
            EditState.initial("AC", budget=2),
            lambda _state, _time: {},
            step_size=0.25,
            stability_hazard=0.05,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            seed=SEED,
            remaining_integrated_hazard_is_zero=lambda _state, _time: True,
        )


@pytest.mark.parametrize("invalid_integral", (-1.0, float("nan"), float("inf")))
def test_invalid_numeric_remaining_hazard_evidence_fails_closed(
    invalid_integral: float,
) -> None:
    with pytest.raises(ValueError):
        constrained_single_event_first_order(
            EditState.initial("AC", budget=2),
            lambda _state, _time: {},
            step_size=0.25,
            stability_hazard=0.05,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            seed=SEED,
            remaining_hazard_verifier=lambda _state, _time: invalid_integral,
        )


def test_future_positive_hazard_cannot_receive_zero_remaining_certificate() -> None:
    def future_positive(state: EditState, time: float):
        rate = 0.0 if time == 0.0 else time * time
        return {AtomicAction(ActionType.STOP): rate}

    certificate = _remaining_certificate(
        future_positive, EditState.initial("AC", budget=2), 0.0
    )
    assert certificate.integral == pytest.approx(1.0 / 3.0, rel=1.0e-12)
    assert not certificate.verified_zero
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        future_positive,
        step_size=0.25,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        remaining_hazard_verifier=lambda state, time: _remaining_certificate(
            future_positive, state, time
        ),
    )
    assert result.steps[0].t_end == 0.25
    assert result.final_state.termination_reason != (
        TerminationReason.FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
    )


def test_zero_integral_termination_serializes_certificate_without_pseudo_step() -> None:
    zero_rates = lambda _state, _time: {}
    verifier = lambda state, time: _remaining_certificate(zero_rates, state, time)
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        zero_rates,
        step_size=0.25,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        remaining_hazard_verifier=verifier,
    )
    record = sampler_result_to_schema_record(
        result,
        zero_rates,
        trajectory_id="zero-integral",
        source_id="source",
        remaining_hazard_verifier=verifier,
    )
    assert record["steps"] == []
    assert record["remaining_hazard_certificate"]["higher_order"] == 128
    assert record["termination"]["reason"] == (
        "FORCED_ZERO_REMAINING_INTEGRATED_HAZARD"
    )
    assert record["replay"]["status"] == "PASS"
    validate_schema_facing_record(record, "trajectory")


def test_unverified_zero_hazard_reaches_horizon_instead_of_early_termination() -> None:
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        lambda _state, _time: {},
        step_size=0.2,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    assert result.steps[-1].t_end == pytest.approx(1.0)
    assert all(step.outcome == "NO_EVENT" for step in result.steps)
    assert (
        result.final_state.termination_reason == TerminationReason.FORCED_TIME_HORIZON
    )


def test_adaptive_subdivision_bounds_hazard_product_and_event_probability() -> None:
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=2),
        _stop_only_rate(100.0),
        step_size=0.5,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    for step in result.steps:
        if step.total_hazard <= 0.0:
            continue
        assert step.h * step.total_hazard <= 0.05 + 1.0e-15
        assert math.isclose(
            step.event_probability,
            -math.expm1(-step.h * step.total_hazard),
            abs_tol=FLOAT64_ATOL,
            rel_tol=FLOAT64_RTOL,
        )
        assert 0.0 <= step.event_probability <= 1.0


def test_primary_sampler_hard_validity_budget_and_replay_are_100_percent() -> None:
    for seed in range(SEED, SEED + 256):
        initial = EditState.initial("ACG", budget=3)
        result = constrained_single_event_first_order(
            initial,
            _factorized_rate_fn,
            step_size=0.05,
            stability_hazard=0.05,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            seed=seed,
        )
        assert result.sampler == "constrained_single_event_first_order"
        assert result.exact_gillespie is False
        assert MIN_LENGTH <= len(result.final_state.current) <= MAX_LENGTH
        assert result.final_state.history.executed <= initial.initial_budget
        assert result.final_state.remaining_budget == (
            initial.initial_budget - result.final_state.history.executed
        )
        assert result.edit_events == result.final_state.history.executed
        assert result.final_state.phase == Phase.HALTED
        assert result.final_state.termination_reason in set(TerminationReason)
        assert all(
            step.selected_action is None
            or step.selected_action
            in enumerate_legal_actions(
                # Replay validity is fully checked by state hashes below; this
                # membership check covers the initial action-domain contract.
                initial,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
            or step.step > 0
            for step in result.steps
        )
        assert replay_constrained_result(
            result,
            _factorized_rate_fn,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )


def test_primary_sampler_is_deterministic_for_same_seed_and_inputs() -> None:
    kwargs = dict(
        step_size=0.03125,
        stability_hazard=0.05,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    first = constrained_single_event_first_order(
        EditState.initial("ACGU", budget=4), _factorized_rate_fn, **kwargs
    )
    second = constrained_single_event_first_order(
        EditState.initial("ACGU", budget=4), _factorized_rate_fn, **kwargs
    )
    assert first == second
    assert replay_constrained_result(
        first,
        _factorized_rate_fn,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )


def test_replay_recomputes_rng_rates_hazards_probabilities_actions_and_times() -> None:
    rates = _stop_only_rate(100.0)
    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=1),
        rates,
        step_size=0.1,
        stability_hazard=100.0,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    assert result.steps and result.steps[0].selected_action == AtomicAction(
        ActionType.STOP
    )
    assert replay_constrained_result(result, rates)
    first = result.steps[0]
    corrupt_logs = (
        replace(first, total_hazard=first.total_hazard + 1.0),
        replace(first, event_probability=first.event_probability / 2.0),
        replace(first, event_draw=float(first.event_draw) / 2.0),
        replace(first, action_draw=float(first.action_draw) / 2.0),
        replace(first, candidate_actions_hash="0" * 64),
        replace(first, candidate_rates_hash="1" * 64),
        replace(first, h=first.h / 2.0),
        replace(first, t_end=first.t_end / 2.0),
        replace(first, adaptive_subdivision_count=first.adaptive_subdivision_count + 1),
        replace(first, outcome="NO_EVENT"),
        replace(first, after_hash="2" * 64),
    )
    for corrupt in corrupt_logs:
        assert not replay_constrained_result(
            replace(result, steps=(corrupt,) + result.steps[1:]), rates
        )
    assert not replay_constrained_result(replace(result, seed=result.seed + 1), rates)
    assert not replay_constrained_result(result, _stop_only_rate(99.0))


def test_forced_budget_is_separate_from_schema_compatible_edit_step() -> None:
    def sub_only(state: EditState, _time: float):
        return {AtomicAction(ActionType.SUB, 0, "G"): 100.0}

    result = constrained_single_event_first_order(
        EditState.initial("AC", budget=1),
        sub_only,
        step_size=0.1,
        stability_hazard=100.0,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    assert result.steps[-1].outcome == "SUB"
    assert result.steps[-1].after_hash == result.termination_before_hash
    assert result.final_state.termination_reason == TerminationReason.FORCED_BUDGET
    assert all(step.h > 0.0 for step in result.steps)
    record = sampler_result_to_schema_record(
        result,
        sub_only,
        trajectory_id="forced-budget",
        source_id="source",
    )
    assert record["steps"][-1]["outcome"] == "SUB"
    assert record["termination"]["reason"] == "FORCED_BUDGET"
    assert record["replay"]["status"] == "PASS"
    validate_schema_facing_record(record, "trajectory")


def test_paper_parallel_validity_is_reported_separately_without_repair_claim() -> None:
    def high_rates(state: EditState, _time: float):
        return {
            action: 100.0
            for action in enumerate_legal_actions(
                state,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
            )
        }

    result = paper_first_order_parallel(
        EditState.initial("ACG", budget=2),
        high_rates,
        step_size=0.1,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        horizon=0.2,
    )
    assert result.sampler == "paper_first_order_parallel"
    assert result.exact_gillespie is False
    assert result.invalid_joint_proposals > 0
    assert any(
        step.outcome == "INVALID_JOINT_PROPOSAL_REPORTED" for step in result.steps
    )
    assert replay_paper_result(result, high_rates)


def test_del_and_sub_use_simultaneous_pre_step_coordinates() -> None:
    def staged_rates(state: EditState, time: float):
        if time == 0.0:
            return {AtomicAction(ActionType.SUB, 0, "G"): 1.0e6}
        # Both actions are defined against pre-step "GC".  Deleting original
        # token 0 must not retarget the substitution of original token 1.
        return {
            AtomicAction(ActionType.DEL, 0): 1.0e6,
            AtomicAction(ActionType.SUB, 1, "A"): 1.0e6,
        }

    result = paper_first_order_parallel(
        EditState.initial("AC", budget=3),
        staged_rates,
        step_size=0.5,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        horizon=1.0,
    )
    assert result.steps[-1].outcome == "PARALLEL_EVENTS_APPLIED"
    assert result.final_state.current == "A"
    assert result.invalid_joint_proposals == 0
    assert replay_paper_result(result, staged_rates)


@pytest.mark.parametrize(
    ("source", "actions", "expected"),
    (
        (
            "AC",
            (AtomicAction(ActionType.DEL, 0), AtomicAction(ActionType.SUB, 1, "A")),
            "A",
        ),
        (
            "AC",
            (AtomicAction(ActionType.SUB, 0, "G"), AtomicAction(ActionType.DEL, 1)),
            "G",
        ),
        (
            "ACG",
            (AtomicAction(ActionType.DEL, 1), AtomicAction(ActionType.INS, 0, "U")),
            "UAG",
        ),
        (
            "AC",
            (AtomicAction(ActionType.DEL, 0), AtomicAction(ActionType.INS, 1, "U")),
            "UC",
        ),
        (
            "AC",
            (AtomicAction(ActionType.DEL, 1), AtomicAction(ActionType.INS, 1, "U")),
            "AU",
        ),
        (
            "ACG",
            (
                AtomicAction(ActionType.SUB, 1, "U"),
                AtomicAction(ActionType.INS, 2, "A"),
                AtomicAction(ActionType.DEL, 0),
            ),
            "UAG",
        ),
        (
            "AC",
            (
                AtomicAction(ActionType.INS, 0, "G"),
                AtomicAction(ActionType.INS, 2, "U"),
            ),
            "GACU",
        ),
        (
            "AC",
            (
                AtomicAction(ActionType.DEL, 0),
                AtomicAction(ActionType.DEL, 1),
                AtomicAction(ActionType.INS, 1, "U"),
            ),
            "U",
        ),
    ),
)
def test_parallel_mixed_edits_have_frozen_pre_step_coordinate_semantics(
    source: str,
    actions: tuple[AtomicAction, ...],
    expected: str,
) -> None:
    def selected_rates(_state: EditState, _time: float):
        return {action: 1.0e6 for action in actions}

    result = paper_first_order_parallel(
        EditState.initial(source, budget=len(actions)),
        selected_rates,
        step_size=0.25,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        horizon=0.25,
    )
    assert result.steps[0].outcome == "PARALLEL_EVENTS_APPLIED"
    assert result.final_state.current == expected
    assert result.final_state.history.executed == len(actions)
    assert replay_paper_result(result, selected_rates)


def _reference_parallel_sequence(
    source: str, actions: tuple[AtomicAction, ...]
) -> str | None:
    token_positions: set[int] = set()
    gap_positions: set[int] = set()
    substitutions: dict[int, str] = {}
    deletions: set[int] = set()
    insertions: dict[int, str] = {}
    for action in actions:
        position = int(action.position)
        occupied = gap_positions if action.kind == ActionType.INS else token_positions
        if position in occupied:
            return None
        occupied.add(position)
        if action.kind == ActionType.INS:
            insertions[position] = str(action.token)
        elif action.kind == ActionType.SUB:
            substitutions[position] = str(action.token)
        else:
            deletions.add(position)
    output: list[str] = []
    if 0 in insertions:
        output.append(insertions[0])
    for index, token in enumerate(source):
        if index not in deletions:
            output.append(substitutions.get(index, token))
        if index + 1 in insertions:
            output.append(insertions[index + 1])
    return "".join(output)


def test_parallel_coordinate_transform_is_exhaustive_for_all_legal_pairs_and_triples() -> (
    None
):
    source = "ACG"
    initial = EditState.initial(source, budget=3)
    actions = tuple(
        action
        for action in enumerate_legal_actions(
            initial, min_length=MIN_LENGTH, max_length=MAX_LENGTH
        )
        if action.kind != ActionType.STOP
    )
    checked = 0
    for size in (2, 3):
        for combination in itertools.combinations(actions, size):
            expected = _reference_parallel_sequence(source, combination)
            if expected is None or not MIN_LENGTH <= len(expected) <= MAX_LENGTH:
                continue

            def selected_rates(_state: EditState, _time: float, selected=combination):
                return {action: 1.0e6 for action in selected}

            result = paper_first_order_parallel(
                initial,
                selected_rates,
                step_size=0.25,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                seed=SEED,
                horizon=0.25,
            )
            assert result.steps[0].outcome == "PARALLEL_EVENTS_APPLIED"
            assert result.final_state.current == expected
            assert replay_paper_result(result, selected_rates)
            checked += 1
    assert checked > 2_500


def test_conflicting_parallel_proposal_is_reported_without_partial_mutation() -> None:
    actions = (
        AtomicAction(ActionType.SUB, 0, "G"),
        AtomicAction(ActionType.DEL, 0),
    )

    def conflicting(_state: EditState, _time: float):
        return {action: 1.0e6 for action in actions}

    result = paper_first_order_parallel(
        EditState.initial("AC", budget=2),
        conflicting,
        step_size=0.25,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        horizon=0.25,
    )
    step = result.steps[0]
    assert step.outcome == "INVALID_JOINT_PROPOSAL_REPORTED"
    assert step.before_hash == step.after_hash
    assert result.edit_events == 0
    assert result.final_state.current == "AC"
    assert replay_paper_result(result, conflicting)


def test_paper_replay_and_schema_reject_corrupt_parallel_random_ledger() -> None:
    actions = (
        AtomicAction(ActionType.DEL, 0),
        AtomicAction(ActionType.SUB, 1, "A"),
    )

    def selected_rates(_state: EditState, _time: float):
        return {action: 1.0e6 for action in actions}

    result = paper_first_order_parallel(
        EditState.initial("AC", budget=2),
        selected_rates,
        step_size=0.25,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        seed=SEED,
        horizon=0.25,
    )
    assert replay_paper_result(result, selected_rates)
    first = result.steps[0]
    corrupted_draws = list(first.parallel_draws)
    corrupted_draws[0] = (corrupted_draws[0][0], corrupted_draws[0][1] / 2.0)
    corrupt = replace(first, parallel_draws=tuple(corrupted_draws))
    assert not replay_paper_result(replace(result, steps=(corrupt,)), selected_rates)
    assert not replay_paper_result(result, lambda _state, _time: {actions[0]: 1.0})
    record = sampler_result_to_schema_record(
        result,
        selected_rates,
        trajectory_id="paper-mixed",
        source_id="source",
    )
    assert record["steps"][0]["parallel_trials"]
    assert record["steps"][0]["parallel_actions"]
    assert record["replay"]["status"] == "PASS"
    validate_schema_facing_record(record, "trajectory")


def test_step_halving_constant_hazard_converges_to_integrated_hazard_reference() -> (
    None
):
    hazard = 1.2
    exact_event_fraction = 1.0 - math.exp(-hazard)
    fractions = []
    for step_size in (0.125, 0.0625, 0.03125):
        events = 0
        sample_count = 4_096
        for offset in range(sample_count):
            result = constrained_single_event_first_order(
                EditState.initial("AC", budget=1),
                _stop_only_rate(hazard),
                step_size=step_size,
                stability_hazard=0.05,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                seed=SEED + offset,
            )
            events += int(
                result.final_state.termination_reason == TerminationReason.LEARNED_STOP
            )
        fraction = events / sample_count
        fractions.append(fraction)
        assert abs(fraction - exact_event_fraction) < 0.03
    assert abs(fractions[-1] - fractions[-2]) < 0.03


def test_sampler_rejects_nonzero_rate_for_hard_masked_action() -> None:
    state = EditState.initial("A", budget=1)

    def illegal_rate(_state: EditState, _time: float):
        return {AtomicAction(ActionType.DEL, 0): 1.0}

    with pytest.raises(ValueError):
        constrained_single_event_first_order(
            state,
            illegal_rate,
            step_size=0.1,
            stability_hazard=0.05,
            min_length=1,
            max_length=4,
            seed=SEED,
        )
