"""MK0-03/04 alignment, product path, switch-clock and schedule oracles."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from mrna_editflow.core.mk0.alignment_coupling import (
    BLANK,
    alignment_actions,
    build_alignment,
    changed_indices,
    coupling_manifest_record,
    joint_path_probability,
    reconstruct_alignment,
    remaining_switches,
    sample_optimal_alignment,
    sample_switch_clocks,
    switched_alignment_state,
)
from mrna_editflow.core.mk0.schedule import (
    cubic_schedule,
    evaluate_schedule,
    linear_schedule,
    rho,
)
from mrna_editflow.core.mk0.state_action import replay_actions
from mrna_editflow.core.mk0.types import EditState

from .conftest import FLOAT64_ATOL, FLOAT64_RTOL, SEED


def _tiny_ac_sequences() -> tuple[str, ...]:
    return tuple(
        "".join(tokens)
        for length in (1, 2, 3)
        for tokens in itertools.product("AC", repeat=length)
    )


def _levenshtein_distance(source: str, target: str) -> int:
    previous = list(range(len(target) + 1))
    for i, source_token in enumerate(source, start=1):
        current = [i]
        for j, target_token in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (source_token != target_token),
                )
            )
        previous = current
    return previous[-1]


def _remove_blanks(tokens) -> str:
    return "".join(token for token in tokens if token != BLANK)


def test_exhaustive_196_tiny_alignments_reconstruct_and_are_optimal() -> None:
    sequences = _tiny_ac_sequences()
    assert len(sequences) == 14
    observed_hashes: dict[tuple[str, str], str] = {}
    for source, target in itertools.product(sequences, repeat=2):
        alignment = build_alignment(source, target)
        assert reconstruct_alignment(alignment) == (source, target)
        assert alignment.cost == _levenshtein_distance(source, target)
        assert alignment.path_is_observed is False
        assert alignment.path_semantics == "latent_algorithmic"
        assert len(alignment.alignment_hash) == 64
        assert (
            alignment.alignment_hash == build_alignment(source, target).alignment_hash
        )
        observed_hashes[(source, target)] = alignment.alignment_hash
    assert len(observed_hashes) == 196


def test_alignment_actions_replay_source_to_target_for_all_tiny_pairs() -> None:
    for source, target in itertools.product(_tiny_ac_sequences(), repeat=2):
        alignment = build_alignment(source, target)
        actions = alignment_actions(alignment)
        initial = EditState.initial(source, budget=len(actions))
        final, _ = replay_actions(
            initial,
            actions,
            min_length=0,
            max_length=6,
        )
        assert final.current == target
        assert final.history.executed == alignment.cost == len(actions)
        assert final.remaining_budget == 0


@pytest.mark.parametrize(
    ("source", "target"),
    (("AA", "A"), ("A", "AA"), ("AAA", "AA"), ("AA", "AAA"), ("ACA", "AAC")),
)
def test_sampled_optimal_alignment_sensitivity_stays_optimal(
    source: str, target: str
) -> None:
    rng = random.Random(SEED)
    hashes = set()
    for _ in range(512):
        alignment = sample_optimal_alignment(source, target, rng=rng)
        assert reconstruct_alignment(alignment) == (source, target)
        assert alignment.cost == _levenshtein_distance(source, target)
        assert alignment.coupling_type == "sampled_optimal_sensitivity"
        assert alignment.path_is_observed is False
        hashes.add(alignment.alignment_hash)
    # Ambiguous repeated-symbol indels must expose more than one latent path.
    if source != "ACA":
        assert len(hashes) > 1


def test_coupling_manifest_never_calls_constructed_path_observed() -> None:
    record = coupling_manifest_record(
        build_alignment("AA", "A"), source_id="source", target_id="target"
    )
    assert record["path_is_observed"] is False
    assert record["path_semantics"] == "latent_algorithmic"
    assert record["coupling_type"] == "canonical_optimal"
    assert record["alignment_cost"] == 1


@pytest.mark.parametrize(("source", "target"), (("AT", "AC"), ("AC", "N"), ("X", "U")))
def test_alignment_rejects_non_rna_alphabet(source: str, target: str) -> None:
    with pytest.raises(ValueError):
        build_alignment(source, target)


@pytest.mark.parametrize("schedule", ("cubic", "linear"))
@pytest.mark.parametrize("t", (0.0, 0.125, 0.5, 0.875, 1.0))
def test_joint_path_is_normalized_product_with_correct_time_direction(
    schedule: str, t: float
) -> None:
    alignment = build_alignment("ACA", "AAC")
    coordinate_supports = []
    for column in alignment.columns:
        coordinate_supports.append(tuple({column.source_token, column.target_token}))
    probability_sum = math.fsum(
        joint_path_probability(alignment, z, t, schedule=schedule)
        for z in itertools.product(*coordinate_supports)
    )
    assert math.isclose(
        probability_sum, 1.0, abs_tol=FLOAT64_ATOL, rel_tol=FLOAT64_RTOL
    )

    source_state = tuple(column.source_token for column in alignment.columns)
    target_state = tuple(column.target_token for column in alignment.columns)
    if t == 0.0:
        assert (
            joint_path_probability(alignment, source_state, t, schedule=schedule) == 1.0
        )
    if t == 1.0:
        assert (
            joint_path_probability(alignment, target_state, t, schedule=schedule) == 1.0
        )


def test_independent_switch_clock_empirical_joint_matches_product_path() -> None:
    alignment = build_alignment("AAA", "CCCA")
    changed = changed_indices(alignment)
    assert len(changed) >= 3
    rng = random.Random(SEED)
    sample_count = 20_000
    t = 0.6
    switched = np.zeros((sample_count, len(changed)), dtype=np.float64)
    for row in range(sample_count):
        clocks = sample_switch_clocks(alignment, rng=rng, schedule="cubic")
        switched[row] = [float(clocks[index] <= t) for index in changed]
    expected = t**3
    assert np.max(np.abs(switched.mean(axis=0) - expected)) < 0.015
    covariance = np.cov(switched, rowvar=False, ddof=0)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.max(np.abs(off_diagonal)) < 0.01
    empirical_all = float(np.all(switched == 1.0, axis=1).mean())
    assert abs(empirical_all - expected ** len(changed)) < 0.015


def test_remaining_switches_and_switched_state_follow_clocks_exactly() -> None:
    alignment = build_alignment("AC", "CGA")
    changed = changed_indices(alignment)
    clocks = {
        index: (rank + 1) / (len(changed) + 1) for rank, index in enumerate(changed)
    }
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        remaining = remaining_switches(alignment, clocks, t)
        assert remaining == tuple(index for index in changed if clocks[index] > t)
        state = switched_alignment_state(alignment, clocks, t)
        expected = tuple(
            (
                column.target_token
                if index in clocks and clocks[index] <= t
                else column.source_token
            )
            for index, column in enumerate(alignment.columns)
        )
        assert state == expected
    assert _remove_blanks(switched_alignment_state(alignment, clocks, 0.0)) == "AC"
    assert _remove_blanks(switched_alignment_state(alignment, clocks, 1.0)) == "CGA"


@pytest.mark.parametrize("t", np.linspace(0.0, 0.95, 20))
def test_cubic_and_linear_schedule_formula_derivative_and_rho(t: float) -> None:
    cubic_kappa, cubic_derivative = cubic_schedule(float(t))
    linear_kappa, linear_derivative = linear_schedule(float(t))
    assert math.isclose(cubic_kappa, t**3, abs_tol=FLOAT64_ATOL)
    assert math.isclose(cubic_derivative, 3.0 * t**2, abs_tol=FLOAT64_ATOL)
    assert math.isclose(linear_kappa, t, abs_tol=FLOAT64_ATOL)
    assert linear_derivative == 1.0
    assert math.isclose(
        rho(float(t), name="cubic"),
        cubic_derivative / (1.0 - cubic_kappa),
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )
    assert math.isclose(
        rho(float(t), name="linear"),
        linear_derivative / (1.0 - linear_kappa),
        abs_tol=FLOAT64_ATOL,
        rel_tol=FLOAT64_RTOL,
    )
    if 1.0e-5 < t < 0.95:
        epsilon = 1.0e-6
        numerical = (
            cubic_schedule(float(t + epsilon))[0]
            - cubic_schedule(float(t - epsilon))[0]
        ) / (2.0 * epsilon)
        assert math.isclose(
            numerical,
            cubic_derivative,
            abs_tol=1.0e-8,
            rel_tol=1.0e-7,
        )


def test_endpoint_is_clipped_and_auditable_without_silent_singularity() -> None:
    value = evaluate_schedule(1.0, name="cubic", time_eps=1.0e-4)
    assert value.t_requested == 1.0
    assert value.t_evaluated == 0.9999
    assert value.endpoint_clipped is True
    assert math.isfinite(value.rho) and value.rho > 0.0
    unclipped = evaluate_schedule(0.5, name="cubic", time_eps=1.0e-4)
    assert unclipped.t_evaluated == 0.5
    assert unclipped.endpoint_clipped is False


def test_unknown_schedule_never_silently_falls_back_to_linear() -> None:
    alignment = build_alignment("AC", "CA")
    with pytest.raises(ValueError):
        joint_path_probability(
            alignment,
            tuple(column.source_token for column in alignment.columns),
            0.5,
            schedule="unregistered_schedule",
        )
    with pytest.raises(ValueError):
        sample_switch_clocks(
            alignment, rng=random.Random(SEED), schedule="unregistered_schedule"
        )


@pytest.mark.parametrize("bad_time", (-1.0, 1.1, math.nan, math.inf))
def test_schedule_rejects_invalid_external_time(bad_time: float) -> None:
    with pytest.raises(ValueError):
        evaluate_schedule(bad_time)
