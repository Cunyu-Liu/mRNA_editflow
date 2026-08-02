"""Fault injections for CPU acceptance domains that previously over-declared PASS."""

from __future__ import annotations

import hashlib
import random

import pytest
import yaml

from mrna_editflow.core.mk0.types import EditState
from scripts.mk0 import run_mk0_cpu_acceptance as CPU


REPEATED_CASES = (
    ("AA", "A"),
    ("A", "AA"),
    ("AAA", "AA"),
    ("AA", "AAA"),
    ("ACA", "AAC"),
)


def _gate(gate_id: str):
    config = yaml.safe_load(
        (CPU.REPO_ROOT / "configs" / "math" / "math_kernel_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    return next(gate for gate in config["acceptance"]["gates"] if gate["id"] == gate_id)


def test_repaired_gate_domains_and_tolerances_are_frozen() -> None:
    assert (
        _gate("M09")["sample_count"],
        _gate("M09")["atol"],
        _gate("M09")["rtol"],
    ) == (
        2006,
        1.0e-10,
        1.0e-8,
    )
    assert _gate("M12")["sample_count"] == 64
    assert _gate("M13")["sample_count"] == 64
    assert _gate("M14")["sample_count"] == 1176
    assert _gate("M17")["sample_count"] == 320
    assert (
        _gate("M19")["sample_count"],
        _gate("M19")["atol"],
        _gate("M19")["rtol"],
    ) == (
        384,
        1.0e-6,
        1.0e-5,
    )
    assert (
        _gate("M21")["sample_count"],
        _gate("M21")["atol"],
        _gate("M21")["rtol"],
    ) == (
        24576,
        1.0e-8,
        1.0e-7,
    )
    assert (
        _gate("M22")["sample_count"],
        _gate("M22")["atol"],
        _gate("M22")["rtol"],
    ) == (
        24576,
        0.02,
        0.02,
    )


def test_m09_wrong_analytic_derivative_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CPU.cubic_schedule

    def wrong_derivative(time: float):
        kappa, derivative = original(time)
        return kappa, derivative + 1.0e-3

    monkeypatch.setattr(CPU, "cubic_schedule", wrong_derivative)
    with pytest.raises(CPU.AcceptanceFailure, match="M09 finite-difference"):
        CPU._m09_schedule_endpoint_derivative_audit()


def test_m12_nonzero_cubic_t0_hazard_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CPU.rho

    def wrong_rho(time: float, *, name: str = "cubic", time_eps: float = 1.0e-4):
        if name == "cubic" and time == 0.0:
            return 1.0
        return original(time, name=name, time_eps=time_eps)

    monkeypatch.setattr(CPU, "rho", wrong_rho)
    with pytest.raises(CPU.AcceptanceFailure, match="M12 zero-instantaneous"):
        CPU._m12_zero_instantaneous_hazard_audit()


def test_m13_context_or_seed_repetition_cannot_inflate_fixture_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = CPU._m13_fixture_state("zero_integral", 0, "A")
    monkeypatch.setattr(
        CPU,
        "_m13_fixture_state",
        lambda _stratum, _index, _sequence: repeated,
    )
    with pytest.raises(CPU.AcceptanceFailure, match="distinct state/fixture domain"):
        CPU._m13_remaining_integrated_hazard_audit()


def test_m13_zeroed_future_hazard_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CPU,
        "_m13_future_rate_factory",
        lambda **_parameters: (lambda _state, _time: {}),
    )
    with pytest.raises(CPU.AcceptanceFailure, match="M13 remaining-integrated"):
        CPU._m13_remaining_integrated_hazard_audit()


def test_operational_fixture_id_ignores_decorative_context() -> None:
    first = EditState.initial(
        "AC",
        budget=2,
        context={"batch": "decorative-a"},
    )
    second = EditState.initial(
        "AC",
        budget=2,
        context={"batch": "decorative-b"},
    )
    first_id = CPU._operational_fixture_id(
        first,
        case="context_negative_control",
        attempted_action_keys=("SUB:0:C",),
        min_length=1,
        max_length=4,
    )
    second_id = CPU._operational_fixture_id(
        second,
        case="context_negative_control",
        attempted_action_keys=("SUB:0:C",),
        min_length=1,
        max_length=4,
    )
    assert first.state_hash != second.state_hash
    assert first_id == second_id


def test_invalid_action_payload_domain_is_real_and_state_independent() -> None:
    attempts = CPU._m11_invalid_action_payloads()
    payload_ids = [
        CPU._invalid_action_payload_id(payload) for _category, payload in attempts
    ]
    assert len(attempts) == len(set(payload_ids)) == 588
    for _category, payload in attempts:
        with pytest.raises((TypeError, ValueError)):
            CPU.AtomicAction(**payload)


def test_m11_duplicate_invalid_payload_domain_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active, _halted = CPU.tiny_active_halted_rate_states()
    monkeypatch.setattr(CPU, "_invalid_action_payload_id", lambda _payload: "duplicate")
    with pytest.raises(CPU.AcceptanceFailure, match="M11 operational fixture domain"):
        CPU._m11_mask_audit(active)


def test_m14_uniform_distribution_replacement_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def uniform_distribution(rates):
        probability = 1.0 / len(rates)
        return {action: probability for action in rates}

    monkeypatch.setattr(CPU, "conditioned_event_distribution", uniform_distribution)
    with pytest.raises(CPU.AcceptanceFailure, match="M14 conditioned action law"):
        CPU._m14_conditioned_distribution_audit(list(CPU.tiny_active_rate_states()))


def test_m17_dropped_transition_aggregate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CPU.aggregate_transition_rates

    def dropped_aggregate(*args, **kwargs):
        result = original(*args, **kwargs)
        if result:
            result.pop(next(iter(result)))
        return result

    monkeypatch.setattr(CPU, "aggregate_transition_rates", dropped_aggregate)
    with pytest.raises(CPU.AcceptanceFailure, match="M17 transition parameter"):
        CPU._m17_transition_parameter_audit(REPEATED_CASES)


def test_m19_biased_finite_difference_gradient_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = CPU._central_finite_difference_gradient

    def biased_gradient(*args, **kwargs):
        return [value + 1.0e-2 for value in original(*args, **kwargs)]

    monkeypatch.setattr(CPU, "_central_finite_difference_gradient", biased_gradient)
    with pytest.raises(CPU.AcceptanceFailure, match="M19 finite-difference"):
        CPU._m19_finite_difference_gradient_audit(random.Random(CPU.SEED))


def test_m21_completion_dependent_dwell_fails_structural_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completion_dependent_target(clocks, *, gamma_ref, rng):
        del rng
        completion = max(clocks.values(), default=0.0)
        dwell = (1.0 + completion) / gamma_ref
        latent = completion + dwell
        return CPU.StopTarget(
            completion,
            dwell,
            latent,
            min(latent, 1.0),
            latent < 1.0,
        )

    monkeypatch.setattr(CPU, "sample_stop_target", completion_dependent_target)
    with pytest.raises(CPU.AcceptanceFailure, match="M21 structural"):
        CPU._m21_dwell_independence_audit(random.Random(CPU.SEED))


def test_m22_one_bad_gamma_event_fraction_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gamma = {}
    for gamma in (8.0, 16.0, 32.0):
        fake_gamma[str(gamma)] = {
            "sample_count": 8192,
            "completion_dwell_pearson_correlation_diagnostic": 0.0,
            "observed_mean_dwell": 1.0 / gamma,
            "observed_event_fraction": 0.75 if gamma == 8.0 else 0.5,
            "expected_event_fraction": 0.5,
            "event_fraction_absolute_error": 0.25 if gamma == 8.0 else 0.0,
        }
    monkeypatch.setattr(
        CPU,
        "_m21_dwell_independence_audit",
        lambda _rng: {"gamma_dwell_independence": fake_gamma},
    )
    with pytest.raises(CPU.AcceptanceFailure, match="M22 gamma 8/16/32"):
        CPU.run_stop_audits(random.Random(CPU.SEED))


def test_m24_duplicate_operational_ids_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def case_only_id(_state, *, case, **_kwargs):
        return hashlib.sha256(case.encode("utf-8")).hexdigest()

    monkeypatch.setattr(CPU, "_operational_fixture_id", case_only_id)
    with pytest.raises(CPU.AcceptanceFailure, match="M24 unique operational"):
        CPU.run_stop_audits(random.Random(CPU.SEED))
