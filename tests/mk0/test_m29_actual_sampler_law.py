"""Fault-injection regression for the actual-trajectory M29 sampler gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_cpu_runner():
    path = ROOT / "scripts" / "mk0" / "run_mk0_cpu_acceptance.py"
    spec = importlib.util.spec_from_file_location("mk0_m29_cpu_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU_RUNNER = _load_cpu_runner()


@pytest.fixture(scope="module")
def actual_sampler_law_report():
    return CPU_RUNNER._run_m29_actual_sampler_law_audit()


def test_m29_binds_actual_trajectories_to_independent_action_and_law_oracles(
    actual_sampler_law_report,
) -> None:
    report = actual_sampler_law_report
    assert report["status"] == "PASS"
    assert report["trajectory_denominator"] == 2048
    assert report["stratum_count"] == 4
    assert report["trajectories_per_stratum"] == 512
    assert report["paired_seed_design"] is True
    assert report["step_kernel_check_denominator"] > 0
    assert report["step_kernel_mismatch_count"] == 0
    assert report["step_kernel_agreement_fraction"] == 1.0
    assert report["action_oracle_check_denominator"] > 0
    assert report["action_oracle_mismatch_count"] == 0
    assert report["action_oracle_agreement_fraction"] == 1.0
    assert report["trajectory_kernel_replay_mismatch_count"] == 0
    assert report["empirical_distribution_role"] == (
        "diagnostic_only_not_a_gate_condition"
    )
    assert report["empirical_distribution_can_grant_pass"] is False
    assert report["failure_count"] == 0
    for field in (
        "input_stream_sha256",
        "actual_terminal_stream_sha256",
        "actual_first_action_stream_sha256",
        "step_kernel_oracle_stream_sha256",
        "action_oracle_stream_sha256",
        "expected_law_stream_sha256",
        "audit_binding_sha256",
    ):
        assert len(report[field]) == 64
        int(report[field], 16)


def test_m29_fault_injection_biased_draw_action_must_fail(monkeypatch) -> None:
    sampler_globals = CPU_RUNNER.constrained_single_event_first_order.__globals__

    def always_first_action(distribution, _draw):
        return min(distribution, key=lambda action: action.key)

    monkeypatch.setitem(sampler_globals, "_draw_action", always_first_action)
    report = CPU_RUNNER._run_m29_actual_sampler_law_audit()

    assert report["status"] == "FAIL"
    assert report["failure_count"] > 0
    assert report["action_oracle_mismatch_count"] > 0
    assert report["action_oracle_agreement_fraction"] < 1.0
    assert report["empirical_distribution_can_grant_pass"] is False
