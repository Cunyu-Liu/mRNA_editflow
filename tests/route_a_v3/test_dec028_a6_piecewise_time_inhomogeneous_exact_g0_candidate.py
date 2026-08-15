from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py"
SPEC = importlib.util.spec_from_file_location("dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate", SCRIPT_PATH)
assert SPEC and SPEC.loader
A6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A6
SPEC.loader.exec_module(A6)


def config() -> dict[str, Any]:
    return A6.load_config(CONFIG_PATH)


def test_disk_candidate_runs_exact96_without_general_time_or_a6_claim(capsys: pytest.CaptureFixture[str]) -> None:
    candidate = config()
    assert candidate["authority_status"] == "NON_AUTHORITATIVE_G0_NONLEARNED_SYNTHETIC_TIME_INHOMOGENEOUS_PREPARATION"
    assert candidate["synthetic_fixture_generator"]["expected_fixture_count"] == 96
    assert candidate["time_inhomogeneous_reference_contract"]["contract_wide_general_time_inhomogeneous_exactness"] == "NOT_ESTABLISHED_NOT_CLAIMED"
    assert A6.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "G0_NONLEARNED_SYNTHETIC_PIECEWISE_TIME_INHOMOGENEOUS_REFERENCE_PASS_NOT_A6_PASS"
    assert report["fixture_count"] == 96
    assert report["budgets"] == [1, 3, 5]
    assert report["source_lengths"] == [2, 3, 4, 5]
    assert report["time_scope"] == "FINITE_PIECEWISE_CONSTANT_CONTINUOUS_ALGORITHMIC_TIME_SCHEDULE_WITH_HOMOGENEOUS_ABSORBING_TAIL_ONLY"
    assert report["contract_wide_general_time_inhomogeneous_exactness"] == "NOT_ESTABLISHED_NOT_CLAIMED"
    assert report["physical_kinetics_claim"] == "NOT_CLAIMED"
    assert report["a6_evidence_status"] == "IN_PROGRESS"
    assert report["l3_claim_status"] == "NOT_ESTABLISHED"
    assert report["max_uniformization_error_bound"] <= 3e-13
    assert report["max_terminal_mass_shortfall"] <= report["max_uniformization_error_bound"] + 1e-12
    assert report["max_independent_terminal_distribution_tv"] <= 1e-10
    assert all(value == 0 or value is False for value in report["runtime_truth"].values())


def test_all_96_cases_have_bounded_reference_error_and_independent_rk4_agreement() -> None:
    candidate = config()
    cases = A6.generate_cases(candidate)
    assert len(cases) == 96
    assert {case.budget for case in cases} == {1, 3, 5}
    assert {len(case.source) for case in cases} == {2, 3, 4, 5}
    outcomes = [A6.run_case(case, candidate) for case in cases]
    assert all(outcome.state_count > 0 for outcome in outcomes)
    assert all(outcome.maximum_exit_rate > 0.0 for outcome in outcomes)
    assert all(outcome.uniformization_error_bound <= 3e-13 for outcome in outcomes)
    assert all(outcome.terminal_mass_shortfall <= outcome.uniformization_error_bound + 1e-12 for outcome in outcomes)
    assert all(outcome.terminal_distribution_tv <= 1e-10 for outcome in outcomes)
    assert all(outcome.rk4_refinement_steps >= 8 for outcome in outcomes)


def test_schedule_changes_relative_action_probabilities_not_only_holding_time() -> None:
    candidate = config()
    case = next(case for case in A6.generate_cases(candidate) if len(case.source) == 5 and case.budget == 5 and case.variant == 3)
    root = A6.initial_state(case)
    reference = candidate["time_inhomogeneous_reference_contract"]
    early = A6.normalized_transition_probabilities(A6.canonical_transitions(root, case, candidate, reference["finite_segments"][0]))
    middle = A6.normalized_transition_probabilities(A6.canonical_transitions(root, case, candidate, reference["finite_segments"][1]))
    assert early != middle
    early_stop = next(probability for state, probability in early.items() if state.terminal_cause == "EXPLICIT_STOP")
    middle_stop = next(probability for state, probability in middle.items() if state.terminal_cause == "EXPLICIT_STOP")
    assert early_stop > middle_stop


def test_raw_aliases_aggregate_before_normalization_and_stop_remains_positive() -> None:
    candidate = config()
    case = next(case for case in A6.generate_cases(candidate) if len(case.source) == 5 and case.budget == 5)
    root = A6.initial_state(case)
    schedule = candidate["time_inhomogeneous_reference_contract"]["finite_segments"][0]
    transitions = A6.canonical_transitions(root, case, candidate, schedule)
    assert len(transitions) == len(case.source) + 1
    assert any(item.next_state.terminal_cause == "EXPLICIT_STOP" for item in transitions)
    for transition in transitions:
        assert transition.raw_alias_ids == ("ALIAS_PRIMARY", "ALIAS_SECONDARY")
        assert transition.rate == pytest.approx(transition.raw_rate_sum, abs=1e-15)
        assert transition.rate > candidate["transition_contract"]["support_floor"]


def test_source_anchoring_repeated_edit_and_illegal_raw_actions_are_rejected() -> None:
    candidate = config()
    case = next(case for case in A6.generate_cases(candidate) if case.budget == 3)
    root = A6.initial_state(case)
    first_edit = next(action for action in A6.legal_raw_actions(root, candidate) if action.action_type == A6.EDIT)
    edited = A6.apply_raw_action(root, first_edit, candidate)
    with pytest.raises(A6.CandidateError, match="illegal"):
        A6.apply_raw_action(edited, first_edit, candidate)
    malformed = A6.State(root.source, root.source, (0,), root.remaining_budget)
    with pytest.raises(A6.CandidateError, match="source-relative legal"):
        A6.validate_state(malformed, candidate)
    with pytest.raises(A6.CandidateError, match="illegal"):
        A6.apply_raw_action(root, A6.RawAction(A6.EDIT, "ALIAS_PRIMARY", len(root.source)), candidate)


def test_config_rejects_authority_schedule_scope_and_runtime_lock_drift() -> None:
    candidate = config()
    mutations = (
        (("static_authority", "current_qualified_counts", "a1"), 2),
        (("static_authority", "effective_active_amendment_decision_ids"), list(A6.BEFORE_DECISION_IDS) + ["V3-DEC-028"]),
        (("synthetic_fixture_generator", "expected_fixture_count"), 95),
        (("time_inhomogeneous_reference_contract", "finite_segments", 1, "stop_multiplier"), 1.45),
        (("time_inhomogeneous_reference_contract", "contract_wide_general_time_inhomogeneous_exactness"), "PASS"),
        (("runtime_truth", "cuda_probe_calls"), 1),
        (("runtime_truth", "a6_pass_asserted"), True),
    )
    for keys, value in mutations:
        drifted = copy.deepcopy(candidate)
        target: Any = drifted
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        with pytest.raises(A6.CandidateError):
            A6.validate_config(drifted)


def test_module_has_no_model_device_or_dataset_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"torch", "tensorflow", "jax", "numpy", "pandas", "subprocess"}
