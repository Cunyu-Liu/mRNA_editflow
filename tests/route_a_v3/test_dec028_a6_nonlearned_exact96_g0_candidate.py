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
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec028_a6_nonlearned_exact96_g0_candidate_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec028_a6_nonlearned_exact96_g0_candidate.py"
SPEC = importlib.util.spec_from_file_location("dec028_a6_nonlearned_exact96_g0_candidate", SCRIPT_PATH)
assert SPEC and SPEC.loader
A6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A6
SPEC.loader.exec_module(A6)


def config() -> dict[str, Any]:
    return A6.load_config(CONFIG_PATH)


def test_disk_candidate_runs_exact96_in_memory_and_does_not_claim_general_time_or_a6_pass(capsys: pytest.CaptureFixture[str]) -> None:
    candidate = config()
    assert candidate["authority_status"] == "NON_AUTHORITATIVE_G0_NONLEARNED_SYNTHETIC_PREPARATION"
    assert candidate["synthetic_fixture_generator"]["expected_fixture_count"] == 96
    assert candidate["reference_contract"]["general_time_inhomogeneous_exactness"] == "NOT_ESTABLISHED_NOT_CLAIMED"
    assert A6.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "G0_NONLEARNED_SYNTHETIC_EXACT96_REFERENCE_PASS_NOT_A6_PASS"
    assert report["fixture_count"] == 96
    assert report["budgets"] == [1, 3, 5]
    assert report["source_lengths"] == [2, 3, 4, 5]
    assert report["aggregate_state_count"] > 0
    assert report["max_terminal_distribution_absolute_error"] <= 1e-12
    assert report["time_scope"] == "SYNTHETIC_TIME_HOMOGENEOUS_JUMP_CHAIN_EXACT_REFERENCE_ONLY"
    assert report["general_time_inhomogeneous_exactness"] == "NOT_ESTABLISHED_NOT_CLAIMED"
    assert report["a6_evidence_status"] == "IN_PROGRESS"
    assert report["l3_claim_status"] == "NOT_ESTABLISHED"
    assert all(value == 0 or value is False for value in report["runtime_truth"].values())


def test_generated_fixture_closure_is_exact96_and_all_cases_match_dp_with_enumeration() -> None:
    candidate = config()
    cases = A6.generate_cases(candidate)
    assert len(cases) == 96
    assert len({case.case_id for case in cases}) == 96
    assert {case.budget for case in cases} == {1, 3, 5}
    assert {len(case.source) for case in cases} == {2, 3, 4, 5}
    for case in cases:
        graph = A6.build_graph(case, candidate)
        A6.validate_graph_invariants(graph, candidate)
        dp = A6.terminal_distribution_dp(graph)
        enumeration = A6.terminal_distribution_enumeration(graph)
        assert set(dp) == set(enumeration)
        assert sum(dp.values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(enumeration.values()) == pytest.approx(1.0, abs=1e-12)
        for state in dp:
            assert dp[state] == pytest.approx(enumeration[state], abs=1e-12)


def test_raw_aliases_are_aggregated_before_normalization_and_stop_is_competing_positive_rate() -> None:
    candidate = config()
    case = next(case for case in A6.generate_cases(candidate) if len(case.source) == 5 and case.budget == 5)
    graph = A6.build_graph(case, candidate)
    root = graph.root
    transitions = graph.outgoing[root]
    assert any(item.next_state.terminal_cause == "EXPLICIT_STOP" for item in transitions)
    assert len(transitions) == len(case.source) + 1
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
    with pytest.raises(A6.CandidateError, match="edited base"):
        A6.validate_state(malformed, candidate)
    with pytest.raises(A6.CandidateError, match="illegal"):
        A6.apply_raw_action(root, A6.RawAction(A6.EDIT, "ALIAS_PRIMARY", len(root.source)), candidate)


def test_config_rejects_authority_fixture_time_scope_and_runtime_lock_drift() -> None:
    candidate = config()
    for keys, value in (
        (("static_authority", "current_qualified_counts", "a1"), 2),
        (("static_authority", "effective_active_amendment_decision_ids"), list(A6.BEFORE_DECISION_IDS) + ["V3-DEC-028"]),
        (("synthetic_fixture_generator", "expected_fixture_count"), 95),
        (("reference_contract", "general_time_inhomogeneous_exactness"), "PASS"),
        (("runtime_truth", "cuda_probe_calls"), 1),
        (("runtime_truth", "a6_pass_asserted"), True),
    ):
        altered = copy.deepcopy(candidate)
        cursor: dict[str, Any] = altered
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        with pytest.raises(A6.CandidateError):
            A6.validate_config(altered)


def test_source_has_only_static_standard_library_dependencies() -> None:
    module = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for statement in module.body:
        if isinstance(statement, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            imported.add(statement.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "copy", "hashlib", "json", "math", "sys", "dataclasses", "pathlib", "typing"}
