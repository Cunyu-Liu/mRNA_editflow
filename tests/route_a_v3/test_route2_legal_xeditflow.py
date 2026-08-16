from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "core/route2_legal_xeditflow.py"
SPEC = importlib.util.spec_from_file_location("route2_legal_xeditflow_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FLOW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FLOW
SPEC.loader.exec_module(FLOW)

STOP = FLOW.STOP
FlowState = FLOW.FlowState
LegalAction = FLOW.LegalAction
LegalFlowError = FLOW.LegalFlowError
apply_action = FLOW.apply_action
exact_terminal_distribution = FLOW.exact_terminal_distribution
initial_state = FLOW.initial_state
legal_actions = FLOW.legal_actions
positive_rates = FLOW.positive_rates
replay_source_relative = FLOW.replay_source_relative
sample_trajectory = FLOW.sample_trajectory
numerical_failure_state = FLOW.numerical_failure_state


def _unit_rates(state, actions):
    return {action: 1.0 for action in actions}


def test_source_anchored_sub_stop_and_no_reedit_or_revert() -> None:
    root = initial_state("AC", budget=2, assay_id="a", context_id="c")
    assert len(legal_actions(root)) == 7
    first = apply_action(root, LegalAction("SUB", 0, "G"))
    assert first.current_sequence == "GC"
    assert first.source_relative_edits == ((0, "G"),)
    ids = {action.action_id for action in legal_actions(first)}
    assert ids == {STOP, "SUB:1:A", "SUB:1:G", "SUB:1:U"}
    with pytest.raises(LegalFlowError, match="illegal action"):
        apply_action(first, LegalAction("SUB", 0, "A"))
    with pytest.raises(LegalFlowError, match="edited twice"):
        replay_source_relative("AC", ((0, "G"), (0, "U")))


def test_terminal_causes_are_absorbing() -> None:
    zero = initial_state("A", budget=0, assay_id="a", context_id="c")
    assert zero.terminal_cause == "BUDGET_EXHAUSTED"
    assert legal_actions(zero) == ()
    root = initial_state("A", budget=2, assay_id="a", context_id="c")
    stopped = apply_action(root, LegalAction(STOP))
    assert stopped.terminal_cause == "EXPLICIT_STOP"
    edited = apply_action(root, LegalAction("SUB", 0, "C"))
    assert edited.terminal_cause == "NO_LEGAL_ACTION"
    assert legal_actions(edited) == ()
    numerical = numerical_failure_state(root)
    assert numerical.terminal_cause == "NUMERICAL_FAILURE"
    assert legal_actions(numerical) == ()
    assert numerical.current_sequence == root.current_sequence


def test_rates_are_requested_for_exact_legal_set_and_have_support() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")

    def scorer(state, actions):
        assert {action.action_id for action in actions} == {STOP, "SUB:0:C", "SUB:0:G", "SUB:0:U"}
        return {action: 0.0 for action in actions}

    rates = positive_rates(root, scorer, support_floor=0.25)
    assert {rate for _, rate in rates} == {0.25}

    def missing_stop(state, actions):
        return {action: 1.0 for action in actions if action.kind != STOP}

    with pytest.raises(LegalFlowError, match="exactly"):
        positive_rates(root, missing_stop)


def test_exact_small_graph_absorbs_all_mass() -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")
    terminal = exact_terminal_distribution(root, _unit_rates, support_floor=1e-8)
    assert len(terminal) == 4
    assert math.fsum(terminal.values()) == pytest.approx(1.0, abs=1e-12)
    assert {state.terminal_cause for state in terminal} == {
        "EXPLICIT_STOP",
        "BUDGET_EXHAUSTED",
    }
    assert sorted(terminal.values()) == pytest.approx([0.25] * 4, abs=1e-12)


def test_sampled_trajectories_replay_and_end_legally() -> None:
    root = initial_state("ACG", budget=2, assay_id="a", context_id="c")
    for seed in range(20):
        trajectory = sample_trajectory(root, _unit_rates, seed=seed)
        assert trajectory[-1].terminal_cause in {
            "EXPLICIT_STOP",
            "BUDGET_EXHAUSTED",
        }
        for parent, child in zip(trajectory, trajectory[1:]):
            assert child in {apply_action(parent, action) for action in legal_actions(parent)}
            assert child.current_sequence == replay_source_relative(
                child.source_sequence, child.source_relative_edits
            )


@pytest.mark.parametrize("bad_rate", [-1.0, float("nan"), float("inf")])
def test_invalid_learned_rates_fail(bad_rate: float) -> None:
    root = initial_state("A", budget=1, assay_id="a", context_id="c")

    def scorer(state, actions):
        return {action: bad_rate for action in actions}

    with pytest.raises(LegalFlowError, match="invalid learned base rate"):
        positive_rates(root, scorer)


def test_inconsistent_state_is_rejected() -> None:
    state = FlowState("A", "C", (), 1, "a", "c")
    with pytest.raises(LegalFlowError, match="does not replay"):
        legal_actions(state)
