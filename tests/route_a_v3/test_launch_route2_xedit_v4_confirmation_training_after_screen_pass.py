from __future__ import annotations

from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_training_after_screen_pass as launcher


def _gate(component: str, *, passed: bool) -> dict[str, object]:
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return {
        "status": f"{prefix}_V4_SCREEN_{'PASS' if passed else 'NO_GO'}",
        "confirmation_authorized": passed,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_confirmation_launcher_uses_current_head_formal_scheduler() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    assert launcher.CONFIRMATION_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_training_scheduler.py"
    )


def test_confirmation_gate_selection_is_component_independent() -> None:
    assert launcher.gate_passed("critic", _gate("critic", passed=True)) is True
    assert launcher.gate_passed("setflow", _gate("setflow", passed=False)) is False


def test_confirmation_gate_rejects_inconsistent_or_protected_authorization() -> None:
    inconsistent = _gate("critic", passed=True)
    inconsistent["confirmation_authorized"] = False
    with pytest.raises(Exception, match="inconsistent"):
        launcher.gate_passed("critic", inconsistent)

    protected = _gate("setflow", passed=True)
    protected["development_test_outcome_reads"] = 1
    with pytest.raises(Exception, match="protected outcome"):
        launcher.gate_passed("setflow", protected)


def test_confirmation_seed_sets_are_exact_and_additional_seed_is_absent() -> None:
    assert launcher.CRITIC_SEEDS == (20260908, 20260909, 20260910)
    assert launcher.SETFLOW_SEEDS == (20260912, 20260913, 20260914)


def test_confirmation_launcher_consumes_dual_head_screen_authorizations() -> None:
    source = open(launcher.__file__, encoding="utf-8").read()
    assert "def run(current_head: str, experiment_head: str)" in source
    assert "screen_{experiment_head}_runner_{current_head}/critic.json" in source
    assert "screen_{experiment_head}_runner_{current_head}/setflow.json" in source


def test_confirmation_training_records_memory_without_using_it_as_a_gate() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert '"diagnostic_peak_plus_two_gib_mib_by_gpu"' in source
    assert "free_memory[gpu] >=" not in source
