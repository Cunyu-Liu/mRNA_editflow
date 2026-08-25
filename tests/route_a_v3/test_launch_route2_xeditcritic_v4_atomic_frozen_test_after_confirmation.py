from __future__ import annotations

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_atomic_frozen_test_after_confirmation as launcher


def _posttraining(*, eligible: bool, terminal: str = "SUMMARY") -> dict[str, object]:
    return {
        "eligible_components": ["critic"] if eligible else ["setflow"],
        "adjudications": {
            "critic": {"terminal_artifact_kind": terminal}
        },
    }


def _gate(status: str) -> dict[str, object]:
    passed = status == "XEDITCRITIC_V4_THREE_SEED_PASS"
    return {
        "status": status,
        "required_seeds": [20260908, 20260909, 20260910],
        "development_test_authorized": passed,
        "atomic_development_test_only": passed,
        "additional_seed_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_atomic_test_decision_launches_only_exact_three_seed_pass() -> None:
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True), _gate("XEDITCRITIC_V4_THREE_SEED_PASS")
    ) == "LAUNCH_EXACT_ATOMIC_TEST"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True), _gate("XEDITCRITIC_V4_THREE_SEED_NO_GO")
    ) == "NOT_AUTHORIZED_CRITIC_THREE_SEED_NO_GO"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=False), None
    ) == "NOT_AUTHORIZED_CRITIC_SCREEN_NO_GO"
    assert launcher.atomic_test_decision(
        _posttraining(eligible=True, terminal="FAILURE"), None
    ) == "NOT_AUTHORIZED_CRITIC_CONFIRMATION_TECHNICAL_FAILURE"


def test_atomic_test_decision_rejects_relaxed_authorization() -> None:
    gate = _gate("XEDITCRITIC_V4_THREE_SEED_PASS")
    gate["additional_seed_authorized"] = True
    with pytest.raises(Exception, match="authorization changed"):
        launcher.atomic_test_decision(_posttraining(eligible=True), gate)


def test_atomic_test_gpu_selection_uses_only_zero_to_five_and_maximum_free() -> None:
    free = {0: 20_000, 1: 35_000, 2: 34_000, 3: 10_000, 4: 36_000, 5: 36_000, 6: 80_000}
    assert launcher.select_gpu(free, required_mib=30_000) == 4
    with pytest.raises(Exception, match="no GPU 0–5"):
        launcher.select_gpu({gpu: 10_000 for gpu in range(8)}, required_mib=30_000)


def test_atomic_test_launcher_uses_formal_current_head_job_runner() -> None:
    assert launcher.JOB_RUNNER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_atomic_frozen_test_job.py"
    )
