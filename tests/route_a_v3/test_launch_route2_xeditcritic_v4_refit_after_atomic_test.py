from __future__ import annotations

import pytest

import scripts.route_a_v3.launch_route2_xeditcritic_v4_refit_after_atomic_test as launcher


def _receipt(status: str) -> dict[str, object]:
    passed = status == "XEDITCRITIC_V4_POSTTEST_AUTHORIZED"
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1",
        "status": status,
        "required_seeds": [20260908, 20260909, 20260910],
        "frozen_test_gate_status": (
            "XEDITCRITIC_V4_FROZEN_TEST_PASS"
            if passed
            else "XEDITCRITIC_V4_FROZEN_TEST_NO_GO"
        ),
        "all_development_refit_authorized": passed,
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
        "test_bottom_six_cache_persisted": False,
        "development_test_metrics_in_receipt": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def test_refit_decision_requires_atomic_result_and_passing_receipt() -> None:
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "RESULT"},
        _receipt("XEDITCRITIC_V4_POSTTEST_AUTHORIZED"),
    ) == "LAUNCH_EXACT_THREE_REFITS"
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "RESULT"},
        _receipt("XEDITCRITIC_V4_POSTTEST_NOT_AUTHORIZED"),
    ) == "REFIT_NOT_AUTHORIZED_FROZEN_TEST_NO_GO"
    assert launcher.refit_decision(
        {"terminal_artifact_kind": "FAILURE"}, None
    ) == "REFIT_NOT_AUTHORIZED_ATOMIC_TEST_TECHNICAL_FAILURE"


def test_refit_decision_rejects_receipt_with_test_metrics() -> None:
    receipt = _receipt("XEDITCRITIC_V4_POSTTEST_AUTHORIZED")
    receipt["development_test_metrics_in_receipt"] = True
    with pytest.raises(Exception, match="receipt changed"):
        launcher.refit_decision({"terminal_artifact_kind": "RESULT"}, receipt)


def test_refit_manifest_is_exact_three_seed_full_only() -> None:
    payload = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_job_manifest.v1",
        "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": [20260908, 20260909, 20260910],
        "refit_pass_count": 8,
        "job_count": 3,
        "jobs": [
            {"seed": seed, "run_id": "v4_full"}
            for seed in (20260908, 20260909, 20260910)
        ],
    }
    assert len(launcher.validate_refit_manifest(payload)) == 3
    payload["jobs"].append({"seed": 20260911, "run_id": "v4_full"})
    with pytest.raises(Exception, match="job set changed"):
        launcher.validate_refit_manifest(payload)


def test_refit_launcher_uses_formal_current_head_scheduler() -> None:
    assert launcher.REFIT_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xeditcritic_v4_refit_scheduler.py"
    )


def test_refit_gpu_selection_uses_three_best_eligible_zero_to_five() -> None:
    free = {0: 31_000, 1: 36_000, 2: 35_000, 3: 34_000, 4: 10_000, 5: 36_000, 6: 90_000}
    assert launcher.select_refit_gpus(free, required_mib=30_000) == (1, 5, 2)
    with pytest.raises(Exception, match="fewer than three"):
        launcher.select_refit_gpus(
            {0: 31_000, 1: 29_000, 2: 35_000, 3: 20_000, 4: 10_000, 5: 15_000},
            required_mib=30_000,
        )
