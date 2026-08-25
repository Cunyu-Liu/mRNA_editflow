from __future__ import annotations

import scripts.route_a_v3.launch_route2_xedit_v4_postscreen_after_screen_terminal as launcher


def test_postscreen_launcher_uses_current_head_formal_coordinator() -> None:
    assert launcher.POSTSCREEN_COORDINATOR == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_postscreen_adjudication_scheduler.py"
    )


def test_postscreen_expected_screen_job_set_is_exact() -> None:
    jobs = launcher.expected_screen_jobs()
    assert len(jobs) == 10
    assert {job for job in jobs if job.startswith("critic:")} == {
        "critic:c0_v4",
        "critic:v4_full",
        "critic:v4_source_only",
        "critic:v4_edit_metadata_only",
        "critic:v4_no_candidate_sequence",
        "critic:v4_candidate_bundle_permutation",
        "critic:v4_no_cross",
        "critic:v4_no_moe",
    }
    assert {job for job in jobs if job.startswith("setflow:")} == {
        "setflow:v4_full",
        "setflow:v4_single_mode",
    }
