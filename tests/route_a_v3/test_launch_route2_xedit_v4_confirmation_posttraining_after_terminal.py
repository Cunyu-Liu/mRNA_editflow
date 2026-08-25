from __future__ import annotations

import pytest

import scripts.route_a_v3.launch_route2_xedit_v4_confirmation_posttraining_after_terminal as launcher


def test_confirmation_posttraining_launcher_uses_current_head_scheduler() -> None:
    assert launcher.POSTTRAINING_SCHEDULER == (
        launcher.WORKTREE
        / "scripts/route_a_v3/run_route2_xedit_v4_confirmation_posttraining_scheduler.py"
    )


def test_confirmation_training_job_set_is_component_exact() -> None:
    critic = launcher.expected_training_job_keys(("critic",))
    setflow = launcher.expected_training_job_keys(("setflow",))
    assert critic == {
        f"critic:{seed}:{run_id}"
        for seed in (20260908, 20260909, 20260910)
        for run_id in ("v4_full", "c0_v4")
    }
    assert setflow == {
        f"setflow:{seed}:v4_full" for seed in (20260912, 20260913, 20260914)
    }
    assert launcher.expected_training_job_keys(("critic", "setflow")) == (
        critic | setflow
    )


def test_confirmation_setflow_validation_assignment_is_exact_and_balanced() -> None:
    assignments = launcher.validation_assignments(
        (20260912, 20260913, 20260914)
    )
    jobs = [job for rows in assignments.values() for job in rows]
    assert set(jobs) == {
        (seed, checkpoint_pass)
        for seed in (20260912, 20260913, 20260914)
        for checkpoint_pass in (4, 6, 8, 10)
    }
    assert len(jobs) == 12
    assert set(assignments) == set(range(6))
    assert all(len(rows) == 2 for rows in assignments.values())


def test_confirmation_setflow_validation_rejects_additional_seed() -> None:
    with pytest.raises(Exception, match="seed changed"):
        launcher.validation_assignments((20260912, 20260915))


def test_confirmation_runtime_rejects_nonterminal_job() -> None:
    runtime = {
        "status": "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL",
        "git_head": "a" * 40,
        "eligible_components": ["setflow"],
        "jobs": {
            f"setflow:{seed}:v4_full": {
                "terminal_artifact_kind": "SUMMARY"
            }
            for seed in (20260912, 20260913, 20260914)
        },
        "active_performance_output_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    runtime["jobs"]["setflow:20260914:v4_full"]["terminal_artifact_kind"] = None
    with pytest.raises(Exception, match="lacks an exact terminal"):
        launcher.validate_confirmation_runtime(runtime, head="a" * 40)
