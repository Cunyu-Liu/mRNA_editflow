from __future__ import annotations

import copy

import pytest

from core.route2_xeditcritic_gate_v3 import adjudicate_critic_screen_v3


TASKS = [f"task-{index}" for index in range(9)]


def _summary(
    run_id: str,
    *,
    arm: str,
    spearman: float,
    mae: float = 1.0,
    task_spearman: float | None = None,
    eligible_tasks: list[str] | None = None,
) -> dict:
    per_task = spearman if task_spearman is None else task_spearman
    return {
        "status": "TERMINAL_SCREEN_ARM_COMPLETE",
        "run_id": run_id,
        "arm": arm,
        "seed": 20260830,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "candidate_permutation_summary": {
            "eligible_tasks": eligible_tasks or [],
        },
        "final_validation": {
            "task_count": 9,
            "task_macro_spearman": spearman,
            "task_macro_standardized_mae": mae,
            "positive_task_count": 9 if per_task > 0 else 0,
            "prediction_std": 0.2,
            "tasks": {
                task: {"spearman": per_task, "standardized_mae": mae}
                for task in TASKS
            },
        },
    }


def _screen() -> dict[str, dict]:
    result = {
        "c0": _summary("c0", arm="C0", spearman=0.20, task_spearman=0.20, mae=1.2),
        "c1": _summary("c1", arm="C1", spearman=0.21),
    }
    for arm, score in (("c2", 0.31), ("c3", 0.32)):
        result[arm] = _summary(arm, arm=arm.upper(), spearman=score, task_spearman=score)
        result[f"{arm}_source_only"] = _summary(f"{arm}_source_only", arm=arm.upper(), spearman=0.10)
        result[f"{arm}_edit_metadata_only"] = _summary(f"{arm}_edit_metadata_only", arm=arm.upper(), spearman=0.11)
        result[f"{arm}_no_candidate_sequence"] = _summary(f"{arm}_no_candidate_sequence", arm=arm.upper(), spearman=0.12)
        result[f"{arm}_candidate_bundle_permutation"] = _summary(
            f"{arm}_candidate_bundle_permutation",
            arm=arm.upper(),
            spearman=0.05,
            task_spearman=0.05,
            eligible_tasks=TASKS[:6],
        )
    return result


def test_screen_selects_higher_arm_only_beyond_material_difference() -> None:
    result = adjudicate_critic_screen_v3(_screen())
    assert result["status"] == "XEDITCRITIC_V3_SCREEN_PASS"
    assert result["selected_arm"] == "C3"
    tied = _screen()
    tied["c3"]["final_validation"]["task_macro_spearman"] = 0.314
    assert adjudicate_critic_screen_v3(tied)["selected_arm"] == "C2"


def test_any_failed_strict_criterion_makes_candidate_ineligible() -> None:
    summaries = _screen()
    summaries["c2"]["final_validation"]["positive_task_count"] = 7
    summaries["c3"]["final_validation"]["task_macro_standardized_mae"] = 1.8
    result = adjudicate_critic_screen_v3(summaries)
    assert result["status"] == "XEDITCRITIC_V3_SCREEN_NO_GO"
    assert result["confirmation_authorized"] is False


def test_missing_or_unauthorized_artifact_hard_fails() -> None:
    summaries = _screen()
    summaries.pop("c1")
    with pytest.raises(Exception, match="incomplete or unauthorized"):
        adjudicate_critic_screen_v3(summaries)


def test_protected_outcome_read_hard_fails() -> None:
    summaries = _screen()
    summaries["c3"]["development_test_outcomes_accessed"] = True
    with pytest.raises(Exception, match="Development TEST"):
        adjudicate_critic_screen_v3(summaries)
