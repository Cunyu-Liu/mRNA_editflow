from __future__ import annotations

import copy

import pytest

from core.route2_xeditcritic_gate_v3 import (
    adjudicate_critic_confirmation_v3,
    adjudicate_critic_frozen_test_v3,
    adjudicate_critic_loso_v3,
    adjudicate_critic_readiness_v3,
    adjudicate_critic_screen_v3,
    paired_source_group_task_macro_bootstrap_v3,
)


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
    suffix_to_control = {
        "_source_only": "SOURCE_ONLY",
        "_edit_metadata_only": "EDIT_METADATA_ONLY",
        "_no_candidate_sequence": "NO_CANDIDATE_SEQUENCE",
    }
    control_mode = next(
        (mode for suffix, mode in suffix_to_control.items() if run_id.endswith(suffix)),
        "NONE",
    )
    is_permutation = run_id.endswith("_candidate_bundle_permutation")
    applicable = eligible_tasks or []
    return {
        "status": "TERMINAL_SCREEN_ARM_COMPLETE",
        "run_id": run_id,
        "arm": arm,
        "control_mode": control_mode,
        "candidate_bundle_permutation": is_permutation,
        "seed": 20260830,
        "train_record_count": 89580,
        "validation_record_count": 18293,
        "pass_count": 8,
        "selected_pass": 8,
        "update_count": 22416,
        "precision": "BF16",
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "training_scope": "FROZEN_TRAIN_VALIDATION",
        "trainable_parameter_count": 30_000_000 if arm in {"C2", "C3"} else 500_000,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "candidate_permutation_summary": {
            "exact_source_task_strata": is_permutation,
            "complete_candidate_bundle_permuted": is_permutation,
            "recipient_count": 100 if is_permutation else 0,
            "changed_candidate_sequence_count": 99 if is_permutation else 0,
            "eligible_task_count": len(applicable),
            "eligible_tasks": applicable,
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


def test_misidentified_control_or_partial_permutation_hard_fails() -> None:
    summaries = _screen()
    summaries["c2_source_only"]["control_mode"] = "NONE"
    with pytest.raises(Exception, match="control identity"):
        adjudicate_critic_screen_v3(summaries)
    summaries = _screen()
    summaries["c3_candidate_bundle_permutation"]["candidate_permutation_summary"][
        "complete_candidate_bundle_permuted"
    ] = False
    with pytest.raises(Exception, match="complete bundle"):
        adjudicate_critic_screen_v3(summaries)


def test_parameter_or_budget_mismatched_control_hard_fails() -> None:
    summaries = _screen()
    summaries["c2_no_candidate_sequence"]["trainable_parameter_count"] -= 1
    with pytest.raises(Exception, match="parameter matched"):
        adjudicate_critic_screen_v3(summaries)
    summaries = _screen()
    summaries["c3_edit_metadata_only"]["update_count"] -= 1
    with pytest.raises(Exception, match="training budget"):
        adjudicate_critic_screen_v3(summaries)


def test_protected_outcome_read_hard_fails() -> None:
    summaries = _screen()
    summaries["c3"]["development_test_outcomes_accessed"] = True
    with pytest.raises(Exception, match="Development TEST"):
        adjudicate_critic_screen_v3(summaries)


def _confirmation_summary(seed: int, arm: str, spearman: float) -> dict:
    return {
        "status": "TERMINAL_CONFIRMATION_ARM_COMPLETE",
        "run_id": arm.lower(),
        "arm": arm,
        "control_mode": "NONE",
        "candidate_bundle_permutation": False,
        "seed": seed,
        "train_record_count": 89580,
        "validation_record_count": 18293,
        "pass_count": 8,
        "selected_pass": 8,
        "update_count": 22416,
        "precision": "BF16",
        "cuda_training_tensors_verified": True,
        "cpu_fallback_used": False,
        "training_scope": "FROZEN_TRAIN_VALIDATION",
        "parameter_changed": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
        "final_validation": {
            "task_count": 9,
            "task_macro_spearman": spearman,
            "task_macro_standardized_mae": 1.0 if arm != "C0" else 1.2,
            "positive_task_count": 9,
            "prediction_std": 0.2,
            "tasks": {
                task: {
                    "spearman": spearman,
                    "standardized_mae": 1.0 if arm != "C0" else 1.2,
                }
                for task in TASKS
            },
        },
    }


def _confirmation_payloads() -> dict[int, dict]:
    return {
        seed: {
            "candidate_summary": _confirmation_summary(seed, "C2", 0.32),
            "baseline_summary": _confirmation_summary(seed, "C0", 0.20),
            "bootstrap": {
                "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
                "task_count": 9,
                "bootstrap_iterations": 10000,
                "defined_bootstrap_iterations": 10000,
                "point_task_macro_spearman_difference": 0.12,
                "task_macro_spearman_difference_ci_95": [0.03, 0.20],
            },
        }
        for seed in (20260831, 20260901, 20260902)
    }


def test_confirmation_gate_requires_all_seed_and_cohort_thresholds() -> None:
    result = adjudicate_critic_confirmation_v3(
        _confirmation_payloads(), selected_arm="C2"
    )
    assert result["status"] == "XEDITCRITIC_V3_THREE_SEED_PASS"
    assert result["development_test_authorized"] is True
    assert result["additional_seed_authorized"] is False
    failed = _confirmation_payloads()
    failed[20260901]["bootstrap"]["task_macro_spearman_difference_ci_95"][0] = 0.0
    result = adjudicate_critic_confirmation_v3(failed, selected_arm="C2")
    assert result["status"] == "XEDITCRITIC_V3_THREE_SEED_NO_GO"
    assert result["development_test_authorized"] is False


def test_confirmation_gate_rejects_missing_or_extra_seed() -> None:
    payloads = _confirmation_payloads()
    payloads.pop(20260902)
    with pytest.raises(Exception, match="exactly the three frozen seeds"):
        adjudicate_critic_confirmation_v3(payloads, selected_arm="C2")


def test_confirmation_rejects_wrong_control_identity_or_budget() -> None:
    payloads = _confirmation_payloads()
    payloads[20260901]["baseline_summary"]["control_mode"] = "SOURCE_ONLY"
    with pytest.raises(Exception, match="run/control identity"):
        adjudicate_critic_confirmation_v3(payloads, selected_arm="C2")
    payloads = _confirmation_payloads()
    payloads[20260902]["candidate_summary"]["update_count"] -= 1
    with pytest.raises(Exception, match="training budget"):
        adjudicate_critic_confirmation_v3(payloads, selected_arm="C2")


def test_confirmation_bootstrap_is_task_stratified_and_exactly_paired() -> None:
    candidate = []
    baseline = []
    for task_index, task in enumerate(TASKS):
        for group_index in range(2):
            for row_index, target in enumerate((-1.0, 0.0, 1.0)):
                record_id = f"{task_index}-{group_index}-{row_index}"
                common = {
                    "record_id": record_id,
                    "source_group_id": f"{task}-group-{group_index}",
                    "task_id": task,
                    "target": target,
                    "scaled_target": target,
                }
                candidate.append({**common, "prediction": target})
                baseline.append({**common, "prediction": -target})
    result = paired_source_group_task_macro_bootstrap_v3(
        candidate, baseline, iterations=1000, seed=20260831
    )
    assert result["analysis_unit"] == "SOURCE_GROUP_WITHIN_TASK"
    assert result["task_count"] == 9
    assert result["source_group_count"] == 18
    assert result["defined_bootstrap_iterations"] == 1000
    assert result["task_macro_spearman_difference_ci_95"][0] > 0.0
    baseline[0]["task_id"] = "mismatch"
    with pytest.raises(Exception, match="field differs"):
        paired_source_group_task_macro_bootstrap_v3(
            candidate, baseline, iterations=1000, seed=20260831
        )


def _frozen_test_summary(spearman: float, mae: float) -> dict:
    return {
        "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_EVALUATION_COMPLETE",
        "test_record_count": 18292,
        "development_test_outcomes_accessed": True,
        "development_test_access_event_count": 1,
        "new_final_evaluation_outcomes_accessed": False,
        "general_test_projection_persisted": False,
        "test_metrics": {
            "task_count": 9,
            "task_macro_spearman": spearman,
            "task_macro_standardized_mae": mae,
            "positive_task_count": 9,
        },
    }


def test_frozen_test_gate_requires_single_atomic_access_and_strict_metrics() -> None:
    result = adjudicate_critic_frozen_test_v3(
        _frozen_test_summary(0.32, 1.0),
        _frozen_test_summary(0.20, 1.2),
        {
            "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
            "task_macro_spearman_difference_ci_95": [0.02, 0.20],
        },
    )
    assert result["status"] == "XEDITCRITIC_V3_FROZEN_TEST_PASS"
    assert result["all_development_refit_authorized"] is True
    failed = _frozen_test_summary(0.32, 1.0)
    failed["general_test_projection_persisted"] = True
    with pytest.raises(Exception, match="general TEST projection"):
        adjudicate_critic_frozen_test_v3(
            failed,
            _frozen_test_summary(0.20, 1.2),
            {
                "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
                "task_macro_spearman_difference_ci_95": [0.02, 0.20],
            },
        )


def _loso_results():
    studies = ["GSE269595", "s2", "s3", "s4", "s5", "s6", "s7"]
    return {
        seed: {
            "status": "XEDITCRITIC_V3_PAIRED_LOSO_COMPLETE",
            "held_out_study_count": 7,
            "model_study_macro_spearman": 0.27,
            "baseline_study_macro_spearman": 0.20,
            "fold_margins": {study: 0.07 for study in studies},
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        for seed in (20260831, 20260901, 20260902)
    }


def test_loso_and_readiness_require_every_predecessor() -> None:
    loso = adjudicate_critic_loso_v3(_loso_results())
    assert loso["status"] == "XEDITCRITIC_V3_LOSO_PASS"
    three = {"status": "XEDITCRITIC_V3_THREE_SEED_PASS", "development_test_authorized": True}
    test = {"status": "XEDITCRITIC_V3_FROZEN_TEST_PASS", "all_development_refit_authorized": True}
    refit = {
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "required_seeds": [20260831, 20260901, 20260902],
        "completed_refit_count": 3,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    readiness = adjudicate_critic_readiness_v3(three, test, refit, loso)
    assert readiness["status"] == "CRITIC_READY_FOR_GUIDANCE"
    refit["completed_refit_count"] = 2
    readiness = adjudicate_critic_readiness_v3(three, test, refit, loso)
    assert readiness["status"] == "CRITIC_NOT_READY_FOR_GUIDANCE"
    assert readiness["guidance_authorized"] is False


def test_loso_dense_study_stress_failure_is_terminal_no_go() -> None:
    results = _loso_results()
    results[20260901]["fold_margins"]["GSE269595"] = 0.0
    result = adjudicate_critic_loso_v3(results)
    assert result["status"] == "XEDITCRITIC_V3_LOSO_NO_GO"
    assert result["guidance_readiness_authorized"] is False
