"""Frozen screen and confirmation gate logic for XEditCritic V3."""

from __future__ import annotations

import math
from typing import Any, Mapping


class XEditCriticGateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticGateError(message)


def _validate_screen_summary(
    summary: Mapping[str, Any], *, expected_seed: int
) -> Mapping[str, Any]:
    _require(summary.get("status") == "TERMINAL_SCREEN_ARM_COMPLETE", "screen arm is not terminal-complete")
    _require(int(summary.get("seed", -1)) == expected_seed, "screen seed differs from the freeze")
    _require(summary.get("development_test_outcomes_accessed") is False, "screen arm accessed Development TEST outcome")
    _require(summary.get("new_final_evaluation_outcomes_accessed") is False, "screen arm accessed Evaluation outcome")
    final = summary.get("final_validation")
    _require(isinstance(final, Mapping), "screen arm lacks final Validation metrics")
    _require(int(final.get("task_count", 0)) == 9, "screen arm does not cover all nine Validation tasks")
    prediction_std = float(final.get("prediction_std", float("nan")))
    _require(math.isfinite(prediction_std) and prediction_std > 0.0, "prediction spread is zero or nonfinite")
    return final


def evaluate_screen_candidate_v3(
    *,
    candidate_summary: Mapping[str, Any],
    c0_summary: Mapping[str, Any],
    control_summaries: Mapping[str, Mapping[str, Any]],
    permutation_summary: Mapping[str, Any],
    expected_seed: int = 20260830,
) -> dict[str, Any]:
    arm = str(candidate_summary.get("arm"))
    _require(arm in {"C2", "C3"}, "only C2/C3 may enter the screen gate")
    candidate = _validate_screen_summary(candidate_summary, expected_seed=expected_seed)
    baseline = _validate_screen_summary(c0_summary, expected_seed=expected_seed)
    required_controls = {
        "SOURCE_ONLY",
        "EDIT_METADATA_ONLY",
        "NO_CANDIDATE_SEQUENCE",
    }
    _require(set(control_summaries) == required_controls, "candidate-information controls are incomplete")
    controls = {
        name: _validate_screen_summary(summary, expected_seed=expected_seed)
        for name, summary in control_summaries.items()
    }
    permutation = _validate_screen_summary(
        permutation_summary, expected_seed=expected_seed
    )
    candidate_tasks = candidate["tasks"]
    baseline_tasks = baseline["tasks"]
    _require(set(candidate_tasks) == set(baseline_tasks), "candidate/baseline task sets differ")
    task_wins = sum(
        candidate_tasks[task]["spearman"] > baseline_tasks[task]["spearman"]
        for task in candidate_tasks
    )
    permutation_tasks = set(
        permutation_summary.get("candidate_permutation_summary", {}).get(
            "eligible_tasks", []
        )
    )
    _require(len(permutation_tasks) >= 2, "candidate permutation has fewer than two applicable tasks")
    _require(permutation_tasks <= set(candidate_tasks), "permutation task is absent from candidate metrics")
    permutation_wins = sum(
        candidate_tasks[task]["spearman"]
        > permutation["tasks"][task]["spearman"]
        for task in sorted(permutation_tasks)
        if candidate_tasks[task]["spearman"] is not None
        and permutation["tasks"][task]["spearman"] is not None
    )
    candidate_spearman = float(candidate["task_macro_spearman"])
    baseline_spearman = float(baseline["task_macro_spearman"])
    candidate_mae = float(candidate["task_macro_standardized_mae"])
    baseline_mae = float(baseline["task_macro_standardized_mae"])
    criteria = {
        "task_macro_spearman_at_least_0_25": candidate_spearman >= 0.25,
        "margin_over_c0_at_least_0_08": candidate_spearman - baseline_spearman >= 0.08,
        "task_macro_standardized_mae_at_most_1_70": candidate_mae <= 1.70,
        "mae_not_worse_than_c0": candidate_mae <= baseline_mae,
        "positive_task_count_at_least_8": int(candidate["positive_task_count"]) >= 8,
        "task_wins_over_c0_at_least_6": task_wins >= 6,
        "beats_source_only": candidate_spearman > float(controls["SOURCE_ONLY"]["task_macro_spearman"]),
        "beats_edit_metadata_only": candidate_spearman > float(controls["EDIT_METADATA_ONLY"]["task_macro_spearman"]),
        "beats_no_candidate_sequence": candidate_spearman > float(controls["NO_CANDIDATE_SEQUENCE"]["task_macro_spearman"]),
        "permutation_wins_at_least_2": permutation_wins >= 2,
        "prediction_spread_finite_nonzero": True,
        "protected_outcome_reads_zero": True,
    }
    return {
        "arm": arm,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "task_macro_spearman": candidate_spearman,
        "margin_over_c0": candidate_spearman - baseline_spearman,
        "task_macro_standardized_mae": candidate_mae,
        "positive_task_count": int(candidate["positive_task_count"]),
        "task_wins_over_c0": task_wins,
        "permutation_applicable_task_count": len(permutation_tasks),
        "permutation_task_wins": permutation_wins,
        "control_task_macro_spearmans": {
            name: float(metrics["task_macro_spearman"])
            for name, metrics in sorted(controls.items())
        },
    }


def adjudicate_critic_screen_v3(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    expected_seed: int = 20260830,
) -> dict[str, Any]:
    required = {
        "c0",
        "c1",
        "c2",
        "c2_source_only",
        "c2_edit_metadata_only",
        "c2_no_candidate_sequence",
        "c2_candidate_bundle_permutation",
        "c3",
        "c3_source_only",
        "c3_edit_metadata_only",
        "c3_no_candidate_sequence",
        "c3_candidate_bundle_permutation",
    }
    _require(set(summaries) == required, "screen artifact set is incomplete or unauthorized")
    # C1 is diagnostic-only but must be the frozen arm and outcome-clean.
    _validate_screen_summary(summaries["c1"], expected_seed=expected_seed)
    results = {}
    for arm in ("c2", "c3"):
        results[arm.upper()] = evaluate_screen_candidate_v3(
            candidate_summary=summaries[arm],
            c0_summary=summaries["c0"],
            control_summaries={
                "SOURCE_ONLY": summaries[f"{arm}_source_only"],
                "EDIT_METADATA_ONLY": summaries[f"{arm}_edit_metadata_only"],
                "NO_CANDIDATE_SEQUENCE": summaries[f"{arm}_no_candidate_sequence"],
            },
            permutation_summary=summaries[f"{arm}_candidate_bundle_permutation"],
            expected_seed=expected_seed,
        )
    eligible = [arm for arm, result in results.items() if result["passed"]]
    if not eligible:
        status = "XEDITCRITIC_V3_SCREEN_NO_GO"
        selected = None
    elif len(eligible) == 1:
        status = "XEDITCRITIC_V3_SCREEN_PASS"
        selected = eligible[0]
    else:
        difference = abs(
            results["C2"]["task_macro_spearman"]
            - results["C3"]["task_macro_spearman"]
        )
        selected = (
            max(eligible, key=lambda arm: results[arm]["task_macro_spearman"])
            if difference > 0.005
            else "C2"
        )
        status = "XEDITCRITIC_V3_SCREEN_PASS"
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_screen_gate.v1",
        "status": status,
        "screen_seed": expected_seed,
        "eligible_arms": eligible,
        "selected_arm": selected,
        "selection_material_difference": 0.005,
        "tie_choice": "C2",
        "arm_results": results,
        "confirmation_authorized": selected is not None,
        "development_test_authorized": False,
        "new_final_evaluation_authorized": False,
    }
