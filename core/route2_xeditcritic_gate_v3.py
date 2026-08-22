"""Frozen screen and confirmation gate logic for XEditCritic V3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr


class XEditCriticGateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticGateError(message)


def _validate_screen_summary(
    summary: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_arm: str,
    expected_run_id: str,
    expected_control_mode: str = "NONE",
    expected_candidate_permutation: bool = False,
) -> Mapping[str, Any]:
    _require(summary.get("status") == "TERMINAL_SCREEN_ARM_COMPLETE", "screen arm is not terminal-complete")
    _require(int(summary.get("seed", -1)) == expected_seed, "screen seed differs from the freeze")
    _require(str(summary.get("arm")) == expected_arm, "screen arm identity differs")
    _require(str(summary.get("run_id")) == expected_run_id, "screen run identity differs")
    _require(
        str(summary.get("control_mode")) == expected_control_mode,
        "screen control identity differs",
    )
    _require(
        summary.get("candidate_bundle_permutation")
        is expected_candidate_permutation,
        "screen candidate-permutation identity differs",
    )
    _require(
        int(summary.get("train_record_count", -1)) == 89580
        and int(summary.get("validation_record_count", -1)) == 18293,
        "screen TRAIN/Validation inventory differs",
    )
    _require(
        int(summary.get("pass_count", -1)) == 8
        and int(summary.get("selected_pass", -1)) == 8
        and int(summary.get("update_count", -1)) == 22416,
        "screen training budget differs",
    )
    _require(
        summary.get("precision") == "BF16"
        and summary.get("cuda_training_tensors_verified") is True
        and summary.get("cpu_fallback_used") is False
        and summary.get("training_scope") == "FROZEN_TRAIN_VALIDATION",
        "screen CUDA/precision/training scope differs",
    )
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
    arm_id = arm.lower()
    candidate = _validate_screen_summary(
        candidate_summary,
        expected_seed=expected_seed,
        expected_arm=arm,
        expected_run_id=arm_id,
    )
    baseline = _validate_screen_summary(
        c0_summary,
        expected_seed=expected_seed,
        expected_arm="C0",
        expected_run_id="c0",
    )
    required_controls = {
        "SOURCE_ONLY",
        "EDIT_METADATA_ONLY",
        "NO_CANDIDATE_SEQUENCE",
    }
    _require(set(control_summaries) == required_controls, "candidate-information controls are incomplete")
    controls = {
        name: _validate_screen_summary(
            summary,
            expected_seed=expected_seed,
            expected_arm=arm,
            expected_run_id=f"{arm_id}_{name.lower()}",
            expected_control_mode=name,
        )
        for name, summary in control_summaries.items()
    }
    permutation = _validate_screen_summary(
        permutation_summary,
        expected_seed=expected_seed,
        expected_arm=arm,
        expected_run_id=f"{arm_id}_candidate_bundle_permutation",
        expected_candidate_permutation=True,
    )
    candidate_tasks = candidate["tasks"]
    baseline_tasks = baseline["tasks"]
    _require(set(candidate_tasks) == set(baseline_tasks), "candidate/baseline task sets differ")
    _require(
        all(set(metrics["tasks"]) == set(candidate_tasks) for metrics in controls.values())
        and set(permutation["tasks"]) == set(candidate_tasks),
        "candidate/control task sets differ",
    )
    matched_budget_fields = (
        "train_record_count",
        "validation_record_count",
        "pass_count",
        "selected_pass",
        "update_count",
    )
    _require(
        all(
            summary[field] == candidate_summary[field]
            for summary in (*control_summaries.values(), permutation_summary)
            for field in matched_budget_fields
        ),
        "candidate/control training budgets differ",
    )
    candidate_parameter_count = int(
        candidate_summary.get(
            "total_trainable_parameter_count",
            candidate_summary.get("trainable_parameter_count", -1),
        )
    )
    _require(candidate_parameter_count > 0, "candidate parameter count is absent")
    _require(
        all(
            int(
                summary.get(
                    "total_trainable_parameter_count",
                    summary.get("trainable_parameter_count", -1),
                )
            )
            == candidate_parameter_count
            for summary in (*control_summaries.values(), permutation_summary)
        ),
        "candidate-information control is not parameter matched",
    )
    task_wins = sum(
        candidate_tasks[task]["spearman"] > baseline_tasks[task]["spearman"]
        for task in candidate_tasks
    )
    permutation_evidence = permutation_summary.get("candidate_permutation_summary", {})
    _require(
        permutation_evidence.get("exact_source_task_strata") is True
        and permutation_evidence.get("complete_candidate_bundle_permuted") is True
        and int(permutation_evidence.get("recipient_count", 0)) > 0
        and int(permutation_evidence.get("changed_candidate_sequence_count", 0)) > 0,
        "candidate permutation did not permute the complete bundle",
    )
    eligible_task_list = permutation_evidence.get("eligible_tasks", [])
    _require(
        isinstance(eligible_task_list, list)
        and len(eligible_task_list) == len(set(eligible_task_list))
        and int(permutation_evidence.get("eligible_task_count", -1))
        == len(eligible_task_list),
        "candidate permutation eligible-task inventory differs",
    )
    permutation_tasks = set(eligible_task_list)
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
    _validate_screen_summary(
        summaries["c1"],
        expected_seed=expected_seed,
        expected_arm="C1",
        expected_run_id="c1",
    )
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


CONFIRMATION_SEEDS_V3 = (20260831, 20260901, 20260902)


def _spearman(values: list[float], predictions: list[float]) -> float | None:
    if len(values) < 3 or np.std(values) == 0.0 or np.std(predictions) == 0.0:
        return None
    result = float(spearmanr(values, predictions).statistic)
    return result if math.isfinite(result) else None


def _task_macro_spearman_from_rows(
    rows: list[Mapping[str, Any]], prediction_key: str
) -> float | None:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    correlations = []
    for task in sorted(by_task):
        members = by_task[task]
        correlation = _spearman(
            [float(row["target"]) for row in members],
            [float(row[prediction_key]) for row in members],
        )
        if correlation is None:
            return None
        correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else None


def paired_source_group_task_macro_bootstrap_v3(
    candidate_rows: list[Mapping[str, Any]],
    baseline_rows: list[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired, task-stratified source-group bootstrap of task-macro Spearman."""
    _require(iterations >= 1000, "confirmation bootstrap budget is below 1000")
    candidate_by_id = {str(row["record_id"]): row for row in candidate_rows}
    baseline_by_id = {str(row["record_id"]): row for row in baseline_rows}
    _require(
        len(candidate_by_id) == len(candidate_rows)
        and len(baseline_by_id) == len(baseline_rows),
        "confirmation prediction record is duplicated",
    )
    _require(
        set(candidate_by_id) == set(baseline_by_id) and candidate_by_id,
        "candidate/baseline confirmation records are not exactly paired",
    )
    aligned: list[dict[str, Any]] = []
    for record_id in sorted(candidate_by_id):
        candidate = candidate_by_id[record_id]
        baseline = baseline_by_id[record_id]
        for field in ("source_group_id", "task_id", "target", "scaled_target"):
            _require(
                candidate[field] == baseline[field],
                f"candidate/baseline confirmation field differs: {record_id}/{field}",
            )
        aligned.append(
            {
                "record_id": record_id,
                "source_group_id": str(candidate["source_group_id"]),
                "task_id": str(candidate["task_id"]),
                "target": float(candidate["target"]),
                "candidate_prediction": float(candidate["prediction"]),
                "baseline_prediction": float(baseline["prediction"]),
            }
        )
    by_task_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in aligned:
        by_task_group[row["task_id"]][row["source_group_id"]].append(row)
    _require(len(by_task_group) == 9, "confirmation bootstrap does not cover nine tasks")
    _require(
        all(len(groups) >= 2 for groups in by_task_group.values()),
        "confirmation bootstrap task has fewer than two source groups",
    )
    point_candidate = _task_macro_spearman_from_rows(
        aligned, "candidate_prediction"
    )
    point_baseline = _task_macro_spearman_from_rows(aligned, "baseline_prediction")
    _require(
        point_candidate is not None and point_baseline is not None,
        "confirmation point task-macro Spearman is undefined",
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled: list[Mapping[str, Any]] = []
        for task in sorted(by_task_group):
            groups = by_task_group[task]
            keys = sorted(groups)
            indices = rng.integers(0, len(keys), size=len(keys))
            sampled.extend(
                row for index in indices for row in groups[keys[int(index)]]
            )
        candidate_value = _task_macro_spearman_from_rows(
            sampled, "candidate_prediction"
        )
        baseline_value = _task_macro_spearman_from_rows(
            sampled, "baseline_prediction"
        )
        if candidate_value is not None and baseline_value is not None:
            differences.append(candidate_value - baseline_value)
    _require(
        len(differences) >= int(iterations * 0.95),
        "too few defined confirmation bootstrap iterations",
    )
    values = np.asarray(differences, dtype=float)
    return {
        "analysis_unit": "SOURCE_GROUP_WITHIN_TASK",
        "task_count": len(by_task_group),
        "source_group_count": sum(len(groups) for groups in by_task_group.values()),
        "bootstrap_iterations": iterations,
        "defined_bootstrap_iterations": len(differences),
        "point_task_macro_spearman_difference": point_candidate - point_baseline,
        "task_macro_spearman_difference_ci_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def _validate_confirmation_summary_v3(
    summary: Mapping[str, Any], *, seed: int, expected_arm: str
) -> Mapping[str, Any]:
    _require(
        summary.get("status") == "TERMINAL_CONFIRMATION_ARM_COMPLETE",
        "confirmation arm is not terminal-complete",
    )
    _require(int(summary.get("seed", -1)) == seed, "confirmation seed differs")
    _require(str(summary.get("arm")) == expected_arm, "confirmation arm differs")
    _require(
        str(summary.get("run_id")) == expected_arm.lower()
        and summary.get("control_mode") == "NONE"
        and summary.get("candidate_bundle_permutation") is False,
        "confirmation run/control identity differs",
    )
    _require(
        int(summary.get("train_record_count", -1)) == 89580
        and int(summary.get("validation_record_count", -1)) == 18293
        and int(summary.get("pass_count", -1)) == 8
        and int(summary.get("selected_pass", -1)) == 8
        and int(summary.get("update_count", -1)) == 22416,
        "confirmation split or training budget differs",
    )
    _require(
        summary.get("precision") == "BF16"
        and summary.get("cuda_training_tensors_verified") is True
        and summary.get("cpu_fallback_used") is False
        and summary.get("training_scope") == "FROZEN_TRAIN_VALIDATION",
        "confirmation CUDA/precision/training scope differs",
    )
    _require(
        summary.get("parameter_changed") is True
        or (
            summary.get("head_parameter_changed") is True
            and summary.get("lora_parameter_changed") is True
        ),
        "confirmation performed no verified parameter update",
    )
    _require(
        summary.get("development_test_outcomes_accessed") is False,
        "confirmation accessed Development TEST outcome",
    )
    _require(
        summary.get("new_final_evaluation_outcomes_accessed") is False,
        "confirmation accessed Evaluation outcome",
    )
    final = summary.get("final_validation")
    _require(isinstance(final, Mapping), "confirmation lacks final Validation metrics")
    _require(int(final.get("task_count", 0)) == 9, "confirmation lacks nine tasks")
    spread = float(final.get("prediction_std", float("nan")))
    _require(math.isfinite(spread) and spread > 0.0, "confirmation prediction spread is invalid")
    return final


def adjudicate_critic_confirmation_v3(
    seed_payloads: Mapping[int, Mapping[str, Any]],
    *,
    selected_arm: str,
    required_seeds: tuple[int, ...] = CONFIRMATION_SEEDS_V3,
) -> dict[str, Any]:
    _require(selected_arm in {"C2", "C3"}, "confirmation selected arm is not C2/C3")
    _require(
        tuple(sorted(seed_payloads)) == tuple(sorted(required_seeds))
        and len(seed_payloads) == 3,
        "confirmation requires exactly the three frozen seeds",
    )
    seed_results: dict[str, dict[str, Any]] = {}
    for seed in required_seeds:
        payload = seed_payloads[seed]
        _require(
            set(payload) == {"candidate_summary", "baseline_summary", "bootstrap"},
            f"confirmation seed payload is incomplete: {seed}",
        )
        candidate = _validate_confirmation_summary_v3(
            payload["candidate_summary"], seed=seed, expected_arm=selected_arm
        )
        baseline = _validate_confirmation_summary_v3(
            payload["baseline_summary"], seed=seed, expected_arm="C0"
        )
        _require(
            set(candidate["tasks"]) == set(baseline["tasks"]),
            f"confirmation task inventory differs: {seed}",
        )
        _require(
            all(
                payload["candidate_summary"][field]
                == payload["baseline_summary"][field]
                for field in (
                    "train_record_count",
                    "validation_record_count",
                    "pass_count",
                    "selected_pass",
                    "update_count",
                )
            ),
            f"confirmation candidate/baseline budget differs: {seed}",
        )
        task_wins = sum(
            candidate["tasks"][task]["spearman"]
            > baseline["tasks"][task]["spearman"]
            for task in candidate["tasks"]
        )
        candidate_spearman = float(candidate["task_macro_spearman"])
        baseline_spearman = float(baseline["task_macro_spearman"])
        candidate_mae = float(candidate["task_macro_standardized_mae"])
        baseline_mae = float(baseline["task_macro_standardized_mae"])
        bootstrap = payload["bootstrap"]
        _require(
            bootstrap.get("analysis_unit") == "SOURCE_GROUP_WITHIN_TASK",
            f"confirmation bootstrap unit differs: {seed}",
        )
        _require(
            int(bootstrap.get("task_count", -1)) == 9
            and int(bootstrap.get("bootstrap_iterations", 0)) >= 1000
            and int(bootstrap.get("defined_bootstrap_iterations", 0))
            >= int(bootstrap.get("bootstrap_iterations", 0) * 0.95),
            f"confirmation bootstrap coverage differs: {seed}",
        )
        ci = bootstrap.get("task_macro_spearman_difference_ci_95")
        _require(
            isinstance(ci, list) and len(ci) == 2,
            f"confirmation bootstrap CI is absent: {seed}",
        )
        ci_lower = float(ci[0])
        bootstrap_point = float(
            bootstrap.get("point_task_macro_spearman_difference", float("nan"))
        )
        _require(
            math.isclose(
                bootstrap_point,
                candidate_spearman - baseline_spearman,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"confirmation bootstrap point estimate differs from summaries: {seed}",
        )
        checks = {
            "task_macro_spearman_at_least_0_25": candidate_spearman >= 0.25,
            "margin_over_c0_at_least_0_07": candidate_spearman - baseline_spearman >= 0.07,
            "positive_task_count_at_least_8": int(candidate["positive_task_count"]) >= 8,
            "task_wins_over_c0_at_least_6": task_wins >= 6,
            "task_macro_standardized_mae_at_most_1_70": candidate_mae <= 1.70,
            "mae_not_worse_than_c0": candidate_mae <= baseline_mae,
            "paired_bootstrap_ci_lower_bound_positive": math.isfinite(ci_lower)
            and ci_lower > 0.0,
            "protected_outcome_reads_zero": True,
        }
        seed_results[str(seed)] = {
            "candidate_task_macro_spearman": candidate_spearman,
            "baseline_task_macro_spearman": baseline_spearman,
            "margin_over_c0": candidate_spearman - baseline_spearman,
            "candidate_task_macro_standardized_mae": candidate_mae,
            "baseline_task_macro_standardized_mae": baseline_mae,
            "positive_task_count": int(candidate["positive_task_count"]),
            "task_wins_over_c0": task_wins,
            "paired_bootstrap_ci_95": [float(ci[0]), float(ci[1])],
            "checks": checks,
            "passed": all(checks.values()),
        }
    spearmans = [
        row["candidate_task_macro_spearman"] for row in seed_results.values()
    ]
    margins = [row["margin_over_c0"] for row in seed_results.values()]
    cohort_checks = {
        "exact_three_frozen_seeds": True,
        "all_seed_checks_pass": all(row["passed"] for row in seed_results.values()),
        "median_task_macro_spearman_at_least_0_30": float(np.median(spearmans))
        >= 0.30,
        "median_margin_over_c0_at_least_0_10": float(np.median(margins)) >= 0.10,
    }
    passed = all(cohort_checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_three_seed_gate.v1",
        "status": (
            "XEDITCRITIC_V3_THREE_SEED_PASS"
            if passed
            else "XEDITCRITIC_V3_THREE_SEED_NO_GO"
        ),
        "selected_arm": selected_arm,
        "required_seeds": list(required_seeds),
        "seed_results": seed_results,
        "median_task_macro_spearman": float(np.median(spearmans)),
        "median_margin_over_c0": float(np.median(margins)),
        "cohort_checks": cohort_checks,
        "development_test_authorized": passed,
        "all_development_refit_authorized": False,
        "loso_authorized": False,
        "guidance_authorized": False,
        "new_final_evaluation_authorized": False,
        "additional_seed_authorized": False,
    }


def adjudicate_critic_frozen_test_v3(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    for label, summary in (("candidate", candidate), ("baseline", baseline)):
        _require(
            summary.get("status") == "ATOMIC_FROZEN_DEVELOPMENT_TEST_EVALUATION_COMPLETE",
            f"{label} frozen TEST summary is not terminal",
        )
        _require(
            int(summary.get("test_record_count", -1)) == 18292,
            f"{label} frozen TEST count differs",
        )
        _require(
            summary.get("development_test_outcomes_accessed") is True
            and int(summary.get("development_test_access_event_count", -1)) == 1,
            f"{label} frozen TEST access was not atomic",
        )
        _require(
            summary.get("new_final_evaluation_outcomes_accessed") is False,
            f"{label} frozen TEST evaluation accessed new Evaluation",
        )
        _require(
            summary.get("general_test_projection_persisted") is False,
            f"{label} left a general TEST projection",
        )
    candidate_metrics = candidate.get("test_metrics")
    baseline_metrics = baseline.get("test_metrics")
    _require(
        isinstance(candidate_metrics, Mapping) and isinstance(baseline_metrics, Mapping),
        "frozen TEST metrics are missing",
    )
    _require(
        int(candidate_metrics.get("task_count", -1))
        == int(baseline_metrics.get("task_count", -2))
        == 9,
        "frozen TEST does not cover nine tasks",
    )
    candidate_spearman = float(candidate_metrics["task_macro_spearman"])
    baseline_spearman = float(baseline_metrics["task_macro_spearman"])
    candidate_mae = float(candidate_metrics["task_macro_standardized_mae"])
    baseline_mae = float(baseline_metrics["task_macro_standardized_mae"])
    ci = bootstrap.get("task_macro_spearman_difference_ci_95")
    _require(
        bootstrap.get("analysis_unit") == "SOURCE_GROUP_WITHIN_TASK"
        and isinstance(ci, list)
        and len(ci) == 2,
        "frozen TEST paired bootstrap differs",
    )
    checks = {
        "task_macro_spearman_at_least_0_25": candidate_spearman >= 0.25,
        "margin_over_c0_at_least_0_07": candidate_spearman - baseline_spearman >= 0.07,
        "task_macro_standardized_mae_at_most_1_70": candidate_mae <= 1.70,
        "mae_not_worse_than_c0": candidate_mae <= baseline_mae,
        "positive_task_count_at_least_8": int(candidate_metrics["positive_task_count"]) >= 8,
        "paired_bootstrap_ci_lower_bound_positive": float(ci[0]) > 0.0,
        "single_atomic_test_access": True,
        "general_test_projection_not_persisted": True,
        "new_final_evaluation_read_zero": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_frozen_test_gate.v1",
        "status": (
            "XEDITCRITIC_V3_FROZEN_TEST_PASS"
            if passed
            else "XEDITCRITIC_V3_FROZEN_TEST_NO_GO"
        ),
        "candidate_task_macro_spearman": candidate_spearman,
        "baseline_task_macro_spearman": baseline_spearman,
        "margin_over_c0": candidate_spearman - baseline_spearman,
        "candidate_task_macro_standardized_mae": candidate_mae,
        "baseline_task_macro_standardized_mae": baseline_mae,
        "paired_bootstrap_ci_95": [float(ci[0]), float(ci[1])],
        "checks": checks,
        "all_development_refit_authorized": passed,
        "loso_authorized": False,
        "guidance_authorized": False,
        "new_final_evaluation_authorized": False,
    }


def adjudicate_critic_loso_v3(
    seed_results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    required_seeds = set(CONFIRMATION_SEEDS_V3)
    _require(
        set(seed_results) == required_seeds,
        "Critic V3 LOSO requires exactly the three frozen seeds",
    )
    rows = {}
    model_spearmans = []
    for seed in CONFIRMATION_SEEDS_V3:
        result = seed_results[seed]
        _require(
            result.get("status") == "XEDITCRITIC_V3_PAIRED_LOSO_COMPLETE",
            f"Critic V3 LOSO seed is incomplete: {seed}",
        )
        _require(
            int(result.get("held_out_study_count", -1)) == 7,
            f"Critic V3 LOSO study count differs: {seed}",
        )
        _require(
            result.get("development_test_outcomes_accessed") is False
            and result.get("new_final_evaluation_outcomes_accessed") is False,
            f"Critic V3 LOSO accessed protected outcome: {seed}",
        )
        fold_margins = result.get("fold_margins")
        _require(
            isinstance(fold_margins, Mapping)
            and len(fold_margins) == 7
            and "GSE269595" in fold_margins,
            f"Critic V3 LOSO fold inventory differs: {seed}",
        )
        margins = [float(value) for value in fold_margins.values()]
        _require(all(math.isfinite(value) for value in margins), f"Critic V3 LOSO margin is nonfinite: {seed}")
        model = float(result["model_study_macro_spearman"])
        baseline = float(result["baseline_study_macro_spearman"])
        model_spearmans.append(model)
        checks = {
            "study_macro_spearman_at_least_0_20": model >= 0.20,
            "margin_over_c0_at_least_0_05": model - baseline >= 0.05,
            "positive_fold_margin_count_at_least_6": sum(value > 0.0 for value in margins) >= 6,
            "median_fold_margin_positive": float(np.median(margins)) > 0.0,
            "leave_gse269595_out_margin_positive": float(fold_margins["GSE269595"]) > 0.0,
            "protected_outcome_reads_zero": True,
        }
        rows[str(seed)] = {
            "model_study_macro_spearman": model,
            "baseline_study_macro_spearman": baseline,
            "margin_over_c0": model - baseline,
            "fold_margins": {key: float(value) for key, value in sorted(fold_margins.items())},
            "checks": checks,
            "passed": all(checks.values()),
        }
    median_spearman = float(np.median(model_spearmans))
    cohort_checks = {
        "all_three_seed_checks_pass": all(row["passed"] for row in rows.values()),
        "median_study_macro_spearman_at_least_0_25": median_spearman >= 0.25,
    }
    passed = all(cohort_checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_loso_gate.v1",
        "status": (
            "XEDITCRITIC_V3_LOSO_PASS"
            if passed
            else "XEDITCRITIC_V3_LOSO_NO_GO"
        ),
        "required_seeds": list(CONFIRMATION_SEEDS_V3),
        "seed_results": rows,
        "median_study_macro_spearman": median_spearman,
        "cohort_checks": cohort_checks,
        "guidance_readiness_authorized": passed,
        "new_final_evaluation_authorized": False,
    }


def adjudicate_critic_readiness_v3(
    three_seed_gate: Mapping[str, Any],
    frozen_test_gate: Mapping[str, Any],
    refit_manifest: Mapping[str, Any],
    loso_gate: Mapping[str, Any],
) -> dict[str, Any]:
    three_seed_passed = (
        three_seed_gate.get("status") == "XEDITCRITIC_V3_THREE_SEED_PASS"
        and three_seed_gate.get("development_test_authorized") is True
    )
    frozen_test_passed = (
        frozen_test_gate.get("status") == "XEDITCRITIC_V3_FROZEN_TEST_PASS"
        and frozen_test_gate.get("all_development_refit_authorized") is True
    )
    refit_complete = (
        refit_manifest.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit_manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V3)
        and int(refit_manifest.get("completed_refit_count", -1)) == 3
        and refit_manifest.get("development_test_outcomes_accessed_during_refit") is False
        and refit_manifest.get("new_final_evaluation_outcomes_accessed") is False
    )
    loso_passed = (
        loso_gate.get("status") == "XEDITCRITIC_V3_LOSO_PASS"
        and loso_gate.get("guidance_readiness_authorized") is True
    )
    ready = three_seed_passed and frozen_test_passed and refit_complete and loso_passed
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_guidance_readiness.v1",
        "status": "CRITIC_READY_FOR_GUIDANCE" if ready else "CRITIC_NOT_READY_FOR_GUIDANCE",
        "three_seed_passed": three_seed_passed,
        "frozen_test_passed": frozen_test_passed,
        "all_development_refit_complete": refit_complete,
        "loso_readiness_passed": loso_passed,
        "guidance_authorized": ready,
        "new_final_evaluation_authorized": False,
        "submission_ready": False,
    }
