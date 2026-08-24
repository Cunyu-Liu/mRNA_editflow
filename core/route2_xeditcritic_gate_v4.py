"""Strict terminal screen gate for XEditCritic V4."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from core.route2_xeditcritic_gate_v3 import (
    paired_source_group_task_macro_bootstrap_v3,
)


CONFIRMATION_SEEDS_V4 = (20260908, 20260909, 20260910)
LOSO_STUDIES_V4 = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "GSE186455",
    "GSE256185",
    "GSE269595",
)


class XEditCriticGateV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticGateV4Error(message)


def _validate_summary_v4(
    summary: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    selected_physical_batch: int,
    formal_parameter_count: int,
    expected_seed: int = 20260907,
    expected_schema: str = "route_a_v3_route2_xeditcritic_v4_screen_run.v1",
    expected_status: str = "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE",
    expected_run_stage: str | None = None,
) -> Mapping[str, Any]:
    run_id = str(run["run_id"])
    _require(summary.get("schema_version") == expected_schema, f"{run_id} summary schema changed")
    _require(summary.get("status") == expected_status, f"{run_id} is not terminal complete")
    if expected_run_stage is not None:
        _require(summary.get("run_stage") == expected_run_stage, f"{run_id} run stage changed")
    _require(str(summary.get("run_id")) == run_id, f"{run_id} summary identity changed")
    _require(str(summary.get("model_kind")) == str(run["model"]), f"{run_id} model kind changed")
    expected_control = (
        "CANDIDATE_BUNDLE_PERMUTATION"
        if bool(run.get("candidate_bundle_permutation", False))
        else str(run["control"])
    )
    _require(str(summary.get("control_mode")) == expected_control, f"{run_id} control identity changed")
    _require(str(summary.get("mechanism_mode")) == str(run["mechanism"]), f"{run_id} mechanism identity changed")
    _require(summary.get("candidate_bundle_permutation") is bool(run.get("candidate_bundle_permutation", False)), f"{run_id} permutation identity changed")
    _require(int(summary.get("seed", -1)) == expected_seed, f"{run_id} seed changed")
    _require(int(summary.get("train_record_count", -1)) == 89580 and int(summary.get("validation_record_count", -1)) == 18293, f"{run_id} projection inventory changed")
    _require(int(summary.get("pass_count", -1)) == 8 and int(summary.get("selected_pass", -1)) == 8 and int(summary.get("update_count", -1)) == 22416, f"{run_id} training budget changed")
    _require(summary.get("selection_policy") == "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION", f"{run_id} checkpoint selection changed")
    _require(int(summary.get("physical_batch_size", -1)) == selected_physical_batch and int(summary.get("effective_batch_size", -1)) == 32, f"{run_id} batch geometry changed")
    _require(int(summary.get("singleton_forward_count", -1)) == 0, f"{run_id} used singleton forwards")
    _require(summary.get("precision") == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE" and summary.get("cpu_fallback_used") is False, f"{run_id} precision or device policy changed")
    _require(summary.get("parameter_changed") is True, f"{run_id} lacks a verified parameter update")
    _require(int(summary.get("development_test_outcome_reads", -1)) == 0, f"{run_id} accessed Development TEST outcome")
    _require(int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0, f"{run_id} accessed new Evaluation outcome")
    passes = summary.get("passes", [])
    _require(len(passes) == 8 and all(row.get("validation_metric_read") is False for row in passes), f"{run_id} read active Validation metrics")
    capacity = summary.get("capacity", {})
    parameter_count = int(capacity.get("trainable_parameter_count", -1))
    if str(run["model"]) == "V4-FULL":
        _require(parameter_count == formal_parameter_count, f"{run_id} parameter count differs from formal preflight")
        _require(165_000_000 <= parameter_count <= 175_000_000, f"{run_id} missed the 165–175M design target")
    else:
        _require(str(run["model"]) == "C0-V4" and parameter_count > 0, "C0-V4 capacity identity changed")
    peak_gib = float(summary.get("peak_vram_bytes", -1)) / 1024**3
    _require(math.isfinite(peak_gib) and 0.0 < peak_gib <= 35.0, f"{run_id} peak memory is invalid or exceeds 35 GiB")
    final = summary.get("final_validation")
    _require(isinstance(final, Mapping), f"{run_id} final Validation metrics are absent")
    _require(int(final.get("task_count", -1)) == 9 and len(final.get("tasks", {})) == 9, f"{run_id} does not cover nine tasks")
    spread = float(final.get("prediction_std", float("nan")))
    _require(math.isfinite(spread) and spread > 0.0, f"{run_id} prediction spread is zero or nonfinite")
    for task, metrics in final["tasks"].items():
        spearman = metrics.get("spearman")
        mae = metrics.get("standardized_mae")
        _require(spearman is not None and math.isfinite(float(spearman)), f"{run_id}/{task} Spearman is undefined")
        _require(mae is not None and math.isfinite(float(mae)), f"{run_id}/{task} standardized MAE is undefined")
    task_spearmans = [
        float(metrics["spearman"])
        for metrics in final["tasks"].values()
    ]
    task_maes = [
        float(metrics["standardized_mae"])
        for metrics in final["tasks"].values()
    ]
    _require(
        math.isclose(
            float(final.get("task_macro_spearman", float("nan"))),
            float(np.mean(task_spearmans)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{run_id} task-macro Spearman is inconsistent with task rows",
    )
    _require(
        math.isclose(
            float(final.get("task_macro_standardized_mae", float("nan"))),
            float(np.mean(task_maes)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{run_id} task-macro standardized MAE is inconsistent with task rows",
    )
    _require(
        int(final.get("positive_task_count", -1))
        == sum(value > 0 for value in task_spearmans),
        f"{run_id} positive-task count is inconsistent with task rows",
    )
    return final


def evaluate_xeditcritic_v4_screen(
    config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    c3_reference_spearman: float,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    required_runs = {str(run["run_id"]): run for run in config["required_screen_runs"]}
    _require(set(summaries) == set(required_runs), "Critic V4 terminal summary package is incomplete")
    _require(preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS" and preflight.get("passed") is True, "Critic V4 formal preflight is not PASS")
    _require(int(preflight.get("development_test_outcome_reads", -1)) == 0 and int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0, "Critic V4 preflight protected reads are nonzero")
    formal_count = int(preflight.get("trainable_parameter_count", -1))
    physical_batch = int(preflight.get("selected_physical_batch", -1))
    finals = {
        run_id: _validate_summary_v4(
            summaries[run_id],
            run,
            selected_physical_batch=physical_batch,
            formal_parameter_count=formal_count,
        )
        for run_id, run in required_runs.items()
    }
    full = finals["v4_full"]
    c0 = finals["c0_v4"]
    source = finals["v4_source_only"]
    metadata = finals["v4_edit_metadata_only"]
    no_candidate = finals["v4_no_candidate_sequence"]
    permutation = finals["v4_candidate_bundle_permutation"]
    no_cross = finals["v4_no_cross"]
    no_moe = finals["v4_no_moe"]
    task_set = set(full["tasks"])
    _require(all(set(metrics["tasks"]) == task_set for metrics in finals.values()), "Critic V4 task inventories differ")
    gate = config["screen_gate"]
    applicable = list(gate["permutation_applicable_tasks"])
    _require(len(applicable) == 6 and len(set(applicable)) == 6 and set(applicable) <= task_set, "permutation applicable-task freeze is invalid")
    permutation_evidence = summaries["v4_candidate_bundle_permutation"].get("candidate_permutation_summary", {})
    _require(permutation_evidence.get("complete_candidate_bundle_permuted") is True and permutation_evidence.get("exact_source_task_strata") is True, "candidate permutation did not replace the complete exact-stratum bundle")
    _require(set(permutation_evidence.get("eligible_tasks", [])) == set(applicable), "candidate permutation applicable tasks differ from the freeze")
    full_rho = float(full["task_macro_spearman"])
    c0_rho = float(c0["task_macro_spearman"])
    threshold = max(0.30, float(c3_reference_spearman) + 0.05, c0_rho + 0.10)
    c0_task_wins = sum(
        float(full["tasks"][task]["spearman"])
        > float(c0["tasks"][task]["spearman"])
        for task in sorted(task_set)
    )
    permutation_task_wins = sum(
        float(full["tasks"][task]["spearman"])
        > float(permutation["tasks"][task]["spearman"])
        for task in applicable
    )
    checks = {
        "minimum_spearman_formula": full_rho >= threshold,
        "standardized_mae_ceiling": float(full["task_macro_standardized_mae"]) <= 1.70,
        "standardized_mae_not_worse_than_c0": float(full["task_macro_standardized_mae"]) <= float(c0["task_macro_standardized_mae"]),
        "positive_task_breadth": int(full["positive_task_count"]) >= 8,
        "minimum_tasks_won_over_c0": c0_task_wins >= 6,
        "beats_source_only": full_rho > float(source["task_macro_spearman"]),
        "beats_edit_metadata_only": full_rho > float(metadata["task_macro_spearman"]),
        "beats_no_candidate_sequence": full_rho > float(no_candidate["task_macro_spearman"]),
        "permutation_aggregate_margin": full_rho - float(permutation["task_macro_spearman"]) >= 0.05,
        "permutation_five_of_six_tasks": permutation_task_wins >= 5,
        "no_cross_margin": full_rho - float(no_cross["task_macro_spearman"]) >= 0.02,
        "no_moe_margin": full_rho - float(no_moe["task_macro_spearman"]) >= 0.02,
        "protected_reads_zero": True,
        "formal_parameter_batch_memory_update_identity": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
        "status": "XEDITCRITIC_V4_SCREEN_PASS" if passed else "XEDITCRITIC_V4_SCREEN_NO_GO",
        "passed": passed,
        "selectable_model": "V4-FULL" if passed else None,
        "screen_seed": 20260907,
        "c3_reference_task_macro_spearman": float(c3_reference_spearman),
        "c0_v4_task_macro_spearman": c0_rho,
        "v4_full_task_macro_spearman": full_rho,
        "minimum_required_task_macro_spearman": threshold,
        "v4_full_task_macro_standardized_mae": float(full["task_macro_standardized_mae"]),
        "c0_task_win_count": c0_task_wins,
        "permutation_applicable_tasks": applicable,
        "permutation_task_win_count": permutation_task_wins,
        "margins": {
            "over_c0_v4": full_rho - c0_rho,
            "over_source_only": full_rho - float(source["task_macro_spearman"]),
            "over_edit_metadata_only": full_rho - float(metadata["task_macro_spearman"]),
            "over_no_candidate_sequence": full_rho - float(no_candidate["task_macro_spearman"]),
            "over_candidate_bundle_permutation": full_rho - float(permutation["task_macro_spearman"]),
            "over_no_cross": full_rho - float(no_cross["task_macro_spearman"]),
            "over_no_moe": full_rho - float(no_moe["task_macro_spearman"]),
        },
        "checks": checks,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "confirmation_authorized": passed,
        "development_test_authorized": False,
    }


def build_critic_confirmation_seed_payload_v4(
    candidate_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    candidate_prediction_rows: list[Mapping[str, Any]],
    baseline_prediction_rows: list[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _require(seed in {20260908, 20260909, 20260910}, "Critic V4 confirmation seed is undeclared")
    _require(
        bootstrap_seed == seed * 100 + 1,
        "Critic V4 confirmation bootstrap seed changed",
    )
    bootstrap = paired_source_group_task_macro_bootstrap_v3(
        candidate_prediction_rows,
        baseline_prediction_rows,
        iterations=10_000,
        seed=bootstrap_seed,
    )
    return {
        "candidate_summary": dict(candidate_summary),
        "baseline_summary": dict(baseline_summary),
        "bootstrap": bootstrap,
    }


def adjudicate_critic_confirmation_v4(
    config: Mapping[str, Any],
    seed_payloads: Mapping[int, Mapping[str, Any]],
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    required_seeds = (20260908, 20260909, 20260910)
    _require(
        tuple(sorted(seed_payloads)) == required_seeds and len(seed_payloads) == 3,
        "Critic V4 confirmation requires exactly the three frozen seeds",
    )
    _require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True
        and int(preflight.get("development_test_outcome_reads", -1)) == 0
        and int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 confirmation preflight is invalid or reports a protected read",
    )
    formal_count = int(preflight.get("trainable_parameter_count", -1))
    physical_batch = int(preflight.get("selected_physical_batch", -1))
    runs = {str(row["run_id"]): row for row in config["required_screen_runs"]}
    _require(
        {"v4_full", "c0_v4"} <= set(runs),
        "Critic V4 confirmation full or matched baseline spec is absent",
    )
    seed_results: dict[str, Any] = {}
    for seed in required_seeds:
        payload = seed_payloads[seed]
        _require(
            set(payload) == {"candidate_summary", "baseline_summary", "bootstrap"},
            f"Critic V4 confirmation payload is incomplete: {seed}",
        )
        candidate = _validate_summary_v4(
            payload["candidate_summary"],
            runs["v4_full"],
            selected_physical_batch=physical_batch,
            formal_parameter_count=formal_count,
            expected_seed=seed,
            expected_schema="route_a_v3_route2_xeditcritic_v4_confirmation_run.v1",
            expected_status="TERMINAL_XEDITCRITIC_V4_CONFIRMATION_RUN_COMPLETE",
            expected_run_stage="CONFIRMATION",
        )
        baseline = _validate_summary_v4(
            payload["baseline_summary"],
            runs["c0_v4"],
            selected_physical_batch=physical_batch,
            formal_parameter_count=formal_count,
            expected_seed=seed,
            expected_schema="route_a_v3_route2_xeditcritic_v4_confirmation_run.v1",
            expected_status="TERMINAL_XEDITCRITIC_V4_CONFIRMATION_RUN_COMPLETE",
            expected_run_stage="CONFIRMATION",
        )
        _require(
            all(
                payload["candidate_summary"].get(field)
                == payload["baseline_summary"].get(field)
                for field in (
                    "train_record_count",
                    "validation_record_count",
                    "pass_count",
                    "selected_pass",
                    "update_count",
                    "physical_batch_size",
                    "effective_batch_size",
                )
            ),
            f"Critic V4 confirmation candidate/baseline budget differs: {seed}",
        )
        candidate_rho = float(candidate["task_macro_spearman"])
        baseline_rho = float(baseline["task_macro_spearman"])
        candidate_mae = float(candidate["task_macro_standardized_mae"])
        baseline_mae = float(baseline["task_macro_standardized_mae"])
        task_wins = sum(
            float(candidate["tasks"][task]["spearman"])
            > float(baseline["tasks"][task]["spearman"])
            for task in candidate["tasks"]
        )
        bootstrap = payload["bootstrap"]
        _require(
            bootstrap.get("analysis_unit") == "SOURCE_GROUP_WITHIN_TASK"
            and int(bootstrap.get("task_count", -1)) == 9
            and int(bootstrap.get("bootstrap_iterations", -1)) == 10_000
            and int(bootstrap.get("defined_bootstrap_iterations", 0)) >= 9_500,
            f"Critic V4 confirmation bootstrap identity changed: {seed}",
        )
        ci = bootstrap.get("task_macro_spearman_difference_ci_95")
        _require(isinstance(ci, list) and len(ci) == 2, f"Critic V4 confirmation CI is absent: {seed}")
        margin = candidate_rho - baseline_rho
        _require(
            math.isclose(
                float(bootstrap.get("point_task_macro_spearman_difference", float("nan"))),
                margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"Critic V4 confirmation bootstrap point differs: {seed}",
        )
        checks = {
            "task_macro_spearman_at_least_0_30": candidate_rho >= 0.30,
            "margin_over_c0_v4_at_least_0_10": margin >= 0.10,
            "task_macro_standardized_mae_at_most_1_70": candidate_mae <= 1.70,
            "mae_not_worse_than_c0_v4": candidate_mae <= baseline_mae,
            "positive_task_count_at_least_8": int(candidate["positive_task_count"]) >= 8,
            "task_wins_over_c0_v4_at_least_6": task_wins >= 6,
            "paired_bootstrap_ci_lower_bound_positive": math.isfinite(float(ci[0]))
            and float(ci[0]) > 0.0,
            "protected_reads_zero": True,
        }
        seed_results[str(seed)] = {
            "candidate_task_macro_spearman": candidate_rho,
            "baseline_task_macro_spearman": baseline_rho,
            "margin_over_c0_v4": margin,
            "candidate_task_macro_standardized_mae": candidate_mae,
            "baseline_task_macro_standardized_mae": baseline_mae,
            "positive_task_count": int(candidate["positive_task_count"]),
            "task_wins_over_c0_v4": task_wins,
            "paired_bootstrap_ci_95": [float(ci[0]), float(ci[1])],
            "checks": checks,
            "passed": all(checks.values()),
        }
    spearmans = [row["candidate_task_macro_spearman"] for row in seed_results.values()]
    margins = [row["margin_over_c0_v4"] for row in seed_results.values()]
    cohort_checks = {
        "exact_three_frozen_seeds": True,
        "all_seed_checks_pass": all(row["passed"] for row in seed_results.values()),
        "median_task_macro_spearman_at_least_0_35": float(np.median(spearmans)) >= 0.35,
        "median_margin_over_c0_v4_at_least_0_12": float(np.median(margins)) >= 0.12,
    }
    passed = all(cohort_checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_three_seed_gate.v1",
        "status": "XEDITCRITIC_V4_THREE_SEED_PASS"
        if passed
        else "XEDITCRITIC_V4_THREE_SEED_NO_GO",
        "required_seeds": list(required_seeds),
        "seed_results": seed_results,
        "cohort_checks": cohort_checks,
        "development_test_authorized": passed,
        "atomic_development_test_only": passed,
        "additional_seed_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def adjudicate_critic_frozen_test_v4(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    for label, summary in (("candidate", candidate), ("baseline", baseline)):
        _require(
            summary.get("status")
            == "ATOMIC_FROZEN_DEVELOPMENT_TEST_EVALUATION_COMPLETE"
            and int(summary.get("test_record_count", -1)) == 18_292,
            f"{label} Critic V4 frozen TEST summary is not exact terminal",
        )
        _require(
            summary.get("development_test_outcomes_accessed") is True
            and int(summary.get("development_test_access_event_count", -1)) == 1
            and summary.get("general_test_projection_persisted") is False
            and summary.get("new_final_evaluation_outcomes_accessed") is False,
            f"{label} Critic V4 frozen TEST access was not single and atomic",
        )
    candidate_metrics = candidate.get("test_metrics")
    baseline_metrics = baseline.get("test_metrics")
    _require(
        isinstance(candidate_metrics, Mapping)
        and isinstance(baseline_metrics, Mapping)
        and int(candidate_metrics.get("task_count", -1))
        == int(baseline_metrics.get("task_count", -2))
        == 9,
        "Critic V4 frozen TEST metrics do not cover nine tasks",
    )
    candidate_rho = float(candidate_metrics["task_macro_spearman"])
    baseline_rho = float(baseline_metrics["task_macro_spearman"])
    candidate_mae = float(candidate_metrics["task_macro_standardized_mae"])
    baseline_mae = float(baseline_metrics["task_macro_standardized_mae"])
    ci = bootstrap.get("task_macro_spearman_difference_ci_95")
    _require(
        bootstrap.get("analysis_unit") == "SOURCE_GROUP_WITHIN_TASK"
        and int(bootstrap.get("bootstrap_iterations", -1)) == 10_000
        and isinstance(ci, list)
        and len(ci) == 2,
        "Critic V4 frozen TEST paired bootstrap identity changed",
    )
    margin = candidate_rho - baseline_rho
    _require(
        math.isclose(
            float(bootstrap.get("point_task_macro_spearman_difference", float("nan"))),
            margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Critic V4 frozen TEST bootstrap point differs from summaries",
    )
    checks = {
        "task_macro_spearman_at_least_0_30": candidate_rho >= 0.30,
        "margin_over_c0_v4_at_least_0_10": margin >= 0.10,
        "task_macro_standardized_mae_at_most_1_70": candidate_mae <= 1.70,
        "mae_not_worse_than_c0_v4": candidate_mae <= baseline_mae,
        "positive_task_count_at_least_8": int(candidate_metrics["positive_task_count"]) >= 8,
        "paired_bootstrap_ci_lower_bound_positive": math.isfinite(float(ci[0]))
        and float(ci[0]) > 0.0,
        "single_atomic_test_access": True,
        "general_test_projection_not_persisted": True,
        "new_final_evaluation_read_zero": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_frozen_test_gate.v1",
        "status": "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        if passed
        else "XEDITCRITIC_V4_FROZEN_TEST_NO_GO",
        "candidate_task_macro_spearman": candidate_rho,
        "baseline_task_macro_spearman": baseline_rho,
        "margin_over_c0_v4": margin,
        "candidate_task_macro_standardized_mae": candidate_mae,
        "baseline_task_macro_standardized_mae": baseline_mae,
        "paired_bootstrap_ci_95": [float(ci[0]), float(ci[1])],
        "checks": checks,
        "all_development_refit_authorized": passed,
        "loso_authorized": False,
        "guidance_authorized": False,
        "new_final_evaluation_authorized": False,
    }


def adjudicate_critic_loso_v4(
    seed_results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        tuple(sorted(seed_results)) == CONFIRMATION_SEEDS_V4,
        "Critic V4 LOSO requires exactly the three frozen seeds",
    )
    rows: dict[str, Any] = {}
    model_spearmans = []
    for seed in CONFIRMATION_SEEDS_V4:
        result = seed_results[seed]
        _require(
            result.get("status") == "XEDITCRITIC_V4_PAIRED_LOSO_COMPLETE"
            and int(result.get("held_out_study_count", -1)) == 7,
            f"Critic V4 LOSO seed is not exact terminal: {seed}",
        )
        _require(
            result.get("development_test_outcomes_accessed_during_loso") is False
            and result.get("new_final_evaluation_outcomes_accessed") is False,
            f"Critic V4 LOSO accessed protected outcome: {seed}",
        )
        fold_margins = result.get("fold_margins")
        _require(
            isinstance(fold_margins, Mapping)
            and set(fold_margins) == set(LOSO_STUDIES_V4),
            f"Critic V4 LOSO fold inventory changed: {seed}",
        )
        margins = [float(fold_margins[study]) for study in LOSO_STUDIES_V4]
        _require(
            all(math.isfinite(value) for value in margins),
            f"Critic V4 LOSO margin is nonfinite: {seed}",
        )
        model = float(result["model_study_macro_spearman"])
        baseline = float(result["baseline_study_macro_spearman"])
        _require(
            math.isfinite(model) and math.isfinite(baseline),
            f"Critic V4 LOSO study-macro Spearman is nonfinite: {seed}",
        )
        margin = model - baseline
        checks = {
            "study_macro_spearman_at_least_0_25": model >= 0.25,
            "margin_over_c0_v4_at_least_0_07": margin >= 0.07,
            "positive_fold_margin_count_at_least_6": sum(
                value > 0.0 for value in margins
            )
            >= 6,
            "median_fold_margin_positive": float(np.median(margins)) > 0.0,
            "leave_gse269595_out_margin_positive": float(
                fold_margins["GSE269595"]
            )
            > 0.0,
            "protected_outcome_reads_zero": True,
        }
        model_spearmans.append(model)
        rows[str(seed)] = {
            "model_study_macro_spearman": model,
            "baseline_study_macro_spearman": baseline,
            "margin_over_c0_v4": margin,
            "fold_margins": {
                study: float(fold_margins[study]) for study in LOSO_STUDIES_V4
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    median_spearman = float(np.median(model_spearmans))
    cohort_checks = {
        "exact_three_frozen_seeds": True,
        "all_three_seed_checks_pass": all(row["passed"] for row in rows.values()),
        "median_study_macro_spearman_at_least_0_30": median_spearman >= 0.30,
    }
    passed = all(cohort_checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_gate.v1",
        "status": "XEDITCRITIC_V4_LOSO_PASS"
        if passed
        else "XEDITCRITIC_V4_LOSO_NO_GO",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "held_out_studies": list(LOSO_STUDIES_V4),
        "seed_results": rows,
        "median_study_macro_spearman": median_spearman,
        "cohort_checks": cohort_checks,
        "guidance_readiness_authorized": passed,
        "new_final_evaluation_authorized": False,
    }


def adjudicate_critic_readiness_v4(
    three_seed_gate: Mapping[str, Any],
    frozen_test_gate: Mapping[str, Any],
    refit_manifest: Mapping[str, Any],
    loso_gate: Mapping[str, Any],
) -> dict[str, Any]:
    three_seed_passed = (
        three_seed_gate.get("status") == "XEDITCRITIC_V4_THREE_SEED_PASS"
        and three_seed_gate.get("development_test_authorized") is True
        and three_seed_gate.get("atomic_development_test_only") is True
        and three_seed_gate.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
    )
    frozen_test_passed = (
        frozen_test_gate.get("status") == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        and frozen_test_gate.get("all_development_refit_authorized") is True
    )
    refit_complete = (
        refit_manifest.get("status")
        == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit_manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and int(refit_manifest.get("completed_refit_count", -1)) == 3
        and int(refit_manifest.get("refit_pass_count", -1)) == 8
        and refit_manifest.get("development_test_outcomes_accessed_during_refit")
        is False
        and refit_manifest.get("new_final_evaluation_outcomes_accessed") is False
    )
    loso_passed = (
        loso_gate.get("status") == "XEDITCRITIC_V4_LOSO_PASS"
        and loso_gate.get("guidance_readiness_authorized") is True
        and loso_gate.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and loso_gate.get("held_out_studies") == list(LOSO_STUDIES_V4)
    )
    ready = three_seed_passed and frozen_test_passed and refit_complete and loso_passed
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_guidance_readiness.v1",
        "status": "CRITIC_V4_READY_FOR_GUIDANCE"
        if ready
        else "CRITIC_V4_NOT_READY_FOR_GUIDANCE",
        "three_seed_passed": three_seed_passed,
        "frozen_test_passed": frozen_test_passed,
        "all_development_refit_complete": refit_complete,
        "loso_readiness_passed": loso_passed,
        "development_test_access_event_count": 1 if frozen_test_passed else 0,
        "general_test_projection_persisted": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
        "guidance_authorized": ready,
        "new_final_evaluation_authorized": False,
        "submission_ready": False,
    }
