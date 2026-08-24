"""Strict terminal screen gate for XEditCritic V4."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


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
) -> Mapping[str, Any]:
    run_id = str(run["run_id"])
    _require(summary.get("schema_version") == "route_a_v3_route2_xeditcritic_v4_screen_run.v1", f"{run_id} summary schema changed")
    _require(summary.get("status") == "TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE", f"{run_id} is not terminal complete")
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
    _require(int(summary.get("seed", -1)) == 20260907, f"{run_id} seed changed")
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
