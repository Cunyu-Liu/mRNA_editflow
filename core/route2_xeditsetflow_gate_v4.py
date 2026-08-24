"""Strict terminal checkpoint selection and screen gate for SetFlow V4."""

from __future__ import annotations

import math
from typing import Any, Mapping


class XEditSetFlowGateV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowGateV4Error(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def _require_compute_identity_v4(
    summary: Mapping[str, Any], *, mode_count: int
) -> None:
    compute = summary.get("compute")
    _require(isinstance(compute, Mapping), "SetFlow V4 compute record is absent")
    _require(
        int(compute.get("trajectory_count", -1)) == 28_512
        and int(compute.get("candidate_count", -1)) == 28_512
        and int(compute.get("critic_forward_count", -1)) == 0
        and int(compute.get("independent_evaluator_forward_count", -1)) == 0,
        "SetFlow V4 compute budget or prohibited forward count changed",
    )
    common_states = int(compute.get("common_nll_trunk_forward_state_count", -1))
    _require(
        int(compute.get("common_nll_trunk_forward_batch_count", -1)) == 996
        and common_states == 31_848
        and int(compute.get("common_nll_mode_head_forward_state_count", -1))
        == common_states * mode_count,
        "SetFlow V4 common-NLL mode-head compute is undercounted",
    )
    for section in ("root_prior", "primary_generation", "replay_generation"):
        row = compute.get(section)
        _require(isinstance(row, Mapping), f"SetFlow V4 {section} compute is absent")
        trunk_states = int(row.get("trunk_forward_state_count", -1))
        _require(
            int(row.get("trunk_forward_batch_count", -1)) > 0
            and trunk_states > 0
            and int(row.get("mode_head_forward_state_count", -1))
            == trunk_states * mode_count,
            f"SetFlow V4 {section} mode-head compute is undercounted",
        )
    root = compute["root_prior"]
    primary = compute["primary_generation"]
    replay = compute["replay_generation"]
    _require(
        int(root["trunk_forward_batch_count"]) == 14
        and int(root["trunk_forward_state_count"]) == 891,
        "SetFlow V4 root-prior compute differs from 891 sources at batch64",
    )
    _require(
        int(primary["trunk_forward_state_count"])
        == int(replay["trunk_forward_state_count"]),
        "SetFlow V4 exact replay compute differs from primary generation",
    )


def validate_checkpoint_summary_identity_v4(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    checkpoint_pass: int,
) -> dict[str, Any]:
    mode_count = 8 if run_id == "v4_full" else 1
    selectable = run_id == "v4_full"
    _require(
        summary.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_checkpoint_validation.v1"
        and summary.get("status")
        == "TERMINAL_XEDITSETFLOW_V4_CHECKPOINT_VALIDATION_COMPLETE",
        "SetFlow V4 checkpoint validation is not terminal complete",
    )
    _require(
        summary.get("run_id") == run_id
        and int(summary.get("checkpoint_pass", -1)) == checkpoint_pass
        and int(summary.get("seed", -1)) == 20260911
        and int(summary.get("mode_count", -1)) == mode_count
        and summary.get("selectable") is selectable,
        "SetFlow V4 checkpoint validation identity changed",
    )
    _require(
        int(summary.get("source_count", -1)) == 891
        and int(summary.get("trajectory_count", -1)) == 28_512
        and int(summary.get("candidate_count", -1)) == 28_512
        and int(summary.get("candidate_cap_per_source", -1)) == 32
        and int(summary.get("duplicate_retry_or_rejection_count", -1)) == 0,
        "SetFlow V4 source, trajectory, candidate, or retry budget changed",
    )
    _require(
        summary.get("training_summary_status")
        == "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"
        and summary.get("precision") == "BF16"
        and summary.get("cpu_fallback_used") is False
        and int(summary.get("parameter_update_count", -1)) == 0,
        "SetFlow V4 checkpoint validation provenance changed",
    )
    _require(
        summary.get("g0_status") == "FLOW_G0_READY"
        and _finite(summary.get("wall_time_seconds"), "wall time") > 0.0
        and int(summary.get("peak_vram_bytes", 0)) > 0,
        "SetFlow V4 G0 status, wall time, or peak VRAM is absent",
    )
    _require(
        summary.get("critic_used") is False
        and summary.get("independent_evaluator_used") is False
        and int(summary.get("development_test_outcome_reads", -1)) == 0
        and int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4 checkpoint validation used prohibited information",
    )
    common = summary.get("common_validation")
    _require(
        isinstance(common, Mapping)
        and int(common.get("validation_candidate_record_count", -1)) == 15_924
        and int(common.get("validation_states_per_record", -1)) == 2,
        "SetFlow V4 common Validation cohort changed",
    )
    allocations = summary.get("aggregate_mode_allocations")
    _require(isinstance(allocations, Mapping), "SetFlow V4 mode allocations are absent")
    expected_keys = {str(index) for index in range(mode_count)}
    _require(set(allocations) == expected_keys, "SetFlow V4 mode allocation keys changed")
    _require(
        sum(int(value) for value in allocations.values()) == 28_512
        and all(int(value) >= 891 for value in allocations.values()),
        "SetFlow V4 omitted a mode or changed trajectory budget",
    )
    small_graph = summary.get("small_graph_reference")
    _require(
        isinstance(small_graph, Mapping)
        and small_graph.get("status") == "PASS"
        and int(small_graph.get("mode_count", -1)) == mode_count
        and _finite(small_graph.get("total_variation"), "small-graph TV")
        <= _finite(small_graph.get("tolerance"), "small-graph tolerance"),
        "SetFlow V4 small-graph mixture exactness is absent or failed",
    )
    _require_compute_identity_v4(summary, mode_count=mode_count)
    nll = _finite(
        summary.get("common_validation_set_marginal_nll"), "common Validation NLL"
    )
    _require(
        math.isclose(
            nll,
            _finite(
                common.get("common_validation_set_marginal_nll"),
                "nested common Validation NLL",
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "SetFlow V4 common-NLL summary fields disagree",
    )
    recovery = _finite(
        summary.get("source_macro_candidate_recovery_rate"), "recovery"
    )
    top_k = _finite(
        summary.get("source_macro_measured_top_k_recovery_at_k"), "top-k recovery"
    )
    unique = _finite(
        summary.get("source_macro_unique_candidate_rate"), "unique rate"
    )
    checks = {
        "common_nll_at_most_2_06809": nll <= 2.06809,
        "recovery_at_least_0_35": recovery >= 0.35,
        "top_k_recovery_at_least_0_20": top_k >= 0.20,
        "unique_candidate_rate_at_least_0_90": unique >= 0.90,
        "hard_legality_100pct": _finite(
            summary.get("hard_legality_rate"), "hard legality"
        )
        == 1.0,
        "edit_budget_violation_zero": int(
            summary.get("edit_budget_violation_count", -1)
        )
        == 0,
        "candidate_budget_violation_zero": int(
            summary.get("candidate_budget_violation_count", -1)
        )
        == 0,
        "trajectory_replay_failure_zero": int(
            summary.get("trajectory_replay_failure_count", -1)
        )
        == 0,
        "numerical_failure_zero": int(summary.get("numerical_failure_count", -1))
        == 0,
        "small_graph_exact": small_graph.get("status") == "PASS",
    }
    return {
        "run_id": run_id,
        "checkpoint_pass": checkpoint_pass,
        "common_validation_set_marginal_nll": nll,
        "source_macro_candidate_recovery_rate": recovery,
        "source_macro_measured_top_k_recovery_at_k": top_k,
        "source_macro_unique_candidate_rate": unique,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def select_checkpoint_v4(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    _require(set(rows) == {4, 6, 8, 10}, "SetFlow V4 checkpoint set is incomplete")
    eligible = [dict(rows[checkpoint_pass]) for checkpoint_pass in (4, 6, 8, 10) if rows[checkpoint_pass]["eligible"]]
    nll_selected = min(
        (dict(rows[checkpoint_pass]) for checkpoint_pass in (4, 6, 8, 10)),
        key=lambda row: (
            row["common_validation_set_marginal_nll"],
            row["checkpoint_pass"],
        ),
    )
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                -row["source_macro_candidate_recovery_rate"],
                -row["source_macro_measured_top_k_recovery_at_k"],
                row["common_validation_set_marginal_nll"],
                row["checkpoint_pass"],
            ),
        )
    return {
        "eligible_checkpoint_passes": [row["checkpoint_pass"] for row in eligible],
        "generation_constrained_selected_checkpoint": selected,
        "nll_selected_checkpoint": nll_selected,
        "nll_only_selection_differs": selected is not None
        and selected["checkpoint_pass"] != nll_selected["checkpoint_pass"],
    }


def adjudicate_setflow_screen_v4(
    config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    _require(
        set(summaries) == {"v4_full", "v4_single_mode"},
        "SetFlow V4 screen runs are incomplete",
    )
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    selections: dict[str, dict[str, Any]] = {}
    for run_id in ("v4_full", "v4_single_mode"):
        _require(
            set(summaries[run_id]) == {4, 6, 8, 10},
            f"SetFlow V4 checkpoint validations are incomplete: {run_id}",
        )
        rows[run_id] = {
            checkpoint_pass: validate_checkpoint_summary_identity_v4(
                summaries[run_id][checkpoint_pass],
                run_id=run_id,
                checkpoint_pass=checkpoint_pass,
            )
            for checkpoint_pass in (4, 6, 8, 10)
        }
        selections[run_id] = select_checkpoint_v4(rows[run_id])
    full = selections["v4_full"]["generation_constrained_selected_checkpoint"]
    single = selections["v4_single_mode"][
        "generation_constrained_selected_checkpoint"
    ]
    reference = config["terminal_f2_reference"]
    checks = {
        "full_has_eligible_checkpoint": full is not None,
        "single_mode_has_eligible_checkpoint": single is not None,
    }
    if full is not None and single is not None:
        checks.update(
            {
                "recovery_margin_over_terminal_f2_at_least_0_05": full[
                    "source_macro_candidate_recovery_rate"
                ]
                - float(reference["source_macro_recovery"])
                >= 0.05,
                "top_k_margin_over_terminal_f2_at_least_0_03": full[
                    "source_macro_measured_top_k_recovery_at_k"
                ]
                - float(reference["source_macro_top_k_recovery"])
                >= 0.03,
                "unique_margin_over_terminal_f2_at_least_0_15": full[
                    "source_macro_unique_candidate_rate"
                ]
                - float(reference["source_macro_unique_candidate_rate"])
                >= 0.15,
                "recovery_margin_over_single_mode_at_least_0_03": full[
                    "source_macro_candidate_recovery_rate"
                ]
                - single["source_macro_candidate_recovery_rate"]
                >= 0.03,
                "unique_margin_over_single_mode_at_least_0_05": full[
                    "source_macro_unique_candidate_rate"
                ]
                - single["source_macro_unique_candidate_rate"]
                >= 0.05,
            }
        )
    else:
        checks.update(
            {
                "recovery_margin_over_terminal_f2_at_least_0_05": False,
                "top_k_margin_over_terminal_f2_at_least_0_03": False,
                "unique_margin_over_terminal_f2_at_least_0_15": False,
                "recovery_margin_over_single_mode_at_least_0_03": False,
                "unique_margin_over_single_mode_at_least_0_05": False,
            }
        )
    passed = all(checks.values())
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1",
        "status": "XEDITSETFLOW_V4_SCREEN_PASS"
        if passed
        else "XEDITSETFLOW_V4_SCREEN_NO_GO",
        "screen_seed": 20260911,
        "checkpoint_rows": rows,
        "checkpoint_decisions": selections,
        "terminal_f2_reference": dict(reference),
        "screen_checks": checks,
        "selected_checkpoint_pass": None if full is None else full["checkpoint_pass"],
        "confirmation_authorized": passed,
        "confirmation_seeds": [20260912, 20260913, 20260914]
        if passed
        else [],
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def technical_failure_gate_v4(failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(failures), "SetFlow V4 technical failure gate has no failure")
    for failure in failures:
        _require(
            int(failure.get("development_test_outcome_reads", -1)) == 0
            and int(failure.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "SetFlow V4 technical failure artifact has unauthorized protected reads",
        )
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_screen_gate.v1",
        "status": "XEDITSETFLOW_V4_SCREEN_NO_GO",
        "screen_seed": 20260911,
        "reason": "ONE_OR_MORE_FROZEN_TRAINING_OR_CHECKPOINT_VALIDATION_RUNS_FAILED_TECHNICALLY",
        "technical_failures": [dict(row) for row in failures],
        "selected_checkpoint_pass": None,
        "confirmation_authorized": False,
        "confirmation_seeds": [],
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
