"""Strict frozen checkpoint selection and screen gate for SetFlow V4 S1."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Mapping


from core.route2_xedit_v4_interfaces import SetFlowCheckpointDecisionV4
from core.route2_xeditsetflow_gate_v4 import (
    XEditSetFlowGateV4Error,
    paired_bootstrap_recovery_improvement_v4,
)


OBJECTIVE_IDENTITY = "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY"
OBJECTIVE_WEIGHT = 0.05
CONFIRMATION_SEEDS = (20260912, 20260913, 20260914)
SCREEN_RUNNER_GIT_HEAD = "ebf99ebf8a253ad27e311e555121d328df8fae10"
CONFIRMATION_BOOTSTRAP_REPLICATES = 10_000
CONFIRMATION_BOOTSTRAP_SEED = 2026091102
CONFIRMATION_BOOTSTRAP_STATISTIC = (
    "SOURCE_MACRO_CANDIDATE_RECOVERY_DIFFERENCE_V4_S1_FULL_MINUS_TERMINAL_F2"
)
MATCHED_INITIALIZATION_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_matched_initialization.v1"
)
MATCHED_INITIALIZATION_DIGEST_ALGORITHM = "sha256"
MATCHED_INITIALIZATION_DIGEST_INPUT_FIELDS = ["name", "dtype", "shape", "bytes"]
MATCHED_INITIALIZATION_ROUTER_PROJECTION = {
    "mode_router.weight": "canonical_full.mode_router.weight[0:1]",
    "mode_router.bias": "canonical_full.mode_router.bias[0:1]",
}
MATCHED_INITIALIZATION_MODEL_ROLES = {
    "v4_s1_full": "CANONICAL_FULL_MODEL",
    "v4_s1_single_mode": "PROJECTED_FROM_CANONICAL_FULL_MODE_ZERO",
}


class XEditSetFlowGateS1Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowGateS1Error(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def require_matched_initialization_evidence_s1(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate the fail-closed, cross-process matched-initialization record."""

    evidence = payload.get("matched_initialization")
    _require(isinstance(evidence, Mapping), f"{label} matched initialization is absent")
    _require(
        evidence.get("schema_version") == MATCHED_INITIALIZATION_SCHEMA
        and evidence.get("canonical_run_id") == "v4_s1_full"
        and evidence.get("projection_target_run_id") == "v4_s1_single_mode"
        and evidence.get("model_roles") == MATCHED_INITIALIZATION_MODEL_ROLES,
        f"{label} matched-initialization identity changed",
    )
    digest = evidence.get("canonical_state_digest")
    _require(
        evidence.get("canonical_state_digest_algorithm")
        == MATCHED_INITIALIZATION_DIGEST_ALGORITHM
        and evidence.get("canonical_state_digest_input_fields")
        == MATCHED_INITIALIZATION_DIGEST_INPUT_FIELDS
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"{label} canonical initialization digest is invalid",
    )
    count_names = (
        "canonical_state_tensor_count",
        "canonical_state_element_count",
        "comparable_parameter_tensor_count",
        "comparable_parameter_element_count",
        "comparable_buffer_tensor_count",
        "comparable_buffer_element_count",
        "comparable_tensor_count",
        "comparable_element_count",
        "full_only_state_tensor_count",
        "full_only_state_element_count",
    )
    counts: dict[str, int] = {}
    for name in count_names:
        value = evidence.get(name)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label} matched-initialization count is invalid: {name}",
        )
        counts[name] = value
    _require(
        counts["comparable_parameter_tensor_count"] > 0
        and counts["comparable_parameter_element_count"] > 0
        and counts["comparable_tensor_count"]
        == counts["comparable_parameter_tensor_count"]
        + counts["comparable_buffer_tensor_count"]
        and counts["comparable_element_count"]
        == counts["comparable_parameter_element_count"]
        + counts["comparable_buffer_element_count"]
        and counts["canonical_state_tensor_count"]
        >= counts["comparable_tensor_count"]
        and counts["canonical_state_element_count"]
        >= counts["comparable_element_count"]
        and counts["canonical_state_tensor_count"]
        == counts["comparable_tensor_count"]
        + counts["full_only_state_tensor_count"]
        and counts["canonical_state_element_count"]
        == counts["comparable_element_count"]
        + counts["full_only_state_element_count"],
        f"{label} matched-initialization count geometry changed",
    )
    _require(
        evidence.get("router_projection") == MATCHED_INITIALIZATION_ROUTER_PROJECTION
        and evidence.get("unmapped_target_names") == []
        and evidence.get("mismatched_target_names") == []
        and evidence.get("all_equal") is True,
        f"{label} comparable initialization is not exactly matched",
    )
    return copy.deepcopy(dict(evidence))


def _require_compute_identity_s1(
    summary: Mapping[str, Any], *, mode_count: int
) -> None:
    compute = summary.get("compute")
    _require(isinstance(compute, Mapping), "SetFlow V4 S1 compute record is absent")
    _require(
        int(compute.get("trajectory_count", -1)) == 28_512
        and int(compute.get("candidate_count", -1)) == 28_512
        and int(compute.get("critic_forward_count", -1)) == 0
        and int(compute.get("independent_evaluator_forward_count", -1)) == 0,
        "SetFlow V4 S1 compute budget or prohibited forward count changed",
    )
    common_states = int(compute.get("common_nll_trunk_forward_state_count", -1))
    _require(
        int(compute.get("common_nll_trunk_forward_batch_count", -1)) == 996
        and common_states == 31_848
        and int(compute.get("common_nll_mode_head_forward_state_count", -1))
        == common_states * mode_count,
        "SetFlow V4 S1 common-NLL mode-head compute is undercounted",
    )
    for section in ("root_prior", "primary_generation", "replay_generation"):
        row = compute.get(section)
        _require(isinstance(row, Mapping), f"SetFlow V4 S1 {section} compute is absent")
        trunk_states = int(row.get("trunk_forward_state_count", -1))
        _require(
            int(row.get("trunk_forward_batch_count", -1)) > 0
            and trunk_states > 0
            and int(row.get("mode_head_forward_state_count", -1))
            == trunk_states * mode_count,
            f"SetFlow V4 S1 {section} mode-head compute is undercounted",
        )
    root = compute["root_prior"]
    primary = compute["primary_generation"]
    replay = compute["replay_generation"]
    _require(
        int(root["trunk_forward_batch_count"]) == 14
        and int(root["trunk_forward_state_count"]) == 891,
        "SetFlow V4 S1 root-prior compute differs from 891 sources at batch64",
    )
    _require(
        int(primary["trunk_forward_state_count"])
        == int(replay["trunk_forward_state_count"]),
        "SetFlow V4 S1 exact replay compute differs from primary generation",
    )


def validate_checkpoint_summary_identity_s1(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    checkpoint_pass: int,
    expected_seed: int = 20260911,
    expected_run_stage: str = "SCREEN",
) -> dict[str, Any]:
    mode_count = 8 if run_id == "v4_s1_full" else 1
    selectable = run_id == "v4_s1_full"
    matched_initialization = require_matched_initialization_evidence_s1(
        summary,
        label=f"SetFlow V4 S1 {run_id} pass {checkpoint_pass} Validation",
    )
    _require(
        summary.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_checkpoint_validation.v1"
        and summary.get("status")
        == "TERMINAL_XEDITSETFLOW_V4_S1_CHECKPOINT_VALIDATION_COMPLETE",
        "SetFlow V4 S1 checkpoint validation is not terminal complete",
    )
    _require(
        summary.get("run_id") == run_id
        and summary.get("run_stage") == expected_run_stage
        and int(summary.get("checkpoint_pass", -1)) == checkpoint_pass
        and int(summary.get("seed", -1)) == expected_seed
        and int(summary.get("mode_count", -1)) == mode_count
        and summary.get("selectable") is selectable,
        "SetFlow V4 S1 checkpoint validation identity changed",
    )
    _require(
        summary.get("objective_identity") == OBJECTIVE_IDENTITY
        and _finite(
            summary.get("cross_state_candidate_mode_responsibility_weight"),
            "S1 responsibility weight",
        )
        == OBJECTIVE_WEIGHT
        and isinstance(summary.get("active_responsibility_constraint_count"), int)
        and not isinstance(summary.get("active_responsibility_constraint_count"), bool)
        and int(summary["active_responsibility_constraint_count"]) >= 0,
        "SetFlow V4 S1 objective identity, weight, or constraint count changed",
    )
    _require(
        int(summary.get("source_count", -1)) == 891
        and int(summary.get("trajectory_count", -1)) == 28_512
        and int(summary.get("candidate_count", -1)) == 28_512
        and int(summary.get("candidate_cap_per_source", -1)) == 32
        and int(summary.get("duplicate_retry_or_rejection_count", -1)) == 0,
        "SetFlow V4 S1 source, trajectory, candidate, or retry budget changed",
    )
    _require(
        summary.get("training_summary_status")
        == "TERMINAL_XEDITSETFLOW_V4_S1_TRAINING_COMPLETE_PENDING_VALIDATION"
        and summary.get("precision") == "BF16"
        and summary.get("cpu_fallback_used") is False
        and int(summary.get("parameter_update_count", -1)) == 0,
        "SetFlow V4 S1 checkpoint validation provenance changed",
    )
    if expected_run_stage == "CONFIRMATION":
        _require(
            summary.get("selected_model") == "v4_s1_full"
            and summary.get("parameter_initialization_seed") == expected_seed
            and summary.get(
                "parameter_initialization_seed_applied_before_model_construction"
            )
            is True,
            "SetFlow V4 S1 confirmation parameter initialization provenance changed",
        )
        physical_gpu_index = summary.get("physical_gpu_index")
        _require(
            isinstance(physical_gpu_index, int)
            and not isinstance(physical_gpu_index, bool)
            and physical_gpu_index in range(6)
            and summary.get("torch_device") == f"cuda:{physical_gpu_index}"
            and "A100" in str(summary.get("device_name", ""))
            and summary.get("precision") == "BF16"
            and summary.get("cuda_available") is True
            and summary.get("bf16_supported") is True
            and summary.get("cpu_fallback_used") is False
            and summary.get("cuda_device_index") == physical_gpu_index
            and "A100" in str(summary.get("cuda_device_name", ""))
            and bool(summary.get("cuda_device_uuid"))
            and bool(summary.get("declared_physical_gpu_uuid"))
            and summary.get("cuda_parent_uuid_matches_declared_physical_index")
            is True,
            "SetFlow V4 S1 confirmation Validation CUDA/A100/BF16 evidence changed",
        )
        training_device = str(summary.get("training_torch_device", ""))
        _require(
            re.fullmatch(r"cuda:[0-5]", training_device) is not None
            and "A100" in str(summary.get("training_device_name", ""))
            and summary.get("training_precision") == "BF16"
            and summary.get("training_cuda_available") is True
            and summary.get("training_bf16_supported") is True
            and summary.get("training_cpu_fallback_used") is False,
            "SetFlow V4 S1 confirmation training CUDA/A100/BF16 evidence changed",
        )
    _require(
        summary.get("g0_status")
        in {"FLOW_G0_READY", "FLOW_G0_VALIDATION_FAIL"}
        and _finite(summary.get("wall_time_seconds"), "wall time") > 0.0
        and int(summary.get("peak_vram_bytes", 0)) > 0,
        "SetFlow V4 S1 G0 status, wall time, or peak VRAM is absent",
    )
    _require(
        summary.get("critic_used") is False
        and summary.get("independent_evaluator_used") is False
        and int(summary.get("development_test_outcome_reads", -1)) == 0
        and int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "SetFlow V4 S1 checkpoint validation used prohibited information",
    )
    common = summary.get("common_validation")
    _require(
        isinstance(common, Mapping)
        and int(common.get("validation_candidate_record_count", -1)) == 15_924
        and int(common.get("validation_states_per_record", -1)) == 2,
        "SetFlow V4 S1 common Validation cohort changed",
    )
    allocations = summary.get("aggregate_mode_allocations")
    _require(isinstance(allocations, Mapping), "SetFlow V4 S1 mode allocations are absent")
    expected_keys = {str(index) for index in range(mode_count)}
    _require(set(allocations) == expected_keys, "SetFlow V4 S1 mode allocation keys changed")
    _require(
        sum(int(value) for value in allocations.values()) == 28_512
        and all(int(value) >= 891 for value in allocations.values()),
        "SetFlow V4 S1 omitted a mode or changed trajectory budget",
    )
    small_graph = summary.get("small_graph_reference")
    _require(
        isinstance(small_graph, Mapping)
        and small_graph.get("status") == "PASS"
        and int(small_graph.get("mode_count", -1)) == mode_count
        and _finite(small_graph.get("total_variation"), "small-graph TV")
        <= _finite(small_graph.get("tolerance"), "small-graph tolerance"),
        "SetFlow V4 S1 small-graph mixture exactness is absent or failed",
    )
    _require_compute_identity_s1(summary, mode_count=mode_count)
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
        "SetFlow V4 S1 common-NLL summary fields disagree",
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
    correctness_ready = all(
        checks[name]
        for name in (
            "hard_legality_100pct",
            "edit_budget_violation_zero",
            "candidate_budget_violation_zero",
            "trajectory_replay_failure_zero",
            "numerical_failure_zero",
            "small_graph_exact",
        )
    )
    _require(
        summary.get("g0_status")
        == ("FLOW_G0_READY" if correctness_ready else "FLOW_G0_VALIDATION_FAIL"),
        "SetFlow V4 S1 G0 status disagrees with terminal correctness checks",
    )
    return {
        "run_id": run_id,
        "checkpoint_pass": checkpoint_pass,
        "matched_initialization": matched_initialization,
        "common_validation_set_marginal_nll": nll,
        "source_macro_candidate_recovery_rate": recovery,
        "source_macro_measured_top_k_recovery_at_k": top_k,
        "source_macro_unique_candidate_rate": unique,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def select_checkpoint_s1(
    rows: Mapping[int, Mapping[str, Any]]
) -> SetFlowCheckpointDecisionV4:
    _require(set(rows) == {4, 6, 8, 10}, "SetFlow V4 S1 checkpoint set is incomplete")
    eligible = [
        dict(rows[checkpoint_pass])
        for checkpoint_pass in (4, 6, 8, 10)
        if rows[checkpoint_pass]["eligible"]
    ]
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


def adjudicate_setflow_screen_s1(
    config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    _require(
        set(summaries) == {"v4_s1_full", "v4_s1_single_mode"},
        "SetFlow V4 S1 screen runs are incomplete",
    )
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    selections: dict[str, dict[str, Any]] = {}
    matched_initialization_by_run: dict[str, dict[str, Any]] = {}
    for run_id in ("v4_s1_full", "v4_s1_single_mode"):
        _require(
            set(summaries[run_id]) == {4, 6, 8, 10},
            f"SetFlow V4 S1 checkpoint validations are incomplete: {run_id}",
        )
        rows[run_id] = {
            checkpoint_pass: validate_checkpoint_summary_identity_s1(
                summaries[run_id][checkpoint_pass],
                run_id=run_id,
                checkpoint_pass=checkpoint_pass,
            )
            for checkpoint_pass in (4, 6, 8, 10)
        }
        matched_initialization_by_run[run_id] = copy.deepcopy(
            rows[run_id][4]["matched_initialization"]
        )
        _require(
            all(
                rows[run_id][checkpoint_pass]["matched_initialization"]
                == matched_initialization_by_run[run_id]
                for checkpoint_pass in (4, 6, 8, 10)
            ),
            f"SetFlow V4 S1 matched-initialization evidence drifted across checkpoints: {run_id}",
        )
        selections[run_id] = select_checkpoint_s1(rows[run_id])
    _require(
        matched_initialization_by_run["v4_s1_full"]
        == matched_initialization_by_run["v4_s1_single_mode"],
        "SetFlow V4 S1 canonical initialization differs across full and single-mode processes",
    )
    full = selections["v4_s1_full"]["generation_constrained_selected_checkpoint"]
    single = selections["v4_s1_single_mode"][
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
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1",
        "status": "XEDITSETFLOW_V4_S1_SCREEN_PASS"
        if passed
        else "XEDITSETFLOW_V4_S1_SCREEN_NO_GO",
        "screen_seed": 20260911,
        "matched_initialization": copy.deepcopy(
            matched_initialization_by_run["v4_s1_full"]
        ),
        "checkpoint_rows": rows,
        "checkpoint_decisions": selections,
        "terminal_f2_reference": dict(reference),
        "screen_checks": checks,
        "selected_checkpoint_pass": None if full is None else full["checkpoint_pass"],
        "legacy_v4_confirmation_authorized": False,
        "confirmation_authorized": False,
        "confirmation_seeds": [],
        "successor_protocol_required": passed,
        "s1_mechanics_screen_passed": passed,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _validate_terminal_f2_reference_s1(
    terminal_f2_summary: Mapping[str, Any],
) -> None:
    _require(
        terminal_f2_summary.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_unguided_validation.v3"
        and terminal_f2_summary.get("status") == "FLOW_G0_READY"
        and terminal_f2_summary.get("arm") == "f2"
        and int(terminal_f2_summary.get("seed", -1)) == 20260903
        and int(terminal_f2_summary.get("source_count", -1)) == 891
        and int(terminal_f2_summary.get("candidate_count", -1)) == 28_512,
        "terminal F2 reference identity changed",
    )
    _require(
        terminal_f2_summary.get("development_test_outcomes_accessed") is False
        and int(terminal_f2_summary.get("evaluation_records_read", -1)) == 0
        and terminal_f2_summary.get("evaluation_outcomes_accessed") is False
        and terminal_f2_summary.get("guided_critic_used") is False
        and terminal_f2_summary.get("independent_evaluator_used") is False,
        "terminal F2 reference used prohibited outcomes or evaluators",
    )


def _validate_confirmation_config_s1(
    config: Mapping[str, Any], *, seed: int
) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_runtime.v1"
        and config.get("status") == "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED"
        and config.get("run_stage") == "CONFIRMATION"
        and int(config.get("training_seed", -1)) == seed
        and config.get("selected_model") == "v4_s1_full"
        and config.get("required_confirmation_seeds")
        == list(CONFIRMATION_SEEDS)
        and config.get("additional_seed_authorized") is False,
        f"SetFlow V4 S1 confirmation config changed: {seed}",
    )
    _require(
        config.get("objective_identity") == OBJECTIVE_IDENTITY
        and _finite(
            config.get("cross_state_candidate_mode_responsibility_weight"),
            "S1 confirmation responsibility weight",
        )
        == OBJECTIVE_WEIGHT
        and config.get("screen_runner_git_head") == SCREEN_RUNNER_GIT_HEAD
        and isinstance(config.get("screen_gate_path"), str)
        and bool(config.get("screen_gate_path"))
        and int(config.get("screen_selected_checkpoint_pass", -1))
        in {4, 6, 8, 10},
        f"SetFlow V4 S1 confirmation lineage changed: {seed}",
    )
    confirmation_runner_git_head = config.get("confirmation_runner_git_head")
    output_root = Path(str(config.get("output_root", "")))
    validation_output_root = Path(str(config.get("validation_output_root", "")))
    screen_provenance = config.get("screen_provenance")
    _require(
        isinstance(confirmation_runner_git_head, str)
        and re.fullmatch(r"[0-9a-f]{40}", confirmation_runner_git_head) is not None
        and output_root.is_absolute()
        and validation_output_root.is_absolute()
        and isinstance(screen_provenance, Mapping)
        and screen_provenance.get("screen_runner_git_head")
        == SCREEN_RUNNER_GIT_HEAD
        and screen_provenance.get("screen_gate_path")
        == config.get("screen_gate_path")
        and int(screen_provenance.get("screen_selected_checkpoint_pass", -1))
        == int(config.get("screen_selected_checkpoint_pass", -2)),
        f"SetFlow V4 S1 confirmation runner or screen provenance changed: {seed}",
    )
    _require(
        config.get("development_test_outcomes_accessed") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False
        and int(config.get("development_test_outcome_reads", -1)) == 0
        and int(config.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"SetFlow V4 S1 confirmation config reports a protected read: {seed}",
    )


def adjudicate_setflow_confirmation_s1(
    configs: Mapping[int, Mapping[str, Any]],
    summaries: Mapping[int, Mapping[int, Mapping[str, Any]]],
    terminal_f2_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Adjudicate the frozen three-seed, full-only S1 confirmation package."""

    _validate_terminal_f2_reference_s1(terminal_f2_summary)
    _require(
        set(configs) == set(summaries) == set(CONFIRMATION_SEEDS),
        "SetFlow V4 S1 confirmation seed cohort is incomplete or contains an extra seed",
    )
    _require(
        len(
            {
                (
                    config.get("screen_gate_path"),
                    config.get("screen_runner_git_head"),
                    config.get("screen_selected_checkpoint_pass"),
                )
                for config in configs.values()
            }
        )
        == 1,
        "SetFlow V4 S1 confirmation configs disagree on frozen screen lineage",
    )
    first_config = configs[CONFIRMATION_SEEDS[0]]
    first_screen_provenance = first_config.get("screen_provenance")
    first_runner_head = first_config.get("confirmation_runner_git_head")
    _require(
        all(
            config.get("screen_provenance") == first_screen_provenance
            and config.get("confirmation_runner_git_head") == first_runner_head
            for config in configs.values()
        ),
        "SetFlow V4 S1 confirmation configs disagree on full screen or runner provenance",
    )
    seed_results: dict[str, Any] = {}
    all_passed = True
    validation_git_heads: set[str] = set()
    training_git_heads: set[str] = set()
    for seed in CONFIRMATION_SEEDS:
        config = configs[seed]
        _validate_confirmation_config_s1(config, seed=seed)
        _require(
            set(summaries[seed]) == {4, 6, 8, 10},
            f"SetFlow V4 S1 confirmation checkpoint package is incomplete: {seed}",
        )
        training_root = Path(str(config["output_root"])) / "v4_s1_full"
        validation_root = Path(str(config["validation_output_root"])) / "v4_s1_full"
        expected_training_head = str(config["confirmation_runner_git_head"])
        rows: dict[int, dict[str, Any]] = {}
        for checkpoint_pass in (4, 6, 8, 10):
            summary = summaries[seed][checkpoint_pass]
            expected_checkpoint_path = str(
                training_root / f"pass_{checkpoint_pass}.pt"
            )
            expected_training_summary_path = str(
                training_root / "training_summary.json"
            )
            expected_validation_summary_path = str(
                validation_root
                / f"pass_{checkpoint_pass}"
                / "validation_summary.json"
            )
            _require(
                summary.get("checkpoint_path") == expected_checkpoint_path
                and summary.get("training_summary_path")
                == expected_training_summary_path
                and summary.get("validation_summary_path")
                == expected_validation_summary_path,
                f"SetFlow V4 S1 confirmation artifact path changed: {seed} pass {checkpoint_pass}",
            )
            training_git_head = summary.get("training_git_head")
            validation_git_head = summary.get("validation_git_head")
            _require(
                training_git_head == expected_training_head
                and isinstance(validation_git_head, str)
                and re.fullmatch(r"[0-9a-f]{40}", validation_git_head) is not None
                and summary.get("training_and_validation_git_heads_differ")
                == (training_git_head != validation_git_head),
                f"SetFlow V4 S1 confirmation training/Validation Git lineage changed: {seed} pass {checkpoint_pass}",
            )
            training_git_heads.add(training_git_head)
            validation_git_heads.add(validation_git_head)
            rows[checkpoint_pass] = validate_checkpoint_summary_identity_s1(
                summary,
                run_id="v4_s1_full",
                checkpoint_pass=checkpoint_pass,
                expected_seed=seed,
                expected_run_stage="CONFIRMATION",
            )
        decision = select_checkpoint_s1(rows)
        selected = decision["generation_constrained_selected_checkpoint"]
        checks = {"has_eligible_checkpoint": selected is not None}
        bootstrap = None
        selected_checkpoint_path = None
        selected_validation_summary_path = None
        selected_training_summary_path = None
        if selected is not None:
            selected_summary = summaries[seed][selected["checkpoint_pass"]]
            selected_checkpoint_path = selected_summary.get("checkpoint_path")
            selected_validation_summary_path = selected_summary.get(
                "validation_summary_path"
            )
            selected_training_summary_path = selected_summary.get(
                "training_summary_path"
            )
            _require(
                all(
                    isinstance(value, str) and bool(value)
                    for value in (
                        selected_checkpoint_path,
                        selected_validation_summary_path,
                        selected_training_summary_path,
                    )
                ),
                f"SetFlow V4 S1 selected checkpoint lineage is absent: {seed}",
            )
            reference_recovery = _finite(
                terminal_f2_summary.get("source_macro_candidate_recovery_rate"),
                "terminal F2 recovery",
            )
            reference_top_k = _finite(
                terminal_f2_summary.get(
                    "source_macro_measured_top_k_recovery_at_k"
                ),
                "terminal F2 top-k recovery",
            )
            reference_unique = _finite(
                terminal_f2_summary.get("source_macro_unique_candidate_rate"),
                "terminal F2 unique rate",
            )
            try:
                reused_bootstrap = paired_bootstrap_recovery_improvement_v4(
                    selected_summary,
                    terminal_f2_summary,
                    replicates=CONFIRMATION_BOOTSTRAP_REPLICATES,
                    seed=CONFIRMATION_BOOTSTRAP_SEED,
                )
            except XEditSetFlowGateV4Error as error:
                raise XEditSetFlowGateS1Error(str(error)) from error
            _require(
                reused_bootstrap.get("replicates")
                == CONFIRMATION_BOOTSTRAP_REPLICATES
                and reused_bootstrap.get("seed") == CONFIRMATION_BOOTSTRAP_SEED
                and reused_bootstrap.get("statistic")
                == "SOURCE_MACRO_CANDIDATE_RECOVERY_DIFFERENCE_V4_MINUS_TERMINAL_F2",
                "SetFlow V4 S1 reused bootstrap numeric contract changed",
            )
            bootstrap = {
                **reused_bootstrap,
                "statistic": CONFIRMATION_BOOTSTRAP_STATISTIC,
            }
            checks.update(
                {
                    "recovery_margin_over_terminal_f2_at_least_0_05": selected[
                        "source_macro_candidate_recovery_rate"
                    ]
                    - reference_recovery
                    >= 0.05,
                    "top_k_margin_over_terminal_f2_at_least_0_03": selected[
                        "source_macro_measured_top_k_recovery_at_k"
                    ]
                    - reference_top_k
                    >= 0.03,
                    "unique_margin_over_terminal_f2_at_least_0_15": selected[
                        "source_macro_unique_candidate_rate"
                    ]
                    - reference_unique
                    >= 0.15,
                    "paired_bootstrap_recovery_ci_lower_bound_positive": bootstrap[
                        "ci_lower_bound_strictly_greater_than_zero"
                    ],
                    "s1_responsibility_constraints_active": int(
                        selected_summary.get(
                            "active_responsibility_constraint_count", 0
                        )
                    )
                    > 0,
                }
            )
        else:
            checks.update(
                {
                    "recovery_margin_over_terminal_f2_at_least_0_05": False,
                    "top_k_margin_over_terminal_f2_at_least_0_03": False,
                    "unique_margin_over_terminal_f2_at_least_0_15": False,
                    "paired_bootstrap_recovery_ci_lower_bound_positive": False,
                    "s1_responsibility_constraints_active": False,
                }
            )
        passed = all(checks.values())
        all_passed = all_passed and passed
        seed_results[str(seed)] = {
            "checkpoint_rows": rows,
            "checkpoint_decision": decision,
            "selected_checkpoint_pass": None
            if selected is None
            else selected["checkpoint_pass"],
            "selected_checkpoint_path": selected_checkpoint_path,
            "selected_validation_summary_path": selected_validation_summary_path,
            "selected_training_summary_path": selected_training_summary_path,
            "paired_bootstrap_recovery_improvement": bootstrap,
            "checks": checks,
            "passed": passed,
        }
    _require(
        training_git_heads == {first_runner_head}
        and len(validation_git_heads) == 1,
        "SetFlow V4 S1 confirmation package Git lineage is inconsistent across 12 validations",
    )
    validation_git_head = next(iter(validation_git_heads))
    return {
        "schema_version": "route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1",
        "status": "XEDITSETFLOW_V4_G0_READY"
        if all_passed
        else "XEDITSETFLOW_V4_CONFIRMATION_NO_GO",
        "required_seeds": list(CONFIRMATION_SEEDS),
        "selected_model": "v4_s1_full",
        "objective_identity": OBJECTIVE_IDENTITY,
        "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
        "screen_runner_git_head": SCREEN_RUNNER_GIT_HEAD,
        "screen_gate_path": configs[CONFIRMATION_SEEDS[0]]["screen_gate_path"],
        "screen_selected_checkpoint_pass": configs[CONFIRMATION_SEEDS[0]][
            "screen_selected_checkpoint_pass"
        ],
        "screen_provenance": copy.deepcopy(first_screen_provenance),
        "confirmation_runner_git_head": first_runner_head,
        "training_git_head": first_runner_head,
        "validation_git_head": validation_git_head,
        "training_and_validation_git_heads_differ": (
            first_runner_head != validation_git_head
        ),
        "seed_results": seed_results,
        "additional_seed_authorized": False,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "critic_used": False,
        "independent_evaluator_used": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "claim_boundary": "SETFLOW_S1_COMPONENT_READINESS_ONLY_NOT_FINAL_SCIENTIFIC_EVIDENCE",
    }
