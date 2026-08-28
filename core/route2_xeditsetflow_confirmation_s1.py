"""Frozen S1 confirmation config derivation for the single canonical launcher.

This module has no command-line entry point.  The confirmation launcher must
provide the exact tracked screen artifacts; no directory discovery is used.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.route2_xeditsetflow_gate_s1 import select_checkpoint_s1


SCREEN_CONFIG_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_config.v1"
)
CONFIRMATION_PROTOCOL_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_protocol.v1"
)
CONFIRMATION_RUNTIME_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_runtime.v1"
)
CONFIRMATION_PROTOCOL_STATUS = (
    "FROZEN_PROSPECTIVE_BEFORE_S1_SCREEN_TERMINAL_OR_CONFIRMATION_OUTCOME_READ"
)
CONFIRMATION_RUNTIME_STATUS = "FROZEN_S1_CONFIRMATION_CONFIG_NOT_STARTED"
OBJECTIVE_IDENTITY = (
    "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY"
)
OBJECTIVE_WEIGHT = 0.05
SCREEN_HEAD = "930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8"
SCREEN_RUN_IDS = ("v4_s1_full", "v4_s1_single_mode")
CONFIRMATION_RUN_ID = "v4_s1_full"
CONFIRMATION_SEEDS = (20260912, 20260913, 20260914)
CHECKPOINT_PASSES = (4, 6, 8, 10)


class XEditSetFlowConfirmationS1Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowConfirmationS1Error(message)


def _protected_reads_zero(payload: Mapping[str, Any], label: str) -> None:
    _require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a protected outcome read",
    )


def _format_path(template: Any, runner_git_head: str) -> Path:
    return Path(str(template).format(runner_git_head=runner_git_head))


def _jobs(queues: Any, label: str) -> list[Mapping[str, Any]]:
    _require(isinstance(queues, list), f"{label} queues are absent")
    result: list[Mapping[str, Any]] = []
    for queue in queues:
        _require(isinstance(queue, Mapping), f"{label} queue is invalid")
        jobs = queue.get("jobs")
        _require(isinstance(jobs, list), f"{label} job list is absent")
        _require(
            all(isinstance(job, Mapping) for job in jobs),
            f"{label} job row is invalid",
        )
        result.extend(jobs)
    return result


def _normalize_checkpoint_rows(rows: Any, run_id: str) -> dict[int, dict[str, Any]]:
    _require(isinstance(rows, Mapping), f"screen checkpoint rows are absent: {run_id}")
    normalized: dict[int, dict[str, Any]] = {}
    for key, value in rows.items():
        _require(isinstance(value, Mapping), f"screen checkpoint row is invalid: {run_id}")
        try:
            checkpoint_pass = int(key)
        except (TypeError, ValueError) as error:
            raise XEditSetFlowConfirmationS1Error(
                f"screen checkpoint pass is invalid: {run_id}"
            ) from error
        normalized[checkpoint_pass] = dict(value)
    _require(
        set(normalized) == set(CHECKPOINT_PASSES),
        f"screen checkpoint rows are incomplete: {run_id}",
    )
    return normalized


def _validate_protocol_and_base(
    base: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    _require(
        base.get("schema_version") == SCREEN_CONFIG_SCHEMA,
        "unexpected SetFlow V4 S1 screen config",
    )
    _require(
        protocol.get("schema_version") == CONFIRMATION_PROTOCOL_SCHEMA
        and protocol.get("status") == CONFIRMATION_PROTOCOL_STATUS,
        "SetFlow V4 S1 confirmation protocol is not prospectively frozen",
    )
    _require(
        protocol.get("selected_model") == CONFIRMATION_RUN_ID
        and tuple(protocol.get("required_seeds", ())) == CONFIRMATION_SEEDS
        and protocol.get("additional_seed_authorized") is False,
        "SetFlow V4 S1 confirmation model or seed cohort changed",
    )
    design = protocol.get("confirmation_design")
    _require(
        isinstance(design, Mapping)
        and int(design.get("training_job_count", -1)) == 3
        and design.get("training_run_ids") == [CONFIRMATION_RUN_ID] * 3
        and int(design.get("single_mode_training_job_count", -1)) == 0
        and int(design.get("checkpoint_validation_job_count", -1)) == 12
        and design.get("checkpoint_validation_passes_per_seed")
        == list(CHECKPOINT_PASSES)
        and design.get("screen_full_and_single_mode_decisions_role")
        == "BOUND_PROVENANCE_ONLY"
        and design.get("screen_selected_checkpoint_pass_role")
        == "BOUND_PROVENANCE_ONLY"
        and design.get("screen_checkpoint_never_substitutes_for_confirmation_checkpoint")
        is True,
        "SetFlow V4 S1 confirmation job geometry or provenance role changed",
    )
    objective = protocol.get("objective")
    base_objective = base.get("objective")
    _require(
        isinstance(objective, Mapping)
        and isinstance(base_objective, Mapping)
        and objective.get("identity") == OBJECTIVE_IDENTITY
        and base_objective.get("identity") == OBJECTIVE_IDENTITY
        and float(objective.get("cross_state_candidate_mode_responsibility_weight", -1.0))
        == OBJECTIVE_WEIGHT
        and float(base_objective.get("cross_state_candidate_mode_responsibility_weight", -1.0))
        == OBJECTIVE_WEIGHT
        and objective.get("weight_sweep_authorized") is False
        and base_objective.get("cross_state_candidate_mode_responsibility_weight_sweep")
        is False,
        "SetFlow V4 S1 confirmation objective identity or fixed weight changed",
    )
    training = base.get("training")
    policy = protocol.get("training_policy")
    _require(
        isinstance(training, Mapping)
        and isinstance(policy, Mapping)
        and int(training.get("pass_count", -1)) == int(policy.get("passes", -1)) == 10
        and training.get("saved_checkpoint_passes")
        == policy.get("saved_checkpoint_passes")
        == list(CHECKPOINT_PASSES)
        and float(training.get("learning_rate", -1.0))
        == float(policy.get("learning_rate", -2.0))
        and float(training.get("weight_decay", -1.0))
        == float(policy.get("weight_decay", -2.0))
        and float(training.get("gradient_clip_norm", -1.0))
        == float(policy.get("gradient_clip_norm", -2.0))
        and float(training.get("warmup_fraction", -1.0))
        == float(policy.get("warmup_fraction", -2.0))
        and training.get("decay") == policy.get("decay")
        and training.get("validation_generation_during_training") is False
        and policy.get("validation_generation_during_training") is False,
        "SetFlow V4 S1 confirmation training policy changed",
    )
    gpu = protocol.get("gpu_policy")
    _require(
        isinstance(gpu, Mapping)
        and gpu.get("physical_gpu_scope") == list(range(6))
        and gpu.get("cuda_bf16_only") is True
        and gpu.get("cpu_fallback") is False
        and gpu.get("free_or_estimated_memory_gate") is False
        and gpu.get("free_or_estimated_memory_sorting") is False,
        "SetFlow V4 S1 confirmation GPU or no-VRAM-gate policy changed",
    )
    failure = protocol.get("package_failure_policy")
    _require(
        isinstance(failure, Mapping)
        and failure.get("first_terminal_failure_stops_pending_launches") is True
        and failure.get("pending_status_after_failure")
        == "NOT_RUN_AFTER_TERMINAL_FAILURE"
        and failure.get("technical_failure_is_scientific_no_go") is False
        and failure.get("adjudication_after_incomplete_package") is False
        and failure.get("same_family_reuse_authorized") is False,
        "SetFlow V4 S1 confirmation package failure policy changed",
    )
    _protected_reads_zero(protocol, "confirmation protocol")


def validate_screen_pass_barrier_s1(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_schedule: Mapping[str, Any],
    screen_runtime_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    screen_schedule_path: Path,
    screen_runtime_config_path: Path,
    screen_authorization_path: Path,
    screen_gate_path: Path,
) -> dict[str, Any]:
    """Validate the exact terminal S1 PASS bundle without scanning its family."""

    _validate_protocol_and_base(base, protocol)
    provenance = protocol.get("screen_provenance")
    _require(isinstance(provenance, Mapping), "S1 screen provenance is absent")
    _require(
        provenance.get("screen_runner_git_head") == SCREEN_HEAD
        and Path(str(provenance.get("schedule_path"))) == screen_schedule_path
        and Path(str(provenance.get("runtime_path")))
        == Path(str(screen_schedule.get("runtime_manifest")))
        and Path(str(provenance.get("runtime_config_path")))
        == screen_runtime_config_path
        and Path(str(provenance.get("authorization_path")))
        == screen_authorization_path
        and Path(str(provenance.get("screen_gate_path"))) == screen_gate_path,
        "S1 screen HEAD or explicit artifact path changed",
    )
    _require(
        screen_schedule.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_screen_schedule.v1"
        and screen_schedule.get("status")
        == "FROZEN_XEDITSETFLOW_V4_S1_SCREEN_SCHEDULE"
        and screen_schedule.get("git_head") == SCREEN_HEAD
        and Path(str(screen_schedule.get("runtime_config")))
        == screen_runtime_config_path
        and Path(str(screen_schedule.get("authorization")))
        == screen_authorization_path,
        "S1 screen schedule HEAD or binding changed",
    )
    _require(
        screen_schedule.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            screen_schedule.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT,
        "S1 screen schedule objective provenance changed",
    )
    adjudication = screen_schedule.get("adjudication")
    _require(
        isinstance(adjudication, Mapping)
        and Path(str(adjudication.get("gate_path"))) == screen_gate_path,
        "S1 screen schedule gate path changed",
    )
    _protected_reads_zero(screen_schedule, "S1 screen schedule")
    _require(
        screen_runtime_config.get("schema_version") == SCREEN_CONFIG_SCHEMA
        and screen_runtime_config.get("run_stage") == "SCREEN"
        and screen_runtime_config.get("runner_git_head") == SCREEN_HEAD
        and Path(str(screen_runtime_config.get("screen_gate_output_path")))
        == screen_gate_path,
        "S1 screen runtime config HEAD, stage, or gate path changed",
    )
    runtime_objective = screen_runtime_config.get("objective")
    _require(
        isinstance(runtime_objective, Mapping)
        and runtime_objective.get("identity") == OBJECTIVE_IDENTITY
        and float(
            runtime_objective.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT,
        "S1 screen runtime objective provenance changed",
    )
    _require(
        screen_runtime_config.get("development_test_outcomes_accessed") is False
        and screen_runtime_config.get("new_final_evaluation_outcomes_accessed")
        is False,
        "S1 screen runtime config reports a protected outcome read",
    )
    _require(
        screen_authorization.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_screen_launch_authorization.v1"
        and screen_authorization.get("status")
        == "XEDITSETFLOW_V4_S1_SCREEN_LAUNCH_AUTHORIZED"
        and screen_authorization.get("authorized_git_head") == SCREEN_HEAD
        and screen_authorization.get("authorized_run_ids") == list(SCREEN_RUN_IDS)
        and int(screen_authorization.get("screen_seed", -1)) == 20260911
        and screen_authorization.get("objective_identity") == OBJECTIVE_IDENTITY
        and float(
            screen_authorization.get(
                "cross_state_candidate_mode_responsibility_weight", -1.0
            )
        )
        == OBJECTIVE_WEIGHT,
        "S1 screen authorization identity changed",
    )
    _protected_reads_zero(screen_authorization, "S1 screen authorization")
    _require(
        screen_gate.get("schema_version")
        == "route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1"
        and screen_gate.get("status") == "XEDITSETFLOW_V4_S1_SCREEN_PASS"
        and int(screen_gate.get("screen_seed", -1)) == 20260911
        and screen_gate.get("successor_protocol_required") is True
        and screen_gate.get("s1_mechanics_screen_passed") is True
        and screen_gate.get("confirmation_authorized") is False
        and screen_gate.get("legacy_v4_confirmation_authorized") is False
        and screen_gate.get("confirmation_seeds") == []
        and screen_gate.get("additional_seed_authorized") is False,
        "S1 screen gate is not the exact PASS-only successor barrier",
    )
    _protected_reads_zero(screen_gate, "S1 screen gate")

    rows_payload = screen_gate.get("checkpoint_rows")
    decisions = screen_gate.get("checkpoint_decisions")
    _require(
        isinstance(rows_payload, Mapping)
        and set(rows_payload) == set(SCREEN_RUN_IDS)
        and isinstance(decisions, Mapping)
        and set(decisions) == set(SCREEN_RUN_IDS),
        "S1 screen full/single checkpoint provenance is incomplete",
    )
    normalized_rows = {
        run_id: _normalize_checkpoint_rows(rows_payload[run_id], run_id)
        for run_id in SCREEN_RUN_IDS
    }
    checkpoint_check_keys = {
        "common_nll_at_most_2_06809",
        "recovery_at_least_0_35",
        "top_k_recovery_at_least_0_20",
        "unique_candidate_rate_at_least_0_90",
        "hard_legality_100pct",
        "edit_budget_violation_zero",
        "candidate_budget_violation_zero",
        "trajectory_replay_failure_zero",
        "numerical_failure_zero",
        "small_graph_exact",
    }
    for run_id, rows in normalized_rows.items():
        for checkpoint_pass, row in rows.items():
            checks = row.get("checks")
            _require(
                row.get("run_id") == run_id
                and int(row.get("checkpoint_pass", -1)) == checkpoint_pass
                and isinstance(checks, Mapping)
                and set(checks) == checkpoint_check_keys
                and all(isinstance(value, bool) for value in checks.values()),
                f"S1 screen checkpoint gate row identity changed: {run_id} pass {checkpoint_pass}",
            )
            expected_threshold_checks = {
                "common_nll_at_most_2_06809": float(
                    row["common_validation_set_marginal_nll"]
                )
                <= 2.06809,
                "recovery_at_least_0_35": float(
                    row["source_macro_candidate_recovery_rate"]
                )
                >= 0.35,
                "top_k_recovery_at_least_0_20": float(
                    row["source_macro_measured_top_k_recovery_at_k"]
                )
                >= 0.20,
                "unique_candidate_rate_at_least_0_90": float(
                    row["source_macro_unique_candidate_rate"]
                )
                >= 0.90,
            }
            _require(
                all(checks[key] is value for key, value in expected_threshold_checks.items())
                and row.get("eligible") is all(checks.values()),
                f"S1 screen checkpoint gate checks disagree with frozen rows: {run_id} pass {checkpoint_pass}",
            )
    recomputed_decisions = {
        run_id: select_checkpoint_s1(normalized_rows[run_id])
        for run_id in SCREEN_RUN_IDS
    }
    _require(
        dict(decisions) == recomputed_decisions,
        "S1 screen checkpoint decision provenance drifted from frozen rows",
    )
    full_selected = recomputed_decisions[CONFIRMATION_RUN_ID][
        "generation_constrained_selected_checkpoint"
    ]
    single_selected = recomputed_decisions["v4_s1_single_mode"][
        "generation_constrained_selected_checkpoint"
    ]
    _require(
        isinstance(full_selected, Mapping)
        and isinstance(single_selected, Mapping)
        and int(screen_gate.get("selected_checkpoint_pass", -1))
        == int(full_selected.get("checkpoint_pass", -2)),
        "S1 screen selected checkpoint provenance is inconsistent",
    )
    reference = base.get("terminal_f2_reference")
    _require(
        isinstance(reference, Mapping)
        and screen_gate.get("terminal_f2_reference") == dict(reference),
        "S1 screen terminal F2 reference provenance changed",
    )
    expected_screen_checks = {
        "full_has_eligible_checkpoint": True,
        "single_mode_has_eligible_checkpoint": True,
        "recovery_margin_over_terminal_f2_at_least_0_05": float(
            full_selected["source_macro_candidate_recovery_rate"]
        )
        - float(reference["source_macro_recovery"])
        >= 0.05,
        "top_k_margin_over_terminal_f2_at_least_0_03": float(
            full_selected["source_macro_measured_top_k_recovery_at_k"]
        )
        - float(reference["source_macro_top_k_recovery"])
        >= 0.03,
        "unique_margin_over_terminal_f2_at_least_0_15": float(
            full_selected["source_macro_unique_candidate_rate"]
        )
        - float(reference["source_macro_unique_candidate_rate"])
        >= 0.15,
        "recovery_margin_over_single_mode_at_least_0_03": float(
            full_selected["source_macro_candidate_recovery_rate"]
        )
        - float(single_selected["source_macro_candidate_recovery_rate"])
        >= 0.03,
        "unique_margin_over_single_mode_at_least_0_05": float(
            full_selected["source_macro_unique_candidate_rate"]
        )
        - float(single_selected["source_macro_unique_candidate_rate"])
        >= 0.05,
    }
    _require(
        screen_gate.get("screen_checks") == expected_screen_checks
        and all(expected_screen_checks.values()),
        "S1 screen PASS checks disagree with frozen rows or margins",
    )

    output_root = Path(str(screen_runtime_config.get("output_root")))
    validation_root = Path(str(screen_runtime_config.get("validation_output_root")))
    training_jobs = _jobs(screen_schedule.get("training_queues"), "screen training")
    _require(
        len(training_jobs) == 2
        and {str(job.get("run_id")) for job in training_jobs} == set(SCREEN_RUN_IDS),
        "S1 screen training schedule is not exact full plus single-mode",
    )
    for job in training_jobs:
        run_id = str(job["run_id"])
        _require(
            Path(str(job.get("terminal_summary")))
            == output_root / run_id / "training_summary.json",
            f"S1 screen training summary path changed: {run_id}",
        )
    validation_jobs = _jobs(
        screen_schedule.get("validation_queues"), "screen validation"
    )
    expected_validation_jobs = {
        (run_id, checkpoint_pass)
        for run_id in SCREEN_RUN_IDS
        for checkpoint_pass in CHECKPOINT_PASSES
    }
    observed_validation_jobs = {
        (str(job.get("run_id")), int(job.get("checkpoint_pass", -1)))
        for job in validation_jobs
    }
    _require(
        len(validation_jobs) == 8
        and observed_validation_jobs == expected_validation_jobs,
        "S1 screen Validation schedule is not the frozen eight jobs",
    )
    for job in validation_jobs:
        run_id = str(job["run_id"])
        checkpoint_pass = int(job["checkpoint_pass"])
        _require(
            Path(str(job.get("terminal_summary")))
            == validation_root
            / run_id
            / f"pass_{checkpoint_pass}"
            / "validation_summary.json",
            f"S1 screen Validation summary path changed: {run_id} pass {checkpoint_pass}",
        )
    return {
        "screen_runner_git_head": SCREEN_HEAD,
        "schedule_path": str(screen_schedule_path),
        "runtime_path": str(provenance["runtime_path"]),
        "runtime_config_path": str(screen_runtime_config_path),
        "authorization_path": str(screen_authorization_path),
        "screen_gate_path": str(screen_gate_path),
        "checkpoint_decisions": copy.deepcopy(recomputed_decisions),
        "screen_selected_checkpoint_pass": int(full_selected["checkpoint_pass"]),
        "role": "BOUND_SCREEN_FULL_AND_SINGLE_MODE_PROVENANCE_ONLY",
    }


def build_confirmation_configs_s1(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    screen_schedule: Mapping[str, Any],
    screen_runtime_config: Mapping[str, Any],
    screen_authorization: Mapping[str, Any],
    screen_gate: Mapping[str, Any],
    *,
    screen_schedule_path: Path,
    screen_runtime_config_path: Path,
    screen_authorization_path: Path,
    screen_gate_path: Path,
    confirmation_runner_git_head: str,
) -> list[dict[str, Any]]:
    """Build exactly three full-only confirmation runtime configs."""

    _require(
        re.fullmatch(r"[0-9a-f]{40}", confirmation_runner_git_head) is not None,
        "S1 confirmation runner Git HEAD is invalid",
    )
    screen_provenance = validate_screen_pass_barrier_s1(
        base,
        protocol,
        screen_schedule,
        screen_runtime_config,
        screen_authorization,
        screen_gate,
        screen_schedule_path=screen_schedule_path,
        screen_runtime_config_path=screen_runtime_config_path,
        screen_authorization_path=screen_authorization_path,
        screen_gate_path=screen_gate_path,
    )
    outputs = protocol["runner_outputs"]
    training_root = _format_path(
        outputs["training_runtime_root_template"], confirmation_runner_git_head
    )
    confirmation_gate_output = _format_path(
        outputs["confirmation_gate_output_template"], confirmation_runner_git_head
    )
    configs: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        seed_root = training_root / f"seed_{seed}"
        config = copy.deepcopy(dict(base))
        config.update(
            {
                "schema_version": CONFIRMATION_RUNTIME_SCHEMA,
                "status": CONFIRMATION_RUNTIME_STATUS,
                "run_stage": "CONFIRMATION",
                "training_seed": seed,
                "selected_model": CONFIRMATION_RUN_ID,
                "required_confirmation_seeds": list(CONFIRMATION_SEEDS),
                "additional_seed_authorized": False,
                "objective_identity": OBJECTIVE_IDENTITY,
                "cross_state_candidate_mode_responsibility_weight": OBJECTIVE_WEIGHT,
                "confirmation_runner_git_head": confirmation_runner_git_head,
                "screen_gate_path": str(screen_gate_path),
                "screen_runner_git_head": SCREEN_HEAD,
                "screen_selected_checkpoint_pass": screen_provenance[
                    "screen_selected_checkpoint_pass"
                ],
                "screen_provenance": copy.deepcopy(screen_provenance),
                "output_root": str(seed_root),
                "validation_output_root": str(
                    seed_root / "outcome_free_validation_generation"
                ),
                "confirmation_gate_output": str(confirmation_gate_output),
                "confirmation_training_job_count": 3,
                "confirmation_checkpoint_validation_job_count": 12,
                "development_test_outcomes_accessed": False,
                "new_final_evaluation_outcomes_accessed": False,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            }
        )
        configs.append(config)
    return configs


def materialize_confirmation_configs_s1(
    configs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    confirmation_runner_git_head: str,
) -> dict[str, Any]:
    """Atomically publish one fresh three-config package and never reuse it."""

    _require(
        re.fullmatch(r"[0-9a-f]{40}", confirmation_runner_git_head) is not None,
        "S1 confirmation runner Git HEAD is invalid",
    )
    _require(len(configs) == 3, "S1 confirmation must contain exactly three configs")
    seeds = [int(config.get("training_seed", -1)) for config in configs]
    _require(
        tuple(seeds) == CONFIRMATION_SEEDS
        and all(
            config.get("schema_version") == CONFIRMATION_RUNTIME_SCHEMA
            and config.get("status") == CONFIRMATION_RUNTIME_STATUS
            and config.get("run_stage") == "CONFIRMATION"
            and config.get("selected_model") == CONFIRMATION_RUN_ID
            and config.get("confirmation_runner_git_head")
            == confirmation_runner_git_head
            for config in configs
        ),
        "S1 confirmation config package identity changed",
    )
    outputs = protocol.get("runner_outputs")
    _require(isinstance(outputs, Mapping), "S1 confirmation output contract is absent")
    config_root = _format_path(
        outputs["runtime_config_root_template"], confirmation_runner_git_head
    )
    training_root = _format_path(
        outputs["training_runtime_root_template"], confirmation_runner_git_head
    )
    posttraining_root = _format_path(
        outputs["posttraining_runtime_root_template"], confirmation_runner_git_head
    )
    for path, label in (
        (config_root, "config root"),
        (training_root, "training family root"),
        (posttraining_root, "posttraining family root"),
    ):
        _require(not path.exists(), f"S1 confirmation {label} exists: {path}")
        partial = path.with_name(path.name + ".partial")
        _require(
            not partial.exists(),
            f"S1 confirmation partial {label} exists: {partial}",
        )
    for path, label in (
        (
            _format_path(
                outputs["authorization_output_template"],
                confirmation_runner_git_head,
            ),
            "authorization",
        ),
        (
            _format_path(
                outputs["prelaunch_failure_template"],
                confirmation_runner_git_head,
            ),
            "prelaunch failure",
        ),
    ):
        _require(not path.exists(), f"S1 confirmation {label} exists: {path}")
        partial = path.with_suffix(path.suffix + ".partial")
        _require(
            not partial.exists(),
            f"S1 confirmation partial {label} exists: {partial}",
        )
    staging = config_root.with_name(config_root.name + ".partial")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    config_paths: list[str] = []
    for config in configs:
        seed = int(config["training_seed"])
        final_path = config_root / f"seed_{seed}.json"
        (staging / final_path.name).write_text(
            json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_paths.append(str(final_path))
    manifest = {
        "schema_version": (
            "route_a_v3_route2_xeditsetflow_v4_s1_confirmation_config_manifest.v1"
        ),
        "status": "THREE_S1_CONFIRMATION_CONFIGS_PREPARED_NOT_STARTED",
        "confirmation_runner_git_head": confirmation_runner_git_head,
        "selected_model": CONFIRMATION_RUN_ID,
        "required_seeds": list(CONFIRMATION_SEEDS),
        "training_job_count": 3,
        "single_mode_training_job_count": 0,
        "checkpoint_validation_job_count": 12,
        "config_paths": config_paths,
        "development_test_authorized": False,
        "guidance_authorized": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, config_root)
    return manifest
