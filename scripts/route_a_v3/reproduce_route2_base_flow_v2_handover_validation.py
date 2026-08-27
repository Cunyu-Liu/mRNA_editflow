#!/usr/bin/env python3
"""Reproduce the frozen Base Flow V2 Development aggregates after Final terminal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3 import evaluate_route2_generation_v1 as evaluator


METHOD_ID = "unguided_learned_base_flow_g0"
FINAL_LAUNCH_SCHEMA = "route_a_v3_route2_xeditflow_v4_final_launch.v1"
FINAL_RUNTIME_SCHEMA = "route_a_v3_route2_xeditflow_v4_final_runtime.v1"
FINAL_ADJUDICATION_SCHEMA = (
    "route_a_v3_route2_xeditflow_final_adjudication.v4"
)
FINAL_GATE_SCHEMA = "route_a_v3_route2_xeditflow_v4_three_seed_gate.v1"
FINAL_TERMINAL_STATUS = "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL"
FINAL_GATE_STATUSES = {"XEDITFLOW_V4_PASS", "XEDITFLOW_V4_NO_GO"}
TERMINAL_CONFIG_SCHEMA = "route_a_v3_route2_base_flow_g0_validation_config.v1"
TERMINAL_TRAINING_SCHEMA = "route_a_v3_route2_base_flow_g0_training.v1"
TERMINAL_VALIDATION_SCHEMA = "route_a_v3_route2_base_flow_g0_validation.v1"
EXPECTED_SOURCE_COUNT = 891
EXPECTED_CANDIDATE_COUNT = 28_512
EXPECTED_CANDIDATE_CAP = 32
EXPECTED_FINAL_JOB_COUNT = 98
FLOAT_ABS_TOLERANCE = 1e-6
OPEN_SUPPORT_STATUS = "UNDEFINED_OPEN_SUPPORT_HAS_UNKNOWN_OUTCOMES"


class BaseFlowV2HandoverReproductionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaseFlowV2HandoverReproductionError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaseFlowV2HandoverReproductionError(
            f"cannot read {label}: {path}"
        ) from exc
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _same_path(recorded: Any, supplied: Path, label: str) -> None:
    _require(isinstance(recorded, str) and bool(recorded), f"{label} is absent")
    _require(
        Path(recorded).resolve() == supplied.resolve(),
        f"{label} differs from the supplied path",
    )


def _require_zero(value: Any, label: str) -> None:
    _require(_is_int(value) and value == 0, f"{label} is not exact zero")


def observe_git_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    head_command = ["git", "rev-parse", "HEAD"]
    status_command = ["git", "status", "--porcelain"]
    head = subprocess.run(
        head_command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        status_command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "head_command": head_command,
        "head_return_code": int(head.returncode),
        "head_stdout": head.stdout,
        "head_stderr": head.stderr,
        "observed_git_head": (
            head.stdout.strip() if head.returncode == 0 else None
        ),
        "status_command": status_command,
        "status_return_code": int(status.returncode),
        "status_stdout": status.stdout,
        "status_stderr": status.stderr,
        "worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }


def validate_git_identity(
    observation: Mapping[str, Any], expected_head: str
) -> None:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected current Git HEAD is invalid",
    )
    _require(
        observation.get("head_return_code") == 0,
        "current Git HEAD query failed",
    )
    _require(
        observation.get("observed_git_head") == expected_head,
        "current Git HEAD differs from expected exact HEAD",
    )
    _require(
        observation.get("status_return_code") == 0,
        "current Git status query failed",
    )
    _require(
        observation.get("worktree_clean") is True,
        "current worktree is not clean",
    )


def validate_final_authority(
    launch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    runtime_path: Path,
    adjudication_path: Path,
) -> dict[str, Any]:
    """Validate every postterminal authority field before protected rows open."""
    _require(
        launch.get("schema_version") == FINAL_LAUNCH_SCHEMA
        and launch.get("status") == "XEDITFLOW_V4_FINAL_SCHEDULER_LAUNCHED",
        "Final launch receipt schema or status differs",
    )
    _same_path(launch.get("runtime_manifest"), runtime_path, "Final runtime identity")
    _same_path(
        launch.get("final_adjudication_path"),
        adjudication_path,
        "Final adjudication identity",
    )
    _require(
        launch.get("development_test_reopened") is False,
        "Final launch receipt reports Development TEST reopening",
    )
    _require_zero(
        launch.get("new_final_evaluation_outcome_reads"),
        "Final launch new Evaluation outcome reads",
    )

    _require(
        runtime.get("schema_version") == FINAL_RUNTIME_SCHEMA
        and runtime.get("status") == FINAL_TERMINAL_STATUS,
        "Final runtime is not the exact terminal runtime",
    )
    _require(
        runtime.get("first_terminal_failure") is None,
        "Final runtime records a terminal failure",
    )
    for key in ("git_head", "experiment_head", "guidance_runner_head"):
        _require(
            runtime.get(key) == launch.get(key),
            f"Final launch/runtime {key} identity differs",
        )
    jobs = runtime.get("jobs")
    _require(
        isinstance(jobs, Mapping) and len(jobs) == EXPECTED_FINAL_JOB_COUNT,
        "Final runtime does not contain exactly 98 jobs",
    )
    for job_key, row in jobs.items():
        _require(isinstance(row, Mapping), f"Final job row is invalid: {job_key}")
        _require(
            row.get("status") == "TERMINAL_COMPLETE"
            and _is_int(row.get("return_code"))
            and row.get("return_code") == 0
            and row.get("terminal_artifact_kind") == "SUCCESS",
            f"Final job is not an exact success: {job_key}",
        )
    for key in (
        "active_performance_output_read",
        "development_test_reopened",
        "development_test_outcomes_accessed_after_atomic_test",
    ):
        _require(runtime.get(key) is False, f"Final runtime protected flag is set: {key}")
    _require_zero(
        runtime.get("new_final_evaluation_outcome_reads"),
        "Final runtime new Evaluation outcome reads",
    )

    _require(
        adjudication.get("schema_version") == FINAL_ADJUDICATION_SCHEMA
        and adjudication.get("status") == FINAL_TERMINAL_STATUS,
        "Final adjudication is not exact terminal V4 adjudication",
    )
    gate = adjudication.get("gate")
    _require(isinstance(gate, Mapping), "Final adjudication gate is absent")
    gate_status = gate.get("status")
    _require(
        gate.get("schema_version") == FINAL_GATE_SCHEMA
        and gate_status in FINAL_GATE_STATUSES,
        "Final adjudication gate is neither exact PASS nor exact NO_GO",
    )
    passed = gate_status == "XEDITFLOW_V4_PASS"
    for payload, label in ((gate, "Final gate"), (adjudication, "Final adjudication")):
        _require(
            payload.get("new_final_evaluation_authorized") is passed,
            f"{label} new Evaluation authorization contradicts gate status",
        )
        _require(
            payload.get("additional_training_seed_authorized") is False,
            f"{label} authorizes an additional training seed",
        )
        _require(
            payload.get("submission_ready") is False,
            f"{label} incorrectly reports submission readiness",
        )
    _require(
        adjudication.get("predictor_generator_baselines_metrics_policy_frozen")
        is True,
        "Final adjudication does not preserve the frozen policy",
    )
    _require(
        adjudication.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and adjudication.get("new_final_evaluation_outcomes_accessed") is False,
        "Final adjudication reports a protected outcome read",
    )
    return {
        "final_gate_status": str(gate_status),
        "final_gate_passed": passed,
        "final_git_head": str(runtime["git_head"]),
        "final_job_count": len(jobs),
    }


def validate_terminal_provenance(
    config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    training_attempt: Mapping[str, Any],
    *,
    config_path: Path,
    training_summary_path: Path,
    training_attempt_path: Path,
    source_manifest_path: Path,
    candidates_path: Path,
) -> dict[str, Any]:
    """Bind terminal rows to their original Base Flow training and config."""
    _require(
        config.get("schema_version") == TERMINAL_CONFIG_SCHEMA,
        "terminal Base Flow config schema differs",
    )
    _require(
        config_path.resolve()
        == (candidates_path.parent / "validation_config.json").resolve(),
        "terminal config is not the producer validation_config.json",
    )
    _require(
        config.get("guided_critic_used") is False
        and config.get("evaluation_outcomes_accessed") is False,
        "terminal Base Flow config crosses its Development-only boundary",
    )
    _same_path(
        config.get("source_eligibility_manifest"),
        source_manifest_path,
        "terminal Base Flow source manifest",
    )
    _same_path(
        config.get("output_directory"),
        candidates_path.parent,
        "terminal Base Flow candidate output directory",
    )
    checkpoint_path = Path(str(config.get("checkpoint_path", "")))
    _require(str(checkpoint_path) not in {"", "."}, "terminal checkpoint path is absent")
    seed = config.get("seed")
    _require(_is_int(seed), "terminal Base Flow seed is invalid")
    device = config.get("device")
    physical_gpu_index = config.get("physical_gpu_index")
    _require(
        isinstance(device, str)
        and re.fullmatch(r"cuda:[0-9]+", device) is not None
        and _is_int(physical_gpu_index)
        and int(device.split(":", 1)[1]) == physical_gpu_index,
        "terminal candidate generation lacks exact CUDA device provenance",
    )

    _require(
        training_summary.get("schema_version") == TERMINAL_TRAINING_SCHEMA
        and training_summary.get("status")
        == "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE",
        "Base Flow training provenance is not terminal",
    )
    _require(
        training_summary_path.parent == checkpoint_path.parent
        and training_attempt_path.parent == checkpoint_path.parent,
        "training summary/attempt and checkpoint directories differ",
    )
    _require(
        training_summary.get("seed") == seed,
        "terminal candidate and checkpoint training seeds differ",
    )
    training_device = training_summary.get("torch_device")
    training_gpu_index = training_summary.get("physical_gpu_index")
    _require(
        isinstance(training_device, str)
        and re.fullmatch(r"cuda:[0-9]+", training_device) is not None
        and _is_int(training_gpu_index)
        and int(training_device.split(":", 1)[1]) == training_gpu_index
        and training_summary.get("cpu_fallback_used") is False
        and training_summary.get("cuda_training_tensors_verified") is True
        and training_summary.get("parameter_changed") is True
        and _is_int(training_summary.get("optimizer_steps"))
        and int(training_summary["optimizer_steps"]) > 0,
        "checkpoint provenance does not prove a learned CUDA update",
    )
    _require(
        training_summary.get("development_test_outcomes_evaluated") is False
        and training_summary.get("guided_critic_used") is False
        and training_summary.get("biological_optimization_established") is False,
        "checkpoint provenance crosses the frozen scientific boundary",
    )
    _require_zero(
        training_summary.get("evaluation_records_read"),
        "Base Flow training Evaluation records read",
    )
    code_commit = training_attempt.get("code_commit")
    _require(
        training_attempt.get("status") == "COMPLETED"
        and training_attempt.get("seed") == seed
        and training_attempt.get("output_directory")
        == str(checkpoint_path.parent)
        and training_attempt.get("device") == training_device
        and training_attempt.get("physical_gpu_index") == training_gpu_index
        and _is_int(training_attempt.get("evaluation_record_count"))
        and training_attempt.get("evaluation_record_count") == 0
        and re.fullmatch(r"[0-9a-f]{40}", str(code_commit)) is not None,
        "original training-attempt provenance is inconsistent",
    )
    return {
        "method_id": METHOD_ID,
        "terminal_candidates_path": str(candidates_path),
        "original_checkpoint_path": str(checkpoint_path),
        "original_config_path": str(config_path),
        "original_training_summary_path": str(training_summary_path),
        "original_training_attempt_path": str(training_attempt_path),
        "original_seed": int(seed),
        "original_code_commit": str(code_commit),
        "original_training_device": str(training_device),
        "original_training_physical_gpu_index": int(training_gpu_index),
        "original_training_cpu_fallback_used": False,
        "original_cuda_training_tensors_verified": True,
    }


def validate_terminal_validation_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    candidates_path: Path,
    config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_optimizer_steps = summary.get(
        "checkpoint_training_optimizer_steps"
    )
    terminal_optimizer_steps = training_summary.get("optimizer_steps")
    _require(
        summary_path.resolve()
        == (candidates_path.parent / "validation_summary.json").resolve(),
        "terminal validation summary is not beside terminal candidates",
    )
    _require(
        summary.get("schema_version") == TERMINAL_VALIDATION_SCHEMA
        and summary.get("status") == "FLOW_G0_READY",
        "Base Flow producer validation summary is not exact terminal FLOW_G0_READY",
    )
    _require(
        summary.get("source_budget_cohort_count") == EXPECTED_SOURCE_COUNT
        and summary.get("trajectory_count") == EXPECTED_CANDIDATE_COUNT,
        "Base Flow producer validation geometry differs",
    )
    _require(
        _finite(summary.get("hard_legality_rate"), "terminal hard legality")
        == 1.0,
        "Base Flow producer hard legality is not exact one",
    )
    for key in (
        "edit_budget_violation_count",
        "candidate_budget_violation_count",
        "numerical_failure_count",
        "trajectory_replay_failure_count",
    ):
        _require_zero(summary.get(key), f"terminal validation {key}")
    _require(
        summary.get("cpu_fallback_used") is False
        and summary.get("device") == config.get("device")
        and summary.get("physical_gpu_index")
        == config.get("physical_gpu_index"),
        "Base Flow producer validation CUDA provenance differs from config",
    )
    _require(
        summary.get("learned_parameter_update_checkpoint_loaded") is True
        and summary.get("checkpoint_gpu_parameter_update_provenance_verified")
        is True
        and summary.get("checkpoint_training_device")
        == training_summary.get("torch_device")
        and summary.get("checkpoint_training_physical_gpu_index")
        == training_summary.get("physical_gpu_index")
        and summary.get("checkpoint_cpu_fallback_used") is False
        and summary.get("checkpoint_cpu_fallback_used")
        == training_summary.get("cpu_fallback_used")
        and summary.get("checkpoint_training_seed")
        == training_summary.get("seed")
        and _is_int(checkpoint_optimizer_steps)
        and checkpoint_optimizer_steps > 0
        and _is_int(terminal_optimizer_steps)
        and checkpoint_optimizer_steps <= terminal_optimizer_steps
        and summary.get("checkpoint_parameter_changed") is True
        and summary.get("checkpoint_parameter_changed")
        == training_summary.get("parameter_changed")
        and summary.get("checkpoint_cuda_training_tensors_verified") is True
        and summary.get("checkpoint_cuda_training_tensors_verified")
        == training_summary.get("cuda_training_tensors_verified")
        and summary.get("checkpoint_training_cuda_device_index")
        == training_summary.get("cuda_device_index")
        and summary.get("checkpoint_training_cuda_device_uuid")
        == training_summary.get("cuda_device_uuid")
        and _finite(
            summary.get("checkpoint_training_cuda_total_memory_mb"),
            "terminal checkpoint CUDA total memory",
        )
        == _finite(
            training_summary.get("cuda_total_memory_mb"),
            "training-summary CUDA total memory",
        ),
        "terminal checkpoint GPU provenance differs from training summary",
    )
    _require_zero(
        summary.get("evaluation_outcomes_read"),
        "terminal validation Evaluation outcome reads",
    )
    _require(
        summary.get("guided_critic_used") is False
        and summary.get("generated_candidates_grant_canonical_credit") is False
        and summary.get("biological_optimization_established") is False,
        "terminal validation crosses its Development engineering boundary",
    )
    return {
        "terminal_validation_summary_path": str(summary_path),
        "terminal_validation_status": "FLOW_G0_READY",
        "terminal_validation_source_count": EXPECTED_SOURCE_COUNT,
        "terminal_validation_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "terminal_validation_hard_legality_rate": 1.0,
        "terminal_validation_failure_counts": {
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "numerical_failure_count": 0,
            "trajectory_replay_failure_count": 0,
        },
        "terminal_validation_device": str(summary["device"]),
        "terminal_validation_physical_gpu_index": int(
            summary["physical_gpu_index"]
        ),
        "terminal_validation_cpu_fallback_used": False,
    }


def read_expected_row(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if row.get("method_id") == METHOD_ID
            ]
    except OSError as exc:
        raise BaseFlowV2HandoverReproductionError(
            f"cannot read tracked expected CSV: {path}"
        ) from exc
    _require(
        len(rows) == 1,
        f"tracked expected CSV must contain exactly one {METHOD_ID} row",
    )
    return rows[0]


def recompute_terminal_metrics(
    source_manifest_path: Path,
    candidates_path: Path,
    measured_neighborhood_path: Path,
    *,
    k: int,
) -> dict[str, Any]:
    """Use the shared evaluator itself; no frozen evaluation JSON is accepted."""
    _require(_is_int(k) and k > 0, "measured-neighborhood k must be positive")
    sources = evaluator.load_source_manifest(source_manifest_path)
    candidates = evaluator._read_jsonl(candidates_path)
    measured_rows = evaluator._read_jsonl(measured_neighborhood_path)
    evaluator.validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    generation = evaluator.evaluate_generation(sources, candidates)
    measured = evaluator.measured_neighborhood_metrics(
        sources,
        candidates,
        measured_rows,
        k=k,
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    return {
        "schema_version": (
            "route_a_v3_route2_base_flow_v2_handover_recomputed_metrics.v1"
        ),
        "status": "BASE_FLOW_V2_HANDOVER_R3_METRICS_RECOMPUTED",
        "generation": generation,
        "measured_neighborhood": measured,
        "measured_neighborhood_pool": "DEVELOPMENT",
        "evaluation_release_state": "CLOSED",
        "shared_evaluator_module": (
            "scripts.route_a_v3.evaluate_route2_generation_v1"
        ),
        "frozen_evaluation_json_copied": False,
    }


def _expected_int(row: Mapping[str, str], key: str) -> int:
    raw = row.get(key)
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise BaseFlowV2HandoverReproductionError(
            f"tracked expected {key} is not an integer"
        ) from exc
    return value


def _expected_float(row: Mapping[str, str], key: str) -> float:
    raw = row.get(key)
    try:
        value = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise BaseFlowV2HandoverReproductionError(
            f"tracked expected {key} is not numeric"
        ) from exc
    _require(math.isfinite(value), f"tracked expected {key} is not finite")
    return value


def compare_to_tracked_expected(
    metrics: Mapping[str, Any], expected_row: Mapping[str, str]
) -> dict[str, Any]:
    generation = metrics.get("generation")
    measured = metrics.get("measured_neighborhood")
    _require(
        isinstance(generation, Mapping) and isinstance(measured, Mapping),
        "recomputed metrics are incomplete",
    )
    _require(
        generation.get("method_id") == METHOD_ID,
        "recomputed method identity differs",
    )

    expected_discrete = {
        "source_count": _expected_int(expected_row, "source_count"),
        "candidate_count": _expected_int(expected_row, "candidate_count"),
        "candidate_cap_per_source": _expected_int(
            expected_row, "candidate_cap_per_source"
        ),
        "edit_budget_violation_count": _expected_int(
            expected_row, "edit_budget_violation_count"
        ),
        "candidate_budget_violation_count": _expected_int(
            expected_row, "candidate_budget_violation_count"
        ),
    }
    _require(
        expected_discrete
        == {
            "source_count": EXPECTED_SOURCE_COUNT,
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "candidate_cap_per_source": EXPECTED_CANDIDATE_CAP,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
        },
        "tracked expected Base Flow row changed its frozen discrete contract",
    )
    per_source = generation.get("per_source")
    _require(isinstance(per_source, Mapping), "generation per-source metrics are absent")
    candidate_budgets = {
        row.get("candidate_budget")
        for row in per_source.values()
        if isinstance(row, Mapping)
    }
    candidate_counts = [
        row.get("candidate_count")
        for row in per_source.values()
        if isinstance(row, Mapping)
    ]
    _require(
        len(candidate_counts) == EXPECTED_SOURCE_COUNT
        and candidate_budgets == {EXPECTED_CANDIDATE_CAP}
        and all(
            _is_int(value) and 0 < value <= EXPECTED_CANDIDATE_CAP
            for value in candidate_counts
        ),
        "recomputed per-source candidate cap differs",
    )
    observed_discrete = {
        "source_count": generation.get("source_count"),
        "candidate_count": generation.get("candidate_count"),
        "candidate_cap_per_source": next(iter(candidate_budgets)),
        "edit_budget_violation_count": generation.get(
            "edit_budget_violation_count"
        ),
        "candidate_budget_violation_count": generation.get(
            "candidate_budget_violation_count"
        ),
    }
    _require(
        observed_discrete == expected_discrete,
        "recomputed discrete Base Flow metrics differ from tracked expected row",
    )
    expected_legality = _expected_float(expected_row, "hard_legality_rate")
    observed_legality = _finite(
        generation.get("hard_legality_rate"), "recomputed hard legality rate"
    )
    _require(
        expected_legality == 1.0 and observed_legality == 1.0,
        "hard legality is not exact one",
    )

    float_fields = {
        "source_macro_unique_candidate_rate": (
            generation.get("source_macro_unique_candidate_rate"),
            _expected_float(expected_row, "source_macro_unique_candidate_rate"),
        ),
        "source_macro_candidate_recovery_rate": (
            measured.get("source_macro_candidate_recovery_rate"),
            _expected_float(
                expected_row, "source_macro_candidate_recovery_rate"
            ),
        ),
        "source_macro_measured_top_k_recovery_at_k": (
            measured.get("source_macro_measured_top_k_recovery_at_k"),
            _expected_float(
                expected_row, "source_macro_measured_top_k_recovery_at_k"
            ),
        ),
    }
    float_checks: dict[str, Any] = {}
    for key, (observed_raw, expected) in float_fields.items():
        observed = _finite(observed_raw, f"recomputed {key}")
        absolute_error = abs(observed - expected)
        _require(
            absolute_error <= FLOAT_ABS_TOLERANCE,
            f"recomputed {key} exceeds absolute tolerance",
        )
        float_checks[key] = {
            "observed": observed,
            "expected": expected,
            "absolute_error": absolute_error,
            "absolute_tolerance": FLOAT_ABS_TOLERANCE,
            "matched": True,
        }

    _require(
        measured.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT"
        and measured.get("unknown_generated_candidates_are_zero_gain") is False,
        "measured-neighborhood support boundary differs",
    )
    _require(
        measured.get("source_macro_closed_measured_ndcg_at_k") is None
        and measured.get("source_closed_measured_ndcg_defined_count") == 0,
        "closed measured NDCG must remain undefined for every source",
    )
    measured_per_source = measured.get("per_source")
    _require(
        isinstance(measured_per_source, Mapping)
        and len(measured_per_source) == EXPECTED_SOURCE_COUNT
        and all(
            isinstance(row, Mapping)
            and row.get("closed_measured_ndcg_at_k") is None
            and row.get("closed_measured_ndcg_status") == OPEN_SUPPORT_STATUS
            for row in measured_per_source.values()
        ),
        "per-source closed measured NDCG is not uniformly undefined",
    )
    return {
        "schema_version": (
            "route_a_v3_route2_base_flow_v2_handover_comparison.v1"
        ),
        "status": "BASE_FLOW_V2_HANDOVER_R3_MATCH",
        "method_id": METHOD_ID,
        "tracked_expected_discrete": expected_discrete,
        "recomputed_discrete": observed_discrete,
        "hard_legality_rate": {
            "observed": observed_legality,
            "expected": expected_legality,
            "comparison": "EXACT",
            "matched": True,
        },
        "continuous_metrics": float_checks,
        "closed_measured_ndcg_at_k": None,
        "closed_measured_ndcg_status": OPEN_SUPPORT_STATUS,
        "source_closed_measured_ndcg_defined_count": 0,
        "matched": True,
    }


def _ensure_new_output(output_dir: Path) -> Path:
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    _require(
        not partial_dir.exists(),
        f"partial output directory already exists: {partial_dir}",
    )
    return partial_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _readme(
    *,
    final_gate_status: str,
    current_head: str,
    comparison: Mapping[str, Any],
    terminal_provenance: Mapping[str, Any],
) -> str:
    gate_line = (
        "The frozen Final Development gate is PASS."
        if final_gate_status == "XEDITFLOW_V4_PASS"
        else "The frozen Final Development gate is NO_GO."
    )
    return f"""# Base Flow V2 handover R3 reproduction

{gate_line} This directory independently recomputes the historical Base Flow V2
Development aggregation from terminal candidate, source-manifest, and measured-
neighborhood rows using `evaluate_route2_generation_v1.py`.

- Current clean exact HEAD: `{current_head}`
- Comparison status: `{comparison['status']}`
- Terminal producer summary: `{terminal_provenance['terminal_validation_summary_path']}`
- Terminal producer status: `FLOW_G0_READY`
- Terminal producer geometry: 891 sources / 28,512 candidates / hard legality 1.0
- Terminal producer failure counts: edit-budget 0 / candidate-budget 0 / numerical 0 / replay 0
- Scientific scope: Development-only aggregation reproduction
- Development TEST opened: no
- New Evaluation opened: no
- Training or parameter update: no
- Generation or independent-scorer rerun: no
- Model forward: no
- GPU/CUDA operation: no
- CPU fallback: no; CPU aggregation is the declared execution mode
- Additional protected outcome reads: 0

This artifact reproduces a historical engineering baseline row. It does not turn
that row into a final biological claim, does not copy a frozen evaluation JSON,
and does not alter the Final Development PASS/NO_GO adjudication.
"""


def write_output_directory(
    output_dir: Path,
    *,
    command: Sequence[str],
    git_observation: Mapping[str, Any],
    current_head: str,
    authority: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    partial_dir = _ensure_new_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    terminal_stdout = {
        "status": "BASE_FLOW_V2_HANDOVER_R3_REPRODUCTION_COMPLETE",
        "method_id": METHOD_ID,
        "final_gate_status": authority["final_gate_status"],
        "comparison_status": comparison["status"],
        "output_dir": str(output_dir),
    }
    (partial_dir / "command.txt").write_text(
        shlex.join(list(command)) + "\n", encoding="utf-8"
    )
    (partial_dir / "environment.txt").write_text(
        "\n".join(
            (
                "execution_mode=CPU_AGGREGATION_ONLY",
                f"python_version={platform.python_version()}",
                f"platform={os.uname().sysname}-{os.uname().release}-{os.uname().machine}",
                "training_run=false",
                "gpu_validation_run=false",
                "generation_run=false",
                "independent_scorer_run=false",
                "model_forward_run=false",
                "cuda_queried=false",
                "cpu_fallback_used=false",
                "additional_protected_outcome_reads=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (partial_dir / "git_status.txt").write_text(
        "\n".join(
            (
                f"observed_git_head={current_head}",
                "worktree_clean=true",
                "status_porcelain_begin",
                str(git_observation.get("status_stdout", "")).rstrip("\n"),
                "status_porcelain_end",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        partial_dir / "config.snapshot.json",
        {
            "schema_version": (
                "route_a_v3_route2_base_flow_v2_handover_config_snapshot.v1"
            ),
            "status": "BASE_FLOW_V2_HANDOVER_R3_CONFIG_SNAPSHOTTED",
            "current_git_head": current_head,
            "current_worktree_clean": True,
            "authority": dict(authority),
            "terminal_provenance": dict(provenance),
            **dict(config_snapshot),
            "development_only": True,
            "development_test_opened": False,
            "new_evaluation_opened": False,
            "additional_protected_outcome_reads": 0,
            "training_run": False,
            "gpu_validation_run": False,
            "model_forward_run": False,
            "cuda_queried": False,
            "cpu_fallback_used": False,
        },
    )
    _write_json(partial_dir / "metrics.json", metrics)
    _write_json(partial_dir / "comparison_to_frozen.json", comparison)
    (partial_dir / "stdout.log").write_text(
        json.dumps(terminal_stdout, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partial_dir / "README.md").write_text(
        _readme(
            final_gate_status=str(authority["final_gate_status"]),
            current_head=current_head,
            comparison=comparison,
            terminal_provenance=provenance,
        ),
        encoding="utf-8",
    )
    os.replace(partial_dir, output_dir)
    return terminal_stdout


def run_reproduction(
    *,
    final_launch_receipt_path: Path,
    final_runtime_path: Path,
    final_adjudication_path: Path,
    terminal_config_path: Path,
    terminal_validation_summary_path: Path,
    terminal_training_summary_path: Path,
    terminal_provenance_path: Path,
    source_manifest_path: Path,
    terminal_candidates_path: Path,
    development_measured_neighborhood_path: Path,
    tracked_expected_csv_path: Path,
    expected_head: str,
    output_dir: Path,
    k: int = 10,
    command: Sequence[str] = (),
    git_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_new_output(output_dir)
    launch = _read_json(final_launch_receipt_path, "Final launch receipt")
    runtime = _read_json(final_runtime_path, "Final runtime")
    adjudication = _read_json(final_adjudication_path, "Final adjudication")
    authority = validate_final_authority(
        launch,
        runtime,
        adjudication,
        runtime_path=final_runtime_path,
        adjudication_path=final_adjudication_path,
    )
    terminal_config = _read_json(terminal_config_path, "terminal Base Flow config")
    terminal_validation_summary = _read_json(
        terminal_validation_summary_path,
        "terminal Base Flow producer validation summary",
    )
    terminal_training_summary = _read_json(
        terminal_training_summary_path, "Base Flow training summary"
    )
    terminal_provenance_payload = _read_json(
        terminal_provenance_path, "Base Flow training-attempt provenance"
    )
    observation = dict(git_observation or observe_git_identity())
    validate_git_identity(observation, expected_head)
    provenance = validate_terminal_provenance(
        terminal_config,
        terminal_training_summary,
        terminal_provenance_payload,
        config_path=terminal_config_path,
        training_summary_path=terminal_training_summary_path,
        training_attempt_path=terminal_provenance_path,
        source_manifest_path=source_manifest_path,
        candidates_path=terminal_candidates_path,
    )
    provenance.update(
        validate_terminal_validation_summary(
            terminal_validation_summary,
            summary_path=terminal_validation_summary_path,
            candidates_path=terminal_candidates_path,
            config=terminal_config,
            training_summary=terminal_training_summary,
        )
    )
    expected_row = read_expected_row(tracked_expected_csv_path)

    # Final authority and all cross-file identities are settled above. Only now
    # may the Development source, terminal candidate, and measured rows open.
    metrics = recompute_terminal_metrics(
        source_manifest_path,
        terminal_candidates_path,
        development_measured_neighborhood_path,
        k=k,
    )
    comparison = compare_to_tracked_expected(metrics, expected_row)
    metrics_payload = {
        **dict(metrics),
        "status": "BASE_FLOW_V2_HANDOVER_R3_REPRODUCTION_COMPLETE",
        "current_git_head": expected_head,
        "current_worktree_clean": True,
        "final_gate_status": authority["final_gate_status"],
        "terminal_provenance": provenance,
        "comparison_status": comparison["status"],
        "scientific_scope": "DEVELOPMENT_ONLY_AGGREGATION_REPRODUCTION",
        "development_test_opened": False,
        "new_evaluation_opened": False,
        "additional_protected_outcome_reads": 0,
        "training_run": False,
        "gpu_validation_run": False,
        "generation_run": False,
        "independent_scorer_run": False,
        "model_forward_run": False,
        "cuda_queried": False,
        "cpu_fallback_used": False,
    }
    config_snapshot = {
        "input_paths": {
            "final_launch_receipt": str(final_launch_receipt_path),
            "final_runtime": str(final_runtime_path),
            "final_adjudication": str(final_adjudication_path),
            "terminal_config": str(terminal_config_path),
            "terminal_validation_summary": str(
                terminal_validation_summary_path
            ),
            "terminal_training_summary": str(terminal_training_summary_path),
            "terminal_training_attempt": str(terminal_provenance_path),
            "source_manifest": str(source_manifest_path),
            "terminal_candidates": str(terminal_candidates_path),
            "development_measured_neighborhood": str(
                development_measured_neighborhood_path
            ),
            "tracked_expected_csv": str(tracked_expected_csv_path),
        },
        "terminal_config": dict(terminal_config),
        "terminal_validation_summary": dict(terminal_validation_summary),
        "terminal_training_summary": dict(terminal_training_summary),
        "terminal_training_attempt": dict(terminal_provenance_payload),
        "measured_neighborhood_k": int(k),
        "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "evaluation_release_state": "CLOSED",
    }
    return write_output_directory(
        output_dir,
        command=command,
        git_observation=observation,
        current_head=expected_head,
        authority=authority,
        provenance=provenance,
        config_snapshot=config_snapshot,
        metrics=metrics_payload,
        comparison=comparison,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-launch-receipt", required=True, type=Path)
    parser.add_argument("--final-runtime", required=True, type=Path)
    parser.add_argument("--final-adjudication", required=True, type=Path)
    parser.add_argument("--terminal-config", required=True, type=Path)
    parser.add_argument("--terminal-validation-summary", required=True, type=Path)
    parser.add_argument("--terminal-training-summary", required=True, type=Path)
    parser.add_argument("--terminal-provenance", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--terminal-candidates", required=True, type=Path)
    parser.add_argument(
        "--development-measured-neighborhood", required=True, type=Path
    )
    parser.add_argument("--tracked-expected-csv", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(sys.argv[1:] if argv is None else argv)
    result = run_reproduction(
        final_launch_receipt_path=arguments.final_launch_receipt,
        final_runtime_path=arguments.final_runtime,
        final_adjudication_path=arguments.final_adjudication,
        terminal_config_path=arguments.terminal_config,
        terminal_validation_summary_path=arguments.terminal_validation_summary,
        terminal_training_summary_path=arguments.terminal_training_summary,
        terminal_provenance_path=arguments.terminal_provenance,
        source_manifest_path=arguments.source_manifest,
        terminal_candidates_path=arguments.terminal_candidates,
        development_measured_neighborhood_path=(
            arguments.development_measured_neighborhood
        ),
        tracked_expected_csv_path=arguments.tracked_expected_csv,
        expected_head=arguments.expected_head,
        output_dir=arguments.output_dir,
        k=arguments.k,
        command=command,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
