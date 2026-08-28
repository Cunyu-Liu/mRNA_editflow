#!/usr/bin/env python3
"""Adjudicate exact Critic V4 all-Development refit or paired LOSO jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v4 import (
    CONFIRMATION_SEEDS_V4,
    LOSO_STUDIES_V4,
    adjudicate_critic_loso_v4,
)


class CriticPosttestAdjudicationV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticPosttestAdjudicationV4Error(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def _terminal_job(job: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    summary = Path(job["summary_path"])
    failure = Path(job["failure_path"])
    _require(
        int(summary.exists()) + int(failure.exists()) == 1,
        "Critic V4 posttest job is not exactly terminal",
    )
    return ("failure", _read(failure)) if failure.exists() else ("summary", _read(summary))


def _validate_parameter_update_terminal_v4(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    stage: str,
) -> Mapping[str, Any]:
    """Bind one explicit manifest job to its config, attempt, and checkpoint."""

    seed = int(job["seed"])
    run_id = str(job["run_id"])
    runner_head = str(manifest.get("runner_git_head", ""))
    _require(
        re.fullmatch(r"[0-9a-f]{40}", runner_head) is not None,
        "Critic V4 posttest runner Git HEAD is invalid",
    )
    config_path = Path(str(job.get("config_path", "")))
    _require(config_path.is_file(), "Critic V4 posttest job config is absent")
    config = _read(config_path)
    expected_runs = ["v4_full"] if stage == "REFIT" else ["v4_full", "c0_v4"]
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1"
        and config.get("run_stage") == stage
        and int(config.get("training_seed", -1)) == seed
        and config.get("posttest_runner_git_head") == runner_head
        and config.get("required_posttest_run_ids") == expected_runs
        and run_id in expected_runs,
        "Critic V4 posttest config identity changed",
    )
    if stage == "LOSO":
        _require(
            config.get("held_out_study") == job.get("held_out_study")
            and config.get("held_out_study_scale_policy")
            == "UNKNOWN_STUDY_SCALE_FIXED_1",
            "Critic V4 LOSO config holdout identity changed",
        )
    output_directory = Path(str(config["output_root"])) / run_id
    summary_path = output_directory / "run_summary.json"
    failure_path = output_directory / "failure.json"
    checkpoint_path = output_directory / "final_pass_8_checkpoint.pt"
    attempt_path = output_directory / "training_attempt.json"
    _require(
        Path(str(job["summary_path"])) == summary_path
        and Path(str(job["failure_path"])) == failure_path,
        "Critic V4 posttest manifest path binding changed",
    )
    _require(
        checkpoint_path.is_file() and attempt_path.is_file(),
        "Critic V4 posttest checkpoint or completed attempt is absent",
    )
    geometry = config["data_geometry"]
    physical_gpu_index = int(summary.get("physical_gpu_index", -1))
    run_spec = next(
        (
            row
            for row in config["required_screen_runs"]
            if str(row["run_id"]) == run_id
        ),
        None,
    )
    _require(run_spec is not None, "Critic V4 posttest run spec is absent")
    expected_scope = (
        "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
        if str(run_spec["model"]) == "C0-V4"
        else "NOT_CLAIMED_PARAMETER_MATCHED_DIFFERENT_MODULE"
        if str(run_spec["mechanism"]) == "NO_CROSS"
        else "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"
    )
    _require(
        summary.get("schema_version")
        == f"route_a_v3_route2_xeditcritic_v4_{stage.lower()}_run.v2"
        and summary.get("status")
        == f"TERMINAL_XEDITCRITIC_V4_{stage}_RUN_COMPLETE"
        and summary.get("run_stage") == stage
        and summary.get("run_id") == run_id
        and int(summary.get("seed", -1)) == seed
        and int(summary.get("parameter_initialization_seed", -1)) == seed
        and summary.get(
            "parameter_initialization_seed_applied_before_model_construction"
        )
        is True
        and summary.get("parameter_initialization_tensor_identity_scope")
        == expected_scope
        and summary.get("training_git_head") == runner_head
        and summary.get("cuda_available") is True
        and 0 <= physical_gpu_index <= 5
        and summary.get("cuda_device") == f"cuda:{physical_gpu_index}"
        and "A100" in str(summary.get("cuda_device_name", ""))
        and summary.get("a100_device_verified") is True
        and summary.get("bf16_supported") is True
        and summary.get("precision") == "BF16_FORWARD_FP32_EFFECTIVE_OBJECTIVE"
        and summary.get("cpu_fallback_used") is False
        and summary.get("parameter_changed") is True
        and int(summary.get("train_record_count", -1))
        == int(geometry["expected_train_count"])
        and int(summary.get("validation_record_count", -1))
        == int(geometry["expected_validation_count"])
        and int(summary.get("pass_count", -1)) == int(geometry["pass_count"]) == 8
        and int(summary.get("selected_pass", -1)) == 8
        and int(summary.get("update_count", -1))
        == int(geometry["total_optimizer_updates"])
        and int(summary.get("effective_batch_size", -1)) == 32
        and int(summary.get("physical_batch_size", -1)) in {4, 8, 16, 32}
        and summary.get("selection_policy")
        == "FINAL_PASS_8_FIXED_NO_TEST_OR_VALIDATION_SELECTION"
        and summary.get("output_directory") == str(output_directory)
        and summary.get("training_summary_path") == str(summary_path)
        and summary.get("checkpoint_path") == str(checkpoint_path)
        and summary.get("training_attempt_path") == str(attempt_path)
        and int(summary.get("development_test_outcome_reads", -1)) == 0
        and int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 posttest terminal parameter-update evidence changed",
    )
    attempt = _read(attempt_path)
    _require(
        attempt.get("status") == "COMPLETED"
        and attempt.get("code_commit") == runner_head
        and attempt.get("training_git_head") == runner_head
        and int(attempt.get("seed", -1)) == seed
        and int(attempt.get("parameter_initialization_seed", -1)) == seed
        and attempt.get(
            "parameter_initialization_seed_applied_before_model_construction"
        )
        is True
        and attempt.get("parameter_initialization_tensor_identity_scope")
        == expected_scope
        and attempt.get("cuda_available") is True
        and attempt.get("device") == f"cuda:{physical_gpu_index}"
        and "A100" in str(attempt.get("cuda_device_name", ""))
        and attempt.get("a100_device_verified") is True
        and attempt.get("bf16_supported") is True
        and attempt.get("training_precision") == "BF16"
        and attempt.get("cpu_fallback_used") is False
        and int(attempt.get("optimizer_steps", -1))
        == int(geometry["total_optimizer_updates"])
        and int(attempt.get("selected_epoch", -1)) == 8
        and attempt.get("output_directory") == str(output_directory)
        and attempt.get("training_summary_path") == str(summary_path)
        and attempt.get("checkpoint_path") == str(checkpoint_path)
        and attempt.get("training_attempt_path") == str(attempt_path),
        "Critic V4 completed training attempt evidence changed",
    )
    return summary


def adjudicate_refits_v4(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        manifest.get("status") == "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and re.fullmatch(
            r"[0-9a-f]{40}", str(manifest.get("runner_git_head", ""))
        )
        is not None
        and int(manifest.get("refit_pass_count", -1)) == 8,
        "Critic V4 refit manifest changed",
    )
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 3, "Critic V4 refit requires three jobs")
    completed = []
    failures = []
    for job in jobs:
        kind, terminal = _terminal_job(job)
        if kind == "failure":
            _require(
                int(terminal.get("development_test_outcome_reads", -1)) == 0
                and int(terminal.get("new_final_evaluation_outcome_reads", -1)) == 0,
                "Critic V4 refit failure reports a protected read",
            )
            failures.append({"seed": int(job["seed"]), **dict(terminal)})
            continue
        terminal = _validate_parameter_update_terminal_v4(
            manifest, job, terminal, stage="REFIT"
        )
        completed.append(
            {
                "seed": int(job["seed"]),
                "checkpoint_path": terminal["checkpoint_path"],
                "physical_batch_size": int(terminal["physical_batch_size"]),
            }
        )
    passed = not failures and {row["seed"] for row in completed} == set(CONFIRMATION_SEEDS_V4)
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_manifest.v1",
        "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        if passed
        else "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_NO_GO",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "refit_pass_count": 8,
        "completed_refit_count": len(completed),
        "checkpoints": completed,
        "technical_failures": failures,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
        "loso_authorized": passed,
    }


def adjudicate_loso_jobs_v4(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        manifest.get("status") == "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and re.fullmatch(
            r"[0-9a-f]{40}", str(manifest.get("runner_git_head", ""))
        )
        is not None
        and manifest.get("held_out_studies") == list(LOSO_STUDIES_V4),
        "Critic V4 LOSO manifest changed",
    )
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 42, "Critic V4 LOSO requires 42 paired jobs")
    summaries: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    failures = []
    for job in jobs:
        identity = (int(job["seed"]), str(job["held_out_study"]), str(job["run_id"]))
        _require(identity not in summaries, "Critic V4 LOSO job identity is duplicated")
        kind, terminal = _terminal_job(job)
        if kind == "failure":
            _require(
                int(terminal.get("development_test_outcome_reads", -1)) == 0
                and int(terminal.get("new_final_evaluation_outcome_reads", -1)) == 0,
                "Critic V4 LOSO failure reports a protected read",
            )
            failures.append(
                {"seed": identity[0], "held_out_study": identity[1], "run_id": identity[2], **dict(terminal)}
            )
            continue
        terminal = _validate_parameter_update_terminal_v4(
            manifest, job, terminal, stage="LOSO"
        )
        summaries[identity] = terminal
    if failures:
        gate = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_gate.v1",
            "status": "XEDITCRITIC_V4_LOSO_NO_GO",
            "reason": "ONE_OR_MORE_FROZEN_LOSO_RUNS_FAILED_TECHNICALLY",
            "required_seeds": list(CONFIRMATION_SEEDS_V4),
            "held_out_studies": list(LOSO_STUDIES_V4),
            "technical_failures": failures,
            "guidance_readiness_authorized": False,
            "new_final_evaluation_authorized": False,
        }
        seed_results = {}
    else:
        _require(len(summaries) == 42, "Critic V4 LOSO terminal inventory is incomplete")
        seed_results = {}
        for seed in CONFIRMATION_SEEDS_V4:
            folds = {}
            for study in LOSO_STUDIES_V4:
                model = summaries[(seed, study, "v4_full")]["final_validation"]
                baseline = summaries[(seed, study, "c0_v4")]["final_validation"]
                model_rho = float(model["task_macro_spearman"])
                baseline_rho = float(baseline["task_macro_spearman"])
                folds[study] = {
                    "model_spearman": model_rho,
                    "baseline_spearman": baseline_rho,
                    "margin": model_rho - baseline_rho,
                }
            seed_results[seed] = {
                "status": "XEDITCRITIC_V4_PAIRED_LOSO_COMPLETE",
                "held_out_study_count": 7,
                "model_study_macro_spearman": float(
                    np.mean([row["model_spearman"] for row in folds.values()])
                ),
                "baseline_study_macro_spearman": float(
                    np.mean([row["baseline_spearman"] for row in folds.values()])
                ),
                "fold_margins": {study: row["margin"] for study, row in folds.items()},
                "folds": folds,
                "development_test_outcomes_accessed_during_loso": False,
                "new_final_evaluation_outcomes_accessed": False,
            }
        gate = adjudicate_critic_loso_v4(seed_results)
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_adjudication.v1",
        "status": "XEDITCRITIC_V4_LOSO_TERMINAL",
        "seed_results": seed_results,
        "technical_failures": failures,
        "loso_gate": gate,
        "development_test_outcomes_accessed_during_loso": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("REFIT", "LOSO"))
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"Critic V4 posttest adjudication exists: {arguments.output}")
    manifest = _read(arguments.manifest)
    result = (
        adjudicate_refits_v4(manifest)
        if arguments.mode == "REFIT"
        else adjudicate_loso_jobs_v4(manifest)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
