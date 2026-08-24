#!/usr/bin/env python3
"""Adjudicate exact Critic V4 all-Development refit or paired LOSO jobs."""

from __future__ import annotations

import argparse
import json
import os
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


def adjudicate_refits_v4(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        manifest.get("status") == "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED"
        and manifest.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
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
        _require(
            terminal.get("status") == "TERMINAL_XEDITCRITIC_V4_REFIT_RUN_COMPLETE"
            and terminal.get("run_stage") == "REFIT"
            and terminal.get("run_id") == "v4_full"
            and int(terminal.get("seed", -1)) == int(job["seed"])
            and int(terminal.get("train_record_count", -1)) == 107_873
            and int(terminal.get("validation_record_count", -1)) == 0
            and int(terminal.get("pass_count", -1)) == 8
            and int(terminal.get("selected_pass", -1)) == 8
            and int(terminal.get("physical_batch_size", -1)) in {4, 8, 16, 32}
            and terminal.get("development_test_outcome_reads") == 0
            and terminal.get("new_final_evaluation_outcome_reads") == 0,
            "Critic V4 refit terminal identity changed",
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
        _require(
            terminal.get("status") == "TERMINAL_XEDITCRITIC_V4_LOSO_RUN_COMPLETE"
            and terminal.get("run_stage") == "LOSO"
            and int(terminal.get("seed", -1)) == identity[0]
            and terminal.get("held_out_study") == identity[1]
            and terminal.get("run_id") == identity[2]
            and terminal.get("held_out_study_scale_policy")
            == "UNKNOWN_STUDY_SCALE_FIXED_1"
            and int(terminal.get("pass_count", -1)) == 8
            and int(terminal.get("selected_pass", -1)) == 8
            and int(terminal.get("development_test_outcome_reads", -1)) == 0
            and int(terminal.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "Critic V4 LOSO terminal identity changed",
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
