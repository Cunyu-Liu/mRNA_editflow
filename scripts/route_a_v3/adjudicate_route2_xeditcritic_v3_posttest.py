#!/usr/bin/env python3
"""Adjudicate all-Development refits or paired seven-study LOSO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v3 import adjudicate_critic_loso_v3
from core.route2_xeditcritic_ledger_v3 import POSTTEST_STUDIES_V3


SEEDS = (20260831, 20260901, 20260902)


class CriticPosttestAdjudicationV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticPosttestAdjudicationV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def adjudicate_refits_v3(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("status") == "XEDITCRITIC_V3_REFIT_CONFIGS_PREPARED", "refit job manifest is incomplete")
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 3, "refit requires exactly three jobs")
    completed = []
    for job in jobs:
        summary = _json(Path(job["summary_path"]))
        _require(summary.get("status") == "TERMINAL_REFIT_ARM_COMPLETE", "refit run is incomplete")
        _require(summary.get("arm") == manifest["selected_arm"] and int(summary.get("seed", -1)) == int(job["seed"]), "refit run identity differs")
        _require(int(summary.get("train_record_count", -1)) == 107873 and int(summary.get("validation_record_count", -1)) == 0, "refit record scope differs")
        _require(int(summary.get("selected_pass", -1)) == int(manifest["refit_pass_count"]), "refit pass count differs")
        _require(summary.get("development_test_outcomes_accessed") is False and summary.get("new_final_evaluation_outcomes_accessed") is False, "refit accessed protected outcome")
        completed.append({"seed": int(job["seed"]), "checkpoint_path": summary["checkpoint_path"]})
    _require({row["seed"] for row in completed} == set(SEEDS), "refit seed inventory differs")
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_refit_manifest.v1",
        "status": "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE",
        "selected_arm": manifest["selected_arm"],
        "required_seeds": list(SEEDS),
        "refit_pass_count": int(manifest["refit_pass_count"]),
        "completed_refit_count": 3,
        "checkpoints": completed,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
        "loso_authorized": True,
    }


def adjudicate_loso_jobs_v3(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("status") == "XEDITCRITIC_V3_LOSO_CONFIGS_PREPARED", "LOSO job manifest is incomplete")
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list) and len(jobs) == 42, "LOSO requires exactly 42 paired jobs")
    by_identity = {}
    for job in jobs:
        identity = (int(job["seed"]), str(job["held_out_study"]), str(job["arm"]))
        _require(identity not in by_identity, "LOSO job identity is duplicated")
        summary = _json(Path(job["summary_path"]))
        _require(summary.get("status") == "TERMINAL_LOSO_ARM_COMPLETE", "LOSO run is incomplete")
        _require(int(summary.get("seed", -1)) == identity[0] and summary.get("arm") == identity[2] and summary.get("held_out_study") == identity[1], "LOSO run identity differs")
        _require(summary.get("held_out_study_scale_policy") == "UNKNOWN_STUDY_SCALE_FIXED_1", "LOSO held-out study scale differs")
        _require(summary.get("development_test_outcomes_accessed") is False and summary.get("new_final_evaluation_outcomes_accessed") is False, "LOSO accessed protected outcome")
        by_identity[identity] = summary
    selected = str(manifest["selected_arm"])
    seed_payloads = {}
    for seed in SEEDS:
        folds = {}
        for study in POSTTEST_STUDIES_V3:
            model = by_identity[(seed, study, selected)]["final_validation"]
            baseline = by_identity[(seed, study, "C0")]["final_validation"]
            model_spearman = float(model["task_macro_spearman"])
            baseline_spearman = float(baseline["task_macro_spearman"])
            folds[study] = {
                "model_spearman": model_spearman,
                "baseline_spearman": baseline_spearman,
                "margin": model_spearman - baseline_spearman,
            }
        seed_payloads[seed] = {
            "status": "XEDITCRITIC_V3_PAIRED_LOSO_COMPLETE",
            "held_out_study_count": 7,
            "model_study_macro_spearman": float(np.mean([row["model_spearman"] for row in folds.values()])),
            "baseline_study_macro_spearman": float(np.mean([row["baseline_spearman"] for row in folds.values()])),
            "fold_margins": {study: row["margin"] for study, row in folds.items()},
            "folds": folds,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    gate = adjudicate_critic_loso_v3(seed_payloads)
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_loso_adjudication.v1",
        "status": "XEDITCRITIC_V3_LOSO_TERMINAL",
        "selected_arm": selected,
        "seed_results": seed_payloads,
        "loso_gate": gate,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("REFIT", "LOSO"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"post-TEST adjudication output exists: {args.output}")
    manifest = _json(args.manifest)
    result = adjudicate_refits_v3(manifest) if args.mode == "REFIT" else adjudicate_loso_jobs_v3(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
