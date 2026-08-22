#!/usr/bin/env python3
"""Prepare authorized all-Development refit or seven-study LOSO jobs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_xeditcritic_ledger_v3 import POSTTEST_STUDIES_V3
from core.route2_xeditcritic_training_data_v3 import records_from_projection_rows


SEEDS = (20260831, 20260901, 20260902)


class CriticPosttestPrepareV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticPosttestPrepareV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _posttest_selection(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    three = _json(Path(config["three_seed_gate_path"]))
    _require(three.get("status") == "XEDITCRITIC_V3_THREE_SEED_PASS", "post-TEST configs require three-seed PASS")
    selected = str(three.get("selected_arm"))
    _require(selected in {"C2", "C3"}, "post-TEST selected arm differs")
    atomic = _json(Path(config["atomic_frozen_test_path"]))
    gate = atomic.get("frozen_test_gate")
    _require(
        atomic.get("status") == "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL"
        and isinstance(gate, Mapping)
        and gate.get("status") == "XEDITCRITIC_V3_FROZEN_TEST_PASS"
        and gate.get("all_development_refit_authorized") is True,
        "post-TEST configs require frozen TEST PASS",
    )
    return selected, gate


def freeze_refit_pass_count_v3(config: Mapping[str, Any], *, selected_arm: str) -> int:
    summaries = config.get("confirmation_candidate_summaries")
    _require(isinstance(summaries, list) and len(summaries) == 3, "refit requires three confirmation summaries")
    selected_passes = []
    seen = set()
    for spec in summaries:
        seed = int(spec["seed"])
        _require(seed in SEEDS and seed not in seen, "refit confirmation seed differs")
        seen.add(seed)
        summary = _json(Path(spec["summary_path"]))
        _require(summary.get("status") == "TERMINAL_CONFIRMATION_ARM_COMPLETE", "refit confirmation run is incomplete")
        _require(summary.get("arm") == selected_arm and int(summary.get("seed", -1)) == seed, "refit confirmation identity differs")
        _require(summary.get("development_test_outcomes_accessed") is False and summary.get("new_final_evaluation_outcomes_accessed") is False, "refit confirmation summary accessed protected outcome")
        selected_passes.append(int(summary.get("selected_pass", -1)))
    _require(seen == set(SEEDS) and all(value > 0 for value in selected_passes), "refit selected-pass inventory differs")
    median = statistics.median(selected_passes)
    _require(float(median).is_integer(), "refit selected-pass median is not an integer")
    return int(median)


def _base_training_config(
    config: Mapping[str, Any],
    *,
    selected_arm: str,
    seed: int,
    run_stage: str,
    output_root: Path,
    expected_train_count: int,
    expected_validation_count: int,
    passes: int,
) -> dict[str, Any]:
    template = config.get("training_template")
    _require(isinstance(template, Mapping), "post-TEST training template is absent")
    return {
        **dict(template),
        "run_stage": run_stage,
        "seed": seed,
        "selected_arm": selected_arm,
        "three_seed_gate_path": str(config["three_seed_gate_path"]),
        "atomic_frozen_test_path": str(config["atomic_frozen_test_path"]),
        "projection_paths": list(config["projection_paths"]),
        "edit_site_cache": str(config["edit_site_cache"]),
        "expected_record_count": int(config["expected_record_count"]),
        "expected_train_count": expected_train_count,
        "expected_validation_count": expected_validation_count,
        "withheld_development_test_record_count": 18292,
        "passes": passes,
        "output_root": str(output_root),
        "experiment_ledger_path": str(config["experiment_ledger_path"]),
    }


def prepare_refit_configs_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditcritic_v3_posttest_prepare.v1", "unexpected post-TEST prepare schema")
    _require(config.get("mode") == "REFIT", "post-TEST prepare mode is not REFIT")
    selected, _ = _posttest_selection(config)
    passes = freeze_refit_pass_count_v3(config, selected_arm=selected)
    total = int(config["expected_record_count"])
    _require(total == 107873, "all-Development record count changed")
    gpu_indices = [int(value) for value in config["physical_gpu_indices"]]
    _require(bool(gpu_indices) and set(gpu_indices) <= set(range(6)), "refit GPU scope differs")
    jobs = []
    root = Path(config["output_root"])
    for index, seed in enumerate(SEEDS):
        output = root / f"seed{seed}"
        jobs.append({
            "seed": seed,
            "arm": selected,
            "physical_gpu_index": gpu_indices[index % len(gpu_indices)],
            "config": _base_training_config(
                config,
                selected_arm=selected,
                seed=seed,
                run_stage="REFIT",
                output_root=output,
                expected_train_count=total,
                expected_validation_count=0,
                passes=passes,
            ),
            "summary_path": str(output / selected.lower() / "run_summary.json"),
        })
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_refit_job_manifest.v1",
        "status": "XEDITCRITIC_V3_REFIT_CONFIGS_PREPARED",
        "selected_arm": selected,
        "required_seeds": list(SEEDS),
        "refit_pass_count": passes,
        "job_count": 3,
        "jobs": jobs,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def prepare_loso_configs_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditcritic_v3_posttest_prepare.v1", "unexpected post-TEST prepare schema")
    _require(config.get("mode") == "LOSO", "post-TEST prepare mode is not LOSO")
    selected, _ = _posttest_selection(config)
    refit = _json(Path(config["refit_manifest_path"]))
    _require(
        refit.get("status") == "XEDITCRITIC_V3_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit.get("required_seeds") == list(SEEDS)
        and int(refit.get("completed_refit_count", -1)) == 3,
        "LOSO requires all three refits",
    )
    passes = int(refit["refit_pass_count"])
    records = records_from_projection_rows(
        load_projection_rows([Path(path) for path in config["projection_paths"]])
    )
    _require(len(records) == int(config["expected_record_count"]) == 107873, "LOSO all-Development count changed")
    study_counts = Counter(record.study for record in records)
    _require(set(study_counts) == set(POSTTEST_STUDIES_V3), "LOSO study inventory changed")
    gpu_indices = [int(value) for value in config["physical_gpu_indices"]]
    _require(bool(gpu_indices) and set(gpu_indices) <= set(range(6)), "LOSO GPU scope differs")
    jobs = []
    root = Path(config["output_root"])
    for seed in SEEDS:
        for study in POSTTEST_STUDIES_V3:
            for arm in (selected, "C0"):
                output = root / f"seed{seed}" / study
                training = _base_training_config(
                    config,
                    selected_arm=selected,
                    seed=seed,
                    run_stage="LOSO",
                    output_root=output,
                    expected_train_count=len(records) - study_counts[study],
                    expected_validation_count=study_counts[study],
                    passes=passes,
                )
                training.update({
                    "held_out_study": study,
                    "refit_manifest_path": str(config["refit_manifest_path"]),
                })
                jobs.append({
                    "seed": seed,
                    "held_out_study": study,
                    "arm": arm,
                    "physical_gpu_index": gpu_indices[len(jobs) % len(gpu_indices)],
                    "config": training,
                    "summary_path": str(output / arm.lower() / "run_summary.json"),
                })
    _require(len(jobs) == 42, "LOSO job count differs")
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_loso_job_manifest.v1",
        "status": "XEDITCRITIC_V3_LOSO_CONFIGS_PREPARED",
        "selected_arm": selected,
        "required_seeds": list(SEEDS),
        "held_out_studies": list(POSTTEST_STUDIES_V3),
        "study_record_counts": dict(sorted(study_counts.items())),
        "refit_pass_count": passes,
        "job_count": 42,
        "jobs": jobs,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_manifest_v3(payload: Mapping[str, Any], output_dir: Path) -> None:
    _require(not output_dir.exists(), f"post-TEST config output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for index, job in enumerate(payload["jobs"]):
        label = f"{index:02d}_seed{job['seed']}_{job['arm'].lower()}"
        if "held_out_study" in job:
            label += "_" + job["held_out_study"]
        (output_dir / f"{label}.json").write_text(
            json.dumps(job["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    payload = prepare_refit_configs_v3(config) if config.get("mode") == "REFIT" else prepare_loso_configs_v3(config)
    write_manifest_v3(payload, args.output_dir)
    print(json.dumps({key: value for key, value in payload.items() if key != "jobs"}, sort_keys=True))


if __name__ == "__main__":
    main()
