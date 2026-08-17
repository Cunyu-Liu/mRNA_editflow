#!/usr/bin/env python3
"""Run the frozen 190-source exhaustive reference after the full suite succeeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "run_route2_search_generation_baselines_v1.py"
SCORE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "score_route2_generation_independent_evaluator_v1.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "evaluate_route2_generation_v1.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "compare_route2_exhaustive_small_space_reference_v1.py"


class ExhaustiveReferenceSuiteError(RuntimeError):
    pass


class StageFailure(ExhaustiveReferenceSuiteError):
    def __init__(self, result: Mapping[str, Any]):
        self.result = dict(result)
        super().__init__(f"stage failed: {self.result['stage']}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExhaustiveReferenceSuiteError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_inputs(
    config: Mapping[str, Any],
    scoring: Mapping[str, Any],
    protocol: Mapping[str, Any],
    full_suite: Mapping[str, Any],
    evaluator_adjudication: Mapping[str, Any],
) -> None:
    _require(
        config.get("schema_version") == "route_a_v3_route2_exhaustive_small_space_reference.v1",
        "exhaustive reference config schema changed",
    )
    _require(
        config.get("scientific_role")
        == "REAL_SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_NOT_FULL_COHORT_STRONGEST_SELECTOR",
        "exhaustive scientific role changed",
    )
    _require(
        full_suite.get("status") == "MATCHED_GENERATION_BASELINE_SUITE_COMPLETED",
        "full matched generation suite is not complete",
    )
    _require(full_suite.get("evaluation_outcomes_accessed") is False, "full suite accessed Evaluation")
    _require(full_suite.get("guided_xeditflow_run") is False, "full suite ran guided XEditFlow")
    _require(
        evaluator_adjudication.get("status") == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
        and evaluator_adjudication.get("candidate_rerun_authorized") is True,
        "independent evaluator did not authorize candidate rerun",
    )
    _require(
        evaluator_adjudication.get("evaluation_outcomes_accessed") is False,
        "independent evaluator accessed Evaluation",
    )
    _require(config.get("source_cohort_count") == 190, "exhaustive source cohort count changed")
    _require(config.get("legal_space_size_per_source") == 151, "exhaustive legal space changed")
    _require(config.get("candidate_budget_per_source") == protocol.get("candidate_budget_per_source") == 32, "candidate budget changed")
    _require(config.get("critic_forward_budget_per_source") == protocol.get("search_critic_forward_budget_per_source") == 256, "critic budget changed")
    _require(config.get("forward_equivalent_budget_per_source") == protocol.get("forward_equivalent_budget_per_source") == 320, "total forward budget changed")
    _require(config.get("candidate_budget_per_source") <= config.get("critic_forward_budget_per_source"), "candidate budget exceeds critic budget")
    _require(config.get("guiding_checkpoint_path") == protocol.get("guiding_checkpoint_path"), "guiding checkpoint changed")
    _require(config.get("independent_evaluator_checkpoint_path") == protocol.get("independent_evaluator_checkpoint_path"), "independent evaluator changed")
    _require(scoring.get("evaluator_checkpoint_path") == config.get("independent_evaluator_checkpoint_path"), "scoring evaluator changed")
    _require(scoring.get("guiding_checkpoint_path") == config.get("guiding_checkpoint_path"), "scoring guide changed")
    _require(scoring.get("source_manifest_path") == config.get("source_manifest_path"), "scoring source subset changed")
    _require(scoring.get("candidate_path") == config.get("candidate_output_path"), "scoring candidate path changed")
    _require(scoring.get("output_path") == config.get("independent_scored_output_path"), "scoring output path changed")
    _require(scoring.get("evaluator_frozen_before_candidate_generation") is True, "evaluator was not pre-frozen")
    _require(scoring.get("evaluation_outcomes_used_to_select_evaluator") == 0, "Evaluation selected evaluator")
    _require(protocol.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT", "candidate support mode changed")
    _require(protocol.get("evaluation_release_state") == "CLOSED", "Evaluation release state changed")
    _require(config.get("device") == protocol.get("execution_device") == scoring.get("device") == "cuda:6", "execution device changed")
    _require(config.get("physical_gpu_index") == protocol.get("physical_gpu_index") == scoring.get("physical_gpu_index") == 6, "physical GPU changed")
    _require(config.get("evaluation_outcomes_accessed") is False, "exhaustive config accessed Evaluation")
    _require(config.get("full_cohort_strongest_selector_eligible") is False, "exhaustive reference entered strongest selector")
    _require(config.get("guided_xeditflow_allowed") is False, "guided XEditFlow was enabled")


def build_commands(
    config: Mapping[str, Any],
    scoring_config_path: Path,
    protocol: Mapping[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": "exhaustive_candidate_generation",
            "command": [
                sys.executable,
                str(SEARCH_SCRIPT),
                "--source-manifest", str(config["source_manifest_path"]),
                "--checkpoint", str(config["guiding_checkpoint_path"]),
                "--device", str(config["device"]),
                "--physical-gpu-index", str(config["physical_gpu_index"]),
                "--method", "exhaustive",
                "--max-critic-forwards", str(config["critic_forward_budget_per_source"]),
                "--beam-width", str(config["beam_width"]),
                "--genetic-population-size", str(config["genetic_population_size"]),
                "--oversample-factor", str(config["oversample_factor"]),
                "--exhaustive-space-limit", str(config["exhaustive_space_limit"]),
                "--seed", str(config["seed"]),
                "--output", str(config["candidate_output_path"]),
            ],
        },
        {
            "stage": "independent_evaluator_scoring",
            "command": [
                sys.executable,
                str(SCORE_SCRIPT),
                "--config", str(scoring_config_path),
                "--output", str(config["independent_scored_output_path"]),
            ],
        },
        {
            "stage": "development_open_support_evaluation",
            "command": [
                sys.executable,
                str(EVALUATE_SCRIPT),
                "--source-manifest", str(config["source_manifest_path"]),
                "--candidates", str(config["independent_scored_output_path"]),
                "--measured-neighborhood", str(config["measured_neighborhood_path"]),
                "--measured-neighborhood-pool", "DEVELOPMENT",
                "--candidate-support-mode", str(protocol["candidate_support_mode"]),
                "--evaluation-release-state", "CLOSED",
                "--k", "10",
                "--output", str(config["generation_evaluation_output_path"]),
            ],
        },
        {
            "stage": "small_space_reference_comparison",
            "command": [
                sys.executable,
                str(COMPARE_SCRIPT),
                "--config", str(config_path),
                "--output", str(config["comparison_output_path"]),
            ],
        },
    ]


def run_stage(spec: Mapping[str, Any], log_directory: Path) -> dict[str, Any]:
    stage = str(spec["stage"])
    stdout_path = log_directory / f"{stage}.stdout.log"
    stderr_path = log_directory / f"{stage}.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            list(spec["command"]),
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    result = {
        "stage": stage,
        "return_code": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "wall_time_seconds": time.time() - started,
    }
    if completed.returncode != 0:
        raise StageFailure(result)
    return result


def validate_comparison_output(comparison: Mapping[str, Any]) -> None:
    _require(
        comparison.get("status") == "SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_COMPLETED",
        "comparison did not reach its complete status",
    )
    _require(
        comparison.get("measured_neighborhood_comparison_included") is True,
        "comparison omitted Development measured-neighborhood evidence",
    )
    _require(
        comparison.get("measured_candidate_support_mode") == "OPEN_GENERATED_SUPPORT",
        "comparison measured-support mode changed",
    )
    _require(
        comparison.get("unknown_generated_outcomes_treated_as_zero") is False,
        "comparison treated unknown generated outcomes as zero",
    )
    _require(
        comparison.get("measured_superiority_claim_established") is False,
        "open-support comparison claimed measured superiority",
    )


def execute(config_path: Path, output_summary_path: Path | None = None) -> dict[str, Any]:
    config = _read_json(config_path)
    scoring_path = REPO_ROOT / str(config["independent_evaluator_scoring_config"])
    protocol_path = REPO_ROOT / str(config["matched_compute_protocol_config"])
    scoring = _read_json(scoring_path)
    protocol = _read_json(protocol_path)
    full_suite = _read_json(Path(str(config["full_cohort_suite_summary_path"])))
    adjudication = _read_json(Path(str(config["independent_evaluator_adjudication_path"])))
    validate_inputs(config, scoring, protocol, full_suite, adjudication)

    output_summary = output_summary_path or Path(str(config["execution_summary_path"]))
    _require(not output_summary.exists(), f"output summary already exists: {output_summary}")
    for key in (
        "candidate_output_path",
        "independent_scored_output_path",
        "generation_evaluation_output_path",
        "comparison_output_path",
    ):
        _require(not Path(str(config[key])).exists(), f"stage output already exists: {config[key]}")
    _require(Path(str(config["measured_neighborhood_path"])).is_file(), "exhaustive measured neighborhood is absent")
    log_directory = Path(str(config["execution_log_directory"]))
    _require(not log_directory.exists(), f"execution log directory already exists: {log_directory}")
    log_directory.mkdir(parents=True)

    stages: list[dict[str, Any]] = []
    started = time.time()
    try:
        for spec in build_commands(config, scoring_path, protocol, config_path):
            stages.append(run_stage(spec, log_directory))
        comparison = _read_json(Path(str(config["comparison_output_path"])))
        validate_comparison_output(comparison)
        summary = {
            "schema_version": "route_a_v3_route2_exhaustive_small_space_reference_suite.v1",
            "status": "EXHAUSTIVE_SMALL_SPACE_REFERENCE_SUITE_COMPLETED",
            "source_count": comparison["source_count"],
            "method_count": len(comparison["method_summaries"]),
            "physical_gpu_index": config["physical_gpu_index"],
            "device": config["device"],
            "cpu_fallback_used": False,
            "evaluation_outcomes_accessed": False,
            "guided_xeditflow_run": False,
            "measured_neighborhood_comparison_included": True,
            "measured_candidate_support_mode": comparison["measured_candidate_support_mode"],
            "unknown_generated_outcomes_treated_as_zero": False,
            "measured_superiority_claim_established": False,
            "stage_results": stages,
            "comparison_output_path": config["comparison_output_path"],
            "wall_time_seconds": time.time() - started,
            "scientific_claim_status": comparison["scientific_claim_status"],
        }
    except Exception as exc:
        if isinstance(exc, StageFailure):
            stages.append(exc.result)
        summary = {
            "schema_version": "route_a_v3_route2_exhaustive_small_space_reference_suite.v1",
            "status": "EXHAUSTIVE_SMALL_SPACE_REFERENCE_SUITE_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "evaluation_outcomes_accessed": False,
            "guided_xeditflow_run": False,
            "stage_results": stages,
            "wall_time_seconds": time.time() - started,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    args = parser.parse_args()
    execute(args.config, args.output_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
