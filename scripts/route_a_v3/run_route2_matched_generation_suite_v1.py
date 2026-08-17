#!/usr/bin/env python3
"""Run the frozen Route 2 matched-budget generation baseline suite."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "run_route2_search_generation_baselines_v1.py"
FLOW_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "run_route2_base_flow_g0_validation_v1.py"
SCORE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "score_route2_generation_independent_evaluator_v1.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "evaluate_route2_generation_v1.py"
COMPOSE_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "build_route2_generation_baseline_selection_input_v2.py"
SELECT_SCRIPT = REPO_ROOT / "scripts" / "route_a_v3" / "select_route2_strongest_generation_baseline_v1.py"
UNGUIDED_METHOD = "unguided_learned_base_flow_g0"
PARALLEL_QUALITY_ONLY = "PARALLEL_SHARED_GPU_QUALITY_ONLY"
SERIAL_RUNTIME_VALID = "SERIAL_SAME_GPU_RUNTIME_VALID"


class MatchedGenerationSuiteError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedGenerationSuiteError(message)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"input is not an object: {path}")
    return payload


def validate_suite_inputs(
    protocol: Mapping[str, Any],
    jobs: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    flow_config: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_generation_matched_compute_repair_protocol.v1",
        "unexpected matched-compute protocol schema",
    )
    _require(
        jobs.get("schema_version")
        == "route_a_v3_route2_generation_independent_evaluator_jobs.v1",
        "unexpected evaluator-jobs schema",
    )
    _require(
        adjudication.get("status") == "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED"
        and adjudication.get("candidate_rerun_authorized") is True,
        "candidate generation is not authorized by the frozen evaluator adjudication",
    )
    _require(
        adjudication.get("development_test_outcomes_accessed") is False
        and adjudication.get("evaluation_outcomes_accessed") is False,
        "evaluator qualification accessed protected outcomes",
    )
    _require(protocol.get("evaluation_release_state") == "CLOSED", "Evaluation is not closed")
    _require(protocol.get("evaluation_outcomes_accessed") is False, "Evaluation was accessed")
    _require(protocol.get("guided_xeditflow_allowed") is False, "guided XEditFlow entered baseline suite")
    _require(
        protocol.get("independent_evaluator_must_be_frozen_before_candidate_generation") is True,
        "independent evaluator is not required to be pre-frozen",
    )
    execution_mode = str(protocol.get("gpu_stage_execution_mode", PARALLEL_QUALITY_ONLY))
    _require(
        execution_mode in {PARALLEL_QUALITY_ONLY, SERIAL_RUNTIME_VALID},
        "unknown GPU-stage execution mode",
    )
    runtime_comparison_valid = execution_mode == SERIAL_RUNTIME_VALID
    if runtime_comparison_valid:
        _require(protocol.get("same_gpu_cohort_required") is True, "runtime comparison does not require one GPU cohort")
        _require(protocol.get("parallel_gpu_jobs_allowed") is False, "runtime-valid suite permits concurrent GPU jobs")
        _require(protocol.get("runtime_comparison_allowed") is True, "runtime comparison is not authorized")
        _require(protocol.get("per_method_wall_time_required") is True, "per-method wall time is not required")
        _require(protocol.get("per_method_peak_vram_required") is True, "per-method peak VRAM is not required")
    else:
        _require(
            protocol.get("runtime_comparison_allowed", False) is False,
            "parallel shared-GPU execution cannot authorize runtime comparison",
        )

    required_methods = [str(value) for value in protocol["required_method_ids"]]
    job_by_method = {str(row["method_id"]): dict(row) for row in jobs["jobs"]}
    _require(len(job_by_method) == len(jobs["jobs"]), "evaluator job method is duplicated")
    _require(set(job_by_method) == set(required_methods), "evaluator jobs do not cover required methods")
    _require(UNGUIDED_METHOD in required_methods, "unguided Flow G0 method is absent")
    _require(len(required_methods) == 7, "matched suite must contain exactly seven full-cohort methods")

    shared_bindings = {
        "source_manifest_path": str(protocol["source_manifest_path"]),
        "evaluator_checkpoint_path": str(protocol["independent_evaluator_checkpoint_path"]),
        "guiding_checkpoint_path": str(protocol["guiding_checkpoint_path"]),
        "device": str(protocol["execution_device"]),
        "physical_gpu_index": int(protocol["physical_gpu_index"]),
    }
    for key, expected in shared_bindings.items():
        _require(jobs.get(key) == expected, f"evaluator jobs differ from protocol: {key}")
    _require(jobs.get("evaluator_frozen_before_candidate_generation") is True, "jobs do not freeze evaluator")
    _require(jobs.get("evaluation_outcomes_used_to_select_evaluator") == 0, "Evaluation selected evaluator")

    flow_output = Path(str(flow_config["output_directory"])) / "trajectories.private.jsonl"
    _require(
        Path(str(job_by_method[UNGUIDED_METHOD]["candidate_path"])) == flow_output,
        "Flow G0 output does not match the evaluator job candidate path",
    )
    _require(flow_config.get("guided_critic_used") is False, "guided critic entered unguided Flow G0")
    _require(flow_config.get("evaluation_outcomes_accessed") is False, "Flow G0 accessed Evaluation")
    _require(flow_config.get("device") == shared_bindings["device"], "Flow G0 device differs")
    _require(
        int(flow_config.get("physical_gpu_index")) == shared_bindings["physical_gpu_index"],
        "Flow G0 physical GPU differs",
    )
    _require(
        flow_config.get("source_eligibility_manifest") == shared_bindings["source_manifest_path"],
        "Flow G0 source manifest differs",
    )
    _require(int(flow_config.get("seed")) == int(protocol["seed"]), "Flow G0 seed differs")

    return {
        "required_methods": required_methods,
        "job_by_method": job_by_method,
        "shared_bindings": shared_bindings,
        "gpu_stage_execution_mode": execution_mode,
        "runtime_comparison_valid": runtime_comparison_valid,
    }


def build_generation_commands(
    protocol: Mapping[str, Any],
    flow_config_path: Path,
    suite: Mapping[str, Any],
) -> list[dict[str, Any]]:
    commands = []
    for method_id in suite["required_methods"]:
        job = suite["job_by_method"][method_id]
        if method_id == UNGUIDED_METHOD:
            command = [sys.executable, str(FLOW_SCRIPT), "--config", str(flow_config_path)]
        else:
            command = [
                sys.executable,
                str(SEARCH_SCRIPT),
                "--source-manifest",
                str(protocol["source_manifest_path"]),
                "--checkpoint",
                str(protocol["guiding_checkpoint_path"]),
                "--device",
                str(protocol["execution_device"]),
                "--physical-gpu-index",
                str(protocol["physical_gpu_index"]),
                "--method",
                method_id,
                "--max-critic-forwards",
                str(protocol["search_critic_forward_budget_per_source"]),
                "--beam-width",
                str(protocol["beam_width"]),
                "--genetic-population-size",
                str(protocol["genetic_population_size"]),
                "--oversample-factor",
                str(protocol["oversample_factor"]),
                "--exhaustive-space-limit",
                str(protocol["exhaustive_space_limit"]),
                "--seed",
                str(protocol["seed"]),
                "--output",
                str(job["candidate_path"]),
            ]
        commands.append({"name": method_id, "command": command})
    return commands


def build_scoring_configs(
    jobs: Mapping[str, Any],
    suite: Mapping[str, Any],
    config_directory: Path,
) -> list[dict[str, Any]]:
    config_directory.mkdir(parents=True, exist_ok=True)
    specs = []
    for method_id in suite["required_methods"]:
        job = suite["job_by_method"][method_id]
        config_path = config_directory / f"{method_id}_independent_evaluator_config_v1.json"
        _require(not config_path.exists(), f"scoring config already exists: {config_path}")
        config = {
            "schema_version": "route_a_v3_route2_generation_independent_evaluator_job.v1",
            "method_id": method_id,
            "evaluator_checkpoint_path": jobs["evaluator_checkpoint_path"],
            "guiding_checkpoint_path": jobs["guiding_checkpoint_path"],
            "source_manifest_path": jobs["source_manifest_path"],
            "evaluator_frozen_before_candidate_generation": jobs[
                "evaluator_frozen_before_candidate_generation"
            ],
            "evaluation_outcomes_used_to_select_evaluator": jobs[
                "evaluation_outcomes_used_to_select_evaluator"
            ],
            "device": jobs["device"],
            "physical_gpu_index": jobs["physical_gpu_index"],
            "candidate_path": job["candidate_path"],
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        specs.append(
            {
                "name": method_id,
                "command": [
                    sys.executable,
                    str(SCORE_SCRIPT),
                    "--config",
                    str(config_path),
                    "--output",
                    str(job["output_path"]),
                ],
            }
        )
    return specs


def build_evaluation_commands(
    protocol: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evaluation_root = Path(str(protocol["independent_evaluation_output_root"]))
    commands = []
    for method_id in suite["required_methods"]:
        job = suite["job_by_method"][method_id]
        commands.append(
            {
                "name": method_id,
                "command": [
                    sys.executable,
                    str(EVALUATE_SCRIPT),
                    "--source-manifest",
                    str(protocol["source_manifest_path"]),
                    "--candidates",
                    str(job["output_path"]),
                    "--measured-neighborhood",
                    str(protocol["measured_neighborhood_path"]),
                    "--measured-neighborhood-pool",
                    "DEVELOPMENT",
                    "--candidate-support-mode",
                    str(protocol["candidate_support_mode"]),
                    "--evaluation-release-state",
                    "CLOSED",
                    "--k",
                    "10",
                    "--output",
                    str(evaluation_root / f"{method_id}_evaluation_v2.json"),
                ],
            }
        )
    return commands


def run_parallel_stage(
    stage_name: str,
    specs: Sequence[Mapping[str, Any]],
    log_directory: Path,
) -> list[dict[str, Any]]:
    log_directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processes = []
    print(json.dumps({"event": "STAGE_STARTED", "stage": stage_name, "job_count": len(specs)}), flush=True)
    for spec in specs:
        stdout_path = log_directory / f"{stage_name}.{spec['name']}.stdout.log"
        stderr_path = log_directory / f"{stage_name}.{spec['name']}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            list(spec["command"]),
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        processes.append((spec, process, stdout_handle, stderr_handle, stdout_path, stderr_path))

    results = []
    for spec, process, stdout_handle, stderr_handle, stdout_path, stderr_path in processes:
        return_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        results.append(
            {
                "name": str(spec["name"]),
                "return_code": return_code,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "wall_time_seconds": None,
                "runtime_measurement_valid": False,
            }
        )
    print(
        json.dumps(
            {
                "event": "STAGE_COMPLETED",
                "stage": stage_name,
                "wall_time_seconds": time.time() - started,
                "return_codes": {result["name"]: result["return_code"] for result in results},
            }
        ),
        flush=True,
    )
    _require(all(result["return_code"] == 0 for result in results), f"stage failed: {stage_name}")
    return results


def run_serial_stage(
    stage_name: str,
    specs: Sequence[Mapping[str, Any]],
    log_directory: Path,
) -> list[dict[str, Any]]:
    """Run GPU jobs one at a time so per-method wall time is interpretable."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stage_started = time.time()
    results = []
    print(
        json.dumps(
            {
                "event": "STAGE_STARTED",
                "stage": stage_name,
                "job_count": len(specs),
                "execution_mode": SERIAL_RUNTIME_VALID,
            }
        ),
        flush=True,
    )
    for spec in specs:
        stdout_path = log_directory / f"{stage_name}.{spec['name']}.stdout.log"
        stderr_path = log_directory / f"{stage_name}.{spec['name']}.stderr.log"
        job_started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.run(
                list(spec["command"]),
                cwd=REPO_ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        result = {
            "name": str(spec["name"]),
            "return_code": process.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "wall_time_seconds": time.time() - job_started,
            "runtime_measurement_valid": True,
        }
        results.append(result)
        _require(process.returncode == 0, f"stage failed: {stage_name}/{spec['name']}")
    print(
        json.dumps(
            {
                "event": "STAGE_COMPLETED",
                "stage": stage_name,
                "execution_mode": SERIAL_RUNTIME_VALID,
                "wall_time_seconds": time.time() - stage_started,
                "return_codes": {result["name"]: result["return_code"] for result in results},
            }
        ),
        flush=True,
    )
    return results


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} is not finite/nonnegative")
    return result


def build_method_runtime_summary(
    protocol: Mapping[str, Any],
    flow_config: Mapping[str, Any],
    suite: Mapping[str, Any],
    stage_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not suite["runtime_comparison_valid"]:
        return {
            "gpu_stage_execution_mode": suite["gpu_stage_execution_mode"],
            "runtime_comparison_valid": False,
            "runtime_invalid_reason": "GPU_METHODS_EXECUTED_CONCURRENTLY_ON_ONE_DEVICE",
            "method_runtime": None,
        }

    candidate_results = {str(row["name"]): row for row in stage_results["candidate_generation"]}
    scoring_results = {str(row["name"]): row for row in stage_results["independent_evaluator_scoring"]}
    _require(set(candidate_results) == set(suite["required_methods"]), "candidate runtime coverage differs")
    _require(set(scoring_results) == set(suite["required_methods"]), "scoring runtime coverage differs")
    evaluation_root = Path(str(protocol["independent_evaluation_output_root"]))
    method_runtime: dict[str, dict[str, float]] = {}
    for method_id in suite["required_methods"]:
        candidate_result = candidate_results[method_id]
        scoring_result = scoring_results[method_id]
        _require(candidate_result.get("runtime_measurement_valid") is True, f"candidate runtime is invalid: {method_id}")
        _require(scoring_result.get("runtime_measurement_valid") is True, f"scoring runtime is invalid: {method_id}")
        candidate_wall = _finite_nonnegative(
            candidate_result.get("wall_time_seconds"), f"candidate wall time: {method_id}"
        )
        scoring_wall = _finite_nonnegative(
            scoring_result.get("wall_time_seconds"), f"scoring wall time: {method_id}"
        )
        evaluation = _read(evaluation_root / f"{method_id}_evaluation_v2.json")
        per_source = evaluation["generation"]["per_source"]
        candidate_peaks = [
            _finite_nonnegative(row["peak_vram_mb"], f"candidate peak VRAM: {method_id}")
            for row in per_source.values()
            if row.get("peak_vram_mb") is not None
        ]
        if method_id == UNGUIDED_METHOD:
            flow_summary = _read(Path(str(flow_config["output_directory"])) / "final_summary.json")
            candidate_peak = _finite_nonnegative(
                flow_summary.get("peak_vram_mb"), f"Flow candidate peak VRAM: {method_id}"
            )
        else:
            _require(candidate_peaks, f"candidate peak VRAM is absent: {method_id}")
            candidate_peak = max(candidate_peaks)
        scored_output = Path(str(suite["job_by_method"][method_id]["output_path"]))
        scoring_summary = _read(scored_output.with_suffix(scored_output.suffix + ".summary.json"))
        scoring_peak = _finite_nonnegative(
            scoring_summary.get("peak_vram_mb"), f"scoring peak VRAM: {method_id}"
        )
        method_runtime[method_id] = {
            "candidate_generation_wall_time_seconds": candidate_wall,
            "candidate_generation_peak_vram_mb": candidate_peak,
            "independent_evaluator_scoring_wall_time_seconds": scoring_wall,
            "independent_evaluator_scoring_peak_vram_mb": scoring_peak,
            "total_serial_gpu_wall_time_seconds": candidate_wall + scoring_wall,
            "peak_vram_mb": max(candidate_peak, scoring_peak),
        }
    return {
        "gpu_stage_execution_mode": suite["gpu_stage_execution_mode"],
        "runtime_comparison_valid": True,
        "runtime_invalid_reason": None,
        "method_runtime": method_runtime,
    }


def run_serial_command(name: str, command: Sequence[str], log_directory: Path) -> dict[str, Any]:
    return run_parallel_stage(name, [{"name": name, "command": list(command)}], log_directory)[0]


def execute(
    *,
    protocol_path: Path,
    jobs_path: Path,
    evaluator_adjudication_path: Path,
    flow_config_path: Path,
    output_summary_path: Path,
) -> dict[str, Any]:
    _require(not output_summary_path.exists(), f"output summary already exists: {output_summary_path}")
    protocol = _read(protocol_path)
    jobs = _read(jobs_path)
    adjudication = _read(evaluator_adjudication_path)
    flow_config = _read(flow_config_path)
    suite = validate_suite_inputs(protocol, jobs, adjudication, flow_config)

    evaluation_root = Path(str(protocol["independent_evaluation_output_root"]))
    log_directory = evaluation_root / "suite_logs_v1"
    scoring_config_directory = evaluation_root / "independent_evaluator_job_configs_v1"
    selection_input_path = evaluation_root / "baseline_selection_input_v2.json"
    strongest_path = evaluation_root / "strongest_generation_baseline_v2.json"
    stage_results: dict[str, Any] = {}
    started = time.time()
    gpu_stage_runner = run_serial_stage if suite["runtime_comparison_valid"] else run_parallel_stage

    try:
        stage_results["candidate_generation"] = gpu_stage_runner(
            "candidate_generation",
            build_generation_commands(protocol, flow_config_path, suite),
            log_directory,
        )
        stage_results["independent_evaluator_scoring"] = gpu_stage_runner(
            "independent_evaluator_scoring",
            build_scoring_configs(jobs, suite, scoring_config_directory),
            log_directory,
        )
        stage_results["development_open_support_evaluation"] = run_parallel_stage(
            "development_open_support_evaluation",
            build_evaluation_commands(protocol, suite),
            log_directory,
        )
        stage_results["selection_input"] = run_serial_command(
            "selection_input",
            [
                sys.executable,
                str(COMPOSE_SCRIPT),
                "--protocol",
                str(protocol_path),
                "--evaluator-jobs",
                str(jobs_path),
                "--output",
                str(selection_input_path),
            ],
            log_directory,
        )
        stage_results["strongest_selection"] = run_serial_command(
            "strongest_selection",
            [
                sys.executable,
                str(SELECT_SCRIPT),
                "--input",
                str(selection_input_path),
                "--output",
                str(strongest_path),
            ],
            log_directory,
        )
        strongest = _read(strongest_path)
        runtime_summary = build_method_runtime_summary(
            protocol,
            flow_config,
            suite,
            stage_results,
        )
        summary = {
            "schema_version": "route_a_v3_route2_matched_generation_suite.v1",
            "status": "MATCHED_GENERATION_BASELINE_SUITE_COMPLETED",
            "required_method_ids": suite["required_methods"],
            "source_manifest_path": protocol["source_manifest_path"],
            "candidate_budget_per_source": protocol["candidate_budget_per_source"],
            "critic_forward_budget_per_source": protocol["search_critic_forward_budget_per_source"],
            "forward_equivalent_budget_per_source": protocol["forward_equivalent_budget_per_source"],
            "candidate_support_mode": protocol["candidate_support_mode"],
            "strongest_generation_baseline_id": strongest["strongest_generation_baseline_id"],
            "point_leader_method_id": strongest["point_leader_method_id"],
            "strongest_output_path": str(strongest_path),
            "selection_input_path": str(selection_input_path),
            "physical_gpu_index": protocol["physical_gpu_index"],
            "execution_device": protocol["execution_device"],
            "cpu_fallback_allowed": False,
            "evaluation_outcomes_accessed": False,
            "guided_xeditflow_run": False,
            "stage_results": stage_results,
            "wall_time_seconds": time.time() - started,
            "scientific_claim_status": "INDEPENDENT_EVALUATOR_ONLY_MEASURED_OUTCOME_NOT_ESTABLISHED",
            **runtime_summary,
        }
    except Exception as exc:
        summary = {
            "schema_version": "route_a_v3_route2_matched_generation_suite.v1",
            "status": "MATCHED_GENERATION_BASELINE_SUITE_FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "stage_results": stage_results,
            "evaluation_outcomes_accessed": False,
            "guided_xeditflow_run": False,
            "wall_time_seconds": time.time() - started,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluator-jobs", type=Path, required=True)
    parser.add_argument("--evaluator-adjudication", type=Path, required=True)
    parser.add_argument("--flow-config", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    execute(
        protocol_path=args.protocol,
        jobs_path=args.evaluator_jobs,
        evaluator_adjudication_path=args.evaluator_adjudication,
        flow_config_path=args.flow_config,
        output_summary_path=args.output_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
