#!/usr/bin/env python3
"""Launch the frozen V4 three-seed matched-compute comparison package."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROTOCOL = WORKTREE / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
PREPARER = WORKTREE / "scripts/route_a_v3/prepare_route2_xeditflow_final_generation_configs_v4.py"
SCHEDULER = WORKTREE / "scripts/route_a_v3/run_route2_xeditflow_v4_final_scheduler.py"
SEEDS = (20260912, 20260913, 20260914)
METHODS = (
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
)


class XEditFlowV4FinalLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowV4FinalLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def directory_failure(output: Path) -> Path:
    return output.with_name(output.name + ".failed.json")


def gpu_free_memory_mib() -> dict[int, int]:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free = (value.strip() for value in line.split(",", maxsplit=1))
        values[int(index)] = int(free)
    return values


def _config_job(
    *,
    key: str,
    script: str,
    config_path: Path,
    config: Mapping[str, Any],
    log_root: Path,
    failure_root: Path,
    output_kind: str,
    gpu_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    command_line = [str(PYTHON), str(WORKTREE / "scripts/route_a_v3" / script)]
    output: Path
    if output_kind == "directory":
        output = Path(str(config["output_dir"]))
        command_line += ["--config", str(config_path), "--output-dir", str(output)]
        success = output / "run_summary.json"
        failure = directory_failure(output)
    elif output_kind == "file":
        output = Path(str(config["output_path"]))
        command_line += ["--config", str(config_path), "--output", str(output)]
        success = output
        failure = failure_root / f"{key}.failed.json"
    elif output_kind == "independent_evaluator":
        output = Path(str(config["output_path"]))
        command_line += ["--config", str(config_path), "--output", str(output)]
        success = output.with_suffix(output.suffix + ".summary.json")
        failure = output.with_suffix(output.suffix + ".failed.json")
    elif output_kind == "adapter":
        output = Path(str(config["output_dir"]))
        command_line += ["--config", str(config_path)]
        success = output / "adapter_summary.json"
        failure = failure_root / f"{key}.failed.json"
    elif output_kind == "timing":
        output = Path(str(config["output_dir"]))
        command_line += ["--config", str(config_path)]
        success = output / "run_summary.json"
        failure = failure_root / f"{key}.failed.json"
    else:
        raise XEditFlowV4FinalLaunchError(f"unknown job output kind: {output_kind}")
    return {
        "job_key": key,
        "command": command_line,
        "physical_gpu_indices": list(gpu_indices),
        "success_path": str(success),
        "failure_path": str(failure),
        "log_path": str(log_root / f"{key.replace(':', '_')}.log"),
    }


def validate_manifest(manifest: Mapping[str, Any], config_root: Path) -> None:
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_generation_manifest.v4"
        and manifest.get("status")
        == "XEDITFLOW_V4_FINAL_GENERATION_CONFIGS_PREPARED_NOT_STARTED"
        and tuple(manifest.get("required_base_flow_training_seeds", ())) == SEEDS
        and int(manifest.get("written_runtime_config_count", -1)) == 97
        and len(manifest.get("written_runtime_config_paths", ())) == 97
        and all(
            Path(str(path)).parent == config_root and Path(str(path)).is_file()
            for path in manifest.get("written_runtime_config_paths", ())
        )
        and manifest.get("same_decoder_seed_streams_across_methods_and_seeds")
        is True
        and int(manifest.get("candidate_cap_per_source", -1)) == 32
        and int(manifest.get("forward_equivalent_ceiling_per_source", -1)) == 320
        and manifest.get("final_three_seed_gate_may_run_only_after_all_seed_jobs_terminal")
        is True
        and manifest.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and manifest.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 final generation manifest changed",
    )


def build_schedule(
    manifest: Mapping[str, Any],
    *,
    config_root: Path,
    log_root: Path,
    failure_root: Path,
    runtime_manifest: Path,
    current_head: str,
    experiment_head: str,
    guidance_runner_head: str,
    diagnostic_peak_plus_two_gib_mib: int,
    free_memory_mib: Mapping[int, int],
) -> dict[str, Any]:
    validate_manifest(manifest, config_root)

    def config(name: str) -> tuple[Path, dict[str, Any]]:
        path = config_root / name
        require(path.is_file(), f"V4 final runtime config is absent: {path}")
        return path, read_json(path)

    prerequisite_queues: list[dict[str, Any]] = []
    for seed, value_gpu, critic_gpu in ((20260913, 0, 4), (20260914, 1, 5)):
        jobs: list[dict[str, Any]] = []
        specs = (
            (
                "value_rollout",
                "generate_route2_xeditflow_value_rollouts_v4.py",
                "directory",
                (value_gpu,),
            ),
            (
                "value_critic_score",
                "score_route2_xeditflow_value_rollouts_v4.py",
                "directory",
                (critic_gpu,),
            ),
            (
                "value_target",
                "build_route2_xeditflow_final_value_target_v4.py",
                "directory",
                (),
            ),
            (
                "value_training",
                "train_route2_xeditflow_value_v4.py",
                "directory",
                (value_gpu,),
            ),
        )
        for suffix, script, kind, gpus in specs:
            path, payload = config(f"seed_{seed}_{suffix}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:{suffix}",
                    script=script,
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind=kind,
                    gpu_indices=gpus,
                )
            )
        prerequisite_queues.append(
            {"queue_key": f"value_seed_{seed}", "jobs": jobs}
        )
    timing_path, timing = config("strongest_matched_baseline_timing.json")
    prerequisite_queues.append(
        {
            "queue_key": "strongest_timing",
            "jobs": [
                _config_job(
                    key="strongest_timing",
                    script="run_route2_xeditflow_strongest_timing_v4.py",
                    config_path=timing_path,
                    config=timing,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="timing",
                    gpu_indices=(2,),
                )
            ],
        }
    )

    seed_chains: list[dict[str, Any]] = []
    for seed, generation_gpu, critic_gpu in zip(SEEDS, (0, 1, 2), (3, 4, 5), strict=True):
        jobs: list[dict[str, Any]] = []
        path, payload = config(f"seed_{seed}_strongest_adapter.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:strongest_adapter",
                script="adapt_route2_xeditflow_strongest_baseline_v4.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="adapter",
            )
        )
        path, payload = config(f"seed_{seed}_full_smc.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:generation:full_soft_value_smc",
                script="run_route2_xeditflow_smc_v4.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="directory",
                gpu_indices=(generation_gpu,),
            )
        )
        for method in METHODS[1:]:
            path, payload = config(f"seed_{seed}_control_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:generation:{method}",
                    script="run_route2_xeditflow_matched_controls_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="directory",
                    gpu_indices=(generation_gpu,),
                )
            )
        for method in METHODS:
            path, payload = config(f"seed_{seed}_terminal_critic_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:terminal_critic:{method}",
                    script="score_route2_xeditflow_candidates_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="directory",
                    gpu_indices=(critic_gpu,),
                )
            )
        for method in METHODS:
            path, payload = config(f"seed_{seed}_open_metric_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:open:{method}",
                    script="evaluate_route2_xeditflow_open_generation_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="file",
                )
            )
        for method in METHODS[:2]:
            path, payload = config(f"seed_{seed}_closed_exact_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:closed_exact:{method}",
                    script="evaluate_route2_xeditflow_closed_neighborhood_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="directory",
                    gpu_indices=(generation_gpu,),
                )
            )
        for method in METHODS[2:]:
            path, payload = config(f"seed_{seed}_closed_control_score_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:closed_score:{method}",
                    script="score_route2_xeditflow_closed_controls_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="directory",
                    gpu_indices=(critic_gpu,),
                )
            )
        for method in (*METHODS[2:], "strongest_matched_baseline"):
            path, payload = config(f"seed_{seed}_closed_metric_{method}.json")
            jobs.append(
                _config_job(
                    key=f"seed_{seed}:closed_metric:{method}",
                    script="evaluate_route2_xeditflow_closed_scores_v4.py",
                    config_path=path,
                    config=payload,
                    log_root=log_root,
                    failure_root=failure_root,
                    output_kind="file",
                )
            )
        path, payload = config(f"seed_{seed}_independent_evaluator.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:independent_evaluator",
                script="score_route2_generation_independent_evaluator_v1.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="independent_evaluator",
                gpu_indices=(critic_gpu,),
            )
        )
        path, payload = config(f"seed_{seed}_independent_evaluator_comparison.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:independent_evaluator_comparison",
                script="compare_route2_xeditflow_independent_evaluator_v4.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="file",
            )
        )
        path, payload = config(f"seed_{seed}_equal_wall_time.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:equal_wall_time",
                script="build_route2_xeditflow_equal_wall_time_sensitivity_v4.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="file",
            )
        )
        path, payload = config(f"seed_{seed}_final_seed_evidence.json")
        jobs.append(
            _config_job(
                key=f"seed_{seed}:final_evidence",
                script="assemble_route2_xeditflow_final_seed_evidence_v4.py",
                config_path=path,
                config=payload,
                log_root=log_root,
                failure_root=failure_root,
                output_kind="directory",
            )
        )
        require(len(jobs) == 29, f"V4 final seed job count differs: {seed}")
        seed_chains.append({"queue_key": f"seed_{seed}", "jobs": jobs})

    compose_path, compose = config("final_comparison_compose.json")
    comparison_manifest = Path(str(compose["output_path"]))
    adjudication_output = Path(str(manifest["final_adjudication_output_path"]))
    finalization_jobs = [
        {
            "job_key": "compose_final_comparison",
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/compose_route2_xeditflow_final_comparison_manifest_v4.py"),
                "--config",
                str(compose_path),
            ],
            "physical_gpu_indices": [],
            "success_path": str(comparison_manifest),
            "failure_path": str(failure_root / "compose_final_comparison.failed.json"),
            "log_path": str(log_root / "compose_final_comparison.log"),
        },
        {
            "job_key": "adjudicate_final_comparison",
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditflow_final_v4.py"),
                "--manifest",
                str(comparison_manifest),
                "--output",
                str(adjudication_output),
            ],
            "physical_gpu_indices": [],
            "success_path": str(adjudication_output),
            "failure_path": str(failure_root / "adjudicate_final_comparison.failed.json"),
            "log_path": str(log_root / "adjudicate_final_comparison.log"),
        },
    ]
    return {
        "schema_version": "route_a_v3_route2_xeditflow_v4_final_schedule.v1",
        "status": "FROZEN_THREE_SEED_MATCHED_COMPUTE_SCHEDULE",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "guidance_runner_head": guidance_runner_head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "config_manifest_path": str(config_root / "manifest.json"),
        "prerequisite_queues": prerequisite_queues,
        "seed_chains": seed_chains,
        "finalization_jobs": finalization_jobs,
        "diagnostic_peak_plus_two_gib_mib": int(
            diagnostic_peak_plus_two_gib_mib
        ),
        "gpu_free_memory_mib_before_launch": {
            str(key): int(value) for key, value in free_memory_mib.items()
        },
        "free_memory_gate_applied": False,
        "gpu_selection_policy": "FROZEN_PHYSICAL_GPU_ASSIGNMENT",
        "active_performance_output_read": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(
    current_head: str,
    experiment_head: str,
    guidance_runner_head: str,
    strongest_closed_score_table: Path,
) -> dict[str, Any]:
    for value, label in (
        (current_head, "current"),
        (experiment_head, "experiment"),
        (guidance_runner_head, "guidance runner"),
    ):
        require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, f"{label} Git HEAD is invalid")
    require(
        all(path.is_file() for path in (PYTHON, PROTOCOL, PREPARER, SCHEDULER)),
        "formal V4 final runtime is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head
        and not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is not clean at expected current HEAD",
    )
    protocol = read_json(PROTOCOL)
    gate_path = Path(str(protocol["guidance_screen_output_root"])) / "guidance_screen_gate.json"
    gate = read_json(gate_path)
    require(
        gate.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
        and gate.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and int(gate.get("base_flow_training_seed", -1)) == 20260912
        and int(gate.get("combination_count", -1)) == 18,
        "V4 guidance screen is not terminal and frozen",
    )
    guidance_runtime = read_json(
        ROOT
        / "experiments/xedit_v4"
        / f"guidance_screen_execution_{experiment_head}_runner_{guidance_runner_head}"
        / "runtime.json"
    )
    require(
        guidance_runtime.get("status") == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
        and guidance_runtime.get("git_head") == guidance_runner_head
        and guidance_runtime.get("experiment_head") == experiment_head
        and guidance_runtime.get("development_test_reopened") is False
        and guidance_runtime.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and int(guidance_runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 guidance screen runtime identity or protected-read boundary differs",
    )
    critic_preflight = read_json(
        ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_5/preflight.json"
    )
    setflow_preflight = read_json(
        ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_5/preflight.json"
    )
    require(
        critic_preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and critic_preflight.get("git_head") == experiment_head
        and setflow_preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and setflow_preflight.get("git_head") == experiment_head,
        "V4 final preflight identities changed",
    )
    required_mib = math.ceil(
        (
            max(
                float(critic_preflight["selected_peak_allocated_gib"]),
                float(setflow_preflight["peak_memory_allocated_gib"]),
            )
            + 2.0
        )
        * 1024
    )
    free_memory = gpu_free_memory_mib()
    require(set(free_memory).issuperset(range(6)), "physical GPU inventory 0-5 is incomplete")
    config_root = Path(str(protocol["runtime_config_root"])).parent / "final_three_seed_v1"
    output_root = Path(str(protocol["guidance_screen_output_root"])).parent / "final_three_seed"
    runtime_root = (
        ROOT
        / "experiments/xedit_v4"
        / f"final_execution_{experiment_head}_guidance_{guidance_runner_head}_runner_{current_head}"
    )
    require(
        strongest_closed_score_table.is_file()
        and str(strongest_closed_score_table).startswith(str(ROOT) + "/"),
        "pre-frozen strongest closed score table is absent or outside Route 2",
    )
    require(
        not config_root.exists() and not output_root.exists() and not runtime_root.exists(),
        "V4 final configs, outputs, or runtime already exist",
    )
    command(
        [
            str(PYTHON),
            str(PREPARER),
            "--protocol",
            str(PROTOCOL),
            "--critic-readiness",
            str(protocol["critic_readiness_path"]),
            "--setflow-confirmation",
            str(protocol["setflow_confirmation_path"]),
            "--critic-refit-manifest",
            str(protocol["critic_refit_manifest_path"]),
            "--source-data-audit",
            str(protocol["source_level_data_audit_path"]),
            "--guidance-screen-gate",
            str(gate_path),
            "--strongest-closed-score-table",
            str(strongest_closed_score_table),
            "--generation-gpus",
            "0",
            "1",
            "2",
            "--critic-gpus",
            "3",
            "4",
            "5",
            "--value-gpus",
            "0",
            "1",
            "--strongest-timing-gpu",
            "2",
            "--output-dir",
            str(config_root),
        ]
    )
    manifest = read_json(config_root / "manifest.json")
    validate_manifest(manifest, config_root)
    log_root = (
        ROOT
        / "logs/xedit_v4"
        / f"final_execution_{experiment_head}_guidance_{guidance_runner_head}_runner_{current_head}"
    )
    failure_root = runtime_root / "failures"
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule = build_schedule(
        manifest,
        config_root=config_root,
        log_root=log_root,
        failure_root=failure_root,
        runtime_manifest=runtime_manifest,
        current_head=current_head,
        experiment_head=experiment_head,
        guidance_runner_head=guidance_runner_head,
        diagnostic_peak_plus_two_gib_mib=required_mib,
        free_memory_mib=free_memory,
    )
    schedule_path = runtime_root / "schedule.json"
    write_atomic(schedule_path, schedule)
    wrapper_log = log_root / "scheduler.wrapper.log"
    stream = wrapper_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_final_launch.v1",
        "status": "XEDITFLOW_V4_FINAL_SCHEDULER_LAUNCHED",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "guidance_runner_head": guidance_runner_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "final_adjudication_path": str(manifest["final_adjudication_output_path"]),
        "development_test_reopened": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--experiment-head", required=True)
    parser.add_argument("--guidance-runner-head", required=True)
    parser.add_argument("--strongest-closed-score-table", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.expected_head,
                arguments.experiment_head,
                arguments.guidance_runner_head,
                arguments.strongest_closed_score_table,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
