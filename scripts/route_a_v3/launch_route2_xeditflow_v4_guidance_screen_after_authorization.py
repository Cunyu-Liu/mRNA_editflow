#!/usr/bin/env python3
"""Launch the exact V4 value and 18-combination guidance screen package."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROTOCOL = WORKTREE / "configs/route_a_v3_route2_xeditflow_v4_guidance_protocol_v1.json"
PREPARER = WORKTREE / "scripts/route_a_v3/prepare_route2_xeditflow_v4_value_configs.py"
SCHEDULER = WORKTREE / "scripts/route_a_v3/run_route2_xeditflow_v4_guidance_screen_scheduler.py"


class XEditFlowV4GuidanceScreenLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowV4GuidanceScreenLaunchError(message)


def command(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=check,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


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


def fixed_guidance_gpu_assignment() -> tuple[int, ...]:
    return tuple(index % 6 for index in range(18))


def validate_manifest(
    manifest: Mapping[str, Any], *, config_root: Path
) -> None:
    require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_value_config_manifest.v1"
        and manifest.get("status")
        == "XEDITFLOW_V4_VALUE_CONFIGS_PREPARED_NOT_STARTED"
        and int(manifest.get("base_flow_training_seed", -1)) == 20260912
        and int(manifest.get("rollout_job_count", -1)) == 1
        and int(manifest.get("critic_score_job_count", -1)) == 1
        and int(manifest.get("value_target_package_count", -1)) == 6
        and int(manifest.get("value_training_job_count", -1)) == 6
        and int(manifest.get("later_guidance_combination_count", -1)) == 18
        and tuple(manifest.get("guidance_gpu_assignment", ()))
        == fixed_guidance_gpu_assignment()
        and manifest.get("beta_max_used_in_value_target_or_training") is False
        and manifest.get("independent_evaluator_used") is False
        and manifest.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and manifest.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 guidance value-config manifest changed",
    )
    path_fields = (
        "value_training_config_paths",
        "guidance_smc_config_paths",
        "guidance_critic_config_paths",
        "guidance_closed_config_paths",
        "guidance_open_metric_config_paths",
        "guidance_independent_evaluator_config_paths",
        "guidance_independent_evaluator_comparison_config_paths",
    )
    expected_lengths = (6, 18, 18, 18, 18, 18, 18)
    for field, length in zip(path_fields, expected_lengths, strict=True):
        paths = tuple(Path(value) for value in manifest.get(field, ()))
        require(
            len(paths) == length
            and all(path.parent == config_root and path.is_file() for path in paths),
            f"V4 guidance manifest {field} changed",
        )
    require(
        len(manifest.get("guidance_result_paths", ())) == 18,
        "V4 guidance manifest result grid changed",
    )


def directory_failure(output: Path) -> Path:
    return output.with_name(output.name + ".failed.json")


def run(current_head: str, experiment_head: str) -> dict[str, Any]:
    for value, label in (
        (current_head, "current"),
        (experiment_head, "experiment"),
    ):
        require(
            re.fullmatch(r"[0-9a-f]{40}", value) is not None,
            f"{label} Git HEAD is invalid",
        )
    require(
        PYTHON.is_file() and PROTOCOL.is_file() and PREPARER.is_file() and SCHEDULER.is_file(),
        "formal V4 guidance screen runtime is absent",
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    authorization_decision = read_json(
        ROOT
        / f"experiments/xedit_v4/guidance_authorization_{experiment_head}/decision.json"
    )
    require(
        authorization_decision.get("status") == "XEDITFLOW_V4_GUIDANCE_AUTHORIZED"
        and authorization_decision.get("git_head") == experiment_head
        and authorization_decision.get("critic_readiness_state") == "READY"
        and authorization_decision.get("setflow_readiness_state") == "READY"
        and authorization_decision.get("guidance_authorized") is True
        and authorization_decision.get("guidance_training_or_sampling_launched")
        is False
        and authorization_decision.get("development_test_reopened") is False
        and authorization_decision.get(
            "development_test_outcomes_accessed_after_atomic_test"
        )
        is False
        and int(authorization_decision.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "V4 joint guidance authorization decision is absent or changed",
    )
    protocol = read_json(PROTOCOL)
    authorization = Path(protocol["authorization_output"])
    require(authorization.is_file(), "V4 joint guidance authorization is absent")
    config_root = Path(protocol["runtime_config_root"])
    output_root = Path(protocol["guidance_screen_output_root"])
    runtime_root = (
        ROOT
        / "experiments/xedit_v4"
        / f"guidance_screen_execution_{experiment_head}_runner_{current_head}"
    )
    require(
        not config_root.exists() and not output_root.exists() and not runtime_root.exists(),
        "V4 guidance screen config, output, or runtime already exists",
    )
    critic_preflight = read_json(
        ROOT / "experiments/xeditcritic_v4/screen_seed_20260907/preflight_attempt_3/preflight.json"
    )
    setflow_preflight = read_json(
        ROOT / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight_attempt_3/preflight.json"
    )
    require(
        critic_preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and critic_preflight.get("git_head") == experiment_head
        and setflow_preflight.get("status") == "XEDITSETFLOW_V4_PREFLIGHT_PASS"
        and setflow_preflight.get("git_head") == experiment_head,
        "V4 guidance screen preflight identities changed",
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
    require(
        all(free_memory[gpu] >= required_mib for gpu in range(6)),
        "not every GPU 0-5 has enough memory for the fixed six-way guidance screen",
    )
    primary_gpu = min(range(6), key=lambda gpu: (-free_memory[gpu], gpu))
    command(
        [
            str(PYTHON),
            str(PREPARER),
            "--protocol", str(PROTOCOL),
            "--rollout-gpu", str(primary_gpu),
            "--critic-gpu", str(primary_gpu),
            "--value-gpus", "0", "1", "2", "3", "4", "5",
            "--guidance-gpus",
            *(str(value) for value in fixed_guidance_gpu_assignment()),
        ]
    )
    manifest_path = config_root / "manifest.json"
    manifest = read_json(manifest_path)
    validate_manifest(manifest, config_root=config_root)

    log_root = (
        ROOT
        / "logs/xedit_v4"
        / f"guidance_screen_execution_{experiment_head}_runner_{current_head}"
    )
    failure_root = runtime_root / "failures"
    rollout_config_path = Path(manifest["config_paths"]["value_rollout.json"])
    score_config_path = Path(manifest["config_paths"]["value_critic_score.json"])
    target_config_path = Path(manifest["config_paths"]["value_target_grid.json"])
    rollout_config = read_json(rollout_config_path)
    score_config = read_json(score_config_path)
    target_config = read_json(target_config_path)
    rollout_output = Path(rollout_config["output_dir"])
    score_output = Path(score_config["output_dir"])
    target_output = Path(target_config["output_root"])
    serial = [
        {
            "job_key": "value_rollouts",
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/generate_route2_xeditflow_value_rollouts_v4.py"),
                "--config", str(rollout_config_path),
                "--output-dir", str(rollout_output),
            ],
            "success_path": str(rollout_output / "run_summary.json"),
            "failure_path": str(directory_failure(rollout_output)),
            "log_path": str(log_root / "value_rollouts.log"),
        },
        {
            "job_key": "value_critic_scoring",
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/score_route2_xeditflow_value_rollouts_v4.py"),
                "--config", str(score_config_path),
                "--output-dir", str(score_output),
            ],
            "success_path": str(score_output / "run_summary.json"),
            "failure_path": str(directory_failure(score_output)),
            "log_path": str(log_root / "value_critic_scoring.log"),
        },
        {
            "job_key": "value_target_grid",
            "command": [
                str(PYTHON),
                str(WORKTREE / "scripts/route_a_v3/build_route2_xeditflow_value_targets_v4.py"),
                "--config", str(target_config_path),
                "--output-root", str(target_output),
            ],
            "success_path": str(target_output / "manifest.json"),
            "failure_path": str(failure_root / "value_target_grid.failed.json"),
            "log_path": str(log_root / "value_target_grid.log"),
        },
    ]
    value_queues = []
    for gpu, config_path_value in enumerate(manifest["value_training_config_paths"]):
        config_path = Path(config_path_value)
        config = read_json(config_path)
        require(
            int(config["physical_gpu_index"]) == gpu,
            "V4 value training GPU assignment changed",
        )
        output = Path(config["output_dir"])
        value_queues.append(
            {
                "physical_gpu_index": gpu,
                "jobs": [
                    {
                        "job_key": f"value_train:{config_path.stem}",
                        "command": [
                            str(PYTHON),
                            str(WORKTREE / "scripts/route_a_v3/train_route2_xeditflow_value_v4.py"),
                            "--config", str(config_path),
                            "--output-dir", str(output),
                        ],
                        "success_path": str(output / "run_summary.json"),
                        "failure_path": str(directory_failure(output)),
                        "log_path": str(log_root / f"{config_path.stem}.log"),
                    }
                ],
            }
        )

    config_lists = [
        manifest["guidance_smc_config_paths"],
        manifest["guidance_critic_config_paths"],
        manifest["guidance_closed_config_paths"],
        manifest["guidance_open_metric_config_paths"],
        manifest["guidance_independent_evaluator_config_paths"],
        manifest["guidance_independent_evaluator_comparison_config_paths"],
    ]
    queues = {gpu: [] for gpu in range(6)}
    for index, paths in enumerate(zip(*config_lists, strict=True)):
        smc_path, critic_path, closed_path, open_path, evaluator_path, comparison_path = (
            Path(value) for value in paths
        )
        smc = read_json(smc_path)
        critic = read_json(critic_path)
        closed = read_json(closed_path)
        open_metric = read_json(open_path)
        evaluator = read_json(evaluator_path)
        comparison = read_json(comparison_path)
        gpu = fixed_guidance_gpu_assignment()[index]
        require(
            all(
                int(config["physical_gpu_index"]) == gpu
                for config in (smc, critic, closed, evaluator)
            ),
            "V4 guidance chain GPU assignment changed",
        )
        combination_id = smc_path.stem.removeprefix("smc_")
        smc_output = Path(smc["output_dir"])
        critic_output = Path(critic["output_dir"])
        closed_output = Path(closed["output_dir"])
        evaluator_output = Path(evaluator["output_path"])
        jobs = [
            {
                "job_key": f"guidance:{combination_id}:smc",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/run_route2_xeditflow_smc_v4.py"), "--config", str(smc_path), "--output-dir", str(smc_output)],
                "success_path": str(smc_output / "run_summary.json"),
                "failure_path": str(directory_failure(smc_output)),
                "log_path": str(log_root / f"{combination_id}_smc.log"),
            },
            {
                "job_key": f"guidance:{combination_id}:critic",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/score_route2_xeditflow_candidates_v4.py"), "--config", str(critic_path), "--output-dir", str(critic_output)],
                "success_path": str(critic_output / "run_summary.json"),
                "failure_path": str(directory_failure(critic_output)),
                "log_path": str(log_root / f"{combination_id}_critic.log"),
            },
            {
                "job_key": f"guidance:{combination_id}:closed",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/evaluate_route2_xeditflow_closed_neighborhood_v4.py"), "--config", str(closed_path), "--output-dir", str(closed_output)],
                "success_path": str(closed_output / "run_summary.json"),
                "failure_path": str(directory_failure(closed_output)),
                "log_path": str(log_root / f"{combination_id}_closed.log"),
            },
            {
                "job_key": f"guidance:{combination_id}:open",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/evaluate_route2_xeditflow_open_generation_v4.py"), "--config", str(open_path), "--output", str(open_metric["output_path"])],
                "success_path": str(open_metric["output_path"]),
                "failure_path": str(failure_root / f"{combination_id}_open.failed.json"),
                "log_path": str(log_root / f"{combination_id}_open.log"),
            },
            {
                "job_key": f"guidance:{combination_id}:evaluator",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/score_route2_generation_independent_evaluator_v1.py"), "--config", str(evaluator_path), "--output", str(evaluator_output)],
                "success_path": str(evaluator_output.with_suffix(evaluator_output.suffix + ".summary.json")),
                "failure_path": str(evaluator_output.with_suffix(evaluator_output.suffix + ".failed.json")),
                "log_path": str(log_root / f"{combination_id}_evaluator.log"),
            },
            {
                "job_key": f"guidance:{combination_id}:comparison",
                "command": [str(PYTHON), str(WORKTREE / "scripts/route_a_v3/compare_route2_xeditflow_independent_evaluator_v4.py"), "--config", str(comparison_path), "--output", str(comparison["output_path"])],
                "success_path": str(comparison["output_path"]),
                "failure_path": str(failure_root / f"{combination_id}_comparison.failed.json"),
                "log_path": str(log_root / f"{combination_id}_comparison.log"),
            },
        ]
        queues[gpu].append({"combination_id": combination_id, "jobs": jobs})
    require(
        all(len(chains) == 3 for chains in queues.values()),
        "V4 guidance queues are not three chains per GPU",
    )
    gate_output = output_root / "guidance_screen_gate.json"
    adjudication = {
        "job_key": "guidance_screen_adjudication",
        "command": [
            str(PYTHON),
            str(WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditflow_guidance_screen_v4.py"),
            "--manifest", str(manifest_path),
            "--output", str(gate_output),
        ],
        "success_path": str(gate_output),
        "failure_path": str(failure_root / "guidance_screen_adjudication.failed.json"),
        "log_path": str(log_root / "guidance_screen_adjudication.log"),
    }
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_schedule.v1",
        "status": "FROZEN_VALUE_AND_EXACT_18_COMBINATION_SCHEDULE",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "manifest_path": str(manifest_path),
        "serial_value_prerequisites": serial,
        "value_training_queues": value_queues,
        "guidance_queues": [
            {"physical_gpu_index": gpu, "chains": queues[gpu]}
            for gpu in range(6)
        ],
        "adjudication": adjudication,
        "required_free_memory_mib": required_mib,
        "gpu_free_memory_mib_before_launch": free_memory,
        "active_performance_output_read": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }
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
        "schema_version": "route_a_v3_route2_xeditflow_v4_guidance_screen_launch.v1",
        "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_SCHEDULER_LAUNCHED",
        "git_head": current_head,
        "experiment_head": experiment_head,
        "scheduler_pid": process.pid,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "wrapper_log": str(wrapper_log),
        "guidance_screen_gate_path": str(gate_output),
        "development_test_reopened": False,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--experiment-head", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.expected_head, arguments.experiment_head),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
