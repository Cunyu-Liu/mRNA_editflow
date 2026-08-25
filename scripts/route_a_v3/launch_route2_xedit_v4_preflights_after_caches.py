#!/usr/bin/env python3
"""Authorize and launch both V4 preflights after exact-HEAD caches finish."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
C3_REFERENCE = (
    ROOT
    / "experiments/xeditcritic_v3/screen_seed_20260830/"
    "c3_v4_reference_read_once.json"
)
PREFLIGHT_JOB_RUNNER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_preflight_job.py"
)
PREFLIGHT_SEQUENCE_RUNNER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xedit_v4_preflight_sequence.py"
)


class XEditV4PreflightLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4PreflightLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def write_atomic(path: Path, payload: dict[str, object]) -> None:
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
        index, free = (part.strip() for part in line.split(",", maxsplit=1))
        values[int(index)] = int(free)
    return values


def require_preflight_gpu_availability(
    free_memory: dict[int, int],
    *,
    critic_gpu: int,
    setflow_gpu: int,
    critic_minimum_free_mib: int,
    setflow_minimum_free_mib: int,
) -> None:
    requirements = (
        ("Critic", critic_gpu, critic_minimum_free_mib),
        ("SetFlow", setflow_gpu, setflow_minimum_free_mib),
    )
    failures = [
        f"{component} GPU{gpu} has {free_memory.get(gpu, -1)} MiB free; "
        f"requires at least {minimum} MiB"
        for component, gpu, minimum in requirements
        if free_memory.get(gpu, -1) < minimum
    ]
    if failures:
        snapshot = {gpu: free_memory.get(gpu, -1) for gpu in range(6)}
        raise XEditV4PreflightLaunchError(
            "; ".join(failures)
            + f"; allowed_gpu_free_memory_mib={json.dumps(snapshot, sort_keys=True)}"
        )


def require_preflight_gpu_layout(
    *, critic_gpu: int, setflow_gpu: int, sequential_single_gpu: bool
) -> None:
    require(critic_gpu in range(6), "Critic GPU is outside physical GPU 0–5")
    require(setflow_gpu in range(6), "SetFlow GPU is outside physical GPU 0–5")
    if sequential_single_gpu:
        require(
            critic_gpu == setflow_gpu,
            "sequential single-GPU preflights require one shared GPU",
        )
    else:
        require(critic_gpu != setflow_gpu, "concurrent preflights require distinct GPUs")


def require_summary(path: Path, *, expected_head: str, component: str) -> None:
    require(path.is_file(), f"{component} cache summary is absent")
    summary = json.loads(path.read_text(encoding="utf-8"))
    require(
        str(summary.get("git_head")) == expected_head,
        f"{component} cache summary HEAD changed",
    )
    if component == "critic":
        require(
            summary.get("development_test_outcomes_accessed") is False,
            "critic cache summary reports a Development TEST read",
        )
        require(
            summary.get("evaluation_outcomes_accessed") is False,
            "critic cache summary reports an Evaluation read",
        )
    else:
        require(
            int(summary.get("development_test_outcome_reads", -1)) == 0,
            "setflow cache adoption reports a Development TEST read",
        )
        require(
            int(summary.get("new_final_evaluation_outcome_reads", -1)) == 0,
            "setflow cache adoption reports a new Evaluation read",
        )


def expected_authorization_status(component: str) -> str:
    require(component in {"critic", "setflow"}, "unknown V4 preflight component")
    prefix = "XEDITCRITIC" if component == "critic" else "XEDITSETFLOW"
    return f"{prefix}_V4_PREFLIGHT_AUTHORIZED"


def component_paths() -> dict[str, dict[str, Path | int]]:
    return {
        "critic": {
            "config": WORKTREE
            / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json",
            "preflight": WORKTREE
            / "scripts/route_a_v3/preflight_route2_xeditcritic_v4.py",
            "cache_summary": ROOT
            / "pretrained_features/xeditcritic_v4/"
            "frozen_bottom_six_chunk_cache_v1.summary.json",
            "cache_failure": ROOT
            / "pretrained_features/xeditcritic_v4/"
            "frozen_bottom_six_chunk_cache_v1.failure.json",
            "output": ROOT
            / "experiments/xeditcritic_v4/screen_seed_20260907/preflight.json",
            "failure": ROOT
            / "experiments/xeditcritic_v4/screen_seed_20260907/preflight.failure.json",
        },
        "setflow": {
            "config": WORKTREE
            / "configs/route_a_v3_route2_xeditsetflow_v4_screen_v1.json",
            "preflight": WORKTREE
            / "scripts/route_a_v3/preflight_route2_xeditsetflow_v4.py",
            "cache_summary": ROOT
            / "pretrained_features/xeditsetflow_v4/"
            "source_token_cache_v3_adoption_receipt_v1.json",
            "cache_failure": ROOT
            / "pretrained_features/xeditsetflow_v4/"
            "source_token_cache_v3_adoption_receipt_v1.failure.json",
            "output": ROOT
            / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight.json",
            "failure": ROOT
            / "experiments/xeditsetflow_v4/screen_seed_20260911/preflight.failure.json",
        },
    }


def preflight_job_command(
    component: str,
    paths: dict[str, Path | int],
    *,
    authorization: Path,
    runtime: Path,
    log: Path,
    current_head: str,
) -> list[str]:
    return [
        str(PYTHON),
        str(PREFLIGHT_JOB_RUNNER),
        "--component",
        component,
        "--python",
        str(PYTHON),
        "--preflight",
        str(paths["preflight"]),
        "--config",
        str(paths["config"]),
        "--authorization",
        str(authorization),
        "--physical-gpu-index",
        str(paths["gpu"]),
        "--output",
        str(paths["output"]),
        "--failure",
        str(paths["failure"]),
        "--runtime",
        str(runtime),
        "--log",
        str(log),
        "--git-head",
        current_head,
    ]


def run(
    current_head: str,
    experiment_head: str,
    *,
    critic_gpu: int,
    setflow_gpu: int,
    sequential_single_gpu: bool = False,
    critic_minimum_free_mib: int = 38000,
    setflow_minimum_free_mib: int = 20000,
) -> dict[str, object]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None,
        "expected current Git HEAD is invalid",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", experiment_head) is not None,
        "expected cache experiment HEAD is invalid",
    )
    require(PYTHON.is_file(), "formal Python is absent")
    require(PREFLIGHT_JOB_RUNNER.is_file(), "current-HEAD preflight job runner is absent")
    if sequential_single_gpu:
        require(
            PREFLIGHT_SEQUENCE_RUNNER.is_file(),
            "current-HEAD sequential preflight runner is absent",
        )
    require_preflight_gpu_layout(
        critic_gpu=critic_gpu,
        setflow_gpu=setflow_gpu,
        sequential_single_gpu=sequential_single_gpu,
    )
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head,
        "A100 worktree is not at expected current HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    require(C3_REFERENCE.is_file(), "C3 read-once reference is absent")

    a100_audit = ROOT / f"audits/a100_current_head_v4/sync_tests_{current_head}.json"
    require(a100_audit.is_file(), "exact current-HEAD A100 test audit is absent")
    free_memory = gpu_free_memory_mib()
    require_preflight_gpu_availability(
        free_memory,
        critic_gpu=critic_gpu,
        setflow_gpu=setflow_gpu,
        critic_minimum_free_mib=critic_minimum_free_mib,
        setflow_minimum_free_mib=setflow_minimum_free_mib,
    )

    authorization_root = (
        ROOT
        / "authorizations/xedit_v4"
        / f"preflight_{experiment_head}_runner_{current_head}"
    )
    authorization_staging_root = authorization_root.with_name(
        authorization_root.name + ".partial"
    )
    runtime_root = (
        ROOT
        / "experiments/xedit_v4"
        / f"preflight_launch_{experiment_head}_runner_{current_head}"
    )
    log_root = (
        ROOT
        / "logs/xedit_v4"
        / f"preflight_launch_{experiment_head}_runner_{current_head}"
    )
    require(not authorization_root.exists(), "preflight authorizations already exist")
    require(
        not authorization_staging_root.exists(),
        "partial preflight authorization package already exists",
    )
    require(not runtime_root.exists(), "preflight launch runtime already exists")

    authorizer = (
        WORKTREE / "scripts/route_a_v3/authorize_route2_xedit_v4_screen_stages.py"
    )
    require(authorizer.is_file(), "current-HEAD preflight authorizer is absent")
    components = component_paths()
    components["critic"]["gpu"] = critic_gpu
    components["setflow"]["gpu"] = setflow_gpu
    for component, paths in components.items():
        for key in ("config", "preflight"):
            require(Path(paths[key]).is_file(), f"{component} {key} is absent")
        require(
            not Path(paths["cache_failure"]).exists(),
            f"{component} cache has a technical failure",
        )
        require_summary(
            Path(paths["cache_summary"]),
            expected_head=experiment_head,
            component=component,
        )
        require(
            not Path(paths["output"]).exists(),
            f"{component} preflight output already exists",
        )
        require(
            not Path(paths["failure"]).exists(),
            f"{component} preflight failure already exists",
        )
        authorization = authorization_staging_root / f"{component}.json"
        command(
            [
                str(PYTHON),
                str(authorizer),
                "--component",
                component,
                "--stage",
                "preflight",
                "--screen-config",
                str(paths["config"]),
                "--c3-reference",
                str(C3_REFERENCE),
                "--a100-audit",
                str(a100_audit),
                "--cache-summary",
                str(paths["cache_summary"]),
                "--cache-experiment-head",
                experiment_head,
                "--output",
                str(authorization),
            ]
        )
        require(authorization.is_file(), f"{component} preflight authorization is absent")
        payload = json.loads(authorization.read_text(encoding="utf-8"))
        require(
            payload.get("status") == expected_authorization_status(component)
            and payload.get("component") == component
            and payload.get("authorized_git_head") == current_head
            and payload.get("cache_experiment_head") == experiment_head,
            f"{component} preflight authorization content is invalid",
        )
    os.replace(authorization_staging_root, authorization_root)

    launches: dict[str, object] = {}
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    if sequential_single_gpu:
        sequence_jobs: list[dict[str, object]] = []
        for order, (component, paths) in enumerate(components.items()):
            authorization = authorization_root / f"{component}.json"
            runtime = runtime_root / f"{component}.runtime.json"
            log = log_root / f"{component}.log"
            wrapper_log = log_root / f"{component}.wrapper.log"
            gpu = int(paths["gpu"])
            sequence_jobs.append(
                {
                    "component": component,
                    "order": order,
                    "physical_gpu_index": gpu,
                    "command": preflight_job_command(
                        component,
                        paths,
                        authorization=authorization,
                        runtime=runtime,
                        log=log,
                        current_head=current_head,
                    ),
                    "output": str(paths["output"]),
                    "failure": str(paths["failure"]),
                    "runtime": str(runtime),
                    "wrapper_log": str(wrapper_log),
                }
            )
        sequence_config = runtime_root / "sequence_config.json"
        sequence_runtime = runtime_root / "sequence.runtime.json"
        sequence_failure = runtime_root / "sequence.failure.json"
        sequence_wrapper_log = log_root / "sequence.wrapper.log"
        write_atomic(
            sequence_config,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_sequence_config.v1",
                "status": "V4_PREFLIGHT_SEQUENCE_PREPARED",
                "git_head": current_head,
                "experiment_head": experiment_head,
                "physical_gpu_index": critic_gpu,
                "component_order": ["critic", "setflow"],
                "jobs": sequence_jobs,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        stream = sequence_wrapper_log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(PREFLIGHT_SEQUENCE_RUNNER),
                "--config",
                str(sequence_config),
                "--runtime",
                str(sequence_runtime),
                "--failure",
                str(sequence_failure),
                "--git-head",
                current_head,
            ],
            cwd=WORKTREE,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stream.close()
        for job in sequence_jobs:
            component = str(job["component"])
            launches[component] = {
                "wrapper_pid": process.pid,
                "shared_sequence_scheduler": True,
                "launch_order": int(job["order"]),
                "physical_gpu_index": critic_gpu,
                "free_memory_mib_before_launch": free_memory[critic_gpu],
                "authorization": str(authorization_root / f"{component}.json"),
                "runtime": str(job["runtime"]),
                "output": str(job["output"]),
                "failure": str(job["failure"]),
                "preflight_log": str(log_root / f"{component}.log"),
                "wrapper_log": str(job["wrapper_log"]),
            }
        sequence = {
            "scheduler_pid": process.pid,
            "config": str(sequence_config),
            "runtime": str(sequence_runtime),
            "failure": str(sequence_failure),
            "wrapper_log": str(sequence_wrapper_log),
        }
        launch_mode = "SEQUENTIAL_SINGLE_GPU"
    else:
        sequence = None
        launch_mode = "CONCURRENT_DISTINCT_GPUS"
        for component, paths in components.items():
            authorization = authorization_root / f"{component}.json"
            runtime = runtime_root / f"{component}.runtime.json"
            log = log_root / f"{component}.log"
            wrapper_log = log_root / f"{component}.wrapper.log"
            stream = wrapper_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                preflight_job_command(
                    component,
                    paths,
                    authorization=authorization,
                    runtime=runtime,
                    log=log,
                    current_head=current_head,
                ),
                cwd=WORKTREE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stream.close()
            gpu = int(paths["gpu"])
            launches[component] = {
                "wrapper_pid": process.pid,
                "shared_sequence_scheduler": False,
                "physical_gpu_index": gpu,
                "free_memory_mib_before_launch": free_memory[gpu],
                "authorization": str(authorization),
                "runtime": str(runtime),
                "output": str(paths["output"]),
                "failure": str(paths["failure"]),
                "preflight_log": str(log),
                "wrapper_log": str(wrapper_log),
            }

    manifest = runtime_root / "launch_manifest.json"
    write_atomic(
        manifest,
        {
            "schema_version": "route_a_v3_route2_xedit_v4_preflight_launch_manifest.v1",
            "status": "V4_PREFLIGHT_JOBS_LAUNCHED",
            "git_head": current_head,
            "experiment_head": experiment_head,
            "launch_mode": launch_mode,
            "sequence": sequence,
            "c3_reference": str(C3_REFERENCE),
            "a100_audit": str(a100_audit),
            "gpu_free_memory_mib_before_launch": free_memory,
            "jobs": launches,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return {"manifest": str(manifest), "jobs": launches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--experiment-head", required=True)
    parser.add_argument("--critic-gpu", type=int, required=True)
    parser.add_argument("--setflow-gpu", type=int, required=True)
    parser.add_argument("--sequential-single-gpu", action="store_true")
    parser.add_argument("--critic-minimum-free-mib", type=int, default=38000)
    parser.add_argument("--setflow-minimum-free-mib", type=int, default=20000)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.expected_head,
                arguments.experiment_head,
                critic_gpu=arguments.critic_gpu,
                setflow_gpu=arguments.setflow_gpu,
                sequential_single_gpu=arguments.sequential_single_gpu,
                critic_minimum_free_mib=arguments.critic_minimum_free_mib,
                setflow_minimum_free_mib=arguments.setflow_minimum_free_mib,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
