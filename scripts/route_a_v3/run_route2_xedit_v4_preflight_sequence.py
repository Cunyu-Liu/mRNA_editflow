#!/usr/bin/env python3
"""Run the two authorized V4 preflights sequentially on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
EXPECTED_COMPONENT_ORDER = ("critic", "setflow")


class XEditV4PreflightSequenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditV4PreflightSequenceError(message)


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def load_and_validate_config(path: Path, *, git_head: str) -> dict[str, Any]:
    require(path.is_file(), "sequential preflight config is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xedit_v4_preflight_sequence_config.v1",
        "sequential preflight config schema changed",
    )
    require(payload.get("git_head") == git_head, "sequential preflight HEAD changed")
    gpu = int(payload.get("physical_gpu_index", -1))
    require(gpu in range(6), "sequential preflight GPU is outside physical GPU 0–5")
    jobs = payload.get("jobs")
    require(isinstance(jobs, list), "sequential preflight jobs are absent")
    require(
        tuple(job.get("component") for job in jobs if isinstance(job, dict))
        == EXPECTED_COMPONENT_ORDER,
        "sequential preflight component order changed",
    )
    for job in jobs:
        require(isinstance(job, dict), "sequential preflight job is invalid")
        require(
            int(job.get("physical_gpu_index", -1)) == gpu,
            "sequential preflight job GPU changed",
        )
        command = job.get("command")
        require(
            isinstance(command, list)
            and command
            and all(isinstance(value, str) and value for value in command),
            "sequential preflight job command is invalid",
        )
        for field in ("output", "failure", "runtime", "wrapper_log"):
            require(
                isinstance(job.get(field), str) and bool(job[field]),
                f"sequential preflight job {field} is invalid",
            )
    return payload


def current_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def run(config_path: Path, runtime_path: Path, *, git_head: str) -> dict[str, object]:
    require(current_git_head() == git_head, "A100 worktree HEAD changed")
    config = load_and_validate_config(config_path, git_head=git_head)
    require(not runtime_path.exists(), "sequential preflight runtime already exists")
    started = time.time()
    completed: list[dict[str, object]] = []
    for order, job in enumerate(config["jobs"]):
        component = str(job["component"])
        output = Path(job["output"])
        failure = Path(job["failure"])
        require(
            not output.exists() and not failure.exists(),
            f"{component} preflight already has a terminal artifact",
        )
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_sequence_runtime.v1",
                "status": "RUNNING",
                "git_head": git_head,
                "physical_gpu_index": int(config["physical_gpu_index"]),
                "scheduler_pid": os.getpid(),
                "current_component": component,
                "current_order": order,
                "completed": completed,
                "started_unix_seconds": started,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        wrapper_log = Path(job["wrapper_log"])
        wrapper_log.parent.mkdir(parents=True, exist_ok=True)
        with wrapper_log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                list(job["command"]),
                cwd=WORKTREE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        terminal_count = int(output.is_file()) + int(failure.is_file())
        require(
            terminal_count == 1,
            f"{component} sequential preflight did not publish exactly one terminal",
        )
        completed.append(
            {
                "component": component,
                "order": order,
                "return_code": int(result.returncode),
                "terminal_kind": "OUTPUT" if output.is_file() else "FAILURE",
                "output": str(output),
                "failure": str(failure),
                "runtime": str(job["runtime"]),
            }
        )
    terminal = {
        "schema_version": "route_a_v3_route2_xedit_v4_preflight_sequence_runtime.v1",
        "status": "TERMINAL_COMPLETE",
        "git_head": git_head,
        "physical_gpu_index": int(config["physical_gpu_index"]),
        "scheduler_pid": os.getpid(),
        "component_order": list(EXPECTED_COMPONENT_ORDER),
        "completed": completed,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_path, terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    arguments = parser.parse_args()
    if arguments.failure.exists():
        raise XEditV4PreflightSequenceError(
            "sequential preflight failure already exists"
        )
    try:
        result = run(arguments.config, arguments.runtime, git_head=arguments.git_head)
    except Exception as exc:
        write_atomic(
            arguments.failure,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_sequence_failure.v1",
                "status": "TECHNICAL_FAILURE",
                "git_head": arguments.git_head,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_path": str(arguments.runtime),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
