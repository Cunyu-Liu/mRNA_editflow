#!/usr/bin/env python3
"""Run one authorized V4 preflight and publish exactly one terminal state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


class XEditV4PreflightJobError(RuntimeError):
    pass


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def terminal_kind(*, output_present: bool) -> str:
    """An atomically published PASS/PAUSE output is the unique terminal."""

    return "OUTPUT" if output_present else "FAILURE"


def run(arguments: argparse.Namespace) -> None:
    if arguments.output.exists() or arguments.failure.exists():
        raise XEditV4PreflightJobError("preflight already has a terminal artifact")
    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [
        str(arguments.python),
        str(arguments.preflight),
        "--config",
        str(arguments.config),
        "--authorization",
        str(arguments.authorization),
        "--physical-gpu-index",
        str(arguments.physical_gpu_index),
    ]
    with arguments.log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_atomic(
            arguments.runtime,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_runtime.v1",
                "status": "RUNNING",
                "component": arguments.component,
                "wrapper_pid": os.getpid(),
                "preflight_pid": process.pid,
                "git_head": arguments.git_head,
                "physical_gpu_index": arguments.physical_gpu_index,
                "started_unix_seconds": started,
                "output_path": str(arguments.output),
                "failure_path": str(arguments.failure),
                "log_path": str(arguments.log),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        return_code = process.wait()

    # Each formal preflight atomically publishes its complete PASS or PAUSE
    # document as its final substantive action.  That document remains the
    # unique terminal even if stdout/teardown later changes the return code.
    if terminal_kind(output_present=arguments.output.is_file()) == "OUTPUT":
        result = json.loads(arguments.output.read_text(encoding="utf-8"))
        write_atomic(
            arguments.runtime,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_runtime.v1",
                "status": "TERMINAL_COMPLETE",
                "component": arguments.component,
                "preflight_status": result.get("status"),
                "preflight_passed": result.get("passed"),
                "wrapper_pid": os.getpid(),
                "preflight_pid": process.pid,
                "git_head": arguments.git_head,
                "physical_gpu_index": arguments.physical_gpu_index,
                "started_unix_seconds": started,
                "finished_unix_seconds": time.time(),
                "preflight_return_code": return_code,
                "output_path": str(arguments.output),
                "failure_path": str(arguments.failure),
                "failure_published": False,
                "log_path": str(arguments.log),
                "development_test_outcome_reads": int(
                    result.get("development_test_outcome_reads", -1)
                ),
                "new_final_evaluation_outcome_reads": int(
                    result.get("new_final_evaluation_outcome_reads", -1)
                ),
            },
        )
        return

    failure = {
        "schema_version": "route_a_v3_route2_xedit_v4_preflight_failure.v1",
        "status": "TECHNICAL_FAILURE",
        "component": arguments.component,
        "wrapper_pid": os.getpid(),
        "preflight_pid": process.pid,
        "git_head": arguments.git_head,
        "physical_gpu_index": arguments.physical_gpu_index,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "return_code": return_code,
        "output_present": False,
        "log_path": str(arguments.log),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(arguments.failure, failure)
    write_atomic(
        arguments.runtime,
        {
            **failure,
            "schema_version": "route_a_v3_route2_xedit_v4_preflight_runtime.v1",
        },
    )
    raise SystemExit(return_code if return_code != 0 else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("critic", "setflow"), required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
