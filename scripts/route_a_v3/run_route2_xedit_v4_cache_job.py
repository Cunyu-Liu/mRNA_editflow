#!/usr/bin/env python3
"""Run one authorized V4 cache builder and publish exactly one terminal state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


class XEditV4CacheJobError(RuntimeError):
    pass


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def terminal_kind(*, summary_present: bool) -> str:
    """An atomically published builder summary is the unique success terminal."""

    return "SUMMARY" if summary_present else "FAILURE"


def run(arguments: argparse.Namespace) -> None:
    if arguments.summary.exists() or arguments.failure.exists():
        raise XEditV4CacheJobError("cache job already has a terminal artifact")
    arguments.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [
        str(arguments.python),
        str(arguments.builder),
        "--config",
        str(arguments.config),
        "--authorization",
        str(arguments.authorization),
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
                "schema_version": "route_a_v3_route2_xedit_v4_cache_runtime.v1",
                "status": "RUNNING",
                "component": arguments.component,
                "wrapper_pid": os.getpid(),
                "builder_pid": process.pid,
                "git_head": arguments.git_head,
                "started_unix_seconds": started,
                "summary_path": str(arguments.summary),
                "failure_path": str(arguments.failure),
                "log_path": str(arguments.log),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        return_code = process.wait()

    # Both builders publish their complete summary atomically as their final
    # substantive action.  Once that summary exists it is the unique terminal
    # state even if a later stdout/teardown error changes the process return code.
    if terminal_kind(summary_present=arguments.summary.is_file()) == "SUMMARY":
        write_atomic(
            arguments.runtime,
            {
                "schema_version": "route_a_v3_route2_xedit_v4_cache_runtime.v1",
                "status": "TERMINAL_COMPLETE",
                "component": arguments.component,
                "wrapper_pid": os.getpid(),
                "builder_pid": process.pid,
                "git_head": arguments.git_head,
                "started_unix_seconds": started,
                "finished_unix_seconds": time.time(),
                "builder_return_code": return_code,
                "summary_path": str(arguments.summary),
                "failure_path": str(arguments.failure),
                "failure_published": False,
                "log_path": str(arguments.log),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        return

    failure = {
        "schema_version": "route_a_v3_route2_xedit_v4_cache_failure.v1",
        "status": "TECHNICAL_FAILURE",
        "component": arguments.component,
        "wrapper_pid": os.getpid(),
        "builder_pid": process.pid,
        "git_head": arguments.git_head,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "return_code": return_code,
        "summary_present": False,
        "log_path": str(arguments.log),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(arguments.failure, failure)
    write_atomic(
        arguments.runtime,
        {
            **failure,
            "schema_version": "route_a_v3_route2_xedit_v4_cache_runtime.v1",
        },
    )
    raise SystemExit(return_code if return_code != 0 else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("critic", "setflow"), required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    arguments = parser.parse_args()
    run(arguments)


if __name__ == "__main__":
    main()
