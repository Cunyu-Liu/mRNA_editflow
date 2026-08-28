#!/usr/bin/env python3
"""Run the single V4 atomic TEST process and preserve one terminal artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


class XEditCriticV4AtomicTestJobLaunchError(RuntimeError):
    pass


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def write_new_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
        raise XEditCriticV4AtomicTestJobLaunchError(
            f"terminal artifact already exists: {path}"
        )
    write_atomic(path, payload)


def inspect_worktree_identity(job: dict[str, Any]) -> None:
    """Fail before TEST access unless the frozen job still owns a clean HEAD."""

    expected_head = str(job["git_head"])
    worktree = Path(job["worktree"])
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST job Git HEAD is invalid"
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise XEditCriticV4AtomicTestJobLaunchError(
            f"atomic TEST worktree identity command failed: {error}"
        ) from error
    if head.returncode != 0:
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST worktree HEAD could not be read: "
            f"{head.stderr.strip()}"
        )
    if status.returncode != 0:
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST worktree cleanliness could not be read: "
            f"{status.stderr.strip()}"
        )
    if head.stdout.strip() != expected_head:
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST worktree is not at the job-fixed exact HEAD"
        )
    if status.stdout.strip():
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST worktree is not clean"
        )


def write_pre_access_terminal_failure(
    job: dict[str, Any],
    *,
    started: float,
    failure_stage: str,
    error: Exception,
) -> None:
    runtime = Path(job["runtime_manifest"])
    output = Path(job["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    failure = output / "failure.json"
    write_new_atomic(
        failure,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v4_atomic_frozen_test_"
                "pre_access_failure.v1"
            ),
            "status": (
                "ATOMIC_FROZEN_TEST_PRE_ACCESS_TECHNICAL_FAILURE_"
                "NO_AUTOMATIC_RETRY"
            ),
            "failure_stage": failure_stage,
            "git_head": str(job["git_head"]),
            "worktree": str(job["worktree"]),
            "runner_command": list(job["command"]),
            "error_type": type(error).__name__,
            "error": str(error),
            "gpu_job_started": False,
            "development_test_access_started": False,
            "development_test_access_event_count": 0,
            "development_test_outcome_reads": 0,
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
            "new_final_evaluation_outcomes_accessed": False,
            "new_final_evaluation_outcome_reads": 0,
            "automatic_retry_attempted": False,
        },
    )
    write_atomic(
        runtime,
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v4_atomic_test_job_runtime.v1"
            ),
            "status": "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TECHNICAL_FAILURE",
            "job_pid": os.getpid(),
            "git_head": str(job["git_head"]),
            "worktree": str(job["worktree"]),
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "output_directory": str(output),
            "return_code": None,
            "terminal_artifact_kind": "FAILURE",
            "failure_stage": failure_stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "gpu_job_started": False,
            "development_test_access_started": False,
            "development_test_access_event_count_observed_by_wrapper": 0,
            "development_test_outcome_reads": 0,
            "started_unix_seconds": started,
            "finished_unix_seconds": time.time(),
            "active_performance_output_read": False,
            "terminal_payload_content_read_by_wrapper": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def exact_terminal(output_directory: Path) -> str | None:
    result = output_directory / "atomic_frozen_test.json"
    failure = output_directory / "failure.json"
    if result.exists() == failure.exists():
        return None
    return "RESULT" if result.exists() else "FAILURE"


def require_fresh_job_family(runtime: Path, output: Path) -> None:
    paths = (
        runtime,
        runtime.with_suffix(runtime.suffix + ".partial"),
        output / "atomic_frozen_test.json",
        output / "atomic_frozen_test.json.partial",
        output / "failure.json",
        output / "failure.json.partial",
    )
    if any(path.exists() for path in paths):
        raise XEditCriticV4AtomicTestJobLaunchError(
            "atomic TEST job family already exists and cannot be reused"
        )


def run(job: dict[str, Any]) -> None:
    runtime = Path(job["runtime_manifest"])
    output = Path(job["output_directory"])
    log = Path(job["log_path"])
    started = time.time()
    require_fresh_job_family(runtime, output)
    try:
        inspect_worktree_identity(job)
    except Exception as error:
        write_pre_access_terminal_failure(
            job,
            started=started,
            failure_stage="JOB_FIXED_WORKTREE_IDENTITY",
            error=error,
        )
        raise
    write_atomic(
        runtime,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_job_runtime.v1",
            "status": "XEDITCRITIC_V4_ATOMIC_TEST_JOB_RUNNING",
            "job_pid": os.getpid(),
            "git_head": job["git_head"],
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "output_directory": str(output),
            "started_unix_seconds": started,
            "active_performance_output_read": False,
            "development_test_access_event_count_observed_by_wrapper": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                list(job["command"]),
                cwd=Path(job["worktree"]),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as error:
            write_pre_access_terminal_failure(
                job,
                started=started,
                failure_stage="FORMAL_TEST_RUNNER_PROCESS_LAUNCH",
                error=error,
            )
            raise
        return_code = process.wait()
    terminal = exact_terminal(output)
    if terminal is None and not (output / "atomic_frozen_test.json").exists():
        output.mkdir(parents=True, exist_ok=True)
        failure = output / "failure.json"
        if not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditcritic_v4_atomic_frozen_test_failure.v1"
                    ),
                    "status": "ATOMIC_FROZEN_TEST_TERMINAL_FAILURE_NO_AUTOMATIC_RETRY",
                    "failure_stage": "FORMAL_JOB_WRAPPER_BEFORE_EXACT_RUNNER_TERMINAL",
                    "return_code": int(return_code),
                    "development_test_access_started": (
                        output / "authorization_consumed.json"
                    ).exists(),
                    "development_test_access_event_count": int(
                        (output / "authorization_consumed.json").exists()
                    ),
                    "general_test_projection_persisted": False,
                    "test_bottom_six_cache_persisted": False,
                    "new_final_evaluation_outcomes_accessed": False,
                },
            )
        terminal = exact_terminal(output)
    write_atomic(
        runtime,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_job_runtime.v1",
            "status": (
                "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TERMINAL"
                if terminal is not None
                else "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TECHNICAL_FAILURE"
            ),
            "job_pid": os.getpid(),
            "git_head": job["git_head"],
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "output_directory": str(output),
            "return_code": int(return_code),
            "terminal_artifact_kind": terminal,
            "started_unix_seconds": started,
            "finished_unix_seconds": time.time(),
            "active_performance_output_read": False,
            "terminal_payload_content_read_by_wrapper": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.job.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
