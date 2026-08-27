#!/usr/bin/env python3
"""Run the frozen V4 three-seed matched-compute comparison package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def observe_worktree_identity(worktree: Path) -> dict[str, Any]:
    head_command = ["git", "rev-parse", "HEAD"]
    status_command = ["git", "status", "--porcelain"]
    head = subprocess.run(
        head_command,
        cwd=worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        status_command,
        cwd=worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "head_command": head_command,
        "head_return_code": int(head.returncode),
        "head_stdout": head.stdout,
        "head_stderr": head.stderr,
        "observed_git_head": head.stdout.strip() if head.returncode == 0 else None,
        "status_command": status_command,
        "status_return_code": int(status.returncode),
        "status_stdout": status.stdout,
        "status_stderr": status.stderr,
        "worktree_clean": status.returncode == 0 and not status.stdout.strip(),
    }


def worktree_identity_error(
    observation: Mapping[str, Any], expected_head: str
) -> str | None:
    if int(observation["head_return_code"]) != 0:
        return "GIT_HEAD_QUERY_FAILED"
    if observation.get("observed_git_head") != expected_head:
        return "GIT_HEAD_MISMATCH"
    if int(observation["status_return_code"]) != 0:
        return "GIT_STATUS_QUERY_FAILED"
    if observation.get("worktree_clean") is not True:
        return "WORKTREE_DIRTY"
    return None


def _all_jobs(schedule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jobs: list[Mapping[str, Any]] = []
    for queue in schedule["prerequisite_queues"]:
        jobs.extend(queue["jobs"])
    for chain in schedule["seed_chains"]:
        jobs.extend(chain["jobs"])
    jobs.extend(schedule["finalization_jobs"])
    return jobs


def run(schedule: Mapping[str, Any]) -> None:
    runtime_path = Path(str(schedule["runtime_manifest"]))
    worktree = Path(str(schedule["worktree"]))
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states = {
        str(job["job_key"]): {
            "status": "PENDING",
            "physical_gpu_indices": list(job.get("physical_gpu_indices", ())),
            "success_path": str(job["success_path"]),
            "failure_path": str(job["failure_path"]),
            "log_path": str(job["log_path"]),
        }
        for job in _all_jobs(schedule)
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v4_final_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "experiment_head": schedule["experiment_head"],
                "guidance_runner_head": schedule["guidance_runner_head"],
                "jobs": states,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_reopened": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def close_missing(job: Mapping[str, Any], return_code: int) -> str:
        success = Path(str(job["success_path"]))
        failure = Path(str(job["failure_path"]))
        if success.exists() and not failure.exists():
            return "SUCCESS"
        if failure.exists() and not success.exists():
            return "FAILURE"
        if not success.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": (
                        "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
                    ),
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "job_key": job["job_key"],
                    "return_code": int(return_code),
                    "development_test_outcomes_accessed_after_atomic_test": False,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
            return "FAILURE"
        return "DOUBLE_TERMINAL_FAILURE"

    def stop_pending_after_failure() -> None:
        for row in states.values():
            if row["status"] == "PENDING":
                row.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_FINAL_JOB_TECHNICAL_FAILURE",
                    }
                )

    def register_terminal_failure(
        job: Mapping[str, Any],
        *,
        return_code: int | None,
        terminal_artifact_kind: str,
        failure_stage: str,
        identity_observation: Mapping[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "job_key": str(job["job_key"]),
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal_artifact_kind,
                    "failure_stage": failure_stage,
                    "success_path": str(job["success_path"]),
                    "failure_path": str(job["failure_path"]),
                    "log_path": str(job["log_path"]),
                }
            )
            if identity_observation is not None:
                first_terminal_failure.update(
                    {
                        "expected_git_head": schedule["git_head"],
                        "observed_git_head": identity_observation.get(
                            "observed_git_head"
                        ),
                    }
                )
        terminal_failure.set()
        stop_pending_after_failure()

    def close_worktree_identity_failure(
        job: Mapping[str, Any],
        observation: Mapping[str, Any],
        error: str,
    ) -> None:
        failure = Path(str(job["failure_path"]))
        write_atomic(
            failure,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
                ),
                "status": "TERMINAL_WORKTREE_IDENTITY_FAILURE",
                "failure_stage": "PRE_JOB_WORKTREE_IDENTITY",
                "job_key": job["job_key"],
                "error": error,
                "expected_git_head": schedule["git_head"],
                **dict(observation),
                "job_process_started": False,
                "cpu_fallback_used": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def close_job_exception_failure(
        job: Mapping[str, Any],
        *,
        status: str,
        failure_stage: str,
        error: Exception,
        job_process_started: bool,
        observed_return_code: int | None = None,
        cleanup: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": (
                "route_a_v3_route2_xeditflow_v4_final_job_failure.v1"
            ),
            "status": status,
            "failure_stage": failure_stage,
            "job_key": job["job_key"],
            "exception_type": type(error).__name__,
            "error": str(error),
            "job_process_started": job_process_started,
            "observed_return_code": observed_return_code,
            "cpu_fallback_used": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
        if cleanup is not None:
            payload["cleanup"] = dict(cleanup)
        write_atomic(Path(str(job["failure_path"])), payload)

    def stop_process_after_wait_exception(
        process: subprocess.Popen[str],
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "return_code_before_stop": process.poll(),
            "terminate_attempted": False,
            "kill_attempted": False,
        }
        if details["return_code_before_stop"] is None:
            try:
                process.terminate()
                details["terminate_attempted"] = True
                process.wait(timeout=5)
            except Exception as terminate_error:
                details["terminate_error_type"] = type(terminate_error).__name__
                details["terminate_error"] = str(terminate_error)
                try:
                    process.kill()
                    details["kill_attempted"] = True
                    process.wait(timeout=5)
                except Exception as kill_error:
                    details["kill_error_type"] = type(kill_error).__name__
                    details["kill_error"] = str(kill_error)
        details["return_code_after_stop"] = process.poll()
        return details

    def execute(job: Mapping[str, Any]) -> bool:
        key = str(job["job_key"])
        stream = None
        process: subprocess.Popen[str] | None = None
        with lock:
            if terminal_failure.is_set():
                return False
            observation = observe_worktree_identity(worktree)
            identity_error = worktree_identity_error(
                observation, str(schedule["git_head"])
            )
            if identity_error is not None:
                close_worktree_identity_failure(job, observation, identity_error)
                states[key].update(
                    {
                        "status": "TERMINAL_FAILURE",
                        "return_code": None,
                        "terminal_artifact_kind": "FAILURE",
                        "failure_stage": "PRE_JOB_WORKTREE_IDENTITY",
                        "finished_unix_seconds": time.time(),
                    }
                )
                register_terminal_failure(
                    job,
                    return_code=None,
                    terminal_artifact_kind="FAILURE",
                    failure_stage="PRE_JOB_WORKTREE_IDENTITY",
                    identity_observation=observation,
                )
                publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
                return False
            try:
                log_path = Path(str(job["log_path"]))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    list(job["command"]),
                    cwd=worktree,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as error:
                start_close: dict[str, Any] | None = None
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as close_error:
                        start_close = {
                            "log_close_error_type": type(close_error).__name__,
                            "log_close_error": str(close_error),
                        }
                close_job_exception_failure(
                    job,
                    status="TERMINAL_JOB_PROCESS_START_FAILURE",
                    failure_stage="JOB_PROCESS_START",
                    error=error,
                    job_process_started=False,
                    cleanup=start_close,
                )
                states[key].update(
                    {
                        "status": "TERMINAL_FAILURE",
                        "return_code": None,
                        "terminal_artifact_kind": "FAILURE",
                        "failure_stage": "JOB_PROCESS_START",
                        "finished_unix_seconds": time.time(),
                    }
                )
                register_terminal_failure(
                    job,
                    return_code=None,
                    terminal_artifact_kind="FAILURE",
                    failure_stage="JOB_PROCESS_START",
                )
                publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
                return False
            states[key].update(
                {"status": "RUNNING", "started_unix_seconds": time.time()}
            )
            publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
        assert process is not None and stream is not None
        try:
            return_code = process.wait()
            stream.close()
        except Exception as error:
            with lock:
                terminal_failure.set()
            process_stop = stop_process_after_wait_exception(process)
            try:
                stream.close()
            except Exception as close_error:
                process_stop["log_close_error_type"] = type(close_error).__name__
                process_stop["log_close_error"] = str(close_error)
            observed_return_code = process.poll()
            close_job_exception_failure(
                job,
                status="TERMINAL_JOB_PROCESS_WAIT_FAILURE",
                failure_stage="JOB_PROCESS_WAIT",
                error=error,
                job_process_started=True,
                observed_return_code=observed_return_code,
                cleanup=process_stop,
            )
            with lock:
                states[key].update(
                    {
                        "status": "TERMINAL_FAILURE",
                        "return_code": observed_return_code,
                        "terminal_artifact_kind": "FAILURE",
                        "failure_stage": "JOB_PROCESS_WAIT",
                        "finished_unix_seconds": time.time(),
                    }
                )
                register_terminal_failure(
                    job,
                    return_code=observed_return_code,
                    terminal_artifact_kind="FAILURE",
                    failure_stage="JOB_PROCESS_WAIT",
                )
                publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
            return False
        with lock:
            terminal = close_missing(job, return_code)
            if terminal != "SUCCESS":
                terminal_failure.set()
            states[key].update(
                {
                    "status": (
                        "TERMINAL_COMPLETE"
                        if terminal == "SUCCESS"
                        else "TERMINAL_FAILURE"
                    ),
                    "return_code": int(return_code),
                    "terminal_artifact_kind": terminal,
                    "finished_unix_seconds": time.time(),
                }
            )
            if terminal != "SUCCESS":
                register_terminal_failure(
                    job,
                    return_code=int(return_code),
                    terminal_artifact_kind=terminal,
                    failure_stage="JOB_TERMINAL_ARTIFACT",
                )
            publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
        return terminal == "SUCCESS"

    def run_queue(queue: Mapping[str, Any], results: dict[str, bool]) -> None:
        complete = True
        for job in queue["jobs"]:
            if terminal_failure.is_set():
                complete = False
                break
            complete = execute(job)
            if not complete:
                break
        results[str(queue["queue_key"])] = complete

    publish("XEDITFLOW_V4_FINAL_COMPARISON_RUNNING")
    prerequisite_results: dict[str, bool] = {}
    threads = [
        threading.Thread(target=run_queue, args=(queue, prerequisite_results))
        for queue in schedule["prerequisite_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if terminal_failure.is_set() or not prerequisite_results or not all(
        prerequisite_results.values()
    ):
        stop_pending_after_failure()
        publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
        return

    seed_results: dict[str, bool] = {}
    threads = [
        threading.Thread(target=run_queue, args=(chain, seed_results))
        for chain in schedule["seed_chains"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if (
        terminal_failure.is_set()
        or set(seed_results)
        != {"seed_20260912", "seed_20260913", "seed_20260914"}
        or not all(seed_results.values())
    ):
        stop_pending_after_failure()
        publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
        return

    for job in schedule["finalization_jobs"]:
        if not execute(job):
            stop_pending_after_failure()
            publish("XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE")
            return
    publish("XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
