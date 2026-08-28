#!/usr/bin/env python3
"""Run the frozen 42-job Critic V4 LOSO package and compose readiness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def write_failure_once(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        return
    write_atomic(path, payload)


def exact_terminal(summary_path: str, failure_path: str) -> str | None:
    summary = Path(summary_path)
    failure = Path(failure_path)
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.wait()


def inspect_worktree_identity(worktree: Path, expected_head: str) -> dict[str, Any] | None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    observed_head = head.stdout.strip() if head.returncode == 0 else None
    dirty = porcelain.stdout.splitlines() if porcelain.returncode == 0 else None
    if (
        head.returncode == 0
        and porcelain.returncode == 0
        and observed_head == expected_head
        and dirty == []
    ):
        return None
    return {
        "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
        "expected_git_head": expected_head,
        "observed_git_head": observed_head,
        "head_return_code": int(head.returncode),
        "porcelain_return_code": int(porcelain.returncode),
        "porcelain_lines": dirty,
    }


def close_missing_job(job: dict[str, Any], *, return_code: int) -> str | None:
    terminal = exact_terminal(job["summary_path"], job["failure_path"])
    if terminal is not None:
        return terminal
    summary = Path(job["summary_path"])
    failure = Path(job["failure_path"])
    if not summary.exists() and not failure.exists():
        write_atomic(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_run_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_stage": "LOSO",
                "run_id": job["run_id"],
                "seed": int(job["seed"]),
                "held_out_study": job["held_out_study"],
                "return_code": int(return_code),
                "failure_stage": "FORMAL_LOSO_SCHEDULER",
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    return exact_terminal(job["summary_path"], job["failure_path"])


def close_job_spawn_failure(
    job: dict[str, Any], *, error: Exception
) -> str | None:
    summary = Path(job["summary_path"])
    failure = Path(job["failure_path"])
    if not summary.exists() and not failure.exists():
        write_failure_once(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_run_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "run_stage": "LOSO",
                "run_id": job["run_id"],
                "seed": int(job["seed"]),
                "held_out_study": job["held_out_study"],
                "return_code": None,
                "failure_stage": "FORMAL_LOSO_JOB_SPAWN",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    return exact_terminal(job["summary_path"], job["failure_path"])


def write_barrier_failure(
    spec: dict[str, Any],
    *,
    barrier: str,
    failure_stage: str,
    error: Exception | None = None,
    identity_failure: dict[str, Any] | None = None,
) -> str | None:
    summary = Path(spec["summary_path"])
    failure = Path(spec["failure_path"])
    if not summary.exists() and not failure.exists():
        if barrier == "LOSO_ADJUDICATION":
            schema = (
                "route_a_v3_route2_xeditcritic_v4_"
                "loso_adjudication_failure.v1"
            )
            protected = {"development_test_outcome_reads_during_loso": 0}
        else:
            schema = "route_a_v3_route2_xeditcritic_v4_readiness_failure.v1"
            protected = {
                "development_test_outcome_reads_after_atomic_test": 0
            }
        payload: dict[str, Any] = {
            "schema_version": schema,
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "return_code": None,
            "failure_stage": failure_stage,
            **protected,
            "new_final_evaluation_outcome_reads": 0,
        }
        if error is not None:
            payload.update(
                {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            )
        if identity_failure is not None:
            payload.update(identity_failure)
        write_failure_once(failure, payload)
    return exact_terminal(str(summary), str(failure))


def run(schedule: dict[str, Any]) -> None:
    runtime_path = Path(schedule["runtime_manifest"])
    worktree = Path(schedule["worktree"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states = {
        job["job_key"]: {
            "seed": int(job["seed"]),
            "held_out_study": job["held_out_study"],
            "run_id": job["run_id"],
            "physical_gpu_index": int(queue["physical_gpu_index"]),
            "status": "PENDING",
            "summary_path": job["summary_path"],
            "failure_path": job["failure_path"],
            "log_path": job["log_path"],
        }
        for queue in schedule["gpu_queues"]
        for job in queue["jobs"]
    }
    adjudication: dict[str, Any] = {
        "status": "PENDING",
        "summary_path": schedule["loso_adjudication"]["summary_path"],
        "failure_path": schedule["loso_adjudication"]["failure_path"],
    }
    readiness: dict[str, Any] = {
        "status": "PENDING",
        "summary_path": schedule["readiness"]["summary_path"],
        "failure_path": schedule["readiness"]["failure_path"],
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_runtime.v1",
                "status": status,
                "scheduler_pid": os.getpid(),
                "git_head": schedule["git_head"],
                "jobs": states,
                "loso_adjudication": adjudication,
                "readiness": readiness,
                "first_terminal_failure": first_terminal_failure or None,
                "active_performance_output_read": False,
                "development_test_access_event_count_before_loso": 1,
                "development_test_outcome_reads_during_loso": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

    def run_queue(queue: dict[str, Any]) -> None:
        jobs = list(queue["jobs"])
        for index, job in enumerate(jobs):
            key = str(job["job_key"])
            with lock:
                if terminal_failure.is_set():
                    for skipped in jobs[index:]:
                        states[str(skipped["job_key"])].update(
                            {
                                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                                "terminal_artifact_kind": None,
                                "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                            }
                        )
                    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
                    return
                identity_failure = inspect_worktree_identity(
                    worktree, str(schedule["git_head"])
                )
                if identity_failure is not None:
                    states[key].update(
                        {
                            "status": "TECHNICAL_FAILURE",
                            "return_code": None,
                            "terminal_artifact_kind": exact_terminal(
                                job["summary_path"], job["failure_path"]
                            ),
                            "finished_unix_seconds": time.time(),
                            **identity_failure,
                        }
                    )
                    if not first_terminal_failure:
                        first_terminal_failure.update(
                            {
                                "job_key": key,
                                "seed": int(job["seed"]),
                                "held_out_study": job["held_out_study"],
                                "run_id": job["run_id"],
                                "summary_path": job["summary_path"],
                                "failure_path": job["failure_path"],
                                "log_path": job["log_path"],
                                **identity_failure,
                            }
                        )
                    terminal_failure.set()
                    for skipped in jobs[index + 1 :]:
                        states[str(skipped["job_key"])].update(
                            {
                                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                                "terminal_artifact_kind": None,
                                "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                            }
                        )
                    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
                    return
                states[key].update(
                    {"status": "RUNNING", "started_unix_seconds": time.time()}
                )
                publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
            try:
                return_code = run_logged(
                    list(job["command"]),
                    cwd=worktree,
                    log=Path(job["log_path"]),
                )
            except Exception as error:
                terminal = close_job_spawn_failure(job, error=error)
                with lock:
                    states[key].update(
                        {
                            "status": "TECHNICAL_FAILURE",
                            "return_code": None,
                            "terminal_artifact_kind": terminal,
                            "failure_stage": "FORMAL_LOSO_JOB_SPAWN",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "finished_unix_seconds": time.time(),
                        }
                    )
                    if not first_terminal_failure:
                        first_terminal_failure.update(
                            {
                                "job_key": key,
                                "seed": int(job["seed"]),
                                "held_out_study": job["held_out_study"],
                                "run_id": job["run_id"],
                                "return_code": None,
                                "terminal_artifact_kind": terminal,
                                "failure_reason": "PROCESS_SPAWN_EXCEPTION",
                                "failure_stage": "FORMAL_LOSO_JOB_SPAWN",
                                "exception_type": type(error).__name__,
                                "exception_message": str(error),
                                "summary_path": job["summary_path"],
                                "failure_path": job["failure_path"],
                                "log_path": job["log_path"],
                            }
                        )
                    terminal_failure.set()
                    for skipped in jobs[index + 1 :]:
                        states[str(skipped["job_key"])].update(
                            {
                                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                                "terminal_artifact_kind": None,
                                "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                            }
                        )
                    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
                return
            terminal = close_missing_job(job, return_code=return_code)
            with lock:
                succeeded = terminal == "SUMMARY" and return_code == 0
                status = "TERMINAL_COMPLETE" if succeeded else "TECHNICAL_FAILURE"
                states[key].update(
                    {
                        "status": status,
                        "return_code": int(return_code),
                        "terminal_artifact_kind": terminal,
                        "finished_unix_seconds": time.time(),
                    }
                )
                if not succeeded:
                    if not first_terminal_failure:
                        first_terminal_failure.update(
                            {
                                "job_key": key,
                                "seed": int(job["seed"]),
                                "held_out_study": job["held_out_study"],
                                "run_id": job["run_id"],
                                "return_code": int(return_code),
                                "terminal_artifact_kind": terminal,
                                "failure_reason": (
                                    "NONZERO_RETURN_CODE_WITH_SUMMARY"
                                    if terminal == "SUMMARY" and return_code != 0
                                    else "NO_UNIQUE_SUCCESS_SUMMARY"
                                ),
                                "summary_path": job["summary_path"],
                                "failure_path": job["failure_path"],
                                "log_path": job["log_path"],
                            }
                        )
                    terminal_failure.set()
                publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

    threads = [
        threading.Thread(
            target=run_queue,
            args=(queue,),
            name=f"gpu-{queue['physical_gpu_index']}",
        )
        for queue in schedule["gpu_queues"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if terminal_failure.is_set():
        for state in states.values():
            if state.get("status") == "PENDING":
                state.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_LOSO_JOB_TECHNICAL_FAILURE",
                    }
                )
        adjudication.update(
            {
                "status": "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
            }
        )
        readiness.update(
            {
                "status": "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )
        publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
        return

    spec = schedule["loso_adjudication"]
    identity_failure = inspect_worktree_identity(
        worktree, str(schedule["git_head"])
    )
    if identity_failure is not None:
        terminal = write_barrier_failure(
            spec,
            barrier="LOSO_ADJUDICATION",
            failure_stage="LOSO_ADJUDICATION_WORKTREE_IDENTITY",
            identity_failure=identity_failure,
        )
        adjudication.update(
            {
                "status": "TECHNICAL_FAILURE",
                "return_code": None,
                "terminal_artifact_kind": terminal,
                "finished_unix_seconds": time.time(),
                **identity_failure,
            }
        )
        readiness.update(
            {
                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )
        first_terminal_failure.update(
            {
                "job_key": "LOSO_ADJUDICATION",
                "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
                "return_code": None,
                "terminal_artifact_kind": terminal,
                "summary_path": spec["summary_path"],
                "failure_path": spec["failure_path"],
                "log_path": spec["log_path"],
                **identity_failure,
            }
        )
        publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
        return
    adjudication["status"] = "RUNNING"
    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
    try:
        return_code = run_logged(
            list(spec["command"]), cwd=worktree, log=Path(spec["log_path"])
        )
    except Exception as error:
        terminal = write_barrier_failure(
            spec,
            barrier="LOSO_ADJUDICATION",
            failure_stage="LOSO_ADJUDICATION_SPAWN",
            error=error,
        )
        adjudication.update(
            {
                "status": "TECHNICAL_FAILURE",
                "return_code": None,
                "terminal_artifact_kind": terminal,
                "failure_stage": "LOSO_ADJUDICATION_SPAWN",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "finished_unix_seconds": time.time(),
            }
        )
        readiness.update(
            {
                "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )
        first_terminal_failure.update(
            {
                "job_key": "LOSO_ADJUDICATION",
                "failure_reason": "PROCESS_SPAWN_EXCEPTION",
                "return_code": None,
                "terminal_artifact_kind": terminal,
                "summary_path": spec["summary_path"],
                "failure_path": spec["failure_path"],
                "log_path": spec["log_path"],
                "failure_stage": "LOSO_ADJUDICATION_SPAWN",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        )
        publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
        return
    summary = Path(spec["summary_path"])
    failure = Path(spec["failure_path"])
    if not summary.exists() and not failure.exists():
        write_atomic(
            failure,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_adjudication_failure.v1",
                "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                "return_code": int(return_code),
                "development_test_outcome_reads_during_loso": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
    terminal = exact_terminal(str(summary), str(failure))
    adjudication_succeeded = terminal == "SUMMARY" and return_code == 0
    adjudication.update(
        {
            "status": "TERMINAL_COMPLETE" if adjudication_succeeded else "TECHNICAL_FAILURE",
            "return_code": int(return_code),
            "terminal_artifact_kind": terminal,
            "finished_unix_seconds": time.time(),
        }
    )
    publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")

    if adjudication_succeeded:
        spec = schedule["readiness"]
        identity_failure = inspect_worktree_identity(
            worktree, str(schedule["git_head"])
        )
        if identity_failure is not None:
            readiness_terminal = write_barrier_failure(
                spec,
                barrier="READINESS",
                failure_stage="READINESS_WORKTREE_IDENTITY",
                identity_failure=identity_failure,
            )
            readiness.update(
                {
                    "status": "TECHNICAL_FAILURE",
                    "return_code": None,
                    "terminal_artifact_kind": readiness_terminal,
                    "guidance_authorized": False,
                    "finished_unix_seconds": time.time(),
                    **identity_failure,
                }
            )
            first_terminal_failure.update(
                {
                    "job_key": "READINESS",
                    "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
                    "return_code": None,
                    "terminal_artifact_kind": readiness_terminal,
                    "summary_path": spec["summary_path"],
                    "failure_path": spec["failure_path"],
                    "log_path": spec["log_path"],
                    **identity_failure,
                }
            )
            publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
            return
        readiness["status"] = "RUNNING"
        publish("XEDITCRITIC_V4_LOSO_SCHEDULER_RUNNING")
        try:
            return_code = run_logged(
                list(spec["command"]),
                cwd=worktree,
                log=Path(spec["log_path"]),
            )
        except Exception as error:
            readiness_terminal = write_barrier_failure(
                spec,
                barrier="READINESS",
                failure_stage="READINESS_SPAWN",
                error=error,
            )
            readiness.update(
                {
                    "status": "TECHNICAL_FAILURE",
                    "return_code": None,
                    "terminal_artifact_kind": readiness_terminal,
                    "guidance_authorized": False,
                    "failure_stage": "READINESS_SPAWN",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "finished_unix_seconds": time.time(),
                }
            )
            first_terminal_failure.update(
                {
                    "job_key": "READINESS",
                    "failure_reason": "PROCESS_SPAWN_EXCEPTION",
                    "return_code": None,
                    "terminal_artifact_kind": readiness_terminal,
                    "summary_path": spec["summary_path"],
                    "failure_path": spec["failure_path"],
                    "log_path": spec["log_path"],
                    "failure_stage": "READINESS_SPAWN",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            )
            publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
            return
        summary = Path(spec["summary_path"])
        failure = Path(spec["failure_path"])
        if not summary.exists() and not failure.exists():
            write_atomic(
                failure,
                {
                    "schema_version": "route_a_v3_route2_xeditcritic_v4_readiness_failure.v1",
                    "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                    "return_code": int(return_code),
                    "development_test_outcome_reads_after_atomic_test": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
            )
        readiness_terminal = exact_terminal(str(summary), str(failure))
        readiness_status = None
        guidance_authorized = False
        readiness_succeeded = readiness_terminal == "SUMMARY" and return_code == 0
        if readiness_succeeded:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            readiness_status = payload.get("status")
            guidance_authorized = payload.get("guidance_authorized") is True
        readiness.update(
            {
                "status": (
                    "TERMINAL_COMPLETE" if readiness_succeeded else "TECHNICAL_FAILURE"
                ),
                "return_code": int(return_code),
                "terminal_artifact_kind": readiness_terminal,
                "readiness_status": readiness_status,
                "guidance_authorized": guidance_authorized,
                "finished_unix_seconds": time.time(),
            }
        )
    else:
        readiness.update(
            {
                "status": "NOT_RUN_LOSO_ADJUDICATION_TECHNICAL_FAILURE",
                "terminal_artifact_kind": None,
                "guidance_authorized": False,
            }
        )

    if not adjudication_succeeded or not readiness.get("status") == "TERMINAL_COMPLETE":
        failure_stage = (
            "LOSO_ADJUDICATION" if not adjudication_succeeded else "READINESS"
        )
        failed_row = adjudication if not adjudication_succeeded else readiness
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "job_key": failure_stage,
                    "failure_reason": (
                        "NONZERO_RETURN_CODE_WITH_SUMMARY"
                        if failed_row.get("terminal_artifact_kind") == "SUMMARY"
                        and int(failed_row.get("return_code", 0)) != 0
                        else "NO_UNIQUE_SUCCESS_SUMMARY"
                    ),
                    "return_code": failed_row.get("return_code"),
                    "terminal_artifact_kind": failed_row.get(
                        "terminal_artifact_kind"
                    ),
                }
            )
        if not adjudication_succeeded:
            readiness.update(
                {
                    "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                    "terminal_artifact_kind": None,
                    "guidance_authorized": False,
                }
            )
        publish("XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE")
        return

    exact_jobs = all(
        row.get("terminal_artifact_kind") in {"SUMMARY", "FAILURE"}
        for row in states.values()
    )
    publish(
        "CRITIC_V4_READY_FOR_GUIDANCE"
        if exact_jobs and readiness.get("guidance_authorized") is True
        else "CRITIC_V4_NOT_READY_FOR_GUIDANCE"
        if exact_jobs and readiness.get("terminal_artifact_kind") == "SUMMARY"
        else "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
