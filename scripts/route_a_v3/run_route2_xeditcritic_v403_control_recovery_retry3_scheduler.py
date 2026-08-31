#!/usr/bin/env python3
"""Run the six retry3 Critic V4 controls after retry2 terminal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping


HISTORICAL_FULL_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
HISTORICAL_C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
PRIOR_FAILED_CONTROL_GIT_HEAD = "a21ae2a47b3275519611ad834660813534b38c41"
RETRY3_LICENSED_HEAD = "a21ae2a47b3275519611ad834660813534b38c41"
CONTROL_RETRY_ORDINAL = 3
CONTROL_RETRY_IDENTITY = "v403_control_recovery_retry3"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
TRAINING_WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
TRAINER = TRAINING_WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
FULL_TERMINAL_AUDIT = (
    TRAINING_WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json"
)
CONTROL_RUN_IDS = (
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
PHYSICAL_GPU_INDICES = (2, 3, 5, 2, 3, 5)
CONTROL_WAVES = (
    CONTROL_RUN_IDS[:3],
    CONTROL_RUN_IDS[3:],
)
PRIOR_FAILED_CONTROL_RUNTIME = (
    Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
    / "experiments/xeditcritic_v4/"
    f"v403_control_recovery_retry2_runner_{PRIOR_FAILED_CONTROL_GIT_HEAD}/runtime.json"
)
PRIOR_CONTROL_OOM_TERMINAL_RECEIPT = (
    Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
    / "audits/xeditcritic_v4/"
    f"v403_control_recovery_retry2_runner_{PRIOR_FAILED_CONTROL_GIT_HEAD}_terminal.json"
)


class XEditCriticV403ControlRecoverySchedulerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403ControlRecoverySchedulerError(message)


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def terminal_observation(output_directory: Path) -> tuple[str | None, str | None]:
    summary_exists = (output_directory / "run_summary.json").exists()
    failure_exists = (output_directory / "failure.json").exists()
    if summary_exists and failure_exists:
        return None, "DOUBLE_TERMINAL_ARTIFACT"
    if summary_exists:
        return "SUMMARY", None
    if failure_exists:
        return "FAILURE", None
    return None, "MISSING_TERMINAL_ARTIFACT"


def terminal_kind(output_directory: Path) -> str | None:
    return terminal_observation(output_directory)[0]


def inspect_worktree_identity(
    worktree: Path, *, expected_head: str
) -> dict[str, Any] | None:
    try:
        observed_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=worktree, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], cwd=worktree, check=True,
            capture_output=True, text=True,
        ).stdout
    except Exception as error:
        return {
            "reason": "WORKTREE_IDENTITY_INSPECTION_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if observed_head != expected_head:
        return {
            "reason": "WORKTREE_HEAD_MISMATCH",
            "expected_git_head": expected_head,
            "observed_git_head": observed_head,
        }
    if porcelain.strip():
        return {
            "reason": "WORKTREE_NOT_CLEAN",
            "expected_git_head": expected_head,
            "observed_git_head": observed_head,
        }
    return None


def publish_scheduler_prelaunch_failure(
    job: Mapping[str, Any], *, inspection: Mapping[str, Any]
) -> str | None:
    output = Path(str(job["output_directory"]))
    if output.exists():
        return terminal_kind(output)
    write_atomic(
        output / "failure.json",
        {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v403_"
                "control_recovery_scheduler_failure.v1"
            ),
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "failure_stage": "CONTROL_RECOVERY_SCHEDULER_PRELAUNCH",
            "run_id": str(job["run_id"]),
            "seed": 20260907,
            "training_git_head": str(job["training_git_head"]),
            "cpu_fallback_used": False,
            "worktree_inspection": dict(inspection),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return terminal_kind(output)


def _command_value(command: list[Any], flag: str) -> str:
    require(flag in command, f"control command lacks {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"control command lacks {flag} value")
    return str(command[index + 1])


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_control_recovery_schedule.v1"
        and schedule.get("status")
        == "XEDITCRITIC_V403_CONTROL_RECOVERY_SCHEDULED",
        "V4.0.3 control recovery schedule identity is invalid",
    )
    current_head = str(schedule.get("current_git_head", ""))
    require(
        re.fullmatch(r"[0-9a-f]{40}", current_head) is not None
        and current_head == RETRY3_LICENSED_HEAD
        and current_head
        not in {
            HISTORICAL_FULL_GIT_HEAD,
            HISTORICAL_C0_GIT_HEAD,
        },
        "control retry runner HEAD must be the pinned Retry2 baseline",
    )
    require(
        schedule.get("historical_full_git_head") == HISTORICAL_FULL_GIT_HEAD
        and schedule.get("historical_c0_git_head") == HISTORICAL_C0_GIT_HEAD
        and schedule.get("runner_git_head") == current_head
        and schedule.get("training_code_git_head") == current_head
        and schedule.get("orchestration_git_head") == current_head
        and Path(str(schedule.get("training_worktree"))) == TRAINING_WORKTREE
        and Path(str(schedule.get("historical_full_terminal_audit")))
        == FULL_TERMINAL_AUDIT,
        "historical provenance or current control runner binding is invalid",
    )
    require(
        schedule.get("retry_ordinal") == CONTROL_RETRY_ORDINAL
        and schedule.get("retry_identity") == CONTROL_RETRY_IDENTITY
        and schedule.get("prior_failed_control_git_head")
        == PRIOR_FAILED_CONTROL_GIT_HEAD
        and Path(str(schedule.get("prior_failed_control_runtime")))
        == PRIOR_FAILED_CONTROL_RUNTIME
        and Path(str(schedule.get("prior_control_oom_terminal_receipt")))
        == PRIOR_CONTROL_OOM_TERMINAL_RECEIPT
        and schedule.get("control_waves")
        == [list(wave) for wave in CONTROL_WAVES]
        and schedule.get("wave1_requires_wave0_all_summaries") is True
        and schedule.get("pytorch_cuda_alloc_conf")
        == PYTORCH_CUDA_ALLOC_CONF,
        "V4.0.3 control retry identity or frozen wave policy is invalid",
    )
    jobs = schedule.get("jobs")
    require(isinstance(jobs, list), "V4.0.3 control jobs are absent")
    require(
        tuple(str(job.get("run_id")) for job in jobs) == CONTROL_RUN_IDS,
        "V4.0.3 recovery is not the exact six-control package",
    )
    require(
        tuple(int(job.get("physical_gpu_index", -1)) for job in jobs)
        == PHYSICAL_GPU_INDICES,
        "V4.0.3 control GPU inventory or assignment changed",
    )
    require(
        len({str(job.get("output_directory")) for job in jobs}) == 6
        and len({str(job.get("training_attempt_id")) for job in jobs}) == 6,
        "V4.0.3 control output directories or attempt ids are not unique",
    )
    require(
        [int(job.get("wave_index", -1)) for job in jobs]
        == [0, 0, 0, 1, 1, 1]
        and all(
            str(job.get("training_attempt_id", "")).endswith(
                f"::{CONTROL_RETRY_IDENTITY}_{current_head}"
            )
            and job.get("process_environment")
            == {"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF}
            for job in jobs
        ),
        "V4.0.3 control retry job wave, attempt, or allocator binding changed",
    )
    config_path = str(schedule.get("screen_config"))
    authorization_path = str(schedule.get("launch_authorization"))
    require(
        all(
            job.get("training_git_head") == current_head
            and list(job.get("command", []))[:2]
            == [str(PYTHON), str(TRAINER)]
            and _command_value(job["command"], "--config") == config_path
            and _command_value(job["command"], "--launch-authorization")
            == authorization_path
            and _command_value(job["command"], "--run-id")
            == str(job["run_id"])
            and _command_value(job["command"], "--physical-gpu-index")
            == str(job["physical_gpu_index"])
            and _command_value(job["command"], "--training-attempt-id")
            == str(job["training_attempt_id"])
            for job in jobs
        ),
        "V4.0.3 control command is not bound to the current licensed trainer",
    )
    require(
        not ({"c0_v4", "v4_full"} & {str(job["run_id"]) for job in jobs}),
        "V4.0.3 control schedule would retrain C0 or full",
    )
    inventory = schedule.get("cuda_bf16_inventory")
    require(
        isinstance(inventory, list)
        and [int(row.get("physical_gpu_index", -1)) for row in inventory]
        == list(PHYSICAL_GPU_INDICES)
        and all(row.get("bf16_supported") is True for row in inventory)
        and all(row.get("bf16_tensor_probe") is True for row in inventory)
        and all(row.get("cpu_fallback_used", False) is False for row in inventory),
        "V4.0.3 CUDA/BF16 inventory is not exact",
    )
    require(
        schedule.get("full_retrained") is False
        and schedule.get("c0_retrained") is False
        and schedule.get("old_v402_stopped_process_resumed") is False
        and schedule.get("prior_family_reused") is False
        and schedule.get("all_six_controls_retrained") is True
        and schedule.get("free_memory_gate_applied") is False
        and int(schedule.get("terminal_artifact_payloads_read_by_scheduler", -1)) == 0
        and int(
            schedule.get("historical_terminal_payloads_read_before_cross_root", -1)
        )
        == 0
        and int(schedule.get("development_test_outcome_reads", -1)) == 0
        and int(schedule.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4.0.3 control schedule violates a recovery isolation boundary",
    )


def run(schedule: Mapping[str, Any]) -> None:
    validate_schedule(schedule)
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    runtime_path = Path(str(schedule["runtime_manifest"]))
    worktree = Path(str(schedule["training_worktree"]))
    current_head = str(schedule["training_code_git_head"])
    lock = threading.Lock()
    terminal_failure = threading.Event()
    first_terminal_failure: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {
        str(job["run_id"]): {
            "run_id": str(job["run_id"]),
            "physical_gpu_index": int(job["physical_gpu_index"]),
            "status": "PENDING",
            "output_directory": str(job["output_directory"]),
            "log_path": str(job["log_path"]),
            "training_attempt_id": str(job["training_attempt_id"]),
            "training_git_head": str(job["training_git_head"]),
            "wave_index": int(job["wave_index"]),
        }
        for job in schedule["jobs"]
    }

    def publish(status: str) -> None:
        write_atomic(
            runtime_path,
            {
                "schema_version": (
                    "route_a_v3_route2_xeditcritic_v403_"
                    "control_recovery_runtime.v1"
                ),
                "status": status,
                "scheduler_pid": os.getpid(),
                "historical_full_git_head": HISTORICAL_FULL_GIT_HEAD,
                "historical_c0_git_head": HISTORICAL_C0_GIT_HEAD,
                "current_git_head": current_head,
                "runner_git_head": current_head,
                "orchestration_git_head": schedule["orchestration_git_head"],
                "training_code_git_head": current_head,
                "training_worktree": str(worktree),
                "retry_ordinal": CONTROL_RETRY_ORDINAL,
                "retry_identity": CONTROL_RETRY_IDENTITY,
                "prior_failed_control_git_head": PRIOR_FAILED_CONTROL_GIT_HEAD,
                "prior_failed_control_runtime": str(
                    PRIOR_FAILED_CONTROL_RUNTIME
                ),
                "prior_control_oom_terminal_receipt": str(
                    PRIOR_CONTROL_OOM_TERMINAL_RECEIPT
                ),
                "ordered_control_run_ids": list(CONTROL_RUN_IDS),
                "control_waves": [list(wave) for wave in CONTROL_WAVES],
                "wave1_requires_wave0_all_summaries": True,
                "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
                "jobs": states,
                "first_terminal_failure": first_terminal_failure or None,
                "cross_root_adjudication_run": False,
                "full_retrained": False,
                "c0_retrained": False,
                "old_v402_stopped_process_resumed": False,
                "prior_family_reused": False,
                "all_six_controls_retrained": True,
                "free_memory_gate_applied": False,
                "terminal_artifact_payloads_read_by_scheduler": 0,
                "historical_terminal_payloads_read_before_cross_root": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )

    def record_failure(
        run_id: str, reason: str, terminal: str | None,
        return_code: int | None = None,
        inspection: Mapping[str, Any] | None = None,
    ) -> None:
        if not first_terminal_failure:
            first_terminal_failure.update(
                {
                    "run_id": run_id,
                    "reason": reason,
                    "return_code": return_code,
                    "terminal_artifact_kind": terminal,
                    "output_directory": states[run_id]["output_directory"],
                    "log_path": states[run_id]["log_path"],
                    "worktree_inspection": dict(inspection) if inspection else None,
                }
            )
        terminal_failure.set()

    def mark_pending_not_run() -> None:
        for state in states.values():
            if state.get("status") == "PENDING":
                state.update(
                    {
                        "status": "NOT_RUN_AFTER_TERMINAL_FAILURE",
                        "terminal_artifact_kind": None,
                        "stop_reason": "EARLIER_CONTROL_JOB_TECHNICAL_FAILURE",
                    }
                )

    jobs_by_run_id = {
        str(job["run_id"]): job for job in schedule["jobs"]
    }
    publish("XEDITCRITIC_V403_CONTROL_RECOVERY_STARTING")

    for wave_index, wave_run_ids in enumerate(CONTROL_WAVES):
        if terminal_failure.is_set():
            with lock:
                mark_pending_not_run()
            break

        wave_processes: dict[str, tuple[subprocess.Popen[str], Any]] = {}
        for run_id in wave_run_ids:
            job = jobs_by_run_id[run_id]
            with lock:
                if terminal_failure.is_set():
                    states[run_id].update(
                        status="NOT_RUN_AFTER_TERMINAL_FAILURE",
                        terminal_artifact_kind=None,
                        stop_reason="EARLIER_CONTROL_JOB_TECHNICAL_FAILURE",
                    )
                    continue
                output = Path(str(job["output_directory"]))
                if output.exists():
                    terminal, issue = terminal_observation(output)
                    states[run_id].update(
                        status="TECHNICAL_FAILURE",
                        terminal_artifact_kind=terminal,
                        finished_unix_seconds=time.time(),
                    )
                    record_failure(
                        run_id,
                        issue or "PREEXISTING_OUTPUT_DIRECTORY",
                        terminal,
                    )
                    continue
                inspection = inspect_worktree_identity(
                    worktree, expected_head=current_head
                )
                if inspection is not None:
                    terminal = publish_scheduler_prelaunch_failure(
                        job, inspection=inspection
                    )
                    states[run_id].update(
                        status="TECHNICAL_FAILURE",
                        terminal_artifact_kind=terminal,
                        finished_unix_seconds=time.time(),
                        worktree_inspection=dict(inspection),
                    )
                    record_failure(
                        run_id,
                        str(inspection["reason"]),
                        terminal,
                        inspection=inspection,
                    )
                    continue
                log = Path(str(job["log_path"]))
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = None
                try:
                    stream = log.open("w", encoding="utf-8")
                    process_environment = os.environ.copy()
                    process_environment.update(job["process_environment"])
                    process = subprocess.Popen(
                        list(job["command"]),
                        cwd=worktree,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                        env=process_environment,
                    )
                except Exception as error:
                    if stream is not None:
                        stream.close()
                    inspection = {
                        "reason": "JOB_PROCESS_LAUNCH_FAILED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    terminal = publish_scheduler_prelaunch_failure(
                        job, inspection=inspection
                    )
                    states[run_id].update(
                        status="TECHNICAL_FAILURE",
                        terminal_artifact_kind=terminal,
                        finished_unix_seconds=time.time(),
                        worktree_inspection=inspection,
                    )
                    record_failure(
                        run_id,
                        "JOB_PROCESS_LAUNCH_FAILED",
                        terminal,
                        inspection=inspection,
                    )
                    continue
                wave_processes[run_id] = (process, stream)
                states[run_id].update(
                    status="RUNNING",
                    training_pid=process.pid,
                    started_unix_seconds=time.time(),
                    launched_wave_index=wave_index,
                )
                publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")

        if terminal_failure.is_set():
            with lock:
                mark_pending_not_run()
        publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")

        # Every process in a started wave is allowed to reach its natural
        # terminal state.  No job in the next wave is spawned until all three
        # current-wave processes have been observed.
        for run_id in wave_run_ids:
            process_stream = wave_processes.get(run_id)
            if process_stream is None:
                continue
            job = jobs_by_run_id[run_id]
            process, stream = process_stream
            try:
                return_code = process.wait()
            finally:
                stream.close()
            kind, issue = terminal_observation(
                Path(str(job["output_directory"]))
            )
            successful = kind == "SUMMARY" and return_code == 0
            with lock:
                states[run_id].update(
                    status=(
                        "TERMINAL_SUMMARY"
                        if successful
                        else "TECHNICAL_FAILURE"
                    ),
                    return_code=return_code,
                    terminal_artifact_kind=kind,
                    finished_unix_seconds=time.time(),
                )
                if not successful:
                    reason = (
                        "JOB_NONZERO_RETURN_CODE"
                        if kind == "SUMMARY" and return_code != 0
                        else "JOB_TERMINAL_FAILURE_ARTIFACT"
                        if kind == "FAILURE"
                        else f"JOB_{issue}"
                    )
                    record_failure(run_id, reason, kind, int(return_code))
                publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")

        with lock:
            if terminal_failure.is_set():
                mark_pending_not_run()
                publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")
                break
            wave_succeeded = all(
                states[run_id].get("status") == "TERMINAL_SUMMARY"
                and states[run_id].get("terminal_artifact_kind") == "SUMMARY"
                and int(states[run_id].get("return_code", -1)) == 0
                for run_id in wave_run_ids
            )
            if not wave_succeeded:
                record_failure(
                    wave_run_ids[0],
                    "WAVE_DID_NOT_REACH_EXACT_ALL_SUMMARIES",
                    states[wave_run_ids[0]].get("terminal_artifact_kind"),
                )
                mark_pending_not_run()
                publish("XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING")
                break

    with lock:
        if terminal_failure.is_set():
            mark_pending_not_run()
            publish("XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE")
            return
        exact_summaries = all(
            states[run_id].get("status") == "TERMINAL_SUMMARY"
            and states[run_id].get("terminal_artifact_kind") == "SUMMARY"
            and int(states[run_id].get("return_code", -1)) == 0
            for run_id in CONTROL_RUN_IDS
        )
        publish(
            "XEDITCRITIC_V403_CONTROL_RECOVERY_ALL_SIX_SUMMARIES_TERMINAL"
            if exact_summaries
            else "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    arguments = parser.parse_args()
    run(json.loads(arguments.schedule.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
