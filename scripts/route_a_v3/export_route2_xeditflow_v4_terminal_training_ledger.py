#!/usr/bin/env python3
"""Export the narrow, terminal-only V4 parameter-updating training ledger.

The exporter deliberately performs no discovery.  An explicit inventory names
only a frozen schedule and job key for every training attempt; all other paths
and identities are derived from those schedules and cross-checked against the
per-run ``training_attempt.json`` and terminal training summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


class XEditFlowV4TerminalTrainingLedgerError(RuntimeError):
    pass


EXPECTED_FAMILY_COUNTS = {
    "critic_screen": 8,
    "critic_confirmation": 6,
    "critic_refit": 3,
    "critic_loso": 42,
    "setflow_screen": 2,
    "setflow_confirmation": 3,
    "guidance_value": 6,
    "final_value": 2,
}
EXPECTED_TOTAL_ATTEMPTS = 72

INVENTORY_SCHEMA = (
    "route_a_v3_route2_xeditflow_v4_terminal_training_inventory.v1"
)
INVENTORY_STATUS = "EXPLICIT_COMPLETE_72_PARAMETER_UPDATING_ATTEMPTS"
OUTPUT_SCHEMA = "route_a_v3_route2_xeditflow_v4_terminal_training_ledger.v1"
OUTPUT_STATUS = "XEDITFLOW_V4_TERMINAL_TRAINING_LEDGER_COMPLETE"

SCREEN_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xedit_v4_screen_package_schedule.v1"
)
CRITIC_V402_RECOVERY_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v402_recovery_schedule.v1"
)
CRITIC_V403_FULL_RECOVERY_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_full_recovery_schedule.v1"
)
CRITIC_V403_CONTROL_RECOVERY_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_control_recovery_schedule.v1"
)
CONFIRMATION_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xedit_v4_confirmation_training_schedule.v1"
)
SETFLOW_V403_RECOVERED_CONFIRMATION_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xeditsetflow_v403_"
    "recovered_confirmation_training_schedule.v1"
)
REFIT_SCHEDULE_SCHEMA = "route_a_v3_route2_xeditcritic_v4_refit_schedule.v1"
LOSO_SCHEDULE_SCHEMA = "route_a_v3_route2_xeditcritic_v4_loso_schedule.v1"
GUIDANCE_SCHEDULE_SCHEMA = (
    "route_a_v3_route2_xeditflow_v4_guidance_screen_schedule.v1"
)
FINAL_SCHEDULE_SCHEMA = "route_a_v3_route2_xeditflow_v4_final_schedule.v1"

CRITIC_TRAINER = "train_route2_xeditcritic_v4.py"
SETFLOW_TRAINER = "train_route2_xeditsetflow_v4.py"
VALUE_TRAINER = "train_route2_xeditflow_value_v4.py"

HISTORICAL_CRITIC_V402_C0_HEAD = (
    "93703adec7a4c76b4466d3aaae8684620bee985a"
)
HISTORICAL_CRITIC_V403_FULL_HEAD = (
    "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
)
CRITIC_TERMINAL_EVIDENCE_V1 = "v1"
CRITIC_TERMINAL_EVIDENCE_V2 = "v2"

CRITIC_SCREEN_RUN_IDS = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
CRITIC_CONTROL_RUN_IDS = CRITIC_SCREEN_RUN_IDS[2:]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowV4TerminalTrainingLedgerError(message)


def _counter_is(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise XEditFlowV4TerminalTrainingLedgerError(
            f"{label} is unreadable JSON: {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def _path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} is not a path")
    return Path(value)


def _head(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} is not an exact Git HEAD",
    )
    return value


def _command_value(command: Sequence[Any], flag: str, label: str) -> str:
    values = [index for index, value in enumerate(command) if value == flag]
    _require(len(values) == 1, f"{label} does not contain exactly one {flag}")
    index = values[0]
    _require(index + 1 < len(command), f"{label} has no value after {flag}")
    value = command[index + 1]
    _require(isinstance(value, str) and bool(value), f"{label} {flag} is invalid")
    return value


def _validate_common_schedule_boundary(
    schedule: Mapping[str, Any], *, after_atomic_test: bool
) -> None:
    _require(
        schedule.get("free_memory_gate_applied") is False,
        "training schedule applied a free-memory gate",
    )
    if "active_performance_output_read" in schedule:
        _require(
            schedule.get("active_performance_output_read") is False,
            "training schedule reports an active performance-output read",
        )
    if after_atomic_test:
        if "development_test_outcomes_accessed_after_atomic_test" in schedule:
            _require(
                schedule.get("development_test_outcomes_accessed_after_atomic_test")
                is False,
                "post-atomic schedule reports a Development TEST outcome read",
            )
        if "development_test_outcome_reads_during_refit" in schedule:
            _require(
                _counter_is(
                    schedule.get("development_test_outcome_reads_during_refit"), 0
                ),
                "refit schedule reports a Development TEST outcome read",
            )
        if "development_test_outcome_reads_during_loso" in schedule:
            _require(
                _counter_is(
                    schedule.get("development_test_outcome_reads_during_loso"), 0
                ),
                "LOSO schedule reports a Development TEST outcome read",
            )
    for key in (
        "development_test_outcome_reads",
        "new_final_evaluation_outcome_reads",
    ):
        if key in schedule:
            _require(
                _counter_is(schedule.get(key), 0),
                f"training schedule protected-read counter differs: {key}",
            )


def _queue_jobs(
    schedule: Mapping[str, Any], *, queues_key: str = "gpu_queues"
) -> list[tuple[Mapping[str, Any], int]]:
    queues = schedule.get(queues_key)
    _require(isinstance(queues, list), f"schedule {queues_key} is not a list")
    rows: list[tuple[Mapping[str, Any], int]] = []
    for queue in queues:
        _require(isinstance(queue, Mapping), "schedule queue is not an object")
        gpu = queue.get("physical_gpu_index")
        _require(type(gpu) is int and 0 <= gpu <= 5, "schedule GPU is outside 0-5")
        jobs = queue.get("jobs")
        _require(isinstance(jobs, list), "schedule queue jobs are not a list")
        for job in jobs:
            _require(isinstance(job, Mapping), "schedule job is not an object")
            rows.append((job, gpu))
    return rows


def _final_jobs(schedule: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    prerequisite = schedule.get("prerequisite_queues")
    seed_chains = schedule.get("seed_chains")
    finalization = schedule.get("finalization_jobs")
    _require(
        isinstance(prerequisite, list)
        and isinstance(seed_chains, list)
        and isinstance(finalization, list),
        "final schedule job containers differ",
    )
    _require(
        len(prerequisite) == 3
        and [len(queue.get("jobs", ())) for queue in prerequisite] == [4, 4, 1],
        "final prerequisite queue geometry differs",
    )
    _require(
        len(seed_chains) == 3
        and all(len(chain.get("jobs", ())) == 29 for chain in seed_chains),
        "final seed-chain geometry differs",
    )
    _require(len(finalization) == 2, "finalization job count differs")
    jobs: list[Mapping[str, Any]] = []
    for container in (*prerequisite, *seed_chains):
        _require(isinstance(container, Mapping), "final job container is invalid")
        for job in container["jobs"]:
            _require(isinstance(job, Mapping), "final schedule job is invalid")
            jobs.append(job)
    for job in finalization:
        _require(isinstance(job, Mapping), "finalization job is invalid")
        jobs.append(job)
    keys = [job.get("job_key") for job in jobs]
    _require(
        len(jobs) == 98
        and all(isinstance(key, str) and key for key in keys)
        and len(set(keys)) == 98,
        "final schedule does not contain 98 unique jobs",
    )
    return jobs


def _validate_schedule_and_training_jobs(
    schedule_path: Path,
    *,
    final_schedule_path: Path,
    preloaded_schedule: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    schedule = (
        dict(preloaded_schedule)
        if preloaded_schedule is not None
        else _read_json(schedule_path, "frozen training schedule")
    )
    schema = schedule.get("schema_version")
    families: dict[str, list[dict[str, Any]]] = {}

    def normalized(
        job: Mapping[str, Any],
        gpu: int,
        *,
        family: str,
        trainer: str,
        summary_path: Path,
        output_directory: Path,
        failure_path: Path,
        training_git_head: str,
        job_key: str | None = None,
        historical_free_memory_gate_applied: bool = False,
        historical_free_memory_gate_policy: str | None = None,
        critic_terminal_evidence_version: str | None = None,
    ) -> dict[str, Any]:
        key = job_key if job_key is not None else job.get("job_key")
        command = job.get("command")
        _require(isinstance(key, str) and bool(key), "schedule job_key is invalid")
        _require(
            isinstance(command, list)
            and len(command) >= 4
            and isinstance(command[1], str)
            and Path(command[1]).name == trainer,
            f"training command differs for {key}",
        )
        config_path = Path(_command_value(command, "--config", f"job {key}"))
        log_path = _path(job.get("log_path"), f"job {key} log_path")
        if "--run-id" in command and "run_id" in job:
            _require(
                _command_value(command, "--run-id", f"job {key}")
                == str(job["run_id"]),
                f"training command run_id differs for {key}",
            )
        if "--physical-gpu-index" in command:
            _require(
                _command_value(command, "--physical-gpu-index", f"job {key}")
                == str(gpu),
                f"training command GPU differs for {key}",
            )
        if "--output-dir" in command:
            _require(
                _command_value(command, "--output-dir", f"job {key}")
                == str(output_directory),
                f"training command output directory differs for {key}",
            )
        attempt_flag_count = command.count("--training-attempt-id")
        _require(
            attempt_flag_count in {0, 1},
            f"training command attempt identity differs for {key}",
        )
        expected_attempt_id = (
            _command_value(command, "--training-attempt-id", f"job {key}")
            if attempt_flag_count == 1
            else None
        )
        if "training_attempt_id" in job:
            _require(
                isinstance(job.get("training_attempt_id"), str)
                and bool(job.get("training_attempt_id")),
                f"schedule training_attempt_id differs for {key}",
            )
            _require(
                expected_attempt_id == job.get("training_attempt_id"),
                f"schedule and command training_attempt_id differ for {key}",
            )
        if family.startswith("critic_"):
            _require(
                critic_terminal_evidence_version
                in {
                    CRITIC_TERMINAL_EVIDENCE_V1,
                    CRITIC_TERMINAL_EVIDENCE_V2,
                },
                f"Critic terminal-evidence version is absent for {key}",
            )
            run_id = job.get("run_id")
            _require(
                isinstance(run_id, str) and bool(run_id),
                f"Critic schedule run_id is invalid for {key}",
            )
        else:
            _require(
                critic_terminal_evidence_version is None,
                f"non-Critic job declares Critic terminal evidence for {key}",
            )
            run_id = job.get("run_id")
        return {
            "family": family,
            "job_key": key,
            "schedule_path": schedule_path,
            "schedule_schema_version": schema,
            "training_git_head": training_git_head,
            "command": list(command),
            "config_path": config_path,
            "physical_gpu_index": gpu,
            "output_directory": output_directory,
            "terminal_summary_path": summary_path,
            "failure_path": failure_path,
            "log_path": log_path,
            "expected_attempt_id": expected_attempt_id,
            "run_id": run_id,
            "critic_terminal_evidence_version": (
                critic_terminal_evidence_version
            ),
            "historical_free_memory_gate_applied": (
                historical_free_memory_gate_applied
            ),
            "historical_free_memory_gate_policy": (
                historical_free_memory_gate_policy
            ),
        }

    if schema == SCREEN_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "screen schedule git_head")
        _require(
            schedule.get("status") == "FROZEN_SCREEN_PACKAGE_SCHEDULE",
            "screen schedule is not frozen",
        )
        _require(
            "free_memory_gate_applied" not in schedule
            and schedule.get("gpu_assignment_policy")
            == "ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB",
            "historical screen free-memory gate identity differs",
        )
        _require(
            schedule.get("active_performance_output_read") is False
            and _counter_is(schedule.get("development_test_outcome_reads"), 0)
            and _counter_is(schedule.get("new_final_evaluation_outcome_reads"), 0),
            "historical screen schedule violates a protected-read boundary",
        )
        free_memory = schedule.get("gpu_free_memory_mib_before_launch")
        critic_required = schedule.get("critic_required_free_memory_mib")
        setflow_required = schedule.get("setflow_required_free_memory_mib")
        _require(
            isinstance(free_memory, Mapping)
            and set(free_memory) == {str(gpu) for gpu in range(6)}
            and all(
                type(free_memory[str(gpu)]) is int
                and free_memory[str(gpu)] > 0
                for gpu in range(6)
            )
            and type(critic_required) is int
            and critic_required > 0
            and type(setflow_required) is int
            and setflow_required > 0,
            "historical screen measured-memory gate fields differ",
        )
        rows = _queue_jobs(schedule)
        by_component = Counter(job.get("component") for job, _ in rows)
        _require(
            by_component == {"critic": 8, "setflow": 2},
            "screen schedule training inventory differs",
        )
        all_keys = [job.get("job_key") for job, _ in rows]
        _require(
            set(all_keys)
            == {
                *(f"critic:{run_id}" for run_id in CRITIC_SCREEN_RUN_IDS),
                "setflow:v4_full",
                "setflow:v4_single_mode",
            },
            "screen schedule job identities differ",
        )
        for job, gpu in rows:
            component = str(job["component"])
            required = critic_required if component == "critic" else setflow_required
            _require(
                free_memory[str(gpu)] >= required,
                f"historical screen GPU assignment violates measured gate for {job.get('job_key')}",
            )
            if component != "setflow":
                continue
            output = _path(job.get("output_directory"), "screen output_directory")
            families.setdefault("setflow_screen", []).append(
                normalized(
                    job,
                    gpu,
                    family="setflow_screen",
                    trainer=SETFLOW_TRAINER,
                    summary_path=output / "training_summary.json",
                    output_directory=output,
                    failure_path=output / "failure.json",
                    training_git_head=head,
                    historical_free_memory_gate_applied=True,
                    historical_free_memory_gate_policy=(
                        "ANY_PHYSICAL_GPU_0_TO_5_MEETING_MEASURED_PEAK_PLUS_2_GIB"
                    ),
                )
            )
    elif schema == CRITIC_V402_RECOVERY_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "V4.0.2 recovery git_head")
        _require(
            head == HISTORICAL_CRITIC_V402_C0_HEAD
            and schedule.get("status") == "FROZEN_V402_RECOVERY_SCHEDULE",
            "V4.0.2 recovery schedule is not frozen",
        )
        _require(
            schedule.get("active_performance_output_read") is False
            and schedule.get("setflow_jobs_stopped_modified_or_restarted") is False
            and _counter_is(schedule.get("terminal_artifact_payloads_read"), 0)
            and _counter_is(schedule.get("development_test_outcome_reads"), 0)
            and _counter_is(schedule.get("new_final_evaluation_outcome_reads"), 0),
            "V4.0.2 recovery schedule violates an isolation boundary",
        )
        free_mib = schedule.get("gpu5_free_memory_mib_before_launch")
        required_bytes = schedule.get("required_free_memory_bytes")
        _require(
            type(free_mib) is int
            and free_mib > 0
            and type(required_bytes) is int
            and required_bytes > 0
            and free_mib * 1024**2 >= required_bytes
            and schedule.get("required_free_memory_rule")
            == "TRAIN_ONLY_SMOKE_PEAK_PLUS_2_GIB",
            "V4.0.2 historical free-memory gate provenance differs",
        )
        _require(
            schedule.get("physical_gpu_index") == 5,
            "V4.0.2 recovery is not bound to physical GPU5",
        )
        jobs = schedule.get("jobs")
        _require(
            isinstance(jobs, list)
            and all(isinstance(job, Mapping) for job in jobs)
            and tuple(str(job.get("run_id")) for job in jobs)
            == CRITIC_SCREEN_RUN_IDS
            and tuple(str(job.get("job_key")) for job in jobs)
            == tuple(f"critic:{run_id}" for run_id in CRITIC_SCREEN_RUN_IDS)
            and len({str(job.get("output_directory")) for job in jobs}) == 8,
            "V4.0.2 recovery is not the exact ordered eight-arm package",
        )
        c0 = jobs[0]
        _require(
            c0.get("run_id") == "c0_v4"
            and c0.get("job_key") == "critic:c0_v4",
            "historical V4.0.2 C0 evidence identity differs",
        )
        output = _path(c0.get("output_directory"), "V4.0.2 C0 output_directory")
        families["critic_screen"] = [
            normalized(
                c0,
                5,
                family="critic_screen",
                trainer=CRITIC_TRAINER,
                summary_path=output / "run_summary.json",
                output_directory=output,
                failure_path=output / "failure.json",
                training_git_head=head,
                historical_free_memory_gate_applied=True,
                historical_free_memory_gate_policy=(
                    "TRAIN_ONLY_SMOKE_PEAK_PLUS_2_GIB"
                ),
                critic_terminal_evidence_version=(
                    CRITIC_TERMINAL_EVIDENCE_V1
                ),
            )
        ]
    elif schema == CRITIC_V403_FULL_RECOVERY_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "V4.0.3 full recovery git_head")
        _require(
            head == HISTORICAL_CRITIC_V403_FULL_HEAD
            and schedule.get("status")
            == "XEDITCRITIC_V403_FULL_RECOVERY_SCHEDULED"
            and schedule.get("run_id") == "v4_full",
            "V4.0.3 full recovery schedule identity differs",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=False)
        gpu = schedule.get("physical_gpu_index")
        _require(
            type(gpu) is int and 0 <= gpu <= 5,
            "V4.0.3 full recovery GPU differs",
        )
        job = dict(schedule)
        job["job_key"] = "critic:v4_full"
        output = _path(
            schedule.get("output_directory"),
            "V4.0.3 full recovery output_directory",
        )
        _require(
            _path(schedule.get("screen_config"), "V4.0.3 full screen_config")
            == Path(_command_value(schedule["command"], "--config", "V4.0.3 full")),
            "V4.0.3 full recovery config binding differs",
        )
        families["critic_screen"] = [
            normalized(
                job,
                gpu,
                family="critic_screen",
                trainer=CRITIC_TRAINER,
                summary_path=output / "run_summary.json",
                output_directory=output,
                failure_path=output / "failure.json",
                training_git_head=head,
                critic_terminal_evidence_version=(
                    CRITIC_TERMINAL_EVIDENCE_V1
                ),
            )
        ]
    elif schema == CRITIC_V403_CONTROL_RECOVERY_SCHEDULE_SCHEMA:
        training_head = _head(
            schedule.get("training_code_git_head"),
            "V4.0.3 control training_code_git_head",
        )
        _head(
            schedule.get("orchestration_git_head"),
            "V4.0.3 control orchestration_git_head",
        )
        _require(
            schedule.get("status")
            == "XEDITCRITIC_V403_CONTROL_RECOVERY_SCHEDULED",
            "V4.0.3 control recovery schedule identity differs",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=False)
        _require(
            schedule.get("full_retrained") is False
            and schedule.get("c0_retrained") is False
            and schedule.get("old_v402_stopped_process_resumed") is False
            and _counter_is(
                schedule.get("terminal_artifact_payloads_read_by_scheduler"), 0
            ),
            "V4.0.3 control recovery isolation boundary differs",
        )
        jobs = schedule.get("jobs")
        _require(
            isinstance(jobs, list)
            and all(isinstance(job, Mapping) for job in jobs)
            and tuple(str(job.get("run_id")) for job in jobs)
            == CRITIC_CONTROL_RUN_IDS
            and tuple(job.get("physical_gpu_index") for job in jobs)
            == tuple(range(6))
            and len({str(job.get("output_directory")) for job in jobs}) == 6
            and len({str(job.get("training_attempt_id")) for job in jobs}) == 6,
            "V4.0.3 recovery is not the exact six-control package",
        )
        inventory = schedule.get("cuda_bf16_inventory")
        _require(
            isinstance(inventory, list)
            and [row.get("physical_gpu_index") for row in inventory]
            == list(range(6))
            and all(row.get("bf16_supported") is True for row in inventory)
            and all(row.get("bf16_tensor_probe") is True for row in inventory)
            and all(row.get("cpu_fallback_used", False) is False for row in inventory),
            "V4.0.3 control CUDA/BF16 inventory differs",
        )
        config_path = _path(
            schedule.get("screen_config"), "V4.0.3 control screen_config"
        )
        for job in jobs:
            run_id = str(job["run_id"])
            gpu = int(job["physical_gpu_index"])
            _require(
                Path(_command_value(job["command"], "--config", run_id))
                == config_path,
                f"V4.0.3 control config binding differs for {run_id}",
            )
            output = _path(
                job.get("output_directory"),
                f"V4.0.3 control output_directory for {run_id}",
            )
            families.setdefault("critic_screen", []).append(
                normalized(
                    job,
                    gpu,
                    family="critic_screen",
                    trainer=CRITIC_TRAINER,
                    summary_path=output / "run_summary.json",
                    output_directory=output,
                    failure_path=output / "failure.json",
                    training_git_head=training_head,
                    job_key=f"critic:{run_id}",
                    critic_terminal_evidence_version=(
                        CRITIC_TERMINAL_EVIDENCE_V2
                    ),
                )
            )
    elif schema == CONFIRMATION_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "confirmation schedule git_head")
        _require(
            schedule.get("status") == "FROZEN_CONFIRMATION_TRAINING_SCHEDULE",
            "confirmation schedule is not frozen",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=False)
        rows = _queue_jobs(schedule)
        by_component = Counter(job.get("component") for job, _ in rows)
        _require(
            schedule.get("eligible_components") == ["critic"]
            and by_component == {"critic": 6},
            "Critic-only confirmation schedule training inventory differs",
        )
        for job, gpu in rows:
            output = _path(
                job.get("output_directory"), "confirmation output_directory"
            )
            families.setdefault("critic_confirmation", []).append(
                normalized(
                    job,
                    gpu,
                    family="critic_confirmation",
                    trainer=CRITIC_TRAINER,
                    summary_path=output / "run_summary.json",
                    output_directory=output,
                    failure_path=output / "failure.json",
                    training_git_head=head,
                    critic_terminal_evidence_version=(
                        CRITIC_TERMINAL_EVIDENCE_V2
                    ),
                )
            )
    elif schema == SETFLOW_V403_RECOVERED_CONFIRMATION_SCHEDULE_SCHEMA:
        head = _head(
            schedule.get("git_head"),
            "recovered SetFlow confirmation runner git_head",
        )
        _head(
            schedule.get("training_git_head"),
            "recovered SetFlow confirmation training_git_head",
        )
        _head(
            schedule.get("validation_git_head"),
            "recovered SetFlow confirmation validation_git_head",
        )
        _head(
            schedule.get("experiment_head"),
            "recovered SetFlow confirmation experiment_head",
        )
        _require(
            schedule.get("status")
            == "FROZEN_RECOVERY_DERIVED_CONFIRMATION_TRAINING_SCHEDULE"
            and schedule.get("eligible_components") == ["setflow"]
            and schedule.get("training_reused_from_screen") is False
            and schedule.get("screen_training_reused_by_recovery") is True
            and _counter_is(schedule.get("recovery_parameter_update_count"), 0),
            "recovered SetFlow confirmation schedule identity differs",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=False)
        rows = _queue_jobs(schedule)
        _require(
            len(rows) == 3
            and all(job.get("component") == "setflow" for job, _ in rows)
            and len({job.get("training_seed") for job, _ in rows}) == 3,
            "recovered SetFlow confirmation job inventory differs",
        )
        probes = schedule.get("cuda_bf16_probes")
        _require(
            isinstance(probes, Mapping),
            "recovered SetFlow confirmation CUDA/BF16 probes are absent",
        )
        for job, gpu in rows:
            probe = probes.get(str(gpu))
            _require(
                isinstance(probe, Mapping)
                and probe.get("physical_gpu_index") == gpu
                and probe.get("device_type") == "cuda"
                and probe.get("cuda_available") is True
                and probe.get("bf16_supported") is True
                and probe.get("cpu_fallback_used") is False,
                f"recovered SetFlow confirmation GPU probe differs for GPU {gpu}",
            )
            output = _path(
                job.get("output_directory"),
                "recovered SetFlow confirmation output_directory",
            )
            families.setdefault("setflow_confirmation", []).append(
                normalized(
                    job,
                    gpu,
                    family="setflow_confirmation",
                    trainer=SETFLOW_TRAINER,
                    summary_path=output / "training_summary.json",
                    output_directory=output,
                    failure_path=output / "failure.json",
                    # The recovered trainer executes from the runner worktree;
                    # training_attempt.code_commit therefore records git_head.
                    training_git_head=head,
                )
            )
    elif schema in {REFIT_SCHEDULE_SCHEMA, LOSO_SCHEDULE_SCHEMA}:
        head = _head(schedule.get("git_head"), "post-test schedule git_head")
        is_refit = schema == REFIT_SCHEDULE_SCHEMA
        family = "critic_refit" if is_refit else "critic_loso"
        expected_status = (
            "FROZEN_EXACT_THREE_REFIT_SCHEDULE"
            if is_refit
            else "FROZEN_EXACT_42_JOB_LOSO_SCHEDULE"
        )
        _require(schedule.get("status") == expected_status, f"{family} schedule is not frozen")
        _validate_common_schedule_boundary(schedule, after_atomic_test=True)
        access_key = (
            "development_test_access_event_count_before_refit"
            if is_refit
            else "development_test_access_event_count_before_loso"
        )
        _require(
            _counter_is(schedule.get(access_key), 1),
            f"{family} schedule lost the single Atomic TEST access event",
        )
        rows = _queue_jobs(schedule)
        _require(
            len(rows) == EXPECTED_FAMILY_COUNTS[family],
            f"{family} job count differs",
        )
        for job, gpu in rows:
            output = _path(job.get("summary_path"), f"{family} summary_path").parent
            families.setdefault(family, []).append(
                normalized(
                    job,
                    gpu,
                    family=family,
                    trainer=CRITIC_TRAINER,
                    summary_path=_path(
                        job.get("summary_path"), f"{family} summary_path"
                    ),
                    output_directory=output,
                    failure_path=_path(
                        job.get("failure_path"), f"{family} failure_path"
                    ),
                    training_git_head=head,
                    critic_terminal_evidence_version=(
                        CRITIC_TERMINAL_EVIDENCE_V2
                    ),
                )
            )
    elif schema == GUIDANCE_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "guidance schedule git_head")
        _require(
            schedule.get("status")
            == "FROZEN_VALUE_AND_EXACT_18_COMBINATION_SCHEDULE",
            "guidance schedule is not frozen",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=True)
        rows = _queue_jobs(schedule, queues_key="value_training_queues")
        _require(len(rows) == 6, "guidance value-training job count differs")
        for job, gpu in rows:
            summary = _path(job.get("success_path"), "guidance value success_path")
            output = summary.parent
            families.setdefault("guidance_value", []).append(
                normalized(
                    job,
                    gpu,
                    family="guidance_value",
                    trainer=VALUE_TRAINER,
                    summary_path=summary,
                    output_directory=output,
                    failure_path=_path(
                        job.get("failure_path"), "guidance value failure_path"
                    ),
                    training_git_head=head,
                )
            )
    elif schema == FINAL_SCHEDULE_SCHEMA:
        head = _head(schedule.get("git_head"), "final schedule git_head")
        _require(
            schedule_path == final_schedule_path,
            "final_value inventory does not use the authoritative final schedule",
        )
        _require(
            schedule.get("status") == "FROZEN_THREE_SEED_MATCHED_COMPUTE_SCHEDULE",
            "final schedule is not frozen",
        )
        _validate_common_schedule_boundary(schedule, after_atomic_test=True)
        jobs = _final_jobs(schedule)
        training_jobs = [
            job for job in jobs if str(job.get("job_key", "")).endswith(":value_training")
        ]
        _require(
            [job.get("job_key") for job in training_jobs]
            == ["seed_20260913:value_training", "seed_20260914:value_training"],
            "final value-training job identity differs",
        )
        for job in training_jobs:
            indices = job.get("physical_gpu_indices")
            _require(
                isinstance(indices, list)
                and len(indices) == 1
                and type(indices[0]) is int
                and 0 <= indices[0] <= 5,
                "final value-training GPU identity differs",
            )
            summary = _path(job.get("success_path"), "final value success_path")
            families.setdefault("final_value", []).append(
                normalized(
                    job,
                    indices[0],
                    family="final_value",
                    trainer=VALUE_TRAINER,
                    summary_path=summary,
                    output_directory=summary.parent,
                    failure_path=_path(
                        job.get("failure_path"), "final value failure_path"
                    ),
                    training_git_head=head,
                )
            )
    else:
        raise XEditFlowV4TerminalTrainingLedgerError(
            f"inventory refers to a non-allowlisted training schedule: {schedule_path}"
        )

    for family, jobs in families.items():
        keys = [str(job["job_key"]) for job in jobs]
        _require(
            len(keys) == len(set(keys)), f"{family} schedule contains duplicate job keys"
        )
    return schedule, families


def _attempt_seed(
    family: str,
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> int:
    if family in {"critic_screen", "setflow_screen"}:
        training = config.get("training")
        _require(
            isinstance(training, Mapping),
            f"screen training configuration is absent for {family}",
        )
        config_seed = training.get("screen_seed")
    elif family in {"guidance_value", "final_value"}:
        config_seed = config.get("base_flow_training_seed")
    else:
        config_seed = config.get("training_seed")
    summary_key = (
        "base_flow_training_seed" if family in {"guidance_value", "final_value"} else "seed"
    )
    summary_seed = summary.get(summary_key)
    _require(
        type(config_seed) is int and summary_seed == config_seed,
        f"training seed differs across config and summary for {family}",
    )
    attempt_seed = attempt.get("seed")
    _require(
        attempt_seed in {None, "", config_seed},
        f"training seed differs in training_attempt.json for {family}",
    )
    return config_seed


def _critic_v1_legacy_identity(job: Mapping[str, Any]) -> bool:
    schedule_schema = job.get("schedule_schema_version")
    job_key = job.get("job_key")
    head = job.get("training_git_head")
    return (
        schedule_schema == CRITIC_V402_RECOVERY_SCHEDULE_SCHEMA
        and job_key == "critic:c0_v4"
        and head == HISTORICAL_CRITIC_V402_C0_HEAD
    ) or (
        schedule_schema == CRITIC_V403_FULL_RECOVERY_SCHEDULE_SCHEMA
        and job_key == "critic:v4_full"
        and head == HISTORICAL_CRITIC_V403_FULL_HEAD
    )


def _critic_initialization_scope(run_id: str) -> str:
    if run_id == "c0_v4":
        return "NOT_CLAIMED_DIFFERENT_C0_ARCHITECTURE"
    if run_id == "v4_no_cross":
        return "NOT_CLAIMED_PARAMETER_MATCHED_DIFFERENT_MODULE"
    return "SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE"


def _validate_critic_v2_terminal_evidence(
    job: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    summary: Mapping[str, Any],
    seed: int,
    output_directory: Path,
    summary_path: Path,
    checkpoint_path: Path,
    attempt_path: Path,
) -> None:
    family = str(job["family"])
    key = str(job["job_key"])
    stage = family.removeprefix("critic_").upper()
    run_id = str(job["run_id"])
    gpu = int(job["physical_gpu_index"])
    device = f"cuda:{gpu}"
    head = str(job["training_git_head"])
    expected_scope = _critic_initialization_scope(run_id)
    runner_head_key = {
        "SCREEN": "runner_git_head",
        "CONFIRMATION": "confirmation_runner_git_head",
        "REFIT": "posttest_runner_git_head",
        "LOSO": "posttest_runner_git_head",
    }[stage]
    data_geometry = config.get("data_geometry")
    update_budget = (
        data_geometry.get("total_optimizer_updates")
        if isinstance(data_geometry, Mapping)
        else None
    )

    _require(
        config.get(runner_head_key) == head,
        f"Critic v2 config runner HEAD differs from schedule for {key}",
    )
    _require(
        isinstance(update_budget, int)
        and not isinstance(update_budget, bool)
        and update_budget > 0
        and summary.get("update_count") == update_budget
        and attempt.get("optimizer_steps") == update_budget,
        f"Critic v2 parameter-update budget differs for {key}",
    )
    _require(
        summary.get("run_id") == run_id
        and attempt.get("baseline_id")
        == f"xeditcritic_v4_{run_id}_seed{seed}",
        f"Critic v2 run identity differs for {key}",
    )
    for payload, label in ((summary, "summary"), (attempt, "training attempt")):
        _require(
            payload.get("parameter_initialization_seed") == seed
            and payload.get(
                "parameter_initialization_seed_applied_before_model_construction"
            )
            is True
            and payload.get("parameter_initialization_tensor_identity_scope")
            == expected_scope,
            f"Critic v2 initialization provenance differs in {label} for {key}",
        )
        payload_device = (
            payload.get("cuda_device")
            if label == "summary"
            else payload.get("device")
        )
        _require(
            payload.get("cuda_available") is True
            and payload_device == device
            and isinstance(payload.get("cuda_device_name"), str)
            and "A100" in str(payload.get("cuda_device_name"))
            and payload.get("a100_device_verified") is True
            and payload.get("bf16_supported") is True
            and payload.get("cpu_fallback_used") is False
            and payload.get("training_git_head") == head,
            f"Critic v2 CUDA/BF16/HEAD provenance differs in {label} for {key}",
        )

    expected_paths = {
        "output_directory": str(output_directory),
        "training_summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "training_attempt_path": str(attempt_path),
    }
    for payload, label in ((summary, "summary"), (attempt, "training attempt")):
        _require(
            all(payload.get(name) == value for name, value in expected_paths.items()),
            f"Critic v2 terminal paths differ in {label} for {key}",
        )
    _require(
        attempt.get("seed") == seed,
        f"Critic v2 training-attempt seed differs for {key}",
    )


def _validate_summary(
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    output_directory: Path,
) -> tuple[Path, str]:
    family = str(job["family"])
    if family.startswith("critic_"):
        stage = family.removeprefix("critic_").upper()
        evidence_version = job.get("critic_terminal_evidence_version")
        if evidence_version == CRITIC_TERMINAL_EVIDENCE_V1:
            _require(
                _critic_v1_legacy_identity(job),
                f"{family} v1 evidence is outside the exact historical jobs",
            )
        else:
            _require(
                evidence_version == CRITIC_TERMINAL_EVIDENCE_V2
                and not _critic_v1_legacy_identity(job),
                f"{family} terminal-evidence policy differs",
            )
        expected_schema = (
            f"route_a_v3_route2_xeditcritic_v4_{stage.lower()}_run."
            f"{evidence_version}"
        )
        _require(
            summary.get("schema_version") == expected_schema
            and summary.get("status")
            == f"TERMINAL_XEDITCRITIC_V4_{stage}_RUN_COMPLETE"
            and summary.get("run_stage") == stage,
            f"{family} terminal summary differs",
        )
        _require(
            _counter_is(summary.get("development_test_outcome_reads"), 0)
            and _counter_is(summary.get("new_final_evaluation_outcome_reads"), 0),
            f"{family} terminal summary reports a protected outcome read",
        )
        _require(
            isinstance(summary.get("cuda_device_name"), str)
            and bool(summary.get("cuda_device_name")),
            f"{family} terminal summary lacks CUDA device provenance",
        )
        checkpoint = _path(summary.get("checkpoint_path"), f"{family} checkpoint")
    elif family.startswith("setflow_"):
        stage = family.removeprefix("setflow_").upper()
        _require(
            summary.get("schema_version")
            == "route_a_v3_route2_xeditsetflow_v4_training_summary.v1"
            and summary.get("status")
            == "TERMINAL_XEDITSETFLOW_V4_TRAINING_COMPLETE_PENDING_VALIDATION"
            and summary.get("run_stage") == stage,
            f"{family} terminal summary differs",
        )
        _require(
            _counter_is(summary.get("development_test_outcome_reads"), 0)
            and _counter_is(summary.get("new_final_evaluation_outcome_reads"), 0),
            f"{family} terminal summary reports a protected outcome read",
        )
        checkpoints = summary.get("saved_checkpoint_paths")
        _require(
            isinstance(checkpoints, Mapping) and set(checkpoints) == {"4", "6", "8", "10"},
            f"{family} terminal checkpoint package differs",
        )
        checkpoint = _path(checkpoints["10"], f"{family} pass-10 checkpoint")
    else:
        _require(
            summary.get("schema_version")
            == "route_a_v3_route2_xeditflow_value_training.v4"
            and summary.get("status") == "XEDITFLOW_V4_VALUE_TRAINING_COMPLETE",
            f"{family} terminal summary differs",
        )
        _require(
            summary.get("development_test_outcomes_accessed_after_atomic_test")
            is False
            and summary.get("new_final_evaluation_outcomes_accessed") is False,
            f"{family} terminal summary reports a protected outcome read",
        )
        checkpoint = output_directory / "value_checkpoint.pt"
    _require(
        summary.get("parameter_changed") is True,
        f"{family} did not update parameters",
    )
    _require(
        summary.get("cpu_fallback_used") is False,
        f"{family} used CPU fallback",
    )
    precision = summary.get("training_precision", summary.get("precision"))
    _require(
        isinstance(precision, str) and "BF16" in precision.upper(),
        f"{family} terminal summary is not BF16",
    )
    _require(
        checkpoint.parent == output_directory,
        f"{family} checkpoint is outside its output directory",
    )
    return checkpoint, str(summary.get("schema_version"))


def _validate_attempt(job: Mapping[str, Any]) -> dict[str, Any]:
    family = str(job["family"])
    key = str(job["job_key"])
    output_directory = Path(job["output_directory"])
    config_path = Path(job["config_path"])
    attempt_path = output_directory / "training_attempt.json"
    summary_path = Path(job["terminal_summary_path"])
    failure_path = Path(job["failure_path"])
    log_path = Path(job["log_path"])
    _require(output_directory.is_dir(), f"training output directory is absent: {key}")
    for path, label in (
        (config_path, "training config"),
        (attempt_path, "training attempt"),
        (summary_path, "terminal training summary"),
        (log_path, "training log"),
    ):
        _require(path.is_file(), f"{label} is absent for {key}: {path}")
    _require(
        not failure_path.exists(), f"success and failure both exist for training job {key}"
    )
    config = _read_json(config_path, f"training config for {key}")
    attempt = _read_json(attempt_path, f"training_attempt.json for {key}")
    summary = _read_json(summary_path, f"terminal summary for {key}")
    checkpoint, terminal_evidence_schema = _validate_summary(
        job, summary, output_directory=output_directory
    )
    _require(checkpoint.is_file(), f"checkpoint is absent for {key}: {checkpoint}")

    attempt_id = attempt.get("attempt_id")
    _require(
        isinstance(attempt_id, str) and bool(attempt_id),
        f"training attempt_id is absent for {key}",
    )
    _require(attempt.get("status") == "COMPLETED", f"training attempt is not COMPLETED: {key}")
    started = attempt.get("started_at")
    completed = attempt.get("completed_at")
    _require(
        isinstance(started, str)
        and bool(started)
        and isinstance(completed, str)
        and bool(completed),
        f"training start/end timestamps are absent for {key}",
    )
    head = _head(attempt.get("code_commit"), f"training attempt code_commit for {key}")
    _require(
        head == job["training_git_head"],
        f"training attempt commit differs from schedule for {key}",
    )
    expected_attempt_id = job.get("expected_attempt_id")
    if expected_attempt_id is not None:
        _require(
            attempt_id == expected_attempt_id,
            f"training attempt_id differs from schedule for {key}",
        )
    _require(
        attempt.get("output_directory") == str(output_directory),
        f"training attempt output directory differs for {key}",
    )
    _require(
        attempt.get("evaluation_record_count") in {0, "0"},
        f"training attempt reports Evaluation records for {key}",
    )

    gpu = job["physical_gpu_index"]
    device = f"cuda:{gpu}"
    _require(
        attempt.get("physical_gpu_index") == gpu and attempt.get("device") == device,
        f"training attempt GPU/device differs for {key}",
    )
    if "physical_gpu_index" in config:
        _require(
            config.get("physical_gpu_index") == gpu,
            f"training config GPU differs for {key}",
        )
    if "device" in config:
        _require(config.get("device") == device, f"training config device differs for {key}")
    if "physical_gpu_index" in summary:
        _require(
            summary.get("physical_gpu_index") == gpu,
            f"terminal summary GPU differs for {key}",
        )
    summary_device = summary.get("torch_device", summary.get("device"))
    if summary_device is not None:
        _require(summary_device == device, f"terminal summary device differs for {key}")
    attempt_precision = attempt.get("training_precision")
    _require(
        isinstance(attempt_precision, str) and "BF16" in attempt_precision.upper(),
        f"training attempt is not BF16 for {key}",
    )
    seed = _attempt_seed(family, config, attempt, summary)
    if (
        family.startswith("critic_")
        and job.get("critic_terminal_evidence_version")
        == CRITIC_TERMINAL_EVIDENCE_V2
    ):
        _validate_critic_v2_terminal_evidence(
            job,
            config=config,
            attempt=attempt,
            summary=summary,
            seed=seed,
            output_directory=output_directory,
            summary_path=summary_path,
            checkpoint_path=checkpoint,
            attempt_path=attempt_path,
        )

    return {
        "attempt_id": attempt_id,
        "family": family,
        "schedule_path": str(job["schedule_path"]),
        "schedule_job_key": key,
        "code_commit": head,
        "config_path": str(config_path),
        "seed": seed,
        "physical_gpu_index": gpu,
        "device": device,
        "cuda_verified": True,
        "bf16_verified": True,
        "cpu_fallback_used": False,
        "parameter_changed": True,
        "started_at": started,
        "completed_at": completed,
        "output_directory": str(output_directory),
        "log_path": str(log_path),
        "checkpoint_path": str(checkpoint),
        "result_path": str(summary_path),
        "terminal_evidence_schema_version": terminal_evidence_schema,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
        "historical_free_memory_gate_applied": job[
            "historical_free_memory_gate_applied"
        ],
        "historical_free_memory_gate_policy": job[
            "historical_free_memory_gate_policy"
        ],
    }


def _validate_inventory(
    inventory_path: Path,
    *,
    final_schedule_path: Path,
    final_schedule: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    inventory = _read_json(inventory_path, "explicit training inventory")
    _require(
        inventory.get("schema_version") == INVENTORY_SCHEMA
        and inventory.get("status") == INVENTORY_STATUS,
        "explicit training inventory schema/status differs",
    )
    identities = inventory.get("attempts")
    _require(
        isinstance(identities, list) and len(identities) == EXPECTED_TOTAL_ATTEMPTS,
        "explicit training inventory does not contain 72 attempts",
    )
    parsed: list[tuple[str, Path, str]] = []
    for row in identities:
        _require(
            isinstance(row, Mapping)
            and set(row) == {"family", "schedule_path", "job_key"},
            "training inventory row must contain only family/schedule_path/job_key",
        )
        family = row.get("family")
        _require(family in EXPECTED_FAMILY_COUNTS, "training inventory family differs")
        schedule_path = _path(row.get("schedule_path"), "inventory schedule_path")
        key = row.get("job_key")
        _require(isinstance(key, str) and bool(key), "training inventory job_key differs")
        parsed.append((str(family), schedule_path, key))
    _require(
        len(set(parsed)) == EXPECTED_TOTAL_ATTEMPTS,
        "training inventory contains a duplicate schedule job identity",
    )
    counts = Counter(family for family, _, _ in parsed)
    _require(dict(counts) == EXPECTED_FAMILY_COUNTS, "training family counts differ")
    _require(
        all(
            path == final_schedule_path
            for family, path, _ in parsed
            if family == "final_value"
        ),
        "final value identities use a non-authoritative final schedule",
    )

    schedule_cache: dict[Path, tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]] = {}
    for path in sorted({path for _, path, _ in parsed}, key=str):
        schedule_cache[path] = _validate_schedule_and_training_jobs(
            path,
            final_schedule_path=final_schedule_path,
            preloaded_schedule=(
                final_schedule if path == final_schedule_path else None
            ),
        )
    jobs_by_identity: dict[tuple[str, Path, str], dict[str, Any]] = {}
    for path, (_, families) in schedule_cache.items():
        for family, jobs in families.items():
            for job in jobs:
                identity = (family, path, str(job["job_key"]))
                _require(
                    identity not in jobs_by_identity,
                    "allowlisted schedules contain a duplicate training identity",
                )
                jobs_by_identity[identity] = job
    for identity in parsed:
        _require(
            identity in jobs_by_identity,
            "explicit inventory row is not an allowlisted schedule training job: "
            f"{identity[0]}/{identity[2]}",
        )

    rows = [_validate_attempt(jobs_by_identity[identity]) for identity in parsed]
    attempt_ids = [row["attempt_id"] for row in rows]
    _require(
        len(set(attempt_ids)) == EXPECTED_TOTAL_ATTEMPTS,
        "training_attempt.json attempt_id is duplicated",
    )
    schedules = {path: value[0] for path, value in schedule_cache.items()}
    return rows, schedules


def _validate_posttest_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path, "outcome-free Atomic TEST authorization receipt")
    _require(
        receipt.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_authorization_receipt.v1"
        and receipt.get("status") == "XEDITCRITIC_V4_POSTTEST_AUTHORIZED"
        and receipt.get("frozen_test_gate_status")
        == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        and receipt.get("all_development_refit_authorized") is True
        and _counter_is(receipt.get("development_test_access_event_count"), 1)
        and receipt.get("general_test_projection_persisted") is False
        and receipt.get("test_bottom_six_cache_persisted") is False
        and receipt.get("development_test_metrics_in_receipt") is False
        and receipt.get("new_final_evaluation_outcomes_accessed") is False,
        "outcome-free Atomic TEST authorization receipt differs",
    )
    return receipt


def _validate_final_runtime_boundary(
    *,
    launch_path: Path,
    schedule_path: Path,
    runtime_path: Path,
    adjudication_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Mapping[str, Any]]]:
    # This ordering is a read boundary: no historical inventory, training
    # terminal summary, Atomic TEST receipt, or adjudication is read until the
    # authoritative Final runtime proves exact 98/98 terminal success.
    schedule = _read_json(schedule_path, "authoritative final schedule")
    _require(
        schedule.get("schema_version") == FINAL_SCHEDULE_SCHEMA
        and schedule.get("status") == "FROZEN_THREE_SEED_MATCHED_COMPUTE_SCHEDULE",
        "authoritative final schedule differs",
    )
    jobs = _final_jobs(schedule)
    launch = _read_json(launch_path, "final launch receipt")
    _require(
        launch.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_final_launch.v1"
        and launch.get("status") == "XEDITFLOW_V4_FINAL_SCHEDULER_LAUNCHED"
        and launch.get("git_head") == schedule.get("git_head")
        and _path(launch.get("schedule_path"), "final launch schedule_path")
        == schedule_path
        and _path(launch.get("runtime_manifest"), "final launch runtime_manifest")
        == runtime_path
        and _path(
            launch.get("final_adjudication_path"),
            "final launch final_adjudication_path",
        )
        == adjudication_path
        and launch.get("development_test_reopened") is False
        and _counter_is(launch.get("new_final_evaluation_outcome_reads"), 0),
        "final launch identity or protected-read boundary differs",
    )
    _require(
        _path(schedule.get("runtime_manifest"), "final runtime_manifest") == runtime_path,
        "final runtime path differs from the final schedule",
    )
    runtime = _read_json(runtime_path, "final runtime")
    _require(
        runtime.get("schema_version")
        == "route_a_v3_route2_xeditflow_v4_final_runtime.v1"
        and runtime.get("status") == "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL",
        "final runtime is not exact terminal",
    )
    _require(
        runtime.get("git_head") == schedule.get("git_head")
        and runtime.get("first_terminal_failure") is None
        and runtime.get("active_performance_output_read") is False
        and runtime.get("development_test_reopened") is False
        and runtime.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and _counter_is(runtime.get("new_final_evaluation_outcome_reads"), 0),
        "final runtime identity or protected-read boundary differs",
    )
    states = runtime.get("jobs")
    expected = {str(job["job_key"]): job for job in jobs}
    _require(
        isinstance(states, Mapping)
        and len(states) == 98
        and set(states) == set(expected),
        "final runtime does not contain the exact 98-job inventory",
    )
    for key, job in expected.items():
        state = states[key]
        _require(
            isinstance(state, Mapping)
            and state.get("status") == "TERMINAL_COMPLETE"
            and state.get("terminal_artifact_kind") == "SUCCESS"
            and _counter_is(state.get("return_code"), 0),
            f"final runtime job is not TERMINAL_COMPLETE/SUCCESS: {key}",
        )
        for field in ("success_path", "failure_path", "log_path"):
            _require(
                state.get(field) == job.get(field),
                f"final runtime path differs from schedule for {key}/{field}",
            )
        success = _path(job.get("success_path"), f"final {key} success_path")
        failure = _path(job.get("failure_path"), f"final {key} failure_path")
        log = _path(job.get("log_path"), f"final {key} log_path")
        _require(success.is_file(), f"final terminal artifact is absent: {key}")
        _require(not failure.exists(), f"final job has both success and failure: {key}")
        _require(log.is_file(), f"final job log is absent: {key}")

    adjudication_job = expected.get("adjudicate_final_comparison")
    _require(
        adjudication_job is not None
        and adjudication_path
        == _path(adjudication_job.get("success_path"), "final adjudication success_path"),
        "final adjudication path differs from the final schedule",
    )
    return schedule, runtime, jobs


def _validate_final_adjudication(path: Path) -> dict[str, Any]:
    adjudication = _read_json(path, "final adjudication")
    gate = adjudication.get("gate")
    _require(
        adjudication.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_adjudication.v4"
        and adjudication.get("status") == "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL"
        and isinstance(gate, Mapping)
        and gate.get("status") in {"XEDITFLOW_V4_PASS", "XEDITFLOW_V4_NO_GO"},
        "final adjudication is not terminal PASS/NO_GO",
    )
    passed = gate.get("status") == "XEDITFLOW_V4_PASS"
    _require(
        gate.get("new_final_evaluation_authorized") is passed
        and adjudication.get("new_final_evaluation_authorized") is passed
        and adjudication.get("additional_training_seed_authorized") is False
        and adjudication.get("submission_ready") is False
        and adjudication.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and adjudication.get("new_final_evaluation_outcomes_accessed") is False,
        "final adjudication authorization or protected-read boundary differs",
    )
    return adjudication


def _dependency_kind(job_key: str) -> str:
    key = job_key.lower()
    if "timing" in key or "equal_wall_time" in key:
        return "TIMING"
    if "adjudicate" in key or "compose" in key or "final_evidence" in key:
        return "ADJUDICATION"
    if any(value in key for value in ("critic", "evaluator", "metric", "closed", "open")):
        return "EVALUATOR"
    if any(value in key for value in ("rollout", "generation", "adapter")):
        return "GENERATION"
    return "FROZEN_DEPENDENCY"


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    _require(not path.exists(), f"terminal training ledger already exists: {path}")
    _require(not partial.exists(), f"terminal training ledger partial exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def export_terminal_training_ledger(
    *,
    final_launch_path: Path,
    final_schedule_path: Path,
    final_runtime_path: Path,
    final_adjudication_path: Path,
    training_inventory_path: Path,
    posttest_authorization_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not output_path.exists(), f"terminal training ledger already exists: {output_path}")
    _require(not partial.exists(), f"terminal training ledger partial exists: {partial}")

    final_schedule, _, final_jobs = _validate_final_runtime_boundary(
        launch_path=final_launch_path,
        schedule_path=final_schedule_path,
        runtime_path=final_runtime_path,
        adjudication_path=final_adjudication_path,
    )
    attempts, schedules = _validate_inventory(
        training_inventory_path,
        final_schedule_path=final_schedule_path,
        final_schedule=final_schedule,
    )
    _require(
        final_schedule_path in schedules,
        "authoritative final schedule is absent from the training inventory",
    )
    receipt = _validate_posttest_receipt(posttest_authorization_receipt_path)
    adjudication = _validate_final_adjudication(final_adjudication_path)
    counts = Counter(row["family"] for row in attempts)
    _require(
        len(attempts) == EXPECTED_TOTAL_ATTEMPTS
        and dict(counts) == EXPECTED_FAMILY_COUNTS,
        "validated terminal training rows do not close the 72-attempt inventory",
    )
    gate_status = str(adjudication["gate"]["status"])
    non_training = [
        job
        for job in final_jobs
        if not str(job["job_key"]).endswith(":value_training")
    ]
    dependency_counts = Counter(
        _dependency_kind(str(job["job_key"])) for job in non_training
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "final_gate_status": gate_status,
        "excellent_development_result": gate_status == "XEDITFLOW_V4_PASS",
        "submission_ready": False,
        "parameter_updating_attempt_count": EXPECTED_TOTAL_ATTEMPTS,
        "family_counts": dict(counts),
        "training_attempts": attempts,
        "stage_barriers": {
            "atomic_frozen_development_test": {
                "outcome_free_authorization_receipt_path": str(
                    posttest_authorization_receipt_path
                ),
                "authorization_status": receipt["status"],
                "frozen_test_gate_status": receipt["frozen_test_gate_status"],
                "development_test_access_event_count": 1,
                "metrics_present_in_receipt": False,
                "training_attempt_row_count": 0,
            },
            "validation": {
                "classification": "FROZEN_NON_PARAMETER_UPDATING_STAGE",
                "training_attempt_row_count": 0,
                "outcome_payload_read_by_exporter": False,
            },
            "post_atomic_development_test_reopened": False,
            "new_final_evaluation_outcome_reads": 0,
        },
        "frozen_dependencies": {
            "excluded_from_training_rows": [
                "ATOMIC_TEST",
                "VALIDATION",
                "GENERATION",
                "EVALUATOR",
                "TIMING",
                "ADJUDICATION",
            ],
            "final_schedule_non_parameter_updating_job_count": len(non_training),
            "final_schedule_non_parameter_updating_counts_by_kind": dict(
                dependency_counts
            ),
            "final_schedule_non_parameter_updating_job_keys": [
                str(job["job_key"]) for job in non_training
            ],
        },
        "source_artifacts": {
            "final_launch_path": str(final_launch_path),
            "final_schedule_path": str(final_schedule_path),
            "final_runtime_path": str(final_runtime_path),
            "final_adjudication_path": str(final_adjudication_path),
            "training_inventory_path": str(training_inventory_path),
        },
        "exporter_protected_reads": {
            "active_performance_output_reads": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
            "private_or_outcome_payload_reads": 0,
            "log_payload_reads": 0,
            "checkpoint_payload_reads": 0,
        },
    }
    _write_once(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-launch", required=True, type=Path)
    parser.add_argument("--final-schedule", required=True, type=Path)
    parser.add_argument("--final-runtime", required=True, type=Path)
    parser.add_argument("--final-adjudication", required=True, type=Path)
    parser.add_argument("--training-inventory", required=True, type=Path)
    parser.add_argument(
        "--posttest-authorization-receipt", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = export_terminal_training_ledger(
        final_launch_path=arguments.final_launch,
        final_schedule_path=arguments.final_schedule,
        final_runtime_path=arguments.final_runtime,
        final_adjudication_path=arguments.final_adjudication,
        training_inventory_path=arguments.training_inventory,
        posttest_authorization_receipt_path=(
            arguments.posttest_authorization_receipt
        ),
        output_path=arguments.output,
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
