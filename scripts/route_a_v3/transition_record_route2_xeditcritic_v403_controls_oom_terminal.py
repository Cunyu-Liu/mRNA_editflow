#!/usr/bin/env python3
"""Freeze the failed Y3 Critic-controls package before an independent retry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OLD_HEAD = "ebf99ebf8a253ad27e311e555121d328df8fae10"
HISTORICAL_FULL_GIT_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
HISTORICAL_C0_GIT_HEAD = "93703adec7a4c76b4466d3aaae8684620bee985a"
CONTROL_RUN_IDS = (
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
PHYSICAL_GPU_INDICES = (0, 1, 2, 3, 4, 5)
RUNTIME_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_control_recovery_runtime.v1"
)
RUNNING_STATUS = "XEDITCRITIC_V403_CONTROL_RECOVERY_RUNNING"
TECHNICAL_TERMINAL_STATUS = (
    "XEDITCRITIC_V403_CONTROL_RECOVERY_TECHNICAL_FAILURE"
)
RUNNING_OBSERVED_STATUS = (
    "XEDITCRITIC_V403_CONTROL_RECOVERY_STILL_RUNNING"
)
RECEIPT_SCHEMA = (
    "route_a_v3_route2_xeditcritic_v403_"
    "control_recovery_oom_terminal_receipt.v1"
)
RECEIPT_STATUS = (
    "XEDITCRITIC_V403_CONTROL_RECOVERY_OOM_TERMINAL_RECORDED"
)
LICENSED_WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_v403_no_vram_gate_20260827"
)
OLD_RUNTIME_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"v403_control_recovery_runner_{OLD_HEAD}"
)
OLD_RUNTIME = OLD_RUNTIME_ROOT / "runtime.json"
OLD_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_control_recovery_{OLD_HEAD}"
)
OLD_LOG_ROOT = (
    ROOT
    / "logs/xeditcritic_v4/"
    f"v403_control_recovery_{OLD_HEAD}"
)
CROSS_ROOT_GATE = (
    ROOT
    / "experiments/xeditcritic_v4/"
    f"screen_seed_20260907_v403_cross_root_controls_{OLD_HEAD}/"
    "screen_gate.json"
)
CANONICAL_RECEIPT = (
    ROOT
    / "audits/xeditcritic_v4/"
    f"v403_control_recovery_runner_{OLD_HEAD}_oom_terminal.json"
)
EXPECTED_FIRST_FAILURE = {
    "run_id": "v4_source_only",
    "reason": "JOB_TERMINAL_FAILURE_ARTIFACT",
    "return_code": 1,
    "terminal_artifact_kind": "FAILURE",
}


class CriticControlsOomTerminalError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticControlsOomTerminalError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _partial(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".partial")


def write_new_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"OOM terminal receipt already exists: {path}")
    partial = _partial(path)
    require(not partial.exists(), f"partial OOM terminal receipt already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def scheduler_process_is_alive(pid: int) -> bool:
    """Return whether PID still names this exact Y3 controls scheduler."""

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "args="],
        text=True,
        capture_output=True,
        check=False,
    )
    command_line = result.stdout.strip()
    error = result.stderr.strip()
    if result.returncode == 1 and not command_line and not error:
        return False
    require(
        result.returncode == 0,
        f"scheduler process inspection failed: {error or result.returncode}",
    )
    return (
        "run_route2_xeditcritic_v403_control_recovery_scheduler.py"
        in command_line
        and OLD_HEAD in command_line
    )


def _protected_reads_zero(payload: Mapping[str, Any], label: str) -> None:
    require(
        int(payload.get("development_test_outcome_reads", -1)) == 0
        and int(payload.get("new_final_evaluation_outcome_reads", -1)) == 0,
        f"{label} reports a protected outcome read",
    )


def _runtime_identity(runtime: Mapping[str, Any]) -> None:
    require(
        runtime.get("schema_version") == RUNTIME_SCHEMA,
        "old controls runtime schema changed",
    )
    require(
        runtime.get("historical_full_git_head") == HISTORICAL_FULL_GIT_HEAD
        and runtime.get("historical_c0_git_head") == HISTORICAL_C0_GIT_HEAD
        and runtime.get("current_git_head") == OLD_HEAD
        and runtime.get("runner_git_head") == OLD_HEAD
        and runtime.get("orchestration_git_head") == OLD_HEAD
        and runtime.get("training_code_git_head") == OLD_HEAD,
        "old controls runtime lineage changed",
    )
    require(
        Path(str(runtime.get("training_worktree", ""))) == LICENSED_WORKTREE,
        "old controls training worktree changed",
    )
    require(
        runtime.get("ordered_control_run_ids") == list(CONTROL_RUN_IDS),
        "old controls run order changed",
    )
    require(
        runtime.get("cross_root_adjudication_run") is False,
        "old controls runtime reports cross-root adjudication",
    )
    require(
        runtime.get("full_retrained") is False
        and runtime.get("c0_retrained") is False
        and runtime.get("old_v402_stopped_process_resumed") is False,
        "old controls runtime changed historical producer isolation",
    )
    require(
        runtime.get("free_memory_gate_applied") is False,
        "old controls runtime used a free-memory gate",
    )
    require(
        int(runtime.get("terminal_artifact_payloads_read_by_scheduler", -1)) == 0
        and int(
            runtime.get("historical_terminal_payloads_read_before_cross_root", -1)
        )
        == 0,
        "old controls runtime reports an unauthorized terminal payload read",
    )
    _protected_reads_zero(runtime, "old controls runtime")


def _terminal_paths(output_directory: Path) -> tuple[Path, Path]:
    return output_directory / "run_summary.json", output_directory / "failure.json"


def _inspect_job(
    run_id: str,
    row: Mapping[str, Any],
    *,
    expected_gpu: int,
) -> dict[str, Any]:
    require(row.get("run_id") == run_id, f"old controls job identity changed: {run_id}")
    require(
        int(row.get("physical_gpu_index", -1)) == expected_gpu,
        f"old controls GPU assignment changed: {run_id}",
    )
    output = Path(str(row.get("output_directory", "")))
    log = Path(str(row.get("log_path", "")))
    require(
        output == OLD_OUTPUT_ROOT / run_id,
        f"old controls output directory changed: {run_id}",
    )
    require(
        log == OLD_LOG_ROOT / f"{run_id}.log",
        f"old controls log path changed: {run_id}",
    )
    require(
        row.get("training_git_head") == OLD_HEAD,
        f"old controls training HEAD changed: {run_id}",
    )
    expected_attempt_id = (
        "xeditcritic_v4_screen_seed20260907::"
        f"{run_id}::v403_control_recovery_{OLD_HEAD}"
    )
    require(
        row.get("training_attempt_id") == expected_attempt_id,
        f"old controls attempt identity changed: {run_id}",
    )
    status = row.get("status")
    require(
        status not in {"PENDING", "RUNNING"},
        f"old controls job is not terminal: {run_id}",
    )
    require(
        status in {"TERMINAL_SUMMARY", "TECHNICAL_FAILURE"},
        f"old controls job has an ambiguous terminal status: {run_id}",
    )
    summary, failure = _terminal_paths(output)
    require(
        not _partial(summary).exists() and not _partial(failure).exists(),
        f"old controls job has a partial terminal artifact: {run_id}",
    )
    terminal_count = int(summary.is_file()) + int(failure.is_file())
    require(
        terminal_count == 1,
        f"old controls job lacks a unique terminal artifact: {run_id}",
    )
    observed_kind = "SUMMARY" if summary.is_file() else "FAILURE"
    require(
        row.get("terminal_artifact_kind") == observed_kind,
        f"old controls runtime/artifact terminal kind differs: {run_id}",
    )
    return_code = row.get("return_code")
    require(
        isinstance(return_code, int) and not isinstance(return_code, bool),
        f"old controls return code is absent: {run_id}",
    )
    if status == "TERMINAL_SUMMARY":
        require(
            observed_kind == "SUMMARY" and return_code == 0,
            f"old controls success row is not exact: {run_id}",
        )
    else:
        require(
            observed_kind == "FAILURE" and return_code != 0,
            f"old controls failure row points to a successful terminal: {run_id}",
        )
    return {
        "run_id": run_id,
        "physical_gpu_index": expected_gpu,
        "status": status,
        "return_code": return_code,
        "terminal_artifact_kind": observed_kind,
        "terminal_artifact": str(summary if observed_kind == "SUMMARY" else failure),
        "output_directory": str(output),
        "log_path": str(log),
    }


def _inspect_jobs(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = runtime.get("jobs")
    require(isinstance(jobs, Mapping), "old controls runtime job table is absent")
    # The scheduler writes JSON with sort_keys=True; the authoritative order is
    # the separate ordered_control_run_ids field validated above.
    require(
        set(jobs) == set(CONTROL_RUN_IDS),
        "old controls runtime job inventory changed",
    )
    rows: list[dict[str, Any]] = []
    for run_id, expected_gpu in zip(
        CONTROL_RUN_IDS, PHYSICAL_GPU_INDICES, strict=True
    ):
        row = jobs[run_id]
        require(
            isinstance(row, Mapping),
            f"old controls job row is not an object: {run_id}",
        )
        rows.append(_inspect_job(run_id, row, expected_gpu=expected_gpu))
    return rows


def _inspect_first_failure(
    runtime: Mapping[str, Any], jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    first = runtime.get("first_terminal_failure")
    require(
        isinstance(first, Mapping) and bool(first),
        "old controls runtime lacks one first terminal failure",
    )
    for key, expected in EXPECTED_FIRST_FAILURE.items():
        require(
            first.get(key) == expected,
            f"old controls first terminal failure changed: {key}",
        )
    run_id = str(first["run_id"])
    bound = [row for row in jobs if row["run_id"] == run_id]
    require(
        len(bound) == 1
        and bound[0]["status"] == "TECHNICAL_FAILURE"
        and bound[0]["return_code"] == first.get("return_code")
        and bound[0]["terminal_artifact_kind"]
        == first.get("terminal_artifact_kind")
        and first.get("output_directory") == bound[0]["output_directory"]
        and first.get("log_path") == bound[0]["log_path"]
        and first.get("worktree_inspection") is None,
        "old controls first terminal failure is not stably bound",
    )
    return {
        "run_id": run_id,
        "reason": str(first["reason"]),
        "return_code": int(first["return_code"]),
        "terminal_artifact_kind": str(first["terminal_artifact_kind"]),
        "output_directory": str(first["output_directory"]),
        "log_path": str(first["log_path"]),
    }


def _require_cross_root_absent() -> None:
    failure = CROSS_ROOT_GATE.with_suffix(".failed.json")
    for path in (
        CROSS_ROOT_GATE,
        _partial(CROSS_ROOT_GATE),
        failure,
        _partial(failure),
    ):
        require(
            not path.exists(),
            f"cross-root gate or adjudication evidence already exists: {path}",
        )


def run(
    *,
    runtime_path: Path = OLD_RUNTIME,
    receipt_path: Path = CANONICAL_RECEIPT,
    process_is_alive: Callable[[int], bool] = scheduler_process_is_alive,
) -> dict[str, Any]:
    require(runtime_path == OLD_RUNTIME, "old controls runtime path is not canonical")
    require(receipt_path == CANONICAL_RECEIPT, "OOM terminal receipt path is not canonical")
    require(not receipt_path.exists(), f"OOM terminal receipt already exists: {receipt_path}")
    require(
        not _partial(receipt_path).exists(),
        f"partial OOM terminal receipt already exists: {_partial(receipt_path)}",
    )

    # This is the transition's only JSON payload read. Terminal artifacts are
    # checked by existence and never reopened after their prior diagnosis.
    runtime = read_json(runtime_path)
    _runtime_identity(runtime)
    runtime_status = runtime.get("status")
    if runtime_status == RUNNING_STATUS:
        return {
            "status": RUNNING_OBSERVED_STATUS,
            "runtime_status": RUNNING_STATUS,
            "receipt_written": False,
        }
    require(
        runtime_status == TECHNICAL_TERMINAL_STATUS,
        "old controls runtime is not the exact technical terminal",
    )
    require(
        not _partial(runtime_path).exists(),
        "terminal old controls runtime has a partial sibling",
    )
    scheduler_pid = runtime.get("scheduler_pid")
    require(
        isinstance(scheduler_pid, int)
        and not isinstance(scheduler_pid, bool)
        and scheduler_pid > 0,
        "old controls scheduler PID is absent",
    )
    require(
        not process_is_alive(scheduler_pid),
        "old controls scheduler process is still alive",
    )
    jobs = _inspect_jobs(runtime)
    first_failure = _inspect_first_failure(runtime, jobs)
    _require_cross_root_absent()
    technical_failure_run_ids = [
        row["run_id"] for row in jobs if row["status"] == "TECHNICAL_FAILURE"
    ]
    terminal_summary_run_ids = [
        row["run_id"] for row in jobs if row["status"] == "TERMINAL_SUMMARY"
    ]
    require(
        technical_failure_run_ids
        and len(technical_failure_run_ids) + len(terminal_summary_run_ids)
        == len(CONTROL_RUN_IDS),
        "old controls terminal job partition is incomplete",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "terminal_class": "TECHNICAL_FAILURE_TERMINAL",
        "old_current_git_head": OLD_HEAD,
        "old_runner_git_head": OLD_HEAD,
        "old_orchestration_git_head": OLD_HEAD,
        "old_training_code_git_head": OLD_HEAD,
        "old_runtime_path": str(OLD_RUNTIME),
        "old_runtime_status": TECHNICAL_TERMINAL_STATUS,
        "scheduler_pid": scheduler_pid,
        "scheduler_process_gone": True,
        "ordered_control_run_ids": list(CONTROL_RUN_IDS),
        "terminal_jobs": jobs,
        "technical_failure_run_ids": technical_failure_run_ids,
        "terminal_summary_run_ids": terminal_summary_run_ids,
        "first_terminal_failure": first_failure,
        "cross_root_adjudication_run": False,
        "cross_root_gate_path": str(CROSS_ROOT_GATE),
        "cross_root_gate_absent": True,
        "free_memory_gate_applied": False,
        "terminal_artifact_payloads_read_by_transition": 0,
        "historical_terminal_payloads_read_before_cross_root": 0,
        "successor_authorized": False,
        "same_family_retry_authorized": False,
        "new_independent_retry_eligible": True,
        "old_family_artifacts_read_only": True,
        "old_runtime_read_count_this_transition": 1,
        "gpu_inventory_or_probe_executed": False,
        "gpu_or_model_execution_started": False,
        "protected_outcome_payload_read": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_new_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    print(json.dumps(run(), sort_keys=True))


if __name__ == "__main__":
    main()
