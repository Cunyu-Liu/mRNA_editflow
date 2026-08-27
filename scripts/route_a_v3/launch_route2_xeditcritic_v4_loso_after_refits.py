#!/usr/bin/env python3
"""Prepare, authorize, and launch the exact 42-job Critic V4 LOSO package."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
LOSO_SCHEDULER = (
    WORKTREE / "scripts/route_a_v3/run_route2_xeditcritic_v4_loso_scheduler.py"
)
SEEDS = (20260908, 20260909, 20260910)
STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "GSE186455",
    "GSE256185",
    "GSE269595",
)
GPU_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
)


class XEditCriticV4LosoLaunchError(RuntimeError):
    pass


class XEditCriticV4LosoGpuInventoryError(XEditCriticV4LosoLaunchError):
    def __init__(
        self,
        message: str,
        *,
        reason: str,
        return_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        missing_physical_gpus: tuple[int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.command_line = GPU_INVENTORY_COMMAND
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.missing_physical_gpus = missing_physical_gpus


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV4LosoLaunchError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments, cwd=WORKTREE, text=True, capture_output=True, check=True
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def sibling_failure_path(runtime_root: Path) -> Path:
    return runtime_root.with_name(runtime_root.name + ".failed.json")


def require_fresh_prelaunch_family(runtime_root: Path) -> Path:
    failure = sibling_failure_path(runtime_root)
    partial = failure.with_suffix(failure.suffix + ".partial")
    require(
        not failure.exists() and not partial.exists(),
        "Critic prelaunch failure evidence already exists; use a new retry family",
    )
    return failure


def gpu_free_memory_mib(
    required_physical_gpus: Sequence[int],
) -> dict[int, int]:
    try:
        result = subprocess.run(
            list(GPU_INVENTORY_COMMAND),
            cwd=WORKTREE,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise XEditCriticV4LosoGpuInventoryError(
            f"nvidia-smi could not be executed: {error}",
            reason="COMMAND_EXECUTION_FAILED",
        ) from error
    if result.returncode != 0:
        raise XEditCriticV4LosoGpuInventoryError(
            f"nvidia-smi exited with return code {result.returncode}",
            reason="NONZERO_RETURN_CODE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    values: dict[int, int] = {}
    try:
        for line in result.stdout.splitlines():
            index, free = (part.strip() for part in line.split(",", maxsplit=1))
            values[int(index)] = int(free)
    except (TypeError, ValueError) as error:
        raise XEditCriticV4LosoGpuInventoryError(
            f"nvidia-smi inventory could not be parsed: {error}",
            reason="OUTPUT_PARSE_FAILED",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ) from error
    required = tuple(int(gpu) for gpu in required_physical_gpus)
    missing = tuple(sorted(set(required) - set(values)))
    if missing:
        raise XEditCriticV4LosoGpuInventoryError(
            f"configured physical GPU inventory is incomplete; missing {list(missing)}",
            reason="PHYSICAL_GPU_INVENTORY_INCOMPLETE",
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            missing_physical_gpus=missing,
        )
    return values


def write_prelaunch_failure_evidence(
    path: Path,
    *,
    expected_head: str,
    runtime_root: Path,
    error: XEditCriticV4LosoGpuInventoryError,
) -> None:
    require(
        not runtime_root.exists(),
        "Critic LOSO runtime root exists before prelaunch failure evidence",
    )
    require(
        not path.exists()
        and not path.with_suffix(path.suffix + ".partial").exists(),
        "Critic prelaunch failure evidence already exists; use a new retry family",
    )
    write_atomic(
        path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_prelaunch_failure.v1",
            "status": "XEDITCRITIC_PRELAUNCH_GPU_OR_CUDA_FAILURE",
            "launcher": "loso",
            "failure_stage": "INVENTORY",
            "expected_head": expected_head,
            "command": list(error.command_line),
            "return_code": error.return_code,
            "stdout": error.stdout,
            "stderr": error.stderr,
            "reason": error.reason,
            "missing_physical_gpus": list(error.missing_physical_gpus),
            "error_type": type(error).__name__,
            "error": str(error),
            "intended_runtime_root": str(runtime_root),
            "runtime_root_created": False,
            "jobs_started": 0,
            "cpu_fallback_used": False,
            "free_memory_gate_applied": False,
            "automatic_retry_attempted": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def eligible_loso_gpus(
    physical_gpu_indices: Sequence[int], inventory: Mapping[int, int]
) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(gpu) for gpu in physical_gpu_indices))
    require(
        bool(values) and all(0 <= gpu <= 5 for gpu in values),
        "no valid LOSO GPU 0–5 is configured",
    )
    require(all(gpu in inventory for gpu in values), "a configured LOSO GPU is absent")
    return values


def validate_loso_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_loso_job_manifest.v1"
        and payload.get("status")
        == "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED"
        and payload.get("required_seeds") == list(SEEDS)
        and payload.get("held_out_studies") == list(STUDIES)
        and int(payload.get("job_count", -1)) == 42,
        "Critic V4 LOSO manifest changed",
    )
    jobs = payload.get("jobs", [])
    identities = {
        (int(job["seed"]), job["held_out_study"], job["run_id"])
        for job in jobs
    }
    expected = {
        (seed, study, run_id)
        for seed in SEEDS
        for study in STUDIES
        for run_id in ("v4_full", "c0_v4")
    }
    require(len(jobs) == 42 and identities == expected, "Critic V4 LOSO job set changed")
    return list(jobs)


def run(head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "expected Git HEAD is invalid")
    require(PYTHON.is_file() and LOSO_SCHEDULER.is_file(), "formal LOSO runtime is absent")
    require(
        command(["git", "rev-parse", "HEAD"]).stdout.strip() == head,
        "A100 worktree is not at expected HEAD",
    )
    require(
        not command(["git", "status", "--porcelain"]).stdout.strip(),
        "A100 worktree is dirty",
    )
    refit_runtime_path = ROOT / f"experiments/xedit_v4/refit_execution_{head}/runtime.json"
    refit_runtime = read_json(refit_runtime_path)
    require(
        refit_runtime.get("status")
        == "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED"
        and refit_runtime.get("git_head") == head
        and refit_runtime.get("adjudication", {}).get("loso_authorized") is True
        and refit_runtime.get("active_performance_output_read") is False
        and int(refit_runtime.get("development_test_access_event_count_before_refit", -1)) == 1
        and int(refit_runtime.get("development_test_outcome_reads_during_refit", -1)) == 0
        and int(refit_runtime.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 exact three-refit predecessor is absent",
    )
    protocol = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
    protocol_payload = read_json(protocol)
    refit_manifest = Path(
        protocol_payload["all_development_refit"]["terminal_manifest_output"]
    )
    refit = read_json(refit_manifest)
    require(
        refit.get("status") == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit.get("required_seeds") == list(SEEDS)
        and int(refit.get("completed_refit_count", -1)) == 3
        and int(refit.get("refit_pass_count", -1)) == 8
        and refit.get("loso_authorized") is True
        and refit.get("development_test_outcomes_accessed_during_refit") is False
        and refit.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 refit terminal manifest changed",
    )
    runtime_root = ROOT / f"experiments/xedit_v4/loso_execution_{head}"
    require(not runtime_root.exists(), "Critic V4 LOSO execution runtime exists")
    prelaunch_failure_path = require_fresh_prelaunch_family(runtime_root)
    loso_config_root = Path(
        protocol_payload["test_preserving_loso"]["runtime_config_root"]
    )
    loso_run_root = Path(protocol_payload["test_preserving_loso"]["run_root"])
    loso_output = Path(
        protocol_payload["test_preserving_loso"]["adjudication_output"]
    )
    readiness_output = Path(protocol_payload["readiness_output"])
    for path in (loso_config_root, loso_run_root, loso_output, readiness_output):
        require(not path.exists(), f"Critic V4 LOSO artifact already exists: {path}")
    preflight = read_json(Path(protocol_payload["formal_preflight_path"]))
    required_mib = math.ceil(
        (float(preflight["selected_peak_allocated_gib"]) + 2.0) * 1024
    )
    configured_gpus = tuple(int(gpu) for gpu in protocol_payload["physical_gpu_indices"])
    try:
        free_memory = gpu_free_memory_mib(configured_gpus)
    except XEditCriticV4LosoGpuInventoryError as error:
        write_prelaunch_failure_evidence(
            prelaunch_failure_path,
            expected_head=head,
            runtime_root=runtime_root,
            error=error,
        )
        raise
    selected_gpus = eligible_loso_gpus(
        configured_gpus, free_memory
    )
    prepare = WORKTREE / "scripts/route_a_v3/prepare_route2_xeditcritic_v4_posttest_configs.py"
    authorize = WORKTREE / "scripts/route_a_v3/authorize_route2_xeditcritic_v4_posttest.py"
    trainer = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
    adjudicator = WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_posttest.py"
    readiness = WORKTREE / "scripts/route_a_v3/adjudicate_route2_xeditcritic_v4_readiness.py"
    command(
        [
            str(PYTHON), str(prepare),
            "--protocol", str(protocol),
            "--mode", "LOSO",
            "--refit-manifest", str(refit_manifest),
        ]
    )
    manifest_path = loso_config_root / "manifest.json"
    jobs = validate_loso_manifest(read_json(manifest_path))
    command(
        [
            str(PYTHON), str(authorize),
            "--protocol", str(protocol),
            "--stage", "LOSO",
            "--refit-manifest", str(refit_manifest),
        ]
    )
    authorization = Path(
        protocol_payload["test_preserving_loso"]["authorization_output"]
    )
    authorized = read_json(authorization)
    require(
        authorized.get("status") == "XEDITCRITIC_V4_LOSO_LAUNCH_AUTHORIZED"
        and authorized.get("authorized_git_head") == head
        and authorized.get("authorized_seeds") == list(SEEDS)
        and authorized.get("authorized_run_ids") == ["v4_full", "c0_v4"]
        and authorized.get("authorized_held_out_studies") == list(STUDIES)
        and authorized.get("all_three_refits_complete") is True
        and int(authorized.get("development_test_access_event_count_before_posttest", -1)) == 1
        and int(authorized.get("development_test_outcome_reads_during_posttest", -1)) == 0
        and int(authorized.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "Critic V4 LOSO launch authorization changed",
    )
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in selected_gpus}
    log_root = ROOT / f"logs/xedit_v4/loso_execution_{head}"
    for index, job in enumerate(jobs):
        gpu = selected_gpus[index % len(selected_gpus)]
        queues[gpu].append(
            {
                "job_key": f"loso:{job['seed']}:{job['held_out_study']}:{job['run_id']}",
                "seed": int(job["seed"]),
                "held_out_study": job["held_out_study"],
                "run_id": job["run_id"],
                "summary_path": job["summary_path"],
                "failure_path": job["failure_path"],
                "log_path": str(
                    log_root
                    / f"seed_{job['seed']}_{job['held_out_study']}_{job['run_id']}.log"
                ),
                "command": [
                    str(PYTHON), str(trainer),
                    "--config", job["config_path"],
                    "--run-id", job["run_id"],
                    "--physical-gpu-index", str(gpu),
                    "--launch-authorization", str(authorization),
                ],
            }
        )
    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True, exist_ok=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_schedule.v1",
        "status": "FROZEN_EXACT_42_JOB_LOSO_SCHEDULE",
        "git_head": head,
        "worktree": str(WORKTREE),
        "runtime_manifest": str(runtime_manifest),
        "selected_physical_gpus": list(selected_gpus),
        "gpu_free_memory_mib_before_launch": free_memory,
        "diagnostic_peak_plus_two_gib_mib": required_mib,
        "free_memory_gate_applied": False,
        "gpu_selection_policy": "FROZEN_PROTOCOL_PHYSICAL_GPU_ORDER",
        "gpu_queues": [
            {"physical_gpu_index": gpu, "jobs": rows}
            for gpu, rows in queues.items()
        ],
        "loso_adjudication": {
            "summary_path": str(loso_output),
            "failure_path": str(runtime_root / "loso_adjudication.failed.json"),
            "log_path": str(log_root / "loso_adjudication.log"),
            "command": [
                str(PYTHON), str(adjudicator),
                "--manifest", str(manifest_path),
                "--mode", "LOSO",
                "--output", str(loso_output),
            ],
        },
        "readiness": {
            "summary_path": str(readiness_output),
            "failure_path": str(runtime_root / "readiness.failed.json"),
            "log_path": str(log_root / "readiness.log"),
            "command": [
                str(PYTHON), str(readiness),
                "--protocol", str(protocol),
                "--three-seed-gate", protocol_payload["three_seed_gate_path"],
                "--posttest-authorization-receipt", protocol_payload["posttest_authorization_receipt_path"],
                "--refit-manifest", str(refit_manifest),
                "--loso-adjudication", str(loso_output),
            ],
        },
        "development_test_access_event_count_before_loso": 1,
        "development_test_outcome_reads_during_loso": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    schedule_path = runtime_root / "schedule.json"
    write_atomic(schedule_path, schedule)
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(LOSO_SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    result = {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_launch.v1",
        "status": "XEDITCRITIC_V4_EXACT_42_JOB_LOSO_LAUNCHED",
        "git_head": head,
        "scheduler_pid": process.pid,
        "loso_jobs_launched": 42,
        "selected_physical_gpus": list(selected_gpus),
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
        "development_test_access_event_count_before_loso": 1,
        "development_test_outcome_reads_during_loso": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
