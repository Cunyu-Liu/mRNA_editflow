#!/usr/bin/env python3
"""Launch one repaired Critic V4 full arm after the exact CUDA replay smoke."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping


WORKTREE = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BASE_CONFIG = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
RUN_ID = "v4_full"
OLD_OUTPUT_ROOT = ROOT / "experiments/xeditcritic_v4/screen_seed_20260907"
V402_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4/"
    "screen_seed_20260907_v402_recovery_runner_"
    "93703adec7a4c76b4466d3aaae8684620bee985a"
)
PREFLIGHT = OLD_OUTPUT_ROOT / "preflight_attempt_5/preflight.json"
SMOKE_PROVENANCE_HEAD = "f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea"
CANONICAL_RECOVERY_HEAD = SMOKE_PROVENANCE_HEAD
CANONICAL_ATTEMPT_ID = (
    "xeditcritic_v4_screen_seed20260907::v4_full::"
    f"v403_rng_replay_fix_{CANONICAL_RECOVERY_HEAD}"
)
CANONICAL_OUTPUT_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4"
    / f"screen_seed_20260907_v403_rng_replay_fix_{CANONICAL_RECOVERY_HEAD}"
)
CANONICAL_RUNTIME_ROOT = (
    ROOT
    / "experiments/xeditcritic_v4"
    / f"v403_rng_replay_fix_runner_{CANONICAL_RECOVERY_HEAD}"
)
CANONICAL_AUTHORIZATION_ROOT = (
    ROOT
    / "authorizations/xeditcritic_v4"
    / f"v403_rng_replay_fix_{CANONICAL_RECOVERY_HEAD}"
)
CANONICAL_LOG_ROOT = (
    ROOT
    / "logs/xeditcritic_v4"
    / f"v403_rng_replay_fix_{CANONICAL_RECOVERY_HEAD}"
)
SMOKE_TRAINING_SEMANTIC_PATHS = (
    "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json",
    "core",
    "scripts/route_a_v3/preflight_route2_xeditcritic_v4.py",
    "scripts/route_a_v3/smoke_route2_xeditcritic_v402_recovery.py",
    "scripts/route_a_v3/train_route2_xeditcritic_v3.py",
    "scripts/route_a_v3/train_route2_xeditcritic_v4.py",
    ":(glob)scripts/route_a_v3/route2_mrnabert_*.py",
)


class XEditCriticV403FullRecoveryLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV403FullRecoveryLaunchError(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def require_canonical_attempt_unconsumed(current_head: str) -> None:
    """Keep the f34 seed-20260907 full recovery a true one-shot attempt."""

    runtime_path = CANONICAL_RUNTIME_ROOT / "runtime.json"
    schedule_path = CANONICAL_RUNTIME_ROOT / "schedule.json"
    launch_path = CANONICAL_RUNTIME_ROOT / "launch.json"
    authorization_path = CANONICAL_AUTHORIZATION_ROOT / "launch_authorization.json"
    if runtime_path.is_file():
        runtime = read_json(runtime_path)
        require(
            runtime.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1"
            and runtime.get("run_id") == RUN_ID
            and runtime.get("git_head") == CANONICAL_RECOVERY_HEAD
            and runtime.get("status")
            in {
                "XEDITCRITIC_V403_FULL_RECOVERY_RUNNING",
                "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL",
                "XEDITCRITIC_V403_FULL_RECOVERY_NO_TERMINAL_ARTIFACT",
            },
            "canonical Critic V4.0.3 runtime identity is invalid; use an explicit retry family",
        )
    if schedule_path.is_file():
        schedule = read_json(schedule_path)
        require(
            schedule.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v403_full_recovery_schedule.v1"
            and schedule.get("run_id") == RUN_ID
            and schedule.get("git_head") == CANONICAL_RECOVERY_HEAD
            and CANONICAL_ATTEMPT_ID in list(schedule.get("command", [])),
            "canonical Critic V4.0.3 schedule identity is invalid; use an explicit retry family",
        )
    consumed_paths = tuple(
        path
        for path in (
            CANONICAL_OUTPUT_ROOT,
            CANONICAL_RUNTIME_ROOT,
            CANONICAL_AUTHORIZATION_ROOT,
            CANONICAL_LOG_ROOT,
            launch_path,
            authorization_path,
        )
        if path.exists()
    )
    require(
        not consumed_paths,
        "canonical Critic V4.0.3 f34 seed-20260907 v4_full attempt is already "
        "RUNNING, terminal, or consumed; use an explicit new retry family: "
        + ", ".join(str(path) for path in consumed_paths),
    )
    require(
        current_head == CANONICAL_RECOVERY_HEAD,
        "this one-shot Critic recovery launcher is bound to the canonical f34 "
        "attempt; use an explicit new retry family for another HEAD",
    )


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=True,
    )


def build_recovery_config(
    base: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    recovery = copy.deepcopy(dict(base))
    recovery["output_root"] = str(output_root)
    recovery["screen_gate_output"] = str(output_root / "screen_gate.json")
    allowed = {"output_root", "screen_gate_output"}
    base_science = {key: value for key, value in base.items() if key not in allowed}
    recovery_science = {
        key: value for key, value in recovery.items() if key not in allowed
    }
    require(base_science == recovery_science, "V4.0.3 changed a scientific config field")
    require(
        str(base["output_root"]) == str(OLD_OUTPUT_ROOT),
        "base Critic output root changed",
    )
    return recovery


def gpu_free_memory_bytes(physical_gpu_index: int) -> int:
    result = command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index, free = (part.strip() for part in line.split(",", maxsplit=1))
        values[int(index)] = int(free) * 1024**2
    require(physical_gpu_index in values, "selected physical GPU is absent")
    return values[physical_gpu_index]


def smoke_memory_diagnostic_bytes(smoke: Mapping[str, Any]) -> tuple[int, str]:
    """Read either V4.0.3 smoke diagnostic spelling without making it a gate."""

    for field in (
        "diagnostic_peak_plus_two_gib_bytes",
        "required_free_memory_bytes",
    ):
        value = smoke.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, field
    raise XEditCriticV403FullRecoveryLaunchError(
        "V4.0.3 smoke lacks a positive peak-plus-two-GiB diagnostic"
    )


def require_smoke_training_semantics_unchanged(current_head: str) -> dict[str, Any]:
    """Bind the f34 GPU smoke to a later launcher with unchanged training code."""

    changed = command(
        [
            "git",
            "diff",
            "--name-only",
            SMOKE_PROVENANCE_HEAD,
            current_head,
            "--",
            *SMOKE_TRAINING_SEMANTIC_PATHS,
        ]
    ).stdout.splitlines()
    require(
        not changed,
        "Critic training semantics changed after the f34 GPU smoke: "
        + ", ".join(changed),
    )
    return {
        "smoke_git_head": SMOKE_PROVENANCE_HEAD,
        "launcher_git_head": current_head,
        "training_semantic_paths": list(SMOKE_TRAINING_SEMANTIC_PATHS),
        "training_semantic_diff_paths": changed,
        "training_semantics_unchanged": True,
    }


def build_launch_authorization(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    current_head: str,
    physical_gpu_index: int,
) -> dict[str, Any]:
    run_ids = [str(row["run_id"]) for row in config["required_screen_runs"]]
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_head,
        "preflight_runner_git_head": str(preflight["git_head"]),
        "authorized_run_ids": run_ids,
        "barriers": {
            "all_five_c3_jobs_terminal": True,
            "c3_terminal_summaries_read_exactly_once": True,
            "a100_current_head_focused_tests_passed": True,
            "a100_current_head_v332_tests_passed": True,
            "bottom_six_cache_terminal_complete": True,
            "formal_parameter_preflight_passed": True,
            "formal_memory_preflight_passed": True,
            "cache_online_equivalence_passed": True,
        },
        "v403_rng_replay_recovery": {
            "run_id": RUN_ID,
            "physical_gpu_index": physical_gpu_index,
            "strict_full_model_rng_replay_smoke_passed": True,
            "strict_full_model_rng_replay_smoke_git_head": SMOKE_PROVENANCE_HEAD,
            "scientific_config_changed": False,
            "historical_c0_reference_reused": True,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def terminal_kind(output: Path) -> str | None:
    summary = output / "run_summary.json"
    failure = output / "failure.json"
    if summary.exists() == failure.exists():
        return None
    return "SUMMARY" if summary.exists() else "FAILURE"


def run_worker(schedule_path: Path) -> None:
    schedule = read_json(schedule_path)
    require(
        schedule.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v403_full_recovery_schedule.v1",
        "V4.0.3 worker schedule schema changed",
    )
    require(schedule.get("run_id") == RUN_ID, "V4.0.3 worker run changed")
    runtime_path = Path(str(schedule["runtime_manifest"]))
    output = Path(str(schedule["output_directory"]))
    log = Path(str(schedule["log_path"]))
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(schedule["command"]),
            cwd=Path(str(schedule["worktree"])),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        write_atomic(
            runtime_path,
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1",
                "status": "XEDITCRITIC_V403_FULL_RECOVERY_RUNNING",
                "worker_pid": os.getpid(),
                "training_pid": process.pid,
                "run_id": RUN_ID,
                "physical_gpu_index": schedule["physical_gpu_index"],
                "git_head": schedule["git_head"],
                "started_unix_seconds": started,
                "terminal_artifact_payloads_read": 0,
                "development_test_outcome_reads": 0,
                "new_final_evaluation_outcome_reads": 0,
            },
        )
        return_code = process.wait()
    kind = terminal_kind(output)
    write_atomic(
        runtime_path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v403_full_recovery_runtime.v1",
            "status": (
                "XEDITCRITIC_V403_FULL_RECOVERY_TERMINAL"
                if kind is not None
                else "XEDITCRITIC_V403_FULL_RECOVERY_NO_TERMINAL_ARTIFACT"
            ),
            "worker_pid": os.getpid(),
            "run_id": RUN_ID,
            "physical_gpu_index": schedule["physical_gpu_index"],
            "git_head": schedule["git_head"],
            "return_code": return_code,
            "terminal_artifact_kind": kind,
            "started_unix_seconds": started,
            "finished_unix_seconds": time.time(),
            "terminal_artifact_payloads_read": 0,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )


def launch(expected_head: str, physical_gpu_index: int) -> dict[str, Any]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected HEAD is invalid",
    )
    require(0 <= physical_gpu_index <= 5, "Critic V4 physical GPU must be 0–5")
    require(PYTHON.is_file() and BASE_CONFIG.is_file() and TRAINER.is_file(), "V4.0.3 code or Python is absent")
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    require(command(["git", "rev-parse", "HEAD"]).stdout.strip() == expected_head, "V4.0.3 worktree is at another HEAD")
    require(not command(["git", "status", "--porcelain"]).stdout.strip(), "V4.0.3 worktree is dirty")
    require_canonical_attempt_unconsumed(expected_head)

    preflight = read_json(PREFLIGHT)
    require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True,
        "formal Critic preflight is not PASS",
    )
    smoke_path = (
        ROOT
        / "audits/xeditcritic_v4"
        / f"v403_rng_replay_smoke_{SMOKE_PROVENANCE_HEAD}.json"
    )
    smoke = read_json(smoke_path)
    require(
        smoke.get("status") == "XEDITCRITIC_V403_FULL_MODEL_RNG_REPLAY_SMOKE_PASS"
        and str(smoke.get("git_head")) == SMOKE_PROVENANCE_HEAD
        and smoke.get("strict_replay_prediction_equal") is True
        and smoke.get("retained_graph_prediction_equal_to_replay") is True
        and smoke.get("retained_graph_parameter_gradients_equal_to_replay") is True
        and smoke.get("retained_graph_gradient_norm_equal_to_replay") is True
        and smoke.get("retained_graph_rng_terminal_state_equal_to_replay") is True
        and float(smoke.get("router_balance_weight_exercised", 0.0)) > 0
        and int(smoke.get("formal_training_forward_count_per_update", -1)) == 1
        and int(smoke.get("full_cache_validation_count_before_batching", -1)) == 1
        and smoke.get("full_cache_validation_per_batch") is False
        and smoke.get("optimizer_state_materialized") is True
        and smoke.get("target_value_accessed") is False
        and smoke.get("validation_metric_read") is False
        and int(smoke.get("development_test_outcome_reads", -1)) == 0
        and int(smoke.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "f34 full-model RNG replay smoke is absent or invalid",
    )
    smoke_provenance = require_smoke_training_semantics_unchanged(expected_head)
    historical_c0 = V402_OUTPUT_ROOT / "c0_v4/run_summary.json"
    require(historical_c0.is_file(), "matched V4.0.2 C0 terminal summary is absent")

    output_root = (
        ROOT
        / "experiments/xeditcritic_v4"
        / f"screen_seed_20260907_v403_rng_replay_fix_{expected_head}"
    )
    runtime_root = (
        ROOT
        / "experiments/xeditcritic_v4"
        / f"v403_rng_replay_fix_runner_{expected_head}"
    )
    authorization_root = (
        ROOT
        / "authorizations/xeditcritic_v4"
        / f"v403_rng_replay_fix_{expected_head}"
    )
    log_root = (
        ROOT / "logs/xeditcritic_v4" / f"v403_rng_replay_fix_{expected_head}"
    )
    for path, label in (
        (output_root, "V4.0.3 output root"),
        (runtime_root, "V4.0.3 runtime root"),
        (authorization_root, "V4.0.3 authorization root"),
    ):
        require(not path.exists(), f"{label} already exists")

    free_memory_bytes = gpu_free_memory_bytes(physical_gpu_index)
    diagnostic_peak_plus_two_gib_bytes, smoke_memory_diagnostic_field = (
        smoke_memory_diagnostic_bytes(smoke)
    )
    base = read_json(BASE_CONFIG)
    recovery_config = build_recovery_config(base, output_root)
    authorization_root.mkdir(parents=True)
    config_path = authorization_root / "screen_config.json"
    authorization_path = authorization_root / "launch_authorization.json"
    write_atomic(config_path, recovery_config)
    write_atomic(
        authorization_path,
        build_launch_authorization(
            recovery_config,
            preflight,
            current_head=expected_head,
            physical_gpu_index=physical_gpu_index,
        ),
    )

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    runtime_manifest = runtime_root / "runtime.json"
    schedule_path = runtime_root / "schedule.json"
    log_path = log_root / f"{RUN_ID}.log"
    output_directory = output_root / RUN_ID
    schedule = {
        "schema_version": "route_a_v3_route2_xeditcritic_v403_full_recovery_schedule.v1",
        "status": "XEDITCRITIC_V403_FULL_RECOVERY_SCHEDULED",
        "git_head": expected_head,
        "worktree": str(WORKTREE),
        "run_id": RUN_ID,
        "physical_gpu_index": physical_gpu_index,
        "output_directory": str(output_directory),
        "runtime_manifest": str(runtime_manifest),
        "log_path": str(log_path),
        "screen_config": str(config_path),
        "launch_authorization": str(authorization_path),
        "strict_rng_replay_smoke": str(smoke_path),
        "strict_rng_replay_smoke_provenance": smoke_provenance,
        "smoke_memory_diagnostic_source_field": smoke_memory_diagnostic_field,
        "historical_c0_reference": str(historical_c0),
        "diagnostic_peak_plus_two_gib_bytes": diagnostic_peak_plus_two_gib_bytes,
        "free_memory_bytes_before_launch": free_memory_bytes,
        "free_memory_gate_applied": False,
        "command": [
            str(PYTHON),
            str(TRAINER),
            "--config",
            str(config_path),
            "--run-id",
            RUN_ID,
            "--physical-gpu-index",
            str(physical_gpu_index),
            "--launch-authorization",
            str(authorization_path),
            "--training-attempt-id",
            (
                "xeditcritic_v4_screen_seed20260907::v4_full::"
                f"v403_rng_replay_fix_{expected_head}"
            ),
        ],
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    stream = (log_root / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "--worker-schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch_payload = {
        "schema_version": "route_a_v3_route2_xeditcritic_v403_full_recovery_launch.v1",
        "status": "XEDITCRITIC_V403_FULL_RECOVERY_LAUNCHED",
        "git_head": expected_head,
        "worker_pid": process.pid,
        "run_id": RUN_ID,
        "physical_gpu_index": physical_gpu_index,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "strict_rng_replay_smoke": str(smoke_path),
        "strict_rng_replay_smoke_provenance": smoke_provenance,
        "historical_c0_reference": str(historical_c0),
        "scientific_config_changed": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch_payload)
    return launch_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--worker-schedule", type=Path)
    arguments = parser.parse_args()
    if arguments.worker_schedule is not None:
        require(arguments.expected_head is None and arguments.physical_gpu_index is None, "worker arguments are mixed with launch arguments")
        run_worker(arguments.worker_schedule)
        return
    require(arguments.expected_head is not None, "--expected-head is required")
    require(arguments.physical_gpu_index is not None, "--physical-gpu-index is required")
    print(
        json.dumps(
            launch(arguments.expected_head, arguments.physical_gpu_index),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
