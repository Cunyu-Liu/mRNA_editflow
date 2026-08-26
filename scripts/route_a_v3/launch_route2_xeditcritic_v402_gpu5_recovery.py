#!/usr/bin/env python3
"""Verify and detach the single V4.0.2 eight-arm Critic recovery on GPU5."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_v402_recovery_20260826"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BASE_CONFIG = WORKTREE / "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json"
TRAINER = WORKTREE / "scripts/route_a_v3/train_route2_xeditcritic_v4.py"
SCHEDULER = (
    WORKTREE
    / "scripts/route_a_v3/run_route2_xeditcritic_v402_recovery_scheduler.py"
)
AMENDMENT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v402_technical_recovery_amendment_v1.json"
)
LOCAL_SMOKE_RECEIPT = (
    WORKTREE
    / "audits/route_a_v3_route2_xeditcritic_v402_gpu5_smoke_terminal_v1.json"
)
DIAGNOSIS = (
    ROOT
    / "audits/xeditcritic_v4/v402_failure_diagnosis_read_once.json"
)
PREFLIGHT = (
    ROOT
    / "experiments/xeditcritic_v4/screen_seed_20260907/"
    "preflight_attempt_5/preflight.json"
)
OLD_OUTPUT_ROOT = (
    ROOT / "experiments/xeditcritic_v4/screen_seed_20260907"
)
SINGLE_LAUNCH_MARKER = (
    ROOT / "authorizations/xeditcritic_v4/v402_recovery_launch_consumed.json"
)
RUN_IDS = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)
CACHE_EXPERIMENT_HEAD = "a7ef72fac23cd5b25dcc6c8d560236b97fa8b09d"
CRITIC_TEST_PATTERNS = (
    "tests/route_a_v3/*xeditcritic_v4*.py",
    "tests/route_a_v3/*xeditcritic_v402*.py",
    "tests/route_a_v3/test_route2_xeditcritic_batch_v4.py",
    "tests/route_a_v3/test_route2_bottom_encoder_chunk_cache_v4.py",
    "tests/route_a_v3/test_route2_mrnabert_bottom_six_encoder_v4.py",
    "tests/route_a_v3/test_route2_xedit_v4_interfaces.py",
    "tests/route_a_v3/test_route2_xedit_v4_method_repair_protocol_v1.py",
    "tests/route_a_v3/test_authorize_route2_xedit_v4_screen_stages.py",
)
V332_TEST_PATTERNS = ("tests/route_a_v3/*v332*.py",)


class XEditCriticV402RecoveryLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticV402RecoveryLaunchError(message)


def command(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=check,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def test_files(patterns: tuple[str, ...]) -> list[str]:
    values: set[str] = set()
    for pattern in patterns:
        values.update(
            str(Path(path).relative_to(WORKTREE))
            for path in glob.glob(str(WORKTREE / pattern))
        )
    require(bool(values), "A100 recovery test selection is empty")
    return sorted(values)


def run_suite(label: str, patterns: tuple[str, ...]) -> dict[str, Any]:
    files = test_files(patterns)
    result = command([str(PYTHON), "-m", "pytest", "-q", *files], check=False)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    print(output, flush=True)
    match = re.search(r"(?m)(\d+) passed(?:,| in)", output)
    passed = int(match.group(1)) if match else 0
    require(result.returncode == 0 and passed > 0, f"A100 {label} tests failed")
    return {"label": label, "passed": passed, "failed": 0, "file_count": len(files)}


def build_recovery_config(
    base: Mapping[str, Any], recovery_output_root: Path
) -> dict[str, Any]:
    required = [str(row["run_id"]) for row in base["required_screen_runs"]]
    require(required == list(RUN_IDS), "base Critic screen arm order changed")
    require(
        str(base["output_root"]) == str(OLD_OUTPUT_ROOT),
        "base Critic screen output root changed",
    )
    recovery = copy.deepcopy(dict(base))
    recovery["output_root"] = str(recovery_output_root)
    recovery["screen_gate_output"] = str(recovery_output_root / "screen_gate.json")
    assert_scientific_config_unchanged(base, recovery)
    return recovery


def assert_scientific_config_unchanged(
    base: Mapping[str, Any], recovery: Mapping[str, Any]
) -> None:
    allowed = {"output_root", "screen_gate_output"}
    base_science = {key: value for key, value in base.items() if key not in allowed}
    recovery_science = {
        key: value for key, value in recovery.items() if key not in allowed
    }
    require(
        base_science == recovery_science,
        "V4.0.2 recovery changed a scientific config field",
    )
    require(
        str(base["output_root"]) != str(recovery["output_root"])
        and str(base["screen_gate_output"])
        != str(recovery["screen_gate_output"]),
        "V4.0.2 recovery did not isolate its output paths",
    )


def gpu5_free_memory_mib() -> int:
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
        values[int(index)] = int(free)
    require(set(values).issuperset(range(6)), "physical GPU inventory 0–5 is incomplete")
    return values[5]


def validate_prerequisites(current_head: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(PYTHON.is_file(), "formal Python 3.10 is absent")
    require(BASE_CONFIG.is_file() and TRAINER.is_file() and SCHEDULER.is_file(), "recovery code is incomplete")
    require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    require(command(["git", "rev-parse", "HEAD"]).stdout.strip() == current_head, "recovery worktree is not at expected HEAD")
    require(not command(["git", "status", "--porcelain"]).stdout.strip(), "recovery worktree is dirty")
    require(not SINGLE_LAUNCH_MARKER.exists(), "the single V4.0.2 recovery launch is already consumed")

    amendment = read_json(AMENDMENT)
    require(
        amendment.get("status")
        == "PROSPECTIVELY_FROZEN_BEFORE_FAILURE_PAYLOAD_READ_OR_RECOVERY_PARAMETER_UPDATE"
        and amendment.get("decision")
        == "AUTHORIZE_NARROW_CRITIC_TECHNICAL_RECOVERY_ON_GPU5"
        and amendment["recovery_package"].get("required_arms") == list(RUN_IDS)
        and amendment["recovery_package"].get("single_technical_recovery_attempt") is True,
        "V4.0.2 prospective amendment is absent or changed",
    )
    diagnosis = read_json(DIAGNOSIS)
    require(
        diagnosis.get("status")
        == "XEDITCRITIC_V402_FAILURE_DIAGNOSIS_READ_ONCE_COMPLETE"
        and int(diagnosis.get("terminal_failure_payloads_read_count", -1)) == 8
        and int(diagnosis.get("terminal_summary_artifacts_present", -1)) == 0
        and diagnosis.get("valid_validation_performance_summary_present") is False,
        "V4.0.2 technical-recovery diagnosis is not eligible",
    )
    for run_id in RUN_IDS:
        require(
            (OLD_OUTPUT_ROOT / run_id / "failure.json").is_file()
            and not (OLD_OUTPUT_ROOT / run_id / "run_summary.json").exists(),
            f"historical Critic terminal artifact identity changed: {run_id}",
        )

    smoke = read_json(LOCAL_SMOKE_RECEIPT)
    require(
        smoke.get("status")
        == "XEDITCRITIC_V402_RECOVERY_TRAIN_ONLY_CUDA_SMOKE_PASS"
        and int(smoke.get("physical_gpu_index", -1)) == 5
        and int(smoke.get("trainable_parameters", -1)) == 170481957
        and int(smoke.get("effective_batch_size", -1)) == 32
        and int(smoke.get("passes_checked", -1)) == 8
        and int(smoke.get("optimizer_updates_per_pass", -1)) == 2802
        and int(smoke.get("maximum_record_repeats_per_pass", -1)) == 4
        and smoke.get("sampler_target_value_accessed") is False
        and smoke.get("validation_metric_accessed") is False
        and int(smoke.get("development_test_outcome_reads", -1)) == 0
        and int(smoke.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4.0.2 TRAIN-only GPU5 smoke receipt is absent or changed",
    )
    require(Path(str(smoke["remote_terminal_artifact"])).is_file(), "remote GPU5 smoke terminal artifact is absent")
    smoke_head = str(smoke["git_head"])
    changed_scientific = command(
        [
            "git",
            "diff",
            "--name-only",
            f"{smoke_head}..{current_head}",
            "--",
            "core/route2_xeditcritic_training_v4.py",
            "scripts/route_a_v3/train_route2_xeditcritic_v4.py",
            "configs/route_a_v3_route2_xeditcritic_v4_screen_v1.json",
        ]
    ).stdout.strip()
    require(not changed_scientific, "smoke-tested Critic training or scientific config changed")

    preflight = read_json(PREFLIGHT)
    require(
        preflight.get("status") == "XEDITCRITIC_V4_PREFLIGHT_PASS"
        and preflight.get("passed") is True
        and int(preflight.get("selected_physical_batch", -1)) == 32
        and 165000000 <= int(preflight.get("trainable_parameter_count", -1)) <= 175000000
        and 0.0 < float(preflight.get("selected_peak_allocated_gib", -1)) <= 35.0
        and int(preflight.get("development_test_outcome_reads", -1)) == 0
        and int(preflight.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "formal Critic V4 preflight is absent or changed",
    )
    return smoke, preflight


def build_launch_authorization(
    *, current_head: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1",
        "status": "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED",
        "authorized_git_head": current_head,
        "preflight_runner_git_head": str(preflight["git_head"]),
        "cache_experiment_head": CACHE_EXPERIMENT_HEAD,
        "authorized_run_ids": list(RUN_IDS),
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
        "v402_technical_recovery": {
            "single_complete_eight_arm_package": True,
            "physical_gpu_index": 5,
            "old_failure_artifacts_retained": True,
            "setflow_jobs_stopped_modified_or_restarted": False,
            "scientific_config_changed": False,
        },
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(expected_head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None, "expected HEAD is invalid")
    smoke, preflight = validate_prerequisites(expected_head)
    critic_tests = run_suite("critic_v402_recovery_focused", CRITIC_TEST_PATTERNS)
    v332_tests = run_suite("exact_v332", V332_TEST_PATTERNS)
    require(int(v332_tests["passed"]) == 96, "A100 exact V3.3.2 cohort is not 96/96")
    require(not command(["git", "status", "--porcelain"]).stdout.strip(), "A100 tests changed the recovery worktree")

    recovery_output_root = (
        ROOT
        / "experiments/xeditcritic_v4"
        / f"screen_seed_20260907_v402_recovery_runner_{expected_head}"
    )
    runtime_root = (
        ROOT
        / "experiments/xeditcritic_v4"
        / f"v402_recovery_package_runner_{expected_head}"
    )
    log_root = ROOT / "logs/xeditcritic_v4" / f"v402_recovery_runner_{expected_head}"
    authorization_root = ROOT / "authorizations/xeditcritic_v4" / f"v402_recovery_runner_{expected_head}"
    for path, label in (
        (recovery_output_root, "recovery output root"),
        (runtime_root, "recovery runtime root"),
        (authorization_root, "recovery authorization root"),
    ):
        require(not path.exists(), f"{label} already exists")

    base = read_json(BASE_CONFIG)
    recovery_config = build_recovery_config(base, recovery_output_root)
    required_free_bytes = int(smoke["launch_required_free_memory_bytes"])
    free_memory_mib = gpu5_free_memory_mib()
    require(
        free_memory_mib * 1024**2 >= required_free_bytes,
        "physical GPU5 lacks the smoke-measured peak plus 2 GiB",
    )

    authorization_root.mkdir(parents=True)
    config_path = authorization_root / "screen_config.json"
    authorization_path = authorization_root / "launch_authorization.json"
    a100_audit_path = authorization_root / "a100_current_head_tests.json"
    write_atomic(config_path, recovery_config)
    write_atomic(
        authorization_path,
        build_launch_authorization(current_head=expected_head, preflight=preflight),
    )
    write_atomic(
        a100_audit_path,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v402_a100_current_head_tests.v1",
            "status": "XEDITCRITIC_V402_A100_CURRENT_HEAD_TESTS_PASS",
            "git_head": expected_head,
            "worktree": str(WORKTREE),
            "critic_focused": critic_tests,
            "exact_v332": v332_tests,
            "worktree_clean_after": True,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )

    runtime_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    schedule_path = runtime_root / "schedule.json"
    runtime_manifest = runtime_root / "runtime.json"
    jobs = [
        {
            "job_key": f"critic:{run_id}",
            "run_id": run_id,
            "output_directory": str(recovery_output_root / run_id),
            "log_path": str(log_root / f"{run_id}.log"),
            "command": [
                str(PYTHON),
                str(TRAINER),
                "--config",
                str(config_path),
                "--run-id",
                run_id,
                "--physical-gpu-index",
                "5",
                "--launch-authorization",
                str(authorization_path),
            ],
        }
        for run_id in RUN_IDS
    ]
    schedule = {
        "schema_version": "route_a_v3_route2_xeditcritic_v402_recovery_schedule.v1",
        "status": "FROZEN_V402_RECOVERY_SCHEDULE",
        "git_head": expected_head,
        "worktree": str(WORKTREE),
        "physical_gpu_index": 5,
        "gpu5_free_memory_mib_before_launch": free_memory_mib,
        "required_free_memory_bytes": required_free_bytes,
        "required_free_memory_rule": "TRAIN_ONLY_SMOKE_PEAK_PLUS_2_GIB",
        "runtime_manifest": str(runtime_manifest),
        "jobs": jobs,
        "terminal_artifact_payloads_read": 0,
        "active_performance_output_read": False,
        "setflow_jobs_stopped_modified_or_restarted": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(schedule_path, schedule)
    write_exclusive(
        SINGLE_LAUNCH_MARKER,
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v402_recovery_launch_consumed.v1",
            "status": "XEDITCRITIC_V402_SINGLE_RECOVERY_LAUNCH_CONSUMED",
            "git_head": expected_head,
            "schedule_path": str(schedule_path),
            "physical_gpu_index": 5,
            "authorized_run_ids": list(RUN_IDS),
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    scheduler_log = log_root / "scheduler.log"
    stream = scheduler_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(SCHEDULER), "--schedule", str(schedule_path)],
        cwd=WORKTREE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    launch = {
        "schema_version": "route_a_v3_route2_xeditcritic_v402_recovery_launch.v1",
        "status": "XEDITCRITIC_V402_RECOVERY_SCHEDULER_LAUNCHED",
        "git_head": expected_head,
        "scheduler_pid": process.pid,
        "physical_gpu_index": 5,
        "schedule_path": str(schedule_path),
        "runtime_manifest": str(runtime_manifest),
        "scheduler_log": str(scheduler_log),
        "screen_config": str(config_path),
        "launch_authorization": str(authorization_path),
        "a100_test_audit": str(a100_audit_path),
        "old_failure_artifacts_retained": True,
        "setflow_jobs_stopped_modified_or_restarted": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    write_atomic(runtime_root / "launch.json", launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
