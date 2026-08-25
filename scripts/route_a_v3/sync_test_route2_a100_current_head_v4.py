#!/usr/bin/env python3
"""Sync the post-C3 A100 worktree and record exact current-HEAD tests."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path


WORKTREE = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/"
    "route_a_v3_route2_method_repair_20260817"
)
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
BRANCH = "route-a-v3-route2-method-repair-20260817"
EXPERIMENT_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "xeditcritic_v3/screen_seed_20260830"
)
RUN_IDS = (
    "c3",
    "c3_source_only",
    "c3_edit_metadata_only",
    "c3_no_candidate_sequence",
    "c3_candidate_bundle_permutation",
)
OLD_LAUNCH_PROCESSES = (
    (2443206, "c3"),
    (2443207, "c3_source_only"),
    (2443208, "c3_edit_metadata_only"),
    (2529140, "c3_no_candidate_sequence"),
    (2592082, "c3_candidate_bundle_permutation"),
)
OLD_TRAINER_ENTRYPOINT = "train_route2_xeditcritic_v3_c3_online.py"
CRITIC_TEST_PATTERNS = (
    "tests/route_a_v3/*xeditcritic_v4*.py",
    "tests/route_a_v3/test_adjudicate_route2_xeditcritic_v3_c3_v4_reference.py",
    "tests/route_a_v3/test_sync_test_route2_a100_current_head_v4.py",
    "tests/route_a_v3/test_route2_xeditcritic_batch_v4.py",
    "tests/route_a_v3/test_route2_bottom_encoder_chunk_cache_v4.py",
    "tests/route_a_v3/test_route2_mrnabert_bottom_six_encoder_v4.py",
    "tests/route_a_v3/test_route2_xedit_v4_interfaces.py",
    "tests/route_a_v3/test_authorize_route2_xedit_v4_screen_stages.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_cache_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_caches_after_a100_sync.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_preflights_after_caches.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_screen_package_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_screens_after_preflights.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_postscreen_adjudication_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_postscreen_after_screen_terminal.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_confirmation_training_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py",
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness.py",
)
SETFLOW_TEST_PATTERNS = (
    "tests/route_a_v3/*xeditsetflow_v4*.py",
    "tests/route_a_v3/test_route2_xeditsetflow_training_v4.py",
    "tests/route_a_v3/test_route2_source_token_cache_v3.py",
    "tests/route_a_v3/test_route2_xedit_v4_interfaces.py",
    "tests/route_a_v3/test_authorize_route2_xedit_v4_screen_stages.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_cache_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_caches_after_a100_sync.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_preflights_after_caches.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_screen_package_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_screens_after_preflights.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_postscreen_adjudication_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_postscreen_after_screen_terminal.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_confirmation_training_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_confirmation_training_after_screen_pass.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_confirmation_posttraining_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_confirmation_posttraining_after_terminal.py",
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_guidance_authorization_after_dual_readiness.py",
)
V332_TEST_PATTERNS = ("tests/route_a_v3/*v332*.py",)


class SyncTestError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SyncTestError(message)


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


def git(*arguments: str) -> str:
    return command(["git", *arguments]).stdout.strip()


def command_is_registered_old_c3(arguments: list[str], *, run_id: str) -> bool:
    return any(OLD_TRAINER_ENTRYPOINT in value for value in arguments) and any(
        arguments[index] == "--run-id"
        and index + 1 < len(arguments)
        and arguments[index + 1] == run_id
        for index in range(len(arguments))
    )


def registered_old_process_active(pid: int, *, run_id: str) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    arguments = [
        value.decode("utf-8", errors="replace")
        for value in raw.split(b"\0")
        if value
    ]
    return command_is_registered_old_c3(arguments, run_id=run_id)


def exact_terminal_package() -> bool:
    return all(
        (EXPERIMENT_ROOT / run_id / "run_summary.json").exists()
        != (EXPERIMENT_ROOT / run_id / "failure.json").exists()
        for run_id in RUN_IDS
    )


def test_files(patterns: tuple[str, ...]) -> list[str]:
    values: set[str] = set()
    for pattern in patterns:
        values.update(
            str(Path(path).relative_to(WORKTREE))
            for path in glob.glob(str(WORKTREE / pattern))
        )
    require(bool(values), "A100 focused test file selection is empty")
    return sorted(values)


def run_suite(label: str, patterns: tuple[str, ...]) -> dict[str, object]:
    files = test_files(patterns)
    result = command([str(PYTHON), "-m", "pytest", "-q", *files], check=False)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    print(output, flush=True)
    match = re.search(r"(?m)(\d+) passed(?:,| in)", output)
    passed = int(match.group(1)) if match else 0
    require(result.returncode == 0 and passed > 0, f"A100 {label} tests failed")
    return {
        "label": label,
        "passed": passed,
        "failed": 0,
        "file_count": len(files),
    }


def run(expected_head: str) -> dict[str, object]:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected HEAD is invalid",
    )
    require(WORKTREE.is_dir() and PYTHON.is_file(), "A100 worktree or Python is absent")
    require(git("branch", "--show-current") == BRANCH, "A100 branch identity changed")
    require(exact_terminal_package(), "five-run C3 terminal package is incomplete")
    require(
        (EXPERIMENT_ROOT / "c3_v4_reference_read_once.json").is_file(),
        "C3 read-once reference is absent before A100 sync",
    )
    active = [
        {"pid": pid, "run_id": run_id}
        for pid, run_id in OLD_LAUNCH_PROCESSES
        if registered_old_process_active(pid, run_id=run_id)
    ]
    require(not active, f"old launch-head processes remain active: {active}")
    status_before = git("status", "--porcelain")
    require(not status_before, "A100 worktree is dirty before current-HEAD sync")
    head_before = git("rev-parse", "HEAD")
    command(["git", "fetch", "origin", BRANCH])
    command(["git", "pull", "--ff-only", "origin", BRANCH])
    head_after = git("rev-parse", "HEAD")
    require(head_after == expected_head, "A100 sync did not reach the exact requested HEAD")
    require(not git("status", "--porcelain"), "A100 worktree is dirty after sync")

    critic = run_suite("critic_v4_focused", CRITIC_TEST_PATTERNS)
    setflow = run_suite("setflow_v4_focused", SETFLOW_TEST_PATTERNS)
    v332 = run_suite("exact_v332", V332_TEST_PATTERNS)
    require(v332["passed"] == 96, "A100 exact V3.3.2 cohort is not 96/96")
    require(not git("status", "--porcelain"), "A100 tests changed the worktree")

    result = {
        "schema_version": "route_a_v3_route2_a100_current_head_sync_tests_v4.v1",
        "status": "A100_CURRENT_HEAD_SYNCED_AND_V4_TESTS_PASS",
        "repository_sync": {
            "remote_worktree": str(WORKTREE),
            "branch": BRANCH,
            "upstream": f"origin/{BRANCH}",
            "head_before": head_before,
            "head_after": head_after,
            "old_launch_jobs_active_before_sync": False,
            "remote_worktree_clean_before": True,
            "remote_worktree_clean_after": True,
            "sync_method": "git pull --ff-only",
            "shared_history_rewritten": False,
        },
        "a100_current_head_verification": {
            "python": str(PYTHON),
            "verified_git_head": head_after,
            "critic_focused_total_passed": critic["passed"],
            "critic_focused_failed": critic["failed"],
            "setflow_focused_passed": setflow["passed"],
            "setflow_focused_failed": setflow["failed"],
            "exact_v332_passed": v332["passed"],
            "exact_v332_failed": v332["failed"],
            "test_file_counts": {
                "critic": critic["file_count"],
                "setflow": setflow["file_count"],
                "v332": v332["file_count"],
            },
            "sequential_execution": True,
        },
        "protected_data": {
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        },
    }
    audit_root = Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/"
        "a100_current_head_v4"
    )
    audit_root.mkdir(parents=True, exist_ok=True)
    output = audit_root / f"sync_tests_{head_after}.json"
    require(not output.exists(), f"A100 current-HEAD audit already exists: {output}")
    partial = output.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    result["audit_path"] = str(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.expected_head), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
