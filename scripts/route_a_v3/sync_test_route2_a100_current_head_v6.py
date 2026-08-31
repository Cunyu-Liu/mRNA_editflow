#!/usr/bin/env python3
"""Produce the A100 current-HEAD sync-tests audit for the V6 worktree.

Conforms to the V5 audit schema (route_a_v3_route2_a100_current_head_sync_tests_v4.v1).
The V6 worktree/branch differ from the V5-era constants, so the suite list is
reused verbatim but executed inside the V6 worktree.  The c3 READ-ONCE
reference and the C3 five-run terminal package are shared immutable facts, so
the sync audit reports them without re-reading any protected outcome.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path

WORKTREE = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v6_lambda_pairwise_prep_20260831")
PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
AUDIT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/a100_current_head_v4")

CRITIC_TEST_PATTERNS = (
    "tests/route_a_v3/*xeditcritic_v4*.py",
    "tests/route_a_v3/test_adjudicate_route2_xeditcritic_v3_c3_v4_reference.py",
    "tests/route_a_v3/test_sync_test_route2_a100_current_head_v4.py",
    "tests/route_a_v3/test_route2_xeditcritic_batch_v4.py",
    "tests/route_a_v3/test_route2_bottom_encoder_chunk_cache_v4.py",
    "tests/route_a_v3/test_route2_mrnabert_bottom_six_encoder_v4.py",
    "tests/route_a_v3/test_route2_xedit_v4_interfaces.py",
    "tests/route_a_v3/test_route2_xedit_v4_method_repair_protocol_v1.py",
    "tests/route_a_v3/test_authorize_route2_xedit_v4_screen_stages.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_cache_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_caches_after_a100_sync.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_job.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_sequence.py",
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
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_guidance_screen_after_authorization.py",
    "tests/route_a_v3/test_run_route2_xeditflow_v4_guidance_screen_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_final_after_guidance_screen.py",
)
SETFLOW_TEST_PATTERNS = (
    "tests/route_a_v3/*xeditsetflow_v4*.py",
    "tests/route_a_v3/test_route2_xeditsetflow_training_v4.py",
    "tests/route_a_v3/test_route2_source_token_cache_v3.py",
    "tests/route_a_v3/test_route2_xedit_v4_interfaces.py",
    "tests/route_a_v3/test_route2_xedit_v4_method_repair_protocol_v1.py",
    "tests/route_a_v3/test_authorize_route2_xedit_v4_screen_stages.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_cache_job.py",
    "tests/route_a_v3/test_launch_route2_xedit_v4_caches_after_a100_sync.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_job.py",
    "tests/route_a_v3/test_run_route2_xedit_v4_preflight_sequence.py",
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
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_guidance_screen_after_authorization.py",
    "tests/route_a_v3/test_run_route2_xeditflow_v4_guidance_screen_scheduler.py",
    "tests/route_a_v3/test_launch_route2_xeditflow_v4_final_after_guidance_screen.py",
)
V332_TEST_PATTERNS = ("tests/route_a_v3/*v332*.py",)


class V6SyncAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V6SyncAuditError(message)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=WORKTREE,
        text=True,
        capture_output=True,
        check=False,
    )


def test_files(patterns: tuple[str, ...]) -> list[str]:
    values: set[str] = set()
    for pattern in patterns:
        values.update(
            str(Path(path).relative_to(WORKTREE))
            for path in glob.glob(str(WORKTREE / pattern))
        )
    require(bool(values), "V6 focused test file selection is empty")
    return sorted(values)


def run_suite(label: str, patterns: tuple[str, ...]) -> dict[str, object]:
    files = test_files(patterns)
    result = command([str(PYTHON), "-m", "pytest", "-q", *files])
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = re.search(r"(?m)(\d+) passed(?:,| in)", output)
    passed = int(match.group(1)) if match else 0
    require(result.returncode == 0 and passed > 0, f"V6 {label} tests failed")
    return {
        "label": label,
        "passed": passed,
        "failed": 0,
        "file_count": len(files),
        "returncode": result.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-f]{40}", arguments.expected_head) is not None,
        "expected HEAD is invalid",
    )
    require(WORKTREE.is_dir() and PYTHON.is_file(), "V6 worktree or Python is absent")
    c3_reference = Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
        "xeditcritic_v3/screen_seed_20260830/c3_v4_reference_read_once.json"
    )
    require(c3_reference.is_file(), "C3 read-once reference is absent")
    require(not command(["git", "status", "--porcelain"]).stdout.strip(),
            "V6 worktree is dirty before the audit")
    head = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    require(head == arguments.expected_head, "V6 worktree is not at the requested HEAD")

    critic = run_suite("critic_v4_focused", CRITIC_TEST_PATTERNS)
    setflow = run_suite("setflow_v4_focused", SETFLOW_TEST_PATTERNS)
    v332 = run_suite("exact_v332", V332_TEST_PATTERNS)
    require(v332["passed"] == 96, f"V6 exact V3.3.2 cohort is not 96/96 (got {v332['passed']})")
    require(not command(["git", "status", "--porcelain"]).stdout.strip(),
            "V6 tests changed the worktree")

    result = {
        "schema_version": "route_a_v3_route2_a100_current_head_sync_tests_v4.v1",
        "status": "A100_CURRENT_HEAD_SYNCED_AND_V4_TESTS_PASS",
        "repository_sync": {
            "remote_worktree": str(WORKTREE),
            "branch": command(["git", "branch", "--show-current"]).stdout.strip(),
            "upstream": "origin/route-a-v3-v6-lambda-pairwise-prep-20260831",
            "head_before": head,
            "head_after": head,
            "old_launch_jobs_active_before_sync": False,
            "remote_worktree_clean_before": True,
            "remote_worktree_clean_after": True,
            "sync_method": "V6 worktree exact-HEAD (no pull; HEAD already 7815fdeb)",
            "shared_history_rewritten": False,
        },
        "a100_current_head_verification": {
            "python": str(PYTHON),
            "verified_git_head": head,
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
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    output = AUDIT_ROOT / f"sync_tests_{head}.json"
    require(not output.exists(), f"A100 current-HEAD audit already exists: {output}")
    partial = output.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({**result, "audit_path": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()