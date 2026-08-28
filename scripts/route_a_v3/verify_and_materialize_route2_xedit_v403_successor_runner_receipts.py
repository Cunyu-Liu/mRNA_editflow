#!/usr/bin/env python3
"""Verify the licensed successor runner and publish its two formal receipts.

This one-shot is deliberately limited to repository tests and receipt consumer
validation.  It does not inspect an experiment runtime, probe CUDA, or launch a
training/validation family.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import scripts.route_a_v3.authorize_route2_xeditsetflow_v403_recovered_confirmation as setflow_authorizer
import scripts.route_a_v3.launch_route2_xeditcritic_v403_confirmation_after_cross_root_screen as critic_confirmation
import scripts.route_a_v3.launch_route2_xeditcritic_v403_controls_after_full as critic_controls
import scripts.route_a_v3.launch_route2_xeditsetflow_s1_screen_after_v403_terminal as setflow_s1_screen


PYTHON = Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10")
LICENSED_BRANCH = "route-a-v3-v403-no-vram-gate-20260827"
DIRECT_TEST_MODULE = (
    "test_verify_and_materialize_route2_xedit_v403_successor_runner_receipts.py"
)
V332_LITERAL_GLOB = "tests/route_a_v3/*v332*.py"
TERMINAL_SCHEMA = (
    "route_a_v3_route2_xedit_v403_successor_runner_receipt_materialization.v1"
)
TERMINAL_MATERIALIZED = (
    "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_RECEIPTS_MATERIALIZED"
)
TERMINAL_VALIDATED = (
    "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_RECEIPTS_VALIDATED"
)
TERMINAL_FAILED = (
    "XEDIT_V403_SUCCESSOR_RUNNER_VERIFICATION_RECEIPTS_FAILED"
)


class SuccessorRunnerReceiptError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorRunnerReceiptError(message)


def run_command(
    arguments: Sequence[str], *, cwd: Path = WORKTREE
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def require_exact_clean_pushed_identity(expected_head: str) -> None:
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None,
        "expected Git HEAD must be an exact 40-character lowercase commit",
    )
    commands_and_expected = (
        (("git", "rev-parse", "HEAD"), expected_head, "worktree HEAD changed"),
        (
            ("git", "status", "--porcelain"),
            "",
            "licensed worktree is dirty",
        ),
        (
            ("git", "branch", "--show-current"),
            LICENSED_BRANCH,
            "worktree is not on the licensed branch",
        ),
        (
            ("git", "rev-parse", f"origin/{LICENSED_BRANCH}"),
            expected_head,
            "licensed exact HEAD is not the current origin branch identity",
        ),
    )
    for command_line, expected, failure in commands_and_expected:
        result = run_command(command_line)
        require(
            result.returncode == 0,
            f"Git identity command failed ({shlex.join(command_line)}): "
            f"{result.stderr.strip()}",
        )
        require(result.stdout.strip() == expected, failure)


def require_exact_focused_group_contract() -> tuple[tuple[str, ...], ...]:
    critic_groups = critic_confirmation.FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    setflow_groups = setflow_authorizer.FOCUSED_GROUP_REQUIRED_TEST_MARKERS
    require(
        critic_groups == setflow_groups,
        "strict Critic and SetFlow focused group contracts differ",
    )
    require(
        isinstance(critic_groups, tuple)
        and len(critic_groups)
        == critic_confirmation.FOCUSED_PROCESS_GROUP_COUNT
        == setflow_authorizer.FOCUSED_PROCESS_GROUP_COUNT
        == 8
        and all(
            isinstance(group, tuple)
            and group
            and all(isinstance(marker, str) and marker for marker in group)
            for group in critic_groups
        ),
        "focused group contract is not the exact eight non-empty groups",
    )
    return critic_groups


def focused_group_arguments() -> list[list[str]]:
    groups = require_exact_focused_group_contract()
    commands: list[list[str]] = []
    for index, markers in enumerate(groups):
        modules = [f"tests/route_a_v3/{marker}" for marker in markers]
        if index == len(groups) - 1:
            modules.append(f"tests/route_a_v3/{DIRECT_TEST_MODULE}")
        commands.append(
            [str(PYTHON), "-m", "pytest", "-q", *modules]
        )
    return commands


def parse_pytest_counts(output: str, *, label: str) -> tuple[int, int]:
    passed_matches = re.findall(r"(?<![A-Za-z])(\d+)\s+passed\b", output)
    failed_matches = re.findall(r"(?<![A-Za-z])(\d+)\s+failed\b", output)
    require(passed_matches, f"{label} has no parseable positive pytest PASS count")
    passed = int(passed_matches[-1])
    failed = int(failed_matches[-1]) if failed_matches else 0
    require(passed > 0, f"{label} reports no passing tests")
    return passed, failed


def _terminate_started_processes(processes: Sequence[Any]) -> None:
    for process in processes:
        if getattr(process, "poll", lambda: None)() is None:
            try:
                process.terminate()
            except OSError:
                pass
    for process in processes:
        try:
            process.communicate()
        except OSError:
            pass


def run_focused_groups() -> dict[str, Any]:
    argument_groups = focused_group_arguments()
    processes: list[subprocess.Popen[str]] = []
    try:
        for arguments in argument_groups:
            processes.append(
                subprocess.Popen(
                    arguments,
                    cwd=WORKTREE,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
    except Exception:
        _terminate_started_processes(processes)
        raise

    group_counts: list[int] = []
    failures: list[str] = []
    for index, (arguments, process) in enumerate(
        zip(argument_groups, processes, strict=True), start=1
    ):
        stdout, stderr = process.communicate()
        combined = "\n".join((stdout or "", stderr or ""))
        try:
            passed, failed = parse_pytest_counts(
                combined, label=f"focused group {index}"
            )
        except SuccessorRunnerReceiptError as error:
            failures.append(str(error))
            continue
        group_counts.append(passed)
        if process.returncode != 0 or failed != 0:
            failures.append(
                f"focused group {index} failed: returncode={process.returncode}, "
                f"passed={passed}, failed={failed}; output={combined[-4000:]}"
            )
    require(not failures, "; ".join(failures))
    require(
        len(group_counts) == 8 and all(value > 0 for value in group_counts),
        "focused cohort did not return eight positive group PASS counts",
    )
    return {
        "command": [shlex.join(arguments) for arguments in argument_groups],
        "isolated_process_groups": True,
        "group_passed_counts": group_counts,
        "passed_count": sum(group_counts),
        "failed_count": 0,
        "passed": True,
    }


def run_v332_cohort() -> dict[str, Any]:
    matched = sorted(WORKTREE.glob(V332_LITERAL_GLOB))
    require(matched, f"literal V3.3.2 cohort matched no tests: {V332_LITERAL_GLOB}")
    relative = [str(path.relative_to(WORKTREE)) for path in matched]
    execution_arguments = [
        str(PYTHON),
        "-m",
        "pytest",
        "-q",
        *relative,
    ]
    result = run_command(execution_arguments)
    combined = "\n".join((result.stdout or "", result.stderr or ""))
    passed, failed = parse_pytest_counts(combined, label="V3.3.2 cohort")
    require(
        result.returncode == 0 and passed == 96 and failed == 0,
        "V3.3.2 cohort is not exact 96 PASS / 0 FAIL: "
        f"returncode={result.returncode}, passed={passed}, failed={failed}; "
        f"output={combined[-4000:]}",
    )
    return {
        "command": [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            V332_LITERAL_GLOB,
        ],
        "passed_count": passed,
        "failed_count": failed,
        "passed": True,
    }


def canonical_receipt_paths(expected_head: str) -> tuple[Path, Path]:
    critic_path = critic_confirmation.runner_verification_receipt_path(
        expected_head
    )
    s1_shared, s1_setflow = setflow_s1_screen.expected_receipt_paths(
        expected_head
    )
    require(
        critic_path == s1_shared,
        "Critic and S1 consumers disagree on the canonical shared receipt path",
    )
    return critic_path, s1_setflow


def require_fresh_receipt_targets(paths: Sequence[Path]) -> None:
    for path in paths:
        partial = path.with_suffix(path.suffix + ".partial")
        require(not path.exists(), f"receipt already exists: {path}")
        require(not partial.exists(), f"partial receipt already exists: {partial}")


def build_receipts(
    expected_head: str,
    *,
    focused_tests: Mapping[str, Any],
    v332_tests: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = {
        "runner_git_head": expected_head,
        "worktree_clean": True,
        "licensed_branch": LICENSED_BRANCH,
        "origin_branch_git_head": expected_head,
        "focused_tests": copy.deepcopy(dict(focused_tests)),
        "v332_tests": copy.deepcopy(dict(v332_tests)),
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    shared = {
        "schema_version": critic_confirmation.RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": critic_confirmation.RUNNER_VERIFICATION_RECEIPT_PASS,
        **copy.deepcopy(evidence),
    }
    setflow = {
        "schema_version": setflow_authorizer.RUNNER_VERIFICATION_RECEIPT_SCHEMA,
        "status": setflow_authorizer.RUNNER_VERIFICATION_RECEIPT_PASS,
        **copy.deepcopy(evidence),
    }
    return shared, setflow


def validate_in_memory_receipts(
    expected_head: str,
    shared_path: Path,
    setflow_path: Path,
    shared: Mapping[str, Any],
    setflow: Mapping[str, Any],
) -> None:
    require_exact_focused_group_contract()
    critic_confirmation.validate_runner_verification_receipt(
        shared,
        runner_head=expected_head,
        receipt_path=shared_path,
    )
    setflow_authorizer.require_runner_verification_receipt_v403(
        setflow,
        current_runner_head=expected_head,
    )
    shared_evidence = {
        key: value
        for key, value in shared.items()
        if key not in {"schema_version", "status"}
    }
    setflow_evidence = {
        key: value
        for key, value in setflow.items()
        if key not in {"schema_version", "status"}
    }
    require(
        shared_evidence == setflow_evidence,
        "shared and SetFlow receipts do not contain identical actual evidence",
    )
    require(
        shared_path == canonical_receipt_paths(expected_head)[0]
        and setflow_path == canonical_receipt_paths(expected_head)[1],
        "receipt path is not canonical",
    )


def _write_receipts_atomically(
    items: Sequence[tuple[Path, Mapping[str, Any]]]
) -> None:
    require_fresh_receipt_targets([path for path, _ in items])
    partials: list[Path] = []
    published: list[Path] = []
    try:
        for path, payload in items:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".partial")
            partial.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            partials.append(partial)
        for (path, _), partial in zip(items, partials, strict=True):
            os.replace(partial, path)
            published.append(path)
    except Exception:
        for partial in partials:
            if partial.exists():
                partial.unlink()
        for path in published:
            if path.exists():
                path.unlink()
        raise


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"receipt is not a JSON object: {path}")
    return payload


def validate_production_consumers(
    expected_head: str, shared_path: Path, setflow_path: Path
) -> dict[str, Any]:
    shared = read_json_object(shared_path)
    setflow = read_json_object(setflow_path)
    validate_in_memory_receipts(
        expected_head,
        shared_path,
        setflow_path,
        shared,
        setflow,
    )
    critic_controls.validate_historical_full_terminal_audit()
    critic_controls.validate_training_source(expected_head)
    s1_config = setflow_s1_screen.read_json(setflow_s1_screen.CONFIG)
    setflow_s1_screen.validate_config(s1_config)
    s1_repo_fact_audits = setflow_s1_screen.validate_repo_fact_audits(s1_config)
    old_s1_terminal_invalidation = (
        setflow_s1_screen.consume_old_s1_terminal_invalidation_receipt(
            setflow_s1_screen.OLD_S1_TERMINAL_INVALIDATION_RECEIPT
        )
    )
    s1 = setflow_s1_screen.consume_receipts(
        expected_head, shared_path, setflow_path
    )
    return {
        "critic_historical_full_terminal_audit_validation": True,
        "critic_training_source_validation": True,
        "critic_controls_receipt_validation": True,
        "s1_repo_fact_audit_validation": sorted(s1_repo_fact_audits),
        "s1_old_terminal_invalidation_validation": old_s1_terminal_invalidation,
        "s1_pre_gpu_receipt_validation": s1,
        "gpu_launcher_invoked": False,
        "runtime_read": False,
    }


def verify_or_validate(
    expected_head: str, *, validate_receipts_only: bool = False
) -> dict[str, Any]:
    require_exact_clean_pushed_identity(expected_head)
    require_exact_focused_group_contract()
    shared_path, setflow_path = canonical_receipt_paths(expected_head)

    if validate_receipts_only:
        for path in (shared_path, setflow_path):
            require(path.is_file(), f"receipt is absent: {path}")
            require(
                not path.with_suffix(path.suffix + ".partial").exists(),
                f"partial receipt conflicts with terminal receipt: {path}",
            )
        consumers = validate_production_consumers(
            expected_head, shared_path, setflow_path
        )
        return {
            "schema_version": TERMINAL_SCHEMA,
            "status": TERMINAL_VALIDATED,
            "mode": "VALIDATE_RECEIPTS_ONLY",
            "runner_git_head": expected_head,
            "shared_receipt": str(shared_path),
            "setflow_receipt": str(setflow_path),
            "consumers": consumers,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }

    require(PYTHON.is_file(), f"formal repository Python is absent: {PYTHON}")
    require_fresh_receipt_targets((shared_path, setflow_path))
    focused = run_focused_groups()
    v332 = run_v332_cohort()
    require_exact_clean_pushed_identity(expected_head)
    shared, setflow = build_receipts(
        expected_head, focused_tests=focused, v332_tests=v332
    )
    validate_in_memory_receipts(
        expected_head,
        shared_path,
        setflow_path,
        shared,
        setflow,
    )
    _write_receipts_atomically(
        ((shared_path, shared), (setflow_path, setflow))
    )
    # The receipts have already passed both strict in-memory validators.  Keep
    # them immutable if a later production-consumer preflight fails so the
    # operator can use --validate-receipts-only after correcting that external
    # condition instead of rerunning all eight groups and V3.3.2.
    consumers = validate_production_consumers(
        expected_head, shared_path, setflow_path
    )
    return {
        "schema_version": TERMINAL_SCHEMA,
        "status": TERMINAL_MATERIALIZED,
        "mode": "VERIFY_AND_MATERIALIZE",
        "runner_git_head": expected_head,
        "focused_passed_count": focused["passed_count"],
        "focused_group_passed_counts": focused["group_passed_counts"],
        "v332_passed_count": v332["passed_count"],
        "shared_receipt": str(shared_path),
        "setflow_receipt": str(setflow_path),
        "consumers": consumers,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--expected-head", required=True)
    result.add_argument(
        "--validate-receipts-only",
        action="store_true",
        help="validate existing canonical receipts without running pytest",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        terminal = verify_or_validate(
            arguments.expected_head,
            validate_receipts_only=arguments.validate_receipts_only,
        )
    except Exception as error:
        canonical_receipts_present: list[str] = []
        if re.fullmatch(r"[0-9a-f]{40}", arguments.expected_head):
            try:
                canonical_receipts_present = [
                    str(path)
                    for path in canonical_receipt_paths(arguments.expected_head)
                    if path.exists()
                ]
            except Exception:
                canonical_receipts_present = []
        print(
            json.dumps(
                {
                    "schema_version": TERMINAL_SCHEMA,
                    "status": TERMINAL_FAILED,
                    "runner_git_head": arguments.expected_head,
                    "validate_receipts_only": arguments.validate_receipts_only,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "canonical_pass_receipts_present": (
                        canonical_receipts_present
                    ),
                    "gpu_launcher_invoked": False,
                    "runtime_read": False,
                    "development_test_outcome_reads": 0,
                    "new_final_evaluation_outcome_reads": 0,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
