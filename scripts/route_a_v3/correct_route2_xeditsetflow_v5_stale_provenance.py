#!/usr/bin/env python3
"""Correct stale Git provenance in terminal SetFlow V5 training attempt records.

Root cause (2026-09-02 screen failure at VALIDATION):
  The 2026-09-01 screen launch aborted once at 13:18 CST from worktree HEAD
  17c31468fb9d8f7c429c2440621026ac28946ffd (GPU6/7 MIG capacity problem, fixed
  by commit a5cdeccfa981bcc07d930f3b0ea7236344bbb5d6 at 13:39:43).  The
  scheduler self-cleaned the staging output directories and relaunched at
  ~13:42 from the authorized HEAD a5cdeccf..., and both launches share the
  deterministic attempt_id ``xeditsetflow_v5_{run_id}_seed{training_seed}``.
  ``record_training_attempt`` preserves ``started_at``/``code_commit`` from
  the first ledger row it sees, so the terminal ``training_attempt.json`` of
  every arm inherited the ABORTED attempt's ``code_commit`` (17c31468...) and
  ``started_at`` (13:18) even though the checkpoints were trained from the
  authorized HEAD a5cdeccf... starting ~13:42.

  Evidence that the checkpoints were trained from a5cdeccf...:
    * training_config.json (written by the training process after its
      launch-time authorization check) records authorized_git_head=a5cdeccf...
    * the launch authorization file records authorized_git_head=a5cdeccf...
      authorized_at 2026-09-01T13:40:16+0800
    * per arm, completed_at - wall_time_seconds lands at 13:42-13:44 CST,
      after the a5cdeccf commit and authorization, and disagrees with the
      preserved started_at of 13:18 (internally inconsistent record).

This correction makes each attempt record internally consistent:
  code_commit := training_config.json:authorized_git_head (the code that ran)
  started_at  := completed_at - wall_time_seconds (the run's true start)
and applies the same fix to the shared ledger CSV rows, then moves the twelve
stale ``pass_*.failed.json`` artifacts (failures of the provenance check this
correction resolves) into a subfolder so any later failure can record
cleanly.  Everything is audited in provenance_correction_*.json at the screen
root.  Idempotent: arms already carrying the authorized head are skipped.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

STALE_COMMIT = "17c31468fb9d8f7c429c2440621026ac28946ffd"
ARMS = ("b_fix1", "b_fix2", "b_fix3", "b_arch1")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
TS_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class CorrectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(str(value), TS_FORMAT)


def _format_ts(value: datetime) -> str:
    return value.strftime(TS_FORMAT)


def _git_show_time(worktree: Path, commit: str) -> str:
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ci", commit],
        cwd=worktree, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def correct_arm(
    screen_root: Path,
    arm: str,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    directory = screen_root / arm
    training_config = _read_json(directory / "training_config.json")
    attempt = _read_json(directory / "training_attempt.json")
    authorized_head = str(training_config.get("authorized_git_head", ""))
    _require(
        COMMIT_RE.fullmatch(authorized_head) is not None,
        f"authorized_git_head is not a commit: {arm}",
    )
    current_commit = str(attempt.get("code_commit", ""))
    if current_commit == authorized_head:
        # Already corrected (idempotent rerun).
        return None
    _require(
        current_commit == STALE_COMMIT,
        f"code_commit is neither the authorized head nor the known stale "
        f"commit {STALE_COMMIT}: {arm} has {current_commit}",
    )
    _require(
        str(attempt.get("status")) == "COMPLETED",
        f"attempt is not COMPLETED: {arm}",
    )
    started_old = str(attempt.get("started_at", ""))
    completed = _parse_ts(attempt["completed_at"])
    wall = float(attempt["wall_time_seconds"])
    started_new = completed - timedelta(seconds=wall)
    drift = abs((completed - _parse_ts(started_old) - timedelta(seconds=wall)).total_seconds())
    _require(
        drift > 60.0,
        f"started_at already agrees with completed_at-wall_time: {arm}",
    )
    corrected = dict(attempt)
    corrected["code_commit"] = authorized_head
    corrected["started_at"] = _format_ts(started_new)
    if not dry_run:
        partial = directory / "training_attempt.json.partial"
        partial.write_text(
            json.dumps(corrected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, directory / "training_attempt.json")
    return {
        "arm": arm,
        "authorized_git_head": authorized_head,
        "code_commit_before": current_commit,
        "code_commit_after": authorized_head,
        "started_at_before": started_old,
        "started_at_after": corrected["started_at"],
        "started_at_drift_seconds_resolved": drift,
        "completed_at": str(attempt["completed_at"]),
        "wall_time_seconds": wall,
    }


def correct_ledger(
    ledger_path: Path,
    fixes: Mapping[str, Mapping[str, str]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    changed: dict[str, dict[str, str]] = {}
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with ledger_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
        _require("code_commit" in columns and "started_at" in columns, "ledger columns missing")
        for row in rows:
            attempt_id = str(row.get("attempt_id", ""))
            for arm, fix in fixes.items():
                expected_id = f"xeditsetflow_v5_{arm}_seed20260915"
                if attempt_id != expected_id:
                    continue
                if str(row.get("code_commit", "")) != STALE_COMMIT:
                    continue
                row["code_commit"] = fix["code_commit_after"]
                row["started_at"] = fix["started_at_after"]
                changed[arm] = {
                    "attempt_id": attempt_id,
                    "code_commit_after": fix["code_commit_after"],
                    "started_at_after": fix["started_at_after"],
                }
        if not dry_run:
            partial = ledger_path.with_suffix(ledger_path.suffix + ".partial")
            with partial.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(partial, ledger_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {"ledger_rows_corrected": sorted(changed), "details": changed}


def archive_stale_failures(
    validation_root: Path,
    *,
    dry_run: bool,
) -> list[str]:
    archive = validation_root / "attempt1_stale_provenance_failures_20260902"
    moved: list[str] = []
    for arm in ARMS:
        arm_dir = validation_root / arm
        if not arm_dir.is_dir():
            continue
        for failed in sorted(arm_dir.glob("pass_*.failed.json")):
            if not dry_run:
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(failed), str(archive / f"{arm}_{failed.name}"))
            moved.append(f"{arm}/{failed.name}")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    screen_root = arguments.screen_root.resolve()
    dry_run = bool(arguments.dry_run)

    fixes: dict[str, dict[str, str]] = {}
    arm_reports: list[dict[str, Any]] = []
    for arm in ARMS:
        report = correct_arm(screen_root, arm, dry_run=dry_run)
        if report is None:
            print(f"skip {arm}: already carries the authorized head")
            continue
        arm_reports.append(report)
        fixes[arm] = {
            "code_commit_after": report["code_commit_after"],
            "started_at_after": report["started_at_after"],
        }
    _require(fixes, "nothing to correct (all arms already carry the authorized head)")

    ledger_report = correct_ledger(arguments.ledger.resolve(), fixes, dry_run=dry_run)
    _require(
        set(ledger_report["ledger_rows_corrected"]) == set(fixes),
        f"ledger correction incomplete: {ledger_report['ledger_rows_corrected']}",
    )
    moved = archive_stale_failures(
        screen_root / "outcome_free_validation_generation", dry_run=dry_run
    )

    worktree = Path(__file__).resolve().parents[2]
    audit = {
        "schema_version": "route_a_v3_route2_xeditsetflow_v5_provenance_correction.v1",
        "corrected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": (
            "terminal training_attempt.json inherited code_commit/started_at from "
            "an aborted 2026-09-01T13:18CST pre-fix launch (worktree HEAD "
            f"{STALE_COMMIT}) because record_training_attempt preserved the first "
            "ledger row for the shared deterministic attempt_id; the checkpoints "
            "were trained from the authorized HEAD after relaunch at ~13:42CST"
        ),
        "evidence": {
            "authorized_git_head": fixes["b_fix1"]["code_commit_after"],
            "authorization_authorized_at": "2026-09-01T13:40:16+0800",
            "commit_a5cdecc_time": _git_show_time(worktree, fixes["b_fix1"]["code_commit_after"]),
            "commit_17c3146_time": _git_show_time(worktree, STALE_COMMIT),
            "internal_inconsistency_resolved": "started_at + wall_time != completed_at before correction",
        },
        "arms": arm_reports,
        "ledger": ledger_report,
        "archived_stale_validation_failures": moved,
        "dry_run": dry_run,
    }
    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        audit_path = screen_root / f"provenance_correction_{stamp}.json"
        partial = audit_path.with_suffix(audit_path.suffix + ".partial")
        partial.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, audit_path)
        print(json.dumps({"status": "CORRECTED", "audit": str(audit_path)}, sort_keys=True))
    else:
        print(json.dumps({"status": "DRY_RUN", "audit_payload": audit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
