from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xeditcritic_v4_refit_scheduler as scheduler


def _writer(path: Path | None, payload: dict[str, object], exit_code: int) -> list[str]:
    action = ""
    if path is not None:
        action = (
            f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
            f"p.write_text(json.dumps({payload!r})+'\\n');"
        )
    return [
        sys.executable,
        "-c",
        f"import json,pathlib,sys;{action}sys.exit({exit_code})",
    ]


def test_refit_scheduler_preserves_terminals_and_authorizes_loso(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    failure = tmp_path / "run/failure.json"
    manifest = tmp_path / "refit_manifest.json"
    adjudication_failure = tmp_path / "refit_adjudication.failed.json"
    schedule = {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "refit:20260908:v4_full",
                        "seed": 20260908,
                        "summary_path": str(summary),
                        "failure_path": str(failure),
                        "log_path": str(tmp_path / "refit.log"),
                        "command": _writer(summary, {}, 0),
                    }
                ],
            }
        ],
        "adjudication": {
            "manifest_path": str(manifest),
            "failure_path": str(adjudication_failure),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(
                manifest,
                {
                    "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
                    "loso_authorized": True,
                },
                0,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED"
    assert payload["jobs"]["refit:20260908:v4_full"]["terminal_artifact_kind"] == "SUMMARY"
    assert payload["jobs"]["refit:20260908:v4_full"]["return_code"] == 0
    assert payload["adjudication"]["terminal_artifact_kind"] == "SUMMARY"
    assert payload["adjudication"]["return_code"] == 0
    assert payload["adjudication"]["loso_authorized"] is True
    assert not failure.exists()
    assert not adjudication_failure.exists()


def test_refit_scheduler_closes_missing_job_and_stops_before_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    failure = tmp_path / "run/failure.json"
    manifest = tmp_path / "refit_manifest.json"
    schedule = {
        "git_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "refit:20260908:v4_full",
                        "seed": 20260908,
                        "summary_path": str(summary),
                        "failure_path": str(failure),
                        "log_path": str(tmp_path / "refit.log"),
                        "command": _writer(None, {}, 3),
                    }
                ],
            }
        ],
        "adjudication": {
            "manifest_path": str(manifest),
            "failure_path": str(tmp_path / "adjudication.failed.json"),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(
                manifest,
                {
                    "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_NO_GO",
                    "loso_authorized": False,
                },
                0,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["jobs"]["refit:20260908:v4_full"]["terminal_artifact_kind"] == "FAILURE"
    assert failure.is_file()
    assert payload["jobs"]["refit:20260908:v4_full"]["status"] == "TECHNICAL_FAILURE"
    assert payload["adjudication"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert not manifest.exists()


def test_refit_nonzero_with_summary_is_technical_and_stops_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    first_summary = tmp_path / "first/run_summary.json"
    second_summary = tmp_path / "second/run_summary.json"
    adjudication = tmp_path / "adjudication.json"
    jobs = []
    for key, seed, summary, exit_code in (
        ("refit:20260908:v4_full", 20260908, first_summary, 4),
        ("refit:20260909:v4_full", 20260909, second_summary, 0),
    ):
        jobs.append(
            {
                "job_key": key,
                "seed": seed,
                "summary_path": str(summary),
                "failure_path": str(summary.parent / "failure.json"),
                "log_path": str(summary.parent / "job.log"),
                "command": _writer(summary, {}, exit_code),
            }
        )
    schedule = {
        "git_head": "c" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [{"physical_gpu_index": 0, "jobs": jobs}],
        "adjudication": {
            "manifest_path": str(adjudication),
            "failure_path": str(tmp_path / "adjudication.failed.json"),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(adjudication, {"loso_authorized": True}, 0),
        },
    }
    scheduler.run(schedule)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["jobs"][jobs[0]["job_key"]]["status"] == "TECHNICAL_FAILURE"
    assert payload["jobs"][jobs[1]["job_key"]]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "NONZERO_RETURN_CODE_WITH_SUMMARY"
    )
    assert not second_summary.exists()
    assert not adjudication.exists()


def test_refit_worktree_drift_stops_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree_identity",
        lambda *_: {
            "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
            "expected_git_head": "d" * 40,
            "observed_git_head": "e" * 40,
            "head_return_code": 0,
            "porcelain_return_code": 0,
            "porcelain_lines": [],
        },
    )
    marker = tmp_path / "launched"
    runtime = tmp_path / "runtime.json"
    job = {
        "job_key": "refit:20260908:v4_full",
        "seed": 20260908,
        "summary_path": str(tmp_path / "run/run_summary.json"),
        "failure_path": str(tmp_path / "run/failure.json"),
        "log_path": str(tmp_path / "run.log"),
        "command": _writer(marker, {}, 0),
    }
    scheduler.run(
        {
            "git_head": "d" * 40,
            "worktree": str(tmp_path),
            "runtime_manifest": str(runtime),
            "gpu_queues": [{"physical_gpu_index": 0, "jobs": [job]}],
            "adjudication": {
                "manifest_path": str(tmp_path / "adjudication.json"),
                "failure_path": str(tmp_path / "adjudication.failed.json"),
                "log_path": str(tmp_path / "adjudication.log"),
                "command": _writer(None, {}, 0),
            },
        }
    )
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "SCHEDULE_WORKTREE_IDENTITY_DRIFT"
    )
    assert not marker.exists()


def test_refit_job_spawn_exception_writes_failure_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    monkeypatch.setattr(
        scheduler,
        "run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("trainer spawn failed")
        ),
    )
    runtime = tmp_path / "runtime.json"
    failure = tmp_path / "run/failure.json"
    manifest = tmp_path / "adjudication.json"
    job = {
        "job_key": "refit:20260908:v4_full",
        "seed": 20260908,
        "summary_path": str(tmp_path / "run/run_summary.json"),
        "failure_path": str(failure),
        "log_path": str(tmp_path / "run.log"),
        "command": ["missing-trainer"],
    }
    scheduler.run(
        {
            "git_head": "f" * 40,
            "worktree": str(tmp_path),
            "runtime_manifest": str(runtime),
            "gpu_queues": [{"physical_gpu_index": 0, "jobs": [job]}],
            "adjudication": {
                "manifest_path": str(manifest),
                "failure_path": str(tmp_path / "adjudication.failed.json"),
                "log_path": str(tmp_path / "adjudication.log"),
                "command": _writer(manifest, {"loso_authorized": True}, 0),
            },
        }
    )

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["jobs"][job["job_key"]]["status"] == "TECHNICAL_FAILURE"
    assert payload["jobs"][job["job_key"]]["return_code"] is None
    assert payload["adjudication"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "PROCESS_SPAWN_EXCEPTION"
    )
    assert evidence["failure_stage"] == "FORMAL_REFIT_JOB_SPAWN"
    assert evidence["return_code"] is None
    assert evidence["exception_type"] == "FileNotFoundError"
    assert evidence["exception_message"] == "trainer spawn failed"
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0
    assert not manifest.exists()


def test_refit_adjudication_spawn_exception_is_technical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    original_run_logged = scheduler.run_logged

    def fail_adjudication(command, *, cwd, log):
        if log.name == "adjudication.log":
            raise OSError("adjudication spawn failed")
        return original_run_logged(command, cwd=cwd, log=log)

    monkeypatch.setattr(scheduler, "run_logged", fail_adjudication)
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    adjudication_failure = tmp_path / "adjudication.failed.json"
    job = {
        "job_key": "refit:20260908:v4_full",
        "seed": 20260908,
        "summary_path": str(summary),
        "failure_path": str(tmp_path / "run/failure.json"),
        "log_path": str(tmp_path / "run.log"),
        "command": _writer(summary, {}, 0),
    }
    scheduler.run(
        {
            "git_head": "1" * 40,
            "worktree": str(tmp_path),
            "runtime_manifest": str(runtime),
            "gpu_queues": [{"physical_gpu_index": 0, "jobs": [job]}],
            "adjudication": {
                "manifest_path": str(tmp_path / "adjudication.json"),
                "failure_path": str(adjudication_failure),
                "log_path": str(tmp_path / "adjudication.log"),
                "command": ["missing-adjudicator"],
            },
        }
    )

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(adjudication_failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["jobs"][job["job_key"]]["status"] == "TERMINAL_COMPLETE"
    assert payload["adjudication"]["status"] == "TECHNICAL_FAILURE"
    assert payload["adjudication"]["return_code"] is None
    assert payload["first_terminal_failure"]["job_key"] == "REFIT_ADJUDICATION"
    assert evidence["failure_stage"] == "REFIT_ADJUDICATION_SPAWN"
    assert evidence["return_code"] is None
    assert evidence["exception_type"] == "OSError"
    assert evidence["exception_message"] == "adjudication spawn failed"
    assert evidence["development_test_outcome_reads_during_refit"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


def test_refit_adjudication_identity_drift_writes_failure_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_calls = 0

    def identity(*_):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            return None
        return {
            "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
            "expected_git_head": "2" * 40,
            "observed_git_head": "3" * 40,
            "head_return_code": 0,
            "porcelain_return_code": 0,
            "porcelain_lines": [],
        }

    monkeypatch.setattr(scheduler, "inspect_worktree_identity", identity)
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    marker = tmp_path / "adjudication-launched"
    adjudication_failure = tmp_path / "adjudication.failed.json"
    job = {
        "job_key": "refit:20260908:v4_full",
        "seed": 20260908,
        "summary_path": str(summary),
        "failure_path": str(tmp_path / "run/failure.json"),
        "log_path": str(tmp_path / "run.log"),
        "command": _writer(summary, {}, 0),
    }
    scheduler.run(
        {
            "git_head": "2" * 40,
            "worktree": str(tmp_path),
            "runtime_manifest": str(runtime),
            "gpu_queues": [{"physical_gpu_index": 0, "jobs": [job]}],
            "adjudication": {
                "manifest_path": str(tmp_path / "adjudication.json"),
                "failure_path": str(adjudication_failure),
                "log_path": str(tmp_path / "adjudication.log"),
                "command": _writer(marker, {}, 0),
            },
        }
    )

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(adjudication_failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TECHNICAL_FAILURE"
    assert payload["adjudication"]["status"] == "TECHNICAL_FAILURE"
    assert payload["adjudication"]["return_code"] is None
    assert payload["first_terminal_failure"]["job_key"] == "REFIT_ADJUDICATION"
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "SCHEDULE_WORKTREE_IDENTITY_DRIFT"
    )
    assert evidence["failure_stage"] == (
        "REFIT_ADJUDICATION_WORKTREE_IDENTITY"
    )
    assert evidence["return_code"] is None
    assert not marker.exists()


def test_refit_exact_terminal_rejects_double_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failure = tmp_path / "failure.json"
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
    summary.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) == "SUMMARY"
    failure.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
