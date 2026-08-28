from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xeditcritic_v4_loso_scheduler as scheduler


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


def _successful_schedule(tmp_path: Path) -> dict[str, object]:
    job_summary = tmp_path / "job/run_summary.json"
    loso_summary = tmp_path / "loso.json"
    readiness_summary = tmp_path / "readiness.json"
    return {
        "git_head": "f" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "loso:20260908:GSE200304:v4_full",
                        "seed": 20260908,
                        "held_out_study": "GSE200304",
                        "run_id": "v4_full",
                        "summary_path": str(job_summary),
                        "failure_path": str(tmp_path / "job/failure.json"),
                        "log_path": str(tmp_path / "job.log"),
                        "command": _writer(job_summary, {}, 0),
                    }
                ],
            }
        ],
        "loso_adjudication": {
            "summary_path": str(loso_summary),
            "failure_path": str(tmp_path / "loso.failed.json"),
            "log_path": str(tmp_path / "loso.log"),
            "command": _writer(
                loso_summary,
                {"status": "XEDITCRITIC_V4_LOSO_TERMINAL"},
                0,
            ),
        },
        "readiness": {
            "summary_path": str(readiness_summary),
            "failure_path": str(tmp_path / "readiness.failed.json"),
            "log_path": str(tmp_path / "readiness.log"),
            "command": _writer(
                readiness_summary,
                {
                    "status": "CRITIC_V4_READY_FOR_GUIDANCE",
                    "guidance_authorized": True,
                },
                0,
            ),
        },
    }


def test_loso_scheduler_preserves_terminals_and_composes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    job_summary = tmp_path / "job/run_summary.json"
    job_failure = tmp_path / "job/failure.json"
    loso = tmp_path / "loso.json"
    readiness = tmp_path / "readiness.json"
    schedule = {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "loso:20260908:GSE200304:v4_full",
                        "seed": 20260908,
                        "held_out_study": "GSE200304",
                        "run_id": "v4_full",
                        "summary_path": str(job_summary),
                        "failure_path": str(job_failure),
                        "log_path": str(tmp_path / "job.log"),
                        "command": _writer(job_summary, {}, 0),
                    }
                ],
            }
        ],
        "loso_adjudication": {
            "summary_path": str(loso),
            "failure_path": str(tmp_path / "loso.failed.json"),
            "log_path": str(tmp_path / "loso.log"),
            "command": _writer(loso, {"status": "XEDITCRITIC_V4_LOSO_TERMINAL"}, 0),
        },
        "readiness": {
            "summary_path": str(readiness),
            "failure_path": str(tmp_path / "readiness.failed.json"),
            "log_path": str(tmp_path / "readiness.log"),
            "command": _writer(
                readiness,
                {"status": "CRITIC_V4_READY_FOR_GUIDANCE", "guidance_authorized": True},
                0,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "CRITIC_V4_READY_FOR_GUIDANCE"
    assert payload["jobs"]["loso:20260908:GSE200304:v4_full"]["terminal_artifact_kind"] == "SUMMARY"
    assert payload["loso_adjudication"]["return_code"] == 0
    assert payload["readiness"]["return_code"] == 0
    assert payload["readiness"]["guidance_authorized"] is True
    assert not job_failure.exists()


def test_loso_scheduler_stops_queue_after_first_technical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    job_summary = tmp_path / "job/run_summary.json"
    job_failure = tmp_path / "job/failure.json"
    skipped_summary = tmp_path / "skipped/run_summary.json"
    skipped_failure = tmp_path / "skipped/failure.json"
    loso = tmp_path / "loso.json"
    readiness = tmp_path / "readiness.json"
    schedule = {
        "git_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "loso:20260908:GSE200304:v4_full",
                        "seed": 20260908,
                        "held_out_study": "GSE200304",
                        "run_id": "v4_full",
                        "summary_path": str(job_summary),
                        "failure_path": str(job_failure),
                        "log_path": str(tmp_path / "job.log"),
                        "command": _writer(None, {}, 3),
                    },
                    {
                        "job_key": "loso:20260908:GSE114002:v4_full",
                        "seed": 20260908,
                        "held_out_study": "GSE114002",
                        "run_id": "v4_full",
                        "summary_path": str(skipped_summary),
                        "failure_path": str(skipped_failure),
                        "log_path": str(tmp_path / "skipped.log"),
                        "command": _writer(skipped_summary, {}, 0),
                    },
                ],
            }
        ],
        "loso_adjudication": {
            "summary_path": str(loso),
            "failure_path": str(tmp_path / "loso.failed.json"),
            "log_path": str(tmp_path / "loso.log"),
            "command": _writer(loso, {"status": "XEDITCRITIC_V4_LOSO_TERMINAL"}, 0),
        },
        "readiness": {
            "summary_path": str(readiness),
            "failure_path": str(tmp_path / "readiness.failed.json"),
            "log_path": str(tmp_path / "readiness.log"),
            "command": _writer(
                readiness,
                {"status": "CRITIC_V4_NOT_READY_FOR_GUIDANCE", "guidance_authorized": False},
                0,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    failed = payload["jobs"]["loso:20260908:GSE200304:v4_full"]
    assert failed["status"] == "TECHNICAL_FAILURE"
    assert failed["terminal_artifact_kind"] == "FAILURE"
    skipped = payload["jobs"]["loso:20260908:GSE114002:v4_full"]
    assert skipped["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert skipped["terminal_artifact_kind"] is None
    assert payload["first_terminal_failure"]["job_key"] == (
        "loso:20260908:GSE200304:v4_full"
    )
    assert job_failure.is_file()
    assert not skipped_summary.exists()
    assert not skipped_failure.exists()
    assert not loso.exists()
    assert not readiness.exists()
    assert payload["loso_adjudication"]["status"] == (
        "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE"
    )
    assert payload["readiness"]["status"] == (
        "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE"
    )
    assert payload["readiness"]["guidance_authorized"] is False


def test_loso_nonzero_with_summary_is_technical_and_stops_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    runtime = tmp_path / "runtime.json"
    first_summary = tmp_path / "first/run_summary.json"
    second_summary = tmp_path / "second/run_summary.json"
    loso = tmp_path / "loso.json"
    readiness = tmp_path / "readiness.json"
    first = {
        "job_key": "loso:20260908:GSE200304:v4_full",
        "seed": 20260908,
        "held_out_study": "GSE200304",
        "run_id": "v4_full",
        "summary_path": str(first_summary),
        "failure_path": str(first_summary.parent / "failure.json"),
        "log_path": str(first_summary.parent / "job.log"),
        "command": _writer(first_summary, {}, 4),
    }
    second = {
        "job_key": "loso:20260908:GSE114002:v4_full",
        "seed": 20260908,
        "held_out_study": "GSE114002",
        "run_id": "v4_full",
        "summary_path": str(second_summary),
        "failure_path": str(second_summary.parent / "failure.json"),
        "log_path": str(second_summary.parent / "job.log"),
        "command": _writer(second_summary, {}, 0),
    }
    schedule = {
        "git_head": "c" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [{"physical_gpu_index": 0, "jobs": [first, second]}],
        "loso_adjudication": {
            "summary_path": str(loso),
            "failure_path": str(tmp_path / "loso.failed.json"),
            "log_path": str(tmp_path / "loso.log"),
            "command": _writer(loso, {}, 0),
        },
        "readiness": {
            "summary_path": str(readiness),
            "failure_path": str(tmp_path / "readiness.failed.json"),
            "log_path": str(tmp_path / "readiness.log"),
            "command": _writer(readiness, {}, 0),
        },
    }
    scheduler.run(schedule)
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload["jobs"][first["job_key"]]["status"] == "TECHNICAL_FAILURE"
    assert payload["jobs"][second["job_key"]]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "NONZERO_RETURN_CODE_WITH_SUMMARY"
    )
    assert not second_summary.exists()
    assert not loso.exists()
    assert not readiness.exists()


def test_loso_worktree_drift_stops_before_popen(
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
    runtime = tmp_path / "runtime.json"
    marker = tmp_path / "launched"
    job = {
        "job_key": "loso:20260908:GSE200304:v4_full",
        "seed": 20260908,
        "held_out_study": "GSE200304",
        "run_id": "v4_full",
        "summary_path": str(tmp_path / "job/run_summary.json"),
        "failure_path": str(tmp_path / "job/failure.json"),
        "log_path": str(tmp_path / "job.log"),
        "command": _writer(marker, {}, 0),
    }
    scheduler.run(
        {
            "git_head": "d" * 40,
            "worktree": str(tmp_path),
            "runtime_manifest": str(runtime),
            "gpu_queues": [{"physical_gpu_index": 0, "jobs": [job]}],
            "loso_adjudication": {
                "summary_path": str(tmp_path / "loso.json"),
                "failure_path": str(tmp_path / "loso.failed.json"),
                "log_path": str(tmp_path / "loso.log"),
                "command": _writer(None, {}, 0),
            },
            "readiness": {
                "summary_path": str(tmp_path / "readiness.json"),
                "failure_path": str(tmp_path / "readiness.failed.json"),
                "log_path": str(tmp_path / "readiness.log"),
                "command": _writer(None, {}, 0),
            },
        }
    )
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "SCHEDULE_WORKTREE_IDENTITY_DRIFT"
    )
    assert not marker.exists()


def test_loso_job_spawn_exception_writes_failure_and_stops_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    monkeypatch.setattr(
        scheduler,
        "run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("LOSO trainer spawn failed")
        ),
    )
    schedule = _successful_schedule(tmp_path)
    second_summary = tmp_path / "second/run_summary.json"
    second = {
        "job_key": "loso:20260908:GSE114002:v4_full",
        "seed": 20260908,
        "held_out_study": "GSE114002",
        "run_id": "v4_full",
        "summary_path": str(second_summary),
        "failure_path": str(tmp_path / "second/failure.json"),
        "log_path": str(tmp_path / "second.log"),
        "command": _writer(second_summary, {}, 0),
    }
    schedule["gpu_queues"][0]["jobs"].append(second)
    scheduler.run(schedule)

    runtime = Path(schedule["runtime_manifest"])
    first = schedule["gpu_queues"][0]["jobs"][0]
    failure = Path(first["failure_path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload["jobs"][first["job_key"]]["status"] == "TECHNICAL_FAILURE"
    assert payload["jobs"][first["job_key"]]["return_code"] is None
    assert payload["jobs"][second["job_key"]]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert payload["loso_adjudication"]["status"] == (
        "NOT_RUN_LOSO_JOB_TECHNICAL_FAILURE"
    )
    assert payload["readiness"]["guidance_authorized"] is False
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "PROCESS_SPAWN_EXCEPTION"
    )
    assert evidence["failure_stage"] == "FORMAL_LOSO_JOB_SPAWN"
    assert evidence["return_code"] is None
    assert evidence["exception_type"] == "FileNotFoundError"
    assert evidence["exception_message"] == "LOSO trainer spawn failed"
    assert evidence["development_test_outcome_reads"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0
    assert not second_summary.exists()


def test_loso_adjudication_spawn_exception_is_technical_and_blocks_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    original_run_logged = scheduler.run_logged

    def fail_adjudication(command, *, cwd, log):
        if log.name == "loso.log":
            raise OSError("LOSO adjudication spawn failed")
        return original_run_logged(command, cwd=cwd, log=log)

    monkeypatch.setattr(scheduler, "run_logged", fail_adjudication)
    schedule = _successful_schedule(tmp_path)
    scheduler.run(schedule)

    runtime = Path(schedule["runtime_manifest"])
    failure = Path(schedule["loso_adjudication"]["failure_path"])
    readiness_summary = Path(schedule["readiness"]["summary_path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload["loso_adjudication"]["status"] == "TECHNICAL_FAILURE"
    assert payload["loso_adjudication"]["return_code"] is None
    assert payload["readiness"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert payload["readiness"]["guidance_authorized"] is False
    assert payload["first_terminal_failure"]["job_key"] == "LOSO_ADJUDICATION"
    assert evidence["failure_stage"] == "LOSO_ADJUDICATION_SPAWN"
    assert evidence["return_code"] is None
    assert evidence["exception_type"] == "OSError"
    assert evidence["exception_message"] == "LOSO adjudication spawn failed"
    assert evidence["development_test_outcome_reads_during_loso"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0
    assert not readiness_summary.exists()


def test_loso_readiness_spawn_exception_is_technical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_: None)
    original_run_logged = scheduler.run_logged

    def fail_readiness(command, *, cwd, log):
        if log.name == "readiness.log":
            raise RuntimeError("readiness spawn failed")
        return original_run_logged(command, cwd=cwd, log=log)

    monkeypatch.setattr(scheduler, "run_logged", fail_readiness)
    schedule = _successful_schedule(tmp_path)
    scheduler.run(schedule)

    runtime = Path(schedule["runtime_manifest"])
    failure = Path(schedule["readiness"]["failure_path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload["loso_adjudication"]["status"] == "TERMINAL_COMPLETE"
    assert payload["readiness"]["status"] == "TECHNICAL_FAILURE"
    assert payload["readiness"]["return_code"] is None
    assert payload["readiness"]["guidance_authorized"] is False
    assert payload["first_terminal_failure"]["job_key"] == "READINESS"
    assert evidence["failure_stage"] == "READINESS_SPAWN"
    assert evidence["return_code"] is None
    assert evidence["exception_type"] == "RuntimeError"
    assert evidence["exception_message"] == "readiness spawn failed"
    assert evidence["development_test_outcome_reads_after_atomic_test"] == 0
    assert evidence["new_final_evaluation_outcome_reads"] == 0


@pytest.mark.parametrize(
    ("drift_call", "barrier", "failure_stage"),
    (
        (2, "LOSO_ADJUDICATION", "LOSO_ADJUDICATION_WORKTREE_IDENTITY"),
        (3, "READINESS", "READINESS_WORKTREE_IDENTITY"),
    ),
)
def test_loso_barrier_identity_drift_writes_failure_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_call: int,
    barrier: str,
    failure_stage: str,
) -> None:
    identity_calls = 0

    def identity(*_):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls != drift_call:
            return None
        return {
            "failure_reason": "SCHEDULE_WORKTREE_IDENTITY_DRIFT",
            "expected_git_head": "f" * 40,
            "observed_git_head": "e" * 40,
            "head_return_code": 0,
            "porcelain_return_code": 0,
            "porcelain_lines": [],
        }

    monkeypatch.setattr(scheduler, "inspect_worktree_identity", identity)
    schedule = _successful_schedule(tmp_path)
    scheduler.run(schedule)

    runtime = Path(schedule["runtime_manifest"])
    spec_key = "loso_adjudication" if barrier == "LOSO_ADJUDICATION" else "readiness"
    spec = schedule[spec_key]
    failure = Path(spec["failure_path"])
    summary = Path(spec["summary_path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    evidence = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE"
    assert payload[spec_key]["status"] == "TECHNICAL_FAILURE"
    assert payload[spec_key]["return_code"] is None
    assert payload["first_terminal_failure"]["job_key"] == barrier
    assert payload["first_terminal_failure"]["failure_reason"] == (
        "SCHEDULE_WORKTREE_IDENTITY_DRIFT"
    )
    assert evidence["failure_stage"] == failure_stage
    assert evidence["return_code"] is None
    assert not summary.exists()
    if barrier == "LOSO_ADJUDICATION":
        assert payload["readiness"]["status"] == (
            "NOT_RUN_AFTER_TERMINAL_FAILURE"
        )
        assert payload["readiness"]["guidance_authorized"] is False


def test_loso_exact_terminal_rejects_double_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failure = tmp_path / "failure.json"
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
    summary.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) == "SUMMARY"
    failure.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
