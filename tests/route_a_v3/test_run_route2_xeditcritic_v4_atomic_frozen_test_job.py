from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xeditcritic_v4_atomic_frozen_test_job as job_runner


def _writer(path: Path | None, *, exit_code: int) -> list[str]:
    action = ""
    if path is not None:
        action = (
            f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_text('{}\\n');"
        )
    return [
        sys.executable,
        "-c",
        f"import pathlib,sys;{action}sys.exit({exit_code})",
    ]


def _job(tmp_path: Path, command: list[str]) -> dict[str, object]:
    return {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "physical_gpu_index": 2,
        "output_directory": str(tmp_path / "output"),
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "log_path": str(tmp_path / "job.log"),
        "command": command,
    }


def test_atomic_test_job_preserves_result_after_late_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_runner, "inspect_worktree_identity", lambda job: None)
    result = tmp_path / "output/atomic_frozen_test.json"
    job_runner.run(_job(tmp_path, _writer(result, exit_code=7)))

    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["status"] == "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TERMINAL"
    assert runtime["terminal_artifact_kind"] == "RESULT"
    assert runtime["return_code"] == 7
    assert result.is_file()
    assert not (tmp_path / "output/failure.json").exists()
    assert runtime["terminal_payload_content_read_by_wrapper"] is False


def test_atomic_test_job_closes_preflight_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_runner, "inspect_worktree_identity", lambda job: None)
    job_runner.run(_job(tmp_path, _writer(None, exit_code=3)))

    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (tmp_path / "output/failure.json").read_text(encoding="utf-8")
    )
    assert runtime["status"] == "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TERMINAL"
    assert runtime["terminal_artifact_kind"] == "FAILURE"
    assert failure["status"] == "ATOMIC_FROZEN_TEST_TERMINAL_FAILURE_NO_AUTOMATIC_RETRY"
    assert failure["development_test_access_started"] is False
    assert failure["development_test_access_event_count"] == 0


def test_atomic_test_job_rejects_double_terminal(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "atomic_frozen_test.json").write_text("{}\n", encoding="utf-8")
    (output / "failure.json").write_text("{}\n", encoding="utf-8")
    assert job_runner.exact_terminal(output) is None


def test_atomic_job_identity_drift_fails_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path, _writer(None, exit_code=0))

    def drift(_job: dict[str, object]) -> None:
        raise job_runner.XEditCriticV4AtomicTestJobLaunchError(
            "job-fixed HEAD drift"
        )

    monkeypatch.setattr(job_runner, "inspect_worktree_identity", drift)
    monkeypatch.setattr(
        job_runner.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TEST runner Popen must not run after identity drift")
        ),
    )
    with pytest.raises(
        job_runner.XEditCriticV4AtomicTestJobLaunchError,
        match="job-fixed HEAD drift",
    ):
        job_runner.run(job)

    failure = json.loads(
        (tmp_path / "output/failure.json").read_text(encoding="utf-8")
    )
    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "JOB_FIXED_WORKTREE_IDENTITY"
    assert failure["development_test_access_started"] is False
    assert failure["development_test_access_event_count"] == 0
    assert failure["development_test_outcome_reads"] == 0
    assert runtime["status"] == "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TECHNICAL_FAILURE"
    assert runtime["terminal_artifact_kind"] == "FAILURE"
    assert runtime["gpu_job_started"] is False
    assert runtime["development_test_access_started"] is False


def test_atomic_job_identity_inspection_rejects_nonfixed_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path, _writer(None, exit_code=0))

    def inspect(arguments, **kwargs):
        stdout = "b" * 40 + "\n" if arguments[1:3] == ["rev-parse", "HEAD"] else ""
        return job_runner.subprocess.CompletedProcess(
            arguments,
            0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(job_runner.subprocess, "run", inspect)
    with pytest.raises(
        job_runner.XEditCriticV4AtomicTestJobLaunchError,
        match="job-fixed exact HEAD",
    ):
        job_runner.inspect_worktree_identity(job)


def test_atomic_job_popen_failure_is_terminal_and_not_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(tmp_path, _writer(None, exit_code=0))
    monkeypatch.setattr(job_runner, "inspect_worktree_identity", lambda value: None)

    def fail(*args, **kwargs):
        raise OSError("formal TEST runner spawn failed")

    monkeypatch.setattr(job_runner.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failed"):
        job_runner.run(job)

    failure_path = tmp_path / "output/failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "FORMAL_TEST_RUNNER_PROCESS_LAUNCH"
    assert failure["error_type"] == "OSError"
    assert failure["gpu_job_started"] is False
    assert failure["development_test_access_started"] is False
    assert failure["development_test_outcome_reads"] == 0
    assert runtime["status"] == "XEDITCRITIC_V4_ATOMIC_TEST_JOB_TECHNICAL_FAILURE"
    assert runtime["terminal_artifact_kind"] == "FAILURE"
    assert runtime["new_final_evaluation_outcome_reads"] == 0

    with pytest.raises(
        job_runner.XEditCriticV4AtomicTestJobLaunchError,
        match="job family already exists",
    ):
        job_runner.run(job)
