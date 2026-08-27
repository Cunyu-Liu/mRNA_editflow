from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_confirmation_posttraining_scheduler as scheduler


def _writer(
    path: Path | None,
    *,
    exit_code: int,
    marker: Path | None = None,
    wait_for: Path | None = None,
    wait_for_runtime_failure: Path | None = None,
) -> list[str]:
    wait = ""
    if wait_for is not None:
        wait = (
            f"deadline=time.time()+5;target=pathlib.Path({str(wait_for)!r});"
            "\nwhile not target.exists() and time.time()<deadline:time.sleep(.01);"
            "\nassert target.exists();"
        )
    runtime_wait = ""
    if wait_for_runtime_failure is not None:
        runtime_wait = (
            f"deadline=time.time()+5;runtime=pathlib.Path({str(wait_for_runtime_failure)!r});"
            "payload={};"
            "\nwhile time.time()<deadline:"
            "\n try:payload=json.loads(runtime.read_text())"
            "\n except (FileNotFoundError,json.JSONDecodeError):payload={}"
            "\n if payload.get('first_terminal_failure'):break"
            "\n time.sleep(.01)"
            "\nassert payload.get('first_terminal_failure');"
        )
    mark = ""
    if marker is not None:
        mark = (
            f"m=pathlib.Path({str(marker)!r});m.parent.mkdir(parents=True,exist_ok=True);"
            "m.write_text('started');"
        )
    action = ""
    if path is not None:
        action = (
            f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_text('{}\\n');"
        )
    source = (
        f"import json,pathlib,sys,time;{wait}{mark}{runtime_wait}"
        f"{action}sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def test_confirmation_posttraining_preserves_atomic_gate_and_validation_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler, "inspect_worktree_identity", lambda *args, **kwargs: None
    )
    runtime = tmp_path / "runtime.json"
    validation_summary = tmp_path / "validation/pass_4/validation_summary.json"
    validation_failure = tmp_path / "validation/pass_4.failed.json"
    inflight_summary = tmp_path / "validation/inflight/validation_summary.json"
    inflight_failure = tmp_path / "validation/inflight.failed.json"
    skipped_summary = tmp_path / "validation/pass_8/validation_summary.json"
    skipped_failure = tmp_path / "validation/pass_8.failed.json"
    inflight_marker = tmp_path / "validation/inflight.marker"
    critic_gate = tmp_path / "critic_gate.json"
    critic_failure = tmp_path / "critic_gate.failed.json"
    schedule = {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["critic"],
        "validation_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "setflow:20260912:pass_4",
                        "training_seed": 20260912,
                        "checkpoint_pass": 4,
                        "physical_gpu_index": 0,
                        "terminal_summary": str(validation_summary),
                        "terminal_failure": str(validation_failure),
                        "log_path": str(tmp_path / "validation.log"),
                        "command": _writer(
                            None, exit_code=7, wait_for=inflight_marker
                        ),
                    }
                ],
            },
            {
                "physical_gpu_index": 1,
                "jobs": [
                    {
                        "job_key": "setflow:20260913:pass_4",
                        "training_seed": 20260913,
                        "checkpoint_pass": 4,
                        "physical_gpu_index": 1,
                        "terminal_summary": str(inflight_summary),
                        "terminal_failure": str(inflight_failure),
                        "log_path": str(tmp_path / "inflight_validation.log"),
                        "command": _writer(
                            inflight_summary,
                            exit_code=0,
                            marker=inflight_marker,
                            wait_for_runtime_failure=runtime,
                        ),
                    },
                    {
                        "job_key": "setflow:20260913:pass_8",
                        "training_seed": 20260913,
                        "checkpoint_pass": 8,
                        "physical_gpu_index": 1,
                        "terminal_summary": str(skipped_summary),
                        "terminal_failure": str(skipped_failure),
                        "log_path": str(tmp_path / "skipped_validation.log"),
                        "command": _writer(skipped_summary, exit_code=0),
                    }
                ],
            }
        ],
        "adjudications": {
            "critic": {
                "gate_path": str(critic_gate),
                "failure_path": str(critic_failure),
                "log_path": str(tmp_path / "critic.log"),
                "command": _writer(critic_gate, exit_code=9),
            }
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE"
    validation = payload["validation_jobs"]["setflow:20260912:pass_4"]
    assert validation["terminal_artifact_kind"] == "FAILURE"
    assert validation_failure.is_file()
    assert (
        payload["validation_jobs"]["setflow:20260913:pass_4"][
            "terminal_artifact_kind"
        ]
        == "SUMMARY"
    )
    skipped = payload["validation_jobs"]["setflow:20260913:pass_8"]
    assert skipped["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert skipped["terminal_artifact_kind"] is None
    assert not skipped_summary.exists()
    assert not skipped_failure.exists()
    adjudication = payload["adjudications"]["critic"]
    assert adjudication["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert adjudication["terminal_artifact_kind"] is None
    assert not critic_gate.exists()
    assert not critic_failure.exists()
    assert payload["first_terminal_failure"]["stage"] == "VALIDATION"
    assert payload["active_performance_output_read"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0


def test_confirmation_posttraining_publishes_adjudication_failure_when_gate_absent(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler, "inspect_worktree_identity", lambda *args, **kwargs: None
    )
    runtime = tmp_path / "runtime.json"
    gate = tmp_path / "setflow_gate.json"
    failure = tmp_path / "setflow_gate.failed.json"
    schedule = {
        "git_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["setflow"],
        "validation_queues": [],
        "adjudications": {
            "setflow": {
                "gate_path": str(gate),
                "failure_path": str(failure),
                "log_path": str(tmp_path / "setflow.log"),
                "command": _writer(None, exit_code=3),
            }
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE"
    assert payload["adjudications"]["setflow"]["terminal_artifact_kind"] == "FAILURE"
    assert failure.is_file()
    assert not gate.exists()
    assert payload["first_terminal_failure"]["stage"] == "ADJUDICATION"


def test_confirmation_validation_summary_survives_late_nonzero_exit(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler, "inspect_worktree_identity", lambda *args, **kwargs: None
    )
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "validation/pass_4/validation_summary.json"
    failure = tmp_path / "validation/pass_4.failed.json"
    gate = tmp_path / "setflow_gate.json"
    gate_failure = tmp_path / "setflow_gate.failed.json"
    schedule = {
        "git_head": "c" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["setflow"],
        "validation_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "setflow:20260912:pass_4",
                        "training_seed": 20260912,
                        "checkpoint_pass": 4,
                        "physical_gpu_index": 0,
                        "terminal_summary": str(summary),
                        "terminal_failure": str(failure),
                        "log_path": str(tmp_path / "validation.log"),
                        "command": _writer(summary, exit_code=7),
                    }
                ],
            }
        ],
        "adjudications": {
            "setflow": {
                "gate_path": str(gate),
                "failure_path": str(gate_failure),
                "log_path": str(tmp_path / "setflow.log"),
                "command": _writer(gate, exit_code=0),
            }
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
    validation = payload["validation_jobs"]["setflow:20260912:pass_4"]
    assert validation["terminal_artifact_kind"] == "SUMMARY"
    assert validation["return_code"] == 7
    assert summary.is_file()
    assert not failure.exists()


def test_confirmation_posttraining_records_worktree_mismatch_before_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    inspection = {
        "reason": "WORKTREE_NOT_CLEAN",
        "expected_git_head": "d" * 40,
        "observed_git_head": "d" * 40,
    }
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree_identity",
        lambda *args, **kwargs: inspection,
    )
    runtime = tmp_path / "runtime.json"
    validation_summary = tmp_path / "validation/pass_4/validation_summary.json"
    validation_failure = tmp_path / "validation/pass_4.failed.json"
    gate = tmp_path / "setflow_gate.json"
    gate_failure = tmp_path / "setflow_gate.failed.json"
    schedule = {
        "git_head": "d" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["setflow"],
        "validation_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "setflow:20260912:pass_4",
                        "training_seed": 20260912,
                        "checkpoint_pass": 4,
                        "physical_gpu_index": 0,
                        "terminal_summary": str(validation_summary),
                        "terminal_failure": str(validation_failure),
                        "log_path": str(tmp_path / "validation.log"),
                        "command": _writer(validation_summary, exit_code=0),
                    }
                ],
            }
        ],
        "adjudications": {
            "setflow": {
                "gate_path": str(gate),
                "failure_path": str(gate_failure),
                "log_path": str(tmp_path / "setflow.log"),
                "command": _writer(gate, exit_code=0),
            }
        },
    }

    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_TECHNICAL_FAILURE"
    assert payload["first_terminal_failure"]["reason"] == "WORKTREE_NOT_CLEAN"
    validation = payload["validation_jobs"]["setflow:20260912:pass_4"]
    assert validation["terminal_artifact_kind"] == "FAILURE"
    failure = json.loads(validation_failure.read_text(encoding="utf-8"))
    assert failure["worktree_inspection"] == inspection
    assert payload["adjudications"]["setflow"]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert not validation_summary.exists()
    assert not gate.exists()


def test_exact_terminal_rejects_absent_or_double_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failure = tmp_path / "failure.json"
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
    summary.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) == "SUMMARY"
    failure.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
