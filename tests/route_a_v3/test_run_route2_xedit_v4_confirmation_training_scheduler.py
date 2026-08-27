from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_confirmation_training_scheduler as scheduler


def _writer(
    output: Path,
    artifact: str,
    marker: Path,
    exit_code: int,
    *,
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
    source = (
        "import json,pathlib,sys,time;"
        f"{wait}"
        f"out=pathlib.Path({str(output)!r});out.mkdir(parents=True);"
        f"pathlib.Path({str(marker)!r}).write_text(str(time.time()));"
        f"{runtime_wait}"
        f"(out/{artifact!r}).write_text('{{}}\\n');sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def test_confirmation_scheduler_stops_all_queues_after_first_technical_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler, "inspect_worktree_identity", lambda *args, **kwargs: None
    )
    runtime = tmp_path / "runtime.json"
    critic_output = tmp_path / "critic"
    setflow_output = tmp_path / "setflow"
    skipped_output = tmp_path / "skipped"
    critic_marker = tmp_path / "critic.marker"
    setflow_marker = tmp_path / "setflow.marker"
    skipped_marker = tmp_path / "skipped.marker"
    schedule = {
        "git_head": "a" * 40,
        "experiment_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["critic", "setflow"],
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "critic:20260908:v4_full",
                        "component": "critic",
                        "training_seed": 20260908,
                        "run_id": "v4_full",
                        "output_directory": str(critic_output),
                        "log_path": str(tmp_path / "critic.log"),
                        "command": _writer(
                            critic_output,
                            "failure.json",
                            critic_marker,
                            7,
                            wait_for=setflow_marker,
                        ),
                    }
                ],
            },
            {
                "physical_gpu_index": 1,
                "jobs": [
                    {
                        "job_key": "setflow:20260912:v4_full",
                        "component": "setflow",
                        "training_seed": 20260912,
                        "run_id": "v4_full",
                        "output_directory": str(setflow_output),
                        "log_path": str(tmp_path / "setflow.log"),
                        "command": _writer(
                            setflow_output,
                            "training_summary.json",
                            setflow_marker,
                            0,
                            wait_for_runtime_failure=runtime,
                        ),
                    },
                    {
                        "job_key": "setflow:20260913:v4_full",
                        "component": "setflow",
                        "training_seed": 20260913,
                        "run_id": "v4_full",
                        "output_directory": str(skipped_output),
                        "log_path": str(tmp_path / "skipped.log"),
                        "command": _writer(
                            skipped_output,
                            "training_summary.json",
                            skipped_marker,
                            0,
                        ),
                    },
                ],
            },
        ],
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_TRAINING_TECHNICAL_FAILURE"
    assert (
        payload["jobs"]["critic:20260908:v4_full"]["terminal_artifact_kind"]
        == "FAILURE"
    )
    assert (
        payload["jobs"]["setflow:20260912:v4_full"]["terminal_artifact_kind"]
        == "SUMMARY"
    )
    skipped = payload["jobs"]["setflow:20260913:v4_full"]
    assert skipped["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert skipped["terminal_artifact_kind"] is None
    assert not skipped_marker.exists()
    assert payload["first_terminal_failure"]["job_key"] == (
        "critic:20260908:v4_full"
    )
    assert payload["active_performance_output_read"] is False
    assert payload["experiment_head"] == "b" * 40
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0


def test_confirmation_scheduler_records_worktree_mismatch_before_launch(
    tmp_path: Path, monkeypatch,
) -> None:
    inspection = {
        "reason": "WORKTREE_HEAD_MISMATCH",
        "expected_git_head": "a" * 40,
        "observed_git_head": "b" * 40,
    }
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree_identity",
        lambda *args, **kwargs: inspection,
    )
    runtime = tmp_path / "runtime.json"
    output = tmp_path / "critic"
    marker = tmp_path / "critic.marker"
    schedule = {
        "git_head": "a" * 40,
        "experiment_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "eligible_components": ["critic"],
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "critic:20260908:v4_full",
                        "component": "critic",
                        "training_seed": 20260908,
                        "run_id": "v4_full",
                        "output_directory": str(output),
                        "log_path": str(tmp_path / "critic.log"),
                        "command": _writer(output, "run_summary.json", marker, 0),
                    }
                ],
            }
        ],
    }

    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_TRAINING_TECHNICAL_FAILURE"
    assert payload["first_terminal_failure"]["reason"] == "WORKTREE_HEAD_MISMATCH"
    assert (
        payload["jobs"]["critic:20260908:v4_full"]["terminal_artifact_kind"]
        == "FAILURE"
    )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["worktree_inspection"] == inspection
    assert not marker.exists()


def test_confirmation_terminal_kind_rejects_double_terminal(tmp_path: Path) -> None:
    job = {"component": "critic", "output_directory": str(tmp_path)}
    assert scheduler.terminal_kind(job) is None
    (tmp_path / "run_summary.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) == "SUMMARY"
    (tmp_path / "failure.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) is None
