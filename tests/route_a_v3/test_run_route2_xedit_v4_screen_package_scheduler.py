from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_screen_package_scheduler as scheduler


def _writer_command(
    output: Path, artifact: str, marker: Path, *, exit_code: int
) -> list[str]:
    source = (
        "import pathlib,sys,time;"
        f"out=pathlib.Path({str(output)!r});out.mkdir(parents=True);"
        f"pathlib.Path({str(marker)!r}).write_text(str(time.time()));"
        f"(out/{artifact!r}).write_text('{{}}\\n');"
        f"sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def test_scheduler_preserves_queue_order_and_accepts_exact_failure_terminal(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    first_output = tmp_path / "critic_first"
    failure_output = tmp_path / "critic_failure"
    setflow_output = tmp_path / "setflow"
    first_marker = tmp_path / "first.marker"
    failure_marker = tmp_path / "failure.marker"
    setflow_marker = tmp_path / "setflow.marker"
    schedule = {
        "git_head": "0" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "critic:first",
                        "component": "critic",
                        "run_id": "first",
                        "output_directory": str(first_output),
                        "log_path": str(tmp_path / "first.log"),
                        "command": _writer_command(
                            first_output, "run_summary.json", first_marker, exit_code=0
                        ),
                    },
                    {
                        "job_key": "critic:failure",
                        "component": "critic",
                        "run_id": "failure",
                        "output_directory": str(failure_output),
                        "log_path": str(tmp_path / "failure.log"),
                        "command": _writer_command(
                            failure_output, "failure.json", failure_marker, exit_code=7
                        ),
                    },
                ],
            },
            {
                "physical_gpu_index": 1,
                "jobs": [
                    {
                        "job_key": "setflow:full",
                        "component": "setflow",
                        "run_id": "full",
                        "output_directory": str(setflow_output),
                        "log_path": str(tmp_path / "setflow.log"),
                        "command": _writer_command(
                            setflow_output,
                            "training_summary.json",
                            setflow_marker,
                            exit_code=0,
                        ),
                    }
                ],
            },
        ],
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_SCREEN_PACKAGE_ALL_JOBS_TERMINAL"
    assert payload["active_performance_output_read"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0
    jobs = payload["jobs"]
    assert jobs["critic:first"]["terminal_artifact_kind"] == "SUMMARY"
    assert jobs["critic:failure"]["terminal_artifact_kind"] == "FAILURE"
    assert jobs["critic:failure"]["return_code"] == 7
    assert jobs["setflow:full"]["terminal_artifact_kind"] == "SUMMARY"
    assert float(failure_marker.read_text()) >= float(first_marker.read_text())


def test_terminal_kind_rejects_absent_or_double_terminal(tmp_path: Path) -> None:
    job = {
        "component": "critic",
        "output_directory": str(tmp_path),
    }
    assert scheduler.terminal_kind(job) is None
    (tmp_path / "run_summary.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) == "SUMMARY"
    (tmp_path / "failure.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) is None
