from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xeditcritic_v402_recovery_scheduler as scheduler


def _writer(output: Path, artifact: str, marker: Path, exit_code: int) -> list[str]:
    source = (
        "import pathlib,sys,time;"
        f"out=pathlib.Path({str(output)!r});out.mkdir(parents=True);"
        f"pathlib.Path({str(marker)!r}).write_text(str(time.time()));"
        f"(out/{artifact!r}).write_text('{{}}\\n');"
        f"sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def _schedule(tmp_path: Path) -> dict:
    jobs = []
    for index, run_id in enumerate(scheduler.RUN_IDS):
        output = tmp_path / run_id
        artifact = "failure.json" if index == 2 else "run_summary.json"
        jobs.append(
            {
                "run_id": run_id,
                "output_directory": str(output),
                "log_path": str(tmp_path / f"{run_id}.log"),
                "command": _writer(
                    output,
                    artifact,
                    tmp_path / f"{run_id}.marker",
                    7 if artifact == "failure.json" else 0,
                ),
            }
        )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v402_recovery_schedule.v1"
        ),
        "status": "FROZEN_V402_RECOVERY_SCHEDULE",
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "physical_gpu_index": 5,
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "jobs": jobs,
        "active_performance_output_read": False,
        "setflow_jobs_stopped_modified_or_restarted": False,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_scheduler_completes_all_eight_in_order_after_one_failure(
    tmp_path: Path,
) -> None:
    schedule = _schedule(tmp_path)
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITCRITIC_V402_RECOVERY_ALL_EIGHT_TERMINAL"
    assert set(runtime["jobs"]) == set(scheduler.RUN_IDS)
    assert runtime["jobs"][scheduler.RUN_IDS[2]]["terminal_artifact_kind"] == (
        "FAILURE"
    )
    assert runtime["jobs"][scheduler.RUN_IDS[-1]]["terminal_artifact_kind"] == (
        "SUMMARY"
    )
    timestamps = [
        float((tmp_path / f"{run_id}.marker").read_text())
        for run_id in scheduler.RUN_IDS
    ]
    assert timestamps == sorted(timestamps)
    assert runtime["terminal_artifact_payloads_read"] == 0
    assert runtime["active_performance_output_read"] is False
    assert runtime["setflow_jobs_stopped_modified_or_restarted"] is False
    assert runtime["development_test_outcome_reads"] == 0
    assert runtime["new_final_evaluation_outcome_reads"] == 0


def test_schedule_rejects_missing_arm_or_non_gpu5(tmp_path: Path) -> None:
    schedule = _schedule(tmp_path)
    schedule["jobs"] = schedule["jobs"][:-1]
    with pytest.raises(Exception, match="eight-arm"):
        scheduler.validate_schedule(schedule)
    schedule = _schedule(tmp_path)
    schedule["physical_gpu_index"] = 4
    with pytest.raises(Exception, match="GPU5"):
        scheduler.validate_schedule(schedule)


def test_terminal_kind_requires_exactly_one_terminal_artifact(tmp_path: Path) -> None:
    job = {"output_directory": str(tmp_path)}
    assert scheduler.terminal_kind(job) is None
    (tmp_path / "run_summary.json").write_text("{}\n")
    assert scheduler.terminal_kind(job) == "SUMMARY"
    (tmp_path / "failure.json").write_text("{}\n")
    assert scheduler.terminal_kind(job) is None
