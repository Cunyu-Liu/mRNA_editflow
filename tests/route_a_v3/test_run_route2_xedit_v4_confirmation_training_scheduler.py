from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_confirmation_training_scheduler as scheduler


def _writer(output: Path, artifact: str, marker: Path, exit_code: int) -> list[str]:
    source = (
        "import pathlib,sys,time;"
        f"out=pathlib.Path({str(output)!r});out.mkdir(parents=True);"
        f"pathlib.Path({str(marker)!r}).write_text(str(time.time()));"
        f"(out/{artifact!r}).write_text('{{}}\\n');sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def test_confirmation_scheduler_keeps_exact_terminals_and_queue_order(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    critic_output = tmp_path / "critic"
    setflow_output = tmp_path / "setflow"
    critic_marker = tmp_path / "critic.marker"
    setflow_marker = tmp_path / "setflow.marker"
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
                            critic_output, "failure.json", critic_marker, 7
                        ),
                    },
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
                        ),
                    },
                ],
            }
        ],
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_CONFIRMATION_TRAINING_ALL_JOBS_TERMINAL"
    assert payload["jobs"]["critic:20260908:v4_full"]["terminal_artifact_kind"] == "FAILURE"
    assert payload["jobs"]["setflow:20260912:v4_full"]["terminal_artifact_kind"] == "SUMMARY"
    assert float(setflow_marker.read_text()) >= float(critic_marker.read_text())
    assert payload["active_performance_output_read"] is False
    assert payload["experiment_head"] == "b" * 40
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0


def test_confirmation_terminal_kind_rejects_double_terminal(tmp_path: Path) -> None:
    job = {"component": "critic", "output_directory": str(tmp_path)}
    assert scheduler.terminal_kind(job) is None
    (tmp_path / "run_summary.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) == "SUMMARY"
    (tmp_path / "failure.json").write_text("{}\n", encoding="utf-8")
    assert scheduler.terminal_kind(job) is None
