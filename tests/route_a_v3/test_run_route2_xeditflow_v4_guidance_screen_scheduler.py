from __future__ import annotations

import json
from pathlib import Path

import scripts.route_a_v3.run_route2_xeditflow_v4_guidance_screen_scheduler as scheduler


def _job(tmp_path: Path, key: str) -> dict[str, object]:
    return {
        "job_key": key,
        "command": ["produce", str(tmp_path / f"{key}.success")],
        "success_path": str(tmp_path / f"{key}.success"),
        "failure_path": str(tmp_path / f"{key}.failure"),
        "log_path": str(tmp_path / f"{key}.log"),
    }


def _schedule(tmp_path: Path) -> dict[str, object]:
    chains = [
        {
            "combination_id": f"combination_{index}",
            "jobs": [_job(tmp_path, f"guidance_{index}")],
        }
        for index in range(18)
    ]
    return {
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "worktree": str(tmp_path),
        "git_head": "a" * 40,
        "experiment_head": "b" * 40,
        "serial_value_prerequisites": [_job(tmp_path, "prerequisite")],
        "value_training_queues": [
            {"physical_gpu_index": 0, "jobs": [_job(tmp_path, "value")]}
        ],
        "guidance_queues": [
            {"physical_gpu_index": 0, "chains": chains}
        ],
        "adjudication": _job(tmp_path, "adjudication"),
    }


def test_guidance_scheduler_runs_exact_terminal_chain_without_reading_curves(
    tmp_path: Path, monkeypatch,
) -> None:
    def fake_run(command: list[str], *, cwd: Path, log: Path) -> int:
        Path(command[1]).write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(scheduler, "run_logged", fake_run)
    scheduler.run(_schedule(tmp_path))
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN"
    assert len(runtime["jobs"]) == 20
    assert all(
        row["terminal_artifact_kind"] == "SUCCESS"
        for row in runtime["jobs"].values()
    )
    assert runtime["adjudication"]["terminal_artifact_kind"] == "SUCCESS"
    assert runtime["active_performance_output_read"] is False
    assert runtime["development_test_reopened"] is False


def test_guidance_scheduler_stops_after_missing_prerequisite_terminal(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler,
        "run_logged",
        lambda command, *, cwd, log: 7,
    )
    schedule = _schedule(tmp_path)
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    prerequisite = runtime["jobs"]["prerequisite"]
    assert prerequisite["terminal_artifact_kind"] == "FAILURE"
    assert Path(prerequisite["failure_path"]).is_file()
    assert runtime["jobs"]["value"]["status"] == "NOT_RUN_PREREQUISITE_FAILURE"
