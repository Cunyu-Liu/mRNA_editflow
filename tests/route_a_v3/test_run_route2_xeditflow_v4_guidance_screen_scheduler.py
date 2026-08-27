from __future__ import annotations

import json
import threading
import types
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


def _install_fake_execution(
    monkeypatch,
    wait,
    launched: list[str] | None = None,
) -> list[str]:
    launches = launched if launched is not None else []

    def fake_start(command: list[str], *, cwd: Path, log: Path):
        del cwd, log
        launches.append(Path(command[1]).stem)
        return command, None

    monkeypatch.setattr(scheduler, "start_logged", fake_start)
    monkeypatch.setattr(
        scheduler,
        "wait_logged",
        lambda process, stream: wait(process),
    )
    return launches


def test_guidance_scheduler_runs_exact_terminal_chain_without_reading_curves(
    tmp_path: Path, monkeypatch,
) -> None:
    def fake_wait(command: list[str]) -> int:
        Path(command[1]).write_text("{}\n", encoding="utf-8")
        return 0

    _install_fake_execution(monkeypatch, fake_wait)
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree",
        lambda worktree, expected_head: {"exact_clean_head": True},
    )
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
    _install_fake_execution(monkeypatch, lambda command: 7)
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree",
        lambda worktree, expected_head: {"exact_clean_head": True},
    )
    schedule = _schedule(tmp_path)
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    prerequisite = runtime["jobs"]["prerequisite"]
    assert prerequisite["terminal_artifact_kind"] == "FAILURE"
    assert Path(prerequisite["failure_path"]).is_file()
    assert runtime["jobs"]["value"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert runtime["adjudication"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"


def test_guidance_scheduler_stops_all_queues_after_first_technical_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    schedule = _schedule(tmp_path)
    schedule["value_training_queues"] = [
        {
            "physical_gpu_index": 0,
            "jobs": [
                _job(tmp_path, "value_failure"),
                _job(tmp_path, "value_after_failure"),
            ],
        },
        {
            "physical_gpu_index": 1,
            "jobs": [
                _job(tmp_path, "value_in_flight"),
                _job(tmp_path, "value_after_in_flight"),
            ],
        },
    ]
    real_event = threading.Event
    real_lock = threading.Lock
    in_flight_started = real_event()
    failure_artifact_written = real_event()
    package_failure_event_set = real_event()
    coordination_failures: list[str] = []
    invoked: list[str] = []
    inspection_count = 0
    original_write_atomic = scheduler.write_atomic

    def observe_write(path: Path, payload: dict[str, object]) -> None:
        original_write_atomic(path, payload)
        failure_job = schedule["value_training_queues"][0]["jobs"][0]
        if path == Path(failure_job["failure_path"]):
            failure_artifact_written.set()

    class TrackingLock:
        def __init__(self) -> None:
            self.inner = real_lock()
            self.owner: int | None = None

        def __enter__(self):
            self.inner.acquire()
            self.owner = threading.get_ident()
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            self.owner = None
            self.inner.release()

        def held_by_current_thread(self) -> bool:
            return self.owner == threading.get_ident()

    start_lock: TrackingLock | None = None

    def lock_factory() -> TrackingLock:
        nonlocal start_lock
        assert start_lock is None
        start_lock = TrackingLock()
        return start_lock

    class TrackingPackageFailureEvent:
        def __init__(self) -> None:
            self.inner = real_event()

        def is_set(self) -> bool:
            return self.inner.is_set()

        def set(self) -> None:
            if start_lock is None or not start_lock.held_by_current_thread():
                coordination_failures.append(
                    "package failure Event was set outside the launch lock"
                )
            self.inner.set()
            package_failure_event_set.set()

    package_event_created = False

    def event_factory():
        nonlocal package_event_created
        if not package_event_created:
            package_event_created = True
            return TrackingPackageFailureEvent()
        return real_event()

    def fake_wait(command: list[str]) -> int:
        key = Path(command[1]).stem
        if key == "value_failure":
            if not in_flight_started.wait(timeout=5):
                coordination_failures.append("in-flight job did not start")
            return 7
        if key == "value_in_flight":
            in_flight_started.set()
            if not failure_artifact_written.wait(timeout=5):
                coordination_failures.append("failure artifact was not written")
            if not package_failure_event_set.wait(timeout=5):
                coordination_failures.append("package failure Event was not set")
        Path(command[1]).write_text("{}\n", encoding="utf-8")
        return 0

    def inspect(worktree: Path, expected_head: str) -> dict[str, object]:
        nonlocal inspection_count
        del worktree, expected_head
        inspection_count += 1
        return {"exact_clean_head": True}

    monkeypatch.setattr(scheduler, "write_atomic", observe_write)
    monkeypatch.setattr(
        scheduler,
        "threading",
        types.SimpleNamespace(
            Lock=lock_factory,
            Event=event_factory,
            Thread=threading.Thread,
        ),
    )
    _install_fake_execution(monkeypatch, fake_wait, invoked)
    monkeypatch.setattr(scheduler, "inspect_worktree", inspect)
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert not coordination_failures
    assert failure_artifact_written.is_set()
    assert package_failure_event_set.is_set()
    assert inspection_count == 3
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["job_key"] == "value_failure"
    assert runtime["jobs"]["value_in_flight"]["status"] == "TERMINAL_COMPLETE"
    assert runtime["jobs"]["value_after_failure"]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert runtime["jobs"]["value_after_in_flight"]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert "value_after_failure" not in invoked
    assert "value_after_in_flight" not in invoked
    assert "adjudication" not in invoked


def test_guidance_scheduler_rechecks_exact_clean_head_before_each_popen(
    tmp_path: Path, monkeypatch,
) -> None:
    invoked: list[str] = []
    inspection_count = 0

    def fake_wait(command: list[str]) -> int:
        key = Path(command[1]).stem
        Path(command[1]).write_text("{}\n", encoding="utf-8")
        return 0

    def inspect(worktree: Path, expected_head: str) -> dict[str, object]:
        nonlocal inspection_count
        inspection_count += 1
        if inspection_count == 1:
            return {"exact_clean_head": True}
        return {
            "exact_clean_head": False,
            "expected_git_head": expected_head,
            "observed_git_head": "c" * 40,
            "git_status_porcelain": " M tracked.py\n",
            "git_head_return_code": 0,
            "git_status_return_code": 0,
            "git_head_stderr": "",
            "git_status_stderr": "",
            "inspection_error": None,
        }

    _install_fake_execution(monkeypatch, fake_wait, invoked)
    monkeypatch.setattr(scheduler, "inspect_worktree", inspect)
    scheduler.run(_schedule(tmp_path))
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    value = runtime["jobs"]["value"]
    failure = json.loads(Path(value["failure_path"]).read_text())
    assert invoked == ["prerequisite"]
    assert inspection_count == 2
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["failure_reason"] == (
        "WORKTREE_NOT_AT_EXACT_CLEAN_SCHEDULE_HEAD"
    )
    assert value["process_started"] is False
    assert failure["status"] == "TERMINAL_WORKTREE_STATE_FAILURE"
    assert failure["expected_git_head"] == "a" * 40
    assert failure["observed_git_head"] == "c" * 40


def test_guidance_scheduler_records_process_start_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        scheduler,
        "inspect_worktree",
        lambda worktree, expected_head: {"exact_clean_head": True},
    )

    def fail_start(command: list[str], *, cwd: Path, log: Path):
        del command, cwd, log
        raise OSError("cannot start")

    monkeypatch.setattr(scheduler, "start_logged", fail_start)
    scheduler.run(_schedule(tmp_path))
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    prerequisite = runtime["jobs"]["prerequisite"]
    failure = json.loads(Path(prerequisite["failure_path"]).read_text())
    assert runtime["status"] == "XEDITFLOW_V4_GUIDANCE_SCREEN_TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["job_key"] == "prerequisite"
    assert runtime["first_terminal_failure"]["failure_reason"] == (
        "PROCESS_START_FAILURE"
    )
    assert prerequisite["process_started"] is False
    assert failure["failure_stage"] == "PRE_POPEN_PROCESS_START"
    assert failure["process_started"] is False
