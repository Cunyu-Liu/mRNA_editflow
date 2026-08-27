from __future__ import annotations

import json
import inspect
import runpy
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import scripts.route_a_v3.launch_route2_xeditflow_v4_final_after_guidance_screen as launcher
import scripts.route_a_v3.run_route2_xeditflow_v4_final_scheduler as final_scheduler
from scripts.route_a_v3.launch_route2_xeditflow_v4_final_after_guidance_screen import build_schedule
from scripts.route_a_v3.run_route2_xeditflow_v4_final_scheduler import run


ROOT = Path(__file__).resolve().parents[2]


def _prepared_manifest(tmp_path: Path) -> tuple[dict, Path]:
    helpers = runpy.run_path(
        str(
            ROOT
            / "tests/route_a_v3/"
            "test_prepare_route2_xeditflow_final_generation_configs_v4.py"
        )
    )
    payload = helpers["_payload"]()
    config_root = tmp_path / "configs"
    payload["runtime_config_root"] = str(config_root)
    helpers["write_final_generation_configs_v4"](payload, config_root)
    return json.loads((config_root / "manifest.json").read_text()), config_root


def test_v4_final_launcher_builds_exact_three_seed_job_graph(tmp_path: Path) -> None:
    manifest, config_root = _prepared_manifest(tmp_path)
    schedule = build_schedule(
        manifest,
        config_root=config_root,
        log_root=tmp_path / "logs",
        failure_root=tmp_path / "failures",
        runtime_manifest=tmp_path / "runtime.json",
        current_head="a" * 40,
        experiment_head="b" * 40,
        guidance_runner_head="c" * 40,
        diagnostic_peak_plus_two_gib_mib=30_000,
        free_memory_mib={gpu: 40_000 for gpu in range(6)},
    )
    assert [row["queue_key"] for row in schedule["prerequisite_queues"]] == [
        "value_seed_20260913",
        "value_seed_20260914",
        "strongest_timing",
    ]
    assert [row["queue_key"] for row in schedule["seed_chains"]] == [
        "seed_20260912",
        "seed_20260913",
        "seed_20260914",
    ]
    assert all(len(row["jobs"]) == 29 for row in schedule["seed_chains"])
    assert len(schedule["finalization_jobs"]) == 2
    all_jobs = [
        job
        for row in schedule["prerequisite_queues"] + schedule["seed_chains"]
        for job in row["jobs"]
    ] + schedule["finalization_jobs"]
    assert len(all_jobs) == 98
    assert len({job["job_key"] for job in all_jobs}) == 98
    assert all(
        not set(job["physical_gpu_indices"]) - set(range(6)) for job in all_jobs
    )
    seed_12 = schedule["seed_chains"][0]["jobs"]
    assert seed_12[0]["job_key"].endswith("strongest_adapter")
    assert seed_12[-1]["job_key"].endswith("final_evidence")
    assert seed_12[-1]["success_path"].endswith(
        "/final_evidence/seed_manifest_row.json"
    )
    assert seed_12[-1]["failure_path"].endswith("/final_evidence.failed.json")
    assert schedule["finalization_jobs"][-1]["job_key"] == (
        "adjudicate_final_comparison"
    )
    assert schedule["development_test_outcomes_accessed_after_atomic_test"] is False
    assert schedule["new_final_evaluation_outcome_reads"] == 0
    assert schedule["free_memory_gate_applied"] is False
    assert schedule["diagnostic_peak_plus_two_gib_mib"] == 30_000


def test_v4_final_launcher_uses_own_repo_and_does_not_memory_gate() -> None:
    assert launcher.WORKTREE == Path(launcher.__file__).resolve().parents[2]
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert '"free_memory_gate_applied": False' in source
    assert "all(free_memory[gpu]" not in source


def test_v4_final_can_reuse_derived_protocol_and_distinct_preflight_heads() -> None:
    parameters = inspect.signature(launcher.run).parameters
    assert {
        "protocol_path",
        "guidance_runtime_path",
        "critic_preflight_path",
        "critic_preflight_head",
        "setflow_preflight_path",
        "setflow_preflight_head",
        "execution_runtime_root",
        "execution_log_root",
    } <= set(parameters)
    assert parameters["protocol_path"].default == launcher.PROTOCOL
    assert parameters["critic_preflight_head"].default is None
    assert parameters["setflow_preflight_head"].default is None


def _job(
    tmp_path: Path,
    key: str,
    *,
    succeed: bool = True,
) -> dict:
    success = tmp_path / f"{key}.success"
    failure = tmp_path / f"{key}.failure"
    if succeed:
        code = (
            "from pathlib import Path; "
            f"Path({str(success)!r}).write_text('ok', encoding='utf-8')"
        )
    else:
        code = "raise SystemExit(7)"
    return {
        "job_key": key,
        "command": [sys.executable, "-c", code],
        "physical_gpu_indices": [],
        "success_path": str(success),
        "failure_path": str(failure),
        "log_path": str(tmp_path / f"{key}.log"),
    }


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex-test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=repo,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return repo, head


def _scheduler_fixture(tmp_path: Path, *, fail_seed: bool = False) -> dict:
    repo, head = _git_repo(tmp_path)
    return {
        "git_head": head,
        "experiment_head": "b" * 40,
        "guidance_runner_head": "c" * 40,
        "worktree": str(repo),
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "prerequisite_queues": [
            {"queue_key": "prerequisite", "jobs": [_job(tmp_path, "pre")]}
        ],
        "seed_chains": [
            {
                "queue_key": f"seed_{seed}",
                "jobs": [
                    _job(
                        tmp_path,
                        f"seed_{seed}",
                        succeed=not (fail_seed and seed == 20260913),
                    )
                ],
            }
            for seed in (20260912, 20260913, 20260914)
        ],
        "finalization_jobs": [
            _job(tmp_path, "compose"),
            _job(tmp_path, "adjudicate"),
        ],
    }


def _clean_identity(schedule: dict) -> dict:
    return {
        "head_return_code": 0,
        "observed_git_head": schedule["git_head"],
        "status_return_code": 0,
        "worktree_clean": True,
    }


def test_v4_final_scheduler_closes_only_after_all_three_seed_chains(
    tmp_path: Path,
) -> None:
    schedule = _scheduler_fixture(tmp_path)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL"
    assert all(
        row["status"] == "TERMINAL_COMPLETE" for row in runtime["jobs"].values()
    )
    assert runtime["active_performance_output_read"] is False


def test_v4_final_scheduler_preserves_failure_and_does_not_adjudicate(
    tmp_path: Path,
) -> None:
    schedule = _scheduler_fixture(tmp_path, fail_seed=True)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE"
    assert runtime["jobs"]["seed_20260913"]["status"] == "TERMINAL_FAILURE"
    assert runtime["jobs"]["compose"]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert runtime["jobs"]["adjudicate"]["status"] == (
        "NOT_RUN_AFTER_TERMINAL_FAILURE"
    )
    assert runtime["first_terminal_failure"]["job_key"] == "seed_20260913"
    assert Path(runtime["jobs"]["seed_20260913"]["failure_path"]).is_file()


def test_v4_final_scheduler_stops_all_pending_after_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _scheduler_fixture(tmp_path)
    first_failure = _job(tmp_path, "first_failure", succeed=False)
    first_failure["command"] = ["test-job", "first_failure"]
    already_flying = _job(tmp_path, "already_flying")
    already_flying["command"] = ["test-job", "already_flying"]
    schedule["prerequisite_queues"] = [
        {
            "queue_key": "fails_first",
            "jobs": [
                first_failure,
                _job(tmp_path, "never_after_failure"),
            ],
        },
        {
            "queue_key": "already_flying",
            "jobs": [
                already_flying,
                _job(tmp_path, "never_after_flying"),
            ],
        },
    ]
    real_event = threading.Event
    real_lock = threading.Lock
    flying_started = real_event()
    failure_published = real_event()
    coordination_failures: list[str] = []
    original_write_atomic = final_scheduler.write_atomic

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

    def event_factory() -> TrackingPackageFailureEvent:
        return TrackingPackageFailureEvent()

    def coordinated_write_atomic(path: Path, payload: dict) -> None:
        original_write_atomic(path, payload)
        if (
            path == Path(schedule["runtime_manifest"])
            and payload.get("first_terminal_failure") is not None
        ):
            failure_published.set()

    launched: list[list[str]] = []

    class CoordinatedProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.returncode: int | None = None
            if command == ["test-job", "already_flying"]:
                flying_started.set()

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is None
            if self.command == ["test-job", "first_failure"]:
                if not flying_started.wait(timeout=5):
                    coordination_failures.append(
                        "already-flying job did not start"
                    )
                self.returncode = 7
                return self.returncode
            if self.command == ["test-job", "already_flying"]:
                if not failure_published.wait(timeout=5):
                    coordination_failures.append(
                        "first failure was not published"
                    )
                Path(already_flying["success_path"]).write_text(
                    "ok", encoding="utf-8"
                )
                self.returncode = 0
                return self.returncode
            coordination_failures.append(
                f"unexpected job launch: {self.command!r}"
            )
            self.returncode = 9
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

    def coordinated_popen(command: list[str], **kwargs: object) -> CoordinatedProcess:
        del kwargs
        launched.append(list(command))
        return CoordinatedProcess(list(command))

    monkeypatch.setattr(final_scheduler, "write_atomic", coordinated_write_atomic)
    monkeypatch.setattr(
        final_scheduler,
        "threading",
        types.SimpleNamespace(
            Lock=lock_factory,
            Event=event_factory,
            Thread=threading.Thread,
        ),
    )
    monkeypatch.setattr(
        final_scheduler,
        "observe_worktree_identity",
        lambda worktree: _clean_identity(schedule),
    )
    monkeypatch.setattr(final_scheduler.subprocess, "Popen", coordinated_popen)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert not coordination_failures
    assert flying_started.is_set()
    assert failure_published.is_set()
    assert launched == [
        ["test-job", "first_failure"],
        ["test-job", "already_flying"],
    ] or launched == [
        ["test-job", "already_flying"],
        ["test-job", "first_failure"],
    ]
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["job_key"] == "first_failure"
    assert runtime["jobs"]["already_flying"]["status"] == "TERMINAL_COMPLETE"
    for key in (
        "never_after_failure",
        "never_after_flying",
        "seed_20260912",
        "seed_20260913",
        "seed_20260914",
        "compose",
        "adjudicate",
    ):
        assert runtime["jobs"][key]["status"] == (
            "NOT_RUN_AFTER_TERMINAL_FAILURE"
        )
        assert not Path(runtime["jobs"][key]["success_path"]).exists()


def test_v4_final_scheduler_popen_error_is_terminal_and_stops_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _scheduler_fixture(tmp_path)
    launch_attempts: list[list[str]] = []

    def fail_popen(command: list[str], **kwargs: object) -> None:
        del kwargs
        launch_attempts.append(list(command))
        raise OSError("deterministic process launch failure")

    monkeypatch.setattr(
        final_scheduler,
        "observe_worktree_identity",
        lambda worktree: _clean_identity(schedule),
    )
    monkeypatch.setattr(final_scheduler.subprocess, "Popen", fail_popen)
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    failed_job = schedule["prerequisite_queues"][0]["jobs"][0]
    failure = json.loads(Path(failed_job["failure_path"]).read_text())

    assert launch_attempts == [failed_job["command"]]
    assert runtime["status"] == "XEDITFLOW_V4_FINAL_COMPARISON_TECHNICAL_FAILURE"
    assert runtime["first_terminal_failure"]["job_key"] == failed_job["job_key"]
    assert runtime["first_terminal_failure"]["failure_stage"] == (
        "JOB_PROCESS_START"
    )
    assert runtime["jobs"][failed_job["job_key"]]["status"] == (
        "TERMINAL_FAILURE"
    )
    assert runtime["jobs"][failed_job["job_key"]]["failure_stage"] == (
        "JOB_PROCESS_START"
    )
    assert failure["status"] == "TERMINAL_JOB_PROCESS_START_FAILURE"
    assert failure["exception_type"] == "OSError"
    assert failure["error"] == "deterministic process launch failure"
    assert failure["job_process_started"] is False
    assert all(row["status"] != "RUNNING" for row in runtime["jobs"].values())
    assert all(
        row["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
        for key, row in runtime["jobs"].items()
        if key != failed_job["job_key"]
    )


def test_v4_final_scheduler_refuses_dirty_worktree_before_next_job(
    tmp_path: Path,
) -> None:
    schedule = _scheduler_fixture(tmp_path)
    repo = Path(schedule["worktree"])
    first = _job(tmp_path, "dirties_after_success")
    first["command"] = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({first['success_path']!r}).write_text('ok', encoding='utf-8'); "
            f"Path({str(repo / 'tracked.txt')!r}).write_text('dirty\\n', encoding='utf-8')"
        ),
    ]
    refused = _job(tmp_path, "refused_dirty_job")
    schedule["prerequisite_queues"] = [
        {"queue_key": "dirty_sequence", "jobs": [first, refused]}
    ]
    run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["jobs"]["dirties_after_success"]["status"] == (
        "TERMINAL_COMPLETE"
    )
    assert runtime["jobs"]["refused_dirty_job"]["status"] == "TERMINAL_FAILURE"
    assert runtime["jobs"]["refused_dirty_job"]["failure_stage"] == (
        "PRE_JOB_WORKTREE_IDENTITY"
    )
    assert runtime["first_terminal_failure"]["job_key"] == "refused_dirty_job"
    assert runtime["first_terminal_failure"]["observed_git_head"] == (
        schedule["git_head"]
    )
    failure = json.loads(Path(refused["failure_path"]).read_text())
    assert failure["status"] == "TERMINAL_WORKTREE_IDENTITY_FAILURE"
    assert failure["error"] == "WORKTREE_DIRTY"
    assert failure["job_process_started"] is False
    assert not Path(refused["success_path"]).exists()


def test_v4_final_prelaunch_gpu_inventory_failure_is_sibling_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "a" * 40
    experiment_head = "b" * 40
    guidance_head = "c" * 40
    critic_head = "d" * 40
    setflow_head = "e" * 40
    for name in ("python", "preparer.py", "scheduler.py"):
        (tmp_path / name).write_text("\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "PYTHON", tmp_path / "python")
    monkeypatch.setattr(launcher, "PREPARER", tmp_path / "preparer.py")
    monkeypatch.setattr(launcher, "SCHEDULER", tmp_path / "scheduler.py")

    guidance_output = tmp_path / "guidance_output"
    guidance_output.mkdir()
    (guidance_output / "guidance_screen_gate.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "route_a_v3_route2_xeditflow_v4_guidance_screen_gate.v1"
                ),
                "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
                "base_flow_training_seed": 20260912,
                "combination_count": 18,
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "guidance_screen_output_root": str(guidance_output),
                "runtime_config_root": str(tmp_path / "runtime_configs" / "guidance"),
            }
        ),
        encoding="utf-8",
    )
    guidance_runtime = tmp_path / "guidance_runtime.json"
    guidance_runtime.write_text(
        json.dumps(
            {
                "status": "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
                "git_head": guidance_head,
                "experiment_head": experiment_head,
                "development_test_reopened": False,
                "development_test_outcomes_accessed_after_atomic_test": False,
                "new_final_evaluation_outcome_reads": 0,
            }
        ),
        encoding="utf-8",
    )
    critic_preflight = tmp_path / "critic_preflight.json"
    critic_preflight.write_text(
        json.dumps(
            {
                "status": "XEDITCRITIC_V4_PREFLIGHT_PASS",
                "git_head": critic_head,
                "selected_peak_allocated_gib": 10.0,
            }
        ),
        encoding="utf-8",
    )
    setflow_preflight = tmp_path / "setflow_preflight.json"
    setflow_preflight.write_text(
        json.dumps(
            {
                "status": "XEDITSETFLOW_V4_PREFLIGHT_PASS",
                "git_head": setflow_head,
                "peak_memory_allocated_gib": 11.0,
            }
        ),
        encoding="utf-8",
    )

    def fake_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        stdout = head + "\n" if arguments == ["git", "rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    monkeypatch.setattr(launcher, "command", fake_command)
    inventory_error = launcher.XEditFlowV4GpuInventoryError(
        "nvidia-smi failed",
        return_code=9,
        stdout="probe stdout\n",
        stderr="probe stderr\n",
    )

    def fail_inventory() -> dict[int, int]:
        raise inventory_error

    existing_runtime_root = tmp_path / "existing_final_execution"
    existing_runtime_root.mkdir()
    existing_probe_called = False

    def unexpected_existing_runtime_probe() -> dict[int, int]:
        nonlocal existing_probe_called
        existing_probe_called = True
        return {gpu: 40_000 for gpu in range(6)}

    monkeypatch.setattr(
        launcher, "gpu_free_memory_mib", unexpected_existing_runtime_probe
    )
    with pytest.raises(
        launcher.XEditFlowV4FinalLaunchError,
        match="final runtime already exists",
    ):
        launcher.run(
            head,
            experiment_head,
            guidance_head,
            tmp_path / "not_read_for_existing_runtime.json",
            protocol_path=protocol,
            guidance_runtime_path=guidance_runtime,
            critic_preflight_path=critic_preflight,
            critic_preflight_head=critic_head,
            setflow_preflight_path=setflow_preflight,
            setflow_preflight_head=setflow_head,
            execution_runtime_root=existing_runtime_root,
            execution_log_root=tmp_path / "existing_logs",
        )
    assert existing_probe_called is False
    assert not (tmp_path / "existing_final_execution.failed.json").exists()

    monkeypatch.setattr(launcher, "gpu_free_memory_mib", fail_inventory)
    runtime_root = tmp_path / "final_execution"
    with pytest.raises(launcher.XEditFlowV4GpuInventoryError):
        launcher.run(
            head,
            experiment_head,
            guidance_head,
            tmp_path / "not_read_before_inventory.json",
            protocol_path=protocol,
            guidance_runtime_path=guidance_runtime,
            critic_preflight_path=critic_preflight,
            critic_preflight_head=critic_head,
            setflow_preflight_path=setflow_preflight,
            setflow_preflight_head=setflow_head,
            execution_runtime_root=runtime_root,
            execution_log_root=tmp_path / "logs",
        )
    assert not runtime_root.exists()
    evidence_path = tmp_path / "final_execution.failed.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == (
        "XEDITFLOW_V4_FINAL_PRELAUNCH_GPU_INVENTORY_FAILURE"
    )
    assert evidence["runtime_root_created"] is False
    assert evidence["command"] == list(launcher.GPU_INVENTORY_COMMAND)
    assert evidence["return_code"] == 9
    assert evidence["stdout"] == "probe stdout\n"
    assert evidence["stderr"] == "probe stderr\n"
    assert evidence["git_head"] == head
    assert evidence["experiment_head"] == experiment_head
    assert evidence["guidance_runner_head"] == guidance_head
    assert evidence["error"] == "nvidia-smi failed"
    assert evidence["automatic_retry_attempted"] is False
    assert evidence["cpu_fallback_used"] is False
    assert evidence["development_test_reopened"] is False
    assert evidence["development_test_outcomes_accessed_after_atomic_test"] is False
    assert evidence["new_final_evaluation_outcome_reads"] == 0

    original_evidence = evidence_path.read_text(encoding="utf-8")
    probe_called = False

    def unexpected_inventory_probe() -> dict[int, int]:
        nonlocal probe_called
        probe_called = True
        return {gpu: 40_000 for gpu in range(6)}

    monkeypatch.setattr(
        launcher, "gpu_free_memory_mib", unexpected_inventory_probe
    )
    with pytest.raises(
        launcher.XEditFlowV4FinalLaunchError,
        match="prelaunch failure evidence already exists",
    ):
        launcher.run(
            head,
            experiment_head,
            guidance_head,
            tmp_path / "still_not_read.json",
            protocol_path=protocol,
            guidance_runtime_path=guidance_runtime,
            critic_preflight_path=critic_preflight,
            critic_preflight_head=critic_head,
            setflow_preflight_path=setflow_preflight,
            setflow_preflight_head=setflow_head,
            execution_runtime_root=runtime_root,
            execution_log_root=tmp_path / "logs",
        )
    assert probe_called is False
    assert evidence_path.read_text(encoding="utf-8") == original_evidence
