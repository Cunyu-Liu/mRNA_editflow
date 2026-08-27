from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xeditsetflow_s1_screen_scheduler as scheduler


def _writer(path: Path, payload: dict, exit_code: int = 0) -> list[str]:
    source = (
        "import json,pathlib,sys;"
        f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
        f"p.write_text(json.dumps({payload!r})+'\\n');sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def _schedule(tmp_path: Path, *, fail_training: bool = False) -> dict:
    training_queues = []
    for gpu, run_id in enumerate(scheduler.RUN_IDS):
        output = tmp_path / "training" / run_id
        summary = output / "training_summary.json"
        failure = output / "failure.json"
        terminal = failure if fail_training and run_id == "v4_s1_full" else summary
        training_queues.append({"physical_gpu_index": gpu, "jobs": [{
            "job_key": f"training:{run_id}", "run_id": run_id,
            "physical_gpu_index": gpu, "terminal_summary": str(summary),
            "terminal_failure": str(failure), "log_path": str(tmp_path / f"{run_id}.log"),
            "command": _writer(terminal, {"status": "TERMINAL"}, 1 if terminal == failure else 0),
        }]})
    jobs_by_gpu = {gpu: [] for gpu in range(6)}
    index = 0
    for run_id in scheduler.RUN_IDS:
        for checkpoint_pass in scheduler.CHECKPOINT_PASSES:
            output = tmp_path / "validation" / run_id / f"pass_{checkpoint_pass}"
            summary = output / "validation_summary.json"
            failure = output.with_name(output.name + ".failed.json")
            gpu = index % 6
            jobs_by_gpu[gpu].append({
                "job_key": f"validation:{run_id}:pass_{checkpoint_pass}",
                "run_id": run_id, "checkpoint_pass": checkpoint_pass,
                "physical_gpu_index": gpu, "terminal_summary": str(summary),
                "terminal_failure": str(failure), "log_path": str(tmp_path / f"v{index}.log"),
                "command": _writer(summary, {"status": "TERMINAL"}),
            })
            index += 1
    gate = tmp_path / "screen_gate.json"
    return {
        "git_head": "a" * 40, "worktree": str(tmp_path),
        "runtime_manifest": str(tmp_path / "runtime.json"),
        "objective_identity": "XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY",
        "cross_state_candidate_mode_responsibility_weight": .05,
        "training_queues": training_queues,
        "validation_queues": [{"physical_gpu_index": gpu, "jobs": jobs} for gpu, jobs in jobs_by_gpu.items()],
        "adjudication": {"gate_path": str(gate), "failure_path": str(gate.with_name(gate.name + ".failed.json")),
                         "log_path": str(tmp_path / "gate.log"),
                         "command": _writer(gate, {"status": "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"})},
    }


def test_s1_scheduler_runs_two_training_then_eight_validations_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_a, **_k: None)
    schedule = _schedule(tmp_path)
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == scheduler.TERMINAL_STATUS
    assert len(runtime["training_jobs"]) == 2
    assert len(runtime["validation_jobs"]) == 8
    assert all(row["terminal_artifact_kind"] == "SUMMARY" for row in runtime["training_jobs"].values())
    assert all(row["terminal_artifact_kind"] == "SUMMARY" for row in runtime["validation_jobs"].values())
    assert runtime["adjudication"]["status"] == "TERMINAL_COMPLETE"
    assert runtime["first_terminal_failure"] is None


def test_s1_scheduler_first_training_failure_stops_all_pending_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_a, **_k: None)
    scheduler.run(_schedule(tmp_path, fail_training=True))
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == scheduler.FAILURE_STATUS
    assert runtime["first_terminal_failure"]["stage"] == "TRAINING"
    assert all(row["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE" for row in runtime["validation_jobs"].values())
    assert runtime["adjudication"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
    assert not (tmp_path / "screen_gate.json").exists()


def test_s1_scheduler_rejects_summary_when_process_exit_is_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_a, **_k: None)
    schedule = _schedule(tmp_path)
    full_job = schedule["training_queues"][0]["jobs"][0]
    full_job["command"] = _writer(
        Path(full_job["terminal_summary"]),
        {"status": "TERMINAL"},
        exit_code=7,
    )
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == scheduler.FAILURE_STATUS
    assert runtime["first_terminal_failure"] == {
        "stage": "TRAINING",
        "job_key": "training:v4_s1_full",
        "run_id": "v4_s1_full",
        "reason": "JOB_NO_EXACT_SUCCESS_TERMINAL",
        "return_code": 7,
        "terminal_artifact_kind": "SUMMARY",
    }
    assert all(
        row["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"
        for row in runtime["validation_jobs"].values()
    )
    assert runtime["adjudication"]["status"] == "NOT_RUN_AFTER_TERMINAL_FAILURE"


def test_s1_scheduler_preserves_partial_terminal_evidence_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_a, **_k: None)
    schedule = _schedule(tmp_path)
    full_job = schedule["training_queues"][0]["jobs"][0]
    failure_path = Path(full_job["terminal_failure"])
    partial_path = failure_path.with_suffix(failure_path.suffix + ".partial")
    full_job["command"] = _writer(
        partial_path,
        {"status": "ORIGINAL_PARTIAL_EVIDENCE"},
        exit_code=8,
    )
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["status"] == scheduler.FAILURE_STATUS
    assert not failure_path.exists()
    assert json.loads(partial_path.read_text())["status"] == "ORIGINAL_PARTIAL_EVIDENCE"
    assert runtime["first_terminal_failure"]["details"] == {
        "partial_terminal_artifacts": [str(partial_path)]
    }


def test_s1_scheduler_records_adjudication_nonzero_as_first_technical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "inspect_worktree_identity", lambda *_a, **_k: None)
    schedule = _schedule(tmp_path)
    gate = Path(schedule["adjudication"]["gate_path"])
    schedule["adjudication"]["command"] = _writer(
        gate,
        {"status": "XEDITSETFLOW_V4_S1_SCREEN_NO_GO"},
        exit_code=9,
    )
    scheduler.run(schedule)
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    failure = Path(schedule["adjudication"]["failure_path"])
    assert runtime["status"] == scheduler.FAILURE_STATUS
    assert failure.is_file()
    assert runtime["first_terminal_failure"] == {
        "stage": "ADJUDICATION",
        "job_key": "adjudication",
        "reason": "ADJUDICATION_NONZERO_RETURN_CODE",
        "return_code": 9,
        "terminal_artifact_kind": "DOUBLE_TERMINAL",
    }
    assert runtime["adjudication"]["status"] == "TECHNICAL_FAILURE"


def test_s1_scheduler_rejects_incomplete_inventory() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        scheduler.validate_schedule_inventory({"training_queues": [], "validation_queues": []})


def test_s1_scheduler_never_kills_inflight_processes() -> None:
    source = Path(scheduler.__file__).read_text()
    assert "NOT_RUN_AFTER_TERMINAL_FAILURE" in source
    assert ".kill(" not in source and ".terminate(" not in source
