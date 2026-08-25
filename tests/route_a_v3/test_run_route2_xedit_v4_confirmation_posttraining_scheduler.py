from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_confirmation_posttraining_scheduler as scheduler


def _writer(path: Path | None, *, exit_code: int) -> list[str]:
    action = ""
    if path is not None:
        action = (
            f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
            "p.write_text('{}\\n');"
        )
    source = f"import pathlib,sys;{action}sys.exit({exit_code})"
    return [sys.executable, "-c", source]


def test_confirmation_posttraining_preserves_atomic_gate_and_validation_failure(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    validation_summary = tmp_path / "validation/pass_4/validation_summary.json"
    validation_failure = tmp_path / "validation/pass_4.failed.json"
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
                        "command": _writer(None, exit_code=7),
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
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
    validation = payload["validation_jobs"]["setflow:20260912:pass_4"]
    assert validation["terminal_artifact_kind"] == "FAILURE"
    assert validation_failure.is_file()
    adjudication = payload["adjudications"]["critic"]
    assert adjudication["terminal_artifact_kind"] == "SUMMARY"
    assert adjudication["return_code"] == 9
    assert not critic_failure.exists()
    assert payload["active_performance_output_read"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0


def test_confirmation_posttraining_publishes_adjudication_failure_when_gate_absent(
    tmp_path: Path,
) -> None:
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
    assert payload["status"] == "V4_CONFIRMATION_POSTTRAINING_ALL_TERMINAL"
    assert payload["adjudications"]["setflow"]["terminal_artifact_kind"] == "FAILURE"
    assert failure.is_file()
    assert not gate.exists()


def test_confirmation_validation_summary_survives_late_nonzero_exit(
    tmp_path: Path,
) -> None:
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


def test_exact_terminal_rejects_absent_or_double_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failure = tmp_path / "failure.json"
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
    summary.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) == "SUMMARY"
    failure.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
