from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.transition_record_route2_xeditcritic_v403_controls_oom_terminal as transition


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Production scheduler artifacts use sort_keys=True.
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _bind_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime_root" / "runtime.json"
    receipt = tmp_path / "audits" / "oom_terminal.json"
    monkeypatch.setattr(transition, "OLD_RUNTIME_ROOT", runtime.parent)
    monkeypatch.setattr(transition, "OLD_RUNTIME", runtime)
    monkeypatch.setattr(transition, "CANONICAL_RECEIPT", receipt)
    monkeypatch.setattr(transition, "OLD_OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(transition, "OLD_LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(
        transition, "CROSS_ROOT_GATE", tmp_path / "cross_root" / "screen_gate.json"
    )
    monkeypatch.setattr(transition, "LICENSED_WORKTREE", tmp_path / "licensed")
    return runtime, receipt


def _runtime_payload() -> dict:
    jobs = {}
    for index, run_id in enumerate(transition.CONTROL_RUN_IDS):
        output = transition.OLD_OUTPUT_ROOT / run_id
        log = transition.OLD_LOG_ROOT / f"{run_id}.log"
        failed = index < 3
        terminal = output / ("failure.json" if failed else "run_summary.json")
        # Invalid JSON is intentional: the transition must not reopen consumed
        # terminal payloads merely to freeze the runtime-level evidence.
        _write(terminal, "terminal-payload-must-not-be-read")
        jobs[run_id] = {
            "run_id": run_id,
            "physical_gpu_index": index,
            "status": "TECHNICAL_FAILURE" if failed else "TERMINAL_SUMMARY",
            "output_directory": str(output),
            "log_path": str(log),
            "training_attempt_id": (
                "xeditcritic_v4_screen_seed20260907::"
                f"{run_id}::v403_control_recovery_{transition.OLD_HEAD}"
            ),
            "training_git_head": transition.OLD_HEAD,
            "return_code": 1 if failed else 0,
            "terminal_artifact_kind": "FAILURE" if failed else "SUMMARY",
            "finished_unix_seconds": 1.0 + index,
        }
    first_run_id = transition.CONTROL_RUN_IDS[0]
    return {
        "schema_version": transition.RUNTIME_SCHEMA,
        "status": transition.TECHNICAL_TERMINAL_STATUS,
        "scheduler_pid": 4321,
        "historical_full_git_head": transition.HISTORICAL_FULL_GIT_HEAD,
        "historical_c0_git_head": transition.HISTORICAL_C0_GIT_HEAD,
        "current_git_head": transition.OLD_HEAD,
        "runner_git_head": transition.OLD_HEAD,
        "orchestration_git_head": transition.OLD_HEAD,
        "training_code_git_head": transition.OLD_HEAD,
        "training_worktree": str(transition.LICENSED_WORKTREE),
        "ordered_control_run_ids": list(transition.CONTROL_RUN_IDS),
        "jobs": jobs,
        "first_terminal_failure": {
            **transition.EXPECTED_FIRST_FAILURE,
            "output_directory": jobs[first_run_id]["output_directory"],
            "log_path": jobs[first_run_id]["log_path"],
            "worktree_inspection": None,
        },
        "cross_root_adjudication_run": False,
        "full_retrained": False,
        "c0_retrained": False,
        "old_v402_stopped_process_resumed": False,
        "free_memory_gate_applied": False,
        "terminal_artifact_payloads_read_by_scheduler": 0,
        "historical_terminal_payloads_read_before_cross_root": 0,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def _run_bound(runtime: Path, receipt: Path, *, process_alive: bool = False) -> dict:
    return transition.run(
        runtime_path=runtime,
        receipt_path=receipt,
        process_is_alive=lambda _pid: process_alive,
    )


def test_running_runtime_is_read_once_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    payload = _runtime_payload()
    payload["status"] = transition.RUNNING_STATUS
    _write(runtime, payload)
    original_read = transition.read_json
    reads: list[Path] = []

    def counted_read(path: Path) -> dict:
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(transition, "read_json", counted_read)
    result = _run_bound(runtime, receipt)
    assert result == {
        "status": transition.RUNNING_OBSERVED_STATUS,
        "runtime_status": transition.RUNNING_STATUS,
        "receipt_written": False,
    }
    assert reads == [runtime]
    assert not receipt.exists() and not receipt.with_suffix(".json.partial").exists()


def test_exact_technical_terminal_writes_non_authorizing_retry_eligibility_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    _write(runtime, _runtime_payload())
    original_read = transition.read_json
    reads: list[Path] = []

    def counted_read(path: Path) -> dict:
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(transition, "read_json", counted_read)
    result = _run_bound(runtime, receipt)
    assert result["schema_version"] == transition.RECEIPT_SCHEMA
    assert result["status"] == transition.RECEIPT_STATUS
    assert result["old_runtime_status"] == transition.TECHNICAL_TERMINAL_STATUS
    assert result["scheduler_process_gone"] is True
    assert result["technical_failure_run_ids"] == list(
        transition.CONTROL_RUN_IDS[:3]
    )
    assert result["terminal_summary_run_ids"] == list(
        transition.CONTROL_RUN_IDS[3:]
    )
    assert result["first_terminal_failure"] == {
        "run_id": "v4_source_only",
        "reason": "JOB_TERMINAL_FAILURE_ARTIFACT",
        "return_code": 1,
        "terminal_artifact_kind": "FAILURE",
        "output_directory": str(transition.OLD_OUTPUT_ROOT / "v4_source_only"),
        "log_path": str(transition.OLD_LOG_ROOT / "v4_source_only.log"),
    }
    assert result["cross_root_adjudication_run"] is False
    assert result["cross_root_gate_absent"] is True
    assert result["successor_authorized"] is False
    assert result["same_family_retry_authorized"] is False
    assert result["new_independent_retry_eligible"] is True
    assert result["terminal_artifact_payloads_read_by_transition"] == 0
    assert result["development_test_outcome_reads"] == 0
    assert result["new_final_evaluation_outcome_reads"] == 0
    assert reads == [runtime]
    assert json.loads(receipt.read_text(encoding="utf-8")) == result
    assert not receipt.with_suffix(".json.partial").exists()


@pytest.mark.parametrize("existing_partial", [False, True])
def test_existing_receipt_or_partial_refuses_before_runtime_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_partial: bool,
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    occupied = receipt.with_suffix(".json.partial") if existing_partial else receipt
    _write(occupied, {"status": "IMMUTABLE_EXISTING_EVIDENCE"})
    monkeypatch.setattr(
        transition,
        "read_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("runtime was read")),
    )
    with pytest.raises(transition.CriticControlsOomTerminalError, match="already exists"):
        _run_bound(runtime, receipt)


def test_repeated_call_refuses_before_second_runtime_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    _write(runtime, _runtime_payload())
    _run_bound(runtime, receipt)
    monkeypatch.setattr(
        transition,
        "read_json",
        lambda _path: (_ for _ in ()).throw(AssertionError("runtime was reread")),
    )
    with pytest.raises(transition.CriticControlsOomTerminalError, match="already exists"):
        _run_bound(runtime, receipt)


def test_terminal_refuses_while_recorded_scheduler_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    _write(runtime, _runtime_payload())
    with pytest.raises(
        transition.CriticControlsOomTerminalError, match="still alive"
    ):
        _run_bound(runtime, receipt, process_alive=True)
    assert not receipt.exists()


@pytest.mark.parametrize("problem", ["double", "partial"])
def test_double_or_partial_job_terminal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    payload = _runtime_payload()
    output = transition.OLD_OUTPUT_ROOT / transition.CONTROL_RUN_IDS[3]
    if problem == "double":
        _write(output / "failure.json", "second-terminal")
    else:
        _write(output / "run_summary.json.partial", "unfinished-terminal")
    _write(runtime, payload)
    pattern = "unique terminal" if problem == "double" else "partial terminal"
    with pytest.raises(transition.CriticControlsOomTerminalError, match=pattern):
        _run_bound(runtime, receipt)
    assert not receipt.exists()


def test_first_failure_must_remain_stably_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    payload = _runtime_payload()
    payload["first_terminal_failure"]["run_id"] = "v4_edit_metadata_only"
    _write(runtime, payload)
    with pytest.raises(
        transition.CriticControlsOomTerminalError,
        match="first terminal failure changed",
    ):
        _run_bound(runtime, receipt)


def test_technical_failure_requires_failure_artifact_and_nonzero_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    payload = _runtime_payload()
    run_id = "v4_edit_metadata_only"
    row = payload["jobs"][run_id]
    output = Path(row["output_directory"])
    (output / "failure.json").unlink()
    _write(output / "run_summary.json", "contradictory-nonzero-summary")
    row["terminal_artifact_kind"] = "SUMMARY"
    row["return_code"] = 1
    _write(runtime, payload)

    with pytest.raises(
        transition.CriticControlsOomTerminalError,
        match="failure row points to a successful terminal",
    ):
        _run_bound(runtime, receipt)


def test_cross_root_gate_or_failure_evidence_must_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, receipt = _bind_paths(monkeypatch, tmp_path)
    _write(runtime, _runtime_payload())
    _write(transition.CROSS_ROOT_GATE.with_suffix(".failed.json"), {"status": "FAILED"})
    with pytest.raises(
        transition.CriticControlsOomTerminalError,
        match="cross-root gate or adjudication evidence",
    ):
        _run_bound(runtime, receipt)


@pytest.mark.parametrize(
    ("command_line", "expected"),
    [
        (
            "/python run_route2_xeditcritic_v403_control_recovery_scheduler.py "
            f"--schedule /runner_{transition.OLD_HEAD}/schedule.json",
            True,
        ),
        ("/python unrelated_reused_pid.py", False),
        ("", False),
    ],
)
def test_scheduler_process_check_distinguishes_reused_pid(
    monkeypatch: pytest.MonkeyPatch, command_line: str, expected: bool
) -> None:
    monkeypatch.setattr(
        transition.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0 if command_line else 1,
            stdout=command_line,
            stderr="",
        ),
    )
    assert transition.scheduler_process_is_alive(4321) is expected


def test_transition_contains_no_gpu_or_model_execution() -> None:
    source = Path(transition.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "nvidia-smi",
        "import torch",
        "torch.load",
        "model.forward",
        "development_test_path",
        "evaluation_outcomes_accessed",
    ):
        assert forbidden not in source
