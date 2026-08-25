from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xedit_v4_preflight_job as preflight_job
from scripts.route_a_v3.run_route2_xedit_v4_preflight_job import terminal_kind


def test_atomic_output_is_unique_terminal_even_after_nonzero_process_exit() -> None:
    assert terminal_kind(output_present=True) == "OUTPUT"


def test_absent_output_requires_technical_failure_terminal() -> None:
    assert terminal_kind(output_present=False) == "FAILURE"


def _arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        component="critic",
        python=Path("/formal/python"),
        preflight=Path("/formal/preflight.py"),
        config=Path("/formal/config.json"),
        authorization=Path("/formal/authorization.json"),
        physical_gpu_index=3,
        output=tmp_path / "preflight.json",
        failure=tmp_path / "preflight.failure.json",
        runtime=tmp_path / "runtime.json",
        log=tmp_path / "preflight.log",
        git_head="a" * 40,
    )


def test_nonzero_exit_after_atomic_pause_does_not_publish_second_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)

    class Process:
        pid = 123

        def wait(self) -> int:
            arguments.output.write_text(
                json.dumps(
                    {
                        "status": "XEDITCRITIC_V4_PREFLIGHT_PAUSE",
                        "passed": False,
                        "development_test_outcome_reads": 0,
                        "new_final_evaluation_outcome_reads": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 7

    monkeypatch.setattr(
        preflight_job.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    preflight_job.run(arguments)

    assert arguments.output.is_file()
    assert not arguments.failure.exists()
    runtime = json.loads(arguments.runtime.read_text(encoding="utf-8"))
    assert runtime["status"] == "TERMINAL_COMPLETE"
    assert runtime["preflight_status"] == "XEDITCRITIC_V4_PREFLIGHT_PAUSE"
    assert runtime["preflight_passed"] is False
    assert runtime["preflight_return_code"] == 7
    assert runtime["failure_published"] is False


def test_zero_exit_without_atomic_output_publishes_only_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)

    class Process:
        pid = 456

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        preflight_job.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    with pytest.raises(SystemExit) as error:
        preflight_job.run(arguments)

    assert error.value.code == 1
    assert not arguments.output.exists()
    assert arguments.failure.is_file()
    failure = json.loads(arguments.failure.read_text(encoding="utf-8"))
    assert failure["status"] == "TECHNICAL_FAILURE"
