from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.route_a_v3.run_route2_xedit_v4_cache_job as cache_job
from scripts.route_a_v3.run_route2_xedit_v4_cache_job import terminal_kind


def test_atomic_summary_is_unique_terminal_even_after_nonzero_process_exit() -> None:
    assert terminal_kind(summary_present=True) == "SUMMARY"


def test_absent_summary_requires_technical_failure_terminal() -> None:
    assert terminal_kind(summary_present=False) == "FAILURE"


def _arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        component="critic",
        python=Path("/formal/python"),
        builder=Path("/formal/builder.py"),
        config=Path("/formal/config.json"),
        authorization=Path("/formal/authorization.json"),
        summary=tmp_path / "summary.json",
        failure=tmp_path / "failure.json",
        runtime=tmp_path / "runtime.json",
        log=tmp_path / "builder.log",
        git_head="a" * 40,
    )


def test_nonzero_exit_after_atomic_summary_does_not_publish_second_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)

    class Process:
        pid = 123

        def wait(self) -> int:
            arguments.summary.write_text("{}\n", encoding="utf-8")
            return 7

    monkeypatch.setattr(cache_job.subprocess, "Popen", lambda *args, **kwargs: Process())
    cache_job.run(arguments)

    assert arguments.summary.is_file()
    assert not arguments.failure.exists()
    runtime = json.loads(arguments.runtime.read_text(encoding="utf-8"))
    assert runtime["status"] == "TERMINAL_COMPLETE"
    assert runtime["builder_return_code"] == 7
    assert runtime["failure_published"] is False


def test_zero_exit_without_atomic_summary_publishes_only_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)

    class Process:
        pid = 456

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(cache_job.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(SystemExit) as error:
        cache_job.run(arguments)

    assert error.value.code == 1
    assert not arguments.summary.exists()
    assert arguments.failure.is_file()
    failure = json.loads(arguments.failure.read_text(encoding="utf-8"))
    assert failure["status"] == "TECHNICAL_FAILURE"
