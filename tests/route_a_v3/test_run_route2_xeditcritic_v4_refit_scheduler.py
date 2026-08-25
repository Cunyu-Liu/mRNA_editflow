from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xeditcritic_v4_refit_scheduler as scheduler


def _writer(path: Path | None, payload: dict[str, object], exit_code: int) -> list[str]:
    action = ""
    if path is not None:
        action = (
            f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
            f"p.write_text(json.dumps({payload!r})+'\\n');"
        )
    return [
        sys.executable,
        "-c",
        f"import json,pathlib,sys;{action}sys.exit({exit_code})",
    ]


def test_refit_scheduler_preserves_terminals_and_authorizes_loso(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    failure = tmp_path / "run/failure.json"
    manifest = tmp_path / "refit_manifest.json"
    adjudication_failure = tmp_path / "refit_adjudication.failed.json"
    schedule = {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "refit:20260908:v4_full",
                        "seed": 20260908,
                        "summary_path": str(summary),
                        "failure_path": str(failure),
                        "log_path": str(tmp_path / "refit.log"),
                        "command": _writer(summary, {}, 7),
                    }
                ],
            }
        ],
        "adjudication": {
            "manifest_path": str(manifest),
            "failure_path": str(adjudication_failure),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(
                manifest,
                {
                    "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE",
                    "loso_authorized": True,
                },
                9,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_ALL_TERMINAL_LOSO_AUTHORIZED"
    assert payload["jobs"]["refit:20260908:v4_full"]["terminal_artifact_kind"] == "SUMMARY"
    assert payload["jobs"]["refit:20260908:v4_full"]["return_code"] == 7
    assert payload["adjudication"]["terminal_artifact_kind"] == "SUMMARY"
    assert payload["adjudication"]["return_code"] == 9
    assert payload["adjudication"]["loso_authorized"] is True
    assert not failure.exists()
    assert not adjudication_failure.exists()


def test_refit_scheduler_closes_missing_job_and_no_go(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "run/run_summary.json"
    failure = tmp_path / "run/failure.json"
    manifest = tmp_path / "refit_manifest.json"
    schedule = {
        "git_head": "b" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "gpu_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "refit:20260908:v4_full",
                        "seed": 20260908,
                        "summary_path": str(summary),
                        "failure_path": str(failure),
                        "log_path": str(tmp_path / "refit.log"),
                        "command": _writer(None, {}, 3),
                    }
                ],
            }
        ],
        "adjudication": {
            "manifest_path": str(manifest),
            "failure_path": str(tmp_path / "adjudication.failed.json"),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(
                manifest,
                {
                    "status": "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_NO_GO",
                    "loso_authorized": False,
                },
                0,
            ),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITCRITIC_V4_REFIT_TERMINAL_NO_GO"
    assert payload["jobs"]["refit:20260908:v4_full"]["terminal_artifact_kind"] == "FAILURE"
    assert failure.is_file()
    assert payload["adjudication"]["loso_authorized"] is False


def test_refit_exact_terminal_rejects_double_artifact(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    failure = tmp_path / "failure.json"
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
    summary.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) == "SUMMARY"
    failure.write_text("{}\n", encoding="utf-8")
    assert scheduler.exact_terminal(str(summary), str(failure)) is None
