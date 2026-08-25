from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.route_a_v3.run_route2_xedit_v4_preflight_sequence as sequence


def _config(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    jobs: list[dict[str, object]] = []
    for order, component in enumerate(("critic", "setflow")):
        jobs.append(
            {
                "component": component,
                "order": order,
                "physical_gpu_index": 0,
                "command": [f"{component}-command"],
                "output": str(tmp_path / f"{component}.json"),
                "failure": str(tmp_path / f"{component}.failure.json"),
                "runtime": str(tmp_path / f"{component}.runtime.json"),
                "wrapper_log": str(tmp_path / f"{component}.wrapper.log"),
            }
        )
    path = tmp_path / "sequence_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "route_a_v3_route2_xedit_v4_preflight_sequence_config.v1",
                "git_head": "a" * 40,
                "physical_gpu_index": 0,
                "jobs": jobs,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path, jobs


def test_sequence_runs_critic_then_setflow_and_continues_after_component_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _config(tmp_path)
    calls: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        component = command[0].removesuffix("-command")
        calls.append(component)
        job = next(value for value in jobs if value["component"] == component)
        terminal_field = "output" if component == "critic" else "failure"
        Path(str(job[terminal_field])).write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0 if component == "critic" else 7)

    monkeypatch.setattr(sequence, "current_git_head", lambda: "a" * 40)
    monkeypatch.setattr(sequence.subprocess, "run", fake_run)
    runtime = tmp_path / "sequence.runtime.json"
    result = sequence.run(config, runtime, git_head="a" * 40)

    assert calls == ["critic", "setflow"]
    assert result["status"] == "TERMINAL_COMPLETE"
    assert [value["terminal_kind"] for value in result["completed"]] == [
        "OUTPUT",
        "FAILURE",
    ]
    assert json.loads(runtime.read_text(encoding="utf-8")) == result


def test_sequence_config_rejects_reordered_or_cross_gpu_jobs(tmp_path: Path) -> None:
    config, jobs = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["jobs"] = list(reversed(jobs))
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(sequence.XEditV4PreflightSequenceError, match="order"):
        sequence.load_and_validate_config(config, git_head="a" * 40)

    payload["jobs"] = jobs
    payload["jobs"][1]["physical_gpu_index"] = 3
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(sequence.XEditV4PreflightSequenceError, match="GPU changed"):
        sequence.load_and_validate_config(config, git_head="a" * 40)


def test_sequence_main_publishes_scheduler_failure_without_fabricating_component_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, jobs = _config(tmp_path)
    runtime = tmp_path / "sequence.runtime.json"
    failure = tmp_path / "sequence.failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sequence",
            "--config",
            str(config),
            "--runtime",
            str(runtime),
            "--failure",
            str(failure),
            "--git-head",
            "a" * 40,
        ],
    )
    monkeypatch.setattr(
        sequence,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sequence.XEditV4PreflightSequenceError("synthetic failure")
        ),
    )
    with pytest.raises(sequence.XEditV4PreflightSequenceError, match="synthetic"):
        sequence.main()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["status"] == "TECHNICAL_FAILURE"
    assert payload["development_test_outcome_reads"] == 0
    assert not any(Path(str(job["output"])).exists() for job in jobs)
    assert not any(Path(str(job["failure"])).exists() for job in jobs)
