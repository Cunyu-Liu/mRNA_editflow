from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xedit_v4_postscreen_adjudication_scheduler as coordinator


def _writer(path: Path, payload: dict[str, object], *, exit_code: int = 0) -> list[str]:
    source = (
        "import json,pathlib,sys;"
        f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
        f"p.write_text(json.dumps({payload!r})+'\\n');sys.exit({exit_code})"
    )
    return [sys.executable, "-c", source]


def test_postscreen_accepts_atomic_gates_despite_late_nonzero_exit(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    critic_gate = tmp_path / "critic_gate.json"
    setflow_gate = tmp_path / "setflow_gate.json"
    full_summary = tmp_path / "validation/full/pass_4/validation_summary.json"
    full_failure = tmp_path / "validation/full/pass_4.failed.json"
    schedule = {
        "git_head": "a" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "critic_adjudication": {
            "gate_path": str(critic_gate),
            "log_path": str(tmp_path / "critic.log"),
            "command": _writer(
                critic_gate,
                {"status": "XEDITCRITIC_V4_SCREEN_NO_GO"},
                exit_code=7,
            ),
        },
        "validation_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "full:pass_4",
                        "run_id": "full",
                        "checkpoint_pass": 4,
                        "terminal_summary": str(full_summary),
                        "terminal_failure": str(full_failure),
                        "log_path": str(tmp_path / "full.log"),
                        "command": _writer(
                            full_summary, {"status": "TERMINAL_COMPLETE"}
                        ),
                    }
                ],
            }
        ],
        "setflow_adjudication": {
            "gate_path": str(setflow_gate),
            "log_path": str(tmp_path / "setflow.log"),
            "command": _writer(
                setflow_gate,
                {"status": "XEDITSETFLOW_V4_SCREEN_NO_GO"},
                exit_code=9,
            ),
        },
    }
    coordinator.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "V4_POSTSCREEN_ALL_TERMINAL"
    assert payload["critic_adjudication"]["status"] == "TERMINAL_COMPLETE"
    assert payload["critic_adjudication"]["return_code"] == 7
    assert payload["setflow_adjudication"]["status"] == "TERMINAL_COMPLETE"
    assert payload["setflow_adjudication"]["return_code"] == 9
    assert payload["active_performance_output_read"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0


def test_gate_terminal_requires_published_gate() -> None:
    assert coordinator.gate_terminal(gate_present=True) == "TERMINAL_COMPLETE"
    assert coordinator.gate_terminal(gate_present=False) == "TECHNICAL_FAILURE"
