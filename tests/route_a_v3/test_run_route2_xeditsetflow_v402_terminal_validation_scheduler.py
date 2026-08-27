from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.route_a_v3.run_route2_xeditsetflow_v402_terminal_validation_scheduler as scheduler


def _writer(path: Path, payload: dict[str, object]) -> list[str]:
    source = (
        "import json,pathlib;"
        f"p=pathlib.Path({str(path)!r});p.parent.mkdir(parents=True,exist_ok=True);"
        f"p.write_text(json.dumps({payload!r})+'\\n')"
    )
    return [sys.executable, "-c", source]


def test_setflow_only_scheduler_publishes_terminal_gate_without_critic_read(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.json"
    summary = tmp_path / "validation/v4_full/pass_4/validation_summary.json"
    failure = tmp_path / "validation/v4_full/pass_4.failed.json"
    gate = tmp_path / "screen_gate.json"
    schedule = {
        "git_head": "a" * 40,
        "source_screen_head": "b" * 40,
        "experiment_head": "c" * 40,
        "worktree": str(tmp_path),
        "runtime_manifest": str(runtime),
        "validation_queues": [
            {
                "physical_gpu_index": 0,
                "jobs": [
                    {
                        "job_key": "v4_full:pass_4",
                        "run_id": "v4_full",
                        "checkpoint_pass": 4,
                        "terminal_summary": str(summary),
                        "terminal_failure": str(failure),
                        "log_path": str(tmp_path / "validation.log"),
                        "command": _writer(summary, {"status": "TERMINAL"}),
                    }
                ],
            }
        ],
        "setflow_adjudication": {
            "gate_path": str(gate),
            "log_path": str(tmp_path / "adjudication.log"),
            "command": _writer(gate, {"status": "XEDITSETFLOW_V4_SCREEN_NO_GO"}),
        },
    }
    scheduler.run(schedule)

    payload = json.loads(runtime.read_text(encoding="utf-8"))
    assert payload["status"] == "XEDITSETFLOW_V402_VALIDATION_AND_GATE_TERMINAL"
    assert payload["setflow_adjudication"]["gate_present"] is True
    assert payload["critic_failure_payload_reads"] == 0
    assert payload["active_performance_output_read"] is False
    assert payload["development_test_outcome_reads"] == 0
    assert payload["new_final_evaluation_outcome_reads"] == 0
