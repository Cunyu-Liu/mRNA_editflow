#!/usr/bin/env python3
"""Adjudicate the eight frozen Critic V4 screen terminal artifacts once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditcritic_gate_v4 import evaluate_xeditcritic_v4_screen


class XEditCriticAdjudicationV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditCriticAdjudicationV4Error(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not a JSON object: {path}")
    return payload


def run(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["screen_gate_output"])
    _require(not output.exists(), "Critic V4 screen gate already exists")
    partial = output.with_suffix(output.suffix + ".partial")
    _require(not partial.exists(), "partial Critic V4 screen gate already exists")
    root = Path(config["output_root"])
    summaries: dict[str, dict[str, Any]] = {}
    technical_failures: dict[str, dict[str, Any]] = {}
    for row in config["required_screen_runs"]:
        run_id = str(row["run_id"])
        summary_path = root / run_id / "run_summary.json"
        failure_path = root / run_id / "failure.json"
        _require(summary_path.exists() != failure_path.exists(), f"{run_id} is not exactly one terminal state")
        if failure_path.exists():
            technical_failures[run_id] = _load(failure_path)
        else:
            summaries[run_id] = _load(summary_path)
    reference = _load(Path(config["c3_read_once_reference_adjudication"]))
    _require(reference.get("status") == "C3_V4_REFERENCE_READ_ONCE_COMPLETE", "C3 read-once reference adjudication is absent")
    _require(reference.get("terminal_summaries_read_count") == 5, "C3 terminal summaries were not read exactly once as a five-run package")
    _require(int(reference.get("development_test_outcome_reads", -1)) == 0 and int(reference.get("new_final_evaluation_outcome_reads", -1)) == 0, "C3 reference adjudication protected reads are nonzero")
    if technical_failures:
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_screen_gate.v1",
            "status": "XEDITCRITIC_V4_SCREEN_NO_GO",
            "passed": False,
            "reason": "ONE_OR_MORE_FROZEN_SCREEN_RUNS_TERMINATED_WITH_TECHNICAL_FAILURE",
            "technical_failure_run_ids": sorted(technical_failures),
            "technical_failures": technical_failures,
            "confirmation_authorized": False,
            "development_test_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
    else:
        result = evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=float(reference["c3_reference_task_macro_spearman"]),
            preflight=_load(Path(config["preflight_output"])),
        )
        result["c3_reference_adjudication_path"] = config["c3_read_once_reference_adjudication"]
        result["c3_terminal_summaries_reread_by_v4_gate"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _load(arguments.config)
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
