#!/usr/bin/env python3
"""Atomically adjudicate the frozen SetFlow V4 full/single checkpoint package."""

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

from core.route2_xeditsetflow_gate_v4 import (
    adjudicate_setflow_screen_v4,
    technical_failure_gate_v4,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"artifact is not an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _read(arguments.config)
    output = Path(config["screen_gate_output_path"])
    if output.exists():
        raise RuntimeError(f"terminal SetFlow V4 screen gate already exists: {output}")
    failures: list[dict[str, Any]] = []
    for run_id in ("v4_full", "v4_single_mode"):
        training_directory = Path(config["output_root"]) / run_id
        training_summary = training_directory / "training_summary.json"
        training_failure = training_directory / "failure.json"
        terminal_count = int(training_summary.exists()) + int(training_failure.exists())
        if terminal_count != 1:
            raise RuntimeError(f"SetFlow V4 training is not exactly terminal: {run_id}")
        if training_failure.exists():
            failures.append(
                {"stage": "TRAINING", "run_id": run_id, **_read(training_failure)}
            )
    if failures:
        result = technical_failure_gate_v4(failures)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        partial.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, output)
        print(json.dumps(result, sort_keys=True))
        return
    summaries: dict[str, dict[int, dict[str, Any]]] = {
        "v4_full": {},
        "v4_single_mode": {},
    }
    for run_id in ("v4_full", "v4_single_mode"):
        for checkpoint_pass in (4, 6, 8, 10):
            directory = (
                Path(config["validation_output_root"])
                / run_id
                / f"pass_{checkpoint_pass}"
            )
            summary_path = directory / "validation_summary.json"
            failure_path = directory.with_name(directory.name + ".failed.json")
            terminal_count = int(summary_path.exists()) + int(failure_path.exists())
            if terminal_count != 1:
                raise RuntimeError(
                    f"SetFlow V4 checkpoint validation is not exactly terminal: {run_id}/pass{checkpoint_pass}"
                )
            if failure_path.exists():
                failures.append(
                    {
                        "stage": "CHECKPOINT_VALIDATION",
                        "run_id": run_id,
                        "checkpoint_pass": checkpoint_pass,
                        **_read(failure_path),
                    }
                )
            else:
                summaries[run_id][checkpoint_pass] = _read(summary_path)
    result = technical_failure_gate_v4(failures) if failures else adjudicate_setflow_screen_v4(config, summaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
