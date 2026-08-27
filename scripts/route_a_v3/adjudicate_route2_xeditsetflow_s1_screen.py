#!/usr/bin/env python3
"""Atomically adjudicate the isolated SetFlow V4 S1 S1 full/single checkpoint package."""

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

from core.route2_xeditsetflow_gate_s1 import (
    adjudicate_setflow_screen_s1,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"artifact is not an object: {path}")
    return payload


def _write_atomic_terminal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"terminal SetFlow V4 S1 screen gate already exists: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"partial SetFlow V4 S1 screen gate already exists: {partial}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _read(arguments.config)
    output = Path(config["screen_gate_output_path"])
    if output.exists():
        raise RuntimeError(f"terminal SetFlow V4 S1 screen gate already exists: {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"partial SetFlow V4 S1 screen gate already exists: {partial}")
    failures: list[dict[str, Any]] = []
    for run_id in ("v4_s1_full", "v4_s1_single_mode"):
        training_directory = Path(config["output_root"]) / run_id
        training_summary = training_directory / "training_summary.json"
        training_failure = training_directory / "failure.json"
        terminal_count = int(training_summary.exists()) + int(training_failure.exists())
        if terminal_count != 1:
            raise RuntimeError(f"SetFlow V4 S1 training is not exactly terminal: {run_id}")
        if training_failure.exists():
            failures.append(
                {"stage": "TRAINING", "run_id": run_id, **_read(training_failure)}
            )
    if failures:
        raise RuntimeError(
            "SetFlow V4 S1 training package is technically incomplete; "
            "scientific adjudication is forbidden"
        )
    summaries: dict[str, dict[int, dict[str, Any]]] = {
        "v4_s1_full": {},
        "v4_s1_single_mode": {},
    }
    for run_id in ("v4_s1_full", "v4_s1_single_mode"):
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
                    f"SetFlow V4 S1 checkpoint validation is not exactly terminal: {run_id}/pass{checkpoint_pass}"
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
    if failures:
        raise RuntimeError(
            "SetFlow V4 S1 Validation package is technically incomplete; "
            "scientific adjudication is forbidden"
        )
    result = adjudicate_setflow_screen_s1(config, summaries)
    _write_atomic_terminal(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
