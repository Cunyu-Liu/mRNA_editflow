#!/usr/bin/env python3
"""Consume the eight terminal Critic V4 failures exactly once for V4.0.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


RUN_IDS = (
    "c0_v4",
    "v4_full",
    "v4_source_only",
    "v4_edit_metadata_only",
    "v4_no_candidate_sequence",
    "v4_candidate_bundle_permutation",
    "v4_no_cross",
    "v4_no_moe",
)


class V402FailureDiagnosisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V402FailureDiagnosisError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"failure artifact is not an object: {path}")
    return payload


def consumption_marker_path(output: Path) -> Path:
    return Path(str(output) + ".consumption_started.json")


def _publish_consumption_marker(
    output: Path,
    failure_paths: Mapping[str, Path],
) -> Path:
    marker = consumption_marker_path(output)
    payload = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v402_failure_diagnosis_"
            "consumption_started.v1"
        ),
        "status": "XEDITCRITIC_V402_FAILURE_CONSUMPTION_STARTED",
        "diagnosis_output": str(output),
        "terminal_failures": {
            run_id: str(path) for run_id, path in failure_paths.items()
        },
        "terminal_payload_content_included": False,
        "automatic_retry_if_diagnosis_absent": False,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise V402FailureDiagnosisError(
            "Critic V4.0.2 failure consumption already started; automatic reread "
            f"is forbidden: {marker}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return marker


def _terminal_failure_paths(screen_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for run_id in RUN_IDS:
        directory = screen_root / run_id
        failure = directory / "failure.json"
        summary = directory / "run_summary.json"
        _require(
            failure.is_file() and not summary.exists(),
            f"Critic arm is not failure-only terminal: {run_id}",
        )
        paths[run_id] = failure
    return paths


def run(*, screen_root: Path, output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"V4.0.2 diagnosis already exists: {output}")
    _require(
        not output.with_suffix(output.suffix + ".partial").exists(),
        f"V4.0.2 diagnosis partial already exists: {output}",
    )
    failure_paths = _terminal_failure_paths(screen_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = _publish_consumption_marker(output, failure_paths)
    payloads = {run_id: _read(path) for run_id, path in failure_paths.items()}
    result = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v402_failure_diagnosis_read_once.v1"
        ),
        "status": "XEDITCRITIC_V402_FAILURE_DIAGNOSIS_READ_ONCE_COMPLETE",
        "screen_root": str(screen_root),
        "run_ids": list(RUN_IDS),
        "terminal_failure_payloads_read_count": len(payloads),
        "terminal_summary_artifacts_present": 0,
        "valid_validation_performance_summary_present": False,
        "consumption_marker": str(marker),
        "automatic_retry_authorized": False,
        "failure_payloads": payloads,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(screen_root=arguments.screen_root, output=arguments.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
