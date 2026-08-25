#!/usr/bin/env python3
"""Read the five terminal C3 artifacts once and freeze the V4 reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class C3V4ReferenceAdjudicationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3V4ReferenceAdjudicationError(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"terminal artifact is not an object: {path}")
    return payload


def _consumption_marker_path(output: Path) -> Path:
    return Path(str(output) + ".consumption_started.json")


def _publish_consumption_marker(
    output: Path,
    terminal_paths: Mapping[str, tuple[str, Path]],
) -> Path:
    """Irreversibly mark the logical read before opening any terminal payload."""

    marker = _consumption_marker_path(output)
    payload = {
        "schema_version": (
            "route_a_v3_route2_xeditcritic_v3_c3_v4_reference_"
            "consumption_started.v1"
        ),
        "status": "C3_V4_REFERENCE_TERMINAL_CONSUMPTION_STARTED",
        "reference_output": str(output),
        "terminal_artifacts": {
            run_id: {"terminal_kind": kind, "path": str(path)}
            for run_id, (kind, path) in terminal_paths.items()
        },
        "terminal_payload_content_included": False,
        "automatic_retry_if_reference_absent": False,
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise C3V4ReferenceAdjudicationError(
            "C3 terminal consumption already started without a published "
            f"reference; automatic reread is forbidden: {marker}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return marker


def _terminal_kind(run_directory: Path) -> tuple[str, Path]:
    summary = run_directory / "run_summary.json"
    failure = run_directory / "failure.json"
    _require(
        summary.exists() != failure.exists(),
        f"C3 run is not exactly one terminal state: {run_directory.name}",
    )
    return ("SUMMARY", summary) if summary.exists() else ("FAILURE", failure)


def _protected_reads_zero(payload: Mapping[str, Any], *, run_id: str) -> None:
    _require(
        payload.get("development_test_outcomes_accessed") is False
        and payload.get("new_final_evaluation_outcomes_accessed") is False,
        f"C3 terminal artifact reports a protected outcome read: {run_id}",
    )


def _task_macro_spearman(summary: Mapping[str, Any], *, label: str) -> float:
    metrics = summary.get("final_validation")
    _require(isinstance(metrics, Mapping), f"{label} final Validation metrics are absent")
    value = metrics.get("task_macro_spearman")
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} task-macro Spearman is invalid",
    )
    return float(value)


def adjudicate_c3_v4_reference(
    config: Mapping[str, Any],
    terminal_payloads: Mapping[str, tuple[str, Mapping[str, Any]]],
    fallback_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    run_ids = tuple(str(value) for value in config["launch_barrier"]["c3_required_run_ids"])
    _require(
        set(terminal_payloads) == set(run_ids) and len(terminal_payloads) == 5,
        "C3 read-once package is not the exact five frozen runs",
    )
    terminal_rows: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        kind, payload = terminal_payloads[run_id]
        _require(kind in {"SUMMARY", "FAILURE"}, f"unknown C3 terminal kind: {run_id}")
        _protected_reads_zero(payload, run_id=run_id)
        if kind == "SUMMARY":
            _require(
                payload.get("status") == "TERMINAL_SCREEN_ARM_COMPLETE",
                f"C3 summary is not a terminal screen result: {run_id}",
            )
        else:
            _require(
                payload.get("status") == "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
                f"C3 failure artifact is not terminal: {run_id}",
            )
        terminal_rows[run_id] = {
            "terminal_kind": kind,
            "status": payload.get("status"),
            "control_mode": payload.get("control_mode"),
            "candidate_bundle_permutation": bool(
                payload.get("candidate_bundle_permutation", False)
            ),
            "task_macro_spearman": (
                _task_macro_spearman(payload, label=run_id)
                if kind == "SUMMARY"
                else None
            ),
        }

    full_kind, full_payload = terminal_payloads["c3"]
    if full_kind == "SUMMARY":
        reference = _task_macro_spearman(full_payload, label="C3 full")
        reference_source = "C3_FULL_TERMINAL_VALIDATION"
        fallback_used = False
    else:
        _require(
            fallback_summary is not None,
            "C3 full failed technically and the predeclared fallback is absent",
        )
        _protected_reads_zero(fallback_summary, run_id="predeclared C2 fallback")
        reference = _task_macro_spearman(
            fallback_summary, label="predeclared C2 fallback"
        )
        reference_source = "PREDECLARED_HIGHEST_VALID_V3_FULL_DIAGNOSTIC_C2"
        fallback_used = True

    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_c3_v4_reference_read_once.v1",
        "status": "C3_V4_REFERENCE_READ_ONCE_COMPLETE",
        "terminal_summaries_read_count": 5,
        "terminal_run_ids": list(run_ids),
        "terminal_rows": terminal_rows,
        "c3_full_technical_failure": full_kind == "FAILURE",
        "predeclared_fallback_used": fallback_used,
        "reference_source": reference_source,
        "c3_reference_task_macro_spearman": reference,
        "c3_confirmation_authorized": False,
        "c3_development_test_authorized": False,
        "c3_refit_loso_or_guidance_authorized": False,
        "c3_terminal_artifacts_retained": True,
        "development_test_outcome_reads": 0,
        "new_final_evaluation_outcome_reads": 0,
    }


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["c3_read_once_reference_adjudication"])
    _require(not output.exists(), f"C3 V4 reference already exists: {output}")
    root = Path(config["c3_reference"]["preferred_terminal_summary"]).parents[1]
    run_ids = tuple(str(value) for value in config["launch_barrier"]["c3_required_run_ids"])

    # Resolve all terminal paths before opening any of the five payloads.  A
    # partially terminal package therefore cannot consume the one-time read.
    terminal_paths = {
        run_id: _terminal_kind(root / run_id) for run_id in run_ids
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_consumption_marker(output, terminal_paths)
    terminal_payloads = {
        run_id: (kind, _read(path))
        for run_id, (kind, path) in terminal_paths.items()
    }
    fallback = None
    if terminal_paths["c3"][0] == "FAILURE":
        fallback = _read(
            Path(config["c3_reference"]["predeclared_fallback_terminal_summary"])
        )
    result = adjudicate_c3_v4_reference(config, terminal_payloads, fallback)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _read(arguments.config)
    print(json.dumps(run(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
