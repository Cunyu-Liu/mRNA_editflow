#!/usr/bin/env python3
"""Adjudicate the eight frozen Critic V4 screen terminal artifacts once."""

from __future__ import annotations

import argparse
import json
import os
import re
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


def _authorized_training_heads(
    config: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    *,
    expected_runner_head: str,
) -> dict[str, str]:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_runner_head) is not None,
        "expected Critic V4 runner Git HEAD is invalid",
    )
    required_run_ids = {
        str(row["run_id"]) for row in config["required_screen_runs"]
    }
    heads: dict[str, str] = {}
    required_barriers = (
        "all_five_c3_jobs_terminal",
        "c3_terminal_summaries_read_exactly_once",
        "a100_current_head_focused_tests_passed",
        "a100_current_head_v332_tests_passed",
        "bottom_six_cache_terminal_complete",
        "formal_parameter_preflight_passed",
        "formal_memory_preflight_passed",
        "cache_online_equivalence_passed",
    )
    for run_id, summary in summaries.items():
        authorization_path = Path(str(summary.get("launch_authorization_path", "")))
        _require(
            authorization_path.is_file(),
            f"{run_id} launch authorization is absent",
        )
        authorization = _load(authorization_path)
        barriers = authorization.get("barriers", {})
        _require(
            authorization.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_screen_launch_authorization.v1"
            and authorization.get("status")
            == "XEDITCRITIC_V4_SCREEN_LAUNCH_AUTHORIZED"
            and authorization.get("authorized_git_head") == expected_runner_head
            and set(authorization.get("authorized_run_ids", []))
            == required_run_ids
            and run_id in authorization.get("authorized_run_ids", [])
            and all(barriers.get(name) is True for name in required_barriers)
            and int(authorization.get("development_test_outcome_reads", -1)) == 0
            and int(authorization.get("new_final_evaluation_outcome_reads", -1))
            == 0,
            f"{run_id} launch authorization or runner-HEAD binding changed",
        )
        heads[run_id] = expected_runner_head
    return heads


def run(
    config: dict[str, Any], *, expected_runner_head: str
) -> dict[str, Any]:
    output = Path(config["screen_gate_output"])
    _require(not output.exists(), "Critic V4 screen gate already exists")
    partial = output.with_suffix(output.suffix + ".partial")
    _require(not partial.exists(), "partial Critic V4 screen gate already exists")
    technical_failure_output = output.with_name(output.stem + ".failed.json")
    _require(
        not technical_failure_output.exists(),
        "Critic V4 screen technical-failure artifact already exists",
    )
    technical_failure_partial = technical_failure_output.with_suffix(
        technical_failure_output.suffix + ".partial"
    )
    _require(
        not technical_failure_partial.exists(),
        "partial Critic V4 screen technical-failure artifact already exists",
    )
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
    if technical_failures:
        result = {
            "schema_version": (
                "route_a_v3_route2_xeditcritic_v4_screen_technical_failure.v1"
            ),
            "status": "XEDITCRITIC_V4_SCREEN_TECHNICAL_FAILURE",
            "passed": False,
            "reason": "ONE_OR_MORE_FROZEN_SCREEN_RUNS_TERMINATED_WITH_TECHNICAL_FAILURE",
            "technical_failure_run_ids": sorted(technical_failures),
            "technical_failures": technical_failures,
            "confirmation_authorized": False,
            "development_test_authorized": False,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
        technical_failure_output.parent.mkdir(parents=True, exist_ok=True)
        technical_failure_partial.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(technical_failure_partial, technical_failure_output)
        raise XEditCriticAdjudicationV4Error(
            "Critic V4 screen terminated technically; scientific gate was not written"
        )
    else:
        reference = _load(Path(config["c3_read_once_reference_adjudication"]))
        _require(reference.get("status") == "C3_V4_REFERENCE_READ_ONCE_COMPLETE", "C3 read-once reference adjudication is absent")
        _require(reference.get("terminal_summaries_read_count") == 5, "C3 terminal summaries were not read exactly once as a five-run package")
        _require(int(reference.get("development_test_outcome_reads", -1)) == 0 and int(reference.get("new_final_evaluation_outcome_reads", -1)) == 0, "C3 reference adjudication protected reads are nonzero")
        expected_training_git_heads = _authorized_training_heads(
            config,
            summaries,
            expected_runner_head=expected_runner_head,
        )
        result = evaluate_xeditcritic_v4_screen(
            config,
            summaries,
            c3_reference_spearman=float(reference["c3_reference_task_macro_spearman"]),
            preflight=_load(Path(config["preflight_output"])),
            expected_training_git_heads=expected_training_git_heads,
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
    parser.add_argument("--expected-runner-head", required=True)
    arguments = parser.parse_args()
    config = _load(arguments.config)
    print(
        json.dumps(
            run(config, expected_runner_head=arguments.expected_runner_head),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
