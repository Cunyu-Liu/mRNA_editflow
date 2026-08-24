#!/usr/bin/env python3
"""Build V4 A100 equal-wall sensitivity from reconciled final compute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_equal_wall_time_v3 import (
    METHODS_V3,
    equal_wall_time_sensitivity_v3,
)
from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4
from scripts.route_a_v3.build_route2_xeditflow_equal_wall_time_sensitivity_v3 import (
    normalize_matched_compute_rows_v3,
    normalize_search_candidate_rows_v3,
)


MATCHED_COMPUTE_SCORED_JSONL = "MATCHED_COMPUTE_SCORED_JSONL"
SEARCH_CANDIDATE_JSONL = "SEARCH_CANDIDATE_JSONL"


class XEditFlowEqualWallTimeBuildV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowEqualWallTimeBuildV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        bool(rows) and all(isinstance(row, dict) for row in rows),
        f"JSONL input is empty or invalid: {path}",
    )
    return rows


def _source_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [str(row.get("source_key")) for row in rows]
    _require(
        len(result) == 891 and len(set(result)) == 891,
        "V4 equal-wall source manifest is not the frozen 891-source cohort",
    )
    return result


def _validate_scored_compute_rows_v4(
    rows: Sequence[Mapping[str, Any]], *, method: str, source_order: Sequence[str]
) -> None:
    _require(
        len(rows) == 891
        and [str(row.get("source_key")) for row in rows] == list(source_order),
        f"V4 reconciled compute source inventory differs: {method}",
    )
    for row in rows:
        failures = row.get("failure_counters")
        critic_calls = row.get("critic_forwards_by_member")
        _require(
            row.get("schema_version") == "MatchedComputeRecordV4"
            and row.get("terminal_critic_reservation_reconciled") is True
            and row.get("terminal_critic_forwards_are_reserved_pending_scoring")
            is False
            and row.get("trajectory_critic_forwards_preserved_during_reconciliation")
            is True
            and isinstance(critic_calls, list)
            and len(critic_calls) == 3
            and all(int(value) >= 0 for value in critic_calls)
            and all(
                int(row.get(key, -1)) >= 0
                for key in ("trunk_forwards", "mode_forwards", "value_forwards")
            )
            and 0 <= int(row.get("total_forward_equivalents", -1)) <= 320
            and isinstance(failures, Mapping)
            and all(int(value) == 0 for value in failures.values()),
            f"V4 reconciled compute accounting differs: {method}",
        )


def build_equal_wall_time_sensitivity_v4(
    config: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    timing_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    closed_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_equal_wall_time_config.v4",
        "unexpected V4 equal-wall config schema",
    )
    seed = int(config.get("base_flow_training_seed", -1))
    _require(seed in BASE_FLOW_SEEDS_V4, "V4 equal-wall SetFlow seed differs")
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 equal-wall config accessed protected outcomes",
    )
    method_configs = config.get("methods")
    _require(
        isinstance(method_configs, Mapping) and set(method_configs) == METHODS_V3,
        "V4 equal-wall method inventory differs",
    )
    source_order = _source_order(
        list(source_rows)
        if source_rows is not None
        else _jsonl(Path(str(config["source_manifest_path"])))
    )
    normalized_rows: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for method in sorted(METHODS_V3):
        method_config = method_configs[method]
        format_name = str(method_config["timing_format"])
        expected_format = (
            SEARCH_CANDIDATE_JSONL
            if method == "strongest_matched_baseline"
            else MATCHED_COMPUTE_SCORED_JSONL
        )
        _require(
            format_name == expected_format,
            f"V4 equal-wall timing format differs: {method}",
        )
        rows = (
            list(timing_rows[method])
            if timing_rows is not None
            else _jsonl(Path(str(method_config["timing_path"])))
        )
        if method == "strongest_matched_baseline":
            normalized_rows[method] = normalize_search_candidate_rows_v3(
                rows, method=method, source_order=source_order
            )
        else:
            _validate_scored_compute_rows_v4(
                rows, method=method, source_order=source_order
            )
            normalized_rows[method] = normalize_matched_compute_rows_v3(
                rows, method=method, source_order=source_order
            )
        closed = (
            closed_results[method]
            if closed_results is not None
            else _json(Path(str(method_config["closed_summary_path"])))
        )
        _require(
            closed.get("status") == "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE"
            and str(closed.get("method_id")) == method
            and int(closed.get("base_flow_training_seed", -1)) == seed
            and closed.get("development_test_outcomes_accessed_after_atomic_test")
            is False
            and closed.get("new_final_evaluation_outcomes_accessed") is False,
            f"V4 equal-wall closed evidence differs: {method}",
        )
        results[method] = {
            **closed,
            "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        inputs[method] = {
            "timing_path": str(method_config["timing_path"]),
            "timing_format": format_name,
            "closed_summary_path": str(method_config["closed_summary_path"]),
        }
    legacy = equal_wall_time_sensitivity_v3(
        normalized_rows,
        results,
        source_order=source_order,
        base_flow_training_seed=20260904,
    )
    return {
        **legacy,
        "schema_version": "route_a_v3_route2_xeditflow_equal_wall_time_sensitivity.v4",
        "status": "XEDITFLOW_V4_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE",
        "base_flow_training_seed": seed,
        "timing_inputs": inputs,
        "five_v4_methods_use_terminal_scoring_reconciled_compute": True,
        "all_network_forwards_separately_charged": True,
        "matched_compute_schema": "MatchedComputeRecordV4",
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(
        not arguments.output.exists(),
        f"V4 equal-wall output already exists: {arguments.output}",
    )
    result = build_equal_wall_time_sensitivity_v4(_json(arguments.config))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "common_source_keys"},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
