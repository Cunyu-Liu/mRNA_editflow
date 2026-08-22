#!/usr/bin/env python3
"""Build the frozen A100 common-prefix equal-wall-time sensitivity artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_equal_wall_time_v3 import (
    EQUAL_WALL_TIME_SCOPE_V3,
    METHODS_V3,
    equal_wall_time_sensitivity_v3,
)


MATCHED_COMPUTE_JSONL = "MATCHED_COMPUTE_JSONL"
SEARCH_CANDIDATE_JSONL = "SEARCH_CANDIDATE_JSONL"


class XEditFlowEqualWallTimeBuildV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowEqualWallTimeBuildV3Error(message)


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
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL is empty or invalid: {path}")
    return rows


def _positive(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{label} is not finite-positive")
    return result


def _zero(value: Any, label: str) -> None:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == 0.0,
        f"{label} is not zero",
    )


def frozen_source_order_v3(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    source_order = [str(row.get("source_key")) for row in rows]
    _require(
        len(source_order) == 891 and len(set(source_order)) == 891,
        "equal-wall source manifest is not the frozen 891-source cohort",
    )
    return source_order


def _normalized_row(row: Mapping[str, Any], *, method: str) -> dict[str, Any]:
    accelerator = row.get("source_cuda_device_name", row.get("cuda_device_name"))
    _require(
        isinstance(accelerator, str) and "A100" in accelerator.upper(),
        f"equal-wall raw timing is not from A100: {method}",
    )
    return {
        "source_key": str(row["source_key"]),
        "source_wall_time_seconds": _positive(
            row.get("source_equal_wall_time_seconds"),
            f"equal-wall raw source time {method}",
        ),
        "wall_time_scope": str(row.get("source_equal_wall_time_scope")),
        "accelerator_name": accelerator,
        "peak_vram_mb": _positive(
            row.get("source_equal_wall_peak_vram_mb"),
            f"equal-wall raw peak VRAM {method}",
        ),
    }


def normalize_matched_compute_rows_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    source_order: Sequence[str],
) -> list[dict[str, Any]]:
    _require(len(rows) == 891, f"matched-compute timing row count differs: {method}")
    _require(
        [str(row.get("source_key")) for row in rows] == list(source_order),
        f"matched-compute timing source order differs: {method}",
    )
    normalized = [_normalized_row(row, method=method) for row in rows]
    _require(
        {row["wall_time_scope"] for row in normalized} == {EQUAL_WALL_TIME_SCOPE_V3},
        f"matched-compute timing scope differs: {method}",
    )
    return normalized


def normalize_search_candidate_rows_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    source_order: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    observed_order: list[str] = []
    for row in rows:
        source_key = str(row.get("source_key"))
        if source_key not in grouped:
            grouped[source_key] = []
            observed_order.append(source_key)
        grouped[source_key].append(row)
    _require(observed_order == list(source_order), f"search timing source order differs: {method}")
    normalized = []
    for source_key in source_order:
        source_rows = grouped[source_key]
        first = source_rows[0]
        _require(
            first.get("source_equal_wall_time_scope") == EQUAL_WALL_TIME_SCOPE_V3,
            f"search first-row timing scope differs: {method}/{source_key}",
        )
        normalized.append(_normalized_row(first, method=method))
        for row in source_rows[1:]:
            _zero(
                row.get("source_equal_wall_time_seconds"),
                f"search repeated source timing {method}/{source_key}",
            )
            _zero(
                row.get("source_equal_wall_peak_vram_mb"),
                f"search repeated source peak VRAM {method}/{source_key}",
            )
            _require(
                row.get("source_equal_wall_time_scope") == "COUNTED_ON_SOURCE_FIRST_ROW",
                f"search repeated source timing scope differs: {method}/{source_key}",
            )
    return normalized


def build_equal_wall_time_sensitivity_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_equal_wall_time_config.v1",
        "unexpected equal-wall config schema",
    )
    _require(
        config.get("development_test_outcomes_accessed") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "equal-wall config accessed protected outcomes",
    )
    method_configs = config.get("methods")
    _require(
        isinstance(method_configs, Mapping) and set(method_configs) == METHODS_V3,
        "equal-wall config method inventory differs",
    )
    source_order = frozen_source_order_v3(_jsonl(Path(str(config["source_manifest_path"]))))
    time_rows = {}
    closed_results = {}
    timing_inputs = {}
    for method in sorted(METHODS_V3):
        method_config = method_configs[method]
        timing_path = Path(str(method_config["timing_path"]))
        timing_format = str(method_config["timing_format"])
        raw_rows = _jsonl(timing_path)
        expected_format = (
            SEARCH_CANDIDATE_JSONL
            if method == "strongest_matched_baseline"
            else MATCHED_COMPUTE_JSONL
        )
        _require(timing_format == expected_format, f"equal-wall timing format differs: {method}")
        if timing_format == MATCHED_COMPUTE_JSONL:
            normalized = normalize_matched_compute_rows_v3(
                raw_rows, method=method, source_order=source_order
            )
        else:
            normalized = normalize_search_candidate_rows_v3(
                raw_rows, method=method, source_order=source_order
            )
        time_rows[method] = normalized
        closed_results[method] = _json(Path(str(method_config["closed_summary_path"])))
        timing_inputs[method] = {
            "timing_path": str(timing_path),
            "timing_format": timing_format,
            "closed_summary_path": str(method_config["closed_summary_path"]),
        }
    result = equal_wall_time_sensitivity_v3(
        time_rows,
        closed_results,
        source_order=source_order,
        base_flow_training_seed=int(config["base_flow_training_seed"]),
    )
    result["timing_inputs"] = timing_inputs
    result["timing_scope_definition"] = (
        "METHOD_REQUIRED_CANDIDATE_GENERATION_INCLUDING_REPLAY_WHERE_APPLICABLE_"
        "AND_SELECTION_SCORING;_POSTHOC_DIAGNOSTIC_SCORING_EXCLUDED"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"equal-wall output already exists: {args.output}")
    result = build_equal_wall_time_sensitivity_v3(_json(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "common_source_keys"}, sort_keys=True))


if __name__ == "__main__":
    main()
