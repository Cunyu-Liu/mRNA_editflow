#!/usr/bin/env python3
"""Evaluate frozen method scores on the common closed measured neighborhood."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from core.route2_closed_neighborhood_v3 import closed_neighborhood_metrics_v1


class XEditFlowClosedScoresV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowClosedScoresV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def evaluate_closed_method_scores_v3(
    config: Mapping[str, Any],
    measured_rows: list[Mapping[str, Any]],
    score_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_closed_score_config.v1", "unexpected closed-score config")
    _require(config.get("pool_assignment") == "DEVELOPMENT" and config.get("split") == "VALIDATION", "closed-score cohort differs")
    _require(config.get("analysis_unit") == "SOURCE", "closed-score analysis unit differs")
    _require(config.get("undefined_source_policy") == "EXCLUDE_NOT_ZERO_FILL", "closed-score undefined policy differs")
    _require(config.get("score_transform") == "SOURCEWISE_EXP_SHIFTED_MAX", "closed-score transform differs")
    method_id = str(config["method_id"])
    scores = {}
    for row in score_rows:
        key = (str(row["source_key"]), str(row["candidate_sequence"]))
        _require(key not in scores, f"closed method score is duplicated: {key}")
        value = float(row["frozen_method_score"])
        _require(math.isfinite(value), f"closed method score is nonfinite: {key}")
        scores[key] = value
    _require(bool(scores), "closed method score table is empty")
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    measured_keys = set()
    for row in measured_rows:
        _require(row.get("pool_assignment") == "DEVELOPMENT" and row.get("split") == "VALIDATION", "closed-score measured row differs")
        key = (str(row["source_key"]), str(row["candidate_sequence"]))
        _require(key not in measured_keys, f"closed measured candidate is duplicated: {key}")
        _require(key in scores, f"frozen method score does not cover measured candidate: {key}")
        measured_keys.add(key)
        by_source[key[0]].append(row)
    _require(bool(by_source), "closed-score measured neighborhood is empty")
    _require(set(scores) == measured_keys, "frozen method score table does not exactly match measured candidates")
    normalized_rows = []
    for source_key, rows in sorted(by_source.items()):
        source_scores = [scores[(source_key, str(row["candidate_sequence"]))] for row in rows]
        maximum = max(source_scores)
        positive = [math.exp(max(value - maximum, -80.0)) for value in source_scores]
        _require(all(math.isfinite(value) and value > 0.0 for value in positive), f"closed transformed score is invalid: {source_key}")
        for row, raw, weight in zip(rows, source_scores, positive):
            normalized_rows.append(
                {
                    **dict(row),
                    "terminal_probability": weight,
                    "frozen_method_score": raw,
                    "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
                }
            )
    metrics = closed_neighborhood_metrics_v1(normalized_rows)
    return {
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood.v3",
        "status": "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE",
        "method_id": method_id,
        "base_flow_training_seed": int(config["base_flow_training_seed"]),
        **metrics,
        "measured_candidate_count": len(normalized_rows),
        "ranking_input": "FROZEN_METHOD_SCORE",
        "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
        "undefined_sources_are_not_filled_with_zero": True,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"closed-score output exists: {args.output}")
    config = _json(args.config)
    result = evaluate_closed_method_scores_v3(
        config,
        _jsonl(Path(config["measured_neighborhood_path"])),
        _jsonl(Path(config["score_table_path"])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
