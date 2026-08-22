#!/usr/bin/env python3
"""Evaluate XEditFlow V3 open-support recovery without zero-filling outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
    measured_neighborhood_metrics,
    validate_measured_pool,
)


class XEditFlowOpenGenerationV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowOpenGenerationV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def evaluate_open_generation_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_open_generation_config.v1", "unexpected open-generation config schema")
    _require(config.get("pool_assignment") == "DEVELOPMENT", "open-generation pool differs")
    _require(config.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT", "open-generation support mode differs")
    _require(config.get("undefined_outcome_policy") == "UNKNOWN_NOT_ZERO", "open-generation unknown-outcome policy differs")
    sources = load_source_manifest(Path(config["source_eligibility_manifest"]))
    candidates = _jsonl(Path(config["candidate_path"]))
    measured_rows = _jsonl(Path(config["measured_neighborhood_path"]))
    validate_measured_pool(measured_rows, "DEVELOPMENT", "CLOSED")
    generation = evaluate_generation(sources, candidates)
    measured = measured_neighborhood_metrics(
        sources,
        candidates,
        measured_rows,
        k=int(config.get("measured_top_k", 10)),
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_metrics.v3",
        "status": "XEDITFLOW_V3_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": generation["method_id"],
        "source_count": generation["source_count"],
        "source_macro_candidate_recovery": measured["source_macro_candidate_recovery_rate"],
        "source_macro_top_k_recovery": measured["source_macro_measured_top_k_recovery_at_k"],
        "source_macro_unique_candidate_rate": generation["source_macro_unique_candidate_rate"],
        "hard_legality_rate": generation["hard_legality_rate"],
        "edit_budget_violation_count": generation["edit_budget_violation_count"],
        "candidate_budget_violation_count": generation["candidate_budget_violation_count"],
        "closed_ndcg_defined_count": measured["source_closed_measured_ndcg_defined_count"],
        "closed_ndcg_is_not_defined_on_open_support": measured["source_macro_closed_measured_ndcg_at_k"] is None,
        "unknown_generated_candidates_are_zero_gain": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"open-generation output exists: {args.output}")
    result = evaluate_open_generation_v3(_json(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
