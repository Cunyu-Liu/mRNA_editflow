#!/usr/bin/env python3
"""Evaluate V4 open-support recovery without zero-filling unknown outcomes."""

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


class XEditFlowOpenGenerationV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowOpenGenerationV4Error(message)


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


def validate_open_generation_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_open_generation_config.v4",
        "unexpected V4 open-generation config",
    )
    _require(
        config.get("pool_assignment") == "DEVELOPMENT"
        and config.get("candidate_support_mode") == "OPEN_GENERATED_SUPPORT"
        and config.get("undefined_outcome_policy") == "UNKNOWN_NOT_ZERO",
        "V4 open-generation cohort or support policy differs",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) == 20260912
        and float(config.get("kappa", -1)) in {0.0, 0.5, 1.0}
        and float(config.get("temperature", -1)) in {0.5, 1.0}
        and float(config.get("beta_max", -1)) in {0.5, 1.0, 2.0}
        and bool(str(config.get("method_id", ""))),
        "V4 open-generation combination differs",
    )
    _require(
        int(config.get("measured_top_k", -1)) == 10
        and config.get("critic_self_score_used_for_ranking") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 open-generation metric or protected-input policy differs",
    )


def evaluate_open_generation_v4(
    config: Mapping[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    measured_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_open_generation_config_v4(config)
    sources = load_source_manifest(Path(config["source_eligibility_manifest"]))
    candidate_rows = (
        _jsonl(Path(config["candidate_path"]))
        if candidates is None
        else candidates
    )
    measured = (
        _jsonl(Path(config["measured_neighborhood_path"]))
        if measured_rows is None
        else measured_rows
    )
    _require(
        all(
            str(row.get("method_id")) == str(config["method_id"])
            and int(row.get("base_flow_training_seed", -1)) == 20260912
            and float(row.get("kappa", -1)) == float(config["kappa"])
            and float(row.get("temperature", -1))
            == float(config["temperature"])
            and float(row.get("beta_max", -1)) == float(config["beta_max"])
            and row.get("critic_self_score_used_for_generation_or_selection")
            is False
            for row in candidate_rows
        ),
        "V4 open-generation candidate combination differs",
    )
    validate_measured_pool(measured, "DEVELOPMENT", "CLOSED")
    generation = evaluate_generation(sources, candidate_rows)
    measured_metrics = measured_neighborhood_metrics(
        sources,
        candidate_rows,
        measured,
        k=10,
        candidate_support_mode="OPEN_GENERATED_SUPPORT",
    )
    _require(
        generation["method_id"] == str(config["method_id"]),
        "V4 open-generation method identity differs",
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_metrics.v4",
        "status": "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": generation["method_id"],
        "base_flow_training_seed": 20260912,
        "kappa": float(config["kappa"]),
        "temperature": float(config["temperature"]),
        "beta_max": float(config["beta_max"]),
        "source_count": generation["source_count"],
        "source_macro_candidate_recovery": measured_metrics[
            "source_macro_candidate_recovery_rate"
        ],
        "source_macro_top_k_recovery": measured_metrics[
            "source_macro_measured_top_k_recovery_at_k"
        ],
        "source_macro_unique_candidate_rate": generation[
            "source_macro_unique_candidate_rate"
        ],
        "hard_legality_rate": generation["hard_legality_rate"],
        "edit_budget_violation_count": generation["edit_budget_violation_count"],
        "candidate_budget_violation_count": generation[
            "candidate_budget_violation_count"
        ],
        "closed_ndcg_defined_count": measured_metrics[
            "source_closed_measured_ndcg_defined_count"
        ],
        "closed_ndcg_is_not_defined_on_open_support": measured_metrics[
            "source_macro_closed_measured_ndcg_at_k"
        ]
        is None,
        "unknown_generated_candidates_are_zero_gain": False,
        "ranking_input": "GENERATION_SCORE_NOT_CRITIC_SELF_SCORE",
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"V4 open metric output exists: {arguments.output}")
    result = evaluate_open_generation_v4(_json(arguments.config))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
