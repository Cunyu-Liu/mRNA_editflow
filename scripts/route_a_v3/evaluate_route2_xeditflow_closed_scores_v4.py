#!/usr/bin/env python3
"""Evaluate frozen V4 control scores on the common measured neighborhood."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4
from scripts.route_a_v3.evaluate_route2_xeditflow_closed_scores_v3 import (
    evaluate_closed_method_scores_v3,
)
from scripts.route_a_v3.score_route2_xeditflow_closed_controls_v4 import (
    CLOSED_CRITIC_METHODS_V4,
)


CLOSED_SCORED_METHODS_V4 = CLOSED_CRITIC_METHODS_V4 | {
    "strongest_matched_baseline"
}
STRONGEST_CLOSED_SCORE_PRODUCER_SEED_V4 = 20260904


class XEditFlowClosedScoresV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowClosedScoresV4Error(message)


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


def evaluate_closed_method_scores_v4(
    config: Mapping[str, Any],
    measured_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    score_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_closed_score_config.v4",
        "unexpected V4 closed-score config",
    )
    method = str(config.get("method_id"))
    seed = int(config.get("base_flow_training_seed", -1))
    _require(
        method in CLOSED_SCORED_METHODS_V4 and seed in BASE_FLOW_SEEDS_V4,
        "V4 closed-score method or seed differs",
    )
    _require(
        config.get("pool_assignment") == "DEVELOPMENT"
        and config.get("split") == "VALIDATION"
        and config.get("analysis_unit") == "SOURCE"
        and config.get("undefined_source_policy") == "EXCLUDE_NOT_ZERO_FILL"
        and config.get("score_transform") == "SOURCEWISE_EXP_SHIFTED_MAX",
        "V4 closed-score cohort or metric policy differs",
    )
    _require(
        config.get("development_test_outcomes_accessed_after_atomic_test") is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 closed-score accessed a protected outcome",
    )
    if method in CLOSED_CRITIC_METHODS_V4:
        combination = (
            float(config.get("kappa", -1)),
            float(config.get("temperature", -1)),
            float(config.get("beta_max", -1)),
        )
        _require(
            combination[0] in {0.0, 0.5, 1.0}
            and combination[1] in {0.5, 1.0}
            and combination[2] in {0.5, 1.0, 2.0}
            and isinstance(score_summary, Mapping)
            and score_summary.get("status")
            == "XEDITFLOW_V4_CLOSED_CONTROL_SCORES_COMPLETE"
            and str(score_summary.get("method_id")) == method
            and int(score_summary.get("base_flow_training_seed", -1)) == seed
            and tuple(
                float(score_summary.get(key, -1))
                for key in ("kappa", "temperature", "beta_max")
            )
            == combination
            and score_summary.get("measured_outcome_used_to_construct_score")
            is False
            and score_summary.get("independent_evaluator_used") is False,
            "V4 closed-control score provenance differs",
        )
        _require(
            all(
                str(row.get("method_id")) == method
                and int(row.get("base_flow_training_seed", -1)) == seed
                and tuple(
                    float(row.get(key, -1))
                    for key in ("kappa", "temperature", "beta_max")
                )
                == combination
                and row.get("measured_outcome_used_to_construct_score") is False
                and row.get("independent_evaluator_used") is False
                for row in score_rows
            ),
            "V4 closed-control score row provenance differs",
        )
        score_table_method = method
    else:
        _require(
            config.get("strongest_baseline_frozen_before_v4_candidate_generation")
            is True
            and config.get("baseline_reselected_for_v4") is False,
            "V4 strongest closed baseline was not pre-frozen",
        )
        score_table_method = str(config.get("score_table_method_id"))
        _require(
            score_table_method == "strongest_matched_baseline",
            "V4 strongest baseline score-table identity differs",
        )
        _require(
            isinstance(score_summary, Mapping)
            and score_summary.get("schema_version")
            == "route_a_v3_route2_xeditflow_closed_frozen_scores.v3"
            and score_summary.get("status")
            == "XEDITFLOW_V3_CLOSED_FROZEN_SCORES_COMPLETE"
            and score_summary.get("method_id") == "strongest_matched_baseline"
            and int(score_summary.get("base_flow_training_seed", -1))
            == STRONGEST_CLOSED_SCORE_PRODUCER_SEED_V4
            and int(score_summary.get("source_count", -1)) == 891
            and score_summary.get("score_path") == config.get("score_table_path")
            and score_summary.get("score_provider")
            == "FROZEN_GENETIC_GUIDING_CHECKPOINT"
            and score_summary.get("frozen_baseline_reselected") is False
            and score_summary.get("measured_outcome_used_for_score") is False
            and score_summary.get("development_test_outcomes_accessed") is False
            and score_summary.get("new_final_evaluation_outcomes_accessed")
            is False
            and score_summary.get("cpu_fallback_used") is False
            and int(score_summary.get("cuda_device_index", -1)) in range(6)
            and bool(str(score_summary.get("cuda_device_name", "")).strip())
            and score_summary.get(
                "cuda_parent_uuid_matches_declared_physical_index"
            )
            is True,
            "V4 strongest closed-score producer lineage differs",
        )
    legacy_config = {
        "schema_version": "route_a_v3_route2_xeditflow_closed_score_config.v1",
        "method_id": method,
        "score_table_method_id": score_table_method,
        "base_flow_training_seed": seed,
        "pool_assignment": "DEVELOPMENT",
        "split": "VALIDATION",
        "analysis_unit": "SOURCE",
        "undefined_source_policy": "EXCLUDE_NOT_ZERO_FILL",
        "score_transform": "SOURCEWISE_EXP_SHIFTED_MAX",
    }
    result = evaluate_closed_method_scores_v3(
        legacy_config,
        [dict(row) for row in measured_rows],
        [dict(row) for row in score_rows],
    )
    result_v4 = {
        **result,
        "schema_version": "route_a_v3_route2_xeditflow_closed_neighborhood.v4",
        "status": "XEDITFLOW_V4_CLOSED_NEIGHBORHOOD_COMPLETE",
        "base_flow_training_seed": seed,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    if method in CLOSED_CRITIC_METHODS_V4:
        result_v4.update(
            {
                "kappa": combination[0],
                "temperature": combination[1],
                "beta_max": combination[2],
            }
        )
    return result_v4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(not arguments.output.exists(), f"V4 closed-score output exists: {arguments.output}")
    config = _json(arguments.config)
    summary = (
        _json(Path(config["score_summary_path"]))
        if "score_summary_path" in config
        else None
    )
    result = evaluate_closed_method_scores_v4(
        config,
        _jsonl(Path(config["measured_neighborhood_path"])),
        _jsonl(Path(config["score_table_path"])),
        score_summary=summary,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
