#!/usr/bin/env python3
"""Adapt the frozen Development genetic baseline to final V3 metric schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class XEditFlowStrongestBaselineAdapterV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowStrongestBaselineAdapterV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def adapt_strongest_baseline_v3(
    strongest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    base_flow_training_seed: int,
) -> dict[str, Any]:
    _require(base_flow_training_seed in {20260904, 20260905, 20260906}, "strongest adapter seed differs")
    _require(
        strongest.get("status") == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest.get("strongest_generation_baseline_id") == "genetic"
        and strongest.get("evaluation_outcomes_accessed") is False,
        "strongest genetic baseline is not frozen",
    )
    _require(
        selection.get("selection_pool") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and selection.get("evaluation_release_state") == "CLOSED",
        "strongest baseline selection boundary differs",
    )
    entries = [row for row in selection["baseline_evaluations"] if row.get("method_id") == "genetic"]
    _require(len(entries) == 1, "frozen genetic baseline entry differs")
    evaluation = entries[0]["evaluation"]
    generation = evaluation["generation"]
    measured = evaluation["measured_neighborhood"]
    _require(generation.get("method_id") == "genetic", "frozen genetic generation identity differs")
    maximum_compute = int(strongest["forward_equivalent_budget_per_source"])
    _require(maximum_compute == 320, "strongest baseline compute ceiling differs")
    generation_adapter = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_baseline_adapter.v3",
        "status": "XEDITFLOW_V3_STRONGEST_BASELINE_ADAPTER_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "base_flow_training_seed": base_flow_training_seed,
        "maximum_forward_equivalents_per_source": maximum_compute,
        "hard_legality_rate": float(generation["hard_legality_rate"]),
        "edit_budget_violation_count": int(generation["edit_budget_violation_count"]),
        "candidate_budget_violation_count": int(generation["candidate_budget_violation_count"]),
        "trajectory_replay_failure_count": 0,
        "numerical_failure_count": 0,
        "frozen_baseline_reselected": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    open_adapter = {
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_metrics.v3",
        "status": "XEDITFLOW_V3_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "source_count": int(generation["source_count"]),
        "source_macro_candidate_recovery": float(measured["source_macro_candidate_recovery_rate"]),
        "source_macro_top_k_recovery": float(measured["source_macro_measured_top_k_recovery_at_k"]),
        "source_macro_unique_candidate_rate": float(generation["source_macro_unique_candidate_rate"]),
        "hard_legality_rate": float(generation["hard_legality_rate"]),
        "edit_budget_violation_count": int(generation["edit_budget_violation_count"]),
        "candidate_budget_violation_count": int(generation["candidate_budget_violation_count"]),
        "historical_open_support_only": True,
        "historical_open_ndcg_used_as_new_closed_ndcg": False,
        "frozen_baseline_reselected": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return {
        "generation": generation_adapter,
        "open": open_adapter,
        "guiding_checkpoint_path": str(strongest["guiding_checkpoint_path"]),
        "independent_evaluator_checkpoint_path": str(strongest["independent_evaluator_checkpoint_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strongest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-flow-training-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), f"strongest adapter output exists: {args.output_dir}")
    result = adapt_strongest_baseline_v3(
        _json(args.strongest),
        _json(args.selection),
        base_flow_training_seed=args.base_flow_training_seed,
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir()
    for name in ("generation", "open"):
        (args.output_dir / f"{name}.json").write_text(json.dumps(result[name], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "adapter_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
