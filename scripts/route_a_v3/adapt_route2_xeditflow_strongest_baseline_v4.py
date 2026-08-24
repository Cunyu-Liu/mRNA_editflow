#!/usr/bin/env python3
"""Adapt the pre-V4 frozen strongest baseline to V4 final metric schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4


class XEditFlowStrongestBaselineAdapterV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowStrongestBaselineAdapterV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def adapt_strongest_baseline_v4(
    strongest: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    base_flow_training_seed: int,
) -> dict[str, Any]:
    _require(
        base_flow_training_seed in BASE_FLOW_SEEDS_V4,
        "V4 strongest adapter seed differs",
    )
    _require(
        strongest.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest.get("strongest_generation_baseline_id") == "genetic"
        and strongest.get("evaluation_outcomes_accessed") is False
        and int(strongest.get("forward_equivalent_budget_per_source", -1)) == 320,
        "V4 strongest genetic baseline is not prospectively frozen",
    )
    _require(
        selection.get("selection_pool") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and selection.get("evaluation_release_state") == "CLOSED",
        "V4 strongest baseline selection boundary differs",
    )
    entries = [
        row
        for row in selection.get("baseline_evaluations", ())
        if row.get("method_id") == "genetic"
    ]
    _require(len(entries) == 1, "V4 frozen genetic baseline entry differs")
    evaluation = entries[0]["evaluation"]
    generation = evaluation["generation"]
    measured = evaluation["measured_neighborhood"]
    _require(
        generation.get("method_id") == "genetic",
        "V4 frozen genetic generation identity differs",
    )
    generation_adapter = {
        "schema_version": "route_a_v3_route2_xeditflow_strongest_baseline_adapter.v4",
        "status": "XEDITFLOW_V4_STRONGEST_BASELINE_ADAPTER_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "base_flow_training_seed": base_flow_training_seed,
        "maximum_forward_equivalents_per_source": 320,
        "hard_legality_rate": float(generation["hard_legality_rate"]),
        "edit_budget_violation_count": int(generation["edit_budget_violation_count"]),
        "candidate_budget_violation_count": int(
            generation["candidate_budget_violation_count"]
        ),
        "trajectory_replay_failure_count": 0,
        "numerical_failure_count": 0,
        "frozen_baseline_reselected_for_v4": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    open_adapter = {
        "schema_version": "route_a_v3_route2_xeditflow_open_generation_metrics.v4",
        "status": "XEDITFLOW_V4_OPEN_GENERATION_METRICS_COMPLETE",
        "method_id": "strongest_matched_baseline",
        "underlying_method_id": "genetic",
        "base_flow_training_seed": base_flow_training_seed,
        "source_count": int(generation["source_count"]),
        "source_macro_candidate_recovery": float(
            measured["source_macro_candidate_recovery_rate"]
        ),
        "source_macro_top_k_recovery": float(
            measured["source_macro_measured_top_k_recovery_at_k"]
        ),
        "source_macro_unique_candidate_rate": float(
            generation["source_macro_unique_candidate_rate"]
        ),
        "hard_legality_rate": float(generation["hard_legality_rate"]),
        "edit_budget_violation_count": int(generation["edit_budget_violation_count"]),
        "candidate_budget_violation_count": int(
            generation["candidate_budget_violation_count"]
        ),
        "historical_open_support_only": True,
        "historical_open_ndcg_used_as_new_closed_ndcg": False,
        "frozen_baseline_reselected_for_v4": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return {
        "generation": generation_adapter,
        "open": open_adapter,
        "guiding_checkpoint_path": str(strongest["guiding_checkpoint_path"]),
        "independent_evaluator_checkpoint_path": str(
            strongest["independent_evaluator_checkpoint_path"]
        ),
    }


def validate_strongest_adapter_config_v4(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_strongest_baseline_adapter_config.v4",
        "unexpected V4 strongest adapter config schema",
    )
    _require(
        int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 strongest adapter config seed differs",
    )
    for field in (
        "strongest_generation_baseline_path",
        "baseline_selection_input_path",
        "output_dir",
    ):
        _require(
            str(config.get(field, "")).startswith(
                "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
            ),
            f"V4 strongest adapter {field} left Route 2 /mnt",
        )
    _require(
        config.get("strongest_baseline_reselected_for_v4") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and config.get("new_final_evaluation_outcomes_accessed") is False,
        "V4 strongest adapter protected-input policy differs",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    config = _json(arguments.config)
    validate_strongest_adapter_config_v4(config)
    output_dir = Path(str(config["output_dir"]))
    _require(
        not output_dir.exists(),
        f"V4 strongest adapter output exists: {output_dir}",
    )
    result = adapt_strongest_baseline_v4(
        _json(Path(str(config["strongest_generation_baseline_path"]))),
        _json(Path(str(config["baseline_selection_input_path"]))),
        base_flow_training_seed=int(config["base_flow_training_seed"]),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    for name in ("generation", "open"):
        (output_dir / f"{name}.json").write_text(
            json.dumps(result[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "adapter_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
