#!/usr/bin/env python3
"""Atomically assemble and adjudicate the V4 three-seed final comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v4 import adjudicate_guided_three_seed_v4
from core.route2_xeditflow_value_training_v4 import (
    BASE_FLOW_SEEDS_V4,
    validate_value_training_provenance_v4,
)
from scripts.route_a_v3.assemble_route2_xeditflow_final_seed_evidence_v4 import (
    METHODS_V4,
)


class XEditFlowFinalAdjudicationV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalAdjudicationV4Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def assemble_final_payloads_v4(
    manifest: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    _require(
        manifest.get("schema_version")
        == "route_a_v3_route2_xeditflow_final_comparison_manifest.v4"
        and manifest.get("status")
        == "XEDITFLOW_V4_FINAL_COMPARISON_RESULTS_COMPLETE"
        and manifest.get("guidance_screen_status")
        == "XEDITFLOW_V4_GUIDANCE_SCREEN_FROZEN",
        "V4 final comparison manifest is incomplete or unfrozen",
    )
    rows = manifest.get("seeds")
    selected_combination = tuple(
        float(value) for value in manifest.get("selected_combination", ())
    )
    _require(
        isinstance(rows, list) and len(rows) == 3,
        "V4 final comparison requires exactly three seed rows",
    )
    _require(
        len(selected_combination) == 3
        and selected_combination[0] in {0.0, 0.5, 1.0}
        and selected_combination[1] in {0.5, 1.0}
        and selected_combination[2] in {0.5, 1.0, 2.0},
        "V4 final comparison selected combination differs",
    )
    value_checkpoint_paths = manifest.get("value_checkpoint_paths")
    _require(
        isinstance(value_checkpoint_paths, Mapping)
        and set(value_checkpoint_paths)
        == {str(seed) for seed in BASE_FLOW_SEEDS_V4},
        "V4 final value checkpoint inventory differs",
    )
    value_checkpoint_paths = {
        str(seed): str(value_checkpoint_paths[str(seed)])
        for seed in BASE_FLOW_SEEDS_V4
    }
    _require(
        manifest.get("guidance_screen_value_checkpoint_path")
        == value_checkpoint_paths["20260912"],
        "V4 final guidance-screen value checkpoint path differs",
    )
    validate_value_training_provenance_v4(
        manifest.get("guidance_screen_value_training_provenance", {}),
        base_flow_training_seed=20260912,
        value_checkpoint_path=value_checkpoint_paths["20260912"],
    )
    payloads: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("base_flow_training_seed", -1))
        _require(
            seed in BASE_FLOW_SEEDS_V4 and seed not in payloads,
            "V4 final comparison seed differs or is duplicated",
        )
        _require(
            row.get("value_checkpoint_path") == value_checkpoint_paths[str(seed)],
            f"V4 final value checkpoint path differs: {seed}",
        )
        validate_value_training_provenance_v4(
            row.get("value_training_provenance", {}),
            base_flow_training_seed=seed,
            value_checkpoint_path=value_checkpoint_paths[str(seed)],
        )
        method_specs = row.get("methods")
        _require(
            isinstance(method_specs, Mapping) and set(method_specs) == METHODS_V4,
            f"V4 final comparison method inventory differs: {seed}",
        )
        methods: dict[str, Any] = {}
        for method, path in method_specs.items():
            result = _json(Path(path))
            _require(
                result.get("status")
                == "XEDITFLOW_V4_MATCHED_METHOD_METRICS_COMPLETE"
                and result.get("method_role") == method
                and int(result.get("base_flow_training_seed", -1)) == seed
                and tuple(
                    float(value)
                    for value in result.get("selected_combination", ())
                )
                == selected_combination
                and result.get(
                    "development_test_outcomes_accessed_after_atomic_test"
                )
                is False
                and result.get("new_final_evaluation_outcomes_accessed") is False,
                f"V4 final method metrics differ or accessed outcomes: {seed}/{method}",
            )
            methods[method] = result["metrics"]
        bootstrap = _json(Path(row["paired_bootstrap_path"]))
        _require(
            bootstrap.get("status")
            == "XEDITFLOW_V4_SOURCE_PAIRED_BOOTSTRAP_COMPLETE"
            and bootstrap.get("analysis_unit") == "SOURCE"
            and int(bootstrap.get("base_flow_training_seed", -1)) == seed
            and int(bootstrap.get("bootstrap_iterations", -1)) == 10_000
            and tuple(
                float(value)
                for value in bootstrap.get("selected_combination", ())
            )
            == selected_combination
            and bootstrap.get("closed_method_source_support_exactly_matched")
            is True
            and bootstrap.get("undefined_closed_sources_filled_with_zero") is False
            and int(bootstrap.get("closed_source_count", -1))
            >= int(bootstrap.get("defined_closed_source_count", -1))
            >= 2
            and bootstrap.get("setflow_mode_is_fixed_trajectory_state") is True
            and bootstrap.get("free_action_ratio_head_used") is False
            and bootstrap.get("all_network_forwards_separately_charged") is True
            and bootstrap.get("matched_compute_schema") == "MatchedComputeRecordV4"
            and bootstrap.get("independent_evaluator_in_gradient") is False
            and bootstrap.get(
                "development_test_outcomes_accessed_after_atomic_test"
            )
            is False
            and int(bootstrap.get("new_final_evaluation_outcome_reads", -1)) == 0,
            f"V4 final paired-bootstrap evidence differs: {seed}",
        )
        equal_wall = _json(Path(row["equal_wall_time_sensitivity_path"]))
        _require(
            equal_wall.get("status")
            == "XEDITFLOW_V4_EQUAL_WALL_TIME_SENSITIVITY_COMPLETE"
            and int(equal_wall.get("base_flow_training_seed", -1)) == seed
            and isinstance(equal_wall.get("methods"), Mapping)
            and set(equal_wall["methods"]) == METHODS_V4
            and 2 <= int(equal_wall.get("common_source_prefix_count", -1)) <= 891
            and equal_wall.get("all_network_forwards_separately_charged") is True
            and equal_wall.get("matched_compute_schema") == "MatchedComputeRecordV4"
            and equal_wall.get(
                "development_test_outcomes_accessed_after_atomic_test"
            )
            is False
            and equal_wall.get("new_final_evaluation_outcomes_accessed") is False,
            f"V4 final equal-wall evidence differs: {seed}",
        )
        payloads[seed] = {
            "methods": methods,
            "source_paired_ndcg_improvement_ci_95": bootstrap[
                "source_paired_ndcg_improvement_ci_95"
            ],
            "source_paired_independent_evaluator_margin_ci_95": bootstrap[
                "source_paired_independent_evaluator_margin_ci_95"
            ],
            "critic_self_score_increased": bool(
                bootstrap["critic_self_score_increased"]
            ),
            "all_methods_matched_compute_ceiling_met": bool(
                bootstrap["all_methods_matched_compute_ceiling_met"]
            ),
            "setflow_mode_is_fixed_trajectory_state": True,
            "free_action_ratio_head_used": False,
            "all_network_forwards_separately_charged": True,
            "matched_compute_schema": "MatchedComputeRecordV4",
            "equal_wall_time_sensitivity_complete": True,
            "independent_evaluator_in_gradient": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        }
    _require(
        set(payloads) == set(BASE_FLOW_SEEDS_V4),
        "V4 final comparison seed inventory differs",
    )
    return payloads


def adjudicate_final_manifest_v4(manifest: Mapping[str, Any]) -> dict[str, Any]:
    gate = adjudicate_guided_three_seed_v4(assemble_final_payloads_v4(manifest))
    rows = manifest["seeds"]
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_adjudication.v4",
        "status": "XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL",
        "gate": gate,
        "value_checkpoint_paths": dict(manifest["value_checkpoint_paths"]),
        "value_training_provenance_by_seed": {
            str(row["base_flow_training_seed"]): row["value_training_provenance"]
            for row in rows
        },
        "predictor_generator_baselines_metrics_policy_frozen": True,
        "new_final_evaluation_authorized": gate[
            "new_final_evaluation_authorized"
        ],
        "additional_training_seed_authorized": False,
        "submission_ready": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    _require(
        not arguments.output.exists(),
        f"terminal V4 final adjudication exists: {arguments.output}",
    )
    result = adjudicate_final_manifest_v4(_json(arguments.manifest))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial = arguments.output.with_suffix(arguments.output.suffix + ".partial")
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, arguments.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
