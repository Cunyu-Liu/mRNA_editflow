#!/usr/bin/env python3
"""Assemble and adjudicate the exact three-seed matched XEditFlow comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import adjudicate_guided_three_seed_v3


SEEDS = (20260904, 20260905, 20260906)
METHODS = {
    "full_soft_value_smc",
    "unguided_setflow",
    "first_order_guidance",
    "simple_rate_guidance",
    "generate_then_rerank",
    "strongest_matched_baseline",
}


class XEditFlowFinalAdjudicationV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowFinalAdjudicationV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def assemble_final_payloads_v3(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    _require(manifest.get("schema_version") == "route_a_v3_route2_xeditflow_final_comparison_manifest.v1", "unexpected final comparison manifest schema")
    _require(manifest.get("status") == "XEDITFLOW_V3_FINAL_COMPARISON_RESULTS_COMPLETE", "final comparison manifest is incomplete")
    _require(manifest.get("guidance_screen_status") == "XEDITFLOW_V3_GUIDANCE_SCREEN_FROZEN", "final comparison guidance combination is not frozen")
    rows = manifest.get("seeds")
    _require(isinstance(rows, list) and len(rows) == 3, "final comparison requires exactly three seed rows")
    payloads = {}
    for row in rows:
        seed = int(row["base_flow_training_seed"])
        _require(seed in SEEDS and seed not in payloads, "final comparison seed differs or is duplicated")
        specs = row.get("methods")
        _require(isinstance(specs, Mapping) and set(specs) == METHODS, "final comparison method inventory differs")
        methods = {}
        for method, path in specs.items():
            result = _json(Path(path))
            _require(result.get("status") == "XEDITFLOW_V3_MATCHED_METHOD_METRICS_COMPLETE", f"final method metrics are incomplete: {seed}/{method}")
            _require(result.get("method_role") == method and int(result.get("base_flow_training_seed", -1)) == seed, f"final method identity differs: {seed}/{method}")
            _require(result.get("development_test_outcomes_accessed") is False and result.get("new_final_evaluation_outcomes_accessed") is False, f"final method accessed protected outcome: {seed}/{method}")
            methods[method] = result["metrics"]
        bootstrap = _json(Path(row["paired_bootstrap_path"]))
        _require(bootstrap.get("status") == "XEDITFLOW_V3_SOURCE_PAIRED_BOOTSTRAP_COMPLETE", f"final bootstrap is incomplete: {seed}")
        _require(bootstrap.get("analysis_unit") == "SOURCE", f"final bootstrap unit differs: {seed}")
        _require(bootstrap.get("development_test_outcomes_accessed") is False and bootstrap.get("new_final_evaluation_outcomes_accessed") is False, f"final bootstrap accessed protected outcome: {seed}")
        payloads[seed] = {
            "methods": methods,
            "source_paired_ndcg_improvement_ci_95": bootstrap["source_paired_ndcg_improvement_ci_95"],
            "source_paired_independent_evaluator_margin_ci_95": bootstrap["source_paired_independent_evaluator_margin_ci_95"],
            "critic_self_score_increased": bool(bootstrap["critic_self_score_increased"]),
            "all_methods_matched_compute_ceiling_met": bool(bootstrap["all_methods_matched_compute_ceiling_met"]),
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    _require(set(payloads) == set(SEEDS), "final comparison seed inventory differs")
    return payloads


def adjudicate_final_manifest_v3(manifest: Mapping[str, Any]) -> dict[str, Any]:
    gate = adjudicate_guided_three_seed_v3(assemble_final_payloads_v3(manifest))
    return {
        "schema_version": "route_a_v3_route2_xeditflow_final_adjudication.v1",
        "status": "XEDITFLOW_V3_FINAL_COMPARISON_TERMINAL",
        "gate": gate,
        "predictor_generator_baselines_metrics_policy_frozen": True,
        "new_final_evaluation_authorized": gate["new_final_evaluation_authorized"],
        "submission_ready": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"final XEditFlow adjudication exists: {args.output}")
    result = adjudicate_final_manifest_v3(_json(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
