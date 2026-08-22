#!/usr/bin/env python3
"""Adjudicate the exact eighteen Development guidance combinations once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_xeditflow_gate_v3 import GUIDANCE_GRID_V3, adjudicate_guidance_screen_v3


class GuidanceScreenAdjudicationV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidanceScreenAdjudicationV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def assemble_screen_results_v3(manifest: Mapping[str, Any]) -> dict[tuple[float, float, float], dict[str, Any]]:
    _require(manifest.get("schema_version") == "route_a_v3_route2_xeditflow_guidance_screen_manifest.v1", "unexpected guidance manifest schema")
    _require(manifest.get("status") == "XEDITFLOW_V3_GUIDANCE_SCREEN_CONFIGS_PREPARED", "guidance manifest is incomplete")
    jobs = manifest.get("guidance_jobs")
    _require(isinstance(jobs, list) and len(jobs) == 18, "guidance manifest must contain 18 jobs")
    results = {}
    for job in jobs:
        combination = tuple(float(value) for value in job["combination"])
        _require(combination in GUIDANCE_GRID_V3 and combination not in results, "guidance manifest combination differs")
        closed = _json(Path(job["closed_config"]["output_dir"]) / "run_summary.json")
        smc = _json(Path(job["smc_config"]["output_dir"]) / "run_summary.json")
        open_metrics = _json(Path(job["open_generation_metric_path"]))
        evaluator = _json(Path(job["independent_evaluator_metric_path"]))
        _require(closed.get("status") == "XEDITFLOW_V3_CLOSED_NEIGHBORHOOD_COMPLETE", "guidance closed result is incomplete")
        _require(smc.get("status") == "XEDITFLOW_V3_SMC_GENERATION_COMPLETE", "guidance SMC result is incomplete")
        _require(all(payload.get("development_test_outcomes_accessed") is False and payload.get("new_final_evaluation_outcomes_accessed") is False for payload in (closed, smc, open_metrics, evaluator)), "guidance screen accessed protected outcome")
        results[combination] = {
            "status": "XEDITFLOW_V3_GUIDANCE_SCREEN_COMBINATION_COMPLETE",
            "base_flow_training_seed": 20260904,
            "combination": list(combination),
            "closed_source_macro_ndcg": closed["source_macro_ndcg"],
            "closed_source_macro_normalized_regret": closed["source_macro_normalized_regret"],
            "independent_evaluator_paired_margin": evaluator["paired_margin_over_strongest_baseline"],
            "open_source_macro_candidate_recovery": open_metrics["source_macro_candidate_recovery"],
            "total_forward_equivalents": smc["maximum_forward_equivalents_per_source"] + int(smc["reserved_terminal_critic_forwards_per_source"]),
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
    _require(set(results) == set(GUIDANCE_GRID_V3), "guidance result grid differs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"terminal guidance adjudication exists: {args.output}")
    gate = adjudicate_guidance_screen_v3(assemble_screen_results_v3(_json(args.manifest)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
