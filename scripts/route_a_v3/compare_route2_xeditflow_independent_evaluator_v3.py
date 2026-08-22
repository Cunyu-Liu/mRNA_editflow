#!/usr/bin/env python3
"""Compute the source-paired independent-evaluator margin over the frozen baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
)


class XEditFlowIndependentEvaluatorV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowIndependentEvaluatorV3Error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"JSONL input is empty or invalid: {path}")
    return rows


def _finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def paired_mean_bootstrap_ci_v3(
    differences: Sequence[float], *, iterations: int, seed: int
) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "paired evaluator differences are invalid")
    _require(iterations == 10_000, "paired evaluator bootstrap iterations changed")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare_independent_evaluator_v3(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v1", "unexpected independent-evaluator comparison schema")
    strongest = _json(Path(config["strongest_baseline_path"]))
    _require(
        strongest.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest generation baseline is not frozen",
    )
    _require(strongest.get("evaluation_outcomes_accessed") is False, "strongest baseline accessed Evaluation")
    strongest_id = str(strongest["strongest_generation_baseline_id"])
    selection = _json(Path(config["baseline_selection_input_path"]))
    _require(selection.get("selection_pool") == "DEVELOPMENT_VALIDATION", "baseline selection pool differs")
    _require(selection.get("evaluation_release_state") == "CLOSED", "baseline selection opened Evaluation")
    entries = [row for row in selection["baseline_evaluations"] if str(row["method_id"]) == strongest_id]
    _require(len(entries) == 1, "strongest baseline evaluation entry differs")
    baseline_generation = entries[0]["evaluation"]["generation"]
    _require(baseline_generation.get("method_id") == strongest_id, "strongest baseline method identity differs")
    sources = load_source_manifest(Path(config["source_eligibility_manifest"]))
    guided = evaluate_generation(sources, _jsonl(Path(config["guided_scored_candidate_path"])))
    _require(set(guided["per_source"]) == set(baseline_generation["per_source"]) == set(sources), "independent-evaluator source pairing differs")
    differences = {}
    guided_uplifts = {}
    baseline_uplifts = {}
    for source_key in sorted(sources):
        guided_score = guided["per_source"][source_key]["independent_evaluator_score"]
        baseline_score = baseline_generation["per_source"][source_key]["independent_evaluator_score"]
        _require(guided_score is not None and baseline_score is not None, "independent evaluator score is absent")
        left = _finite(guided_score["max_uplift_over_source"], "guided evaluator uplift")
        right = _finite(baseline_score["max_uplift_over_source"], "baseline evaluator uplift")
        guided_uplifts[source_key] = left
        baseline_uplifts[source_key] = right
        differences[source_key] = left - right
    values = list(differences.values())
    ci = paired_mean_bootstrap_ci_v3(
        values,
        iterations=int(config["bootstrap_iterations"]),
        seed=int(config["bootstrap_seed"]),
    )
    return {
        "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_comparison.v3",
        "status": "XEDITFLOW_V3_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE",
        "method_id": guided["method_id"],
        "strongest_baseline_id": strongest_id,
        "source_count": len(values),
        "guided_source_macro_max_uplift": float(np.mean(list(guided_uplifts.values()))),
        "baseline_source_macro_max_uplift": float(np.mean(list(baseline_uplifts.values()))),
        "paired_margin_over_strongest_baseline": float(np.mean(values)),
        "source_paired_margin_ci_95": ci,
        "per_source_paired_margin": differences,
        "analysis_unit": "SOURCE",
        "independent_evaluator_used_for_gradient": False,
        "development_test_outcomes_accessed": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"independent-evaluator comparison exists: {args.output}")
    result = compare_independent_evaluator_v3(_json(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
