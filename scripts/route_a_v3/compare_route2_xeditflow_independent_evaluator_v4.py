#!/usr/bin/env python3
"""Compare a V4 guidance candidate set with the pre-frozen strongest baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_a_v3.compare_route2_xeditflow_independent_evaluator_v3 import (
    paired_mean_bootstrap_ci_v3,
)
from scripts.route_a_v3.evaluate_route2_generation_v1 import (
    evaluate_generation,
    load_source_manifest,
)
from core.route2_xeditflow_value_training_v4 import BASE_FLOW_SEEDS_V4


class XEditFlowIndependentEvaluatorV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowIndependentEvaluatorV4Error(message)


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


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def compare_independent_evaluator_v4(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        config.get("schema_version")
        == "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v4",
        "unexpected V4 independent-evaluator comparison schema",
    )
    combination = tuple(float(value) for value in config.get("combination", ()))
    _require(
        len(combination) == 3
        and combination[0] in {0.0, 0.5, 1.0}
        and combination[1] in {0.5, 1.0}
        and combination[2] in {0.5, 1.0, 2.0}
        and int(config.get("base_flow_training_seed", -1)) in BASE_FLOW_SEEDS_V4,
        "V4 evaluator comparison identity differs",
    )
    _require(
        int(config.get("bootstrap_iterations", -1)) == 10_000
        and config.get("independent_evaluator_in_gradient") is False
        and config.get("development_test_outcomes_accessed_after_atomic_test")
        is False
        and int(config.get("new_final_evaluation_outcome_reads", -1)) == 0,
        "V4 evaluator comparison protected-input policy differs",
    )
    strongest = _json(Path(str(config["strongest_baseline_path"])))
    _require(
        strongest.get("status")
        == "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
        and strongest.get("evaluation_outcomes_accessed") is False
        and bool(
            str(strongest.get("independent_evaluator_checkpoint_path", "")).strip()
        ),
        "V4 strongest generation baseline is not frozen and outcome-isolated",
    )
    strongest_id = str(strongest["strongest_generation_baseline_id"])
    evaluator_checkpoint_path = str(
        strongest["independent_evaluator_checkpoint_path"]
    )
    selection = _json(Path(str(config["baseline_selection_input_path"])))
    _require(
        selection.get("selection_pool") == "DEVELOPMENT_MEASURED_NEIGHBORHOOD"
        and selection.get("evaluation_release_state") == "CLOSED",
        "V4 baseline selection pool or release state differs",
    )
    entries = [
        row
        for row in selection["baseline_evaluations"]
        if str(row["method_id"]) == strongest_id
    ]
    _require(len(entries) == 1, "V4 strongest baseline evaluation entry differs")
    baseline_generation = entries[0]["evaluation"]["generation"]
    _require(
        baseline_generation.get("method_id") == strongest_id,
        "V4 strongest baseline method identity differs",
    )
    scoring_summary = _json(Path(str(config["guided_scoring_summary_path"])))
    _require(
        scoring_summary.get("status")
        == "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE"
        and scoring_summary.get("evaluator_frozen_before_candidate_generation")
        is True
        and scoring_summary.get("guiding_checkpoint_distinct") is True
        and len(scoring_summary.get("guiding_checkpoint_paths", ())) == 3
        and scoring_summary.get("evaluator_checkpoint_path")
        == evaluator_checkpoint_path
        and scoring_summary.get("evaluator_result_stage")
        == "FROZEN_DEVELOPMENT_VALIDATION"
        and scoring_summary.get("selection_score_scale")
        == "TRAIN_TASK_ROBUST_STANDARDIZED"
        and scoring_summary.get("cpu_fallback_used") is False
        and scoring_summary.get("independent_evaluator_in_gradient") is False
        and scoring_summary.get(
            "development_test_outcomes_accessed_after_atomic_test"
        )
        is False
        and int(scoring_summary.get("new_final_evaluation_outcome_reads", -1))
        == 0,
        "V4 independent evaluator scoring is incomplete or contaminated",
    )
    candidate_rows = _jsonl(Path(str(config["guided_scored_candidate_path"])))
    expected_method = str(config["method_id"])
    base_flow_seed = int(config["base_flow_training_seed"])
    _require(
        all(
            str(row.get("method_id")) == expected_method
            and int(row.get("base_flow_training_seed", -1)) == base_flow_seed
            and tuple(
                float(row[key]) for key in ("kappa", "temperature", "beta_max")
            )
            == combination
            and row.get("independent_evaluator_score") is not None
            for row in candidate_rows
        ),
        "V4 evaluator-scored candidate combination differs",
    )
    sources = load_source_manifest(Path(str(config["source_eligibility_manifest"])))
    guided = evaluate_generation(sources, candidate_rows)
    _require(
        guided["method_id"] == expected_method
        and set(guided["per_source"])
        == set(baseline_generation["per_source"])
        == set(sources),
        "V4 independent-evaluator source pairing differs",
    )
    differences: dict[str, float] = {}
    guided_uplifts: dict[str, float] = {}
    baseline_uplifts: dict[str, float] = {}
    for source_key in sorted(sources):
        guided_score = guided["per_source"][source_key][
            "independent_evaluator_score"
        ]
        baseline_score = baseline_generation["per_source"][source_key][
            "independent_evaluator_score"
        ]
        _require(
            guided_score is not None and baseline_score is not None,
            "V4 independent evaluator score is absent",
        )
        guided_uplift = _finite(
            guided_score["max_uplift_over_source"], "guided evaluator uplift"
        )
        baseline_uplift = _finite(
            baseline_score["max_uplift_over_source"],
            "baseline evaluator uplift",
        )
        guided_uplifts[source_key] = guided_uplift
        baseline_uplifts[source_key] = baseline_uplift
        differences[source_key] = guided_uplift - baseline_uplift
    values = list(differences.values())
    ci = paired_mean_bootstrap_ci_v3(
        values,
        iterations=10_000,
        seed=int(config["bootstrap_seed"]),
    )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_independent_evaluator_comparison.v4"
        ),
        "status": "XEDITFLOW_V4_INDEPENDENT_EVALUATOR_COMPARISON_COMPLETE",
        "method_id": expected_method,
        "base_flow_training_seed": base_flow_seed,
        "combination": list(combination),
        "strongest_baseline_id": strongest_id,
        "independent_evaluator_checkpoint_path": evaluator_checkpoint_path,
        "evaluator_result_stage": scoring_summary["evaluator_result_stage"],
        "selection_score_scale": scoring_summary["selection_score_scale"],
        "cpu_fallback_used": False,
        "source_count": len(values),
        "guided_source_macro_max_uplift": float(
            np.mean(list(guided_uplifts.values()))
        ),
        "baseline_source_macro_max_uplift": float(
            np.mean(list(baseline_uplifts.values()))
        ),
        "paired_margin_over_strongest_baseline": float(np.mean(values)),
        "source_paired_margin_ci_95": ci,
        "per_source_paired_margin": differences,
        "analysis_unit": "SOURCE",
        "independent_evaluator_used_for_gradient": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _json(args.config)
    _require(
        args.output == Path(str(config["output_path"])),
        "V4 evaluator comparison output differs from its frozen config",
    )
    _require(
        not args.output.exists(),
        f"terminal V4 evaluator comparison exists: {args.output}",
    )
    result = compare_independent_evaluator_v4(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
