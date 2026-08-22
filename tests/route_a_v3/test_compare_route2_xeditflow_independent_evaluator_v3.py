from __future__ import annotations

import json

from scripts.route_a_v3.compare_route2_xeditflow_independent_evaluator_v3 import (
    compare_independent_evaluator_v3,
)


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_independent_evaluator_margin_is_source_paired(tmp_path) -> None:
    sources = tmp_path / "sources.jsonl"
    guided = tmp_path / "guided.jsonl"
    strongest = tmp_path / "strongest.json"
    selection = tmp_path / "selection.json"
    _write_jsonl(sources, [{
        "source_key": "s",
        "source_sequence": "AC",
        "edit_budget": 1,
        "candidate_budget": 2,
        "source_independent_evaluator_score": 0.0,
    }])
    _write_jsonl(guided, [{
        "method_id": "guided",
        "source_key": "s",
        "candidate_sequence": "AA",
        "terminal_cause": "BUDGET_EXHAUSTED",
        "generation_score": 1.0,
        "source_independent_evaluator_score": 0.0,
        "independent_evaluator_score": 0.4,
    }])
    strongest.write_text(json.dumps({
        "status": "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY",
        "strongest_generation_baseline_id": "genetic",
        "evaluation_outcomes_accessed": False,
    }), encoding="utf-8")
    baseline_generation = {
        "method_id": "genetic",
        "per_source": {"s": {"independent_evaluator_score": {"max_uplift_over_source": 0.1}}},
    }
    selection.write_text(json.dumps({
        "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
        "evaluation_release_state": "CLOSED",
        "baseline_evaluations": [{
            "method_id": "genetic",
            "evaluation": {"generation": baseline_generation},
        }],
    }), encoding="utf-8")
    result = compare_independent_evaluator_v3({
        "schema_version": "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v1",
        "strongest_baseline_path": str(strongest),
        "baseline_selection_input_path": str(selection),
        "source_eligibility_manifest": str(sources),
        "guided_scored_candidate_path": str(guided),
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 20261001,
    })
    assert abs(result["paired_margin_over_strongest_baseline"] - 0.3) < 1e-12
    assert result["source_paired_margin_ci_95"] == [0.4 - 0.1, 0.4 - 0.1]
    assert result["analysis_unit"] == "SOURCE"
