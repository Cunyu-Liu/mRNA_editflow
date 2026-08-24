from __future__ import annotations

import copy
import json

import pytest

from scripts.route_a_v3.compare_route2_xeditflow_independent_evaluator_v4 import (
    compare_independent_evaluator_v4,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path):
    sources = tmp_path / "sources.jsonl"
    guided = tmp_path / "guided.jsonl"
    strongest = tmp_path / "strongest.json"
    selection = tmp_path / "selection.json"
    scoring = tmp_path / "scoring.json"
    _write_jsonl(
        sources,
        [
            {
                "source_key": "s",
                "source_sequence": "AC",
                "edit_budget": 1,
                "candidate_budget": 2,
                "source_independent_evaluator_score": 0.0,
            }
        ],
    )
    _write_jsonl(
        guided,
        [
            {
                "method_id": "xeditflow_v4_guidance_screen_k",
                "base_flow_training_seed": 20260912,
                "kappa": 0.5,
                "temperature": 1.0,
                "beta_max": 2.0,
                "source_key": "s",
                "candidate_sequence": "AA",
                "terminal_cause": "BUDGET_EXHAUSTED",
                "generation_score": 1.0,
                "source_independent_evaluator_score": 0.0,
                "independent_evaluator_score": 0.4,
            }
        ],
    )
    _write_json(
        strongest,
        {
            "status": (
                "DEVELOPMENT_STRONGEST_GENERATION_BASELINE_FROZEN_INDEPENDENT_EVALUATOR_ONLY"
            ),
            "strongest_generation_baseline_id": "genetic",
            "evaluation_outcomes_accessed": False,
        },
    )
    _write_json(
        selection,
        {
            "selection_pool": "DEVELOPMENT_MEASURED_NEIGHBORHOOD",
            "evaluation_release_state": "CLOSED",
            "baseline_evaluations": [
                {
                    "method_id": "genetic",
                    "evaluation": {
                        "generation": {
                            "method_id": "genetic",
                            "per_source": {
                                "s": {
                                    "independent_evaluator_score": {
                                        "max_uplift_over_source": 0.1
                                    }
                                }
                            },
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        scoring,
        {
            "status": "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE",
            "evaluator_frozen_before_candidate_generation": True,
            "guiding_checkpoint_distinct": True,
            "guiding_checkpoint_paths": ["a", "b", "c"],
            "independent_evaluator_in_gradient": False,
            "development_test_outcomes_accessed_after_atomic_test": False,
            "new_final_evaluation_outcome_reads": 0,
        },
    )
    return {
        "schema_version": (
            "route_a_v3_route2_xeditflow_independent_evaluator_comparison_config.v4"
        ),
        "method_id": "xeditflow_v4_guidance_screen_k",
        "base_flow_training_seed": 20260912,
        "combination": [0.5, 1.0, 2.0],
        "strongest_baseline_path": str(strongest),
        "baseline_selection_input_path": str(selection),
        "source_eligibility_manifest": str(sources),
        "guided_scored_candidate_path": str(guided),
        "guided_scoring_summary_path": str(scoring),
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 20261001,
        "independent_evaluator_in_gradient": False,
        "development_test_outcomes_accessed_after_atomic_test": False,
        "new_final_evaluation_outcome_reads": 0,
    }


def test_v4_independent_evaluator_margin_is_source_paired_and_frozen(
    tmp_path,
) -> None:
    result = compare_independent_evaluator_v4(_fixture(tmp_path))
    assert abs(result["paired_margin_over_strongest_baseline"] - 0.3) < 1e-12
    assert result["source_paired_margin_ci_95"] == pytest.approx([0.3, 0.3])
    assert result["independent_evaluator_used_for_gradient"] is False


def test_v4_independent_evaluator_rejects_gradient_or_combination_drift(
    tmp_path,
) -> None:
    config = _fixture(tmp_path)
    changed = copy.deepcopy(config)
    changed["independent_evaluator_in_gradient"] = True
    with pytest.raises(Exception, match="protected-input"):
        compare_independent_evaluator_v4(changed)
    changed = copy.deepcopy(config)
    changed["combination"] = [0.5, 1.0, 1.0]
    with pytest.raises(Exception, match="combination differs"):
        compare_independent_evaluator_v4(changed)
