from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_mrnabert_generation_comparison_suite_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("mrnabert_generation_comparison_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUIDED = "frozen_mrnabert_critic_v2_guided_xeditflow_v1"
BASELINES = (
    "random_legal", "greedy", "beam", "genetic", "local_search",
    "generate_then_rerank", "unguided_learned_base_flow_g0",
)


def _config():
    return {
        "schema_version": "route_a_v3_route2_mrnabert_critic_v2_generation_comparison_protocol.v1",
        "status": "WAITING_FOR_CRITIC_V2_GUIDED_AND_MATCHED_CANDIDATE_GENERATION",
        "guided_method_id": GUIDED,
        "required_baseline_method_ids": list(BASELINES),
        "candidate_paths": {method: f"/{method}.jsonl" for method in (GUIDED, *BASELINES)},
        "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
        "measured_neighborhood_pool": "DEVELOPMENT",
        "evaluation_release_state": "CLOSED",
        "evaluation_outcomes_accessed": False,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 7,
    }


def _evaluation(method, uplift_by_source, budget=100):
    per_source = {}
    measured = {}
    for key, uplift in uplift_by_source.items():
        is_guided = method == GUIDED
        is_unguided = method == "unguided_learned_base_flow_g0"
        generator = 10 if is_guided else (7 if is_unguided else 0)
        critic = 90 if is_guided else (0 if is_unguided else 95)
        per_source[key] = {
            "candidate_count": 32,
            "candidate_budget": 32,
            "independent_evaluator_score": {"count": 32, "max_uplift_over_source": uplift},
            "compute": {
                "generator_nfe": generator,
                "critic_forwards": critic,
                "independent_evaluator_forwards": 20,
                "critic_forward_budget": None if is_unguided else budget,
            },
        }
        measured[key] = {}
    return {
        "schema_version": "route_a_v3_route2_generation_evaluation.v2",
        "evaluation_release_state": "CLOSED",
        "measured_neighborhood_pool": "DEVELOPMENT",
        "generation": {
            "method_id": method,
            "hard_legality_rate": 1.0,
            "edit_budget_violation_count": 0,
            "candidate_budget_violation_count": 0,
            "generated_candidates_grant_canonical_credit": False,
            "source_macro_unique_candidate_rate": 0.8,
            "source_macro_pairwise_hamming_diversity": 0.1,
            "per_source": per_source,
        },
        "measured_neighborhood": {
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "unknown_generated_candidates_are_zero_gain": False,
            "source_macro_measured_top_k_recovery_at_k": 0.1,
            "source_macro_candidate_recovery_rate": 0.2,
            "per_source": measured,
        },
    }


def test_guided_is_compared_only_after_strongest_baseline_is_frozen() -> None:
    module = _load()
    budgets = {"S1": 100, "S2": 100, "S3": 100}
    evaluations = {
        GUIDED: _evaluation(GUIDED, {"S1": 1.0, "S2": 1.1, "S3": 1.2})
    }
    for index, method in enumerate(BASELINES):
        center = 0.3 + index * 0.05
        evaluations[method] = _evaluation(
            method, {"S1": center, "S2": center + 0.01, "S3": center - 0.01}
        )
    result = module.select_comparison(_config(), evaluations, budgets)
    assert result["guided_advantage_ci_excludes_zero"] is True
    assert result["status"] == "DEVELOPMENT_INDEPENDENT_EVALUATOR_GUIDED_ADVANTAGE"
    assert result["strongest_generation_baseline_id"] in BASELINES
    assert result["measured_biological_improvement_established"] is False


def test_guided_compute_must_equal_frozen_per_source_budget() -> None:
    module = _load()
    budgets = {"S1": 99}
    evaluations = {GUIDED: _evaluation(GUIDED, {"S1": 1.0})}
    evaluations.update({method: _evaluation(method, {"S1": 0.0}) for method in BASELINES})
    with pytest.raises(module.GenerationComparisonError, match="matched total budget|guided accounting"):
        module.select_comparison(_config(), evaluations, budgets)


def test_repository_config_uses_only_critic_v2_candidate_paths() -> None:
    module = _load()
    current = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_critic_v2_generation_comparison_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    historical = json.loads(
        (
            ROOT
            / "configs/route_a_v3_route2_mrnabert_generation_comparison_development_gpu0_v1.json"
        ).read_text(encoding="utf-8")
    )
    module.validate_config_boundary(current)
    assert historical["status"] == (
        "RETIRED_HISTORICAL_V1_GUIDED_CANDIDATE_PATH_NOT_AUTHORIZED"
    )
    with pytest.raises(module.GenerationComparisonError, match="schema"):
        module.validate_config_boundary(historical)
