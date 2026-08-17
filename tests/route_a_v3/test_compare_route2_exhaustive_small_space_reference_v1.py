import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "route_a_v3" / "compare_route2_exhaustive_small_space_reference_v1.py"
CONFIG = ROOT / "configs" / "route_a_v3_route2_exhaustive_small_space_reference_seed20260816_gpu6_v1.json"
SCORING = ROOT / "configs" / "route_a_v3_route2_exhaustive_small_space_independent_evaluator_gpu6_v1.json"
PROTOCOL = ROOT / "configs" / "route_a_v3_route2_generation_matched_compute_repair_protocol_v1.json"
JOBS = ROOT / "configs" / "route_a_v3_route2_generation_independent_evaluator_jobs_gpu6_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("exhaustive_reference_comparison_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(source_key: str, candidate: str, critic: float, evaluator: float, method: str) -> dict:
    return {
        "source_key": source_key,
        "candidate_sequence": candidate,
        "critic_score": critic,
        "independent_evaluator_score": evaluator,
        "source_independent_evaluator_score": 0.0,
        "method_id": method,
    }


def _evaluation(method_id: str, per_source: dict[str, dict]) -> dict:
    return {
        "schema_version": "route_a_v3_route2_generation_evaluation.v2",
        "evaluation_release_state": "CLOSED",
        "measured_neighborhood_pool": "DEVELOPMENT",
        "generation": {"method_id": method_id},
        "measured_neighborhood": {
            "candidate_support_mode": "OPEN_GENERATED_SUPPORT",
            "unknown_generated_candidates_are_zero_gain": False,
            "source_closed_measured_ndcg_defined_count": 0,
            "source_macro_closed_measured_ndcg_at_k": None,
            "per_source": per_source,
        },
    }


def _measured(candidate_recovery: float, top_k_recovery: float, ndcg, regret) -> dict:
    return {
        "candidate_recovery_rate": candidate_recovery,
        "measured_top_k_recovery_at_k": top_k_recovery,
        "recovered_measured_ndcg_at_k": ndcg,
        "normalized_regret": regret,
    }


def _toy_inputs():
    config = {
        "schema_version": "route_a_v3_route2_exhaustive_small_space_reference.v1",
        "scientific_role": "REAL_SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_NOT_FULL_COHORT_STRONGEST_SELECTOR",
        "source_cohort_count": 2,
        "candidate_budget_per_source": 32,
        "legal_space_size_per_source": 151,
        "critic_forward_budget_per_source": 256,
        "evaluation_outcomes_accessed": False,
        "full_cohort_strongest_selector_eligible": False,
        "guided_xeditflow_allowed": False,
    }
    sources = [
        {"source_key": "s1", "candidate_budget": 32},
        {"source_key": "s2", "candidate_budget": 32},
    ]
    exhaustive = [
        _row("s1", "AAAA", 1.0, 0.20, "exhaustive"),
        _row("s1", "AAAC", 0.7, 0.25, "exhaustive"),
        _row("s2", "CCCC", 0.9, 0.10, "exhaustive"),
        _row("s2", "CCCA", 0.6, 0.15, "exhaustive"),
    ]
    methods = {
        "random_legal": [
            _row("s1", "AAAA", 1.0, 0.20, "random_legal"),
            _row("s1", "AAAG", 0.4, 0.30, "random_legal"),
            _row("s2", "CCCG", 0.5, 0.20, "random_legal"),
        ],
        "greedy": [
            _row("s1", "AAAC", 0.7, 0.25, "greedy"),
            _row("s2", "CCCC", 0.9, 0.10, "greedy"),
        ],
    }
    scoring = {
        "status": "FROZEN_INDEPENDENT_EVALUATOR_SCORING_COMPLETE",
        "cpu_fallback_used": False,
    }
    suite = {"status": "MATCHED_GENERATION_BASELINE_SUITE_COMPLETED"}
    exhaustive_evaluation = _evaluation(
        "exhaustive",
        {
            "s1": _measured(0.50, 0.50, 0.80, None),
            "s2": _measured(0.25, 0.50, 0.60, 0.40),
        },
    )
    method_evaluations = {
        "random_legal": _evaluation(
            "random_legal",
            {
                "s1": _measured(0.25, 0.25, 0.70, None),
                "s2": _measured(0.50, 0.75, 0.80, 0.20),
            },
        ),
        "greedy": _evaluation(
            "greedy",
            {
                "s1": _measured(0.50, 0.50, None, None),
                "s2": _measured(0.25, 0.25, 0.55, None),
            },
        ),
    }
    return (
        config,
        sources,
        exhaustive,
        methods,
        exhaustive_evaluation,
        method_evaluations,
        scoring,
        suite,
    )


def test_real_reference_config_matches_frozen_protocol_and_jobs() -> None:
    config = _load(CONFIG)
    scoring = _load(SCORING)
    protocol = _load(PROTOCOL)
    jobs = _load(JOBS)

    assert config["source_cohort_count"] == 190
    assert config["legal_space_size_per_source"] == 151
    assert config["candidate_budget_per_source"] == protocol["candidate_budget_per_source"] == 32
    assert config["critic_forward_budget_per_source"] == protocol["search_critic_forward_budget_per_source"] == 256
    assert config["forward_equivalent_budget_per_source"] == protocol["forward_equivalent_budget_per_source"] == 320
    assert config["guiding_checkpoint_path"] == jobs["guiding_checkpoint_path"]
    assert config["independent_evaluator_checkpoint_path"] == jobs["evaluator_checkpoint_path"]
    assert config["independent_evaluator_scoring_config"] == str(SCORING.relative_to(ROOT))
    assert config["matched_compute_protocol_config"] == str(PROTOCOL.relative_to(ROOT))
    assert scoring["evaluator_checkpoint_path"] == config["independent_evaluator_checkpoint_path"]
    assert scoring["guiding_checkpoint_path"] == config["guiding_checkpoint_path"]
    assert scoring["source_manifest_path"] == config["source_manifest_path"]
    assert scoring["candidate_path"] == config["candidate_output_path"]
    assert scoring["output_path"] == config["independent_scored_output_path"]
    assert scoring["evaluator_frozen_before_candidate_generation"] is True
    assert scoring["evaluation_outcomes_used_to_select_evaluator"] == 0
    assert config["device"] == "cuda:6"
    assert config["physical_gpu_index"] == 6
    assert scoring["device"] == config["device"]
    assert scoring["physical_gpu_index"] == config["physical_gpu_index"]
    assert config["measured_neighborhood_path"].endswith(
        "/exhaustive_small_space_critic256_v1/measured_neighborhood.private.jsonl"
    )
    assert config["runner_script"] == (
        "scripts/route_a_v3/run_route2_exhaustive_small_space_reference_suite_v1.py"
    )
    assert config["full_cohort_strongest_selector_eligible"] is False


def test_comparison_reports_critic_gap_recovery_and_evaluator_alignment() -> None:
    module = _load_module()
    (
        config,
        sources,
        exhaustive,
        methods,
        exhaustive_evaluation,
        method_evaluations,
        scoring,
        suite,
    ) = _toy_inputs()
    result = module.compare(
        config=config,
        source_rows=sources,
        exhaustive_rows=exhaustive,
        full_method_rows=methods,
        exhaustive_evaluation=exhaustive_evaluation,
        full_method_evaluations=method_evaluations,
        exhaustive_scoring_summary=scoring,
        full_suite_summary=suite,
    )

    assert result["status"] == "SMALL_SPACE_EXHAUSTIVE_GUIDING_CRITIC_REFERENCE_COMPLETED"
    by_method = {row["method_id"]: row for row in result["method_summaries"]}
    assert by_method["random_legal"]["source_macro_critic_optimality_gap"] == pytest.approx(0.2)
    assert by_method["random_legal"]["critic_optimum_recovery_rate"] == pytest.approx(0.5)
    assert by_method["random_legal"][
        "source_macro_independent_evaluator_advantage_over_exhaustive_critic_top32"
    ] == pytest.approx(0.05)
    assert by_method["random_legal"][
        "source_macro_candidate_recovery_advantage_over_exhaustive_critic_top32"
    ] == pytest.approx(0.0)
    assert by_method["random_legal"][
        "source_macro_measured_top_k_recovery_advantage_over_exhaustive_critic_top32"
    ] == pytest.approx(0.0)
    assert by_method["random_legal"][
        "source_macro_recovered_measured_ndcg_advantage_over_exhaustive_critic_top32"
    ] == pytest.approx(0.05)
    assert by_method["random_legal"][
        "source_macro_normalized_regret_advantage_over_exhaustive_critic_top32"
    ] == pytest.approx(0.20)
    assert by_method["greedy"]["source_macro_critic_optimality_gap"] == pytest.approx(0.15)
    assert by_method["greedy"]["critic_optimum_recovery_rate"] == pytest.approx(0.5)
    assert result["full_cohort_strongest_selector_eligible"] is False
    assert result["measured_neighborhood_comparison_included"] is True
    assert result["unknown_generated_outcomes_treated_as_zero"] is False
    assert result["measured_superiority_claim_established"] is False
    assert result["evaluation_outcomes_accessed"] is False


def test_comparison_rejects_candidate_that_exceeds_exhaustive_critic_optimum() -> None:
    module = _load_module()
    (
        config,
        sources,
        exhaustive,
        methods,
        exhaustive_evaluation,
        method_evaluations,
        scoring,
        suite,
    ) = _toy_inputs()
    methods["random_legal"][0]["critic_score"] = 1.1
    with pytest.raises(module.ExhaustiveReferenceComparisonError):
        module.compare(
            config=config,
            source_rows=sources,
            exhaustive_rows=exhaustive,
            full_method_rows=methods,
            exhaustive_evaluation=exhaustive_evaluation,
            full_method_evaluations=method_evaluations,
            exhaustive_scoring_summary=scoring,
            full_suite_summary=suite,
        )


def test_comparison_rejects_open_support_that_assigns_unknown_outcomes_zero() -> None:
    module = _load_module()
    (
        config,
        sources,
        exhaustive,
        methods,
        exhaustive_evaluation,
        method_evaluations,
        scoring,
        suite,
    ) = _toy_inputs()
    exhaustive_evaluation["measured_neighborhood"][
        "unknown_generated_candidates_are_zero_gain"
    ] = True
    with pytest.raises(module.ExhaustiveReferenceComparisonError):
        module.compare(
            config=config,
            source_rows=sources,
            exhaustive_rows=exhaustive,
            full_method_rows=methods,
            exhaustive_evaluation=exhaustive_evaluation,
            full_method_evaluations=method_evaluations,
            exhaustive_scoring_summary=scoring,
            full_suite_summary=suite,
        )
