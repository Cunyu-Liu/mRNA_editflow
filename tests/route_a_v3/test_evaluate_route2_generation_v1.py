from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/evaluate_route2_generation_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_route2_generation_v1_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sources():
    return {
        "s1": {
            "source_sequence": "AAAA",
            "edit_budget": 2,
            "candidate_budget": 3,
            "source_critic_score": 0.0,
            "source_independent_evaluator_score": 0.5,
        }
    }


def _candidates():
    base = {
        "method_id": "beam",
        "source_key": "s1",
        "generator_nfe": 1,
        "critic_forwards": 1,
        "independent_evaluator_forwards": 1,
        "terminal_cause": "EXPLICIT_STOP",
        "source_critic_score": 0.0,
        "critic_forward_budget": 10,
    }
    return [
        dict(base, candidate_sequence="CAAA", critic_score=2.0, independent_evaluator_score=1.5),
        dict(base, candidate_sequence="CCAA", critic_score=1.0, independent_evaluator_score=1.0),
        dict(base, candidate_sequence="CAAA", critic_score=2.0, independent_evaluator_score=1.5),
    ]


def test_generation_metrics_keep_legality_diversity_cost_and_no_credit() -> None:
    module = _load()
    result = module.evaluate_generation(_sources(), _candidates())
    assert result["hard_legality_rate"] == 1.0
    assert result["edit_budget_violation_count"] == 0
    assert result["candidate_budget_violation_count"] == 0
    assert result["source_macro_unique_candidate_rate"] == pytest.approx(2 / 3)
    assert result["generated_candidates_grant_canonical_credit"] is False
    source = result["per_source"]["s1"]
    assert source["critic_score"]["max_uplift_over_source"] == 2.0
    assert source["compute"]["total_forward_equivalents"] == 9.0
    assert source["compute"]["critic_forward_budget"] == 10


def test_candidate_rows_can_supply_consistent_independent_source_score() -> None:
    module = _load()
    sources = _sources()
    sources["s1"]["source_independent_evaluator_score"] = None
    rows = [dict(row, source_independent_evaluator_score=0.5) for row in _candidates()]
    result = module.evaluate_generation(sources, rows)["per_source"]["s1"]
    assert result["independent_evaluator_score"]["max_uplift_over_source"] == 1.0


def test_illegal_and_over_budget_candidates_are_reported_not_hidden() -> None:
    module = _load()
    rows = _candidates()
    rows[0] = dict(rows[0], candidate_sequence="CCCC")
    result = module.evaluate_generation(_sources(), rows)
    assert result["hard_legality_rate"] == pytest.approx(2 / 3)
    assert result["edit_budget_violation_count"] == 1


def test_measured_neighborhood_recovery_is_separate_from_self_score() -> None:
    module = _load()
    measured = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "measured_direction_normalized_delta": 1.0, "pool_assignment": "DEVELOPMENT"},
        {"source_key": "s1", "candidate_sequence": "CCAA", "measured_direction_normalized_delta": 3.0, "pool_assignment": "DEVELOPMENT"},
        {"source_key": "s1", "candidate_sequence": "GAAA", "measured_direction_normalized_delta": 2.0, "pool_assignment": "DEVELOPMENT"},
    ]
    result = module.measured_neighborhood_metrics(_sources(), _candidates(), measured, k=1)
    source = result["per_source"]["s1"]
    assert source["candidate_recovery_rate"] == pytest.approx(2 / 3)
    assert source["measured_top_k_recall"] == 0.0
    assert source["measured_ndcg_at_k"] == 0.0
    assert source["normalized_regret"] == 1.0


def test_unmeasured_top_candidate_has_no_fabricated_measured_regret() -> None:
    module = _load()
    measured = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "measured_direction_normalized_delta": 1.0},
        {"source_key": "s1", "candidate_sequence": "CCAA", "measured_direction_normalized_delta": 3.0},
    ]
    candidates = _candidates()
    candidates[0] = dict(candidates[0], candidate_sequence="GAAA", critic_score=10.0)
    summary = module.measured_neighborhood_metrics(_sources(), candidates, measured, k=1)
    result = summary["per_source"]["s1"]
    assert result["measured_ndcg_at_k"] == 0.0
    assert result["selected_measured_outcome"] is None
    assert result["normalized_regret"] is None
    assert summary["source_normalized_regret_defined_count"] == 0
    assert summary["source_macro_normalized_regret"] is None


def test_unguided_generation_probability_can_rank_measured_candidates_without_critic() -> None:
    module = _load()
    measured = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "measured_direction_normalized_delta": 1.0},
        {"source_key": "s1", "candidate_sequence": "CCAA", "measured_direction_normalized_delta": 3.0},
    ]
    candidates = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "generation_score": -2.0},
        {"source_key": "s1", "candidate_sequence": "CCAA", "generation_score": -0.1},
    ]
    result = module.measured_neighborhood_metrics(_sources(), candidates, measured, k=1)["per_source"]["s1"]
    assert result["ranking_score_field"] == "generation_score"
    assert result["measured_ndcg_at_k"] == 1.0
    assert result["normalized_regret"] == 0.0


def test_measured_ranking_is_order_invariant_for_score_ties() -> None:
    module = _load()
    measured = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "measured_direction_normalized_delta": 1.0},
        {"source_key": "s1", "candidate_sequence": "GAAA", "measured_direction_normalized_delta": 0.0},
    ]
    candidates = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "generation_score": 0.0},
        {"source_key": "s1", "candidate_sequence": "GAAA", "generation_score": 0.0},
    ]
    first = module.measured_neighborhood_metrics(_sources(), candidates, measured, k=1)["per_source"]["s1"]
    second = module.measured_neighborhood_metrics(_sources(), list(reversed(candidates)), measured, k=1)["per_source"]["s1"]
    assert first == second
    assert first["measured_ndcg_at_k"] == pytest.approx(0.5)
    assert first["measured_top_k_recall"] == pytest.approx(0.5)
    assert first["normalized_regret"] == pytest.approx(0.5)


def test_unmeasured_member_of_top_score_tie_keeps_regret_undefined() -> None:
    module = _load()
    measured = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "measured_direction_normalized_delta": 1.0},
        {"source_key": "s1", "candidate_sequence": "GAAA", "measured_direction_normalized_delta": 0.0},
    ]
    candidates = [
        {"source_key": "s1", "candidate_sequence": "CAAA", "generation_score": 1.0},
        {"source_key": "s1", "candidate_sequence": "UAAA", "generation_score": 1.0},
    ]
    result = module.measured_neighborhood_metrics(_sources(), candidates, measured, k=1)["per_source"]["s1"]
    assert result["selected_measured_outcome"] is None
    assert result["normalized_regret"] is None


def test_cli_keeps_measured_evaluation_closed_before_freeze(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    manifest = tmp_path / "sources.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    measured = tmp_path / "measured.jsonl"
    manifest.write_text(json.dumps({"source_key": "s1", **_sources()["s1"]}) + "\n")
    candidates.write_text("\n".join(json.dumps(row) for row in _candidates()) + "\n")
    measured.write_text(json.dumps({
        "source_key": "s1",
        "candidate_sequence": "CAAA",
        "measured_direction_normalized_delta": 1.0,
        "pool_assignment": "EVALUATION",
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--source-manifest", str(manifest), "--candidates", str(candidates),
        "--measured-neighborhood", str(measured), "--measured-neighborhood-pool", "EVALUATION",
        "--output", str(tmp_path / "out.json"),
    ])
    with pytest.raises(module.GenerationEvaluationError, match="remain closed"):
        module.main()


def test_cli_allows_development_measured_neighborhood_while_evaluation_closed(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    manifest = tmp_path / "sources.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    measured = tmp_path / "measured.jsonl"
    output = tmp_path / "out.json"
    manifest.write_text(json.dumps({"source_key": "s1", **_sources()["s1"]}) + "\n")
    candidates.write_text("\n".join(json.dumps(row) for row in _candidates()) + "\n")
    measured.write_text(json.dumps({
        "source_key": "s1", "candidate_sequence": "CAAA",
        "measured_direction_normalized_delta": 1.0, "pool_assignment": "DEVELOPMENT",
    }) + "\n")
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--source-manifest", str(manifest), "--candidates", str(candidates),
        "--measured-neighborhood", str(measured), "--measured-neighborhood-pool", "DEVELOPMENT",
        "--evaluation-release-state", "CLOSED", "--output", str(output),
    ])
    assert module.main() == 0
    assert json.loads(output.read_text())["measured_neighborhood_pool"] == "DEVELOPMENT"
