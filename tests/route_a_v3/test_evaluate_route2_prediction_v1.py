from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/evaluate_route2_prediction_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("evaluate_route2_prediction_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observations():
    return [
        {"canonical_record_id": "A1", "study_unit_id": "S", "source_id": "A", "biological_context_id": "C", "endpoint_id": "E", "stratum": ("S", "3UTR", "E"), "observed": -1.0},
        {"canonical_record_id": "A2", "study_unit_id": "S", "source_id": "A", "biological_context_id": "C", "endpoint_id": "E", "stratum": ("S", "3UTR", "E"), "observed": 0.0},
        {"canonical_record_id": "A3", "study_unit_id": "S", "source_id": "A", "biological_context_id": "C", "endpoint_id": "E", "stratum": ("S", "3UTR", "E"), "observed": 2.0},
        {"canonical_record_id": "B1", "study_unit_id": "S", "source_id": "B", "biological_context_id": "C", "endpoint_id": "E", "stratum": ("S", "3UTR", "E"), "observed": 0.5},
        {"canonical_record_id": "B2", "study_unit_id": "S", "source_id": "B", "biological_context_id": "C", "endpoint_id": "E", "stratum": ("S", "3UTR", "E"), "observed": 1.5},
    ]


def test_perfect_predictions_close_numeric_and_ranking_metrics() -> None:
    module = _module()
    observations = _observations()
    predictions = {row["canonical_record_id"]: row["observed"] for row in observations}
    result = module.evaluate(observations, predictions, k=1)
    assert result["overall_numeric"]["mae"] == 0.0
    assert result["overall_numeric"]["spearman"] == pytest.approx(1.0)
    assert result["rankable_source_group_count"] == 2
    assert result["source_macro_ndcg_at_k"] == 1.0
    assert result["source_macro_top_k_recall"] == 1.0
    assert result["source_macro_normalized_regret"] == 0.0
    assert result["source_macro_top_1_accuracy"] == 1.0
    assert result["source_macro_pairwise_accuracy"] == 1.0
    assert result["overall_numeric"]["sign_accuracy"] == 1.0
    assert set(result["task_numeric"]) == {"3UTR|E"}
    assert result["task_count"] == result["task_spearman_defined_count"] == 1
    assert result["task_macro_spearman"] == pytest.approx(1.0)


def test_ranking_ndcg_shifts_negative_relevance_without_changing_order() -> None:
    module = _module()
    observed = np.asarray([-4.0, -2.0, -3.0])
    perfect = module.ranking_metrics(observed, observed.copy(), 3)
    reversed_result = module.ranking_metrics(observed, -observed, 3)
    assert perfect["ndcg_at_k"] == 1.0
    assert reversed_result["ndcg_at_k"] < 1.0
    assert reversed_result["normalized_regret"] == 1.0


def test_ranking_metrics_are_order_invariant_for_prediction_ties() -> None:
    module = _module()
    observed = np.asarray([1.0, 0.0])
    predicted = np.asarray([0.0, 0.0])
    first = module.ranking_metrics(observed, predicted, 1)
    second = module.ranking_metrics(observed[::-1], predicted[::-1], 1)
    assert first == second
    assert first["ndcg_at_k"] == pytest.approx(0.5)
    assert first["top_k_recall"] == pytest.approx(0.5)
    assert first["top_1_accuracy"] == pytest.approx(0.5)
    assert first["normalized_regret"] == pytest.approx(0.5)


def test_source_group_bootstrap_reports_spearman_and_mae_improvements() -> None:
    module = _module()
    observations = _observations()
    model = {row["canonical_record_id"]: row["observed"] for row in observations}
    baseline = {row["canonical_record_id"]: -row["observed"] for row in observations}
    comparison = module.paired_group_bootstrap(observations, model, baseline, 200, 7, 1)
    assert comparison["analysis_unit"] == "SOURCE_GROUP"
    assert comparison["task_macro_spearman"]["improvement"] > 0.0
    assert comparison["task_macro_spearman"]["defined_bootstrap_iterations"] > 0
    assert comparison["task_macro_spearman"]["bootstrap_ci_95"][0] > 0.0
    assert comparison["mae"]["mean_improvement"] > 0.0
    assert comparison["mae"]["bootstrap_ci_95"][0] > 0.0
    assert comparison["ranking"]["rankable_source_group_count"] == 2
    assert comparison["ranking"]["ndcg"]["mean_improvement"] >= 0.0
    assert comparison["ranking"]["regret_reduction"]["mean_improvement"] > 0.0
    assert comparison["ranking"]["top_1"]["mean_improvement"] > 0.0


def test_manifest_can_select_loso_study_across_fixed_splits(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"canonical_record_id": "a", "split": "TRAIN", "study_unit_id": "H", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "b", "split": "TEST", "study_unit_id": "H", "pool_assignment": "DEVELOPMENT"},
        {"canonical_record_id": "c", "split": "VALIDATION", "study_unit_id": "O", "pool_assignment": "DEVELOPMENT"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected, pools = module.load_manifest(path, requested_study="H")
    assert selected == {"a", "b"}
    assert pools == {"DEVELOPMENT"}
    with pytest.raises(module.EvaluationError, match="exactly one"):
        module.load_manifest(path, requested_split="TRAIN", requested_study="H")


def test_evaluation_study_selection_remains_closed_until_full_freeze(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "evaluation_manifest.jsonl"
    path.write_text(json.dumps({
        "canonical_record_id": "e",
        "split": "EVALUATION_ZERO_SHOT",
        "study_unit_id": "GSE232572",
        "pool_assignment": "EVALUATION",
    }) + "\n")
    selected, pools = module.load_manifest(path, requested_study="GSE232572")
    assert selected == {"e"}
    with pytest.raises(module.EvaluationError, match="remain closed"):
        module.require_evaluation_release(pools, "CLOSED")
    module.require_evaluation_release(pools, "PREDICTOR_GENERATOR_AND_BASELINES_FROZEN")
