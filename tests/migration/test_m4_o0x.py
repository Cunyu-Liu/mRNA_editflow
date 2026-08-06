"""O0-X closed measured-space optimization unit tests (pure, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.o0x_search import (  # noqa: E402
    ndcg_at_k, top_decile_recall, rank_metrics,
    run_source_headline, aggregate_strategy, run_benchmark_o0x,
)
from scripts.m4_sparse.train import build_folds  # noqa: E402


def _rec(delta, study, source_id, edit_count=1, endpoint="ep_x", benchmark="5U-A1"):
    s = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    c = s[:10] + "G" + s[11:]
    return {
        "delta": float(delta), "study": study, "benchmark": benchmark,
        "source_id": source_id, "endpoint": endpoint,
        "source_sequence": s, "candidate_sequence": c,
        "edit_list": [{"op": "SUB", "pos": 10, "token": "G"}],
        "edit_count": edit_count,
        "source_value": 0.5,
        "candidate_value": float(delta) + 0.5,
    }


# ---- NDCG / top-decile / rank metrics ----
def test_ndcg_perfect_ordering_is_1():
    # rel is the relevance of a ranked list (best first -> descending);
    # min-max normalized per pool, so a perfect ordering gives 1.0.
    assert ndcg_at_k(np.array([3.0, 2.0, 1.0]), 10) == pytest.approx(1.0)


def test_ndcg_handles_negative_deltas():
    # signed (possibly all-negative) deltas must not collapse NDCG to 0 for a
    # perfect ordering; min-max normalization makes it well-defined.
    assert ndcg_at_k(np.array([-1.0, -2.0, -3.0]), 10) == pytest.approx(1.0)


def test_ndcg_reversed_heavy_tail_lower():
    rel = np.array([5.0, 1.0, 1.0, 1.0])
    perfect = ndcg_at_k(rel, 4)
    reversed_order = np.array([5.0, 1.0, 1.0, 1.0])[::-1]
    bad = ndcg_at_k(reversed_order, 4)
    assert perfect > bad


def test_top_decile_recall_full_recovery():
    deltas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    order = np.argsort(deltas)[::-1]
    assert top_decile_recall(deltas, order) == pytest.approx(1.0)


def test_rank_metrics_exact_ceiling():
    deltas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    order = np.argsort(deltas)[::-1]
    m = rank_metrics(deltas, order, k=5)
    assert m["ndcg_at_k"] == pytest.approx(1.0)
    assert m["top_decile_recall"] == pytest.approx(1.0)
    assert m["normalized_regret"] == pytest.approx(0.0)


# ---- run_source_headline ----
def test_source_headline_has_all_strategies_no_scorer():
    recs = [_rec(d, "S1", "src1", edit_count=i) for i, d in enumerate([1.0, 3.0, 2.0])]
    out = run_source_headline(recs, None, k=10)
    assert set(out["strategies"].keys()) >= {
        "exact_enumeration", "random_legal", "greedy", "beam"}
    # exact enumeration is the oracle ceiling
    exact = out["strategies"]["exact_enumeration"]
    assert exact["ndcg_at_k"] == pytest.approx(1.0)
    assert exact["normalized_regret"] == pytest.approx(0.0)
    # query count = pool size, forward equivalents = 0 for non-model strategies
    assert exact["query_count"] == 3
    assert exact["forward_equivalents"] == 0


def test_source_headline_adds_rerank_with_scorer():
    recs = [_rec(d, "S1", "src1") for d in [1.0, 3.0, 2.0]]
    pred = np.array([1.0, 3.0, 2.0])  # perfect: higher pred for higher delta
    out = run_source_headline(recs, pred, k=10)
    rk = out["strategies"]["sparseeditformer_rerank"]
    assert rk["ndcg_at_k"] == pytest.approx(1.0)
    assert rk["forward_equivalents"] == 3
    assert rk["query_count"] == 0


def test_singleton_source_excluded_from_headline():
    # run_source_headline is only called for non-singleton, but verify a
    # one-record pool ranks consistently (no crash, n=1).
    recs = [_rec(2.0, "S1", "src1")]
    out = run_source_headline(recs, np.array([2.0]), k=10)
    assert out["n_candidates"] == 1
    assert out["strategies"]["exact_enumeration"]["n"] == 1


# ---- aggregation ----
def test_aggregate_strategy_macro():
    ms = [
        {"n": 3, "ndcg_at_k": 0.8, "top_decile_recall": 0.6,
         "enrichment_at_k": 1.2, "normalized_regret": 0.1,
         "query_count": 3, "forward_equivalents": 0},
        {"n": 2, "ndcg_at_k": 0.6, "top_decile_recall": 0.4,
         "enrichment_at_k": 1.0, "normalized_regret": 0.3,
         "query_count": 2, "forward_equivalents": 0},
    ]
    agg = aggregate_strategy(ms)
    assert agg["n_sources"] == 2
    assert agg["macro_ndcg_at_k"] == pytest.approx(0.7)
    assert agg["total_query_count"] == 5
    assert agg["total_forward_equivalents"] == 0


def test_aggregate_strategy_handles_none_metrics():
    ms = [{"n": 3, "ndcg_at_k": None, "top_decile_recall": 0.6,
           "enrichment_at_k": None, "normalized_regret": 0.1,
           "query_count": 3, "forward_equivalents": 0}]
    agg = aggregate_strategy(ms)
    assert agg["macro_ndcg_at_k"] is None
    assert agg["macro_top_decile_recall"] == pytest.approx(0.6)


# ---- run_benchmark_o0x ----
def test_run_benchmark_o0x_study_macro_and_singleton():
    rows = []
    # S1: one non-singleton source (3 candidates) + one singleton
    rows += [_rec(d, "S1", "s1a", edit_count=i) for i, d in enumerate([1.0, 3.0, 2.0])]
    rows += [_rec(5.0, "S1", "s1b")]
    # S2: one non-singleton source
    rows += [_rec(d, "S2", "s2a", edit_count=i) for i, d in enumerate([2.0, 4.0, 3.0])]
    folds = build_folds(rows, "5U-A1", split="S4")

    def predict_fn(test_rows):
        # perfect predictor: return delta directly
        return np.array([r["delta"] for r in test_rows])

    res = run_benchmark_o0x(rows, folds, predict_fn=predict_fn, k=10)
    assert res["benchmark"] == "5U-A1"
    assert res["summary"]["n_pools_total"] == 3
    assert res["summary"]["n_singleton_sources"] == 1
    assert res["summary"]["n_headline_sources"] == 2
    sm = res["study_macro"]
    assert "sparseeditformer_rerank" in sm
    assert sm["sparseeditformer_rerank"]["macro_over_studies_ndcg_at_k"] == pytest.approx(1.0)
    assert sm["exact_enumeration"]["macro_over_studies_normalized_regret"] == pytest.approx(0.0)


def test_run_benchmark_o0x_without_scorer_skips_rerank():
    rows = [_rec(d, "S1", "s1a", edit_count=i) for i, d in enumerate([1.0, 3.0, 2.0])]
    folds = build_folds(rows, "5U-A1", split="S4")
    res = run_benchmark_o0x(rows, folds, predict_fn=None, k=10)
    assert "sparseeditformer_rerank" not in res["study_macro"]
    assert "exact_enumeration" in res["study_macro"]