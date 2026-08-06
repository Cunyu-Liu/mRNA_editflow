"""M4 evaluation / S4-splitter / reranker unit tests (pure, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.evaluate import (  # noqa: E402
    context_metrics, macro_metrics, abs_candidate_baseline,
)
from scripts.m4_sparse.rerank import (  # noqa: E402
    ndcg_at_k, rerank_sources, top_decile_recall,
)
from scripts.m4_sparse.train import build_folds  # noqa: E402


def _rec(delta, study, source_id, endpoint="ep_x", benchmark="5U-A1"):
    s = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    c = s[:10] + "G" + s[11:]
    return {
        "delta": float(delta), "study": study, "benchmark": benchmark,
        "source_id": source_id, "endpoint": endpoint,
        "source_sequence": s, "candidate_sequence": c,
        "edit_list": [{"op": "SUB", "pos": 10, "token": "G"}],
        "source_value": 0.5,
        "candidate_value": float(delta) + 0.5,
    }


# ---- context metrics ----
def test_context_metrics_perfect():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = context_metrics(true, pred)
    assert m["delta_spearman"] == pytest.approx(1.0)
    assert m["sign_accuracy"] == 1.0
    # top10_enrichment = mean(true_delta[pred_top])/mean(true_delta[true_top]);
    # for a perfect ordering this equals 1.0 (predicted top == true top).
    assert m["top10_enrichment"] == pytest.approx(1.0)
    # top10pct_enrichment is ENRICHMENT-OVER-RANDOM:
    #   mean(true_delta[pred_top_10%]) / mean(true_delta[all]).
    # For true=[1,2,3,4,5], top-10% = {5.0}, overall mean = 3.0 -> 5/3.
    assert m["top10pct_enrichment"] == pytest.approx(5.0 / 3.0)


def test_context_metrics_reversed():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    m = context_metrics(true, pred)
    assert m["delta_spearman"] == pytest.approx(-1.0)


def test_context_metrics_small_context_spearman_none():
    m = context_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))  # n<3
    assert m["delta_spearman"] is None
    assert m["sign_accuracy"] is not None


def test_macro_metrics_excludes_none():
    cms = [
        {"delta_spearman": 0.5, "n": 10},
        {"delta_spearman": 0.7, "n": 20},
        {"delta_spearman": None, "n": 3},
    ]
    m = macro_metrics(cms)
    assert abs(m["macro_delta_spearman"] - 0.6) < 1e-9


# ---- S4 splitter ----
def test_build_folds_study_disjoint_no_source_overlap():
    rows = [_rec(float(i), study="S%d" % (i % 3), source_id="src_%d_%d" % (i % 3, i))
            for i in range(30)]
    folds = build_folds(rows, "5U-A1")
    assert len(folds) == 3
    for f in folds:
        train_studies = {r["study"] for r in f["train"]}
        test_studies = {r["study"] for r in f["test"]}
        assert not (train_studies & test_studies), "study overlap in S4 fold"
        train_src = {r["source_id"] for r in f["train"]}
        test_src = {r["source_id"] for r in f["test"]}
        assert not (train_src & test_src), "source overlap in S4 fold"


def test_build_folds_covers_every_record_exactly_once():
    rows = [_rec(float(i), study="S%d" % (i % 3), source_id="src%d" % i)
            for i in range(30)]
    folds = build_folds(rows, "5U-A1")
    seen = set()
    for f in folds:
        for r in f["test"]:
            seen.add(r["source_id"])
    assert len(seen) == len(rows), "test folds must partition all records"


def test_build_folds_keeps_benchmark_separate():
    a = [_rec(1.0, study="S1", source_id="s%d" % i, benchmark="5U-A1") for i in range(4)]
    b = [_rec(1.0, study="S1", source_id="t%d" % i, benchmark="3U-A1") for i in range(4)]
    folds = build_folds(a + b, "5U-A1")
    assert len(folds) == 1
    assert all(r["benchmark"] == "5U-A1" for r in folds[0]["test"])


# ---- reranker ----
def test_ndcg_at_k_perfect_and_reversed():
    rel = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert abs(ndcg_at_k(rel, 5) - 1.0) < 1e-9
    assert ndcg_at_k(rel[::-1], 5) < 1.0


def test_rerank_sources_perfect_ordering():
    # three sources, each with graded candidates; pred ranks by true delta
    records, pred = [], []
    for sid in ("A", "B", "C"):
        for j in range(4):
            records.append(_rec(float(j), study="S1", source_id=sid))
            pred.append(float(j))
    out = rerank_sources(records, np.array(pred))
    assert out["n_sources"] == 3
    assert abs(out["macro_ndcg_at_10"] - 1.0) < 1e-9
    assert abs(out["macro_top_decile_recall"] - 1.0) < 1e-9


def test_rerank_sources_ignores_singletons():
    records = [_rec(1.0, study="S1", source_id="single")]
    out = rerank_sources(records, np.array([0.5]))
    assert out["n_sources"] == 0
    assert out["n_singleton_sources"] == 1
    assert out["macro_ndcg_at_10"] is None


def test_top_decile_recall_perfect():
    rel = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    assert abs(top_decile_recall(rel) - 1.0) < 1e-9


# ---- abs_candidate baseline ----
def test_abs_candidate_finite_and_uses_measured_source():
    train = [_rec(1.0, study="S1", source_id="a%d" % i) for i in range(20)]
    train += [_rec(2.0, study="S1", source_id="b%d" % i) for i in range(20)]
    test = [_rec(0.0, study="S2", source_id="z%d" % i) for i in range(10)]
    pred = abs_candidate_baseline(train, test)
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
