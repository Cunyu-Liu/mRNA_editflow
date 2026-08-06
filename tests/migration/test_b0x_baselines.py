"""B0-X baseline metric computation tests (pure functions, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

B0X = Path(__file__).resolve().parents[2] / "scripts" / "b0x"
sys.path.insert(0, str(B0X))

import run_effect_baselines as R  # noqa: E402
import run_search_baselines as S  # noqa: E402


def _rec(delta, source_id="src1", candidate_value=None, source_value=None):
    return {
        "delta": delta,
        "candidate_value": candidate_value if candidate_value is not None else delta,
        "source_value": source_value,
        "source_sequence": "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",
        "candidate_sequence": "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGG",
        "edit_list": [{"op": "SUB", "pos": 3, "token": "G"}],
        "source_id": source_id,
        "study": "TEST",
        "endpoint": "ep_x",
    }


def test_context_metrics_perfect():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # perfect monotone
    m = R.context_metrics(true, pred)
    assert m["delta_spearman"] == pytest.approx(1.0)
    assert m["sign_accuracy"] == 1.0
    assert m["top10_enrichment"] == 1.0


def test_context_metrics_reversed():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # anti-monotone
    m = R.context_metrics(true, pred)
    assert m["delta_spearman"] == pytest.approx(-1.0)


def test_macro_averages():
    cms = [
        {"delta_spearman": 0.5, "n": 10},
        {"delta_spearman": 0.7, "n": 20},
        {"delta_spearman": None, "n": 3},
    ]
    m = R._macro(cms)
    # None values excluded
    assert abs(m["macro_delta_spearman"] - 0.6) < 1e-9


def test_ndcg_at_k_perfect():
    rel = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert abs(S.ndcg_at_k(rel, 5) - 1.0) < 1e-9


def test_rank_metrics_oracle():
    deltas = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    order = np.argsort(deltas)[::-1]  # oracle
    m = S.rank_metrics(deltas, order, k=5)
    assert m["top_decile_recall"] == 1.0
    assert m["normalized_regret"] == 0.0
    assert abs(m["ndcg_at_k"] - 1.0) < 1e-9


def test_bl_mean():
    train = [_rec(1.0), _rec(3.0), _rec(5.0)]
    test = [_rec(0), _rec(0)]
    pred = R.bl_mean(train, test)
    assert np.allclose(pred, 3.0)


def test_bl_source_mean():
    train = [_rec(1.0, source_id="a"), _rec(3.0, source_id="a"), _rec(10.0, source_id="b")]
    test = [_rec(0, source_id="a"), _rec(0, source_id="zzz")]
    pred = R.bl_source_mean(train, test)
    # a -> mean(1,3)=2 ; unseen zzz -> global mean (1+3+10)/3
    assert abs(pred[0] - 2.0) < 1e-9
    assert abs(pred[1] - (14.0 / 3.0)) < 1e-9


def test_bl_abs_minus_abs_runs():
    # RidgeCV must not receive unsupported kwargs (regression for random_state bug).
    train = [_rec(d, candidate_value=d + 1.0, source_value=1.0) for d in (1.0, 2.0, 5.0)]
    test = [_rec(0, candidate_value=None, source_value=None)]
    pred = R.bl_abs_minus_abs(train, test)
    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))


def test_bl_ridge_feature_runs_with_empty_token():
    # Regression: edit_features with an empty token must not break feature baselines.
    tr = [_rec(d, candidate_value=None, source_value=None) for d in (1.0, 2.0, 3.0)]
    te = [_rec(0, candidate_value=None, source_value=None)]
    for row in tr + te:
        row["edit_list"] = [{"op": "SUB", "pos": 3, "token": ""}]
    pred = R.bl_diff_features(tr, te)
    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))