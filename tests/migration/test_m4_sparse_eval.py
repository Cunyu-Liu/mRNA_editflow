"""M4 evaluation / S4-splitter / reranker unit tests (pure, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.evaluate import (  # noqa: E402
    context_metrics, macro_metrics, abs_candidate_baseline, predict_model,
    predict_model_outputs,
)
from scripts.m4_sparse import config as C  # noqa: E402
from scripts.m4_sparse.dataset import UNKNOWN_CONTEXT, build_vocab  # noqa: E402
from scripts.m4_sparse.rerank import (  # noqa: E402
    ndcg_at_k, rerank_sources, top_decile_recall,
)
from scripts.m4_sparse.train import build_folds  # noqa: E402


def _rec(delta, study, source_id, endpoint="ep_x", benchmark="5U-A1"):
    s = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    pos = 8 + int(abs(float(delta))) % 12
    token = {"A": "C", "C": "G", "G": "U", "U": "A"}[s[pos]]
    c = s[:pos] + token + s[pos + 1:]
    return {
        "delta": float(delta), "study": study, "benchmark": benchmark,
        "source_id": source_id, "endpoint": endpoint,
        "context_id": "ctx",
        "source_sequence": s, "candidate_sequence": c,
        "edit_list": [{"op": "SUB", "pos": pos, "token": token}],
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
    assert m["top10_enrichment"] is None  # k=10 covers this entire context
    assert m["top10pct_enrichment"] is None  # frozen signed-ratio gate invalidated
    assert m["top10pct_oracle_normalized_excess_gain"] == pytest.approx(1.0)
    assert m["top10pct_enrichment_status"].startswith("HOLD_")


def test_context_metrics_reversed():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    m = context_metrics(true, pred)
    assert m["delta_spearman"] == pytest.approx(-1.0)
    assert m["top10pct_oracle_normalized_excess_gain"] < 1.0


def test_signed_top10pct_gain_is_order_monotone_for_negative_domain():
    true = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    perfect = context_metrics(true, true)
    reversed_ = context_metrics(true, true[::-1])
    assert perfect["top10pct_oracle_normalized_excess_gain"] == pytest.approx(1.0)
    assert (perfect["top10pct_oracle_normalized_excess_gain"]
            > reversed_["top10pct_oracle_normalized_excess_gain"])
    assert perfect["top10pct_enrichment"] is None


def test_fixed_top10_signed_ratio_is_held_and_safe_diagnostic_is_monotone_negative_domain():
    true = -np.arange(1.0, 21.0)
    perfect = context_metrics(true, true)
    reversed_ = context_metrics(true, true[::-1])
    assert perfect["top10_enrichment"] is None
    assert reversed_["top10_enrichment"] is None
    assert perfect["top10_enrichment_status"].startswith("HOLD_")
    assert perfect["top10_oracle_normalized_excess_gain"] == pytest.approx(1.0)
    assert (perfect["top10_oracle_normalized_excess_gain"]
            > reversed_["top10_oracle_normalized_excess_gain"])
    # This is the exact backwards behavior that disqualifies the old ratio.
    assert (reversed_["top10_enrichment_legacy_signed_oracle_ratio"]
            > perfect["top10_enrichment_legacy_signed_oracle_ratio"])


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


def test_build_folds_excludes_source_sequence_alias_across_studies():
    held = _rec(1.0, study="held", source_id="held_id")
    alias = _rec(2.0, study="train", source_id="different_id")
    distinct = _rec(3.0, study="train", source_id="distinct_id")
    distinct["source_sequence"] = "U" * len(distinct["source_sequence"])
    fold = next(f for f in build_folds([held, alias, distinct], "5U-A1")
                if f["held_out_study"] == "held")
    assert alias not in fold["train"]
    assert distinct in fold["train"]
    assert fold["n_train_excluded_source_overlap"] == 1


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


def test_predict_model_routes_s4_study_to_trained_unknown_and_passes_masks():
    rows = [_rec(1.0, study="held", source_id="a"),
            _rec(-1.0, study="held", source_id="b")]
    vocab = build_vocab(rows)
    cfg = C.get_config()
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])
    cfg.SIGN_TEMPERATURE = 1.0
    cfg.SIGN_DECISION_RULE = C.SIGN_DECISION_RULE

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.study = None
            self.masks = None

        def forward(self, src, cand, edit, study, endpoint, bench,
                    src_pad_mask=None, cand_pad_mask=None):
            self.study = study.detach().clone()
            self.masks = (src_pad_mask.detach().clone(),
                          cand_pad_mask.detach().clone())
            z = src[:, 0, 0] * 0.0
            return {"mean": z, "logvar": z, "sign": z, "rank": z}

    model = CaptureModel()
    pred = predict_model(model, rows, vocab, cfg, torch.device("cpu"))
    assert pred.shape == (2,)
    assert torch.all(model.study == vocab["study"][UNKNOWN_CONTEXT])
    assert model.masks[0].dtype == torch.bool
    assert model.masks[1].dtype == torch.bool


def test_reported_model_sign_accuracy_is_derived_from_delta_mean_not_independent_head():
    rows = [_rec(1.0, study="held", source_id="positive"),
            _rec(-1.0, study="held", source_id="negative")]
    vocab = build_vocab(rows)
    cfg = C.get_config()
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])
    cfg.SIGN_TEMPERATURE = 2.0
    cfg.SIGN_DECISION_RULE = C.SIGN_DECISION_RULE

    class OpposedHeads(torch.nn.Module):
        def forward(self, src, cand, edit, study, endpoint, bench,
                    src_pad_mask=None, cand_pad_mask=None):
            # The predicted delta mean gets both signs right.  A contradictory
            # legacy sign output must be ignored by the R3-P0 evaluator.
            mean = torch.tensor([1.0, -1.0], device=src.device)[:src.size(0)]
            sign = torch.tensor([-10.0, 10.0], device=src.device)[:src.size(0)]
            zero = mean * 0.0
            return {"mean": mean, "logvar": zero, "sign": sign, "rank": zero}

    outputs = predict_model_outputs(
        OpposedHeads(), rows, vocab, cfg, torch.device("cpu"))
    true = np.asarray([row["delta"] for row in rows])
    reported = context_metrics(
        true, outputs["delta"], sign_prediction=outputs["sign_decision"],
        sign_rule=outputs["sign_decision_rule"])
    assert reported["sign_accuracy"] == 1.0
    assert reported["sign_decision_rule"] == cfg.SIGN_DECISION_RULE


def test_positive_temperature_changes_probability_not_hard_sign():
    rows = [_rec(1.0, study="held", source_id="positive"),
            _rec(-1.0, study="held", source_id="negative")]
    vocab = build_vocab(rows)

    class FixedMean(torch.nn.Module):
        def forward(self, src, cand, edit, study, endpoint, bench,
                    src_pad_mask=None, cand_pad_mask=None):
            mean = torch.tensor([2.0, -1.0], device=src.device)[:src.size(0)]
            zero = mean * 0.0
            return {"mean": mean, "logvar": zero,
                    "sign": -mean, "rank": -mean}

    cfg = C.get_config()
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])
    cfg.SIGN_DECISION_RULE = C.SIGN_DECISION_RULE
    cfg.SIGN_TEMPERATURE = 1.0
    cold = predict_model_outputs(
        FixedMean(), rows, vocab, cfg, torch.device("cpu"))
    cfg.SIGN_TEMPERATURE = 7.0
    warm = predict_model_outputs(
        FixedMean(), rows, vocab, cfg, torch.device("cpu"))
    assert np.array_equal(cold["sign_decision"], warm["sign_decision"])
    assert np.array_equal(cold["sign_logit"], warm["sign_logit"])
    assert not np.allclose(cold["sign_probability"],
                           warm["sign_probability"])
