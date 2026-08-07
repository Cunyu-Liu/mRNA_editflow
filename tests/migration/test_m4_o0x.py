"""O0-X R1 pool-safety and ranking metric tests (synthetic ordinary data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.o0x_search import (  # noqa: E402
    NDCG_GAIN_DEFINITION,
    SIGNED_MEAN_ENRICHMENT_STATUS,
    aggregate_strategy,
    candidate_identity,
    ndcg_at_k,
    pool_key,
    pool_seed,
    rank_metrics,
    run_benchmark_o0x,
    run_source_headline,
    top_decile_recall,
)
from scripts.m4_sparse.rerank import rerank_sources  # noqa: E402
from scripts.m4_sparse.train import build_folds  # noqa: E402


def _rec(
    delta,
    study,
    source_id,
    edit_count=1,
    endpoint="ep_x",
    benchmark="5U-A1",
    candidate_pos=None,
    source_sequence=None,
    **extra,
):
    source = source_sequence or (
        "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    )
    pos = (
        int(candidate_pos)
        if candidate_pos is not None
        else 5 + int(round(abs(float(delta)) * 10)) % min(50, len(source) - 5)
    )
    pos = min(max(pos, 0), len(source) - 1)
    replacement = {"A": "C", "C": "G", "G": "U", "U": "A"}.get(source[pos], "A")
    candidate = source[:pos] + replacement + source[pos + 1:]
    row = {
        "delta": float(delta),
        "study": study,
        "benchmark": benchmark,
        "source_id": source_id,
        "endpoint": endpoint,
        "context_id": extra.pop("context_id", "ctx"),
        "source_sequence": source,
        "candidate_sequence": candidate,
        "candidate_id": f"cand::{source_id}::{pos}",
        "edit_list": [{"op": "SUB", "pos": pos, "token": replacement}],
        "edit_count": edit_count,
        "source_value": 0.5,
        "candidate_value": float(delta) + 0.5,
    }
    row.update(extra)
    return row


# ---- identity / key invariants ----
def test_pool_key_is_endpoint_and_context_aware():
    a = _rec(1.0, "S1", "src1", endpoint="ep_a", context_id="ctx")
    b = _rec(1.0, "S1", "src1", endpoint="ep_b", context_id="ctx")
    c = _rec(1.0, "S1", "src1", endpoint="ep_a", context_id="ctx2")
    assert pool_key(a) != pool_key(b)
    assert pool_key(a) != pool_key(c)


def test_pool_key_fails_closed_without_source_or_endpoint():
    missing_source = _rec(1.0, "S1", "src1")
    missing_source["source_id"] = None
    missing_endpoint = _rec(1.0, "S1", "src1")
    missing_endpoint["endpoint"] = None
    assert pool_key(missing_source) is None
    assert pool_key(missing_endpoint) is None


def test_pool_key_fails_closed_without_context_or_with_context_conflict():
    missing = _rec(1.0, "S1", "src1", context_id=None)
    conflicting = _rec(
        1.0, "S1", "src1", context_id="ctx_a", assay_context="ctx_b"
    )
    assert pool_key(missing) is None
    assert pool_key(conflicting) is None


def test_candidate_identity_uses_sequence_not_observation_id():
    a = _rec(1.0, "S1", "src1", candidate_pos=8)
    b = dict(a, candidate_id="another_observation_id")
    assert candidate_identity(a) == candidate_identity(b)


# ---- NDCG / top-decile / rank metrics ----
def test_ndcg_perfect_ordering_is_1():
    assert ndcg_at_k(np.array([3.0, 2.0, 1.0]), 10) == pytest.approx(1.0)


def test_ndcg_transforms_negative_deltas_before_gain():
    assert ndcg_at_k(np.array([-1.0, -2.0, -3.0]), 10) == pytest.approx(1.0)


def test_ndcg_equal_relevance_is_explicitly_undefined():
    assert ndcg_at_k(np.array([1.0, 1.0]), 10) is None


def test_ndcg_reversed_heavy_tail_lower():
    rel = np.array([5.0, 1.0, 1.0, 1.0])
    perfect = ndcg_at_k(rel, 4)
    bad = ndcg_at_k(rel[::-1], 4)
    assert perfect > bad


def test_top_decile_recall_full_recovery():
    deltas = np.arange(1.0, 11.0)
    order = np.argsort(deltas)[::-1]
    assert top_decile_recall(deltas, order) == pytest.approx(1.0)


def test_rank_metrics_exact_ceiling_and_explicit_k_coverage():
    deltas = np.arange(1.0, 6.0)
    order = np.argsort(deltas)[::-1]
    metrics = rank_metrics(deltas, order, k=10)
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert metrics["ndcg_gain_definition"] == NDCG_GAIN_DEFINITION
    assert metrics["top_decile_recall"] == pytest.approx(1.0)
    assert metrics["normalized_regret"] == pytest.approx(0.0)
    assert metrics["effective_k"] == 5
    assert metrics["k_covers_pool"] is True
    assert metrics["oracle_normalized_gain_at_k"] is None
    assert metrics["oracle_normalized_gain_status"] == "UNDEFINED_K_COVERS_POOL"


def test_rank_metrics_topk_gain_is_bounded_and_oracle_normalized():
    deltas = np.arange(1.0, 6.0)
    exact = np.argsort(deltas)[::-1]
    reversed_order = exact[::-1]
    perfect = rank_metrics(deltas, exact, k=2)
    bad = rank_metrics(deltas, reversed_order, k=2)
    assert perfect["oracle_normalized_gain_at_k"] == pytest.approx(1.0)
    assert 0.0 <= bad["oracle_normalized_gain_at_k"] < 1.0


def test_signed_mean_enrichment_is_hold_and_negative_domain_gain_is_monotone():
    deltas = np.asarray([-5.0, -4.0, -3.0, -2.0, -1.0])
    oracle = np.argsort(-deltas, kind="stable")
    reversed_order = oracle[::-1]
    good = rank_metrics(deltas, oracle, k=2)
    bad = rank_metrics(deltas, reversed_order, k=2)

    # The bounded excess-gain definition preserves direction even when every
    # measured effect is negative.  Raw signed mean ratios do not, so they are
    # not emitted as a numeric gate value.
    assert good["oracle_normalized_gain_at_k"] == pytest.approx(1.0)
    assert good["oracle_normalized_gain_at_k"] > bad[
        "oracle_normalized_gain_at_k"
    ]
    assert good["signed_mean_enrichment_at_k"] is None
    assert good["signed_mean_enrichment_status"] == SIGNED_MEAN_ENRICHMENT_STATUS


def test_rank_metrics_rejects_incomplete_order():
    with pytest.raises(ValueError, match="complete permutation"):
        rank_metrics(np.array([1.0, 2.0]), np.array([1, 1]), k=1)


# ---- run_source_headline ----
def test_source_headline_has_all_strategies_no_scorer():
    records = [
        _rec(delta, "S1", "src1", edit_count=index, candidate_pos=10 + index)
        for index, delta in enumerate([1.0, 3.0, 2.0])
    ]
    result = run_source_headline(records, None, k=2)
    assert result["status"] == "ELIGIBLE"
    assert set(result["strategies"]) >= {
        "exact_enumeration", "random_legal", "greedy", "beam"
    }
    exact = result["strategies"]["exact_enumeration"]
    assert exact["ndcg_at_k"] == pytest.approx(1.0)
    assert exact["normalized_regret"] == pytest.approx(0.0)
    assert exact["query_count"] == 3
    assert exact["forward_equivalents"] == 0


def test_source_headline_adds_rerank_with_scorer():
    records = [
        _rec(delta, "S1", "src1", candidate_pos=10 + index)
        for index, delta in enumerate([1.0, 3.0, 2.0])
    ]
    prediction = np.array([1.0, 3.0, 2.0])
    result = run_source_headline(records, prediction, k=2)
    rerank = result["strategies"]["sparseeditformer_rerank"]
    assert rerank["ndcg_at_k"] == pytest.approx(1.0)
    assert rerank["forward_equivalents"] == 3
    assert rerank["query_count"] == 0


def test_source_headline_is_invariant_to_input_row_order():
    records = [
        _rec(delta, "S1", "src1", edit_count=index, candidate_pos=10 + index)
        for index, delta in enumerate([1.0, 4.0, 2.0, 3.0])
    ]
    prediction = np.asarray([1.0, 4.0, 2.0, 3.0])
    forward = run_source_headline(records, prediction, k=2, random_seed=2718)
    reverse = run_source_headline(
        list(reversed(records)), prediction[::-1], k=2, random_seed=2718
    )
    assert forward == reverse


def test_random_seed_is_deterministic_and_pool_specific():
    key_a = ("5U-A1", "S1", "ep", "ctx", "src_a")
    key_b = ("5U-A1", "S1", "ep", "ctx", "src_b")
    assert pool_seed(42, key_a) == pool_seed(42, key_a)
    assert pool_seed(42, key_a) != pool_seed(42, key_b)


def test_singleton_source_is_explicitly_excluded():
    result = run_source_headline([_rec(2.0, "S1", "src1")], np.array([2.0]))
    assert result == {
        "status": "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE",
        "n_candidates": 1,
        "strategies": {},
    }


def test_equal_relevance_source_is_explicitly_excluded():
    records = [
        _rec(2.0, "S1", "src1", candidate_pos=10),
        _rec(2.0, "S1", "src1", candidate_pos=11),
    ]
    result = run_source_headline(records, None)
    assert result["status"] == "EXCLUDED_DEGENERATE_EQUAL_RELEVANCE"
    assert result["strategies"] == {}


# ---- aggregation ----
def test_aggregate_strategy_macro_and_k_coverage():
    metrics = [
        {"n": 3, "ndcg_at_k": 0.8, "top_decile_recall": 0.6,
         "oracle_normalized_gain_at_k": 0.9, "enrichment_at_k": 0.9,
         "normalized_regret": 0.1, "k_covers_pool": False,
         "query_count": 3, "forward_equivalents": 0},
        {"n": 2, "ndcg_at_k": 0.6, "top_decile_recall": 0.4,
         "oracle_normalized_gain_at_k": None, "enrichment_at_k": None,
         "normalized_regret": 0.3, "k_covers_pool": True,
         "query_count": 2, "forward_equivalents": 0},
    ]
    aggregate = aggregate_strategy(metrics)
    assert aggregate["n_sources"] == 2
    assert aggregate["macro_ndcg_at_k"] == pytest.approx(0.7)
    assert aggregate["macro_oracle_normalized_gain_at_k"] == pytest.approx(0.9)
    assert aggregate["n_pools_where_k_covers_pool"] == 1
    assert aggregate["total_query_count"] == 5


# ---- run_benchmark_o0x pool eligibility ----
def test_run_benchmark_o0x_study_macro_and_singleton():
    rows = []
    rows += [
        _rec(delta, "S1", "s1a", edit_count=index, candidate_pos=10 + index)
        for index, delta in enumerate([1.0, 3.0, 2.0])
    ]
    rows += [_rec(5.0, "S1", "s1b", candidate_pos=20)]
    rows += [
        _rec(delta, "S2", "s2a", edit_count=index, candidate_pos=20 + index)
        for index, delta in enumerate([2.0, 4.0, 3.0])
    ]
    folds = build_folds(rows, "5U-A1", split="S4")

    def predict_fn(test_rows):
        return np.array([row["delta"] for row in test_rows])

    result = run_benchmark_o0x(rows, folds, predict_fn=predict_fn, k=2)
    assert result["benchmark"] == "5U-A1"
    assert result["summary"]["n_pools_total"] == 3
    assert result["summary"]["n_singleton_sources"] == 1
    assert result["summary"]["n_headline_sources"] == 2
    study_macro = result["study_macro"]
    assert study_macro["sparseeditformer_rerank"][
        "macro_over_studies_ndcg_at_k"
    ] == pytest.approx(1.0)
    assert study_macro["exact_enumeration"][
        "macro_over_studies_normalized_regret"
    ] == pytest.approx(0.0)


def test_same_candidate_in_two_endpoints_never_forms_pool():
    rows = [
        _rec(1.0, "S1", "src", endpoint="HEK", candidate_pos=10),
        _rec(2.0, "S1", "src", endpoint="SH", candidate_pos=10),
    ]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None, k=10)
    assert result["summary"]["n_pools_total"] == 2
    assert result["summary"]["n_singleton_sources"] == 2
    assert result["summary"]["n_headline_sources"] == 0
    assert result["summary"]["n_legacy_source_ids_spanning_endpoints"] == 1
    assert result["study_macro"] == {}


def test_duplicate_candidate_rows_are_collapsed_before_pool_size():
    duplicate = _rec(1.0, "S1", "src", candidate_pos=10)
    rows = [duplicate, dict(duplicate)]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["summary"]["n_pools_total"] == 1
    assert result["summary"]["n_duplicate_candidate_rows_collapsed"] == 1
    assert result["summary"]["n_singleton_sources"] == 1
    assert result["summary"]["n_headline_sources"] == 0


def test_conflicting_duplicate_candidate_invalidates_pool():
    first = _rec(1.0, "S1", "src", candidate_pos=10)
    conflicting = dict(first, delta=2.0)
    rows = [first, conflicting, _rec(3.0, "S1", "src", candidate_pos=11)]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    per_study = result["per_study"]["S1"]
    assert per_study["n_invalid_conflicting_duplicate_candidate"] == 1
    assert result["summary"]["n_headline_sources"] == 0


def test_source_sequence_mismatch_invalidates_source_id_pool():
    rows = [
        _rec(1.0, "S1", "src", candidate_pos=10),
        _rec(2.0, "S1", "src", candidate_pos=11,
             source_sequence="UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"),
    ]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["per_study"]["S1"]["n_invalid_source_sequence_mismatch"] == 1
    assert result["summary"]["n_headline_sources"] == 0


def test_missing_source_sequence_invalidates_pool_and_is_counted():
    first = _rec(1.0, "S1", "src", candidate_pos=10)
    second = _rec(2.0, "S1", "src", candidate_pos=11)
    first["source_sequence"] = None
    rows = [first, second]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["summary"]["n_invalid_missing_source_sequence"] == 1
    assert result["summary"]["n_headline_sources"] == 0


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_source_value_invalidates_pool(bad_value):
    rows = [
        _rec(1.0, "S1", "src", candidate_pos=10, source_value=bad_value),
        _rec(2.0, "S1", "src", candidate_pos=11),
    ]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["summary"]["n_invalid_missing_or_nonfinite_source_value"] == 1
    assert result["summary"]["n_headline_sources"] == 0


def test_conflicting_source_value_invalidates_pool_and_is_counted():
    rows = [
        _rec(1.0, "S1", "src", candidate_pos=10, source_value=0.5),
        _rec(2.0, "S1", "src", candidate_pos=11, source_value=0.75),
    ]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["summary"]["n_invalid_source_value_conflict"] == 1
    assert result["summary"]["n_headline_sources"] == 0


@pytest.mark.parametrize(
    ("field", "expected_atom"),
    [("source_id", "source_id"), ("context_id", "context")],
)
def test_missing_pool_atoms_are_reported_not_silently_coalesced(field, expected_atom):
    rows = [
        _rec(1.0, "S1", "src", candidate_pos=10),
        _rec(2.0, "S1", "src", candidate_pos=11),
    ]
    for row in rows:
        row[field] = None
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None)
    assert result["summary"]["n_rows_missing_pool_key"] == 2
    assert result["summary"]["missing_pool_atom_counts"][expected_atom] == 2
    assert result["summary"]["n_pools_total"] == 0


def test_direct_o0_and_rerank_entrypoints_reject_protected_rows():
    protected = _rec(1.0, "S1", "src", candidate_pos=10)
    protected["metadata"] = {"origin": {"accession": "GSE246381"}}
    folds = [{"held_out_study": "S1", "test": [protected]}]
    with pytest.raises(ValueError, match="protected_field_value"):
        run_benchmark_o0x([protected], folds, predict_fn=None)
    with pytest.raises(ValueError, match="protected_field_value"):
        rerank_sources([protected], np.asarray([1.0]))


def test_run_benchmark_o0x_without_scorer_skips_rerank():
    rows = [
        _rec(delta, "S1", "s1a", edit_count=index, candidate_pos=10 + index)
        for index, delta in enumerate([1.0, 3.0, 2.0])
    ]
    folds = build_folds(rows, "5U-A1", split="S4")
    result = run_benchmark_o0x(rows, folds, predict_fn=None, k=2)
    assert "sparseeditformer_rerank" not in result["study_macro"]
    assert "exact_enumeration" in result["study_macro"]
