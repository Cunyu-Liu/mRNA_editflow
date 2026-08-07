"""Independent top-K paired reranker over R1-safe measured candidate pools.

Given a trained SparseEditFormer and a set of (source, candidate) records,
group candidates by benchmark, study, endpoint, assay context and source ID,
then require consistent source sequence/value provenance.  Rank each eligible
pool by model-predicted delta and report ordinary-data diagnostics.  These
developmental pools remain explicitly unqualified for publication claims.

NDCG relevance is min-max normalized per source pool so the (2**rel - 1) gain
term stays numerically bounded even for large measured deltas.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from .o0x_search import (
    METRIC_CONTRACT_VERSION,
    POOL_KEY_DEFINITION,
    _prepare_pool,
    pool_key,
    rank_metrics,
)
from .ordinary_io import assert_ordinary_rows


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """relevance sorted best->worst according to the ranked order."""
    relevance = np.asarray(relevance, dtype=float)
    k = min(k, len(relevance))
    if k == 0 or len(relevance) == 0:
        return 0.0
    dcg = 0.0
    for i in range(k):
        dcg += (2 ** relevance[i] - 1) / np.log2(i + 2)
    ideal = np.sort(relevance)[::-1][:k]
    idcg = 0.0
    for i in range(k):
        idcg += (2 ** ideal[i] - 1) / np.log2(i + 2)
    return float(dcg / idcg) if idcg > 0 else 0.0


def top_decile_recall(relevance_desc: np.ndarray) -> float:
    """Fraction of the measured top-decile that the model placed in its top decile."""
    n = len(relevance_desc)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(n * 0.10)))
    true_top = np.argsort(relevance_desc)[::-1][:k]
    pred_top = set(range(k))
    return float(len({int(i) for i in true_top} & pred_top) / k)


def rerank_sources(records: List[dict], delta_pred: np.ndarray,
                   k_ndcg: int = 10) -> Dict:
    """Rank only R1-qualified endpoint/context-aware candidate pools.

    records and delta_pred must be aligned (same order).  Returns per-source
    ranking metrics macro-averaged over non-singleton sources.
    """
    assert_ordinary_rows(records, purpose="R1 reranker")
    if np.asarray(delta_pred).shape != (len(records),):
        raise ValueError("delta_pred must contain one score per record")
    groups = defaultdict(list)
    missing_pool_key = 0
    for i, r in enumerate(records):
        key = pool_key(r)
        if key is None:
            missing_pool_key += 1
            continue
        groups[key].append((i, r))
    per_source = []
    status_counts = defaultdict(int)
    for key, items in groups.items():
        prepared = _prepare_pool(items)
        status_counts[prepared["status"]] += 1
        if prepared["status"] != "ELIGIBLE":
            continue
        unique = prepared["entries"]
        deltas = np.asarray([row["delta"] for _, row in unique], dtype=float)
        scores = np.asarray([delta_pred[index] for index, _ in unique], dtype=float)
        order = np.argsort(-scores, kind="stable")
        metrics = rank_metrics(deltas, order, k_ndcg)
        per_source.append({
            "pool_key": list(key),
            "n_candidates": len(unique),
            "ndcg_at_10": metrics["ndcg_at_k"],
            "top_decile_recall": metrics["top_decile_recall"],
            "oracle_normalized_gain_at_k": metrics[
                "oracle_normalized_gain_at_k"],
            "normalized_regret": metrics["normalized_regret"],
        })
    if not per_source:
        return {"n_sources": 0,
                "n_singleton_sources": status_counts[
                    "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE"],
                "macro_ndcg_at_10": None, "macro_top_decile_recall": None,
                "per_source": [], "pool_status_counts": dict(status_counts),
                "n_rows_missing_pool_key": missing_pool_key,
                "metric_contract_version": METRIC_CONTRACT_VERSION,
                "pool_key_definition": POOL_KEY_DEFINITION,
                "publication_grouping_qualified": False}
    return {
        "n_sources": len(per_source),
        "n_singleton_sources": status_counts[
            "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE"],
        "macro_ndcg_at_10": float(np.mean([s["ndcg_at_10"] for s in per_source])),
        "macro_top_decile_recall": float(np.mean([s["top_decile_recall"] for s in per_source])),
        "macro_oracle_normalized_gain_at_k": float(np.mean([
            s["oracle_normalized_gain_at_k"] for s in per_source
            if s["oracle_normalized_gain_at_k"] is not None
        ])) if any(s["oracle_normalized_gain_at_k"] is not None
                  for s in per_source) else None,
        "pool_status_counts": dict(status_counts),
        "n_rows_missing_pool_key": missing_pool_key,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "pool_key_definition": POOL_KEY_DEFINITION,
        "publication_grouping_qualified": False,
        "per_source": per_source,
    }
