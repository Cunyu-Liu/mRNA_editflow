"""Independent top-K paired reranker over a source is measured candidate pool.

Given a trained SparseEditFormer and a set of (source, candidate) records,
group candidates by source_id and rank each source is candidate pool by the
model-predicted delta.  Reports NDCG@10 and top-decile recall vs the MEASURED
delta (macro over non-singleton sources).  This is the measured-neighborhood
ranking task; per B0-X, only 5U-A1 has non-singleton measured sources.

NDCG relevance is min-max normalized per source pool so the (2**rel - 1) gain
term stays numerically bounded even for large measured deltas.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

import numpy as np


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
    """Group records by source_id and rank candidates by predicted delta.

    records and delta_pred must be aligned (same order).  Returns per-source
    ranking metrics macro-averaged over non-singleton sources.
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["source_id"]].append((i, r["delta"]))
    per_source = []
    for sid, items in groups.items():
        if len(items) < 2:
            continue
        order = sorted(items, key=lambda t: delta_pred[t[0]], reverse=True)
        rel = np.array([t[1] for t in order], dtype=float)
        rmin, rmax = float(rel.min()), float(rel.max())
        rel_norm = (rel - rmin) / (rmax - rmin + 1e-9) if rmax > rmin else rel - rmin
        per_source.append({
            "source_id": sid,
            "n_candidates": len(items),
            "ndcg_at_10": ndcg_at_k(rel_norm, k_ndcg),
            "top_decile_recall": top_decile_recall(rel),
        })
    if not per_source:
        return {"n_sources": 0, "n_singleton_sources": len(groups),
                "macro_ndcg_at_10": None, "macro_top_decile_recall": None,
                "per_source": []}
    return {
        "n_sources": len(per_source),
        "n_singleton_sources": len(groups) - len(per_source),
        "macro_ndcg_at_10": float(np.mean([s["ndcg_at_10"] for s in per_source])),
        "macro_top_decile_recall": float(np.mean([s["top_decile_recall"] for s in per_source])),
        "per_source": per_source,
    }
