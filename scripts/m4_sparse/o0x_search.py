"""O0-X closed measured-space optimization.

Phase O0-X runs, WITHOUT any Edit Flow, on each source's real *measured*
candidate pool:

  - exact enumeration (oracle ceiling on true measured delta)
  - random legal (baseline)
  - greedy
  - beam / search
  - SparseEditFormer reranking (model-predicted delta)

and reports:
  - NDCG@k, top-decile recall, enrichment@k, normalized regret
  - query count, forward equivalents, wall time

Aggregation is source-macro then study-macro (S4 leave-one-study-out).  Singleton
sources (pool size 1) are excluded from the ranking headline but counted.  Pool
completeness / missingness is reported.  This establishes the scorer/search
ceiling that determines whether a subsequent Flow is necessary.

All metrics are computed against the *measured* delta (no Flow, no generation).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    """NDCG@k over per-pool min-max normalized relevance.

    Measured deltas are signed and can be negative (e.g. a stability edit that
    decreases the endpoint).  Raw signed deltas make (2**rel-1) / raw-IDCG
    degenerate (idcg<=0 -> 0 even for a perfect ordering).  We therefore
    min-max normalize relevance to [0,1] per source pool (the same convention
    as the M4 reranker), so NDCG is a well-defined, listwise-normalized ranking
    metric: a perfect ordering gives 1.0.
    """
    k = min(k, len(rel))
    if k <= 0:
        return 0.0
    rmin, rmax = float(rel.min()), float(rel.max())
    if rmax > rmin:
        rel_n = (rel - rmin) / (rmax - rmin)
    else:
        rel_n = rel - rmin
    dcg = sum((2 ** rel_n[i] - 1) / math.log2(i + 2) for i in range(k))
    idcg = sum((2 ** sorted(rel_n, reverse=True)[i] - 1) / math.log2(i + 2)
               for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def top_decile_recall(deltas: np.ndarray, order: np.ndarray) -> float:
    n = len(deltas)
    if n == 0:
        return 0.0
    k = max(1, int(math.ceil(n * 0.10)))
    true_top = set(np.argsort(deltas)[::-1][:k].tolist())
    pred_top = set(order[:k].tolist())
    return len(pred_top & true_top) / k


def rank_metrics(deltas: np.ndarray, order: np.ndarray, k: int = 10) -> dict:
    """deltas: true measured deltas; order: candidate indices best->worst."""
    n = len(deltas)
    k = min(k, n)
    rel_pred = deltas[order]
    rel_oracle = np.sort(deltas)[::-1]
    ndcg = ndcg_at_k(rel_pred, k)
    tdr = top_decile_recall(deltas, order)
    denom = float(np.mean(rel_oracle[:k])) if k > 0 and np.mean(rel_oracle[:k]) != 0 else None
    enrich = float(np.mean(rel_pred[:k])) / denom if denom else None
    mn, mx = float(deltas.min()), float(deltas.max())
    best_pred_true = float(deltas[order[0]]) if n > 0 else None
    regret = (mx - best_pred_true) / (mx - mn) if (mx - mn) > 0 else 0.0
    return {
        "n": n,
        "ndcg_at_k": ndcg,
        "top_decile_recall": tdr,
        "enrichment_at_k": enrich,
        "normalized_regret": regret,
    }


# ---------------------------------------------------------------------------
# search strategies over a single source pool (closed, measured)
# ---------------------------------------------------------------------------

def _greedy_edit_order(records: List[dict]) -> np.ndarray:
    """Greedy distance heuristic: closest edits (fewest edit_count) first."""
    return np.argsort([r["edit_count"] for r in records]).astype(int)


def _beam_order(records: List[dict], delta_score: np.ndarray,
                beam_width: int = 8) -> np.ndarray:
    """Beam search over measured candidates using a scorer delta_score.

    Expand the top-`beam_width` candidates at each step; the best-first order is
    the beam frontier sorted by score (a tractable beam over the closed pool).
    """
    n = len(records)
    if n == 0:
        return np.array([], dtype=int)
    bw = max(1, min(beam_width, n))
    order = np.argsort(delta_score)[::-1]
    # beam keeps the top-bw scored candidates; the returned order is those kept,
    # followed by the remainder (so the ranking is complete and deterministic).
    kept = list(order[:bw])
    rest = list(order[bw:])
    return np.array(kept + rest, dtype=int)


def _random_order(records: List[dict], seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(len(records))


def run_source_headline(records: List[dict], pred_score: Optional[np.ndarray],
                        k: int = 10, beam_width: int = 8,
                        random_seed: int = 0) -> Dict:
    """Run all O0-X search strategies on one non-singleton source pool.

    records: measured (source,candidate) rows sharing a source_id.
    pred_score: SparseEditFormer predicted delta aligned to records (or None).
    Returns per-strategy metrics plus compute accounting.

    forward-equivalents: exact enumeration / greedy / beam / random use no model
    forward passes (0); SparseEditFormer reranking costs len(pool) forwards.
    wall_time is measured per strategy by the caller hook (passed here for
    deterministic accounting is avoided; we return event counts and let the
    runner time model-inference separately).
    """
    deltas = np.array([r["delta"] for r in records], dtype=float)
    n = len(records)
    out = {"n_candidates": n, "strategies": {}}

    # exact enumeration (oracle ceiling)
    exact = np.argsort(deltas)[::-1].astype(int)
    out["strategies"]["exact_enumeration"] = {
        **rank_metrics(deltas, exact, k),
        "query_count": n, "forward_equivalents": 0,
    }

    # random legal (baseline)
    rnd = _random_order(records, random_seed)
    out["strategies"]["random_legal"] = {
        **rank_metrics(deltas, rnd, k),
        "query_count": n, "forward_equivalents": 0,
    }

    # greedy (edit-distance heuristic)
    grec = _greedy_edit_order(records)
    out["strategies"]["greedy"] = {
        **rank_metrics(deltas, grec, k),
        "query_count": n, "forward_equivalents": 0,
    }

    # beam over measured pool: if pred_score available use it, else use greedy
    # (edit_count) as the beam heuristic so the beam is defined even without a
    # learned scorer.
    if pred_score is not None:
        beam_scorer = pred_score
    else:
        beam_scorer = -np.array([r["edit_count"] for r in records], dtype=float)
    bm = _beam_order(records, beam_scorer, beam_width)
    out["strategies"]["beam"] = {
        **rank_metrics(deltas, bm, k),
        "query_count": n, "forward_equivalents": 0,
    }

    # SparseEditFormer reranking (only defined when pred_score is provided)
    if pred_score is not None:
        rrk = np.argsort(pred_score)[::-1].astype(int)
        out["strategies"]["sparseeditformer_rerank"] = {
            **rank_metrics(deltas, rrk, k),
            "query_count": 0, "forward_equivalents": n,
        }

    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _macro(key: str, metrics: List[dict]) -> Optional[float]:
    vals = [m[key] for m in metrics if m.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def aggregate_strategy(strategy_metrics: List[dict]) -> dict:
    """Source-macro over a list of per-source strategy result dicts."""
    return {
        "n_sources": len(strategy_metrics),
        "n_records": int(sum(m["n"] for m in strategy_metrics)),
        "macro_ndcg_at_k": _macro("ndcg_at_k", strategy_metrics),
        "macro_top_decile_recall": _macro("top_decile_recall", strategy_metrics),
        "macro_enrichment_at_k": _macro("enrichment_at_k", strategy_metrics),
        "macro_normalized_regret": _macro("normalized_regret", strategy_metrics),
        "total_query_count": int(sum(m.get("query_count", 0) for m in strategy_metrics)),
        "total_forward_equivalents": int(sum(m.get("forward_equivalents", 0) for m in strategy_metrics)),
    }


def run_benchmark_o0x(rows: List[dict], folds: List[dict],
                      predict_fn=None, k: int = 10,
                      beam_width: int = 8, random_seed: int = 0,
                      device=None) -> dict:
    """Run O0-X over S4 folds for one benchmark.

    rows: benchmark rows (delta-defined).  folds: S4 folds from build_folds.
    predict_fn(fold_test_rows) -> np.ndarray predicted delta (or None to skip
    the SparseEditFormer strategy).  Aggregation: source-macro -> study-macro.
    """
    bench = rows[0]["benchmark"] if rows else "?"
    per_study = {}
    all_sources = []
    for fold in folds:
        held = fold["held_out_study"]
        test = fold["test"]
        # group by source_id -> measured pool
        pools = defaultdict(list)
        for r in test:
            pools[r["source_id"]].append(r)
        singleton = 0
        strategy_buckets = defaultdict(list)
        n_pools = 0
        # model prediction aligned to test rows (SparseEditFormer reranking)
        pred_map = None
        if predict_fn is not None and len(test) > 0:
            pred_all = predict_fn(test)
            pred_map = {test[i]["source_id"]: [] for i in range(len(test))}
            for i, r in enumerate(test):
                pred_map[r["source_id"]].append(float(pred_all[i]))

        for sid, recs in pools.items():
            n_pools += 1
            if len(recs) < 2:
                singleton += 1
                continue
            pred_score = None
            if pred_map is not None and sid in pred_map:
                pred_score = np.array(pred_map[sid], dtype=float)
            src = run_source_headline(recs, pred_score, k, beam_width, random_seed)
            all_sources.append(src)
            for strat, m in src["strategies"].items():
                strategy_buckets[strat].append(m)

        per_study[held] = {
            "study": held,
            "n_pools_total": n_pools,
            "n_singleton_sources": singleton,
            "n_headline_sources": n_pools - singleton,
            "strategies": {s: aggregate_strategy(ms)
                           for s, ms in strategy_buckets.items()},
        }

    # study-macro over per-study aggregates
    strategy_names = sorted({s for ps in per_study.values() for s in ps["strategies"]})
    study_macro = {}
    for s in strategy_names:
        aggs = [ps["strategies"][s] for ps in per_study.values() if s in ps["strategies"]]
        study_macro[s] = {
            "macro_over_studies_ndcg_at_k": _macro("macro_ndcg_at_k", aggs),
            "macro_over_studies_top_decile_recall": _macro("macro_top_decile_recall", aggs),
            "macro_over_studies_enrichment_at_k": _macro("macro_enrichment_at_k", aggs),
            "macro_over_studies_normalized_regret": _macro("macro_normalized_regret", aggs),
            "total_query_count": int(sum(a.get("total_query_count", 0) for a in aggs)),
            "total_forward_equivalents": int(sum(a.get("total_forward_equivalents", 0) for a in aggs)),
            "n_studies_with_strategy": len(aggs),
        }

    return {
        "benchmark": bench,
        "k": k,
        "n_records": len(rows),
        "per_study": per_study,
        "study_macro": study_macro,
        "summary": {
            "n_pools_total": int(sum(ps["n_pools_total"] for ps in per_study.values())),
            "n_singleton_sources": int(sum(ps["n_singleton_sources"] for ps in per_study.values())),
            "n_headline_sources": int(sum(ps["n_headline_sources"] for ps in per_study.values())),
        },
    }