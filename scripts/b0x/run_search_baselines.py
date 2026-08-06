"""B0-X search / optimization baselines.

For the measured candidate pool within each source, establish the search
headroom / measured-space ceiling using:
  - random_legal   : a random ranking of the measured candidates
  - greedy         : a distance-based greedy selection (closest edits first)
  - exact          : exact enumeration (oracle ranking by true delta) = ceiling

Singleton sources (pool size 1) are excluded from the ranking headline but
counted in the denominator report.

Metrics (per source pool, macro-averaged):
  - NDCG@k             : normalized discounted cumulative gain over ORACLE rel
  - top_decile_recall  : fraction of true top-decile recovered in pred top-decile
  - enrichment@k       : mean(true delta of pred top-k) / mean(true delta of oracle top-k)
  - normalized_regret  : (max_true - best_pred_true) / (max_true - min_true)

Usage:
    python -m scripts.b0x.run_search_baselines --dataset artifacts/b0x/effect_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from run_effect_baselines import load_dataset  # noqa: E402


def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel))
    if k <= 0:
        return 0.0
    dcg = sum(rel[i] / math.log2(i + 2) for i in range(k))
    idcg = sum(sorted(rel, reverse=True)[i] / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def rank_metrics(deltas: np.ndarray, pred_order: np.ndarray, k: int) -> dict:
    """deltas: true deltas; pred_order: indices sorted by predicted rank (best first)."""
    n = len(deltas)
    k = min(k, n)
    true_order = np.argsort(deltas)[::-1]
    rel_oracle = np.sort(deltas)[::-1]
    rel_pred = deltas[pred_order]
    ndcg = ndcg_at_k(rel_pred, k)
    # top-decile recall
    k_dec = max(1, int(math.ceil(n / 10)))
    true_top = set(true_order[:k_dec].tolist())
    pred_top = set(pred_order[:k_dec].tolist())
    tdr = len(pred_top & true_top) / k_dec
    # enrichment@k
    denom = float(np.mean(rel_oracle[:k])) if k > 0 and np.mean(rel_oracle[:k]) != 0 else None
    enrich = float(np.mean(rel_pred[:k])) / denom if denom else None
    # normalized regret
    mn, mx = float(np.min(deltas)), float(np.max(deltas))
    best_pred_true = float(deltas[pred_order[0]]) if n > 0 else None
    regret = (mx - best_pred_true) / (mx - mn) if (mx - mn) > 0 else 0.0
    return {
        "n": n,
        "ndcg_at_k": ndcg,
        "top_decile_recall": tdr,
        "enrichment_at_k": enrich,
        "normalized_regret": regret,
    }


def _mean_plain(metrics: List[dict]) -> dict:
    """Average plain metric keys over a list of per-trial/per-source dicts."""
    def _mean(key):
        vals = [m[key] for m in metrics if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "ndcg_at_k": _mean("ndcg_at_k"),
        "top_decile_recall": _mean("top_decile_recall"),
        "enrichment_at_k": _mean("enrichment_at_k"),
        "normalized_regret": _mean("normalized_regret"),
    }


def _avg_metrics(metrics: List[dict]) -> dict:
    def _mean(key):
        vals = [m[key] for m in metrics if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_sources": len(metrics),
        "n_records": int(sum(m["n"] for m in metrics)),
        "macro_ndcg_at_k": _mean("ndcg_at_k"),
        "macro_top_decile_recall": _mean("top_decile_recall"),
        "macro_enrichment_at_k": _mean("enrichment_at_k"),
        "macro_normalized_regret": _mean("normalized_regret"),
    }


def greedy_rank(records: List[dict]) -> np.ndarray:
    """Greedy distance heuristic: order candidates by ascending edit distance to
    the source (closest edits first), breaking ties by candidate presentation."""
    order = np.argsort([r["edit_count"] for r in records])
    return order.astype(int)


def run_search(rows: List[dict], k: int = 10, random_trials: int = 3) -> dict:
    pools: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        pools[r["source_id"]].append(r)

    per_bl: Dict[str, list] = defaultdict(list)
    singleton_counts = 0
    for src, recs in pools.items():
        if len(recs) < 2:
            singleton_counts += 1
            continue
        deltas = np.array([r["delta"] for r in recs], dtype=float)
        n = len(recs)

        # random legal
        rng = np.random.default_rng(0)
        rmets = []
        for _ in range(random_trials):
            perm = rng.permutation(n)
            rmets.append(rank_metrics(deltas, perm, k))
        # average random metrics over trials, keeping plain (non-macro) keys
        avg_rand = _mean_plain(rmets)
        avg_rand["n"] = n
        per_bl["random_legal"].append(avg_rand)

        # greedy
        grec = greedy_rank(recs)
        per_bl["greedy"].append(rank_metrics(deltas, grec, k))

        # exact (oracle ceiling)
        exact = np.argsort(deltas)[::-1].astype(int)
        per_bl["exact_enumeration"].append(rank_metrics(deltas, exact, k))

    results = {bl: _avg_metrics(ms) for bl, ms in per_bl.items()}
    return {
        "k": k,
        "random_trials": random_trials,
        "n_sources_total": len(pools),
        "n_singleton_sources": singleton_counts,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/b0x"))
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    out = {"benchmarks": {}}
    for benchmark in ("5U-A1", "3U-A1"):
        bench_rows = [r for r in rows if r["benchmark"] == benchmark]
        out["benchmarks"][benchmark] = run_search(bench_rows, k=args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "search_baseline_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())