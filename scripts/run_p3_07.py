#!/usr/bin/env python
"""P3-07: Strong-search ceiling and RL necessity gate — main driver.

Runs the full P3-07 experiment on the P3-01 benchmark with the P3-02
cross-fitted oracle ensemble as the delta oracle:

1. Load benchmark (measured + proxy tiers), rebuild the P3-02 ensemble
   deterministically (frozen config: seed 42, 5 folds, 4 architectures).
   NOTE: checkpoints/p3_delta_oracles/*.npz lack normalization stats
   (_y_mean/_y_std/_bias are not serialized), so the ensemble is refit in
   memory with the identical frozen P3-02 pipeline. This is a deterministic
   reproduction, not a re-experiment.
2. Train Stage B ranker (benchmark pairs) and DAgger ranker (train mothers).
3. Run 10 baselines x query budgets {32,128,512,2048} x edit budgets
   {1,3,5,10} on held-out test mothers.
4. Exact evaluation: exact one-edit (all sources), exact two-edit (subset),
   tiny-MDP DP (subset, budget 2). Compute regrets.
5. Algorithm semantics (greedy intensity / stochastic CTMC / finite-horizon
   optimal / exact terminal marginal / beam) on 2 test sources.
6. Cost accounting + RL necessity decision (Route A/B/C).

Usage:
    python scripts/run_p3_07.py \
        --benchmark-dir data/p3/benchmark \
        --output-json docs/p3_07_search_results.json \
        --n-test-sources 24 --n-train-sources 24

    # Smoke test (synthetic oracle, no data needed):
    python scripts/run_p3_07.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
# rl/__init__.py imports `mrna_editflow.rl.*`; the package resolves via the
# mrna_editflow symlink in the repo root's parent directory (same mechanism
# pytest uses via rootdir __init__.py prepend-import).
sys.path.insert(0, str(_REPO_ROOT.parent))
sys.path.insert(0, str(_REPO_ROOT))

from core.constants import START_CODON
from core.schema import MRNARecord
from rl.p3_06_mdp import RewardV3Config
from rl.p3_07_search import (
    BASELINE_METHODS,
    EnsembleDeltaOracle,
    LinearDeltaRanker,
    SyntheticDeltaOracle,
    best_single_edit,
    compare_algorithm_semantics,
    compute_regrets,
    exact_one_edit_optimum,
    exact_two_edit_optimum,
    run_all_baselines,
    tiny_mdp_value_iteration,
    train_dagger_ranker,
)

QUERY_BUDGETS = (32, 128, 512, 2048)
EDIT_BUDGETS = (1, 3, 5, 10)
CFG = RewardV3Config(context="protein_output_focused")

# Inert placeholder CDS/3'UTR: task_a allows ONLY 5'UTR substitutions, so the
# CDS is never touched by the action space; it exists only to satisfy the
# MRNARecord schema. Documented in p3_07_search_protocol.md.
INERT_CDS = START_CODON + "GCU" * 4 + "UAA"
INERT_THREE_UTR = "UGCU"


def source_to_record(source_id: str, five_utr: str) -> MRNARecord:
    return MRNARecord(
        transcript_id=source_id,
        five_utr=five_utr,
        cds=INERT_CDS,
        three_utr=INERT_THREE_UTR,
        metadata={"inert_cds": True, "task": "task_a_five_utr_only"},
    )


# ---------------------------------------------------------------------------
# Ensemble construction (deterministic reproduction of P3-02 fold models)
# ---------------------------------------------------------------------------

def build_ensemble_predict_fns(benchmark_dir: str, max_proxy: int = 10000, seed: int = 42):
    from collections import defaultdict
    from core.p3_02_delta_oracle import (
        CrossFitConfig, load_benchmark, batch_extract_features,
        build_oracle_ensemble,
    )

    tiers = load_benchmark(benchmark_dir, tiers=("measured", "proxy"))
    measured = tiers.get("measured", [])
    proxy = tiers.get("proxy", [])
    train_recs = [r for r in measured if r.split_role in ("train", "val")]
    rng = np.random.RandomState(seed)
    if proxy:
        idx = rng.choice(len(proxy), min(len(proxy), max_proxy), replace=False)
        train_recs = train_recs + [proxy[i] for i in idx]

    config = CrossFitConfig(n_folds=5, seed=seed, hidden_dim=64, lr=1e-3,
                            n_epochs=100, max_seq_len=100)
    feats = batch_extract_features(train_recs, config.max_seq_len)
    labels = feats["delta"]

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(train_recs):
        groups[rec.source_id].append(i)
    group_ids = list(groups.keys())
    rng2 = np.random.RandomState(seed)
    rng2.shuffle(group_ids)
    n_folds = config.n_folds
    fold_size = len(group_ids) // n_folds
    folds = []
    for k in range(n_folds):
        s = k * fold_size
        e = (k + 1) * fold_size if k < n_folds - 1 else len(group_ids)
        idxs: List[int] = []
        for gid in group_ids[s:e]:
            idxs.extend(groups[gid])
        folds.append(np.array(sorted(idxs)))

    ensemble = build_oracle_ensemble(feats, labels, folds, config)

    predict_fns = []
    for name in ensemble["model_names"]:
        fold_models = ensemble["per_model_models"][name]

        def fn(batch_feats, fms=fold_models):
            preds = [m.predict_delta(batch_feats) for m in fms.values()]
            return np.mean(preds, axis=0)

        predict_fns.append(fn)
    return predict_fns, ensemble


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def select_sources(benchmark_dir: str, n_test: int, n_train: int, seed: int = 0):
    from core.p3_02_delta_oracle import load_benchmark_tier

    measured = load_benchmark_tier(os.path.join(benchmark_dir, "measured_tier.jsonl"))
    anchors: Dict[str, Dict[str, str]] = {}
    for r in measured:
        if r.edit_count == 0 and r.edit_type == "wild_type_anchor":
            anchors[r.source_id] = {
                "seq": r.source_sequence,
                "split": r.split_role,
                "family": r.family_cluster_id,
            }
    test_ids = sorted(s for s, v in anchors.items() if v["split"] == "test")
    train_ids = sorted(s for s, v in anchors.items() if v["split"] == "train")
    rng = np.random.RandomState(seed)
    rng.shuffle(test_ids)
    rng.shuffle(train_ids)
    test_srcs = [source_to_record(s, anchors[s]["seq"]) for s in test_ids[:n_test]]
    train_srcs = [source_to_record(s, anchors[s]["seq"]) for s in train_ids[:n_train]]
    return test_srcs, train_srcs, {s: anchors[s] for s in test_ids[:n_test]}


def load_ranker_training_pairs(benchmark_dir: str, max_pairs: int = 20000, seed: int = 42):
    from core.p3_02_delta_oracle import load_benchmark

    tiers = load_benchmark(benchmark_dir, tiers=("measured", "proxy"))
    recs = [r for r in tiers.get("measured", []) if r.split_role in ("train", "val")]
    proxy = [r for r in tiers.get("proxy", []) if r.split_role in ("train", "val")]
    rng = np.random.RandomState(seed)
    if proxy:
        idx = rng.choice(len(proxy), min(len(proxy), max_pairs), replace=False)
        recs = recs + [proxy[i] for i in idx]
    srcs = [r.source_sequence for r in recs]
    cands = [r.candidate_sequence for r in recs]
    tgts = [r.delta for r in recs]
    return srcs, cands, tgts


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke(output_json: str):
    print("[SMOKE] synthetic oracle")
    src = source_to_record("smoke_src", "ACGUACGUAC")
    ranker = LinearDeltaRanker()
    oracle = SyntheticDeltaOracle(seed=0)
    # ranker training pairs from synthetic oracle
    from rl.p3_07_search import legal_actions
    from rl.p3_06_mdp import apply_edit_action
    srcs, cands, tgts = [src.five_utr], [src.five_utr], [0.0]
    for a in legal_actions(src):
        if a.is_stop():
            continue
        child = apply_edit_action(src, a)
        m, u = oracle.score(src, child, purpose="eval")
        srcs.append(src.five_utr); cands.append(child.five_utr); tgts.append(m - u)
    ranker.fit(srcs, cands, tgts)

    results = run_all_baselines(
        src, lambda: SyntheticDeltaOracle(seed=0),
        query_budget=128, edit_budget=3, seed=0,
        stage_b_ranker=ranker, dagger_ranker=ranker,
    )
    dp = tiny_mdp_value_iteration(src, SyntheticDeltaOracle(seed=0), edit_budget=2)
    regrets = compute_regrets(results, dp["optimal_value"])
    sem = compare_algorithm_semantics(src, SyntheticDeltaOracle(seed=0), edit_budget=2)
    out = {
        "smoke": True,
        "n_baselines": len(results),
        "results": [r.to_dict() for r in results],
        "dp_optimal_value": dp["optimal_value"],
        "dp_n_states": dp["n_states"],
        "regrets": regrets,
        "semantics_expected_returns": sem["expected_returns"],
        "semantics_constraint_validity": sem["constraint_validity"],
    }
    _write_json(output_json, out)
    print(f"[SMOKE] wrote {output_json}; baselines={len(results)}, "
          f"dp_states={dp['n_states']}, constraint_validity={sem['constraint_validity']}")


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def run_real(args):
    t_start = time.time()
    print("=" * 70)
    print("P3-07: strong-search ceiling and RL necessity gate (real oracle)")
    print("=" * 70)

    print("\n[1] Building P3-02 ensemble (deterministic refit, frozen config)")
    t0 = time.time()
    predict_fns, ensemble = build_ensemble_predict_fns(
        args.benchmark_dir, max_proxy=args.max_proxy, seed=42
    )
    print(f"    ensemble built in {time.time() - t0:.1f}s; "
          f"architectures={ensemble['model_names']}")

    def oracle_factory():
        return EnsembleDeltaOracle(predict_fns, max_seq_len=100)

    print("\n[2] Selecting sources (test mothers for search, train mothers for DAgger)")
    test_srcs, train_srcs, test_meta = select_sources(
        args.benchmark_dir, args.n_test_sources, args.n_train_sources, seed=args.seed
    )
    print(f"    test sources: {len(test_srcs)}, train sources: {len(train_srcs)}")

    print("\n[3] Training Stage B ranker on benchmark pairs")
    t0 = time.time()
    p_srcs, p_cands, p_tgts = load_ranker_training_pairs(
        args.benchmark_dir, max_pairs=args.max_ranker_pairs, seed=42
    )
    stage_b_ranker = LinearDeltaRanker().fit(p_srcs, p_cands, p_tgts)
    print(f"    ranker trained on {len(p_tgts)} pairs in {time.time() - t0:.1f}s")

    print("\n[4] Training DAgger ranker (2 rounds on train mothers)")
    t0 = time.time()
    dagger_oracle = oracle_factory()
    dagger_out = train_dagger_ranker(
        train_srcs, dagger_oracle, n_rounds=2, edits_per_round=2,
        max_actions_per_state=32, seed=args.seed,
        training_query_budget=args.dagger_training_budget,
    )
    dagger_ranker = dagger_out["ranker"]
    print(f"    DAgger done in {time.time() - t0:.1f}s; "
          f"training_oracle_calls={dagger_out['training_oracle_calls']}, "
          f"pairs={dagger_out['n_pairs']}")

    print("\n[5] Baseline grid: 10 methods x query budgets x edit budgets")
    grid_results: List[Dict[str, Any]] = []
    for si, src in enumerate(test_srcs):
        for qb in QUERY_BUDGETS:
            for eb in EDIT_BUDGETS:
                res = run_all_baselines(
                    src, oracle_factory,
                    query_budget=qb, edit_budget=eb, seed=args.seed + si,
                    stage_b_ranker=stage_b_ranker, dagger_ranker=dagger_ranker,
                )
                for r in res:
                    d = r.to_dict()
                    d["source_index"] = si
                    grid_results.append(d)
        print(f"    source {si + 1}/{len(test_srcs)} done "
              f"({src.transcript_id}, utr_len={len(src.five_utr)})")

    print("\n[6] Exact evaluation")
    exact_one: Dict[str, Any] = {}
    for si, src in enumerate(test_srcs):
        out = exact_one_edit_optimum(src, oracle_factory())
        exact_one[src.transcript_id] = {
            "optimum_score": out["optimum_score"],
            "n_evaluated": out["n_evaluated"],
        }
    print(f"    exact one-edit on {len(exact_one)} sources")

    n_two = min(args.n_two_edit_sources, len(test_srcs))
    exact_two: Dict[str, Any] = {}
    for src in test_srcs[:n_two]:
        t0 = time.time()
        out = exact_two_edit_optimum(src, oracle_factory())
        exact_two[src.transcript_id] = {
            "optimum_score": out["optimum_score"],
            "n_evaluated": out["n_evaluated"],
            "wall_clock_sec": time.time() - t0,
        }
    print(f"    exact two-edit on {n_two} sources")

    print("\n[7] Tiny-MDP DP (budget 2) + regrets")
    n_dp = min(args.n_dp_sources, len(test_srcs))
    dp_results: Dict[str, Any] = {}
    regret_table: Dict[str, Any] = {}
    for src in test_srcs[:n_dp]:
        t0 = time.time()
        dp = tiny_mdp_value_iteration(src, oracle_factory(), edit_budget=2)
        dp_results[src.transcript_id] = {
            "optimal_value": dp["optimal_value"],
            "n_states": dp["n_states"],
            "optimal_edits": dp["optimal_edits"],
            "wall_clock_sec": time.time() - t0,
        }
        # regrets of each budget-2-compatible method at each query budget
        src_grid = [g for g in grid_results
                    if g["source_id"] == src.transcript_id and g["edit_budget"] >= 2]
        # regret uses best over edit_budget >= 2 rows per (method, qb)
        by_mq: Dict[str, float] = {}
        for g in src_grid:
            key = f"{g['method']}@qb{g['query_budget']}"
            by_mq[key] = max(by_mq.get(key, -1e18), g["best_score"])
        regret_table[src.transcript_id] = {
            k: dp["optimal_value"] - v for k, v in by_mq.items()
        }
    print(f"    DP on {n_dp} sources")

    print("\n[8] Algorithm semantics on 2 sources (budgets 1 and 2)")
    semantics: Dict[str, Any] = {}
    for src in test_srcs[:2]:
        for eb in (1, 2):
            t0 = time.time()
            sem = compare_algorithm_semantics(
                src, oracle_factory(), edit_budget=eb, beam_width=4, beta=4.0
            )
            semantics[f"{src.transcript_id}__budget{eb}"] = {
                "n_states": sem["n_states"],
                "expected_returns": sem["expected_returns"],
                "terminal_kl": sem["terminal_kl"],
                "action_kl": sem["action_kl"],
                "argmax_agreement": sem["argmax_agreement"],
                "constraint_validity": sem["constraint_validity"],
                "wall_clock_sec": time.time() - t0,
            }

    print("\n[9] Cost accounting + RL necessity decision")
    decision = make_decision(
        grid_results, exact_one, regret_table, dagger_out, test_srcs,
    )

    total_wall = time.time() - t_start
    out = {
        "phase": "P3-07",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "query_budgets": list(QUERY_BUDGETS),
            "edit_budgets": list(EDIT_BUDGETS),
            "n_test_sources": len(test_srcs),
            "n_train_sources": len(train_srcs),
            "reward": "reward_v3 protein_output_focused (LCB + edit cost)",
            "oracle": "p3_02 cross-fitted ensemble (4 architectures x 5 folds), deterministic refit",
            "baselines": list(BASELINE_METHODS),
            "seed": args.seed,
            "dagger_training_oracle_calls": dagger_out["training_oracle_calls"],
            "gpu": "none (CPU-only numpy MLP ensemble); gpu_memory_bytes=0",
        },
        "grid_results": grid_results,
        "exact_one_edit": exact_one,
        "exact_two_edit": exact_two,
        "tiny_mdp_dp": dp_results,
        "regret_table": regret_table,
        "algorithm_semantics": semantics,
        "rl_necessity_decision": decision,
        "total_wall_clock_sec": total_wall,
    }
    _write_json(args.output_json, out)
    print(f"\n[done] wrote {args.output_json} in {total_wall:.1f}s")
    print(f"DECISION: {decision['route']} — {decision['rationale']}")


# ---------------------------------------------------------------------------
# Decision logic (pre-registered thresholds, see p3_07_search_protocol.md)
# ---------------------------------------------------------------------------

def make_decision(
    grid_results: List[Dict[str, Any]],
    exact_one: Dict[str, Any],
    regret_table: Dict[str, Any],
    dagger_out: Dict[str, Any],
    test_srcs: Sequence[MRNARecord],
) -> Dict[str, Any]:
    """Route A/B/C per the frozen P3-07 decision rule.

    - Route C (No-RL): strong search (or ranker+limited search) reaches
      within 5% of the achievable improvement range at query budget <= 128
      for >= 80% of sources.
    - Route B (RL amortization): near-optimal needs budget >= 512, but
      ranker+limited search matches it at much lower amortized cost.
    - Route A (RL quality): even budget-2048 search leaves > 10% regret —
      search is insufficient within practical budgets.
    """
    # Per-source achievable range: exact one-edit optimum - source score(=0 baseline)
    # Source score is 0 by construction (delta=0, edit cost 0). Range proxy:
    # use the exact one-edit optimum as the scale of achievable improvement.
    scale = {sid: max(abs(v["optimum_score"]), 1e-6) for sid, v in exact_one.items()}

    # Aggregate best score per (method, query_budget) over sources
    by_mq: Dict[str, Dict[str, List[float]]] = {}
    for g in grid_results:
        by_mq.setdefault(g["method"], {}).setdefault(str(g["query_budget"]), []).append(
            g["best_score"] / max(scale.get(g["source_id"], 1e-6), 1e-6)
        )
    norm_score = {m: {qb: float(np.mean(v)) for qb, v in qbs.items()}
                  for m, qbs in by_mq.items()}

    # Regret stats on DP sources (budget >= 2)
    regret_stats: Dict[str, Dict[str, float]] = {}
    if regret_table:
        keys = sorted({k for rt in regret_table.values() for k in rt})
        for k in keys:
            vals = [rt[k] for rt in regret_table.values() if k in rt]
            if vals:
                regret_stats[k] = {
                    "mean": float(np.mean(vals)),
                    "max": float(np.max(vals)),
                    "frac_zero": float(np.mean([abs(v) < 1e-9 for v in vals])),
                }

    # Decision inputs
    best_search_128 = max(
        (v.get("128", -1e18) for m, v in norm_score.items()
         if m not in ("stage_b_ranker", "dagger_ranker")),
        default=-1e18,
    )
    best_search_2048 = max(
        (v.get("2048", -1e18) for m, v in norm_score.items()), default=-1e18
    )
    ranker_limited_128 = norm_score.get("dagger_ranker_plus_limited_search", {}).get("128", -1e18)

    # Reference: what fraction of the exact one-edit optimum do methods reach?
    exact_ref = 1.0  # normalized scale: exact one-edit == 1.0 by construction
    reach_128 = best_search_128 / exact_ref if exact_ref else 0.0
    reach_2048 = best_search_2048 / exact_ref if exact_ref else 0.0

    if reach_2048 < 0.90:
        route = "RL_ROUTE_A"
        rationale = (
            f"Even query-budget-2048 search reaches only {reach_2048:.2%} of the "
            "exact one-edit reference (normalized); strong search is insufficient "
            "within practical budgets, so RL has a potential quality advantage."
        )
    elif reach_128 >= 0.95 or ranker_limited_128 >= 0.95:
        # near-optimal at low budget -> no RL needed UNLESS cost asymmetry
        if ranker_limited_128 >= 0.95 and best_search_128 < 0.95:
            route = "RL_ROUTE_B"
            rationale = (
                f"DAgger ranker + limited search reaches {ranker_limited_128:.2%} "
                f"at budget 128 while unguided strong search reaches {reach_128:.2%}; "
                "a learned policy amortizes search cost. Compute break-even scale."
            )
        else:
            route = "NO_RL_ROUTE_C"
            rationale = (
                f"Strong search already reaches {reach_128:.2%} of the exact "
                "reference at query budget 128; no quality or cost gap for RL to fill."
            )
    else:
        route = "RL_ROUTE_B"
        rationale = (
            f"Search reaches {reach_2048:.2%} only at budget 2048 "
            f"(budget 128: {reach_128:.2%}); quality is achievable but expensive, "
            "so the RL story is amortized constrained optimization."
        )

    # Break-even: dagger training cost / per-cargo search savings
    dagger_calls = dagger_out.get("training_oracle_calls", 0)
    breakeven = None
    if route == "RL_ROUTE_B":
        per_cargo_search = 2048
        per_cargo_policy = 32
        if per_cargo_search > per_cargo_policy:
            breakeven = dagger_calls / (per_cargo_search - per_cargo_policy)

    return {
        "route": route,
        "rationale": rationale,
        "normalized_reach": {
            "best_search_qb128": reach_128,
            "best_search_qb2048": reach_2048,
            "dagger_plus_limited_qb128": ranker_limited_128,
        },
        "norm_score_by_method_budget": norm_score,
        "regret_stats_vs_dp": regret_stats,
        "lifecycle_cost": {
            "dagger_training_oracle_calls": dagger_calls,
            "ensemble_training_oracle_calls": 0,
            "note": "ensemble trained on benchmark labels (no oracle queries); "
                    "DAgger training queries amortized over designed cargos",
            "breakeven_designed_cargos": breakeven,
        },
        "decision_rule": {
            "route_A": "best_search_qb2048 < 0.90 of exact one-edit reference",
            "route_C": "best_search_qb128 >= 0.95 (or dagger+limited >= 0.95 with cheap search also >= 0.95)",
            "route_B": "otherwise (quality achievable but expensive)",
        },
    }


def _write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description="P3-07 strong-search ceiling driver")
    ap.add_argument("--benchmark-dir", default="data/p3/benchmark")
    ap.add_argument("--output-json", default="docs/p3_07_search_results.json")
    ap.add_argument("--n-test-sources", type=int, default=24)
    ap.add_argument("--n-train-sources", type=int, default=24)
    ap.add_argument("--n-two-edit-sources", type=int, default=6)
    ap.add_argument("--n-dp-sources", type=int, default=2)
    ap.add_argument("--max-proxy", type=int, default=10000)
    ap.add_argument("--max-ranker-pairs", type=int, default=20000)
    ap.add_argument("--dagger-training-budget", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke(args.output_json)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
