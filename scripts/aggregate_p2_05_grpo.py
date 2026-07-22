#!/usr/bin/env python
"""P2-05 GRPO Pilot Aggregation Script.

Aggregates results from 3 policy seeds of the P2-05 GRPO pilot, computing
family-cluster bootstrap confidence intervals and determining whether GRPO
outperforms baselines on equal query budget.

Usage:
    python scripts/aggregate_p2_05_grpo.py \\
        --seed-dirs /path/to/seed0 /path/to/seed1 /path/to/seed2 \\
        --output docs/p2_05_grpo_pilot_results.json

Each seed directory should contain:
    - curves.jsonl   (per-iteration training metrics)
    - trajectories.jsonl  (per-iteration trajectory data)
    - run_metadata.json   (run metadata)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_curves(curves_path: Path) -> List[Dict[str, Any]]:
    """Load curves.jsonl and return list of per-iteration metric dicts."""
    curves = []
    with curves_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                curves.append(json.loads(line))
    return curves


def load_trajectories(traj_path: Path) -> List[Dict[str, Any]]:
    """Load trajectories.jsonl."""
    trajs = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(json.loads(line))
    return trajs


def extract_metric_series(curves: List[Dict[str, Any]], metric: str) -> List[float]:
    """Extract a single metric as a time series from curves."""
    return [c.get(metric, float("nan")) for c in curves]


def compute_final_metric(curves: List[Dict[str, Any]], metric: str) -> float:
    """Get the last-iteration value of a metric."""
    if not curves:
        return float("nan")
    return curves[-1].get(metric, float("nan"))


def compute_best_metric(curves: List[Dict[str, Any]], metric: str, maximize: bool = True) -> float:
    """Get the best (max or min) value of a metric across all iterations."""
    vals = extract_metric_series(curves, metric)
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return max(vals) if maximize else min(vals)


def family_cluster_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of ``values``.

    This is a family-cluster bootstrap: each value is treated as one
    independent cluster (policy seed). Resampling is done at the cluster
    level with replacement.
    """
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_bootstrap_pvalue(
    treatment: List[float],
    baseline: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """One-sided paired bootstrap p-value: P(baseline >= treatment | H0).

    Tests whether treatment mean is significantly greater than baseline mean.
    """
    if len(treatment) != len(baseline) or len(treatment) == 0:
        return 1.0
    arr_t = np.array(treatment, dtype=float)
    arr_b = np.array(baseline, dtype=float)
    observed_diff = arr_t.mean() - arr_b.mean()
    # Under null, swap treatment/baseline within each pair with 50% prob.
    rng = np.random.RandomState(seed)
    n = len(arr_t)
    count = 0
    for _ in range(n_bootstrap):
        swap = rng.random(n) < 0.5
        t_null = np.where(swap, arr_b, arr_t)
        b_null = np.where(swap, arr_t, arr_b)
        diff_null = t_null.mean() - b_null.mean()
        if diff_null >= observed_diff:
            count += 1
    return count / n_bootstrap


def aggregate_seed_results(seed_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Aggregate curves and trajectory data across multiple policy seeds.

    Returns a dict with per-seed metrics and cross-seed statistics.
    """
    per_seed: List[Dict[str, Any]] = []
    for sd in seed_dirs:
        curves_path = sd / "curves.jsonl"
        traj_path = sd / "trajectories.jsonl"
        meta_path = sd / "run_metadata.json"

        seed_info: Dict[str, Any] = {"seed_dir": str(sd)}

        if curves_path.exists():
            curves = load_curves(curves_path)
            seed_info["n_iters"] = len(curves)
            seed_info["final_mean_return"] = compute_final_metric(curves, "mean_return")
            seed_info["best_mean_return"] = compute_best_metric(curves, "mean_return", maximize=True)
            seed_info["final_loss"] = compute_final_metric(curves, "loss")
            seed_info["final_mean_kl"] = compute_final_metric(curves, "mean_kl")
            seed_info["final_mean_entropy"] = compute_final_metric(curves, "mean_entropy")
            seed_info["final_mean_advantage"] = compute_final_metric(curves, "mean_advantage")
            seed_info["final_return_std_mean"] = compute_final_metric(curves, "return_std_mean")
            # Improvement: final - initial mean_return
            if curves:
                initial = curves[0].get("mean_return", float("nan"))
                final = curves[-1].get("mean_return", float("nan"))
                seed_info["improvement"] = final - initial
                seed_info["initial_mean_return"] = initial
            else:
                seed_info["improvement"] = float("nan")
                seed_info["initial_mean_return"] = float("nan")
        else:
            seed_info["n_iters"] = 0
            seed_info["final_mean_return"] = float("nan")
            seed_info["best_mean_return"] = float("nan")

        if meta_path.exists():
            with meta_path.open() as f:
                seed_info["metadata"] = json.load(f)
        else:
            seed_info["metadata"] = None

        # Trajectory-level: compute mean reward per seed
        if traj_path.exists():
            trajs = load_trajectories(traj_path)
            all_rewards: List[float] = []
            for entry in trajs:
                for t in entry.get("transitions", []):
                    r = t.get("reward")
                    if r is not None and not math.isnan(r):
                        all_rewards.append(float(r))
            seed_info["traj_mean_reward"] = float(np.mean(all_rewards)) if all_rewards else float("nan")
            seed_info["traj_n_transitions"] = len(all_rewards)
        else:
            seed_info["traj_mean_reward"] = float("nan")
            seed_info["traj_n_transitions"] = 0

        per_seed.append(seed_info)

    # Cross-seed statistics on final_mean_return
    final_returns = [s["final_mean_return"] for s in per_seed if not math.isnan(s["final_mean_return"])]
    best_returns = [s["best_mean_return"] for s in per_seed if not math.isnan(s["best_mean_return"])]
    improvements = [s["improvement"] for s in per_seed if not math.isnan(s["improvement"])]

    ci_final = family_cluster_bootstrap_ci(final_returns)
    ci_best = family_cluster_bootstrap_ci(best_returns)
    ci_improvement = family_cluster_bootstrap_ci(improvements)

    result = {
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "cross_seed": {
            "final_mean_return": {
                "mean": float(np.mean(final_returns)) if final_returns else float("nan"),
                "std": float(np.std(final_returns, ddof=1)) if len(final_returns) > 1 else 0.0,
                "ci_95": {"low": ci_final[0], "high": ci_final[1]},
                "n": len(final_returns),
            },
            "best_mean_return": {
                "mean": float(np.mean(best_returns)) if best_returns else float("nan"),
                "std": float(np.std(best_returns, ddof=1)) if len(best_returns) > 1 else 0.0,
                "ci_95": {"low": ci_best[0], "high": ci_best[1]},
                "n": len(best_returns),
            },
            "improvement": {
                "mean": float(np.mean(improvements)) if improvements else float("nan"),
                "std": float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0,
                "ci_95": {"low": ci_improvement[0], "high": ci_improvement[1]},
                "n": len(improvements),
            },
        },
    }
    return result


def determine_verdict(
    aggregated: Dict[str, Any],
    baseline_final_return: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Determine whether GRPO outperforms baseline or is degraded.

    Verdict:
    - "improves": GRPO final mean_return CI lower bound > baseline
    - "degraded": GRPO final mean_return CI upper bound < baseline
    - "inconclusive": CI overlaps baseline
    """
    cs = aggregated.get("cross_seed", {}).get("final_mean_return", {})
    mean = cs.get("mean", float("nan"))
    ci = cs.get("ci_95", {"low": float("nan"), "high": float("nan")})

    if baseline_final_return is None or math.isnan(baseline_final_return):
        return {
            "verdict": "no_baseline",
            "reason": "No baseline provided for comparison",
            "grpo_mean": mean,
            "grpo_ci": ci,
            "baseline": None,
        }

    if math.isnan(mean) or math.isnan(ci["low"]) or math.isnan(ci["high"]):
        return {
            "verdict": "inconclusive",
            "reason": "GRPO metrics are NaN",
            "grpo_mean": mean,
            "grpo_ci": ci,
            "baseline": baseline_final_return,
        }

    if ci["low"] > baseline_final_return:
        verdict = "improves"
        reason = "GRPO CI lower bound (%.6f) > baseline (%.6f)" % (ci["low"], baseline_final_return)
    elif ci["high"] < baseline_final_return:
        verdict = "degraded"
        reason = "GRPO CI upper bound (%.6f) < baseline (%.6f)" % (ci["high"], baseline_final_return)
    else:
        verdict = "inconclusive"
        reason = "GRPO CI [%.6f, %.6f] overlaps baseline (%.6f)" % (ci["low"], ci["high"], baseline_final_return)

    return {
        "verdict": verdict,
        "reason": reason,
        "grpo_mean": mean,
        "grpo_ci": ci,
        "baseline": baseline_final_return,
        "alpha": alpha,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate P2-05 GRPO pilot results across policy seeds"
    )
    parser.add_argument("--seed-dirs", nargs="+", required=True,
        help="Directories containing curves.jsonl, trajectories.jsonl, run_metadata.json")
    parser.add_argument("--baseline-final-return", type=float, default=None,
        help="Baseline (e.g. EMA-REINFORCE) final mean_return for comparison")
    parser.add_argument("--output", required=True,
        help="Output JSON path for aggregated results")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations for CI")
    args = parser.parse_args(argv)

    seed_dirs = [Path(d) for d in args.seed_dirs]
    for sd in seed_dirs:
        if not sd.exists():
            print("WARNING: seed directory does not exist: %s" % sd, file=sys.stderr)

    aggregated = aggregate_seed_results(seed_dirs)

    verdict = determine_verdict(
        aggregated,
        baseline_final_return=args.baseline_final_return,
    )

    output = {
        "schema_version": "1.0",
        "reward_qualifier": "predicted_te_internal_proxy",
        "n_policy_seeds": len(seed_dirs),
        "aggregated": aggregated,
        "verdict": verdict,
        "n_bootstrap": args.n_bootstrap,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print("WROTE:", out_path)
    print("n_seeds:", aggregated["n_seeds"])
    cs = aggregated.get("cross_seed", {}).get("final_mean_return", {})
    print("final_mean_return: mean=%.6f CI=[%.6f, %.6f]" % (
        cs.get("mean", float("nan")),
        cs.get("ci_95", {}).get("low", float("nan")),
        cs.get("ci_95", {}).get("high", float("nan")),
    ))
    print("verdict:", verdict["verdict"])
    print("reason:", verdict["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
