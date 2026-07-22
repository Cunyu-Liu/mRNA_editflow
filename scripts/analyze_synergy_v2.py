"""P2-01: Statistical analysis for cross-region synergy panel v2.

Reads the panel results JSON from ``run_counterfactual_panel_v2_parallel.py``
and computes:
  - syn_sum significance test (one-sample t-test, H0: syn_sum = 0)
  - Family-cluster bootstrap 95% CI on syn_sum (cluster = gene family proxy)
  - Cohen's d effect size
  - Permutation test (robustness check)
  - Region-pair decomposition (syn_5c, syn_c3, syn_53)
  - GO / borderline / NO-GO decision

Decision rules (pre-registered):
  - GO       : d > 0.5  AND p < 0.001
  - borderline: d in (0.2, 0.5) AND p < 0.05
  - NO-GO    : p > 0.05  (or d <= 0.2)

All ``improves TE/stability/expression`` claims are ``predicted/internal proxy``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Family clustering (proxy via transcript_id hash)
# ---------------------------------------------------------------------------

def family_id(transcript_id: str, n_families: int = 100) -> int:
    """Assign a family ID via SHA-256 hash of transcript_id.

    This is a PROXY for gene family clustering. In production, use actual
    gene family annotations (e.g., from Ensembl BioMart).
    """
    h = hashlib.sha256(transcript_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_families


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d_one_sample(x: List[float], mu: float = 0.0) -> float:
    """Cohen's d for one-sample test (mean - mu) / std."""
    if len(x) < 2:
        return 0.0
    m = statistics.mean(x)
    s = statistics.stdev(x)
    if s == 0:
        return 0.0
    return (m - mu) / s


def t_test_one_sample(x: List[float], mu: float = 0.0) -> Tuple[float, float]:
    """One-sample t-test (H0: mean(x) = mu). Returns (t, two-sided p)."""
    n = len(x)
    if n < 2:
        return (0.0, 1.0)
    m = statistics.mean(x)
    s = statistics.stdev(x)
    if s == 0:
        return (float("inf") if m != mu else 0.0, 0.0 if m != mu else 1.0)
    t = (m - mu) / (s / math.sqrt(n))
    if _HAS_SCIPY:
        p = 2.0 * sp_stats.t.sf(abs(t), df=n - 1)
    else:
        # Normal approximation for large n.
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return (float(t), float(p))


def family_cluster_bootstrap_ci(
    values: List[float],
    families: List[int],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> Tuple[float, float]:
    """Family-cluster bootstrap CI on the mean.

    Resamples FAMILIES (not individual samples) with replacement, then
    pools all values within the resampled families. This accounts for
    within-family correlation.

    Args:
        values: per-sample values
        families: per-sample family IDs (same length as values)
        n_bootstrap: number of bootstrap iterations
        confidence: confidence level (0.95 = 95% CI)
        seed: RNG seed

    Returns:
        (lower, upper) bounds of the CI.
    """
    rng = np.random.default_rng(seed)
    # Group values by family.
    family_to_values: Dict[int, List[float]] = {}
    for v, f in zip(values, families):
        family_to_values.setdefault(f, []).append(v)
    unique_families = list(family_to_values.keys())
    n_families = len(unique_families)
    if n_families < 2:
        return (0.0, 0.0)

    arr_values = np.array(values, dtype=np.float64)
    arr_families = np.array(families, dtype=np.int64)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Resample families with replacement.
        sampled_families = rng.choice(unique_families, size=n_families, replace=True)
        # Pool values from sampled families.
        pooled = []
        for f in sampled_families:
            pooled.extend(family_to_values[f])
        bootstrap_means[b] = float(np.mean(pooled)) if pooled else 0.0

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def permutation_test(
    values: List[float],
    mu: float = 0.0,
    n_permutations: int = 10000,
    seed: int = 1729,
) -> float:
    """One-sample permutation test: H0: mean(values) = mu.

    Under H0, the sign of (value - mu) is exchangeable. We flip signs randomly
    and compute the mean, then compare to the observed mean.

    Returns two-sided p-value.
    """
    if len(values) < 2:
        return 1.0
    arr = np.array(values, dtype=np.float64) - mu
    observed = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(arr))
        perm_mean = float(np.mean(arr * signs))
        if abs(perm_mean) >= abs(observed):
            count += 1
    return (count + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def make_decision(d: float, p: float, alpha_go: float = 0.001,
                  alpha_borderline: float = 0.05) -> str:
    """GO / borderline / NO-GO decision.

    Pre-registered rules:
      - GO       : d > 0.5  AND p < alpha_go (0.001)
      - borderline: d in (0.2, 0.5) AND p < alpha_borderline (0.05)
      - NO-GO    : p > alpha_borderline  (or d <= 0.2)
    """
    if p > alpha_borderline:
        return "NO-GO"
    if d <= 0.2:
        return "NO-GO"
    if d > 0.5 and p < alpha_go:
        return "GO"
    if 0.2 < d <= 0.5 and p < alpha_borderline:
        return "borderline"
    # Edge cases.
    if d > 0.5 and p < alpha_borderline:
        return "borderline"  # large effect but not strict alpha
    return "NO-GO"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_panel(results_path: str, output_path: str,
                  n_bootstrap: int = 10000,
                  n_permutations: int = 10000,
                  n_families: int = 100,
                  seed: int = 1729) -> Dict[str, Any]:
    """Analyze the v2 panel results and produce the finding report data."""
    with open(results_path, "r") as f:
        data = json.load(f)

    panels = data["panels"]
    n = len(panels)
    print(f"[analyze] Loaded {n} panels from {results_path}")

    # Extract per-panel synergy scores.
    syn_sums = [p["synergy_scores"]["syn_sum"] for p in panels]
    syn_means = [p["synergy_scores"]["syn_mean"] for p in panels]
    syn_bests = [p["synergy_scores"]["syn_best"] for p in panels]
    syn_vs_wts = [p["synergy_scores"]["syn_vs_wt"] for p in panels]
    syn_5c = [p["synergy_scores"]["syn_5c"] for p in panels]
    syn_c3 = [p["synergy_scores"]["syn_c3"] for p in panels]
    syn_53 = [p["synergy_scores"]["syn_53"] for p in panels]
    syn_5c3 = [p["synergy_scores"]["syn_5c3"] for p in panels]

    # Deltas per arm.
    delta_5 = [p["synergy_scores"]["delta_5utr_vs_wt"] for p in panels]
    delta_c = [p["synergy_scores"]["delta_cds_vs_wt"] for p in panels]
    delta_3 = [p["synergy_scores"]["delta_3utr_vs_wt"] for p in panels]
    delta_5c = [p["synergy_scores"]["delta_pair_5_cds_vs_wt"] for p in panels]
    delta_c3 = [p["synergy_scores"]["delta_pair_c_3_vs_wt"] for p in panels]
    delta_53 = [p["synergy_scores"]["delta_pair_5_3_vs_wt"] for p in panels]
    delta_j = [p["synergy_scores"]["delta_joint_vs_wt"] for p in panels]

    # Family IDs.
    families = [family_id(p["transcript_id"], n_families) for p in panels]

    # --- syn_sum statistics ---
    t_stat, t_pval = t_test_one_sample(syn_sums, mu=0.0)
    d_syn = cohens_d_one_sample(syn_sums, mu=0.0)
    ci_lower, ci_upper = family_cluster_bootstrap_ci(
        syn_sums, families, n_bootstrap=n_bootstrap, seed=seed
    )
    perm_p = permutation_test(syn_sums, mu=0.0, n_permutations=n_permutations, seed=seed)

    # --- Pairwise synergy statistics ---
    def _stats(x: List[float]) -> Dict[str, float]:
        t, p = t_test_one_sample(x, mu=0.0)
        cd = cohens_d_one_sample(x, mu=0.0)
        ci_l, ci_u = family_cluster_bootstrap_ci(
            x, families, n_bootstrap=n_bootstrap, seed=seed
        )
        return {
            "mean": float(statistics.mean(x)),
            "std": float(statistics.stdev(x)) if len(x) >= 2 else 0.0,
            "median": float(statistics.median(x)),
            "t_stat": float(t),
            "t_pvalue": float(p),
            "cohens_d": float(cd),
            "bootstrap_ci_lower": float(ci_l),
            "bootstrap_ci_upper": float(ci_u),
        }

    syn_5c_stats = _stats(syn_5c)
    syn_c3_stats = _stats(syn_c3)
    syn_53_stats = _stats(syn_53)
    syn_5c3_stats = _stats(syn_5c3)

    # --- Per-arm improvement ---
    def _arm_stats(x: List[float]) -> Dict[str, float]:
        return {
            "mean": float(statistics.mean(x)),
            "std": float(statistics.stdev(x)) if len(x) >= 2 else 0.0,
            "median": float(statistics.median(x)),
        }

    arm_improvement = {
        "single_5utr": _arm_stats(delta_5),
        "single_cds": _arm_stats(delta_c),
        "single_3utr": _arm_stats(delta_3),
        "pair_5_cds": _arm_stats(delta_5c),
        "pair_c_3": _arm_stats(delta_c3),
        "pair_5_3": _arm_stats(delta_53),
        "joint": _arm_stats(delta_j),
    }

    # --- Decision ---
    decision = make_decision(d_syn, t_pval)
    print(f"[analyze] syn_sum: mean={statistics.mean(syn_sums):+.6f}, "
          f"d={d_syn:+.4f}, t={t_stat:+.4f}, p={t_pval:.6f}, "
          f"CI=[{ci_lower:+.6f}, {ci_upper:+.6f}], decision={decision}")
    print(f"[analyze] permutation p-value: {perm_p:.6f}")
    print(f"[analyze] Pairwise: syn_5c={syn_5c_stats['mean']:+.6f} (d={syn_5c_stats['cohens_d']:+.4f}), "
          f"syn_c3={syn_c3_stats['mean']:+.6f} (d={syn_c3_stats['cohens_d']:+.4f}), "
          f"syn_53={syn_53_stats['mean']:+.6f} (d={syn_53_stats['cohens_d']:+.4f})")

    result = {
        "n_wild_types": n,
        "n_families": n_families,
        "syn_sum_stats": {
            "mean": float(statistics.mean(syn_sums)),
            "std": float(statistics.stdev(syn_sums)) if len(syn_sums) >= 2 else 0.0,
            "median": float(statistics.median(syn_sums)),
            "t_stat": float(t_stat),
            "t_pvalue": float(t_pval),
            "cohens_d": float(d_syn),
            "bootstrap_ci_lower": float(ci_lower),
            "bootstrap_ci_upper": float(ci_upper),
            "permutation_pvalue": float(perm_p),
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_permutations,
        },
        "syn_mean_stats": _stats(syn_means),
        "syn_best_stats": _stats(syn_bests),
        "syn_vs_wt_stats": _stats(syn_vs_wts),
        "pairwise_synergy": {
            "syn_5c_5utr_x_cds": syn_5c_stats,
            "syn_c3_cds_x_3utr": syn_c3_stats,
            "syn_53_5utr_x_3utr": syn_53_stats,
            "syn_5c3_triple": syn_5c3_stats,
        },
        "arm_improvement": arm_improvement,
        "decision": {
            "verdict": decision,
            "rules": {
                "GO": "d > 0.5 AND p < 0.001",
                "borderline": "d in (0.2, 0.5) AND p < 0.05",
                "NO-GO": "p > 0.05 OR d <= 0.2",
            },
            "d_observed": float(d_syn),
            "p_observed": float(t_pval),
            "alpha_go": 0.001,
            "alpha_borderline": 0.05,
        },
        "oracle_version": "multi_region_v2_p2_01",
        "config": data.get("config", {}),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[analyze] Wrote {output_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2-01 statistical analysis for cross-region synergy panel v2"
    )
    parser.add_argument("--results", required=True,
                        help="Path to panel results JSON")
    parser.add_argument("--output", required=True,
                        help="Path to analysis output JSON")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--n-families", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    analyze_panel(
        results_path=args.results,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        n_families=args.n_families,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
