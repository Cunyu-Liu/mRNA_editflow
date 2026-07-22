"""P2-03: Aggregate leakage-free headline results across baselines.

Reads eval_summary.json from each baseline's seed directories and produces:
  1. A headline JSON (benchmark/dev/leakage_free_headline_preliminary/headline.json)
  2. A markdown summary (docs/p2_03_leakage_free_headline_preliminary.md)

Primary endpoint (pre-registered): delta_oracle_te_vs_source (predicted TE
improvement over source, internal proxy).
Secondary endpoints: oracle_ensemble_te, edit_distance, mfe_proxy.

Note: This is a PRELIMINARY aggregation (1 training seed × 10 decoder seeds).
The full P2-03 spec requires ≥3 training seeds × 10 decoder seeds with
family-cluster bootstrap CI. Phase C will add the remaining training seeds.

Usage:
    python -m scripts.aggregate_p2_03_headline \\
        --root benchmark/dev/leakage_free_headline_preliminary \\
        --out-json benchmark/dev/leakage_free_headline_preliminary/headline.json \\
        --out-md docs/p2_03_leakage_free_headline_preliminary.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Pre-registered endpoints.
PRIMARY_ENDPOINT = "delta_oracle_te_vs_source"
SECONDARY_ENDPOINTS = ["oracle_ensemble_te", "edit_distance", "mfe_proxy"]
# Qualifier per project constraint.
REWARD_QUALIFIER = "predicted_te_internal_proxy"


def load_eval_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Load an eval_summary.json, returning None on error."""
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
        return None


def extract_metric(eval_summary: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Extract a metric value from an eval_summary.json.

    Looks in bootstrap_ci[metric_name].mean first, then metrics[metric_name].
    Returns None if not found.
    """
    bci = eval_summary.get("bootstrap_ci", {})
    if metric_name in bci:
        return float(bci[metric_name].get("mean"))
    metrics = eval_summary.get("metrics", {})
    if metric_name in metrics:
        val = metrics[metric_name]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and "mean" in val:
            return float(val["mean"])
    return None


def load_progress_deltas(baseline_dir: Path) -> Dict[int, float]:
    """Load delta_oracle_te_vs_source from multiseed_progress.jsonl.

    Returns a mapping {decoder_seed: delta_value}.
    """
    progress_path = baseline_dir / "multiseed_progress.jsonl"
    if not progress_path.exists():
        return {}
    deltas: Dict[int, float] = {}
    with progress_path.open("r") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                if "delta_oracle_te_vs_source" in d and "decoder_seed" in d:
                    deltas[int(d["decoder_seed"])] = float(d["delta_oracle_te_vs_source"])
            except (json.JSONDecodeError, ValueError):
                continue
    return deltas


def extract_delta(eval_summary: Dict[str, Any]) -> Optional[float]:
    """Extract delta_oracle_te_vs_source from eval_summary.json.

    This is the primary endpoint: predicted TE improvement over source.
    Falls back to None if not present (caller should use progress JSONL).
    """
    # Direct field in bootstrap_ci.
    bci = eval_summary.get("bootstrap_ci", {})
    if PRIMARY_ENDPOINT in bci:
        return float(bci[PRIMARY_ENDPOINT].get("mean"))
    # Try metrics.
    metrics = eval_summary.get("metrics", {})
    if PRIMARY_ENDPOINT in metrics:
        val = metrics[PRIMARY_ENDPOINT]
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and "mean" in val:
            return float(val["mean"])
    return None


def aggregate_baseline(
    baseline_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Aggregate results for a single baseline across all decoder seeds.

    Returns a dict with:
      - baseline: str
      - n_seeds: int
      - primary: {mean, std, ci_low, ci_high, values}
      - secondary: {metric: {mean, std, ci_low, ci_high, values}}
      - per_seed: [{seed, primary, secondary}]
    """
    seed_dirs = sorted([d for d in baseline_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    if not seed_dirs:
        return None

    # Load delta_oracle_te_vs_source from multiseed_progress.jsonl as fallback.
    progress_deltas = load_progress_deltas(baseline_dir)

    per_seed: List[Dict[str, Any]] = []
    primary_values: List[float] = []
    secondary_values: Dict[str, List[float]] = {m: [] for m in SECONDARY_ENDPOINTS}

    for sd in seed_dirs:
        eval_path = sd / "eval_summary.json"
        ev = load_eval_summary(eval_path)
        if ev is None:
            continue
        seed = int(sd.name.replace("seed_", ""))
        primary = extract_delta(ev)
        if primary is None and seed in progress_deltas:
            primary = progress_deltas[seed]
        if primary is not None:
            primary_values.append(primary)
        secondary: Dict[str, Optional[float]] = {}
        for m in SECONDARY_ENDPOINTS:
            v = extract_metric(ev, m)
            secondary[m] = v
            if v is not None:
                secondary_values[m].append(v)
        per_seed.append({
            "seed": seed,
            "primary": primary,
            "secondary": secondary,
        })

    if not primary_values:
        return None

    def bootstrap_ci(values: List[float], n_boot: int = 10000) -> Tuple[float, float, float]:
        arr = np.array(values)
        mean = float(arr.mean())
        if len(arr) < 2:
            return mean, mean, mean
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        return mean, ci_low, ci_high

    primary_mean, primary_ci_low, primary_ci_high = bootstrap_ci(primary_values)
    secondary_agg: Dict[str, Dict[str, float]] = {}
    for m, vals in secondary_values.items():
        if vals:
            m_mean, m_low, m_high = bootstrap_ci(vals)
            secondary_agg[m] = {
                "mean": m_mean,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": m_low,
                "ci_high": m_high,
                "n": len(vals),
            }

    return {
        "baseline": baseline_dir.name,
        "n_seeds": len(primary_values),
        "primary": {
            "endpoint": PRIMARY_ENDPOINT,
            "qualifier": REWARD_QUALIFIER,
            "mean": primary_mean,
            "std": float(np.std(primary_values, ddof=1)) if len(primary_values) > 1 else 0.0,
            "ci_low": primary_ci_low,
            "ci_high": primary_ci_high,
            "ci_level": 0.95,
            "values": primary_values,
        },
        "secondary": secondary_agg,
        "per_seed": per_seed,
    }


def pairwise_compare(
    baseline_a: Dict[str, Any],
    baseline_b: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Paired comparison of two baselines on the primary endpoint.

    Uses the per-seed primary values. If seeds don't match, falls back to
    unpaired bootstrap.
    """
    a_vals = baseline_a["primary"]["values"]
    b_vals = baseline_b["primary"]["values"]
    if not a_vals or not b_vals:
        return None

    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    diff_mean = float(b_arr.mean() - a_arr.mean())

    # Paired if same length.
    if len(a_arr) == len(b_arr):
        diffs = b_arr - a_arr
        # Paired bootstrap.
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot_diffs = np.array([
            rng.choice(diffs, size=len(diffs), replace=True).mean()
            for _ in range(n_boot)
        ])
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        # p-value (two-sided): fraction of bootstrap diffs crossing 0.
        p_value = float(
            2 * min(
                (boot_diffs <= 0).mean(),
                (boot_diffs >= 0).mean(),
            )
        )
        test_type = "paired_bootstrap"
    else:
        # Unpaired bootstrap.
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot_a = np.array([
            rng.choice(a_arr, size=len(a_arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        boot_b = np.array([
            rng.choice(b_arr, size=len(b_arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        boot_diffs = boot_b - boot_a
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        p_value = float(
            2 * min(
                (boot_diffs <= 0).mean(),
                (boot_diffs >= 0).mean(),
            )
        )
        test_type = "unpaired_bootstrap"

    return {
        "baseline_a": baseline_a["baseline"],
        "baseline_b": baseline_b["baseline"],
        "diff_mean": diff_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "test_type": test_type,
        "significant": p_value < 0.05,
    }


def aggregate_all(root: Path) -> Dict[str, Any]:
    """Aggregate all baselines under root."""
    baseline_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and "_seed" in d.name
    ])
    if not baseline_dirs:
        return {"error": f"no baseline dirs found under {root}"}

    baselines: List[Dict[str, Any]] = []
    for bd in baseline_dirs:
        agg = aggregate_baseline(bd)
        if agg is not None:
            baselines.append(agg)

    if not baselines:
        return {"error": "no valid eval results found"}

    # Pairwise comparisons (all pairs).
    comparisons: List[Dict[str, Any]] = []
    for i in range(len(baselines)):
        for j in range(i + 1, len(baselines)):
            cmp = pairwise_compare(baselines[i], baselines[j])
            if cmp is not None:
                comparisons.append(cmp)

    # Rank by primary mean.
    ranked = sorted(baselines, key=lambda b: b["primary"]["mean"], reverse=True)

    return {
        "root": str(root),
        "n_baselines": len(baselines),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_qualifier": REWARD_QUALIFIER,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "n_training_seeds": 1,  # preliminary; Phase C will add 2 more
        "n_decoder_seeds": baselines[0]["n_seeds"] if baselines else 0,
        "baselines": baselines,
        "ranked_by_primary": [b["baseline"] for b in ranked],
        "pairwise_comparisons": comparisons,
        "note": (
            "PRELIMINARY (1 training seed × 10 decoder seeds). "
            "Full P2-03 requires ≥3 training seeds with family-cluster "
            "bootstrap CI. Phase C will add remaining training seeds."
        ),
    }


def write_markdown(agg: Dict[str, Any], out_path: Path) -> None:
    """Write a markdown summary of the aggregation."""
    lines: List[str] = []
    lines.append("# P2-03: Leakage-Free Headline (Preliminary)")
    lines.append("")
    lines.append(f"**Status**: PRELIMINARY — {agg.get('n_training_seeds', 1)} training seed × "
                 f"{agg.get('n_decoder_seeds', 0)} decoder seeds.")
    lines.append(f"**Primary endpoint**: `{agg['primary_endpoint']}` "
                 f"(qualifier: `{agg['primary_qualifier']}`)")
    lines.append(f"**Secondary endpoints**: {', '.join(f'`{m}`' for m in agg['secondary_endpoints'])}")
    lines.append("")
    lines.append("## Ranking (by primary endpoint)")
    lines.append("")
    lines.append("| Rank | Baseline | Primary mean | 95% CI | n seeds |")
    lines.append("|-----:|----------|-------------:|--------|--------:|")
    for rank, b in enumerate(
        sorted(agg["baselines"], key=lambda x: x["primary"]["mean"], reverse=True),
        start=1,
    ):
        p = b["primary"]
        lines.append(
            f"| {rank} | {b['baseline']} | {p['mean']:.6f} | "
            f"[{p['ci_low']:.6f}, {p['ci_high']:.6f}] | {b['n_seeds']} |"
        )
    lines.append("")
    lines.append("## Secondary endpoints")
    lines.append("")
    for m in agg["secondary_endpoints"]:
        lines.append(f"### `{m}`")
        lines.append("")
        lines.append("| Baseline | Mean | 95% CI | n |")
        lines.append("|----------|-----:|--------|---:|")
        for b in sorted(agg["baselines"], key=lambda x: x["primary"]["mean"], reverse=True):
            if m in b["secondary"]:
                s = b["secondary"][m]
                lines.append(
                    f"| {b['baseline']} | {s['mean']:.4f} | "
                    f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}] | {s['n']} |"
                )
        lines.append("")
    lines.append("## Pairwise comparisons (primary endpoint)")
    lines.append("")
    lines.append("| Baseline A | Baseline B | Δ (B−A) | 95% CI | p-value | Sig.? |")
    lines.append("|------------|------------|--------:|--------|--------:|:-----:|")
    for c in agg.get("pairwise_comparisons", []):
        sig = "✓" if c["significant"] else ""
        lines.append(
            f"| {c['baseline_a']} | {c['baseline_b']} | "
            f"{c['diff_mean']:.6f} | "
            f"[{c['ci_low']:.6f}, {c['ci_high']:.6f}] | "
            f"{c['p_value']:.4f} | {sig} |"
        )
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(agg.get("note", ""))
    lines.append("")
    lines.append("**Constraint compliance**:")
    lines.append("- All claims use `predicted_te_internal_proxy` qualifier.")
    lines.append("- Split contract enforced (--train-idx/--val-idx/--test-idx).")
    lines.append("- Preliminary: 1 training seed only. Phase C will add seeds 1, 2.")

    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate P2-03 headline results.")
    parser.add_argument(
        "--root",
        default="benchmark/dev/leakage_free_headline_preliminary",
        help="Root directory containing baseline_seedN/ subdirectories.",
    )
    parser.add_argument(
        "--out-json",
        default="benchmark/dev/leakage_free_headline_preliminary/headline.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        default="docs/p2_03_leakage_free_headline_preliminary.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        return 1

    agg = aggregate_all(root)
    if "error" in agg:
        print(f"ERROR: {agg['error']}", file=sys.stderr)
        return 1

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(agg, fh, indent=2, sort_keys=True)
    print(f"Wrote JSON: {out_json} ({out_json.stat().st_size} bytes)")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(agg, out_md)
    print(f"Wrote markdown: {out_md} ({out_md.stat().st_size} bytes)")

    # Print ranking to stdout.
    print("\n=== Ranking (by primary endpoint) ===")
    for rank, b in enumerate(
        sorted(agg["baselines"], key=lambda x: x["primary"]["mean"], reverse=True),
        start=1,
    ):
        p = b["primary"]
        print(f"  {rank}. {b['baseline']}: {p['mean']:.6f} "
              f"[{p['ci_low']:.6f}, {p['ci_high']:.6f}] (n={b['n_seeds']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
