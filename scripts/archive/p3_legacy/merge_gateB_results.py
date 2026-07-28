#!/usr/bin/env python
"""Merge Gate B results from multiple GPUs and evaluate against criteria.

Usage:
    python scripts/merge_gateB_results.py \
        --gpu6-json docs/p3_08_grpo_results_gateB_gpu6.json \
        --gpu7-json docs/p3_08_grpo_results_gateB_gpu7.json \
        --output-json docs/p3_08_grpo_results_gateB.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_p3_08 import evaluate_gate_a


def merge_results(gpu6: Dict[str, Any], gpu7: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two GPU results into a single Gate B result."""
    all_results = gpu6.get("results", []) + gpu7.get("results", [])
    all_failed = gpu6.get("failed_seeds", []) + gpu7.get("failed_seeds", [])

    # Use config from gpu6 (should be identical)
    config = gpu6.get("config", {})

    # Merge train logs for health report
    total_wall = gpu6.get("total_wall_clock_sec", 0) + gpu7.get("total_wall_clock_sec", 0)

    merged = {
        "gate": "B",
        "config": config,
        "n_seeds": len(all_results),
        "n_seeds_completed": len(all_results),
        "n_seeds_failed": len(all_failed),
        "verdict": "PENDING",  # Will be set after evaluation
        "criteria": {},
        "results": all_results,
        "failed_seeds": all_failed,
        "total_wall_clock_sec": total_wall,
        "gpu_breakdown": {
            "gpu6": {
                "seeds": [r["seed"] for r in gpu6.get("results", [])],
                "n_completed": len(gpu6.get("results", [])),
                "n_failed": len(gpu6.get("failed_seeds", [])),
                "wall_clock_sec": gpu6.get("total_wall_clock_sec", 0),
            },
            "gpu7": {
                "seeds": [r["seed"] for r in gpu7.get("results", [])],
                "n_completed": len(gpu7.get("results", [])),
                "n_failed": len(gpu7.get("failed_seeds", [])),
                "wall_clock_sec": gpu7.get("total_wall_clock_sec", 0),
            },
        },
    }
    return merged


def evaluate_gate_b(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Gate B criteria (same as Gate A but for 10 seeds)."""
    results = merged["results"]
    criteria = evaluate_gate_a(results)
    merged["criteria"] = criteria
    merged["verdict"] = "PASS" if criteria.get("all_pass", False) else "FAIL"
    return merged


def generate_horizontal_comparison(merged: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate horizontal comparison table across all seeds."""
    rows = []
    for r in merged["results"]:
        fv = r.get("final_validation", {})
        ws = r.get("warm_start_validation", {})
        tls = r.get("train_log_summary", {})
        rows.append({
            "seed": r["seed"],
            "n_updates": r.get("n_updates", 0),
            "wall_clock_sec": r.get("wall_clock_sec", 0),
            "warm_start_reward": ws.get("mean_reward", 0),
            "final_reward": fv.get("mean_reward", 0),
            "reward_improvement": fv.get("mean_reward", 0) - ws.get("mean_reward", 0),
            "pos_rate": fv.get("positive_improvement_rate", 0),
            "stop_rate": fv.get("stop_at_root_rate", 0),
            "constraint_validity": fv.get("constraint_validity", 0),
            "mean_loss": tls.get("mean_loss", 0),
            "mean_kl": tls.get("mean_kl", 0),
            "mean_entropy": tls.get("mean_entropy", 0),
            "n_skipped": tls.get("n_skipped", 0),
            "n_updated": tls.get("n_updated", 0),
        })
    # Sort by seed
    rows.sort(key=lambda x: x["seed"])
    return rows


def compute_summary_stats(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics across all seeds."""
    results = merged["results"]
    if not results:
        return {}

    pos_rates = [r["final_validation"]["positive_improvement_rate"] for r in results]
    stop_rates = [r["final_validation"]["stop_at_root_rate"] for r in results]
    constraints = [r["final_validation"]["constraint_validity"] for r in results]
    rewards = [r["final_validation"]["mean_reward"] for r in results]
    warm_rewards = [r["warm_start_validation"]["mean_reward"] for r in results]
    improvements = [f - w for f, w in zip(rewards, warm_rewards)]
    wall_times = [r.get("wall_clock_sec", 0) for r in results]

    return {
        "n_seeds": len(results),
        "pos_rate": {
            "mean": float(np.mean(pos_rates)),
            "std": float(np.std(pos_rates)),
            "min": float(np.min(pos_rates)),
            "max": float(np.max(pos_rates)),
            "values": pos_rates,
        },
        "stop_rate": {
            "mean": float(np.mean(stop_rates)),
            "std": float(np.std(stop_rates)),
            "max": float(np.max(stop_rates)),
            "values": stop_rates,
        },
        "constraint_validity": {
            "mean": float(np.mean(constraints)),
            "min": float(np.min(constraints)),
            "values": constraints,
        },
        "final_reward": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "values": rewards,
        },
        "reward_improvement": {
            "mean": float(np.mean(improvements)),
            "std": float(np.std(improvements)),
            "min": float(np.min(improvements)),
            "max": float(np.max(improvements)),
            "values": improvements,
        },
        "wall_clock": {
            "total": float(np.sum(wall_times)),
            "mean": float(np.mean(wall_times)),
            "max": float(np.max(wall_times)),
            "values": wall_times,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu6-json", required=True, help="Path to GPU 6 results JSON")
    parser.add_argument("--gpu7-json", required=True, help="Path to GPU 7 results JSON")
    parser.add_argument("--output-json", required=True, help="Path to output merged JSON")
    args = parser.parse_args()

    print(f"[1] Loading GPU 6 results: {args.gpu6_json}")
    with open(args.gpu6_json) as f:
        gpu6 = json.load(f)
    print(f"    {len(gpu6.get('results', []))} seeds, verdict={gpu6.get('verdict')}")

    print(f"[2] Loading GPU 7 results: {args.gpu7_json}")
    with open(args.gpu7_json) as f:
        gpu7 = json.load(f)
    print(f"    {len(gpu7.get('results', []))} seeds, verdict={gpu7.get('verdict')}")

    print(f"[3] Merging results")
    merged = merge_results(gpu6, gpu7)
    print(f"    Total: {merged['n_seeds']} seeds, {merged['n_seeds_failed']} failed")

    print(f"[4] Evaluating Gate B criteria")
    merged = evaluate_gate_b(merged)
    print(f"    Verdict: {merged['verdict']}")
    for k, v in merged["criteria"].items():
        if k != "all_pass":
            print(f"      {k}: {v}")

    print(f"[5] Generating horizontal comparison table")
    comparison = generate_horizontal_comparison(merged)
    merged["horizontal_comparison"] = comparison

    print(f"[6] Computing summary statistics")
    stats = compute_summary_stats(merged)
    merged["summary_stats"] = stats

    print(f"\nSummary:")
    if stats:
        print(f"  pos_rate: mean={stats['pos_rate']['mean']:.2%}, "
              f"min={stats['pos_rate']['min']:.2%}, max={stats['pos_rate']['max']:.2%}")
        print(f"  stop_rate: mean={stats['stop_rate']['mean']:.2%}, "
              f"max={stats['stop_rate']['max']:.2%}")
        print(f"  constraint: mean={stats['constraint_validity']['mean']:.2%}")
        print(f"  reward_improvement: mean={stats['reward_improvement']['mean']:.6f}")
        print(f"  total_wall_clock: {stats['wall_clock']['total']:.0f}s "
              f"({stats['wall_clock']['total']/3600:.1f}h)")

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\n[7] Results written to {args.output_json}")

    # Print horizontal comparison table
    print(f"\n{'='*120}")
    print(f"Horizontal Comparison Table (10 seeds)")
    print(f"{'='*120}")
    print(f"{'Seed':>6} {'Updates':>8} {'Wall(s)':>8} {'WarmR':>10} {'FinalR':>10} {'Improv':>10} "
          f"{'PosRate':>8} {'StopRt':>8} {'Constr':>8} {'MeanKL':>8} {'NSkip':>6}")
    print(f"{'-'*120}")
    for row in comparison:
        print(f"{row['seed']:>6} {row['n_updates']:>8} {row['wall_clock_sec']:>8.0f} "
              f"{row['warm_start_reward']:>10.6f} {row['final_reward']:>10.6f} "
              f"{row['reward_improvement']:>10.6f} {row['pos_rate']:>8.2%} "
              f"{row['stop_rate']:>8.2%} {row['constraint_validity']:>8.2%} "
              f"{row['mean_kl']:>8.4f} {row['n_skipped']:>6}")
    print(f"{'='*120}")

    return merged


if __name__ == "__main__":
    main()
