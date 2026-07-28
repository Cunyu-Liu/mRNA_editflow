#!/usr/bin/env python3
"""P3-08 Checkpoint Selection Script.

Selects the best checkpoint from Gate B training based on:
1. Hard constraints = 100% (mandatory filter)
2. Independent-oracle LCB improvement (val_reward with uncertainty penalty)
3. Positive-improvement rate (higher is better)
4. Reward per edit (val_reward / edit_budget)
5. STOP calibration (stop_root close to 0, but not 100%)
6. KL constraint (mean KL during training, lower is better)
7. OOD performance (test-set metrics if available)

Usage:
    python scripts/select_p3_08_checkpoint.py \\
        --gpu6-log /tmp/p3_08_gateB/gateB_gpu6.log \\
        --gpu1-log /tmp/p3_08_gateB/gateB_gpu1_fix.log \\
        --gpu6-ckpt-dir checkpoints/p3_08_gateB_gpu6 \\
        --gpu1-ckpt-dir checkpoints/p3_08_gateB_gpu1 \\
        --output docs/p3_08_checkpoint_selection.json
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_training_log(log_path: str) -> List[Dict]:
    """Parse training log to extract per-step and validation metrics.

    Returns list of validation entries:
    [{"seed": int, "step": int, "val_reward": float, "pos_rate": float,
      "stop_root": float, "constraint": float}, ...]
    """
    validations = []
    current_seed = None

    with open(log_path) as f:
        for line in f:
            # Track current seed
            seed_match = re.search(r"\[Seed \d+/\d+\] seed=(\d+)", line)
            if seed_match:
                current_seed = int(seed_match.group(1))

            # Parse validation lines
            val_match = re.search(
                r"\[seed=(\d+)\] step (\d+): val_reward=([-\d.]+), "
                r"pos_rate=([\d.]+)%, stop_root=([\d.]+)%, constraint=([\d.]+)%",
                line,
            )
            if val_match:
                seed = int(val_match.group(1))
                step = int(val_match.group(2))
                val_reward = float(val_match.group(3))
                pos_rate = float(val_match.group(4)) / 100.0
                stop_root = float(val_match.group(5)) / 100.0
                constraint = float(val_match.group(6)) / 100.0
                validations.append({
                    "seed": seed,
                    "step": step,
                    "val_reward": val_reward,
                    "pos_rate": pos_rate,
                    "stop_root": stop_root,
                    "constraint": constraint,
                })

    return validations


def parse_kl_stats(log_path: str) -> Dict[int, Dict]:
    """Parse training log to extract KL statistics per seed.

    Returns {seed: {"mean_kl": float, "max_kl": float, "n_skips": int, "n_resets": int}}
    """
    stats = {}
    current_seed = None
    kl_values = []
    n_skips = 0
    n_resets = 0

    with open(log_path) as f:
        for line in f:
            seed_match = re.search(r"\[Seed \d+/\d+\] seed=(\d+)", line)
            if seed_match:
                if current_seed is not None:
                    stats[current_seed] = {
                        "mean_kl": sum(kl_values) / len(kl_values) if kl_values else 0,
                        "max_kl": max(kl_values) if kl_values else 0,
                        "n_skips": n_skips,
                        "n_resets": n_resets,
                    }
                current_seed = int(seed_match.group(1))
                kl_values = []
                n_skips = 0
                n_resets = 0

            # Parse KL from step lines
            kl_match = re.search(r"kl=([\d.]+)", line)
            if kl_match:
                kl_values.append(float(kl_match.group(1)))

            if "KL_SKIP" in line:
                n_skips += 1
            if "REFERENCE RESET" in line:
                n_resets += 1

    # Don't forget the last seed
    if current_seed is not None:
        stats[current_seed] = {
            "mean_kl": sum(kl_values) / len(kl_values) if kl_values else 0,
            "max_kl": max(kl_values) if kl_values else 0,
            "n_skips": n_skips,
            "n_resets": n_resets,
        }

    return stats


def find_checkpoints(ckpt_dir: str) -> List[Dict]:
    """Find checkpoint files in directory.

    Returns [{"seed": int, "step": int, "path": str}, ...]
    """
    checkpoints = []
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return checkpoints

    for f in sorted(ckpt_path.glob("grpo_seed*_step*.pt")):
        match = re.search(r"grpo_seed(\d+)_step(\d+)\.pt", f.name)
        if match:
            checkpoints.append({
                "seed": int(match.group(1)),
                "step": int(match.group(2)),
                "path": str(f),
            })

    return checkpoints


def compute_lcb(val_reward: float, n_trajectories: int = 192, alpha: float = 0.95) -> float:
    """Compute lower confidence bound on val_reward.

    LCB = mean - z * std / sqrt(n)
    Since we don't have per-trajectory rewards, use a conservative penalty.
    """
    # Conservative std estimate: 0.05 (typical for normalized rewards)
    std_est = 0.05
    z = 1.645  # 95% one-sided
    lcb = val_reward - z * std_est / math.sqrt(n_trajectories)
    return lcb


def select_best_checkpoint(
    validations: List[Dict],
    kl_stats: Dict[int, Dict],
    checkpoints: List[Dict],
    warm_start_reward: float = -0.10,
) -> Dict:
    """Select best checkpoint based on multiple criteria.

    Criteria (in priority order):
    1. Hard constraints = 100% (mandatory filter)
    2. Independent-oracle LCB improvement (val_reward LCB > warm_start)
    3. Positive-improvement rate (higher is better)
    4. Reward per edit (val_reward / edit_budget, edit_budget=1)
    5. STOP calibration (stop_root < 0.1, ideally 0)
    6. KL constraint (mean_kl < 0.15 during training)
    7. OOD performance (not available, skip)
    """
    # Match checkpoints to validation metrics
    ckpt_metrics = []
    for ckpt in checkpoints:
        # Find validation at this step (or closest before)
        matching_vals = [
            v for v in validations
            if v["seed"] == ckpt["seed"] and v["step"] == ckpt["step"]
        ]
        if not matching_vals:
            # Find closest validation before this step
            earlier_vals = [
                v for v in validations
                if v["seed"] == ckpt["seed"] and v["step"] <= ckpt["step"]
            ]
            if earlier_vals:
                matching_vals = [max(earlier_vals, key=lambda v: v["step"])]

        if not matching_vals:
            continue

        val = matching_vals[0]
        seed_kl = kl_stats.get(ckpt["seed"], {"mean_kl": 0, "n_skips": 0, "n_resets": 0})

        # Compute metrics
        lcb = compute_lcb(val["val_reward"])
        lcb_improvement = lcb - warm_start_reward
        reward_per_edit = val["val_reward"] / 1.0  # edit_budget=1

        ckpt_metrics.append({
            **ckpt,
            "val_reward": val["val_reward"],
            "pos_rate": val["pos_rate"],
            "stop_root": val["stop_root"],
            "constraint": val["constraint"],
            "lcb": lcb,
            "lcb_improvement": lcb_improvement,
            "reward_per_edit": reward_per_edit,
            "mean_kl": seed_kl["mean_kl"],
            "n_kl_skips": seed_kl["n_skips"],
            "n_ref_resets": seed_kl["n_resets"],
            # Scoring (higher is better)
            "score": (
                (1.0 if val["constraint"] >= 1.0 else 0.0) * 100  # Hard constraint
                + lcb_improvement * 50  # LCB improvement
                + val["pos_rate"] * 20  # Positive improvement rate
                + reward_per_edit * 10  # Reward per edit
                - val["stop_root"] * 30  # STOP penalty
                - min(seed_kl["mean_kl"], 0.15) * 20  # KL penalty (capped)
            ),
        })

    if not ckpt_metrics:
        return {"error": "No checkpoints with matching validation metrics found"}

    # Filter: hard constraints = 100%
    valid_ckpts = [c for c in ckpt_metrics if c["constraint"] >= 1.0]
    if not valid_ckpts:
        return {"error": "No checkpoints with 100% constraint validity"}

    # Rank by score
    ranked = sorted(valid_ckpts, key=lambda c: c["score"], reverse=True)

    # Best per seed
    best_per_seed = {}
    for c in ranked:
        if c["seed"] not in best_per_seed:
            best_per_seed[c["seed"]] = c

    # Overall best
    best = ranked[0]

    return {
        "best_checkpoint": best,
        "best_per_seed": best_per_seed,
        "all_ranked": ranked,
        "n_total": len(ckpt_metrics),
        "n_valid": len(valid_ckpts),
        "n_filtered": len(ckpt_metrics) - len(valid_ckpts),
    }


def main():
    parser = argparse.ArgumentParser(description="P3-08 Checkpoint Selection")
    parser.add_argument("--gpu6-log", required=True, help="GPU 6 training log path")
    parser.add_argument("--gpu1-log", required=True, help="GPU 1 training log path")
    parser.add_argument("--gpu6-ckpt-dir", required=True, help="GPU 6 checkpoint directory")
    parser.add_argument("--gpu1-ckpt-dir", required=True, help="GPU 1 checkpoint directory")
    parser.add_argument("--output", default="docs/p3_08_checkpoint_selection.json",
                        help="Output JSON path")
    parser.add_argument("--warm-start-reward", type=float, default=-0.10,
                        help="Warm start validation reward for LCB comparison")
    args = parser.parse_args()

    # Parse logs
    print("Parsing training logs...")
    gpu6_vals = parse_training_log(args.gpu6_log)
    gpu1_vals = parse_training_log(args.gpu1_log)
    gpu6_kl = parse_kl_stats(args.gpu6_log)
    gpu1_kl = parse_kl_stats(args.gpu1_log)

    all_vals = gpu6_vals + gpu1_vals
    all_kl = {**gpu6_kl, **gpu1_kl}

    print(f"  GPU 6: {len(gpu6_vals)} validation entries, {len(gpu6_kl)} seeds")
    print(f"  GPU 1: {len(gpu1_vals)} validation entries, {len(gpu1_kl)} seeds")

    # Find checkpoints
    print("\nFinding checkpoints...")
    gpu6_ckpts = find_checkpoints(args.gpu6_ckpt_dir)
    gpu1_ckpts = find_checkpoints(args.gpu1_ckpt_dir)
    all_ckpts = gpu6_ckpts + gpu1_ckpts

    print(f"  GPU 6: {len(gpu6_ckpts)} checkpoints")
    print(f"  GPU 1: {len(gpu1_ckpts)} checkpoints")
    print(f"  Total: {len(all_ckpts)} checkpoints")

    # Select best
    print("\nSelecting best checkpoint...")
    result = select_best_checkpoint(all_vals, all_kl, all_ckpts, args.warm_start_reward)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    # Print results
    best = result["best_checkpoint"]
    print(f"\n=== Best Checkpoint ===")
    print(f"  Seed: {best['seed']}")
    print(f"  Step: {best['step']}")
    print(f"  Path: {best['path']}")
    print(f"  Val Reward: {best['val_reward']:.6f}")
    print(f"  LCB: {best['lcb']:.6f}")
    print(f"  LCB Improvement: {best['lcb_improvement']:.6f}")
    print(f"  Pos Rate: {best['pos_rate']:.2%}")
    print(f"  Stop Root: {best['stop_root']:.2%}")
    print(f"  Constraint: {best['constraint']:.2%}")
    print(f"  Mean KL: {best['mean_kl']:.4f}")
    print(f"  KL Skips: {best['n_kl_skips']}")
    print(f"  Ref Resets: {best['n_ref_resets']}")
    print(f"  Score: {best['score']:.4f}")

    print(f"\n=== Per-Seed Best ===")
    print(f"{'Seed':>6} {'Step':>6} {'Val Reward':>12} {'Pos Rate':>10} {'Stop':>8} {'KL':>8} {'Score':>10}")
    for seed in sorted(result["best_per_seed"].keys()):
        c = result["best_per_seed"][seed]
        print(f"{c['seed']:>6} {c['step']:>6} {c['val_reward']:>12.6f} "
              f"{c['pos_rate']:>10.2%} {c['stop_root']:>8.2%} "
              f"{c['mean_kl']:>8.4f} {c['score']:>10.4f}")

    print(f"\n=== Summary ===")
    print(f"  Total checkpoints: {result['n_total']}")
    print(f"  Valid (100% constraint): {result['n_valid']}")
    print(f"  Filtered out: {result['n_filtered']}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
