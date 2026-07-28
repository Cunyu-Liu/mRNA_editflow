#!/usr/bin/env python
"""Generate Gate B training health report from merged results.

Usage:
    python scripts/generate_gateB_health_report.py \
        --merged-json docs/p3_08_grpo_results_gateB.json \
        --output docs/p3_08_grpo_training_health_gateB.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def generate_report(merged: Dict[str, Any]) -> str:
    """Generate markdown health report."""
    criteria = merged.get("criteria", {})
    stats = merged.get("summary_stats", {})
    comparison = merged.get("horizontal_comparison", [])
    gpu_breakdown = merged.get("gpu_breakdown", {})
    config = merged.get("config", {})

    verdict = merged.get("verdict", "UNKNOWN")
    n_seeds = merged.get("n_seeds", 0)
    n_failed = merged.get("n_seeds_failed", 0)
    total_wall = merged.get("total_wall_clock_sec", 0)

    lines = []
    lines.append("# P3-08 Gate B Training Health Report")
    lines.append("")
    lines.append(f"**Document**: `docs/p3_08_grpo_training_health_gateB.md`")
    lines.append(f"**Phase**: P3-08 Production GRPO / Amortized Policy Training")
    lines.append(f"**Gate**: B (10-seed paper run, 5000 updates, edit_budget=1)")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Verdict**: **{verdict}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Configuration
    lines.append("## 1. Training Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| n_updates | {config.get('n_updates', 'N/A')} |")
    lines.append(f"| edit_budget | {config.get('edit_budget', 'N/A')} |")
    lines.append(f"| sources_per_batch | {config.get('sources_per_batch', 'N/A')} |")
    lines.append(f"| group_size | {config.get('group_size', 'N/A')} |")
    lines.append(f"| lr | {config.get('lr', 'N/A')} |")
    lines.append(f"| beta_kl | {config.get('beta_kl', 'N/A')} |")
    lines.append(f"| beta_entropy | {config.get('beta_entropy', 'N/A')} |")
    lines.append(f"| max_kl | {config.get('max_kl', 'N/A')} |")
    lines.append(f"| clip_epsilon | {config.get('clip_epsilon', 'N/A')} |")
    lines.append(f"| warmup_steps | {config.get('warmup_steps', 'N/A')} |")
    lines.append(f"| stop_penalty | {config.get('stop_penalty', 'N/A')} |")
    lines.append(f"| validation_interval | {config.get('validation_interval', 'N/A')} |")
    lines.append(f"| checkpoint_interval | {config.get('checkpoint_interval', 'N/A')} |")
    lines.append("")

    # GPU breakdown
    lines.append("## 2. GPU Breakdown")
    lines.append("")
    lines.append("| GPU | Seeds | Completed | Failed | Wall Clock (s) | Wall Clock (h) |")
    lines.append("|-----|-------|-----------|--------|----------------|----------------|")
    for gpu_name, gpu_info in gpu_breakdown.items():
        seeds = ", ".join(str(s) for s in gpu_info.get("seeds", []))
        wall = gpu_info.get("wall_clock_sec", 0)
        lines.append(f"| {gpu_name} | {seeds} | {gpu_info.get('n_completed', 0)} | "
                     f"{gpu_info.get('n_failed', 0)} | {wall:.0f} | {wall/3600:.1f} |")
    lines.append(f"| **Total** | **{n_seeds}** | **{n_seeds}** | **{n_failed}** | "
                 f"**{total_wall:.0f}** | **{total_wall/3600:.1f}** |")
    lines.append("")

    # Gate criteria evaluation
    lines.append("## 3. Gate B Criteria Evaluation")
    lines.append("")
    lines.append("| Criterion | Threshold | Result | Pass |")
    lines.append("|-----------|-----------|--------|------|")

    # no_collapse
    finite_fracs = criteria.get("finite_fractions", [])
    min_frac = min(finite_fracs) if finite_fracs else 0
    lines.append(f"| no_collapse | ≥99% finite | min={min_frac:.4f} | "
                 f"{'✅' if criteria.get('no_collapse') else '❌'} |")

    # hard_constraints_100
    constrs = criteria.get("constraint_validities", [])
    min_constr = min(constrs) if constrs else 0
    lines.append(f"| hard_constraints_100 | 100% | min={min_constr:.4f} | "
                 f"{'✅' if criteria.get('hard_constraints_100') else '❌'} |")

    # two_thirds_beat_warm
    beat = criteria.get("beat_warm_start", [])
    n_beat = sum(beat) if beat else 0
    lines.append(f"| two_thirds_beat_warm | ≥{2*n_seeds//3}/{n_seeds} | {n_beat}/{n_seeds} | "
                 f"{'✅' if criteria.get('two_thirds_beat_warm') else '❌'} |")

    # no_reward_hacking
    pos_rates = criteria.get("positive_rates", [])
    min_pos = min(pos_rates) if pos_rates else 0
    lines.append(f"| no_reward_hacking | pos_rate≥30% | min={min_pos:.2%} | "
                 f"{'✅' if criteria.get('no_reward_hacking') else '❌'} |")

    # stop_not_collapsed
    stop_rates = criteria.get("stop_rates", [])
    max_stop = max(stop_rates) if stop_rates else 1
    lines.append(f"| stop_not_collapsed | <90% | max={max_stop:.2%} | "
                 f"{'✅' if criteria.get('stop_not_collapsed') else '❌'} |")

    # kl_normal
    mean_kls = criteria.get("mean_kls", [])
    max_kl = max(mean_kls) if mean_kls else 1
    lines.append(f"| kl_normal | <0.5 | max={max_kl:.4f} | "
                 f"{'✅' if criteria.get('kl_normal') else '❌'} |")

    # clip_normal
    mean_clips = criteria.get("mean_clips", [])
    max_clip = max(mean_clips) if mean_clips else 0
    lines.append(f"| clip_normal | [0.0, 0.5] | max={max_clip:.4f} | "
                 f"{'✅' if criteria.get('clip_normal') else '❌'} |")

    lines.append(f"| **Overall** | | | **{verdict}** |")
    lines.append("")

    # Summary statistics
    lines.append("## 4. Summary Statistics (10 seeds)")
    lines.append("")
    if stats:
        lines.append("### 4.1 Positive Improvement Rate")
        pr = stats.get("pos_rate", {})
        lines.append(f"- Mean: {pr.get('mean', 0):.2%}")
        lines.append(f"- Std: {pr.get('std', 0):.2%}")
        lines.append(f"- Min: {pr.get('min', 0):.2%}")
        lines.append(f"- Max: {pr.get('max', 0):.2%}")
        lines.append(f"- Values: {[f'{v:.2%}' for v in pr.get('values', [])]}")
        lines.append("")

        lines.append("### 4.2 STOP Rate")
        sr = stats.get("stop_rate", {})
        lines.append(f"- Mean: {sr.get('mean', 0):.2%}")
        lines.append(f"- Max: {sr.get('max', 0):.2%}")
        lines.append(f"- Values: {[f'{v:.2%}' for v in sr.get('values', [])]}")
        lines.append("")

        lines.append("### 4.3 Constraint Validity")
        cv = stats.get("constraint_validity", {})
        lines.append(f"- Mean: {cv.get('mean', 0):.2%}")
        lines.append(f"- Min: {cv.get('min', 0):.2%}")
        lines.append("")

        lines.append("### 4.4 Reward Improvement (Final - Warm Start)")
        ri = stats.get("reward_improvement", {})
        lines.append(f"- Mean: {ri.get('mean', 0):.6f}")
        lines.append(f"- Std: {ri.get('std', 0):.6f}")
        lines.append(f"- Min: {ri.get('min', 0):.6f}")
        lines.append(f"- Max: {ri.get('max', 0):.6f}")
        lines.append("")

        lines.append("### 4.5 Wall Clock Time")
        wc = stats.get("wall_clock", {})
        lines.append(f"- Total: {wc.get('total', 0):.0f}s ({wc.get('total', 0)/3600:.1f}h)")
        lines.append(f"- Mean per seed: {wc.get('mean', 0):.0f}s ({wc.get('mean', 0)/3600:.1f}h)")
        lines.append(f"- Max: {wc.get('max', 0):.0f}s ({wc.get('max', 0)/3600:.1f}h)")
        lines.append("")

    # Horizontal comparison table
    lines.append("## 5. Horizontal Comparison Table")
    lines.append("")
    lines.append("| Seed | Updates | Wall(s) | Warm Reward | Final Reward | Improvement | "
                 "Pos Rate | Stop Rate | Constraint | Mean KL | N Skip |")
    lines.append("|------|---------|---------|-------------|--------------|-------------|"
                 "----------|-----------|------------|---------|--------|")
    for row in comparison:
        lines.append(
            f"| {row['seed']} | {row['n_updates']} | {row['wall_clock_sec']:.0f} | "
            f"{row['warm_start_reward']:.6f} | {row['final_reward']:.6f} | "
            f"{row['reward_improvement']:.6f} | {row['pos_rate']:.2%} | "
            f"{row['stop_rate']:.2%} | {row['constraint_validity']:.2%} | "
            f"{row['mean_kl']:.4f} | {row['n_skipped']} |"
        )
    lines.append("")

    # Per-seed validation trajectories
    lines.append("## 6. Per-Seed Validation Trajectories")
    lines.append("")
    for r in merged.get("results", []):
        seed = r["seed"]
        val_log = r.get("validation_log_summary", [])
        if not val_log:
            continue
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| Step | Mean Reward | Pos Rate | Constraint |")
        lines.append("|------|-------------|----------|------------|")
        for v in val_log:
            lines.append(f"| {v['step']} | {v['mean_reward']:.6f} | "
                         f"{v['positive_improvement_rate']:.2%} | "
                         f"{v['constraint_validity']:.2%} |")
        lines.append("")

    # KL controller analysis
    lines.append("## 7. KL Controller Analysis")
    lines.append("")
    lines.append("The Adaptive KL Controller uses a 4-tier strategy:")
    lines.append("- **MIN_COEFFICIENT=0.3**: Strong baseline penalty")
    lines.append("- **Proactive tier at 0.5×max_kl**: 1.5× coefficient increase")
    lines.append("- **max_kl threshold**: 2× coefficient increase")
    lines.append("- **Skip at 1.3×max_kl**: Hard reject + 2× coefficient (cap 2.0)")
    lines.append("- **Reference reset after 30 consecutive KL_SKIPs**: Breaks deadlocks")
    lines.append("")
    lines.append("| Seed | N Updates | N Skipped | N Updated | Skip Rate |")
    lines.append("|------|-----------|-----------|-----------|-----------|")
    for row in comparison:
        n_upd = row["n_updated"]
        n_skip = row["n_skipped"]
        total = n_upd + n_skip
        skip_rate = n_skip / total if total > 0 else 0
        lines.append(f"| {row['seed']} | {total} | {n_skip} | {n_upd} | {skip_rate:.2%} |")
    lines.append("")

    # Conclusion
    lines.append("## 8. Conclusion")
    lines.append("")
    if verdict == "PASS":
        lines.append(f"**Gate B verdict: PASS**")
        lines.append("")
        lines.append(f"All 7 criteria satisfied across {n_seeds} seeds:")
        lines.append(f"- No numerical collapse (all loss/kl finite)")
        lines.append(f"- 100% constraint validity (protein identity + length preserved)")
        lines.append(f"- ≥2/3 seeds beat warm start baseline")
        lines.append(f"- Positive improvement rate ≥30% (mean={stats.get('pos_rate', {}).get('mean', 0):.2%})")
        lines.append(f"- STOP rate <90% (max={stats.get('stop_rate', {}).get('max', 0):.2%})")
        lines.append(f"- KL normal (max={max(mean_kls) if mean_kls else 0:.4f} < 0.5)")
        lines.append(f"- Clip fraction normal (max={max(mean_clips) if mean_clips else 0:.4f} ≤ 0.5)")
        lines.append("")
        lines.append("The production GRPO policy demonstrates stable training across 10 seeds "
                     "with consistent improvement over warm start baseline. The reference reset "
                     "mechanism successfully breaks KL deadlocks without losing learned progress.")
    else:
        lines.append(f"**Gate B verdict: {verdict}**")
        lines.append("")
        lines.append("Some criteria not satisfied. See table above for details.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by `scripts/generate_gateB_health_report.py` at "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-json", required=True, help="Path to merged results JSON")
    parser.add_argument("--output", required=True, help="Path to output markdown report")
    args = parser.parse_args()

    print(f"[1] Loading merged results: {args.merged_json}")
    with open(args.merged_json) as f:
        merged = json.load(f)
    print(f"    {merged.get('n_seeds', 0)} seeds, verdict={merged.get('verdict')}")

    print(f"[2] Generating health report")
    report = generate_report(merged)

    print(f"[3] Writing report to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"    Done ({len(report)} bytes)")

    return report


if __name__ == "__main__":
    main()
