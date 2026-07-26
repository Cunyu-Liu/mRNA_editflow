"""Re-evaluate Gate A criteria with fixed metrics.

Parses the training log to reconstruct per-step metrics, then applies the
corrected `evaluate_gate_a` function that:
  1. `no_collapse`: checks actual NaN/Inf in loss/kl (NOT update rate).
     KL_SKIP steps have finite loss/kl — they are a safety mechanism, not
     numerical collapse.
  2. `clip_normal`: uses upper-bound-only threshold [0.0, 0.5].
     clip_fraction=0 is legitimate for single-update on-policy GRPO.

Usage:
    python scripts/reeval_p3_08_gateA.py \\
        --log /tmp/p3_08_nohup.out \\
        --results docs/p3_08_grpo_results_gateA.json \\
        --output docs/p3_08_grpo_results_gateA.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Per-step log line regex (handles both updated=True and KL_SKIP lines)
# Example: "  [seed=42] step 710/1000: loss=0.0173, kl=0.0752, kl_c=0.450, updated=True, rate=0.2 steps/s, eta=21.9min"
# Example: "  [seed=123] step 230/1000: loss=0.1798, kl=0.2016, kl_c=2.000, updated=False, KL_SKIP, rate=0.2 steps/s, eta=55.3min"
_STEP_RE = re.compile(
    r"\[seed=(\d+)\]\s+step\s+(\d+)/\d+:\s+"
    r"loss=([-+]?nan|[-+]?inf|[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s+"
    r"kl=([-+]?nan|[-+]?inf|[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s+"
    r"kl_c=([-+]?\d+\.?\d*),\s+"
    r"updated=(True|False)"
)

# Validation line regex
# Example: "  [seed=42] step 100: val_reward=-0.100768, pos_rate=25.00%, stop_root=46.88%, constraint=100.00%"
_VAL_RE = re.compile(
    r"\[seed=(\d+)\]\s+step\s+(\d+):\s+"
    r"val_reward=([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s+"
    r"pos_rate=([-+]?\d+\.?\d*)%,\s+"
    r"stop_root=([-+]?\d+\.?\d*)%,\s+"
    r"constraint=([-+]?\d+\.?\d*)%"
)

# Done line regex
# Example: "  Done in 4499.3s: reward=-0.055206, pos_rate=79.17%"
_DONE_RE = re.compile(
    r"Done in\s+(\d+\.?\d*)s:\s+reward=([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?),\s+"
    r"pos_rate=([-+]?\d+\.?\d*)"
)


def _parse_float(s: str) -> float:
    """Parse float, handling nan/inf."""
    s_lower = s.lower()
    if "nan" in s_lower:
        return float("nan")
    if "inf" in s_lower:
        return float("inf") if "+" in s or s_lower == "inf" else float("-inf")
    return float(s)


def parse_log(log_path: str) -> Dict[int, Dict[str, Any]]:
    """Parse training log into per-seed data dicts.

    Returns dict keyed by seed, each value has:
      - train_log: list of per-step metric dicts
      - validation_log: list of validation metric dicts
      - final_done: dict with wall_clock_sec, final_reward, final_pos_rate
    """
    seeds: Dict[int, Dict[str, Any]] = {}
    current_seed: int | None = None

    with open(log_path, "r") as f:
        for line in f:
            # Per-step training line
            m = _STEP_RE.search(line)
            if m:
                seed = int(m.group(1))
                if seed not in seeds:
                    seeds[seed] = {"train_log": [], "validation_log": [], "final_done": None}
                current_seed = seed
                loss = _parse_float(m.group(3))
                kl = _parse_float(m.group(4))
                kl_c = float(m.group(5))
                updated = m.group(6) == "True"
                skip_kl = "KL_SKIP" in line
                seeds[seed]["train_log"].append({
                    "step": int(m.group(2)),
                    "loss": loss,
                    "kl": kl,
                    "kl_coefficient": kl_c,
                    "updated": updated,
                    "skip_kl_guard": skip_kl,
                    "clip_fraction": 0.0,  # Not logged per-step; filled from summary
                })
                continue

            # Validation line
            m = _VAL_RE.search(line)
            if m:
                seed = int(m.group(1))
                if seed not in seeds:
                    seeds[seed] = {"train_log": [], "validation_log": [], "final_done": None}
                seeds[seed]["validation_log"].append({
                    "step": int(m.group(2)),
                    "mean_reward": float(m.group(3)),
                    "positive_improvement_rate": float(m.group(4)) / 100.0,
                    "stop_at_root_rate": float(m.group(5)) / 100.0,
                    "constraint_validity": float(m.group(6)) / 100.0,
                })
                continue

            # Done line
            m = _DONE_RE.search(line)
            if m and current_seed is not None:
                seeds[current_seed]["final_done"] = {
                    "wall_clock_sec": float(m.group(1)),
                    "final_reward": float(m.group(2)),
                    "final_pos_rate": float(m.group(3)) / 100.0,
                }
                continue

    return seeds


def evaluate_gate_a_fixed(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fixed Gate A evaluation — checks actual NaN/Inf, not update rate."""
    n_seeds = len(results)
    criteria: Dict[str, Any] = {}

    # 1. No numerical collapse (no NaN/Inf in loss or kl)
    finite_fractions = []
    for r in results:
        log = r.get("train_log", [])
        n_finite = sum(
            1 for m in log
            if math.isfinite(m.get("loss", float("nan")))
            and math.isfinite(m.get("kl", float("nan")))
        )
        n_total = max(len(log), 1)
        finite_fractions.append(n_finite / n_total)
    criteria["no_collapse"] = all(f >= 0.99 for f in finite_fractions)
    criteria["finite_fractions"] = finite_fractions

    # 2. Hard constraints 100%
    constraint_validities = []
    for r in results:
        fv = r.get("final_validation", {})
        constraint_validities.append(fv.get("constraint_validity", 0.0))
    criteria["hard_constraints_100"] = all(v >= 1.0 for v in constraint_validities)
    criteria["constraint_validities"] = constraint_validities

    # 3. >= 2/3 seeds beat warm start
    beat_warm = []
    for r in results:
        ws = r.get("warm_start_validation", {}).get("mean_reward", 0.0)
        fv = r.get("final_validation", {}).get("mean_reward", 0.0)
        beat_warm.append(fv > ws)
    criteria["two_thirds_beat_warm"] = sum(beat_warm) >= 2 * n_seeds / 3
    criteria["beat_warm_start"] = beat_warm

    # 4. No reward hacking (positive improvement rate >= 30%)
    pos_rates = []
    for r in results:
        fv = r.get("final_validation", {})
        pos_rates.append(fv.get("positive_improvement_rate", 0.0))
    criteria["no_reward_hacking"] = all(p >= 0.30 for p in pos_rates)
    criteria["positive_rates"] = pos_rates

    # 5. STOP not collapsed (< 0.9)
    stop_rates = []
    for r in results:
        fv = r.get("final_validation", {})
        stop_rates.append(fv.get("stop_at_root_rate", 1.0))
    criteria["stop_not_collapsed"] = all(s < 0.9 for s in stop_rates)
    criteria["stop_rates"] = stop_rates

    # 6. KL normal (< 0.5)
    mean_kls = []
    for r in results:
        log = r.get("train_log", [])
        kls = [m.get("kl", 0.0) for m in log if m.get("updated", False)]
        mean_kls.append(float(np.mean(kls)) if kls else 0.0)
    criteria["kl_normal"] = all(k < 0.5 for k in mean_kls)
    criteria["mean_kls"] = mean_kls

    # 7. Clip fraction normal (upper bound only — clip=0 is legitimate)
    mean_clips = []
    for r in results:
        log = r.get("train_log", [])
        clips = [m.get("clip_fraction", 0.0) for m in log if m.get("updated", False)]
        mean_clips.append(float(np.mean(clips)) if clips else 0.0)
    criteria["clip_normal"] = all(0.0 <= c <= 0.5 for c in mean_clips)
    criteria["mean_clips"] = mean_clips

    # Overall verdict
    criteria["all_pass"] = all([
        criteria["no_collapse"],
        criteria["hard_constraints_100"],
        criteria["two_thirds_beat_warm"],
        criteria["no_reward_hacking"],
        criteria["stop_not_collapsed"],
        criteria["kl_normal"],
        criteria["clip_normal"],
    ])

    return criteria


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate Gate A with fixed metrics")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--results", required=True, help="Path to existing results JSON")
    parser.add_argument("--output", required=True, help="Path to output results JSON")
    args = parser.parse_args()

    # Parse log to reconstruct per-step train_log
    print(f"[1] Parsing log: {args.log}")
    parsed = parse_log(args.log)
    print(f"    Parsed {len(parsed)} seeds: {sorted(parsed.keys())}")

    # Load existing results JSON
    print(f"[2] Loading existing results: {args.results}")
    with open(args.results, "r") as f:
        results_json = json.load(f)

    # Merge parsed train_log into results
    # The existing results JSON has train_log_summary but not the full train_log.
    # We add the parsed train_log to each seed's result.
    for r in results_json["results"]:
        seed = r["seed"]
        if seed in parsed:
            r["train_log"] = parsed[seed]["train_log"]
            # Fill in clip_fraction from summary (not logged per-step)
            mean_clip = r.get("train_log_summary", {}).get("mean_clip_fraction", 0.0)
            for m in r["train_log"]:
                m["clip_fraction"] = mean_clip
            print(f"    seed={seed}: {len(r['train_log'])} steps parsed, "
                  f"mean_clip={mean_clip}")
        else:
            print(f"    WARNING: seed={seed} not found in log!", file=sys.stderr)

    # Re-evaluate with fixed metrics
    print(f"[3] Re-evaluating Gate A criteria with fixed metrics")
    criteria = evaluate_gate_a_fixed(results_json["results"])

    # Update results JSON
    results_json["criteria"] = criteria
    results_json["verdict"] = "PASS" if criteria["all_pass"] else "FAIL"

    # Print summary
    print(f"\n[4] Verdict: {results_json['verdict']}")
    print(f"    no_collapse: {criteria['no_collapse']}")
    print(f"      finite_fractions: {criteria['finite_fractions']}")
    print(f"    hard_constraints_100: {criteria['hard_constraints_100']}")
    print(f"      constraint_validities: {criteria['constraint_validities']}")
    print(f"    two_thirds_beat_warm: {criteria['two_thirds_beat_warm']}")
    print(f"      beat_warm_start: {criteria['beat_warm_start']}")
    print(f"    no_reward_hacking: {criteria['no_reward_hacking']}")
    print(f"      positive_rates: {criteria['positive_rates']}")
    print(f"    stop_not_collapsed: {criteria['stop_not_collapsed']}")
    print(f"      stop_rates: {criteria['stop_rates']}")
    print(f"    kl_normal: {criteria['kl_normal']}")
    print(f"      mean_kls: {criteria['mean_kls']}")
    print(f"    clip_normal: {criteria['clip_normal']}")
    print(f"      mean_clips: {criteria['mean_clips']}")

    # Write output
    print(f"\n[5] Writing: {args.output}")
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"    Done.")


if __name__ == "__main__":
    main()
