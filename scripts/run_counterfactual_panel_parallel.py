#!/usr/bin/env python3
"""P1-13: Parallel counterfactual cross-region synergy edit panel.

This is a parallelized version of ``scripts/run_counterfactual_panel.py``
that uses ``multiprocessing.Pool`` to process wild-types concurrently.
It produces IDENTICAL results to the serial version (same per-wild-type
seeds: ``seed + i * 1000``), just faster.

Usage
-----
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \\
    /home/cunyuliu/miniconda3/envs/editflow/bin/python \\
    scripts/run_counterfactual_panel_parallel.py \\
        --records data/processed/gencode_human_transcripts.records.jsonl \\
        --n-wild-types 1000 --max-steps 8 --n-workers 16 \\
        --output docs/cross_region_synergy_panel_results_1000.json \\
        --seed 1729
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.eval.oracle import LocalTranslationOracle

# Import from the serial script to reuse all core logic.
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from run_counterfactual_panel import (  # noqa: E402
    run_panel_for_wild_type,
    panel_to_dict,
    aggregate_stats,
    load_wild_type_records,
    WildTypePanel,
)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init(max_utr_len: int, seed: int, torch_threads: int) -> None:
    """Initialize each worker process: set torch threads + seeds."""
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)


def _worker_task(args: Tuple[MRNARecord, int, int, int, int]) -> Dict[str, Any]:
    """Run the 5-arm panel for one wild-type.

    Args:
        (record, global_idx, seed_base, max_steps, max_utr_len)
    Returns:
        panel_dict (with global_idx embedded for reordering)
    """
    record, global_idx, seed_base, max_steps, max_utr_len = args
    device = torch.device("cpu")
    # Per-wild-type seed (matches serial version: seed + i * 1000).
    seed_i = seed_base + global_idx * 1000
    # CRITICAL: Seed the global torch RNG before creating TinyTrainableModel,
    # so that model initialization is deterministic and matches the serial
    # version's state at wild-type `global_idx`. In the serial version, the
    # global RNG advances as each wild-type creates a fresh model; here we
    # reproduce that by seeding before each model creation.
    torch.manual_seed(seed_i)
    # Each worker creates its own oracle (lightweight, no shared state).
    # Oracle seed must match serial version (uses args.seed, not seed_i).
    oracle = LocalTranslationOracle(max_len=max_utr_len, seed=seed_base)
    panel = run_panel_for_wild_type(
        record=record,
        oracle=oracle,
        max_steps=max_steps,
        seed=seed_i,
        device=device,
    )
    d = panel_to_dict(panel)
    d["_global_idx"] = global_idx
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="P1-13 parallel counterfactual synergy panel"
    )
    parser.add_argument(
        "--records",
        default="mrna_editflow/data/processed/gencode_human_transcripts.head64.records.jsonl",
        help="Path to wild-type records JSONL.",
    )
    parser.add_argument("--n-wild-types", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-utr-len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="Per-worker torch intra-op threads.")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="Pool.imap_unordered chunksize.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[panel] Loading records from {args.records}")
    records = load_wild_type_records(args.records, max_utr_len=args.max_utr_len)
    print(f"[panel] Loaded {len(records)} valid wild-type records "
          f"(5'UTR <= {args.max_utr_len} nt, no T)")

    if len(records) < args.n_wild_types:
        print(f"[panel] Warning: only {len(records)} records available, using all.")
        args.n_wild_types = len(records)

    records = records[: args.n_wild_types]
    print(f"[panel] Running panel on {len(records)} wild-types, "
          f"max_steps={args.max_steps}, n_workers={args.n_workers}")

    tasks: List[Tuple[MRNARecord, int, int, int, int]] = [
        (record, i, args.seed, args.max_steps, args.max_utr_len)
        for i, record in enumerate(records)
    ]

    t0 = time.time()
    results: List[Dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]
    done = 0

    with Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(args.max_utr_len, args.seed, args.torch_threads),
    ) as pool:
        for panel_dict in pool.imap_unordered(_worker_task, tasks,
                                              chunksize=args.chunksize):
            idx = panel_dict.pop("_global_idx")
            results[idx] = panel_dict
            done += 1
            elapsed = time.time() - t0
            if done <= 5 or done % 50 == 0 or done == len(tasks):
                syn_sum = panel_dict.get("synergy_scores", {}).get("syn_sum", 0.0)
                syn_wt = panel_dict.get("synergy_scores", {}).get("syn_vs_wt", 0.0)
                rate = done / max(elapsed, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"[panel] {done}/{len(tasks)} "
                      f"syn_sum={syn_sum:+.4f} syn_vs_wt={syn_wt:+.4f} "
                      f"({elapsed:.1f}s elapsed, {rate:.1f}/s, ETA {eta:.0f}s)")

    # Filter out any None (failed) entries.
    panel_dicts = [r for r in results if r is not None]
    if not panel_dicts:
        print("[panel] No panels collected. Aborting.")
        return 1

    # Reconstruct WildTypePanel objects for aggregate_stats.
    # We can't easily reconstruct from dicts, so recompute stats from dicts.
    # Instead, we'll build WildTypePanel objects from the dict structure.
    # Actually, aggregate_stats takes List[WildTypePanel]. The dict structure
    # from panel_to_dict has all the same fields. Let's reconstruct.
    from run_counterfactual_panel import ArmResult
    panels: List[WildTypePanel] = []
    for d in panel_dicts:
        wt = ArmResult(**d["wild_type"])
        s5 = ArmResult(**d["single_5utr"])
        sc = ArmResult(**d["single_cds"])
        s3 = ArmResult(**d["single_3utr"])
        jt = ArmResult(**d["joint"])
        p = WildTypePanel(
            transcript_id=d["transcript_id"],
            wild_type=wt,
            single_5utr=s5,
            single_cds=sc,
            single_3utr=s3,
            joint=jt,
            synergy_scores=d["synergy_scores"],
        )
        panels.append(p)

    stats = aggregate_stats(panels)

    output = {
        "config": {
            "records_path": args.records,
            "n_wild_types": args.n_wild_types,
            "max_steps": args.max_steps,
            "max_utr_len": args.max_utr_len,
            "seed": args.seed,
            "n_workers": args.n_workers,
            "torch_threads_per_worker": args.torch_threads,
            "parallel": True,
        },
        "stats": asdict(stats),
        "panels": panel_dicts,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"[panel] Wrote {args.output}")

    # Print summary.
    print(f"\n[panel] === Summary ({len(panels)} wild-types) ===")
    print(f"  syn_sum  : mean={stats.syn_sum_mean:+.4f} "
          f"std={stats.syn_sum_std:.4f} median={stats.syn_sum_median:+.4f}")
    print(f"  syn_vs_wt: mean={stats.syn_vs_wt_mean:+.4f} "
          f"std={stats.syn_vs_wt_std:.4f} median={stats.syn_vs_wt_median:+.4f}")
    print(f"  t-stat (syn_sum=0): {stats.t_stat_syn_sum:.4f} "
          f"p={stats.t_pvalue_syn_sum:.4g}")
    print(f"  Cohen's d (syn_sum): {stats.cohens_d_syn_sum:.4f}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
