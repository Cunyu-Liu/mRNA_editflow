"""P2-01: Parallel counterfactual synergy panel v2.

Parallelized version of ``run_counterfactual_panel_v2.py`` using
``multiprocessing.Pool``. Produces IDENTICAL results to the serial version
(same per-wild-type seeds: ``seed + i * 1000``), just faster.

Usage
-----
    PYTHONHASHSEED=0 PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \\
    /home/cunyuliu/miniconda3/envs/editflow/bin/python -u \\
      scripts/run_counterfactual_panel_v2_parallel.py \\
        --records data/processed/gencode_human_transcripts.records.jsonl \\
        --n-wild-types 1000 --max-steps 8 --n-workers 16 --torch-threads 4 \\
        --output docs/cross_region_synergy_panel_results_v2.json \\
        --seed 1729
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.schema import MRNARecord

_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from run_counterfactual_panel_v2 import (  # noqa: E402
    run_panel_for_wild_type_v2,
    panel_to_dict_v2,
    load_wild_type_records_v2,
    WildTypePanelV2,
    build_multi_region_mask,
    OracleMDPV2,
    ARM_SPECS,
)

# Global state for workers (initialized in _worker_init).
_ORACLE = None
_REPO_ROOT_W = None
_DEVICE_W = "cpu"


def _worker_init(
    repo_root: str,
    seed: int,
    torch_threads: int,
    device: str,
    skip_cnn: bool,
) -> None:
    """Initialize each worker: set torch threads + load oracle."""
    global _ORACLE, _REPO_ROOT_W, _DEVICE_W
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    _REPO_ROOT_W = repo_root
    _DEVICE_W = device
    from mrna_editflow.eval.multi_region_oracle import (
        build_default_multi_region_oracle,
    )
    _ORACLE = build_default_multi_region_oracle(
        repo_root=repo_root,
        device=device,
        skip_cnn=skip_cnn,
    )


def _worker_task(args: Tuple[MRNARecord, int, int, int]) -> Dict[str, Any]:
    """Run the 8-arm panel for one wild-type."""
    global _ORACLE, _DEVICE_W
    record, global_idx, seed_base, max_steps = args
    device = torch.device(_DEVICE_W)
    seed_i = seed_base + global_idx * 1000
    torch.manual_seed(seed_i)
    panel = run_panel_for_wild_type_v2(
        record=record,
        oracle=_ORACLE,
        max_steps=max_steps,
        seed=seed_i,
        device=device,
    )
    d = panel_to_dict_v2(panel)
    d["_global_idx"] = global_idx
    return d


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2-01 parallel counterfactual synergy panel v2"
    )
    parser.add_argument(
        "--records",
        default="data/processed/gencode_human_transcripts.records.jsonl",
    )
    parser.add_argument("--n-wild-types", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-utr-len", type=int, default=500)
    parser.add_argument("--max-cds-len", type=int, default=3000)
    parser.add_argument("--max-3utr-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--repo-root", default=_REPO_ROOT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[panel_v2] Loading records from {args.records}")
    records = load_wild_type_records_v2(
        args.records,
        max_utr_len=args.max_utr_len,
        max_cds_len=args.max_cds_len,
        max_3utr_len=args.max_3utr_len,
    )
    print(f"[panel_v2] {len(records)} valid records after filters")

    if len(records) < args.n_wild_types:
        print(f"[panel_v2] Warning: only {len(records)} records, using all.")
        args.n_wild_types = len(records)
    records = records[: args.n_wild_types]

    print(f"[panel_v2] Running {len(records)} wild-types x 8 arms, "
          f"n_workers={args.n_workers}, skip_cnn={args.skip_cnn}")

    tasks: List[Tuple[MRNARecord, int, int, int]] = [
        (record, i, args.seed, args.max_steps)
        for i, record in enumerate(records)
    ]

    t0 = time.time()
    results: List[Dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]
    done = 0

    with Pool(
        processes=args.n_workers,
        initializer=_worker_init,
        initargs=(
            args.repo_root,
            args.seed,
            args.torch_threads,
            args.device,
            args.skip_cnn,
        ),
    ) as pool:
        for panel_dict in pool.imap_unordered(
            _worker_task, tasks, chunksize=args.chunksize
        ):
            idx = panel_dict.pop("_global_idx")
            results[idx] = panel_dict
            done += 1
            elapsed = time.time() - t0
            if done <= 5 or done % 50 == 0 or done == len(tasks):
                syn = panel_dict.get("synergy_scores", {}).get("syn_sum", 0.0)
                rate = done / max(elapsed, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"[panel_v2] {done}/{len(tasks)} "
                      f"syn_sum={syn:+.6f} ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)")

    panel_dicts = [r for r in results if r is not None]
    if not panel_dicts:
        print("[panel_v2] No panels collected. Aborting.")
        return 1

    output = {
        "config": {
            "records_path": args.records,
            "n_wild_types": args.n_wild_types,
            "max_steps": args.max_steps,
            "max_utr_len": args.max_utr_len,
            "max_cds_len": args.max_cds_len,
            "max_3utr_len": args.max_3utr_len,
            "seed": args.seed,
            "skip_cnn": args.skip_cnn,
            "n_workers": args.n_workers,
            "torch_threads_per_worker": args.torch_threads,
            "parallel": True,
            "oracle_version": "multi_region_v2_p2_01",
            "n_arms": 8,
        },
        "panels": panel_dicts,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    elapsed = time.time() - t0
    print(f"[panel_v2] Wrote {args.output} ({len(panel_dicts)} panels, {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
