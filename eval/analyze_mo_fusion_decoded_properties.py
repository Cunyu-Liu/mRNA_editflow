#!/usr/bin/env python3
"""Aggregate decoded-output multi-objective properties across the 4 fusion-mode
benchmarks (te_only / scalar / pareto / grpo) for a given data slice.

The multiseed aggregate only exposes oracle TE/MRL, but the multi-objective
upgrade's value lies in the OTHER decoded-sequence properties (uAUG, start
accessibility, GC, CAI). Those are retained per seed under
``eval_summary.json::metrics``. This script means them across the 10 seeds per
mode so the fusion tradeoff is visible, not just "TE slightly lower".

CPU-only, reads finished eval_summary.json files. Usage:
    python3 analyze_mo_fusion_decoded_properties.py [BENCH_ROOT] [SLICE]
where SLICE defaults to head256 (e.g. head1024 for the scaled slice).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

from mrna_editflow.eval.run_eval import paired_permutation_pvalue

MODES = ("te_only", "scalar", "pareto", "grpo")
# metric group -> key within that group (all are decoded top-1 output properties)
PROPS = {
    "delta_oracle_te_vs_source": ("aggregate", "delta_oracle_te_vs_source"),
    "mean_oracle_te": ("aggregate", "mean_oracle_te"),
    "mean_oracle_mrl": ("aggregate", "mean_oracle_mrl"),
    "mean_uaug_count": ("metrics", "kozak_uaug", "mean_uaug_count"),
    "uaug_fraction": ("metrics", "kozak_uaug", "uaug_fraction"),
    "mean_start_accessibility": ("metrics", "structure", "mean_start_accessibility"),
    "candidate_mean_gc": ("metrics", "distribution", "candidate_mean_gc"),
    "mean_cai": ("metrics", "cai", "mean_cai"),
    "mean_kozak_score": ("metrics", "kozak_uaug", "mean_kozak_score"),
}


def _dig(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, dict):
        cur = cur.get("mean")
    return cur if isinstance(cur, (int, float)) else None


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mode_values(bench_root: str, mode: str, slice_name: str = "head256", top_k: int = 64) -> dict[str, float]:
    d = os.path.join(bench_root, f"multiseed_t5_public_{slice_name}_mo_{mode}_top{top_k}")
    seed_files = sorted(glob.glob(os.path.join(d, "seed_*", "eval_summary.json")))
    agg_path = os.path.join(d, "multiseed_summary.json")
    agg = _read_json(agg_path)["aggregate"] if os.path.exists(agg_path) else {}
    out: dict[str, float] = {"n_seeds": float(len(seed_files))}
    for prop, path in PROPS.items():
        if path[0] == "aggregate":
            v = agg.get(path[1], {})
            out[prop] = float(v.get("mean")) if isinstance(v, dict) and "mean" in v else float("nan")
        else:
            vals = []
            for f in seed_files:
                try:
                    val = _dig(_read_json(f), path)
                except (OSError, json.JSONDecodeError):
                    val = None
                if val is not None:
                    vals.append(float(val))
            out[prop] = statistics.mean(vals) if vals else float("nan")
    return out


def mode_seed_vectors(
    bench_root: str, mode: str, slice_name: str = "head256", top_k: int = 64
) -> dict[str, list[float]]:
    """Per-seed value lists for the metrics-group decoded properties, ordered by seed.

    These feed a seed-paired permutation test vs the te_only control so the
    multi-objective non-TE tradeoffs (uAUG/accessibility/GC/CAI) carry the same
    10-seed paired significance standard applied to oracle TE, not just raw means.
    """
    d = os.path.join(bench_root, f"multiseed_t5_public_{slice_name}_mo_{mode}_top{top_k}")
    seed_files = sorted(glob.glob(os.path.join(d, "seed_*", "eval_summary.json")))
    vectors: dict[str, list[float]] = {p: [] for p, path in PROPS.items() if path[0] == "metrics"}
    for f in seed_files:
        try:
            payload = _read_json(f)
        except (OSError, json.JSONDecodeError):
            payload = None
        for prop, path in PROPS.items():
            if path[0] != "metrics":
                continue
            val = _dig(payload, path) if payload is not None else None
            vectors[prop].append(float(val) if val is not None else float("nan"))
    return vectors


def main(argv):
    bench_root = argv[1] if len(argv) > 1 else "benchmark"
    slice_name = argv[2] if len(argv) > 2 else "head256"
    rows = {m: mode_values(bench_root, m, slice_name) for m in MODES}
    base = rows["te_only"]
    print(f"slice={slice_name}")
    print(f"{'metric':<28}" + "".join(f"{m:>14}" for m in MODES))
    for prop in PROPS:
        line = f"{prop:<28}"
        for m in MODES:
            line += f"{rows[m].get(prop, float('nan')):>14.5f}"
        print(line)
    print("\n=== deltas vs te_only (fusion - control) ===")
    print(f"{'metric':<28}" + "".join(f"{m:>14}" for m in ("scalar", "pareto", "grpo")))
    for prop in PROPS:
        line = f"{prop:<28}"
        for m in ("scalar", "pareto", "grpo"):
            d = rows[m].get(prop, float("nan")) - base.get(prop, float("nan"))
            line += f"{d:>+14.5f}"
        print(line)

    # Seed-paired permutation significance for the per-seed decoded properties,
    # matching the 10-seed paired standard used for oracle TE. Aggregate-group
    # metrics (oracle TE/MRL) already get paired p via compare_benchmarks.
    vecs = {m: mode_seed_vectors(bench_root, m, slice_name) for m in MODES}
    metric_props = [p for p, path in PROPS.items() if path[0] == "metrics"]
    if metric_props:
        print("\n=== seed-paired permutation p vs te_only (per-seed decoded props) ===")
        print(f"{'metric':<28}" + "".join(f"{m:>14}" for m in ("scalar", "pareto", "grpo")))
        for prop in metric_props:
            line = f"{prop:<28}"
            for m in ("scalar", "pareto", "grpo"):
                cand = vecs[m].get(prop, [])
                ctrl = vecs["te_only"].get(prop, [])
                n = min(len(cand), len(ctrl))
                if n == 0:
                    line += f"{'nan':>14}"
                else:
                    p = paired_permutation_pvalue(cand[:n], ctrl[:n])
                    line += f"{p:>14.5f}"
            print(line)
    print(f"\nseeds per mode: " + ", ".join(f"{m}={int(rows[m]['n_seeds'])}" for m in MODES))


if __name__ == "__main__":
    main(sys.argv)
