#!/usr/bin/env python3
"""P1-3 (audit A9): per-cell bootstrap CI + MDE for the 15 new-row cells.

5 families x (M1 / M6 / S1 two arms) = 15 cells. For each cell:
  - point rho = task Spearman(delta_hat, observed) on full rows (recomputed;
    cross-checked against matrix_v2_results.json archived values)
  - 95% bootstrap CI: source-group cluster bootstrap, 2,000 iterations,
    seed 20260920 (resample source groups with replacement; keep all rows
    within a group; recompute Spearman per iteration)
  - MDE: minimum detectable |rho| at alpha=0.05 two-sided given observed n
    (Fisher z: |z| >= 1.959964 => MDE = tanh(1.959964/sqrt(n-3)))
  - interpretation boundary statement: CI crossing zero vs near-zero point
    reads (|rho| small != 0 is power-limited, not evidence of absence)

Row-level caliber note: M1 source groups are per-file prefix/Hamming groups
(source_id in predictions.jsonl); M6 source group = Family; S1 source group
= gene x arm (source_id). Bootstrap uses source_id as the cluster variable
(read-only from predictions.jsonl). Zero new forward passes; CPU only;
protected TEST reads = 0. No verdict changes (matrix values stay frozen;
this is a power-statement layer).

Outputs -> benchmark_v2/leaderboard_matrix_v2/analysis_cell_power_v1/power.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CELLS_DIR = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"
ARCHIVE = MNT / "benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json"
OUT = MNT / "benchmark_v2/leaderboard_matrix_v2/analysis_cell_power_v1"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["gemorna", "hydrarna", "lamar_utr5te", "utr_insight", "utr_stcnet"]
TARGET_CELLS = ["M1_MRL_EVAL_ROW|5UTR", "M6_NDD5UTR_EVAL_ROW|5UTR", "S1_STABILITY_EVAL_ROW|3UTR", "S1_STABILITY_EVAL_ROW|5UTR"]
N_BOOT = 2000
SEED = 20260920
Z_CRIT = 1.959963984540054  # two-sided alpha=0.05


def cell_rho(rows):
    pred = np.array([r["delta_hat"] for r in rows], float)
    obs = np.array([r["observed"] for r in rows], float)
    return float(spearmanr(pred, obs).statistic)


def main():
    arch = json.load(open(ARCHIVE))["cells"]
    results = {
        "schema_version": "route_a_v3_cell_power_v1.v1",
        "task": "P1-3 (audit A9): per-cell bootstrap CI + MDE for 15 new-row cells (5 families x M1/M6/S1)",
        "date": str(_date.today()),
        "protocol": {
            "bootstrap": "source-group cluster bootstrap: resample source groups with replacement (cluster = source_id from predictions.jsonl), keep all rows within group, recompute task Spearman per iteration",
            "iterations": N_BOOT,
            "seed": SEED,
            "ci": "percentile 2.5/97.5",
            "mde": "Fisher z: MDE = tanh(1.959964/sqrt(n-3)) at alpha=0.05 two-sided, given observed n (row-level n; conservative for cluster data)",
            "discipline": "zero new forward passes (recompute from archived predictions.jsonl only); read-only; protected TEST reads = 0; no verdict changes - power-statement layer only",
        },
        "cells": {},
    }
    for fam in FAMILIES:
        path = CELLS_DIR / fam / "predictions.jsonl"
        by_cell = defaultdict(list)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d["cell"] in TARGET_CELLS:
                    by_cell[d["cell"]].append(d)
        for cell, rows in by_cell.items():
            rho = cell_rho(rows)
            arch_rho = arch.get(cell, {}).get(fam, {}).get("task_macro_spearman")
            groups = defaultdict(list)
            for r in rows:
                groups[r["source_id"]].append(r)
            gkeys = sorted(groups.keys())
            rng = random.Random(SEED)
            boots = []
            G = len(gkeys)
            for _ in range(N_BOOT):
                sample_rows = []
                for _ in range(G):
                    k = gkeys[rng.randrange(G)]
                    sample_rows.extend(groups[k])
                boots.append(cell_rho(sample_rows))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            n = len(rows)
            mde = float(np.tanh(Z_CRIT / np.sqrt(n - 3))) if n > 3 else float("nan")
            crosses_zero = bool(lo <= 0.0 <= hi)
            near_zero = abs(rho) < 0.05
            results["cells"][f"{cell}|{fam}"] = {
                "rho_point": rho,
                "rho_archived": arch_rho,
                "abs_delta_vs_archive": abs(rho - arch_rho) if arch_rho is not None else None,
                "n_rows": n,
                "n_source_groups": G,
                "bootstrap_ci95": [float(lo), float(hi)],
                "ci_width": float(hi - lo),
                "mde_0.05_two_sided": mde,
                "ci_crosses_zero": crosses_zero,
                "near_zero_point": near_zero,
            }
            print(f"{cell}|{fam}: rho={rho:+.4f} CI[{lo:+.4f},{hi:+.4f}] width={hi-lo:.4f} n={n} G={G} MDE={mde:.4f} cross0={crosses_zero}")

    results["interpretation_boundary_statement"] = (
        "CI-crossing-zero vs near-zero reading boundary: a cell with |rho| < MDE is power-limited - the data cannot distinguish the observed value from zero at alpha=0.05, so the near-zero read is consistent-with-zero, NOT proven-zero. "
        "The collective <0.09 pattern across all 15 cells is itself the consistency evidence (family-level agreement), but individual cells with CI width > 0.15 (M6 n=800 and S1 arms) should be read as power-limited. "
        "M1 cells (n=2,805) have the tightest CIs and still show |rho| <= 0.089 - the M1 near-zero band is not a power artifact. No frozen verdicts are changed by this layer."
    )
    widths = [v["ci_width"] for v in results["cells"].values()]
    results["summary"] = {
        "n_cells": len(results["cells"]),
        "ci_width_min": min(widths),
        "ci_width_median": float(np.median(widths)),
        "ci_width_max": max(widths),
        "all_15_cross_zero_or_power_limited": all(v["ci_crosses_zero"] or abs(v["rho_point"]) < v["mde_0.05_two_sided"] for v in results["cells"].values()),
        "max_abs_rho": max(abs(v["rho_point"]) for v in results["cells"].values()),
    }
    with open(OUT / "power.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("saved:", OUT / "power.json")


if __name__ == "__main__":
    main()
