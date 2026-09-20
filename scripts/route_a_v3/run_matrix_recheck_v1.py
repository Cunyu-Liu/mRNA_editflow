#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2.3: independent recheck of benchmark v2 matrix (5 families x 13 cells).

Recalculates task Spearman rho for every cell directly from per-family
predictions.jsonl (delta_hat vs observed), WITHOUT reading any metrics values,
then compares against the archived value (delta vs matrix_v2_results.json /
per-family cells.json) with |Delta| <= 1e-6 -> PASS.

Discipline: reads only existing artifacts; no GPU; protected reads = 0.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CELLS_ROOT = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"
OUT_DIR = MNT / "benchmark_v2/leaderboard_matrix_v2/analysis_matrix_recheck_v1"

FAMILIES = ["lamar_utr5te", "hydrarna", "gemorna", "utr_stcnet", "utr_insight"]
TOL = 1e-6

CELL_ORDER = [
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD",
    "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
    "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
    "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
    "GSE149487|5UTR|te_log2_polysome_over_totalrna",
    "GSE149487|5UTR|transcript_log2_totalrna_over_dna",
    "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE",
    "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES",
    "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES",
    "M1_MRL_EVAL_ROW|5UTR",
    "M6_NDD5UTR_EVAL_ROW|5UTR",
    "S1_STABILITY_EVAL_ROW|3UTR",
    "S1_STABILITY_EVAL_ROW|5UTR",
]


def recompute_cell(observed, delta_hat):
    """Mirror the runner's task-macro Spearman: single task group per cell, so
    task_macro_spearman == overall spearman over all finite observed rows."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(delta_hat, dtype=float)
    finite = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[finite], pred[finite]
    if len(obs) < 3 or obs.std() == 0.0 or pred.std() == 0.0:
        return None, int(finite.sum())
    value = spearmanr(obs, pred).statistic
    return (None if not math.isfinite(float(value)) else float(value)), int(finite.sum())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells_by_family = {}
    for fam in FAMILIES:
        per_cell = {}
        with (CELLS_ROOT / fam / "predictions.jsonl").open() as handle:
            for line in handle:
                rec = json.loads(line)
                per_cell.setdefault(rec["cell"], []).append((rec["observed"], rec["delta_hat"]))
        cells_by_family[fam] = per_cell

    results = {}
    n_pass = n_total = 0
    for cell in CELL_ORDER:
        results[cell] = {}
        for fam in FAMILIES:
            rows = cells_by_family[fam].get(cell, [])
            rho, n = recompute_cell([r[0] for r in rows], [r[1] for r in rows])
            entry = {
                "n_prediction_rows": len(rows),
                "n_evaluated_recomputed": n,
                "rho_recomputed": rho,
            }
            n_total += 1
            if rho is not None:
                entry["abs_delta"] = None  # filled below from archived value
            results[cell][fam] = entry

    # archived values (matrix_v2_results.json) — compared AFTER recomputation
    archived = json.loads((MNT / "benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json").read_text())
    deviants = []
    for cell in CELL_ORDER:
        for fam in FAMILIES:
            arch = archived["cells"][cell][fam].get("task_macro_spearman")
            entry = results[cell][fam]
            entry["rho_archived"] = arch
            rec = entry["rho_recomputed"]
            if rec is None or arch is None:
                entry["status"] = "FAIL_UNDEFINED"
                deviants.append({"cell": cell, "family": fam, "reason": "undefined rho", "recomputed": rec, "archived": arch})
            else:
                diff = abs(rec - arch)
                entry["abs_delta"] = diff
                if diff <= TOL:
                    entry["status"] = "PASS"
                    n_pass += 1
                else:
                    entry["status"] = "FAIL"
                    deviants.append({"cell": cell, "family": fam, "recomputed": rec, "archived": arch, "abs_delta": diff})

    out = {
        "schema_version": "benchmark_v2_matrix_recheck.v1",
        "task": "Task 2.3: independent cell-by-cell Spearman recheck from predictions.jsonl",
        "recompute_method": "scipy.stats.spearmanr(observed, delta_hat) over all finite rows of each cell (mirrors evaluate_route2_prediction_v1 single-task-group caliber; runner stored task_macro_spearman = overall for one-task cells)",
        "tolerance": TOL,
        "families": FAMILIES,
        "cell_grid": CELL_ORDER,
        "counts": {
            "cells_total": n_total,
            "pass": n_pass,
            "fail": n_total - n_pass,
            "pass_rate": round(n_pass / n_total, 6) if n_total else None,
        },
        "deviants": deviants,
        "per_cell": results,
        "protected_test_reads": 0,
    }
    (OUT_DIR / "recheck_results.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out["counts"], indent=1))
    print("deviants:", json.dumps(deviants, indent=1))
    print("saved:", OUT_DIR / "recheck_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
