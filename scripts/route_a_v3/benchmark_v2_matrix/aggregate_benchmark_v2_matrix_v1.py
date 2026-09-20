#!/usr/bin/env python3
"""Aggregate benchmark v2 matrix cells -> matrix_v2_results.json + summary.md.

Prereg: append-only, structured missing-cell annotation, per-cell adaptation
declaration, VALIDATION only. This script only READS the per-family cells.json
files and writes the leaderboard aggregates (no re-scoring).
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CELLS_ROOT = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"
OUT_ROOT = MNT / "benchmark_v2/leaderboard_matrix_v2"

FAMILIES = ["lamar_utr5te", "hydrarna", "gemorna", "utr_stcnet", "utr_insight"]

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

# frozen v1 leaderboard reference numbers (GSE114002 VALIDATION, Task-1
# evaluator, same split) for the M1/MRL key-readout comparison requested by
# the task brief; source: route2 v332 baseline matrix (frozen-9 classical).
FROZEN_V1_REFERENCE = {
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD": {
        "optimus_frozen": 0.3132,
    },
}

FAMILY_LABELS = {
    "lamar_utr5te": "LAMAR-UTR5TEPred",
    "hydrarna": "HydraRNA",
    "gemorna": "GEMORNA (5'/3' dual)",
    "utr_stcnet": "UTR-STCNet (MPRA-H)",
    "utr_insight": "UTR-Insight",
}


def main() -> int:
    results = {}
    for fam in FAMILIES:
        path = CELLS_ROOT / fam / "cells.json"
        if not path.exists():
            results[fam] = {"status": "NOT_RUN"}
            continue
        results[fam] = json.loads(path.read_text())

    matrix = {}
    for fam in FAMILIES:
        fam_res = results[fam]
        if fam_res.get("status") == "NOT_RUN":
            for cell in CELL_ORDER:
                matrix.setdefault(cell, {})[fam] = {"status": "NOT_RUN"}
            continue
        for cell, v in fam_res.get("matrix", {}).items():
            matrix.setdefault(cell, {})[fam] = v

    n_cells_total = len(CELL_ORDER)
    n_ok = n_port_fail = n_not_run = 0
    miss_stats = []
    for cell in CELL_ORDER:
        for fam in FAMILIES:
            v = matrix.get(cell, {}).get(fam, {"status": "MISSING_CELL"})
            st = v.get("status", "MISSING_CELL")
            if st == "OK":
                n_ok += 1
                if v.get("miss_count", 0) > 0:
                    miss_stats.append({"cell": cell, "family": fam, "miss_count": v["miss_count"], "miss_rate": v.get("miss_rate")})
            elif st == "PORT_FAILURE":
                n_port_fail += 1
            else:
                n_not_run += 1

    out = {
        "schema_version": "benchmark_v2_leaderboard_matrix.v1",
        "prereg": "docs/paper/benchmark_v2_matrix_row_prereg_v1.md (frozen) + docs/paper/benchmark_v2_port_ledger_v1.md (incl. license-waiver addendum 2026-09-20)",
        "evaluator": "Task-1 evaluate_route2_prediction_v1.evaluate (same instance, K=10, source-group within-source Spearman)",
        "split": "VALIDATION only for the 9 existing tasks; M1/M6/S1 new rows full (append-only, Task 3.5)",
        "protected_test_reads": 0,
        "cell_grid": CELL_ORDER,
        "families": FAMILY_LABELS,
        "cells": matrix,
        "family_meta": {fam: (results[fam].get("family_meta") if isinstance(results[fam], dict) else None) for fam in FAMILIES},
        "family_unit_tests": json.loads((OUT_ROOT / "family_unit_tests.json").read_text())["results"] if (OUT_ROOT / "family_unit_tests.json").exists() else None,
        "frozen_v1_reference": FROZEN_V1_REFERENCE,
        "counts": {
            "cells_per_family": n_cells_total,
            "families": len(FAMILIES),
            "total_cells": n_cells_total * len(FAMILIES),
            "OK": n_ok,
            "PORT_FAILURE": n_port_fail,
            "NOT_RUN_or_MISSING": n_not_run,
        },
        "miss_statistics": miss_stats,
    }
    (OUT_ROOT / "matrix_v2_results.json").write_text(json.dumps(out, indent=1, sort_keys=True))

    # ---------------- summary.md ----------------
    lines = []
    lines.append("# Benchmark v2 leaderboard matrix (Task 2.2 + 3.5) — VALIDATION only")
    lines.append("")
    lines.append("- Prereg: `benchmark_v2_matrix_row_prereg_v1.md` (frozen-delta zero-tuning, append-only, Task-1 evaluator)")
    lines.append("- Port ledger: `benchmark_v2_port_ledger_v1.md` (license waived by user decision 2026-09-20, journal batch 110)")
    lines.append("- Delta: `delta_hat = pred(cand) - pred(src)`; absolute predictions in per-family predictions.jsonl")
    lines.append("- Evaluator: Task-1 `evaluate_route2_prediction_v1.evaluate`, K=10, source-group within-source Spearman + task-macro Spearman")
    lines.append("- Splits: VALIDATION only (9 existing tasks); TEST untouched (0 reads). M1/M6/S1 are new append-only eval rows (Task 3.5)")
    lines.append("")
    lines.append("## Matrix (task-macro Spearman on VALIDATION delta)")
    lines.append("")
    header = "| cell | " + " | ".join(FAMILY_LABELS[f] for f in FAMILIES) + " |"
    sep = "|---" * (len(FAMILIES) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for cell in CELL_ORDER:
        vals = []
        for fam in FAMILIES:
            v = matrix.get(cell, {}).get(fam, {})
            st = v.get("status", "MISSING")
            if st == "OK":
                rho = v.get("task_macro_spearman")
                n = v.get("n_evaluated")
                high = "†" if v.get("HIGH_MISS") else ""
                vals.append(f"{rho:.4f}{high} (n={n})" if rho is not None else f"NONE (n={n})")
            elif st == "PORT_FAILURE":
                vals.append("PORT_FAILURE")
            else:
                vals.append(st)
        lines.append(f"| {cell} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("† = HIGH_MISS (miss rate > 15%, prereg §1.5)")
    lines.append("")
    lines.append("## Missing / failure structured statistics (prereg §1.5)")
    lines.append("")
    lines.append(f"- total cells: {n_cells_total * len(FAMILIES)} ({len(FAMILIES)} families x {n_cells_total} cells)")
    lines.append(f"- OK: {n_ok}; PORT_FAILURE: {n_port_fail}; NOT_RUN/MISSING: {n_not_run}")
    if miss_stats:
        lines.append("- per-cell miss records:")
        for m in miss_stats:
            lines.append(f"  - {m['cell']} / {m['family']}: miss={m['miss_count']} rate={m['miss_rate']:.3f}")
    else:
        lines.append("- no missing predictions (all rows scored; adapters map unknown chars deterministically, no rejections)")
    lines.append("")
    lines.append("## Key readouts (M1 MRL row, GSE232927 3-context, n=2805)")
    lines.append("")
    m1 = matrix.get("M1_MRL_EVAL_ROW|5UTR", {})
    lines.append("| family | M1 task-macro Spearman | n |")
    lines.append("|---|---|---|")
    for fam in FAMILIES:
        v = m1.get(fam, {})
        if v.get("status") == "OK":
            lines.append(f"| {FAMILY_LABELS[fam]} | {v.get('task_macro_spearman'):.4f} | {v.get('n_evaluated')} |")
        else:
            lines.append(f"| {FAMILY_LABELS[fam]} | {v.get('status')} | - |")
    lines.append(f"| frozen-Optimus (v1 reference, GSE114002 VALIDATION) | {FROZEN_V1_REFERENCE['GSE114002|5UTR|MEAN_RIBOSOME_LOAD']['optimus_frozen']:.4f} | 730 |")
    lines.append("")
    lines.append("## Per-family input adaptation declarations")
    lines.append("")
    for fam in FAMILIES:
        meta = results[fam].get("family_meta") if isinstance(results[fam], dict) else None
        if meta:
            lines.append(f"### {FAMILY_LABELS[fam]}")
            lines.append(f"- weights: `{meta.get('weights_path')}`")
            lines.append(f"- license: {meta.get('license')}")
            lines.append(f"- paradigm: {meta.get('paradigm')}")
            lines.append(f"- adaptation: {meta.get('input_adaptation')}")
            lines.append("")
    lines.append("## Unit test evidence (prereg §3.2)")
    lines.append("")
    ut = out.get("family_unit_tests")
    if ut:
        for fam, u in ut.items():
            lines.append(f"- {fam}: {u.get('status')} — {u.get('check')}")
    lines.append("")
    (OUT_ROOT / "matrix_v2_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_ROOT}/matrix_v2_results.json and matrix_v2_summary.md")
    print(json.dumps(out["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
