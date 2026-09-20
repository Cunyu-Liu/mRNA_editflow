#!/usr/bin/env python3
"""Build benchmark_v2_matrix/matrix_summary_v1.json (14 families x task cells).

Task 2.2 + 3.5 summary table: 5 new families (this run, frozen-delta zero-tuning,
VALIDATION only + M1/M6/S1 new rows) + 9 existing frozen families (ALREADY_DONE -
archived values imported by reference, not re-run; append-only discipline).

Grid note: user brief says "14 families x 12 tasks"; the executed v2 grid is
14 families x 13 cells (9 existing VALIDATION tasks + M1/M6/S1 new rows where
S1 splits into 3UTR+5UTR arms). This summary keeps the 13-cell executed grid and
records the mapping explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
CELLS = MNT / "benchmark_v2/leaderboard_matrix_v2/cells"
EXP = MNT / "experiments"
OUT_DIR = MNT / "benchmark_v2/leaderboard_matrix_v2"

NEW_FAMILIES = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]

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

UT = json.loads((OUT_DIR / "family_unit_tests.json").read_text())["results"]


def new_family_cells():
    out = {}
    for fam in NEW_FAMILIES:
        d = json.loads((CELLS / fam / "cells.json").read_text())
        out[fam] = d
    return out


def _load(path):
    return json.loads(path.read_text())


def existing_family_cells():
    """Import archived VALIDATION values for the 9 frozen families by reference."""
    cells = {}

    def add(fam, cell, value, source, note=None, status="ALREADY_DONE"):
        rec = {"status": status, "spearman": value, "source": source}
        if note:
            rec["note"] = note
        cells.setdefault(fam, {})[cell] = rec

    mrl = _load(EXP / "analysis_frozen_delta_gse114002_20260903/frozen_delta_results.json")["results"]
    add("optimus5prime", "GSE114002|5UTR|MEAN_RIBOSOME_LOAD", mrl["optimus5prime"]["task_macro_spearman"],
        "experiments/analysis_frozen_delta_gse114002_20260903/frozen_delta_results.json")
    add("framepool", "GSE114002|5UTR|MEAN_RIBOSOME_LOAD", mrl["framepool"]["task_macro_spearman"],
        "experiments/analysis_frozen_delta_gse114002_20260903/frozen_delta_results.json")

    fc = _load(EXP / "analysis_frozen_delta_full_coverage_20260904/frozen_delta_results.json")["results"]
    FC_SRC = "experiments/analysis_frozen_delta_full_coverage_20260904/frozen_delta_results.json"
    for task, cell in [
        ("gse149487_rna", "GSE149487|5UTR|transcript_log2_totalrna_over_dna"),
        ("gse149487_te", "GSE149487|5UTR|te_log2_polysome_over_totalrna"),
        ("gse200304_te", "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"),
        ("gse186455", "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE"),
        ("polya_gse269595", "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS"),
        ("mprau_encsr854ruf", "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE"),
        ("half_life_5utr", "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES"),
        ("half_life_3utr", "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES"),
    ]:
        for model in ("utrlm", "rnafm"):
            r = fc[task][model]
            add(model, cell, r["task_macro_spearman"], FC_SRC)

    lamar = _load(EXP / "analysis_lamar_frozen_delta_20260909/frozen_delta_results.json")
    tn = lamar["metrics"]["task_numeric"]
    add("lamar_utr3deg", "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES", tn["3UTR|RNA_HALF_LIFE_MINUTES"]["spearman"],
        "experiments/analysis_lamar_frozen_delta_20260909/frozen_delta_results.json",
        note="LAMAR UTR3DegPred (3'UTR half-life, BEAS-2B) - separate row from lamar_utr5te")
    add("lamar_utr3deg", "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES", tn["5UTR|RNA_HALF_LIFE_MINUTES"]["spearman"],
        "experiments/analysis_lamar_frozen_delta_20260909/frozen_delta_results.json",
        note="3'UTR degradation model applied to 5'UTR arm (recorded as-is, archived run)")

    ribonn = _load(EXP / "analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")["results"]
    add("ribonn", "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY", ribonn["gse200304_te"]["task_macro_spearman"],
        "experiments/analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")
    add("ribonn", "GSE149487|5UTR|te_log2_polysome_over_totalrna", ribonn["gse149487_te"]["task_macro_spearman"],
        "experiments/analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")
    add("ribonn", "GSE149487|5UTR|transcript_log2_totalrna_over_dna", ribonn["gse149487_rna"]["task_macro_spearman"],
        "experiments/analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")
    add("ribonn", "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE", ribonn["gse186455"]["task_macro_spearman"],
        "experiments/analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")

    saluki = _load(EXP / "analysis_saluki_frozen_gse217518_20260903/frozen_delta_results.json")
    stn = saluki["metrics"]["task_numeric"]
    add("saluki", "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES", stn["3UTR|RNA_HALF_LIFE_MINUTES"]["spearman"],
        "experiments/analysis_saluki_frozen_gse217518_20260903/frozen_delta_results.json")
    add("saluki", "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES", stn["5UTR|RNA_HALF_LIFE_MINUTES"]["spearman"],
        "experiments/analysis_saluki_frozen_gse217518_20260903/frozen_delta_results.json")

    apa2 = _load(EXP / "analysis_aparent2_frozen_delta_20260906/result.json")
    add("aparent2", "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", apa2["task_macro_spearman"],
        "experiments/analysis_aparent2_frozen_delta_20260906/result.json")

    mb = _load(EXP / "analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")["tasks"]
    add("mrnabert_raw", "GSE114002|5UTR|MEAN_RIBOSOME_LOAD", mb["MRL"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS", mb["polyA"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE", mb["MPRAU"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE", mb["REFALT"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY", mb["TE200304"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE149487|5UTR|te_log2_polysome_over_totalrna", mb["TE149487"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE149487|5UTR|transcript_log2_totalrna_over_dna", mb["RNA149487"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES", mb["HL5"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    add("mrnabert_raw", "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES", mb["HL3"]["spearman"],
        "experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
    return cells


EXISTING_FAMILIES = [
    "optimus5prime", "framepool", "aparent2", "utrlm", "rnafm",
    "saluki", "ribonn", "mrnabert_raw", "lamar_utr3deg",
]


def main() -> int:
    new_cells = new_family_cells()
    existing_cells = existing_family_cells()

    grid = {}
    for cell in CELL_ORDER:
        row = {}
        for fam in NEW_FAMILIES:
            v = new_cells[fam]["matrix"].get(cell, {})
            if v.get("status") == "OK":
                row[fam] = {
                    "status": "DONE",
                    "spearman": v.get("task_macro_spearman"),
                    "n": v.get("n_evaluated"),
                    "unit_test_level": UT.get(fam, {}).get("status"),
                }
            else:
                row[fam] = {"status": v.get("status", "MISSING_CELL")}
        for fam in EXISTING_FAMILIES:
            if fam in existing_cells and cell in existing_cells[fam]:
                rec = existing_cells[fam][cell]
                row[fam] = {"status": "ALREADY_DONE", "spearman": rec["spearman"], "source": rec["source"], **({"note": rec["note"]} if rec.get("note") else {})}
            else:
                row[fam] = {
                    "status": "NOT_ARCHIVED_FOR_CELL",
                    "note": "existing frozen family: no archived VALIDATION number for this cell at the time of this summary (frozen-9 rows were task-scoped; M1/M6/S1 new rows post-date them)",
                }
        grid[cell] = row

    summary = {
        "schema_version": "benchmark_v2_matrix_summary.v1",
        "prereg": "docs/paper/benchmark_v2_matrix_row_prereg_v1.md + docs/paper/benchmark_v2_port_ledger_v1.md (license waived 2026-09-20)",
        "protocol": "frozen-delta zero-tuning; delta_hat = pred(cand) - pred(src); Task-1 evaluator K=10; VALIDATION only; protected TEST reads = 0",
        "grid_note": "user brief 12-task grid = 9 existing VALIDATION tasks + M1/M6/S1; executed grid = 13 cells (S1 splits into 3UTR and 5UTR arms)",
        "new_families": {
            fam: {
                "status": "DONE",
                "unit_test_level": ("EXACT" if fam == "utr_insight" else "SMOKE"),
                "unit_test": UT.get(fam, {}),
                "cells_ok": sum(1 for v in new_cells[fam]["matrix"].values() if v.get("status") == "OK"),
                "predictions_rows": sum(1 for _ in (CELLS / fam / "predictions.jsonl").open()),
                "cpu_fallback_used": new_cells[fam]["cpu_fallback_used"],
                "cuda_provenance": new_cells[fam]["cuda_provenance"],
                "family_meta": new_cells[fam]["family_meta"],
            }
            for fam in NEW_FAMILIES
        },
        "existing_families": {
            fam: {"status": "ALREADY_DONE", "archived_cells": sorted(existing_cells.get(fam, {}).keys())}
            for fam in EXISTING_FAMILIES
        },
        "matrix": grid,
        "counts": {
            "families_total": 14,
            "new_families": 5,
            "existing_families": 9,
            "cells": len(CELL_ORDER),
            "new_family_cells_ok": sum(1 for fam in NEW_FAMILIES for c in CELL_ORDER if grid[c][fam]["status"] == "DONE"),
        },
    }
    out = MNT / "benchmark_v2_matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix_summary_v1.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print("wrote", out / "matrix_summary_v1.json")
    print(json.dumps(summary["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
