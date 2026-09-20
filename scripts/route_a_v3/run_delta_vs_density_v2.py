#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2.4: delta-density v2 — held-out prediction test per frozen framework
docs/paper/delta_density_prereg_framework_v1.md.

Frozen fit: 25 external rows (analysis_delta_vs_density_20260915), OLS on
log10(density) -> rho. New 65 matrix-v2 cells are held-out: predicted, never
refit. Verdict gate: >=70% of new cells within |drho| <= 0.10; otherwise FAIL
-> downgrade clause. NO row selection, NO post-hoc band adjustment.

Density calibers (recorded, not tuned):
- 9 existing tasks: same within-source candidate density table as the 25-row
  fit set (frozen TRAIN caliber; GSE149487 two cells = same-caliber recomputed
  TRAIN candidates-per-source, see density_caliber.json).
- M1/M6/S1: their own eval-row pairing densities from row_manifest.json.

Exploratory (NOT a gate): group in-band rate by whether the eval task's
endpoint domain is inside the model family's training corpus domain.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OLD_DIR = MNT / "experiments/analysis_delta_vs_density_20260915"
MATRIX = MNT / "benchmark_v2/leaderboard_matrix_v2"
OUT_DIR = MNT / "experiments/analysis_delta_vs_density_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOL = 0.10
GATE = 0.70
MIN_SAMPLE = 10

FAMILIES = ["lamar_utr5te", "hydrarna", "gemorna", "utr_stcnet", "utr_insight"]
FAMILY_LABELS = {
    "lamar_utr5te": "LAMAR-UTR5TEPred",
    "hydrarna": "HydraRNA",
    "gemorna": "GEMORNA",
    "utr_stcnet": "UTR-STCNet",
    "utr_insight": "UTR-Insight",
}

DENSITY_EXISTING = {
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD": 4.9,
    "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS": 126.7,
    "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE": 6.1,
    "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY": 1.0,
    "GSE149487|5UTR|te_log2_polysome_over_totalrna": 1.0,
    "GSE149487|5UTR|transcript_log2_totalrna_over_dna": 1.0,
    "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE": 2.1,
    "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES": 1.7,
    "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES": 1.6,
}
DENSITY_NEW_ROWS = {
    "M1_MRL_EVAL_ROW|5UTR": 1.0014,
    "M6_NDD5UTR_EVAL_ROW|5UTR": 1.5717,
    "S1_STABILITY_EVAL_ROW|3UTR": 3.6682,
    "S1_STABILITY_EVAL_ROW|5UTR": 3.6682,
}
DENSITY_SOURCES = {
    **{k: "25-row fit set density table (frozen TRAIN cand/source, analysis_delta_vs_density_20260915)"
       for k in ["GSE114002|5UTR|MEAN_RIBOSOME_LOAD",
                 "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
                 "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
                 "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
                 "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE",
                 "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES",
                 "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES"]},
    **{k: "same-caliber recomputed: GSE149487 TRAIN has no source groups (0 candidates/source); 48 pair records, density 1.0 = 1 candidate per source record"
       for k in ["GSE149487|5UTR|te_log2_polysome_over_totalrna",
                 "GSE149487|5UTR|transcript_log2_totalrna_over_dna"]},
    "M1_MRL_EVAL_ROW|5UTR": "row_manifest.json cand_per_source_density.mean (eval-row pairing caliber, declared in row_definition.md)",
    "M6_NDD5UTR_EVAL_ROW|5UTR": "row_manifest.json cand_per_source_density.mean (eval-row pairing caliber)",
    "S1_STABILITY_EVAL_ROW|3UTR": "row_manifest.json cand_per_source_density.mean (eval-row pairing caliber)",
    "S1_STABILITY_EVAL_ROW|5UTR": "row_manifest.json cand_per_source_density.mean (eval-row pairing caliber)",
}

# ---- task-domain-in-training-corpus map (exploratory grouping only) ----
# Endpoint domains per family (from port ledger + official repo preprocessing):
# MRL-domain = human designed-library MPRA (GSM3130443-style 5'UTR MRL);
# TE-domain = mammalian 5'UTR TE; polyA = 3'UTR proximal polyA usage.
# HydraRNA/GEMORNA-3'head/GEMORNA-5'head: corpus-specific as declared.
TASK_DOMAIN = {
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD": "MRL",
    "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS": "polyA",
    "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE": "MPRAU_skew",
    "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY": "TE",
    "GSE149487|5UTR|te_log2_polysome_over_totalrna": "TE",
    "GSE149487|5UTR|transcript_log2_totalrna_over_dna": "RNA_abundance",
    "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE": "MPRA_refalt",
    "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES": "stability",
    "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES": "stability",
    "M1_MRL_EVAL_ROW|5UTR": "MRL",
    "M6_NDD5UTR_EVAL_ROW|5UTR": "TE",
    "S1_STABILITY_EVAL_ROW|3UTR": "stability",
    "S1_STABILITY_EVAL_ROW|5UTR": "stability",
}
IN_DOMAIN = {
    # family -> endpoint domains numerically supervised in the family's
    # training corpus, judged from official training data provenance
    # (port ledger + official repo preprocessing):
    # UTR-STCNet MPRA-H = GSM3130443 human_utrs+snv (MRL);
    # UTR-Insight = GSM3130435 50-nt random library (MRL);
    # GEMORNA = 5'UTR human MPRA MRL + 3'UTR MRL-scale corpus (MRL);
    # LAMAR-UTR5TEPred = mammalian 5'UTR TE fine-tune (TE; GSE200304 is
    # 3UTR-TE -> endpoint family matched, region mismatch noted);
    # HydraRNA = full-length RNA LM pretraining, no numeric UTR-task corpus.
    "utr_stcnet": {"MRL"},
    "utr_insight": {"MRL"},
    "gemorna": {"MRL"},
    "lamar_utr5te": {"TE"},
    "hydrarna": set(),
}


def main():
    # 1) frozen 25-row fit
    old = json.loads((OLD_DIR / "delta_vs_density_data.json").read_text())
    old_rows = old["rows"]
    xs = np.log10([old["density_table"][r["task"]] for r in old_rows])
    ys = np.asarray([r["rho"] for r in old_rows], dtype=float)
    coef = np.polyfit(xs, ys, 1)
    r_check = float(np.corrcoef(xs, ys)[0, 1])

    # 2) 65 held-out cells
    matrix = json.loads((MATRIX / "matrix_v2_results.json").read_text())
    new_cells = []
    for cell in matrix["cell_grid"]:
        for fam in FAMILIES:
            v = matrix["cells"][cell][fam]
            if v.get("status") != "OK":
                continue
            rho = v.get("task_macro_spearman")
            if rho is None:
                continue
            density = DENSITY_EXISTING.get(cell, DENSITY_NEW_ROWS.get(cell))
            x = np.log10(density)
            pred = float(np.polyval(coef, x))
            diff = rho - pred
            new_cells.append({
                "family": fam,
                "cell": cell,
                "rho_observed": float(rho),
                "rho_predicted": round(pred, 6),
                "delta_rho": round(float(diff), 6),
                "abs_delta_rho": round(abs(float(diff)), 6),
                "in_band": bool(abs(diff) <= TOL),
                "density": density,
                "task_domain": TASK_DOMAIN[cell],
                "domain_in_family_training_corpus": TASK_DOMAIN[cell] in IN_DOMAIN[fam],
            })

    n = len(new_cells)
    n_in = sum(1 for c in new_cells if c["in_band"])
    in_rate = n_in / n if n else None
    if n < MIN_SAMPLE:
        verdict = "DEFERRED (sample below prereg minimum 10)"
    else:
        verdict = "PASS" if in_rate >= GATE else "FAIL"
    downgrade = verdict == "FAIL"

    # 3) exploratory domain grouping
    in_dom = [c for c in new_cells if c["domain_in_family_training_corpus"]]
    out_dom = [c for c in new_cells if not c["domain_in_family_training_corpus"]]
    above = [c for c in new_cells if c["delta_rho"] > TOL]
    below = [c for c in new_cells if c["delta_rho"] < -TOL]
    exploratory = {
        "note": "EXPLORATORY — not a preregistered gate; grouping by whether the eval task's endpoint domain is numerically inside the family's training corpus (source: port ledger + official repo preprocessing provenance)",
        "in_domain": {
            "n_cells": len(in_dom),
            "n_in_band": sum(1 for c in in_dom if c["in_band"]),
            "in_band_rate": round(sum(1 for c in in_dom if c["in_band"]) / len(in_dom), 4) if in_dom else None,
            "cells": [f"{c['family']}|{c['cell']}" for c in in_dom],
        },
        "out_of_domain": {
            "n_cells": len(out_dom),
            "n_in_band": sum(1 for c in out_dom if c["in_band"]),
            "in_band_rate": round(sum(1 for c in out_dom if c["in_band"]) / len(out_dom), 4) if out_dom else None,
        },
        "deviation_direction_decomposition": {
             "above_band": {
                 "n_cells": len(above),
                 "n_in_domain": sum(1 for c in above if c["domain_in_family_training_corpus"]),
                 "mean_rho": round(sum(c["rho_observed"] for c in above) / len(above), 4) if above else None,
                 "cells": [f"{c['family']}|{c['cell']}|rho={c['rho_observed']:+.4f}" for c in sorted(above, key=lambda c: -c["delta_rho"])],
             },
             "below_band": {
                 "n_cells": len(below),
                 "n_in_domain": sum(1 for c in below if c["domain_in_family_training_corpus"]),
                 "mean_rho": round(sum(c["rho_observed"] for c in below) / len(below), 4) if below else None,
                 "cells": [f"{c['family']}|{c['cell']}|rho={c['rho_observed']:+.4f}" for c in sorted(below, key=lambda c: c["delta_rho"])],
             },
         },
    }
    by_cell = {}
    for c in new_cells:
        by_cell.setdefault(c["cell"], []).append(c)
    exploratory["per_cell_in_band_count"] = {
        cell: f"{sum(1 for c in cs if c['in_band'])}/{len(cs)}" for cell, cs in sorted(by_cell.items())
    }
    by_fam = {}
    for c in new_cells:
        by_fam.setdefault(c["family"], []).append(c)
    exploratory["per_family_in_band_count"] = {
        fam: f"{sum(1 for c in cs if c['in_band'])}/{len(cs)}" for fam, cs in sorted(by_fam.items())
    }

    out = {
        "schema_version": "delta_vs_density_heldout_v2.v1",
        "task": "Task 2.4: delta-density v2 held-out prediction test (frozen framework delta_density_prereg_framework_v1.md)",
        "framework": {
            "source": "docs/paper/delta_density_prereg_framework_v1.md (frozen)",
            "tolerance_band": TOL,
            "pass_gate": GATE,
            "min_sample": MIN_SAMPLE,
            "fit": "frozen 25 external rows, OLS on log10(density) -> rho; new rows never refit",
        },
        "fit": {
            "n_rows": len(old_rows),
            "slope_log10": round(float(coef[0]), 6),
            "intercept": round(float(coef[1]), 6),
            "pearson_log10_r_recomputed": round(r_check, 4),
            "pearson_log10_r_archived": old["stats"]["pearson_log10_density_r"],
            "slope_10x_density": round(float(coef[0]), 4),
        },
        "heldout": {
            "n_cells": n,
            "n_in_band": n_in,
            "in_band_rate": round(in_rate, 4) if in_rate is not None else None,
            "tolerance_band": TOL,
            "pass_gate": GATE,
            "verdict": verdict,
            "downgrade_triggered": downgrade,
            "downgrade_action": "per framework clause (c): density single-factor claim downgraded to task-specific phenomenon; wording updated; no row removal / band widening / gate lowering permitted" if downgrade else None,
        },
        "density_caliber": {cell: {"density": DENSITY_EXISTING.get(cell, DENSITY_NEW_ROWS.get(cell)), "source": DENSITY_SOURCES[cell]} for cell in list(DENSITY_EXISTING) + list(DENSITY_NEW_ROWS)},
        "cells": new_cells,
        "exploratory_domain_grouping": exploratory,
        "protected_test_reads": 0,
    }
    (OUT_DIR / "delta_vs_density_v2_results.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False))
    (OUT_DIR / "delta_vs_density_v2_cells.json").write_text(
        json.dumps(new_cells, indent=1, ensure_ascii=False))

    # 4) figure: 25 old rows + 65 new cells + prediction band
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 11, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    xx = np.logspace(np.log10(0.85), np.log10(200), 200)
    yy = np.polyval(coef, np.log10(xx))
    ax.plot(xx, yy, ls="--", lw=1.4, color="#5B8DB8",
            label=f"frozen fit (25 rows): slope={coef[0]:.3f}/decade, r={r_check:.2f}")
    ax.fill_between(xx, yy - TOL, yy + TOL, color="#5B8DB8", alpha=0.16,
                    label=f"prereg tolerance band ±{TOL}")
    old_x = [old["density_table"][r["task"]] for r in old_rows]
    old_y = [r["rho"] for r in old_rows]
    ax.scatter(old_x, old_y, s=46, marker="o", color="#8A8A8A", alpha=0.85,
               edgecolors="black", linewidths=0.4, zorder=3,
               label="fit set: 25 external frozen-Δ rows (2026-09-15)")
    FAM_COLOR = {"lamar_utr5te": "#E67E22", "hydrarna": "#9B59B6", "gemorna": "#16A085",
                 "utr_stcnet": "#C0392B", "utr_insight": "#2980B9"}
    FAM_MARKER = {"lamar_utr5te": "s", "hydrarna": "D", "gemorna": "^",
                  "utr_stcnet": "*", "utr_insight": "v"}
    for fam in FAMILIES:
        fc = [c for c in new_cells if c["family"] == fam]
        if not fc:
            continue
        ax.scatter([c["density"] for c in fc], [c["rho_observed"] for c in fc],
                   s=120 if fam == "utr_stcnet" else 64,
                   marker=FAM_MARKER[fam], color=FAM_COLOR[fam], alpha=0.9,
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"{FAMILY_LABELS[fam]} (held-out)")
    # annotate the two most extreme deviants
    devs = sorted(new_cells, key=lambda c: -c["abs_delta_rho"])[:4]
    for c in devs:
        ax.annotate(f"{FAMILY_LABELS[c['family']][:8]} {c['rho_observed']:+.2f}",
                    xy=(c["density"], c["rho_observed"]),
                    xytext=(c["density"] * 0.55, c["rho_observed"] - 0.13),
                    fontsize=8.5, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7))
    ax.set_xscale("log")
    ax.set_xlim(0.85, 220)
    ax.set_ylim(-0.55, 1.0)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Within-source candidate density (candidates per source, log scale)")
    ax.set_ylabel("Delta-prediction Spearman ρ")
    ax.set_title(f"Delta-density v2 held-out test: 65 benchmark-v2 matrix cells vs frozen 25-row fit\n"
                 f"verdict: {verdict} — {n_in}/{n} in band ({in_rate:.1%}) | prereg gate ≥{GATE:.0%} within ±{TOL}",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=8.6, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_vs_density_v2_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "delta_vs_density_v2_scatter.pdf", bbox_inches="tight")

    print(json.dumps({
        "verdict": verdict,
        "n_cells": n,
        "n_in_band": n_in,
        "in_band_rate": round(in_rate, 4),
        "fit": out["fit"],
        "exploratory_in_domain_rate": exploratory["in_domain"]["in_band_rate"],
        "exploratory_out_domain_rate": exploratory["out_of_domain"]["in_band_rate"],
    }, indent=1))
    print("saved:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
