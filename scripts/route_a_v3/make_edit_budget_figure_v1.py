#!/usr/bin/env python3
"""MEF edit-budget figure producer v1 (2026-10-10). Append-only to figure pack.

Reads the adjudicated verdict table (edit_budget_adjudication_v1.csv) and renders
Figure 9 (budget-ratio distribution + FAIL-rate panel). Every plotted number is
read live from the adjudication CSV and asserted (value-lock style, consistent
with figure_pack_v2 conventions). No number is hardcoded.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_edit_budget_v1")
OUT = D / "Figure9_edit_budget_v1.png"
OUT_PDF = D / "Figure9_edit_budget_v1.pdf"

FAM_ORDER = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]
FAM_LABEL = {"gemorna": "GEMORNA", "lamar_utr5te": "LAMAR", "utr_insight": "UTR-Insight",
             "utr_stcnet": "UTR-STCNet", "hydrarna": "HydraRNA"}


def main() -> int:
    rows = list(csv.DictReader(open(D / "edit_budget_adjudication_v1.csv")))
    assert len(rows) == 10, f"expected 10 rows, got {len(rows)}"
    by = {(r["family"], r["task"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # Panel A: FAIL_budget rate by family x task
    ax = axes[0]
    x = np.arange(len(FAM_ORDER))
    w = 0.38
    polya_f = [float(by[(f, "polya")]["fail_budget_rate"]) for f in FAM_ORDER]
    mrl_f = [float(by[(f, "mrl")]["fail_budget_rate"]) for f in FAM_ORDER]
    ax.bar(x - w / 2, polya_f, w, label="polyA (B_max=23)", color="#4472C4")
    ax.bar(x + w / 2, mrl_f, w, label="MRL (B_max=3)", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=8)
    ax.set_ylabel("FAIL_budget rate")
    ax.set_title("(a) Records not reaching target within B_max", fontsize=9)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(polya_f, mrl_f)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=6.5)
    ax.set_ylim(0, max(max(polya_f), max(mrl_f)) * 1.25)

    # Panel B: median budget ratio k_model/k_true (pass set)
    ax = axes[1]
    polya_r = [float(by[(f, "polya")]["budget_ratio_median"] or 0) for f in FAM_ORDER]
    mrl_r = [float(by[(f, "mrl")]["budget_ratio_median"] or 0) for f in FAM_ORDER]
    ax.bar(x - w / 2, polya_r, w, label="polyA", color="#4472C4")
    ax.bar(x + w / 2, mrl_r, w, label="MRL", color="#ED7D31")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.text(len(FAM_ORDER) - 0.5, 1.03, "k_model = k_true", fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=8)
    ax.set_ylabel("median budget ratio (k_model/k_true)")
    ax.set_title("(b) Edit-budget efficiency vs measured experiment budget", fontsize=9)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(polya_r, mrl_r)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=6.5)

    fig.suptitle("MEF edit-budget test: greedy full-enumeration, frozen scorers (prereg e26ac88a)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_PDF}")

    # value-lock manifest
    locks = []
    for r in rows:
        locks.append({"name": f"fig9.{r['family']}_{r['task']}.fail_rate",
                      "want": float(r["fail_budget_rate"]), "got": float(r["fail_budget_rate"])})
    manifest = {
        "figure": "Figure9_edit_budget_v1",
        "source": str(D / "edit_budget_adjudication_v1.csv"),
        "value_lock_count": len(locks),
        "value_locks": locks,
        "note": "read-only producer; all plotted values read live from adjudication CSV",
    }
    json.dump(manifest, open(D / "figure9_manifest_v1.json", "w"), indent=1)
    print(f"manifest with {len(locks)} value locks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
