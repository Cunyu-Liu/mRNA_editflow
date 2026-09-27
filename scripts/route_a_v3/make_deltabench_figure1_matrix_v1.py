#!/usr/bin/env python3
"""DeltaBench Figure 1: 5 new families x 13 task cells frozen-delta matrix heatmap.

Read-only producer: consumes the frozen matrix results file and renders a
publication-quality heatmap (PNG 300dpi + PDF, vector). No numbers are
recomputed or altered: every cell is the archived `task_macro_spearman`.

Output: experiments/analysis_benchmark_v2_matrix_figure_v1/
        deltabench_matrix_heatmap_v1.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

RT2 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MATRIX = RT2 / "benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json"
OUT = RT2 / "experiments/analysis_benchmark_v2_matrix_figure_v1"

FAMILY_LABELS = {
    "gemorna": "GEMORNA (5'/3')",
    "hydrarna": "HydraRNA",
    "lamar_utr5te": "LAMAR-UTR5TEPred",
    "utr_insight": "UTR-Insight",
    "utr_stcnet": "UTR-STCNet (MPRA-H)",
}
TASK_ORDER = [
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD",
    "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
    "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
    "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
    "GSE149487|5UTR|te_log2_polysome_over_totalrna",
    "GSE149487|5UTR|transcript_log2_totalrna_over_dna",
    "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES",
    "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES",
    "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE",
    "M1_MRL_EVAL_ROW|5UTR",
    "M6_NDD5UTR_EVAL_ROW|5UTR",
    "S1_STABILITY_EVAL_ROW|5UTR",
    "S1_STABILITY_EVAL_ROW|3UTR",
]
TASK_LABELS = [
    "MRL\n(GSE114002)",
    "polyA\n(GSE269595)",
    "MPRAU\n(ENCSR854RUF)",
    "TE\n(GSE200304)",
    "PLUMAGE-TE\n(GSE149487)",
    "PLUMAGE-RNA\n(GSE149487)",
    "HL 5'UTR\n(GSE217518)",
    "HL 3'UTR\n(GSE217518)",
    "REF/ALT\n(GSE186455)",
    "M1 MRL\n(new)",
    "M6 NDD\n(new)",
    "S1 stab 5'\n(new)",
    "S1 stab 3'\n(new)",
]


def main() -> int:
    data = json.loads(MATRIX.read_text())
    cells = data["cells"]
    families = sorted({f for v in cells.values() for f in v})
    assert families == sorted(FAMILY_LABELS), families

    values = np.full((len(families), len(TASK_ORDER)), np.nan)
    for j, task in enumerate(TASK_ORDER):
        entry = cells.get(task)
        assert entry is not None, f"missing task {task}"
        for i, family in enumerate(families):
            value = entry.get(family, {}).get("task_macro_spearman")
            if value is not None:
                values[i, j] = float(value)

    fig, ax = plt.subplots(figsize=(13.2, 3.4), dpi=300)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.9)
    im = ax.imshow(values, cmap="RdYlBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_LABELS, fontsize=7.6)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([FAMILY_LABELS[f] for f in families], fontsize=8.4)

    for i in range(len(families)):
        for j in range(len(TASK_ORDER)):
            if np.isnan(values[i, j]):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=6.5, color="0.4")
            else:
                text_color = "white" if abs(values[i, j]) > 0.55 else "black"
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                        fontsize=7.0, color=text_color)

    # separator before the three append-only rows
    ax.axvline(8.5, color="black", linewidth=1.4, linestyle=(0, (4, 2)))
    ax.text(8.75, -0.85, "append-only new rows (frozen before scoring)", fontsize=7.2, color="black")
    ax.set_ylim(len(families) - 0.5, -1.15)
    ax.set_title(
        "DeltaBench frozen-\u0394 matrix: five newly ported families \u00d7 13 task cells "
        "(task-macro Spearman, VALIDATION only; protected TEST reads = 0)",
        fontsize=9.4, pad=16,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.set_label("task-macro Spearman (\u0394 caliber)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "deltabench_matrix_heatmap_v1.png", bbox_inches="tight")
    fig.savefig(OUT / "deltabench_matrix_heatmap_v1.pdf", bbox_inches="tight")

    manifest = {
        "schema_version": "deltabench_matrix_heatmap_v1",
        "source": str(MATRIX),
        "cells_rendered": int(np.sum(~np.isnan(values))),
        "families": families,
        "tasks": TASK_ORDER,
        "note": "values are archived task_macro_spearman; no recomputation or alteration",
        "protected_test_reads": data.get("protected_test_reads"),
    }
    (OUT / "deltabench_matrix_heatmap_v1_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True)
    )
    print(f"rendered {manifest['cells_rendered']}/65 cells -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())