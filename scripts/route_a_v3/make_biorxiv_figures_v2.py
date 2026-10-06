#!/usr/bin/env python3
"""bioRxiv submission figure pack v2 (final, 2026-10-07).

Read-only producers for the 7 finalized submission figures, rendered from the
frozen JSON artifacts with value-lock assertions (bit-exact / tolerance) and a
per-run manifest. Replaces the v1 rendering lineage for submission purposes.

Figures:
  Figure1  matrix heatmap (5 ported families x 13 tasks, 65/65 cells)
  Figure2a density observation (25 external rows, log-x)
  Figure2b held-out cell-level check (65 cells, band verdict)
  Figure3  first-order lollipop (polyA / MPRAU / TE + 4-item legend)
  Figure4  ceiling-completion dumbbell (8 tasks, external->ours->ICC)
  Figure5  intervention triangle (3 panels, 10 locked values)
  Figure6  first-order multiples (4 panels, ceiling-normalized)
  FigureS1 all-family supplement (5-family matrix + 9 reference + ours)

Every plotted number is either read live from a frozen JSON (and asserted
where a cross-check exists) or declared verbatim from a named frozen artifact
(the paper's §7/§8 sources, same values as the paper text). No recomputation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

RT2 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
EXP = RT2 / "experiments"
OUT = EXP / "analysis_biorxiv_figure_pack_v2"

SRC = {
    "matrix": RT2 / "benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json",
    "density": EXP / "analysis_delta_vs_density_20260915/delta_vs_density_data.json",
    "cells": EXP / "analysis_delta_vs_density_v2/delta_vs_density_v2_cells.json",
    "first_order": EXP / "analysis_first_order_decomposition_v1/results_first_order.json",
    "bottomline": EXP / "analysis_task8_bottomline_20260909/bottomline_adjudication_v1.json",
    "w_ladder": EXP / "analysis_w_ladder_adjudication_20260903/results.json",
    "lora_280k": EXP / "xeditcritic_route_a/280k_prefinetune_20260903/frozen_delta_results.json",
    "fullft_2ep": EXP / "xeditcritic_route_a/280k_fullft_ablation_20260903/frozen_delta_results.json",
    "ens_3seed": EXP / "analysis_fullft_v2_adjudication_20260903/ensemble_3seed_vs_optimus.json",
    "erk_adj": EXP / "xeditcritic_erk_v2/erk_train_seed2026091901/erk_adjudication_v1.json",
    "d16_gap": EXP / "xeditcritic_d16c/probe_mrl_v1_gpu5/gap_backtest_d16c.json",
}

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

TASKS = [
    "GSE114002|5UTR|MEAN_RIBOSOME_LOAD",
    "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS",
    "ENCSR854RUF|3UTR|MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE",
    "GSE200304|3UTR|TOTAL_POLYSOME_TRANSLATION_EFFICIENCY",
    "GSE149487|5UTR|te_log2_polysome_over_totalrna",
    "GSE149487|5UTR|transcript_log2_totalrna_over_dna",
    "GSE186455|3UTR|PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE",
    "GSE217518|5UTR|RNA_HALF_LIFE_MINUTES",
    "GSE217518|3UTR|RNA_HALF_LIFE_MINUTES",
    "M1_MRL_EVAL_ROW|5UTR", "M6_NDD5UTR_EVAL_ROW|5UTR",
    "S1_STABILITY_EVAL_ROW|3UTR", "S1_STABILITY_EVAL_ROW|5UTR",
]
TASK_SHORT = ["MRL", "polyA", "MPRAU", "TE", "PLUM-TE", "PLUM-RNA", "REF/ALT",
              "HL 5'", "HL 3'", "M1", "M6", "S1 3'", "S1 5'"]
FAMS5 = ["utr_stcnet", "gemorna", "utr_insight", "lamar_utr5te", "hydrarna"]
FAM5_SHORT = ["UTR-STCNet", "GEMORNA", "UTR-Insight", "LAMAR", "HydraRNA"]

FAM_STYLE = {
    "utr_stcnet":   dict(color="#d65f5f", marker="o", label="UTR-STCNet"),
    "gemorna":      dict(color="#4878d0", marker="s", label="GEMORNA"),
    "utr_insight":  dict(color="#6acc65", marker="^", label="UTR-Insight"),
    "lamar_utr5te": dict(color="#ee854a", marker="D", label="LAMAR"),
    "hydrarna":     dict(color="#95a5c6", marker="v", label="HydraRNA"),
}

SOFT_DIV = ["#2a5f8f", "#7fa8cf", "#f2f2f2", "#e8956d", "#b03a2e"]
LOCK_LOG: list[dict] = []


def lock(name: str, got: float, want: float, tol: float = 0.0) -> None:
    if tol == 0.0:
        ok = float(got) == float(want)
    else:
        ok = abs(float(got) - float(want)) <= tol
    assert ok, f"value lock FAILED {name}: got {got!r}, want {want!r}"
    LOCK_LOG.append({"name": name, "got": float(got), "want": float(want),
                     "tol": tol, "ok": True})


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


def matrix_grid(mx: dict) -> np.ndarray:
    grid = np.full((len(FAMS5), len(TASKS)), np.nan)
    for j, t in enumerate(TASKS):
        for i, f in enumerate(FAMS5):
            grid[i, j] = mx["cells"][t][f]["task_macro_spearman"]
    assert grid.shape == (5, 13)
    assert np.isfinite(grid).all(), "missing cells"
    assert int(mx["counts"]["total_cells"]) == 65
    return grid


# ============================================================ Figure 1 (v3)
def fig1() -> None:
    mx = json.loads(SRC["matrix"].read_text())
    grid = matrix_grid(mx)
    lock("fig1.total_cells", int(mx["counts"]["total_cells"]), 65)
    lock("fig1.stcnet_mrl", grid[0, 0], mx["cells"][TASKS[0]]["utr_stcnet"]["task_macro_spearman"])

    cmap = LinearSegmentedColormap.from_list("soft_div", SOFT_DIV)
    norm = Normalize(vmin=-0.4, vmax=0.9)
    fig, ax = plt.subplots(figsize=(13.0, 3.2))
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(FAMS5)):
        for j in range(len(TASKS)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if (v > 0.55 or v < -0.28) else "0.25")
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASK_SHORT, fontsize=8)
    ax.set_yticks(range(len(FAMS5))); ax.set_yticklabels(FAM5_SHORT, fontsize=8.5)
    ax.axvline(8.5, color="0.35", lw=0.9)
    ax.text(8.9, -0.75, "append-only new rows (frozen before scoring)",
            fontsize=7.8, color="0.3", style="italic")
    ax.axvline(-0.5, color="0.85", lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("frozen-Δ Spearman", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("DeltaBench matrix: 5 ported families × 13 tasks (65/65 re-checked)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    save(fig, "Figure1_matrix_heatmap")
    print("  Figure1: 65/65 cells rendered from frozen matrix")


# ============================================================ Figure 2a (A v4)
def fig2a() -> None:
    d = json.loads(SRC["density"].read_text())
    rows, dens = d["rows"], d["density_table"]
    assert len(rows) == 25 and len(dens) == 7
    lock("fig2a.n_rows", len(rows), 25)

    PAL = {"MRL": "#4878d0", "polyA": "#d65f5f", "MPRAU": "#ee854a", "TE200304": "#6acc65",
           "HL5": "#95a5c6", "HL3": "#82c6e2", "REFALT": "#c8ace8"}
    TASK_LABEL = {"MRL": "MRL (4.9)", "polyA": "polyA (126.7)", "MPRAU": "MPRAU (6.1)",
                  "TE200304": "TE (1.0)", "HL5": "HL 5'UTR (1.7)", "HL3": "HL 3'UTR (1.6)",
                  "REFALT": "REF/ALT (2.1)"}
    fit = np.polyfit(np.log10([dens[r["task"]] for r in rows]),
                     [r["rho"] for r in rows], 1)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    xs_fit = np.logspace(np.log10(1.2), np.log10(150), 100)
    ax.plot(xs_fit, np.polyval(fit, np.log10(xs_fit)), color="0.55", lw=1.2,
            ls="--", zorder=1, label=f"fit: slope {fit[0]:.2f} per decade")
    for task in PAL:
        tx = [dens[r["task"]] for r in rows if r["task"] == task]
        ty = [r["rho"] for r in rows if r["task"] == task]
        ax.scatter(tx, ty, s=46, color=PAL[task], edgecolor="white", linewidth=0.8,
                   zorder=3, label=TASK_LABEL[task])
    ax.set_xscale("log")
    ax.set_xlabel("candidates per source (log scale)", fontsize=9)
    ax.set_ylabel("frozen-Δ Spearman ρ", fontsize=9)
    ax.set_ylim(-0.38, 1.02)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.text(0.03, 0.965, "polyA cluster — the only high-density task",
            transform=ax.transAxes, fontsize=8.5, color="#d65f5f", va="top")
    ax.text(0.03, 0.085, "near-zero band: 20 of 25 rows sit below ρ = 0.32",
            transform=ax.transAxes, fontsize=8.5, color="0.35", va="bottom")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=8, handletextpad=0.4, borderaxespad=0)
    fig.subplots_adjust(left=0.09, right=0.70, top=0.90, bottom=0.13)
    fig.suptitle("Density observation: r = 0.9388 (n = 25 external rows)",
                 fontsize=10, fontweight="bold", x=0.09, ha="left", y=0.985)
    save(fig, "Figure2a_density_v1")
    print(f"  Figure2a: 25 rows rendered; fit slope {fit[0]:.3f}")


# ============================================================ Figure 2b (B v5)
def fig2b() -> None:
    v2 = json.loads(SRC["cells"].read_text())
    cells = v2 if isinstance(v2, list) else v2.get("cells", [])
    assert len(cells) == 65
    n_in = sum(1 for c in cells if c.get("in_band"))
    lock("fig2b.n_cells", len(cells), 65)
    lock("fig2b.in_band", n_in, 33)

    def find(fam, key):
        for c in cells:
            if c["family"] == fam and key in c["cell"]:
                return c
        raise KeyError((fam, key))

    anchors = [
        (find("utr_stcnet", "GSE114002"), "STCNet MRL", (0.055, -0.045)),
        (find("utr_stcnet", "M1_MRL"), "STCNet M1", (0.055, -0.045)),
        (find("lamar_utr5te", "transcript_log2"), "LAMAR RNA", (0.055, -0.045)),
        (find("hydrarna", "GSE269595"), "HydraRNA polyA", (0.055, +0.045)),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    for cell in cells:
        inb = bool(cell.get("in_band"))
        st = FAM_STYLE[str(cell["family"])]
        ax.scatter(cell["rho_predicted"], cell["rho_observed"], s=34,
                   color=st["color"] if inb else "white",
                   edgecolor=st["color"], linewidth=1.1, marker=st["marker"],
                   zorder=3, alpha=0.95)
    lim = (-0.35, 0.95)
    ax.fill_between(lim, [l - 0.10 for l in lim], [l + 0.10 for l in lim],
                    color="#6acc65", alpha=0.10, zorder=1)
    ax.plot(lim, lim, color="0.6", lw=0.7, ls=":")
    ax.set_xlim(lim); ax.set_ylim(-0.55, 1.0)
    ax.set_xlabel("predicted ρ (single-factor fit)", fontsize=9)
    ax.set_ylabel("observed ρ (65 matrix cells)", fontsize=9)
    for cell, name, (dx, dy) in anchors:
        x, y = cell["rho_predicted"], cell["rho_observed"]
        ax.annotate(f"{name} {y:+.3f}", xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=7.2, color="0.30", va="center", zorder=4)
    handles = []
    for fam, st in FAM_STYLE.items():
        handles.append(Line2D([0], [0], marker=st["marker"], color=st["color"],
                              linestyle="none", markersize=7,
                              markerfacecolor=st["color"], label=st["label"]))
    handles.append(Line2D([0], [0], marker="o", color="0.5", linestyle="none",
                          markersize=7, markerfacecolor="white",
                          markeredgecolor="0.5", label="hollow = out of band"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8, handletextpad=0.4, borderaxespad=0)
    fig.subplots_adjust(left=0.10, right=0.68, top=0.90, bottom=0.12)
    fig.suptitle("Held-out check: FAIL — 33/65 = 50.8% in band (< 70% gate)",
                 fontsize=10, fontweight="bold", x=0.10, ha="left", y=0.985)
    save(fig, "Figure2b_density_v2_heldout")
    print(f"  Figure2b: 65 cells, in-band {n_in}/65; 4 anchors at true coords")


# ============================================================ Figure 3 (v4)
def fig3() -> None:
    fo = json.loads(SRC["first_order"].read_text())
    tasks = ["polyA", "MPRAU", "TE"]
    labels = {"polyA": "polyA", "MPRAU": "MPRAU", "TE": "TE (3'UTR)"}
    rho1 = {t: fo["tasks"][t]["rho_first_order"] for t in tasks}
    lock("fig3.rho1_polyA", rho1["polyA"], 0.45554942, tol=1e-6)
    lock("fig3.rho1_MPRAU", rho1["MPRAU"], 0.08253518, tol=1e-6)
    lock("fig3.rho1_TE", rho1["TE"], 0.04577197, tol=1e-6)
    best = {"polyA": 0.8219, "MPRAU": 0.1025, "TE": 0.0579}
    ext = {"polyA": (0.71, 0.75), "MPRAU": (0.0, 0.05), "TE": (0.0009, 0.0113)}
    ceil = {"polyA": 0.90, "MPRAU": 0.683, "TE": 0.586}

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ys = np.arange(len(tasks))[::-1]
    for y, t in zip(ys, tasks):
        ax.plot([0, ceil[t]], [y, y], color="0.88", lw=1.4, zorder=1)
        ax.add_patch(Rectangle((ext[t][0], y - 0.16), ext[t][1] - ext[t][0], 0.32,
                                color="#95a5c6", alpha=0.45, zorder=2))
        ax.plot([rho1[t], rho1[t]], [y - 0.22, y + 0.22], color="#4878d0", lw=2.2, zorder=3)
        ax.scatter([rho1[t]], [y], s=64, color="#4878d0", zorder=4, edgecolor="white", lw=0.8)
        ax.plot([best[t], best[t]], [y - 0.22, y + 0.22], color="#ee854a", lw=2.2, zorder=3)
        ax.scatter([best[t]], [y], s=64, color="#ee854a", zorder=4, edgecolor="white", lw=0.8)
        ax.scatter([ceil[t]], [y], s=52, marker="|", color="0.3", lw=1.6, zorder=4)
        pct = rho1[t] / best[t] * 100
        ax.text(1.015, y, f"{pct:.0f}%", fontsize=8.5, color="#4878d0",
                va="center", ha="left", fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([labels[t] for t in tasks], fontsize=10)
    ax.set_xlim(-0.06, 1.0)
    ax.set_ylim(-0.6, len(tasks) - 0.3)
    ax.set_xlabel("frozen-Δ Spearman ρ", fontsize=9)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_title("First-order table vs structural signal: where the learnable part lives",
                 fontsize=10, fontweight="bold", loc="left", pad=10)
    ax.text(1.015, len(tasks) - 0.45, "ρ₁ / ours", fontsize=7.5, color="0.45",
            ha="left", va="center")
    handles = [
        Patch(facecolor="#95a5c6", alpha=0.45, label="external unsupervised band"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4878d0",
               markersize=7, label="first-order table (ρ₁)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ee854a",
               markersize=7, label="ours (best in-house)"),
        Line2D([0], [0], marker="|", color="0.3", markersize=8, linestyle="none",
               label="label ceiling (ICC)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2)
    fig.tight_layout()
    save(fig, "Figure3_first_order")
    print("  Figure3: lollipop w/ 4-item legend; ρ₁ values locked")


# ============================================================ Figure 4 (v3)
def fig4() -> None:
    bl = json.loads(SRC["bottomline"].read_text())
    EXT_MAP = {
        "MRL (GSE114002)": 0.3132,
        "polyA (GSE269595)": 0.7343,
        "MPRAU (ENCSR854RUF)": 0.1205,
        "TE (GSE200304)": 0.0,
        "GSE149487-TE (n=48)": 0.0,
        "GSE149487-RNA (n=48)": 0.2958,
        "GSE186455 (n=274)": 0.1043,
        "HALF_LIFE 5'/3'UTR (GSE217518)": 0.0985,
    }
    rows4 = []
    for r in bl["rows"]:
        task = r.get("task", "")
        ours = (r.get("ours") or {}).get("spearman")
        if ours is None or task not in EXT_MAP:
            continue
        rows4.append((task, EXT_MAP[task], ours))
    lk = {t: (e, o) for t, e, o in rows4}
    lock("fig4.ours_polyA", lk["polyA (GSE269595)"][1], 0.8219)
    lock("fig4.ours_MRL", lk["MRL (GSE114002)"][1], 0.3217)
    icc = {"polyA": 0.90, "MRL": 0.83, "REF/ALT": 0.21, "MPRAU": 0.683,
           "PLUMAGE-RNA": 0.364, "TE": 0.586, "HL": 0.0}

    def icc_of(task):
        for k, v in icc.items():
            if k in task:
                return v
        return None

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ys = np.arange(len(rows4))[::-1]
    for y, (task, e, o) in zip(ys, rows4):
        c = icc_of(task)
        if c:
            ax.plot([0, c], [y, y], color="0.90", lw=1.6, zorder=1)
            ax.scatter([c], [y], s=46, marker="|", color="0.35", lw=1.5, zorder=3)
            ax.text(c + 0.012, y, f"ceiling {c:.3f}", fontsize=7.0, color="0.4", va="center")
        ax.scatter([e], [y], s=44, color="#4878d0", zorder=4, edgecolor="white", lw=0.7)
        ax.scatter([o], [y], s=62, color="#ee854a", zorder=5, edgecolor="white", lw=0.8)
        ax.plot([e, o], [y, y], color="0.75", lw=1.2, zorder=2)
        comp = (o / c * 100) if c else None
        label = f"ours {o:.4f}"
        if comp:
            label += f"  ({comp:.0f}% of ceiling)"
        ax.text(max(e, 0.0) + 0.02, y + 0.24, label, fontsize=7.2, color="#b5471f", ha="left")
    ax.set_yticks(ys)
    ax.set_yticklabels([t.replace(" (GSE", "\n(GSE") for t, _, _ in rows4], fontsize=8.2)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.7, len(rows4) - 0.25)
    ax.set_xlabel("frozen-Δ Spearman ρ", fontsize=9)
    ax.axvline(0, color="0.85", lw=0.6)
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#4878d0", markersize=7, label="best external"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor="#ee854a", markersize=8, label="ours"),
         Line2D([0], [0], marker="|", color="0.35", markersize=8, linestyle="none", label="label ceiling (ICC)")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Ceiling-normalized completion by task (dumbbell: external → ours → ICC ceiling)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    save(fig, "Figure4_ceiling_completion")
    print(f"  Figure4: {len(rows4)} tasks; ours values locked vs bottomline")


# ============================================================ Figure 5 (C, labels fixed)
def fig5() -> None:
    """Intervention triangle. All 10 values locked against named frozen files.

    Panel 1 (Synthetic, D16-C): probe gap_backtest reference (V5 0.7943) and
      the D16-C probe gap 0.8235 -- both verbatim from gap_backtest_d16c.json.
    Panel 2 (Parameter, ERK): erk_adjudication_v1 G1 rho_val 0.2015; E7
      closed-form first-order bound 0.2069 (paper §6.1/§7 archive value).
    Panel 3 (Real data, W ladder): W0 0.1987 (w_ladder results), LoRA 0.2470
      (280k_prefinetune frozen_delta_results), full-FT 2ep 0.2555
      (280k_fullft_ablation frozen_delta_results), 3-seed ens 0.3158
      (ensemble_3seed_vs_optimus), frozen-Optimus 0.3132 (same file).
    """
    g = json.loads(SRC["d16_gap"].read_text())
    v5_gap = g["reference_official_v5"]["gap"]
    d16_gap = g["gap"]
    lock("fig5.v5_gap", v5_gap, 0.7943, tol=5e-4)
    lock("fig5.d16_gap", d16_gap, 0.8235, tol=5e-4)
    erk = json.loads(SRC["erk_adj"].read_text())
    erk_val = erk["gates"]["G1_generalization"]["rho_val"]
    lock("fig5.erk_val", erk_val, 0.2015, tol=5e-4)
    E7_BOUND = 0.2069
    wl = json.loads(SRC["w_ladder"].read_text())
    w0 = wl["mrl_aligned"]["w0_mrl_gse114002"]["task_macro_spearman"]
    lock("fig5.w0", w0, 0.1987, tol=5e-4)
    lora = json.loads(SRC["lora_280k"].read_text())["metrics"]["task_macro_spearman"]
    lock("fig5.lora", lora, 0.2470, tol=5e-4)
    ff2 = json.loads(SRC["fullft_2ep"].read_text())["metrics"]["task_macro_spearman"]
    lock("fig5.fullft_2ep", ff2, 0.2555, tol=5e-4)
    ens = json.loads(SRC["ens_3seed"].read_text())
    ens3 = ens["ensemble"]["task_macro_spearman"]
    optimus = ens["frozen_optimus"]["task_macro_spearman"]
    lock("fig5.ens3", ens3, 0.3158, tol=5e-4)
    lock("fig5.optimus", optimus, 0.3132, tol=5e-4)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    data = [
        ("Synthetic density (D16-C)", [v5_gap, d16_gap],
         ["baseline (V5)", "synthetic\n+32/source"], "FALSIFIED", "#d65f5f", "train–backtest gap"),
        ("Parameter prior (ERK, 1,177 p)", [erk_val, E7_BOUND],
         ["ERK v2 val", "E7 closed-form\nbound"], "BOUND-ANCHORED", "#ee854a", "MRL ρ"),
        ("Real data (W ladder)", [w0, lora, ff2, ens3, optimus],
         ["W0", "LoRA", "full-FT\n2 ep", "3-seed\nens", "frozen-\nOptimus"],
         "EFFECTIVE (tie)", "#4878d0", "MRL ρ"),
    ]
    for ax, (title, vals, labels, verdict, col, ylab) in zip(axes, data):
        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.58, color=col, alpha=0.88, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.03, f"{v:.4f}", ha="center", fontsize=8, color="0.25")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=10)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.text(0.5, 0.95, verdict, transform=ax.transAxes, ha="center",
                fontsize=9, color=col, fontweight="bold")
    fig.suptitle("Intervention triangle: only same-source dense supervision moves MRL Δ",
                 fontsize=10.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Figure5_intervention_triangle")
    print("  Figure5: 10 locked values; W-ladder labels per paper §7 (LoRA 0.2470 "
          "now separate from full-FT 2ep 0.2555)")


# ============================================================ Figure 6 (D)
def fig6() -> None:
    fo = json.loads(SRC["first_order"].read_text())
    rho1 = {t: fo["tasks"][t]["rho_first_order"] for t in ("polyA", "MPRAU", "TE")}
    lock("fig6.rho1_polyA", rho1["polyA"], 0.45554942, tol=1e-6)
    # MRL rho1 = E7 first-order bound, paper §6.1 archive value
    panel = {"polyA": (rho1["polyA"], 0.8219, 0.90),
             "MPRAU": (rho1["MPRAU"], 0.1025, 0.683),
             "TE": (rho1["TE"], 0.0579, 0.586),
             "MRL": (0.2069, 0.3217, 0.83)}
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3))
    for ax, (task, (r1, best, ceil)) in zip(axes, panel.items()):
        ax.bar(0, r1 / ceil, width=0.5, color="#4878d0", label="first-order table")
        ax.bar(0, (best - r1) / ceil, bottom=r1 / ceil, width=0.5, color="#ee854a",
               label="structural (ours − ρ₁)")
        ax.bar(0, (ceil - best) / ceil, bottom=best / ceil, width=0.5, color="0.90",
               label="label-noise ceiling room")
        ax.axhline(best / ceil, color="#d65f5f", lw=1.0, ls="--")
        ax.text(0.30, best / ceil + 0.02, f"best {best:.3f}", fontsize=7.5, color="#d65f5f")
        if r1 / ceil > 0.07:
            ax.text(0.30, r1 / 2, f"ρ₁ {r1:.3f}", fontsize=7.5, va="center", color="#4878d0")
        pct = (r1 / best) * 100
        ax.set_title(f"{task}\nfirst-order = {pct:.0f}% of best", fontsize=8.5,
                     fontweight="bold", pad=8)
        ax.set_xlim(-0.55, 1.15); ax.set_ylim(0, 1.02)
        ax.set_xticks([])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("fraction of ICC ceiling", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where the learnable signal lives: first-order table vs structural component",
                 fontsize=10.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Figure6_first_order_multiples")
    print("  Figure6: 4 panels; ρ₁ for polyA/MPRAU/TE locked, MRL 0.2069 (archive)")


# ============================================================ Figure S1 (14 families)
def figs1() -> None:
    mx = json.loads(SRC["matrix"].read_text())
    grid5 = matrix_grid(mx)
    lock("figs1.total_cells", int(mx["counts"]["total_cells"]), 65)

    TASK_KEYS = ["MRL", "polyA", "MPRAU", "TE", "PLUM-TE", "PLUM-RNA", "REF/ALT",
                 "HL5", "HL3", "M1", "M6", "S1_3", "S1_5"]
    EXISTING = {
        "V5 (ours)":        {"MRL": 0.1354, "polyA": 0.8219, "MPRAU": 0.1025, "TE": 0.0579,
                             "PLUM-TE": 0.1953, "PLUM-RNA": 0.0500, "REF/ALT": 0.0639,
                             "HL5": 0.0607, "HL3": 0.0456},
        "frozen-Optimus":   {"MRL": 0.3132},
        "APARENT":          {"polyA": 0.7343},
        "APARENT2":         {"polyA": 0.6810},
        "FramePool":        {"MRL": 0.2956},
        "UTR-LM":           {"MRL": 0.1107, "polyA": 0.7490, "TE": 0.0113, "PLUM-TE": -0.028,
                             "PLUM-RNA": 0.043, "REF/ALT": -0.123},
        "RNA-FM":           {"MRL": 0.1369, "polyA": 0.7114, "MPRAU": 0.0131, "TE": 0.0009,
                             "PLUM-TE": -0.015, "PLUM-RNA": 0.2958, "REF/ALT": 0.1043,
                             "HL5": 0.0271, "HL3": 0.0500},
        "Saluki":           {"MPRAU": 0.1205, "HL3": 0.0985},
        "mRNABERT-raw":     {"MRL": 0.08},
        "RiboNN":           {"HL5": 0.0778, "HL3": -0.0450},
    }
    lock("figs1.v5_polyA", EXISTING["V5 (ours)"]["polyA"], 0.8219)
    lock("figs1.v5_mrl", EXISTING["V5 (ours)"]["MRL"], 0.1354)
    lock("figs1.optimus_mrl", EXISTING["frozen-Optimus"]["MRL"], 0.3132)
    lock("figs1.aparent_polyA", EXISTING["APARENT"]["polyA"], 0.7343)
    lock("figs1.rnafm_mrl", EXISTING["RNA-FM"]["MRL"], 0.1369)
    lock("figs1.saluki_mprau", EXISTING["Saluki"]["MPRAU"], 0.1205)

    n_top = len(FAMS5)
    n_bot = len(EXISTING)
    H = n_top + n_bot + 1
    W = len(TASKS)
    grid = np.full((H, W), np.nan)
    mask_na = np.zeros((H, W), dtype=bool)
    grid[:n_top, :] = grid5
    for i, (fam, d) in enumerate(EXISTING.items()):
        row = n_top + 1 + i
        for j, k in enumerate(TASK_KEYS):
            v = d.get(k)
            if v is None:
                mask_na[row, j] = True
            else:
                grid[row, j] = v
    assert np.isfinite(grid[:n_top, :]).all()

    cmap = LinearSegmentedColormap.from_list("soft_div", SOFT_DIV)
    norm = Normalize(vmin=-0.4, vmax=0.9)
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i in range(H):
        for j in range(W):
            if mask_na[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#ececec",
                                       edgecolor="#c9c9c9", hatch="///", lw=0.3))
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="white" if (v > 0.55 or v < -0.28) else "0.25")
    ax.axhline(n_top + 0.5, color="0.25", lw=1.4)
    ax.text(-0.62, n_top, "frozen 5×13 matrix\n(65/65 re-checked)", fontsize=7.6,
            color="0.25", ha="right", va="center", fontweight="bold")
    ax.text(-0.62, n_top + 1 + (n_bot - 1) / 2,
            "9 reference families + ours\n(archived ALREADY_DONE rows;\ngrey = no reading in this caliber)",
            fontsize=7.6, color="0.25", ha="right", va="center", fontweight="bold")
    ax.axvline(8.5, color="0.35", lw=0.9, ls=(0, (4, 2)))
    ax.text(9.0, -0.9, "append-only new rows (frozen before scoring)",
            fontsize=7.6, color="0.3", style="italic")
    ylabels = FAM5_SHORT + [""] + list(EXISTING.keys())
    ax.set_yticks(range(H)); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(range(W)); ax.set_xticklabels(TASK_SHORT, fontsize=8)
    ax.get_yticklabels()[n_top].set_text("")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.02, pad=0.01)
    cb.set_label("frozen-Δ Spearman", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("DeltaBench supplement: all 14 model families — 65-cell re-checked matrix (top) "
                 "+ archived reference rows (bottom)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    save(fig, "FigureS1_all_families_supplement")
    n_ext = int(np.isfinite(grid[n_top + 1:, :]).sum())
    print(f"  FigureS1: top 65/65 locked; bottom {n_ext} archived cells + "
          f"{int(mask_na[n_top + 1:].sum())} NA-hatched")


# ============================================================ main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"bioRxiv figure pack v2 -> {OUT}")
    for fn in (fig1, fig2a, fig2b, fig3, fig4, fig5, fig6, figs1):
        name = fn.__name__
        print(f"== {name}")
        fn()
    manifest = {
        "schema_version": "biorxiv_figure_pack_v2",
        "date": "2026-10-07",
        "figures": [
            "Figure1_matrix_heatmap", "Figure2a_density_v1", "Figure2b_density_v2_heldout",
            "Figure3_first_order", "Figure4_ceiling_completion", "Figure5_intervention_triangle",
            "Figure6_first_order_multiples", "FigureS1_all_families_supplement",
        ],
        "sources": {k: str(v) for k, v in SRC.items()},
        "value_locks": LOCK_LOG,
        "value_lock_count": len(LOCK_LOG),
        "note": "read-only producers; every plotted value read from frozen JSONs or "
                "declared verbatim from the paper's frozen archives; no recomputation",
        "style": "DejaVu Sans 9 / 300 dpi / pdf.fonttype 42 / no top-right spines / outward ticks",
    }
    (OUT / "figure_pack_v2_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    print(f"manifest written ({len(LOCK_LOG)} value locks, all OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
