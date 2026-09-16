#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analysis_delta_vs_density (paper version, English labels)
Figure: external frozen-Δ performance vs within-source candidate density (log x) + V5 gap.
All data from frozen artifacts; no new training.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_delta_vs_density_20260915"
os.makedirs(OUT_DIR, exist_ok=True)

DENSITY = {"MRL": 4.9, "HL5": 1.7, "HL3": 1.6, "MPRAU": 6.1, "TE200304": 1.0,
           "REFALT": 2.1, "polyA": 126.7}
TASK_LABEL = {
    "MRL": "MRL", "HL5": "HL-5'UTR", "HL3": "HL-3'UTR", "MPRAU": "MPRAU",
    "TE200304": "TE", "REFALT": "REF/ALT", "polyA": "polyA",
}

def load(path):
    with open(path) as f:
        return json.load(f)

rows = []
d = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_frozen_delta_full_coverage_20260904/frozen_delta_results.json")
TASKMAP = {"gse114002_mrl": "MRL", "mprau_encsr854ruf": "MPRAU", "polya_gse269595": "polyA",
           "half_life_5utr": "HL5", "half_life_3utr": "HL3", "gse200304_te": "TE200304",
           "gse186455": "REFALT"}
for task_key, models in d["results"].items():
    t = TASKMAP.get(task_key)
    if t is None:
        continue
    for model, blob in models.items():
        v = blob.get("task_macro_spearman")
        if v is not None:
            rows.append((t, model, float(v)))

d2 = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_frozen_delta_gse114002_20260903/frozen_delta_results.json")
for m, blob in d2["results"].items():
    if isinstance(blob, dict) and "task_macro_spearman" in blob:
        rows.append(("MRL", "optimus5prime" if "optimus" in m else m, float(blob["task_macro_spearman"])))

d3 = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_mrnabert_raw_frozen9/mrnabert_raw_frozen9_results.json")
T3 = {"MRL": "MRL", "MPRAU": "MPRAU", "polyA": "polyA", "REFALT": "REFALT",
      "HL5": "HL5", "HL3": "HL3", "TE200304": "TE200304"}
for t, v in d3["tasks"].items():
    tt = T3.get(t)
    if tt:
        rows.append((tt, "mrnabert_raw", float(v["spearman"])))

d4 = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_ribonn_frozen_te_20260913/frozen_delta_results.json")
for task_key, blob in d4["results"].items():
    t = {"gse200304_te": "TE200304"}.get(task_key)
    if t and isinstance(blob, dict) and "task_macro_spearman" in blob:
        rows.append((t, "ribonn", float(blob["task_macro_spearman"])))

d5 = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_saluki_frozen_mprau_20260903/frozen_delta_results.json")
rows.append(("MPRAU", "saluki_frozen", float(d5["mprau_pair_mean"]["pair_mean_spearman"])))

d6 = load("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_aparent2_frozen_delta_20260906/result.json")
rows.append(("polyA", "aparent2", float(d6["task_macro_spearman"])))
rows.append(("polyA", "aparent2019", float(d6["references"]["aparent_2019_spearman"])))

bt = load("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902/docs/training_journal/v5_train_backtest_20260912.json")
V5 = {t: bt["tasks"][t] for t in ["MRL", "polyA", "MPRAU", "HL5", "HL3"]}

xs = [DENSITY[t] for t, _, _ in rows]
ys = [r for _, _, r in rows]
sp_rho, sp_p = stats.spearmanr(xs, ys)
kd_tau, kd_p = stats.kendalltau(xs, ys)
xlog = np.log10([x for x in xs])
pr_r, pr_p = stats.pearsonr(xlog, ys)

# linear fit in log-space for panel A
coef = np.polyfit(xlog, ys, 1)
xx_fit = np.logspace(np.log10(0.9), np.log10(140), 60)

plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 11, "axes.linewidth": 0.9})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4))

MARKERS = {"rnafm": "o", "utrlm": "s", "mrnabert_raw": "^", "optimus5prime": "D",
           "framepool": "v", "saluki_frozen": "P", "ribonn": "X", "aparent2": "p",
           "aparent2019": "*"}
NICE = {"rnafm": "RNA-FM", "utrlm": "UTR-LM", "mrnabert_raw": "mRNABERT (raw)",
        "optimus5prime": "Optimus-5'", "framepool": "FramePool", "saluki_frozen": "Saluki",
        "ribonn": "RiboNN", "aparent2": "APARENT2", "aparent2019": "APARENT (2019)"}

# Panel A
ax1.axhline(0, color="#999999", lw=0.8, ls="--")
ax1.plot(xx_fit, np.polyval(coef, np.log10(xx_fit)), ls="--", lw=1.2, color="#5B8DB8",
         label=f"OLS on log$_{{10}}$ density: r={pr_r:.2f}, p={pr_p:.0e}")
for m in MARKERS:
    xs_m = [DENSITY[t] for t, mm, _ in rows if mm == m]
    ys_m = [r for _, mm, r in rows if mm == m]
    if xs_m:
        ax1.scatter(xs_m, ys_m, marker=MARKERS[m], s=72, label=NICE[m],
                    alpha=0.9, edgecolors="black", linewidths=0.5, zorder=4)
ax1.scatter([DENSITY["polyA"]], [0.8219], marker="*", s=260, color="#C0392B",
            edgecolors="black", linewidths=0.6, label="mRNA-EditFlow V5 (ours)", zorder=6)
ax1.scatter([DENSITY["MRL"]], [0.1354], marker="*", s=260, color="#C0392B",
            edgecolors="black", linewidths=0.6, zorder=6)
ax1.set_xscale("log")
ax1.set_xlim(0.8, 180)
ax1.set_xlabel("Within-source candidate density (candidates per source, log scale)")
ax1.set_ylabel("Delta-prediction Spearman $\\rho$ (validation)")
ax1.set_title(f"(a) Frozen-$\\Delta$ performance vs data regime  (n = {len(rows)} model-task rows)\n"
              f"Spearman $\\rho$ = {sp_rho:.2f} (p = {sp_p:.1e});  Kendall $\\tau$ = {kd_tau:.2f}")
ax1.annotate("polyA cluster\n(external 0.68-0.75, ours 0.82)", xy=(126.7, 0.72),
             xytext=(11, 0.60), fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
ax1.annotate("failure band (density 1-6)\nexternal $\\rho \\approx -0.12$ to $+0.31$",
             xy=(4.0, 0.02), xytext=(1.0, 0.45), fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
ax1.legend(loc="lower right", fontsize=8.2, ncol=2, framealpha=0.92)
ax1.grid(True, which="both", alpha=0.25, lw=0.4)

# Panel B
tasks_b = ["MRL", "MPRAU", "HL3", "HL5", "polyA"]
xb = np.array([DENSITY[t] for t in tasks_b], dtype=float)
yb = np.array([V5[t]["gap"] for t in tasks_b])
sp_gap, sp_gap_p = stats.spearmanr(xb, yb)
ax2.axhline(0.30, color="#C0392B", lw=1.0, ls=":", label="D16-C G1 gate (gap $\\leq$ 0.30)")
ax2.scatter(xb, yb, s=120, color="#1F4E78", edgecolors="black", linewidths=0.6, zorder=4)
for t, x, y in zip(tasks_b, xb, yb):
    ha = "left" if t == "polyA" else "center"
    xt = x * 1.06 if t == "polyA" else x
    yt = y - 0.03 if t in ("HL3", "MRL") else y + 0.015
    ax2.annotate(f"{TASK_LABEL[t]}\ngap {y:.2f}", xy=(x, y), xytext=(xt, yt),
                 fontsize=9.5, ha=ha)
zz = np.polyfit(np.log10(xb), yb, 1)
xx2 = np.logspace(np.log10(0.9), np.log10(150), 50)
ax2.plot(xx2, np.polyval(zz, np.log10(xx2)), ls="--", lw=1.1, color="#888888",
         label=f"log-linear trend (Spearman $\\rho$ = {sp_gap:.2f}, p = {sp_gap_p:.1e})")
ax2.set_xscale("log")
ax2.set_xlim(0.8, 200)
ax2.set_ylim(-0.06, 0.88)
ax2.set_xlabel("Within-source candidate density (candidates per source, log scale)")
ax2.set_ylabel("V5 train-backtest gap ($\\rho_{train} - \\rho_{val}$)")
ax2.set_title("(b) Memorization (train-val gap) follows the same regime\n"
              "polyA gap 0.03 (generalizes) vs MRL gap 0.79 (memorizes)")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, which="both", alpha=0.25, lw=0.4)

fig.suptitle("Delta prediction is regime-limited: published models fail outside dense within-source data",
             fontsize=13, y=1.00)
fig.tight_layout()
png = os.path.join(OUT_DIR, "delta_vs_density_scatter.png")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "delta_vs_density_scatter.pdf"), bbox_inches="tight")

out = {
    "schema": "delta_vs_density_analysis.v1",
    "date": "2026-09-15",
    "caliber_note": ("external rows = frozen-Delta (frozen backbone, linear/ridge readout fit on TRAIN only); "
                     "V5 = supervised in-domain; density = amendment sec2 frozen TRAIN cand/source; "
                     "Saluki MPRAU row = pair-mean caliber; APARENT 2019 = reference row from aparent2 artifact"),
    "density_table": DENSITY,
    "rows": [{"task": t, "model": m, "rho": round(r, 4)} for t, m, r in rows],
    "stats": {
        "n_rows": len(rows),
        "spearman_rho_density_vs_external": round(sp_rho, 4),
        "spearman_p": float(sp_p),
        "kendall_tau": round(kd_tau, 4),
        "kendall_p": float(kd_p),
        "pearson_log10_density_r": round(pr_r, 4),
        "pearson_p": float(pr_p),
        "spearman_gap_vs_density": round(sp_gap, 4),
        "spearman_gap_p": float(sp_gap_p),
    },
    "v5_backtest": {t: {k: round(v, 4) if isinstance(v, float) else v for k, v in blob.items()}
                    for t, blob in V5.items()},
    "outputs": {"figure_png": png, "figure_pdf": os.path.join(OUT_DIR, "delta_vs_density_scatter.pdf")},
    "protected_reads": 0,
}
with open(os.path.join(OUT_DIR, "delta_vs_density_data.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("stats:", json.dumps(out["stats"], indent=1))
print("saved:", OUT_DIR)
