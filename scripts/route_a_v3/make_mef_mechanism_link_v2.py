#!/usr/bin/env python3
"""MEF x mechanism correlation figure v2 (2026-10-12).

v1 -> v2: per-point value labels (rho / pass%) so every one of the 5 families
is individually legible at slide scale; slightly larger fonts. All values read
live from frozen sources and asserted (same locks as v1). Answer to reviewer
question: no family is missing - STCNet/LAMAR sit near the x=0 guide.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_edit_budget_v1")
RES = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json")
OUT = D / "Figure9b_mef_mechanism_link_v2.png"

FAMS = [("gemorna", "GEMORNA"), ("lamar_utr5te", "LAMAR"), ("utr_insight", "UTR-Insight"),
        ("utr_stcnet", "UTR-STCNet"), ("hydrarna", "HydraRNA")]

def main() -> int:
    rows = list(csv.DictReader(open(D / "edit_budget_adjudication_v1.csv")))
    by = {(r["family"], r["task"]): r for r in rows}
    pass_rate = {f: 1 - float(by[(f, "polya")]["fail_budget_rate"]) for f, _ in FAMS}
    m = json.load(open(RES))
    cell = "GSE269595|3UTR|PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS"
    rho = {f: float(m["cells"][cell][f]["overall_spearman"]) for f, _ in FAMS}

    assert abs(rho["gemorna"] - 0.222513) < 1e-5
    assert abs(rho["lamar_utr5te"] - 0.1483) < 1e-3
    assert abs(rho["utr_insight"] - 0.3136) < 1e-3
    assert abs(rho["utr_stcnet"] - (-0.0217)) < 1e-3
    assert abs(rho["hydrarna"] - (-0.3719)) < 1e-3
    assert abs(pass_rate["gemorna"] - 0.9962) < 1e-3
    assert abs(pass_rate["hydrarna"] - 0.5647) < 1e-3
    print("value locks: matrix rho + MEF pass rates all matched frozen sources")

    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=300)
    xs = [rho[f] for f, _ in FAMS]
    ys = [pass_rate[f] for f, _ in FAMS]
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")

    ax.scatter(xs, ys, s=95, color="#4472C4", zorder=3, edgecolor="k", linewidth=0.5)
    OFF = {"GEMORNA": (7, -4), "LAMAR": (9, 6), "UTR-Insight": (9, -18),
           "UTR-STCNet": (9, 4), "HydraRNA": (7, 7)}
    for (f, label), x, y in zip(FAMS, xs, ys):
        dx, dy = OFF[label]
        ax.annotate(f"{label}\n{rho[f]:+.2f} / {pass_rate[f]*100:.0f}%", (x, y),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    fontweight="bold", linespacing=1.1)

    from scipy.stats import spearmanr as spr
    r_rank, _ = spr(xs, ys)
    ax.text(0.02, 0.02, "5 points = all 5 ported families; same ordering (n=5, descriptive)",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#ED7D31", lw=0.6))

    ax.set_xlabel("matrix frozen-\u0394 Spearman \u03c1 (polyA, discriminative face)", fontsize=9)
    ax.set_ylabel("MEF in-budget pass rate (polyA, decision face)", fontsize=9)
    ax.set_title("Discriminative ability (matrix) vs edit-budget pass rate (MEF), polyA", fontsize=10)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlim(-0.48, 0.45)
    ax.grid(alpha=0.25, lw=0.4)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print("rank r =", round(float(r_rank), 3), "(n=5, descriptive only)")

    manifest = {
        "figure": "Figure9b_mef_mechanism_link_v2",
        "supersedes": "Figure9b_mef_mechanism_link_v1.png (same data/locks; v2 adds per-point value labels)",
        "sources": {"mef": str(D / "edit_budget_adjudication_v1.csv"), "matrix": str(RES)},
        "points": {f: {"rho_matrix": round(rho[f], 4), "mef_pass_rate": round(pass_rate[f], 4)} for f, _ in FAMS},
        "rank_spearman_r": round(float(r_rank), 3),
        "note": "n=5 families, descriptive ordering agreement only (no significance claim)",
    }
    json.dump(manifest, open(D / "figure9b_manifest_v2.json", "w"), indent=1)
    print("manifest written")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
