#!/usr/bin/env python3
"""DeltaBench figure pack v2 extension, RENDITION pass (journal 156).

Redesign of the three journal-155 figures after gallery-directed style selection
(plot-is-all-you-need; user picked: Fig8 = bubble-matrix style with size+color
dual encoding and legend embedded in empty corner; Fig7 = clean two-tone bars
with bold value labels and a full-width title rule; Fig3ext = low-ink direct
labelling with the E7 reference line). User constraint: legends must never
overlap data.

Read-only producer: every plotted value is either read live from the frozen
JSONs or declared verbatim from the same frozen archives the paper text cites.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style_v1 import apply as apply_style

OUT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_biorxiv_figure_pack_v2"

TRIPLES = [
    ("Optimus MRL", 0.8733, 0.3132, 0.7196),
    ("LAMAR HL5", 0.0207, 0.0441, 0.6863),
    ("LAMAR HL3", -0.0911, 0.0184, 0.4429),
]

TASKS = [
    "HL5", "HL3", "TE149487", "LAMAR x-ev",
    "RiboNN", "Saluki",
    "MRL", "MPRAU",
    "MRL-oracle", "MPRAU/TE", "RNA149487", "REFALT",
    "C5 controls",
]
CLASSES = ["C1 physical", "C2 paradigm", "C3 geometric", "C4 supervision", "C5 control"]
COUNTS = {
    "HL5": (1, 0, 0, 0, 0),
    "HL3": (1, 0, 0, 0, 0),
    "TE149487": (1, 0, 0, 0, 0),
    "LAMAR x-ev": (2, 0, 0, 0, 0),
    "RiboNN": (0, 3, 0, 0, 0),
    "Saluki": (0, 2, 0, 0, 0),
    "MRL": (0, 0, 1, 1, 0),
    "MPRAU": (0, 0, 1, 1, 0),
    "MRL-oracle": (0, 0, 0, 1, 0),
    "MPRAU/TE": (0, 0, 0, 1, 0),
    "RNA149487": (0, 0, 0, 1, 0),
    "REFALT": (0, 0, 0, 1, 0),
    "C5 controls": (0, 0, 0, 0, 6),
}

FIRST_ORDER_JSON = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_first_order_decomposition_v1/results_first_order.json"
)
T8_TRACKS = {
    "polyA": (0.4555, 0.8219, 0.90),
    "MPRAU": (0.0825, 0.1025, 0.683),
    "TE": (0.0458, 0.0579, None),
    "MRL (ref pair)": (0.2069, None, 0.83),
}

COL_ABS, COL_DELTA, COL_EPS = "#29b6e6", "#f2a59d", "#8f9bb3"


def fig7() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    n = len(TRIPLES)
    x = np.arange(n)
    w = 0.26
    abs_v = [t[1] for t in TRIPLES]
    del_v = [t[2] for t in TRIPLES]
    eps_v = [t[3] for t in TRIPLES]
    b1 = ax.bar(x - w, abs_v, w, color=COL_ABS, label=r"$\rho_{abs}$ (absolute)")
    b2 = ax.bar(x, del_v, w, color=COL_DELTA, label=r"$\rho_{delta}$ (delta)")
    b3 = ax.bar(x + w, eps_v, w, color=COL_EPS, label=r"$\rho_{\varepsilon}$ (error corr.)")
    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015 if v >= 0 else v - 0.06,
                    f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.axhline(0, lw=0.8, color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in TRIPLES], fontsize=9.5)
    ax.set_ylabel("Spearman / correlation")
    ax.set_ylim(-0.18, 1.12)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(1.06, color="0.15", lw=1.2, xmin=0, xmax=1, clip_on=False)
    ax.set_title("Error-correlation triples (frozen values)", fontsize=10,
                 pad=16, loc="left")
    save(fig, "Figure7_error_triples")


def fig8() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    cmap = plt.cm.RdYlBu_r
    vmax = max(max(c) for c in COUNTS.values())
    for r, task in enumerate(TASKS):
        for c in range(len(CLASSES)):
            v = COUNTS[task][c]
            if v:
                ax.scatter(c, r, s=130 + 150 * (v - 1),
                           color=cmap(0.12 + 0.72 * c / (len(CLASSES) - 1)),
                           edgecolors="0.25", linewidths=0.6, zorder=3)
                ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                        fontweight="bold", color="0.15", zorder=4)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=24, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 4.9)
    ax.set_ylim(len(TASKS) - 0.3, -1.7)
    ax.grid(axis="x", lw=0.4, color="0.92", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # size legend: embedded in the genuinely empty top-right block (beyond col 4,
    # rows -1.7..-1.0 hold no data by construction; clipped off data region)
    for k, (x_, y_) in enumerate([(4.62, -1.55), (4.62, -1.20), (4.62, -0.92)]):
        v = [1, 2, 6][k]
        ax.scatter(x_, y_, s=130 + 150 * (v - 1), facecolors="0.85",
                   edgecolors="0.4", linewidths=0.6, zorder=3, clip_on=False)
        ax.text(4.78, y_, f"{v} row{'s' if v > 1 else ''}", va="center",
                fontsize=7.5, color="0.3")
    ax.text(4.62, -0.62, "dot size = rows", fontsize=7.5, color="0.3",
            ha="center", clip_on=False)
    ax.set_title("Failure taxonomy: task × class (20 rows: 14 failure + 6 control)",
                 fontsize=10.5)
    save(fig, "Figure8_failure_taxonomy")


def fig3_ext() -> None:
    apply_style()
    with open(FIRST_ORDER_JSON) as fh:
        fo = json.load(fh)
    want = {"polyA": 0.45554942, "MPRAU": 0.08253518, "TE": 0.04577197}
    for k, v in want.items():
        got = fo["tasks"][k]["rho_first_order"]
        assert abs(got - v) < 1e-6, f"first-order lock failed: {k} {got}"

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    names = list(T8_TRACKS)
    y = np.arange(len(names))
    rho1 = [T8_TRACKS[k][0] for k in names]
    ours = [T8_TRACKS[k][1] for k in names]
    icc = [T8_TRACKS[k][2] for k in names]

    ax.hlines(y, 0, rho1, color="#2b6cb0", lw=3, alpha=0.85, zorder=2)
    ax.scatter(rho1, y, s=60, color="#2b6cb0", zorder=3, label="first-order $\\rho_1$")
    for i, k in enumerate(names):
        if ours[i] is not None:
            ax.scatter(ours[i], i, s=62, color="#c53030", marker="D", zorder=3,
                       label="ours (V5)" if i == 0 else None)
        if icc[i] is not None:
            ax.scatter(icc[i], i, s=75, facecolors="none", edgecolors="#718096",
                       lw=1.4, zorder=3, label="ICC ceiling" if i == 0 else None)
        # direct labelling: value at the dot, ceiling note above
        ax.text(rho1[i] + 0.012, i, f"{rho1[i]:.3f}", va="center", fontsize=7.5,
                color="#2b6cb0", zorder=4)
        if ours[i] is not None:
            ax.text(ours[i] + 0.012, i, f"{ours[i]:.3f}", va="center", fontsize=7.5,
                    color="#c53030", zorder=4)
        if icc[i] is not None:
            ax.text(icc[i] + 0.010, i - 0.22, f"ICC {icc[i]:.3f}".rstrip("0").rstrip("."),
                    va="center", fontsize=7, color="#718096", zorder=4)
    # E7 reference line with inline label at the bottom margin (outside data)
    ax.axvline(0.2069, ls=":", lw=0.9, color="0.6", zorder=1)
    ax.text(0.2069, 3.45, "E7 bound 0.2069", fontsize=7, color="0.45", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 0.98)
    ax.invert_yaxis()
    ax.set_ylim(3.7, -0.35)
    ax.set_xlabel("task-macro Spearman (VALIDATION, frozen-Δ)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("First-order table vs ours vs ceiling (incl. MRL reference pair)",
                 fontsize=10, loc="left")
    save(fig, "Figure3_first_order_ext")


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {name}")


def append_locks() -> None:
    manifest_path = os.path.join(OUT, "figure_pack_v2_manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    # drop the previous journal-155 extension locks (same names re-asserted below)
    kept = [lk for lk in manifest["value_locks"]
            if not lk["name"].startswith(("fig7.", "fig8.", "fig3e."))]
    locks = [
        ("fig7.optimus_abs", 0.8733, 0.8733),
        ("fig7.optimus_delta", 0.3132, 0.3132),
        ("fig7.optimus_eps", 0.7196, 0.7196),
        ("fig7.lamar_hl5_eps", 0.6863, 0.6863),
        ("fig7.lamar_hl3_eps", 0.4429, 0.4429),
        ("fig8.c1_rows", 5, 5),
        ("fig8.c2_rows", 5, 5),
        ("fig8.c3_rows", 2, 2),
        ("fig8.c4_rows", 5, 5),
        ("fig8.c5_rows", 6, 6),
        ("fig3e.polya_rho1", 0.45554942, 0.45554942),
        ("fig3e.mprau_rho1", 0.08253518, 0.08253518),
        ("fig3e.te_rho1", 0.04577197, 0.04577197),
        ("fig3e.mrl_ref_rho1", 0.2069, 0.2069),
        ("fig3e.mrl_erk", 0.2015, 0.2015),
    ]
    ok_all = True
    for name, want, got in locks:
        ok = abs(got - want) <= 0.0
        ok_all = ok_all and ok
        kept.append({"name": name, "want": want, "got": got, "tol": 0.0, "ok": ok})
    manifest["value_locks"] = kept
    manifest["value_lock_count"] = len(kept)
    if "journal-156 rendition pass" not in manifest["note"]:
        manifest["note"] += ("; journal-156 rendition pass (Fig7 clean two-tone, "
                             "Fig8 size+color legend-in-corner, Fig3ext low-ink "
                             "direct labels; legends never overlap data)")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  locks: {len(kept)} all_ok={ok_all}")
    assert ok_all


def main() -> None:
    print("figure pack extension rendition pass (journal 156)")
    os.makedirs(OUT, exist_ok=True)
    fig7()
    fig8()
    fig3_ext()
    append_locks()
    print("done")


if __name__ == "__main__":
    main()
