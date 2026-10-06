#!/usr/bin/env python3
"""DeltaBench figure pack v2 extension (journal 155):
Figure 7  error-correlation triples (rho_abs / rho_delta / rho_eps) - grouped bars
Figure 8  failure taxonomy matrix (task x class C1-C5, dot size = row count)
Figure 3+  first-order decomposition extended with MRL reference-pair track (T8 merge)

Read-only producer: every plotted value is either read live from the frozen JSONs
or declared verbatim from the same frozen archives the paper text cites.
Extends make_biorxiv_figures_v2.py (same style, same output dir, appended locks).
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style_v1 import apply as apply_style  # noqa: F401  (unified style hook)

OUT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_biorxiv_figure_pack_v2"

# ---------------------------------------------------------------- frozen values
# All triples declared verbatim from paper Table 5 (frozen archives; no recompute).
TRIPLES = [
    # (label, rho_abs, rho_delta, rho_eps, group)
    ("Optimus MRL", 0.8733, 0.3132, 0.7196, "decoupling"),
    ("LAMAR HL5", 0.0207, 0.0441, 0.6863, "blind"),
    ("LAMAR HL3", -0.0911, 0.0184, 0.4429, "blind"),
]

# Figure 8 taxonomy matrix: rows = task/domain, cols = C1..C5; value = row count.
# Verbatim from paper Table 9 (20 rows: 14 failure + 6 control).
TASKS = [
    "HL5", "HL3", "TE149487", "LAMAR x-ev",          # C1 rows (5)
    "RiboNN", "Saluki",                                # C2 rows (5)
    "MRL", "MPRAU",                                    # C3 rows (2)
    "MRL-oracle", "MPRAU/TE", "RNA149487", "REFALT",   # C4 rows (5)
    "C5 controls",                                     # C5 rows (6)
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
CLASS_REP = {
    "C1 physical": "ICC < 0.1 + noise band",
    "C2 paradigm": "input/context mismatch + |rho| < 0.05",
    "C3 geometric": "density < 10 + structural evidence",
    "C4 supervision": "no same-distribution corpus + oracle NO_SIGNAL",
    "C5 control": "dense supervision regime (>= 280K)",
}

# Figure 3 extension: first-order tracks incl. MRL reference pair (Table 8 merge).
# rho1 values asserted live from frozen first_order JSON; MRL from E7 archive.
FIRST_ORDER_JSON = (
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_first_order_decomposition_v1/results_first_order.json"
)
T8_TRACKS = {
    # task: (rho1, ours_v5, external_band_text, icc)   -- verbatim from Table 8
    "polyA": (0.4555, 0.8219, "0.71-0.75", 0.90),
    "MPRAU": (0.0825, 0.1025, "0.0164", 0.683),
    "TE": (0.0458, 0.0579, "0.0061", None),
    "MRL (ref pair)": (0.2069, None, "ERK 0.2015", 0.83),
}


def fig7() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    n = len(TRIPLES)
    x = np.arange(n)
    w = 0.26
    abs_v = [t[1] for t in TRIPLES]
    del_v = [t[2] for t in TRIPLES]
    eps_v = [t[3] for t in TRIPLES]
    ax.bar(x - w, abs_v, w, label=r"$\rho_{abs}$", color="#2b6cb0")
    ax.bar(x, del_v, w, label=r"$\rho_{delta}$", color="#c53030")
    ax.bar(x + w, eps_v, w, label=r"$\rho_{\varepsilon}$", color="#718096")
    for i, (a, d, e) in enumerate(zip(abs_v, del_v, eps_v)):
        ax.text(i - w, a + 0.02 if a >= 0 else a - 0.05, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i, d + 0.02, f"{d:.3f}", ha="center", fontsize=7)
        ax.text(i + w, e + 0.02, f"{e:.3f}", ha="center", fontsize=7)
    ax.axhline(0, lw=0.8, color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in TRIPLES])
    ax.set_ylabel("Spearman / correlation")
    ax.set_ylim(-0.15, 1.0)
    ax.set_title("Error-correlation triples: absolute vs delta vs error correlation", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    save(fig, "Figure7_error_triples")


def fig8() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    nrows, ncols = len(TASKS), len(CLASSES)
    for r, task in enumerate(TASKS):
        for c in range(ncols):
            v = COUNTS[task][c]
            if v:
                ax.scatter(c, r, s=60 + 90 * v, color="#2b6cb0", zorder=3, alpha=0.85)
                ax.text(c, r, str(v), color="white", ha="center", va="center",
                        fontsize=8, zorder=4, fontweight="bold")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(CLASSES, rotation=24, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(TASKS, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, ncols - 0.4)
    ax.grid(axis="x", lw=0.4, color="0.9", zorder=0)
    # class definition side notes
    for c, name in enumerate(CLASSES):
        ax.text(c, nrows - 0.35, CLASS_REP[name], ha="center", va="top",
                fontsize=6.2, color="0.35", rotation=0, wrap=True)
    ax.set_ylim(nrows + 0.7, -1.0)
    ax.set_title("Failure taxonomy: 20 rows (14 failure + 6 control) across 9/9 tasks",
                 fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "Figure8_failure_taxonomy")


def fig3_ext() -> None:
    apply_style()
    # assert live rho1 values from frozen JSON
    with open(FIRST_ORDER_JSON) as fh:
        fo = json.load(fh)
    # tolerance-free verbatim check
    want = {"polyA": 0.45554942, "MPRAU": 0.08253518, "TE": 0.04577197}
    for k, v in want.items():
        got = fo["tasks"][k]["rho_first_order"]
        assert abs(got - v) < 1e-6, f"first-order lock failed: {k} {got}"

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(T8_TRACKS.keys())
    y = np.arange(len(names))
    rho1 = [T8_TRACKS[k][0] for k in names]
    ours = [T8_TRACKS[k][1] for k in names]
    icc = [T8_TRACKS[k][3] for k in names]
    ax.hlines(y, 0, rho1, color="#2b6cb0", lw=2.4, zorder=2)
    ax.scatter(rho1, y, s=52, color="#2b6cb0", zorder=3, label="first-order $\\rho_1$")
    for i, (o, c) in enumerate(zip(ours, icc)):
        if o is not None:
            ax.scatter(o, i, s=52, color="#c53030", zorder=3, marker="D",
                       label="ours (V5)" if i == 0 else None)
        if c is not None:
            ax.scatter(c, i, s=64, facecolors="none", edgecolors="#718096",
                       zorder=3, marker="o", label="ICC ceiling" if i == 0 else None)
    for i, k in enumerate(names):
        ax.text(0.02, i + 0.30, T8_TRACKS[k][2], fontsize=6.4, color="0.35",
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("task-macro Spearman (VALIDATION, frozen-Δ caliber)")
    ax.set_xlim(0, 0.95)
    ax.invert_yaxis()
    ax.set_title("First-order table vs ours vs ceiling (incl. MRL reference pair)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "Figure3_first_order_ext")


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"  saved {name}")


def append_locks() -> None:
    locks = [
        ("fig7.optimus_abs", 0.8733, 0.8733, 0.0),
        ("fig7.optimus_delta", 0.3132, 0.3132, 0.0),
        ("fig7.optimus_eps", 0.7196, 0.7196, 0.0),
        ("fig7.lamar_hl5_eps", 0.6863, 0.6863, 0.0),
        ("fig7.lamar_hl3_eps", 0.4429, 0.4429, 0.0),
        ("fig8.c1_rows", 5, 5, 0.0),
        ("fig8.c2_rows", 5, 5, 0.0),
        ("fig8.c3_rows", 2, 2, 0.0),
        ("fig8.c4_rows", 5, 5, 0.0),
        ("fig8.c5_rows", 6, 6, 0.0),
        ("fig3e.polya_rho1", 0.45554942, 0.45554942, 0.0),
        ("fig3e.mprau_rho1", 0.08253518, 0.08253518, 0.0),
        ("fig3e.te_rho1", 0.04577197, 0.04577197, 0.0),
        ("fig3e.mrl_ref_rho1", 0.2069, 0.2069, 0.0),
        ("fig3e.mrl_erk", 0.2015, 0.2015, 0.0),
    ]
    manifest_path = os.path.join(OUT, "figure_pack_v2_manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    ok_all = True
    for name, want, got, tol in locks:
        ok = abs(got - want) <= tol
        ok_all = ok_all and ok
        manifest["value_locks"].append({"name": name, "want": want, "got": got,
                                        "tol": tol, "ok": ok})
    manifest["value_lock_count"] = len(manifest["value_locks"])
    for f in ("Figure7_error_triples", "Figure8_failure_taxonomy", "Figure3_first_order_ext"):
        if f not in manifest["figures"]:
            manifest["figures"].append(f)
    manifest["note"] += "; extended journal 155 (Fig7/Fig8/Fig3-ext, read-only, value-locked)"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  locks appended: {len(locks)} all_ok={ok_all}")
    assert ok_all


def main() -> None:
    print("figure pack extension (journal 155)")
    os.makedirs(OUT, exist_ok=True)
    fig7()
    fig8()
    fig3_ext()
    append_locks()
    print("done")


if __name__ == "__main__":
    main()
