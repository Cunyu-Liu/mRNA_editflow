#!/usr/bin/env python3
"""P1-1 (audit A4): M1/M6/S1 density dual-caliber sensitivity recompute.

Reads the frozen delta_vs_density_v2 cells (65 held-out rows), replaces ONLY
the M1/M6/S1 density inputs with alternative calibers (predeclared in
docs/paper/density_sensitivity_mini_prereg_v1.md), re-derives rho_predicted
with the frozen fit (slope_log10=0.366274, intercept=-0.085673) and recomputes
in-band flags for the 20 affected cells (45 other cells untouched).

Sensitivity-only: the original verdict (FAIL, in-band 33/65) is NOT changed.

Calibers (computed read-only from archived artifacts):
  M1  training-side full-pool pairing density (prefix-15 groups + global
      Hamming-1 fallback on all 1,128,562 kept rows):
      9,578 candidates / 9,538 sources = 1.0042 (vs eval-row 1.0014)
  M6  full 1,507 variant-key Family density: 1,507/763 = 1.9751
      (vs sampled 800/509 = 1.5717)
  S1  eligible 3,742 rows variant-ID-level gene x arm group density:
      3,717 variants / 1,519 groups = 2.4470 (vs sub-row 5,572/1,519 = 3.6682)

Outputs -> experiments/analysis_delta_vs_density_v2/sensitivity_density_caliber/
(results.json + comparison table md). CPU only, TEST reads = 0.
"""
from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
SRC = MNT / "experiments/analysis_delta_vs_density_v2/delta_vs_density_v2_results.json"
OUT = MNT / "experiments/analysis_delta_vs_density_v2/sensitivity_density_caliber"
OUT.mkdir(parents=True, exist_ok=True)

SLOPE_LOG10 = 0.366274
INTERCEPT = -0.085673
TOL = 0.1
GATE = 0.7

NEW_DENSITY = {
    "M1_MRL_EVAL_ROW|5UTR": {"density": 1.0042, "caliber": "TRAIN-side full-pool pairing (prefix-15 groups + global Hamming-1 fallback over all 1,128,562 kept rows): 9,578 candidates / 9,538 sources"},
    "M6_NDD5UTR_EVAL_ROW|5UTR": {"density": 1.9751, "caliber": "full 1,507 variant-key Family density: 1,507 keys / 763 families"},
    "S1_STABILITY_EVAL_ROW|3UTR": {"density": 2.4470, "caliber": "eligible 3,742 rows variant-ID-level gene x arm group density: 3,717 distinct variants / 1,519 groups"},
    "S1_STABILITY_EVAL_ROW|5UTR": {"density": 2.4470, "caliber": "same as 3UTR (row_manifest carries one shared density)"},
}
ROW_CELLS = ["M1_MRL_EVAL_ROW|5UTR", "M6_NDD5UTR_EVAL_ROW|5UTR", "S1_STABILITY_EVAL_ROW|3UTR", "S1_STABILITY_EVAL_ROW|5UTR"]


def main():
    d = json.load(open(SRC))
    cells = d["cells"]
    orig_in_band = sum(1 for c in cells if c["in_band"])
    changed = []
    per_cell = {}
    for c in cells:
        cell = c["cell"]
        row_key = "|".join(cell.split("|")[:2]) if any(cell.startswith(r.split("|")[0]) for r in ROW_CELLS) else None
        # match M1/M6/S1 rows by study-unit prefix
        target = None
        for rk in NEW_DENSITY:
            prefix = rk.split("|")[0]
            if cell.startswith(prefix + "|"):
                target = rk
                break
        key = cell + "|" + c["family"]
        if target is None:
            per_cell[key] = {"cell": cell, "family": c["family"], "rho_observed": c["rho_observed"], "density": c["density"], "rho_predicted": c["rho_predicted"], "in_band_orig": c["in_band"], "in_band_sens": c["in_band"], "changed": False}
            continue
        new_d = NEW_DENSITY[target]["density"]
        import math
        rho_pred_new = SLOPE_LOG10 * math.log10(new_d) + INTERCEPT
        in_band_new = abs(c["rho_observed"] - rho_pred_new) <= TOL
        per_cell[key] = {
            "cell": cell,
            "family": c["family"],
            "rho_observed": c["rho_observed"],
            "density_orig": c["density"],
            "density_sens": new_d,
            "rho_predicted_orig": c["rho_predicted"],
            "rho_predicted_sens": round(rho_pred_new, 6),
            "in_band_orig": c["in_band"],
            "in_band_sens": bool(in_band_new),
            "flipped": bool(in_band_new != c["in_band"]),
        }
        if in_band_new != c["in_band"]:
            changed.append({"cell": cell, "family": c["family"], "direction": "OUT->IN" if in_band_new else "IN->OUT"})

    sens_in_band = sum(1 for v in per_cell.values() if v.get("in_band_sens"))
    n_flipped = len(changed)
    delta_rate = sens_in_band / 65 - orig_in_band / 65

    if abs(delta_rate) < 0.05 and sens_in_band / 65 < GATE:
        interp = "(a) FAIL verdict robust to caliber switch (in-band change <5pp, still below gate)"
    elif sens_in_band / 65 >= GATE:
        interp = "(c) caliber-dependent boundary case: sensitivity caliber passes gate BUT verdict unchanged per prereg sec.1 - flagged for main-session adjudication"
    else:
        interp = "(b) magnitude-sensitive: in-band change >=5pp but still below gate; downgrade clause remains in force"

    out = {
        "schema_version": "route_a_v3_density_caliber_sensitivity.v1",
        "task": "P1-1 (audit A4): M1/M6/S1 density dual-caliber sensitivity",
        "date": str(_date.today()),
        "prereg": "docs/paper/density_sensitivity_mini_prereg_v1.md (frozen before computation)",
        "discipline": "sensitivity-only report; original verdict (FAIL 33/65=50.77%) and downgrade clause NOT changed; frozen fit slope/intercept/tolerance/gate untouched; 45 non-M1/M6/S1 cells untouched",
        "frozen_fit": {"slope_log10": SLOPE_LOG10, "intercept": INTERCEPT, "tolerance_band": TOL, "pass_gate": GATE},
        "caliber_replacements": NEW_DENSITY,
        "original": {"in_band": orig_in_band, "n_cells": 65, "in_band_rate": orig_in_band / 65, "verdict": d["heldout"]["verdict"]},
        "sensitivity": {"in_band": sens_in_band, "n_cells": 65, "in_band_rate": round(sens_in_band / 65, 6), "n_flipped": n_flipped, "flipped_cells": changed, "delta_rate_pp": round(delta_rate * 100, 2)},
        "interpretation": interp,
        "per_cell": per_cell,
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    lines = [
        "# 密度双口径敏感性（P1-1，审计 A4）",
        "",
        "Prereg: docs/paper/density_sensitivity_mini_prereg_v1.md（判定以原口径为准，本分析只作敏感性报告）。",
        "",
        f"| 口径 | in-band | 率 | verdict |",
        f"|---|---|---|---|",
        f"| 原口径（row_manifest eval-row） | {orig_in_band}/65 | {orig_in_band/65:.4f} | FAIL（冻结不动） |",
        f"| 敏感性口径（M1 训练配对 1.0042 / M6 全量 1.9751 / S1 变异ID级 2.4470） | {sens_in_band}/65 | {sens_in_band/65:.4f} | 敏感性读数（不改判） |",
        "",
        f"翻转格数：{n_flipped}" + (f"（{'；'.join(c['cell']+' '+c['family']+' '+c['direction'] for c in changed)}）" if changed else ""),
        "",
        "解读：" + interp,
        "",
        "## 20 个受影响格明细（M1 5 / M6 5 / S1 10）",
        "",
        "| cell | family | rho_obs | density_orig→sens | rho_pred_orig→sens | in_band orig→sens |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in per_cell.items():
        if "density_sens" in v:
            lines.append(f"| {v['cell']} | {v['family']} | {v['rho_observed']:.4f} | {v['density_orig']}→{v['density_sens']} | {v['rho_predicted_orig']:.4f}→{v['rho_predicted_sens']:.4f} | {v['in_band_orig']}→{v['in_band_sens']} |")
    with open(OUT / "comparison_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved:", OUT / "results.json")
    print("saved:", OUT / "comparison_table.md")
    print(f"orig in-band {orig_in_band}/65 -> sens in-band {sens_in_band}/65 (flipped {n_flipped}); {interp}")


if __name__ == "__main__":
    main()
