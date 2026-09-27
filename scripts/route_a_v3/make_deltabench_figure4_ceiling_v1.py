#!/usr/bin/env python3
"""DeltaBench Figure 4: ceiling-normalized completion by task (learnability map).

Read-only producer. Values are copied from frozen artifacts:
  - ours / external / internal: analysis_task8_bottomline_20260909/bottomline_adjudication_v1.json
  - label ICC ceilings: mechanism_results_v2.json -> label_icc_reference (via draft Table 3)
No recomputation. Tasks with ICC <= 0 are shown as N/A (no normalization possible).

Output: experiments/analysis_benchmark_v2_matrix_figure_v1/deltabench_ceiling_completion_v1.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RT2 = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
BOTTOMLINE = RT2 / "experiments/analysis_task8_bottomline_20260909/bottomline_adjudication_v1.json"
OUT = RT2 / "experiments/analysis_benchmark_v2_matrix_figure_v1"

# task label -> (ours, ICC ceiling) ; ICC values per frozen Table 3 / label_icc_reference
ROWS = [
    ("MRL (GSE114002)", 0.3217, 0.83),
    ("polyA (GSE269595)", 0.8219, 0.90),
    ("MPRAU (ENCSR854RUF)", 0.1351, 0.683),
    ("TE (GSE200304)", 0.0579, 0.586),
    ("REF/ALT (GSE186455)", 0.0639, 0.21),
    ("PLUMAGE-RNA (GSE149487)", 0.0500, 0.364),
    ("HALF_LIFE 5'/3'UTR", 0.0, None),      # ICC ~ 0.001-0.013 -> N/A
    ("PLUMAGE-TE (GSE149487)", 0.1953, None),  # ICC -0.274 -> N/A
]


def main() -> int:
    data = json.loads(BOTTOMLINE.read_text())
    by_task = {row["task"]: row for row in data["rows"]}
    # explicit display-label -> frozen bottomline task-key map (no fuzzy matching)
    key_map = {
        "MRL (GSE114002)": "MRL (GSE114002)",
        "polyA (GSE269595)": "polyA (GSE269595)",
        "MPRAU (ENCSR854RUF)": "MPRAU (ENCSR854RUF)",
        "TE (GSE200304)": "TE (GSE200304)",
        "REF/ALT (GSE186455)": "GSE186455 (n=274)",
        "PLUMAGE-RNA (GSE149487)": "GSE149487-RNA (n=48)",
        "PLUMAGE-TE (GSE149487)": "GSE149487-TE (n=48)",
        "HALF_LIFE 5'/3'UTR": "HALF_LIFE 5'/3'UTR (GSE217518)",
    }
    checks = {label: ours for label, ours, _ in ROWS}
    for label, ours in checks.items():
        key = key_map[label]
        assert key in by_task, f"missing bottomline row for {label} -> {key}"
        frozen = by_task[key]["ours"]["spearman"]
        assert abs(frozen - ours) < 5e-5, f"{label}: frozen {frozen} vs figure {ours}"

    labeled = [(t, o, icc) for t, o, icc in ROWS if icc is not None]
    labeled.sort(key=lambda x: x[1] / x[2])
    na_rows = [(t, o) for t, o, icc in ROWS if icc is None]

    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=300)
    labels = [t for t, _, _ in labeled]
    completion = [100.0 * o / icc for _, o, icc in labeled]
    bars = ax.barh(range(len(labeled)), completion, color="#2b6cb0", height=0.62)
    for i, (task, ours, icc) in enumerate(labeled):
        ax.text(completion[i] + 1.2, i, f"{completion[i]:.1f}%  ({ours:.4g}/{icc:.3g})",
                va="center", fontsize=7.6)
    ax.set_yticks(range(len(labeled)))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("ceiling-normalized completion (ours / label ICC, %)", fontsize=8.6)
    ax.set_xlim(0, 100)
    ax.axvline(60, color="0.35", linestyle="--", linewidth=0.9)
    ax.text(61, 1.4, "Tier-A 60% reference", fontsize=7.2, color="0.3",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_title(
        "DeltaBench learnability map: label-ceiling completion by task\n"
        "(VALIDATION; ours = best in-house row; ICC = split-half / meta-analytic label reliability)",
        fontsize=9.6,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    ax.text(0.005, -0.30,
            "N/A (no normalization): " + "; ".join(f"{t} (ours {o:.4g}, ICC \u2248 0 or negative)" for t, o in na_rows),
            transform=ax.transAxes, fontsize=7.0, color="0.25")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "deltabench_ceiling_completion_v1.png", bbox_inches="tight")
    fig.savefig(OUT / "deltabench_ceiling_completion_v1.pdf", bbox_inches="tight")
    manifest = {
        "schema_version": "deltabench_ceiling_completion_v1",
        "source_bottomline": str(BOTTOMLINE),
        "source_icc": "mechanism_results_v2.json -> label_icc_reference",
        "rows": [{"task": t, "ours": o, "icc": icc, "completion_pct": 100.0 * o / icc if icc else None}
                 for t, o, icc in ROWS],
        "note": "no recomputation; values copied from frozen artifacts and asserted against the bottomline rows",
    }
    (OUT / "deltabench_ceiling_completion_v1_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("rendered", len(labeled), "labeled rows + ", len(na_rows), "N/A rows ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())