#!/usr/bin/env python3
"""Baseline Task 4.2: consolidated closed-NDCG ranking table (Table 3/4 NDCG
columns) — frozen + fine-tuned + internal controls + generation line.

All values previously adjudicated on file; this compiles the single ranking
view the task asked to freeze at Task 6 time (blocked then on in-flight rows —
all terminal now). Sources:
- analysis_frozen_delta_gse114002_20260903 (MRL frozen NDCG@10)
- analysis_fullft_v2_adjudication_20260903 (Route A ensemble NDCG@10)
- analysis_baseline_k_sensitivity_20260908 (V5 per-task NDCG@10, R4-closed)
- analysis_task1_alignment_20260902 (APARENT polyA NDCG@10)
- analysis_baseline_table4_20260908 + table4_closed_ndcg_v1.json (generation line)

Output: experiments/analysis_task8_bottomline_20260909/closed_ndcg_ranking_v1.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments")
OUT = MNT / "analysis_task8_bottomline_20260909"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- prediction line: MRL (the multi-candidate task where NDCG is defined)
    mrl_frozen = json.load(open(MNT / "analysis_frozen_delta_gse114002_20260903/frozen_delta_results.json"))
    route_a = json.load(open(MNT / "analysis_fullft_v2_adjudication_20260903/ensemble_3seed_vs_optimus.json"))
    ks = json.load(open(MNT / "analysis_baseline_k_sensitivity_20260908/k_sensitivity.json"))

    mrl_rows = []
    for name, r in mrl_frozen["results"].items():
        mrl_rows.append({
            "mode": "frozen-Δ", "method": name, "task": "MRL",
            "ndcg_at_10": round(r["ndcg_at_10"], 4),
            "top_1": round(r["top_1"], 4),
            "spearman": round(r["task_macro_spearman"], 4),
        })
    mrl_rows.append({
        "mode": "fine-tuned (3-seed ensemble)", "method": "Route A full-FT V2", "task": "MRL",
        "ndcg_at_10": round(route_a["ensemble"]["ndcg_at_10"], 4),
        "top_1": round(route_a["ensemble"]["top_1"], 4),
        "spearman": round(route_a["ensemble"]["task_macro_spearman"], 4),
    })
    v5_mrl = ks["rows"]["critic_v5__mrl_gse114002"]["K=10"]
    mrl_rows.append({
        "mode": "critic (frozen terminal)", "method": "critic V5", "task": "MRL",
        "ndcg_at_10": round(v5_mrl["ndcg_at_k"], 4),
        "top_1": round(v5_mrl["top_1"], 4),
        "spearman": round(v5_mrl["task_macro_spearman"], 4),
    })
    mrl_rows.sort(key=lambda r: -r["ndcg_at_10"])

    # --- V5 per-task NDCG (where defined; TE family single-record-source => null)
    v5_rows = []
    for key, val in ks["rows"].items():
        if not key.startswith("critic_v5__"):
            continue
        task = key.replace("critic_v5__", "")
        k10 = val["K=10"]
        v5_rows.append({
            "task": task,
            "ndcg_at_10": None if k10["ndcg_at_k"] is None else round(k10["ndcg_at_k"], 4),
            "spearman": round(k10["task_macro_spearman"], 4),
            "note": "single-record-source task — NDCG undefined" if k10["ndcg_at_k"] is None else "",
        })

    # --- polyA external reference (APARENT, from task1 aligned eval)
    t1 = json.load(open(MNT / "analysis_task1_alignment_20260902/task1_alignment_results.json"))
    polya_external = None
    ae = t1.get("aligned_eval")
    if isinstance(ae, dict):
        for k, v in ae.items():
            if "aparent" in str(k).lower():
                polya_external = {"row": k, "detail": v}
                break

    # --- generation line (P0-3 + P0-3b)
    gen_closed = None
    p = MNT / "analysis_baseline_table4_20260908/table4_closed_ndcg_v1.json"
    if p.exists():
        gen_closed = json.load(open(p))

    payload = {
        "schema_version": "route_a_v3_task42_closed_ndcg_ranking_v1",
        "prediction_line_mrl": mrl_rows,
        "prediction_line_v5_per_task": v5_rows,
        "polya_external_reference": polya_external,
        "generation_line_closed_ndcg": gen_closed,
        "notes": [
            "NDCG@10 defined only for multi-candidate-per-source tasks (MRL 730, polyA 2,628, MPRAU, HL, GSE186455); TE-family tasks are single-candidate-per-source — ranking degenerate, column null by protocol (not missing data).",
            "MRL NDCG ordering (Optimus 0.8655 > FramePool 0.8647 > Route A ens 0.8624 > V5 0.8350) differs from Spearman ordering (V9-1a ens 0.3217 > Route A 0.3158 > Optimus 0.3132 > FramePool 0.2956 > V5 0.1354): decision-calibre vs association-calibre divergence is itself H2 material.",
            "Generation-line closed NDCG@10 (hit-set calibre): V5-guided 0.1240 > V8 specialist 0.1187 > unguided 0.1180 > V8 joint 0.0273.",
            "Task 4.2 closes with this table; deep-link values identical to their on-file sources (R4 K-sensitivity alignment already verified V5 rows).",
        ],
    }
    (OUT / "closed_ndcg_ranking_v1.json").write_text(json.dumps(payload, indent=1))

    md = ["# Task 4.2 consolidated closed-NDCG ranking (frozen + fine-tuned + critic + generation)", "",
          "## MRL (multi-candidate ranking task)", "",
          "| mode | method | NDCG@10 | top-1 | Spearman |", "|---|---|---|---|---|"]
    for r in mrl_rows:
        md.append(f"| {r['mode']} | {r['method']} | {r['ndcg_at_10']} | {r['top_1']} | {r['spearman']} |")
    md += ["", "## critic V5 per-task NDCG@10", "",
           "| task | NDCG@10 | Spearman | note |", "|---|---|---|---|"]
    for r in v5_rows:
        md.append(f"| {r['task']} | {r['ndcg_at_10'] if r['ndcg_at_10'] is not None else 'null'} | {r['spearman']} | {r['note']} |")
    md += ["", "## Generation line (closed NDCG@10, hit-set calibre)", ""]
    if gen_closed:
        md.append("```json\n" + json.dumps(gen_closed, indent=1)[:800] + "\n```")
    else:
        md.append("(see analysis_baseline_table4_20260908/table4_closed_ndcg_v1.json)")
    md += [""] + [f"- {n}" for n in payload["notes"]]
    (OUT / "closed_ndcg_ranking_v1.md").write_text("\n".join(md))
    print(f"MRL NDCG order: {' > '.join(r['method'] + ' ' + str(r['ndcg_at_10']) for r in mrl_rows)}")
    print(f"wrote {OUT}/closed_ndcg_ranking_v1.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
