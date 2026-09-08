#!/usr/bin/env python3
"""Task 11.1 / 12.3 LOSO-lite Table 5 draft (interpretation B material).

ZERO-INFERENCE compilation of on-file per-study numbers (no protocol
invention; the A/B adjudication remains a user decision). Rows:
  - critic V5 multitask per-study VALIDATION (in-domain, NOT zero-shot —
    wording clause from the 2026-09-04 clarification memo applies)
  - external same-pool frozen rows where they exist (3 studies)
  - internal control targets (9 tasks)
  - ceiling column where a frozen product exists

This draft exists so that EITHER adjudication outcome can proceed:
  A (true LOSO retrain) -> this table becomes the reference/companion;
  B (lite composition)  -> this table IS Table 5 with corrected wording.

Output: experiments/analysis_task11_loso_lite_20260909/table5_loso_lite_draft_v1.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_task11_loso_lite_20260909"

# per-study critic rows (V5 multitask VALIDATION, from task1 alignment / k-sensitivity products)
CRITIC_V5 = {
    # study: (task label, V5 spearman, source)
    "GSE114002": ("MRL", 0.1354, "k_sensitivity critic_v5__mrl"),
    "GSE269595": ("polyA", 0.8219, "k_sensitivity critic_v5__polya"),
    "ENCSR854RUF": ("MPRAU", 0.0732, "k_sensitivity critic_v5__mprau (cell) / pair-mean 0.1025"),
    "GSE217518-5UTR": ("HALF_LIFE 5UTR", 0.0607, "k_sensitivity critic_v5__half_life_5utr"),
    "GSE217518-3UTR": ("HALF_LIFE 3UTR", 0.0456, "k_sensitivity critic_v5__half_life_3utr"),
    "GSE200304": ("TE", 0.0579, "k_sensitivity critic_v5__gse200304_te"),
    "GSE149487-TE": ("TE (PLUMAGE)", 0.1953, "k_sensitivity critic_v5__gse149487_te"),
    "GSE149487-RNA": ("RNA", 0.0500, "k_sensitivity critic_v5__gse149487_rna"),
    "GSE186455": ("REF/ALT", 0.0639, "k_sensitivity critic_v5__gse186455"),
}

EXTERNAL_SAME_POOL = {
    "GSE114002": [
        ("frozen-Optimus", 0.3132), ("FramePool", 0.2956), ("UTR-LM frozen", 0.1107),
        ("RNA-FM frozen", 0.122), ("matched-FT Optimus", 0.2977), ("matched-FT FramePool", 0.2300),
        ("V9-1a 3-seed ens (ours-2)", 0.3217), ("Route A full-FT ens (ours-2)", 0.3158),
    ],
    "GSE269595": [
        ("APARENT frozen", 0.7343), ("APARENT2 frozen", 0.6810), ("UTR-LM frozen", 0.7490),
        ("RNA-FM frozen", 0.7114),
    ],
    "ENCSR854RUF": [
        ("Saluki frozen (weak-control)", 0.1205), ("RNA-FM matched-FT", -0.0747),
        ("UTR-LM matched-FT", -0.1066), ("CMS prior (3 arms)", 0.033),
        ("NT-v2 LLR", 0.0183), ("HyenaDNA LLR", 0.0450),
        ("s_mprau_in 5-seed ens (ours-2)", 0.1351),
    ],
    "GSE217518": [
        ("Saluki frozen 3UTR", 0.0985), ("Saluki frozen 5UTR", 0.0193),
        ("LAMAR frozen 3UTR", 0.0184), ("LAMAR frozen 5UTR", 0.0441),
        ("UTR-LM frozen", -0.0199), ("RNA-FM frozen", 0.0271),
        ("Saluki-FT 5UTR (matched)", 0.0489), ("Saluki-FT 3UTR (matched)", -0.0160),
    ],
    # GSE200304/GSE149487/GSE186455: no external same-pool row (frozen-Δ full coverage rows exist but are cross-study generalist probes; per protocol not pre-filled)
}

INTERNAL_TARGETS = {
    "GSE114002": 0.1192, "GSE269595": 0.7308, "ENCSR854RUF": 0.0248,
    "GSE217518": 0.0, "GSE200304": -0.0266, "GSE149487-TE": 0.1747,
    "GSE149487-RNA": 0.2230, "GSE186455": -0.0052,
}

CEILINGS = {
    "GSE114002": 0.83, "GSE269595": 0.90, "ENCSR854RUF": 0.683,
    "GSE217518": "ICC≈0.001-0.013 (unlearnable)",
    # MRL/polyA/MPRAU/HL ceilings frozen; TE-family ceilings absent on file (honest blank)
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for study, (task, v5, src) in CRITIC_V5.items():
        key = "GSE217518" if study.startswith("GSE217518") else (
            "GSE149487" if study.startswith("GSE149487") else study)
        ext = EXTERNAL_SAME_POOL.get(key) or []
        ext_str = "; ".join(f"{n} {v}" for n, v in ext) if ext else "无外部同池行（协议不预填）"
        rows.append({
            "study": study, "task": task,
            "critic_v5_in_domain": v5,   # wording clause: IN-DOMAIN, not zero-shot
            "external_same_pool": ext_str,
            "internal_target": INTERNAL_TARGETS.get(key if key in INTERNAL_TARGETS else study,
                                                   INTERNAL_TARGETS.get(study)),
            "ceiling": CEILINGS.get(key if key in CEILINGS else study, "—"),
            "note": src,
        })

    payload = {
        "schema_version": "route_a_v3_task11_loso_lite_table5_draft_v1",
        "status": "DRAFT pending user A/B adjudication (2026-09-04 clarification memo)",
        "wording_clause": "critic 列为多任务 in-domain VALIDATION（非 zero-shot）；若裁决为解释 B，Table 5 表头必须写 in-domain 而非 zero-shot（澄清备忘措辞陷阱条款）",
        "interpretation_A_note": "true LOSO 7-fold retrain (42-job infra, loso_folds.jsonl frozen) — not executed; if adjudicated, ~2-3 GPU-days",
        "interpretation_B_note": "this composition IS Table 5 (zero inference); multi-task gain column = per-study V5 vs best external same-pool",
        "rows": rows,
        "multi_task_gain_column": {
            r["study"]: {
                "v5": r["critic_v5_in_domain"],
                "best_external": (max(v for _, v in EXTERNAL_SAME_POOL[r["study"].split("-")[0]])
                                  if r["study"].split("-")[0] in EXTERNAL_SAME_POOL else None),
                "note": "R2 fairness response: multitask gain only meaningful where an external same-pool row exists",
            } for r in rows
        },
    }
    (OUT / "table5_loso_lite_draft_v1.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = ["# Table 5 LOSO-lite draft（解释 B 素材，A/B 裁决前预编制）", "",
          "**措辞条款**：critic 列 = 多任务 in-domain VALIDATION（非 zero-shot）。", "",
          "| study | task | critic V5 (in-domain) | 外部同池行 | 内靶 | 天花板 |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['study']} | {r['task']} | {r['critic_v5_in_domain']} | {r['external_same_pool'][:80]}{'…' if len(r['external_same_pool'])>80 else ''} | {r['internal_target']} | {r['ceiling']} |")
    md += ["", "多任务增益列（R2 回应）：仅在有外部同池行的 study 有意义（MRL/polyA/MPRAU/HL）。",
           "", "裁决路径：A = 真 LOSO 7 折重训（42-job 基础设施在档，~2-3 GPU·天）；B = 本表即 Table 5（表头改 in-domain）。"]
    (OUT / "table5_loso_lite_draft_v1.md").write_text("\n".join(md))
    print(f"{len(rows)} rows composed; wrote {OUT}/table5_loso_lite_draft_v1.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
