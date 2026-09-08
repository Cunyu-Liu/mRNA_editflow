#!/usr/bin/env python3
"""Baseline Task 8.1: consolidated bottom-line adjudication (Phase 2 closure).

Spec requirement "底线闭合判定（Phase 2）": 9-task full-coverage adjudication
(external target + control internal target) + macro + generation-line
closed-form, with per-row seeds status, CI status, and Holm correction status.
Task 8.2 (gap list + W ladder) is compiled alongside — the W ladder already
ran to terminal (W0/W1'/W2/W4/W-gen), so gaps map to the amendment options.

All numbers are previously adjudicated on-file products (frozen evaluator,
VALIDATION, FINAL-PASS/EPOCH-FIXED); this script compiles the consolidated
table with no new computation. Sources: holm_bottom_line.json (Task 12.1),
power_statement_v1.json (Task 12.2), frozen_delta_results.json family,
task6 leaderboard freeze, Table 4 v1 + closed-NDCG (P0-3/3b), V9 terminal
adjudications (batches 63/64).

Output: experiments/analysis_task8_bottomline_20260909/bottomline_adjudication_v1.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_task8_bottomline_20260909"

ROWS = [
    # task, ours (method, value, seeds), strongest external (method, value),
    # internal target, verdict, holm, power
    {
        "task": "MRL (GSE114002)",
        "ours": {"method": "V9-1a 3-seed ensemble (z-mean)", "spearman": 0.3217, "seeds": "3/3 protocol met; Route A full-FT V2 3-seed 0.3158 concordant"},
        "external": {"method": "frozen-Optimus", "spearman": 0.3132},
        "internal_target": 0.1192,
        "ci_status": "exact bootstrap Δ+0.0094 CI [-0.050, +0.070] crosses zero",
        "holm": "p=0.767 / holm_p=1.0 (not significant)",
        "power": "MDE_z 0.147 at n=730 — tie consistent with effects up to ~0.15",
        "verdict": "TIE-POINT-AHEAD (statistically indistinguishable; never claim 'no difference')",
    },
    {
        "task": "polyA (GSE269595)",
        "ours": {"method": "critic V5 (frozen terminal)", "spearman": 0.8219, "seeds": "single-run frozen; V9-1a 3-seed 0.8049-0.8633 + V8 h_bench9 0.8067 concordant (4 independent trainings)"},
        "external": {"method": "APARENT frozen-delta", "spearman": 0.7343},
        "internal_target": 0.7308,
        "ci_status": "Δ+0.088 CI [0.069, 0.120] excludes zero",
        "holm": "raw_p=0.001 / holm_p=0.004 — SIGNIFICANT (family-wise)",
        "power": "~100% at n=2,628 (only non-limited row)",
        "verdict": "PASS — the single family-wise significant win (91% of 0.90 ceiling)",
    },
    {
        "task": "MPRAU (ENCSR854RUF)",
        "ours": {"method": "s_mprau_in 5-seed ensemble (in-house)", "spearman": 0.1351, "seeds": "5/5 seeds > V5 0.1025; first in-house row above V5"},
        "external": {"method": "Saluki frozen (weak-control clause)", "spearman": 0.1205},
        "internal_target": 0.1025,
        "ci_status": "vs Saluki: Δ+0.0032 CI [-0.055, +0.062] crosses zero (exact bootstrap); vs V5 historical Δ+0.0325 CI [-0.023, +0.088] crosses zero",
        "holm": "p=0.895 / holm_p=1.0 (not significant)",
        "power": "MDE_z 0.088 at n=2,008 — near-band (+0.035) unverifiable at this pool size",
        "verdict": "POINT-AHEAD / POWER-LIMITED (D5 near band not met: CI crosses; far band learning-side blocked — 15.6 readiness report)",
    },
    {
        "task": "TE (GSE200304)",
        "ours": {"method": "critic V5", "spearman": 0.0579, "seeds": "single-run frozen"},
        "external": {"method": "LLR family (NT-v2/HyenaDNA) ~0", "spearman": 0.0},
        "internal_target": -0.0266,
        "ci_status": "ci-derived (stored adjudication win)",
        "holm": "ci-derived row (no exact test — no per-record external predictions persisted)",
        "power": "67% at n=1,614 — borderline limited; stored CI governs",
        "verdict": "PASS-DIRECTIONAL (beats internal target; no task-specific external model exists — RiboNN weights BLOCKED_ON_USER_TRANSFER)",
    },
    {
        "task": "GSE149487-TE (n=48)",
        "ours": {"method": "critic V5", "spearman": 0.1953, "seeds": "single-run frozen"},
        "external": {"method": "UTR-LM frozen -0.028 / RNA-FM frozen -0.015", "spearman": -0.015},
        "internal_target": 0.1747,
        "ci_status": "ci-derived win (+0.021)",
        "holm": "ci-derived; power-limited",
        "power": "5% at n=48 (MDE 0.59) — directional only",
        "verdict": "PASS-DIRECTIONAL / POWER-LIMITED (n=48 standard clause)",
    },
    {
        "task": "GSE149487-RNA (n=48)",
        "ours": {"method": "critic V5", "spearman": 0.0500, "seeds": "single-run frozen"},
        "external": {"method": "RNA-FM frozen", "spearman": 0.2958},
        "internal_target": 0.2230,
        "ci_status": "registered loss (-0.246)",
        "holm": "raw_p=0.30 / holm_p=0.9",
        "power": "23% at n=48; n~245 needed to confirm the loss",
        "verdict": "FAIL-DIRECTIONAL (registered loss to RNA-FM frozen AND internal target; small-sample clause; flagged as our gap row)",
    },
    {
        "task": "GSE186455 (n=274)",
        "ours": {"method": "critic V5", "spearman": 0.0639, "seeds": "single-run frozen"},
        "external": {"method": "RNA-FM frozen", "spearman": 0.1043},
        "internal_target": -0.0052,
        "ci_status": "win vs internal (+0.069); loss vs RNA-FM (-0.040)",
        "holm": "ci-derived; power-limited",
        "power": "8-13% at n=274",
        "verdict": "MIXED-DIRECTIONAL (beats internal target; below best external frozen row; both power-limited)",
    },
    {
        "task": "HALF_LIFE 5'/3'UTR (GSE217518)",
        "ours": {"method": "critic V5", "spearman": 0.0, "seeds": "—"},
        "external": {"method": "Saluki frozen 0.0985 (3UTR) / LAMAR 0.018 (3UTR), 0.044 (5UTR) / LM frozen ≈0 / LLR ≈0", "spearman": 0.0985},
        "internal_target": 0.0,
        "ci_status": "label ICC 0.001-0.013 (measurement repeatability floor)",
        "holm": "registered-unlearnable — no significance test (spec D2 attribution clause)",
        "power": "not applicable — the constraint is the label, not n",
        "verdict": "UNLEARNABLE (registered; 7-way external evidence incl. 3 official in-domain specialists — absolute endpoint learnable ≠ variant-differential learnable)",
    },
    {
        "task": "Macro (9 tasks)",
        "ours": {"method": "critic V5", "spearman": 0.1671, "seeds": "single-run frozen; V9-1a 3-seed mean 0.1916 > 0.167"},
        "external": {"method": "—", "spearman": None},
        "internal_target": 0.1317,
        "ci_status": "ci-derived win (+0.035)",
        "holm": "ci-derived",
        "power": "macro row — F9 dilution caveat on file",
        "verdict": "PASS-DIRECTIONAL (V5 macro beats global_scaled internal target; V9-1a 3-seed 0.192 exceeds it further)",
    },
]

GENERATION = {
    "calibre": "closed-form recovery@10 (Table 4 v1 + P0-3b closed NDCG@10 hit-set)",
    "rows": [
        {"method": "unguided", "recovery_at_10": 0.12046, "closed_ndcg_at_10": 0.1180},
        {"method": "V5-guided (B2)", "recovery_at_10": 0.12626, "closed_ndcg_at_10": 0.1240, "gate": "B2 FAIL: Δ+0.0058 CI crosses zero"},
        {"method": "V8-S specialist guided (Stage 3)", "recovery_at_10": 0.12495, "closed_ndcg_at_10": 0.1187, "gate": "FAIL: Δ CI crosses zero; SetFlow V6 project vetoed"},
        {"method": "V8-S joint guided (Stage 3)", "recovery_at_10": 0.12336, "closed_ndcg_at_10": 0.0273, "gate": "FAIL"},
        {"method": "TreeG (reversed selection)", "recovery_at_10": 0.0045, "gate": "FAIL (reversed)"},
    ],
    "beta_sweep": "SetFlow V6 Phase C C2 in flight (5-β x calib100; amendment B-tier gate Δsc-hit@1 + Δsupport) — final guided row pending its terminal",
    "verdict": "NO GUIDED CONFIGURATION PASSES the B2 original gate; 76% zero-hit source dominance = fourth independent evidence of the coverage constraint; zero-training task-routed committee (M3) 0.0715 probe-side is the only configuration above V5 single 0.0614 on the probe calibre",
}

W_LADDER = {
    "status": "executed to terminal before this adjudication (spec order inverted in practice)",
    "W0": "from-scratch 0.1987 (MRL) — architecture extracts more than Optimus arch but ceiling is the prior",
    "W1'": "LoRA 0.1486 / head-only 0.1336 — domain multitask init does not close MRL gap (gap = 280K external prior)",
    "W2": "Route A full-FT V2 3-seed ensemble 0.3158 — statistical tie with frozen-Optimus, point ahead",
    "W4": "V7 loss-mechanism 0.0766 MPRAU — negative, closed",
    "W-gen": "B2 FAIL (Δ+0.0058 CI crossing); Stage 3 double-arm FAIL; β sweep in flight",
    "V9": "adapter-zoo: seesaw broken 3-seed stable (MRL 0.322/polyA 0.84-0.86/macro 0.192) but gate-1 off-manifold FAIL 3/3; data arm D1/D2/D3 terminal (joint-constraint conclusion)",
}

GAPS_8_2 = [
    "GSE149487-RNA: single directional loss row (n=48) — our only per-task loss; needs n~245 to confirm or a stronger in-house row (V9-1a did not lift it)",
    "GSE186455: below best external frozen (RNA-FM 0.1043 vs 0.0639) — power-limited, directional",
    "MPRAU significance: s_mprau_in 0.1351 CI crosses zero — near band unverifiable at n=2,008; far band blocked on learning-side data (15.6)",
    "HALF_LIFE: unlearnable (label ICC) — no closure possible on current data",
    "Generation line: no guided configuration passes B2; β sweep terminal pending",
    "MRL: family-wise tie (not a loss, not a significant win) — claims limited to 'point estimate ahead, statistically indistinguishable'",
]

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "route_a_v3_task8_bottomline_adjudication_v1",
        "protocol": "frozen evaluator, VALIDATION splits, FINAL-PASS/EPOCH-FIXED; per-row seeds/CI/Holm status reported as adjudicated on file (Task 12.1/12.2 products)",
        "rows": ROWS,
        "generation_line": GENERATION,
        "w_ladder": W_LADDER,
        "gaps_task_8_2": GAPS_8_2,
        "headline_summary": (
            "1 task family-wise significant (polyA, Holm p=0.004); 1 statistical tie point-ahead (MRL); "
            "1 point-ahead power-limited (MPRAU); 4 directional (TE-family wins, 1 registered loss row); "
            "1 unlearnable (HALF_LIFE, label-limited). Generation line: no guided config passes B2. "
            "Bottom line met on polyA; MRL tied at parity; remaining gaps are data-side (power/labels/coverage), "
            "not method-side — consistent with the three-body attribution chain."
        ),
    }
    (OUT / "bottomline_adjudication_v1.json").write_text(json.dumps(payload, indent=1))

    md = ["# Task 8.1 consolidated bottom-line adjudication (Phase 2 closure)", "",
          "| task | ours (seeds) | strongest external | internal target | Holm | verdict |",
          "|---|---|---|---|---|---|"]
    for r in ROWS:
        ours = f"{r['ours']['method']} {r['ours']['spearman']:.4f}" if r["ours"]["spearman"] is not None else r["ours"]["method"]
        ext = f"{r['external']['method']} {r['external']['spearman']:.4f}" if r["external"]["spearman"] is not None else "—"
        md.append(f"| {r['task']} | {ours} | {ext} | {r['internal_target']} | {r['holm']} | {r['verdict']} |")
    md += ["", "## Generation line (closed-form)", "",
           "| method | recovery@10 | closed NDCG@10 | gate |", "|---|---|---|---|"]
    for g in GENERATION["rows"]:
        md.append(f"| {g['method']} | {g.get('recovery_at_10', '—')} | {g.get('closed_ndcg_at_10', '—')} | {g.get('gate', '—')} |")
    md += ["", f"β sweep: {GENERATION['beta_sweep']}", "",
           "## W ladder (executed, terminal)", ""]
    for k, v in W_LADDER.items():
        if k != "status":
            md.append(f"- **{k}**: {v}")
    md += ["", "## Task 8.2 gap list", ""]
    md += [f"- {g}" for g in GAPS_8_2]
    md += ["", "## Headline", "", payload["headline_summary"], ""]
    (OUT / "bottomline_adjudication_v1.md").write_text("\n".join(md))
    print(payload["headline_summary"])
    print(f"wrote {OUT}/bottomline_adjudication_v1.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
