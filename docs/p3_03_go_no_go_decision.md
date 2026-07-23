# P3-03: GO / PARTIAL / NO-GO Decision Framework

**Phase:** P3-03
**Status:** FROZEN (pre-registered)
**Date:** 2026-07-23
**Depends on:** docs/p3_03_experiment_preregistration.md, docs/p3_03_analysis_plan.md

---

## Purpose

This document defines the **deterministic decision criteria** for translating P3-03 wet-lab analysis results into a GO, PARTIAL, or NO-GO verdict. The criteria are frozen before wet-lab execution and cannot be modified post-hoc.

---

## Decision Matrix

### GO (Proceed to P3-04 full-scale GRPO)

All four conditions must be met:

| # | Condition | Metric | Threshold | Source Analysis |
|---|-----------|--------|-----------|-----------------|
| G1 | Top-ranked edits significantly outperform random on ≥1 cargo | top_k_enrichment OR guided_vs_random_ttest | enrichment > 1.5 OR ttest_p < 0.05 | Analysis 3 |
| G2 | Two cargos' overall direction not systematically opposite | per_cargo_sign_agreement | ≥ 1 cargo has sign_accuracy > 0.60 AND cargos agree on direction (both positive or both neutral) | Analysis 2 |
| G3 | Oracle enriches measured beneficial edits | beneficial_precision | > 0.50 (better than random) | Analysis 2 |
| G4 | Substantial local-edit headroom exists | max_region_measured_delta | |measured_delta| > 0.1 × WT_value for ≥1 region arm | Analysis 5 |

**GO → P3-04 (full-scale GRPO) unblocked.**

---

### PARTIAL (Shrink primary task, proceed with reduced scope)

Either of the following:

| # | Condition | Interpretation |
|---|-----------|----------------|
| P1 | Only 5′UTR OR only CDS effective (not both) | Shrink to effective region only |
| P2 | Only effective on single cargo (not both) | Shrink to in-distribution cargo only |

**PARTIAL → P3-04 with reduced scope:**
- If P1: restrict GRPO action space to effective region only
- If P2: restrict GRPO to in-distribution cargo family; do not claim cross-cargo generalization
- Update P3-00A frozen contract via amendment procedure

---

### NO-GO (Pause RL, retrain Oracle or expand intervention data)

Any of the following:

| # | Condition | Metric | Threshold | Source Analysis |
|---|-----------|--------|-----------|-----------------|
| N1 | Predicted improvements do not transfer | delta_spearman | ρ < 0.2 (pooled) | Analysis 1 |
| N2 | Adversarial high-score sequences perform poorly | hacking_flag | True (predicted > 0, measured ≤ 0) | Analysis 6 |
| N3 | Random edits indistinguishable from model edits | guided_vs_random_ttest | p > 0.20 (no signal) | Analysis 3 |
| N4 | Model ranking ≈ random | sign_accuracy | < 0.50 (worse than coin flip) | Analysis 2 |

**NO-GO → RL paused. Required remediation:**
1. Retrain oracle with expanded intervention data (P3-01 expansion)
2. Investigate feature engineering (GC/length over-reliance)
3. Re-run P3-02 with improved oracle
4. Re-attempt P3-03 before P3-04

---

## Decision Flow

```
                    ┌─────────────────────────┐
                    │  P3-03 Analysis Complete │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Check NO-GO conditions  │
                    │  (N1 OR N2 OR N3 OR N4)  │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
              YES  │                      NO   │
          ┌────────▼────────┐    ┌─────────────▼─────────────┐
          │  NO-GO          │    │  Check PARTIAL conditions  │
          │  → Pause RL     │    │  (P1 OR P2)                │
          │  → Retrain      │    └─────────────┬─────────────┘
          │    Oracle       │                  │
          └─────────────────┘     ┌────────────┴────────────┐
                                  │                         │
                             YES  │                    NO   │
                         ┌────────▼────────┐   ┌────────────▼────────────┐
                         │  PARTIAL        │   │  Check GO conditions     │
                         │  → Shrink scope │   │  (G1 AND G2 AND G3 AND G4│
                         │  → P3-04 reduced│   └────────────┬────────────┘
                         └─────────────────┘                │
                                                   ┌─────────┴─────────┐
                                                   │                   │
                                              ALL  │              ANY FAIL
                                                   │                   │
                                          ┌────────▼────────┐  ┌───────▼───────┐
                                          │  GO             │  │  PARTIAL      │
                                          │  → P3-04 full   │  │  (conservative│
                                          │    GRPO         │  │   fallback)   │
                                          └─────────────────┘  └───────────────┘
```

**Conservative default:** If GO conditions are partially met (some pass, some fail) but NO-GO conditions are not triggered, default to PARTIAL.

---

## Tie-Breaking Rules

| Situation | Rule |
|-----------|------|
| G1 passes on EGFP but not mCherry | GO if G2-G4 pass (in-distribution is primary) |
| N2 (hacking) triggered but N1 not | NO-GO (reward hacking is a hard stop) |
| P1 and P2 both triggered | PARTIAL with most restrictive scope |
| Sign accuracy exactly 0.50 | Treat as NO-GO (no signal) |
| Exactly 2 of 4 GO conditions pass | PARTIAL (conservative default) |

---

## Secondary Analyses (Informative, Not Gate-Blocking)

| Analysis | Informs |
|----------|---------|
| Analysis 4 (Oracle comparison) | Whether to use P3-02 or independent oracle as RL reward |
| Analysis 6 (Adversarial) | Whether RL needs anti-hacking reward shaping |
| Analysis 7 (Non-additivity) | Whether to use greedy or beam search in RL action selection |

These do not affect the GO/PARTIAL/NO-GO verdict but inform P3-04 design.

---

## Verdict Recording

The final verdict is recorded in `docs/p3_03_go_no_go_decision.md` (this document, updated post-analysis) with:

```json
{
  "phase": "P3-03",
  "verdict": "GO | PARTIAL | NO_GO",
  "verdict_date": "<ISO timestamp>",
  "gating_conditions": {
    "G1": {"passed": true/false, "value": ..., "threshold": ...},
    "G2": {"passed": true/false, "value": ..., "threshold": ...},
    "G3": {"passed": true/false, "value": ..., "threshold": ...},
    "G4": {"passed": true/false, "value": ..., "threshold": ...}
  },
  "no_go_conditions": {
    "N1": {"triggered": true/false, "value": ..., "threshold": ...},
    "N2": {"triggered": true/false, "value": ..., "threshold": ...},
    "N3": {"triggered": true/false, "value": ..., "threshold": ...},
    "N4": {"triggered": true/false, "value": ..., "threshold": ...}
  },
  "partial_conditions": {
    "P1": {"triggered": true/false, "details": ...},
    "P2": {"triggered": true/false, "details": ...}
  },
  "next_phase": "P3-04 | P3-02_rework | P3-01_expansion",
  "scope_modification": "<if PARTIAL, describe reduced scope>",
  "analysis_results_ref": "results/p3_03/p3_03_analysis_results.json",
  "amendments": []
}
```

---

## Post-Verdict Actions

### If GO
1. Unlock P3-04 (full-scale GRPO)
2. Use P3-02 oracle as RL reward signal
3. Include anti-hacking constraints if Analysis 6 flagged any concern
4. Use greedy/beam search based on Analysis 7 non-additivity findings

### If PARTIAL
1. Amend P3-00A frozen contract (scope reduction)
2. Update P3-04 action space to effective region only
3. Do not claim cross-cargo generalization if P2 triggered
4. Proceed to P3-04 with reduced scope

### If NO-GO
1. Pause all RL training
2. Diagnose oracle failure mode (GC over-reliance, insufficient features, data leakage)
3. Expand P3-01 intervention data (more measured records)
4. Re-run P3-02 with improved oracle
5. Re-attempt P3-03 before any P3-04

---

## Status

**Current status:** AWAITING_WET_LAB_DATA

This document will be updated with the final verdict after wet-lab measurements are collected and analyzed. The criteria above are frozen and cannot be changed post-hoc.
