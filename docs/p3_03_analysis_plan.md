# P3-03: Analysis Plan

**Phase:** P3-03
**Status:** FROZEN (pre-registered)
**Date:** 2026-07-23
**Depends on:** docs/p3_03_experiment_preregistration.md, docs/p3_03_sequence_manifest.json

---

## Overview

This document specifies the **seven key analyses** that will be executed once wet-lab measurements are uploaded. All analyses are pre-registered before unblinding. The GO/PARTIAL/NO-GO criteria in `docs/p3_03_go_no_go_decision.md` depend on the outputs of these analyses.

---

## Data Inputs

| Input | Source | Format |
|-------|--------|--------|
| Sequence manifest | docs/p3_03_sequence_manifest.json | JSON with 24 sequences × 3 replicates |
| Predicted deltas | Embedded in manifest | per-arm predicted_delta, prediction_std |
| Measured readouts | Wet-lab upload | per-well: protein_output, mRNA_abundance, cell_viability at 4h/8h/24h/48h |
| Oracle checkpoint | checkpoints/p3_delta_oracles/ | P3-02 cross-fitted ensemble |
| Independent oracle | eval/oracle.py (LocalTranslationOracle) | GBT-based UTR oracle |

---

## Preprocessing

### 1. Aggregation
- Per-arm measured values = median across ≥3 replicates
- Per-arm AUC = trapezoidal integration over [4h, 8h, 24h, 48h]
- Apparent TE = protein_output_AUC / mRNA_abundance_AUC
- Apparent half-life = exponential decay constant fit from 24h→48h

### 2. Delta Computation
- **Measured delta** = measured_value(arm) − measured_value(A01_wt)
- **Predicted delta** = predicted_delta(arm) from manifest (already computed)

### 3. Exclusion Application
- Apply pre-specified exclusion criteria (viability < 50% WT, failed transfection)
- Report exclusion count per arm

---

## Seven Key Analyses

### Analysis 1: Predicted Delta vs Measured Delta Correlation

**Question:** Does the oracle's predicted local-delta correlate with measured biological delta?

**Method:**
- Compute Pearson and Spearman correlation between predicted_delta and measured_delta across all 24 arms (pooled across cargos)
- Compute per-cargo correlation (12 arms each)
- Bootstrap 95% CI (1000 resamples, clustered by cargo)

**Metrics:**
| Metric | Definition |
|--------|-----------|
| delta_spearman | Spearman ρ(predicted_delta, measured_delta) |
| delta_pearson | Pearson r(predicted_delta, measured_delta) |
| per_cargo_spearman | Spearman ρ per cargo |
| bootstrap_ci_95 | [2.5%, 97.5%] percentile bootstrap CI |

**Pass threshold:** Spearman ρ > 0.3 with 95% CI excluding 0 (pooled)

---

### Analysis 2: Beneficial-Edit Sign Accuracy

**Question:** When the oracle predicts a beneficial edit (predicted_delta > 0), does the measured delta also tend to be positive?

**Method:**
- Define predicted-beneficial arms: predicted_delta > 0 (excluding A01 WT)
- Define measured-beneficial arms: measured_delta > 0
- Sign accuracy = fraction of arms where sign(predicted_delta) == sign(measured_delta)
- Binomial test against H0: p = 0.5

**Metrics:**
| Metric | Definition |
|--------|-----------|
| sign_accuracy | P(sign(pred) == sign(measured)) |
| beneficial_precision | P(measured > 0 | predicted > 0) |
| harmful_recall | P(measured < 0 | predicted < 0) |
| binomial_p | p-value of sign test vs H0: p=0.5 |

**Pass threshold:** sign_accuracy > 0.60, binomial_p < 0.10

---

### Analysis 3: Top-Ranked Enrichment

**Question:** Are the top-ranked predicted edits enriched for measured-beneficial edits compared to random?

**Method:**
- Rank arms by predicted_delta within each cargo
- Define top-k = top 30% (≈4 arms per cargo)
- Enrichment = (measured-beneficial rate in top-k) / (measured-beneficial rate in bottom-k)
- Fisher's exact test for significance
- Compare A04/A05/A06/A07 (prediction-guided) vs A02/A03 (random)

**Metrics:**
| Metric | Definition |
|--------|-----------|
| top_k_enrichment | beneficial_rate(top-k) / beneficial_rate(bottom-k) |
| fisher_p | Fisher's exact test p-value |
| guided_vs_random_ttest | t-test: measured_delta(guided arms) vs measured_delta(random arms) |

**Pass threshold:** top_k_enrichment > 1.5 OR fisher_p < 0.05

---

### Analysis 4: Training Oracle vs Independent Oracle

**Question:** Which oracle (P3-02 trained local-delta ensemble vs independent GBT UTR oracle) is closer to experimental measurements?

**Method:**
- For each arm, compute:
  - P3-02 predicted_delta (from manifest)
  - Independent oracle predicted_delta = predict(candidate) − predict(source)
- Compute correlation of each oracle's predictions with measured_delta
- Compute Brier score for sign prediction
- Compare via Diebold-Mariano test for predictive accuracy

**Metrics:**
| Metric | Definition |
|--------|-----------|
| p302_spearman | Spearman ρ(P3-02_delta, measured_delta) |
| independent_spearman | Spearman ρ(independent_delta, measured_delta) |
| p302_brier | Brier score for sign prediction |
| independent_brier | Brier score for sign prediction |
| dm_stat | Diebold-Mariano test statistic |

**Pass threshold:** P3-02 Spearman ≥ independent Spearman (P3-02 not worse)

---

### Analysis 5: Region-Specific Real Effects (5′UTR / CDS / Joint)

**Question:** Which region (5′UTR, CDS, joint) shows the largest realizable measured headroom?

**Method:**
- Group arms by region:
  - 5′UTR: A08 (five_utr_only_best)
  - CDS: A09 (cds_only_best)
  - Joint: A10 (joint_best)
- Compute measured_delta for each region arm vs A01
- Paired comparison: region arm vs corresponding random arm (A08 vs A02, A09 vs A02, A10 vs A03)
- Two-way ANOVA: measured_delta ~ region × cargo

**Metrics:**
| Metric | Definition |
|--------|-----------|
| utr_measured_delta | measured_delta(A08) |
| cds_measured_delta | measured_delta(A09) |
| joint_measured_delta | measured_delta(A10) |
| region_anova_p | ANOVA p-value for region effect |
| best_region | argmax of |measured_delta| across regions |

**Pass threshold:** At least one region shows |measured_delta| > 0.1 × WT_value with paired p < 0.05

---

### Analysis 6: Adversarial Reward Hacking Exposure

**Question:** Does the adversarial candidate (A12, high predicted delta via GC-maximizing) actually perform well, or does it expose reward hacking?

**Method:**
- Compare A12 (adversarial) vs A04 (best predicted one-edit) vs A01 (WT)
- If A12 measured_delta < 0 while predicted_delta > 0 → reward hacking confirmed
- If A12 measured_delta ≈ A04 → GC heuristic is a valid signal
- Compute: hacking_signal = sign(predicted_delta_A12) × (predicted_delta_A12 − measured_delta_A12)

**Metrics:**
| Metric | Definition |
|--------|-----------|
| adversarial_pred_delta | predicted_delta(A12) |
| adversarial_meas_delta | measured_delta(A12) |
| hacking_signal | max(0, predicted − measured) for A12 |
| hacking_flag | True if predicted > 0 and measured ≤ 0 |

**Interpretation:**
- hacking_flag = True → oracle overfits to GC heuristic; RL must include anti-hacking constraints
- hacking_flag = False → GC heuristic is partially valid; less concerning

---

### Analysis 7: Single-Edit vs Multi-Edit Non-Additivity

**Question:** Are multi-edit effects additive (sum of single-edit deltas) or do they show non-additive interactions?

**Method:**
- For arms with multiple edits (A03, A05, A06, A07, A10):
  - Compute predicted_additive = sum of individual single-edit predicted deltas
  - Compute measured_additive = sum of individual single-edit measured deltas (estimated from A02/A04)
  - Compute actual_multi = measured_delta(multi-edit arm)
  - Non-additivity = actual_multi − measured_additive
- Test: is |non_additivity| significantly different from 0? (one-sample t-test)

**Metrics:**
| Metric | Definition |
|--------|-----------|
| predicted_additive | Sum of single-edit predicted deltas |
| measured_additive | Sum of single-edit measured deltas |
| actual_multi | Measured delta of multi-edit arm |
| non_additivity | actual_multi − measured_additive |
| additivity_ttest_p | t-test p-value for H0: non_additivity = 0 |

**Interpretation:**
- Non-additivity ≈ 0 → edits are independent; greedy search is valid
- Non-additivity > 0 → synergistic; beam/joint search valuable
- Non-additivity < 0 → antagonistic; edit budget should be conservative

---

## Multiple Testing Correction

- Analyses 1–3 are the **primary** analyses for the GO gate
- Analyses 4–7 are **secondary** (informative, not gate-blocking)
- Primary analyses: Bonferroni correction (α = 0.05 / 3 = 0.0167)
- Secondary analyses: reported at nominal α = 0.05 with correction noted

---

## Reporting

All analyses produce a JSON results file `results/p3_03/p3_03_analysis_results.json` with:

```json
{
  "analysis_1_correlation": { ... },
  "analysis_2_sign_accuracy": { ... },
  "analysis_3_top_k_enrichment": { ... },
  "analysis_4_oracle_comparison": { ... },
  "analysis_5_region_effects": { ... },
  "analysis_6_adversarial": { ... },
  "analysis_7_nonadditivity": { ... },
  "exclusions": { ... },
  "cargo_summary": { ... }
}
```

This file feeds directly into `docs/p3_03_go_no_go_decision.md` for the gate verdict.

---

## Code

Analysis code: `scripts/run_p3_03_analysis.py` (to be executed after wet-lab data upload)

```bash
python scripts/run_p3_03_analysis.py \
  --manifest docs/p3_03_sequence_manifest.json \
  --measurements <uploaded_raw_data.csv> \
  --independent-oracle eval/oracle.py \
  --output results/p3_03/p3_03_analysis_results.json
```
