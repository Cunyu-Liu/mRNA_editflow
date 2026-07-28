# P3-11 Statistical Analysis Plan (Pre-Registered)

> Created: 2026-07-26T03:16:33Z
> Status: **PRE-REGISTERED** — no changes after data collection

## Primary Model: Mixed-Effects Regression

Per P3-11 spec (lines 2700-2735), the primary statistical model is a
mixed-effects regression with the following specification:

### Fixed Effects (Factors)

| Factor | Type | Levels |
|--------|------|--------|
| method | categorical | wt, random_legal, best_single_edit, ranker, strong_search, mef_policy, mef_policy_plus_search, single_region, joint_region, adversarial_control |
| edit_budget | continuous | 0–5 (number of edits per design) |
| region | categorical | five_utr_only, five_utr_plus_cds |
| cargo | categorical | reporter_protein, secreted_protein, functional_editing_related_protein |
| cell_context | categorical | context_A, context_B |
| time | continuous | time-course measurement points (for P3-11B readouts) |

### Random Effects

| Random Effect | Grouping | Justification |
|---------------|----------|---------------|
| design | (1 \| design) | Multiple measurements per design (time course, replicates) |
| biological_replicate | (1 \| replicate) | Biological variation across replicates |
| experimental_batch | (1 \| batch) | Batch effects in wet-lab execution |

### Model Formula (R/lme4 syntax)

```r
# Primary endpoint: protein output (log-transformed)
lmer(
  log_protein_output ~ method * edit_budget * region + cargo + cell_context + time +
    (1 | design) + (1 | replicate) + (1 | batch),
  data = wet_lab_results
)
```

### Reporting (per spec lines 2729-2734)

| Metric | Definition |
|--------|-----------|
| effect size | Cohen's d or marginal R² for method main effect |
| confidence interval | 95% CI for each method vs WT contrast |
| adjusted p-value | Benjamini-Hochberg FDR correction across 10 arm comparisons |
| positive-response rate | Fraction of designs with >1.5× WT protein output |
| cargo heterogeneity | I² statistic for cross-cargo effect variance |

## Pre-Registered Contrasts

1. **MEF policy vs WT** (primary): H₀: Δ_protein = 0; H₁: Δ_protein > 0
2. **MEF policy vs random_legal**: tests whether learned policy beats random
3. **MEF policy vs strong_search**: tests whether policy adds value over search
4. **MEF policy + search vs MEF policy**: tests search refinement value
5. **joint_region vs single_region**: tests cross-region synergy
6. **adversarial_control vs WT**: confirms adversarial patterns are deleterious
7. **Top-k enrichment**: binomial test for over-representation of policy arm in top 10%

## Multiple Testing Correction

- 7 pre-registered contrasts
- Benjamini-Hochberg FDR at q = 0.05
- No post-hoc contrasts without correction

## Power Analysis (pre-registered)

- Minimum detectable effect: 1.5× WT (Cohen's d ≈ 0.5)
- With 500 designs × 10 arms × 3 replicates: >80% power at α = 0.05
- With 5 cargos × 8 designs × 7 methods × 3 replicates: >75% power for cargo heterogeneity

## Outlier Handling (pre-registered)

- Computational: designs with predicted_delta > 3σ from arm mean flagged before wet-lab
- Wet-lab: measurements >3 MAD from per-batch median flagged as outliers
- Outliers are NOT removed; they are reported with sensitivity analysis (with/without)

## Failure Handling (pre-registered)

- Arm failures (oracle budget exhausted, policy error) are recorded with error traces
- Failed designs use WT fallback (documented per arm)
- No post-hoc replacement of failed designs
- Sensitivity analysis: primary analysis includes failures; secondary analysis excludes them

## Qualifier

All predicted deltas are "predicted / internal proxy" until wet-lab validation
is complete. No claim of "improves TE/stability/expression" without the
"predicted" qualifier until P3-11 wet-lab data is collected (constraint #23).
