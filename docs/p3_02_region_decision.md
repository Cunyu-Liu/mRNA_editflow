# P3-02: Region Sensitivity Analysis and Decision

## Summary

**Verdict:** The current oracle does NOT demonstrate region-specific sensitivity. All perturbation types produce nearly identical mean deltas (~0.029), indicating the model has not learned position-specific or region-specific sequence-function relationships.

## Perturbation Results

| Perturbation Type | Mean Delta | Std Delta | N Samples |
|-------------------|-----------|-----------|-----------|
| five_utr_single_sub | 0.0293 | 0.0001 | 50 |
| start_proximal_cds | 0.0294 | 0.0000 | 4 |
| middle_cds | 0.0294 | 0.0000 | 4 |
| late_cds | 0.0294 | 0.0000 | 4 |
| joint_5utr_cds | 0.0294 | 0.0000 | 4 |
| matched_random | 0.0293 | 0.0001 | 50 |

## Sensitivity Checks

| Check | Result | Interpretation |
|-------|--------|----------------|
| Position sensitive | **False** (std=8.3e-5) | Model gives same prediction regardless of edit position |
| GC-only risk | **True** | 5'UTR and matched-random give identical patterns |
| Length-only risk | **True** | Std of deltas < 0.01 for same-length edits |
| CDS start vs late | 4.8e-6 (negligible) | No difference between start and late CDS positions |
| CDS position aware | **False** | Model does not distinguish CDS positions |
| Source aware | **False** | Model gives same prediction across different source sequences |

## Analysis

### 1. Is the model sensitive to edit position?
**No.** The position sensitivity std is 8.3e-5, meaning predictions vary by <0.01% across positions. The model treats all edit positions identically.

### 2. Does it only learn GC?
**Likely yes.** The 5'UTR single substitution and matched random substitution produce identical mean deltas (0.0293 vs 0.0293), suggesting the model responds to GC content changes rather than position-specific sequence context.

### 3. Does it only learn length?
**Likely yes.** All perturbations produce the same delta regardless of the specific edit, suggesting the model has learned a constant offset rather than length-dependent effects. The std is near zero for all perturbation types.

### 4. Does it only read CDS first few nt?
**Cannot determine meaningfully** — CDS perturbation samples are too few (n=4 each) due to most test sequences being 5'UTR-only (50nt, no CDS). The CDS start vs late difference is negligible (4.8e-6).

### 5. Does it ignore source sequence?
**Yes.** The model produces nearly identical deltas (std < 0.001) across different source sequences, indicating it has not learned source-dependent effects.

## Root Cause Analysis

The uniform predictions stem from:
1. **Insufficient features:** 22 lightweight features (GC, length, k-mer) lack the resolution to capture position-specific effects.
2. **Insufficient training:** Only 20 epochs on ~6K records (4.8K measured + 2K proxy).
3. **Model capacity:** Single hidden layer (64 units) may be too shallow.
4. **Data characteristics:** All measured records are 50nt 5'UTR-only with single-nt substitutions — limited diversity.

## Region Decision

### Current State
- **5'UTR:** Oracle does not show position-specific sensitivity within 5'UTR.
- **CDS:** Insufficient data to evaluate (most test sequences have no CDS).
- **Joint:** Not meaningfully testable with current data.

### Recommendation
1. **Task A (5'UTR-only):** Do NOT proceed to RL with current oracle. The oracle cannot distinguish beneficial from harmful 5'UTR edits.
2. **Task B (CDS-only):** Cannot evaluate — need CDS-containing benchmark records.
3. **Task C (Joint):** Blocked by both A and B.
4. **Next steps:**
   - Upgrade oracle with deeper architecture (GBT/CNN) and richer features (340-feature extractor)
   - Use full proxy dataset (473K records) for training
   - Validate against locked Oracle #3 (Pearson 0.434) as independent reference
   - Re-run P3-02 with improved oracle before any RL training

## Gate Impact

The sensitivity results directly contribute to the PARTIAL verdict:
- `not_gc_only`: **FAIL** (gc_only_risk=True)
- `not_length_only`: **FAIL** (length_only_risk=True)
- These two failures mean the gate cannot reach GO (maximum 3/5 criteria pass).
