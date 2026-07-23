# P3-02: Optimization Headroom Analysis

## Summary

**Verdict:** Headroom exists but is marginal and potentially artifactual.

All 5 search methods find positive candidates in 100% of the 20 test sources, but the mean best delta is only ~0.03 (on a ~5.0 MRL scale), representing <1% improvement. The multi-edit advantage over single-edit is negligible (2.6e-5).

## Search Results

| Method | Mean Best Delta | Median | Max | Positive % | N Evaluated |
|--------|----------------|--------|-----|-----------|-------------|
| exact_one_edit | 0.0295 | 0.0295 | 0.0295 | 100% | 150 |
| greedy | 0.0296 | 0.0295 | 0.0296 | 100% | 450 |
| beam_search | 0.0295 | 0.0295 | 0.0296 | 100% | 600 |
| simulated_annealing | 0.0294 | 0.0294 | 0.0295 | 100% | 44 |
| mcts | 0.0295 | 0.0295 | 0.0295 | 100% | 50 |

## Key Findings

### Q1: Best legal edit improvement over source?
- **Mean:** +0.030 MRL (~0.6% of source value ~5.0)
- **Max:** +0.030 MRL
- All methods converge to nearly identical deltas, suggesting the oracle predicts a uniform small positive bias for any edit.

### Q2: Fraction of sources with positive candidates?
- **100%** — every source has at least one positive candidate.
- However, this may reflect oracle bias (predicting everything as slightly positive) rather than true headroom.

### Q3: Is gain concentrated in few families?
- Cannot determine with current sample size (20 sources).
- The uniformity of deltas across sources suggests no concentration pattern.

### Q4: 5'UTR vs CDS contribution?
- All edits were 5'UTR-only (Task A active task scope).
- CDS contribution not evaluated in this run.
- Region sensitivity tests show no position-specific signal (all regions produce identical mean deltas).

### Q5: Does joint search exceed single-region search?
- Multi-edit advantage: +2.6e-5 (greedy vs exact_one_edit)
- MCTS vs greedy: -3.0e-5
- **No meaningful advantage from multi-edit or advanced search.**

## Interpretation

The uniform ~0.03 delta across all sources, positions, and methods is suspicious:
1. The oracle may have learned a constant positive bias rather than position-specific effects.
2. The sensitivity analysis confirms: position_sensitive=False, GC-only risk=True, length-only risk=True.
3. The model's predictions are nearly identical regardless of edit position or nucleotide change.

This suggests the current oracle has NOT learned meaningful sequence-function relationships — it predicts a small positive delta for any edit, which is not actionable for RL.

## Recommendations

1. **Do not enter full joint RL** with the current oracle.
2. **Improve feature engineering:** The 22 lightweight features may be insufficient. Consider:
   - K-mer features (3-mer, 4-mer)
   - Position-specific one-hot encoding (currently unused in the MLP)
   - RNA secondary structure proxies (MFE, accessibility)
3. **Increase training data:** Only 2,000 proxy records were used (subsampled from 473K). Full proxy training may improve signal.
4. **Use deeper models:** The single-hidden-layer MLP (64 units) may be insufficient. Consider:
   - sklearn MLPRegressor with 2-3 hidden layers
   - Gradient boosted trees (LightGBM, available as Oracle #3)
   - 1D CNN over one-hot sequence encoding
5. **Validate with the locked GBT Oracle #3** (340 features, Pearson 0.434) as an independent reference.
6. **Re-run headroom search** after oracle improvement to verify non-uniform predictions.

## Confidence Assessment

| Question | Answer | Confidence |
|----------|--------|------------|
| Headroom exists? | Yes (technically) | LOW — may be artifact |
| Headroom is actionable? | No | HIGH |
| Multi-edit helps? | No | HIGH |
| Region matters? | Unknown (5'UTR-only) | N/A |
| Should proceed to RL? | **No** | HIGH |
