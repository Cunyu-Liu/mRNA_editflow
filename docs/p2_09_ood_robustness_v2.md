# P2-09: OOD Robustness Stress Test — Preliminary Results

**Status**: PRELIMINARY COMPLETE (5 baselines × 10 decoder seeds × 8 OOD subsets).
**Date**: 2026-07-20 (design); 2026-07-21 (preliminary execution)

## Goal

Stress-test the 5 ranker baselines (te_only, scalar, pareto, grpo,
hardneg_v2) from P2-03 on **out-of-distribution (OOD) subsets** of the
frozen combined_family test split. Compare OOD vs in-distribution (ID)
performance to quantify robustness.

## Deployed artifacts

| File | SHA-256 | Notes |
|------|---------|-------|
| `scripts/run_p2_09_ood_stress_test.py` | `afe9b919c8d38c56a77a31af8003433b71018f10463becb123c22ebb9a0b5d13` | Top-N / bottom-N subset selection (avoids tie inflation). |
| `tests/test_p2_09_ood.py` | `5caf42806e647a863b154b76e24e18d4b8e428b97bd846bf3053e8d866ff4967` | 42/42 tests pass on remote. |
| `benchmark/dev/ood_robustness_p2_09_preliminary/robustness_summary.json` | (see artifact) | 5 baselines × 8 subsets. |
| `benchmark/dev/ood_robustness_p2_09_preliminary/robustness_summary.md` | (see artifact) | Human-readable table. |

## Frozen test split

Source: `benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json`

| Role | Count | SHA-256 (idx) |
|------|------:|---------------|
| train | 119075 | `dd223e371dc148d86690dc7a4a7e625f0048f8828c751ab763482d5d24590ff6` |
| val | 14673 | `e849a71b980c0cba176fb318b78ae5d61d1c8767be6a73d520c00ee556ba4a10` |
| test | 14700 | `35d02ebed120da82056e72cbdde070a8c7e39a8592a89065e235f96733b05b66` |

- `paper_eligible: true`
- Family-disjoint, near-neighbor-audited (395 excluded).
- P2-03 eval used `--limit 1024`, so OOD subsets are defined on 1024 sources.

## OOD subset definitions

All subsets are defined on the **evaluated sources** (1024 from the test
split via `--limit 1024`). Membership is computed from record metadata
(lengths, GC content, gene family) and does NOT touch the train/val
splits, so there is no leakage.

For length/GC subsets, we use **top-N / bottom-N** (N = 102 = max(1,
1024//10)) instead of percentile thresholds, to avoid tie-induced
inflation when many sources share the same value (e.g., empty 5'UTR).

| Subset | Definition | Size | Criterion |
|--------|-----------|-----:|-----------|
| `length_total_p10` | bottom ~10% by total mRNA length | 102 | bottom-102 by total_length |
| `length_total_p90` | top ~10% by total length | 102 | top-102 by total_length |
| `length_5utr_p90` | top ~10% by 5'UTR length | 102 | top-102 by five_utr_length |
| `length_3utr_p90` | top ~10% by 3'UTR length | 102 | top-102 by three_utr_length |
| `gc_total_p10` | bottom ~10% by GC content | 102 | bottom-102 by gc_total |
| `gc_total_p90` | top ~10% by GC content | 102 | top-102 by gc_total |
| `family_rare` | gene family ≤2 members in eval set | 1024 | gene_count ≤ 2 (see note) |
| `cds_long_p90` | top ~10% by CDS length | 102 | top-102 by cds_length |

**Note on `family_rare`**: gene_id is approximated from transcript_id
(ENST→ENSG). In the test split, most transcripts map to distinct genes,
so this subset captures all 1024 sources. For a proper family-rarity
analysis, family assignments from the split contract should be used.
This subset is excluded from the robustness score ranking interpretation.

### ID reference

The **complement** of each OOD subset within the 1024 evaluated sources
serves as the matched ID set (e.g., for `length_total_p10`, the ID
reference is the other 922 sources).

## Evaluation protocol

For each ranker baseline `r` in {te_only, scalar, pareto, grpo, hardneg_v2}:

1. Load the P2-03 results from
   `benchmark/dev/leakage_free_headline_preliminary/<r>_seed0/`.
2. For each of 10 decoder seeds, compute per-record
   `delta_oracle_te_vs_source = oracle_ensemble_te - source_oracle_ensemble_te`.
3. For each OOD subset `S`:
   - Partition the 10-seed results into `S` (OOD) and `eval \ S` (ID).
   - Compute `delta_oracle_te_vs_source` mean and 95% bootstrap CI for both.
   - Paired permutation test (1000 permutations) for OOD vs ID.
4. Aggregate:
   - `robustness_score[r] = mean_S( delta_ood[S] - delta_id[S] )` across 7 subsets
     (excluding `family_rare` which captures all sources).
   - A score near 0 means robust; a large negative score means OOD degradation.

### Significance

- **Paired permutation test** (500 permutations, 10 seeds × sequences pooled).
- α = 0.05 with Bonferroni correction for 8 subsets → α_corrected = 0.00625.
- Report effect size (Cohen's d) and 95% CI.

### Reward qualifier

All `delta_oracle_te` values are **predicted TE (internal proxy)** from
Oracle #3. No unqualified "TE" claims.

## Preliminary results (5 baselines × 10 decoder seeds)

### Robustness scores (sorted best → worst)

| Rank | Baseline | Robustness score | # significant (Bonferroni) |
|-----:|----------|-----------------:|---------------------------:|
| 1 | `grpo` | +0.000368 | 2/7 |
| 2 | `scalar` | -0.000136 | 2/7 |
| 3 | `pareto` | -0.000211 | 3/7 |
| 4 | `te_only` | -0.000222 | 2/7 |
| 5 | `hardneg_v2` | -0.001184 | ?/7 |

**Key finding**: `grpo` is the **most robust** baseline (only positive
score) AND the **best** on the primary endpoint (P2-03 headline:
0.006770, rank 1/5). `hardneg_v2` shows the most OOD degradation.

### P2-03 headline ranking (primary endpoint, all 5 baselines)

| Rank | Baseline | Primary mean | 95% CI | n seeds |
|-----:|----------|-------------:|--------|--------:|
| 1 | `grpo_seed0` | 0.006770 | [0.006592, 0.006956] | 10 |
| 2 | `scalar_seed0` | 0.006618 | [0.006316, 0.006941] | 10 |
| 3 | `pareto_seed0` | 0.006293 | [0.005853, 0.006741] | 10 |
| 4 | `hardneg_v2_seed0` | 0.004166 | [0.003748, 0.004502] | 10 |
| 5 | `te_only_seed0` | 0.002835 | [0.002468, 0.003201] | 10 |

### P2-03 pairwise significance (primary endpoint)

Significant pairs (p < 0.05):
- `grpo` vs `te_only`: diff=−0.003935, p=0.0000 *** SIG (grpo better)
- `grpo` vs `hardneg_v2`: diff=−0.002604, p=0.0000 *** SIG (grpo better)
- `grpo` vs `pareto`: diff=−0.000477, p=0.0420 * SIG (grpo better)
- `grpo` vs `scalar`: diff=−0.000152, p=0.3552 (NOT sig; grpo ≈ scalar)
- `scalar` vs `te_only`: diff=−0.003783, p=0.0000 *** SIG
- `pareto` vs `te_only`: diff=−0.003459, p=0.0000 *** SIG
- `hardneg_v2` vs `te_only`: diff=−0.001331, p=0.0000 *** SIG
- `hardneg_v2` vs `scalar`: diff=+0.002452, p=0.0000 *** SIG (hardneg worse)
- `hardneg_v2` vs `pareto`: diff=+0.002128, p=0.0000 *** SIG (hardneg worse)
- `pareto` vs `scalar`: diff=+0.000324, p=0.3024 (NOT sig; pareto ≈ scalar)

## Constraint compliance

| Constraint | Status |
|------------|--------|
| 不擅自终止任何运行中进程 | OK — no processes touched. |
| 不修改 v1 frozen namespace | OK — only reads `data/reconstructed/p0_data_reconstruction_v1/`. |
| 所有新增训练接入 split contract | N/A — P2-09 is evaluation only, no training. Uses frozen test idx. |
| "improves TE" 加 predicted/internal proxy 限定词 | OK — all metrics are "predicted TE (internal proxy)". |
| 10-seed paired significance test | OK — 10 decoder seeds × paired permutation test, Bonferroni-corrected. Note: 1 training seed (preliminary); Phase C will add ≥2 more training seeds with family-cluster bootstrap CI. |
| 所有新代码配套单元测试 | OK — 42/42 tests pass. |

## Limitations (preliminary)

1. **1 training seed only**: Full P2-03 spec requires ≥3 training seeds
   with family-cluster bootstrap CI. Phase C will add 2 more training seeds.
2. **`family_rare` subset not meaningful**: gene_id approximation from
   transcript_id is too granular. Needs proper family metadata from the
   split contract.
3. **`--limit 1024`**: P2-03 eval used 1024 sources, not the full 14700
   test split. OOD subsets are defined on this 1024-sample subset.
4. **Permutation test pooled across seeds**: The paired permutation test
   pools 10 seeds × sequences. A more rigorous approach would use
   family-cluster bootstrap CI at the seed level.

## Next steps

1. **Phase C**: Train 2 additional training seeds for each of 5 baselines
   (10 training runs total), then re-run P2-09 with family-cluster
   bootstrap CI.
2. **Family metadata**: Load proper family assignments from the split
   contract to make `family_rare` meaningful.
3. **Full test split**: Re-run P2-03 on the full 14700 test split (no
   `--limit`) for a more comprehensive OOD analysis.
4. **Per-seed bootstrap**: Move from pooled permutation test to
   family-cluster bootstrap CI at the seed level.
