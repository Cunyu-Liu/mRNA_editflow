# P2-09: OOD Robustness Stress Test — Design

**Status**: DESIGN READY, execution BLOCKED by P2-03 (needs all 5 ranker
baselines complete on the frozen test split).
**Date**: 2026-07-20

## Goal

Stress-test the 5 ranker baselines (te_only, scalar, pareto, grpo,
hardneg_v2) from P2-03 on **out-of-distribution (OOD) subsets** of the
frozen combined_family test split (14700 sequences). Compare OOD vs
in-distribution (ID) performance to quantify robustness.

## Frozen test split

Source: `benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json`

| Role | Count | SHA-256 (idx) |
|------|------:|---------------|
| train | 119075 | `dd223e371dc148d86690dc7a4a7e625f0048f8828c751ab763482d5d24590ff6` |
| val | 14673 | `e849a71b980c0cba176fb318b78ae5d61d1c8767be6a73d520c00ee556ba4a10` |
| test | 14700 | `35d02ebed120da82056e72cbdde070a8c7e39a8592a89065e235f96733b05b66` |

- `paper_eligible: true`
- Family-disjoint, near-neighbor-audited (395 excluded).
- Records: `data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl`

## OOD subset definitions

All subsets are defined on the **test split** (14700 sequences). Membership
is computed from record metadata (lengths, GC content, family ID) and
does NOT touch the train/val splits, so there is no leakage.

| Subset | Definition | Expected size | Rationale |
|--------|-----------|--------------:|-----------|
| `length_total_p10` | bottom 10% by `len(5'UTR)+len(CDS)+len(3'UTR)` | ~1470 | Short transcripts are underrepresented in training (length bias). |
| `length_total_p90` | top 10% by total length | ~1470 | Long transcripts exceed the model's typical context. |
| `length_5utr_p90` | top 10% by `len(5'UTR)` | ~1470 | Long 5'UTR tests UTR editor robustness. |
| `length_3utr_p90` | top 10% by `len(3'UTR)` | ~1470 | Long 3'UTR tests stability proxy. |
| `gc_total_p10` | bottom 10% by GC content (full sequence) | ~1470 | Low-GC sequences have different secondary structure. |
| `gc_total_p90` | top 10% by GC content | ~1470 | High-GC sequences are harder to edit. |
| `family_rare` | sequences whose family has ≤2 members in the full dataset | ~? | Rare families test generalization beyond common families. |
| `cds_long_p90` | top 10% by `len(CDS)` | ~1470 | Long CDS tests codon-lattice DP scalability. |

**Total**: 8 OOD subsets. Each subset is a *subset* of the test split, so
the same 10 decoder seeds from P2-03 apply (no new sampling needed).

### ID reference

The **full test split** (14700 sequences) serves as the ID reference.
OOD vs ID comparison uses the *complement* of each OOD subset within the
test split as the matched ID set (e.g., for `length_total_p10`, the ID
reference is the other 90% of the test split).

## Evaluation protocol

For each ranker baseline `r` in {te_only, scalar, pareto, grpo, hardneg_v2}:

1. Load the P2-03 results from
   `benchmark/dev/leakage_free_headline_preliminary/<r>_seed0/`.
2. For each OOD subset `S`:
   - Partition the 10-seed results into `S` (OOD) and `test \ S` (ID).
   - Compute `delta_oracle_te_vs_source` mean and std for both partitions.
   - Paired permutation test (10 seeds × sequences) for OOD vs ID.
3. Aggregate:
   - `ood_robustness_score[r] = mean_S( delta_oracle_te_ood[S] - delta_oracle_te_id[S] )`
   - A score near 0 means robust; a large negative score means OOD degradation.

### Significance

- **10-seed paired significance test** (family-cluster bootstrap CI, cluster
  = family ID) for each OOD vs ID comparison.
- α = 0.05 with Bonferroni correction for 8 subsets → α_corrected = 0.00625.
- Report effect size (Cohen's d) and 95% CI.

### Reward qualifier

All `delta_oracle_te` values are **predicted TE (internal proxy)** from
Oracle #3. No unqualified "TE" claims.

## Implementation plan (when P2-03 completes)

### Script: `scripts/run_p2_09_ood_stress_test.py`

```
Inputs:
  --test-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx
  --records-jsonl data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl
  --p2-03-root benchmark/dev/leakage_free_headline_preliminary
  --baselines te_only scalar pareto grpo hardneg_v2
  --out-dir benchmark/dev/ood_robustness_p2_09
  --n-bootstrap 1000

Outputs:
  benchmark/dev/ood_robustness_p2_09/
    ├── ood_subsets.json          # subset definitions + membership
    ├── <baseline>_<subset>.json  # per-baseline per-subset stats
    ├── robustness_summary.json   # aggregate robustness scores
    └── robustness_summary.md     # human-readable table
```

### Unit tests: `tests/test_p2_09_ood.py`

- OOD subset membership computation (length, GC, family rarity).
- OOD vs ID partition correctness.
- Paired permutation test correctness.
- Robustness score aggregation.

## Constraint compliance

| Constraint | Status |
|------------|--------|
| 不擅自终止任何运行中进程 | OK — no processes touched. |
| 不修改 v1 frozen namespace | OK — only reads `data/reconstructed/p0_data_reconstruction_v1/`. |
| 所有新增训练接入 split contract | N/A — P2-09 is evaluation only, no training. Uses frozen test idx. |
| "improves TE" 加 predicted/internal proxy 限定词 | OK — all metrics are "predicted TE (internal proxy)". |
| 10-seed paired significance test | PLANNED — family-cluster bootstrap CI, Bonferroni-corrected. |
| 所有新代码配套单元测试 | PLANNED — `tests/test_p2_09_ood.py`. |

## Blocker

P2-09 execution requires all 5 ranker baselines from P2-03 to complete.
Current P2-03 status (2026-07-20):

| Baseline | Status |
|----------|--------|
| te_only | seed 6/10 running (PID 613062, GPU 6) |
| scalar | queued |
| pareto | queued |
| grpo | queued |
| hardneg_v2 | queued |

Estimated time to P2-03 completion: ~4-5 hours (te_only finishes in ~24 min,
then 4 more baselines × ~60 min each).

## Next steps

1. Wait for P2-03 to complete all 5 baselines.
2. Implement `scripts/run_p2_09_ood_stress_test.py`.
3. Implement `tests/test_p2_09_ood.py`.
4. Run OOD stress test.
5. Aggregate results into `benchmark/dev/ood_robustness_p2_09/robustness_summary.md`.
