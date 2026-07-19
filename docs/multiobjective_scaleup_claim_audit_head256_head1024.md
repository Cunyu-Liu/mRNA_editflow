# Multi-Objective Scale-Up Claim Audit

- Primary metric: `delta_oracle_te_vs_source`
- Comparison rows ready: True
- Full hard-constraint claim audit ready: True
- head256 fusion vs te_only all strict: True
- head1024 vs te_only strict claim allowed: False
- head1024 vs te_only best signal: borderline_positive
- head1024 vs hardneg_v2 all strict: True
- Claim policy: head256 fusion-vs-control claims may be strict only where paired p < 0.05. The completed head1024 scale-up versus the stronger te_only control is not strictly significant; pareto is borderline/trend only, while all head1024 fusion modes remain strictly positive versus the older hardneg_v2 baseline.

## Comparison Ledger

| comparison | run | delta | paired p | signal | allowed language | source |
|---|---|---:|---:|---|---|---|
| head256_mo_scalar_top64_vs_te_only | mo_scalar_top64 | 0.00739118 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| head256_mo_pareto_top64_vs_te_only | mo_pareto_top64 | 0.00680289 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| head256_mo_grpo_top64_vs_te_only | mo_grpo_top64 | 0.00765385 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| head256_mo_scalar_top64_vs_hardneg_v2 | mo_scalar_top64 | 0.00584541 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_scalar_vs_hardneg_v2_head256.json` |
| head256_mo_pareto_vs_hardneg_v2 | mo_pareto | 0.00525712 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_pareto_vs_hardneg_v2_head256.json` |
| head256_mo_grpo_vs_hardneg_v2 | mo_grpo | 0.00610809 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_grpo_vs_hardneg_v2_head256.json` |
| head256_grpo_vs_scalar | mo_grpo | 0.000262674 | 0.434783 | positive_not_significant | numeric_positive_only_no_significance_claim | `benchmark/compare_grpo_vs_scalar_head256.json` |
| head256_scalar_vs_pareto | mo_scalar | 0.000588287 | 0.213893 | positive_not_significant | numeric_positive_only_no_significance_claim | `benchmark/compare_scalar_vs_pareto_head256.json` |
| head1024_mo_scalar_top64_vs_te_only | mo_scalar_top64 | 8.39869e-05 | 0.744628 | positive_not_significant | numeric_positive_only_no_significance_claim | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| head1024_mo_pareto_top64_vs_te_only | mo_pareto_top64 | 0.000807924 | 0.0504748 | borderline_positive | trend_or_borderline_only_no_strict_significance | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| head1024_mo_grpo_top64_vs_te_only | mo_grpo_top64 | 5.52153e-05 | 0.898051 | positive_not_significant | numeric_positive_only_no_significance_claim | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| head1024_mo_scalar_top64_vs_hardneg_v2 | mo_scalar_top64 | 0.00469441 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |
| head1024_mo_pareto_top64_vs_hardneg_v2 | mo_pareto_top64 | 0.00541835 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |
| head1024_mo_grpo_top64_vs_hardneg_v2 | mo_grpo_top64 | 0.00466564 | 0.00449775 | strict_positive | may_claim_strict_significant_positive | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |

## Summary Constraint Evidence

| slice | mode | exists | constraints exact 1 | missing metrics | path |
|---|---|---:|---:|---|---|
| head256 | te_only | True | True | `[]` | `benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json` |
| head256 | scalar | True | True | `[]` | `benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json` |
| head256 | pareto | True | True | `[]` | `benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json` |
| head256 | grpo | True | True | `[]` | `benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json` |
| head256 | hardneg_v2 | True | True | `[]` | `benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json` |
| head1024 | te_only | True | True | `[]` | `benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json` |
| head1024 | scalar | True | True | `[]` | `benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json` |
| head1024 | pareto | True | True | `[]` | `benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json` |
| head1024 | grpo | True | True | `[]` | `benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json` |
| head1024 | hardneg_v2 | True | True | `[]` | `benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json` |
