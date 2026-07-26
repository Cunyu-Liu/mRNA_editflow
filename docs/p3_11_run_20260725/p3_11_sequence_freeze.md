# P3-11 Sequence Freeze (Pre-Registration)

> Created: 2026-07-25T14:45:10Z
> Status: **FROZEN** — no changes after wet-lab data collection

## Purpose

This document freezes all computational artifacts for prospective wet-lab
validation, per P3-11 spec (lines 2678-2696). **No candidate may be changed
after experimental data is observed.**

## Frozen Items (11 per spec)

| # | Item | Value |
|---|------|-------|
| 1 | all source sequences | Frozen in p3_11_pooled_designs.json (per-source five_utr sequences) |
| 2 | all candidate sequences | Frozen in p3_11_pooled_designs.json and p3_11_full_length_designs.json |
| 3 | model checkpoint | `checkpoints/p3_08_gateB_gpu6/grpo_seed42_step4000.pt (SHA-256: e4ae3d6d8c4ee876f1d855e61cc3f1bbcba8c25061c811502e4a37f138f4f2fc)` |
| 4 | selection rule | Greedy decode (MEF policy) / beam search / oracle ranking — frozen per arm |
| 5 | excluded motifs | motif_policy_v1: hard_forbidden motifs excluded from action space; guarded_risk motifs tracked |
| 6 | primary endpoint | Protein output (predicted delta ranking vs WT, validated by wet-lab TE) |
| 7 | secondary endpoints | top-k enrichment (fraction of top-10% designs per arm), edit-budget Pareto (delta per edit), region interactions (5'UTR vs joint-region), Oracle transfer (training→independent consistency), cargo heterogeneity (cross-cargo effect variance) |
| 8 | sample size (pooled) | 100 sources × 10 arms |
| 8 | sample size (full-length) | 5 cargos × 7 methods × 8 designs × 2 contexts × 3 replicates |
| 9 | outlier rule | Pre-registered: designs with predicted_delta > 3σ from arm mean flagged; wet-lab outliers defined as >3 MAD from median per batch |
| 10 | failure handling | Pre-registered: arm failures (oracle budget exhausted, policy error) recorded with error trace; no post-hoc replacement of failed designs; WT fallback documented per arm |
| 11 | statistical model | Mixed-effects model: delta ~ method * edit_budget * region + (1|design) + (1|replicate) + (1|batch); factors: method, edit budget, region, cargo, cell context, time; random effects: design, biological replicate, experimental batch |

## Arms (P3-11A, 10 per spec)

| Arm | Description |
|-----|-------------|
| wt | Wild-type (no edits) |
| random_legal | Random legal edits (uniform sampling) |
| best_single_edit | Best single edit by oracle score |
| ranker | Greedy hill-climbing (beam_width=1) |
| strong_search | P3-07 beam search (width=8, budget=500) |
| mef_policy | P3-08 GRPO policy greedy decode |
| mef_policy_plus_search | MEF policy + beam search refinement |
| single_region | 5'UTR-only (Task A, same as mef_policy) |
| joint_region | 5'UTR + CDS synonymous (CAI improvement) |
| adversarial_control | CA repeat injection (P3-09 adversarial pattern) |

## Delta Ranking (predicted / internal proxy)

| Rank | Arm | Predicted Δ mean |
|------|-----|-----------------|
| 1 | ranker | 0.005924 |
| 2 | best_single_edit | 0.002063 |
| 3 | random_legal | 0.000992 |
| 4 | wt | 0.000000 |
| 5 | strong_search | 0.000000 |
| 6 | mef_policy | 0.000000 |
| 7 | mef_policy_plus_search | 0.000000 |
| 8 | single_region | 0.000000 |
| 9 | joint_region | 0.000000 |
| 10 | adversarial_control | 0.000000 |

## Top-k Enrichment (top 10% by predicted delta)

| Arm | Top-k fraction |
|-----|---------------|
| wt | 0.9700 |
| random_legal | 0.0100 |
| best_single_edit | 0.0100 |
| ranker | 0.0100 |
| strong_search | 0.0000 |
| mef_policy | 0.0000 |
| mef_policy_plus_search | 0.0000 |
| single_region | 0.0000 |
| joint_region | 0.0000 |
| adversarial_control | 0.0000 |

## Edit-Budget Pareto

| Arm | Mean edits | Mean Δ |
|-----|-----------|--------|
| wt | 0.00 | 0.000000 |
| random_legal | 0.03 | 0.000992 |
| best_single_edit | 0.01 | 0.002063 |
| ranker | 0.03 | 0.005924 |
| strong_search | 0.00 | 0.000000 |
| mef_policy | 0.00 | 0.000000 |
| mef_policy_plus_search | 0.00 | 0.000000 |
| single_region | 0.00 | 0.000000 |
| joint_region | 0.00 | 0.000000 |
| adversarial_control | 0.00 | 0.000000 |

## P3-11B Cargo Validation

- Cargos: 5 (reporter_protein, secreted_protein, functional_editing_related_protein)
- Methods: wt, best_single_edit, ranker, strong_search, mef_policy, mef_policy_plus_search, adversarial_control
- Designs per method/cargo: 8
- Cell contexts: 2
- Biological replicates: 3
- Total designs: 280
- Readouts: protein_output_time_course, mRNA_abundance, apparent_half_life, translation_efficiency, dose_response, cell_viability, IVT_yield, dsRNA_innate_immune_readout

## Integrity Guarantee

Per spec line 2696: **"不得在看到实验数据后更换 candidate"**
(No candidate may be changed after seeing experimental data.)

All candidate sequences are frozen in the JSON artifacts with their SHA-256
hashes recorded. Any discrepancy between frozen sequences and wet-lab
sequences must be documented as a protocol deviation.
