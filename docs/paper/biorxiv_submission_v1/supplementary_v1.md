# DeltaBench — Supplementary Material v1 (bioRxiv submission)

## S1. Pre-registration document list (frozen before computation)

### 10.4 Pre-registration document list

All decision thresholds were frozen before computation. The full commit-hash list (commits that introduced each frozen document on the W0 branch `route-a-v3-w0-diagnosis-20260902`): `delta_density_prereg_framework_v1` (682b8c80); `delta_validity_oracle_prereg_v1` (a2a2919a); `first_order_decomposition_prereg_v1` (8f80d7e6); `benchmark_v2_matrix_row_prereg_v1` + `benchmark_v2_port_ledger_v1` (a14fb447); `m1_intervention_arm_amendment_v1` (b2e141cd); `polya_3seed_mini_prereg_v1` (73d47cdf); `density_sensitivity_mini_prereg_v1` + `first_order_train_alpha_sensitivity_prereg_v1` (60ff6999); `route2_critic_d16c_within_source_amendment_v1` (0809f5f6); `route2_critic_erk_v2_amendment_v1` (e343d4a3); `route2_w_ladder_amendment_v1` (7303417c); `utr_editflow_goal_v2_amendment_pivot_v1` (682b8c80); `route2_setflow_comb_mechanism_prereg_v1` (eb69e6bb, v8-stage1 worktree). Scope boundary: the family range is maintained at 14 (5 new + 9 existing); M5 is not included (no pre-registered row / port evidence). The batch-2 generalist-backbone append (7 adapters, frozen-probe caliber) is governed by `benchmark_v2_generalist_rows_amendment_v2.md` (DESIGN FROZEN); its execution commit is `6ef7cc6c` (journal 134; port-validation |Δ| = 2.4e-06 against the archived RNA-FM MRL row).


## S2. Reproduction entry points

### 10.5 Reproduction entry points

Row-construction scripts (`build_{m1,m6,s1}_*_v1.py`, with construction seeds archived for re-runs); the matrix execution / aggregation / re-check scripts (eight files); the analysis scripts (e.g. `run_delta_vs_density_v2.py`). **R2 internal target rows** cover all nine tasks (global-scaled internal control, macro 0.1317). **LOSO reference (Table 5 supplement, protocol B — frozen decision).** Two protocols were pre-registered: (A) true LOSO 7-fold retraining (42-job infrastructure archived) and (B) a LOSO-lite compiled table with an in-domain header; **protocol B was adopted** and its draft is the frozen LOSO Table 5. It lists, for all nine tasks: the multi-task in-domain critic V5 reading (explicitly not zero-shot), the external same-pool rows where such rows exist (MRL / polyA / MPRAU / HALF_LIFE only; TE / RNA / REF-ALT have no external same-pool rows and are declared as such), the internal target, and the ceiling. Worked rows: MRL 0.1354 vs frozen-Optimus 0.3132 / FramePool 0.2956 / UTR-LM 0.1107, internal 0.1192, ceiling 0.83; polyA 0.8219 vs APARENT 0.7343 / APARENT2 0.6810 / UTR-LM 0.7490, internal 0.7308, ceiling 0.90; MPRAU 0.0732 (cell caliber) vs Saluki 0.1205 weak-control, internal 0.0248, ceiling 0.683; HALF_LIFE 5′UTR 0.0607 and 3′UTR 0.0456 vs Saluki 3′UTR 0.0985, internal 0.0, ceiling ≈ 0.001–0.013 (physical unlearnability); GSE200304 TE 0.0579 vs internal −0.0266; PLUMAGE TE 0.1953 vs internal 0.1747; PLUMAGE RNA 0.0500 vs internal 0.2230 (the single registered loss, n = 48, power-limited); REF/ALT 0.0639 vs internal −0.0052.


## S3. Figure provenance

All four main figures are rendered from producer scripts with archived manifests and value locks:
- Figure 1:  (65/65 cells asserted vs matrix_v2_results.json)
- Figure 2a/2b:  /  (r=0.9388 recomputed=archived; verdict FAIL 33/65 identical)
- Figure 3:  (values from frozen results_first_order.json)
- Figure 4:  (frozen-vs-figure |Δ|<5e-5 assertions)
Unified style:  (DejaVu Sans base 9, dpi 300, fonttype 42).

## S4. Gate P unblinding receipt summary

One-shot TEST read (18,292 rows) of the frozen V5 main row, 2026-10-07, per ; receipt with sha256s at ; consistency 3/4 CONFIRMED (polyA 0.8205 vs 0.8219; polyA strongest; macro Δ−0.032) + 1 honest deviation (small-sample sign stability, magnitudes in near-zero band).
