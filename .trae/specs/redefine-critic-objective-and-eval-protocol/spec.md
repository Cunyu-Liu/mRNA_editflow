# Spec — Redefine Critic Objective and Eval Protocol (V6)

> **Recovered 2026-08-31 by the successor session.** The original three-file
> spec was referenced by [`CURRENT_HANDOVER_STATUS_20260831.md`]
> but was not present on the A100 worktree or the local Mac. This file is
> reconstructed verbatim-in-content from that authoritative handover document
> (§2 facts, §3 decisions, §4 gates, §5 execution lists). It is the frozen
> execution basis for V6–V8; do not rewrite gates after the fact.

## 0. Purpose

Make the Critic's learning objective and evaluation口径 match D0/D1/D2 so
that a Cleaner Shared-Effect model has a measured chance of exceeding V5's
within-source/pair-level 0.103 (pair-mean rho) with statistical evidence.

## 1. Facts (F1–F16, from 2026-08-31 deep read-only diagnosis)

### F1–F4: Label signal-to-noise ceilings
| Task | Ceiling | Basis | V5 now | Progress |
|---|---|---|---|---|
| MPRAU allelic skew (ENCSR854RUF, 2008 variants × 6 cells) | **0.683** split-half [0.671, 0.710] | each pair measured in exactly 6 cells | pair-level 0.103 | 15% |
| PROXIMAL_POLYA | 0.90 (ICC) | pair repeat structure | 0.82 | 91% ✅ |
| MRL | unknown (no repeats) | — | within-source 0.118 | track |
| RNA_HALF_LIFE (two regions) | 0.0014 / 0.013 (ICC) | labels ≈ noise | ≈0 | physical ceiling ≈0; must attribute |
| TE / abundance (GSE149487) | no repeats; zero-shot eval task | — | training set has 0 rows | — |

### F5–F9: Mechanism attribution (why V5 is a "compressor")
- F5: model per-variant prediction std = 0.018 vs label std = 0.126 → the model
  does not learn cross-cell differences.
- F6: prediction pairwise std 0.108 vs label 0.269 → compresses ~60% of variance.
- F7: 10-bucket analysis: only the highest prediction bucket separates (0.068);
  the other 9 buckets sit near zero → mid/low-effect ordering ≈ random, the
  direct lesion of Spearman.
- F8: study_scale = only 7 scalars (code review verdict) → "scale absorbs
  variance" shortcut hypothesis rejected.
- F9: pair-mean supervision is the correct mechanism: single-cell reliability
  0.417 → 6-cell mean via Spearman-Brown → 0.683 ceiling.

### F10–F12: EditFlow (FAIR) paper exploration
- F10: original guidance = classifier-free CFG only (naive-rate CFG optimal);
  no reward/critic/potential anywhere in the paper.
- F11: our Potential-style guidance `U_q = U_p * e^{β[V(s′)−V(s)]}` is our own
  design beyond the paper.
- F12: CFG-style guidance is Plan B (no critic needed).

### F13–F16: Data facts
- F13: three-axis long tail: task-level (62%/29%/…/0.2%), group-level (6/9
  tasks have only 1 candidate per source), effect-magnitude (10-bucket).
- F14: true scale: 89,580 rows = 43,730 unique sequences (12,635
  near-duplicate components); independent effect observations only 5–8k;
  parameters:observations ≈ 20–30:1.
- F15: train soft_spearman 0.235 vs val 0.167 + loss not converged +
  mid/low under-learning = local overfit AND global under-fit coexist.
- F16: `different_source_group_pair_indices` is greedy disjoint matching
  (32 rows → 16 pairs, not C(32,2)) → pair coverage only N/2 per batch; V7
  candidate.

## 2. Decisions (frozen 2026-08-31)

### D0 — Critic role: B ≈ C > A
- **B** = experiment prioritization ranker (**task families**: 5'UTR MRL /
  3'UTR stability / polyA usage / variant ordering / LOSO transfer — NOT only
  5U-MRL-REFINE).
- **C** = cross-context variant effect annotator (context = cell line / study /
  assay / time point; cross-cell is only the one window with a quantifiable
  ceiling today — do not extrapolate).
- **A** = guidance scorer (blocked by SetFlow readiness; 6/9 tasks lack
  within-source evaluation structure).
- Current phase: make the critic itself trustworthy (B/C); this is A's
  prerequisite.

### D1 — Learning objective: shared effect + conditioning (two tracks)
- Main supervision: **pair-mean** (MPRAU 12,048 rows: each row's label → its
  six-cell pair mean; row count unchanged).
- Auxiliary head: **cell offset** (predict each cell's offset relative to the
  pair mean; HEK293FT vs others only 0.17–0.20 correlated → discrimination is
  the auxiliary acceptance criterion).
- Additive (approved by user): **per-task rank transform** at train time
  (intra-task label rank-Gaussian/quantile transform flattens the effect long
  tail; Spearman invariant; MAE double-currency).
- Implemented: **LambdaRankIC displacement-weighted pairwise** (λ = |soft_rank
  gap|·|midrank gap|, detached; commit `7f01d17c`).
- Honest statement: success is not guaranteed; this is a hypothesis test
  backed by a 0.68 ceiling.

### D2 — Evaluation口径: ceiling-normalized pass
- Pass = each task approaches its own physical ceiling (Tier A target ≥60%),
  **not** high scores on all tasks (RNA_HALF_LIFE ICC≈0: any model ≤≈0; must
  attribute "limited by measurement repeatability").
- polyA at 91% → moved out of the improvement narrative, into benchmark.
- Macro-average demoted to secondary; mandatory per-task table +
  source-group bootstrap CI.
- Decision metrics: hit@1 / NDCG@K (Recovery@K is inflated in 2–3 candidate
  neighborhoods — do not use).

### User rulings (final unless the user changes them)
- Ensemble (v5+f34 average): NOT doing it ("没什么用"). 3-seed start gate =
  V6 main criterion passing the gate itself.
- SSL pre-training: NOT doing it (mRNABERT already strong; strong init does
  not solve downstream imbalance; top-6 already trainable).
- SWA: not in V6 main line (demoted to zero-cost offline post-analysis).
- H2 (validation-peak reselection): permanently removed.

## 3. V6 acceptance gates (pre-registered; cannot be changed post-hoc)

| Gate | Definition |
|---|---|
| Main | pair-mean ρ vs V5's 0.103: pair-level bootstrap CI does not cross zero |
| Auxiliary | cell-offset head HEK293FT discrimination (AUC and/or ρ) |
| Ceiling | ceiling-normalized MPRAU 15% → 40%+ |
| MAE | double currency (transformed + original scale) |
| Discipline | CUDA/BF16 only; CPU fallback = stop + evidence; protected
  reads (DEV TEST / Eval outcome) = 0; exact-HEAD per arm; pre-registered
  gates never modified |

## 4. Wave 1/2/3 execution checklist

### Wave 1 (immediately parallel; CPU + GPU prep)
- W1-a pair-mean label aggregation switch (config default off): off = V5
  bit-identical; on = 12,048 MPRAU rows → 2,008 pair means.
- W1-b cell-offset auxiliary head: unit tests + HEK293FT discrimination
  computable.
- W1-c per-task rank transform switch: test intra-task monotone mapping.
- W1-d per-pass checkpoint (1–8) + per-pass validation metrics persisted
  (never touches TEST).
- W1-e eval output extension: within-source ρ / pair-mean ρ + ceiling ratio /
  hit@K / NDCG@K / Tier-B marker in run_summary.
- W1-f V6 first training (all approved items) on GPU (~15.5h); pre-registered
  gate acceptance.

### Wave 2 (W1-f starts → fill remaining GPUs immediately)
- W2-a H3 weight ablation ×3 (0.5 / 0.75 / 1.0), same seed 20260907; rest
  identical to W1-f; one arm per GPU.
- W2-b 2 additional seeds (3-seed total) of the V6 target arm.
- W2-c SWA offline post-analysis (pass 6/7/8 weight averaging).
- W2-d expert routing utilization diagnostics (read W1-f artifacts only).

### Wave 3 (after Wave 2 terminals)
- 3-seed ensemble average + common main-metric adjudication.
- H3 attribution with all three arms terminal.
- V7 candidate selection (gradient-norm-scaled loss first / BiLB4MTL /
  DB-MTL) after the V6 main metric lands.
- V8 (LoRA-ized top-6 / block reduction; protocol revision required to break
  120–180M parameter contract) only if both V6 and V7 are insufficient.
- SetFlow guidance (Potential-style) chain only when Critic passes AND
  SetFlow readiness passes.

## 5. Execution discipline (never violated)
1. Training/GPU validation must be CUDA (BF16); CPU fallback → stop + evidence.
2. Protected reads (Development TEST / Evaluation outcome) = 0.
3. Exact-HEAD: each arm in its own worktree + own config/auth + dual runner
   receipts; new HEAD never inherits old receipts.
4. V4.0.3/S1/v5/f34 historical terminals: no backfill, no rerun.
5. Per-pass validation metrics persisted (prerequisite for H1).
6. Pre-registered gates never modified post-hoc; negative results recorded,
   never renegotiated.
7. Parallel arms never share a GPU (one arm per card; expandable_segments:True;
   replicate retry1's PHYSICAL_GPU_INDICES pattern).