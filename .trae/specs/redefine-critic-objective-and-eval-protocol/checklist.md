# Checklist — V6 redefine critic objective & eval protocol

> Checkboxes are the gate-of-record for entering each later phase. Nothing may
> be marked done without an artifact path / test output to back it.

## A. Decision signoff (frozen 2026-08-31)
- [x] D0: role = prioritization ranker + cross-context annotator (B/C), not A.
- [x] D1: pair-mean main supervision; cell-offset aux head; per-task rank
      transform; LambdaRankIC pairwise (commit `7f01d17c`).
- [x] D2: ceiling-normalized pass; per-task table + source-group bootstrap CI;
      hit@1 / NDCG@K decision metrics.
- [x] User rulings recorded (no ensemble / no SSL / no SWA main / H2 removed).

## B. Free verification (read-only)
- [x] V5 terminal summary read correctly (task-macro 0.16709 / MAE 1.93476).
- [x] f34 same-seed comparison + bootstrap CI recorded ([−0.027, +0.040]).
- [x] Ceilings per task: 0.683 / 0.90 / ≈0 attributed.
- [x] No protected reads performed during diagnosis.

## C. Code implementation
- [x] LambdaRankIC default off = bit-identical to V5 (24 v4/v5 regressions +
      5 new tests pass).
- [x] W1-a pair-mean switch: default off identity; 12,048 → 2,008 collapse.
- [x] W1-b cell-offset head: forward shape, gradient direction, HEK293FT
      discrimination AUC tests pass.
- [x] W1-c rank-Gaussian switch: monotone intra-task transform tests pass.
- [x] W1-d per-pass checkpoint + per-pass validation metrics gated
      implementation present.
- [x] W1-e extended metrics (within-source ρ, pair-mean ρ + ceiling ratio,
      hit@K, NDCG@K, Tier-B) implemented, default off.
- [ ] Full regression suite green on the frozen V6 HEAD (core critic suites).

## D. Training protocol compliance
- [ ] CUDA/BF16 verified; CPU fallback evidence retained if any.
- [ ] Per-arm independent worktree + config + authorization + dual receipts.
- [ ] No two arms share a GPU; `expandable_segments:True`.
- [ ] Protected outcome reads = 0 across all runs.
- [ ] Pre-registered gates untouched.

## E. Training runs
- [ ] W1-f V6 first training launched on GPU and passed pre-registered gate.
- [ ] W2-a H3 ablation ×3 launched on separate GPUs.
- [ ] W2-b second and third seeds launched.
- [ ] SWA offline post-analysis produced.
- [ ] Expert routing utilization diagnostics produced.
- [ ] Wave 3: 3-seed ensemble + main-metric adjudication with bootstrap CI.

## F. 3-seeds readiness (start gate)
- [ ] 3 seeds terminal for the V6 target arm.
- [ ] Main pair-mean ρ CI vs 0.103 adjudicated and recorded (positive or
      negative, honestly).
- [ ] Code committed and pushed to GitHub after each confirmed task.