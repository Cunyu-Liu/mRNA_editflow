# Tasks — V6 redefine critic objective & eval protocol

> State legend: [x] done · [~] in progress · [ ] pending. Recovered 2026-08-31
> from `CURRENT_HANDOVER_STATUS_20260831.md` §5. Dependencies are strict.

## Task 0 — Spec & decision freeze
- [x] Freeze D0 (B≈C>A), D1 (pair-mean + offset + rank + LambdaRankIC), D2
      (ceiling-normalized, hit@1/NDCG@K, bootstrap CI).
- [x] Record user rulings (no ensemble, no SSL, no SWA in main, H2 removed).
- [x] Pre-register V6 acceptance gates (main = pair-mean ρ CI not crossing
      zero vs 0.103; aux = HEK293FT discrimination; ceiling 15%→40%+).

## Task 1 — Free verification (read-only, 2026-08-31)
- [x] V5 v5_full terminal read: task-macro 0.16709 / MAE 1.93476.
- [x] Same-seed f34 comparison: +4.07% ρ, −3.99% MAE; bootstrap CI
      [−0.027, +0.040], p=0.743 → no statistical evidence.
- [x] Ceiling analysis: MPRAU 0.683, polyA 0.90, HALF_LIFE ≈0 → gates by task.
- [x] F1–F16 diagnosis locked (compressor mechanism, mid/low lesion, data
      long-tails).

## Task 2 — Code implementation (V6 worktree)
- [x] SubTask 2.0 LambdaRankIC displacement-weighted pairwise (commit
      `7f01d17c`) + 5 unit tests; existing 24 v4/v5 tests pass; default off =
      bit-identical to V5.
- [x] SubTask 2.1 W1-a pair-mean label aggregation switch
      (`core/route2_xeditcritic_pair_mean_v6.py`); tests incl. 12,048 → 2,008
      collapse; default off.
- [x] SubTask 2.2 W1-b cell-offset auxiliary head
      (`core/route2_xeditcritic_cell_offset_v6.py`) + model wiring + HEK293FT
      discrimination probe; default off.
- [x] SubTask 2.3 W1-c per-task rank-Gaussian transform switch; tests; default
      off.
- [x] SubTask 2.4 W1-d per-pass checkpoint (pass_1..8) + per-pass validation
      metrics persisted; gated by config, default off.
- [x] SubTask 2.5 W1-e eval output extension: within-source ρ / pair-mean ρ +
      ceiling ratio / hit@K / NDCG@K / Tier-B marker in run_summary; gated,
      default off.
- [ ] Freeze training config for the V6 target arm (pair-mean ON + offset ON +
      rank ON + LambdaRankIC ON + within-source 0.5), build frozen launcher +
      authorization, run full regression on the frozen HEAD.

## Task 3 — Training protocol compliance (GPU)
- [ ] W1-f V6 first training (~15.5h, one A100 card).
- [ ] W2-a H3 weight ablation ×3 (0.5 / 0.75 / 1.0), same seed 20260907,
      separate cards.
- [ ] W2-b 2 additional seeds (3-seed family).
- [ ] W2-c SWA offline pass 6/7/8 averaging (CPU).
- [ ] W2-d expert routing diagnostics (CPU, read-only W1-f artifacts).
- [ ] Per-arm dual receipts + exact-HEAD + CUDA/BF16 verification + protected
      reads = 0.

## Task 4 — Acceptance & adjudication (Wave 3)
- [ ] 3-seed ensemble + common main-metric adjudication.
- [ ] H3 attribution with the three arms terminal.
- [ ] V6 main gate: pair-mean ρ vs 0.103, pair-level bootstrap CI not crossing
      zero; record negative results honestly if failed.
- [ ] Auxiliary gate: HEK293FT discrimination AUC/ρ.
- [ ] Ceiling-normalized progress 15% → 40%+ for MPRAU.
- [ ] V7 candidate selection only after V6 main metric is adjudicated.

## Task 5 — Governance
- [ ] Push committed code to GitHub after each confirmed task.
- [ ] Keep code/commits under /home; large data/weights/artifacts under
      /mnt/cunyuliu.
- [ ] Update the rapid iteration log (Iteration U) with F1–F16 + D0/D1/D2 +
      the Wave 1/2/3 checklist.
- [ ] No TEST / Evaluation outcome reads at any point.