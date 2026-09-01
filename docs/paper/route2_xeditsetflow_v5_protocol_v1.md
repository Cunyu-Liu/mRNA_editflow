# SetFlow V5 protocol v1 — base-model repair ablation screen

**Status:** FROZEN_PROSPECTIVE (2026-09-01; D1/D2/D5 decided, D3/D4 numbers here)
**Family:** `xeditsetflow_v5` (new family; V4/V4-S1 frozen terminal records untouched)
**Worktree:** `route_a_v3_setflow_v5_base_fix_20260901`
**GPU policy:** CUDA BF16 only; CPU fallback → immediate stop with evidence; GPU scope 0–7; no memory gate.

## 1. Objectives (guided-generation prep)

- **Base model role (V5):** legal, diverse, natural candidate generator — NOT a recovery oracle.
- **Recovery role (V5):** deferred to the guided stage (Task 4) using the frozen critic.
- This screen selects the repaired base model among the four ablation arms.

## 2. Pre-registered arms (one arm per GPU; F9)

| arm | mode_count | coverage_weight | arch profile | params (preflight-bound) | test |
|---|---|---|---|---|---|
| b_fix1 | 1 | 0.10 | V4_FULL (640/18/2560) | ≈98.6M | F1+F2+F3+F5 |
| b_fix2 | 1 | 0.00 | V4_FULL | ≈98.6M | same minus coverage |
| b_fix3 | 2 | 0.10 | V4_FULL | ≈100.0M | mixture re-check |
| b_arch1 | 1 | 0.10 | A1 (384/6/1536) | ≈12.7M | A1 small model |

Each arm: same screen seed 20260915, same train/validation projections, same
sampler, same LR/warmup/decay. All arms run concurrently on distinct GPUs.

**F-series configuration (all in screen config):**
- F1 early-stop schedule: `pass_count=6`, `saved_checkpoint_passes=[2,4,6]`; selection rule pre-registered (see §4).
- F2 coverage weight per arm (0.10 / 0.00 / 0.10).
- F3 validation temperature knobs: `mode_prior_temperature=1.0`, `stop_rate_scale=1.0` in the screen run (identity); a 5×5 frozen sweep is run afterward for diagnostics only.
- F4 single-mode default (mode_count=1); b_fix3 re-checks 2-mode.
- F5 regularization: weight_decay 0.001 (vs V4's 0.0001), dropout per profile.
- F6 validation dry-run smoke: every GPU long task is preceded by the config/schema/authorization chain guard + pytest green.
- F8 convergence curves: training summary records per-pass train loss; validation summaries record per-pass validation NLL; Gate B0 uses last-2-pass criterion.
- F9 multi-GPU: one arm per GPU, concurrent smoke-then-launch.

## 3. Data & protected boundaries

- TRAIN: 89,580 candidate rows (outcome-free) · VALIDATION projection: 18,293 rows / 15,327 source records / 891 eligible sources.
- TEST (18,292) / final Evaluation outcome: **protected reads = 0** throughout.
- Measured-neighborhood pool: DEVELOPMENT CLOSED; critic/independent-evaluator never enter training gradient.

## 4. Checkpoint selection rule (pre-registered, H2 ban honored)

For each arm, after ALL checkpoints pass outcome-free validation:
**selected checkpoint = argmin common-validation NLL among saved passes
(2,4,6) whose hard-legality rate == 1.0; ties → earliest pass.**
No post-hoc peak picking.

## 5. Gates (numbers frozen here; not amendable after launch)

- **Gate B0 (convergence):** training loss converged — relative drop of the mean
  train total loss over the last 2 passes < 5%. `converged` must be true.
- **Gate B1 (base quality):** selected checkpoint has
  `common NLL <= 2.068` AND `unique >= 0.85` AND `hard_legality == 1.0`.
- **Gate B2 (guided delta, Task 4):** guided recovery − unguided recovery >= +0.05
  with source-group bootstrap 95% CI not crossing zero; hit@1 not worse.
- **Gate B3 (system terminal, Task 4):** guided recovery >= 0.35.
- **Param band:** A1 arms 5M–20M; V4-size arms 80M–150M (preflight-bound).

Failure handling: a single arm over Gate B0/B1 reports a negative result
recorded with both curves; it does not block other arms. If no arm passes B1,
the screen is NO-GO and the fallback chain A1→A2→A3 is invoked with the B0/B1
evidence (per D5).

## 6. Authorization chain (exact-HEAD)

- Preflight (source audit + capacity + BF16 batch) bound to the worktree HEAD.
- Screen launch authorization binds the exact HEAD + the 4-arm package.
- Runner receipts: training summary + per-pass checkpoints; validation summary per pass.
- Dry-run smoke precedes all long GPU tasks (F6).

## 7. Deliverables

1. Four per-arm `training_summary.json` (with Gate B0) + per-pass checkpoints.
2. Outcome-free validation summaries per arm/pass (NLL / recovery / top-k / unique / legality).
3. `screen_gate.json` adjudicating Gates B0/B1 with selected checkpoint per arm.
4. Iteration-log Iteration U entry + handover appendix.
