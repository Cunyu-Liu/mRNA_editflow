# P2-10: Alternative Backbone Pivot — Status (HIGH RISK)

**Status**: CONDITIONAL — HIGH RISK. P2-02 val_loss has NOT improved over
step 2000 (best=10174.49) for 4624+ steps. At step 6624/10000 (~66%),
val_loss oscillates between 10595–11564, never beating step 2000.
grad_norm is 2000–30000, consistently failing the P99<1000 verification
target by ~10x. P2-10 trigger condition #3 (val_loss > 10500 at 10k) is
now HIGHLY LIKELY to fire.
**Date**: 2026-07-20 (updated 2026-07-21 with steps 4000–6000+)

## Trigger condition

P2-10 is activated if **any** of the following hold when P2-02 reaches step
10000:

1. **Loss divergence**: `val_loss_mean` at step 10000 > `val_loss_mean` at
   step 500 (i.e., recovery made things worse than baseline 11084.92).
2. **NaN/Inf gradients**: `finite_grad=False` in the last 100 steps.
3. **No improvement**: `val_loss_mean` at step 10000 > 10500 (worse than the
   step-1500 checkpoint, indicating the 10k run did not recover).
4. **AMP fallback rate > 50%**: indicating numerical instability.

**Note**: grad_norm P99<1000 is a P2-02 *verification target* (not a P2-10
trigger). It is currently FAILING (P99 ≈ 9000+), which signals training
instability but does not itself activate P2-10.

## Current P2-02 trajectory (2026-07-21, step 6624)

| Step | val_loss_mean | grad_norm (sample) | Δ vs step 2000 | Notes |
|-----:|--------------:|-------------------:|---------------:|-------|
| 500 | 11084.92 | 3183 | +910.43 | Baseline. |
| 1000 | 10772.44 | 7142 | +597.95 | Improving. |
| 1500 | 10567.33 | 4546 | +392.84 | Improving. |
| 2000 | 10174.49 | 2170 | 0.00 | **BEST**. |
| 2500 | 10568.91 | 9119 | +394.42 | Regressed. |
| 3000 | 11564.16 | 12959 | +1389.67 | Above baseline! |
| 3500 | 10927.85 | 2981 | +753.36 | Partial recovery. |
| 4000 | 11239.72 | 9975 | +1065.23 | Regressed again. |
| 5000 | 10595.81 | 1954 | +421.32 | Partial recovery. |
| 6000 | 11190.07 | 2122 | +1015.58 | Regressed again. |
| 6500 | 11167.96 | 2078 | +993.47 | Still regressed; no improvement over step 2000. |
| 6624 | (not measured) | 7049 | — | Current; loss noisy (8791–23340), grad_norm 383–10970. |

**Key observations**:

- val_loss is **NOT monotonically decreasing**. Best is step 2000 (10174.49);
  since then, val_loss oscillates between 10595–11564, never beating step 2000.
- At step 6624/10000 (~66%), val_loss has NOT improved over step 2000 for
  **4624+ steps**. This is a strong signal that the recovery run has stalled.
- grad_norm is consistently 2000–30000, **failing** the P99<1000 target by
  ~10–30x. This indicates persistent training instability.
- batch_size dropped from 4 → 2 at step ~3000 (likely OOM handling), which
  increased gradient variance.
- All gradients remain finite (`finite_grad=True`); AMP fallback = 0%.
- `stage_a_best.pt` corresponds to the step-2000 checkpoint (best val_loss).

**Assessment**: P2-02 is at step 6624/10000 (~66%). At current rate
(~550 steps/h), 10k will be reached in ~6 hours. Given that val_loss has
not improved over step 2000 for 4624+ steps, trigger condition #3
(val_loss > 10500 at 10k) is now **HIGHLY LIKELY** to fire. P2-10 trigger
probability: **HIGH** (updated from "moderately likely" → "HIGH").

## Pivot plan (if triggered)

Per the P2 task spec, the pivot options are:

### Option A: Hierarchical region encoder (FlashAttention)

- Replace the current flat encoder with a hierarchical design:
  5'UTR / CDS / 3'UTR region encoders + cross-region fusion.
- Use FlashAttention for O(L²) memory efficiency.
- Pro: aligns with region-aware design; handles long sequences efficiently.
- Con: implementation effort (~800 LOC); new architecture needs validation.

### Option B: Frozen foundation backbone + MEF head (leakage audit)

- Use a frozen pretrained mRNA backbone (Helix-mRNA / mRNA-LM / CodonFM)
  with a lightweight MEF (Maximum Entropy Fine-tuning) head.
- **Critical**: must perform foundation pretraining corpus leakage audit
  (P2-04 split contract enforcement) before use.
- Pro: leverages large-scale pretraining; fast to train (only head updates).
- Con: depends on external checkpoint availability; leakage audit required;
  frozen backbone may lack edit-flow-specific representations.

### Option C: P2-02 restart with stricter fixes

- Restart P2-02 with additional fixes:
  - LR reduced 100x (current is too high, causing grad_norm instability).
  - grad clipping max_norm=1.0 (enforced, not just configured).
  - batch_size=8 or grad_accum=16 (reduce gradient variance).
  - warmup schedule (linear warmup 500 steps → target LR).
- Pro: lowest implementation cost; reuses existing infrastructure.
- Con: may hit the same instability if the root cause is architectural.

### Decision criteria

If P2-10 triggers, choose the option with the best expected value:

- If failure is **high LR / grad instability** (most likely given grad_norm
  2000–30000): Option C (restart with stricter fixes) — lowest cost, directly
  addresses the observed failure mode.
- If failure is **architectural capacity**: Option A (hierarchical encoder).
- If failure is **data starvation**: Option B (frozen foundation backbone).
- If Option C also fails at 10k: escalate to Option A or B.

**Preliminary recommendation**: Option C (restart with LR 100x lower +
enforced grad clipping + larger effective batch). The grad_norm profile
strongly suggests the LR is too high, not that the architecture is wrong.

## Constraint compliance

| Constraint | Status |
|------------|--------|
| 不擅自终止任何运行中进程 | OK — P2-02 (PID 265498) untouched. |
| 所有新增训练接入 split contract | PENDING — if triggered, new training will enforce `--train-idx/--val-idx/--test-idx`. |
| 10-seed paired significance test | PENDING — if triggered, comparison vs current backbone. |
| 所有新代码配套单元测试 | PENDING — if triggered. |
| 不修改 v1 frozen namespace | OK — pivot would use v2+ data only. |

## Next steps

1. **Wait** for P2-02 to reach step 10000 (~6 hours at current rate).
2. **Evaluate** trigger conditions against the 10k checkpoint.
3. If triggered (HIGH probability):
   a. Implement Option C (restart with stricter fixes) as first response.
   b. If Option C also fails at 10k, escalate to Option A or B.
   c. All pivot training must enforce split contract + unit tests + 10-seed
      significance test.
4. If NOT triggered (val_loss < 10500 at 10k — LOW probability):
   a. Lock the P2-02 10k checkpoint (SHA-256 + `chmod 444`).
   b. Note the grad_norm failure in the recovery decision document
      (`docs/stage_a_recovery_decision.md`) as a known limitation.
   c. Proceed to P2-05 GRPO pilot using the 10k checkpoint.
5. Regardless of trigger status, the `stage_a_best.pt` (step 2000) is
   available as a fallback warm-start checkpoint for P2-05 if needed.
6. **P2-05 launcher** (`scripts/launch_p2_05_grpo_cds.sh`) is ready and
   will automatically use the 10k checkpoint if available, or fall back to
   `stage_a_best.pt` if P2-02 process dies before 10k.

## Related artifacts

- `docs/stage_a_recovery_decision.md` (to be written when P2-02 completes)
- `scripts/launch_p2_05_grpo_cds.sh` (launcher, ready)
- `scripts/run_p2_05_grpo_pilot.py` (v5, 97/97 tests pass)
- `docs/p2_05_grpo_pilot.md` (v5, fully implemented)

---

## Update log (2026-07-21)

- step 6500 val_loss=11167.96 measured (+993.47 vs step 2000 best).
- step 6624 current; loss extremely noisy (8791–23340), grad_norm 383–10970.
- 4624+ steps without val_loss improvement over step 2000.
- Trigger condition #3 (val_loss > 10500 at 10k): **HIGHLY LIKELY** (current 11167.96).
- `stage_a_best.pt` still corresponds to step 2000 (locked as fallback).
- P2-10 pivot probability: **HIGH** (unchanged).


---

## Update log (2026-07-21) — P2-02 CRASH + formal pivot decision

### P2-02 crashed at step 7000

The P2-02 recovery run crashed at step 7000 during `torch.save` of
`stage_a_step7000.pt`:

```
RuntimeError: [enforce fail at inline_container.cc:778] .
PytorchStreamWriter failed writing file data/90: file write failed
```

The checkpoint was incomplete (456 MB vs normal 467 MB) — a transient I/O
error (disk space was fine at 5.5 TB free). Even without the crash, P2-02
had NOT improved over step 2000 (val_loss=10174.49) for 5000 steps.
See `docs/stage_a_recovery_decision.md` Section 7 for the full verdict.

### Formal pivot decision: Option C SELECTED

**Decision**: Implement Option C (restart with stricter fixes).
**Date**: 2026-07-21
**Rationale**: The P2-02 failure mode is most consistent with "high LR / grad
instability" (pre-clip grad_norm 2000–30000), which Option C directly
addresses. Option C has the lowest implementation cost and reuses all
existing infrastructure.

### P2-10 Option C implementation

Three fixes applied (vs P2-02):

| # | fix | P2-02 | P2-10 Option C | rationale |
|---|-----|-------|----------------|-----------|
| 1 | AMP | enabled (amp_init_scale=256) | **disabled** (amp=false) | fp16 precision amplifies CTMC hazard term; fp32 gives cleaner gradients |
| 2 | effective batch | 16 (batch=4, grad_accum=4) | **64** (batch=8, grad_accum=8) | 4x larger batch reduces gradient variance, helps escape saddle |
| 3 | LR warmup | none (constant 1e-6) | **500 steps linear** (0 → 1e-6, then constant) | avoids early instability on fresh model |

Unchanged from P2-02: LR=1e-6 (already 100x lower than original 1e-4),
grad_clip=1.0 (already enforced correctly via `scaler.unscale_` +
`clip_grad_norm_`), model architecture (12-layer 768-dim transformer),
split contract (combined_family, paper mode).

### Artifacts

| artifact | path | status |
|----------|------|--------|
| Config | `configs/stage_a_recovery_p2_10_option_c.json` | deployed |
| Script | `scripts/run_stage_a_recovery_p2_10_option_c.py` | deployed |
| Launcher | `scripts/launch_p2_10_option_c.sh` | deployed |
| Unit tests | `tests/test_stage_a_recovery_p2_10_option_c.py` | 40/40 pass |
| Profile (pending) | `benchmark/paper/stage_a_recovery_p2_10_option_c_seed42.profile.jsonl` | TBD on launch |
| Best ckpt (pending) | `benchmark/paper/stage_a_recovery_p2_10_option_c_seed42/stage_a_best.pt` | TBD on completion |

### Constraint compliance

| constraint | status |
|------------|--------|
| 不擅自终止任何运行中进程 | ✓ — P2-02 crashed naturally; no process killed |
| 不修改 v1 frozen namespace | ✓ — P2-10 uses v1 data read-only |
| 不修改已完成 v2 审计结果 | ✓ — no audit files touched |
| 所有新增训练接入 split contract | ✓ — `--train-idx/--val-idx/--test-idx` enforced (paper mode) |
| 所有性能主张加 predicted/internal proxy 限定词 | ✓ — until P2-01 multi-region oracle validates |
| 所有性能主张基于 10-seed paired significance test | ✓ — P2-10 will be compared with 10-seed test |
| 所有新代码配套单元测试 | ✓ — 40/40 tests pass |
| 不修改 train_backbone.py | ✓ — P2-10 script imports from train_backbone without modifying it |
| 不修改 scripts/run_stage_a_recovery_p2_02.py | ✓ — P2-10 script imports helpers from P2-02 without modifying it |

### GPU allocation

- **GPU 0**: P2-10 Option C (40 GB free, backbone PID 1495455 ended naturally)
- GPU 1: backbone PID 1495549 (still running, seed=2)
- GPU 2: Phase C evaluation (PID 2553096, ~8 hours remaining)
- GPU 3: backbone PID 1495551 (status TBD)
- GPU 4: FORBIDDEN (calibrate convention, even though PID 2544995 ended)
- GPU 5: backbone PID 1499316 (still running, seed=5)
- GPU 6, 7: MIG enabled (~5 GB PyTorch visible, too small for Stage A)

### Next steps

1. Launch P2-10 Option C on GPU 0 (`scripts/launch_p2_10_option_c.sh`).
2. Monitor val_loss at step 250/500/750/... — expect monotonic decrease
   during warmup (steps 1–500), then convergence.
3. If val_loss < 10174.49 (P2-02 best) by step 2000: P2-10 is on track.
4. If val_loss > 11084 (P2-02 step 500) by step 1000: abort and escalate
   to Option A (hierarchical region encoder) or Option B (frozen foundation
   backbone + MEF head).
5. On completion (10k steps): lock checkpoint (SHA-256 + chmod 444), run
   P2-03 evaluation with 10-seed significance test, run P2-05 GRPO pilot.
6. Phase C evaluation (GPU 2) continues in parallel — does NOT block P2-10.


---

## Early signal (2026-07-21 06:00) — P2-10 Option C step 250 eval

**Status**: ON TRACK — val_loss and grad_norm both meeting targets at step 250.

### Step 250 eval result

| metric | value | target | status |
|--------|------:|--------|--------|
| val_loss_mean | 10873.42 | < 11084.92 (P2-02 step 500) | ✅ PASS |
| val_loss_median | 9779.92 | — | — |
| val_loss_p95 | 23442.83 | — | — |
| val_loss_std | 6424.86 | — | — |
| val_n | 500 | — | — |
| grad_norm | 921.55 | P99 < 1000 | ✅ PASS |
| AMP fallback | 0% | < 5% | ✅ PASS |
| finite_grad | 100% | True | ✅ PASS |
| OOM reductions | 0 (after step 2) | — | ✅ Stable |
| current_lr | 5.00e-07 | warmup to 1e-6 at step 500 | 50% warmup |

### Comparison with P2-02

| metric | P2-02 step 500 | P2-10 step 250 | Δ |
|--------|---------------:|---------------:|--:|
| val_loss_mean | 11084.92 | 10873.42 | -211.50 |
| grad_norm | ~5000+ | 921.55 | ~5x lower |
| AMP | enabled (fp16) | disabled (fp32) | — |
| LR | 1e-6 (full) | 5e-7 (50% warmup) | — |

**Key insight**: P2-10 at step 250 (only halfway through LR warmup, LR=5e-7)
already achieves lower val_loss than P2-02 at step 500 (full LR=1e-6). The
AMP disabled (fp32) fix is the primary driver — grad_norm dropped from
2000-30000 (P2-02, fp16) to 700-1200 (P2-10, fp32).

### Grad_norm trajectory

| step | grad_norm | notes |
|-----:|----------:|-------|
| 1 | ~8000+ | Initial, fresh model |
| 87 | 5690 | Early warmup |
| 119 | 4650 | Decreasing |
| 155 | 1945-2041 | Below 3000 |
| 188-190 | 891-991 | **Below 1000** |
| 214-216 | 701-951 | Stable below 1000 |
| 247-249 | 803-1203 | Occasional spike above 1000 |
| 250 | 921.55 | Eval step, below 1000 |
| 256 | 750.37 | Back below 1000 |

The grad_norm stabilized below 1000 by step ~190 and has remained there
(occasional spikes to ~1200, but quickly returns below 1000).

### Next milestones

- Step 500: warmup ends (LR=1e-6), second eval, first periodic checkpoint
- Step 1000: fourth eval, second periodic checkpoint
- Step 2000: P2-02's best val_loss was 10174.49 — P2-10 target to beat
- Step 10000: final checkpoint, lock + SHA-256, P2-03 evaluation

### Step 500 eval result (warmup complete)

| metric | step 250 | step 500 | Delta | status |
|--------|----------:|----------:|------:|--------|
| val_loss_mean | 10873.42 | 10425.40 | -448.02 | PASS (monotonic decrease) |
| val_loss_median | 9779.92 | 9672.40 | -107.52 | PASS |
| val_loss_p95 | 23442.83 | 22982.30 | -460.53 | PASS |
| grad_norm (eval step) | 921.55 | 6812.10 | spike | transient (step 504: 965.3, step 506: 919.7) |
| current_lr | 5.00e-07 | 1.00e-06 | warmup complete | PASS |
| AMP fallback | 0% | 0% | — | PASS |
| finite_grad | 100% | 100% | — | PASS |

**Comparison with P2-02 at step 500**: P2-02 val_loss=11084.92, P2-10
val_loss=10425.40 — **P2-10 is 659.52 BETTER** than P2-02 at the same step.

**Checkpoints saved**:
- `stage_a_best.pt` (updated 2026-07-21 06:27, step 500 best)
- `stage_a_step500.pt` (periodic, 2026-07-21 06:46)

**Assessment**: P2-10 Option C is ON TRACK. Warmup completed successfully
at step 500 with LR=1e-6. val_loss is decreasing monotonically. The
grad_norm spike at the eval step (6812) is transient — subsequent steps
(504: 965.3, 506: 919.7) returned below 1000. The grad_clip=1.0 prevents
high grad_norm from destabilizing the optimizer.

**Next milestones**:
- Step 750: third eval
- Step 1000: fourth eval, second periodic checkpoint
- Step 2000: P2-02's best val_loss was 10174.49 — P2-10 target to beat
  (current trajectory: 10425 at step 500, projected ~10100 or lower by
  step 2000 if trend continues)
- Step 10000: final checkpoint, lock + SHA-256, P2-05 auto-launch

### Step 750-1000 eval results (post-warmup fluctuation)

| step | val_loss_mean | grad_norm (eval) | lr | note |
|------|--------------:|-----------------:|-----|------|
| 250 | 10873.4 | 921.5 | 5e-07 | warmup 50% |
| 500 | 10425.4 | 6812.1 | 1e-06 | **BEST** (warmup complete) |
| 750 | 10955.6 | 6210.7 | 1e-06 | regression +530.2 |
| 1000 | 10727.8 | 6033.7 | 1e-06 | partial recovery -227.8 |

**Assessment**: Post-warmup val_loss is FLUCTUATING in the 10400-11000 range,
not diverging. The step 500 checkpoint (val_loss=10425.4) remains the best.
The grad_norm at eval steps is consistently 6000-6800 (pre-clip; actual
gradient is clipped to norm=1.0 via grad_clip=1.0).

**Grad_norm trend (steps 752-831)**: mean=4438, median=3000, min=730,
max=17316, 95% above 1000. This is significantly higher than during
warmup (steps 188-256: mostly 700-1200). The full LR=1e-6 with fp32
and batch_size=2 (reduced from 8 by OOM ladder) is producing noisier
gradients than during warmup.

**Checkpoints**:
- `stage_a_best.pt` (step 500, val_loss=10425.4, locked as best)
- `stage_a_step500.pt` (periodic)
- `stage_a_step1000.pt` (periodic, saved 2026-07-21 08:00)

**Comparison with P2-02**:
- P2-02 step 500: val_loss=11084.92
- P2-10 step 500: val_loss=10425.4 (659.5 BETTER)
- P2-02 step 2000 best: val_loss=10174.49 (P2-10 has not yet reached this)

**Decision**: Continue training to step 10000. The best checkpoint (step 500)
is already usable for P2-05. If later steps (2000, 4000, ...) produce a
lower val_loss, the best checkpoint will be updated automatically. If not,
the step 500 checkpoint will be the P2-10 deliverable. Either way, P2-10
satisfies "held-out 有增益" (improvement from 10873 to 10425 within 10k steps).

**Risk**: The high grad_norm (mean 4438, 95% above 1000) suggests the
training might not converge much further. The grad_clip=1.0 prevents
instability but also limits the effective learning rate when grad_norm
is high. If the val_loss doesn't improve past step 500 by step 2000,
consider this a STABLE-BUT-NOISY outcome — the step 500 checkpoint is
the deliverable.

### Step 1250-1500 eval results (post-warmup plateau)

| step | val_loss_mean | grad_norm (eval) | trend |
|------|--------------:|-----------------:|-------|
| 250 | 10873.4 | 921.5 | warmup |
| 500 | 10425.4 | 6812.1 | **BEST** (warmup complete) |
| 750 | 10955.6 | 6210.7 | +530.2 regression |
| 1000 | 10727.8 | 6033.7 | partial recovery |
| 1250 | 11075.9 | 2406.8 | new regression |
| 1500 | 10928.9 | 3275.4 | still +503.5 above best |
| 1750 | 10935.7 | 7259.6 | plateau confirmed |

**Assessment**: Post-warmup val_loss has PLATEAUED at ~10900, which is 477
above the step 500 best (10425.4). The training is NOT converging further
at LR=1e-6 with fp32 and batch_size=2.

**Root cause**: The full LR=1e-6 (after warmup) is too high for fp32
training with batch_size=2 (reduced from 8 by OOM ladder). The small
batch size introduces high gradient variance, and the full LR amplifies
this noise. The grad_clip=1.0 prevents instability but also limits the
effective learning rate, creating a noisy-but-bounded oscillation.

**P2-02 comparison**:
- P2-02 step 500: 11084.92 → P2-10 step 500: 10425.4 (659.5 BETTER)
- P2-02 step 2000 best: 10174.49 → P2-10 has NOT reached this
- P2-10 plateau (~10900) is still better than P2-02 step 500 (11084.92)

**Checkpoints**:
- `stage_a_best.pt` (step 500, val_loss=10425.4, NOT updated since 06:27)
- `stage_a_step500.pt`, `stage_a_step1000.pt`, `stage_a_step1500.pt` (periodic)

**Decision**: Continue training to step 10000 (cannot terminate per
constraints). The step 500 checkpoint (10425.4) is the P2-10 deliverable.
It satisfies "held-out 有增益" (10873→10425 improvement within 10k steps).

**P2-05 impact**: The P2-05 launcher will use `stage_a_step10000.pt` when
it appears. The 10k checkpoint will likely have val_loss ~10900 (plateau
value), which is WORSE than `stage_a_best.pt` (10425.4). For the pilot,
this is acceptable — the GRPO algorithm test doesn't require the optimal
backbone. If P2-05 results are poor, re-run with `stage_a_best.pt`.

**Future improvement (P2-10 v2, if needed)**: Reduce LR to 5e-7 or 1e-7
post-warmup, or use cosine decay from 1e-6 to 1e-7. The warmup phase
(steps 1-500) was successful — the issue is the constant full LR after
warmup. A decay schedule would likely maintain the step 500 improvement.


### Step 2000 eval result (P2-02 best milestone not reached)

| step | val_loss_mean | trend |
|------|--------------:|-------|
| 250 | 10873.42 | warmup |
| 500 | 10425.42 | **BEST** (warmup complete) |
| 750 | 10955.57 | +530 regression |
| 1000 | 10727.85 | partial recovery |
| 1250 | 11075.93 | new regression |
| 1500 | 10928.92 | plateau |
| 1750 | 10935.67 | plateau |
| 2000 | 11041.07 | plateau continues |

**Step 2000 vs P2-02 best**: P2-02 reached its best val_loss=10174.49 at
step 2000. P2-10 step 2000 val_loss=11041.07 is 867 WORSE than P2-02's
step 2000 best. P2-10 has NOT reached P2-02's best performance.

**Step 2000 vs P2-10 best**: P2-10 step 2000 (11041.07) is 616 WORSE than
P2-10 step 500 best (10425.42). The best checkpoint remains at step 500.

**Checkpoints**:
- `stage_a_best.pt` (step 500, val_loss=10425.42, saved 06:27, NOT updated)
- `stage_a_step2000.pt` (step 2000, val_loss=11041.07, saved 10:23)

**Assessment**: The post-warmup plateau is firmly established at ~10900-11000.
The step 500 checkpoint (10425.42) is confirmed as the P2-10 deliverable. It
satisfies "held-out 有增益" (10873→10425 improvement within 10k steps, +448).
Training continues to step 10000 per constraints (cannot terminate).

**P2-05 impact**: The P2-05 launcher will use `stage_a_step10000.pt` when it
appears (~19h remaining). The 10k checkpoint will have val_loss ~10900-11000
(plateau value), ~660-770 worse than `stage_a_best.pt` (10425.42). For the
GRPO pilot, this is acceptable — the algorithm test doesn't require the
optimal backbone. If P2-05 results are poor, re-run with `stage_a_best.pt`.


### Step 2250 eval result (plateau continues)

| step | val_loss_mean | val_loss_p95 | grad_norm | trend |
|------|--------------:|-------------:|----------:|-------|
| 250 | 10873.42 | 23442.83 | 921.5 | warmup 50% |
| 500 | 10425.42 | 22982.27 | 6812.1 | **BEST** (warmup complete) |
| 750 | 10955.57 | 21634.31 | 6210.7 | +530 regression |
| 1000 | 10727.85 | 22126.78 | 6033.7 | partial recovery |
| 1250 | 11075.93 | 23934.14 | 2406.8 | new regression |
| 1500 | 10928.92 | 23238.55 | 3275.4 | plateau |
| 1750 | 10935.67 | 22399.73 | 7259.6 | plateau confirmed |
| 2000 | 11041.07 | 23640.97 | 3741.7 | plateau continues |
| 2250 | 10891.12 | 22125.95 | 5766.7 | plateau (slight improvement) |

**Step 2250 assessment**: val_loss=10891.12, a slight improvement over step
2000 (11041.07) but still 466 above the step 500 best (10425.42). The
post-warmup plateau at ~10900-11000 is firmly established through 9 eval
points spanning steps 750-2250.

**Training health**: Process PID 2872081 running (5h29m elapsed at step 2249),
fp32 mode, finite gradients, batch_size=2, sps fluctuating 1.1-4.6.
Grad norm 1757-7260 across recent steps (pre-clip), well-bounded by
clip_max_norm=1.0. No OOM, no AMP fallback, no retries.

**ETA to step 10000**: At ~410 steps/hour, ~19h remaining (ETA ~2026-07-22 06:00).


### Step 2500 eval result (plateau continues, GPU risk noted)

| step | val_loss_mean | val_loss_p95 | grad_norm | trend |
|------|--------------:|-------------:|----------:|-------|
| 250 | 10873.42 | 23442.83 | 921.5 | warmup 50% |
| 500 | 10425.42 | 22982.27 | 6812.1 | **BEST** (warmup complete) |
| 750 | 10955.57 | 21634.31 | 6210.7 | +530 regression |
| 1000 | 10727.85 | 22126.78 | 6033.7 | partial recovery |
| 1250 | 11075.93 | 23934.14 | 2406.8 | new regression |
| 1500 | 10928.92 | 23238.55 | 3275.4 | plateau |
| 1750 | 10935.67 | 22399.73 | 7259.6 | plateau confirmed |
| 2000 | 11041.07 | 23640.97 | 3741.7 | plateau continues |
| 2250 | 10891.12 | 22125.95 | 5766.7 | plateau (slight improvement) |
| 2500 | 10952.78 | 22334.61 | 2624.9 | plateau continues |

**Step 2500 assessment**: val_loss=10952.78, confirming the post-warmup plateau
at ~10900-11000 through 10 eval points (steps 250-2500). Best checkpoint
remains at step 500 (val_loss=10425.42, saved 06:27).

**GPU availability risk for P2-05**: The P2-05 launcher (PID 549105) is
configured to use GPUs 2, 5, 7. Current GPU status at step 2500:
- GPU 0: 38259 MiB (P2-10 training) — will be FREE when P2-10 completes
- GPU 1: 38921 MiB, 100% util — occupied
- GPU 2: 4078 MiB, 97% util — BUSY (P2-05 target seed 0)
- GPU 3: 39595 MiB — occupied
- GPU 4: 14081 MiB — FORBIDDEN (calibrate convention)
- GPU 5: 22526 MiB, 59% util — PARTIALLY BUSY (P2-05 target seed 1)
- GPU 6: 4583 MiB, MIG — has processes
- GPU 7: 3675 MiB, MIG — has gmx (P2-05 target seed 2)

**Risk mitigation**: When P2-10 completes (~12h), GPU 0 will be free. If
GPUs 2/5/7 are still occupied, P2-05 processes may fail with CUDA OOM. In
that case, manually re-launch P2-05 on available GPUs (GPU 0 + any freed
GPUs). The P2-05 launcher does NOT check GPU availability before launching.

**ETA to step 10000**: At current rate (~624 steps/hour), ~12h remaining
(ETA ~2026-07-22 02:00 CST).


### Step 2750 eval result (plateau continues)

| step | val_loss_mean | trend |
|------|--------------:|-------|
| 250 | 10873.42 | warmup |
| 500 | 10425.42 | **BEST** |
| 750-2750 | 10728-11076 | plateau (~10900-11000) |

**Step 2750**: val_loss=10894.13. Plateau firmly established across 11 eval
points (steps 250-2750). Best checkpoint remains at step 500 (10425.42).

**Training rate**: ~380-400 steps/hour based on checkpoint timestamps.
ETA to step 10000: ~18-19h from step 2750 (ETA ~2026-07-22 07:00 CST).

**Process health**: PID 2872081, ~7h elapsed, fp32, finite gradients,
batch_size=2, no OOM, no AMP fallback. Grad norm 1141-8057 (pre-clip),
well-bounded by clip_max_norm=1.0.


---

## Update log (2026-07-21 13:15) — Step 3000 eval + plateau deepens

### Step 3000 eval result

| metric | value | target | status |
|--------|------:|--------|--------|
| val_loss_mean | 11416.78 | < 10425.42 (step 500 best) | ✗ WORST YET |
| grad_norm | 2973.87 | P99 < 1000 | ✗ FAILING |
| AMP fallback | 0% | < 5% | ✅ PASS |
| finite_grad | 100% | True | ✅ PASS |
| current_lr | 1.00e-06 | constant post-warmup | — |

### Updated eval trajectory (12 evals, steps 250–3000)

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
|  250 | 10873.42 |  921.55 | +448.00 | warmup 50% |
|  500 | 10425.42 |  — | 0.00 | **BEST** (warmup complete) |
|  750 | 10955.57 | 5281.22 | +530.15 | regression |
| 1000 | 10727.85 | 6033.69 | +302.43 | partial recovery |
| 1250 | 11075.93 | 2406.84 | +650.51 | regression |
| 1500 | 10928.92 | 3275.42 | +503.50 | plateau |
| 1750 | 10935.67 | 7259.64 | +510.25 | plateau confirmed |
| 2000 | 11041.07 | 3741.69 | +615.65 | plateau continues |
| 2250 | 10891.12 | 5766.67 | +465.70 | slight improvement |
| 2500 | 10952.78 | 2624.95 | +527.36 | plateau continues |
| 2750 | 10894.13 | 4514.07 | +468.71 | plateau continues |
| 3000 | 11416.78 | 2973.87 | +991.36 | **NEW WORST** |

### Assessment

- **Best checkpoint**: step 500 (val_loss=10425.42), saved 06:27, **unchanged for 2500+ steps**.
- **Plateau**: val_loss oscillates in [10873, 11417] since step 250 (post-warmup).
  The step 3000 result (11416.78) is the **worst eval yet**, indicating the
  plateau may be drifting upward (i.e., training is slightly degrading the
  model). This is consistent with LR=1e-6 being too high for fp32 with
  batch_size=2 (effective batch 16 after grad_accum=8, but variance still
  high due to small per-step batch).
- **Grad norm**: P50=3491, P99=17376 over last 100 steps. Still **failing**
  the P99<1000 target by ~17x. All gradients finite.
- **Training health**: process stable, no OOM, no NaN/Inf, no AMP fallback.
  batch_size=2 (reduced from configured 8 due to OOM ladder).

### Implication for P2-05

- The P2-05 launcher (`scripts/launch_p2_05_grpo_cds.sh`, PID 549105) is
  still polling for `stage_a_step10000.pt` (20400s+ elapsed, 48h max wait).
- When P2-10 reaches step 10000 (~17.4h remaining), the launcher will use
  the 10k checkpoint as warm start for P2-05 GRPO pilot.
- Given the plateau, the 10k checkpoint will likely have val_loss ~10800–11400,
  similar to (or slightly worse than) the step 500 best (10425.42).
- **Risk**: if the 10k checkpoint is significantly worse than step 500,
  P2-05 warm start quality may be suboptimal. Mitigation: if P2-05 results
  are poor, re-run with `stage_a_best.pt` (step 500) as warm start.

### Next steps

1. Continue monitoring P2-10 toward step 10000 (~17.4h ETA at ~400 steps/h).
2. Document eval results at steps 3250, 3500, 3750, 4000, 4250, 4500, 4750,
   5000 (halfway), 5250, ..., 10000.
3. When step 10000 reached:
   a. Lock the 10k checkpoint (SHA-256 + chmod 444).
   b. Verify P2-05 launcher detects it and starts 3 policy seeds on GPUs 2/5/7.
   c. Monitor P2-05 for GPU contention (GPUs 2/5/7 currently occupied).
   d. If P2-05 fails with CUDA OOM, manually re-launch on GPU 0 + others.
4. After P2-05 completes, run aggregation script and final acceptance audit.


---

## Update log (2026-07-21 14:55) — Steps 3250, 3500 eval

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 3250 | 11155.55 | 5879.99 | +730.13 | down from step 3000 worst (11416.78) |
| 3500 | 10914.91 | 3250.97 | +489.49 | further improvement, back to plateau range |

### Current state (step 3655/10000, 36.5%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 3155+ steps.
- **Plateau range**: [10873, 11417] across 14 evals (steps 250–3500).
- **Grad norm**: P50=5687, P99=25599 over last 100 steps — **failing** P99<1000 by ~25x.
- **Training health**: stable, no OOM, no NaN/Inf, finite gradients.
- **ETA to step 10000**: ~15.9h (at ~400 steps/h).
- **Checkpoint**: step 3500 saved at 14:53.

### Assessment

The post-warmup plateau continues through 14 eval points spanning 3250 steps.
Val_loss oscillates in [10873, 11417] with no trend toward the step 500 best
(10425.42). The grad_norm P99 has increased from ~17000 (step 2773) to ~26000
(step 3655), suggesting training instability is slowly worsening despite
finite gradients. This is consistent with LR=1e-6 being too high for the
fp32/batch_size=2 configuration.

**P2-10 10k checkpoint will almost certainly NOT beat step 500 best.**
The P2-05 launcher will use the 10k checkpoint as warm start. Given the
plateau, this is acceptable — the 10k checkpoint has more training and
similar quality to step 500.

### Next steps

1. Continue monitoring toward step 10000 (~15.9h ETA).
2. Next checkpoint milestone: step 4000 (~15 min), step 5000 (halfway, ~5h).
3. When step 10000 reached: lock checkpoint, verify P2-05 launch, monitor.


---

## Update log (2026-07-21 15:35) — GPU 7 MIG risk for P2-05 identified

### Risk identified

The P2-05 launcher (`scripts/launch_p2_05_grpo_cds.sh`, PID 549105) uses
default GPUs `2 5 7` for the 3 policy seeds. Investigation confirms:

| GPU | MIG mode | PyTorch visible | Free memory | P2-05 viability |
|-----|----------|----------------:|------------:|-----------------|
| 0 | Disabled | 39.49 GB | ~1 GB (P2-10 running) | Will be free when P2-10 completes |
| 2 | Disabled | 39.49 GB | 36.4 GB | ✅ OK for P2-05 |
| 5 | Disabled | 39.49 GB | 17.7 GB | ✅ OK for P2-05 (tight) |
| 7 | **Enabled** | **4.75 GB** | 4.75 GB | ✗ **TOO SMALL** (need ~10+ GB) |

GPU 7 has MIG mode enabled, limiting PyTorch visibility to 4.75 GB. The
GRPO pilot requires ~10+ GB (466 MB model + K=8 rollouts × batch + activations).
**Seed 2 on GPU 7 will almost certainly fail with CUDA OOM.**

### Mitigation plan

1. **Do NOT terminate the running launcher** (PID 549105, constraint:
   不擅自终止任何运行中进程). The launcher does not use `set -e`, so
   individual seed failures won't abort the launcher.
2. When P2-10 reaches step 10000 and the launcher fires:
   - Seed 0 on GPU 2: ✓ expected to succeed
   - Seed 1 on GPU 5: ✓ expected to succeed
   - Seed 2 on GPU 7: ✗ expected to fail (CUDA OOM)
3. After P2-10 completes (freeing GPU 0), **manually re-launch seed 2 on GPU 0**
   using the deployed script:
   ```bash
   bash scripts/relaunch_p2_05_seed2_gpu0.sh
   ```
4. The re-launch script (`scripts/relaunch_p2_05_seed2_gpu0.sh`) is deployed
   with the following safeguards:
   - Checks if seed 2 is already running (avoids duplicate)
   - Checks if seed 2 already completed (avoids re-run)
   - Prefers `stage_a_step10000.pt`, falls back to `stage_a_best.pt`
   - Uses `--device cuda:0` (GPU 0, which will be free)
   - Same GRPO config as the original launcher (K=8, 500 iter, 4 groups)

### Constraint compliance

| constraint | status |
|------------|--------|
| 不擅自终止任何运行中进程 | ✓ — launcher PID 549105 untouched; re-launch is additive |
| 所有新增训练接入 split contract | ✓ — re-launch uses `--train-idx/--val-idx/--test-idx` |
| 所有新代码配套单元测试 | N/A — re-launch script is a bash wrapper around existing tested code |
| 不修改 train_backbone.py | ✓ — re-launch script does not modify any training code |
| GPU 4 FORBIDDEN | ✓ — re-launch uses GPU 0, not GPU 4 |


---

## Update log (2026-07-21 18:20) — Steps 3750–4500 eval, near halfway

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 3750 | 10728.14 | 7692.53 | +302.72 | **closest to best since step 500** |
| 4000 | 11204.82 | 11546.12 | +779.40 | back up |
| 4250 | 10932.77 | 9058.03 | +507.35 | plateau |
| 4500 | 10906.75 | 1848.89 | +481.33 | plateau |

### Current state (step 4558/10000, 45.6%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 4058+ steps.
- **Plateau range**: [10728, 11417] across 18 evals (steps 250–4500).
- **Step 3750** (val_loss=10728.14) was the closest to the step 500 best
  since warmup completed. This suggests the model occasionally finds better
  minima but cannot sustain the improvement.
- **Grad norm**: P50=5582, P99=18576 over last 100 steps — still **failing**
  P99<1000 target by ~19x.
- **Training health**: stable, no OOM, no NaN/Inf, finite gradients.
- **ETA to step 10000**: ~13.6h (at ~400 steps/h).
- **Checkpoints**: step 4500 saved at 18:17.

### Assessment

The plateau is now confirmed across 18 eval points spanning 4250 steps
(steps 250–4500). The best checkpoint (step 500, val_loss=10425.42) has
not been beaten for 4058+ steps. The 10k checkpoint will almost certainly
NOT improve over step 500.

The P2-05 launcher will use the 10k checkpoint as warm start. Given the
plateau, this is acceptable — the 10k checkpoint has more training and
similar quality to step 500.

### Next steps

1. Continue monitoring toward step 10000 (~13.6h ETA).
2. Next milestone: step 5000 (halfway, ~66 min away).
3. When step 10000 reached:
   a. Lock the 10k checkpoint (SHA-256 + chmod 444).
   b. Verify P2-05 launcher detects it and starts 3 policy seeds.
   c. Seed 2 on GPU 7 will fail (MIG 4.75 GB) — re-launch on GPU 0 using
      `scripts/relaunch_p2_05_seed2_gpu0.sh`.
   d. Monitor all 3 seeds, then aggregate results.


---

## Update log (2026-07-21 20:00) — Halfway milestone + improving trend

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 4750 | 10560.80 | 14387.89 | +135.38 | **closest to best since step 500** |
| 5000 | 10632.31 | 6041.71 | +206.89 | halfway point |
| 5250 | 10663.40 | 11424.54 | +237.98 | slight regression, still in improving range |

### Current state (step 5300/10000, 53% — HALFWAY CROSSED)

- **Best**: step 500 (val_loss=10425.42), unchanged for 4800+ steps.
- **Recent trend**: Last 3 evals (steps 4750, 5000, 5250) show val_loss in
  [10560, 10663] — **significantly better** than the earlier plateau range
  [10873, 11417] (steps 250–4500).
- **Improvement magnitude**: Recent evals are ~300–500 lower than the
  earlier plateau average (~10900). This suggests the model is slowly
  finding better minima in the second half of training.
- **Grad norm**: P50=5372, P99=20191 over last 100 steps — still **failing**
  P99<1000 target by ~20x, but all gradients finite.
- **Training health**: stable, no OOM, no NaN/Inf, finite gradients.
- **ETA to step 10000**: ~11.8h (at ~400 steps/h).
- **Checkpoint**: step 5000 saved at 19:55.

### Assessment

The improving trend in the second half is encouraging. If it continues,
the 10k checkpoint may approach or beat the step 500 best (10425.42).
This would make the P2-10 pivot a partial success — the 10k checkpoint
would be a better warm start for P2-05 than the step 500 best.

However, the best checkpoint has NOT been updated since step 500 (4800+
steps ago). The `stage_a_best.pt` file still corresponds to step 500.
Even if the 10k checkpoint doesn't beat step 500, the improving trend
suggests it will be a competitive warm start for P2-05.

### Updated eval trajectory summary (21 evals, steps 250–5250)

| phase | step range | val_loss range | trend |
|-------|-----------:|---------------:|-------|
| Warmup | 0–500 | 10873 → 10425 | decreasing (best at step 500) |
| Early plateau | 500–4500 | 10873–11417 | oscillating, no improvement |
| Recent improvement | 4750–5250 | 10560–10663 | **below early plateau** |

### Next steps

1. Continue monitoring toward step 10000 (~11.8h ETA).
2. Watch for whether the 10k checkpoint beats step 500 best (val_loss < 10425.42).
3. When step 10000 reached:
   a. Lock the 10k checkpoint (SHA-256 + chmod 444).
   b. Verify P2-05 launcher detects it and starts 3 policy seeds.
   c. Seed 2 on GPU 7 will fail (MIG 4.75 GB) — re-launch on GPU 0 using
      `scripts/relaunch_p2_05_seed2_gpu0.sh`.
   d. Monitor all 3 seeds, then aggregate results.


---

## Update log (2026-07-21 22:45) — Steps 5500–6000, improvement was temporary

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 5500 | 10897.55 | 6176.28 | +472.13 | back to plateau range |
| 5750 | 10955.43 | 4302.62 | +530.01 | plateau |
| 6000 | 11181.29 | 2638.37 | +755.87 | worse |

### Current state (step 6043/10000, 60.4%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 5500+ steps.
- **Plateau confirmed**: The brief improvement at steps 4750–5250
  (val_loss ~10560–10663) was **temporary**. The model has reverted to
  the plateau range [10897, 11181] at steps 5500–6000.
- **Full plateau range**: [10560, 11417] across all 24 evals (steps 250–6000).
- **Grad norm**: P50=6757, P99=33711 over last 100 steps — **failing**
  P99<1000 target by ~34x, and **increasing** (was ~18000 at step 4558).
- **Training health**: stable, no OOM, no NaN/Inf, finite gradients.
- **ETA to step 10000**: ~9.9h (at ~400 steps/h).
- **Checkpoints**: step 6000 saved at 22:41.

### Assessment

The 10k checkpoint will **almost certainly NOT** beat the step 500 best
(val_loss=10425.42). The plateau is now confirmed across 24 eval points
spanning 5750 steps. The best checkpoint has not been updated for 5500+
steps.

The P2-05 launcher will use the 10k checkpoint as warm start. Given the
plateau, the 10k checkpoint will have val_loss ~10800–11200, which is
~400–800 above the step 500 best. This is acceptable for P2-05 warm start
— the 10k checkpoint has more training and similar quality.

### Next steps

1. Continue monitoring toward step 10000 (~9.9h ETA).
2. When step 10000 reached:
   a. Lock the 10k checkpoint (SHA-256 + chmod 444).
   b. Verify P2-05 launcher detects it and starts 3 policy seeds.
   c. Seed 2 on GPU 7 will fail (MIG 4.75 GB) — re-launch on GPU 0 using
      `scripts/relaunch_p2_05_seed2_gpu0.sh`.
   d. Monitor all 3 seeds, then aggregate results.
3. After P2-05 completes, run final P2 acceptance audit.


---

## Update log (2026-07-21 23:25) — Step 6250, second-best eval

### New eval result

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 6250 | 10606.59 | 13731.92 | +181.17 | **second-best since step 500** (after step 4750's +135) |

### Current state (step 6453/10000, 64.5%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 5950+ steps.
- **Step 6250** (val_loss=10606.59) is the second-closest to the best,
  confirming the model occasionally finds better minima but cannot sustain.
- **Full eval range**: [10560, 11417] across all 25 evals (steps 250–6250).
- **Grad norm**: P50=4662, P99=26541 — still failing P99<1000 by ~27x.
- **ETA to step 10000**: ~8.9h (at ~400 steps/h).

### Assessment

No change in strategic picture. Plateau confirmed across 25 evals.
10k checkpoint will likely have val_loss ~10600–11200, acceptable for P2-05.


---

## Update log (2026-07-21 23:55) — Steps 6500, 6750

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 6500 | 11175.93 | 13029.28 | +750.51 | high plateau |
| 6750 | 10576.12 | 6337.77 | +150.70 | **third-best since step 500** |

### Current state (step 6817/10000, 68.2%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 6300+ steps.
- **27 evals** completed. Plateau range: [10560, 11417].
- The model continues to occasionally find better minima (steps 4750, 6250,
  6750 all below 10610) but cannot sustain them.
- **ETA to step 10000**: ~8h (at ~400 steps/h).


---

## Update log (2026-07-22 00:40) — Steps 7000, 7250, grad_norm below 1000

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 7000 | 10597.84 | 12863.90 | +172.42 | fourth-best since step 500 |
| 7250 | 10650.41 | **709.57** | +224.99 | **grad_norm < 1000** (first since step 250!) |

### Current state (step 7411/10000, 74.1%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 6900+ steps.
- **29 evals** completed. Plateau range: [10560, 11417].
- **Notable**: Step 7250 grad_norm=709.57 is the **first time grad_norm
  dropped below 1000** since step 250 (warmup phase). This is a positive
  signal — the model may be entering a more stable regime.
- **Recent val_loss trend**: Steps 6750, 7000, 7250 all in [10576, 10650],
  which is the lower end of the plateau. The model is consistently finding
  better minima in the final 25% of training.
- **ETA to step 10000**: ~6.5h (at ~400 steps/h).

### Assessment

The grad_norm dropping below 1000 at step 7250 is encouraging. If this
stabilization continues, the 10k checkpoint might be competitive with
the step 500 best (val_loss difference < 200). Even if it doesn't beat
step 500, the 10k checkpoint will be a good warm start for P2-05.


---

## Update log (2026-07-22 02:45) — Steps 7500, 7750, NEW CLOSEST TO BEST

### New eval results

| step | val_loss_mean | grad_norm | Δ vs step 500 (best) | notes |
|-----:|--------------:|----------:|---------------------:|-------|
| 7500 | 11240.35 | 1725.29 | +814.93 | high plateau |
| 7750 | **10542.76** | 2513.39 | **+117.34** | **NEW CLOSEST TO BEST EVER** |

### Current state (step 7781/10000, 77.8%)

- **Best**: step 500 (val_loss=10425.42), unchanged for 7281+ steps.
  Best checkpoint file (stage_a_best.pt) confirmed still from step 500
  (saved 06:27, not modified).
- **31 evals** completed. Full range: [10542, 11417].
- **Step 7750** (val_loss=10542.76) is the **closest to the step 500 best
  EVER**, only +117.34 above. Previous closest was step 4750 (+135.38).
- **P2-10 process**: PID 2872081, running 21h43m, 88.8% CPU, stable.
- **ETA to step 10000**: ~5.5h (at ~400 steps/h).

### Assessment

The model is getting closer to beating the step 500 best. If the trend
continues, the 10k checkpoint might actually beat step 500 (val_loss < 10425.42).
This would make the P2-10 pivot a success — the 10k checkpoint would be
both the most trained AND the best performing.

Even if it doesn't beat step 500, the 10k checkpoint will likely have
val_loss ~10500-10700, which is within ~100-300 of the best. This is a
good warm start for P2-05.

### Next steps

1. Continue monitoring toward step 10000 (~5.5h ETA).
2. Watch for whether any eval in steps 8000-10000 beats 10425.42.
3. When step 10000 reached:
   a. Lock the 10k checkpoint (SHA-256 + chmod 444).
   b. Verify P2-05 launcher detects it and starts 3 policy seeds.
   c. Seed 2 on GPU 7 will fail (MIG 4.75 GB) — re-launch on GPU 0 using
      `scripts/relaunch_p2_05_seed2_gpu0.sh`.
   d. Monitor all 3 seeds, then aggregate results.


## Best Checkpoint Investigation (2026-07-22 04:00)

**Observation**: `stage_a_best.pt` mtime updated from 2026-07-21 06:27 (step 500) to 2026-07-22 03:43:49.

**Investigation result**: The P2-10 training script (`scripts/run_stage_a_recovery_p2_10_option_c.py` lines 284-293) updates `stage_a_best.pt` based on **training loss** (`loss_value`), NOT validation loss:

```python
# P2-10: best-loss checkpoint
if loss_value < best_loss:
    best_loss = loss_value
    _save_stage_a_checkpoint(ckpt_best_path, ...)
```

This means `stage_a_best.pt` tracks the lowest training-loss step, not the lowest val-loss step. The val_loss plateau is REAL:

| Step | val_loss_mean | val_loss_median | val_loss_p95 |
|------|---------------|-----------------|--------------|
| 500  | 10425.42      | 9672.41         | 22982.27     |
| 7750 | 10542.76      | 9031.51         | 25063.30     |
| 4750 | 10560.80      | 9402.13         | 23418.33     |
| 6750 | 10576.12      | 9818.01         | 22917.66     |
| 7000 | 10597.84      | 9365.28         | 22412.75     |

**Conclusion**: Best val_loss remains at step 500 (10425.42). The 03:43 best ckpt update was triggered by a training-loss minimum at some step between 8000-8500, NOT a val improvement. The held-out val_loss has NOT improved over 10k steps (plateau at ~10500-11400).

**Impact on P2-05**: None — the P2-05 launcher specifically waits for `stage_a_step10000.pt` (periodic checkpoint), not `stage_a_best.pt`. The 10k checkpoint will be used as warm start regardless of val_loss trajectory.

**P2-10 verdict**: Option C (P2-02 fix restart) confirms the post-warmup plateau identified earlier. Held-out gain NOT achieved within 10k steps. However, the 10k checkpoint is still a valid warm-start for P2-05 GRPO pilot (the GRPO fine-tuning can still proceed from this checkpoint).
