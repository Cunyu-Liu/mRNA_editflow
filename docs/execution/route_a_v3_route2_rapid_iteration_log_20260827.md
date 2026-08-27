# Route 2 rapid iteration log — 2026-08-27

This is the current execution record for fast model iteration. It supplements
the historical protocol documents; it does not overwrite old failures, gates,
or terminal artifacts.

## Required record for every run

- run ID and objective;
- code commit and parent checkpoint/training commit;
- data split and protected-outcome read count;
- model/config, seed, optimizer/update budget when training occurs;
- physical GPU, start/end, wall time, peak memory;
- terminal summary or failure path;
- primary metrics and matched comparison;
- conclusion and next action.

## Iteration 0 — SetFlow V4 validation-only recovery

- Objective: recover the missing performance evaluation for the already
  terminal `v4_full` and `v4_single_mode` pass 4/6/8/10 checkpoints.
- Training commit: `edad89392077a0cf56e84dfcf94335606dd2b05a`.
- Parameter updates: 0; both training runs and all eight checkpoints are reused.
- Data: frozen Development Validation only; Development TEST reads 0; new final
  Evaluation reads 0.
- Metrics: common Validation NLL, recovery, top-k recovery, unique rate,
  legality, replay/numerical failures, and full-vs-single margins.
- Historical artifact policy: retain the original eight HEAD-mismatch failures
  and original technical `XEDITSETFLOW_V4_SCREEN_NO_GO`; write all recovery
  outputs to a new directory.
- Recovery implementation commit:
  `37c5901000cf6bef1606f05af242512f1342ceb6`.
- A100 verification: imports resolved from the detached recovery worktree;
  SetFlow V4 focused 156/156, exact V3.3.2 96/96, and compile passed under
  Python 3.10.20.
- Launch status: `XEDITSETFLOW_V403_VALIDATION_RECOVERY_LAUNCHED`; scheduler PID
  `1357620`; GPU0–4 eligible; five checkpoint jobs started in parallel and the
  remaining three queued per GPU. Runtime manifest:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v4/v403_validation_recovery_37c5901000cf6bef1606f05af242512f1342ceb6/runtime.json`.
- Current status: the first five checkpoint validations remain active; no
  performance artifact has been read while active.
- Conclusion: no SetFlow model-performance conclusion until the eight
  Validation jobs and recovery gate reach terminal.

## Iteration 1 — Critic V4 full RNG-replay repair

- Objective: repair the technical failure that stopped `v4_full` before its
  first optimizer update, then run the smallest real full-model replay check
  before restarting the full arm.
- Historical failure: V4.0.2 `v4_full`, seed `20260907`, stopped after
  168.58 seconds with `XEditCriticTrainingV4Error: Critic V4 RNG replay changed
  a stochastic prediction`. Failure artifact:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v4/screen_seed_20260907_v402_recovery_runner_93703adec7a4c76b4466d3aaae8684620bee985a/v4_full/failure.json`.
- Root cause: prediction collection used `torch.no_grad()`, while replay used a
  grad-enabled forward. The full model conditionally enables its mRNABERT
  upper-six and twelve edit-block activation-checkpoint paths only when gradients
  are enabled, so collection and replay executed different paths despite
  identical model parameters, batches, and restored RNG. C0 lacks that
  grad-sensitive checkpoint path and therefore did not expose the regression.
- Corrective change: collect with `torch.enable_grad()`, detach the prediction,
  and immediately release the collection graph. Exact `torch.equal` replay,
  dropout, activation checkpointing, RNG capture/restore, architecture, loss,
  seed, batch geometry, and scientific config remain unchanged.
- Repair commit: `a3dd9a9bd3ca727b88c0a053c4dd06fa959cf001`.
- Minimal full-recovery launcher commit:
  `7d7b7b9a8edc7cbc29590a096d13d8b494ef2909`.
- Verification:
  - the new regression failed on the old path because all collection forwards
    were no-grad and passed after the repair;
  - A100 Critic training helper tests: 17/17;
  - A100 Critic focused cohort at the repair commit: 137/137;
  - exact V3.3.2 cohort: 96/96;
  - current-head launcher plus repaired replay regression: 3/3;
  - remote imports resolve from the exact detached V4.0.3 worktree.
- Protected data: technical checks use TRAIN geometry only and synthetic VJP;
  Development TEST reads 0; new final Evaluation reads 0.
- Current status: code and launch path are ready. The target-free 170.48M
  parameter CUDA replay smoke is waiting for an in-scope GPU0–5 to become free;
  GPU0–4 currently run SetFlow checkpoint validation and GPU5 runs the existing
  Critic `v4_source_only` arm. GPU6–7 are outside the frozen Critic scope and are
  not used.
- Next action: run the strict full-model replay smoke on the first freed GPU,
  then immediately launch only the repaired `v4_full` arm. Reuse the completed
  matched V4.0.2 C0 summary because this repair changes only the full model's
  grad-sensitive checkpoint path; keep the existing source-only/control queue
  as complementary diagnostics.

### Iteration 1 launch and throughput follow-up

- Final implementation commit:
  `f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea`.
- Additional change: validate the immutable 9.65 GB bottom-six cache once
  before batching instead of rescanning it for every batch; use one retained
  graph forward when the physical batch already equals the fixed effective
  batch of 32; keep the strict replay path for smaller physical batches.
- Exact-head A100 verification: 217 Critic-focused tests and 96 exact V3.3.2
  tests passed. The 170,481,957-parameter BF16 CUDA smoke confirmed identical
  retained/replay predictions, parameter gradients, gradient norm, CPU/CUDA RNG
  terminal state, exercised router-balance loss, materialized AdamW state, and
  `cpu_fallback_used=false`. Evidence:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/xeditcritic_v4/v403_rng_replay_smoke_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea.json`.
- Formal run: `v4_full`, seed `20260907`, 8 passes, 22,416 fixed optimizer
  updates, physical/effective batch 32, BF16 AdamW on physical GPU5.
- Start: `2026-08-27 13:06:36 +08:00`; worker PID `1521028`; trainer PID
  `1521031`; unique attempt ID
  `xeditcritic_v4_screen_seed20260907::v4_full::v403_rng_replay_fix_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea`.
- Runtime:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v4/v403_rng_replay_fix_runner_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea/runtime.json`.
- Output:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v4/screen_seed_20260907_v403_rng_replay_fix_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea/v4_full`.
- Resource coordination: the older V4.0.2 `v4_source_only` process was retained
  with `SIGSTOP`; it was not killed and its artifacts were not modified.
- Protected data: Development TEST reads 0; new final Evaluation reads 0.
- Last launch-window observation: Critic runtime was running on CUDA with no
  summary or failure artifact. No active training metric was read and no
  performance conclusion is available before terminal Validation evidence.

## Iteration 2 — Remove remaining-free-memory launch gates

- Objective: apply the operator instruction that any configured GPU with CUDA
  memory may be used; current remaining memory must not be a launch-authorization
  threshold.
- Integrated recovery branch:
  `route-a-v3-v403-no-vram-gate-20260827`; SetFlow/Critic merge commit
  `6ff81ad3`; policy-fix commit
  `6e5dc4a4b95432fae568d7602d9d07efafaa4578`.
- SetFlow change: use the frozen configured physical GPU scope, including GPU5,
  without filtering by `preflight peak + 2 GiB`.
- Critic change: both the full-model smoke and formal launcher retain free/total
  memory and `peak + 2 GiB` as diagnostics only; neither value can authorize or
  reject launch. Real CUDA allocation or OOM remains fail-closed evidence.
- CUDA/BF16 policy is unchanged: no CPU fallback, no `CUDA_VISIBLE_DEVICES`
  remapping, explicit physical device identity, and terminal failure evidence on
  CUDA errors.
- Verification: the two directly affected test files passed 6/6 under the A100
  Python 3.10 environment; `git diff --check` passed. No neural parameter update,
  protected outcome read, or new experiment was performed for this policy fix.
- Active-run policy: the already running SetFlow `37c59010` and Critic `f34ab7d8`
  worktrees and artifacts remain unchanged for exact provenance.
- Conclusion: future launch code records memory telemetry but no longer treats
  free-memory headroom as a gate. Existing active scientific results remain
  pending their frozen Development Validation terminal artifacts.
