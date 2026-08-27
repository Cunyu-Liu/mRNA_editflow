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
- Runtime config:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runtime_configs/xeditsetflow_v4/v403_validation_recovery_37c5901000cf6bef1606f05af242512f1342ceb6.json`.
- Start evidence: scheduler and the first five validation processes were alive
  from `2026-08-27 12:23:49 +08:00`; the runtime/schedule files were materialized
  at approximately `12:21:38 +08:00`.
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
- Runtime screen config:
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/authorizations/xeditcritic_v4/v403_rng_replay_fix_f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea/screen_config.json`.
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

## Iteration 3 — Terminal-transition readiness while GPU jobs run

- Objective: remove the concrete blockers that would otherwise stop the two
  active V4.0.3 runs from entering their next contract stage after terminal,
  without reading active performance payloads, touching the processes, or
  opening Development TEST/new Evaluation.
- Low-frequency snapshot: one read-only check at `2026-08-27 13:31:53 +08:00`.
  SetFlow runtime remained
  `XEDITSETFLOW_V403_VALIDATION_RECOVERY_RUNNING`: five validation PIDs were
  present on physical GPU0–4 at approximately 2.88 GiB each and three jobs were
  still pending. Critic runtime remained
  `XEDITCRITIC_V403_FULL_RECOVERY_RUNNING`: trainer PID `1521031` was present on
  physical GPU5 at approximately 14.91 GiB with `device=cuda:5` and BF16.
- Failure evidence in that snapshot: no OOM, CUDA unavailable, CPU fallback,
  traceback, runtime failure, or terminal failure artifact was present. This is
  an in-progress device/status observation, not a technical or scientific PASS.
- Protected outcomes: Development TEST reads 0; new final Evaluation reads 0;
  active performance payloads were not read.
- Launcher/provenance repair commit: `45842293`. Fourteen currently reachable V4
  launchers now resolve their actual repository root, retain CUDA/BF16
  fail-closed behavior, record memory telemetry only, and never filter, sort, or
  reject a configured GPU by free-memory headroom. The existing f34 CUDA replay
  smoke is reused only after a cheap training-semantics provenance comparison;
  this avoids repeating the 170.48M-parameter smoke for launcher-only changes.
- Critic control-transition commit: `3b452a90`. A new fail-closed path waits for
  the current f34 `v4_full` exact terminal summary, then launches exactly the six
  affected non-C0 controls with the same f34 trainer/config on GPU0–5. It does
  not retrain `v4_full` or C0 and does not resume the old SIGSTOP
  `v4_source_only`, because all six controls use the repaired V4-FULL retained-
  graph path. A cross-root adjudicator combines the historical matched C0, the
  current f34 full result, and the six repaired controls only after all eight are
  unique successful terminals, then calls the unchanged frozen screen gate once.
- SetFlow confirmation-transition commit: `9dc31780`. A derived protocol,
  recovery-aware authorizer, and component-only launcher retain the original
  training HEAD `edad89392077a0cf56e84dfcf94335606dd2b05a`, validation HEAD
  `37c5901000cf6bef1606f05af242512f1342ceb6`, and actual future confirmation
  runner HEAD as three explicit provenance identities. The scientific
  thresholds, training policy, checkpoints, paired bootstrap, and seeds
  `20260912/20260913/20260914` are unchanged. Launch remains impossible before
  the recovered eight-job gate is terminal PASS.
- Focused verification: 74 launcher tests, 8 Critic transition tests, and 26
  SetFlow transition/prepare/train-interface tests passed. These CPU-native unit
  tests prove orchestration and fail-closed boundaries only; they are not model
  or scientific evidence. No additional GPU run was started for this code work.
- Milestone review found no P0 and four reachable P1 orchestration defects; all
  four were fixed before push. Commit `58c34341` requires the current Critic full
  arm to match the frozen seed, 8-pass/22,416-update budget, data counts, batch,
  checkpoint policy, and runtime/summary/authorization GPU identity before the
  six-control package may start. It also derives every cross-root arm HEAD from
  its real launch authorization instead of a caller-supplied claim. The two
  corrected Critic transition test files passed 25/25.
- Commit `a839a71d` makes both original recovery attempts true one-shot
  identities: changing the orchestration HEAD cannot relaunch the canonical f34
  Critic full arm or the canonical 37c SetFlow 2x4 validation cohort. A technical
  failure requires an explicit new retry family. The same commit stops the
  SetFlow confirmation authorizer from inheriting current-HEAD test booleans
  from the old screen HEAD; the new runner must provide an exact-HEAD, clean-
  worktree CPU verification receipt with focused and V3.3.2 tests passing. The
  affected authorizer/launcher cohorts passed 31/31, 8/8, and 16/16.
- The exact-runner verification receipt is materialized only after the final
  documentation commit and final CPU test pass, so its `runner_git_head` names
  the actual launchable HEAD. Its immutable audit path is
  `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/xeditsetflow_v4/confirmation_v403_recovered_runner_verification_{runner_git_head}.json`.
- Monitoring: a single two-hour heartbeat is the only progress monitor. It is
  pinned to the two exact active runtime paths, never opens protected outcomes,
  and will use the repaired no-memory-gate path only for new or still-unstarted
  work; it does not hot-replace the current processes.
- Conclusion: both active jobs remain scientifically pending. At terminal,
  SetFlow is adjudicated once by its recovery scheduler; Critic full is first
  checked as one technical arm, then the six repaired controls and the exact
  eight-arm screen gate are required. No smoke, training-set value, single arm,
  or three-seed Development result is a final project conclusion.

## Iteration 4 — Complete the V4.0.3 terminal-to-confirmation and guidance successors

- Objective: make every already-frozen V4.0.3 terminal PASS actionable without
  changing a scientific threshold, seed, budget, protected-data boundary, or
  the running SetFlow/Critic attempts.
- Critic successor: the recovered eight-arm cross-root screen PASS now has a
  one-shot launcher for the frozen three-seed confirmation package
  (`20260908/20260909/20260910`, `v4_full` plus matched `c0_v4`, physical
  GPU0-5). It requires a clean exact runner HEAD, the shared exact-HEAD CPU
  verification receipt, the persisted source authorizations, and real
  CUDA/BF16 probes. Free-memory telemetry is diagnostic only.
- Persisted-gate correction: review found that the upstream atomic JSON writer
  sorts object keys, while the first successor draft treated
  `arm_sources` object order as the frozen arm order. The consumer now checks
  the exact arm-key set and uses the explicit persisted `ordered_run_ids` for
  order. A real `sort_keys=True` write/read round trip is covered so an actual
  PASS gate cannot be rejected only because JSON object keys were reordered.
- SetFlow successor: three recovered confirmation training terminals now bind
  exactly 12 Validation jobs (`3 seeds x passes 4/6/8/10`) and the existing
  atomic G0 adjudicator. Training, orchestration, and validation runner HEADs
  remain distinct provenance identities; all checkpoints require successful
  CUDA/BF16 training summaries and the recovered protocol/authorization.
- Guidance successor: the recovery-aware bridge consumes the SetFlow G0 PASS,
  Critic post-test readiness/refit, the two frozen preflights/caches, and the
  shared exact-HEAD verification receipt. Before any 18-cell guidance candidate
  generation, it produces one shared pre-frozen strongest-baseline closed score
  table from the already selected genetic guiding checkpoint on the common
  Development Validation measured neighborhood. This table is outcome-free,
  CUDA/BF16-only, uses `method_id=strongest_matched_baseline` with
  `frozen_method_score`, and never participates in guidance-winner selection.
- Guidance-selection correction: an earlier draft incorrectly treated the 18
  screen `closed_candidate_probabilities.private.jsonl` files as candidate
  strongest-baseline score tables. Those files contain screen-combination
  terminal probabilities and are not schema-compatible with the pre-frozen
  baseline. The final successor now binds the single independently frozen
  baseline table; only `kappa`, `temperature`, and `beta_max` remain unresolved
  until the terminal frozen guidance gate selects one combination.
- Downstream compatibility: the existing V4 authorization, guidance screen,
  and final launchers accept explicit recovery-aware protocol/runtime/preflight
  paths while preserving their legacy defaults. No compatibility layer or
  alternate scientific route was introduced.
- Verification scope: syntax, focused orchestration, persisted-artifact,
  scheduler/adjudicator, and exact V3.3.2 CPU-native tests are required on the
  final committed HEAD before its shared runner receipt is materialized. These
  tests establish launch readiness only; they are not GPU validation or model
  evidence.
- Protected outcomes and active runs: Development TEST reads 0; new final
  Evaluation reads 0. No active metric, curve, stdout/stderr, or protected
  outcome was opened for this work, and no running process or artifact family
  was changed. The existing two-hour heartbeat remains the only progress
  monitor.
- Conclusion: the current SetFlow and Critic experiments remain scientifically
  pending. Their terminal gates, the frozen three-seed confirmations, the
  atomic TEST/post-test readiness stages, and the final matched three-seed
  comparison must still complete before any Development result can be called
  excellent; external confirmation still requires a new outcome-unexposed
  Evaluation cohort.
