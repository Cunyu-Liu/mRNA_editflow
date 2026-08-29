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

## Iteration 5 — Bind every future LOSO job to clean code and stop on first technical failure

- Objective: close the two remaining reachable refit/LOSO execution gaps before
  that successor stage starts, without touching either active V4.0.3 recovery
  process or changing any scientific contract term.
- Exact-code correction: every newly started Critic V4 training process now
  requires both the authorized Git HEAD and a clean worktree before it reads the
  launch authorization, initializes CUDA, or creates an output directory. This
  prevents a long 42-job LOSO queue from mixing uncommitted code under one
  recorded commit identity.
- First-failure correction: the six-GPU LOSO scheduler now uses one package-wide
  stop event. A unique failure artifact or a non-exact terminal state records
  the first affected job, return code, summary/failure paths, and log; no later
  queued job is launched. Jobs already in flight are allowed to reach their
  exact terminal artifacts, after which adjudication and readiness are skipped
  and the package publishes `XEDITCRITIC_V4_LOSO_TECHNICAL_FAILURE`.
- Preserved semantics: a unique summary remains authoritative even when its
  subprocess return code is nonzero; exact terminal job status remains backward
  compatible and `terminal_artifact_kind` continues to distinguish SUMMARY from
  FAILURE. The frozen three seeds, seven studies, paired `v4_full`/`c0_v4` arms,
  42-job count, fixed pass 8, GPU order, CUDA/BF16 requirement, and all gates are
  unchanged. Free-memory values remain diagnostics and never authorize launch.
- Verification: the two directly affected test modules pass 14/14 in the A100
  Python 3.10 environment, including clean/dirty worktree refusal, exact-summary
  compatibility, first-failure evidence, skipped queued work, and suppression
  of LOSO adjudication/readiness after technical failure. These are CPU-native
  orchestration tests, not model validation or scientific evidence.
- Active and protected scope: the current SetFlow `37c59010` and Critic
  `f34ab7d8` processes and artifact families are unchanged. No active training
  metric, Development TEST outcome, or new final Evaluation outcome was read;
  the two-hour heartbeat remains the only training-progress monitor.
- Conclusion: future refit/LOSO execution will either use one exact clean code
  identity throughout or terminate promptly with durable first-failure
  evidence. Current model quality remains pending the frozen terminal and
  scientific gates.

## Iteration 6 — Make every frozen V4 successor fail closed as one package

- Objective: remove the remaining deterministic and package-orchestration
  blockers from the confirmation, posttraining, guidance-screen, and final
  comparison successors before any of those long queues starts. No active
  V4.0.3 process, frozen scientific threshold, seed, budget, precision mode,
  protected-data boundary, or GPU assignment is changed.
- Final-evidence correction: a successful per-seed final evidence job is now
  recognized by its actual atomic `seed_manifest_row.json` terminal artifact.
  The previous scheduler contract expected a nonexistent `run_summary.json`
  from this directory-producing job and would therefore have converted every
  successful final seed into a manufactured technical failure.
- Package-wide first-failure correction: confirmation training,
  posttraining/Validation, guidance screening, and the 98-job matched-final
  comparison now share one stop signal and one durable first-terminal-failure
  record per package. Jobs already in flight may finish, but no pending job is
  started after a technical failure or non-exact terminal state. Downstream
  adjudication is skipped for a technically incomplete package. A scientifically
  valid terminal adjudication, including a frozen scientific NO_GO, remains a
  successful execution and is not reclassified as an infrastructure failure.
- Exact-code correction: every process launch in those future queues rechecks
  that the worktree is clean and still equals the schedule's frozen Git HEAD.
  A mismatch is recorded as that job's technical failure before process launch,
  so one schedule cannot silently combine code identities while repository work
  continues in parallel.
- GPU prelaunch evidence: failures while querying or parsing the final package's
  CUDA inventory now produce a sibling prelaunch-failure record with the command,
  captured output, expected and observed code identity, and explicit no-CPU-
  fallback status. Available memory remains diagnostic only: no free-memory,
  estimated-memory, or headroom gate was introduced. The frozen genetic timing
  baseline remains real-CUDA FP32 timing-only as required by its protocol; neural
  jobs remain CUDA/BF16 and fail closed on CPU fallback.
- Scientific meaning: the only excellent Development result remains a terminal
  `XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL` adjudication whose frozen gate is
  `XEDITFLOW_V4_PASS`, backed by all three matched seeds and their paired
  uncertainty, regret, recovery, evaluator, legality, failure, budget, and
  provenance checks. A smoke test, proxy, training metric, atomic TEST, Critic
  screen, G0, guidance screen, or single seed is not promoted to that conclusion.
  Even a Development PASS remains non-external while the current named cohorts
  are outcome-exposed or invalid; submission readiness requires a genuinely new
  outcome-unexposed Evaluation cohort.
- Verification: the directly affected scheduler/launcher cohort passed 22/22,
  and the complete V4 successor orchestration cohort passed 131/131 in six
  isolated Python processes. A deliberately attempted single-process aggregate
  imported already-cached modules from a legacy worktree after an existing test
  helper changed Python import state; it was rejected as non-authoritative, and
  the established isolated-process layout removed that cross-worktree test
  contamination. The exact V3.3.2 CPU-native regression cohort and both exact-
  HEAD repeats remain required after commit before new runner receipts are
  materialized. These tests certify orchestration only and are not GPU
  validation or scientific evidence.
- Active and protected scope: no active training metric, curve, stdout/stderr,
  Development TEST outcome, or final Evaluation outcome was opened while making
  these corrections. The existing two-hour heartbeat remains the only progress
  monitor for the two active V4.0.3 recoveries.
- Conclusion: the corrected successor design either advances on one exact code
  identity with complete terminal artifacts or stops at the first technical
  failure with evidence. Model quality itself is still pending the already-
  frozen GPU and scientific gates.

## Iteration 7 — Close the handover, GPU-evidence, and final-science identity gaps

- Objective: remove only the reachable blockers found by three independent
  static audits before any frozen successor or final comparison starts. The two
  active V4.0.3 attempts, their artifacts, all frozen thresholds, seeds,
  budgets, precision modes, and protected-data boundaries remain unchanged.
- Current execution entry: `CURRENT_EXECUTION_INDEX.md` is now the sole current
  repository entry point. The dated 2026-08-20 handover is explicitly marked as
  a read-only historical snapshot, so a new operator cannot mistake its old
  branch, worktree, launcher commands, or scientific route for the current
  V4.0.3 path.
- Critic successor correction: repaired-screen provenance remains fixed at the
  true f34 training HEAD, while the confirmation training-semantics comparison
  now starts from the independently audited a305 safety baseline. This preserves
  the trainer/core/config diff barrier without treating the later clean-worktree
  execution check as a model-semantic change. A real committed-Git regression
  supplements the mocked diff tests; the affected module passed 22/22.
- GPU prelaunch evidence: generic confirmation training now inventories GPUs
  before publishing its one-shot authorization package. Generic guidance and
  posttraining, plus recovered SetFlow confirmation and posttraining, record
  command, return code, output, parse/missing-device reason, code identity,
  protected-read state, and explicit CPU-fallback status for inventory and
  CUDA/BF16 probe failures. Existing evidence prevents same-family overwrite.
  The five affected launcher modules passed 56/56.
- Final timing evidence: the final schedule passes its declared
  `failures/strongest_timing.failed.json` path directly to the frozen genetic
  timing wrapper. A lower-level CUDA or execution failure is bridged there with
  the original producer evidence, including a true CPU-fallback observation,
  instead of being replaced by a generic scheduler failure. The wrapper and
  final-launcher interface passed 10/10 targeted tests.
- Final scientific identity: the seed assembler now recomputes source-macro
  normalized regret and top-1 recall from their existing eligible per-source
  rows and requires 1e-12 agreement with producer aggregates. Every generated
  method's closed, open, generation, and terminal evidence is bound to the one
  selected `kappa/temperature/beta_max`; the pre-frozen strongest baseline is
  deliberately combination-independent. The closed-score controls retain the
  already-validated combination fields.
- Frozen comparator identity: final configs must use the protocol-fixed
  strongest score table and its existing V4.0.3 producer summary. The consumer
  verifies the fixed producer seed, 891-source coverage, score path/provider,
  CUDA/no-CPU-fallback state, no reselection, no measured outcomes, and protected
  reads. Guided and strongest margins also require the same independent
  evaluator checkpoint, frozen Development Validation stage, score scale, and
  no CPU fallback. The five directly affected final-science modules passed
  34/34 without changing a scientific threshold.
- Memory policy: GPU free-memory and peak-memory values remain diagnostics only.
  No launcher filters, sorts, rejects, or authorizes a configured GPU by free or
  estimated memory; no new memory threshold was added.
- Verification and publication: the containing commit must be clean and pushed,
  then the complete successor focused cohort must run as isolated Python
  processes together with the 96 V3.3.2 regression tests. Exact-HEAD shared and
  SetFlow receipts are materialized only from those final committed results and
  accepted by their real consumers before the two-hour heartbeat is updated.
  Targeted tests above are engineering evidence only, never model or scientific
  evidence.
- Active and protected scope: no active runtime, metric, curve, stdout/stderr,
  Development TEST outcome, or new final Evaluation outcome was read while
  making these corrections. No training or GPU validation was launched. The
  single two-hour heartbeat remains the only progress monitor.
- Conclusion: these repairs make a future Development PASS auditable, but do not
  claim that PASS in advance. An excellent Development result still requires
  `XEDITFLOW_V4_FINAL_COMPARISON_TERMINAL` together with `XEDITFLOW_V4_PASS` on
  all frozen final evidence. External scientific confirmation remains locked
  until a genuinely new outcome-unexposed Evaluation cohort is available.

## Iteration 8 — Make the next exact-HEAD handoff receipt and active cursor durable

- Objective: close only the remaining reachable handoff and successor-monitor
  gaps found while the two-hour heartbeat continues to own all active training
  reads. No active V4.0.3 runtime, metric, curve, log, protected outcome, frozen
  threshold, seed, budget, precision mode, or current artifact family is changed.
- Receipt-coverage correction: the repository current index now requires eight
  isolated focused-test processes rather than the obsolete six-process layout.
  Both Critic and SetFlow receipt consumers require an internally consistent
  eight-group record, the complete set of current group-defining test modules,
  at least the c5db 203-test coverage floor, and exactly 96 V3.3.2 tests. A
  positive count alone can no longer authorize a future runner, while additional
  tests can still increase the exact-HEAD receipt count without weakening the
  frozen minimum coverage.
- Critic GPU prelaunch evidence: controls, atomic frozen TEST, three-refit, and
  LOSO now reserve deterministic `<runtime_root>.failed.json` sibling evidence
  before inventory or CUDA/BF16 probing. Inventory execution, return-code,
  parsing, configured-device absence, and controls child-probe failures stop
  before runtime creation or job launch and retain command/output/device/code
  identity. Existing failure or partial evidence closes the family; a retry
  requires a distinct family. No free- or estimated-memory gate or GPU sorting
  was introduced.
- Active-cursor correction: the two runtimes named when the heartbeat was first
  deployed are explicitly initial cursors, not a permanent active set. Every
  successful successor launch must promote its returned `runtime_manifest`,
  atomic `job_runtime`, or guidance `guidance_screen_runtime_path` into the next
  wake's active-cursor set until exact terminal state. Existing runtime, launch,
  decision, failure, or partial evidence forbids relaunching the same family.
- Handover locator: the repository current index now names the canonical main
  contract, V4 prospective protocol, and frozen guidance config. The original
  2026-08-26 handover package receives one current-status addendum and a README
  link; its older audit files remain immutable historical snapshots.
- Remaining handover evidence: independent frozen Validation reproduction and a
  narrow critical-run ledger export remain intentionally deferred until Final
  Development terminal, when they can use terminal read-only evidence without
  competing with or snapshotting the active package. Any model forward in that
  reproduction must use real CUDA and stop on CPU fallback.
- External Evaluation metadata: no registered study is presently both
  outcome-unexposed, convertible, independent, and rights-cleared. GSE232572 is
  exposed, E-MTAB-10902 is currently unconvertible, and GSE246381 remains sealed.
  GSE113849 is only a possible metadata-admission subject; historical exposure,
  provenance independence, and study-specific rights must close before any
  outcome read. This does not gate Development PASS and continues to keep
  `submission_ready=false`.
- Verification discipline: the directly affected receipt and Critic launcher
  tests are run before commit. After the containing commit, the complete focused
  cohort is rerun in eight isolated Python processes together with the 96-test
  V3.3.2 cohort; only those committed results may create the next two receipts.
  These are orchestration tests, not GPU validation or scientific evidence.
- Conclusion: the current Development result remains scientifically pending.
  The only excellent-result criterion is still terminal Final adjudication plus
  frozen `XEDITFLOW_V4_PASS`; none of the handoff or orchestration evidence in
  this iteration substitutes for that result.

## Iteration 9 — Reject and repair a stale exact-HEAD test receipt fixture

- Objective: preserve fail-closed verification after the Iteration 8 receipt
  contract was committed. This is an orchestration-test correction only; it does
  not inspect or change any model runtime, metric, threshold, seed, budget,
  precision mode, or scientific artifact.
- Detected failure: the first committed eight-process focused run at
  `7b21bf9f2449190e656c3c24a47f687531e81a2d` was not accepted. Group 1 reported
  101 passes and two failures because one guidance-launcher test module still
  built the former single-command, 22-test mock runner receipt. The tightened
  consumer correctly rejected that fixture before the tests reached their
  intended stale-V3.3.2 and launch assertions. The other seven focused processes
  reported 188 passes and zero failures, and the separate V3.3.2 cohort reported
  96 passes, but no receipt is created from this mixed run.
- Minimal repair: only the stale test fixture was updated to model the current
  eight command groups, all required module markers, the 203-test historical
  floor, and exactly 96 V3.3.2 tests. Its deliberate stale-HEAD, failed-focused,
  and 95-test mutations remain intact. No production launcher or scientific
  rule changed.
- Pre-final verification: the repaired test module reported 9/9 passes and its
  complete Group 1 reported 103/103 passes in separate CPU-native processes.
  These checks establish the local fix only. The final containing commit must
  still rerun all eight focused groups and the complete V3.3.2 cohort; only that
  exact clean HEAD may be represented by shared Critic and SetFlow receipts.
- Active and protected scope: no active runtime, training log, metric, curve,
  Development TEST outcome, or new final Evaluation outcome was read. No GPU
  training or validation was launched, and the two-hour heartbeat remains the
  sole progress monitor.
- Conclusion: this iteration fixes test-evidence fidelity, not model quality.
  Development remains pending until the frozen Final comparison is terminal and
  reports `XEDITFLOW_V4_PASS`.

## Iteration 10 — Prepare terminal-only handover evidence without touching active training

- Objective: remove the avoidable delay between Final Development terminal and
  the two remaining machine-executable handover checks. This iteration prepares
  code and synthetic tests only. It does not inspect any active runtime, metric,
  curve, log, checkpoint payload, Development TEST outcome, or new Evaluation
  outcome, and it does not launch training or GPU Validation.
- R3 scope decision: the minimum contract-complete independent reproduction is
  a fresh aggregation of the historical Base Flow V2 terminal candidate rows,
  Development source manifest, and measured-neighborhood rows through the
  shared evaluator. It does not copy the frozen evaluation JSON, retrain Base
  Flow, rerun the other six methods, invoke the independent scorer, load a
  checkpoint, or perform model forward. The runner binds the terminal config,
  training summary, training attempt, checkpoint locator, seed, and historical
  commit as provenance, then requires exact source/candidate geometry, exact
  legality and violation counts, 1e-6 continuous-metric agreement, and an
  undefined rather than zero closed measured NDCG.
- Terminal gate: the R3 runner opens historical source/candidate/measured rows
  only after the Final launch receipt, 98-job runtime, exact adjudication, clean
  current HEAD, and protected-outcome fields agree. Both scientific PASS and
  scientific NO_GO may produce handover evidence; RUNNING, technical failure,
  missing or contradictory terminal evidence stops before those rows open.
  Output is a new atomic `handover_validation` directory and never overwrites a
  completed or partial family.
- Narrow training ledger: the exporter records exactly 72 parameter-updating
  attempts consumed by Final lineage: Critic screen 8, confirmation 6, refit 3,
  LOSO 42; SetFlow screen 2 and confirmation 3; guidance value 6 and final value
  2. A strict explicit inventory identifies only family, frozen schedule, and
  job key; config, output, GPU, log, attempt, summary, and checkpoint identity
  are derived from allowlisted schedules and cross-checked. Atomic TEST,
  Validation, generation, evaluator inference, timing, and adjudication remain
  stage barriers or frozen dependencies, not fabricated training rows.
- Ledger boundaries: the exporter requires all 98 Final jobs to have unique
  successful terminal artifacts, preserves the one authorized atomic TEST event
  separately, requires no post-atomic TEST reopening and zero new Evaluation
  reads, and distinguishes PASS from NO_GO while keeping
  `submission_ready=false`. It does not scan `/mnt`, read private/log/checkpoint
  payloads, mutate the central CSV or per-run records, or add checksums.
- Receipt coverage: the two new direct test modules are added to focused process
  group 8 in both independent receipt consumers and every current positive mock
  receipt. A future exact-HEAD receipt must therefore prove these closeout tools
  were tested; the historical 203 floor, eight-process layout, and exact 96
  V3.3.2 requirement remain unchanged.
- Verification discipline: direct synthetic tests exercise terminal PASS/NO_GO,
  pre-read rejection, real shared metric reduction, tolerance/undefined rules,
  exact 72-row inventory, CUDA/CPU-fallback and protected-read inconsistencies,
  and non-overwrite publication. After the containing commit, all eight focused
  groups and the 96-test V3.3.2 cohort must run again before new receipts and the
  heartbeat may advance to that HEAD.
- Conclusion: this code shortens terminal handover closure but creates no model
  result. The real R3 directory and terminal ledger remain gated on Final exact
  terminal; the excellent-result criterion remains frozen Final PASS only.

## Iteration 11 — Freeze the independent SetFlow V4-S1 mechanics successor

- Terminal facts: the V4.0.3 SetFlow recovery is now exact terminal with eight
  unique Validation summaries, frozen gate `XEDITSETFLOW_V4_SCREEN_NO_GO`, and
  `confirmation_authorized=false`. Protected, Development TEST, and new
  Evaluation reads are zero. This is a successfully executed scientific NO-GO,
  not a technical failure; the old runtime, gate, training, and Validation paths
  remain immutable and no legacy confirmation/posttraining family exists.
- Critic closure: V4.0.3 `v4_full` independently reached a unique terminal
  summary at seed 20260907, pass 8, 22,416 optimizer updates, physical/effective
  batch 32, A100 GPU5, CUDA/BF16, no CPU fallback, f34 authorization, and zero
  protected/TEST/Evaluation reads. Six controls are paused because they cannot
  restore SetFlow dual-readiness. This single arm is not a Critic screen PASS and
  is not a final model result.
- Prospective method decision: a new independent SetFlow family is frozen under
  the existing master contract rather than retrying or reinterpreting the old
  NO-GO. It contains exactly `v4_s1_full` and `v4_s1_single_mode`, reuses screen
  seed 20260911, and adds no extra screen seed.
- S1 mechanism: duplicate-collapsed terminal edit sets retain their target-only
  canonical index after sorting by edit count and edit tuple. Each expanded
  source draw carries explicit slots 0/1/2/3 and a fresh occurrence identity;
  padding repeats never share an occurrence. The detached slot-0 `EMPTY`
  candidate-mode posterior is the target for forward KL to the same candidate's
  compatible nonroot, nonstructural state posteriors. Reduction is fixed as
  state mean, candidate mean, occurrence mean.
- Frozen delta and controls: the S1 term has prospective weight 0.05 with no
  sweep. The existing V4 common/coverage/count weights remain 1.0/0.50/0.20 and
  mode-information remains 0.05 for full and exactly 0.0 for single-mode.
  Architecture, sampler, source repeat cap, AdamW schedule, ten passes, batch
  32, checkpoints 4/6/8/10, candidate generation, and selection order are
  unchanged.
- Scientific gates: Validation remains the same 891×32 cohort. Absolute floors
  remain NLL≤2.06809, recovery≥0.35, top-k≥0.20, unique≥0.90, legality=1.0,
  and all failures zero. Terminal-F2 margins remain 0.05/0.03/0.15 and full over
  single-mode margins remain 0.03 recovery and 0.05 unique. No threshold is
  weakened.
- Execution boundary: the one-shot S1 launcher requires a clean pushed exact
  HEAD, complete isolated focused tests, exactly 96 V3.3.2 tests, and both
  exact-HEAD receipts. Training and GPU Validation are CUDA/BF16 only with no
  CPU fallback. GPU memory is diagnostic only and cannot gate, sort, authorize,
  or reject a configured GPU. Any existing or partial family forbids reuse.
- Runner audit closure: real tracked config and terminal-fact audits are now
  exercised directly by launcher tests. Parameter attempts include the exact
  runner HEAD so a distinct retry cannot overwrite the earlier ledger row; a
  success summary is published only after terminal ledger bookkeeping. Job
  success requires a zero process exit and one SUMMARY. Original `.partial`
  evidence is never overwritten, and training, Validation, adjudication, CUDA
  probe, or scheduler-start failures retain the unique first failure, command,
  device, and protected-read state. A completed Validation with a failed frozen
  legality/budget/replay/numerical check remains a scientific ineligible row and
  can yield `XEDITSETFLOW_V4_S1_SCREEN_NO_GO`; a technically incomplete package
  skips adjudication and cannot be relabeled as scientific NO-GO.
- Claim boundary: no S1 optimizer, GPU Validation, Development TEST, or new
  Evaluation read is claimed by this freeze. An S1 screen PASS is still only a
  screen result; it can open only a new S1-bound successor under all unchanged
  downstream gates. Excellent Development evidence still requires terminal
  Final adjudication with `XEDITFLOW_V4_PASS`; `submission_ready=false` remains.
- Evidence: `configs/route_a_v3_route2_xeditsetflow_v4_s1_mechanics_screen_v1.json`,
  `audits/route_a_v3_route2_xeditsetflow_v403_recovered_screen_terminal_nogo_v1.json`,
  `audits/route_a_v3_route2_xeditcritic_v403_full_terminal_v1.json`, and
  `audits/route_a_v3_route2_xeditsetflow_v4_s1_freeze_and_runner_v1.json`.

## Iteration 12 — Reject the first S1 exact-HEAD cohort and renew the Critic-neutral baseline

- Rejected cohort: the first committed S1 candidate HEAD
  `708e2843b4b4a6f36796db5c21b6e99469138f3b` did not receive a runner receipt.
  Focused group 1 reported 122 passes and one failure because the existing
  Critic confirmation guard correctly detected four changed paths under its
  broad `core` semantic scope. Group 7 reported 95 passes and one failure
  because a controls test captured old module constants as function defaults;
  after the real `v4_full` became terminal, its monkeypatched paths no longer
  stopped the test before process creation. Groups 2/3/4/5/6/8 reported
  14/61/26/14/8/58 passes, and the independent V3.3.2 cohort reported 96/96.
  Mixed results are not combined, no receipt is materialized, and no GPU family
  is launched from this commit.
- Critic-neutral re-audit: the four semantic-scope changes since the prior
  baseline are the SetFlow batch typed fields, SetFlow V4 state identity and
  collation, and the two new S1-only modules. No Critic config, trainer,
  preflight, objective, sampler, or confirmation derivation changed. HEAD
  `708e2843b4b4a6f36796db5c21b6e99469138f3b` is therefore recorded as the new
  Critic-neutral safety baseline by
  `audits/route_a_v3_route2_xeditcritic_v403_confirmation_training_semantics_reaudit_708e2843b4b4a6f36796db5c21b6e99469138f3b.json`;
  the real confirmation consumer validates this audit before comparing future
  semantic paths.
- Controls test repair: full and historical-C0 validators now resolve their
  default artifact paths at call time. Synthetic tests can therefore substitute
  isolated paths even after the real full becomes terminal; production paths
  and scientific behavior are unchanged.
- Next gate: the containing repair commit must rerun all eight isolated focused
  groups and the exact 96 V3.3.2 tests. Only a wholly passing new HEAD may create
  the two receipts, be pushed, or launch S1.

## Iteration 13 — Fail closed on unmatched S1 initialization and prepare an independent seeded retry

- Active execution fact: the S1 canonical launcher consumed exactly once from
  clean pushed HEAD `930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8` and created the
  independent screen family
  `s1_screen_seed_20260911_runner_930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8`.
  It contains exactly two training jobs and eight checkpoint Validation jobs,
  with the frozen objective, weight 0.05, seed 20260911, passes 4/6/8/10,
  batch 32, CUDA/A100/BF16-only policy, no CPU fallback, zero protected reads,
  and no free-memory gate. The two-hour heartbeat remains its only monitor.
- Late static defect: review of the exact trainer used by that family established
  that the model was constructed before `torch.manual_seed` and
  `torch.cuda.manual_seed_all` were applied. Because full and single-mode were
  separate processes, the nominal screen seed did not control parameter
  initialization or provide auditable matched initialization. This is a real
  scientific-execution defect independent of any observed metric. The frozen
  repository evidence is
  `audits/route_a_v3_route2_xeditsetflow_v4_s1_seed_initialization_repair_v1.json`.
- Immutable-family disposition: the 930 family is not killed, relaunched, or
  overwritten; already-started jobs may naturally reach terminal. Its runtime,
  summaries, failures, and gate remain immutable evidence. However, even a
  formally emitted screen PASS cannot authorize confirmation or any successor.
  The canonical confirmation launcher explicitly rejects the audit field
  `affected_family_can_authorize_successor=false`. A formal NO-GO is likewise
  not converted into a corrected scientific conclusion.
- Corrected screen protocol: after the old family reaches exact terminal, the
  repair may enter the permitted execution branch only as a new clean pushed
  HEAD. One independent retry family will reuse exactly seed 20260911, the same
  full and single-mode arms, objective, weight 0.05, ten passes, batch 32,
  checkpoint passes, 891×32 Validation and unchanged absolute/F2/single-mode
  margins. There is no additional screen seed, weight sweep, threshold reduction
  or artifact overwrite. CPU and CUDA seeds are now applied before any model
  construction and are recorded through training, checkpoint, Validation and
  gate lineage.
- S1-bound confirmation freeze: preparation commits
  `02e77f5f`, `95f38aec`, `a66f0c5e`, `82567f68`, `26234264`, `69e9af4f`,
  `16f595db4b83dd67197a8487f7d1f5302fc771b1` and
  `409e2d30` implement a prospective confirmation protocol, helper, exact
  three-seed training launcher, exact 12-job posttraining launcher and atomic
  adjudicator. Confirmation is intentionally locked until a future corrected
  screen HEAD/path is prospectively bound and reaches exact PASS. It trains only
  `v4_s1_full` at seeds 20260912/20260913/20260914, does not retrain
  single-mode, and does not preselect a confirmation checkpoint.
- Lineage and scheduler repair: launcher authorization is bound to the permitted
  execution branch rather than the preparation branch. Screen runtime jobs are
  bound to their exact schedule job keys, run IDs, GPUs, checkpoint passes and
  terminal paths. Posttraining accepts only the canonical training schedule,
  manifest, authorization and runtime paths. GPU 0–5 inventory and required
  CUDA/BF16 probes complete before any live screen/training family is consumed
  or new family is materialized. GPU memory remains diagnostic only. Existing
  final, failure, authorization, prelaunch or partial evidence closes a family;
  first technical failure stops future pending launches without relabeling a
  scientific result.
- Engineering verification: the seed/CUDA lineage modules passed 52 focused
  test nodes; the tracked seed-repair audit and corrected-screen launcher tests
  passed 9/9; the integrated S1 preparation suite passed 122/122; and after the
  final canonical-path/preflight changes the four directly affected modules
  passed 28/28. An independent static reviewer found no remaining launch or
  lineage blocker and separately ran 25/25 directly affected tests. These are
  CPU-native engineering checks, not model Validation or scientific evidence.
- Publication gate: the preparation branch is not an execution entry and this
  iteration creates no new `/mnt` family. After the old family terminal, the
  repair must be merged into `route-a-v3-v403-no-vram-gate-20260827`, committed
  and pushed, then pass all eight isolated focused groups and exactly 96 V3.3.2
  tests. Only the resulting exact HEAD may receive both runner receipts and
  launch the corrected screen retry.
- Protected scope and conclusion: this preparation read no active runtime,
  training log, metric, curve, Development TEST outcome or new Evaluation
  outcome and launched no GPU job. It prevents an unmatched screen from being
  promoted; it does not claim a model-quality result. Excellent Development
  evidence still requires terminal Final adjudication with frozen
  `XEDITFLOW_V4_PASS`, and `submission_ready=false` remains.

## Iteration 14 — Put Critic readiness on the multi-GPU critical path

- Scheduling decision: Critic readiness is now the first blocking path for
  value targets and Guidance. Base SetFlow mechanics remains independent and
  may run in parallel, but no value training or guided sampling may begin until
  the Critic has passed its own controls, confirmation, atomic frozen TEST,
  refit and LOSO gates and has been frozen. This changes execution priority,
  not the frozen scientific thresholds. It does not reinterpret the old
  SetFlow V4.0.3 NO-GO or the old single-arm Critic full summary.
- Multi-GPU plan: the six previously unstarted Critic controls are fixed to
  physical GPUs 0–5, one arm per GPU. A screen PASS opens three confirmation
  seeds, each with full and matched C0, again six jobs across six GPUs. The
  post-test chain uses three concurrent refit GPUs followed by a 42-job LOSO
  multi-GPU queue. The corrected SetFlow S1 retry uses its own frozen multi-GPU
  queues concurrently when the execution branch is available. GPU inventory is
  an identity check only; free or predicted memory never sorts, filters or
  gates a configured device.
- Historical/current separation: the already existing C0 producer remains
  exact head `93703adec7a4c76b4466d3aaae8684620bee985a` and repaired full remains
  exact head `f34ab7d865bb2477bfe24c1d0a7c9f5301a24cea`. They are the only allowed
  Critic v1 screen summaries. All six new controls use the current licensed
  clean HEAD, current trainer/worktree and v2 evidence. The controls launcher
  consumes the current shared runner receipt rather than inheriting f34 test
  booleans. Historical terminal payloads are not reopened at launch; the repo
  full-terminal audit is the barrier, and the eight summaries are consumed
  only once when the cross-root scientific gate is actually run.
- Code baseline: prep commit
  `eba5b17431cb8e19202e5ea788fd419338da2d66` is the immutable code/test
  baseline X for this iteration. It applies CPU/CUDA seed before every new
  Critic or SetFlow model construction, projects one canonical full SetFlow
  initialization into the single-mode arm, and carries exact initialization,
  device, BF16/no-CPU, HEAD, output, summary, checkpoint and attempt lineage.
  Value training now carries the same seed-before-model and checkpoint
  provenance through Guidance and Final evidence.
- Critic gate and package semantics: legacy compatibility is limited to the
  real C0/full pair. Controls, confirmation, refit and LOSO require v2. A job is
  successful only with zero return code and one unique SUMMARY. Before each
  pending process starts, the scheduler holds the shared lock and rechecks the
  schedule-fixed clean HEAD. The first spawn, CUDA, OOM, nonzero-return,
  FAILURE, double-terminal, missing-terminal or worktree-drift event stops new
  pending jobs, preserves one `first_terminal_failure`, lets jobs already in
  flight finish, and skips incomplete-package adjudication/readiness. Technical
  failures are no longer emitted as scientific NO-GO.
- Evidence and closeout repair: Critic screen peak VRAM is now a finite positive
  diagnostic only; the former 35-GiB ceiling is removed. The terminal ledger
  accepts v1 only for the exact historical C0/full identities, requires v2 for
  all new Critic attempts, and proves
  `summary.update_count == attempt.optimizer_steps ==
  config.data_geometry.total_optimizer_updates`. Its 72-attempt inventory
  remains explicit and does not scan directories or read logs/checkpoint
  payloads. Final value evidence remains tied to the exact frozen evaluator and
  checkpoint lineage rather than Critic self-evaluation.
- Engineering verification: direct stable results included controls 31/31,
  cross-root transition 10/10, confirmation launcher 46/46 with only the
  deliberately deferred exact-HEAD semantic-audit node deselected,
  refit/LOSO 10/10, screen mixed-root/technical classification 2/2, and ledger
  budget-drift checks 5/5. The integrated 18-module batch reported 239 passes
  and two stale ledger-fixture failures: the fixture still expected current
  controls to use f34 and rewrote the already-correct current HEAD as its
  supposed drift. Those two assertions were corrected; the exact affected
  three-node run then passed 3/3. `git diff --check`, the no-AppleDouble check,
  the no-directory-discovery ledger check and the no-VRAM-gate diff check all
  passed. These are CPU-native engineering checks, not model Validation.
- Semantic audit boundary: relative to the previous 708 baseline, the Critic
  successor path set has 20 exact changed paths. The v2 audit must state
  truthfully that initialization order, terminal/provenance schema, gate
  acceptance and scheduler failure semantics changed, while model architecture,
  forward/loss, data, sampler, batch, seed cohort, optimizer-update budget and
  scientific thresholds did not. It must keep f34 historical provenance,
  baseline X and the eventual audit-only runner HEAD Y as three distinct
  identities. Only an X-to-Y empty semantic-path diff may authorize launch.
- Publication and launch boundary: this prep commit is not an execution entry
  and no receipt or GPU family is created from it. After the v2 audit-only
  commit is tested and pushed, the old 930 SetFlow family must still reach exact
  terminal before the prep branch can enter the licensed execution branch.
  The merged clean pushed HEAD must then pass all eight isolated focused groups
  and exactly 96 V3.3.2 tests and materialize both exact-HEAD receipts. Only
  then may the six-card Critic controls and the corrected multi-GPU SetFlow S1
  retry launch in parallel as independent one-shot families.
- Protected scope and claim boundary: this iteration did not read the active
  930 runtime, any training log or checkpoint payload, Development TEST or new
  Evaluation outcome, and launched no GPU job. Critic readiness, S1 screen,
  atomic TEST, G0 and Guidance remain intermediate evidence. The excellent
  Development claim still requires the terminal 98-job Final adjudication with
  frozen `XEDITFLOW_V4_PASS` and an independent evaluator;
  `submission_ready=false` remains.

## Iteration 15 — Fail closed before multi-GPU Critic launch

- Candidate admission preflight: the clean pushed preparation candidate
  `b7f7c122299e71380aadef015498490e5e8dfeba` was tested without reading or
  writing `/mnt`. Eight isolated focused processes reported group results
  145/2, 16/0, 129/0, 27/0, 17/0, 10/0, 105/0 and 81/0, for 530 passes and
  two failures. The exact `tests/route_a_v3/*v332*.py` glob remained 96/96.
  Both focused failures came from one stale synthetic runner-receipt fixture:
  its third command group omitted the four already-required SetFlow S1
  confirmation modules. After repairing only that fixture coverage, the two
  failed nodes passed 2/2. This preflight was diagnostic evidence, not a formal
  `/mnt` receipt.
- Keep/discard decision: `b7f7c122...` was discarded as an executable
  candidate even though most tests passed. A targeted read-only review found
  reachable failure paths that could leave a one-shot family permanently
  ambiguous, compute a cross-root gate with drifted code, continue refit/LOSO
  after a missing process, or consume the unique frozen TEST with the wrong
  checkout. No threshold, seed, objective, data split or model result was used
  in this decision.
- One-shot launch closure: controls, Critic confirmation, confirmation
  posttraining, Atomic TEST, refit and LOSO launchers now write a family-local,
  non-overwriting `scheduler_launch.failed.json` if the scheduler/wrapper
  process cannot start after schedule/authorization/attempt artifacts have
  consumed the family. The failure records the exact command, HEAD, worktree,
  intended runtime, created artifact paths and original exception, with no PID,
  no false `LAUNCHED`, `gpu_job_started=false`, protected reads 0 and no
  automatic same-family retry.
- Atomic TEST pre-access closure: the Atomic job wrapper now verifies the
  job-fixed exact clean HEAD before starting the formal TEST runner. Identity
  drift or runner spawn failure writes `failure.json` plus a terminal technical
  runtime with `development_test_access_started=false`, access count 0 and
  outcome reads 0. A consumed failure family cannot be entered again.
- Long-package closure: refit and LOSO training-thread spawn exceptions now
  create exact job failure evidence, set the unique `first_terminal_failure`,
  stop pending work and skip adjudication/readiness. Refit adjudication, LOSO
  adjudication and LOSO readiness also re-check the schedule-fixed clean HEAD
  immediately before their process starts; identity or spawn failure is
  technical, never scientific NO-GO/READY, and leaves Guidance unauthorized.
- Cross-root one-read closure: before any eight-arm terminal payload is read,
  the transition now verifies the controls HEAD against the current clean
  checkout. Any identity, validation, evaluation or gate-write exception writes
  the gate sibling `.failed.json`; a later call rejects that family before any
  terminal payload reread. A successfully computed scientific PASS or NO-GO
  continues to use the normal immutable gate.
- New immutable code baseline: commit
  `f1a2328db57e1bd20fcc5cd5e6a23abcf4c62b66` contains the 21 implementation
  and direct-test changes. Direct verification was 155/155 for the six
  launchers plus Atomic wrapper, 29/29 for refit/LOSO/cross-root, and 9/9 for
  the Guidance runner-receipt consumer. `git diff --check` passed, AppleDouble
  files were absent, and an independent read-only review found no remaining
  launch-decision blocker in the bounded Critic-first path.
- Audit boundary: this code baseline changes controls/refit/LOSO/transition
  files inside the frozen Critic semantic pathspec. The previous
  `eba5b174...` audit cannot be reused mechanically. A new v2 audit must compare
  the same 708 baseline to `f1a2328d...`, classify the additional process-launch,
  barrier-identity and one-shot evidence semantics, and prove an empty frozen
  path diff from this code baseline to the eventual audit-only runner HEAD.
- Protected and scientific boundary: no active 930 runtime, training log,
  checkpoint payload, Development TEST or new Evaluation outcome was read; no
  GPU family or TEST runner was started. All results above are CPU-native
  engineering evidence. They do not establish Critic readiness, SetFlow
  readiness, Guidance success or an excellent Development result. The latter
  still requires the terminal three-seed 98-job Final gate
  `XEDITFLOW_V4_PASS` with the independent evaluator, and
  `submission_ready=false` remains.

## Iteration 16 — Make terminal handoff and formal admission executable

- Readiness finding: the clean pushed audit-only candidate `70652f557...` had
  correct training semantics but two reachable operational gaps. The corrected
  S1 launcher depended on heartbeat prose, rather than a canonical artifact, to
  prove that the invalid 930 family was terminal. The required eight isolated
  focused groups, exact 96 V3.3.2 tests and dual exact-HEAD receipts had strict
  consumers but no tracked producer; an automatic wake would otherwise have to
  reconstruct ad-hoc commands and could accidentally select the narrower
  controls marker subset. The candidate was therefore superseded before any
  GPU family or formal receipt was created.
- Old-family terminal transition: a new one-shot transition is now the sole
  old-runtime reader. It refuses an existing receipt or `.partial` before the
  runtime read, reads that runtime once, writes nothing while RUNNING, and only
  freezes one of the two exact terminal classes after the 2 training + 8
  Validation inventory, unique terminal artifacts, gate or first technical
  failure, scheduler exit and zero protected reads close. Its receipt preserves
  the nominal terminal only as invalid execution evidence and fixes successor
  and same-family-retry authorization to false. Process-inspection command
  failure is fail-closed, not interpreted as scheduler exit.
- Corrected S1 guard: the canonical retry launcher explicitly rejects the 930
  HEAD and consumes that immutable invalidation receipt after tracked repo
  facts but before current-HEAD runner receipts, GPU inventory/probes or family
  creation. It never rereads the old runtime. Scientific PASS/NO-GO and a
  technical terminal can both close the predecessor execution, but neither can
  authorize a scientific successor.
- Formal admission producer: the new
  `verify_and_materialize_route2_xedit_v403_successor_runner_receipts.py`
  asserts that the strict Critic and SetFlow eight-group contracts are equal,
  starts all eight isolated pytest processes before collecting them, records
  their actual positive PASS counts, then runs the literal V3.3.2 cohort and
  requires exact 96/0. Only then does it publish the distinct shared and
  SetFlow receipt schemas and run the production Critic-controls and S1
  pre-GPU consumer paths. A later external consumer-preflight failure preserves
  already strictly validated receipts so `--validate-receipts-only` can retry
  without repeating the test cohort.
- GPU scheduling conclusion: no code change was needed. After formal admission,
  Critic controls launch six concurrent arms on physical GPUs 0–5. Corrected
  SetFlow launches full/single training on GPUs 0/1, followed by six concurrent
  Validation queues over GPUs 0–5. The families use disjoint artifact roots and
  may share fixed physical GPUs; no scheduler sorts, filters or gates by free or
  predicted memory. Launcher calls should be serialized for unambiguous receipt
  consumption, while their background GPU schedulers overlap.
- New immutable code baseline: commit
  `7b4a445d3979aa7f832c52a16bd5bd67885e10b7` contains the terminal transition,
  launcher barrier, focused markers, formal admission producer and direct
  tests. Root verification was 24/24 for the two new tools, 163/163 for the
  integrated seven-module consumer set, and 72/72 for the committed-head Critic
  semantic consumer. `git diff --check` passed; two independent read-only
  reviewers reported PASS. The frozen Critic pathspec diff from `f1a2328d...`
  to this baseline is empty, so the X2 scientific/training semantic audit
  remains valid.
- Deferred formal evidence: the complete eight-process focused run, exact 96
  V3.3.2 run and dual `/mnt` receipts were deliberately not executed on the
  preparation branch. They are valid only after the old 930 terminal receipt,
  main-branch fast-forward/push and clean exact-HEAD identity checks. This
  iteration read no active runtime or terminal metric, touched no Development
  TEST/new Evaluation outcome, and launched no GPU. It is engineering evidence,
  not Critic/SetFlow readiness or an excellent Development result;
  `submission_ready=false` remains.

## Iteration 17 — Isolate the Critic controls CUDA-OOM retry

- Formal launch evidence: licensed clean pushed HEAD
  `ebf99ebf8a253ad27e311e555121d328df8fae10` passed eight isolated focused
  groups 158/17/149/27/22/10/110/92 (585/585 total) and the exact V3.3.2
  cohort 96/96. Both exact-HEAD receipts were materialized and accepted before
  the Critic and corrected SetFlow launchers created their independent
  families. The Critic scheduler started all six controls on physical GPUs
  0–5; SetFlow full/single started concurrently on GPUs 0/1. No free-memory
  query, ranking, filter or threshold was used.
- Technical failure: source-only on GPU0, edit-metadata-only on GPU1 and
  no-candidate-sequence on GPU2 each wrote a unique CUDA OOM failure. The
  first immutable failure is source-only with return code 1. At the first
  low-frequency failure window the other three controls were already in
  flight on GPUs 3/4/5 and therefore continued naturally; no pending job was
  started after the failure became observable. Cross-root adjudication and
  confirmation are forbidden for this technically incomplete package. The
  three OOM artifacts are technical evidence, not a scientific NO-GO, and
  Development TEST/new Evaluation reads remain 0.
- Causal scope: two failed controls shared GPU0/1 with the independent SetFlow
  jobs; all three failures were CUDA allocator-capacity failures, while the
  exact amount of fragmentation is not inferred from the terminal evidence.
  The retry does not convert these diagnostics into a memory gate. It leaves
  model architecture, forward/loss, data, sampler, seed 20260907, batch 32,
  passes, optimizer-update budget and scientific thresholds unchanged. The
  only allocator change is the documented PyTorch
  `expandable_segments:True` mode.
- Retry isolation: a new canonical transition will be the sole old-runtime
  reader once that scheduler reaches exact technical terminal. It validates
  the six-arm inventory, terminal uniqueness, scheduler exit, unchanged first
  failure, absent cross-root gate and protected reads 0, then writes one
  immutable receipt with same-family and scientific-successor authorization
  false and new-independent-retry eligibility true. It never reads the old
  failure or summary payloads.
- Fixed multi-GPU waves: retry1 uses a new exact HEAD and disjoint output,
  runtime, authorization, log, gate and attempt identities. All six current
  controls are retrained under that one HEAD. Wave 0 launches the original
  GPU0/1/2 arms concurrently; only three exact successful summaries permit
  wave 1 to launch the original GPU3/4/5 arms concurrently. A wave-0 first
  failure marks all wave-1 jobs `NOT_RUN_AFTER_TERMINAL_FAILURE`; already
  running jobs finish naturally. The mapping is literal and makes no live
  memory decision.
- Technical-semantics freeze: immutable baseline
  `793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110` is recorded by
  `audits/route_a_v3_route2_xeditcritic_v403_controls_oom_retry_training_semantics_793eedfb4b84e8c0dbd5a30bdf79c8923ddf8110.json`.
  The existing f1a v2 audit remains independently validated. Under its frozen
  pathspec, f1a→baseline changes exactly the controls launcher and scheduler,
  classified as immutable OOM/new-family/no-VRAM-gate lineage and fixed
  two-wave/allocator/package-first-failure execution; baseline→confirmation
  runner must be empty. Model implementation/architecture/forward/loss,
  scientific config, data, sampler, batch, seed cohort, update budget,
  thresholds and GPU0–5 mapping are unchanged. Protected reads are 0 and this
  audit claims no model result or scientific authorization.
- Execution boundary: implementation occurs only on prep branch
  `route-a-v3-v403-controls-oom-retry-prep-20260828`. It may be committed,
  tested and pushed there but cannot launch GPU work. Corrected SetFlow keeps
  the licensed main worktree frozen at Y3. After SetFlow and the old Critic
  package are exact terminal, main may fast-forward only, push the prep HEAD,
  run the complete eight-group focused cohort and exact 96 V3.3.2 tests, and
  materialize new dual receipts before one retry launch. None of the present
  engineering work is Critic readiness, Guidance evidence or an excellent
  Development result.

## Iteration 18 — Freeze the corrected SetFlow screen-to-confirmation bridge

- Production baseline: S3
  `19bc3ed4dd3ee5647e3d3304c10dc9914f885e68` follows Z2
  `26fdbcb38090cf98e68425bebabd084a374447c4`. Under the frozen production
  pathspec, Z2→S3 changes exactly four paths: the S1 confirmation protocol,
  confirmation core, gate core and trainer. The protocol diff is restricted to
  six `screen_provenance` fields, each replacing only the invalid 930 family
  identity with corrected-screen producer
  `ebf99ebf8a253ad27e311e555121d328df8fae10`.
- Immutable audit: the tracked artifact is
  `audits/route_a_v3_route2_xeditsetflow_s1_corrected_screen_confirmation_provenance_19bc3ed4dd3ee5647e3d3304c10dc9914f885e68.json`
  with status
  `XEDITSETFLOW_V4_S1_CORRECTED_SCREEN_CONFIRMATION_PROVENANCE_PASS`. It keeps
  the old 930 invalidation and seed-initialization repair authoritative and
  explicitly denies both successor and confirmation authorization.
- Dual-consumer barrier: the SetFlow confirmation launcher validates the audit,
  exact Z2→S3 four-path diff and an empty S3→current diff before any GPU probe,
  then records the baseline head/path/status in authorization, schedule and
  launch receipt. The Critic confirmation launcher independently preserves its
  f1a and Z1 audits, accepts exactly the two SetFlow core paths from Z1→S3 as
  provenance-only and Critic-objective-neutral, and requires no later drift in
  its frozen pathspec. The runner HEAD may advance; the corrected screen
  producer remains ebf and 930 remains forbidden.
- Scientific boundary: objective/weight, seed cohorts, passes, batch, checkpoint
  set, bootstrap, thresholds, GPU0–5 mapping, no-VRAM-gate, package-first-failure
  and protected-outcome policies are unchanged. This CPU-native provenance
  bridge reads no active runtime or protected outcome, authorizes no GPU family,
  and reports no model result; `submission_ready=false` remains.
## Iteration N — 2026-08-29: Critic Y3 controls OOM 冻结 + retry1 GPU 重映射

- Objective: 冻结 Y3 controls 的 CUDA OOM 技术失败，并为 retry1 提供可执行的 GPU 调度。
- Y3 controls（HEAD ebf99ebf）六臂固定 GPU0–5 并发：GPU3/4/5 三臂（candidate_bundle_permutation、
  no_cross、no_moe）TERMINAL_SUMMARY；GPU0/1/2 三臂（source_only、edit_metadata_only、
  no_candidate_sequence）CUDA OOM（error_type=OutOfMemoryError，backward 阶段无法分配 428–756 MiB）。
  失败根因：GPU0/1/2 被 tokenizer-benchmark 与 S1 训练长期占用，空闲显存不足；free_memory_gate_applied=false。
  该 package 定为技术失败，非科学 NO-GO。
- 唯一 transition 已执行：transition_record_route2_xeditcritic_v403_controls_oom_terminal.py
  → receipt audits/xeditcritic_v4/v403_control_recovery_runner_ebf99ebf8a253ad27e311e555121d328df8fae10_oom_terminal.json
  status=XEDITCRITIC_V403_CONTROL_RECOVERY_OOM_TERMINAL_RECORDED,
  new_independent_retry_eligible=true, same_family_retry_authorized=false, protected reads=0。
- retry1 GPU 映射调整（2026-08-29）：原冻结「GPU0–2 / GPU3–5」两波在 GPU0/1（仅约 5–8 GiB 空闲）必立即
  OOM；调整为两波均用 GPU2/3/5（9.6/30.3/22.1 GiB 空闲），wave 1 三臂全成功后再复用同卡跑 wave 2。
  PHYSICAL_GPU_INDICES = (2,3,5,2,3,5)；一臂一卡、expandable_segments、不筛卡、不改任何科学字段。
- Tests: launcher + transition 相关 focused 52/52 passed。
- Conclusion: Y3 旧 package 已永久冻结；retry1 将以全新 roots/attempt IDs 六臂全量重跑。
