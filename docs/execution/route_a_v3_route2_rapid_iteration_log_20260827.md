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
