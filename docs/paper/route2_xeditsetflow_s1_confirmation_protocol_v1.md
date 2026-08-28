# Route 2 XEditSetFlow V4-S1 confirmation protocol v1

Status: `FROZEN_PROSPECTIVE_BEFORE_S1_SCREEN_TERMINAL_OR_CONFIRMATION_OUTCOME_READ`

This protocol was fixed while the independent V4-S1 mechanics screen was still
running. No active screen runtime, gate, training curve, checkpoint metric,
Development TEST outcome, or new Evaluation outcome was used to choose this
design. It is subordinate to the tracked Route 2 master protocol and does not
reinterpret the completed V4.0.3 SetFlow scientific NO_GO.

A later static implementation audit found that screen runner HEAD
`930fccf468c14378b3dd2fd2caf3aaa3cc2eb3c8` set the declared seed only after
constructing randomly initialized model layers. Its full and single-mode arms
ran in separate processes, so their nominal seed did not establish matched
parameter initialization. That family remains immutable execution evidence,
and any nominal gate it writes is not a successor authority for this protocol.
The confirmation design below remains prospectively frozen, but it can be
activated only by a new independent screen family that applies the same seed
before model construction. The retry must keep seed `20260911`, both arms,
weight `0.05`, all thresholds, and every other frozen screen setting unchanged;
it is a technical correction, not an extra seed, sweep, or reinterpretation of
the original evidence.

## Decision and question

The confirmation question is deliberately narrow: if the already frozen S1
screen reaches its exact scientific PASS, does a freshly trained
`v4_s1_full` model reproduce the unchanged SetFlow component-readiness
criteria across all three predeclared confirmation seeds?

The decision is to retain the existing downstream SetFlow confirmation
contract exactly:

- training seeds `20260912`, `20260913`, and `20260914`;
- one `v4_s1_full` parameter-update training per seed, for exactly three
  training jobs;
- checkpoint passes `4`, `6`, `8`, and `10` per seed, for exactly twelve
  outcome-free Development Validation jobs;
- no fourth seed, replacement seed, early stopping, adaptive coefficient,
  weight sweep, or post-outcome arm selection.

The screen's `v4_s1_single_mode` arm remains required screen provenance but is
not retrained during confirmation. The screen already tests the frozen
full-versus-single mechanics margin; confirmation retains the historical
full-only three-seed readiness design. Consequently, confirmation can support
cross-training-seed stability of the full S1 model, but cannot establish that
the 0.05 S1 term alone causally produced an effect.

## Entry barrier

No confirmation artifact may be materialized unless the independent S1 family
has one exact terminal runtime and one exact screen gate satisfying all of the
following:

- runtime status `XEDITSETFLOW_V4_S1_SCREEN_AND_GATE_TERMINAL`;
- two of two training jobs and eight of eight checkpoint Validation jobs have
  unique zero-exit SUMMARY terminals;
- no FAILURE, double terminal, missing terminal, partial conflict, or
  `first_terminal_failure`;
- gate schema `route_a_v3_route2_xeditsetflow_v4_s1_screen_gate.v1` and status
  `XEDITSETFLOW_V4_S1_SCREEN_PASS`;
- `s1_mechanics_screen_passed=true` and
  `successor_protocol_required=true`;
- `confirmation_authorized=false`, `confirmation_seeds=[]`, and
  `legacy_v4_confirmation_authorized=false`, proving this is a new successor
  rather than an illicit use of the legacy launcher;
- screen seed `20260911`, objective
  `XEDITSETFLOW_V4_S1_CROSS_STATE_CANDIDATE_MODE_RESPONSIBILITY`, fixed weight
  `0.05`, and a selected screen checkpoint in `{4,6,8,10}`;
- every successful screen training summary and checkpoint records
  `parameter_initialization_seed=20260911` and
  `parameter_initialization_seed_applied_before_model_construction=true`;
- Development TEST and new Evaluation outcome reads both equal zero.

The screen-selected pass and both screen checkpoint decisions are frozen into
confirmation provenance. They do not select or reuse any confirmation
checkpoint and the screen weights are not carried into the new three training
runs.

## Frozen model and training policy

Every confirmation job trains `v4_s1_full` from its own seed with the existing
S1 architecture and objective:

- eight latent modes, full mode-information weight `0.05`, and cross-state
  candidate-mode responsibility weight `0.05`;
- canonical TRAIN-only candidate identity, four state slots per source
  occurrence, detached EMPTY-state target, forward KL, and the frozen
  state-to-candidate-to-occurrence reduction;
- AdamW, learning rate `0.0002`, weight decay `0.0001`, gradient clip `1.0`,
  warmup fraction `0.05`, and cosine decay to ten percent of the initial rate;
- ten complete passes, physical and effective state batch `32`, source repeat
  cap `4`, no early stopping, and no Validation generation during training;
- only passes `4`, `6`, `8`, and `10` are saved;
- each confirmation seed is applied to CPU and CUDA RNG state before any model
  layer is constructed, and that ordering is recorded in the attempt,
  checkpoints, training summary, and Validation lineage;
- active S1 responsibility constraints must be positive in every successful
  confirmation training summary.

Training may consume TRAIN projections and the frozen source-token cache only.
It may not read active Validation performance, Critic scores, independent
evaluator scores, Development TEST, guidance output, or new Evaluation.

## Frozen Validation and checkpoint selection

Posttraining starts only after all three training jobs have unique zero-exit
SUMMARY terminals with exact S1 identity, CUDA/BF16 evidence, nonzero optimizer
updates, four exact checkpoint paths, no CPU fallback, and protected reads of
zero.

Each of the twelve checkpoint jobs performs the existing outcome-free
Development Validation on exactly 891 eligible sources with exactly 32
trajectories per source, for 28,512 candidates. Decoder streams, common
Validation states, measured neighborhood, candidate cap, mode allocation,
replay, and no-retry/no-duplicate-rejection policy remain frozen. Each job has
parameter-update count zero and uses neither Critic nor an independent
evaluator.

For each training seed independently, checkpoint eligibility requires:

- common Validation set-marginal NLL at most `2.06809`;
- source-macro recovery at least `0.35`;
- source-macro measured top-k recovery at least `0.20`;
- source-macro unique-candidate rate at least `0.90`;
- hard legality exactly `1.0`;
- edit-budget, candidate-budget, trajectory-replay, numerical, and frozen
  small-graph correctness failures all equal zero.

Eligible checkpoints are ordered by maximum recovery, maximum top-k recovery,
minimum common NLL, then earliest pass. The screen-selected pass never enters
this ordering. A separate NLL-only choice remains a read-only diagnostic and
cannot replace the generation-constrained selection.

## Three-seed scientific gate

After all twelve Validation summaries are unique technical successes, every
seed must independently satisfy all of the following relative to the frozen
terminal F2 Development reference:

- recovery margin at least `0.05`;
- top-k recovery margin at least `0.03`;
- unique-candidate margin at least `0.15`;
- source-paired recovery bootstrap over the exact 891-source key set, with
  10,000 replicates, fixed RNG seed `2026091102`, percentile two-sided 95%
  interval, and lower bound strictly greater than zero.

All three seeds must pass. Means, pooled sources, the best two seeds, or one
exceptionally strong seed cannot rescue a failed seed. A complete package that
misses any scientific criterion is a scientific
`XEDITSETFLOW_V4_CONFIRMATION_NO_GO`; it does not authorize an extra seed or a
threshold change. A complete three-seed PASS is
`XEDITSETFLOW_V4_G0_READY` and establishes SetFlow component readiness only.

The gate retains schema
`route_a_v3_route2_xeditsetflow_v4_confirmation_gate.v1` for downstream
readiness compatibility, while adding the exact external model identity
`v4_s1_full`, the S1 objective and weight, screen provenance, and each selected
checkpoint, training-summary, and Validation-summary path. Internal reuse of
the legacy `v4_full` architecture builder must never alter external S1 lineage.

## Technical execution contract

Training and GPU Validation require real CUDA on configured physical GPUs,
NVIDIA A100 identity, BF16, and no CPU fallback. Before any new family root,
config, authorization, schedule, or runtime is created, the one-shot launcher
must inventory configured GPUs 0–5 and run CUDA/A100/BF16 probes on every GPU
that will receive a job. Inventory execution, return-code, parse, missing-device,
probe, CUDA, BF16, OOM, or CPU-fallback failures fail closed and write a unique,
non-overwriting sibling technical-failure artifact.

Free, used, total, predicted, required, or reserved memory is diagnostic only.
It cannot filter, rank, sort, authorize, or reject a configured GPU. Training
jobs use three GPUs in configured order; twelve Validation jobs use GPUs 0–5
in fixed round-robin order, two jobs per GPU.

The generic schedulers use package-level first-failure semantics. Success
requires both process return code zero and one unique SUMMARY with no FAILURE.
Any nonzero exit, FAILURE, double terminal, missing terminal, process-launch
failure, or fixed-worktree identity failure records the unique
`first_terminal_failure`, lets already-running jobs finish naturally, marks
pending jobs `NOT_RUN_AFTER_TERMINAL_FAILURE`, and prevents subsequent
adjudication. Technical failure never creates or masquerades as a scientific
NO_GO gate. A retry, if scientifically unchanged and separately authorized,
must use a new non-overwriting family.

Every new code HEAD must be a clean pushed exact HEAD and must pass the complete
isolated focused cohort plus exactly 96 V3.3.2 tests, with both exact-HEAD runner
receipts accepted by the actual launcher consumers, before GPU work starts.

## Claim and downstream boundary

S1 screen PASS, S1 confirmation PASS, and `XEDITSETFLOW_V4_G0_READY` are not an
excellent Development result, do not open Development TEST or new Evaluation,
do not establish Critic readiness, and do not by themselves authorize guidance
or Final comparison. Guidance must later use an S1-aware checkpoint loader;
silently labeling an S1 checkpoint as legacy `v4_full` is prohibited.

The sole excellent Development criterion remains exact terminal Final
adjudication with frozen gate `XEDITFLOW_V4_PASS`. Even that remains
`submission_ready=false` until a lawful, new, outcome-unexposed Evaluation is
available.

## Considered and rejected expansions

An independent reviewer preferred retraining matched single-mode controls for
all three confirmation seeds. That would answer a broader mechanism-package
replication question, but it would expand the already frozen downstream
inventory from 3 training/12 Validation jobs to 6/24 and introduce additional
post-screen comparisons. It is not required for the present full-model
readiness decision and is rejected for this protocol. Its concern is retained
as a claim limitation.

An eight-mode S1-off arm would better isolate the incremental S1 loss, but it
was not part of the frozen S1 screen family. Adding it now would create a new
mechanism study and cannot affect this readiness gate. Weight sweeps and extra
screen seeds are likewise rejected because they would turn confirmation into
post-outcome method selection.
