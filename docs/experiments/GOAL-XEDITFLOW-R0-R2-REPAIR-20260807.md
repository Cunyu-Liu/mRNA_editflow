---
run_id: GOAL-XEDITFLOW-R0-R2-REPAIR-20260807
status: REPAIR_IMPLEMENTED_NOT_SCIENTIFIC_GO
started_utc: 2026-08-07T11:22:39Z
base_commit: bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c
branch: xeditflow-r0-r2-repair-20260807t112239z
worktree: /home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_r0_r2_repair_20260807T112239Z
authority_binding: docs/contracts/xeditflow_v2_source_binding.yaml
decision_log: docs/execution/xeditflow_repair_decision_log.yaml
sealed_formal_evaluator_invoked: false
sealed_access_intent_written: false
sealed_reservation_written: false
training_started: false
git_push_authorized: false
---

# R0-R2 repair runbook

## Objective

Repair the deterministic governance, ordinary-data grouping/metric, and model
semantics/input/context defects identified in the 2026-08-07 read-only audit.
This run does not amend the original sign gate, overwrite historical results,
consume the formal sealed final, or make a scientific GO claim.

## Safety preflight

- Main repository HEAD at preflight: `2420dd85466d3fa1d82194a6a6bb0fe67dd396e8`.
- Migration base HEAD: `bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c`.
- The main repository retained its pre-existing untracked canonical snapshots
  and legacy `LinearDesign` state; none were modified.
- The migration worktree retained its pre-existing untracked `runlogs/`.
- GPU 0-7 had unrelated active compute jobs.  This run started no training and
  terminated no process.
- No repository `AGENTS.md` was present in the audited tree.

## Baseline verification

Before repair, the existing targeted engineering suite passed despite the
scientific/semantic defects:

```text
tests/migration/test_e0x.py
tests/migration/test_m4_data_readiness.py
tests/migration/test_m4_o0x.py
tests/migration/test_m4_sparse_model.py
tests/migration/test_m4_sparse_eval.py
tests/migration/test_g1x.py

94 passed in 4.55s
```

This establishes that the new tests must cover previously untested failure
modes; the old green suite is not evidence of scientific validity.

## Planned stages

### R0

- Bind the local V2 source proposal hash to the remote executable authority
  without claiming byte equivalence or changing the original gate.
- Add an append-only repair decision overlay rather than rewriting the frozen
  migration decision log.
- Hard-disable formal sealed execution before access intent/reservation while
  the evaluator is unimplemented.
- Add no-side-effect synthetic/ordinary preflight and lineage manifests.

### R1

- Rebuild candidate pools with endpoint-aware source identity and at least two
  distinct candidates.
- Reject duplicate-candidate/two-endpoint pseudo-pools and degenerate metrics.
- Report ordinary grouping atoms, study dominance, edit budget, and provenance
  gaps without reading sealed member data.

### R2

- Separate candidate-value, source-value, delta, sign, and grouping targets.
- Restrict ranking to within-source/context candidate sets and inverse
  consistency to delta-native prediction.
- Add padding masks, edit-visible encoding, trained unknown context handling,
  and fail-closed generation context behavior.

## Artifact locations

No large or ignored result is committed.  Final test logs and the generated
lineage manifest will be recorded under a new run-specific repair artifact root
and reported with absolute paths and SHA-256 after verification.

## Current status

`REPAIR_IMPLEMENTED_NOT_SCIENTIFIC_GO`: the deterministic R0-R2 repair is
implemented in the isolated worktree.  No model training, formal sealed
evaluation, access intent, reservation, push, or scientific GO occurred.

The repair has already established the following non-negotiable boundaries:

- The 5-prime `sign_accuracy >= 0.60` hard gate is retained.  R0-R2 does not
  amend the contract or convert the historical failure into a PASS.
- The historical signed top-10-percent mean ratio is retained only as a legacy
  diagnostic.  It is not a valid activation gate on signed outcomes; the
  primary enrichment field is therefore `HOLD` until a stable replacement is
  prospectively preregistered.
- Formal sealed activation requires exact preregistration bytes, a hash-bound
  complete ordinary H1-H3 GO manifest, matching R2 data/checkpoint/vocabulary
  schemas, a real aggregate-only evaluator fixture, and a fresh `UNSEALED`
  state.  The present production evaluator remains blocked.
- Pre-R2 checkpoints are incompatible with the edit-centred, explicit-mask,
  trained-UNK and prospective sign-decision contract.  They are rejected
  before model construction; fresh ordinary training is required.
- The R2 single-window encoder is substitution-only and fails closed for empty,
  unequal-length, out-of-range or wider-than-window edit scripts.  Multi-window
  support remains future work and cannot be claimed by this repair.
- G1 scientific guidance and X0 transfer are not rerun in R0-R2.  Encoder
  compatibility fixes do not validate historical generation-quality claims.

Accordingly, engineering completion will be reported as
`REPAIR_IMPLEMENTED_NOT_SCIENTIFIC_GO`; effect improvement, sealed transfer,
SOTA and publication readiness remain `NOT_RUN` pending fresh R2 training and
independent grouped evaluation.

## Implemented repair

### R0 governance and one-shot protection

- Bound the source proposal SHA-256 to the single active executable authority
  without claiming byte equivalence or relaxing the 0.60 sign gate.
- Required exact canonical preregistration bytes and loaded semantics, an
  embedded full-family ordinary H1-H3 GO aggregate, recomputed Holm/verdict,
  matching dataset/checkpoint/vocabulary/input schemas, and a fixture-bound
  aggregate-only evaluator before reservation can be considered.
- Enforced an exact H1/H2/H3 evaluator allowlist and every frozen effect,
  guidance, OOD, legality, length and budget gate.  Incomplete evidence is
  `HOLD_REPAIR`; complete failing evidence is `NO_GO`; only a complete all-pass
  family can be `GO`.
- Refused to resume pre-existing `ACCESS_INTENT` or `RESERVED` state, and made
  internal/sealed outputs exclusive-create under separate path policies.
- Hashed and parsed preregistration and ordinary activation files from the same
  byte snapshot to close the concurrent-replacement boundary.

### R1 ordinary grouping and metrics

- All ordinary runners now reject protected input, checkpoint and output paths
  before opening or creating them, including resolved symlink targets and any
  protected identity/role field in nested JSON rows.
- Candidate pools require benchmark, study, endpoint, context, source ID,
  source sequence, a consistent finite source anchor, and at least two distinct
  canonical candidates.  Ties and random baselines are deterministic and
  invariant to input row order.
- The unstable signed-mean top-10-percent ratio is preserved only as a legacy
  diagnostic.  The active enrichment field is `HOLD` until a stable signed-gain
  definition is prospectively preregistered.

### R2 model semantics and compatibility

- Separated candidate value, source anchor, delta, sign and safe rank groups;
  sign and rank targets no longer inherit candidate-value semantics.
- Added shared edit-centred source/candidate windows, explicit padding masks,
  strict edit-script validation, trainable unknown contexts, source-disjoint
  folds, grouped train/dev splitting and group-preserving rank batches.
- A prospective frozen sign-head rule now drives the reported model sign metric;
  its calibration and vocabulary/input contracts are checkpoint-bound.
- Every M4/O0/G1/E0 checkpoint consumer rejects pre-R2 checkpoints before model
  construction.  G1 critic scoring shares the R2 encoder, masks and UNK routing.

## Ordinary data audit

Input:

```text
/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z/artifacts/b0x/effect_dataset.jsonl
106,659 rows
SHA-256 f23a9fdd54a8ead90dccf793a073ccc6ed804a9b760a266c77b02e1fb1007ba1
```

Final aggregate audit:

```text
/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_r0_r2_repair_20260807T112239Z/artifacts/repairs/GOAL-XEDITFLOW-R0-R2-REPAIR-20260807/r1_ordinary_grouping_and_r2_input_audit_v3.json
SHA-256 52f3d93c9fce3b3fabc62c5e75136b968b31392ea41360dd1360569dda5e69c7
```

Findings:

- All 103,199 delta-defined rows lack both `context_id` and `assay_context`.
  The repaired safe pool builder therefore produces zero publication-qualified
  candidate pools.  The previous 1,630 endpoint-mixed legacy groups cannot be
  used as O0/G1 headline evidence.
- 5-prime data remain highly study-dominated: GSE114002 contributes 55,043 of
  60,237 rows (91.38%).
- The R2 single-window SUB schema represents 73,024 of 103,199 rows (70.76%).
  The 30,175 incompatible rows comprise 15,082 incomplete edit scripts, 13,284
  non-SUB scripts and 1,809 unequal-length pairs.
- Coverage is study-dependent.  GSE232571, GSE261709 and GSE298114 have 0%
  compatible rows, so a complete 3-prime S4 claim is currently impossible.
  For the 5-prime studies, compatibility is 75.87% for GSE114002 and 95.05%
  for GSE217518; any pilot must bind exact inclusion/exclusion manifests.

## Verification and sealed non-consumption

- Local final migration suite: `357 passed`; the sole local warning is an
  existing F0 tensor-to-scalar warning outside this repair.
- Remote final verification uses the existing `editflow` environment with
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` so the
  shared server is not oversubscribed.  The first unrestricted-thread attempt
  was gracefully stopped after `294 passed` when it spawned 191 CPU threads;
  its interrupted log was preserved and was not treated as final evidence.
- E0 `--dry-run` returned code 4 and `status: BLOCKED`.  Its safety report says
  no dataset, restricted store, access log, member data, output or reservation
  was read/written.  The blockers include the evaluator stub, stale formal
  command, absent ordinary full-family GO and absent fixture binding.
- The real sealed access log was checked by hash and line count only.  Before
  and after R0-R2 it remained byte-identical:

```text
SHA-256 073bf90a8cb7f68b7c6b8e518ee06bec3bb664103ef9b38fac12bcb3e6109216
1,187 lines
```

## Scientific disposition and next gate

R0-R2 removes deterministic false-pass and semantic bugs, but it does not yet
improve the observed model effect.  The next authorized scientific step should
be a new ordinary-only R3 training plan with two separate tracks:

1. a 5-prime R2 pilot on a hash-bound, coverage-audited SUB inclusion manifest,
   with study-macro S4 reporting, multi-seed confidence intervals, group-level
   permutation/bootstrap and the prospective sign-head rule frozen in advance;
2. a data-reconstruction/multi-window-indel track for the incompatible rows and
   missing context/provenance atoms, especially the three 3-prime studies with
   zero compatible rows.

O0/G1 headline optimization, X0 transfer and sealed E0-X remain blocked until
those gates are satisfied.  No legacy checkpoint or legacy enrichment value
may be used to bypass them.
