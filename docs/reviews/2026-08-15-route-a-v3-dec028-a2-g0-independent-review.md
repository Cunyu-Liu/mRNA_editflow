# DEC028 A2 G0 current-HEAD review handoff

## Scope

This record is limited to the source-level A2 G0 candidate at
scripts/route_a_v3/a2_g0_synthetic_evaluator_candidate.py and its frozen
configuration at configs/route_a_v3_a2_g0_evaluator_candidate_v1.json. It is
not an A2 qualification, a real evaluator run, a power result, or a biological
claim.

## Reviewable findings

- The configuration states DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL,
  NON_AUTHORITATIVE, and INACTIVE_EVALUATOR_CANDIDATE.
- The only executable mode is validate-only. It creates a built-in synthetic
  fixture and emits a stdout record; it has no project-row or materialization
  interface.
- Component grouping consumes only synthetic source/candidate identifiers. The
  split API accepts no endpoint outcome argument and marks its synthetic plan
  outcome_values_consumed=false and final_assignment_materialized=false.
- Direction normalization is exactly candidate minus source. Missing,
  nonnumeric, and nonfinite source endpoint, candidate endpoint, or biological
  SE values fail before a delta is produced; no missing value is treated as
  zero.
- The candidate requires a finite nonnegative biological SE and an explicit
  biological replicate count of at least two for its synthetic metadata
  fixture.
- The source imports no learner, project data reader, CUDA probe, checkpoint
  tool, or runtime output writer. Focused tests inspect that boundary.
- The separate static-audit recorder writes only aggregate validation,
  manifest, acceptance, and log files to the caller-supplied artifact root.
  It refuses an existing run directory and leaves the acceptance verdict at
  PASS_STATIC_SYNTHETIC_G0_ONLY rather than A2 PASS.

## Required independent checks

A reviewer who did not author this candidate must run the focused A2 test
module and inspect the endpoint/direction and component-split logic. That
reviewer must decide whether the implementation agrees with the DEC028 SS1
scope before recording an independent verdict.

## Current verdict

PENDING_INDEPENDENT_REVIEW. The candidate is only a static/synthetic G0
preparation. It does not establish A2, A6, SS1 completion, any data
qualification, or any scientific result.
