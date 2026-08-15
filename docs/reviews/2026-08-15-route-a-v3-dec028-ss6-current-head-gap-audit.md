# DEC028 SS6 current-HEAD nonlearned exactness gap audit

**Status:** `NOT_READY_FOR_FORWARD_PORT`

## Scope

This audit evaluates only the repository's existing A6 nonlearned CPU exact-DAG
and legal-CTMC partial implementations against current DEC028 authority. It is
not a production run, a learned run, or an A6/L3/scientific PASS.

## Current evidence

The frozen exact-DAG synthetic module completed:

```text
tests/route_a_v3/test_a6_cpu_exact_absorbing_dag.py
12 passed
```

Its evidence is limited by its own static contract to a time-homogeneous,
synthetic toy graph: exactly six cases covering source lengths 2 and 3 with
budgets 0, 1, and 2. It records general time-inhomogeneous exactness as
`NOT_RUN`, flow-base legal CTMC evidence as `NOT_RUN`, A6 as `IN_PROGRESS`,
and L3 as `NOT_ESTABLISHED`.

The legal-CTMC partial module completed five tests but failed its sixth:

```text
tests/route_a_v3/test_a6_cpu_legal_ctmc_partial.py
5 passed, 1 failed
```

The failing test was
`test_production_authority_exact_base_i_b_and_dependency_blobs`. Its explicit
failure was `AuthorityError: authority leaf drift: configs/route_a_v3.yaml`.
The old partial module binds its active config to
`c908ac57b7c9667398f616a0ccf7101b41451b80bf169e768131844d3b63a678`, while
the current DEC028 static authority has root-config SHA-256
`1f11e6a84ed394aecc5ef7a5626b7a07b2a877a4aa8c2a4c67a3d79e9771aca8`.

The tests used synthetic fixtures and temporary directories only. No production
publisher was invoked; no project data, CUDA/device context, model, optimizer,
checkpoint, runtime artifact, or parameter update occurred.

## Decision

Do not replace the old bound hash with the current DEC028 hash. That change
would falsely forward-port a historical production authority without a new
candidate, current-head review, or DEC028 binding.

The existing evidence supports only these retained facts:

- support-floor, hard-legality, STOP/terminal precedence, budget accounting,
  raw-alias aggregation, and small time-homogeneous exact-DAG fixtures were
  implemented and tested in their historical scope;
- no 96-graph exact-reference suite is established;
- no `{1,3,5}` budget suite is established by the six-case fixture;
- general time-inhomogeneous exactness is not established;
- no current-HEAD DEC028 production authority exists for this publisher.

## Required successor work

Any SS6 continuation must be a new, non-authoritative DEC028 G0 candidate with
its own current-head static binding and distinct review. It must separately
implement and test the promised source-anchored acyclic edit DAG, budgets
`{1,3,5}`, support floor, STOP/legality/alias semantics, independent exact
reference, a declared 96-graph fixture generator, and a truthfully bounded
time-inhomogeneous method. Until then it must remain synthetic/nonlearned CPU,
with no project rows, CUDA, model construction, checkpoint, parameter update,
A6 PASS, L3 claim, or A7 change.
