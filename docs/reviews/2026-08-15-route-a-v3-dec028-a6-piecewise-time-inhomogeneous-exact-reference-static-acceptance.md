# DEC028 A6 piecewise time-inhomogeneous reference — static acceptance record

**Record type:** `DEC028_A6_G0_STATIC_SYNTHETIC_ACCEPTANCE`

**Decision:** `STATIC_VERIFIED_G0_REFERENCE_CANDIDATE_NOT_A6_PASS`

**Implementation source commit:** `a5526b7df47e4aa8c8562a88c7f6570dba5ae9bc`

## Evidence manifest

- Config: `configs/route_a_v3_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate_v1.json`
- Module: `scripts/route_a_v3/dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py`
- Focused test: `tests/route_a_v3/test_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py`
- Implementation plan: `docs/plans/2026-08-15-route-a-v3-dec028-a6-piecewise-time-inhomogeneous-exact-reference-implementation.md`
- Current static binding consumed by the candidate: `configs/route_a_v3.yaml` and `docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml`.

## Acceptance evidence

The committed source was tested in the isolated DEC028 authority-repair worktree with:

```text
PYTHONDONTWRITEBYTECODE=1 /home/cunyuliu/miniconda3/envs/editflow/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/route_a_v3/test_dec028_a6_piecewise_time_inhomogeneous_exact_g0_candidate.py
```

Result: `7 passed in 22.68s`.

The test constructs exactly 96 synthetic source-anchored edit DAG fixtures with
source lengths `{2,3,4,5}`, budgets `{1,3,5}`, and eight deterministic variants.
It verifies that the finite continuous algorithmic-time schedule changes the
relative edit/STOP probabilities rather than merely scaling holding time; raw
aliases aggregate by full successor state before normalization; source-relative
legality and no-reedit behavior reject violations; and authority/count/schedule/
runtime-lock drift fails closed.

The candidate reports a maximum recorded uniformization truncation bound of
approximately `1.68e-13`, a terminal-mass shortfall no larger than that bound,
and maximum independently refined RK4 terminal-TV of approximately `8.46e-14`.
The homogeneous tail DP is separately compared with complete root-path
enumeration for every fixture.

## Scope and retained non-results

This is synthetic, nonlearned, CPU-only, in-memory evidence for a finite
piecewise-constant continuous algorithmic-time schedule followed by a
homogeneous absorbing tail. It is not physical kinetics and does not establish
arbitrary/general time-varying-rate exactness. The candidate and its test do
not read project rows, sequences, membership, trajectories, private/sealed
payloads, checkpoints, or runtime artifacts; they do not probe CUDA or create
models, optimizers, parameter updates, GPU runs, or output files.

Accordingly, the following remain exact:

- `scientific_claim_status=NOT_ESTABLISHED`;
- contract-wide `general_time_inhomogeneous_exactness=NOT_ESTABLISHED_NOT_CLAIMED`;
- `a6_evidence_status=IN_PROGRESS`, `A6 PASS` not asserted, and `L3` not established;
- `A7` remains locked; no P0/materialization/SS4/SS5/G1/sealed transition is authorized;
- DEC028 remains a pending successor; the effective active decision chain ends at DEC027 with counts `1/1/0/6547`.

## Required next decision

This record is **not** the distinct A6 review required by the contract. A
reviewer independent of the implementation must assess the stated continuous-
time scope, uniformization error accounting, tail-DP/path-enumeration agreement,
and prohibition boundary before the candidate may be cited beyond
`G0_NONLEARNED_SYNTHETIC_PREPARATION`. No runtime authority, data access, CUDA
operation, or learned run follows from this record.
