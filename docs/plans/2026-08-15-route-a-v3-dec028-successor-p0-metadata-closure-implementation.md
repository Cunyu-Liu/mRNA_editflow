# DEC028 successor P0 metadata-only closure plan

**Status:** `STATIC_TEMPLATE_AND_OWNER_INPUT_HANDOFF_ONLY_NOT_A_PRODUCTION_P0_RECORD`

## Purpose

This plan closes the gap between the DEC028 11-axis P0 schema and a future
metadata-only closure package. It does not materialize the 6,547 members, read
rows, create a split assignment, run a model, probe CUDA, write a runtime
record, or make a scientific claim. The companion template is
`configs/route_a_v3_dec028_successor_p0_metadata_closure_template_v1.json`;
its validator is
`scripts/route_a_v3/dec028_successor_p0_metadata_closure_template.py`.

## Current template result

The template intentionally preserves the contractual starting state:

- `PASS`: P0.3 exposure role, P0.6 scratch-only route, P0.11 persistent locks;
- `UNKNOWN_NOT_ASSERTED`: P0.2 prior-use attestation;
- `FAIL_CLOSED`: P0.1, P0.4, P0.5, P0.7, P0.8, P0.9, and P0.10.

This is a `3 PASS / 7 FAIL_CLOSED / 1 UNKNOWN_NOT_ASSERTED` static template.
It is neither a new DEC026 result nor a production execution of DEC028 P0.

## Required owner/aggregate closure inputs

| Gate | Responsible role | Minimum aggregate binding | Prohibited at this stage |
| --- | --- | --- | --- |
| P0.1 | Data custodian and project owner | 6,547 membership-contract reference, source lineage, and one unexercised prospective materialization recipe | rows, member IDs, sequences, endpoint values |
| P0.2 | Project owner | Full prior analytic-use disclosure; scope must be `DISCLOSED_EXPOSED_DEVELOPMENT_ONLY`; predecessor `UNKNOWN_NOT_ASSERTED` remains historical | any assertion of untouched or confirmatory status without disclosure |
| P0.4 | Project owner or rights holder | Intended internal process/train/evaluate rights and output-boundary binding | inference from public availability or old provenance alone |
| P0.5 | Data custodian and scientific owner | Complete future row-contract schema binding: authority, membership, endpoint, group, SE, rights, exposure | actual materialization, member/row payload, split assignment |
| P0.7 | Method owner | Outcome-blind algorithm, grouping keys, salt, fold/subrole contract, and `split_assignment_count=0` | component graph from real membership, assignments, outcome-conditioned tuning |

P0.3, P0.6, and P0.11 must retain their existing inherited facts. P0.8, P0.9,
and P0.10 require separately bound single-run policy, critic gate bundle, and
source-relative critic implementation; their failure codes must not be softened
by A2/A6 G0 preparation evidence.

## Future production gate

Only after DEC028 activation and the required distinct reviews may an authorized
metadata-only P0 producer consume a complete aggregate package. Any missing,
unknown, partial, unexpected, or non-PASS gate returns
`STOP_BEFORE_DATA_CUDA_MODEL` before materialization, CUDA, model construction,
optimizer, checkpoint, parameter update, training, or runtime output.

Even an all-PASS package yields only
`ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED`. It does not start
materialization, launch G1, unlock A7, change the 1/1/0/6547 counts, or make a
scientific claim. Those later actions need their own authorities and conformance
gates under the contract.

## Verification and handoff

The template's focused test exercises group closure, status semantics, initial
failure state, synthetic all-PASS non-launch behavior, persistent locks, and
aggregate-only owner-input boundaries. A distinct reviewer should assess the
future production package and P0.9/P0.10 bindings; this plan is not that review.
