# Route A V3 A2 G0 evaluator/split/power candidate — review handoff

Status: `DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL`

Authority status: `NON_AUTHORITATIVE`

Candidate ID: `ROUTE_A_V3_A2_G0_EVALUATOR_SPLIT_POWER_CANDIDATE_V1`

## Design-time state checked

Before implementing this candidate, the current local execution snapshot and
the A1/A2 gate-bridge proposal were read. The design-time scientific projection
remains:

- qualified ordinary studies: `1 / 3`;
- qualified A1 studies: `1 / 2`;
- qualified true-A2 studies: `0 / 1`;
- canonical records: `6,547`;
- final A2 membership and final split assignments: not frozen;
- training, GPU work, model/checkpoint selection, A7, and the next phase: not
  authorized by this candidate.

This implementation changes none of those values.

## Implemented G0 surface

The candidate freezes and exercises only synthetic interfaces:

1. Source-group and known-duplicate edges are combined into transitive
   connected components. Components are indivisible split atoms.
2. A deterministic five-fold, outcome-blind synthetic split planner balances
   whole components. It rejects outcome-bearing structural input and reports
   only aggregate fold counts, component-size histograms, and zero-leakage
   counts. It does not return record keys, component keys, or assignments.
3. The synthetic split salt is explicitly not the final A2 salt. Final A2
   membership, final salt, and final assignments remain `NOT_FROZEN`.
4. A dataset-specific endpoint/effect/SE manifest interface requires a declared
   endpoint scale and transform, direction normalization, candidate-minus-source
   effect semantics on the transformed scale, post-dedup source-group analysis,
   at least three independent biological replicates, a biological-replicate
   standard error, and pre-frozen missing/nonfinite/censoring rules. Technical
   replicates cannot count as biological replicates.
5. The evaluator schema has one primary metric,
   `WITHIN_STUDY_SPEARMAN`, at the post-dedup independent source-group unit with
   average ranks for ties and one source group per vote. It also reports a
   within-scale mean absolute effect error and aggregate mean observed standard
   error. Synthetic evaluation outputs contain no source-group keys, row-level
   effects, or row-level SEs and cannot select a model/checkpoint.
6. The Bonett–Wright/Fisher-z planning function uses the existing frozen
   assumptions: Spearman alternative `rho=0.25`, two-sided `alpha=0.05`, target
   power `0.80`, confidence level `0.95`, and maximum full CI width `0.30`.
   Under the frozen formulas, `N=155` passes power but not precision, while
   `N=156` is the first post-dedup independent-source-group N that passes both.
7. The command-line interface supports only `--validate-only`. It reads the
   fixed candidate config, performs no synthetic or project row processing, and
   writes nothing except one aggregate JSON object to standard output.

## Fail-closed boundary

The implementation rejects unknown duplicate references, duplicate structural
keys, extra/outcome fields in structural input, fewer connected components than
required nonempty folds, non-synthetic scope or identifiers, inconsistent
endpoint direction, technical-replicate substitution, fewer than three
biological replicates, unprefrozen censoring, missing/nonfinite numeric values,
nonpositive SEs, repeated evaluator source groups, and constant-rank inputs for
which Spearman is undefined.

`UNKNOWN_NOT_ASSERTED` and `NOT_RUN` are not PASS. A successful config or
synthetic-fixture validation is only
`PASS_DRAFT_INTERFACE_ONLY_NOT_ACTIVE_NOT_SCIENTIFIC_GATE` or
`PASS_SYNTHETIC_INTERFACE_ONLY_NOT_QUALIFICATION`; it is not A2 qualification,
power evidence, final membership, credit, canonical evidence, or a next-stage
unlock.

## Explicitly not implemented or authorized

- no real dataset, project row, member ID, sequence, endpoint, effect, or SE
  read;
- no real split or final assignment publication;
- no final A2 membership freeze;
- no formal qualification or power gate execution;
- no training, parameter update, CUDA/GPU access, checkpoint operation, or
  model selection;
- no qualification, credit, canonical, A6/L3/A7, claim, or phase transition.

The Python helper APIs are implementation candidates, not an active data
reader. Real-data use requires a separately reviewed successor with explicit
dataset-specific authority and a final prospective membership/salt freeze.

## Focused verification

- Candidate JSON parsed successfully.
- The module and focused test file compiled with a temporary bytecode cache.
- `26` focused tests passed.
- The tests exercised the first-pass `N=156` boundary, component transitivity,
  aggregate-only split output, zero source/duplicate/component leakage,
  endpoint/effect/SE closure, aggregate evaluator behavior, negative cases,
  validate-only zero-row/zero-artifact behavior, and absence of data/training/
  GPU/network/file-write imports or calls.

A repository-wide suite was not run because the candidate is an isolated
four-file G0 surface and the focused suite directly covers its supported
interfaces. No claim is made about unrelated project code.
