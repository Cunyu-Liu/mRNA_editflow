# Benchmark v2

This directory is the frozen, source-relative benchmark registry used by the
Phase 1--5 experiments.  `records.jsonl` is the canonical store and each
manifest is an index into it.  The five final roles are deliberately
fail-closed: training/evaluation code must pass an explicit
`allow_final_labels=True` only after model and candidate selection are frozen.

Roles:

- `train`, `val`: source-disjoint development data;
- `test_id`: source-disjoint in-distribution final test;
- `test_family`: cargo/protein-family holdout when independent families exist;
- `test_context`, `test_assay`: context/assay holdouts only when the source
  registry contains a genuine independent axis;
- `test_ood`: GC/length tail or other declared distribution shift.

An empty final role is a valid audit result when the current registry cannot
support that scientific claim.  It is never filled with proxy data silently.
