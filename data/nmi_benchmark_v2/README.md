# Benchmark v2

This directory is the frozen, source-relative benchmark registry used by the
Phase 1--5 experiments.  `records.jsonl` is the canonical store and each
manifest is an index into it.  The five final roles are deliberately
fail-closed: training/evaluation code must pass an explicit
`allow_final_labels=True` only after model and candidate selection are frozen.

Roles:

- `train`, `val`: source-disjoint development data;
- `test_id`: source-disjoint in-distribution final test;
- `test_family`: cargo/protein-family holdout from the frozen P3 source split;
- `test_context`, `test_assay`: independent absolute-property context/assay
  shift tasks. They are present for axis coverage but are explicitly not
  source-matched local-delta ground truth.
- `test_ood`: GC/length tail or other declared distribution shift.

`task_kind=local_delta` and `data_layer=C_source_matched_intervention` are the
only records eligible for biological local-delta metrics. Public absolute
libraries are registered as Layer B and retain their measured values for
absolute-property evaluation only. Proxy and unlabeled records are assets,
not biological ground truth.

`manifests/prospective.json` is an empty Layer D intake and remains unfrozen
until model development, candidate selection, and all development audits are
complete. An empty or proxy-only axis never authorizes a SOTA/OOD claim.
