# Benchmark v2

This directory is the frozen, source-relative benchmark registry used by the
Phase 1--5 experiments.  `records.jsonl` is the canonical store and each
manifest is an index into it.  The five final roles are deliberately
fail-closed: training/evaluation code must pass an explicit
`allow_final_labels=True` only after model and candidate selection are frozen.

Roles:

- `train`, `val`: source-disjoint development data;
- `test_id`: source-disjoint in-distribution final test;
- `test_family`: exact protein-family absolute-property holdout (CodonBERT
  CDS joined to P0 GENCODE/RefSeq protein metadata) plus measured mCherry
  cargo-family holdout against eGFP development libraries, and the untouched
  raw-library `family_cluster_id` local-delta holdout. The protein/cargo
  family results are absolute-property only; the local-delta subset is not
  cargo/protein-family disjoint.
- `test_context`, `test_assay`: source-matched context/assay shift records plus
  independent absolute-property records. These are measured axis interventions
  but are explicitly not nucleotide-edit local-delta ground truth.
- `test_ood`: declared GC or uAUG-motif local-delta shift plus measured
  varying-length absolute-property records. Mouse 5′UTRs are available as
  Level A observational assets, but species-shift source-matched local-delta
  labels are not available.

`task_kind=local_delta` and `data_layer=C_source_matched_intervention` are the
only records eligible for biological nucleotide local-delta metrics. The
`context_delta` and `assay_delta` records are retained for axis-shift audits,
not editing claims. Public absolute
libraries are registered as Layer B and retain their measured values for
absolute-property evaluation only. Proxy and unlabeled records are assets,
not biological ground truth.

`manifests/prospective.json` is an empty Layer D intake and remains unfrozen
until model development, candidate selection, and all development audits are
complete. An empty or proxy-only axis never authorizes a SOTA/OOD claim.
