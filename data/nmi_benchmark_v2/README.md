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
  CDS joined to P0 GENCODE/RefSeq protein metadata), measured mCherry
  cargo-family holdout against eGFP development libraries, and 804
  source-matched tdTomato local-delta interventions from GSE246381. The
  local-delta family key is disjoint from development eGFP records.
- `test_context`, `test_assay`: source-matched context/assay shift records plus
  independent absolute-property records. These are measured axis interventions
  but are explicitly not nucleotide-edit local-delta ground truth.
- `test_ood`: declared GC/uAUG local-delta shift, 703 source-matched mouse
  GSE246381 reporter interventions tagged `species_tail`, and a 453-record
  `length_tail` subset, plus measured varying-length absolute-property
  records. GENCODE mouse 5′UTRs remain Level A observational assets only.

`task_kind=local_delta` and `data_layer=C_source_matched_intervention` are the
only records eligible for biological nucleotide local-delta metrics. The
`context_delta` and `assay_delta` records are retained for axis-shift audits,
not editing claims. Public absolute
libraries are registered as Layer B and retain their measured values for
absolute-property evaluation only. Proxy and unlabeled records are assets,
not biological ground truth. GSE246381 paired labels are deposited reporter
abundance derived from mean sample-normalized UMI CPM; they are not direct
protein-abundance measurements. Full legal DP/beam references remain blocked
until every legal action has a measured label.

`manifests/prospective.json` is an empty Layer D intake and remains unfrozen
until model development, candidate selection, and all development audits are
complete. An empty or proxy-only axis never authorizes a SOTA/OOD claim.
