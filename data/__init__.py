"""Data pipeline: download, clean, dedup/split, features, datasets.

Public entry points (importable as ``mrna_editflow.data.<name>``):

* :func:`~mrna_editflow.data.download_mrna.synthesize_corpus` — offline corpus.
* :func:`~mrna_editflow.data.clean_mrna.clean_corpus` /
  :func:`~mrna_editflow.data.clean_mrna.clean_record`.
* :func:`~mrna_editflow.data.dedup_split.family_disjoint_split`.
* :func:`~mrna_editflow.data.precompute_features.precompute_corpus` /
  :func:`~mrna_editflow.data.precompute_features.load_features`.
* :func:`~mrna_editflow.data.public_pipeline.build_public_corpus` — GENCODE
  FASTA -> cleaned JSONL + provenance manifest.
* :func:`~mrna_editflow.data.download_mrna.load_records_jsonl` /
  :func:`~mrna_editflow.data.download_mrna.write_records_jsonl`.
* :func:`~mrna_editflow.data.leakage_audit.audit_leakage` — k-mer nearest
  neighbour leakage audit before frozen-foundation-model evaluation.
* :mod:`~mrna_editflow.data.reconstruction` — transport-verified, untruncated
  canonical public records and explicitly lineage-bound derived views.
* :class:`~mrna_editflow.data.mrna_dataset.MRNADataset`,
  :func:`~mrna_editflow.data.mrna_dataset.collate_fn`,
  :class:`~mrna_editflow.data.mrna_dataset.LengthBucketBatchSampler`.
"""
