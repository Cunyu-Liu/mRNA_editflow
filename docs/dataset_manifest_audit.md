# Dataset Manifest Audit

- Claim policy: Dataset manifest audit is reproducibility-governance evidence. Do not claim data scale-up or leakage-safe training until every required data version has source URL, SHA256, raw/clean record counts, drop statistics, split statistics, and record-file SHA verification where applicable.
- Complete manifests: `1/4`; all complete: `False`; ready for data-scale claim: `False`
- Incomplete datasets: `['refseq_human_rna', 'mpra_te', 'stability_half_life']`
- Pending split sidecars: `['refseq_human_rna']`

| Dataset | Manifest | Complete | Missing fields | Records SHA verified | Split sidecar |
|---|---|---:|---|---:|---|
| gencode_human_transcripts | `data/processed/gencode_human_transcripts.data_manifest.json` | `True` | `[]` | `True` | `benchmark/gencode_family_leakage_protocol/report.json` |
| refseq_human_rna | `None` | `False` | `['manifest', 'source_url', 'raw_file_sha256', 'records_sha256', 'record_counts', 'drop_stats', 'split_stats']` | `None` | `pending:benchmark/refseq_family_leakage_protocol/status.json` |
| mpra_te | `None` | `False` | `['manifest', 'source_url', 'raw_file_sha256', 'records_sha256', 'record_counts', 'drop_stats', 'split_stats']` | `None` | `None` |
| stability_half_life | `None` | `False` | `['manifest', 'source_url', 'raw_file_sha256', 'records_sha256', 'record_counts', 'drop_stats', 'split_stats']` | `None` | `None` |
