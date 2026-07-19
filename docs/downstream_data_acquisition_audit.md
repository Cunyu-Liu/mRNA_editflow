# Downstream Data Acquisition Audit

- Claim policy: Downstream data acquisition audit is planning and provenance evidence. Do not claim real MPRA/TE or stability performance until source tables are downloaded with license/source URLs, official splits are preserved, dataset manifests are complete, held-out predictor reports are generated, and leakage documentation is attached.
- Source tables present: `0/2`; schema ready: `0/2`; complete manifests: `0/2`; predictor audits ready: `0/2`
- Ready for real TE/stability claim: `False`
- Incomplete datasets: `['mpra_te', 'stability_half_life']`

| Dataset | Status | Raw candidates | Manifest complete | Predictor ready | Missing gates |
|---|---|---:|---:|---:|---|
| mpra_te | needs_source_table_download | 0 | `False` | `False` | `['source_table', 'schema_valid_official_split', 'complete_dataset_manifest', 'heldout_predictor_report', 'leakage_documentation']` |
| stability_half_life | needs_source_table_download | 0 | `False` | `False` | `['source_table', 'schema_valid_official_split', 'complete_dataset_manifest', 'heldout_predictor_report', 'leakage_documentation']` |
