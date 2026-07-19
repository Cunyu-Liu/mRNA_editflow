# External SOTA Real-Run Audit

- Claim policy: External real-run audit validates adapter-written measured outputs against the standardized input pack and metric schema. Passing this audit permits reporting external metric rows, but does not by itself prove that MEF beats external SOTA methods.
- Input pack: `benchmark/external_sota/input_pack_t5_head1024/summary.json`; ready: `True`; sha256: `b0ccc98c07dbb67a7479595de2b7cf860e54e7fb5c9a295ebc6e67bd21313408`
- Measured models: `4` / `6`; invalid: `1`; missing: `1`; protocol-fidelity sufficient: `1`
- Ready for external real metric table: `False`; ready for external SOTA claim: `False`

| Model | Task family | Status | Outputs | Success | Constraints | Protocol fidelity | Failure reasons |
|---|---|---|---:|---:|---|---|---|
| LinearDesign | cds_protein_conditioned | `measured` | 1024 / 1024 | 1.0000 | `True` | unspecified (sufficient=False) |  |
| EnsembleDesign | cds_protein_conditioned | `invalid` | 1016 / 1024 | 0.9922 | `True` | official_code_budgeted_beam100_iter3_runs1_vs_paper_default_beam200_iter30_runs20 (sufficient=False) | expected_output_transcript_coverage_mismatch |
| codonGPT | cds_protein_conditioned | `measured` | 1024 / 1024 | 1.0000 | `True` | official_hf_pretrained_checkpoint_synonymous_masked_sampling (sufficient=False) |  |
| Prot2RNA | cds_protein_conditioned | `missing` | 0 / 1024 | 0.0000 | `False` | None (sufficient=None) | summary_missing, outputs_jsonl_missing |
| UTailoR | utr5_only | `measured` | 315 / 1024 | 1.0000 | `True` | official_public_code_and_weights_strict_25_100_nt_shared_public_subset_not_paper_dataset (sufficient=False) |  |
| UTRGAN | utr5_only | `measured` | 1024 / 1024 | 1.0000 | `True` | official_code_paper_default_10000_steps (sufficient=True) |  |
