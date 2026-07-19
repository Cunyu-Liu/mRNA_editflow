# Paper Table 3: External Baseline Readiness

- Claim policy: Table 3 is an external-baseline readiness/protocol table. Do not report external TE/F1/runtime performance or SOTA comparisons until a row is executable_ready and a real adapter writes measured outputs under benchmark/external_sota/.
- Task: `T5`; executable ready: `5` / `6`; real metric table ready: `False`
- Dataset: split=`public_head1024`, records=`1024`, sha256=`8e2a1b6de75d418bafe42241fe18d36093fa50fde732b56866a2e4e8fc1231ec`
- Hardware: label=`a100-server`, host=`bms-18937653-012`, machine=`x86_64`

## External Input Pack

- Present: `True`; ready for external real run: `True`; path: `benchmark/external_sota/input_pack_t5_head1024/summary.json`
- CDS/protein-conditioned rows: `1024`; 5'UTR-only rows: `1024`; skipped invalid CDS: `0`; sha256: `b0ccc98c07dbb67a7479595de2b7cf860e54e7fb5c9a295ebc6e67bd21313408`

## External Real-Run Audit

- Present: `True`; audit complete: `True`; path: `docs/external_sota_real_run_audit.json`
- Measured models: `4` / `6`; invalid: `1`; missing: `1`; real metric table ready: `False`; sha256: `9bcc75d455eb38877e7458785ebe45fad81d96f876101646ef60861b953236ab`

## Model Set Consistency

- Consistent: `True`; dry-run: `['EnsembleDesign', 'LinearDesign', 'Prot2RNA', 'UTRGAN', 'UTailoR', 'codonGPT']`; input pack: `['EnsembleDesign', 'LinearDesign', 'Prot2RNA', 'UTRGAN', 'UTailoR', 'codonGPT']`; real-run audit: `['EnsembleDesign', 'LinearDesign', 'Prot2RNA', 'UTRGAN', 'UTailoR', 'codonGPT']`
- Missing from dry-run: `[]`; missing from input pack: `[]`; missing from real-run audit: `[]`

| Model | Dry-run status | Real-run status | Candidate audit | Dataset/runtime audit | Protocol gap | Claim language |
|---|---|---|---|---|---|---|
| LinearDesign | `executable_ready` | measured; metric_ready=True; success=1.0; constraints=True | LINEARDESIGN_BIN=executable | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: LinearDesign is a deterministic CDS optimizer with strong biological validation; it is not trained as a stochastic full-transcript edit model and does not optimize UTR edits. | measured_external_metrics_available_not_sota_claim |
| EnsembleDesign | `executable_ready` | invalid; metric_ready=False; success=0.9921875; constraints=True | ENSEMBLEDESIGN_BIN=executable | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: EnsembleDesign targets structure/codon optimization in the protein-conditioned CDS lattice; MEF should compare against it on CDS structure objectives, not full UTR editing tasks. | adapter_executable_but_real_metrics_still_required |
| codonGPT | `executable_ready` | measured; metric_ready=True; success=1.0; constraints=True | CODONGPT_BIN=executable | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: codonGPT focuses on CDS-constrained synonymous generation; it does not jointly edit 5'UTR, CDS and 3'UTR under one region-aware variable-length process. | measured_external_metrics_available_not_sota_claim |
| Prot2RNA | `not_configured` | missing; metric_ready=False; success=None; constraints=False | PROT2RNA_BIN=env_unset; prot2rna=path_not_found; Prot2RNA=path_not_found | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: Prot2RNA is protein/CDS-conditioned and does not directly model UTR editing or full-transcript variable-length tasks; comparisons should isolate CDS design from full mRNA generation. | not_configured_no_external_metric_claim |
| UTailoR | `executable_ready` | measured; metric_ready=True; success=1.0; constraints=True | UTAILOR_BIN=executable | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: UTailoR is focused on 5'UTR optimization; it does not preserve or edit CDS/3'UTR with an explicit CTMC grammar. | measured_external_metrics_available_not_sota_claim |
| UTRGAN | `executable_ready` | measured; metric_ready=True; success=1.0; constraints=True | UTRGAN_BIN=executable | split=public_head1024; records=1024; dry_run_elapsed_s=0.007081896066665649; hardware=a100-server | Protocol difference: UTRGAN optimizes/generates UTRs rather than full 5'UTR-CDS-3'UTR transcripts; it has no codon-frame edit operator and is not evaluated on variable-length edit tasks. | measured_external_metrics_available_not_sota_claim |
