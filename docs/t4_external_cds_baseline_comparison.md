# T4 External CDS Baseline Comparison

- Claim policy: This table supports descriptive T4 CDS comparison only. Do not claim MEF superiority over LinearDesign, EnsembleDesign, or codonGPT until matched per-record paired tests, common objectives, and complete external outputs are available.
- Descriptive table ready: `True`; MEF superiority claim ready: `False`

| Method | Status | n | CAI | delta CAI | GC | GC3 | Codon accuracy | Codon KL | Pair KL | Protein exact-1 | MFE | EFE | sec/seq |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_source | `measured_internal_reference` | 1024 | 0.693702 | 0.000000 | 0.566811 | 0.667178 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | NA | NA | NA |
| MEF_protein_conditioned_CDS | `measured_internal_proxy` | 1024 | 1.000000 | 0.306298 | 0.688764 | 1.000000 | 0.510972 | 0.704436 | 1.404453 | 1.000000 | NA | NA | NA |
| MEF_codon_lattice_DP_budget3 | `measured_internal_proxy` | 1024 | 0.718100 | 0.024399 | 0.573655 | NA | NA | NA | NA | 1.000000 | NA | NA | NA |
| LinearDesign_official | `measured_external` | 1024 | 0.784660 | 0.090958 | 0.624985 | 0.824934 | NA | 1.416190 | 12.634971 | 1.000000 | -517.089258 | NA | 78.624449 |
| EnsembleDesign_official | `executable_ready_metrics_pending` | 1016 | 0.677068 | -0.016634 | 0.598681 | 0.770205 | NA | 1.826930 | 13.624174 | 1.000000 | NA | -488.009360 | 212.981474 |
| codonGPT_official_HF_pretrained | `measured_external_pretrained_checkpoint_10seed` | 1024 | 0.704488 | 0.010786 | 0.564751 | 0.662419 | 0.420297 | 1.653550 | 13.338071 | 1.000000 | NA | NA | 0.230512 |

## codonGPT Seed Variability

- Seeds: `10`; CAI delta 95% CI: [0.009121, 0.012387]; sign-flip p vs native: `0.004498`.
- This quantifies the public pretrained checkpoint's sampling variability. It is not the unreleased RL policy result.
