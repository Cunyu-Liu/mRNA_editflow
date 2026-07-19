# Paper Table 4: Architecture Ablation

- Claim policy: Table 4 is an architecture-ablation table over existing proxy/offline artifacts. Negative or non-significant modules must be reported as such. Do not turn region-adapter or cascade trends into positive SOTA claims.
- Modules: `5`; signals: `{'negative_ablation': 1, 'constructive_constraint_positive': 1, 'recall_positive_regret_mixed': 1, 'trend_not_significant': 1, 'not_positive': 1}`; positive SOTA claim ready: `False`

| Module | Scope | Main result | Constraints / scope | Signal | Claim language |
|---|---|---|---|---|---|
| Region adapters / FiLM-style conditioning | T5 head256/head1024 region-specialized adapters | head256 best=region_adapter_utr3_top64 delta=-0.00543 p=0.00450; head1024 best=region_adapter_all_top64 delta=-0.00439 p=0.00450 | head256 exact1=True; head1024 exact1=True | negative_ablation | Report as failed/negative ablation; hard constraints safe but TE regresses. |
| Synonymous codon mask / codon-lattice DP | T4 CDS-only synonymous optimization | protein_identity=1.0000; delta_CAI=0.02682; delta_GC=0.00645; mean_codon_changes=3.0000 | protein identity = 1.0 | constructive_constraint_positive | Supports exact CDS protein-preserving optimization; not external LinearDesign/codonGPT. |
| Source-aware teacher / recall ranker | T5 proposal-ranking head64 diagnostic | base recall=0.03279; ranker recall=0.42623; source-aware recall=0.75410; source-aware regret=0.03088 | proposal-ranking diagnostic only | recall_positive_regret_mixed | Source-aware teacher improves oracle-best recall, but does not by itself prove top-1 TE gain. |
| Source-aware cascade recall -> precision | T5 cascade decoding head64 | old cascade vs sequential delta=0.00092 p=0.09545; oracle-best recall k64=0.80328; win_fraction=0.54688 | legal/protein/budget/frame = 1.0 in multiseed comparisons | trend_not_significant | Old cascade has small non-significant trend; use cautious language only. |
| Stage A 10k recall -> hardneg_v2 precision cascade | T5 head256 10k recall cascade | delta=-0.00048; p=0.36632; run_mean=0.00455 | legal/protein/budget/frame = 1.0 | not_positive | 10k recall did not improve top-1 TE over hardneg_v2; report as negative cascade ablation. |
