# Paper Table 2: T1-T7 Main Results

- Claim policy: Table 2 is an internal proxy/constraint table. It supports constrained local optimization claims only. It does not support wet-lab validation, external SOTA reproduction, speed SOTA, or full de novo mRNA generation claims.
- Paired p available tasks: `['T1', 'T7']`; NA tasks: `['T2', 'T3', 'T4', 'T5', 'T6']`
- External SOTA claim ready: `False`; wet-lab claim ready: `False`; full de novo claim ready: `False`

| Task | Main result | Hard constraints / scope | paired p | Claim language |
|---|---|---|---|---|
| T1 Validity / Oracle TE | legal=1.0000; delta_TE=0.01114; mean_TE=0.79097; mean_MRL=8.3923 | legal/protein/budget/frame = 1.0 | mo_grpo vs te_only delta_oracle_te_vs_source p=0.00450; mo_grpo vs hardneg_v2 delta_oracle_te_vs_source p=0.00450 | Strict positive proxy-TE claim vs te_only and hardneg_v2; not wet-lab expression. |
| T2 Distribution preservation | kmer_JS=0.00001; codon_KL=0.00000; GC_length_dist=0.00225; length_delta=0.00000 | distribution-collapse flag = False | NA_distribution_audit_no_paired_claim | No local distribution-collapse signal; still needs external split alignment. |
| T3 Diversity / Novelty | mean_novelty=0.00326; exact_match=0.01172; unique=0.99609; pairwise=0.51777 (64/32640 sampled) | novelty exact; pairwise sampled | NA_novelty_audit_no_paired_claim | Novel but not de novo: exact source matches are non-zero. |
| T4 Protein identity / CDS CAI-GC | protein_identity_exact_1=True; local_DP_delta_CAI=0.02682; protein_conditioned_delta_CAI=0.27092 | protein identity = 1.0 | NA_constructive_DP_hard_constraint | Exact protein-identity and CAI/GC proxy claim; not external LinearDesign/codonGPT/MFE. |
| T5 Edit budget | budget3_delta_TE=0.01114; budget3_edit_distance=2.9527; budget10_delta_TE=0.02796; within_budget=1.0000 | legal/protein/budget/frame = 1.0 | NA_curve_audit_no_single_paired_claim | Budget curve supports controllable local edits; larger budgets increase edits. |
| T6 Length control | head256_max_abs_error=0.35156; head1024_max_abs_error=0.52051; deltas=-30/-15/0/+15/+30 | legal/protein/budget/frame = 1.0 | NA_control_curve_no_single_paired_claim | Accurate length control under hard constraints; positive lengthening hurts proxy TE. |
| T7 Motif / Frame | frame=1.0000; uAUG_presence=0.34375; constructive_insert_success=0.99336; constructive_excise_success=0.99336 | frame/protein/budget = 1.0 in constructive edit benchmark | mo_grpo vs te_only uAUG_presence_fraction p=0.04048 | Frame is controlled; grpo increases uAUG vs te_only, so do not claim uAUG safety improvement. |
