# mRNA-EditFlow SOTA Readiness Audit

- Slice: head256
- Top-k: 64
- All ready for SOTA claim audit: True
- Positive SOTA claim ready: False
- Internal proxy constrained-optimization claim ready: True
- External SOTA metric claim ready: False
- Full de novo claim ready: False
- Real TE/stability claim ready: False
- True scale-law claim ready: False
- Wet-lab claim ready: False
- Positive SOTA blockers: ['external_sota_real_metrics_missing', 'full_de_novo_evidence_missing_or_overclaim_flagged', 'real_mpra_te_or_stability_data_missing', 'true_data_model_step_scale_law_missing', 'head1024_fusion_vs_strong_te_only_not_strict', 'wet_lab_validation_missing']
- Allowed claim scope: Constrained local full-length mRNA optimization/reranking with proxy/offline T1-T7 evidence. Do not state full de novo, wet-lab, external SOTA, or true scale-law claims until the blocking gates clear.
- Sections ready: 6/6
- Pending sections: []
- Claim policy: Ready only means evidence is complete and hard constraints are auditable; effect sizes and p-values still determine whether claims are positive, borderline, non-significant, or negative.

| section | ready | key status |
|---|---:|---|
| region_adapter | True | compare files 5/5; constraints exact 1=True; persisted audit=True |
| protein_conditioned_gc_sweep | True | points=8; identity exact 1=True; pareto metadata=True; persisted audit=True |
| external_sota_protocol | True | protocol ready=True; input pack ready=True; real-run audit=True; model set consistent=True; measured=3/6; T5 UTR descriptive/model-only=True/False; input rows=1024/1024; models=6; executable ready=4; real metrics not claimed |
| multiobjective_scaleup_claims | True | comparison rows=True; hard constraints=True; head256 strict=True; head1024 vs te_only strict=False; best signal=borderline_positive |
| frozen_foundation_protocol | True | protocol ready=True; leakage gate=True; matched budget=True; real/stub arms=1/2; real metrics not claimed |
| t1_t7_evidence_bundle | True | reports ready=9/9; failed checks=[]; proxy/constraint reports only |

## Missing Artifacts

| section | path |
|---|---|
