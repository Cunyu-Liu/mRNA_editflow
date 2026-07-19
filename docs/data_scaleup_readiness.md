# Data Scale-Up Readiness

- Claim policy: Data scale-up readiness is infrastructure and corpus-governance evidence. Do not claim RefSeq-scale training, real MPRA/TE/stability prediction, family-disjoint leakage-safe splits, or a data-scale law until the raw files, cleaned records, manifests, split/leakage audits, and downstream T1-T7/scale-law evaluations are complete.
- Ready for data scale claim: `False`; RefSeq corpus ready: `False`; real TE/stability ready: `False`; family leakage ready: `False`
- GENCODE manifest ready: `True`; GENCODE local records exist: `True`; Stage A sweep status: `queued_or_running`
- Missing or incomplete: `['refseq_official_corpus', 'stage_a_data_model_step_sweep', 'stage_a_downstream_evaluation', 'real_mpra_te_stability_data', 'family_disjoint_split_and_leakage_audit', 'all_dataset_manifests']`

| Area | Status | Ready | Evidence | Claim language |
|---|---|---|---|---|
| GENCODE base corpus | manifest_and_local_records_verified | `True` | clean_n=54680; records_sha=5a74b0ea8d40065fd44383625ab17a4b4bc6e33b7185b67056edda7ecb7448d3 | GENCODE manifest/SHA evidence is present; local records availability is reported separately. |
| RefSeq corpus scale-up | queued_or_running | `False` | raw=True; records=False; manifest=False; last_event=build_start | RefSeq parser/build queue readiness only until official raw, records, and manifest exist. |
| Stage A data/model/step sweep | queued_or_running | `False` | runs=None/8; last_event=None; last_loadavg=None | Controlled axes are queue evidence until all runs and downstream audits complete. |
| Stage A downstream evaluation | blocked_on_stage_a_sweep | `False` | training=0/8; downstream=0/8; aggregate=False; trend=False | Stage A checkpoints need proposal-ranking, T1-T7 aggregate, runtime, and trend audits before any scale-law claim. |
| MPRA/TE/stability data | synthetic_predictor_smoke_only_no_real_data | `False` | tooling=True; predictor_tooling=True; stability_tooling=True; manifest_builder=True; mpra_table=False; stability_table=False; real_artifacts=0; te_predictor_audit=False; stability_predictor_audit=False; synthetic_smoke=(True, True) | No real TE/stability predictor claim without external data and validation artifacts. |
| Family-disjoint split and leakage | blocked_on_refseq_records_real_gencode_split_ready | `False` | split_files=2; cross_corpus_reports=1; protocol_reports=2; real_protocol_reports=1; split_protocol_ready=True; protocol_smoke=False | Leakage tooling exists, but GENCODE/RefSeq family-disjoint evidence awaits RefSeq records. |
| Dataset manifests | incomplete | `False` | gencode_human_transcripts=manifest_and_local_records_verified; refseq_human_rna=queued_or_running; contract=1/4; incomplete=['refseq_human_rna', 'mpra_te', 'stability_half_life'] | Every data version needs source URL, SHA256, record counts, drop stats, and split stats. |
