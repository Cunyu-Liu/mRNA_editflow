# mRNA-EditFlow SOTA Gap Report

## Measured MEF Evidence

| Evidence | Metric | Baseline | Run | Delta | 95% CI | paired p | n | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TE-ranker fair head decoding (n=32) | `delta_oracle_te_vs_source` | -0.00526 | 0.00520 | 0.01046 | [0.00752, 0.01297] | 0.00450 | 10 | `benchmark/t5_ranker_comparison.json` |
| TE-ranker fair head decoding (n=256) | `delta_oracle_te_vs_source` | -0.00154 | 0.00406 | 0.00560 | [0.00470, 0.00660] | 0.00450 | 10 | `benchmark/t5_ranker_full1k_head256_comparison.json` |
| All-legal-proposal TE oracle upper bound (n=32) | `delta_oracle_te_vs_source` | -0.00016 | 0.06930 | 0.06946 | [0.06942, 0.06950] | 0.00450 | 10 | `benchmark/t5_guidance_comparison.json` |
| Cascade decoding source-aware->sequential top64 (n=256) | `delta_oracle_te_vs_source` | 0.00391 | 0.00482 | 0.00092 | [0.00006, 0.00187] | 0.09545 | 10 | `benchmark/compare_t5_head256_cascade_vs_seq_top64.json` |
| Cascade decoding source-aware->sequential top64 (n=256) | `mean_oracle_te` | 0.78374 | 0.78466 | 0.00092 | [0.00006, 0.00187] | 0.09545 | 10 | `benchmark/compare_t5_head256_cascade_vs_seq_top64.json` |
| Hard-negative v2 direct top64 decoding (n=256) | `delta_oracle_te_vs_source` | 0.00391 | 0.00503 | 0.00112 | [0.00041, 0.00180] | 0.02049 | 10 | `benchmark/compare_t5_head256_hardneg_v2_top64.json` |
| Hard-negative v2 direct top64 decoding (n=256) | `mean_oracle_te` | 0.78374 | 0.78486 | 0.00112 | [0.00041, 0.00180] | 0.02049 | 10 | `benchmark/compare_t5_head256_hardneg_v2_top64.json` |
| Hard-negative v2 direct top64 decoding (n=1024) | `delta_oracle_te_vs_source` | 0.00239 | 0.00385 | 0.00147 | [0.00114, 0.00179] | 0.00450 | 10 | `benchmark/compare_t5_head1024_hardneg_v2_top64.json` |
| Hard-negative v2 direct top64 decoding (n=1024) | `mean_oracle_te` | 0.79342 | 0.79489 | 0.00147 | [0.00114, 0.00179] | 0.00450 | 10 | `benchmark/compare_t5_head1024_hardneg_v2_top64.json` |
| Multi-objective grpo-fusion ranker top64 (vs single-TE control) (n=256) | `delta_oracle_te_vs_source` | 0.00348 | 0.01114 | 0.00765 | [0.00718, 0.00815] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| Multi-objective scalar-fusion ranker top64 (vs single-TE control) (n=256) | `delta_oracle_te_vs_source` | 0.00348 | 0.01087 | 0.00739 | [0.00694, 0.00785] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| Multi-objective pareto-fusion ranker top64 (vs single-TE control) (n=256) | `delta_oracle_te_vs_source` | 0.00348 | 0.01029 | 0.00680 | [0.00608, 0.00756] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head256.json` |
| Multi-objective grpo-fusion ranker top64 scale-up (vs single-TE control) (n=1024) | `delta_oracle_te_vs_source` | 0.00846 | 0.00852 | 0.00006 | [-0.00058, 0.00061] | 0.89805 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| Multi-objective scalar-fusion ranker top64 scale-up (vs single-TE control) (n=1024) | `delta_oracle_te_vs_source` | 0.00846 | 0.00855 | 0.00008 | [-0.00045, 0.00050] | 0.74463 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| Multi-objective pareto-fusion ranker top64 scale-up (vs single-TE control) (n=1024) | `delta_oracle_te_vs_source` | 0.00846 | 0.00927 | 0.00081 | [0.00018, 0.00145] | 0.05047 | 10 | `benchmark/compare_mo_fusion_vs_te_only_head1024.json` |
| Multi-objective scalar-fusion ranker top64 scale-up (vs prior champion hardneg_v2) (n=1024) | `delta_oracle_te_vs_source` | 0.00385 | 0.00855 | 0.00469 | [0.00407, 0.00527] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |
| Multi-objective pareto-fusion ranker top64 scale-up (vs prior champion hardneg_v2) (n=1024) | `delta_oracle_te_vs_source` | 0.00385 | 0.00927 | 0.00542 | [0.00481, 0.00603] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |
| Multi-objective grpo-fusion ranker top64 scale-up (vs prior champion hardneg_v2) (n=1024) | `delta_oracle_te_vs_source` | 0.00385 | 0.00852 | 0.00467 | [0.00406, 0.00520] | 0.00450 | 10 | `benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json` |
| Region-specialized 5UTR adapter top64 (vs hardneg_v2) (n=256) | `delta_oracle_te_vs_source` | 0.00503 | -0.00119 | -0.00622 | [-0.00694, -0.00551] | 0.00450 | 10 | `benchmark/region_adapter_vs_hardneg_v2_top64_head256.json` |
| Region-specialized CDS adapter top64 (vs hardneg_v2) (n=256) | `delta_oracle_te_vs_source` | 0.00503 | -0.00055 | -0.00557 | [-0.00641, -0.00475] | 0.00450 | 10 | `benchmark/region_adapter_vs_hardneg_v2_top64_head256.json` |
| Region-specialized 3UTR adapter top64 (vs hardneg_v2) (n=256) | `delta_oracle_te_vs_source` | 0.00503 | -0.00040 | -0.00543 | [-0.00646, -0.00444] | 0.00450 | 10 | `benchmark/region_adapter_vs_hardneg_v2_top64_head256.json` |
| Region-specialized all-region adapters top64 (vs hardneg_v2) (n=256) | `delta_oracle_te_vs_source` | 0.00503 | -0.00157 | -0.00659 | [-0.00773, -0.00566] | 0.00450 | 10 | `benchmark/region_adapter_vs_hardneg_v2_top64_head256.json` |
| Region-specialized all-region adapters top64 (vs MO-GRPO) (n=256) | `delta_oracle_te_vs_source` | 0.01114 | -0.00157 | -0.01270 | [-0.01350, -0.01190] | 0.00450 | 10 | `benchmark/region_adapter_vs_mo_grpo_top64_head256.json` |
| Region-specialized all-region adapters top64 (vs MO-scalar) (n=256) | `delta_oracle_te_vs_source` | 0.01087 | -0.00157 | -0.01244 | [-0.01359, -0.01127] | 0.00450 | 10 | `benchmark/region_adapter_vs_mo_scalar_top64_head256.json` |
| Region-specialized all-region adapters top64 (vs MO-pareto) (n=256) | `delta_oracle_te_vs_source` | 0.01029 | -0.00157 | -0.01185 | [-0.01279, -0.01099] | 0.00450 | 10 | `benchmark/region_adapter_vs_mo_pareto_top64_head256.json` |
| Region-specialized all-region adapters top64 (vs MO te_only) (n=256) | `delta_oracle_te_vs_source` | 0.00348 | -0.00157 | -0.00505 | [-0.00589, -0.00409] | 0.00450 | 10 | `benchmark/region_adapter_vs_mo_te_only_top64_head256.json` |
| Head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.02985 | -0.00827 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json` |
| Head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.42623 | 0.39344 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json` |
| UTR-teacher head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.03212 | -0.00599 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_utr_teacher_head64.json` |
| UTR-teacher head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.70492 | 0.67213 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_utr_teacher_head64.json` |
| Hybrid-teacher head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.03156 | -0.00655 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_hybrid_teacher_head64.json` |
| Hybrid-teacher head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.47541 | 0.44262 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_hybrid_teacher_head64.json` |
| Full-then-UTR head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.02798 | -0.01014 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_full1k_then_utr_teacher_head64.json` |
| Full-then-UTR head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.40984 | 0.37705 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_full1k_then_utr_teacher_head64.json` |
| Source-aware hybrid head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.03088 | -0.00724 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json` |
| Source-aware hybrid head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.75410 | 0.72131 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json` |
| Cascade hard-negative v2 head64 proposal-ranking regret | `mean_model_regret` | 0.03812 | 0.02526 | -0.01285 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_cascade_hardneg_teacher_head64.json` |
| Cascade hard-negative v2 head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.03279 | 0.73770 | 0.70492 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_full1k_head64.json vs benchmark/proposal_ranking_t5_cascade_hardneg_teacher_head64.json` |
| Stage A 10k head1024 proposal-ranking regret | `mean_model_regret` | 0.00000 | 0.04029 | 0.04029 | NA | NA | 1024 | `benchmark/proposal_ranking_t5_stage_a10k_head1024.json` |
| Stage A 10k head1024 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.00000 | 0.05263 | 0.05263 | NA | NA | 1024 | `benchmark/proposal_ranking_t5_stage_a10k_head1024.json` |
| Stage A 10k teacher-ranker head64 proposal-ranking regret | `mean_model_regret` | 0.03309 | 0.03298 | -0.00011 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_stage_a10k_head64.json vs benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.json` |
| Stage A 10k teacher-ranker head64 oracle-best-in-model-top32 fraction | `oracle_best_in_model_top_k_fraction` | 0.01639 | 0.19672 | 0.18033 | NA | NA | 64 | `benchmark/proposal_ranking_t5_base_stage_a10k_head64.json vs benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.json` |
| Cascade source-aware->sequential head64 regret (k=32) | `mean_cascade_regret` | 0.03088 | 0.02970 | -0.00118 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_sequential_head64.json` |
| Cascade source-aware->sequential oracle-best recall (k=32) | `oracle_best_in_recall_top_k_fraction` | 0.00000 | 0.75410 | 0.75410 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_sequential_head64.json` |
| Cascade source-aware->sequential head64 regret (k=64) | `mean_cascade_regret` | 0.02798 | 0.02788 | -0.00009 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_sequential_head64_k64.json` |
| Cascade source-aware->sequential oracle-best recall (k=64) | `oracle_best_in_recall_top_k_fraction` | 0.00000 | 0.80328 | 0.80328 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_sequential_head64_k64.json` |
| Cascade source-aware->hard-negative v2 head64 regret (k=64) | `mean_cascade_regret` | 0.02526 | 0.02664 | 0.00138 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_hardneg_head64_k64.json` |
| Cascade source-aware->hard-negative v2 oracle-best recall (k=64) | `oracle_best_in_recall_top_k_fraction` | 0.00000 | 0.80328 | 0.80328 | NA | NA | 64 | `benchmark/cascade_sourceaware_to_hardneg_head64_k64.json` |
| CDS codon-lattice DP CAI | `mean_cai` | 0.67322 | 0.70004 | 0.02682 | NA | NA | 256 | `benchmark/codon_lattice_dp_head256.json` |
| CDS codon-lattice DP GC shift | `mean_gc` | 0.61309 | 0.61954 | 0.00645 | NA | NA | 256 | `benchmark/codon_lattice_dp_head256.json` |
| Foundation leakage audit head256 vs GENCODE | `flagged_fraction` | 0.00000 | 1.0000 | 1.0000 | NA | NA | 256 | `benchmark/leakage_ranker_head256_vs_gencode.json` |
| Foundation leakage audit exact matches | `exact_match_count` | 0.00000 | 256.0000 | 256.0000 | NA | NA | 256 | `benchmark/leakage_ranker_head256_vs_gencode.json` |
| UTR local-search TE baseline | `mean_oracle_te` | 0.77984 | 0.86725 | 0.08741 | NA | NA | 256 | `benchmark/utr_local_search_head256.json` |
| UTR one-step teacher headroom | `best_one_step_oracle_te` | 0.77838 | 0.82835 | 0.04997 | NA | NA | 256 | `benchmark/utr_teacher_head256.json` |

## Oracle Gap Health Check

- All-legal oracle delta_TE: `0.06930`
- Best ranker delta_TE: `0.01114`
- Remaining oracle gap: `0.05816`
- Caution: The upper bound is an oracle-guided target. Treat the gap as a health check unless generated under the same guarded config as the model-only comparison.

## External SOTA Integration Table

| Method | Venue/year | Scope | Accuracy/F1 signal | Speed/scale | MEF gap/action | Registered | Citation |
|---|---|---|---|---|---|---:|---|
| LinearDesign | Nature 2023 | protein-conditioned CDS structure/codon dynamic programming | Not an accuracy/F1 task; optimize folding/codon objective. | Dynamic-programming CDS optimizer; strong deterministic baseline. | Run CDS-only T4 benchmark with CAI/MFE/proxy TE and wall-clock comparison. | yes | https://doi.org/10.1038/s41586-023-06127-z |
| EnsembleDesign | Bioinformatics 2025 | CDS ensemble free-energy optimization | Objective-optimization task, not classification. | External lattice parser; runtime benchmark pending. | Add ensemble-free-energy objective to T4/CDS benchmark. | yes | https://doi.org/10.1093/bioinformatics/btaf245 |
| mRNA-LM | Nucleic Acids Research 2025 | full-length mRNA analysis and representation | Task dependent; compare frozen-probe accuracy/F1 on mRNABench-style tasks. | Foundation encoder; adapter/probe runtime pending. | Install checkpoint or embedding cache and run frozen-probe plus MEF-head ablation. | yes | https://doi.org/10.1093/nar/gkaf044 |
| codonGPT | Nucleic Acids Research 2025 | codon-level generative LM plus RL for CDS design | Generative/RL task; use reward, validity and protein-preservation metrics. | Scalable codon LM; exact local throughput pending. | Compare CDS-only T4 generation, then show MEF advantage from joint UTR editing. | yes | https://doi.org/10.1093/nar/gkaf1345 |
| GEMORNA | Science 2025 | de novo full-length mRNA design (encoder-decoder CDS + decoder-only UTR, then combined) | Generation task; report wet-lab/oracle protein output, CAI, MFE, rare-codon, and TE proxy. | Zero-shot autoregressive decoders; modular per-region generation. | Align T4 CDS-only + T5 UTR-only design under one protocol; MEF must show full-length constrained-edit advantage vs GEMORNA's modular combine, and report that MEF preserves 100% protein identity (hard constraint) rather than free CDS regeneration. | no | https://doi.org/10.1126/science.adr8470 |
| mRNA-GPT | ICLR 2026 submission | end-to-end full-length mRNA design/optimization (decoder-only, joint 5'UTR+CDS+3'UTR) | Generation task; report predicted TE/half-life, diversity, cross-region interaction gain, protein preservation. | Decoder-only LM; flexible single-region / full-length / conditional generation modes. | This is the closest full-length rival: run head-to-head full-length T1-T7 with matched proxy oracle; MEF's differentiator is constrained edit-flow (protein/frame/budget hard constraints + interpretable edit distance) vs mRNA-GPT's free autoregressive generation. Quantify TE delta AND constraint satisfaction jointly. | no | https://openreview.net/pdf?id=juUrI9kCBw |
| ProMORNA | arXiv 2026 | protein-conditioned de novo full-length mRNA via multi-objective RL (BART encoder-decoder + MO-GRPO) | Multi-objective generation; report per-objective Pareto frontier (TE, half-life, immune-safety) and protein-conditioned validity. | GRPO RL over protein prompts; no wild-type template needed at inference. | Directly informs MEF upgrade #1 (multi-objective) and #3 (protein-conditioned): adopt per-metric advantage standardization as an alternative to scalar/Pareto fusion, and align MEF protein-conditioned CDS + multi-objective ranker against ProMORNA's Pareto-frontier metric on a shared held-out protein. | no | https://arxiv.org/abs/2605.01513 |
| RNAGenScape | ICML 2025 GenBio workshop / arXiv 2025 | property-guided optimization + interpolation of existing mRNAs via manifold Langevin dynamics (organized autoencoder + manifold projector) | Optimization task; report property gain, success rate, on-manifold biological plausibility, and interpolation smoothness. | Latent Langevin dynamics; efficient inference, no explicit score/density learning. | Closest philosophy to MEF ('optimize, not invent' from real sequences): benchmark MEF's discrete constrained edit-flow trajectories against RNAGenScape's continuous latent trajectories on shared 5'UTR TE optimization; report interpretability (edit distance/budget) and constraint guarantees MEF adds over latent projection. | no | https://arxiv.org/abs/2510.24736 |
| CodonFM | NVIDIA open model 2025 | codon-level RNA foundation encoder | Task dependent; compare property-prediction accuracy/F1 after frozen probing. | Large RefSeq-scale pretraining; embedding-cache runtime pending. | Add leakage-audited frozen embeddings and measure TE/stability probe transfer. | yes | https://developer.nvidia.com/blog/introducing-the-codonfm-open-model-for-rna-design-and-analysis/ |
| UTailoR | iScience 2025 | 5'UTR discriminative and generative optimization | Predictor/generator task; use TE proxy and held-out predictor agreement. | 5'UTR-specific service/model; local throughput pending. | Run T5/T7 UTR-only benchmark and verify MEF preserves full transcript constraints. | yes | https://doi.org/10.1016/j.isci.2025.113544 |
| Helix-mRNA | arXiv 2025 | long-context full mRNA foundation encoder | Task dependent; compare frozen-probe accuracy/F1 and generation-head transfer. | Long-context embedding cache needed for fair runtime. | Use as frozen backbone and compare against backbone=none under same trainable budget. | yes | https://arxiv.org/abs/2502.13785 |
| Prot2RNA | OpenReview 2026 submission | protein-conditioned CDS diffusion | Use codon accuracy, codon-usage profile and protein-preservation metrics. | External diffusion model; runtime pending. | Isolate CDS-conditioned T4 benchmark; report that Prot2RNA has no UTR edit task. | yes | https://openreview.net/forum?id=BPNK5HDEMh |

## External SOTA Dry-Run Readiness

- Status: `dry_run_complete`; task: `T5`; ready: `5` / 6

- Dataset audit: split=`public_head1024`, seed=`0`, records=`1024` / 1024, sha256=`8e2a1b6de75d418bafe42241fe18d36093fa50fde732b56866a2e4e8fc1231ec`

- Hardware audit: label=`a100-server`, hostname=`bms-18937653-012`, machine=`x86_64`

- Real metric policy: Do not report accuracy/F1/TE/runtime metrics for an external method until its row status is executable_ready and a real adapter writes measured outputs under benchmark/external_sota/.

| Model | Status | Command candidates | Protocol scope | Next setup |
|---|---|---|---|---|
| LinearDesign | `executable_ready` | `LINEARDESIGN_BIN, lineardesign, LinearDesign` | Protocol difference: LinearDesign is a deterministic CDS optimizer with strong biological validation; it is not trained as a stochastic full-transcript edit model and does not optimize UTR edits. | Run the real adapter and write measured outputs under `benchmark/external_sota/`. Candidate audit: LINEARDESIGN_BIN=executable. |
| EnsembleDesign | `executable_ready` | `ENSEMBLEDESIGN_BIN, ensembledesign, EnsembleDesign` | Protocol difference: EnsembleDesign targets structure/codon optimization in the protein-conditioned CDS lattice; MEF should compare against it on CDS structure objectives, not full UTR editing tasks. | Run the real adapter and write measured outputs under `benchmark/external_sota/`. Candidate audit: ENSEMBLEDESIGN_BIN=executable. |
| codonGPT | `executable_ready` | `CODONGPT_BIN, codongpt, codonGPT` | Protocol difference: codonGPT focuses on CDS-constrained synonymous generation; it does not jointly edit 5'UTR, CDS and 3'UTR under one region-aware variable-length process. | Run the real adapter and write measured outputs under `benchmark/external_sota/`. Candidate audit: CODONGPT_BIN=executable. |
| Prot2RNA | `not_configured` | `PROT2RNA_BIN, prot2rna, Prot2RNA` | Protocol difference: Prot2RNA is protein/CDS-conditioned and does not directly model UTR editing or full-transcript variable-length tasks; comparisons should isolate CDS design from full mRNA generation. | Set one candidate as an environment variable or install it on PATH, then rerun. Candidate audit: PROT2RNA_BIN=env_unset; prot2rna=path_not_found; Prot2RNA=path_not_found. |
| UTailoR | `executable_ready` | `UTAILOR_BIN, utailor, UTailoR` | Protocol difference: UTailoR is focused on 5'UTR optimization; it does not preserve or edit CDS/3'UTR with an explicit CTMC grammar. | Run the real adapter and write measured outputs under `benchmark/external_sota/`. Candidate audit: UTAILOR_BIN=executable. |
| UTRGAN | `executable_ready` | `UTRGAN_BIN, utrgan, UTRGAN` | Protocol difference: UTRGAN optimizes/generates UTRs rather than full 5'UTR-CDS-3'UTR transcripts; it has no codon-frame edit operator and is not evaluated on variable-length edit tasks. | Run the real adapter and write measured outputs under `benchmark/external_sota/`. Candidate audit: UTRGAN_BIN=executable. |

## External SOTA Input Pack

- Ready for external real run: `True`; ready for external SOTA claim: `False`
- Dataset: split=`public_head1024`, seed=`0`, records=`1024` / 1024, sha256=`8e2a1b6de75d418bafe42241fe18d36093fa50fde732b56866a2e4e8fc1231ec`
- Rows: CDS/protein-conditioned=`1024`, 5'UTR-only=`1024`, skipped invalid CDS=`0`
- Pack SHA: CDS=`77e9d89aa5736fe995bfbbbd7f314456d315f73017a8d3d45c7c5745699d60c4`, 5'UTR=`83d5032a6a4123970e1361dce2fba5ddb76c1bdab7f729cc7f85f0a65a0c1ca1`, metric schema=`1ee80c0db59f29ffa4b29c76faf4103a4c6a27ef5cb25a52940b72a61617967a`
- Claim policy: External SOTA input packs are reproducibility/preflight artifacts only. They define the exact input rows and expected output schema for external tools, but do not imply that LinearDesign, EnsembleDesign, codonGPT, Prot2RNA, UTailoR or UTRGAN has been executed, reproduced, or beaten.

## External SOTA Real-Run Audit

- Audit complete: `True`; real metric table ready: `False`; external metric claim ready: `False`
- Measured models: `4` / `6`; invalid: `1`; missing: `1`

| Model | Task family | Status | Outputs | Constraints | Reasons |
|---|---|---|---:|---|---|
| LinearDesign | cds_protein_conditioned | `measured` | 1024 / 1024 | `True` |  |
| EnsembleDesign | cds_protein_conditioned | `invalid` | 1016 / 1024 | `True` | expected_output_transcript_coverage_mismatch |
| codonGPT | cds_protein_conditioned | `measured` | 1024 / 1024 | `True` |  |
| Prot2RNA | cds_protein_conditioned | `missing` | 0 / 1024 | `False` | summary_missing, outputs_jsonl_missing |
| UTailoR | utr5_only | `measured` | 315 / 1024 | `True` |  |
| UTRGAN | utr5_only | `measured` | 1024 / 1024 | `True` |  |

## T5 External 5'UTR Comparison

- Descriptive table ready: `True`; model-only head-to-head ready: `False`; MEF superiority claim ready: `False`
- Hard constraints exact-1: `True`; paired inference ready: `False`

| Method | Status | n | TE | delta TE | uAUG | Kozak | access | UTR edit | CDS/3UTR/protein |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| native_source | `measured_reference` | 1024 | 0.79104 | 0.00000 | 0.54785 | 0.60156 | 0.57959 | 0.00000 | 1.0000/1.0000/1.0000 |
| MEF_full_length_mo_pareto_top64 | `measured_internal_model_context` | 1024 | 0.80031 | 0.00927 | 0.51172 | 0.59761 | 0.63356 | NA | 1.0000/NA/1.0000 |
| MEF_region_adapter_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.78626 | -0.00478 | 0.55186 | 0.59204 | 0.61824 | 2.7871 | 1.0000/1.0000/1.0000 |
| MEF_pure_utr_teacher_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.79422 | 0.00318 | 0.57529 | 0.59688 | 0.62835 | 2.7963 | 1.0000/1.0000/1.0000 |
| MEF_full_then_utr_teacher_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.79401 | 0.00298 | 0.56641 | 0.59946 | 0.62805 | 2.7476 | 1.0000/1.0000/1.0000 |
| MEF_pure_utr_teacher_budget5_utailor_strict_25_100nt | `measured_internal_model_10seed_utailor_protocol_subset_budget5` | 315 | 0.84100 | 0.00777 | 0.29206 | 0.63302 | 0.61250 | 4.7962 | 1.0000/1.0000/1.0000 |
| MEF_utr5_constrained_local_search_budget3 | `measured_internal_oracle_guided_ceiling` | 1024 | 0.87406 | 0.08302 | 0.03613 | 0.69385 | 0.64997 | 2.9375 | 1.0000/1.0000/1.0000 |
| UTailoR_official_strict_25_100nt | `measured_external_protocol_subset` | 315 | 0.86933 | 0.03611 | 0.06984 | 0.70476 | 0.57633 | 4.4190 | 1.0000/1.0000/1.0000 |
| UTRGAN_official_budgeted_10_steps | `measured_external_budgeted` | 1024 | 0.81976 | 0.02872 | 0.37598 | 0.53467 | 0.63195 | 70.9980 | 1.0000/1.0000/1.0000 |
| UTRGAN_official_paper_default_10000_steps | `measured_external_paper_default` | 1024 | 0.83172 | 0.04068 | 0.38086 | 0.57129 | 0.65263 | 72.4854 | 1.0000/1.0000/1.0000 |

- Remaining fairness action: The MEF hard-budget-5 UTailoR strict-subset run narrows the edit-effort mismatch but is not an exact budget match because official UTailoR is not hard-capped per record. UTRGAN paper-default is complete; paired inference remains invalid because UTRGAN is not source-conditioned.

## Dataset Survey / Leakage Alignment

- Status: `ready`; protocol ready: `True`; path: `docs/mrna_dataset_survey.md`
- SHA-256: `aedd0cf616fdf448510d240251262cb8d05f7499ce2bd5cb9bbec61833e7a00e`
- Covered methods: `['GEMORNA', 'mRNA-GPT', 'ProMORNA', 'RNAGenScape', 'codonGPT']`; missing methods: `[]`
- Mentions leakage/split/license controls: `True`
- Claim policy: Dataset survey is protocol/landscape evidence only. Do not claim external SOTA reproduction until official weights/executables and leakage-free splits are configured.

## Unified SOTA Readiness

- All ready for SOTA claim audit: `True`
- Positive SOTA claim ready: `False`
- Internal proxy constrained-optimization claim ready: `True`
- External SOTA metric claim ready: `False`
- Full de novo claim ready: `False`
- Real TE/stability claim ready: `False`
- True scale-law claim ready: `False`
- Wet-lab claim ready: `False`
- Positive SOTA blockers: `['external_sota_real_metrics_missing', 'full_de_novo_evidence_missing_or_overclaim_flagged', 'real_mpra_te_or_stability_data_missing', 'true_data_model_step_scale_law_missing', 'head1024_fusion_vs_strong_te_only_not_strict', 'wet_lab_validation_missing']`
- Allowed claim scope: Constrained local full-length mRNA optimization/reranking with proxy/offline T1-T7 evidence. Do not state full de novo, wet-lab, external SOTA, or true scale-law claims until the blocking gates clear.
- Sections ready: `6` / `6`
- Pending sections: `[]`
- Claim policy: Ready only means evidence is complete and hard constraints are auditable; effect sizes and p-values still determine whether claims are positive, borderline, non-significant, or negative.

| Section | Ready | Key status |
|---|---:|---|
| external_sota_protocol | `True` | protocol ready=True; input pack ready=True; real-run audit=True; measured=3/6; models=6; executable ready=4; real metrics not claimed |
| frozen_foundation_protocol | `True` | protocol ready=True; leakage gate=True; matched budget=True; real/stub arms=1/2; real metrics not claimed |
| multiobjective_scaleup_claims | `True` | comparison rows=True; hard constraints=True; head256 strict=True; head1024 vs te_only strict=False; best signal=borderline_positive |
| protein_conditioned_gc_sweep | `True` | points=8; identity exact 1=True; pareto metadata=True |
| region_adapter | `True` | compare files 5/5; constraints exact 1=True |
| t1_t7_evidence_bundle | `True` | reports ready=9/9; failed checks=[]; proxy/constraint reports only |

## Next Gates

- Finish EnsembleDesign, UTRGAN paper-default, and codonGPT 10-seed runs; obtain Prot2RNA artifacts if released.
- Add frozen mRNA foundation embeddings with leakage audits.
- Scale beyond the 1000-step Stage A snapshot and repeat proposal-ranking distillation.
