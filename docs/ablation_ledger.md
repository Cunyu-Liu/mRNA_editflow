# mRNA-EditFlow Ablation Ledger

This ledger tracks Task 6 baselines and ablations. The local status is limited
to CPU/offline smoke runs; paper-grade claims still require matched data splits,
multi-seed training and bootstrap/significance analysis.

Smoke entry point:

```bash
mrna_editflow/scripts/run_ablation.sh --dry-run
mrna_editflow/scripts/run_ablation.sh --smoke
```

Local evidence snapshot (offline/synthetic only):

- `mrna_editflow/scripts/run_ablation.sh --dry-run`: command matrix enumerates all Task 6 ablation switches and exits 0 locally.
- `mrna_editflow/scripts/run_ablation.sh --smoke`: executable smoke path covers masked diffusion, AR LM, mRNABench synthetic probe and external-model protocol records.
- `mrna_editflow/benchmark/paper_table.md`: persisted synthetic `eval.run_eval` table with bootstrap CI and paired p-value plumbing.
- `mrna_editflow/benchmark/results.json`: persisted synthetic benchmark JSON generated from guided synthetic candidates.
- `mrna_editflow/benchmark/mrnabench_probe.json`: synthetic fallback linear-probe run; current local trainable probe params = 32.
- `mrna_editflow/benchmark/parameter_efficiency.json`: backbone-family protocol and external-model protocol-difference record.
- `mrna_editflow/benchmark/test_coverage_manifest.json`: requirement-to-test matrix only; no line coverage measurement is claimed.

Scope note: these artifacts close local smoke/dry-run evidence. They do not
claim real external mRNABench scores, real ViennaRNA/RNAfold execution, or
paper-grade multi-seed external-model performance.

| Ablation | Scientific question | Command or switch | Expected metrics | Status |
| --- | --- | --- | --- | --- |
| RoPE on/off | Does rotary position encoding improve long transcript modelling versus absolute positions? | `ModelConfig(use_rope=True/False)` | validation loss, legality, k-mer JS, length control, downstream oracle score | Planned; listed in dry-run |
| Region FiLM on/off | Does explicit 5UTR/CDS/3UTR conditioning help heterogeneous mRNA editing? | `ModelConfig(use_region_film=True/False)` | region-specific edit error, CDS legality, UTR motif preservation, oracle score | Planned; listed in dry-run |
| Codon constraint on/off | Does synonymous codon-lattice masking preserve protein/frame constraints? | `ModelConfig(use_codon_constraint=True/False)` | valid CDS fraction, frame intact fraction, protein identity, edit budget | Planned; listed in dry-run |
| Whole-codon indel on/off | Are frame-safe whole-codon CDS indels useful beyond strict CDS lock? | `ModelConfig(codon_indel=True/False)` | frame intact fraction, protein identity, length control, CDS edit rate | Planned; listed in dry-run |
| Aux structure on/off | Do MFE/start-accessibility auxiliary heads improve biologically useful designs? | `ModelConfig(use_aux_struct=True/False, aux_loss_weight)` | start accessibility proxy, MFE proxy, oracle TE/MRL, training stability | Planned; listed in dry-run |
| Guidance scale | Does property/oracle guidance improve objectives without breaking legality? | `sample_mrna(..., guidance_scale, target_te, target_start_accessibility, oracle)` with `LocalTranslationOracle` rerank | oracle score, legality, novelty, diversity, edit budget | Local sampler support + unittest smoke; persisted synthetic eval in `benchmark/results.json` |
| EditFlow vs masked diffusion | Does variable-length edit flow outperform a fixed-canvas denoising baseline? | `train_stage_a/sample_mrna` vs `train_masked_diffusion/sample_masked_diffusion` | legality, length control, diversity, k-mer JS, oracle score, runtime | Smoke implemented |
| Three-way coupling ratio | Which mixture of empty growth, corruption refinement and ortholog coupling is most robust? | `CouplingConfig(1,0,0)`, `(0,1,0)`, `(0,0,1)`, default mixed | validation loss, edit budget, ortholog conservation proxy, novelty | Planned; ortholog arm listed |
| Backbone family | How much signal comes from the encoder versus the EditFlow head? | `mrna_editflow.eval.frozen_backbone_comparison` over `BackboneConfig(name='none')`, mRNA-native, ncRNA-control | probe score, generation metrics, trainable params, runtime | Leakage-gated matched-budget harness implemented (`eval/frozen_backbone_comparison.py`, 9 tests): identical trainable budget across arms, leakage gate refuses unfair comparison, real (`none`) vs offline stub arms tagged `valid_quality_signal`. Real external checkpoints still pending at the `FrozenBackbone._try_load_pretrained` seam |
| AR LM baseline | How does left-to-right generation compare to edit operations? | `train_ar_lm/sample_ar_lm` | likelihood loss, legality, length control, diversity, oracle score | Smoke implemented |
| External models | Which published systems are directly comparable, and where do protocols differ? | `list_external_results(task_id)` | protocol coverage, required inputs, unavailable checkpoint notes | Offline `ExternalResult` records implemented; protocol differences persisted in `benchmark/parameter_efficiency.json` |
| mRNABench probe | Can frozen/proxy representations support benchmark-style linear probing? | `run_mrnabench_probe(records=None)` | trainable params, probe loss, task accuracy once labels exist | Synthetic fallback implemented; `benchmark/mrnabench_probe.json` records 32 trainable params |

Minimum completion criteria for paper tables:

1. Same train/validation/test split for MEF and every trainable baseline.
2. At least 10 seeds for headline comparisons.
3. Matched parameter budget or explicit parameter-count normalization.
4. Bootstrap confidence intervals and paired significance where paired samples exist.
5. Protocol-difference notes for every external model that cannot be run locally.
