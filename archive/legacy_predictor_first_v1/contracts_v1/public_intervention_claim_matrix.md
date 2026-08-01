# Claim Matrix — public_intervention_contract_v1

**Contract:** `configs/public_intervention_contract.yaml` (`public_intervention_contract_v1`)

This matrix fixes what the project may claim, the evidence each claim
requires, and the claims that are permanently forbidden because no new
wet-lab experiments will be performed.

## 1. Allowed claims

| ID | Claim | Type | Required evidence | Boundary wording |
|---|---|---|---|---|
| C1 | We introduce a harmonized benchmark of experimentally measured mRNA variant effects (mRNA-EditBench) and a source-conditioned sparse action model (SparseEditFormer) that predicts local intervention effects across unseen genes, studies and cellular contexts. | Primary | All four sub-benchmarks built with data cards; Split A–E results; macro-averaged metrics vs baselines | "predicted local intervention effects on measured public endpoints" |
| C2 | Explicit `source + edit action` modeling is more reliable than subtracting two absolute property scores. | Secondary | Head-to-head vs candidate-absolute and Siamese subtraction baselines on EditBench-5U-Natural, with source-group bootstrap CIs | limited to measured endpoints |
| C3 | Source cache + sparse action scoring ranks candidates at large scale without re-running a full foundation model per candidate. | Secondary | Compute-matched cost analysis (encodings per source, wall-clock, memory) vs full re-encoding | efficiency claim, not efficacy |
| C4 | Assay-aware uncertainty and abstention reduce false-beneficial predictions under cross-experiment generalization. | Secondary | ECE, coverage–risk curves, selective-prediction task T7 across studies/contexts | "false-beneficial predictions on measured endpoints" |
| C5 | Explicit legal geometry of UTR nucleotide actions and CDS synonymous-codon actions improves validity and interpretability. | Secondary | Action-space validity audit; cross-region reuse analysis; mechanism/attribution analysis | no wet-lab validation implied |
| C6 | Dense synthetic intervention pretraining (EditBench-5U-Dense) improves transfer to natural variants. | Secondary | Curriculum comparison T0-03 (with/without dense pretraining) on natural 5'UTR benchmark | "on measured public endpoints" |
| C7 | SparseEditFlow adds value over the direct sparse scorer. | Conditional | Matched-compute experiment F0-04/F0-05; if no gain, Flow is dropped and the direct-scorer results remain reported | decided by experiment, never预设 |

## 2. Targets tied to claims

| Target set | Threshold |
|---|---|
| Minimum scientific validity (>= 2 independent 5'UTR studies) | delta Spearman >= 0.30; sign accuracy >= 0.60; top-10% enrichment >= 1.5; beat strongest non-foundation baseline |
| Submission stretch | macro delta Spearman >= 0.35; macro sign accuracy >= 0.65; top-10% enrichment >= 1.75; ECE <= 0.10; +0.05 Spearman vs strongest executable baseline with source-group bootstrap 95% CI lower bound > 0 |
| External temporal (sealed GSE246381) | delta Spearman >= 0.25; sign accuracy >= 0.60; top-10% enrichment >= 1.5 — immutable after unseal |
| Optimization (measured candidate space only) | top-10 recall >= 0.70; NDCG >= strongest baseline + 0.05; normalized regret <= 0.10 |

## 3. Forbidden claims (permanent)

1. The model improves protein output of real therapeutic mRNA.
2. Model-discovered candidates are experimentally validated / effective.
3. The model outperforms real experimental screening.
4. MRL improvement equals protein-output improvement.
5. TE improvement equals protein-output improvement.
6. Half-life improvement necessarily improves protein output.
7. General full-length therapeutic mRNA optimization.
8. Observational data treated as intervention labels.
9. Random 50-nt UTR library called WT–mutant pairs.
10. Same-protein synonymous families treated as independent pair samples.
11. Merging half-life and expression into one label.
12. Merging 3'UTR and 5'UTR labels into one regression.
13. Reporting only pooled metrics or only the best seed.
14. Tuning on sealed labels; modifying the test set after external failure.
15. Deleting direct-scorer results because Flow shows no gain.
16. Any therapeutic-improvement claim without experimental support.

## 4. Evidence status ledger

| Claim | Status | Evidence pointer |
|---|---|---|
| C1–C7 | PENDING | populated as Phases D0–O0 complete (see `docs/execution/task_registry.yaml`) |

*No wet-lab claim will ever be added to this matrix.*
