# P3-02: Local-Delta Oracle Protocol

## Phase Goal

Answer two critical questions:
1. Can edit effects be predicted?
2. Is there optimization headroom worth pursuing?

If this phase fails, stop expanding RL.

## Data Sources

| Tier | Records | Label Type | Qualifier |
|------|---------|------------|-----------|
| measured | 4,802 | wet-lab MPRA ribosome load (delta) | wet-lab measured |
| proxy | 473,228 (sampled 2,000) | CNN-50mer ensemble prediction (delta) | predicted/internal proxy |
| unlabeled | 839,288 | null (excluded) | N/A |

**Splits (measured):** train=3,411, val=573, test=486, ood=332
**Split contract:** family-disjoint (source_id never crosses roles), mmseqs-based clustering.

## Task 1: Prediction Model Architectures

Four structurally different models, all trained with 3-fold cross-fitting:

| Model | Input | Output | Architecture |
|-------|-------|--------|-------------|
| Absolute | candidate_sequence features | predicted absolute value | 22→64→1 ReLU MLP |
| Difference | (candidate-source) diff + edit features | predicted delta | 34→64→1 ReLU MLP |
| Siamese | encoder(source), encoder(candidate) | delta via shared encoder | 22→64 shared, diff→1 |
| Edit-conditioned | source features + edit tokens | predicted delta | 34→64→1 ReLU MLP |

**Features (22 per sequence):** length, GC, GC-first-10, AUG-position, nt-frequencies (4), top-dinucleotide-frequencies (6), max-polyN-runs (4), Shannon-entropy, GC-windows (3).

**Edit features (12):** edit-count, position-mean/std/min/max, region-distribution, GC-change-fraction, top-transition-frequencies (4).

## Task 2: Local-Delta Metrics

Primary metrics (in order of importance):
1. **Sign accuracy** — fraction of correct beneficial/harmful direction
2. **Beneficial-edit precision** — of predicted-beneficial, fraction truly beneficial
3. **Top-k enrichment** — enrichment of true-beneficial in top-k predicted

Secondary metrics: delta Spearman, delta Pearson, pairwise ranking AUC, calibration error, false-positive beneficial rate, source-normalized RMSE, MAE, RMSE.

## Task 3: Cross-Fitted Oracle Ensemble

- **Cross-fitting:** 3-fold, family-group-respecting (source_id-level)
- **Ensemble:** mean of 4 model OOF predictions
- **Uncertainty:** std across 4 models
- **Disagreement:** max − min across 4 models
- **Independence:** models are structurally different (different input representations), not same-architecture-different-seed

## Task 4: Region Sensitivity

6 controlled perturbation types × 5 sensitivity checks:

| Perturbation | Description |
|-------------|-------------|
| five_utr_single_sub | Random nt substitution in 5'UTR |
| start_proximal_cds | Synonymous codon in codons 1-5 |
| middle_cds | Synonymous codon in middle third |
| late_cds | Synonymous codon in last third |
| joint_5utr_cds | Combined 5'UTR + CDS edit |
| matched_random | Same-position random substitution (control) |

**Checks:** position sensitivity, GC-only risk, length-only risk, CDS start-vs-late, source awareness.

## Task 5: Optimization Headroom

5 search strategies on 20 frozen test sources:

| Method | Description |
|--------|-------------|
| exact_one_edit | Full enumeration of single-nt substitutions |
| greedy | Step-wise best single edit (max 3) |
| beam_search | Beam width=3, max depth=2 |
| simulated_annealing | 100 iterations, temp=1.0, cooling=0.95 |
| mcts | UCB1 tree search, 50 simulations, max depth=3 |

## GO/NO-GO Gate

### GO (all required)
- Sign accuracy > 0.55 on independent test
- Top-k enrichment > 1.0
- Strong search finds positive candidates in >20% of sources
- Gain not explained by single GC/length heuristic
- At least one region has stable headroom

### PARTIAL
- Local effect predictable but only in one region, OR
- Headroom exists but concentrated in few cargo, OR
- Training Oracle and independent Oracle agreement limited

### NO-GO
- Sign accuracy near random (<0.52)
- No top-k enrichment
- Strong search cannot exceed source
- Independent Oracle direction frequently reverses

## Configuration

```
n_folds: 3
n_epochs: 20
hidden_dim: 64
max_proxy: 2000
n_sensitivity_samples: 50
n_headroom_sources: 20
seed: 42
```

## Deliverables

- `docs/p3_02_delta_oracle_protocol.md` (this file)
- `docs/p3_02_delta_oracle_results.json` (machine-readable results)
- `docs/p3_02_headroom_analysis.md` (headroom analysis)
- `docs/p3_02_region_decision.md` (region sensitivity + decision)
- `checkpoints/p3_delta_oracles/` (4 model checkpoints, fold-0)
- `results/p3_02/test_predictions.npz` (raw test predictions)
