# P1-04: Cross-Fitted Predictor Ensemble Design

**Task ID**: P1-04
**Goal**: Train cross-fitted predictor ensemble (≥2 architectures, ≥2 training datasets) for TE/MRL/stability prediction, with held-out + OOD metrics, uncertainty, calibration, and applicability domain.
**Status**: DESIGN PHASE (acquisition complete; implementation pending)
**Date**: 2026-07-19
**Depends on**: P1-02A (Sample 2019 + Cao 2021), P1-03 (Saluki + CodonBERT)
**Blocks**: P1-05 (independent final oracle), P1-06 (reward-hacking audit)

---

## 1. Motivation

The current `eval/oracle.py` is a deterministic feature-regressor with optional tiny 5'UTR CNN. It is intentionally weak (single architecture, single data source, no cross-fitting) and was flagged in `docs/next_steps_sota_roadmap.md` §1.1 as:
> "ranker teacher 与 evaluation oracle circular；同 checkpoint 兼作生成 field 和 rank score"

P1-04 builds a **proper cross-fitted predictor ensemble** with:
- Multiple architectures (CNN + Transformer)
- Multiple training datasets (Sample 2019 + Cao 2021 + Saluki + CodonBERT)
- K-fold cross-fitting for unbiased held-out predictions
- Frozen test lock via deterministic SHA-256 split
- Per-fold uncertainty, calibration, and applicability domain reporting

This ensemble serves as the **training-time teacher** (oracle #1 of three). The **independent final oracle** (P1-05, oracle #3) will use a separate frozen artifact with hidden test labels.

---

## 2. Requirements (from roadmap §13.1)

| Requirement | How P1-04 satisfies it |
|-------------|------------------------|
| ≥2 architectures | (A) CNN-50mer (Optimus100K-style); (B) Transformer-UTR (UTR-LM-style) |
| ≥2 training data | (1) Sample 2019 random_50mer + designed; (2) Cao 2021 GENCODE 5'UTR + Ribo-seq TE; (3) Saluki half-life; (4) CodonBERT stability |
| Cross-fitted | k=5 folds on train split; held-out predictions stacked for downstream teacher signal |
| Held-out metrics | Per-fold Pearson r, Spearman ρ, R², MAE on val + test |
| OOD metrics | Cross-dataset evaluation: train on Sample 2019 → eval on Cao 2021; train on Saluki human → eval on Saluki mouse; etc. |
| Uncertainty | Ensemble disagreement (std across folds) + deep ensembles (M=3 seeds per arch) |
| Calibration | Expected Calibration Error (ECE) on test; reliability diagram |
| Applicability domain | Distance-to-training-distribution via k-NN in embedding space (Gini-style abstention threshold) |

---

## 3. Data Inputs

### 3.1 TE / MRL Labels

| Dataset | Records | Label | Sequence | Source |
|---------|---------|-------|----------|--------|
| Sample 2019 random_50mer | 2,188,769 | `rl` (mean ribosome load, normalized) | 50-nt 5'UTR | P1-02A ✓ |
| Sample 2019 designed | 100,017 | `rl` | 50-nt human-SNV 5'UTR | P1-02A ✓ |
| Sample 2019 variable_length | 106,530 | `rl` | 25–100 nt 5'UTR | P1-02A ✓ |
| Cao 2021 GENCODE 5'UTR | 96,015 | (no direct RL; used for context/embedding) | 5'UTR + 15bp CDS | P1-02A ✓ |
| Cao 2021 Ribo-seq TE | 165,946 | TE (ribosome / mRNA RPKM) | (no UTR seq; join by gene ID) | P1-02A ✓ |

### 3.2 Stability / Half-Life Labels

| Dataset | Records | Label | Sequence | Source |
|---------|---------|-------|----------|--------|
| Saluki human | 12,968 | PCA-1 of half-life (39 datasets) | 12,288-nt spliced mRNA | P1-03 ✓ |
| Saluki mouse | 13,738 | PCA-1 of half-life (27 datasets) | 12,288-nt spliced mRNA | P1-03 ✓ |
| CodonBERT mRNA_Stability | 65,356 | stability value (codon-comp) | CDS-length sequences | P1-03 ✓ |
| CodonBERT CoV_Vaccine | 2,400 | degradation rate at multiple positions | 130-nt RNA fragments | P1-03 ✓ |

**Total**: 2,660,439 records across 4 datasets (subject to GSM3130440 completion; +280k once it lands).

---

## 4. Architecture

### 4.1 Architecture A: CNN-50mer (Optimus100K-style)

**Purpose**: Strong, fast-training baseline. Replicates Sample 2019 NBT architecture.

```
Input: 50-nt one-hot (N, 4, 50)
├── Conv1d(4 → 64, kernel=8, stride=1, pad=same) + BN + ReLU
├── Conv1d(64 → 64, kernel=8, stride=1, pad=same) + BN + ReLU
├── Conv1d(64 → 128, kernel=8, stride=1, pad=same) + BN + ReLU
├── MaxPool1d(kernel=2, stride=2)
├── Conv1d(128 → 128, kernel=8, stride=1, pad=same) + BN + ReLU
├── Conv1d(128 → 256, kernel=8, stride=1, pad=same) + BN + ReLU
├── MaxPool1d(kernel=2, stride=2)
├── Flatten
├── Linear(256 * 12 → 256) + ReLU + Dropout(0.3)
├── Linear(256 → 1)  # scalar regression head
```

**Training**: AdamW, lr=1e-3, weight_decay=1e-5, batch=512, 30 epochs, MSE loss on log1p(rl) standardized.

### 4.2 Architecture B: Transformer-UTR (UTR-LM-style)

**Purpose**: Captures long-range dependencies; supports variable-length inputs.

```
Input: variable-length UTR (N, L_max=100) tokenized as BPE tokens
├── Token embedding (vocab=64, dim=128)
├── Positional embedding (RoPE, max_len=128)
├── 4 × TransformerEncoderLayer(d_model=128, nhead=8, dim_ff=512, dropout=0.1)
├── Mean-pool over valid tokens
├── Linear(128 → 1) + ReLU gate (uncertainty head: Linear(128 → 2) → μ, log σ²)
```

**Training**: AdamW, lr=3e-4, warmup 1k steps, cosine decay, batch=256, 50 epochs, NLL loss for Gaussian head.

### 4.3 Architecture C (optional): CNN-Transformer Hybrid

For full-length mRNA (Saluki, 12k nt): use 1D conv stack to downsample → transformer on compressed tokens.

**Status**: deferred unless A/B underfit Saluki.

---

## 5. Cross-Fitting Protocol

### 5.1 Split Contract

All datasets use the same deterministic SHA-256 hashing as in P1-02A/P1-03 manifests:
- Hash key: uppercase sequence (Sample 2019, CodonBERT) or uppercase gene name (Saluki)
- Bucket boundaries: `<0.8` train, `<0.9` val, else test
- **Test split is FROZEN** — no model in P1-04 is allowed to train on it
- Val split is used for early stopping and hyperparameter selection
- Train split is used for k-fold cross-fitting

### 5.2 k-Fold Cross-Fitting (k=5)

For each (architecture, dataset) pair:
1. Shuffle train split with fixed seed (seed=42)
2. Partition into k=5 contiguous folds F_0, ..., F_4
3. For each fold i ∈ {0..4}:
   - Train on union(F_j for j ≠ i)
   - Predict on F_i (held-out in-fold predictions) → store
   - Predict on val + test → store
4. Concatenate in-fold predictions → unbiased teacher signal for the entire train split
5. Concatenate val/test predictions across folds → mean prediction + std (uncertainty)

### 5.3 Deep Ensemble (M=3 seeds)

For each (architecture, dataset, fold) triple:
- Train M=3 random initialization seeds
- Mean prediction = ensemble mean
- Epistemic uncertainty = std across (fold × seed) = std across 15 predictions

---

## 6. Evaluation Protocol

### 6.1 Per-Fold Metrics

For each (architecture, dataset, fold) on **val + test** splits:
- Pearson r
- Spearman ρ
- R² (coefficient of determination)
- MAE (mean absolute error)
- RMSE

### 6.2 OOD Metrics

Cross-dataset evaluation matrix (train → eval):

| Train ↓ \ Eval → | Sample 2019 | Cao 2021 TE | Saluki human | Saluki mouse | CodonBERT stab |
|-------------------|-------------|-------------|--------------|--------------|-----------------|
| Sample 2019 | (in-dist) | OOD-1 | OOD-2 | OOD-3 | OOD-4 |
| Cao 2021 | OOD-5 | (in-dist) | OOD-6 | OOD-7 | OOD-8 |
| Saluki human | OOD-9 | OOD-10 | (in-dist) | cross-species | OOD-11 |
| Saluki mouse | OOD-12 | OOD-13 | cross-species | (in-dist) | OOD-14 |
| CodonBERT stab | OOD-15 | OOD-16 | OOD-17 | OOD-18 | (in-dist) |

Each OOD cell reports: Pearson r, Spearman ρ, R², MAE, fraction abstained (via applicability domain).

### 6.3 Uncertainty Metrics

- **Ensemble disagreement**: std(μ_i for i in folds×seeds) per sample
- **Calibration**: Expected Calibration Error (ECE) with 10 equal-width bins on |μ - y| vs predicted σ
- **Reliability diagram**: per-bin empirical error vs predicted σ

### 6.4 Applicability Domain

For each test sample x:
1. Compute embedding z = encoder(x) at penultimate layer
2. Find k=10 nearest neighbors in training embeddings (cosine distance)
3. mean_dist = mean(d(z, NN_i) for i in 1..10)
4. Abstain if mean_dist > τ (τ = 95th percentile of train→train mean_dist)

Report:
- Fraction abstained at τ
- Pearson r on non-abstained subset
- Coverage-accuracy curve (vary τ from 0 to ∞)

---

## 7. Deliverables

### 7.1 Code

```
src/mrna_editflow/predictors/
├── __init__.py
├── base.py                    # PredictorBase interface
├── cnn_50mer.py               # Architecture A
├── transformer_utr.py         # Architecture B
├── crossfit.py                # k-fold cross-fitting harness
├── ensemble.py                # Deep ensemble aggregation
├── uncertainty.py             # ECE, reliability, applicability domain
└── data_loaders.py            # Unified loader for Sample 2019 / Cao 2021 / Saluki / CodonBERT
```

### 7.2 Checkpoints

```
ckpts/p1_04_predictors/
├── cnn_50mer__sample2019__fold0_seed42.pt
├── cnn_50mer__sample2019__fold1_seed42.pt
├── ...
├── transformer_utr__sample2019__fold0_seed42.pt
├── ...
├── cnn_50mer__saluki_human__fold0_seed42.pt
├── ...
└── ensemble_manifest.json     # All checkpoints + metrics + SHAs
```

### 7.3 Artifacts

- `docs/p1_04_predictor_ensemble_report.md` — full report with tables and figures
- `docs/p1_04_predictor_ensemble_metrics.json` — machine-readable metrics (per-arch, per-dataset, per-fold, OOD, calibration)
- `data/processed/p1_04_teacher_predictions.parquet` — held-out predictions for downstream RL teacher

### 7.4 Test Contract

- `tests/test_p1_04_predictor_ensemble.py`:
  - Predictor determinism (same input → same output given fixed seed)
  - Cross-fitting: train/val/test split disjoint
  - Test lock: no checkpoint trained on test split (verified via data provenance manifest)
  - Uncertainty calibration: ECE within ±0.05 of bootstrap CI
  - Applicability domain: abstained fraction within [0.01, 0.10]

---

## 8. Implementation Plan

### Phase 1: Scaffolding (Day 1)
- Define `PredictorBase` interface (`fit`, `predict`, `predict_with_uncertainty`, `save`, `load`)
- Implement unified data loader (handles 4 datasets, applies deterministic split)
- Implement k-fold cross-fitting harness
- Write tests for split disjointness and determinism

### Phase 2: Architecture A (Day 2)
- Implement CNN-50mer
- Train on Sample 2019 (3M records, k=5 folds × M=3 seeds = 15 checkpoints)
- Train on Saluki human (13k records, k=5 folds × M=3 seeds)
- Record per-fold + OOD metrics

### Phase 3: Architecture B (Day 3-4)
- Implement Transformer-UTR
- Train on Sample 2019 + Saluki human
- Train on CodonBERT mRNA_Stability (65k records)
- Record per-fold + OOD metrics

### Phase 4: Aggregation & Report (Day 5)
- Build ensemble aggregation (mean + std across 30+ checkpoints)
- Compute ECE, reliability diagrams, applicability domain
- Write `docs/p1_04_predictor_ensemble_report.md`
- Generate `data/processed/p1_04_teacher_predictions.parquet`

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Saluki sequences too long (12k nt) for Transformer | Use conv-downsample front-end (Architecture C) or restrict to CDS+UTR windows |
| Sample 2019 + Saluki OOD performance near zero | Acceptable; report as negative result; teacher signal still useful in-distribution |
| Compute budget exceeds 5-day plan | Prioritize Architecture A on Sample 2019 (most data, fastest training); defer B/C if needed |
| Cross-fitting leakage via sequence similarity | Audit with `leakage_audit.py`; if violations, switch to family-disjoint split (already implemented in P0 v2) |
| Calibration poorly defined for regression | Use quantile-based calibration (Pinball loss) instead of classification ECE if needed |

---

## 10. Acceptance Criteria

- [ ] ≥2 architectures trained on ≥2 datasets each
- [ ] k=5 cross-fitting completed for all (arch, dataset) pairs
- [ ] Held-out test metrics (Pearson r, R², MAE) reported per (arch, dataset)
- [ ] OOD metrics reported for all 25 cross-dataset cells
- [ ] Uncertainty (ensemble std) computed per sample
- [ ] Calibration (ECE or quantile) reported per (arch, dataset)
- [ ] Applicability domain (k-NN distance) computed per sample
- [ ] All checkpoints + manifest written to `ckpts/p1_04_predictors/`
- [ ] `docs/p1_04_predictor_ensemble_report.md` written
- [ ] `data/processed/p1_04_teacher_predictions.parquet` written
- [ ] Tests pass: split disjointness, determinism, test lock

---

## 11. Three-Oracle Contract (preview for P1-05)

P1-04 produces **oracle #1 (training teacher)** — cross-fitted ensemble with publicly visible test labels.

P1-05 will produce:
- **Oracle #2 (selection oracle)** — used for hyperparameter / model selection; same data as #1 but different random seeds and frozen weights; hidden val labels
- **Oracle #3 (final independent oracle)** — frozen artifact with hidden test labels, different architecture (e.g., gradient-boosted trees on hand-engineered features), different data source (e.g., Leppek 2022 PERSIST-Seq or external MPRA not used in #1/#2), weights locked before any RL training

This three-oracle separation ensures:
- No circularity between training and evaluation
- Final reported metrics come from a model that has never seen test labels in any form
- Selection bias cannot inflate reported metrics

See `docs/independent_oracle_design_v1.md` (to be written in P1-05).
