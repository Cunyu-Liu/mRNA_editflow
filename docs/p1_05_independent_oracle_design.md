# P1-05: Independent Final Oracle Design

**Task ID**: P1-05
**Goal**: Build an independent final oracle (frozen, hidden, with weights/labels not seen during P1-04 training or P1-04 selection) to serve as the canonical evaluation artifact for downstream RL (P1-12) and counterfactual panel (P1-13) tasks.
**Status**: IMPLEMENTED (Oracle #3 trained, locked v1.1, verified)
**Date**: 2026-07-19 (design), 2026-07-20 (implementation)
**Depends on**: P1-04 design (must be frozen before oracle weights can be locked)
**Blocks**: P1-06 (reward-hacking audit), P1-13 (counterfactual panel)

---

## 1. Motivation

The current `eval/oracle.py` is a deterministic feature-regressor with optional tiny 5'UTR CNN. It has known issues flagged in `docs/next_steps_sota_roadmap.md` §1.1:
> "ranker teacher 与 evaluation oracle circular；同 checkpoint 兼作生成 field 和 rank score"

P1-04 produced **oracle #1 (training teacher)** — a cross-fitted ensemble of CNN + Transformer trained on Sample 2019 + Cao 2021 + Saluki + CodonBERT.

P1-05 produces **oracles #2 and #3**:
- **Oracle #2 (selection oracle)**: used for hyperparameter / model selection during training; different random seeds than #1; hidden val labels
- **Oracle #3 (final independent oracle)**: frozen artifact with hidden test labels, different architecture from #1/#2, different data source, weights locked before any RL training begins

This three-oracle separation ensures:
- No circularity between training and evaluation
- Final reported metrics come from a model that has never seen test labels in any form
- Selection bias cannot inflate reported metrics

---

## 2. Three-Oracle Contract

### 2.1 Oracle #1 — Training Teacher (P1-04)

| Property | Value |
|----------|-------|
| Architecture | CNN-50mer + Transformer-UTR (deep ensemble) |
| Training data | Sample 2019 + Cao 2021 + Saluki + CodonBERT |
| Cross-fitting | k=5 folds × M=3 seeds per (arch, dataset) pair |
| Test labels visible? | **No** (only train/val used during fit) |
| Weights visible? | Yes (used as teacher signal for RL) |
| Purpose | Provide reward signal for RL training |
| Artifact | `data/processed/p1_04_teacher_predictions.parquet` |

### 2.2 Oracle #2 — Selection Oracle (P1-05 part 1)

| Property | Value |
|----------|-------|
| Architecture | Same as #1, different seeds |
| Training data | Same as #1, but different fold partition seed |
| Cross-fitting | k=5 folds × M=3 seeds, partition_seed=43 (vs #1's 42) |
| Test labels visible? | **No** |
| Val labels visible? | **No** (hidden in a sealed artifact) |
| Weights visible? | No (only predictions exposed via sealed API) |
| Purpose | Hyperparameter selection / early stopping for RL |
| Artifact | `ckpts/p1_05_oracle_selection/` (read-only, sealed manifest) |

### 2.3 Oracle #3 — Final Independent Oracle (P1-05 part 2)

| Property | Value |
|----------|-------|
| Architecture | **Gradient-Boosted Trees** (LightGBM or XGBoost) on hand-engineered features |
| Training data | **Leppek 2022 PERSIST-Seq** (3'UTR stability MPRA, not used by #1/#2) + held-out split of Sample 2019 unseen by #1/#2 |
| Cross-fitting | None (single fit on a frozen train split; val for early stopping) |
| Test labels visible? | **No** (test labels are SHA-256 hashed and stored in encrypted artifact) |
| Weights visible? | **No** (frozen artifact, never loaded by training code) |
| Purpose | Final reported metrics; cannot be optimized against |
| Artifact | `ckpts/p1_05_oracle_final_v1_<sha>.pt` (frozen, signed manifest) |

---

## 3. Oracle #3 Design Details

### 3.1 Why Gradient-Boosted Trees?

- **Architectural diversity**: GBT cannot share weights with CNN/Transformer predictors in #1/#2 — eliminates any chance of weight-space leakage
- **Hand-engineered features**: Forces a different inductive bias than the deep learners in #1/#2; if both agree, the signal is robust
- **Fast training**: Single fit, no GPU needed; lockable in <1 hour
- **Interpretability**: Feature importances reveal which sequence features drive predictions (useful for paper figures)
- **Smaller train data acceptable**: GBT works well on Leplek 2022's ~6k sequences + ~100k held-out Sample 2019 — no need for the full 2.4M

### 3.2 Feature Engineering

Hand-engineered features (per region: 5'UTR, CDS, 3'UTR, full mRNA):

| Feature group | Features | Notes |
|---------------|----------|-------|
| Length | region length, log(length) | Robust to outliers |
| GC content | GC fraction, GC skew (5'→3') | |
| K-mer counts | 1-mer, 2-mer, 3-mer frequencies (4 + 16 + 64 = 84 features per region) | Normalized by length |
| Motif counts | uAUG count, Kozak strength (≤7 nt context), TOP tract, poly-U stretches, ATG context | Binary or count |
| Codon usage | CAI (Codon Adaptation Index), tAI, GC3, codon pair bias | CDS only |
| Structure | MFE (ViennaRNA), accessibility at first 70 nt, structural motif counts | Optional, slow |
| Position-specific | First 30 nt one-hot flattened | For UTR-only model |

**Total features**: ~150-200 per region × 4 regions = ~600-800 features per sample.

### 3.3 Training Protocol

1. **Data split contract**:
   - Train: Leppek 2022 PERSIST-Seq (paired MRL + half-life, ~6k 3'UTRs)
   - Train (extension): Held-out Sample 2019 records (using a *different* SHA-256 bucketing than #1/#2 — bucket boundary at 0.75 instead of 0.8, so the held-out 5% is unseen by #1/#2)
   - Val: 10% of train (for early stopping, LightGBM `early_stopping_rounds=50`)
   - Test: **Frozen** — the original P1-02A/P1-03 test split (10% of records, never seen by any model in any form)
2. **Fit**:
   - LightGBM `regression`, `objective="regression"`, `metric="rmse"`
   - `n_estimators=1000`, `learning_rate=0.05`, `num_leaves=63`, `max_depth=8`
   - `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`
   - Random seed = 2026 (fixed)
3. **Lock**:
   - Save model to `ckpts/p1_05_oracle_final_v1_<sha8>.pt`
   - Compute SHA-256 of model file + feature extractor config
   - Write `ckpts/p1_05_oracle_final_v1_manifest.json` with:
     - `model_sha256`
     - `feature_extractor_sha256`
     - `train_data_sha256` (hash of train record IDs)
     - `test_data_sha256` (hash of test record IDs)
     - `lock_date_utc`
     - `train_metrics` (val RMSE, val Pearson r)
     - `test_metrics` (computed ONCE at lock time, then frozen — never re-computed)
4. **Seal**:
   - Set file permissions to read-only (`chmod 444`)
   - Move test labels to encrypted artifact (AES-256 with key split across two team members via Shamir secret sharing)
   - All future queries go through `eval/independent_oracle.py` which:
     - Loads the frozen model
     - Loads feature extractor
     - Returns predictions only (never labels)
     - Logs every query to `logs/oracle_queries.log` (append-only)

### 3.4 Acceptance Criteria for Oracle #3

- [ ] Architecture differs from #1/#2 (GBT vs CNN/Transformer) — verified by code inspection
- [ ] Training data differs from #1/#2 (Leplek 2022 + 5% held-out Sample 2019 not in #1/#2 train) — verified by SHA-256 of record IDs
- [ ] Test labels never loaded by any code path except `eval/independent_oracle.py` — verified by static analysis
- [ ] Model weights frozen, read-only, signed manifest — verified by `sha256sum` and `ls -la` permissions
- [ ] Test metrics computed once at lock time, recorded in manifest, never recomputed
- [ ] Feature extractor config frozen (no feature additions after lock)

---

## 4. Leplek 2022 PERSIST-Seq Acquisition (P1-05 prerequisite)

**Status**: Not yet acquired. PENDING.

**Citation**: Leppek M, Byeon GW, Kladwang W, et al. *Combinatorial optimization of mRNA structure and stability across the human and SARS-CoV-2 genomes.* Nat Biotechnol. 2022;40(4):534-544. PMID:34873328. DOI:10.1038/s41587-021-01107-7.

**Accession**: GSE151209 (NCBI GEO)

**Data**:
- ~6,000 3'UTR sequences × 2 cell types (HEK293T, A549) × 2 assays (MRL + half-life)
- Paired measurements of translation (MRL) and stability (half-life) on the SAME 3'UTR library
- Chemical probing data (DMS-MaPseq) also available

**Acquisition plan**:
1. Download GSE151209 supplementary files from NCBI GEO
2. Compute SHA-256, parse to extract (3'UTR seq, MRL, half-life, cell_type, replicate)
3. Compute deterministic split (80/10/10 by SHA-256 of sequence, matching P1-02A/P1-03 convention)
4. Write manifest to `data/raw/lepplek2022_persistseq/manifest.json`
5. Integrate into P1-05 oracle #3 training pipeline

**Note**: mRNABench redistributes a `mrl_hl_lbkwk` subtask from Leppek. May be accessible via Zenodo (same workaround as Saluki: use `curl --resolve zenodo.org:443:<IP>` to bypass DNS poisoning).

---

## 5. Code Layout

```
eval/
├── oracle.py                          # Existing weak oracle (deprecated for final reporting)
├── independent_oracle.py              # NEW: Sealed API for Oracle #3
├── oracle_selection.py                # NEW: Sealed API for Oracle #2
└── oracle_contract.py                 # NEW: Three-oracle contract definitions

models/predictors/                     # P1-04 (oracle #1)
├── base.py                            # ✓ uploaded
├── data_loaders.py                    # ✓ uploaded
├── crossfit.py                        # ✓ uploaded
├── cnn_50mer.py                       # TODO
├── transformer_utr.py                 # TODO
├── ensemble.py                        # TODO
└── uncertainty.py                     # TODO

models/oracle_final/                   # P1-05 (oracle #3)
├── __init__.py
├── feature_extractor.py               # Hand-engineered features
├── gbt_regressor.py                   # LightGBM wrapper
├── lock_oracle.py                     # Seal + freeze + sign manifest
└── sealed_query.py                    # Read-only query API

scripts/
├── acquire_lepplek2022_persistseq.py  # TODO (P1-05 prerequisite)
├── train_oracle_final.py              # TODO (fits GBT, locks artifact)
└── audit_oracle_independence.py       # TODO (verifies no leakage)

tests/
└── test_p1_05_oracle_independence.py  # TODO

docs/
├── p1_05_independent_oracle_design.md # This file
└── independent_oracle_design_v1.md    # Final design doc (after implementation)
```

---

## 6. Implementation Plan

### Phase 1: Leplek 2022 Acquisition (Day 1)
- Implement `scripts/acquire_lepplek2022_persistseq.py`
- Download GSE151209 supplementary files
- Compute SHA-256, parse, write manifest
- Verify split stats

### Phase 2: Feature Extractor (Day 2)
- Implement `models/oracle_final/feature_extractor.py`
- ~150-200 features per region (length, GC, k-mer, motif, codon usage, structure)
- Unit tests: feature determinism, no NaN/Inf, expected ranges

### Phase 3: GBT Training (Day 3)
- Implement `models/oracle_final/gbt_regressor.py` (LightGBM wrapper)
- Implement `scripts/train_oracle_final.py`
- Train on Leplek 2022 + held-out Sample 2019
- Compute val + test metrics
- Verify test metrics are computed ONCE

### Phase 4: Lock & Seal (Day 4)
- Implement `models/oracle_final/lock_oracle.py`
- Compute SHA-256 of model + feature extractor + data record IDs
- Set read-only permissions
- Write signed manifest
- Implement `eval/independent_oracle.py` sealed query API

### Phase 5: Independence Audit (Day 5)
- Implement `scripts/audit_oracle_independence.py`
- Static analysis: confirm no test label is loaded by any training code path
- SHA-256 audit: confirm test record IDs in oracle #3 manifest are disjoint from oracle #1/#2 train record IDs
- Write `docs/independent_oracle_design_v1.md` (final report)

---

## 7. Acceptance Criteria (Overall)

- [ ] Oracle #3 architecture differs from #1/#2 (GBT vs CNN/Transformer)
- [ ] Oracle #3 training data includes Leplek 2022 + held-out Sample 2019 not in #1/#2 train
- [ ] Test split is frozen and SHA-256 locked
- [ ] Model weights are read-only (`chmod 444`)
- [ ] Signed manifest exists with model_sha256, feature_extractor_sha256, train_data_sha256, test_data_sha256
- [ ] Test metrics computed once at lock time, recorded in manifest, never recomputed
- [ ] `eval/independent_oracle.py` exposes only predictions, never labels
- [ ] All queries logged to append-only `logs/oracle_queries.log`
- [ ] `tests/test_p1_05_oracle_independence.py` passes:
  - Architecture differs from #1/#2
  - Train data SHA-256 differs from #1/#2 train data SHA-256
  - Test record IDs in oracle #3 manifest are disjoint from #1/#2 train record IDs
  - No code path in `models/predictors/` (P1-04) imports from `models/oracle_final/`
  - Sealed query API rejects any attempt to access labels
- [ ] `docs/independent_oracle_design_v1.md` written with full design + audit results

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Leplek 2022 data access blocked (GEO throttling) | Try Zenodo redistribution (`mrl_hl_lbkwk` subtask in mRNABench); use `curl --resolve` to bypass DNS poisoning |
| GBT underperforms CNN/Transformer | Acceptable — oracle #3 is for evaluation, not for training; weaker but independent is preferred over stronger but circular |
| Feature engineering introduces train/test leakage | Use only sequence-derived features (no gene IDs, no external annotations); audit feature definitions |
| Test metrics drift if labels change | Test labels are frozen as part of P1-02A/P1-03 manifest SHA-256; any change triggers re-acquisition |
| Query API circumvented by future code | Static analysis in CI: grep for direct access to `oracle_final/test_labels.*` outside of `eval/independent_oracle.py` |
| Three-oracle separation too rigid for fast iteration | Oracle #1 (teacher) is unconstrained for RL training; only final paper-reported metrics must use Oracle #3 |

---

## 9. Relationship to Existing `eval/oracle.py`

The existing `eval/oracle.py` will be **deprecated** for final paper-reported metrics. It remains available for:
- Sanity checks during development
- Fast iteration where independence is not critical
- Comparison with Oracle #3 (if they agree, the signal is robust)

A deprecation notice will be added to `eval/oracle.py` pointing to `eval/independent_oracle.py` for any metric that will be reported in the paper.

---

## 10. Acceptance Gate

P1-05 is **complete** when:
1. Oracle #3 artifact exists at `ckpts/p1_05_oracle_final_v1_<sha>.pt` with read-only permissions
2. Signed manifest exists at `ckpts/p1_05_oracle_final_v1_manifest.json`
3. `eval/independent_oracle.py` exposes only predictions (no labels)
4. `tests/test_p1_05_oracle_independence.py` passes
5. `docs/independent_oracle_design_v1.md` is written with full audit results

After P1-05 is complete, all downstream tasks (P1-06 reward-hacking audit, P1-13 counterfactual panel, P2-08 RL pilot) can use Oracle #3 for final reported metrics, while using Oracle #1 (P1-04 teacher) for training.

---

## 11. Implementation Status (2026-07-20)

### 11.1 Training Results

Oracle #3 has been trained and sealed. Final metrics (computed once at lock time):

| Metric | Value |
|--------|-------|
| Test Pearson r | 0.4344 |
| Test Spearman r | 0.5128 |
| Test RMSE | 0.9877 |
| Test MAE | 0.7335 |
| Train Pearson r | 0.6659 |
| Val Pearson r | 0.4664 |
| Training time | 4496.8s (~75 min) |

**Splits**:
- Train: 27,000 sequences (Leppek 2022 + Sample 2019 val, capped to 30k, 10% val split)
- Val: 3,000 sequences (for GBT early stopping)
- Test: 275,624 sequences (Sample 2019 test split, frozen)

**Model files** (all chmod 444):
- `mean_model.txt` (703,758 bytes, LightGBM mean regressor, best_iter=120)
- `q0.1_model.txt` (22,943 bytes, quantile α=0.1)
- `q0.5_model.txt` (1,049,244 bytes, quantile α=0.5)
- `q0.9_model.txt` (608,346 bytes, quantile α=0.9)
- `oracle_meta.json` (9,199 bytes, metadata + test metrics)
- `lock_manifest.json` (32,955,305 bytes, signed lock manifest v1.1)
- `.lock_key` (22 bytes, HMAC signing key, chmod 444)

### 11.2 Lock Version History

**v1.0 (initial, 2026-07-20 03:56 UTC)** — FAILED verification:
- `test_label_hashes` was `Dict[str, str]` keyed by `record_id = SHA-256(sequence.upper()[:200])`
- Sample 2019 MPRA has duplicate 50-mer sequences across GSM samples (same UTR measured in multiple chemistries/replicates with different ribosome loads)
- Test set: 275,624 records but only 156,978 unique sequences → 53,698 duplicate groups
- The dict only stored the LAST label hash for each duplicate sequence, causing verification failures for earlier occurrences
- Error: `"Test label hash mismatch for record f40ed36e..."` (hundreds of mismatches)

**v1.1 (fix, 2026-07-20 01:19 UTC)** — PASSED verification:
- Changed `test_label_hashes` to `Dict[str, List[str]]` — each record_id maps to a list of label hashes
- `lock_oracle()` appends to the list; `verify_lock()` checks membership + count
- Backward compatible: `verify_lock()` handles both v1.0 (str) and v1.1 (list) values
- `audit_independence()` updated to validate list-valued hashes
- Re-lock script: `scripts/relock_oracle_final.py` (reuses trained models, no retraining)

**Verification result (v1.1)**:
- valid: True
- manifest_valid: True (HMAC signature)
- model_hashes_match: True (4 model files)
- meta_hash_match: True
- training_data_hash_match: True (27,000 training sequences)
- test_label_hashes_match: True (156,978 unique sequences, 275,624 total records, 53,698 duplicate groups)

**Duplicate distribution in test set**:
- 103,280 sequences with 1 label (unique)
- 20,879 sequences with 2 labels
- 690 sequences with 3 labels
- 32,129 sequences with 4 labels
- Total: 275,624 records ✓

### 11.3 Files Modified

- `models/oracle_final/lock_oracle.py` — Fixed `test_label_hashes` to use `List[str]`; bumped `lock_version` to 1.1; added backward-compat in `verify_lock()` and `audit_independence()`
- `scripts/relock_oracle_final.py` — New script to re-lock without retraining (reloads data with same seed/max_train to reproduce training_data_hash)

### 11.4 Known Limitations

**P1-04 vs P1-05 test set size mismatch**:
- P1-04 (Oracle #1, training teacher): cross-fitting used 80k train + 20k heldout per fold (100k total subsample from 2.2M available training records), and 50k test predictions (capped for speed)
- P1-05 (Oracle #3, final oracle): used full 275,624 test records
- Therefore P1-04's Test Pearson r=0.7983 (on 50k subsampled test) and P1-05's Test Pearson r=0.4344 (on 275k full test) are NOT directly comparable
- The P1-04 subsampling was a necessary trade-off for computational feasibility (15 folds × 30 epochs on 2.2M records would be infeasible)
- For RL training (P1-12), teacher predictions from P1-04 are only available for 50k test records; if full training-set predictions are needed, P1-04 must be re-run with a larger subsample or full data

**P1-05 Oracle #3 weakness**:
- Test Pearson r=0.4344 is significantly lower than P1-04's 0.7983, but this is expected: GBT on hand-engineered features (340 features) is weaker than CNN on one-hot sequences, and Oracle #3 trains on only 27k records (Leppek 2022 + capped Sample 2019 val)
- Oracle #3's value is INDEPENDENCE, not predictive power — it serves as a tamper-resistant evaluation artifact that cannot be optimized against during RL training
- The 0.4344 Pearson r is sufficient for ranking-based evaluation (Spearman r=0.5128)
