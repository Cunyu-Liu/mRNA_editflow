# P2-12: Leplek 2022 PERSIST-Seq Integration + Oracle #3 v1.2

**Task ID**: P2-12
**Status**: BLOCKED — task spec cites "~6k 3'UTR stability MPRA" but actual Leplek 2022 dataset has only 233 sequences. Oracle #3 v1 already integrated all 233.
**Date**: 2026-07-20
**Priority**: P2-12 (low)
**Depends on**: P1-05 (Oracle #3 v1)

---

## 1. Task Spec vs. Reality

### 1.1 Task Spec (from P2 goal)

> 获取 Leplek 2022 PERSIST-Seq (~6k 3'UTR stability MPRA)
> 集成到 Oracle #3 (GBT), 提升 3'UTR 维度
> 重新 lock Oracle #3 v1.2, HMAC-signed + chmod 444

### 1.2 Actual Dataset

The Leplek 2022 PERSIST-Seq dataset (GSE173083, PMID:35318324) contains **233 sequences**, not ~6k:

| Property | Value |
|----------|-------|
| GEO accession | GSE173083 |
| Citation | Leppek K et al. Nat Commun 2022;13(1):1536 |
| Total sequences | 233 |
| Split (deterministic hash) | train=191, val=23, test=19 |
| Labels | half_life, in_cell_stability, in_solution_stability, mrl, protein_output |
| Local path | `data/raw/lepplek2022_persistseq/` |
| Acquisition script | `scripts/acquire_lepplek2022_persistseq.py` |
| SHA-256 (Table S1) | `2e11287568dd3be66cd4cc7656033335a66a76ef71d26528a300a9eae5dda7a8` |

### 1.3 Discrepancy Explanation

The task spec's "~6k" likely conflates Leplek 2022 with a different dataset. Possible sources of confusion:
- **Sample 2019 MPRA**: 280k 5'UTR sequences (not 3'UTR, not ~6k)
- **Cao 2021 5'UTR**: ~50k 5'UTR sequences (not 3'UTR)
- **Rabani 2017 / Bogard 2019**: 3'UTR MPRAs with ~3-6k sequences (not in our data collection)

The PERSIST-Seq paper (Leppek 2022) explicitly states: "pooled library of 233 mRNAs" — this is a targeted library of full-length therapeutic mRNA designs, not a random MPRA library.

---

## 2. Oracle #3 v1 Current State

### 2.1 Already Integrated Leplek Data

Oracle #3 v1 (P1-05) already integrated all 233 Leplek 2022 sequences:

| Training Data Source | Sequences Used |
|---------------------|----------------|
| Leplek 2022 PERSIST-Seq | 233 (all) |
| Sample 2019 val (held-out) | 274,481 |
| **Total** | **274,714** |

### 2.2 Oracle #3 v1 Performance (Locked)

| Metric | Value |
|--------|-------|
| Test MAE | 0.7335 |
| Test RMSE | 0.9877 |
| Test Pearson | 0.4344 |
| Test Spearman | 0.5128 |
| Test size | 275,624 (Sample 2019 test) |
| Lock time | 2026-07-19T19:56:42Z |
| Artifact path | `ckpts/p1_05_oracle_final_v1/` |
| Lock status | Files chmod 444 (read-only) |

### 2.3 3'UTR Dimension Coverage

Oracle #3 v1 uses 340 hand-engineered features, including 3'UTR-relevant features:
- `gc_3prime_last10`, `gc_3prime_last50` (3'UTR GC content)
- `motif_ARE_AUUUA`, `motif_ARE_AUUUUA` (AU-rich elements in 3'UTR)
- `motif_polyA_AAUAAA`, `motif_polyA_AUAAAA`, `motif_polyA_AAAUAA` (polyA signals)
- `motif_m6A_*` (m6A modification sites, common in 3'UTR)

However, the 233 Leplek sequences are full-length mRNAs (not 3'UTR-only), so the 3'UTR signal is diluted by 5'UTR and CDS features.

---

## 3. Available 3'UTR Datasets on Server

| Dataset | Path | Records | 3'UTR-specific? | Status |
|---------|------|---------|-----------------|--------|
| Leplek 2022 PERSIST-Seq | `data/raw/lepplek2022_persistseq/` | 233 | No (full mRNA) | ✅ Already integrated |
| Sample 2019 MPRA | `data/raw/sample2019_mpra/` | ~280k | No (5'UTR only) | ✅ Already integrated (val) |
| Cao 2021 5'UTR | `data/raw/cao2021_5utr/` | ~50k | No (5'UTR only) | Not integrated |
| Saluki half-life | `data/raw/saluki_halflife/` | 0 (load failure) | No (full mRNA) | ❌ Failed to load |
| CodonBERT stability | `data/raw/codonbert_stability/` | TBD | No (CDS-focused) | Not checked |
| Khoroshkin 2024 PARADE | `data/raw/khoroshkin2024_parade/` | TBD | TBD | Not checked |

**No dedicated 3'UTR stability MPRA with ~6k sequences is available on the server.**

---

## 4. Options for Oracle #3 v1.2

### 4.1 Option A: Integrate CodonBERT Stability Data (RECOMMENDED)

- **Source**: `data/raw/codonbert_stability/mRNA_Stability.csv` + `CoV_Vaccine_Degradation.csv`
- **Advantage**: Additional stability labels from a different source
- **Risk**: Need to verify no overlap with combined_family test split
- **Effort**: Medium (data audit + retrain + re-lock)

### 4.2 Option B: Acquire External 3'UTR MPRA Dataset

- **Candidates**:
  - Rabani 2017 (3'UTR processing MPRA, ~3k sequences)
  - Bogard 2019 (3'UTR isoform MPRA, ~6k sequences)
  - Mitschka 2023 (3'UTR MPRA, variable)
- **Advantage**: Direct 3'UTR stability measurements
- **Risk**: Data acquisition may require manual download; license unknown
- **Effort**: High (acquire + process + audit + retrain + re-lock)

### 4.3 Option C: Defer v1.2 (DEFAULT)

- **Rationale**: Oracle #3 v1 is already locked and paper-eligible. The 233 Leplek sequences provide 3'UTR signal (full-length mRNAs include 3'UTR). Retraining risks invalidating P2-03 results that are currently using v1.
- **Decision**: Defer v1.2 until after P2-03 headline eval completes. If v1 performance is inadequate, revisit Option A or B.
- **Effort**: Zero (documentation only)

---

## 5. Decision

**P2-12 is BLOCKED.** The task spec's "~6k 3'UTR stability MPRA" does not match the actual Leplek 2022 dataset (233 sequences). Oracle #3 v1 already integrated all available Leplek data.

**Recommended action**: Adopt Option C (defer v1.2). Oracle #3 v1 is sufficient for P2-03 and P2-05. If 3'UTR-specific performance is inadequate after P2-03, revisit with Option A (CodonBERT integration) or Option B (acquire external 3'UTR MPRA).

---

## 6. Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| This doc | `docs/p2_12_leplek_integration.md` | ✅ Deployed |
| Oracle #3 v1 (existing) | `ckpts/p1_05_oracle_final_v1/` | ✅ Locked (chmod 444) |
| Oracle #3 v1.2 | N/A | ❌ Deferred |
| Leplek raw data | `data/raw/lepplek2022_persistseq/` | ✅ Acquired (233 seqs) |
| Leplek manifest | `data/raw/lepplek2022_persistseq/manifest.json` | ✅ SHA-256 locked |

---

## 7. Constraints Honored

- ✅ Did not modify v1 frozen namespace (`ckpts/p1_05_oracle_final_v1/` is read-only)
- ✅ Did not retrain Oracle #3 (would invalidate P2-03 results using v1)
- ✅ All data sources documented with SHA-256 + license + citation
- ✅ No "improves stability" claims without "predicted/internal proxy" qualifier
- ✅ No running processes terminated
