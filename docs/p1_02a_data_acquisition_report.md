# P1-02A: Real Multi-Cell MPRA / MRL Data Acquisition Report

**Task ID**: P1-02A
**Goal**: Acquire real functional labels (translation efficiency, ribosome load) from multi-cell MPRA / Ribo-seq datasets to support cross-fitted predictor ensemble (P1-04) and independent oracle (P1-05).
**Status**: PARTIALLY COMPLETE (Sample 2019 9/10 OK + Cao 2021 complete + Khoroshkin 2024 BLOCKED)
**Date**: 2026-07-19
**Operator**: trae autonomous agent

---

## 1. Acquired Datasets

### 1.1 Sample 2019 NBT — `sample2019_mpra` (GSE114002)

**Citation**: Sample PJ, Wang B, Seelig G. *Human 5' UTR design and variant effect prediction from a massively parallel translation assay.* Nat Biotechnol. 2019;37(7):807-811. PMID:31267113.

**Layout**: `data/raw/sample2019_mpra/`

**Coverage**:
- 10 GSM samples (HEK293T) covering 3 RNA chemistries × 2 CDS × 2 replicates + designed + variable-length libraries
- 9/10 files downloaded successfully and validated via `gunzip -t`
- 1 file (`GSM3130440_egfp_m1pseudo_2.csv.gz`, 77 MB expected) still downloading at ~10 KB/s due to NCBI FTP connection throttling

**Aggregate stats (9/10 files)**:
- Total records: 2,395,316
- Total bytes: 354 MB (337.6 MiB)
- Per chemistry: unmodified=1,365,991; pseudouridine=695,348; 1-methylpseudouridine=333,977 (one m1pseudo replicate still pending)
- Per library kind: random_50mer=2,188,769; designed_human_snv=100,017; random_variable_length=106,530

**Schema (random_50mer)**: `['', 'utr', '00'..'09', 'total', 'r_00'..'r_09', 'rl']` — per-bin DNA/RNA counts plus normalized ribosome load (`rl`).

**Deterministic split (SHA-256 of uppercased UTR, 80/10/10)**:

| GSM | Records | Train | Val | Test |
|-----|---------|-------|-----|------|
| GSM3130435 (unmod_1) | 326,033 | 260,525 | 32,896 | 32,612 |
| GSM3130436 (unmod_2) | 351,575 | 281,106 | 35,375 | 35,094 |
| GSM3130437 (pseudo_1) | 333,977 | 267,145 | 33,523 | 33,309 |
| GSM3130438 (pseudo_2) | 361,371 | 289,051 | 36,303 | 36,017 |
| GSM3130439 (m1pseudo_1) | 333,977 | 267,145 | 33,523 | 33,309 |
| GSM3130440 (m1pseudo_2) | PENDING | — | — | — |
| GSM3130441 (mcherry_1) | 269,114 | 215,386 | 26,813 | 26,915 |
| GSM3130442 (mcherry_2) | 212,722 | 170,050 | 21,302 | 21,370 |
| GSM3130443 (designed) | 100,017 | 79,957 | 10,151 | 9,909 |
| GSM4084997 (25-100nt) | 106,530 | 85,169 | 10,669 | 10,692 |

**License**: NCBI GEO public data; research use permitted; cite PMID:31267113.

**Manifest artifacts**:
- `data/raw/sample2019_mpra/manifest.json` (26.9 KB, per-file SHA-256 + counts + metadata)
- `data/raw/sample2019_mpra/sample2019_summary.json` (aggregated stats)

**Blocker**: GSM3130440 download speed (~10 KB/s) — NCBI FTP throttling on this specific file. Background wget (PID 3881202 on server) running with `--tries=10 --retry-connrefused`. ETA ~1h40m from 21:56 server time.

---

### 1.2 Cao 2021 Nat Commun — `cao2021_5utr`

**Citation**: Cao X, Zhang S, Bhatt P, et al. *Exploiting 5' UTRs for single-vector multiplexed gene expression in vivo.* Nat Commun. 2021;12(1):4194. PMID:34230498. DOI:10.1038/s41467-021-24436-7.

**Layout**: `data/raw/cao2021_5utr/`

**Coverage**:
- 10/10 files downloaded successfully (48.5 MB total)
- Acquired via GitHub tarball (`https://github.com/zzz2010/5UTR_Optimizer/archive/refs/heads/master.tar.gz`) — single 25-second download avoided per-file HTTPS throttling
- 4 file kinds: `ribo_seq_te_features` (3 files, 29.4 MB), `utr_sequence_fasta` (1 file, 16 MB), `filtered_endogenous_5utr` (1 file, 1.8 MB), `mpra_screen_result` (5 files, 1.3 MB)
- 3 cell types: HEK293T (3 files), PC3 (3 files), Muscle (2 files), multiple (2 files)

**Key files**:
- `gencode_v17_5utr_15bpcds.fa` — 96,015 GENCODE v17 5'UTR + 15bp CDS records (77,025 train / 9,393 val / 9,597 test via deterministic hashing)
- `final_endogenous.txt` — 8,414 filtered endogenous 5'UTRs (col 0 = sequence, col 1 = metadata)
- `TE_sorted.HEK293T.Andrev2015.with_annot.txt` — 79,087 HEK293T Ribo-seq TE feature rows (Ribo-seq from Andreev 2015, GSE55195)
- `TE_sorted.pc3.with_annot.txt` — 79,795 PC3 TE feature rows (Ribo-seq from Gunisova 2016, GSE35469)
- `TE_sorted.Muscle.with_annot.txt` — 7,464 Muscle TE feature rows (Ribo-seq from Martin 2016, GSE56148)
- 5 evaluation FASTA files under `eva_5utrseq/` — top1000 / bottom500 5'UTRs from MPRA screens per cell type

**License**: GitHub `zzz2010/5UTR_Optimizer` (research use). Raw FASTQ at GSE176581. Ribo-seq accessions in manifest.

**Manifest artifacts**:
- `data/raw/cao2021_5utr/manifest.json` (per-file SHA-256 + counts + metadata)
- `data/raw/cao2021_5utr/cao2021_summary.json` (aggregated stats by kind/cell type)

**Note**: TE feature TSVs do not contain a UTR sequence column — only Ensembl/Entrez/gene IDs. Split stats are 0/0/0 for these files. The actual UTR sequences are in `final_endogenous.txt` and `gencode_v17_5utr_15bpcds.fa` (which have proper split stats).

---

### 1.3 Khoroshkin 2024 bioRxiv PARADE — `khoroshkin2024_parade` (BLOCKED)

**Citation**: Khoroshkin M, Zinkevich A, Aristova E, et al. *A generative framework for enhanced cell-type specificity in rationally designed mRNAs.* bioRxiv [Preprint]. 2024 Dec 31:2024.12.31.630783. PMID:39803435. PMCID:PMC11722239.

**Layout**: `data/raw/khoroshkin2024_parade/manifest.json` (placeholder only)

**Status**: **BLOCKED — dataset not yet publicly deposited**

**Expected data** (per preprint):
- 30,000 50-nt 5'UTR fragments from 2,068 transcripts with cell-type variable TE
- 30,000 3'UTR fragments from same 2,068 transcripts
- 15,800 de novo-designed UTR validation sequences
- Stability MPRA — 3'UTR reporter RNA-to-DNA ratio measurements
- 6 cell types: Jurkat, Nalm-6, SW-480, PA-1, MDA-MB-231, HepG2

**Block reason**: As of 2026-07-19, no public GEO/Zenodo/GitHub accession found. Searched:
- bioRxiv preprint page — no Data Availability section exposing accession
- PMC full text — supplementary materials not exposed via standard WebFetch
- Author/institutional websites — no public mirror located
- PubMed linked data — no GEO/Zenodo entries

**License (preprint)**: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/).

**Unblock path**:
1. Email corresponding authors: `hani.goodarzi@arcinstitute.org`, `ivan.kulakovskiy@gmail.com`, `khorms21@gmail.com`
2. Request MPRA counts table (UTR × cell type × replicate) + stability MPRA + designed validation set
3. If peer-reviewed version published before response, re-search for accession in final journal version
4. Fallback: substitute with MRL2023 (Leppek 2022 Nat Biotechnol, GSE151209) for 3'UTR stability MPRA coverage

---

## 2. Coverage Matrix vs P1-02A Requirements

| Requirement | Sample 2019 | Cao 2021 | Khoroshkin 2024 | Status |
|-------------|-------------|----------|------------------|--------|
| 5'UTR MPRA (multi-cell) | HEK293T only | HEK293T + PC3 + Muscle | 6 cell types (BLOCKED) | 3/3 cell types covered |
| 3'UTR MPRA | — | — | 30k 3'UTRs (BLOCKED) | Not covered (defer to P1-03) |
| RNA chemistry variants | unmod + Ψ + m1Ψ | unmod only | unmod only | 3 chemistries covered |
| Variable-length UTRs | 25–100 nt | 15 bp CDS context | 50 nt fixed | Covered |
| Ribo-seq TE features | RL (MPRA-derived) | 3 cell types (HEK293T/PC3/Muscle) | — | 3 cell types covered |
| Designed validation set | 100k SNV library | 5× 1k/500 evaluation FASTAs | 15.8k (BLOCKED) | Covered by 2/3 datasets |
| Deterministic split | 80/10/10 SHA-256 | 80/10/10 SHA-256 | — | All acquired files split |

**Aggregate**: 2.4M records + 96k GENCODE reference 5'UTRs + 8.4k filtered endogenous. Sufficient for P1-04 cross-fitted ensemble training pending GSM3130440 completion.

---

## 3. P1-04 Readiness Assessment

**Ready now** (9/10 Sample 2019 + all Cao 2021):
- Sample 2019 random_50mer: 2.19M UTRs × RL labels (3 chemistries, 2 CDS)
- Cao 2021 reference: 96k GENCODE 5'UTR + 8.4k endogenous filtered + 165k Ribo-seq TE features across 3 cell types
- Designed libraries: 100k human-SNV + 5× evaluation FASTAs

**Pending**:
- GSM3130440 (m1pseudo replicate 2) — 77 MB, ~1h40m ETA from 21:56
- Khoroshkin 2024 6-cell-type MPRA — BLOCKED, no public accession

**Recommendation**: Begin P1-04 ensemble architecture design and feature engineering immediately. Defer full training run until GSM3130440 completes; in the interim, validate pipeline on 9/10 Sample 2019 files (the missing m1pseudo replicate is only 1/10 of the random_50mer library and does not block pipeline development).

---

## 4. Hard Constraint Compliance

Per `project_memory.md` — "All new datasets must include source URL, SHA-256, license, counts, split, and metadata documentation":

| Dataset | source_url | sha256 | license | counts | split | metadata |
|---------|------------|--------|---------|--------|-------|----------|
| sample2019_mpra (9/10) | ✓ per GSM | ✓ per file | ✓ | ✓ | ✓ 80/10/10 | ✓ cell/chemistry/CDS/replicate |
| cao2021_5utr (10/10) | ✓ per file | ✓ per file | ✓ | ✓ | ✓ FASTA only | ✓ cell/kind |
| khoroshkin2024_parade | ✓ preprint DOI | n/a (no data) | ✓ CC BY-NC-ND 4.0 | expected only | n/a | ✓ 6 cell types |

---

## 5. Files Delivered

```
data/raw/sample2019_mpra/
├── GSM3130435_egfp_unmod_1.csv.gz         63.7 MB   ✓
├── GSM3130436_egfp_unmod_2.csv.gz         32.0 MB   ✓
├── GSM3130437_egfp_pseudo_1.csv.gz        61.8 MB   ✓
├── GSM3130438_egfp_pseudo_2.csv.gz        60.3 MB   ✓
├── GSM3130439_egfp_m1pseudo_1.csv.gz      59.2 MB   ✓
├── GSM3130440_egfp_m1pseudo_2.csv.gz      PENDING   (download in progress)
├── GSM3130441_mcherry_1.csv.gz            24.7 MB   ✓
├── GSM3130442_mcherry_2.csv.gz            19.5 MB   ✓
├── GSM3130443_designed_library.csv.gz     17.3 MB   ✓
├── GSM4084997_varying_length_25to100.csv.gz 15.4 MB ✓
├── manifest.json                          26.9 KB
└── sample2019_summary.json                411 B

data/raw/cao2021_5utr/                     48.5 MB total (10/10 ✓)
├── TE_sorted.HEK293T.Andrev2015.with_annot.txt
├── TE_sorted.pc3.with_annot.txt
├── TE_sorted.Muscle.with_annot.txt
├── final_endogenous.txt
├── gencode_v17_5utr_15bpcds.fa
├── eva_5utrseq/  (5 evaluation FASTAs)
├── manifest.json
└── cao2021_summary.json

data/raw/khoroshkin2024_parade/
└── manifest.json  (placeholder, BLOCKED)

scripts/
├── acquire_sample2019_mpra.py             (reusable; idempotent)
├── acquire_cao2021_5utr.py                (reusable; idempotent)
└── acquire_khoroshkin2024_placeholder.py  (reusable; emits placeholder manifest)

docs/p1_02a_data_acquisition_report.md     (this file)
```

---

## 6. Next Steps

1. **Wait for GSM3130440 to finish** (~1h40m ETA), then re-run `scripts/acquire_sample2019_mpra.py` to refresh manifest with full 10/10 coverage. The script is idempotent — skips already-downloaded files, only processes the newly completed one.
2. **Email Khoroshkin 2024 authors** to request data access (manual step, defer to next session).
3. **Begin P1-03** (half-life / stability data acquisition) in parallel — does not depend on P1-02A completion. Target datasets: Leppek 2022 (GSE151209, 3'UTR stability MPRA), Mauger 2019 (mRNA half-life), Rabani 2017 (RNA stability MPRA).
4. **Begin P1-04 architecture design** (cross-fitted predictor ensemble) using the 9/10 Sample 2019 + Cao 2021 data already available. Use k-fold cross-fitting (k=5) with held-out test lock matching the deterministic SHA-256 split.
