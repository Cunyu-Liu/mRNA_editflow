# D0-05 Current Candidates Audit Report

> **Task:** D0-05 (Phase D0)
> **Contract:** `utr_editflow_contract_v2` (FROZEN)
> **Audit date:** 2026-08-01
> **Git commit at start:** `1d91ed1`
> **Decision log:** `docs/decision_log.md` → `DEC-UTR-EF-V2-20260801-D0-05-CANDIDATES-AUDIT`
> **Structured role table:** `docs/data/d0_05_dataset_role_table.yaml`

---

## 1. Audit Methodology

This audit reviews all 9 candidate datasets in `data_registry/intervention_candidates.yaml` and assigns each a v2-compliant role (D_A–D_E) and evidence grade (E0–E6) with explicit allowed/forbidden claims. The audit corrects three categories of v1-era issues:

1. **Grade scheme migration**: v1 `A1/A2/B1/B2` → v2 `E0-E6`
2. **Role scheme migration**: v1 descriptive roles → v2 contract-defined data roles (D_A–D_E)
3. **Scope violation correction**: GSE246381 `sealed` → `historically_exposed (E4)`; GSE207584/GSE173083 CDS/full-length → D_A (out-of-scope)

**Audit principles:**
- No dataset is promoted by size alone (acceptance criterion 2)
- Every dataset has non-empty allowed/forbidden claims (acceptance criterion 1)
- CDS/full-length datasets are capped at D_A regardless of quality (v2 §4.2)
- GSE246381 is fixed at E4 regardless of data volume (v2 §2)
- Raw reads (ENCSR854RUF) cannot provide causal labels (D_A, not D_C)

---

## 2. Per-Dataset Audit

### 2.1 GSE114002 — Sample et al. 2019 (5'UTR)

| Field | Value |
|---|---|
| Region | 5'UTR |
| Endpoint | mean_ribosome_loading |
| Variants | 3577 natural variants |
| v1 role/grade | primary_benchmark_component / A1 |
| **v2 role** | **D_C** (source-matched measured interventions) |
| **v2 grade** | **E2** (retrospective measured) |
| Allowed claims | 5'UTR edit-effect on MRL (E2); generative edit-trajectory learning (E1); MRL prediction as auxiliary critic (E1) |
| Forbidden claims | therapeutic efficacy (E6 not in scope); MRL = protein output; prospective improvement from retrospective; "the first" claims |
| Size note | 3577 variants — sufficient for D_C; NOT promoted to D_D (moderate, not dense tiling) |

**Role change reason:** v2 replaces v1 A/B grade scheme with E0-E6. Paired WT-mutant 5'UTR MPRA with MRL endpoint is the canonical source-matched measured intervention → D_C at E2.

### 2.2 GSE149487 — PLUMAGE (5'UTR)

| Field | Value |
|---|---|
| Region | 5'UTR |
| Endpoint | transcript_abundance; translation_efficiency |
| Variants | 545 somatic mutations / 914 synthetic 5'UTR (WT+mutant) |
| v1 role/grade | primary_benchmark_component / A1 |
| **v2 role** | **D_C** |
| **v2 grade** | **E2** |
| Allowed claims | 5'UTR edit-effect on TE/transcript abundance (E2); somatic mutation edit-trajectory learning (E1); TE prediction as auxiliary critic (E1) |
| Forbidden claims | therapeutic efficacy (E6); TE = protein output; prospective improvement; prostate cancer therapeutic claim |
| Size note | 545 mutations — small but source-matched; NOT promoted to D_D (low count, not dense) |

### 2.3 GSE246381 — NDD (5'UTR) — HISTORICALLY EXPOSED

| Field | Value |
|---|---|
| Region | 5'UTR |
| Endpoint | transcript_abundance; 80S_monosome_polysome |
| Variants | 997 NDD family 5'UTR mutations (6 replicates) |
| v1 role/grade | **sealed_external_test** / A1 |
| **v2 role** | **D_E** (historically-exposed external) |
| **v2 grade** | **E4** (historically exposed) |
| Allowed claims | cross-study 5'UTR transfer evaluation (E4); transfer from GSE114002/GSE149487 (E1); NDD mutation spectrum analysis (E4) |
| Forbidden claims | **"sealed/untouched/never-seen"** (v2 §2); using labels for new training/hyperparameter selection (v2 §2); therapeutic efficacy (E6); untouched independent validation |
| Size note | 997 mutations — role is D_E regardless of size; historical exposure is fixed by v2 §2, not by data volume |

**Critical fix:** v1 labeled this dataset `sealed_external_test`. v2 §2 explicitly states GSE246381 is `historically_exposed (E4)` because labels were read by the v1 predictor-first workflow during benchmark construction. The words "sealed", "untouched", "never-seen external test" are FORBIDDEN in the active tree.

### 2.4 GSE145046 — Dense Library (5'UTR) — DENSE LANDSCAPE

| Field | Value |
|---|---|
| Region | 5'UTR |
| Endpoint | ribosome_free_monosome_polysome; fluorescence; in_cell/in_vitro half_life |
| Variants | >1,000,000 designed 10-nt randomized variants |
| v1 role/grade | large_scale_pretraining / A2 |
| **v2 role** | **D_D** (dense measured landscape) |
| **v2 grade** | **E2** |
| Allowed claims | dense 5'UTR landscape pretraining for generative prior (E2); multi-step edit-trajectory learning (E1); sequence-function landscape modeling (E2) |
| Forbidden claims | dense library implies natural source-matched edit-effect (scaffold bias); therapeutic efficacy (E6); **promoted to D_C by size alone** (FORBIDDEN) |
| Size note | >1M variants — largest dataset; explicitly NOT promoted from D_D to D_C. Size qualifies for dense landscape (D_D) but does NOT change role to source-matched intervention (D_C). |

**Key anti-promotion check:** This is the primary "no dataset auto-promoted by size" test case. Despite having >1M variants (1000x more than GSE149487), it stays at D_D because the variants are randomized on a fixed scaffold, not source-matched natural interventions.

### 2.5 GSE217518 (3'UTR)

| Field | Value |
|---|---|
| Region | 3'UTR |
| Endpoint | decay_constant; half_life |
| Variants | 6555 disease-relevant UTR variants (WT+mutant) |
| v1 role/grade | cross_region_benchmark / A1 |
| **v2 role** | **D_C** (primary supervised for 3'UTR) |
| **v2 grade** | **E2** |
| Allowed claims | 3'UTR edit-effect on decay/half-life (E2); 3'UTR edit-trajectory learning (E1); stability prediction as auxiliary critic (E1) |
| Forbidden claims | therapeutic stability (E6); half-life = efficacy; prospective improvement |
| Size note | 6555 variants — sufficient for D_C; NOT promoted to D_D (moderate density) |

**Role change reason:** In v2, 3'UTR is a primary scope region (not a "cross-region" add-on). Renamed from `cross_region_benchmark` to D_C.

### 2.6 GSE200304 (3'UTR)

| Field | Value |
|---|---|
| Region | 3'UTR |
| Endpoint | translation_efficiency; steady_state_rna; mrna_stability |
| Variants | 6892 patient mutations (WT/mutant 201-nt pairs) |
| v1 role/grade | cross_region_benchmark / A1 |
| **v2 role** | **D_C** |
| **v2 grade** | **E2** |
| Allowed claims | 3'UTR edit-effect on TE/RNA/stability (E2); patient mutation edit-trajectory learning (E1); stability/TE prediction as auxiliary critic (E1) |
| Forbidden claims | therapeutic efficacy (E6); TE/stability = protein output; prospective improvement; patient disease therapeutic claim |
| Size note | 6892 mutations — sufficient for D_C; NOT promoted to D_D |

### 2.7 ENCSR854RUF — MPRAu (3'UTR) — DUAL ROLE

| Field | Value |
|---|---|
| Region | 3'UTR |
| Endpoint | allele_specific_rna_abundance |
| Variants | 12173 3'UTR variants (6 cell lines) + 62 raw fastq.gz (~357GB) |
| v1 role/grade | cross_region_benchmark / A1 |
| **v2 role (processed)** | **D_C** (primary supervised for 3'UTR) |
| **v2 role (raw reads)** | **D_A** (observational pretraining only) |
| **v2 grade (processed)** | **E2** |
| **v2 grade (raw reads)** | **E1** (internal computational) |
| Allowed claims | 3'UTR edit-effect from processed table (E2); observational pretraining from raw reads (E1); 6-cell-line context conditioning (E1) |
| Forbidden claims | **extracting wt-mutant causal labels from raw reads** (FORBIDDEN — D_A only); raw reads provide multi-step real trajectories (they do NOT); raw reads provide prospective improvement (E1 only); therapeutic efficacy (E6) |
| Size note | 12173 variants (processed) + ~357GB raw reads. The large raw-read volume does NOT promote raw reads from D_A to D_C. File size does not override role definition. |

**Critical dual-role clarification:** The ENCODE raw reads download was corrected from "deferred" to "COMPLETE" in `DEC-UTR-EF-V2-20260801-ENCODE-STATUS-CORRECTION`, but this only fixes the download record. It does NOT promote raw reads to a higher evidence grade. Raw reads remain D_A (observational pretraining, E1) — they cannot provide wt-mutant causal labels, multi-step real trajectories, or prospective improvement evidence.

### 2.8 GSE207584 — iCodon (CDS) — OUT OF SCOPE

| Field | Value |
|---|---|
| Region | **CDS** (out of scope, v2 §4.2) |
| Endpoint | mrna_decay_2h_5h_8h |
| Variants | 1395 synthesized synonymous CDS |
| v1 role/grade | **codon_benchmark** / B1 |
| **v2 role** | **D_A** (observational pretraining only) |
| **v2 grade** | **E2** (retrospective measured, used as D_A) |
| Allowed claims | observational CDS pretraining for foundation model (E2, as D_A); codon-level representation learning (E1) |
| Forbidden claims | **CDS synonymous edit-effect benchmark** (FORBIDDEN, v2 §4.2); **CDS grammar validation** (FORBIDDEN); **CDS edit-trajectory generation** (FORBIDDEN); promoted to primary by variant count (FORBIDDEN); CDS therapeutic efficacy (E6 + out of scope) |
| Size note | 1395 CDS — moderate size; explicitly NOT promoted to D_C/D_D. CDS is out of scope, so dataset is capped at D_A regardless of count or quality. |

**Critical scope fix:** v1 treated CDS as a benchmark component (`codon_benchmark`). v2 §4.2 explicitly excludes CDS synonymous generation from scope. The dataset is downgraded to D_A (observational pretraining only).

### 2.9 GSE173083 — PERSIST-seq (full-length) — OUT OF SCOPE

| Field | Value |
|---|---|
| Region | **full_length** (out of scope, v2 §4.2) |
| Endpoint | ribosome_load; in_cell_stability; in_solution_stability |
| Variants | 233 full-length mRNA constructs (24 CDS designs) |
| v1 role/grade | **full_length_transfer_benchmark** / B2 |
| **v2 role** | **D_A** (observational pretraining only) |
| **v2 grade** | **E2** (retrospective measured, used as D_A) |
| Allowed claims | observational full-length mRNA pretraining for foundation model (E2, as D_A); full-length representation learning (E1) |
| Forbidden claims | **full-length mRNA edit-effect benchmark** (FORBIDDEN, v2 §4.2); **full-length joint optimization** (FORBIDDEN); **cross-region full-transcript synergy** (FORBIDDEN); promoted to primary by quality (FORBIDDEN); full-length therapeutic efficacy (E6 + out of scope) |
| Size note | 233 constructs — small; explicitly NOT promoted. Full-length is out of scope, so dataset is capped at D_A. |

**Critical scope fix:** v1 treated full-length as a transfer benchmark. v2 §4.2 explicitly excludes full-length mRNA joint optimization and cross-region full-transcript synergy. Downgraded to D_A.

---

## 3. Acceptance Criteria Verification

| Criterion | Check | Result |
|---|---|---|
| each dataset has allowed/forbidden claim | All 9 datasets have non-empty `allowed_claims` (≥2 items) and `forbidden_claims` (≥3 items) | **PASS** |
| no dataset auto-promoted by size | GSE145046 (>1M) stays D_D not D_C; ENCSR854RUF raw (~357GB) stays D_A not D_C; GSE207584 (1395 CDS) stays D_A not D_C; GSE173083 (233 FL) stays D_A not D_C | **PASS** |

---

## 4. Summary Statistics

| v2 role | Count | Datasets |
|---|---|---|
| D_A (observational) | 3 | GSE207584, GSE173083, ENCSR854RUF(raw) |
| D_C (primary supervised) | 6 | GSE114002, GSE149487, GSE217518, GSE200304, ENCSR854RUF(processed) |
| D_D (dense landscape) | 1 | GSE145046 |
| D_E (historically exposed) | 1 | GSE246381 |

| v2 evidence grade | Count | Datasets |
|---|---|---|
| E1 (internal computational) | 1 | ENCSR854RUF(raw reads) |
| E2 (retrospective measured) | 9 | all datasets (as observational or supervised) |
| E4 (historically exposed) | 1 | GSE246381 |
| E6 (prospective experimental) | 0 | not in scope |

---

## 5. Scope Violations Fixed

| Dataset | v1 violation | v2 fix | Contract clause |
|---|---|---|---|
| GSE246381 | `sealed_external_test` | D_E (E4, historically_exposed) | v2 §2 |
| GSE207584 | `codon_benchmark` (CDS in scope) | D_A (CDS out of scope) | v2 §4.2 |
| GSE173083 | `full_length_transfer_benchmark` (full-length in scope) | D_A (full-length out of scope) | v2 §4.2 |

---

## 6. Size Promotion Prevented

| Dataset | Size | Tempting promotion | Actual role | Reason |
|---|---|---|---|---|
| GSE145046 | >1,000,000 variants | D_C (primary supervised) | D_D (dense landscape) | Randomized scaffold, not source-matched natural interventions |
| ENCSR854RUF raw | ~357 GB (62 fastq.gz) | D_C (supervised) | D_A (observational) | File size does not override role; raw reads have no causal labels |
| GSE207584 | 1395 CDS | D_C (supervised) | D_A (observational) | CDS is out of scope; scope overrides size |
| GSE173083 | 233 full-length | D_C (supervised) | D_A (observational) | Full-length is out of scope; scope overrides size |

---

## 7. Next Steps

D0-05 completion unblocks:
- **D1-01** (canonical records: download/extract/clean/build_source_candidate/build_edit_scripts per dataset)
- **D1-02** (exposure ledger: data/data_exposure_ledger.jsonl)
- **B0-01** (generative UTR benchmark construction — closed_measured_pool / heldout_generative / open_legal_generation)

The role table in `docs/data/d0_05_dataset_role_table.yaml` is the authoritative reference for all downstream tasks. Any downstream task that uses a dataset MUST check its v2 role and allowed/forbidden claims before proceeding.
