# Full-Length Joint mRNA MPRA Dataset Design v1

**Status**: Design + statistical plan FROZEN. Wet-lab execution NOT in scope (design only).
**Date**: 2026-07-19
**Author**: trae agent (autonomous execution, P1-02B)
**Barrier**: 壁垒 2 (cross-region synergy mechanism discovery) — data foundation
**SHA-256 (this doc)**: (recorded on upload)

---

## 0. Executive Summary

This document specifies the design and pre-registered statistical analysis plan for a **self-built full-length joint mRNA MPRA (massively parallel reporter assay)** dataset. The dataset is the **key data barrier** for 壁垒 2 (cross-region synergy): it is the first MPRA to jointly vary all three mRNA regions (5'UTR, CDS, 3'UTR) in a full-length therapeutic mRNA context, enabling direct measurement of cross-region synergy.

**Design summary**:
- **5 arms × 1000–5000 sequences per arm** (5,000–25,000 total)
- **Arms**: wild-type / MEF-edited / GEMORNA-de-novo / LinearDesign-CDS-only / random-legal-edits
- **Cargo**: 5–10 therapeutic proteins (ACTB, HLA-A, eGFP, luciferase, SARS-CoV-2 spike, factor IX, etc.)
- **Readouts**: MRL (ribosome load), TE (translational efficiency), half-life (stability), protein expression (flow cytometry / ELISA)
- **Cells**: HEK293T (baseline), HepG2 (liver cargo), primary T cells (therapeutic context)
- **Statistical plan**: pre-registered, with family-level inference, OOD stress, and synergy decomposition

**Why this is a 6–12 month competitor replication barrier**:
1. **No public full-length joint mRNA MPRA exists**. All public MPRAs (Sample 2019, NatureComm 2024) are 5'UTR-only with random or truncated sequences.
2. **Joint variation of 3 regions requires full-length mRNA synthesis** (expensive, ~$50–200/sequence for 1–4 kb mRNAs), which no academic lab has done at MPRA scale (>1000 sequences).
3. **The synergy signal is only detectable with joint variation** — single-region MPRAs cannot measure cross-region interactions by construction.
4. **MEF's RL policy is the only method that can generate principled joint edits** (vs random or single-region baselines), giving MEF a unique algorithmic advantage on this dataset.

---

## 1. Scientific Question

**Primary question**: Does joint editing of multiple mRNA regions (5'UTR + CDS + 3'UTR) produce **synergistic** improvements in translational efficiency (TE), stability (half-life), and protein expression — i.e., improvements that exceed the sum of single-region edits?

**Secondary questions**:
1. Which region pairs (5'UTR × CDS, CDS × 3'UTR, 5'UTR × 3'UTR) contribute most to synergy?
2. Does MEF's RL policy discover synergistic edits that single-region baselines miss?
3. Are synergistic effects consistent across cell types (HEK293T vs HepG2 vs primary T cells)?
4. Are synergistic effects consistent across cargos (cytoplasmic vs secreted vs membrane)?

**Hypothesis** (pre-registered): Joint editing produces positive synergy (syn_sum > 0) with effect size Cohen's d > 0.5, detectable at α = 0.001 with N = 1000 sequences per arm.

---

## 2. Experimental Design

### 2.1 Five Arms

| Arm | Description | Edit Region | Method | N sequences |
|---|---|---|---|---|
| **A1: wild-type** | Natural human mRNA sequences from GENCODE/RefSeq | none | canonical record (long-view, P1-11) | 1000–5000 |
| **A2: MEF-edited** | MEF RL policy edits on wild-type starting points | 5'UTR + CDS + 3'UTR (joint) | trained MEF policy (CTO + synergy RL) | 1000–5000 |
| **A3: GEMORNA-de-novo** | GEMORNA generates full mRNA from scratch | de novo | GEMORNA (Science 2025) | 1000–5000 |
| **A4: LinearDesign-CDS-only** | LinearDesign optimizes CDS only, UTRs from wild-type | CDS only | LinearDesign (Nature 2023) | 1000–5000 |
| **A5: random-legal-edits** | Random legal edits (matched edit budget to A2) | 5'UTR + CDS + 3'UTR (joint) | random policy with legal mask | 1000–5000 |

**Rationale for arms**:
- **A1** (wild-type): baseline, natural mRNA performance.
- **A2** (MEF-edited): the method under test — joint RL editing.
- **A3** (GEMORNA): SOTA de-novo baseline — generates full mRNA without wild-type anchor.
- **A4** (LinearDesign): SOTA CDS-only baseline — isolates CDS optimization contribution.
- **A5** (random-legal): ablation — controls for edit budget; if A2 ≈ A5, MEF adds no value beyond random legal edits.

### 2.2 Cargo Selection

Select 5–10 therapeutic cargos spanning diverse protein classes:

| Cargo | Protein class | Length (nt) | Rationale |
|---|---|---|---|
| eGFP | reporter | 720 | standard benchmark, high dynamic range |
| Firefly luciferase | reporter | 1650 | standard stability/TE readout |
| Erythropoietin (EPO) | secreted cytokine | 580 | therapeutic, secreted |
| Factor IX | secreted enzyme | 1380 | therapeutic, liver-targeted |
| SARS-CoV-2 spike (truncated) | membrane viral | 2400 | therapeutic, membrane-anchored |
| ACTB | cytoplasmic structural | 1130 | housekeeping, codon-optimized benchmark |
| HLA-A | membrane immune | 1080 | therapeutic, immune context |
| anti-CD19 CAR | membrane therapeutic | 1350 | therapeutic, T-cell context |
| alpha-galactosidase A | lysosomal enzyme | 1290 | therapeutic, rare disease |
| luciferase + PEST | reporter destabilized | 1800 | stability-sensitive readout |

**Cargo stratification**: Each cargo × arm × cell type gets N/cargo sequences. With 10 cargos and 1000 sequences per arm, that's 100 sequences per cargo per arm.

### 2.3 Cell Types

| Cell | Type | Rationale |
|---|---|---|
| HEK293T | immortalized, easy transfection | baseline, standard MPRA cell |
| HepG2 | hepatocyte-like | liver cargo (Factor IX, alpha-gal) |
| Primary human T cells | primary, therapeutic | CAR-T context, translational relevance |

**Cell stratification**: All cargos × all arms in HEK293T (full panel). HepG2 and primary T cells get a subset (3–5 cargos each, selected by therapeutic relevance).

### 2.4 Readouts

| Readout | Method | Timepoints | Cells |
|---|---|---|---|
| MRL (ribosome load) | polysome profiling + RNA-seq | 24h post-transfection | HEK293T, HepG2 |
| TE (translational efficiency) | Ribo-seq / polysome ratio | 24h | HEK293T, HepG2 |
| mRNA half-life | actinomycin D chase + qPCR | 4h, 8h, 12h, 24h | HEK293T, HepG2 |
| Protein expression | flow cytometry (intracellular) + ELISA (secreted) | 24h, 48h, 72h | all 3 cells |

### 2.5 mRNA Synthesis

- **Template**: linearized plasmid or PCR product
- **Synthesis**: T7 or SP6 in vitro transcription (IVT)
- **Cap**: CleanCap (trinucleotide cap 1, anti-reverse cap analog not needed)
- **Poly(A)**: 120 nt poly(A) tail (encoded in template, not enzymatic)
- **Modification**: N1-methylpseudouridine (m1Ψ) for all cargos (therapeutic standard)
- **Purification**: cellulose-based dsRNA removal + silica column
- **QC**: Bioanalyzer (size), Nanodrop (concentration), ddPCR (integrity)

### 2.6 Transfection

- **HEK293T**: Lipofectamine 3000, 100 ng mRNA per well (96-well), 3 biological replicates
- **HepG2**: Lipofectamine MessengerMAX, 200 ng per well, 3 replicates
- **Primary T cells**: Lonza Nucleofector (primary cell protocol), 500 ng per reaction, 3 donors × 2 technical replicates

---

## 3. Sequence Generation Protocol

### 3.1 Wild-type Pool (A1)

Source: P1-11 long-view reconstruction (`data/reconstructed/p1_long_view/combined/long_view.records.jsonl`, 271,877 records).

Selection:
- Stratified by cargo: for each cargo, select 100 wild-type human transcripts whose CDS matches the cargo's protein (by Ensembl/UniProt mapping).
- If cargo is non-human (eGFP, luciferase, SARS-CoV-2 spike), use a humanized CDS as the "wild-type".
- Length filter: 5'UTR ≤ 512, CDS ≤ 3072, 3'UTR ≤ 1024 (P1-11 caps).
- Family-disjoint: use v1 frozen family split to ensure no family overlap with training data.

### 3.2 MEF-edited (A2)

- Starting point: A1 wild-type sequences.
- Policy: trained MEF policy (CTO + synergy RL, P1-12) with `--inference-mode calibrated_marginal`.
- Edit budget: max 3 edits per region (9 total), enforced by CTO constraint.
- Generate 1 edited variant per wild-type (matched N).

### 3.3 GEMORNA-de-novo (A3)

- Input: protein sequence + expression context (cell type, cargo).
- Output: full-length mRNA (5'UTR + CDS + 3'UTR).
- Use GEMORNA public checkpoint (Science 2025).
- Generate 1 de-novo sequence per wild-type (matched N).

### 3.4 LinearDesign-CDS-only (A4)

- Input: wild-type 5'UTR + 3'UTR + protein sequence.
- CDS: LinearDesign-optimized (Nature 2023) with default CAI/MFE weights.
- Output: wild-type 5'UTR + LinearDesign CDS + wild-type 3'UTR.
- Generate 1 CDS-optimized variant per wild-type (matched N).

### 3.5 Random-legal-edits (A5)

- Starting point: A1 wild-type sequences.
- Policy: random policy with `build_legal_action_mask` (P1-07).
- Edit budget: matched to A2 (same number of edits per sequence).
- Generate 1 random-legal variant per wild-type (matched N).

### 3.6 Matching

All 5 arms have **matched N** and **matched edit budget** (where applicable). The wild-type sequence is the **anchor** for A2, A4, A5 (paired design). A3 (GEMORNA) is unpaired (de-novo), but matched on cargo and length.

---

## 4. Pre-Registered Statistical Analysis Plan

**This section is FROZEN as of 2026-07-19. Any deviation must be documented as a protocol amendment with justification.**

### 4.1 Primary Endpoint

**Synergy score** (per sequence):

```
syn_sum = R(A2) - (R(A2_5utr_only) + R(A2_cds_only) + R(A2_3utr_only))
```

where `R(·)` is the readout (MRL, TE, half-life, or protein expression), and `A2_*_only` are **counterfactual single-region edits** derived from the same MEF policy with region-restricted masks (P1-13 protocol).

**Note**: This requires an **auxiliary counterfactual panel** (5 arms × 3 single-region variants = 8 variants per wild-type in the auxiliary panel). The main panel (A1–A5) measures method-vs-method; the auxiliary panel measures synergy decomposition within MEF.

### 4.2 Secondary Endpoints

1. **Method comparison** (A2 vs A3, A4, A5):
   - `delta_MEF_GEMORNA = R(A2) - R(A3)`
   - `delta_MEF_LinearDesign = R(A2) - R(A4)`
   - `delta_MEF_random = R(A2) - R(A5)`

2. **Region-pair synergy** (from auxiliary panel):
   - `syn_5utr_cds = R(5UTR+CDS) - R(5UTR) - R(CDS) + R(wild-type)`
   - `syn_cds_3utr = R(CDS+3UTR) - R(CDS) - R(3UTR) + R(wild-type)`
   - `syn_5utr_3utr = R(5UTR+3UTR) - R(5UTR) - R(3UTR) + R(wild-type)`

3. **OOD stress**:
   - Per-cargo: leave-one-cargo-out (LOCO) — train MEF on 9 cargos, test on 1.
   - Per-cell: train on HEK293T, test on HepG2 / primary T cells.

### 4.3 Hypothesis Testing

| Test | H0 | H1 | α | Power | N needed |
|---|---|---|---|---|---|
| Primary synergy | syn_sum = 0 | syn_sum > 0 | 0.001 | 0.8 | ~500 per arm (d=0.5) |
| MEF > GEMORNA | delta = 0 | delta > 0 | 0.001 | 0.8 | ~1000 per arm (d=0.35) |
| MEF > LinearDesign | delta = 0 | delta > 0 | 0.001 | 0.8 | ~1000 per arm (d=0.35) |
| MEF > random | delta = 0 | delta > 0 | 0.001 | 0.8 | ~500 per arm (d=0.5) |

**Multiple testing correction**: Benjamini-Hochberg FDR at q = 0.05 across all primary + secondary tests.

**Family-level inference**: Cluster sequences by gene family (v1 frozen family assignments). Report family-level paired t-test in addition to sequence-level.

### 4.4 Effect Size

- **Cohen's d** (paired): mean(delta) / std(delta).
- **Interpretation**: d > 0.5 = medium, d > 0.8 = large.
- **Pre-registered threshold**: d > 0.5 for primary synergy to claim "positive finding".

### 4.5 Robustness Checks

1. **Permutation test**: 10,000 permutations of arm labels, recompute syn_sum, compare to observed.
2. **Bootstrap CI**: 10,000 bootstrap resamples, report 95% CI for syn_sum and deltas.
3. **Length-stratified analysis**: bin sequences by total length (quartiles), recompute syn_sum per bin.
4. **Cargo-stratified analysis**: per-cargo syn_sum, report consistency.
5. **Cell-type-stratified analysis**: per-cell syn_sum, report consistency.

### 4.6 Quality Control

- **Transfection efficiency**: >80% (HEK293T), >60% (HepG2), >40% (primary T cells). Wells below threshold are excluded.
- **RNA integrity**: Bioanalyzer RIN > 7. Samples below are excluded.
- **Read depth**: polysome RNA-seq ≥ 10M reads per sample; Ribo-seq ≥ 20M reads per sample.
- **Replicate concordance**: Pearson r > 0.9 between biological replicates. Outliers excluded.
- **Batch effects**: include batch as covariate in linear model. Report batch-corrected and uncorrected results.

### 4.7 Sample Size Justification

With N = 1000 per arm:
- Detect d = 0.35 at α = 0.001, power = 0.8 (one-sided paired t-test).
- Detect d = 0.5 at α = 0.001, power = 0.95.

With N = 5000 per arm:
- Detect d = 0.15 at α = 0.001, power = 0.8.

**Recommended N**: 1000 per arm for primary analysis, expandable to 5000 if budget allows.

---

## 5. Data Schema

### 5.1 Per-sequence Metadata

```json
{
  "sequence_id": "mpra_v1_<arm>_<cargo>_<idx>",
  "arm": "A1|A2|A3|A4|A5",
  "cargo": "eGFP|luciferase|EPO|...",
  "wild_type_anchor_id": "<transcript_id or null for A3>",
  "five_utr": "<RNA sequence, max 512 nt>",
  "cds": "<RNA sequence, max 3072 nt, starts with AUG, length multiple of 3>",
  "three_utr": "<RNA sequence, max 1024 nt>",
  "edit_budget_used": "<int, 0 for A1, matched for A2/A5>",
  "method_metadata": {
    "policy_checkpoint": "<SHA-256 or null>",
    "inference_mode": "calibrated_marginal|legacy_direct",
    "seed": "<int>"
  },
  "source": {
    "url": "<source URL or null>",
    "sha256": "<source SHA-256 or null>",
    "license": "<license or null>"
  }
}
```

### 5.2 Per-readout Metadata

```json
{
  "sequence_id": "mpra_v1_...",
  "readout": "MRL|TE|half_life|protein_expression",
  "cell_type": "HEK293T|HepG2|primary_T",
  "replicate": 1|2|3,
  "timepoint_h": 4|8|12|24|48|72,
  "value": <float>,
  "unit": "<unit>",
  "qc_flag": "pass|fail|exclude",
  "batch": "<batch ID>"
}
```

### 5.3 Manifest

```json
{
  "dataset_name": "full_length_joint_mrna_mpra_v1",
  "version": "1.0",
  "date_frozen": "2026-07-19",
  "design_doc_sha256": "<this doc>",
  "n_sequences_per_arm": 1000,
  "n_arms": 5,
  "n_cargos": 10,
  "n_cell_types": 3,
  "n_readouts": 4,
  "arms": ["A1_wild_type", "A2_MEF_edited", "A3_GEMORNA_de_novo", "A4_LinearDesign_CDS_only", "A5_random_legal"],
  "cargos": ["eGFP", "luciferase", "EPO", "Factor_IX", "SARS_CoV_2_spike_trunc", "ACTB", "HLA_A", "anti_CD19_CAR", "alpha_gal_A", "luciferase_PEST"],
  "cell_types": ["HEK293T", "HepG2", "primary_T"],
  "readouts": ["MRL", "TE", "half_life", "protein_expression"],
  "statistical_plan": {
    "primary_endpoint": "syn_sum",
    "alpha": 0.001,
    "power": 0.8,
    "multiple_testing_correction": "Benjamini-Hochberg FDR q=0.05",
    "effect_size_threshold": 0.5,
    "robustness_checks": ["permutation_test", "bootstrap_ci", "length_stratified", "cargo_stratified", "cell_type_stratified"]
  },
  "constraints": [
    "Family-disjoint (v1 frozen family split)",
    "Edit budget matched (A2 vs A5)",
    "N matched across arms",
    "3 biological replicates per (sequence, cell, readout)"
  ]
}
```

---

## 6. Budget and Timeline

### 6.1 Cost Estimate ( rough order of magnitude)

| Item | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| mRNA synthesis (1–4 kb, m1Ψ, CleanCap) | $50–200 | 5,000–25,000 | $250K–$5M |
| Transfection reagents | $1–5 per well | 15,000–75,000 wells | $15K–$375K |
| Polysome profiling + RNA-seq | $200 per sample | 1,000–5,000 | $200K–$1M |
| Ribo-seq | $500 per sample | 500–2,500 | $250K–$1.25M |
| Flow cytometry | $5 per well | 15,000–75,000 | $75K–$375K |
| ELISA | $10 per well | 5,000–25,000 | $50K–$250K |
| Primary T cells | $2,000 per donor | 3–10 donors | $6K–$20K |
| Bioinformatics | — | — | $50K–$100K |
| **Total (rough)** | | | **$900K–$8.5M** |

**Recommended scope for v1**: 1000 sequences per arm × 5 arms × 5 cargos × 1 cell type (HEK293T) = 25,000 sequences, ~$2–3M.

### 6.2 Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Design freeze (this doc) | done | this doc |
| Sequence generation (computational) | 2 weeks | 5 arms × 1000–5000 sequences |
| Pilot (50 sequences × 5 arms, HEK293T only) | 4 weeks | pilot data, QC validation |
| Full panel (1000–5000 × 5 arms, HEK293T) | 12 weeks | primary data |
| Secondary cells (HepG2, primary T) | 8 weeks | cross-cell data |
| Bioinformatics + stats | 4 weeks | analysis report |
| **Total** | **~30 weeks (7.5 months)** | **full dataset + report** |

---

## 7. Connection to P1 / P2

### 7.1 Upstream (P1)

| Task | Role |
|---|---|
| P1-00 (Stage A health) | Trained policy for A2 (MEF-edited). Currently STOP. |
| P1-04 (cross-fitted ensemble) | Oracle for sequence selection (which wild-types to include). |
| P1-07 (action distribution) | Legal action mask for A5 (random-legal). |
| P1-11 (long-view reconstruction) | Wild-type pool for A1 (271,877 records available). |
| P1-12 (CTO + synergy RL) | Policy for A2 (CTO constraint + synergy reward). |
| P1-13 (counterfactual panel) | Auxiliary panel for synergy decomposition (in silico). |

### 7.2 Downstream (P2+)

| Task | Role |
|---|---|
| P2-01 (headline evidence) | This dataset provides the headline evidence for 壁垒 2. |
| P2-08 (large-scale RL) | If synergy is detected, the dataset validates the synergy RL training signal. |
| P3-02 (external SOTA fairness) | A3 (GEMORNA) and A4 (LinearDesign) are SOTA baselines. |
| P5 (submission) | This dataset is the core data contribution for top-journal submission. |

---

## 8. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| mRNA synthesis fails for long cargos (>3 kb) | medium | high | Use truncated cargos; cap CDS at 3072 nt (P1-11 cap) |
| Transfection efficiency varies across arms | medium | high | Include transfection marker (e.g., co-transfect GFP plasmid); normalize by transfection efficiency |
| Polysome profiling has low dynamic range | low | medium | Use Ribo-seq as backup (higher sensitivity) |
| Primary T cell donor variability | high | medium | 3+ donors, mixed-effect model with donor as random effect |
| GEMORNA checkpoint not public | medium | high | Degradation: replace A3 with mRNA-GPT or ProMORNA; document in `data_acquisition_blocker.md` |
| MEF policy not trained (P1-00 STOP) | high | critical | Degradation: use P1-12 tiny-trained policy on synthetic MDP; document as limitation |
| Synergy not detected (null result) | medium | high | Pre-registered null finding + methodology contribution; shift main claim to 壁垒 4 + 壁垒 1 |

---

## 9. Ethical and Regulatory Considerations

- **IRB**: Primary human T cells require IRB approval and informed consent. Use de-identified leukapheresis products from healthy donors.
- **Biosafety**: SARS-CoV-2 spike (truncated, no furin cleavage site) requires BSL-2.
- **Dual use**: mRNA sequences for therapeutic cargos (EPO, Factor IX, CAR) have dual-use potential. Sequence data will be released under a responsible-disclosure license (e.g., CC-BY-NC for academic use, commercial license required for therapeutic development).

---

## 10. Data Release Plan

| Artifact | Release timing | License |
|---|---|---|
| Design doc (this file) | immediate (frozen) | CC-BY 4.0 |
| Sequence data (5 arms) | after primary analysis | CC-BY-NC 4.0 |
| Readout data (MRL/TE/half-life/protein) | after primary analysis | CC-BY-NC 4.0 |
| Analysis code | after primary analysis | MIT |
| Pre-registered analysis plan | immediate (this doc, Section 4) | CC-BY 4.0 |

**Pre-registration**: This design doc is the pre-registration. It will be timestamped (SHA-256 + date) and deposited to a pre-registration server (OSF or AsPredicted) before wet-lab execution begins.

---

## 11. Constraints Honored

- ✅ Design + statistical plan FROZEN (Section 4). Wet-lab execution NOT in scope.
- ✅ All sequence sources record source URL / SHA-256 / license / counts / split / metadata (Section 5).
- ✅ Family-disjoint split enforced (v1 frozen family split, Section 3.1).
- ✅ Edit budget matched across arms (A2 vs A5, Section 3.6).
- ✅ "improves TE/stability/expression" claims deferred until readout data is available; current doc uses "predicted" or "hypothesized" language.
- ✅ No Stage A training disturbance (P1-00 STOP respected; A2 policy is a separate computational step).

---

## 12. Next Steps

1. **Immediate**: Deposit this design doc to OSF for timestamped pre-registration.
2. **After P1-04 + P1-05**: Use the cross-fitted ensemble to select the 1000 wild-type sequences that maximize oracle uncertainty (active learning) — this ensures the panel probes the most informative region of sequence space.
3. **After P1-00 restart or tiny-policy training**: Generate A2 (MEF-edited) sequences using the trained policy.
4. **Pilot (50 sequences × 5 arms)**: Validate transfection, readout dynamic range, and QC thresholds before committing to the full panel.
5. **Full panel**: Execute according to this design, analyze according to Section 4.

---

**End of design document. This design and statistical plan is FROZEN as of 2026-07-19.**
