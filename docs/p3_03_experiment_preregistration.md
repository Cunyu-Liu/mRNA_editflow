# P3-03: Early Prospective Falsification — Experiment Pre-Registration

**Phase:** P3-03
**Status:** PRE-REGISTERED (locked before wet-lab execution)
**Date:** 2026-07-23
**Depends on:** P3-01 (benchmark freeze), P3-02 (local-delta oracle)
**Unlocks:** P3-04 (full-scale GRPO) upon GO/PARTIAL verdict

---

## 1. Scientific Purpose

This is a **risk-elimination experiment**, not a final paper wet-lab validation. The goal is to answer three minimal questions before committing months of compute to full-scale GRPO training:

1. **Oracle directional credibility:** Are the P3-02 local-delta oracle's predicted edit directions consistent with measured biological effects?
2. **Minimal-edit detectability:** Do minimal local edits (1–5 nt substitutions) produce measurable changes in protein output / mRNA abundance?
3. **Region priority:** Which region (5′UTR, CDS, joint) shows the largest realizable headroom?

If the oracle's predicted improvements do not transfer to wet-lab measurements, RL expansion must be paused and the oracle retrained or intervention data expanded.

---

## 2. Pre-Registration Lock

The following are **frozen before any wet-lab execution**:

| Item | Locked Value |
|------|-------------|
| Cargos | 2 (EGFP, mCherry) |
| Cell context | HEK293T |
| Total unique sequences | 24 (2 cargos × 12 arms) |
| Biological replicates | ≥ 3 per sequence |
| Time points | 4h, 8h, 24h, 48h |
| Readouts | protein output, mRNA abundance, cell viability |
| Arms per cargo | 12 (A01–A12, see §5) |
| Analysis plan | docs/p3_03_analysis_plan.md (frozen) |
| GO/PARTIAL/NO-GO criteria | docs/p3_03_go_no_go_decision.md (frozen) |
| Oracle checkpoint | P3-02 cross-fitted ensemble (checkpoints/p3_delta_oracles/) |
| Sequence manifest SHA-256 | Recorded in docs/p3_03_sequence_manifest.json |

**Any deviation from this pre-registration must be documented as an amendment with justification.**

---

## 3. Cargo Definitions

### Cargo 1: EGFP (Enhanced Green Fluorescent Protein)

| Property | Value |
|----------|-------|
| cargo_id | EGFP |
| Length | 239 aa |
| Source | P3-01 measured tier (Sample2019 MPRA), test split, WT anchor (edit_count=0) |
| 5′UTR length | 50 nt |
| Selection criteria | Moderate GC content (~50%), standard reporter |
| Role | In-distribution test — oracle trained on this data family |

### Cargo 2: mCherry (Red Fluorescent Protein)

| Property | Value |
|----------|-------|
| cargo_id | mCherry |
| Length | 236 aa |
| Source | P3-01 proxy tier, test split, longer 5′UTR (>50 nt) |
| Selection criteria | Different GC profile (~30%), different spectral property |
| Role | Out-of-distribution test — proxy-tier predictions, different length/property cargo |

**Rationale:** One standard reporter (EGFP, in-distribution) and one different-length/property cargo (mCherry, OOD) to test whether oracle predictions generalize across cargo contexts.

---

## 4. Cell Context and Readouts

### Cell Context
- **Cell line:** HEK293T
- **Transfection:** Lipofectamine-based transient transfection
- **Culture:** Standard DMEM + 10% FBS

### Readouts (Minimum)

| Readout | Method | Purpose |
|---------|--------|---------|
| Protein output | Flow cytometry (fluorescence) or Western blot | Primary endpoint |
| mRNA abundance | RT-qPCR | Translation efficiency denominator |
| Cell viability | CellTiter-Glo or equivalent | Confound control |

### Time Points (Recommended)

| Time | Rationale |
|------|-----------|
| 4h | Early kinetic — captures translation onset |
| 8h | Mid-early — steady-state approaching |
| 24h | Standard — most MPRA data measured here |
| 48h | Late — degradation/half-life signal |

### Computed Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Protein-output AUC | Trapezoidal integration over time points | Cumulative protein yield |
| mRNA-abundance AUC | Trapezoidal integration over time points | Cumulative mRNA level |
| Apparent TE | protein-output AUC / mRNA-abundance AUC | Translation efficiency proxy |
| Apparent half-life | Exponential decay fit from 24h→48h | Stability proxy |

---

## 5. Experimental Arms

Each cargo receives 12 arms. Arms are pre-registered before wet-lab execution.

| Arm | Name | Description | Selection Method |
|-----|------|-------------|-----------------|
| A01 | wt_source | Wild-type source sequence (no edits) | Baseline anchor |
| A02 | random_one_edit | Random legal single-nt substitution in 5′UTR | Random (seed-locked) |
| A03 | random_three_edit | Random legal 3-nt substitutions in 5′UTR | Random (seed-locked) |
| A04 | best_predicted_one_edit | Oracle-predicted best single-nt edit | Exact enumeration over 5′UTR |
| A05 | best_predicted_three_edit | Oracle-predicted best 3-nt edits | Greedy max-3 over 5′UTR |
| A06 | greedy_best | Greedy search best (max 5 edits) | Greedy max-5 over 5′UTR |
| A07 | beam_search_best | Beam search best (beam=5, max 3 edits) | Beam search over 5′UTR |
| A08 | five_utr_only_best | Best edit restricted to 5′UTR region | Exact enumeration, 5′UTR-only |
| A09 | cds_only_best | Best synonymous codon edit in CDS | Exact enumeration, synonymous CDS |
| A10 | joint_best | Best joint 5′UTR + CDS edit | Greedy joint 5′UTR + CDS |
| A11 | high_disagreement_negative | High ensemble disagreement, predicted negative | Max-disagreement negative control |
| A12 | adversarial_high_reward | High predicted delta but suspicious features | GC-maximizing reward-hacking test |

### Arm Categories

| Category | Arms | Purpose |
|----------|------|---------|
| Baseline | A01 | WT reference |
| Random controls | A02, A03 | Null hypothesis — random edits |
| Prediction-guided | A04–A07 | Oracle directional credibility |
| Region-specific | A08, A09, A10 | Region priority comparison |
| Negative control | A11 | Disagreement calibration |
| Adversarial | A12 | Reward hacking exposure |

**Note:** This phase does NOT include a GRPO candidate. GRPO is deferred to P3-04 pending GO/PARTIAL verdict.

---

## 6. Replication and Randomization

| Parameter | Value |
|-----------|-------|
| Biological replicates | ≥ 3 per sequence |
| Total unique sequences | 24 (2 × 12) |
| Total wells | ≥ 72 (24 × 3) |
| Plate layout | Randomized well assignment (seed-locked in manifest) |
| Batch design | Single batch per time point to avoid batch effects |

### Randomization
- Well positions are randomly assigned (seed=42, recorded in manifest)
- Replicates are distributed across plate regions to avoid positional bias
- Plate map is frozen in `docs/p3_03_sequence_manifest.json` under `well_layout`

---

## 7. Blinding

- **Sequence identity:** Wet-lab operator receives sequences with blinded arm labels (A01–A12 shuffled)
- **Predicted deltas:** Not shared with wet-lab operator
- **Unblinding:** Occurs only after all raw measurements are uploaded and SHA-256 verified

---

## 8. Sample Size Justification

| Factor | Value |
|--------|-------|
| Cargos | 2 (minimum for cross-cargo consistency check) |
| Arms per cargo | 12 (covers baseline + random + prediction-guided + region + controls) |
| Replicates | 3 (minimum for sign-test power > 0.8 with effect size d > 0.5) |
| Total sequences | 24 (within 48–96 range) |
| Total wells | 72 (within 96-well plate capacity) |

**Power analysis:** With n=3 replicates and a paired comparison (A04 vs A02), a one-sided sign test has >80% power to detect a standardized effect size of d=0.5 at α=0.05. This is sufficient for directional (not magnitude) falsification.

---

## 9. Data Integrity

| Item | Method |
|------|--------|
| Sequence manifest SHA-256 | Computed and recorded at generation time |
| Manifest immutability | Any post-hoc sequence change invalidates the pre-registration |
| Raw data upload | Measurements uploaded with well_id → arm_id mapping |
| Reproducibility | All selection scripts version-controlled (commit hash recorded) |

---

## 10. Exclusions and Outliers

### Pre-specified Exclusion Criteria
1. **Cell viability < 50% of WT:** Well excluded from analysis (toxicity confound)
2. **Failed transfection (viability < 20% of plate median):** Well excluded
3. **Technical failure (no signal):** Well excluded, recorded with reason

### Outlier Handling
- No post-hoc outlier removal beyond pre-specified exclusion criteria
- All excluded wells reported transparently with reason codes
- If >20% of wells excluded, experiment flagged for re-run

---

## 11. Timeline

| Step | Action | Dependency |
|------|--------|------------|
| 1 | Pre-registration lock (this document) | P3-02 complete |
| 2 | Sequence manifest generation | This document |
| 3 | Wet-lab execution | Manifest frozen |
| 4 | Raw data upload | Wet-lab complete |
| 5 | Analysis execution (analysis_plan.md) | Raw data uploaded |
| 6 | GO/PARTIAL/NO-GO decision (go_no_go_decision.md) | Analysis complete |
| 7 | Unblinding → P3-04 or pause | Decision recorded |

---

## 12. Risk Register

| Risk | Mitigation |
|------|-----------|
| Oracle overfits to GC/length heuristic | Adversarial arm A12 tests this explicitly |
| Minimal edits too small to measure | 3-edit and 5-edit arms (A03, A05, A06) provide larger signal |
| Single cargo bias | Two cargos with different properties |
| Proxy-tier cargo (mCherry) less reliable | Interpretation weights EGFP as primary, mCherry as secondary |
| Plate/batch effects | Randomized layout, single batch per time point |

---

## 13. Amendment Procedure

Any change to this pre-registration after wet-lab execution begins must:
1. Be documented as a numbered amendment
2. Include justification and date
3. Be approved before analysis execution
4. Be reflected in the GO/PARTIAL/NO-GO decision document

---

**Pre-registration fingerprint:**
- Document: `docs/p3_03_experiment_preregistration.md`
- Analysis plan: `docs/p3_03_analysis_plan.md`
- Decision criteria: `docs/p3_03_go_no_go_decision.md`
- Sequence manifest: `docs/p3_03_sequence_manifest.json` (SHA-256 locked)
