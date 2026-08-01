# Cross-Region Synergy Protocol v1

**Status**: P1-12 (Innovation 2) — protocol frozen, tiny-MDP convergence validated.

**Scope**: This document specifies the experimental protocol for measuring and optimizing cross-region synergy in mRNA design. It covers (1) the counterfactual edit experiment panel (P1-13), (2) the synergy score definition, (3) the statistical analysis plan, and (4) the mechanism interpretation framework.

---

## 1. Background

### 1.1 The Synergy Question

mRNA design involves editing three regions: 5'UTR, CDS, and 3'UTR. Each region contributes to functional outcomes (translation efficiency, stability, expression) through distinct mechanisms:

- **5'UTR**: ribosome recruitment, scanning efficiency, secondary structure
- **CDS**: codon usage, elongation speed, co-translational folding
- **3'UTR**: mRNA stability, miRNA binding, polyadenylation

The *additive hypothesis* assumes the joint reward equals the sum of single-region rewards:
```
R_joint = R_5'UTR + R_CDS + R_3'UTR
```

The *synergy hypothesis* posits that joint editing yields a disproportionate benefit:
```
R_joint > R_5'UTR + R_CDS + R_3'UTR
```

If synergy exists, it implies **mechanistic cross-talk** between regions — e.g., 5'UTR secondary structure modulates the effect of CDS codon usage on translation efficiency.

### 1.2 Why This Matters

Synergy is the core scientific claim of the project (壁垒 2 — Cross-Region Synergy). If synergy is significant:

1. **Scientific contribution**: First quantitative evidence of cross-region synergy in mRNA design.
2. **Methodological contribution**: The counterfactual synergy RL algorithm (Innovation 2) is a novel contribution to RL theory.
3. **Practical contribution**: Joint editing is necessary for optimal mRNA design — single-region optimization is provably suboptimal.

If synergy is NOT significant (null finding):
- The project pivots to 壁垒 4 (RL algorithm innovation) + 壁垒 1 (regulatory-grade minimal-edit) as the main claims.
- Innovation 2 (synergy RL) remains a methodological contribution, submittable to NeurIPS/ICML independently.

---

## 2. Counterfactual Edit Experiment Panel (P1-13)

### 2.1 Experimental Design

**Panel**: 1000 wild-type mRNAs × 5 arms = 5000 total sequences.

**Arms**:
1. **Wild-type**: no edits (control)
2. **Single-5'UTR**: edit only 5'UTR (region-restricted mask)
3. **Single-CDS**: edit only CDS (region-restricted mask, synonymous-only)
4. **Single-3'UTR**: edit only 3'UTR (region-restricted mask)
5. **Joint**: edit all regions (full legal action mask)

**Edit budget**: 3 edits per sequence (matching the project's `--edit-budget 3` default).

**Wild-type source**: GENCODE/RefSeq canonical records from the P0 Data Reconstruction v1 frozen namespace.

**Selection criteria**:
- 5'UTR length 50..500 nt
- CDS length 300..3000 nt (multiple of 3)
- 3'UTR length 50..1000 nt
- No ambiguous bases (N, R, Y, etc.)
- Stratified sampling across gene families (to avoid family-specific bias)

### 2.2 Reward Function

The reward is the **independent final oracle** (P1-05, frozen and hidden from the policy):
```
R(τ) = oracle_score(final_sequence)
```

The oracle is a cross-fitted predictor ensemble (P1-04) covering:
- Translation efficiency (TE)
- Ribosome load (MRL)
- Stability (half-life)
- CAI (codon adaptation index)
- MFE (minimum free energy of secondary structure)

The oracle score is a weighted combination:
```
oracle_score = w_TE · normalized_TE + w_MRL · normalized_MRL
             + w_stability · normalized_stability
             + w_CAI · normalized_CAI + w_MFE · normalized_MFE
```

Weights are frozen before the experiment and recorded in `docs/independent_oracle_design_v1.md`.

### 2.3 Counterfactual Rollout Collection

For each wild-type mRNA `i` and arm `a`:

1. Initialize the policy from the trained checkpoint (P1-12 synergy RL).
2. Set the region-restricted mask based on arm `a`.
3. Sample 1 trajectory with edit budget = 3.
4. Record the final sequence and oracle score.

**CRN trick**: All 5 arms for the same wild-type mRNA share the same RNG seed. This ensures the trajectories differ only in the action mask (region restriction), not in sampling noise.

---

## 3. Synergy Score Definition

### 3.1 Per-mRNA Synergy Score

For each wild-type mRNA `i`:
```
synergy_i = R_joint_i - Σ_a R_single_a_i
```
where `R_joint_i` is the oracle score of the joint edit, and `R_single_a_i` is the oracle score of single-region edit `a`.

### 3.2 Population Synergy Score

The population synergy score is the mean over all wild-type mRNAs:
```
synergy = (1/N) · Σ_i synergy_i
```

### 3.3 Normalized Synergy Score

To control for the scale of oracle scores, we also compute a normalized version:
```
synergy_normalized = synergy / std(R_joint)
```

---

## 4. Statistical Analysis Plan

### 4.1 Primary Hypothesis Test

**H0**: `synergy ≤ 0` (no synergy, additive or sub-additive)
**H1**: `synergy > 0` (positive synergy)

**Test**: One-sample t-test on `synergy_i` across all 1000 wild-type mRNAs.

**Significance level**: α = 0.001 (Bonferroni-corrected for multiple comparisons across oracle dimensions).

**Effect size**: Cohen's d = `mean(synergy_i) / std(synergy_i)`.

### 4.2 Secondary Analyses

1. **Per-region contribution**: Decompose `synergy_i` into contributions from each region pair (5'UTR×CDS, 5'UTR×3'UTR, CDS×3'UTR) using a 2-way ANOVA.

2. **Per-oracle-dimension synergy**: Compute synergy separately for TE, MRL, stability, CAI, MFE. Identify which functional dimensions exhibit synergy.

3. **Dose-response curve**: Vary the edit budget (1, 2, 3, 5, 10) and measure how synergy scales. Synergy should increase with edit budget if cross-region mechanisms are at play.

4. **Sequence-feature regression**: Regress `synergy_i` on sequence features (5'UTR length, CDS GC content, 3'UTR motif count) to identify predictors of synergy.

### 4.3 Multiple Comparison Correction

- Primary test: Bonferroni correction for 5 oracle dimensions → α = 0.001.
- Secondary analyses: Benjamini-Hochberg FDR at q = 0.05.

### 4.4 Sample Size Justification

With N = 1000 wild-type mRNAs and expected effect size d = 0.2 (small-to-medium), the power to detect synergy at α = 0.001 is:
```
power = Φ(d · √N - z_{1-α}) ≈ Φ(0.2 · 31.6 - 3.09) ≈ Φ(3.23) ≈ 0.9994
```
So N = 1000 is well-powered even for small effect sizes.

---

## 5. Mechanism Interpretation Framework

### 5.1 Synergy Mechanism Hypotheses

If synergy is significant, we investigate the underlying mechanisms:

**Hypothesis M1: 5'UTR structure × CDS codon usage.**
- 5'UTR secondary structure affects ribosome scanning efficiency.
- CDS codon usage affects elongation speed.
- Synergy: structured 5'UTR + optimized CDS codon usage may yield super-additive TE improvement.

**Hypothesis M2: CDS codon usage × 3'UTR stability.**
- CDS codon usage affects co-translational folding.
- 3'UTR affects mRNA stability (half-life).
- Synergy: optimized CDS folding + enhanced stability may yield super-additive expression improvement.

**Hypothesis M3: 5'UTR accessibility × 3'UTR miRNA avoidance.**
- 5'UTR accessibility affects ribosome recruitment.
- 3'UTR miRNA binding sites affect degradation.
- Synergy: accessible 5'UTR + miRNA-avoidant 3'UTR may yield super-additive TE × stability.

### 5.2 Mechanism Validation

For each hypothesis, we compute:
1. **Correlation**: `corr(synergy_i, mechanism_feature_i)` where `mechanism_feature` is a quantitative measure (e.g., 5'UTR MFE, CDS CAI, 3'UTR miRNA site count).
2. **Stratified analysis**: Split wild-type mRNAs into high/low mechanism_feature groups and compare synergy scores.
3. **Mediation analysis**: Test whether the mechanism feature mediates the synergy effect.

### 5.3 Negative Result Interpretation

If synergy is NOT significant:
1. **Check panel design**: Ensure the edit budget is sufficient (try budget = 5, 10).
2. **Check oracle**: Ensure the oracle captures cross-region effects (not just independent per-region predictors).
3. **Check wild-type selection**: Ensure wild-types are diverse (not all from the same gene family).
4. **Accept null finding**: If all checks pass and synergy is still not significant, report this as a null finding. The project pivots to 壁垒 4 + 壁垒 1 as main claims. Innovation 2 (synergy RL) remains a methodological contribution.

---

## 6. Deliverables

### 6.1 Artifacts

- `docs/cross_region_synergy_finding_v1.md` — Full report with synergy scores, statistical tests, mechanism analysis.
- `data/counterfactual_edit_panel_v1.jsonl` — 5000 sequences (1000 wild-types × 5 arms) with oracle scores.
- `data/counterfactual_edit_panel_v1_manifest.json` — Manifest with SHA-256, source URLs, license, counts, split.

### 6.2 Acceptance Criteria

- 1000 wild-type mRNAs × 5 arms = 5000 sequences collected.
- Synergy score computed for each wild-type.
- Primary hypothesis test (t-test) performed with p-value reported.
- Effect size (Cohen's d) reported.
- If synergy significant: at least one mechanism hypothesis tested.
- If synergy not significant: panel design checked, null finding documented.

---

## 7. Protocol Freeze

This protocol is **frozen** as of 2026-07-19. Any changes require:
1. Documenting the change in `docs/cross_region_synergy_protocol_v1_changelog.md`.
2. Re-freezing the protocol with a new version number (v2).
3. Re-running the entire panel with the new protocol.

**Frozen parameters**:
- N = 1000 wild-type mRNAs
- 5 arms (wild-type, single-5'UTR, single-CDS, single-3'UTR, joint)
- Edit budget = 3
- CRN trick: shared RNG seed across arms
- Oracle: independent final oracle (P1-05)
- Significance level: α = 0.001 (Bonferroni)
- Synergy score: `R_joint - Σ R_single`

---

**Document SHA-256**: to be computed after final review.
**Authors**: mRNA-EditFlow team
**Last updated**: 2026-07-19
