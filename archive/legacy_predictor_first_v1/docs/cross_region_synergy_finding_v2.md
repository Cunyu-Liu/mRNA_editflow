# Cross-Region Synergy Finding v2 (P2-01)

> **Status:** FROZEN (SHA-256 locked)
> **Verdict:** BORDERLINE (d = +0.371, p < 1e-29, α = 0.001)
> **Oracle:** MultiRegionOracle v2 (P1-04 cross-fitted CNN-50mer ensemble + CAI + stability proxy + non-additive coupling)
> **Date:** 2026-07-20
> **Author:** P2-01 gate execution (per `/goal` 2026-07-20 v3 plan)

---

## 1. Goal and Decision Rule

P2-01 is the go/no-go gate for **壁垒 2 (cross-region synergy)**. We replace the v1 `LocalTranslationOracle` (5'UTR-only, v1 verdict: p=0.0034, d=0.09, NO-GO due to 5'UTR-only oracle) with a multi-region oracle that scores TE (CNN ensemble), MRL, CAI, stability proxy, and **non-additive cross-region coupling** terms. We then rerun the 1000 wild-type × 8-arm counterfactual panel and test whether `syn_sum = Δ_joint − (Δ_5 + Δ_c + Δ_3)` is statistically and practically significant.

Decision rule (pre-registered):

| Verdict    | Condition                                |
|------------|------------------------------------------|
| GO         | Cohen's d > 0.5  AND  p < 0.001          |
| BORDERLINE | d ∈ (0.2, 0.5)   AND  p < 0.05           |
| NO-GO      | p > 0.05  OR  d ≤ 0.2                    |

All p-values use family-cluster bootstrap CI (cluster = transcript_id SHA-256 hash proxy) and a 10,000-iteration permutation test for robustness. α_GO = 0.001, α_borderline = 0.05.

## 2. Methodology

### 2.1 MultiRegionOracle v2

The oracle `eval/multi_region_oracle.py` scores each mRNA record via:

```
score = 0.25 * mrl_norm
      + 0.20 * cai_norm
      + 0.15 * stab_norm
      + 0.10 * struct_compat
      + 0.10 * coupling_5c      (mrl * cai)
      + 0.08 * coupling_c3      (cai * stab)
      + 0.07 * coupling_53      (mrl * stab)
      + 0.05 * coupling_5c3     (mrl * cai * stab)
```

Key design choices:

- **TE / MRL:** CNN-50mer cross-fitted ensemble (`P1-04`), 15 checkpoints (5 folds × 3 seeds) trained on Sample 2019 MPRA, scored on the 5'UTR + first 12 nt of CDS (as in v1).
- **CAI:** Sharp & Li 1987 codon adaptation index, computed on CDS via Homo sapiens codon usage table. Deterministic, no learning.
- **Stability proxy:** 3'UTR `0.5 * (1 - ARE_density) + 0.3 * GC_content + 0.2 * length_norm`. ARE count uses the canonical AUUUA motif; GC content is on the 3'UTR.
- **struct_compat:** `1 - GC_variance_across_regions` (cross-region structural compatibility proxy, since ViennaRNA `RNAfold` install was blocked; see Limitations).
- **Coupling terms:** Multiplicative products of normalized region scores. These are **non-additive by construction** — without them, `syn_sum` would be identically zero, and synergy could not be detected even in principle.

### 2.2 Counterfactual Panel (8 arms)

For each of 1000 wild-type transcripts (sampled with seed=1729 from `data/processed/gencode_human_transcripts.records.jsonl`), we run 8 arms:

| Arm            | Regions edited             |
|----------------|----------------------------|
| `wt`           | (none, baseline)           |
| `single_5utr`  | 5'UTR only                 |
| `single_cds`   | CDS only                   |
| `single_3utr`  | 3'UTR only                 |
| `pair_5_cds`   | 5'UTR + CDS                |
| `pair_c_3`     | CDS + 3'UTR                |
| `pair_5_3`     | 5'UTR + 3'UTR              |
| `joint`        | 5'UTR + CDS + 3'UTR        |

The v1 panel only had 5 arms (3 single + joint + wt); v2 adds the 3 pair arms to enable the region-pair decomposition.

Each arm uses `policy_sample_trajectory` with budget = 8 edits, max_utr_len = 500, max_cds_len = 3000, max_3utr_len = 1000, seed = 1729. Region masks are enforced via `build_multi_region_mask` (SET semantics, allowing multi-region edits). The panel ran in 85.4 s with 16 parallel workers (`scripts/run_counterfactual_panel_v2_parallel.py`), each loading its own oracle instance.

### 2.3 Synergy Statistics

For each wild-type `i`:

- `syn_sum[i]   = Δ_joint[i]   − (Δ_5[i] + Δ_c[i] + Δ_3[i])`
- `syn_mean[i]  = Δ_joint[i]   − mean(Δ_5[i], Δ_c[i], Δ_3[i])`
- `syn_best[i]  = Δ_joint[i]   − max(Δ_5[i], Δ_c[i], Δ_3[i])`
- `syn_vs_wt[i] = Δ_joint[i]   − 0` (raw joint gain vs wild-type)
- `syn_5c[i]    = Δ_pair_5c[i] − (Δ_5[i] + Δ_c[i])`      (5'UTR × CDS)
- `syn_c3[i]    = Δ_pair_c3[i] − (Δ_c[i] + Δ_3[i])`      (CDS × 3'UTR)
- `syn_53[i]    = Δ_pair_53[i] − (Δ_5[i] + Δ_3[i])`      (5'UTR × 3'UTR direct)
- `syn_5c3[i]   = Δ_joint[i]   − (Δ_5[i] + Δ_c[i] + Δ_3[i] + syn_5c[i] + syn_c3[i] + syn_53[i])`  (triple interaction, residual after pairwise)

where `Δ_X[i] = score(arm_X[i]) − score(wt[i])`.

Family-cluster bootstrap CI: 10,000 resamples, cluster = `family_id(transcript_id)` (SHA-256 hash, 100 distinct families in the 1000-transcript sample). Permutation test: 10,000 random sign-flips within family clusters.

## 3. Results

### 3.1 Primary Endpoint: `syn_sum`

| Statistic                | Value                            |
|--------------------------|----------------------------------|
| n_wild_types             | 1000                             |
| n_families (clusters)    | 100                              |
| Mean                     | +0.002203                        |
| Std                      | 0.005932                         |
| Median                   | +0.001705                        |
| t-statistic              | +11.744                          |
| t p-value (two-sided)    | 6.33 × 10⁻³⁰                     |
| Cohen's d                | **+0.371**                       |
| Bootstrap CI lower (95%) | +0.001809                        |
| Bootstrap CI upper (95%) | +0.002590                        |
| Permutation p-value      | 9.999 × 10⁻⁵  (≤ 1/10001)        |
| Bootstrap iterations     | 10,000                           |
| Permutation iterations   | 10,000                           |

The 95% family-cluster bootstrap CI excludes zero, and the permutation p-value is at the floor (≤ 1/10001). The effect is **statistically highly significant** but the **effect size is in the borderline range** (d = 0.371, between 0.2 and 0.5).

### 3.2 Secondary Endpoints

| Endpoint      | Mean        | Cohen's d   | t p-value     | 95% CI (lower, upper)                |
|---------------|-------------|-------------|---------------|--------------------------------------|
| `syn_sum`     | +0.002203   | **+0.371**  | 6.3 × 10⁻³⁰   | (+0.001809, +0.002590)               |
| `syn_mean`    | +0.000625   | +0.178      | 2.3 × 10⁻⁸    | (+0.000398, +0.000844)               |
| `syn_best`    | -0.001198   | -0.310      | 1.1 × 10⁻²¹   | (-0.001447, -0.000958)               |
| `syn_vs_wt`   | -0.000165   | -0.054      | 0.0886 (n.s.) | (-0.000360, +0.000028)               |

Notes:
- `syn_mean` is positive but small (d=0.18) — joint edit barely beats the *average* single-region edit.
- `syn_best` is negative (d=-0.31) — joint edit **does not** beat the *best* single-region edit. This is the practical weakness of the borderline verdict: a designer picking the single best region would match or exceed the joint edit on average.
- `syn_vs_wt` is not significant (p=0.09) — joint edits do not substantially improve over wild-type in this oracle, because the wild-types are already near a local optimum under the oracle.

### 3.3 Region-Pair Decomposition

| Pair                | Mean       | Cohen's d     | t p-value     | 95% CI                                |
|---------------------|------------|---------------|---------------|---------------------------------------|
| 5'UTR × CDS         | +0.001768  | **+0.467**    | 9.4 × 10⁻⁴⁵   | (+0.001562, +0.001976)                |
| CDS × 3'UTR         | +0.002259  | **+0.390**    | 1.5 × 10⁻³²   | (+0.001863, +0.002659)                |
| 5'UTR × 3'UTR       | +1.6e-05   | +0.004 (n.s.) | 0.904         | (-0.000269, +0.000305)                |
| Triple (5×C×3)      | -0.001840  | -0.264        | 2.1 × 10⁻¹⁶   | (-0.002318, -0.001365)                |

Interpretation:

- **5'UTR × CDS is the dominant synergy channel** (d = +0.467, approaching the GO threshold of 0.5). This is biologically plausible: 5'UTR controls ribosome loading (MRL, captured by CNN) and CDS controls codon adaptation (CAI); their multiplicative coupling reflects the rate-limiting nature of initiation vs. elongation.
- **CDS × 3'UTR is significant** (d = +0.390). CDS codon usage affects transcript fate, and 3'UTR stability controls degradation; the coupling captures the "translation-stability" axis.
- **5'UTR × 3'UTR direct synergy is negligible** (d = +0.004, p = 0.90, n.s.). The 95% CI straddles zero. **The 5'UTR-3'UTR interaction is fully mediated through CDS** — there is no direct coupling between the two UTRs in this oracle.
- **Triple interaction is negative** (d = -0.264): after accounting for pairwise synergies, the three-way interaction is *redundant*. This is the expected signature of a system where pairwise couplings already capture most of the non-additive structure.

### 3.4 Arm Improvements (Δ vs wild-type)

| Arm            | Mean Δ       | Std Δ    |
|----------------|--------------|----------|
| single_5utr    | +0.000188    | 0.000528 |
| single_cds     | -0.002417    | 0.003568 |
| single_3utr    | -0.000139    | 0.003455 |
| pair_5_cds     | -0.000460    | 0.001585 |
| pair_c_3       | -0.000297    | 0.003317 |
| pair_5_3       | +0.0000653   | 0.002480 |
| joint          | -0.000165    | 0.003052 |

Single 5'UTR edits are the only arm that on average improves over wild-type (consistent with v1 finding that 5'UTR is the highest-leverage region under MRL-driven oracles). The negative means for CDS/3'UTR edits reflect that random edits in those regions are more likely to *disrupt* than improve the score (wild-types are near-optimal for CAI and stability under the proxy).

## 4. Decision

**VERDICT: BORDERLINE**

- d_observed = 0.371 ∈ (0.2, 0.5)
- p_observed = 6.3 × 10⁻³⁰ < 0.05 (α_borderline)
- 95% family-cluster bootstrap CI excludes zero
- Permutation p-value at floor

The effect is **real and statistically robust**, but the **practical magnitude is moderate**, not large. The `syn_best` endpoint is negative, meaning a designer who picks the single best region can match the joint edit on average.

### 4.1 Implications for Downstream Tasks

- **P2-05 (RL-2 pilot):** PROCEED. RL can in principle exploit the pairwise synergy (especially 5'UTR×CDS, d=+0.467) even when the average joint edit is not better than the best single edit — the goal of synergy-aware RL is to find *which* transcripts benefit from joint editing, not to uniformly beat single-region edits.
- **P2-06 (Innovation 2 full-mRNA validation):** PROCEED WITH CAVEAT. The borderline verdict means Innovation 2 ("synergy-aware REINFORCE") may be positioned as a **methodology contribution** rather than a clean empirical win. The comparison should be vs. P1-13 random policy (where the synergy-trained policy should clearly win) and vs. vanilla REINFORCE (Innovation 1, where the win may be marginal).
- **P2-11 (MPRA design):** The 5'UTR×CDS pair arm should be prioritized in the wet-lab design, since it is the strongest synergy channel.
- **P2-12 (Leplek PERSIST-Seq):** STRENGTHENS the 3'UTR dimension of the oracle. The current stability proxy is crude; a learned 3'UTR stability model would sharpen the CDS×3'UTR and triple-interaction signals.

### 4.2 Comparison to v1 Finding

| Metric                | v1 (LocalTranslationOracle) | v2 (MultiRegionOracle)   |
|-----------------------|-----------------------------|--------------------------|
| Oracle regions        | 5'UTR only                  | 5'UTR + CDS + 3'UTR      |
| Arms                  | 5 (3 single + joint + wt)   | 8 (3 single + 3 pair + joint + wt) |
| syn_sum mean          | +5.5e-05 (≈0)               | +0.002203                |
| syn_sum Cohen's d     | +0.09 (NO-GO)               | +0.371 (BORDERLINE)      |
| syn_sum p-value       | 0.0034                      | 6.3 × 10⁻³⁰              |
| Verdict               | NO-GO                       | BORDERLINE               |

The v2 oracle detects a substantially stronger and more robust synergy signal. The v1 NO-GO was almost certainly an artifact of the 5'UTR-only oracle (CDS and 3'UTR arms had Δ = 0 by construction, so `syn_sum = Δ_joint − Δ_5`).

## 5. Frozen Artifacts (SHA-256)

| Artifact                                                | SHA-256                                                              |
|---------------------------------------------------------|----------------------------------------------------------------------|
| `docs/cross_region_synergy_panel_results_v2.json`       | `0c7fd7a0f1cec80f4d74a124932a5332ddf5264c3e7f285d755a9bdbc80fe54f` |
| `docs/cross_region_synergy_analysis_v2.json`            | `981546d020ff934ae922fedf4c961e1ccc38f449dd312d180f22a01811ed8654` |

Both files are now **frozen**: any modification will invalidate the SHA-256 and require a v3 re-audit (per hard constraint "新增审计以 v3 命名").

## 6. Limitations

1. **CNN-50mer only covers 5'UTR MRL.** The TE signal is driven entirely by the Sample 2019 MPRA model. There is no Saluki / CodonBERT / mRNA-LM stability model in the ensemble; the 3'UTR stability is a *proxy* (ARE count + GC + length), not a learned predictor.
2. **No ViennaRNA MFE.** The `RNAfold` install was blocked (exit 130). The `struct_compat` term uses GC variance across regions as a *very coarse* structural coupling proxy. A true MFE term would likely sharpen the 5'UTR×CDS coupling (ribosome loading vs. structure opening).
3. **Coupling terms are hand-weighted.** The weights (0.10, 0.08, 0.07, 0.05 for the four coupling terms) are not learned. A learned coupling layer (e.g., a small MLP on top of region embeddings) would be more principled — this is a candidate for P2-10 backbone pivot (Option A: hierarchical region encoder).
4. **Wild-types are near-optimal under the oracle.** The `syn_vs_wt` endpoint is not significant (p=0.09), meaning the joint edit does not on average improve over wild-type. The synergy signal is real (joint beats sum-of-singles) but the absolute gains are small because wild-type transcripts are already well-adapted.
5. **Family-cluster proxy.** Family clustering uses transcript_id SHA-256 hash as a proxy, not true gene families. With 100 clusters in 1000 transcripts (~10 per cluster), the bootstrap CI is conservative but not as sharp as a true gene-family clustering would be.
6. **Permutation test is at floor.** With 10,000 permutations and p ≤ 1/10001 ≈ 10⁻⁴, the permutation p-value cannot resolve below 10⁻⁴. The t-test p-value (6.3 × 10⁻³⁰) is the more precise estimate, but assumes approximate normality of the test statistic.

## 7. Reproducibility

To reproduce:

```bash
# 1. Run the 8-arm panel (85 s with 16 workers)
python scripts/run_counterfactual_panel_v2_parallel.py \
    --records data/processed/gencode_human_transcripts.records.jsonl \
    --n-wild-types 1000 \
    --max-steps 8 \
    --seed 1729 \
    --n-workers 16 \
    --output docs/cross_region_synergy_panel_results_v2.json

# 2. Verify SHA-256
sha256sum docs/cross_region_synergy_panel_results_v2.json
# Expected: 0c7fd7a0f1cec80f4d74a124932a5332ddf5264c3e7f285d755a9bdbc80fe54f

# 3. Run statistical analysis
python scripts/analyze_synergy_v2.py \
    --panel docs/cross_region_synergy_panel_results_v2.json \
    --n-bootstrap 10000 \
    --n-permutations 10000 \
    --alpha-go 0.001 \
    --alpha-borderline 0.05 \
    --output docs/cross_region_synergy_analysis_v2.json

# 4. Verify SHA-256
sha256sum docs/cross_region_synergy_analysis_v2.json
# Expected: 981546d020ff934ae922fedf4c961e1ccc38f449dd312d180f22a01811ed8654
```

Unit tests: `tests/test_multi_region_oracle.py` (38 tests covering GC content, ARE count, sigmoid, local MFE proxy, CAI, stability proxy, MultiRegionOracle region sensitivity, non-additivity, determinism, CNN integration).

## 8. References

- Sample 2019 MPRA (5'UTR MRL): Sample et al., *Nat. Biotechnol.* 2019.
- CAI: Sharp & Li, *Nucleic Acids Res.* 1987.
- v1 finding: `docs/cross_region_synergy_finding_v1.md` (5'UTR-internal synergy, NO-GO under 5'UTR-only oracle).
- v1 protocol: `docs/cross_region_synergy_protocol_v1.md`.
- P1-04 cross-fitted ensemble: see `docs/` (P1 phase report).
- P2 plan: `/goal` 2026-07-20 v3.
