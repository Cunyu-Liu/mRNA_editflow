# P3-00 Hypothesis Preregistration

- **Date**: 2026-07-22 (amended by P3-00A)
- **phase_status**: PASS
- **scientific_validation_status**: PENDING — H1–H6 are preregistered, not validated. P3-00 PASS must never be cited as experimental support for any of them.
- **Status**: PREREGISTERED — thresholds below are frozen before P3-01/P3-02/P3-03 data is examined. Modifying a threshold after seeing results is forbidden (global hard constraint #22).
- **Primary contract**: `configs/p3_primary_task.yaml`
- **Claim policy**: every result is reported at its Claim-Ladder level (C0–C6). Nothing below C5 may be written as a wet-lab improvement.

Each hypothesis lists: supporting evidence / opposing evidence / validation experiment / minimum effect size / statistical method / GO / PARTIAL / NO-GO. A NO-GO on H1–H3 pauses all RL work; a NO-GO on H4 stops RL specifically (project continues as ranker+search); H5/H6 are optional unlock hypotheses and never gate the core paper.

---

## H1 — Local-edit effect existence (local-effect existence)

**Statement.** Within the legal edit neighborhood of a source mRNA (≤10 nt substitutions: 5′UTR substitutions and synonymous CDS substitutions), at least a subset of edits produces a measurable, reproducible change in protein output.

### Supporting evidence
- Sample 2019 MPRA (~280k random 50 nt 5′UTRs): dense single-nucleotide neighborhoods show large MRL differences between near-identical sequences.
- UTR-LM: among 211 prospectively tested 5′UTR designs, top designs exceeded optimized therapeutic UTRs — sequence-level headroom exists in 5′UTR.
- PERSIST-seq (233 full-length mRNAs): 5′UTR variants produced the largest ribosome-load changes across regions.
- Large-scale 3′UTR variant screens (MapUTR-like): many single variants significantly alter mRNA abundance.

### Opposing evidence
- PERSIST-seq: some high-ribosome-load sequences did **not** yield higher total protein output; MRL ≠ protein output.
- Effects may be small relative to assay noise for ≤10 edits, especially synonymous CDS edits.
- Effect direction may be strongly context-dependent (cell type, reporter backbone, cargo).

### Validation experiment
- P3-01: controlled neighborhoods — all legal single edits, random/structure-guided/top-ranked double edits, and matched negative edits per eligible source, split by region (5′UTR, first 30 codons, first 50 codons, remaining CDS, joint).
- P3-03: prospective falsification panel — 2 cargos × arms {WT, random 1-edit, random 3-edit, best-predicted 1-edit, best-predicted 3-edit, region bests, joint best, negative controls} × ≥3 biological replicates; readouts at 4h/8h/24h/48h → protein-output AUC, mRNA-abundance AUC, apparent half-life, apparent TE.

### Minimum effect size
- In-silico (P3-01/02): ≥30% of evaluated sources have ≥1 legal single edit whose predicted |delta| exceeds the oracle's calibrated noise floor.
- Prospective (P3-03): ≥1 region arm shows |Δprotein-output AUC| ≥ 2× assay SD vs WT, and the best-predicted edit beats the matched random-edit distribution.

### Statistical method
- Per-source paired contrasts of edit vs matched negative edits; mixed-effects model with source as random effect; Benjamini–Hochberg FDR across sources; assay noise estimated from WT replicates.

### GO
Measurable, direction-consistent local-edit signal in ≥1 region on both cargos (prospective) or on frozen test sources (in-silico, pending P3-03).

### PARTIAL
Signal confined to one region (e.g., 5′UTR only) or one cargo → shrink primary task to that region/cargo class.

### NO-GO
Best-edit distribution is indistinguishable from matched random-edit distribution in all regions → **pause all RL**; redirect effort to intervention-data acquisition (Level C) before any policy training.

---

## H2 — Local-delta predictability (local-delta predictability)

**Statement.** A model can predict the sign, ranking, and enrichment of local edit effects on unseen sources/cargo families — i.e., `protein_output(candidate) − protein_output(source)` — not merely absolute expression.

### Supporting evidence
- Optimus 5-Prime: high MRL prediction accuracy on held-out random 50 nt 5′UTRs.
- UTR-LM: representation transfer to unseen 5′UTRs with prospective confirmation.
- RiboNN: transcriptome-scale sequence→TE modeling is data-feasible (3,819 datasets, >140 cell types).

### Opposing evidence
- Absolute-expression prediction ≠ delta prediction; predicting cross-construct rankings can succeed while within-source edit sign prediction fails.
- eFold lesson (project memory): 99.7% overlap between a public test set and training data inflated apparent predictability — naive splits overstate H2.
- Project internal audits: training-oracle → independent-oracle transfer was only PARTIAL in P2 dashboards.

### Validation experiment
- P3-02 Task 1–2: compare Absolute / Difference / Siamese / Edit-conditioned architectures on source-disjoint validation and cargo-family-disjoint test splits; primary metrics: sign accuracy, beneficial-edit precision, top-k enrichment (secondary: delta Spearman/Pearson, pairwise AUC, calibration, source-normalized RMSE).
- P3-02 Task 3: cross-fitted oracle ensemble (CNN / foundation-encoder / structure-aware) with frozen independent oracle; disagreement score tracked.
- P3-02 Task 4: region sensitivity controls (GC-only, length-only, first-nt-only, source-agnostic ablations).

### Minimum effect size
- Sign accuracy ≥ 0.60 and significantly > 0.50 (one-sided binomial, BH-FDR < 0.05) on the family-disjoint test set.
- Top-10% predicted edits enriched ≥ 1.5× for measured/cross-fitted beneficial edits.
- Delta Spearman ≥ 0.30 on family-disjoint test.

### Statistical method
- Binomial test for sign accuracy vs 0.50; hypergeometric/Fisher exact for top-k enrichment; bootstrap CIs over source groups (group-aware resampling, never over individual candidates).

### GO
All three thresholds met on family-disjoint test **and** sign direction agrees between training oracle and frozen independent oracle on shared candidates.

### PARTIAL
Thresholds met in only one region, or train/independent oracle agreement is limited → shrink task scope; do not enter full joint RL.

### NO-GO
Sign accuracy ≈ random, top-k enrichment absent, or independent-oracle direction frequently reverses → **pause all RL**; acquire Level-C intervention data before any policy work.

---

## H3 — Optimization headroom (optimization headroom)

**Statement.** Strong search over the legal neighborhood finds candidates that beat the source on an **independent** oracle or measured assay — the task contains exploitable headroom.

### Supporting evidence
- UTR-LM's best designs beat mature therapeutic UTRs prospectively.
- LinearDesign wet-lab gains from synonymous recoding.
- GEMORNA/mRNAutilus show large full-redesign headroom (de novo upper bound).

### Opposing evidence
- P2-05 GRPO pilot: mean return 0.0274, improvement CI [+0.0015, +0.0076] — headroom on the then-current oracle was tiny.
- Headroom on the training oracle may not transfer to independent oracles (reward hacking).
- Headroom may concentrate in a few cargo families, limiting generality.

### Validation experiment
- P3-02 Task 5: on frozen validation/test sources — exact one-edit enumeration, exact two-edit enumeration where tractable, greedy, beam search, simulated annealing, MCTS, oracle-guided local search; report per-region and joint headroom, fraction of sources with positive candidates, and family concentration.
- Candidates confirmed on cross-fitted independent oracle; stratified controls for GC content and length.

### Minimum effect size
- ≥30% of test sources admit a legal candidate with positive delta confirmed on the independent oracle.
- Median best-candidate improvement ≥ 5% predicted protein-output proxy (internal proxy claim level C2/C3 only).
- Gains remain after GC/length stratified controls (not fully explained by a single heuristic).

### Statistical method
- Bootstrap CIs over source groups; regression of gain on GC/length covariates to test heuristic confounding; paired comparison of joint vs best single-region search.

### GO
Headroom confirmed on independent oracle in ≥1 region, broadly distributed across families.

### PARTIAL
Headroom confined to few cargos or one region → shrink task; no full joint RL.

### NO-GO
Even strong search cannot stably beat the source on the independent oracle → **pause all RL**; the task lacks optimizable structure at current oracle quality; improve oracles/data first.

---

## H4 — RL quality/amortization value (RL quality/amortization value)

**Statement.** A learned policy provides either (H4-A) higher quality than the best strong-search baseline at equal inference-time query budget, or (H4-B) approximately equal quality at substantially lower inference cost (amortization).

### Supporting evidence
- P3-00-RL correctness gate (12/12) established a formally correct GRPO production path — the tool is now valid to test.
- Sequential minimal-edit MDP with budgets ≤10 is a natural fit for amortized policies; CTMC action semantics support principled sampling.

### Opposing evidence
- P2-05 effect sizes were tiny; P2-01 synergy BORDERLINE.
- With edit budget 1–3, exact enumeration is cheap — RL has no role where enumeration is tractable.
- Ranker + limited search may already sit near the optimum (P3-07 Route C).

### Validation experiment
- P3-07: query budgets {32, 128, 512, 2048} × edit budgets {1, 3, 5, 10}; baselines {random legal, best single edit, greedy, Stage-B ranker, beam, SA, MCTS, oracle-guided local search, DAgger ranker ± limited search}; exact one/two-edit optima and tiny-MDP DP for optimality-gap accounting; full cost report (training oracle calls, training compute, inference calls, break-even cargo count).

### Minimum effect size
- H4-A (quality): RL ≥ best search baseline by ≥5% relative at matched 512-query budget, paired bootstrap 95% CI over sources excludes 0, across ≥3 seeds.
- H4-B (amortization): RL within 2% of best-search quality with ≥10× fewer inference oracle calls, and break-even deployment scale ≤ a pre-registered cargo count.

### Statistical method
- Paired bootstrap over source groups; per-seed replication; pre-registered budgets — no post-hoc budget selection (global constraint #17).

### GO
H4-A holds → P3-08 with "RL discovers better designs" framing (Route A). Or only H4-B holds → P3-08 with "RL amortizes expensive constrained optimization" framing (Route B).

### PARTIAL
Amortization only, or quality gain confined to large edit budgets → narrow the claim accordingly.

### NO-GO
Ranker + limited search is near-optimal and RL shows no quality or cost advantage (Route C) → **stop expanding GRPO**; project proceeds as local-delta prediction + benchmark + strong constrained search + prospective validation (still a complete paper). This is an RL-specific stop, not a project stop.

---

## H5 — Cross-region synergy (cross-region synergy) — OPTIONAL UNLOCK

**Statement.** Joint 5′UTR+CDS editing yields super-additive gains: `Δjoint > Δ5′UTR + ΔCDS`, replicating on independent oracles and prospective assay.

### Supporting evidence
- P2-01 (MultiRegionOracle v2): BORDERLINE positive interaction, d = +0.371, p < 1e-29.
- PERSIST-seq: UTR, CDS structure, stability, and protein output interact; in-cell stability is a major driver of output.

### Opposing evidence
- P2-01 was BORDERLINE, not PASS; the interaction may be a training-oracle artifact.
- Additivity is common in reporter systems; super-additivity is the exception.

### Validation experiment
- P3-10: counterfactual panel — 1000 wild-type sources × 5 arms {WT, single-5′UTR, single-CDS, single-3′UTR (exploratory only), joint} on ≥2 independent oracles, plus a prospective assay arm; pre-registered interaction contrast.

### Minimum effect size
- Interaction effect |d| ≥ 0.2 with consistent sign across independent oracles; prospective arm direction-consistent.

### Statistical method
- Pre-registered interaction contrast in a linear/mixed model; bootstrap over sources; cross-oracle sign consistency test.

### GO
Interaction replicates on ≥2 independent oracles **and** prospective panel → C6 claim allowed.

### PARTIAL
Interaction on training oracle only → report as internal-proxy observation (C2), no mechanism claim.

### NO-GO
Joint ≤ best single-region → drop synergy narrative; project remains single-region or additive joint. Core paper unaffected.

---

## H6 — Full-transcript extension value (full-transcript extension value) — OPTIONAL UNLOCK

**Statement.** Adding 3′UTR editing to the primary task improves held-out objectives, justified only after a 3′UTR local-delta oracle meets H2-level metrics and joint-policy gains transfer (H5 GO).

### Supporting evidence
- Large-scale 3′UTR variant screens: single variants measurably alter mRNA abundance.
- PERSIST-seq: in-cell stability (partly 3′UTR-driven) is a major protein-output driver.

### Opposing evidence
- Current project reward sensitivity is weakest for 3′UTR (P3-00 §2.3).
- No source-matched local 3′UTR edit labels in project assets.
- 3′UTR effects are highly cell-type-specific (RBP/miRNA/splicing risk); larger action space raises reward-hacking risk.

### Validation experiment
- Gated two-step: (1) train/evaluate a 3′UTR local-delta oracle against H2 thresholds (sign accuracy ≥ 0.60, enrichment ≥ 1.5×, Spearman ≥ 0.30 on family-disjoint test); (2) if and only if step 1 passes, compare Task C policy vs Task C+3′UTR policy on held-out sources with independent-oracle confirmation.

### Minimum effect size
- Step 1: H2 thresholds met for 3′UTR deltas.
- Step 2: ≥5% held-out objective improvement over the Task C policy, confirmed on the independent oracle.

### Statistical method
- As H2 (step 1); paired bootstrap over sources with independent-oracle confirmation (step 2).

### GO
Both steps pass → unlock `three_utr_status: locked_extension → enabled` via a new contract version (not a silent edit).

### PARTIAL
Step 1 passes, step 2 fails → 3′UTR remains a prediction-side result, not an editing action.

### NO-GO
Step 1 fails → 3′UTR permanently excluded from the primary task; documented as a negative result (global constraint #16).

---

## RL pause conditions (summary — what stops RL)

| Trigger | Scope | Action |
|---|---|---|
| H1 NO-GO: best edits indistinguishable from random edits | all RL | pause RL; acquire Level-C intervention data |
| H2 NO-GO: sign accuracy ≈ random / no enrichment / oracle reversal | all RL | pause RL; retrain/rebuild oracles and data |
| H3 NO-GO: strong search cannot beat source on independent oracle | all RL | pause RL; task lacks headroom at current oracle quality |
| H4 NO-GO (Route C): ranker+search near-optimal, no RL advantage | RL only | stop GRPO expansion; no-RL paper route |
| P3-03 adversarial arm exposes reward hacking (high training-reward candidates fail prospectively) | all RL | pause RL; fix reward/oracle before any scale-up |
| H5 NO-GO | synergy claim only | drop synergy narrative; RL may continue single-region |
| H6 NO-GO | 3′UTR extension only | 3′UTR excluded; RL unaffected |

**Non-negotiable**: full-transcript synergy (H5) and 3′UTR extension (H6) are unlock hypotheses. They are never assumed true by default, and their failure never blocks the core minimal-edit paper.
