# DeltaBench: 14 Model Families × 13 Tasks for mRNA Edit-Effect Deltas — and Why Absolute Scores Do Not Transfer

*Draft v1 (full text). Source-relative benchmark + systematic failure-attribution analysis.*

> **Title note.** Candidate **T2** is recommended: it carries both the quantitative subject (14 model families × 13 tasks) and the mechanism hook ("why absolute scores do not transfer"). Alternatives retained for a venue-specific final choice: **T1** — "DeltaBench: A Source-Relative Benchmark for Predicting the Effects of mRNA Sequence Edits" (benchmark-primary, flat); **T3** — "The Limits of Delta: A Source-Relative Benchmark and Systematic Failure-Attribution Analysis for mRNA Edit-Effect Prediction" (mechanism-track framing). Final selection is deferred to the PI.

> **Scope and discipline statement.** Every number in this manuscript is drawn from frozen, pre-registered experiments evaluated on **VALIDATION splits only**; **protected TEST reads = 0**. All matrix readouts use the frozen-Δ (zero-fine-tuning) caliber. Failure verdicts follow pre-registered decision rules that were frozen before computation. Where a number is registered but not yet finalized, a PENDING marker is used and no number is invented.

---

## Abstract

mRNA edit-prioritization tasks, from variant ranking to generative design guidance, ultimately consume the *ranking of differences between sequences* (Δ), yet the field's leaderboards predominantly score absolute prediction accuracy — a capability that is not the one downstream users need. We provide a textbook-scale decoupling: a frozen Optimus model reaches ρ_abs = 0.873 on its absolute MRL task while its source-relative delta correlation collapses to ρ_delta = 0.313, with error correlation ρ_ε = 0.720, showing that absolute ranking ability and delta ranking ability are distinct competencies that today's benchmarks do not separate. We deliver **DeltaBench**, a source-relative benchmark built from a canonical record abstraction (source sequence / candidate sequence / `direction_normalized_delta` / `source_group_id`) with an explicit dual task (absolute and frozen-Δ reported side by side), 13 evaluation cells (9 existing VALIDATION tasks plus three append-only rows M1/M6/S1), and a 14-family frozen-Δ matrix comprising 65 new cells that are 65/65 independently re-checked to floating-point-identical precision. The central matrix reading is strong in-domain but weak cross-library: UTR-STCNet reaches 0.8144 on the MRL row (GSE114002) but only 0.0672 on the cross-library M1 row (GSE232927), and all 730/730 evaluation sequences for the in-domain cell are empirically contained in that model family's training corpus; on the three appended rows the five newly ported families collectively stay below 0.09. Our mechanism analysis — the analytical core of the paper — proceeds in three rings: (i) differential validity, decomposed into five quantitative error sources together with an oracle probe arm that is decisive (MRL NO_SIGNAL; polyA HEAD/EQUIVALENT); (ii) a two-factor regularity in which the single-factor density hypothesis **fails** its held-out test (50.77% < 70% gate, honestly downgraded to a task-specific phenomenon) while the two-factor grouping of deviation by supervision regime is supported (EXPLORATORY), backed by an intervention triangle (synthetic density increase falsified, G1 FAIL gap 0.8235; parameter-side half-effect anchored at the first-order bound, ERK 0.2015 ≈ 0.207; same-domain real-data ladder fully effective, 0.1987 → 0.3158 statistically tied with the frozen reference 0.3132, while a pre-registered cross-library real-data arm fails its direction gate 0.2739 vs the 0.3158+0.01 band, bounding the density factor's independent contribution); and (iii) a first-order / structural decomposition plus a four-class failure taxonomy. Finally, polyA is the sole task in which the ceiling ruler becomes a non-trivial reading: V5 reaches 0.8219, i.e. 91.3% of the ICC-0.90 label ceiling, with Holm p = 0.004 as the only family-wise significant win in the whole nine-task family, and with the decision-caliber result reported honestly as mixed; this positive result is paired with a model-selection taxonomy (C1 physical / C2 paradigm / C3 geometric / C4 supervision / C5 control-regime) and a "data-side triple screening" (ICC / density / corpus) that should precede architecture selection.

**Keywords:** mRNA edit-effect prediction; source-relative benchmarking; delta learning; failure attribution; supervision regime; data geometry.

---

## §1 Introduction

### 1.1 The edit-prioritization task and the need for Δ ranking

Edit-effect prediction models are increasingly deployed as *prioritizers*: given a source sequence and a set of candidate edits, downstream pipelines — variant-priority triage, guide construction for generative design — consume the *ordering of candidate differences*, i.e. the rank of Δy = y(candidate) − y(source), not the absolute activity of any single sequence. The two quantities are, however, routinely conflated. Our opening case makes the distinction concrete. For the MRL task (GSE114002) a frozen Optimus model achieves ρ_abs = 0.8733 on the absolute endpoint while its delta correlation is only ρ_delta = 0.3132 and its cross-arm error correlation is ρ_ε = 0.7196. The model "reads absolute values" well and "reads differences" poorly; these are two competencies, and a leaderboard that reports only the first does not measure the second.

### 1.2 The gap in existing benchmarks

Current absolute-endpoint leaderboards do not distinguish the two capabilities, and no widely used benchmark formalizes the source-relative quantity at all. Three concrete deficiencies follow. First, there is no source-relative formalization: without a `delta_hat = pred(candidate) − pred(source)` definition anchored in a canonical paired record, absolute and delta abilities cannot be separated. Second, there is no controlled design over supervision regime × data geometry: in-domain supervised rows and out-of-domain rows are interspersed in a single table, so a reading such as UTR-STCNet's 0.8144 on MRL — a number that partly reflects corpus-level memorization, since 730/730 evaluation sequences fall inside its training corpus — is invisible as a *regime* property on conventional leaderboards. Third, there is no failure-attribution discipline: without pre-registered decisions, a high absolute score is silently extrapolated into an assumption that deltas are usable.

### 1.3 Contributions

We make four contributions.

**C1 — Benchmark v2 definition.** A canonical record, a dual task (absolute and frozen-Δ), 13 evaluation cells (9 existing tasks plus three append-only rows), and a frozen-Δ evaluation protocol with a two-tier adjudication discipline and ceiling normalization.

**C2 — The 14-family × 13-cell matrix.** Five new families are ported (each passing a family unit test and with all input-adaptation decisions fully declared) alongside 9 existing reference families; the resulting matrix contains 65 newly computed cells, and an independent re-check script recomputes all of them from archived predictions at floating-point-identical precision (65/65).

**C3 — A three-ring attribution framework, including the two-factor regularity.** Ring 0, differential validity (five error sources plus a decisive oracle arm); then the two-factor regularity — an observation, a held-out failure that is honestly downgraded, and an intervention triangle supporting ρ ≈ f(supervision regime × data geometry) at EXPLORATORY grade; then a first-order/structural decomposition and a four-class failure taxonomy.

**C4 — The polyA positive result pushed to 91.3% of ceiling.** On the single task (of 13) whose label ICC is 0.90, V5 reaches 0.8219, i.e. 91.3% ceiling completion, closing more than half (0.0876/0.1657 = 52.9%) of the remaining learnable space, with Holm p = 0.004 as the only family-wise significant win, and with the decision-caliber result reported honestly as mixed (attributed to supervision regime). Mechanism attribution (first-order 55.4% plus combined signal plus a multi-task isolation causal chain) and two application outlets (generation-line guidance and committee routing with a CI excluding zero) complete a full positive-result chapter.

### 1.4 Structure of the paper

Section 2 defines the benchmark (canonical record, 13 tasks, four-axis differentiation, evaluation protocol, new-row discipline). Section 3 presents the matrix and its two key readings (in-domain strong / cross-library weak; new rows collectively near zero). Section 4 explains why absolute scores do not transfer (Ring 0 differential validity). Section 5 develops the two-factor regularity and the intervention triangle. Section 6 gives the first-order/structural decomposition. Section 7 presents the failure taxonomy. Section 8 is the polyA positive-result chapter. Section 9 discusses model selection and limitations. Section 10 details the methods.

---

## §2 The Benchmark

### 2.1 Definition and the canonical record

The unit of the benchmark is a *paired record* over a (source, candidate) pair. Each record carries a fourteen-field schema including `source_sequence`, `candidate_sequence`, `source_relative_edits`, `direction_normalized_delta`, `source_group_id`, `endpoint_descriptor`, `biological_context`, `eval_split_status`, and `provenance`. The source-relative quantity is defined as Δy = y(candidate) − y(source) at the record level (LEVEL_DIFFERENCE caliber). Records are grouped by `source_group_id` so that within-source calibers can be computed in addition to the overall caliber. **Dual task declaration:** the absolute task (ρ_abs) and the frozen-Δ task (ρ_delta) are reported as separate columns computed on the same evaluation surface, never merged.

### 2.2 The 13 tasks (9 existing + 3 new rows; S1 two arms)

**Nine existing VALIDATION tasks.** MRL (GSE114002, n = 730); polyA (GSE269595, n = 2628); MPRAU (ENCSR854RUF, n = 12048); HL5 (GSE217518, n = 400); HL3 (same library, n = 503); TE200304 (n = 1614); TE149487 (n = 48); RNA149487 (n = 48); REFALT (GSE186455, n = 274).

**Three append-only new rows (seed frozen 2026-09-20).** **M1** — an MRL row (GSE232927, Castillo-Hair 2024): 2,805 rows over 2,801 sources across three cellular contexts (HEPG2 800 + T_CELL 1600 + HEK293T 405), candidate-per-source density 1.0014. **M6** — an NDD 5′UTR row (GSE246381, Plassmeyer 2025): 800 rows over 509 family sources; its seal/unseal history is recorded in provenance and `historical_exposure` is reported per contract. **S1** — a stability row (Su 2025, eLife): 5,572 sub-rows split into two assay arms (SH 2,792 / HEK 2,780), density 3.67 over 1,519 sources, with 1,330 protected records excluded and a Dao cryptic-splicing QC flag-keep of 834/1,871.

**Table 1. Benchmark composition (13 evaluation cells).**

| Cell | Source / study | n (VALIDATION) | Notes |
|---|---|---|---|
| MRL | GSE114002 | 730 | in-domain reference row |
| polyA | GSE269595 | 2,628 | 78 source groups; ICC 0.90 |
| MPRAU | ENCSR854RUF | 12,048 | pair-mean caliber, 2,008 variants |
| HL5 | GSE217518 | 400 | ICC 0.0014 |
| HL3 | same library | 503 | ICC 0.013 |
| TE200304 | — | 1,614 | ICC 0.586 |
| TE149487 | — | 48 | ICC −0.274 |
| RNA149487 | — | 48 | ICC 0.364 |
| REFALT | GSE186455 | 274 | ICC 0.21 |
| M1 | GSE232927 | 2,805 | 2,801 sources; density 1.0014; 3 contexts |
| M6 | GSE246381 | 800 | 509 family sources |
| S1 (SH / HEK) | Su 2025 eLife | 5,572 sub-rows | density 3.67; 1,330 excluded |

### 2.3 Four-axis differentiation

Table 2 summarizes the four axes that distinguish DeltaBench from conventional absolute-endpoint benchmarks, together with a leakage/rights boundary column.

**Table 2. Four-axis differentiation.**

| Axis | DeltaBench | Conventional absolute leaderboards |
|---|---|---|
| ① Source-relative formalization | Δ defined at the record level (source/candidate paired record) | endpoint score only |
| ② Dual-task reporting | absolute (ρ_abs) and frozen-Δ (ρ_delta) reported side by side | single absolute score |
| ③ Ceiling normalization | ICC task ceiling + explicit N/A handling | none; scores read without a noise ceiling |
| ④ Two-tier adjudication discipline | tier-1 small-calibration falsification right + tier-2 full-set verdict | none |
| Leakage / rights boundary | pigeonhole audit + study-specific rights review | typically undeclared |

### 2.4 Evaluation protocol

**Frozen-Δ caliber.** Each ported family is run with its official published weights, with no fine-tuning, no retraining, no gradient updates, and no output post-calibration; only input adaptation and dtype/device movement are permitted. All per-family input adaptations are declared exhaustively: GEMORNA uses a dual 5′/3′ head; HydraRNA uses frozen embeddings read out at the first coordinate; STCNet uses a padd-120 one-hot port; UTR-Insight uses a 50-nt padded prefix with the GSM3130435 branch; LAMAR uses the EsmTokenizer with 1026 truncation.

**Two-tier adjudication.** A tier-1 small-calibration set (e.g. polyA `calib100`) may raise a signal, but that signal must be re-checked on the tier-2 full set (891 sources) before it is credited; the discipline case is the Optimus calib100 signal being falsified by the 891-source full set.

**Budget parity (fairness declaration).** All matrix rows are scored under the same information budget (VALIDATION-only reading, one frozen evaluator instance, identical record sets), the same split discipline (the frozen near-duplicate-component manifest), and the same tuning budget — zero tuning: official published weights, no per-row hyper-parameter search, no output post-calibration. For the project's own in-domain rows, hyper-parameters and training budgets are frozen in the pre-registration documents (§10.4) and are never re-searched per task; for the ported families, no matched fine-tuning was performed, and the "you under-tuned the external models" objection is answered by the pre-registered matched-FT control rather than by ad-hoc tuning (MPRAU matched-FT arms for UTR-LM / RNA-FM: three seeds, all negative, the R5 closure). Budget asymmetry is declared where it genuinely exists: APARENT's decision-caliber advantage on polyA traces to its 2.74M within-assay training corpus — a supervision-regime property reported openly in §8.2 — not to an evaluation-budget difference.

**Ceiling normalization.** Each task has a label ICC ceiling: MRL 0.83, polyA 0.90, MPRAU 0.683, TE200304 0.586, HL5 0.0014, HL3 0.013 (and TE149487 −0.274, RNA149487 0.364, REFALT 0.21 in the full table). Rows whose ICC is ≈ 0 are assigned normalization **N/A**, so that negative denominators are never used to manufacture inflated readings. The unified source for these ICC values is the label-ICC table (`mechanism_results_v2.json → label_icc_reference`); the GSE149487/GSE200304 entries additionally have an independent recomputation cross-check.

**Table 3. Label ICC ceilings.**

| Task | ICC | Task | ICC |
|---|---|---|---|
| MRL | 0.83 | HL5 | 0.0014 |
| polyA | 0.90 | HL3 | 0.013 |
| MPRAU | 0.683 | TE149487 | −0.274 |
| TE200304 | 0.586 | RNA149487 | 0.364 |
| REFALT | 0.21 | | |

### 2.5 New-row construction discipline

New rows are append-only (they do not enter any existing manifest), with construction seeds frozen. A pigeonhole leakage audit returns zero protected overlap for M1/M6/S1 (M1: 4 TRAIN-overlapping sequences flagged `keep=0` and thus naturally excluded; M6: zero hits at both locus and sequence level; S1: zero after the 1,330-record hard exclusion). TRAIN overlaps are only *marked*, not removed, for the existing rows, while the four M1 TRAIN overlaps fall out by construction. Extreme values are preserved as-is (e.g. the S1 t05 anomalous magnitude; robust statistics are recommended when reading S1).

---

## §3 Results: the matrix

### 3.1 The 14 × 13 matrix

The matrix comprises five newly ported families measured across 13 cells plus 9 existing reference families taken as archived `ALREADY_DONE` rows. **Figure 1** renders the complete 65-cell frozen-Δ block (the five new families × all 13 task cells; every cell is the archived `task_macro_spearman`, no recomputation). Table 4 collects the key newly computed readouts with exact values.

**Table 4. Key frozen-Δ matrix readouts (VALIDATION).**

| Cell / family | Value | Note |
|---|---|---|
| UTR-STCNet, MRL row | 0.8144 | in-domain; 730/730 eval sequences inside training corpus |
| UTR-STCNet, M1 row | 0.0672 | cross-library (same MRL domain, different study) |
| GEMORNA, M1 row | 0.0889 | maximum over the five new families on M1 |
| UTR-Insight, MRL row | 0.2739 | same-library weak-supervision control |
| frozen-Optimus, MRL row | 0.3132 | supervision-regime control (v1 reference) |
| M1 row, five new families | 0.0226 / −0.0215 / 0.0889 / 0.0672 / 0.0575 | all < 0.09 |
| M6 row, max over five new families | 0.0464 | all < 0.09 |
| S1 row, max over five new families (both arms) | 0.0450 | all < 0.09 |

By family, UTR-STCNet lands 4/13 cells in-band (with extreme deviations in both directions), while the remaining families land 7–8/13 cells in-band; this per-family count bridges to the density analysis of §5.

### 3.2 Re-check declaration

An independent re-check script, which does **not** read the archived metric values, recomputes each of the 65 cells directly from `predictions.jsonl` (137,350 rows). All 65/65 cells pass, with a whole-matrix max |Δ| = 0.0 (floating-point-exact), including the extreme STCNet MRL cell (0.814430506981109). This establishes that the reported metrics are exactly reproducible from the archived predictions.

### 3.3 Key reading I: strong in-domain, weak cross-library

UTR-STCNet achieves 0.8144 on the in-domain MRL row (GSE114002, n = 730) but only 0.0672 on the cross-library M1 row (GSE232927, n = 2,805) — even though both rows live in the same MRL domain. The cross-library reading falls back to the general-family level. Crucially, 730/730 evaluation sequences (667 train_val + 63 test) for the in-domain cell are empirically contained within that family's MPRA-H training corpus, so part of the in-domain supervised gain is bound to corpus-level memorization. A same-library weak-supervision control (UTR-Insight, random 50-nt library) reaches only 0.2739 on the same cell, isolating the effect of supervision *density* from that of domain membership.

### 3.4 Key reading II: the appended rows are collectively near zero

On the three appended rows the five new families stay below 0.09 (M1 max 0.0889, GEMORNA; M6 max 0.0464; S1 max 0.0450). The correct reading is that model families **without same-distribution dense supervision** are collectively constrained on the new rows; the frozen-Optimus MRL reference (0.3132) serves as a supervision-regime control. This must not be phrased as "all baselines fail to predict deltas": the phenomenon is a property of the supervision regime, and Optimus/APARENT are the regime contrast rows. A refinement from the per-cell power analysis is recorded in §9: a majority of the 15 new-row cells are near zero, but on the M1 row GEMORNA (ρ = 0.0889, CI [0.052, 0.127]) and one further 5′UTR family carry a small yet significant positive signal.

### 3.5 Batch-2 generalist backbones: the same regularity on a second architecture cohort (append-only rows)

To test whether the §5 two-factor regularity is an artifact of one architecture cohort, we appended a second, structurally diverse batch of generalist backbones under the frozen probe protocol (frozen backbone + linear probe on embedding differences, TRAIN fit / VALIDATION eval, protected TEST reads = 0, pre-registered in `benchmark_v2_generalist_rows_amendment_v2.md`): Orthrus 4-track and 6-track (Mamba/SSM, 10.2M; the 6-track input uses the official CDS-frame + splice-site channels, zeros for UTR-only sequences), CodonFM-80M (codon-level encoder), mRNA-LM 5′UTR and 3′UTR (Sanofi official checkpoints, WordLevel character tokenizer per the official `OneModel.py` recipe), CaLM (codon LM, multimolecule re-export with provenance recorded), and LucaOne (1.58B gene–protein pretrained, official FTP checkpoint, byte-exact sha256). All seven adapters were smoke-tested (single-sequence forward), and the runner glue was port-validated against the archived RNA-FM MRL row (|Δ| = 2.4×10⁻⁶ ≤ 10⁻⁵).

**Table 4b. Batch-2 frozen-Δ readouts, 7 backbones × 9 Development tasks (task-macro Spearman, VALIDATION).**

| Backbone | MRL | polyA | MPRAU | TE-3′ | PLUMAGE-TE (LOSO) | PLUMAGE-RNA (LOSO) | REF/ALT | HL-5′ | HL-3′ |
|---|---|---|---|---|---|---|---|---|---|
| Orthrus 4-track | +0.033 | +0.467 | −0.005 | −0.004 | +0.017 | −0.253 | −0.083 | −0.016 | −0.018 |
| Orthrus 6-track | +0.203 | +0.773 | +0.050 | +0.027 | −0.113 | +0.052 | −0.083 | −0.052 | −0.016 |
| CodonFM-80M | +0.154 | +0.695 | +0.054 | +0.046 | −0.125 | +0.162 | +0.007 | −0.004 | −0.047 |
| mRNA-LM 5′UTR | +0.100 | +0.752 | +0.077 | +0.034 | +0.150 | −0.361 | −0.016 | −0.050 | −0.021 |
| mRNA-LM 3′UTR | +0.172 | +0.788 | +0.087 | +0.030 | −0.102 | −0.001 | −0.051 | −0.055 | −0.007 |
| CaLM | +0.080 | +0.682 | +0.006 | +0.015 | −0.084 | −0.184 | −0.013 | −0.013 | −0.002 |
| LucaOne (1.58B) | +0.126 | +0.770 | +0.041 | +0.039 | −0.057 | +0.138 | +0.014 | +0.004 | +0.003 |

Readings, with the same discipline as the batch-0/1/3 rows (no peak-picking, no significance claims on n = 48 LOSO cells):

1. **polyA is again the only high band.** All seven backbones land in +0.467…+0.788 on the high-ceiling × high-density polyA task, versus |ρ| ≤ 0.09 elsewhere except the two MRL rows noted below. The batch-0/1/3 generalist band is reproduced without exception on a second, structurally diverse cohort.
2. **Half-life cells are 14/14 near zero.** The label-ICC ≈ 0 physical-unlearnability reading extends to Mamba/SSM, codon-level, and gene–protein pretrained backbones.
3. **LOSO sign flips recur.** PLUMAGE-RNA spans −0.361…+0.138 across backbones (n = 48 per cell), the same backbone/seed sensitivity documented for the batch-3 rows; we register the extremes as noteworthy cells without significance claims.
4. **No backbone-scale effect.** LucaOne (1.58B) shows the highest 9-cell mean (+0.120) but no cell outside the cohort's own band — capacity does not buy Δ capability.
5. **Architecture-internal control.** Orthrus 4-track → 6-track moves polyA +0.467 → +0.773 (the official CDS-frame/splice channels carry polyA-relevant signal) and MRL +0.033 → +0.203, the only visible MRL lift in this cohort; this within-family input-representation contrast is consistent with the §6 finding that polyA's learnable component is partly first-order positional/table signal.

Sources (append-only, numbers copied verbatim from frozen artifacts): `analysis_generalist_rows_v2/{batch2a_orthrus_v1, batch2b_codonmrna_v1, batch2b2_mrnalm_v1, batch2c_calm_v1, batch2d_lucaone_v1}/` and the reporting-only aggregation `batch2_matrix_v1.json`. One incident is disclosed for reproducibility: the initial mRNA-LM runs produced degenerate embeddings (a WordLevel tokenizer with whitespace pre-tokenization collapsed each unspaced sequence into a single `[UNK]` token, Spearman = None with zero prediction variance); the fix — space-joined character input per the official recipe — was verified by a non-degeneracy test (pairwise embedding distances 18.05–28.17) before the clean rerun. The failed run's artifacts were removed; the matrix above is the post-fix rerun.

---

## §4 Why absolute scores do not transfer (Ring 0: differential validity)

### 4.1 Mathematical decomposition

Under a bounded linear read-out, Δŷ = w·Δh, and hence Δŷ − Δy = ε_c − ε_s: the differential error is exactly the *difference* of the two arms' errors, with the shared bias cancelling. This yields a favorable condition (ρ_ε → 1, errors cancel) and a catastrophic condition (ρ_ε → 0, or an error-difference term uncorrelated with the target). Absolute accuracy constrains neither: a model can be accurate on levels while its arm-errors are uncorrelated in the difference.

### 4.2 Five error sources

**(1) Error-correlation triples.** Optimus MRL shows 0.8733 / 0.3132 / 0.7196: partial cancellation, but insufficient. In "fully blind" rows, high ρ_ε coexists with no signal — LAMAR HL5 gives 0.0207 / 0.0441 / 0.6863, and HL3 gives −0.0911 / 0.0184 / 0.4429 — i.e. shared error with no shared signal.

**(2) Overall vs within-source divergence.** For polyA the overall caliber is 0.71–0.75 while the within-source caliber is only 0.19–0.23 (UTR-LM 0.7490 → 0.1934; RNA-FM 0.7114 → 0.2335 over 78 groups). The overall figure is inflated by between-source composition, whereas edit prioritization needs the within-source caliber.

**(3) "Broad and shallow" probe geometry.** Cosine alignment is 0.79/0.81 while RNA-FM's participation ratio is 258.7, top-10 share is 0.0955, and the per-dimension gain is 0.0189 — the representation is broad but shallow (SNR proxy under an isotropy assumption; the 16 rows of unarchived probe weights are registered but not enumerated).

**(4) Oracle arm (decisive).** On MRL the oracle probe is NO_SIGNAL: RNA-FM three-seed mean 0.0869 (per-seed 0.1369 / 0.0640 / 0.0597, the seed-sensitive pseudo-signal shown in full) and UTR-LM 0.0700 — the representation does not contain delta signal, and changing the supervision form does not help. On polyA the probe is HEAD/EQUIVALENT (RNA-FM +0.0070, UTR-LM −0.0054): the signal genuinely exists in the representation and is already delivered.

**(5) Context mismatch and label noise.** RiboNN shows four near-zero rows (−0.0106 / −0.0290 / +0.0778 / −0.0450; a full-length TE model applied to fragments). On the data side, GSE149487-TE has ICC −0.274 with a split-half of 0.050, closing the lower bound; the reverse case is GSE200304 (ICC 0.586 is measurable while the strongest delta model reads ≤ 0.011), locating the deficit on the representation side.

A consistency re-check recomputes all ρ_delta values at 30/30 rows with |Δ| ≤ 1.1e-16.

**Table 5. Five error sources (representative values).**

| Source | Mechanism | Representative values |
|---|---|---|
| Error-correlation triple | shared bias, insufficient cancellation | Optimus 0.8733 / 0.3132 / 0.7196 |
| Blind rows | high ρ_ε, no signal | LAMAR HL5 0.0207/0.0441/0.6863; HL3 −0.0911/0.0184/0.4429 |
| Overall vs within | between-source inflation | UTR-LM 0.7490→0.1934; RNA-FM 0.7114→0.2335 (78 groups) |
| Broad-and-shallow probe | isotropy-limited SNR | RNA-FM PR 258.7; top-10 0.0955; cos 0.8102; per-dim 0.0189 |
| Oracle (MRL) | NO_SIGNAL | RNA-FM mean 0.0869 (0.1369/0.0640/0.0597); UTR-LM 0.0700 |
| Oracle (polyA) | HEAD/EQUIVALENT | RNA-FM +0.0070; UTR-LM −0.0054 |
| Context mismatch | granularity mismatch | RiboNN −0.0106/−0.0290/+0.0778/−0.0450 |
| Label lower bound | noise floor | ICC −0.274; split-half 0.050 |
| Representation-side reverse | measurable ceiling, no signal | GSE200304 ICC 0.586, strongest Δ ≤ 0.011 |

### 4.3 Protocol discovery (disclosed honestly)

The existing frozen-Δ LM rows are implemented *as* a Δh → Δy regression, which is inconsistent with parts of their docstrings; under a linear read-out the two supervision forms are mathematically equivalent and do not affect any archived value. This is recorded in the protocol note (`results_oracle_probe.json → supervision_form_audit_note`).

### 4.4 Operational boundaries

We summarize five valid conditions, five failure modes, and three operational criteria: (i) use the within-source caliber and inspect the ρ_ε triple first; (ii) declare ICC and context match before modeling; (iii) a high absolute score must not be extrapolated into delta usability.

---

## §5 The two-factor regularity

### 5.1 Observation

Across 25 external rows × 9 task families, log10(density) versus ρ_delta yields a Pearson r = 0.9388 (p = 3.9e-12, n = 25), with a scatter of 25 points and a frozen fit line. The reading is stratified: the five rows with ρ ≥ 0.68 all lie on polyA (density 126.7, a lone order-of-magnitude outlier), while the twenty rows with density ≤ 6.1 all read ≤ 0.32. The correlation is between-strata, not within.

### 5.2 Held-out test (honest downgrade)

Under the frozen pre-registered framework (tolerance band |Δρ| ≤ 0.10; PASS gate ≥ 70%; no row-picking, append-only, no band changes), all 65 new cells are predicted without back-feeding. The result is in-band 33/65 = **50.77% < 70% → verdict FAIL**, and the downgrade clause triggers: the "single-factor density regularity" is downgraded to a **task-specific phenomenon**, the phrase "cross-task universal regularity" is thereafter prohibited, and the only upgrade path is append-only plus re-evaluation at the next decision point. The frozen fit parameters are slope 0.3663 per decade and intercept −0.0857. This subsection is an honest negative result and the disciplining anchor of §5. A dual-caliber sensitivity recomputation (training-pair density caliber) raises the in-band rate from 50.77% to 60.00% (+9.2 pp, six flipped cells all on the new rows); the FAIL verdict is robust to the caliber choice.

**Table 6. Density–learnability: observation and held-out test.**

| Quantity | Value |
|---|---|
| Original correlation | r = 0.9388, p = 3.9e-12, n = 25 |
| Frozen fit | slope 0.3663 / decade; intercept −0.0857 |
| Tolerance / gate | |Δρ| ≤ 0.10; PASS ≥ 70% |
| Held-out verdict | 33/65 = 50.77% → FAIL (downgrade triggered) |
| Dual-caliber sensitivity | 60.00% (+9.2 pp; 6 flipped cells, all new rows) |

### 5.3 Two-factor grouping (EXPLORATORY, not a decision gate)

Grouping by whether the evaluation endpoint domain lies *inside* the model's training corpus: of the 21 below-band cells, **0** are in-domain (all five polyA new-family cells fall below band, with frozen predictions ρ ≈ 0.68 against observed values from −0.37 to +0.31); of the 11 above-band cells, **6** are in-domain (STCNet-MRL 0.8144 exceeds the band by +0.65; four MRL-domain families form a floor on M1). The in-band rate is 3/9 = 33.3% in-domain versus 30/56 = 53.6% out-of-domain. Same-density contrasts sharpen the point: at density 4.9, in-domain dense supervision reads 0.81 versus −0.02 out-of-domain; at density 126.7, polyA reads 0.68–0.75 for the old rows versus 0.15–0.31 for the new rows. The EXPLORATORY conclusion is that deviations group systematically by supervision regime, i.e. ρ = f(supervision regime × data geometry), and that the 25-row fit (r = 0.9388) reflects co-variation of the two factors.

### 5.4 The intervention chain (observation–intervention triangle)

**Synthetic side (falsification).** D16-C synthetically increases density: G1 FAIL (gap 0.8235 ≥ baseline 0.7943); G2 PASS (polyA 0.81338, non-destructive); G4 sc-hit1 = 0 (81/81 support, 0 hits). This falsifies the "increase density and it is fixed" phrasing.

**Parameter side (half-effect).** ERK v2 — the parameter-side upper-bound row, a **1,177-parameter** model — supplies an explicit first-order prior: the gap halves from 0.7943 to 0.3815, and validation reaches 0.2015 — the historical MRL best, sitting at the closed-form first-order + context bound E7 = 0.2069 (a bound-anchored result: the gain is explained by a first-order prior, i.e. source-residual shrinkage rather than new discriminative signal); G4 = 0 (231/231 support, exact match 1); G3 not triggered (partial path). This bound-anchored reading, together with the E7 anchor (0.207) and the three-generation G4 zeros, leaves **no claim of remaining model-side headroom**.

**Real-data side (fully effective).** The W ladder is 0.1987 → 0.2470 (LoRA) → 0.2555 (full-FT, 2 epochs) → 0.3158 (3-seed ensemble) ≈ frozen-Optimus 0.3132. The endpoint is a statistical tie, Δ+0.0027 CI [−0.045, +0.048], crossing zero; this is a statistical tie with a point estimate ahead, and — given the 730-record power — we do not claim a significant surpass (we do not write "no difference").

**Triangle.** The causal carrier of the density correlation is the geometric coverage of real supervision data (H-geometry); H-density is excluded by the synthetic falsification; H-selection is retained as a boundary condition (polyA mechanism locality).

**Table 7. Intervention triangle.**

| Arm | Result | Interpretation |
|---|---|---|
| D16-C (synthetic) | G1 FAIL gap 0.8235 (baseline 0.7943); G2 PASS polyA 0.81338; G4 0 (81/81) | "increase density" falsified |
| ERK v2 (parameter) | gap 0.7943→0.3815; val 0.2015 ≈ E7 0.2069; G4 0 (231/231); G3 not triggered (1,177-parameter upper bound) | half-effect; bound-anchored |
| W ladder (real data) | 0.1987 / 0.2470 / 0.2555 / 0.3158 ≈ 0.3132 | fully effective; tie |
| Tie caliber | Δ+0.0027 CI [−0.045, +0.048] | statistical tie, point estimate ahead |

### 5.5 The three-generation zero chain (G4)

Three successive generations — D15-2 polyA V5 → D16-C probe → ERK v2 — each return graded sc-hit@1 = 0 (support rates 100%, hit 0). This locates the structural-ordering deficit at the data/knowledge level: MRL rank signal is dominated by source means.

### 5.6 M1 intervention arm (result slot)

**Results (frozen harvest, recomputed from archived predictions with bitwise match; `m1_adjudication_v1.json`).**

| Gate | Frozen reading | Verdict |
|---|---|---|
| **G1 direction** | MRL VALIDATION frozen-Δ task-macro ρ = **0.2739**; baseline band 0.3158 + 0.01 = 0.3258; gain = **−0.0419** | **NEGATIVE** (direction not positive) |
| **G2 non-destruction (mechanical)** | polyA VALIDATION recomputation 0.3710 vs frozen V5 main row 0.8219; drop 0.4509 > tolerance 0.02 | **FAIL (mechanical)** |
| **G2 context row (report-only)** | arm 0.3710 vs the Route A V2 baseline recompute 0.0903 on the same caliber: **+0.2807** | not part of the gate verdict (semantics note below) |
| **G3 efficiency (report-only, no threshold)** | 5.03 h wall-clock; 45,828 steps; 977,608 training rows per epoch (677,608 base + 300,000 M1) | reported |
| **G4 mechanism** | not run — prereg executes G4 only when G1 is direction-positive | NOT TRIGGERED |

**Reading (negative direction is the deliverable, stated as such).** Adding a real 300,000-row external-domain MRL corpus on top of the same-domain library does **not** transfer to within-source Δ ordering on the MRL benchmark: the arm lands 0.2739, i.e. 0.0419 below the frozen V2 three-seed ensemble (0.3158) and below all three V2 single-seed values (0.2873 / 0.3157 / 0.3198). Together with the synthetic-side falsification (D16-C, G1 FAIL gap 0.8235) and the parameter-side bound-anchored half-effect (ERK 0.2015 ≈ first-order bound 0.2069), the intervention triangle now reads: **supervision volume/density on the data side does not unlock MRL Δ**, which is consistent with the geometric reading (MRL within-source density 4.9 per source; the learnable part is essentially the first-order table). The single-variable design plus the frozen FINAL-EPOCH(6) rule precludes reading this as a tuning artifact; the pre-registered interpretation framework already classifies a FAIL as evidence that the density factor's independent contribution is bounded.

**Cross-task side signal (report-only, explicitly not a gate).** The same arm evaluated on polyA under the same frozen-Δ caliber reads 0.3710 against its own V2 lineage's 0.0903 — i.e. the added M1 corpus is far from inert; it materially reshapes cross-domain behaviour (a supervision-regime transfer effect). The mechanical G2 verdict is **not** taken from this row: per the amendment, G2 mechanically compares against the V5 multi-task main row, while the arm inherits V2's MRL-only design, so the informative non-destruction reading is this contrast row.

**Declarations.** (i) G1 is a single-seed direction read by prereg; the 3-seed CI branch was never entered because the direction is negative. (ii) No peak-picking: the verdict uses the final epoch only. (iii) Two further seeds (20260904 / 20260905) were trained under the pre-committed compute clause of amendment v2; because G1 is negative they remain **trained-not-analyzed** — archived, excluded from every table, figure and claim, and not used to dilute the G1 reading. (iv) Reporting gap: the archived efficiency block records `peak_cuda_memory_gb = 0.0` — the value was never captured by the runner — so no peak-memory number is claimed here.

**Pre-registered design.** The M1 arm is a mechanism-intervention verification arm (not a SOTA push), pre-registered before launch (freeze commit b2e141cd). It is a single-variable data-side intervention: the MRL delta pipeline is re-trained with real high-density external MRL data (M1, GSE232927 Castillo-Hair 2024) added to the training side, and the question is whether real dense external MRL data improves delta discriminability in the direction predicted by the two-factor regularity. Configuration is cloned from Route A full-FT V2 except for the single data-side increment: backbone mRNABERT + mean-pool linear head, full fine-tuning, base library of 677,608 clean rows, 6 epochs under a fixed FINAL-EPOCH(6) protocol (per-epoch metrics diagnostic only, no peak-picking), batch 128 / lr 2e-5 / weight-decay 1e-4 / AdamW; the increment is a 300,000-row downsample (seed 20260920) of the HepG2 r1 + T-cell r1/r2 defined-end library, giving 977,608 rows per epoch, with joint z-scoring of labels and joint shuffling. Leakage discipline excludes all 4,848 unique evaluation-row sequences and the four TRAIN-overlapping pigeonhole-flagged sequences. The four frozen gates are: **G1** direction gate — single-seed MRL VALIDATION frozen-Δ task macro Spearman, direction positive if gain > +0.01 over the 0.3158 baseline (with the V2 single-seed spectrum, e.g. seed 20260903 = 0.3198, reported as a fluctuation band); **G2** non-destruction gate — polyA main-row recomputation回落 ≤ 0.02 relative to the V5 0.8219 caliber; **G3** efficiency gate (reported, no threshold: wall-clock, peak memory, rows/epoch); **G4** mechanism gate (executed only if G1 is direction-positive) — a directional comparison on the M1 evaluation row against the in-domain/cross-library STCNet phenomenon. The pre-registered interpretation framework treats a FAIL as a valid deliverable (evidence that the density factor's independent contribution is bounded), and the pre-registered contrast anchor is the Route A V2 baseline's M1-row reading 0.0916 (n = 2,805), above the five external families (0.0889 / 0.0672 / 0.0575 / 0.0226 / −0.0215). The arm ran to its frozen terminal state (seed 20260920, 6/6 epochs, FINAL-EPOCH(6)) and was harvested; results below.

### 5.7 Summary

The evidentiary grades are stated explicitly: the single-factor density claim is **downgraded** (held-out FAIL, honestly reported); the two-factor claim is supported by three lines — observation, the intervention triangle, and the held-out deviation pattern — and is labeled EXPLORATORY.

---

## §6 Signal decomposition (first-order / structural)

### 6.1 The first-order closed-form energy table (method)

We replicate E7: a position-block × 64-context design matrix plus a per-source background term b[source], fit by ridge regression in closed form, with the ridge α selected on VALIDATION over the grid {0.3, 1, 3, 10, 30, 100}. The MRL reference pair (0.2069 vs ERK 0.2015) is maintained from the existing archive. A TRAIN-only α sensitivity check (5-fold GroupKFold cross-validation) recomputes the three tasks: polyA Δ = 0.0 (α = 100 in both cases, reading unchanged), MPRAU −0.0003, TE −0.0044 — the decomposition reading is fully robust to α selection, so the optimistic-bias caveat can be withdrawn.

### 6.2 Per-task results

**polyA.** First-order additive table reads 0.4555 = 55.4% of V5's 0.8219. Comparison rows: the externally supervised band 0.71–0.75; APARENT 0.7343; V5 0.8219; ICC 0.90.

**MPRAU.** First-order reads 0.0825 (pair-mean; record-level 0.0630) = 80.5% of V5's 0.1025. The external band is ≈ 0.016 with a CI crossing zero (0.0164); Saluki weak-control 0.1205; ICC 0.683.

**TE.** First-order reads 0.0458 = 79.1% of V5's 0.0579; the external band is 0.0009–0.0113 (0.0061).

**Table 8. First-order decomposition by task.**

| Task | First-order ρ₁ | Ours (V5) | Ratio | External band | ICC |
|---|---|---|---|---|---|
| polyA | 0.4555 | 0.8219 | 55.4% | 0.71–0.75 (APARENT 0.7343) | 0.90 |
| MPRAU | 0.0825 (pair-mean; record 0.0630) | 0.1025 | 80.5% | 0.0164 (CI crossing zero) | 0.683 |
| TE | 0.0458 | 0.0579 | 79.1% | 0.0061 (0.0009–0.0113) | — |
| MRL (reference pair) | 0.2069 | — | — | ERK 0.2015 | 0.83 |

### 6.3 Task heterogeneity (main conclusion)

The decomposition narrative does not transfer across tasks. On polyA the first-order term explains only 0.46, leaving 0.27+ as non-first-order (structural / contextual-composition) signal that external LMs partially capture; on MPRAU and TE our rows are almost entirely first-order signal (V5 increments of only +0.02 and +0.01), so "structural / higher-order gain" must be stated with restraint on these tasks. This is complementary to the G4 structural blindness and the delta-validity within-source finding.

### 6.4 Two-sided reading of external rows

"No dense supervision" ≠ "no signal": polyA external rows (0.71–0.75) generally *exceed* the first-order bound (general-LM pretraining transfer already exceeds the first-order table), whereas MPRAU/TE external rows read ≈ 0 (they do not even capture first-order signal). The per-task edit density is consistent with the first-order coverage gap: polyA averages 8.5 edits/record, the only task with combinatorial space.

### 6.5 Decision table (pre-registered rules)

A = |ρ₁ − band median| ≤ 0.05; B = dense-row − ρ₁ > 0.05. polyA is partially satisfied (✗A 0.2747 / ✓B 0.2788); MPRAU is not established (first-order exceeds the band by 0.0661); TE is partial (B not applicable). The task-level gap is polyA 0.2747 versus MPRAU increment 0.0200 and TE increment 0.0121.

---

## §7 A taxonomy of failure

### 7.1 Classification rules (rules before labels)

**C1 physical failure** — label ICC < 0.1 and a whole-row noise band. **C2 paradigm failure** — input-distribution / context mismatch plus a whole-row |ρ| < 0.05 band. **C3 geometric failure** — density < 10 plus structural evidence. **C4 supervision failure** — no same-distribution corpus plus oracle NO_SIGNAL or an all-zero same-pool result. **C5 non-failure control** — dense-supervision regime rows (≥ 280K).

### 7.2 Main table (20 rows: 14 failure + 6 control; covering 9/9 tasks)

**Table 9. Failure taxonomy.**

| Class | Definition | Rows | Representative values |
|---|---|---|---|
| C1 physical failure | ICC < 0.1 + noise band | 5 | HL5 (ICC 0.0014); HL3 (0.013); TE149487 (ICC −0.274, split-half 0.050); LAMAR cross-evidence 2 (HL5 ρ_ε 0.6863; HL3 0.4429) |
| C2 paradigm failure | input/context mismatch + |ρ| < 0.05 | 5 | RiboNN 3 (−0.0106/−0.0290/+0.0778/−0.0450); Saluki 2 (MPRAU 0.1205, weak-control CI [0.077, 0.164]) |
| C3 geometric failure | density < 10 + structural evidence | 2 | MRL (density 4.9; G4 three-generation 0; E7 0.207); MPRAU (6.1; CI crossing zero) |
| C4 supervision failure | no same-distribution corpus + oracle NO_SIGNAL / all-zero pool | 5 | MRL-oracle NO_SIGNAL (0.0869 / 0.0700); MPRAU; TE; RNA149487 (edge); REFALT (edge) |
| C5 non-failure control | dense supervision regime (≥ 280K) | 6 | Optimus 0.3132; ours 3-seed 0.3158; APARENT 0.7343; APARENT2 0.6810; V5 0.8219; general LM polyA 0.7114–0.7490 |

### 7.3 Task-level roll-up and cross-class annotation

Primary and secondary classes are layered: MRL is C3-primary with C4-secondary (geometry first, representation-side no-signal second). Three edge decisions: GSE149487-RNA → C4-primary with C1-secondary (power-limited); GSE186455 → C4; LAMAR → C1 with ρ_ε cross-evidence.

### 7.4 Control rows and the anti-one-sided narrative

The same general LM that reads 0.71–0.75 on polyA (density 126.7, 2.74M corpus, ICC 0.9) with an EQUIVALENT oracle, reads across zero on low-density tasks without same-distribution corpora (RNA-FM: polyA 0.7114 vs MRL 0.1369 (seed-sensitive) vs HL ≈ 0). The failure attribution therefore lives in the *data regime*, not in an "all general LMs are inadequate" assertion.

### 7.5 Practical implications (toward a model-selection guide)

C2 → align input granularity; C3 → within-source dense candidate supervision with pair-mean/within-source caliber; C4 → run a low-cost oracle probe first (NO_SIGNAL ≤ 0.10 is a stop-loss) before committing, or await corpus, or use end-to-end FT (with the caveat that MPRAU matched-FT is negative across all three seeds, pair-mean −0.075 to −0.107: FT does not rescue it); C5 → find the corpus before discussing architecture.

---

## §8 polyA: from boundary endpoint to a full positive-result chapter

### 8.1 The ceiling ruler

**Task side.** polyA (GSE269595, n = 2,628, 78 source groups) is the only one of the 13 cells with ICC = 0.90 (unified label-ICC source), so its label signal-to-noise ceiling is explicit; the other tasks (ICC 0.001–0.83) do not support the same "approachable ceiling" reading. ICC is a *necessary but not sufficient* upper bound.

**Geometry side.** The edit density is 126.7 candidates/source (the lone order-of-magnitude outlier in the 25-row fit set), with an average of 8.5 edits/record — the only task with combinatorial space among the 13.

**Corpus side.** The APARENT lineage carries a 2.74M within-source dense-supervision corpus (density and in-domain supervision co-vary in the fit set; §5.3's two factors are simultaneously in place on this task).

**Completion reading.** V5 0.8219 / ICC 0.90 = **91.3% completion**; the remaining learnable space is 0.90 − 0.7343 (strongest external row) = 0.1657, of which V5 closes 0.0876 → **more than half (52.9%) of the remaining learnable space**. The external frozen band (0.681–0.749) sits on the same task and band but one tier lower; the first-order closed-form starting point is 0.4555 (first-order-only) / 0.488 (E7-a v1 pos×base).

**Table 10. polyA ceiling accounting.**

| Quantity | Value |
|---|---|
| Label ICC ceiling | 0.90 |
| V5 | 0.8219 = 91.3% of ceiling |
| Strongest external | 0.7343 (APARENT) |
| Remaining learnable space | 0.1657 |
| Closed by V5 | 0.0876 (52.9%) |
| APARENT2 (stronger supervision lineage) | 0.6810 |
| External frozen band | 0.681–0.749 |
| First-order start | 0.4555 / 0.488 |

**Results (frozen harvest; `polya_3seed_harvest_v1.json`).**

| Seed | polyA VALIDATION Spearman | top-1 | NDCG@10 |
|---|---|---|---|
| seed20260907 (main row, unchanged) | 0.8219 | 0.5007 | 0.8710 |
| seed20260921 | 0.8191 | 0.5375 | 0.8820 |
| seed20260922 | 0.8178 | 0.5643 | 0.8831 |
| **new row `polyA-V5-3seed-mean`** | **0.8196** (range [0.8178, 0.8219]) | — | — |

The three-seed mean is 0.819573 with range width 0.0041 — far inside the pre-registered 0.03 variance bound — and the source-group paired bootstrap (2,000 iterations, seed 20260920) gives Δ vs APARENT (recomputed 0.734315) = **+0.085258, CI95 [+0.062098, +0.107759], excluding zero** (cross-check with the frozen protocol seed 20260816: +0.085388, CI [+0.062160, +0.110269]). The Holm family direction recomputes to raw p = 0.0005 → **holm p = 0.002, significant** (the frozen table row stays at 0.001; the recomputation is a direction check, not a replacement). Both pre-registered upgrade conditions are met (mean ≥ 0.80 and CI excluding zero), so the polyA claim is upgraded to a **three-seed robust** reading and this attack surface is closed. Per the prereg the main row is never replaced: 0.8219 remains the frozen headline and `polyA-V5-3seed-mean` is the appended robustness row. The decision-caliber mixed result (§8.2) is unaffected by the supplement and is reported there in the same paragraph as the rank-caliber win.
>
> Pre-registered supplement (mini-prereg, freeze 73d47cdf): the V5 polyA main row is a single-seed frozen terminal state (0.8219). The supplement re-runs the identical V5 configuration with two additional seeds (20260921 / 20260922, giving a three-seed set 20260907 + the two new seeds), with the only change being the seed. The main row 0.8219 is retained unchanged under all circumstances; a new row `polyA-V5-3seed-mean` (with CI and per-seed values) is appended, framed as a robustness reading and never a SOTA push. The reported main judgment is the three-seed mean ± range against the main caliber (GSE269595 VALIDATION n = 2,628 overall Spearman), with a source-group paired bootstrap 95% CI (2,000 iterations, seed 20260920). Upgrade rule: if the three-seed mean is ≥ 0.80 and the CI excludes zero, 0.8219 is upgraded to a "three-seed robust" claim (attack surface SEALED); if the seed variance is large (range > 0.03), the claim is honestly rewritten as an interval statement without selection; a failed seed is recorded as FAILED with a "2-seed + declaration" downgrade, never silently retried with a different seed.

### 8.2 Results (dual caliber reported separately)

**Rank caliber (primary).** V5 0.8219 vs APARENT frozen-delta 0.7343: **Δ+0.0876, bootstrap CI [0.0630, 0.1141], excluding zero, Holm p = 0.004 — the only family-wise significant win across the whole bottom-line nine-task family** (the family is the Task 12.1 adjudication table of nine ours-vs-strongest-external pairings; polyA raw_p = 0.001 → Holm p = 0.004; n = 2,628 VALIDATION records, source-group paired bootstrap 2,000 iterations, seed 20260816).

**Decision caliber (honestly mixed, same paragraph requirement).** Top-1: V5 0.5007 vs APARENT 0.6011 (**Δ−0.1004, CI [−0.1426, −0.0576], significantly behind**, over 747 rankable source groups); NDCG Δ−0.0195, CI [−0.0318, −0.0065], significantly behind. Attribution (within the pre-registered frame): APARENT was trained on a 2.74M same-assay within-source dense corpus, to which the decision caliber (within-group top-1 fine-ranking) is more sensitive; V5's rank-caliber win reflects stronger ordering, while the decision-caliber deficit reflects weaker within-group fine-ranking relative to a purpose-built supervised model. The two calibers are not contradictory: they probe different capability faces of the task. **Both the rank-caliber significant win (Δ+0.0876 CI [+0.0630, +0.1141]) and the decision-caliber significant deficit (Δ−0.1004 CI [−0.1426, −0.0576]) are stated together here.**

**Ceiling comparison row.** APARENT2 (multimolecule Kowalski 2024 lineage) frozen-Δ reads 0.6810 — the stronger-supervision 0.7-generation version reads *lower* than the 2019 version's frozen-Δ reading; this is honestly registered without interpretation.

**Seed declaration (honest).** The V5 polyA main row is a frozen single-seed terminal training (0.8219) and remains the headline; the paired external row is likewise a frozen single pass. The evaluation-surface CI is given by source-group paired bootstrap; the *training-seed* variance is now supplied by the three-seed supplement in §8.1 (mean 0.8196, range 0.0041), so this item is closed for polyA and remains open only for the matrix's new-family cells (§9.3).

**Table 11. polyA dual-caliber results.**

| Caliber | V5 | APARENT | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|
| Spearman (rank, primary) | 0.8219 | 0.7343 | +0.0876 | [+0.0630, +0.1141] | significantly ahead (Holm p = 0.004) |
| Top-1 (decision) | 0.5007 | 0.6011 | −0.1004 | [−0.1426, −0.0576] | significantly behind |
| NDCG | — | — | −0.0195 | [−0.0318, −0.0065] | significantly behind |
| APARENT2 (ceiling row) | — | 0.6810 | — | — | registered, no interpretation |

### 8.3 Why: the mechanism attribution chain

**(1) First-order closed form (55.4%).** The first-order additive table 0.4555 = 55.4% of V5's 0.8219; the remaining 0.27+ is non-first-order (structural / contextual-composition) signal — exactly the portion by which external general LMs (0.71–0.75) exceed the first-order bound (composition signal partially captured by pretrained representations) plus V5's task-specific gain (+0.07). This is mutually corroborated with G4 structural blindness (three-generation graded sc-hit@1 = 0): V5's within-group gain comes from data geometry, not from a "structural ordering capability."

**(2) The only combinatorial-signal space.** polyA averages 8.5 edits/record (the only task among the 13 with combinatorial space) → its first-order coverage gap is the largest (0.2747), appearing precisely on the task with combinatorial space; MPRAU/TE have their first-order terms already explaining ~80% (V5 increments of only +0.02/+0.01), which throws the task heterogeneity into relief.

**(3) Multi-task isolation causal chain (intervention evidence).** CMS mixing損 on the polyA side is −0.2261 (option-3 CMS augmentation adjudication per-task side-effects: polyA −0.2261 / MRL +0.1809 vs the control recipe task profile) — introducing heterogeneous-corpus mixed supervision collapses the polyA caliber; by contrast, W0 single-task from-scratch MRL training reads 0.1987 vs V5 multi-task 0.1354 = **+47% relative improvement** (multi-task dilution confirmed); the V6–V9 multi-generation mixed-training failure history (V8-S specialization cost polyA −0.16; V9 S1/M6 holdout unlearnable) is mentioned in one sentence (archived in journal batches 76/85/86). Single-task isolation plus same-assay dense supervision is the training-side condition for polyA's high completion.

### 8.4 Application outlets (two levels: generation-line guidance; committee routing)

**Generation line (COMB 2×2, D7).** Exploration (B) / guided (C) / both-arms (D) over 891 sources × 3 seeds, tier-2 full set with the two-tier adjudication discipline. The sc-caliber near-perfect additivity fingerprint is **B+C +0.02974 vs D +0.02972** (difference 0.00002); verdict H_IND_or_NEGATIVE (H-int not significant; exploration gain fully retained, guidance non-destructive, D ≈ C). The polyA third-caliber three rows in the D arm (APARENT frozen scoring, n = 20 report-only) are D_main +0.3527 / seed16 +0.4210 / seed17 +0.4844 (B +0.2328 / C +0.0596 for reference); n = 20 is the *entire* polyA source set within the 891 pool, so the expansion ceiling is exactly 20. The unguided baseline is 0.12046 (891 sources, 28,512 candidates, uniqueness 0.8572, legality 1.0). The critic self-score, the independent evaluator, and the measured calibers are kept separate, and H_IND is reported as-is (near-perfect additivity is positive evidence of orthogonality, not a failure).

**Committee routing (D8).** A constructive zero-peek task-routing scheme (dev-only selection, holdout evaluation, paired bootstrap 2,000 iterations, seed 20260816) gives **Δ+0.0362, CI [+0.0090, +0.0632], excluding zero** on holdout n = 446 (vs single-V5 probe@1 0.0443 → 0.0806; per-task routing MRL → v8_hbench9 / MPRAU → v8_smprau_in / polyA → v8_smprau_in / HL → full). Reported honestly alongside it, the dev-routed variant gives +0.0045, CI [0.0000, +0.0112], crossing zero (dev-selected task-level routing generalizes insufficiently); the constructive version is the pre-registered constructive routing (not a data-peeking selection), and the two variants are layered and reported. Both are reported — never only one.

**Outlet positioning.** Both outlets in §8.4 are demonstrations of the mechanism conclusions, not engineering claims. The committee-routing CI lower bound (+0.009) is close to zero, so the effect-size claim must carry a power boundary (see audit A6).

**Table 12. Application outlets.**

| Outlet | Variant | Δ | 95% CI | n | Reading |
|---|---|---|---|---|---|
| Generation line (COMB) | B+C vs D additivity | +0.02974 vs +0.02972 | — | 891 × 3 seeds | H_IND_or_NEGATIVE |
| Generation line (polyA D arm) | D_main / seed16 / seed17 | +0.3527 / +0.4210 / +0.4844 | — | 20 (report-only) | all-pool polyA sources |
| Committee routing | constructive zero-peek | +0.0362 | [+0.0090, +0.0632] | 446 | excludes zero |
| Committee routing | dev-routed | +0.0045 | [0.0000, +0.0112] | 446 | crosses zero |

---

## §9 Discussion

### 9.1 A model-selection guide

Select by task regime first. C1 tasks (ICC < 0.1): do not select a model; change the measurement or add replicates. C2: align input granularity first (fragment vs full length). C3: within-source dense candidate supervision with pair-mean/within-source caliber. C4: run a low-cost oracle probe first (NO_SIGNAL ≤ 0.10 is a stop-loss) before committing. C5: corpus volume determines success (external references at 280K / 2.74M). A companion "data-side triple screening" (ICC / density / corpus-domain mapping) should precede architecture selection.

### 9.2 Impact on absolute-score leaderboard practice

Between-source composition inflates overall scores (polyA overall 0.71–0.75 vs within-source 0.19–0.23), and in-domain corpus binding (STCNet 0.8144 with 730/730) is invisible without a source-relative protocol. We recommend that benchmarks report the dual task, the within-source caliber, and a corpus-overlap declaration.

### 9.3 Limitations (itemized honestly)

**Rights / data.** GEMORNA / UTR-STCNet / UTR-Insight have no declared license (user exemption decision 2026-09-20, port-ledger addendum commit 5a8e30eb); M6 has a seal/unseal history (`historical_exposure`; the words sealed/untouched/never-seen are prohibited); the payload release boundary stands at 0 authorized public study-payload rows (v332 rights table).

**Single-seed probes.** The 65 new-family matrix cells are single-seed frozen-Δ evaluations (row-construction seed frozen 20260920; model inference is deterministic but row sampling is single). The batch-2/3 generalist probe rows (RiNALMo/ERNIE-RNA and the 7 bespoke-adapter backbones of §3.5) are likewise single-seed frozen-Δ evaluations. The oracle / first-order analyses are mostly three-seed (the MRL seed-sensitive pseudo-signal is itself the evidence for single-seed artifacts); the matrix has no multi-seed variance band (registered honestly). The V5 polyA main row 0.8219 is a frozen single-seed terminal state, but its training-seed variance is now measured by the three-seed supplement (§8.1: mean 0.8196, range 0.0041, Δ vs APARENT CI [+0.0621, +0.1078] excluding zero), which closes the polyA part of this limitation; the paired APARENT row remains a single pass. **ICC provenance items:** MRL 0.83 and REFALT 0.21 are historical reference values whose computation procedure / n are not archived — **PROVENANCE_UNRESOLVED** — and are treated as limitations, not re-run. (By contrast, the recomputation block closes 7/9 tasks: polyA 0.9041 ≈ 0.90 reproduced consistently; TE200304 floating-point-consistent; MPRAU bracketed at 0.7273 [0.694, 0.766], covering the 0.683 magnitude; HL5/HL3 near-zero confirmed.) **Inference determinism:** four of five families are bitwise deterministic; STCNet's stochasticity is by design (official DPC clustering adds tie noise; cell spread 8.4e-04 measured on the MRL cell), and the archived 0.8144 lies within the re-run interval; per-cell power on the new rows shows a median CI width of 0.077 and median MDE ≈ 0.037.

**Honest handling of the FAIL verdict.** The single-factor density held-out FAIL has been executed per the downgrade clause (the only upgrade path is append-only plus re-evaluation at the next decision point); the two-factor claim is EXPLORATORY (not a pre-registered decision gate); the M1 intervention arm is now backfilled (§5.6) and reports a **negative direction gate** with the pre-registered FAIL branch — real-data corpus addition did not move MRL Δ, which bounds the density factor's independent contribution and is consistent with the geometric reading.

**VALIDATION-only.** All matrix / leaderboard numbers are from VALIDATION splits; protected TEST reads = 0; the G4 structural probe's three-generation zero holds only under the D15-2 caliber.

**Fairness boundary (R12 residual).** The frozen-Δ zero-tuning caliber is defensible against the "you under-tuned the external models" objection only indirectly (via the matched-FT evidence, R1/R5); new-family matched-FT pairings were not executed (out of matrix prereg scope, registered honestly). Domain mapping is documentation-level for 4/5 families and measured-level for 1. **Rights/payload (R13 residual).** The availability statement cannot be finalized until study-specific rights review completes; exemption decisions are not publication authorization.

### 9.4 Relation to existing benchmarks and models

The four-axis differentiation (Table 2) positions DeltaBench against conventional absolute-endpoint benchmarks along source-relative formalization, dual-task reporting, ceiling normalization, and two-tier adjudication. Concretely, the underlying assays of several benchmark studies publish rich measured libraries without a source-relative formalization — the 280K random 5′UTR library (Sample et al. 2019), the PLUMAGE screens (Lim et al. 2021), the prostate 3′UTR reporter library (Schuster et al. 2023), the HGMD/ClinVar UTR stability panel (Su et al. 2025) and the MPRAu allelic panel (Xue et al.) are re-used here as canonical records with Δ labels rather than as absolute endpoints. Published UTR predictors are scored on identical VALIDATION records in one frozen-Δ caliber (UTR-LM; RNA-FM; APARENT; APARENT2; Saluki; Optimus; FramePool; RiboNN), so the absolute-vs-delta decoupling of §4 is read on the same records instead of across incompatible reportings. Recently released UTR predictors are ported with fully declared input adaptations and unit-tested against official outputs (GEMORNA, Science 2025; LAMAR, bioRxiv 2024; UTR-STCNet, IEEE BIBM 2025; UTR-Insight; HydraRNA), and their matrix rows (Table 4) locate the in-domain / cross-library boundary quantitatively. Ceiling-normalized learnability accounting (Table 3) and the two-tier adjudication discipline (§2.4) are not standard in prior UTR model reporting, and the three append-only evaluation rows (Castillo-Hair 2024; Plassmeyer 2025; Su et al. 2025) provide fresh surfaces that were frozen before any model was scored on them. The full reference list (including the model papers for the ported baselines and the right/permission statements) is assembled at the submission stage; no external reference numbers are re-cited here, to keep every reported quantity traceable to this project's frozen artifacts.

---

## §10 Methods

### 10.1 Data pipeline

The pipeline has three layers. The **canonical layer** holds nine study converters (e.g. the GSE114002 mother-family identity / candidate rules). The **projection layer** (`development_train_validation_v1`) carries the `direction_normalized_delta` labels. The **new evaluation-row layer** (M1/M6/S1, append-only) declares each row's caliber fully: M1 uses prefix-family plus Hamming-1 fallback pairing; M6 uses a SIC → fraction mapping with a pseudocount of 1.0 and a 155-nt window; S1 uses canonical_115 dual-assay sub-rows with the protected exclusion and the Dao QC flag-keep.

### 10.2 Evaluator

All evaluation uses the same Task-1 evaluator instance (`evaluate_route2_prediction_v1`, K = 10), computing source-group within-source Spearman and task-macro Spearman; the MPRAU caliber is pair-mean (2,008 variants). The **independent re-check protocol** does not read archived metrics and recomputes every cell from `predictions.jsonl` with a |Δ| ≤ 1e-6 gate (achieved 65/65 with max |Δ| = 0.0).

### 10.3 External model porting

The port ledger records 5 PORT_READY families plus the license disposition. Per-family port adaptations are declared in the matrix summary's verbatim blocks. The unit-test protocol verifies official example values and strict loading (UTR-Insight official CSV alignment ρ = 0.96; RiboNN official predictions aligned to model-0 max diff 0.0034 and model-1 max diff 0.0012, reproducing the official Pearson 0.758). The HydraRNA MIG environment fix (Triton autotuner single-device visibility) is declared with zero numerical change.

### 10.4 Pre-registration document list

All decision thresholds were frozen before computation. The full commit-hash list (commits that introduced each frozen document on the W0 branch `route-a-v3-w0-diagnosis-20260902`): `delta_density_prereg_framework_v1` (682b8c80); `delta_validity_oracle_prereg_v1` (a2a2919a); `first_order_decomposition_prereg_v1` (8f80d7e6); `benchmark_v2_matrix_row_prereg_v1` + `benchmark_v2_port_ledger_v1` (a14fb447); `m1_intervention_arm_amendment_v1` (b2e141cd); `polya_3seed_mini_prereg_v1` (73d47cdf); `density_sensitivity_mini_prereg_v1` + `first_order_train_alpha_sensitivity_prereg_v1` (60ff6999); `route2_critic_d16c_within_source_amendment_v1` (0809f5f6); `route2_critic_erk_v2_amendment_v1` (e343d4a3); `route2_w_ladder_amendment_v1` (7303417c); `utr_editflow_goal_v2_amendment_pivot_v1` (682b8c80); `route2_setflow_comb_mechanism_prereg_v1` (eb69e6bb, v8-stage1 worktree). Scope boundary: the family range is maintained at 14 (5 new + 9 existing); M5 is not included (no pre-registered row / port evidence). The batch-2 generalist-backbone append (7 adapters, frozen-probe caliber) is governed by `benchmark_v2_generalist_rows_amendment_v2.md` (DESIGN FROZEN); its execution commit is `6ef7cc6c` (journal 134; port-validation |Δ| = 2.4e-06 against the archived RNA-FM MRL row).

### 10.5 Reproduction entry points

Row-construction scripts (`build_{m1,m6,s1}_*_v1.py`, with construction seeds archived for re-runs); the matrix execution / aggregation / re-check scripts (eight files); the analysis scripts (e.g. `run_delta_vs_density_v2.py`). **R2 internal target rows** cover all nine tasks (global-scaled internal control, macro 0.1317). **LOSO reference (Table 5 supplement, protocol B — frozen decision).** Two protocols were pre-registered: (A) true LOSO 7-fold retraining (42-job infrastructure archived) and (B) a LOSO-lite compiled table with an in-domain header; **protocol B was adopted** and its draft is the frozen LOSO Table 5. It lists, for all nine tasks: the multi-task in-domain critic V5 reading (explicitly not zero-shot), the external same-pool rows where such rows exist (MRL / polyA / MPRAU / HALF_LIFE only; TE / RNA / REF-ALT have no external same-pool rows and are declared as such), the internal target, and the ceiling. Worked rows: MRL 0.1354 vs frozen-Optimus 0.3132 / FramePool 0.2956 / UTR-LM 0.1107, internal 0.1192, ceiling 0.83; polyA 0.8219 vs APARENT 0.7343 / APARENT2 0.6810 / UTR-LM 0.7490, internal 0.7308, ceiling 0.90; MPRAU 0.0732 (cell caliber) vs Saluki 0.1205 weak-control, internal 0.0248, ceiling 0.683; HALF_LIFE 5′UTR 0.0607 and 3′UTR 0.0456 vs Saluki 3′UTR 0.0985, internal 0.0, ceiling ≈ 0.001–0.013 (physical unlearnability); GSE200304 TE 0.0579 vs internal −0.0266; PLUMAGE TE 0.1953 vs internal 0.1747; PLUMAGE RNA 0.0500 vs internal 0.2230 (the single registered loss, n = 48, power-limited); REF/ALT 0.0639 vs internal −0.0052.

### 10.6 Leakage and rights audit

Pigeonhole audit: S1/M6 flagged = 0; M1 has 4 TRAIN overlaps with keep = 0. The **STCNet old three-arm INVALID case** is archived (training-set leakage determination; three arms 0.8135 / 0.2667 / 0.2120 all excluded). The rights-boundary table is the v332 study-rights table (public release: 0 authorized rows).

### 10.7 §8 support protocol blocks

**polyA main-row statistics.** Source-group paired bootstrap (2,000 iterations, seed 20260816) plus a Holm step-down family = the bottom-line nine-task pairings. The polyA main caliber is the GSE269595 VALIDATION n = 2,628 overall Spearman; the decision caliber includes top-1 and NDCG@10 (K = 10). The 3-seed supplement uses the V5 configuration with only the seed changed (main row unchanged).

**COMB 2×2 tier-2 protocol.** 891 sources × 3 seeds; tier-1 calib100 falsification right; repaired adjudicator fingerprint. Calibers are separated: critic self-score / independent evaluator / measured.

**Committee routing dev/holdout protocol.** Stratified 50/50 split (seed 20260909), dev-only argmax, constructive zero-peek variant definition, paired bootstrap (seed 20260816).

**New-data admission: four gates.** Before a corpus may enter training: (1) a manifest (URL + SHA256 + row count + split); (2) an R3 pigeonhole check against the full protected split; (3) cryptic-splicing QC; (4) an effect-size stratification audit (the CMS lesson). The M1 corpus passed all four gates (pigeonhole exact overlap: 4 rows, all in TRAIN; near-duplicate audit: 0 full-length overlaps) before entering training; on the evaluation side, 4,848 unique sequences (with the four TRAIN-overlapping sequences pigeonhole-flagged) were excluded throughout.

**STCNet stochasticity note.** STCNet / the new families are evaluated with a **single frozen forward pass**; the official stochastic DPC clustering introduces run-to-run spread (8.4e-04 measured on the MRL cell), and the archived 0.8144 lies within the re-run interval. Rows with a miss rate > 15% carry the HIGH_MISS marker.

**Freeze discipline and limitations recap.** Frozen-Δ protocol (official weights, zero tuning, input adaptation declared per family); two-tier adjudication (tier-1 calibration falsified by tier-2 full set); ceiling normalization (ICC ceilings with N/A handling for ≈0 rows); four admission gates; STCNet stochasticity as above. Limitations include the single-seed matrix rows, the VALIDATION-only scope (protected TEST reads = 0), and the MRL 0.83 / REFALT 0.21 ICC **PROVENANCE_UNRESOLVED** items (historical reference values whose computation is unarchived).

---

## §11 Conclusion

DeltaBench reframes mRNA edit-effect prediction around the source-relative quantity that downstream users actually consume, and demonstrates — with a frozen matrix, a rigorous re-check, and a pre-registered adjudication discipline — that absolute accuracy and delta accuracy are distinct competencies. The matrix's headline is a supervision-regime effect (strong in-domain, weak cross-library), and the mechanism analysis attributes systematic delta-learning failures to the interaction of supervision regime and data geometry while honestly downgrading the single-factor density claim after a held-out failure. polyA stands as the single task where the label ceiling becomes a non-trivial reading: V5 reaches 91.3% of an ICC-0.90 ceiling, the only family-wise significant win, with a decision-caliber deficit reported alongside it. Two application demonstrations — generation-line guidance and committee routing — show that the mechanism conclusions are actionable, with explicitly bounded effect sizes.

---

## Figures

**Figure 1. DeltaBench frozen-Δ matrix: five newly ported families × 13 task cells** (task-macro Spearman, VALIDATION only; protected TEST reads = 0). Every cell is the archived value from `benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json`; the append-only rows (M1/M6/S1) were frozen before any model was scored on them; the dashed separator marks the new-row block. Rendered artifact: `experiments/analysis_benchmark_v2_matrix_figure_v1/deltabench_matrix_heatmap_v1.{png,pdf}` (producer `make_deltabench_figure1_matrix_v1.py`, 65/65 cells; manifest archived). Headline reading: strong in-domain (UTR-STCNet MRL 0.814) versus weak cross-library (same family on M1 0.067), and the new-row block collectively near zero.

**Figure 2. Density–learnability observation and its held-out test** — 25 external frozen-Δ rows, r = 0.9388 (p = 3.9e-12), with the held-out cell-level test (33/65 = 50.77% in-band → FAIL, honest downgrade per the pre-registered clause). Archived artifacts: `experiments/analysis_delta_vs_density_20260915/delta_vs_density_scatter.{png,pdf}` and `experiments/analysis_delta_vs_density_v2/delta_vs_density_v2_scatter.{png,pdf}`.

**Figure 3. First-order decomposition vs external rows vs ceiling, by task** — polyA (ρ₁ = 0.4555) shows the largest above-first-order structure headroom; MPRAU and TE are close to their first-order readings. Archived artifact: `experiments/analysis_first_order_decomposition_v1/first_order_vs_external_vs_ceiling.{png,pdf}`.

**Figure 4. Ceiling-normalized learnability map (label-ceiling completion by task)** — best in-house row vs the label ICC ceiling (VALIDATION): polyA 91.3% (0.8219 / 0.90) is the only high-completion task; MRL 38.8% (0.3217 / 0.83); REF/ALT 30.4% (0.0639 / 0.21); MPRAU 19.8% (0.1351 / 0.683); PLUMAGE-RNA 13.7% (0.0500 / 0.364); TE 9.9% (0.0579 / 0.586); HALF_LIFE and PLUMAGE-TE carry **no normalization** (ICC ≈ 0 or negative). Archived artifact: `experiments/analysis_benchmark_v2_matrix_figure_v1/deltabench_ceiling_completion_v1.{png,pdf}` (producer `make_deltabench_figure4_ceiling_v1.py`, which asserts every plotted value against `bottomline_adjudication_v1.json` and the label-ICC table at render time).

- Submission-stage figure work: unified styling, vector-only panels, alt-text, and per-figure manifests (the existing products above already carry their producer scripts and data manifests).

---

## Open items / placeholders

- **§5.6** — M1 intervention four gates: **RESOLVED** (negative direction −0.0419; G2 mechanical FAIL with a report-only context row +0.2807; G4 not triggered).
- **§8.1** — polyA 3-seed supplement: **RESOLVED** (mean 0.8196, range 0.0041, Δ vs APARENT CI excluding zero; new row `polyA-V5-3seed-mean`; main row unchanged).
- **§9.3 / §10.6** — the final availability statement remains gated on study-specific rights review (14/14 studies have no named accountable reviewer yet; exemption decisions are not publication authorization). 【USER-SIDE, awaiting assignment】
- **Reference list + figure set** — assembled at the submission stage; all reference *numbers* are already in-table and traceable to frozen artifacts. Figure inventory (mechanism png/pdf products) is listed in the project's `experiments/analysis_delta_vs_density_20260915/` and the matrix summary; final figure rendering is a submission-stage task.
- **Resolved in draft v1.1 (this revision)**: full pre-registration commit-hash list (§10.4), LOSO-lite Table 5 protocol-B adjudication (§10.5), external model / benchmark relation paragraph (§9.4).