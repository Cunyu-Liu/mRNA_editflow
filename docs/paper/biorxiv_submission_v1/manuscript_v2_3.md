# DeltaBench: A Source-Relative Benchmark Reveals Why Absolute-Score Models Fail to Predict mRNA Edit Effects

*Version v2.3 (2026-10-08): full-manuscript expression-level revision of v2.2 (post-unblinding) — paragraph restructuring, sentence splitting, terminology alignment, internal-note removal; all numbers, verdicts, citations and figure facts unchanged.*

> **Scope and discipline statement.** Every number in this manuscript is drawn from frozen, pre-registered experiments. All matrix and mechanism readouts are computed on **VALIDATION splits** under the frozen-Δ (zero-fine-tuning) caliber, with verdicts following decision rules frozen before computation. The protected TEST split (18,292 rows) was read **exactly once** — after the full draft was complete, per the frozen unblinding prereg (§9.3; receipt summary in Supplementary S4). Headline claims carry their TEST-confirmed status inline. No tuning, selection, or wording decision was made after that read. Where a number is registered but not yet finalized, a PENDING marker is used and no number is invented.

---

## Abstract

mRNA edit-prioritization pipelines — from variant ranking to generative design — consume the *ranking of differences* (Δ) between sequences, yet the field's leaderboards score only absolute prediction accuracy. We show these are distinct competencies. A frozen Optimus model reaches ρ = 0.873 on absolute MRL prediction while its source-relative delta correlation collapses to 0.313. No public resource measures the delta quantity, and the underlying paired libraries are scarce; we therefore curate one. We introduce **DeltaBench**, a source-relative benchmark built from 15 public MPRA studies into 126,165 canonical paired records (near-duplicate-component splits; three append-only rows frozen before any model was scored), covering 13 evaluation cells and released with seed-frozen converters, frozen manifests, and the full re-check pipeline as a reusable community asset. Scoring 14 model families under a frozen-delta, zero-fine-tuning protocol yields a 65-cell matrix (independently re-checked to floating-point identity) with two headline readings: in-domain supervision is strong (UTR-STCNet 0.814 on MRL, with 730/730 evaluation sequences inside its training corpus) but collapses cross-library (0.067); without same-distribution dense supervision, 51 of 65 cells read |ρ| < 0.1. A pre-registered mechanism analysis attributes the pattern to an interaction of supervision regime and data geometry. The single-factor density hypothesis fails its held-out test and is honestly downgraded; a three-arm intervention triangle (synthetic-density falsified; parameter-side bound-anchored; same-domain data effective, cross-library corpus not) supports the two-factor regularity, which reproduces on 11 probe backbones spanning a second architecture cohort. First-order/structural decomposition and a four-class failure taxonomy turn the regularity into model-selection rules. On the one task passing every gate (polyA; ICC 0.90), our model reaches 0.8219 — 91.3% of the label ceiling and the only family-wise significant win — confirmed by a one-shot TEST unblinding (0.8205).

**Keywords:** mRNA edit-effect prediction; source-relative benchmarking; delta learning; failure attribution; supervision regime; data geometry.

---

## §1 Introduction

### 1.1 The edit-prioritization task and the need for Δ ranking

Edit-effect prediction models are increasingly deployed as *prioritizers*: given a source sequence and a set of candidate edits, downstream pipelines (variant-priority triage, guide construction for generative design) consume the *ordering of candidate differences* — the rank of Δy = y(candidate) − y(source) — not the absolute activity of any single sequence. The two quantities are, however, routinely conflated. Our opening case makes the distinction concrete. For the MRL task (GSE114002) a frozen Optimus model [18] achieves ρ_abs = 0.8733 on the absolute endpoint, while its delta correlation is only ρ_delta = 0.3132 and its cross-arm error correlation is ρ_ε = 0.7196. The model "reads absolute values" well and "reads differences" poorly. These are two competencies, and a leaderboard that reports only the first does not measure the second.

### 1.2 The gap in existing benchmarks

Current absolute-endpoint leaderboards do not distinguish the two capabilities, and no widely used benchmark formalizes the source-relative quantity at all. Three concrete deficiencies follow. First, there is no source-relative formalization: without a `delta_hat = pred(candidate) − pred(source)` definition anchored in a canonical paired record, absolute and delta abilities cannot be separated. Second, there is no controlled design over supervision regime × data geometry: in-domain supervised rows and out-of-domain rows are interspersed in a single table, so a reading such as UTR-STCNet's [30] 0.8144 on MRL — a number that partly reflects corpus-level memorization, since 730/730 evaluation sequences fall inside its training corpus — is invisible as a *regime* property on conventional leaderboards. Third, there is no failure-attribution discipline: without pre-registered decisions, a high absolute score is silently extrapolated into an assumption that deltas are usable.

### 1.3 Contributions

We make four contributions.

**C1 — the DeltaBench dataset and benchmark definition.** A curated, reusable benchmark resource that the field currently lacks: 126,165 canonical paired records built from 15 public MPRA studies (the largest source-relative collection of its kind), a canonical record schema, a dual task (absolute and frozen-Δ), 13 evaluation cells (nine existing tasks plus three append-only rows), and a frozen-Δ evaluation protocol (published weights, zero fine-tuning) with a two-tier adjudication discipline and ceiling normalization.

**C2 — The 14-family × 13-cell matrix.** Five new families are ported (each passing a family unit test, with all input-adaptation decisions fully declared) alongside nine existing reference families. The resulting matrix contains 65 newly computed cells, and an independent re-check script recomputes all of them from archived predictions at floating-point-identical precision (65/65).

**C3 — A three-ring attribution framework, including the two-factor regularity.** Ring 0, differential validity (five error sources plus a decisive oracle arm); then the two-factor regularity — an observation, a held-out failure that is honestly downgraded, and an intervention triangle supporting ρ ≈ f(supervision regime × data geometry) at EXPLORATORY grade; then a first-order/structural decomposition and a four-class failure taxonomy.

**C4 — The polyA positive result pushed to 91.3% of ceiling.** On the single task (of 13) whose label ICC is 0.90, V5 reaches 0.8219, i.e. 91.3% ceiling completion, closing more than half (0.0876/0.1657 = 52.9%) of the remaining learnable space, with Holm p = 0.004 as the only family-wise significant win, and with the decision-caliber result reported honestly as mixed (attributed to supervision regime). Mechanism attribution (first-order 55.4% plus combined signal plus a multi-task isolation causal chain) and two application outlets (generation-line guidance and committee routing with a CI excluding zero) complete a full positive-result chapter.

### 1.4 Structure of the paper — a single reading line, six linked moves

The paper is organized as one causal chain; each section ends by opening the question the next section answers:

1. **The phenomenon (§2–3):** a benchmark formalizing the source-relative Δ task and the 14-family × 13-cell matrix. The matrix's two readings (in-domain strong 0.814 / cross-library collapsed 0.067; new rows collectively near zero) raise the paper's driving question: *why does absolute ability not transfer to delta ability, and what does?*
2. **The first mechanism ring (§4):** differential validity. The 0.873 → 0.313 decoupling and the five error sources show the failure is structural, not incidental — and §4.5 answers the two strongest a-priori objections (embedding difference ≠ property difference; no-delta-training) with the probe/oracle evidence before the reader can raise them.
3. **The second ring (§5):** the two-factor regularity. The near-zero band is not uniform noise: supervision regime × data geometry groups it, the single-factor density hypothesis fails its held-out test (honestly downgraded), and the intervention triangle closes the loop — same-source dense supervision works, heterogeneous corpus does not.
4. **The third ring (§6):** signal decomposition. Where the learnable part lives (first-order table vs structural signal) determines whether more data or more model is the right investment (polyA 55% structural; MPRAU/TE 79–81% table).
5. **The positive branch (§7–8):** failure taxonomy turns the rings into a selection guide; polyA is the boundary case carried to 91.3% of the label ceiling with two working application outlets; the regularity is actionable, not just descriptive.
6. **The deliverables (§9–10):** model-selection guide, data-side triple screening, and the pre-registration / re-check apparatus that makes every number in the chain auditable.

A reader who takes only one line away: *absolute leaderboards do not measure the quantity edit pipelines consume; we measure it, explain what governs it (regime × geometry, proven by intervention), and show when it can be pushed to the measurement ceiling (polyA) — with the decision rules for which case you are in.*

---

## §2 The Benchmark

### 2.0 Why this dataset had to be built, and how

No public resource measures the quantity edit pipelines consume. Source-relative delta evaluation requires measured libraries in which multiple candidate sequences share one measured source under the same assay — a property that public MPRA deposits rarely preserve, because most assays are designed either as random libraries without shared sources or as single-mutant scans without within-source density. We therefore curated one from scratch: a 15-study screen of GEO/ENCODE/EBI (the full scarcity argument, acquisition protocol, and admission gates are in §10.1), yielding 126,165 canonical paired records with direction-normalized delta labels, near-duplicate-component splits, and three append-only evaluation rows frozen before any model was scored on them. The result is released as a reusable community asset — per-study seed-frozen converters, frozen manifests, all model predictions, and the full re-check pipeline — so that future edit-effect models can be scored on identical records under the same frozen-delta protocol without re-deriving the curation (§Data availability).

### 2.1 Definition and the canonical record

The unit of the benchmark is a *paired record* over a (source, candidate) pair. Each record carries a fourteen-field schema including `source_sequence`, `candidate_sequence`, `source_relative_edits`, `direction_normalized_delta`, `source_group_id`, `endpoint_descriptor`, `biological_context`, `eval_split_status`, and `provenance`. The source-relative quantity is defined as Δy = y(candidate) − y(source) at the record level (LEVEL_DIFFERENCE caliber). Records are grouped by `source_group_id` so that within-source calibers can be computed in addition to the overall caliber. **Dual task declaration:** the absolute task (ρ_abs) and the frozen-Δ task (ρ_delta) are reported as separate columns computed on the same evaluation surface, never merged.

### 2.2 The 13 tasks (9 existing + 3 new rows; S1 two arms)

Throughout the paper we use *cells* when referring to the matrix's evaluation units and *tasks* when referring to the underlying assay endpoints; the 13 cells comprise the nine existing tasks plus three append-only rows, with S1 carrying two assay arms.

**Nine existing VALIDATION tasks.** MRL (GSE114002 [1], n = 730); polyA (GSE269595 [3], n = 2628); MPRAU (ENCSR854RUF [4], n = 12048); HL5 (GSE217518 [7], n = 400); HL3 (same library, n = 503); TE200304 [5] (n = 1614); TE149487 [6] (n = 48); RNA149487 [6] (n = 48); REFALT (GSE186455 [8], n = 274).

**Three append-only new rows (seed frozen 2026-09-20).** **M1** — an MRL row from GSE232927 [2] (Castillo-Hair 2024): 2,805 rows over 2,801 sources across three cellular contexts (HEPG2 800 + T_CELL 1600 + HEK293T 405), with a candidate-per-source density of 1.0014.

**M6** — an NDD 5′UTR row from GSE246381 [9] (Plassmeyer 2025): 800 rows over 509 family sources; its seal/unseal history is recorded in provenance and `historical_exposure` is reported per contract.

**S1** — a stability row from Su 2025 [7] (eLife): 5,572 sub-rows split into two assay arms (SH 2,792 / HEK 2,780), density 3.67 over 1,519 sources, with 1,330 protected records excluded and a Dao cryptic-splicing QC flag-keep of 834/1,871.

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

**Frozen-Δ caliber.** Each ported family is run with its official published weights, with no fine-tuning, no retraining, no gradient updates, and no output post-calibration; only input adaptation and dtype/device movement are permitted. All per-family input adaptations are declared exhaustively: GEMORNA [29] uses a dual 5′/3′ head; HydraRNA [26] uses frozen embeddings read out at the first coordinate; STCNet uses a padd-120 one-hot port; UTR-Insight [31] uses a 50-nt padded prefix with the GSM3130435 branch; LAMAR [28] uses the EsmTokenizer with 1026 truncation.

**Two-tier adjudication.** A tier-1 small-calibration set (e.g. polyA `calib100`) may raise a signal, but that signal must be re-checked on the tier-2 full set (891 sources) before it is credited; the discipline case is the Optimus calib100 signal being falsified by the 891-source full set.

**Budget parity (fairness declaration).** All matrix rows are scored under the same information budget (VALIDATION-only reading, one frozen evaluator instance, identical record sets), the same split discipline (the frozen near-duplicate-component manifest), and the same tuning budget — zero tuning: official published weights, no per-row hyper-parameter search, no output post-calibration. For the project's own in-domain rows, hyper-parameters and training budgets are frozen in the pre-registration documents (§10.4) and are never re-searched per task. For the ported families, no matched fine-tuning was performed; the "you under-tuned the external models" objection is answered by the pre-registered matched-FT control rather than by ad-hoc tuning (MPRAU matched-FT arms for UTR-LM / RNA-FM: three seeds, all negative — the R5 closure). Budget asymmetry is declared where it genuinely exists: APARENT's decision-caliber advantage on polyA traces to its 2.74M within-assay training corpus — a supervision-regime property reported openly in §8.2 — not to an evaluation-budget difference.

**Ceiling normalization.** Each task has a label ICC ceiling: MRL 0.83, polyA 0.90, MPRAU 0.683, TE200304 0.586, HL5 0.0014, HL3 0.013 (and TE149487 −0.274, RNA149487 0.364, REFALT 0.21 in the full table). Table 3 covers the nine existing tasks; the three append-only rows (M1/M6/S1) carry no archived label-ICC entry and are therefore reported without ceiling normalization. Rows whose ICC is ≈ 0 are assigned normalization **N/A**, so that negative denominators are never used to manufacture inflated readings. The unified source for these ICC values is the label-ICC table (`mechanism_results_v2.json → label_icc_reference`); the GSE149487/GSE200304 entries additionally have an independent recomputation cross-check.

**Table 3. Label ICC ceilings.**

| Task | ICC | Task | ICC |
|---|---|---|---|
| MRL | 0.83 | HL5 | 0.0014 |
| polyA | 0.90 | HL3 | 0.013 |
| MPRAU | 0.683 | TE149487 | −0.274 |
| TE200304 | 0.586 | RNA149487 | 0.364 |
| REFALT | 0.21 | | |

### 2.5 New-row construction discipline

New rows are append-only (they do not enter any existing manifest), with construction seeds frozen. A pigeonhole leakage audit returns zero protected overlap for M1/M6/S1 (M1: 4 TRAIN-overlapping sequences flagged `keep=0` and thus naturally excluded; M6: zero hits at both locus and sequence level; S1: zero after the 1,330-record hard exclusion). TRAIN overlaps are only *marked*, not removed, for the existing rows, while the four M1 TRAIN overlaps fall out by construction. Extreme values are preserved as-is (e.g. the S1 t05 anomalous magnitude; robust statistics are recommended when reading S1). With the instrument defined  (records, splits, calibers, and admission discipline) the next section reports what the frozen models actually do on it: the 65-cell matrix whose two readings pose the paper's driving question.

---

## §3 Results: the matrix

### 3.1 The 14 × 13 matrix

The matrix comprises five newly ported families measured across 13 cells plus nine existing reference families taken as archived `ALREADY_DONE` rows — the nine reference *families* are a different set from the nine existing *tasks*, and the coincidence of counts is incidental. **Figure 1** renders the complete 65-cell frozen-Δ block (the five new families × all 13 task cells; every cell is the archived `task_macro_spearman`, no recomputation). Table 4 collects the key newly computed readouts with exact values.

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

An independent re-check script, which does **not** read the archived metric values, recomputes each of the 65 cells directly from `predictions.jsonl` (137,350 rows). All 65/65 cells pass, with a whole-matrix max |Δ| = 0.0 (floating-point-exact), including the extreme STCNet MRL cell (0.814430506981109). This establishes that the reported metrics are exactly reproducible from the archived predictions. The same 65 cells are rendered in Figure 1 (five ported families) and, together with the archived reference rows, in Figure S1 (all 14 families).

### 3.3 Key reading I: strong in-domain, weak cross-library

UTR-STCNet achieves 0.8144 on the in-domain MRL row (GSE114002, n = 730) but only 0.0672 on the cross-library M1 row (GSE232927, n = 2,805), even though both rows live in the same MRL domain. The cross-library reading falls back to the general-family level. And 730/730 evaluation sequences (667 train_val + 63 test) for the in-domain cell are empirically contained within that family's MPRA-H training corpus, so part of the in-domain supervised gain is bound to corpus-level memorization. A same-library weak-supervision control (UTR-Insight, random 50-nt library) reaches only 0.2739 on the same cell, isolating the effect of supervision *density* from that of domain membership.

### 3.4 Key reading II: the appended rows are collectively near zero

On the three appended rows the five new families stay below 0.09 (M1 max 0.0889, GEMORNA; M6 max 0.0464; S1 max 0.0450). Across the full 65-cell matrix, 51 of 65 cells read |ρ| < 0.1 (Figure S1 renders the complete distribution). The correct reading is that model families **without same-distribution dense supervision** are collectively constrained on the new rows; the frozen-Optimus MRL reference (0.3132) serves as a supervision-regime control. This must not be phrased as "all baselines fail to predict deltas": the phenomenon is a property of the supervision regime, and Optimus/APARENT are the regime contrast rows. A refinement from the per-cell power analysis is recorded in §9: a majority of the 15 new-row cells are near zero, but on the M1 row GEMORNA (ρ = 0.0889, CI [0.052, 0.127]) and one further 5′UTR family carry a small yet significant positive signal.

### 3.5 Batch-2 generalist backbones: the same regularity on a second architecture cohort (append-only rows)

To test whether the §5 two-factor regularity is an artifact of one architecture cohort, we appended a second, structurally diverse batch of generalist backbones under the frozen probe protocol (frozen backbone + linear probe on embedding differences, TRAIN fit / VALIDATION eval, protected TEST reads = 0, pre-registered in `benchmark_v2_generalist_rows_amendment_v2.md`). The first cohort — the "batch-0/1/3" rows already frozen into the project's generalist band — comprises four backbones (RiNALMo micro/mega/giga and ERNIE-RNA). The second cohort comprises seven: Orthrus 4-track and 6-track (Mamba/SSM, 10.2M; the 6-track input uses the official CDS-frame + splice-site channels, zeros for UTR-only sequences), CodonFM-80M (codon-level encoder), mRNA-LM 5′UTR and 3′UTR (Sanofi official checkpoints, WordLevel character tokenizer per the official `OneModel.py` recipe), CaLM (codon LM, multimolecule re-export with provenance recorded), and LucaOne (1.58B gene–protein pretrained, official FTP checkpoint, byte-exact sha256). All seven adapters were smoke-tested (single-sequence forward), and the runner glue was port-validated against the archived RNA-FM MRL row (|Δ| = 2.4×10⁻⁶ ≤ 10⁻⁵).

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

Sources (append-only, numbers copied verbatim from frozen artifacts): `analysis_generalist_rows_v2/{batch2a_orthrus_v1, batch2b_codonmrna_v1, batch2b2_mrnalm_v1, batch2c_calm_v1, batch2d_lucaone_v1}/` and the reporting-only aggregation `batch2_matrix_v1.json`. One incident is disclosed for reproducibility: the initial mRNA-LM runs produced degenerate embeddings (a WordLevel tokenizer with whitespace pre-tokenization collapsed each unspaced sequence into a single `[UNK]` token, Spearman = None with zero prediction variance); the fix  (space-joined character input per the official recipe) was verified by a non-degeneracy test (pairwise embedding distances 18.05–28.17) before the clean rerun. The failed run's artifacts were removed; the matrix above is the post-fix rerun.

Why does absolute ability not transfer — is the failure structural or incidental? §4 opens with the mathematical answer and ends with two objections answered (§4.5); what remains unexplained by error structure alone, why the zero band is *organized* rather than uniform, is §5's question.

---

## §4 Why absolute scores do not transfer (Ring 0: differential validity)

### 4.1 Mathematical decomposition

Under a bounded linear read-out, Δŷ = w·Δh, and hence Δŷ − Δy = ε_c − ε_s: the differential error is exactly the *difference* of the two arms' errors, with the shared bias cancelling. This yields a favorable condition (ρ_ε → 1, errors cancel) and a catastrophic condition (ρ_ε → 0, or an error-difference term uncorrelated with the target). Absolute accuracy constrains neither: a model can be accurate on levels while its arm-errors are uncorrelated in the difference.

### 4.2 Five error sources

**(1) Error-correlation triples.** Optimus MRL shows 0.8733 / 0.3132 / 0.7196: partial cancellation, but insufficient. In "fully blind" rows, high ρ_ε coexists with no signal: LAMAR HL5 gives 0.0207 / 0.0441 / 0.6863, and HL3 gives −0.0911 / 0.0184 / 0.4429 — i.e. shared error with no shared signal.

**(2) Overall vs within-source divergence.** For polyA the overall caliber is 0.71–0.75 while the within-source caliber is only 0.19–0.23 (UTR-LM 0.7490 → 0.1934; RNA-FM 0.7114 → 0.2335 over 78 groups). The overall figure is inflated by between-source composition, whereas edit prioritization needs the within-source caliber.

**(3) "Broad and shallow" probe geometry.** Cosine alignment is 0.79/0.81 while RNA-FM's participation ratio is 258.7, top-10 share is 0.0955, and the per-dimension gain is 0.0189 — the representation is broad but shallow (SNR proxy under an isotropy assumption; the 16 rows of unarchived probe weights are registered but not enumerated).

**(4) Oracle arm (decisive).** On MRL the oracle probe is NO_SIGNAL: RNA-FM three-seed mean 0.0869 (per-seed 0.1369 / 0.0640 / 0.0597, the seed-sensitive pseudo-signal shown in full) and UTR-LM 0.0700 — the representation does not contain delta signal, and changing the supervision form does not help. On polyA the probe is HEAD/EQUIVALENT (RNA-FM +0.0070, UTR-LM −0.0054): the signal genuinely exists in the representation and is already delivered.

**(5) Context mismatch and label noise.** RiboNN shows four near-zero rows (−0.0106 / −0.0290 / +0.0778 / −0.0450; a full-length TE model applied to fragments). On the data side, GSE149487-TE has ICC −0.274 with a split-half of 0.050, closing the lower bound; the reverse case is GSE200304 (ICC 0.586 is measurable while the strongest delta model reads ≤ 0.011), locating the deficit on the representation side.

A consistency re-check recomputes all ρ_delta values at 30/30 rows with |Δ| ≤ 1.1e-16. **Figure 7** renders the three error-correlation triples as grouped bars (ρ_abs / ρ_delta / ρ_ε per row, values locked to the frozen archives): the Optimus MRL decoupling case (0.8733 / 0.3132 / 0.7196) and the two LAMAR blind rows (HL5 0.0207 / 0.0441 / 0.6863; HL3 −0.0911 / 0.0184 / 0.4429) — high ρ_ε coexisting with no delta signal. The remaining error-source rows (overall-vs-within divergence, probe geometry, oracle verdicts, context mismatch, label noise) are carried by the prose above with their frozen values; the full enumeration is retained in the archived claim–evidence table.

### 4.3 Protocol discovery (disclosed honestly)

The existing frozen-Δ LM rows are implemented *as* a Δh → Δy regression, which is inconsistent with parts of their docstrings; under a linear read-out the two supervision forms are mathematically equivalent and do not affect any archived value. This is recorded in the protocol note (`results_oracle_probe.json → supervision_form_audit_note`).

### 4.4 Operational boundaries

We summarize five valid conditions, five failure modes, and three operational criteria: (i) use the within-source caliber and inspect the ρ_ε triple first; (ii) declare ICC and context match before modeling; (iii) a high absolute score must not be extrapolated into delta usability.

---

### 4.5 Two anticipated reviewer questions, answered with the project's own evidence

**Q1 — "Is a difference in embeddings necessarily a difference in properties?"** No — and that non-identity is precisely this paper's subject matter, not an objection to it. Three layers of evidence support the distinction.

First, *the decoupling itself*: a frozen Optimus [18] reads absolute MRL ranking at ρ_abs = 0.873 while its Δ ranking sits at ρ_delta = 0.313 with cross-arm error correlation 0.720. Embedding-level accuracy coexists with near-collapse at the delta level, so the two are empirically separable competencies.

Second, *the task-level condition*: the first-order decomposition (§6) quantifies when alignment is high. The learnable component on polyA is 55% first-order table signal, and the 7/7 batch-2 backbones [23-27, 32-34] reach +0.75 on polyA with a 641-parameter linear head (approaching APARENT's [16] 0.734 in-domain), while on MPRAU/TE 79–81% of the learnable part is the table and external backbones stay ≈ 0. Alignment between embedding difference and property difference is therefore *measurable per task*, not assumed.

Third, *the probe protocol is itself the hypothesis test*: if embedding differences carried no delta-relevant information, the frozen probe would read ≈ 0 everywhere. It reads 0.75–0.79 on polyA and ≈ 0 on 8 of 9 tasks for 11 consecutive backbones (batches 0–3 + batch 2) — a systematic stress test of exactly this objection, with a positive and a negative branch both present.

**Q2 — "None of these backbones was trained on delta data, so of course they fail — isn't that expected?"** Expected is our result, stated quantitatively — but the shallow version of this explanation is falsified by three controls inside the benchmark.

(i) *The probe arm supplies delta supervision* [20, 21]: the linear head is fit on each task's TRAIN split, so "never saw delta data" does not hold — supervision is present at read-out. The backbones still read ≈ 0 on 8/9 tasks, which locates the deficit in the frozen representation (no linearly accessible edit-sensitive signal), not in the absence of supervision. The oracle arm makes this decisive: MRL NO_SIGNAL (three-seed 0.0869 mean, seed-sensitive pseudo-signal shown in full) versus polyA HEAD/EQUIVALENT — same probe, two different verdicts, so the protocol discriminates rather than uniformly failing.

(ii) *Same supervision, different outcome*: polyA 0.75 vs MRL 0.04–0.13 under identical probe supervision. The operative variable is the supervision regime × data geometry (corpus density 126.7 vs 4.9), which is the two-factor regularity of §5 — a quantitative prediction that held on a second architecture cohort (§3.5: 7/7 backbones reproduce the band).

(iii) *The intervention triangle closes the loop*: adding heterogeneous dense corpus (G1 synthetic, direction gate negative) does not fix it; only same-source dense supervision does (the W-ladder 0.1987 → 0.3158 run). So the practitioner-facing answer is inverted: the missing ingredient is not "more pretraining" but *same-source high-density paired libraries* — which is the data-side triple screening we deliver in §9. In short, "no delta in training" predicts the zero band, but the benchmark turns that expectation into a measurable regularity with regime contrast rows (UTR-STCNet 0.814 in-domain vs 0.067 cross-library) and falsifies its shallow reading.

Error structure explains the decoupling but not the pattern of which cells survive (polyA) and which collapse (MRL). §5 shows the band is governed by supervision regime × data geometry, with intervention-level evidence.

---

## §5 The two-factor regularity

### 5.1 Observation

Across 25 external rows × 9 task families, log10(density) versus ρ_delta yields a Pearson r = 0.9388 (p = 3.9e-12, n = 25), with a scatter of 25 points and a frozen fit line (Figure 2a). The reading is stratified: the five rows with ρ ≥ 0.68 all lie on polyA (density 126.7, a lone order-of-magnitude outlier), while the twenty rows with density ≤ 6.1 all read ≤ 0.32. The correlation is between-strata, not within.

### 5.2 Held-out test (honest downgrade)

Under the frozen pre-registered framework (tolerance band |Δρ| ≤ 0.10; PASS gate ≥ 70%; no row-picking, append-only, no band changes), all 65 new cells are predicted without back-feeding. The result is in-band 33/65 = **50.77% < 70% → verdict FAIL** (Figure 2b), and the downgrade clause triggers: the "single-factor density regularity" is downgraded to a **task-specific phenomenon**, the phrase "cross-task universal regularity" is thereafter prohibited, and the only upgrade path is append-only plus re-evaluation at the next decision point. The frozen fit parameters are slope 0.3663 per decade and intercept −0.0857 (the observation's r = 0.9388, p = 3.9e-12, n = 25 is stated in §5.1; Figure 2a/2b render the scatter, the fit line, and the in-band check). This subsection is an honest negative result and the disciplining anchor of §5. A dual-caliber sensitivity recomputation (training-pair density caliber) raises the in-band rate from 50.77% to 60.00% (+9.2 pp, six flipped cells all on the new rows); the FAIL verdict is robust to the caliber choice.

### 5.3 Two-factor grouping (EXPLORATORY, not a decision gate)

Grouping by whether the evaluation endpoint domain lies *inside* the model's training corpus: of the 21 below-band cells, **0** are in-domain (all five polyA new-family cells fall below band, with frozen predictions ρ ≈ 0.68 against observed values from −0.37 to +0.31); of the 11 above-band cells, **6** are in-domain (STCNet-MRL 0.8144 exceeds the band by +0.65; four MRL-domain families form a floor on M1). The in-band rate is 3/9 = 33.3% in-domain versus 30/56 = 53.6% out-of-domain.

Same-density contrasts sharpen the point: at density 4.9, in-domain dense supervision reads 0.81 versus −0.02 out-of-domain; at density 126.7, polyA reads 0.68–0.75 for the old rows versus 0.15–0.31 for the new rows. The EXPLORATORY conclusion is that deviations group systematically by supervision regime, i.e. ρ = f(supervision regime × data geometry), and that the 25-row fit (r = 0.9388) reflects co-variation of the two factors.

### 5.4 The intervention chain (observation–intervention triangle)

**Synthetic side (falsification).** D16-C synthetically increases density: G1 FAIL (gap 0.8235 ≥ baseline 0.7943; Figure 5, synthetic panel); G2 PASS (polyA 0.81338, non-destructive); G4 sc-hit1 = 0 (81/81 support, 0 hits). This falsifies the "increase density and it is fixed" phrasing.

**Parameter side (half-effect).** ERK v2 (the parameter-side upper-bound row, a **1,177-parameter** model; Figure 5, parameter panel) supplies an explicit first-order prior: the gap halves from 0.7943 to 0.3815, and validation reaches 0.2015 — the historical MRL best, sitting at the closed-form first-order + context bound E7 = 0.2069 (a bound-anchored result: the gain is explained by a first-order prior, i.e. source-residual shrinkage rather than new discriminative signal); G4 = 0 (231/231 support, exact match 1); G3 not triggered (partial path). This bound-anchored reading, together with the E7 anchor (0.207) and the three-generation G4 zeros, leaves **no claim of remaining model-side headroom**.

**Real-data side (fully effective).** The W ladder is 0.1987 → 0.2470 (LoRA) → 0.2555 (full-FT, 2 epochs) → 0.3158 (3-seed ensemble) ≈ frozen-Optimus 0.3132 (Figure 5, real-data panel). The endpoint is a statistical tie, Δ+0.0027 CI [−0.045, +0.048], crossing zero; this is a statistical tie with a point estimate ahead, and given the 730-record power we do not claim a significant surpass (we do not write "no difference").

**Triangle.** The causal carrier of the density correlation is the geometric coverage of real supervision data (H-geometry); H-density is excluded by the synthetic falsification; H-selection is retained as a boundary condition (polyA mechanism locality).

**Table 7. Intervention triangle.**

| Arm | Result | Interpretation |
|---|---|---|
| D16-C (synthetic) | G1 FAIL gap 0.8235 (baseline 0.7943); G2 PASS polyA 0.81338; G4 0 (81/81) | "increase density" falsified |
| ERK v2 (parameter) | gap 0.7943→0.3815; val 0.2015 ≈ E7 0.2069; G4 0 (231/231); G3 not triggered (1,177-parameter upper bound) | half-effect; bound-anchored |
| W ladder (real data) | 0.1987 / 0.2470 / 0.2555 / 0.3158 ≈ 0.3132 | fully effective; tie |
| Tie caliber | Δ+0.0027 CI [−0.045, +0.048] | statistical tie, point estimate ahead |

### 5.5 The three-generation zero chain (G4)

Three successive generations (D15-2 polyA V5 → D16-C probe → ERK v2) each return graded sc-hit@1 = 0 (support rates 100%, hit 0). This locates the structural-ordering deficit at the data/knowledge level: MRL rank signal is dominated by source means.

### 5.6 M1 intervention arm (result slot)

**Pre-registered design.** The M1 arm is a mechanism-intervention verification arm (not a SOTA push), pre-registered before launch (freeze commit b2e141cd). It is a single-variable data-side intervention: the MRL delta pipeline is re-trained with real high-density external MRL data (M1, GSE232927 Castillo-Hair 2024) added to the training side. The question is whether real dense external MRL data improves delta discriminability in the direction predicted by the two-factor regularity.

**Configuration.** The configuration is cloned from Route A full-FT V2 (the project's MRL-only fine-tuning lineage) except for the single data-side increment: backbone mRNABERT + mean-pool linear head, full fine-tuning, base library of 677,608 clean rows, 6 epochs under a fixed FINAL-EPOCH(6) protocol (per-epoch metrics diagnostic only, no peak-picking), batch 128 / lr 2e-5 / weight-decay 1e-4 / AdamW. The increment is a 300,000-row downsample (seed 20260920) of the HepG2 r1 + T-cell r1/r2 defined-end library, giving 977,608 rows per epoch, with joint z-scoring of labels and joint shuffling. Leakage discipline excludes all 4,848 unique evaluation-row sequences and the four TRAIN-overlapping pigeonhole-flagged sequences.

**The four frozen gates.**

- **G1 (direction):** single-seed MRL VALIDATION frozen-Δ task macro Spearman, direction positive if the gain exceeds +0.01 over the 0.3158 baseline; the V2 single-seed spectrum (e.g. seed 20260903 = 0.3198) is reported as a fluctuation band.
- **G2 (non-destruction):** polyA main-row recomputation must not drop more than 0.02 relative to the V5 0.8219 caliber — the V5 multi-task main row is the frozen anchor because it is the paper's headline polyA row, even though the arm inherits V2's MRL-only design.
- **G3 (efficiency, reported with no threshold):** wall-clock, peak memory, rows per epoch.
- **G4 (mechanism, executed only if G1 is direction-positive):** a directional comparison on the M1 evaluation row against the in-domain/cross-library STCNet phenomenon.

The pre-registered interpretation framework treats a FAIL as a valid deliverable (evidence that the density factor's independent contribution is bounded), and the pre-registered contrast anchor is the Route A V2 baseline's M1-row reading 0.0916 (n = 2,805), above the five external families (0.0889 / 0.0672 / 0.0575 / 0.0226 / −0.0215). The arm ran to its frozen terminal state (seed 20260920, 6/6 epochs, FINAL-EPOCH(6)) and was harvested.

**Results (frozen harvest, recomputed from archived predictions with bitwise match; `m1_adjudication_v1.json`).**

| Gate | Frozen reading | Verdict |
|---|---|---|
| **G1 direction** | MRL VALIDATION frozen-Δ task-macro ρ = **0.2739**; baseline band 0.3158 + 0.01 = 0.3258; gain = **−0.0419** | **NEGATIVE** (direction not positive) |
| **G2 non-destruction (mechanical)** | polyA VALIDATION recomputation 0.3710 vs frozen V5 main row 0.8219; drop 0.4509 > tolerance 0.02 | **FAIL (mechanical)** |
| **G2 context row (report-only)** | arm 0.3710 vs the Route A V2 baseline recompute 0.0903 on the same caliber: **+0.2807** | not part of the gate verdict |
| **G3 efficiency (report-only, no threshold)** | 5.03 h wall-clock; 45,828 steps; 977,608 training rows per epoch (677,608 base + 300,000 M1) | reported |
| **G4 mechanism** | not run — prereg executes G4 only when G1 is direction-positive | NOT TRIGGERED |

**Reading (negative direction is the deliverable, stated as such).** Adding a real 300,000-row external-domain MRL corpus on top of the same-domain library does **not** transfer to within-source Δ ordering on the MRL benchmark. The arm lands 0.2739 — 0.0419 below the frozen V2 three-seed ensemble (0.3158) and below all three V2 single-seed values (0.2873 / 0.3157 / 0.3198). Together with the synthetic-side falsification (D16-C, G1 FAIL gap 0.8235) and the parameter-side bound-anchored half-effect (ERK 0.2015 ≈ first-order bound 0.2069), the intervention triangle now reads: **supervision volume/density on the data side does not unlock MRL Δ**, consistent with the geometric reading (MRL within-source density 4.9 per source; the learnable part is essentially the first-order table). The single-variable design plus the frozen FINAL-EPOCH(6) rule precludes reading this as a tuning artifact; the pre-registered interpretation framework already classifies a FAIL as evidence that the density factor's independent contribution is bounded.

**Why G2 fails mechanically while the context row is strongly positive.** The mechanical G2 verdict is **not** taken from the contrast row: per the amendment, G2 mechanically compares against the V5 multi-task main row (the frozen anchor), while the arm inherits V2's MRL-only design. On that same MRL-only caliber, the same arm evaluated on polyA reads 0.3710 against its own V2 lineage's 0.0903 (**+0.2807**) — i.e. the added M1 corpus is far from inert; it materially reshapes cross-domain behaviour (a supervision-regime transfer effect), and this informative non-destruction reading is the contrast row, reported alongside the mechanical verdict rather than replacing it.

**Declarations.** (i) G1 is a single-seed direction read by prereg; the 3-seed CI branch was never entered because the direction is negative. (ii) No peak-picking: the verdict uses the final epoch only. (iii) Two further seeds (20260904 / 20260905) were trained under the pre-committed compute clause of amendment v2; because G1 is negative they remain **trained-not-analyzed** — archived, excluded from every table, figure and claim, and not used to dilute the G1 reading. (iv) Reporting gap: the archived efficiency block records `peak_cuda_memory_gb = 0.0` — the value was never captured by the runner, so no peak-memory number is claimed here.

### 5.7 Summary

The evidentiary grades are stated explicitly: the single-factor density claim is **downgraded** (held-out FAIL, honestly reported); the two-factor claim is supported by three lines of evidence: the observation, the intervention triangle, and the held-out deviation pattern — and is labeled EXPLORATORY.

The regularity says *whether* a task's delta is learnable in principle; the next question is *where the learnable signal lives* (table or structure), because that decides data vs model investment. §6 decomposes it.

---

## §6 Signal decomposition (first-order / structural)

### 6.1 The first-order closed-form energy table (method)

We replicate E7: a position-block × 64-context design matrix plus a per-source background term b[source], fit by ridge regression in closed form, with the ridge α selected on VALIDATION over the grid {0.3, 1, 3, 10, 30, 100}. The MRL reference pair (0.2069 vs ERK 0.2015) is maintained from the existing archive. A TRAIN-only α sensitivity check (5-fold GroupKFold cross-validation) recomputes the three tasks: polyA Δ = 0.0 (α = 100 in both cases, reading unchanged), MPRAU −0.0003, TE −0.0044 — the decomposition reading is fully robust to α selection, so the optimistic-bias caveat can be withdrawn.

### 6.2 Per-task results

**Figure 3 (extended render)** carries the full first-order accounting per task as a lollipop chart — first-order ρ₁, ours (V5), and the ICC ceiling per track, with the MRL reference pair (E7 0.2069 vs ERK 0.2015) included as a fourth track; ρ₁ values are read live from `analysis_first_order_decomposition_v1/results_first_order.json` and asserted at render time, and the per-task first-order share of ours (polyA 55%, MPRAU 80%, TE 79%) is annotated in the right margin.

**polyA.** First-order additive table reads 0.4555 = 55.4% of V5's 0.8219 (Figure 3, polyA track; Figure 6, polyA panel). Comparison rows: the externally supervised band 0.71–0.75; APARENT 0.7343; V5 0.8219; ICC 0.90.

**MPRAU.** First-order reads 0.0825 (pair-mean; record-level 0.0630) = 80.5% of V5's 0.1025 (Figure 3, MPRAU track; Figure 6, MPRAU panel). The external band is ≈ 0.016 with a CI crossing zero (0.0164); Saluki weak-control 0.1205; ICC 0.683.

**TE.** First-order reads 0.0458 = 79.1% of V5's 0.0579 (Figure 3, TE track; Figure 6, TE panel); the external band is 0.0009–0.0113 (0.0061).

**MRL (reference pair).** The E7 closed-form first-order bound reads 0.2069, with ERK v2's frozen validation at 0.2015 anchoring the parameter-side rung (Figure 3, MRL track; Figure 5, parameter panel); ICC 0.83.

### 6.3 Task heterogeneity (main conclusion)

The decomposition narrative does not transfer across tasks. On polyA the first-order term explains only 0.46, leaving 0.27+ as non-first-order (structural / contextual-composition) signal that external LMs partially capture; on MPRAU and TE our rows are almost entirely first-order signal (V5 increments of only +0.02 and +0.01), so "structural / higher-order gain" must be stated with restraint on these tasks. This is complementary to the G4 structural blindness and the delta-validity within-source finding.

### 6.4 Two-sided reading of external rows

"No dense supervision" ≠ "no signal": polyA external rows (0.71–0.75) generally *exceed* the first-order bound (general-LM pretraining transfer already exceeds the first-order table), whereas MPRAU/TE external rows read ≈ 0 (they do not even capture first-order signal). The per-task edit density is consistent with the first-order coverage gap: polyA averages 8.5 edits/record, the only task with combinatorial space.

### 6.5 Decision table (pre-registered rules)

A = |ρ₁ − band median| ≤ 0.05; B = dense-row − ρ₁ > 0.05. polyA is partially satisfied (✗A 0.2747 / ✓B 0.2788); MPRAU is not established (first-order exceeds the band by 0.0661); TE is partial (B not applicable). The task-level gap is polyA 0.2747 versus MPRAU increment 0.0200 and TE increment 0.0121.

The decomposition + regularity jointly predict which failure mode each task is in; §7 turns that into the four-class taxonomy and the practical selection guide, and §8 takes the one boundary case (polyA) to its positive endpoint.

---

## §7 A taxonomy of failure

### 7.1 Classification rules (rules before labels)

**C1 physical failure**: label ICC < 0.1 and a whole-row noise band. **C2 paradigm failure**: input-distribution / context mismatch plus a whole-row |ρ| < 0.05 band. **C3 geometric failure**: density < 10 plus structural evidence. **C4 supervision failure**: no same-distribution corpus plus oracle NO_SIGNAL or an all-zero same-pool result. **C5 non-failure control**: dense-supervision regime rows (≥ 280K).

### 7.2 Main classification (20 rows: 14 failure + 6 control; covering 9/9 tasks)

**Figure 8** renders the taxonomy as a task × class matrix: rows are the affected tasks/domains, columns are the five classes (C1–C5, definitions in the figure and §7.1), and each dot's size encodes the number of rows (5 + 5 + 2 + 5 + 6 = 20). Representative frozen values per class. C1 — HL5 (ICC 0.0014), HL3 (0.013), TE149487 (ICC −0.274, split-half 0.050), LAMAR cross-evidence 2 (HL5 ρ_ε 0.6863; HL3 0.4429). C2 — RiboNN 3 (−0.0106/−0.0290/+0.0778/−0.0450), Saluki 2 (MPRAU 0.1205, weak-control CI [0.077, 0.164]). C3 — MRL (density 4.9; G4 three-generation 0; E7 0.207), MPRAU (6.1; CI crossing zero). C4 — MRL-oracle NO_SIGNAL (0.0869 / 0.0700), MPRAU, TE, RNA149487 (edge), REFALT (edge). C5 — Optimus 0.3132, ours 3-seed 0.3158, APARENT 0.7343, APARENT2 0.6810, V5 0.8219, general LM polyA 0.7114–0.7490.

### 7.3 Task-level roll-up and cross-class annotation

Primary and secondary classes are layered: MRL is C3-primary with C4-secondary (geometry first, representation-side no-signal second). Three edge decisions: GSE149487-RNA → C4-primary with C1-secondary (power-limited); GSE186455 → C4; LAMAR → C1 with ρ_ε cross-evidence.

### 7.4 Control rows and the anti-one-sided narrative

The same general LM that reads 0.71–0.75 on polyA (density 126.7, 2.74M corpus, ICC 0.9) with an EQUIVALENT oracle, reads across zero on low-density tasks without same-distribution corpora (RNA-FM: polyA 0.7114 vs MRL 0.1369 (seed-sensitive) vs HL ≈ 0). The failure attribution therefore lives in the *data regime*, not in an "all general LMs are inadequate" assertion.

### 7.5 Practical implications (toward a model-selection guide)

C2 → align input granularity; C3 → within-source dense candidate supervision with pair-mean/within-source caliber; C4 → run a low-cost oracle probe first (NO_SIGNAL ≤ 0.10 is a stop-loss) before committing, or await corpus, or use end-to-end FT (with the caveat that MPRAU matched-FT is negative across all three seeds, pair-mean −0.075 to −0.107: FT does not rescue it); C5 → find the corpus before discussing architecture.

The taxonomy equips the practitioner to classify any new task before modeling; §8 demonstrates the full pipeline on the sole task that passes all gates — from ceiling ruler to 91.3% completion and two application outlets.

---

## §8 polyA: from boundary endpoint to a full positive-result chapter

### 8.1 The ceiling ruler

**Task side.** polyA (GSE269595 [3], n = 2,628, 78 source groups) is the only one of the 13 cells with ICC = 0.90 (unified label-ICC source), so its label signal-to-noise ceiling is explicit; the other tasks (ICC 0.001–0.83) do not support the same "approachable ceiling" reading. ICC is a *necessary but not sufficient* upper bound.

**Geometry side.** The edit density is 126.7 candidates/source (the lone order-of-magnitude outlier in the 25-row fit set), with an average of 8.5 edits/record — the only task with combinatorial space among the 13.

**Corpus side.** The APARENT lineage [16] carries a 2.74M within-source dense-supervision corpus (density and in-domain supervision co-vary in the fit set; §5.3's two factors are simultaneously in place on this task).

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

The three-seed mean is 0.819573 with range width 0.0041, far inside the pre-registered 0.03 variance bound, and the source-group paired bootstrap (2,000 iterations, seed 20260920) gives Δ vs APARENT (recomputed 0.734315) = **+0.085258, CI95 [+0.062098, +0.107759], excluding zero** (cross-check with the frozen protocol seed 20260816: +0.085388, CI [+0.062160, +0.110269]). The Holm family direction recomputes to raw p = 0.0005 → **holm p = 0.002, significant** (the frozen table row stays at 0.001; the recomputation is a direction check, not a replacement). Both pre-registered upgrade conditions are met (mean ≥ 0.80 and CI excluding zero), so the polyA claim is upgraded to a **three-seed robust** reading and this attack surface is closed. Per the prereg the main row is never replaced: 0.8219 remains the frozen headline and `polyA-V5-3seed-mean` is the appended robustness row. The decision-caliber mixed result (§8.2) is unaffected by the supplement and is reported there in the same paragraph as the rank-caliber win.
>
> Pre-registered supplement (mini-prereg, freeze 73d47cdf): the V5 polyA main row is a single-seed frozen terminal state (0.8219). The supplement re-runs the identical V5 configuration with two additional seeds (20260921 / 20260922, giving a three-seed set 20260907 + the two new seeds), with the only change being the seed. The main row 0.8219 is retained unchanged under all circumstances; a new row `polyA-V5-3seed-mean` (with CI and per-seed values) is appended, framed as a robustness reading and never a SOTA push. The reported main judgment is the three-seed mean ± range against the main caliber (GSE269595 VALIDATION n = 2,628 overall Spearman), with a source-group paired bootstrap 95% CI (2,000 iterations, seed 20260920). Upgrade rule: if the three-seed mean is ≥ 0.80 and the CI excludes zero, 0.8219 is upgraded to a "three-seed robust" claim (attack surface SEALED); if the seed variance is large (range > 0.03), the claim is honestly rewritten as an interval statement without selection; a failed seed is recorded as FAILED with a "2-seed + declaration" downgrade, never silently retried with a different seed.

### 8.2 Results (dual caliber reported separately)

**Rank caliber (primary).** V5 0.8219 vs APARENT frozen-delta 0.7343: **Δ+0.0876, bootstrap CI [0.0630, 0.1141], excluding zero, Holm p = 0.004 — the only family-wise significant win across the whole bottom-line nine-task family.** The TEST-confirmed status comes from the one-shot Gate P unblinding (receipt on file): polyA TEST Spearman 0.8205 vs VALIDATION 0.8219, Δ −0.0014. The family is the bottom-line adjudication table of nine ours-vs-strongest-external pairings; polyA raw_p = 0.001 → Holm p = 0.004. The evaluation surface is n = 2,628 VALIDATION records with a source-group paired bootstrap (2,000 iterations, seed 20260816).

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

**(3) Multi-task isolation causal chain (intervention evidence).** CMS mixing on the polyA side is −0.2261 (option-3 CMS augmentation adjudication per-task side-effects: polyA −0.2261 / MRL +0.1809 vs the control recipe task profile) — introducing heterogeneous-corpus mixed supervision collapses the polyA caliber. By contrast, W0 single-task from-scratch MRL training reads 0.1987 vs V5 multi-task 0.1354 = **+47% relative improvement** (multi-task dilution confirmed); the V6–V9 multi-generation mixed-training failure history (V8-S specialization cost polyA −0.16; V9 S1/M6 holdout unlearnable) is mentioned in one sentence (archived in journal batches 76/85/86). Single-task isolation plus same-assay dense supervision is the training-side condition for polyA's high completion.

### 8.4 Application outlets (two levels: generation-line guidance; committee routing)

**Generation line (the COMB 2×2 mechanism experiment).** Exploration (B) / guided (C) / both-arms (D) over 891 sources × 3 seeds, tier-2 full set with the two-tier adjudication discipline. The sc-caliber near-perfect additivity fingerprint is **B+C +0.02974 vs D +0.02972** (difference 0.00002); verdict H_IND_or_NEGATIVE (H-int not significant; exploration gain fully retained, guidance non-destructive, D ≈ C). The polyA third-caliber three rows in the D arm (APARENT frozen scoring, n = 20 report-only) are D_main +0.3527 / seed16 +0.4210 / seed17 +0.4844 (B +0.2328 / C +0.0596 for reference); n = 20 is the *entire* polyA source set within the 891 pool, so the expansion ceiling is exactly 20. The unguided baseline is 0.12046 (891 sources, 28,512 candidates, uniqueness 0.8572, legality 1.0). The critic self-score, the independent evaluator, and the measured calibers are kept separate, and H_IND is reported as-is (near-perfect additivity is positive evidence of orthogonality, not a failure).

**Committee routing (the zero-peek task-routing experiment).** A constructive zero-peek task-routing scheme (dev-only selection, holdout evaluation, paired bootstrap 2,000 iterations, seed 20260816) gives **Δ+0.0362, CI [+0.0090, +0.0632], excluding zero** on holdout n = 446 (vs single-V5 probe@1 0.0443 → 0.0806; per-task routing MRL → v8_hbench9 / MPRAU → v8_smprau_in / polyA → v8_smprau_in / HL → full). Reported honestly alongside it, the dev-routed variant gives +0.0045, CI [0.0000, +0.0112], crossing zero (dev-selected task-level routing generalizes insufficiently); the constructive version is the pre-registered constructive routing (not a data-peeking selection), and the two variants are layered and reported. Both are reported — never only one.

**Outlet positioning.** Both outlets in §8.4 are demonstrations of the mechanism conclusions, not engineering claims. The committee-routing CI lower bound (+0.009) is close to zero, so the effect-size claim must carry a power boundary (see the rigor-audit item A6, §9.3).

**Table 12. Application outlets.**

| Outlet | Variant | Δ | 95% CI | n | Reading |
|---|---|---|---|---|---|
| Generation line (COMB) | B+C vs D additivity | +0.02974 vs +0.02972 | — | 891 × 3 seeds | H_IND_or_NEGATIVE |
| Generation line (polyA D arm) | D_main / seed16 / seed17 | +0.3527 / +0.4210 / +0.4844 | — | 20 (report-only) | all-pool polyA sources |
| Committee routing | constructive zero-peek | +0.0362 | [+0.0090, +0.0632] | 446 | excludes zero |
| Committee routing | dev-routed | +0.0045 | [0.0000, +0.0112] | 446 | crosses zero |

polyA closes the positive branch; §9 assembles the general guide (C1–C5, triple screening), confronts limitations honestly, and §10 supplies the reproduction apparatus behind every number cited in the chain.

---

## §9 Discussion

### 9.1 A model-selection guide

Select by task regime first. C1 tasks (ICC < 0.1): do not select a model; change the measurement or add replicates. C2: align input granularity first (fragment vs full length). C3: within-source dense candidate supervision with pair-mean/within-source caliber. C4: run a low-cost oracle probe first (NO_SIGNAL ≤ 0.10 is a stop-loss) before committing. C5: corpus volume determines success (external references at 280K / 2.74M). A companion "data-side triple screening" (ICC / density / corpus-domain mapping) should precede architecture selection.

### 9.2 Impact on absolute-score leaderboard practice

Between-source composition inflates overall scores (polyA overall 0.71–0.75 vs within-source 0.19–0.23), and in-domain corpus binding (STCNet 0.8144 with 730/730) is invisible without a source-relative protocol. We recommend that benchmarks report the dual task, the within-source caliber, and a corpus-overlap declaration.

### 9.3 Limitations (itemized honestly)

**Rights / data.** GEMORNA / UTR-STCNet / UTR-Insight have no declared license (user exemption decision 2026-09-20, port-ledger addendum commit 5a8e30eb); M6 has a seal/unseal history (`historical_exposure`; the words sealed/untouched/never-seen are prohibited); the payload release boundary stands at 0 authorized public study-payload rows (v332 rights table).

**Single-seed probes.** The 65 new-family matrix cells are single-seed frozen-Δ evaluations (row-construction seed frozen 20260920; model inference is deterministic but row sampling is single). The batch-2/3 generalist probe rows (RiNALMo/ERNIE-RNA and the 7 bespoke-adapter backbones of §3.5) are likewise single-seed frozen-Δ evaluations. The oracle / first-order analyses are mostly three-seed (the MRL seed-sensitive pseudo-signal is itself the evidence for single-seed artifacts); the matrix has no multi-seed variance band (registered honestly).

The V5 polyA main row 0.8219 is a frozen single-seed terminal state, but its training-seed variance is now measured by the three-seed supplement (§8.1: mean 0.8196, range 0.0041, Δ vs APARENT CI [+0.0621, +0.1078] excluding zero), which closes the polyA part of this limitation; the paired APARENT row remains a single pass.

**ICC provenance items.** MRL 0.83 and REFALT 0.21 are historical reference values whose computation procedure / n are not archived — **PROVENANCE_UNRESOLVED** — and are treated as limitations, not re-run. (By contrast, the recomputation block closes 7/9 tasks: polyA 0.9041 ≈ 0.90 reproduced consistently; TE200304 floating-point-consistent; MPRAU bracketed at 0.7273 [0.694, 0.766], covering the 0.683 magnitude; HL5/HL3 near-zero confirmed.)

**Inference determinism.** Four of five families are bitwise deterministic; STCNet's stochasticity is by design (official DPC clustering adds tie noise; cell spread 8.4e-04 measured on the MRL cell), and the archived 0.8144 lies within the re-run interval; per-cell power on the new rows shows a median CI width of 0.077 and median MDE ≈ 0.037.

**Honest handling of the FAIL verdict.** The single-factor density held-out FAIL has been executed per the downgrade clause (the only upgrade path is append-only plus re-evaluation at the next decision point); the two-factor claim is EXPLORATORY (not a pre-registered decision gate). The M1 intervention arm is now backfilled (§5.6) and reports a **negative direction gate** with the pre-registered FAIL branch — real-data corpus addition did not move MRL Δ, which bounds the density factor's independent contribution and is consistent with the geometric reading.

**VALIDATION-only -> updated 2026-10-07 (Gate P unblinded).** All matrix / leaderboard numbers above are from VALIDATION splits; the protected TEST split (18,292 rows) was read exactly once after full-paper v2, per the frozen unblinding prereg (receipt: manifest/checkpoint/record-set sha256s on file). Outcome: the headline claims are TEST-confirmed (polyA 0.8205 vs 0.8219, Δ −0.0014; polyA remains the strongest task by a wide margin; macro 0.1353 vs 0.167, Δ −0.032), while the small-sample rows reproduce the sign-instability already registered above (PLUMAGE-TE 0.1953 → −0.047 at n=48; REFALT 0.0639 → −0.077 at n=274) with all non-polyA magnitudes staying inside the near-zero band (|ρ| ≤ 0.18) — the band-structure reading holds at the magnitude level, and the sign-level limitation is now double-split evidence. The G4 structural probe's three-generation zero holds only under the D15-2 caliber.

**Fairness boundary (R12 residual).** The frozen-Δ zero-tuning caliber is defensible against the "you under-tuned the external models" objection only indirectly (via the matched-FT evidence, R1/R5); new-family matched-FT pairings were not executed (out of matrix prereg scope, registered honestly). Domain mapping is documentation-level for 4/5 families and measured-level for 1. **Rights/payload (R13 residual).** The availability statement cannot be finalized until study-specific rights review completes; exemption decisions are not publication authorization.

### 9.4 Relation to existing benchmarks and models

The four-axis differentiation (Table 2) positions DeltaBench against conventional absolute-endpoint benchmarks along source-relative formalization, dual-task reporting, ceiling normalization, and two-tier adjudication.

Concretely, the underlying assays of several benchmark studies publish rich measured libraries without a source-relative formalization — the 280K random 5′UTR library (Sample et al. 2019 [1]), the PLUMAGE screens (Lim et al. 2021 [6]), the prostate 3′UTR reporter library (Schuster et al. 2023 [5]), the HGMD/ClinVar UTR stability panel (Su et al. 2025 [7]) and the MPRAu allelic panel (Xue et al. [4]) — all re-used here as canonical records with Δ labels rather than as absolute endpoints. Published UTR predictors are scored on identical VALIDATION records in one frozen-Δ caliber (UTR-LM [21]; RNA-FM [20]; APARENT [16]; APARENT2 [17]; Saluki [19]; Optimus [18]; FramePool [18-family]; RiboNN [35]). The absolute-vs-delta decoupling of §4 is therefore read on the same records instead of across incompatible reportings.

Recently released UTR predictors are ported with fully declared input adaptations and unit-tested against official outputs (GEMORNA [29]; LAMAR [28]; UTR-STCNet [30]; UTR-Insight [31]; HydraRNA [26]), and their matrix rows (Table 4) locate the in-domain / cross-library boundary quantitatively. Ceiling-normalized learnability accounting (Table 3) and the two-tier adjudication discipline (§2.4) are not standard in prior UTR model reporting, and the three append-only evaluation rows (Castillo-Hair 2024; Plassmeyer 2025; Su et al. 2025) provide fresh surfaces that were frozen before any model was scored on them. The full reference list (including the model papers for the ported baselines and the right/permission statements) is assembled at the submission stage; no external reference numbers are re-cited here, to keep every reported quantity traceable to this project's frozen artifacts.

---

### 9.5 Anticipated reviewer objections and where the evidence answers them

| # | Objection | Answer (evidence locus) |
|---|---|---|
| O1 | "Embedding difference does not imply property difference — the probe premise is untested." | The premise is the paper's measured subject: the 0.873→0.313 decoupling with ρ_ε = 0.720 (§4.1); the first-order condition per task (§6.2); and the probe itself as a two-branch test — polyA HEAD/EQUIVALENT vs MRL NO_SIGNAL oracle verdicts under the same protocol (§4.4). See §4.5 Q1 for the full chain. |
| O2 | "The ported backbones never saw delta data — near-zero is expected and uninteresting." | The probe arm *supplies* delta supervision at read-out and still reads ≈0 on 8/9 tasks, locating the deficit in the frozen representation; same-supervision contrast (polyA 0.75 vs MRL 0.04–0.13) makes the supervision regime × geometry the operative variable; the M1/G1 intervention arms close the loop (heterogeneous corpus fails, same-source dense supervision works). Full chain in §4.5 Q2. |
| O3 | "In-domain 0.814 may be corpus memorization." | 730/730 evaluation sequences are empirically contained in UTR-STCNet's training corpus; the cross-library M1 row collapses to 0.067 (§3.3); a same-library weak-supervision control (UTR-Insight 0.2739) separates density from domain membership. |
| O4 | "Single-seed probes make the near-zero band unreliable." | Registered as a limitation (§9.3); the MRL seed-sensitive pseudo-signal (0.1369/0.0640/0.0597) is itself shown in full; the LOSO sign flips are registered as noteworthy cells without significance claims; polyA's key claim is closed by the three-seed supplement (mean 0.8196, range 0.0041). |
| O5 | "The batch-2 adapters may mis-port the backbones." | Port discipline: 7/7 adapters unit-referenced to official code paths, strict loading (missing = 0), port-validated runner (|Δ| = 2.4×10⁻⁶ vs the archived RNA-FM row); the one incident (WordLevel tokenizer collapse to [UNK]) was caught, fixed per the official recipe, and disclosed in §3.5 with a non-degeneracy verification. |

---

## §10 Methods

### 10.1 Dataset construction

**Why curated collection is scarce.** Source-relative delta evaluation requires measured libraries in which multiple candidates share one measured source under the same assay — a property most public MPRA deposits lack. Screening the public repositories under the frozen audit yielded 15 studies that qualify (12 NCBI GEO series, 1 ENCODE experiment, 1 EMBL-EBI BioStudies study, plus their SRA/BioProject records): eight study units contribute qualified paired records; the remaining studies serve as training corpus, historical-transfer diagnostics, or are sealed/unconvertible with reasons recorded per study in the qualification table. No synthetic or generated candidate contributes any benchmark credit.

**Collection.** Every study was acquired through its official public route (GEO/ENCODE/EBI), with per-study converters written against the primary deposit (not intermediate files); acquisition manifests record URL, SHA256, row counts, and split assignment for each. Data-use terms were reviewed study-by-study (GEO disclaimer, ENCODE TOS, EMBL-EBI terms; §10.6): 15/15 studies permit analysis and publication; raw payloads are not redistributed, and the release instead ships the seed-frozen converters that regenerate every canonical record from the public accessions.

**Construction.** The pipeline has three layers. The **canonical layer** holds the nine study converters (e.g. the GSE114002 mother-family identity / candidate rules). The **projection layer** (`development_train_validation_v1`) carries the `direction_normalized_delta` labels over 126,165 records (89,580 TRAIN / 18,293 VALIDATION / 18,292 TEST) under the frozen near-duplicate-component split. The **new evaluation-row layer** (M1/M6/S1, append-only) declares each row's caliber fully: M1 uses prefix-family plus Hamming-1 fallback pairing; M6 uses a SIC → fraction mapping with a pseudocount of 1.0 and a 155-nt window; S1 uses canonical_115 dual-assay sub-rows with the protected exclusion and the Dao QC flag-keep. New-data admission passes four gates (manifest; pigeonhole leakage check; cryptic-splicing QC; effect-size stratification audit); the M1 corpus passed all four before entering training.

### 10.2 Evaluator

All evaluation uses the same Task-1 evaluator instance (`evaluate_route2_prediction_v1`, K = 10), computing source-group within-source Spearman and task-macro Spearman; the MPRAU caliber is pair-mean (2,008 variants). The **independent re-check protocol** does not read archived metrics and recomputes every cell from `predictions.jsonl` with a |Δ| ≤ 1e-6 gate (achieved 65/65 with max |Δ| = 0.0).

### 10.3 External model porting

The port ledger records 5 PORT_READY families plus the license disposition. Per-family port adaptations are declared in the matrix summary's verbatim blocks. The unit-test protocol verifies official example values and strict loading (UTR-Insight official CSV alignment ρ = 0.96; RiboNN official predictions aligned to model-0 max diff 0.0034 and model-1 max diff 0.0012, reproducing the official Pearson 0.758). The HydraRNA MIG environment fix (Triton autotuner single-device visibility) is declared with zero numerical change.

### 10.4 Pre-registration document list

All decision thresholds were frozen before computation. The full commit-hash list (commits that introduced each frozen document on the W0 branch `route-a-v3-w0-diagnosis-20260902`): `delta_density_prereg_framework_v1` (682b8c80); `delta_validity_oracle_prereg_v1` (a2a2919a); `first_order_decomposition_prereg_v1` (8f80d7e6); `benchmark_v2_matrix_row_prereg_v1` + `benchmark_v2_port_ledger_v1` (a14fb447); `m1_intervention_arm_amendment_v1` (b2e141cd); `polya_3seed_mini_prereg_v1` (73d47cdf); `density_sensitivity_mini_prereg_v1` + `first_order_train_alpha_sensitivity_prereg_v1` (60ff6999); `route2_critic_d16c_within_source_amendment_v1` (0809f5f6); `route2_critic_erk_v2_amendment_v1` (e343d4a3); `route2_w_ladder_amendment_v1` (7303417c); `utr_editflow_goal_v2_amendment_pivot_v1` (682b8c80); `route2_setflow_comb_mechanism_prereg_v1` (eb69e6bb, v8-stage1 worktree). Scope boundary: the family range is maintained at 14 (5 new + 9 existing); M5 is not included (no pre-registered row / port evidence). The batch-2 generalist-backbone append (7 adapters, frozen-probe caliber) is governed by `benchmark_v2_generalist_rows_amendment_v2.md` (DESIGN FROZEN); its execution commit is `6ef7cc6c` (journal 134; port-validation |Δ| = 2.4e-06 against the archived RNA-FM MRL row).

### 10.5 Reproduction entry points

Row-construction scripts (`build_{m1,m6,s1}_*_v1.py`, with construction seeds archived for re-runs); the matrix execution / aggregation / re-check scripts (eight files); the analysis scripts (e.g. `run_delta_vs_density_v2.py`). **R2 internal target rows** cover all nine tasks (global-scaled internal control, macro 0.1317). The full LOSO reference table (protocol B — the frozen LOSO Table 5, with the per-task readings for all nine tasks) is reproduced in Supplementary S2; here we summarize its structure. It lists, for all nine tasks: the multi-task in-domain critic V5 reading (explicitly not zero-shot), the external same-pool rows where such rows exist (MRL / polyA / MPRAU / HALF_LIFE only; TE / RNA / REF-ALT have no external same-pool rows and are declared as such), the internal target, and the ceiling. Representative worked rows: MRL 0.1354 vs frozen-Optimus 0.3132 / FramePool 0.2956 / UTR-LM 0.1107, internal 0.1192, ceiling 0.83; polyA 0.8219 vs APARENT 0.7343 / APARENT2 0.6810 / UTR-LM 0.7490, internal 0.7308, ceiling 0.90.

### 10.6 Leakage and rights audit

Pigeonhole audit: S1/M6 flagged = 0; M1 has 4 TRAIN overlaps with keep = 0. The **STCNet old three-arm INVALID case** is archived (training-set leakage determination; three arms 0.8135 / 0.2667 / 0.2120 all excluded). The rights-boundary table is the v332 study-rights table (public release: 0 authorized rows). The accountable rights review (owner: TRAE agent, user-assigned, 2026-10-06) over all 15 studies is archived as `data_rights_audit_v1.md` (commit 2d7abc5d) with the filled review CSV (commit c2cb7355): 15/15 analysis-and-publication use permitted; payload redistribution conservatively not authorized.

### 10.7 §8 support protocol blocks

**polyA main-row statistics.** Source-group paired bootstrap (2,000 iterations, seed 20260816) plus a Holm step-down family = the bottom-line nine-task pairings. The polyA main caliber is the GSE269595 VALIDATION n = 2,628 overall Spearman; the decision caliber includes top-1 and NDCG@10 (K = 10). The 3-seed supplement uses the V5 configuration with only the seed changed (main row unchanged).

**COMB 2×2 tier-2 protocol.** 891 sources × 3 seeds; tier-1 calib100 falsification right; repaired adjudicator fingerprint. Calibers are separated: critic self-score / independent evaluator / measured.

**Committee routing dev/holdout protocol.** Stratified 50/50 split (seed 20260909), dev-only argmax, constructive zero-peek variant definition, paired bootstrap (seed 20260816).

**New-data admission: four gates.** Before a corpus may enter training: (1) a manifest (URL + SHA256 + row count + split); (2) an R3 pigeonhole check against the full protected split; (3) cryptic-splicing QC; (4) an effect-size stratification audit (the CMS lesson). The M1 corpus passed all four gates (pigeonhole exact overlap: 4 rows, all in TRAIN; near-duplicate audit: 0 full-length overlaps) before entering training; on the evaluation side, 4,848 unique sequences (with the four TRAIN-overlapping sequences pigeonhole-flagged) were excluded throughout.

**STCNet stochasticity note.** STCNet / the new families are evaluated with a **single frozen forward pass**; the official stochastic DPC clustering introduces run-to-run spread (8.4e-04 measured on the MRL cell), and the archived 0.8144 lies within the re-run interval. Rows with a miss rate > 15% carry the HIGH_MISS marker.

**Freeze discipline and limitations recap.** Frozen-Δ protocol (official weights, zero tuning, input adaptation declared per family); two-tier adjudication (tier-1 calibration falsified by tier-2 full set); ceiling normalization (ICC ceilings with N/A handling for ≈0 rows); four admission gates; STCNet stochasticity as above. Limitations include the single-seed matrix rows, the VALIDATION-only scope (protected TEST reads = 0), and the MRL 0.83 / REFALT 0.21 ICC **PROVENANCE_UNRESOLVED** items (historical reference values whose computation is unarchived).

---

## §11 Conclusion

DeltaBench reframes mRNA edit-effect prediction around the source-relative quantity that downstream users actually consume. With a frozen matrix, a rigorous re-check, and a pre-registered adjudication discipline, it demonstrates that absolute accuracy and delta accuracy are distinct competencies. The matrix's headline is a supervision-regime effect: strong in-domain, weak cross-library. The mechanism analysis attributes systematic delta-learning failures to the interaction of supervision regime and data geometry, while honestly downgrading the single-factor density claim after a held-out failure. polyA is the single task where the label ceiling becomes a non-trivial reading: V5 reaches 91.3% of an ICC-0.90 ceiling, the only family-wise significant win, with a decision-caliber deficit reported alongside it. Two application demonstrations (generation-line guidance and committee routing) show that the mechanism conclusions are actionable, with explicitly bounded effect sizes.

---

## Data availability

DeltaBench is released as a curated, reusable benchmark resource. All evaluation records are derived from 15 publicly deposited studies (12 NCBI GEO series, 1 ENCODE experiment, 1 EMBL-EBI BioStudies/ArrayExpress study, and their associated SRA/BioProject records), each cited by accession in Supplementary Table S1 together with the primary publication. Consistent with the NCBI GEO disclaimer (submitters may retain IP rights; NCBI cannot grant unrestricted redistribution permission) and the ENCODE/EMBL-EBI data-use terms, **raw source payloads are not redistributed with this manuscript**. Instead, we release: (i) per-study converters and row-construction scripts (seed-frozen, deterministic) that regenerate every canonical evaluation record from the public accessions; (ii) the frozen evaluation manifests (record IDs, splits, and grouping) with per-record summary statistics; (iii) all model predictions, metrics, and adjudication JSONs; and (iv) the full reproduction pipeline. This converter-based release makes the entire 126,165-record benchmark reconstructible and re-usable by future edit-effect models without re-deriving the curation. No "available on request" channel is promised for raw payloads; access follows each provider's public route.

## Code availability

All pipeline code (converters, row construction, frozen-delta evaluators, probe protocol, adjudication and re-check scripts, figure producers) is released at the project repository (GitHub, `Cunyu-Liu/mRNA_editflow`, branch `route-a-v3-w0-diagnosis-20260902` and the setflow branch) under the repository license; every frozen artifact referenced in the text carries its SHA-tagged producer script. Third-party model weights are loaded from their official releases under each provider's terms (MIT / Apache-2.0 / NVIDIA Open Model / AGPL-3.0 as registered in the port ledger); weights are never redistributed by this project.

## References

**Evaluation studies (by benchmark role).**

1. Sample, P. J. et al. Human 5' UTR design and variant effect prediction from a massively parallel translation assay. *Nat. Biotechnol.* **37**, 803–809 (2019). doi:10.1038/s41587-019-0164-5 (GSE114002; MRL row; 280K library.)
2. Castillo-Hair, S. et al. Optimizing 5'UTRs for mRNA-delivered gene editing using deep learning. *Nat. Commun.* **15**, 5284 (2024). doi:10.1038/s41467-024-49508-2 (GSE232927; M1 row + M1 intervention corpus.)
3. Kowalski, M. H. et al. Multiplexed single-cell characterization of alternative polyadenylation regulators (CPA-Perturb-seq). *Cell* **187**, 4408–4425 (2024). doi:10.1016/j.cell.2024.06.005 (GSE269595; polyA row.)
4. Griesemer, D. et al. Genome-wide functional screen of 3'UTR variants uncovers causal variants for human disease and evolution. *Cell* **184**, 5247–5260 (2021). doi:10.1016/j.cell.2021.08.025 (ENCSR854RUF / ENCODE; MPRAU row.)
5. Schuster, S. L. et al. Multi-level functional genomics reveals molecular and cellular oncogenic drivers in prostate cancer. *Cell Rep.* **42**, 112840 (2023). doi:10.1016/j.celrep.2023.112840 (GSE200304; TE 3'UTR row.)
6. Lim, Y. et al. Multiplexed functional genomic analysis of 5' untranslated region mutations across the spectrum of human prostate cancer (PLUMAGE). *Nat. Commun.* **12**, 4217 (2021). doi:10.1038/s41467-021-24445-6 (GSE149487; LOSO TE/RNA rows.)
7. Su, J. Y. et al. Multiplexed assays of human disease-relevant mutations reveal UTR dimer composition as a major determinant of RNA stability. *eLife* **13**, e97682 (2025). doi:10.7554/eLife.97682 (GSE217518; HL rows + S1 stability row.)
8. Lagunas, T., Jr. et al. A Cre-dependent massively parallel reporter assay allows for cell-type specific assessment of the functional effects of 3'UTR genetic variants in vivo. *Commun. Biol.* **6**, 1151 (2023). doi:10.1038/s42003-023-05483-w (GSE186455; REF/ALT row.)
9. Plassmeyer, S. P. et al. A massively parallel screen of 5'UTR mutations identifies variants impacting translation and protein production in neurodevelopmental disorder genes. *medRxiv* (2023). doi:10.1101/2023.11.02.23297961 (GSE246381; M6 row; preprint as deposited.)
10. Lewis, C. J. T. et al. Quantitative profiling of human translation initiation reveals elements that potently regulate endogenous and therapeutically modified mRNAs. *Mol. Cell* **85**, 445–459 (2025). doi:10.1016/j.molcel.2024.11.030 (GSE256185; training corpus, DART.)
11. Fu, T. et al. Massively parallel screen uncovers many rare 3'UTR variants regulating mRNA abundance of cancer driver genes. *Nat. Commun.* **15**, 3335 (2024). doi:10.1038/s41467-024-46795-7 (GSE232572; training corpus.)
12. Jia, L. et al. Decoding mRNA translatability and stability from the 5' UTR. *Nat. Struct. Mol. Biol.* **27**, 814–821 (2020). doi:10.1038/s41594-020-0465-x (GSE145046; training corpus.)
13. Diez, M. et al. iCodon customizes gene expression based on the codon composition. *Sci. Rep.* **12**, 12126 (2022). doi:10.1038/s41598-022-15526-7 (GSE207584; training corpus, synonymous codon library.)
14. Miliotis, C. et al. Determinants of gastric cancer immune escape identified from non-coding immune-landscape quantitative trait loci. *Nat. Commun.* **15**, 4319 (2024). doi:10.1038/s41467-024-48436-5 (GSE261709; training corpus, 3'UTR ilQTL MPRA.)
15. Mendonsa, S., von Kügelgen, N., Dantsuji, S., Ron, M., Breimann, L., Baranovskii, A., Lödige, I., Kirchner, M., Fischer, M., Zerna, N., Bujanic, L., Mertins, P., Ulitsky, I. & Chekulaeva, M. Massively parallel identification of mRNA localization elements in primary cortical neurons. *Nat. Neurosci.* **26**, 394–405 (2023). doi:10.1038/s41593-022-01243-x (E-MTAB-10902 is this study's ArrayExpress accession; training corpus, N-zip MPRA.)


## Cover letter (bioRxiv submission draft)

Dear bioRxiv team,

We submit "DeltaBench: A Source-Relative Benchmark Reveals Why Absolute-Score Models Fail to Predict mRNA Edit Effects" for consideration as a preprint.

mRNA edit-prioritization pipelines consume the ranking of sequence *differences*, but current leaderboards score only absolute accuracy — and no public dataset exists to score the delta quantity itself, because the required paired libraries (multiple measured candidates per source, same assay) are scarce in public MPRA deposits. Our headline decoupling — 0.873 absolute vs 0.313 delta for the same frozen model — shows these are distinct competencies. DeltaBench closes the resource gap: 126,165 curated paired records from 15 public studies, released with seed-frozen converters and the full re-check pipeline as a reusable community asset. It measures the quantity practitioners actually use, attributes its failures to supervision regime × data geometry via pre-registered interventions, and demonstrates the sole passing task (polyA) at 91.3% of its measurement ceiling, confirmed by a one-shot held-out TEST unblinding with a full audit receipt.

All evaluation is built from 15 public studies (cited by accession; raw payloads not redistributed, per provider terms — converters and manifests are released instead), all decision rules were frozen before computation, and every number is traceable to archived artifacts.

Sincerely,
The Authors

## bioRxiv submission checklist (W4)

- [x] Manuscript (this draft, v2.3 post-unblinding; abstract ~250 words)
- [x] Title — **T1′ FINAL** (user decision 2026-10-07; alternates retired)
- [x] Six + two new main figures + FigureS1 as vector PDFs (producers make_biorxiv_figures_v2.py + _v3_ext.py; 41 value locks, all OK; FigureS1 = 14-family supplement view; Figure 7/8 + Figure 3 extended render added journal 155)
- [x] Data availability + Code availability statements (payload-not-redistributed route)
- [x] Full source-verified reference list (40 entries) with in-text wiring
- [ ] Author list / affiliations / corresponding email — **USER-SIDE**
- [ ] Supplement file assembly (prereg list §10.4 + reproduction entry points §10.5 as standalone PDF) — ready to export
- [x] final figure upload set: experiments/analysis_biorxiv_figure_pack_v2/ (Fig1/2a/2b/3/4/5/6 + FigureS1, all .png+.pdf, server-rendered and value-locked)

**Evaluated models.**

16. Bogard, N., Linder, J., Rosenberg, A. B. & Seelig, G. A deep neural network for predicting and engineering alternative polyadenylation. *Cell* **178**, 91–106 (2019). doi:10.1016/j.cell.2019.04.046 (APARENT; 2.74M corpus.)
17. Linder, J. & Seelig, G. APARENT2. In: *Sparse and Crepuscular Structures in Biology* / as released at github.com/johli/aparent2 (multimolecule re-export, 2024). (APARENT2 frozen-delta row.)
18. Sample, P. J. et al. Optimus 5'UTR CNN — architecture as described in ref. 1; no official public weights (community retrains). (Optimus rows per port ledger.)
19. Agarwal, V. & Kelley, D. R. The genetic and biochemical determinants of mRNA degradation rates in mammals. *Genome Biol.* **23**, 245 (2022). doi:10.1186/s13059-022-02811-x (Saluki; half-life model.)
20. Chen, J. et al. Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions. *arXiv* 2204.00300 (2022); official repo ml4bio/RNA-FM; multimolecule re-release (AGPL-3.0, RNACentral 23.7M). (RNA-FM rows.)
21. Chu, Y. et al. A 5' UTR language model for decoding untranslated regions of mRNA and function predictions. *Nat. Mach. Intell.* **6**, 449–460 (2024). doi:10.1038/s42256-024-00823-9 (UTR-LM.)
22. Xiong, Y. et al. mRNABERT: advancing mRNA sequence design with a universal language model and codesign. *Nat. Commun.* **16**, 10371 (2025). doi:10.1038/s41467-025-65340-8 (mRNABERT-raw backbone of V5.)
23. Penić, R. J. et al. RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks. *Nat. Commun.* **16**, 5671 (2025). doi:10.1038/s41467-025-60872-5 (RiNALMo micro/mega/giga.)
24. Yin, W. et al. ERNIE-RNA: an RNA language model with structure-enhanced representations. *Nat. Commun.* **16**, 10076 (2025). doi:10.1038/s41467-025-64972-0 (ERNIE-RNA.)
25. Li, S. et al. mRNA-LM: full-length integrated SLM for mRNA analysis. *Nucleic Acids Res.* **53**, gkaf044 (2025). doi:10.1093/nar/gkaf044 (mRNA-LM 5'/3'UTR segments.)
26. Li, G. et al. HydraRNA: a hybrid architecture based full-length RNA language model. *Genome Biol.* **26**, 383 (2025). doi:10.1186/s13059-025-03853-7 (HydraRNA.)
27. Fradkin, P. et al. Orthrus: toward evolutionary and functional RNA foundation models. *Nat. Methods* **23**, 935–945 (2026). doi:10.1038/s41592-026-03064-3 (Orthrus 4/6-track.)
28. LAMAR-UTR5TEPred — mutation-effect pretrained UTR predictor, ported per the frozen port ledger (MIT license; EsmTokenizer, 1026-nt truncation). Full bibliographic citation to be completed at submission from the ledger's registered source. (LAMAR row.) 【submission-stage completion】
29. Zhao, H. et al. GEMORNA (generative mRNA designer with 5'/3'UTR scoring heads). *Science* (2025) — as registered in the port ledger (no declared license; exemption decision recorded). (GEMORNA rows.)
30. UTR-STCNet. Lin, Y. et al. arXiv 2507.16801 (2025) — UTR-STCNet (MPRA-H corpus, BIBM 2025 line). (UTR-STCNet row.)
31. UTR-Insight. Chen, Z. et al. *BMC Genomics* (2025) — integrating deep learning for efficient 5'UTR discovery and design; PMC11796101. (UTR-Insight row.)
32. CodonFM. Hie, B. et al. Codon language models (NVIDIA-Digital-Bio/CodonFM, NV-CodonFM-Encodon-80M-v1; NVIDIA Open Model License). (CodonFM row.)
33. CaLM. Yang, J. et al. *bioRxiv* (2024); OPIG CaLM codon MLM (multimolecule re-export, AGPL). (CaLM row.)
34. LucaOne. Zhang, Y. et al. arXiv 2405.07632 (2024); official FTP checkpoint (Apache-2.0). (LucaOne row.)
35. RiboNN. *Nat. Biotechnol.* (2026) — 3,819 Ribo-seq datasets, full-length TE model; as registered in the port ledger. (RiboNN rows.)

**Methods and infrastructure.**

36. NCBI GEO disclaimer (data use policy), current version 2024-07-16; https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html.
37. ENCODE portal terms of service / data use policy; https://www.encodeproject.org/help/rest-api/ ("freely download, analyze and publish ... no restriction").
38. EMBL-EBI terms of use and BioStudies record E-MTAB-10902 (per-study license field absent; EBI-wide free-use-with-citation terms).
39. Dao, A. et al. Cryptic splicing QC context (U-rich 3'UTR MPRA artifacts), 2025 — as cited in row 7's supplement.
40. MultiMolecule project (Chen, Z. & Zhu, S. Y., Zenodo 12638419, 2024) — re-release channel for several RNA backbones used here.

> Reference-list provenance note: every bibliographic field above was verified against PubMed E-utilities, the ENCODE REST API, BioStudies, and provider pages on 2026-10-06 (the rights-audit session); entries intentionally kept minimal for preprint and to be formatted to the target journal's style at submission. In-text citation wiring is in place (20 anchor sites asserted); the single remaining reference item is [28] LAMAR, whose full bibliographic citation is completed at submission from the port ledger's registered source.

## Figures

The submission carries eight main figures plus one supplement figure. Figures 1–6 and S1 are rendered by the consolidated producer `scripts/route_a_v3/make_biorxiv_figures_v2.py`; Figures 7, 8, and the Figure 3 extended render are added by the extension producer `make_biorxiv_figures_v3_ext.py` (journal 155; read-only, same style, same output dir). Output lives in `experiments/analysis_biorxiv_figure_pack_v2/`, manifest `figure_pack_v2_manifest.json` with 41 value-lock assertions (26 + 15 extension), all passing at render time; unified style: DejaVu Sans 9, 300 dpi, vector PDF, fonttype 42, no top/right spines. Every plotted number is either read live from a frozen JSON (matrix, density, cells, first-order, bottomline, W-ladder, ERK, D16-C, 3-seed-ensemble files) or declared verbatim from the same frozen archives the paper text cites; nothing is recomputed.

**Figure 1. DeltaBench frozen-Δ matrix: five newly ported families × 13 task cells** (task-macro Spearman, VALIDATION only; TEST reads = 0 at matrix build, the split subsequently read exactly once per the frozen unblinding prereg and the matrix not recomputed). Every cell is the archived value from `benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json` (65/65 asserted); the separator marks the append-only new-row block (M1/M6/S1, frozen before any model was scored on them). Headline reading: strong in-domain (UTR-STCNet MRL 0.814) versus weak cross-library (same family on M1 0.067), new rows collectively near zero. Heatmap with in-cell values and soft diverging colormap.

**Figure 2. Density–learnability observation (panel a) and its held-out test (panel b).** (a) 25 external frozen-Δ rows against candidates-per-source (log x): r = 0.9388 (p = 3.9e-12); polyA (density 126.7) is the only high-density task — the observation. (b) The pre-registered held-out cell-level check: 65 matrix cells, predicted ρ (single-factor fit) vs observed ρ with a ±0.10 band; 33/65 = 50.77% in-band → **FAIL** against the 70% gate, the honest downgrade per the frozen clause. Source JSONs: `analysis_delta_vs_density_20260915/delta_vs_density_data.json` (25 rows asserted) and `analysis_delta_vs_density_v2/delta_vs_density_v2_cells.json` (65 cells, in-band count 33 asserted); the four direct labels sit at the true cell coordinates.

**Figure 3. First-order decomposition vs external band vs ours vs ceiling, by task** (lollipop form; polyA / MPRAU / TE). ρ₁ values are read from `analysis_first_order_decomposition_v1/results_first_order.json` and asserted at render time (polyA 0.45554942, MPRAU 0.08253518, TE 0.04577197); the external unsupervised band, our best in-house row, and the ICC label ceiling complete each track; the right margin reports ρ₁ as a share of ours (polyA 55%, MPRAU 80%, TE 79%). polyA shows the largest above-first-order structural headroom; MPRAU and TE sit close to their first-order readings.

**Figure 4. Ceiling-normalized learnability map (dumbbell: best external → ours → ICC ceiling, per task).** Our rows are asserted against `analysis_task8_bottomline_20260909/bottomline_adjudication_v1.json` (polyA 0.8219, MRL 0.3217 exact); external anchors are the paper-§8 frozen values (frozen-Optimus 0.3132, APARENT 0.7343, Saluki 0.1205, RNA-FM 0.2958 / 0.1043, LLR-family 0.0). Completion: polyA 91.3% (0.8219 / 0.90) is the only high-completion task; MRL 38.8% (0.3217 / 0.83); REF/ALT 30.4% (0.0639 / 0.21); MPRAU 19.8% (0.1351 / 0.683); PLUMAGE-RNA 13.7% (0.0500 / 0.364); TE 9.9% (0.0579 / 0.586); HALF_LIFE and PLUMAGE-TE carry **no normalization** (ICC ≈ 0 or negative).

**Figure 5. Intervention triangle: synthetic / parameter / real-data arms (MRL Δ).** Ten values, each locked to a named frozen artifact at render time. Synthetic panel — V5 gap 0.7943 and D16-C gap 0.8235 (`xeditcritic_d16c/probe_mrl_v1_gpu5/gap_backtest_d16c.json`; FALSIFIED). Parameter panel — ERK v2 val 0.2015 (`xeditcritic_erk_v2/erk_train_seed2026091901/erk_adjudication_v1.json`) vs the E7 closed-form first-order bound 0.2069 (§6.1 archive; BOUND-ANCHORED). Real-data panel — the W ladder W0 0.1987 → LoRA 0.2470 (`280k_prefinetune_20260903/frozen_delta_results.json`) → full-FT 2 ep 0.2555 (`280k_fullft_ablation_20260903/frozen_delta_results.json`) → 3-seed ensemble 0.3158 ≈ frozen-Optimus 0.3132 (`analysis_fullft_v2_adjudication_20260903/ensemble_3seed_vs_optimus.json`; statistical tie, Δ CI crossing zero; EFFECTIVE). The panel encodes the full §7 ladder exactly as the text states it, with the separate LoRA and full-FT rungs.

**Figure 6. Where the learnable signal lives: first-order table vs structural component (four panels, ceiling-normalized).** For polyA / MPRAU / TE, ρ₁ is the frozen first-order reading (asserted at render time); for MRL, ρ₁ is the §6.1 E7 archive bound 0.2069. Each panel decomposes the ICC ceiling into the first-order share, the structural share (ours − ρ₁), and the label-noise room; per-task first-order share of best: polyA 55%, MPRAU 80%, TE 79%, MRL 64%.

**Figure 7. Error-correlation triples (§4).** Grouped bars of the ρ_abs / ρ_delta / ρ_ε triple per row, values locked to the frozen archives: the Optimus MRL decoupling case (0.8733 / 0.3132 / 0.7196) and the two LAMAR blind rows (HL5 0.0207 / 0.0441 / 0.6863; HL3 −0.0911 / 0.0184 / 0.4429) — high ρ_ε coexisting with no delta signal, the visual form of the §4.1–4.2 argument that absolute accuracy constrains neither the favorable nor the catastrophic error-correlation condition.

**Figure 8. Failure taxonomy matrix (§7).** Task × class dot matrix for the 20 taxonomy rows (14 failure + 6 control; C1–C5 column definitions as in §7.1): rows are the affected tasks/domains, dot size encodes the row count per cell (5 / 5 / 2 / 5 / 6). The matrix form makes the anti-one-sided structure visible at a glance — control rows (C5) sit in the same plane as the failure classes, and every one of the 9/9 tasks is covered by at least one class.

**Figure 3 (extended render; §6.2).** The Figure 3 lollipop chart re-rendered with a fourth MRL track carrying the reference pair (E7 closed-form first-order bound 0.2069 vs ERK v2 0.2015, ICC 0.83), merging the former Table 8 into the figure; ρ₁ values are read live from `analysis_first_order_decomposition_v1/results_first_order.json` and asserted at render time (polyA 0.45554942, MPRAU 0.08253518, TE 0.04577197), with the per-task first-order share of ours (polyA 55%, MPRAU 80%, TE 79%) in the right margin.

**Figure S1 (supplement). DeltaBench across all 14 model families.** Top block: the frozen 5 × 13 matrix (65/65 re-checked cells, same values as Figure 1). Bottom block: the 9 existing reference families + V5 (ours), cells drawn where archived `ALREADY_DONE`/LOSO/bottomline readings exist (33 cells); grey hatched cells = no reading in this caliber (structured NA — paradigm mismatch or not measured), not zero. This is the full-family view behind the abstract's "51 of 65 cells read |ρ| < 0.1".

- Submission-stage figure work: unified styling, vector-only panels, alt-text, and the consolidated producer manifest (41 value locks, all OK).