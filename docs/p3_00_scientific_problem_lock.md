# P3-00 Scientific Problem Lock

- **Date**: 2026-07-22 (amended by P3-00A Frozen Publication Contract Alignment Patch)
- **phase_status**: PASS
- **scientific_validation_status**: PENDING — H1–H6 are preregistered, not validated. P3-00 PASS must never be cited as experimental support.
- **Contract**: `configs/p3_frozen_research_contract.yaml` (p3_contract_v2), `configs/p3_primary_task.yaml` (p3_task_v2)
- **Inputs**: `提示词/mrna的 rl 的后续优化的分阶段提示词.md` (P3-00 spec), `docs/p3_00_scientific_landscape.tsv`, `docs/p3_00_hypothesis_preregistration.md`
- **P3-00A amendment scope**: method/task hierarchy fixed; totals arithmetic corrected (A=41, B=29, C=29, D=21); scoring robustness analysis added (§2.5); motif policy referenced as versioned `motif_policy_v1`; closest baseline registered. No scientific gate threshold changed; no model/GRPO/data/training code touched.

---

## 1. Locked scientific question

> Given a source mRNA that already expresses the target protein, in a fixed cargo and cell context, can ≤ k nucleotide substitutions — starting with 5′UTR substitutions, with protein-preserving synonymous CDS substitutions as fallback/extension — significantly improve real protein output while keeping protein sequence, transcript length, and key manufacturing constraints unchanged?

- `edit_budget k ∈ {1, 3, 5, 10}` (frozen in `configs/p3_primary_task.yaml`).
- Primary endpoint: **protein output over time** (protein-output AUC, or normalized abundance at pre-registered time points). TE-only reward is forbidden: PERSIST-seq shows ribosome load, stability, and total protein output are not equivalent.
- **Method/task separation (P3-00A)**: the paper's primary method is the heterogeneous grammar-constrained Edit Flow over the full mRNA state space (full-length modeling, protein-conditioned CDS generation, region infilling, source-conditioned editing). Task A below is the **first application task, not the whole method scope**.
- 3′UTR: `locked_extension` — not in the primary task (weakest current reward sensitivity; no source-matched local labels; cell-type-specific RBP/miRNA/splicing risk; larger reward-hacking surface).
- Prediction target: local counterfactual `Δprotein_output = output(candidate) − output(source)`, not absolute expression.

### Why this question (evidence, not narrative)
1. **Real application problem**: therapeutic mRNA programs optimize existing constructs; wholesale replacement carries new immunogenicity/structure/manufacturing/IP risk and validation cost. Few-edit optimization is controllable, cheap to validate, interpretable, and easy to control experimentally.
2. **Headroom exists**: UTR-LM's prospective panel beat optimized therapeutic UTRs; PERSIST-seq shows 5′UTR drives the largest ribosome-load changes; LinearDesign validates synonymous-recoding gains.
3. **Differentiation**: full-length de novo + RL is crowded in 2026 (GEMORNA, mRNA-GPT, ProMORNA, mRNAutilus). The defensible niche is source-conditioned, minimal-edit, hard-constrained, local-counterfactual, uncertainty-aware optimization with prospective validation.
4. **Missing asset**: not another absolute TE predictor, but a model + benchmark for *few-edit local effects around a given source*.

---

## 2. Candidate task scoring (Task 3)

Scale: 1–5, 5 = most favourable. `reward-hacking risk` is scored as *favourability* (5 = low risk). Scores are evidence-based per `docs/p3_00_scientific_landscape.tsv` and project memory; narrative appeal is explicitly excluded (spec: "不得仅凭故事吸引力选择 Task D").

| Dimension | A: 5′UTR-only | B: CDS-only synonymous | C: 5′UTR+CDS joint | D: full-transcript |
|---|---:|---:|---:|---:|
| experimental data availability | 5 — Sample 2019 ~280k, Cao 2021 ~50k, UTR-LM 211 | 3 — abundant observational codon data; few perturbation sets | 3 — union of A+B, no joint local-edit sets | 2 — PERSIST-seq only 233 full-length |
| local-edit label availability | 4 — dense single-nt MPRA neighborhoods (reporter context) | 2 — source-matched synonymous delta data sparse | 2 — no source-matched joint-edit labels | 1 — no source-matched 3′UTR local labels in assets |
| predictability | 5 — Optimus 5-Prime / UTR-LM validated | 3 — CAI/tAI/MFE proxies good; protein-output delta weaker | 3 — inherits weakest region | 2 — 3′UTR effects context-specific, weakest oracle sensitivity |
| biological effect size | 4 — largest ribosome-load changes (PERSIST-seq) | 3 — smaller per edit; cumulative CDS gains shown (LinearDesign) | 4 — additive at minimum; synergy unproven (P2-01 BORDERLINE) | 3 — 3′UTR real but context-dependent |
| action-space tractability | 5 — short region, small neighborhood, budget ≤10 cheap | 3 — long region but synonymous constraint + codon lattice keeps it tractable | 3 — product of two spaces, still enumerable at k≤3 | 2 — largest space; enumeration intractable beyond k=1 |
| external baseline availability | 5 — UTailoR, UTR-LM, Optimus 5-Prime | 5 — LinearDesign, EnsembleDesign, codonGPT | 4 — region baselines composable; joint-edit baselines few | 3 — de-novo full-length baselines exist but are not edit-based |
| wet-lab feasibility | 5 — short oligos, established reporter assays | 4 — standard gene synthesis, higher per-variant cost | 4 — same synthesis, combined constructs | 3 — more constructs, more confounds |
| novelty | 3 — crowded (UTR-LM, UTailoR, RNAGenScape) | 3 — LinearDesign/GEMORNA CDS strong | 4 — minimal-edit joint with hard constraints is open | 4 — but crowded by de-novo full-length generators |
| reward-hacking risk (5 = low) | 5 — well-validated oracles, small space | 3 — CAI/GC heuristics easily hacked; needs guards | 2 — larger space, more hacking surface | 1 — largest surface; weakest oracles |
| **Total (45 max)** | **41** | **29** | **29** | **21** |

Totals corrected by P3-00A (previously mis-summed as 36/30/29/21; per-dimension scores unchanged). Ranking: **A (41) > B = C (29) > D (21)**.

### 2.5 Scoring robustness (P3-00A Task 4)

**Hard-floor eligibility.** Pre-registered floors: a task is ineligible if any of {experimental data availability, local-edit label availability, predictability, reward-hacking favourability} < 2.

| Task | min over floored dims | Eligible |
|---|---:|---|
| A | 4 | yes |
| B | 2 | yes |
| C | 2 | yes |
| D | 1 (local-edit labels = 1; reward-hacking = 1) | **NO — structurally ineligible regardless of total** |

**Leave-one-criterion-out (LOCO).** Dropping each of the 9 dimensions in turn and re-totalling: A ranks first in all 9 scenarios (A exceeds B/C by 12 points overall; the largest single-dimension A-advantage is 3 points, so no single deletion can flip the lead). Secondary instability exists and is recorded honestly: B and C are tied at 29 overall, and their relative order swaps depending on the dropped dimension (dropping `external baseline availability` → C 25 > B 24; dropping `novelty` → B 26 > C 25; dropping `wet-lab feasibility` or `action-space tractability` → B = C tie). This instability affects only the fallback ordering, never the preferred task.

**Alternative-weight sensitivity.** Re-totalling under pre-registered alternative weightings:

| Weighting scheme | A | B | C | D | A first? |
|---|---:|---:|---:|---:|---|
| equal (baseline) | 41 | 29 | 29 | 21 | yes |
| data-first (×2 on dims 1–2) | 50 | 34 | 34 | 24 | yes |
| risk-averse (×2 on dims 3, 9) | 51 | 35 | 34 | 24 | yes |
| novelty-seeking (×2 on dim 8) | 44 | 32 | 33 | 25 | yes |
| feasibility-first (×2 on dims 5, 7) | 51 | 36 | 36 | 26 | yes |

**Conclusion**: Task A ranks first under every tested weighting and every LOCO scenario, and passes hard-floor eligibility. The selection is stable; no ranking instability affects the preferred task. The B/C tie is recorded, not hidden, and is resolved by governance (fallback activation order), not by re-scoring.

### Decision

- **active_primary: Task A (5′UTR-only)** — highest score (41/45), first under all robustness checks. P3-01→P3-03 falsification runs 5′UTR-first. The operational contract (`allowed_actions` in `configs/p3_primary_task.yaml`) contains **only** `five_utr_substitution`; no CDS action is part of the active task.
- **frozen_fallback: Task B (CDS-only synonymous)** — activates only if Task A fails H1/H2 while CDS passes; activation requires an amendment per `docs/p3_00_change_governance.md` (contract re-versioning; claim level stays ≤ C3 until prospective CDS evidence exists).
- **locked_extension: Task C (5′UTR+CDS joint)** — unlocks only when H1, H2, and H3 have each passed in **both** the 5′UTR and the CDS scope (6 conditions frozen in `application_tasks.task_c.unlock_requires`). Joint before that is forbidden.
- **rejected_for_primary_paper: Task D (full-transcript)** — lowest score (21/45) and hard-floor ineligible. Remains `locked_extension` at the method level; unlock conditions per H6 only. Full-transcript synergy is an optional hypothesis (H5/H6), never a default premise.

---

## 3. Falsifiability summary

All six hypotheses (H1–H6) are preregistered with explicit thresholds in `docs/p3_00_hypothesis_preregistration.md`:

| Hypothesis | Falsifier (what would kill it) | Gate phase |
|---|---|---|
| H1 local-effect existence | best-edit distribution ≈ matched random-edit distribution | P3-01/03 |
| H2 local-delta predictability | sign accuracy ≈ 0.50, no top-k enrichment, oracle reversal | P3-02 |
| H3 optimization headroom | strong search cannot beat source on independent oracle | P3-02 |
| H4 RL quality/amortization value | ranker+limited search near-optimal; no quality/cost edge (Route C) | P3-07 |
| H5 cross-region synergy | joint ≤ best single-region on independent oracles | P3-10 |
| H6 full-transcript extension value | 3′UTR delta oracle fails H2 thresholds | P3-10 |

Thresholds were frozen on 2026-07-22 before P3-01+ data is examined; post-hoc threshold edits or exclusion of failing families are forbidden (global constraints #17, #22).

---

## 4. Minimum publishable unit (Task 5)

The first paper is viable if and only if all five components exist:

1. **Source-conditioned minimal-edit benchmark** (P3-01): local-edit benchmark aligned to the frozen contract, with Level A–D data separation, group-aware splits (train source-disjoint / val source-disjoint / test cargo-family-disjoint / OOD shifted), frozen manifests with hashes.
2. **Reliable local-delta predictor** (P3-02): meets H2 thresholds on family-disjoint test; cross-fitted oracle ensemble with a frozen independent oracle.
3. **Strong search or learned policy** (P3-02/P3-07/P3-08): constrained search at minimum; RL only if H4 passes — Route A (quality) or Route B (amortization). Route C (no-RL) remains a complete paper. **RL is not a paper blocker.**
4. **Independent predictor transfer** (P3-02/P3-09): designs keep improvement direction on a predictor/assay model not used in training (Claim level C3). All matched-budget comparisons include the registered closest baseline **Generic FlexFlow-mRNA** before external baselines.
5. **At least one prospective assay** (P3-03, expanded in P3-11): measured protein output, mRNA abundance, cell viability; pre-registered time points (Claim level C5 for the assayed panel only).

Explicitly **not required** for the first paper: full-transcript editing, 3′UTR, multi-objective preference, cross-region synergy, GRPO, top-tier-scale wet-lab. These are upgrade items gated by H5/H6 and H4, not minimum conditions.

### Claim-Ladder discipline
- C0–C1 after P3-01/P3-02 (benchmark + prediction).
- C2 always written as `predicted / internal proxy`.
- C3 requires the frozen independent oracle.
- C4 requires pre-registered matched-budget comparisons (ranker gain / search gain / RL quality gain / RL amortization gain stated separately).
- C5 only from prospective measured endpoints; C6 only with independent-oracle + prospective synergy support.
- Full ladder: `docs/p3_00_claim_ladder.md`.

---

## 5. RL pause conditions (explicit)

RL work (any policy training beyond correctness smoke tests) pauses when **any** of the following holds:

1. **H1 NO-GO** — local edits show no measurable signal vs matched random edits (P3-03 prospective, or P3-01/02 in-silico pending P3-03).
2. **H2 NO-GO** — sign accuracy ≈ random, no beneficial-edit enrichment, or independent-oracle direction reversal (P3-02).
3. **H3 NO-GO** — strong search finds no stable headroom on the independent oracle (P3-02).
4. **H4 NO-GO / Route C** — ranker + limited search near-optimal with no RL quality or amortization advantage (P3-07): stop GRPO expansion specifically; project continues as benchmark + delta prediction + constrained search + prospective validation.
5. **Reward hacking exposed** — P3-03/P3-09 adversarial high-training-reward candidates fail prospectively: pause RL, fix reward/oracle, re-gate.

Recovery from any pause requires new evidence (Level-C data, rebuilt oracles, or fixed reward) and a fresh gate run — never a threshold edit.

---

## 6. Defaults explicitly rejected

The following are **not** premises of P3 (each is a falsifiable hypothesis or an unlock):

- full-transcript joint optimization beats region-specific optimization by default (H5/H6);
- cross-region synergy exists with large effect (H5; P2-01 evidence is BORDERLINE, d = +0.371);
- the current oracle deserves full RL optimization (H2/H3 gates first);
- stronger backbones automatically improve Edit-Flow (P3-05 gate);
- GRPO beats strong local search (H4);
- all three regions belong in one MDP now (Task D rejected);
- multi-objective reward should be optimized from the start (single primary endpoint: protein output).

---

## 7. Constraint versioning (P3-00A Task 5)

The unversioned `no_forbidden_motif` hard constraint is replaced by **`motif_policy_v1`** (`configs/p3_primary_task.yaml#motif_policy`), with three tiers:

- **hard_forbidden** — universally invalid sequences (premature stop, cryptic splice sites, upstream in-frame start codon, IVT-blocking homopolymer ≥6 nt): excluded **by the action space**, never by reward penalty (global constraint #15). Referenced in `hard_constraints` as `motif_policy_v1_hard_forbidden`.
- **guarded_risk** — context-dependent regulatory risks (cell-specific miRNA seed creation, RBP site disruption, m6A motif change, uAUG context change): surfaced to oracle/reviewer and tracked; not auto-illegal.
- **soft_objective** — sequence-quality preferences (local GC extremes, 4–5 nt homopolymers, codon-pair deoptimization): reward shaping only; never legality criteria.

Not all motif risk is a universal hard constraint; the three tiers are versioned together and amended only via `docs/p3_00_change_governance.md`.
