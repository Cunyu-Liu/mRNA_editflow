# P3-00 Frozen Scientific Question

- **Date**: 2026-07-22 (amended by P3-00A)
- **phase_status**: PASS
- **scientific_validation_status**: PENDING
- **Contract**: `configs/p3_frozen_research_contract.yaml` (p3_contract_v2)

This document freezes what the paper line claims to study. P3-00A aligns the previously completed P3-00 scientific-question gate with the final frozen publication contract. No new scientific survey was performed; no model, GRPO, data, or training code was modified.

---

## 1. The method (Base Model scope)

The primary method is a **heterogeneous grammar-constrained Edit Flow for full-length protein-coding mRNA**. The Base Model studies the complete heterogeneous state space of mRNA:

- **full-length mRNA modeling** — one generative/editing process over UTR nucleotides and CDS codon grammar jointly;
- **protein-conditioned CDS generation** — CDS as a frame-locked codon lattice conditioned on target protein;
- **region infilling** — 5′UTR/3′UTR as length-variable regulatory canvases;
- **source-conditioned editing** — minimal, budgeted, legality-guaranteed edits of an existing mRNA.

The method contribution is the *grammar-constrained heterogeneous flow itself*: different regions obey different grammars (free nucleotide substitution vs frame-locked synonymous codon substitution), and the flow respects those grammars by construction, not by post-hoc filtering or reward penalty.

## 2. The frozen scientific question (application level)

> Given a source mRNA that already expresses the target protein, in fixed cargo and cell context, can ≤ k nucleotide substitutions — starting with 5′UTR substitutions — significantly improve real protein output while keeping protein sequence, transcript length, and key manufacturing constraints unchanged?

`k ∈ {1, 3, 5, 10}`. Primary endpoint: protein output over time (AUC or pre-registered time-point abundance). Prediction target: local counterfactual `Δ = output(candidate) − output(source)`.

## 3. Task hierarchy (frozen)

| Task | Name | Status | Role |
|---|---|---|---|
| A | 5′UTR minimal substitution | **active_primary** | first application task; all P3-01→P3-03 falsification runs here first |
| B | CDS synonymous minimal substitution | **frozen_fallback** | activates only if Task A fails H1/H2 while CDS passes |
| C | 5′UTR+CDS joint substitution | **locked_extension** | unlocks only when H1, H2, H3 pass in **both** regions |
| D | full-transcript editing | **rejected_for_primary_paper** | hard-floor ineligible (no local 3′UTR labels; highest reward-hacking risk) |

**Task A is the first application task, not the whole method scope.** The Base Model's full-length grammar flow remains the method contribution even if later tasks shrink to a single region.

## 4. Closest baseline

**Generic FlexFlow-mRNA** is registered as the closest baseline: the same model class (full-length mRNA flow) *without* the heterogeneous grammar-constrained edit contract — no region grammar, no frame lock, no minimal-edit budget semantics. Every matched-budget comparison must include it before any external baseline (LinearDesign, UTR-LM, etc.), because it isolates the contribution of the grammar-constrained contract from the contribution of the underlying flow architecture.

## 5. What is NOT a paper blocker

- **RL is not a paper blocker.** `rl_status: conditional_extension`. If P3-07 shows ranker + limited search is near-optimal (Route C), the paper proceeds as benchmark + local-delta prediction + constrained search + prospective validation. RL enters only via Route A (quality gain) or Route B (amortization gain).
- **3′UTR is not a paper blocker.** `three_utr_status: locked_extension`; unlock requires a 3′UTR local-delta oracle meeting H2 thresholds plus transferable joint gains (H6).
- **Cross-region synergy is not a paper blocker.** `synergy_status: conditional_extension`; it is an optional unlock hypothesis (H5), never a default premise.

## 6. Why the alternative routes were not chosen

| Route not taken | Reason (evidence, not narrative) |
|---|---|
| Full-transcript first (Task D as primary) | Hard-floor ineligible: no source-matched local 3′UTR edit labels in project assets; weakest current reward sensitivity; highest reward-hacking score risk (1/5); PERSIST-seq offers only 233 full-length sequences |
| CDS synonymous as primary | Local-edit label availability 2/5 (no source-matched synonymous delta datasets); smaller per-edit effects; kept as frozen fallback because external baselines (LinearDesign) and tractability are strong |
| Joint 5′UTR+CDS as primary from day one | No joint local-edit labels; synergy evidence is BORDERLINE (P2-01, d=+0.371, internal oracle only); joint action space doubles reward-hacking surface before H1/H2 are established per region |
| RL-first scale-up | P2-05 effect sizes were tiny (mean return 0.0274, CI [+0.0015, +0.0076]); with edit budget ≤3 exact enumeration is cheap; RL value is itself a hypothesis (H4) gated at P3-07 |
| De novo full-length generation as differentiation | 2026 field is crowded (GEMORNA, mRNA-GPT, ProMORNA, mRNAutilus); wholesale replacement carries new immunogenicity/structure/manufacturing/IP risk that minimal edit avoids |
| Multi-objective reward from project start | TE, stability, and protein output are not equivalent (PERSIST-seq); arbitrary proxy weighted sums invite reward hacking; single primary endpoint (protein output) with secondary tracking is falsifiable |

## 7. Relationship to P3-00 gate

P3-00 PASS means: the contract is frozen, tasks are evidence-scored, preferred/fallback tasks are chosen, hypotheses are falsifiable, and RL pause conditions are explicit. It does **not** mean H1–H6 are experimentally supported — every hypothesis remains `scientific_validation_status: PENDING` until its gate phase (P3-01…P3-10) produces evidence.
