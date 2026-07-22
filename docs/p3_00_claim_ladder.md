# P3-00 Claim Ladder (frozen)

- **Date**: 2026-07-22 (amended by P3-00A)
- **phase_status**: PASS
- **scientific_validation_status**: PENDING
- **Contract**: `configs/p3_frozen_research_contract.yaml`

The claim ladder maps evidence levels to allowed statements. No statement may exceed its evidence level. Global hard constraint #23 applies: every predicted functional improvement must carry a `predicted` / `internal proxy` / `independent predictor` / `wet-lab measured` qualifier.

## Status vocabulary (binding)

- `phase_status: PASS` — a phase's engineering/contract deliverables are complete and verified.
- `scientific_validation_status: PENDING` — the underlying hypotheses (H1–H6) have **not** been experimentally or independently validated.

A phase PASS must never be written as scientific validation. In particular, **P3-00 PASS does not imply H1–H6 hold**; it only means the contract, scoring, hypotheses, and pause conditions are frozen.

## The ladder

| Level | Name | Allowed claim | Forbidden while at this level |
|---|---|---|---|
| C0 | Problem definition holds | Established a source-conditioned minimal-edit benchmark: legal actions, budgets, metrics, splits | Any functional or predictive claim |
| C1 | Local edits are predictable | Model predicts edit sign/ranking on the independent (family-disjoint) test set | Claiming real mRNA improvement; claiming optimization value |
| C2 | Internal computational optimization holds | Improved **predicted protein-output proxy on the internal training oracle** — must be written as `predicted / internal proxy` | Omitting the internal-proxy qualifier; claiming transfer |
| C3 | Independent-oracle transfer holds | Designs keep improvement direction on a predictor/assay model not used in training (`independent predictor` qualifier) | Claiming wet-lab efficacy; claiming RL necessity |
| C4 | Equal-budget optimization value holds | At matched edit/query budgets, model/policy beats pre-registered baselines; gains reported separately as `ranker gain` / `search gain` / `RL quality gain` / `RL amortization gain` | Mixing gain types into one number; post-hoc budget selection |
| C5 | Prospective experimental improvement holds | Real protein output, mRNA abundance, half-life, or TE improved (`wet-lab measured`), for the assayed panel only | Generalizing beyond assayed cargos/contexts |
| C6 | Cross-region mechanism holds | Transferable 5′UTR–CDS (or full-transcript) synergistic editing mechanism exists | Claiming mechanism from internal-oracle synergy alone |

## Gate mapping

| Claim | Requires | Gate phase |
|---|---|---|
| C0 | benchmark frozen, manifests hashed | P3-01 |
| C1 | H2 thresholds on family-disjoint test | P3-02 |
| C2 | H3 headroom on training oracle | P3-02 |
| C3 | H3 confirmed on frozen independent oracle | P3-02 / P3-09 |
| C4 | H4 Route A or B at pre-registered budgets | P3-07 / P3-08 |
| C5 | prospective assay panel meets pre-registered criteria | P3-03 / P3-11 |
| C6 | H5 replicated on ≥2 independent oracles + prospective arm | P3-10 |

## Current standing (2026-07-22)

- All claims: **none above C0**. P3-00 established the contract only.
- `max_claim_level_without_prospective: C2_internal_proxy` (per `configs/p3_primary_task.yaml`).
- Any document claiming improvement before P3-02/P3-03 evidence exists must use `predicted / internal proxy` wording.
