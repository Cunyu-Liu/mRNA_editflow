# P3-00 Change Governance (frozen)

- **Date**: 2026-07-22 (created by P3-00A)
- **phase_status**: PASS
- **scientific_validation_status**: PENDING — this document governs procedures only; it makes no experimental claim.
- **Applies to**: all P3 frozen artifacts

## 1. Frozen artifacts

| Artifact | Path |
|---|---|
| Master research contract | `configs/p3_frozen_research_contract.yaml` |
| Primary task contract | `configs/p3_primary_task.yaml` |
| Frozen scientific question | `docs/p3_00_frozen_scientific_question.md` |
| Claim ladder | `docs/p3_00_claim_ladder.md` |
| Hypothesis preregistration | `docs/p3_00_hypothesis_preregistration.md` |
| Task scoring + problem lock | `docs/p3_00_scientific_problem_lock.md` |
| Landscape matrix | `docs/p3_00_scientific_landscape.tsv` |
| GO/NO-GO matrix | `docs/p3_00_go_no_go_matrix.json` |
| Contract tests | `tests/test_p3_00_task_contract.py` |

## 2. Rules

1. **No silent edits.** A frozen artifact changes only through an explicit amendment (§3). Direct edits, renames, deletions, and "clarifications" without an amendment record are forbidden (global hard constraint #3).
2. **No threshold shopping.** Gate thresholds frozen in the preregistration may never be edited after the corresponding data has been seen; excluding failing families/seeds/sequences to reach GO is forbidden (global constraints #16, #17, #22).
3. **Unlock, don't assume.** `locked_extension` / `conditional_extension` items (3′UTR, Task C joint, RL, synergy) activate only when their preregistered gate passes. An amendment cannot unlock them without the gate evidence.
4. **Fail closed.** If a frozen artifact is missing, corrupted, or fails hash verification, dependent paper runs must refuse to execute (global constraints #7, #10).
5. **Hash integrity.** Every amendment recomputes and re-embeds SHA-256 digests in `docs/p3_00_go_no_go_matrix.json` and re-runs `pytest tests/test_p3_00_task_contract.py` to green before commit.

## 3. Amendment procedure

An amendment **must** contain all of the following:

1. **Amendment ID** — e.g., `p3_00a_frozen_publication_contract_alignment`.
2. **Rationale** — why the change is needed; what it does *not* change.
3. **Scope** — exact files and sections touched; explicit statement that model/GRPO/data/training code is untouched (if true).
4. **Gate evidence** — if the amendment changes task status, hypothesis thresholds, or unlocks an extension: the gate result that authorizes it. Contract-alignment amendments (like P3-00A) instead state that no scientific gate is affected.
5. **Version bump** — `task_version` / contract `version` incremented; old version referenced under `supersedes`.
6. **Test delta** — new/updated contract tests covering the amended semantics.
7. **Hash refresh** — updated digests embedded in the gate matrix; full test suite green.

## 4. Amendment log

| Amendment | Date | Version change | Rationale | Gate affected |
|---|---|---|---|---|
| p3_00 (initial freeze) | 2026-07-22 | — → p3_contract_v1 | Lock scientific question, task contract, hypotheses, pause conditions | Created P3-00 gate |
| p3_00a_frozen_publication_contract_alignment | 2026-07-22 | p3_contract_v1 → p3_contract_v2 | Align contract with frozen publication line: method/task hierarchy (Base Model scope vs Task A–D), Generic FlexFlow-mRNA registered as closest baseline, phase-vs-scientific status separation, scoring robustness analysis, versioned 3-tier motif policy replacing `no_forbidden_motif` | None — no scientific gate threshold changed; H1–H6 remain PENDING |

## 5. Interpretation disputes

If a frozen text is ambiguous, the resolution order is: (1) machine-readable contract (`configs/p3_frozen_research_contract.yaml`, `configs/p3_primary_task.yaml`); (2) gate matrix JSON; (3) preregistration; (4) narrative docs. Any interpretation that would weaken a gate, a split contract, or a claim-ladder qualifier is automatically invalid.
