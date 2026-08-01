# SUPERSEDED — Legacy Predictor-First / SparseEditFormer / RL Route

**Status:** SUPERSEDED_LEGACY
**Superseded by:** `utr_editflow_contract_v2` (active contract)
**Archived on:** 2026-08-01
**Decision:** `docs/decision_log.md` → `DEC-UTR-EF-V2-20260801-LEGACY-ARCHIVAL`
**Approved by user:** 2026-08-01

## Why archived

This subtree contains artifacts from the **predictor-first / SparseEditForm / RL route**, which conflicts with the active contract `utr_editflow_contract_v2` on multiple core boundaries:

| Boundary | Legacy route | Active v2 contract |
|---|---|---|
| Primary method | SparseEditForm predictor; Flow optional | Edit Flow is primary, not optional |
| Scope | 5′UTR + 3′UTR + CDS synonymous + full-length | 5′UTR + 3′UTR only (CDS/full-length out of scope, §4.2) |
| GSE246381 | Sealed external test | Historically exposed (E4); labels forbidden for new training/hyperparameter selection |
| Scientific question | Local-delta prediction + transfer | Source-conditioned generative edit-trajectory distribution |
| Hypotheses | Q1–Q5 (prediction-oriented) | H1–H8 (generative + architecture + constraint + control + search + transfer + foundation + region) |
| Benchmark | EditBench (delta-prediction CSV) | Generative UTR Benchmark (closed_measured_pool / heldout_generative / open_legal_generation) |
| RL | GRPO / DAgger / policy / synergy | RL is NOT the central methodological story of v2 |

## Contents

| Subtree | Contents |
|---|---|
| `contracts_v1/` | `public_intervention_contract.yaml`, `public_intervention_claim_matrix.md`, `public_intervention_scientific_question.md`, `task_registry.yaml` |
| `benchmark/` | `benchmark/`, `benchmark_v21/`, `nmi_benchmark_v2/` |
| `rl/` | grpo / dagger / policy / cto / synergy legacy RL code |
| `training_scripts/` | `train_grpo.py`, `train_dagger_ranker.py`, `train_proposal_ranker.py`, `train_adapter.py`, `train_backbone.py`, `sample.py` |
| `configs/` | `nmi_split_v2.yaml`, `paired_delta/`, `stage_a_*.json` |
| `ckpts/` | Legacy checkpoints (proposal_ranker_t5_*, region_adapter_t5_*, phase_c_seed*, p1_*, phase2_stage_a_*) |
| `docs/` | SOTA roadmap, cross-region synergy findings, RL blocker docs |
| `audits/` | SOTA gap / readiness / multiobjective audit scripts |

## Rules

- **Historical reference only.** Must not be cited as active evidence.
- **Must not be reactivated** without a new amendment to `utr_editflow_contract_v2`.
- **Must not be merged** into the v2 main line.
- Git history is preserved (used `git mv` for tracked paths; `mv` for gitignored `ckpts/`).
- Negative results and failed seeds archived here remain visible per the project's evidence-preservation rule.
