# v2 Contract Conflict Matrix

**Contract:** `utr_editflow_contract_v2`
**Created:** 2026-08-01 (Phase C0, task C0-02)
**Decision log:** `docs/decision_log.md` → `DEC-UTR-EF-V2-20260801-CONTRACT-INTEGRATION`

This matrix records every conflict between the now-archived `public_intervention_contract_v1` (Layer 1) and the active `utr_editflow_contract_v2` (Layer 2), and the disposition taken during Phase C0. It also records conflicts found in repository state (README, task registry, code references) and how they were resolved.

---

## 1. Contract-level conflicts (v1 → v2)

| # | Dimension | v1 (`public_intervention_contract_v1`) | v2 (`utr_editflow_contract_v2`) | Disposition |
|---|---|---|---|---|
| 1 | Primary method | SparseEditForm predictor; Flow optional / fallback | Edit Flow is primary, NOT optional, NOT a fallback (§1.1) | v1 archived under `archive/legacy_predictor_first_v1/contracts_v1/`; v2 is the only active contract |
| 2 | Scope | 5′UTR + 3′UTR + CDS synonymous + full-length | 5′UTR + 3′UTR only; CDS/full-length out-of-scope (§4.2) | All CDS/full-length benchmark and config content archived |
| 3 | GSE246381 status | Sealed external test (untouched) | Historically exposed (E4); labels forbidden for new training & hyperparameter selection | All "sealed/untouched" wording removed from active tree; `forbidden_wording` enforced in v2 contract §2 |
| 4 | Scientific question | Local-delta prediction + transfer (Q1–Q5) | Source-conditioned generative edit-trajectory distribution (H1–H8) | v1 scientific question + claim matrix archived; v2 scientific question + claim matrix active |
| 5 | Hypotheses | Q1–Q5 (prediction-oriented) | H1–H8 (generative + architecture + constraint + control + search + transfer + foundation + region) | v1 hypotheses archived; v2 hypotheses are the only active set |
| 6 | Benchmark | EditBench (delta-prediction CSV) | Generative UTR Benchmark (closed_measured_pool / heldout_generative / open_legal_generation) | `benchmark/`, `benchmark_v21/`, `data/nmi_benchmark_v2/` archived |
| 7 | Evidence grades | A1/A2/B1/B2/C/D | E0–E6 (max E3–E5; GSE246381 fixed at E4; E6 not in scope) | v1 grade scheme archived; v2 grade scheme is the only active one |
| 8 | Phase structure | No explicit phase gates; P3-00A execution order | C0→D0→D1→B0→FM0→MK0→EF0→GP0→FC0→ME0→MB0→TR0→ER0→PP0 + MK0/MB0-Freeze mandatory gates | v1 task registry archived; v2 task registry is the only active one |
| 9 | Evaluation metrics | delta_spearman, sign_accuracy (prediction) | Pareto frontier (function-control / legality / diversity / edit-cost / generation-efficiency) + 100% hard constraints + generative likelihood/recovery/calibration | v1 metrics retired; v2 metrics are the only active ones |
| 10 | RL role | GRPO/DAgger/policy as central method | RL is NOT the central methodological story of v2 | `rl/`, `train_grpo.py`, `train_dagger_ranker.py` etc. archived |
| 11 | Checkpoints | proposal_ranker_t5_*, region_adapter_t5_*, phase_c_seed*, p1_*, phase2_stage_a_* | None yet (FM0/MK0/EF0/GP0 not started) | All legacy checkpoints moved to `archive/legacy_predictor_first_v1/ckpts/` |
| 12 | README | "full-length protein-coding mRNA generation", CDS/5′UTR/CDS/3′UTR infilling, P3-00A, RL contracts | UTR-only, Edit Flow primary, no CDS/full-length/RL, GPU-only | README replaced (1605 → 207 lines) |
| 13 | Contract ID in active code | `public_intervention_contract_v1` in data_registry, scripts, tests | `utr_editflow_contract_v2` everywhere in active tree | All active code refs updated; v1 test file archived |
| 14 | Authority hierarchy | Ambiguous (P3 archived → v1 active → v2 unlanded) | Single active contract: Goal → contract_v2.yaml → scientific_question_v2 → claim_matrix_v2 → task_registry_v2 → per-experiment artifacts | Three-layer ambiguity resolved; only v2 is active |
| 15 | ENCODE status | manifest: downloaded=false, deferred; verification: PARTIAL | manifest: RAW_READS_COMPLETE, downloaded=true, provider_md5 verified; verification: COMPLETE | All 4 evidence files updated (manifest, processed_manifest, download_verification, task_registry) |
| 16 | Goal §1.5 / §D0-04 | "62 ENCODE raw files downloading, task continues" | "62 ENCODE raw files download complete (in reconstructed/, ~357 GB, provider_md5 + size verified)" | Local Goal document updated |

---

## 2. Repository-state conflicts found and resolved

| # | Location | Conflict | Resolution |
|---|---|---|---|
| R1 | `README.md` | Title claimed "full-length protein-coding mRNA generation"; body contained CDS/5′UTR/CDS/3′UTR infilling, P3-00A, RL contracts | Replaced with v2-aligned README (UTR-only, Edit Flow primary) |
| R2 | `docs/execution/task_registry.yaml` (v1) | Only R0/D0 tasks, no C0 tasks; controlled by v1 contract | Archived to `archive/legacy_predictor_first_v1/contracts_v1/`; replaced by `docs/execution/task_registry_v2.yaml` |
| R3 | `configs/public_intervention_contract.yaml` | Active config for predictor-first route | Archived to `archive/legacy_predictor_first_v1/contracts_v1/` |
| R4 | `docs/public_intervention_scientific_question.md`, `docs/public_intervention_claim_matrix.md` | Active v1 scientific question + claim matrix | Archived to `archive/legacy_predictor_first_v1/contracts_v1/` |
| R5 | `configs/archive/p3_legacy/SUPERSEDED.md`, `docs/archive/p3_legacy/SUPERSEDED.md` | SUPERSEDED chain broken: pointed to v1 as successor, but v1 is now also superseded | Appended second-level supersession note pointing to v2; chain is now P3 → v1 → v2(ACTIVE) |
| R6 | `data_registry/intervention_candidates.yaml`, `data_registry/unavailable.yaml`, `data_registry/search_artifacts/intervention_candidates.raw.json` | `contract_id: public_intervention_contract_v1` | Updated to `utr_editflow_contract_v2` |
| R7 | `scripts/data/download_common.py`, `scripts/data/download_ena_reconstruction.py` | USER_AGENT string referenced v1 | Updated to `mrna-editflow-d0/2.0 (utr_editflow_contract_v2)` |
| R8 | `scripts/data/systematic_search.py` | Wrote v1 contract_id into registry; comment referenced v1 doc | Updated to write v2 contract_id; comment references v2 doc |
| R9 | `tests/test_public_intervention_contract.py` | Entire test file validated v1 contract (now archived) | Archived to `archive/legacy_predictor_first_v1/tests/` |
| R10 | `tests/test_systematic_search.py` | Assertion expected v1 contract_id | Updated to expect v2 contract_id |
| R11 | `tests/test_missing_dataset_acquisition.py` | Assertion expected v1 contract_id in unavailable.yaml | Updated to expect v2 contract_id |
| R12 | `tests/test_validate_registry.py` | Test fixture used v1 contract_id | Updated to v2 contract_id |
| R13 | `docs/data/missing_dataset_acquisition.md`, `docs/data/systematic_search_protocol.md` | Contract header referenced v1 | Updated to v2 |
| R14 | `benchmark/`, `benchmark_v21/`, `data/nmi_benchmark_v2/` | Legacy predictor-first / delta-prediction benchmarks in active tree | Archived to `archive/legacy_predictor_first_v1/benchmark/` |
| R15 | `rl/` | Legacy GRPO/DAgger/policy/cto/synergy code in active tree | Archived to `archive/legacy_predictor_first_v1/rl/` |
| R16 | `train_grpo.py`, `train_dagger_ranker.py`, `train_proposal_ranker.py`, `train_adapter.py`, `train_backbone.py`, `sample.py` | Legacy training scripts in repo root | Archived to `archive/legacy_predictor_first_v1/training_scripts/` |
| R17 | `configs/nmi_split_v2.yaml`, `configs/paired_delta/`, `configs/stage_a_*.json` | Legacy configs in active configs/ | Archived to `archive/legacy_predictor_first_v1/configs/` |
| R18 | `audit_sota_readiness.py`, `sota_gap_report.py`, `harvest_sota_artifacts.sh`, `audit_multiobjective_scaleup_claims.py` | Legacy SOTA/multiobjective audit scripts in repo root | Archived to `archive/legacy_predictor_first_v1/audits/` |
| R19 | `docs/next_steps_sota_roadmap.md`, `docs/codongpt_rl_reproduction_blocker.md`, `docs/cross_region_synergy_*` | Legacy SOTA/RL/cross-region docs in active docs/ | Archived to `archive/legacy_predictor_first_v1/docs/` |
| R20 | `ckpts/` (gitignored, ~4.4 GB) | Legacy checkpoints in active tree | Moved (plain mv) to `archive/legacy_predictor_first_v1/ckpts/` |
| R21 | `data/p0/ENCSR854RUF/manifest.json` | 62 files `downloaded=false`, `defer_reason` set | All 62 files `downloaded=true`, `defer_reason` removed, `download_location` added, top-level `status: RAW_READS_COMPLETE` |
| R22 | `data/p0/ENCSR854RUF/processed/processed_manifest.json` | `status: PROCESSED_DATA_COMPLETE_RAW_READS_INCOMPLETE` | `status: PROCESSED_DATA_COMPLETE_RAW_READS_COMPLETE` |
| R23 | `docs/data/download_verification.md` | `verdict: PARTIAL`, `files deferred: 62` | `verdict: COMPLETE`, `files deferred: 0`, ENCSR854RUF row `complete: 62` |
| R24 | `docs/execution/task_registry.yaml` D0-03, D0-04 | `status: IN_PROGRESS`, report "raw reads incomplete" | `status: DONE`, report "all 62 ENCODE raw fastq.gz downloaded and verified" |

---

## 3. Conflicts intentionally NOT resolved (historical records)

The following v1 references are intentionally preserved as historical records and do NOT need fixing:

| Location | Reason for preservation |
|---|---|
| `docs/plans/2026-07-28-phase-d0-data-discovery.md` | Dated historical plan document — records what was planned at the time |
| `docs/plans/2026-07-28-public-intervention-contract.md` | Dated historical plan document — records the v1 contract creation |
| `tests/test_audit_legacy_references.py` (fixture lines) | Test fixtures that intentionally simulate a legacy repo to validate the audit script's behavior |
| `archive/legacy_predictor_first_v1/contracts_v1/*` | The v1 contract files themselves — archived as historical reference |
| `configs/archive/p3_legacy/SUPERSEDED.md`, `docs/archive/p3_legacy/SUPERSEDED.md` | Legacy marker files — updated with second-level supersession note, original content preserved |
| `configs/utr_editflow_contract_v2.yaml` §supersedes | v2 contract intentionally names v1 as superseded |
| `README.md` §2 | README intentionally names v1 as superseded |
| `docs/utr_editflow_scientific_question_v2.md` §header | v2 doc intentionally names v1 as superseded |
| `docs/decision_log.md` | Decision log records the v1→v2 transition (intentional) |

---

## 4. Final state

- **Active contract count:** 1 (`utr_editflow_contract_v2`)
- **Active contract conflicts:** 0
- **Active predictor-only fallback clauses:** 0
- **Active Flow-optional clauses:** 0
- **Active CDS/full-length Phase-1 tasks:** 0
- **Active GSE246381 sealed/untouched wording:** 0
- **Active v1 contract_id references in code:** 0 (excluding intentional historical records in §3)
- **SUPERSEDED chain:** P3/NMI legacy → public_intervention_contract_v1 → **utr_editflow_contract_v2 (ACTIVE)**

This matrix is **FROZEN** as of 2026-08-01. Future contract amendments must be recorded as new entries in `docs/decision_log.md` and reflected in an updated conflict-matrix addendum, not by rewriting this file.
