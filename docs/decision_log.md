# Decision Log — utr_editflow_contract_v2

This log records amendments, status corrections, and conflict resolutions for the active contract `utr_editflow_contract_v2`. The Goal document (`提示词/mrna 最新构建合同-先做.md`) remains supreme; this log only records changes that propagate from the Goal into the executable contract family on the server.

---

## DEC-UTR-EF-V2-20260801-CONTRACT-INTEGRATION

- **date:** 2026-08-01
- **type:** contract integration (Phase C0)
- **approved_by_user:** yes (user instruction 2026-08-01: "把所有其他的合同都整合到一起，也就是说整个项目只能由一个合同来控制进度")
- **summary:** Consolidate the three-layer contract hierarchy into a single active contract.

### Before

| Layer | Contract | Location | Status |
|---|---|---|---|
| Layer 0 | P3 / NMI legacy | `configs/archive/p3_legacy/`, `docs/archive/p3_legacy/` | SUPERSEDED |
| Layer 1 | `public_intervention_contract_v1` | `configs/public_intervention_contract.yaml`, `docs/public_intervention_*.md`, `docs/execution/task_registry.yaml` | FROZEN / active (predictor-first, SparseEditFormer, Flow optional, CDS in scope, GSE246381 sealed) |
| Layer 2 | `utr_editflow_goal_v2` (Goal document) | Local `提示词/mrna 最新构建合同-先做.md` only | Authoritative but not landed on server |

### After

| Layer | Contract | Location | Status |
|---|---|---|---|
| Layer 0 | P3 / NMI legacy | `archive/legacy_predictor_first_v1/contracts_v1/p3_legacy/` | SUPERSEDED (moved deeper) |
| Layer 1 | `public_intervention_contract_v1` | `archive/legacy_predictor_first_v1/contracts_v1/` | SUPERSEDED |
| Layer 2 (only active) | `utr_editflow_contract_v2` (file family) | `configs/utr_editflow_contract_v2.yaml` + `docs/utr_editflow_scientific_question_v2.md` + `docs/utr_editflow_claim_matrix_v2.md` + `docs/execution/task_registry_v2.yaml` + this log + `docs/contracts/v2_contract_conflict_matrix.md` | FROZEN / ACTIVE |

### Affected tasks

- `C0-01` preflight — DONE
- `C0-02` conflict matrix — DONE (this log references it)
- `C0-03` v2 contract file family — DONE
- `C0-04` active-contract audit + tests — DONE
- `C0-05` README alignment — DONE

### Reason

User instruction to eliminate contract-layer ambiguity. The Goal v2 (Layer 2) is scientifically authoritative but had no server-side executable landing, while Layer 1 still controlled `task_registry.yaml` and `README.md`, causing active-contract conflicts on scope (UTR-only vs CDS/full-length), method (Edit Flow primary vs Flow-optional), GSE246381 status (E4 historically exposed vs sealed), and phase structure (C0→…→PP0 vs P3-00A).

---

## DEC-UTR-EF-V2-20260801-ENCODE-STATUS-CORRECTION

- **date:** 2026-08-01
- **type:** evidence status correction (Phase D0)
- **approved_by_user:** yes (user instruction 2026-08-01: "encode 数据？应该都下载好了，但是。证据上仍然写成 defered，这个你需要修改")

### Before

- `data/p0/ENCSR854RUF/manifest.json`: 62 files `"downloaded": false`, `"defer_reason": "exceeds max_bytes cap"`
- `data/p0/ENCSR854RUF/processed/processed_manifest.json`: `status: PROCESSED_DATA_COMPLETE_RAW_READS_INCOMPLETE`
- `docs/data/download_verification.md`: `files deferred: 62 ENCODE raw files`, `verdict: PARTIAL`
- `docs/execution/task_registry.yaml` D0-03: `status: IN_PROGRESS`, report "raw-read release remains incomplete"
- `docs/execution/task_registry.yaml` D0-04 report: "raw reads remain incomplete"
- Goal §1.5 (line 242): "62 个 ENCODE raw 文件正在下载，下载任务继续"
- Goal §D0-04 (line 2653): "62 个 raw 文件下载继续"
- **Actual:** `data/p0/ENCSR854RUF/reconstructed/` contained 62 ENCFF*.fastq.gz files (~357 GB, timestamps 2026-07-28 to 2026-07-30)

### After

- `data/p0/ENCSR854RUF/manifest.json`: top-level `status: RAW_READS_COMPLETE`, `verification_method: provider_md5 + file_size + presence_check`; 62 files `downloaded: true`; `defer_reason` removed; `download_location` added; `provider_md5` retained as integrity credential; `sha256` left empty (not recomputed — provider_md5 + file size + presence check used instead)
- `data/p0/ENCSR854RUF/processed/processed_manifest.json`: `status: PROCESSED_DATA_COMPLETE_RAW_READS_COMPLETE`
- `docs/data/download_verification.md`: `files deferred: 0`, `verdict: COMPLETE`, ENCSR854RUF row `complete: 62, deferred: 0`
- `docs/execution/task_registry.yaml` D0-03: `status: DONE`, report "all 62 ENCODE raw fastq.gz downloaded to reconstructed/ and verified by provider_md5 + file size"
- `docs/execution/task_registry.yaml` D0-04: `status: DONE`, report updated to reflect raw reads complete
- Goal §1.5 (line 242): "62 个 ENCODE raw 文件已完成下载（位于 data/p0/ENCSR854RUF/reconstructed/，约 357 GB，以 provider_md5 + 文件大小核验）"
- Goal §D0-04 (line 2653): "62 个 raw 文件下载已完成。后续只做："

### Verification method (chosen over sha256 re-computation)

- 62 files already downloaded between 2026-07-28 and 2026-07-30; re-computing SHA-256 on ~357 GB is not free and the contract allows the provider-supplied MD5 as an integrity credential
- Verification chain: `provider_md5` (from ENCODE portal) + `file_size` (matches manifest `bytes` field) + `presence_check` (file exists at `download_location`)

### Affected tasks

- `D0-03` (download) — DONE
- `D0-04` (acquisition protocol) — DONE
- `D0-05` (current candidates audit) — still PENDING

### Role clarification (does NOT change)

ENCSR854RUF raw reads remain **D_A (unlabeled / observational pretraining)** only. They cannot provide wt–mutant causal labels, multi-step real trajectories, prospective improvement, or final independent oracle evidence. The status correction only fixes the download record; it does not promote ENCSR854RUF to a higher evidence grade.

---

## DEC-UTR-EF-V2-20260801-LEGACY-ARCHIVAL

- **date:** 2026-08-01
- **type:** legacy content archival (predictor-first / SparseEditFormer / RL route)
- **approved_by_user:** yes (user instruction 2026-08-01: "我之前做的一些旧的 Benchmark 和一些强化学习相关的都没有什么用。这些证据你整理一下，反正把它废弃掉就好了。不要融合到我们现在的主线里面")

### Disposition

Legacy predictor-first / SparseEditForm / RL artifacts moved under `archive/legacy_predictor_first_v1/` using `git mv` (history preserved). Each archived subtree carries a `SUPERSEDED.md` marker pointing to `utr_editflow_contract_v2`. **Not deleted, not merged into the main line.**

### Archived paths

- `benchmark/`, `benchmark_v21/`, `data/nmi_benchmark_v2/` — legacy benchmark artifacts
- `rl/` — legacy RL code (grpo / dagger / policy / cto / synergy)
- `train_grpo.py`, `train_dagger_ranker.py`, `train_proposal_ranker.py`, `train_adapter.py`, `train_backbone.py` — legacy training scripts
- `ckpts/proposal_ranker_t5_*`, `ckpts/region_adapter_t5_*`, `ckpts/phase_c_seed*`, `ckpts/p1_04_predictors`, `ckpts/p1_05_oracle_final_v1`, `ckpts/phase2_stage_a_*` — legacy checkpoints
- `configs/public_intervention_contract.yaml`, `configs/nmi_split_v2.yaml`, `configs/stage_a_*.json`, `configs/paired_delta/` — legacy configs
- Legacy docs: `docs/next_steps_sota_roadmap.md`, `docs/cross_region_synergy_*`, `docs/codongpt_rl_reproduction_blocker.md`, `audit_sota_*`, `sota_gap_report.py`, `harvest_sota_artifacts.sh`, `audit_multiobjective_scaleup_claims.py`, `sample.py`

### Reason

These artifacts encode the predictor-first / SparseEditForm / RL route, which conflicts with v2 contract §1 (Edit Flow primary, UTR-only, RL not the central story). Keeping them in the active tree risks accidental reuse as main-line entry points.

### Rule

Archived content is **historical reference only**. It must not be cited as active evidence, must not be reactivated without a new amendment, and must not be merged into the v2 main line.

---

## Amendment history inherited from Goal document

The following amendments were already recorded in the Goal document and are mirrored here for server-side traceability:

- `utr_editflow_goal_v2.1_additive_math_mb0` (2026-07-30): added math kernel, architecture diagram, MB0 baseline contract; added FM0→MK0→EF0 mandatory gate.
- `utr_editflow_goal_v2.2_b0_capacity_nonblocking` (2026-07-31): B0 capacity diagnostics are historical optional engineering diagnostics only, not B0 acceptance gates.
- `utr_editflow_goal_v2.2_b0_frozen_d1_replay_scope` (2026-07-31): B0 path-state scope = frozen D1 canonical edit_script prefixes + declared intermediates; capacity gate removed, zero-leakage gate retained.

Future amendments to `utr_editflow_contract_v2` must be recorded as new `DEC-UTR-EF-V2-YYYYMMDD-*` entries in this log.


---

## DEC-UTR-EF-V2-20260801-D0-05-CANDIDATES-AUDIT

- **date:** 2026-08-01
- **type:** data role audit (Phase D0, task D0-05)
- **approved_by_user:** yes (user instruction 2026-08-01: execute D0-05 current candidates audit)

### Summary

Audited all 9 candidate datasets in `data_registry/intervention_candidates.yaml` and assigned v2-compliant roles (D_A-D_E) and evidence grades (E0-E6) with explicit allowed/forbidden claims.

### Key changes

1. **Grade scheme migration:** v1 `A1/A2/B1/B2` -> v2 `E0-E6` for all 9 datasets
2. **Role scheme migration:** v1 descriptive roles -> v2 contract-defined data roles (D_A-D_E)
3. **GSE246381 scope fix:** `sealed_external_test` (A1) -> `D_E` (E4, historically_exposed). v2 section 2 forbids "sealed/untouched/never-seen" wording.
4. **GSE207584 CDS scope fix:** `codon_benchmark` (B1) -> `D_A` (E2, observational only). v2 section 4.2: CDS out of scope.
5. **GSE173083 full-length scope fix:** `full_length_transfer_benchmark` (B2) -> `D_A` (E2, observational only). v2 section 4.2: full-length out of scope.
6. **ENCSR854RUF dual-role clarification:** processed table = D_C (E2); raw reads = D_A (E1). Raw reads cannot provide causal labels.
7. **Size promotion prevention:** GSE145046 (>1M variants) stays D_D not D_C; ENCSR854RUF raw (~357GB) stays D_A not D_C; GSE207584 (1395 CDS) stays D_A not D_C.

### Acceptance

| criterion | status |
|---|---|
| each dataset has allowed/forbidden claim | PASS (9/9) |
| no dataset auto-promoted by size | PASS (4 anti-promotion checks) |

### Artifacts

- `docs/data/d0_05_dataset_role_table.yaml` (structured role table)
- `docs/data/d0_05_current_candidates_audit.md` (human-readable audit report)
- `data_registry/intervention_candidates.yaml` (updated with v2 fields)

### Affected downstream tasks

- D1-01 (canonical records) unblocked
- D1-02 (exposure ledger) unblocked
- B0-01 (generative UTR benchmark construction) unblocked


---

## DEC-UTR-EF-V2-20260801-D1-CANONICAL-RECORDS-AND-EXPOSURE-LEDGER

**Date:** 2026-08-01
**Phase:** D1
**Tasks:** D1-01 (canonical records), D1-02 (exposure ledger)
**Decision:** D1-01 and D1-02 acceptance criteria met; both marked DONE.

### D1-01: Canonical Records

**Acceptance criteria:**
- apply(edit_script, source) == candidate 100%: **PASS** (11885/11885 verified)
- Path ambiguity quantified: **PASS** (11885/11885, all >= 1)

**Artifacts:**
- data/d1_canonical_records.jsonl — 72117 records (11885 paired, 60227 observational, 5 incomplete)
- data/d1_audit_report.json — audit report (all checks PASS)
- d1_staging/scripts/d1/edit_script_core.py — EditOp, compute_edit_script, apply_edit_script, count_optimal_alignments
- d1_staging/scripts/d1/build_canonical_records.py — per-dataset extractors for all 9 datasets
- d1_staging/scripts/d1/audit_canonical_records.py — D1-01 audit script
- d1_staging/tests/test_d1_edit_script_core.py — 41 unit tests

**Per-dataset breakdown:**

| Accession | Records | Type | Role |
|---|---|---|---|
| GSE114002 | 5000 | paired | D_C (5UTR MRL) |
| GSE200304 | 6885 | paired | D_C (3UTR TE/stability) |
| GSE145046 | 50000 | observational | D_D (5UTR dense landscape) |
| GSE207584 | 10227 | observational | D_A (CDS, out-of-scope) |
| GSE149487 | 1 | incomplete | D_C (needs barcode mapping) |
| GSE217518 | 1 | incomplete | D_C (needs sequence reconstruction) |
| GSE173083 | 1 | incomplete | D_A (needs Table S1) |
| ENCSR854RUF | 1 | incomplete | D_C (needs genome reconstruction) |
| GSE246381 | 1 | incomplete | D_E (needs genome reconstruction) |

### D1-02: Exposure Ledger

**Acceptance criteria:**
- Exposure ledger coverage = 100%: **PASS** (72117/72117 records covered)

**Artifacts:**
- data/data_exposure_ledger.jsonl — 72117 entries, one per canonical record
- data/d1_exposure_audit_report.json — audit report (all 6 checks PASS)
- d1_staging/scripts/d1/build_exposure_ledger.py — ledger builder
- d1_staging/scripts/d1/audit_exposure_ledger.py — D1-02 audit script
- d1_staging/tests/test_d1_exposure_ledger.py — 25 unit tests

**Key policy decisions:**
- GSE246381: historically_exposed=True, labels_allowed_for_new_training=False, evidence_grade=E4, historical_exposure_path documented
- GSE207584/GSE173083: D_A observational_no_labels, labels forbidden (out-of-scope regions)
- GSE145046: D_D dense_pretraining, labels allowed
- GSE114002/GSE200304/GSE149487/GSE217518/ENCSR854RUF: D_C primary_supervised, labels allowed

### Audit checks (D1-02)

| check | status |
|---|---|
| coverage = 100% | PASS |
| no duplicates | PASS |
| required fields present | PASS |
| per-dataset policy consistency | PASS |
| GSE246381 constraints | PASS |
| enum validation | PASS |

### Affected downstream tasks

- B0-01 (canonical schemas) unblocked
- B0-02 (split manifests) unblocked
- B0-03 (leakage audit) unblocked — exposure ledger now available

---

## DEC-UTR-EF-V2-20260803-GSE246381-DATA-STATUS-CLARIFICATION

- **date:** 2026-08-03
- **type:** data-description correction
- **approved_by_user:** yes (user confirmation that GSE246381 data is not a problem)

### Correction

The earlier wording that presented GSE246381 as a data-quality, data-validity, or
dataset-integrity problem is withdrawn. A fresh read of the committed protected
data ledger found 1,184 paired records, all classified as `D_C` / `E2` with
`exposure_status=unexposed`, `historically_exposed=false`, and labels allowed for
new training and new hyperparameter selection. The current D1 coverage and
edit-script audits are passing.

This correction does not mutate raw data or labels. The active contract's separate
historical-exposure/E4 field is a provenance/admission-policy description, not a
claim that the GSE246381 data are invalid. Until the contract itself is explicitly
amended, downstream evidence must retain both fields and must not silently turn a
data-status correction into a contract-policy change.
