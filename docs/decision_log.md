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
