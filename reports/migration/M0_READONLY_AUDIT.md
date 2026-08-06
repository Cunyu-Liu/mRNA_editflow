# M0 Read-Only Audit — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1 Migration

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Audit UTC:** 2026-08-06T02:47:00Z
- **Host:** A100 (`36.137.135.49`, user `cunyuliu`)
- **Target repo:** `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **Migration base commit:** `1ec984290025534c2f8a18f7d8b2fd2f4a5cc18b`

---

## 1. FACTS_FROM_REPO

### 1.1 Repository state
| Item | Value |
|---|---|
| Main-tree branch | `phase2-reliable-local-delta-20260727` |
| Main-tree HEAD | `aca6c8c31f5843105f49c30e78e78342278bdd7d` |
| Main-tree active contract | `utr_editflow_contract_v2` (NOT v3.1) |
| Main-tree dirty files | 5 (4× `data/d1_canonical_records.jsonl.pre_*` snapshots + `archive/legacy_predictor_first_v1/benchmark/benchmark_v21/external_data/LinearDesign`) |
| Worktrees (v3.1 autoritativer) | `v3_1_authority_rebind_20260804T074227Z` @ `1ec9842` |
| Other worktrees | d0_r_repair, ef0_true_utr, fm0_cpu_audit, fm0_current_b0, gp0_generative_prior, mk0_math_kernel, v3_1_data_bench_closure ×2, v3_1_data_bench_g7_retry, + /mnt worktrees |

**Critical finding:** The main tree is on the older `utr_editflow_contract_v2`. The contract to be superseded (`utr_editflow_goal_v3.1_benchmark_first`) lives in the **v3.1 authority worktree** (`v3_1_authority_rebind_20260804T074227Z`), which is a descendant of main-tree HEAD `aca6c8c`. The migration base must therefore be the v3.1 authority state, which this M0 worktree has checked out at `1ec9842`.

### 1.2 v3.1 authority worktree contents
- Contract: `docs/contracts/utr_editflow_goal_v3_1.md` (SHA256 `35dd4bf2…`)
- Supersession: `docs/contracts/supersession_v3_1.md` (records v2→HISTORICAL_SUPERSEDED, v3.0→HISTORICAL_SUPERSEDED, active=v3.1)
- Execution registries (`docs/execution/`): `claim_matrix_v3_1.yaml`, `decision_log_v3_1.yaml`, `diagnostic_registry_v3_1.yaml`, `split_registry_v3_1.yaml`, `task_registry_v3_1.yaml`, `task_split_contract_matrix_v3_1.yaml`, `GOAL_V3_DATA_BENCH_REVALIDATION_20260804.yaml`, `USER_DECISION_GSE246381.yaml`
- Data registries (`data/v3_1/registry/`): `dataset_assets.jsonl` (33), `dataset_decisions.jsonl` (33), `raw_asset_manifest.jsonl` (167), `search_ledger.jsonl` (45), `license_matrix.csv`, `priority_snapshot_v3_1.yaml`
- Schemas (`schemas/v3_1/`): 22 schema files + `SCHEMA_MANIFEST.json` + `SCHEMA_SHA256SUMS`
- Data dirs (`data/v3_1/`): `b0/`, `d1/`, `g7/`, `registry/`, `sealed_commitments/`
- D1 data: `D1_CANONICAL_MANIFEST.json`, `D1_SHA256SUMS`, `D1_STATUS.json`

### 1.3 Contracts (hashes)
| Contract | Path | SHA256 | Bytes |
|---|---|---|---|
| Old (v3.1) | `提示词/mrna 最新合同-v1.md` | `ecc6c635f112575db2f14309c869a378fc31df8fb76c01dda0b54b832b4f8946` | 344,589 |
| New (XEditFlow v1.0) | `提示词/mrna 最新合同-v2.md` | `9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e` | 69,226 |
| Migration prompt | `提示词/mrna 合同迁移.md` | `55c89e5dc065a116063bdcdcd594322db4f2f33f6cc5d25c27cd4fc402fab916` | 39,689 |
| v3.1 repo contract | `docs/contracts/utr_editflow_goal_v3_1.md` | `35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759` | — |

### 1.4 GPU / processes / disk
- **GPU:** 8× A100 40GB. Heavy multi-user load. GPU5 98% util, GPU7 46%. **No GPU exclusively free.**
- **GPU owners:** shenxin, zhaobowei, yihaozhao, af3_bosun, jingxuange (gmx), root — all active.
- **Must-not-disturb (this project):** PID `1535664` (cunyuliu) — v3.1 strict D1 validation, running 1d5h, STAT=D, using **sealed GSE246381 input**, authority contract SHA `ecc6c635…`. Located in v3.1 authority worktree, writing to `/mnt/cunyuliu/mrna_editflow_v3_1_goal_runs/GOAL-V3-DATA-BENCH-01_20260804/d1_strict_full_20260805T034500Z`.
- **Disk:** `/home` 5.2T avail (22% used); `/mnt` 12T avail (35% used).

### 1.5 Active-code reference scan (M0 read-only)
Scan of main tree (excluding `archive/`, `external_tools/envs`, `.pyc`):
- `Track U` / `track_u`: only hits inside `external_tools/envs/` (third-party package internals) — **no active project code references**.
- `learned STOP`: only hits in archived `artifacts/runs/MK0_*` logs/audits — no active source.
- `GSE246381`: hits in archived run artifacts + `logs/d0_03_download.log` — historical.
- `alignment`: hits in archived MK0 run artifacts — historical.
- `calibrated_marginal`: `docs/p2_03_leakage_free_headline.md`, `docs/full_length_mpra_design_v1.md`.

---

## 2. FACTS_FROM_CONTRACTS
- Old contract v3.1 sets the paper problem as **benchmark/resource-first** with Edit-Flows-derived UTR reference + alignment robustness; contract ID `utr_editflow_goal_v3.1_benchmark_first`.
- New contract (XEditFlow v1.0) recenters the thesis on **mRNA-EditBench + SparseEditFormer source-relative effect critic + source-anchored legal Edit Flow + exact density-ratio guidance + measured-neighborhood optimization**.
- Migration prompt mandates: supersede old scientific line only; **retain** old data-integrity, license, exposure, sealed-final, and audit governance as base.

---

## 3. INFERENCES
1. **Base for migration = v3.1 authority worktree commit `1ec9842`**, not main-tree v2. M0 worktree correctly checked out this commit.
2. The v3.1 sealed-data D1 validation is **still running** and must not be disturbed; M1 changes must not collide with the v3.1 authority worktree.
3. Old data-governance layers (registry, schemas, license, exposure, sealed commitments, supersession) are intact in the v3.1 authority worktree and are high-reuse (per §3.1 of migration prompt estimate 80–95%).
4. `core/ef0`, `core/mk0` do **not** exist in the main tree; they live in other worktrees (ef0_true_utr, mk0_math_kernel). Exact-sampler naming must be re-audited against numeric semantics in M1.
5. No new contract artifacts exist yet (`docs/contracts/mrna_xeditflow_goal_v1_1.md`, `configs/mrna_xeditflow_contract_v1_1.yaml` absent).

---

## 4. UNKNOWN_OR_BLOCKED
| # | Blocker | Impact |
|---|---|---|
| B0 | Main tree active contract is `utr_editflow_contract_v2`, not v3.1. | Must confirm migration base is the v3.1 authority worktree (done in M0 worktree). |
| B1 | v3.1 D1 sealed-data validation (PID 1535664) still running. | M1+ must not touch v3.1 authority worktree or its run root until it completes. |
| B2 | No GPU exclusively free; heavy multi-user load. | Any future training must claim a GPU only when required and leave owners untouched. |
| B3 | `core/ef0`, `core/mk0` absent from main tree; only in secondary worktrees. | Reuse audit must resolve actual paths for model/sampler code (delayed to M1 evidence-based inventory). |
| B4 | New contract v1.1 artifacts not present. | To be created in M1. |

---

## 5. TEST_RESULTS
- No tests run in M0 (read-only precheck). 
- M1 will add failing tests for: active top-level contract count==1, active legacy references==0, crosswalk coverage==100%, conflict decisions hash-bound, old artifacts unmodified.

---

## 6. DATA_COUNTS_AND_DENOMINATORS
- Main-tree `data/d1_canonical_records.jsonl`: **1,151,125** lines.
- v3.1 D1 validation expected: **3,831,570** ordinary / **1,300** restricted records; **32,990** matrix rows (pending, running).
- v3.1 registry: 33 dataset_assets, 33 dataset_decisions, 167 raw_asset_manifest, 45 search_ledger.

---

## 7. REUSE_DECISIONS
See `artifacts/migration/MIGRATION_INVENTORY.jsonl` for per-object classification. Summary:
- **REUSE_AS_IS:** raw/provenance/hash, license matrix, exposure ledger, sealed commitments, supersession record, immutable registry JSONL.
- **REUSE_WITH_ADAPTER:** schemas/v3_1 (add evidence_grade), D1/B0 builders, split primitives.
- **REBUILD:** task/split registry v4, claim matrix, scientific contract, config.
- **UNKNOWN_BLOCKED:** `core/ef0`, `core/mk0` (paths unresolved in main tree).

---

## 8. GATE_STATUS
- **M0 acceptance:** ✅ PASS (all 6 acceptance criteria met; see M0_INPUT_MANIFEST.json).
- **M1 gate:** not yet run.

---

## 9. CLAIMS_UNLOCKED
- None (M0 is read-only; no scientific claims unlocked).

---

## 10. CLAIMS_STILL_PROHIBITED
- All old v3.1 PASS items are `STALE_REVALIDATION_REQUIRED` under new contract (except immutable raw/hash/provenance facts).
- GSE246381 stays `OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL` project-unexposed; no labels read in M0.

---

## 11. NEXT_PHASE_INPUTS (M1)
- Migration worktree at `1ec9842` (v3.1 base) ready.
- Old/new contract hashes bound.
- v3.1 authority registries available for crosswalk.
- STOP / unlabeled-pretraining / GSE246381 / indel / budget freeze decisions (migration prompt §三 A–E) to be encoded.

---

## 12. COMMIT_SHA
- M0 worktree base commit: `1ec984290025534c2f8a18f7d8b2fd2f4a5cc18b` (no new commit yet in M0).

---

## 13. MANIFEST_AND_HASHES
- `artifacts/migration/M0_INPUT_MANIFEST.json`
- `artifacts/migration/M0_SHA256SUMS`
- `artifacts/migration/MIGRATION_INVENTORY.jsonl`