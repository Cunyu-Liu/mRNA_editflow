# M2 Data Schema & Canonical Compatibility Migration — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Phase:** M2 (数据 schema 与 canonical 兼容迁移)
- **UTC:** 2026-08-06
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **M1 commit:** `633f7e8` · **M2 base:** `633f7e8`

---

## 1. FACTS_FROM_REPO
- 主树只读未改；M2 全部在隔离 worktree 内完成。
- `schemas/v3_1/` 21 个 schema + `SCHEMA_MANIFEST.json` + `SCHEMA_SHA256SUMS` 原样保留，未修改、未删除（git status 无 v3_1 变更）。
- `data/v3_1/registry/`（dataset_assets.jsonl 33 行、dataset_decisions.jsonl、license_matrix.csv、priority_snapshot_v3_1.yaml、raw_asset_manifest.jsonl、search_ledger.jsonl）原样保留。
- 新增 `schemas/xedit_v1_1/` 命名空间（22 个 schema 文件：21 个 rebind + 1 个正交轴参考 schema）。
- 新增 `docs/execution/xeditflow_asset_role_assignment.yaml`（33 个 P0/P1 资产逐一定级）。

## 2. FACTS_FROM_CONTRACTS
- 新权威合同 `mrna_xeditflow_goal_v1_1`（SHA `fc9c1c88…`）为唯一 ACTIVE 权威。
- 旧合同 `utr_editflow_goal_v3.1_benchmark_first`（SHA `ecc6c635…`）标记 HISTORICAL_SUPERSEDED。
- 迁移提示词 §七 Phase M2 要求：新建 `schemas/xedit_v1_1/`、保留旧 schema、注入正交轴、每个 P0/P1 资产定级为四类之一。

## 3. INFERENCES
- A1/A2/B1/B2/C/D 作为**正交轴**叠加在保留的 E/F 之上，不替代 E/F（`scientific_track` 与 `intervention_evidence_grade` 并存）。
- 新轴全部为 optional，保证既有 frozen v3_1 行在新命名空间下仍有效（`additionalProperties` 兼容）。
- 旧多实体 schema 未压成单 flat 表：21 个实体 schema 逐一 rebind 到 xedit_v1_1，仅对 5 个核心数据实体注入正交轴。

## 4. UNKNOWN_OR_BLOCKED
- 无新增 blocker。19 个 PENDING_BLOCKED 资产均带 reason（可 REBOUND），非静默丢弃。
- GSE200302/303/217530（GSE200304 子系列）映射未决，标 PENDING_BLOCKED；GSE145046 需完成 label join 后才计 functional example。

## 5. FILES_READ
- `schemas/v3_1/*`（21 schema + manifest + sha256sums）、`data/v3_1/registry/*`。
- `docs/contracts/mrna_xeditflow_goal_v1_1.md`、`configs/mrna_xeditflow_contract_v1_1.yaml`。

## 6. FILES_CHANGED
- `schemas/xedit_v1_1/`（新增，22 文件：21 个 v3_1 rebind + `xedit_orthogonal_axes.schema.json` + `SCHEMA_MANIFEST.json` + `SCHEMA_SHA256SUMS`）。
- `docs/execution/xeditflow_asset_role_assignment.yaml`（新增，33 资产定级）。
- `scripts/m2_build_xedit_v1_1_schemas.py`、`scripts/m2_build_asset_role_assignment.py`（新增，构建脚本文档化）。
- `tests/migration/test_m2_migration.py`（新增，21 个 M2 gate 测试）。

## 7. COMMANDS_RUN
- `python3 scripts/m2_build_xedit_v1_1_schemas.py`（生成 xedit_v1_1 命名空间）。
- `python3 scripts/m2_build_asset_role_assignment.py`（生成资产定级）。
- `conda activate editflow; python -m pytest tests/migration/ -q`（运行迁移测试）。

## 8. TEST_RESULTS
- `tests/migration/test_m2_migration.py`：**21/21 PASS**（editflow env, pytest 9.1.1）。
- `tests/migration/` 全量：**36/36 PASS**（M1 15 + M2 21，无回归）。
- 关键断言：xedit_v1_1 命名空间与 manifest hash 一致；v3_1 原样保留；5 个核心实体含全部 9 正交轴且新轴 optional；evidence_grade 枚举 A1–D、track 枚举 E/F/AUX/REFERENCE；33 资产全部四类之一且带 evidence；GSE246381 sealed 隔离；GSE207584 不自动解锁 B1；GSE145046 计为 PENDING_BLOCKED。

## 9. DATA_COUNTS_AND_DENOMINATORS
- xedit_v1_1 schema 数：22（21 实体 rebind + 1 正交轴参考）。
- P0/P1 资产定级：**33** 项。
  - `ACCEPTED_FOR_NEW_ROLE`：14
  - `PENDING_BLOCKED`：19（均带 reason）
  - `REFERENCE_ONLY`：0（当前无仅参考资产，语义由 PENDING/ACCEPTED 覆盖）
  - `EXCLUDED_WITH_EVIDENCE`：0（无证据排除项，全部保留或 blocked）
- 核心数据实体（注入 9 正交轴）：dataset_asset、sequence_entity、functional_observation、utr_edit_relation_candidate、utr_edit_pair（5 个）。

## 10. REUSE_DECISIONS
- `schemas/v3_1/*`：**REUSE_AS_IS**（原样保留，git 未改）。
- `schemas/v3_1/* → schemas/xedit_v1_1/*`：**REUSE_WITH_ADAPTER**（rebind contract_id/version + 注入正交轴）。
- `data/v3_1/registry/*`：**REUSE_AS_IS**（frozen 保留）。
- 数据资产：见 `xeditflow_asset_role_assignment.yaml`（ACCEPTED / PENDING 逐项）。

## 11. GATE_STATUS
- **M2 通过**：`MIGRATION_READY_FOR_DATA_REBUILD`（进入 M3 前提满足）。

## 12. CLAIMS_UNLOCKED
- 无新论文 claim 解锁。M2 仅完成数据 schema 兼容迁移与资产定级，不改变效应/生成 claim 状态。

## 13. CLAIMS_STILL_PROHIBITED
- L4（真实生物/治疗改善）PROHIBITED；无新增湿实验。
- GSE207584 未解锁为 B1；GSE145046 未计为 functional example。

## 14. NEXT_PHASE_INPUTS
- M3 输入：`schemas/xedit_v1_1/` 命名空间 + `xeditflow_asset_role_assignment.yaml` + 已建 TaskRegistry v4 / SplitRegistry v4 / task×split matrix，用于 mRNA-EditBench v2 子 benchmark 落地。
- GSE246381 保持 restricted shard，final 前 ordinary loader 返回 0 行。

## 15. COMMIT_SHA
- M2 commit：见下方 `## 16` 之后的 COMMIT（本报告提交后回填）。

## 16. MANIFEST_AND_HASHES
- xedit_v1_1 `SCHEMA_MANIFEST.json`：`11b14927e0497306ca6ae85526e122cf53bf0ab57d9dfaa85aa054a3c8c8bc8d`
- xedit_v1_1 `SCHEMA_SHA256SUMS`：`2ce53d4c5d12225a1b1b9efb8ed5ca2fa91b2865306b76d49ed7e2befcdd073a`
- `xedit_orthogonal_axes.schema.json`：`394f7a9be68e96c1488c116b764616ab514a8425a33cce3bcaa355a9e9fc8869`
- `xeditflow_asset_role_assignment.yaml`：`ee7e20adfed8050ab774cfc912bf5ec2be5eb920d4d50657af48c35c0583df8f`
- `test_m2_migration.py`：`a62eb6d9475e30d2f2f59ee84276611d17c487f58c56927faead115c7aeb632e`
- v3_1 `SCHEMA_MANIFEST.json`（保留）：`1ca045ccf76949f45c529c4edb27b7072c8e3328743e28c9faf59bfe20ce16a7`