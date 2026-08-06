# M1 Contract Supersession — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Phase:** M1 (顶层合同 supersession + crosswalk + 失败测试)
- **UTC:** 2026-08-06
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **M0 base commit:** `1ec9842` · **M1 commit:** `722935d`

---

## 1. FACTS_FROM_REPO
- 主树只读，未修改；M1 全部在隔离 worktree 内完成。
- 旧合同权威文档 `docs/contracts/utr_editflow_goal_v3_1.md` 在 supersession 标注前 SHA256=`35dd4bf2…`。
- 迁移前已存在 v3.1 治理文件（`docs/execution/*_v3_1.yaml`、`schemas/v3_1/*`、`data/v3_1/*`）原样保留，未改动。

## 2. FACTS_FROM_CONTRACTS
- 迁移提示词 `提示词/mrna 合同迁移.md` SHA256=`55c89e5…`。
- 旧合同（本地提示词）`mrna 最新合同-v1.md` SHA256=`ecc6c635…`（合同 ID `utr_editflow_goal_v3.1_benchmark_first`）。
- 新合同 `mrna 最新合同-v2.md` SHA256=`9c79edd8…`。
- 新权威合同 `docs/contracts/mrna_xeditflow_goal_v1_1.md` SHA256=`fc9c1c88…`。
- 新可执行合同 `configs/mrna_xeditflow_contract_v1_1.yaml` SHA256=`b3be70e7…`。

## 3. INFERENCES
- 迁移只 supersede 顶层科学主线；provenance/license/exposure/sealed-final/split/conservation/audit 治理按 hash 原样继承。
- 顶层超指令成功后，旧合同退为 `HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1`，正文不改写（仅 7 行表头标注）。
- Crosswalk 逐条映射 old RQ/Hypothesis/Claim/E-F/AUX/REFERENCE/12 old tasks/10 old splits/P0 blockers/model components/GSE246381/STOP/Track U/alignment/G0-G7/PR1-PR3。

## 4. UNKNOWN_OR_BLOCKED
- 无。5 项冲突（STOP/unlabeled/GSE246381/budget/indel）已按提示词冻结，全部进入 decision log 与 config。

## 5. FILES_READ
- `提示词/mrna 合同迁移.md`、`提示词/mrna 最新合同-v1.md`、`提示词/mrna 最新合同-v2.md`。
- `docs/contracts/utr_editflow_goal_v3_1.md`（旧合同正文）、`docs/execution/*_v3_1.yaml`、`schemas/v3_1/*`。

## 6. FILES_CHANGED
- `docs/contracts/mrna_xeditflow_goal_v1_1.md`（新增权威合同）。
- `configs/mrna_xeditflow_contract_v1_1.yaml`（新增可执行合同）。
- `docs/contracts/supersession_mrna_editflow_v3_1_to_xeditflow_v1_1.md`（新增 supersession 记录，填充实际 SHA）。
- `docs/contracts/utr_editflow_goal_v3_1.md`（仅表头 7 行加 HISTORICAL_SUPERSEDED 标注，正文未动）。
- `docs/execution/xeditflow_migration_decision_log.yaml`、`old_to_new_contract_crosswalk.csv`、`xeditflow_claim_matrix.yaml`、`xeditflow_task_registry.yaml`、`xeditflow_split_registry.yaml`、`xeditflow_task_split_matrix.yaml`（新增）。
- `tests/migration/test_m1_migration_contract.py`（新增 M1 gate 测试）。

## 7. COMMANDS_RUN
- `scp` 传输 M1 产物到 worktree；`python m1_apply_supersession.py` 加表头并回填 SHA；`pytest tests/migration/test_m1_migration_contract.py -v`；`git add/commit`。

## 8. TEST_RESULTS
- `tests/migration/test_m1_migration_contract.py`：**15/15 PASS**（editflow env, pytest 9.1.1）。
- 关键断言：single authoritative contract=1；active 权威层不把旧 contract ID 标为 ACTIVE；旧合同 md 标记 HISTORICAL_SUPERSEDED；crosswalk 覆盖；决策日志冻结；task/split registry expected-set 闭合；task×split FK closure；sealed S6 隔离。

## 9. DATA_COUNTS_AND_DENOMINATORS
- Crosswalk 行数：68（含旧合同 supersession、Track U、STOP 行）。
- Task registry：primary=4、secondary=6、theory=2（合计 12）。
- Split registry：8 splits（S1–S8）；sealed external = S6，不进入 task activation/metric/calibration/model-selection。
- Claim matrix：11 条 claim（L0–L4 + C_*），L4=PROHIBITED。

## 10. REUSE_DECISIONS
- 治理层（provenance/license/exposure/sealed/split/audit）：RETENTION（按 hash 继承）。
- 顶层科学主线（RQ/Hypothesis/Claim/方法中心）：SUPERSEDED_BY_NEW_METHOD 或 RETAIN_AS_SECONDARY。
- 数据资产（GSE145046/114002/149487/217518/200304/232572/186455/173083/207584/246381）：REUSE_WITH_ADAPTER / RETAIN_WITH_RENAME / RETAIN_EXACT。
- Track U、learned STOP：PROHIBITED。

## 11. GATE_STATUS
- **M1 通过**：`MIGRATION_READY_FOR_DATA_REBUILD`（进入 M2 前提满足）。

## 12. CLAIMS_UNLOCKED
- 无新论文 claim 解锁；M1 只完成合同 supersession，不改变数据/效应 claim 状态。（旧 PASS 默认 STALE_REVALIDATION_REQUIRED。）

## 13. CLAIMS_STILL_PROHIBITED
- L4（真实生物/治疗改善）PROHIBITED；无新增湿实验。
- exact 仅表示 learned density-ratio/path 假设下的数学 exactness，不表示真实生物最优。

## 14. NEXT_PHASE_INPUTS
- M2 输入：保留的 `schemas/v3_1/`、`data/v3_1/` canonical 管线、P0/P1 资产清单、新增 evidence_grade/track/role 正交轴映射。
- M3 输入：TaskRegistry v4 / SplitRegistry v4 / task×split matrix（已建），用于 mRNA-EditBench v2 子 benchmark 落地。

## 15. COMMIT_SHA
- M1 commit：`722935d`（11 files changed, 968 insertions）。

## 16. MANIFEST_AND_HASHES
- 新权威合同 md：`fc9c1c882efbaa4c1e86f4da2e1be64e219755fb9c5941da4b4309793d3d8c2f`
- 新可执行合同 yaml：`b3be70e765fb8285996487815ee6a4494ca4cc7fb503dae2901b40a0382d83cf`
- 旧合同正文 supersession 前 SHA：`35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759`
- 迁移提示词：`55c89e5dc065a116063bdcdcd594322db4f2f33f6cc5d25c27cd4fc402fab916`
- 旧合同（本地提示词）：`ecc6c635f112575db2f14309c869a378fc31df8fb76c01dda0b54b832b4f8946`