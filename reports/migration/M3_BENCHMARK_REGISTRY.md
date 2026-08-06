# M3 Benchmark & Registry Migration — mRNA-EditFlow v3.1 → mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Phase:** M3 (mRNA-EditBench v2 子 benchmark + task/split registry 落地)
- **UTC:** 2026-08-06
- **Migration worktree:** `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z`
- **Migration branch:** `xeditflow-migration-20260806T024650Z`
- **M2 commit:** `2641661` · **M3 base:** `2641661`

---

## 1. FACTS_FROM_REPO
- 主树只读未改；M3 全部在隔离 worktree 内完成。
- M1 已建 TaskRegistry v4 / SplitRegistry v4 / task×split matrix（`docs/execution/xeditflow_task_registry.yaml`、`xeditflow_split_registry.yaml`、`xeditflow_task_split_matrix.yaml`）原样复用。
- M2 已建 `docs/execution/xeditflow_asset_role_assignment.yaml`（33 个 P0/P1 资产定级）作为 benchmark 资产绑定的唯一来源。
- 新增 `docs/execution/xeditflow_benchmark_registry.yaml` 定义 mRNA-EditBench v2（4 个子 benchmark）。

## 2. FACTS_FROM_CONTRACTS
- 新权威合同 `mrna_xeditflow_goal_v1_1`（SHA `fc9c1c88…`）为唯一 ACTIVE 权威。
- 迁移提示词 §七 Phase M3 要求：构建 mRNA-EditBench v2 + 新 task/split registry；不得对不存在合格数据的子 benchmark 造空 PASS。
- 新合同任务命名：`T5_SOURCE_RELATIVE_EFFECT`、`T5_SELECTIVE_EFFECT`、`T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION`、`T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION`、`CDS_SYNONYMOUS_FAMILY_RANKING` 等。

## 3. INFERENCES
- 4 个子 benchmark 按 region × evidence_grade 划分：5U-A1-Natural、5U-A2-Dense、3U-A1-Variant、CDS-B1-Synonymous。
- 5'UTR 与 3'UTR 是独立 endpoint heads，池不混用（ENCSR854RUF 为 3'UTR 资产，只进 3U-A1，绝不进 5U 池）。
- CDS-B1 无合格数据（GSE207584 为 PENDING_BLOCKED legacy 责任），标 **DORMANT** 且不绑定任何资产，绝不 fabricate PASS。
- GSE246381 为 sealed external final candidate，不进任何 benchmark 池、不进 activation/metric/calibration/model-selection，ordinary loader final 前返回 0 行，final evaluator 上限 1。

## 4. UNKNOWN_OR_BLOCKED
- 无阻塞。CDS-B1 明确 DORMANT + status_reason，等 rebuilt sequence/family/label 后才可解锁为 B1。
- 5U-A2 当前仅 GSE330741 一项 ACCEPTED；其余 A2 候选（如 GSE246381）因 CRITIC_AUX/sealed 角色不进入 EFFECT_PRIMARY 池。

## 5. FILES_READ
- `docs/execution/xeditflow_task_registry.yaml`、`xeditflow_split_registry.yaml`、`xeditflow_task_split_matrix.yaml`、`xeditflow_asset_role_assignment.yaml`。
- `docs/contracts/mrna_xeditflow_goal_v1_1.md`、`configs/mrna_xeditflow_contract_v1_1.yaml`。

## 6. FILES_CHANGED
- `docs/execution/xeditflow_benchmark_registry.yaml`（新增，mRNA-EditBench v2 定义）。
- `scripts/m3_build_benchmark_registry.py`（新增，构建脚本，含 expected-set/FK/asset 绑定断言）。
- `tests/migration/test_m3_migration.py`（新增，11 个 M3 gate 测试）。

## 7. COMMANDS_RUN
- `python3 scripts/m3_build_benchmark_registry.py`（重新生成 benchmark registry，修复排序与 ENCSR854RUF 跨区泄漏）。
- `conda activate editflow; python -m pytest tests/migration/test_m3_migration.py -v`（运行 M3 测试）。
- `python -m pytest tests/migration/ -v`（全量迁移测试回归）。

## 8. TEST_RESULTS
- `tests/migration/test_m3_migration.py`：**11/11 PASS**（editflow env, pytest 9.1.1）。
- `tests/migration/` 全量：**47/47 PASS**（M1 15 + M2 21 + M3 11，无回归）。
- 关键断言：expected_benchmark_id 闭合（4 个子 benchmark）；task/split FK 闭合（每个绑定的 task/split 存在于 v4 registry 且被 task×split matrix 允许）；asset 绑定全部为 ACCEPTED_FOR_NEW_ROLE；5U/3U 池不混用且 ENCSR854RUF 不进 5U；CDS-B1 DORMANT 不 fabricate PASS 且不绑定资产；GSE246381 sealed 隔离（不进池/activation/metric/calibration/model-selection）；S6 sealed split 不进 activation。

## 9. DATA_COUNTS_AND_DENOMINATORS
- mRNA-EditBench v2 子 benchmark 数：**4**。
  - `EditBench-5U-A1-Natural`：ACTIVE，9 个 A1 资产（GSE114002/186455/200304/217518/232571/232572/261709/288185/298114），4 个 T5 primary tasks，splits S1–S5。
  - `EditBench-5U-A2-Dense`：ACTIVE，1 个 A2 资产（GSE330741），2 个 T5 tasks，splits S1–S4。
  - `EditBench-3U-A1-Variant`：ACTIVE，1 个 3'UTR 资产（ENCSR854RUF），T3_EFFECT_TRANSFER + CROSS_REGION_TRANSFER，splits S4–S5。
  - `EditBench-CDS-B1-Synonymous`：**DORMANT**，0 资产，CDS_SYNONYMOUS_FAMILY_RANKING，split S7。
- 所有子 benchmark sealed_external = S6（不进 activation）。
- GSE246381：SEALED_EXTERNAL_FINAL_CANDIDATE，final_evaluator_count_max=1。

## 10. REUSE_DECISIONS
- TaskRegistry v4 / SplitRegistry v4 / task×split matrix：**REUSE_AS_IS**（M1 已建，M3 直接绑定，未改动）。
- `xeditflow_asset_role_assignment.yaml`：**REUSE_AS_IS**（M2 已建，作为 asset 绑定唯一来源）。
- 旧 benchmark 治理（21 schemas/12 tasks/10 splits/120-row matrix）：**REUSE_AS_IS / RETAIN_HISTORICAL**（不删除，不替换；新建 v2 registry 与其并存）。
- 数据资产：逐项由 M2 asset role 决定（见 `xeditflow_asset_role_assignment.yaml`）。

## 11. GATE_STATUS
- **M3 通过**：`MIGRATION_READY_FOR_MODEL_LINE_SUPERSESSION`（顶层科学主线 supersession 的 registry 前提满足）。

## 12. CLAIMS_UNLOCKED
- 无新论文 claim 解锁。M3 仅完成 mRNA-EditBench v2 子 benchmark 与 registry 落地，不改变效应/生成 claim 状态。

## 13. CLAIMS_STILL_PROHIBITED
- L4（真实生物/治疗改善）PROHIBITED；无新增湿实验。
- CDS-B1 不因存在 GSE207584 legacy 数据而自动解锁（DORMANT）。
- exact 仅表示 learned density-ratio/path 假设下的数学 exactness，不表示真实生物最优。

## 14. NEXT_PHASE_INPUTS
- M4 输入：mRNA-EditBench v2 registry + SparseEditFormer 模型 line supersession + exact guidance 理论包。
- 正式训练前：不启动训练，不打开 GSE246381 final，不把旧 PASS 自动继承为新合同 PASS。
- CDS-B1 需 rebuilt sequence/family/label 后才可激活。

## 15. COMMIT_SHA
- M3 产物 commit：`187c95f`（4 files changed, 521 insertions）。

## 16. MANIFEST_AND_HASHES
- `xeditflow_benchmark_registry.yaml`：`c9453f6b531de690fac88a095d1533af59be14a7bf6709b21054e7b30451efe3`
- `scripts/m3_build_benchmark_registry.py`：`e88802681fcc39b52d695f9f6f220706b1031352e2108aa003de9c46f051776d`
- `tests/migration/test_m3_migration.py`：`058d80a601f8fdaf416baf98d8a93f8368c32d29b7c1cfdea242ea93d25390c9`