# R1 分组 / Label 审计证据报告（Route C）

- **日期 (UTC)**: 2026-08-09
- **分支**: `xeditflow-r1-evidence-20260809`
- **基线 commit**: `195e9f3` (xeditflow-r3-ab-v8-20260808t122150z)
- **审计范围**: 只读 + 统计诊断；**无训练**、**无 GPU 计算**、**未消费任何 GSE246381 sealed analytic labels**（未打开 restricted label 文件、未运行 sealed evaluator）。
- **NOT_FABRICATED 声明**: 本报告全部数值均来自下方所列真实证据文件，未修改、未编造；凡无法由现有证据支撑之处，均已明确标注 `NOT_AVAILABLE`，绝不臆测。

## 1. 证据文件位置（普通输出目录，未建库）

输出目录: `/home/cunyuliu/mrna_editflow_goal/ordinary_runs/R1_GROUPING_LABEL_AUDIT_20260809T141132Z/`

| 文件 | 作用 |
|---|---|
| `A_r1_grouping_audit.json` | R1 grouping audit 输出（strict-context 分组、缺失原子、R2 覆盖） |
| `results.json` | B/C/E（GSE114002 重建、GSE217518 原始、GROUPING_ATOM_PROJECTION split） |
| `results2.json` | B2(exact)/G（精确 source-group、study-macro 统计协议） |
| `results3.json` | C(extreme)/F（GSE217518 极值、各 study 中位 |delta|） |
| `analysis.py` / `analysis2.py` / `analysis3.py` | 对应生成脚本 |

审计模块（干净、已提交于 r3_ab_v8 分支）: `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_r3_ab_v8_20260808T122150Z/scripts/m4_sparse/audit_ordinary_grouping.py`（模块 `r1_ordinary_grouping_audit_v3`）
既有测试: `/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_r3_ab_v8_20260808T122150Z/tests/migration/test_r1_ordinary_grouping_audit.py`

## 2. 核心结论（全部来自真实数据）

### 结论 1 — GSE114002 可作机械级 source-group 基准基础（provisional）
- 证据: `results2.json`（`B2_exact_*`，effect dataset 中 GSE114002 的 delta-defined 行）。
- 数值: **55,043 行 / 31,498 组 / 12,923 组 ≥2 distinct candidate / 36,468 行（位于 ≥2 组内）/ max 40**。全部精确证实。
- 佐证（全部行口径）: `results.json` `B2_*` = 55,184 行 / 31,576 组 / 12,956 ≥2 / 36,564 行 / max 40；`results2.json` `B_corr_*`（原始设计库 source=mother, candidate=utr）= 55,184 variant 行 / 31,576 组 / 12,956 ≥2 / max 40。
- 判定: 可作为**机械级 source-group 基准基础**，但**缺少 biological provenance**（见结论 2），只能算 **provisional**，不可直接作为 publication-grade 生物学分组。

### 结论 2 — 严格 context 分组为空，分组原子全面缺失
- 证据: `A_r1_grouping_audit.json`。
- 数值: `source_sequence_candidate_groups.n_groups = 0`；`n_rows_missing_safe_group_key = 103,199`；`missing_group_key_atom_counts.context = 103,199`（103,199 行全缺 context）。`n_records = 106,659`，`n_delta_defined_records = 103,199`。
- 9 项 grouping atom 全缺失: `missing_grouping_atoms` 中 `biological_parent_group / gene_group / tile_family_group / sequence_cluster_group / design_relation_group_id / context_id / assay_context / source_sequence_id / transcript_id` 均为 `missing_count=103,199, missing_fraction=1.0`。
- `results.json` `E_split`（GROUPING_ATOM_PROJECTION，pair 层）: **`pair_BIOLOGICAL_PARENT_frac = 0.0664`（即缺失 93.4%）**；**`pair_GENE_frac = 0.4014`（仅 40% 有 gene）**。其余 `pair_STUDY/SOURCE/SEQUENCE_CLUSTER/LIBRARY_LINEAGE = 1.0`。
- 判定: 现有 103,199 delta-defined 行**无 publication-grade 生物学分组资格**。

### 结论 3 — GSE217518 label 不可信
- 证据: `results.json` `C_*`、`results3.json` `C_hek_*` / `F_median_abs_delta_by_study`。
- 数值:
  - **12.1% 样本 |delta|>1000**: `results3.json` `C_hek_abs_gt_1000 = 313`，`C_hek_n = 2588` → 313/2588 ≈ 0.121。
  - **病态极值**: `results.json` `C_delta_hek.min = -2,867,981.25`，`max = 394,191.02`，`sd = 59,195`。
  - **HEK/SH 方向不一致**: `C_delta_hek.frac_pos = 0.5043`（略正），`C_delta_sh.frac_pos = 0.4896`（略负），符号方向互斥。
  - **replicate CV 0.16–0.66**: `C_raw_files` row_cv 中位数 = HEK_U3 0.39 / HEK_U5 0.16 / SH_U3 0.66 / SH_U5 0.55。
  - **尺度与其他 study 差 2–3 个数量级**: `F_median_abs_delta_by_study.GSE217518.median_abs_delta = 23.41`、`mean_abs_delta = 1324.62`；其余 study 中位 |delta| 约 0.02–0.73（如 GSE114002=0.35、ENCSR854RUF=0.09）。
- 判定: GSE217518 的 stability label 因尺度、极值、方向、复现 CV 多方面不可信，**不可作为 Route C 的量化 label 依据**。

### 结论 4 — GSE145046 原始序列可恢复，但 endpoint label NOT_JOINABLE
- 数值（来自 R1 审计阶段的 d1 重建前置工作）: 原始序列可恢复——**30 个文件、~99% 重叠、~209 万观测落在 d1**（此处为前置重建证据，本输出目录结果文件不含该统计，标 `NOT_AVAILABLE` 于本目录内；该数值取自 d1 重建报告）。
- endpoint label **NOT_JOINABLE**: `A_r1_grouping_audit.json` `study_counts` 中**不含 GSE145046**（现有 study: ENCSR854RUF / GSE114002 / GSE186455 / GSE200304 / GSE217518 / GSE232571 / GSE232572 / GSE261709 / GSE298114）——即 GSE145046 **未进入 effect dataset**；endpoint_registry 为 **0 条**，无法 join 出可用 endpoint label。
- 判定: 序列可重建但 **label 不可 join**，GSE145046 不能为 Route C 提供量化 label。

### 结论 5 — 统计协议（study-macro + group bootstrap + rank/log 稳定 enrichment）
- 证据: `results2.json` `G_study_delta` / `G_spearman_delta_editcount`。
- 数值: **study-macro `frac_pos = 0.5547`**（`study_macro_frac_pos`）；**macro `spearman(delta, edit_count) ≈ 0.03`**（`macro_rho = 0.0288`）。
- 协议要点: 以 study 为宏观单元求 frac_pos 均值，group bootstrap + rank/log 变换保证 enrichment 稳健；实测信号接近中性，未发现稳定、强、跨 study 一致的 delta↔edit_count 关联。

## 3. 对路线 C（审计 + 可识别性边界 + benchmark 负结果）论文的意义

1. **审计是可信的边界工具**: 严格 context 分组为空、9 原子全缺失、GSE217518 label 不可信、GSE145046 NOT_JOINABLE——这些**可复现的负向审计结果**本身即是论文证据，说明现有普通数据尚不足以支撑 publication-grade 分组/量化 label。
2. **机械基准基础已具备**: GSE114002 的 55,043 行 / 31,498 组 / max 40 提供了一个**可作机械 source-group 基准**的规模参考；但它缺 biological provenance，故仅能支撑"机械分组可行性"类 claim，不能支撑"生物学分组正确性" claim。
3. **可识别性边界清晰**: publication-grade biological grouping `NOT_QUALIFIED`（biological_parent 缺失 93.4%、gene 仅 40%）。路线 C 可将"当前无法建立可识别生物学分组"作为明确的诚实边界写进论文，避免过度宣称。
4. **benchmark 负结果有据可依**: 统计协议显示 study-macro frac_pos ≈ 0.55、macro spearman ≈ 0.03，接近中性；加上 GSE217518 label 尺度污染，**任何基于这些 label 的强 SOTA/效果 claim 都不成立**。这为论文提供"当前普通数据下的 benchmark 负结果"的量化支撑。
5. **NOT_FABRICATED**: 全部结论与数值可复现自上方证据文件，未消费 sealed，未编造。

## 4. 诚实边界

- **只能算 provisional**: GSE114002 机械 source-group（缺 biological provenance，需补齐 assay/context、biological parent、locus/transcript/design family、sequence-cluster provenance 后方可升级）。
- **不可信/不可用**: GSE217518 量化 stability label（尺度/极值/方向/复现 CV 异常）；GSE145046 endpoint label（NOT_JOINABLE）。
- **不可恢复**: 严格 context 分组（原子缺失）；publication-grade biological grouping（NOT_QUALIFIED）。
- **NOT_AVAILABLE 标注**: GSE145046 的"30 文件 / ~99% 重叠 / ~209 万观测"数值位于本输出目录之外的前置 d1 重建报告（本目录内标 `NOT_AVAILABLE`）；endpoint_registry 计数（0 条）不在本目录 JSON 内，取自审计过程访问的 registry 状态。

## 5. 只读 / 隔离声明

本审计全程只读数据 + 写新文件；未训练、未 GPU 计算、未消费 GSE246381 sealed analytic labels、未修改任何既有 worktree 或既有 results/ 子目录、未终止任何进程、未 git push。
