# S1 Stability Eval Row — row_definition.md（口径全声明）

- **row_key**: `S1_STABILITY_EVAL_ROW`
- **study_unit**: SU2025（Su 2025, eLife 97682；与 frozen GSE217518 同库——审计已证实）
- **构建日期**: 2026-09-20；构建脚本：`/home/cunyuliu/mrna_editflow_goal/benchmark_v2_rows/build_s1_stability_eval_row_v1.py`（可复跑，无抽样——全量 eligible 集即评测行）
- **split 状态**: `NEW_EVAL_ROW` — 不进既有 development manifest；只作 frozen-Δ 评测行（只增行条款）。

## 1. 数据源（只读）

- `route2/audits/candidate_datasets_v1/s1_su2025_stability_extracted_pairs_v1/s1_variant_sequence_pairs.v1.jsonl`（5,072 行：canonical_115 + task_155 双口径 ref/alt 序列、SH/HEK 双 assay 表型 t05_WT/mt/pval/sig、逐行 audit_hits 与 split 归属）。
- 审计 verdict（`S1_SU_2025_STABILITY.pigeonhole_audit.v2.json`）："Any future use of S1 rows as additive candidates must exclude/de-duplicate rows overlapping GSE217518 VALIDATION/TEST (4 sequence-level protected rows + 1,328 mutant-id-level protected rows)"。

## 2. 受保护行排除（逐行判定，按 verdict 口径）

- 规则：`protected_overlap == true`（序列级）**或** `gse217518_mutant_id_splits ∩ {VALIDATION, TEST} ≠ ∅`（变异 ID 级）→ 排除。
- 实测：5,072 − **1,330**（联合去重：序列级 4 + 变异 ID 级 1,328，二者重叠 2 行）= **3,742 eligible 行**（3'UTR 1,871 / 5'UTR 1,871）。
- 排除是硬排除（不打 tag 保留，直接不进入评测行）——与 verdict "exclude/de-duplicate" 一致。TRAIN 重叠行（mutant-id TRAIN 1,450 行）**保留**（同库 TRAIN 重复不构成 VAL/TEST 泄漏；评测集含与训练同库的行是 S1 的固有性质，审计已声明 same-library overlap expected）。

## 3. Dao cryptic splicing QC（仅 3'UTR 臂 1,871 eligible 行）

**筛查实现**（脚本内嵌，可复跑）：
- U-rich：滑窗 10nt，窗口内 T/U 计数 ≥ 7（占比 ≥ 0.7）→ u_rich flag（ref 与 alt 序列任一命中即算）。
- 5' 剪接供体样信号：序列内 GT（含 U→T 归一）位置 i，下游 ≤40nt 内存在 AG → (i, j) 信号对。
- **risk flag = u_rich AND splice_like_signal 同时成立**。

**处置决定：flag 但保留 + 标注（FLAG_KEEP_ANNOTATE），不排除。理由**：
1. Dao 2025 (Nat Commun) 的排除规则针对**合成 3'UTR MPRA 文库插入**中隐匿剪接导致的测量伪影（插入序列被细胞剪接机器识别为内含子，条目丰度被剪接事件污染）；Su 2025 的行是**内源基因组 UTR 变异对**（真实基因座位 ±57nt 上下文），测量体系为报告基因 half-life，不存在"插入序列被当作内含子剪掉"的同构伪影路径。
2. U-rich 基序在天然 3'UTR 常见（polyA 信号邻域、AU-rich 元件），强行排除会引入对 UTR 组成的选择偏置，破坏评测集对内源变异空间的代表性。
3. 保留 flag（行内 `tags: cryptic_splice_risk_dao_qc` + `dao_qc` 字段含窗口参数与信号坐标）使评测后可做敏感性分析（flagged 子集 vs 非 flagged 子集的 Δ 可分性对比）。

**实测（1,871 行 3'UTR eligible）**：flagged 834 / pass 1,037；分解：u_rich_only 25、splice_signal_only 990（两者单独出现均不 flag）。flag 比例 ~45%，其中 splice_signal_only 的 GT..AG 在随机序列中的期望命中率高（155nt 窗口 GT 期望 ~7 个），故联合条件（u_rich AND signal）才是有效筛查器——834 行为最终 flagged 集。5'UTR 臂 1,871 行不做该 QC（`5UTR_qc_na`，审计口径：Dao 门适用于 3'UTR MPRA 候选）。

## 4. 行结构：双 assay sub-row（biological_context 区分）

每个 eligible 行 × 有表型的 assay 展开为 sub-row（镜像 GSE217518 canonical converter 的 context 双行结构：HEK293T / SH_SY5Y 两 context 各成行）：

- SH sub-row：biological_context = `SH_SY5Y`，用 t05_WT_SH / t05_mt_SH
- HEK sub-row：biological_context = `HEK293T`，用 t05_WT_HEK / t05_mt_HEK
- 单 assay 表型缺失（null）时只发另一 assay 的 sub-row（SH null 950 / HEK null 962 行——源表缺测如实，不插补）

序列：canonical_115 对（ref/alt，115nt，变异锚定窗口末端 19nt 前——审计验证其精确复现 GSE217518 canonical 源窗口口径）；task_155 对未使用（单一窗口约定，记录）。DNA 字母表（源提取自 hg38）。

## 5. Δy 口径（一句话）

**Δy = t05_mt − t05_WT（分钟，RNA half-life 差，HIGHER_IS_BETTER）**，镜像 GSE217518 canonical 的 endpoint_id=RNA_HALF_LIFE_MINUTES、LEVEL_DIFFERENCE 口径（assay_id=UTR_HALF_LIFE_MPRA 同名镜像）。

注意（如实声明）：源表 t05 存在异常量级值（dy min ≈ −2.87M 分钟 / max ≈ +5.7 万分钟），**原样保留不清洗**（评测行如实反映源测量；评测聚合时建议使用稳健统计或预声明裁剪规则，此提示写入本文件供评测方决策）。

## 6. 统计（row_manifest.json 同步）

- 5,572 eval sub-rows（SH 2,792 / HEK 2,780）；1,519 源组（gene × arm）；密度 mean cand/源 = **3.67**（max 150，PTEN 5'UTR 等多变异基因族——天然多候选结构，如实保留）。
- arm 分布：sub-row 层面 3'UTR ≈ 2,754 / 5'UTR ≈ 2,818（eligible 行层面 1,871/1,871 均衡）。

## 7. 字段 schema

统一行 schema（row_id / study_unit / source_sequence / candidate_sequence / source_relative_edits / task_id / source_group_id / assay / biological_context / region / endpoint_descriptor / direction_normalized_delta / source_endpoint_value / candidate_endpoint_value / endpoint_direction / eval_split_status / tags / provenance），另含 `variant_id / variant_name / variant_class / phenotype_pval / phenotype_sig / dao_qc`（可追溯与敏感性分析）。region 按 arm（3UTR/5UTR）；task_id = `RNA_HALF_LIFE_MINUTES::region=<arm>`。

**edit 表示约定**：等长变异（4,821 sub-row，SNV/等长 COMPLEX）= 逐位 SUB；长度变化变异（751 sub-row，DEL 410 / COMPLEX 280 / INS 61——115nt 窗口在 indel 后序列长度 116-118 等）= 单条 `LENGTH_CHANGE` 操作（首分歧位 + 尾段替换，镜像 GSE186455 canonical 语义）。终验 0 mismatch。

## 8. 纪律声明

源 jsonl（审计产物）只读；不写既有 manifest；产物只增（`route2/benchmark_v2/s1_stability_eval_row/`）；protected reads = 0（受保护行已在构建时排除，评测行集内 0 行触及 GSE217518 VAL/TEST 受保护 split）；脚本与文档不 commit；本行为评测行不进训练管线。
