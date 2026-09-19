# M1 MRL Eval Row — row_definition.md（口径全声明）

- **row_key**: `M1_MRL_EVAL_ROW`
- **study_unit**: GSE232927（Castillo-Hair 2024，三细胞 5'UTR MRL 文库：HepG2 defined_end / 原代 T cell defined_end r1+r2 / HEK293T random_end N25 r1+r2 + N50 r1）
- **构建日期**: 2026-09-20；构建脚本：`/home/cunyuliu/mrna_editflow_goal/benchmark_v2_rows/build_m1_mrl_eval_row_v1.py`（可复跑，seed 落盘）
- **split 状态**: `NEW_EVAL_ROW` — 本行不进入既有 `route2_development_frozen_v1` development manifest，只作 frozen-Δ 评测行（只增行条款：不修改任何既有 manifest/训练管线）。

## 1. 数据源（只读）

- 转换产物：`route2/external_model_assets/castillohair2024/mrl_converted_20260909/GSE232927_processed_*_mrl.tsv.gz`（6 文件，1,128,562 kept 行；MRL 标签 = `rl_paper`，conversion_summary.json 声明为权威值；`rl_recomputed` 仅为 sanity 列不使用）。
- 泄漏审计依据：`route2/audits/candidate_datasets_v1/M1_CASTILLOHAIR_2024.pigeonhole_audit.v2.json`（exact 全 frozen corpus 重叠 4 行，全部 TRAIN split；gate 未触发）。
- 近重复审计：`external_model_assets/candidate_datasets/castillohair2024/audit_20260909/near_duplicate_audit_result.json`（mmseqs2 linsearch 0.8/0.8，120 行命中全为局部短比对，无全长度重叠，判定非泄漏）。

## 2. 行单位与配对口径（镜像 GSE114002 既有做法）

GSE114002 canonical converter（`route_a_v3_route2_gse114002_converter_v1.json`）把 280K 库拆成 source/candidate 的规则是：`identity_rule = DESIGNED_TRUE_AND_UTR_EQUALS_MOTHER`（mother 序列 = source），`candidate_rule = DESIGNED_FALSE_AND_EQUAL_LENGTH_HAMMING_DISTANCE_1_TO_3`（同长 Hamming 1-3 变体 = candidate），按 mother family（共享 mother 序列）建 source_group。M1 没有设计注记列（无 mother/design 标志），因此以序列结构本身镜像该逻辑：

1. **prefix group（mother-family 模拟）**：每文件内共享 15nt 前缀的 kept 序列组成一个 group。defined_end 文库（hepg2_r1 / tcell_r1 / tcell_r2）实测存在真实前缀家族（prefix-15 多成员组数千个）；random_end 文库（N25/N50）前缀几乎全唯一（生日碰撞级）。
2. **组内配对**：source = 组内字典序最小序列（mother 模拟）；candidate = 组内与 source Hamming 距离 = 1 的成员（GSE114002 candidate rule 的保守收窄：1-3 → 1，评测行确定性优先）。每个 candidate 序列至多被使用一次（跨 group 不重复为 candidate）。
3. **全局 Hamming-1 兜底（随机库）**：未被 prefix 配对消耗的序列，用 deletion-neighborhood 哈希做全局 Hamming-1 邻居检测；每条序列作为 source 与其全部未消耗的 Hamming-1 邻居配对（同样每 candidate 至多用一次）。随机库中的 Hamming-1 对既可能是真实变体兄弟也可能是生日碰撞（25nt 库 N≈110K 时碰撞期望非零），**两种成因均如实保留并在 row_manifest 的 per-file 统计中记录 pairs_found / global_hamming1_pairs，评测解读时知悉**。

### K 候选/源密度统计（per file）

| file | kept | pairs_found | 其中 global_hamming1 | 抽样 | 发出行 |
|---|---|---|---|---|---|
| hepg2_r1 | 288,572 | 5,025 | ~5,000 | 800 | 800 |
| tcell_r1 | 236,221 | 1,788 | ~1,700 | 800 | 800 |
| tcell_r2 | 264,916 | 2,360 | ~2,300 | 800 | 800 |
| N25_r1 | 110,663 | 19 | 19 | 全部 | 19 |
| N25_r2 | 79,501 | 12 | 12 | 全部 | 12 |
| N50_r1 | 148,689 | 374 | 374 | 374 | 374 |

最终密度：2,805 行 / 2,801 源组，mean cand/源 = **1.0014**（max 3）。GSE114002 的 ~4 cand/源密度由其设计性变体库产生；M1 随机库的天然 Hamming-1 兄弟密度即如此，如实记录不人为放大。

## 3. Δy 口径（一句话）

**Δy = rl_paper(candidate) − rl_paper(source)**（MEAN_RIBOSOME_LOAD，HIGHER_IS_BETTER；absolute level difference，无标准化/无对数变换——直接镜像 GSE114002 canonical 的 direction_normalized_delta = cand − source 口径）。

## 4. VALIDATION 口径评测子集（冻结）

- 从 6 文件 1,128,562 kept 行的配对池（9,578 对）中抽样：seed = **20260920**（`random.Random(20260920)`，per-file 顺序抽样，脚本内嵌、可复跑）；每文件目标 ~800 行（建议区间 600-900 内），随机库文件全量纳入（数量 < 800）。
- 最终 **2,805 行 / 2,801 源组 / 3 细胞上下文**（HEPG2 800 / PRIMARY_T_CELL 1,600 / HEK293T 405）。

## 5. TRAIN 重叠排除

审计 v2 的 4 条 TRAIN 重叠序列（tcell_r1 ×2、tcell_r2 ×2）在 kept 集内检索：4 条均 `keep=0`（低于论文 total≥100 reads 阈值），**天然不在配对池**。脚本仍内置显式 `cross_study_duplicate` 标记逻辑（命中即打 tag 排除出评测子集），本次实际触发 0 次（`total_train_overlap_tagged_excluded: 0`）。语义：TRAIN 重叠不构成 VALIDATION/TEST 泄漏（审计 gate 为 BLOCK 条件是 VAL/TEST 重叠），行内不排除 TRAIN 重叠只做标记；但本行是评测行且 4 条已天然不在池内，故无实际影响。

## 6. 字段 schema（镜像既有 canonical/projection 口径）

`row_id / study_unit / source_sequence / candidate_sequence / source_relative_edits / task_id / source_group_id / assay / biological_context / region / endpoint_descriptor{endpoint_id, quantity_family, measurement_form, numerator_family, denominator_family} / direction_normalized_delta / source_endpoint_value / candidate_endpoint_value / endpoint_direction / eval_split_status / pair_kind / tags / provenance{...}`

- `source_sequence`/`candidate_sequence`：DNA 字母表（源表即 DNA 记法；既有 canonical GSE114002 同为 DNA 记法，镜像保持）。
- `endpoint_descriptor`：复用 `route_a_v3_route2_endpoint_descriptors_v1.json` 中 MEAN_RIBOSOME_LOAD 定义（LEVEL_DIFFERENCE）。
- `biological_context`：HEPG2_CELL_LINE / PRIMARY_T_CELL / HEK293T_CELL_LINE（tcell r1/r2 同为 T 细胞上下文，行 ID 与 source_group_id 均含文件标签可区分重复）。

## 7. 纪律声明

- 源数据与审计产物只读（`mrl_converted_20260909/`、`audits/` 未被写入）；protected reads = 0；本行不写入任何既有 manifest；产物只增（`route2/benchmark_v2/m1_mrl_eval_row/` 新目录）。
- 构建脚本与本文件不 commit（等主会话统一）。
- 本行为评测行（模型族 frozen-Δ 评测），不进训练管线。
