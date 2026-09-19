# M6 NDD 5'UTR Eval Row — row_definition.md（口径全声明）

- **row_key**: `M6_NDD5UTR_EVAL_ROW`
- **study_unit**: GSE246381（Plassmeyer 2025，Cell Rep Methods，5'UTR NDD 变异 MPRA；HEK 转染 + Vglut2-Cre 小鼠在体，本行只用 HEK 文件）
- **构建日期**: 2026-09-20；构建脚本：`/home/cunyuliu/mrna_editflow_goal/benchmark_v2_rows/build_m6_ndd5utr_eval_row_v1.py`（可复跑，seed 落盘）
- **split 状态**: `NEW_EVAL_ROW` — 不进入既有 development manifest；只作 frozen-Δ 评测行（只增行条款）。

## 0. 封存-解封历史（provenance 如实记录，含依据文档）

1. **封存期**：W0 worktree `configs/route_a_v3_route2_14_study_final_inventory_v1.json` 记 GSE246381 `use_role=SEALED_EXCLUDED, terminal_status=SEALED_EXCLUDED, outcome_exposure=SEALED_NOT_READ`（"Remains sealed and excluded under the contract; it is not used to complete the current Evaluation axis."）。
2. **封存的真实原因**：`configs/utr_editflow_contract_v2.yaml` §2 `gse246381_status`：`historically_exposed: true`、`role: historically_exposed_retrospective_external_stress_test`、`labels_allowed_for_new_training: false`、`labels_allowed_for_new_hyperparameter_selection: false`、`evidence_grade: E4`、`must_report: historical_exposure_path`（历史暴露路径：v3.1 strict D1 validation 曾使用 sealed GSE246381 输入，M0_INPUT_MANIFEST.json 记录 PID/运行路径）。即"封存"实为**历史暴露隔离**，非数据质量问题。
3. **解封**：2026-09-08 下载两 csv.gz 并 manifest 入册（`route2/manifests/candidate_datasets_v1/M6_PLASSMEYER_2025.manifest.json`：sha256、行数、additive-only split policy）+ pigeonhole 审计 v1/v2 PASS（variant-locus 级 vs GSE200304，0 重叠）。
4. **本行用途边界**：标签**只用于 frozen-Δ 评测**（与 contract 的 stress-test 定位一致：不训练、不选超参）。行内 provenance.sealed_history 按上述文档原文如实引用；评测报告如引用本行，须按 contract §2 汇报 historical_exposure_path，且禁用 "sealed/untouched/never-seen" 措辞。

## 1. 行单位与 QC

- 源：`GSE246381_hek_combined_umi_counts.csv.gz`（30,140 数据行 = **1,507 variant-keys × (10 REF + 10 ALT) barcode 行** + 1,350 `Control;Basal` 行 + 1,500 Shuf 行；对照行永不进入本评测行）。
- variant-key = (chrom, pos, ref, alt, Family=#, ENST)。REF/ALT 等位基因对天然构成 source/candidate 对（同 ENST 5'UTR 上下文的等位替换）。
- **QC 门槛（行进入条件）**：聚合（10 barcodes 求和）后 `DNA_ref + DNA_alt ≥ 10` 且 `(Poly_ref+Mono_ref) + (Poly_alt+Mono_alt) ≥ 10`。实测：**1,507/1,507 全过**（0 fail DNA、0 fail RNA）—— 因 10-barcode 聚合后计数普遍两位数以上。阈值（10/10）取任务要求区间（10-30）的下沿，理由：聚合口径下更严阈值（如 30）经抽查亦几乎不改变行集，取下沿保留统计功效；阈值已声明可复算。

## 2. Δy 口径（先定义后计算）

42 个 HEK 样本列按 SIC→fraction 映射分 7 组 × 6 reps（映射证据链：GEO GSM titles → SRA run 原始 fastq 文件名 `SIC####_HVF3VDSXY_...`，已固化 `benchmark_v2_rows/hek_sic_map.json`）：

| fraction | GEO title | 用途 |
|---|---|---|
| DNA（6 列） | HEK, Rep N, DNA | QC 基线（质粒输入对照） |
| Monosome（6 列） | HEK, Rep N, **80S_RNA** | Δy 分母 |
| Polysome（6 列） | HEK, Rep N, Polysome_RNA | Δy 分子 |
| TotalRNA/40S/TRAP/Plasmid（各 6 列） | — | 本行不使用（40S=43S PIC，TRAP=翻译核糖体亲和纯化，另属其他 endpoint 口径，未混入） |

**Δy = log2((POLY_alt + pc)/(MONO_alt + pc)) − log2((POLY_ref + pc)/(MONO_ref + pc))，pc = 1.0（UMI 计数伪计数，声明）**

- POLY/MONO = 6 个对应列、跨 10 barcodes 的 UMI 计数总和（生物学重复合并进入负载估计；per-rep 不单独成行，重复间方差不进入行内——如需 SE 可后续按 per-rep 扩展，本行不含）。
- 语义：ALT 的"多聚核糖体/单核糖体负载比"相对 REF 的对数变化（polysome translation efficiency 的 proxy；无剪接环境的报告基因体系）。endpoint_descriptor: quantity_family=TRANSLATION_EFFICIENCY, measurement_form=LOG2_FOLD, numerator=POLYSOME_RNA, denominator=MONOSOME_RNA, pseudocount=1.0。
- 实测：dy_mean ≈ **+0.0136**（近零，符合 5'UTR NDD 变异多数功能中性的文献预期）。

## 3. 序列窗口（S1 task_155 镜像）

- hg38（`route2/reference/hg38.fa.gz`）plus 链 [pos−78, pos+77) 0-based 闭区间（155nt，变异在 offset 77 居中）——镜像 S1 审计 task_155 提取规则；REF 窗口即 source_sequence，ALT = 窗口内 offset 77 处 ref→alt 替换。
- 等位方向：SeqID 无 strand 字段，等位按给定方向先试 plus；失败则试 revcomp(ref)/revcomp(alt)（minus fallback，行内 `allele_orientation` 字段记录）。实测 **1,507/1,507 plus-strand 直接匹配（minus fallback 0 次，0 行丢弃）**——说明表内 ref/alt 均为 plus 链记法（与 S1 的 strand 字段不同，如实记录）。
- hg38 为 soft-masked，提取统一 `.upper()`（首跑曾因此 100% allele mismatch，已修复）。

## 4. 重叠复审（本行构建时重跑的简版审计）

- **locus 级**：(chrom,pos) vs GSE200304 canonical ids（chr:pos）∪ GSE232572 sequence_id loci → 14,611 loci 池 → **0 hits**（复现 pigeonhole v2 结论）。
- **sequence 级**：1,507 × 2 条 155nt 窗口 vs 保守 frozen 池（GSE114002 + ENCSR854RUF + GSE200304 + GSE217518 + GSE186455 + GSE232572 全 canonical source/candidate 序列 + projection VALIDATION 序列，77,036 keys）→ **0 hits**。
- 结论：0 受保护重叠，无行因复审被排除。

## 5. VALIDATION 口径评测子集（冻结）

- QC+复审全过 1,507 行中抽样 **800 行**（seed **20260920**，`random.Random(20260920).sample`，脚本内嵌）；源组 = Family 标签（763 families；本行 509 个源组进入抽样集），密度 mean cand/源 = **1.57**（一个 family 含多 variant-key 时形成多行，如实保留——与 M1 的"文件内独立建组"纪律对应，M6 的组即论文的 Family）。
- Vglut 文件（小鼠在体，CreON/CreOFF 两态）**不进入本行**：在体神经元分馏只有 Monosome+Polysome 两 fraction 且 Cre 重组状态引入额外维度，Δy 口径与 HEK 不同质；留给后续独立行（row_definition 此处声明不越界）。

## 6. 字段 schema

同统一行 schema（row_id / study_unit / source_sequence / candidate_sequence / source_relative_edits / task_id / source_group_id / assay / biological_context / region / endpoint_descriptor / direction_normalized_delta / endpoint_direction / eval_split_status / tags / provenance），另含 `raw_counts{ref/alt × poly/mono/dna}`（可复算 Δy 的原始聚合计数）与 `variant_locus{chrom,pos,ref,alt,enst}`（坐标可追溯）。region=5UTR；biological_context=HEK_CELL_LINE。

**edit 表示约定**：等长变异（741 行）= 逐位 SUB（position/source_base/candidate_base）；长度变化变异（59 行，ref≠alt 长度的 deletion 型）= 单条 `LENGTH_CHANGE` 操作（首分歧位 position + source_segment/candidate_segment 尾段替换，镜像 GSE186455 canonical edit_operations 的 LENGTH_CHANGE 语义）——避免 zip 级联伪差异。构建脚本内嵌同款函数，复跑一致（已终验 0 mismatch）。

## 7. 纪律声明

源 csv.gz 与 hg38 只读；不写既有 manifest/审计产物；产物只增（`route2/benchmark_v2/m6_ndd5utr_eval_row/`）；protected reads = 0；脚本与文档不 commit；本行为评测行不进训练管线（并按封存史条款：其标签亦不可用于任何新训练/超参选择）。
