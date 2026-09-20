# DeltaBench 主论文骨架 v1（BENCHMARK_PAPER 主线 · Task 4.1，含 Task 4.2 攻击面自查）

- **版本**: v1（2026-09-20）
- **定位（PI 答复后定稿）**: benchmark 为第一主体交付（BENCHMARK_PAPER 主线），delta 可学性「系统性失效归因」为分析核心（spec pivot-delta-benchmark-mechanism v4 + amendment v1 生效；PI 三问用户直答：①benchmark 为主 ②机理归因型论文接受为最终交付 ③polyA 小论文叙事另议）。
- **本文件性质**: 章节级大纲 + 每节素材映射 + claim-证据表（数字与 locator 一一对应）；**不是全文撰写**。全文撰写为后续任务。
- **素材根（绝对路径）**: /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/（下文 locator 以 RT2/ 缩写）；W0 worktree 文档根 docs/paper/。
- **口径纪律**: 全部数字 = VALIDATION split、frozen-Δ 口径、预注册判定规则；protected TEST reads = 0；骨架中每个数字均标 locator；不确定/未落盘处如实标注，不编数。

---

## Title / Abstract 位

### 候选标题（3 个）

| # | 方向 | 标题 |
|---|---|---|
| T1 | benchmark 主体直陈 | DeltaBench: A Source-Relative Benchmark for Predicting the Effects of mRNA Sequence Edits |
| T2 | 矩阵 + 归因并列 | DeltaBench: 14 Model Families × 13 Tasks for mRNA Edit-Effect Deltas — and Why Absolute Scores Do Not Transfer |
| T3 | 归因副标题方向 | The Limits of Delta: A Source-Relative Benchmark and Systematic Failure-Attribution Analysis for mRNA Edit-Effect Prediction |

- 建议：T2 兼容量化主体（14×13）与归因钩子（绝对分不迁移）；T3 作备选（若 venue 偏 mechanism-track）。终选待 PI。

### Abstract 骨架（5 句式）

1. **问题**：mRNA 编辑优先级任务需要的是对变体间差异（Δ）的排序能力；我们给出教科书式脱钩证据——frozen Optimus 绝对任务 ρ_abs=0.873 而 ρ_delta=0.313，现有基准缺 source-relative 形式化，绝对分排行榜不测这一能力。
2. **交付**：DeltaBench——canonical record（source/candidate/direction_normalized_delta）+ 双任务（绝对 / frozen-Δ）+ 13 评测格（9 既有 VALIDATION 任务 + M1/M6/S1 三新行）+ 14 模型族 frozen-Δ 矩阵（65 新格 65/65 复核浮点级一致）。
3. **关键读数**：域内强跨库弱（UTR-STCNet MRL 域内 0.8144、跨库 M1 仅 0.0672；730/730 评测序列实测在其训练语料内）；新评测行上 5 新族集体 <0.09。
4. **归因（分析核心）**：三环——(i) 差分有效性五误差源 + oracle 臂（MRL NO_SIGNAL / polyA EQUIVALENT）；(ii) 两因子规律：密度单因子 held-out 检验 FAIL（50.77%<70%）如实降级为任务特异性现象，双因子分组（EXPLORATORY）显示偏离按监督体制系统化分裂；合成提密度干预证伪（G1 FAIL gap 0.8235）、参数侧半效（ERK 0.2015 贴一阶上界 0.207）、真实数据阶梯完全有效（0.1987→0.3158≈0.3132）；(iii) 一阶分解 + 四类失效分类。
5. **实践出口**：模型选择分类表（C1 物理 / C2 范式 / C3 几何 / C4 监督 / C5 对照体制）+ 数据侧三体检（ICC / 密度 / 语料）先于模型选型。

- 措辞注意：Abstract 句 3「集体 <0.09」仅指 5 新族 × 3 新行格，不得外推为全矩阵；句 4 FAIL 如实、不隐藏。

---

## §1 Introduction

### 小节大纲

- **1.1 编辑优先级任务与 Δ 排序需求**：变异/编辑效应预测的下游用例（变异优先级排序、生成线引导）真正消费的是 Δy 的秩；引子案例 Optimus MRL（0.873 abs / 0.313 delta / 0.720 ρ_ε 三元组）——「能读绝对值」与「能读差分」是两种能力。
- **1.2 现有基准的缺口**：绝对端点 leaderboard 不区分两能力；无 source-relative 形式化（delta_hat = pred(cand) − pred(src)、canonical record、source_group）；无监督体制 × 数据几何的对照设计（域内监督行与域外行混排导致读数误读——STCNet 0.8144 的语料级记忆属性在既有榜单上不可见）。
- **1.3 贡献列表（4 项）**：
  - C-a **Benchmark v2 定义**：canonical record / 双任务 / 13 任务 / frozen-Δ 评测协议（两级判定纪律 + 天花板归一化）。
  - C-b **14 族 × 13 格矩阵**：5 新族移植（单测 PASS + 端口适配全声明）+ 9 既有族引用行；65/65 独立复核浮点级一致。
  - C-c **三环归因框架**：R0 差分有效性（五误差源 + oracle 臂）→ 两因子规律（观测 + held-out + 干预三角）→ 一阶分解 + 四类失效分类。
  - C-d **双因子规律**：ρ ≈ f(监督体制 × 数据几何)（EXPLORATORY 支撑），密度单因子已按预注册降级条款如实降级。
- **1.4 论文结构图**（一段）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 1.1 | Optimus 三元组 | RT2/experiments/analysis_delta_validity_v1/results_error_correlation.json → rows[0] |
| 1.2 | STCNet 语料重叠实测 | RT2/experiments/analysis_delta_vs_density_v2/summary.md §4.2 |
| 1.3 C-b | 矩阵 + 复核 | RT2/benchmark_v2/leaderboard_matrix_v2/matrix_v2_summary.md + analysis_matrix_recheck_v1/recheck_results.json |
| 1.3 C-c | R0/两因子/分解/分类 | 素材 3/2/6/4/5（见各节） |

### claim-证据表

| # | claim（骨架级） | 证据数字 | locator |
|---|---|---|---|
| 1.1 | 绝对能力与差分能力脱钩的教科书案例 | 0.8733 / 0.3132 / 0.7196 | results_error_correlation.json → rows[0] |
| 1.2 | 域内高分行有语料绑定属性（引子） | STCNet MRL 0.8144；730/730 重叠 | matrix_v2_summary.md；analysis_delta_vs_density_v2/summary.md §4.2 |
| 1.3 | 新族在新行上集体近零 | M1/M6/S1 全部格 <0.09（最大 GEMORNA-M1 0.0889） | matrix_v2_summary.md（M1/M6/S1 各 5 行） |

---

## §2 The Benchmark

### 小节大纲

- **2.1 定义与 canonical record**：行单位 = (source, candidate) 配对 record，字段 schema 14 件套（source_sequence / candidate_sequence / source_relative_edits / direction_normalized_delta / source_group_id / endpoint_descriptor / biological_context / eval_split_status / provenance 等）；Δy = cand − source（LEVEL_DIFFERENCE 口径）；**双任务声明**：绝对任务（ρ_abs）与 frozen-Δ 任务（ρ_delta）分列报告，同一评测面上可同时计算。
- **2.2 13 任务表（9 既有 + 3 新行，S1 双臂）**：
  - 9 既有 VALIDATION：MRL(GSE114002, n=730) / polyA(GSE269595, n=2628) / MPRAU(ENCSR854RUF, n=12048) / HL5(GSE217518, n=400) / HL3(同库, n=503) / TE200304(n=1614) / TE149487(n=48) / RNA149487(n=48) / REFALT(GSE186455, n=274)。
  - 3 新行（append-only，seed 冻结 20260920）：**M1** MRL 行（GSE232927 Castillo-Hair 2024；2,805 行/2,801 源/3 细胞上下文；密度 1.0014）；**M6** NDD 5UTR 行（GSE246381 Plassmeyer 2025；800 行/509 Family 源；封存-解封史入 provenance，historical_exposure 按契约汇报）；**S1** 稳定性行（Su 2025 eLife；5,572 sub-row SH 2,792/HEK 2,780；受保护排除 1,330；Dao cryptic QC flag-keep 834/1,871）。
- **2.3 四轴差异表（竞品对比 Table 1 底稿更新）**：四轴 = ①source-relative 形式化（Δ 定义在 record 层）②双任务报告（abs/Δ 分列）③天花板归一化（ICC 上界 + N/A 处理）④两级判定纪律（tier-1 小样校准否证权 + tier-2 全量判定）+ 泄漏/权利边界列。底稿 = SPECS_BASELINE_LEADERBOARD BENCHMARK_GUIDE 四轴差异（引用不改）。
- **2.4 评测协议**：
  - frozen-Δ 口径：官方权重零调参，端口适配逐族全声明（GEMORNA 5/3 双头、HydraRNA 冻结嵌入首坐标读出、STCNet padd120 one-hot、Insight 50nt pad 前缀 + GSM3130435 分支、LAMAR EsmTokenizer 1026 截断）。
  - 两级判定纪律：tier-1（小样校准集，如 polyA calib100）发现信号须被 tier-2 全量（891 源）复核——Optimus calib100 信号被 891 否证的纪律案例。
  - 天花板归一化：ICC 任务上界（MRL 0.83 / polyA 0.90 / MPRAU 0.683 / TE200304 0.586 / HL ICC≈0.001-0.013）；ICC≈0 行归一化 N/A（不用负数分母制造放大假读数）。
- **2.5 新行构建纪律**：只增行（不进既有 manifest）、种子冻结、鸽笼泄漏审计（M1/M6/S1 全 0 受保护重叠）、TRAIN 重叠只标记不排除（M1 4 条 keep=0 天然出局）、极端值原样保留（S1 t05 异常量级，建议稳健统计）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 2.1/2.2 既有 9 任务 | 榜单冻结表 | RT2/experiments/task6_leaderboard_freeze_20260903/TASK6_leaderboard_freeze_20260903.md §1 |
| 2.2 M1 | 行定义 | RT2/benchmark_v2/m1_mrl_eval_row/row_definition.md |
| 2.2 M6 | 行定义（含封存-解封史） | RT2/benchmark_v2/m6_ndd5utr_eval_row/row_definition.md §0 |
| 2.2 S1 | 行定义（Dao QC + 受保护排除） | RT2/benchmark_v2/s1_stability_eval_row/row_definition.md §2-3 |
| 2.3 四轴 | BENCHMARK_GUIDE（引用不改） | SPECS_BASELINE_LEADERBOARD/BENCHMARK_GUIDE.md（journal 批次 143 行入册；§9 全表同源） |
| 2.4 端口声明 | 矩阵 summary 逐族块 | RT2/benchmark_v2/leaderboard_matrix_v2/matrix_v2_summary.md（Per-family adaptation declarations） |
| 2.4 两级纪律 | COMB 终表 tier-1 案例 | 素材 7（journal 批次 104/105：calib100 Optimus 信号被 891 否证） |
| 2.5 | 新行三份 row_definition | 同 2.2 三文件 |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 2.1 | 13 格 = 9 既有 + 3 新行（S1 双臂） | grid_note 原文 | RT2/benchmark_v2_matrix/matrix_summary_v1.json → counts/grid_note |
| 2.2 | M1 规模与密度 | 2,805 行 / 2,801 源 / 密度 1.0014 / HEPG2 800 + T_CELL 1600 + HEK293T 405 | m1_mrl_eval_row/row_definition.md §2/§4 |
| 2.2 | M6 规模 | 800 行 / 509 源 / 1,507 variant 全过 QC / 0 重叠 | m6_ndd5utr_eval_row/row_definition.md §1/§4 |
| 2.2 | S1 规模与排除 | 5,572 sub-row / 1,519 源 / 密度 3.67 / 排除 1,330 / flagged 834 | s1_stability_eval_row/row_definition.md §3/§6 |
| 2.4 | 单测 5 族全 PASS | family_unit_tests.json | RT2/benchmark_v2/leaderboard_matrix_v2/family_unit_tests.json |
| 2.5 | 泄漏审计 0 | M1 4 条 TRAIN 重叠 keep=0；M6 locus/sequence 双级 0 hits；S1 1,330 硬排除后 0 | 三份 row_definition |

---

## §3 Results: the matrix

### 小节大纲

- **3.1 主表（14 族 × 13 格）**：5 新族实测矩阵表（任务宏 Spearman on VALIDATION delta）整表灌入；9 既有族按 matrix_summary_v1.json ALREADY_DONE 引用行（引用不改）。主表数字 = matrix_v2_summary.md 表格 65 格 + 既有族冻结行。
- **3.2 复核声明（一段）**：独立复核脚本不读入档指标、从 predictions.jsonl（137,350 行）逐格重算，65/65 PASS，全矩阵 max |Δ| = 0.0（浮点级精确一致，含 STCNet MRL 0.814430506981109 极端格）。
- **3.3 关键读数一：域内强跨库弱**：STCNet MRL（GSE114002）0.8144 vs 同族 M1（GSE232927）0.0672——语料同为 MRL 域，跨库即回落到通用族水平；730/730 评测序列实测在其 MPRA-H 训练语料内（667 train_val + 63 test）→ 域内监督收益部分绑定语料级记忆。同库弱监督对照：UTR-Insight（随机 50nt 文库）同格仅 0.2739。
- **3.4 关键读数二：新行集体近零**：M1/M6/S1 全部格最大 0.0889（GEMORNA-M1）；无同分布稠密监督的模型族在新行上集体受限——对照行（frozen-Optimus MRL 0.3132 域内参考）以监督体制解释，非「全部 baseline 失效」（措辞冻结条款）。
- **3.5 按族汇总**：STCNet 4/13 in-band（两极偏离）；其余族 7-8/13（素材 2 §3 逐格计数，作为 §5 的数据桥）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 3.1 主表 | 矩阵 65 格 | RT2/benchmark_v2/leaderboard_matrix_v2/matrix_v2_summary.md（Matrix 表） |
| 3.1 既有族 | 9 族 ALREADY_DONE 归档行 | RT2/benchmark_v2_matrix/matrix_summary_v1.json → existing_families |
| 3.2 | 复核 | RT2/benchmark_v2/leaderboard_matrix_v2/analysis_matrix_recheck_v1/recheck_results.json |
| 3.3 | STCNet 两读数 | 0.8144 / 0.0672 / 730/730 | matrix_v2_summary.md；analysis_delta_vs_density_v2/summary.md §4.2 |
| 3.4 | 新行近零 | M1 5 行（0.0226/-0.0215/0.0889/0.0672/0.0575）；M6 5 行；S1 两臂各 5 行 | matrix_v2_summary.md |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 3.2 | 65/65 浮点级一致 | 65 OK；max abs_delta = 0.0 | recheck_results.json |
| 3.3 | 域内强 | 0.8144（n=730） | matrix_v2_summary.md |
| 3.3 | 跨库弱 | 0.0672（n=2805） | 同上 |
| 3.3 | 语料重叠 | 730/730（667 train_val + 63 test） | analysis_delta_vs_density_v2/summary.md §4.2 |
| 3.3 | 同库弱监督对照 | UTR-Insight 0.2739 | matrix_v2_summary.md |
| 3.4 | 新行近零 | M1 最大 0.0889；M6 最大 0.0464；S1 最大 0.0450 | matrix_v2_summary.md |
| 3.4 | 域内参考对照 | frozen-Optimus MRL 0.3132（v1 reference） | matrix_v2_summary.md（Key readouts 块） |

- 措辞注意（§3 全节）：禁写「所有 baseline 都无法预测 delta」；写「无同分布稠密监督的模型族集体受限，Optimus/APARENT 为监督体制对照行」。

---

## §4 Why absolute scores do not transfer（R0 差分有效性 · 素材 3）

### 小节大纲

- **4.1 数学分解（半节）**：Δŷ − Δy = ε_c − ε_s（偏置精确相消；线性读出下 Δŷ = w·Δh）；差分误差 = 两臂误差之差 → 有利条件（ρ_ε→1 抵消）/灾难条件（ρ_ε→0 或误差差带无关结构）。
- **4.2 五误差源实证**（每源：机制 + 定量 + locator）：
  1. 误差相关三元组：Optimus 0.873/0.313/0.720（部分抵消、不够）；LAMAR 全盲行（ρ_ε 0.686/0.443 而无信号 = 误差共享非信号）。
  2. 组内 vs 整体口径分叉：polyA 整体 0.71-0.75 vs 组内 0.19-0.23（源间成分虚高；编辑优先级需要组内口径）。
  3. probe 几何「宽而浅」：cos 对齐 0.79/0.81 但 RNA-FM PR 258.7 / top-10 9.6% / 每维增益 0.019（SNR proxy 标注各向同性假设；16 行未存档 probe 权重只登记不编数）。
  4. oracle 臂（决定性）：MRL NO_SIGNAL（RNA-FM 0.0869 / UTR-LM 0.0700 三 seed 均值；per-seed 0.1370/0.0640/0.0597 全列——seed 敏感伪信号如实呈现）→ 表征不含 delta 信号、换监督形式无效；polyA HEAD/EQUIVALENT（+0.0070/−0.0054）→ 信号真实存在于表征且已兑现。
  5. 上下文失配 + 标签噪声：RiboNN 四行近零（−0.0106/−0.0290/+0.0778/−0.0450，全长 TE 模型 vs 片段）；GSE149487-TE ICC −0.274 / split-half 0.050（数据侧下界封死）；GSE200304 反例（ICC 0.586 可测而模型无信号 → 表征侧）。
- **4.3 协议发现（如实披露段）**：既有 frozen-Δ LM 行实现代码本身即 Δh→Δy 回归，与部分 docstring 不一致；线性读出下两监督形式数学等价、不影响存档数值；已入 protocol note（results_oracle_probe.json → supervision_form_audit_note）。
- **4.4 操作性边界（收束半节）**：5 有效条件 / 5 失效形态 / 3 操作判据（组内口径 + ρ_ε 三元组先行；ICC 与上下文匹配声明先行；绝对分高不得外推差分可用）。

### 素材映射

整章 = 素材 3 RT2/experiments/analysis_delta_validity_v1/section_draft_why_abs_not_delta.md（10 节结构，骨架 §4.1-4.4 对应该底稿 §2/§3-§7/§6 披露段/§8）。

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 4.2-1 | 三元组脱钩 | 0.8733 / 0.3132 / 0.7196 | results_error_correlation.json → rows[0] |
| 4.2-1 | 全盲行误差共享 | LAMAR HL5 0.0207/0.0441/0.6863；HL3 −0.0911/0.0184/0.4429 | 同上 rows[2-3] |
| 4.2-2 | 口径分叉 | UTR-LM 0.7490→组内 0.1934；RNA-FM 0.7114→组内 0.2335（78 组） | 同上（polya 两行） |
| 4.2-3 | 宽而浅 | RNA-FM PR 258.7 / top-10 0.0955 / cos 0.8102 / 每维 0.0189 | results_projection_decoding.json → archived_probe_weights.RNA-FM |
| 4.2-4 | MRL NO_SIGNAL | RNA-FM 均值 0.0869（per-seed 0.1369/0.0640/0.0597）；UTR-LM 0.0700 | oracle_probe/results_oracle_probe.json → results.gse114002_mrl |
| 4.2-4 | polyA EQUIVALENT | RNA-FM +0.0070 / UTR-LM −0.0054（HEAD/EQUIVALENT） | 同上 results.polya_gse269595 |
| 4.2-5 | 上下文失配 | RiboNN −0.0106/−0.0290/+0.0778/−0.0450 | results_context_mismatch.json → ribonn_rows |
| 4.2-5 | 标签下界 | ICC −0.274 / split-half 0.050 | results_label_noise.json → table[0] |
| 4.2-5 | 反例（表征侧） | GSE200304 ICC 0.586，最强 Δ ≤0.011 | 同上 |
| 4.2 复核 | ρ_delta 重算一致 | 30/30 行 |Δ|≤1.1e-16 | recheck_rho_delta_consistency.json |

---

## §5 The two-factor regularity（素材 2 + 6）

### 小节大纲

- **5.1 观测（v1 原始观测）**：25 外部行 × 9 任务族，log10(density) vs ρ_delta：Pearson r=0.9388（p=3.9e-12）；散点图（25 灰点 + 冻结拟合线）；读法：≥0.68 的 5 行全在 polyA（密度 126.7 唯一离群），密度 ≤6.1 的 20 行全 ≤0.32——相关是分层的。
- **5.2 held-out 检验（诚实降级如实写）**：冻结预注册框架（容差带 |Δρ|≤0.10、PASS 门 ≥70%、禁挑行/只增行/禁改带）；65 新格一律只预测不回灌；结果 in-band 33/65 = **50.77% < 70% → verdict FAIL**；降级条款触发执行：「密度单因子规律」降级为**任务特异性现象**，此后禁用「跨任务普适规律」表述，唯一升级路径 = 只增行 + 下一判定点重评。**本小节为如实负结果，是 §5 的纪律锚**。
- **5.3 双因子分组读数（EXPLORATORY，非判定门）**：按「评测任务 endpoint 域是否在模型训练语料内」分组——below-band 21 格 **0 格域内**（polyA 5 族全低于带：冻结预测 ρ≈0.68 vs 实测 −0.37~+0.31）；above-band 11 格中 **6 格域内**（STCNet-MRL 0.8144 超带 +0.65；M1 四个 MRL 域内族托底）；域内 3/9=33.3% vs 域外 30/56=53.6%。**同密度对照**：密度 4.9 下域内稠密监督 0.81 vs 域外 −0.02；密度 126.7 下 polyA 域内 0.68-0.75（旧行）vs 域外 0.15-0.31（新行）。结论（EXPLORATORY 标注）：偏离按监督体制系统化分组，ρ = f(监督体制 × 数据几何)；25 行拟合集 r=0.9388 系两因子共变。
- **5.4 干预链（观测-干预三角）**：
  - 合成侧证伪：D16-C 合成提密度，G1 FAIL（gap 0.8235 ≥ 基线 0.7943）；G2 PASS（polyA 0.81338 非破坏）；G4 sc-hit1=0（81/81 support、0 命中）→「提密度即可修复」被证伪（措辞冻结条款）。
  - 参数侧半效：ERK v2 显式一阶先验，gap 0.7943→0.3815 减半、val 0.2015 = MRL 历史最强 ≈ E7 一阶+上下文闭式上界 0.207（贴界锚点：增益可被一阶先验解释 = 收缩源残差非新判别信号）；G4=0（231/231 support、exact match 1）；G3 未触发（partial-path）。
  - 真实数据侧完全有效：W 阶梯 0.1987 → 0.2470（LoRA）→ 0.2555（full-FT 2ep）→ 0.3158（3-seed）≈ frozen-Optimus 0.3132（统计平局 Δ+0.0027 CI[−0.045,+0.048] 跨零——**平局措辞条款：禁写 no difference，写统计平局、点估计领先、730-record 检验力不足不宣称显著超越**）。
  - 三角合并：密度相关的因果载体 = 真实监督数据的几何覆盖（H-geometry），排除 H-density（合成证伪），H-selection 保留为边界条件（polyA 机制局部性）。
- **5.5 G4 三代零信号链**：D15-2 polyA V5 → D16-C probe → ERK v2 三代 graded sc-hit@1 = 0（支持率 100% 命中 0 的稳定形态）→ 结构排序能力层归因：MRL rank 信号由源均值主导。
- **5.6 M1 干预臂结果位（空槽）**：

  > **[SLOT — 待 xeditcritic_m1_intervention 终态回填，本骨架不写数字]**
  > 预注册四门（docs/paper/m1_intervention_arm_amendment_v1.md，发射前冻结 commit b2e141cd）：G1 方向门（vs 3-seed ens 0.3158 基线，单 seed 增益 > +0.01 判方向正）/ G2 非破坏门（polyA 0.8219 口径回落 ≤ 0.02）/ G3 效率门 / G4 机理门（M1 评测行方向性对照，与 STCNet 域内/跨库现象对照）。解释框架预登记：增益 FAIL 也是有效交付（密度因子独立贡献受限的干预证据）。训练臂状态 = WAITING/排队（seed 20260920），收割由后续会话终态执行。

- **5.7 小节收束**：两因子主张的证据等级声明——密度单因子：已降级（held-out FAIL 如实）；双因子：观测 + 干预三角 + held-out 偏离模式三线支持，EXPLORATORY 等级如实标注。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 5.1 | v1 观测 | RT2/experiments/analysis_delta_vs_density_20260915/delta_vs_density_data.json + 散点图 |
| 5.2/5.3 | held-out v2 | RT2/experiments/analysis_delta_vs_density_v2/{summary.md, delta_vs_density_v2_results.json, delta_vs_density_v2_cells.json, 散点图} |
| 5.4 干预三角 | 机理链 A | RT2/experiments/analysis_mechanism_chain_a_v1/mechanism_chain_a_draft.md §2-4 |
| 5.4 D16-C | adjudication | RT2/experiments/xeditcritic_d16c/probe_mrl_v1_gpu5/{adjudication_d16c.json, gap_backtest_d16c.json} |
| 5.4 ERK | adjudication | RT2/experiments/xeditcritic_erk_v2/erk_train_seed2026091901/{erk_adjudication_v1.json, gap_backtest_erk_v2.json} |
| 5.4 W 阶梯 | 终判表 | RT2/experiments/analysis_task8_bottomline_20260909/bottomline_adjudication_v1.json + xeditcritic_route_a/280k_prefinetune_20260903/frozen_delta_results.json |
| 5.5 | G4 链 | 三份 structure_probe_g4 JSON（d16c/erk + 基线） |
| 5.6 | M1 空槽 | docs/paper/m1_intervention_arm_amendment_v1.md（W0）+ RT2/experiments/xeditcritic_m1_intervention/seed_20260920/heartbeat.json |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 5.1 | 原始强相关 | r=0.9388，p=3.9e-12，n=25 | delta_vs_density_data.json → stats |
| 5.2 | held-out FAIL | 33/65 = 50.77% < 70% 门；verdict FAIL；downgrade_triggered=true | delta_vs_density_v2_results.json → heldout |
| 5.2 | 冻结拟合参数 | 斜率 0.3663/decade，截距 −0.0857 | 同上 → fit |
| 5.3 | below-band 0 域内 | 21 格全域外 | 同上 → exploratory_domain_grouping.deviation_direction_decomposition |
| 5.3 | above-band 6/11 域内 | STCNet +0.647 等 | 同上 |
| 5.3 | 分组率 | 域内 3/9=33.3% vs 域外 30/56=53.6% | 同上 → in_domain/out_of_domain |
| 5.3 | 同密度对照 | MRL 4.9：0.81 vs −0.02；polyA 126.7：域内 0.68-0.75 vs 域外 0.15-0.31 | analysis_delta_vs_density_v2/summary.md §4.2 |
| 5.4 | 合成证伪 | D16-C gap 0.8235（G1 FAIL）；G2 polyA 0.81338 | adjudication_d16c.json + gap_backtest_d16c.json |
| 5.4 | 参数半效 | ERK gap 0.3815 / val 0.2015 / G4=0 / G3 未触发 | erk_adjudication_v1.json + gap_backtest_erk_v2.json |
| 5.4 | 上界贴界 | 0.2015 ≈ E7 0.2069 | analysis_erk_e7_probes/e7_v2_probe_results.json → probes.MRL__E7a_v2 |
| 5.4 | W 阶梯 | 0.1987 / 0.2470 / 0.2555 / 0.3158 / 0.3132 | bottomline_adjudication_v1.json + 280k_prefinetune_20260903/frozen_delta_results.json |
| 5.4 | 平局口径 | Δ+0.0027 CI[−0.045,+0.048] 跨零 | analysis_fullft_v2_adjudication_20260903/ensemble_3seed_vs_optimus.json |
| 5.5 | 三代零 | D15-2 0.0 / D16-C 0.0（81/81）/ ERK 0.0（231/231） | 三份 structure_probe JSON |
| 5.6 | 空槽不写数字 | — | amendment 冻结 commit b2e141cd |

- 措辞注意（§5 全节）：密度单因子 = 降级后措辞（「在已观测 25 外部行集合内成立的密度-可学性现象；65 格 held-out 判 FAIL（50.77%<70%），已按预注册降级条款降级为任务特异性现象」）；双因子主张标注 EXPLORATORY 支撑；禁写「提密度即可修复」；W 阶梯终点平局禁写 no difference。

---

## §6 Signal decomposition（一阶/结构分解 · 素材 4）

### 小节大纲

- **6.1 一阶闭式能量表口径（方法段）**：E7 复刻（pos-block × 64 上下文组合 + per-source 背景 b[source]，Ridge 闭式，alpha 网格 VALIDATION 选优）；MRL 参照对（0.2069 vs ERK 0.2015）维持既有档案。
- **6.2 逐任务结果三表**：polyA 一阶 0.4555（=V5 的 55.4%；对照：无稠密监督外部带 0.71-0.75、APARENT 0.7343、V5 0.8219、ICC 0.90）；MPRAU 一阶 0.0825（pair-mean；=V5 的 80.5%；外部带 ≈0.016 CI 跨零；Saluki 弱对照 0.1205；ICC 0.683）；TE 一阶 0.0458（=V5 的 79.1%；外部带 0.0009-0.0113）。
- **6.3 任务异质性（主结论）**：分解叙事不可跨任务外推——polyA 一阶只解释 0.46（0.27+ 属非一阶/结构信号，外部 LM 已拿到）；MPRAU/TE 我方行几乎就是一阶信号（V5 增量仅 +0.02/+0.01），「结构/高阶增益」主张在这些任务上要克制表述；与 G4 结构盲、delta_validity 组内口径发现互补。
- **6.4 外部行的双面读法（如实报告）**：「无稠密监督」≠「无信号」——polyA 外部行 0.71-0.75 普遍高于一阶上限（通用 LM 预训练迁移已超过一阶表）；MPRAU/TE 外部行 ≈0（连一阶信号都拿不到）。任务编辑密度（polyA 平均 8.5 edits/record 唯一有组合空间）与一阶覆盖缺口一致。
- **6.5 判定总表（预注册规则）**：A=|ρ₁−带中位|≤0.05；B=稠密行−ρ₁>0.05：polyA 部分成立（✗A 0.2747 / ✓B 0.2788）；MPRAU 不成立（一阶超带 0.0661）；TE 部分成立（B 不适用）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 6.1-6.5 | 一阶分解 summary | RT2/experiments/analysis_first_order_decomposition_v1/summary.md（含三表 + 判定总表） |
| 6.2 对照行 | polyA 外部带 / Saluki | analysis_delta_vs_density_20260915/delta_vs_density_data.json → rows（polyA 5 行）+ analysis_saluki_frozen_mprau_20260903/frozen_delta_results.json:mprau_pair_mean |
| 6.4 | 与 G4/组内口径互补 | 素材 6 §5 + 素材 3 §4 |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 6.2 | polyA 一阶 | 0.4555 = V5 0.8219 的 55.4% | analysis_first_order_decomposition_v1/summary.md §2 polyA 表 |
| 6.2 | MPRAU 一阶 | 0.0825 = V5 0.1025 的 80.5%（record 级 0.0630） | 同上 §2 MPRAU 表 |
| 6.2 | TE 一阶 | 0.0458 = V5 0.0579 的 79.1% | 同上 §2 TE 表 |
| 6.2 | 外部带 | polyA 0.71-0.75 / MPRAU 0.0164 / TE 0.0061 | 同上（各表对照行） |
| 6.3 | 任务异质性 | polyA 缺口 0.2747 vs MPRAU 增量 0.0200 / TE 增量 0.0121 | 同上 §3 判定总表 |
| 6.4 | 外部行双面 | polyA 外部 5 行 ≥0.681；MPRAU/TE 外部 CI 跨零 | delta_vs_density_data.json → rows |
| 6.1 | MRL 参照对（既有档案） | 一阶上下文 0.2069 / ERK 0.2015 | e7_v2_probe_results.json → probes.MRL__E7a_v2 + erk_train_summary.json |

- 措辞注意（§6）：MPRAU/TE 上「我方行≈一阶信号」如实写（结构增益主张克制）；polyA 结构信号空间最大（V5 距 ICC 0.90 仅 0.078）不得外推到其他任务。

---

## §7 A taxonomy of failure（素材 5）

### 小节大纲

- **7.1 分类规则（先写规则后分类，半节表）**：C1 物理失败（标签 ICC<0.1 且全行噪声带）/ C2 范式失败（输入分布/上下文失配 + 全行 |ρ|<0.05 带）/ C3 几何失败（密度<10 + 结构证据支持）/ C4 监督失败（无同分布语料 + oracle NO_SIGNAL 或同池全零）/ C5 非失败对照组（稠密监督体制行 ≥280K）。
- **7.2 主分类表（20 行 = 失效 14 + 对照 6，覆盖 9/9 任务）**：C1 5 行（HL5/HL3/TE149487 + LAMAR 交叉证据 2 行）；C2 5 行（RiboNN 3 + Saluki 2）；C3 2 行（MRL/MPRAU）；C4 5 行（MRL-oracle/MPRAU/TE/RNA149487 边缘/REFALT 边缘）；C5 6 行（Optimus/我方 3-seed/APARENT/APARENT2/V5/通用 LM polyA 行）。
- **7.3 任务级汇总 + 跨类标注**：主类+次类分层（MRL = C3 主 + C4 次：几何先行、表征无信号第二约束）；边缘 case 三决定（GSE149487-RNA → C4 主 C1 次 power-limited；GSE186455 → C4；LAMAR → C1 + ρ_ε 交叉证据）。
- **7.4 对照组叙事职能（防一边倒段）**：同一套通用 LM 在 polyA（密度 126.7、2.74M 语料、ICC 0.9）上 0.71-0.75 与 oracle EQUIVALENT，在低密度无同分布语料任务上全面跨零——失效归因在数据体制，不在「通用 LM 不行」类断言（措辞冻结条款落点）。
- **7.5 实践启示（模型选择指南的雏形，通向 §9）**：C2 → 输入粒度对齐；C3 → within-source 稠密候选监督 + pair-mean 口径；C4 → oracle 探针先立项止损 / 等语料 / 端到端 FT（MPRAU matched-FT 三 seed 全负 = FT 不保救）；C5 → 先找语料再谈架构。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 7.1-7.5 | 失效分类表全文 | RT2/experiments/analysis_failure_taxonomy_v1/failure_taxonomy_v1.md（+ 同名 JSON 结构化版） |
| 7.2 | 20 行主表 | 同上 §1（C1-C5 五表） |
| 7.3 | 边缘决定 | 同上 §1 末三段 |
| 7.4 | 对照职能 | 同上 §1 C5 表后叙事段 |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 7.1 | C1 判定阈值 | HL5 ICC 0.0014 / HL3 0.013 / TE149487 −0.274 | mechanism_results_v2.json:label_icc_reference + analysis_ceiling_icc_20260907/ceiling_icc_results.json |
| 7.2 | C2 代表行 | RiboNN −0.0106/−0.0290/+0.0778/−0.0450；Saluki MPRAU 0.1205（weak-control CI[0.077,0.164]） | analysis_ribonn_frozen_te_20260913/ + analysis_saluki_frozen_mprau_20260903/ |
| 7.2 | C3 双行 | MRL 密度 4.9 + G4 三代 0 + E7 0.207；MPRAU 6.1 + CI 跨零 | delta_vs_density_data.json:density_table + e7_v2_probe_results.json |
| 7.2 | C4 oracle 行 | NO_SIGNAL 0.0869/0.0700 | oracle_probe/results_oracle_probe.json |
| 7.2 | C5 对照行 | Optimus 0.3132 / ens 0.3158 / APARENT 0.7343 / V5 0.8219 / 通用 LM polyA 0.7114-0.7490 | failure_taxonomy_v1.md §1 C5 表（各行 locator 已内标） |
| 7.4 | 同套 LM 反差 | RNA-FM：polyA 0.7114 vs MRL 0.1369（seed 敏感）vs HL ≈0 | failure_taxonomy_v1.md（C5 行 23） |
| 7.5 | matched-FT 全负 | MPRAU 三 seed pair-mean −0.075~−0.107 | analysis_mprau_matched_ft_external_20260904{,_seed20260911,_seed20260915}/ |

- 措辞注意（§7）：失效行与对照行并列；Saluki 0.1205 正读数保留 + weak-control 口径标注；RNA-FM 0.2958 / 0.1043 两个「唯一高行」保留 + 族间分歧 + power-limited 标注；天花板归一化 ICC≈0/负值行 N/A。

---

## §8 Case study: polyA（收缩版 · 素材 7）

### 小节大纲（收缩边界：只收 91% 完成度 + 生成线应用一节 COMB 可加性；polyA 机制全章留位不展开——小论文另议）

- **8.1 polyA 作为机理链边界端点（半节）**：密度 126.7 唯一数量级离群；5 frozen 外部行 0.681-0.749 与 V5 内部 0.8219 同带；V5 距 ICC 0.90 仅 0.078（完成度 91%）；一阶闭式先验 0.488 起点（E7-a v1）/ 0.4555（一阶-only 扩展）——机制可加局部性解释 frozen 高判别；MRL/MPRAU 一阶 0.159/0.041 反衬。
- **8.2 生成线应用一节：COMB 可加性（B+C≈D）**：2×2 干预（explore B / guide C / 双臂 D）三 seed 终表——sc calibre 近可加指纹 B+C +0.02974 vs D +0.02972；verdict H_IND_or_NEGATIVE（H-int 不显著）；探索增益全保留、guide 非破坏（D~C 模式）；双口径镜像分歧现象学（C 正于 sc/recovery、B 正于 optimus）；polyA 报告行（D_main +0.3527 / seed16 +0.4210 / seed17 +0.4844）。
- **8.3 两级判定纪律案例（半节，挂 §2.4 协议的实例）**：tier-1 calib100 上 Optimus 信号被 tier-2 891 源全量否证——tier-1 只有否证权、无确认权（判定纪律的可操作展示）。
- **8.4 留位声明（一段）**：polyA 机制章（V5 0.8219 显著胜 APARENT Δ+0.088 CI[0.063,0.114]、Holm p=0.004；决策口径 top-1 混合结果如实——Δ−0.100 CI[−0.143,−0.058] 显著负）在主论文只留位置与一句话结论，**完整机制叙事为 polyA 独立小论文（PI 三问 ③，叙事讨论中）**。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 8.1 | 边界端点 | 素材 6 analysis_mechanism_chain_a_v1/mechanism_chain_a_draft.md §6 + 素材 4（一阶） |
| 8.2 | COMB 终表 | 素材 7：journal 批次 103-105（COMB 2×2 891 终表三 seed；B+C +0.02974 vs D +0.02972；H_IND_or_NEGATIVE；polyA 报告行三数）；RT2/experiments/xeditsetflow_v5/（COMB tier-2 产物树） |
| 8.2 | 生成线基线 | TASK6_leaderboard_freeze_20260903.md §2（unguided 0.12046 / Gate B2-B3 预注册带） |
| 8.3 | 两级纪律 | 素材 7 journal 批次 104/105（tier-1 calib100 Optimus 被 891 否证） |
| 8.4 | polyA 机制结论 | analysis_task1_alignment_20260902/task1_alignment_results.json（paired_bootstrap）+ analysis_aparent2_frozen_delta_20260906/result.json |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 8.1 | 91% 完成度 | V5 0.8219 / ICC 0.90（缺口 0.078） | v5_full/run_summary.json + failure_taxonomy_v1.md C5 行 22 |
| 8.1 | frozen 同带 | 5 行 0.681-0.749 | delta_vs_density_data.json → rows（polyA 块） |
| 8.1 | 一阶起点 | 0.488（E7-a v1）/ 0.4555（一阶-only） | e7_probe_results.json + analysis_first_order_decomposition_v1/ |
| 8.2 | 近可加指纹 | B+C +0.02974 vs D +0.02972（sc calibre） | COMB 终表（journal 批次 105 存档） |
| 8.2 | H_IND verdict | H-int 不显著；sc gate 1/3 臂、optimus 0/3 | 同上（批次 105） |
| 8.2 | polyA 报告行 | D_main +0.3527 / seed16 +0.4210 / seed17 +0.4844 | 同上 |
| 8.2 | 生成线基线 | unguided 0.12046（891 源，28,512 候选，unique 0.8572/legality 1.0） | TASK6_leaderboard_freeze_20260903.md §2 |
| 8.3 | 纪律案例 | calib100 信号 → 891 否证 | journal 批次 104/105 |
| 8.4 | 显著胜（秩口径） | Δ+0.088 CI[0.063,0.114]；Holm p=0.004 | task1_alignment_results.json |
| 8.4 | 决策口径混合 | top-1 Δ−0.100 CI[−0.143,−0.058] 显著负 | 同上 |

- 措辞注意（§8）：三口径（秩 Spearman / 决策 top-1 / NDCG）分列不混排；决策口径显著负如实登记（混合结果）；COMB 只写 sc calibre 可加性主张 + 双口径分歧现象学，不外推「生成成功」。

---

## §9 Discussion

### 小节大纲

- **9.1 模型选择指南（分类表建议）**：按任务体制先选表——C1 任务（ICC<0.1）：不选模型，改测量/加重复；C2：输入粒度对齐优先（片段级 vs 全长）；C3：within-source 稠密候选监督 + pair-mean/组内口径评估；C4：先跑小成本 oracle 探针（NO_SIGNAL ≤0.10 即止损）再立项；C5：语料体量决定成败（280K/2.74M 外标杆）。附「数据侧三体检」（ICC / 密度 / 语料域）先于模型选型的操作流程。
- **9.2 对绝对分排行榜实践的影响**：源间成分虚高（polyA 整体 0.71-0.75 vs 组内 0.19-0.23）；域内语料绑定（STCNet 0.8144 的 730/730）在无 source-relative 协议下不可见；建议基准报告双任务 + 组内口径 + 语料重叠声明。
- **9.3 局限（逐项诚实）**：
  - **rights/数据权利**：GEMORNA/UTR-STCNet/UTR-Insight license 无声明（用户豁免决策 2026-09-20，port ledger addendum commit 5a8e30eb）；M6 封存-解封史（historical_exposure，禁用 sealed/untouched/never-seen 措辞）；payload 发布边界（public study-payload release 授权 0 行，v332 rights 表）。
  - **单 seed 探针**：新族矩阵 65 格为单 seed frozen-Δ 评测（行构建 seed 冻结 20260920；模型推理确定性但行抽样单一）；oracle/一阶分解等分析多为 3-seed（MRL oracle seed 敏感发现本身即单 seed 伪信号的证据）；matrix 无多 seed 方差带（如实登记）。
  - **VERDICT FAIL 的诚实处理**：密度单因子 held-out FAIL 已按降级条款执行（唯一升级路径 = 只增行 + 下一判定点重评）；双因子主张为 EXPLORATORY 等级（非预注册判定门）；M1 干预臂结果未回填（空槽）。
  - **VALIDATION-only**：全部矩阵/榜单数字为 VALIDATION split；protected TEST reads = 0（TEST 未开）；G4 结构探针三代零但仅限 D15-2 口径。
- **9.4 与既有基准/模型文献的关系**（一段，引用 BENCHMARK_GUIDE 竞品轴；不引具体外部文献数字——留全文撰写期）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 9.1 | 分类表建议 | 素材 5 failure_taxonomy_v1.md（C1-C5 建议块） |
| 9.2 | 源间虚高 + 语料绑定 | 素材 3 §4 + 素材 2 §4.2 |
| 9.3 rights | port ledger + M6 provenance + v332 rights 表 | docs/paper/benchmark_v2_port_ledger_v1.md（W0）+ m6_ndd5utr_eval_row/row_definition.md §0 + docs/paper/route2_v332_study_rights_*_v1.csv（W0） |
| 9.3 单 seed | matrix 协议 | docs/paper/benchmark_v2_matrix_row_prereg_v1.md（W0） |
| 9.3 FAIL 诚实 | 降级执行清单 | analysis_delta_vs_density_v2/summary.md §4.3 |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 9.1 | 三体检判据 | ICC<0.1 / 密度<10 / 语料域映射 | failure_taxonomy_v1.md §0 量化阈值备注 |
| 9.2 | 源间虚高量级 | 0.71-0.75 → 0.19-0.23 | 素材 3 §4 表 |
| 9.2 | 语料绑定量级 | 0.8144 vs 0.0672（同族跨库） | 素材 1 + 2 |
| 9.3 | 豁免记录 | 3 族 license 用户豁免（commit 5a8e30eb） | port ledger addendum |
| 9.3 | FAIL 执行 | verdict FAIL + downgrade_triggered=true | delta_vs_density_v2_results.json |

---

## §10 Methods

### 小节大纲

- **10.1 数据管线**：canonical 层（9 研究 converter：GSE114002 mother-family identity/candidate 规则等）→ projection 层（development_train_validation_v1，direction_normalized_delta 标签）→ 新评测行层（M1/M6/S1 append-only，三份 row_definition 口径全声明：M1 前缀家族+Hamming-1 兜底；M6 SIC→fraction 映射 + 伪计数 1.0 + 155nt 窗口；S1 canonical_115 双 assay sub-row + 受保护排除 + Dao QC flag-keep）。
- **10.2 评估器**：Task-1 evaluate_route2_prediction_v1.evaluate（同一实例、K=10；source-group within-source Spearman + 任务宏 Spearman）；MPRAU pair-mean 口径（2,008 变体）；独立复核协议（不读入档指标、predictions.jsonl 重算、|Δ|≤1e-6 门）。
- **10.3 外部模型移植**：port ledger（5 PORT_READY + license 处置）；逐族端口适配声明（matrix_v2_summary 原文块）；单测协议（官方例值/strict load/官方 CSV 对齐 rho=0.96）；HydraRNA MIG env 修复声明（Triton autotuner 单设备可见性，零数值变更）。
- **10.4 预注册文档清单**（先于计算冻结的完整列表）：delta_density_prereg_framework_v1 / delta_validity_oracle_prereg_v1（commit a2a2919a）/ first_order_decomposition_prereg_v1（8f80d7e6）/ benchmark_v2_matrix_row_prereg_v1 / m1_intervention_arm_amendment_v1（b2e141cd）等——逐项 commit 哈希列表（全文撰写期补全）。
- **10.5 复现入口**：构建脚本（scripts/route_a_v3/benchmark_v2_rows/build_{m1,m6,s1}_*_v1.py，seed 落盘可复跑）；矩阵执行/聚合/复核脚本（scripts/route_a_v3/benchmark_v2_matrix/ 八件）；分析脚本（run_delta_vs_density_v2.py 等）；**R2 内靶行**（global_scaled 内靶对照 9 任务全覆盖，macro 0.1317）；**LOSO 引用**（LOSO-lite Table 5：A = 真 LOSO 7 折重训 / B = in-domain 表头改注——裁决未定，引用不改）。
- **10.6 泄漏与权利审计**：鸽笼审计（S1/M6 flagged=0；M1 4 条 TRAIN keep=0）；STCNet 旧三臂 INVALID 案例归档（R3 训练集泄漏判定，commit 08ff6c2f）；权利边界表（v332 study rights 表；public release 0 行授权）。

### 素材映射

| 小节 | 素材 | locator |
|---|---|---|
| 10.1 | canonical/projection | RT2/canonical/ + RT2/projections/xedit_v3/development_train_validation_v1 |
| 10.1 | 三新行 | 三份 row_definition（§2 同源） |
| 10.2 | 评估器 | matrix_v2_summary.md 头部协议块 + recheck_results.json |
| 10.3 | 移植 | docs/paper/benchmark_v2_port_ledger_v1.md（W0）+ family_unit_tests.json |
| 10.4 | 预注册清单 | W0 docs/paper/ 预注册文件集（commit 链见 journal 批次 107-112） |
| 10.5 | R2 内靶 | task1_internal_controls_per_task.json（榜单 §1.5） |
| 10.5 | LOSO | RT2/experiments/analysis_task11_loso_lite_20260909/table5_loso_lite_draft_v1.md |
| 10.6 | 审计 | S1/M6 pigeonhole v2 + frozen_delta_results.json（STCNet INVALID 判定） |

### claim-证据表

| # | claim | 证据数字 | locator |
|---|---|---|---|
| 10.2 | 评估器一致性 | 65/65 max |Δ|=0.0 | recheck_results.json |
| 10.3 | 单测 | 5/5 PASS（Insight 对齐 rho=0.96） | family_unit_tests.json |
| 10.5 | 内靶对照 | macro 0.1317；9 任务逐行 | task1_internal_controls_per_task.json |
| 10.5 | LOSO-lite | Table 5 草表 9 study 行 | table5_loso_lite_draft_v1.md |
| 10.6 | 鸽笼 0 | S1 flagged=0 / M6 flagged=0 | pigeonhole v2 审计 JSON |
| 10.6 | STCNet INVALID 先例 | 三臂 0.8135/0.2667/0.2120 全排除 | frozen_delta_results.json（泄漏判定） |

---

## 附 A：R1-R13 攻击面自查表（Task 4.2）

> 状态口径：CLOSED = 已有实证/协议闭合；OPEN = 待回填/待裁决。R1-R11 = 既有攻击面（journal 与 spec v4 证据链入档）；R12/R13 = 本骨架新增。

| # | 攻击面 | 质疑内容 | 闭合证据（数字 + locator） | 状态 |
|---|---|---|---|---|
| R1 | 外部模型未充分调参 | 「frozen 行低估外部模型」 | MRL matched-FT 双臂终态：外部模型获对等微调机会且退化（薄数据微调退化跨阵营对称）；frozen 平局 + matched-FT 我方领先（journal 批次六十；analysis_mrl_matched_ft_20260909/） | CLOSED |
| R2 | 内靶对照缺失 | 「我方行无内部对照」 | 内靶 global_scaled 9 任务全覆盖 macro 0.1317；V5 胜 8/9 内靶（唯一负 GSE149487-RNA n=48 如实）；Table 5 LOSO-lite 草表（task1_internal_controls_per_task.json + table5_loso_lite_draft_v1.md） | CLOSED |
| R3 | 训练集泄漏 | 「高分行是记忆」 | STCNet 旧三臂 INVALID 判定归档（0.8135 系 designed_library 训练，commit 08ff6c2f 全排除）；新矩阵 STCNet 0.8144 如实标注语料重叠 730/730（域内属性披露而非隐藏）；S1/M6/M1 鸽笼 flagged=0 | CLOSED |
| R4 | 指标口径不稳定 | 「NDCG@K 结论依赖 K」 | K 敏感性 29 行：K≥5 全行逐位稳定（K=3 仅微降）；V5 各行与榜单参照精确一致（analysis_baseline_k_sensitivity_20260908/） | CLOSED |
| R5 | 弱对照空白 | 「MPRAU 无外部对照」 | Saluki MPRAU weak-control 0.1205 CI[0.077,0.164]（preregistered 弱对照口径如实）；matched-FT 三 seed 全负实证闭环（analysis_mprau_matched_ft_external_20260904*/） | CLOSED |
| R6 | 移植保真度 | 「端口适配改了模型」 | RiboNN 官方预测对齐（model0 max diff 0.0034 / model1 0.0012，官方 pearson 0.758 复现）；矩阵 5 族单测全 PASS（Insight 官方 CSV 对齐 rho=0.96）；端口适配逐族全声明 | CLOSED |
| R7 | seed 数不足 | 「单 seed 不可复现」 | 底线判定 ≥3 seeds/方法外部同等待遇协议（MPRAU matched-FT 3 seeds；MRL 3-seed ens）；MRL oracle seed 敏感伪信号发现本身入档（0.1370→0.0640/0.0597）；残余：矩阵 65 格单 seed 行抽样（§9.3 如实登记） | CLOSED（带残余声明） |
| R8 | 判定门事后调整 | 「阈值改过」 | 全部门阈值先于计算冻结（预注册文档 commit 链：a2a2919a / 8f80d7e6 / b2e141cd；delta-density 框架容差 0.10 / 门 70% 未触碰）；FAIL verdict 机械执行 | CLOSED |
| R9 | 评测面污染 | 「TEST 泄漏」 | protected TEST reads = 0（全程 VALIDATION only，各任务纪律终验段落）；只增行条款（新行不进 manifest） | CLOSED |
| R10 | 措辞过度 | 「全部失效式断言」 | 措辞冻结条款全文自查（本文件附 B）；失效行与对照行并列；Optimus/APARENT 为监督体制对照行 | CLOSED |
| R11 | 干预解释混淆 | 「训练失败/度量伪影/选择效应」 | G2 非破坏门全 PASS；gap 协议镜像；非可加性 +0.0044 排除；H-selection 保留为边界条件（机理链 A §7 替代解释逐项排除） | CLOSED |
| R12 | 评测协议公平性（本骨架新增） | 「frozen-Δ 零调参是否亏待新族？密度口径是否可比？域映射是否循环论证？」 | (a) frozen 零调参 + matched-FT 对位证据（R1/R5 双线：换监督也不解决——薄数据微调退化对称）；(b) 密度口径逐行声明（9 既有任务同款 within-source 口径 + M1/M6/S1 各自 row_manifest cand_per_source_density；M1 密度 1.0014 的评测行口径与训练配对密度区分声明）；(c) 域映射独立来源（port ledger §2.7-2.10 + 官方 repo 预处理脚本；STCNet 重叠为实测非循环） | OPEN（部分闭合）：frozen-Δ 公平性靠 R1/R5 间接闭合；新族 matched-FT 对位未执行（matrix prereg 范围外，如实登记）；域映射 4/5 族文档级、1 族实测级。全文撰写期回应：§9.3 增补「零调参口径的公平性边界」段 + R12 残余清单 |
| R13 | rights 与 payload 边界（本骨架新增） | 「数据发布合法性？权重再分发？」 | (a) 5 新族 license 处置：LAMAR MIT / HydraRNA fairseq-MIT 无条件；GEMORNA/STCNet/Insight 无 license 文件——用户豁免决策入档（commit 5a8e30eb，port ledger addendum）但**论文发布前需重审**；(b) M6 封存-解封史按 contract §2 汇报 historical_exposure_path，禁用 sealed/untouched 措辞；(c) v332 rights 表现状：public study-payload release 授权 0 行（HUMAN_REVIEW_PENDING；无 availability-on-request 承诺）；(d) 复现包边界 = 代码 + 指纹/审计 JSON（不含受保护 payload），accountable license review 为发布前置 | OPEN：论文 availability statement 需 study-specific rights review 后才能定稿（v332 instructions 原文要求）；豁免决策 ≠ 发布授权——如实分层 |

- R1-R13 汇总：CLOSED 11（R7 带残余声明）/ OPEN 2（R12 部分闭合、R13 待发布前重审）。

---

## 附 B：措辞冻结条款自查清单（骨架全文自查，逐项打勾）

依据：journal 批次 107-108 措辞冻结条款 + 素材 3/5/6 各底稿的自查段（骨架继承其结论，全文撰写期对成稿再执行一轮）。

| # | 冻结条款 | 自查结果 | 落点 |
|---|---|---|---|
| B1 | 禁写「所有 baseline 都无法预测 delta」/「all baselines fail」 | **通过**：骨架全文检索无此表述；统一措辞 = 「无同分布稠密监督的模型族集体受限，Optimus/APARENT 为监督体制对照行」（§3.4 / §7.4 / R10） | §3.4、§7.4、附 A R10 |
| B2 | delta-density 措辞 = 单因子已降级如实 + 双因子主张（标注 EXPLORATORY 支撑） | **通过**：§5.2 FAIL 50.77% + 降级条款执行如实写（禁「跨任务普适规律」）；§5.3/5.7 双因子主张全部带 EXPLORATORY 标注；Abstract 句 4 同口径 | §5.2、§5.3、§5.7、Abstract |
| B3 | MRL 平局禁写 no difference | **通过**：W 阶梯终点措辞 = 「统计平局、点估计领先、730-record 检验力不足不宣称显著超越」（Δ+0.0027 CI[−0.045,+0.048] 跨零如实） | §5.4、§8 无冲突 |
| B4 | 生成线三口径分列 | **通过**：§8.4 秩 Spearman / 决策 top-1 / NDCG 分列；决策口径显著负（Δ−0.100 CI[−0.143,−0.058]）如实登记为混合结果 | §8.2、§8.4 |
| B5 | HL 附不可学归因 | **通过**：HL5/HL3 归 C1 物理失败（ICC 0.0014/0.013），「任何模型在该任务 ≈0 是 benchmark 结论而非模型失败」的归因声明随行；天花板归一化 N/A 处理（不用负数分母） | §2.4、§7.1、§7.2 |
| B6 | 禁写「提密度即可修复」 | **通过**：§5.4 合成侧 D16-C G1 FAIL（gap 0.8235）已证伪该表述；全文无「提高密度即可修复」语态；因果语态一律用「真实监督数据的几何覆盖」 | §5.4、R11 |
| B7 | polyA 平局/胜出措辞 | **通过**：V5 vs APARENT Δ+0.088 CI[0.063,0.114] 写「显著胜（秩口径）」+ 决策口径混合结果如实；不写「全面超越」 | §8.4 |
| B8 | M1 空槽不写数字 | **通过**：§5.6 只写预注册门与状态，无任何结果数字 | §5.6 |
| B9 | 数字与 locator 一一对应 | **通过**：正文每个数字均出现在对应小节 claim-证据表（含 locator）；无来源数字未标注；未编造不可算量（16 行未存档 probe 权重只登记） | 全文各 claim-证据表 |
| B10 | 实证范围声明 | **通过**：骨架头部口径纪律声明（VALIDATION only、frozen-Δ、实证范围 = 9 frozen 族 × 5 任务族 × 新矩阵 5 族 × 13 格）；不主张「embedding 差分等价于性质差」类一般性命题 | 文件头、§4 |
| B11 | S1/M6/M1 泄漏与权利措辞 | **通过**：M6 禁用 sealed/untouched/never-seen（封存-解封史按契约汇报）；S1 受保护排除如实（1,330）；M1 TRAIN 重叠只标记不排除的语义如实 | §2.2、§9.3、R13 |
| B12 | STCNet 域内属性披露而非隐藏 | **通过**：0.8144 与 730/730 重叠并列披露（域内监督体制行），并作 §5.3 双因子证据；旧三臂 INVALID 判定史在 §10.6/R3 归档 | §3.3、§5.3、R3 |

---

## 附 C：骨架统计（自查产出）

- **章节结构**：Title/Abstract 位 + §1-§10 + 附 A（R1-R13）+ 附 B（措辞自查 12 条）。
- **claim-证据表覆盖统计**（骨架级 claim 总数，含各章 claim-证据表行）：

| 章 | claim 行数 | 备注 |
|---|---|---|
| §1 | 3 | 引言级 |
| §2 | 6 | 定义/规模/单测/审计 |
| §3 | 8 | 矩阵主表 + 两关键读数 |
| §4 | 10 | 五误差源 + 复核 |
| §5 | 14 | 观测/FAIL/分组/干预链（含空槽声明行） |
| §6 | 7 | 一阶分解 |
| §7 | 7 | 失效分类 |
| §8 | 10 | polyA 案例 + COMB |
| §9 | 5 | 讨论级 |
| §10 | 6 | 方法级 |
| **合计** | **76** | 逐行带 locator |

- **素材覆盖**：任务书素材清单 1-9 全部映射（素材 7 拆入 §2.3/§2.4/§8/§10，其余一一对应；既有冻结榜单 §9 全表经 TASK6 冻结表 + matrix_summary_v1.json ALREADY_DONE 块引用不改）。
- **R1-R13 状态**：CLOSED 11（R7 带残余声明）/ OPEN 2（R12 部分闭合 / R13 待发布前重审）。
- **措辞自查**：附 B 12/12 通过。
- **后续任务（骨架 → 全文）**：①标题终选（PI）②各章全文撰写（按素材灌入）③R12/R13 残余处理（零调参公平性段 + availability statement rights review）④§5.6 M1 空槽回填（干预臂终态后）⑤LOSO A/B 裁决（Table 5 定稿）⑥图表清单（散点图/矩阵热图/分解条形图）。

（骨架 v1 完）
