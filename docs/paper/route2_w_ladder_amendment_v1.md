# Route 2 W 阶梯 Amendment 预注册 v1（W0 诊断 + W1' 两阶段）

> 依据：spec `define-downstream-tasks-and-benchmark-comparison` v5.1 §四（协议前置："W0/W1' 均为协议外新实验形态 → Phase 2 启动前统一 amendment 预注册"）
> 冻结时间：2026-09-02（Asia/Shanghai）
> 状态：**FROZEN**——本 amendment 连同 W0 训练配置在同一 commit 冻结；启动后零修改
> 纪律：protected reads = 0（TEST 在 Gate P 拍板前不碰）；CUDA BF16-only；预注册门槛不事后改

## 1. W0 单任务架构诊断（本 commit 启动）

**目的**：定位 MRL/polyA 对位差距来源——critic 架构本身 vs 多任务训练配方（spec 坑 D）。

**设计**（两臂，同一冻结管线 `train_route2_xeditcritic_v4.py` + `study_filter` 新配置键）：

| 臂 | run_id | study | TRAIN/VAL | updates/pass × 8 pass = total | GPU |
|---|---|---|---|---|---|
| W0-MRL | `w0_mrl_gse114002` | GSE114002（MEAN_RIBOSOME_LOAD::region=0）| 2,443 / 730 | 77 × 8 = 616 | 6 |
| W0-polyA | `w0_polya_gse269595` | GSE269595（PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1）| 25,710 / 2,628 | 804 × 8 = 6,432 | 7 |

**不变量**（与 V6 screen 完全一致，保证架构对位纯净）：
- 架构：V6-FULL 170,679,590 trainable（含 cell_offset_head_v6）；bottom-6 mRNABERT 冻结缓存复用全量数据集缓存（107,873 rec / 43,730 unique seq，superset 查找）
- vocab（study/assay/context/quantity/measurement/numerator/denominator）在**全量 projection** 上构建——模型容量头与多任务 screen preflight 精确一致（参数量逐位相等）
- 训练配方：seed 20260907；8-pass 冻结 loss schedule（pass1-2 / pass3-8）；FINAL_PASS_8_FIXED 选择；AdamW 分层 LR（top-6 1e-5 / trunk+head 2e-4 / experts+router 1e-4）+ cosine decay 至 10% + 5% warmup；pair-mean 与 rank-Gaussian 对非 MPRAU 任务为恒等映射（已核验代码路径）
- 唯一 delta = 数据作用域（study_filter 后的 TRAIN/VALIDATION split）与相应 data_geometry 计数

**预注册判定带**（同评估器 `evaluate_route2_prediction_v1.py`、VALIDATION split、K=10、与 Task 1 对齐行同记录集）：
- W0-MRL：validation Spearman ≥ 0.28（≈ Optimus adapter 0.3132 − 0.034）→ **架构无虞，差距主因多任务配方** → W1' 优先推进；< 0.20 → **架构可疑** → W2（per-task 容量）/架构侧修订优先；[0.20, 0.28) → 混合，两线并行
- W0-polyA：top-1 ≥ 0.55 且 NDCG@10 ≥ 0.885（APARENT 0.6011/0.8906 邻域）→ 架构无虞；top-1 < 0.50 → 架构可疑
- W0 结果**不回填** V5/V6 多任务行；只作诊断入档与 W 阶梯路由依据

## 2. W1' 两阶段（W0 出数后启动，本节为预注册）

- 阶段 A（已存在）：多任务终态 checkpoint（V5/V6/V7 过门者）作为初始化
- 阶段 B：per-task **LoRA/head-only 微调，冻结骨干，禁止全参**（坑 A：GSE114002 TRAIN 2,443 行 vs 170M 参数 ≈ 70,000:1 过拟合比）
- 措辞：supervised cross-endpoint transfer（多任务阶段 = 其他 8 任务带标签 TRAIN split），**非**自监督预训练（坑 B）
- 架构对架构对照：最强外部架构（Optimus）同跑两阶段（多任务 → per-task 微调）作对照行
- 泄漏边界（坑 C）：目标任务在两阶段均只暴露 TRAIN split；VAL/TEST 纪律不变
- 对位预算：与外部 adapter 同数据同更新步数（协议 §三）；≥3 seeds 判定（W1' 首轮单 seed 探路，过带后再补 seeds——探路轮不作为底线判定依据，仅路由）

## 3. 记录与收割

- 训练过程与结论逐轮记入 `docs/training_journal/`（本 worktree）并随 commit 上传 GitHub
- W0 终态（run_summary.json）出现后 1 个工作日内收割：同口径对位表 + 预注册判定带结论
- 若 W0-MRL 达到"架构无虞"带，W1'-MRL 立即启动（LoRA 微调配置另行 commit 预注册）

## 4. 代码 delta 说明（本 commit）

`scripts/route_a_v3/train_route2_xeditcritic_v4.py` 四处向后兼容修改：
1. `_require_bottom_six_preflight_identity_v4`：cache identity 的 record/unique-sequence 计数改为从 `data_geometry` 显式键读取（默认值 = 原行为）
2. 记录加载后新增可选 `study_filter` 过滤（无该键 = 原行为，逐位一致）
3. vocab 构建改在全量 records 上（study_filter 存在时；否则原行为）
4. 数据集构建处 cache identity 同 1（默认值 = 原行为）

无 `study_filter` 键的既有 config（V6/V7）行为逐位不变——已通过 py_compile + 既有 focused 单测验证。

## 5. Addendum A（2026-09-02 15:45，首次发射尝试后、任何训练开始前）

首次 W0-MRL 发射在数据集构建阶段被 bottom-six cache view 的精确覆盖检查拦截（要求 projection 记录集与缓存记录集完全相等）。补充第 5 处向后兼容代码 delta：

5. `core/route2_xeditcritic_batch_v4.py` `FrozenBottomEncoderChunkCacheViewV4`：覆盖不变量由"精确相等"放宽为"子集覆盖"（缓存 ⊇ projection）——本质不变量（每个 projection 记录都有 donor）保持；全量数据集运行行为不变；缓存为冻结 outcome-free 特征，超集安全。对应单测 `test_v4_cache_view_requires_full_projection_coverage` 更新为双断言（子集通过 + 缺失记录仍拒绝）。

首次失败已如实入账本（screen_w0_mrl_gse114002_928f676c → FAILED, TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE），失败目录保留为 `w0_mrl_gse114002_aborted_by_cache_view_subset_20260902`。本 Addendum 于重发射前冻结。

## 6. Addendum B（2026-09-02 晚，W1' 两臂终态收割后、发射前）

**触发**：W0-MRL 终态 0.1987，pass 级 soft-spearman loss 单调下降 0.62→0.33 未平台化 → 预算限制假说待检验。用户批准"现在启动"（2026-09-02 晚）。

**新增 W0-continue 臂（run_id `w0_continue_mrl_gse114002`）**：
- 初始化 = W0-MRL `final_pass_8_checkpoint.pt` 严格加载（state_dict 509 键，含 cell_offset_head；arch/config 与 W0 逐字段一致）
- **全参数继续训练**（无冻结、无 LoRA）；标准三组优化器（head 2e-4 / semantic 1e-4 / upper-6 1e-5）+ 全新 cosine warm-restart 调度；同 seed 20260907、同 8-pass 冻结 loss schedule、同 study_filter（GSE114002）/data_geometry（77 updates/pass × 8 = 616）
- **与 W1' 禁全参条款的关系（纪律说明）**：坑 A 条款针对"多任务 init 适配到 2.4K 行小任务"的过拟合场景（V5 多任务表征 + 全参微调）；本臂是**同一单任务训练的预算延长**——W0 已在完全相同的 2,443 行 TRAIN 上训练 616 updates 且验证指标仍在改善区间，继续训练不改变参数:观测比的过拟合结构。本臂按协议外新形态特此预注册。

**预注册判定带**（同评估器/VALIDATION split/K=10；对照 W0 0.1987 / Optimus adapter 0.3132 / 内靶 0.1192）：
- **≥0.25**：预算是主要限制 → W0-continue 路线加码（更长训练），W2 容量线并行
- **0.20–0.25**：预算部分限制，混合归因
- **≈0.20 或更低（验证回落）**：预算不是限制（过拟合已开始）→ 差距主因架构/先验 → 强化 D3 路线 A（外部大库预微调）优先级
