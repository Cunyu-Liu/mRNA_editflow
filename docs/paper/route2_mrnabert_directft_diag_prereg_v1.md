# 预注册：mRNABERT 直调双臂 × 三任务诊断实验（Phase 0 Task 3，2026-09-04）

> 定位：**诊断性实验（Phase 0）**，只填证据矩阵格子，不动任何已冻结 gate。所有判定门均为诊断带（diagnostic band），不触发任何 S1/gate 状态变更。
> 依据：SPECS_OPTIMIZATION_V2_20260904 Phase 0 Task 3；路线 A 必要性分析（route2_route_a_necessity_certainty_analysis.md）。
> 纪律（不可违反）：CUDA BF16-only（禁 CPU 训练）；protected reads=0（不碰 TEST split）；不干预任何在途进程（B2 guided PID 1909082 在 GPU1；既有 watcher 勿动）；产物写 /mnt，代码提交本 worktree 并 push GitHub；预注册后启动。

## 1. 目的：填证据矩阵唯一未测格子

证据矩阵（MRL GSE114002 VALIDATION，frozen Task-1 evaluator K=10）现有行：

| init / 训练方式 | MRL VALIDATION Spearman |
|---|---|
| W0 从零训练（170M critic，同构单任务重训） | 0.1987 |
| W1-lora（V5 critic init + LoRA） | 0.1486 |
| Step-2（mRNABERT 280K 预微调 → MRL 任务适配） | 0.2159 |
| Route A zero-shot（280K frozen-delta，无任务数据） | 0.2470–0.3158 |
| **本实验（mRNABERT 原始预训练权重 init + 任务数据微调）** | **?（待填）** |

本格子回答：**mRNABERT 的原始预训练权重（不做任何域库预微调）+ 任务数据 LoRA 微调**，在各任务上能到什么水平。它与 W0（同预算从零）配对分离「预训练 init 的贡献」；与 Step-2 配对分离「280K 域库预微调的增量贡献」；与 Route A zero-shot 配对分离「任务数据微调的增量贡献」。

## 2. 臂 A（端到端）× 3 任务

- **任务**：`mrl_gse114002`、`polya_gse269595`、`mprau_encsr854ruf`，各单任务独立训练，互不共享。
- **Init**：mRNABERT 原始预训练权重（`external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40/pytorch_model.bin`），加载方式沿用 280K 预微调脚本已验证模式：`AutoModel.from_config` + 手动 `torch.load` 剥 `bert.` 前缀 + `flash_attn_qkvpacked_func = None`（首发射教训：`AutoModel.from_pretrained` 与自定义 ALiBi 不兼容会崩溃）。**不做 280K 预微调，不做 LoRA merge**——原始权重即冻结起点。
- **适配器**：LoRA r=16 α=32 dropout=0.05，注入位置照 Route A：每层 Wqkv / attn-out.dense / mlp.gated_layers / mlp.wo，全 12 层 × 4 = 48 Linears（与 Step-2 同款 `wrap_lora`）。readout = masked-mean-pool + 线性 head（head 新鲜初始化）。
- **可训练参数**：仅 fresh LoRA + head（W1' 禁全参条款延续）。
- **监督目标**：该任务 (source, candidate) 对的 `direction_normalized_delta`（Δy 回归/排序），prediction = f(candidate) − f(source)（frozen-delta 结构，训练与评估同构）。
- **Loss（完整镜像 Step-2，无简化）**：huber δ=1.0 + 跨源 pairwise softplus + soft-Spearman（温度 0.2）+ within-source 0.5 权重；两段调度：pass 1–2 = {huber 1.0, pairwise 0.25, soft_spearman 0.0}，pass 3–8 = {huber 1.0, pairwise 0.5, soft_spearman 0.25}。
- **预算**：8 passes，batch 32，LoRA lr 1e-4 / head lr 2e-4，AdamW wd 1e-4，cosine 衰减至初始 10% + 5% warmup，grad clip 1.0，BF16 autocast。
  - MRL：TRAIN 2,443 行 → 77 updates/pass × 8（与 W0-MRL 同预算）
  - polyA：TRAIN 25,710 行 → 804 updates/pass × 8（与 W0-polyA 同预算）
  - MPRAU：注册预算 12,048 行/pass → **377 updates/pass × 8**（spec 记 ≈376，按 ceil(12048/32)=377 与 W0 系 ceil 规则一致）。**实际 TRAIN 池为 55,704 行（9,284 变体 × 6 细胞）**，每 pass 从池内均匀无放回抽取 12,048 行（每 pass 新抽样，8 pass 期望覆盖 ~88% 行）。此差异如实记录：spec 写 12,048 行系将 VALIDATION 结构（2,008 × 6）误记为 TRAIN；按注册预算字面值执行。
- **seed**：20260907（W 系列统一 seed，取自 W0 config `screen_seed`；Step-2 用 20260903，差异记录）。
- **per-pass VALIDATION 指标落盘**（诊断曲线，不做 peak-picking）。
- **选择规则**：FINAL-PASS-8-FIXED（终态 = pass 8 结束时权重，禁 peak-picking）。

## 3. 臂 B（两阶段 embedding）× 3 任务

- **输入**：对应任务臂 A 终态 checkpoint（pass 8 固定），整体冻结。
- **embedding 提取**：对 TRAIN + VALIDATION 全部行的 source/candidate 各提 768 维池化表征：
  1. **masked-mean-pool**（与臂 A readout 同构）；
  2. **edit-centered pooling**：hidden states 在 `source_relative_edits` 各编辑位 ±16 token 窗并集上的均值（tokenizer 为逐核苷酸词级，token 位 = 核苷酸位 + 1 [CLS] 偏移，位置映射已验证）。若某行缺失 edits 字段则回退 masked-mean 并计数落盘；若该池化在此架构上不可用，则只用 masked-mean 并如实记录。
- **预测器特征**：`[e_source; e_candidate; e_source − e_candidate]`（2304 维），TRAIN 拟合 StandardScaler 固定变换。
- **轻预测器三选一（sklearn，全部固定超参）**：
  - ridge：`Ridge(alpha=1.0)`
  - mlp：`MLPRegressor(hidden_layer_sizes=(256,64), early_stopping=True, max_iter=500, random_state=20260907)`
  - gbdt：`HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=20260907)`
- **网格**：{池化} × {预测器} = 2 × 3 = 6 配置，其余全部固定。
- **拟合数据**：任务 TRAIN 全池（MPRAU 为全部 55,704 行；臂 B 不受臂 A 的 per-pass 子采样预算约束——embedding 提取与预测器拟合均无 pass 概念，如实记录此不对称）。
- **选择规则（预注册）**：MRL/polyA 取 VALIDATION task_macro_spearman 最高者；MPRAU 取 VALIDATION 变体 pair-mean ρ 最高者（与该臂主判据同口径）。注意：臂 B 报告的 VALIDATION 数值是 6 配置网格最大值，**选择有偏**，解读时须声明；臂 A 无此问题。

## 4. 评估口径（与历史行同口径，绝不偷换）

- **MRL / polyA**：frozen Task-1 evaluator（`evaluate_route2_prediction_v1.py`），VALIDATION split，K=10：per-task Spearman（`task_macro_spearman`）+ top-1（`source_macro_top_1_accuracy`）+ NDCG@10（`source_macro_ndcg_at_k`）。预测 = f(candidate) − f(source) frozen-delta。
- **MPRAU 主判据**：变体 pair-mean ρ——record_id 去 `:context:` 后缀分组（2,008 变体，≥2 context 才计入），每变体 target 均值 vs prediction 均值，变体层 Spearman。实现镜像 `w_ladder_adjudication.py` / `adjudicate_v7_mprau_vs_v5_v1.py`。**绝不用 run_summary 的池化 pair_mean_spearman 冒充**（历史口径陷阱）。
- **MPRAU 对照**：vs V5 多任务 0.1025 的 paired bootstrap（2,000 iters，seed 20260816，共享变体上重采样，V5 预测文件 `xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl`）。
- **参考行（MRL）**：W0 0.1987 / W1-lora 0.1486 / Step-2 0.2159 / Route A zero-shot 0.2470（280K frozen）/ frozen-FramePool 0.2956 / frozen-Optimus 0.3132。
- **参考行（polyA）**：APARENT adapter top-1 0.6011 / NDCG@10 0.8906；W0-polyA Spearman 0.8142。
- **参考行（MPRAU）**：V5 多任务 0.1025；W-ladder 各臂 0.0510–0.0883；ceiling 0.683。

## 5. 判定与对照（诊断带，均为描述性，不动冻结 gate）

- **MRL 臂预判带 0.15–0.22**：> 0.22 → mRNABERT 原始权重先验 + 任务微调即超过从零天花板（H-prior 获支持，280K 增量存疑）；< 0.15 → 原始 init 不及从零（预训练权重与任务结构不匹配嫌疑）；0.15–0.22 → 与 W0 大致持平，先验贡献中性。
- **polyA 臂预判带 ≥ 0.81**（Spearman，对标 W0-polyA 0.8142）。
- **MPRAU 臂主判据**：vs V5 0.1025 的 paired bootstrap CI 排除 0 且方向为正 → 单任务直接微调优于多任务 V5；否则如实报告（包括显著为负）。
- **预测方向错误不是失败**：本格子无论结果如何均如实填入矩阵；负结果同样关闭一个假说分支。

## 6. 产物路径

- 臂 A/B 结果：`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/directft_diag_20260904/{task}_arm_{a,b}/`（results JSON、checkpoint、predictions.jsonl、per-pass 曲线）。
- 训练日志：`.../directft_diag_20260904/logs/`。
- launcher watcher：`/home/cunyuliu/mrna_editflow_goal/monitor/directft_launch_watcher.sh`，日志 `monitor/directft_launch_watcher.log`。排队：MRL-A → MPRAU-A → polyA-A → 三个 B；启动条件 = 完整卡（GPU 0–5，非 MIG）显存余量 ≥ 20GB 且该卡无我方 python 进程（跳过 watcher 链占用卡，含 B2 终态后将被 APA 占用的 GPU1）；出现第二张空卡时并行第二队列。

## 7. 与 Step-2 配方的差异清单（诚实声明）

1. init：Step-2 = 280K 预微调 checkpoint（先 merge 旧 LoRA 再 fresh LoRA）；本实验 = mRNABERT 原始预训练权重直接 + fresh LoRA + fresh head，无 280K 阶段。
2. 任务范围：Step-2 仅 MRL；本实验三任务（MRL/polyA/MPRAU）各自独立。
3. seed：Step-2 = 20260903；本实验 = 20260907（W 系列）。
4. MPRAU 预算实现：TRAIN 池 55,704 行，每 pass 无放回抽 12,048 行（377 updates/pass），见 §2。
5. loss/优化器/调度/注入位置/选择规则：与 Step-2 完全一致（无简化）。
