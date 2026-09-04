# 预注册：MPRAU 任务外部通用 LM 对位微调双臂（RNA-FM / UTR-LM，2026-09-04）

> 定位：**Baseline 补强 P0（D6 已批准）**。把 R5 报告的 MPRAU「外部行结构性空白」从不可检验的主张升级为可检验主张：外部通用 RNA LM 在与我方 per-task 微调臂**完全相同的任务数据与更新预算**下对位微调，若仍做不动 → 空白主张成立；若反超 → 如实报告并进 W 阶梯。
> 依据：SPECS_OPTIMIZATION_V2 spec §三「公平预算协议」；对照臂 = directft 臂 A（mRNABERT，`route2_mrnabert_directft_diag_prereg_v1.md`，脚本 `run_route2_mrnabert_directft_arm_a_v1.py`，其 loss/数据/评估实现直接复用）。
> 纪律（不可违反）：VALIDATION only、protected reads=0（不碰 TEST split）；CUDA BF16-only（禁 CPU 训练）；不干预任何在途进程（GPU0 p4_train、GPU1 B2 guided PID 1909082 等勿动；本实验只用 GPU3/GPU4）；产物写 /mnt，代码与文档提交本 worktree 并 push GitHub；**预注册后启动**。

## 1. 任务与现状

- 任务：`mprau_encsr854ruf`（ENCSR854RUF，3'UTR 变体等位偏移 MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE，133bp oligo，6 细胞系 context）。
- TRAIN 池 55,704 行（9,284 变体 × 6 context）；VALIDATION 12,048 行（2,008 变体 × 6 context）。
- 现有 VALIDATION 行：V5 多任务 0.1025（pair-mean ρ）；W-ladder 各臂 0.0510–0.0883；mRNABERT direct-FT 臂 A（LoRA，同预算）−0.0908；Saluki frozen 弱对照 0.1205（口径不匹配：3'UTR 降解模型 vs 等位偏移端点）；ceiling 0.683。**外部通用 RNA LM 的对位微调行 = 空白**。

## 2. 两臂定义（共同协议，仅 backbone 与官方加载方式不同）

- **RNA-FM 臂**（GPU3）：multimolecule 官方转换权重 `external_model_assets/rnafm`（RnaFmModel，99,521,920 参数，12 层 × 640 hidden，RNA 字母表 U）。加载 = `RnaTokenizer.from_pretrained` + `RnaFmModel.from_pretrained`（项目 frozen 基线已验证模式，pooler 缺失无碍——readout 不用 pooler）。
- **UTR-LM 臂**（GPU4）：官方 SISS checkpoint `external_model_assets/utrlm/Model/Pretrained/ESM2SISS_FS4.1_..._epoch93.pkl`（1,208,559 参数，6 层 × 16 heads × 128 embed，rotary 位置编码，DNA 字母表 ACGT）。加载 = vendored 官方 esm 包（`external_models/utrlm/Scripts`）`ESM2(num_layers=6, embed_dim=128, attention_heads=16)` + 剥 `module.` 前缀 + `strict=True`（项目 frozen 基线 `run_route2_utrlm_baseline_v1.py` 已验证模式；UTR-LM 输入需 U→T）。
- **Readout（两臂同构，fresh 初始化）**：
  - RNA-FM：非特殊 token 的 masked-mean-pool（与本项目 frozen RNA-FM 基线同口径）→ 线性 head（640→1）。
  - UTR-LM：BOS token 表征 `representations[6][:, 0]`（官方下游 `--bos_emb` 模式，与本项目 frozen UTR-LM 基线同口径）→ 线性 head（128→1）。
  - 线性 head 结构统一（pool → Linear→1），不用官方下游的隐层 head：保证两臂与我方 directft 臂 A 的 readout 同构，「matched」解释性优先；预训练 artifact head（lm_head/contact_head/supervised_linear/structure_linear）不参与损失图。
- **监督目标与预测结构**：`direction_normalized_delta`（source-relative Δy）；prediction = f(candidate) − f(source)（frozen-delta 结构，训练与评估同构，镜像臂 A）。

## 3. 微调方式选择（预注册声明）

- **两臂均取全参微调（full-parameter fine-tuning）**，理由：
  - RNA-FM（multimolecule 官方）：README 下游示例 = `RnaFmForSequencePrediction.from_pretrained` + HF Trainer 常规训练（无 LoRA 示例）→ 官方推荐即全参。
  - UTR-LM（官方仓库）：下游脚本 `Finetune_extract_append_predictor_*.py` 的 `--finetune` 分支 = 整个 ESM2 backbone + head 全部 `requires_grad=True` → 官方推荐即全参（不带 `--finetune` 则为 frozen probe，已被本项目 frozen 基线覆盖）。
- **优化器（协议统一项，非官方逐字复制）**：AdamW wd 1e-4，cosine 衰减至初始 10% + 5% warmup，grad clip 1.0，BF16 autocast——与我方臂 A 及 W 系列同构，保证「除 backbone 外唯一差异」的可比性；官方 UTR-LM 用 SGD momentum 0.9，如实记入差异清单（§7）。
- **学习率（一次性预注册，不做调参）**：
  - RNA-FM（99.5M 全参）：backbone 2e-5 / head 1e-4（~100M 预训练 LM 全参微调的标准量级；head 新鲜初始化需更高）。
  - UTR-LM（1.2M 全参）：backbone 1e-4 / head 2e-4（微型模型，与臂 A 的 LoRA/head 组学习率同值，量级保守于官方 SGD lr 0.1 的有效步长）。

## 4. 预算（与 directft 臂 A 完全一致）

- **12,048 行/pass × 8 passes**（每 pass 从 55,704 行 TRAIN 池内均匀无放回抽取，`torch.randperm`，每 pass 新抽样）× **batch 32 = 377 updates/pass × 8 = 3,016 updates**。
- seed 20260907（W 系列统一，与我方臂对齐）。
- 选择规则：**FINAL-PASS-8-FIXED**（终态 = pass 8 结束时权重，禁 peak-picking）；per-pass VALIDATION 曲线落盘仅供记录。

## 5. Loss（完整镜像 directft 臂 A / Step-2 配方，无简化）

- huber(δ=1.0)（权重 1.0）
- 跨源 pairwise softplus：pass 1–2 权重 0.25 / pass 3–8 权重 0.5
- within-source pairwise softplus：恒 0.5
- soft-Spearman（温度 0.2）：pass 1–2 权重 0.0 / pass 3–8 权重 0.25
- 分组键 = `source_group_id`（batch 内 C(n,2) 全对，跨源/同源分桶，与臂 A 逐行同实现）。

## 6. 评估口径（与历史行同口径，绝不偷换）

- **主判据 = 变体 pair-mean ρ**：record_id 去 `:context:` 后缀分组（2,008 变体，≥2 context 才计入），每变体 target 均值 vs prediction 均值，变体层 Spearman。实现镜像臂 A 脚本（`mprau_variant_table` / `pair_mean_rho`，算法同 `adjudicate_v7_mprau_vs_v5_v1.py` / W-ladder adjudication）。**绝不用 run_summary 的池化 pair_mean_spearman 冒充**。
- **paired bootstrap（2,000 iters，seed 20260816，共享变体上重采样）**：
  - vs **V5 多任务 0.1025**：V5 预测 `experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl`；
  - vs **Saluki frozen 0.1205**：预测 `experiments/analysis_saluki_frozen_mprau_20260903/predictions.jsonl`（target 由 VALIDATION 投影行 join）。
  - 报告 Δ 与 95% CI，CI 排除 0 且方向为正 → 显著反超；CI 排除 0 且方向为负 → 显著不及；否则无显著差异。
- 次级口径（记录用）：frozen Task-1 evaluator（K=10）`task_macro_spearman` / `source_macro_top_1_accuracy` / `source_macro_ndcg_at_k`，预测同 frozen-delta。
- VALIDATION only；主判据只看 pass 8 终态。

## 7. 判定（预注册）

| 结果形态 | 判定 |
|---|---|
| 两臂 vs V5 与 vs Saluki 的 CI 均不排除 0（或显著为负），且 pair-mean ρ 未超 W-ladder 上沿 0.0883 量级 | **空白主张成立**：外部通用 LM 对位微调（全参、同预算）也做不动 MPRAU，R5 结构性空白升级为可检验结论 |
| 任一臂 pair-mean ρ 反超 V5 0.1025 且 bootstrap CI 排除 0（方向为正） | **如实报告反超**，该臂进 W 阶梯候选，空白主张对该 backbone 不成立 |
| 介于其间（数值高于 W-ladder 但未显著超 V5/Saluki） | 如实报告边际改善，空白主张弱化为「无显著外部增益」 |

- 对照参考行：V5 0.1025 / Saluki frozen 0.1205（弱对照，口径不匹配声明保留）/ mRNABERT direct-FT 臂 A −0.0908 / W-ladder 0.0510–0.0883 / ceiling 0.683。
- 负结果与预测方向错误均如实填入；本实验不动任何冻结 gate。

## 8. 与官方配方 / 臂 A 的差异清单（诚实声明）

1. 微调方式：臂 A = LoRA r16 α32（mRNABERT ~86M，W1' 禁全参条款延续至我方模型）；本实验两臂 = 全参（各自官方推荐模式，预注册选择）。
2. 优化器：UTR-LM 官方 = SGD momentum 0.9（lr 0.1–0.001 视命令而异）；本实验 = AdamW+cosine（协议统一项，理由见 §3）。
3. head：官方下游 head 含隐层（nodes=40 + dropout）；本实验 = 单线性 head（跨臂同构，matched 可比性优先）。
4. readout：RNA-FM = masked-mean-pool；UTR-LM = BOS 表征（各自与本项目 frozen 基线口径一致）。
5. 其余（数据、预算、loss、调度、seed、评估、选择规则）与臂 A 完全一致。

## 9. 产物路径与执行

- 脚本：`scripts/route_a_v3/run_route2_mprau_matched_ft_external_v1.py`（`--model rnafm|utrlm --physical-gpu-index N [--smoke]`）。
- 产物：`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_mprau_matched_ft_external_20260904/{rnafm,utrlm}/`（results JSON、checkpoint、predictions.jsonl、per-pass 曲线）；日志 `.../logs/`。
- 执行：冒烟（~100 步，/tmp 产物）通过后，GPU3 = RNA-FM 臂、GPU4 = UTR-LM 臂，nohup 后台，预计各 1h 内量级；出数即收割写训练日志并 commit+push。
