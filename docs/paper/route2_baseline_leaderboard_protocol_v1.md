# Route 2 Baseline Leaderboard 评估协议 v1（已冻结）

> Spec 依据：`define-downstream-tasks-and-benchmark-comparison` v5（已批准）§二/§三
> 状态：**FROZEN v1**——2026-09-02 冻结（worktree route_a_v3_baseline_leaderboard_20260902，基 e57d1fb7）；spec v5.1 获胜阶梯与公平预算条款随本协议一并冻结
> 预注册纪律：冻结后零修改；adapter 规则评估开始后零修改

## 1. 评估器与口径（已核验存在的实现）

- 评估器：`scripts/route_a_v3/evaluate_route2_prediction_v1.py`（schema `route_a_v3_route2_prediction_evaluation.v1`）
- Split：VALIDATION（18,293 rec）；**TEST 全程不碰**（Gate P 拍板前 protected reads = 0）
- K = 10（与 Track B 既有行一致）；K 敏感性分析补 K∈{3,5}
- 指标：task Spearman + within-source ρ + NDCG@K + normalized regret + top-1 + pairwise accuracy；source-group paired bootstrap 95% CI（2,000 iters，seed 20260816）+ Holm 多重比较
- 决策口径（承 Critic V6 spec D2）：hit@1/NDCG 优先，Spearman 双报告

## 2. 两种外部模式（R1）

| 模式 | 定义 | 现状 |
|---|---|---|
| **frozen zero-shot/delta** | 官方预训练权重直接打分，Δ = score(candidate) − score(source)，任务内排序 | **未跑（H2 测试核心）** |
| fine-tuned adapter | 官方架构 + 我方任务数据同预算训练 | 已有 5 行（APARENT 0.734 / Optimus 0.313 / FramePool 0.296 / UTR-LM 0.111 / RNA-FM 0.122、0.137）|

frozen 模式 adapter 规则（预注册，评估后零修改）：
- 输入：source_sequence 与 candidate_sequence 按各官方模型的 native 输入格式（one-hot / tokenizer）
- 长度截断：按各模型官方上限（Optimus 50nt 居中截断；FramePool 变长 native；APARENT 186nt polyA 窗口；Saluki 3'UTR 窗口按官方；UTR-LM native）
- 评分：模型原生输出标量；Δ = score(candidate) − score(source)；不引入任何我方训练
- 超参：无（zero-shot）；逐行声明权重来源 commit/revision

## 3. 公平预算协议（fine-tuned 模式与 Phase 2 对位，R1/R2/R7）

1. 对位预算：外部与我方 per-task 微调用相同任务 TRAIN split 数据、相同更新步数/前向预算；超参 = 官方推荐 + 同网格 HPO，逐行声明
2. 迁移数据等价性：我方两阶段的"多任务阶段"用其他 8 任务带标签 TRAIN split（supervised cross-endpoint transfer，**非**自监督预训练，论文措辞遵守）；不计入对位微调预算；**架构对架构条款**：最强外部架构同跑两阶段作对照行
3. seeds：底线最终判定 ≥3 seeds/方法（外部同等待遇）；CI + Holm
4. 生成线：matched 前向预算（前向等价物列）；closed-form 指标为主，evaluator uplift 辅助（STOP-05）

## 4. 覆盖边界（R5/R10，预注册）

- MPRAU（ENCSR854RUF，63% 数据）：无专用外部序列 scorer——结构性空白，预注册声明；Saluki-delta 为弱对照（3'UTR 降解口径 ≠ allelic skew 口径，标注口径差异）
- FunUV：分类口径（functional vs not）与本基准回归口径不同——登记为 related-work 对照，不入主表
- mRNABERT：我方 critic 的冻结 bottom encoder（bottom-6）；外部对照行用其官方 fine-tuned 形态时单列声明（R10）
- HALF_LIFE 两任务：标签 ICC≈0，一切结果附"受限于测量可重复性"归因，不参与方法优劣叙事

## 5. Saluki 移植条目（Task 3）

- 挂载点：`run_route2_external_prediction_baselines_v1.py`（现有 Optimus5Prime/FramePool native torch port 先例：`class Optimus5Prime` L140 / `class FramePool` L170 + h5 权重加载 `_load_array`）
- 资产：`external_model_assets/` 现无 saluki——需从官方仓库取权重（fetch 脚本先例：`fetch_route2_external_model_assets_v1.sh`）
- 作用域：GSE217518 两 region（3'UTR 稳定性族）+ MPRAU 弱对照
- 验收：单测（port 与官方参考输出一致容差内）+ 冒烟 + 移植记录入 ledger；失败登记不硬凑

## 6. 榜单冻结（Task 6）

- 汇总：5 外部模型 × 两模式 + native 对齐状态 + 内部 control + 内靶 + 我方冻结终态行（critic V5/V6、SetFlow f2、unguided；标注"迭代中"）
- 冻结即预注册靶子（含生成线 closed-form 靶），此后零修改
- 落盘：`docs/paper/route2_v332_baseline_leaderboard_v1.csv` + 本协议一起 commit

## 7. 今日已完成的对齐基线（引用，不需重跑）

见 `experiments/analysis_task1_alignment_20260902/`：critic V5/V6 × {GSE269595, GSE114002} 同口径行 + paired bootstrap vs APARENT/Optimus + 内靶 per-task（global_scaled 等 4 control 臂）+ MPRAU pair-mean（V5 0.1025 / V6-7815fdeb 0.0510）
