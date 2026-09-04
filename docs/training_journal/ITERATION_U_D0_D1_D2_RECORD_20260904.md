# A100 迭代记录 U：D0/D1/D2 补录（2026-09-04）

> 定位：SPECS_CRITIC_V6 遗留 Task 1——把 Iteration U/V/W 期间已定案但未入 A100 仓库留档的决策与事实补录进 W0 worktree。本文只登记既有事实（record-only），不含新结论、不改任何 gate/门槛。
> 出处：交接包 SPECS_CRITIC_V6/spec.md（D0/D1/D2 节，2026-09-01 冻结）；EditFlow 论文（Havasi et al., FAIR）探索结论 F13；`TRAINING_LOG_202609.md` 2026-09-01 ~ 09-03 各批次。
> 背景：V6 线已于 2026-09-03 终局负结果收官（MPRAU pair-mean ρ：V6 首训 0.0510、H3 三臂 0.0883/0.0839/0.0510、V7 0.0766，全部未过 V5 0.1025 主判据参照门）。本文为该收官前的决策链补档。

## 0. D0/D1/D2 冻结一览（2026-09-01）

- **D0（作用定位）**：Critic 作用优先级 B ≈ C > A（详见 §1）。
- **D1（学习目标）**：pair-mean 监督（六细胞共享效应为主目标）+ 细胞偏移辅助头（cell offset auxiliary head）+ per-task rank 变换。实现载体：`core/route2_xeditcritic_pair_mean_v6.py`（pair-mean 标签映射 / per-task rank-Gaussian / 扩展评估指标）+ V6 训练配置开关（默认 OFF，显式启用才生效）。
- **D2（评估口径）**：任务分层 + 分任务共主指标 + 天花板归一化——polyA 完成度 91% 达标，移出改进叙事；MPRAU 完成度 15% → 目标 40%；HALF_LIFE ICC≈0 归因为物理不可学，不进改进叙事。run_summary 扩展指标的任务分列落地见 2026-09-04 Task 3.1 修复（commit 41ec7e54，`extended_validation_metrics` 池化字段加 `_pooled_legacy` 后缀 + 按任务分列列）。

## 1. Critic 作用优先级声明（D0）

**定案：B ≈ C > A**（定案时间 2026-09-01，出处：交接包 SPECS_CRITIC_V6/spec.md D0 节）。

三作用定义：

| 作用 | 定义 | 备注 |
|---|---|---|
| **A = 引导打分器** | 为 SetFlow 生成线提供 potential 式率引导的打分函数 V(s) | V6 线负结果后，B2 guided 线的引导 critic 采用 V5 终态（最强多任务 critic） |
| **B = 实验优先排序器任务族** | 排序"下一个做什么实验/变体"的任务族：5'UTR MRL、3'UTR stability、polyA 使用、变体排序、跨研究迁移 | **B 是一个"作用"而非单一任务**——是任务族层面的作用定位 |
| **C = 跨 context 变体效应注释器** | 跨 context 的变体效应泛化注释 | context = 细胞系/研究/assay/时间点等维度，**不只跨细胞** |

优先级含义：B 与 C 是 critic 的主要价值来源（≈ 同级），A（引导打分器）次之。

## 2. EditFlow 论文探索结论（F13）

对 EditFlow 论文（Havasi et al., FAIR）的探索结论：

1. **flow 定义**：其 flow = CTMC over 变长序列（ins/del/sub 编辑速率），训练用 Discrete Flow Matching。
2. **guidance 全部是 classifier-free guidance（CFG）**：训练时 drop 条件、采样时组合条件/无条件 rate；论文结论为 naïve rate CFG 最优。
3. **全文无 reward/critic/potential**：论文不含任何 reward 模型、critic 或 potential 引导机制。
4. **Potential 式引导是本项目超出论文的自有设计**：U_q = U_p·e^{β[V(s′)−V(s)]}（Nisonoff ICLR 2025 率修正形式），为本项目在 SetFlow 生成线上引入的自有机制（现役 B2 guided 线）。
5. **CFG 定位为 Plan B**：触发条件 = 主判据连续未过门且架构侧无可试项。登记文档见 `docs/paper/route2_cfg_guidance_plan_b_registration_v1.md`（Task 7，2026-09-04 同批提交）。

## 3. 架构路线登记（V6 批复 → V7/V8 现状）

| 路线 | 内容 | 状态（截至 2026-09-04） |
|---|---|---|
| **V6 已批三项** | (1) pair-mean 监督 + 细胞偏移头；(2) per-task rank 变换；(3) LambdaRankIC 位移加权 pairwise（commit 7f01d17c） | 已执行；2026-09-03 终局负结果收官（H3 三臂 0.0883/0.0839/0.0510，全未过门；3-seeds confirmation 不启动） |
| **V7** | loss 机制梯度范数缩放（W4 机制） | 已执行；2026-09-03 终态 0.0766 负结果。**BiLB4MTL / DB-MTL 不再启动** |
| **V8（当时定义）** | LoRA 化容量方向；当时标注"需协议修订" | 2026-09-04 已重定义为**联合先验注入攻坚线**，正在另一条线推进（见后续增补，本文不展开） |
| **SSL** | 自监督预训练线 | 不进主线：判别证据链（Stage 0a/0b）指向外部库**监督**先验承载差距（280K 大库 frozen 即达 0.3132，架构 from-scratch 仅 0.0984），且 mRNABERT 预训练权重已作表征基座在用——新建自监督线无判别性证据支撑 |
| **SWA** | 随机权重平均 | 不进主线：属权重平均技巧，不改变学习目标与容量结构，作为主判据攻坚杠杆有限；降级为**零成本离线分析**（对既有终态 checkpoint 离线评估），2026-09-04 当日并行执行中 |

## 4. F1-F16 事实索引（浓缩表）

> 出处：交接包 SPECS_CRITIC_V6 F1-F16（逐条一行浓缩；详版见 spec 原文）。

| # | 事实 |
|---|---|
| F1 | MPRAU：2,008 变体 × 6 细胞 |
| F2 | 标签信噪天花板（split-half ρ）0.683 |
| F3 | V5 MPRAU pair-mean ρ 0.1025 ≈ 天花板完成度 15%（目标 40%） |
| F4 | 细胞盲 |
| F5 | HEK293FT 特异 |
| F6 | polyA 近饱和（完成度 91%），移出改进叙事 |
| F7 | MRL 瓶颈 |
| F8 | V5-f34 增益属噪声 |
| F9 | 宏平均稀释 |
| F10 | 6/9 任务每源仅 1 候选 |
| F11 | 无 per-pass 曲线（V6 已修复） |
| F12 | 天花板差异（任务间天花板不同，需归一化口径） |
| F13 | EditFlow 论文无 critic/reward/potential（详见 §2） |
| F14 | study_scale 捷径排除 + pair-mean 机制修正（Spearman-Brown 0.417→0.683） |
| F15 | 效应量长尾 10 桶 |
| F16 | 数据真实规模：43,730 独立序列 / 5-8k 独立效应观测 |

---
*登记：2026-09-04，W0 worktree `route_a_v3_w0_diagnosis_20260902`（SPECS_CRITIC_V6 遗留 Task 1）。*
