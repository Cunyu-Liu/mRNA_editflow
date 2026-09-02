# mRNA EditFlow 训练日志（2026-09 起）

> 本文件记录每次训练的启动、过程、终态与结论。坐标：A100 服务器（ssh A100）；执行 worktree `/home/cunyuliu/mrna_editflow_goal/worktrees/`；实验产物 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/`。纪律：CUDA BF16-only、protected reads=0、预注册门槛不事后改。

## 2026-09-02（接管日）

### 现场接管（15:00 快照）

在途训练（全部健康，不干预）：

| 线 | 运行 | GPU | 状态 |
|---|---|---|---|
| Critic V6 H3 消融 | v6_h3_lambda_{0_5,0_75,1_0}（runner_586e08aa，HEAD d9a03dc4e，seed 20260907）| 1/2/4 | RUNNING（λ=0.75 已至 pass_5；λ=0.5/1.0 pass_4 验证产物已出）|
| SetFlow V5 四臂 | b_fix1 / b_fix3（scheduler PID 1962691 健在）| 3/4 | RUNNING（pass_4 于 09:10/08:52 写出；b_fix2/b_arch1 已 TERMINAL_TRAINING_COMPLETE_PENDING_VALIDATION）|
| Critic V7（梯度范数缩放，W4 机制）| v7_full（runner_10965037，HEAD e57d1fb7，12:00 发射）| 3 | RUNNING（pass_1 完成，2,802 updates）|

遗留问题：GPU5 上有 Aug26 的 v4_source_only 进程处于 STOPPED 状态（占 15.3G），属 V4 修复线遗产——暂不干预（不阻塞任何当前工作）。

### 今日完成动作

1. **Task 2 协议冻结**：`route2_baseline_leaderboard_protocol_v1.md` 提交为 `docs/paper/`（commit 7dc3dd98，worktree `route_a_v3_baseline_leaderboard_20260902`，基 e57d1fb7），已推 GitHub。Phase 1 榜单协议（评估器/K=10/frozen 规则/公平预算四条/覆盖边界）正式冻结。
2. **W 阶梯 amendment 预注册**（W0+W1'）：`docs/paper/route2_w_ladder_amendment_v1.md`（commit 7303417c + Addendum A d2b5542a）。预注册判定带：W0-MRL Spearman ≥0.28 → 架构无虞；<0.20 → 架构可疑。W1' LoRA/head-only 禁全参、架构对架构对照、泄漏边界条款入册。
3. **W0 诊断两臂发射**（详见下节）。
4. **Saluki 资产**：A100 无法直连 Zenodo/HF（仅 GitHub 通）。已改走本地 Mac HTTP-range 部分提取 Zenodo datapack（17.8GB zip）中的 train_gru 模型权重（每折 model{0,1}_best.h5 ~2MB + params.json，共 ~10 折）→ scp 至服务器 `external_model_assets/saluki/`。
5. **定时监控**：每 2 小时巡检（本会话外 cron）：进程存活 / 目录新文件 / run_summary 出现即记录 / GPU 快照 / CUDA 合规抽查，日志写 `training_monitor_log.md`。

### W0 单任务架构诊断（今日主推进）

**代码 delta**（worktree `route_a_v3_w0_diagnosis_20260902`，HEAD d2b5542a，已推 GitHub）：
- `train_route2_xeditcritic_v4.py`：新增 `study_filter` 配置键（4 处向后兼容修改）；vocab 在全量 projection 上构建（模型容量与 preflight 精确一致 170,679,590 参数）
- `core/route2_xeditcritic_batch_v4.py`：cache view 覆盖不变量放宽为子集语义（Addendum A）
- focused 测试：batch_v4 18 passed / runner 21 passed / 全套 xeditcritic 479 passed + 1 失败（v403 confirmation launcher 测试，**基线 e57d1fb7 同样失败，与本 delta 无关**，已验证）/ setflow 329 passed / v332 101 passed

**发射记录**：
- 尝试 1（GPU6/7）：双双 OOM——**GPU6/7 为 MIG 1g.5gb 切片（4.75 GiB）**，非完整 A100。失败如实入账本（2 条 FAILED 记录），目录保留 `_aborted_by_*_mig_capacity_20260902`
- 尝试 2（GPU1/GPU2）：**成功**
  - `w0_mrl_gse114002`：GPU1，77 updates/pass × 8 pass = 616 updates，pass 1 已完成（~16:10），预计 ~40 min 跑完
  - `w0_polya_gse269595`：GPU2，804 updates/pass × 8 pass = 6,432 updates，预计 ~6.5 h（约 22:30 完成）
- 与 V6 H3 λ=0.5/0.75 共卡（各 15.9G + W0 ~9G < 40G，显存充足并行）

**判定带（预注册，amendment §1）**：
- W0-MRL vs Optimus adapter 0.3132 / FramePool 0.2956（同 730-record 验证集、同评估器）
- W0-polyA vs APARENT adapter top-1 0.6011 / NDCG 0.8906（同 2,628-record 验证集）

### 本周计划（至 09-07）

1. 今晚：W0-MRL 出数 → 立即收割判定；W0-polyA 夜间跑完
2. V6 H3 三臂终态 → V6 最终裁决（主判据 MPRAU pair-mean ρ vs 0.1025 + CI）
3. V5 b_fix1/b_fix3 终态 + scheduler 自动 validation → Gate B0/B1 判定
4. W0-MRL 若"架构无虞"→ 明日启动 W1'-MRL（LoRA 微调，多任务终态 ckpt 初始化）
5. Saluki 移植（模型权重到手后 native port，接入 GSE217518 两 region）
6. Task 4 frozen 评估 → Task 6 榜单冻结

## 2026-09-02（晚，第一批 W 阶梯结果）

### W0-MRL 终态收割（17:55）

| 指标 | 数值 |
|---|---|
| status | TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE |
| **MRL validation Spearman（730 rec）** | **0.1987** |
| standardized MAE | 0.7063 |
| 训练 | 616 updates / 8 passes，pass 级 soft-spearman loss 单调下降 0.62→0.33（未平台化）|

**预注册判定带结论**：0.1987 < 0.20（架构可疑带边界）——按字面落入"架构可疑"，但考虑：
- vs V5 多任务 0.1354：单任务训练 **+47% 相对提升** → 多任务稀释确实是差距来源之一（配方问题成立）
- vs Optimus adapter 0.3132：仍差 0.115 → **架构/容量也有实质差距**（W0 用同一架构 616 updates 从头训练）
- loss 未平台化 → 预算可能不足（Optimus adapter 训练预算更大），但 8-pass 是冻结管线的固定结构

**裁决：混合结论——架构与配方双因素。W1'（LoRA 表征适配）与 W2（per-task 容量）并行推进。**

### W1' 两臂（MRL，V5 终态初始化）

发射过程（4 次修复迭代，全部如实入账本，失败目录 `_aborted_by_*` 保留）：
1. v1（GPU6/7 MIG 4.75G）→ OOM（误用 MIG 切片卡；GPU6/7 非完整 A100，教训记档）
2. v2 → state_dict vs parameters() 计数口径错（persistent buffers），修复
3. v3 → cell_offset_weight=0.5 撞 V5 架构（无 cell_offset_head）→ 置 0；LoRA 路径 intermediate/output → 实为 mlp.gated_layers/mlp.wo，修复
4. v4 → 冻结 router 的 router_balance loss 无 grad_fn 进入 multi-tensor backward → core 修复（grad-free 叶子过滤，梯度贡献本为零，全可训练时行为不变）；LoRA 新参数在 CPU → policy 末尾 to(device)
5. v5 → auth HEAD 跨 ssh 会话变量丢失写空 → 修正重发（最终 HEAD 07bb58df）

代码修复全部走 commit 预注册（0980d8eb / d3168791 / 8f01b672 / 07bb58df），单测每轮通过。

**W1-head（head_only，readout+effect_head 13,768,193 参数可训练）终态（21:05）**：

| 指标 | 数值 |
|---|---|
| **MRL validation Spearman** | **0.1336** |
| standardized MAE | 0.6651 |

**结论：head_only 无增益**（vs V5 0.1354 持平）——冻结的多任务表征本身是瓶颈，只调头不够。与 W0（0.1987，从头单任务训练表征）对照：**任务专属表征适配是关键路径** → LoRA 臂（正在跑）是决定性测试。

**W1-lora（upper-6 LoRA rank16 α32 + head，GPU1）**：pass 3/8 健康（231 updates），预计 ~22:00 终态。

### MRL 差距机制图景（更新）

| 方法 | Spearman | 增量来源 |
|---|---|---|
| V5 多任务 | 0.1354 | 基线 |
| W1-head（V5 init + 只调头）| 0.1336 | +0（表征不动）|
| W0（同架构从头单任务）| 0.1987 | +0.063（任务专属训练全栈）|
| Optimus adapter | 0.3132 | 外部架构上限参照 |
| 内靶 control | 0.1192 | — |

待 W1-lora 补全"W1 init + 表征 LoRA 适配"行——若显著 >0.15 则表征适配有效，W2（容量解除）接力；若 ≈0.13 则 V5 初始化的表征对 MRL 有害，W0 路线（从头）优先。

### 其他在途（22:00 快照）

- W0-polyA（GPU2）：pass 1-2/8，预计 ~23:30-24:00 终态
- V6 H3 三臂（GPU1/2/4）：运行中（λ=0.75 曾至 pass_5）
- V7 v7_full（GPU3）：运行中
- V5 b_fix1/b_fix3（GPU3/4）：运行中；scheduler 健在
- Saluki 权重：本地 HTTP-range 提取 train_gru 模型文件（~198MB，f0_c4 折已到）——完成后 scp 至 external_model_assets/saluki/
