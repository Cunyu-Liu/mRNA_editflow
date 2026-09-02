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

### W0-MRL 终态收割（服务器时间 16:28，run_summary.json mtime 核验）

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

**W1-head（head_only，readout+effect_head 13,768,193 参数可训练）终态（服务器时间 17:36，mtime 核验）**：

| 指标 | 数值 |
|---|---|
| **MRL validation Spearman** | **0.1336** |
| standardized MAE | 0.6651 |

**结论：head_only 无增益**（vs V5 0.1354 持平）——冻结的多任务表征本身是瓶颈，只调头不够。与 W0（0.1987，从头单任务训练表征）对照：**任务专属表征适配是关键路径** → LoRA 臂是决定性测试。

**W1-lora（upper-6 LoRA rank16 α32 + head，GPU1）**：pass 3/8 健康（231 updates），实际服务器时间 18:11 终态（mtime 核验）。

### MRL 差距机制图景（中间版，W1-lora 出数前）

| 方法 | Spearman | 增量来源 |
|---|---|---|
| V5 多任务 | 0.1354 | 基线 |
| W1-head（V5 init + 只调头）| 0.1336 | +0（表征不动）|
| W0（同架构从头单任务）| 0.1987 | +0.063（任务专属训练全栈）|
| Optimus adapter | 0.3132 | 外部架构上限参照 |
| 内靶 control | 0.1192 | — |

待 W1-lora 补全"W1 init + 表征 LoRA 适配"行——若显著 >0.15 则表征适配有效，W2（容量解除）接力；若 ≈0.13 则 V5 初始化的表征对 MRL 有害，W0 路线（从头）优先。（注：本段为出数前的预判记录，终态数据见下节终版图景。）

### W1-lora 终态（服务器时间 18:11，mtime 核验）

| 指标 | 数值 |
|---|---|
| status | TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE |
| **MRL validation Spearman** | **0.1486** |
| standardized MAE | 0.6876 |
| W1 细节 | LoRA 24 Linears（rank16，1,474,560 参数）+ head；可训练 15,242,753 |

### MRL 完整机制图景（终版，2026-09-02）

| 方法 | Spearman | 判读 |
|---|---|---|
| 内靶 control | 0.1192 | 下界 |
| V5 多任务 | 0.1354 | 基线 |
| W1-head（V5 init + 只调头）| 0.1336 | +0.000——表征不动则无增益 |
| W1-lora（V5 init + LoRA 适配 + head）| 0.1486 | +0.013——LoRA 有限增益 |
| W0（同架构从头单任务 616 updates）| **0.1987** | +0.063——任务专属从头训练最优 |
| Optimus adapter（0.5M 参数 CNN）| **0.3132** | 外部对位靶 |

**W 阶梯裁决（MRL，2026-09-02）**：
1. **W1' 路线对 MRL 无效**：V5 多任务初始化不是好的 MRL 起点（W1-head ≈ W1-lora ≈ V5 << W0）——多任务表征对 MRL 有干扰，小容量适配救不回
2. **W0 路线有效但仍不足**：从头单任务 +0.063，但距 Optimus 仍差 0.115 → **架构/容量层面差距确认**（170M mRNABERT 系 vs 0.5M CNN 在 MRL 上落后）
3. **下一步（W2 设计，预注册后启动）**：(a) W0-continue 臂——W0 终态初始化 + 继续训练（检验预算是否是 W0 的限制；复用 W1' init 机制，仅需 config）；(b) W2 容量方向——per-task 容量解除/架构侧修订（edit_blocks 97M 是容量大头，MRL 或许需要不同分配）
4. polyA 判定带等 W0-polyA 终态（预计 ~22:00 服务器时间）

### 其他在途（18:57 快照）

- W0-polyA（GPU2）：pass 4/8（3,216 updates，~45 min/pass）
- V6 H3 三臂（GPU1/2/4）、V7（GPU3）、V5 b_fix1/b_fix3（GPU3/4）：运行中；2 小时巡检 cron 持续跟踪

**时间戳勘误（2026-09-02 晚）**：本日志初稿三处终态时间按会话时钟估计有误，已按服务器 run_summary.json mtime 修正为实测值：W0-MRL 16:28 / W1-head 17:36 / W1-lora 18:11。W0-polyA 18:57 快照为 pass 4/8（约 45 min/pass），预计 ~22:00 服务器时间终态。

## 2026-09-02（深夜，第三批：W0-continue + Saluki port + D3 增补阅读）

### 交接文档更新阅读（用户 17:41 更新）

1. **新增 `SPECS_BASELINE_LEADERBOARD/BENCHMARK_GUIDE.md`**：benchmark 全量参考（三层数据结构/14 研究/9 任务/指标口径/天花板归一化/权利边界/四轴差异）——作为 Task 6 榜单冻结与论文 Table 底稿依据。
2. **`SPECS_CRITIC_V6/spec.md` 2026-09-02 增补**：mRNABERT 跨任务微调提案（路线 A 外部大库预微调→LoRA / 路线 B 域内多任务）+ **决策点 D3 待拍板**。与今日 W 阶梯数据的交叉：W1-lora（0.1486）≈ 路线 B 近似模拟已测弱；W0（0.1987）确认架构/先验差距；Optimus 0.3132 的赢法 = 280K 大库先验 → **路线 A 是唯一未测试的差距来源**。D3 按 spec 硬约束等 V6 H3 裁决后呈报证据包拍板。

### W0-continue 臂发射（Addendum B，用户批准"现在启动"）

- **触发**：W0-MRL loss 未平台化（0.62→0.33）→ 预算限制假说
- **代码**：`full_continue` 模式加入 W1 policy（严格加载 W0 终态、全参数可训练、标准三组优化器 + 全新 cosine warm-restart）；commit 10fced68 + ae2b43d4
- **发射**：`w0_continue_mrl_gse114002`，GPU1，19:0x 服务器时间，pass 1 顺利完成；20:0x 至 pass 3/8；预计 ~21:00 终态
- **预注册判定带**（Addendum B）：≥0.25 预算是主要限制；0.20-0.25 部分限制；≈0.20 或回落 → 差距主因架构/先验（强化 D3 路线 A）

### Saluki native port（Task 3 主体完成）

- **资产**：51 折 × 2 模型权重 + params.json 已在 `external_model_assets/saluki/datasets/deeplearning/train_gru/`（Zenodo 本地 HTTP-range 提取）
- **架构考据**（以 h5 内嵌 model_config 为准，43 层）：conv k5 无偏置 → LN(0.007) → ReLU → 6×[conv k5 → Dropout0.3 → MaxPool2 → LN → ReLU] → GRU64(reset_after) → BN → ReLU → Dense64 → BN → ReLU → Dense1；输入 6 通道 [A,C,G,T/U,frame,splice5p]
- **两个移植陷阱已定案**：(1) Keras GRU 列序 [z,r,h] → PyTorch 行序 [r,z,n] 重排；(2) 指示通道极性——中间一度按 master 代码的 `tf.one_hot(v,1)` 推断为"普通位 1.0"，**官方测试集取证推翻**：2022 版为原始值（codon/splice 位=1.0，UTR-only 输入全 0），详见下方 parity 条目
- **第三个发现**：架构最小长度 = **320**（每池化块的 conv 也 −4；官方推理长度 12288 右侧零填充）——短 UTR 输入必须 pad
- **交付**：`core/route2_saluki_port_v1.py` + 5 单测（含 Keras reset_after GRU 单步公式精确匹配——最大风险点验证 ✓）+ GPU 冒烟脚本；commit c00dcce7
- **GPU 冒烟通过**：f7_c0/model0 真实权重，CUDA，长度 350/512/1000 输出有限
- **官方预测对齐（parity）——数值级达成（R6 黄金证据）**：通过官方 f7_c0 测试集取证定位三个根因并修复：(1) GRU `go_backwards=True`（rnann.py L44 + 保存 config 证实）——port 时间翻转，终态读序列起点（前向方向会把所有输入坍缩为常数 1.794）；(2) 2022 版指示通道为**原始值**（codon/splice 位=1.0）——master 仓库后期的 `tf.one_hot(v,1)` 反转与冻结权重不符（用原始极性复现官方发表 pearson 0.758/0.706）；(3) 官方 tfr 为 zlib 压缩 + model1 的输出层名为 dense_2（Keras 自动命名漂移）。**对齐结果（各 128 条，右填充 12288）**：model0 vs test0/preds max diff 0.0034 / mean 0.0015 / 100% ≤0.01；model1 vs test1/preds max 0.0012 / mean 0.0004 / 100% ≤0.01；交叉组合不匹配证实 model_M↔data_M↔test_M 配对。佐证：pearson(tfr targets, 官方 preds)=0.7626 vs 官方 acc.txt 0.758。commit 8e871062
- **过程中的方法学副产品**：Zenodo 18.75GB zip 的分块抓取器（512KB/块、逐块重试、ZIP64 解析——单流读取在 ~3MB 处必断）；无依赖 TFRecord/protobuf 解析器（含 zlib 流）

### 执行顺序确认（用户问询后拍板）

用户批准：W2（W0-continue）与 Saluki port 不等在途、立即启动；W0-polyA/V6/V5/V7 收割等各自终态。

### W0-continue 终态收割（服务器时间 20:08，run_summary mtime 核验）

| 指标 | 数值 |
|---|---|
| status | TERMINAL_XEDITCRITIC_V4_SCREEN_RUN_COMPLETE |
| **MRL validation Spearman** | **0.1800**（W0 = 0.1987 → **回落 −0.019**）|
| standardized MAE | 0.7135（W0 = 0.7063 → 略升）|
| 训练 dynamics | huber 持续下降 0.240→0.150；pairwise/soft-spearman 自 pass 5 起回升（0.366→0.405 / 0.164→0.186）——训练仍在拟合，验证已过拟合 |

**Addendum B 预注册判定带裁决**：0.1800 落入"≈0.20 或更低（验证回落）"带 → **预算不是 MRL 差距的限制因素**。

### MRL 差距最终机制图景（W 阶梯在本架构内搜索穷尽，2026-09-02 终版）

| 方法 | Spearman | 判读 |
|---|---|---|
| 内靶 control | 0.1192 | 下界 |
| V5 多任务 | 0.1354 | 基线 |
| W1-head（V5 init + 只调头）| 0.1336 | 表征不动则无增益 |
| W1-lora（V5 init + LoRA）| 0.1486 | 域内多任务先验弱（≈路线 B 模拟）|
| **W0-continue（W0 + 8 pass 预算延长）** | **0.1800 ↓** | 预算延长 → 验证回落（过拟合）|
| **W0（同架构从头单任务 616 updates）** | **0.1987** | **本架构 + 2,443 行数据的实际最优** |
| Optimus adapter（280K 大库先验 + 0.5M CNN）| **0.3132** | 外部对位靶 |

**W 阶梯 MRL 总结论（2026-09-02）**：
1. 域内先验（多任务 init/LoRA 适配）与训练预算（continue 回落）均已排除为 MRL 差距的可行来源
2. 从头单任务 0.1987 是该 170M 架构在此数据量上的天花板邻域；距 Optimus 0.3132 的 0.115 差距只能来自**先验来源（外部大库监督）或架构本身**
3. **D3 证据链闭合**：W1-lora（路线 B 模拟）弱 + W0-continue（预算）排除 + Optimus 赢法 = 280K 大库 → 路线 A（外部大库预微调 → LoRA 迁移）是唯一未测试且证据指向的路径。待 V6 H3 终态后按 spec 硬约束呈报 D3 拍板
4. 后续若 D3 批准路线 A：Step 1 需要 Sample 2019 280K 5'UTR 文库（GSE114002 原生配套，GEO 可得——A100 GitHub/NCBI 连通性需验证）
