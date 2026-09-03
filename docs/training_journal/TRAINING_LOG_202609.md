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

## 2026-09-03（凌晨-晨，第四批：一夜终态正式收割）

### 收割总览

一夜之间全部在途出数：W0-polyA（23:21 终态）、V6 H3 三臂（~01:30 终态）、SetFlow V5 12/12 验证收敛 + gate adjudicate（screen_gate.json 落盘）。V7 仍在跑（02:04 快照 pass 5/8）。

### ⚠️ 口径陷阱事件（先记录，本批最重要的一课）

正式裁决发现 **run_summary 的 `extended_validation_metrics.pair_mean_spearman` 是全任务池化值**（2,660 对 = MPRAU 2,008 变体 + polyA 321 + 其他 ~331），**不是预注册的 MPRAU 主判据**。监控 cron 曾把 H3 池化值 0.2334 与 V5 的 MPRAU 专口径 0.1025 对比得出"2.1-2.3× 提升"——**苹果比橘子，结论错误**。

口径验证（三重交叉确认）：
- 按变体口径（record_id 去 context 后缀分组，2,008 变体）算 V5 = 0.1025，与 Task 1 冻结参考 0.10254 **精确一致** ✓
- H3 λ=1.0 臂（= V6 首训同 seed 复跑）算出 0.0510，与 spec G6 记载的 V6_full 0.05105 **精确一致** ✓
- Task 1 的 critic_v5/critic_v6_full mprau_pair 段直接对上 ✓

**教训入档**：(1) 池化指标不得当任务专口径主判据使用；(2) run_summary 扩展指标的 pair_mean 字段需在下一 family 修复为按任务分列（已知问题，本批不做代码手术——正式裁决脚本 `analysis_w_ladder_adjudication_20260903/results.json` 为 canonical）；(3) 监控 cron prompt 已加口径警示。

### V6 H3 最终裁决（负结果，预注册主判据：MPRAU 变体 pair-mean ρ，paired bootstrap 2,000 iters vs V5）

| 臂 | MPRAU pair-mean ρ | Δ vs V5 0.1025 | 95% CI | 判定 |
|---|---|---|---|---|
| λ=0.5 | 0.0883 | −0.0142 | [−0.049, +0.019] | CI 跨零 → **未过门** |
| λ=0.75 | 0.0839 | −0.0186 | [−0.054, +0.017] | CI 跨零 → **未过门** |
| λ=1.0 | 0.0510 | −0.0515 | [−0.084, −0.019] | **显著更差** → 未过门 |

**V6 线终局结论**：V6 首训（0.0510，spec G6 已载）+ H3 λ 扫描（0.0883/0.0839/0.0510）全部未过 MPRAU 主判据门 → **loss 机制线（pair-mean 监督 + rank 变换 + LambdaRankIC + within-source 权重扫描）负结果收官**。按 spec："未过门 → 记录负结果，回到 D1 备选"。λ 趋势：λ=0.5 略好于 λ=1.0（+0.037），但都不及 V5 基线。3-seeds confirmation 不启动。天花板完成度：λ=0.5 = 12.9%（目标 40%）。

### SetFlow V5 screen gate：**PASS**（生成线重大正结果）

screen_gate.json（recovery 恢复流程自动 adjudicate，2026-09-03 凌晨落盘）：

| 臂 | Profile | B1 NLL（阈 2.068）| unique（阈 0.85）| legality | B1 判定 |
|---|---|---|---|---|---|
| b_arch1 | A1 | 2.3538 ✗ | 0.7063 ✗ | 1.0 | FAIL |
| b_fix1 | V4_FULL | 2.0884 ✗（差 0.020）| 0.8283 ✗ | 1.0 | FAIL |
| **b_fix2** | V4_FULL | **2.0670 ✓**（压线过）| **0.8572 ✓** | **1.0 ✓** | **PASS** |
| b_fix3 | V4_FULL | **2.0366**（最优）✓ | 0.7708 ✗ | 1.0 | FAIL（多样性不足）|

- 整体 status = **XEDITSETFLOW_V5_SCREEN_PASS**，stage_acceptance = BASE_MODEL_REPAIR_SELECTION，confirmation_authorized = **true**，protected reads = 0
- 选中 b_fix2 的 **pass-2 checkpoint**（NLL/unique 随训练恶化：pass 2→6 NLL 2.067→5.796——四臂共同模式，快速过拟合，B0 均未收敛；b_fix2 是早期 checkpoint 过门）
- b_fix3 训练质量最好（NLL 最低）但生成多样性不达标（unique 0.77）——修复方向间的 trade-off 实证
- **下一阶段（已授权）**：guided generation（Gate B2：guided vs unguided recovery Δ≥+0.05 CI 不跨零；B3：guided recovery ≥0.35）——需要冻结 critic 做 potential 式率修正；critic 候选 = V5 终态（V6 线已负，V5 为最强多任务 critic）

### W0-polyA 判定带（Task-1 同口径对齐评估，K=10）

| 指标 | W0-polyA | 判定带参照 |
|---|---|---|
| Spearman | **0.8142** | APARENT 0.7343（Δ+0.080，CI [+0.055, +0.106] 显著胜）|
| top-1 | 0.5080 | 过带线 0.55 / 可疑线 0.50 / APARENT 0.6011（Δ−0.093 CI [−0.134,−0.052] 显著负）|
| NDCG@10 | 0.8702 | 过带线 0.885 / APARENT 0.8906（Δ−0.020 CI [−0.033,−0.008] 显著负）|

**判定：MIXED 带**（top-1 0.508 落在 0.50-0.55 之间）。解读：单任务从头训练 0.5080 ≈ V5 多任务 0.5007 / V6 0.5482——**polyA 任务上架构无碍**（无可疑信号，接近饱和任务），对 APARENT 的决策口径差距与多任务线同构（配方/监督问题而非架构问题）。

### MRL W 阶梯全臂同口径对齐评估（GSE114002，K=10，统一口径终版）

| 方法 | Spearman | top-1 | NDCG@10 |
|---|---|---|---|
| 内靶 control | 0.1192 | — | — |
| V5 多任务 | 0.1354 | 0.3810 | — |
| W1-head | 0.1336 | 0.3961 | 0.8360 |
| W1-lora | 0.1486 | 0.4026 | 0.8385 |
| W0-continue | 0.1800 | 0.4286 | 0.8473 |
| **W0（从头单任务）** | **0.1987** | 0.3983 | 0.8475 |
| Optimus adapter | 0.3132 | 0.4069 | — |

（注意 top-1 口径上 W0-continue 0.4286 已超 Optimus 0.4069——Spearman 与决策口径的分歧，与 polyA 的模式同构，入档备查。）

### D3 决策包（证据链闭合，待用户拍板）

四条独立证据全部指向同一结论：
1. **V6/H3 loss 机制线负结果**（本批）：pair-mean/rank/λ 扫描全未过门
2. **W1' 域内先验弱**（0.1336/0.1486 ≈ V5 基线）：多任务初始化不解决 MRL
3. **W0 预算限制排除**（continue 0.1800 回落）：从头训练 0.1987 是本架构天花板邻域
4. **Optimus 的赢法 = 280K 外部大库监督先验**——唯一未测试的差距来源

→ **路线 A（外部大库预微调 → per-task LoRA 迁移）是证据指向的路径**（spec 2026-09-02 增补 D3 选项 1）。V7 若也负，则 loss 线证据完全闭合。待用户拍板后起草路线 A 预注册（Step 1 需 Sample 2019 280K 5'UTR 文库，GEO 可得性待验证）。

### 当前在途

- V7 v7_full：pass 5/8（GPU3），预计今日内终态——loss 线最后一块拼图
- GPU1/2/4 已释放（V6 H3 三臂 + SetFlow 验证完成）

## 2026-09-03（晨，第五批：D3 拍板 + Stage 0a 判别实验）

### D3 拍板与路线 A 前置分析

用户批准 A+B，但要求先论证路线 A 的必要性与确定性（"不清楚用这么大量数据集微调会有什么后果"）。交付 `docs/paper/route2_route_a_necessity_certainty_analysis.md`（commit 00d03adc）：五项后果（泄漏硬门/H2 口径风险/遗忘→LoRA/措辞/算力）+ 分阶段 GO/NO-GO 设计（Stage 0 判别先行）。

### Stage 0a：frozen-Optimus/FramePool delta（决定性结果）

**方法**：官方 280K 预训练权重直接打分 source/candidate（零任务微调），delta 评估同口径（GSE114002 VALIDATION，K=10）。

| 模型 | frozen delta Spearman | 原"adapter"行 |
|---|---|---|
| Optimus | **0.3132** | 0.3132（完全一致）|
| FramePool | **0.2956** | 0.2956（完全一致）|

**验证**：spearman(frozen_delta, HPO adapter 预测) = **1.000000**，pearson = 1.0——Track B 的"adapter"行实为 frozen 权重 delta + 线性校准（对 Spearman 不变）。**榜单标签需修正**（Task 6 时改为 frozen-delta 口径）。

**科学结论（路线 A 确定性跃升）**：
1. **280K 外部大库先验单独（零任务训练）= 0.3132 全部性能**——MRL 差距的主因确认是外部库先验，不是任务微调
2. 按预注册决策规则（0a ≥ 0.15 → GO）：**路线 A 高信心 GO**
3. 路线 A 的核心问题精确化为：**mRNABERT 吃同样 280K 监督能否达到 CNN 同等水平**——Step 1 产物可直接用同一 frozen-delta 协议评估
4. H2 假说获重要数据点：绝对 MRL 预测器的差分直接携带 Δy 排序信号（0.3132）

### 并行推进状态

- Saluki frozen-delta GSE217518 全量（100 checkpoint）在跑（GPU4，PID 811572）
- SetFlow guided generation 侦察完成：guidance 核心（potential_guided_rates_v3/v4 + SMC）与 critic 接口（FrozenRoute2MRNABERTCritic）现成，需写 SetFlow V5 guided runner（采样器无 critic 钩子）；b_fix2 pass_2.pt 就位
- Stage 0b（Optimus from-scratch 对照）与 Stage 0c（280K 泄漏审计）待做

## 2026-09-03（上午，第六批：Stage 0b/0c 完成 + 路线 A Stage 1 发射）

### Stage 0b：Optimus 架构 from-scratch 对照（GPU2，~10 min）

同 Optimus5Prime 架构随机初始化，仅在 GSE114002 TRAIN（2,443 对，绝对端点回归）上训练 300 epochs：

| 方法 | Spearman | 结论 |
|---|---|---|
| **Optimus 架构 from-scratch（2.4K 任务数据）** | **0.0984** | 低于内靶 control 0.1192 |
| frozen-Optimus（280K 先验，零任务训练）| 0.3132 | 全部性能来自先验 |
| W0（170M critic from-scratch 同数据）| 0.1987 | 大架构从 2.4K 提取更多 |

**判别结论**：架构单独买不到性能（0.098 << 0.313），280K 先验承载全部——按预注册决策规则（0a=0.3132≥0.15 且 0b=0.0984<0.20）**路线 A GO 确认（高信心）**。副产品洞见：W0 从头 0.1987 > Optimus 从头 0.0984，说明 170M 表征从小数据提取能力强于 0.5M CNN——喂上 280K 后有超越 0.3132 的可能。

### Stage 0c：280K 文库获取 + 泄漏审计（硬门通过）

- **获取**：GEO GSE114002 supplementary egfp_unmod_1/2（GSM3130435/36，95MB gz）。NCBI 单流限速 24KB/s → 写并行分块下载器（16×2MB range 请求，~7 分钟完成）
- **数据结构**：CSV 每行 (utr 50nt, 14 个分数占比, counts, **rl 列 = 预计算 MRL**)——无需从 counts 重建
- **规模**：两重复并集 677,608 条序列
- **泄漏审计（3-block 鸽笼 seeding，≤2 mismatches/50-mer = ≥96% 同源）**：677,608 文库序列 vs 4,858 条受保护序列（GSE114002 全 split source+candidate）→ **flagged = 0**，随机 50-mer 与人源 UTR 窗口零碰撞。审计 JSON 落盘 `xeditcritic_route_a/280k_prefinetune_20260903/leakage_audit.json`

### 路线 A Stage 1：mRNABERT 280K LoRA 预微调（发射）

- **配置**：mRNABERT 全 12 层 + LoRA r16 α32 dropout0.05（Wqkv/attn-out/mlp-gated/mlp-wo，48 Linears）+ masked mean pool + linear head；可训练 3,046,657 参数
- **目标**：677,608 条 (utr → standardized rl) 监督回归（supervised domain-library pre-finetuning，措辞纪律遵守）
- **训练**：2 epochs / batch 128 / AdamW lr1e-4 wd1e-4 / cosine+5% warmup / bf16 autocast / seed 20260903；GPU2
- **修复记录**：首发射崩溃（`AutoModel.from_pretrained` 与自定义 ALiBi 代码 meta 初始化不兼容）→ 改用工作管线的加载模式（`from_config` + 手动加载 `pytorch_model.bin` 剥 `bert.` 前缀 + flash_attn 置 None）后正常
- **进度**：MSE 1.02 → 0.67（step 800/10.6K），GPU2 73% 利用率，预计 ~50 min
- **评估（训练后自动）**：frozen-delta 协议（GSE114002 VALIDATION，K=10）——直接对标 frozen-Optimus 0.3132

### SetFlow guided generation（B2）委托执行中

后台 agent 在 setflow worktree 实现 V5 guided runner（b_fix2 pass_2 + FrozenRoute2MRNABERTCritic potential 引导 + B2 adjudication），完成后收割。

### V7

pass 7/8，即将终态——监控 cron 跟踪。
## 2026-09-03（午后，第七批：full-FT 消融收割 + B2 全量发射 + full-FT V2 预算扩展）

### Full-FT 消融终态收割（04:11 出数，本轮入档）

Route A Step-1 全参消融臂（113,389,825 可训练参数，同 280K 清洗文库 677,608 条，2 epochs，lr 2e-5，seed 20260903）frozen-delta 评估（GSE114002 VALIDATION，K=10）：

| 方法 | Spearman | top-1 | NDCG@10 |
|---|---|---|---|
| critic V5 多任务 | 0.1354 | 0.3810 | — |
| W0 从头单任务 | 0.1987 | 0.3983 | 0.8475 |
| Route A Step-1 LoRA（2ep）| 0.2470 | — | — |
| **Route A full-FT（2ep）** | **0.2555** | **0.4351** | 0.8550 |
| frozen FramePool（280K 先验）| 0.2956 | 0.2485 | 0.8647 |
| frozen Optimus（280K 先验）| 0.3132 | 0.4069 | 0.8655 |

**裁决**：(1) full-FT > LoRA（+0.0085）——容量限制部分成立但幅度小；(2) **top-1 0.4351 超 frozen-Optimus 0.4069（决策口径首次超越外部最强行）**，NDCG 0.8550 vs 0.8655 微差、Spearman 差 0.058——三口径分裂，按预注册决策口径（hit@1/NDCG 优先）为混合结果，不宣称过线；(3) 剩余 Spearman 差距归因候选 = 收敛不足（2ep）/架构归纳偏置。

### Full-FT V2 预算扩展发射（12:23，GPU5，PID 1933044）

- **预注册**：`run_route2_mrnabert_280k_fullft_v2.py`——2ep→6ep 预算扩展，其余全部相同（seed 20260903 / batch 128 / lr 2e-5 / cosine+5% warmup / bf16）；**主判据 = FINAL-EPOCH-6-FIXED frozen-delta**（防 peak-picking，沿 FINAL_PASS_8_FIXED 先例）；每 epoch checkpoint + 每 epoch frozen-delta 诊断曲线（收敛归因用）
- 动机：full-FT 2ep 已是房内最佳 0.2555 且训练 loss 无平台证据；测试剩余 0.058 Spearman 差距中收敛不足成分
- 产物目录：`experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/`（含 training_losses.jsonl 每 50 步 + epoch_frozen_delta_metrics.jsonl）
- 预计 ~6h 终态（每 epoch ~45min + 评估）

### SetFlow V5 B2 全量发射（12:20，GPU1，PID 1909082）

- 冒烟（8 源，guided wall 550s）通过后发射全量：`b2_full_891`（891 源 × 双臂 unguided+guided × 32 trajectories/source，critic = V5 终态 frozen，β = G0 冻结奖励策略，seed 链沿 screen）
- git HEAD c30904a1（setflow worktree，V5 guided runner + V5 critic potential 支持）
- 预计 ~17-20h（guided 臂为主）
- Gate B2 判定：Δrecovery ≥ +0.05 且 CI 不跨零且 hit@1 不劣化；B3：guided recovery ≥ 0.35
- 中途 sanity：self-pair adjudication（guided=unguided）delta=0 已确认（`smoke_b2_adjudication_self_pair.json` / `b2_baseline_self_pair_full_891.json`）
- **注意**：首发射因预建输出目录被 runner 拒绝（要求目录不存在）——重启一次，无科学影响

### 监控与治理（本轮）

- 服务器监控 cron（每 2h）原指向 v403/s1 旧 runtime，本轮更新为当前在途（fullft_v2 日志 + B2 运行 + GPU 快照）
- 客户端（本地 TRAE）设置定时监控任务跟踪两条训练线
## 2026-09-03（下午，第八批：Full-FT V2 终态收割 + 显著性裁决）

### Full-FT V2（6-epoch 预算扩展）终态收割

预注册主判据 FINAL-EPOCH-6-FIXED（GSE114002 VALIDATION，K=10 frozen-delta，seed 20260903）：

| epoch | Spearman | top-1 | NDCG@10 |
|---|---|---|---|
| 1 | 0.2705 | 0.4502 | 0.8641 |
| 2 | 0.2846 | 0.4156 | 0.8559 |
| 3 | 0.3292 | 0.4524 | 0.8641 |
| 4 | 0.3169 | 0.4177 | 0.8601 |
| 5 | 0.3169 | 0.4567 | 0.8597 |
| **6（主判据）** | **0.3198** | **0.4481** | **0.8613** |

### Paired bootstrap 显著性裁决（source-group 重采样 2,000 iters，seed 20260816）

**vs frozen-Optimus（0.3132）**：
- ΔSpearman +0.0066，CI [−0.0465, +0.0588] 跨零 → **统计平局**
- Δtop-1 +0.0411，CI [−0.0216, +0.1104] 跨零 → 平局
- ΔNDCG −0.0042，CI [−0.0198, +0.0124] 跨零 → 平局
- **判定：三口径统计不可区分（点估计两胜一平）——房内模型首次追平最强外部行；不得宣称显著超越**（730-record validation 对 ~0.05 级 delta 检验力不足，与 Stage 2 adjudication 结论一致）

**vs critic V5（0.1354）**：
- ΔSpearman **+0.1844，CI [+0.0874, +0.2774] 不跨零 → 显著提升 ✓**
- ΔNDCG +0.0263，CI [+0.0031, +0.0498] 不跨零 → 显著提升 ✓
- Δtop-1 +0.0671，CI [−0.0130, +0.1494] 跨零
- **判定：对自家多任务前身的提升统计显著**

### 结论（W 阶梯 MRL 侧阶段收割）

1. **MRL 差距从 −0.178（V5 vs Optimus，显著落后）→ 统计平局点估计领先（full-FT V2 6ep）**：Route A（外部 280K 大库监督预微调 + 零任务训练）机制成立，+0.184 Spearman 提升 CI 不跨零
2. 收敛归因：epoch 3 已达峰值域（0.3292），epoch 4-6 稳定在 0.317-0.320 平台——6ep 足够收敛，剩余 vs Optimus 的点估计差为噪声/检验力问题
3. 底线判定（spec v5.1：必须超过所有 baseline）：MRL 行**未完全过线**（平局非超越）——需多 seed 平均或更大 validation 才能分辨；但"从显著落后到统计平局+点估计领先"是 W 阶梯实质进展，作为本周一版结果的 MRL 主体
4. 证据：`experiments/analysis_fullft_v2_adjudication_20260903/{adjudication_results.json, vs_critic_v5.json}`
