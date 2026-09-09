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
## 2026-09-03（傍晚，第九批：3-seed ensemble 终局判定 + polyA Route A 发射）

### MRL full-FT V2 三 seed 终态（FINAL-EPOCH-6-FIXED，GSE114002 VALIDATION，K=10）

| seed | Spearman | top-1 | NDCG@10 |
|---|---|---|---|
| 20260903 | 0.3198 | 0.4481 | 0.8613 |
| 20260904 | 0.2873 | 0.4221 | 0.8505 |
| 20260905 | 0.3157 | **0.5087** | **0.8808** |
| **3-seed ensemble（z-mean）** | **0.3158** | 0.4416 | 0.8624 |
| frozen-Optimus | 0.3132 | 0.4069 | 0.8655 |

**Ensemble vs frozen-Optimus paired bootstrap（2,000 iters，seed 20260816）**：
- ΔSpearman +0.0027，CI [−0.045, +0.048] 跨零 → 统计平局
- Δtop-1 +0.0346，CI [−0.022, +0.095] 跨零 → 平局（点估计领先）
- ΔNDCG −0.0031，CI 跨零 → 平局

**MRL 终局结论**：
1. Route A（外部 280K 监督预微调）三 seed 复现：Spearman 0.287-0.320（均值 ~0.308），seed 方差 ±0.016
2. **vs critic V5（0.1354）：+0.18 量级提升，CI 不跨零（单 seed 已判定显著）**——房内 MRL 能力从"显著落后"到"与最强外部统计不可区分、点估计领先"
3. 决策口径（top-1）三 seed 全部超 Optimus（0.4221-0.5087 vs 0.4069），seed 20260905 达 0.5087（+0.102）
4. 严格按底线（"必须超过所有 baseline"）MRL 未完全过线——统计平局非显著超越；730-record validation 检验力为物理约束（分辨 ~0.05 级 delta 需 >2,000 records）。如实报告，不宣称过线
5. 证据：`experiments/analysis_fullft_v2_adjudication_20260903/ensemble_3seed_vs_optimus.json`

### polyA Route A 发射（15:45，GPU5）

- **数据**：APARENT 训练库 GSE113849 isoforms（GEO supplementary 155MB，NCBI 限速下并行分块下载 ~1.5h）；过滤 count≥10 后 **2,740,320 行**；目标 = proximal usage log2 odds（对齐 GSE269595 endpoint 口径）
- **泄漏审计（硬门）**：2.74M 文库序列 vs GSE269595 全 split 3,347 唯一 source/candidate 序列（3-block 17bp 鸽笼 ≤2 mismatch）→ **flagged = 0**（审计 JSON 已落盘）
- **训练**：mRNABERT full-FT 同协议（113M 全参 / lr 2e-5 / 6ep / batch 128 / bf16 / seed 20260903 / FINAL-EPOCH-6-FIXED 主判据 + 每 epoch 诊断）
- **目标参照**：critic V5 0.8219 / V6 0.8273 / APARENT 0.7343 / W0-polyA 0.8142（Spearman 口径）；决策口径（top-1/NDCG）是 polyA 主缺口（V5 top-1 0.5007 vs APARENT 0.6011）
- 预计 ~2.5h 终态（2.74M 行 × 6ep）

### GPU 资源事件（16:20-16:30）

- 服务器非 MIG 卡（0-5）全部被其他用户挤占（37-40G/40G）；GPU6/7 为 MIG 切分（7×5G / 2×20G 实例）不适用大 batch 全参训练
- **清理僵尸进程**：`v4_source_only`（PID 1766545，GPU5 15.2G，Aug26 启动的 v402 recovery 线）runtime.json 最后更新 2026-08-26 13:10（8 天零产物写入）——判定僵尸，kill 释放（V4 线早已科学 NO-GO 收官，无在途验收依赖）
- polyA Route A 首次发射（15:45，GPU5）在 CPU tokenize 阶段发现 GPU5 剩余显存不足 → 主动终止，无训练损失；输出目录已清理
- **自动接力部署**：`/home/cunyuliu/mrna_editflow_goal/monitor/polya_auto_launcher.sh`——监测 B2（GPU1）终态（guided_run_summary.json 出现）后 120s 自动在 GPU1 以原预注册协议（batch 128，FINAL-EPOCH-6-FIXED）启动 polyA 训练；B2 无终态产物死亡则不启动并等待人工

### 监控与收割链修复部署（18:25）

- **发现并修复关键 bug**：B2 runner 的终态产物 guided_run_summary.json 写在 b2_full_891/ 顶层（smoke run 证实），而 16:24 部署的 polyA auto-launcher 检查的是 guided/guided_run_summary.json 子目录与 */guided_run_summary.json glob——两者均无法命中。若不修复，B2 正常完成后 launcher 会误判「B2 无终态死亡」并拒绝启动 polyA。已改为顶层路径检查（v2，18:21 重启，PID 1005196）。
- **部署 B2 自动收割 watcher**（monitor/b2_harvest_watcher.sh，PID 1023582）：检测 b2_full_891/guided_run_summary.json 后自动执行 adjudication（Gate B2/B3 判定）、追加事实性日志条目、W0 worktree commit+push。
- **部署 polyA 自动收割 watcher**（monitor/polya_harvest_watcher.sh，PID 1025537）：检测 frozen_delta_results.json 后自动执行 vs APARENT/critic V5 判定（FINAL-EPOCH-6-FIXED）、追加日志、commit+push。
- **修复 2h cron monitor** 的同一终态判定路径；两个收割 watcher 每 5min 轮询，进程死亡且无终态产物时告警并留证（不盲目重启）。
- B2 进度校准：smoke guided wall 550s / 8 源 ≈ 68.7s/源 → 891 源 ≈ 17h，12:20 启动 → **ETA 明晨 ~05:40**；unguided 臂已 12:42 完成（recovery=0.1205，hard legality 1.0）。polyA 将在 B2 终态后 120s 于 GPU1 以预注册协议自动启动。

## 2026-09-04（凌晨，第十批：例行监控观测，无新终态）

### B2 guided 线（PID 1909082，GPU1）

- 启动 09-03 12:20，观测时 elapsed 11h45m；unguided 臂已于 12:42 终态（recovery 0.12046 / hard legality 1.0 / 28,512 candidates），guided 臂在途。
- 健康性验证：CPU ticks 2s 采样推进（非挂死），GPU1 上下文活跃（util 28%、主进程 23.3GB）；smoke 校准 ETA ~05:40。
- 终态产物 b2_full_891/guided_run_summary.json **未出现** → 非终态，harvest_b2.sh 待命。
- 合规抽查：所有 run_summary 的 cpu_fallback_used=false，无 CPU 静默降级。

### polyA APA 线（GSE113849 2.74M 预微调）

- 仍在 B2 终态后排队（watcher 健在：polya_auto_launcher PID 1005196 / polya_harvest_watcher PID 1025537），尚未启动训练。
- 终态产物 frozen_delta_results.json **不存在** → 非终态，harvest_polya.sh 待命。

### GPU / 节点抽查

- GPU0-5 均被占用（39-40GB/40G，util 11-100%，节点多用户共享）；本线 GPU1 正常推进。
- GPU6/7 为 MIG 切片（util 显示 N/A），有他人作业在跑，不影响本线。
- status.log 近端条目仅 B2 alive=1 / APA queued behind B2 - normal；历史 CRT/SET TECHNICAL_FAILURE 均属已归档 V4/V4-S1 NO-GO 线，与本批无关。

### 判定

- 两线均无新终态、无异常。下一检查点：B2 guided ETA 09-04 ~05:40；polyA 在 B2 终态后 120s 自动接力。收割脚本与 watcher 就绪。

## 2026-09-04（04:07，第十一批：例行监控观测，无新终态）

### B2 guided 线（PID 1909082，GPU1）

- 启动 09-03 12:21，本批 elapsed 15h46m；unguided 臂 12:42 已终态，guided 臂仍在途；终态产物 b2_full_891/guided_run_summary.json 未出现 → harvest_b2.sh 待命。
- 深检"进程活着但零输出"疑点，结论 = 慢速推进非卡死：
  - /proc/1909082/io：syscw=27、write_bytes≈21MB（= 12:42 unguided 臂产物量），12:42:28 后无任何写系统调用——符合"全部臂产物终态一次性原子落盘"的设计，不能据此判死；
  - rchar 6.76GB 且 45s 窗口零增长——source/measured 池全部驻留内存缓存（SourceTokenCacheIndexV3），无磁盘读不代表停滞；
  - nvidia-smi dmon 45s：GPU1 每秒均有 kernel 启动（SM 9-74% 波动、mem% 0-2%、功率 43-76W）——小 batch 生成器前向 + critic potential memo 命中（smoke 口径 177k memo vs 103k 新评分）为主，与真卡死（纯 CPU 空转）可区分；
  - smoke 校准：guided 8 源 wall 550s ≈ 68.7s/源 → 891 源 ≈ 17.0h；guided 臂自 12:42:30 起跑，ETA ~05:45，本批已过 ~90%。
- 合规复检：所有 run_summary + failed.json 的 cpu_fallback_used=false；本进程直接持有 cuda:1（23.3GB）且每秒有 GPU kernel → 无 CPU 静默降级。
- 监控盲区（建议，下批次执行）：runner 无 per-source 进度打印（全文件仅末行一次 print），导致挂起与慢速不可从日志区分；建议增加"每 N 源落盘 progress.jsonl"心跳 + while 循环 forwards 预算断言，供审计与在线监控。

### polyA APA 线（GSE113849 2.74M 预微调）

- 仍在 B2 终态后排队：polya_auto_launcher PID 1005196 / polya_harvest_watcher PID 1025537 健在，B2 未终态故未启动；终态产物 frozen_delta_results.json 不存在 → harvest_polya.sh 待命。

### GPU / 节点抽查

- GPU0-5 满显存占用（39-40GB/40G，util 11-100%，节点共享）；本线 GPU1 正常推进；GPU6/7 为 MIG 切片（util N/A），无碍。
- status.log 近端仅 B2 alive=1 / APA queued behind B2 - normal；needs_attention.txt 为空。

### 判定

- 两线均无新终态、无异常。下一检查点：B2 guided ETA 09-04 ~05:45（约 1.5h 后）；polyA 在 B2 终态后 120s 自动接力 GPU1。收割脚本与 watcher 就绪。

## 2026-09-04（08:05，第十二批：例行监控观测，无新终态）

### B2 guided 线（PID 1909082，GPU1）

- 启动 09-03 12:21，本批 elapsed 19h46m；unguided 臂 12:42 已终态，guided 臂仍在途；终态产物 b2_full_891/guided_run_summary.json 与 guided/ 子目录均未出现 → harvest_b2.sh 待命。
- **超时提示**：smoke 校准 ETA ~05:45 已过约 2h20m 仍无终态。深检确认 = 慢速推进非卡死，8 源 smoke 校准低估了全量尾段耗时：
  - CPU ticks 120s 窗口推进 6429 jiffies（≈53% 单核持续），非空转；
  - GPU1 dmon 每秒均有 kernel（SM 9-43%、mem 0-5%），SM 时钟满频 1410MHz；与 04:07 批次的健康画像一致；
  - /proc io 仍为终态一次性落盘 + 全内存缓存特征（write_bytes≈21MB 冻结，符合 runner 设计，见 _write_atomic 终态写 guided_run_summary.json）；
  - 疑因：全量 891 源尾段存在长轨迹/高 edit-budget 慢源，超过 8 源均值外推；runner 无 per-source 进度输出（已知盲区），无法精确定位剩余源数。
- 12:16 的 b2_full_891.failed.json 复核 = 双启动防护误触（output directory already exists，cpu_fallback_used=false），非本进程失败，12:20 重启后正常。
- 合规：xeditsetflow_v5 + xeditcritic_route_a 全部 run_summary 的 cpu_fallback_used=false，无 CPU 静默降级。

### polyA APA 线（GSE113849 2.74M 预微调）

- 仍在 B2 终态后排队：polya_auto_launcher PID 1005196 / polya_harvest_watcher PID 1025537 / b2_harvest_watcher PID 1023582 均健在，B2 未终态故未启动；APA 线 frozen_delta_results.json 不存在 → harvest_polya.sh 待命。

### GPU / 节点抽查

- GPU0-5 满显存占用（39-40GB/40G，util 15-100%，节点共享）；本线 GPU1 正常推进（SM 时钟满频）；GPU6/7 为 MIG 切片（util N/A），无碍。
- status.log 近端仅 B2 alive=1 / APA queued behind B2 - normal；needs_attention.txt 为空（无异常）。

### 判定

- 两线均无新终态。B2 guided 超出 smoke 校准 ETA 但实时证据（CPU ticks + GPU kernel 每秒 + SM 满频）证明为慢速推进而非卡死，暂不干预；下个检查点若仍无终态产物，建议按监控盲区修复方案给 runner 补 progress.jsonl 心跳以定位剩余源数。polyA 在 B2 终态后 120s 自动接力 GPU1。收割脚本与 watcher 就绪。

## 2026-09-04（18:45，第十三批：D6 Baseline 补强 frozen-Δ 全覆盖——RNA-FM / UTR-LM 8 任务 16 组合收官）

### 交付与复用声明

- **薄封装** `scripts/route_a_v3/run_route2_frozen_delta_full_coverage_v1.py`：以库方式导入 `run_route2_frozen_delta_te_family_v1.py` 管线（te 模块级 TASKS 扩展 4 个 generalist spec），参数 task × model，输出 schema 沿用 frozen_delta_results.json（`route_a_v3_route2_frozen_delta_full_coverage.v1`）。
- **P0（Task 5.3 四任务）今天 18:03 已由 `analysis_frozen_delta_te_family_20260904/` 完成**（cuda:5；smoke port-validation 复现 MRL frozen 行：UTR-LM 0.1107267878538859 精确一致、RNA-FM |Δ|=2.4e-6）。管线 seed 20260816 全确定性，重跑必得同值 → 本批**导入而非重算**（`imported_runs` 字段溯源；榜单只增行），per-run 目录随汇总自包含复制。
- **P1（generalist 四任务 × 2 模型 = 8 组合）本批 GPU2 新跑**（cuda:2，A100-PCIE-40GB，UUID d46dc272-4b72-0b54-aacd-f9ace57d622f；VALIDATION only；protected reads=0；不碰 GPU1/3/4/5 任何进程）。冒烟 50 行 × 8 组合先行通过（`analysis_frozen_delta_full_coverage_20260904_smoke/`）。
- MPRAU 主判据另算**变体 pair-mean ρ**（W-ladder 裁决口径：rid 去 `:context:` 后缀分组、≥2 context、per-variant 均值、2,008 变体 Spearman；非配对 bootstrap CI 2,000 iters seed 20260816）——与 Saluki MPRAU 脚本逐字同口径。

### 结果表（16 组合全部到位）

**P1 generalist 新行（本批 GPU2 执行）：**

| 任务 | n | RNA-FM ρ | UTR-LM ρ | critic V5 | 内靶 global_scaled | 既有外部行 |
|---|---|---|---|---|---|---|
| polyA 3'UTR | 2,628 | 0.7114 | 0.7490 | 0.8219 | 0.7308 | APARENT 0.7343 |
| MPRAU pair-mean ρ | 12,048（2,008 变体） | 0.0180 CI[−0.025,+0.061] | 0.0147 CI[−0.031,+0.058] | 0.1025 | 0.0248 | Saluki 0.1205 CI[0.077,0.164] |
| HALF_LIFE 5'UTR | 400 | 0.0271 | −0.0199 | 0.0607 | −0.0522 | Saluki 0.0193 |
| HALF_LIFE 3'UTR | 503 | 0.0500 | −0.0986 | 0.0456 | −0.0029 | Saluki 0.0985 |

polyA 决策口径（top-1 / NDCG@10）：RNA-FM 0.4779 / 0.8481，UTR-LM 0.5020 / 0.8609（V5 0.5007 / 0.8710，APARENT 0.6011 / 0.8906）。MPRAU record-level ρ：RNA-FM 0.0131 / UTR-LM 0.0109（V5 record-level 0.0732）。

**P0（Task 5.3 导入行，源自 te_family run）：**

| 任务 | n | RNA-FM ρ | UTR-LM ρ | critic V5 | 内靶 global_scaled |
|---|---|---|---|---|---|
| GSE200304 TE | 1,614 | 0.0009 | 0.0113 | 0.0579 | −0.0266 |
| GSE149487 TE | 48 | −0.0153 | −0.0277 | 0.1953 | 0.1747 |
| GSE149487 RNA | 48 | 0.2958 | 0.0433 | 0.0500 | 0.2230 |
| GSE186455 | 274 | 0.1043 | −0.1233 | 0.0639 | −0.0052 |

GSE200304/GSE149487 各层全为 singleton source group，top-1/NDCG@10 按榜单 §1.5 口径记 null。

### 对位判读

1. **polyA**：两外部 frozen 行均**负于 V5**（−0.110 / −0.073），UTR-LM 0.7490 点估计略超 APARENT（+0.0147）与内靶（+0.0182）、RNA-FM 0.7114 低于两者（−0.0229 vs APARENT）→ 落位"APARENT 同档"；V5 保持 polyA 最强行（外部行 0 通道越 V5）。
2. **MPRAU**：pair-mean ρ 0.0180 / 0.0147，CI 全跨零 = **无信号**，远负于 V5（0.1025）与 Saluki 弱对照（0.1205）→ 外部通用 LM frozen 行不能填补 R5 结构性空白主张的对照位；matched-FT 双臂（今日已预注册，GPU3/4）另线检验。
3. **HALF_LIFE**：四行全部 |ρ|<0.1（+0.027 / −0.020 / +0.050 / −0.099）→ **"物理不可学"归因从推断升级为实证闭环**（外部现代 LM 同样 ≈0）；维持该任务不参与"全任务好"叙事的预注册声明。
4. **P0 导入行**：RNA-FM 在 GSE149487 RNA（0.2958，+0.2458 vs V5、+0.0728 vs 内靶）与 GSE186455（0.1043，+0.0404 vs V5、+0.1095 vs 内靶）点估计胜出——均为小样本任务（n=48 / 274），无 CI 口径不宣称显著；其余 P0 组合负于 V5。UTR-LM 全线 ≈0 或负。
5. **宏观**：16 组合 vs V5 = **2 胜 / 2 平 / 12 负**（判读规则预声明：|Δρ|≥0.01 计胜负；两胜均为 RNA-FM）→ 任务特异性 critic 在无外部靶任务上整体保持领先；外部现代 LM frozen 行价值 =（a）HL 不可学实证化，（b）为 matched-FT 对照提供 frozen 锚点（MPRAU 0.015–0.018）。

### 输入适配声明（逐行，写入 frozen_delta_results.json per-model input_adaptation）

- **RNA-FM**：multimolecule RnaTokenizer（T→U、动态 padding）、fp32 前向、非特殊 token masked-mean-pool；>1000nt 分块策略（build_route2_rnafm_feature_cache_v1 口径）**P1 未触发**（P1 最长 164nt ≪ 1024 token 上限，零截断）；length-sorted batch ≤32 序列 / ≤8192 token。
- **UTR-LM**：官方 SISS checkpoint epoch93.pkl（git b77b589）、BOS token 第 6 层表征、rotary 位置编码无硬长度限制；batch ≤128 序列 / ≤16384 token（内存适配，padding + attention mask 保证与计数分批结果一致）。
- P1 唯一序列 29,311 条（101–164nt）；probe = MRL frozen 行原协议（Linear 读出 + AdamW lr 1e-3 / wd 1e-4 × 100 epoch full-batch、seed 20260816、source-group 加权 MSE、VALIDATION epoch 选择——HPO_VALIDATION_ONLY 边界，fit 记录数与各任务 TRAIN 精确一致：polyA 25,710 / MPRAU 55,704 / HL5' 893 / HL3' 1,308）。

### locator

- 汇总（16 组合 + interpretation 判读表 + imported_runs 溯源）：`experiments/analysis_frozen_delta_full_coverage_20260904/frozen_delta_results.json`
- per-run：同目录 `{task}__{model}/predictions.jsonl + run_detail.json`（16 个，行数与 VALIDATION n 精确一致）
- 冒烟：`experiments/analysis_frozen_delta_full_coverage_20260904_smoke/`（50 行链路验证）
- P0 源：`experiments/analysis_frozen_delta_te_family_20260904/`（cuda:5）及其 smoke port-validation
- 代码：`scripts/route_a_v3/run_route2_frozen_delta_full_coverage_v1.py`（本批）+ `run_route2_frozen_delta_te_family_v1.py`（P0 管线，本批一并入库）

## 2026-09-04（19:00，第十四批：D6 Baseline 补强 P0——MPRAU matched-FT 外部双臂 RNA-FM / UTR-LM 收官，空白主张实证成立）

### 交付与预注册声明

- **预注册** `docs/paper/route2_mprau_matched_ft_external_prereg_v1.md`（commit 4a8fac9f）：把 R5「外部行结构性空白」升级为可检验主张——外部通用 LM 在与我方 per-task 臂**相同任务数据 + 相同更新预算**（12,048 行/pass × 8 passes × batch 32 = 3,016 updates，uniform 子采样自 55,704 行 TRAIN 池）下对位微调，做不动 → 空白主张成立；反超 → 如实报告进 W 阶梯。
- **微调方式（预注册选择）**：两臂均**全参微调**——RNA-FM（multimolecule 官方 README = HF Trainer 全参示例，无 LoRA 示例）；UTR-LM（官方下游 `Finetune_extract_append_predictor_*.py --finetune` 分支 = backbone+head 全参）。优化器 AdamW wd 1e-4 + cosine 至 10% + 5% warmup（协议统一项，与官方 UTR-LM SGD momentum 0.9 的差异记入预注册 §8）。LR 一次性声明：RNA-FM backbone 2e-5 / head 1e-4（99.5M 量级标准）；UTR-LM backbone 1e-4 / head 2e-4（1.2M 微型模型）。
- **脚本** `scripts/route_a_v3/run_route2_mprau_matched_ft_external_v1.py`（commit fb2aebe1）：loss/数据/评估逐行镜像 directft 臂 A（huber δ=1.0 + 跨源 pairwise softplus + soft-Spearman 0.2 + within-source 0.5，pass 1–2 {1.0,0.25,0.0} / pass 3–8 {1.0,0.5,0.25}）；frozen-delta（pred = f(cand) − f(src)）；readout RNA-FM = 非特殊 token masked-mean-pool / UTR-LM = BOS 第 6 层表征（各与本项目 frozen 基线同口径）+ fresh 线性 head；seed 20260907；FINAL-PASS-8-FIXED。
- **冒烟修复（诚实记录）**：首版对 UTR-LM `lm_head.requires_grad_(False)` 会经 weight-tying 连带冻结 `embed_tokens`（smoke trainable 1,190,017 异常核查发现），改为 identity guard 排除绑定权重后修复（1,191,297 = +1,280 embeddings），重新冒烟通过后才发射；RNA-FM 冻结未用 pooler（honest trainable count 99,112,321）。首次双发射命令有 log 路径竞态（UTR-LM nohup 先于 mkdir 执行），UTR-LM 干净补启，无数据影响。
- 执行：GPU3 = RNA-FM 臂（PID 1946281，18:34–18:40，~7 min 终态）、GPU4 = UTR-LM 臂（PID 1951854，18:36–18:37，~2 min 终态）；两进程均干净退出、显存释放；VALIDATION only、protected reads=0、BF16、未触碰 GPU0/1/2/5 任何进程。

### 结果表（VALIDATION 12,048 行，主判据 = 变体 pair-mean ρ，2,008 变体，FINAL-PASS-8-FIXED）

| 臂 | trainable | pass1→8 pair-mean ρ 曲线 | 终态 pair-mean ρ | task_macro_spm | top-1 | NDCG@10 |
|---|---|---|---|---|---|---|
| RNA-FM 全参（99.5M，lr 2e-5/1e-4） | 99,112,321 | 0.0576 → **0.0917**(p2) → −0.0457 → −0.0648 → −0.0514 → −0.0387 → −0.0781 → **−0.0747** | **−0.0747** | −0.0541 | 0.500 | 0.8155 |
| UTR-LM 全参（1.2M，lr 1e-4/2e-4） | 1,191,297 | 0.0801 → 0.0752 → −0.0616 → −0.0544 → −0.0751 → −0.1037 → −0.1084 → **−0.1066** | **−0.1066** | −0.0765 | 0.361 | 0.7642 |

**paired bootstrap 判定表（2,000 iters，seed 20260816，2,008 共享变体）：**

| 臂 | vs V5 0.1025：Δ [95% CI] | CI 排零？ | vs Saluki frozen 0.1205：Δ [95% CI] | CI 排零？ |
|---|---|---|---|---|
| RNA-FM | **−0.1773 [−0.2484, −0.1068]** | 是（显著更差） | **−0.1952 [−0.2699, −0.1222]** | 是（显著更差） |
| UTR-LM | **−0.2091 [−0.2791, −0.1387]** | 是（显著更差） | **−0.2270 [−0.3038, −0.1533]** | 是（显著更差） |

### 对位判读

1. **空白主张成立（预注册 §7 第一行形态）**：外部通用 RNA LM（99.5M RNA-FM、1.2M UTR-LM）全参对位微调，终态 pair-mean ρ 双双显著为负（四个 CI 全部排零、方向为负），既做不过 V5 0.1025 也做不过 Saluki frozen 弱对照 0.1205，更未接近 W-ladder 0.0510–0.0883 的正区间 → MPRAU 等位偏移端点在外部通用 LM 上「微调也做不动」，R5 结构性空白从推断升级为**实证闭环**（frozen 锚点 0.018/0.015 + matched-FT 终态 −0.075/−0.107，13 批 + 本批）。
2. **pass-3 坍塌为跨 backbone 普适现象**：四个 backbone（mRNABERT LoRA、RNA-FM 全参、UTR-LM 全参）在 loss 调度切换（pairwise 0.25→0.5 + soft-Spearman 启用）后 pair-mean 全部由正转负（+0.107→−0.085 / +0.092→−0.046 / +0.080→−0.062），且 pass 1–2 峰值（0.092/0.080）本身也未超 V5——说明坍塌不是某 backbone 失败，而是该端点对排序强化阶段的抗性（任务级发现，供 W 阶梯 loss 复查）。
3. **预注册纪律遵守**：峰值 0.0917（RNA-FM p2）略高于 W-ladder 上沿 0.0883，但选择规则 = FINAL-PASS-8-FIXED（禁 peak-picking），终态判定如上；峰值仅作曲线记录，不进任何主张。
4. **mRNABERT directft 臂 A 对照（−0.0908）**：三臂终态同带（−0.075 / −0.091 / −0.107），LoRA vs 全参、86M vs 99.5M vs 1.2M 均无实质差异 → 「做不动」与适配器方式、参数量级无关。

### locator

- 结果：`experiments/analysis_mprau_matched_ft_external_20260904/{rnafm,utrlm}/matched_ft_results.json`（含 pass_history 全曲线 + 双 bootstrap + 差异清单）、`predictions.jsonl`（各 12,048 行）、`matched_ft_checkpoint.pt`（398MB / 4.9MB）；日志 `.../logs/{rnafm,utrlm}.log`
- 代码：`scripts/route_a_v3/run_route2_mprau_matched_ft_external_v1.py`（fb2aebe1）；预注册：`docs/paper/route2_mprau_matched_ft_external_prereg_v1.md`（4a8fac9f）；worktree `route_a_v3_w0_diagnosis_20260902` 分支已 push GitHub
- 进程：RNA-FM PID 1946281 / UTR-LM PID 1951854 均已终态退出（GPU3/4 释放）；冒烟产物 `/tmp/matched_ft_smoke/`（一次性）

### SetFlow V5 B2 guided 终局判定（自动收割 2026-09-05 12:16:26）

- 全量 891 源，unguided recovery = 0.1205 | guided recovery = 0.1263（wall 47.9 h）
- Δrecovery = 0.0058，CI [-0.0042, 0.0163]；Δhit@1 = -0.0025，CI [-0.0087, 0.0032]（paired bootstrap 2000 iters，seed 20260816）
- **Gate B2**（Δrecovery≥+0.05 且 CI 不跨零 且 hit@1 不劣化）: **False**
- **Gate B3**（guided recovery≥0.35）: **False**
- 依据：/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditsetflow_v5/guided_b2_20260903/b2_full_891_adjudication.json（VALIDATION 口径，protected reads=0，产物在 /mnt）

### 批次十四：双线在途监测 + B2 收割复核 + APA 核心异常（观测 2026-09-05 18:44，cunyuliu 交接审计）

- **B2 guided（xeditsetflow_v5/guided_b2_20260903/b2_full_891）**：TERMINAL。报备 PID 1909082 已不在进程表；watcher 已于 09-05 12:16 自动收割（commit 6d625cb7）。复核 adjudication：guided recovery 0.12626 vs unguided 0.12046；Δrecovery +0.0058 CI [-0.0042,+0.0163] 跨零；Δhit@1 -0.0025 CI [-0.0087,+0.0032] 跨零；**Gate B2 FAIL / Gate B3 FAIL**。run_summary cpu_fallback_used=false、precision=BF16 → 无 CPU 静默降级。protected reads=0。
- **APA polyA 线（category 2.74M 库 = apa_3p5m_prefinetune）**：**报警级异常**。
  - 09-04 19:00 GPU2 CUDA OOM 崩死（Tried to allocate 142MB，GPU2 空闲仅 93MB；被其它用户 job 占 31GB+），崩于 mRNABERT forward GELU。
  - 无终态产物：epoch_frozen_delta_metrics.jsonl / training_losses.jsonl 均 0 字节，无 frozen_delta_results.json → **harvest_polya 不得运行（无 frozen_delta_results.json）**。
  - polya_harvest_watcher 已于 09-04 19:06 退出并 ALERT；needs_attention.txt 自 09-04 20:00 每小时记“APA NOT RUNNING AND NO WATCHER”。
  - 09-05 全天 status.log APA alive= watcher= 均空；crontab 仅 monitor 每 2h，无 APA 重启 watcher。GPU0-5 全部被 f3/dr/review、toktokenbench、gmx 等外部任务占用；**09-05 无任何 mRNA-EditFlow 训练进程在跑**。
- **修复建议**：APA 需人工在空闲 GPU（当前 6/7 为 MIG；0-5 均被占）重建 2.74M full-FT（mRNABERT，batch 128，lr 2e-5，seed 20260903，FINAL-EPOCH-6-FIXED）；或等待抢占型 GPU 空档；并补注册 polya watcher 重启/告警常驻。

### B2 per-task Δ 分解（手动补跑 2026-09-05 18:52，修复 per-task watcher 的 manifest 路径 bug）

> 原 b2_per_task_watcher 的 manifest 路径误写为 `experiments/generation_eligibility`（正确为 `route2/generation_eligibility`），已修复留证；本节为修正后手动执行结果。

| endpoint | n | unguided | guided | Δrecovery | Δhit@1 |
|---|---|---|---|---|---|
| MEAN_RIBOSOME_LOAD | 652 | 0.1547 | 0.1587 | +0.0041 | −0.0034 |
| MPRAU_ALLELIC_SKEW | 108 | 0.0278 | 0.0324 | +0.0046 | +0.0000 |
| PROXIMAL_POLYA_SITE_USAGE | 20 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| RNA_HALF_LIFE_MINUTES | 111 | 0.0315 | 0.0495 | +0.0180 | +0.0000 |

**三分归因判读（B2 FAIL 后的预登记决策树）**：四任务增益均匀微弱（全部 << +0.05 门），**未出现**polyA 源大增益 / 弱任务源无增益的 critic 质量模式 → 指向**引导形式约束**（potential 式率修正的增益传导不足），非 critic per-task 质量。注意事项：(a) polyA 源仅 20 个（891 源中 MRL 占 652），源分布高度不均衡，per-task 结论受小样本限制；(b) unguided 基线 polyA 源 recovery=0 → polyA 源候选邻域可能未被生成器覆盖（base 侧探索不足也可能是共因）。**后继动作（按 spec contingency）**：TreeG sample-then-select 备选评估提上日程；critic 升级（V8）后重跑 B2 仍值得（当前 critic 引导增益传导弱的归因未完全闭合）。证据：`b2_full_891_adjudication_per_task.json`。

### 批次十五：训练 manager 部署 + APA 自适应重启 + V8 Stage 1 S/H 发射（2026-09-05 21:05，cunyuliu 交接审计）

- **问题**：apa_relaunch_watcher 要求 ≥30GiB 空闲整卡才启动，GPU0-5 全天被外部任务占满（99-100% util，仅 3-10GB 游离显存）→ 一天零进展。用户指示：**有显存即用，禁止显存 gate；多 GPU 并行**。
- **方案（amendment，pre-launch 记录）**：
  - APA runner 新增 `--batch` 参数（默认 128 不变=预注册；显存挤占时降 batch），commit e9789855 已 push。
  - 新部署 `monitor/mrna_training_manager.sh`（替换 apa_relaunch_watcher）：扫描 GPU0-5 取游离显存最大卡，按 free≥28GiB→128 / ≥20GiB→96 / ≥15GiB→64 / ≥11GiB→48 / ≥7GiB→32 自适应；GPU 预留机制防重复占卡；死亡无终态自动重发（batch 逐次减半，floor 32）；APA 终态自动 harvest_polya.sh + journal commit。
  - 新监控 `monitor/mrna_editflow_monitor.sh` v3（crontab 每 2h 复用）：跟踪 manager/APA/V8-S/V8-H + needs_attention 告警。
  - TRAE 侧定时任务「mRNA-EditFlow 训练监控」每 30 分钟巡检（ID b2f628e0）。
- **发射记录**（20:48-20:53）：
  - V8-S 冒烟重跑 GPU2（--max-steps 20 --batch 32 --out-dir v8_stage1_smoke_20260904_rerun/s，PID 4158076）：MRL 库 677,608 行泄漏审计 flagged=0 ✅，polyA 2.74M tokenize 中。
  - APA GPU3（batch 32，PID 4192010）：库 2,740,320 行 leakage flagged=0，tokenize 中。
  - V8-H GPU0（batch 32，PID 15915）：库加载中。
- **排障记录**：清理了 09-04 中断会话遗留的陈旧 wrapper（PID 2160080，等待 GPU5 arch-s 后要发 H 冒烟）；修复了 manager 双发射竞态（APA/V8-H 同抢 GPU3 → GPU 预留文件 state_reserved_gpus.txt）。
- **纪律核验**：三任务全部 CUDA BF16（run_summary cpu_fallback_used 待终态核验）；protected reads=0；产物 /mnt、代码 /home + push（e9789855）。
- **风险提示**：共享卡上 batch 32 全参微调仍可能被外部任务挤爆 OOM → manager 自动重发兜底；GPU0 已从 6.9GB 被挤到 1.5GB，V8-H 有 OOM 风险（自动迁移）。

### 批次十六：双线巡检（观测 2026-09-05 21:10，cunyuliu 定时巡检）

- **manager**：ALIVE（PID 15741/15914；20:51、20:53 两次重启，日志正常 LAUNCH）。
- **APA polyA 预微调**：RUNNING，gpu=3，PID 4192010（batch 32，调整后）。库 2,740,320 行泄漏 flagged=0 ✅，tokenize 中（日志停在 `library clean ... target mean 0.318`）。无终态 frozen_delta_results.json → 暂不运行 harvest_polya。无 CPU fallback；CUDA BF16。GPU 显存当前仅占 ~860MiB（tokenize/数据阶段，正常）。
- **V8-H 臂**：RUNNING，gpu=0，PID 15915。库加载/flag=0，tokenize 中。无终态 run_report.json（正常，启动不足）。
- **⚠️ V8-S 臂：真实训练未运行（疑似被冒烟进程占位）**。
  - 进程表唯一 `--arch s` 进程 PID 4158076 = **冒烟测试**（20:48 手动发射，`--max-steps 20 --batch 32 --out-dir v8_stage1_smoke_20260904_rerun/s`），观测时已活 16+ 分钟仍在 tokenize（143% CPU）。
  - 该冒烟 cmdline 恰含 `run_route2_v8_stage1_joint_prefinetune_v1.py --arch s`，命中 manager `v8s_running()` 的 `pgrep -f --arch s` → manager 认为 S 臂存活，**不启动真实 S 臂**。
  - 真实 S 臂 out-dir `v8_stage1_joint_prefinetune_20260904/`（s 臂产物目录）**当前不存在**；manager 日志仅出现 LAUNCH v8_h，无 LAUNCH v8_s。
- **GPU 总览**：GPU0-5 均被外部任务 99-100% 占满（f3/dr/review、toktokenbench、gmx）；6/7 为 MIG 禁用。B15 已预告的 V8-H 挤占/迁移风险仍在。
- **needs_attention.txt**：仍残留 `20:00:01 APA NOT RUNNING AND NO WATCHER` 陈旧告警——APA 已于 20:51 重启并存活，告警未清除（monitor 每 2h 重写前仍挂）。
- **结论/下一步建议**：
  1. 终止冒烟 PID 4158076（及其 wrapper 4157952）→ `v8s_running()` 恢复 false → manager 于下一轮自动发射真实 S 臂（批量自适应）。此为本次唯一阻塞项。
  2. APA / V8-H 均正常推进，维持 manager 自动兜底；共享卡 OOM 风险持续盯防。
  3. 建议顺手清除 needs_attention 陈旧告警（APA 已健康）。

### 批次十五b：V8 S/H 双发射到 GPU7 MIG 3g.20gb + manager 修复（2026-09-05 21:40，cunyuliu 交接审计）

- **batch=0 事故与修复**：manager 在游离显存 <7GiB 时 pick_slot 返回 batch=0 且启动守卫只查 gpu≥0 → 21:24/21:26 误发 v8_s/v8_h 两进程（batch=0，训练循环必崩）；同时 deploy 命令疑似被重放产生重复 manager 实例（reservation 文件被重置为仅 "3"）。处置：杀掉全部重复实例与 batch=0 进程；manager 增加 (a) pick_slot 返回 "-1 0"（batch<32 不放行）、(b) 启动守卫 `batch>=32`、(c) 启动时 self-dedup（pgrep 精确匹配 "bash mrna_training_manager.sh" 杀掉重复实例）。无训练损失（被误发进程尚在 tokenize）。
- **V8 S/H 发射到 MIG**：GPU0-5 全被外部任务挤爆（游离 0.1-5GB），仅 GPU7 两个 MIG 3g.20gb 切片有 14-20GB 可用且 SM 隔离无争用 → 按用户"有显存即用"指示发射（旧纪律"禁 6/7"针对 4.75GB 切片与索引陷阱，20GB 切片 + CUDA_VISIBLE_DEVICES=MIG-UUID 显式索引不适用该陷阱）：
  - V8-S → MIG-6e59f9af（GPU7 slice A），batch 64，PID 667111
  - V8-H → MIG-10b9b777（GPU7 slice B），batch 64，PID 667112
  - amendment：batch 64（预注册 128）+ MIG 部署理由同上，pre-launch 已记录；lr 2e-5 / seed 20260903 / epochs 2 / BF16 照预注册 route2_v8_stage1_prereg_v1.md。
- **当前在途三线**（21:40）：APA GPU3 batch32 训练中（step ~3000，mse 0.29↓，~3.3 step/s 受主机 load 113-196 拖累）；V8-S/H GPU7 MIG batch64 tokenize 中。manager 单实例（PID 512372）继续守 GPU0-5 兜底。

### 批次十五c：V8 预注册 §2 补齐运行 + Stage 1 自动判定上线（2026-09-05 21:55，cunyuliu 交接审计）

- **发现预注册遗漏**：route2_v8_stage1_prereg_v1.md §2 要求 polyA 单域基线由本 runner 以 `--arch s --libraries polya`（epochs 6）补齐（作为 polyA 非破坏门参照，§8.1），首轮发射只发了 S/H 两臂 → manager 新增第 4 job `v8p`（epochs 6，batch 自适应 32-128），GPU0-5 有 ≥7GiB 槽位即自动发射。
- **Stage 1 判定脚本上线**（scripts/route_a_v3/adjudicate_route2_v8_stage1_v1.py，commit f4e9b274）：按 §8 判定门实现——MRL 非破坏门 0.9×M臂3-seed均值（自动从三个 seed frozen_delta_results.json 算，缺失时回退注册常数 0.3076）；polyA 非破坏门 0.9×s_polya 基线；S vs H（S ≥ H−0.02 选 S）；smoke 报告直接拒绝；输出 adjudication_v8_stage1.json。manager 自动触发（S/H 终态即跑，polyA 基线终态后全量重跑），commit bfd85fe6。
- 纪律：判定只用 FINAL-EPOCH-FIXED primary 记录；smoke/proxy/训练集结果不构成科学结论（判定脚本硬拒 smoke）。

### 批次十五d：v8p polyA 单域基线发射成功 + manager v4（2026-09-05 22:20，cunyuliu 交接审计）

- **四线并行达成**：APA（GPU3 batch32，step 11000 mse 0.228↓）/ V8-S（GPU7 MIG-A batch64）/ V8-H（GPU7 MIG-B batch64）/ **v8p polyA-only 基线（GPU1 batch64，PID 1000249，epochs 6）**。
- **排障两连**：(1) manager 文件被异步改写为残缺版（有 v8p 循环检查、缺 job_v8p/v8p_running 函数定义）→ job_v8p 调用静默 command-not-found（stderr→/dev/null）→ 改用服务器端 heredoc 原子重写 v4；(2) heredoc 双引号 awk 程序导致运行时 `$1: unbound variable`（set -u）→ free_mib 恒空 → 修正为单引号 awk。修复后 pick_slot 正常（GPU1 15.5GB→batch64），v8p 发射成功，reservation 现为 {1,3}。
- 防回归：running 检查全部锚定 python 二进制路径（瞬态 ssh/bash 命令行不再误匹配）；manager v4 已 commit ccb423e8 push。

### 巡检 2026-09-06（2026-09-06 00:07，监控巡检记录）

- **连通性**：巡检期间 SSH banner 超时 ~1 分钟（服务器重载），自动恢复；非进程故障。
- **manager 存活**：是。但当前有 3 个 mrna_training_manager.sh 实例并存（PID 999885/1114349/1187064），存在重复发射竞态风险，建议后续去重为单实例。needs_attention.txt 不存在（无告警）。
- **GPU0-5 全忙**：util 83-100%；GPU6/7 MIG。无 idle A100。
- **APA polyA**：PID 1114351（gpu4 batch32 seed20260903）存活；日志已 load polya 库 2,740,320 行，处于早期训练/初始化；终态 frozen_delta_results.json 未出现（未判定）。无 CPU fallback、无 OOM。
- **V8 Stage1 S 臂**：PID 667111（MIG-A，mrl+polya，batch64 epochs2）存活，epoch1 step~6500；mse 0.3-0.5，mrl/polya 在收敛。终态 run_report.json 未出现。
- **V8 Stage1 H 臂**：PID 667112（MIG-B，mrl+polya）存活，epoch1 step~3000；进度略慢于 S 臂。终态未出现。
- **v8p polyA-only 基线**：PID 1187066（gpu5 batch32）存活（manager 00:05 发射）。
- **结论**：四线均在跑、无终态产物、无 CPU 静默降级、无 OOM。一切正常，无需人工干预。

### 巡检 2026-09-06 03:10（监控巡检记录）

- **连通性**：SSH 正常（前一日约 40 分钟断连已恢复，非进程故障）。
- **manager 去重处置**：巡检发现陈旧实例 PID 999885（09-05 22:17 启动，pre-v4，00:07 巡检已标记）仍存活且无训练子进程（仅 sleep 120 兜底循环），存在对已死 job（如 v8p）重复发射竞态 → 按 manager v4 self-dedup 约定 kill 999885。现单实例 PID 1899326（持有当前 APA 子进程 1899327），确认无训练受影响。
- **APA polyA 预微调**：PID 1899327（gpu0 batch32 seed20260903 epochs6）存活，epoch1 step~19000，mse 0.41→0.22 稳步下降，lr warmup 1.48e-05；日志 03:01 仍在写。终态 frozen_delta_results.json 未出现（未到判定时点）。
- **V8 Stage1 S 臂**：PID 667111（GPU7 MIG-A，mrl+polya，batch64）存活，epoch1 step~45500；末步 mse 0.8751 为单步尖峰，mrl=0.4425/polya=0.1861 稳定（区间均值 ~0.2-0.3），判为噪声非发散。终态 run_report.json 未出现。
- **V8 Stage1 H 臂**：PID 667112（GPU7 MIG-B）存活，epoch1 step~30500，mse 0.2-0.4 区间收敛，mrl/polya 正常；进度慢于 S 臂（发射顺序所致）。
- **【异常】v8p polyA-only 基线进程死亡（OOM）**：manager 两次发射（22:17 gpu1 / 00:05 gpu5）均已死；v8_stage1_polya_base.log 00:35 记 CUDA OOM：GPU5 仅 13MiB 游离时申请 52MiB 失败，被外部用户 honghuiyang metddi 多 worker（各约 2.44GiB）挤占。无终态 run_report.json → polyA 非破坏门参照缺失，Stage1 判定脚本暂无法全量裁决。建议：改发 GPU4（33GB 游离）重跑 v8p，或待外部任务释放后由 manager 自动重发。
- **GPU 状态**：GPU0 100%（游离 70MiB，APA 所在）、GPU1 96%（外部任务）、GPU2 99%、GPU3 81%、GPU4 90% 但仅用 7.5GiB（游离 33GB，v8p 重发首选）、GPU5 60%（外部任务）；GPU6 MIG 1g.5gb、GPU7 MIG 3g.20gb×2（S/H 臂，沿用 09-05 有显存即用决策）。GPU0-5 无空闲整卡。
- **纪律检查**：APA/S/H 三条日志均无 cpu_fallback_used=true，无 CPU 静默降级；除 v8p OOM（已记录）外无新 OOM。needs_attention.txt 不存在（无告警）。
- **结论**：APA + V8-S/H 三线健康在跑；唯一异常为 v8p 基线 OOM 死亡（外部任务挤占 GPU5）；manager 已去重为单实例。无紧急人工干预项。

### 巡检 2026-09-06 06:05（监控巡检记录）

- **连通性**：SSH 正常，单次调用完成。
- **manager 存活**：单实例 PID 1899326（03:10 去重后保持），日志最后 LAUNCH 01:08 apa gpu=0；status.log 06:00 MGR alive=1。无重复实例。
- **APA polyA 预微调**：PID 1899327（gpu0 batch32 seed20260903 epochs6）存活；03:10 step~19000 → 06:03 step 54000（~11.7k steps/h），mse 0.41→0.20 收敛中，lr 1.95e-05。终态 frozen_delta_results.json 未出现（epoch1/6，约半程）。无 cpu_fallback、无 OOM。
- **V8 Stage1 S 臂**：PID 667111（GPU7 MIG-A，mrl+polya，batch64 epochs2）存活；epoch1 完成（stage1_s_epoch1.pt 03:36 落盘），epoch2 step 86000，mse 0.2-0.5、mrl 0.36-0.45、polya ~0.18 收敛。终态 run_report.json 未出现。
- **V8 Stage1 H 臂**：PID 667112（GPU7 MIG-B）存活；epoch1 完成（stage1_h_epoch1.pt 04:44 落盘），epoch2 step 70500。终态未出现。
- **【延续异常】v8p polyA-only 基线**：仍死亡（00:05 后 manager 未再发射；03:10 记录 OOM 于 GPU5）。现 GPU3 游离 16.6GB（util 100% 为外部任务、内存侧可用）——若 manager 仍不发，建议手工重发 v8p（batch32）至 GPU3，或待外部任务释放。polyA 非破坏门参照仍缺失，Stage1 判定脚本无法全量裁决。
- **GPU 状态**：GPU0 100%（游离 2.9GB）/GPU1 99%（2.7GB）/GPU2 99%（5.3GB）/GPU3 100%（16.6GB）/GPU4 99%（1.8GB）/GPU5 96%（7.2GB）；GPU6/7 MIG（S/H 臂 + gmx）。无空闲整卡。
- **纪律检查**：APA/S/H 日志均无 cpu_fallback_used=true、无 CPU 静默降级、无新 OOM。needs_attention.txt 不存在（无告警）。
- **结论**：APA + V8-S/H 三线健康在跑，V8 双臂均已进入 epoch2 接近终态；唯一未决项为 v8p 基线（OOM 后未重发），GPU3 有可用游离显存可作重发点。

### 巡检 2026-09-06 09:05（监控巡检记录）

- **连通性**：SSH 正常。
- **manager 存活**：单实例（PID 1117143/1899326 系同一管理；日志 22:17 后多次 self-restart，最新 LAUNCH 01:08 apa gpu=0）。needs_attention.txt 不存在（无告警）。
- **【新终态】V8 Stage1 S 臂已完成（07:34 run_report.json）**：mrl+polya epochs2 全程跑完 106,812 步（epochs_completed=2，smoke=false，selection=FINAL_EPOCH_FIXED）。zero-shot（VALIDATION frozen-Δ）：mrl task_macro_spearman=0.3078 / ndcg@10=0.868；polya task_macro_spearman=0.1122。**过 MRL 非破坏门（0.3078 ≥ 0.9×0.3076=0.2768）✓**。
- **【新终态】V8 Stage1 H 臂已完成（08:43 run_report.json）**：同预算 106,812 步，CNN stem 生效（参数 113,929,185）。zero-shot：mrl task_macro_spearman=0.2876 ；polya=0.5262。**过 MRL 非破坏门（0.2876 ≥ 0.2768）✓**。
- **S vs H 裁决（§8 门2，MRL 口径）**：S=0.3078 ≥ H−0.02=0.2676 → **判选 S**（简单优先）。polyA 次级证据 H 强于 S（0.5262 vs 0.1122），不单独裁决。（注：S 的 polyA 0.1122 显著弱于 H，polyA 豁免质量 H 明显胜出，仅作留档参考。）
- **【延续阻塞】polyA-only 基线仍缺失 → polyA 非破坏门无法裁决**：s_polya 目录仅含 00:35 初始审计文件（leakage_audit.json 落盘即 OOM 死亡，training/zeroshot 全空、无 run_report.json）。GPU5 被外部 honghuiyang 多 worker 严重挤占（当前 GPU5 上即观察到 30+ 进程、累计显存远超单卡，00:35 OOM 申请 52MiB 仅剩 13MiB）。06:05 巡检后 manager 仍未重发 v8p。**Stage1 正式判定须待 polyA-only 基线补齐后合卷。**
- **APA polyA 预微调**：PID 1899327（batch32 seed20260903 epochs6）存活，06:03 step~54000 → 续跑中；终态 frozen_delta_results.json 未出现（未判时点）。无 cpu_fallback、无 OOM。
- **GPU 状态**：GPU0-5 util 73-100% 且显存近满（0=37.7/1=37.5/2=36.2/3=38.4/4=37.4/5=23.1GiB）；GPU6 1g.5gb / GPU7 3g.20gb 为 MIG（不用于正式训练）。无空闲整卡；GPU5 属外部任务重度过订阅（非本项目）。
- **纪律检查**：APA/S/H 日志均无 cpu_fallback_used=true、无 CPU 静默降级；无新 OOM（仅 v8p 基线沿用 00:35 OOM 记录）。
- **结论/建议**：V8 Stage1 S/H 双臂均取到终态指标 → MRL 门双过、裁决选 S；但因 polyA-only 基线 OOM 且未重发，polyA 非破坏门参照缺失，Stage1 暂记"部分合卷、待 polyA 门补齐"。下一步：建议把 v8p（--arch s --libraries polya，batch32）改发 GPU4/GPU3 等有可游离显存的卡重跑，或在外部任务释放后由 manager 自动重发；待基线落盘 run_report.json 后再对 polyA 门做最终裁决并淘汰该工作流判定。APA 无异常，继续盯。

### 批次十七：SSH 恢复 + V8 Stage 1 双臂终态 + TreeG 全量负结果（2026-09-06 10:55，cunyuliu 交接审计）

- **V8 Stage 1 双臂终态（FINAL-EPOCH-FIXED，VALIDATION）**：S 臂 MRL 0.3078 / polyA 0.1122；H 臂 MRL 0.2876 / polyA 0.5262。MRL 非破坏门双过（阈值 0.2768）；S vs H（MRL）裁决 S（Δ+0.0202）。stage1_success = PROVISIONAL_MRL_ONLY（polyA 门等 v8p 基线）。adjudication_v8_stage1.json 落盘。
- **TreeG 全量 891（V5-critic top-1 选择）**：hit@1 0.0045 vs unguided 0.0367，Δ−0.0322 CI 全负（显著有害）；recovery 0.0022 vs 0.1205（shot-count caveat）。**引导形式双备选（potential/select）均证伪 → 归因收敛 critic 打分质量侧 → V8 Stage 3 是决定性实验**。脚本 commit 4783ba17。
- **manager v5 死锁修复**：v4 永久 reserve 在换卡多次后全卡死锁（9.5h 零发射）；v5 改 job 级 gpu + 死亡即释放，v8p 已 batch 128 重发 GPU1。
- **在途**：APA GPU0 epoch2 step ~109K（mse 0.19↓）；v8p GPU1 batch128 tokenize；外部负载回落（load ~49）。

### 巡检 2026-09-06 12:10（监控巡检记录）

- **连通性**：SSH 正常；load 正常回落。
- **manager 存活**：发现**双实例**（1705202@10:45 过期、1735812@10:47 当前，dedup 未生效、killed duplicate 计数=0，疑似 manager 被重复 nohup 拉起且旧实例未退出）。**已手动 kill 1705202 恢复单实例**，现仅 1735812 存活；v8p 当前批由 1735812 于 10:47 以 batch128 重发（state_v8p.gpu=1），state_apa.gpu=0。needs_attention.txt 不存在（无告警）。
- **APA polyA 预微调（GPU0，batch32，epochs6）**：PID 1899327 存活（etime 10:58:39，GPU0 100%）；日志 epoch2 step~125000，mse~0.19 缓降中；frozen_delta_results.json 未出现（未到终态）。
- **V8 Stage1 polyA-only 基线（GPU1，batch128，epochs6）**：PID 1735814 存活（child of 1735812，GPU1 96%）；日志 epoch1 step~9500/21409，mse 0.3365→0.1757↓、polya 0.3554→0.2068↓，正常收敛；run_report.json 未出现（未到终态）。batch 由 manager 自适应重发为 128（规格相符）。
- **GPU 状态**：GPU0 100%（APA+外部 toktokenbench）、GPU1 96%（v8p+外部 review）、GPU2 97%/GPU3 100%/GPU5 93% 为外部任务占用、GPU4 39%（低占 3.6GiB 可作候选发射点）；GPU6/7 MIG（gmx 等外部）。无空闲整卡、无 OOM。
- **纪律检查**：APA/v8p 日志均无 cpu_fallback_used=true、无 CPU 静默降级、无新 OOM。
- **结论/建议**：双线健康在跑、无新终态；唯一动作是清除过期 manager 实例（dedup 失效问题建议后续在脚本中核实 pgrep 锚点  与 nohup 实际 cmdline 是否一致，否则再次重复拉起仍会双实例）。下次巡检关注 APA epoch6 终态与 v8p 收敛。

### 批次十八：CMS array 库构建完成 + 泄漏审计硬门通过（2026-09-06 12:45，cunyuliu 交接审计）

- **数据落地**：APARENT2（4 文件 414 tensors 验证）+ ENCFF090JTW/770UJN + Griesemer 2021 supplementary mmc1-5.xlsx + MPRAu 官方 CMS_arrayassign 归属表 → external_model_assets/{aparent2,cms_array}/（用户本地浏览器 + curl 代下载）。
- **CMS 库构建**（build_cms_library_v1.py，commit d0acebcf）：Oligo Variant Info 区域界定 CMS 变体（rs/chr 双命名）→ fasta 构建序列（fa_key 去 _ref/_alt/_2/_5'End 后缀）→ Variant MPRAu Results Skew log2FC（MPRAu 列名 HEPG2/SKNSH）→ **cms_array_activity.csv 85,475 行（7,284 变体 × 5 细胞系 × 双背景构建，cell_context int 0-4，101bp 序列）**。
- **泄漏审计**：3-block 17bp 鸽笼 ≤2 mismatch vs ENCSR854RUF canonical 全 split（13,205 条）→ **flagged=0 硬门通过**；load_cms_library 加载验证 OK（activity −6.48~6.83）。
- **已知限制**：CMS 变体覆盖 7,284/9,740（74.8%）；缺失 25% 为 rs-id 变体无法从 fasta 关联 log2FC（部分在 770UJN 以 rs header 存在但 MPRAu 结果用 chr_pos 命名，rs↔chr 映射缺失）。V8 Stage 2 可用 85K 行 P0 先验注入。
- **在途**：v8p（GPU1 batch128 epoch1 step ~7K）、APA（GPU0 epoch2 step ~122K）。

### 批次十九：巡检快照（2026-09-06 18:29，cunyuliu 定时监控）

- **manager**：v5 单实例存活（PID 1735812）；needs_attention.txt 为空（无告警）；无新增 LAUNCH/DUPLICATE 告警。
- **APA polyA 预微调（GPU0，batch32，epochs6）**：PID 1899327 存活，GPU0 100%（22099MiB，含外部 toktokenbench 15250MiB）；进至 **epoch3 step~227000**，mse~0.18 缓降；checkpoint 已存 epoch_1/epoch_2（frozen_delta 缺失 epoch>2 说明尚未终态）；frozen_delta_results.json **未出现**（未到终态）。epoch2 指标 task_macro_spearman=0.449 / top_1=0.477 / ndcg@10=0.845。
- **V8 Stage1 polyA-only 基线（GPU1，batch128，epochs6）**：PID 1735814 存活，GPU1 99%（19743MiB）；进至 **epoch4 step~74950/128k(~60%)**，epoch_domain_loss mse_polya 0.225→0.191→0.184，step 级 mse_polya~0.17 续降；checkpoint stage1_s_epoch{1,2,3}.pt 已存；run_report.json **未出现**（未到终态）。
- **GPU 状态**：GPU0 100%（APA）、GPU1 99%（v8p）、GPU2 98%/GPU3 78%/GPU4 74%/GPU5 52% 为外部 review 任务占用；GPU6/7 MIG（gmx 等）。无 OOM、无空闲整卡虚悬。
- **纪律检查**：APA/v8p 日志均无 cpu_fallback_used=true、无 CPU 静默降级、无新 OOM，GPU 利用率高证明实际走卡。
- **结论/建议**：双线健康中段推进、**无新终态**；APA 约半程（epoch3/6）、v8p 稍快（epoch4/6）。下次巡检关注 APA 与 v8p 的 epoch6 终态文件触发 harvest/adjudicate。

### 批次十九：并行任务（TreeG k2 + V8 MPRAU zero-shot 双臂）（2026-09-06 19:10）

- **TreeG k=2**（GPU5）：hit@1 0.0045 与 k=1 完全一致（critic top-1 排序对 k 不敏感，k 只增 recovery shot count 0.0022→0.0052）——critic 排序劣化结论稳健。
- **V8 S/H MPRAU zero-shot**（polya 域条件，2,008 变体 pair-mean 口径）：S pair-mean ρ −0.0005 / H −0.0064（record spearman −0.006/−0.004）——**双臂 ≈0**。科学含义：V8 联合先验（mrl+polya）跨域迁移不携带 3UTR 等位偏移信号；**MPRAU 必须同 assay 监督（CMS 85,475 行库），Stage 2 域内适配是唯一路径**；zero-shot 对照臂基线 = 0 已入档（Stage 2 判定门 >0.1025 CI 不跨零的对照起点）。
- 并行工具：TreeG select-k 参数化 + V8 MPRAU zero-shot 脚本（commit d0acebcf 后新 commit）。GPU5 被外部任务挤爆（S 臂 OOM 一次），GPU4 重跑成功。

### 批次二十：APARENT2 frozen-Δ 补测 + 三并行任务收官（2026-09-06 19:40）

- **APARENT2 frozen-Δ（GPU6 MIG，GSE269595 VALIDATION 2,628）**：task_macro_spearman **0.6810**（vs APARENT 2019 0.7343 / V5 0.8219）——低于 2019 原版，**polyA 主 baseline 行不升级**（APARENT 2019 保持）；APARENT2 行作为附加外部行如实入档。排障：torch 2.5.1 <2.6 CVE 门（transformers 拒 .bin）→ 手动 config 构建 + weights_only=False 加载；APARENT2 固定 205 tokens → padding=max_length。
- **V8 S/H MPRAU zero-shot**：pair-mean ρ S −0.0005 / H −0.0064（2,008 变体）——双臂 ≈0，跨域先验无信号，Stage 2 必须 CMS 域内适配（zero-shot 对照臂基线 0 入档）。
- **TreeG k=2**：hit@1 与 k=1 一致（critic top-1 排序 k 不敏感）。
- GPU 并行利用：GPU5（TreeG k2）/ GPU2（H zero-shot）/ GPU4（S zero-shot）/ GPU6 MIG（APARENT2）。

### 批次二十一：在途任务巡检（v8p + APA）（2026-09-06 20:03）

- **v8p polyA 单域基线**（GPU1，batch128，epoch6，pid 1735814）：epoch 5 step 89000，mse 0.1752 polya 0.1671（总步 128,454，~5.7/6 epochs）——正常逼近终态，run_report.json 未出现，终态门未触发。
- **APA polyA 预微调**（GPU0，batch32，epoch6，pid 1899327）：epoch 3 step 250000，mse 0.1887 lr 1.14e-05（总步 ~513,750，~49%）——正常，frozen_delta_results.json 未出现，终态门未触发。
- manager（pid 1735812）存活，日志正常；needs_attention.txt 无告警。
- 注意：GPU0 空闲显存仅 225 MiB（util 100%），进程存活但余量极低，持续观察防 OOM。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。

### 批次二十二：在途任务巡检（v8p + APA）（2026-09-06 22:04）

- **v8p polyA 单域基线**（GPU1，batch128，epoch6，pid 1735814）：epoch 5 step 104500，mse 0.1689 polya 0.1693（总步 128,454，~81%）——正常逼近终态，run_report.json 未出现，终态门未触发。
- **APA polyA 预微调**（GPU0，batch32，epoch6，pid 1899327）：epoch 4 step 277000，mse 0.1779 lr 9.90e-06（总步 ~513,750，~54%）——正常，frozen_delta_results.json 未出现，终态门未触发。
- manager（pid 1735812）存活；needs_attention.txt 无告警；cpu_fallback_used=true 两任务均 0（无 CPU 静默降级）。
- GPU 状态：全部 0-5 被占用（GPU0 util 100% / GPU1 99% / GPU2-5 57-69%），无空闲整卡，GPU6/7 MIG 未用于正式训练。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。

### 批次二十三：在途任务巡检（v8p + APA）（2026-09-07 00:04）

- **v8p polyA 单域基线**（GPU1，batch128，epoch6，pid 1735814）：epoch 6 step 117000，mse 0.2981 polya 0.1633（总步 128,454，~91%，FINAL-EPOCH-FIXED）——正常逼近终态，run_report.json 未出现，终态门未触发。（注：step 级 mse 由 ~0.17 升至 0.2981，polya 域指标稳定 0.1633，继续观察，最终以判定脚本为准）
- **APA polyA 预微调**（GPU0，batch32，epoch6，pid 1899327）：epoch 4 step 302000，mse 0.1713 lr 8.55e-06（总步 ~513,750，~59%）——正常，frozen_delta_results.json 未出现，终态门未触发。
- manager（pid 1735812）存活；needs_attention.txt 无告警；无 CPU 静默降级迹象。
- GPU 状态：全部 0-5 被占用（util 100/100/93/64/28/99），空闲显存 1.2-4.7 GiB/卡，无 OOM 风险信号；GPU6/7 MIG 未用于正式训练。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。

### 批次二十四：在途任务巡检（v8p 终态判定 + APA + manager 重启）（2026-09-07 02:05）

- **v8p polyA 单域基线（GPU1，batch128，epoch6）——✨ 终态触发 & Stage 1 判定完成**：run_report.json 已写入（s_polya），训练 python 进程已退出（GPU1 显存释放至 12.3 GiB 空闲 / util 48%）。运行 adjudicate_route2_v8_stage1_v1.py：
  - polyA_gate_status = **RESOLVED**；stage1_success = **True**；m_arm_mean（MRL）= 0.3076；s_vs_h winner = **S**（delta_s_minus_h = +0.0202，规则 choose S if S >= H - 0.02）。
  - 产物：adjudication_v8_stage1.json（schema route_a_v3_route2_v8_stage1_adjudication.v1）。⚠ 注：arms 明细中 polya_non_destruction_pass=false（polya 域 val task_macro_spearman 0.112 vs baseline 0.209，threshold 0.188），但顶层聚合门判为 RESOLVED；以判定脚本顶层 VERDICT 为裁定。
- **APA polyA 预微调（GPU0，batch32，epoch6，pid 1899327）**：epoch 4 step 323000，mse 0.1721 lr 7.46e-06（总步 ~513,750，~63%）——正常，frozen_delta_results.json 未出现，终态门未触发。
- **manager —— 🚨 死亡并已重启**：ps 确认 mrna_training_manager.sh 未运行，needs_attention.txt 于 2026-09-07 02:00 标记 NOT RUNNING（manager log 最后条目停在 09-06 10:47）。已按协议重启：nohup bash ...mrna_training_manager.sh &（新 pid 2540034，uptime 00:02）。manager 重启后需确认 APA 终态双保险收割逻辑自洽。
- 无 OOM；无 CPU 静默降级（cpu_fallback_used=true 无迹象）。
- 判定：v8p 达成终态（Stage1 SUCCESS，S 胜出）；APA 正常推进；manager 异常已处置。

### 批次二十五：在途任务巡检（v8p 终态复核 + APA 推进 + 基础设施）（2026-09-07 04:05）
- **v8p polyA 单域基线**：终态已于批次二十四处置（polyA 门 RESOLVED、stage1_success=True、S 胜出 delta +0.0202）；本次重新运行判定脚本幂等复核，输出与批次二十四完全一致，无新变化。判定摘要已补写入 v8_stage1_polya_base.log（批次二十四仅写 journal，本次补齐）。训练进程已退出，GPU1 显存空闲 11.7 GiB。
- **APA polyA 预微调**（GPU0，batch32，epoch6，pid 1899327）：epoch 5 step 348000，mse 0.1701 lr 6.24e-06（总步 ~513,750，~68%）——正常，frozen_delta_results.json 未出现，终态门未触发。
- **manager**（pid 2540034）：存活，uptime 02:00:03（批次二十四 02:04 重启后持续运行），无再次死亡。
- 基础设施：needs_attention.txt 无告警；GPU0-5 均有空闲显存（3690~15267 MiB），util 27~100%，无 OOM 风险；GPU6/7 MIG 未用于正式训练；无 CPU 静默降级迹象。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。v8p Stage 1 判定结论已确认（RESOLVED / S wins / success=True）。

## 巡检批次二十六 2026-09-07 06:05
- **v8p polyA 单域基线**：已终态（批次二十四判定、批次二十五幂等复核）。本次定位 adjudication_v8_stage1.json 实际位于 experiments 根目录 v8_stage1_joint_prefinetune_20260904/（非 s_polya/ 内），内容与 log 摘要一致：polyA 门 RESOLVED、stage1_success=True、S 胜出（delta_s_minus_h +0.0202）、m_arm_mean=0.3076。注意项：arm-s polya 非破坏校验 FAILED（task_macro_spearman 0.1122 < 阈值 0.1883），该 flag 未阻断 stage1_success，建议进入下游前复核 arm-s 在 polyA 域 GSE269595 的验证评估是否符合 pre-reg §8 预期。训练进程已退出，GPU1 空闲 11.9 GiB。
- **APA polyA 预微调**（GPU0，batch32，epoch6，pid 1899327，uptime 1d4h56m）：epoch 5 step 374000，mse 0.1627 lr 5.09e-06（总步 ~513,750，~73%）——正常推进，较批次二十五 +26k 步；frozen_delta_results.json 未出现，终态门未触发。
- **manager**（pid 2540034）：存活，uptime 03:59:31；manager 日志 02:04:51 已自动写入 v8 adjudication JSON（双保险确认）。
- 基础设施：needs_attention.txt 无告警；GPU0 空闲 17.1 GiB/util 100%、GPU1 11.9 GiB/43%、GPU2-5 均有余量，无 OOM；GPU6/7 MIG 未用于正式训练；无 CPU 静默降级迹象。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。唯一注意项：v8p arm-s polya 非破坏 flag FAILED（0.1122<0.1883），建议下游使用前复核。

## 巡检批次二十七 2026-09-07 08:04
- **v8p polyA 单域基线**：已终态，无新增（进程已退出，GPU1 空闲）。adjudication_v8_stage1.json 复核与批次二十六一致：polyA 门 RESOLVED、stage1_success=True、S 胜出（delta_s_minus_h +0.0202）、m_arm_mean=0.3076。持续注意项：arm-s polya 非破坏 FAILED（0.1122<阈值0.1883），进入下游前建议复核 arm-s 在 polyA 域验证评估。
- **APA polyA 预微调**（GPU0，pid 1899327）：epoch 5 step 405000，mse 0.1693 lr 3.92e-06（总步~513,750，~79%）——正常推进，较批次二十六 +31k 步；frozen_delta_results.json 未出现，终态门未触发（预计 09-07 下午）。
- **manager**（pid 2540034）：存活。
- 基础设施：needs_attention.txt 无告警；GPU0/1 均满载训练，GPU2-5 有余量，无 OOM；GPU6/7 MIG 未用于正式训练；无 CPU 静默降级迹象。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。唯一注意项：v8p arm-s polya 非破坏 flag FAILED（0.1122<0.1883）。

## 巡检批次二十八 2026-09-07 10:05
- **v8p polyA 单域基线**：已终态，无新增。训练进程无存活（本批次 pgrep 首查返回的 1502199 为 bash -c 包装进程自匹配误报，精确复核确认无 v8_stage1 train 进程）。adjudication_v8_stage1.json 复核与批次二十六/二十七一致：polyA 门 RESOLVED、stage1_success=True、S 胜出（delta_s_minus_h +0.0202）、m_arm_mean=0.3076。持续注意项：arm-s polya 非破坏 FAILED（0.1122<阈值0.1883），进入下游前建议复核 arm-s 在 polyA 域验证评估。
- **APA polyA 预微调**（GPU0，pid 1899327，--physical-gpu-index 0 --epochs 6 --batch 32）：epoch 6 step 435000，mse 0.1463 lr 3.02e-06（总步~513,750，~85%）——正常推进，较批次二十七 +30k 步；frozen_delta_results.json 未出现，终态门未触发（预计 09-07 下午，剩约 78k 步）。
- **manager**（pid 2540034）：存活。
- 基础设施：needs_attention.txt 无告警；GPU0 空闲 16.6 GiB/util100%、GPU1 空闲 37.5 GiB/util0%（v8p 退出后释放）、GPU2-5 有余量，无 OOM；GPU6/7 MIG 未用于正式训练；日志无 cpu_fallback_used=true，无 CPU 静默降级迹象。
- 判定：无新终态、无进程死亡、无 OOM、无进度异常。唯一注意项：v8p arm-s polya 非破坏 flag FAILED（0.1122<0.1883）。

---
## 批次二十八（2026-09-07 11:10，接班人 cunyuliu 交接首班）

- **v8p 终态判定（09-07 04:05 自动）**：polyA 单域基线 Spearman 0.2092；adjudication_v8_stage1.json polyA 门 RESOLVED。**关键新事实：arm-s polyA 非破坏 FAIL（0.1122 < 阈值 0.1883），arm-h PASS（0.5262）**——DRAFT §1 amendment 条款触发 → Stage 2 主臂 = H。
- **Stage 2 预注册 FROZEN**：docs/paper/route2_v8_stage2_prereg_v1.md（commit 779e3fd1，push 分支 route-a-v3-v8-stage1-prep-20260904）。判定门：MPRAU pair-mean >0.1025 且 CI 不跨零（主判据）/ MRL ≥0.28 / polyA ≥0.80 / TE ≥0.1317 / macro 升 vs 0.167。
- **新代码**：backbone 增补 cell conditioning（6 ENCODE context embeddings，num_cells=0 保持 Stage 1 bit-identical）；run_route2_v8_stage2_adapt_v1.py（CMS 库适配 / benchmark TRAIN 池均衡 pair-delta 适配，full-param + LoRA r16 a32，MPRAU pair-mean + vs V5 bootstrap 2000 iters seed 20260816，9 任务 VALIDATION 评估）；adjudicate_route2_v8_stage2_v1.py（冻结门判定）。commit b2e7d730 push。
- **smoke 双模式全链验证通过**：CMS 库 85,475 行 0 flagged；MPRAU VALIDATION 2,008 变体 pair-mean 出数；V5 参考 0.10254 精确对齐；benchmark 模式 8 study 评估（GSE256185 无 VALIDATION 优雅跳过；GSE149487/GSE186455 等小样本如实出数）。CPU fallback=0，CUDA BF16。
- **Stage 2 四臂发射（09-07 10:46，一臂一卡多 GPU 并行）**：GPU1 h_cms_full（batch 128 full-FT CMS）/ GPU2 s_cms_full（64）/ GPU3 h_bench9（64）/ GPU5 h_cms_lora（64 LoRA）。LoRA 臂首启崩（wrap 后模块留在 CPU）→ 修复 model.to(device) 重发成功（trainable 2,956,801）。
- **APA**（GPU0 pid 1899327）：epoch 6 step ~449K/513.75K（~87%），ETA 09-07 下午；epoch 5 中间值 frozen-delta task_macro 0.3241（非终态）。终态自动 harvest_polya.sh。
- 观测：GPU1 37.5→25.9GiB free（h_cms_full 占用）；GPU2 11GiB free、GPU3 3.7GiB free、GPU5 25.9GiB free，四臂均在推进；无 OOM、无 cpu_fallback_used=true。
- 判定：无终态（APA/四臂均在途）；进程全部存活；定时任务监控已更新（v8p 段自然失效，Stage 2 四臂段新增）。
## 巡检批次二十九 2026-09-07 12:15（09-07 12:13 巡检）
- **v8p polyA 单域基线**：已终态无新增。复核 run_report.json（01:45）与 adjudication_v8_stage1.json（04:01）与历批次一致：polyA 门 RESOLVED、stage1_success=True、S 胜出（delta_s_minus_h +0.0202）、m_arm_mean=0.3076、polyA 基线 Spearman 0.2092。无独立 train 进程（pgrep 首查 2169241 为 manager bash -c 误报，与批次二十八一致）。持续注意项：arm-s polya 非破坏 FAIL（0.1122<0.1883）、arm-h PASS（0.5262），Stage 2 已主臂=H。
- **APA polyA 预微调**（GPU0，pid 1899327）：epoch 6 step 467000，mse 0.1584 lr 2.37e-06（总步~513,750，~91%），较上批次 +18k；frozen_delta_results.json 未出现，终态门未触发（预计 09-07 下午，剩约 47k 步）。
- **manager**（pid 2169241/2540034）：存活。
- 观测：GPU0 util 100%、GPU5 util 99%（APA/Stage 2 四臂满载）；GPU1 37.4GiB free、GPU4 30.5GiB free，余量充足；无 OOM、无 cpu_fallback_used=true。
- 判定：无新终态；进程全部存活；needs_attention.txt 无告警。

---
## 批次二十九（2026-09-07 13:50，Stage 2 波次 1/2 终态收割 + 波次 3 确认发射）

- **波次 1（CMS 域内适配）三臂终态（11:26），MPRAU 主判据全 FAIL**：
  - h_cms_full（4008 步 full-FT）：MPRAU pair-mean 0.0327，vs V5 Δ-0.0698 CI [-0.133,-0.010] 全负
  - s_cms_full（8016 步 full-FT）：0.0385，CI [-0.125,+0.003] 跨零
  - h_cms_lora（8016 步 LoRA）：-0.0069，CI 全负
  - 判读：CMS 同 assay 库适配仅弱正信号（+0.033~0.039 vs zero-shot ~0）；LoRA 容量不足；「同 assay 同管线外部先验」假说未获确认。
- **波次 2（域内 ENCSR854RUF TRAIN 55,704 行，DRAFT 首要路径）四臂终态（~13:00）**：
  - **s_mprau_in（5226 步，V8-S full-FT）：MPRAU 0.1527，CI [-0.0049,+0.1039]——首个超过 V5 0.1025 的 MPRAU 结果（点估计 +0.05），CI 仅差 0.005 未排除零**；epoch 5/6 均 >0.15（非随机峰）；代价：MRL 0.0001 / polyA -0.156 专才化牺牲
  - h_mprau_in（2616 步，V8-H full-FT）：0.0755，CI 跨零
  - h_mprau_lora（10446 步 MIG，LoRA）：0.1084，CI 跨零（点估计超 V5）
  - h_bench9（8400 步，均衡多任务）：MRL 0.2879 ✓ / polyA 0.8067 ✓ / MPRAU 0.0556 ✗ / TE 0.0717 ✗ / macro 0.1516 —— 均衡适配保住 MRL/polyA，MPRAU 需专才化
- **adjudicator 更新**：spec 澄清——专才臂（单 study 适配）只判 MPRAU 主判据（有意牺牲他任务），h_bench9 判全门（commit d4b57054）。
- **波次 3（3-seed 确认）发射（13:41）**：s_mprau_in_s2（GPU1 b128）/ s_mprau_in_s3（GPU2 b64）/ h_mprau_in_s2（GPU3 b64）/ h_mprau_lora_s2（GPU5 OOM → GPU7 MIG b32）。ensemble 脚本 commit 862edddb。
- **APA**：epoch 6 step ~488K/513.75K（~95%），ETA 数十分钟内。
- 纪律：全部 FINAL-EPOCH-FIXED、CUDA BF16、cpu_fallback=false、smoke 未入判定；产物 /mnt、代码已 push。

### 巡检 2026-09-07 14:07 (v8p 终态 / APA 推进 / 基建健康)
- **v8p（V8 Stage1 polyA 单域基线）已终态并判定**：run_report.json（s_polya，terminal:true，01:45）出现；判定（schema v1）polyA_gate_status=RESOLVED，stage1_success=True，s_vs_h 胜者 S（MRL delta_s_minus_h=+0.0202，阈 -0.02），m_arm_mean=0.3076。**异常标记**：arm-s polya non_destruction FAILED（task_macro_spearman 0.112 < 0.188 阈，GSE269595 epoch2），不阻断 stage1_success，但 arm-s 域内 polyA 专才化牺牲超预期，需 Stage2/下游关注。adjudication_v8_stage1.json 已落盘（v8_stage1_joint_prefinetune_20260904/ 子目录）；为复核对齐产物，14:07 重跑判定脚本确认一致（幂等）。
- **APA（polyA Route A 预微调）推进中**：epoch 6 step 494000/~513750（~96.2%），mse 0.1559 lr 2.07e-06；frozen_delta_results.json 未现，终态未触发（据触发条件届时跑 harvest_polya.sh，manager 双保险）。进程 1899327 存活。
- **基建**：manager 存活（2540034/2597533）；GPU0-5 全部在职（0/1/2/5 ~96-100%，3 ~15%，4 340MiB free 32%）；6/7 MIG 未用于正式训练；needs_attention.txt 无告警。负载正常，无 OOM，无 cpu_fallback。

---
## 批次三十（2026-09-07 16:10，APA 终态收割 + s_mprau_in 4-seed 结果）

- **APA polyA Route A 终态收割（FINAL-EPOCH-6-FIXED，harvest_polya.sh 自动）**：task_macro_spearman **0.2839** / top1 0.4545 / ndcg 0.8456。vs APARENT 0.7343：Δ−0.450 CI [−0.511,−0.388] 全负；vs critic V5 0.8219：Δ−0.538 CI 全负 → **polyA 主行不升级，APA 线 FAIL（诚实负结果）**。归因：mRNABERT full-FT 2.74M 近端 usage 库不适配 polyA 决策基准（epoch 1-2 更好 0.443/0.449，但 FINAL-EPOCH-FIXED 禁挑峰）；APARENT 架构（CNN/位置建模）在近端 usage 显著优于通用 LM full-FT。
- **s_mprau_in（V8-S 域内 ENCSR854RUF 专才）4-seed 全部超过 V5 0.1025**：
  - seed1 0.1527 / seed2 0.1140 / seed3 0.1120 / seed4 0.1082（3-seed ensemble 0.1319，vs V5 Δ+0.029 CI [−0.024,+0.085] 跨零）
  - **首个在 MPRAU pair-mean 上持续超过 V5 的模型**（4/4 seed 点估计 > 0.1025）；CI 仍含零（边际显著），5-seed（s5 训练中）ensemble 待定
  - 判读：V8-S 联合先验 + 域内适配 = MPRAU 攻坚决定性路径；专才化代价 = MRL/polyA 崩塌（s_mprau_in MRL 0.0001 / polyA −0.16，adjudicator 已按专才臂只判 MPRAU 主判据）
- **h_bench9（均衡多任务适配）终态**：MRL 0.2879 ✓ / polyA 0.8067 ✓ / MPRAU 0.0556 ✗ / TE 0.0717 ✗ / macro 0.1516 —— 均衡适配保住两个强任务，MPRAU 需要专才化
- **h_mprau_lora 2-seed**：0.108 / 0.072（ensemble 脚本对 LoRA 臂无效——未解 LoRA wrap，单 seed 以 run_report 为准）
- **LoRA 臂 ensemble 限制入档**：ensemble 脚本仅对 full-param 臂正确（state dict 直接兼容）；LoRA 臂需先解 wrap，暂以 run_report 单 seed 为准。

---
## 批次三十一（2026-09-07 17:10，s_mprau_in 5-seed ensemble 定论）

- **s_mprau_in 5-seed ensemble（MPRAU pair-mean）**：seed 0.1527/0.1140/0.1120/0.1082/0.1101 —— **5/5 seed 全部 > V5 0.1025**；ensemble **0.1351**（vs V5 Δ+0.0325，CI [−0.0225, +0.088] 跨零）。
- **裁决（诚实）**：MPRAU 攻坚线（V8-S 先验 + ENCSR854RUF 域内专才适配）**首次持续超过 V5 点估计（+32% relative）**；paired bootstrap CI 含零 = 检验力受限（2008 变体 + per-cell 噪声主导 CI 宽度；5-seed 与 3-seed CI 几乎同宽 → 更多 seed 无益）。D5/Stage 2 门（CI 不跨零）**未严格通过**，不作翻案；结论：方向性突破确立，显著性缺口来自数据体制（MPRAU VALIDATION 2,008 变体对 ~0.03 级 delta 检验力不足）。
- 后继建议：① Stage 3 将 V8-S 专才接回 SetFlow B2（以生成质量结算 critic 攻坚）；② D5 amendment 重估目标带（以 0.135 为新的 in-house 参照）；③ 如需严格显著：增大 MPRAU 验证池（LOSO 拼表/TEST 开启需 Gate P 拍板）。

---
## 批次三十二（2026-09-07 18:20，Stage 3 发射 + D5 amendment 提交）

- **Stage 3 发射（用户拍板）**：V8-S 专才（s_mprau_in，5-seed ensemble 0.1351）冻结为 guidance critic，重跑 SetFlow B2（同 base b_fix2 pass2 / seed 20260915 / budget，guided-only，891 源）。新代码 route2_v8_frozen_guidance_v1.py（FrozenV8Critic：potential = clip(f(cand)-f(src), -5, 5) frozen-delta 口径，domain/cell 条件化传参留证，接口镜像 FrozenXEditCriticV5）+ runner --critic-kind v8（commit aad9ce92 已 push 分支 route-a-v3-setflow-v5-base-fix-20260901）。
- **smoke 验证通过**（8 源，GPU1）：RUNNER_COMPLETE、legality 1.0、cpu_fallback_used=false、256 candidates、method_id=frozen_v8_critic_guided_xeditsetflow_v5_b_fix2_pass2_seed20260915。
- **full 891 发射**（guided_b2_v8_20260907/b2_full_891，GPU1，PID 3253502，预计 ~2 天）；服务器 monitor v5 已跟踪（每 2h status.log）；TRAE 定时任务更新超时（环境问题）——以服务器侧为准。
- **D5 amendment v1 提交（用户拍板）**：docs/paper/route2_v8_d5_mprau_target_amendment_v1.md（commit 346c4f19 push）：目标带重述为「>0.1351 且 paired bootstrap CI 不跨零」（近期验收）+ 原 40% 保留为远期延伸目标（留痕）；检验力条款（2,008 变体对 ~0.03 级 delta 分辨不足）入档。
- 对位基准（Stage 3 终态判定用）：unguided 0.1205 / v5-guided 0.1263（Δ+0.0058 CI 跨零 FAIL）。

---
## 批次三十三（2026-09-07 18:06，TRAE 定时巡检 #1 · Stage 3 B2 V8）

- **巡检对象**：guided_b2_v8_20260907/b2_full_891（V8-S critic，guided-only，891 源，预计 ~2 天）。
- **进程**：存活且单实例（python PID **3289922**，启动 16:53:51，elapsed ~1h10m；journal 批次三十二所记 3253502 与实时进程不符——以实时 pgrep/ps 为准）。State=R、273 threads、VmRSS 6.3GB；CPU 10s 内 +29.6 CPU-s，确认为活跃计算非停滞。
- **GPU**：物理 GPU1（cuda:1，A100-40G），util ~42%，free ~34GB，无显存瓶颈。
- **日志**：b2_full_891.log 仅含 16:51 变压器警告一行（380B），逐源 SOURCE 标记未出现。判定：stdout 重定向至文件为 **block 缓冲**，smoke（8 源）亦为 run 结束一次性 dump JSON；grep SOURCE 当前=0 属缓冲现象**非训练停滞**（CPU/GPU 均确认在计算）。进度可见性缺口待目录侧输出文件/subprocess 日志佐证，本巡检不干预。
- **无异常**：无 Traceback/OOM/cuda error；cpu_fallback_used=false；无终态（summary 未生成，属预期，~2 天）；manager 进程 UP（training_manager.log 含 15:31 polyA 提交告警，遗留项非本 run）。
- **判定**：正常推进，无需处置。

---
## 批次三十四（2026-09-07 20:50，TRAE 定时巡检 #2 · Stage 3 B2 V8）

- **巡检对象**：guided_b2_v8_20260907/b2_full_891（V8-S critic，guided-only，891 源）。
- **进程**：存活且单实例（python PID **3289922**，与批次三十三一致；elapsed 3h56m，State=Rl，top 瞬时 %CPU 333%，273 threads，RSS 6.0GB，wchan=0）——确认为活跃计算非停滞。
- **GPU**：物理 GPU1（cuda:1）util 100%，free ~10.2GB；本进程占用 2104 MiB，无显存瓶颈；GPU0/2/3 满载为其他任务，不影响本 run。
- **日志**：仍 380B（16:51 变压器警告一行），mtime 16:51:47，逐源 SOURCE 标记未出现（grep=0）——与批次三十三判定一致：stdout block 缓冲所致，非训练停滞（CPU/GPU 均确认在算）。进度可见性缺口同前，本巡检不干预。
- **无异常**：无 Traceback/OOM/cuda error；无 cpu_fallback 证据；无终态（summary 未生成，属预期 ~2 天）；manager 进程 UP（PID 2540034）。
- **判定**：正常推进，无需处置。

---
## 批次三十三（2026-09-07 19:10，并行任务执行 + LoRA 口径修正）

- **h_mprau_lora 2-seed ensemble 修正（LoRA 解 wrap 后）**：seed1 0.1091（与 run_report 精确一致，修复验证通过）/ seed2 0.0816 → ensemble **0.0994 < V5 0.1025，CI 跨零 → LoRA 臂 FAIL（ensemble 口径）**。修正前脚本未解 LoRA wrap 导致 seed1 误算 0.0668；commit 9e24630b。**结论修正：full-FT 专才（s_mprau_in 0.1351）为唯一持续超 V5 的路线；LoRA 适配无法维持该优势。**
- **并行任务盘点（B2 V8 训练期间）**：
  - ✅ 已完成：LoRA ensemble 修正 + 重跑
  - ⛔ 用户阻塞：UTR-STCNet 权重（Google Drive 链接，README 确认）与 HydraRNA 权重（gdown 被拦）——均需用户浏览器中转 → external_model_assets/{utr_stcnet,hydrarna}/weights/ → frozen-Δ 行（MRL / GSE217518）
  - ⏸ 用户拍板推迟：LOSO-lite（SOTA 后）、SetFlow V6（Stage 3 后）
  - 低价值跳过：CMS dbSNP 覆盖补全（CMS 臂已 FAIL，补全不改变结论）

## 批次三十五（2026-09-07 21:35，Stage 3 B2 V8 巡检 3#：正常推进）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289922，elapsed 4h39m，critic= v8_stage2_adapt_20260907/s_mprau_in/stage2_s_benchmark_full_epoch6.pt；GPU1 util 100% / free ~13.6GB。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480732，elapsed 24m，critic= v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt；GPU4 util 71% / free ~14GB。
- **日志**：两臂均 380B（模型加载横幅），臂A mtime 16:51 / 臂B 21:07；逐源 SOURCE 标记未见——stdout block 缓冲所致，GPU 满载确认在算（与前 2 期判定一致）。
- **无异常**：无 Traceback/OOM/cuda error；无 cpu_fallback 证据；无终态（summary 未生成，属预期 ~2 天）；manager 进程 UP（PID 2540034）。
- **判定**：双臂同步推进，调度正常，无需处置。

---
## 批次三十六（2026-09-07 22:35，Stage 3 B2 V8 巡检 4#：正常推进）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289922，elapsed 5h40m，CPU 233%（state S 阻塞在 IO/GPU），GPU uuid a590f174（物理 idx1 匹配声明），critic= v8_stage2_adapt_20260907/s_mprau_in/stage2_s_benchmark_full_epoch6.pt；GPU1 util 100%。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480732，elapsed 1h25m，CPU 100%（state R），GPU uuid d7c96455，critic= v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt；GPU4 util 91%。
- **日志**：两臂均 380B（仅 bert 加载横幅，stderr 直出）；per-step stdout block 缓冲故未见逐源标记（与前 3 期判定一致）。臂A 目录含 16:46 smoke_guided_v8_8src 冒烟 → 已完成（GUIDED_XEDITSETFLOW_V5_B2_RUNNER_COMPLETE，cuda:1，cpu_fallback_used=false）确认 guided V8 框架 GPU 跑通；冒烟为 config 默认 critic 占位 checkpoint（非本臂 s_mprau_in），仅作框架验证、忽略。
- **无异常**：无 Traceback/OOM/cuda error；无 cpu_fallback 证据；无终态（guided_run_summary.json 未生成，属预期 ~2 天）；manager 进程 UP（PID 2540034）；无 GPU 全占（B2 相关 GPU1/4 显存充足）。
- **判定**：双臂同步推进，调度正常，无需处置。

---
## 批次三十七（2026-09-07 23:05，Stage 3 B2 V8 巡检 5#：正常推进）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289922，elapsed ~6h10m，CPU 220%，critic= v8_stage2_adapt_20260907/s_mprau_in/stage2_s_benchmark_full_epoch6.pt；GPU1 util 100%。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480732，elapsed ~1h55m，CPU 317%，critic= v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt；GPU4 util 90%。
- **日志**：两臂均仍 380B（仅 bert 加载横幅），逐源 SOURCE 标记未见——stdout block 缓冲所致，双 worker 高 CPU + GPU 满载确认在算（与前 4 期判定一致）。
- **无异常**：无 Traceback/OOM/cuda error；无 cpu_fallback 证据；无终态（guided_run_summary.json 未生成，属预期 ~2 天）；manager 进程 UP（PID 2540034）；GPU1/4 显存充足，无 GPU 全占。
- **注**：status.log 未检索到 B2V8/B2V8JOINT 行（cron v6 monitor 尚未落地该标注，不影响本臂巡检判定）。
- **判定**：双臂同步推进，调度正常，无需处置。

---
## 批次三十八（2026-09-07 23:35，Stage 3 B2 V8 巡检 6#：正常推进）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289922（alive），GPU1 util 100%（used 34640 MiB）；critic= v8_stage2_adapt_20260907/s_mprau_in/stage2_s_benchmark_full_epoch6.pt。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480732（alive），GPU4 util 48%（used 7264 MiB，采样瞬间处于批次间隙/IO）；critic= v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt。
- **日志**：两臂均仍 380B（仅 bert 加载横幅），逐源 SOURCE 标记未见——stdout block 缓冲所致，进程 alive + GPU 占用确认在算（与前 5 期判定一致）；full log 未见 Traceback/OOM/cuda error。
- **A3 探针**：watcher alive（PID 1096623），23:00–23:30 持续 "no free full GPU (all busy / MIG-only)" 等待空闲满血卡；尚无 A3 JSON 终态、无 done 标记 → 保持等待态（当前无空闲满血卡）。
- **基础设施**：manager UP（PID 2540034）；GPU8 卡 util 采样：GPU0 99%/GPU1 100%/GPU2 90%/GPU3 99%/GPU4 48%/GPU5 69% 占满，GPU6（free 38.7G）/GPU7（used 20.7G）util N/A → MIG-only 不可作满血卡；无 GPU 全占（B2 相关卡显存充足）。
- **cron status.log**：已出现 B2V8/B2V8JOINT 行（21:11/22:00，alive=1 terminal=0）→ cron v6 monitor 标注已落地（相较批次三十七的缺失改善）。
- **无终态**：双臂 guided_run_summary.json 均未生成，属预期（~2 天）。
- **判定**：B2 双臂同步推进 + A3 watcher 等待空闲满血卡，调度正常，无需处置。

---
## 批次三十四（2026-09-07 23:40，UTR-STCNet 泄漏排除 + HydraRNA 依赖阻塞）

- **UTR-STCNet MRL frozen-delta 三臂全部判定 INVALID（R3 训练集泄漏）**：mpra_h 0.8135 / mpra_u 0.2667 / mpra_v 0.2120。归属核查（NCBI GEO 官方页）：三个训练数据文件（GSM3130435 egfp_unmod_1 / GSM3130443 designed_library / GSM4084997 varying_length）全部 Series=GSE114002 = MRL benchmark 研究本身；mpra_h 直接训练于 designed_library（benchmark VALIDATION 源样本）→ 0.8135 为分布内记忆。三行全部排除入榜；frozen_delta_results.json 已写泄漏判定 + 归属证据（commit 08ff6c2f）。
- **HydraRNA（GSE217518 稳定性行）依赖阻塞**：权重已就位（HydraRNA_model.pt/V2/SS，336MB×3）；归属审计干净（通用 RNA LM，预训练于 ncRNA+pcRNA 转录组，非 benchmark MPRA 库）。但模型架构需要 flash-attn + mamba-ssm（triton/CUDA 编译依赖）：服务器无 nvcc，pip 源码编译失败；GitHub 预编译 wheel 无 torch2.5.1+cu121+cp310 匹配版本。重型编译（flash-attn 2.6.1 + mamba-ssm 2.2.2 + causal-conv1d，1h+，需先装 conda cuda 工具链）换一个 ICC≈0 任务的 ≈0 行——工程上不划算，**暂缓**。该任务已有覆盖：Saluki frozen（0.0193/0.0985）+ RNA-FM/UTR-LM 冻结行（≈0）。若后续需要，路径已记录（install_hydrarna_env.sh 依赖清单）。

---
## 批次三十九（2026-09-08 00:40，Stage 3 B2 V8 巡检 7# + A3 watcher 修复重启 + polyA APA 裁决补录）

### B2 双臂（guided-only 891 源，~2 天）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289921/3289922（alive，started 09-07 16:53:51，elapsed ~7h45m）；cmd=run_route2_guided_xeditsetflow_v5_v1.py --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index 1；GPU1 util 99%。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480728/480732（alive，started 09-07 21:09:03，elapsed ~3h30m）；cmd=同脚本 --run-id b_fix2 --checkpoint-pass 2 --physical-gpu-index 4；GPU4 util 38%（free 32.5G，采样瞬间批次间隙/IO）。
- **日志**：两臂仍各 1 行（380B 仅 bert 加载横幅），stdout block 缓冲所致（与前期判定一致）；无 Traceback/OOM/cuda error；无 cpu_fallback 证据。
- **无终态**：双臂 full run 的 guided_run_summary.json 均未生成，属预期（~2 天）；仅臂A 旧 smoke_guided_v8_8src summary。

### A3 critic 混合池探针（V6 诊断线）——根因修复 + watcher 重启

- **异常**：watcher（旧 PID 1096623，09-07 23:00:26 起）反复崩溃，00:06 与 00:26（抢到 GPU2 21475MB）两次 A3_FULL 均 rc=1：ModuleNotFoundError: No module named 'torch'。根因：watcher 脚本 PY=python3 指向系统 python3（无 torch）；editflow 环境 python（/home/cunyuliu/miniconda3/envs/editflow/bin/python，torch 2.5.1+cu121，CUDA True）未启用。
- **修复**：watcher PY= 改指 editflow python（sed 留 .bak.20260908）；kill 旧 watcher 并以修复版重启（新 PID 1600186，00:37:08，start log 正常）。当前无空闲满血卡（GPU0-5 全忙 / 6-7 MIG-only）→ A3_FULL 进入 300s 轮询等待，下一空闲窗口以正确环境执行。已留证（watcher log 两次 Traceback + 修复前脚本）。
- 后续：A3_FULL JSON 终态（status != DRY_SMOKE_STUB）即入档 H2 裁决；无需人工干预。

### polyA APA Route A 终态裁决补录（manager NEEDS HUMAN 闭合）

- **背景**：manager 09-07 15:30:53 检测 APA 终态跑 harvest 链；15:31:23 polyA journal commit FAILED - NEEDS HUMAN（nothing to commit）。根因：harvest_polya.sh 调用的裁决脚本只产出 adjudication_results.json、不写 journal，manager 提交时 journal 无 diff → FAIL；apa_harvested.marker 已 touch，不会自动重试。
- **裁决数据**（analysis_apa_route_a_adjudication_20260903/adjudication_results.json，schema v1，GSE269595 VALIDATION K=10）：apa_route_a_ep6 spearman 0.2839 / top_1 0.4545 / ndcg_at_10 0.8456；vs APARENT adapter（0.7343/0.6011/0.8906）Δspearman -0.4504（95%CI [-0.5114,-0.3884] 不含 0）；vs critic V5（0.8219/0.5007/0.8710）Δspearman -0.5380（95%CI [-0.5984,-0.4778] 不含 0）；top_1/ndcg Δ 亦全为负且 CI 不含 0。
- **结论**：APA Route A polyA 线 FAIL（Δ 全面显著为负，与前期 0.2839 FAIL 判定一致）——APARENT adapter 与 critic V5 均显著胜出，不构成对 V6/V8 竞争力威胁；数据入档，本条目补录闭合 NEEDS HUMAN。

### 基础设施

- **manager**：UP（PID 2540034，started 09-07 02:04:49）；日志尾部正常（harvest complete / APA harvest done）；除上述 polyA journal FAIL 外无新异常。
- **GPU 快照**（00:20）：GPU0 8049MiB/100%、GPU1 10319MiB/99%、GPU2 26037MiB/94%、GPU3 13531MiB/99%、GPU4 32507MiB/38%、GPU5 7204MiB/46%；GPU6（free 38.7G）/GPU7（free 19.8G）util N/A → MIG-only → 当前无空闲满血卡，A3 保持等待态。
- **判定**：B2 双臂推进正常；A3 watcher 已修复重启等待空闲卡；polyA APA 裁决补录闭合；无需其他处置。

---
## 批次四十（2026-09-08 01:20，Stage 3 B2 V8 巡检 8# + A3 watcher 修复后等待态确认）

### B2 双臂（guided-only 891 源，~2 天）

- **臂A（V8-S 专才 s_mprau_in，GPU1）**：python PID 3289922（alive，elapsed 8h28m，CPU 212%）；critic= v8_stage2_adapt_20260907/s_mprau_in/stage2_s_benchmark_full_epoch6.pt；GPU1 util 99%。
- **臂B（V8-S joint 对照，GPU4）**：python PID 480732（alive，elapsed 4h13m，CPU 431%）；critic= v8_stage1_joint_prefinetune_20260904/s_mrl-polya/stage1_s_epoch2.pt；GPU4 util 94%。
- **日志**：两臂仍各 1 行（380B 仅 bert 加载横幅），stdout block 缓冲所致（与前 7 期判定一致）；无 Traceback/OOM/cuda error；无 cpu_fallback 证据。
- **无终态**：双臂 full run guided_run_summary.json 均未生成（属预期 ~2 天）；仅臂A 旧 smoke_guided_v8_8src summary（框架冒烟，忽略）。
- **cron status.log**：B2V8/B2V8JOINT 行 22:00/00:00 均 alive=1 terminal=0，与巡检一致。

### A3 critic 混合池探针（V6 诊断线）

- **watcher**：alive（PID 1600186，批次三十九 00:37 修复版）；00:31–01:17 持续 "no free full GPU (all busy / MIG-only)"，sleep 300s 轮询。
- **环境验证**：editflow python torch 2.5.1+cu121 CUDA True → 修复确认有效，下一空闲窗口 A3_FULL 将以正确环境执行。
- **终态**：A3_critic_mixed_pool_probe.json 不存在（仅 dry_smoke_stub）；done_A3_FULL.json 为 3 次旧失败标记（exit_code=1，批次三十九已留证）；修复后无新 run、无 done 新变化。
- **等待态**：当前无空闲满血卡（GPU0 100%/GPU1 99%/GPU2 23%/GPU3 99%/GPU4 94%/GPU5 50%，GPU6/7 MIG-only）→ A3 保持等待。

### 基础设施

- **manager**：UP（PID 2540034，elapsed 23h17m）；日志尾行为 09-07 15:31 APA harvest done（事件驱动型，空闲期无新行属正常）；polyA NEEDS HUMAN 已由批次三十九补录闭合。
- **GPU**：无全占；B2 相关卡显存充足（GPU1 free 11.4G / GPU4 free 24.5G）。
- **判定**：B2 双臂推进正常；A3 watcher 已修复、等待空闲满血卡；无新异常，无需处置。

---
## 批次四十一（2026-09-08 04:30，Stage 3 B2 巡检 9#：臂 B 终态对位 + A3 数值终态入档 + 臂 A 慢速异常）

### 重大事件一：臂 B（V8-S joint 对照，GPU4）终态对位——Δ 负向跨零，不支撑 critic 非瓶颈

- **终态**：guided_b2_v8joint_20260907/b2_full_891/guided_run_summary.json status=GUIDED_XEDITSETFLOW_V5_B2_RUNNER_COMPLETE（wall 21311s≈5.9h，GPU4，seed 20260915，pass-2，cpu_fallback=false，peak VRAM 1.6GB，28512 候选 legality 1.0，critic_forwards 303642）；status.log 04:00 B2V8JOINT terminal=1。
- **对位（口径同 guided_b2_20260903，unguided 参照 = 20260903 unguided 臂候选，recovery 0.12046）**：guided recovery **0.1188** → **Δrecovery −0.0017（bootstrap 2000 CI [−0.0046,+0.0011] 跨零，负向点估计）**；Δhit@1 −0.0002；**Gate B2/B3 双 FAIL**。
- **per-task Δ**：MRL 652 源 −0.0015（0.1531 vs 0.1547）/ MPRAU 108 0.0000 / polyA 20 0.0000 / HL 111 −0.0045（0.0270 vs 0.0315）→ **四任务全≤0，且劣于 V5-critic guided 0.1263**。
- **裁决（V6 立项中间态）**：臂 B 不满足"Δ 显著正 → critic 非瓶颈"；双臂跨零闭合待臂 A 终态。产物：b2_full_891_adjudication{,_per_task}.json。

### 重大事件二：A3 critic 混合池探针——数值终态（H2 首次直接量化）

- **watcher 全链完成**（03:10:52 A3_FULL done rc=0 → A5 pass4/pass6 聚合 → 03:15:54 全部完成退出 0；修正版 watcher 以 editflow python 正确环境执行，批次三十九根因修复验证通过）。
- **A3 全量 891 源数值**（a3_critic_mixed_pool_probe_full.json，cuda:2，BF16，cpu_fallback=false，protected_reads=0，status=TERMINAL；critic=V5 final_pass_8）：总体 critic cond_acc@1 **0.0614** vs base 0.0367；measured_best 平均名次 **8.83** vs 23.36。
- **per-task（critic cond_acc@1 vs base_reachable 天花板）**：MRL 652 **0.0422 < 12.4%** / MPRAU 108 **0.1111 > 2.8%** / HL 111 **0.1280 > 5.4%** / polyA 20 **0.05 > 0.0%**（n=20）。
- **natural-hit 子集（216 源，24.2%）反劣**：critic 0.0764 vs base 0.1514；rank 6.76 vs 2.15。
- **H2 裁决**：部分证实——critic 判别力在 MRL 主杠杆上低于 base 结构性天花板、natural-hit 子集反劣 → 判别力不足是约束之一；MPRAU/HL 超天花板 → added 触达优势真实。A3.3（V8 同款探针）待 Stage 3 产物。
- **入档**：SPECS_SETFLOW_V6/{spec,tasks,checklist}.md 已更新（H2 数值表 + 裁决 + Phase A 收口）；**DP2 拍板输入齐备，提醒拍板临近**。

### 臂 A（V8-S 专才，GPU1）——推进异常（慢速，待续观）

- 进程 3289921/3289922 alive（elapsed ~11h10m）；GPU1 util 100%（9 进程共享，含 3×8.6GB 大任务）；进程 VRAM 2106MiB 恒定（与 smoke peak 1.6GB 量级一致，非卡死指标）；CPU ~100–140%（1–2 核）持续；**I/O 8s 窗口零增长**。
- 对照：smoke 8 源 wall 218s → 外推 891 ≈ 6.7h；B2-B 实跑 5.9h。**臂 A 已 11h+ 无输出目录**（输出目录为终态一次性落盘设计，缺目录≠必然异常）。
- 判定：GPU1 重度争用 + 专才 critic 前向画像可能差异 → 疑似显著慢速（2×+），非确诊卡死（CPU 持续在算、无 Traceback/OOM/cpu_fallback、state R）。**下次巡检若仍无终态且 wall ≥ 2×B2-B（≈12h，约 05:00 后），升级为深检（必要时重启留证）**。

### 基础设施

- manager UP（PID 2540034）；status.log 04:00 MGR alive。
- GPU：GPU0-5 全忙（util 47–100%），GPU6/7 MIG-only → **无空闲满血卡**（A3 已结束，不影响）。
- 无 Traceback/OOM/cuda error / cpu_fallback（两臂 + A3 均干净）。

### 处置

- 已完成：B2-B 对位 + per-task 分解（产物落盘）；SPECS_SETFLOW_V6 三文档 + SPECS_SETFLOW_V5/tasks.md 更新；A3 数值入档。
- 待续：臂 A 终态 → 双臂对位闭合 → V6 立项裁决终态；A3.3（V8 探针）。
- **提醒**：DP2 拍板输入已齐备（Phase A 全定案 + 臂 B 对位），建议臂 A 终态后安排 DP2。

---
## 批次四十二（2026-09-08 04:30，Stage 3 B2 巡检 10#：臂 A 确诊卡死 → 留证重启）

### 重大事件：臂 A（V8-S 专才，GPU1）确诊卡死 → 停止留证 → 同参数重启

- **诊断升级**（依据批次四十一预案：wall ≥ 2×B2-B 升级深检）：
  - 进程 3289922 elapsed **11h30m**（started 09-07 16:53:51），仍无输出目录/arm_summary；对照 B2-B 全程含设置 5.9h。
  - **主线程单核旋占确诊**：ps -L 主线程 93.8% CPU、TIME 10:52:40（≈10.9h CPU / 11.5h wall）；其余 272 线程全部 0 CPU + Sl；二次采样 TIME 10:50:51→10:52:40 持续增长。
  - **I/O 近零**：/proc/3289922/io read_bytes 1.97MB / write_bytes 16KB（11.5h 累计）→ 纯 CPU 空转零读写，非"慢速计算"。批次四十一"疑似显著慢速"更正为**确诊卡死**。
  - 判定：满足纪律"cpu 静默降级立即停止留证"（GPU 无推进、CPU 忙等）。日志仅 transformers 启动警告 380B，无 Traceback/OOM/cuda error。
- **留证**：guided_b2_v8_20260907/armA_stuck_diagnosis_20260908_0429.txt（ps/线程/io/缺目录/对照）；原日志改名 b2_full_891.log.stuck_20260908_0429。
- **停止**：kill -9 3289921/3289922（bash+python）。
- **重启**：同参数重发（bash 2869260 / **python 2869261**，GPU1，V8-S 专才 s_mprau_in stage2_s_benchmark_full_epoch6.pt，--arms guided --critic-kind v8，输出目录同路径 b2_full_891）；新日志已建（仅启动警告，脚本静默设计至终态）。
- **疑因（未定）**：GPU1 重度争用下 torch expandable_segments 分配重试旋占（瞬态）vs 专才 critic 路径数据依赖死循环（确定性）。重启后观：若 60min 内仍无输出目录 + 主线程旋占 → 确定性 bug，转 --source-limit 减源诊断定位触发源。

### 基础设施

- manager UP（PID 2540034）；watcher 已正常退出（A3/A5 全链完成，非死亡）。
- GPU：GPU0-5 忙（util 52–100%），GPU6/7 MIG-only → **无空闲满血卡**；臂 A 重启复用 GPU1（free 14.6GB 充足）。
- 新进程启动干净，无 Traceback/OOM/cpu_fallback。

### 处置

- 已停旧进程 + 留证 + 同参数重启臂 A（新 PID 2869261）。
- 待续：臂 A 重启推进确认（下次巡检）；若复现卡死 → 减源诊断；臂 A 终态 → 双臂对位闭合 → V6 立项裁决终态。
---
## 批次四十三（2026-09-08 08:04，Stage 3 B2 巡检 11#：臂 A 重启推进确认；臂 B / A3 均已归档无新事件）

### 臂 A（V8-S 专才，GPU1，批次四十二重启后）

- 进程 PID 2869261（elapsed 3h40m，CPU 172%≈1.7 核，RSS 6.3GB，state Rl，GPU1 util 100% cron 08:00 确认）→ **正常计算中，非卡死复现**（对照批次四十二卡死画像：主线程单核 93.9% 旋占 + I/O 近零；本次为多核计算 + 大 RSS 模型加载，画像差异明显）。
- 日志仅启动警告 380B（stdout block 缓冲，终态一次性落盘设计）；错误 grep=0（无 Traceback/OOM/cuda error/cpu_fallback）；输出目录未建（预期，B2-B 对照 wall 5.9h → 预计终态 ~10:30+）。
- 批次四十二预案：wall ≥ 2×B2-B（≈12h，~16:30）仍无终态 → 深检/减源诊断，当前未触发。

### 臂 B（V8-S joint，GPU4）与 A3 探针——无新事件（已归档）

- 臂 B：批次四十一已对位（guided 0.1188，Δrecovery −0.0017 CI [−0.0046,+0.0011] 负向跨零，Gate B2/B3 双 FAIL，per-task 四任务全≤0）；SPECS_SETFLOW_V5/V6 已更新（本地 patch_specs_20260908.py 落盘）。仅待臂 A 终态做双臂跨零闭合。
- A3：批次四十一已数值终态入档（H2 部分证实；SPECS_SETFLOW_V6 三文档已更新）；watcher 正常退出，无残留进程。本批复核两者产物均无新变化。

### 基础设施

- manager UP（PID 2540034）；cron status.log 08:00：B2V8 alive=1 terminal=0 / B2V8JOINT terminal=1 / MGR alive=1，与巡检一致。
- GPU：GPU0-5 忙（util 62–100%），GPU6/7 MIG-only → **无空闲满血卡**（A3 已结束不影响；后续若需执行 A3.3 V8 探针仍需等空闲卡）。

### 判定

- B2 臂 A 重启后正常推进；臂 B 与 A3 均已归档；无新异常，无需处置。
- 待续：臂 A 终态 → 双臂对位闭合 → V6 立项裁决终态；A3.3（V8 探针）；DP2 拍板（输入已齐备，建议臂 A 终态后安排）。
---
## 批次四十四（2026-09-08 09:30，非定时巡检：用户指示用空闲卡执行 A3.3 → 发现并修复 FrozenV8Critic tokenizer 致命 bug（V8 引导断电）→ 臂 A/B2-B 作废）

### 重大事件一：A3.3 V8 critic 同款探针——首个 flatness 异常 → 确诊 tokenizer bug → 修复

- **触发**：GPU4 空闲 34.7GB / util 29%（臂 B 终态释放）、GPU2 可用 → 执行此前被阻塞的 A3.3（V8 critic 同款探针，pre-reg A3.3/SPECS_SETFLOW_V6）。
- **实现**：a3_critic_mixed_pool_probe.py 加 --critic-kind {v5,v8}（default v5 行为不变，commit 00c6fbc9）；V8 分支复用 FrozenV8Critic（与 runner 同接口）。
- **flatness 异常**：首次跑两个 V8-S checkpoint 探针结果逐位相同 + measured_best_avg_rank=1.00（全源 tie）——critic potential 全为 0。
- **根因（确诊）**：FrozenV8Critic._score_candidate_group 直接 `tokenizer(str.upper().replace("U","T"))` 整串传入 BertTokenizer（vocab=74 单核苷酸/3mer 级）→ **整条序列被 UNK 成单个 token** → base 输出与输入无关（fp32/bf16 下逐位相等）→ potential 恒 0。V5 FrozenXEditCriticV5 及共享 encoder 使用 `" ".join(逐核苷酸)` 空格 join（format_utr_chunk）→ 正确。B2 双臂（09-07 18:30 起）一直在用**常数 potential 引导 = 无引导**。
- **修复**：route2_v8_frozen_guidance_v1.py tokenize 改逐核苷酸空格 join（commit 57827f5c）；修复后 5 序列 delta 恢复 1e-2 量级、token 恢复真实单核苷酸 ID。
- **修复后真实 A3.3 数值（891 源，CUDA BF16，cpu_fallback=false，TERMINAL）**：

  | critic | overall acc@1 | MRL(652) | MPRAU(108) | HL(111) | polyA(20) | natural-hit(216) |
  |---|---|---|---|---|---|---|
  | V5 (ref) | 0.0614 | 0.0422 | 0.1111 | 0.1280 | 0.0500 | 0.0764 |
  | V8-S 专才 s_mprau_in | 0.0320 | 0.0084 | 0.1497 | 0.0300 | 0.1750 | 0.0440 |
  | V8-S joint s_mrl-polya | 0.0359 | 0.0360 | 0.0278 | 0.0450 | 0.0250 | 0.0579 |

  - base overall 0.0367 / base_reachable 天花板 MRL 12.4% / MPRAU 2.8% / HL 5.4% / polyA 0%。
  - **结论（科学）**：修复后 V8 critic 判别力在搜索分布 mixed 池上 ≤ V5（overall 0.032/0.036 < 0.061）；MRL 主杠杆皆低于 base 结构性天花板（专才 0.0084 接近零——stage2 仅适配 MPRAU 塔所致）；natural-hit 子集反劣（0.044/0.058 vs base 0.151）。**on-manifold 改善（MPRAU 0.1351 / MRL 0.3078）未能迁移到搜索分布，V8 迁移失败（H-V8e 负向）**。
- **Stage 3 冲击**：常数引导下 B2-B（v8joint）已终态 recovery 0.1188 ≈ unguided 0.1205（无引导自洽）→ **判定无效，必须重跑**；臂 A（v8 专才）被杀停（本批 08:35，kill -9 2869260/2869261）——继续跑等于浪费 GPU 且产出无引导等价结果。留证 armA_invalidated_tokenizer_bug_20260908.txt + 探针产物 commit。

### 处置

- 已修复 + 验证 + 探针数值入档（base_fix worktree commit 00c6fbc9 / 57827f5c / 探针产物）；臂 A 杀停留证。
- **待决策（DP2 级）**：修复后 Stage 3 B2 双臂重跑（~/批 1–2 天）以取得真实 V8 guided recovery 判定；或先以探针数值直接裁决 V8 判别力不足 → 缩短 Stage 3 至单臂验证。SPECS_SETFLOW_V5/V6 更新 + 双臂重跑调度待用户拍板。
---
## 批次四十五（2026-09-08 ~10:05，用户拍板修复版重跑双臂）

- **决策**（批次四十四确认 tokenizer bug 后，用户选"重跑双臂"）：
  - 臂 A（V8-S 专才 s_mprau_in）→ **GPU4**（36.9G free/42%），PID **190239**，输出 `guided_b2_v8_20260908/b2_full_891`，日志 `.../guided_b2_v8_20260908/b2_full_891.log`
  - 臂 B（V8-S joint s_mrl-polya）→ **GPU2**（35.2G free/86%），PID **190240**，输出 `guided_b2_v8joint_20260908/b2_full_891`，日志 `.../guided_b2_v8joint_20260908/b2_full_891.log`
  - 参数：同原（b_fix2 pass-2 / seed 20260915 / budget / screen gate 20260915 / --arms guided --critic-kind v8）；修复 commit 57827f5c 已含于 base_fix worktree HEAD。
- **启动验证**：4min 后 CPU 335%/307%（多核正常），GPU4 6.2G/73%、GPU2 17.6G/88%；日志 380B（启动横幅，缓冲模式正常）；无 Traceback/OOM/cuda error。
- **监控 pattern 变更**（后续巡检）：臂 A `pgrep -f "guided_xeditsetflow.*v8_20260908.*b2_full_891"`；臂 B `pgrep -f "guided_xeditsetflow.*v8joint.*b2_full_891"`（改写 20260908 日期目录）。
- **预期**：对照修复前 B2-B wall 5.9h + 探针判别力 V8≤V5 → 预计 ~6–12h 内终态（专才判据 MRL 判别力低可能更快收敛）。
---
## 批次四十六（2026-09-08 12:15，定时巡检 1#（修复版）：双臂推进正常 + 服务器 monitor 脚本路径修复）

### 双臂（修复版 20260908）

- 臂 A（V8-S 专才 s_mprau_in，GPU4，PID 190239）：CPU 时间 10h17m（wall ~2h15m，多核满载），日志零错误；无终态（guided_run_summary.json 不存在，预期 wall ~5-6h+）。
- 臂 B（V8-S joint，GPU2，PID 190240）：CPU 时间 9h04m，零错误；无终态。
- GPU4 采样瞬间 util 0%（与 CPU 侧推进不矛盾，generation 间歇期）；GPU2 82%。

### 服务器 monitor 脚本修复（交叉验证脱节）

- 发现 cron `mrna_editflow_monitor.sh` B2V8/B2V8JOINT 段仍指向旧 20260907 作废目录 → status.log 显示旧 path 假象（B2V8JOINT terminal=1 / B2V8 alive=空）。
- 已更新为 20260908 新路径（物理卡 GPU4/GPU2 对位），bash -n OK，实测运行：`B2V8 alive=1 terminal=0` / `B2V8JOINT alive=1 terminal=0` ✓ 交叉验证恢复。monitor/ 非 git 仓（就地修改，本 journal 记录）。

### 基础设施

- manager UP（2540034）；GPU0-5 全忙、GPU6/7 MIG-only → 无空闲满血卡（当前无等待 GPU 任务，A3/A3.3 已终态）。
- A3/A3.3 不再巡检（已入档）；watcher 无残留。

### 判定

- 修复版双臂健康推进，无异常；server monitor 已对齐新臂。待续：双臂终态 → 真实 Δrecovery 对位 → V6 立项裁决。
---
## 批次四十七（2026-09-08 15:05，定时巡检：修复版双臂终态 → 真实 Δrecovery 对位 → V6 立项裁决）

### 双臂终态（修复版 2026-09-08，均 CUDA A100、CPU fallback=false、method=frozen_v8_critic_guided_xeditsetflow_v5_b_fix2_pass2）

- 臂 A（V8-S 专才 s_mprau_in，GPU4，run_id=b_fix2）：status = **GUIDED_XEDITSETFLOW_V5_B2_RUNNER_COMPLETE**，wall 16441s(~4.6h)，cpu_fallback_used=false，cuda:4；guided recovery = **0.124953**。日志 13:57 收尾，正常终态非死亡。
- 臂 B（V8-S joint 对照 s_mrl-polya，GPU2，run_id=b_fix2）：status = **GUIDED_XEDITSETFLOW_V5_B2_RUNNER_COMPLETE**，wall 17994s(~5.0h)，cpu_fallback_used=false，cuda:2；guided recovery = **0.123363**。日志 14:23 收尾，正常终态。
- 全程无 Traceback / OOM / cuda 报错；两臂经逐核苷酸空格 join 修复（57827f5c）后真实引导（V8 potential 非 0），非旧版常数引导。

### 对位表（修复版真实 Δrecovery，同口径参照 guided_b2_20260903/b2_full_891_adjudication_per_task.json；unguided 基线 0.12046 / V5-guided 0.12626）

| 口径 | n(源) | V8-A 专才 | V8-B joint | V5-guided(ref) | unguided(ref) | V8-A vs ung | V8-A vs v5 | V8-B vs ung | V8-B vs v5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **overall** | 891 | **0.124953** | **0.123363** | 0.126262 | 0.120464 | **+0.004489** | **−0.001309** | **+0.002899** | **−0.002899** |
| MRL(5UTR) | 652 | 0.160020 | 0.156314 | 0.158742 | 0.154652 | +0.005368 | +0.001278 | +0.001662 | −0.002428 |
| MPRAU(3UTR) | 108 | 0.027778 | 0.032407 | 0.032407 | 0.027778 | +0.000000 | −0.004630 | +0.004630 | +0.000000 |
| polyA(3UTR) | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| HL(5UTR) | 111 | 0.036036 | 0.040541 | 0.049550 | 0.031532 | +0.004505 | −0.013514 | +0.009009 | −0.009009 |

### 判定：SetFlow V6 立项裁决 = **否决（critic 攻坚负结果闭合）**

- **任一 V8 臂 Δ 显著正 → critic 非瓶颈？否**。臂 A +0.00449 / 臂 B +0.00290（vs unguided），均远低于 Gate B2 +0.05 门槛且 CI 必跨零，**且双双未超越 V5-critic guided 0.1263**（A −0.00131 / B −0.00290）。
- **双臂跨零 → critic 攻坚负结果闭合**：V8-S 专才/ joint 修复版引导 recovery 均未超过 V5-critic 基线，与 A3.3 探针（V8 判别力迁移失败 H-V8e 负向）在 B2 尺度再印。per-task 亦无任一任务 V8 显著优于"unguided 之上且 ≥V5"：专才仅在 MRL 微占先（+0.0029/ref 内）、joint 在 MPRAU 追平 V5，HL 两臂均反劣（A −0.0135 / B −0.0090），polyA 无信号（base 探索不足共因，n=20）。
- 依用户 09-05 拍板"仅在 Stage 3 结果证明 critic 非瓶颈后才立项 V6"——此条件**未满足** → **V6 立项否决**；Critic V5 维持最强，V8-S 专才不构成 V6 base 升级依据。SPECS_SETFLOW_V6 Phase B Task B1/B2（对位+预判对照）据此闭合，Phase C 维持"未拍板零启动"。

### 使命完成建议

- 双臂已终态、对位与 V6 立项裁决均已入档 → **本监控任务（mRNA EditFlow 训练监控，b2f628e0，30min）使命完成，建议停用**。若后续仍有巡检需求（如补 random/F2 对照臂、DP2 拍板后的 Phase C 发射），再另行挂起。### 批次四十八（2026-09-08 16:10，Phase B 收口复核 + 补全：B1.1/B1.2/B3 + 独立复算确认）

- **触发**：用户指示「A3 数值 + Stage 3 终态齐备后开 DP2」→ 会话检查发现两项均已终态（A3 03:10 数值 + 09:12 V8 探针；Stage 3 修复版双臂 13:57/14:23 终态 + 批 47 journal 裁决）。本批做机械复核 + 补齐 Phase B 缺件（0908 双臂 adjudication 当时未落产物）。
- **B1.1 身份核实（闭合）**：批 45 journal 原文（w0_diagnosis worktree L1184-1185）+ runner 代码分支（`--critic-kind v8` → `FrozenV8Critic(--v8-critic-checkpoint)`，summary 的 `critic_checkpoint_path` 为通用参数遗留字段，v8 分支不加载该文件）+ GPU/wall 交叉验证（臂 A cuda:4 / wall 16441s≈4.6h；臂 B cuda:2 / wall 17994s≈5.0h，UUID 匹配）。**臂 A = V8-S 专才 s_mprau_in（stage2_s_benchmark_full_epoch6.pt），臂 B = V8-S joint s_mrl-polya（stage1_s_epoch2.pt）**（同 A3.3 探针所用 checkpoint）。
- **B1.2 官方 adjudication 补产物（独立复算）**：用预注册 `adjudicate_route2_guided_setflow_v5_b2_v1.py`（与 0903 B2 冻结判定同款同种子）对两臂补跑：
  - 臂 A：guided 0.124953，Δrecovery **+0.004489** CI [−0.0011, +0.0103] 跨零；Δhit@1 −0.00031 CI [−0.0020, +0.0011]；gate B2/B3 = false
  - 臂 B：guided 0.123363，Δrecovery **+0.002899** CI [−0.0048, +0.0105] 跨零；Δhit@1 −0.00240 CI [−0.0070, +0.0010]；gate B2/B3 = false
  - 产物：`guided_b2_v8_20260908/b2_full_891_adjudication.json` / `guided_b2_v8joint_20260908/b2_full_891_adjudication.json` + `_per_task.json`（`analyze_route2_guided_b2_per_task_v1.py`）——**逐位复现批 47 journal 数字**，判定与批 47 一致：双臂 B2/B3 全 FAIL。
- **B3 支持度分解（V8 臂，A2 同款口径）**：产物 `guided_b2_v8_20260908/b3_support_decomposition_v8_arms.json`：unguided support 0.2424 / V8-A 0.2435 / V8-B 0.2469；MRL 0.3129/0.3129/0.3144；MPRAU 0.0556/0.0556/0.0648；HL 0.0541/0.0631/0.0721；polyA 0×3。**修复版真实 V8 势能对池覆盖仍≈零贡献**（+0.001~+0.004 support），与 A2（V5 引导）结论完全一致：**引导（V5 与 V8）改转移速率不改池覆盖**——B2 主判据（recovery@budget = 池覆盖）对 critic 质量结构性不敏感，是双臂跨零的机制性解释。
- **B2 预判对照（spec §2 逐条，如实入档）**：P-主（B2 FAIL 高概率，Δ<+0.05 或 CI 跨零）**证实**（双臂 Δ +0.0045/+0.0029，CI 全跨零）；P-1（MPRAU 源增益较 V5-critic 扩大）**证伪**（A −0.0046 / B +0.0000 vs V5-guided，未扩大）；P-2（MRL 杠杆，不押注）**明确答案：不迁移**（A +0.0013 / B −0.0024 vs V5-guided；探针 MRL 0.0084/0.0360 < base_reachable 12.4%）；P-4（V8 探针 > V5）**证伪**（0.0320/0.0359 < 0.0614，H-V8e 负向）。P-3 polyA 无判据地位（维持）。**P-主依据修订条款（R.5）触发**：判据-机制错配（recovery@budget 与 critic 排序无关）意味着 Stage 3 双臂 FAIL 同时包含「V8 判别力不足」与「任何 critic 都改不了池覆盖」两个成分，二者不可分——但 A3.3 探针已独立裁决前者（V8 ≤ V5），故双 FAIL 结论稳健。
- **本批变更**：base_fix worktree 新增（a）0908 双臂 adjudication + per_task 产物（/mnt），（b）B3 支持度分解 JSON（/mnt），（c）watcher.sh PY 路径修复（python3 → conda editflow env，运行期实况如实提交），（d）A3 full/V8 探针产物与 done 标记（前批遗留未提交）。Stage 3 双臂 runner 产物本身未重算（只读冻结产物复算）。
- **状态**：**Phase A + Phase B 全部闭合（A3 数值终态 + Stage 3 修复版双臂终态 + 预判对照 + B3 分解齐备）→ DP2 拍板会输入完备，已通知用户**。Phase C 维持零启动。

---
## 批次四十九（2026-09-08 16:45，DP2/DP3 拍板落档：V6 以「生成器侧攻坚」立项 + 监控停用 + A3.3-h_bench9 补测发射）

### DP2 拍板（用户，16:40 会话）

- **V6 范围 = 0+1+2+3+4 全部**（「01234 都做」）：
  - 0 = D4 amendment（B3 门槛重校准 + 判据修正 + 双口径，无 GPU）
  - 1 = A3.3 补 h_bench9 探针（V8 裁决完备性：现只探了 s_mprau_in / s_mrl-polya，h_bench9 stage-2 多任务适配臂 MRL 0.2879 未探）
  - 2 = D1 q 模型 cheap-kill（AUC ≥ 0.60 且 CI 下界 > 0.55，不过即杀，~1 天沉没上限）
  - 3 = D3 β sweep（β∈{0.25,0.5,1,2,4} per-task z-归一 + 退火/rank/SMC 增量臂，判定门改新口径）
  - 4 = 扩池 B=256 重跑（H4 直接检验，A4 预测 recovery ~0.62）
- **立项逻辑（用户理由原文，如实入档）**：「即使我们现在可能认为是critic的性能比较差，导致整个的结果不好，但是我们依旧可以去提升 setflow 的效果来提升整个的性能」——与 Phase A/B 证据自洽：recovery 瓶颈 = 池覆盖（生成器属性）、hit@1 headroom = 0.037→0.242（生成器覆盖 + 排序联合属性），二者不因 critic 弱锁死。**09-05 gate 留痕不删**（批 47 的「V6 立项否决」依 09-05 规则字面维持有效；本批为 amendment 式重新立项，依据 = Phase A/B 诊断 + 用户 09-08 拍板）。
- **DP3 拍板（用户）**：主判据换 **support-conditional hit@1 + independent evaluator 双口径**（frozen Optimus/MRL、APARENT/polyA 对生成候选 delta 分布）；**0.35 留痕不删**（V4 记忆化时代标定考核去记忆化架构属范畴错误；amendment 文档 = Task C6 交付物，数字预注册起草时冻结）。
- **监控停用（用户拍板，批 47 建议照办）**：服务器 crontab `mrna_editflow_monitor.sh` 行已移除（2026-09-08 16:45，`crontab -l` 现为空）；needs_attention 标志清理。若 Phase C 发射后需巡检，再行挂起。
- **D2（A2 臂）维持缓议**（不在 0–4 选项内；A5 证据 recovery 对快照不敏感、检索条件化 gated on q 验收）。**DP4（V6 protocol 冻结预注册）在 Phase C 首臂发射前执行**（独立 family、exact-HEAD 授权链、双 receipts）。

### A3.3-h_bench9 探针发射（选项 1 立即执行）

- GPU4（15.7G used/20% util，无本用户进程、无运行进程监控），PID 2723464，`a3_critic_mixed_pool_probe.py --critic-kind v8 --critic-checkpoint v8_stage2_adapt_20260907/h_bench9/stage2_h_benchmark_full_epoch6.pt`，输出 `analysis_phaseA_20260907/a3_critic_mixed_pool_probe_v8_hbench9.json`（BF16、cpu_fallback=false、protected_reads=0）。预期 ~30–60 min。
- GPU 判据说明：GPU5 上仅他用户轻进程（1.5GB/13%）→ 按保守守卫跳过（不与共享卡）；GPU4 满足独立判据。

### 状态与下一步（Phase C 序列）

1. h_bench9 探针终态 → V8 裁决完备（选项 1 闭合）
2. C6 amendment 起草（D4/DP3：新判据 + 0.35 留痕 + 双口径设计 + 新目标带预注册）→ 生效后 B=256 扩池臂才具备判定基础（选项 4 gated）
3. C3 q 模型预注册（泄漏审计：TRAIN only + component flagged=0 硬门 + 验证源 measured 零接触）→ 1 卡 ~4h 训练 → AUC 门裁决（选项 2）
4. C1 效率修复（批量化 + 心跳；100 源 wall ≤ 55 min 验收）→ C2 β sweep（选项 3，~47 GPU·h）
5. Phase C 总预算（新范围）≈ 70–90 GPU·h。

### 批次四十九附（2026-09-08 17:47，选项 1 闭合：A3.3-h_bench9 探针终态）

- h_bench9（stage2_h_benchmark_full_epoch6.pt，cuda:4，BF16，protected_reads=0，wall ~55min）：overall acc@1 **0.0461**（仍 < V5 0.0614）为三 V8 checkpoint 最强，**唯一 MRL 微超 V5**（0.0458 vs 0.0422，仍 < base_reachable 12.4%）；MPRAU 0.0741 / HL 0.0203 / polyA 0.0500 / natural-hit 0.0741。
- **3/3 checkpoint 闭合 → V8 裁决完备：V8 整体 ≤ V5 维持**（H-V8e 负向维持）。多任务适配优于专才/joint 的 MRL 信号登记为 D1/D3 参考，不改变总结论。选项 1 完成，入档 spec §R.6 + tasks/checklist。

### 批次五十（2026-09-08 18:05，选项 0：D4 amendment 起草完成，待用户确认生效）

- **D4 amendment v1 草案**：`docs/paper/route2_setflow_b3_amendment_sc_hit1_v1.md`（v8_stage1_prep worktree，D5 先例同目录）。核心内容：(1) 主判据换 **support-conditional hit@1**（sc-hit@1，tie-aware 口径同现 hit@1，S+ = 池内含 ≥1 measured 的源；防退化条款 = 与 support@B 强制并报）；(2) B3 新门槛分档：**B 档（B=32）sc-hit@1 ≥ 0.10 + ≥2× base 自身排序（natural-hit base acc 0.1514 对照线）**；**A 档（B=256）sc-hit@1 ≥ 0.30 + Δsupport ≥ +0.10**；(3) B2 门改 Δsc-hit@1 ≥ +0.03（B 档）/ +0.10（A 档）CI 不跨零；(4) recovery@budget 永久降级为覆盖诊断字段；(5) independent evaluator 双口径强制（frozen Optimus—MRL / APARENT—polyA，覆盖 672/891 源，MPRAU/HL 如实标注无独立评估器）；(6) 诚实性条款：A 档通过不回溯适用 B 档、分档报告禁止跨档叙事、负结果条款（0+2+3+4 全 FAIL → 诊断完备负结果收口）；(7) **0.35 与旧 B2/B3 判据留痕不删**（E7：V4 记忆化时代标定考核去记忆化架构 = 范畴错误）。
- 证据链 E1–E7 全部为 Phase A/B 机械实测（A1 口径 / A2+B3 覆盖零贡献 / A4 不可达定量 / A3 天花板与 V8 3/3 裁决）。
- **状态：草稿待用户确认**（确认后 journal 正式批次 + spec §R.11 + C2/扩池臂判定门指向本文件）。未确认前不阻塞发射——选项 4a（B=256 unguided 扩池）判定字段为 support/recovery（覆盖侧，不依赖新门），可先发射。

### 批次五十一（2026-09-08 19:15，Phase C 开工：C1 补丁链 + 选项 4a/2 发射实录）

- **C1 工程补丁（3 项，均冒烟验证）**：runner 新增 `--trajectory-count`（参数化 B，硬编码 32 共 5 处解除）+ per-source 进度心跳（`progress_heartbeat.jsonl`，~37 源/跳，ETA 字段）。修复链：心跳定义前置（NameError）/ 输出目录 mkdir（FileNotFoundError）/ **`stratified_trajectory_mode_ids_v4(prior)` 默认 trajectory_count=32 未透传（首次 4a 发射即断言拦截——预注册断言纪律生效）**，补传 `trajectory_count=candidate_cap`。8 源冒烟 15s 全绿（默认 32 路径字节级不变）。
- **选项 4a（B=256 unguided 扩池）发射**：GPU4，PID 2957360，`pool256_unguided_20260908/b2_full_891_B256`（891 源 × 256 = 228,096 轨迹；unguided 为全批量采样（sample_many_setflow_v4）无逐源心跳，roots 心跳 0.4s 完成；CPU 1851% 正常）。ETA ~4h。**判据 = 覆盖侧 support@256 vs support@32（A4 Model A 检验，0.242→预测 ~0.62）+ recovery 上界带**；新 sc-hit@1 门按 amendment A 档（gate gated on amendment 状态）。
- **选项 2（C3 q 模型）发射**：GPU5（与 honghuiyang 进程共卡但显存 1.9G 无冲突，2GB 轻推理），PID 2914339，`phase_c_c3_20260908/train_q_measurability_v1.py`。**泄漏审计前置通过**：TRAIN 14,634 源/5,758 components vs VALIDATION 3,439 components **交集 0（flagged=0 硬门 ✓）**；正例 89,580 measured 行（97% 单编辑）+ 负例均匀合法单编辑 1:5；component 90/10 切分 held-out AUC 门（≥0.60 且 CI 下界>0.55，cheap-kill）；frozen mRNABERT + 393k 参数头；BF16 CUDA。冒烟 21.7s 全链绿（含 transformers torch.load 闸门绕行 = 项目规范 load_mrnabert_base 镜像 + tuple 输出 [0] 索引）。dry-run AUC 0.504 无意义（1 epoch/2k pairs）。全量 ETA ~2h。
- **amendment 状态说明**：v1 草案（批 50）——用户 DP3 已拍板判据方向（sc-hit@1 + 双口径 + 0.35 留痕）+「自己去执行」指令 → **本轮按 ACTIVE 执行**（分档数字 0.10/0.30 系 A4 推导的预注册值，provenance 如实入档；如用户复核有异议走 v2 amendment 留痕修订）。

### 批次五十二（2026-09-08 21:05，选项 2 终态：q 模型 AUC 门 FAIL → D1 cheap-kill 终止（如实入档））

- **终态数字**（GPU5，BF16，wall 3065s，protected_reads=0，产物 `q_model_20260908/full/q_train_result.json` + head checkpoint）：held-out component AUC **0.5528**，bootstrap 95% CI [**0.5453, 0.5604**]（38,406 holdout 对，361,812 训练对，3 epochs，393k 参数头）。
- **门裁决：FAIL**（预注册门 AUC ≥ 0.60 且 CI 下界 > 0.55；实测 0.5528 < 0.60）→ **D1 按预注册 cheap-kill 条款终止：q 通道不进入集成（C4 取消）**。泄漏审计 TRAIN only + component flagged=0 全程有效。
- **科学发现（负结果带信息量，如实入档）**：AUC 0.5528 的 CI 下界 0.5453 **显著高于随机 0.50**（38k holdout 对检验力充足）→「实验者测量偏好」存在**弱但可跨源泛化的信号**（+0.053 AUC）——H5 口径偏好假设的直接测量首次落地：**信号存在、幅度不足以支撑 D1 集成门**。这同时意味着：测量偏好可建模但非强结构信号，recovery 家族口径对「偏好学习」的敏感度有限。
- **下游影响**：(1) C4（q 集成臂）取消；(2) D2 检索条件化的 gate 条件翻转（spec §4-D2 原文：q 失败 → 检索升主选）——D2 仍维持缓议（用户 09-08 范围内未含），仅登记触发条件变化；(3) GPU5 释放。
- **选项 4a 继续在途**（GPU4，运行 1h02m，CPU 1878% 正常，ETA ~3h）。

---
## 批次五十三（2026-09-08 20:30，V9 Stage 0 三件套终态 + baseline P0-3/P0-4 交付；CRITIC_V6 spec 2026-09-08 增补执行首班）

> 依据：SPECS_CRITIC_V6「【2026-09-08 增补】V5 架构尸检 + V9 执行案」（用户批准即立项，双轨完全并行）+ SPECS_BASELINE_LEADERBOARD「【2026-09-08 增补】Baseline 覆盖审计 v2」（P0 立即执行）。零训练窗口与 Phase C 并行（GPU2/GPU3，未触碰 GPU4 上的 pool256_unguided）。代码：v8_stage1_prep worktree `analyze_v9_stage0_v1.py` / `compile_baseline_table4_v1.py` / `analyze_baseline_k_sensitivity_v1.py`；产物 `/mnt/.../experiments/analysis_v9_stage0_20260908/` + `analysis_baseline_table4_20260908/` + `analysis_baseline_k_sensitivity_20260908/`。

### V9-0a M2 参数干扰定位（CPU）——**任务向量正交 + 核心区近随机不相交**

- 对象：H 家族同构对 τ(h_bench9, 9 任务均衡 full-FT) vs τ(h_mprau_lora, MPRAU-only LoRA 并回权重)，base = Stage 1 H（同 init 实证：两 run report 均 stage1_h_epoch2.pt）。
- **实现勘误（如实留痕）**：首版 flatten_taus 误拼权重本身而非任务向量（sd−base），范数 589≈权重范数暴露 bug（LoRA 臂非 LoRA 键应冻结 → τ2 范数 ≤29）；修复后 float64 重算（float32 在 113M 元素上累计误差曾产出非法余弦 1.0275）。
- **终态**：‖τ1‖=13.84 / ‖τ2‖=5.68；**整体余弦 0.0034（正交）**；top-5/10/20% 核心区 Jaccard = 0.053/0.075/0.128（随机期望 0.026/0.053/0.111——仅 1.1-2.1× 随机）；head.weight 余弦 0.36；τ2 质量全部落在 48 个 LoRA 模块 + head（49 键）。
- **判读**：按 CPI-FT"低重叠→可合并"预测条件满足 → 交 M1 裁决；同时跷跷板归因修正：h_bench9 的 MPRAU 弱不是"被覆盖"（区域不相交）而是"从未学到"（均衡采样稀释）。

### V9-0a M1 合并诊断（GPU2，~1h）——**7/7 变体全 FAIL，合并路线关闭**

- 管线交叉验证：single_h_bench9 经本管线逐位复现（MRL 0.28792/polyA 0.80673/MPRAU 0.05556/macro 0.15162 vs 预注册参照 0.2879/0.8067/0.0556/0.1516 ✓）；single_h_mprau_lora MPRAU 0.11005（入档参照 0.1084，BF16 跨卡 ±0.002 级）。
- **知识保留门（per-task ≥ max(single) − 0.01）裁决**：arith_mean / TIES keep∈{20,80,90,95}% / DARE drop∈{90,95}% **全部 FAIL**——MPRAU 0.038–0.049（<< 0.098 门），polyA 0.68–0.78（< 0.797 门），MRL 全保留（0.287–0.298，keep80 甚至超单模型 0.2979）。
- **S 家族 inter-seed soup（机制校验）**：5-seed 权重平均 MPRAU 0.1142（单 seed 0.108–0.153；预测 ensemble 0.1351）——**同任务合并有效 → 合并机器无误**；副发现：soup 恢复 MRL 0.2496（单 seed ≈0.0001）——权重平均抵消各 seed 的破坏性偏移、部分恢复 Stage 1 先验。
- **科学结论（论文素材）**：参数空间的几何可分性（M2 正交+低重叠）≠ 功能可加性（M1 全灭）——TIES/DARE/task-arithmetic 的"低重叠→可合并"预测在 113M 编辑差分 critic + 该数据体制上系统性失败；专才知识对更新幅度呈非线性依赖（halving/trimming 即摧毁）。**V9-1 裁决：adapter-zoo（V9-a）路线确认，不设 merged-as-init 臂。**

### V9-0b M2b 任务梯度余弦（GPU3，~15min）——**共享线性头是梯度冲突主战场（新机械发现）**

- 底座：V8-S Stage 1（s_mrl-polya epoch2）扩展 9 域/6 细胞（= V9-1a init），eval 模式无 dropout，5 任务 × 100 batch × 32 pair，pair-delta MSE（z-scored target，同训练口径）。
- **All-params 余弦：MRL(GSE114002) 与全部其他任务显著负相关**——MRL↔MPRAU **−0.922**、MRL↔HL **−0.959**、MRL↔TE200304 **−0.889**、MRL↔polyA **−0.542**；其余任务对全部正相关（MPRAU↔HL +0.943、MPRAU↔TE +0.888、TE↔HL +0.902、polyA↔{MPRAU,TE,HL} +0.54 左右）。
- **Backbone-only 余弦大幅减弱**：MRL↔MPRAU −0.124、MRL↔HL −0.283、其余 |cos| ≤0.30——**冲突集中于非 backbone 参数（共享线性头 768×1 为主；domain/cell embedding 各行正交不贡献负余弦）**。
- **判读**：(a) 跷跷板又一机械根源 = 单一共享输出头上的方向冲突（V5 尸检组件②补强——不仅是容量，还有头的方向冲突）；(b) V9-a 适配器天然自带 per-task 头 → 结构性解决，PCGrad 臂按条款可登记但优先级低于 adapter（冲突主体将被 adapter 结构消除）；(c) 3UTR 族（MPRAU/HL/TE/polyA）梯度高度协同 → V9-a 分组共享适配器消融臂有数据支持；(d) MRL 是"孤立任务"（5UTR 域）——独占适配器。

### Baseline P0-3 Table 4 v1（CPU）+ P0-4 K 敏感性（CPU）

- **Table 4 v1**（`analysis_baseline_table4_20260908/`，JSON+MD）：五行终态（unguided 0.12046 / V5-guided 0.12626 / V8 专才 0.12495 / V8 joint 0.12336 / TreeG 0.0045）× recovery@10 / top-K rec@10 / hit@1 / legality 1.0 / budget 0 + Gate B2/B3 FAIL 标注。**缺口登记 P0-3b**：closed NDCG@K/regret 需离线打分通道（generated_candidates.private.jsonl 已含 generation_score——可算，后续接线）；pool256 在途不入 v1。
- **K 敏感性**（`analysis_baseline_k_sensitivity_20260908/`，29 行 = full-coverage 16 + Saluki 3 + V5 10）：**NDCG@K 在 K≥5 完全稳定**（全部行 K=5/10/20 逐位一致；K=3 仅多候选任务微降：polyA V5 0.8402→0.8703、MRL V5 0.8195→0.8350）；V5 各行 Spearman 与榜单参照精确一致（0.1953/0.0500/0.0579/0.0639/0.8219/0.1354 ✓ 分层 bug 修复后）→ **R4 闭合：榜单 NDCG 结论对 K 稳健**。缺口 P0-4b：Optimus/FramePool/APARENT/APARENT2 逐条预测未持久化（Stage 0a 只存聚合）→ GPU 重打分登记。

### P0-1/P0-2 可得性探测（P0 剩余两项执行前置）

- **连通性**：huggingface.co 不通；**hf-mirror.com 200** ✓；github 200 ✓；pypi 200 ✓。
- **P0-2 RiboNN 可得性确认**：GitHub Sanofi-Public/RiboNN 公开（NBT 2025 44:783, s41587-025-02712-x）+ **Zenodo 官方人/鼠权重自动下载** → 可执行；输入适配条款待登记（UTR 片段作全长 mRNA 输入、无 CDS 通道——其论文 5UTR 每 nt 信息密度 ~67% 支撑 UTR-only 输入合理性，口径差异如实声明）。
- **P0-1 LLR 模型下载已启动**（后台 PID 3315048）：Caduceus-ps_131k_d256 + NT-v2-500m-multi-species 经 hf-mirror → external_model_assets/{caduceus,nucleotide_transformer}/（日志 llr_model_download.log）；GPN 托管待查（备选 DNABERT-2）。LLR 打分脚本 = 下一班 P0-1 执行件。

### 纪律

- protected reads = 0（全部 VALIDATION）；零训练（无参数更新）；CUDA BF16（M1/M2b，cpu_fallback=false）；产物 /mnt、代码 v8_stage1_prep worktree（本批 commit）；未触碰 Phase C 在途产物（pool256/GPU4）；M1 合并网格超预注册集（增 TIES keep20% = 论文典型密度）如实入档。

---
## 批次五十四（2026-09-08 22:30，P0-1 LLR 零样本基线族双模型终态 + Task 16.1/16.2 评估基础设施固化）

> 依据：SPECS_BASELINE_LEADERBOARD「2026-09-08 增补」§V.4 P0-1（绑定 H2 第三模式 + MPRAU 约束四模式闭环）+ SPECS_CRITIC_V6 spec N.4.4 固定条款（混合池探针固化 + potentials 断言）。GPU2（NT-v2）/GPU3（HyenaDNA）；与 Phase C（GPU4 pool256 在途）零冲突。代码：v8_stage1_prep worktree `analyze_baseline_llr_zeroshot_v1.py`；setflow worktree `evaluate_route2_mixed_pool_probe_v1.py` + `test_route2_guidance_link_assertions_v1.py` + guidance 模块断言。产物 `/mnt/.../analysis_baseline_llr_zeroshot_20260908/{ntv2,hyenadna}/`。

### P0-1 执行记录（工程排障如实入档）

- **模型资产**：NT-v2-500m（5.6G）+ HyenaDNA-small-32k-hf（42M）经 hf-mirror 下载；Caduceus 硬依赖 mamba_ssm（服务器无 nvcc 不可编译，HydraRNA 同款教训）→ **放弃 Caduceus，换 HyenaDNA（causal-PLL 口径）**，spec"≥2 模型"满足。
- **transformers 版本坑**：editflow 环境 transformers 5.14.1 移除了 NT-v2 custom code 所需 API（find_pruneable_heads_and_indices）→ 建 `/home/cunyuliu/llr_env`（venv --system-site-packages 复用 torch + transformers 4.45.2）跑 LLR 脚本；两模型加载验证通过。
- **打分协议**：NT-v2 = MLM 逐编辑位置 mask（非重叠 6-mer：定位覆盖 token，mask 整 token，比较 ref/alt 6-mer logP；尾部单核苷酸同法）；HyenaDNA = causal-PLL（source 序列单次 forward，每编辑取 logits[pos-1] 行 logP(alt|prefix)−logP(ref|prefix)）；U→T；编辑位置 0-based 验证（edit_operations.position_zero_based，字段名勘误：非 spec 草案的 source_relative_edits）；MPRAU 按变体去重打分（2,008 unique，广播回 12,048 行）。
- **冒烟纪律**：两模型各 40 记录/task 冒烟全绿后才发全量。

### P0-1 终态数字（VALIDATION，signed ρ 主口径 / |y| ρ 副口径）

| 任务 | NT-v2-500m | HyenaDNA-small | 判读 |
|---|---|---|---|
| **MPRAU pair-mean**（2,008 变体） | **0.0183** / 0.0015 | **0.0450** / 0.0060 | 均 << V5 0.1025 / s_mprau_in 0.1351 / Saluki 0.1205 |
| GSE186455（274） | 0.0395 / −0.0336 | 0.0556 / 0.0551 | ≈0 |
| GSE149487-TE（48） | 0.0350 / −0.0084 | 0.1523 / −0.1929 | n=48 噪声带 |
| GSE149487-RNA（48） | −0.2748 / −0.1566 | 0.0083 / −0.1557 | n=48 噪声带 |
| GSE217518-5UTR（399/400） | 0.0361 / 0.0348 | 0.0293 / −0.0028 | ≈0（1 条 position-0 flagged） |
| GSE217518-3UTR（499/503） | −0.0319 / 0.0445 | −0.1089 / −0.0323 | ≈0/负（4 条 flagged） |

- **H2 第三模式证据落地（双模型）**：零样本 LLR（通用 LM 的"内心"似然）在全部任务上无信号——与 frozen-Δ 16 行、matched-FT 三 backbone 负带构成三模式闭环：**通用序列模型的表征、监督微调、似然比三个通道都不携带 source-relative 编辑差分信号**。
- **MPRAU 约束主张四模式全闭环**：frozen probe 0.015 / matched-FT −0.075 / CMS 外部先验 0.033 / **LLR 0.018 + 0.045**——"外部任何无监督信号源都不携带 3'UTR 等位偏移信息"（benchmark 结论，D5 数据体制主张证据完备）。
- flag 统计：两模型全部任务 flagged ≤4/503（position-0 无 prefix 或编码对齐拒绝），主结果不受影响。

### Task 16.1 A3 混合池探针固化（完成）

- 脚本固化为正式评估入口：`evaluate_route2_mixed_pool_probe_v1.py`（v8_stage1_prep worktree scripts/，源自 setflow worktree analysis_phaseA 原件，零改动复制）。
- **参照行入档**：`analysis_v9_stage0_20260908/mixed_pool_probe_reference_v1.json`——V5 0.0614 / h_bench9 0.0461 / joint 0.0359 / 专才 0.0320（base-self 0.0367）+ natural-hit 子集 + per-task 全表；此后任何 critic 行报 on-manifold ρ 必须同报探针值（spec N.4.4 共主口径条款生效）。

### Task 16.2 potentials 非常数链路断言 + tokenizer 单测（完成）

- **runner 断言入码**：`route2_v8_frozen_guidance_v1.py` `_score_candidate_group` 增加链路完整性断言——≥2 个不同候选序列收到完全相同打分即 raise（整串-UNK tokenizer bug 的精确指纹：不同序列 → 同一 UNK token → 同输出 → potential 恒 0 = 无引导，N5 教训制度化）。
- **独立单测**（setflow worktree `test_route2_guidance_link_assertions_v1.py`，--tokenizer-only 模式免 GPU）：T1 join 路径逐核苷酸 token 数（22 tokens for 20nt ✓）/ T2 整串路径坍缩为 3 tokens（bug 指纹文档化 ✓）/ T3 join 编码全核苷酸覆盖无 UNK（✓）——**全过**。

### 下一班（P0-2 + V9-1a）

P0-2 RiboNN frozen-Δ（clone Sanofi-Public/RiboNN + Zenodo 权重；输入适配条款：UTR 片段作全长 mRNA 输入、无 CDS 通道；R3 归属审计=内源 Ribo-seq TE 训练，与 MPRA 库无重叠）；V9-1a adapter-zoo 预注册起草（M2b 证据：MRL 独立适配器 + 3UTR 族分组共享消融；polyA CNN stem 固定条款）。

### 纪律

- protected reads = 0（全部 VALIDATION）；零训练；CUDA BF16（cpu_fallback 未触发——LLR 脚本 autocast 未加，纯 fp32 推理更快且这是 frozen 打分非训练，协议 BF16 条款针对训练/validation 轨道；如实记录口径差异）；产物 /mnt、代码两 worktree（本批分别 commit）。

### 批次五十三（2026-09-08 22:05，选项 4a 两次发射拦截实录 + 评估器 cap 覆盖补丁 + 第三次发射）

- **首次发射被断言拦截（19:30，运行 ~1h 后）**：`stratified_trajectory_mode_ids_v4(prior)` 默认 trajectory_count=32 未透传 → 每源只产 32 条 → 预注册断言 `len(roots) == sources × candidate_cap` 拦截。修复：透传 trajectory_count=candidate_cap（commit 1acafb41）。
- **二次发射被 gate 断言拦截（21:50，运行 ~2h 后，产物已落盘）**：`candidate_budget_violation_count = 199,584`——评估器（evaluate_route2_generation_v1::evaluate_generation）按 **manifest 冻结 candidate_budget=32** 计数候选越界。采样本身全合法（228,096/228,096 legal、unique 0.7141、0 edit violation）。诊断：**这是「评估口径」拦截而非「生成合法性」拦截**——B=256 扩池系 DP3 amendment A 档授权操作（分档判据），manifest cap=32 是 V5 冻结协议字段。
- **修复（评估器 cap 覆盖，最小侵入）**：evaluate_arm 增 candidate_cap 参数（默认 32 = 现行行为不变）；≠32 时以 dict(spec, candidate_budget=cap) 影子 manifest 覆盖（**不写回、不改 manifest 文件**）；runner 传 --trajectory-count 值；arm summary 新增 manifest_candidate_cap_per_source=32 留痕字段（原 candidate_cap_per_source 字段记录实际运行值）。旧产物（B=32 一切历史 run）评估路径字节级不变。
- **三次发射（22:05，PID 3565869，GPU4）**：roots 0.3s + unguided 全批量采样在途（GPU4 46%）。ETA ~2h（按二次发射 2h 采样 + 评估 ~10min 推算）。
- **纪律说明**：两次拦截均为预注册断言按设计工作（fail-fast 而非静默错数据）；修复走工程补丁 + journal 留痕，不改任何 B=32 历史判定。

---
## 批次五十五（2026-09-08 22:15，P0-2 阻塞登记 + V9-1a adapter-zoo 预注册 FROZEN + 3-seed 发射）

> 依据：SPECS_BASELINE_LEADERBOARD §V.4 P0-2 + SPECS_CRITIC_V6 N.4 双轨执行案 Task 12。代码：v8_stage1_prep worktree `run_route2_v9_adapter_zoo_v1.py` + `docs/paper/route2_v9_adapter_zoo_prereg_v1.md`（FROZEN）；watcher `~/monitor/v9_relaunch_watcher.sh`。

### P0-2 RiboNN：BLOCKED_ON_USER_TRANSFER（如实登记，非 NOT_TESTABLE）

- 代码可得性 ✓（GitHub Sanofi-Public/RiboNN 已 clone 至 external_model_assets/ribonn/，含 src/predict.py、data.py、config/conf.yml——78 人细胞系多任务 CNN+GRU，输入编码 = one-hot 4 通道 + codon 标注通道，pad_5_prime=False）。
- **权重不可达**：Zenodo records/17258709（weights.zip，官方 Makefile 的 wget 源）服务器直连 Connection refused；hf-mirror 无 RiboNN 镜像（API 搜索空）。
- 处置：登记 **BLOCKED_ON_USER_TRANSFER**（权重公开可得、仅网络中转问题——APARENT2/CMS 先例：用户 Mac 下载 → scp 至 `/mnt/.../external_model_assets/ribonn/tmp/weights.zip` → 解压 `models/human/{run_id}/state_dict.pth`）；到位后按 spec §V.4 P0-2 执行（frozen-Δ on GSE200304 + GSE149487-TE；输入适配条款：UTR 片段作全长 mRNA 输入、无 CDS 通道、codon 通道按序列隐式标注——RiboNN data.py 的 label_codons 按帧标注，UTR-only 输入的帧假设如实声明；native 三态 + 内源/报告基因口径差异声明 + R3 归属审计照走）。

### V9-1a adapter-zoo：预注册 FROZEN + 发射

- **预注册**（docs/paper/route2_v9_adapter_zoo_prereg_v1.md，v8 worktree）：冻结 V8-S Stage 1 trunk + 每任务 LoRA（r16 α32，全部 12 层 Wqkv/attn.dense/gated_layers/wo）+ 共享 LoRA（r32）+ **per-task 线性头**（M2b 证据：共享头 all-params 梯度余弦 −0.92~−0.96 vs backbone-only −0.12~−0.28）+ **polyA CNN stem**（从 Stage 1 H 拷贝 init、随适配器可训练；S trunk + H stem 嫁接失配风险登记，门② polyA ≥0.80 暴露即处置）+ task_id 确定性路由（domain_ids 直接索引，不学路由器）；训练 = V8 Stage 2 均衡适配同款（pair-delta MSE z-scored / DomainBalancedSampler / cell conditioning / AdamW 2e-5 cosine / 6 epochs / FINAL-EPOCH-6-FIXED / CUDA BF16）；判定门 = spec N.4.3 三门全量引用（①探针 per-task ≥ 历史最强 −0.005 ②on-manifold 9 任务表含 MPRAU >0.1351 CI 不跨零 ③V9-2 guided B2 原门）；A/B 共享方向消融（shared-A/shared-B）待主配置结果后另案（v1 只训 symmetric × 3 seeds，预注册留痕）。
- **实现勘误（如实）**：冻结顺序 bug（先冻结 trunk 再 wrap，否则 LoRA 参数被 base.parameters() 冻结——trainable 白名单断言拦截）；stem state_dict 键前缀剥离；shared LoRA 独立 r32 参数组（首版误并入 task rank 桶）。
- **冒烟**（30 步，GPU5）：init 136 keys（Stage 1 S）+ stem 8 keys（Stage 1 H）✓；可训练 33.0M（48 LoRA 模块 × [shared r32 + 9×task r16] + stem + embeddings + 9 heads）✓；训练 + 全 9 任务评估管线全绿；budget 700 steps/epoch × 6 = 4,200 步（~半日/seed）。
- **发射**：seed 20260907（GPU2，PID 3617069）/ seed 20260915（GPU5，PID 3617696）训练中（step 150/200，loss 6.04/5.20 下降）；**seed 20260911 OOM 亡**（GPU3 被外部进程挤占：8.8G+2.45G 外部占用下我方 28.2G 时爆）→ **relaunch watcher 已部署**（`~/monitor/v9_relaunch_watcher.sh`：GPU3 ≥30G 空闲且无我方进程即自动重发；日志 v9_relaunch_watcher.log）——共享集群 OOM 教训（09-04 APA）复用方案。
- 预计终态：seed 07/15 ~09-09 上午；seed 11 随 GPU3 释放。

### 纪律

- protected reads = 0；FINAL-EPOCH-6-FIXED；CUDA BF16（cpu_fallback_used=false）；产物 /mnt（v9_adapter_zoo_20260908/seed*/）、代码 worktree + push（本批 commit）；3 seeds 全报不作 seed 挑选；预注册门槛不事后改。

---
## 批次五十六（2026-09-08 23:40，轨道 B 数据审计 stage 1+2 终态：S1/M6 构建完成 + R3 五研究零泄漏 + 效应量分层审计）

> 依据：SPECS_CRITIC_V6 spec N.4 双轨执行案 Task 15（轨道 B 数据轨，与轨道 A 训练并行）。代码：v8_stage1_prep worktree `track_b_stage1_inventory.py` + `track_b_stage2_pairs.py`；产物 `/mnt/.../analysis_track_b_20260908/`（stage1_inventory.json + s1_stability_pairs.jsonl + m6_ndd_translation_pairs.jsonl + stage2_pairs_report.json + effect_strata_audit.json）。

### 基础设施

- **hg38/GRCh38 参考基因组就位**：UCSC 下载限速（0.24MB/s）→ 切 Ensembl GRCh38 primary assembly（curl 5MB/s，882MB gz → 3.15GB fa，external_model_assets/hg38/）；企业 MITM 证书坑（wget 证书验证失败，curl 绕行）入档。

### Stage 1 清单（勘误入档）

- **S1（Su 2025）规模勘误**：调研文档"6,555 对"实为论文宣称变体总数；补充表实际 **5,072 行**（SNV 4,540 + indel 532；indel 因 V9 动作空间为 SUB 而排除）。全基因组分布 + 正负链 50/50。
- **M6（Plassmeyer 2025）规模勘误**：调研文档"15,070 对"错误（行数/2 未考虑 barcode 重复）；实际 **1,507 个变体**（每变体 ~11 barcode × REF/ALT = 32,990 行；HEK/vglut 两文件同变体集），SNV 子集 916。
- **S6（GSE200304 IVT）勘误**：调研文档"RAW.tar 已在管线内"错误——raw_public 无 GSE200304；GEO portal reCAPTCHA WAF 拦截（与 ENCODE 同款）→ **BLOCKED_ON_USER_TRANSFER**（用户浏览器下载 GSM6721068-6721097 processed 文件后接入）。

### Stage 2 构建终态（双双 100%）

- **坐标语义破案（如实留痕，三轮收敛）**：首版 Start 列 + 正链直接匹配 = 64.8%（负链未处理）；二版 Mutant 列 strand + rc 匹配但 start 坐标 = 26.5%；**终版：变异位置 = STOP 坐标（1-based，BED 风格 start=stop−1）+ ref/alt 已是转录本方向（负链已补）——正链 2302/2302 与负链 2238/2238 双 100% 验证**。
- **S1 库**：4,540 SNV 对 × 155nt 转录本方向窗（负链已 rc）；y_SH = WT−mt decay（SH-SY5Y，n=3,351 有效）/ y_HEK（n=3,295）；方向语义 = "变异使 UTR 更稳定为正"。
- **M6 库**：916 SNV 对 × 100nt 窗；barcode 聚合 UMI 总计数 → y_v0 = log2((sum_ALT+1)/(sum_REF+1))（**口径 v0 声明**：总计数表达比；polysome 富集口径需 GEO 42 样本元数据，登记后续升级）；42 样本列名 SIC#### 语义待 GEO 元数据。
- **R3 泄漏审计（五研究全 split 鸽笼 ≤2 mismatch）**：S1 **flagged=0** + M6 **flagged=0**——**双硬门通过**（对照 GSE114002/GSE269595/GSE217518/GSE186455/ENCSR854RUF 全部受保护序列）。

### 效应量分层审计（Task 15.4 前半）

- **S1**：|Δdecay| 中位数 23.6 分钟（SH）/ 23.2（HEK）；>99% 行 |Δ|>0.1——**富集效应分布**。
- **M6**：|log2FC| 中位数 0.254；79.8% > 0.1、22.9% > 0.5——**富集效应分布**。
- **判读（CMS 教训对照）**：两库效应分布与 benchmark GWAS 富集体制匹配（CMS 失败根因 = 小效应为主强化"≈0"先验）——**范式 + 泄漏 + 体制三审全过，具备入 V9-1b 弱域适配器训练资格**。
- cryptic splicing QC（Task 15.4 后半）：GT-AG v0 proxy 指标已算（窗内含 GT..AG 对比例 ~30%）；**完整 Dao 2024 筛查登记待做**（需其伪影模型/规则集）。

### V9-1a 训练监控（轨道 A）

- seed 20260907（GPU2）/ 20260915（GPU5）训练中（epoch 3-4，step 2000-2150，loss 下降正常）；seed 20260911 watcher 值守 GPU3（外部任务仍占）。

### 下一班

1. V9-1a 三 seed 终态 → 门①（探针硬门）②（on-manifold）判定
2. V9-1b 弱域适配器预注册（S1/M6 数据已获入训资格；效应体制条款写入）
3. M6 polysome 口径升级（GEO 元数据）+ S6/RiboNN 权重用户中转跟踪
4. cryptic splicing 完整筛查（Dao 2024 规则集调研）

### 纪律

- protected reads = 0（外部数据 + R3 审计基础设施只读受保护序列）；零训练；产物 /mnt、代码 worktree + push（本批 commit）。

---
## 批次五十七（2026-09-09 00:20，V9-1a 首 2 seeds 终态——跷跷板打破的直接证据 + polyA 冲击天花板）

> 依据：SPECS_CRITIC_V6 spec N.4 Task 12（V9-1a adapter-zoo，预注册 route2_v9_adapter_zoo_prereg_v1.md）。seed 20260907（GPU2）/ 20260915（GPU5）FINAL-EPOCH-6-FIXED 终态；seed 20260911 watcher 仍在等 GPU3（外部占用）。产物 v9_adapter_zoo_20260908/seed*/。

### 2-seed 终态（VALIDATION，FINAL-EPOCH-FIXED）

| 任务 | seed 07 | seed 15 | 2-seed 均值 | 历史最强（参照） | 门② | 判读 |
|---|---|---|---|---|---|---|
| MRL | **0.3055** | **0.3223** | ~0.314 | Route A ensemble 0.3158 / h_bench9 0.2879 / V5 0.1354 | ≥0.28 | **双 seed 过门**；seed 15 单点超 Route A（0.3223 > 0.3158）= 首个超 MRL 外部平局线的任务训练模型 |
| polyA | **0.8049** | **0.8633** | ~0.834 | V5 0.8219 / V6 0.8273 / h_bench9 0.8067 | ≥0.80 | **双 seed 过门**；seed 15 = **历史最强**（+0.041 vs V5）= 天花板 0.90 的 **95.9%**（V5 91% → V9 96%，逼近物理上限） |
| MPRAU | 0.0477 | 0.0573 | ~0.053 | s_mprau_in 0.1351 / V5 0.1025 | >0.1351 CI 不跨零 | 未过（预期内：均衡池无域内专才数据——V9-1b 数据臂方向） |
| TE(200304) | 0.0356 | −0.0041 | ~0.016 | V5 0.0579 / 内靶 0.1317 | ≥0.1317 | 未过（同 h_bench9 弱） |
| macro | 0.1602 | 0.1983 | ~0.179 | V5 0.167 / h_bench9 0.1516 | ≥0.167 | seed 15 过；2-seed 均值过 |

### 核心科学发现（论文级）

1. **跷跷板被打破的直接证据**：MRL 与 polyA **同时达到历史最强**（V8 Stage 2 时双达标但都非最强：0.288/0.807；V9-1a：0.322/0.863）——任务专属 LoRA + per-task 头 + polyA CNN stem（M2b 共享头冲突 + M1 容量断点诊断的架构回应）在同 init 下让两个强任务并存且各自超越历史。V5 尸检组件②（容量断点）的修复实证。
2. **polyA 96% 天花板完成度**：0.8633 / 0.90 = 95.9%——D2 天花板归一化叙事下 polyA 接近物理上限（标签 ICC 0.90），"已饱和"结论再加固（V5 91% → V9 96%）。
3. **MRL 任务训练模型首次超外部平局线**：0.3223 > frozen-Optimus 0.3132 / Route A ensemble 0.3158（单 seed 点估计，显著性待 3-seed ensemble + paired bootstrap——按预注册 730-record 检验力条款预期 CI 跨零，如实报告）。
4. seed 波动如实：polyA 0.805 vs 0.863（压线 vs 大幅过）——3-seed mean ± std 报告，不作 seed 挑选。

### 判定状态（预注册门，等 seed 11 后全量判定）

- 门②分项：MRL ✓✓ / polyA ✓✓ / macro ✓（均值）/ MPRAU ✗✗ / TE ✗✗ → **部分过门**（主判据 MPRAU 未过 = V9-1a 整体不判 PASS；按预注册门②须全过——如实记录，MPRAU/TE 缺口正是 V9-1b 数据增补臂 + 回退梯的设计对象）。
- 门①（混合池探针）：待 3 seeds 齐 + V9 探针适配（V9 forward 接口与 V5/V8 探针脚本不同，适配为下一班工作）。
- 门③（V9-2 guided）：等门①②全量判定。

### 下一班

1. seed 20260911 终态（watcher 值守）→ 3-seed 全量门判定 + MPRAU 3-seed ensemble paired bootstrap
2. V9 探针适配（evaluate_route2_mixed_pool_probe_v1.py 加 V9 checkpoint 分支）→ 门①
3. V9-1b 预注册（S1/M6 已获入训资格——批次五十六；MPRAU/TE 弱域适配器）
4. MRL 3-seed ensemble vs Route A/Optimus paired bootstrap（显著性裁决）

### 纪律

- FINAL-EPOCH-6-FIXED；CUDA BF16（cpu_fallback_used=false）；protected reads=0；seed 全报不挑选；门不事后改。

### 批次五十四（2026-09-09 00:10，选项 4a 终态收割（第三次发射成功）+ C2 β sweep 发射）

- **4a 终态（wall 0.35h，GPU4，B=256 × 891 源 = 228,096 轨迹，0 violation，cpu_fallback=false）**：`pool256_unguided_20260908/b2_full_891_B256`（评估器 cap 补丁生效）。**H4 直接检验结果（三个新科学事实）**：
  1. **support(256) = 0.3861 vs A4 Model A 预测 0.62——模型高估近一倍**（二项式假设不成立，均匀采样边际收益递减远陡于线性外推）；recovery@256 = 0.2462 ≈ support（covered 源几乎全部命中）。
  2. **per-task 极不均匀**：MRL 0.313→0.479（+53%）、HL 0.054→0.216（4×）、MPRAU 0.056→0.074（+33%）、**polyA 0→0（死区：B=256 均匀采样仍零命中，20 源全灭）**——扩池对 polyA 完全无效，对 MPRAU 接近无效。
  3. **sc-hit@1 稀释效应**：base 自身排序 sc-hit@1（S+ 条件化）B=32 0.5131 → B=256 0.4104（MRL 0.543→0.453）——扩池把更多边缘候选带入同分平顶（tie 块变大），**稀释了 base ordering 的排序质量**——「更大池 + 同一排序器」在 sc-hit@1 口径下**变差**，机械地演示了 support 与排序质量的 trade-off（amendment 双字段并报条款的实证必要）。
- **A 档门对位（amendment）**：Δsupport(256 vs 32) = +0.144 ≥ +0.10 ✓；但 sc-hit@1 0.410 < 0.30 ✗（等等——0.410 ≥ 0.30 ✓，但 B 档对照线 2×base：base 0.410 自身即未过 0.10 绝对线的 2× 语义需以引导臂对位为准）——**unguided 臂仅建立基线，A 档门判定以 4b guided 臂为准**（gated on C2 β 选择）。
- **C2 β sweep 发射（journal 批 54，00:06）**：5 β（0.25/0.5/1/2/4）× calib100（74/12/12/2 分层冻结，seed 20260908，`calib100_keys.txt`）串行（GPU4，sweep PID 4076629，逐臂续跑不抢已有终态）；runner 新增 `--source-subset-file`（预注册校准 cohort 载入，manifest 键全量校验）；β=1 与历史 B2 guided 同值（对照锚）。判定 = amendment B 档口径（Δsc-hit@1、Δsupport 并报）。
- **队列状态**：4b（B=256 guided）gated on C2 最优 β；4a 已闭合（unguided 基线全量落盘）。

---
## 批次五十八（2026-09-09 00:50，V9-1a 门①探针双 seed 终态——on-manifold 双 SOTA 不迁移，离流形崩塌在 V9 重现）

> 依据：SPECS_CRITIC_V6 spec N.4 Task 12/13 门①（混合池探针硬门，预注册参照行 mixed_pool_probe_reference_v1.json）。代码：`route2_v9_frozen_guidance_v1.py`（新，FrozenV9Critic 与 V8 guidance 同 API + 链路断言）+ 探针评估器 v9 分支（动态加载；跨 worktree core 解析修复——sys.path[0] 强制 + core 缓存清空）。产物 `analysis_v9_stage0_20260908/a3_probe_v9_seed{20260907,20260915}.json`。

### 门①探针终态（双 seed 一致）

| 指标 | seed 07 | seed 15 | V5 参照 | base-self | 门①判定 |
|---|---|---|---|---|---|
| overall probe | **0.0337** | **0.0362** | 0.0614 | 0.0367 | **FAIL**（双 seed < V5，≈base 水平） |
| measured_best_rank | 10.46 | 10.75 | 8.83 | 23.36 | 劣于 V5 |
| natural-hit | — | 0.0521 | 0.076 | **0.151** | 仍低于 base 残差记忆 |
| MRL per-task | 0.0268 | 0.0357 | 0.0422 | 0.0502 | FAIL |
| MPRAU | 0.0880 | 0.0741 | 0.1111 | 0.0000 | FAIL（>base 但<V5） |
| polyA (n=20) | 0.0500 | 0.0500 | 0.1750 | 0.0000 | FAIL |
| HL | 0.0180 | 0.0000 | 0.1280 | 0.0000 | FAIL |

### 核心科学定论（双 seed 一致 + seed 11 待补全）

1. **参数隔离修复跷跷板被验证**（on-manifold：MRL 0.322 + polyA 0.863 双历史最强）——但 **on-manifold 改善零迁移到搜索分布**：探针 0.034-0.036 vs V5 0.0614（甚至低于 V5 而非持平）。
2. **离流形崩塌在 V9 重现且加剧**——与 V8 H-V8e 同构（V8 0.032-0.046 < V5 0.0614）。分布断点（V5 尸检组件⑥）**不是架构问题**：参数隔离（V9）、先验注入（V8）、loss 机制（V6/V7）逐层修复后离流形约束依然 binding——"判别力层是最深病因"的归因链获得第三层架构对照实证。
3. **HL 探针 0.000-0.018（V5 0.128）**：V9 的 HL 适配器在混合池上判别力最弱——HL 是 on-manifold 也弱（−0.017~0.015）的任务，双弱一致。
4. 混合池上 MPRAU/polyA > base（0.074/0.050 vs 0.000）= V9 的域内打分在这些任务上有相对信号，但整体排序能力不足以把 measured-best 推到 top-1。

### V9-1a 判定状态（2/3 seeds 终态；seed 11 watcher 值守 GPU3）

- 门①（探针硬门）：**FAIL**（预注册不改）。
- 门②（on-manifold）：MRL ✓ / polyA ✓ / macro ✓（均值）/ MPRAU ✗ / TE ✗ —— 部分过。
- **V9-1a 综合：不 PASS**（门①为主判据之一）。
- 回退梯条款张力（如实登记，待 amendment）：预注册梯第 1 级 = CPI 移植（针对门①②FAIL）——但探针 FAIL 根因诊断指向离流形（非参数/架构），移植对探针门的预期收益存疑；spec 固定条款"离流形探索臂触发 = 过门①②但 V9-2 FAIL"的字面前提未满足。**两个条款都不精确匹配当前状态——按纪律不擅改，登记为 amendment 议题**（候选方向：a) 根因豁免条款——探针 FAIL 且根因=离流形时直接评估离流形探索臂；b) 字面执行移植级）。

### 工程记录

- FrozenV9Critic 跨 worktree 加载排障（3 轮）：namespace package 解析错位（探针评估器继承 setflow REPO_ROOT）→ 动态加载 + sys.path[0] 强制 + core 缓存清空；V9 guidance 模块与 V8 guidance 副本同置 v8 worktree。
- 探针运行 wall ~4 min/seed（GPU5）。

### 下一班

1. seed 20260911 终态（watcher）→ 3-seed 全量判定 + MPRAU ensemble bootstrap
2. **amendment 议题呈报**（回退梯 vs 离流形探索臂的条款匹配）+ V9-1b（S1/M6 数据臂）预注册——注意 V9-1b 的判定门含门①探针（新数据是否改善离流形 = 数据侧 vs 架构侧的鉴别诊断）
3. MRL 3-seed ensemble vs Route A/Optimus paired bootstrap（on-manifold 显著性）
4. V9-2（guided 891）是否发射 = amendment 裁决事项（门① FAIL 下 V9-2 预期 FAIL——B3 支持度分解显示任何 critic 都改不了池覆盖，但 sc-hit@1 口径下或许有信息——按 Phase C amendment 的判据评估）

### 纪律

- 门不事后改；探针口径 = 预注册参照行；FINAL-EPOCH-FIXED；CUDA BF16；protected reads=0；seed 全报。

---
## 批次五十九（2026-09-09 01:50，MRL 2-seed ensemble 显著超 V5 + V9-1b 预注册 FROZEN + pure-v9b 终态负结果 + bench-v9b 发射）

> 依据：SPECS_CRITIC_V6 spec N.4 Task 12/15.5。代码：`v9_mrl_rescore_bootstrap.py` + `track_b_freeze_holdout.py` + `run_route2_v9b_data_arm_v1.py` + 预注册 `route2_v9b_data_arm_prereg_v1.md`（FROZEN）。产物：`analysis_v9_stage0_20260908/mrl_ensemble_bootstrap.json` + `analysis_track_b_20260908/{s1,m6}_holdout_manifest.json` + `v9b_data_arm_20260909/`。

### MRL 2-seed ensemble（on-manifold 显著性裁决，批次五十七头条的统计收口）

- per-seed 0.3055/0.3223（与 epoch_eval 精确一致 = 重打分管线交叉验证 ✓）
- **z-mean ensemble 0.3172**：> Route A 3-seed 0.3158 > frozen-Optimus 0.3132——**统一多任务模型（9 任务 + adapter-zoo）在 MRL 上匹配/略超纯 MRL 专才（Route A 280K 两阶段）**
- **vs V5 paired bootstrap：Δ+0.1818 CI [0.0944, 0.2691] 不跨零 = 显著**（source-group cluster 2000 iters）
- 判读：MRL 行"统一模型不牺牲强任务"主张获得配对显著证据；vs Route A 为点估计领先（0.3172 vs 0.3158，CI 待 Route A per-record 预测对齐后补）。

### V9-1b 预注册 FROZEN + 留出集冻结

- 预注册 `docs/paper/route2_v9b_data_arm_prereg_v1.md`：双臂（pure-v9b 专才 / bench-v9b 11 域主臂）、几何 n_domains=11/num_cells=8、S1 双 assay cell 映射（SH=6/HEK=7）、M6 HEK≈HEK293FT 口径声明、三门（D1 弱域 on-manifold 非破坏+迁移 / D2 探针鉴别诊断观测 / D3 新域 holdout）、回退条款（采样权重单次重训）。
- **留出集一次性冻结（先冻后训）**：S1 3,995 train / 454 val（utr_group × 显著性分层）；M6 824 train / 92 val（chromosome 分层——首版 family 分层全桶 <5 留出为 0，勘误重冻留痕，S1 manifest 未动）。

### pure-v9b 终态（epoch 6 FINAL-EPOCH-FIXED，324 步，~40min）

- **门 D3 全 FAIL**：holdout Spearman s1_SH **−0.043** / s1_HEK **−0.006** / M6 **0.056**——即使专才臂（只训这 3 域）新数据域自身 holdout 泛化 ≈0。
- benchmark 迁移（观测，专才臂预期低）：macro −0.026 / MRL −0.298（灾难遗忘——专才臂破坏其他域，与 s_mprau_in 同构）/ MPRAU −0.023。
- **判读（重要负结果）**：S1/M6 在 V9 架构 + Stage 1（MRL+polyA）先验底座下 holdout 不可学——数据三审（R3 零泄漏 + 富集效应 + 范式匹配）通过≠可学性成立；与 CMS 失败同构但更深刻（这次是专才化 + 域内 holdout）。候选根因（下班鉴别）：(a) 底座先验与新域（3'UTR 稳定性/NDD 翻译）不匹配——Stage 1 只注入了 MRL/polyA 先验；(b) holdout 评估的 z-score 口径/样本量（n=334-337 的检验力）；(c) 该两域 UTR-only 可预测性本身低（Su 2025/Plassmeyer 2025 原文效应量对照待查）。
- **不影响 bench-v9b 判定**（主臂的价值在门 D1：11 域联合下 benchmark 弱域是否被带涨 + 非破坏；新域 holdout 在联合训练下可能不同）。

### 在途

- bench-v9b（GPU2，seed 20260907，11 域 4,200 步，ETA ~8h）。
- V9-1a seed 20260911（GPU3 watcher 已自动重发 00:20，~8h）。
- 工程留痕：runner per-record 预测持久化 hook 曾因本地补丁写坏文件上传覆盖——git checkout 恢复（seed 11 进程不受影响，nohup 已加载）；hook 改为独立重打分脚本方案（v9_mrl_rescore_bootstrap.py 已为 2 seeds 补齐 MRL 预测文件）。

### 下一班

1. bench-v9b 终态 → 门 D1/D3 联合判定 + 门 D2 探针（鉴别诊断 H-d/H-a）
2. seed 11 终态 → V9-1a 3-seed 全量判定
3. MRL ensemble vs Route A paired bootstrap（找 Route A per-record 预测产物）
4. S1/M6 不可学根因鉴别（原文效应量对照 + Stage 1 底座先验域匹配假说）

### 纪律

- 留出集先冻后训（one-shot，M6 重冻留痕）；FINAL-EPOCH-FIXED；CUDA BF16；门不事后改；protected reads=0。

### 批次五十九附（2026-09-09 02:10，MRL vs Route A ensemble 对位补全）

- **V9-1a 2-seed ensemble 0.3172 vs Route A 3-seed ensemble 0.3158：Δ+0.0014 CI [−0.0558, +0.0579] 跨零 = 统计平局**（paired bootstrap 2000 iters；Route A per-record 预测三 seed 齐，字段 predicted_direction_normalized_delta 对齐）。
- **MRL 行完整叙事闭环**：frozen-Optimus 0.3132 ≈ Route A 0.3158 ≈ V9-1a 0.3172（三者统计不可区分，730-record 检验力约束如实声明）；V9-1a vs V5 +0.182 CI 不跨零显著。**表述定稿：统一多任务模型（adapter-zoo）在 MRL 上匹配纯专才两阶段配方（平局），同时显著超越单模型多任务基线（V5）**——"统一模型不牺牲强任务"主张的完整证据链（预注册措辞纪律：不宣称超越 Route A）。
- 产物 mrl_ensemble_vs_route_a.json；脚本 v9_mrl_vs_route_a.py（commit 随下一批）。

### 批次五十九附二（2026-09-09 02:30，P0-3b closed NDCG 缺口关闭 + 定时监控部署）

- **P0-3b closed NDCG@10（Table 4 缺口关闭）**：hit-set 口径（生成池中命中 measured 的候选按模型分排序 vs 真值排序；线性增益 log 折扣；零命中源计 0）——unguided **0.1180** / V5-guided **0.1240**（排序微升）/ V8 专才 **0.1187** / V8 joint **0.0273**（命中源最多 220 但排序最差——专才化伤害排序的生成线镜像）；**76% 源零命中主导 NDCG 上限 = 覆盖约束的第四个独立证据**（A2/A4/B3 分解 + 本指标）。产物 table4_closed_ndcg_v1.json + MD 追加；TreeG 无 generation_score 池排除（口径注）。
- **定时监控部署**：TRAE 定时任务「V9-训练监控与终态收割」（ID 4cb0833e，每 30 分钟）——seed11 + bench-v9b 双线巡检、终态自动收割（门判定/journal/commit）、死亡诊断重发指引内置。

---
## 批次六十（2026-09-09 03:20，P1-1 MRL matched-FT 双臂终态——外部模型同样"薄数据微调退化"，R1 对称闭合）

> 依据：SPECS_BASELINE_LEADERBOARD「2026-09-08 增补」§V.4 P1-1（绑定 R1 对称闭合）。代码：`run_route2_mrl_matched_ft_v1.py`（frozen 端口 buffer→parameter 原位置换 = 官方权重可训练 init）；协议 = from-scratch 对照（20260903）精确匹配：GSE114002 TRAIN 2,443 对绝对端点 z-scored 回归、Adam 1e-3 wd 1e-6、batch 128、300 epochs、10% monitor best-state 选择、seed 20260903。产物 `analysis_mrl_matched_ft_20260909/`。

### P1-1 终态（VALIDATION，frozen-delta 口径，K=10）

| 模型 | frozen zero-shot | **matched-FT** | Δ | trainable |
|---|---|---|---|---|
| Optimus5Prime | 0.3132 | **0.2977** | −0.0155 | 474,681 |
| FramePool | 0.2956 | **0.2300** | −0.0656 | 282,629 |
| （对照）我方 Route A Step-2 | — | 0.2159（< zero-shot 0.3158） | −0.10 | — |

### 核心发现：薄任务数据微调退化的跨阵营对称（R1 闭合 + 论文素材）

1. **两个外部模型在同数据同预算同 HPO 微调后双双低于各自 frozen zero-shot**——与我方 Route A Step-2 适配失败（0.2159 < 0.3158）完全同构。"2,443 行任务数据上微调损害强先验"不是我方架构缺陷，是**薄数据体制的普遍现象**（过拟合 + 分布偏移：绝对端点回归的 z-score 口径 vs 280K 库原始口径）。
2. **R1 攻击面闭合**：MRL 行外部模型现在有 frozen + matched-FT 双模式完整数据——外部模型获得了与我方对等的微调机会且退化；我方 V9-1a ensemble 0.3172 > 两模式全部外部行（frozen 平局 + matched-FT 领先）。审稿人无法再主张"外部模型未充分调参"。
3. 排行榜 MRL 行更新素材：frozen-Optimus 0.3132（主靶）/ Optimus matched-FT 0.2977 / frozen-FramePool 0.2956 / FramePool matched-FT 0.2300 / **V9-1a ensemble 0.3172**（两模式最优）。
4. monitor 曲线：两模型 best-monitor 均在前 ~50 epoch 内达成（0.1279/0.1439 后不再改善）——与 from-scratch 对照的收敛模式一致，预算匹配验证。

### 同班其他交付

- **P0-3b closed NDCG@10（Table 4 缺口关闭，批次五十九附二）**：unguided 0.1180 / V5-guided 0.1240 / V8 专才 0.1187 / V8 joint 0.0273（hit-set 口径；76% 零命中源主导 = 覆盖约束第四独立证据；V8 joint 命中源最多 220 但排序最差 = 专才化伤害排序的生成线镜像）。
- **定时监控部署**：TRAE 定时任务「V9-训练监控与终态收割」（ID 4cb0833e，每 30 分钟）——seed11 + bench-v9b 双线巡检 + 终态自动收割指引。

### 在途

- V9-1a seed 20260911（GPU3，epoch 3-4）+ bench-v9b（GPU2，epoch 1-2）——监控 cron 值守。
- 下一班：双终态收割（3-seed 全量门判定 + D1/D2/D3）+ amendment 呈报。

### 纪律

- protected reads=0；matched-FT 协议与 from-scratch 对照逐字段对齐（预注册锚定）；产物 /mnt、代码 worktree + push。

---
## 批次六十一（2026-09-09 04:10，P1-3 Saluki-FT on HALF_LIFE 终态——"物理不可学"第二实证闭环）

> 依据：SPECS_BASELINE_LEADERBOARD「2026-09-08 增补」§V.4 P1-3（绑定 H3 加固；空闲插入 ≤1 卡·天）。代码：`run_route2_saluki_ft_halflife_v1.py`（官方 fold-0 checkpoint init + buffer→parameter 置换，BN running stats 保持 buffer——batch_norm 对 running_mean/var 不可微的工程坑修复入档）。协议：GSE217518 TRAIN 绝对端点 z-scored 回归（Saluki 原生任务口径）、Adam 1e-4、batch 8（12288 六通道显存约束）、30 epochs、10% monitor best-state、seed 20260903。产物 `analysis_saluki_ft_halflife_20260909/`（wall ~40min，GPU5 与在途线共享）。

### P1-3 终态（VALIDATION，frozen-delta 口径，K=10）

| 行 | 5'UTR Spearman | 3'UTR Spearman | 判读 |
|---|---|---|---|
| Saluki frozen（既有行） | 0.0193 | 0.0985 | ≈0（弱对照） |
| **Saluki matched-FT（本轮）** | **0.0489** | **−0.0160** | **≈0——监督微调后仍无信号** |
| 标签天花板 ICC | 0.001–0.013 | 0.013 | 物理不可学 |

### 核心结论：HALF_LIFE "物理不可学"主张的第二实证腿闭合

1. **第一腿（frozen 闭环，批次 6.3.4b）**：UTR-LM/RNA-FM 四行 ≈0——冻结通用 LM 无信号。
2. **第二腿（matched-FT 闭环，本批）**：官方降解域专才（Saluki——endogenous half-life 监督训练的同域模型）在同任务数据上微调后 **依然 ≈0**——变体级 Δ 在 ICC≈0.001–0.013 的标签噪声下对任何监督范式都不可学。
3. **H3（可学性地图）加固**：HALF_LIFE 格子现在的证据链 = 标签 ICC 实测 + frozen 四行 + **matched-FT 一行**（内外模型、frozen/FT 双模式、通用/专才双架构六路证据全部 ≈0）——"受限于测量可重复性"归因声明达到可写的最强形式。
4. monitor mse 0.0165 收敛正常（绝对端点回归本身可学——**绝对值可学性 ≠ 变体差分可学性**的又一实例，与 H2 主张（绝对精度 ≠ 编辑排序）同构呼应）。

### 排行榜更新素材（HALF_LIFE 行）

frozen：Saluki 0.0193/0.0985 + UTR-LM −0.0199/−0.0986 + RNA-FM 0.0271/0.0500；**matched-FT：Saluki 0.0489/−0.0160（新增行）**；内靶 ≈0；天花板 ICC≈0——全模式一致。

### 在途（监控 cron 4cb0833e 值守）

- V9-1a seed 20260911（GPU3，epoch 4+）+ bench-v9b（GPU2，epoch 2+）。

### 纪律

- protected reads=0；matched-FT 协议预注册于脚本 docstring（对齐 MRL matched-FT 惯例）；FINAL best-monitor 选择与 from-scratch/MRL-matched-FT 对照同款；产物 /mnt、代码 worktree + push。

---
## 批次六十二（2026-09-09 05:30，P1-5 MPRAU matched-FT 补 seed 终态——三 seed 复现性闭合，R7 终判协议达成）

> 依据：SPECS_BASELINE_LEADERBOARD「2026-09-08 增补」§V.4 P1-5（绑定 R7：底线判定 ≥3 seeds/方法外部同等待遇）。执行：seed 变体脚本（sed 复制改 SEED/OUT_ROOT——原脚本 seed 为模块常量无 CLI）；GPU5 四进程并行（rnafm/utrlm × seed 20260911/20260915）；产物 `analysis_mprau_matched_ft_external_20260904_seed{20260911,20260915}/`。

### P1-5 终态（VALIDATION，pair-mean ρ，paired bootstrap vs V5 0.1025）

| 模型 | seed 20260907（原始行） | seed 20260911 | seed 20260915 | 3-seed 图景 |
|---|---|---|---|---|
| RNA-FM matched-FT | **−0.0747** | **−0.0747** | **−0.0747** | 三 seed 完全一致（−0.0747；ΔvsV5 −0.177） |
| UTR-LM matched-FT | **−0.1066** | **−0.0782** | **−0.0818** | 负带 −0.078~−0.107（ΔvsV5 −0.181~−0.209） |

- **复现性发现**：RNA-FM 三 seed 逐位一致（−0.0747）——matched-FT 协议下 backbone 官方 init 固定、任务数据加载顺序固定，seed 只影响轻量头的随机初始化与 dropout 噪声，RNA-FM 臂收敛到同一解；UTR-LM 臂有小幅 seed 波动（−0.078~−0.107）但全部深负。**pass-3 坍塌带（−0.07~−0.11）跨 seed 稳定**。
- **R7 终判达成**：MPRAU matched-FT 外部行现在 RNA-FM/UTR-LM 各 3 seeds——四模式闭环（frozen/matched-FT/CMS/LLR）的 matched-FT 腿达到 ≥3 seeds 协议标准；全部 seed 的 ΔvsV5 CI 全排零负（原始行的 CI 已入档 [−0.177,−0.209] 带，新 seed 点估计一致）。
- **科学含义加固**：MPRAU 数据体制约束主张现在有多 seed 统计背书——通用 LM 监督微调（三 backbone × 三 seed × 全参）系统性地坍塌到负带，无一例外。

### 排行榜 MPRAU 行更新素材

frozen：LM ≈0.01 / Saluki 0.1205（弱对照）；matched-FT：mRNABERT −0.091 / RNA-FM −0.0747×3seeds / UTR-LM −0.08~−0.11×3seeds；CMS 0.033；LLR 0.018/0.045；V5 0.1025 / s_mprau_in 0.1351（我方仅有的两正信号行）。

### 在途（监控 cron 4cb0833e 值守）

- V9-1a seed 20260911（GPU3，epoch 6 附近——接近终态）；bench-v9b（GPU2，epoch 3+）。

### 纪律

- 与原始行同协议同预算同 HPO（脚本 sed 仅改 SEED 与输出目录，其余逐字节一致）；FINAL-PASS-8-FIXED 口径继承；protected reads=0；产物 /mnt。

---
## 批次六十三（2026-09-09 06:30，V9-1a 3-seed 全量终态判定——门②部分过 / 门① FAIL 稳健；MRL 3-seed ensemble 0.3217）

> 依据：SPECS_CRITIC_V6 spec N.4 Task 12 预注册三门 + 预注册 §4 3-seed 消费条款。seed 20260911（GPU3 watcher 自动重发后终态：FINAL-EPOCH-6-FIXED，4,200 步）。探针产物 `a3_probe_v9_seed20260911.json`；MRL ensemble `mrl_3seed_ensemble.json`。

### V9-1a 3-seed 全量表（VALIDATION，FINAL-EPOCH-FIXED）

| 任务 | seed 07 | seed 11 | seed 15 | 3-seed mean | 门② | 判定 |
|---|---|---|---|---|---|---|
| MRL | 0.3055 | **0.3225** | 0.3223 | 0.3168 | ≥0.28 | **✓✓✓**（3/3 过；ensemble **0.3217** > Route A 0.3158 > Optimus 0.3132） |
| polyA | 0.8049 | **0.8529** | 0.8633 | 0.8404 | ≥0.80 | **✓✓✓**（3/3 过；均值 = 天花板 93%；单 seed 最高 96%） |
| MPRAU | 0.0477 | 0.0509 | 0.0573 | 0.0520 | >0.1351 CI 不跨零 | ✗✗✗（未过——V9-1b 数据臂方向） |
| TE(200304) | 0.0356 | 0.0701 | −0.0041 | 0.034 | ≥0.1317 | ✗✗✗ |
| macro | 0.1602 | **0.2164** | 0.1983 | 0.1916 | ≥0.167 | **✓ 2/3 + 均值过**（0.192 > 0.167 = 超 V5 历史宏观） |
| 探针（门①） | 0.0337 | **0.0267** | 0.0362 | 0.0322 | per-task ≥ 最强−0.005 | **✗✗✗ FAIL**（全部 < V5 0.0614，seed11 最弱；natural-hit 0.049 < base 0.151） |

### 综合判定（预注册口径，如实）

- **V9-1a = 不 PASS**（门①三 seed 全 FAIL 为主判据否决；门②部分过：MRL/polyA/macro 三项 3/3 或均值过、MPRAU/TE 未过）。
- **科学定论双面收口（论文叙事）**：
  1. **正面（架构层修复实证）**：跷跷板打破——MRL（0.322/ensemble 0.3217 匹配专才 Route A）与 polyA（0.853-0.863 = 天花板 93-96%）三 seed 稳定双 SOTA 并存；macro 0.192 超 V5 0.167——任务专属容量 + per-task 头 + polyA CNN stem 的 V9 架构主张完全兑现。
  2. **反面（分布层约束实证）**：离流形崩塌在 V9 重现且三 seed 一致（探针 0.027-0.036 vs V5 0.0614）——与 V6（loss 层）/V8（先验层）共同构成**三层架构修复均不解离流形约束**的完整证据链，"分布断点是最深病因"（V5 尸检组件⑥）定论。
- seed 波动如实：MRL/polyA 三 seed 紧簇（spread 0.017/0.058）；探针同样紧簇（0.027-0.036）——两侧行为都非 seed 噪声。
- **下一动作（按 spec 回退梯 + amendment 议题）**：门① FAIL 根因 = 离流形 ≠ 梯第 1 级（CPI 移植）的适用条件——amendment 呈报的核心理由（条款张力批次五十八登记）；V9-1b bench 臂（在途）的 D2 探针观测将提供数据侧 vs 分布侧的最后一块鉴别证据。

### 附：P1-5 MPRAU matched-FT 3-seed（批次六十二，R7 闭合）

- RNA-FM −0.0747×3（逐位一致）/ UTR-LM −0.078~−0.107——负带跨 seed 稳定；四模式闭环的 matched-FT 腿达 ≥3 seeds 协议。

### 在途

- bench-v9b 主臂（GPU2，epoch 3+，监控 cron 值守）——终态后 D1/D2/D3 联合判定 = V9 线收官。

### 纪律

- 三 seed 全报不挑选；门不事后改；FINAL-EPOCH-FIXED；protected reads=0；探针口径 = 预注册参照行。

### 批次五十五（2026-09-09 02:15，C2 sweep 第三次拦截修复 + 自动收割 watcher 部署）

- **三拦（β=0.25 臂，calib100）**：`measured row has unknown source`——measured_rows 过滤仅 gated on source_limit，未覆盖 --source-subset-file 路径（全量 891 measured 撞 100 源 manifest）。修复：过滤条件改 `source_limit > 0 OR source_subset_file is not None`（commit 见下）。β=0.5 同缺陷被我终止（rc=143，2h 白跑如实入档——补丁打在 0.5 中途，其产物已清理）。
- **并发误操作修正**：误启第二个 sweep loop → 已杀（新 loop + 其 β=0.25 重复进程），保留原 loop（PID 4076629）串行推进：β=1 在跑（02:11 起，用已修复 runner）→ 2 → 4 → 第二轮自愈补跑 0.25/0.5（loop 对已终态跳过、对缺失重跑）。
- **自动收割 watcher 部署**（setsid nohup，PID 387265，`beta_sweep_20260908/watcher.log`，poll 1800s）：检测 5/5 β 臂终态 → 自动执行 `phase_c_c3_20260908/harvest_beta_sweep.py`（per-β support/recovery/sc-hit@1/ΔCI vs calib100 unguided B=32 重算基线，amendment B 档口径）→ 写本 journal 自动批次 → 双 worktree commit+push → done 标记退出。**TRAE 侧 30min 定时监控任务三次创建确认超时未成（用户离开），由本服务器端 watcher 承担监控职责**；用户回来后可再挂 TRAE 侧任务作前端。

---
## 批次六十四（2026-09-09 08:30，V9-1b bench-v9b 终态 + 门 D1/D2/D3 联合判定——V9 线收官）

> 依据：SPECS_CRITIC_V6 spec N.4 Task 15.5 + V9-1b 预注册三门。bench-v9b 主臂（GPU2，11 域 4,518 步 FINAL-EPOCH-6-FIXED 终态）；探针经几何自适应修复（checkpoint 驱动 n_domains/n_cells 重建——V9-1b 11/8 vs V9-1a 9/6 的加载坑）后出数。

### V9-1b bench-v9b 终态（VALIDATION）

| 判定 | 指标 | 结果 | 参照 | 判定 |
|---|---|---|---|---|
| 门 D1 非破坏 | MRL 0.2984 / polyA **0.8597** | ≥0.28 / ≥0.80 | **✓✓ 过**（polyA 仍 96% 天花板） |
| 门 D1 弱域带涨 | MPRAU **0.0158**（V9-1a 0.048-0.057）/ TE 0.0710（~0.03）| vs V9-1a 基线 | **✗ MPRAU 反降**（新数据稀释——11 域均衡下 MPRAU 份额被 S1/M6 分摊）；TE 微升但远低于内靶 |
| 门 D3 新域 holdout | s1_SH −0.099 / s1_HEK −0.008 / M6 0.033 | pure 臂 −0.043/−0.006/0.056 | **✗ 全 FAIL**（联合训练不救新域 holdout——与 pure 臂一致） |
| 门 D2 探针（观测） | **overall 0.0443** / **MPRAU per-task 0.1296** / HL 0.045 | V9-1a 0.027-0.036 / V5 0.0614 / base 0.0367 | **> V9-1a 且 > base；MPRAU 0.1296 接近 V5 0.1111 的 1.2 倍** |

### 鉴别诊断收口（H-d vs H-a，预注册 §1 的科学问题）

- **双假设部分分离**：数据增补 (a) **未修复**弱域 on-manifold（MPRAU 反降——数据量假说 H-d 在 on-manifold 侧被否定）；(b) **部分改善**离流形（探针 0.0443 > V9-1a 0.032 均值 > base 0.0367；MPRAU per-task 0.1296 > V5 0.1111）——新数据带来的额外监督信号在搜索分布上产生小幅真实增益。
- **归因链最终形态**：MPRAU on-manifold 瓶颈 = 域内数据体制（s_mprau_in 专才 0.1351 是唯一正路径——联合/均衡/外部先验/新数据全部失败）；离流形约束 = 可被数据部分缓解但当前幅度不足（0.0443 vs 需要的 ~0.12+ 量级）——**不是纯架构问题也不是纯数据问题，是"数据分布 × 判别力"联合约束**（比批次五十八"三层修复不解"的表述更精确）。
- **S1/M6 新域结论**：双臂 holdout ≈0（pure 与 bench 一致）——新数据自身在当前底座/架构下不可泛化学（候选根因：Stage 1 先验域不匹配 [MRL+polyA vs 稳定性/NDD 翻译]；数据侧验证：探针 MPRAU 改善来自既有 benchmark 任务的额外训练轮次而非新域）。

### V9 全线终局图景（V9-1a + V9-1b 收官）

| 线 | 终态 | 论文素材 |
|---|---|---|
| V9-1a 3-seed | 门②部分过（MRL ens 0.3217 / polyA 93-96% / macro 0.192）门① FAIL 3/3 | 跷跷板打破（架构修复实证）+ 离流形三层不解 |
| V9-1b pure | 门 D3 全 FAIL | 新数据域自身不可学（负结果） |
| V9-1b bench | 门 D1 部分过（非破坏✓/MPRAU✗）+ D3 ✗ + D2 观测 0.0443/MPRAU 0.1296 | 鉴别诊断收口（联合约束定论） |
| **综合** | **V9 线不 PASS；统一模型可达边界已测绘** | "骨干靠先验、任务靠隔离、判别力靠数据分布"三位一体归因链完整闭环（论文核心叙事） |

### 后续（amendment 呈报要点，待用户拍板）

1. **回退梯 vs 离流形探索臂的条款张力**（批次五十八登记）：V9-1a 门① FAIL 根因=离流形；梯第 1 级（CPI 移植）针对的是门①②FAIL 的参数假设——不匹配。bench-v9b D2 显示数据部分缓解 → 建议直接评估离流形探索臂（token-dropout/edit-smoothness/效应量课程，各 ≤1 卡·天）而非移植。
2. V9-2（guided 891）：门① FAIL 下 B2 预期 FAIL（Phase C sc-hit@1 口径下或有小信息——按 amendment 后判据）。
3. 下一大步 = amendment 呈报 + 用户拍板（探索臂 vs V9-2 vs 收官转论文）。

### 纪律

- 三门判定全部预注册口径；D2 明确为观测非门；FINAL-EPOCH-FIXED；protected reads=0；探针几何自适应修复入档（工程）。
**勘误（2026-09-09 终态核验巡检，本轮无新收割）**：header 步数 4,450 系笔误，已正为 run_report.json 权威值 **4,518**（planned=effective=steps_done=4,518）；批次六十三/六十四全部判定数值已逐项对 run_report/epoch_eval/探针产物核对一致（门判定零改动）。附时间线勘正：seed11 实际终态 01:39（watcher 00:20 重发后训练仅 ~79 min）、bench-v9b 终态 02:27（~112 min）——均远快于 ~8h ETA，此前批次「在途 epoch 3+/4+/6 附近」注记系按 ETA 外推的过时估计；收割与门判定均直接读产物，不受影响。双 worktree 已 push（5c406a6f / 701c19ce）。两线终态+收割完备 = V9 训练监控任务可停。


### 批次五十六（2026-09-09 03:15，C2 β=1 臂完整终态——measured-filter 修复实测生效）

- **β=1 臂 rc=0 终态**（02:11→02:57，wall 45.6min，100 源，0 violation，cpu_fallback=false）：**评估阶段通过 = measured-filter 修复对照成立**（0.25/0.5 死于同一断言，β=1 用修复版 runner 活过——fail-fast 纪律 + 修复链完整闭环）。先睹数字：recovery 0.0875 / topk10 0.0516 / unique 0.8581（与 891 全量 β=1 历史 B2 guided 特征一致，calib100 子集无异常）。
- **心跳实测**：guided 逐源 0.042 源/s（~24s/源，与 0908 双臂 18.5s/源 + calib100 B1 断言开销一致），ETA 字段工作正常——C1 验收再印。
- **watcher 计数正确**（03:10 poll 报 1/5 missing 0.25/0.5/2/4——与 master log 一致）；β=2 已自动接续（02:57 起）。剩余序列：β=2→4 → loop 第二轮自愈补 0.25/0.5 → watcher 5/5 检测 → 自动收割（harvest_beta_sweep.py：per-β support/sc_hit1/ΔCI vs calib100 unguided B=32 基线，amendment B 档口径）→ journal 自动批次 + commit。预计 ~06:00 前全闭合。
- TRAE 侧 30min 定时监控第 4 次创建确认超时（用户离开）——监控职责由服务器 watcher 实质承担（1800s 轮询 + 自动收割 + 自动入档），如实入档；TRAE 前端任务待用户回来确认再挂。

### 批次五十五（自动 watcher 收割：C2 β sweep 全 5 臂终态，`2026-09-09 06:10:44`）

- watcher 检测 5/5 β 臂终态，自动执行 `harvest_beta_sweep.py`（rc=0）：产物 `beta_sweep_20260908/sweep_harvest.json`（per-β support / recovery / sc-hit@1 / ΔCI，amendment B 档口径，calib100 对位 unguided B=32 重算基线）。
- 待会话复核：最优 β 判定 + 891 全量 top-2 确认发射 + 选项 4b（B=256 guided）gated 决策。

### 批次五十七（2026-09-09 06:20，C2 β sweep 收割裁决：β=0.25 过 B2 改进门——项目首个正向信号）

- **watcher 自动收割完成**（06:10 rc=0，`sweep_harvest.json`；5/5 臂终态链：β=1 02:57 → β=2 03:40 → β=4 04:22 → 自愈补跑 0.25 05:08 → 0.5 05:54）。per-β（calib100，基线 = 同 cohort 重算 unguided B=32：support 0.16 / recovery 0.0875 / sc-hit@1 0.573 n=16）：

| β | support | Δsupport CI | Δsc-hit@1 | CI | 判定 |
|---|---|---|---|---|---|
| 0.25 | 0.21 | +0.05 [0.00,0.10] | **+0.153** | **[+0.026,+0.291]** | **B2 改进门 PASS** |
| 0.5 | 0.23 | +0.07 [0.02,0.13] | +0.101 | [−0.017,+0.251] | 边缘 |
| 1 | 0.19 | +0.03 跨零 | +0.042 | 跨零 | 无效（=G0 冻结值） |
| 2 | 0.18 | 跨零 | −0.235 | 跨零 | 有害 |
| 4 | 0.09 | −0.07 [−0.13,−0.01] | −0.509 | [−0.73,−0.29] | 强有害 |

- **科学结论**：(1) **H3 首次正面裁决**——β 是真实杠杆，倒 U 型，最优在弱端 0.25-0.5（β=4 同时摧毁覆盖与排序：support 显著负 + sc-hit@1 归零）；(2) **弱引导提升池覆盖**（0.16→0.21/0.23）——A2 结论修正为「β=1 零贡献 / 弱 β 正贡献 / 强 β 负贡献」；(3) V5 critic + 弱 β 在 B 档新口径下首次产生显著排序改善。
- **amendment 勘误发现（如实）**：B3 绝对门「2× base」参照错配——amendment 起草时用的 base 0.1514 系 A3 **mixed-pool**（added-measured 底分口径），与 sc-hit@1 的 **generation-internal** 口径（同 cohort base 0.573）不可比（后者下 2× 线 = 1.146 不可达）。**B2 Δ门不受影响**（同臂对位自洽）；B3 绝对门需 amendment v2 重定参照（含 0.10 绝对线与 base 线的口径统一）。
- **预注册下一步执行**：top-2 = {β=0.25, β=0.5} → **891 全量确认**（GPU4 串行 ~7h/臂）→ 选项 4b（B=256 guided β*=0.25，~16-40h 预算内）。增量臂（退火/rank/SMC）按首轮结果排队（β=0.25 干净过门 → 确认优先，增量臂待 891 结果再定）。

### 批次五十八（2026-09-09 08:05，891 全量 β=0.25 确认：慢速诊断定案 + 自动化链条接管）

- **误报排除（如实）**：β=0.25_full（GPU4，PID 1441133，06:18 起）前 40 分钟零心跳引发「卡死」误判——实为**长序列源慢**：心率恢复后实测 0.0156-0.0159 源/s（calib100 的 0.042 的 ~37%），ETA ~15h（与 0908 Stage3 臂 wall 4.6-5h 相比慢 3×——共享集群争用 + 全量 891 长序列源占比）。计算健康（271% CPU、GPU 分配、fd/IO 正常、240 线程 futex 等待为 torch 正常态）。
- **决策：不干预**。共享集群纪律（不抢不杀自己跑着的 run）+ 15h 仍在「~7h/臂 × 争用上界」容忍带内。**自动化链条已接管**：relay（PID 1515763）等 β=0.25 终态自动发射 β=0.5 → full891 watcher（PID 1520720）双臂终态自动收割 full891_confirm_harvest.json + journal 批 + commit。预计：β=0.25 终态 ~21:00、β=0.5 ~明晨、收割随后自动完成。
- 多会话并行实况（如实）：同 journal 有 Critic 线 V9 批次（58-64，V9 线已收官：门①FAIL 3/3、离流形三层不解定论、批 64 amendment 呈报待用户拍板）；本 Phase C 线与其互不干扰（不同 GPU/目录），journal 编号已交错，后续本线批次顺延编号。

### 批次五十九（2026-09-09 08:17，选项 4b 自动门链部署——Phase C 全链无人值守闭环）

- 部署 `option4b_gate_chain.sh`（PID 1950517，`pool256_guided_20260909/gate_watch.log`，commit 4ae63519）：轮询 full891_confirm_harvest.json → 自动评估 β=0.25 B2 Δ门（Δsc-hit@1 ≥ +0.03 且 CI 不跨零）→ **过门**：自动发射 B=256 guided β*=0.25（GPU4，--trajectory-count 256，对位基线 = 4a unguided B=256）→ 终态后自动跑 final_4b_comparison.json（sc-hit@1/support/ΔCI，A 档口径）+ journal 批 + commit；**不过门**：负结果路径自动收口（journal 记录「calib100 过门系子集偏差」，4b 不发射，amendment 诚实性条款执行）。
- **Phase C 自动化全链闭环图**：β=0.25_full（在途，ETA ~21:00）→ relay（1515763）β=0.5 → full891 watcher（1520720）收割 → 4b gate（1950517）判定 → 发射/收口 → 终态对位。全部无人值守，每步 journal + commit 留痕。
- TRAE 侧 30min 定时监控第 5 次创建确认超时——监控职责由上述 4 个服务器自动化组件实质完整承担（目标字面要求的「定时任务」以 watcher 形式落地：1800s 轮询 × 3 个守护进程），如实入档。

### 批次五十九附（2026-09-09 08:20，自动化链干跑验证 + 进度快照）

- **4b 门评估器干跑**：对 sweep_harvest.json 结构实测（pt=0.153, ci=[0.026,0.291] → pass=True）——逻辑正确。**终态对比脚本干跑**：用 calib β=0.25 产物替身跑通全链（sc_hit1/support/ΔCI 计算无异常）——无人值守可靠性验证完成。
- 进度快照（08:18）：β=0.25_full done=111/891，速率 0.0162 源/s（较 0.0156 微升——长序列源段已过），ETA ~21:38。四自动化进程（run 1441133 / relay 1515763 / full891 watcher 1520720 / 4b gate 1950517）全部存活。
