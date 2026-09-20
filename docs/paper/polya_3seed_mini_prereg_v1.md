# V5 polyA 3-seed 补跑 mini prereg v1（P0-1 / rigor audit A1 最大攻击面封堵）

- **change-id**: v5-polya-3seed-20260920
- **落盘日期**: 2026-09-20（训练发射前冻结；发射后本文件零修改）
- **状态**: PREREGISTERED（发射前冻结）
- **上游判定输入**: `docs/paper/deltabench_rigor_audit_v1.md` A1（V5 polyA 主行单 seed = 全骨架最锋利攻击面）+ 补跑清单 P0-1
- **背景事实（实查）**: polyA 主行 0.8219 = critic V5（frozen terminal）`PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1` VALIDATION n=2,628 overall Spearman = 0.8218686245779881（`adjudicate_route2_apa_route_a_v1.py` critic_v5 行）；原 seed = 20260907（`experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c…/v5_full/training_attempt.json` `parameter_initialization_seed`，SHARED_V4_CONSTRUCTOR_WITHIN_IDENTICAL_ARCHITECTURE）；V5 训练实测 55,636.7s ≈ 15.5h/seed（`run_summary.json` elapsed_seconds，8 pass / 22,416 updates / A100-40G 整卡 BF16 / peak 8.3GB）→ 2 新 seed ≈ 31h 总量，1 天内/seed，**不触发降级预案**。
- **角色边界**: 稳健性补跑臂（seed 方差带），**不是 SOTA 冲刺、不是主行替换**。

## 1. 训练配置（与 V5 主行完全同款，唯一改动面 = seed）

### 1.1 不变量（逐项克隆自冻结的 V5 screen_config）

| 项 | 值 | 对齐依据 |
|---|---|---|
| 训练核心 | `train_route2_xeditcritic_v4.py` 的 V4-FULL 通道共享核心（`XEditCriticV4` + `TrainableMRNABERTUpperSixEncoderV4` + bottom_six 冻结缓存），**代码路径逐字复用 import，不改任何模型/损失/优化器代码** | 主行 runner 同源 |
| 模型 | V4-FULL：mRNABERT 冻结底六层（缓存 `frozen_bottom_six_chunk_cache_v1.pt`）+ 可训上六层 + 12 edit blocks + 语义专家 4×top-2 + readout 2560；可训参数 170,481,957（容量门 165-175M 复核） | screen_config architecture + preflight 实测 |
| 多任务数据 | 9 任务 canonical TRAIN 107,873 行（TRAIN 89,580 / VALIDATION 18,293 / withheld TEST 18,292），projections `development_train_validation_v1/{train,validation}.jsonl` | screen_config data_geometry |
| 采样/几何 | `FixedEffectiveTaskBatchSamplerV4`：SQRT 任务大小、任务同质、source-group 均衡；effective batch 32 / physical batch 32 / repeat cap 4 / 8 pass / 2,802 updates/pass / 22,416 总 updates | screen_config data_geometry |
| 优化器 | AdamW wd 1e-4；lr 三组 {mrnabert_top_six 1e-5, new_head_and_v4_trunk 2e-4, semantic_experts_and_router 1e-4}；warmup 5% + cosine to 10%；grad clip 1.0 | screen_config training |
| 损失 | STANDARDIZED_HUBER + CROSS_GROUP_PAIRWISE + SOFT_SPEARMAN + WITHIN_SOURCE（pass1-2: huber1.0/pair0.25/wsr0.5；pass3-8: huber1.0/pair0.5/router0.01/soft0.25/wsr0.5，温度 0.2）；TASK_ROBUST_STANDARDIZED 聚合 | screen_config training loss 权重表 |
| 目标缩放 | TRAIN task-robust scaler（floor 0.001），region-global fallback | 主行同款 |
| 精度/后端 | BF16 autocast 前向 + FP32 有效目标；PYTORCH_SDPA_AUTO；activation checkpointing on | training_attempt.json |
| 终态协议 | **FINAL-PASS-8-FIXED**（final_pass_8_checkpoint.pt；无验证峰值重选） | screen_config checkpoint_selection |
| 终态评测 | VALIDATION 18,293 行 `final_validation_predictions.jsonl`（逐行 target/prediction/scaled 对）+ `validation_metrics`（9 任务 task_macro） | 主行同款产物 schema |

### 1.2 唯一改动面

- **seed**: 新增 2 枚 = **20260921、20260922**（`_set_seed` 注入点 = 模型构建前，`parameter_initialization_seed_applied_before_model_construction=True` 口径复刻；同时驱动 sampler 与 dropout RNG）。声明：**3-seed = 原 20260907 + 新 20260921/20260922**。新 seed 与 V4 协议既有 seed 族（screen 20260907 / confirmation 20260908-10）不重叠、与既有 MPRAU/MRL 多 seed 族不重叠，独立性如实登记。
- **输出根**: `experiments/xeditcritic_v5_polya_3seed/seed_<s>/`（新建，不动 xeditcritic_v5 主行目录）。
- **runner**: bespoke `run_route2_xeditcritic_v5_polya_3seed_v1.py`——因官方 `train_route2_xeditcritic_v4.py` SCREEN 通道硬门锁 `seed==20260907` + 旧 HEAD launch-authorization + clean worktree（新 seed 无法走旧授权），本 runner 以库方式 import 官方 runner 的共享函数集（模型构建/优化器组/损失目标/sampler/评测/checkpoint schema/实验 ledger），**绕过且仅绕过 SCREEN 通道四门（seed 白名单、旧 HEAD 授权、preflight 绑定、clean worktree）**，其余纪律逐项保留（CUDA A100 硬门、BF16 硬门、ledger 上报、TEST/Eval reads=0、append-only、失败终态落盘）。此路径先例 = M1 干预臂 bespoke runner（journal 批次 112）。

### 1.3 发射配置

- **序列化排队**: watcher `launch_v5_polya_3seed_watcher.sh`（模式抄 M1 watcher）：先轮询 M1 心跳（`xeditcritic_m1_intervention/seed_20260920/heartbeat.json`），**仅当 M1 进入 TRAINING/RUNNING 状态（其 watcher 完成抢卡发射）后**，polyA watcher 才开始自己的抢卡循环（每 600s 查 GPU 0-5 空闲整卡 <2GB 且 0 进程 + 120s 二次确认）。**两 seed 严格串行：seed 20260921 训练进程到达终态（DONE/FAILED 或进程消失）后才排队 seed 20260922**。全程不与 M1 竞争同一张卡（M1 训练占卡期间 GPU 0-5 无空闲整卡，polyA watcher 自然 WAITING；M1 终态释放卡后才抢）。
- **GPU**: A100-40G 整卡（GPU 0-5 范围，物理索引直通，禁 CUDA_VISIBLE_DEVICES 重映射——官方 runner 同款断言）。
- **时长预算**: ~15.5h/seed（对齐主行 55,636s 实测）；两 seed 串行 + 排队等待，总墙钟视 GPU 释放时间 1-3 天。
- **心跳**: 每完成一个 pass 写 `heartbeat.json`（status/seed/pass/UTC/elapsed），run_summary.json 终态落盘。

## 2. 冻结判定（发射后零修改）

### 2.1 报告主判定（polyA 稳健性，主指标）

- **指标**: GSE269595 VALIDATION n=2,628 polyA overall Spearman（= 主行 0.8219 口径：`final_validation_predictions.jsonl` polyA 行的 target vs prediction Spearman）。
- **报告项**: 3-seed 均值 ± range（原 0.8219 + 两新 seed 点估计）；逐 seed 值全报，**不挑选**。
- **CI**: 3-seed 均值的 source-group paired bootstrap 95% CI（2,000 iters，seed 20260920），与 APARENT 0.7343 frozen-delta 的 Δ 对比（Δ = 3-seed 均值 − 0.7343，bootstrap 同口径）。
- **榜单处理（冻结）**: **主行 0.8219 保持不变**（单 seed frozen terminal 原值原样保留）；新增行 = polyA-V5-3seed-mean（附 CI + 逐 seed 值），附注于主行脚注（"3-seed 稳健性见新行"）。**任何情况下不以 3-seed 均值替换主行**。

### 2.2 Holm 家族方向重算（报告项）

- 以新 3-seed 均值为 polyA ours 值，重算 bottom-line 9 任务 Holm 家族中 polyA 行的显著性方向（家族其余 8 行原值不动）；仅报告方向（SIGNIFICANT / not significant），raw_p 依赖逐 record 配对重算，在评测收割会话执行。
- 若 3-seed 均值使 polyA vs APARENT Δ 的 bootstrap CI 仍排零 → 方向维持 SIGNIFICANT（升格 3-seed 稳健）；若跨零 → 如实改写为区间主张（0.8219 单 seed 声明保留）。

### 2.3 polyA 评测口径（训练终态后执行，本任务负责发射与排队机制）

- frozen-Δ 口径对 VALIDATION 2,628 行（polyA cell 的 source/candidate 对，pred(cand)−pred(src)）+ 双口径（top-1 / NDCG@10，K=10，`evaluate_route2_prediction_v1.py` 零改动）。
- 由 watcher 在两 seed 均终态后自动触发，或后续会话收割（本任务如实报告发射状态即可）。

### 2.4 禁改条款

- 预注册门发射后零修改；FINAL-PASS-8-FIXED 禁挑峰；逐 pass 指标仅诊断（V4 协议 pass_rows validation_metric_read=False 口径保持——本 runner 不做逐 pass 验证读取，与主行一致）。
- protected TEST reads = 0（TEST 18,292 行全程不可触碰）；Evaluation 行（M1/M6/S1）不读取。
- 训练中不读任何评测行 outcome；评测仅在训练终态后执行。

## 3. 解释框架（预登记）

1. **若三 seed 均值 ≥ 0.80 且 CI 排零** → 0.8219 升格「3-seed 稳健」声明（A1 攻击面 SEALED）。
2. **若 seed 方差大（range > 0.03）** → 如实改写为区间主张（骨架 §8.2 已有单 seed 显式声明，替换为区间表述），**不隐藏不挑选**。
3. **若某 seed 训练失败**（OOM/非终态）→ 该 seed 如实记 FAILED，3-seed 判定降级为「2-seed + 声明」，不静默重试换 seed。
4. 训练时长若实测远超 15.5h/seed × 2（>2 天/seed），如实报告并降级为「单 seed 补 + 声明」预案（不硬来）——以主行实测 55,636s 为基准，预期不触发。

## 4. 交付物清单

| 交付物 | 路径 |
|---|---|
| 训练 runner | `scripts/route_a_v3/run_route2_xeditcritic_v5_polya_3seed_v1.py` |
| 排队 watcher | `scripts/route_a_v3/launch_v5_polya_3seed_watcher.sh` |
| 本 prereg | `docs/paper/polya_3seed_mini_prereg_v1.md` |
| seed 产物 | `experiments/xeditcritic_v5_polya_3seed/seed_20260921/`、`seed_20260922/`（heartbeat/training_attempt/run_summary/final_pass_8_checkpoint/final_validation_predictions） |
| 确定性检查（同批 P2-1） | `experiments/analysis_forward_determinism_v1/determinism.json` |
