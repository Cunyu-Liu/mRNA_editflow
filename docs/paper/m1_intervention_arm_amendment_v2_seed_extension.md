# M1 干预臂 3-seed 扩展 Amendment v2（**ACTIVE — 已冻结**）

- **change-id**: m1-intervention-arm-seed-extension-v2-20260928
- **状态**: **ACTIVE — FROZEN**（冻结时间 2026-09-28T19:15Z / 2026-09-28 03:15 CST；本文件冻结于任何扩展臂训练发射之前，且任何扩展臂数据/G1 读数均未被读取）
- **上游**: `docs/paper/m1_intervention_arm_amendment_v1.md`（M1 四门 + 单 seed 方向判定，已冻结执行中）
- **取代**: `docs/paper/m1_intervention_arm_amendment_v2_seed_extension_DRAFT.md`（DRAFT 版，保留为存档；其 §1 种子消解、§2 配置、§3 判定全部逐字继承，仅 §4 发射时机条款在本 ACTIVE 版中修订）

## 1. 背景与 v1 口径歧义的消解

- v1 §1.1 判定口径行原文：「单 seed 判方向；3-seed（20260903/20260904/20260905 复用两枚 + 新 seed 20260920）后 CI 判显著（复用 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 3-seed ensemble 口径，bootstrap 2000 / seed 20260816）」。
- **歧义**："复用两枚" 未指名哪两枚。**v2 消解（发射前冻结，禁止事后改）**：扩展臂种子值 = **20260904 与 20260905**；理由（固定条款）：20260903 是 v1 明示的「V2 单 seed 参照点 0.3198」来源，保持不克隆以免参照与臂混淆；20260904/20260905 来自 V2 seed 族、与 V2 的对应 run 构成 2/3 配对种子，最大化"同架构同训练、仅数据面不同"的可比性。
- **两臂构成（冻结）**：
  - M1-3seed 集合 = {**20260920**（v1 主臂）, **20260904**, **20260905**}（后两枚为 v2 扩展训练臂）。
  - V2-3seed 参照集合 = {20260903, 20260904, 20260905}（既有存档 ensemble 0.3158，不重训）。

## 2. 扩展臂训练配置（与 v1 逐项同款，唯一改动面 = seed）

| 项 | 值 | 依据 |
|---|---|---|
| runner | `scripts/route_a_v3/run_route2_m1_intervention_fullft_v1.py`（逐字复用，传 `--seed`，未传 `--m1-seed`/`--m1-rows` 即默认 20260920/300000） | v1 同款 |
| 数据 | 280K 干净库 677,608 + M1 下采样 300,000（**`--m1-seed` 默认 20260920 固定不动**，与 v1 逐位一致）；评测行 4,848 唯一序列 + 4 条鸽笼标记全程排除 | v1 §1.2 |
| 超参/调度/精度 | batch 128 / lr 2e-5 / wd 1e-4 / AdamW warmup5%+cosine / BF16 / 6 epochs / FINAL-EPOCH-6-FIXED | v1 §1.1 |
| 唯一改动面 | `--seed 20260904` / `--seed 20260905`（`torch.manual_seed(seed)` 于模型头初始化前执行并驱动 `torch.randperm` 的 batch 顺序） | 本 v2 §1 |
| 产物根 | `experiments/xeditcritic_m1_intervention/seed_20260904/`、`seed_20260905/`（append-only，不动 20260920 目录） | append-only |
| GPU/预算 | A100-40G（有 ≥10GB 自由显存即用——用户 09-28 共享卡政策）；~5-6.5h/seed；有卡即并行发射 | 项目纪律 |
| 硬门 | CUDA 不可用即 runner 内 `SystemExit("CUDA unavailable - GPU required")` 停留证；protected TEST reads = 0 | 项目纪律 |

## 3. 冻结判定（激活后零修改）

- **主判定（3-seed ensemble 对比）**：M1-3seed z-mean（PER_SEED_ZSCORE_MEAN 口径，复用既有 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 脚本口径）对 **V2-3seed ensemble 0.3158289984824722**（`ensemble_3seed_vs_optimus.json` 存档值）做 source-group paired bootstrap（2,000 iters，seed 20260816，K=10，GSE114002 VALIDATION 730 行）→ **Δ point + CI95**；CI 排零为正 = 显著改善成立。
- **报告项**：逐 seed 值全报（三枚）；单 seed 谱（V2 谱 0.3198/0.2873/0.3157 作波动带参照）；G2/G3 按 v1 同款（扩展臂各自 polyA 复测 + V2 context 行由收割脚本自动产出）。
- **禁改条款**：FINAL-EPOCH-6-FIXED；不得以扩展结果替换 v1 单 seed 判定读数（v1 判定读数保留，3-seed 为追加行）；任何门槛调整须 v3 amendment。

## 4. 发射与执行清单

### 4.1 发射时机（本 ACTIVE 版的唯一实质修订，发射前冻结）

- 扩展臂训练 = **方向无关的算力预提交（compute pre-commitment）**：发射不等待 G1 读数。理由（固定条款）：(a) 扩展臂训练配置与 G1 结果无关（同 runner、同数据、仅 seed 不同），不存在按结果定制训练的可能；(b) G1 读数只能在 seed 20260920 终态后获得，而 v1 明令 FINAL-EPOCH-6-FIXED、禁止中间读数，故"先读 G1 再发射"必然以拖延算力为代价；(c) 用户 09-28 指令要求 GPU 闲置显存即刻投入使用、禁用显存侧 gate。
- **仅分析激活依赖 G1**（见 4.2），故本修订不改变任何判定规则或报告口径。
- 本修订为**发射前冻结**：冻结时点早于任一扩展臂的第一个 heartbeat，且任何扩展臂数据未被读取、G1 未被读取——不构成事后改。

### 4.2 分析激活（唯一依赖 G1 的环节）

- `m1_adjudication_v1.json → G1.direction_positive == true` 时：执行 3-seed ensemble 收割（§3 主判定），结果作为 §5.6 的**显著性追加行**写入 draft；同时逐 seed 值全报。
- `G1.direction_positive == false` 时：扩展臂归档为 **trained-not-analyzed**——不进入任何表格、不进入 draft、不参与任何 robustness 叙述；journal 注明"方向负，扩展臂未用（预提交算力损耗已记）"。**不得**以扩展臂结果替换或"稀释" v1 单 seed 判定读数。
- 自动收割器须双条件同时成立才可执行 3-seed 收割：**G1 方向为正** 且 **三枚臂均达终态**（`run_summary.json` 存在）。任一不满足则跳过并在日志注明。

### 4.3 执行清单

1. [x] 本文件冻结 ACTIVE + commit（2026-09-28T19:15Z，先于发射）。
2. [x] 发射两臂（`--seed 20260904` / `--seed 20260905`，各自独立卡，发射记录见 journal 批次 127 与各 arm `training_pid.txt`）。
3. [ ] 两臂终态 → 3-seed 收割脚本（`scripts/route_a_v3/harvest_route2_m1_intervention_3seed_v1.py`，z-mean ensemble + paired bootstrap，复用 V2 ensemble 口径）。
4. [ ] 结果回填 §5.6（v1 单 seed 判定的**显著性追加行**）+ journal 批次。

## 5. 纪律自检

- [x] 触发条款唯一且可机检（G1.direction_positive 与三枚终态）
- [x] 种子歧义在发射前以固定条款消解（无事后裁量）
- [x] 本文件冻结早于发射；发射后不再修改本文件（新信息进 journal）
- [x] protected TEST reads = 0；扩展臂仅读 VALIDATION
- [x] 不替换 v1 判定读数；FINAL-EPOCH-6-FIXED 不变