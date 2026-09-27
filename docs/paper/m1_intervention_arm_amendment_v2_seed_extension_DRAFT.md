# M1 干预臂 3-seed 扩展 Amendment v2（DRAFT — 条件激活）

- **change-id**: m1-intervention-arm-seed-extension-v2-20260928
- **落盘日期**: 2026-09-28（**草拟**；**仅在下方触发条件满足时冻结激活**——激活动作 = 发射前 commit 并在本文件顶部标记 ACTIVE+时间戳）
- **状态**: **DRAFT — NOT ACTIVE**（未发射、不占 GPU、不改变 v1 任何门限）
- **上游**: `docs/paper/m1_intervention_arm_amendment_v1.md`（M1 四门 + 单 seed 方向判定，已冻结执行中）
- **触发条件（唯一）**: M1 臂 seed 20260920 终态后，**G1 方向为正**（即 `m1_adjudication_v1.json → G1.direction_positive == true`，判定值 > 0.3158 + 0.01）。G1 不为正则本 amendment 不激活（文件保留为未用草稿，journal 注明）。

## 1. 背景与 v1 口径歧义的消解

- v1 §1.1 判定口径行原文：「单 seed 判方向；3-seed（20260903/20260904/20260905 复用两枚 + 新 seed 20260920）后 CI 判显著（复用 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 3-seed ensemble 口径，bootstrap 2000 / seed 20260816）」。
- **歧义**："复用两枚" 未指名哪两枚。**v2 消解（发射前冻结，禁止事后改）**：扩展臂种子值 = **20260904 与 20260905**；理由（固定条款）：20260903 是 v1 明示的「V2 单 seed 参照点 0.3198」来源，保持不克隆以免参照与臂混淆；20260904/20260905 来自 V2 seed 族、与 V2 的对应 run 构成 2/3 配对种子，最大化"同架构同训练、仅数据面不同"的可比性。
- **两臂构成（冻结）**：
  - M1-3seed 集合 = {**20260920**（v1 主臂，已完成）, **20260904**, **20260905**}（后两枚为本 v2 扩展训练臂）。
  - V2-3seed 参照集合 = {20260903, 20260904, 20260905}（既有存档 ensemble 0.3158，不重训）。

## 2. 扩展臂训练配置（与 v1 逐项同款，唯一改动面 = seed）

| 项 | 值 | 依据 |
|---|---|---|
| runner | `scripts/route_a_v3/run_route2_m1_intervention_fullft_v1.py`（逐字复用，传 `--seed`） | v1 同款 |
| 数据 | 280K 干净库 677,608 + M1 下采样 300,000（seed 20260920 固定，与 v1 逐位一致）；评测行 4,848 唯一序列 + 4 条鸽笼标记全程排除 | v1 §1.2 |
| 超参/调度/精度 | batch 128 / lr 2e-5 / wd 1e-4 / AdamW warmup5%+cosine / BF16 / 6 epochs / FINAL-EPOCH-6-FIXED | v1 §1.1 |
| 唯一改动面 | `--seed 20260904` / `--seed 20260905`（作用于模型构建前，驱动初始化与 batch 顺序 RNG） | 本 v2 §1 |
| 产物根 | `experiments/xeditcritic_m1_intervention/seed_20260904/`、`seed_20260905/`（新建，不动 20260920 目录） | append-only |
| GPU/预算 | A100-40G 整卡（有 ≥10GB 自由显存即用——用户 09-28 共享卡政策）；~5-6.5h/seed；有卡即并行发射 | 项目纪律 |
| 硬门 | CUDA 不可用即停留证；cpu_fallback 硬拒；protected TEST reads = 0 | 项目纪律 |

## 3. 冻结判定（激活后零修改）

- **主判定（3-seed ensemble 对比）**：M1-3seed z-mean（PER_SEED_ZSCORE_MEAN 口径，复用既有 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 脚本口径）对 **V2-3seed ensemble 0.3158289984824722**（`ensemble_3seed_vs_optimus.json` 存档值）做 source-group paired bootstrap（2,000 iters，seed 20260816，K=10，GSE114002 VALIDATION 730 行）→ **Δ point + CI95**；CI 排零为正 = 显著改善成立。
- **报告项**：逐 seed 值全报（三枚）；单 seed 谱（V2 谱 0.3198/0.2873/0.3157 作波动带参照）；G2/G3 按 v1 同款（扩展臂各自 polyA 复测 + V2 context 行由收割脚本自动产出）。
- **G4 条件不变**：沿用 v1（仅当方向正触发；本次激活前提即为方向正，故 G4 在 v1 收割时已触发/未触发按 v1 执行，不重复）。
- **禁改条款**：FINAL-EPOCH-6-FIXED；不得以扩展结果替换 v1 单 seed 判定读数（v1 判定读数保留，3-seed 为追加行）；任何门槛调整须 v3 amendment。

## 4. 激活与执行清单（仅在触发条件满足时执行）

1. [ ] 核验 `m1_adjudication_v1.json → G1.direction_positive == true`（否则本文件不激活，记录 journal 后归档）。
2. [ ] 本文件顶部改 `ACTIVE + 时间戳`，同 commit 冻结（含 §1 种子消解与 §2 配置）。
3. [ ] 发射 watcher（模式抄既有 watcher；独立卡获取 + flock 序列化防同卡竞争——2026-09-28 OOM 教训）。
4. [ ] 两臂终态 → 3-seed 收割脚本（z-mean ensemble + paired bootstrap，复用 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 口径，输入换为 M1 臂三枚 predictions）。
5. [ ] 结果回填 §5.6（作为 v1 单 seed 判定的**显著性追加行**）+ journal 批次。

## 5. 纪律自检

- [x] 本文件为 DRAFT：未发射、未占卡、未改动 v1 门限与产物
- [x] 触发条件唯一且可机检（G1.direction_positive）
- [x] 种子歧义在发射前以固定条款消解（无事后裁量）
- [x] protected TEST reads = 0；扩展臂仅读 VALIDATION