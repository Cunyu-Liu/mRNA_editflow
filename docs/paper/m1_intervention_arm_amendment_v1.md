# M1 真实稠密数据干预验证臂（Task 3.4 / R6 方向 A 收编）预注册 Amendment v1

- **change-id**: m1-intervention-arm-route-a-20260920
- **落盘日期**: 2026-09-20（发射前落盘；发射后本文件零修改）
- **状态**: PREREGISTERED（训练发射前冻结；门值冻结）
- **主管线**: Route A full-FT V2（`run_route2_mrnabert_280k_fullft_v2.py`，产物 `xeditcritic_route_a/280k_fullft_v2_6ep_20260903` 及 seed 20260904/20260905）
- **本臂角色**: 机理干预验证臂（R6 收编、方向 A 收编），**不是 SOTA 冲刺**。测试命题：在既有 MRL delta 管线训练侧加入真实高密度外部 MRL 语料（M1），能否按双因子规律（监督体制 × 数据几何，spec v4 / delta-density v2 held-out 支持）预测的方向改善 delta 判别力。
- **上游判定输入**: Task 2.4 判定（`analysis_delta_vs_density_v2`）：密度单因子降级为任务特异性现象；双因子分组成立（above-band 域内 6/11 vs below-band 域内 0/21）。STCNet 域内 MRL 0.8144 与跨库 M1 仅 0.0672 的对照提示域内监督收益部分绑定语料级记忆——本臂以自有干预实验正面检验"加入域内语料能否兑现增益"。

## 1. 训练配置（与 Route A full-FT V2 同款 + 单一数据侧增量）

### 1.1 不变量（克隆自 full-FT V2，逐项对齐）

| 项 | 值 | 对齐依据 |
|---|---|---|
| backbone | mrnabert_a1eb7df…（`AutoModel` + MeanPoolRegressor 线性头，full-FT） | V2 同源 |
| 训练库（基座部分） | sample280K 677,608 clean 行（GSM3130435/36，mean-rep 聚合，3-block 近邻泄流审计排除后） | V2 同款 `load_library`+`audit_leakage`，逐字复用 |
| epochs | 6，FINAL-EPOCH(6)-FIXED 判定（逐 epoch 指标仅诊断，禁后验挑峰） | V2 同款 |
| batch / lr / weight-decay / 优化器 | 128 / 2e-5 / 1e-4 / AdamW（warmup 5% + cosine） | V2 同款 |
| 序列输入口径 | 空格分隔大写 DNA 50nt（tokenizer add_special_tokens、无截断） | V2 同款 `format_sequence` |
| MRL 主评测 | GSE114002 VALIDATION 730 行 frozen-delta（K=10 task macro Spearman，`evaluate_route2_prediction_v1.py` 零改动） | V2 同款 |
| 判定口径 | 单 seed 判方向；3-seed（20260903/20260904/20260905 复用两枚 + 新 seed 20260920）后 CI 判显著（复用 `adjudicate_route2_fullft_v2_3seed_ensemble_v1.py` 3-seed ensemble 口径，bootstrap 2000 / seed 20260816） | V2/裁定管线同款 |
| 泄流纪律 | MRL 库保留 V2 泄流审计；**M1 侧 = 排除 benchmark_v2/m1_mrl_eval_row 全部 4,848 条唯一序列（sources+candidates）+ 4 条 TRAIN 重叠鸽笼标记序列** | 见 §1.2 |

### 1.2 增量（唯一改动面：数据侧，M1 并入 TRAIN）

- **M1 语料**: `route2/external_model_assets/castillohair2024/mrl_converted_20260909/`（GSE232927 Castillo-Hair 2024；鸽笼审计 PASS）。文件选择 = **HepG2 r1 + T-cell r1/r2**（defined_end 文库，M1 评测行主构成域；HEK293T random_end 三文件不进入本臂——评测行 HEK293T 405 行的域内覆盖让渡给对照解读，记入局限）。**与 v8_m1_prefinetune 先例（6 文件 589,503 行）的偏差已如实登记**：先例为多域联合预训练口径（V8 系，domain slot m1 复用 MRL readout），本臂为 Route A 单任务 delta 管线收编口径。
- **行选择**: keep=1 行；rl_paper 为标签（权威列，V8 先例与评测行同口径）。
- **下采样**: 目标 300,000 行（建议区间 200-400K 中值，控算力），seed=20260920 固定，`rng.sample(sorted(pool))` 确定性。对三文件池（约 789,709 行）均匀池化后无放回抽样。
- **序列口径镜像**: M1 utr 列 = 50nt 大写 DNA 与 GSE114002 canonical 5'UTR 50nt 同长度同字母表；空格分隔 tokenization 与 280K 库逐字同款（"canonical 5'UTR 50nt 口径"镜像达成——无长度再投影必要，如实记录）。
- **标签标准化**: M1 rl_paper 与 280K rl 拼接后**联合 z-score**（单一回归头单一输出分布，最简设计；对"双分布分头"的替代设计记为未采用备选，理由 = 改动最小化条款）。
- **顺序**: 每 epoch 联合池 shuffle（V2 同款 `torch.randperm` 语义），M1 与 280K 行同批混洗，无 domain-balanced sampler（对齐 V2 极简批次语义）。
- **排除规则（泄流）**: ① m1_mrl_eval_row 全部唯一序列 4,848 条（跨三文件）从训练池排除（评测行冻结的对面条款）；② 4 条 TRAIN 重叠鸽笼标记序列（tcell_r1×2 + tcell_r2×2）排除（该 4 条本为 keep=0，双保险如实登记）。

### 1.3 发射配置

- **seed**: 20260920（单 seed 起步；3-seed 扩展为后续会话收割决策）
- **GPU**: 整卡（W0 统计判卡；无空闲则排队等待并如实报告 WAITING）
- **时长预算**: 对齐 V2 单 seed 量级（V2 ≈ 4.5-5.5 小时；本臂行数 +44% ≈ 6.5-8 小时，A100-40G 整卡）
- **产物根**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_m1_intervention/seed_20260920/`

## 2. 冻结判定门（四门，发射后零修改）

### G1 增益方向门（主门）
- **指标**: MRL VALIDATION frozen-delta task macro Spearman（GSE114002 VALIDATION 730 行，K=10，V2 同口径）。
- **基线**: 0.3158（Route A full-FT V2 3-seed ensemble，PER_SEED_ZSCORE_MEAN；`ensemble_3seed_vs_optimus.json` 入档 0.3158289984824722）。
- **判定**: 单 seed（20260920）判方向——**增益 > +0.01 = 方向正**；≤ +0.01 = 方向负/中性。单 seed 结论仅为方向读数；3-seed 后 CI（bootstrap 2000）判显著。参照单点：V2 单 seed 20260903 = 0.3198。
- **说明**: ensemble 0.3158 为 3-seed 均值参照而非单 seed 比较点，单 seed 方向判读时同时报 V2 单 seed 谱（0.3198 及 seed 20260904/20260905 两点）作波动带参照。

### G2 非破坏门
- **指标**: polyA VALIDATION 主行 task macro Spearman（GSE269595 VALIDATION，V5 0.8219 口径参照，`analysis_apa_route_a_adjudication_20260903` critic_v5 = 0.8218686245779881）。
- **判定**: 本臂模型在 polyA 主行上复测，回落 ≤ 0.02（相对 0.8219 口径）视为非破坏。本臂 polyA 复测为收割时零调参 frozen-Δ 推理（复用既有评测管线口径）。
- **预期说明**: 本臂训练仅含 MRL 域数据（280K + M1），无 polyA 域训练——G2 检验的是 MRL 域内加料是否通过共享表征/负迁移损害跨任务可用性；Route A full-FT V2 本身无 polyA 域训练，其 polyA 参照为冻结 mrnabert 初始表征 + V5 判定面，故 G2 以 V5 口径为静态参照面如实报告落差（预期方向：若 M1 语料不引入灾难性表征退化，回落应主要反映"MRL 域过训漂移"幅度）。

### G3 效率门（报告项，不设阈值）
- 训练时长（墙钟）、峰值显存（`torch.cuda.max_memory_allocated`）、训练行数（677,608 + 300,000 = 977,608/epoch）与 V2 单 seed 对照表如实报告。

### G4 机理门（方向性对照，门值不设硬阈值）
- **触发条件**: 仅当 G1 单 seed 方向正（> +0.01）时执行；方向负/中性则记录"不触发"。
- **内容**: 在 M1 评测行（`route2/benchmark_v2/m1_mrl_eval_row/`，2,805 行/2,801 源）上复测 delta 排序（frozen-Δ 零调参，与 benchmark_v2 矩阵同口径），方向性对照 = **域内语料应同向改善**（若 GSE114002 域增益成立，M1 行亦应相对基线模型的 M1 行读数改善）。结果如实报，无硬阈值。
- **对照锚点**: STCNet 0.8144（域内 MRL 历史最强）与跨库 M1 0.0672 的"域内语料绑定"现象——本臂是"把域外语料变成域内语料"的正面对照实验；本臂 M1 行读数（加料后）vs 全外部基线谱（矩阵 14 族 M1 列）的相对位置一并报告。

## 3. 解释框架（预登记，发射前冻结）

1. **增益 FAIL 也是有效交付**：真实高密度外部语料未改善 → 双因子规律中"密度/域内监督"因子独立贡献受限的干预证据；与 Task 2.4 held-out 发现（密度与域内监督共变、单因子失准）互为印证，写入机制章节而非丢弃。
2. **G4 与 STCNet 现象对照**: STCNet 域内 0.8144 / 跨库 0.0672 提示其收益部分绑定"评测序列直接在训练语料内"的语料级记忆；本臂把语料真正加入训练侧但**不包含评测序列本身**（评测行排除条款），是记忆绑定假设的解耦检验——若本臂增益接近零而 STCNet 域内强，则支持"语料级记忆绑定"强于"表征级域适应"的解读。
3. **单 seed 局限如实登记**: 单 seed 方向读数受 seed 波动影响（V2 三 seed 单点谱待收割时报告）；3-seed 扩展与终判收割由后续会话在训练终态后执行（本任务只负责发射与预注册）。
4. **禁挑峰条款**: FINAL-EPOCH(6)-FIXED；逐 epoch frozen-delta 指标仅诊断收敛归因，不进入判定。

## 4. 交付物清单

- 训练产物：`xeditcritic_m1_intervention/seed_20260920/`（run_summary JSON、逐 epoch checkpoint、epoch_frozen_delta_metrics.jsonl、training_losses.jsonl、心跳文件、log）
- 本 amendment（发射前 commit）
- 训练脚本 `scripts/route_a_v3/run_route2_m1_intervention_fullft_v1.py`（W0 worktree，commit）
- journal 批次一百一十二发射记录

## 5. 纪律条款（自检清单）

- [x] protected TEST reads = 0（训练/评测仅 VALIDATION 面；canonical GSE114002 读取仅取 VALIDATION 730 行，与 V2 同款）
- [x] M1 行只进 TRAIN 侧；benchmark_v2/m1_mrl_eval_row 评测行冻结不动（训练池排除 4,848 条唯一序列）
- [x] 预注册门发射后零修改（本文件落盘于发射前，commit 顺序 = amendment 先于训练发射 commit）
- [x] CUDA 硬门（无 GPU 不发射，如实报告 WAITING）
- [x] 模型/损失/超参/优化器零改动（唯一改动面 = 数据侧加载与拼接、输出产物路径、心跳）
