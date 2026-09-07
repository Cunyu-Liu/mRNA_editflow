# V8 Stage 2 多任务适配预注册（FROZEN v1，2026-09-07）

> **状态：FROZEN**。触发条件满足：Stage 1 全量判定 RESOLVED（`adjudication_v8_stage1.json`，
> 2026-09-07 04:05 自动落盘）。本文档由 DRAFT v1（commit de1f2170）按预注册条款冻结，
> 仅补入 Stage 1 全量判定数字、架构选择 amendment 的触发结果、预算对账与臂定义；判定门
> 与训练设计照 DRAFT 不变。冻结后门槛不事后改（修订走 amendment，留痕）。
> 依据：SPECS_CRITIC_V6/spec.md V8 攻坚线 Stage 2 + `mprau_prior_survey_20260904.md`。

## 0. Stage 1 全量判定（2026-09-07 04:05，本预注册的输入）

| 臂 | MRL spearman（GSE114002 VALIDATION, n=730）| polyA spearman（GSE269595 VALIDATION, n=2,628）| MRL 非破坏门 0.2768 | polyA 非破坏门 0.1883 | S vs H |
|---|---|---|---|---|---|
| S（纯 mRNABERT 联合）| **0.3078** | 0.1122 | ✓ | **✗ FAIL（0.1122 < 0.1883）** | MRL winner（Δ+0.0202 ≥ −0.02）|
| H（hybrid CNN stem）| 0.2876 | **0.5262** | ✓ | ✓ | — |

- polyA 单域基线（v8p，`--arch s --libraries polya`，epochs 6，FINAL-EPOCH-FIXED）：Spearman **0.2092**（阈值 = 0.9 × 0.2092 = 0.1883）。
- `stage1_success = true`（MRL 主判据）；S 臂 polyA 非破坏门 FAIL，H 臂 PASS。
- 补充：V8 S/H 权重 MPRAU zero-shot pair-mean ≈ 0（S −0.0005 / H −0.0064）——跨域先验不携带等位偏移信号，Stage 2 必须用 CMS 域内适配（zero-shot 对照臂基线 0 入档）。

## 1. 架构选择 amendment（DRAFT §1 条款触发，2026-09-07 冻结）

DRAFT §1 预注册条款原文：「若 polyA 非破坏门在 S 上 FAIL 而 H 上 PASS，按 amendment
将主臂改为 H（或 S+H 双适配臂），MRL 0.02 容忍带条款不覆盖跨域架构选择」。

- **触发结果**：S 上 FAIL（0.1122 < 0.1883）、H 上 PASS（0.5262 ≥ 0.1883）→ **条款触发**。
- **决策**：主臂 = **H**；S 保留为**对照臂**（S+H 双适配臂，二者同库同预算同 seed）。
- 理由（证据链）：① Stage 2 polyA 门 ≥0.80 要求适配不得破坏 polyA，S 起点 0.1122 远低于
  H 起点 0.5262，H 是唯一可达目标带的架构；② polyA 域 H 完胜（Δ+0.414）呼应 Saluki
  conv 论断与 W2 混合架构依据；③ Stage 2 判定门全部按 DRAFT §3 不变，本 amendment 只改
  主臂选择，不改任何门槛。

## 2. 训练设计（照 DRAFT §2 冻结）

- 数据级任务均衡采样（`DomainBalancedSampler`，每 batch 等额任务配额）。
- **细胞条件输入保留原始每细胞标签**（不聚合）：V8 骨干新增可选 cell embedding（6 个
  ENCODE 上下文：GM12878/HEK293FT/HEPG2/HMEC/K562/SKNSH），加到 pooled 表示；CMS 与
  ENCSR854RUF 行均保留 per-cell label。
- 适配方式对照：**full-param 臂 + LoRA 臂（r16 α32，qkv+o+mlp）**；zero-shot 对照臂必设。
- MPRAU：CMS array 85,475 行库（域内适配，泄漏审计 flagged=0 硬门已于 09-06 通过；
  同 assay 同管线先验 = 唯一未测试先验来源，V8 Stage 1 是决定性实验的前提）。
- 训练配方：AdamW lr 2e-5 wd 1e-4、cosine 至 10%（5% warmup）、BF16 autocast、
  seed 20260907、epochs 6（batch 自适应，pre-launch amendment：free≥28GiB→128 /
  ≥20GiB→96 / ≥15GiB→64 / ≥11GiB→48 / ≥7GiB→32，与 APA 政策一致）、FINAL-EPOCH-FIXED。

## 3. 判定门（冻结，不可事后改）

| 门 | 阈值 | 口径 |
|---|---|---|
| MRL | ≥0.28 | GSE114002 VALIDATION task_macro_spearman（frozen-delta，K=10）|
| polyA | ≥0.80 | GSE269595 VALIDATION task_macro_spearman（frozen-delta）|
| **MPRAU（主判据）** | **>0.1025 且 CI 不跨零** | ENCSR854RUF VALIDATION variant pair-mean ρ（2,008 变体），paired bootstrap 2,000 iters seed 20260816 vs V5 |
| TE 族 | ≥ 内靶 0.1317 | GSE217518/GSE256185 等 TE family VALIDATION task_macro |
| macro | 显著升 vs 0.167 | 全部 VALIDATION 任务 macro（paired bootstrap seed 20260816）|

- 纪律：smoke/proxy/训练集结果不构成科学结论；FINAL-EPOCH-FIXED 禁 peak-picking；
  protected reads = 0（TEST 不碰）；CUDA BF16-only（cpu_fallback_used=false 留证）。

## 4. 臂定义与预算对账（冻结）

| 臂 | arch | init | 适配方式 | 数据 | cell 条件 | 目的 |
|---|---|---|---|---|---|---|
| **h_cms_full（主臂）** | H | Stage1 H epoch2 | full-param | CMS 85,475 行（domain cms=2）| 是 | MPRAU 域内适配主判据 |
| s_cms_full（对照）| S | Stage1 S epoch2 | full-param | CMS 85,475 行 | 是 | S+H 双适配对照 |
| h_cms_lora（对照）| H | Stage1 H epoch2 | LoRA r16 α32 | CMS 85,475 行 | 是 | 适配方式对照（参数量级）|
| **h_bench9（多任务臂）** | H | Stage1 H epoch2 | full-param | benchmark TRAIN 池（8 study，pair-delta 均衡采样；MPRAU 55,704 行 per-cell）| 是（MPRAU）| 多任务适配全门判定 |
| zero-shot 对照 | — | Stage1 S/H 直接 | 无 | 9 任务 VALIDATION | — | 基线对照（MRL S 0.3078 / polyA H 0.5262 / MPRAU ≈0）|

预算（每臂，batch 自适应后以实际 run_report 为准）：

- cms 臂：85,475 行/域，steps/epoch = ceil(85,475/batch)；6 epochs。
- bench 臂：TRAIN 池行数（实测 development_manifest TRAIN）：ENCSR854RUF 55,704 /
  GSE114002 2,443 / GSE186455 204 / GSE200304 3,318 / GSE217518 2,201 / GSE256185（同池）/
  GSE269595 25,710；GSE149487 无 TRAIN（96 VALIDATION 仅评估）。均衡采样 steps/epoch =
  ceil(Σ行数/batch)；6 epochs。
- LoRA 臂参数量级：trainable ≈ r16 秩残差 × 4 目标 × 12 层 + 头/domain/cell embeddings，
  与 full-param（113.9M）数量级对照。

## 5. 已知限制（冻结时声明）

- Stage 1 双臂仅 2 epochs（106,812 步）；Stage 2 从 epoch-2 权重适配 6 epochs。
- CMS 覆盖 7,284/9,740 变体（74.8%），缺失 25% 为 rs-id 变体（已知限制，不影响本预注册判定）。
- polyA 门参照 V5 0.8219 为既有冻结行；本预注册不重测 V5，仅引用。

## 6. 后继（预注册）

- Stage 2 终态后 24h 内执行 Stage 3：V8 冻结 → SetFlow B2 重跑（同 base/seed/预算）→
  V8-critic vs V5-critic guided Δrecovery 逐任务对位（SetFlow 侧预算另立）。
- D5（MPRAU 目标带 40%）在 Stage 2 + CMS 首臂数据出齐后 amendment 修订（旧 40% 不删）。

## 7. 波次 2 增补（2026-09-07 11:40，预注册澄清 + CMS 首波结果）

- **DRAFT §2 澄清**：原 DRAFT「MPRAU：用 V8 权重 + ENCSR854RUF（TRAIN 55,704）适配；若适配仍负，CMS array 注入为注册后继」——即**域内 ENCSR854RUF TRAIN 是首要路径，CMS 是注册后继**。波次 1 先发 CMS 臂系执行顺序调整（runner 共用），现按 DRAFT 补发域内臂。
- **波次 1（CMS 域内适配）三臂已终态（09-07 11:26）**：
  - h_cms_full（full-FT，4008 步）：MPRAU pair-mean **0.0327**，vs V5 Δ−0.0698 CI [−0.133, −0.010] 全负 → **主判据 FAIL（显著低于 V5）**
  - s_cms_full（full-FT，8016 步）：**0.0385**，CI [−0.125, +0.003] 跨零 → FAIL
  - h_cms_lora（LoRA，8016 步）：**−0.0069**，CI 全负 → FAIL
  - 判读：CMS 同 assay 库适配产生弱正信号（full-FT +0.033~0.039 vs zero-shot ≈0），但远低于 V5 0.1025；LoRA 容量不足（−0.007）。**CMS 作为外部 MPRAU 先验的假说未获确认**（适配后仍显著低于域内训练 V5）。
- **波次 2（域内 ENCSR854RUF TRAIN 55,704 行，DRAFT 首要路径）**：
  - h_mprau_in（GPU1，batch 128，full-FT）：发射 11:33
  - s_mprau_in（GPU2，batch 64，full-FT）：发射 11:33
  - h_mprau_lora（GPU5，batch 96，LoRA）：发射 11:45
  - h_bench9（GPU3，batch 64，均衡多任务适配，epoch 3 在途）
- 判定门不变（§3）；波次 2 结果并入 adjudication_v8_stage2.json（adjudicator 自动扫描 ARM_ORDER 已含新臂，需将新臂名加入扫描表）。
