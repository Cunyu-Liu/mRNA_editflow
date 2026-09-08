# V9-1b 数据增补臂预注册（route2_v9_adapter_zoo_data v1）

**状态**：FROZEN（2026-09-09；SPECS_CRITIC_V6 spec N.4 双轨执行案 Task 15.5——轨道 B 数据审计通过后 V9-1b 为独立增补臂）
**依据**：批次五十六（S1/M6 构建 + R3 五研究零泄漏 + 富集效应审计通过）；V9-1a 2-seed 终态（批次五十七/五十八：门②部分过 = MPRAU/TE 弱域缺口，门①探针 FAIL = 离流形崩塌）

## 1. 科学问题（本臂的双重鉴别诊断价值）

V9-1a 门①（混合池探针）FAIL 与门②的 MPRAU/TE 缺口呈现**不同根因假设**：
- **H-d（数据侧）**：弱域缺口（MPRAU 0.048-0.057 / TE ~0）源于训练池内该域的变体级监督量不足（MPRAU 有效 2,008 变体 × 均衡稀释；TE 族薄数据）——数据增补直接修复。
- **H-a（架构/分布侧）**：探针 FAIL 是离流形协变量偏移（V5 尸检组件⑥），与数据量无关——数据增补不改善探针（新数据仍来自"实验者选择偏倚"分布）。

**本臂的鉴别诊断设计**：S1/M6 增补后 (a) 弱域 on-manifold 提升 + 探针不变 → 两假设可分（弱域=数据问题、离流形=分布问题，分别修）；(b) 双双不变 → 数据体制约束结论再加固；(c) 双双提升 → 离流形也是数据量问题（修正归因链）。无论哪个结果都是论文素材。

## 2. 数据（批次五十六产物，全部已过三审）

| 库 | 规模 | 窗口 | 标签 | R3 | 效应体制 |
|---|---|---|---|---|---|
| S1（Su 2025 稳定性） | 4,540 SNV 对 | 155nt 转录本方向 | y_SH = WT−mt decay（SH-SY5Y，n=3,351）/ y_HEK（n=3,295）双 assay | **flagged=0** | 富集（中位 23.6） |
| M6（Plassmeyer 2025 NDD 翻译） | 916 SNV 对 | 100nt | y_v0 = log2((sum_ALT+1)/(sum_REF+1)) 总计数口径（polysome 口径待 GEO 元数据，如实标注） | **flagged=0** | 富集（中位 0.254） |

- **S1 双 assay = context 维度**（同变体双读出，映射到 cell conditioning：SH→新增 cell id 6，HEK→cell id 7；复用 V8 六 ENCODE cell embedding 机制，num_cells=8）。
- **M6 cell = HEK293T**（42 SIC 样本全 HEK 系；context id 复用 CELL_IDS["HEK293FT"]=1 的语义近似——**口径声明：HEK293 ≠ HEK293FT，作为 context 近似使用并在产物中标注**）。
- **z-score 标准化**：per-library 标准化（与 V8 Stage 2 per-study 同法）。
- **域分配**：S1 → 新 domain id 9（stability_v9b）；M6 → 新 domain id 10（ndd_te_v9b）——独立于既有 9 域，各自专属 LoRA 组 + 头（V9 几何直接扩展 n_domains=11）。
- **S6 不入**（BLOCKED_ON_USER_TRANSFER；到位后另行增补）。

## 3. 训练协议（V9-1a 同款，仅数据侧变化）

- 底座：V8-S Stage 1（s_mrl-polya epoch2）+ V9 几何扩展（n_domains=11, num_cells=8）。
- **两臂设计**：
  - **臂 1（pure-v9b）**：只训 S1+M6 两个新域（与 s_mprau_in 专才同构——数据臂的"专才版"）。
  - **臂 2（bench+v9b）**：benchmark 9 域 + S1 + M6 全域均衡（11 域 DomainBalancedSampler）——主臂（统一模型主张的完整形态）。
- 超参：AdamW lr 2e-5 wd 1e-4 cosine、batch 128 域均衡、epochs 6、seed 20260907（单 seed 首轮；过门后加 2 seeds——V9-1a 已消耗 3 seeds 配额，本臂按"首轮单 seed + 过门扩 seed"模式省预算）、FINAL-EPOCH-6-FIXED。
- CUDA BF16；protected reads=0；产物 /mnt、代码 /home worktree + push。

## 4. 判定门（预注册，不改）

- **门 D1（弱域 on-manifold，主判据）**：
  - S1 域：y_SH Spearman（held-out？**无独立验证集——S1/M6 为全量训练数据，评估用留一交叉验证？不——为保持与 V8/V9 口径一致：训练后评估仅报训练域的 VALIDATION 结构不存在，故门 D1 改为**：MPRAU pair-mean vs V9-1a 基线（0.048-0.057）与 TE(200304) vs V9-1a 基线（~0.016）——**臂 2 的既有弱域是否被新数据带涨**（跨域迁移增益）+ 臂 1/2 在 benchmark 9 任务 VALIDATION 全表不劣化（MRL ≥0.28 / polyA ≥0.80 非破坏门沿用）。
- **门 D2（探针鉴别诊断，非门而是观测）**：臂 2 终态跑混合池探针（门①同口径）——结果只做鉴别诊断登记（H-d/H-a 分离），不作 pass/fail 门（避免用观测门翻案门① FAIL 的预注册语义）。
- **门 D3（新域自身可学性）**：臂 1 的 S1 y_SH 训练 Spearman（train-fit 报告，标注"非泛化证据"）+ 10% 留出验证 Spearman（**新增：S1/M6 各留 10% 作 VALIDATION_SPLIT，训练用 90%——修正 §2 的全量训练表述；留出集入库时冻结**）。
- 失败回退：门 D1 非破坏 FAIL → 调整混合比例（v9b:bench 采样权重 1:1 → 1:2）单臂重训一次；仍 FAIL → 数据臂结论如实入档（负结果）。

## 5. 留出集冻结（防泄漏条款）

- S1：4,540 对 → 随机分层留出 454 对（按 utr_group × assay 显著性分层）→ VALIDATION_SPLIT 标记入库时一次性冻结（seed 20260908）。
- M6：916 对 → 留出 92 对（按 family 分层）→ 同上。
- 训练只用 TRAIN_SPLIT 子集；门 D3 评估只用 VALIDATION_SPLIT。
- 留出集落盘：`analysis_track_b_20260908/{s1,m6}_holdout_manifest.json`（此文件为预注册一部分，先冻结后训练）。

## 6. 预注册时点状态

- V9-1a 2-seed 终态已知（门②部分过/门① FAIL）；seed 11 在途——本预注册不依赖 seed 11 结果（数据臂的判定参照 = 已终态的 2-seed 数字）。
- 写死条款：双臂设计、留出集冻结先行、n_domains=11/num_cells=8 几何、S1 双 assay cell 映射、M6 HEK 近似口径声明。
