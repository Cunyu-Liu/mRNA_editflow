# R3X V6 ordinary-only 模型效果预注册提案

状态：`PROPOSED_NOT_ACTIVE`

当前结论：`SCIENTIFIC_HOLD / MODEL_EFFECT_NOT_ESTABLISHED / ORDINARY_NOT_RUN`

硬边界：

- 仅允许 ordinary train/development；
- `component/cell-balanced sign_accuracy >= 0.60` 保持不变；
- hard sign 始终是 `raw predicted delta mean >= 0`；
- 不访问 ordinary internal-test outcome，不访问 sealed，不解锁 O0/G1/E0；
- 不根据 development 结果增加 seed、阈值、loss、架构或删除困难 cell；
- C3 仍为 `PROPOSED_REDLINE_NOT_ACTIVE`，guidance 仅可描述为
  `DGM/Doob-derived adaptation`，不得声称 exact target sampler。

本文档登记可证伪的实验和停止规则，不是合同修订、训练授权或效果结果。

## 1. 当前证据与未知项

| 项目 | 当前证据状态 | 允许的表述 |
|---|---|---|
| A0 exact potential 的 identity/swap 代数性质 | 合成单元测试支持 | 工程/代数性质成立 |
| 旧 A1 global-context site-sum exact potential | 已实现、机制反例已证实非 bounded-local | 仅作 development diagnostic |
| A1 V2 bounded-local-conv/global exact potential | 已实现 finite-RF development 候选，未训练 | primary 方法候选，不是改进结果 |
| edit-aware FullPair V2 | 已实现 development 候选，未训练 | 更强 comparator 候选 |
| scalable Ridge / chunked exact NN / weighted HGB | 数值核与合成测试支持 | 规模风险已部分修复，真实规模仍未验证 |
| neural method→architecture registry V2 | exact-class/CPU-float32/实际参数检查支持 | 防止旧 method ID、reference class 或 dtype/device 被静默重映射；不是训练许可 |
| canonical information-set fingerprint | 冻结定义推导与交换反例支持 | 拒绝调用者任意自报同信息集；不是独立 data trust root |
| strongest baseline selector V3 | 合成 evidence replay 支持 | 可选 comparator；不代表 baseline 或 candidate 过 gate |
| ordinary sign learnability/noise ceiling | `NOT_RUN` | 不知道模型上限是否超过 0.60 |
| A0/A1/B 的真实 ordinary 效果 | `NOT_RUN` | 不得声称提升、领先或可发表 |

历史 V1 架构和新 V6 候选必须分开记录：

- `R3X_FULL_PAIR_UNRESTRICTED_SIAMESE_V1` 是 pooled historical control；
- `R3X_FULL_PAIR_EDIT_AWARE_UNRESTRICTED_SIAMESE_V2` 是新增候选，只有经
  versioned method/policy authority 纳入后才可进入 strongest claim；
- `R3X_A_PARAMETER_MATCHED_UNRESTRICTED_SIAMESE_V1` 是历史
  condition-invariant residual；
- `R3X_A1_SITE_ADDITIVE_LOCAL_GLOBAL_EXACT_POTENTIAL_V1` 的 site energy 输入全局
  Transformer hidden；单编辑可改变远端全部 site energy，因此只能称
  `GLOBAL_CONTEXT_SITE_SUM_DIAGNOSTIC`，不得称 bounded-local；
- `R3X_A1_BOUNDED_LOCAL_GLOBAL_EXACT_POTENTIAL_V2` 才是 H1 的 primary：其
  depthwise-separable local branch 的冻结 receptive-field radius 为 2，并保留
  `phi(candidate,z)-phi(source,z)`；完整 mean 仍含全局 Transformer potential，故准确
  表述是“带 radius-2 局部分支的 global exact-potential hybrid”，不能声称整个 critic
  或 uncertainty 是 finite-RF、远背景不变或 length-extension invariant；两个编辑的
  local receptive fields 在距离 4 时仍可能重叠，严格 local-interaction exclusion 只在
  整数距离至少 5 时成立；
- condition-aware V2 control 与 A1-matched control 是新归因候选，不能静默替换
  旧 artifact。

方法新颖性边界：把参考与突变序列分别评分后相减并不新颖。Optimus 5-Prime、
FramePool 与 Saluki 都已用 reference/alternative prediction difference 做 UTR variant
effect；因此本文不能把“delta of two sequence scores”或普通卷积局部特征写成创新。
潜在方法贡献只能来自：ordinary paired-effect benchmark、严格 source-component/cell
estimand、可检验的 finite-RF exact-potential 约束及其 matched-control 归因。相关 primary
prior art：Sample et al., 2019, DOI
[10.1038/s41587-019-0164-5](https://doi.org/10.1038/s41587-019-0164-5)；
Karollus et al., 2021, DOI
[10.1371/journal.pcbi.1008982](https://doi.org/10.1371/journal.pcbi.1008982)；
Agarwal & Kelley, 2022,
[Saluki code/data](https://github.com/vagarwal87/saluki_paper)。2024–2025 年后的
translation comparator 审计还必须覆盖
[UTR-LM](https://www.nature.com/articles/s42256-024-00823-9)、
[RiboNN](https://www.nature.com/articles/s41587-025-02712-x) 与公开的
[RNA gLM benchmark](https://www.nature.com/articles/s41467-025-66899-y)。这些模型的
原论文性能不能直接搬到本项目；只有在输入、estimand、license、预训练/微调 exposure 与
同一 split 全部兼容时，才可作为本项目 comparator。

## 2. 可证伪假设

### H0：标签的可重复方向上限不足以支撑 0.60

- 假设：近零效应、replicate 不一致或 estimand 尺度问题是主要瓶颈。
- 预测：独立 replicate/sign concordance、train-only noise ceiling或跨重复测量方向一致性
  接近或低于 0.60；简单 sign prior 已接近可达上限。
- 反证：replicate 方向一致性明显高于 0.60，且强基线仍显著低于该上限。
- 替代解释：某一大 cell 的类别先验或重复伪样本抬高表面一致性。
- 停止规则：若合格重复测量无法定义，标记 `NOISE_CEILING_UNIDENTIFIED`；不得用
  多数类比例冒充可学习精度上界。

### H1：旧表示把稀疏编辑信号稀释在全局均值中

- 假设：1/3/5 个编辑在长序列均值池化后贡献随长度衰减。
- 预测：edit-aware FullPair V2 和 bounded-local A1 V2 的增益在长序列、低 edit count strata 更明显；
  wrong-local/position-shuffle 会消除增益。
- 反证：新增局部分支在两个独立 primary cell 中均无稳定增益，或 wrong-local 与正确
  local 表现相同/更好。
- 替代解释：新增分支只学习 endpoint/context 对 study 的代理。
- 停止规则：任一 primary cell undefined、A1−A0 paired interval 跨零、wrong-local
  约等于 A1、或 UNK-only 暴露 proxy 时，停止扩展局部架构。

### H2：当前 probit sign loss 的尺度导致梯度饱和

- 假设：`Phi(mu / detached_sigma / temperature)` 在极端尺度下饱和，而 hard sign只由
  `mu` 决定。
- 唯一候选：与当前 loss 比较 `FIXED_TRAIN_SCALE_LOGIT_BCE(mu / tau_train)`；
  `tau_train` 只从 train target scale 冻结。
- 预测：固定尺度 logit 在相同五 seed、epoch 和预算下提高 development sign，且不恶化
  mean/NLL 到预注册容忍范围之外。
- 反证：五模型 ensemble 的 paired sign 增益不稳定或只由单一 cell 驱动。
- 停止规则：不得在 development 后再增加温度、loss weight 或第三个 sign loss 候选。

### H3：B-additive 比序列势函数更贴近真实稀疏编辑算子

- 假设：source-context hidden state 上的 ref/alt/position additive contribution 比 A1 更适合
  1/3/5-edit effect。
- 预测：B-additive 在 A continuation 合格后胜过 A1，并在 edit-position/token shuffle 后失效。
- 反证：B 不胜 A1/strongest comparator，或 shuffle control 不降低效果。
- 停止规则：A 未通过 0.60 与冻结 paired rule 时 B 固定 `NOT_RUN`；action-pair support
  未闭合时 interaction 固定 `NOT_RUN_ACTION_SUPPORT`。

## 3. 冻结执行顺序

### Phase 0：可学习性与数据诊断

只在 train/development authority 内计算：

1. 每个 `estimand × publication context × region × edit_count` cell 的正/负/零支持；
2. component-balanced majority prior 与 edit-only sign diagnostic；
3. replicate/sign concordance、`|delta|` 与可用 measurement noise；
4. sign accuracy 随 `|delta|`、sequence length、edit count 和 cell 的分层；
5. endpoint/context 到 study 的确定性 proxy 审计。

这些输出只解释可识别性，不能替代原 0.60 gate，也不能据此筛行。

### Phase 1：论文级强基线

合同 exact-eight 仍需全部运行并 exact replay。scalar fitted-state V2 已把 prior、Ridge 与
nearest-neighbor 统一修为 `formal cell → global source component → row` 加权，并把 grouped-CV
改为先生成每行唯一 OOF prediction、再只计算一次完整 macro risk；旧 V1 state 会 fail closed，
必须重新拟合。HGB 目前只有 synthetic-only numeric-tree refit/replay，且 sample weight 只均衡
loss mass；sklearn 的 quantile binning 与 `min_samples_leaf` 仍按原始 rows 工作，不能称
replication-invariant estimator。FullPair neural checkpoint 路线也未闭合，故 exact-eight
仍是 `NOT_COMPLETE`：

1. ZERO_DELTA；
2. SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR；
3. EDIT_ONLY_RIDGE；
4. CANDIDATE_KMER_RIDGE；
5. PAIRED_LOCAL_GLOBAL_KMER_RIDGE；
6. PAIRED_FEATURE_NEAREST_NEIGHBOR；
7. HISTOGRAM_GRADIENT_BOOSTING；
8. FULL_PAIR_UNRESTRICTED_SIAMESE（历史 pooled 版本，作为合同基线）。

论文级方法比较还必须额外运行 edit-aware FullPair V2；未完成 versioned policy binding 前，
它只能在独立 development leaderboard 中出现，不能被偷偷映射到历史方法 ID。

还必须做 endpoint-compatible external comparator audit：translation cell 至少审查
Optimus 5-Prime、FramePool、UTR-LM、RiboNN，以及 RNA gLM benchmark 中预先冻结的一项
可执行代表；half-life cell 至少审查 Saluki。只有 sequence/annotation 输入可合法重建、
license 可执行、训练数据 exposure 可判定且 cell estimand 匹配时才运行；否则逐项
写 `NOT_RUN_INPUT_OR_EXPOSURE_INCOMPATIBLE`。这些 endpoint-specific external models不能因
只覆盖一个 cell 就冒充统一 strongest comparator，但缺失时也不能声称 general SOTA。

selector V3 的 comparator 选择与 qualification 分离：

- comparator 始终按 development sign → Spearman → low MAE → frozen order 选择；
- baseline 自身是否达到 0.60 单独记录；
- baseline 未过 0.60 不阻止训练 A，但 recovery comparator 不能满足 pre-test freeze；
- A 仍必须独立达到 0.60 并在冻结 paired comparison 中胜 comparator；
- HGB 或其他 required evidence 缺失时不得写 strongest claim。

### Phase 2：Route A

固定候选，不得结果后扩表：

1. A0 global-mean exact potential；
2. 旧 global-context site-sum A1 V1，仅作 diagnostic；
3. primary A1 V2：bounded-local depthwise-separable convolution + global exact
   potential，local radius固定为 2，site normalization固定为
   `SUM_OVER_SQRT_VISIBLE_LENGTH`；
4. A1 wrong-local control；
5. A1 condition-UNK-only；
6. A0/A1 各自的 condition-aware、实际冻结配置下参数差不超过 5% unrestricted control；
7. 仅在上述架构冻结后比较当前 probit 与一个 fixed-train-scale logit loss。

每一项使用 seeds `[11, 23, 42, 67, 101]`，评价对象是五模型 prediction ensemble，
不是五个 metric 的逐项平均，也不能挑最好 seed。

### Phase 3：Route B

只有 `A_CONTINUATION_DECISION=GO` 才运行 B-additive。B-interaction 继续 NOT_RUN，直到
per-action-pair factorial support、跨 component/cell 重复和 dominance gate 全部预注册并通过。

## 4. 评价、归因与停止规则

- primary：所有注册 cell 均定义的 component-within-cell balanced sign accuracy；宏平均
  必须 `>=0.60`；
- secondary：within-pool Spearman、component-balanced MAE、NLL/calibration；
- 分层：sequence length × edit count，禁止用大量 1-edit row 掩盖 3/5-edit；
- paired inference：global source-relation component 成簇 bootstrap/permutation；
- strongest win：candidate 对冻结 comparator 的 oriented improvement、CI 下界、Holm p 均合格；
- mechanism attribution：A1 必须胜 A0，且 exact A1 相对 condition-aware unrestricted control
  的结果满足预冻结方向；wrong-local 与 UNK-only 不得复制主增益；
- 任何 test 后调参、cell 丢失、非有限输出、编辑不可见、authority/hash 不一致或信息集不等：
  `STOP_INVALID_DEVELOPMENT_RUN`；
- development 合格最多得到 `DEVELOPMENT_CANDIDATE_FROZEN`，不是 V2 GO、sealed GO、
  SOTA 或期刊发表资格。

## 5. 当前决策

`decision = HOLD_PENDING_TRUSTED_DEVELOPMENT_EXECUTION_AND_REPLAYABLE_EXACT_EIGHT`

当前 development run-plan 只达到可重放 scaffold，固定
`training_ready=false/execution_authorized=false`；它不是训练许可证。已允许继续实现
可重放 baseline state、A/A-control decision 和 B-additive conditional runner；当前没有授权读取
internal-test outcome、访问 sealed、激活 C3 或降低 0.60 gate。
