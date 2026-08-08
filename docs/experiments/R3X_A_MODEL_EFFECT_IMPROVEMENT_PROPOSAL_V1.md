# R3X A 模型效果改进提案 V1

状态：`PROPOSED_NOT_ACTIVE`

数据边界：`ORDINARY_TRAIN_AND_DEVELOPMENT_ONLY`

确认性门槛：`macro_sign_accuracy >= 0.60`（保持不变）
禁止事项：不得读取 ordinary internal-test outcome；不得访问 sealed；不得用 test 后调 seed、阈值、loss 或架构。

## 1. 焦点问题

当前 A 通过共享序列编码器和全局均值池化定义
`delta_hat = phi(candidate, condition) - phi(source, condition)`。这保证 identity 为零和 swap 反对称，但在长序列上，1/3/5 个替换对均值池化表示的贡献可能随序列长度衰减。需要判断 sign 表现不足主要来自这种稀释，还是来自 endpoint/研究混杂、标签尺度、样本噪声或数据资格问题。

本文档只登记可证伪的候选方向，不把任何候选写成已验证改进。

## 2. 独立候选假设

### H-A0：当前全局 potential 已足够

- 类型：零假设/必要对照。
- 预测：在相同数据、信息集、split、seed 和训练预算下，局部增强不稳定或没有一致增益。
- 反证：局部增强在所有合格 development cell 中一致提高 component-balanced sign，并且不依赖 study/context proxy。

### H-A1：局部位点势能分支可恢复稀疏编辑信号

- 类型：模型机制提案。
- 结构：保留当前共享 full-length encoder；为每个可见位置计算同权重 site potential，并以 masked sum 或预冻结的长度归一化聚合到 `phi(sequence, condition)`；source 与 candidate 必须调用完全相同的 `phi`。
- 必须保持：identity mean 精确为零；swap mean 精确反对称；swap log-variance 精确对称；不输入 study ID；不读取 measured source outcome。
- 预测：增益随序列长度和编辑稀疏度增强；distal edit 不再比短序列 edit 系统性更弱。
- 反证：wrong-local/position-shuffle 对照不降，或增益仅出现在与 study 一一对应的 context/endpoint。

### H-A2：endpoint 量纲混合而非编码能力是主要误差源

- 类型：estimand/训练提案。
- 结构：每个冻结 primary estimand 独立训练；仅用 train 分区拟合 target scaler；development 预测反变换后评估。禁止跨 endpoint 共享 raw-delta loss。
- 预测：独立 estimand 训练显著改善 sign 校准并降低某一大 study 对梯度的主导。
- 反证：独立训练后 sign 仍接近随机，或提升来自少数 row-heavy component。

### H-A3：condition embedding 是 study proxy

- 类型：混杂诊断。
- 结构：比较 condition-enabled 与 condition-ablated/UNK-only A；另做 endpoint/context 到 study 的冻结确定性映射审计。
- 预测：若 condition-enabled 增益在 held study 上消失，而 UNK-only 更稳定，则原增益主要是 proxy。
- 决策：任何确定性 study proxy 未闭合前，只能写 development diagnostic，不能写跨 study 泛化。

## 3. 冻结实验顺序

在论文级强基线完成后，按以下顺序运行，不能根据 development 结果改变候选集合：

强基线候选集合至少包含同一 paired-sequence information set 下的
FullPair、paired local/global k-mer Ridge、nearest-neighbor，以及在冻结
Python/scikit-learn 环境审计通过后运行的 HistGradientBoosting。后者若只因
runner 默认 `execution_authorized=false` 而未运行，则“论文级强基线完成”不成立。
R2-style adaptation 在 architecture crosswalk 与 study-proxy 审计闭合前只作
development diagnostic，不参与 strongest-baseline 资格。

1. `A0_GLOBAL_MEAN_POTENTIAL`：当前 A，原样复现。
2. `A1_SITE_ADDITIVE_LOCAL_GLOBAL_POTENTIAL`：唯一架构增量。
3. `A1_WRONG_LOCAL_CONTROL`：在 train source-component 内错配局部分支输入，保持长度/碱基组成边际。
4. `A1_CONDITION_UNK_ONLY`：移除 endpoint/context shortcut 的诊断。
5. 只有 A1 在冻结 development 规则下优于 A0 和 strongest same-information-set baseline，才允许进入 B；否则停止扩展。

每个候选使用固定 seeds `[11, 23, 42, 67, 101]`，以五模型预测 ensemble 作为一个模型评估，不能对每个 metric 挑不同 seed。所有方法必须使用相同 global source-relation component split、相同 estimand/cell、相同 information-set fingerprint 和匹配的训练预算。

## 4. 预先冻结的评价与停止规则

- 主评价：component-within-cell balanced sign accuracy，硬符号规则为 `raw predicted delta mean >= 0`。
- 必要门槛：每个预注册 primary cell 都有定义且宏平均 `>= 0.60`；不能静默丢弃困难 cell。
- 次评价：within-ranking-pool Spearman、component-balanced MAE、Gaussian NLL/校准诊断。
- paired superiority：相对 development 选出的 strongest same-information-set baseline，按 global source-relation component 成簇；方向和 family 必须由冻结 comparison policy 指定。
- `A1 <= A0`、paired interval 跨零、wrong-local 约等于 A1、任一 cell undefined、proxy 审计失败、编辑不可见或输出非有限：`STOP_A1_NO_QUALIFIED_INCREMENT`。
- A1 development 合格只允许写 `DEVELOPMENT_CANDIDATE_FROZEN`，不等于 internal-test PASS、V2 GO 或发表成功。

## 5. 参数匹配与归因

若 A1 被实现，必须同时实现参数量差异不超过 5% 的 unrestricted pair control；不能拿容量更大的 A1 与旧 FullPair 直接归因于 exact potential。A1 只允许新增局部位点势能聚合；foundation embedding、domain adversarial、额外外部数据和 interaction branch 均不得在同一消融中加入。

## 6. 当前决策

`decision = NULL_PENDING_DEVELOPMENT_STRONG_BASELINE_AND_STAGED_DISCLOSURE`。

在 staged development authority、predictor provenance、strongest-baseline selector 和全部 train-only negative controls 闭合前，不实现 primary 替换，不运行真实 ordinary internal-test，也不把 H-A1 写成方法贡献。
