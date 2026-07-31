# mRNA-EditFlow UTR 生成式重构最终 Goal

## source-conditioned、region-aware、grammar-constrained continuous-time Edit Flow

> - **文档性质：**科研问题合同、模型架构合同、数据与评测合同、GPU 实验执行合同、阶段任务清单
> - **适用对象：**Codex / Trae / 代码智能体 / 数据工程人员 / 模型训练人员 / 论文撰写人员
> - **目标仓库：**`/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
> - **本地仓库标识：**`Cunyu-Liu/mRNA_editflow`
> - **合同版本：**`utr_editflow_goal_v2`
> - **制定日期：**2026-07-28
> - **当前范围：**5′UTR 与 3′UTR；全长 mRNA 和 CDS 延后到架构成熟之后
> - **湿实验边界：**当前阶段不开展任何新增湿实验
> - **核心方法：**mRNA-EditFlow 必须是主方法，不得降级为可选附件
> - **基础模型策略：**优先复用现有 RNA/mRNA foundation model；从零训练仅作为必要消融
> - **执行原则：**问题先于数据、证据先于宣传、训练必须使用 GPU、失败证据不可删除、门槛不可事后降低

---

# 0. 文档权威性、读取义务与变更控制

## 0.1 本文档解决什么问题

旧 Goal 的主轴是：

```text
mRNA-EditBench
→ SparseEditFormer
→ 局部干预效应预测
→ Flow 作为可选扩展
```

这一主轴虽然在无新增湿实验条件下较保守，但偏离了项目名称、原始架构和已经确认的研究意图。mRNA-EditFlow 不是一个普通变异效应预测器项目；它必须研究一个生成式、连续时间、受生物语法约束的编辑过程。

本文档正式纠正为：

```text
Foundation model / sequence prior
        ↓
实验效应模型、critic 与评测基础设施
        ↓
mRNA-EditFlow：核心生成模型
        ↓
source-conditioned 合法编辑轨迹与候选分布
        ↓
匹配预算的生成、搜索与外部证据评测
```

效应预测、数据 benchmark 和 foundation model 都是 mRNA-EditFlow 的支撑系统，不得再次取代生成模型成为项目本体。

## 0.2 权威顺序

后续仓库应建立以下权威顺序：

1. 本 Goal 文档所定义的科学思想和不可变边界；
2. `configs/utr_editflow_execution_policy.yaml`；
3. `docs/utr_editflow_scientific_question.md`；
4. `docs/utr_editflow_claim_matrix.md`；
5. `docs/execution/task_registry.yaml`；
6. 单次实验的 frozen config、manifest 和 run artifacts。

如果低层文件与高层文件冲突，必须 fail closed，停止相关任务并修复合同，不得由训练代码自行选择更方便的解释。

## 0.3 每个执行者的读取义务

开始任何数据、模型、训练、评测或论文任务前，执行者必须：

1. 完整阅读本 Goal；
2. 记录本 Goal 的版本和 SHA256；
3. 指明当前所属 Phase、Task ID 和依赖 Gate；
4. 写出本任务如何服务核心科学问题；
5. 明确本任务不允许做什么；
6. 检查仓库 HEAD、工作区状态、现有进程、GPU 和磁盘；
7. 创建或更新 task registry；
8. 只在依赖 Gate 已通过时进入执行。

所有训练 manifest 必须包含：

```yaml
goal_contract:
  id: utr_editflow_goal_v2
  sha256: <actual_sha256>
scientific_question_id: RQ-UTR-EDITFLOW-V2
phase_id:
task_id:
git_commit:
data_manifest_sha256:
split_manifest_sha256:
foundation_checkpoint:
foundation_checkpoint_sha256:
exposure_ledger_version:
```

缺少这些字段的训练可以作为开发 smoke，但不得作为论文证据。

## 0.4 合同变更

以下内容只有在用户明确讨论并确认后才能修改：

- 核心科学问题；
- Edit Flow 的主方法地位；
- UTR-only 的当前范围；
- 无新增湿实验边界；
- 训练必须使用 GPU；
- GSE246381 的历史暴露状态；
- final-label 使用规则；
- primary metrics 和最终 Gate；
- 允许与禁止的论文主张。

合同修改必须新增 decision log，记录：

```yaml
decision_id:
date:
old_text:
new_text:
reason:
evidence:
affected_tasks:
requires_rerun:
approved_by_user:
```

不得静默重写历史，也不得让旧结果看起来天然符合新合同。

---

# 1. 已确认的决策记录

## 1.1 生成式方向不可退让

已经确认：

> mRNA-EditFlow 必须使用 Edit Flow 框架，并充分发挥其架构优势；Flow 不是在 scorer 失败后才考虑的附件，也不是仅用于生成“预测器喜欢的序列”。

生成模型的独立科学价值包括：

- 学习合法序列与编辑轨迹的条件分布；
- 从同一个 source 生成多个不同而合理的候选；
- 建模插入、删除、替换和 STOP；
- 支持变长生成、infilling 和 refinement；
- 学习多步编辑顺序；
- 在每一步实施硬约束；
- 将重复搜索摊销为训练后的快速候选生成；
- 在不同 region、assay、context 和目标条件下实施可控生成。

预测器被误导是评测风险，不是生成模型没有意义的理由。真正禁止的是：

```text
同一个 predictor
既作为唯一 reward
又作为唯一 selector
又作为唯一 final evaluator
```

## 1.2 当前只做 UTR

第一阶段只做：

```text
5′UTR
3′UTR
```

暂不把以下内容纳入第一阶段核心主张：

```text
CDS synonymous generation
protein-conditioned codon flow
5′UTR–CDS–3′UTR joint optimization
full-length therapeutic mRNA generation
cross-region full-transcript synergy
```

当 UTR Edit Flow 的数学语义、训练稳定性、生成质量、条件控制和外部评测全部成熟后，再通过新合同泛化到 CDS 和全长 mRNA。

当前可使用“region-aware UTR Edit Flow”或“heterogeneous regulatory-region Edit Flow”的表述；在 CDS 未验证前，不得声称完整验证了“UTR nucleotide flow + CDS codon flow”的全长异质语法。

## 1.3 数据顺序

已经确认：

> 先冻结科学问题和可证伪假设，再判断什么数据有资格验证它；不得因为当前某个数据最容易获取，就把科学问题改写成那个数据最容易支持的形式。

数据不足时允许：

- 继续系统检索；
- 改善数据重建；
- 增加无标签或弱标签预训练；
- 使用明确标级的替代证据；
- 缩小某条具体 claim；
- 将未通过假设保留为负结果；
- 进入下一条预注册的前进路线。

数据不足时不允许：

- 把单点预测包装成生成式成功；
- 把人工构造邻接称为真实生物编辑轨迹；
- 把预测改善称为实测改善；
- 降低或删除失败 Gate；
- 使用 final/test labels 反向选择数据和模型。

## 1.4 GSE246381 保留但降级

GSE246381 保留，但由于其标签已被旧流程读取并用于既有评测，不得再称为：

```text
sealed
untouched
never-seen external test
```

正式状态为：

```text
historically_exposed = true
role = historically_exposed_retrospective_external_stress_test
labels_allowed_for_new_training = false
labels_allowed_for_new_hyperparameter_selection = false
```

保留的价值：

- 仍然来自独立研究；
- 仍能测试 study/context shift；
- 在新合同锁定后仍可作为 retrospective external evaluation；
- 可以进行错误分析和与历史结果对照。

降级的后果：

- 不能支持 untouched external generalization；
- 不能作为唯一最终证据；
- 必须报告历史暴露路径；
- 论文需要其他严格 held-out split、leave-one-study-out 或新的未暴露外部数据补足。

如果不降级，会把已经发生的标签暴露错误描述成独立验证，构成不可接受的证据夸大。

## 1.5 62 个 ENCODE raw 文件

62 个 ENCODE raw 文件正在下载，下载任务继续，不构成当前讨论阻断。

在完成审计前，它们的预定位是：

```text
candidate_role = unlabeled_or_observational_pretraining
not_primary_intervention_evidence = true
```

它们可以增强：

- UTR 序列先验；
- foundation-model adaptation；
- region/context representation；
- generative denoising 或 corruption pretraining。

它们不能自动提供：

- WT→mutant 因果标签；
- 多步编辑真实轨迹；
- prospective 功能改善；
- final independent oracle。

## 1.6 无新增湿实验

当前合同下：

```text
new_wetlab = forbidden
```

只有当计算结果达到预注册的 exceptional-computational gate 后，才可以另行讨论新的湿实验合同。该讨论不属于当前 Phase，也不得把“未来可能做”写成“当前已有证据”。

## 1.7 Foundation model 优先

主路线优先使用公开可获得的现有模型：

- mRNABERT；
- UTR-LM；
- 3UTRBERT；
- Orthrus；
- 必要时 RNA-FM、RiNALMo、mRNA-LM；
- 任务原生模型如 Optimus 5-Prime、FramePool、Saluki、APARENT2 作为基线或辅助。

默认顺序：

```text
frozen embedding/cache
→ adapter/LoRA
→ last-block partial unfreeze
→ full fine-tuning only if justified
```

从零训练只用于：

- 小模型控制组；
- 架构消融；
- 验证 foundation model 是否真正贡献；
- 许可证或输入不兼容时的明确替代。

不得先验假定 foundation model 一定获胜，所有优势必须在相同 split 和公平预算下验证。

---

# 2. 唯一核心科学问题

## 2.1 Primary Research Question

> **给定一个既有 UTR source、region、assay/context、功能 endpoint 和目标条件，source-conditioned、region-aware、grammar-constrained continuous-time mRNA-EditFlow 能否学习可迁移的合法编辑轨迹分布，并生成多样、稀疏、变长且可控的 5′UTR/3′UTR 候选？**

这个问题研究的不是“能否给一个现成候选打分”，而是：

```text
一个高功能 UTR candidate
是否应当被建模为
从 source 出发、沿合法 edit actions 演化得到的生成结果
```

## 2.2 Comparative Method Question

> **在匹配训练数据、backbone、可训练参数、GPU 预算、candidate budget、oracle-query budget 和约束条件下，mRNA-EditFlow 能否相对 autoregressive generation、masked/discrete diffusion、generic Edit Flow、direct scorer + search 获得更好的“功能控制—合法性—多样性—编辑成本—生成效率”Pareto frontier？**

## 2.3 为什么这个问题有价值

普通 absolute-property predictor 只能回答：

```text
这个完整序列的预测值是多少？
```

而 UTR Edit Flow 研究：

```text
从哪个 source 出发
在什么 context 下
通过哪些合法动作
以什么顺序
用多少次编辑
可以生成怎样的一组候选分布
```

它与纯 GPT/autoregressive de novo generation 的区别是保留 source 身份与局部可审计编辑；与普通搜索的区别是学习可复用的 amortized proposal distribution；与 masked/diffusion 的区别是插入、删除、替换和 STOP 是显式连续时间事件，而不是固定长度 token 修复的后处理。

## 2.4 关键概念边界

本文中的“edit trajectory”是模型的潜变量和算法生成路径，不等同于：

- 细胞中真实发生的 RNA editing 生化过程；
- 进化历史；
- 实验逐步实施的编辑轨迹；
- 已被逐步测量的因果路径。

如果公共数据只有 source 和 final candidate 两端，模型中间路径只能称为：

```text
latent algorithmic edit trajectory
```

不得称为 observed biological trajectory。

## 2.5 当前最大的科学不确定性

当前最没有把握、也最需要由 D0–MB0 回答的不是“生成模型是否有意义”，而是：

1. 公开数据是否包含足够多的 measured multi-edit、indel、variable-length 和同 source 多候选景观；
2. true continuous-time Edit Flow 是否在匹配预算下优于强 scorer/search/AR/diffusion；
3. foundation model、实验库 proposal bias 和 critic 是否会掩盖真正的 Flow 贡献；
4. 5′UTR 与 3′UTR 的共享规律是否足够强，还是主要需要 region-specific adapters；
5. 无新增湿实验时，open-support generation 可以获得多强的可信计算证据。

这些是不确定、可证伪的问题，不能在合同中预写答案。

---

# 3. 可证伪假设

## H1：Edit-process modeling

在相同 backbone、数据和预算下，显式连续时间 edit-rate field 比以下方法更好地建模 held-out source→candidate 分布：

- candidate-only absolute model；
- source/candidate subtraction；
- Siamese difference；
- autoregressive action model；
- masked/discrete diffusion；
- generic unconstrained Edit Flow。

H1 的证据必须包含 held-out generative likelihood/transition reconstruction、candidate recovery、calibration 和多 seed 统计，而不能只看训练 loss。

## H2：Edit Flow 架构不可替代性

以下组件应分别产生可测量贡献：

- source conditioning；
- continuous time；
- insertion rate；
- deletion rate；
- substitution rate；
- STOP；
- variable-length state；
- multi-step trajectory；
- region-conditioned rate field；
- legal action mask；
- edit-budget state；
- target property/context condition。

每个组件必须有消融。若只实现迭代 greedy/top-k substitution，不得把模型称为完整 Edit Flow。

## H3：Hard-constrained validity

在所有生成步骤和最终样本上：

```text
invalid nucleotide = 0
forbidden-position edit = 0
anchor violation = 0
budget violation = 0
length-bound violation = 0
identity edit counted as edit = 0
```

硬约束有效率必须是 100%。软 penalty、生成后修复或删除非法样本不能替代构造性合法动作空间。

## H4：Conditional controllability

在相同 source 上改变 region、assay/context、endpoint 或目标方向时，生成分布应产生可解释且可重复的变化。

必须评估：

- target-direction success；
- condition consistency；
- condition sensitivity；
- condition permutation negative control；
- target strength monotonicity；
- identical-condition reproducibility；
- diversity under fixed condition。

## H5：Generative advantage over search

在匹配 candidate 数、oracle-query 数、wall-clock、GPU-hours 和约束的条件下，Edit Flow 应至少在下列维度形成不被强搜索全面支配的 Pareto frontier：

- high-effect measured-candidate recovery；
- independent-critic score；
- diversity；
- edit cost；
- inference latency；
- oracle-query efficiency。

必须比较 random legal、greedy、beam、best-of-N、simulated annealing、direct scorer exhaustive ranking（可行时）和其他强搜索。

## H6：Cross-source and cross-study transfer

在 source-disjoint、gene-disjoint、study-disjoint、context-disjoint 和 exposure-aware external evaluation 中，Edit Flow 的生成规律应具有迁移性。

不能只报告随机 pair split。

## H7：Foundation-model value

现有 foundation model 应提高表示、sample efficiency 或跨研究泛化；但必须通过：

```text
small from-scratch control
vs frozen foundation
vs adapter/LoRA
vs partial/full fine-tune
```

进行验证。

## H8：5′UTR 与 3′UTR 的统一与差异

两个区域共享：

- source-conditioned edit process；
- insertion/deletion/substitution/STOP 语义；
- continuous-time parameterization；
- 硬约束接口；
- 生成和评测协议。

两个区域保持独立：

- endpoint heads；
- assay/context metadata；
- motif and anchor rules；
- length priors；
- region-specific rate-field adapters；
- 数据 normalization。

不得把 5′UTR MRL、3′UTR abundance、half-life 或其他 endpoint 直接合并为一个统一 expression label。

---

# 4. 项目范围与非范围

## 4.1 In scope

- 5′UTR source-conditioned generation；
- 3′UTR source-conditioned generation；
- substitution、insertion、deletion、STOP；
- variable-length UTR；
- single-step 与 multi-step editing；
- 局部 infilling；
- source-preserving refinement；
- target-conditioned diverse generation；
- foundation-model representation/prior；
- 实验效应 predictor/critic；
- measured-candidate recovery；
- 跨 source、gene、study、context 迁移；
- 不确定性、abstention 和失败分析；
- 匹配预算的生成与搜索比较；
- 公开数据 benchmark 与完整 provenance。

## 4.2 Out of scope for the current contract

- CDS codon flow；
- protein-conditioned synonymous generation；
- 完整全长 mRNA；
- 治疗性 mRNA efficacy；
- 临床或监管有效性；
- 任何新增湿实验；
- “真实提高蛋白产量”的 prospective claim；
- 未测候选的真实生物最优性；
- 把模型轨迹解释为真实分子轨迹；
- 把 attention map 单独解释为生物机制。

## 4.3 Future full-length gate

只有以下条件全部满足后，才能提出 full-length v3 合同：

1. 5′UTR 与 3′UTR 的 true Edit Flow 语义通过；
2. ins/del/sub/STOP 和 variable length 均有非 smoke 证据；
3. H1–H6 中没有 Critical blocker；
4. 生成模型在强基线下不被全面支配；
5. 数据、foundation exposure 和 final evaluator 角色可审计；
6. 至少一个 region 的跨研究生成结果具有统计支持；
7. 用户明确批准扩展。

---

# 5. mRNA-EditFlow 必须充分利用的架构优势

## 5.1 Continuous-time edit process

模型必须输出非负事件率：

\[
\lambda_i^{ins}(t),\quad
\lambda_i^{sub}(t),\quad
\lambda_i^{del}(t),\quad
\lambda^{stop}(t)
\]

以及相应 token/action 分布。训练和采样必须保留连续时间变量 \(t\)，并验证 rate semantics。

必须明确：

- 是使用精确 Gillespie、tau-leaping 还是离散近似；
- 近似误差如何控制；
- 同一步多事件冲突如何处理；
- rate clipping 和数值稳定规则；
- 采样步数与质量/成本关系。

只调用一个名为 `flow` 的模块但实际执行 greedy candidate ranking，不算完成。

## 5.2 Source conditioning

所有核心生成任务必须有明确 source：

```text
p(candidate, trajectory | source, region, context, target, constraints)
```

source 未编辑位置的保持率必须报告。模型不得通过重新生成整条序列规避 minimal-edit 约束。

source \(x_0\) 可以缓存，但当前状态 \(x_t\) 在每次 edit 后必须被正确更新。允许使用增量编码、局部更新或 source-cache + current-state adapter 提高效率；不允许只看初始 source position hidden state、忽略 indel 后的动态坐标和上下文变化。

## 5.3 Variable-length first-class support

插入和删除必须是训练、采样和评测中的一等动作，而不是：

- 只在数据增强中出现；
- 只在单元测试中出现；
- 生成后人工拼接；
- supplementary demo；
- 永远被 action mask 禁用。

如果某一真实功能数据集只有 substitution，可使用无标签/弱标签 UTR 语料训练 variable-length prior，但必须区分：

```text
generative capability evidence
functional intervention evidence
```

## 5.4 Multi-step trajectories

至少支持：

```text
edit budget k ∈ {1, 3, 5}
```

更大预算是否启用由 validation 决定。

必须评估：

- 编辑顺序；
- STOP calibration；
- 循环编辑；
- 逆向编辑；
- 同一位置反复修改；
- 预算使用率；
- 每一步合法性；
- 最终与 source 的真实 edit distance。

## 5.5 Hard action masks

合法动作空间至少包含：

- region boundary；
- 固定/保护位置；
- anchor motifs；
- 允许 alphabet；
- 最大/最小长度；
- edit budget；
- 禁止的 identity action；
- 可选 motif-preservation rules；
- source-relative state tracking。

硬约束必须作用于 rate normalization 之前。不得先给非法动作概率再依赖拒绝采样掩盖问题。

## 5.6 Region-aware rate fields

5′UTR 与 3′UTR 共享主框架，但必须存在显式 region conditioning：

```text
shared Edit Flow trunk
        ├── 5′UTR adapter / rate modulation
        └── 3′UTR adapter / rate modulation
```

必须比较：

- 完全共享；
- 共享 trunk + region adapter；
- 两个独立模型；
- 错误 region label negative control。

## 5.7 Conditional generation

条件可以包括：

- endpoint；
- assay；
- cell/context；
- 目标方向；
- 目标区间或 quantile；
- 最大 edit budget；
- 长度目标；
- 必须保留的 motif。

实现可选择 classifier-free guidance、critic guidance、conditional FiLM、adapter 或其他方法，但必须报告训练条件与采样条件是否一致。

## 5.8 Diversity and amortization

同一个 source/condition 必须能够产生多个候选。必须报告：

- unique candidate rate；
- pairwise edit distance；
- motif/structure diversity；
- mode collapse；
- duplicate rate；
- candidates per second；
- 每个 source 的 amortized cost。

只返回单一 argmax 不能充分证明生成模型价值。

---

# 6. 总体系统架构

## 6.1 四层系统

```text
Layer A — Foundation representation / sequence prior
    mRNABERT / UTR-LM / 3UTRBERT / Orthrus / alternatives

Layer B — Experimental effect system
    paired-delta model
    endpoint-specific heads
    uncertainty
    independent critic(s)

Layer C — mRNA-EditFlow
    continuous-time legal edit rate field
    source conditioning
    region conditioning
    target conditioning
    variable-length multi-step sampling

Layer D — Evaluation and selection
    measured candidate recovery
    independent critic
    external retrospective data
    matched-budget baselines
    calibration and failure analysis
```

## 6.2 Foundation model 不是 Edit Flow 的替代

foundation model 可作为：

- source encoder；
- candidate encoder；
- local token representation；
- 序列 prior；
- 初始化参数；
- 条件 embedding。

但必须由 Edit Flow rate heads 和合法状态/action space 完成生成。不得把直接调用一个 GPT 生成器改名为 Edit Flow。

## 6.3 Effect predictor 的正确角色

效应模型可以承担：

- supervised representation shaping；
- guidance；
- critic；
- reranking；
- uncertainty；
- measured-pair baseline。

不能承担全部角色。至少划分：

```text
Teacher / guidance model
Selection model
Final evaluator
```

当数据有限无法完全独立时，必须报告共享的数据、权重、特征和潜在相关误差，不能使用“independent”一词。

## 6.4 Predictor 与 generator 的联合训练

允许的路线包括：

- 先训练效应模型，再冻结并指导 Flow；
- 多任务共享 foundation encoder，但 heads 和 final evaluator 隔离；
- 无标签 generative pretraining，再进行 effect-conditioned fine-tuning；
- offline preference / pairwise ranking warm start；
- classifier-free conditional flow；
- 受控 RL 或 policy improvement。

任何 RL 都必须晚于：

- true Flow sampler 可运行；
- reward/evaluator 角色已固定；
- 强 search baseline 已建立；
- reward-hacking audit 已建立。

不得把 RL 作为生成性的同义词。

---

# 7. Claim ladder 与允许主张

## 7.1 Evidence grades

```text
E0 — engineering
    unit tests, shape tests, smoke, synthetic sanity

E1 — internal computational
    train/validation, proxy reward, development split

E2 — retrospective measured
    held-out measured source/candidate labels

E3 — cross-study / context
    study-disjoint or context-disjoint measured evidence

E4 — historically exposed external
    independent study but prior label exposure recorded

E5 — untouched external
    genuinely unexposed and frozen before access

E6 — prospective experimental
    new wet-lab measurement
```

当前合同最高预期证据等级为 E3–E5；GSE246381 固定为 E4。E6 不在当前范围。

## 7.2 允许的 primary claim

只有通过对应 Gate 后，才允许主张：

> We introduce a source-conditioned, region-aware and grammar-constrained continuous-time Edit Flow that generates diverse and biologically legal minimal edits for 5′ and 3′ UTRs, and evaluate its controllability and transfer under matched-budget generative and search baselines.

中文边界：

> 我们提出一个以 source 为锚、以合法编辑动作为基本事件、支持变长和多步生成的 UTR Edit Flow，并在公开测量数据和严格计算协议下评估其生成、控制和迁移能力。

## 7.3 条件性 secondary claims

可在证据支持时主张：

- continuous-time edit-process modeling 优于候选独立建模；
- explicit legal action geometry 提高 100% 约束有效性；
- variable-length flow 提高 infilling/refinement 能力；
- Edit Flow 相对强搜索提高 candidate/query efficiency；
- foundation-model initialization 提高 sample efficiency 或迁移；
- region-conditioned adapters 优于完全共享或完全独立模型；
- uncertainty/abstention 降低 false-beneficial selection；
- 公开 UTR intervention 数据可以支持生成式编辑评测。

## 7.4 永久禁止的当前主张

- 生成候选已经提高真实治疗性 mRNA 效果；
- 未测候选具有实验验证的功能改善；
- MRL 等于 protein output；
- TE 等于 protein output；
- half-life 提升必然带来 protein output 提升；
- 预测器高分等于真实生物最优；
- 模型轨迹是实验观测或真实生化轨迹；
- GSE246381 是 untouched sealed test；
- 当前已经完成 full-length mRNA optimization；
- 当前已经验证 CDS grammar；
- 只凭 attention heatmap 建立机制性结论；
- 用同一 predictor 自我指导、自我选择和自我证明；
- 因为采用 Edit Flow 就天然优于 GPT、diffusion 或 search。

## 7.5 UTR-only 的工作量与可发表性判断

5′UTR 与 3′UTR 足以构成一篇完整工作，前提是完成：

- 真正的生成任务，而不是两个单点预测任务；
- variable-length 和 multi-step；
- 5′/3′ region-aware 比较；
- foundation 与 task-native baselines；
- matched-budget generator/search；
- 跨 study/context；
- measured-support 与 open-support 分轨；
- 不确定性、失败和复现；
- 完整 benchmark/provenance。

增加 CDS 或 full-length 并不会自动提高论文质量，反而可能让当前证据失焦。第一篇先把 UTR Edit Flow 做深是明确决策。

没有新增湿实验不等于生成式工作没有发表价值，但会限制：

- 未测候选的生物功能 claim；
- 治疗性/转化 claim；
- 最高层级生物验证强度。

投稿层级取决于真实的生成优势、外部泛化、方法不可替代性和复现质量，不能在结果产生前保证具体期刊。

---

# 8. 问题驱动的数据资格合同

## 8.1 数据不是按“能下载”分类，而是按“能验证哪条假设”分类

每个数据集进入项目前必须回答：

1. 支持哪个 H1–H8？
2. 监督对象是 sequence、pair、edge、family 还是 measured landscape？
3. 是否包含 source？
4. 是否包含 insertion/deletion/multi-edit？
5. 是否支持 variable length？
6. intermediate trajectory 是否真实观测？
7. endpoint 是什么？
8. assay/context 是什么？
9. 训练、选择、final evaluator 中承担什么角色？
10. 是否存在 foundation-model sequence/label exposure？
11. 许可证是否允许使用和发布？
12. 如果缺字段，claim 应如何降级？

没有明确答案的数据不得直接进入训练。

## 8.2 数据功能等级

### D-A：Unlabeled / observational UTR corpus

用途：

- foundation representation；
- generative sequence prior；
- denoising/corruption coupling；
- variable-length syntax；
- region discrimination。

不支持功能改善 claim。

ENCODE 62 个 raw 文件在完成审计前属于此候选等级。

### D-B：Absolute-property measured sequences

用途：

- conditional property representation；
- auxiliary predictor；
- property-stratified generation；
- 生成分布与实验景观对照。

如果没有可靠 source pairing，不得称 intervention。

### D-C：True source–candidate intervention pairs

要求：

- source/candidate sequence；
- 明确 edit；
- 相同 assay/context；
- 分别测量或可靠 paired effect；
- label provenance；
- replicate/QC 信息。

用途：

- paired delta；
- action-conditioned supervision；
- held-out edit recovery；
- 条件生成评测。

### D-D：Dense measured landscape

要求多个相邻或组合序列均被实际测量。

用途：

- measured graph；
- 局部和多步路径构造；
- candidate recovery；
- measured-space optimization；
- 生成与搜索的受控比较。

人工构造的图 edge 是计算邻接，不等同于实验按该顺序进行了编辑。

### D-E：Measured multi-edit / indel landscape

这是验证 Flow 不可替代性的最高优先数据类型。优先寻找：

- 多位点组合库；
- variable-length UTR library；
- insertion/deletion MPRA；
- 同 source 多候选 family；
- 具有真实测量的 edit-distance > 1 候选。

如果最终没有 D-E 数据：

- 仍可训练和研究 Edit Flow；
- 必须用 D-A/D-D 训练 generative capability；
- 用 D-C 验证单步功能 grounding；
- 不得声称已用真实多步干预数据验证多步功能协同。

### D-F：External evidence

进一步标记：

```text
untouched
historically_exposed
study_disjoint
context_disjoint
sequence_exposure_unknown
label_exposure_unknown
```

## 8.3 当前候选数据不等于最终数据合同

当前已知候选包括：

- GSE145046 dense 5′UTR landscape；
- GSE114002 / Sample 2019；
- GSE149487 / PLUMAGE；
- GSE246381；
- GSE217518；
- GSE200304；
- MPRAu / ENCSR854RUF；
- 其他待系统检索的 variable-length、multi-edit、indel 和 UTR design 数据。

这些名称只构成 discovery inventory。必须完成 eligibility matrix 后，才能冻结其角色。

## 8.4 数据保留与“降级”

数据降级不是删除数据，而是降低其可支持的证据等级。

降级必须保留：

- raw files；
- processed files；
- provenance；
- 历史使用记录；
- 为什么降级；
- 仍允许的用途；
- 禁止的用途。

禁止通过删除不理想数据来“提高”模型表现。

---

# 9. 数据 Schema、清洗与 provenance

## 9.1 Canonical record

```yaml
record_id:
dataset_id:
study_id:
assay_id:
context_id:
evidence_grade:
exposure_grade:

region: five_utr | three_utr
organism:
cell_context:
reporter:
cargo:
endpoint:
timepoint:

source_id:
source_sequence:
candidate_sequence:
source_length:
candidate_length:

edit_script:
edit_types:
edit_positions:
reference_alleles:
alternate_alleles:
edit_count:
edit_distance:

source_value_raw:
candidate_value_raw:
delta_raw:
delta_normalized:
effect_standard_error:
replicate_count:

pair_type:
trajectory_observed: false
trajectory_source: latent | constructed | observed

paper_split:
canonical_split:
source_group:
gene_group:
study_group:
context_group:
sequence_cluster:

sequence_provenance:
label_provenance:
download_manifest:
license:
quality_flags:
historical_exposure:
```

## 9.2 Pair types

允许：

```text
true_wt_mutant
dense_measured_neighbor
measured_multi_edit_family
measured_indel_pair
absolute_property_only
unlabeled_pretraining
retrospective_constructed_neighbor
```

`retrospective_constructed_neighbor` 永远不能静默升级为 `true_wt_mutant`。

## 9.3 两套清洗

```text
paper_clean
    复现论文可识别的过滤和标签

canonical
    转换到本项目统一 schema 和 QC
```

两者必须同时保留，不能用 canonical 结果冒充原文复现。

## 9.4 基础 QC

至少执行：

- checksum；
- 压缩包完整性；
- 防 HTML error page；
- sequence uppercase；
- DNA/RNA alphabet normalization；
- IUPAC audit；
- strand validation；
- source/candidate alignment；
- edit script canonicalization；
- region/length validation；
- duplicate/collision；
- barcode/construct mapping；
- replicate consistency；
- low-count uncertainty；
- endpoint transformation audit；
- license audit。

## 9.5 Label 与 endpoint

以下 endpoint 永久分开：

```text
MRL
translation efficiency
RNA abundance
half-life
decay rate
protein abundance
other assay-specific endpoint
```

每个 endpoint 使用独立 head 或明确的 hierarchical distribution。不得创建无生物意义的统一“expression score”作为 primary label。

## 9.6 Rejected table

任何被排除记录必须进入 rejected table，并有稳定 reason code，例如：

```text
VARIANT_SEQUENCE_MISMATCH
SOURCE_UNRECOVERABLE
ENDPOINT_AMBIGUOUS
LOW_COUNT
REPLICATE_FAILURE
LICENSE_BLOCKED
SEQUENCE_COLLISION
EXPOSURE_UNKNOWN
NOT_TRUE_INTERVENTION
```

不得静默删除。

---

# 10. Split、暴露与防泄漏

## 10.1 禁止随机 pair split 作为 headline

正式评测至少包含：

- source-disjoint；
- sequence-cluster-disjoint；
- gene-disjoint（可定义时）；
- study-disjoint；
- context-disjoint；
- region transfer；
- exposure-aware external。

## 10.2 Dense graph leakage

必须防止：

- 同一 node 跨 train/test；
- reverse edge 跨 split；
- 近邻序列跨 split；
- 相同 scaffold/construct family 泄漏；
- 根据 final label 构造 train/test；
- 从 test high performers 反向生成训练 coupling。

## 10.3 Foundation exposure ledger

每个 foundation model 必须记录：

```yaml
model:
checkpoint:
checkpoint_sha256:
license:
pretraining_corpus_known:
pretraining_corpus_version:
sequence_overlap_status:
downstream_label_overlap_status:
published_task_head_used:
allowed_claim:
```

优先使用 base/pretraining checkpoint，而不是已经在相同下游数据上微调过的任务头。

若只能确认标签隔离、不能确认序列未见，必须写：

```text
label-disjoint; sequence exposure unknown
```

不得写 unseen-sequence。

## 10.4 Oracle / critic role matrix

必须建立：

| 角色 | 可用数据 | 可参与调参 | 可参与生成 guidance | 可做 final evaluation |
|---|---|---:|---:|---:|
| teacher | train | 是 | 是 | 否 |
| selection critic | train/validation | 是 | 可选 | 否 |
| final measured labels | final only | 否 | 否 | 是 |
| exposed external | historical exposure recorded | 否 | 否 | 是，需限定 |
| independent critic | fixed before generation evaluation | 否 | 否 | 是 |

任意模型跨角色复用必须记录并降低 independence claim。

---

# 11. Hard constraints、soft preferences 与不确定生物规则

## 11.1 三层规则

必须区分：

### A. Syntactic / contractual hard constraints

可以构造性保证，例如：

- alphabet；
- region boundary；
- 保护位置；
- source anchor；
- 允许的 edit type；
- 最大 edit budget；
- 最大/最小长度；
- 用户明确指定必须保留的 motif；
- 禁止 identity edit；
- STOP 语义。

这些动作在任何状态下非法时，rate 必须为零。

### B. Biological soft preferences

通常不能默认宣布为“非法”，例如：

- GC；
- secondary structure；
- uAUG；
- Kozak strength；
- miRNA/RBP motif；
- G-quadruplex；
- localization motif；
- polyadenylation-related signal；
- 其他 context-dependent feature。

这些因素可以作为：

- condition；
- soft energy/reward；
- learned effect；
- 风险提示；
- 用户可选 hard rule。

只有合同明确要求某 motif 必须保持时，它才升级为该任务的 contractual hard constraint。

### C. Uncertain learned effects

公共数据中方向不稳定、跨 context 反转或证据有限的关系必须保留不确定性，不得固化成 universal rule。

## 11.2 合法性不是功能性

满足 hard constraints 只说明：

```text
candidate is admissible under the contract
```

不说明：

```text
candidate is biologically beneficial
```

合法率与功能指标必须分开报告。

---

# 12. Benchmark 的正式结构

## 12.1 Benchmark 由生成任务和效应任务共同组成

不能再把 benchmark 仅定义为 delta prediction CSV。建议形成：

```text
UTR-EditBench-5U-Gen
UTR-EditBench-3U-Gen
UTR-EditBench-Measured
UTR-EditBench-Open
UTR-EditBench-Transfer
```

## 12.2 UTR-EditBench-5U-Gen

目标：

- 5′UTR source-conditioned generation；
- substitution/indel/variable length；
- single-step/multi-step；
- endpoint/context condition；
- measured-support recovery；
- open-support generation。

5′UTR 是首个主开发域，但不因此成为唯一论文证据。

## 12.3 UTR-EditBench-3U-Gen

目标：

- 验证相同 Edit Flow 主干能否迁移；
- 使用 3′UTR-specific condition/adapter；
- 保持 endpoint 独立；
- 评估完全共享、部分共享和独立模型；
- 验证 region label 是否被真正使用。

3′UTR 不是附带小任务，而是 UTR-only 论文中对方法可迁移性的关键验证域。

## 12.4 Measured-support generation

生成候选落在已实际测量的 candidate support 内时，允许使用真实 measured labels 评估：

- hit/recovery@K；
- NDCG；
- top-quantile enrichment；
- normalized regret；
- sign/direction success；
- measured Pareto frontier；
- source-family best-candidate recall。

final measured labels 不得用于 guidance 或 model selection。

## 12.5 Open-support generation

生成候选没有实验测量时，只允许评估：

- hard validity；
- source preservation；
- edit cost；
- length distribution；
- novelty；
- diversity；
- model likelihood；
- condition consistency；
- independent critic consensus；
- uncertainty/abstention；
- computational efficiency。

不得将 open-support 的 critic score 写成“measured improvement”。

## 12.6 必须包含真正需要生成模型的任务

主 benchmark 不能只包含单点 substitution。必须至少有：

- variable-length infilling；
- insertion；
- deletion；
- edit budget \(k>1\)；
- 同 source 多样化生成；
- target-conditioned sampling；
- STOP；
- repeated multi-step rollout。

如果真实功能数据暂时不能覆盖全部能力，必须用单独的 generative-capability benchmark 补足，但不得与 measured functional evidence 混写。

---

# 13. Foundation model 与横向模型合同

## 13.1 Foundation ladder

### Shared / general mRNA

- mRNABERT；
- Orthrus；
- mRNA-LM（许可证允许时）；
- RNA-FM / RiNALMo（作为通用 RNA 对照）。

### 5′UTR specialist

- UTR-LM；
- Optimus 5-Prime；
- FramePool。

### 3′UTR specialist

- 3UTRBERT；
- Saluki；
- APARENT2（仅适合相关 endpoint）。

任务原生模型与 foundation model 的角色不能混淆。Saluki、Optimus 等可以是强 endpoint baseline，即使它们不是通用 foundation model。

## 13.2 训练顺序

每个适用 backbone 至少评估：

```text
frozen
frozen + cached embedding
adapter/LoRA
partial unfreeze
```

full fine-tune 只有在 validation 明确支持、GPU 预算可承受且 exposure/licensing 合规时启用。

## 13.3 公平性

foundation model 比较至少匹配或报告：

- 输入序列范围；
- 可见 region/context；
- 最大长度；
- 预训练语料；
- 下游标签暴露；
- 可训练参数；
- 训练 token；
- GPU-hours；
- 显存；
- 许可证；
- 是否使用公开任务头；
- 是否使用 protein/cargo 信息。

## 13.4 从零训练

必须保留至少一个合理规模的 from-scratch Edit Flow control，以回答：

```text
性能来自 Edit Flow 结构
还是仅来自 foundation representation？
```

但不得把主要算力投入大规模从零预训练，除非现有模型在输入、许可或结构上确实不适用。

---

# 14. Baseline 合同

## 14.1 Non-generative baselines

- mean/source/group baseline；
- k-mer ridge；
- XGBoost；
- small CNN/Transformer；
- absolute candidate predictor；
- \(f(candidate)-f(source)\)；
- difference features；
- Siamese paired encoder；
- full paired encoder；
- source-cached action scorer；
- top-K paired reranker。

## 14.2 Generative baselines

- autoregressive full-sequence model；
- autoregressive edit-action model；
- masked iterative generation；
- discrete diffusion；
- generic Edit Flow；
- source-conditioned Edit Flow without grammar；
- mRNA-EditFlow full model。

## 14.3 Search/optimization baselines

- random legal；
- exhaustive legal enumeration（可行时）；
- direct scorer ranking；
- greedy；
- beam；
- best-of-N；
- simulated annealing；
- evolutionary search；
- local search；
- predictor-guided gradient/latent optimization（适用时）。

## 14.4 Matched-budget protocol

必须匹配或显式报告：

```text
training examples
foundation backbone
trainable parameters
training tokens/steps
GPU-hours
peak VRAM
number of generated candidates
oracle-query count
wall-clock
action space
hard constraints
edit budget
seed count
model-selection budget
```

不允许只让 Edit Flow 使用更强 backbone、更大候选池或更多 oracle calls 后声称算法优势。

## 14.5 Amortization break-even

Edit Flow 的效率主张必须计入训练成本。

报告：

```text
upfront training cost
per-source sampling cost
per-source search cost
number of new sources required to break even
```

只比较单次采样延迟、不计训练成本，不足以支持 amortized generation claim。

---

# 15. 评测、指标与统计

## 15.1 四类 primary metrics

### A. Generative distribution

- held-out edit NLL / flow objective；
- transition reconstruction；
- source-conditioned candidate recovery；
- sequence likelihood or calibrated surrogate；
- duplication/mode collapse；
- novelty；
- diversity。

### B. Functional control

- measured recovery@K；
- measured NDCG；
- measured top-quantile enrichment；
- normalized regret；
- independent-critic ensemble score；
- target-direction success；
- condition monotonicity；
- calibration；
- coverage–risk。

### C. Constraints and edit cost

- hard validity，目标固定为 100%；
- source preservation；
- 真实 edit distance；
- edit budget violation；
- insertion/deletion/substitution mix；
- length distribution；
- STOP timing；
- edits per successful candidate。

### D. Efficiency

- candidates/second；
- latency/source；
- GPU-hours；
- peak VRAM；
- oracle queries；
- energy/compute proxy（可获得时）；
- amortization break-even。

## 15.2 Secondary effect-model metrics

效应 predictor/critic 可以报告：

- delta Spearman；
- RMSE/MAE；
- sign accuracy；
- pairwise ranking AUC；
- beneficial precision/recall；
- ECE；
- Brier score；
- coverage–risk。

这些指标不能单独证明生成模型成功。

## 15.3 Pareto evaluation

至少构造：

```text
functional quality ↑
diversity ↑
validity ↑
edit cost ↓
oracle queries ↓
latency ↓
```

报告 Pareto frontier 和预注册 hypervolume/reference point。reference point 必须在 final test 之前冻结。

## 15.4 Seeds 与不确定性

- 正式神经网络比较至少 5 个独立训练 seeds；
- 生成采样 seed 与训练 seed 分开；
- 模型 seed 不是生物样本；
- source/group bootstrap 是主要 CI 单位；
- 同一 source 多候选不能当独立生物样本；
- 所有 seed 都报告，不能只选最好 seed。

## 15.5 统计比较

使用与任务匹配的方法：

- source-group bootstrap；
- study-level macro average；
- paired permutation；
- hierarchical/mixed-effects analysis（适用时）；
- 多重比较校正；
- effect size；
- 95% CI；
- non-inferiority margin（预注册时）。

headline advantage 至少要求：

1. strongest executable baseline 可运行；
2. effect direction 在预注册主要 split 一致；
3. source-group 95% CI 支持优势或预注册 non-inferiority；
4. 不由单一 study 或单一 seed 驱动；
5. compute/query budget 公平；
6. 没有 hard-validity violation。

## 15.6 预注册门槛

具体数据相关数值应在 Phase D1 完成数据审计、但在正式模型选择和 final evaluation 前冻结。

不可变的 universal gates：

```text
hard validity = 100%
test-label tuning events = 0
unexplained train/test overlap = 0
formal training CPU fallback = 0
missing formal seeds = 0, unless recorded FAILED_WITH_EVIDENCE
final evaluator used for guidance = false
```

性能阈值不得在看到 final results 后修改。

---

# 16. 最强审稿人反驳与预先回答

## R1：小编辑预算可以穷举，为什么需要生成模型？

必须通过以下证据回答：

- \(k>1\)；
- insertion/deletion；
- variable length；
- 多样化条件生成；
- 大候选空间；
- matched-query efficiency；
- amortization break-even。

如果 headline 只依赖 \(k=1\) substitution，不能充分回答。

## R2：是不是同一个 predictor 自证？

必须通过 oracle-role matrix、final measured labels、independent critic、model/data exposure ledger 和 sensitivity analysis 回答。

## R3：连续时间轨迹是否只是人为插值？

必须：

- 承认 intermediate path 是 latent algorithmic trajectory；
- 验证 rate semantics；
- 与不含 continuous time 的 action model 比较；
- 证明 continuous-time/trajectory 带来质量、控制或效率价值；
- 不制造生物机制主张。

## R4：是否有足够 indel 与 multi-edit 数据？

Phase D0 必须优先检索。若不足，分开报告：

- generative capability；
- measured single-edit grounding；
- dense-landscape evidence；
- 未验证的 multi-edit biological claim。

## R5：“生物硬约束”是否只是人为 motif 规则？

通过三层规则区分回答：syntax/contract hard、biological soft、uncertain learned。

## R6：foundation model 是否见过测试序列或标签？

通过 exposure ledger 回答；未知即写 unknown，不得写 unseen。

## R7：5′与3′是否只是参数共享包装？

通过 shared/adapter/independent ablation、错误 region condition、跨 region transfer 和 endpoint-separated results 回答。

## R8：无湿实验如何称功能优化？

只对 measured-support 使用 measured language；对 open-support 使用 predicted/critic-guided candidate generation。当前不声称 prospective biological improvement。

## R9：生成效率是否忽略训练成本？

报告 full lifecycle cost 和 break-even source count。

---

# 17. Forward-only 科研原则

## 17.1 “只能前进不能后退”的正式定义

只能前进是指：

- 核心科学问题不因结果不理想而改成 predictor-only；
- Edit Flow 主体地位不静默取消；
- hard constraints 不放宽；
- 防泄漏规则不放宽；
- 强基线不删除；
- 预注册指标和 Gate 不事后降低；
- 失败 seed、失败 run 和负结果不删除；
- 每个失败转化为诊断、修复、新 run 或明确负结论；
- 所有替代路线仍服务同一核心问题。

只能前进不表示：

- 带着 NaN/Inf 继续训练；
- 在 GPU/磁盘危险时强行运行；
- 明知数据污染仍继续；
- 禁止安全回滚代码；
- 禁止恢复最后健康 checkpoint；
- 必须把失败写成成功。

安全暂停、回滚到最后可信代码/checkpoint、修复、创建新 run，属于前进。

## 17.2 合法状态

```text
PLANNED
REGISTERED
PREFLIGHT_PASSED
RUNNING
SAFE_PAUSED
FAILED_WITH_EVIDENCE
REPAIR_REQUIRED
SUPERSEDED_WITH_TRACE
VERIFIED
FROZEN
```

`FAILED_WITH_EVIDENCE` 是合法进展，因为它排除了一个错误路线并保留了可审计证据。

## 17.3 禁止的“假前进”

- 删除难例；
- 删除失败 seeds；
- 覆盖失败日志；
- 调低 final threshold；
- 将 test 改成 validation；
- 读取 final labels 后重新选模型；
- 改成更容易的随机 split；
- 用 CPU 偷跑 formal training；
- 因 Flow 较弱而改写成 SparseEditFormer 论文；
- 用更弱 proxy 替代原 endpoint 却保留原 claim；
- 停止运行强基线；
- 只汇报最好 checkpoint 或最好 subgroup。

---

# 18. GPU-only 训练合同

## 18.1 适用范围

以下所有正式训练必须使用 CUDA GPU：

- mRNA-EditFlow；
- foundation model probing/fine-tuning；
- 神经网络 effect predictor/critic；
- 神经网络 reranker；
- autoregressive/diffusion/generic-flow baselines；
- RL/policy training；
- 论文级 neural ablations。

CPU 允许用于：

- 下载；
- 解压；
- 数据清洗；
- schema/QC；
- 传统线性/树模型（若其正式实现本身为 CPU）；
- 单元测试；
- 统计检验；
- 绘图；
- 文档；
- 明确标记的极小 smoke。

神经网络 formal run 禁止 CPU fallback。

## 18.2 GPU preflight

每次训练前必须记录：

```text
GPU index
GPU UUID
GPU model
driver
CUDA version
framework version
free/used VRAM
GPU utilization
all existing GPU processes and owners
CPU RAM
disk availability
```

不得：

- 占用已有未知任务；
- 杀死无关进程；
- 使用模糊 PID；
- 执行 `pkill python`；
- 为了抢 GPU 停止 62 个 ENCODE 下载或其他用户任务；
- 在显存不足时静默切换 CPU。

若资源不足，任务进入 `WAITING_FOR_GPU` 或重新规划 micro-batch；不得干扰其他任务。

## 18.3 CUDA fail-closed

formal run 启动健康检查必须证明：

```text
torch.cuda.is_available() == true
model parameters are on CUDA
input batch is on CUDA
real forward pass on CUDA
real backward pass on CUDA
optimizer update completed
torch.cuda.max_memory_allocated() > 0
CPU fallback count == 0
```

任一失败，run 立即标记 `FAILED_WITH_EVIDENCE`。

## 18.4 OOM 规则

发生 OOM 时：

1. 保存 failure bundle；
2. 不切 CPU；
3. 不覆盖原 run；
4. 创建新 run ID；
5. 可降低 micro-batch；
6. 用梯度累积保持 effective batch；
7. 记录 batch、accumulation 和优化器语义变化；
8. 重新通过 preflight。

---

# 19. 每个 run 的 artifact 合同

## 19.1 绝对路径

```text
PROJECT_ROOT =
/home/cunyuliu/mrna_editflow_goal/mrna_editflow

RUN_ROOT =
/home/cunyuliu/mrna_editflow_goal/mrna_editflow/artifacts/runs/<RUN_ID>
```

## 19.2 Run ID

```text
<phase>_<model>_<dataset>_<split>_<UTC_TIME>_<SHORT_SHA>_s<seed>
```

任何超参数、代码、数据、split、seed 或 backbone 变化都必须使用新 run ID。

## 19.3 Artifact tree

```text
artifacts/runs/<RUN_ID>/
├── status.json
├── run_manifest.json
├── resolved_config.yaml
├── command.txt
├── provenance/
│   ├── goal_contract.sha256
│   ├── data_manifest.json
│   ├── split_manifest.json
│   ├── foundation_manifest.json
│   ├── exposure_ledger.json
│   └── code_manifest.json
├── git/
│   ├── commit.txt
│   ├── diff.patch
│   └── diff.sha256
├── logs/
│   ├── stdout.log
│   ├── stderr.log
│   ├── metrics.jsonl
│   ├── system_metrics.jsonl
│   └── events.jsonl
├── checkpoints/
│   ├── last_healthy.ckpt
│   ├── best_primary.ckpt
│   └── checksums.sha256
├── evaluation/
├── failure/
├── summary.json
└── DONE | FAILED
```

W&B 可以使用，但不能成为唯一证据。没有本地 JSONL、manifest 和 checkpoint 的 run 不具备 paper eligibility。

## 19.4 Manifest 必填字段

- run/task/parent run ID；
- H1–H8 hypothesis；
- evidence level；
- contract hash；
- code commit/diff hash；
- exact command；
- resolved config/hash；
- data/split hash；
- foundation checkpoint/hash/license；
- exposure ledger；
- seed；
- GPU UUID；
- start/end；
- PID/tmux/job ID；
- exit code；
- stop reason；
- artifact checksums；
- paper eligibility；
- known deviations。

## 19.5 工作区保护

- paper run 必须来自可复现的已提交代码；
- development dirty run 必须保存完整 diff 和 hash；
- 训练启动后不得修改其代码、config、data、split；
- 并行开发必须使用隔离 worktree 或不冲突目录；
- 不得 reset、checkout 覆盖或删除用户现有改动；
- 聚焦变更在适当验证后可本地 commit；
- 默认不 push、不创建 PR，除非用户另行要求。

---

# 20. 低频监控与时间管理

## 20.1 不频繁查看的定义

训练启动后：

1. 在 3–5 分钟内做一次健康检查，确认真实 GPU batch、loss 有限、日志和 checkpoint 路径可写；
2. 轻量自动 watchdog 可以每 5 分钟记录进程、GPU、内存、磁盘和 heartbeat；
3. 人工/智能体语义检查默认每 30 分钟最多一次；
4. 超长 run 可放宽到每 60 分钟一次；
5. 预计 30 分钟内完成的任务，健康检查后优先直接等待结束；
6. validation 频率由 frozen config 决定，不因焦虑临时增加；
7. 只有异常、退出、checkpoint 失败或资源危险才打破检查间隔；
8. 禁止持续 `tail -f`、分钟级反复查询或无意义刷新。

自动写入 system metrics 不等于人工频繁查看。

## 20.2 等待期间允许并行的任务

训练等待期间优先推进不争用当前 run 的工作：

- 只读文献与 baseline 审计；
- 数据 provenance/license 核对；
- manifest 和 exposure ledger；
- 文档与 claim–evidence matrix；
- 静态代码检查；
- 隔离目录中的轻量单元测试；
- 下一 Phase 配置草案；
- 固定 fixture 上的评测/绘图代码；
- 与当前训练输入无关的数据字段检查；
- 失败分析模板；
- 论文方法草图。

## 20.3 等待期间禁止并行的任务

- 修改当前 run 使用的代码、data、split、config；
- 在同一 GPU 启动第二个训练；
- 启动会显著争抢 I/O、CPU 或磁盘的任务；
- 切换当前 run worktree 分支；
- 压缩、删除或移动当前 checkpoint；
- 访问不允许读取的 final labels；
- 启动依赖尚未产生结果的下游任务；
- 以“并行”为由降低监控和证据完整性。

没有安全独立任务时，应安静等待，不用频繁查看制造忙碌。

## 20.4 任务资源标签

task registry 必须标记：

```text
GPU_EXCLUSIVE
CPU_LIGHT
CPU_HEAVY
IO_HEAVY
READ_ONLY
MUTATES_CODE
MUTATES_DATA
FINAL_LABEL_ACCESS
```

并声明 `conflict_keys` 和 `allowed_parallel_tasks`。

---

# 21. Stop、pause、resume 与故障规则

## 21.1 立即安全暂停

出现以下任一情况必须安全暂停并保存最后健康 checkpoint：

- loss、gradient、parameter、input 或关键 metric 出现 NaN/Inf；
- CUDA 不可用、设备错配或 CPU fallback；
- unexpected OOM；
- GPU Xid/ECC/illegal memory access；
- contract、code、data、split hash 在运行中变化；
- final-label access violation；
- checkpoint 写入或 checksum 失败；
- 磁盘不足以容纳预计剩余 artifacts；
- 未知进程与当前 run 发生资源冲突；
- 数据泄漏被发现；
- hard constraint violation；
- 进程长期无 heartbeat 和新指标；
- 模型状态损坏。

## 21.2 Stall

默认 stall 判据：

```text
heartbeat age >
max(15 minutes, 3 × recent p95 step duration)
```

并且没有：

- 新 metrics；
- 新 checkpoint；
- 预期中的长计算事件记录。

确认 stall 后只终止当前 run 自身 PID，不得触碰无关进程。

## 21.3 无进展

若 primary validation metric 连续 5 次没有达到预注册最小改善：

- 触发 early-stop/diagnostic gate；
- 保留 best 与 last healthy checkpoint；
- 记录完整曲线；
- 不得临时增加 patience；
- 若需改变 patience，必须创建新 run 并说明理由。

## 21.4 Resume

只有以下完全不变时才能恢复同一 run：

- code；
- data；
- split；
- config；
- seed；
- optimizer semantics；
- foundation checkpoint；
- contract。

任何实验语义变化必须新建 run，并填写 `parent_run_id`。

不能从含 NaN/Inf 或 checksum 失败的 checkpoint 恢复。

## 21.5 不自动重试

run crash 后必须先生成 failure bundle 和诊断。不得静默自动重跑，从而掩盖真实错误或浪费 GPU。

---

# 22. 阶段状态机与 Gate

## 22.1 Run 状态机

```text
REGISTERED
→ PREFLIGHT_PASSED
→ GPU_VERIFIED
→ STARTED
→ RUNNING
→ TRAINING_FINISHED
→ EVALUATED
→ VERIFIED
→ FROZEN
```

异常：

```text
RUNNING
→ SAFE_PAUSED
→ RESUMED_<N>
```

或：

```text
RUNNING
→ FAILED_WITH_EVIDENCE
→ SUPERSEDED_BY_<NEW_RUN_ID>
```

exit code 0 不等于 `VERIFIED`；只有 acceptance 全部满足才可验证。

## 22.2 Phase Gate 原则

每个 Phase 必须有：

- Goal；
- 科学假设；
- 输入；
- 允许动作；
- 禁止动作；
- 输出文件；
- 自动验收；
- 人工审计；
- 失败处理；
- 下一 Phase 解锁条件。

任何依赖未通过时，不得用 TODO 文档或 smoke artifact 伪装成完成。

---

# 23. 三条互不混淆的正式评测轨道

## Track A：Closed-world hidden-label measured pool

定义：

- candidate sequence 可以提前进入 candidate pool；
- final measured label 对 generator、teacher、selector 和超参数选择隐藏；
- candidate IDs、generated ranking、paths 和 checkpoint 在解锁标签前冻结；
- 解锁后直接用 measured labels 评估。

允许主张：

- 在 observed measured candidate pool 中 recovery/enrichment 更高；
- observed-pool NDCG/regret 更好；
- 在该公开 assay 范围内优于 matched-budget baseline。

必须将指标命名为：

```text
observed_pool_normalized_regret
```

不得将其称为 full legal action-space regret。

## Track B：Held-out generative modeling

用于评估：

- conditional likelihood；
- held-out source→candidate recovery；
- edit-script reconstruction；
- STOP；
- edit-count calibration；
- path behavior；
- source fidelity；
- variable-length capability。

这个轨道评估生成模型本身，不自动证明功能改善。

## Track C：Open-world legal generation

允许生成任何合同合法但未被测量的候选。

主要指标：

- 100% hard validity；
- novelty/diversity；
- edit cost；
- condition control；
- independent evaluator panel；
- critic disagreement；
- uncertainty；
- compute/query efficiency。

所有功能性结果必须写为：

```text
predicted
computational
proxy-supported
```

不得写 measured improvement。

---

# 24. Edit-script、coupling 与路径歧义合同

## 24.1 Endpoint pairs 不提供唯一轨迹

source 与 candidate 之间可能存在：

- 多个同成本 alignment；
- indel 在重复序列中的坐标歧义；
- 多编辑动作顺序交换；
- 相同 final candidate 的多条合法路径；
- 先改后改回的循环路径。

因此不得把一次 Levenshtein 回溯得到的 edit script 当成唯一真值。

## 24.2 必须实现

- canonical edit-script representation；
- `apply(edit_script, source) == candidate` 100% 测试；
- 所有等价路径的稳定识别或近似；
- action-order sensitivity；
- path canonicalization sensitivity；
- 可行时对等价路径随机化或边缘化；
- indel 坐标锚定；
- dynamic state 下的坐标更新；
- cycle/no-op detection。

## 24.3 Coupling 类型

必须明确区分：

```text
observed endpoint coupling
constructed alignment coupling
corruption/denoising coupling
dense-landscape coupling
property-conditioned target coupling
```

每条训练 sample 必须记录 coupling type。模型不能把 constructed path 当作实验监督的真实动作顺序。

---

# 25. Library ascertainment 与 proposal-distribution 偏差

不同实验库的候选产生机制完全不同：

- disease variants；
- 均匀随机库；
- 人工 motif library；
- 固定 scaffold combinatorial library；
- 自然变体；
- 模型设计库。

Edit Flow 直接学习 observed candidate distribution 时，可能学到“实验人员如何设计 library”，而不是“高功能编辑如何分布”。

每个数据集必须记录：

```yaml
library_design:
proposal_distribution:
source_selection:
candidate_selection:
positive_negative_balance:
edit_type_coverage:
position_coverage:
length_coverage:
motif_coverage:
known_ascertainment_bias:
```

必须执行：

- candidate proposal 分布审计；
- edit type/position/GC/length/motif/effect direction 分层；
- study/scaffold holdout；
- condition permutation；
- beneficial-only selection sensitivity；
- library-ID shortcut audit。

不得把 observed variant frequency 解释为 biological desirability。

---

# 26. 分阶段执行总览

```text
Phase C0  合同与现实对齐
Phase D0  科学问题驱动的数据发现
Phase D1  数据资格、重建与暴露审计
Phase B0  生成式 UTR benchmark 与 splits
Phase FM0 Foundation model 接入
Phase EF0 True UTR Edit Flow 工程实现
Phase GP0 Generative prior GPU 训练
Phase FC0 Functional conditioning / critic 系统
Phase ME0 Measured-support 与 candidate freeze
Phase MB0 Matched-budget 正式比较
Phase TR0 5′UTR→3′UTR 迁移
Phase ER0 Robustness、failure 与机制分析
Phase PP0 论文、复现与发布
Phase FL0 未来 full-length 决策，不在当前执行范围
```

所有 Phase 均使用 forward-only 状态机。上游 Gate 未通过时，可以继续不冲突的并行准备，但不得产生下游正式科学结论。

---

# 27. Phase C0：合同与现实对齐

## 27.1 Goal

消除当前仓库中 README、`public_intervention_contract_v1`、旧 P3/NMI 文档与本 Goal 之间的冲突，使“Edit Flow 核心、UTR-only、无新增湿实验、GSE246381 历史暴露、GPU-only”成为唯一活动合同。

## 27.2 Tasks

### C0-01 只读 preflight

核验：

```text
pwd
repo root
branch
HEAD
git status
active processes
GPU inventory
disk/RAM
62 ENCODE downloads
current contracts
current task registry
```

不得修改或停止现有进程。

### C0-02 合同冲突矩阵

生成：

```text
docs/contracts/v2_contract_conflict_matrix.md
```

逐条列出：

- 旧 predictor-first 条款；
- Flow optional 条款；
- CDS/full-length 当前任务；
- GSE246381 sealed wording；
- wet-lab wording；
- CPU/GPU 空白；
- active code 读取旧合同的入口。

### C0-03 新合同

新增：

```text
configs/utr_editflow_execution_policy.yaml
docs/utr_editflow_scientific_question.md
docs/utr_editflow_claim_matrix.md
docs/decision_log.md
tests/test_single_contract.py
```

### C0-04 Active contract audit

新增：

```text
scripts/contracts/audit_single_contract.py
tests/test_no_predictor_only_fallback.py
tests/test_gse246381_exposure_status.py
tests/test_utr_only_scope.py
tests/test_gpu_training_contract.py
```

### C0-05 README 对齐

README 必须清楚区分：

- 长期愿景；
- 当前 UTR-only 合同；
- 当前证据；
- 未来 full-length；
- 无新增湿实验。

不得同时保留互相矛盾的“Flow primary”和“Flow optional”段落。

## 27.3 Outputs

```text
configs/utr_editflow_execution_policy.yaml
docs/utr_editflow_scientific_question.md
docs/utr_editflow_claim_matrix.md
docs/contracts/v2_contract_conflict_matrix.md
docs/execution/task_registry.yaml
tests/test_single_contract.py
```

## 27.4 Acceptance

```bash
pytest -q tests/test_single_contract.py
python scripts/contracts/audit_single_contract.py --strict
```

必须满足：

```text
active predictor-only fallback = 0
active Flow-optional clauses = 0
active CDS/full-length Phase-1 tasks = 0
GSE246381 sealed wording = 0
formal neural CPU fallback allowed = 0
active contract ambiguity = 0
```

## 27.5 Boundary

- 不删除旧 Git 历史；
- 旧结果原样归档；
- 不重写旧日志；
- 不宣称 C0 完成了模型；
- 不启动 formal training；
- 不触碰下载任务。

---

# 28. Phase D0：科学问题驱动的数据发现

## 28.1 Goal

建立“H1–H8 假设 → 所需监督 → 候选数据 → 可支持 claim”的矩阵，优先寻找真正验证 variable-length、indel、多步和同 source 多候选生成的数据。

## 28.2 Tasks

### D0-01 Hypothesis-to-data matrix

输出：

```text
docs/data/hypothesis_data_requirement_matrix.md
```

每条假设记录：

- 最低监督；
- 理想监督；
- 不可替代字段；
- 替代证据；
- 替代证据不能支持的 claim。

### D0-02 Dataset capability matrix

输出：

```text
data_registry/dataset_capability_matrix.csv
```

至少包含：

```text
dataset_id
source_exact
candidate_exact
assay_matched
endpoint_explicit
replicate_noise
edit_script_recoverable
edit_script_ambiguity
substitution_coverage
indel_coverage
multi_edit_coverage
variable_length
candidates_per_source
independent_source_groups
study_context_diversity
library_ascertainment
license
historical_exposure
allowed_tasks
forbidden_claims
```

### D0-03 Systematic search

重点检索：

- UTR insertion/deletion MPRA；
- variable-length UTR library；
- combinatorial multi-edit UTR；
- source-family multiple candidates；
- 带 raw counts/replicates 的 dense measured landscape；
- 可作为真正 untouched external 的新数据。

检索来源：

```text
GEO
SRA
ENA
ENCODE
MaveDB
Zenodo
Figshare
official supplementary
official repositories
```

### D0-04 ENCODE inventory

62 个 raw 文件下载继续。完成后只做：

- checksum；
- metadata；
- license；
- assay/context；
- sequence/label role；
- I/O/磁盘规划；
- 是否存在 downstream overlap。

不能直接升级为 intervention data。

### D0-05 Current candidates audit

对 GSE145046、Sample、PLUMAGE、GSE246381、GSE217518、GSE200304、MPRAu 等逐一填表，不预设 primary role。

## 28.3 Acceptance

- 每个 H1–H8 至少有一个候选验证路径；
- indel/multi-edit 数据是否存在有明确结论；
- 所有当前数据都有 allowed/forbidden claim；
- GSE246381 标记历史暴露；
- ENCODE 标记 observational pretraining candidate；
- 没有数据因“规模大”自动成为 primary；
- 没有用 final label 选择数据版本。

## 28.4 Forward path if data are insufficient

如果真实 multi-edit/indel 功能数据不足：

1. 不改变核心问题；
2. 保留 measured single-edit grounding；
3. 用 D-A corpus 训练 variable-length prior；
4. 用 D-D dense landscape训练多步生成；
5. 单独建立 synthetic grammar correctness；
6. 继续检索外部数据；
7. 降低对应 biological claim 等级；
8. 不退回 predictor-only。

---

# 29. Phase D1：数据重建、路径与暴露审计

## 29.1 Goal

将通过 D0 的数据转换为可追溯 canonical records，并明确每条记录的 source、candidate、edit-script ambiguity、endpoint、library proposal 和 exposure。

## 29.2 Pipelines

每个 primary/secondary 数据集按需要建立：

```text
download.py
extract.py
paper_clean.py
canonical_clean.py
build_source_candidate.py
build_edit_scripts.py
reproduce_labels.py
audit_library_design.py
audit_exposure.py
tests/
README.md
manifest.yaml
```

## 29.3 Required artifacts

```text
data/data_exposure_ledger.jsonl
data/library_ascertainment_report.json
data/edit_script_ambiguity_report.json
data/measured_action_coverage_report.json
reports/data_reproduction/summary.csv
```

## 29.4 Acceptance

- 所有 paper-eligible record 有 raw/processed provenance；
- `apply(edit_script, source) == candidate` 100%；
- path ambiguity 有量化；
- source/candidate/endpoint mapping 可重现；
- rejected records 全部有 reason；
- label reproduction 状态明确；
- raw与canonical分开；
- 没有把 constructed path 写成 observed path；
- 没有把 absolute sequence 错写为 intervention。

---

# 30. Phase B0：生成式 UTR Benchmark 与 splits

## 30.1 Goal

建立同时服务 Track A/B/C 的 5′UTR 与 3′UTR benchmark，而不是只服务 delta prediction。

## 30.2 Tasks

### B0-01 Canonical schemas

实现：

```text
schemas/utr_edit_record.schema.json
schemas/edit_script.schema.json
schemas/generation_task.schema.json
```

### B0-02 Split manifests

生成：

```text
splits/5utr_source_disjoint.json
splits/5utr_study_disjoint.json
splits/3utr_source_disjoint.json
splits/3utr_study_disjoint.json
splits/cross_region_transfer.json
```

### B0-03 Leakage audit

必须检查：

- exact source/candidate；
- reverse edge/path；
- intermediate state；
- sequence cluster；
- scaffold；
- gene；
- study；
- context；
- barcode/library batch；
- foundation pretraining overlap。

### B0-04 Evaluation tracks

输出：

```text
evaluation/tracks/closed_measured_pool.yaml
evaluation/tracks/heldout_generative.yaml
evaluation/tracks/open_legal_generation.yaml
```

### B0-05 Data Card

明确 counts、bias、exposure、allowed claims 和 unsupported capabilities。

## 30.3 Acceptance

```text
unexplained overlap = 0
reverse/path leakage = 0
final endpoint as train intermediate = 0
exposure ledger coverage = 100%
track-role ambiguity = 0
```

---

# 31. Phase FM0：Foundation model 接入

## 31.1 Goal

用已有 foundation model 提供强表示和 prior，同时防止模型暴露、许可和任务头污染掩盖 Edit Flow 的真实贡献。

## 31.2 Tasks

- 官方 checkpoint 和 tokenizer；
- revision/hash/license；
- 真实 GPU loader；
- frozen embedding cache；
- LoRA/adapter；
- partial unfreeze；
- 小型 from-scratch control；
- 预训练 corpus overlap；
- task-head exposure。

## 31.3 Paper-mode restrictions

- placeholder 禁止；
- deterministic fake embedding 禁止；
- 模型未实际加载权重禁止；
- CPU foundation training 禁止；
- 未知 sequence exposure 必须披露；
- fine-tuned task head 不能冒充 clean base model。

## 31.4 Acceptance

每个模型完成：

```text
load test
tokenization test
GPU forward test
hash/license manifest
exposure record
embedding determinism
input-length behavior
memory/latency profile
```

---

# 32. Phase EF0：True UTR Edit Flow 工程实现

## 32.1 Goal

实现真正以动态当前状态 \(x_t\)、source \(x_0\)、time \(t\)、region/context/target 为条件的 continuous-time rate field。

## 32.2 Required model interface

\[
\lambda_\theta(a\mid x_t,x_0,r,c,y^\*,b,t)
\]

其中：

- \(x_t\)：动态当前 UTR；
- \(x_0\)：固定 source；
- \(r\)：5′/3′ region；
- \(c\)：assay/context/endpoint；
- \(y^\*\)：目标条件；
- \(b\)：剩余 edit budget；
- \(t\)：连续时间；
- \(a\)：INS/SUB/DEL/STOP。

## 32.3 Tasks

### EF0-01 Dynamic state

- indel 后坐标更新；
- source-to-current mapping；
- protected anchors；
- 真实 edit distance；
- budget state。

### EF0-02 Rate heads

- non-negative rate；
- token/action distribution；
- STOP；
- legal mask before normalization；
- stable log-rate。

### EF0-03 Flow objective

- coupling manifest；
- time sampling；
- bridge/corruption；
- equivalent path handling；
- numerical guards；
- inverse/replay tests。

### EF0-04 Sampler

- sampler 类型显式；
- tau/Gillespie/approximation；
- collision semantics；
- deterministic replay；
- seed control；
- trajectory logging。

### EF0-05 Region adapters

- shared；
- shared+adapter；
- independent control。

## 32.4 Engineering hard gates

- rate 非负；
- legal probabilities normalize；
- all forbidden actions rate 0；
- 100% final hard validity；
- deterministic replay；
- path application exact；
- insertion/deletion/substitution/STOP 都在真实 GPU smoke 被执行；
- variable length 真实变化；
- \(k>1\) rollout；
- 当前实现不是单纯 greedy reranker。

## 32.5 Evidence boundary

EF0 通过只代表 E0 engineering evidence，不代表功能或论文成功。

---

# 33. Phase GP0：Generative prior GPU 训练

## 33.1 Goal

先让 Edit Flow 学会合法、source-conditioned、variable-length 的 UTR 编辑分布，再进行功能条件化。

## 33.2 Curriculum candidates

```text
unlabeled UTR corruption/refinement
→ measured endpoint pairs
→ dense measured landscape
→ target-conditioned generation
```

实际顺序由 D0/D1 数据资格决定。

## 33.3 Required comparisons

- from scratch；
- frozen foundation；
- LoRA/adapter；
- no-source；
- no-time；
- no-indel；
- no-STOP；
- fixed-length；
- no-region adapter。

## 33.4 GPU rules

所有 GP0 run 遵守第 18–21 节。开发 smoke 与 paper run 目录、状态和证据等级完全分开。

## 33.5 Acceptance

- held-out generative metrics 优于 trivial corruption model；
- 无 hard constraint violation；
- 无 mode collapse 到单一候选；
- source preservation 和 edit budget 合格；
- 5′/3′ 分开报告；
- 5 seeds 正式结果在数据与超参数冻结后执行。

---

# 34. Phase FC0：功能条件、效应模型与 critic 系统

## 34.1 Goal

将公开测量 endpoint 用于生成控制，但避免 predictor self-validation。

## 34.2 Effect-model ladder

- trivial/k-mer；
- task-native published models；
- foundation frozen；
- direct delta；
- Siamese；
- explicit action-conditioned critic；
- ensemble/uncertainty。

## 34.3 Role separation

冻结：

```text
C_train
C_select
E_final
```

每次 query 记录：

```text
run_id
source_id
candidate_hash
critic_id
query_time
score
uncertainty
purpose
```

## 34.4 Reward-hacking audit

- critic/evaluator disagreement；
- sequence OOD；
- extreme score；
- motif/GC shortcut；
- condition permutation；
- adversarial candidate；
- ensemble dispersion；
- nearest-neighbor distance；
- score improvement vs likelihood/validity tradeoff。

## 34.5 Acceptance

- 同一 predictor 不作为唯一 final evaluator；
- final labels 没有被 query；
- critic uncertainty 可用；
- disagreement 报告完整；
- generator gains 不只存在于 C_train。

---

# 35. Phase ME0：Measured-support、候选冻结与历史外部集

## 35.1 Goal

在解锁任何 Track A final measured labels 前，冻结模型、候选、排序、路径、预算与统计协议。

## 35.2 Candidate freeze

生成：

```text
evaluation/candidate_freeze.json
evaluation/candidate_freeze.sha256
evaluation/preregistered_metrics.yaml
evaluation/budget_contract.yaml
evaluation/oracle_query_log.jsonl
```

freeze 至少固定：

- source IDs；
- candidate sequences/hashes；
- generated paths；
- model/checkpoint；
- seeds；
- candidate budget；
- query budget；
- baselines；
- primary metrics；
- CI；
- multiple-testing plan；
- invalid candidate penalty；
- stop rules。

## 35.3 GSE246381

固定为：

```yaml
role: historically_exposed_retrospective_external_stress_test
confirmatory_primary: false
model_selection: forbidden
threshold_definition: forbidden
error_analysis: allowed_with_label
```

继续在程序上隔离其标签是良好卫生，但不能称重新 sealed。

## 35.4 Acceptance

- candidate freeze 在 label evaluation 前完成；
- hash 可验证；
- final labels 无 pre-freeze access；
- 历史暴露完整披露；
- primary evidence 不只依赖 GSE246381。

---

# 36. Phase MB0：Matched-budget 正式比较

## 36.1 Goal

回答最关键的问题：为什么需要 Edit Flow，而不是 GPT/AR、diffusion、scorer 或 search。

## 36.2 Budget curves

每种方法至少报告多个预算点：

```text
candidate count N
oracle queries Q
wall-clock
GPU-hours
peak VRAM
```

不能只选一个最有利预算。

## 36.3 Formal baselines

必须包括：

- strongest direct scorer；
- exhaustive/rank-all（有限空间）；
- greedy；
- beam；
- best-of-N；
- simulated annealing/evolutionary；
- AR edit model；
- masked/diffusion；
- generic Edit Flow；
- full mRNA-EditFlow。

## 36.4 Primary gate

在预注册 generative primary metric/Pareto hypervolume 上：

- 相对 strongest matched-budget baseline 进行 source-group paired comparison；
- 95% CI 按预注册规则计算；
- 5′UTR 和 3′UTR 分开；
- 多重比较校正；
- hard validity 必须 100%；
- 训练成本计入 break-even；
- invalid candidates 计入预算，不能静默丢弃。

只有 5′与3′均有支持时，才能使用“UTR-general”。单区域通过只能形成 region-specific evidence。

## 36.5 Failure path

若 Flow 未通过：

1. 保留完整负结果；
2. 分类失败来自 prior、sampler、condition、critic、data、budget 还是 optimization；
3. 进入预注册架构修复；
4. 创建新 run；
5. 不删除强基线；
6. 不改成 predictor-only 项目；
7. 不在论文中虚构优势。

---

# 37. Phase TR0：5′UTR 与 3′UTR 迁移

## 37.1 Goal

检验统一 Edit Flow 是否学习可复用编辑过程，而不是简单拼接两个 endpoint。

## 37.2 Comparisons

```text
5′ only
3′ only
joint fully shared
joint shared trunk + region adapters
cross-region warm start
wrong-region condition
region-label permutation
```

## 37.3 Reporting

- endpoint 分开；
- study macro；
- region-specific failure；
- transfer gain/loss；
- catastrophic interference；
- condition sensitivity；
- shared component attribution。

## 37.4 Gate

如果 joint/shared 不优于独立模型，可以诚实得到“共享有限”的结论，但不能把结果包装为统一 UTR grammar 已被证明。

---

# 38. Phase ER0：Robustness、failure 与机制分析

## 38.1 Failure taxonomy

至少包括：

- source ambiguity；
- path ambiguity；
- assay noise；
- low count；
- domain shift；
- library ascertainment；
- motif shortcut；
- structure shortcut；
- length shortcut；
- critic exploitation；
- mode collapse；
- premature STOP；
- edit cycling；
- indel coordinate failure；
- foundation exposure；
- context reversal；
- region confusion。

## 38.2 Mechanism boundary

允许：

- matched counterfactual；
- in-silico mutagenesis；
- action-rate analysis；
- feature ablation；
- integrated gradients；
- motif/context stratification。

不允许：

- 只凭 attention；
- 将 latent trajectory 称为机制；
- 把相关性写成因果；
- 忽略 assay/context。

## 38.3 Failure card

生成：

```text
docs/failure_card.md
evaluation/critic_evaluator_disagreement.json
evaluation/generative_validity_report.json
evaluation/robustness_matrix.json
```

---

# 39. Phase PP0：论文、复现与发布

## 39.1 论文定位

论文类型：

```text
generative machine learning method
+ UTR editing benchmark
+ cross-study/context evaluation
+ constrained biological sequence generation
```

不是：

```text
therapeutic efficacy paper
prospective wet-lab paper
full-length mRNA design paper
predictor-only paper
```

## 39.2 暂定标题方向

> **mRNA-EditFlow: source-conditioned continuous-time generation of constrained UTR edits**

最终标题必须服从真实结果，不预写 superiority。

## 39.3 主图建议

1. 科学问题与 source-conditioned UTR Edit Flow；
2. 动态 state、INS/SUB/DEL/STOP 与 hard masks；
3. 数据证据等级、三条评测轨和防泄漏；
4. generative distribution 与架构消融；
5. matched-budget 生成/搜索 Pareto frontier；
6. measured-support 与 open-support 结果；
7. 5′UTR/3′UTR transfer、uncertainty 和 failure。

## 39.4 自动生成

```text
scripts/paper/build_tables.py
scripts/paper/build_figures.py
scripts/paper/audit_numbers.py
scripts/paper/audit_claims.py
scripts/paper/audit_exposure_language.py
```

禁止手工复制核心数字。

## 39.5 Submission readiness

至少满足：

- H1–H8 结果完整或负结果边界明确；
- true Flow 语义经测试；
- 5′/3′至少完成独立报告；
- matched-budget baselines 完整；
- 生成指标不只依赖 predictor；
- foundation exposure 披露；
- GSE246381 历史暴露披露；
- no-wetlab claim boundary；
- 5 seeds/CI；
- artifacts、container、data/model card；
- claim–evidence matrix；
- failure card；
- 所有主图可从 frozen artifacts 重建。

---

# 40. 当前最高优先级：下一阶段 Goal

## 40.1 Next Goal

> **在不启动正式训练的前提下，完成 Phase C0 与 D0：把仓库的活动合同彻底改为 UTR-only generative Edit Flow，建立假设—数据资格矩阵、GSE246381 暴露状态、foundation exposure 账本和 GPU/run 执行合同，为 true Edit Flow 实现解锁。**

## 40.2 严格执行顺序

### Step 1：Reality preflight

- 核验远端 repo、HEAD、dirty state；
- 核验下载任务；
- 核验 GPU 和其他进程；
- 核验 active contract/code references；
- 只读，不修改。

### Step 2：保护现状

- 保存当前合同和关键文档 hashes；
- 保存当前 dirty diff；
- 不覆盖现有用户改动；
- 建立隔离 branch/worktree（如需要）；
- 不停止下载和训练进程。

### Step 3：合同 V2

- 写 `utr_editflow_execution_policy.yaml`；
- 写 scientific question；
- 写 claim matrix；
- 写 decision log；
- 写 task registry；
- 写 contract tests。

### Step 4：冲突清理

- 移除 active Flow-optional；
- 移除 active predictor-only fallback；
- 移出 CDS/full-length；
- 更正 GSE246381；
- 统一 README；
- 归档但不删除 legacy。

### Step 5：数据资格矩阵

- 建立 H1–H8 requirements；
- 审计当前候选数据；
- 优先检索 indel/multi-edit/variable-length；
- 记录 library ascertainment；
- 记录 ENCODE 预训练候选角色。

### Step 6：执行合同基础设施

实现或规划：

```text
configs/execution_contract.yaml
schemas/run_manifest.schema.json
docs/execution/state_machine.md
scripts/execution/preflight.py
scripts/execution/launch_gpu_run.py
scripts/execution/monitor_run.py
```

### Step 7：自动验收

- 合同 tests；
- active-reference audit；
- scope audit；
- exposure audit；
- GPU contract audit；
- task registry schema。

### Step 8：聚焦提交

在所有验收通过后：

- 只提交与 C0/D0 相关文件；
- 不夹带 raw data、downloads、checkpoints 或无关 dirty changes；
- 记录 commit hash/title；
- 默认不 push。

## 40.3 Next Goal 完成标准

```text
UTR Edit Flow core status = mandatory
predictor role = support only
current scope = 5′UTR + 3′UTR only
GSE246381 status = historically exposed retrospective
wetlab current scope = none
formal neural training = GPU only
foundation strategy = reuse first
hypothesis-data matrix = complete
dataset capability matrix = complete for current candidates
active contract conflicts = 0
run artifact contract = frozen
```

未满足任何一项，不得进入 EF0/GP0 正式实现和训练。

---

# 41. 完整阶段 Todo Checklist

## C0

- [ ] 远端只读 preflight
- [ ] 当前 contracts/hash inventory
- [ ] V2 conflict matrix
- [ ] V2 YAML contract
- [ ] scientific question v2
- [ ] claim matrix v2
- [ ] decision log
- [ ] task registry v2
- [ ] contract unit tests
- [ ] README alignment
- [ ] legacy active-reference audit
- [ ] GSE246381 exposure test
- [ ] UTR-only scope test
- [ ] GPU-only contract test

## D0

- [ ] H1–H8 data requirements
- [ ] dataset capability matrix
- [ ] multi-edit/indel/variable-length search
- [ ] current candidate audit
- [ ] ENCODE inventory/checksums
- [ ] library proposal metadata
- [ ] licenses
- [ ] candidate untouched external search
- [ ] missing-data forward paths

## D1

- [ ] primary pipelines
- [ ] paper_clean/canonical separation
- [ ] source/candidate recovery
- [ ] label reproduction
- [ ] edit-script canonicalization
- [ ] path ambiguity report
- [ ] library ascertainment report
- [ ] data exposure ledger
- [ ] rejected table
- [ ] action-coverage report

## B0

- [ ] canonical schemas
- [ ] 5′ splits
- [ ] 3′ splits
- [ ] sequence-cluster split
- [ ] node/edge/path leakage
- [ ] pretraining overlap audit
- [ ] Track A/B/C manifests
- [ ] Data Card

## FM0

- [ ] mRNABERT
- [ ] UTR-LM
- [ ] 3UTRBERT
- [ ] Orthrus
- [ ] optional RNA-FM/RiNALMo/mRNA-LM
- [ ] task-native baselines
- [ ] checkpoint hashes
- [ ] tokenizer tests
- [ ] license audit
- [ ] exposure ledger
- [ ] frozen cache
- [ ] LoRA/adapter
- [ ] from-scratch control

## EF0

- [ ] dynamic current state
- [ ] source anchoring
- [ ] INS
- [ ] SUB
- [ ] DEL
- [ ] STOP
- [ ] continuous time
- [ ] non-negative rates
- [ ] legal-mask normalization
- [ ] variable length
- [ ] multi-step
- [ ] region adapters
- [ ] condition interface
- [ ] sampler semantics
- [ ] deterministic replay
- [ ] path application tests
- [ ] failure injection

## GP0

- [ ] GPU run infrastructure
- [ ] unlabeled prior
- [ ] measured-pair coupling
- [ ] dense-landscape coupling
- [ ] no-source ablation
- [ ] no-time ablation
- [ ] no-indel ablation
- [ ] no-STOP ablation
- [ ] fixed-length ablation
- [ ] 5 training seeds
- [ ] generation seeds
- [ ] no mode collapse

## FC0

- [ ] effect baseline ladder
- [ ] C_train
- [ ] C_select
- [ ] E_final
- [ ] oracle query log
- [ ] uncertainty
- [ ] reward-hacking audit
- [ ] critic disagreement
- [ ] OOD
- [ ] condition negative controls

## ME0

- [ ] candidate freeze
- [ ] budget freeze
- [ ] metrics freeze
- [ ] baseline freeze
- [ ] code/data/split/checkpoint hashes
- [ ] hidden-label access audit
- [ ] GSE246381 retrospective protocol

## MB0

- [ ] non-generative baselines
- [ ] AR baseline
- [ ] masked/diffusion baseline
- [ ] generic Flow
- [ ] search baselines
- [ ] query curves
- [ ] compute curves
- [ ] break-even
- [ ] Pareto hypervolume
- [ ] 5′ results
- [ ] 3′ results
- [ ] 5 seeds
- [ ] paired CI

## TR0

- [ ] separate models
- [ ] fully shared
- [ ] shared+adapter
- [ ] wrong-region control
- [ ] region permutation
- [ ] catastrophic interference
- [ ] endpoint-separated report

## ER0

- [ ] failure taxonomy
- [ ] path ambiguity sensitivity
- [ ] library bias
- [ ] critic exploitation
- [ ] mode collapse
- [ ] indel failures
- [ ] uncertainty/abstention
- [ ] mechanism boundary audit
- [ ] failure card

## PP0

- [ ] frozen run registry
- [ ] model card
- [ ] data card
- [ ] exposure card
- [ ] failure card
- [ ] reproducibility report
- [ ] tables from artifacts
- [ ] figures from artifacts
- [ ] claim audit
- [ ] manuscript
- [ ] supplement
- [ ] release/license audit

---

# 42. 最终项目 Goal

构建并严格验证一个真正的生成式 **mRNA-EditFlow**：

```text
source UTR
+ region
+ assay/context/endpoint
+ target condition
+ edit budget
+ hard constraints
        ↓
continuous-time legal edit rate field
        ↓
INS / SUB / DEL / STOP
        ↓
variable-length multi-step trajectories
        ↓
diverse, sparse, source-preserving UTR candidates
```

最终要回答的不是：

> 我们能否训练一个更强的 UTR 分数预测器？

而是：

> **把 UTR 设计建模为从已有 source 出发的连续时间合法编辑过程，是否能比完整序列独立生成、候选事后打分和反复搜索，更有效地学习并产生可控、多样、稀疏、变长且可迁移的候选分布？**

项目必须证明的独特价值来自：

- 动态 source-relative state；
- continuous-time rate semantics；
- insertion/deletion/substitution/STOP；
- variable length；
- multi-step trajectories；
- hard legality；
- region-aware conditioning；
- target-conditioned diversity；
- amortized generation；
- matched-budget generative advantage。

效应预测器、mRNA-EditBench、foundation model、critic、uncertainty 和 search 都服务于这个问题，但不得再次取代它。

当前无新增湿实验，所以最终结论必须严格区分：

```text
measured retrospective evidence
historically exposed external evidence
independent computational evidence
open-support predicted evidence
```

在任何阶段，工程 PASS、GPU smoke、proxy reward、训练集提升、旧文档结论或单一 predictor 高分，都不能冒充最终科学成功。

---

# 43. 参考定位（非穷尽）

- [Optimus 5-Prime](https://www.nature.com/articles/s41587-019-0164-5)
- [FramePool](https://pmc.ncbi.nlm.nih.gov/articles/PMC8136849/)
- [UTR-LM](https://www.nature.com/articles/s42256-024-00823-9)
- [mRNABERT](https://www.nature.com/articles/s41467-025-65340-8)
- [3UTRBERT](https://pmc.ncbi.nlm.nih.gov/articles/PMC11497048/)
- [Saluki](https://pmc.ncbi.nlm.nih.gov/articles/PMC9684954/)
- [APARENT2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9636789/)
- [mRNA-LM](https://academic.oup.com/nar/article/53/3/gkaf044/7997216)
- [Orthrus](https://www.nature.com/articles/s41592-026-03064-3)
- [codonGPT](https://academic.oup.com/nar/article/53/22/gkaf1345/8384118)
- [GEMORNA](https://pubmed.ncbi.nlm.nih.gov/40875799/)
- [SANDSTORM/GARDN](https://www.nature.com/articles/s41467-025-59389-8)

这些工作证明生成式 RNA/mRNA 设计、foundation representation、功能预测和受约束生成均有明确研究价值；本项目的差异化不应写成“第一个生成 mRNA”，而应落在：

```text
source-conditioned
continuous-time edit process
variable-length legal actions
minimal and auditable editing
UTR region conditioning
matched-budget generative evaluation
```

---

# 44. 文档结束声明

本文档是下一阶段执行的思想和边界合同。

后续执行者必须：

- 忠于核心科学问题；
- 使用真正的 Edit Flow；
- 只做当前 UTR 范围；
- 正式训练使用 GPU；
- 低频、事件驱动地监控；
- 等待期间进行安全、独立的并行工作；
- 保留失败证据；
- 只通过修复和新证据前进；
- 不降低门槛；
- 不泄漏 final labels；
- 不把 predictor score 冒充 measurement；
- 不把工程完成冒充科学完成。

如果出现未被本文覆盖的问题，默认动作是：

```text
安全暂停当前受影响任务
→ 保留证据
→ 定位冲突
→ 提出仍服务同一科学问题的前进方案
→ 更新 decision log
→ 取得必要确认
→ 使用新 run/task ID 继续
```

不得通过回到旧 predictor-first 科学问题来规避困难。
