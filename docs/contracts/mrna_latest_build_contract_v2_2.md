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
> - **增补状态（2026-07-30）：**`utr_editflow_goal_v2.1_additive_math_mb0`；本次仅新增数学内核、架构图与 MB0 基线合同，`utr_editflow_goal_v2` 原文逐字保留
> - **新增强制 Gate：**`FM0 → MK0 → EF0`；数学内核未冻结前，不得启动 EF0 正式实现验收或 GP0 正式训练
> - **文献快照边界：**公开模型与代码资格以 2026-07-30 审计快照为准；论文冻结前必须刷新，未检索到精确同任务模型不等于证明其不存在

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
2. `configs/utr_editflow_contract_v2.yaml`；
3. `docs/utr_editflow_scientific_question_v2.md`；
4. `docs/utr_editflow_claim_matrix_v2.md`；
5. `docs/execution/task_registry_v2.yaml`；
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

## 4.4 2026-07-30 创新性与“第一”主张边界

本次定向文献审计未核实到一个同时满足下列全部条件的公开方法：

```text
5′UTR + 3′UTR
source-conditioned sequence editing
continuous-time Edit Flow
INS/SUB/DEL + explicit STOP
stepwise UTR grammar hard mask
region-aware shared trunk/adapters
matched-budget independent evaluation
```

这是“在冻结检索范围内尚未核实”的陈述，不是“世界上不存在”的证明。更重要的是，已经存在会显著收窄本项目创新性表述的直接先例：

- [Edit Flows](https://arxiv.org/abs/2506.09018)：已经提出 variable-length INS/DEL/SUB Edit Flow；
- [EvoFlows](https://arxiv.org/abs/2603.11703)：已经从 template protein 出发进行可控数量的 INS/DEL/SUB sequence-to-sequence 编辑；
- [Flexible Flows for Biological Sequence Design](https://arxiv.org/abs/2606.10543)：已经将 structured coupling、latent edit-based rates 与 guidance 用于多类生物序列设计；
- [pCoMole](https://openreview.net/forum?id=tTILzscPs4)：已经研究由预训练 Edit Flow 驱动、带 hard terminal feasibility 的多目标 biomolecule editing；
- [SPROUT](https://openreview.net/forum?id=4AF7WSp7Cs)：已经研究由 rollout utility 引导的 promoter Edit Flow 编辑。

因此，在没有新证据时禁止使用：

```text
the first Edit Flow for biological sequences
the first source/template-conditioned Edit Flow
the first constrained or function-guided Edit Flow
the first variable-length biological sequence editor
```

本项目可争取、但仍需在论文冻结前重新检索并精确限定量词的贡献是：

```text
UTR-specific source-conditioned Edit Flow
unified but region-aware 5′UTR/3′UTR modeling
stepwise UTR grammar legality rather than terminal-only filtering
audited explicit STOP and forced-termination semantics
measured-pool + held-out-generative + open-support evaluation tracks
matched candidate/query/compute-budget comparison
```

论文主张优先写“we formulate / we develop / we evaluate”，而不是“the first”。若最终保留任何 first claim，必须同时给出检索日期、数据库、query、排除标准和与最近先例的逐字段差异表。

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

## 6.5 模型架构图绘制合同

### 6.5.1 图的唯一叙事目标

架构图只回答一件事：

> **mRNA-EditFlow 如何从固定 source UTR 出发，在区域、功能目标和编辑预算条件下，对动态当前状态执行连续时间的合法 INS/SUB/DEL/STOP，并生成多个有效 UTR 候选。**

该图不得承担数据集、损失函数、全部评测指标和论文结果的说明任务。主图必须是单面板、单主轴、一个清晰回环；详细数学内核放在方法图或补充材料。

### 6.5.2 必须出现与不得出现

必须出现：

- 固定 `Source UTR \(x_0\)` 与动态 `Current state \(x_t\)`；
- region、assay/context/endpoint、target condition 与 edit budget；
- shared foundation encoder / sequence prior；
- region-aware Edit Flow trunk 与 5′/3′ adapter；
- `INS / SUB / DEL / STOP` 四类非负 event-rate heads；
- 在 event normalization 之前将非法 rates 置零的 hard legality mask；
- sampler 执行编辑后回到 \(x_t\) 的连续时间回环；
- 多个 diverse、sparse、source-preserving、valid UTR candidates；
- critic 的可选虚线旁路；
- 与生成器严格单向隔离的 independent evaluator。

不得出现：

- CDS、全长 mRNA、蛋白质、核糖体或湿实验元素；
- 将 critic 画成生成器本体；
- evaluator 回流指导生成器；
- 将 latent edit trajectory 画成实验观测到的生物学轨迹；
- 大段公式、数据表、结果曲线、密集小字和交叉箭头。

### 6.5.3 可直接交给绘图模型的英文提示词

```text
Create a publication-grade scientific architecture diagram for “mRNA-EditFlow” in ultra-high resolution, 7680 × 4320 pixels, 16:9 landscape, pure white background, crisp vector-like flat design, precise alignment, generous whitespace, and a restrained Nature / Nature Machine Intelligence visual style. Use only navy blue, muted teal, soft cyan, warm orange, and neutral gray. No gradients, shadows, 3D effects, or decorative biology.

Use one clean left-to-right pipeline and one central feedback loop:

1. INPUT — Show two short RNA sequence strips: “Source UTR x₀” as a fixed anchor and “Current state xₜ” as a dynamic variable-length sequence. Add three small condition chips: “Region: 5′UTR / 3′UTR”, “Assay / context / endpoint”, and “Target condition + edit budget”.

2. ENCODING — Feed x₀ and xₜ into one shared box labeled “Foundation encoder”. Make it visually clear that x₀ stays fixed while xₜ changes. Fuse the encoded states with time t and the condition chips.

3. CORE GENERATOR — Use one central box labeled “Region-aware Edit Flow”, with two slim alternative paths labeled “5′ adapter” and “3′ adapter”. This is one shared model, not two unrelated networks.

4. EVENT RATES AND LEGALITY — Place four equal compact heads in this exact order: “INS”, “SUB”, “DEL”, “STOP”, grouped under “Non-negative event rates”. Immediately after them place a compact filter labeled “Hard legality mask” with the small subtitle “invalid rates = 0 before event normalization”.

5. SAMPLING LOOP — Send the masked rates into “Continuous-time sampler”. INS, SUB, or DEL goes to a small box labeled “Apply edit; update xₜ, coordinates, length, budget”, followed by one smooth curved arrow back to “Current state xₜ”. Label the loop “continuous-time edit trajectory”. Learned STOP exits the loop. If a forced-termination boundary is shown, use a separate neutral-gray label; never draw or label forced termination as learned STOP.

6. OUTPUT — Show three to five distinct short RNA sequence strips labeled “Diverse, sparse, source-preserving, valid UTR candidates”, with only tiny restrained marks indicating substitutions, insertions, and deletions.

Place a small secondary box above the generator labeled “Frozen guidance critic — optional”, connected to the Edit Flow rate field by one thin dashed one-way arrow. Do not show reranking in this minimal main figure. At the far right, behind a thin vertical gray boundary labeled “frozen evaluation boundary”, place “Independent evaluator — evaluation only”. Draw exactly one one-way arrow from final candidates to the evaluator and absolutely no arrow back.

Use concise, perfectly legible sans-serif typography. Keep all labels horizontal and the total visible text minimal. The hierarchy must be immediately understandable at journal-column size: source/current states → foundation encoder → region-aware Edit Flow → INS/SUB/DEL/STOP rates → hard mask → continuous-time sampling loop → multiple valid UTR candidates. The result must look like a clean single-panel Figure 1 for a high-impact computational biology and machine-learning paper.
```

### 6.5.4 Negative prompt

```text
photorealistic, 3D rendering, glossy surfaces, gradients, drop shadows, neon colors, dark background, cluttered infographic, crowded labels, tiny unreadable text, multiple panels, excessive equations, decorative biology, DNA double helix, chromosome, CRISPR, cell, nucleus, ribosome, protein structure, CDS, ORF, full-length mRNA, wet-lab equipment, crossed arrows, duplicated modules, missing insertion or deletion head, autoregressive GPT decoder as the main model, critic replacing the generator, evaluator connected back to the generator, evaluator used for guidance, post-hoc repair, observed biological trajectory claim, logo, watermark, citation text, spelling errors, garbled mathematical notation
```

### 6.5.5 出图验收

- 必须同时保存 7680×4320 master preview 与可编辑 SVG/PDF vector master；最终论文版文字不得栅格模糊；
- 缩小到单栏宽度后，主流程、四类动作和唯一回环仍可辨认；
- hard mask 必须在 event normalization 前将非法 rate 置零，不能只做到“早于 sampler”；
- critic 必须是虚线旁路，independent evaluator 必须无反馈边；
- forced termination 不得画成 learned STOP；
- 图中只能出现 UTR 范围；
- 若绘图模型生成错误文字，必须在矢量软件中重排文字，不得直接使用乱码图。

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

本次增补加入两个不可跳过的冻结点，但不删除或重排上述原始阶段：

```text
Phase FM0
    ↓
Phase MK0  数学内核冻结：state/action、coupling、path、rates、loss、STOP、sampler
    ↓
Phase EF0

Phase ME0
    ↓
Phase MB0-Freeze  任务、可执行 baseline、published anchor 与 matched budget 冻结
    ↓
Phase MB0-Run
```

`MK0` 未通过时可以继续数据清洗、文献资格审计和基线工程准备，但不得把任意训练 loss 或采样脚本称为“True UTR Edit Flow”。`MB0-Freeze` 未通过时不得查看 final labels 后再选择、删除或改名 baseline。

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
configs/utr_editflow_contract_v2.yaml
docs/utr_editflow_scientific_question_v2.md
docs/utr_editflow_claim_matrix_v2.md
docs/decision_log.md
tests/test_utr_editflow_contract_v2.py
```

### C0-04 Active contract audit

新增：

```text
scripts/contracts/audit_active_contracts.py
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
configs/utr_editflow_contract_v2.yaml
docs/utr_editflow_scientific_question_v2.md
docs/utr_editflow_claim_matrix_v2.md
docs/contracts/v2_contract_conflict_matrix.md
docs/execution/task_registry_v2.yaml
tests/test_utr_editflow_contract_v2.py
```

## 27.4 Acceptance

```bash
pytest -q tests/test_utr_editflow_contract_v2.py
python scripts/contracts/audit_active_contracts.py --strict
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

# 31A. Phase MK0：UTR Edit Flow 数学内核冻结

> **阶段位置：**FM0 之后，EF0 与任何正式 GP0 训练之前。  
> **阶段性质：**冻结数学语义、数值算法、接口和失败语义；不以模型性能为验收目标。  
> **阻断规则：**MK0 Gate 未通过，不得把 greedy/reranker、普通序列模型或仅带 time embedding 的分类器称为 mRNA-EditFlow。

## 31A.1 原始方法与项目扩展边界

原始 Edit Flows 将变长序列生成定义为由 insertion、deletion、substitution 驱动的连续时间 Markov chain，并使用 alignment-augmented flow-matching/Bregman rate objective。原始动作集合没有本项目所定义的显式 `STOP`；论文默认采样实现是固定时间网格上的一阶近似，而不是精确 Gillespie 事件时间模拟。

因此必须永久区分：

```text
原始 Edit Flows 内核：
INS + DEL + SUB
variable-length CTMC rate field
alignment-augmented probability path
Bregman / rate-matching objective
fixed-step first-order sampling approximation

mRNA-EditFlow 项目扩展：
fixed source anchoring
UTR grammar and protected-anchor masks
edit budget
5′/3′ region/context/target conditioning
explicit STOP and forced termination
foundation encoder fusion
optional functional critic guidance
```

未经单独证明和数值验证，不得把 explicit STOP、功能 guidance、预算控制或近似事件模拟写成原论文已有理论结论；不得使用 `exact Gillespie`、`exact CTMC sampling` 或 `observed biological edit trajectory` 等表述。

## 31A.2 状态空间

令 RNA alphabet 与变长 UTR 空间为：

\[
\mathcal V=\{A,C,G,U\},\qquad
\mathcal X=\bigcup_{L=L_{\min}}^{L_{\max}}\mathcal V^L.
\]

为避免时间下标与 source 名称混淆，在数学实现内部固定：

```text
x_src ≡ 文档接口中的固定 source x_0
x_tar ≡ paired/corrupted target x_1
x_t   ≡ t 时刻的动态 current state
```

推理时可见的随机扩展状态定义为：

\[
Y_t=(x_{\mathrm{src}},x_t,M_t^{\mathrm{run}},r,c,y^\*,B,b_t,h_t^{\mathrm{run}},q_t),
\]

时间 \(t\in[0,1]\) 是 time-inhomogeneous CTMC 的外部时钟和 network input，不是由 stochastic jump 更新的 state coordinate。为简化后文网络条件记号，定义：

\[
S_t:=(Y_t,t).
\]

其中：

- \(M_t^{\mathrm{run}}\)：只由 source、current state 和已执行动作确定的 source–current mapping、token origin、gap ID 与保护位映射；
- \(r\in\{5^\prime UTR,3^\prime UTR\}\)；
- \(c\)：assay、cell/context、endpoint 等条件；
- \(y^\*\)：目标方向、区间或 quantile；
- \(B\)：初始原子 edit budget；
- \(b_t=B-N_{\mathrm{executed}}(t)\)：剩余预算；
- \(h_t^{\mathrm{run}}\)：只由已执行动作形成、推理时完整可得的 edit-history 摘要；
- \(q_t\in\{\mathrm{ACTIVE},\mathrm{HALTED}\}\)，其中 HALTED 为吸收态。

必须把运行状态与训练辅助变量彻底隔离：

```text
M_run / h_run:
    inference-visible
    derived only from source, current state and executed actions

Z_aux = (z_src, z_t, z_tar):
    training-only Monte Carlo/alignment variable
    never enters the rate network
```

\(M_t^{\mathrm{run}}\) 与 \(h_t^{\mathrm{run}}\) 严禁包含 target sequence、target alignment、remaining-target-edits 或由其推导的任何 feature。训练时的 \(Z_{\mathrm{aux}}\) 只用于构造 target transition rates/loss，不得输入 encoder、adapter、rate head、STOP head 或 sampler。

由于 rate 可以依赖 mapping、budget 和 history，本项目的数学过程是扩展随机状态空间 \(\mathcal Y\) 上的 time-inhomogeneous CTMC，而不只是序列空间 \(\mathcal X\) 上的 CTMC。必须定义固定外部时刻下的确定性 action update：

\[
T_Y:\mathcal Y\times\mathcal A\rightarrow\mathcal Y,
\qquad
Y_{t^+}=T_Y(Y_{t^-},a).
\]

并把最终 sequence distribution 明确为扩展状态过程对 \(x_t\) 分量的边缘分布。后文为简洁写出的 \(T(S_t,a)=S'\)、\(U_\theta(S'\mid S_t)\) 均是“在同一外部时刻 \(t\) 对 \(Y_t\) 应用 \(T_Y\)”的 shorthand；正式实现和推导必须写成 time-indexed generator \(G_{\theta,t}(Y,Y')\)，不得遗漏显式 \(t\) 依赖。`FAILED_NUMERICAL` 属于 CTMC 外部的 run/trajectory execution status，不参与 generator、likelihood 或有效候选分布。

MK0-v1 中每次 INS/SUB/DEL 消耗一个原子预算；逆向编辑、循环编辑和最终被改回的动作仍消耗预算。STOP 与强制终止不消耗编辑预算。累计动作数、最终 Levenshtein distance 和 source preservation 必须分别报告，不得混成一个指标。

## 31A.3 完整动作空间与硬合法性

对长度为 \(L_t\) 的活动状态：

\[
\begin{aligned}
\mathcal A_{\mathrm{ins}}&=\{\mathrm{INS}(g,v):0\le g\le L_t,\ v\in\mathcal V\},\\
\mathcal A_{\mathrm{sub}}&=\{\mathrm{SUB}(i,v):1\le i\le L_t,\ v\in\mathcal V,\ v\ne x_{t,i}\},\\
\mathcal A_{\mathrm{del}}&=\{\mathrm{DEL}(i):1\le i\le L_t\},\\
\mathcal A(S_t)&=\mathcal A_{\mathrm{ins}}\cup\mathcal A_{\mathrm{sub}}
\cup\mathcal A_{\mathrm{del}}\cup\{\mathrm{STOP}\}.
\end{aligned}
\]

所有位置使用 current-state 坐标，同时通过 \(M_t^{\mathrm{run}}\) 保留 source token ID 或 inserted-event ID。每次 indel 后必须更新 \(x_t\)、\(M_t^{\mathrm{run}}\)、gap ID、保护位、history 和预算。

定义 hard mask：

\[
m_C(S_t,a)\in\{0,1\}.
\]

mask 必须在 event selection/normalization 之前生效，非法动作 rate 严格为零。至少构造性禁止：

- 越过 UTR region boundary；
- 编辑 protected anchor；
- 超出长度上下界或剩余预算；
- identity substitution；
- 删除导致低于最小长度；
- 非法 alphabet/token；
- HALTED 状态中的编辑动作。

若训练 coupling 要求的目标动作被 hard mask 禁止，必须将该 sample/path 记入 rejected 或 coupling-repair ledger；不得静默删除该 target term 后继续训练。

## 31A.4 Source–current coupling、alignment 与路径歧义

对训练 pair \((x_{\mathrm{src}},x_{\mathrm{tar}})\)，构造：

\[
z^{\mathrm{src}},z^{\mathrm{tar}}
\in(\mathcal V\cup\{\epsilon\})^N,
\]

满足移除 blank \(\epsilon\) 后分别恢复 source 与 target。Primary coupling 的起始候选为：

```text
unit-cost Levenshtein optimal alignment
+ deterministic canonical tie-break
```

同时必须实现：

```text
sample-from-optimal-alignments sensitivity
equivalent edit-order sensitivity
```

不得把 canonical alignment 称为唯一真实生物路径。每条 sample 至少记录：

```yaml
source_id:
target_id:
coupling_type:
alignment_algorithm_version:
alignment_cost:
tie_break_rule:
alignment_hash:
path_is_observed: false
path_semantics: latent_algorithmic
```

在 augmented alignment 上定义单调 schedule \(\kappa(t)\)：

\[
p_t(z_i\mid z_i^{\mathrm{src}},z_i^{\mathrm{tar}})
=(1-\kappa(t))\delta_{z_i^{\mathrm{src}}}
+\kappa(t)\delta_{z_i^{\mathrm{tar}}},
\]

MK0-v1 的联合 path 必须显式冻结为 independent switch-clock construction，而不能只给单坐标边缘：

\[
p_t(z\mid z^{\mathrm{src}},z^{\mathrm{tar}})
=
\prod_{i=1}^{N}
p_t(z_i\mid z_i^{\mathrm{src}},z_i^{\mathrm{tar}}).
\]

对需要改变的 aligned coordinates：

\[
\mathcal I_\Delta
=
\{i:z_i^{\mathrm{src}}\ne z_i^{\mathrm{tar}}\},
\]

独立采样 switch clocks：

\[
P(\tau_i\le t)=\kappa(t),\qquad i\in\mathcal I_\Delta,
\]

并定义：

\[
z_{t,i}
=
\begin{cases}
z_i^{\mathrm{tar}}, & \tau_i\le t,\\
z_i^{\mathrm{src}}, & \tau_i>t.
\end{cases}
\]

剩余 target-switch 集合为：

\[
R_t
=
\{i\in\mathcal I_\Delta:\tau_i>t\}.
\]

在所有坐标共享 schedule 时，每个 remaining switch 的 conditional target hazard 明确为：

\[
\rho(t)
=
\frac{\dot\kappa(t)}{1-\kappa(t)},
\qquad
w_t(e_i)=\rho(t),\quad i\in R_t.
\]

若 structured coupling 使用不同坐标 schedule，必须从对应 joint path 推导 \(\rho_i(t)\) 并建立新 kernel hash；不得继续使用未定义的 \(w_t(e)\)。

再由移除 blank 得到 observable \(x_t\)。训练辅助变量记为：

\[
Z_{\mathrm{aux}}=(z^{\mathrm{src}},z_t,z^{\mathrm{tar}}).
\]

它只用于构造 target transition，不得进入 \(u_\theta(\cdot\mid S_t)\)。对任意 next extended state：

\[
U_\theta(S'\mid S_t)
=
\sum_{a:T(S_t,a)=S'}u_\theta(a\mid S_t).
\]

多个 aligned coordinate、重复字符或不同 edit scripts 到达同一 \(S'\) 时，必须先把 target transition weight 与模型 transition rate分别按 \(S'\) 聚合，再取 log。只有在形式证明表明 action-labelled extended states 彼此不同，或 action-level 重复计数与 transition-level 聚合严格等价时，才允许 action-level log-rate。toy oracle 必须包含 repeated-symbol insertion/deletion 和多个 edit scripts 到达同一 observable sequence 的案例。

## 31A.5 时间方向、schedule 与 endpoint

唯一 canonical 方向为：

```text
t = 0：source UTR / source distribution
t = 1：target candidate / target distribution
```

不得混用 diffusion 的 `data → noise` 叙述。若未来训练 reverse process，必须创建独立方向字段、配置、测试和 artifact。

MK0 的 reference schedule 为原始 Edit Flows 实验采用的 cubic candidate：

\[
\kappa(t)=t^3,\qquad
\dot\kappa(t)=3t^2,\qquad
w(t)=\frac{\dot\kappa(t)}{1-\kappa(t)}
=\frac{3t^2}{1-t^3}.
\]

linear \(\kappa(t)=t\) 作为预注册 sensitivity candidate。最终 primary schedule 只能用 development evidence 选择一次，并在 `math_kernel_v1.yaml` 中冻结；final labels 不得参与。网络不得在 \(t=1\) 奇点精确求值；`time_eps`、训练时间分布、endpoint clipping、采样步数和最后求值时间必须写入配置。任何 rate/weight clip 都必须记录命中次数和敏感性，不得静默发生。

## 31A.6 Rate factorization、总 hazard 与 CTMC generator

外部接口 \(\lambda_\theta(a\mid S_t)\) 在实现中展开为：

\[
u_\theta(\mathrm{INS}(g,v)\mid S_t)
=m^{\mathrm{ins}}_{g,v}\,
\lambda^{\mathrm{ins}}_{\theta,g}(S_t)\,
Q^{\mathrm{ins}}_{\theta,g}(v\mid S_t),
\]

\[
u_\theta(\mathrm{SUB}(i,v)\mid S_t)
=m^{\mathrm{sub}}_{i,v}\,
\lambda^{\mathrm{sub}}_{\theta,i}(S_t)\,
Q^{\mathrm{sub}}_{\theta,i}(v\mid S_t),
\]

\[
u_\theta(\mathrm{DEL}(i)\mid S_t)
=m^{\mathrm{del}}_i\lambda^{\mathrm{del}}_{\theta,i}(S_t),
\qquad
u_\theta(\mathrm{STOP}\mid S_t)
=m^{\mathrm{stop}}\lambda^{\mathrm{stop}}_\theta(S_t).
\]

要求：

- 所有 \(\lambda\ge0\)，使用数值稳定的非负参数化；
- \(Q^{\mathrm{ins}}\) 与 \(Q^{\mathrm{sub}}\) 只在合法 nucleotide 上归一化；
- 某 gap/position 没有合法 token 时，相应 operation rate 为零；
- source/current/region/context/target/time/budget 必须真实进入 rate field；
- 不允许先采样非法动作，再靠 rejection 或 post-hoc repair 粉饰合法率。

总 hazard 为：

\[
\Lambda_{\mathrm{all}}(S_t)=
\sum_{a\in\mathcal A_{\mathrm{legal}}(S_t)}
u_\theta(a\mid S_t).
\]

对给定“发生了一个事件”的条件分布：

\[
P_\theta(a\mid \text{event},S_t)=
\frac{u_\theta(a\mid S_t)}{\Lambda_{\mathrm{all}}(S_t)}.
\]

该条件分布只在 \(\Lambda_{\mathrm{all}}(S_t)>0\) 时定义。瞬时总 hazard 为零时禁止执行除法，本时刻事件概率为零并继续推进时间；**不得仅因当前 \(\Lambda_{\mathrm{all}}(S_t)=0\) 就终止**。例如 cubic schedule 在 \(t=0\) 的 target-edit hazard 可以正好为零，但稍后为正。

只有验证从当前时刻到 horizon、沿 no-event state evolution 的剩余 integrated hazard：

\[
H_{\mathrm{rem}}(t)
=
\int_t^1\Lambda_{\mathrm{all}}(S_s,s)\,ds
\]

确实为零，才允许提前标记 `FORCED_ZERO_REMAINING_INTEGRATED_HAZARD`；否则必须以 `NO_EVENT` 推进到下一个时间点并重算 rates。必须区分：

```text
zero edit hazard + positive STOP hazard
zero instantaneous total hazard
zero remaining integrated total hazard
```

必须注意：**CTMC rates 本身不要求加和为 1；只有 conditioned-on-event action distribution 才归一化。** 扩展状态 generator 的对角项为 \(-\Lambda_{\mathrm{all}}\)；off-diagonal 项使用 \(U_\theta(S'\mid S_t)\)，即对到达同一 next extended state 的动作求和。实现需枚举小状态验证 factorized hazard、逐动作 hazard、transition aggregation 和 generator row-sum 一致。

## 31A.7 Edit Flow / Bregman 核心目标

令 \(\mathcal N_{\mathrm{edit}}(S_t)\) 为合法编辑可到达的唯一 next extended states，并把 training auxiliary path 的 target weights 按 next state 聚合：

\[
W_t(S'\mid S_t,Z_{\mathrm{aux}})
=
\sum_{\substack{i\in R_t:\\T(S_t,\bar a(e_i))=S'}}
\rho(t).
\]

忽略与参数无关常数，MK0 的 reference Edit Flow objective 为：

\[
\mathcal L_{\mathrm{EF}}
=
\mathbb E
\left[
\sum_{S'\in\mathcal N_{\mathrm{edit}}(S_t)}
U_\theta(S'\mid S_t)
-
\sum_{S':\,W_t(S'\mid S_t,Z_{\mathrm{aux}})>0}
W_t(S'\mid S_t,Z_{\mathrm{aux}})
\log U_\theta(S'\mid S_t)
\right],
\]

其中：

- \(R_t\)：由 joint switch clocks 明确定义的尚未完成 aligned target switches；
- \(\bar a(e)\)：由辅助 edit 映射出的 observable INS/SUB/DEL；
- \(w_t(e_i)=\rho(t)=\dot\kappa(t)/(1-\kappa(t))\)：由冻结的 joint probability path 推导，不得任意手调；
- \(U_\theta(S'\mid S_t)\)：到达同一 next extended state 的所有 action rates 之和；
- 只有 next state 唯一对应单个 INS/SUB 动作时，log transition rate 才可直接分解为 operation-rate 与 nucleotide-distribution 两项；
- illegal target edit 表示数据、alignment 或约束冲突，必须 fail closed。

必须用可穷举 toy state 核对 repeated symbols、target multiplicity、action-to-next-state aggregation、rate-sum、loss 数值和梯度。一般情况下禁止用 \(\sum_e w_e\log u_\theta(a_e)\) 代替 \((\sum_e w_e)\log(\sum_a u_\theta(a))\)。仅训练 action cross-entropy、denoising loss 或 reward loss后，不得称为 Bregman Edit Flow。

MK0 与 GP0-primary 的总损失固定为：

\[
\mathcal L_{\mathrm{MK0/GP0-primary}}
=\mathcal L_{\mathrm{EF}}
+\alpha_{\mathrm{stop}}\mathcal L_{\mathrm{stop}},
\qquad
\alpha_{\mathrm{cond}}=\alpha_{\mathrm{reg}}=0.
\]

critic reward、final evaluator、RL、额外 condition loss 和未定义 regularizer 不得混入 `math_kernel_v1`。未来任何新增目标必须建立新 kernel version/hash，定义数据、权重、ablation 与 Gate。若没有验证可归一化概率估计，不得把 held-out flow objective 写成 exact NLL。

## 31A.8 STOP 与强制终止

`STOP` 是本项目扩展，不是原始 Edit Flows 的原生动作。它进入吸收态，之后序列不再变化。必须区分：

```text
LEARNED_STOP
FORCED_BUDGET
FORCED_NO_LEGAL_EDIT_ACTION
FORCED_ZERO_REMAINING_INTEGRATED_HAZARD
FORCED_TIME_HORIZON
FAILED_NUMERICAL
```

`FORCED_NO_LEGAL_EDIT_ACTION` 只表示 INS/SUB/DEL 合法集合为空；STOP 是否有正 hazard 单独判断。强制终止不得计为模型正确预测 STOP；fixed-step 中的 `NO_EVENT` 只表示时间前进，本步无编辑，不等于 STOP；NaN、Inf、非法概率或损坏状态必须记为 CTMC 外部的 `FAILED_NUMERICAL`，不得输出成有效 candidate。

`competing-risk STOP` 的比例：

\[
\frac{\lambda_{\mathrm{stop}}}
{\lambda_{\mathrm{stop}}+\Lambda_{\mathrm{edit}}+\varepsilon}
\]

只识别 event type 的相对比例，无法识别 CTMC STOP intensity 的绝对时间尺度；同比例缩放全部 rates 时该比例不变，但停止时间分布会改变。因此它只能作为离散 decoder/event-type 消融，不得成为 explicit CTMC STOP 的 primary objective。

若 STOP 保留在总 hazard 中，MK0 primary 必须使用具有绝对时间尺度、且不与最后一次 edit 同时发生的 survival construction。由 31A.4 的 independent edit clocks 定义结构完成时间：

\[
\tau_{\mathrm{comp}}
=
\max_{i\in\mathcal I_\Delta}\tau_i,
\]

zero-edit pair 取 \(\tau_{\mathrm{comp}}=0\)。然后独立采样：

\[
D\perp\{\tau_i\},\qquad
D\sim\mathrm{Exponential}(\gamma_{\mathrm{ref}}),
\qquad
P(D>0)=1.
\]

构造 latent STOP time、administratively observed time 与 event indicator：

\[
\tau_{\mathrm{stop}}=\tau_{\mathrm{comp}}+D,
\qquad
\tilde\tau=\min(\tau_{\mathrm{stop}},1),
\qquad
\delta=\mathbf 1\{\tau_{\mathrm{stop}}<1\}.
\]

等号事件概率为零。这样 zero-edit pair 也不会在 \(t=0\) 产生 STOP 原子，最后一次 edit 与 STOP 不会构成同刻双跳。

联合 auxiliary path 的 terminal state 必须定义为：

```text
q_t = ACTIVE   for t < tau_stop
q_t = HALTED   for t >= tau_stop when delta = 1
q_t = ACTIVE   through t = 1 when administratively censored
```

HALTED 后序列、mapping、history 和 budget 冻结，合法 edit set 为空；\(\mathcal L_{\mathrm{EF}}\) 不得在 STOP 后继续按原 \(z_t\) 训练编辑。

STOP loss 使用 event 前的 predictable state：

\[
\mathcal L_{\mathrm{stop}}
=
-\delta\log
\lambda_{\mathrm{stop}}(S_{\tilde\tau^-})
+
\int_0^{\tilde\tau}
\mathbf 1\{q_s=\mathrm{ACTIVE}\}
\lambda_{\mathrm{stop}}(S_{s^-})\,ds.
\]

其中 rate network 仍显式接收外部时钟 \(s\)。\(\gamma_{\mathrm{ref}}\)、数值积分网格、quadrature、endpoint、at-risk process、dtype 与 auxiliary censor fraction 必须在 development-only sensitivity 后冻结，并由 event-rate/censor-fraction tiny oracle 验证。

固定 horizon \(1\) 只对这个**人工构造的 STOP target process**构成预先规定的 administrative censoring。`FORCED_BUDGET`、`FORCED_NO_LEGAL_EDIT_ACTION`、`FORCED_ZERO_REMAINING_INTEGRATED_HAZARD` 与实际 inference 的 `FORCED_TIME_HORIZON` 是独立 execution termination reasons；不得据此假设它们是 learned-STOP likelihood 中的非信息 censoring。

上述 dwell-time survival objective 是 MK0-v1 primary candidate；只有 loss、absolute event-rate calibration、censor fraction、premature-STOP、never-STOP 和 numerical-integration Gate 全部通过后才能冻结。competing-risk ratio 保留为明确标注的 non-CTMC STOP ablation。functional STOP——“继续编辑是否还有功能收益”——只能在 FC0 critic 角色冻结后另立版本，不得混入结构性 STOP。

必须报告 learned/forced STOP 比例、premature STOP、never-STOP、预算耗尽、instantaneous-zero-hazard steps、verified zero-remaining-integrated-hazard 和 termination reason 分布。任务配置必须显式声明是否允许 step-0 identity output。

## 31A.9 Foundation encoder 与 \(x_{\mathrm{src}}/x_t\) 融合

冻结双状态表示：

\[
H_{\mathrm{src}}=E_\psi(x_{\mathrm{src}}),\qquad
H_t=E_\psi(x_t).
\]

\(H_{\mathrm{src}}\) 可缓存；\(H_t\) 在每次 edit 后必须完整重算，或使用经过逐层等价性测试的增量更新。reference implementation 优先完整重编码 \(x_t\)，先保证语义正确，再优化速度。

rate head 至少接收：

```text
current token/gap representation
source-aligned representation derived only from M_run
source-current features available identically at inference
region adapter
assay/context/endpoint
target condition
time embedding
remaining budget
edit-history summary
```

禁止只缓存 source embedding 而忽略动态 \(x_t\)，也禁止把 \(Z_{\mathrm{aux}}\)、target alignment 或 remaining target edits 注入 feature；禁止把 foundation model 的直接序列输出改名为 Edit Flow。必须加入 target-alignment leakage test：打乱或替换 training target/alignment 不得改变给定相同 inference-visible \(S_t\) 的 network rates。顺序保持：

```text
frozen foundation
→ adapter/LoRA
→ partial unfreeze
```

from-scratch 小模型保留为结构对照。checkpoint、hash、license、pretraining exposure 与长度行为必须进入 manifest。

## 31A.10 Critic/conditioning 角色边界

MK0 内核必须在没有 critic 时完成合法采样。后续 FC0 可比较：

- classifier-free condition；
- frozen critic guidance；
- post-generation reranking。

必须只选一个 development-primary route 后再冻结 final evaluation；三者不得择优挑 final。Critic 可以改变 sampling preference，但不能改变 hard legality。`E_final` 不得向 generator、sampler 或 selector 提供梯度、分数或查询结果。

同一模型跨 teacher/guidance/selection/evaluation 角色复用时，必须降低 independence claim 并报告共享的数据、权重与特征。所有 guidance/selection query 必须写 append-only log。

## 31A.11 MK0 分阶段 Todo

### MK0-01 原始方法事实矩阵

- 固定 Edit Flows 论文版本、公式、实现 revision 与 hash；
- 标记 `original / project extension / future`；
- 固定禁止使用的 exact/biological trajectory 表述。

### MK0-02 State/action schema

- 定义 state、token origin、gap ID、atomic action、budget 与 termination reason；
- 完成 INS/SUB/DEL/STOP apply、inverse/replay 和 coordinate update。

### MK0-03 Alignment/coupling

- canonical optimal alignment；
- sampled optimal-alignment sensitivity；
- equivalent edit-order sensitivity；
- coupling manifest 与 rejected-path ledger；
- `Z_aux` 与 inference-visible state 隔离测试。

### MK0-04 Probability path 与 schedule

- cubic reference 与 linear sensitivity；
- joint product path 与 independent switch clocks；
- remaining-switch set \(R_t\) 与 \(\rho(t)\) target hazard；
- derivative、endpoint、singularity、time-direction tests；
- 冻结 time sampling、`time_eps` 与 clip policy。

### MK0-05 Rate/Bregman kernel

- factorized rate heads；
- hard mask before event normalization；
- action-to-next-state transition aggregation；
- total hazard 与 extended-state generator；
- brute-force loss/gradient oracle。

### MK0-06 STOP extension

- survival-hazard primary 与 absolute-time-scale oracle；
- post-completion exponential dwell 与 \(\gamma_{\mathrm{ref}}\) sensitivity；
- administrative-censor fraction oracle；
- competing-risk event-type ablation；
- forced termination state machine；
- STOP calibration、premature/never-stop tests；
- 冻结 primary 与 ablation。

### MK0-07 Sampler

- 原论文相容的 fixed-step parallel first-order reference；
- 严格预算的 single-event first-order primary；
- parallel joint-event legality audit；
- deterministic replay；
- step-halving convergence；
- tiny-state numerical event-time reference。

### MK0-08 Foundation fusion

- source cache；
- dynamic current full re-encoding；
- alignment-aware feature gather；
- incremental/full equivalence test；
- foundation/from-scratch GPU smoke。

### MK0-09 Critic boundary

- role matrix；
- query log schema；
- no-critic base sampler test；
- final-evaluator isolation test。

### MK0-10 Freeze

- 数学审计与 failure injection；
- CPU symbolic/unit tests；
- 真实 GPU forward/backward tiny smoke；
- manifest/hash/decision log；
- focused MK0 freeze commit。

CPU 只允许用于符号推导、穷举 oracle 和非神经 unit tests；任何神经网络 forward/backward、smoke 或训练验收必须使用 GPU。

## 31A.12 Sampler 数值合同

至少实现两个明确命名的 sampler：

### A. `paper_first_order_parallel`

- 固定时间网格；
- 使用一阶 event-probability approximation；
- 所有并行事件基于 pre-step state；
- delete/substitute 冲突和 canonical application order 固定；
- manifest 必须写 `exact_gillespie: false`。

该 sampler 只用于原始算法复现和数值对照，不得直接承担严格预算 UTR 的 primary generation Gate。多个单独合法事件的联合结果可能违反预算、长度或 anchor 约束；若要在 UTR task 上保留并行采样，必须在采样前定义 joint-event legality kernel，对整个联合事件实施预算、长度与冲突约束。任何采样后 projection、culling 或 repair 必须另命名为项目近似，不能称为 paper-compatible。该 sampler 与 primary constrained sampler 分别报告 validity 和 convergence。

### B. `constrained_single_event_first_order`

对 substep \(h\)：

\[
P(\text{event}\mid S_t)=1-\exp[-h\Lambda_{\mathrm{all}}(S_t)],
\qquad
P(a\mid\text{event},S_t)=
\frac{u_\theta(a\mid S_t)}{\Lambda_{\mathrm{all}}(S_t)}.
\]

上述 action distribution 只在 \(\Lambda_{\mathrm{all}}>0\) 时求值。若 substep 起点的瞬时总 hazard 为零，则令本步为 `NO_EVENT`、推进到 \(t+h\) 并重算 rates，不得立即终止。MK0-v1 将被选事件应用于 substep 末端 \(t+h\)，不再处理该 substep 的剩余时间，明确命名为 `endpoint single-event frozen-rate approximation`。若改为采样区间内事件时间并继续处理残余区间，必须另命名 `event-driven frozen-rate sampler` 并独立验证。

每个 substep 至多执行一个 INS/SUB/DEL/STOP；当 \(h\Lambda_{\mathrm{all}}\) 超出预注册稳定阈值时自动细分。该方法仍是时间非齐次 rate field 的一阶冻结-rate近似，不得称为 exact Gillespie。正式 hard-constrained UTR 结果默认使用本 sampler。

共同要求：

- hazard accumulation 使用稳定高精度实现；
- 不允许负概率、概率和大于 1 或静默 rate clip；
- 每个 edit 后重算 current state、mask 与 rates；
- seed、time、hazard、candidate action、采样随机数、选中动作和 state hash 可重放；
- 至少三个 step-size 执行 convergence curve；
- tiny sequence 上用 numerical integrated-hazard reference 审计；
- parallel sampler 的 collision semantics 必须冻结；
- invalid action 不得采样后静默删除。

## 31A.13 Acceptance Gate

MK0 只有同时满足以下条件才 PASS：

```text
state/action schema validation = PASS
apply(action, state) exactness = 100%
alignment source→target reconstruction = 100%
constructed path_is_observed = false
target-alignment leakage tests = PASS
joint product path and independent switch clocks = PASS
remaining-switch target hazard rho(t) = derived and tested
time-direction errors = 0
schedule endpoint/derivative tests = PASS
negative rates = 0
masked illegal rates = 0
zero instantaneous hazard advances time without division = PASS
zero remaining integrated hazard termination = separately verified
conditioned-on-event action distribution = normalized when total hazard > 0
factorized hazard ~= enumerated hazard under frozen tolerance
generator row sum ~= 0 under frozen tolerance
repeated-symbol transition aggregation = PASS
Bregman toy loss ~= brute-force oracle under frozen tolerance
finite loss and finite gradients = PASS
survival STOP absolute-hazard oracle = PASS
STOP dwell D independent and positive almost surely = PASS
STOP event-rate and administrative-censor fraction oracle = PASS
HALTED states contribute no edit-flow terms = PASS
learned STOP and forced termination never conflated
primary constrained sampler hard validity = 100%
paper parallel sampler validity = separately reported
budget violations = 0
deterministic replay = 100%
step-halving convergence = PASS under preregistered tolerance
unsupported affirmative exact-Gillespie claims = 0
source-only stale-state encoding tests = PASS
paper-mode placeholder foundation = 0
base generation without critic = PASS
final evaluator used for guidance = false
GPU tiny smoke force-covers INS/SUB/DEL/STOP with nonzero oracle rates = PASS
```

每个 Gate 必须绑定：

```text
test domain
exhaustive or sampled status
sample count
dtype
atol/rtol
seed
failure denominator
artifact path/hash
```

state/action/alignment 在预注册 tiny state space 穷举；真实长度使用 property-based randomized tests。浮点 hazard、loss 和 row-sum 使用冻结容差，不能用未声明的字面相等。四类动作的 GPU smoke 在人工构造的合法非零 oracle rates 下分别强制覆盖，不要求未训练随机模型自然采样到四类动作。exact-Gillespie 文本审计必须区分禁止性/否定性陈述、相关工作引用和对本项目 sampler 的肯定性主张。

任何一项失败，状态为 `FAILED_WITH_EVIDENCE`。保留 failure bundle，修复后使用新 MK0 run ID；不得降低 Gate 或跳过 MK0 进入正式训练。MK0 PASS 只授予 E0 数学/工程证据，不代表功能提升、matched-budget 优势或论文成功。

## 31A.14 必须产出的 Artifacts

```text
docs/math/mk0_original_vs_extension_matrix.md
docs/math/mk0_state_action_spec.md
docs/math/mk0_coupling_probability_path.md
docs/math/mk0_bregman_derivation.md
docs/math/mk0_stop_semantics.md
docs/math/mk0_sampler_semantics.md
docs/math/mk0_foundation_fusion.md
docs/math/mk0_critic_boundary.md

configs/math/math_kernel_v1.yaml
schemas/edit_state_v1.schema.json
schemas/edit_action_v1.schema.json
schemas/edit_trajectory_v1.schema.json
schemas/termination_event_v1.schema.json
schemas/coupling_manifest_v1.schema.json

artifacts/mk0/coupling_manifest.json
artifacts/mk0/target_alignment_leakage_audit.json
artifacts/mk0/transition_aggregation_oracle.json
artifacts/mk0/loss_oracle_report.json
artifacts/mk0/hazard_audit.json
artifacts/mk0/sampler_convergence.json
artifacts/mk0/stop_audit.json
artifacts/mk0/stop_survival_oracle.json
artifacts/mk0/foundation_fusion_audit.json
artifacts/mk0/critic_role_audit.json
artifacts/mk0/mk0_acceptance.json
artifacts/mk0/mk0_freeze_manifest.json
artifacts/mk0/mk0_freeze_manifest.sha256
```

`mk0_freeze_manifest.json` 必须记录 Goal hash、代码 commit、公式版本、coupling、schedule、time direction、sampler、STOP primary、foundation 接口、全部测试和失败记录。EF0、GP0、FC0 正式 run 必须引用该 hash。

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

# 36A. Phase MB0 增补：同任务先例闭合与可执行 baseline 冻结

## 36A.1 对“有没有同任务模型”的正式回答

答案分两层：

1. **完全同任务：**截至 2026-07-30 的冻结检索，尚未核实到同时覆盖本项目全部任务量词的公开模型；
2. **可横向比较的最近邻：**明确存在，而且数量不少，包括 UTR 生成/优化模型、source-relative 生物序列 Edit Flow、通用 Edit Flow、AR/diffusion editor 和 predictor-guided search。

因此，不能得出“没有 baseline”；正确结论是：

> **没有可直接原封不动代替全部主比较的单一模型，但必须建立精确同接口的 controlled baselines，并把符合资格的 published-native/adapted 方法作为公开先例与系统级 comparator。**

完全同任务由以下量词共同定义：

\[
\mathcal T=(x_{\mathrm{src}},r,c,\mathcal A,P_\theta,\mathcal G,B,O,E,S),
\]

其中：

- \(x_{\mathrm{src}}\)：给定真实 source UTR；
- \(r\)：5′UTR 或 3′UTR；
- \(c\)：assay/context/endpoint/target；
- \(\mathcal A\)：INS/SUB/DEL/STOP；
- \(P_\theta\)：continuous-time Edit Flow / CTMC rate field；
- \(\mathcal G\)：每一个中间状态的 grammar hard mask；
- \(B\)：相对 source 的累计动作和 edit budget；
- \(O\)：多个候选及可审计 latent trajectory；
- \(E\)：与 generator/reward 隔离的 final evaluator 或 measured support；
- \(S\)：source-group split、exposure 与 holdout 规则。

只共享 UTR、MRL/TE/stability、生成能力、source editing 或 Edit Flow 中某一部分，均属于“最近邻先例”，不能称完全同任务，也不能从相关工作和 baseline registry 中删除。

## 36A.2 先例与最近邻模型矩阵

| ID | 方法 | 已核实的原生能力 | 与本项目的主要差异 | MB0 身份 |
|---|---|---|---|---|
| `P-EF-00` | [Original Edit Flows](https://arxiv.org/abs/2506.09018) | 以 INS/DEL/SUB 构造 variable-length Edit Flow | 无 UTR-specific condition、逐步 UTR grammar、explicit STOP 与功能评测 | `ARCH_CORE` |
| `P-EF-01` | [EvoFlows](https://arxiv.org/abs/2603.11703) | 从 template protein 出发，执行可控数量的 INS/DEL/SUB | protein 而非 UTR；无本合同的 5′/3′、STOP 与 evaluator 隔离 | `SOURCE_EDIT_PRIOR_ART` |
| `P-EF-02` | [Flexible Flows](https://arxiv.org/abs/2606.10543) | structured coupling、latent edit-based rates、guidance，用于多类生物序列 | 非 UTR source-edit 主任务；无本合同完整组合 | `BIO_EDITFLOW_PRIOR_ART` |
| `P-EF-03` | [pCoMole](https://openreview.net/forum?id=tTILzscPs4) | pretrained Edit Flow、multi-objective steering、hard terminal feasibility | biomolecule compression/optimization，非 UTR；约束主要不是本合同逐步 UTR grammar | `GUIDED_EDITFLOW_PRIOR_ART` |
| `P-EF-04` | [SPROUT](https://openreview.net/forum?id=4AF7WSp7Cs) | promoter Edit Flow 与 rollout-guided multi-objective utility | plant promoter 而非 UTR；任务与 evaluator 不同 | `REGULATORY_EDITFLOW_PRIOR_ART` |
| `P-UTR-01` | [UTailoR](https://pubmed.ncbi.nlm.nih.gov/41069846/) | 给定 5′UTR 的 source-proximal 优化/生成 | 无显式 CTMC INS/DEL/STOP、无 3′任务 | `CLOSEST_5_SOURCE_NATIVE` |
| `P-UTR-02` | [PARADE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722239/) | cell-type conditional 5′/3′UTR 设计；含 diffusion、GA、random 与 motif routes | 主要为 de novo，不是 source-relative Edit Flow | `CLOSEST_5_3_NATIVE` |
| `P-UTR-03` | [RNAdiffusion](https://arxiv.org/abs/2409.09828) | 可变长 RNA latent diffusion；5′UTR MRL/TE guidance | 5′为主；无显式 source-relative edit trajectory | `5_DIFFUSION_NATIVE` |
| `P-UTR-04` | [GARDN/SANDSTORM](https://www.nature.com/articles/s41467-025-59389-8) | predictor-guided generative RNA design，包括 5′UTR 相关任务 | GAN/activation-optimization 路线，非 source-conditioned CTMC | `5_GENERATIVE_NATIVE` |
| `P-UTR-05` | [Smart5UTR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10985129/) | 5′UTR predictor 与 autoencoder/generative design | 特定长度/修饰域；无 3′、CTMC 或显式编辑轨迹 | `5_AUTOENCODER_NATIVE` |
| `P-UTR-06` | [Optimus 5-Prime + GA](https://www.nature.com/articles/s41587-019-0164-5) | 5′UTR MRL predictor 加 genetic algorithm design | predictor+search，不是 amortized Edit Flow | `MANDATORY_5_SEARCH` |
| `P-UTR-07` | [GEMORNA](https://pubmed.ncbi.nlm.nih.gov/40875799/) | 5′UTR、3′UTR、CDS 分模块生成并组装 mRNA | de novo Transformer，不是 source-relative editing | `5_3_DE_NOVO_NATIVE` |
| `P-UTR-08` | [mRNAutilus](https://arxiv.org/abs/2605.31296) | masked discrete diffusion 与 tree guidance 的全长 mRNA 多目标生成 | 全长、de novo、预印本；当前 UTR-only 主任务不匹配 | `EMERGING_REFERENCE` |

上表的 `MB0 身份` 不是可执行资格。是否进入数值主表，仍需通过代码、checkpoint、license、exposure、接口和预算 Gate。无可执行代码的方法仍是强制 prior art，不得为了保持创新性而从 related work 删除。

### 36A.2.1 三个 2026 直接先例的事实、推断与代码状态

必须将 paper fact、项目比较推断和当前代码资格分开：

```yaml
Flexible_Flows:
  primary_source: https://arxiv.org/abs/2606.10543
  version: arXiv:2606.10543v1
  doi: 10.48550/arXiv.2606.10543
  status: preprint
  paper_fact:
    - structured coupling
    - latent edit-based rate parameterization
    - Dirichlet operation control
    - latent/rate guidance
    - evaluation on unconditional/conditional DNA and MHC-peptide generation
  not_established_by_paper:
    - given-template source-relative UTR editing
  code_status_2026_07_30: NOT_VERIFIED_PUBLIC

pCoMole:
  primary_source: https://openreview.net/forum?id=tTILzscPs4
  status: ICLR_2026_DeLTa_workshop_poster
  paper_fact:
    - pretrained Edit Flow
    - augmented Tchebycheff utility
    - Doob-h transform and short Monte Carlo rollout
    - hard terminal feasibility
    - GFP/Cas9/peptide compression
  not_established_by_paper:
    - stepwise UTR grammar constraint
  code_status_2026_07_30: NOT_VERIFIED_PUBLIC

SPROUT:
  primary_source: https://openreview.net/forum?id=4AF7WSp7Cs
  status: GenBio_2026_poster
  paper_fact:
    - plant-promoter Edit Flow
    - rollout-guided multi-objective oracle
    - reported failure to constrain unobserved expression axes when oracles cover only two conditions
  not_established_by_paper:
    - complete context constraint
    - UTR source-edit task
  code_status_2026_07_30: NOT_VERIFIED_PUBLIC
```

在本次 snapshot 中，三者当前统一为：

```text
MANDATORY_PRIOR_ART
+ REFERENCE_ONLY
```

只有后续核实官方 repository、checkpoint、license 并完成适配审计后，才能转为 `CONDITIONAL_METHOD_COMPARATOR`。禁止宽泛 first claim 是本项目依据这些论文与任务矩阵作出的比较推断，不是这些论文自身的事实声明。

本项目不得再写“首个生物序列 Edit Flow”“首个 source-conditioned Edit Flow”或“首个约束/多目标 Edit Flow”。若投稿前增量检索仍支持，可使用更窄的事实描述：

> 在冻结检索范围内，尚未发现把 source anchoring、逐步 UTR grammar hard mask、可变长 INS/SUB/DEL/explicit STOP、5′/3′ region-aware 统一建模和 independent matched-budget evaluation 同时纳入一个方法的公开工作。

## 36A.3 正式任务矩阵

| Task ID | Track | 输入与输出 | 主要评测 | 允许形成的结论 |
|---|---|---|---|---|
| `MB-T1-5/3` | B：held-out generative | source/corruption 或 paired source、region、condition、budget → target/candidate distribution | held-out flow objective、exact/near recovery、edit distance、length/action calibration、validity、diversity | generative mechanism evidence；未验证归一化时不得称 exact NLL |
| `MB-T2-5/3` | A：closed measured pool | 相同 source 与隐藏 label 的固定 candidate pool → 排名/选择 \(N\) 个候选 | NDCG、enrichment、Hit@N、observed-pool normalized regret、source-macro paired CI | measured-pool retrospective evidence |
| `MB-T3-5/3` | C：open legal generation | source、region、context、target、constraints、budget → \(N\) 次 proposal | hard validity、unique/diversity、source preservation、edit cost、condition response、independent evaluator、Pareto hypervolume | open-support computational prediction；不是实验改善 |
| `MB-T4-5/3` | cross-track efficiency | 与 T1–T3 相同 | candidate/query/compute curves、latency、GPU-hours、VRAM、amortization break-even | efficiency evidence |
| `MB-NATIVE-5/3` | published-native | 各论文原生输入与原生输出 | 原论文任务复现与审计 | 文献/系统背景；不得与不同任务主表直接定胜负 |

其中：

- `MB-T2` 的 exhaustive/rank-all 只适用于固定、有限 candidate universe；
- `MB-T3` 不能因 independent evaluator 分数高就写成真实功能提高；
- published paper numbers 不得直接与本项目新数据的结果做统计比较；
- 只有在共同 source、split、constraint、budget、evaluator 上重新运行，才属于正式横向比较；
- 5′与3′始终是独立 Task ID、独立统计检验与独立结果表。

## 36A.4 强制 exact-interface core baselines

下列 baseline 必须在 `baseline_registry_v2.yaml` 中逐项出现；状态只能是 executable、blocked-with-evidence 或 formally-inapplicable，不得静默遗漏：

```text
B00_IDENTITY_NO_EDIT
B01_LEGAL_RANDOM_UNIFORM
B02_LEGAL_RANDOM_EMPIRICAL_PROPOSAL
B03_NEAREST_NEIGHBOR_OR_RETRIEVAL
B04_DIRECT_SCORER_TOPN_OR_RANKALL        # Track A only
B05_DIRECT_SCORER_GREEDY
B06_DIRECT_SCORER_BEAM
B07_DIRECT_SCORER_EVOLUTION_OR_SA
B08_BEST_OF_N_CONDITIONAL_PRIOR
B09_SOURCE_CONDITIONED_AR_EDIT
B10_SOURCE_CONDITIONED_MASKED_DISCRETE_DIFFUSION
B10A_SOURCE_CONDITIONED_FIXED_LENGTH_MASKED_DFM_CONTROL
B11_GENERIC_ORIGINAL_EDIT_FLOW
B12_FULL_UTR_MRNA_EDITFLOW
```

`B12_FULL_UTR_MRNA_EDITFLOW` 中的 `FULL` 表示“当前 UTR 方法的全部已冻结组件开启”，**不表示 full-length mRNA**。

Core controlled comparison 要求：

- 尽可能共享同一 foundation encoder/checkpoint；
- 共享训练数据、split、condition、hard legality、edit budget 与候选输出 schema；
- AR、真正的 masked/discrete diffusion 与 Edit Flow 的 trainable parameter 数和调参预算处于预注册容差内；fixed-length DFM 不能替代 diffusion comparator；
- 所有 search 方法共享同一 generation-time frozen scorer；final evaluator 对训练、搜索、guidance、reranking 和 threshold selection 均不可查询，只在 candidate freeze 后运行；
- “strongest direct scorer”必须由预先冻结的 scorer ladder 在 development set 上确定，不能看 final label 后指定；
- generic Edit Flow 必须尽可能复现原始 objective/sampler，不能故意使用残缺版本；
- no-edit、random 与 retrieval 是 sanity/lower-bound，不得冒充 strongest comparator；
- `NO_GRAMMAR_MASK` 只能作为隔离消融运行，非法候选仍计入失败和预算，不能进入生物候选库。

强制 Edit Flow 消融至少包括：

```text
SUB_ONLY
FIXED_LENGTH
NO_CONTINUOUS_TIME
NO_SOURCE_ANCHOR
NO_REGION_ADAPTER
NO_EXPLICIT_STOP
NO_EDIT_COST
NO_FOUNDATION_INIT
SINGLE_BEST_OUTPUT
```

## 36A.5 Published-native、adapted 与 method-prior comparator

公开方法不能只按论文名字进入主表，必须先获得唯一状态：

```text
NATIVE_EXECUTABLE
ADAPTED_EXECUTABLE
EXPOSED_NATIVE_REFERENCE
REFERENCE_ONLY
INELIGIBLE
```

含义如下：

- `NATIVE_EXECUTABLE`：官方代码/权重与许可可用，原生科学接口无需本质改变；
- `ADAPTED_EXECUTABLE`：加入 source initialization、retraining、condition/length wrapper 或 grammar adapter；结果必须写成 `adapted X`；
- `EXPOSED_NATIVE_REFERENCE`：训练数据、task head 或 predictor 已暴露本项目 final 数据/标签；
- `REFERENCE_ONLY`：只有论文数字、无可恢复实现，或任务不能合法映射；
- `INELIGIBLE`：许可、依赖、输出、final-label requirement 或本质性改造导致不可比。

每个方法必须记录：

```yaml
model_id:
paper_title:
paper_version_date:
peer_review_status:
official_paper_url:
official_code_url:
code_commit:
license:
checkpoint_url:
checkpoint_sha256:
native_region:
native_task:
native_input:
native_output:
native_length_domain:
native_action_space:
training_data_ids:
pretraining_data_known:
final_dataset_overlap:
required_oracles:
adapter_description:
executable_status:
ineligibility_reason:
```

硬规则：

1. source-agnostic de novo 模型不能因生成后按 edit distance 过滤，就改称 source-conditioned；被过滤候选仍消耗预算；
2. 固定长度模型不能用 padding 伪装 INS/DEL；
3. 换 predictor、加 source embedding 或 grammar wrapper 后必须标为 adapted；
4. native 与 adapted 必须分行、分目录、分结果，不得择优合并；
5. final dataset overlap 未核实时 fail closed；
6. 无代码时不得凭论文描述伪造一个“官方模型”；独立重实现必须标 `independent_reimplementation`；
7. 只支持 5′的方法不能充当 3′ headline baseline；
8. 缺 published-native executable 不阻断 core controlled comparison，但阻断“优于该已发表模型”和“SOTA”主张。

5′ published candidate 至少审计 UTailoR、Optimus+GA、Smart5UTR、PARADE、RNAdiffusion、GARDN/SANDSTORM、GEMORNA。3′至少审计 PARADE、GEMORNA，以及 D0 新发现的可执行 3′ predictor+search。EvoFlows、Flexible Flows、pCoMole、SPROUT 必须进入 prior-art registry；若官方实现可合法、无语义扭曲地适配，进入 method comparator，否则为 reference-only。

## 36A.6 Foundation model 归因必须分三条轨

```text
NATIVE_REPRODUCTION
    各论文使用自己的原生任务、数据、backbone/checkpoint
    只回答能否复现；禁止产生跨方法 winner

COMMON_TASK_SYSTEM
    各系统允许使用自己的原生 backbone/checkpoint
    但必须共享 source、split、constraint、evaluator 与 matched budget
    只回答完整系统在共同任务上谁更好

ARCH_CONTROLLED
    同一 foundation backbone、数据、trainable parameters、training
    updates/tokens、新增 GPU compute 与 tuning-trial budget
    回答 Edit Flow 架构本身是否带来优势
```

`NATIVE_REPRODUCTION` 结果不得跨不同原生任务排序。只在 `COMMON_TASK_SYSTEM` 上获胜，只能声称系统级优势；不得把 foundation pretraining、额外数据或更大 backbone 的收益归因给 Edit Flow。只有 `ARCH_CONTROLLED` 的共同任务比较可支持架构归因。

所有方法记录：

- foundation checkpoint/hash；
- known pretraining corpus 与 exposure；
- frozen/LoRA/partial/full fine-tuning；
- total 与 trainable parameters；
- 本项目内 training GPU-hours 与 tuning trial 数；
- 外部预训练成本。

未知外部预训练成本写：

```text
UNKNOWN_EXTERNAL_PRETRAINING_COST
```

不得填零。

## 36A.7 Matched-budget 的精确定义

每个正式结果至少同时报告：

```text
B_N  candidate attempts, valid, unique, duplicate, invalid
B_Q  predictor/reward/critic/rollout/reranking queries
B_E  INS/SUB/DEL counts, cumulative actions, final edit distance
B_C  sampler steps, NFE, GPU-seconds, wall-clock, peak VRAM
B_T  training examples/tokens, epochs, trainable params, GPU-hours, tuning trials
```

Primary matched comparison 固定相同的 candidate-attempt budget \(N\)、oracle-query budget \(Q\) 与 edit budget；无法精确匹配计算架构时，使用预冻结网格报告完整 quality–budget curve 和 Pareto frontier，不能挑单个有利点。

每条 run 至少记录：

```yaml
task_id:
region:
support_track:
model_id:
native_or_adapted:
source_manifest_sha256:
split_manifest_sha256:
evaluator_manifest_sha256:
foundation_checkpoint_sha256:
grammar_sha256:
edit_budget:
candidate_attempts:
valid_candidates:
unique_candidates:
duplicate_candidates:
invalid_candidates:
oracle_queries_by_type:
sampler_steps:
nfe:
total_params:
trainable_params:
training_examples:
training_tokens:
training_gpu_hours:
tuning_trials:
inference_gpu_seconds:
wall_time:
peak_vram:
device:
scheduled_check_interval:
manual_check_events:
alert_triggered_exceptions:
parallel_task_ids:
resource_conflict_audit:
seed:
final_label_access: false
candidate_freeze_sha256:
```

额外硬规则：

- invalid、duplicate、timeout、numerical failure 和 decoder failure 均消耗 attempt 与 compute budget，不得免费补样；
- 所有 scorer/guidance/selection queries 都计入 \(Q\)；
- generator 使用的 reward/critic 不能作为唯一 final evaluator；
- 所有方法在 final label reveal 前冻结 candidates 与 hashes；
- hyperparameter tuning trial budget 必须匹配或显式报告不匹配；
- published-native 的既有湿实验数字只能作背景，不替代共同任务评测；
- 当前没有新增湿实验，open-support 一律标 `computational prediction`。

## 36A.8 5′UTR 与 3′UTR 分开规则

- `MB-*-5` 与 `MB-*-3` 使用独立 data/split/endpoint/evaluator/statistical manifest；
- 5′不能因 3′公开 baseline 少而降低自己的标准，反之亦然；
- 只原生支持 5′的方法不能通过 padding、序列反转或替换 endpoint 伪装成 3′ baseline；
- 两区不得先池化再检验；macro-average 只能是次级汇总；
- 两个 region 的 primary hypotheses 执行预注册 multiplicity correction；
- 只有 5′与3′分别通过时，才能写 `UTR-general`；
- 单区通过只形成 region-specific evidence，另一地区失败原样报告；
- 不得用 5′ final labels 选择 3′模型，也不得反向使用。

## 36A.9 MB0 Freeze Gate 与结果状态

在任何 final run 前，必须冻结：

```text
task matrix
prior-art snapshot
baseline registry and eligibility
source/split/grammar/evaluator manifests
candidate/query/edit/compute budget grid
foundation assignment
hyperparameter tuning budget
primary metrics and Pareto definition
statistical analysis and multiplicity
invalid/duplicate/timeout penalties
candidate freeze procedure
```

正式比较：

- 至少 5 个独立训练/生成种子，按合同区分；
- source-group paired comparison；
- source-group cluster bootstrap 95% CI；
- 与每个 task/region 中表现最强的 eligible matched-budget baseline 比较，而非 baseline 平均值；
- hard grammar validity 必须为 100%；
- architecture claim 只能来自 `ARCH_CONTROLLED`；
- efficiency claim 必须同时报告质量非劣界与 query/GPU/latency 节省；
- final labels 只能在 candidate freeze 后揭示。

每个 `region × primary track` 必须在 final labels 不可见时冻结可计算的判定对象：

```yaml
primary_track:
primary_statistic_id:
statistic_definition:
direction: higher_is_better
effect_difference: metric_ours_minus_metric_baseline
baseline_comparison_rule:
  mode: dev_selected_fixed | simultaneous_all_eligible
superiority_margin_delta_sup:
noninferiority_margin_delta_ni:
resource_metric_id:
resource_improvement_margin_delta_eff:
confidence_level:
simultaneous_inference_method:
multiplicity_family:
```

“表现最强 baseline”不得用 final point estimate 事后挑选。只允许：

1. 在 development set 上按预注册规则选定一个 baseline 并永久冻结；或
2. 对全部 eligible baselines 做 simultaneous inference，并要求对每一个 comparator 都通过。

对 comparison track \(k\in\{\mathrm{COMMON\_TASK\_SYSTEM},\mathrm{ARCH\_CONTROLLED}\}\) 与 region \(r\)，定义 higher-is-better primary effect：

\[
\Delta_{k,r,b}
=
M_{k,r}(\mathrm{mRNA\text{-}EditFlow})-M_{k,r}(b).
\]

`PASS_track_region(k,r)` 的必要且充分合同为：

```text
mandatory comparator coverage = complete
candidate/final-label protocol = PASS
hard grammar validity = 100%
simultaneous 95% CI lower bound for primary Δ > frozen δ_sup
all preregistered fatal guards = PASS
```

若使用 `simultaneous_all_eligible`，上述 CI 条件必须对每个 eligible comparator 成立；若使用 `dev_selected_fixed`，只对冻结的 strongest comparator 推断，但仍完整报告其他 baseline。

效率路径单独定义：

```text
quality lower CI > -δ_ni
AND resource-improvement lower CI > δ_eff
```

它只授予 `EFFICIENCY_SUPPORTED_<track>_<region>`；除非在 final labels 不可见时把效率质量–资源联合统计量预注册为唯一 primary statistic，否则不能替代 `PASS_track_region` 的质量 superiority 条件。T1/T2/T3 的统计量不得事后池化换取 PASS。

进一步定义：

```text
PASS_system_region(r)
    = PASS_track_region(COMMON_TASK_SYSTEM, r)

PASS_arch_region(r)
    = PASS_track_region(ARCH_CONTROLLED, r)
```

允许的终态：

```text
MB0_PASS_5_AND_3
MB0_PASS_5_ONLY
MB0_PASS_3_ONLY
MB0_SYSTEM_ONLY
MB0_NOT_ESTABLISHED
MB0_BLOCKED_COMPARATOR
```

其中 `MB0_NOT_ESTABLISHED` 是合法负结果，不得改任务或删除 baseline；`MB0_BLOCKED_COMPARATOR` 表示强制 core comparator 不完整，禁止最终 superiority/SOTA 主张。

终态必须按以下互斥优先级判定：

```text
Priority 1:
mandatory comparator incomplete for the intended claim
    → MB0_BLOCKED_COMPARATOR

Priority 2:
PASS_arch_region(5′) and PASS_arch_region(3′)
    → MB0_PASS_5_AND_3
PASS_arch_region(5′) only
    → MB0_PASS_5_ONLY
PASS_arch_region(3′) only
    → MB0_PASS_3_ONLY

Priority 3:
no architecture PASS, but PASS_system_region for at least one region
    → MB0_SYSTEM_ONLY
      + system_supported_regions: [explicit list]

Priority 4:
none of the preregistered system or architecture PASS conditions met
    → MB0_NOT_ESTABLISHED
```

当 `MB0_BLOCKED_COMPARATOR` 触发时，仍可报告已完成 region/track 的 development evidence，但全局终态不得同时标为任何 PASS 或 `NOT_ESTABLISHED`。`MB0_PASS_*` 专指 `ARCH_CONTROLLED`；系统级通过但 architecture 未通过只能使用 `MB0_SYSTEM_ONLY`。

## 36A.10 分阶段 Todo

### MB0-A 先例冻结

- [ ] 冻结检索式、数据库、日期、版本、去重和纳入/排除理由；
- [ ] 建立 complete-task quantifier 与 task-similarity matrix；
- [ ] 纳入 36A.2 全部方法及新发现方法；
- [ ] 投稿前执行增量检索和 novelty wording audit。

### MB0-B 代码、许可与 exposure

- [ ] 核验官方代码、commit、license、checkpoint 与 batch interface；
- [ ] 审计 sequence/source/study/pretraining/final-label overlap；
- [ ] 赋予唯一 eligibility status；
- [ ] 保存 blocked baseline 的环境、命令、错误和可复现日志。

### MB0-C 统一接口

- [ ] 输入统一 source FASTA、region、condition、seed、edit/candidate/query budget；
- [ ] 输出统一 candidate、trajectory、invalid reason、oracle-query 与 resource schema；
- [ ] native、adapted、independent reimplementation 分目录；
- [ ] source-edit、de novo native 与 measured-pool 不混表。

### MB0-D Budget/evaluator/statistics freeze

- [ ] 冻结 source、split、grammar、endpoint、final evaluator；
- [ ] 冻结 \(B_N/B_Q/B_E/B_C/B_T\) 网格；
- [ ] 冻结 primary statistic、direction、\(\delta_{\mathrm{sup}}\)、\(\delta_{\mathrm{ni}}\)、\(\delta_{\mathrm{eff}}\)；
- [ ] 冻结 dev-selected 或 simultaneous-all baseline comparison rule；
- [ ] 冻结 CI、multiplicity、non-inferiority 与 failure scoring；
- [ ] 测试 final evaluator 对 generator/guidance/selection 不可见。

### MB0-E GPU smoke 与正式运行

- [ ] 神经网络训练、fine-tuning 和可 GPU 化生成必须使用 GPU；
- [ ] 每个 baseline 先执行小型 smoke，验证 I/O、预算账本与 deterministic seed；
- [ ] smoke 结果不得进入论文主表；
- [ ] 正式 run 不因早期结果更换 baseline；
- [ ] 使用低频、validation-boundary/event-driven 监控；
- [ ] 等待期间并行做 adapter、exposure、统计脚本和报告模板，不争抢同一 GPU。
- [ ] manifest 记录 scheduled check interval、manual checks、alert exceptions 与 parallel task IDs；
- [ ] 完成 parallel-task resource-conflict audit，证明未争抢正式 run 的 GPU/IO。

### MB0-F Candidate freeze 与统一评测

- [ ] 每个方法先冻结 candidates、trajectories 与 hashes；
- [ ] freeze 后才允许 final evaluator/measured label reveal；
- [ ] invalid、duplicate、timeout 不得补样；
- [ ] 统一计算 paired metrics、budget curves、Pareto 与 break-even。

### MB0-G 报告与 claim audit

- [ ] 独立报告 5′、3′；
- [ ] 独立报告 measured、held-out-generative、open-support、published-native；
- [ ] 分开 system 与 architecture advantage；
- [ ] claim ledger 标为 `SUPPORTED / NOT_ESTABLISHED / FORBIDDEN`；
- [ ] 负结果、暴露、缺代码和执行失败进入正文或补充材料。

## 36A.11 强制 Artifacts

```text
artifacts/phase_mb0/prior_art/search_protocol.md
artifacts/phase_mb0/prior_art/search_snapshot.json
artifacts/phase_mb0/prior_art/model_registry.tsv
artifacts/phase_mb0/prior_art/task_similarity_matrix.tsv
artifacts/phase_mb0/eligibility/model_eligibility.json
artifacts/phase_mb0/eligibility/exposure_audit.tsv
artifacts/phase_mb0/contracts/task_matrix.json
artifacts/phase_mb0/contracts/source_manifest.json
artifacts/phase_mb0/contracts/split_manifest.json
artifacts/phase_mb0/contracts/grammar_manifest.json
artifacts/phase_mb0/contracts/evaluator_manifest.json
artifacts/phase_mb0/contracts/budget_contract.yaml
artifacts/phase_mb0/contracts/statistical_analysis_plan.md
artifacts/phase_mb0/adapters/<model_id>/adapter_spec.md
artifacts/phase_mb0/adapters/<model_id>/environment.lock
artifacts/phase_mb0/smoke/<model_id>/<region>/smoke_manifest.json
artifacts/phase_mb0/runs/run_registry.jsonl
artifacts/phase_mb0/runs/<task_id>/<model_id>/<seed>/manifest.json
artifacts/phase_mb0/runs/<task_id>/<model_id>/<seed>/candidates.fasta
artifacts/phase_mb0/runs/<task_id>/<model_id>/<seed>/trajectories.jsonl
artifacts/phase_mb0/runs/<task_id>/<model_id>/<seed>/metrics.json
artifacts/phase_mb0/runs/<task_id>/<model_id>/<seed>/system_metrics.jsonl
artifacts/phase_mb0/freeze/candidate_freeze_manifest.json
artifacts/phase_mb0/analysis/paired_metrics.parquet
artifacts/phase_mb0/analysis/budget_curves.parquet
artifacts/phase_mb0/analysis/statistical_results.json
artifacts/phase_mb0/analysis/pareto_frontiers.json
artifacts/phase_mb0/reports/MB0_REPORT.md
artifacts/phase_mb0/reports/CLAIM_LEDGER.md
```

任何关键 artifact 缺失，相应结果为 `UNVERIFIED`。

## 36A.12 论文 Claim 边界

若相应 Gate 通过，可以写：

- 在冻结的 source-conditioned UTR editing task 与 matched budget 下，mRNA-EditFlow 优于所比较的最强可执行 baseline；
- Edit Flow 在相同 foundation/data/budget 下有架构级优势——仅限 `ARCH_CONTROLLED` 通过；
- 冻结检索未发现满足全部任务量词的既有模型；
- 5′UTR-specific 或 3′UTR-specific——只按各区结果；
- measured-support retrospective evidence 与 open-support computational prediction——严格分开。

无论结果多好，均不得写：

- 首个 UTR 生成模型、生物序列 Edit Flow、source-conditioned Edit Flow 或 constrained Edit Flow；
- 用不同 assay、数据、split 或论文原生数字宣称击败 PARADE/GEMORNA/UTailoR；
- 把 foundation model 或额外数据优势归因于 Edit Flow；
- 把 generator reward model 当作 independent evaluator；
- 把计算预测写成实验验证；
- 只有一个 region 通过时写 UTR-general；
- comparator registry 不完整时写 UTR design SOTA。

本增补最重要的区分是：

```text
“没有发现完全同任务模型”目前有条件成立；
“没有接近先例或可比 baseline”不成立。
```

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

- 写 `utr_editflow_contract_v2.yaml`；
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

## 40.4 C0/D0 通过后的近程 Goal 与后续依赖

第 40.1–40.3 节在其 Gate 完成前继续有效。其通过后必须区分：

```text
early MB0 design / preregistration
≠
final MB0-Freeze
```

近程目标是：

> **先完成 D1/B0/FM0 的真实依赖；在 FM0 通过后完成 MK0 数学闭环；同时只开展 early MB0 design，冻结任务量词、registry/interface schema、预算记账规则和 evaluator-isolation protocol。完整 MB0-Freeze 必须等待 FC0/ME0 的 final evaluator、source/split/grammar 和 candidate-freeze procedure 就绪。**

严格顺序：

```text
1. 只读确认 C0/D0 的真实 Gate 与 artifacts
2. 完成 D1/B0/FM0；其间并行刷新 Edit Flow/UTR comparator 快照
3. early MB0 design：task quantifier、registry schema、adapter schema、budget ledger schema
4. FM0 PASS 后写完 MK0 ADR、schema、公式推导和 candidate-generation configuration schema
5. 完成 tiny-state oracle、alignment/path、hazard/loss/STOP tests
6. 在真实 GPU 上做四类动作的 forward/backward/sampler tiny smoke并冻结 MK0 hash
7. 依次完成 EF0/GP0/FC0/ME0 所需训练与 candidate-generation 准备
8. ME0 与 final-evaluator isolation PASS 后才执行 final MB0-Freeze
9. final MB0-Freeze 后才运行 MB0 formal comparison 和 final-label evaluation
```

Early MB0 阶段不得声称：

```text
baseline eligibility 已最终冻结
source/split/evaluator 已最终冻结
candidates 已冻结
MB0-Freeze 已通过
```

该阶段的允许并行：

- GPU smoke 等待期间进行文献、license、exposure、adapter schema、统计脚本和报告模板工作；
- 不同时启动争抢同一 GPU 的无关训练；
- 训练/神经 forward/backward 必须使用 GPU；
- 监控以 validation boundary、checkpoint、异常事件为主，不得高频轮询。

若 MK0 或 early MB0 design 失败，只能在相同科学问题、任务量词和 Gate 下修复并使用新 run/task ID 前进；不得退回 predictor-only、de novo-only、SUB-only 主项目，也不得删除强 baseline、失败证据或 3′UTR 任务。后续 final MB0-Freeze 失败时遵守 36A 的相同 forward-only 规则。

## 40.5 近程完成标准

```text
D1/B0/FM0 required dependencies = PASS
original Edit Flows vs project extension matrix = frozen
state/action/coupling/probability-path semantics = frozen
rate/Bregman/STOP objective = frozen
sampler approximation and convergence contract = frozen
foundation/current-state fusion = frozen
critic/final-evaluator separation protocol = defined
tiny-state mathematical oracle = PASS
GPU INS/SUB/DEL/STOP tiny smoke = PASS_WITH_ARTIFACTS
GPU run_id and GPU UUID = recorded
CPU fallback count = 0
local metrics/events logs = present
run manifest and hashes = present
failure bundle = present when failed
prior-art snapshot = complete for frozen search
exact-task vs adjacent-task distinction = explicit
mandatory core baseline registry schema = complete
published comparator eligibility schema = complete
matched-budget task/interface/accounting schema = frozen
final MB0-Freeze status = NOT_YET_ELIGIBLE_UNTIL_ME0
final labels accessed = false
```

GPU smoke 的运行证据必须服从第 18–21 节的 artifact、manifest、日志、资源和失败语义合同；截图或口头 PASS 无效。完整 MB0-Freeze 的完成标准仍以 36A.9–36A.11 为唯一权威，不得用本节的 early schema 代替。

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

## MK0（本次新增强制 Gate）

- [ ] original Edit Flows 版本、公式、实现 hash
- [ ] original vs project-extension matrix
- [ ] state/action/budget/termination schemas
- [ ] current-coordinate INS/SUB/DEL/STOP apply/replay
- [ ] source-current alignment mapping
- [ ] canonical optimal alignment
- [ ] equivalent optimal-alignment sensitivity
- [ ] edit-order sensitivity
- [ ] coupling/rejected-path ledger
- [ ] time direction
- [ ] cubic schedule reference
- [ ] linear schedule sensitivity
- [ ] endpoint/time-eps/clip policy
- [ ] factorized operation/token rates
- [ ] hard mask before event normalization
- [ ] total hazard/generator row-sum tests
- [ ] Bregman loss derivation
- [ ] brute-force loss/gradient oracle
- [ ] action-to-next-state transition aggregation
- [ ] repeated-symbol edit ambiguity oracle
- [ ] survival-hazard STOP primary candidate
- [ ] absolute STOP hazard/time calibration
- [ ] post-completion exponential dwell and gamma sensitivity
- [ ] administrative-censor fraction oracle
- [ ] no simultaneous final-edit/STOP or step-0 STOP atom
- [ ] competing-risk event-type ablation
- [ ] learned vs forced termination audit
- [ ] paper-compatible first-order reference sampler
- [ ] parallel joint-event legality audit
- [ ] constrained single-event first-order primary sampler
- [ ] deterministic replay
- [ ] step-halving convergence
- [ ] numerical event-time reference
- [ ] source cache + dynamic-current encoding
- [ ] target-alignment leakage test
- [ ] full/incremental encoding equivalence
- [ ] no-critic base sampler
- [ ] final-evaluator isolation
- [ ] real-GPU forward/backward tiny smoke
- [ ] GPU execution of INS/SUB/DEL/STOP
- [ ] MK0 freeze manifest/hash

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
- [ ] true source-conditioned masked/discrete diffusion comparator
- [ ] fixed-length masked DFM kept separate from diffusion
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
- [ ] frozen complete-task quantifier
- [ ] prior-art search protocol/snapshot
- [ ] EvoFlows/Flexible Flows/pCoMole/SPROUT audit
- [ ] UTailoR/PARADE/RNAdiffusion/GARDN/Smart5UTR/Optimus/GEMORNA audit
- [ ] native vs adapted vs reference-only status
- [ ] official code/checkpoint/license verification
- [ ] comparator exposure audit
- [ ] exact-interface B00–B12 registry
- [ ] task matrix T1–T4 for 5′ and 3′
- [ ] native-reproduction track without cross-method winner
- [ ] common-task system track
- [ ] architecture-controlled track
- [ ] candidate-attempt budget
- [ ] oracle-query budget
- [ ] cumulative-edit budget
- [ ] training/tuning budget
- [ ] invalid/duplicate/timeout budget accounting
- [ ] common candidate/trajectory/resource schema
- [ ] final-evaluator isolation test
- [ ] candidate freeze per method/seed
- [ ] strongest eligible baseline resolved on development only
- [ ] region-specific result state
- [ ] claim ledger

本 `## MB0` 清单仅为导航索引；MB0 完成的唯一判据是 36A.10 中 `MB0-A` 至 `MB0-G` 全部完成，并满足 36A.9 Gate 与 36A.11 artifacts。不得只勾选本汇总表后宣布 MB0 PASS。

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

## 43.1 本次增补的直接方法与 baseline 证据

- [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018)
- [EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](https://arxiv.org/abs/2603.11703)
- [Flexible Flows for Biological Sequence Design](https://arxiv.org/abs/2606.10543)
- [pCoMole: Pareto-Constrained Molecule Editing with Discrete Flows](https://openreview.net/forum?id=tTILzscPs4)
- [SPROUT: Steered Plant Promoter Editing via Rollout-Guided Utility Tilting of Edit Flows](https://openreview.net/forum?id=4AF7WSp7Cs)
- [UTailoR](https://pubmed.ncbi.nlm.nih.gov/41069846/)
- [PARADE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11722239/)
- [PARADE official repository](https://github.com/autosome-ru/parade)
- [RNAdiffusion](https://arxiv.org/abs/2409.09828)
- [GARDN/SANDSTORM official repository](https://github.com/AlexGreenLab/GARDN-SANDSTORM)
- [Smart5UTR official repository](https://github.com/deepomicslab/Smart5UTR)
- [Optimus 5-Prime official repository](https://github.com/pjsample/human_5utr_modeling)
- [GEMORNA official repository](https://github.com/RainaBio/GEMORNA)
- [mRNAutilus](https://arxiv.org/abs/2605.31296)

截至 2026-07-30，本次从 Flexible Flows、pCoMole、SPROUT 的 primary paper/OpenReview 页面未核实到官方公开 repository/checkpoint；三者代码状态固定为 `NOT_VERIFIED_PUBLIC`，当前只能是 `MANDATORY_PRIOR_ART + REFERENCE_ONLY`。这是一项可在后续资格审计中更新的代码事实，不代表论文方法不存在或不可重实现。

这些链接只证明论文/项目定位和后续资格审计入口，不自动证明代码可运行、license 合法、checkpoint 可得、数据无暴露或任务可直接比较。可执行资格只能由 36A.5 的 frozen registry 与运行证据授予。

---

# 43A. 2026-07-30 加法增补决策记录

```yaml
decision_id: DEC-UTR-EF-V2-20260730-MATH-MB0
date: 2026-07-30
change_type: additive_only
base_contract: utr_editflow_goal_v2
amendment_id: utr_editflow_goal_v2.1_additive_math_mb0
old_text_deleted: false
user_approval:
  status: explicit
  scope:
    - add unresolved mathematical kernel goals and todo
    - add architecture-figure prompt
    - clarify and strengthen MB0 baselines
reason:
  - EF0 listed rate/loss/sampler components but did not close their shared mathematical semantics
  - explicit STOP is a project extension and was not separated from original Edit Flows
  - MB0 named broad baseline families without a frozen task/interface/eligibility registry
  - recent biological-sequence Edit Flow prior art narrows novelty claims
affected_phases:
  - FM0
  - MK0
  - EF0
  - GP0
  - FC0
  - ME0
  - MB0
  - PP0
required_rerun:
  past_results: false
  future_formal_runs_must_reference_new_hash: true
goal_sha256_location: external task registry and run manifests
goal_sha256_embedding_note: not embedded here because the document hash is self-referential
```

该增补不否定原 Goal 的科学问题；它把原来未最终确定的数学接口变成可执行 Gate，并把“没有完全同任务模型”与“没有可比较先例”明确分开。

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

---

# 45. 2026-07-31 B0 容量诊断重分类决策

```yaml
decision_id: DEC-UTR-EF-V2-20260731-B0-CAPACITY-NONBLOCKING
date: 2026-07-31
amendment_id: utr_editflow_goal_v2.2_b0_capacity_nonblocking
change_type: explicit_user_authorized_contract_amendment
approved_by_user: explicit
affected_phase:
  - B0
old_active_rule:
  - exact path-state capacity findings could block B0 acceptance
  - max_reachable_states=50000 and related path-state limits were treated as B0 production gates
new_active_rule:
  - B0 capacity diagnostics are historical and optional engineering diagnostics only
  - no path-state capacity result is an active B0 acceptance gate
  - B0 acceptance is defined by Section 30.3 plus the required B0 schema, split, leakage, track, and Data Card artifacts
  - physical disk, memory, and process-safety stop rules remain active for every run
historical_evidence:
  - E1 capacity outputs remain immutable FAILED_WITH_EVIDENCE / diagnostic evidence under their original provenance
  - historical evidence is not deleted, relabeled as a pass, or used to claim that the old gate passed
claim_boundary:
  - a B0 pass under this amendment means benchmark/split/track qualification only
  - it does not claim exact enumeration of all dynamic edit-path states
  - it does not establish Edit Flow efficacy, biological improvement, or MB0 superiority
requires_rerun:
  - true
  - fresh B0 task ID, run ID, contract hash, split manifests, leakage audit, track manifests, Data Card, and acceptance record are required
```

本修订仅改变未来 B0 的工程验收边界：容量诊断不再是 B0 的阻断 Gate。Section 30.3 的零泄漏、100% exposure ledger coverage 与 track-role 无歧义门槛，以及 B0-01 至 B0-05 的全部必需产物，均保持不变。

不得借本修订删除既有容量失败日志、修改其时间戳、伪称旧合同已通过，或将新 B0 benchmark qualification 冒充为 Edit Flow 的性能或科学成功。任何未来 B0 运行必须在 manifest 中引用本修订后的合同哈希，并将旧容量证据列为 historical non-blocking diagnostic。
