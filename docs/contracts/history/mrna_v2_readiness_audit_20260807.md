# mRNA-EditFlow V2 就绪度审计与模型修复决策备忘录

日期：2026-08-07  
审计方式：本地合同与交接文档核验；远端仓库、代码、artifact、ordinary 数据与运行状态的只读核验  
安全边界：未读取或消费 GSE246381 sealed analytic labels；未启动训练；未修改远端文件；未影响现有 GPU 任务

## 一、结论

当前项目不是“模型链路全部完成，只差 `sign_accuracy` 最终判定”。更准确的状态是：

```text
ORIGINAL_M0_EFFECT_GATE_FAIL
+ MODEL_IMPLEMENTATION_DEFECTS_CONFIRMED
+ O0/G1_RANKING_POOL_INVALID
+ G1_PUBLICATION_EVIDENCE_NOT_ESTABLISHED
+ SEALED_EVALUATOR_NOT_IMPLEMENTED_DO_NOT_RUN
= BLOCKED_WITH_EVIDENCE / REPAIR_ROUTE_AVAILABLE
```

数据层应单独标为 `DEVELOPMENT_DATASET_ASSEMBLED`，而不是 `PUBLICATION_GRADE_DATA_BENCHMARK_READY`。旧 v3.1 的严格 `five_utr_e_pairs=0 / NOT_VIABLE` 没有被同口径的新证据推翻；60,237 是在较弱的新资格口径下物化出的 5′UTR delta 行，不是 60,237 个已通过严格 grouping/split gate 的独立 pair。

原始合同下，`sign_accuracy≈0.51 < 0.60` 确实意味着 M0 不能 GO；但现有结果还不能证明“模型在方法上最多只能到 0.52”。多数类比例约 0.52 只是 naive baseline，不是理论上限。当前 sign 失败至少受到以下已确认问题影响：

1. candidate-value 模式把 sign loss 错误地训练成 `sign(candidate_value)`，5′UTR target 因而 100% 为正，而不是学习 `sign(delta)`；
2. GSE217518 中约 77.76% 的编辑位置在第 100 位之后，而模型只读取前 100 nt；
3. leave-one-study-out 测试使用了训练中从未更新的 held-out study embedding；
4. 不同 study/endpoint 的数值尺度和零点差异极大，但模型没有可靠的 assay-scale/anchor 处理；
5. 当前 `0.297` 是 context-macro，不是合同要求的 study-macro；正确的两-study macro 约为 0.2525，且没有 group CI 或多种子证据；
6. 当前 `9.92×` enrichment 来自接近零的 signed overall mean 分母，数值病态，不能支持发表级 claim。

因此，当前最合理的路线不是立即降低 sign gate，也不是消费 sealed，而是先修复 ordinary-internal 的任务定义、输入、分组和评估，再用严格的多种子、group-disjoint 方案决定模型是否真正有能力。

## 二、权威文件与运行快照

### 2.1 本地合同与交接

本地合同：

```text
/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 最新合同-v2.md
SHA-256: 9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e
lines: 2608
document status: PROPOSED_PUBLICATION_CONTRACT
document title version: v1.0
```

本地交接：

```text
/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/reports/migration/HANDOVER_V2_READINESS_20260807.md
SHA-256: 858c5418c1adaca9f2e32f6220f449c01837b3b87929843eb2ddaececaf8d841
lines: 146
```

### 2.2 远端 authority 漂移

远端 active authority 是一份 218 行的派生合同，而不是上述 2,608 行文件的 byte-equivalent copy：

```text
/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z/docs/contracts/mrna_xeditflow_goal_v1_1.md
SHA-256: fc9c1c882ef...
status: ACTIVE_AUTHORITATIVE_CONTRACT
```

本地文件名为 v2、文内为 v1.0、交接称其等同 v1.1，而远端真正 active 的 v1.1 内容和字节又不同。正式修复与 sealed 前必须建立唯一 authority，并把本地 V2 SHA、派生规则、可执行 YAML 和远端合同的关系写清楚。

建议做法：保留所有旧文件字节不变，新建版本化 authority（建议 v1.2 或 V2-execution），将本地 2,608 行合同的 SHA 作为 normative source/annex 绑定，并明确所有 outcome-informed amendment。未获用户确认前不应修改权威合同。

### 2.3 Git 与计算资源

实际仓库：

```text
/home/cunyuliu/mrna_editflow_goal/mrna_editflow
main HEAD: 2420dd85466d3fa1d82194a6a6bb0fe67dd396e8
```

迁移 worktree：

```text
/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z
branch: xeditflow-migration-20260806T024650Z
HEAD: bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c
status: ?? runlogs/
```

`44274c2` 是 handover 提交，但不是当前 HEAD。现有 worktree 和主仓库均有需要保留的未跟踪内容。审计时 GPU 0–7 均有活跃任务，不适合立即启动大规模训练，也不应终止无关进程。

此外，migration branch 没有 upstream；数据重建所在主分支提交并不是 migration branch 的祖先，当前迁移实际引用了旁路文件系统产物。`FINAL_MIGRATION_MANIFEST.json` 主要绑定较早 M0–M3 文件，没有完整绑定后续 effect dataset、checkpoint、B0/M4/O0/G1/E0 artifact 与 handover。`artifacts/` 又被 `.gitignore` 排除，因此“artifacts committed alongside”的表述不成立。修复时必须建立跨 branch、数据、代码、checkpoint 与报告的完整 hash manifest。

## 三、对交接核心结论的判定

| 交接结论 | 审计判定 | 原因 |
|---|---|---|
| 模型与开发链路全部就绪，只差最终 sign 判定 | 不正确 | M4 target/loss、截断、held-out context、O0 grouping、G1 baseline/compute、X0 rank、sealed evaluator 均有未解决问题 |
| `sign_accuracy≈0.51` 未达到 0.60 | 正确 | 原始合同 H1、§10.9、M0 均把 sign 写成硬门槛 |
| sign 失败自动等于最严 `NO_GO_EFFECT` | 过度判定 | 合同 §18 最严 NO-GO 是“仅随机 split 有效”；当前应记录 original gate fail，并进入 HOLD/Repair 审计 |
| T5/T9 允许把 5′ sign 降为 secondary | 不正确 | T5 只规定 coverage-risk；T9 只适用于 CDS family ranking；若降级必须正式、前瞻性修订多个合同段落 |
| internal GO 后 sealed 只是确认，可保证 GO | 错误 | sealed 是独立 one-shot external adjudication，仍可 GO/HOLD/NO-GO |
| selective prediction 已被完全证伪 | 证据不足 | 已保存脚本只重现 direct-delta critic 的 logvar frontier；与 formal E0 candval critic 不一致，`|pred|` 等缺机器可读证据 |
| `0.297` 和 `9.92` 表明除 sign 外都已达标 | 不成立 | 0.297 聚合单位错误且仅略过阈值；9.92 指标病态；缺 group CI、五种子、完整强基线 |
| G1 已展示 real-mRNA guidance value | 不成立 | 同 critic 指导并自评、generate-then-rerank 是 no-op、未匹配 compute、只复用最后 fold |
| X0 必须由 sealed GO 解锁 | 合同无此明文 | 合同要求 primary 冻结且 transfer 不反向修改 primary；负迁移/失败分析仍可执行 |
| CDS-B1 缺 per-variant 序列是 blocker | 大体正确 | 但仍应先查 supplementary、作者代码、设计 grammar 和可逆规则，再宣判不可恢复 |

## 四、数据与候选组：当前最大的科学有效性问题

### 4.1 新的“ready”不等于旧 publication-grade gate 已关闭

历史 G7 的主要问题是 source/gene/study split 与真实 grouping atom 不成立。迁移后的 `m4_data_readiness_audit.py` 主要检查 YAML role 和按 pair-id prefix 的数量，并未完整验证 source/gene/study split、pair provenance 或 grouping atom。

因此，新状态只是通过了一个更弱的 migration readiness check，不能据此宣称旧 blocker 已被科学地解决。

当前 staging 的 103,694 pairs 中，biological parent 缺 96,809、gene 缺 62,069、tile family 缺 91,277、transcript 缺 88,161。继承 registry 中，多项现称 ACTIVE 的资产仍保留 `PENDING / NOT_STARTED` 状态且缺 parser commit/config hash。当前 5′UTR 行里 GSE114002 占约 91.38%，仍高于旧发表 gate 的单 study ≤70% 要求；另有 11,632 行 edit count >5，尚未与合同主预算 `[1,3,5]` 对齐。这些不能由“总行数够大”替代。

### 4.2 现有 O0 的 1,630 个 size-2 pool 不是两个候选

只读核验显示，全部 1,630 个“非 singleton source pool”都具有：

- 相同 candidate ID；
- 相同 candidate sequence；
- 相同 edit list；
- 一条 HEK endpoint 记录和一条 SH endpoint 记录。

也就是说，O0 把同一个变体的两个 endpoint 当成两个候选。由此得到的 NDCG/regret 和 G1 measured-pool ranking 不具备原合同要求的“同 source 候选邻域”含义。

当前机器可读 O0 表面结果是：

```text
5U total source IDs: 58,607
singletons: 56,977
non-singleton pools: 1,630
all non-singleton pool sizes: 2
3U non-singleton pools: 0
```

在统一 min-max NDCG 定义下：

```text
random: 0.8125
SparseEditFormer rerank: 0.8340
exact enumeration: 1.0000
```

即便暂不考虑 pool 错误，真实增量约为 +0.0215 NDCG，低于合同 G1 的 +0.05 要求；之前把 0.834 与旧定义下的 random 0.276 比较属于跨口径比较。

### 4.3 ordinary 数据里存在可修复的真正多候选结构

对 GSE114002 55,043 条 ordinary 记录按 `(source_sequence, endpoint)` 重新聚合得到：

```text
source-sequence + endpoint groups: 31,498
groups with >=2 distinct candidate sequences: 12,923
rows in those groups: 36,468
maximum distinct candidates in one group: 40
```

GSE217518 也有 2,880 个不同 source sequence，其中 305 个对应多个 distinct candidates，最大 25；但其两个 endpoint 必须严格分池，不能再次被当作不同候选。

在这 12,923 个候选组中，记录的 `source_value` 一致。这给出了一条高价值修复路线：重建 canonical source group，而不是继续使用由 variant record 机械生成的 singleton `source_id`。

但在正式采用之前，必须验证相同 source sequence 是否确实表示同一 biological source/locus/assay，而不是不同来源偶然具有相同短序列。建议 group key 至少绑定：

```text
source sequence
+ study
+ assay/context
+ endpoint
+ transcript/locus or design family, if recoverable
```

然后以这个 group atom 做 split、listwise loss、bootstrap 和 ranking evaluation。

### 4.4 GSE217518 数值尺度与异常值需要 raw-level 复核

ordinary 聚合显示：

```text
HEK delta: median 0.111, mean -1286.35, SD 59206.61,
           min -2,867,981, max 394,191
SH delta:  median -0.627, mean 66.24, SD 1829.48,
           min -7,946, max 86,212
```

GSE114002 delta 则大致处于 `[-5.8, 5.9]`。把这些 raw candidate values 放进同一个 absolute-value 网络，再用预测 candidate value 减 measured source，会让 unseen study 的零点和 scale 主导 sign。

必须回到原始定义核对：

- endpoint direction 是否一致；
- ratio/log-ratio/normalized value 是否混用；
- source/candidate 聚合和 replicate normalization；
- 极值是合法 biology、分母接近零，还是解析/变换错误；
- 是否存在可靠 replicate SE，可用来定义 near-zero equivalence margin。

### 4.5 GSE145046 是值得优先恢复的 A2 候选

当前 registry 把 GSE145046 标为 `PENDING_BLOCKED / CRITIC_AUX`，核心缺口是 random-oligo read counts 与 endpoint label 的 paper-faithful join 尚未完成，而不是已经证明数据不可用。原始 read-count 文件存在。

建议把它作为独立 D1-A2 qualification 子任务：

1. 恢复 paper/supplement 中的 oligo ID、sequence、condition、replicate 和 label join；
2. 验证方向、单位、过滤和归一化；
3. 构造真实 dense candidate neighborhood；
4. 在 provenance 和重复性达到要求后，才从 CRITIC_AUX 升级为训练候选。

这比继续在当前错误的 singleton grouping 上调模型更可能带来真实信息增益。

## 五、模型实现中的确定性缺陷

### 5.1 candidate-value 与 delta 语义混用

当前 candidate-value 模式使用 `candidate_value` 作为统一的 `y`，再将它同时送入：

- heteroscedastic regression；
- sign BCE；
- pairwise ranking；
- inverse consistency。

这在数学上不一致：

- sign 应预测 `delta > 0`，而不是 `candidate_value > 0`；
- rank 应只比较同一 source、同一 endpoint 的候选 delta；
- absolute candidate property 不应满足 `f(src,cand)+f(cand,src)=0`；
- inverse consistency 只适用于 delta-native antisymmetric head。

建议显式拆分标签与 head：

```text
y_candidate_value
y_source_value
y_delta
y_sign_delta
y_beneficial_delta
source_group_id
```

架构上可并行比较两条可识别路线：

1. delta-native：直接预测 robust relative effect，使用 antisymmetric pair representation；
2. anchored absolute-value：分别预测 source/candidate property，delta 由相同、可校准的 head 做差；absolute head 不施加 inverse consistency。

### 5.2 截断使模型看不到编辑位点

当前 `MAX_SEQ_LEN=100`。GSE217518：

```text
rows: 5,194
median sequence length: 448
sequence length >100: 99.08%
edit position >=100: 77.76%
```

因此模型在大多数 GSE217518 样本中看不到真实变异碱基。修复选项：

- 动态长度 Transformer + padding mask；
- edit-centered local windows，多 edit 逐个编码后聚合；
- local edit encoder + lightweight global context；
- 对长序列使用相对位置或稀疏 attention。

第一轮优先推荐 edit-centered/local+global 路线，计算可控，也直接验证“看见编辑”是否解决 sign。

### 5.3 padding 未 mask

长度 50 的 GSE114002 被补到 100，但 attention 和 pooling 将 padding 当成有效 token。必须引入显式 padding mask，并确保卷积/attention/pooling 都不使用 padding 位置。

### 5.4 leave-one-study-out 使用随机未训练 context embedding

vocab 在全数据上创建 held-out study ID，但 fold 训练不更新其 embedding，测试时却仍使用这个随机向量。Spearman 对全局 offset 相对不敏感，而 sign 对零点极敏感，这与现有结果模式一致。

修复应采用：

- 显式、训练过的 `UNK_STUDY/UNK_ENDPOINT`；
- training-time context dropout；
- 或移除不可外推的 study identity，改用可观测 assay covariates；
- endpoint-specific adapter/head，但在 unseen study 上必须有冻结、可识别的 scale 处理。

### 5.5 当前训练远未构成强模型 ladder

现有主要模型为 hidden size 64、约 4 epochs、单 seed 42、from-scratch。合同要求的 Huber、beneficial、calibration 等也未完整按语义实现。当前结果不能用来判断更强架构是否无效。

修复后建议按同一 target/anchor、相同调参预算建立：

```text
linear/position/edit-feature baseline
anchored CNN
anchored small Transformer
Siamese/full-pair model
UTR-LM probe or fine-tune, exposure flagged
an untouched foundation backbone where possible
```

GSE114002 序列曾出现在 UTR-LM pretraining 中，因此其 foundation 结果必须标成 sequence-exposed，不能与真正 untouched external generalization 混写。

## 六、统计与评估缺陷

### 6.1 聚合单位错误

当前 E0 把三个 `study|endpoint` context 等权平均：

```text
GSE114002: 0.118803
GSE217518 HEK: 0.255623
GSE217518 SH: 0.516725
context-macro: 0.297050
```

先在 GSE217518 内聚合，再做两-study macro：

```text
GSE217518 mean: 0.386174
two-study macro: (0.118803 + 0.386174) / 2 = 0.252488
```

这个值只比 0.25 高约 0.0025，且没有 group bootstrap CI 或种子方差，不能宣称稳健过线。

### 6.2 permutation p-value 伪重复

当前 `p=1/2001≈0.00049975` 来自 60,237 行内的逐 row shuffle，不是按 source/study 的 paired permutation。5′UTR 只有两个独立 study；以 study 为推断单位时，不可能得到如此小的精确 p-value。

正式统计应包括：

- 每 study 先计算；
- source/group bootstrap；
- 多 seed 的 hierarchical summary；
- paired comparison to strongest matched baseline；
- 不把 rows、variants 或 endpoints 伪装成独立 study replication。

### 6.3 enrichment 定义病态且发生于结果可见之后

初次 M4 结果的 oracle-normalized top10% 指标约为 0.1322。随后定义变为：

```text
mean(delta among predicted top 10%) / mean(delta among all rows)
```

GSE114002 的总体 signed mean 仅约 0.02365，所以 top-set mean 约 0.695 被放大成 29.39，最后 context-macro 为 9.92。分母接近零、可为负或换符号时，这个比率不稳定。

建议前瞻性冻结以下之一：

- beneficial prevalence enrichment，beneficial 由 replicate/noise margin 定义；
- top-q mean effect minus random-selection mean，配 group bootstrap CI；
- NDCG/normalized regret，仅限真实同-source candidate pools；
- 每 study 标准化后再汇总的 top-q effect。

原始 0.1322、后续 9.92 和修订原因都必须保留在 decision log 中，不能覆盖历史。

### 6.4 ordinary internal 已被反复用于选择

当前 folds 已用于模型、loss、confidence、阈值与 metric 选择，因此不能继续称为 untouched internal test。推荐：

- 将其正式改名为 development/validation；
- 若还有足够 grouping atom，保留新的 untouched group-disjoint internal confirmatory set；
- 若独立 study 数不足，则使用严格 nested group CV，并诚实限制推断范围；
- 只有冻结后才运行 confirmatory evaluation。

## 七、G1/Flow 主线为何尚未成立

### 7.1 generate-then-rerank 实际是 no-op

当前实现等价于：

```python
guidance_logits = base_logits + 0.0 * critic_delta
```

因此 generate-then-rerank 与 no-guidance 结果完全相同。measured-pool 分支又通过 special-case 直接使用 critic score，使其与另一个 critic rerank baseline相同。这不是一个真实的强 baseline。

### 7.2 guidance 由同一个 critic 指导并自评

`rate_cfg` 直接最大化 frozen critic score，最终仍用同一 critic score 声称提升；没有 independent evaluator 或真实 measured label。高 log-variance 与 policy entropy collapse 表明可能发生 reward hacking。

因此 `+3.33` 只能写成：

```text
DEVELOPMENT_PROXY / CRITIC-SELF-SCORE IMPROVEMENT
```

不能写成 real-mRNA biological quality 或正式 guidance value。

### 7.3 未做到 matched compute 与 study-macro

现有 artifact 只有总 wall time，没有每策略 NFE、critic forward-equivalents、candidate count、wall time 和 memory。`rate_cfg` 每一步评估所有 legal actions，而 no-guidance 不做 critic forward，计算不匹配。

generation evaluation 位于 fold loop 之外，只复用最后一个 fold，也不是跨 study 汇总。

### 7.4 Flow 的修复与退路

只有 critic 与真实 candidate pool 修复后才值得重做 G1：

1. 真正生成 N candidates，再按冻结 evaluator rerank；
2. 对齐 candidate budget、generator NFE、critic forward-equivalents 和 wall-time frontier；
3. 所有 fold/study 独立运行后汇总；
4. 使用 independent evaluator 或 ordinary measured neighborhood；
5. reward 加入 uncertainty/OOD penalty；
6. 报告 entropy、diversity、duplicate rate、OOD 与 reward-hacking stress test；
7. 若 rate-guidance 不优于真实 reranking，论文主线回退到 critic/benchmark，不强保 Flow。

## 八、sealed 的真实状态与禁区

当前 aggregate commitment 没有显示 analytic labels 或 member rows 已被使用；审计也没有访问 sealed 内容。但正式 runner 的 evaluator 仍是未实现 stub：

```text
_evaluate_sealed_external(...)
  -> raise SealedAccessError("sealed final sequence-level scoring not yet mounted ...")
```

更危险的是，runner 会先登记 `ACCESS_INTENT` 并 reserve one-shot，再进入 evaluator；异常后写 `ABORT`，而 ABORT 不允许重试。

因此当前绝对禁令是：

```text
DO NOT RUN E0 SEALED FINAL
```

这里需要精确区分“正式 sealed-final 未消费”和“restricted 层从未被机器访问”：历史 access log 中存在 allowlisted D1 materialization、aggregate audit 与 restricted finalizer 事件。这些不等于 analytic-label unblinding，也没有消耗正式 E0 evaluator 次数，但说明“restricted shard 从未打开”的绝对表述不准确。当前还存在 home log 与旧 commitment mirror 的 chain-head hash 不一致，需要在正式访问前确定唯一 canonical access-log chain。

正式前必须在不读取 sealed labels 的条件下完成：

- synthetic fixture；
- ordinary mirror 的端到端 scorer；
- `--dry-run` 不写 ACCESS_INTENT；
- checkpoint/config/code/data hash binding；
- state-machine failure injection；
- aggregate-only output验证；
- 权限与审计日志复核。

即使 ordinary 内部全部通过，sealed 仍是独立一次性外部裁决，不能保证 GO。

## 九、建议的 repair-first 执行路线

### Phase R0：治理与保护，不训练

目标：消除 authority 与 sealed 风险。

交付：

- 唯一 authority 与 lineage；
- 原始 M0 failure、错误指标和后续 repair 的 immutable decision log；
- sealed runner 的 hard disable 与 safe dry-run；
- 新的独立 worktree/run root；
- 全部旧 artifact 只读保留。

停止条件：authority 未统一或 dry-run 会触发 reservation 时，不进入正式模型实验。

### Phase R1：数据/grouping 与 label audit，不训练或仅做轻量统计

目标：让 estimand、分组、split、sign 与 ranking 可识别。

工作：

- 重建 GSE114002 canonical source candidate pools；
- endpoint/context 分离；
- GSE217518 raw scale、方向、极值和 replicate 审计；
- GSE145046 label join qualification；
- source/gene/study group split；
- near-zero/noise margin 仅由 ordinary replicate evidence 预定义；
- 修复 study-macro、CI、enrichment 与 paired baseline protocol。

停止条件：若没有可验证的 source grouping 或 label semantics，则不做 listwise/guidance claim。

### Phase R2：确定性模型 bug 修复

目标：建立最小、语义正确、编辑可见的 critic。

工作：

- candidate-value 与 delta label/head 分离；
- sign/beneficial 对 delta；
- rank 仅限同 source pool；
- inverse consistency 仅用于 delta head；
- edit-centered/full-length encoder + padding mask；
- trained UNK/context dropout；
- robust endpoint transform + explicit source anchor；
- 完整 loss 与 prediction artifact。

先做单种子小型 factorial，避免在无效架构上烧 GPU。

### Phase R3：ordinary-only 模型选择与强基线

目标：判断修复是否产生跨 study 的真实信号。

建议逐项消融：

```text
A current reproduction
B + edit-visible encoding/padding mask
C + trained unseen-context handling
D B+C
E D + robust endpoint transform/source numeric anchor
F E + corrected multi-task loss and within-source ranking
G F + foundation backbone ladder
```

对每个候选都使用相同 target、anchor、调参预算和 split。只有 pilot 显示每个 study 都改善，才进入五种子完整运行。

### Phase R4：多种子 confirmatory gate

最低要求：

- 五种子全部报告，不选 best seed；
- 每 study 先计算，再 macro；
- group bootstrap 95% CI；
- strongest matched nonfoundation baseline；
- foundation exposure-aware comparison；
- sign、rank、calibration、OOD 和 uncertainty 一致；
- ordinary confirmatory set 或严格 nested CV；
- 所有实验登记和 artifact hash。

这一步决定是继续 H1 effect 主线、进入 HOLD，还是将论文 claim 前瞻性降级。

### Phase R5：重做 Flow/G1

前置条件：critic 和真实 candidate pool 通过 R4。否则不运行。

若 rate-guidance 在 independent/measured evaluation、matched compute 下不能稳定超过真实 generate-then-rerank，则按合同主动降级 Flow claim。

### Phase R6：冻结后才讨论 sealed

前置条件：

- authority 唯一；
- scorer/runner 无 sealed 数据完成 dry-run；
- dataset/model/config/metric/statistics 全冻结；
- ordinary confirmatory 结果值得一次性外部验证；
- failure fallback 已写入论文计划。

满足这些条件只表示“值得运行”，不表示“保证通过”。

## 十、发表路线与避免“全盘 NO-GO”的方式

科学上不能保证数据结果不出现 NO-GO。可以保证的是，不再因实现错误、泄漏、错误统计或一次性访问浪费而产生伪失败/伪成功，并提前设计三层可发表产物：

### 路线 A：完整 XEditFlow 方法论文

要求：

- critic 跨 study 达标；
- 真实 candidate neighborhoods；
- guidance 在 matched compute、independent/measured evaluator 下优于强 reranking；
- 结果对种子、group bootstrap 与 OOD 稳健。

这是最高目标，但当前证据远未达到。

### 路线 B：数据/benchmark + effect critic 论文

若 effect 模型成立而 Flow 不成立：

- 发布 source-relative effect benchmark；
- 强调 group-aware split、assay harmonization、uncertainty 与 foundation exposure；
- Flow 作为负结果或 appendix，不作为标题 claim。

### 路线 C：数据审计/可识别性/负结果论文

若模型修复后仍不能跨 study：

- 系统展示 source grouping、assay scale、near-zero noise、sequence truncation 与 naive split 如何制造虚假提升；
- 提供可复现 benchmark 和 failure taxonomy；
- 但前提仍是先修复当前 O0 candidate-pool 错误并建立可信数据资产。

这就是“不要出现项目层面的 no go”的正确含义：不是强行让每个 gate PASS，而是让每一种诚实结果都有匹配的、不过度主张的交付路线。

## 十一、建议用户确认的决定

建议现在确认以下路线：

```text
保留原始 sign gate和原始 M0 failure；
不消费 sealed；
不立即做 outcome-informed gate amendment；
先执行 R0–R2：authority/sealed 防护、grouping/label 修复、模型确定性 bug 修复；
用 ordinary-only pilot 决定是否投入五种子与更强 foundation model；
只有 corrected protocol 真正成立后，再决定论文主线与 sealed。
```

如果确认，第一批远端修改应只发生在新的隔离 worktree，保存旧字节和 artifact，先修可测试的确定性问题，不启动大规模训练，也不触碰任何 restricted sealed path。
