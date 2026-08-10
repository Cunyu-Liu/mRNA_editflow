# mRNA-XEditFlow Route A V3：唯一战略目标、科学前进循环与分阶段执行总计划

## 1. 总体目标与已锁定决策

### 最终目标

新建一份完整替代旧执行 authority 的 `mRNA-XEditFlow Route A V3` 总合同。其唯一终局是：

> 在公开、可追溯的 mRNA 干预数据上，建立可跨 source/study 迁移的 source-relative 效应模型，并将冻结、校准后的效应目标通过势函数一致的 density-ratio/Doob 控制注入生物合法、预算受控的 mRNA 编辑 CTMC；在真实 measured candidate neighborhoods、独立 evaluator、严格 OOD 和匹配计算预算下，稳定优于强 search、真实 generate-then-rerank、一阶 guidance、CFG、diffusion/flow 基线，并通过一次性 sealed external adjudication。

路线 A 是不可改变的战略目标，但科学证据状态不得预先写成成功。V3 必须同时锁定：

```yaml
strategic_target:
  value: ROUTE_A_FULL_XEDITFLOW
  mutable: false
  change_authority: USER_ONLY

evidence_status:
  allowed:
    - NOT_RUN
    - IN_PROGRESS
    - PASS
    - FAIL_CURRENT_PROTOCOL
    - FAIL_REPAIRABLE
    - BLOCKED_PENDING_PUBLIC_EVIDENCE
    - TERMINATED_SAFELY_WITH_EVIDENCE

claim_status:
  allowed:
    - NOT_ESTABLISHED
    - ESTABLISHED
    - INVALIDATED_CURRENT_FORMULATION
```

### 用户已确认的决策

- 路线 A 是唯一终局；路线 B/C 只能形成中间论文、数据资产或负结果资源，不能替代或终止路线 A。
- 始终坚持公开数据，不新增湿实验；因此 L4“真实治疗性或实验验证改善”永久不在本版 claim 中。
- 新建 V3 替代合同；两份原文件及历史结果保持原字节、原状态和原失败结论，不原地改写。
- 5′UTR primary edit budget 为 `k ∈ {1,3,5}`；`k=10` 是预注册 secondary efficiency/OOD stress track。
- 普通数据的跨研究门槛为至少三个完成资格审计的非 sealed 独立研究，再加一次 GSE246381 sealed adjudication；至少两个 ordinary study 应提供 A1 source–candidate intervention evidence，至少一个应提供真实 A2 dense candidate neighborhood。
- 废止病态的 `mean(top10% delta)/mean(all delta)=9.92×` 作为正式 gate；改用 `top-10% additive uplift = selected mean delta − random-selection mean delta`，按 source group bootstrap，并保留旧 0.1322/9.92 及变更原因作为不可覆盖历史。
- 3′UTR/CDS 至少一个 secondary region 必须取得有统计支持的迁移收益；优先推进 3′UTR，CDS 数据与 synonymous-codon graph 资格审计并行进行。
- 创新主轴固定为“公开干预数据上的可识别效应学习 + 势函数一致的 legal mRNA 精确控制 + 真实 measured-neighborhood 独立评测”，不声称“首次把 Doob 或 Edit Flow 用于生物序列”。

### A1 方案 A 数据角色修订（用户授权，2026-08-10）

本修订纠正一个已被公开数据结构和可识别性审计证伪的数据角色假设，不改变 Route A 战略目标，也不降低任何 gate：

- `GSE145046` 固定为 `ABSOLUTE_AUXILIARY_ONLY / TRUE_A2_NOT_QUALIFIED`。它是一个固定 reporter scaffold、一个 N10 可变位点上的 absolute outcome landscape，不是可识别的 source-relative intervention neighborhood；不计入 ordinary、A1 或 true-A2 gate，也不得作为 source-relative confirmatory evidence。只有另行闭合 paper-faithful endpoint、license、checkpoint exposure 和 auxiliary-only split 后，才可用于 fixed-scaffold absolute pretraining/development。
- `GSE114002` designed library 固定为 `A2_RECOVERY_CANDIDATE_NOT_QUALIFIED`。当前只授权 source/mother anchor、candidate pool、replicate/SE、license、checkpoint exposure 和 leakage 的正式恢复与资格审计；已知相关序列暴露标记保持 `SEQUENCE_EXPOSED`，即使未来通过，也仅限 within-assay development/optimization 边界，不能提前计入 gate 或冒充独立 confirmatory evidence。
- A1 hard gate 原样保持：至少 3 个 qualified ordinary studies、至少 2 个 A1、至少 1 个真正 source-anchored true A2。当前 qualified 计数不因本修订增加，A2 GPU 训练仍未授权。
- 若 `GSE114002` 的字段 authority、source anchor、multi-candidate pool、replicate/SE、license、checkpoint exposure 或 leakage 无法闭合，则必须引入新的 genuine public A2 study；不得退回用 GSE145046 的百万 rows、Hamming proxy 或同一 scaffold 的 absolute outcomes填补 true-A2 gate。

### A1 GSE200302 官方 metadata 角色 authority 修订（2026-08-10）

- 接受 `GSE200302_SRR_ROLE_AUTHORITY_20260810T230315P0800_e3b724d` 为 `COMMITTED_AND_ACCEPTED / EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED`。该 authority 仅闭合 `GSE200302 / PRJNA824033` 的 24 个官方 SRR/GSM/BioSample 角色映射：`High_Poly`、`Low_Poly`、`pDNA`、`Total_RNA` 四个 measurement family，每个 family 严格包含 replicate 1–6。
- 官方 grid 不含 `80S_RNA`，且 `pDNA != 80S_RNA`。因此历史 preflight 中的 `EXACT_SRR_SAMPLE_ROLES_UNKNOWN` 在当前 authority 下已闭合，但不能将它误写为 raw replay 已可执行；其 successor truth 是 `PRJNA824033` 官方 role grid 与当前 `80S` 预期冲突，且所需 `80S` raw authority 缺失。
- `GSE200304` 仍是 ordinary-study 单元，`GSE200302` 仅作为其 primary subseries authority，不新增独立 study 或 gate credit。本 metadata closure 不是 raw count/xTail replay、study qualification 或 canonical materialization；ordinary/A1/true-A2 贡献仍为 0/0/0，`qualified=false`，training/model selection/next phase 均不授权，canonical record 仍为 0。
- 远端实现与 binding commits `d042d7c1706a80821a19b78334985441bcf6eb86` / `e3b724d00a9e5263b99475b9744fc0bb68a3ab67` 已 push 且 clean，consumer 已独立接受该 bundle；A1 活动 runtime 同步仍为 `PENDING_NO_EVT_035`，本修订不得预造 `EVT-035`。

### 当前活动任务的保护

截至 2026-08-09 23:24 CST，迁移 worktree 的 `scripts/m4_routea/` 正在 GPU 5 上执行一个 pre-V3 development run。该任务：

- 不停止、不修改、不迁移、不高频监控；
- 允许自然完成并保存现有 checkpoint；
- 完成后登记为 `PRE_V3_DEVELOPMENT_ONLY`；
- 因仍使用 `candidate_value`、`MAX_SEQ_LEN=100` 和旧 enrichment 口径，不得用于设定 V3 gate、选择 V3 metric、宣称 candidate-specific effect，或翻案原 M0 failure；
- 新 V3 工作在独立 worktree、独立分支和独立 `/mnt` run root 中进行，不 stage 现有未跟踪文件。

## 2. V3 文档、authority 与公共接口

### 权威文档链

新建并提交：

1. `docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md`：唯一规范性科学与执行合同。
2. `configs/route_a_v3.yaml`：机器可执行的任务、数据、模型、metric、gate、GPU、sealed 与 artifact 约束。
3. 机器可读 companion bundle：task registry、data-role registry、baseline registry、claim–evidence matrix、amendment/decision log 和 schema。

V3 必须完整记录本轮讨论的所有决策、理由和拒绝项，包括：

- 路线 A 固定目标与证据状态分离；
- public-data-only 与 L4 禁令；
- B/C 仅为中间成果；
- `k={1,3,5}` primary、`k=10` secondary；
- 三 ordinary studies + one sealed；
- 3′UTR 优先、CDS 并行资格审计、至少一区迁移成功；
- top-10% additive uplift；
- 原 M0 failure、错误 pool、旧 enrichment、current dev run 的历史定位；
- prior-art collision 和新的创新边界；
- sealed one-shot 的未来显式授权要求。

旧合同不改字节。V3 通过 supersession manifest 绑定：

```text
旧本地合同 SHA
+ 旧远端 v1.1 SHA
+ 原始 M0/O0/G1/E0 状态
+ V3 文档 SHA
+ executable YAML SHA
+ Git commit
```

新工作分支从当前迁移 authority 所在提交 `bbb71dcba6f1e1c9cb75a8a6653f1a4fe4a6ca0c` 创建隔离 worktree；执行前若 authority 已漂移，只允许依据“包含当前 active contract、claim matrix 与完整 lineage 的最新明确提交”更新基点，禁止从脏主 checkout、未跟踪 Route-A 代码或旁路 artifact 猜测 authority。

### 核心数据与运行接口

建立并验证以下公共 schema：

- `CanonicalInterventionRecordV3`
  - source/candidate identity、完整序列、edit set、region、study、assay、context、endpoint、原始单位、paper-faithful transform、delta、replicate/SE、biological source group、data role、exposure、split、provenance、license 和 reject reason。
- `MeasuredCandidatePoolV3`
  - 同 biological source、同 study/assay/context/endpoint 下的不同 candidate sequences；同一 candidate 的不同 endpoint 永不形成候选池。
- `PredictionRecordV3`
  - `μΔ`、aleatoric/epistemic uncertainty、sign/beneficial probability、ranking energy、OOD/abstention、model/data/config hashes 和 exposure stratum。
- `RunManifestV3`
  - task/run/parent IDs、V3 hash、code commit、dataset/split/config/checkpoint hashes、seed、GPU UUID、环境、命令、绝对输出路径、状态、失败类型和恢复分支。
- `ComputeLedgerV3`
  - candidate count、unique count、generator NFE、critic/guidance/reranker forwards、total forward-equivalents、wall time、peak VRAM、GPU 型号和并发条件。
- `GateRecordV3`
  - gate 输入、统计单位、阈值、逐 study 结果、CI、全部 seeds、判定、失败包、下一条 Route-A recovery task 和是否接触 sealed。

### 大型资产与 Git 边界

所有新大型资产进入：

```text
/mnt/cunyuliu/mrna_xeditflow_routea_v3/
  raw/<dataset>/<version>/
  processed/<freeze_id>/
  weights/<model>/<revision>/
  runs/<phase>/<run_id>/
  cache/
  legacy/pre_v3/
```

- raw data、foundation weights、checkpoint、predictions、完整 logs 和大 artifact 不进 GitHub。
- sealed/restricted 数据保持原 ACL 和 canonical restricted path，只在普通 registry 中保存无敏感内容的指针与 custody metadata。
- GitHub remote 固定为 `git@github.com:Cunyu-Liu/mRNA_editflow.git`；每个 accepted task 做 focused commit 并 push 当前 Route-A task branch，不自动 push/merge `main`。
- 每次 push 前检查 secret、restricted path、large binary、无关改动和 `git diff --check`。
- 监控 heartbeat、临时缓存、失败的半成品训练和大日志不制造无意义提交；若失败报告、manifest 或修复代码本身已成为 accepted artifact，则单独提交。
- push 失败时状态为 `IMPLEMENTATION_COMPLETE_DELIVERY_BLOCKED`，保留本地 commit、SHA、branch 和失败证据，不谎报上传成功。
- 哈希只用于首次导入、冻结、传输、最终交付或异常调查，不对普通源码重复设哈希门禁。

## 3. 数据、专用架构、benchmark 与成功标准

### 数据资格与任务可识别性

优先裁定：

- `GSE145046`：保持 `ABSOLUTE_AUXILIARY_ONLY / TRUE_A2_NOT_QUALIFIED`；不计入 ordinary、A1 或 true-A2 gate，不承担 source-relative confirmatory evidence。只有 auxiliary-specific endpoint、license、checkpoint exposure 和 split 全部通过后，才可用于 fixed-scaffold absolute pretraining/development。
- `GSE114002`：designed library 作为 `A2_RECOVERY_CANDIDATE_NOT_QUALIFIED`，优先恢复 mother/source anchor、同 assay candidate pools、replicate/SE、license、checkpoint exposure 与 leakage；random library 只作 absolute auxiliary，natural subset 单独重建。相同短 source sequence 只有在 study、assay、endpoint、locus/transcript/design family 一致后才能合并；所有相关结果保持 `SEQUENCE_EXPOSED`，资格通过前不得训练或计入 gate。若该恢复路线失败，必须引入新的 genuine public A2 study。
- `GSE149487/PLUMAGE`：优先恢复为 A1 ordinary intervention study。
- `GSE217518`：完成 endpoint direction、raw scale、异常值、replicate、full-context reconstruction 和 biological grouping 前只作 development/stress；资格通过后才可进入三个 ordinary studies 的 confirmatory 集合。
- `GSE246381`：保持 sealed；不用于训练、调参、阈值、metric、calibration 或 architecture selection。
- 3′UTR：依次资格审计 GSE217518、GSE200304、ENCSR854RUF、GSE232572、GSE186455。
- CDS：并行恢复 GSE207584/iCodon 和其他 public synonymous-family sequence/label lineage，但不让 CDS blocker 延迟 3′UTR 先行路径。

数据冻结前必须报告：

```text
nominal rows
distinct candidates
biological source groups
gene groups
study groups
eligible multi-candidate pools
edit-count strata
replicate/SE coverage
beneficial/noise-zone balance
post-dedup effective N
foundation exposure
license/redistribution status
```

最低资格规则：

- 至少三个非 sealed 独立 ordinary studies，且 endpoint/assay 分头计算，不能把 endpoints 当 studies。
- 至少两个合格 A1 study 和一个真实 A2 dense neighborhood。
- NDCG 主池至少三个 distinct candidates；两个 candidates 只能用于 pairwise evaluation。
- source/group split、reverse edge、candidate、gene、sequence-cluster 与相应 study/context leakage 必须为零。
- effective N 是否足够，不用 row 数拍脑袋决定：在看模型结果前，用 ordinary replicate/group bootstrap simulation 冻结达到 80% power 和目标 CI 宽度所需的 source-group 数；不足则保持 `BLOCKED_PENDING_PUBLIC_EVIDENCE` 并继续公开数据恢复。
- `k=3/5` 的强 biological optimization claim 必须存在相应 measured multi-edit support；single-edit 数据不得自动证明 epistasis 或多步真实收益。

### Route-A 专用 critic

主 critic 为 `Edit-Set Causal-Contrast Critic`：

```text
source-cached chunked full-context encoder
+ edit-centered local windows
+ permutation-invariant edit-set interaction encoder
+ explicit position/ref→alt/region/start-distance features
+ uncertain mechanism deltas
+ observable assay/context/endpoint conditioning
+ antisymmetric delta head
+ replicate-aware uncertainty
+ OOD/abstention
```

硬要求：

- 不再固定截断前 100 nt；模型必须看到每个 edit，并通过 edit-after-100 fixture。
- padding、attention、pooling 全部使用有效 mask。
- primary zero-shot 模型不使用 study-ID 随机 embedding；只使用可观测 context、训练过的 `UNK_CONTEXT` 和 context dropout。
- 机制特征分三类：hard invariants 进 action graph；structure/motif/uAUG/Kozak 等 proxy 作为带不确定性的输入或诊断；measured endpoint 只能由相应 assay label 监督。
- 数值效应均值采用反对称构造：
  \[
  \mu_\Delta(x_0,x,c)=\frac{H(x_0,x,c)-H(x,x_0,c)}{2}.
  \]
- 方差使用 source/candidate 对称构造；`source=candidate` 时均值必须为零，交换 source/candidate 后均值反号。
- sign、beneficial、rank 均从 delta 或同 source ordinal energy 定义；absolute candidate property 只允许作为独立 auxiliary head，不施加 inverse consistency。
- 多编辑先建 additive single-edit term，再仅在真实 measured support 下启用低秩 interaction residual；支持外的 `k>1` 必须提高 uncertainty 并允许 abstain。
- guidance reward 使用经过 calibration 的 lower confidence bound，不使用裸 critic mean。

若 D1 证明 numeric delta 的零点/尺度无法跨 assay 传递，但真实 candidate pools 成立，则切换为 Route A 内部的 `Assay-Invariant Listwise Energy Critic`：

- 只在同 source/study/assay/context/endpoint 内用 Bradley–Terry、Plackett–Luce/ListMLE 和 noise-aware ties；
- claim 限定为 source-neighborhood ranking/selection；
- 不把 ordinal energy 解释成可跨 assay 比较的绝对生物增量；
- 这属于 estimand redesign，不是路线 B/C。

A1 numeric critic 与 A2 ordinal critic先各做一个同预算 GPU pilot；由预先冻结的 identifiability rule 选择，不同时扩张多个 backbone。

### 势函数一致的预算化 legal XEditFlow

Primary 5′UTR 状态定义为 source-anchored、无环稀疏编辑 DAG：

```text
state =
  source
  + current sequence
  + edited-position set
  + remaining budget
  + assay/context
  + algorithmic time
```

- action 为尚未编辑位置的 `source_base → alt_base` 或显式 `STOP`。
- 同一位置 primary trajectory 不重复编辑，event count 与 net edit count 一致。
- `STOP`、预算耗尽、无合法动作、数值失败是不同终态。
- CTMC time 明确称为生成/控制时间，不宣称是真实 mRNA 物理动力学。
- 所有非禁止合法动作保留正 support floor；hard legality 在 rate normalization 前执行。

对多个 action 指向相同 next state，先聚合 observable transition rate：

\[
U_p(s,s')=\sum_{a:T(s,a)=s'}u_p(a\mid s).
\]

学习标量势函数：

\[
h_t(s)=\mathbb E_p[w(X_T)\mid S_t=s], \qquad V_t(s)=\log h_t(s),
\]

并用：

\[
U_q(s,s',t)=U_p(s,s',t)\exp\{V_t(s')-V_t(s)\}
\]

构造 guided rate。禁止用彼此独立、无法保证路径一致性的自由 action-ratio head作为最终 exact 方法。

“exact”只指给定 frozen base path、terminal tilt、support 和状态定义下的解析 Doob operator；learned `V` 是近似估计，必须单独报告 approximation error。路线 A 不声称首次使用 Doob 编辑流，因为 [pCoMole 已使用 Doob-\(h\) 变换控制 biomolecular Edit Flow](https://openreview.net/forum?id=1mCS10EFRq)。真正需要建立的新贡献是 measured intervention critic、mRNA-specific source/assay state、transition aggregation、预算/STOP 支撑、独立 measured evaluation 和 matched compute。

### Benchmark 与横向比较

建立三个不混写的 benchmark track：

1. `NATIVE_REPRODUCTION`
   - 在原论文任务、原数据和原 metric 上复现官方方法。
2. `COMMON_TASK_SYSTEM`
   - 将可适配方法接入相同 mRNA-EditBench、split、edit budget、candidate pool 和 evaluator。
3. `ARCH_CONTROLLED`
   - 锁定 backbone、数据、参数带、训练 exposure、seeds、HPO、GPU、candidate budget 和 forward-equivalents，只比较 action graph、critic、guidance 与 sampler。

核心 ladder：

- Effect sanity：majority sign、study/source mean、anchor-only、candidate permutation、source/context residual、position、ref→alt、GC/MFE/motif、k-mer ridge。
- Standard/paired：XGBoost、CNN、small Transformer、anchored CNN、Siamese/full-pair、absolute-minus-source、ordinal/listwise。
- Foundation：Optimus 5-Prime、UTR-LM、mRNABERT、适用时 Orthrus，并按 exposed/unknown/untouched 分层；[mRNABench](https://openreview.net/forum?id=JcFz2WkCfy) 只作为 property/representation benchmark 相邻工作，不替代 intervention benchmark。
- Search：exact enumeration、random legal、greedy、beam、GA、simulated annealing/local search、true generate-N-then-rerank。
- Generation/guidance：[Edit Flows](https://arxiv.org/abs/2506.09018)、masked DFM/discrete diffusion、[FlexFlow](https://arxiv.org/abs/2606.10543)、no/rate/latent/first-order guidance、[DGM](https://arxiv.org/abs/2509.21912)、pCoMole 和完整 XEditFlow。
- mRNA-specific adjacent systems：[RNAdiffusion](https://arxiv.org/abs/2409.09828)、[UTailoR](https://pubmed.ncbi.nlm.nih.gov/41069846/)、[mRNAutilus](https://arxiv.org/abs/2605.31296)、[T3PO-mRNA](https://openreview.net/forum?id=sShNT08Boq)、[mRNA-GPT](https://www.biorxiv.org/content/10.64898/2026.03.31.715707v1)。无法在共同任务运行的模型标记 `PAPER_ONLY_TASK_MISMATCH`，不得伪造执行结果。

所有 optimization 方法至少报告：

```text
equal candidate budget
equal generator NFE
equal total forward-equivalents
equal wall time
same GPU cohort
same source pools
same legal action space
same seeds/HPO budget
unique candidate rate
peak VRAM
cold-start and amortized cost
```

### 冻结的 Route-A 成功门槛

Critic/effect：

- per-study 计算后 equal-study macro；
- macro delta Spearman `≥0.25`；
- macro sign accuracy `≥0.60`；
- top-10% additive uplift 的 source-group bootstrap 95% CI lower bound `>0`；
- additive uplift 不得在任何 qualified primary study 中出现超出其预冻结 replicate-noise equivalence margin 的明显负方向；
- 胜 strongest matched executable non-foundation baseline；
- 五 seeds 全报告，不选 best seed；
- calibration、OOD、coverage-risk、anchor-only、candidate permutation 和 residual controls 全通过。

Flow/theory：

- hard legality `=100%`；
- budget violations `=0`；
- toy exact rate relative error `≤1e-5`；
- terminal distribution TV error在预注册数值容差内；
- path-product consistency、transition alias aggregation、support、STOP/budget fixtures 全通过。

Guidance：

- NDCG 至少高于 strongest matched baseline `+0.05`；
- top-decile recall `≥0.70`；
- normalized regret `≤0.10`；
- measured labels或真正独立 evaluator 重现改进；
- OOD 不超过预注册退化 margin；
- quality–cost Pareto frontier 至少在一个主预算轴上显著外移，且其他轴完整报告。

Transfer：

- 3′UTR：region-specific critic/graph 相对 generic shared model 和 region-from-scratch baseline 的主指标 paired group-bootstrap CI lower bound `>0`。
- CDS：family-level NDCG/listwise improvement 相对 codon heuristics、外部 CDS 方法和 from-scratch baseline 的 CI lower bound `>0`。
- 两者至少一个通过；另一个即使失败也必须完整报告，但不允许删除失败结果。

## 4. 分阶段执行 TODO、依赖与并行安排

| 阶段 | 风险 | 主要工作与交付 | 进入下一阶段的硬门槛 |
|---|---:|---|---|
| A0：V3 authority 与现场保护，1–2 天 | 高 | 创建隔离 worktree/branch；写入 V3、YAML、decision/amendment log；登记旧合同、旧 failures 和 active pre-V3 run；sealed runner保持 hard disable；首次 focused commit/push | 单一 authority、lineage/schema valid；不触碰脏 checkout、活动任务和 sealed |
| A1：公开数据 qualification，2–4 周 | 高 | paper-faithful D0/D1；冻结 GSE145046 absolute-auxiliary 边界；恢复 GSE114002 designed-library true-A2 candidate、PLUMAGE、GSE217518 raw scale；3′UTR/CDS 并行资格审计；license/exposure/effective-N 表 | 至少 3 ordinary studies、≥2 A1、≥1 source-anchored true A2；source/endpoint/transform/group 可识别；否则保持 public-evidence blocker |
| A2：benchmark/metric/evaluator 冻结，1 周 | 高 | canonical pools、splits、leakage、top-q additive uplift、noise margins、confirmatory set/nested CV、independent evaluator、compute budget、sealed dry-run spec | 新 ordinary confirmatory set未参与选择，或严格 nested group CV；所有 metrics/statistics 前瞻冻结 |
| A3：critic 语义修复与单种子 GPU pilots，2–3 周 | 中高 | 实现 A1/A2 critic；semantic fixtures；anchor/permutation/residual controls；同预算 baseline ladder；完整 prediction artifact | 每个 qualified study 均有正向信息增益；未胜 anchor-only/强基线则冻结 failure bundle并进入 critic/estimand recovery |
| A4：五种子 confirmatory critic，2–3 周 | 高 | 五 seeds、study macro、group bootstrap、calibration、OOD、abstention、foundation exposure ladder | 完整 effect gate通过；不得用 best seed、旧 folds 或病态 enrichment |
| A5：真实 measured-neighborhood 与 search headroom，1–2 周 | 高 | 重建 O0；exact/random/greedy/beam/GA/local/reranking；source-level NDCG/regret/uplift；判断 Flow 质量或 amortization headroom | 每个 pool 为 distinct candidates；存在可测 headroom或可定义的大空间 amortization任务 |
| A6：G0 理论与 base legal Flow，2–4 周，可从 A1 并行 | 高 | scalar-potential Doob 推导；small-graph exact DP checker；support/STOP/transition aggregation；GPU 训练 base/value network；trajectory/compute ledger | 数学假设、toy exactness、base recovery、100% legality通过 |
| A7：G1 measured matched-compute guidance，2–3 周 | 高 | no/first-order/rate/latent/rerank/exact 全对照；guide–evaluator分离；三种公平预算；reward-hacking/OOD/diversity stress | 完整 guidance 和 cost gates通过；sealed 仍不访问 |
| A8：secondary region，2–4 周 | 高 | 3′UTR优先完整 adapter/head/graph/benchmark；CDS data/action graph并行；transfer vs from-scratch与外部 baselines | 3′UTR 或 CDS 至少一区取得预注册、CI支持的迁移收益 |
| A9：sealed readiness，约 1 周 | 关键 | synthetic fixture、ordinary mirror、failure injection、canonical access-log chain、aggregate-only output、完整 hash freeze、独立综合审查 | runner dry-run不写 ACCESS_INTENT；无 stub；所有 ordinary gates已完成；用户另行显式授权 |
| A10：一次性 E0、论文与发布，2–3 周 | 关键 | 执行 one-shot sealed；生成 tables/figures/claim matrix；model/data/failure cards；release package；最终 GitHub push | 不选择性删除结果；所有 Route-A claim有 evidence cell；clean replay与最终审查通过 |

关键路径预计约 18–28 周，前提是公开数据资格及时闭合；公开数据获取、license 或 effective N blocker 不承诺固定结束日期。

GPU 等待期间并行执行：

- raw provenance、source-group、endpoint 和 license 审计；
- comparator adapter 和文献/代码可运行性 registry；
- small-graph exact DP、定理、数值 fixture；
- sealed runner 静态检查与 ordinary mirror；
- tests、schema、文档、table/figure builders。

并行任务不得修改活动训练已绑定的 code、config、dataset manifest、checkpoint symlink 或 output directory。

任何 gate 失败均执行固定 Route-A recovery loop：

```text
冻结原结果与失败包
→ 标记失败层：data / estimand / critic / graph / theory / evaluator / compute
→ 写出新的可证伪 Route-A hypothesis
→ 只用 ordinary development evidence 做最小 repair pilot
→ 通过后使用未污染 confirmatory evidence重裁决
→ 目标仍为 Route A
```

禁止：

- 降低原 gate；
- 改 endpoint 或 split 制造成功；
- 将已参与选择的 test 重新称为 untouched；
- 选择最佳 seed；
- 删除强 baseline；
- 用 critic self-score代替 measured/independent evidence；
- 重复消费 sealed；
- 在同一失败假设上无上限做超参数扫描。

同一失败机制连续两个预注册 redesign 仍失败时，必须进行一次独立综合审查，并改变假设层或证据来源；不得只增加模型宽度、foundation backbone、RL 或 β sweep。

## 5. 验证、GPU 运行、监控与最终验收

### 风险匹配验证

每个 accepted task 只运行受影响范围的最小验证：

- 文档/config：schema、authority uniqueness、cross-reference、conflict-marker 和 `git diff --check`。
- 数据：paper counts、raw→canonical lineage、source/candidate diff、endpoint direction、group identity、leakage、license 和 rejected-table tests。
- critic：swap antisymmetry、identity-zero、padding invariance、edit-after-100 visibility、unknown-context、target semantics、candidate permutation、anchor-only 和 deterministic prediction artifact。
- Flow：legal action enumerator、budget、STOP、transition alias aggregation、support、path consistency、toy exact distribution、trajectory replay。
- evaluator：guide/evaluator checkpoint、data exposure和architecture separation；training path不得导入 sealed。
- operations：CUDA fail-closed、occupied GPU refusal、resume binding、manifest completeness、large-file/secret/restricted scan和 push verification。

集中式完整验证只安排在：

1. A2 数据/metric/evaluator freeze；
2. A4 critic confirmatory milestone；
3. A7 unsealed Route-A end-to-end milestone；
4. A9 pre-sealed final readiness；
5. A10 release/final delivery。

相同代码版本、相同测试结果不重复运行；最终交付明确报告实际 targeted tests、完整测试/build/lint/E2E、主动跳过的重复验证、独立审查、哈希用途和剩余风险。

### GPU-only 与低频监控

所有更新 neural critic、base flow、guidance/value network、foundation fine-tune 参数的正式 run：

- 必须使用 CUDA GPU；
- 启动时记录 GPU UUID/model、driver、CUDA/PyTorch、device、peak VRAM；
- CUDA 不可用或静默 CPU fallback 时立即 `FAIL_CLOSED`，不得自动改用 CPU；
- 启动前检查真实 PID/owner/显存/利用率，不能把“低显存”解释为空闲授权；
- 不抢占、终止或影响任何无关任务。

CPU 允许用于数据、统计、哈希、Git、small-graph exact enumeration、独立数值 checker和普通单元测试；任何需要学习参数的 toy network仍使用 GPU。

稳定训练的外部检查频率：

1. 启动和首 batch sanity；
2. 首 checkpoint 或约 10–15 分钟检查一次；
3. 稳定后每 30 分钟最多一次只读检查；
4. 仅在 checkpoint、validation、OOM、NaN、process exit 或 completion 事件即时检查；
5. 无状态变化不重复向用户汇报。

OOM、NaN 或崩溃时保存 failure bundle 和最后有效 checkpoint；改变 batch、precision、optimizer、有效 batch size 或数据版本必须使用新 run ID，不盲目覆盖续跑。

### Sealed 与最终 Route-A 验收

GSE246381 one-shot 不随本计划自动执行。只有 A0–A9 全部通过后，提交一份 sealed-readiness report，由用户另行显式授权唯一正式调用。

最终 `ROUTE_A_ESTABLISHED` 必须同时满足：

- 唯一 V3 authority 和完整 lineage；
- 三个 qualified ordinary studies + GSE246381 sealed；
- critic 原 sign/Spearman gate与新 additive-uplift gate；
- anchor/permutation/residual、calibration、OOD、abstention；
- 真实 candidate pools 和 search headroom；
- legal base flow 与 exact-theory/toy gate；
- independent/measured matched-compute guidance superiority；
- 3′UTR/CDS 至少一区迁移成功；
- sealed 达到预注册阈值；
- 全部 claim 可追溯到 frozen evidence；
- 代码、配置、文档、测试和小型 manifests 已通过 focused commits 推送 GitHub；
- 大数据、权重、checkpoint 和正式 run artifacts 已在 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/` 形成可恢复 manifest。

如果某项未通过，Route-A claim 保持 `NOT_ESTABLISHED`，项目进入对应的 Route-A recovery cycle或 `BLOCKED_PENDING_PUBLIC_EVIDENCE`；不改成路线 B/C 终局，也不把意愿、工程通过、GPU smoke、toy exactness、checkpoint hash或自评分数冒充路线 A 已实现。
