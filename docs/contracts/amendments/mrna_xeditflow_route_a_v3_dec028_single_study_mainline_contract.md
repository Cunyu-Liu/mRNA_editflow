# Route A V3 DEC028 候选合同：Single-study source-relative development / engineering-theory 主线

## 0. 文档身份、状态与一句话裁定

| 字段 | 冻结值 |
|---|---|
| Contract ID | `MRNA_XEDITFLOW_ROUTE_A_V3_DEC028_SINGLE_STUDY_MAINLINE_CONTRACT_V1` |
| Decision candidate | `V3-DEC-028` |
| Document status | `DRAFT_FOR_REVIEW_NOT_ACTIVE_PROTOCOL` |
| Authority status | `NON_AUTHORITATIVE_PENDING_APPEND_ONLY_DEC028_MACHINE_AUTHORITY_AND_RUNTIME_SYNC` |
| Owner decision basis | `OWNER_INITIATED_PROSPECTIVE_STRATEGIC_SELECTION_NOT_DEC027_AUTOMATIC_TRIGGER` |
| Current safe execution scope | `DOCUMENTATION_STATIC_SYNTHETIC_ZERO_UPDATE_AND_NONLEARNED_CPU_ONLY_UNDER_EXISTING_AUTHORITY` |
| Independent contract review | `PASS_NO_ACTION_CHANGING_P0_OR_P1` |
| Scientific claim status | `NOT_ESTABLISHED` |
| Snapshot | `A1-EVT-060`；Git `214ee9cd131ed9a99c4d425f22fc9f4f9f184a22`；2026-08-15 CST |

本合同候选作出以下战略裁定：

> Full Route A 的 `3 ordinary / 2 A1 / 1 true-A2` 继续作为最高科学目标；当前可执行主线不再等待所有候选数据集串行资格化，而改为以 `GSE200304` 为唯一科学主锚点的 single-study source-relative development，并让 evaluator、legal CTMC、exact reference、base recovery 和其他 engineering-theory 工作并行推进。其余数据集只承担 development、exploratory evaluator、negative-control 或 future sealed 角色。

这不是把 Full Route A 的 gate 从 `3/2/1` 降为 `1/1/0`，也不是声称原问题已经通过。它是对**当前活跃科学问题和可宣称范围**的前瞻性重定义。旧 gate、旧失败和未来回到 Full Route A 的入口全部保留。

本 Markdown 只完成战略、科学问题、阶段、边界和 TODO 的合同化。它不能单独激活 DEC028，不能产生 runtime event，不能授权数据物化、row access、CUDA、模型构建、参数更新、checkpoint、训练、模型选择、A7 或 sealed access。正式激活必须通过第 16 节定义的完整 append-only authority bundle。

## 1. Authority 层次与不可改写历史

### 1.1 Authority 优先级

正式执行必须区分“不可变战略目标”和“当前 disposition”，不得让较旧 config/registry 表面覆盖较新的已生效 decision/current projection：

1. `docs/goals/MRNA_XEDITFLOW_ROUTE_A_V3.md` 只提供唯一根战略目标和不可变边界；
2. supersession manifest 确定当前有效的 append-only amendment 链；
3. 链上最新已生效 decision 与已结算 runtime current projection 决定 dataset、count、claim、lock 和 phase disposition；
4. `configs/route_a_v3.yaml`、qualification config、data-role/task/split/claim/baseline registries 必须与上述 disposition 同步一致，作为执行接口；
5. dataset-specific terminal reports提供更窄的 route/gate 事实，但不能自行 promotion；
6. 历史审计与公开来源只在更高层未结算对应事实时使用。

如果 latest decision/current projection 与 executable config 或 registry 冲突，必须 `FAIL_CLOSED_AUTHORITY_SURFACES_INCONSISTENT`；不能按“旧 config 优先”执行，也不能挑选更有利的表面。本合同不能创建第二个 active root contract，也不能覆盖根合同。正式 DEC028 只能作为 append-only amendment 改变 operational mainline，并在 activation 前消除所有 authority-surface 冲突。

### 1.2 当前已结算事实

截至 `A1-EVT-060`：

- `qualified ordinary=1`；目标仍为 3；
- `qualified A1=1`；目标仍为 2；
- `qualified true-A2=0`；目标仍为 1；
- canonical records=`6547`；
- 唯一贡献者为 `GSE200304`；
- scientific claim=`NOT_ESTABLISHED`；
- training、GPU、model selection、A7、next phase 全部为 false；
- GSE246381 仍为 sealed final-only；
- 六条 rescue report 的 qualification/credit delta 均为 0；
- DEC027 rescue floor 未达到；
- GSE269595 的 planning geometry 为 effective source-group `N=363`、参考要求 `156`、planning power 约 `0.997759`、full CI width 约 `0.196105`；
- 这只证明 true-A2 continuation geometry 在 stop-rule 意义上 reachable，不是 formal qualification，不产生 role、credit 或 true-A2 PASS；
- 因此 DEC027 的 automatic stop-rule trigger 为 false；
- `conditional_successor_activated=false`；
- `EVT061` 未预分配。

### 1.3 本轮选择的治理语义

本轮 owner 指令被记录为：

```yaml
owner_strategic_selection:
  choice: PREPARE_SINGLE_STUDY_OPERATIONAL_MAINLINE
  basis: EXPLICIT_OWNER_PROSPECTIVE_SELECTION
  dec027_automatic_trigger_claimed: false
  historical_gate_rewrite_allowed: false
  gate_threshold_relaxation_claimed: false
  new_credit_claimed: false
```

在完整 DEC028 authority bundle 生效前，`PREPARE` 不是 `ACTIVATE`。

## 2. 双层路线与成功定义

### 2.1 最高目标：Full Route A

Full Route A 继续要求：

- 至少 3 个完成资格审计、公开、非 sealed、相互独立的 ordinary studies；
- 至少 2 个 source→candidate intervention、source-relative endpoint 的 A1 studies；
- 至少 1 个 source-anchored、每 source 至少 3 个真实 measured candidates 的 true-A2 study；
- source-group power、CI、split、rights、exposure、endpoint 和 independent uncertainty 分别闭合；
- 未来独立的一次 sealed adjudication。

它的状态为：

```yaml
full_route_a:
  lifecycle: RETAINED_HIGHEST_INACTIVE_STRATEGIC_TARGET
  required_counts: {ordinary: 3, a1: 2, true_a2: 1}
  current_counts: {ordinary: 1, a1: 1, true_a2: 0}
  scientific_claim_status: NOT_ESTABLISHED
  gate_relaxation: NONE
```

### 2.2 当前候选 operational mainline

Single-study 主线回答的是一个更窄但可识别的问题：

> 在预先冻结的 `GSE200304_SUPERSERIES_ONE_STUDY`、author-published processed endpoint、scratch-only、random-initialization、zero-external-learned-input 路线内，能否开发和诚实评估 source-relative effect prediction、calibration、uncertainty/abstention 与工程可实现性；同时能否闭合 legal CTMC、exact reference、base recovery、support/STOP/budget/trajectory 等 engineering-theory 证据？

初始状态仍为 `NOT_ESTABLISHED`。合同通过、代码通过、测试通过、GPU run 完成或工程 gate 通过，都不能自动把它改成 scientific PASS。

### 2.3 两条路线不是互斥历史

- Single-study 结果不能被写成 Full Route A PASS；
- Full Route A 候选研究可以继续异步救援，但不能阻塞 Single-study 的静态、合成和工程工作；
- 未来新增合格研究后，可通过独立 amendment 回到 Full Route A；
- Single-study 的失败不终止 Full Route A；
- Full Route A 的未满足也不再让所有允许的 engineering/evaluator 工作停摆。

## 3. 科学 estimand、统计单位和 claim 边界

### 3.1 Active estimand

- 核心 estimand：同一 biological source/context 内 `candidate − source` 的 direction-normalized source-relative endpoint；
- 统计单位：biological source group；
- rows、barcodes、replicates、endpoints、folds、seeds 或 subseries 都不是独立 study；
- technical replicate 不能替代 biological uncertainty；
- missing、censored、undefined 和 nonfinite 永不默认置零；
- GSE200302、PRJNA824033 和 GSE200304 component accessions 只属于一个 superseries study unit。

### 3.2 当前路线允许保留的表述

- `GSE200304_NAMED_DATASET_SOURCE_RELATIVE_DEVELOPMENT`；
- within-study prediction、calibration、coverage-risk、abstention 和 error analysis；
- exposed-development-only 的结果；
- legal CTMC、exact reference、base recovery、support、STOP、budget、trajectory 和 approximation engineering；
- 多数据集的 descriptive stress、schema/provenance robustness、negative-control 和 exploratory evaluator 结果；
- engineering proof-of-concept 或诚实的 negative result。

### 3.3 从 active Single-study claim 中删除或降级的表述

| 原强 claim | Single-study 处理 |
|---|---|
| Cross-study generalization | `NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` |
| Confirmatory biological effect | `NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` |
| Independent replication | `NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` |
| Measured-neighborhood optimization | `NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` |
| Broad/general mRNA transfer | `NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` |
| True-A2 ranking/regret/search established | `PROHIBITED_UNDER_CURRENT_EVIDENCE` |
| Guidance superiority | `PROHIBITED_UNDER_CURRENT_EVIDENCE` |
| Best-candidate discovery | `PROHIBITED_UNDER_CURRENT_EVIDENCE` |
| Untouched external validation | `PROHIBITED` |
| One accession with many rows as multiple studies | `PROHIBITED` |

`NOT_APPLICABLE_FOR_REVISED_ACTIVE_CLAIM` 只能出现在 successor claim applicability 中；历史 Full Route A 的 FAIL、BLOCKED、UNKNOWN、NOT_RUN 原样保留，绝不能回写成 PASS。

### 3.4 不可通过改题放松的识别性底线

以下问题即使在 Single-study development 中也不能豁免：

- source→candidate identity；
- endpoint direction、scale 和 transform；
- biological source group 与独立 uncertainty；
- rights、reuse 与真实 exposure；
- source/family/near-duplicate leakage；
- membership 在 outcome/significance 之前冻结；
- evaluator 与 guide/model-selection 的隔离；
- rows/barcodes/replicates/endpoints 不冒充独立研究；
- pairwise 或 absolute landscape 不冒充 true-A2。

## 4. GSE200304 主锚点合同

### 4.1 已成立但有限的事实

`GSE200304` 当前 selected route 为：

```text
AUTHOR_PUBLISHED_PROCESSED_ENDPOINT
SCRATCH_ONLY_NO_FOUNDATION_NO_EXTERNAL_LEARNED_INPUTS
RANDOM_INITIALIZATION_ONLY
```

当前 scoped qualification 为 `ordinary=1 / A1=1 / true-A2=0`，canonical membership 为 6,547。retained foundation route 仍为 `FAIL_CURRENT_PROTOCOL`。

### 4.2 永久保留的边界

- `GSE200304_SUPERSERIES_ONE_STUDY` 是唯一 study unit；
- 6,547 records 不是 6,547 studies；
- GSE200302/PRJNA824033、modalities、endpoints、replicates 不增加 study credit；
- primary measurement route 在模型结果前冻结，不允许结果后 route switch；
- external checkpoint、warm start、pretrained weight、external learned embedding/feature/logit、pseudolabel、teacher target、checkpoint-derived statistic、learned retrieval/reranker/score 的允许数量均为 0；
- raw replay 只是 reproducibility auxiliary；它尚未运行不推翻 processed-endpoint scoped qualification，但也不能被写成 independent reproduction；
- true-A2 contribution 恒为 0；
- qualification eligible 不等于 materialization authorized；
- dataset qualification 不等于 training rights；
- DEC020 adjudicator 不授权 GPU、model selection、next phase 或 scientific claim。

### 4.3 Exposure 语言

GSE200304 已有 aggregate public structural evidence 参与 protocol design：

- status=`DISCLOSED_NOT_UNTOUCHED`；
- 不得称为 untouched；
- 不得声称 no prior influence；
- 不得把选定 test 重命名为 untouched validation；
- prospective freeze boundary 为 DEC020 forward；
- full prior analytic-use attestation 当前仍不完整。

Single-study 可接受的是**明确披露的 exposed development**，不是伪造 untouched confirmation。

## 5. 14 个研究单元的正交使用角色

下表冻结的是 Single-study successor 的**允许用途**，不覆盖现有 qualification role/status，不产生 ordinary/A1/true-A2/canonical credit。

| Study unit | Single-study use role | 允许用途 | 明确禁止 |
|---|---|---|---|
| GSE200304 | `PRIMARY_SINGLE_STUDY_SOURCE_RELATIVE_DEVELOPMENT` | effect development、calibration、abstention、within-study error analysis、工程 proof-of-concept | cross-study、confirmatory、true-A2、broad generalization |
| GSE232572 | `DEVELOPMENT_ROBUSTNESS_ONLY` | loader、source-relative transform、replicate/SE、missingness/reject、跨实现一致性 | 独立确认、current credit |
| ENCSR854RUF | `DEVELOPMENT_EXPOSURE_POSITIVE_STRESS_ONLY` | ref/alt pairing、endpoint、replicate、exposure-positive robustness | untouched validation、current credit |
| GSE217518 | `DEVELOPMENT_FAIL_CLOSED_CROSSWALK_STRESS_ONLY` | endpoint/QC/rights、identity、crosswalk 缺失 STOP | A1/confirmatory claim |
| GSE149487 | `DEVELOPMENT_RECONSTRUCTION_REGRESSION_ONLY` | paper-faithful reconstruction、scale/transform、loader regression | A1/confirmatory credit |
| GSE269595 | `EXPLORATORY_DENSE_EVALUATOR_ONLY` | NDCG/regret/search-headroom/tie/censor/exposure sensitivity | unbiased true-A2 claim、role或credit |
| GSE114002 | `EXPLORATORY_WITHIN_ASSAY_ORDINAL_EVALUATOR_ONLY` | designed-library geometry、within-assay ordinal stress | confirmatory、跨 assay generalization |
| E-MTAB-10902 | `EXPLORATORY_SMALL_N_DENSE_SMOKE_ONLY` | evaluator qualitative smoke、small-N failure behavior | powered confirmation、把 rows 当 source N |
| GSE261709 | `NEGATIVE_CONTROL_PROVENANCE_IDENTIFIABILITY` | 证明 `7×19,220` geometry 不等于 allele identity；mapping 缺失时 STOP | replacement-A1、row/sequence claim |
| GSE207584 | `NEGATIVE_CONTROL_MAPPING_AND_CARTESIAN_RECONSTRUCTION` | endpoint universe、coverage、cartesian expansion、禁止 row-order pairing | source→candidate/true-A2 claim |
| GSE256185 | `NEGATIVE_CONTROL_POOL_PARSER_REJECT_QA` | pool/parser/reject/nonfinite endpoint QA | biological claim、qualification/credit |
| GSE145046 | `NEGATIVE_CONTROL_ABSOLUTE_VS_SOURCE_RELATIVE` | fixed-scaffold absolute landscape；识别 absolute 与 source-relative 差异 | ordinary/A1/true-A2 credit |
| GSE186455 | `NEGATIVE_CONTROL_LIBRARY_INDEPENDENCE_REFERENCE` | overlap/independence anchor | current credit、confirmatory qualifier |
| GSE246381 | `SEALED_EXTERNAL_FINAL_ONLY` | 非敏感 custody metadata、static guard、未来独立 one-shot | payload/label access、训练、调参、模型/metric/threshold selection、error analysis |

补充候选 `GSE113849` 和 `GSE295080` 只能标记为 `EXTERNAL_PREFLIGHT_CANDIDATE_ONLY_NOT_ACTIVE_STUDY_UNIT`。它们可以为 evaluator/negative-control 提供探索性证据，但不得进入 14-study KPI。

## 6. 非串行依赖架构

所有数据集不再排成一条“全部资格化后才能做模型”的串行链。新依赖图为：

```text
                           ┌─ Full Route A future rescue lane ──> future re-entry amendment
Current EVT060 ──> SS0 ────┼─ A2 evaluator G0 review/binding ───> SS4 evaluator freeze
                           ├─ A6 nonlearned CPU/G0 review ──────> SS6 engineering closure
                           └─ Single-study P0 closure ──────────> materialization/conformance
                                                                  └─ SS4 real split/evaluator/baseline PASS
                                                                       └─ critic implementation/gate review PASS
                                                                            └─ separate exactly-one G1 authority

Other datasets ──> development/evaluator/negative-control evidence only
GSE246381 ───────> sealed guard only; no dependency into SS0–SS8
```

规则：

- Full Route A rescue lane 不阻塞 SS0、G0 review、synthetic evaluator 或 nonlearned CPU theory；
- A2 evaluator 和 A6 implementation 独立 review，不能互相作为 evidence；
- 任一真实 dataset row access 仍需 dataset-specific authority；
- exactly-one G1 的依赖顺序固定为：successor 11/11 P0 → 独立 materialization authority → materialization conformance → SS4 real split/evaluator/baseline PASS → critic implementation/gate review PASS → 独立 exactly-one run authority；
- sealed 永不成为 ordinary development 的隐式依赖。

## 7. 分阶段 Goal 与详尽 TODO

### SS0 — 合同候选、独立审查与 activation package

**Goal：**把新问题、claim、角色和执行边界冻结为可审查的唯一 successor 候选。

**当前可直接执行：**是，仅限文档/静态工作。

**TODO：**

- [x] 记录本合同候选；
- [x] 记录 DEC027 automatic trigger=false；
- [x] 保留 Full Route A 3/2/1；
- [x] 冻结 current counts `1/1/0/6547`；
- [x] 冻结 14-study use-role map；
- [x] 冻结 active/dropped/prohibited claims；
- [x] 对本合同做独立 action-changing review；
- [ ] 只有在 review PASS 后，准备完整 DEC028 machine amendment bundle；
- [ ] 使用 fresh authority/runtime snapshot，分配新的 runtime event；不得预造 EVT061；
- [ ] authority sync 只激活 S0/P0 closure，不启动数据、CUDA 或 G1。

**PASS：**没有第二个 root contract；历史状态未改写；所有锁关闭；current counts 不变。

**STOP：**出现双 active G1、credit 变化、历史 FAIL/UNKNOWN→PASS、DEC027 trigger 被伪写为 true、或任何训练授权。

### SS1 — A2/A6 G0 current-HEAD forward-port 与独立重审

**Goal：**把旁支的 synthetic/zero-update evidence 与 current integration 对齐，但保持 nonactive。

**当前可直接执行：**是，限静态、合成、zero-update、nonlearned CPU。

**TODO：**

- [ ] 比较 current candidate 与已审 staging 语义；漂移则重新审查，不沿用旧 binding；
- [ ] forward-port A2 G0 evaluator candidate/review；
- [ ] forward-port A6 G0 implementation candidate/review；
- [ ] 验证 A2 synthetic split 为 outcome-blind、component-disjoint；
- [ ] 验证 candidate-minus-source、direction-normalized endpoint；
- [ ] 验证 biological SE 规则和 missing/nonfinite 不置零；
- [ ] 验证 A6 zero model/optimizer/CUDA/checkpoint/parameter update；
- [ ] 保持 A2 evaluator 与 A6 learner/guide 的接口隔离；
- [ ] 结论只能是 G0 partial/preparation，不得写 A2/A6 scientific PASS。

**STOP：**synthetic fixture 读取项目 rows；evaluator 接触 guide/model-selection output；产生 runtime/model artifact；或 staging bytes 被当成 active authority。

### SS2 — Single-study successor 11-axis metadata-only P0

**Goal：**冻结旧 DEC026 结果，建立新 scope 的独立、aggregate-only successor P0。

**当前可直接执行：**只能起草 schema、fixture、failure codes；生产 P0 需 DEC028 activation。

**TODO：**见第 8 节逐项表。

**Output：**仅一个 aggregate P0 record；不得包含 row/member ID、sequence、endpoint value、SE、split assignment、model、optimizer、checkpoint 或 device payload。

**PASS：**11/11 的 `status` 字段全部精确为 `PASS`，并且只返回 `ELIGIBLE_TO_REQUEST_MATERIALIZATION_NOT_G1_NOT_LAUNCHED`。scoped PASS 必须另有 `scope` 字段，不能把自定义字符串当作 PASS-equivalent。

**STOP：**任一非 PASS 时，严格停在 materialization、data row、CUDA、model、optimizer、checkpoint、parameter update 和 training 之前。

### SS3 — 6,547 membership-preserving materialization 与 conformance

**Goal：**在 11/11 P0 和独立 materialization authority 后，生成 GSE200304 private development asset。

**当前可直接执行：**否。

**TODO：**

- [ ] 绑定 DEC019 canonical membership contract 和 DEC020 scratch route；
- [ ] 仅允许一次 membership-preserving materialization；
- [ ] 验证 public join 6,772、排除 225 NA、canonical exactly 6,547 的既有规则；
- [ ] 验证 membership 在 effect/significance/label 之前冻结；
- [ ] 验证 source/candidate/endpoint/group/context/SE/rights/exposure 完整；
- [ ] 生成 private asset、aggregate manifest、aggregate reject summary；
- [ ] 不输出 member ID、sequence、row effect/SE 或 split assignment。

**STOP：**行数不为 6,547；任何 add/drop/relabel/resample；rights/schema 不完整；或 model/CUDA 在 conformance 之前初始化。

### SS4 — GSE200304 development split assignment、evaluator 和 baseline freeze

**Goal：**建立单研究、非 confirmatory 的开发 evaluator。

**当前可直接执行：**schema/synthetic/baseline contract 可做；真实 membership/split 不可做。

**TODO：**

- [ ] 按 P0.7 已冻结且从未看 outcome 的算法、grouping keys、salt 和 subrole contract 构建真实 component graph；
- [ ] 生成真实 split assignments；P0 阶段的 `split_assignment_count` 必须为 0；
- [ ] 对真实 assignments 做 source/exact-duplicate/near-duplicate/reverse-pair zero-leakage conformance；
- [ ] 本 successor 明确选择 `ONE_FROZEN_COMPONENT_DISJOINT_TRAIN_CALIBRATION_TEST_SPLIT_V1`，不实例化 nested `5×4` learned CV；
- [ ] one biological group, one vote；
- [ ] primary metric：within-study Spearman；
- [ ] diagnostics：MAE、calibration、coverage-risk、abstention；
- [ ] finite positive SE 与 biological replication gate；
- [ ] missing/censored/nonfinite 永不置零；
- [ ] evaluator 不接收 guide output，不用于 checkpoint/model selection；
- [ ] freeze baseline set、salt、fold、metrics、thresholds、compute ledger；
- [ ] 冻结 `authorized_execution_count=1`、`optimizer_fit_count=1`、`fold_model_count=1`、`checkpoint_count=1`、`final_refit_count=0`、`seed_count=1`；
- [ ] calibration split 只做预先冻结的 calibration，test split 只做一次 terminal evaluation；
- [ ] N=156、80% power、CI width 0.30 只作为 planning/diagnostic，不产生 portfolio credit 或 A2 PASS。

现有 nested `5×4` authority 作为历史/planning 证据保留，但不授权本次 learned execution。未来若要运行 composite CV，必须另立 authority，精确列出全部 optimizer fits、fold checkpoints、compute ledger 和 final-refit policy；不得把多个 fit 藏在“one run”中。

**STOP：**source/duplicate/near-duplicate/reverse-pair/candidate/context leakage；空 split；非法 SE；constant-rank prediction；结果后换 salt/split/metric；额外 fit/refit；或 evaluator 参与模型选择。

### SS5 — Exactly-one source-relative critic G1 development run

**本合同的裁定：**当前唯一 G1 定义为 `GSE200304_SOURCE_RELATIVE_CRITIC_G1`。DEC028 必须明确 supersede DEC026 的 future-G1 task meaning；旧 DEC026 `A6_G1_BRIDGE` 的 `3 PASS / 7 FAIL / 1 UNKNOWN` 和 zero-run 结果保持历史事实，但它不能在 DEC028 下自动解锁或另行再启动一次 A6 learned run。

作出这一选择的原因是：当前 A6 learned base/value draft 的 terminal tilt 需要一个 `FROZEN_CALIBRATED_LOWER_CONFIDENCE_BOUND_FROM_ORDINARY_DEVELOPMENT_ROLE`。在 source-relative critic 尚未运行时，这个输入并不存在。把唯一 G1 先用于 A6 learned base/value 会形成无法诚实关闭的 P0.9/P0.10 循环。Single-study mainline 因而先训练一次 critic，产生 development-only 的冻结 calibration/LCB evidence；A6 learned base/value 继续停在 nonlearned/G0 preparation，且不由本合同授权。

**当前可直接执行：**否。

**前置条件：**

- DEC028 已激活；
- successor P0 11/11 PASS；
- 6,547 materialization conformance PASS；
- split/evaluator/baseline freeze PASS；
- source-relative critic implementation、baselines、evaluator isolation 和 gate bundle 已独立 review/binding PASS；
- 独立 `ACTIVE_FOR_THIS_G1_ONE_RUN_ONLY` authority 已签发。

**必须冻结：**

- one seed；
- one run ID；
- exactly one optimizer fit；
- one pre-frozen train/calibration/test split；
- one architecture、optimizer、LR、schedule、compute budget；
- scratch-only、random initialization、zero external learned input；
- terminal checkpoint only；
- no HPO、no early stopping、no best-checkpoint selection；
- no seed/checkpoint/threshold retry；
- CUDA owner/device preflight；
- aggregate result 与 failure bundle；
- private checkpoint 和 private row predictions。

执行计数必须精确为：

```yaml
authorized_execution_count: 1
optimizer_fit_count: 1
fold_model_count: 1
checkpoint_count: 1
final_refit_count: 0
seed_count: 1
```

**Terminal outputs：**

- development-only aggregate prediction/calibration/error record；
- private terminal critic checkpoint；
- frozen calibration/LCB manifest candidate；
- failure bundle；
- `A6_LEARNED_BASE_VALUE_AUTHORIZED=false`。

critic G1 通过不会自动授权 A6 learned base/value。若未来要消费该 LCB 运行 A6 learned base/value，必须有新的 successor protocol、重新验证输入 binding、独立 P0 和独立 owner run authority；本合同不预授权该运行。

**STOP：**P0 漂移；membership 不一致；CUDA ownership 冲突；NaN/OOM/gate failure；第二 run；结果驱动 retry；或把 development metric 写成 qualification。

### SS6 — A6 nonlearned engineering-theory completion 与 future learned handoff

**Goal：**在不依赖更多 qualified studies 的情况下，先完成 legal CTMC、exact reference、base recovery 和 nonlearned engineering evidence，并为未来 learned-potential successor 准备明确输入合同。

**当前可直接执行：**nonlearned CPU、static、synthetic only。

**TODO：**

- [ ] source-anchored acyclic edit DAG；
- [ ] budgets `{1,3,5}`；
- [ ] STOP、hard legality、alias aggregation；
- [ ] support floor、algorithmic time 语义；不得声称 physical kinetics；
- [ ] exact enumeration/DP independent reference；
- [ ] general time-inhomogeneous exactness；
- [ ] 96-graph exact-reference suite；
- [ ] 冻结 future learned-potential edge error、terminal TV、legality、budget、no-graph-deletion gates，但不执行 learned run；
- [ ] 冻结 future base-then-value、禁止 joint optimization 的 protocol；
- [ ] 验证 future terminal tilt 只能消费 SS5 产生并独立接受的 frozen calibration/LCB manifest；
- [ ] future learned run 仍需新 successor authority，no HPO/best-checkpoint/automatic retry。

现有 CPU exact/flow-base partial evidence只能写 `IN_PROGRESS`。SS6 在本合同下不能执行 learned base/value parameter update。即使 nonlearned 工程 gate 全过，也不自动建立 biological claim、guidance superiority、A7 或 Full Route A PASS。

### SS7 — 其他数据集的非主锚用途

**Goal：**让其他数据集产生真实价值，但不污染主 claim。

**当前可直接执行：**只读 metadata、现有 aggregate report、synthetic fixture；真实 rows 仍需 dataset-specific authority。

**TODO：**

- development：loader、endpoint、SE、missingness、reconstruction regression；
- evaluator：dense geometry、ties、censoring、small-N behavior；
- negative-control：mapping absence、cartesian expansion、absolute-vs-relative、overlap/leakage、parser/reject；
- external candidates：仅 discovery/preflight，不进入 KPI；
- 每个结果标记 `DEVELOPMENT`、`EXPLORATORY` 或 `NEGATIVE_CONTROL`；
- 不从这些结果推导 current credit 或 cross-study confirmation。

### SS8 — Claim adjudication、paper/failure package 与 Full Route A re-entry

**Goal：**选择证据实际支持的最高表述，并保留升级路径。

**TODO：**

- [ ] 汇总 GSE200304 effect development 与 calibration；
- [ ] 汇总 A6 legal CTMC/engineering evidence；
- [ ] 汇总 development/evaluator/negative-control datasets；
- [ ] 对每条 claim 标记 established/not established/invalidated；
- [ ] 若 learned run 失败，发布诚实 failure package，不自动 retry；
- [ ] 若只有 within-study evidence，标题、摘要、图表和讨论不得使用 cross-study/confirmatory/generalization wording；
- [ ] Full Route A re-entry 只能由未来新 qualified independent studies 和独立 amendment 触发；
- [ ] Single-study 结果不自动改变 3/2/1 gate。

### SS9 — Sealed boundary

GSE246381 在 SS0–SS8 中无依赖、无 payload access、无 latent read path。只允许 non-sensitive custody metadata 和 static guard。未来 one-shot sealed final 需要 A0–A9 全部通过、checkpoint overlap audit、sealed readiness report 和单独 A10 owner authorization。Single-study 合同不提供该授权。

## 8. Successor P0：旧结果、关闭动作和不变的失败语义

旧 DEC026 的结算结果固定为 `3 PASS / 7 FAIL_CLOSED / 1 UNKNOWN`，`G1 launched=false`。新合同不得重跑覆盖旧 record；只能创建新的 successor P0 lineage。

| Gate | 旧 DEC026 status | Successor 关闭动作 | 能否因改题豁免 |
|---|---|---|---|
| P0.1 `INPUT_MEMBERSHIP_AND_BINDING` | `FAIL_CLOSED_BINDING_ABSENT` | 绑定 6,547 membership contract、source lineage 和唯一 prospective materialization recipe；不要求提前读 rows | 否 |
| P0.2 `PRIOR_USE_ATTESTATION` | `UNKNOWN_NOT_ASSERTED` | owner 提交 known-use/unknown-use disclosure；successor 只有在披露合同闭合后才写 `status=PASS`、`scope=DISCLOSED_EXPOSED_DEVELOPMENT_ONLY`、`predecessor_historical_status=UNKNOWN_NOT_ASSERTED`。禁止 untouched/confirmatory，旧 UNKNOWN 不被改写 | 只能改变 successor 适用标准，不能伪造历史 PASS |
| P0.3 `EXPOSURE_ROLE` | `PASS` | 继承 `EXPOSED_DEVELOPMENT_ONLY`；不得反复重审已结算事实 | 不适用 |
| P0.4 `RIGHTS` | `FAIL_CLOSED_INTENDED_INTERNAL_TRAIN_RIGHTS_NOT_BOUND` | 绑定 internal process/train/evaluate 权利和输出边界 | 否 |
| P0.5 `SCIENTIFIC_ROW_CONTRACT` | `FAIL_CLOSED_COMPLETE_ROW_CONTRACT_NOT_BOUND` | 冻结完整 schema、authority、membership、endpoint、group、SE、rights、exposure；materialization 后再做 conformance | 否 |
| P0.6 `SCRATCH_ONLY_ROUTE` | `PASS` | 继承 DEC020 zero-external-learned-input/random-init route | 不适用 |
| P0.7 `PROSPECTIVE_SPLIT` | `FAIL_CLOSED_PROSPECTIVE_SPLIT_NOT_FROZEN` | 仅冻结 pre-materialization 的算法、grouping keys、salt、fold/subrole contract 和 zero-leakage rules；`split_assignment_count=0`。真实 component graph、assignments 与 conformance 只在 SS4 执行 | 否 |
| P0.8 `SINGLE_RUN_POLICY` | `FAIL_CLOSED_SINGLE_RUN_POLICY_NOT_ACTIVE` | 冻结 one seed/run/architecture/optimizer/schedule/budget/checkpoint/stop/no-retry，并精确绑定 execution=1、optimizer fit=1、fold model=1、checkpoint=1、final refit=0 | 否 |
| P0.9 `EXECUTABLE_SCIENTIFIC_GATES` | `FAIL_CLOSED_EXECUTABLE_GATE_BUNDLE_NOT_BOUND` | 建立 source-relative critic 的 active、fail-closed G1 gate bundle；A2/A6 G0 evidence只能作为接口输入，不能代替 critic gates | 否 |
| P0.10 `SUCCESSOR_LEARNED_RUN_IMPLEMENTATION` | `FAIL_CLOSED_SUCCESSOR_NOT_BOUND` | DEC028 明确 supersede旧 A6-G1 task meaning，并绑定 `GSE200304_SOURCE_RELATIVE_CRITIC_G1` implementation；A6 learned run保持未授权 | 否 |
| P0.11 `STATE_LOCKS` | `PASS` | 原子保留 no-promotion/no-A7/no-sealed/no-model-selection/no-retry | 不适用 |

任何 gate 缺失、部分、未知或非 PASS，都必须返回 STOP。Successor P0 通过也只产生**申请一次 G1 的资格**，不启动 G1。

## 9. A2 evaluator 与 A6 implementation 的独立边界

### A2 G0

允许：

- synthetic connected-component grouping；
- outcome-blind split planning；
- aggregate evaluator schema；
- candidate-minus-source/direction-normalized metadata validation；
- power/CI planning calculation；
- zero project-row I/O validation。

不允许：

- final membership 或 split assignment；
- real evaluation；
- measured-neighborhood NDCG/regret claim；
- guide-evaluator feedback；
- formal power/A2 PASS；
- training/GPU。

### A6 G0

允许：

- shape/interface plan；
- pure formula validation；
- synthetic CPU exact-DAG adapter；
- support/STOP/budget/alias fixtures；
- zero-update validation。

不允许：

- Torch model、dataloader、optimizer、CUDA probe；
- project data rows；
- checkpoint/runtime output；
- parameter update；
- A6 PASS、L3、A7。

两者的 review 结果不能互为证据；二者都必须 forward-port 到 current authority 后再独立重审。

## 10. Artifact、I/O 与隐私边界

### 10.1 当前允许输出

- 本合同及 review；
- static/synthetic/zero-update test results；
- aggregate metadata、gate、schema、rights 和 split-contract audit；
- aggregate successor P0 candidate schema；
- nonlearned CPU theory/evaluator evidence。

### 10.2 当前禁止触点

- materialization exercise 或 row read；
- member ID、sequence、逐行 abundance/effect/slope/SE；
- split assignment；
- private/sealed payload；
- CUDA/device probe；
- model/optimizer construction；
- checkpoint read/write；
- parameter update/training；
- model/seed/checkpoint/threshold selection；
- qualification、credit、canonical mutation；
- A6/L3/A7/next-phase promotion。

### 10.3 Aggregate-only 不等于弱 provenance

报告可以隐藏 member payload，但仍必须记录 route、gate、fact class、reason、counts、analysis unit 和 authority lineage。不得用 aggregate 输出掩盖无法识别 source/candidate、endpoint 或 independent group 的问题。

## 11. Fail-closed 状态机

```text
DRAFT DEC028
  └─ independent review PASS
       └─ complete machine authority + runtime sync
            └─ SS1 G0 bindings + critic implementation review + SS2 successor P0
                 ├─ any P0 nonpass -> STOP_BEFORE_DATA_CUDA_MODEL
                 └─ 11/11 PASS -> eligible, G1 still not launched
                      └─ materialization authority + conformance
                           ├─ mismatch -> STOP_BEFORE_MODEL
                           └─ SS4 real split/evaluator/baseline PASS
                                └─ critic implementation/gate review PASS
                                     └─ exactly-once source-relative critic G1 authority
                                     ├─ failure -> TERMINATED_SAFELY_WITH_EVIDENCE; no retry
                                     └─ terminal evaluation -> claim adjudication
```

每次 STOP 都保留失败证据和下一条可改变行动的修复，不创建防御性空壳，不为了“推进”把 unknown 写成 pass。

## 12. 验收标准

### 12.1 本合同候选完成标准

- [x] Full Route A 3/2/1 被保留；
- [x] Single-study 被定义为另一科学问题，而不是 gate relaxation；
- [x] DEC027 trigger=false 被诚实记录；
- [x] current counts 为 1/1/0/6547；
- [x] 14 个 canonical study units 各有且只有一个正交 use-role；
- [x] GSE246381 sealed locked；
- [x] old DEC026 3/7/1 和 G1-not-launched 被保留；
- [x] 11 P0、阶段 Goal、TODO、I/O、PASS/STOP 全部写明；
- [x] A2/A6 独立 review；
- [x] 无 data/CUDA/model authority。

### 12.2 正式 DEC028 activation 标准

- complete machine-readable amendment；
- current config、qualification config、supersession、decision log、interim、role/task/split/claim registries 一致；
- registry manifest、validator 和 integrity tests 闭合；
- static validator issue count 0；
- focused authority tests PASS；
- fresh runtime authority sync；
- runtime event 只改变 operational route/phase，不改变 counts、credit、scientific claim 或 locks；
- G1 remains false。

### 12.3 Scientific exit 标准

- 只按实际证据选择 `DEVELOPMENT_RESULT`、`ENGINEERING_METHOD_RESULT`、`NEGATIVE_RESULT` 或 `NOT_ESTABLISHED`；
- 不因工程通过宣称 biological generalization；
- 不因单次 G1 完成宣称 superiority；
- 不因 dense evaluator reachability 宣称 true-A2 qualification；
- 任何更强 claim 需要独立 successor protocol。

## 13. 立即执行队列

按依赖与信息增益排序：

1. **独立审查本合同候选**：只找 action-changing 的 claim/authority/P0 矛盾；
2. **A2/A6 G0 current-HEAD reconciliation**：可立即并行，不等待数据资格；
3. **起草 successor P0 schema 和 failure codes**：不运行生产 P0；
4. **GSE200304 metadata-only closure plan**：P0.1/P0.2/P0.4/P0.5/P0.7 的证据责任人和最小输入；
5. **A2 evaluator/baseline candidate freeze**：synthetic-only；
6. **A6 nonlearned CPU exactness continuation**：不重复已经 PASS 的 exact toy；
7. **准备完整 DEC028 authority bundle**：只有合同 review PASS 后；
8. **激活后运行 successor metadata-only P0 一次**；
9. **任一 P0 非 PASS，停止**；
10. **11/11 PASS 后只申请 materialization authority**；
11. **materialization conformance 与 SS4/critic review 全部 PASS 后，另行申请 exactly-one G1 authority**。

## 14. 外部依赖与 owner-only 输入

以下不能由实现代码推断：

- GSE200304 full prior analytic-use disclosure；
- intended internal process/train/evaluate rights 的最终 owner/legal binding；
- DEC028 active authority 的签发；
- exactly-once G1 的最终 run authority；
- future A6 learned base/value 是否在 critic LCB 独立接受后值得单独申请 successor run；
- future sealed A10 authorization。

缺少这些输入时，继续做允许的静态/合成/非学习工作，并在对应 gate 停止；不得用默认值代替 owner 决策。

## 15. 历史保存、失败处理与 Full Route A re-entry

- DEC020/DEC025/DEC026/DEC027 与 EVT060 原字节和状态保留；
- 旧 P0 record 不覆盖、不重跑成新状态；
- 所有 dataset-specific FAIL/UNKNOWN/NOT_RUN 保留；
- Single-study use-role 不改变 qualification role；
- GSE261709 只有在出现与 exact 19,220-token universe 绑定的 publisher instance mapping 后，才可另行申请重启；
- GSE269595 reachability 不等于 true-A2；
- Full Route A re-entry 需要新的独立合格研究、正式 promotion 和独立 amendment；
- Single-study 结果、失败或 paper 都不会自动关闭 Full Route A。

## 16. 从候选到正式 DEC028 的最小治理路径

单独的本 Markdown 不能激活路线。正式转换至少需要：

1. 新增 machine-readable `docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec028.yaml`，引用本合同；
2. 新增 Single-study machine protocol/current projection；
3. 同步 root executable config、qualification config、supersession、decision log、A1 interim；
4. 在 data-role registry 中新增正交 `single_study_track_use_role`，不覆盖 qualification 字段；
5. 同步 task、split、task-split、claim-evidence registries；
6. 更新 manifest、validator 和 focused integrity tests；
7. 独立 review；
8. fresh runtime replay 后生成未预分配的新 authority-sync event；
9. 保持 `1/1/0/6547`、claim `NOT_ESTABLISHED`、所有 training/GPU/model/A7/sealed locks false；
10. 进入 `SINGLE_STUDY_S0_AUTHORITY_AND_P0_CLOSURE`，而不是 G1。

在上述路径完成前，本合同的唯一合法作用是：统一战略、冻结科学问题、组织并行工作、生成可审计 TODO，并约束现有 authority 已允许的静态/合成/zero-update/nonlearned CPU 工作。
