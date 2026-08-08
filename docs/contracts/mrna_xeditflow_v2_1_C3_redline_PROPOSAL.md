# mRNA-XEditFlow V2.1 C3 Redline Proposal

```yaml
revision_status: PROPOSED_REDLINE_NOT_ACTIVE
active_contract_mutation: NOT_AUTHORIZED
selected_route: BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY
parent_contract_path: /Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 最新合同-v2.md
parent_contract_sha256: 9c79edd819e45551974bcfeb14a400dd504c55c0a7c869e456e638daf49f1c1e
parent_contract_authority_role: SOURCE_PROPOSAL_NOT_ACTIVE
current_active_executable_contract_path: docs/contracts/mrna_xeditflow_goal_v1_1.md
current_active_executable_contract_sha256: fc9c1c882efbaa4c1e86f4da2e1be64e219755fb9c5941da4b4309793d3d8c2f
current_active_executable_config_path: configs/mrna_xeditflow_contract_v1_1.yaml
current_active_executable_config_sha256: b3be70e765fb8285996487815ee6a4494ca4cc7fb503dae2901b40a0382d83cf
proposal_date: 2026-08-07
```

本文件是供逐项审阅的 redline 提案，不是合同，不激活新路线，不替换、覆盖或解释性修改 parent contract。这里绑定的 `mrna 最新合同-v2.md` 本身是 source proposal，不是当前 active executable authority；当前执行权威仍是上述 v1.1 contract/config。任何生效版本都必须另建文件、保留 source proposal 与 active v1.1 的原始字节，显式建立 supersession/lineage，并由用户再次明确批准；只绑定 source proposal 不能激活本提案。

## 1. 冻结且不得修改的边界

以下约束逐字保留：

- macro sign accuracy >= 0.60；
- effect gate 为 conjunctive gate，不得由 Spearman、selection gain、abstention、某个 seed 或新阈值替代；
- 模型、阈值、metric、zero policy、split、seed、route 和 claim 的开发限于 ordinary data；
- sealed 只在所有 ordinary gates、checkpoint、分析代码、阈值与 seeds 冻结后进行一次 primary confirmation；
- sealed 前后都禁止 post-hoc gate substitution；unblinding 后禁止为 confirmatory claim 重新训练、调阈值、换 seed、改 endpoint 或重复 primary access；
- 当前 O0、G1、E0 不因本 redline 解锁。

## 2. 建议替换论文中心问题

### Current direction to delete

> 将“exact density-ratio guidance on a legal edit CTMC”预设为主创新，并把 benchmark、critic、Flow 和 exact guidance 串成一个不可分割的论文成功条件。

### Proposed replacement

> 给定一个来源明确的 source mRNA、冻结的 endpoint/estimand、assay/context、有限编辑预算和严格合法性约束，能否从 provenance-controlled 的 ordinary intervention data 中学习 source-relative measured effect，并在 source-relation-component-disjoint、信息集匹配和计算预算匹配的条件下，比论文级强基线更可靠地预测 effect sign、排序 measured local neighborhoods？只有在 critic gate 与 search-headroom gate 均通过后，才评估 DGM/Doob-derived learned guidance 是否改善 ordinary held-out quality-cost frontier。

建议主线：

```text
mRNA-EditBench authority and provenance
    -> six information-set-matched baseline tracks
    -> preregistered source-conditioned critic candidates
    -> conjunctive ordinary effect gate
    -> measured-neighborhood ranking/search headroom
    -> optional legal Edit Flow
    -> secondary DGM/Doob-derived empirical adaptation
    -> one-time sealed confirmation only after complete freeze
```

该结构允许 Flow/guidance 失败后保留真正通过 effect gate 的 benchmark+critic 结果；它不允许在 effect gate 失败时用 Flow 工程结果或其他指标绕过 sign gate。

## 3. C3 逐字替换提案

### Current heading to delete

> C3. Exact guidance on a legal edit CTMC

### Proposed heading

> C3. DGM/Doob-derived guidance adaptation under mRNA-specific legal support; new-theory claim locked

### Proposed body

**Prior-art disposition.** DGM 已给出离散 flow matching 的 target transition-rate / density-ratio guidance framework；Doob `h`-transform 已覆盖一般 Markov process 与 jump-process conditioning；Edit Flows 已将 insertion/deletion/substitution 写成 sequence-edit CTMC 并使用 auxiliary edit state。因而，terminal reweighting、state-dependent legal action mask、non-Cartesian edit graph、sequence edit CTMC、auxiliary alignment 或一次 learned guidance-head forward 本身都不是本项目的一般理论原创。

**Default project claim.** 项目可以把上述理论适配到 source-anchored mRNA legal edit support，并在冻结 critic、相同 action space、相同 candidate budget、相同总 forward-equivalent 与 wall-time frontier 下，比较 learned rate guidance、first-order approximation、latent/rate CFG、generate-then-rerank、beam、enumeration 和 unguided flow。默认状态为：

```text
C3_GENERAL_EXACT_GUIDANCE_ORIGINALITY
  = REJECT_AS_ORIGINAL_GENERAL_THEORY

C3_MRNA_DGM_DOOB_ADAPTATION
  = LOCKED_PENDING_ORDINARY_EMPIRICAL_GATE

C3_DYNAMIC_LEGAL_GRAPH_NEW_THEORY
  = PROHIBITED_UNLESS_G0_NEW_THEORY_PASS
```

**Allowed wording before a new-theory proof gate.**

> If this proposal is activated and the preregistered ordinary experiment is subsequently run, we will evaluate a DGM/Doob-derived learned rate-guidance adaptation under mRNA-specific legal edit support.

**Disallowed wording before `G0_NEW_THEORY_PASS`.**

```text
first exact guidance for biological edit graphs
novel exact density-ratio guidance
exact real-mRNA guidance
one-pass end-to-end exact algorithm
exactness established by toy terminal replay alone
```

**Learned target boundary.** 当 terminal reward、density ratio、`h` function 或 guidance ratio 来自 learned critic/head 且无生物 target 误差界时，真实 mRNA artifact 必须记录 `target_epistemic_status: LEARNED_SURROGATE_WITHOUT_BIOLOGICAL_ERROR_BOUND`。该 epistemic 状态与计算实现是否相对这个 learned surrogate 精确无关：即使有限 toy reference 相对指定 learned terminal target 完整枚举，也不代表真实生物最优分布已被识别。

## 4. Exactness 三轴标签

所有 G0/G1 artifact、表格、配置与 manuscript claim 必须同时填写三个正交字段，不得只写 bare `exact`，也不得用任一轴推断另一轴：

1. `computational_exactness`：只描述相对已声明 target/path 的计算关系。允许值为 `EXACT_FINITE_ENUMERATION_RELATIVE_TO_DECLARED_TARGET`、`EXACT_BY_THEOREM_IMPLEMENTATION_MATCH`、`FIRST_ORDER_NUMERICAL_APPROXIMATION`、`GENERAL_NUMERICAL_APPROXIMATION` 或 `NOT_ESTABLISHED`。该字段不声明 target 是真实生物 target，也不声明方法原创。
2. `target_epistemic_status`：只描述 target 的识别与知识状态。允许值为 `DECLARED_FINITE_REFERENCE_TARGET`、`DECLARED_MODEL_TARGET`、`EMPIRICAL_ENDPOINT_TARGET_WITH_SAMPLING_UNCERTAINTY`、`LEARNED_SURROGATE_WITHOUT_BIOLOGICAL_ERROR_BOUND` 或 `BIOLOGICAL_OPTIMUM_NOT_IDENTIFIED`。该字段不声明计算精确或方法原创。
3. `method_provenance`：只描述方法来源。允许值为 `FINITE_STATE_REFERENCE_CALCULATION`、`DGM_DOOB_DERIVED_ADAPTATION`、`GENERATE_THEN_RERANK` 或 `PROJECT_NEW_THEORY_AFTER_G0_PASS`。该字段不声明计算精确或生物 target 有效。

例如，有限 toy 完整枚举应写为：

```yaml
computational_exactness: EXACT_FINITE_ENUMERATION_RELATIVE_TO_DECLARED_TARGET
target_epistemic_status: DECLARED_FINITE_REFERENCE_TARGET
method_provenance: FINITE_STATE_REFERENCE_CALCULATION
```

真实 mRNA learned guidance 默认应写为：

```yaml
computational_exactness: NOT_ESTABLISHED
target_epistemic_status: LEARNED_SURROGATE_WITHOUT_BIOLOGICAL_ERROR_BOUND
method_provenance: DGM_DOOB_DERIVED_ADAPTATION
```

若 base generator 实际先生成、critic 只排序，则 `method_provenance` 必须为 `GENERATE_THEN_RERANK`，且不得声称 guided generation。

## 5. Effect model 与强基线 redline

删除任何预先指定 SparseEditFormer 为论文 winner 的措辞，替换为：

> 预注册 candidate comparison 先运行 full-pair unrestricted Siamese 强比较器与 `R2-style no-explicit-study` development adaptation，再运行 A exact potential-difference critic、A low-rank parameter-matched algebraic diagnostic、B sparse edit-set additive critic，以及仅在 action support policy publication-qualified 后运行的 B interaction critic。full-pair 与 A 必须在相同 eligible rows、source relation components、seeds、训练预算和信息集下比较；low-rank diagnostic 与尚未完成 architecture crosswalk/study-proxy audit 的 R2-style adaptation 都不得替代 strongest-baseline gate。

**R2 comparator boundary.** 现有 current R2 包含显式 study embedding，只能作为 `HISTORICAL_DIAGNOSTIC_ONLY`；它不属于 `PAIRED_SEQUENCE_NO_MEASURED_ANCHOR` 同信息集 leaderboard，不得作为 strongest same-information-set baseline，也不得用于满足 baseline-win gate。当前新代码只能称为 `R2-style no-explicit-study cross-attention adaptation`：它没有显式 study-ID 输入，但 endpoint/context 仍可能是 study proxy，且其 shared encoder、pooling、edit summary 与 heads 并非历史 R2 的单变量消融。因此在冻结 architecture crosswalk 与 hash-bound semantic study-proxy audit 前，它也只能是 development diagnostic。若未来建立忠实 `R2-no-study` comparator，必须从头训练、保留历史 R2 除 study branch 外的预注册模块、从 predictive schema/vocabulary/embedding/checkpoint/forward/inference API 移除显式 study，并证明 endpoint/context/region 不构成未披露的 deterministic proxy。study 只可作为 outcome-blind split、balanced sampling、分层评估和审计键；不得进入预测张量、可学习参数或推断 API。

六类信息集分别排行，禁止混榜：

1. no-anchor zero/prior；
2. edit-only；
3. candidate-only；
4. paired sequence without measured anchor；
5. measured-source-anchor-assisted；
6. foundation-exposure-stratified。

“exact antisymmetry”“reference-versus-variant scalar difference”“explicit mutation token”“pairwise epistasis”均已有一般 prior art；A/B 的价值必须由 mRNA-specific source/context/endpoint formulation、严格 leakage controls、ordinary measured effect 增益和可解释负对照建立，不得宣称这些组件本身首创。

## 6. Gate redline

### Effect gate — unchanged and conjunctive

```yaml
macro_delta_spearman: ">=0.25"
macro_sign_accuracy: ">=0.60"
stable_selection_metric: "PASS under a preregistered non-pathological definition"
strongest_same_information_set_baseline_win: true
```

任一项失败均为 `HOLD_M0_REPAIR` 或在重复、预注册 ordinary repairs 后为 `NO_GO_EFFECT`。不得把 full primary sign 改为 selective sign，亦不得通过 abstention 隐藏失败。新的 ordinary repair 必须新建 lineage 并完整重跑强基线。

### Empirical guidance gate — secondary

只有 `GO_EFFECT` 与 measured-neighborhood headroom 均成立后才运行：

```yaml
ndcg_gain_over_strongest_matched_baseline: ">=0.05"
top_decile_recall: ">=0.70"
normalized_regret: "<=0.10"
hard_legality: "100%"
ordinary_ood_margin: "within preregistered bound"
sealed_during_G1: "NOT_RUN"
```

必须匹配 candidate pool、legal support、generator NFE、总 forward-equivalent、wall-time frontier、memory、seeds 和 beta grid；并使用不同于 guidance critic 的 independent evaluator。成功只开放 empirical adaptation claim，不开放 new-theory claim。

### New-theory gate — locked

`G0_NEW_THEORY_PASS` 必须同时具备：

- formal state、continuous-time generator、target/path measure、dynamic legal actions、auxiliary variables、budget/STOP states；
- 相对 DGM、Doob `h`-transform、general jump processes 与 Edit Flows 的严格非平凡差集；
- 不依赖完整 terminal enumeration 的新 factorization/estimator/algorithm；
- exact-`h` correctness theorem；
- learned/approximate `h/r/g` 到 rates、path measure 或 terminal law 的误差传播界；
- 包含 action enumeration、base/critic/guidance/sampler 的完整 complexity；
- 独立 proof review；
- 与主实现不同的 checker，在非等度、dynamic-support 小图上验证实际 transition rates、generator row sum、support、absorbing states 与 terminal law；
- prior-art feature matrix 人工复核后仍有新命题。

缺任一项只能得到 `G0_ADAPTATION_REFERENCE_PASS`，不得恢复 “novel exact guidance” 措辞。

## 7. G0/G1 名称 redline

- `Phase G0: exact guidance theory and toy verification` 改为 `Phase G0: prior-art-bounded guidance theory audit and finite-state reference verification`。
- G0 分成 `G0_ADAPTATION_REFERENCE_PASS` 和 `G0_NEW_THEORY_PASS`；前者不蕴含后者。
- `Stage 3: exact guidance head` 改为 `Stage 3: DGM/Doob-derived learned guidance head`。
- `Phase G1: mRNA exact-guidance integration` 改为 `Phase G1: mRNA legal-support DGM/Doob-derived guidance integration`。
- G1 限 ordinary held-out，不访问 sealed；若不胜 strongest matched search/reranking baseline，停止扩大而不调 metric/reward/beta。

## 8. Claim ladder redline

拆分四层，不得互相代替：

1. `BENCHMARK_CRITIC`：数据 authority + six-track baseline + conjunctive effect gate；
2. `MEASURED_NEIGHBORHOOD_OPTIMIZATION`：在未见 source component 的真实 measured pools 上证明 ranking/search utility；
3. `C3_GUIDANCE_ADAPTATION`：ordinary matched-compute 的 DGM/Doob-derived empirical gain；
4. `C3_NEW_GUIDANCE_THEORY`：只有 `G0_NEW_THEORY_PASS` 后开放。

Flow/guidance 失败不应错误否定真正已通过的 benchmark+critic；benchmark/critic 成功也不得反向开放 theory claim。Effect gate 失败时四层都不能通过 generative engineering 绕开。

## 9. Sealed redline

```text
E0_SEALED_UNLOCK requires:
  GO_DATA
  GO_EFFECT with macro sign accuracy >= 0.60
  stable metric registry FROZEN
  checkpoint/config/seed/code hashes FROZEN
  ordinary OOD and baseline comparison complete
  evaluator and one-shot state machine independently verified
```

sealed 只用于冻结方案的一次确认，不是路线赌博。结果无论 PASS/FAIL 都禁止恢复训练、recalibration、threshold/zero/metric/endpoint/split/seed/checkpoint substitution 或第二次 primary access。任何 unblinding 后分析必须标记 `POST_UNBLINDING_EXPLORATORY`。

## 10. 生效流程

本提案只有在以下全部动作完成后才可能生效：

1. 用户逐项批准 redline；
2. 新建带版本号和 parent lineage 的后继合同，不原地覆盖；
3. 对合同、claim matrix、metric registry、prereg 和 code config 做一致性检查；
4. 记录新旧 SHA-256；
5. 明确说明修订发生在 sealed unblinding 前；
6. 独立审查确认 sign gate、ordinary-only development 和 sealed one-shot 未被降低。

在此之前，权威状态始终是 `PROPOSED_REDLINE_NOT_ACTIVE`。
