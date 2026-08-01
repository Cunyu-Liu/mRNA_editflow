# utr_editflow_contract_v2 — 合同总览（人类可读版）

> **合同 ID：** `utr_editflow_contract_v2`（v2.0.0，FROZEN）
> **创建日期：** 2026-08-01
> **取代：** `public_intervention_contract_v1`（已归档至 `archive/legacy_predictor_first_v1/contracts_v1/`）
> **权威层级：** Goal 文档 → 本合同文件族 → per-experiment 冻结产物
> **来源文件：**
> - `configs/utr_editflow_contract_v2.yaml`
> - `docs/utr_editflow_scientific_question_v2.md`
> - `docs/utr_editflow_claim_matrix_v2.md`
> - `docs/execution/task_registry_v2.yaml`
> - `docs/decision_log.md`
> - `docs/contracts/v2_contract_conflict_matrix.md`

本文件是上述 v2 合同文件族的**整理性可读摘要**，非独立权威源。若与源文件冲突，以源文件为准（fail closed）。

---

## 一、项目定位与不可逾越边界

### 1.1 主线方法（§1.1）

**Edit Flow 是唯一主线方法，不可选、不可降级为 fallback。** 不得把 GPT 生成器改名为 Edit Flow。

研究目标是：给定既有 UTR source、region、assay/context、功能 endpoint 与目标条件，**source-conditioned、region-aware、grammar-constrained continuous-time mRNA-EditFlow** 能否学习可迁移的合法编辑轨迹分布，并生成多样、稀疏、变长且可控的 5′UTR/3′UTR 候选。

### 1.2 范围（§1.2 / §4.2）

| 类别 | 范围内 | 范围外 |
|---|---|---|
| 区域 | 5′UTR、3′UTR | CDS synonymous、full-length mRNA、cross-region full-transcript synergy |
| 实验 | 公开测量数据 + 计算证据 | 新湿实验（E6 not in scope） |
| 方法 | Edit Flow 主线 + 匹配预算基线 | RL 作为主线故事 |
| 训练 | GPU-only | CPU fallback 用于正式神经训练 |

### 1.3 关键策略约束

- **Foundation reuse-first**：frozen embedding → adapter → LoRA → partial unfreeze → full FT only if justified
- **失败证据不可删除**；**Gate 不可事后降低**
- **角色分离**：teacher/guidance、selection、final evaluator 必须分开；foundation 与 effect predictor 是支持系统，不是 Edit Flow 本身

---

## 二、科学问题与可证伪假设

### 2.1 主要科学问题（RQ-UTR-EDITFLOW-V2）

> 给定一个既有 UTR source、region、assay/context、功能 endpoint 和目标条件，source-conditioned、region-aware、grammar-constrained continuous-time mRNA-EditFlow 能否学习可迁移的合法编辑轨迹分布，并生成多样、稀疏、变长且可控的 5′UTR/3′UTR 候选？

### 2.2 比较方法问题

> 在匹配训练数据、backbone、可训练参数、GPU 预算、candidate budget、oracle-query budget 和约束条件下，mRNA-EditFlow 能否相对 AR generation、masked/discrete diffusion、generic Edit Flow、direct scorer + search 获得更好的"功能控制—合法性—多样性—编辑成本—生成效率"Pareto frontier？

### 2.3 概念边界（§2.4）

"Edit trajectory"在本项目 = **latent algorithmic edit trajectory**（模型隐变量与算法生成路径）。它**不是**：
- 生物 RNA editing 生化过程
- 进化历史
- 实验分步编辑轨迹
- 因果测量路径

若公开数据只有 source 和最终 candidate，中间路径只能称为 `latent algorithmic edit trajectory`，不可称为 `observed biological trajectory`。

### 2.4 八条可证伪假设（H1–H8）

| 假设 | 核心主张 | 必需证据 |
|---|---|---|
| **H1** Edit-process modeling | 显式连续时间 edit-rate field 优于 candidate-only、subtraction、Siamese、AR、masked/diffusion、generic Edit Flow | heldout generative likelihood、candidate recovery、calibration、multi-seed |
| **H2** Architecture irreplacability | 每个组件（source conditioning、continuous time、INS/DEL/SUB rates、STOP、variable-length、multi-step、region rate field、legal mask、edit budget、target condition）都有可测量贡献 | per-component ablation |
| **H3** Hard-constrained validity | 每一步和最终样本 100% 合法 | invalid nucleotide=0、forbidden position=0、anchor violation=0、budget violation=0、length violation=0、identity edit counted=0；**soft penalty 不可替代** |
| **H4** Conditional controllability | 改变 region/assay/endpoint/direction 产生可解释可复现的分布变化 | target direction success、condition consistency/sensitivity、permutation negative control、target strength monotonicity、identical-condition reproducibility、diversity under fixed condition |
| **H5** Generative advantage over search | 匹配预算下 Edit Flow 不被强搜索完全支配 | vs random/greedy/beam/best_of_N/simulated_annealing/scorer_exhaustive；维度：high-effect recovery、independent critic、diversity、edit cost、latency、oracle-query efficiency |
| **H6** Cross-source/study transfer | source-disjoint、gene-disjoint、study-disjoint、context-disjoint、exposure-aware 外部评估 | 不可只报 random pair split |
| **H7** Foundation-model value | foundation 提升 representation、sample efficiency 或 cross-study generalization | small from-scratch vs frozen vs adapter/LoRA vs partial/full FT |
| **H8** 5′/3′UTR unify and diff | 共享 edit process + 独立 endpoint heads/motif 规则/length priors/region adapters | **禁止**把 5′UTR MRL、3′UTR abundance、half-life 合并为统一 expression label |

---

## 三、证据等级与数据角色

### 3.1 证据等级（E0–E6）

| 等级 | 定义 | 本项目 |
|---|---|---|
| E0 | 工程（unit test、shape、smoke、synthetic sanity） | — |
| E1 | 内部计算（train/val、proxy reward、dev split） | — |
| E2 | 回顾性测量（held-out measured source/candidate labels） | ✅ 公开测量数据最高等级 |
| E3 | 跨研究/上下文（study-disjoint 或 context-disjoint measured） | ✅ |
| E4 | 历史暴露外部（独立研究但 prior label exposure 已记录） | ✅ GSE246381 固定为此级 |
| E5 | 未接触外部（genuinely unexposed and frozen before access） | ✅ |
| E6 | 前瞻性实验（新湿实验） | ❌ not in scope |

**Max expected grade: E3–E5。**

### 3.2 数据功能等级（D_A–D_E）

| 角色 | 定义 | 允许主张 |
|---|---|---|
| **D_A** | 无标签/观测 UTR 语料 — foundation representation、生成先验、去噪 | **无功能改善主张** |
| **D_B** | 绝对属性测量序列 — 条件属性表示、辅助预测器 | **无干预主张** |
| **D_C** | 源匹配测量干预 — edit-effect 和生成基础的主要监督数据 | 编辑效果、生成基础 |
| **D_D** | 密集测量景观 — 多步生成、密集预训练 | 密集预训练、多步生成 |
| **D_E** | 外部独立 — 跨研究/上下文迁移评估 | 跨研究迁移评估 |

---

## 四、当前数据状态（D0-05 后）

### 4.1 9 个数据集 v2 角色表

| # | Accession | Region | v2 Role | v2 Grade | 关键约束 |
|---|---|---|---|---|---|
| 1 | GSE114002 | 5′UTR | D_C | E2 | primary supervised |
| 2 | GSE149487 | 5′UTR | D_C | E2 | primary supervised |
| 3 | GSE217518 | 3′UTR | D_C | E2 | primary supervised |
| 4 | GSE200304 | 3′UTR | D_C | E2 | primary supervised |
| 5 | ENCSR854RUF (processed) | 3′UTR | D_C | E2 | processed table |
| 6 | GSE145046 | 5′UTR | D_D | E2 | dense landscape |
| 7 | GSE246381 | 5′UTR | D_E | **E4** | historically exposed；标签禁用于新训练/超参选择 |
| 8 | GSE207584 | CDS | D_A | E2 | **CDS out-of-scope**；仅观测预训练 |
| 9 | GSE173083 | full-length | D_A | E2 | **full-length out-of-scope**；仅观测预训练 |
| 10 | ENCSR854RUF (raw reads) | 3′UTR | D_A | E1 | raw reads 仅观测预训练 |

### 4.2 GSE246381 特殊状态（§2）

- `historically_exposed: true`
- 角色：`historically_exposed_retrospective_external_stress_test`
- **标签禁用于新训练和新超参选择**
- 证据等级固定 E4
- **禁止措辞**：`sealed`、`untouched`、`never-seen_external_test`
- 必须报告：`historical_exposure_path`

### 4.3 ENCSR854RUF 状态（§3）

- 62 个 raw fastq.gz 已下载到 `data/p0/ENCSR854RUF/reconstructed/`（~357 GB）
- 核验方法：`provider_md5 + file_size + presence_check`
- 角色：`unlabeled_or_observational_pretraining`（D_A）
- **不可提供**：wt-mutant causal labels、multi-step real trajectory、prospective improvement、final independent oracle
- **可增强**：utr_sequence_prior、foundation_adaptation、region_representation、generative_denoising

---

## 五、系统架构（§6 / §8）

```text
Layer A — Foundation representation / sequence prior
    mRNABERT / UTR-LM / 3UTRBERT / Orthrus / alternatives
    （source encoder、candidate encoder、sequence prior、initialization）

Layer B — Experimental effect system
    paired-delta model、endpoint-specific heads、uncertainty、independent critic(s)

Layer C — mRNA-EditFlow（主线）
    continuous-time legal edit rate field、source/region/target conditioning、
    variable-length multi-step sampling

Layer D — Evaluation and selection
    measured candidate recovery、independent critic、external retrospective、
    matched-budget baselines、calibration、failure analysis
```

**规则**：Foundation 与 effect predictor 是支持系统；Edit Flow rate heads + 合法 state/action space 完成生成。**不得把 GPT 生成器改名为 Edit Flow。**

### 5.1 Edit Flow 架构要求（§11）

| 维度 | 要求 |
|---|---|
| continuous_time | 非负事件率 λ_ins/λ_sub/λ_del/λ_stop + token/action 分布；训练和采样都保留 t |
| source_conditioning | p(candidate, trajectory \| source, region, context, target, constraints)；报告 source preservation rate |
| variable_length | INS/DEL 是一等公民（训练、采样、评估），非后处理 |
| multi_step | 支持 edit budget k ∈ {1, 3, 5}；评估 order、STOP calibration、cycling、reverse、repeated position、budget utilization、per-step legality、real edit distance |
| hard_action_mask | region boundary、protected positions、anchor motifs、allowed alphabet、max/min length、edit budget、forbidden identity、optional motif-preservation、source-relative state tracking；**在 rate normalization 之前 act** |
| region_aware | shared trunk + 5′UTR adapter + 3′UTR adapter；对比 fully-shared vs adapter vs independent vs wrong-region negative control |
| conditional_generation | endpoint、assay、context、target direction、target quantile、max edit budget、length target、must-preserve motif；报告 train vs sample condition consistency |
| diversity | 多 candidate per source/condition；报告 unique rate、pairwise edit distance、motif/structure diversity、mode collapse、duplicate rate、candidates/sec、amortized cost |

---

## 六、主张边界（Claim Matrix）

### 6.1 允许的主要主张（Gate 通过后）

> We introduce a source-conditioned, region-aware and grammar-constrained continuous-time Edit Flow that generates diverse and biologically legal minimal edits for 5′ and 3′ UTRs, and evaluate its controllability and transfer under matched-budget generative and search baselines.

### 6.2 条件性次要主张（S1–S8，需证据支持）

| ID | Claim | Required evidence |
|---|---|---|
| S1 | continuous-time edit-process modeling 优于 candidate-independent modeling | heldout generative likelihood、candidate recovery、calibration、multi-seed |
| S2 | explicit legal action geometry 实现 100% constraint validity | 每步 + 最终样本的 hard-constraint audit |
| S3 | variable-length flow 改善 infilling/refinement | variable-length infilling/refinement benchmark |
| S4 | Edit Flow 相比强搜索改善 candidate/query efficiency | matched-budget Pareto frontier comparison |
| S5 | foundation-model initialization 改善 sample efficiency 或 transfer | from-scratch vs frozen vs adapter vs full-FT |
| S6 | region-conditioned adapters 优于 fully-shared 或 fully-independent | region adapter ablation + wrong-region negative control |
| S7 | uncertainty/abstention 减少 false-beneficial selection | ECE、coverage-risk curves、selective prediction |
| S8 | public UTR intervention data 支持生成式编辑评估 | benchmark construction + provenance audit |

### 6.3 永久禁止的主张

1. 生成 candidate 改善真实 therapeutic mRNA efficacy
2. 未测量 candidate 有实验验证的功能改善
3. MRL = protein output；TE = protein output；half-life 改善必然改善 protein output
4. predictor 高分 = 真实生物最优
5. model trajectory = observed biological trajectory
6. **GSE246381 是 untouched sealed test**
7. full-length mRNA 优化完成；CDS grammar 已验证
8. 仅凭 attention heatmap 得机制结论
9. 同一 predictor 自导、自选、自证
10. Edit Flow 无证据下天然优于 GPT/diffusion/search
11. "the first Edit Flow for biological sequences" / "source-conditioned" / "constrained" / "variable-length"（除非附搜索日期、数据库、查询、排除标准、与最近 prior art 的逐字段差异表）

**论文措辞偏好**：`we formulate / we develop / we evaluate`，而非 `the first`。

---

## 七、Phase 结构与强制 Gate（§9）

```text
C0  合同与现实对齐                          [DONE]
D0  科学问题驱动的数据发现                  [D0-01..04 DONE, D0-05 PENDING→DONE]
D1  数据资格、重建与暴露审计                [PENDING ← 当前执行]
B0  生成式 UTR benchmark 与 splits          [PENDING]
FM0 Foundation model 接入                   [PENDING]  (FM0 → MK0 → EF0 mandatory gate)
MK0 UTR Edit Flow 数学内核冻结              [PENDING]  (gate)
EF0 True UTR Edit Flow 工程实现             [PENDING]
GP0 Generative prior GPU 训练               [PENDING]  (GPU-only)
FC0 Functional conditioning / critic 系统   [PENDING]
ME0 Measured-support 与 candidate freeze    [PENDING]  (ME0 → MB0-Freeze → MB0-Run)
MB0 Matched-budget 正式比较                 [PENDING]  (gate)
TR0 5′UTR → 3′UTR 迁移                       [PENDING]
ER0 Robustness、failure 与机制分析          [PENDING]
PP0 论文、复现与发布                        [PENDING]
FL0 未来 full-length 决策 (not in current scope)
```

### 7.1 强制 Gate

- **FM0 → MK0 → EF0**：数学内核未冻结前，不得启动 EF0 正式实现验收或 GP0 正式训练
- **ME0 → MB0-Freeze → MB0-Run**：MB0-Freeze 未通过时不得查看 final labels 后再选择/删除/改名 baseline

### 7.2 执行规则

Forward-only state machine。上游 Gate 未通过 → 可继续非冲突的并行准备，但不得下下游正式科学结论。

### 7.3 完整任务表（task_registry_v2.yaml 摘要）

| Task | Phase | Status | Dependencies | Acceptance |
|---|---|---|---|---|
| C0-01..05 | C0 | DONE | — | preflight / conflict matrix / v2 文件族 / audit+tests / README |
| D0-01..04 | D0 | DONE | — | hypothesis-data matrix / dataset capability / download / missing acquisition |
| D0-05 | D0 | PENDING→DONE | D0-02 | per-dataset role table；no auto-promote by size |
| **D1-01** | D1 | **PENDING** | D0-05 | apply(edit_script, source)==candidate 100%；path ambiguity quantified |
| **D1-02** | D1 | **PENDING** | D1-01 | exposure ledger coverage=100% |
| B0-01..05 | B0 | PENDING | D1-02 | schemas / splits / leakage audit / eval tracks / data card |
| FM0-01 | FM0 | PENDING | B0-05 | foundation checkpoint + tokenizer + hash/license + GPU loader + exposure audit |
| MK0-01 | MK0 | PENDING | FM0-01 | math kernel frozen + numerical verification |
| EF0-01 | EF0 | PENDING | MK0-01 | H3 hard-constrained validity 100%；not just greedy/top-k |
| GP0-01 | GP0 | PENDING | EF0-01 | training manifest complete；GPU-only |
| FC0-01 | FC0 | PENDING | GP0-01 | role separation audited |
| ME0-01 | ME0 | PENDING | FC0-01 | candidate freeze complete；no final label leakage |
| MB0-Freeze | MB0 | PENDING | ME0-01 | MB0-Freeze Gate passed before viewing final labels |
| MB0-Run | MB0 | PENDING | MB0-Freeze | Pareto frontier comparison complete；H5 evaluated |
| TR0-01 | TR0 | PENDING | MB0-Run | H8 + H6 cross-study transfer evaluated |
| ER0-01 | ER0 | PENDING | TR0-01 | failure card complete；H4 conditional controllability evaluated |
| PP0-01 | PP0 | PENDING | ER0-01 | submission readiness (§39.5) met；all main figures reconstructable |

---

## 八、Evaluation Tracks（§10 / B0-04）

| Track | 定义 |
|---|---|
| closed_measured_pool | measured source-candidate pairs，closed support |
| heldout_generative | held-out source→candidate generative likelihood/recovery |
| open_legal_generation | open-support legal generation under constraints |

---

## 九、Prior Art（§14，audited 2026-07-30）

Directed audit **未发现**满足以下全部条件的方法：5′UTR+3′UTR + source-conditioned + continuous-time Edit Flow + INS/SUB/DEL+STOP + stepwise UTR grammar hard mask + region-aware trunk/adapters + matched-budget evaluation。

直接 precedents：

- Edit Flows (arxiv 2506.09018) — variable-length INS/DEL/SUB Edit Flow
- EvoFlows (arxiv 2603.11703) — template protein INS/DEL/SUB editing
- Flexible Flows for Biological Sequence Design (arxiv 2606.10543)
- pCoMole (OpenReview tTILzscPs4) — pretrained Edit Flow + hard terminal feasibility
- SPROUT (OpenReview 4AF7WSp7Cs) — rollout utility guided promoter Edit Flow

---

## 十、训练 Manifest 必填字段（§15）

Paper-mode runs 必须记录：

- `goal_contract.id` + `sha256`
- `scientific_question_id`: RQ-UTR-EDITFLOW-V2
- `phase_id`, `task_id`, `git_commit`
- `data_manifest_sha256`, `split_manifest_sha256`
- `foundation_checkpoint` + `sha256`（或 `none`）
- `exposure_ledger_version`

**缺失这些字段 → 仅 development smoke，不是 paper evidence。**

---

## 十一、Amendment 历史

| ID | 日期 | 摘要 |
|---|---|---|
| `utr_editflow_goal_v2.1_additive_math_mb0` | 2026-07-30 | 添加 math kernel、architecture diagram、MB0 baseline contract；添加 FM0→MK0→EF0 mandatory gate |
| `utr_editflow_goal_v2.2_b0_capacity_nonblocking` | 2026-07-31 | B0 capacity diagnostics 降为 historical optional engineering diagnostics，非 B0 acceptance gate（DEC-UTR-EF-V2-20260731-B0-CAPACITY-NONBLOCKING） |
| `utr_editflow_goal_v2.2_b0_frozen_d1_replay_scope` | 2026-07-31 | B0 path-state scope = frozen D1 canonical edit_script prefixes + declared intermediates；capacity gate removed，zero-leakage gate retained（DEC-UTR-EF-V2-20260731-B0-FROZEN-REPLAY-SCOPE） |

### 11.1 Decision Log 关键条目

| Decision ID | 摘要 |
|---|---|
| `DEC-UTR-EF-V2-20260801-CONTRACT-INTEGRATION` | 三层合同整合为单一活动合同（Phase C0） |
| `DEC-UTR-EF-V2-20260801-ENCODE-STATUS-CORRECTION` | 修正 ENCSR854RUF 62 文件下载状态（deferred→COMPLETE） |
| `DEC-UTR-EF-V2-20260801-LEGACY-ARCHIVAL` | predictor-first / SparseEditForm / RL artifacts 归档（不融入主线） |

---

## 十二、v1 → v2 冲突矩阵（摘要）

完整矩阵见 `docs/contracts/v2_contract_conflict_matrix.md`（FROZEN 2026-08-01）。

| 维度 | v1 | v2 | 处置 |
|---|---|---|---|
| 主线方法 | SparseEditForm predictor；Flow optional | Edit Flow primary，NOT optional | v1 archived |
| 范围 | 5′+3′+CDS+full-length | 5′+3′ only | CDS/full-length archived |
| GSE246381 | sealed external test | historically exposed (E4) | sealed wording removed |
| 证据等级 | A1/A2/B1/B2/C/D | E0–E6 | v1 grade scheme archived |
| Phase 结构 | P3-00A | C0→…→PP0 + mandatory gates | v1 registry archived |
| RL 角色 | GRPO/DAgger 主线 | RL 非主线 | `rl/` archived |
| ENCODE 状态 | deferred/PARTIAL | COMPLETE | 4 evidence files updated |

**最终状态**：active contract count = 1；active contract conflicts = 0；SUPERSEDED chain: P3/NMI → v1 → **v2 (ACTIVE)**。

---

## 十三、Full-length 未来 Gate（§17）

扩展到 full-length mRNA 需**全部**满足：

1. 5′UTR 和 3′UTR true Edit Flow semantics 通过
2. INS/DEL/SUB/STOP 和 variable length 都有非-smoke 证据
3. H1–H6 无 Critical blocker
4. 生成模型不被 baseline 完全支配
5. Data、foundation exposure、final evaluator 角色 auditable
6. 至少一个 region cross-study generational result 有统计支持
7. 用户明确批准扩展

---

## 十四、治理（Governance）

### 14.1 禁止做法

```text
silently rewriting the primary task
changing thresholds after seeing results
moving failed families out of the test set
dropping failed seeds
weakening strong baselines to preserve a story
claiming measured improvement from an internal predictor
overwriting frozen artifacts without an amendment
renaming a GPT generator as Edit Flow
lowering a gate post-hoc
deleting failure evidence
```

### 14.2 变更规则

活动合同为 `utr_editflow_contract_v2`。变更须遵循合同 amendment 规则并记录于 `docs/decision_log.md`。

### 14.3 项目路线选择

**由证据选择，而非预先固定为 RL、full-transcript 或 synergy 论文。**
