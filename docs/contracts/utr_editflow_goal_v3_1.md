# mRNA-EditFlow UTR Benchmark-First 科研与执行合同

## provenance-controlled source-conditioned UTR editing benchmark、Edit-Flows-derived UTR reference method 与 alignment-robustness study

> - **文档性质：**科学问题合同、数据资源合同、benchmark 合同、模型与评测合同、GPU 执行合同、论文主张边界
> - **权威合同 ID：**`utr_editflow_goal_v3.1_benchmark_first`
> - **文档状态：**`AUTHORITATIVE_REVISED_CONTRACT_APPROVED_NOT_ACTIVATED`；用户已确认 GSE246381 项目侧绝对未暴露、保留 Edit-Flows-derived 方法核心、仅挖掘 E/F 数据并接受其余讨论结论；本合同本身仍不授权任何远端写入、下载、commit 或训练，只有用户后续明确发出“执行 GOAL-V3-DATA-BENCH-01”后才可激活，远端 C3 落盘、supersession 与 hash binding 仍待执行
> - **制定日期：**2026-08-03
> - **目标仓库：**`/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
> - **当前范围：**5′UTR 为主区域；3′UTR 为次区域、迁移与机制扩展；CDS 与全长 mRNA 不属于首篇论文主任务
> - **湿实验边界：**当前没有新增湿实验资源；所有结论限于公开历史 assay 的回顾性计算证据
> - **论文定位：**benchmark/resource-first，Edit Flow 为需要通过强基线检验的方法共同贡献
> - **方法核心：**保留从 Edit Flows 演化而来的 conditional action-rate / alignment-aware auxiliary-process 技术路线；把 endpoint pair 的 alignment、编辑顺序、clock 与 detour 分开审计；不把单条 canonical edit script 当成真实生物轨迹，也不把原始 Edit Flows 已有思想冒充本项目首创
> - **数据策略：**只以 Track E（显式编辑关系、dense landscape、no-edit control）和 Track F（有功能标签但无可靠配对）作为科学数据；不建设 Track U，不用无标签 UTR 序列做项目预训练；数据质量、数据量和有效独立样本量同等重要
> - **评测策略：**5′/3′非对称任务；实验伪影风险分层；source/study/context/family 级拆分；匹配查询与编辑预算
> - **开发策略：**三轮预注册开发，final labels 封存且只打开一次；所有尝试留痕
> - **数据快照边界：**v1 文献与公开资产发现截至 2026-08-03；新增来源进入 v1.1，不回写 v1 final
> - **GSE246381 真相锁：**项目负责人已明确确认该数据从未用于模型训练、调参、错误分析或人工标签检查；此前文档、记忆或代码中的 `E4/HISTORICALLY_EXPOSED` 解释全部作废；pipeline 已物化序列/聚合标签这一事实单独登记
> - **历史合同：**原 v2 文档 SHA256 为 `3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791`，必须保留为只读历史，不得删除或伪装成已满足 v3.1

---

# 0. 文档权威性、术语与变更控制

## 0.1 本合同解决的唯一问题

本项目不再以“是否能把一个通用 Edit Flow 接到 UTR 数据上”为论文问题。v3.1 的核心问题是：

> 能否把异构、强实验库偏差、不同 sequence scope、不同端区和不同 assay context 的公开 UTR 功能数据，构造成一个来源可追溯、角色可解释、任务真实、拆分严格的 source-conditioned editing benchmark；并在 endpoint-only、真实路径未观测的条件下，检验从 Edit Flows 演化而来的 alignment-aware UTR reference method 在各自同任务、匹配预算下何时优于、何时不优于简单而强的生成或预测/搜索基线？

项目必须同时回答三类问题，且不得相互替代：

1. **数据问题：**哪些记录是真正的 source→candidate/ref→alt 功能证据，哪些只是无母本功能序列、随机库、重复测量或仅供参考注释的无标签序列？
2. **科学问题：**5′与3′在何种任务、编辑距离、实验 context 和迁移方向上共享规律，何时不能共享？
3. **方法问题：**在明确继承原始 Edit Flows alignment/auxiliary-process 思想的前提下，本项目的 UTR source conditioning、区域适配、约束动作、路径 proposal/ensemble 与 benchmark 组合是否带来稳定、可校准、跨 study 的增量收益？

如果数据合同不能闭环，模型训练成功不能补偿数据失败；如果方法不能胜过强基线，benchmark/resource 仍可成立，但不得强行宣称方法优越。

## 0.2 权威顺序

执行层必须建立并遵守以下权威顺序：

1. 本 Goal 文档及其 SHA256；
2. `configs/utr_editflow_contract_v3_1.yaml`、decision log 与 claim matrix；
3. v3.1 JSON Schemas 与 dataset-specific adapter contracts；
4. 本合同冻结的 REQUIRED task/split/diagnostic expected sets、immutable `TaskRegistry`/`SplitRegistry` definitions、`task_split_contract_matrix_v3_1.yaml` 与 DiagnosticRegistry；
5. immutable raw/acquisition/rights manifests 与 source-level evidence；
6. immutable technical canonical base tables、observation/relation lifecycle、transformation edges 与 canonical manifest；
7. record-level immutable baseline `EXPOSURE_RECORDS.jsonl`、accepted-E-pair base `USE_ROLES.jsonl` 与 current-canonical projection（sealed objects 使用 §3.2 restricted shard）；
8. append-only ordinary/restricted access logs、逐 checkpoint `foundation_exposure_ledger.jsonl` 与由其确定性生成的 `EFFECTIVE_EXPOSURE_PROJECTION.jsonl`；
9. B0 immutable pre-role facts/`B0_ROLE_DECISION_EVIDENCE.jsonl`；
10. `B0_TRANSACTION_COMMITS.jsonl` 中已 committed transaction 所引用的 append-only `RELATION_ROLE_TRANSITIONS.jsonl`；未 committed events 只属失败 evidence；
11. 由第 6/7/10 项确定性生成的 `EFFECTIVE_ROLE_PROJECTION.jsonl`；
12. `GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl` 及一对一派生的 global `ELIGIBILITY_MANIFEST.jsonl`；
13. outcome-blind activation-calibration mask、B0 `TASK_ACTIVATION_DECISIONS.jsonl`、`SPLIT_ACTIVATION_DECISIONS.jsonl` 与全局 120-row `TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl`；
14. `TASK_ELIGIBILITY_UNIVERSE.jsonl`；
15. `SPLIT_ASSIGNMENTS.jsonl`、sealed commitments、split/evaluator manifests；
16. 单次 run 的 frozen config、代码 commit、status、output manifest、checksums、finalizer 与 DONE；
17. 论文表格、图和文字。

低层文件与高层文件冲突时必须 `FAIL_CLOSED`。派生层绝不能反压或改写权威上游：baseline ExposureRecord/UseRole 不能修改 canonical，access/foundation projection 不能重写 baseline 或 access chain，predecision/transaction 不能修改 source facts，role projection 不能修改 committed transition ledger，eligibility 不能修改 canonical/base role/exposure，activation/applicability decision 不能修改 definition registries，split assignment 不能修改 registries/eligibility，论文不能修改任何 evidence。不得由 parser、trainer、evaluator 或论文作者选择更方便的解释。

## 0.3 强制术语

- `PASS`：当前冻结输入、代码和配置的完整 closure bundle 满足全部适用门槛。
- `REOPENED`：历史证据保留，但不再授权下游阶段或当前论文主张。
- `BLOCKED`：存在 P0 blocker；不得靠改阈值、删困难样本或把 `UNKNOWN/N/A/PENDING` 记为 PASS 解锁。
- `DEVELOPMENT_ONLY`：仅供调试、描述统计和方案选择；不能成为 final benchmark 数字。
- `FORMAL`：绑定干净 commit、冻结数据、冻结配置和完整 artifacts；神经网络训练/推理适用真实 CUDA，CPU-native 数据审计、统计与基线允许在 CPU formal 执行。`FORMAL` 不等于一律 GPU。
- `INDEPENDENT_PARENT`：按生物母本、variant context、gene/tile family 或实验库设计定义的独立单位，不等于 JSONL 行。
- `CONTRACT_VALID`：满足语法、长度、预算、保护位点和 action mask；不等于 biologically functional、safe 或 therapeutic。
- `LATENT_PATH`：从 source 到 candidate 的一种可执行编辑脚本；不是被实验观测的生物轨迹。
- `PIPELINE_MATERIALIZATION`：序列/标签被 parser、builder、canonical 或 split 工具机械处理和落盘；必须登记，但不自动等于用于模型选择或人工看标签。
- `ANALYTIC_LABEL_ACCESS`：标签被训练、调参、模型选择、错误分析、候选选择、人工查看或正式 evaluator 消费；这才改变 project/final access 状态。
- `ANALYTIC_FINAL_LABELS_ACCESSED=false`：仅表示 human-label/train/tune/model-selection/pre-final-error-analysis/one-time-final-evaluator 访问计数均为 0；human-sequence view 另有强制 zero counter；隔离的机器 aggregate QC 与历史 pipeline materialization 必须用独立计数/状态登记，不得用该布尔值隐瞒。

## 0.4 每个执行者的读取与预检义务

开始数据、模型、训练、评测或论文任务前必须：

1. 完整读取本合同并记录合同 ID 与 SHA256；
2. 指明 Phase、Task ID、依赖 Gate、输入与禁止事项；
3. 只读检查仓库 HEAD、工作区、活跃进程/GPU、磁盘、data 与 artifacts；
4. 保护用户现有修改与无关进程；不得 reset、覆盖或杀无关任务；
5. 使用隔离 worktree/run root；
6. 将开发证据、工程验收、正式 scientific gate 和论文 claim 分开；
7. 任一依赖未知或冲突时停止，不得自行降级。

所有 formal manifest 至少包含：

```yaml
goal_contract:
  id: utr_editflow_goal_v3.1_benchmark_first
  sha256: <actual_sha256>
literature_snapshot: 2026-08-03
phase_id: <phase>
task_id: <task>
git_commit: <full_sha>
worktree_dirty: false
artifacts:
  raw_manifest:
    sha256: <sha256-or-null>
    availability: <AVAILABLE_INPUT|NOT_YET_PRODUCED|NOT_APPLICABLE>
    artifact_role: <UPSTREAM_INPUT|CURRENT_PHASE_OUTPUT|NOT_APPLICABLE>
  canonical_manifest:
    sha256: <sha256-or-null>
    availability: <AVAILABLE_INPUT|NOT_YET_PRODUCED|NOT_APPLICABLE>
    artifact_role: <UPSTREAM_INPUT|CURRENT_PHASE_OUTPUT|NOT_APPLICABLE>
  exposure_ledger:
    sha256: <sha256-or-null>
    availability: <AVAILABLE_INPUT|NOT_YET_PRODUCED|NOT_APPLICABLE>
    artifact_role: <UPSTREAM_INPUT|CURRENT_PHASE_OUTPUT|NOT_APPLICABLE>
  split_manifest:
    sha256: <sha256-or-null>
    availability: <AVAILABLE_INPUT|NOT_YET_PRODUCED|NOT_APPLICABLE>
    artifact_role: <UPSTREAM_INPUT|CURRENT_PHASE_OUTPUT|NOT_APPLICABLE>
  foundation_checkpoint:
    sha256: <sha256-or-null>
    availability: <AVAILABLE_INPUT|NOT_YET_PRODUCED|NOT_APPLICABLE>
    artifact_role: <UPSTREAM_INPUT|CURRENT_PHASE_OUTPUT|NOT_APPLICABLE>
config_sha256: <sha256>
runtime_environment_sha256: <sha256>
analytic_final_labels_accessed: <false|true|UNKNOWN_AFTER_FAILED_INTENT>
legacy_pipeline_sequence_materialization: <ABSENT|PRESENT|NOT_APPLICABLE>
legacy_pipeline_label_materialization: <ABSENT|PRESENT|NOT_APPLICABLE>
v3_restricted_builder_machine_access_count: <integer>
v3_aggregate_qc_machine_access_count: <integer>
v3_fm_overlap_machine_access_count: <integer>
v3_b0_eligibility_split_machine_access_count: <integer>
v3_g7_restricted_finalizer_machine_access_count: <integer>
v3_human_sequence_view_count: <integer>
v3_human_label_view_count: <integer>
v3_train_access_count: <integer>
v3_tuning_access_count: <integer>
v3_model_selection_access_count: <integer>
v3_internal_test_evaluator_attempt_count: <integer>
v3_internal_test_evaluator_completion_count: <integer>
v3_pre_final_error_analysis_count: <integer>
v3_one_time_final_attempt_count: <integer>
v3_one_time_final_evaluator_count: <integer>
final_access_status: <SEALED_UNOPENED|FINAL_ACCESS_RESERVED|FINAL_OPENED|FINAL_INVALIDATED>
final_access_log_sha256: <sha256-or-not-applicable>
```

G0 只对启动本阶段时真实存在的 `AVAILABLE_INPUT` 计算并校验 hash；`NOT_YET_PRODUCED` 既不是输入，也不能作为 PASS evidence。当前阶段产出的 canonical、ledger、split、report、STATUS 与 DONE 只能在 G7 `OUTPUT_MANIFEST.json` 中绑定；禁止复用 stale artifact 填充模板，禁止为了满足 schema 预先制造空文件。

access 状态的 phase invariants：C3、D0-R、D1-R、FM0-A、B0-R 与数据 G7 均要求 `analytic_final_labels_accessed=false`、全部 human/train/tune/model-selection/internal-test/pre-final-error-analysis/final-attempt/final-evaluator counters=0、`final_access_status=SEALED_UNOPENED`；五类白名单 machine counters 可为有日志的非负整数。后续 GP0/FC0 对 GSE246381 仍须保持上述 analytic counters=0；ordinary INTERNAL_TEST 另依 §8.4。只有另行授权且模型/config/container/budget/output schema 全冻结后的 MB0，才可原子追加一次 final INTENT：INTENT 立即使 `v3_one_time_final_attempt_count=1`、状态=`FINAL_ACCESS_RESERVED`，且 exposure 对 requested scope保守变为 PRESENT/UNKNOWN；成功 completion 才写 `analytic_final_labels_accessed=true`、`v3_one_time_final_evaluator_count=1`、`final_access_status=FINAL_OPENED`。abort/crash/orphan intent 写 `analytic_final_labels_accessed=true|UNKNOWN_AFTER_FAILED_INTENT` 与 `FINAL_INVALIDATED`，永不恢复 unopened；attempt count>1、evaluator count>1、非法状态跃迁或日志/hash 不一致同样 invalidated。

## 0.5 合同变更与决策日志

以下内容只有用户明确讨论确认后才能改：

- benchmark-first 的论文主线；
- 5′主区域、3′次区域的非对称任务；
- 无新增湿实验边界；
- GSE246381 的“项目侧从未发生 analytic use/exposure”事实、pipeline materialization 及其后续 frozen role；
- Track E/Track F 双轨、拒绝项目无标签预训练与风险分层；
- Edit-Flows-derived conditional action-rate/alignment-aware auxiliary-process 方法核心；
- final-label 一次性访问；
- primary metrics、matched budget 与 stop rules；
- 允许和禁止的论文主张。

任何变更必须新增机器可读 decision log：

```yaml
decision_id: <stable-id>
date: <iso-date>
old_contract_sha256: <sha256>
old_text_or_rule: <rule-id>
new_text_or_rule: <rule-id>
reason: <scientific-or-audit-reason>
evidence_ids: [<id>]
affected_artifacts: [<artifact>]
requires_rerun: [<gate>]
approval_status: <PENDING_USER_ACCEPTANCE|APPROVED_BY_USER>
approved_by_user: <true-only-when-explicitly-approved; omit-while-pending>
```

不得改写旧 artifacts 使其看起来天然符合新合同。旧失败、旧 PASS 和旧报告全部保留，并明确标记当前效力。

---

# 1. 2026-08-03 冻结决策与当前真相锁

## 1.1 用户已确认的十二项决策

| 决策 | 冻结结果 |
|---|---|
| 论文主线 | `BENCHMARK_FIRST`：可审计 benchmark/resource 为主，方法为共同贡献 |
| 湿实验 | 当前明确无新增湿实验；不等待湿实验才推进计算论文 |
| 区域 | 5′为 confirmatory primary；3′为 secondary/transfer/mechanism，不强行对称 |
| 数据 | `E_F_PRIORITY_ONLY`：Track E 为主、Track F 为辅；取消 Track U 和项目无标签预训练 |
| 伪影 | `RISK_STRATIFIED_PRIMARY`：低风险主分析，高风险敏感性/辅助训练 |
| 阶段 | 重开 D0/D1/B0；保留 MK0/EF0 仅限工程证据；正式 GP0 阻断 |
| 方法 | 保留从 Edit Flows 演化来的 conditional action-rate、auxiliary alignment 与 multi-sample estimator 路线，但明确继承关系，不把通用 path/alignment marginalization 本身写成首创 |
| 开发 | 三轮预注册开发；所有 trial 登记；sealed final 只开一次 |
| 数据发布 | versioned benchmark v1；发现快照截至 2026-08-03，后续进入 v1.1 |
| GSE246381 | `PRIOR_ANALYTIC_USE_NONE_CONFIRMED_BY_OWNER`，同时 `PIPELINE_MATERIALIZATION_PRESENT`；任何 E4/historically exposed 解释永久撤销 |
| GSE246381 用途 | 先保留为 sealed external final 候选；D1 起只允许有白名单脚本、hash 与 append-only 日志的 restricted builder/aggregate-QC machine access；正式一次性 final 前禁止 human label view、train/tune/model selection、pre-final error analysis、ordinary loader/report 与 final evaluator access，也不得自动分配 train/dev |
| 合同解释 | 合同是可证伪执行规则，不是事实来源；数据、实现或文献证据与合同冲突时必须修合同并留 decision log |

## 1.2 当前远端审计快照

以下数字仅绑定 2026-08-03 的只读审计，不是 v3 closure：

| 项目 | 当前值 | 合同解释 |
|---|---:|---|
| 主仓库 HEAD | `aca6c8c31f5843105f49c30e78e78342278bdd7d` | 数据修复分支快照 |
| 主仓库 branch | `phase2-reliable-local-delta-20260727` | 工作区仍有未跟踪/legacy 项，formal 必须隔离 |
| canonical rows / SHA256 | 1,151,125 / `2780d5a74a6fdec76b79706011e1b32eabd9d2c112741566870fc6d1add55470` | 不能称为 paired trajectories |
| ledger rows / SHA256 | 1,151,125 / `cdfc55c240db5eb110e49cc1edb36fb2f411f753ff4fd0fec98a19b896a589bf` | record-ID 机械覆盖一致，不证明角色正确 |
| explicit observational / explicit paired | 1,061,899 / 55,184 | 仅为 legacy `record_type` 口径 |
| endpoint-present pair-shaped rows | 89,226 | 等于 55,184 explicit paired + 34,042 missing-record-type/fallback-required；不是 89,226 条都由 fallback 推断 |
| B0 覆盖 paired rows | 85,214 | 无声遗漏 4,012 |
| B0 eval assignments | 250,161 | assignments 不是 unique records，不得当数据规模 |
| 缺 `metadata.record_type` | 34,042 | schema FAIL |
| GSE145046 rows | 1,048,106 | 10-nt input abundance，不是百万 paired edit |
| GSE207584 两端 sequence 均空/未保留 | 10,227 | CDS legacy labels-without-sequence liability，schema FAIL |
| GSE217518 当前 canonical | 3,564 | 只覆盖当前 workbook 行的 70.3%，且未进入 B0 |
| region counts | 5′ 1,106,792；3′ 30,843；CDS 10,227；raw literal `full_length` 3,263（规范化概念 full-length） | CDS/full-length 必须移出 UTR 主轨 |

当前 builder/canonical 至少排除了 GSE114002 的 44,833 个和 GSE246381 的 116 个 identity/WT records；这避免了把 identity 计为真实编辑 pair，但 identity 测量本身不得丢失，应从 raw/paper_clean 恢复到 `no_edit_control` 与 source-measurement 层。不得把这一事实写成“某个修复 commit 删除了 44,833 条”。

当前 paired 结构进一步说明 5′/3′任务不能强行对称：

| 当前开发快照 | 5′UTR | 3′UTR |
|---|---:|---:|
| current pair-shaped records | 58,383 | 30,843 |
| unique source sequences | 33,998 | 30,639 |
| source 至少 2 个 candidates | 13,402 | 151 |
| source 至少 5 个 candidates | 1,163 | 7 |
| 位于 multi-candidate source 的 records | 37,787 | 355 |

当前 B0 的 85,214 records 中约 76% 为 edit distance 1；多步、indel 和多候选证据集中于少数 library。论文必须报告这种 concentration，不能把总 action count 当作跨研究的一般能力。

修复后的 `GSM3130435_egfp_unmod_1.csv.gz` 有 326,033 个 data rows 并通过 gzip/CRC；但当前 builder 只消费 `GSM3130443_designed_library.csv.gz`，当前 canonical provenance 也未绑定 `GSM3130435`。因此“原始文件修好”不等于“当前 canonical 已消费修复”；不得用 mtime 推断消费关系。

补充 raw 审计：GSE114002 的 10 个 gzip 共 2,753,885 rows、1,568,946 跨文件 unique sequences，当前 canonical 只读取 `GSM3130443_designed_library.csv.gz`；GSE145046 的 30 个 gzip 共 30,695,604 rows，valid 10-mer union=1,048,576（完整 `4^10` 空间），当前只读取一个 rep2 input 文件。两者说明 raw coverage 与待恢复信息量很大，但只有完成确定性 join 后的 label-complete sequence×context observations 才能计入功能数据；condition/replicate/input-support rows 既不是 F example，也不是 edit pair。

当前 B0 另有 570 条无可用功能 label、11,701 条 edit distance>5；`study_disjoint` 与 cross-region 仍混杂，gene/context/barcode/foundation 仍存在 `N/A/PENDING+PASS`。这些数字使旧 B0 不能作为 v3.1 benchmark closure。

模型重绑定也尚未完成：当前 GP0 `common.py` 与 `record_gp0_preflight.py` 仍硬编码旧 paired count `134059`/11 accessions，`train_gp0.py` 默认 `max_length=256`；而当前 B0 unique pair-shaped coverage 口径为 85,214，v3 数据又尚未冻结。因此数据 Goal 无权直接解锁 GP0。

## 1.3 阶段状态真相锁

| 阶段 | v3 当前状态 | 允许主张 |
|---|---|---|
| D0 | `REOPENED_P0_BLOCKED` | 公开资产 inventory/recovery 进行中 |
| D1 | `REOPENED_P0_BLOCKED` | 当前 canonical 仅 `DEVELOPMENT_ONLY_UNSEALED` |
| B0 | `REOPENED_P0_BLOCKED` | 当前 split/report 不是正式 benchmark |
| FM0 | `REBIND_REQUIRED` | 历史接入工作保留；必须对 v3 数据重做 exposure |
| MK0 | `ENGINEERING_ACCEPTED_REBIND_REQUIRED` | 仅 E0 math/engineering；不证明训练或科学有效 |
| EF0 | `ENGINEERING_VERIFIED_REBIND_REQUIRED` | 仅 E0 engineering；不补偿数据缺陷 |
| GP0 | `LOCKED_NOT_AUTHORIZED` | 当前只有代码开发痕迹，无 formal scientific run |
| PAPER_CLAIM | `NOT_AUTHORIZED` | 不得声称完整清洗、leakage-free、方法优越或可发表完成 |

历史 MK0/EF0 artifacts 不删除、不伪装重跑。v3 只要求在新数据、合同与统一 commit 上重新绑定并做相称回归测试。

## 1.4 自动重开规则

满足任一条件时自动重开并级联：

1. raw/source 文件新增、修复、替换、hash 改变：重开 D0→D1→B0；
2. parser、normalizer、join、dedup、schema、role 或 canonical builder 改变：重开 D1→B0；
3. exposure、split、grouping、budget、leakage、FM reference 或任务定义改变：重开 B0；
4. contract/canonical/ledger/split/hash 不一致：对应 gate 失效；
5. 报告早于输入或未绑定完整输入 hash：报告 `STALE_INVALIDATED`；
6. 上游阶段重开：所有下游授权自动撤销。

---

# 2. 论文定位、创新边界与可证伪问题

## 2.1 工作论文定位

工作定位为：

> **A provenance-controlled benchmark for source-conditioned 5′UTR editing, with a qualified 3′UTR extension and an Edit-Flows-derived continuous-time reference model.**

中文定位为：

> 一个以实验语义、来源追踪、暴露控制、source/study/family 拆分和匹配预算为核心的 source-conditioned UTR editing benchmark；5′UTR 为主要确认区域，3′UTR 为通过独立资格门槛后保留的扩展区域，并提供一个明确继承自 Edit Flows 的连续时间参考模型。

首篇论文的贡献优先级固定为：

1. 面向 source-conditioned UTR editing 的数据语义、provenance、许可、attrition 和可复现构建；
2. source/study/family 级拆分、模型/标签暴露审计与 reporter-artifact 风险控制；
3. 适应 5′与3′真实数据结构的非对称任务和 matched-budget evaluation；
4. 一个明确继承自 Edit Flows 的 UTR reference model：加入 persistent source/current conditioning、edit budget、contract-valid mask 与 region/context interface；
5. 对 canonical alignment、native/fresh alignment proposal 和 multi-alignment training 的预注册鲁棒性比较；
6. 至少一条独立于拟议方法胜负的可复用经验发现，例如 alignment sensitivity、library shortcut、跨区域负迁移或 source conditioning 的适用边界。

## 2.2 方法继承、项目新增与禁止主张

本项目建立在已发表的 Edit Flows 上。以下内容明确属于上游方法，不作为本项目原创：

- 在可变长 sequence space 上定义 continuous-time Markov chain；
- 使用 substitution、insertion 和 deletion 作为 transition actions；
- 使用 auxiliary alignment/process 构造可训练的 Edit Flow；
- 对辅助时间过程进行边缘化的理论；
- 使用 Bregman divergence/flow-matching objective 训练 action rates；
- Edit Flows 已有的 random/optimal alignment construction、逐训练迭代 auxiliary-variable sampling、switch process 和 first-order sampling recipe；
- Edit Flows 已有的 localized auxiliary CTMC / localized propagation paths 与 edit propagation construction。

本项目新增或系统化的工作限定为：

1. source-conditioned 5′UTR editing benchmark，以及通过独立资格门槛后保留的 3′UTR 扩展；
2. Track E/Track F 数据角色、source/candidate/measurement 语义、provenance、exposure、artifact 和 split 治理；
3. persistent source/current conditioning、explicit edit budget、UTR-specific contract-valid mask 与 region/context interface；
4. 在 UTR endpoint-pair 数据上对 deterministic canonical alignment、native/fresh alignment proposal 与 multi-alignment Monte Carlo training 做匹配计算预算的预注册比较；
5. 将 endpoint generation、measured-pool ranking、effect prediction、region transfer 和 open-generation diagnostics 分离；
6. 对 library shortcut、alignment sensitivity、source conditioning、region transfer 和 foundation exposure 做可复现实证分析。

上述“项目新增”是贡献范围，不自动构成算法首创；对应盲测假设通过后，才允许报告性能增益。

截至 2026-08-03，Edit Flows、RNA/mRNA 生成、局部优化和大型核酸 benchmark 已有直接近邻。因此当前不得写：

- “首个 biological Edit Flow”；
- “首个 UTR 生成/优化模型”；
- “首个同时覆盖 5′和3′UTR 的方法”；
- “首个 source-proximal UTR optimization”；
- “首个 RNA/mRNA mutation benchmark”；
- “首个具有生物约束的 Edit Flow”；
- “学到了真实生物编辑轨迹”。
- “we introduce latent-path/alignment marginalization”；
- “the first model to marginalize UTR edit paths”；
- “a novel CTMC with insertion/deletion/substitution”；
- “a novel Bregman Edit Flow objective”，除非后续确有不同于上游且经理论验证的新目标；
- “path ensemble computes the exact endpoint likelihood”。

完整系统综述、feature matrix 和逐 claim 人类核验完成前，标题、摘要和结论一律不使用 `the first`，也不使用“未发现完全相同组合”替代实质创新比较。允许的英文方法边界是：

> “We build on Edit Flows and develop a source-conditioned UTR adaptation with persistent source/current representations, explicit edit budgets, contract-valid action masks and region/context interfaces. We further evaluate, rather than claim to originate, the role of alignment proposals and multi-alignment Monte Carlo training in endpoint-only UTR data.”

### 2.2A Prior-art feature matrix gate

论文冻结前必须生成 `docs/literature/prior_art_feature_matrix_v3_1.csv`，至少逐原始论文/官方代码核对 Edit Flows（同时 pin arXiv v3 与 NeurIPS/OpenReview provenance）、EvoFlows、Tree-Conditioned Edit Flows、mRNAutilus、Flexible Flows、RNAGenScape、LPDP、STRIDE、UTRGen、NucleoBench、NABench、mRNABench、PARADE、UTailoR、UTRGAN、mRNABERT、UTR-LM。每行至少填写：

```text
source_conditioned_editing
variable_length_ins_del_sub
auxiliary_alignment_or_path_handling
5utr_scope
3utr_scope
explicit_ref_alt_or_source_candidate
dense_library_and_no_edit_controls
functional_labels
source_study_family_disjoint_split
historical_or_foundation_exposure_audit
operationally_sealed_final
matched_search_generation_budget
public_build_adapters_and_provenance
verified_source_locator
verified_by
verified_at
```

`UNKNOWN` 不得填成 `NO`。只有 accountable human 打开 primary source、确认 locator 后才能标 verified。矩阵未闭合时不允许任何首创性表述；若已有工作覆盖同一组合，必须缩小 claim，不得通过改名维持“首个”。

## 2.3 主要科学问题

### RQ1：数据与 benchmark

严格区分 sequence scope、实验单位、source/candidate 方向、上下文、重复测量、historical exposure 和 reporter artifact 后，公开 UTR 数据能否形成可复用且非虚高的编辑 benchmark？

### RQ2：Edit Flows alignment construction 在 UTR endpoint-pair 数据中的鲁棒性

在继承 Edit Flows auxiliary-alignment framework 的前提下，UTR endpoint-pair 学习对 deterministic canonical alignment、上游 native construction、冻结 proposal 和 Monte Carlo alignment sample 数有多敏感？同一 proposal 下增加每个 update 的 alignment 样本能否降低梯度估计方差，并在明确的质量—计算前沿上带来可重复的实践收益？

### RQ3：区域与迁移

5′与3′在 source conditioning、编辑预算、action type 和跨 study/context 泛化上有哪些可共享规律，哪些必须区域专属？

### RQ4：分布建模与功能选择的任务边界

在 held-out endpoint reconstruction/distribution 任务上，Edit-Flows-derived editor 是否优于忠实 native Edit Flow、AR、masked/denoising 与 diffusion 类 editor？在独立的 closed measured-pool 功能任务上，冻结 scorer 与 search/ranking 方法在相同候选、查询和编辑预算下表现如何？两类结果必须分别回答，不能把“复现历史 library endpoint”写成“优化功能”。

## 2.4 可证伪假设

| ID | Confirmatory hypothesis | 失败含义 |
|---|---|---|
| H1A | 在相同 raw endpoint records、task/split、训练预算和 capacity-matched placebo control 下，`UTR_ADAPTED_NATIVE` 相对 `BASIC_PAIR_EF_NATIVE` 在冻结的 T5-Gen-Reconstruct primary metric 上满足 PR1 的增量效应规则；persistent source/context interface 属于被检验的 adaptation，不能同时假定 baseline 已拥有 | 不宣称 UTR adaptation advantage；只称 Edit-Flows-derived UTR reference implementation |
| H1B | 对同一 UTR architecture、updates/seeds/minibatches 与冻结 proposal，`UTR_ADAPTED_FRESH_Q1` 相对 `UTR_ADAPTED_CANONICAL` 改善 PR1 唯一冻结的 T5-Gen-Reconstruct endpoint metric；alignment-choice sensitivity 为机制性 secondary | 若 primary 不成立，proposal randomization 不构成 positive contribution，只报告 sensitivity/null result |
| H1C-M | 对同一 UTR architecture、冻结 proposal、10% update-budget checkpoint、32 个冻结 development minibatches 与 action-rate output-head 参数，在 §10.8.1 alignment-variance/nondegeneracy gates 通过后计算 `NMSE_K`；primary floor-stabilized contrast `log[(NMSE_8+τ_nmse)/(NMSE_1+τ_nmse)]` 的 paired cluster-bootstrap 95% CI 上界<0 | 机制性 variance-reduction claim 不成立；若 proposal 无可识别 alignment variance 则为 `H1C_M_NOT_ELIGIBLE` 而非 negative；不得改 checkpoint、minibatch、parameter subset、reference estimator、K、floor 或 statistic |
| H1C-P | 按 §10.8.2 的共同 FLOP grid，`UTR_ADAPTED_MULTI_QK(K=8)` 相对 `UTR_ADAPTED_FRESH_Q1(K=1)` 的 normalized quality-vs-log(FLOPs) AUC 差之 paired group-aware 95% CI 下界>0，且最大共同 FLOPs 处不劣于冻结 `Δ_quality` | 若只通过 H1C-M 而未通过 H1C-P，只报告 estimator finding；不得声称 practical endpoint advantage或新的 endpoint expectation |
| H2A | 在 T5-Gen-Reconstruct 的非 identity、`d(s,y)≥2` 层，persistent source representation 相对仅依赖 current state 的 capacity-placebo ablation 改善冻结 primary metric；t=0 因 `x_0=s` 不单独作为证据 | 若不成立，不得称 generator persistent-source advantage |
| H2B | 在 T5-Rank/T3-Effect 的同一 frozen measured denominator 上，source–candidate/delta scorer 相对 candidate-only scorer 改善各任务唯一 primary metric | 若不成立，不得称 functional source-conditioning advantage |
| H3 | 在同一冻结 strict test IDs/denominator 上，允许 test-group relatives 进入训练的 row-random protocol 相对 group-disjoint protocol 产生预注册的 apparent-performance inflation；两个 protocol 的数据量差额另做 matched-size control | 若不成立，报告区间，不制造 leakage 故事；不得比较不同 test set 难度后归因 shortcut |
| H4 | 在预注册的低 reporter-artifact 风险 primary subset 上，目标方法效应满足冻结 superiority rule；全量/HIGH/UNKNOWN 仅作风险分层 sensitivity | 若不成立，方法主张不能靠“差异可解释”挽救；只报告风险限定或 null result |
| H5 | 仅在 5′/3′具有相同 task、label semantics 与 common-support IDs 时，`shared_plus_region_adapter` 必须同时满足：相对 `fully_shared` superiority（95% CI 下界>0），以及相对 `independent` non-inferiority（95% CI 下界>`-Δ_NI`）；`Δ_NI` 在 PR1 以同一 generator primary metric 的尺度、cluster-aware precision、oracle/retrieval ceiling 与 smallest practically relevant recovery/similarity effect 冻结且不得参考模型结果 | 任一地区无资格、任一 intersection 条件失败或只靠 pooled 结果成立，均不宣称统一 region model 优势；分别报告 fully-shared、adapter、independent 的真实结果 |
| H6A | 在 T5-Gen-Reconstruct 唯一冻结 primary metric 与 matched generation budget 下，PR2 后以唯一 config/hash 解析的 `FULL_GENERATOR_FINAL_ALIAS` 优于 `BASIC_PAIR_EF_NATIVE` 与预注册最强 AR/denoising/diffusion editor；`UPSTREAM_REPRO_CHECK` 只验证官方实现忠实度 | 若不成立，方法不进入 positive headline；不得改用 T5-Rank 数字补偿，也不得在 final 后把 alias 指向另一配置 |
| H6B | 在 T5-Rank/Closed-Select 唯一冻结 primary metric 与同一 measured pool/budget 下，预注册 scorer/ranking/search 方法相对简单基线有正向效应 | 只报告 benchmark 与强基线结果；不得用 H6A endpoint recovery 支撑功能优化主张 |

以下为 exploratory，不得与 confirmatory 混写：3′多候选 ranking、小数据 source-specific 机制、开放生成的预测器分数、>5 edit budget、desired-outcome/reward-conditioned Edit Flow，以及未经新实验验证的开放式功能优化。STOP 在 v1 明确禁用；只有另立 v1.1 合同后才可能成为未来 exploratory。

H1B 的 primary endpoint result 与 alignment sensitivity 分层报告，不能取二者较好者。H1C-M 是机制性 confirmatory、H1C-P 是实践价值 confirmatory；“multi-sample estimator 既降方差又有实践价值”的组合主张使用 intersection rule，二者均通过且纳入 Holm family 才解锁。若只 H1C-M 通过，claim 固定为 estimator finding。

## 2.5 最大盲区与最大不确定性

当前最大的盲区是把 row count 当作科学有效样本量。有效单位必须同时报告 study、assay/library、independent parent、gene/tile family、context 和 sequence cluster。

当前最没有把握的三件事固定记录为：

1. ENCSR854RUF 是否能在 cryptic-splicing 与 pairing 复核后作为主要 3′ intervention evidence；
2. 严格去除 exposure、library identity 和相似 family 后，3′ indel/multi-edit 的独立证据还剩多少；
3. Edit-Flows-derived UTR adaptation 在 endpoint reconstruction/distribution 上能否超过忠实的 native Edit Flow 和同任务 editors；multi-alignment 是否降低梯度估计方差并改善质量—计算前沿。closed measured-pool 的功能 ranking/search 是另一条 estimand，不能用于给无 reward 条件的 editor 赋予功能优化主张。

这些不确定性是合同要检验的对象，不得用叙事提前消除。

## 2.6 Claim matrix 真相边界

| Claim ID | 候选主张 | 最低解锁条件 | 未满足时固定措辞 |
|---|---|---|---|
| C-BENCH-01 | 建成 provenance-controlled source-conditioned UTR benchmark | C3、D0-R、D1-R、FM0-A、B0-R、G7 全 PASS；E/F 双表与 release 权限闭环 | development dataset/benchmark draft |
| C-DATA-01 | 数据清洗完整 | `COMPLETE_CLEANING_FOR_V1_CONTRACT`；row/edge conservation、schema、pairing、labels、artifact、license、checksums 全闭环 | partially reconstructed / cleaning incomplete |
| C-METHOD-01 | UTR-specific adaptation contribution | 必须明确继承 Edit Flows；`BASIC_PAIR_EF_NATIVE→UTR_ADAPTED_NATIVE` 的冻结 bundle 对照、逐组件消融、capacity placebo 与 H1A frozen final 支撑收益 | Edit-Flows-derived UTR reference implementation |
| C-ALIGN-01 | alignment proposal / multi-sample 具有实践价值 | H1B endpoint primary 通过；multi-sample 组合主张还须 H1C-M 与 H1C-P 按 intersection+Holm 同时通过；不得把它们合成新 marginalization claim | alignment sensitivity / estimator finding / null result |
| C-REGION-01 | 5′主任务有效 | 5′资格、预注册指标、study/source-aware CI 与 matched baselines 通过 | 5′ development evidence only |
| C-REGION-02 | 3′扩展有效 | 3′独立数据资格、task-specific gate 与 region-transfer final 通过 | exploratory/qualified 3′ extension |
| C-REGION-ADAPTER | shared region adapter 有增量价值 | H5 在冻结 generator-reconstruction common support 上同时满足对 `fully_shared` superiority 与对 `independent` non-inferiority，5′/3′分别合格，并纳入 Holm family | region variants benchmarked；no shared-adapter advantage claim |
| C-EXTERNAL-01 | GSE246381 外部评测 | owner-confirmed prior use=`NONE_CONFIRMED`、pipeline materialization 完整登记、QC/许可/逐-checkpoint FM overlap/operational seal 通过并一次性 evaluator 完成 | sealed external candidate or excluded with evidence |
| C-BIO-01 | 功能/治疗改善 | 当前永不解锁；需要新 prospective wet-lab | prohibited claim |

每条论文数字和 factual claim 必须在 `docs/execution/claim_matrix_v3_1.yaml` 中绑定 evidence ID、artifact hash、analysis intent、denominator 和状态；合同、搜索摘要或模型输出本身不是证据。

---

# 3. Evidence、暴露与允许主张

## 3.1 必须拆开的四条证据轴

不得再用单一 `evidence_grade` 同时表达测量质量和模型暴露。下列字段构成彼此正交的权威轴，但不能任意复制到多个 sidecar：`scientific_track/relation_type/relation_acceptance_status/effect_evidence/landscape_role` 属于 technical canonical candidate/pair/observation 语义；project/pipeline/foundation 暴露轴属于以 stable record ID 一对一连接的 `ExposureRecord`；immutable base `future_use_role` 属于 accepted candidate↔pair/`UseRole`，其 current effective value只能由 role-transition ledger 投影得到；`EligibilityRecord` 只表达 §5.7 的 purpose-specific global eligibility，必须引用上述上游 ID/hash，禁止重新定义或覆盖 exposure/base role。所有适用轴都不得缺失或靠 accession 级默认值推断。mixed `DatasetAsset` 只保存 `potential_scientific_tracks` 与这些轴的资产级摘要，真实角色与 use/exposure disposition 必须逐 record 落盘，不能把整项资产压成一个 track：

```yaml
scientific_track: <E|F|AUX>
relation_type: <EXACT_REF_ALT|SOURCE_CANDIDATE|NO_EDIT_CONTROL|NOT_APPLICABLE>
relation_acceptance_status: <CANDIDATE|ACCEPTED|AMBIGUOUS|REJECTED|NOT_APPLICABLE>
effect_evidence: <UNKNOWN|SEQUENCE_ONLY|CANDIDATE_ONLY|BOTH_CROSS_CONTEXT|BOTH_SAME_CONTEXT>
landscape_role: <SPARSE|DENSE|NOT_APPLICABLE>
project_sequence_analytic_exposure: <NONE_CONFIRMED|PRESENT|UNKNOWN>
project_sequence_analytic_use_types: [<PROTOCOL_CALIBRATION|TRAIN|TUNE|MODEL_SELECTION|INTERNAL_TEST_EVALUATOR|ERROR_ANALYSIS|HUMAN_VIEW|FINAL_EVALUATOR|NONE_CONFIRMED|UNKNOWN>]
project_label_analytic_exposure: <NONE_CONFIRMED|PRESENT|UNKNOWN>
project_label_analytic_use_types: [<PROTOCOL_CALIBRATION|TRAIN|TUNE|MODEL_SELECTION|INTERNAL_TEST_EVALUATOR|ERROR_ANALYSIS|HUMAN_VIEW|FINAL_EVALUATOR|NONE_CONFIRMED|UNKNOWN>]
pipeline_sequence_materialization: <ABSENT|PRESENT|UNKNOWN>
pipeline_label_materialization: <ABSENT|PRESENT|UNKNOWN>
foundation_overlap_requirement: <REQUIRED_FM0_A|NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS>
foundation_audit_scope_id: <stable-scope-id>
foundation_overlap_audit_status: <NOT_STARTED|DEFERRED_TO_FM0_A|COMPLETE_PER_CHECKPOINT|NOT_APPLICABLE>
future_use_role: <GENERAL_DEVELOPMENT_POOL|SEALED_EXTERNAL_FINAL_CANDIDATE|SEALED_EXTERNAL_FINAL|EXTERNAL_STRESS_ONLY|EXCLUDED|PENDING>
```

`future_use_role` 是 pair 级的 **global reserved-use/effective-seal role**，不是某个 task/split 下的 train/dev/test assignment。`GENERAL_DEVELOPMENT_POOL` 只表示该对象没有被全局保留为 sealed final/stress/excluded；它本身既不证明可训练，也不指定 TRAIN、DEVELOPMENT 或 INTERNAL_TEST。具体分区的唯一权威是 `TaskEligibilityCell.disposition=ASSIGNED_TO_SPLIT` 所引用的 `assigned_partition_id`；同一 object 在不同 split contracts 可因不同科学 protocol 获得不同 partition role，但在同一 `split_contract_id` 下必须跨 task 保持同一 partition，并服从 §5.7.5 group consistency。global role 永远不能覆盖许可、task eligibility 或 cell assignment，cell assignment 也不能把 `SEALED_EXTERNAL_FINAL*|EXTERNAL_STRESS_ONLY|EXCLUDED` 对象偷偷变成 train/dev。

`NONE_CONFIRMED` 与任何实际 use type 互斥；pipeline materialization 不得自动把 project analytic exposure 改为 `PRESENT`，外部 foundation overlap 也不得改写项目历史。只有实际 analytic consumer 的 append-only access event 才能改变 project/final exposure 轴。

工程证据单列：

- `E0_ENGINEERING`：schema、replay、CUDA、artifact closure；不证明科学有效性。
- `E1_SYNTHETIC`：模拟/合成测试；不证明真实 assay 泛化。
- `E2_RETROSPECTIVE_MEASURED`：公开历史测量数据上的开发/评测。
- `E3_FROZEN_INTERNAL`：合同冻结后未用于选择的 benchmark final。
- `E_EXT_UNEXPOSED`：独立公开历史研究，项目侧训练、调参、错误分析和人工标签检查均未接触标签，经冻结 evaluator 一次性使用；强于内部开发集，但不等于 prospective evidence。GSE246381 使用该语义名，不使用任何 `E4*` 名称。
- `E_EXT_EXPOSED_STRESS`：其他数据若历史标签确已被查看，只能 retrospective stress/error analysis；该类别不适用于 GSE246381。
- `E5_NEW_PROSPECTIVE`：新实验或真正未暴露外部证据；当前不可用且不得假装存在。

## 3.2 GSE246381 项目侧未暴露 truth lock 与使用角色

用户已明确确认：在本项目的历史开发、模型训练、超参数选择、错误分析和人工标签检查中，GSE246381 未被使用或查看。此前把它写为 `E4/HISTORICALLY_EXPOSED` 的合同、报告、记忆和代码解释全部失效，不能继续引用。当前冻结事实为：

```yaml
accession: GSE246381

# 用户确认的历史“使用暴露”事实：
project_sequence_analytic_exposure: NONE_CONFIRMED
project_sequence_analytic_use_types: [NONE_CONFIRMED]
project_label_analytic_exposure: NONE_CONFIRMED
project_label_analytic_use_types: [NONE_CONFIRMED]
owner_prior_sequence_analytic_exposure: NONE_CONFIRMED
owner_prior_label_exposure: NONE_CONFIRMED
formal_training_exposure: NONE_CONFIRMED
formal_hyperparameter_exposure: NONE_CONFIRMED
prior_error_analysis_exposure: NONE_CONFIRMED
human_label_inspection: NONE_CONFIRMED
historically_exposed: false

# 已发生的数据管线物化，不等于训练/调参暴露：
pipeline_sequence_materialization: PRESENT
pipeline_label_materialization: PRESENT
legacy_canonical_membership: PRESENT
legacy_split_membership: PRESENT

# 外部预训练模型的语料重叠属于独立问题；真值按 checkpoint 分行，不能压成一个 accession 级布尔值：
foundation_overlap_requirement: REQUIRED_FM0_A
foundation_audit_scope_id: GSE246381_FROZEN_D1_CLUSTERS_V1
foundation_overlap_audit_status: DEFERRED_TO_FM0_A
foundation_exposure_ledger_status: NOT_YET_MATERIALIZED

project_evidence_status: RETROSPECTIVE_PUBLIC_PROJECT_UNEXPOSED
evidence_grade: E_EXT_UNEXPOSED
future_use_role: SEALED_EXTERNAL_FINAL_CANDIDATE
role_policy_source: V3_1_CONSERVATIVE_FINAL_ISOLATION
labels_allowed_for_training: false
labels_allowed_for_hyperparameter_selection: false
labels_allowed_for_pre_final_error_analysis: false
one_time_final_evaluation_policy: CONDITIONAL_PENDING_GATES
required_final_gates: [D1_QC, PERMITTED_EVALUATION_YES, FM0_A, B0_ISOLATION, EVALUATOR_FREEZE]
```

“未暴露”是历史使用事实，“pipeline materialization”是数据加工事实，“train/dev/final”是未来角色，三者不得混同。当前旧 B0 已把 GSE246381 分配到 train/val/test；这不证明发生过历史模型暴露，但与 v3.1 的 conservative sealed-final role 不一致，因此旧 manifests 必须失效重建。为保护独立外部价值，v1 默认将其保留为 sealed external final candidate；从 D1 重建开始，普通 canonical、普通 loader、notebook 和人工报告不得包含或返回其逐记录 candidate sequence、标签或 source-label join。若数据质量、许可、source/candidate 语义或 task eligibility 不通过，数据对象必须 `EXCLUDED_WITH_EVIDENCE`，不得自动转入训练集；foundation overlap gate 则先逐 checkpoint 排除不合格 model/claim，只有 frozen candidate set 中不存在任何许可且 overlap-clean 的 external candidate、`supervised_F_to_E` 的 F↔E final-lineage 也不干净、并且合同不允许 `from_scratch_E` 回退时，才可因模型路径完全不可用而排除对应 benchmark claim，不能仅因某一个 checkpoint 重叠而全局丢弃数据。

GSE246381 必须从 D1 起进入独立受限 sealed store：目录归属与权限、输入/输出 hash、隔离 builder/evaluator、访问日志和 commitment manifest 全部冻结。隔离 builder 可机械解析逐记录数据以完成 conservation、许可、lineage、FM overlap 与 aggregate-only QC，但向普通工作区只输出预注册聚合计数、失败原因分布、commitment hash 和资格状态；不得回传逐条序列、标签、ID→标签映射或可逆排序。aggregate QC 的 normalization、binning、missingness、pass/fail 与输出白名单必须在首次读取 label summary 前冻结；不得看分布后挑 normalization。旧公开 canonical 中已 materialize 的内容原样留作历史 evidence，但标 `SUPERSEDED_NOT_LOADABLE`，不得成为新训练/开发输入。

每次受限存储访问采用 durable 两阶段 append-only `ACCESS_LOG.jsonl`，不能要求一条“访问前完整 event”同时预知实际 rows touched 与 output hash。访问授权前必须先落盘并 fsync `ACCESS_INTENT`；它按 `exposure_record.schema.json#/$defs/AccessIntent` 验证，至少包含 `event_id/log_sequence_no/predecessor_event_id/predecessor_event_sha256/timestamp/actor_identity/executable_sha256/container_or_environment_sha256/input_manifest_sha256/event_type/requested_sequence_scope/requested_label_scope/requested_sequence_object_set_manifest_sha256/requested_label_object_set_manifest_sha256/output_schema_id/output_schema_sha256/analytic_access/requested_state_transition/reason/event_sha256`。object-set manifests 位于 restricted store，逐 object 绑定 stable ID/hash 与 raw-row→canonical-object transformation contributor closure；普通侧只能看到 hash/aggregate。只有 INTENT 的 executable、input、scope、object sets、event type、state 与 output schema 白名单全部 PASS 后，进程才获准读取。执行结束必须恰好追加一条显式含 `intent_event_id` 的 `ACCESS_COMPLETION` 或 `ACCESS_ABORT`，分别按 `#/$defs/AccessCompletion`/`#/$defs/AccessAbort` 验证；completion 才记录实际 `sequence_rows_touched/label_rows_touched/actual_sequence_object_set_manifest_sha256/actual_label_object_set_manifest_sha256/output_manifest_sha256/realized_state_transition`，actual sets 必须是 requested sets 的子集；abort 记录 failure evidence、可恢复的 partial actual-set hashes 或 `TOUCH_SCOPE_UNKNOWN`，且不伪造 output hash。三类 row 共享同一连续 hash chain；GENESIS 使用固定 predecessor sentinel，`event_sha256` 以 RFC 8785/JCS 覆盖除自身外完整 event。外部 WORM receipt 仍须保存并 hash，不能代替 schema chain。C3 必须提供正常 intent→completion、intent→abort、删行、重排、前驱篡改、orphan intent、双 completion、intent replay、scope subset violation、raw-row/object-FK mismatch、output-schema mismatch 与 crash-after-final-intent fixtures。`event_type` 只能取：

```text
RESTRICTED_BUILDER_PARSE
AGGREGATE_QC_MACHINE
FM_OVERLAP_AGGREGATE
B0_ELIGIBILITY_SPLIT_BUILD
G7_RESTRICTED_FINALIZER
TASK_PROTOCOL_CALIBRATION
INTERNAL_TEST_EVALUATOR
HUMAN_SEQUENCE_VIEW
HUMAN_LABEL_VIEW
TRAIN
TUNE
MODEL_SELECTION
PRE_FINAL_ERROR_ANALYSIS
ONE_TIME_FINAL_EVALUATOR
POST_FINAL_ERROR_ANALYSIS
```

只有前五类在 executable/output-schema 白名单 PASS、普通工作区无 row-level 输出、无人类读取且目的仅为 deterministic build/QC/overlap/eligibility-split build/finalizer replay 时可写 `analytic_access=false`，并分别计入独立 machine-access counter；它们保持项目 prior analytic exposure=`NONE_CONFIRMED`。`B0_ELIGIBILITY_SPLIT_BUILD` 可在 restricted store 内机械读取 row IDs、sequence/label-presence flags、许可/FM/task-gate metadata 并生成 restricted eligibility/role/split artifacts，但向普通工作区只允许输出冻结 schema 的不可逆 commitment、aggregate denominator/counters/status/hash；不得输出逐条 ID、sequence、label、join、排序或可逆 membership。`G7_RESTRICTED_FINALIZER` 只允许从同一 frozen restricted snapshot 重放校验并输出 aggregate closure/hash。`TASK_PROTOCOL_CALIBRATION|INTERNAL_TEST_EVALUATOR` 只能出现在 ordinary nonsealed log，必须写 `analytic_access=true` 并更新普通对象的 exposure projection；它们若出现在 restricted log 立即 `FINAL_INVALIDATED`。其余八类 restricted analytic event 一律 `analytic_access=true`，不得由执行者改写。正式 final 前若 restricted log 出现任何后八类 INTENT（即使随后 abort）、protocol-calibration/internal-test INTENT，或前五类输出越过白名单，立即 `FINAL_INVALIDATED`；不得删除日志或把事件重命名为 QC。

event→state 映射固定如下：普通 analytic event 的 INTENT 一经持久化即 fail closed 改写相应 exposure/final 状态；completion 只能补充实际触达证据，不能把 exposure 恢复为 NONE。`HUMAN_SEQUENCE_VIEW` 使 sequence axis=`PRESENT` 并追加 `HUMAN_VIEW`；`HUMAN_LABEL_VIEW` 使 label axis=`PRESENT` 并追加 `HUMAN_VIEW`；`TASK_PROTOCOL_CALIBRATION|TRAIN|TUNE|MODEL_SELECTION|INTERNAL_TEST_EVALUATOR` 向实际请求的 axes 追加对应 `PROTOCOL_CALIBRATION|TRAIN|TUNE|MODEL_SELECTION|INTERNAL_TEST_EVALUATOR` use type；`PRE_FINAL_ERROR_ANALYSIS` 追加 `ERROR_ANALYSIS` 并 `FINAL_INVALIDATED`；`POST_FINAL_ERROR_ANALYSIS` 仅在 `FINAL_OPENED` 后合法并追加 `ERROR_ANALYSIS`。`ONE_TIME_FINAL_EVALUATOR` 的 INTENT 必须以 compare-and-append 锁将 `SEALED_UNOPENED→FINAL_ACCESS_RESERVED`，同时 `one_time_final_attempt_count=1` 并对 requested axes保守写 exposure=`PRESENT`（若 intent 未能确定 touched scope则写 `UNKNOWN`，但绝不能保留 NONE）；completion 才使其 `FINAL_OPENED` 且 `analytic_final_labels_accessed=true`，abort、crash、超时或 orphan intent 一律 `FINAL_INVALIDATED`，其 actual-access flag 按 partial evidence 为 true 或 UNKNOWN，永不得退回 `SEALED_UNOPENED` 或重试同一 v1 final。前五类 machine event 不进入 analytic use types，只增加各自 machine counter。任何 INTENT 缺 scope、任一 INTENT 无唯一 terminal row、completion scope 超出 intent、raw-row→object scope FK不闭合或 state mapping冲突均 fail closed。每个 D1/B0/G7 restricted commitment 必须绑定 access-log schema hash、row count、first/last event ID、chain-root SHA256 与 WORM receipt hash（若有），并要求 `access_log_missing_event=0`、`access_log_sequence_gap=0`、`access_log_reordered_event=0`、`access_log_predecessor_hash_mismatch=0`、`access_log_event_hash_mismatch=0`、`access_log_output_schema_mismatch=0`、`access_intent_without_terminal=0`、`access_intent_replay=0`、`access_intent_multiple_terminal=0`、`access_completion_scope_exceeds_intent=0`、`access_scope_object_fk_mismatch=0`、`effective_exposure_projection_coverage_mismatch=0`、`one_time_final_reservation_count<=1`。

restricted live access authority 是 sealed cohort root 下唯一 append-only `ACCESS_LOG.jsonl`；它会跨 D1→FM0-A→B0-R→G7 继续追加，因此**不得**把这个可增长路径的当前文件 hash 当作历史 phase bundle 的永恒 hash。每个 phase 必须在 `<sealed_cohort_root>/access_snapshots/<restricted_snapshot_id>/` 以 temp+fsync+atomic-rename 生成且永不覆盖的三件套：`ACCESS_LOG.jsonl`、`ACCESS_MANIFEST.json` 与 `ACCESS_SHA256SUMS`。snapshot `ACCESS_LOG.jsonl` 必须是 live log 从 GENESIS 到冻结 `last_event_id` 的 exact byte-for-byte prefix；不得使用指向 live log 的 symlink/hardlink，不得在后续 phase 追加。每个 D1/FM0/B0/G7 commitment 只绑定本 phase immutable prefix snapshot，后续 live append 不得改变旧 bundle 的任一 byte/hash。

snapshot `ACCESS_MANIFEST.json` 至少含 `manifest_id/contract_id/contract_sha256/run_id/phase/restricted_snapshot_id/cohort_ids/cohort_set_sha256/live_access_log_relpath/snapshot_access_log_relpath/snapshot_access_log_sha256/access_log_schema_id/access_log_schema_sha256/event_count/first_event_id/last_event_id/access_log_chain_root_sha256/requested_object_set_manifest_relpaths/actual_object_set_manifest_relpaths/worm_receipt_relpaths/worm_receipt_sha256s/access_sha256s_relpath/access_sha256s_sha256/live_prefix_match_at_snapshot/manifest_sha256`；所有 relpath 必须相对 sealed cohort root、排序唯一且不能越界，`live_prefix_match_at_snapshot` 必须为 true。`ACCESS_SHA256SUMS` 必须按 UTF-8 relpath 字典序、每行 `<sha256><two-spaces><relative-path>\n` 精确覆盖 snapshot `ACCESS_LOG.jsonl`、该 prefix 引用的全部 requested/actual object-set manifests 与全部 WORM receipts，明确排除自身和 `ACCESS_MANIFEST.json`；所有被引用 object-set manifests 必须复制到本 snapshot 子树或使用 hash-addressed immutable path，禁止引用可覆盖 alias。随后 manifest 单向绑定该 checksum ledger，外层 phase/root manifest 再绑定最终 manifest bytes，避免循环引用。`manifest_sha256` 精确定义为 `SHA256(RFC8785/JCS(完整 JSON object 删除 manifest_sha256 字段后))`；加入该字段后的最终文件 bytes 必须且只能为 `UTF-8 RFC8785/JCS(完整 object)+单个 LF`，checksum ledger/外层 manifest 对这些 full-file bytes 取 SHA256。missing/extra/duplicate path、hash mismatch、manifest/log chain-root mismatch、live-prefix mismatch、旧 snapshot byte drift、mutable object-set alias 或 relpath escape 均 FAIL。任何固定根目录下的另造 manifest/checksum sidecar、未命名 sidecar或可变 snapshot 均不能替代这套 authority。C3 必须包含 manifest self-hash/full-file-hash golden/negative fixture、live-prefix exact-match、后续 append 不改变旧 snapshot、symlink/hardlink rejection、mutable object-set alias rejection 与 snapshot overwrite rejection tests。

GSE246381 的 B0/G7 必须使用 **dual-store**，不能为了 closure 把 sealed member IDs 重新泄露到普通工作区：

```text
ordinary data/v3_1/benchmark/                 # 只含非 sealed row-level objects
  B0_ROLE_DECISION_EVIDENCE.jsonl
  GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl
  ELIGIBILITY_MANIFEST.jsonl
  TASK_ACTIVATION_DECISIONS.jsonl
  SPLIT_ACTIVATION_DECISIONS.jsonl
  TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl
  TASK_ELIGIBILITY_UNIVERSE.jsonl
  RELATION_ROLE_TRANSITIONS.jsonl
  EFFECTIVE_ROLE_PROJECTION.jsonl
  SPLIT_ASSIGNMENTS.jsonl
  B0_ORDINARY_PREPARED_MANIFEST.json
  B0_TRANSACTION_COMMITS.jsonl

<restricted_run_root>/sealed_external/GSE246381/benchmark/
  B0_ROLE_DECISION_EVIDENCE.jsonl
  GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl
  ELIGIBILITY_MANIFEST.jsonl
  TASK_ELIGIBILITY_UNIVERSE.jsonl
  RELATION_ROLE_TRANSITIONS.jsonl
  EFFECTIVE_ROLE_PROJECTION.jsonl
  SPLIT_ASSIGNMENTS.jsonl
  B0_RESTRICTED_PREPARED_MANIFEST.json
  B0_RESTRICTED_MANIFEST.json
  B0_RESTRICTED_SHA256SUMS

ordinary data/v3_1/sealed_commitments/
  GSE246381_B0_AGGREGATE.json
  GSE246381_B0_COMMITMENT.json
  GSE246381_G7_COMMITMENT.json
```

上表是 B0 dual-store 的最低核心树；D1/B0/G7 的**穷尽 authoritative artifacts 清单**分别以 §14.5、§14.7、§14.8 为准，不得把本示意树解释为允许省略。每一个 ordinary row-level artifact——包括 canonical objects/candidates、rejections、transformation/supersession edges、groups/assignments、Exposure/UseRole/effective-exposure/current-leaf projections、B0 evidence/cells/roles/splits以及任何新增 row-level support table——都必须独立运行 accession/ID/source-file/source-lineage negative tests，GSE246381 count=0；只要 artifact 含可连接 member-level locator，就属于 row-level，不因文件名叫 report/registry 而豁免。对应 restricted mirror 不进入普通 Git/worktree、loader、notebook 或报告。`dataset_reconciliation`、`data_units_report`、Data Card 等普通报告只能含预注册 aggregate 且必须通过不可逆性检查；若含 member ID、row locator、可逆排序或逐条失败原因，必须移入 restricted store。普通 commitments 只能含预注册 aggregate denominator/status/counters、Merkle或文件 hash commitment 与 restricted manifest hash，不能含 member IDs、逐条 roles/eligibility/assignments、可逆顺序或 labels。每个 object 恰属于 ordinary 或 restricted shard 之一，`cross_store_object_overlap=0`、`sealed_object_in_ordinary_artifact=0`、`restricted_object_missing_from_commitment_count=0`。B0/G7 manifests 必须原子绑定 ordinary artifacts、restricted manifest commitment 和 access-log hash；closure totals 由 ordinary aggregate 与 restricted allowlisted aggregate相加，但任何 ordinary validator 都不得展开 restricted membership。

v1 的 sealed cohort 外部期望集合精确冻结为 `SEALED_COHORT_IDS_V1=[GSE246381]`；按 UTF-8 ID、行尾 LF 的 set SHA256 为 `275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0`。只有该 cohort 的 accepted E pairs 可以使用 `SEALED_EXTERNAL_FINAL_CANDIDATE|SEALED_EXTERNAL_FINAL` 与上述 restricted shard；其他对象在 v1 不得自行获得 SEALED role。D0 若发现新的 sealed 候选，只能登记 `PENDING_CONTRACT_UPGRADE` 并留在普通训练/评测 universe 之外；必须由用户批准 v1.1 cohort-registry/受限路径/commitment 升级后才能使用，执行者不得照抄 GSE246381 路径或静默扩大 sealed set。强制 `sealed_cohort_expected_set_mismatch=0`、`unexpected_sealed_role_object_count=0`。

当前 `d1_staging/scripts/d1/build_exposure_ledger.py` 与 `d1_staging/tests/test_d1_exposure_ledger.py` 将其历史方向写为 unexposed，这一方向应保留并升级为分轴 schema；不能因旧 v2 合同错误而把代码反向改成 exposed。必须 supersede 的是旧 v2 E4 语义；必须重建的是未来 use role、access policy 与 split。

一次性 final 后必须更新：

```yaml
final_access_status: FINAL_OPENED
labels_allowed_for_future_hyperparameter_selection: false
eligible_as_new_final_in_v1_1: false
post_final_error_analysis_allowed: true
```

它仍是公开历史 assay 的 retrospective external evidence，不是新湿实验或 prospective E5。

## 3.3 允许主张阶梯

| Gate | 允许主张 |
|---|---|
| 仅 MK0/EF0 E0 | 实现了可执行的 source/current-conditioned action-rate 工程原型；不得说模型有效 |
| D0/D1 PASS | 构建流程、数据角色、attrition、provenance 和 license 状态可审计；不得说 benchmark 已优越 |
| B0 PASS | benchmark v1 的任务、split、泄漏和 final 封存闭环 |
| GP0/MB0 PASS | 在预注册指标与匹配预算下报告方法效应量和不确定性 |
| PP0 PASS | 可提交 benchmark/resource + method reference paper；仍无湿实验功能主张 |

永久禁止：

- biological improvement、therapeutic optimization、safety、efficacy；
- predictor score 等同于真实功能改善；
- contract-valid 等同于 biologically legal；
- observational/absolute label 等同于 causal delta；
- latent alignment indel 等同于真实实验 insertion/deletion；
- internal frozen split 等同于 prospective external validation；
- GPU gate、smoke、row count 或 manifest 自洽等同于 publication-ready。

---

# 4. Track E / Track F 数据合同、样本单位与发布边界

本项目 v1 不建立 Track U，不执行项目侧无标签序列预训练。所有数据发现、清洗、规模报告、训练和 benchmark gate 只围绕 Track E 与 Track F。无标签资源不能通过“序列很多”增加论文数据规模。

## 4.1 Track E：Explicit edit / ref-alt / dense landscape

Track E 是 source-conditioned editing benchmark 与 Edit Flow 训练/评价的 primary 数据轨。以下名称不是一个互斥 `track_role`，而是由多条正交字段派生的分析视图：

- `E_DELTA`：source/ref 与 candidate/alt 在同一或预注册可比 assay/context 中均有功能测量，方向和 pairing evidence 明确，才可计算 confirmatory delta；
- `E_LINK`：source→candidate/ref→alt 关系确定，但一侧功能标签缺失或 context 不可直接比较，只能做 endpoint recovery、关系建模或描述性分析；
- `E_DENSE`：一个或少数母本的系统候选景观；candidate rows 必须与 independent parents 分开报告；
- `E_NOEDIT`：identity/WT control，用于 no-op、校准、噪声和 source measurement；不进入要求真实 edit event 的 recovery 分母；
- `E_EXTERNAL_FINAL`：role projection 的 `effective_future_use_role=SEALED_EXTERNAL_FINAL`，且 B0 `EligibilityRecord` 的 effective evaluation status 为 `ELIGIBLE`、通过所有隔离 gate 的视图；
- `E_EXTERNAL_STRESS`：`effective_future_use_role=EXTERNAL_STRESS_ONLY` 且 effective evaluation status=`ELIGIBLE` 的预先定义 sensitivity 视图，不自动支撑泛化主张。

同一 accepted pair 可同时属于 dense landscape、no-edit control 或 delta/link view，并拥有独立的 future use role。例如 GSE246381 在 D1/B0 前的 staging records 只能写 `scientific_track=E`、`relation_acceptance_status=CANDIDATE`、相应 `landscape_role` 与 `future_use_role=SEALED_EXTERNAL_FINAL_CANDIDATE`；通过 relation/observation join 后，只有 accepted records 才进入 pair table 并派生 `E_DELTA/E_LINK`，不得预先把整个 accession 写成已验真的 E_DELTA。

```text
E_DELTA := scientific_track=E
           AND relation_acceptance_status=ACCEPTED
           AND relation_type != NO_EDIT_CONTROL
           AND effect_evidence=BOTH_SAME_CONTEXT
           AND pair_direction_verified=true
           AND confirmatory_delta_eligible=true
E_LINK  := scientific_track=E
           AND relation_acceptance_status=ACCEPTED
           AND relation_type != NO_EDIT_CONTROL
           AND NOT E_DELTA
E_NOEDIT := scientific_track=E
            AND relation_acceptance_status=ACCEPTED
            AND relation_type=NO_EDIT_CONTROL
E_DENSE := scientific_track=E
           AND relation_acceptance_status=ACCEPTED
           AND landscape_role=DENSE
E_EXTERNAL_FINAL := scientific_track=E
                    AND relation_acceptance_status=ACCEPTED
                    AND effective_future_use_role=SEALED_EXTERNAL_FINAL
                    AND effective_evaluation_eligibility=ELIGIBLE
E_EXTERNAL_STRESS := scientific_track=E
                     AND relation_acceptance_status=ACCEPTED
                     AND effective_future_use_role=EXTERNAL_STRESS_ONLY
                     AND effective_evaluation_eligibility=ELIGIBLE
```

尚未验真的 dense/no-edit 对象只能分别报告为 `E_DENSE_CANDIDATE_OVERLAY` 与 `E_NOEDIT_CANDIDATE_OVERLAY` staging counts，必须带 candidate/ambiguous/rejected disposition，永不进入正式 E view、task denominator 或数据规模 headline。

只有 source/ref、candidate/alt、方向、sequence scope、实验 context、source file 和 pairing evidence 可验证的记录才能进入 Track E。raw/barcode/replicate/sample rows 一律不得直接称 pair。

## 4.2 Track F：Function-labeled but source-unresolved

Track F 包括随机 UTR library、自然序列功能测量、无法可靠恢复 source 的设计序列和 candidate-only measured sequence。

允许用途：

- absolute-property prediction；
- supervised auxiliary representation training；
- context-transfer analysis；
- retrieval/scorer training；
- generator critic 的 development-side 训练，但不得与 final evaluator 共用标签角色。

禁止用途：

- 计入 paired edit 数量或 independent edit parent；
- 构造伪 source 或仅凭相似度升级为 Track E；
- 产生 causal/confirmatory delta；
- 作为 latent edit path 的 endpoint pair；
- 与 Track E final 在 exact sequence、near cluster、parent/family、study/library 或 label lineage 上重叠。

## 4.3 无标签资产的 v1 排除规则

GENCODE、RefSeq、Ensembl、UTRdb、RNAcentral 等无标签序列资源在 v1 中统一标记为 `REFERENCE_ONLY_NOT_TRAINING` 或 `OUT_OF_SCOPE_UNLABELED_WATCHLIST`：

- 只可用于 transcript/UTR 坐标恢复、reference allele、strand、release 注释和去重参考；
- 不产生训练 example，不进入 D0/D1/G4 的功能数据穷尽门槛；
- 不进入任何数据规模，不用于项目侧预训练；
- 不得用其规模扩展论文的数据量或 representation claim。

外部已经训练好的 backbone 仍需由 FM0 审计其原始预训练语料与 benchmark overlap；该审计不构成 Track U，也不授权本项目引入无标签语料。

## 4.4 禁止统一 N_total

每个数据版本必须先报告 raw/design 通用口径：

```text
design_rows
downloaded_rows
observed_rows
endpoint_missingness
dropped_rows_by_reason
```

再分别输出两张不能相加的科学规模表：

```text
E_TABLE:
  unique_design_relation_groups
  endpoint_context_pair_instances
  relation_links
  delta_eligible_pairs
  e_link_only
  no_edit_controls
  dense_candidates
  independent_parents
  independent_genes
  sequence_clusters
  assay_contexts
  biological_replicates
  technical_replicates
  barcodes_or_umis

F_TABLE:
  unique_labeled_sequences
  functional_observations
  independent_genes
  sequence_clusters
  assay_contexts
  biological_replicates
  technical_replicates
  barcodes_or_umis
  endpoint_missingness
```

不得用一个总行数描述数据规模。`unique_design_relation_groups` 是去除同一 source→candidate edit 的 endpoint/context 重复后的 sequence-edit unit；`endpoint_context_pair_instances` 是测量实例，两者必须并列，后者不得替代前者作 E 规模 headline或 reconstruction denominator。`dense_candidates`、`no_edit_controls` 是可与 `delta_eligible_pairs/e_link_only` 重叠的 overlay counts，不能相加得到 E 总数。条码/biological replicate/technical replicate/UMI 行、同序列跨 context、dense single-parent library、E/F 角色和 reference-only 资产必须分账；reference-only 无标签序列不得进入科学数据规模。

每个进入合同、Data Card、图表或论文的 N 必须有机器可读记录：`value/unit/denominator_definition/analysis_population/verification_state/evidence_locator/nonadditive_with`。缺任一字段只能写 `UNVERIFIED_COUNT`，不得进入规模 headline。

## 4.5 v1 发布与许可

v1 资产发现快照冻结在 2026-08-03。每个发现资产必须完成：

1. 取得并接入；或
2. 标记 `EXCLUDED_WITH_EVIDENCE`，保留原因、失败日志、规模和来源 hash；或
3. 按原子轴标记 `acquisition_status=METADATA_ONLY|DOWNLOAD_FAILED` 或 `license_status=REVIEW_REQUIRED`，不计入相应用途的可用规模。

可再分发数据才进入 release bundle；不能确认再分发的资产只发布下载 adapter、accession、checksum、转换脚本和 derived non-reconstructive metadata。数据库公开访问、论文开放许可或代码仓库许可均不得自动推断数据可再分发。

---

# 5. Canonical schemas 与数据接口

## 5.1 强制 schema 集合

仓库必须实现并版本化：

1. `dataset_asset.schema.json`
2. `sequence_entity.schema.json`
3. `functional_observation.schema.json`
4. `utr_edit_relation_candidate.schema.json`
5. `utr_edit_pair.schema.json`
6. `edit_path_set.schema.json`
7. `generation_task.schema.json`
8. `exposure_record.schema.json`
9. `split_assignment.schema.json`
10. `rejection_record.schema.json`
11. `use_role.schema.json`
12. `eligibility_record.schema.json`
13. `transformation_edge.schema.json`
14. `reporter_artifact_assessment.schema.json`
15. `group_registry.schema.json`
16. `group_assignment.schema.json`
17. `relation_role_transition.schema.json`
18. `task_registry.schema.json`
19. `split_registry.schema.json`
20. `task_split_applicability.schema.json`
21. `task_eligibility_cell.schema.json`

active root `schemas/v3_1/` 的 schema filename set 必须与上述 21 项完全相等；按 UTF-8 filename 字典序、每项后一个 LF 的 expected-set SHA256 为 `d2e5ddaef3665214007422638df3cc6b0357747aad3911efd4f29319647b1762`。每个 schema 必须有唯一 `$id`、`schema_version=3.1`、`contract_id=utr_editflow_goal_v3.1_benchmark_first`，并进入 `schemas/v3_1/SCHEMA_MANIFEST.json` 与 `SCHEMA_SHA256SUMS`。历史/实验 schema 只能在 active root 外保存；active root 内 missing、duplicate `$id`、unexpected schema、version/contract mismatch 或 hash mismatch 的计数必须全为 0。21 个 schema 各自至少有一个 positive fixture 和一个针对关键约束的 negative fixture；只验证文件存在不能 C3 PASS。

不增加第 22 个 schema；下列 required machine rows 必须由对应 schema 的命名 `$defs` 验证并各有 positive/negative/golden fixture：`functional_observation.schema.json#/$defs/FunctionalObservationCandidate|EndpointRegistryRow`；`exposure_record.schema.json#/$defs/AccessIntent|AccessCompletion|AccessAbort|FoundationExposureAuditRow|EffectiveExposureProjection`；`transformation_edge.schema.json#/$defs/SupersessionEdge|CurrentCanonicalObjectProjection`；`relation_role_transition.schema.json#/$defs/EffectiveRoleProjection|B0PreparedManifest|B0TransactionCommit`；`task_registry.schema.json#/$defs/ActivationCalibrationMaskRow|TaskActivationDecision`；`split_registry.schema.json#/$defs/SplitActivationDecision`；`task_split_applicability.schema.json#/$defs/TaskSplitDefinitionRow|TaskSplitApplicabilityDecision`；`eligibility_record.schema.json#/$defs/B0RoleDecisionEvidence|GlobalEligibilityDecisionEvidence`；`generation_task.schema.json#/$defs/DiagnosticRegistryRow`。missing `$defs`、wrong `$ref`、未验证 required JSONL 或只给主 schema fixture 均使 C3 FAIL。

formal canonical 禁止 legacy fallback 或下游静默推断字段。

## 5.2 DatasetAsset 最低字段

```yaml
asset_id: <stable-versioned-id>
accession: <source-accession>
study_id: <study>
provider: <GEO|SRA|ENCODE|BioStudies|supplement|author-repository|other>
publication_ids: [<DOI-or-PMID>]
bioproject_or_project_id: <id-or-null>
source_url: <url>
source_release: <release-or-unknown>
downloaded_at: <iso-time>
original_filename: <name>
byte_size: <integer>
sha256: <sha256>
provider_checksum: <value-or-null>
license_name: <name-or-UNKNOWN>
license_evidence_url: <url-or-null>
rights_holder: <name-or-UNKNOWN>
license_scope: <DATA|CODE|METADATA|SUPPLEMENT|MULTIPLE|UNKNOWN>
terms_version: <version-or-date-or-UNKNOWN>
license_evidence_sha256: <sha256-or-null>
license_evidence_retrieved_at: <iso-time-or-null>
license_checked_at: <iso-date-or-null>
license_reviewer: <human-or-null>
attribution_or_citation_requirements: [<requirement>]
use_basis_notes: <text-or-null>
use_basis_evidence_ids: [<stable-evidence-id>]
permitted_download: <YES|NO|UNKNOWN>
permitted_processing: <YES|NO|UNKNOWN>
permitted_model_training: <YES|NO|UNKNOWN>
permitted_evaluation: <YES|NO|UNKNOWN>
permitted_derived_release: <YES|NO|UNKNOWN>
permitted_raw_redistribution: <YES|NO|UNKNOWN>
license_status: <VERIFIED|REVIEW_REQUIRED|RESTRICTED|UNKNOWN>
redistribution_status: <ALLOWED|ADAPTER_ONLY|FORBIDDEN|UNKNOWN>
acquisition_status: <NOT_PRESENT|METADATA_ONLY|DOWNLOAD_FAILED|PARTIAL|DOWNLOADED_UNVERIFIED|DOWNLOADED_VERIFIED>
parse_status: <NOT_STARTED|PARTIAL|PARSED|FAILED>
mapping_status: <NOT_STARTED|PARTIAL|DETERMINISTIC|AMBIGUOUS|FAILED>
canonical_status: <NOT_APPLICABLE|PENDING|ACCEPTED|EXCLUDED_WITH_EVIDENCE>
potential_scientific_tracks: [<E|F|AUX|REFERENCE>]
scientific_status: <PENDING|ELIGIBLE_SUBSETS_IDENTIFIED|REFERENCE_ONLY|AUXILIARY_ONLY|EXCLUDED>
release_decision: <INCLUDE_RAW|INCLUDE_DERIVED|ADAPTER_ONLY|DO_NOT_RELEASE|PENDING>
parser_commit: <full-sha-or-null>
parser_config_sha256: <sha256-or-null>
failure_reason: <reason-or-null>
acquisition_attempt_evidence_ids: [<id>]
```

条件约束：只有 `acquisition_status=DOWNLOADED_*` 时 `downloaded_at/original_filename/byte_size/sha256` 必填；`NOT_PRESENT/METADATA_ONLY/DOWNLOAD_FAILED` 时这些文件字段允许 null，但 `source_url`、failure reason、尝试路线与 evidence 必填。`parse_status=NOT_STARTED` 时 parser commit/config 可为 null。资产级 `potential_scientific_tracks` 只表示可能含有哪些子库；E/F/AUX 的真实归属必须在记录级完成，不能把 mixed asset 压成一个角色。

## 5.3 SequenceEntity 最低字段

```yaml
sequence_id: <stable-id>
primary_asset_id: <DatasetAsset.asset_id>
contributing_asset_ids: [<DatasetAsset.asset_id>]
contributing_source_file_sha256s: [<sha256>]
contributor_set_sha256: <sha256>
sequence_reconstruction_rule_id: <stable-id>
sequence_reconstruction_rule_sha256: <sha256>
source_record_id: <provider-record-id>
source_row_locator: <sheet/table/row-or-record-locator>
raw_sequence_sha256: <sha256>
normalized_sequence: <ACGT-sequence-or-null>
normalized_sequence_sha256: <sha256-or-null>
normalization_steps: [UPPERCASE, U_TO_T]
alphabet_status: <EXACT_ACGT|IUPAC_AMBIGUOUS|INVALID>
model_sequence_eligible: <bool>
invalid_symbol_status: <PASS|QUARANTINED>
region: <5UTR|3UTR|CDS_AUX|FULL_LENGTH_AUX>
sequence_scope: <FULL_UTR|UTR_WINDOW|RANDOM_INSERT|OLIGO_CONSTRUCT|FULL_LENGTH|CDS_AUXILIARY>
species: <taxon>
reference_build: <build-or-not-applicable>
transcript_release: <release-or-not-applicable>
strand: <+|-|UNKNOWN|NOT_APPLICABLE>
original_length: <int>
window_start: <int-or-null>
window_end: <int-or-null>
full_sequence_sha256: <sha256-or-null>
scaffold_id: <id-or-null>
editable_mask: <coordinate-set-or-null>
```

不得静默删除非 ACGTU/IUPAC 字符。允许的 U→T/大小写转换必须保留原始 hash、转换列表和长度守恒。只有 `EXACT_ACGT` 默认可进入 sequence model；IUPAC 展开必须另有预注册规则。synthetic/random construct 可使用 `strand=NOT_APPLICABLE`；editable mask 对 GenerationTask/E source-conditioned task 必填，对纯 F observation 可为 null。`primary_asset_id` 只作主要 locator，不能替代完整 contributor set；设计表、FASTA/reference/scaffold、variant map 与 assay support 中实际参与重建者都必须进入 contributors。SequenceEntity contributor set 必须与所有 inbound transformation-edge contributor closure、ExposureRecord rights contributors 完全相等；强制 `sequence_transformation_contributor_set_mismatch=0`、`sequence_exposure_rights_contributor_set_mismatch=0`。

## 5.4 FunctionalObservation 最低字段

任何 sequence×context×endpoint 的 label join proposal 都必须先进入持久化 `functional_observation_candidates.jsonl`，其 row 按 `functional_observation.schema.json#/$defs/FunctionalObservationCandidate` 验证；accepted row 也不得从 lifecycle 删除。最低字段为：

```yaml
observation_candidate_id: <stable-id>
asset_ids: [<DatasetAsset.asset_id>]
contributing_source_file_sha256s: [<sha256>]
contributor_set_sha256: <sha256>
source_unit_ids: [<raw-unit-id>]
sequence_id: <SequenceEntity.sequence_id-or-null>
context_id: <GroupRegistry.EXPERIMENTAL_CONTEXT.group_id>
endpoint_id: <EndpointRegistry.endpoint_id>
join_method_id: <id>
join_method_sha256: <sha256>
observation_acceptance_status: <CANDIDATE|ACCEPTED|AMBIGUOUS|UNMATCHED|REJECTED>
accepted_observation_id: <FunctionalObservation.observation_id-or-null>
terminal_disposition_reason: <controlled-reason-or-null>
source_row_locators: [<locator>]
evidence_ids: [<id>]
parent_candidate_id: <id-or-null>
```

`ACCEPTED` 必须且只能有一个 `accepted_observation_id` 与 reciprocal transformation edge；其他状态必须为 null，并分别保留 reason/evidence。修复 join 语义时创建 parent-linked superseding candidate，禁止改写旧 row。

```yaml
observation_id: <stable-id>
observation_candidate_id: <FunctionalObservationCandidate.observation_candidate_id>
canonical_status: <ACCEPTED>
sequence_id: <id>
primary_label_asset_id: <DatasetAsset.asset_id>
contributing_asset_ids: [<DatasetAsset.asset_id>]
contributing_source_file_sha256s: [<sha256>]
contributor_set_sha256: <sha256>
parent_observation_id: <id-or-null>
scientific_track: <E|F|AUX>
observation_role: <F_FUNCTION_LABEL|E_SOURCE_MEASUREMENT|E_CANDIDATE_MEASUREMENT|E_NOEDIT_MEASUREMENT|AUX_QC>
source_file_sha256: <sha256>
source_record_id: <provider-record-id>
source_row_locator: <sheet/table/row-or-record-locator>
context_id: <GroupRegistry.EXPERIMENTAL_CONTEXT.group_id>
endpoint_id: <EndpointRegistry.endpoint_id>
raw_value: <number-or-null>
normalized_value: <number-or-null>
label_status: <OBSERVED|DERIVED|MISSING|BELOW_COVERAGE|QUARANTINED>
label_unit: <EndpointRegistry.label_unit>
label_transform: <EndpointRegistry.label_transform>
cell_context: <controlled-value-or-UNKNOWN>
assay_context: <controlled-value>
promoter: <value-or-UNKNOWN>
reporter_or_cargo: <value-or-UNKNOWN>
rna_chemistry: <value-or-UNKNOWN>
timepoint: <value-or-UNKNOWN>
source_replicate_label: <source-native-value-or-null>
sample_id: <stable-id-or-null>
biological_replicate_id: <stable-id-or-null>
technical_replicate_id: <stable-id-or-null>
barcode_id: <id-or-null>
coverage_or_umi: <number-or-null>
standard_error: <number-or-null>
missingness_reason: <reason-or-null>
quality_flags: [<flag>]
```

`raw_value=null` 时必须给出 `missingness_reason` 且 `label_status` 不得为 `OBSERVED`。F 只由 `scientific_track=F` 且 label-complete 的 sequence×context observation 构成；input/normalization/support row 只能通过 transformation edge 支撑 observation，不能自身计为 F example。

context 不得再哈希成任意实数当作生物距离。`GroupRegistry(group_type=EXPERIMENTAL_CONTEXT)` 是唯一 ContextRegistry authority：每个 context group 必须记录 raw→canonical mapping、ontology/version、mapping rule ID/hash、`RESOLVED|UNKNOWN|AMBIGUOUS|NOT_APPLICABLE` 状态与 parent hierarchy；observation/relation 只能使用其 FK，禁止拼接自由字符串。模型输入使用 learned categorical embeddings、明确的 missing token 和 held-out-context 规则。另在 `data/v3_1/canonical/ENDPOINT_REGISTRY.jsonl` 冻结每个 `endpoint_id/biological_quantity/raw_field_mappings/label_unit/directionality/label_transform/comparability_scope/aggregation_rule_id+sha256/delta_rule_id+sha256/unknown_or_ambiguous_policy/record_sha256`。跨 study pooling/ranking 只有在该 transform 后 `comparability_scope` 明确兼容时允许；否则只能 study-specific 或预注册 normalized estimand。强制 lifecycle↔observation 双向一一对应与 payload/edge 校验：`accepted_observation_candidate_without_observation=0`、`observation_without_accepted_candidate=0`、`observation_candidate_fk_mismatch=0`、`observation_candidate_payload_mismatch=0`、`observation_candidate_contributor_set_mismatch=0`、`context_registry_fk_mismatch=0`、`context_raw_mapping_merge_or_split_collision=0`、`endpoint_registry_fk_mismatch=0`、`cross_study_endpoint_incomparability_violation=0`；`primary_label_asset_id` 必须属于完整 contributor set，不能用单一主文件丢掉 design/assay/supplement 的限制性 rights。ambiguous/unmatched/pending/rejected 均进入独立 attrition denominator，不能靠 RejectionRecord 省略。

## 5.5 UTREditPair 最低字段

所有 proposed relation 先进入持久化 `UTREditRelationCandidate` lifecycle table；被接受后也不得从 staging history 删除：

```yaml
relation_candidate_id: <stable-id>
parent_relation_candidate_id: <id-or-null>
design_relation_group_id: <stable-id>
contributing_asset_ids: [<DatasetAsset.asset_id>]
contributing_source_file_sha256s: [<sha256>]
contributor_set_sha256: <sha256>
relation_context_key: <study-assay-cell-promoter-cargo-chemistry-time-key>
context_id: <GroupRegistry.EXPERIMENTAL_CONTEXT.group_id>
endpoint_id: <EndpointRegistry.endpoint_id>
label_unit: <EndpointRegistry.label_unit>
label_transform: <EndpointRegistry.label_transform>
delta_rule_id: <EndpointRegistry.delta_rule_id>
delta_rule_sha256: <sha256>
scientific_track: <E>
relation_acceptance_status: <CANDIDATE|ACCEPTED|AMBIGUOUS|REJECTED>
relation_type: <EXACT_REF_ALT|SOURCE_CANDIDATE|NO_EDIT_CONTROL>
effect_evidence: <UNKNOWN|SEQUENCE_ONLY|CANDIDATE_ONLY|BOTH_CROSS_CONTEXT|BOTH_SAME_CONTEXT>
landscape_role: <SPARSE|DENSE|NOT_APPLICABLE>
future_use_role: <GENERAL_DEVELOPMENT_POOL|SEALED_EXTERNAL_FINAL_CANDIDATE|SEALED_EXTERNAL_FINAL|EXTERNAL_STRESS_ONLY|EXCLUDED|PENDING>
source_sequence_id: <SequenceEntity.sequence_id-or-null>
candidate_sequence_id: <SequenceEntity.sequence_id-or-null>
pairing_method: <DESIGN_TABLE|EXPLICIT_ID|VARIANT_RECONSTRUCTION|BARCODE_JOIN|OTHER>
pair_evidence_id: <stable-evidence-id>
terminal_disposition_reason: <reason-or-null>
accepted_pair_id: <pair-id-or-null>
```

`source_sequence_id/candidate_sequence_id` 都是对 §5.3 `SequenceEntity.sequence_id` 的显式外键；合同中不存在未定义的 `SequenceCandidate` 实体，也不得用临时 row index、FASTA ordinal 或文件内局部 ID 代替该外键。

一个 raw design relation 若覆盖多个不等价 context 或 endpoint，必须生成共享 `design_relation_group_id`、但具有不同 `context_id×endpoint_id` 的原子 relation candidates；`relation_context_key` 只是由 ContextRegistry FK 确定性导出的便利键，不得作为第二 authority。不得让一个 candidate 在不透明的一对多展开中生成多个 pairs。只有 `relation_acceptance_status=ACCEPTED`、两端 sequence IDs 非空、`terminal_disposition_reason=null` 且 `accepted_pair_id` 非空的对象才能生成下列正式 pair；staging candidate 不得被 loader、Data Card 或论文称为 accepted pair。

```yaml
pair_id: <stable-id>
parent_pair_id: <id-or-null>
relation_candidate_id: <UTREditRelationCandidate.relation_candidate_id>
design_relation_group_id: <stable-id>
contributing_asset_ids: [<DatasetAsset.asset_id>]
contributing_source_file_sha256s: [<sha256>]
contributor_set_sha256: <sha256>
context_id: <GroupRegistry.EXPERIMENTAL_CONTEXT.group_id>
endpoint_id: <EndpointRegistry.endpoint_id>
label_unit: <EndpointRegistry.label_unit>
label_transform: <EndpointRegistry.label_transform>
delta_rule_id: <EndpointRegistry.delta_rule_id>
delta_rule_sha256: <sha256>
scientific_track: <E>
relation_acceptance_status: <ACCEPTED>
relation_type: <EXACT_REF_ALT|SOURCE_CANDIDATE|NO_EDIT_CONTROL>
effect_evidence: <UNKNOWN|SEQUENCE_ONLY|CANDIDATE_ONLY|BOTH_CROSS_CONTEXT|BOTH_SAME_CONTEXT>
landscape_role: <SPARSE|DENSE|NOT_APPLICABLE>
future_use_role: <GENERAL_DEVELOPMENT_POOL|SEALED_EXTERNAL_FINAL_CANDIDATE|SEALED_EXTERNAL_FINAL|EXTERNAL_STRESS_ONLY|EXCLUDED|PENDING>
source_sequence_id: <id>
candidate_sequence_id: <id>
source_observation_id: <id-or-null>
candidate_observation_id: <id-or-null>
delta: <number-or-null>
delta_standard_error: <number-or-null>
same_assay_context: <bool>
biological_parent_group: <id>
gene_group: <id-or-null>
tile_family_group: <id-or-null>
sequence_cluster_group: <id>
true_length_change: <int>
minimum_edit_distance: <int>
path_ambiguity_count_or_bound: <value>
pair_direction_verified: <bool>
pairing_method: <DESIGN_TABLE|EXPLICIT_ID|VARIANT_RECONSTRUCTION|BARCODE_JOIN|OTHER>
join_keys: [<field>]
pair_evidence_id: <stable-evidence-id>
confirmatory_delta_eligible: <bool>
link_view_eligible: <bool>
permission_evidence_ids: [<asset-license-evidence-id>]
exclusion_reason: <controlled-reason-or-null>
```

training/evaluation/derived-release/raw-redistribution eligibility 不是 immutable technical canonical identity，禁止写回 frozen candidate/pair base row；其权威位置是 §5.7 的 purpose-specific `EligibilityRecord`、`ELIGIBILITY_MANIFEST.jsonl` 与相应 `TaskEligibilityCell`。许可 evidence 可以留在 pair 上作为 provenance 引用，但四类 eligibility 状态必须在 B0 对冻结 canonical 与完整 contributor-rights projection 计算并另行 hash。candidate↔pair identity equality 还必须覆盖 `design_relation_group_id/contributing assets+files+set hash/context_id/endpoint_id/label_unit/label_transform/delta_rule_id+sha256`；source/candidate observations 必须与 pair 的 endpoint、unit、transform 和 context compatibility rule一致，`delta` 必须由冻结 delta rule exact recomputation。多 endpoint/context pairs 是同一 sequence-edit relation group 的重复测量实例；Data Card 同时报告 unique design-relation groups 与 endpoint/context-specific pair instances，禁止以后者膨胀 E 规模。

### 5.5.1 RelationRoleTransition append-only 状态机

本状态机 **只适用于 `relation_acceptance_status=ACCEPTED` 且已一一物化 formal pair 的 candidate↔pair**。未接受的 `CANDIDATE|AMBIGUOUS|REJECTED` 对象没有 pair，不进入 role ledger；它们的 provisional `future_use_role` 不授权任何用途，最终处置只由 lifecycle status 与 `terminal_disposition_reason` 表达。如 provisional role 错误，保留旧对象并创建 parent-linked superseding candidate，禁止原地改写。accepted candidate/pair 首次物化时复制相同 `future_use_role` 作为 immutable base role；后续 effective role 只能由 §3.2 dual-store 的逻辑 `RELATION_ROLE_TRANSITIONS` ledger 重放得到，不得直接改写 candidate/pair base row。非 sealed objects 的唯一 shard 路径为 `data/v3_1/benchmark/RELATION_ROLE_TRANSITIONS.jsonl`；GSE246381 的唯一 shard 路径为 `<restricted_run_root>/sealed_external/GSE246381/benchmark/RELATION_ROLE_TRANSITIONS.jsonl`。同一 object 不能跨 shard 重复。每个事件最低字段为：

```yaml
transition_id: <stable-unique-id>
run_id: <run-id>
transaction_id: <stable-b0-transaction-id>
relation_candidate_id: <id>
pair_id: <id>
sequence_no: <positive-integer-contiguous-within-pair>
predecessor_transition_id: <id-or-GENESIS>
predecessor_event_sha256: <sha256-or-GENESIS>
from_role: <future-use-role>
to_role: <future-use-role>
reason: <controlled-reason>
evidence_id: <stable-id>
evidence_sha256: <sha256>
actor: <service-or-human-id>
timestamp: <iso-time>
code_commit: <full-sha>
config_hash: <sha256>
event_sha256: <sha256-over-rfc8785-event-excluding-this-field>
```

v1 唯一允许的 transition matrix 为：

```text
PENDING -> GENERAL_DEVELOPMENT_POOL
PENDING -> SEALED_EXTERNAL_FINAL_CANDIDATE
PENDING -> EXTERNAL_STRESS_ONLY
PENDING -> EXCLUDED
GENERAL_DEVELOPMENT_POOL -> EXCLUDED
SEALED_EXTERNAL_FINAL_CANDIDATE -> SEALED_EXTERNAL_FINAL
SEALED_EXTERNAL_FINAL_CANDIDATE -> EXCLUDED
EXTERNAL_STRESS_ONLY -> EXCLUDED
```

`SEALED_EXTERNAL_FINAL` 与 `EXCLUDED` 为 terminal role；没有出边。禁止 self-transition、跳过 candidate 直接进入 `SEALED_EXTERNAL_FINAL`、final→development/stress、同一 `sequence_no` 多事件、一个 predecessor 多后继或回写历史。具体 TRAIN/DEVELOPMENT/INTERNAL_TEST 只存在于 `TaskEligibilityCell.assigned_partition_id→SplitRegistry.partition role`，绝不属于本 global-role matrix，也不得回流改写 global role。需要矩阵外改变时，必须保留旧对象并建立 parent-linked superseding candidate/pair 与新 run，不能修改矩阵或旧 ledger 来迁就结果。

事件 hash 的 canonical bytes 冻结为 RFC 8785 JSON Canonicalization Scheme：事件先转为无 `event_sha256` member 的 JSON object，以 UTF-8、无 BOM、无外围空白执行 JCS；SHA256 只覆盖该 canonical object bytes，不覆盖 JSONL 行尾 LF。`timestamp` 必须为 RFC3339 UTC，`sequence_no` 为 JSON integer，其余 schema 禁止 NaN/Infinity。validator 必须提供并通过至少一个 GENESIS event 与一个 chained event 的固定 input/canonical-bytes/SHA256 golden vector；不同语言实现结果不一致即 FAIL。

role event 先写入 unique run staging，只有其 `transaction_id` 出现在唯一全局 root registry `data/v3_1/benchmark/B0_TRANSACTION_COMMITS.jsonl`、且 commit row 同时绑定 ordinary/restricted PREPARED manifests、restricted access-chain root 与 ordinary commitment 后，才成为 committed logical ledger 成员。该 registry 不含 object/member IDs，可安全留在 ordinary path；restricted shard 不另建第二份 commit registry。commit marker 必须最后 append。**外部/权威** loader、current projection、eligibility、Data Card 与 evaluator 在 marker 和两侧 hashes 可验证前一律忽略 staged event；唯一例外是同一 `transaction_id` 的 transaction-internal isolated builder，可只为构建和验证该 transaction 的 Stage3–6 读取 staged role projection，该 projection 只能存在于 staging root、必须标 `UNCOMMITTED_INTERNAL_ONLY`，不得被任何外部消费者视为当前状态。失败 attempt 原样保留为 `UNCOMMITTED_FAILED` evidence，不追加伪 rollback；下一 run 从 last committed projection 继续，未提交 attempt 的 `sequence_no` 不占用 committed sequence。跨文件系统发布遵循 §14.7 two-phase protocol，不能只提交一侧或假装单次 rename 原子覆盖两处。

effective-role projection 必须从冻结 base role 起，只对 committed events 按 `sequence_no` 严格递增，并逐事件验证 transaction commit、predecessor ID/hash、`from_role` 与上一状态、candidate/pair FK、allowed matrix 和 event hash。空 committed ledger 的 effective role 等于 base role。ordinary 与 restricted shard 的权威派生文件分别为同目录 `EFFECTIVE_ROLE_PROJECTION.jsonl`；每个 accepted current-leaf E pair×committed transaction snapshot 恰一 row，按 `relation_role_transition.schema.json#/$defs/EffectiveRoleProjection` 验证，最低字段为 `projection_record_id/transaction_id/run_id/relation_candidate_id/pair_id/base_future_use_role/effective_future_use_role/last_committed_transition_id/last_committed_transition_sha256/committed_transition_count/base_use_role_record_sha256/role_ledger_sha256/root_commit_record_sha256/projection_sha256`。eligibility、split、Data Card 与 evaluator manifest 只能在同一 store 内 join 该投影，不得跨 store 返回 sealed IDs，也不得读取人工更新的缓存字段。若为查询性能生成 candidate/pair current-role view，该 view 必须可由同一 ledger 确定性重建且双方一致；projection 不能反向改写 ledger。

正式验收要求：`duplicate_transition_id=0`；在 committed logical ledger 内 `duplicate_transition_sequence_no=0`、`missing_transition_predecessor=0`、`transition_predecessor_hash_mismatch=0`、`transition_from_role_projection_mismatch=0`、`invalid_role_transition=0`、`forked_role_transition=0`、`unlogged_base_role_mutation=0`、`candidate_pair_effective_role_mismatch=0`；并要求 `duplicate_effective_role_projection_key=0`、`missing_effective_role_projection=0`、`uncommitted_event_in_authoritative_projection=0`、`internal_staged_projection_escaped_count=0`、`committed_bundle_missing_component=0`、`transaction_component_hash_mismatch=0`、`ordinary_restricted_transaction_mismatch=0`。这些计数、raw-event ledger SHA256、commit-registry SHA256 与 committed projection SHA256 必须进入 B0/G7 manifest；任一非零即 FAIL。

### 5.5.2 Record-level ExposureRecord、base UseRole 与 rights projection

旧 `data/data_exposure_ledger.jsonl` 是 immutable historical input，只读保留并由 supersession manifest 标记，不得 replace 或就地“修正”。v3.1 权威 ordinary paths 为：

```text
data/v3_1/canonical/EXPOSURE_RECORDS.jsonl
data/v3_1/canonical/USE_ROLES.jsonl
data/v3_1/canonical/EXPOSURE_USE_MANIFEST.json
data/v3_1/canonical/EXPOSURE_USE_SHA256SUMS
```

GSE246381 对应 row 只存在于 `<restricted_run_root>/sealed_external/GSE246381/canonical/` 下的同名 files；ordinary files 对其 accession/object/source lineage count 必须为 0，只通过 §3.2 commitment 闭合。

`ExposureRecord` 对每个 in-scope `SequenceEntity` 以及进入 scientific canonical lifecycle 的 relation candidate、observation candidate、accepted pair 与 observation 使用稳定 `object_id/object_type` 一对一记录；accepted pair 必须引用并验证其 parent candidate 与两端 sequence exposure records，accepted observation 必须引用其 observation candidate 与 sequence exposure records。这样 standalone F/sequence 以及 ambiguous/unmatched/rejected label join 在尚未形成 accepted observation 前的 analytic access 也不会漏记。最低字段为：

```yaml
exposure_record_id: <stable-id>
object_id: <id>
object_type: <SEQUENCE|RELATION_CANDIDATE|OBSERVATION_CANDIDATE|PAIR|OBSERVATION>
project_sequence_analytic_exposure: <NONE_CONFIRMED|PRESENT|UNKNOWN>
project_sequence_analytic_use_types: [<controlled-use-type>]
project_label_analytic_exposure: <NONE_CONFIRMED|PRESENT|UNKNOWN>
project_label_analytic_use_types: [<controlled-use-type>]
pipeline_sequence_materialization: <ABSENT|PRESENT|UNKNOWN>
pipeline_label_materialization: <ABSENT|PRESENT|UNKNOWN>
foundation_overlap_requirement: <REQUIRED_FM0_A|NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS>
foundation_audit_scope_id: <stable-scope-id>
foundation_overlap_audit_status_at_baseline: <NOT_STARTED|DEFERRED_TO_FM0_A|NOT_APPLICABLE>
contributing_asset_ids: [<DatasetAsset.asset_id>]
contributing_file_sha256s: [<sha256>]
rights_evidence_ids: [<id>]
rights_projection_rule_id: <id>
rights_projection_rule_sha256: <sha256>
rights_override_id: <id-or-null>
rights_override_reviewer: <accountable-id-or-null>
rights_override_scope: <object-and-purpose-scope-or-null>
rights_override_evidence_ids: [<id>]
rights_override_sha256: <sha256-or-null>
permitted_model_training: <YES|NO|UNKNOWN>
permitted_evaluation: <YES|NO|UNKNOWN>
permitted_derived_release: <YES|NO|UNKNOWN>
permitted_raw_redistribution: <YES|NO|UNKNOWN>
evidence_ids: [<id>]
canonical_object_sha256: <sha256>
record_sha256: <sha256>
```

rights projection 默认对每个 purpose 在 **全部 provenance contributors** 上作 fail-closed conjunction：只有每个 contributing asset/file 的相应用途均有 `YES`+evidence 时才为 YES；任一 NO→NO，任一 UNKNOWN 且无 NO→UNKNOWN。一个宽松来源的 YES 不能覆盖另一个来源的 UNKNOWN/NO。例外字段必须 all-null/empty 或 all-present；非空时必须有受控 `rights_override_id`、accountable reviewer、明确法律/授权 evidence、适用 object/purpose 与 JCS hash，且不能靠 parser code 隐式实现。DatasetAsset contributor set、transformation edges 与 ExposureRecord contributor set 必须完全一致。

`ExposureRecord` 是 D1 时点的 immutable **baseline**，不允许在 FM0 或访问发生后原地改行。FM0 的 foundation truth 必须逐 checkpoint 写 `foundation_exposure_ledger.jsonl`，每 row 按 `exposure_record.schema.json#/$defs/FoundationExposureAuditRow` 验证，唯一键为 `object_or_cluster_id × checkpoint_id × checkpoint_revision × weights_sha256 × audit_run_id`，至少含 `sequence_overlap=<NO_DETECTED|DETECTED|UNKNOWN|NOT_APPLICABLE>/label_lineage_overlap=<NO_DETECTED|DETECTED|UNKNOWN|NOT_APPLICABLE>/audit_method_id/evidence_sha256/checkpoint_candidate_eligibility/controlled_reason/record_sha256`。ledger 同时绑定 frozen object/cluster set、checkpoint candidate set 与其 hashes。一个 checkpoint 的 DETECTED/UNKNOWN 只使该 checkpoint/相应 claim 不合格，不能全局丢弃数据或污染另一个 clean checkpoint；最终 alias 只能指向 overlap-clean eligible candidate set，若集合为空则使用 `from_scratch_E`，或仅在 F↔E final lineage audit clean时使用 `supervised_F_to_E` fallback。GSE246381 member-level overlap rows只在 restricted store，ordinary 仅接收按预注册 schema 的 checkpoint-level aggregate/commitment。

baseline ExposureRecord 与 append-only access chain 必须确定性生成 versioned `EFFECTIVE_EXPOSURE_PROJECTION.jsonl`；ordinary path 固定为 `data/v3_1/exposure/projections/<snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`，restricted path 为 `<restricted_run_root>/sealed_external/GSE246381/exposure/projections/<snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`，禁止覆盖前一 phase snapshot。每 row 按 `exposure_record.schema.json#/$defs/EffectiveExposureProjection` 验证，至少含 `object_id/object_type/projection_phase=<D1|FM0_A|B0_R|G7|MODEL_REBIND_OR_LATER>/snapshot_id/baseline_exposure_record_id/baseline_record_sha256/access_log_chain_root_sha256/foundation_exposure_ledger_manifest_sha256/as_of_event_id/effective_project_sequence_analytic_exposure/effective_project_sequence_use_types/effective_project_label_analytic_exposure/effective_project_label_use_types/final_access_status=<SEALED_UNOPENED|FINAL_ACCESS_RESERVED|FINAL_OPENED|FINAL_INVALIDATED>/projection_sha256`。`foundation_exposure_ledger_manifest_sha256` 在 D1 且 FM0 尚未产出时必须且只能为 `NOT_YET_PRODUCED`；FM0 起必须为 64-hex，只有真正无 external weights 的 per-checkpoint row 可用 `NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS`，不能用 sentinel 躲避未知 corpus。ordinary 与 restricted 各自投影，sealed member rows绝不跨 store；restricted ordinary commitment只暴露聚合状态与投影 hash。D1 可在空链/机器 QC 链上得到与 baseline 相同的 analytic axes；B0/G7/finalizer 必须绑定并读取 projection，禁止直接把 baseline 当“当前暴露”。

ordinary live analytic-access authority 固定为 `data/v3_1/exposure/ORDINARY_ACCESS_LOG.jsonl`；restricted live authority 固定为 §3.2 sealed root 下的 `ACCESS_LOG.jsonl`。两者使用同一 intent/completion/abort hash-chain schema，并都必须按 §3.2 的 prefix-snapshot 规则在每个 phase 冻结不可变三件套：ordinary 位于 `data/v3_1/exposure/access_snapshots/<snapshot_id>/ORDINARY_ACCESS_LOG.jsonl|ORDINARY_ACCESS_MANIFEST.json|ORDINARY_ACCESS_SHA256SUMS`，restricted 位于 `<sealed_cohort_root>/access_snapshots/<restricted_snapshot_id>/ACCESS_LOG.jsonl|ACCESS_MANIFEST.json|ACCESS_SHA256SUMS`；普通文件名不同但 manifest 字段、自哈希、non-cyclic checksum、live-prefix 与 immutability 规则完全相同。ordinary log 禁止出现任何 sealed cohort object-set hash。D1 ordinary 没有 analytic event 时必须保存可验证 GENESIS/empty-chain sentinel 及 D1 immutable prefix snapshot；B0 的 `TASK_PROTOCOL_CALIBRATION` 必须记录 requested/actual ordinary object-set manifests，并在 EffectiveExposureProjection 中追加 use type=`PROTOCOL_CALIBRATION`。普通 analytic event 不能借“公开数据”豁免日志。强制 `effective_exposure_projection_missing=0`、`effective_exposure_projection_hash_mismatch=0`、`baseline_exposure_row_mutation_count=0`、`foundation_checkpoint_key_collision=0`、`foundation_candidate_scope_mismatch=0`、`ineligible_checkpoint_selected_count=0`、`final_alias_outside_eligible_checkpoint_set=0`、`ordinary_access_log_contains_sealed_scope=0`、`protocol_calibration_access_unlogged=0`、`access_prefix_snapshot_byte_drift_count=0`、`access_live_prefix_mismatch_count=0`。

`UseRole` 只适用于 accepted E candidate↔pair，一 pair 一 row；它是 immutable base-role integrity sidecar，不是第二个可写 authority：

```yaml
use_role_record_id: <stable-id>
relation_candidate_id: <id>
pair_id: <id>
base_future_use_role: <global-role-enum>
candidate_base_payload_sha256: <sha256>
pair_base_payload_sha256: <sha256>
canonical_manifest_sha256: <sha256>
record_sha256: <sha256>
```

candidate、pair 与 UseRole 的 base role 必须逐字段相等；F observation 不创建 UseRole。current role 仍只来自 committed transition projection。v1 base role 永远不得直接为 `SEALED_EXTERNAL_FINAL`；只有 frozen GSE cohort 可从 `SEALED_EXTERNAL_FINAL_CANDIDATE` 起步，并必须经 committed candidate→final event 后才产生 effective final。强制 `duplicate_exposure_record=0`、`in_scope_sequence_without_exposure_record=0`、`canonical_scientific_object_without_exposure_record=0`、`observation_candidate_without_exposure_record=0`、`exposure_record_without_object=0`、`scientific_object_sequence_exposure_fk_mismatch=0`、`observation_candidate_observation_exposure_mismatch=0`、`candidate_pair_exposure_mismatch=0`、`accepted_pair_without_use_role=0`、`observation_with_use_role=0`、`use_role_base_payload_mismatch=0`、`base_role_sealed_external_final_count=0`、`effective_sealed_final_without_committed_transition=0`、`missing_rights_contributor=0`、`rights_evidence_set_mismatch=0`、`rights_projection_mismatch=0`、`rights_override_field_partial_count=0`、`unauthorized_mixed_rights_override=0`。D1/FM0/B0 每次更新只能通过 parent-linked new run/versioned baseline/ledger/projection 与 manifests 体现，不能改写历史 bytes。

### 5.5.3 Supersession ledger 与 current-leaf projection

“创建 parent-linked superseding object”必须是机器可执行状态，不得只留 prose。relation/observation candidates 的 parent 字段只记录 proposed lineage；真正使旧 object 退出 active universe 的事件必须另写 `data/v3_1/canonical/SUPERSESSION_EDGES.jsonl`（GSE 使用 restricted mirror），每行按 `transformation_edge.schema.json#/$defs/SupersessionEdge` 验证：

```yaml
supersession_edge_id: <stable-id>
object_type: <SEQUENCE|RELATION_CANDIDATE|PAIR|OBSERVATION_CANDIDATE|OBSERVATION>
old_object_id: <id>
new_object_id: <id>
old_object_sha256: <sha256>
new_object_sha256: <sha256>
reason: <controlled-repair-reason>
run_id: <run-id>
code_commit: <full-sha>
config_hash: <sha256>
edge_sha256: <rfc8785-sha256>
```

edge 只有在 new object 已通过本层 schema/lineage/payload checks 后才进入 canonical manifest；失败 repair candidate 可保留 parent ID，但没有 committed supersession edge，不能停用旧 leaf。图必须无环、每个 old object 最多一个 committed successor、每条 chain 恰有一个 current leaf；不允许 fork 后由执行者择优。SequenceEntity 被替换时，所有 current downstream candidate/pair/observation references 必须在同 generation 指向 current sequence leaf；accepted relation candidate 被替换时，其 accepted pair 必须在同 run 以对应 pair edge 被替换，candidate/pair generation index 与 endpoints/base payload保持一致；observation candidate 与 accepted observation 同理。

从 base objects+committed edges 确定性生成 `data/v3_1/canonical/CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl`（restricted mirror 同理）。cardinality 固定为每个 lifecycle chain root×canonical snapshot 恰一 row，而不是每个历史 object 一 row；singleton chain 也必须有 row。每行按 `transformation_edge.schema.json#/$defs/CurrentCanonicalObjectProjection` 验证，最低字段为 `projection_record_id/run_id/canonical_snapshot_id/object_type/chain_root_object_id/chain_root_object_sha256/current_leaf_object_id/current_leaf_object_sha256/generation_index/chain_length/last_supersession_edge_id/last_supersession_edge_sha256/supersession_manifest_sha256/is_current_leaf_accepted/projection_sha256`。immutable ExposureRecord/UseRole 仍覆盖并保留每个历史 lifecycle object；另由 current projection 确定性选择其 current-leaf exposure/use view，只有该 view 可供 active loader、global eligibility 和 denominator 使用。superseded accepted rows及其历史 exposure/use bytes/edge 保留但不得双计。强制 `supersession_cycle_count=0`、`supersession_fork_count=0`、`supersession_missing_object=0`、`supersession_hash_mismatch=0`、`duplicate_current_projection_root=0`、`missing_singleton_current_projection=0`、`multiple_active_leaf_count=0`、`current_downstream_reference_to_superseded_sequence=0`、`candidate_pair_generation_mismatch=0`、`observation_generation_mismatch=0`、`current_exposure_projection_mismatch=0`、`superseded_object_in_active_loader=0`、`superseded_object_in_global_denominator=0`；edge/projection hashes 必须进入 D1/B0/G7 manifests。

`E_DELTA`、`E_LINK`、`E_DENSE`、`E_NOEDIT`、`E_EXTERNAL_FINAL` 与 `E_EXTERNAL_STRESS` 必须由上述正交字段生成只读 derived views，不能回填成一个互斥枚举。任何 `permitted_model_training/evaluation/derived_release/raw_redistribution=UNKNOWN|NO` 的上游 contributor 不得靠 canonical 技术 PASS 自动进入相应用途；eligibility 必须按每个 purpose fail closed。

source 或 candidate 为空的记录不能进入 paired canonical。source==candidate 必须保留为 `NO_EDIT_CONTROL`，只从要求实际 edit event 的特定 metric 分母中排除。

## 5.6 EditPathSet 与 GenerationTask

`EditPathSet` 必须记录 endpoint pair、合法动作字典、路径成本、canonical replay path、多路径生成策略、采样 seed、路径数和 estimator 类型。canonical path 仅用于确定性重放，不得标注 `observed_trajectory=true`。

`GenerationTask` 必须记录：source、region、sequence scope、editable mask、context、endpoint、edit budget、length policy、hard constraints、soft preferences、candidate/query budget、predictor-call budget、stopping policy 与 evaluator version。

`GroupRegistry/GroupAssignment` 必须记录 `group_id/group_type/grouping_method/method_version/thresholds/source_evidence/member_count/ambiguous_membership/parent_group_id` 与逐成员 assignment。`group_type` 受控枚举至少为 `SAMPLE|BIOLOGICAL_REPLICATE|TECHNICAL_REPLICATE|BARCODE|BIOLOGICAL_PARENT|GENE|TRANSCRIPT|TILE_FAMILY|SEQUENCE_CLUSTER|LIBRARY_LINEAGE|EXPERIMENTAL_CONTEXT`。`EXPERIMENTAL_CONTEXT` row 另强制 `raw_context_values/context_components=(cell_type,assay,promoter,reporter_or_cargo,rna_chemistry,timepoint,other)/ontology_ids/ontology_version/mapping_status=<RESOLVED|UNKNOWN|AMBIGUOUS|NOT_APPLICABLE>/mapping_rule_id/mapping_rule_sha256`；raw value 到多个 canonical IDs 或多个不等价 raw values被无证据合并均 FAIL。每个 observation 必须通过 GroupAssignment 与其非空 `sample_id/biological_replicate_id/technical_replicate_id/barcode_id/context_id` 逐一一致；source-native replicate/context label只作 provenance，不得代替类型化 ID。强制测试 biological 与 technical ID 不被同一未解释字符串折叠、parent-child hierarchy 无环、成员计数与 observation 外键一致、context merge/split collision=0。independent parent、gene/tile family、sequence cluster 和 library lineage 均不得只有无来源的字符串 ID。`ReporterArtifactAssessment` 承载 §6.7 字段，PTRE 等 AUX_QC 资产进入该表而不是被迫进入 E/F 主 canonical。

## 5.7 Task/Split registries、global eligibility 与 task cells

### 5.7.1 TaskRegistry 与 v1 外部期望集合

`TaskRegistry` 每个正式任务至少记录：

```yaml
task_id: <stable-id>
task_version: <version>
task_kind: <BENCHMARK_EVALUATION|AUXILIARY_TRAINING>
object_type: <PAIR|OBSERVATION>
scientific_track: <E|F>
region_scope: <5UTR|3UTR|CROSS_REGION|MULTI_REGION>
species_scope_policy_id: <WITHIN_SPECIES_STRATIFIED_V1|other-preapproved-policy>
analysis_unit_id: <stable-id>
analysis_unit_dedup_key_fields: [<field>]
estimand_id: <stable-id>
activation_rule_id: <stable-id>
activation_rule_sha256: <sha256>
required_fields: [<field>]
label_validity_rule_id: <id>
eligibility_rule_id: <id>
eligibility_rule_sha256: <sha256>
candidate_primary_metric_ids: [<id-or-NOT_APPLICABLE_AUXILIARY>]
primary_metric_selection_rule_id: <id>
primary_metric_selection_rule_sha256: <sha256>
training_objective_id: <id-or-NOT_APPLICABLE_EVALUATION>
monitoring_metric_id: <id>
evaluator_output_schema_sha256: <sha256-or-NOT_APPLICABLE_AUXILIARY>
confirmatory_policy_id: <id>
confirmatory_policy_sha256: <sha256>
```

C3 的 `TaskRegistry` 是 immutable **definition registry**，不得在 D0/D1 之前预判任务 ACTIVE/N/A，也不保存结果依赖的最终 metric 分支。`BENCHMARK_EVALUATION` 必须冻结候选 primary metrics、唯一 selection rule 与 evaluator schema；允许 descriptive fallback 的 task 必须把 `DESCRIPTIVE_NO_CONFIRMATORY_PRIMARY` 明确列入 frozen candidate set。`AUXILIARY_TRAINING` 必须冻结训练 objective/monitoring metric，但不能产生 edit-pair benchmark headline。B0 对每个 required task 另生成 append-only、hash-bound `TASK_ACTIVATION_DECISIONS.jsonl`，最低字段为 `task_id/decision_run_id/task_definition_sha256/task_activation_status/activation_reason/activation_input_manifest_sha256/activation_calibration_mask_sha256/activation_calibration_population_sha256/sealed_contribution_count/internal_test_contribution_count/selected_primary_metric_id/confirmatory_status/ordinary_access_event_chain_root_sha256/decision_sha256`；其中 `task_activation_status=<ACTIVE|NOT_APPLICABLE_DATA_GATE>`，`confirmatory_status=<CONFIRMATORY|SECONDARY|EXPLORATORY|DESCRIPTIVE|NOT_APPLICABLE_AUXILIARY|NOT_APPLICABLE_DATA_GATE>`。schema 必须用 `oneOf` 锁定：ACTIVE benchmark 的 metric 必须属于 frozen candidate set且不能用 N/A sentinel；N/A task 必须写 `selected_primary_metric_id=NOT_APPLICABLE_DATA_GATE` 与同名 confirmatory sentinel；ACTIVE auxiliary 必须写 `selected_primary_metric_id=NOT_APPLICABLE_AUXILIARY` 与同名 confirmatory sentinel。任务 activation、metric branch 与 confirmatory status 只能重放 C3 rule并使用 frozen **ordinary nonsealed activation-calibration** population；`sealed_contribution_count=0`、`internal_test_contribution_count=0`，不能读取 restricted cohort count/distribution、model 或 final 结果。sealed aggregate 只允许进入 `sealed_final_v1` split readiness decision，不能影响 task/metric选择。条件任务门槛不满足时保留 definition row，在 decision 写 N/A+reason，禁止删 row、换 object type/track/estimand 或回写 TaskRegistry。T5-Gen exact/similarity/descriptive 分支必须在 B0 以冻结非退化 rule决定，PR1 只能绑定该决定，不能重新选择。

为避免 Stage4 在 split 之前偷看未来 INTERNAL_TEST，C3 必须冻结 outcome-blind `activation_calibration_mask_rule_id/hash`。canonical rule bytes 精确为 `ACTIVATION_CALIBRATION_MASK_V1\nELIGIBLE_SCOPE=ORDINARY_NONSEALED_CURRENT_LEAF_TECHNICAL_ACCEPTED\nCOMPONENT_ATOMS=BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,SEQUENCE_CLUSTER,STUDY,TILE_FAMILY\nCOMPONENT_ID=SHA256_SORTED_MEMBER_IDS\nSELECT=UINT64_BE(SHA256(UTR_EDITFLOW_V3_1_CALIBRATION|COMPONENT_ID)[0:8])%5==0\nCALIBRATION_PARTITION=DEVELOPMENT_ONLY\nOUTCOME_BLIND=true\n`，SHA256=`b2652abda7a2dbb7001e7fb655db9b6ac19f2b8f80fbc65362dc1236fd9781e9`。在 ordinary、nonsealed、current-leaf、技术合格 objects 上，按任一共享 atom 构成 group-connected components，再用上述 fixed salt/rule选出 calibration components；构造 mask 只能读取 IDs/group membership，不能读取 outcome。B0 在任何 label summary 前写并 hash `ACTIVATION_CALIBRATION_MASK.jsonl`，随后对该 mask 的 label access 以 ordinary append-only `TASK_PROTOCOL_CALIBRATION` intent/completion记录到 `data/v3_1/exposure/ORDINARY_ACCESS_LOG.jsonl`，Exposure projection追加 `PROTOCOL_CALIBRATION`。被选 components 在 Stage5 对所有 task/split 只能进入 DEVELOPMENT 或 INELIGIBLE，绝不能进入 TRAIN/INTERNAL_TEST/SEALED_FINAL；未选 components不能被 Stage4读取。若 hash选择导致某 task calibration power不足，按 frozen activation rule降级/N/A，不能看 outcome后换 salt。强制 `activation_calibration_rule_hash_mismatch=0`、`activation_calibration_mask_outcome_access_before_freeze=0`、`activation_calibration_group_overlap_with_internal_test_or_sealed=0`、`activation_calibration_object_assigned_train=0`、`sealed_contribution_to_task_activation_or_metric_selection=0`、`internal_test_contribution_to_task_activation_or_metric_selection=0`。

v1 `REQUIRED_TASK_IDS_V1` 精确冻结为下列 12 个原子 task IDs；UTF-8、字典序、每 ID 后一个 LF 的 ID-set SHA256 为 `b0b43cb76f39b32009e3a6ef8ae6d05395d61bf7baa7480743587e6772447207`：

```text
CROSS_REGION_PROPERTY_F_OBSERVATION
CROSS_REGION_RECONSTRUCT_E_PAIR
F3_OUTCOME_AUX_OBSERVATION
F5_OUTCOME_AUX_OBSERVATION
T3_EFFECT_DELTA_E_PAIR
T3_PROPERTY_E_PAIR
T3_RANK_EXPLORATORY_E_PAIR
T3_RECONSTRUCT_E_PAIR
T5_CONTEXT_E_PAIR
T5_CONTEXT_F_OBSERVATION
T5_GEN_RECONSTRUCT_E_PAIR
T5_RANK_CLOSED_SELECT_E_PAIR
```

task 的 core semantic descriptor 外部映射必须与下表逐行完全一致；canonical descriptor line 为 `task_id|task_kind|object_type|scientific_track|region_scope|estimand_id|activation_rule_id|analysis_unit_id|species_scope_policy_id`，按整行 UTF-8 字典序、行尾 LF 的 descriptor-set SHA256 为 `8f42ef044d8de1a26b9b587587c2de99c6068f67f37e269e226e143333245ba3`：

| task_id | task_kind | object_type | track | region_scope | estimand_id | activation_rule_id | analysis_unit_id | species_scope_policy_id |
|---|---|---|---|---|---|---|---|---|
| CROSS_REGION_PROPERTY_F_OBSERVATION | BENCHMARK_EVALUATION | OBSERVATION | F | CROSS_REGION | CROSS_REGION_ENDPOINT_PROPERTY_TRANSFER | CONDITIONAL_COMMON_SUPPORT | ENDPOINT_CONTEXT_OBSERVATION | WITHIN_SPECIES_STRATIFIED_V1 |
| CROSS_REGION_RECONSTRUCT_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | CROSS_REGION | CROSS_REGION_ENDPOINT_RECONSTRUCTION | CONDITIONAL_COMMON_SUPPORT | UNIQUE_SEQUENCE_EDIT_RELATION | WITHIN_SPECIES_STRATIFIED_V1 |
| F3_OUTCOME_AUX_OBSERVATION | AUXILIARY_TRAINING | OBSERVATION | F | 3UTR | SUPERVISED_ENDPOINT_OUTCOME_AUXILIARY | ALWAYS_ACTIVE | ENDPOINT_CONTEXT_OBSERVATION | WITHIN_SPECIES_STRATIFIED_V1 |
| F5_OUTCOME_AUX_OBSERVATION | AUXILIARY_TRAINING | OBSERVATION | F | 5UTR | SUPERVISED_ENDPOINT_OUTCOME_AUXILIARY | ALWAYS_ACTIVE | ENDPOINT_CONTEXT_OBSERVATION | WITHIN_SPECIES_STRATIFIED_V1 |
| T3_EFFECT_DELTA_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 3UTR | WITHIN_ASSAY_NORMALIZED_DELTA | CONDITIONAL_DELTA_JOIN_GATE | ENDPOINT_CONTEXT_PAIR_INSTANCE | WITHIN_SPECIES_STRATIFIED_V1 |
| T3_PROPERTY_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 3UTR | CANDIDATE_ENDPOINT_PROPERTY | ALWAYS_ACTIVE | ENDPOINT_CONTEXT_PAIR_INSTANCE | WITHIN_SPECIES_STRATIFIED_V1 |
| T3_RANK_EXPLORATORY_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 3UTR | MEASURED_POOL_SOURCE_MACRO_RANKING | ACTIVE_EXPLORATORY_IF_NONEMPTY | ENDPOINT_CONTEXT_PAIR_INSTANCE | WITHIN_SPECIES_STRATIFIED_V1 |
| T3_RECONSTRUCT_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 3UTR | HELDOUT_ENDPOINT_RECONSTRUCTION | ALWAYS_ACTIVE | UNIQUE_SEQUENCE_EDIT_RELATION | WITHIN_SPECIES_STRATIFIED_V1 |
| T5_CONTEXT_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 5UTR | PAIR_CONTEXT_TRANSFER | CONDITIONAL_REPEATED_CONTEXT_GATE | REPEATED_CONTEXT_RELATION_GROUP | WITHIN_SPECIES_STRATIFIED_V1 |
| T5_CONTEXT_F_OBSERVATION | BENCHMARK_EVALUATION | OBSERVATION | F | 5UTR | OBSERVATION_CONTEXT_TRANSFER | CONDITIONAL_REPEATED_CONTEXT_GATE | REPEATED_CONTEXT_OBSERVATION_GROUP | WITHIN_SPECIES_STRATIFIED_V1 |
| T5_GEN_RECONSTRUCT_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 5UTR | HELDOUT_ENDPOINT_RECONSTRUCTION | ALWAYS_ACTIVE | UNIQUE_SEQUENCE_EDIT_RELATION | WITHIN_SPECIES_STRATIFIED_V1 |
| T5_RANK_CLOSED_SELECT_E_PAIR | BENCHMARK_EVALUATION | PAIR | E | 5UTR | MEASURED_POOL_SOURCE_MACRO_RANKING | CONDITIONAL_MULTI_CANDIDATE_GATE | ENDPOINT_CONTEXT_PAIR_INSTANCE | WITHIN_SPECIES_STRATIFIED_V1 |

T3 delta 与 absolute-property、T5 context 的 E/F branch 必须使用不同 IDs，不能在 B0 根据结果让同一 task 换 estimand。`UNIQUE_SEQUENCE_EDIT_RELATION` 的 dedup key 固定为 `design_relation_group_id/source_sequence_id/candidate_sequence_id`，同一 edit 的多 endpoint/context pair instances 只计一次 reconstruction unit；`ENDPOINT_CONTEXT_PAIR_INSTANCE` 使用 `pair_id/endpoint_id/context_id`；`REPEATED_CONTEXT_RELATION_GROUP` 使用 `design_relation_group_id/endpoint_id`；`REPEATED_CONTEXT_OBSERVATION_GROUP` 使用 `sequence_id/endpoint_id`；`ENDPOINT_CONTEXT_OBSERVATION` 使用 `observation_id/endpoint_id/context_id`。所有推断使用 analysis-unit/group-aware CI，不得把 endpoint/context 重复测量当独立 edits。`WITHIN_SPECIES_STRATIFIED_V1` 禁止跨 species 合池作为一个独立样本群；跨 species transfer 必须以后续 contract 注册独立 task/estimand。`F5_OUTCOME_AUX_OBSERVATION/F3_OUTCOME_AUX_OBSERVATION` 为 supervised F auxiliary/scorer 任务，只允许按许可与 split 使用功能 observations，不得称 edit pairs 或方法 headline。

open-generation diagnostics 不进入上述 TaskRegistry 或 E/F `TaskEligibilityCell`，因为其单位是 `GenerationTask`/unique source 而不是 held-out pair。它必须单独冻结在 `docs/execution/diagnostic_registry_v3_1.yaml`，v1 唯一 ID 为 `OPEN_GENERATION_DIAGNOSTIC_E_GENERATION_TASK`；按该 ID 加 LF 的 expected-set SHA256 固定为 `f25c0adc643f38ff26c5e08bf07e4175a4e2571eaae939d61daa91fc6f2aabb2`。每个 registry row 必须按 `generation_task.schema.json#/$defs/DiagnosticRegistryRow` 验证，并冻结 denominator rule 与 claim policy；具体 generation instances 再按该 schema 主体验证。它只报告 §9.4 diagnostics，不产生 benchmark primary claim。C3 必须为该 `$defs` 与 instance schema 各有 positive/negative fixture；`diagnostic_registry_expected_set_mismatch=0`、`diagnostic_registry_expected_set_hash_mismatch=0`、`diagnostic_object_type_mismatch=0`。

### 5.7.2 SplitRegistry、partition 与外部期望集合

`SplitRegistry` 每个 split contract 至少记录：

```yaml
split_contract_id: <stable-id>
split_version: <version>
region_scope: <5UTR|3UTR|CROSS_REGION|MULTI_REGION>
object_scope: <PAIR|OBSERVATION|PAIR_OR_OBSERVATION>
activation_rule_id: <stable-id>
activation_rule_sha256: <sha256>
direction_or_cohort_rule_id: <stable-id-or-NOT_APPLICABLE>
direction_or_cohort_rule_sha256: <sha256-or-NOT_APPLICABLE>
partitions:
  - partition_id: <globally-unique-stable-id>
    partition_role: <TRAIN|DEVELOPMENT|INTERNAL_TEST|SEALED_FINAL|STRESS_ONLY>
grouping_atoms: [<SOURCE|PAIR|BIOLOGICAL_PARENT|GENE|TRANSCRIPT|TILE_FAMILY|SEQUENCE_CLUSTER|LIBRARY_LINEAGE|STUDY|CONTEXT>]
grouping_atoms_by_object_type:
  PAIR: [<REQUIRED atom|NOT_APPLICABLE_SCOPE>]
  OBSERVATION: [<REQUIRED atom|NOT_APPLICABLE_SCOPE>]
grouping_atom_projection_rule_sha256: <sha256>
disjointness_constraints: [<constraint>]
stratification_fields: [<field>]
assignment_algorithm_id: <id>
assignment_algorithm_sha256: <sha256>
seed_or_fold_definition: <value>
sealed_final: <bool>
```

C3 的 `SplitRegistry` 同样是 immutable definition registry，不含数据依赖的 active/N/A 结果。`partition_id` 在整个 registry 内唯一，并确定性写作 `<split_contract_id>::<lowercase-partition-role>`；每个 descriptor 指定的 partition role 恰有一个 partition，不能缺失、重复或额外增加。对 `PAIR_OR_OBSERVATION`，union-level `grouping_atoms` 不能强迫 F observation 伪造 PAIR/SOURCE/BIOLOGICAL_PARENT。`grouping_atoms_by_object_type` 必须按如下冻结规则逐 atom 标 REQUIRED/N/A：PAIR 要求全部 listed atoms；OBSERVATION 对 `PAIR|SOURCE|BIOLOGICAL_PARENT` 写 `NOT_APPLICABLE_SCOPE`，对其余 listed atoms写 REQUIRED；required atom 缺失使相应 task cell=`INELIGIBLE_WITH_REASON` 且不产生 assignment，禁止生成虚构 sentinel group ID。canonical rule bytes 固定为 `PAIR_OR_OBSERVATION_GROUPING_ATOMS_V1\nPAIR:REQUIRE_ALL_LISTED\nOBSERVATION:PAIR=NOT_APPLICABLE;SOURCE=NOT_APPLICABLE;BIOLOGICAL_PARENT=NOT_APPLICABLE;REQUIRE_OTHER_LISTED\nMISSING_REQUIRED_ATOM=TASK_CELL_INELIGIBLE_NO_ASSIGNMENT\nINVENTED_ATOM_FORBIDDEN=true\n`，SHA256=`bd8395ab0ec23d98d7c1b717e7fcb0bdd3df6d18002985624cd9eb41f8bd7983`。单一 object scope 的适用类型全 listed atoms REQUIRED、另一类型全部 `NOT_APPLICABLE_SCOPE`。强制 `grouping_atom_projection_rule_hash_mismatch=0`、`missing_required_grouping_atom=0`（assigned cells中）、`invented_not_applicable_group_id=0`。B0 对 10 个 split 另生成 append-only、hash-bound `SPLIT_ACTIVATION_DECISIONS.jsonl`，字段至少为 `split_contract_id/decision_run_id/split_definition_sha256/split_activation_status/activation_reason/activation_input_manifest_sha256/decision_sha256`，其中 status=`ACTIVE|CONDITIONAL_NOT_QUALIFIED`。`heldout_context` 即使数据门槛失败也必须保留 definition row，只在 B0 decision 中写 `CONDITIONAL_NOT_QUALIFIED`，不得删除或回写 SplitRegistry。v1 `REQUIRED_SPLIT_CONTRACT_IDS_V1` 精确冻结为下列 10 个 IDs；同样按 UTF-8 字典序、每 ID 后 LF 的 set SHA256 为 `b8c6fb2718875862da500c949481d04db08d1d21f94e3d13da49e3ace64ff487`：

```text
3utr_sequence_cluster_disjoint
3utr_source_or_variant_disjoint
3utr_study_disjoint
5utr_sequence_cluster_disjoint
5utr_source_disjoint
5utr_study_disjoint
cross_region_3_to_5
cross_region_5_to_3
heldout_context
sealed_final_v1
```

split core semantic descriptor 必须逐行等于下表。`partition_roles` 与 `grouping_atoms` 均按字典序用逗号无空格连接；canonical line 为 `split_contract_id|activation_rule_id|region_scope|object_scope|direction_or_cohort_rule_id|direction_or_cohort_rule_sha256|partition_roles|grouping_atoms|sealed_final`，按 UTF-8 字典序、行尾 LF 的 descriptor-set SHA256 为 `c8a6c82a9a1ab687ef2c3cb912ed96aae26c73a0662b0ae0911040c37e8ef1fa`：

| split_contract_id | activation_rule_id | region_scope | object_scope | direction_or_cohort_rule_id | direction_or_cohort_rule_sha256 | partition_roles | grouping_atoms | sealed_final |
|---|---|---|---|---|---|---|---|---|
| 3utr_sequence_cluster_disjoint | ALWAYS_ACTIVE | 3UTR | OBSERVATION | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | GENE,LIBRARY_LINEAGE,SEQUENCE_CLUSTER,TILE_FAMILY,TRANSCRIPT | false |
| 3utr_source_or_variant_disjoint | ALWAYS_ACTIVE | 3UTR | PAIR | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,TILE_FAMILY,TRANSCRIPT | false |
| 3utr_study_disjoint | ALWAYS_ACTIVE | 3UTR | PAIR_OR_OBSERVATION | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | false |
| 5utr_sequence_cluster_disjoint | ALWAYS_ACTIVE | 5UTR | OBSERVATION | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | GENE,LIBRARY_LINEAGE,SEQUENCE_CLUSTER,TILE_FAMILY,TRANSCRIPT | false |
| 5utr_source_disjoint | ALWAYS_ACTIVE | 5UTR | PAIR | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,TILE_FAMILY,TRANSCRIPT | false |
| 5utr_study_disjoint | ALWAYS_ACTIVE | 5UTR | PAIR_OR_OBSERVATION | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | false |
| cross_region_3_to_5 | CONDITIONAL_COMMON_SUPPORT | CROSS_REGION | PAIR_OR_OBSERVATION | TRAIN_3UTR_EVALUATE_5UTR | 4a2fc1e997856fa33348ed9adb9e101271ef8bdc61206db22f14b40cdf908c70 | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | false |
| cross_region_5_to_3 | CONDITIONAL_COMMON_SUPPORT | CROSS_REGION | PAIR_OR_OBSERVATION | TRAIN_5UTR_EVALUATE_3UTR | 789cca607b17a30dcea362ade94f74b60037af59ae58e05598f9552cc19c7b16 | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | false |
| heldout_context | CONDITIONAL_CONTEXT_GATE | MULTI_REGION | PAIR_OR_OBSERVATION | NOT_APPLICABLE | NOT_APPLICABLE | DEVELOPMENT,INTERNAL_TEST,TRAIN | BIOLOGICAL_PARENT,CONTEXT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | false |
| sealed_final_v1 | CONDITIONAL_SEALED_COHORT_GATE | 5UTR | PAIR | SEALED_COHORT_IDS_V1_SHA256_275774A99CBE46CD | f0ced6dc8869b040f1197b519403691bd97f07e59906c3c82434606a9861262a | SEALED_FINAL | BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,PAIR,SEQUENCE_CLUSTER,SOURCE,STUDY,TILE_FAMILY,TRANSCRIPT | true |

上述三条非 N/A rule 的 canonical rule bytes 分别固定为 `TRAIN_REGION=3UTR;DEVELOPMENT_REGION=5UTR;INTERNAL_TEST_REGION=5UTR\n`、`TRAIN_REGION=5UTR;DEVELOPMENT_REGION=3UTR;INTERNAL_TEST_REGION=3UTR\n` 与 `SEALED_FINAL_COHORT_SET_SHA256=275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0;OTHER_PARTITIONS_FORBIDDEN=true\n`；hash 必须匹配上表。强制 `direction_rule_hash_mismatch=0`、`directional_partition_region_mismatch=0`、`sealed_cohort_rule_mismatch=0`。

### 5.7.3 C3 Task×Split definition matrix 与 B0 applicability decisions

每个 task 允许关联的 split-contract set 精确冻结如下；按 `task_id|comma-joined-sorted-split-ids`、UTF-8 字典序、行尾 LF 的 allowlist SHA256 为 `02b25e4717e4a7192b658d5e69cdbb198e5b696b3ea520b7a0a887fcf89097ab`：

| task_id | contract-allowed split_contract_ids |
|---|---|
| CROSS_REGION_PROPERTY_F_OBSERVATION | cross_region_3_to_5, cross_region_5_to_3 |
| CROSS_REGION_RECONSTRUCT_E_PAIR | cross_region_3_to_5, cross_region_5_to_3 |
| F3_OUTCOME_AUX_OBSERVATION | 3utr_sequence_cluster_disjoint, 3utr_study_disjoint |
| F5_OUTCOME_AUX_OBSERVATION | 5utr_sequence_cluster_disjoint, 5utr_study_disjoint |
| T3_EFFECT_DELTA_E_PAIR | 3utr_source_or_variant_disjoint, 3utr_study_disjoint |
| T3_PROPERTY_E_PAIR | 3utr_source_or_variant_disjoint, 3utr_study_disjoint |
| T3_RANK_EXPLORATORY_E_PAIR | 3utr_source_or_variant_disjoint, 3utr_study_disjoint |
| T3_RECONSTRUCT_E_PAIR | 3utr_source_or_variant_disjoint, 3utr_study_disjoint |
| T5_CONTEXT_E_PAIR | heldout_context |
| T5_CONTEXT_F_OBSERVATION | heldout_context |
| T5_GEN_RECONSTRUCT_E_PAIR | 5utr_source_disjoint, 5utr_study_disjoint, sealed_final_v1 |
| T5_RANK_CLOSED_SELECT_E_PAIR | 5utr_source_disjoint, 5utr_study_disjoint, sealed_final_v1 |

在 C3 冻结 definition-only `docs/execution/task_split_contract_matrix_v3_1.yaml`。每 row 至少记录：

```yaml
task_id: <TaskRegistry.task_id>
split_contract_id: <SplitRegistry.split_contract_id>
contract_mapping: <ALLOWED|NOT_ALLOWED>
object_type: <PAIR|OBSERVATION>
scientific_track: <E|F>
definition_reason: <controlled-reason>
```

definition matrix 必须对外部 12-task×10-split 笛卡尔积的 **精确 key set** 恰有 120 rows，`contract_mapping` 与上述 allowlist 一致；C3 不写数据依赖的 applicability。B0 在全局普通路径 `data/v3_1/benchmark/TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl` 逐 definition row 生成一条不含 object/member IDs 的 effective decision；它只有一份，绝不按 sealed cohort 分 shard：

```yaml
task_id: <id>
split_contract_id: <id>
definition_matrix_sha256: <sha256>
task_activation_decision_sha256: <sha256>
split_activation_decision_sha256: <sha256>
applicability: <APPLICABLE|NOT_APPLICABLE>
reason: <controlled-reason>
decision_run_id: <run-id>
decision_sha256: <sha256>
```

task decision=`ACTIVE`、split decision=`ACTIVE` 且 contract mapping=`ALLOWED` 时必须 `APPLICABLE`，即使实际对象数为 0；task N/A 或 split conditional-not-qualified 时 allowed row 写 `NOT_APPLICABLE`+reason；`NOT_ALLOWED` 永远 N/A。`ACTIVE` task 必须至少有一个 APPLICABLE split；`NOT_APPLICABLE_DATA_GATE` task 必须恰有 0 个 APPLICABLE rows，且每个 contract-allowed row 的 controlled reason/evidence 完整。禁止用模型结果或对象少来改 definition/allowlist。强制 `duplicate_task_registry_id=0`、`duplicate_split_registry_id=0`、`duplicate_task_split_definition_key=0`、`missing_expected_task_split_key=0`、`unexpected_task_split_key=0`、`task_registry_expected_set_mismatch=0`、`task_registry_semantic_mapping_mismatch=0`、`split_registry_expected_set_mismatch=0`、`split_registry_semantic_mapping_mismatch=0`、`task_split_allowlist_mismatch=0`、`task_split_definition_row_count=120`、`task_split_applicability_decision_row_count=120`、`conditional_task_disposition_missing=0`。

### 5.7.4 Global EligibilityRecord

逻辑 `ELIGIBILITY_MANIFEST` 按 §3.2 dual-store 分为 ordinary `data/v3_1/benchmark/ELIGIBILITY_MANIFEST.jsonl`（非 sealed objects）与 restricted mirror（GSE246381）；两 shard 的每行都必须满足同一 `eligibility_record.schema.json`，且只表达 global object/purpose eligibility，不得含 `task_id/split_contract_id/assigned_partition_id`：

```yaml
eligibility_record_id: <stable-id>
eligibility_run_id: <run-id>
object_id: <accepted-pair-id-or-label-complete-observation-id>
object_type: <PAIR|OBSERVATION>
scientific_track: <E|F>
canonical_status: <ACCEPTED>
global_disposition: <ACTIVE|GLOBALLY_EXCLUDED_WITH_REASON|GLOBAL_PENDING_WITH_REASON>
permitted_model_training: <YES|NO|UNKNOWN>
permitted_evaluation: <YES|NO|UNKNOWN>
permitted_derived_release: <YES|NO|UNKNOWN>
permitted_raw_redistribution: <YES|NO|UNKNOWN>
training_eligibility: <ELIGIBLE|INELIGIBLE|PENDING>
evaluation_eligibility: <ELIGIBLE|INELIGIBLE|PENDING>
derived_release_eligibility: <ELIGIBLE|INELIGIBLE|PENDING>
raw_redistribution_eligibility: <ELIGIBLE|INELIGIBLE|PENDING>
effective_future_use_role: <global-role-enum|NOT_APPLICABLE_OBSERVATION>
use_basis_evidence_ids: [<id>]
exclusion_or_pending_reason: <controlled-reason-or-null>
canonical_manifest_sha256: <sha256>
exposure_use_manifest_sha256: <sha256>
license_matrix_sha256: <sha256>
rights_projection_rule_sha256: <sha256>
rights_contributor_set_sha256: <sha256>
fm0_manifest_sha256: <sha256>
effective_role_projection_sha256: <sha256|NOT_APPLICABLE_OBSERVATION>
global_eligibility_decision_evidence_sha256: <sha256>
code_commit: <full-sha>
config_hash: <sha256>
```

每个 global E `pair_id`/F `observation_id`×`eligibility_run_id` 恰有一行；`ACTIVE` 不得有 global pending/exclusion reason，其他 disposition 必须有 reason。任何 `permitted_*=NO|UNKNOWN` 对相应用途不得为对应 eligibility 的 `ELIGIBLE`，其中 derived release 与 raw redistribution 必须分别判定，正常允许 `derived_release_eligibility=ELIGIBLE` 而 `raw_redistribution_eligibility=INELIGIBLE`，不得压成单一 release 状态。

`global_disposition` 只回答“是否进入 benchmark task-cell universe”，不得由 derived/raw release 权限驱动。冻结 truth table 如下，先分别算 training/evaluation purpose eligibility，再由 object type/effective role 决定 disposition：

| object/role | ACTIVE | GLOBAL_PENDING_WITH_REASON | GLOBALLY_EXCLUDED_WITH_REASON |
|---|---|---|---|
| E + `GENERAL_DEVELOPMENT_POOL` | training 或 evaluation 至少一个 `ELIGIBLE` | 两者均非 ELIGIBLE 且至少一个 `PENDING` | 两者均 `INELIGIBLE` |
| E + `SEALED_EXTERNAL_FINAL_CANDIDATE|SEALED_EXTERNAL_FINAL|EXTERNAL_STRESS_ONLY` | evaluation=`ELIGIBLE` | evaluation=`PENDING` | evaluation=`INELIGIBLE` |
| E + base/effective `PENDING` | 不允许 | 必须 | 不允许，除非先有合法 transition→EXCLUDED |
| E + `EXCLUDED` | 不允许 | 不允许 | 必须 |
| F observation | training 或 evaluation 至少一个 `ELIGIBLE` | 两者均非 ELIGIBLE 且至少一个 `PENDING` | 两者均 `INELIGIBLE` |

`permitted_derived_release/raw_redistribution` 与对应 eligibility 仍须完整计算和报告，但不得把本可 train/evaluate 的对象仅因不可发布 raw 而 global-exclude。foundation overlap 的 purpose input 必须来自 object×checkpoint ledger 与本 transaction 选定 checkpoint candidate set：某一 checkpoint 不合格只排除该 checkpoint/claim；object 只有在当前允许的所有模型路径（含经 F↔E final-lineage audit 合格的 supervised fallback 或 from-scratch fallback）都无法形成合法评测 protocol 时才可因此 global-exclude。强制 `global_disposition_truth_table_mismatch=0`、`release_permission_drove_global_disposition_count=0`、`single_checkpoint_overlap_globally_excluded_data_count=0`。

`eligibility_record.schema.json` 必须使用 `oneOf` 锁定 object-type 条件：`PAIR/E` 必须给出真实 `effective_future_use_role` 与 64-hex projection hash；`OBSERVATION/F` 必须且只能写两个 `NOT_APPLICABLE_OBSERVATION` sentinel，不创建 base role、role transition 或 role projection。`ELIGIBILITY_MANIFEST` 与 global unique-object population 一一对应；task-specific disposition 只在下一节。强制 `duplicate_global_eligibility_record=0`、`global_object_without_eligibility_record=0`、`eligibility_record_without_global_object=0`、`eligibility_input_hash_mismatch=0`、`observation_with_pair_role_or_projection=0`、`pair_missing_role_projection=0`、`release_axis_conflation_count=0`。

### 5.7.5 TaskEligibilityCell、SplitAssignment 与 role-partition compatibility

`TaskEligibilityCell` 最低字段为：

```yaml
eligibility_cell_id: <stable-id>
eligibility_record_id: <EligibilityRecord.eligibility_record_id>
object_id: <pair-id-or-observation-id>
object_type: <PAIR|OBSERVATION>
scientific_track: <E|F>
task_id: <TaskRegistry.task_id>
split_contract_id: <SplitRegistry.split_contract_id>
disposition: <ASSIGNED_TO_SPLIT|INELIGIBLE_WITH_REASON|PENDING_WITH_REASON>
assigned_partition_id: <SplitRegistry.partitions.partition_id-or-null>
disposition_reason: <controlled-reason-or-null>
evidence_ids: [<id>]
task_registry_definition_sha256: <sha256>
split_registry_definition_sha256: <sha256>
task_split_definition_matrix_sha256: <sha256>
task_activation_decisions_sha256: <sha256>
split_activation_decisions_sha256: <sha256>
task_split_applicability_decisions_sha256: <sha256>
eligibility_manifest_sha256: <sha256>
effective_role_projection_sha256: <sha256|NOT_APPLICABLE_OBSERVATION>
canonical_manifest_sha256: <sha256>
```

`ASSIGNED_TO_SPLIT` 要求唯一非空 `assigned_partition_id` 且 `disposition_reason=null`；另外两种状态要求 `assigned_partition_id=null` 且 reason 非空。每个 assigned cell 必须与同一 store shard 的 `SPLIT_ASSIGNMENTS.jsonl` 中恰好一条 `SplitAssignment` 双向对应；ordinary path 为 `data/v3_1/benchmark/SPLIT_ASSIGNMENTS.jsonl`，GSE246381 使用 §3.2 restricted mirror：

```yaml
split_assignment_id: <stable-id>
eligibility_cell_id: <TaskEligibilityCell.eligibility_cell_id>
eligibility_record_id: <EligibilityRecord.eligibility_record_id>
object_id: <id>
task_id: <id>
split_contract_id: <id>
partition_id: <SplitRegistry.partitions.partition_id>
partition_role: <derived-role-must-match-registry>
group_assignment_snapshot_sha256: <sha256>
assignment_algorithm_sha256: <sha256>
assignment_evidence_id: <id>
```

E `PAIR` 的 global effective role 与 partition role 唯一允许矩阵为：

```text
GENERAL_DEVELOPMENT_POOL -> TRAIN | DEVELOPMENT | INTERNAL_TEST
SEALED_EXTERNAL_FINAL -> SEALED_FINAL
EXTERNAL_STRESS_ONLY -> STRESS_ONLY
SEALED_EXTERNAL_FINAL_CANDIDATE | PENDING | EXCLUDED -> NO_ASSIGNMENT_ALLOWED
```

E PAIR 的 TRAIN assignment 还要求 global `training_eligibility=ELIGIBLE`；DEVELOPMENT/INTERNAL_TEST/SEALED_FINAL/STRESS_ONLY assignment 要求 global `evaluation_eligibility=ELIGIBLE`。`SEALED_EXTERNAL_FINAL_CANDIDATE` 必须先通过 committed legal transition 得到 `SEALED_EXTERNAL_FINAL`，随后才可写 SEALED_FINAL cell/assignment。

F `OBSERVATION` 不适用 pair role matrix：`training_eligibility=ELIGIBLE` 只允许 TRAIN，`evaluation_eligibility=ELIGIBLE` 只允许 DEVELOPMENT 或 INTERNAL_TEST；v1 F observation 禁止 SEALED_FINAL/STRESS_ONLY，若未来需要 sealed F final 必须另立合同和 observation-reservation state machine。

partition assignment 对同一 `object_id×split_contract_id` 必须跨 task 一致；同一 split contract 的任一冻结 leakage group/connected component 也必须跨 task 落入同一 partition，不能让同一 object/group 在一个 task 训练、另一个 task 测试。assignment algorithm 先产生 task-independent `split_contract_id×group_component→partition_id` map，cell 只继承该 map。每个模型 train/eval manifest 必须精确绑定一个 `target_task_id×target_split_contract_id`，trainer 只能读取该 target 的 TRAIN assigned cells；禁止 union 不同 split contracts 的 TRAIN rows。若为目标 task 使用 F auxiliary/多任务数据，必须按目标 split 对其 STUDY/PARENT/GENE/TILE/CLUSTER/LIBRARY/CONTEXT 等全部 leakage axes 去污染，任何与目标 DEVELOPMENT/INTERNAL_TEST/SEALED_FINAL group 相连的辅助对象不得训练。跨 split-contract 结果必须分别训练模型，或在 C3/PR1 另行冻结不接触外层 test 的 nested protocol，不能用一个 union-trained checkpoint横跨 contracts。

强制 `partition_fk_mismatch=0`、`assigned_cell_without_split_assignment=0`、`split_assignment_without_assigned_cell=0`、`duplicate_split_assignment_for_cell=0`、`cell_assignment_key_mismatch=0`、`global_role_partition_conflict=0`（只对 E PAIR）、`observation_eligibility_partition_conflict=0`、`global_eligibility_partition_conflict=0`、`object_partition_conflict_across_tasks=0`、`group_partition_conflict_across_tasks=0`、`trainer_union_across_split_contracts_count=0`、`target_manifest_mismatch=0`、`auxiliary_to_target_eval_group_leakage=0`。

任何 task/split/规则/global eligibility/role projection/input hash 改变都使 universe 与所有 assignments `STALE_INVALIDATED`，必须使用新 run 完整重建。

---

# 6. 数据获取、清洗与 provenance 合同

## 6.1 系统检索边界

v1 检索必须覆盖 GEO/SRA/BioProject、ENCODE、ArrayExpress/BioStudies、Zenodo/Dryad、PMC 与期刊 supplements、作者 GitHub、MaveDB、MPRAbase，以及关键论文的 backward/forward citation chaining。

必须保存：检索日期、数据库、完整 query、结果总数、去重规则、title/abstract 排除理由、full-text/data-availability 核验、accession、数据状态、许可状态和最后决定。合同只允许声称“在有界检索快照内系统搜索”，不得声称数学意义上穷尽所有未索引、私有、商业或未发布数据。

## 6.2 确定性 acquisition

- raw 文件只读保存，不覆盖旧版本；
- 每个文件记录 URL、release、时间、大小、provider checksum 与 SHA256；
- `.part`、`.new`、CRC 失败、大小不符和 manifest 不完整均不得进入 parser；
- parser 必须使用显式文件名与 checksum；禁止未排序 `rglob(...)[0]`；
- 下载失败按 retry→alternate mirror→raw reconstruction→author contact→documented unavailable 处理；
- 失败 evidence 不删除。

## 6.3 raw→paper_clean→canonical 三层

每个数据集至少保存：

1. immutable raw；
2. `paper_clean`：尽可能重现论文过滤/标签；
3. v3 canonical：统一 schema、role、scope、group、exposure；
4. rejected/quarantined ledger。

每层有 adapter version、代码 commit、config hash、输入/输出 checksum 和 source-native unit/scientific-object conservation。

## 6.4 Source-native unit 与科学对象守恒

每个 accession、每个 source file、每种原生单位 `unit_type`（如 FASTA record、matrix row、design relation、barcode measurement）必须满足：

```text
raw_units[unit_type] = accepted_raw_units + quarantined_raw_units + explicitly_excluded_raw_units
```

该等式只描述 source-native unit disposition，不能同时承担 one-to-many/many-to-one 变换计数。必须另建 `transformation_edges.jsonl`，逐边记录 `raw_unit_id(s) → sequence_id/observation_id/relation_candidate_id` 以及独立的 `relation_candidate_id → pair_id`、unit type、变换类型、代码 commit、config hash 和证据。一个 raw unit 可产生多个 observations，多个 units 可聚合为一个 observation/relation candidate，但每条边必须可回放。

还必须独立满足三类对象守恒，不能用 raw-row 等式替代：

```text
design_relation_candidates = accepted_relations + ambiguous_relations + rejected_relations + pending_relations
observation_join_candidates = matched_observations + ambiguous_observations + unmatched_observations + rejected_observations + pending_observations
accepted_E_relation_population_unique = E_active_unique + E_globally_excluded_unique + E_global_pending_unique
label_complete_F_observation_population_unique = F_active_unique + F_globally_excluded_unique + F_global_pending_unique
E_task_eligibility_cell_population = E_cell_assigned_to_split + E_cell_ineligible_with_reason + E_cell_pending_with_reason
F_task_eligibility_cell_population = F_cell_assigned_to_split + F_cell_ineligible_with_reason + F_cell_pending_with_reason
```

其中 `pending_relations` 精确对应 staging table 中仍为 `relation_acceptance_status=CANDIDATE` 的对象；它不是 formal accepted pair。`ambiguous_relations` 与 `rejected_relations` 分别对应同名状态；只有 `ACCEPTED` 进入 `accepted_relations`。F observation lifecycle 也固定逐类映射：`matched_observations≡observation_acceptance_status=ACCEPTED`、`ambiguous_observations≡AMBIGUOUS`、`unmatched_observations≡UNMATCHED`、`rejected_observations≡REJECTED`、`pending_observations≡CANDIDATE`。五类 candidate ID set 必须互斥、并集恰等于 observation-join-candidate expected set；任何状态映射不一一对应或 category set mismatch 都使守恒 gate FAIL。

上述前两条是 **global unique-object disposition**：E 只以 technical-canonical `ACCEPTED` 的 formal `pair_id`、F 只以 technical-canonical `ACCEPTED` 且 label-complete 的 `observation_id` 计数，每个 ID 在各自等式中只计一次；未接受、ambiguous、rejected 或 technical-canonical failed 的 relation candidate/observation 只在各自 lifecycle 方程中守恒，不得进入 global eligibility 或 task denominator。`active_unique` 表示 technical canonical 已接受且 global purpose 判定允许进入 task-eligibility universe；`globally_excluded_unique` 表示 technical canonical 已接受、但因许可/允许用途、暴露、foundation-overlap 或全局 role 规则而不能进入任何正式任务；`global_pending_unique` 表示 technical canonical 已接受、但上述 global purpose 判定尚未完成。三者互斥且穷尽；B0 若发现新的 canonical defect，必须使本 run stale 并重开 D1，不能把 canonical failure 包装成 B0 global exclusion。不得把同一对象因参加多个任务而重复计数。

后两条是 **task-specific eligibility-cell disposition**。C3 必须按 §5.7 外部冻结的 12-task/10-split sets、semantic descriptor/allowlist hashes 冻结 immutable definitions 与恰好 120-row `TaskSplitContractMatrix`；B0 在同一 frozen D1/FM0 snapshot 上生成完整 Task/Split activation decisions 与 120-row effective applicability decisions，不能以实际 registry 或本轮对象数自己定义完整性。**唯一生成规则**是：只对 B0 decision 中 `APPLICABLE` 的 row，将所有相应 `object_type×scientific_track` 且 `global_disposition=ACTIVE` 的对象与该 `task_id×split_contract_id` 做确定性 cross-product；每个结果必须生成一个 cell，最小主键严格为 `object_id × task_id × split_contract_id`，并通过 `eligibility_record_id` 连接唯一 global EligibilityRecord。`GLOBALLY_EXCLUDED_WITH_REASON|GLOBAL_PENDING_WITH_REASON` 对象只保留 global EligibilityRecord，绝不生成 cell/assignment；`NOT_APPLICABLE` 只存在于 effective applicability decision 层，cell 自身也没有第四状态。某个 applicable row 只有在相应 active-object 集真实为空时才允许零 cell，并必须显式报告 denominator=0 与 object-set hash；不得用“零 universe”替代应生成的对象。

每个 cell 恰处于 `ASSIGNED_TO_SPLIT|INELIGIBLE_WITH_REASON|PENDING_WITH_REASON` 之一；`ASSIGNED_TO_SPLIT` 还必须记录唯一 split role/partition ID。任一 `task_activation_status=ACTIVE` 的 task 没有至少一个 applicable split、任一 active object 未被至少一个 compatible applicable row 覆盖、任一 expected key 缺失或重复均 FAIL；N/A task 必须恰有零 applicable rows并有完整 reason/evidence。多任务复用只允许在不同 cell 中重复同一 object，不得在 global unique-object 等式中重复。Data Card 必须同时逐 track 报告 global unique-object denominator，以及逐 `task_id×split_contract_id` 的 eligibility-cell denominator、三类 disposition和最终分母；并把 canonical-available、train-eligible、evaluation-eligible、derived-release-eligible、raw-redistribution-eligible 五种规模分列，禁止把两种 denominator 或两类 release 权限合成一个“样本数/releasable”。B0 required universe 要求 `E_global_pending_unique=0`、`F_global_pending_unique=0`，并且每个 applicable formal task/split-contract row 的 `E_cell_pending_with_reason=0/F_cell_pending_with_reason=0`。

强制 bijection + payload identity test：每个 `ACCEPTED` relation candidate 必须恰有一个非空 `accepted_pair_id`、恰有一条 candidate→pair edge，且目标 formal pair 的 `relation_candidate_id` 反向相等；每个 formal pair 必须恰有一个 accepted candidate parent。pair 首次物化时，candidate 与 pair 的 `source_sequence_id`、`candidate_sequence_id`、`scientific_track`、`relation_type`、`effect_evidence`、`landscape_role`、`future_use_role`、`pairing_method`、`pair_evidence_id` 必须逐字段相等，而不是只核 FK。`accepted_candidate_without_pair=0`、`accepted_candidate_multiple_pairs=0`、`pair_without_candidate=0`、`candidate_pair_fk_mismatch=0`、`candidate_pair_payload_mismatch_count=0`；任一非零即 FAIL。

`future_use_role` 的任何变化必须严格服从 §5.5.1 的 append-only `RelationRoleTransition` 状态机、合法转移矩阵、hash chain 和 effective-role projection；原始 candidate/pair base row 与历史事件不可覆盖。首次物化时的 base role 不一致计入 `candidate_pair_payload_mismatch_count`，投影后的 effective role 不一致计入 `candidate_pair_effective_role_mismatch`。其他上述 identity payload 在 pair 物化后不可变；如 provenance 修复导致端点或语义变化，必须废止旧 pair、保留 parent-linked supersession evidence，并重新生成新 candidate/pair，禁止原地改写。

每个 relation candidate/observation/pair 必须有稳定 ID、父对象、处置 reason 与 transformation edge。只有 eligibility cell 的 `ASSIGNED_TO_SPLIT` 且存在唯一合法 SplitAssignment，才表示对象对该 task eligible 且已进入唯一 partition；`INELIGIBLE_WITH_REASON` 必须有受控 reason；formal B0 required universe 的 global 与 per-task cell `pending=0`。强制测试 §5.7 全部 expected-set/global-eligibility/cell/assignment/role-partition counters，以及 `orphan_sequence/observation/relation_candidate/pair/edge=0`、`unassigned_eligible_E/F=0`、`active_object_without_applicable_matrix_row=0`，从结构上同时阻止 E/F 无声遗漏、多任务重复计数、悬空 split 和由 barcode/replicate rows 伪造 pair。

同时报告 unique source-native units、unique sequence、relation candidates、accepted pairs、F observations、independent parent、gene、experimental group、duplicate occurrence、ambiguous/unmatched join。任一 raw、relation、observation、E/F coverage disposition 差额或变换边无法解释，必须将相应轴保持 `parse_status=PARTIAL`、`mapping_status=PARTIAL|AMBIGUOUS` 或 `canonical_status=PENDING`，不得用单一旧状态 `PARSED_UNVERIFIED` 掩盖失败所在轴。

## 6.5 序列与坐标 QC

必须执行：

- A/C/G/T/U 与 IUPAC 状态；
- strand、reference allele、genome/transcript release；
- indel left-normalization 与 multi-allelic 拆分；
- adapter/primer/barcode/constant region 处理；
- full sequence 与 variable region 双 hash；
- window 坐标、original length、scaffold 与 editable mask；
- edit script replay；
- exact、reverse-complement、near-duplicate、overlapping-tile clustering。

不得只保存截断 window 而丢失 full-sequence hash 与映射；不得删除非法字符后把序列当作干净。

## 6.6 标签、重复与 delta

label 聚合必须预先规定 coverage/UMI、replicate 最低数、异常值、聚合函数、标准误、missingness 和 endpoint transform。source 与 candidate 只有在同一可比 assay/context、方向明确且各自有测量时才能计算 delta。

WT/source observation 不得因 source==candidate 被删除。一个 source 有多个 WT replicate 时必须先在 source observation table 中聚合，再通过可审计 join 连接 candidate。

## 6.7 Reporter artifact 风险

对所有 DNA-encoded reporter 至少记录共同 scaffold/adapter/cloning 风险；5′与3′再使用区域专属字段。3′重点审计 cryptic splice、cryptic PAS 和 transcript processing；5′重点审计 uAUG/uORF、Kozak/start-context、cryptic splice/PAS 与 cap/chemistry。至少记录：

```yaml
reporter_backbone_id: <id-or-UNKNOWN>
dna_encoded_or_ivt: <DNA|IVT|UNKNOWN>
cryptic_splice_risk: <LOW|MEDIUM|HIGH|UNKNOWN>
observed_splicing_fraction: <number-or-null>
splice_prediction_method: <method-or-null>
splice_prediction_score: <number-or-null>
cryptic_polyA_risk: <status>
uAUG_or_start_risk: <status>
cloning_or_restriction_risk: <status>
splicing_qc_status: <status>
```

有实测剪接时优先实测；预测器仅作风险标记。`reporter_fully_reconstructed=true` 不自动等于低风险。只有预注册的低风险定义通过者进入低风险主分析；HIGH/UNKNOWN 保留在敏感性分析或独立 risk-aware 训练臂，不能事后删除以制造更好结果。

## 6.8 两级清洗完成状态

`D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT` 只在 provenance、schema、source-candidate binding、reference allele/coordinate、序列、numeric label transform lineage、zero-vs-missing、uncertainty、replicate/context、artifact、dedup、library ascertainment、raw/relation/observation conservation、reject ledger、transformation edges 与 canonical checksums 全部通过后声明。

`COMPLETE_CLEANING_FOR_V1_CONTRACT` 还要求 license/use、exposure/access、eligibility、split/leakage、sealed-final 与 G7 closure 全部通过，只能在 B0+G7 后声明。D1 report、claim matrix 或论文不得提前使用第二个名称。两种状态都不等于数据没有测量误差，也不等于生物真实性得到湿实验验证。

## 6.9 Library ascertainment

每个实验库必须记录 design mechanism、candidate inclusion rule、source coverage、编辑距离/action distribution、候选池是否完整以及 selection propensity 是否可恢复。

- 随机库、motif saturation、模型设计候选、临床 variant 和 natural UTR 不能无条件混池；
- candidate pool ranking 只在冻结的真实 observed pool 内解释；
- open-space generation 不得把 library distribution 当作无偏生物目标分布；
- 若 selection propensity 可可靠估计，可预注册 weighting/sensitivity；若不可估计，必须做 library-stratified 结果与限制说明；
- library ID shortcut 与 candidate-count shortcut 是强制负对照。

---

# 7. 数据资产注册表与接入优先级

本节数字是 2026-08-03 的审计目标口径，不因写入合同而自动成为 canonical truth。每个数字必须由 dataset adapter 重放、守恒表和 checksum 再确认。

## 7.1 DATA-P0：v1 必须接入或有证据排除

| Asset | 区域与角色 | 审计目标口径 | 当前状态 | v1 义务 |
|---|---|---:|---|---|
| GSE114002 / Optimus 5-Prime | 5′；random/natural 为 F，designed/SNV 为 E_LINK/E_DELTA 候选，identity/WT 为 E_NOEDIT | 10 个 CSV 约 275 万下载行、约 157 万 unique sequences；当前 55,184 canonical pair-candidates、约 31,576 inferred parents，均不是已验真 pairs | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARTIAL; mapping_status=PARTIAL; canonical_status=PENDING` | 全文件 inventory/join；sublibrary design 重放；random、replicate、designed/SNV、identity 分层；修复文件实际摄入 |
| GSE145046 | 5′ 10-mer；label-complete sequence×context observations 为 F 候选；input-only/normalization/support rows 不属 F；仅恢复显式 anchor/no-edit 后才可有 E_DENSE 子集 | 30 个 gzip 共 30,695,604 downloaded raw measurement rows；full union 有 1,048,576 valid unique 10-mers；当前单个被消费 input 文件有 1,048,106 rows；当前已验真 F observations 与 E relations 均为 0 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARTIAL; mapping_status=PARTIAL; canonical_status=PENDING` | 按 10-mer 确定性连接全部条件，区分 input-support/functional/replicate，标 `RANDOM_INSERT`；只有 label-complete join 计 F，不得称百万 F examples 或 pairs |
| GSE217518 | 5′+3′ ref/mut；通过 assay join 后为 E_DELTA，WT 为 E_NOEDIT | 官方设计约 6,555 variants；审计目标约 5,917 complete ref-mut relation targets；当前 workbook 5,072 行、canonical 3,564 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARTIAL; mapping_status=PARTIAL; canonical_status=PENDING` | 所有 denominator 并列；逐行 attrition；full sequence+window 坐标；稳定 composite ID；assay mapping |
| GSE232571 | 3′ MapUTR rare/clinical；Track E candidate | 论文 rare-variant denominator 17,301；30,910 FASTA records、30,649 unique；约 14,848 complete relation candidates；另 910 dense records，重叠关系待审计 | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | 新 adapter、pair verifier、C3 dense namespace；解释论文口径差异与 910 overlay |
| GSE232572 | 3′ MapUTR COSMIC；Track E candidate | 论文 somatic-variant denominator 11,929；21,971 unique sequences、12,600 complete relation candidates；当前 9,343 canonical pair-candidates | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING` | 设计/论文/canonical 三口径 attrition 闭合 |
| fast-UTR Siegel 2022 | 3′；E_LINK，只有两端同 context 标签齐全者升级 E_DELTA，其余为 F | 41,255 sequences、27,914 source→candidate links、13,296 parents、4,653 genes | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING; license_status=REVIEW_REQUIRED` | pin commit `4aaeb7e97c5ec093c356f0564f96d87887ee9ab7` 与 file SHA；parent/gene split；Jurkat/Beas2B missingness；splicing risk；无明确许可不得再分发 |
| GSE288185 | 3′；Track E，多 endpoint | 11,955 oligos、4,566 designed WT–mutant relation targets、16 genes | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | MPRA/RBNS/SLAM 共索引；gene/tile family split；高度重叠 tile 防泄漏 |
| GSE256185 / DART | 5′；Track F+E | 51,595 reference FASTA；variant pool 约 652 parents/10,737 candidates；四个功能表 raw rows 36,792/11,008/22,330/11,404，跨表可重叠不可相加；多 cap/m1Ψ context | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | 四表 overlap matrix；parent/candidate/random 分账；chemistry/context transfer |
| GSE232927 | 5′；大规模 F；19 条 de novo designs 仍不自动构成 edits | fixed-end 约 204,803 common variants；N25 约 168k 与 Methods 197,341 冲突；N50 约 149k；19 designs 不是 19 pairs | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | 下载并裁决数目冲突；random 不造 source；19 条只作有标签外测 |
| GSE176581 | 5′；Track F+小 Track E | natural 8,414/TE 6,721；Syn-best 2,388 无 input；Syn-low 1,198 有 input | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | 三子集分离；不能称 12,000 paired |
| ENCSR854RUF | 3′ MPRAu；候选 Track E | 原设计约 30,532 allele reporters/12,173 variants；current canonical rows=11,969，relation/source 口径未验真 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING` | source/alt/label/context/许可闭环；cryptic-splicing 风险；低风险主分析 |
| GSE149487 | 5′ pair-candidate/observational | 当前 448 pair-candidate records + 303 observational records | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING` | 448 全部做 eligibility；293T/PC3 mapping feasibility；不得无声漏出 B0 |
| GSE200304 SuperSeries；成员 GSE200302/GSE200303/GSE217530 | 3′ pair-candidate；Polysome-seq、DNA-seq、RNA-seq 三模态 | current canonical relation-candidate records=6,885；其中约 540 当前无可用 label | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING` | 四 accession 全 inventory；三模态分别 checksum/parse/守恒后再做 source/alt、endpoint、replicate/context join；无 label 不进 functional track |
| GSE186455 | 3′ pair-candidate | 论文设计口径 342 proband + 299 sibling variants；当前 649 canonical records、约 639 inferred relation candidates，口径不得相加 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING` | source/candidate、coordinate-verified length change、label 与 duplicate/group provenance 重审 |
| GSE246381 | 5′ explicit-relation/E_DELTA candidate；future use 为独立 sealed external final candidate | 1,184 reconstructed non-identity relation candidates、约 1,170 inferred pair-candidates、1,151 inferred sources；约 33k uniquely barcoded reporters 不是 33k pairs | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING; license_status=REVIEW_REQUIRED` | owner prior analytic use=`NONE_CONFIRMED`；legacy materialization=`PRESENT`；foundation requirement=`REQUIRED_FM0_A` 且 audit status=`DEFERRED_TO_FM0_A`，结果只能见逐 checkpoint ledger；future role=`SEALED_EXTERNAL_FINAL_CANDIDATE`；隔离 QC/许可/FM overlap 后才逐记录裁决 E_DELTA，一次性 final；禁止 historically exposed/E4X |
| E-MTAB-10902/11572/11575 / N-zip | 3′；基础 tiles 为 F，mutagenized library 为高价值 E_DENSE | 基础 4,813 designs/4,745 QC/99 UTRs；突变库 6,266 designs/5,679 QC；independent parents 约 16 而非 5,679 | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | 三 accession、supplement、WT-family、tile overlap 与 sequence-label join 闭环 |
| GSE330741 | 3′；基础 tiles 为 F，8 element groups 的 single-nt mutagenesis 为 E_DENSE | 审计目标约 6,500 designs、5,818 QC retained；不是 5,818 parents | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | mutation family/design table；与复用 GSE295080 lineage 去重 |
| GSE261709 | 3′ eQTL ref-alt；E_DELTA 候选 | 约 749 variant loci；allele oligo、barcode 和 assay rows 另算 | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | variant-barcode-sequence、cell context 与 triplicate label join |
| GSE298114 | 3′ ref-alt；E_DELTA 候选 | 约 400 variant loci；6 个 GEO samples 不是六倍 pairs | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | hg38、strand、ref allele、164-nt oligo 与 DNA/RNA triplicate 确定性重建 |
| PRJNA1116243 / PTRE-seq | AUX_QC_ONLY；不属于 E/F 主训练 | 642 reporters；splicing-fraction QC 单位，不是 parents/pairs | `acquisition_status=NOT_PRESENT; parse_status=NOT_STARTED; mapping_status=NOT_STARTED; canonical_status=PENDING` | sequence/backbone lineage 映射；输出 reporter artifact assessment；仅支撑 artifact gate |
| GSE207584 | audit=P0 cleanup；science=OUT_OF_SCOPE CDS | 当前 10,227 条 records 的 source 与 candidate sequence 均为空/未保留；属于 legacy labels-without-sequence liability，原 FASTA 可恢复 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=FAILED; canonical_status=PENDING; scientific_status=EXCLUDED` | 从 UTR 主 canonical quarantine/exclude；恢复 FASTA 只为清账，不扩大数据 |
| GSE173083 | audit=P0 cleanup；science=P2/full-mRNA AUX | 当前 canonical 3,263，与约 233-row 主表单位冲突 | `acquisition_status=DOWNLOADED_VERIFIED; parse_status=PARSED; mapping_status=PARTIAL; canonical_status=PENDING; scientific_status=AUXILIARY_ONLY` | 对账实验单位；不得进入 UTR paired 规模、E/F 主规模与主任务 |

DATA-P0 每项最终必须达到 `canonical_status=ACCEPTED` 或 `canonical_status=EXCLUDED_WITH_EVIDENCE`；许可与各 use eligibility 仍单独裁决。P0 表示必须完成审计，不表示必须强行混入同一训练集。

除非逐资产已经记录明确许可名称、证据快照/hash、核验日期和允许用途，否则公开 GEO/ENCODE/BioStudies landing page 或可下载文件一律保持 `license_status=REVIEW_REQUIRED`；公开可访问不等于允许训练、评测、衍生发布或原始再分发。

### 7.1.1 冻结优先级全集；禁止 registry 自证完整

必须创建 `data/v3_1/registry/priority_snapshot_v3_1.yaml`，逐项记录 `asset_group_id/accessions/audit_priority/scientific_priority/required_role/required_action/promotion_condition/expected_source_classes`。v3.1 P0 的冻结 `asset_group_id` 集合为：

```text
GSE114002
GSE145046
GSE217518
GSE232571
GSE232572
FAST_UTR_SIEGEL_2022
GSE288185
GSE256185_DART
GSE232927
GSE176581
ENCSR854RUF
GSE149487
GSE200304
GSE186455
GSE246381
NZIP_EMTAB_10902_11572_11575
GSE330741
GSE261709
GSE298114
PTRE_PRJNA1116243
GSE207584_CLEANUP
GSE173083_CLEANUP
```

`asset_group_id=GSE200304` 的成员集合也属于冻结全集，必须精确为 `accessions=[GSE200304,GSE200302,GSE200303,GSE217530]`；asset-group set equality 与组内 accession set equality 两层都必须 PASS。GSE200302、GSE200303、GSE217530 分别登记 Polysome-seq、DNA-seq、RNA-seq，不得只登记 SuperSeries landing page 就称完整。

v3.1 P1 的冻结 `asset_group_id` 集合为：

```text
GSE194092
GSE270252_270254
GSE173098
GSE295080_ISOMPRA
GSE291719_SONAR
GSE55396_FAST_UTR_2014
PASSPORT_SEQ
SEERS
```

其余冻结类别集合为：

```text
P2_ACQUISITION_WATCHLIST = [PARADE, SALUKI_HALF_LIFE]
REFERENCE_SERVICE = [GENCODE, REFSEQ, ENSEMBL, UTRDB, RNACENTRAL]
ANALYSIS_ONLY_OUT_OF_SCOPE = [CODONBERT, OPENVACCINE, BPRNA_STRUCTURE_ONLY]
SEARCH_NEGATIVE_LEDGER = [MAVEDB, MPRABASE]
```

GSE173083_CLEANUP 与 GSE207584_CLEANUP 只在 P0 audit set 出现一次；它们的 `scientific_priority=P2_AUX|OUT_OF_SCOPE` 是同一 registry record 的科学 overlay，不得复制进 P2 audit set。每个 asset group 只能有一个 `audit_priority`，但可有独立 `scientific_priority/class`。

Gate 必须验证 registry P0、P1、P2 acquisition、reference service、analysis-only 与 search-negative sets 分别与上述显式 ID 集合完全相等；缺项、多项或只靠“等其他资产”均 FAIL。`audit_priority` 与 `scientific_priority` 分开：例如 GSE207584 为 P0 cleanup 但 science out-of-scope，PTRE 为 P0 audit 但 science AUX_QC。

## 7.2 DATA-P1：高价值、需确定性重建

| Asset | 预期角色 | 接入前置条件 |
|---|---|---|
| GSE194092 | F 或在找到明确 anchor 时为 E_DENSE；理论 16,384、16 conditions 共同观测约 12,270，independent parent 约 1 | 锁定 endpoint/anchor；不得全互配或称 12,270 parents；只作 landscape/calibration |
| GSE270252/GSE270254 | F full-length 3′；约 1,401 QC UTR、双 promoter、多 endpoint，pair 数为 0 | 恢复 full-length sequence–barcode；解释 4,245 barcode rows、1,404 IDs、1,401 QC UTR 三口径 |
| GSE173098 | F localization；约 47,989 reporters | 找到完整 oligo map；只有作者设计表提供真实 parent 才能建立 E_LINK，不能因 `designed` 字样造 source |
| GSE295080 / ISOMPRA | WT-family design table 闭合后可为 E_DENSE，否则 F | 完整 design table；pre/post-splice、TRAP、HEK/AAV brain；与 GSE330741 lineage 去重 |
| GSE291719 / SONAR | F；约 467 reporters，多 T-cell context | 找到完整 oligo map 前不进 sequence model；27 GEO samples 不是 `467×27` pairs |
| GSE55396 / Zhao fast-UTR 2014 | 主体 F；明确 mutation clone→parent 的小子集可为 E | 重放 16,332 design sequences、论文约 3,000 regions/2,089 genes、先前 retained 2,828 segments 三口径；连接 steady-state RNA/stability/protein |
| PASSPORT-seq | E_DELTA sealed-external 候选；约 111 3′ variants/17 genes | 恢复稳定 accession、完整 sequence-label table、cell contexts 与许可；通过后可升 P0 |
| SEERS | F only | 只有公开 design+label+许可后接入；论文 in-silico saturation mutagenesis 不得产生实验 E pairs |

P1 数据只有在 sequence–label/barcode 映射无歧义、许可状态可审计、与 v1 final lineage 无冲突后才接入。否则保留为 auxiliary/watchlist，不得用论文声称规模替代下载事实。

v1 冻结前，所有 DATA-P1 必须分别闭合 acquisition/parse/mapping/canonical/license/use 各轴；D0 的 `d0_decision` 只能是 `ACQUIRED_FOR_REBUILD|METADATA_ONLY|MAPPING_UNRESOLVED|EXCLUDED_WITH_EVIDENCE`，不能冒充 D1 的 canonical 状态。凡公开文件已足以确定性恢复 sequence-label 且许可/用途允许者，必须接入相应 E/F 轨，不能只因解析困难而跳过。

## 7.3 外部 final、auxiliary、legacy cleanup 与 out-of-scope watchlist

- GSE173083/PERSIST-seq：科学上最多 full-mRNA F-aux/P2；同时是当前 canonical 的 P0 cleanup liability。必须解释约 233-row 主表与当前 3,263 canonical 单位差异。
- PARADE：若当前只有 metadata/`files=[]`，必须写 `acquisition_status=METADATA_ONLY`；不得把该 D0 acquisition 轴的值误写成 composite/canonical status，也不得称已下载 60k。
- Saluki half-life：`audit_priority=P2`、`scientific_role=F_POTENTIAL`、`acquisition_status=DOWNLOAD_FAILED`；只有取得稳定文件、版本/checksum、sequence-label mapping 与许可/use closure 后才可经 decision log 升 P1，当前不算可用资产。
- CodonBERT：U-track 取消后移出项目数据计划；不得保留为数据规模或项目预训练资产。
- OpenVaccine：analysis-only/out-of-scope whole-RNA asset；不能进入 UTR E/F 核心、paired evidence 或训练规模。
- bpRNA/structure-only/reference assets：只可用于非训练的离线方法学分析、工具验证，或对已经纳入的 E/F sequence 运行冻结的确定性特征计算；其 reference rows 不得成为训练 examples、representation learning 来源或数据规模，也不得通过 join 注入外部标签。任何模型输入结构特征只能由该 E/F record 自身序列与冻结软件生成。
- GENCODE/RefSeq/Ensembl/UTRdb/RNAcentral：`REFERENCE_ONLY_NOT_TRAINING`；只作坐标、release、strand、reference allele 与 UTR 边界服务，不产生训练 example、不计 N。
- GSE207584：CDS cleanup；从主 canonical quarantine/exclude，恢复 FASTA 只为闭合 lineage。
- MaveDB/MPRAbase：保留检索与负结果 ledger；检索命中数不得计为可用数据。

## 7.4 Dataset-specific 最低验收

除通用 gate 外，至少满足：

### GSE114002

- 10 文件逐个 checksum、schema、缺失、非法字符与 cross-file overlap；
- 当前 raw 审计的 19 个非 ACGTU 与 3,193 个缺 RL rows 必须 quarantine/missingness 记账，不能静默修正或丢失；
- identity/WT 作为 `NO_EDIT_CONTROL` 和 source observation 恢复；
- 55,184 current canonical pair-candidate records 的 sublibrary 来源与角色逐条重放；只有 relation evidence、方向和两端序列验证通过者进入 accepted E relation denominator；
- coordinate-verified/source-reported length-changing relation 与等长序列的 latent INS/DEL alignment 分开；
- 当前 11,597 条 `k_edit>5` 不能进入 k≤5 recovery 分母；当前 13,292 条等长 pair-shaped records 的算法 INS/DEL path 不得称真实 length change。

### GSE145046

- 所有 condition 文件显式列入 manifest，不依赖第一个 glob；
- 每个 10-mer 的 input、monosome/polysome/ribosome-free、GFP gating、half-life、replicate/timepoint 结构化；
- join 的 matched/unmatched/duplicate 守恒；
- valid 10-mer union 必须重放为 1,048,576；当前 26,188 个非法或长度≠10 rows 必须 quarantine；
- 零计数与该 condition/file 中未观测必须分开；
- 固定 scaffold、insert 坐标和 `RANDOM_INSERT` scope 明确；
- 不能外推 full 5′UTR 长程语法。

### GSE217518

- 官方设计、workbook、assay table 和最终 canonical denominator 并列；
- 当前 5,072 workbook 行 100% accounted；
- 当前 3,564 successes、1,508 failures、38 reconstructed duplicate-ID occurrences 必须持久化逐行 reason；5′/3′成功、失败、duplicate、unique 分别报告；
- full sequence、原始长度、window 坐标、变异坐标、editable mask 可回放；
- 当前至少 2,336 条触发 >500-nt window 的 records 必须补 window start/end 与 full-sequence hash；
- HEK293T/SH-SY5Y 与 timepoint/barcode 映射只有在唯一 join 时进入主表；
- HEK_U3、SH_U3 各 6,235 rows，HEK_U5、SH_U5 各 6,237 rows，且每条 9 timepoint×barcode measurements，必须结构化连接或明确 unmatched/ambiguous；
- 所有 eligible pair 纳入 B0 或进入 exclusion manifest。

### MapUTR / fast-UTR / ENCSR

- ref/alt 完整配对、strand/reference build、cell context 与 source measurement 可审计；
- overlapping oligo、gene、variant family 原子化 split；
- 对 DNA reporter 执行预注册 cryptic-splicing/PAS 风险；
- ENCSR 未解决 role 前不得作为 headline 3′ indel/multi-edit evidence。

### DART / GSE232927 / GSE176581

- DART 的四个功能表、FASTA、variant-parent table 和 cap/m1Ψ context 必须做 overlap matrix；51,595 FASTA records 不得称 51,595 parents；
- GSE232927 的 fixed-end、N25、N50 与 19 downstream designs 分资产守恒；随机库不得构造伪 source；
- GSE176581 的 natural、Syn-best、Syn-low 分表；只有 Syn-low 的显式 input→candidate 可进入 E_LINK，且两端兼容测量后才可升级 E_DELTA；
- 任何 F 数据进入 supervised auxiliary training 前必须与 E final 做 family/cluster/label-lineage 隔离。

### N-zip / GSE330741

- base tiles 与 mutagenized libraries 分轨；WT-family/group ID 必须从设计表恢复，不能把 mutants 当 independent parents；
- overlapping tiles、transcript、element group、reused sample lineage 原子化 split；
- N-zip 三个 E-MTAB accession 共享 design/sequence namespace，不得重复计数；
- GSE330741 与 GSE295080 的 reused samples/lineage 必须显式去重，只有独立新设计可进入 leave-one-study-out 主张。

### GSE261709 / GSE298114

- variant locus、ref/alt allele、strand、reference build、oligo、barcode、cell/replicate 与功能标签逐层连接；
- loci/pairs 与 barcode/assay rows 分别报告；
- 任一 ref allele 或序列重建歧义进入 quarantine，不得按相似度猜测；
- 只有两端同一可比 context 的测量进入 E_DELTA，其余保持 E_LINK。

### GSE246381

- owner-confirmed prior training/tuning/error-analysis/human-label use 必须固定为 `NONE_CONFIRMED`，同时 pipeline sequence/label materialization=`PRESENT`；禁止任何 E4X/historically-exposed 正向解释；
- project exposure 与 external-foundation overlap 分开；FM0 未完成时只写 `foundation_overlap_requirement=REQUIRED_FM0_A`、audit status=`DEFERRED_TO_FM0_A`，不得伪造一个 accession 级 UNKNOWN 布尔；FM0 后按 checkpoint ledger 判断；
- use role 默认 `SEALED_EXTERNAL_FINAL_CANDIDATE`，B0 冻结前 labels 不得进入 trainer、tuner、error analysis 或人工检查；
- 两个 raw matrices 各 32,990 rows/SeqIDs，30,140 variant、1,350 control、1,500 shuffle 必须分角色；barcode/reporters 不得计为 pairs；
- 1,507 `(variant, ENST)` keys、当前 1,300 reconstructed、207 failures、116 identity/no-edit、1,184 edits 必须守恒并有 reject ledger；
- 347 个长 UTR window 必须保存 full hash、original length、window start/end、variant coordinate 与 editable mask；
- 保留 raw replicate matrix，明确 zero count 与 missing；当前“只累计正值”、HEK 正值均值与 Vglut 直接求和比例的聚合逻辑必须重现论文或被替换为预注册、带 uncertainty 的正规化；
- 当前 13 个聚合标签字段不能直接视为已验证功能真值，必须保存 replicate、library-size normalization、coverage/UMI、SE 与 missingness；
- QC/许可/task eligibility 不通过时 `EXCLUDED_WITH_EVIDENCE`，不得自动转 train；
- 一次性 final 后更新为 `FINAL_OPENED`，v1.1 不得继续作为新的 sealed final。

### PTRE-seq reporter QC

- 只用于目标 reporter 的 splicing/artifact 风险，不进入 E/F 主训练和规模；
- 只有 sequence/backbone lineage 可确定性映射到目标 reporter 时才可传递 QC 标签；
- 642 reporters、splicing-positive counts 和 threshold counts 分开报告，不称 parents/pairs。

### GSE207584 legacy cleanup

- perfect CSV 10,227 rows/955 unique names 必须全部连接 1,395-record、339-nt FASTA；retained auxiliary 空 sequence=0；
- imperfect CSV 1,694 rows 单独清账，不得因 `*perfect*` glob 误选而漏掉；
- 从 UTR 主 canonical 移除；即使恢复完整也只为 `CDS_AUXILIARY`，不增加 E/F 或 UTR 论文规模。

## 7.5 当前数据充足性判断与挖掘优先函数

1. Track F 的功能序列/observation 潜在规模可达百万级，但主要支持 supervised property/context/scorer 任务，不能写成百万 edit pairs。
2. Track E 的 relation links/delta-eligible pairs 在完成 adapter、label join、artifact QC、去重和 lineage split 前不得求和；当前合理预期是 `10^4–10^5` 量级而非百万级，该范围只是 planning estimate，不是论文数字。
3. 真正稀缺的是 independent parents、genes、studies 和 contexts：例如 N-zip 数千 mutants 约来自 16 个 WT parents，GSE194092 上万观测可能约 1 个 parent，DART 上万 candidates 约 652 parents。
4. 数据挖掘优先函数固定为：`新增可确定性 E_DELTA independent parents > 新增 E_LINK parents > 新增 E_DENSE families > 新增有独立 context 的 F > 单纯新增 rows`。
5. 因此 N-zip、GSE330741、GSE261709、GSE298114、MapUTR、GSE217518、ENCSR、GSE200304/186455 优先；structure-only、CodonBERT 和无标签天然 UTR corpus 不消耗 v1 acquisition 预算。
6. 如果清洗后 E independent parents 或 study diversity 不足，论文仍可作为 benchmark/resource，但必须缩小 generation/ranking 与跨区域 claim；不得通过把 F 或 dense candidates 伪装成 pairs 补样本量。

---

# 8. Split、暴露与 benchmark v1

## 8.1 Split 的统计单位

分组层级固定为：

```text
study/accession
  → assay lineage/library/batch
    → gene/transcript/tile family/scaffold
      → biological parent/source
        → candidate/replicate/barcode
```

同一 source 的全部 candidate、同一 ref-alt pair、同一 overlapping tile family、同一高相似 isoform、同一 scaffold/library lineage 必须原子化进入同一 split。

## 8.2 必须分开的 split

1. `5utr_source_disjoint`
2. `5utr_study_disjoint`
3. `5utr_sequence_cluster_disjoint`（Track F/observation）
4. `3utr_source_or_variant_disjoint`
5. `3utr_study_disjoint`
6. `3utr_sequence_cluster_disjoint`（Track F/observation）
7. `cross_region_5_to_3`
8. `cross_region_3_to_5`
9. `heldout_context`；registry row 永久保留，仅在 context 数据量满足门槛时激活，否则 `CONDITIONAL_NOT_QUALIFIED`
10. `sealed_final_v1`

`study_disjoint` 不得通过“5′训练、3′测试”实现；region 与 study 必须可分别归因。cross-region 是单独的 secondary experiment。

## 8.3 泄漏审计

formal split 必须审计：

- exact sequence；
- reverse complement；
- near duplicate/cluster；
- source-candidate pair；
- biological parent；
- gene/transcript/ortholog clan，当主张相应泛化时；
- overlapping tile/scaffold/library；
- assay lineage/context/barcode；
- foundation sequence/label exposure；
- latent-path intermediate sequence exposure。

任何主张依赖的维度缺失时，不能 `N/A+PASS`；必须补数据、阻断该 track 或缩小 claim。

## 8.4 Partition access 与 ordinary evaluator

- `TRAIN` 只能用于参数拟合；不得据 TRAIN metric 选择 final claim threshold。
- `DEVELOPMENT` 可用于调参、配置、metric branch 与 model selection，所有读取必须进入 ordinary access log；一旦读取就保持 development-exposed。
- `INTERNAL_TEST` 在全部 PR2 rounds、checkpoint/config/threshold/metric selection 中必须不可见；只允许 `data/v3_1/evaluation/ORDINARY_EVALUATOR_RUNS.jsonl` 注册的 frozen evaluator 按 `ONE_ATTEMPT_PER_FROZEN_MODEL_FAMILY_AND_PROTOCOL` 运行，先写 `INTERNAL_TEST_EVALUATOR` intent，输出只含冻结 aggregate schema。结果一旦向开发者可见，任何后续模型/config/threshold/metric改动都使该 internal-test evidence降级为 DEVELOPMENT，不能继续称 E3；需要新 frozen internal cohort/version。
- `SEALED_FINAL` 仍服从 §8.5 的独立一次性策略，绝不能因 INTERNAL_TEST 权限而打开。
- ordinary evaluator row 至少绑定 `evaluator_run_id/target_task_id/target_split_contract_id/model_family_id/checkpoint_sha256/config_sha256/container_sha256/budget_sha256/output_schema_sha256/internal_test_object_set_commitment/access_intent_id/access_terminal_event_id/status/result_manifest_sha256`；同一 frozen key 最多一个 intent。强制 `internal_test_visible_before_freeze=0`、`internal_test_attempt_count_per_frozen_key<=1`、`internal_test_output_schema_violation=0`、`internal_test_used_for_subsequent_selection=0`、`internal_test_reclassified_as_development_when_modified=100%`。

## 8.5 Final 封存

- final member IDs、labels、candidate pools 与 evaluator 分离保存；
- 训练代码和开发人员不可通过普通 loader 读取 final labels；
- final manifest 在 PR1 冻结；
- final 由与开发角色分离的 custodian/evaluator command 管理；开发代码只获得按预注册 schema 输出的 aggregate metrics，不返回逐条标签；
- final evaluator 在模型、fallback、config、seeds、container、budget 与 output schema 冻结后只运行一次；
- final 打开后机器状态永久保持 `FINAL_OPENED`，论文/治理语义可称“v1 final 已 consumed”，但不得另造未定义的 `FINAL_CONSUMED` enum；任何模型、数据、阈值、metric 或代码改动均为 post hoc。若继续研发，必须进入 v1.1 的新 final，不得重用 v1 final 调参；
- 这只是 `OPERATIONALLY_SEALED_RETROSPECTIVE_FINAL`，不能称 prospective external validation。

GSE246381 为 `RETROSPECTIVE_PUBLIC_PROJECT_UNEXPOSED` sealed-final candidate；不得归入 historically exposed E4X，也不得在 final 前进入训练、调参、错误分析或人工标签检查。旧 pipeline materialization 与旧 split membership 必须登记并在 v3.1 B0 重建时失效。

---

# 9. 5′/3′非对称任务合同

## 9.1 5′ confirmatory tasks

### T5-Rank / Closed-Select：同 source、真实 measured-pool 功能选择（`T5_RANK_CLOSED_SELECT_E_PAIR`）

- 只允许 `effect_evidence=BOTH_SAME_CONTEXT`、方向验证、`confirmatory_delta_eligible=true` 的 accepted E relation；dense 是可重叠 overlay，不是替代资格；
- primary：每个 source 至少 5 个冻结 candidate；
- sensitivity：2–4 candidate；
- 单 candidate source 不进入 NDCG/regret；
- ranking cutoff 固定记为 `K_rank`，不得与 edit budget `k_edit` 或 alignment samples `K_align` 共用符号；
- primary metric：source-macro `NDCG@K_rank`，relevance transform、ties、少于 `K_rank` candidates 和 missing labels 的处理在 PR1 冻结；
- secondary：source-macro normalized regret、top-`K_rank` hit、study-macro Spearman；
- 若 source measurement 未 join，任务只能称 candidate absolute-property ranking，不称 delta optimization，也不能进入本 confirmatory pool；
- scorer、ranking 与 search 只在完全相同的冻结 measured candidate pool 上比较，真实历史测量给出 NDCG/regret/hit；不得用预测器对未测 open candidates 的分数替代该 ground truth。

### T5-Gen-Reconstruct：held-out endpoint distribution/reconstruction（`T5_GEN_RECONSTRUCT_E_PAIR`）

relation-verified 的 E_LINK/E_DELTA 均可进入，但功能标签不进入 generator objective。给定 source、region、允许 context 和预算 `k_edit∈{1,3,5}`，每个 source 发起固定 `B` 次生成 attempt。PR1 必须在不看 final 的情况下冻结唯一 `k_primary` 与 `B`，或冻结单一预定义 budget-AUC；不得在三种 k/多个 B 中按结果择优。

v1 primary 为 `source-study-macro exact endpoint recovery@B`：以每个冻结非 identity held-out endpoint `(s,y)` 为评测单位，B 次输出中至少一个等于 y 记 1；先在同一 source 内对 endpoints 等权，再按 source、study 分层汇总。必须冻结 `d(s,y)` 的算法与 eligibility，按 distance 分层；identity/no-edit 不进入主 recovery 分母，另报 no-op specificity。duplicate、invalid 与 contract-violating outputs 都消耗 B 且不得 resample；同时报告 duplicate/invalid rate。`best-of-B normalized edit similarity` 为 secondary，使用冻结 normalized Levenshtein。

该任务只回答历史 endpoint distribution/reconstruction，报告 recovery、edit-distance/action/length calibration、contract validity、diversity、duplicate rate、memorization 和 source/study macro performance；只与 `BASIC_PAIR_EF_NATIVE`、AR、masked/denoising、diffusion 等同目标 editor 比较，不与 scorer+search 的功能优化数字混排。

B0 必须仅用 frozen development metadata 与不学习的 oracle/retrieval baselines 重放 C3 的非退化/功效 rule，并在 `TASK_ACTIVATION_DECISIONS.jsonl` 冻结：至少 500 个独立非 identity sources、至少 5 个独立 studies/libraries、每个要进入 action-specific claim 的 action stratum 至少 100 endpoints、所选 B 下预期 exact-hit events≥50，且预注册 cluster-aware CI precision 可接受。若 exact gate 通过，selected branch=`GEN_EXACT_CONFIRMATORY`；若 exact recovery 结构性近零但 frozen normalized-similarity estimand 通过同类 gate，则 selected branch=`GEN_SIMILARITY_CONFIRMATORY`；两者都不通过时 task 仍 `ACTIVE`、`confirmatory_status=DESCRIPTIVE`、selected metric=`DESCRIPTIVE_NO_CONFIRMATORY_PRIMARY`，继续生成 cells/splits但不产生 confirmatory headline。任何 threshold/rule 变更必须有用户批准的 decision log，并使 B0/G7 stale 后从 B0 重建；PR1 只能绑定 B0 decision hash，不得重新选择或修改。final 后永远不得把 secondary metric 升格救结果。H1A/H6A 必须绑定 B0 选定的唯一 confirmatory branch；若为 descriptive 则两项不具 confirmatory 资格。

### T5-Context

仅对同一 sequence 或同一 verified relation group 在多个 context 下有 repeated measurements、且 ContextRegistry 结构完整的数据做 cell/cargo/cap/chemistry transfer。E-pair 与 F-observation 必须分别进入 `T5_CONTEXT_E_PAIR` 与 `T5_CONTEXT_F_OBSERVATION`，并各自独立重放 activation gate：E context 不足只使 E row=`NOT_APPLICABLE_DATA_GATE`，不能连带关闭合格的 F row；F context 不足反之亦然。两个 registry rows 始终保留，不能合并 object type、共享 activation decision 或删 task。

### F5-Outcome-Aux（`F5_OUTCOME_AUX_OBSERVATION`）

5′ Track F label-complete observations 只在 `5utr_sequence_cluster_disjoint/5utr_study_disjoint` 下用于 supervised property/scorer/representation auxiliary training 与 monitoring；它不是 source-conditioned edit task，不产生 pair count、endpoint-recovery 或功能改善 headline。

## 9.2 3′ confirmatory tasks

当前 3′同 source 多候选池很稀疏，不能复制 5′ ranking 作为 primary。

### T3-Effect / Property

- `T3_EFFECT_DELTA_E_PAIR`：只有 source measurement join 完成时预测 within-assay normalized delta；
- `T3_PROPERTY_E_PAIR`：E relation 存在但不能形成可比 delta 时，预测 candidate endpoint property，必须使用独立 task ID；
- Track F absolute-property auxiliary 不借用上述 pair tasks，进入 `F3_OUTCOME_AUX_OBSERVATION`；
- primary metric 必须由数据语义预规则决定：仅关心排序且尺度不可比时用 study-macro Spearman；具有可比、冻结单位与 transform 时用 study-normalized MAE；不能根据开发结果二选一；
- 必做 leave-one-study-out、no-study-ID、source-only、candidate-only、single/multi-edit 分层。

### T3-Reconstruct（`T3_RECONSTRUCT_E_PAIR`）

在 action budget 内做 endpoint reconstruction，分别报告 substitution-only、coordinate-verified/source-reported length-changing、latent alignment indel、single/multi-edit。

### T3-Rank exploratory（`T3_RANK_EXPLORATORY_E_PAIR`）

只有数据冻结后同时达到至少 500 个独立 multi-candidate sources、至少 5 个独立 studies/libraries 且每个 source 至少 5 个有效 candidates，B0 decision 才可按预冻结 policy 将 `confirmatory_status` 升为 `CONFIRMATORY`。只要预定义 exploratory population 非空，task 必须保持 `ACTIVE+EXPLORATORY` 并生成完整 cells/splits/denominator；低于升级阈值不能写 N/A。只有 usable exploratory population 真实为 0 时才允许 `NOT_APPLICABLE_DATA_GATE`，且不再声称报告 formal exploratory subset。当前快照 3′只有 151 个 source≥2、7 个 source≥5，因此预期为 ACTIVE+EXPLORATORY，不能复制 5′ ranking headline。

`F3_OUTCOME_AUX_OBSERVATION` 对 3′ Track F label-complete observations 使用 `3utr_sequence_cluster_disjoint/3utr_study_disjoint`，只承担 supervised auxiliary/scorer/property learning；不成为 edit pair 或 3′ ranking headline。

## 9.3 跨区域任务

E-pair reconstruction 使用 `CROSS_REGION_RECONSTRUCT_E_PAIR`；F-observation property transfer 使用 `CROSS_REGION_PROPERTY_F_OBSERVATION`。两者只回答 `fully_shared`、`shared_plus_region_adapter`、`independent` 三者的计算表现与迁移边界，不能混合 denominator 或互相补偿。不得把迁移性能解释为共享生物机制。任一端区失败时不得报告 `UTR-general advantage`。

## 9.4 Open generation

`OPEN_GENERATION_DIAGNOSTIC_E_GENERATION_TASK` 只存在于独立 `DiagnosticRegistry`，不进入 TaskEligibilityUniverse；以 unique `generation_task_id/source_id` 为单位，仅报告 contract validity、novelty、diversity、edit cost、OOD distance、motif/structure distribution、多个冻结 predictor 的 disagreement 和 uncertainty。不得按 source 的 pair 数重复 denominator。

此处 `novelty` 只指冻结集合上的 sequence diagnostics：相对 train/source/known-endpoint sets 的 exact duplicate rate、相对 frozen development canonical 的 minimum edit distance 与 cluster overlap。若 external-backbone corpus 不可审计则写 `NOT_AVAILABLE_NOT_ASSERTED`；相对 sealed final 的 overlap 只能由一次性 evaluator 输出聚合值。不得把“未命中训练集”写成全球新颖性、生物新颖性或功能新颖性。

同一 predictor 不得同时作为唯一 reward、唯一 selector 和唯一 final evaluator。无新湿实验时，open generation 永不解锁“功能改善”主张。

只有未来另立合同、为 Flow 显式加入与所有方法一致的 frozen desired-outcome objective/reward，并具有相同真实 measured denominator 时，才允许其进入功能 search 的 matched comparison；该 reward-conditioned extension 不属于 v1 primary architecture。

---

# 10. Edit-Flows-derived UTR reference model 与 alignment robustness

## 10.1 Primary model 输入与动作

状态定义为：

```text
source sequence s
current sequence x_t
region r
structured context c
edit budget b
time t
editable mask m
```

主动作空间：

```text
SUB(position, nucleotide)
INS(position, nucleotide)
DEL(position)
```

固定 edit budget/fixed horizon 是 v1 唯一 termination policy。`NO_EDIT_CONTROL` 由 k=0/identity endpoint 表达并只用于 no-op specificity/noise/calibration。STOP 在 v1 全面 disabled，不属于任何 trial 或 headline。

模型输出非负 action rates：

\[
\lambda_\theta(a\mid s,x_t,r,c,b,t,m)\ge 0,
\]

任何非法动作在归一化、采样和 loss 前被 hard mask。hard mask 只包含可确定的语法/合同规则；motif、结构、表达等不确定效果属于 soft preference 或 learned critic。

### 10.1.1 Task→model→loss 强制映射

| 组件 | 允许数据 | 输出与 loss | 允许任务/主张 |
|---|---|---|---|
| `EditFlowGenerator` | relation-verified E_LINK/E_DELTA/no-edit controls；不读取 sealed final | action rates；Edit Flow/Bregman objective | T5-Gen-Reconstruct、T3-Reconstruct、open-generation diagnostics；不直接输出功能分数 |
| `OutcomeScorer` | label-complete F 与 development-side E_DELTA，按 endpoint/context 分头 | scalar/distributional outcome head；冻结的 supervised ranking/regression/calibration loss | T5-Rank/Closed-Select、T3-Effect；不是生成器 flow objective |
| `GeneratorPlusFrozenScorer` | generator 输出 + 与 training reward/final evaluator 隔离的 frozen development scorer | proposal/selection score | 仅预注册 proposal/selection 或 open diagnostics；无真实 measured denominator时不得称功能改善 |
| `SealedFinalEvaluator` | isolated final labels | 预注册 aggregate metrics only | 一次性 final；绝不作为 training reward 或 selector |

若 generator 与 scorer 共享 encoder，PR1 必须冻结共享层、heads、各 loss weight、gradient stop/flow、更新顺序与 task sampling；否则明确分开训练。`full method` 必须指明是哪一行或哪种预注册组合，不得把 flow objective 当 outcome score，也不得在 T5-Gen-Reconstruct 与 T5-Rank/Closed-Select 中择优选一个作为“方法获胜”。

## 10.2 Sequence scope 与长度

- full sequence 永久保留；
- primary edit model 使用可验证、带坐标的最大 640-nt editable window；
- window 的选择只能依赖 source-side 信息、合同预先给定的 editable locus 或冻结的 source-only tiling；不得用 target/candidate 差异定位窗口，否则构成 target leakage；
- 超过 640 nt 的完整 UTR 使用 source-only overlapping tiling 与冻结 aggregation，或进入 `LONG_SEQUENCE_UNSUPPORTED` 分层；不得在训练时临时用 edit-centered crop；
- 必须保存 full-sequence hash、original length、window start/end 和 editable mask；
- 共同支持对照另设 ≤256 nt slice，但不得把该 slice 冒充全数据；
- 任何 cap/drop/truncate 必须在 D1 冻结，训练时不得临时丢记录。

## 10.3 Context 表示

assay、cell、endpoint、promoter、reporter/cargo、chemistry、timepoint、study 使用 learned categorical embeddings、独立 missing token 和 missing mask。禁止把多个类别 hash 成一个连续标量。

只有至少两个可训练 level、每 level 有足够独立 parent、并具有 held-out-level 评测的 context 才能进入 confirmatory conditioning；否则只用于 stratification。

## 10.4 External-backbone ladder；禁止项目侧无标签预训练

v1 不建立 Track U，也不训练新的无标签 foundation encoder。最多比较：

1. `from_scratch_E`：仅使用 Track E train partition；
2. `supervised_F_to_E`：先在与 E/final 隔离的 Track F 功能标签上做 supervised auxiliary training，再在 Track E train partition 适配；
3. `external_general_backbone`：最多一个可审计的 general/full-mRNA 或 general-RNA backbone；
4. `external_region_specialist`：5′与3′各最多一个 specialist sensitivity，不进入完整笛卡尔积。

每个 external backbone 必须审计权重与许可、pretraining corpus 可见范围、benchmark source/candidate exact/near overlap、GSE246381 的 foundation sequence/label overlap、最大长度、参数量、tokens、FLOPs 和 adaptation 方式。

若没有 external backbone 能通过 exposure gate，primary 自动回退到 `from_scratch_E` 或预注册的 `supervised_F_to_E`；不得临时引入无标签语料或恢复 Track U。Track F auxiliary training 只能使用 development-side labels，并与 Track E final 在 sequence cluster、parent/family、study/library 和 label lineage 上隔离；不得称无标签预训练或新 foundation model。

`supervised_F_to_E` 不是一个空路线名。PR1 前必须逐 endpoint/assay 冻结：F auxiliary task 与 head、study 内 normalization、loss/endpoint weights、转移哪些 encoder layers、是否丢弃 F heads、E adaptation schedule，以及 exact/near/family/study/label-lineage isolation。强制比较 `E_only`、`F_to_E`、no-study-ID 与 label/study-permutation negative controls，并报告 negative transfer；不同 assay label 不能未经映射共用同一标量 head。上述合同缺失时只能运行 `from_scratch_E`。

## 10.5 Region variants

实现必须互斥：

- `fully_shared`：无 region input，无 region-specific parameter；
- `shared_plus_region_adapter`：共享主干 + 明确 region-specific learned adapter；
- `independent`：5′/3′独立参数；
- `wrong_region_control`：错误 region label，且匹配 length/study/action distribution。

如果 no-adapter 仍向 head 输入 region one-hot，则不能称 fully shared。当前 2×4 operation gate 只能称轻量 region gate，不能代表完整 adapter 假设。

H5 的唯一 confirmatory component/task 固定为 `EditFlowGenerator` 的 common-support endpoint reconstruction，不在 OutcomeScorer/ranking 与 generator 之间择优；OutcomeScorer region variant 仅为 secondary。base 固定为 `UTR_ADAPTED_FRESH_Q1`，三种 variant 除 region parameterization 外使用完全相同的 backbone revision、initialization policy、alignment proposal、context features、action/length/budget policy 与 loss。external region-specialist backbone 只能作 sensitivity，不能成为 H5 base。

三种 variant 使用完全相同的 common-support train/dev/test record IDs、task/label semantics、normalization、trial/config/seed 数、early-stopping policy、外部 candidate/query/edit budget 与 final evaluator；5′和3′分别计算效应，禁止用 pooled 平均掩盖任一地区失败。primary comparison 是 `shared_plus_region_adapter` 对 `fully_shared` 的 superiority；`independent` 只承担预注册 non-inferiority comparator。H5 必须按 intersection 同时通过二者并纳入 Holm family。PR1 在任何 model result 可见前写明 `Δ_NI` 的数值和与 generator primary metric 完全一致的单位；证据必须来自该 metric 的 cluster-aware precision、oracle/retrieval ceiling、可辨识范围与 smallest practically relevant recovery/similarity effect。只有存在真正重复的 endpoint/generation-evaluation measurement 且证明其 reliability 与该 metric 同尺度时，replicate reliability 才可作为补充；没有可辩护 margin 时 H5 不具 confirmatory 资格。

主分析匹配每个 variant 的 optimizer updates、每个地区看到的 training examples/tokens、evaluation 次数与随机种子；同时报告参数量、FLOPs、显存与 wall time。另做 parameter/FLOP-matched sensitivity，并给 `fully_shared` 加 parameter-matched placebo module 以区分 adapter 语义与纯容量。`independent` 的两套参数不能获得额外 trial 或每地区额外数据轮次；任何不可匹配资源必须进入 ledger，不得称 fully matched。

## 10.6 Adapter 命名

只有实际修改预训练权重矩阵的低秩更新可称 LoRA。encoder 输出后的 rank-k residual bottleneck 必须称 `low-rank residual adapter`。合同、代码、配置和论文命名必须一致。

## 10.7 上游方法继承与辅助变量分层

本模型继承 Edit Flows 的 sequence-space CTMC、INS/DEL/SUB rate parameterization、auxiliary alignment process、auxiliary-process marginalization 和 Bregman flow-matching training construction。上述内容必须引用 Edit Flows，不作为本项目原创。

为避免把不同随机对象混成一个“path”，合同固定区分：

- `alignment z`：source 与 endpoint 的 gap placement 和 token correspondence；
- `switch/order variables τ`：给定 alignment 后各 edit switch 的时间或顺序；
- `CTMC trajectory X_[0,1]`：模型采样时实际经历的随机 sequence states；
- `non-minimal detour d`：在 source→endpoint 最短变换之外额外插入后删除、删除后恢复等回路；v1 primary 默认不启用；
- `endpoint y`：历史实验观测到的 candidate sequence。

实验没有观测 `z`、`τ`、`d` 或 CTMC trajectory。任何算法生成的 alignment、edit script 或 intermediate state 都不是生物轨迹真值。split 必须先冻结，再在各 split 内独立生成且不得复用 path objects；必须审计 train↔dev/final endpoint collision 以及实际 frozen/materialized/sampled intermediate states 的 collision，按预注册规则删训练侧对象或阻断。不得声称所有理论上可能的 latent states 均已证明 non-overlap。

## 10.8 方法归因矩阵、alignment-proposal 与 multi-alignment objective

不得把“上游复现/UTR 适配”与“alignment proposal/样本数”混在同一个比较轴。正式实验固定为两个顺序消融层：

| 条件 | UTR architecture/adaptation | alignment construction | 唯一允许回答的问题 |
|---|---|---|---|
| `UPSTREAM_REPRO_CHECK` | 在上游官方任务、数据切片或官方 fixtures 上复核代码来源、目标与采样行为；不要求与 UTR endpoint task 同接口 | 上游 native | 本项目是否忠实理解/移植上游；只属 provenance/engineering check |
| `BASIC_PAIR_EF_NATIVE` | 为 UTR endpoint pair task 做最小 I/O/长度兼容；`x_0=s`，不保留项目专属 persistent source/current 双表示，不加入 region/context adapter 或项目专属 action-prior；所有最小变化逐项登记 | 上游 native construction | 同任务下的 basic Edit Flow baseline，而不是“原论文原封不动复现” |
| `UTR_ADAPTED_NATIVE` | 加入冻结的 UTR persistent source/current interface、action masks/budgets、region/context interface；每个新增组件另有单因素 ablation，并以 dummy/placebo parameters 做 capacity control | 与 `BASIC_PAIR_EF_NATIVE` 相同的 native construction | UTR adaptation bundle 及各组件是否有增量价值 |
| `UTR_ADAPTED_CANONICAL` | 与 `UTR_ADAPTED_NATIVE` 完全相同 | 每个 endpoint pair 使用冻结 deterministic canonical alignment | canonical choice 的敏感性 |
| `UTR_ADAPTED_FRESH_Q1` | 与 `UTR_ADAPTED_NATIVE` 完全相同 | 每次访问时从冻结 proposal `q(z\|s,y)` 独立采样一条 alignment | proposal randomization 相对 canonical 的影响 |
| `UTR_ADAPTED_MULTI_QK` | 与 `UTR_ADAPTED_NATIVE` 完全相同 | 从同一 `q` 有放回 i.i.d. 采样 `K_align` 次，保留重复样本的 multiplicity，做 pair-normalized Monte Carlo 平均 | 同一目标的 multi-sample estimator 方差与质量—计算权衡 |

`UPSTREAM_REPRO_CHECK` 只验证上游忠实度；`BASIC_PAIR_EF_NATIVE→UTR_ADAPTED_NATIVE` 才是论文的同任务 adaptation 轴；`UTR_ADAPTED_CANONICAL→UTR_ADAPTED_FRESH_Q1→UTR_ADAPTED_MULTI_QK` 是 alignment 轴。不得用 `native/generic` 这一含混标签跳过最小 task adaptation 清单，也不得用跨轴比较把 UTR architecture、proposal randomization和 K 的效果归给同一个创新点。

对固定 proposal `q`：

\[
\widehat{\mathcal L}_{K}(s,y)
=
\frac{1}{K}
\sum_{j=1}^{K}
\mathcal L_{\mathrm{EF}}(s,y,z_j),
\qquad z_j\overset{\mathrm{i.i.d.}}{\sim}q(z\mid s,y).
\]

该目标估计的是给定 alignment proposal 下的 Edit Flow/Bregman objective。它保留为本项目的技术核心与经验假设，但不得称为新的 marginalization theorem、exact endpoint likelihood、真实轨迹边缘似然或对 Edit Flows 上游思想的重新发明。

正式比较必须满足：

1. `NATIVE` 必须绑定 Edit Flows paper revision、official code commit、原任务/`x_0` regime、alignment algorithm/cost、random-vs-optimal construction、default-vs-localized auxiliary process；如果 `UTR_ADAPTED_FRESH_Q1` 的 q 与上游逐迭代 native sampling 实为同分布，则合并条件，不得当作两个 baseline；
2. `q` 的支持、概率或采样算法、alignment cost、seed 与最大尝试次数冻结；primary 为有放回采样，任何 unique sensitivity 另列规则；
3. `UTR_ADAPTED_FRESH_Q1` 与 `UTR_ADAPTED_MULTI_QK` 至少做两类互补比较：`DRAW_MATCHED` 固定总 alignment draws，允许 optimizer updates 不同；`UPDATE_MATCHED` 固定 optimizer updates，明确报告增加的 draws、FLOPs、显存和 wall time；另报告 quality-vs-time/FLOPs Pareto curve；
4. 不得声称在一般情形下同时精确匹配 alignment draws、optimizer updates 与 FLOPs；只有实测三者确实相等并给出 ledger 时，才可描述为 triple-matched；
5. gradient-variance 比较必须在同一冻结模型状态或同一 early checkpoint、相同 endpoint minibatch 与受控随机数下重复估计，报告 estimator、重复次数、CI 和 K；不得用最终指标波动替代梯度方差；
6. operation-order/switch-clock sampling 与 alignment sampling 分别记录，不得合称一个 path sample；
7. `K_align=1` 与 `K_align=8` 是不可筛除的 confirmatory anchors，必须完成冻结 seeds、H1C-M 与 H1C-P；`K_align∈{4,16}` 只作预注册 secondary sensitivity，可在 development 按冻结规则筛选，但不得替换 K=8 primary；
8. duplicates 不是“采样失败”：实现可缓存相同 alignment 的 loss，但必须按出现次数 `count(z)/K` 加权；若 `q` 没有合法支持或无法完成 K 次有效 draw，该 pair/batch 必须 fail closed，不得把较小 `K_eff` 冒充 K；
9. unique-without-replacement 只能作为单独 sensitivity：必须定义 inclusion probability 与正确 importance/inclusion weight；否则明确它改变 proposal/estimand，不得把差异归因于 sample count；
10. 每个 endpoint pair 在总 loss 中权重相同，不得因 alignment 支持更多而获得更高权重；
11. `path-consistency` 只有在给出数学公式、非平凡单元测试、明确匹配对象，并证明不是 Edit Flows 已有 localized auxiliary CTMC/edit propagation 的换名后才能启用，否则从 v1 objective 删除；
12. normalized latent-path likelihood/bound 永久为 exploratory，除非规范化概率、短序列穷举、ESS、偏差和数值稳定性全部验证。

因为 `UTR_ADAPTED_FRESH_Q1` 与 `UTR_ADAPTED_MULTI_QK` 对固定 `q` 估计同一个期望目标，K 增大本身不构成新的 endpoint objective。若 multi-alignment 未改善预注册的质量—计算前沿，只能报告方差/稳定性或 null result；整个 Edit-Flows-derived UTR reference method 和 benchmark 仍按真实结果报告。

### 10.8.1 H1C-M 冻结梯度估计量

H1C-M 必须在任何 H1C 结果可见前把以下对象写入 `h1c_estimator_contract.yaml` 并 hash 冻结：

1. checkpoint 是每个预注册 training seed 在 `UTR_ADAPTED_FRESH_Q1` 总 update budget 恰好 10% 处的状态；不得按 loss 或稳定性挑 checkpoint；若某 seed 在此前失败，该 seed 按失败处理而不是替换；
2. 每个 seed 使用相同的 32 个 development minibatch IDs，按 study/library 与 endpoint minimum edit-distance stratum 确定性分层选取；每个 minibatch 必须只属于一个 outer study/library cluster。若 batching 不能保持同质，必须保存 per-example/per-cluster gradient 后再聚合，禁止给 mixed-study batch 任意指定一个 cluster。所有 K 使用相同 IDs、相同 endpoint weights、相同非 alignment 随机数和相同模型状态；至少 10 个独立 outer clusters，否则 H1C-M 只能 descriptive；
3. primary gradient vector 为 action-rate output head 的全部可训练参数；最后一个共享 encoder block 仅为预注册 secondary，不得替换 primary；
4. 对 `K∈{1,4,8,16}` 各做 `R=64` 个独立 alignment-draw replicates，均从同一个冻结 `q` 有放回 i.i.d. 采样并保留 multiplicity；
5. 令 `g(z)=∇_θ L_EF(s,y,z)`。只有 alignment support 可完整枚举、冻结 proposal 的 normalized probability `q(z|s,y)` 可计算且概率和通过数值检查时，exact reference 才定义为 `g_ref=Σ_z q(z|s,y)g(z)`；严禁对 unique alignments 做无权均匀平均。不能满足时，对同一 `q` 使用与各 `g_K` 独立、但在同一 seed/minibatch/replicate 的各 K 之间共用的一组 `K_ref=256` Monte Carlo reference；
6. 在任何 gradient/K result 可见前，对每个 pair 冻结 positive-probability alignment support size、proposal entropy、effective support size 与 `p_diff=P(z_1≠z_2)`。若 q 概率可得则精确计算；若只有 sampler，则用预先 hash 的 1,024 draws 估计并给 one-sided 95% CI。primary ambiguity pool 要求每 pair support≥2 且 `p_diff` 的 point estimate（exact q）或 CI 下界（sampler-only q）>0.05，并至少有 500 个独立 endpoint pairs、10 个 study/library outer clusters；否则 H1C-M=`NO_ALIGNMENT_AMBIGUITY/H1C_M_NOT_ELIGIBLE`，只报告 support/entropy/raw-MSE diagnostics；
7. 以浮点精度审计和 identical-alignment null control，在任何 K result 可见前冻结 `τ_grad>0` 与 normalized-MSE noise floor `τ_nmse>0`。仅 `||g_ref||²>τ_grad` 的预先确定 units 进入 normalized primary，且至少 80% units 与至少 10 个 outer clusters 必须合格；否则 H1C-M=`LOW_REFERENCE_SIGNAL`。fixed primary units 不得再按 K1/K8 结果筛选；若不足 80% outer clusters 的 cluster-level `NMSE_1>τ_nmse`，则 H1C-M=`NO_IDENTIFIABLE_ALIGNMENT_VARIANCE`。这两种失败都只报告 support、entropy 与全部 units 的 raw MSE；
8. 预先 hash 选定 10% minibatch 做 `K_ref=512` audit。令 `C_ref=log[(NMSE_8+τ_nmse)/(NMSE_1+τ_nmse)]`，`C_256`、`C_512` 分别使用 256/512 reference，并定义 `δ_ref=|C_512-C_256|/max(|C_512|,|C_256|,10^-6)`；若 `δ_ref>0.10` 或二者方向不同，则 H1C-M=`REFERENCE_NOT_CONVERGED`，不得扩大采样后追救 confirmatory result；
9. 对 fixed eligible mask 内的 seed `s`、minibatch/cluster unit `b`、replicate `r` 定义；`S/B/R` 分别为进入固定 primary 的 seed、unit 与 replicate 数

\[
\operatorname{NMSE}_K
=
\frac{1}{SBR}
\sum_{s,b,r}
\frac{\lVert g^{(s,b,r)}_K-g^{(s,b,r)}_{\mathrm{ref}}\rVert_2^2}
{\lVert g^{(s,b,r)}_{\mathrm{ref}}\rVert_2^2}.
\]

primary contrast 固定为预先使用 null-control floor 的 `C=log[(NMSE_8+τ_nmse)/(NMSE_1+τ_nmse)]`；禁止在看到结果后临时增加 epsilon。所有 units 同时报告未归一化 `MSE_K=mean||g_K-g_ref||²`。保持 K 间 pairing，以 study/library 为外层 cluster、seed 与 homogeneous-minibatch/per-cluster gradient 为内层重采样单元，做 2,000 次固定-seed bootstrap。只有 alignment ambiguity、identifiable variance、reference signal、reference convergence gates 全部 PASS，且 C 的 two-sided 95% CI 上界<0，才通过 H1C-M。K=4、16 和 encoder-block 结果全部为 secondary；不得用其中较优者替换 K=8。报告 gradient dimension、checkpoint hashes、minibatch/cluster IDs、support/entropy/p_diff、`τ_grad/τ_nmse`、每个 K 的 draws、FLOPs、显存和 wall time。

### 10.8.2 H1C-P 质量—计算判据

H1C-P 只在一次性 `sealed_final_v1` 上形成 confirmatory 结论；development curves 只能选工程范围，不能 PASS H1C-P。PR1 必须在任何 method result 可见前绑定 T5-Gen-Reconstruct 已通过非退化 gate 后选择的同一个 `GEN_EXACT_CONFIRMATORY` 或 `GEN_SIMILARITY_CONFIRMATORY` primary metric，并冻结共同 FLOP grid、插值规则、最大共同预算、`Δ_quality` 的数值和与该 metric 完全一致的单位。margin 证据必须来自该 metric 的 cluster-aware precision、oracle/retrieval ceiling、可辨识范围与 smallest practically relevant recovery/similarity effect；只有重复 endpoint/generation-evaluation measurement 被证明与该 metric 同尺度时才可补充 replicate reliability，没有可辩护 margin则 H1C-P 不具 confirmatory 资格。K=1/K=8 的全部 frozen configs、seeds、checkpoints 和 budget grid 必须由同一个一次性 final evaluator command 原子评测并只返回预注册 aggregates。

H1C-P 的 primary resource axis 为 cold-start 实测 FLOPs；ledger 必须包括 alignment proposal 构造与 draw、cache 首次构造、forward、backward、optimizer 和冻结评估。缓存命中/摊销另报 warm-cache sensitivity，不得从 primary FLOPs 删除。alignment draws、optimizer updates 与 wall time 是预注册 secondary axes。primary statistic 为共同 `log(FLOPs)` 区间的 normalized quality-AUC 差 `AUC_K8-AUC_K1`，要求 paired group-aware 95% CI 下界>0；在最大共同 FLOPs 处另要求 paired quality difference `Q_K8-Q_K1` 的 95% CI 下界>`-Δ_quality`。两项按 intersection 同时成立才 PASS。曲线不得外推，不得把不同 axis 上各自最有利的点拼成一个 Pareto 胜利；draw-matched、update-matched 与 wall-time 曲线必须完整报告。

## 10.9 STOP 条件

公开 endpoint-only 数据没有真实“何时停止编辑”的观测时间；identity/no-edit 只能校准 no-op specificity，不能提供 termination time。当前 STOP head 未进入主训练损失，也没有可识别的 target，因此 v1 状态固定为 `STOP_DISABLED_FIXED_BUDGET_PRIMARY`，不进入 PR2 trial grid、headline 或一次性 final。

未来若另立 v1.1 exploratory 合同，必须在首次 trial 前给出 synthetic computational stop label 的逐状态生成公式、目标可辨识性说明、脚本 hash、Brier/calibration metric、与 fixed-horizon 的 matched comparison 和明确的失败规则；不能在 five-seed 结果后回写 label/branch。即使通过也只能称 computational termination，不称 biological stopping 或真实编辑动力学。

## 10.10 性能与可扩展性

formal trainer 必须使用 batching/DataLoader、长度分桶、缓存可缓存的 source features，并避免为每个 record 串行枚举全部合法动作。正式数据规模下必须报告 throughput、GPU memory、candidate/sec 和 wall-clock。max-length 冲突、CPU fallback 或 record drop 均 fail closed。

## 10.11 Continuous-time 与 sampler 主张边界

rate field 与总 hazard 必须在 MK0-R 中重新绑定 v3 action/state schema。当前已验证的 first-order constrained sampler 可作为工程实现，但不得称 exact Gillespie 或 exact CTMC endpoint sampler。

只有在等待时间、总 hazard、单事件选择、边界条件和小状态空间分布均通过数值对照后，才可解锁 `exact_ctmc_sampler` 表述。否则论文固定使用 `continuous-time action-rate model with a first-order constrained sampler`。

---

# 11. Baselines、matched budget 与 evaluator

## 11.1 最低 baseline 集

### 无学习/简单监督

- identity/no-edit；
- random legal/edit-distance-matched edit；
- empirical action/source-frequency；
- k-mer/linear；
- tree-based；
- small CNN/Transformer scorer；
- candidate-only 与 source-candidate delta scorer。

### 搜索

- scorer + exact enumeration（空间可穷举时）；
- beam search；
- genetic/CEM；
- AdaBeam/NucleoBench-style search 或可复现同类实现。

### 生成

- autoregressive editor；
- masked/denoising editor；
- diffusion 或最近邻可适配方法；
- `UPSTREAM_REPRO_CHECK`（仅 provenance/engineering，不作为同接口性能 baseline）；
- `BASIC_PAIR_EF_NATIVE`；
- `UTR_ADAPTED_NATIVE`；
- `UTR_ADAPTED_CANONICAL`；
- `UTR_ADAPTED_FRESH_Q1`；
- `UTR_ADAPTED_MULTI_QK`。

任何无法合法取得代码/权重/许可的 published model 可作为 method-prior comparator，不得伪装 exact-interface executable baseline。

## 11.2 Matched-budget 固定字段

只在同一 task/estimand 内做 matched-budget 比较；T5-Gen-Reconstruct editors 与 T5-Rank/Closed-Select search 不相互宣布胜负。每项比较记录：

```yaml
sources: <same IDs>
task_estimand: <same frozen task and ground truth>
candidates_per_source: <B>
max_edit_distance: <k>
action_types: <same support>
length_range: <same common support>
predictor_calls: <same or reported curve>
wall_clock_or_flops: <same or reported curve>
external_outcome_budget: <same edit/candidate/query support>
native_internal_horizon: <reported, not forcibly identical when semantics differ>
native_stopping_policy: <reported; fixed external budget remains comparable>
random_seeds: <same count>
```

匹配的是外部可比较资源和结果支持，而不是强迫不同算法使用语义不等价的内部 horizon/stopping。必须同时报告 quality-vs-query、quality-vs-time、quality-vs-FLOPs 和 amortization break-even；不得只比较单个有利预算点。

## 11.3 Evaluator 边界

- held-out flow/Bregman objective 只能称 surrogate/objective，除非规范化 likelihood 被验证；
- functional score 使用冻结、与 generator 训练 reward 分离的 evaluator；
- 多个 predictor 的 agreement/disagreement 作为 open-generation diagnostics，不是真实 ground truth；
- closed measured-pool 使用真实历史测量，但严格按 source/study macro；
- final evaluator 版本、代码 hash、数据 hash 和 output hash 全绑定。

---

# 12. 统计、负对照与三轮预注册

## 12.1 统计规则

1. 每个 confirmatory task 仅一个 primary metric；
2. source 为内层、study/library 为外层 cluster bootstrap；只有至少 10 个独立外层 clusters 时才把外层 bootstrap 作为主要不确定性估计；不足 10 时改用逐 study/library 结果、leave-one-study-out sensitivity 与 parent-level bootstrap，不作 study-population 泛化；默认 2,000 次并固定 seed；
3. 报告 paired effect size 与 two-sided 95% CI；superiority 的默认必要条件为方向正确且 CI 下界>0，除非 PR1 在 final 前冻结更严格规则；seed 不作为生物样本；
4. confirmatory hypothesis family 使用 Holm 校正；
5. 5′、3′分别报告，pooled 仅 secondary；
6. 所有 denominator、missingness、attrition 和失败 seed 进入报告；
7. 不以 row-level 随机 bootstrap 支撑 headline；
8. 阈值在 final 之前冻结，final 之后不可追改。

## 12.2 强制负对照

- no-source；
- candidate-only/current-only；
- source/candidate swap；
- source permutation；
- library/study-ID shortcut；
- condition permutation；
- region permutation；
- matched wrong-region；
- exposure-aware vs potentially exposed backbone；
- `UTR_ADAPTED_CANONICAL/FRESH_Q1/MULTI_QK` alignment-estimator controls；
- `BASIC_PAIR_EF_NATIVE` vs `UTR_ADAPTED_NATIVE` + component/capacity-placebo ablations；
- simple scorer+search。

source 与 current 在初始状态相同的例子不能单独证明 source conditioning；no-source 必须在中间状态或 endpoint任务上评测。

## 12.3 PR1：数据、任务与 final 封存

在任何 formal method training 前冻结：schema、dataset version、role、source observation join、exposure、license、candidate pools、splits、path proposal、budgets、metrics、baseline、bootstrap、seeds、claims 和 stop rules。

PR1 未通过时，只允许数据修复、read-only audit 和 development smoke。

## 12.4 PR2：三轮开发

三轮顺序固定：

### Round 1 — corrected base

- 修正 context embeddings、长度策略、batching、fixed budget、region variants；
- 按 task→model matrix 分开比较：T5-Gen-Reconstruct 用 AR/denoising/diffusion/`BASIC_PAIR_EF_NATIVE`；T5-Rank/T3-Effect 用 simple/outcome scorers 与 search；另比较 `from_scratch_E` 和通过完整适配合同的 `supervised_F_to_E`；
- 不使用 STOP；
- 每类最多 12 个 one-seed screening configs；top 2 进入 5-seed confirmation。

### Round 2 — alignment / estimator robustness

- `UPSTREAM_REPRO_CHECK` 必须先独立通过；正式同任务比较为 `BASIC_PAIR_EF_NATIVE`、`UTR_ADAPTED_NATIVE`、`UTR_ADAPTED_CANONICAL`、`UTR_ADAPTED_FRESH_Q1`、`UTR_ADAPTED_MULTI_QK`；`K_align∈{1,4,8,16}`；
- v1 删除未定义的 `path-consistency on/off` 占位；只有未来给出公式/测试并证明不同于上游 localized propagation 与 Tree-Conditioned consistency 后，才能经 decision log 新增；
- deterministic canonical、paper/code-pinned native proposal 与冻结 bounded alternative proposal 分开；若 native 与 fresh proposal 同分布则合并条件，不得伪造两个 baseline；
- 执行 `DRAW_MATCHED`、`UPDATE_MATCHED` 与 quality-vs-time/FLOPs Pareto 三类互补比较；pair weights 相同，不要求一般不可能的三重精确匹配；
- K=1/K=8 confirmatory anchors 不参与 one-seed 淘汰，必须无条件完成冻结的 seeds、10% checkpoints、FLOP grid、H1C-M 与一次性 final H1C-P；
- 每类最多 12 个 one-seed configs；top-2 screening 只适用于非-anchor secondary configs，包括 K=4/K=16 与其他 development sensitivities，且不能占用或替代 anchor 的 5-seed/final 预算。

### Round 3 — limited combination

- 只组合 Round 1/2 已独立通过的组件；
- foundation、region adapter、approved context 分别做增量验证；
- STOP 在 v1 固定 disabled，不进入组合；
- 最多 12 个 one-seed configs；最终只冻结 1 个 full model 和 1 个 simple fallback。

Round 3 结束时必须把 `FULL_GENERATOR_FINAL_ALIAS` 原子解析为唯一 model class、完整 config、checkpoint lineage、foundation revision、region variant、alignment variant/K、seed set 与 SHA256；alias 在一次性 final 中对所有 baseline、budget 和 metric 保持不变。simple fallback 使用独立 alias，不得在 final 后替换 full alias 或在不同表格中让同一 alias 指向不同配置。

三轮的 config budget、screening/confirmation 规则和进入下一轮的条件在 Round 1 开始前共同冻结，不得给拟议方法额外 trial 数。每轮使用 immutable round registry；所有 trial 写 append-only global registry，包含假设、配置、seed、输入 hash、指标、成本、失败原因与保留/拒绝决定。不得隐藏失败、重用失败 run root 或用 final 结果选择配置。

## 12.5 PR3：一次性 final

模型、fallback、seeds、环境、candidate budget、evaluator command 和 output schema 全部冻结后，final labels 只打开一次。

结果使用三条正交状态，不再压成一个含混标签：

```yaml
benchmark_status: <POSITIVE|LIMITED|NEGATIVE|INVALID>
method_status: <POSITIVE|EQUIVALENT|INCONCLUSIVE|NEGATIVE|INVALID>
region_scope: <FIVE_PRIME_ONLY|FIVE_PRIMARY_THREE_QUALIFIED|REGION_LIMITED|INVALID>
```

只有三条状态均由预注册 final artifacts 支撑时才能进入 claim matrix；benchmark positive 不自动使 method positive，5′ positive 不自动变为 UTR-general。

`EQUIVALENT` 只在 PR1 冻结 equivalence margin、双侧规则并通过后使用；CI 跨越 superiority 与 equivalence 边界时必须为 `INCONCLUSIVE`。seed aggregation、failed-seed handling、confirmatory family 与 Holm 校正范围均在 final 前冻结。

用户已选择继续迭代方法，但该意图不得污染 v1 final。若 v1 方法未赢，可在公开报告 v1 真实结果后，于 v1.1 使用新数据与新 final 继续；不得反复查看 v1 final 调到胜出。

---

# 13. G0–G7 数据与 benchmark 重新关闭门槛

## G0：Input Provenance Freeze

每个阶段启动前必须创建 immutable `INPUT_MANIFEST.json`，绑定：Git HEAD/实际 source tree、合同/config、parser/builder/auditor/test、每个 raw 文件路径/大小/mtime/SHA256、所有已存在的上游 canonical/ledger/split/FM inputs，以及 runtime/environment。G0 只冻结输入，不预先声称本阶段 output、report、STATUS 或 DONE 已存在。

任何 input 更新使基于它的旧报告自动 `STALE_INVALIDATED`。未排序 glob、ambient install、stale build/lib、dirty worktree、输入 checksum 缺失或不匹配均 FAIL。

本阶段生成的 canonical/report/data card 等输出由 G7 的 `OUTPUT_MANIFEST.json`、`SHA256SUMS`、`STATUS.json` 和 finalizer 绑定。`DONE` 只能在 G7 成功后生成，避免用待生成 output 反向满足 G0。

## G1：Canonical Schema and Scope

必须：

- active `schemas/v3_1/` 精确 21-schema filename/$id/version/hash set 与每个 positive/negative fixture 100% PASS，formal 禁止 legacy inference；
- 缺 `scientific_track/relation_acceptance_status/relation_type/effect_evidence/landscape_role/future_use_role`=0；formal 禁止通过 legacy `record_type` fallback 推断 E relation；accepted relation 空 source/candidate=0；
- 每条记录有 sequence scope、source file/hash、context、transform、group、label status；
- window/insert 有 original length、full hash、坐标、scaffold、editable mask；
- CDS/full-length 与 UTR 主 track 隔离；
- GSE207584 的 source 与 candidate 两端 sequence 均为空/未保留；必须两端分别恢复并验证、重分类或 quarantine，不能只检查 candidate。

## G2：Cleaning Conservation and Quarantine

每 accession/source file 必须分别满足 §6.4 的 raw-unit、declared-relation、measurement-join 与 global unique-object 方程；reason 计数严格闭合；accepted candidate↔formal pair 的 FK 与 identity payload mismatch=0；canonical sequence/observation/relation 与 transformation edge 均无 orphan；预期一对一 join 不得 one-to-many，many-to-one 必须列全原始成员；非法字符不静默删除；label missingness 显式；identity/no-edit 保留；biological/technical replicate、barcode/UMI 与 group 分层。task-specific eligibility-cell 方程由 G5 在 frozen Task/Split/Application registries 上独立闭合，不得用 assignment rows 回填 G2 的 unique-object 等式。任一不守恒即 FAIL。

## G3A：Project/Label Exposure and Sealed-Access Integrity

- measurement、project sequence exposure、project label exposure 与 future use 分字段；
- 合同与 ledger 逐 accession diff=0；
- GSE246381 owner-confirmed prior sequence/label training、tuning、model-selection、error-analysis 与 human-view use=`NONE_CONFIRMED`、pipeline sequence/label materialization=`PRESENT`，不得出现 E4X/historically-exposed；其新 sealed-final role 下 train/hyperparameter/pre-final-error-analysis count=0；
- GSE114002 measurement 与 historical sequence exposure 分开；
- ENCODE raw observational 与 processed paired 资产分开；
- project prior-use UNKNOWN 不得 PASS；
- D1 阶段不写 accession/object 级 scalar overlap；只允许 `foundation_overlap_requirement=REQUIRED_FM0_A` 与 audit status=`DEFERRED_TO_FM0_A`，逐 checkpoint ledger 尚未产出。这不是 G3B PASS，也不阻断 G3A。

## G3B：External-Foundation Overlap

- 只在 frozen D1 sequence clusters 后由 FM0-A 执行；
- GSE246381 与所有 benchmark clusters 对实际候选 external checkpoints 的 exact/near sequence 与可审计 label-lineage overlap 分开；
- `from_scratch_E` 记 `NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS`；`supervised_F_to_E` 另审计 F↔E final lineage；
- external checkpoint 的 corpus/overlap 为 `UNKNOWN/PENDING` 时，该 checkpoint 与相应 final/generalization claim 必须 `INELIGIBLE`；不得把 UNKNOWN 记 PASS；
- B0/G7 前必须形成 G3B PASS，或把所有未闭合 external checkpoints 明确排除并绑定 fallback。

## G4：Data Recovery and Exhaustive Inventory

至少完成：

- `priority_snapshot_v3_1.yaml` 与 §7.1.1 显式列出的 P0/P1/P2-acquisition/reference/analysis-only/search-negative 集合逐项 set equality=TRUE；

- GSE114002 10 文件全 inventory、修复文件实际摄入或明确排除、identity 恢复；
- GSE145046 全 condition/replicate inventory 与 deterministic join；
- GSE217518 各 denominator、5′/3′ attrition、full/window 映射和 assay join 闭环；
- §7.1 列出的全部 22 个 DATA-P0 asset groups 均接入或有证据排除；点名包括 GSE232571/572、fast-UTR、GSE288185、DART、N-zip、GSE330741、GSE261709、GSE298114 与 PTRE，但点名列表不替代冻结全集；
- GSE149487 paired eligibility；
- GSE207584 FASTA 恢复或 auxiliary quarantine；
- 所有 P0 资产达到 `canonical_status=ACCEPTED` 或 `canonical_status=EXCLUDED_WITH_EVIDENCE`；
- 所有 P1 资产完成状态、mapping feasibility、许可和 lineage 审计；可确定性恢复且允许使用者必须进入 E/F 相应轨；无标签 reference-only 资产不属于该 gate。

## G5：Benchmark Coverage, Isolation and Leakage

- split 前验证 21-schema set、§5.7 的 12-task/10-split expected sets、descriptor/allowlist hashes、120-row immutable `TaskSplitContractMatrix`、120-row B0 effective decisions 与 DiagnosticRegistry，再冻结 ordinary/restricted `ELIGIBILITY_MANIFEST.jsonl`、`TASK_ELIGIBILITY_UNIVERSE.jsonl`、role ledger/projection 和 `SPLIT_ASSIGNMENTS.jsonl`；E task-cell object 只能是 accepted `pair_id`，F task-cell object 只能是 label-complete `observation_id`；未接受 relation candidates 只在 lifecycle 层守恒；
- `ELIGIBILITY_MANIFEST` 只含每 global object×run 一条 EligibilityRecord，不含 task/split；TaskEligibilityCell 通过 FK 连接该 record，assigned cell 与 SplitAssignment 双向一一对应，partition ID/role 必须来自 SplitRegistry；
- global unique-object 层每个 E pair/F observation 只计一次；task-cell 层严格按 §6.4 唯一生成规则闭合。每个 expected cell 必须进入 `ASSIGNED_TO_SPLIT`、`INELIGIBLE_WITH_REASON` 或 `PENDING_WITH_REASON`；formal global 与每个 applicable task/split-contract row 的 pending=0；不得把仍 eligible 的对象放入 exclusion、删 expected cell 或用 `NOT_APPLICABLE` cell 制造守恒；
- expected-set/semantic/allowlist/matrix/global-eligibility/cell/assignment FK 与 uniqueness counters 全为 0；`active_object_without_applicable_matrix_row=0`；§5.7 global-role×partition-role 与 eligibility×partition-role conflict=0；
- ordinary `data/v3_1/benchmark/RELATION_ROLE_TRANSITIONS.jsonl` 与 restricted mirror 分别重放 allowed matrix、hash chain、effective-role projection 与 frozen canonical base role；所有 transition integrity counters 和 candidate/pair effective-role mismatch=0；ordinary/restricted ledger hashes 进入 B0/G7 manifest；
- GSE246381 row-level EligibilityRecord/cell/role/split 只在 restricted store；ordinary row-level artifacts 的 GSE count=0，ordinary commitments 与 restricted manifests/access-log hash 联合闭合，dual-store overlap/leakage/missing commitment counters=0；
- 当前漏出的 GSE217518 3,564 successfully reconstructed design/workbook rows 与 GSE149487 448 current pair-candidate records 必须重新裁决；
- within-region study split 与 cross-region split 分离；
- group/source/pair/gene/context/exposure 泄漏按主张需要全部审计；
- `UNKNOWN/N/A/PENDING` 不得作为 required dimension PASS；
- project prior-use UNKNOWN 始终阻断相关用途；foundation overlap 只对“数据×实际选择的 external checkpoint”组合生效：`from_scratch_E` 记 `NOT_APPLICABLE_NO_EXTERNAL_WEIGHTS`，`supervised_F_to_E` 必须审计 F↔E final 的 exact/near/family/label-lineage overlap；任何 external weights 的 corpus/overlap 为 UNKNOWN 时阻断该模型的 final/generalization claim；
- FM overlap 对当前 data/FM artifacts 实际执行；
- report 绑定 canonical、exposure、Task/Split/Application registries、global eligibility、task cells、role ledger/projection、split assignments、ordinary/restricted commitments、contract 与 FM0 全部 hashes。

## G6：Task and Evaluation Validity

- functional track 每条有合法 endpoint/unit/context/label；
- recovery 按 k≤1、k≤3、k≤5、k>5 分层；
- k>5 不进最大 k=5 的可恢复分母；
- true length change 与 latent alignment indel 分开；
- no-edit control 只用于 no-op specificity/noise/calibration，不提供 termination-time label；
- 5′/3′使用各自资格与 primary metrics；
- row-level 与 group-level 指标同时报告，headline 使用 group-aware CI。

## G7：Fresh Data/Benchmark Output Closure and Model-Rebind Handoff

最终 canonical 冻结后，D1、exposure、B0、FM overlap、task validity、leakage 全量重跑；所有数字、hash、HEAD、config 与 G0 `INPUT_MANIFEST.json` 一致；旧 `overall_pass` 不得引用；`data_goal_required_blocker_ids` 全部有 closure evidence，`model_rebind_handoff_blocker_ids` 单独守恒交接；`OUTPUT_MANIFEST.json`、`SHA256SUMS`、`STATUS.json`、gate/test reports由 finalizer闭环，`DONE` 只在全部 data gates 与 publication-grade resource viability 同时满足时生成。

只有 D0-R、D1-R、FM0-A、B0-R 及 G0–G7 全 PASS，且 `resource_viability_status=PUBLICATION_GRADE_CANDIDATE`，数据 Goal 才可写 `DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND`。工程 closure通过但 viability有限时写 `BLOCKED_WITH_EVIDENCE` 并先讨论数据/论文 scope。GP0 仍保持 `LOCKED_NOT_AUTHORIZED`；必须另立并明确授权 FM0-B→MK0/EF0-R→GP0 preflight，关闭 source binding、stale counts、length、task/model 与 method attribution blockers 后，后续 Goal 才能写 `READY_FOR_GP0_DEVELOPMENT`。本数据 Goal 内除冻结 activation-calibration population上的 `TASK_PROTOCOL_CALIBRATION` 外不得用 v3 labels做模型选择；该 population永久 development-exposed。

## 13.1 当前 P0 blocker

| ID | Domain | Blocker | 阻断范围 |
|---|---|---|---|
| P0-01 | DATA | 修复后的 GSE114002 raw 未进入当前 canonical | D0/D1/B0/GP0 |
| P0-02 | DATA | GSE114002 仅用 10 文件中的 1 个；GSE145046 仅用约 30 condition 中的 1 个 | D0/D1/B0/GP0 |
| P0-03 | EXPOSURE/ROLE | GSE246381 的 owner-confirmed prior non-use、pipeline materialization、external-foundation overlap 和 sealed-final role 尚未在 ledger/test/split/report 中分轴一致表达；active contract/config/code/ledger/test/split/report 的 E4X 解释必须移除或替换，不可变历史 artifacts 原样保留并标 `SUPERSEDED_BY_V3_1_OWNER_CONFIRMED_NON_USE`，且不参与 PASS/role/claim | B0/FM0/GP0/final |
| P0-04 | DATA | 34,042 条缺 legacy `record_type`；新 `scientific_track/relation_acceptance_status/relation_type/effect_evidence/landscape_role/future_use_role` 尚未回填，当前仍由 fallback 推断 pair-shaped role | D1/B0/GP0 |
| P0-05 | DATA | GSE207584 10,227 条 source/candidate 两端 sequence 均为空/未保留；CDS labels-without-sequence 与 full-length 越界混入 | D1/B0/GP0 |
| P0-06 | DATA | 非法字符被静默删除，无 quarantine/conservation | D1/B0/GP0 |
| P0-07 | DATA | identity/no-edit controls 在 ingestion 被删除 | D1/B0/GP0 |
| P0-08 | DATA | GSE217518 attrition、重复 ID、窗口坐标和 assay mapping 未闭环 | D1/B0/GP0 |
| P0-09 | BENCHMARK | B0 无声遗漏 4,012 previously classified/inferred pair-shaped records；必须重裁决，不能预计入 accepted-pair denominator | B0/GP0 |
| P0-10 | BENCHMARK | study 与 region split 混杂，gene/context/barcode N/A 被当 PASS | B0/GP0/final |
| P0-11 | BENCHMARK | 无 label 与超预算 records 可进入 evaluation | B0/GP0/final |
| P0-12 | DATA/TASK | 等长 pair 的 latent INS/DEL alignment 被误读为真实 length edit | D1/B0/claim |
| P0-13 | EXPOSURE | 旧 FM0 已存在，但 B0 曾把未完成的 external-foundation overlap 审计以 pending/unknown 状态当作 PASS | B0/GP0/final |
| P0-14 | CLOSURE | D1/exposure/B0 reports 陈旧且未全链 hash 绑定 | 当前阶段 closure |
| P0-15 | DATA | sequence scope 缺失，insert/window/full/CDS 语义混合 | D1/B0/GP0 |
| P0-16 | SOURCE_BINDING | 数据修复分支与 GP0/EF0 工作树不是单一 source/data/contract snapshot | MK0/EF0 rebind/GP0 |
| P0-17 | MODEL | GP0 硬编码旧 paired count/accession、max_length=256，formal preflight 必失败 | GP0 |
| P0-18 | MODEL/CLAIM | 旧 STOP active 路线必须移除/禁用并重绑 fixed-budget；region ablation 无效、LoRA 命名错误，均不能支撑 headline | MK0/EF0 rebind/GP0/claim |
| P0-19 | DATA/CONTRACT_SCOPE | active v3.1 合同、数据/FM 配置、loader、registry 与报告仍可能含 Track U/项目无标签预训练路线，未与 E/F-only 决策一致；本项只在当前 data worktree 的 active artifacts 上关闭 | C3/D0/D1/FM0/data claim |
| P0-20 | METHOD/CLAIM | alignment、switch/order、CTMC trajectory 与 detour 仍可能混称 path；`UPSTREAM_REPRO_CHECK/BASIC_PAIR_EF_NATIVE/UTR_ADAPTED_*` 归因、上游 identity 与 estimator tests 缺失 | MK0/EF0-R/GP0/claim |
| P0-21 | DATA | N-zip、GSE330741、GSE261709、GSE298114 等新 P0 E/F 资产尚未完成 bounded recovery、许可与 lineage 决策 | D0/D1/B0/data-scale claim |
| P0-22 | SCHEMA/ROLE | E evidence、dense/no-edit overlay 与 pair-level global future-use role 曾被单枚举混合；task partition assignment、F observation/F auxiliary task 与 mixed asset 缺记录级正交角色 | C3/D1/B0 |
| P0-23 | STATE_MACHINE | acquisition/parse/mapping/canonical/license/science/release 原子状态与旧复合枚举尚未在 active artifacts 全部统一；future-use role 还必须通过带合法矩阵、hash chain 与 effective projection 的 append-only transition ledger 改变 | C3/D0/D1/B0 |
| P0-24 | FINAL_ISOLATION | GSE246381 restricted sealed store、D1-before-B0 loader isolation、B0/G7 dual-store、五类 machine whitelist/commitment 与 access-event taxonomy 尚未实现 | D1/FM0/B0/final |
| P0-25 | CONSERVATION | raw、relation-candidate、measurement join、accepted candidate↔pair payload identity、global EligibilityRecord/unique-object、外部冻结 Task/Split/Application matrix 派生的 cells、SplitAssignment/role-partition compatibility 与 orphan-edge 独立守恒尚未实现 | D1/B0 |
| P0-26 | LICENSE/USE | training/evaluation/release 权限尚未贯穿 record-level eligibility 与 Data Card 分母 | D0/B0/release |
| P0-27 | DATA_ROLE | GSE145046 input/support rows 当前未完成 label join，不能直接计作 F functional examples | D1/data-scale claim |
| P0-28 | CONTRACT_TEST | literal keyword hit=0 会误伤合同中的禁止/废止文字；必须改为结构化 positive-assignment tests | C3 |
| P0-29 | INVENTORY | P0/P1/P2/reference/aux/watchlist 冻结全集与 registry set-equality gate 尚未实现 | D0/G4 |
| P0-30 | TASK/CONTRACT | active task registry/evaluator contract 将 endpoint reconstruction 与功能 ranking/search estimand 混用；必须在 B0 分成 T5-Gen-Reconstruct 与 T5-Rank/Closed-Select，并明确 generator 无 outcome head/reward时不得用 recovery 支撑 optimization | C3/B0/claim |
| P0-31 | MODEL_REBIND/SCOPE | 历史 MK0/EF0/GP0 worktree 与模型配置/代码仍可能保留 Track U、项目无标签预训练 fallback、旧 task coupling 或 v2 schema；只能在后续 model-rebind Goal 中逐文件移除或隔离 | FM0-B/MK0-R/EF0-R/GP0/claim |

`GOAL-V3-DATA-BENCH-01` 的 required blocker 集与后续 model-rebind handoff 集固定如下，禁止由执行者按方便自行增删或把二者混成“全部 P0 必须本阶段关闭”：

```yaml
data_goal_required_blocker_ids:
  [P0-01, P0-02, P0-03, P0-04, P0-05, P0-06, P0-07, P0-08, P0-09, P0-10,
   P0-11, P0-12, P0-13, P0-14, P0-15, P0-19, P0-21, P0-22, P0-23, P0-24,
   P0-25, P0-26, P0-27, P0-28, P0-29, P0-30]
model_rebind_handoff_blocker_ids: [P0-16, P0-17, P0-18, P0-20, P0-31]
```

DATA/EXPOSURE/BENCHMARK/CONTRACT 域 required blocker 必须在本数据 Goal 的对应 artifacts 中 `CLOSED_WITH_EVIDENCE`；任一仍 OPEN 时只允许 `BLOCKED_WITH_EVIDENCE`。handoff blocker 允许在本 Goal 结束时保持 OPEN，但必须逐项给出当前 evidence、受影响文件、关闭条件和 owner，并保持 GP0 locked；它们不得被误报为本 Goal 失败，也不得被误报为已解决。任一影响 final/claim 的 P0 OPEN 时，论文主表不得冻结。

---

# 14. Phase 与 Task Registry

## 14.1 执行顺序

```text
C3 contract freeze
  → D0-R data exhaustion and asset inventory
  → D1-R canonical rebuild and QC
  → FM0-A exposure preflight
  → B0-R benchmark/splits/final seal
  → FM0-B backbone freeze and adaptation contract
  → MK0/EF0-R regression and source binding
  → GP0-DEV three-round preregistered development
  → FC0 endpoint critics/evaluators
  → MB0 matched-budget one-time final
  → ER0 robustness/failure analysis
  → PP0 paper/release
```

不得并行越过依赖 gate。允许并行的是彼此只读、不会污染 final 的 data inventory、文献复核、schema unit tests 和开发型 profiling。

## 14.2 下一阶段待用户明确激活的 Goal 模板

```yaml
goal_id: GOAL-V3-DATA-BENCH-01
goal_name: E/F data exhaustion, canonical rebuild and benchmark-v1 closure
objective: >
  在不启动 GP0 真实训练、不发生 sealed-final analytic/final-evaluator access、不引入无标签预训练的前提下，
  冻结 v3.1 合同，穷尽并裁决 P0/P1 E/F 资产，重建可审计 canonical，完成
  FM0-A 与 benchmark v1 split/final seal，并生成 data/benchmark closure 或带证据阻断的唯一终态；
  即使成功也只交接给后续 model-rebind Goal，不解锁 GP0-DEV。
execution_authorization: NOT_GRANTED_BY_THIS_DOCUMENT
activation_requires: EXPLICIT_USER_INSTRUCTION_TO_EXECUTE_GOAL_ID
phases_in_scope_after_activation: [C3, D0-R, D1-R, FM0-A, B0-R, G7]
required_blocker_set: data_goal_required_blocker_ids
permitted_open_handoff_set: model_rebind_handoff_blocker_ids
terminal_success: DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND
terminal_success_requires_resource_viability: PUBLICATION_GRADE_CANDIDATE
terminal_failure: BLOCKED_WITH_EVIDENCE
not_authorized: [FM0-B, MK0-R, EF0-R, GP0-DEV_TRAINING, FC0_TRAINING, MB0_FINAL, PR3_FINAL_OPEN, PP0_PUBLIC_RELEASE]
```

该 Goal 的成功不是“模型有效”或“已允许 GP0”，而是“数据与 benchmark 已闭合，且按预注册门槛具有 publication-grade candidate 资源资格，可交给下一张 FM0-B/MK0-R/EF0-R/model-rebind 合同”；该资格仍不保证发表。合同被写入文件不构成执行授权；只有用户后续明确要求执行本 Goal，Codex 才能进行其范围内的远端写入/下载/commit。到达终态后必须停止并交接。

### 14.2.1 工作区与运行根

- 保护主仓库 `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`，先只读预检；
- 使用新隔离 worktree：`/home/cunyuliu/mrna_editflow_goal/worktrees/v3_data_benchmark_<UTC>`；
- 大型 run/artifacts 默认使用：`/mnt/cunyuliu/mrna_editflow_v3_runs/<RUN_ID>`；若 `/mnt` 不可用，必须先选择有空间、归属清楚的新绝对路径并写入 manifest；
- 不 reset 主树、不覆盖现有结果、不删除旧失败、不杀无关进程、不占用有活跃任务的 GPU；
- 每个 phase 使用新 `RUN_ID`，禁止在失败 run root 上原地重试；
- phase 内允许 focused local commit；默认不 push、不建 PR，除非用户另行明确授权。

### 14.2.2 Goal 级禁止事项

- 不下载、构建或训练 Track U；不以无标签 corpus 扩数据规模；
- 不运行 GP0/FC0/MB0 真实标签训练或 final；
- 不读取 GSE246381 标签做调参、错误分析或人工浏览；aggregate-only QC 必须由隔离脚本输出；
- 不把 raw/barcode/replicate/sample rows 改名为 pairs；
- 不静默删除非法字符、identity、困难数据集或 P0 数据；
- 不把 `UNKNOWN/N/A/PENDING` 当 PASS；
- 不降低 gate、改阈值、改 split、删失败 seed 或过滤不利结果制造成功；
- 不把 E0 engineering、smoke、合同自洽或 row count 写成科学成功。

### 14.2.3 当前已知受影响路径；执行前仍须重新确认

主仓库相对路径：

```text
configs/utr_editflow_contract_v2.yaml
d1_staging/scripts/b0/canonical_schemas.py
d1_staging/scripts/d1/edit_script_core.py
d1_staging/scripts/d1/build_canonical_records.py
d1_staging/scripts/d1/build_exposure_ledger.py
d1_staging/scripts/d1/audit_canonical_records.py
d1_staging/tests/test_d1_exposure_ledger.py
d1_staging/scripts/b0/build_split_manifests.py
d1_staging/scripts/b0/audit_split_manifests.py
d1_staging/scripts/b0/leakage_audit.py
d1_staging/scripts/b0/eval_tracks.py
d1_staging/scripts/b0/data_card.py
d1_staging/tests/test_b0_canonical_schemas.py
d1_staging/tests/test_b0_split_manifests.py
d1_staging/tests/test_b0_leakage_audit.py
d1_staging/tests/test_b0_eval_tracks.py
d1_staging/tests/test_b0_data_card.py
data/d1_canonical_records.jsonl
data/data_exposure_ledger.jsonl
data/b0_04_eval_track_manifest.jsonl
data/b0_splits/split_5utr_source_disjoint.jsonl
data/b0_splits/split_3utr_source_disjoint.jsonl
data/b0_splits/split_study_disjoint.jsonl
data/b0_splits/split_cross_region_transfer.jsonl
data/b0_01_audit_report.json
data/b0_02_audit_report.json
data/b0_03_leakage_audit_report.json
data/b0_04_eval_track_audit_report.json
data/b0_05_data_card.json
```

GP0 当前文件位于独立 worktree，后续 model-rebind 至少覆盖：

```text
/home/cunyuliu/mrna_editflow_goal/worktrees/gp0_generative_prior_20260803/scripts/gp0/common.py
/home/cunyuliu/mrna_editflow_goal/worktrees/gp0_generative_prior_20260803/scripts/gp0/record_gp0_preflight.py
/home/cunyuliu/mrna_editflow_goal/worktrees/gp0_generative_prior_20260803/scripts/gp0/train_gp0.py
/home/cunyuliu/mrna_editflow_goal/worktrees/gp0_generative_prior_20260803/scripts/gp0/evaluate_gp0.py

# 后续 source-bound worktree 中至少审计的相对路径
core/ef0/model.py
core/ef0/sampler.py
core/ef0/exact_sampler.py
tests/ef0/test_state_contract.py
core/mk0/bregman.py
core/mk0/stop.py
tests/mk0/test_stop_contract.py
scripts/run_region_adapter_ablation.sh
scripts/eval_region_adapter_ablation.sh
scripts/run_region_adapter_ablation_chain.sh
```

本 Goal 只允许将这些路径列入 blocker/未来修复清单并记录是否存在、当前 hash 与受影响语义；不得在该旧 worktree 或主树修改/启动训练，也不得把它们当作 v3.1 source snapshot。后续 model-rebind 必须从当时绑定的完整 commit 在新隔离 worktree 重新解析相对路径；路径改名只能通过 inventory 证明 successor，不能借“文件不存在”关闭 blocker。

## 14.3 C3：合同、状态机与 claim freeze

### C3-01 合同落盘与 supersession

**Files：**

- Create/replace authoritative: `docs/contracts/utr_editflow_goal_v3_1.md`
- Create: `docs/contracts/supersession_v3_1.md`
- Create/modify: `configs/utr_editflow_contract_v3_1.yaml`
- Create: `docs/execution/decision_log_v3_1.yaml`
- Create: `docs/execution/USER_DECISION_GSE246381.yaml`

**Todo：**

- [ ] 逐字读取本合同并记录本地文件 SHA256；
- [ ] 将 v2/v3.0 标为 `HISTORICAL_SUPERSEDED`，不删除旧文档；
- [ ] 写入 V3-010 至 V3-016 全部最新决策；
- [ ] `USER_DECISION_GSE246381.yaml` 分别记录 owner-confirmed prior non-use、pipeline materialization、foundation overlap 和 conservative sealed-final role；
- [ ] 全仓检索 GSE246381 的 E4X/historically-exposed 旧说法、active Track U 和不实 path-originality 说法；
- [ ] 每个命中分类为 `REPLACE_NOW`、`HISTORICAL_KEEP_WITH_WARNING` 或 `GENERATED_ARTIFACT_INVALIDATED`；
- [ ] 所有 active doc/config 指向同一个 v3.1 ID 与合同 hash sidecar。

### C3-02 Schema、状态与 claim matrix

**Files：**

- Create: `schemas/v3_1/dataset_asset.schema.json`
- Create: `schemas/v3_1/sequence_entity.schema.json`
- Create: `schemas/v3_1/functional_observation.schema.json`
- Create: `schemas/v3_1/utr_edit_relation_candidate.schema.json`
- Create: `schemas/v3_1/utr_edit_pair.schema.json`
- Create: `schemas/v3_1/edit_path_set.schema.json`
- Create: `schemas/v3_1/generation_task.schema.json`
- Create: `schemas/v3_1/relation_role_transition.schema.json`
- Create: `schemas/v3_1/split_assignment.schema.json`
- Create: `schemas/v3_1/rejection_record.schema.json`
- Create: `schemas/v3_1/eligibility_record.schema.json`
- Create: `schemas/v3_1/transformation_edge.schema.json`
- Create: `schemas/v3_1/reporter_artifact_assessment.schema.json`
- Create: `schemas/v3_1/group_registry.schema.json`
- Create: `schemas/v3_1/group_assignment.schema.json`
- Create: `schemas/v3_1/task_registry.schema.json`
- Create: `schemas/v3_1/split_registry.schema.json`
- Create: `schemas/v3_1/task_split_applicability.schema.json`
- Create: `schemas/v3_1/task_eligibility_cell.schema.json`
- Create: `schemas/v3_1/exposure_record.schema.json`
- Create: `schemas/v3_1/use_role.schema.json`
- Create: `schemas/v3_1/SCHEMA_MANIFEST.json`
- Create: `schemas/v3_1/SCHEMA_SHA256SUMS`
- Create: `docs/execution/claim_matrix_v3_1.yaml`
- Create: `docs/execution/task_registry_v3_1.yaml`
- Create: `docs/execution/split_registry_v3_1.yaml`
- Create: `docs/execution/task_split_contract_matrix_v3_1.yaml`
- Create: `docs/execution/diagnostic_registry_v3_1.yaml`
- Create: `docs/execution/resource_viability_rule_v3_1.yaml`

**Todo：**

- [ ] 拆开 `acquisition_status/parse_status/mapping_status/canonical_status/license_status/scientific_status/release_decision`；
- [ ] 拆开 `scientific_track/relation_acceptance_status/relation_type/effect_evidence/landscape_role/future_use_role/confirmatory_delta_eligible`；
- [ ] candidate 与 pair 的两端统一引用 `SequenceEntity.sequence_id`；冻结 endpoint/context/delta identity、candidate→pair bijection、append-only `RelationRoleTransition` schema、allowed transition matrix、hash-chain 与 exact-cardinality effective-role projection；
- [ ] 冻结 global unique-object 与 `object_id×task_id×split_contract_id` eligibility-cell 两层互不替代的计数单位、状态机和 Data Card denominator；Task/Split registries 与完整 applicability matrix 必须 set-equality、FK 和 hash 闭合；
- [ ] 建立 immutable baseline exposure、ordinary/restricted access intent→completion/abort chains、effective exposure projection、逐 checkpoint foundation overlap 与 future role/final access 分层 authority；
- [ ] 在 claim matrix 中逐条列出 evidence ID、所需 gate、允许 wording、禁止 wording 和 unlock status；
- [ ] 标记 benchmark、method、region scope 三条正交 outcome；
- [ ] 规定合同事实冲突时通过 decision log 修合同，禁止 parser 自选解释。

### C3-03 先写失败测试，再修 active artifacts

**Files：**

- Create: `tests/contracts/test_utr_editflow_v3_1_contract.py`
- Create: `scripts/contracts/validate_v3_1_contract.py`

**Required tests：**

- [ ] 结构化断言 active schema/assignments 中 legacy `GSE246381.project_exposure` 字段不存在，`evidence_grade=E4X` 正向赋值为 0；prose 中 `PROHIBITION/HISTORICAL_SUPERSEDED/NEGATED_TRUTH_LOCK` 命中允许存在但必须分类，只有未分类或正向赋值才 FAIL；
- [ ] 直接断言 GSE246381 的 `project_sequence_analytic_exposure=NONE_CONFIRMED`、`project_sequence_analytic_use_types=[NONE_CONFIRMED]`、`project_label_analytic_exposure=NONE_CONFIRMED`、`project_label_analytic_use_types=[NONE_CONFIRMED]`；同时 pipeline sequence/label materialization=`PRESENT`、foundation requirement=`REQUIRED_FM0_A`/audit status=`DEFERRED_TO_FM0_A`、future role=`SEALED_EXTERNAL_FINAL_CANDIDATE`，四轴不得互相覆盖；禁止 accession 级 scalar foundation overlap 替代 checkpoint ledger；
- [ ] 保留当前 ledger/test 的历史 unexposed 方向，只升级字段；不得因 supersede v2 E4 而反向写 exposed；
- [ ] 结构化断言 active `scientific_track=U`、`project_unlabeled_pretraining_enabled=true`、`fallback_to_U_track=true` 均为 0；禁止/历史废止文字不按 literal keyword 误报；
- [ ] external backbone corpus audit 仍存在；
- [ ] Edit Flows inherited/new contribution matrix 完整；
- [ ] `alignment/order/clock/trajectory/detour` 五类对象不能共用含混枚举；
- [ ] candidate→pair 不仅 FK 一一对应，而且端点、track、relation/effect/landscape、immutable base `future_use_role`、pairing method 和 evidence ID 全部相等；另对 effective-role projection 做 candidate↔pair equality；base payload mismatch 或 projection mismatch 任一非零必须 FAIL；
- [ ] active `schemas/v3_1/` 的 21 个 schema filenames 与 §5.1 expected set/hash 完全一致；逐文件 `$id/schema_version/contract_id/hash` 唯一匹配，missing/duplicate/unexpected/version/hash counters=0，并运行每个 schema 的 positive/negative fixtures；
- [ ] Task/Split expected ID sets、task descriptor/analysis-unit/species maps、task→split allowlist 及其冻结 hashes 完全一致；TaskSplitContractMatrix 恰有 120 rows；F auxiliary 与 DiagnosticRegistry expected ID/hash不可删除或换 object type；
- [ ] PAIR_OR_OBSERVATION grouping-atom rule与 outcome-blind activation-calibration rule的 canonical bytes/hashes完全匹配；resource viability thresholds在任何 label访问前冻结；
- [ ] §5.1 全部 required `$defs` 均存在并有 positive/negative/hash-chain/cycle/half-commit fixtures；不得遗留 obsolete `SealedAccessEvent`；
- [ ] 结构化验证 global unique-object 与 TaskEligibilityUniverse 是两层不同规则；实际 task-cell population/zero-pending 只在 B0 物化验收，C3 不伪造尚不存在的 PASS；
- [ ] FORMAL 数据/CPU baseline 不被错误要求 CUDA，FORMAL neural training 必须 CUDA；
- [ ] v3.1 ID、schema、task registry、claim matrix 和决策日志彼此一致。

**Acceptance：**测试先在旧 active artifacts 上产生预期 FAIL，完成最小修复后 PASS；生成 `C3_STATUS.json`、`C3_MANIFEST.json`、`C3_SHA256SUMS`。C3 未 PASS 不得进入 D0-R 写操作。

## 14.4 D0-R：E/F 数据穷尽、许可与资产冻结

### D0-00 只读 preflight

- [ ] 记录 repo branch/full HEAD/dirty files、现有 worktrees、进程、GPU ownership、磁盘与 inode；
- [ ] 记录所有现有 raw、partial、repaired、canonical、ledger、split、FM0 和旧 reports 的绝对路径、size、mtime、SHA256；
- [ ] 不读取或输出原始核苷酸序列；远端审计只返回 aggregate counts/schema/hash；
- [ ] 创建 `INPUT_MANIFEST.json`，未闭合则停止。

### D0-01 当前资产全盘清账

**Lightweight registries：**

- `data/v3_1/registry/dataset_assets.jsonl`
- `data/v3_1/registry/raw_asset_manifest.jsonl`
- `data/v3_1/registry/search_ledger.jsonl`
- `data/v3_1/registry/license_matrix.csv`
- `data/v3_1/registry/dataset_decisions.jsonl`
- `data/v3_1/registry/priority_snapshot_v3_1.yaml`

**Todo：**

- [ ] 对 GSE114002 的 10 文件、GSE145046 全 condition/replicate、GSE217518 workbook/assay/full sequences、GSE232571/572、fast-UTR、DART、ENCSR、GSE200304 及其 GSE200302/GSE200303/GSE217530 三个 SubSeries，以及当前所有 legacy assets 逐文件登记；
- [ ] repaired 文件与当前 canonical 的 consumer relationship 明确为 `CONSUMED/NOT_CONSUMED/SUPERSEDED`；
- [ ] `.part/.new`、CRC 失败、未知 hash 和未排序 glob 结果不得进入 parser；
- [ ] 每个 asset 填 provider、publication/project IDs、URL、retrieved date、checksum、mapping、许可和 release 决策；
- [ ] fast-UTR pin commit `4aaeb7e97c5ec093c356f0564f96d87887ee9ab7`，再记录逐文件 SHA；
- [ ] GSE207584/GSE173083 作为 legacy cleanup liability，不计 UTR E/F 规模。

### D0-02 有界 E/F 公共数据再检索

- [ ] 只检索具有 UTR 功能标签、显式 ref-alt/source-candidate、dense mutagenesis、no-edit control 或 reporter-artifact QC 的资产；
- [ ] 覆盖 GEO/SRA/BioProject、ENCODE、ArrayExpress/BioStudies、Zenodo/Dryad、PMC 与期刊 supplements、作者仓库、MaveDB/MPRAbase 与 backward/forward citations；
- [ ] 强制处理 P0 新增：N-zip、GSE330741、GSE261709、GSE298114、PTRE-seq；
- [ ] bounded 处理 P1：GSE194092、GSE270252/254、GSE173098、GSE295080、GSE291719、GSE55396、PASSPORT-seq、SEERS；Saluki 按 P2 promotion condition 处理；
- [ ] 保存 query、日期、结果数、去重、排除理由、full-text/data availability 与最后决定；
- [ ] 无标签资源只登记为 reference service/watchlist，不下载作训练。
- [ ] 验证 P0/P1/P2/reference/aux/watchlist registry set 与 §7 冻结全集完全相等；不得用 registry 自报“已穷尽”。

### D0-03 许可与使用矩阵

- [ ] 对下载、处理、训练、评测、derived release、raw redistribution 分字段核验；
- [ ] “公开可访问”不能自动升级 `VERIFIED`；无明确证据保持 `license_status=REVIEW_REQUIRED`；
- [ ] 不可再分发者只允许 accession+adapter+checksum+非重构 metadata release；
- [ ] `d0_decision=ACQUIRED_FOR_REBUILD` 只允许在 `permitted_download=YES` 且 `permitted_processing=YES`、并绑定 accountable reviewer 与非空 `use_basis_evidence_ids` 后产生；任一为 `NO|UNKNOWN` 时只能 `METADATA_ONLY` 或 `EXCLUDED_WITH_EVIDENCE`，不得进入 D1 parser；
- [ ] 许可未知不得进入公开 release，是否可在本地研究训练/评测也必须单独裁决。

### D0-04 数据优先级终审

- [ ] 在 `dataset_decisions.jsonl` 中，每个 P0 的 D0 决策为 `ACQUIRED_FOR_REBUILD` 或 `EXCLUDED_WITH_EVIDENCE`；每个 P1 为 `ACQUIRED_FOR_REBUILD`、`METADATA_ONLY`、`MAPPING_UNRESOLVED` 或 `EXCLUDED_WITH_EVIDENCE`；这些只是 `d0_decision`，不得冒充 `canonical_status`；
- [ ] 排除必须有 source、reason code、失败日志、已尝试路线和人工复核字段；
- [ ] 输出 E 规模候选表与 F 规模候选表，禁止跨轨 `N_total`。

**D0-R Acceptance：**P0/P1 状态闭合；所有数字标明 unit/denominator/verification state；每个进入 D1 的 asset 均有 download+processing permission=`YES` 与 evidence ID，并写入 D1 `INPUT_MANIFEST`；许可矩阵闭合；`D0_STATUS.json=PASS`、manifest/checksum/finalizer 完整。下载很多但角色/许可未知仍为 FAIL。

## 14.5 D1-R：canonical 重建与完整清洗

### D1-01 Schema-first 与 adapter contract tests

- [ ] 先为每个 P0 adapter 编写最小 fixture、expected counts/roles、失败/歧义案例和 row-conservation 测试；
- [ ] 旧 canonical 必须使新 schema 测试按预期 FAIL；
- [ ] adapter 禁止通过文件顺序、首个 glob、隐式列名或下游 fallback 猜角色；
- [ ] 每个 parser 输出 raw disposition、transformation edges、quarantine 和 dataset-specific reconciliation。

### D1-02 Sequence 与 source binding

- [ ] 所有 sequence 保存 asset/file/record/row locator、raw/normalized/full/window hash、scope、original length、coords、strand、scaffold、editable mask；
- [ ] 非 ACGTU/IUPAC、方向不明、ref mismatch、坐标不明不得静默修正；
- [ ] source/candidate 只能通过 design table、explicit ID、validated variant reconstruction 或确定性 barcode join 绑定；
- [ ] `source==candidate` 恢复为 E_NOEDIT；空 source/candidate 不进入 E pair；
- [ ] 同长度 alignment INS/DEL 与 coordinate-verified/source-reported length change 分字段。

### D1-03 Observation、replicate 与 delta

- [ ] 所有 endpoint、unit、transform、assay/cell/promoter/reporter/chemistry/timepoint、replicate/barcode/coverage/missingness 结构化；
- [ ] source 与 candidate 都有兼容测量才标 E_DELTA；一侧缺失或 context 不同保持 E_LINK；
- [ ] replicate aggregation 规则、SE/uncertainty 和 outlier policy 预先冻结；
- [ ] barcode/replicate/sample rows 不产生额外 pairs/parents。

### D1-04 Dataset-specific rebuild

- [ ] GSE114002：全 10 文件、sublibrary role、repaired raw consumption、identity/no-edit、约 55,184 pair-candidates 的来源重放；
- [ ] GSE145046：全 condition/replicate join；input-only/normalization/support rows 不计 F；只有 deterministic join 后 label-complete 的 sequence×context observation 进入 F，可靠 pair 默认 0；
- [ ] GSE217518：6,555/5,917/5,072/3,564 等 denominator 逐层对账，38 duplicate IDs、失败行与 window/full mapping 全闭合；
- [ ] MapUTR：设计、论文、FASTA、ref-only/alt-only、canonical 数目分开；
- [ ] fast-UTR/DART/GSE176581/GSE232927：E/F 子库与 parents 分开；
- [ ] ENCSR/GSE186455/GSE149487：source/label/context/missingness 与 artifact 风险闭合；
- [ ] GSE200304 group：SuperSeries 与 GSE200302 Polysome-seq、GSE200303 DNA-seq、GSE217530 RNA-seq 的 accession/file set 全等；三模态分别完成 raw-unit/replicate/missingness 守恒后，才允许 deterministic source/alt/endpoint join；
- [ ] N-zip/GSE330741：mutants 与约 16/少数 parents 分开，lineage/family split key 可重放；
- [ ] GSE261709/GSE298114：locus/ref/alt/strand/build/oligo/barcode/label 确定性重建；
- [ ] GSE288185：MPRA/RBNS/SLAM 的 sequence–label join、gene/tile family 与 overlapping design lineage 闭合；
- [ ] PTRE-seq：输出 `reporter_artifact_assessments.jsonl`，只进入 AUX_QC/artifact gate，不进入 E/F 主 canonical；
- [ ] GSE246381：使用隔离 sealed builder 对两张 32,990-row matrices、1,507 variant keys、1,300 reconstructed、207 failures、116 no-edit、1,184 edits、2×(1,350 controls+1,500 shuffles)、347 windows 做全部守恒；机械处理 replicate/zero/missing/normalization/uncertainty，但普通 canonical/loader/report 只接收 aggregate QC、commitment hash 与状态，不得接收逐条 sequence/label/join；保持 owner-confirmed prior use=`NONE_CONFIRMED`、pipeline materialization=`PRESENT` 和 sealed candidate；
- [ ] GSE207584/GSE173083：从 UTR 主轨移除并闭合 legacy count。

### D1-05 Conservation、artifact 与 duplicate audit

- [ ] 每文件分别闭合 raw-unit、declared-relation-candidate、measurement-join 与 current-leaf technical-accepted universe；D1 不创作 ACTIVE/EXCLUDED/PENDING global disposition；
- [ ] transformation edges 完整支持 one-to-many/many-to-one；每个 accepted raw unit 有 outbound edge 或 support-only disposition，每个 canonical entity 有 inbound edge，orphan=0；
- [ ] 每个 observation join proposal 持久化为 FunctionalObservationCandidate；五态 attrition set-equality闭合，每个 accepted candidate 与 FunctionalObservation reciprocal一一对应、完整 contributor set一致；
- [ ] 每个 accepted candidate 恰好产生一个 formal pair 且反向唯一；两端 sequence FK 与全部 identity payload 逐字段一致，`candidate_pair_fk_mismatch=0`、`candidate_pair_payload_mismatch_count=0`；端点/语义修复只能生成 parent-linked supersession，禁止原地改写；
- [ ] sequence/relation/pair/observation candidate/observation supersession edge、acyclic current-leaf projection与 downstream generation一致；旧 repaired generations保留但 active loader/denominator count=0；
- [ ] ordinary/restricted ExposureRecord 覆盖 in-scope sequence 与两类 candidate/accepted scientific objects；accepted E pair的 UseRole/base role闭合；完整 contributor-rights projection逐 purpose fail closed；历史旧 exposure ledger只读；
- [ ] 分别冻结 E accepted current-leaf pair universe 与 F label-complete accepted current-leaf observation universe，所有对象统一标 `AWAITING_B0_GLOBAL_DISPOSITION`；ACTIVE/EXCLUDED/PENDING 只由 B0 Stage3 创作。此处不得用 task assignment 行数求和，task-specific eligibility-cell universe同样在 B0 物化；
- [ ] exact/reverse-complement/near cluster、pair/source/parent/gene/tile/library lineage 分层；
- [ ] 5′ uAUG/uORF/Kozak 与 3′ splice/PAS 风险分开；fully reconstructed 不自动低风险；
- [ ] HIGH/UNKNOWN 风险保留于敏感性/独立臂，不能事后删除。

### D1-06 Canonical freeze

**Required outputs：**

- `data/v3_1/canonical/sequence_entities.jsonl`
- `data/v3_1/canonical/functional_observation_candidates.jsonl`
- `data/v3_1/canonical/functional_observations.jsonl`
- `data/v3_1/canonical/ENDPOINT_REGISTRY.jsonl`
- `data/v3_1/canonical/utr_edit_relation_candidates.jsonl`
- `data/v3_1/canonical/utr_edit_pairs.jsonl`
- `data/v3_1/canonical/rejections.jsonl`
- `data/v3_1/canonical/transformation_edges.jsonl`
- `data/v3_1/canonical/SUPERSESSION_EDGES.jsonl`
- `data/v3_1/canonical/CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl`
- `data/v3_1/canonical/EXPOSURE_RECORDS.jsonl`
- `data/v3_1/exposure/projections/<d1_snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`
- `data/v3_1/canonical/USE_ROLES.jsonl`
- `data/v3_1/canonical/EXPOSURE_USE_MANIFEST.json`
- `data/v3_1/canonical/EXPOSURE_USE_SHA256SUMS`
- `data/v3_1/exposure/ORDINARY_ACCESS_LOG.jsonl`
- `data/v3_1/exposure/access_snapshots/<d1_snapshot_id>/ORDINARY_ACCESS_LOG.jsonl`
- `data/v3_1/exposure/access_snapshots/<d1_snapshot_id>/ORDINARY_ACCESS_MANIFEST.json`
- `data/v3_1/exposure/access_snapshots/<d1_snapshot_id>/ORDINARY_ACCESS_SHA256SUMS`
- `data/v3_1/canonical/dataset_reconciliation.json`
- `data/v3_1/canonical/data_units_report.json`
- `data/v3_1/canonical/reporter_artifact_assessments.jsonl`
- `data/v3_1/canonical/group_registry.jsonl`
- `data/v3_1/canonical/group_assignments.jsonl`
- `data/v3_1/canonical/CANONICAL_MANIFEST.json`
- `data/v3_1/canonical/CANONICAL_SHA256SUMS`
- `<restricted_run_root>/sealed_external/GSE246381/SEALED_INPUT_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/sequence_entities.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/functional_observation_candidates.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/functional_observations.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/ENDPOINT_REGISTRY.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/utr_edit_relation_candidates.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/utr_edit_pairs.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/rejections.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/transformation_edges.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/SUPERSESSION_EDGES.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/CURRENT_CANONICAL_OBJECT_PROJECTION.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/EXPOSURE_RECORDS.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/exposure/projections/<d1_snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/USE_ROLES.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/group_registry.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/group_assignments.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/dataset_reconciliation.json`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/data_units_report.json`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/EXPOSURE_USE_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/canonical/EXPOSURE_USE_SHA256SUMS`
- `<restricted_run_root>/sealed_external/GSE246381/SEALED_CANONICAL_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/SEALED_CANONICAL_SHA256SUMS`
- `<restricted_run_root>/sealed_external/GSE246381/ACCESS_LOG.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<d1_snapshot_id>/ACCESS_LOG.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<d1_snapshot_id>/ACCESS_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<d1_snapshot_id>/ACCESS_SHA256SUMS`
- `data/v3_1/sealed_commitments/GSE246381_AGGREGATE_QC.json`
- `data/v3_1/sealed_commitments/GSE246381_COMMITMENT.json`

restricted D1 root inventory 的 logical component universe 不得由实现者自行选择，精确为 `ACCESS_LOG/ACCESS_MANIFEST/ACCESS_SHA256SUMS/CURRENT_CANONICAL_OBJECT_PROJECTION/DATASET_RECONCILIATION/DATA_UNITS_REPORT/EFFECTIVE_EXPOSURE_PROJECTION/ENDPOINT_REGISTRY/EXPOSURE_RECORDS/EXPOSURE_USE_MANIFEST/EXPOSURE_USE_SHA256SUMS/FUNCTIONAL_OBSERVATION_CANDIDATES/FUNCTIONAL_OBSERVATIONS/GROUP_ASSIGNMENTS/GROUP_REGISTRY/REJECTIONS/SEALED_CANONICAL_SHA256SUMS/SEALED_INPUT_MANIFEST/SEQUENCE_ENTITIES/SUPERSESSION_EDGES/TRANSFORMATION_EDGES/USE_ROLES/UTR_EDIT_PAIRS/UTR_EDIT_RELATION_CANDIDATES`；按 logical ID UTF-8 字典序、每项 LF 的 set SHA256=`974736d060463b3af090af3dd0c6a0e8bc591305f57f51e0a8cd31751a1ee606`。其中 logical IDs `ACCESS_LOG/ACCESS_MANIFEST/ACCESS_SHA256SUMS` 必须且只能映射到 `access_snapshots/<d1_snapshot_id>/` 下的 immutable prefix 三件套；root live `ACCESS_LOG.jsonl` 是持续增长的 operational authority，不属于该 immutable 24-ID component set，也不得冒充其 `ACCESS_LOG` component。

`SEALED_CANONICAL_SHA256SUMS` 按 UTF-8 relpath 字典序、每行 `<sha256><two-spaces><relative-path>\n` 精确覆盖上述 canonical/input payload files，但排除 access 三个 logical components、`SEALED_CANONICAL_MANIFEST.json` 与 checksum ledger 自身；access prefix 三件套由 §3.2 单独闭合。`SEALED_CANONICAL_MANIFEST.json` 是唯一 restricted D1 root inventory，至少含 `manifest_id/contract_id/contract_sha256/run_id/d1_snapshot_id/access_prefix_snapshot_id/cohort_ids/cohort_set_sha256/logical_component_set_sha256/logical_components[{logical_id,relative_path,sha256,schema_id,schema_sha256}]/sealed_canonical_sha256s_sha256/access_manifest_sha256/access_sha256s_sha256/access_log_chain_root_sha256/exposure_use_manifest_sha256/effective_exposure_projection_sha256/manifest_sha256`，并要求 logical IDs/paths 与本段 exact set 一一对应、无 missing/extra/duplicate/relpath escape；非 JSON/checksum artifacts 的 `schema_id/schema_sha256` 固定写 `NOT_APPLICABLE_NON_JSON`，不得留空或自选 sentinel。该 root manifest 的 `manifest_sha256` 精确定义为 `SHA256(RFC8785/JCS(完整 JSON object 删除 manifest_sha256 字段后))`，加入该字段后的最终文件 bytes 必须且只能为 `UTF-8 RFC8785/JCS(完整 object)+单个 LF`，其 full-file SHA256 由 ordinary commitment 记录；C3 必须给出 self-hash/full-file-hash golden、field-mutation 与 noncanonical-serialization rejection fixtures。

为避免循环 hash，`SEALED_CANONICAL_MANIFEST.json` 不反向包含 ordinary commitment hash；相反，`GSE246381_COMMITMENT.json` 必须绑定该 root manifest 的 full-file SHA256、`SEALED_CANONICAL_SHA256SUMS` SHA256、D1 prefix `ACCESS_MANIFEST.json` full-file SHA256、D1 prefix `ACCESS_SHA256SUMS` SHA256、D1 access-chain root 与 `access_prefix_snapshot_id`，形成唯一非循环发布根。任何把 live log current hash冒充 D1 snapshot、把 access binding 只塞进 canonical manifest、另造未命名 checksum sidecar、覆盖历史 prefix bundle，或让 checksum ledger包含自身/形成循环引用的实现均 FAIL。

普通路径的**每一个 row-level artifact**均不得包含任何 GSE246381 record，至少包括 sequences、observation candidates/observations、EndpointRegistry raw mappings、relation candidates/pairs、rejections、transformation/supersession edges、current-canonical/effective-exposure projections、Exposure/UseRole、group registry/assignments 与未来新增 support table；它们只通过不可逆 commitment/aggregate report 指向 restricted sealed store。每类分别运行 accession/record-ID/source-file/source-lineage negative test，不能只证明 ordinary loader 返回 0；validator 必须从 canonical manifest 枚举全部 row-level outputs 与允许的 aggregate-only outputs做 exact set equality，禁止维护一份会过时的“九类白名单”。ordinary `dataset_reconciliation/data_units_report` 只允许预注册 aggregate；若含 row locator/member ID/reversible order 则只能用 restricted mirror。大型内容可位于 run root，repo 中只存 manifest/hash/adapter；不得未经许可把第三方 raw 跟踪进 Git。

**GSE246381 D1 early-seal 强制验收：**

```text
restricted_store_outside_ordinary_worktree=true
ordinary_loader_gse246381_record_count=0
ordinary_report_row_level_gse246381_count=0
ordinary_sequence_entities_gse246381_record_count=0
ordinary_functional_observations_gse246381_record_count=0
ordinary_functional_observation_candidates_gse246381_record_count=0
ordinary_relation_candidates_gse246381_record_count=0
ordinary_edit_pairs_gse246381_record_count=0
ordinary_exposure_use_supersession_projection_gse246381_record_count=0
ordinary_all_row_level_artifact_negative_lineage_tests=PASS
ordinary_row_level_artifact_expected_set_mismatch=0
active_loader_reachable_legacy_copy_count=0
sealed_manifest_checksum_access_log_closure=PASS
sealed_access_log_jcs_chain_schema_root_closure=PASS
d1_access_prefix_snapshot_immutable=true
d1_access_live_prefix_match=true
d1_access_manifest_self_hash_rule_match=true
historical_access_snapshot_byte_drift_count=0
aggregate_qc_executable_hash_allowlisted=true
aggregate_qc_output_schema_allowlisted=true
v3_restricted_builder_machine_access_count=<logged-integer>
v3_aggregate_qc_machine_access_count=<logged-integer>
v3_fm_overlap_machine_access_count=0
v3_b0_eligibility_split_machine_access_count=0
v3_g7_restricted_finalizer_machine_access_count=0
v3_human_sequence_view_count=0
v3_human_label_view_count=0
v3_train_access_count=0
v3_tuning_access_count=0
v3_model_selection_access_count=0
v3_internal_test_evaluator_attempt_count=0
v3_internal_test_evaluator_completion_count=0
v3_pre_final_error_analysis_count=0
v3_one_time_final_attempt_count=0
v3_one_time_final_evaluator_count=0
```

历史不可变 per-record artifacts 可以保留，但必须移出所有 active loader roots；`SUPERSEDED_NOT_LOADABLE` 标记必须由 negative loader test 证明，不能只靠文档声明。

**D1-R Acceptance：**G0、G1、G2、G3A、G4 PASS；G3B 明确为 `DEFERRED_TO_FM0_A`，不得伪装 PASS。schema/fallback errors=0；raw/relation/observation-candidate lifecycle 与 E accepted-current-leaf/F label-complete-current-leaf technical expected sets conservation=100%，其 B0 disposition均仍为 `AWAITING_B0_GLOBAL_DISPOSITION`；observation-candidate↔observation 与 candidate↔pair endpoint/context/FK/payload/contributor mismatch=0；supersession/current-leaf、baseline/effective Exposure/UseRole/rights 与 access-log chain counters全为0；orphan entities/edges=0；`DATA_GOAL_BLOCKER_CLOSURE.jsonl` 中所有 D1-scoped checklist 均有 phase evidence，但需要 FM0/B0/G7 才能完整关闭的 blocker ID 不得提前写 CLOSED；声明仅为 `D1_CANONICAL_CLEANING_COMPLETE_PRE_SPLIT`。E 表分别给出 relation candidates、accepted current-leaf endpoint/context pair instances、unique design-relation groups、delta-eligible、link-only、no-edit、dense overlay、independent parents/genes/contexts；F 表给出 observation candidates五态、current-leaf unique labeled sequences/functional observations、biological replicates、technical replicates、barcodes/UMIs、contexts与attrition；canonical/exposure-use/supersession manifests/checksums原子冻结。

## 14.6 FM0-A：外部 backbone 暴露预检

- [ ] 生成 ordinary `<fm0_run_root>/foundation_candidates.json`、`foundation_exposure_ledger.jsonl`、`FOUNDATION_EXPOSURE_MANIFEST.json`、`FOUNDATION_EXPOSURE_SHA256SUMS` 与 `foundation_overlap_reports/`；每条 ledger row 以 object/cluster×checkpoint revision/weights hash 为唯一键；
- [ ] GSE246381 member-level ledger/reports 只生成于 `<restricted_run_root>/sealed_external/GSE246381/fm0/` 的同名 restricted files；ordinary 只接收 `data/v3_1/sealed_commitments/GSE246381_FM0_AGGREGATE.json` 与 `GSE246381_FM0_COMMITMENT.json`，不得含 member IDs；
- [ ] 只审计 `from_scratch_E`、`supervised_F_to_E` 与最多一个 general、每区域最多一个 specialist external backbone；
- [ ] 记录 model ID/revision/weights hash/license/pretraining corpus sources；
- [ ] 对 frozen D1 sequence clusters 做 exact/near overlap；
- [ ] 单独记录 GSE246381 foundation sequence/label overlap；项目侧未暴露不能代替该审计；
- [ ] FM0-A 的 ordinary/restricted live access logs 在所有本 phase events 完整 terminal 后分别冻结 `<fm0_snapshot_id>` immutable prefix bundles；FM0 commitments 绑定 prefix manifests/checksums/chain roots，后续 B0/G7 append 不得改变这些 bytes；
- [ ] unknown/PENDING/overlap-positive backbone 只使该 checkpoint/claim INELIGIBLE，不得全局删除数据；final alias 必须属于 frozen overlap-clean eligible checkpoint set；
- [ ] 所有 external backbone 不合格时回退 `from_scratch_E`，或仅在 F↔E final-lineage overlap audit clean 时回退 `supervised_F_to_E`；不得恢复 U-track。

**Acceptance：**G3B PASS，或每个未闭合 external checkpoint 明确 `INELIGIBLE` 且冻结 from-scratch/lineage-clean supervised fallback；ordinary/restricted `foundation_candidates`、per-checkpoint ledgers、overlap reports、manifest/checksums、immutable access-prefix bundles、restricted commitment、license matrix、STATUS/MANIFEST/SHA256SUMS 闭合；`foundation_checkpoint_key_collision=0`、`final_alias_outside_eligible_checkpoint_set=0`、`access_live_prefix_mismatch_count=0`、`historical_access_snapshot_byte_drift_count=0`。FM0-A 不训练模型，不发生 analytic/final-evaluator access；GSE246381 overlap 只由 restricted script 返回 allowlisted aggregates。

## 14.7 B0-R：benchmark v1、split 与 final seal

**Required machine artifacts：**

- `docs/execution/task_registry_v3_1.yaml`
- `docs/execution/split_registry_v3_1.yaml`
- `docs/execution/task_split_contract_matrix_v3_1.yaml`
- `docs/execution/diagnostic_registry_v3_1.yaml`
- `docs/execution/resource_viability_rule_v3_1.yaml`
- `data/v3_1/benchmark/B0_ROLE_DECISION_EVIDENCE.jsonl`
- `data/v3_1/benchmark/GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl`
- `data/v3_1/benchmark/ACTIVATION_CALIBRATION_MASK.jsonl`
- `data/v3_1/benchmark/ELIGIBILITY_MANIFEST.jsonl`
- `data/v3_1/benchmark/TASK_ACTIVATION_DECISIONS.jsonl`
- `data/v3_1/benchmark/SPLIT_ACTIVATION_DECISIONS.jsonl`
- `data/v3_1/benchmark/TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl`
- `data/v3_1/benchmark/TASK_ELIGIBILITY_UNIVERSE.jsonl`
- `data/v3_1/benchmark/RELATION_ROLE_TRANSITIONS.jsonl`
- `data/v3_1/benchmark/EFFECTIVE_ROLE_PROJECTION.jsonl`
- `data/v3_1/benchmark/SPLIT_ASSIGNMENTS.jsonl`
- `data/v3_1/benchmark/B0_ORDINARY_PREPARED_MANIFEST.json`
- `data/v3_1/benchmark/B0_TRANSACTION_COMMITS.jsonl`
- `data/v3_1/benchmark/LEGACY_B0_INVALIDATION_MANIFEST.json`
- `data/v3_1/benchmark/B0_MANIFEST.json`
- `data/v3_1/benchmark/B0_SHA256SUMS`
- `data/v3_1/benchmark/RESOURCE_VIABILITY_ASSESSMENT.json`
- `data/v3_1/exposure/access_snapshots/<b0_snapshot_id>/ORDINARY_ACCESS_LOG.jsonl`
- `data/v3_1/exposure/access_snapshots/<b0_snapshot_id>/ORDINARY_ACCESS_MANIFEST.json`
- `data/v3_1/exposure/access_snapshots/<b0_snapshot_id>/ORDINARY_ACCESS_SHA256SUMS`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/B0_ROLE_DECISION_EVIDENCE.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/ELIGIBILITY_MANIFEST.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/TASK_ELIGIBILITY_UNIVERSE.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/RELATION_ROLE_TRANSITIONS.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/EFFECTIVE_ROLE_PROJECTION.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/SPLIT_ASSIGNMENTS.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/B0_RESTRICTED_PREPARED_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/B0_RESTRICTED_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/benchmark/B0_RESTRICTED_SHA256SUMS`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<b0_snapshot_id>/ACCESS_LOG.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<b0_snapshot_id>/ACCESS_MANIFEST.json`
- `<restricted_run_root>/sealed_external/GSE246381/access_snapshots/<b0_snapshot_id>/ACCESS_SHA256SUMS`
- `data/v3_1/sealed_commitments/GSE246381_B0_AGGREGATE.json`
- `data/v3_1/sealed_commitments/GSE246381_B0_COMMITMENT.json`
- `data/v3_1/exposure/projections/<b0_snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`
- `<restricted_run_root>/sealed_external/GSE246381/exposure/projections/<b0_snapshot_id>/EFFECTIVE_EXPOSURE_PROJECTION.jsonl`

Task/Split activation 与 120-row applicability decisions 是不含 member/object IDs 的全局 registry-level artifacts，只生成一份 ordinary copy；不得按 cohort 分叉成两个相互冲突的 matrix。其他 ordinary row-level artifacts 只含非 sealed objects，GSE246381 row-level artifacts 只含于 restricted shard，普通目录分别证明其 count=0。即使某个 shard 本轮没有角色变化，也必须生成可验证的空 transition ledger、其 SHA256 与由 base roles 重建的 projection；不得以“无事件”为由省略 artifact。ordinary B0 manifest 通过 restricted prepared/commitment/hash 与后者联合闭合，不复制 member IDs。

Stage6 PREPARED 的 logical component universe 不得由实现自己选择。ordinary set 精确为 `ACTIVATION_CALIBRATION_MASK/B0_ROLE_DECISION_EVIDENCE/EFFECTIVE_EXPOSURE_PROJECTION/EFFECTIVE_ROLE_PROJECTION/ELIGIBILITY_MANIFEST/FIVE_SCALE_DATA_CARD/FOUNDATION_EXPOSURE_LEDGER_MANIFEST/GLOBAL_ELIGIBILITY_DECISION_EVIDENCE/GSE246381_B0_AGGREGATE/GSE246381_B0_COMMITMENT/LEGACY_B0_INVALIDATION_MANIFEST/ORDINARY_ACCESS_PREFIX_MANIFEST/RELATION_ROLE_TRANSITIONS/RESOURCE_VIABILITY_ASSESSMENT/SPLIT_ACTIVATION_DECISIONS/SPLIT_ASSIGNMENTS/TASK_ACTIVATION_DECISIONS/TASK_ELIGIBILITY_UNIVERSE/TASK_SPLIT_APPLICABILITY_DECISIONS`；按 logical ID UTF-8 字典序、每项 LF 的 set SHA256=`645042cc476710448f4f5b70c80c8cd624c4ea44177eea48d22233fd575545d8`。restricted set 精确为 `ACCESS_PREFIX_MANIFEST/B0_ROLE_DECISION_EVIDENCE/EFFECTIVE_EXPOSURE_PROJECTION/EFFECTIVE_ROLE_PROJECTION/ELIGIBILITY_MANIFEST/FOUNDATION_EXPOSURE_LEDGER/GLOBAL_ELIGIBILITY_DECISION_EVIDENCE/RELATION_ROLE_TRANSITIONS/SPLIT_ASSIGNMENTS/TASK_ELIGIBILITY_UNIVERSE`，hash=`00ebb4bb9090ed74c2d37a424773edd2b4216e50fec084013d978469fcb9b3ff`。`ORDINARY_ACCESS_PREFIX_MANIFEST` 与 `ACCESS_PREFIX_MANIFEST` 必须分别映射到 `<b0_snapshot_id>` immutable prefix bundle 的 `ORDINARY_ACCESS_MANIFEST.json` 与 `ACCESS_MANIFEST.json`，绝不能映射 live log；manifest 依 §3.2 单向绑定其 snapshot log/checksum ledger/object-set closure。PreparedManifest 必须对每个 logical ID 恰好映射一个 physical path+SHA256；missing/extra/duplicate logical ID、同一 path承担两个 IDs或 hash漂移均 FAIL。PreparedManifest自身、root commit row及其后生成的 final `B0_MANIFEST/B0_SHA256SUMS` 不属于 pre-commit component set，避免自引用；它们分别由 root/finalizer另行绑定。

历史 `data/data_exposure_ledger.jsonl`、`data/b0_04_eval_track_manifest.jsonl` 与 `data/b0_splits/*.jsonl` 必须只读 hash、原样保留，不得 modify/replace。`LEGACY_B0_INVALIDATION_MANIFEST.json` 逐文件记录 old hash、legacy semantic defect、`SUPERSEDED_NOT_LOADABLE`、replacement v3.1 locator/hash 与 loader-negative-test evidence；所有 v3.1 输出只写 `data/v3_1/` 或 restricted run root。任何 active loader 读取旧 B0 文件即 FAIL。

### B0-00 不可循环的原子决策顺序

每个 B0 run 必须在 unique staging root 按下列七阶段执行；任何 stage output 在 root commit row 出现前都不是权威状态，禁止让 role、global eligibility、task activation、cell 与 split 互相循环引用：

1. **Definition/pre-role facts：**验证 C3 definitions/hashes；只对 current-leaf accepted E PAIR，从 frozen canonical/EffectiveExposureProjection contributor-rights/FM0/isolation/evaluator-freeze 与 sealed cohort **evaluation-compatibility predicates** 生成 `B0_ROLE_DECISION_EVIDENCE.jsonl`。这些 predicates 只决定 global sealed/excluded role 与 sealed-final split readiness，不是 TaskActivation/metric facts；F observation 不进入该文件；
2. **Staged role：**`RelationRoleTransition` 仅引用 Stage1 evidence，写 staged events 并重放 staged effective-role projection；event 不携带、不修改 eligibility；
3. **Global purpose evidence/eligibility：**对所有 current-leaf accepted E pair/F observation 生成 `GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl`，随后一对一生成 EligibilityRecord；E 使用 staged projection，F 使用 `NOT_APPLICABLE_OBSERVATION`；global-excluded 对象到此停止，不进入 cells；
4. **Activation/applicability：**先以 C3 outcome-blind rule 生成并 hash ordinary nonsealed `ACTIVATION_CALIBRATION_MASK.jsonl`，记录 `TASK_PROTOCOL_CALIBRATION` access，再在同一 snapshot 上生成 12-row TaskActivationDecision、10-row SplitActivationDecision 与全局 120-row ApplicabilityDecision；task/metric decisions只使用该 frozen calibration population，sealed/internal-test contribution=0；restricted allowlisted aggregate只能决定 `sealed_final_v1` split readiness，且任何 decision都不能反向改变 Stage1/2 role；
5. **Cells/assignments：**对 `global_disposition=ACTIVE × APPLICABLE` 的每个 expected key生成一个 TaskEligibilityCell；只有 cell disposition=`ASSIGNED_TO_SPLIT` 时才生成恰一 SplitAssignment，`INELIGIBLE_WITH_REASON|PENDING_WITH_REASON` 均不得有 assignment；assignment 还必须执行 calibration-component DEVELOPMENT lock 与跨 task 同 split-contract partition一致性；
6. **Prepare/validate：**验证 schema、exact sets/FKs/rights/role/partition/leakage/access chain 与全部 hashes，分别写 immutable ordinary/restricted PREPARED manifests；prepared 不等于 committed；
7. **Two-phase root commit：**coordinator 验证 ordinary prepared hash、restricted prepared hash、restricted access-log chain root、ordinary commitment hash 与 component set 后，最后 append `B0_TRANSACTION_COMMITS.jsonl` root commit row。所有 loader/projection/eligibility 必须同时看到该 row 与两个 prepared hashes 才接受 transaction。

Stage1 row 必须符合 `eligibility_record.schema.json#/$defs/B0RoleDecisionEvidence`，并严格限定 `object_type=PAIR/scientific_track=E`。最低字段为 `evidence_id/run_id/transaction_id/object_id/canonical_gate/training_permission_fact/evaluation_permission_fact/derived_release_permission_fact/raw_redistribution_permission_fact/fm0_gate/sealed_evaluation_compatibility_results/isolation_gate/evaluator_freeze_gate/proposed_role_decision/controlled_reason/canonical_manifest_sha256/effective_exposure_projection_sha256/exposure_use_manifest_sha256/license_matrix_sha256/fm0_manifest_sha256/task_registry_definition_sha256/split_registry_definition_sha256/task_split_definition_matrix_sha256/code_commit/config_hash/evidence_sha256`。`sealed_evaluation_compatibility_results` 每项为 `target_task_id/rule_sha256/pre_role_compatible/reason/input_fact_hash`，只能使用 canonical/rights/FM/task-definition compatibility predicates，明确禁止引用 effective role、EligibilityRecord、Task/Split activation decision、cell、partition、activation-calibration outcomes 或 model/final result。GSE sealed scope 固定为 `T5_GEN_RECONSTRUCT_E_PAIR|T5_RANK_CLOSED_SELECT_E_PAIR`，冻结 aggregation rule 要求至少一个 target compatible；Stage4 不得把这些 sealed facts用于 task activation/metric选择。读取 permission 时只使用上述同名 `*_permission_fact`，不得读取尚未生成的 EligibilityRecord。`proposed_role_decision=<NO_TRANSITION_KEEP_BASE|TRANSITION_TO_SEALED_EXTERNAL_FINAL|TRANSITION_TO_EXCLUDED|OTHER_ALLOWED_MATRIX_TRANSITION>`。

Stage3 row 按 `eligibility_record.schema.json#/$defs/GlobalEligibilityDecisionEvidence` 验证，覆盖 PAIR/OBSERVATION，至少含四类 raw permission facts、完整 contributor set/hash、EffectiveExposure/FM per-checkpoint facts、PAIR projection 或 F sentinel、proposed global disposition 与四类 purpose eligibility inputs；它不含 task cell/partition。Stage4 calibration mask/三类 decision rows 分别按 `task_registry.schema.json#/$defs/ActivationCalibrationMaskRow`、`#/$defs/TaskActivationDecision`、`split_registry.schema.json#/$defs/SplitActivationDecision`、`task_split_applicability.schema.json#/$defs/TaskSplitApplicabilityDecision` 验证。Stage2 projection、Stage6 prepared manifests、Stage7 commit 分别按 `relation_role_transition.schema.json#/$defs/EffectiveRoleProjection`、`#/$defs/B0PreparedManifest` 与 `#/$defs/B0TransactionCommit` 验证。B0PreparedManifest 最低字段为 `prepared_manifest_id/transaction_id/run_id/store_shard=<ORDINARY|RESTRICTED_GSE246381>/parent_committed_transaction_id/component_paths_and_sha256s/component_set_sha256/canonical_snapshot_sha256/effective_exposure_projection_sha256/foundation_ledger_manifest_sha256/access_log_chain_root_sha256/prepared_at/preparer_executable_sha256/validation_report_sha256/prepared_manifest_sha256`；restricted/ordinary field nullability 用 `oneOf` 锁定。

root commit row 至少含 `commit_sequence_no/transaction_id/run_id/parent_committed_transaction_id/predecessor_commit_record_sha256/ordinary_prepared_manifest_sha256/restricted_prepared_manifest_sha256/restricted_access_log_chain_root_sha256/restricted_commitment_sha256/task_activation_decisions_sha256/split_activation_decisions_sha256/task_split_applicability_decisions_sha256/ordinary_component_set_sha256/restricted_component_set_sha256/finalizer_executable_sha256/committed_at/commit_record_sha256`。GENESIS 使用固定 predecessor sentinel；后续 `commit_sequence_no` 必须连续且 parent transaction/hash 唯一，`commit_record_sha256` 用 RFC8785/JCS 覆盖除自身外全 row。ordinary worktree 与 restricted run root 可能跨文件系统，禁止假定一个 rename 能原子覆盖两者；只能先双 PREPARED、再由 root marker 提交。任一阶段失败保留 parent-linked `UNCOMMITTED_FAILED` run；orphan/half-prepared bundle 永远被 loader 忽略，修复使用新 transaction 从 last committed projection 重做，不需要也不得伪 rollback terminal role。

强制 `task_activation_decision_expected_set_mismatch=0`、`duplicate_task_activation_decision=0`、`split_activation_decision_expected_set_mismatch=0`、`duplicate_split_activation_decision=0`、`applicability_decision_key_set_mismatch=0`、`duplicate_applicability_decision=0`、`decision_definition_fk_or_hash_mismatch=0`、`decision_run_snapshot_mismatch=0`、`sealed_contribution_to_task_activation_or_metric_selection=0`、`internal_test_contribution_to_task_activation_or_metric_selection=0`、`selected_metric_not_in_candidate_set=0`、`decision_sha_mismatch=0`、`prepared_component_expected_set_mismatch=0`、`prepared_component_duplicate_logical_id_or_path=0`、`orphan_prepared_bundle_in_loader=0`、`half_commit_detected=0`、`root_commit_component_set_mismatch=0`、`root_commit_sequence_gap=0`、`root_commit_predecessor_hash_mismatch=0`、`root_commit_fork_count=0`、`root_commit_record_hash_mismatch=0`；所有 `$defs` 各有 positive/negative/cycle-dependency/sealed-noninterference/half-commit/hash-chain golden fixtures。

### B0-01 Eligibility freeze

- [ ] 先验证 §5.7 冻结的 12-task/10-split expected sets、task/split semantic descriptor/allowlist hashes、完整 120-row definition matrix 与 DiagnosticRegistry；随后生成并验证 12/10/120-row activation/applicability decisions；registry 缩水、条件 row 删除、F task 缺失、direction/cohort rule 漂移或 object type/track/estimand 漂移均 FAIL；
- [ ] ordinary/restricted 各自先生成 `GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl`，再生成 `ELIGIBILITY_MANIFEST.jsonl`；每个 current-leaf global E pair/F observation 一行，固定 scientific track、canonical/global disposition、四类许可、四类 eligibility、E effective global role/F N/A sentinel、完整 contributor-rights evidence 与全部 input hashes；该文件绝不含 task/split/partition 字段；
- [ ] 从 frozen definitions 与 B0 activation/applicability decisions 按 §6.4/§5.7 唯一规则生成并 hash `TASK_ELIGIBILITY_UNIVERSE.jsonl`；主键严格为 `object_id×task_id×split_contract_id`；只对 global ACTIVE objects 与 decision=`APPLICABLE` 生成三态 cell，N/A 只留在 decision 层，global-excluded/pending 对象绝不生成 cell，禁止靠缺 row 隐去 expected cell；
- [ ] `permitted_*=NO|UNKNOWN` 对相应用途均 fail closed；EligibilityRecord 与 global object 一一对应，TaskEligibilityCell 必须通过 `eligibility_record_id` FK 连接，不得复制一个含混的 per-record `task` 字段；
- [ ] `canonical_status=ACCEPTED` 只证明技术重建，不代表允许 train/eval/derived release/raw redistribution；Data Card 分别报告 canonical-available、train-eligible、evaluation-eligible、derived-release-eligible、raw-redistribution-eligible 五种规模；
- [ ] k_edit>5、无功能 label、ambiguous pair、CDS/full-mRNA auxiliary 不进入不适用的 metric 分母；
- [ ] 两层分别守恒：global 层每个 accepted E pair/label-complete F observation 只计一次，分为 active/globally-excluded/global-pending；cell 层每个 expected key 恰为 assigned-to-one-split/ineligible+reason/pending+reason；禁止用多任务 assignment rows 回填 global denominator；
- [ ] definition/activation/applicability exact-key/uniqueness/FK/hash/selected-metric mismatch 全为 0，global eligibility uniqueness/FK/hash mismatch=0、`duplicate_task_eligibility_cell=0`、`missing_expected_task_eligibility_cell=0`、`globally_excluded_object_with_cell=0`、global pending=0，且每个 applicable formal `task_id×split_contract_id` 的 cell pending=0；禁止漏掉当前 4,012 条一类的 E 缺口或任何 F observation；

### B0-02 Split freeze

- [ ] 分别构建 5′ source/study/F-sequence-cluster、3′ source-or-variant/study/F-sequence-cluster、cross-region、heldout-context conditional row 和 sealed-final；10 个 SplitRegistry IDs 一个不能少；
- [ ] study-disjoint 与 region shift 分开，不能用 5′→3′代替 study split；
- [ ] source/candidate、parent、gene/transcript/tile、sequence cluster、library lineage 原子化；
- [ ] GSE246381 默认只进入 isolated sealed-final candidate manifest；不进入 train/dev。
- [ ] 只有 D1 technical canonical accepted、predecision 中 `evaluation_permission_fact=YES`、G3B/FM0-A、frozen sealed evaluation-compatibility aggregation、B0 isolation 与 evaluator-freeze 全部 PASS 时，staged `RelationRoleTransition` 才提出 GSE246381 role 从 `SEALED_EXTERNAL_FINAL_CANDIDATE` 升级为 `SEALED_EXTERNAL_FINAL`；这些 sealed facts不得参与 task activation/metric选择。Stage3 EligibilityRecord 写 `evaluation_eligibility=ELIGIBLE`，Stage4/5 至少一个 frozen sealed target task ACTIVE/APPLICABLE 且对应 SEALED_FINAL cells/assignments合法后，Stage7 root commit 才使该 transition 生效。任一 pre-role gate 失败时只 staged role→`EXCLUDED`，Stage3 EligibilityRecord 写 `global_disposition=GLOBALLY_EXCLUDED_WITH_REASON` 与 purpose INELIGIBLE，绝不生成对应 cells/assignments；role event 不携带 eligibility，eligibility 也不改 role/canonical，且无论哪条路径都绝不转 train/dev；
- [ ] 生成 `SPLIT_ASSIGNMENTS.jsonl` 并验证 assigned cell↔assignment 双向一一对应、partition FK、group snapshot 和 assignment hash；执行 §5.7 global-role×partition-role 与 global-eligibility×partition-role 矩阵，所有 conflict counters=0；
- [ ] 从 frozen base roles 重放全部 transition hash chain；§5.5.1 的 duplicate/missing/predecessor/invalid/fork/mutation/projection counters 全为 0，ledger/projection SHA256 进入 B0 manifest 与 checksum ledger；

### B0-03 Leakage、exposure 与 coverage

- [ ] exact/reverse/near、source/pair/parent、gene/family、tile/scaffold/library、context/barcode、foundation overlap 与实际冻结/物化/采样的 latent intermediate states 全审计；不得声称覆盖所有理论上可能的 latent paths；
- [ ] required dimension 未知即阻断或缩小 claim，不得 N/A+PASS；
- [ ] GSE246381 active ledger 必须同时表达 prior use=`NONE_CONFIRMED` 与 pipeline materialization=`PRESENT`；不得再出现 E4X；
- [ ] report 必须绑定 current contract/canonical/eligibility/exposure/split/FM0 hashes。

### B0-04 Task/evaluator freeze

- [ ] 按 §5.7 精确 12-task set 分别冻结 E benchmark evaluation、F auxiliary/context/property、cross-region 的资格与 estimand；benchmark tasks 冻结唯一 primary metric，auxiliary tasks 冻结 objective/monitoring 且不产生 primary claim；open-generation 只在 DiagnosticRegistry；禁止跨任务择优；
- [ ] 3′ ranking 未达 500 multi-candidate sources/5 studies/source≥5 时保持 exploratory；
- [ ] 冻结 candidate/query/edit/FLOPs/time budgets、bootstrap、seeds、missingness 和 failure handling；
- [ ] 建立与开发角色分离的 sealed evaluator；从 D1 已开始的限制继续生效，普通 loader 无法读取 GSE246381 的逐条 sequence、labels 或 join；
- [ ] GSE246381 的 schema/mapping/reconstruction/QC defect 只能在 D1 使 technical canonical `PENDING|EXCLUDED_WITH_EVIDENCE`；B0 的许可/FM/global sealed-evaluation-compatibility/isolation 失败不得改写 frozen technical canonical，只能在 restricted EligibilityRecord 写 global exclusion/purpose ineligibility、staged restricted role event 写 effective role=`EXCLUDED`，且不生成任何 cell/assignment；只有对象仍 global ACTIVE 而某个具体 task 不合格时才生成 `INELIGIBLE_WITH_REASON` cell。上述 restricted rows均不进入普通 row-level artifact。B0 若发现新的 canonical defect，必须 `STALE_INVALIDATED` 并以新 run 重开 D1。

### B0-05 Data Card 与 operational seal

- [ ] E 表和 F 表分开；禁止统一 N_total；
- [ ] 每个 N 有 stable concept、unit、denominator、dataset、analysis population 和 evidence locator；
- [ ] 每轨分别列 canonical-available、train-eligible、evaluation-eligible、derived-release-eligible、raw-redistribution-eligible；不得用单一 releasable 合并后两类；
- [ ] 分别报告 global unique-object denominator 与逐 `task_id×split_contract_id` eligibility-cell denominator、三类 disposition 和最终分析分母；禁止把二者合并或把 assignments 当 unique objects；
- [ ] ordinary Data Card 只接收 GSE246381 的 allowlisted aggregate denominator/counters/status/commitment hash，不含 member ID、逐条 role/eligibility/split 或可逆排序；ordinary/restricted totals、manifest hashes 与 access-log counts 联合守恒；
- [ ] 输出 `OPERATIONALLY_SEALED_RETROSPECTIVE_FINAL` manifest、custodian/evaluator policy、commitment hash；
- [ ] 任何 final 访问事件写 append-only access log；意外访问立即使 v1 `FINAL_INVALIDATED`。
- [ ] 生成 `RESOURCE_VIABILITY_ASSESSMENT.json`，与工程/data closure 正交地给出 `resource_viability_status=<PUBLICATION_GRADE_CANDIDATE|LIMITED_DEVELOPMENT_ONLY|NOT_VIABLE>`。`PUBLICATION_GRADE_CANDIDATE` 不是“保证发表”，最低要求为：至少一个 5′ primary benchmark task ACTIVE 且 confirmatory；该 task 在冻结 analysis unit 上 ≥500 个独立 units、≥5 个独立 studies/libraries、任何 action-specific claim 每 stratum≥100、source-disjoint 与 study-disjoint evaluation partitions 均非空、预注册 group-aware CI precision PASS、无单一 study/library 占 eligible denominator>70%；并明确 3′ 为 `QUALIFIED_EXTENSION|EXPLORATORY_ONLY|NOT_VIABLE`。若 closure PASS 但上述不满足、仍有至少一个非空 5′ descriptive/exploratory evaluation，则为 `LIMITED_DEVELOPMENT_ONLY`；若没有任何非空 5′ source/study-disjoint evaluation task，则 `NOT_VIABLE`。所有 counts 使用 task 的 frozen analysis unit/dedup key，不用 rows/barcodes/endpoint重复实例膨胀；阈值只能在 C3 用户批准的 `resource_viability_rule_v3_1.yaml` 中冻结，B0 不得因结果修改。

**B0-R Acceptance：**G0、G1、G2、G3A、G3B、G4、G5、G6 PASS；21-schema/12-task/10-split definitions、semantic/direction/cohort hashes、120-row definition/effective-decision exact key sets、DiagnosticRegistry 与 12/10 activation decisions全闭合；E/F global unique-object coverage=100%，每个 applicable formal task eligibility-cell coverage=100%，global/per-task pending=0；global EligibilityRecord、cell、SplitAssignment、committed role/exposure projections 的 uniqueness/FK/hash/partition-compatibility counters 全为 0；candidate↔pair base payload/effective-role mismatch=0；ordinary/restricted PREPARED+root transaction commit、dual-store overlap/leakage/missing commitment/half-commit counters全为 0；required leakage=0；unknown required dimensions=0；B0 ordinary/restricted access-prefix manifests均映射 `<b0_snapshot_id>` immutable bundles，live-prefix mismatch=0、历史 D1/FM0 snapshot byte drift=0；GSE246381 的 human-sequence/human-label/train/tune/model-selection/internal-test/pre-final-error-analysis/final-attempt/final-evaluator counts 均为 0，restricted-builder/aggregate-QC/FM-overlap/B0-eligibility-split/G7-finalizer machine access 与 JCS append-only chain逐事件一致；five-scale Data Card 与 ordinary+restricted manifests/checksums一致。工程 Acceptance 与 resource viability 分开报告；只有 `PUBLICATION_GRADE_CANDIDATE` 可进入 publication-oriented model-rebind terminal，其他状态不得靠 schema PASS 宣称数据足以发表。

## 14.8 G7：fresh closure 与 Goal 终态

- [ ] 从 frozen inputs 全量重跑 D1 validation、FM0-A overlap、B0 eligibility/splits/leakage/task validity；
- [ ] 旧 D1/B0/exposure reports 全部标 `STALE_INVALIDATED`，不参与 PASS；
- [ ] 运行 contract tests、schema tests、dataset adapter tests、conservation tests、split/leakage tests、final-access tests；
- [ ] ordinary finalizer 重算 observation-candidate↔observation、candidate↔pair endpoint/context identity、supersession current-leaf projection、baseline Exposure/UseRole/rights、ordinary access chain、per-checkpoint foundation ledger、effective exposure、global eligibility evidence/EligibilityRecord、activation-calibration mask与其 access、12/10 activation、120 applicability decisions、TaskEligibilityUniverse、committed role projection、SplitAssignment、cross-task partition consistency与 role/eligibility-partition compatibility；restricted `G7_RESTRICTED_FINALIZER` 在 sealed store 内重放同一链，只向普通工作区输出 `GSE246381_G7_COMMITMENT.json` 与 allowlisted aggregates；禁止只复用阶段报告中的 PASS；
- [ ] 所有 G7 machine events 有唯一 terminal row 后，分别冻结 `<g7_snapshot_id>` ordinary/restricted immutable access-prefix bundles；G7 manifests/commitment 绑定 prefix manifests、checksum ledgers、chain roots与前一 D1/FM0/B0 snapshot IDs/hashes，并证明历史 snapshot byte drift=0；
- [ ] 生成 `OUTPUT_MANIFEST.json`、`STATUS.json`、`SHA256SUMS`、`GOAL_REPORT.md`；
- [ ] 生成 `DATA_GOAL_BLOCKER_CLOSURE.jsonl`，逐项覆盖 `data_goal_required_blocker_ids` 且全部 `CLOSED_WITH_EVIDENCE`；另生成 `MODEL_REBIND_HANDOFF_BLOCKERS.jsonl`，逐项覆盖 `model_rebind_handoff_blocker_ids`，允许 OPEN 但必须有 evidence/current paths/closure condition/owner；两个集合各自 set-equality=TRUE、交集为空；
- [ ] finalizer 逐项核对合同 hash、full commit、clean worktree、inputs/outputs、21-schema expected set及所有 required `$defs` fixtures、12-task/10-split semantic definitions、120-row definition/decision exact keys、DiagnosticRegistry、ordinary/restricted baseline/effective exposure/use/current-leaf/global-eligibility/cell/transition/commit/projection/SplitAssignment hashes、PREPARED/root-commit hash chain/dual-store commitments、失败 parent lineage、`analytic_final_labels_accessed=false`、human-sequence/human-label/train/tune/model-selection/pre-final-error-analysis/final-attempt/final-evaluator counts=0，以及五类 machine access JCS chain/schema/root hash；
- [ ] 只有全部 data gates PASS 且 `resource_viability_status=PUBLICATION_GRADE_CANDIDATE` 才生成 `DONE`，并写 `DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND`；若工程 closure PASS 但 viability=`LIMITED_DEVELOPMENT_ONLY|NOT_VIABLE`，写 `BLOCKED_WITH_EVIDENCE` 与对应 viability status、不生成 DONE，先交用户决定缩小论文范围或扩展数据；GP0 状态始终为 `LOCKED_NOT_AUTHORIZED`；
- [ ] 任一未闭合则写 `BLOCKED_WITH_EVIDENCE`，保留 stderr/exit code/reason，不生成 DONE。

### 14.8.1 Goal 交接清单

- [ ] 当前 branch/full commit 与 isolated worktree；
- [ ] contract/config/21-schema、Task/Split definitions、activation/applicability decisions、DiagnosticRegistry 与 expected-set/semantic hashes；
- [ ] E/F 数据单位总表与逐 dataset reconciliation；
- [ ] P0/P1 资产、许可与 exclusions；
- [ ] canonical/observation+relation lifecycle/supersession/exposure-use/global-eligibility/activation/task-cell/split-assignment/role-transaction/FM0/final manifests，以及 ordinary/restricted PREPARED/root commitments；
- [ ] G0–G7 reports、STATUS、MANIFEST、SHA256SUMS；
- [ ] `DATA_GOAL_BLOCKER_CLOSURE.jsonl` 与 `MODEL_REBIND_HANDOFF_BLOCKERS.jsonl`；
- [ ] 未解决风险和下一阶段禁止事项；
- [ ] 明确声明 `NO_GP0_TRAINING_PERFORMED`、`ANALYTIC_FINAL_LABELS_ACCESSED=false`、`GSE246381_PRIOR_ANALYTIC_USE=NONE_CONFIRMED_BY_OWNER`、`GSE246381_LEGACY_PIPELINE_MATERIALIZATION=PRESENT`、`NO_PROJECT_UNLABELED_PRETRAINING`、`GP0_STATUS=LOCKED_NOT_AUTHORIZED`。

## 14.9 后续阶段路线图；不属于 GOAL-V3-DATA-BENCH-01 范围且仍未授权

### FM0-B

在 B0 split freeze 后完成 common-support profiling、长度/参数/FLOPs 对齐和 adaptation contract，选择最多一个 general primary 与两个 specialist sensitivity；不读取 final labels。

### MK0/EF0-R

保留历史 E0 engineering，不从零重做；重新绑定 v3.1 schemas、fixed-budget primary、INS/SUB/DEL、dynamic current-state、alignment variants、region variants、source-only length policy、batching、CUDA/fallback/artifact closure。Acceptance 仍只为 E0 engineering。

### GP0-DEV

另立授权合同后按 PR2 三轮执行；formal neural training 必须真实 CUDA。每轮 immutable registry、相同 config budget、新 run root、失败保留；不得访问 final。

### FC0

每个 endpoint 单独训练 development evaluator；training reward、candidate selector 与 sealed final evaluator 三角色隔离，报告 reliability ceiling、replicate agreement、calibration、OOD 与 reward-hacking audit。

### MB0 / PR3

模型、fallback、seeds、container、budgets 与 evaluator command 冻结后，执行 matched-budget budget curves、完整 baselines、5 seeds 和一次性 final。final 一旦打开，v1 永久 consumed。

### ER0 / PP0

ER0 固定 failure taxonomy：shortcut、alignment sensitivity、artifact risk、context shift、region transfer、length/action、mode collapse、calibration、library ascertainment 和 data missingness。PP0 交付 versioned adapters/schemas、许可允许的数据或重建工具、code/container、baseline/evaluator、cards、reproducibility、limitations 和所有 negative/null results。

---

# 15. GPU、run、监控与故障合同

## 15.1 GPU-only 范围

formal neural foundation adaptation、GP0、neural critic、neural baseline training 和 neural final generation 必须在真实 CUDA 上。必须验证 model、input、forward、backward/optimizer 的 device，记录 GPU UUID、CUDA/driver、峰值显存和 CPU fallback count=0。`nvidia-smi` 存在不等于训练使用 GPU。

CPU 可用于 schema/replay/unit tests、manifest/audit、统计工具和显式 development smoke；linear/tree/exact-enumeration 等 CPU-native baseline 可 formal 执行，但必须记录 CPU 型号、threads、内存、wall time、代码/数据/config hash，且不得冒充神经 GPU training。

## 15.2 Run artifacts

每个 formal run 使用绝对路径与唯一 run ID：

```text
run_root/
  frozen_config.yaml
  contract.sha256
  source_commit.txt
  data_manifest.json
  eligibility_manifest.jsonl
  split_manifest.json
  exposure_ledger.json
  final_access_log.jsonl
  environment.json
  stdout.log
  stderr.log
  metrics.jsonl
  checkpoints/
  evaluation/
  STATUS.json
  MANIFEST.json
  SHA256SUMS
  DONE
```

`DONE` 只由 finalizer 在全部 gates/checksums/status 一致后生成。测试通过、GPU forward 或 checkpoint 存在不等于 DONE。

## 15.3 低频监控

启动前记录 run ID、PID、GPU、log/metrics/checkpoint 路径。正常长任务低频只读检查；在 exit、NaN/Inf、资源危险、checkpoint、连续 validation 无进展或进程异常时增加诊断。不得频繁读取大文件或干扰无关作业。

## 15.4 安全暂停与恢复

NaN/Inf、CPU fallback、OOM、数据/hash 变化、final label 意外访问、暴露违规、磁盘危险或无法确认 ownership 时立即安全暂停。保留日志和最后 checkpoint；修复后使用新 parent-linked run，不覆盖失败 run。

不自动修改 batch、长度、阈值、数据过滤或 seed 以“跑通”。任何科学相关变更先更新 decision log 并重开相应 gate。

---

# 16. 论文发布、stop rules 与最小可发表包

## 16.1 无湿实验时仍有发表价值的条件

benchmark/method paper 有价值，当且仅当：

1. 数据语义、provenance、许可、attrition、exposure 和 artifact 风险比已有 benchmark 更透明；
2. task 确实是 source-conditioned editing，而不是把随机库包装成 edit；
3. 5′/3′按真实数据结构定义非对称任务；
4. source/study/family split 显著减少 shortcut；
5. strong baselines、matched budgets、sealed final 和 group-aware statistics 完整；
6. 至少回答一条可复用科学/方法问题，即使 proposed method 没赢。

## 16.2 Stop rules

- GSE246381 的 owner-confirmed prior non-use、pipeline materialization、external-foundation overlap、sealed use role、split 和 evaluator policy 未分轴一致，或任何 active artifact 仍写 E4X/historically exposed：停止 formal training 与 final。
- source measurement join 未完成：禁止 delta/improvement 主张。
- DATA-P0 未闭环：禁止 data-scale headline。
- final 不能真正封存：禁止正式 leaderboard/generalization claim。
- `UTR_ADAPTED_MULTI_QK` 只降低 estimator variance、但未改善预注册 quality-vs-draws/updates/FLOPs/time Pareto：不得作 multi-alignment positive headline，只报告 estimator/alignment sensitivity；不得改写成新 marginalization 理论。
- H6A 的 generator 未胜同一 T5-Gen-Reconstruct 目标下的 `BASIC_PAIR_EF_NATIVE` 与最强 editor：不作 generator-positive headline；H6B 只能在 frozen measured pool 内比较 scorer/ranking/search，二者不得互相补偿。
- 3′数据资格不足：降为 exploratory extension。
- 任一区域单独失败：禁止 UTR-general advantage。
- STOP 在 v1 固定 `DISABLED`；任何 STOP positive 或 biological termination 主张均直接违反合同。
- license 不允许发布且无可复现 adapter/evaluator：停止 benchmark release claim。
- 无湿实验：永久禁止 biological/therapeutic improvement。

## 16.3 最小发布包

- versioned benchmark v1 与 DOI/永久 release，或可复现 build adapters；
- source/license/use/redistribution matrix；
- frozen schemas、canonical manifest、exposure ledger、reject ledger；
- source/study/family splits 与 sealed evaluator；
- data card、model card、artifact-risk card；
- baseline API、container/environment lock；
- all-trials registry、统计脚本、final manifests/checksums；
- negative/null/failure 结果；
- limitations 与 prohibited-claims；
- 一条独立于 proposed method 胜负的 empirical finding。

## 16.4 推荐论文骨架

1. 为什么现有 mRNA/RNA benchmark 不能回答 source-conditioned UTR editing；
2. Track E/Track F、sequence scope、provenance、attrition 与 independent-unit accounting；
3. 5′/3′非对称任务和严格 splits；
4. Edit Flows 继承边界、UTR adaptation 与 alignment-proposal robustness；
5. matched-budget baselines 与一次性 final；
6. 共享/迁移、artifact/exposure/path sensitivity 的经验规律；
7. 无湿实验、library ascertainment、measurement error 和 generalization 限制。

---

# 17. Prior-art 与数据来源快照

本节用于发现与合同边界，不代替论文提交前逐条人类核验。提交前必须刷新。

## 17.1 关键方法与 benchmark 近邻

- Edit Flows v3（方法继承主来源）：<https://arxiv.org/html/2506.09018v3>
- EvoFlows（template-conditioned INS/DEL/SUB edit flows）：<https://arxiv.org/abs/2603.11703>
- Tree-Conditioned Edit Flows（tree/branch-conditioned paired edit flow）：<https://arxiv.org/abs/2605.04119>
- mRNAutilus（full-transcript、de novo UTR、guided diffusion/multi-objective 近邻）：<https://arxiv.org/abs/2605.31296>
- RNAGenScape：<https://arxiv.org/abs/2510.24736>
- Flexible Flows for Biological Sequence Design：<https://arxiv.org/abs/2606.10543>
- LPDP：<https://arxiv.org/abs/2605.11368>
- STRIDE：<https://arxiv.org/abs/2603.03573>
- UTRGen：<https://www.biorxiv.org/content/10.64898/2026.06.26.734691v1>
- NucleoBench：<https://www.biorxiv.org/content/10.1101/2025.06.20.660785v3>；<https://github.com/move37-labs/nucleobench>
- NABench：<https://arxiv.org/abs/2511.02888>
- mRNABench：<https://pubmed.ncbi.nlm.nih.gov/40672173/>
- PARADE：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11722239/>
- UTailoR：<https://pubmed.ncbi.nlm.nih.gov/41069846/>
- UTRGAN：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12228966/>
- mRNABERT：<https://www.nature.com/articles/s41467-025-65340-8>
- UTR-LM：<https://www.nature.com/articles/s42256-024-00823-9>

## 17.2 高优先级公开数据

- GSE217518：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217518>
- GSE232571/572：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232571>；<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232572>
- GSE256185：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE256185>
- GSE232927：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232927>
- GSE194092：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194092>
- GSE176581：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176581>
- GSE288185：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288185>
- fast-UTR Siegel：<https://github.com/david-a-siegel/AU-Rich-Elements>；<https://pmc.ncbi.nlm.nih.gov/articles/PMC8728028/>
- ENCSR854RUF：<https://www.encodeproject.org/experiments/ENCSR854RUF/>
- GSE200304 SuperSeries（含 GSE200302/GSE200303/GSE217530）：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200304>
- cryptic-splicing复核：<https://www.nature.com/articles/s41467-025-62000-9>
- GSE270252：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270252>
- GSE173098：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173098>
- GSE330741：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330741>
- GSE261709：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261709>
- GSE295080：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295080>
- GSE291719：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291719>
- GSE298114：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE298114>
- N-zip E-MTAB-10902/11572/11575：<https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10902>；<https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11572>；<https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11575>
- GSE55396 / Zhao fast-UTR：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55396>
- PTRE-seq：<https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1116243>
- PASSPORT-seq：<https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2018.00219/full>

---

# 18. v3.1 Decision Log

```yaml
- decision_id: V3-001-BENCHMARK-FIRST
  date: 2026-08-03
  decision: benchmark/resource is primary; method is conditional co-contribution
  approved_by_user: true
  requires_rerun: [D0, D1, B0]

- decision_id: V3-002-NO-WET-LAB
  date: 2026-08-03
  decision: no new wet-lab evidence; retrospective computational claims only
  approved_by_user: true

- decision_id: V3-003-REGION-ASYMMETRY
  date: 2026-08-03
  decision: 5UTR primary; 3UTR effect/reconstruction/transfer secondary
  approved_by_user: true

- decision_id: V3-004-LAYERED-DATA
  date: 2026-08-03
  decision: retain Track U/F/E separately; no single N_total
  approved_by_user: true
  status: SUPERSEDED
  superseded_by: V3-012-EF-ONLY-NO-UNLABELED-PRETRAINING
  requires_rerun: [D0, D1, B0]

- decision_id: V3-005-ARTIFACT-RISK
  date: 2026-08-03
  decision: low-risk primary; high-risk sensitivity or auxiliary training
  approved_by_user: true
  requires_rerun: [D1, B0]

- decision_id: V3-006-REOPEN-DATA-GATES
  date: 2026-08-03
  decision: reopen D0/D1/B0; preserve MK0/EF0 engineering evidence; block formal GP0
  approved_by_user: true

- decision_id: V3-007-LATENT-PATH
  date: 2026-08-03
  decision: path-ensemble/marginalization is the primary method hypothesis
  approved_by_user: true
  status: TECHNICAL_ROUTE_RETAINED_CLAIM_SEMANTICS_SUPERSEDED
  superseded_by: V3-010-EDIT-FLOWS-DERIVATION

- decision_id: V3-008-THREE-ROUND-SEALED-FINAL
  date: 2026-08-03
  decision: three preregistered development rounds; final labels opened once
  approved_by_user: true

- decision_id: V3-009-VERSIONED-BENCHMARK
  date: 2026-08-03
  decision: literature/data discovery snapshot 2026-08-03; future additions enter v1.1
  approved_by_user: true

- decision_id: V3-010-EDIT-FLOWS-DERIVATION
  date: 2026-08-03
  decision: >
    The CTMC edit framework, INS/DEL/SUB actions, auxiliary alignment process,
    auxiliary-process marginalization and Bregman Edit Flow objective are inherited
    from Edit Flows. The technical route is retained as a source-conditioned UTR
    adaptation and alignment-proposal robustness study; no originality claim is made
    for the inherited core.
  technical_route_retained: true
  approved_by_user: true
  requires_rerun: [C3, MK0-R, EF0-R, GP0-DEV, CLAIM_MATRIX]

- decision_id: V3-011-GSE246381-PROJECT-UNEXPOSED
  date: 2026-08-03
  decision: >
    GSE246381 is absolutely unexposed on the project side. Project exposure and
    external foundation-model overlap are separate axes. Its default v1 use role is
    sealed external final candidate, not historically exposed stress data.
  supersedes: [OLD_GSE246381_HISTORICALLY_EXPOSED_TRUTH_LOCK]
  approved_by_user: true
  requires_rerun: [D1-R, FM0-A, B0-R, GP0_PREFLIGHT, FINAL_POLICY]

- decision_id: V3-012-EF-ONLY-NO-UNLABELED-PRETRAINING
  date: 2026-08-03
  decision: >
    Benchmark v1 retains Track E as primary and Track F as supervised auxiliary.
    Track U and project-side unlabeled pretraining are out of scope. Unlabeled
    references may only support coordinate/reference annotation and never count as data.
  supersedes: V3-004-LAYERED-DATA
  approved_by_user: true
  requires_rerun: [C3, D0-R, D1-R, B0-R, FM0-A, DATA_CARD, PAPER_SCOPE]

- decision_id: V3-013-PROPOSED-NEXT-GOAL-BOUNDARY
  date: 2026-08-03
  decision: >
    The proposed next execution scope, once separately activated by an explicit user
    instruction, ends after C3, D0-R, D1-R, FM0-A, B0-R and G7. Its success state is
    DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND. It does not unlock GP0; FM0-B,
    MK0/EF0-R and a GP0 preflight remain required under a separate goal. This document
    does not itself authorize remote writes, downloads, commits, training or final access.
  approved_by_user: true
  activation_status: NOT_ACTIVATED
  execution_authorization: NOT_GRANTED_BY_THIS_DOCUMENT

- decision_id: V3-014-TASK-ESTIMAND-SEPARATION
  date: 2026-08-03
  decision: >
    Endpoint reconstruction/distribution and functional optimization are separate tasks.
    T5-Gen-Reconstruct compares same-target editors; T5-Rank/Closed-Select compares
    scorers and search on a frozen measured pool. Open generation remains diagnostic.
  reason: the primary Edit Flow has no desired-outcome/reward input or outcome head
  approved_by_user: true
  requires_rerun: [PR1, GP0-DEV, FC0, MB0, CLAIM_MATRIX]

- decision_id: V3-015-ORTHOGONAL-DATA-ROLES-AND-EARLY-SEAL
  date: 2026-08-03
  decision: >
    Scientific track, relation/effect evidence, dense/no-edit overlay, future use role,
    and license eligibility are orthogonal fields. GSE246381 enters a restricted sealed
    store from D1, while ordinary canonical receives aggregate commitments only.
  approved_by_user: true
  requires_rerun: [C3, D1-R, FM0-A, B0-R]

- decision_id: V3-016-METHOD-ATTRIBUTION-AND-ESTIMATOR
  date: 2026-08-03
  decision: >
    Separate upstream reproduction, basic pair Edit Flow, UTR adaptation, alignment
    proposal and multi-sample estimator axes. Multi-Q uses with-replacement iid draws
    with multiplicity and is judged by gradient variance and quality-compute Pareto.
  approved_by_user: true
  requires_rerun: [MK0-R, EF0-R, GP0-DEV, MB0, CLAIM_MATRIX]
```

---

# 19. 本合同的完成标准

本合同的目标不是承诺 Edit Flow 一定获胜，而是建立一个不能靠数据混淆、泄漏、暴露重写、指标挑选或 final 反复访问制造成功的研究系统。

项目完成的最低科学标准是：

1. 公开数据资产按真实语义和独立单位完成可审计分账；
2. v1 benchmark 的数据、split、exposure、artifact、任务和 final 封存闭环；
3. Edit-Flows-derived UTR reference model、`BASIC_PAIR_EF_NATIVE`、UTR adaptation/alignment-estimator variants 与同任务 editors 在 T5-Gen-Reconstruct 上比较；scorer/ranking/search 在独立 closed measured-pool 上比较；
4. 结果无论正负均按预注册规则报告；
5. 所有主张严格限于回顾性计算证据；
6. artifacts、provenance、checksums、失败证据与版本历史足以复现和审计。

在上述条件满足前，不得声称：

```text
DATA_COMPLETE
BENCHMARK_READY
GP0_SCIENTIFIC_PASS
METHOD_SUPERIOR
PUBLICATION_READY
BIOLOGICAL_IMPROVEMENT
```

在条件满足后，允许的最高当前结论仍是：

> **一个无新增湿实验、基于 Track E/Track F 公开历史功能测量、来源与暴露可审计的 source-conditioned 5′UTR editing benchmark，附带通过资格门槛的 3′UTR 扩展，以及一个明确继承自 Edit Flows、在预注册 matched-budget final 中接受检验的 UTR reference model。**

若方法没有胜出，最高结论自动降级为：

> **一个来源、暴露、实验语义和独立样本单位可审计的 source-conditioned UTR editing benchmark，以及一组包含 Edit-Flows-derived model 在内的强参考基线和负结果分析。**
