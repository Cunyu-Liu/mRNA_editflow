# mRNA-XEditFlow 项目科研与执行合同 v1.1

## Exact-Guided, Mechanism-Constrained Edit Flows for Source-Anchored mRNA Redesign

> - **文档性质：**科学问题合同、数据资源合同、benchmark 合同、模型与评测合同、GPU 执行合同、论文主张边界
> - **权威合同 ID：**`mrna_xeditflow_goal_v1_1`
> - **文档状态：**`ACTIVE_AUTHORITATIVE_CONTRACT`
> - **制定日期：**2026-08-06
> - **目标仓库：**`/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
> - **迁移基座：**`utr_editflow_goal_v3.1_benchmark_first`（SHA256 `ecc6c635…`），现标记为 `HISTORICAL_SUPERSEDED_BY_MRNA_XEDITFLOW_V1_1`
> - **迁移提示词：**`提示词/mrna 合同迁移.md`（SHA256 `55c89e5…`）
> - **当前范围：**5′UTR 为主区域；3′UTR 为 transfer track；CDS 仅允许原子级 synonymous-codon substitution 作为 transfer
> - **湿实验边界：**当前没有新增湿实验资源；所有结论限于公开历史 assay 的回顾性计算证据
> - **数据治理基座：**迁移只 supersede 顶层科学主线；**不** supersede 旧合同已经建立的 provenance、license、exposure、sealed-final、split、conservation/守恒 与 audit/审计 标准（以下全篇继承）。

---

# 0. 文档权威性、术语与变更控制

## 0.1 本合同的唯一中心科学问题

> 在 source、assay、context、endpoint 和有限编辑预算明确的条件下，能否学习可跨 source/study 迁移的局部 mRNA 编辑效应，并将该效应作为风险校准密度比精确注入生物合法 CTMC，使生成/搜索过程在相同候选、调用和时间预算下优于 no-guidance、first-order guidance、latent/rate CFG 和 generate-then-rerank？

本项目必须同时回答两类独立问题，且不得相互替代：

1. **数据/预测问题：**source-relative intervention learning 是否跨 source/study 有效？
2. **生成/推断问题：**exact guidance 是否在匹配预算下优于强 search/reranking？

## 0.2 权威顺序

1. 本合同及其 SHA256；
2. `configs/mrna_xeditflow_contract_v1_1.yaml`、decision log 与 claim matrix；
3. xedit_v1_1 JSON Schemas 与 dataset-specific adapter contracts；
4. 本合同冻结的 `TaskRegistry v4` / `SplitRegistry v4` / task×split applicability matrix；
5. 旧合同（v3.1）继承的 immutable raw/acquisition/rights manifests、exposure ledger、sealed commitments、supersession 记录；
6. 旧合同继承的 canonical base tables、observation/relation lifecycle、transformation edges；
7. 单次 run 的 frozen config、代码 commit、status、output manifest、checksums、finalizer 与 DONE；
8. 论文表格、图和文字。

低层与高层冲突时 `FAIL_CLOSED`。派生层不得反压或改写权威上游。

## 0.3 强制术语
沿用旧合同 v3.1 术语语义（`PASS`/`REOPENED`/`BLOCKED`/`DEVELOPMENT_ONLY`/`FORMAL`/`INDEPENDENT_PARENT`/`CONTRACT_VALID`/`LATENT_PATH`/`PIPELINE_MATERIALIZATION`/`ANALYTIC_LABEL_ACCESS`/`ANALYTIC_FINAL_LABELS_ACCESSED`），并新增：

- `EFFECT_PRIMARY`：method_training_role，表示该数据用于 source-relative effect critic 主训练。
- `FLOW_BASE`：method_training_role，表示该数据用于 base legal Edit Flow 训练。
- `CRITIC_AUX`：method_training_role，表示该数据用于 SparseEditFormer 辅助任务。
- `EXACT`：仅表示在明确 learned density-ratio/path 与假设下的数学 exactness，不等于真实生物最优。
- `OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL`：GSE246381 的 claim 措辞；除非有时间证据，不称 temporal/prospective。

---

# 1. 冻结决策（迁移前锁定，见 §2.6 决策日志）

| # | 维度 | 冻结值 |
|---|---|---|
| A | Termination | `primary_termination_policy: FIXED_BUDGET_OR_FIXED_HORIZON`；`learned_general_stop: HOLD_IDENTIFIABILITY_GATE`；`INITIAL_NO_EDIT_AUX` 仅作为非 sampler auxiliary，sampler call count=0 |
| B | 无标签预训练 | `project_unlabeled_pretraining: DISABLED_IN_PRIMARY_V1`；`legal_corruption_adaptation: OPTIONAL_AMENDMENT_AFTER_DATA_AND_OVERLAP_GATE`；不静默恢复 Track U |
| C | GSE246381 | `project_prior_analytic_use: NONE_CONFIRMED`；`pipeline_materialization: PRESENT`；`role: SEALED_EXTERNAL_FINAL_CANDIDATE`；claim 措辞 `OPERATIONALLY_SEALED_RETROSPECTIVE_EXTERNAL` |
| D | 编辑预算 | `primary_edit_budget: [1, 3, 5]`；`exploratory_edit_budget: [10]` |
| E | Indel | primary 5′UTR 仅 substitution；INS/DEL heads 物理禁用；真实 length-change 与 latent alignment indel 继续分账 |

---

# 2. 论文定位与五项主创新

## 2.1 论文主线
```text
mRNA-EditBench
→ SparseEditFormer (source-relative effect critic)
→ measured-space search ceiling
→ source-anchored legal Edit Flow
→ exact density-ratio guidance on a legal edit graph
→ matched-compute measured-neighborhood optimization
→ sealed/OOD evaluation
→ 3′UTR/CDS transfer
```

## 2.2 主创新（C1–C5）
- **C1** Intervention-first mRNA benchmark（mRNA-EditBench：EditBench-5U-A1-Natural / 5U-A2-Dense / 3U-A1-Variant / CDS-B1-Synonymous）。
- **C2** Heterogeneous legal biological edit graph（5′UTR SUB graph、CDS synonymous-codon graph、3′UTR SUB graph、GLOBAL STOP）。
- **C3** Exact density-ratio guidance on a non-Cartesian legal edit CTMC（DGM 推广，理论工作包）。
- **C4** Frozen, calibrated intervention critic（SparseEditFormer），Frozen 后用于 guidance；brace 不拼接多个 foundation。
- **C5** Closed-world measured optimization 作为 primary generative benchmark。

## 2.3 明确不是创新的内容
沿用旧合同边界并扩展：不宣称“首次 Transformer 用于 mRNA”、不拼接多个 foundation embedding、不用 RNAfold/GC/codon 作为 reward、不把 hard mask 本身当创新、不因随机 split 成功而宣称泛化、不对未测序列宣称真实功能改善。

---

# 3. 数据治理基座（继承旧合同，不 supersede）

以下治理机制**按 hash 承接**自旧合同 v3.1，作为本合同的规范附件，不得因论文主线变化而弱化：

- provenance 与 license（raw/paper_clean/canonical 三层、file ID/version/byte size/SHA256）；
- Track E / Track F / AUX / REFERENCE 语义区分；
- raw unit、relation candidate、observation join、accepted pair/F observation 独立守恒；
- source/candidate、replicate、barcode、context、endpoint、gene、family、sequence cluster group registry；
- license / training/evaluation/release 权限 fail-closed projection；
- 项目 exposure 与 foundation overlap 分轴；
- supersession ledger、current-leaf projection、immutable historical artifact；
- GSE246381 restricted dual-store、append-only access intent/completion/abort、一次性 final；
- exact/reverse/near/source/gene/study/library/context leakage audit；
- Data Card、claim-evidence matrix、all-trials registry、G0/G7 manifest 与 hash closure。

## 3.1 新增正交轴（不替代 E/F）
```yaml
scientific_track: E | F | AUX | REFERENCE
intervention_evidence_grade: A1 | A2 | B1 | B2 | C | D
method_training_role: EFFECT_PRIMARY | FLOW_BASE | CRITIC_AUX | TRANSFER | DIAGNOSTIC | EXCLUDED
endpoint_role: DELTA | ABSOLUTE_PROPERTY | FAMILY_RANK | RECONSTRUCTION | NOT_APPLICABLE
critic_eligibility: YES | NO
flow_base_eligibility: YES | NO
guidance_training_eligibility: YES | NO
measured_optimization_eligibility: YES | NO
transfer_eligibility: YES | NO
```

映射：intentionally assayed pair→A1；controlled dense measured landscape→A2；same-protein synonymous family→B1；modular/full-length construct family→B2；source-unresolved function observation→C；unlabeled/reference natural transcript→D/REFERENCE（默认不训练）。

---

# 4. 任务与评测（TaskRegistry v4 摘要）

## 4.1 Primary tasks
1. `T5_SOURCE_RELATIVE_EFFECT`
2. `T5_SELECTIVE_EFFECT`
3. `T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION`
4. `T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION`

## 4.2 Secondary tasks
5. `T5_ENDPOINT_RECONSTRUCTION`
6. `T3_EFFECT_TRANSFER`
7. `T3_RECONSTRUCTION`
8. `CDS_SYNONYMOUS_FAMILY_RANKING`
9. `CROSS_REGION_TRANSFER`
10. `OPEN_WORLD_PROXY_DIAGNOSTIC`

## 4.3 Theory tasks
11. `EXACT_GUIDANCE_TOY_GRAPH`
12. `EXACT_GUIDANCE_MATCHED_COMPUTE`

## 4.4 Effect gate（primary）
```text
macro Δ Spearman ≥ 0.25
macro sign accuracy ≥ 0.60
top-10% enrichment ≥ 1.50
SparseEditFormer > strongest executable non-foundation baseline
```

## 4.5 Exact-guidance gate
```text
NDCG ≥ strongest matched baseline + 0.05
top-10 recall ≥ 0.70
normalized regret ≤ 0.10
hard legality = 100%
sealed/OOD not degraded beyond preregistered margin
```

---

# 5. 模型架构（摘要）
- 模块 A：source-cached region encoder（source 编码一次并缓存）。
- 模块 B：生物坐标系统与 hard invariants（illegal rate=0 在归一化前）。
- 模块 C：region-specific structured coupling（5′UTR/3′UTR nucleotide、CDS synonymous-codon）。
- 模块 D：mechanism-aware action encoder。
- 模块 E：base legal Edit Flow（`uθ=softplus(zθ)`；primary substitution-only；INS/DEL 物理禁用）。
- 模块 F：SparseEditFormer 冻结 critic（μΔ/σΔ/P(beneficial)/P(sign)/ranking/OOD）。
- 模块 G：目标分布与 exact edit guidance（`q_{1,β} ∝ p1 exp(β R_robust)`；guidance ratio head gψ）。
- 模块 H：CTMC sampler、预算与 termination（FIXED_BUDGET_OR_FIXED_HORIZON）。
- 模块 I：独立评估器（measured label / independent evaluator / diagnostic predictors）。

---

# 6. 论文 fallback
- 数据+critic+exact guidance 全通过 → mRNA-XEditFlow 主论文。
- 数据+critic 通过但 Flow/guidance 不通过 → mRNA-EditBench + SparseEditFormer 主论文。
- critic 只在随机 split 有效 → 停止生成主线，回到数据、pair、endpoint 与 leakage 修复。

## 6.5 论文主张边界（Claim ladder，L0–L4）
- **L0 数据与复现：**允许声称构建了可复现、证据分层、泄漏受控的公开 mRNA 干预 benchmark（mRNA-EditBench）。前提：raw provenance、checksum、reproduction report、rejected table、license 与 split audit 完整。
- **L1 效应预测：**允许声称在未见 source/gene/study 上更准确预测 source-relative measured delta，并提供校准不确定性。
- **L2 已测候选空间优化：**允许声称在实际测量的候选集合中更有效地排序或发现高收益局部编辑。这是第一篇论文最重要、最稳健的生成/优化 claim。
- **L3 未测候选的代理优化：**最多声称在冻结的独立预测器与预注册生物约束下生成了更高代理分数的合法候选；必须同时报告 OOD、uncertainty、diversity 与 independent evaluator，不得写成真实功能改善。
- **L4 真实生物/治疗改善：**本版本 **prohibited**（无新增湿实验）；不得声称设计了实验验证的更高蛋白产量或更优治疗性 mRNA。

---

# 7. 不可越过的执行边界（沿用旧合同并扩展）
1. 不把 observational data 当 intervention labels。
2. 不把随机 UTR library 称为 natural WT-mutant pairs。
3. 不把 MRL/TE/half-life 写成 measured protein output。
4. 不跨 endpoint 直接混合标签。
5. 不随机拆分 source/candidate、dense graph edge 或 synonymous family。
6. 不允许 reverse edge 跨 split。
7. 不用 sealed labels 调参、校准或选择 checkpoint。
8. 不在 paper mode 使用 placeholder backbone/evaluator。
9. 不把 5′UTR、3′UTR、CDS 简单拼接为单一无条件任务。
10. 不把 same-protein family 展开后当作独立样本做虚假显著性。
11. 不只报告 pooled metric、最佳 seed 或最佳 guidance scale。
12. 不因为 Flow 无增益而删除 direct scorer/search 结果。
13. 不因为外部测试失败而修改 test set 或阈值。
14. 不以 predictor 自评分证明真实生物改善。
15. 不把 hard legality 作为 reward penalty 替代 action-space guarantee。
16. 不声称 DGM 公式对 Edit Flows 异构编辑图自动成立。
17. 不声称 FlexFlow generic substitution matrix 已捕获 mRNA 机制。
18. 不在 primary task 中启用没有可验证数据支持的 indel。
19. 不在没有显著 headroom 时扩展 full-length joint optimization。
20. 不声称无实验支持的 therapeutic/manufacturing/clinical improvement。

---

# 8. 终态
允许终态：`MIGRATION_READY_FOR_DATA_REBUILD` / `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL` / `EFFECT_MODEL_GO` / `EFFECT_MODEL_NO_GO` / `FLOW_GUIDANCE_GO` / `FLOW_GUIDANCE_NO_GO_FALLBACK_TO_BENCHMARK_CRITIC` / `SEALED_FINAL_COMPLETE` / `BLOCKED_WITH_EVIDENCE`。

---

# 9. 合同变更与决策日志
所有合同冲突必须写入 `docs/execution/xeditflow_migration_decision_log.yaml`。迁移前已锁定五项决策（§1 A–E）。不得由 parser、trainer 或论文作者自行选择方便解释。