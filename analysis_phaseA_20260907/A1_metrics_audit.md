# A1 — Metrics 口径审计 (SetFlow V5 B2/B3 adjudication)

日期 2026-09-07 · 离线 CPU · protected reads = 0（本审计只读 VALIDATION DEVELOPMENT 结构信息与生成的候选池，未触碰 TEST/EVALUATION outcome 标签）

范围：审计 B2 裁决脚本 `scripts/route_a_v3/adjudicate_route2_guided_setflow_v5_b2_v1.py` → 其指标定义委托给 `scripts/route_a_v3/evaluate_route2_generation_v1.py` 的 `measured_neighborhood_metrics` / `evaluate_generation`；核对产物 `b2_full_891_adjudication.json`、`b2_full_891_adjudication_per_task.json`、TreeG `result_full891.json` / `result_k2.json`。

---

## 0. 通用口径（三个 metric 的公共基础）

- **序列归一化**：任何候选/measured 序列先 `.upper().replace("T","U")` 后比较 → 判定“恢复”为**精确全序列字符串匹配**（非 startswith、非 kind、非 edit 距离）。来源：`_measured_pool_by_source`、`measured_neighborhood_metrics` 内 `.upper().replace("T","U")`。
- **生成池去重**：每源内按 `candidate_sequence` 去重，同一序列保留 `generation_score` 最大者（`if previous is None or score > previous[1]`）。
- **排序分数来源**：`score_field = "critic_score" if row.get("critic_score") is not None else "generation_score"`。实际两份候选 JSONL（unguided & guided）的行里**都没有 `critic_score` 字段**（guided 只有 `terminal_potential`），故 unguided 与 guided 两臂的排名都用 `generation_score`。
- **宏口径**：所有表级指标均为 **source-macro = 逐源指标的未加权平均（mean over 891 sources）**，非候选级加权聚合。

---

## 1. recovery@budget（= 代码字段 `candidate_recovery_rate`，即 Gate B3 的“guided recovery”）

- **分子**：每源生成池（去重后）中属于该源真实 measured 候选集合的 `recovered_candidate_count`（`hits = [seq for seq in generated if seq in pool]`）。
- **分母**：该源 **measured 候选数** `measured_candidate_count`（= `len(measured[source_key])`，来自 DEVELOPMENT measured_neighborhood），**不是** budget=32。
- 公式：`recovery_source = hits / measured_candidate_count`；`recovery_macro = mean over sources`。
- **不依赖排序、不依赖 k**：`candidate_recovery_rate` 统计候选出现在 32 池“任意位置”，与 critic 排名无关，`k` 只影响下面 top-K 指标。→ **ρ（critic 判别力）不影响本指标**（A4 关键结论）。
- budget=32 与本指标的关系：32 是生成池容量（`candidate_budget`）。本指标衡量“生成的 32 池对真实 measured 集合的覆盖比例”，并非“预算利用率”。
- 逐源还是全池：逐源（per-source）后宏观平均。
- 恢复判定：精确匹配（见 §0）。
- 产物实测：unguided=0.120464，guided=0.126263；per_task 见 §4。

## 2. measured top-K recovery（= 代码字段 `measured_top_k_recovery_at_k`，k=MEASURED_TOP_K=10）

- **语义**：真实 outcome 值（`measured_direction_normalized_delta`）排前 `top_k` 的 measured 候选，有多大比例出现在生成池按分数排前 `top_k` 的位置里。
- `top_k = min(k, len(pool))`（pool=该源 measured 集合），即**多数源 measured 数<10 时 top_k 收缩为 measured 数**。
- `true_top_eligible`：真实值 ≥ 第 `top_k` 名 cutoff 的 measured 候选（含并列）。
- `generated_inclusion = _top_k_inclusion_probabilities(generated_scores, top_k)`：按降序并列块平分剩余名额的准入概率（tie-aware）。
- 公式：`measured_top_k_recovery = Σ( inclusion_prob(seq) for seq∈generated if seq∈true_top_eligible ) / top_k`。
- 产物实测：unguided=0.063671，guided=0.056981（guided 反而更低）。

## 3. hit@1（= `measured_top_k_recovery_at_k` 取 k=1；产物字段 `hit_at_1`）

- **语义**：tie-aware 的“生成池 top-1 候选是真实 outcome 最高（真最好）的 measured 候选”的概率。
- `top_k=min(1,len(pool))=1`；`true_top_eligible`=真实值==最高值的 measured 候选（并列）；`_top_k_inclusion_probabilities(scores,1)`：生成分数最高并列块内平分 `1/块大小`。
- 公式：`hit_at_1 = Σ over 生成top并列块中属于 true_top_eligible 的 inclusion_prob / 1`。
- 产物实测：unguided=0.036700，guided=0.034231。
- 注意：hit@1 要求预测“**最好那一个**”，比“池里出现任意 measured”（support）难得多。

## 4. unique candidate rate / legality（`evaluate_generation` 输出）

- **legality**：单候选合法 ⇔ 非空 ∧ 字符集⊆{A,C,G,U} ∧ 与 source 等长 ∧ `edit_distance ≤ edit_budget`。
  - `hard_legality_rate = total_legal / total_rows`（全池）；逐源 `legality_rate = legal_candidate_count/rows`。
- **unique_candidate_rate**：`len(unique_legal_sequences)/len(rows)`（逐源，source-macro 平均）。
- **budget 违规**：`candidate_budget_violation = max(0, len(rows)-candidate_budget)`；`edit_budget_violation = count(edit_distance>edit_budget)`。
- 产物实测：两臂 `legality_ok=true`（hard_legality_rate=1.0、两次违规计数=0）。

---

## 5. 产物字段核对表

| 字段（b2_full_891_adjudication.json） | 代码定义 | 产物值 | 核对 |
|---|---|---|---|
| unguided.source_macro_candidate_recovery_rate | §1 | 0.120464 | ✓ |
| guided.source_macro_candidate_recovery_rate | §1 | 0.126263 | ✓ |
| unguided/guided.source_macro_measured_top_k_recovery_at_k | §2 | 0.063671 / 0.056981 | ✓ |
| unguided/guided.hit_at_1 | §3 | 0.036700 / 0.034231 | ✓ |
| gate_b2_passed / gate_b3_passed | 规则见下 | false / false | ✓ |
| per_source[*].{unguided,guided}.{candidate_recovery_rate,measured_top_k_recovery_at_k,hit_at_1,top1_generated_is_measured} | §1–3 | 逐源 891 个 | ✓ |
| bootstrap（2000 iter, seed 20260816, 270 source-groups, SOURCE_GROUP_PAIRED_CLUSTER） | 逐源配对集群 | ✓ | ✓ |
| legality_ok / per-arm 违规 | §4 | true / 0 | ✓ |
| evaluation_outcome_reads / new_final_evaluation_outcome_reads | 纪律计数 | 0 / 0 | ✓（protected reads=0 兑现） |

**per-task（b2_full_891_adjudication_per_task.json，endpoint_id 分桶）**：MRL(652) 0.15465→0.15874、MPRAU(108) 0.02778→0.03241、HL(111) 0.03153→0.04955、polyA(20) 0→0 — 与 A2 口径一致。region 字段从 source_eligibility 冗余读出（仅 MRL/HL 为 5UTR，MPRAU/polyA 为 3UTR）。

---

## 6. 口径不一致 / 歧义发现（A2/A3 必须知悉）

1. **`metric_definitions.recovery` 标签“at k=10”是误导**：`candidate_recovery_rate` 完全不依赖 k（只看 32 池内去重命中），`k=10` 只作用于 `measured_top_k_recovery`。这是产物注释与代码语义不符的一处标签错误，不影响数值。
2. **recovery@budget 的分母是“该源 measured 数”而非 32**。对 measured 数仅 2–4 的源，recovery 只取 {0,1/m,2/m,…,1}，导致小采样容错低、且“恢复比例”严格被 measured 数小量化——任何单源都无法出现连续值。这放大了对偶性波动，应在 A3 的配对 delta 解释中注意。
3. **数据画像与实测不符（重要）**：本 891 源 VALIDATION 实际 measured 数分布为 {2:317, 3:481, 4:93}，平均 **2.75/源**（MRL≈3.0，MPRAU/HL/polyA 各 2.0）。任务背景中“平均 6.1 measured 候选/源、33% 源≤3、TreeG 池=256 候选/源”在**验证集上不成立**（该画像疑似来自训练集 14,634 源；且 256 池产物不存在，见 §7）。
4. **`top1_generated_is_measured` 仅是诊断字段（不 gated）**：它统计“top 生成块中属于**任意 measured** 的比例”，≠ hit@1。不要把二者混为一谈（前者是 support@top，后者要求命中最优 measured）。
5. **hit@1 的 top-1 并列块分母不稳定**：当多候选并列最高生成分数时，inclusion 概率被平分，hit@1 成为期望而非确定性，比较小样本时应留意。

---

## 7. TreeG 产物审计（A2 相关）

`treeg_sample_select_20260905/{full,k2}/result_*.json` 说明：
- TreeG 是**在 unguided 的 32 候选/源池（pool_candidate_count=28512）上的 selection-only**，用 FrozenXEditCriticV5 重打分后按 `--select-k` 取 top-K（full891: k=1；k2: k=2）。
- **不存在“256 候选/源”的 TreeG 池**（脚本注释明确标记 N>32 的样本-再选为“注册的后续工作、本未运行”）。
- `result_full891.json`：select_k=1，candidate_count=891；TreeG macro recovery=0.002245 hit@1=0.004489（vs unguided recovery 0.1205 hit@1 0.0367）→ 因 k=1 样本少导致 shot-count 塌缩；`delta_recovery_vs_unguided` 显著为负（-0.118，CI 不含 0）。`result_k2.json`：recovery=0.00524。
- 结论：TreeG 以当前 crit 判力在 k1 的 selection 质量（hit@1 0.0045）远低于 unguided 的 32 池覆盖，critic 判别力是瓶颈之一，但**不是唯一/主瓶颈**（见 A2/A4：池覆盖 support(32)=24% 才更致命）。