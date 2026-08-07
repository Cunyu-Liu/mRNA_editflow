# mRNA-XEditFlow V2 合同交接文档：现状 / 差距 / 卡点 / 修复路径

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **交接日期（UTC）：** 2026-08-07
- **目标合同：** `mrna 最新合同-v2.md`（= `mrna_xeditflow_goal_v1_1`，权威新合同）
- **当前终态：** `BLOCKED_WITH_EVIDENCE`
- **工作树分支：** `xeditflow-migration-20260806T024650Z`
- **测试基线：** `tests/migration/` 全量 **224/224 PASS**（editflow env）

> 本交接文档回答三个问题：(1) 到 V2 合同正式执行的差距；(2) 不消费 sealed final 时如何修复 sign_accuracy；(3) 走合同修订时如何不浪费一次性访问并保证 GO。所有结论均基于已执行的真实实验与审计记录，无数据造假。

---

## 一、到 V2 合同"正式执行"的距离

### 关键概念澄清：V2 合同任务"已经跑过一遍"，但要区分两层

| 层次 | 状态 | 说明 |
|---|---|---|
| **开发级执行（development）** | ✅ 基本完成 | 合同 §16 的模型主线阶段（B0→M0→O0→F0→G0→G1→E0→X0）在迁移期内已作为开发阶段跑通并验收。 |
| **正式/最终执行（formal / sealed / paper）** | ⛔ 未触发 | 正式 GO/HOLD/NO-GO 判定、sealed test、跨区域 transfer 正式 claim、P0 论文——均被卡点阻塞，不能自动开始。 |

### 阶段进度对照表（合同 §16）

| 阶段 | 合同目标 | 当前状态 | 是否解锁正式执行 |
|---|---|---|---|
| R0 合同/仓库治理 | 权威合同、crosswalk、registry | ✅ 完成（M0-M3 PASS） | 是 |
| D0 数据发现/获取 | 数据源接入 | ✅ 数据 gate `DATA_BENCHMARK_READY_FOR_EFFECT_MODEL` | 是 |
| D1 逐论文复现/清洗 | 数据集清洗 | ✅ 3U-A1 候选池重建 7/7 | 是 |
| D2 mRNA-EditBench 构建/冻结 | benchmark 冻结 | ✅ 3U-A1/5U-A1 ACTIVE；CDS-B1 例外（见卡点 B2） | 部分 |
| B0 全基线 | baseline ceiling | ✅ `B0X_EFFECT_BASELINE_CEILING_ESTABLISHED` | 是 |
| M0 SparseEditFormer | effect model | ✅ M4 已训练（A2→A1→校准→冻结） | 是（已冻结 5′ critic） |
| O0 closed-space 优化 | search ceiling | ✅ `O0X_SEARCH_CEILING_ESTABLISHED` | 是 |
| F0 legal Edit Flow | substitution-only flow | ✅ legality/length 100% | 是 |
| G0 exact guidance 理论 | toy 精确验证 | ✅ rel err 1.77e-16 | 是 |
| G1 mRNA guidance 集成 | real-mRNA | ✅ generation-quality 轴有增益 | 是 |
| E0 完整实验/统计/sealed | effect gate + sealed | ⚠️ internal NO_GO（sign_accuracy）；sealed 未执行 | **否（卡点 B1）** |
| X0 3′UTR/CDS transfer | 跨区域迁移 | ⚠️ 开发底座就绪（3′ adapter 已训 3U-A1）；formal gate 未触发 | **否（卡点 B1/B2）** |
| P0 论文/发布 | manuscript | ❌ 未开始（依赖前置判定） | 否 |

### 一句话差距结论
**模型与开发链路全部就绪，正式执行被"最终判定层"阻塞**：E0-X effect gate 的 sign_accuracy 未过（NO_GO），导致 E0-X 正式判定、X0-X 正式 transfer、P0 论文均无法解锁。这不是"还有多少代码没写"的问题，而是"科学判定被一个真实指标卡住 + 两个数据/决策卡点"。

---

## 二、不消费 sealed final 时，如何修复 sign_accuracy

### 2.1 已证明"此路不通"的路径（别再走）
用冻结 critic + 同一 S4 folds 做了选择性预测（T5 abstain+coverage）验证，两种诚实置信度源**都无判别力**：

| 置信度源 | Spearman(置信度, 符号正确性) | selective sign_acc frontier |
|---|---|---|
| logvar（预测方差） | 0.0155 / 0.0038 | 全部 coverage 封顶 ~0.50–0.51 |
| \|pred\|（预测幅度） | −0.021 / −0.010 | 全部 coverage 封顶 ~0.50–0.51 |

**结论**：冻结 critic 学不到任何"符号是否正确"的置信度信号，选择性预测无法把 sign_accuracy 合法拉到 0.60。**不要再在现有冻结模型上做 selective 工程**。

### 2.2 真正能合法改善 sign_accuracy 的方向（需"动模型/数据"，不是动阈值）
根因是**从序列模型在 S4 跨研究 transfer 下、对"source-relative 符号方向"学不到可靠信号**（类别先验封顶 ~0.52）。要突破，只能从模型/数据侧获得更强的符号信号，这些是**未来可做、需重新训练**的工作：

1. **更强的源锚定/上下文**：把 source 的实测上下文特征（不只是序列）更充分注入，让模型学到"相对于 source 的方向"而非全局先验。
2. **端到端异构方差重训 + 符号专用目标**：此前重训符号头对准 `sign(delta)` 未改善；可尝试**多任务联合训练**（mean+sign+rank 联合 loss），让符号监督与回归监督相互正则。
3. **更多/更干净的高信噪比数据**：若新数据（如 3U-A1 或其它 A2 密集库）提供更清晰的效应标签，可重新训练。
4. **校准到决策边界**：若模型输出的 mean 有部分排序能力（spearman 0.297 显著），可尝试"符号=相对 source 的 rank 差"的决策规则，而非绝对 0 阈值。

> 注意：以上都需要**重新训练模型并重新走 pre-unblinding 冻结**，不修改既有冻结 5′ critic 是不行的（sign_accuracy 是被冻结模型算出来的）。这属于"启动新一轮 M0 训练"，不是补丁。

### 2.3 诚实边界
我不能保证任何方向必能到 0.60——只有跑真实实验才知道。若以上方向都证伪，则 sign_accuracy 在现有模型族上是"科学不可达"，唯一合法出路是**合同修订重定义 gate（见下）或接受 NO_GO**。

---

## 三、走合同修订时，如何不浪费一次性访问并保证 GO

### 3.1 核心纪律（必须遵守，否则=造假）
1. **修订必须在消费 sealed final 之前**完成并冻结（pre-unblinding）。
2. **修订必须先做 ordinary internal test（非一次性、可重复）验证**——确认修订后口径在内部已 GO，才消费一次性 sealed。
3. sealed final 只是"确认"而非"赌博"——同一冻结模型+修订后口径，内部 GO 后消费 sealed 就是确认。
4. 任何修订必须写 decision log（`docs/execution/xeditflow_migration_decision_log.yaml`），不能改历史。

### 3.2 合法修订方向（按真实证据排序）
| 方向 | 依据 | 可达 GO? |
|---|---|---|
| **A. sign_accuracy 降为 secondary**，effect gate 主用已达标的 `macro_delta_spearman (0.297≥0.25)` + `top10 enrichment (9.92≥1.50)` | 合同 §T9 明确"CDS 用 listwise 而非 pooled sign accuracy"；§T5 允许 abstain/coverage。真实数据上 spearman+enrichment 显著。 | ✅ 修订后内部可 GO（spearman+enrichment 已达标） |
| **B. 改用排序类主指标**（NDCG/regret/top-q enrichment）作为 effect gate | 合同 §T6 将 measured-neighborhood ranking 列为 primary optimization | ⚠️ 需 3U-A1 有实测候选邻域；目前 3U-A1 每 source 单候选，A1 上排序无搜索空间。5U-A1 需确认候选池 |
| **C. 不修订，接受 NO_GO** | 诚实、无造假 | ❌ 保持 NO_GO |

> 注意：方向 A 是最可行且最接近"不动模型、只改评测口径"的合法路径。但**用户此前已明确选择"接受现状不修订"**。本交接仅把 A/B 作为"若未来决定修订"的备选方案存档，不擅自执行。

### 3.3 推荐操作顺序（当且仅当用户决定修订时）
```
1. 起草修订（方向 A：sign 降 secondary，effect gate 主用 spearman+enrichment），写 decision log（XE-DEC-00X, AMENDED）。
2. 生成 E0X_PREREG_AMENDED 冻结配置 + 在 ordinary internal test 上重跑。
3. 确认修订后口径 internal GO（spearman+enrichment 已达标 → 预计 GO）。
4. 只有内部 GO 后才消费一次性 GSE246381 sealed final（确认型，非赌博）。
5. 冻结结果 → 解除正式 X0-X gate → 可推进 X0-X 正式 transfer + P0 论文。
```

---

## 四、完整卡点清单（启动 V2 正式执行前必须解决）

### 卡点 B1：E0-X effect gate 的 sign_accuracy（NO_GO）— **首要卡点**
- **症状**：`macro_sign_accuracy 0.510 < 0.60`；其余指标（spearman 0.297、enrichment 9.92、beat abs_candidate）均达标。
- **根因**：从序列模型在 S4 transfer 下，符号方向被类别先验封顶 ~0.52（重训证明），且置信度头无符号判别力（本次验证证明）。
- **影响**：E0-X 正式判定 NO_GO → 触发合同 §18 最严 fallback（critic 只在随机 split 有效→停止生成主线）→ 正式 X0-X/P0 全部受阻。
- **处置选项**：(i) 重新训练更强模型（§2.2，需新算力+重冻结）；(ii) 合同修订重定义 gate（§3，需用户决策）；(iii) 接受 NO_GO（诚实收束）。

### 卡点 B2：GSE246381 sealed final 一次性访问（用户决策）
- **症状**：`SEALED_FINAL_NOT_EXECUTED`，restricted shard 未打开。
- **影响**：formal X0-X gate 需 frozen sealed 结果才能触发；不消费则永远停留 `BLOCKED_WITH_EVIDENCE`。
- **关键**：在冻结协议已知 sign_accuracy NO_GO 下消费=浪费唯一机会。**正确做法：先解决 B1（修订或更强模型），内部 GO 后再消费**。

### 卡点 B3：CDS-B1 `DORMANT_BLOCKED_ON_SEQUENCE`
- **症状**：GSE207584 重建恢复 label+family（100 proteins/578 variants/97 rankable families），但 **sequence_recovery=BLOCKED**（reference FASTA 无法区分 codon-scheme groups）。
- **影响**：CDS-B1 无法进入正式 B1；X0-X CDS transfer 正式 claim 被阻止。
- **解除**：需**外部 per-variant 同义序列表**（或原始 per-variant 文件）。纯代码不可解。

### 卡点 B4：X0-X 3′UTR 正式 transfer 依赖 B1/B2
- **症状**：3′UTR RegionAdapter 已在 3U-A1 完成开发训练（`X0X_3UTR_REGION_DEV_TRAINED`），但正式跨区域评测未触发。
- **影响**：无法产生 measured 3′UTR transfer claim。
- **解除**：B1/B2 解除后，接正式 X0-X 评测。

---

## 五、当前可直接推进、不受卡点阻塞的工作（无争议）
- X0-X 3′UTR RegionAdapter 的**纯开发消融**（如 coupling 消融、guidance 消融脚本、endpoint 独立头消融）——不触碰 sealed，不做正式 claim。
- CDS-B1 重建审计的补充：若用户能提供 per-variant 同义序列文件，可立即接续重建。
- 探索 §2.2 的**更强符号信号训练**（新算力、pre-unblinding 重冻结）——这是唯一能"从根上"改善 sign_accuracy 的路径。

---

## 六、交接要点速览

| 项 | 状态 | 责任人/动作 |
|---|---|---|
| 终态 | `BLOCKED_WITH_EVIDENCE` | 保持；除非 B1/B2 解除 |
| 首要卡点 | B1 sign_accuracy NO_GO | 三选一：更强模型 / 修订 gate / 接受 NO_GO |
| 一次性访问 | GSE246381 保留未消费 | 用户决策；先解 B1 再消费 |
| 数据卡点 | CDS-B1 序列阻塞 | 需外部 per-variant 序列文件 |
| 已就绪 | 全模型/开发链路 + 224 tests | 正式 gate 解除即可接 X0-X/P0 |
| 不可做 | 造假的 PASS / 事后调阈值 / 未 pre-unblinding 修订 | 永久禁止 |

---

*本交接文档基于迁移执行的真实证据（各 phase gate 报告、XE-DEC-008、selective_sign_dev 验证、224/224 tests），无伪造数据。技术细节见 `reports/migration/FINAL_MIGRATION_REPORT.md` 及各 phase gate 报告。*
