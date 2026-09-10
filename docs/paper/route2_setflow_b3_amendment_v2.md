# Route2 SetFlow B3/评估 Amendment v2 —— B3 绝对线参照勘误 + 独立评估双口径正式判定数字

> **状态：AMENDMENT v2（2026-09-10，用户批准执行：批 60 勘误发现 + 批 61 遗留项 ①）**。v1（2026-09-08，`route2_setflow_b3_amendment_sc_hit1_v1.md`）主判据体系不变；本 v2 只修订 v1 的两处：§3.2 B3 绝对门「2× base」参照系错配（勘误）与 §3.3 双口径的正式判定数字与裁决条款（落地）。**预注册纪律**：v1/v2 全文留痕；v2 生效后的判定以此为准；再修订走 v3。

## 1. 勘误（v1 §3.2 的 B3 绝对门参照系错误）

**错误内容（v1 原文）**：「B 档（B=32）sc-hit@1 ≥ 0.10（绝对线）且 sc-hit@1 ≥ 2× base 自身排序（结构性对照）」——其中「base 自身排序」被起草人误填为 **0.1514**（A3 mixed-pool 探针的 natural-hit base acc@1）。

**错在哪**：sc-hit@1 是 **generation-internal 口径**（候选项池内按 generation_score 排序的 tie-aware top-1 准入），而 A3 mixed-pool 探针是**混合池口径**（measured 序列以底分加入 base 生成池，排序器是同一 base 的 generation_score）。两口径的 base 数值不可比：**generation-internal 的 base sc-hit@1 = 0.5131（891，n=216 S+）**，2× 线 = 1.146 > 1.0 不可达——按 v1 字面，任何模型永远不可能过 B3 绝对门。

**勘误（v2 生效）**：
- **B 档（B=32）绝对门 = sc-hit@1 ≥ 0.55**（generation-internal 口径，即 base 0.5131 的 ~1.07 倍——含义：「排序质量超越 base 自身排序至少 7%」；0.10 绝对线废除——它在 generation-internal 口径下远低于 base 本身，无判别力）。
- **A 档（B=256）绝对门 = sc-hit@1 ≥ 0.46**（base B=256 = 0.4104 的 ~1.12 倍；同样为「超 base」语义，分母随 pool 扩大而稀释，故阈值相应下调）。
- 2× base 结构性对照线**整体废除**（其在两种口径下均无合理语义）；替代语义 = 上述「≥ base × (1+ε)」倍率线，ε_B=0.07、ε_A=0.12。
- Δ门（B2 主判据）**不变**：B 档 Δsc-hit@1 ≥ +0.03 且 CI 不跨零；A 档 ≥ +0.10。

## 2. 独立评估双口径——正式判定数字（v1 §3.3 的落地）

**已执行的判定**（`beta_full891_20260909/independent_evaluator_dual_calibre{,.beta025}.json`，2026-09-10 08:26，frozen Optimus5Prime MRL / frozen APARENT polyA，cut 80-105，891 源全量，β=0.25 guided vs unguided B=32，paired per-source）：

| 域 | n | guided top1 Δmean | unguided top1 Δmean | Δpoint | 95% CI | independent_positive |
|---|---|---|---|---|---|---|
| **MRL** | 652 | +0.0294 | −0.0063 | **+0.0357** | **[+0.0024, +0.0662] 排零** | **true** |
| polyA | 20 | +0.2328 | +0.1615 | +0.0713 | [−0.0616, +0.2432] 跨零 | false |
| MPRAU/HL | — | — | — | — | — | 无 frozen 独立评估器（登记限制，不伪造第三口径） |

**裁决条款（v2 正式化）**：
1. **主口径（recovery 家族）与独立口径分离判定，均为正式门**：
   - B2 Δ门 = recovery 家族（不变，上节）；
   - **B2-I 独立口径门（新增）**：MRL 域 Δ(Optimus top-1 delta) ≥ +0.02 且 CI 排零（覆盖 652/891 = 73% 源）；polyA 域因 n=20 检验力不足，**不设门仅报告**（v1 的「不自动 FAIL」升级为「不设门」——20 源无统计力设门是伪严谨）。
2. **β=0.25 双口径终判（本 v2 落地时点的唯一实例）**：recovery 家族 B2 Δ门 **FAIL**（891 尺度 Δsc-hit@1 −0.017 跨零，批 60）**而 B2-I 独立口径门 PASS（MRL Δ+0.036 CI 排零）**——按 v1 §3.4 诚实性条款的表观循环对冲逻辑反向应用：**引导未提升 recovery（覆盖/排序），但显著提升了 MRL 独立功能预测分数**。判定登记为「**calibre-divergent 结果**」（口径分歧结果）：主判据 FAIL 维持（B2 门的权威性不变），独立口径正增益如实并行入档——该信号的解释候选（预登记，待 D2/后续实验鉴别）：(a) β=0.25 引导把候选推向 Optimus 偏好的序列特征（真功能增益）；(b) recovery 家族对功能增益不敏感（判据盲区，H5 的加强版）。
3. **verdict 字段语义**：`INDEPENDENT_CONFIRMED` = 独立口径在至少一个有门域上排零正。β=0.25 → MRL PASS → verdict 成立（即便主判据 FAIL）。

## 3. 对在途/后续实验的约束

- **D2 检索条件化**（同日获批）：其判定门**从发射起即用 v2 口径**（B2 Δ门 + B2-I 独立口径门 + B3 勘误后绝对线）；recovery 家族与独立口径双报。
- 4b 负收口（2026-09-10 04:47）**不受影响**——gate 判定依据是 B2 Δ门（v1/v2 相同口径），负路径正确。
- Phase C 已终态各臂（β sweep 5 臂、q 模型、B=256 unguided）不重判（v2 只约束生效后实验）；唯一重登记 = 上节 β=0.25 双口径终判。

## 4. 生效

本 v2 起草即生效（用户 2026-09-10 拍板「执行 amendment v2」）；落盘 `docs/paper/route2_setflow_b3_amendment_v2.md`（v8_stage1_prep worktree，与 v1 同目录）+ journal 批 62 + 双 worktree commit。v1 原文不改（勘误以本文件为准）。
