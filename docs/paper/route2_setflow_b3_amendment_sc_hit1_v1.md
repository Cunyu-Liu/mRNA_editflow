# Route2 SetFlow B3/评估 Amendment v1 —— 主判据换 support-conditional hit@1 + 独立评估双口径

> **状态：AMENDMENT v1（2026-09-08，起草待用户确认生效）**。触发条件满足：DP2/DP3 拍板（2026-09-08，spec §R.10）——选项 0+1+2+3+4 全部执行，主判据换 support-conditional hit@1 + 双口径，0.35 留痕。修订走 amendment 流程（D5 先例 `route2_v8_d5_mprau_target_amendment_v1.md`），**旧判据与旧门槛不删，留痕**。预注册纪律：本文件经用户确认后生效，判定门变更以本文为准；生效后不得事后修改，再修订走 v2。

## 1. 原判据与原门槛（合同口径，保留留痕）

- **原 B2 主判据**：source_macro candidate_recovery_rate（recovery@budget，k=10 注释）——guided vs unguided Δ ≥ +0.05 且 source-group paired bootstrap 95% CI 不跨零 且 hit@1 不劣化。
- **原 B3 门槛**：guided recovery ≥ **0.35**（V4 协议时代标定：当时 unguided 记忆化基线 0.283–0.292）。
- **历史沿革**：B3 0.35 定标时架构尚未去记忆化（V4 时代 unique 0.41–0.68，靠复现训练见过的模式命中）；V5 换轨去记忆化后该门槛从未重估。

## 2. 修订依据（证据链，全部 Phase A/B 机械实测）

| # | 证据 | 数字 | 来源 |
|---|---|---|---|
| E1 | recovery@budget 与 critic 排序能力零相关（判据-机制错配） | 池内命中与排名无关（排第 32 也算），分母 = 每源 measured 数（2–4） | A1 口径审计（§R.1） |
| E2 | 引导（V5 与修复版 V8）对池覆盖零贡献 | support(32)：unguided 0.2424 / V5-guided 0.242 / V8-A 0.2435 / V8-B 0.2469 | A2 + B3 分解（§R.2/§R.9） |
| E3 | 32 预算下 B3 0.35 对两个口径均数学不可达 | recovery 口径需 B≈120（ρ 无关）；hit@1 口径需 ρ=1.48（不可能） | A4 power study（§R.3） |
| E4 | hit@1 headroom 与上界 | 实测 0.0367；理论上界 = support = 0.242（完美排序）；6.6× headroom | A1/A2 |
| E5 | base 排序的结构性天花板 | base_reachable%（true-best measured 被 base 提出过）：MRL 12.4 / MPRAU 2.8 / HL 5.4 / polyA 0 | A3 结构级（§R.6） |
| E6 | V8 裁决 3/3 闭合 | V8（专才/joint/h_bench9）≤ V5（0.032/0.036/0.046 < 0.061） | A3.3（§R.6） |
| E7 | 阈值历史标定错位 | 0.35 系记忆化时代标定，考核去记忆化架构 = 范畴错误 | F10/§R.3 |

## 3. Amendment 内容（正式修订）

### 3.1 新主判据：support-conditional hit@1（sc-hit@1）

**定义**：hit@1 限制在「池内含 ≥1 measured」的源子集上（即 support 命中的源），tie-aware 口径与现 hit@1 完全一致（top-1 并列块准入概率，`evaluate_route2_generation_v1.py::measured_neighborhood_metrics` 现行实现）：

```
sc-hit@1 = mean_{s ∈ S+} hit@1(s)，S+ = {s : 池内 measured 命中数 ≥ 1}
```

**理由**：hit@1 无条件平均被 76% 零覆盖源稀释成「池覆盖 × 排序」混合物（0.0367 ≈ 0.242 × 排序质量 ≈ 0.152）；条件化后成为**纯排序能力**指标，分母诚实、排序 headroom 可见（上界 1.0）。**必须与 support@B 并报**（防分母漂移：通过把 support 砌高同时排序变差也可能维持 sc-hit@1，双字段强制防该退化模式）。

**计算**：直接复用既有 per_source 字段（all_generated_candidates_measured / measured_candidate_count / hit_at_1 的 per-source 值）——现有产物即可重算，无需新评估代码；adjudicator 补充输出列。

**Bootstrap**：source-group paired（cluster=SOURCE_GROUP_PAIRED，270 组）paired bootstrap，2,000 iters，seed 20260816/20260817（与现 adjudicator 一致）。

### 3.2 B3 新门槛（分档预注册）

| 档 | 条件 | 新门槛 | 依据 |
|---|--- |---|---|
| **B 档 B=32**（现行配置） | B=32 | **sc-hit@1 ≥ 0.10**（绝对线）**且 sc-hit@1 ≥ 2× base 自身排序（结构性对照）** | A4：B=32 下 hit@1 上界 0.242×ρ；诚实带估 0.08–0.12 |
| **A 档 B=256**（选项 4 扩池臂） | B=256 | **sc-hit@1 ≥ 0.30** 且 Δsupport vs B=32 ≥ +0.10 | A4：B=256 → support ~0.62、hit@1 上界 0.62×ρ |

**A 档门槛前置条件**：A 档门仅在「选项 4 扩池臂」运行时生效；该臂诚实定位 = H4 检验（覆盖头room 是否真实存在）+ 新口径压力测试，**不作为通过旧 B3 的捷径**（诚实性条款：见 §3.4）。

**2× base 对照的意义**：base 自身排序在 support 命中源上 = natural-hit 子集 base acc@1 = 0.1514（A3 实测）——这就是「不用引导、靠 base 自家 ordering」的免费线。**任何引导/critic 方案必须显著超过这条线才值得存在**（V5 探针 0.076 / V8 0.044–0.074 均在线下——**这条对照线正是 Phase A 探针已暴露的「引导在 base 可触达源上帮倒忙」问题的量化靶子**）× 这句话对 q 模型/β sweep 等任何新方案一律适用。

### 3.3 独立评估双口径（对冲 H5 表观循环）

recovery 家族（recovery@budget / support / hit@1 家族）共同的口径风险：它们奖励「复现实验者测过什么」。对冲条款（D1 披露条款的正式化）：

1. **主口径**（本 amendment 生效后）：sc-hit@1 + support@B 并报；
2. **辅助口径（强制）**：frozen independent evaluator（frozen Optimus — MRL；frozen APARENT — polyA）对生成候选打 delta 分布；对每源 top-1 候选（tie-aware）报告 **delta 分布 + guided vs unguided paired bootstrap CI**——如果某方案在 recovery 家族上变好、但在独立评估 delta 上不升反降，**该方案的主口径结果如实报告 + 表观循环嫌疑如实标注**（不自动 FAIL，但标注为「口径敏感的正结果」，对冲声明纳入论文披露）。
   - MPRAU/HL 任务无 frozen 独立评估器（域内无公开强先验模型）——如实标注「双口径仅覆盖 MRL/polyA（891 中 672 源）」，不伪造第三口径。

### 3.4 诚实性条款（防「花钱买通过」）

- 扩池臂（A 档）的通过**不回溯适用于 B=32 档的任何结论**；论文口径：B=32 与 B=256 结果分档报告，禁止跨档比较叙事。
- recovery@budget **永久降级为覆盖诊断字段**（不再是任何门的判据）。
- 负结果条款：若 0+2+3+4 全部执行后 B 档与 A 档门均未过，项目状态 =「诊断完备的负结果 + 生成器侧四方向排除」，phase-out 文档如实收口（不追加第五方向）。

### 3.5 判定门汇总（替换表）

| 门 | 旧 | 新 |
|---|---|---|
| B2 主判据 | Δrecovery ≥ +0.05 CI 不跨零 | **Δsc-hit@1 ≥ +0.03 且 CI 不跨零**（B 档），**≥ +0.10**（A 档），hit@1-family 与 support 并报 |
| B3 绝对门 | guided recovery ≥ 0.35 | **sc-hit@1 ≥ 0.10（B 档）/ 0.30（A 档）+ 2× base 对照线** |
| 0.35 绝对门槛 | — | **留痕不删**（V4 记忆化时代标定，历史文档保留） |

> **B2 改判说明**：Δsc-hit@1 门（B 档 +0.03）比旧 Δrecovery +0.05 字面更严不成比例——因为 sc-hit@1 的 headroom（0.152→1.0，6.6×）远大于 recovery 的（0.12→0.24 覆盖上限 × ρ）。+0.03 = 上界的 ~20%，与旧门在旧口径中的相对位置匹配。

## 4. 与其他文档的关系

- 本 amendment **不修改** V5 冻结产物与历史判定（B2/B3 FAIL 记录留痕）；仅约束**本 amendment 生效之后**发射的实验（C2 β sweep / C4 q 集成 / 扩池臂）。
- 选项 4（B=256 扩池）与选项 3（C2 β sweep）的判定门**以本文件为准**；发射前 seed/config 冻结（DP4 预注册纪律）。
- D2（A2 臂）缓议维持；D7 维持登记。

## 5. 生效流程

起草（本文件，2026-10-08 17:5x）→ 用户确认（回复"确认生效"或提出修改）→ journal 批次入档 + spec §R.11 引用 + commit → C2/C4/扩池臂的判定门指向本文件。**未确认前本文件为草稿，不影响在途发射**（当前无在途实验，无回溯风险）。
