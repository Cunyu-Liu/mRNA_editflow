# DeltaBench 主论文证据链严谨性审计 v1（对骨架 v2 逐 claim 攻击面审计）

- **审计对象**: `docs/paper/deltabench_main_paper_skeleton_v2.md`（§8 升格版）+ 其继承的 §5/§6/§2 核心数字。
- **审计立场**: 恶意审稿人视角——假设审稿人拿到 v2 骨架全部 claim-证据表与产物 locator，逐条找「这个数字撑不住」的攻击点。
- **状态分级**:
  - **SEALED（证据闭合）**：数字有落盘产物、口径声明完备、n/seed/家族构成可查，攻击面可当场用既有产物回击。
  - **GAP-可补跑**：证据存在但存在可低成本补强的实验空位（补跑后升 SEALED 或转 limitation）。
  - **GAP-不可补**：无法以新实验闭合（数据权利/n 固定/历史事实），只能声明 limitation。
- **审计方法**：每项 = 攻击面 → 现有证据实查（本审计执行时已逐项打开源 JSON/MD 比对）→ 状态 → 补跑方案与成本 → 优先级（P0 最高）。
- **纪律**: 本审计为只读操作（SSH 查产物 + 本地写文档），无训练无 GPU 前向无既有文件修改；protected TEST reads = 0。

---

## 一、逐项审计表（A1-A10 必查 + 附带 A11 汇总项）

### A1. polyA 91.3% 完成度与 V5 0.8219 的 seed 攻击面（§8.1/§8.2/§1 C4/Abstract 句 5）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | ① ICC 0.90 的计算出处与 n；② V5 0.8219 是单 seed 还是多 seed；③ Holm p=0.004 的家族构成；④ 91.3%/52.9% 的分母语义 |
| **现有证据（实查）** | ① ICC 0.90 出处 = `analysis_delta_failure_mechanism_e0_e1_e3_b2/mechanism_results_v2.json → label_icc_reference`（9 任务 ICC 表：MRL 0.83 / MPRAU 0.683 / polyA 0.90 / TE200304 0.586 / HL5 0.0014 / HL3 0.013 / TE149487 −0.274 / RNA149487 0.364 / REFALT 0.21）。polyA ICC 0.90 在该表中**无 n 字段、无计算过程块**（为 9 任务标签信度引用表；`analysis_ceiling_icc_20260907/ceiling_icc_results.json` 只覆盖 3 任务：TE149487 n=48 k=3 / RNA149487 n=48 k=3 / TE200304 n=1614 meta-analytic——polyA 行不在该复算产物内）。② V5 0.8219 = **单 seed frozen 终态**：`analysis_task8_bottomline_20260909/r1_r11_closure_audit_v1.md` R7 行原文「V5 冻结终态单次训练如实标注」；对位表注「critic V5 (frozen terminal)」。**polyA 主行无 3-seed 版本**（MRL 有 V9-1a 3-seed ens、MPRAU 有 5-seed ens，polyA 行独缺）。③ Holm 家族 = Task 12.1 bottom-line 终判表 **9 任务对位**（`bottomline_adjudication_v1.md`：MRL/polyA/MPRAU/TE/TE149487/RNA149487/REFALT/HL5/HL3 行，polyA raw_p=0.001→holm_p=0.004 唯一显著）。④ 91.3% = 0.8219/0.90；52.9% = (0.8219−0.7343)/(0.90−0.7343) = 0.0876/0.1657——分母为「最强外部行之上的剩余可学空间」，语义为自造指标（合理但须定义句）。 |
| **状态** | **GAP-可补跑（最大攻击面，P0）**——ICC 出处闭合但 n/复算块缺；V5 单 seed 是全骨架最锋利的攻击点：审稿人一句「你的 headline 数字没有训练 seed 方差带」即可要求 major revision。对位双方同为 frozen 单次（APARENT 权重固定）部分回击了公平性质疑，但 0.8219 的可复现性方差未知。 |
| **补跑方案与成本** | **P0 补跑 1（V5 polyA 3-seed）**：复用 V5 训练配方（xeditcritic_v5 v5_screen runner），seed 追加 2 个（如 20260921/20260922），各训 8 pass（V5 单 seed 训练时长历史 ~9h/8-pass，GPU 1 卡），共 ~2 卡日 + 评测（2,628 records frozen-Δ 前向 <10 min）。产出：3-seed 均值 ± range + 逐 seed CI；若三 seed 均值仍 >0.80 且 CI 维持排零，0.8219 升格为「3-seed 稳健」——SEALED。若方差大（如 ±0.03），如实改写为区间主张。**P0 补跑 2（polyA ICC 复算块）**：从 GSE269595 VALIDATION 标签做 split-half/ICC 复算（纯 CPU，分钟级，复用 ceiling_icc 协议），产物补 n 与方法块。成本合计：GPU ~2 卡日 + CPU 分钟级。 |
| **优先级** | **P0**（唯一动摇 headline 的项） |

### A2. STCNet 0.8144 推理确定性（§3.3 域内强读数）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | frozen 行 0.8144 的推理可复现性——同权重两次前向是否逐位一致（不一致则复核声明 65/65「浮点级一致」可能只是同 seed 同 batch 顺序的巧合） |
| **现有证据（实查）** | 复核协议本身已是「重算一致性」证据：`analysis_matrix_recheck_v1/recheck_results.json` = 独立脚本不读入档指标、从 predictions.jsonl 逐格重算，65/65 PASS max |Δ|=0.0（**含 STCNet MRL 0.814430506981109 极端格**）——但这是「从存档预测重算指标」，不是「重新前向」。重新前向级确定性**未单独留证**。矩阵执行时 STCNet 走 padd120 one-hot 端口（CPU/GPU 混合路径在 matrix 执行记录中 cpu_fallback 声明）。 |
| **状态** | **GAP-可补跑（P2，低成本）**——指标重算闭合，但审稿人可追问「re-run 前向是否同值」；paddle/keras 后端有非确定算子风险（见 A3）。 |
| **补跑方案与成本** | 对 STCNet（paddle）+ 2 个代表性族各重跑一次 frozen 前向（同 batch 顺序），比对 predictions 逐位（脚本级 diff）。STCNet 730+2805+5572 records 级前向 ~分钟-小时级（视 paddle CPU/GPU），GPU 需求低（可 CPU）。产出 `determinism_check_v1.json`（max_abs_diff 字段）。 |
| **优先级** | P2 |

### A3. 五新族 65 格的推理数值非确定性风险标注（§3.1/§3.2 + §9.3）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | paddle（STCNet）/ keras-tf（部分族）后端的 cuDNN/TF 算子级非确定性 + HydraRNA Triton（MIG env 修复史：autotuner 单设备可见性——autotune 本身可能引入跨 run 差异）；审稿人主张「65 格单 seed 单次前向，数字不可复现」 |
| **现有证据（实查）** | 三层既有回击：① 端口单测 5/5 PASS（官方例值级对齐：Insight 官方 CSV rho=0.96、RiboNN 官方 pearson 复现）= 与官方实现的数值对齐已证；② 复核脚本 65/65 浮点级一致（predictions→指标链闭合）；③ RiboNN 官方预测对齐 max diff 0.0034/0.0012（model0/model1）。**但「本机两次前向」级 determinism 证据缺**（A2 同源问题，扩展到 5 族）。 |
| **状态** | **GAP-可补跑（P2，随 A2 合并执行）** |
| **补跑方案与成本** | 5 族 × 各 1 次重复前向（复用 A2 determinism_check 脚本扩展 5 族；GPU 0.5 卡日内；HydraRNA 需固定 Triton autotune cache）。若个别族出现 1e-6 级漂移，论文加「数值非确定性上界」脚注（声明 rho 变动 <1e-4 不影响任何结论带）；若 paddle 逐位一致则直接 SEALED。 |
| **优先级** | P2 |

### A4. M1 密度双口径敏感性（§5.2 held-out FAIL 的 1.0014 口径）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | held-out FAIL 判定使用 M1 密度 1.0014，而该数是**评测行构建口径**（row_manifest `cand_per_source_density.mean`）而非 25 行拟合集的 **TRAIN within-source 密度口径**（batch 109 已自登记「非训练配对密度，解读时须区分」）。审稿人主张：口径不一致 → 预测 ρ 偏移 → in-band 率对 FAIL 判定敏感 |
| **现有证据（实查）** | `delta_vs_density_v2_results.json → density_caliber` 逐行声明：9 既有任务 = 25 行拟合集同款 within-source 口径；**M1 = 1.0014 / M6 = 1.5717 / S1 = 3.6682 均为 row_manifest eval-row 口径**（声明字段在但**双口径重算的 in-band 敏感性未执行**）。M1 行 2,805 records/2,801 源 = 密度 1.0014 近恒 1；若改用训练配对口径（GSE232927 源内的候选密度），数值可能不同（M1 配对 = 15nt 前缀家族 + Hamming-1 兜底，天然密度低——口径切换后密度大概率仍 ≈1-2，预测 ρ 变动 ≤0.1 decade 内，5 格判定面是否翻转需实算）。 |
| **状态** | **GAP-可补跑（P1）**——口径声明在（诚实），但敏感性量化缺。审稿人可要求「换口径后 FAIL 是否维持」。注意：**若重算后 FAIL 翻转为 PASS，反而动摇降级叙事**——但按预注册框架「禁改带禁挑行」，补跑属于敏感性分析而非改判（结果无论方向都如实报告，原判定以原口径为准——此点须在补跑 prereg mini-note 里先写明）。 |
| **补跑方案与成本** | M1（+顺带 M6/S1）密度换口径重算：从 M1 训练配对侧（GSE232927 canonical TRAIN 池的 15nt 前缀家族配对）算 within-source 密度 → 代入冻结拟合（斜率 0.3663/截距 −0.0857）→ 重出 65 格（或仅 M1 5 格）in-band 标记 → 报 in-band 率区间（双口径 band）。纯 CPU + 既有 cells JSON，脚本 <1h。**先落 mini prereg note（判定以原口径为准，本补跑为敏感性声明）再执行**。 |
| **优先级** | **P1** |

### A5. 一阶分解 α 的 VALIDATION 选择乐观偏差（§6.1/§8.3 一阶 0.4555）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | alpha 网格（0.3-100）按 VALIDATION Spearman 选优（`first_order_decomposition_prereg_v1` 协议）——用评测 split 选超参 = 一阶上限 0.4555 被乐观偏置抬高；审稿人主张「若 TRAIN-only 选 α，0.4555 会更低，55.4% 结论夸大」。反向攻击同样存在：偏置抬高 → polyA 一阶覆盖被**高估** → 「非一阶信号 0.27+」被低估——对 V5 不利方向。但 §8.2 的对外叙事（一阶解释 55.4%）依赖该数 |
| **现有证据（实查）** | `analysis_first_order_decomposition_v1/summary.md` §1 明示「alpha ∈ (0.3,1,3,10,30,100) 按 VALIDATION Spearman 取最优」（协议如实声明）；TRAIN-only α 敏感性版本**未执行**。MRL 参照对（E7 0.2069 vs ERK 0.2015）同样为 VALIDATION 选 α 但两线同口径互证。 |
| **状态** | **GAP-可补跑（P1）** |
| **补跑方案与成本** | TRAIN-only α 选择版本：Ridge 闭式拟合在 TRAIN 内做 α 选优（可用 TRAIN 内留出或 CV），重出 polyA/MPRAU/TE 三任务 ρ₁。纯 CPU（一阶闭式，分钟级/任务）。预期：α 选择对 Spearman 影响小（网格粗、闭式平滑），0.4555 变动大概率 <0.02；产出 `first_order_train_alpha_sensitivity_v1.json`。若变动大 → §8.3 一阶占比改区间表述。 |
| **优先级** | P1 |

### A6. 委员会路由 +0.0362 的出处/seed/功效（§8.4）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | ① 数字出处与变体选择（constructive vs dev-routed 两版本读数差异大：+0.0362 排零 vs +0.0045 跨零——审稿人质疑「报告了对提交有利的那版」）；② bootstrap seed 与 holdout 划分；③ CI 下限 0.0090 接近零的功效声明 |
| **现有证据（实查）** | ① 出处闭合：`option2_constructive_ci.json` = constructive zero-peek 路由（HL→full / MRL→v8_hbench9 / MPRAU→v8_smprau_in / polyA→v8_smprau_in；「构造性」= 无 dev 数据窥探的先验路由），Δ+0.0362 CI[+0.00897,+0.06315] holdout n=446 排零；dev-routed 版（dev-only argmax 选择）+0.0045 CI[0.0,0.0112] 跨零在 `option2_committee_routing.json` 并存——**两版分层已在产物中，v2 骨架 §8.4 已强制并报**（骨架审计通过，攻击面在「变体定义合法性」：constructive 路由是否预注册？实查：`option2_committee_routing.json → protocol.registration` 字段原文「script committed before probe re-run outputs were read」——预注册链在）。② bootstrap = 2,000 iters seed 20260816 paired per-source；split = stratified by task 50/50 seed 20260909。③ 功效：CI 下限 +0.009 距零 <1/4 CI 宽——效果脆弱性声明缺（n=446 holdout 下 +0.036 点估计的功效可算但未算）。 |
| **状态** | **SEALED（带分层声明义务）**——出处/seed/预注册/双版本并报全部闭合；残余 = 功效脚注（可在写作期用 Fisher z 或 bootstrap 重采样 0.5×CI 声明 MDE），或补 bootstrap seed 扰动敏感性（换 3 个 bootstrap seed 看 CI 稳定性，纯 CPU 分钟级——可选补强非必需）。 |
| **补跑方案与成本** | 可选小补强：bootstrap seed 20260816 → {20260901, 20260902, 20260903} 重采样 CI 四次（读 per_source JSONL 重跑，CPU 分钟级）。若任一 seed CI 触零 → §8.4 措辞降级为「seed 敏感」。 |
| **优先级** | P3（可选） |

### A7. COMB polyA 三行 n=20 report-only 的扩样问题（§8.4）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | D 臂三行（+0.3527/+0.4210/+0.4844）n=20 report-only——审稿人主张「n=20 无统计力，三行数字不可引用」 |
| **现有证据（实查）** | `comb_tier2_polya_aparent.json` 原文：n=20 stratum、report_only=true（prereg sec.3 条款）、caliber = frozen APARENT base proximal_log2_odds (cut 80..105) tie-aware top-1。**n=20 的构成实查**：委员会路由 per_task_counts 证实 891 池 = HL 111 + MPRAU 108 + MRL 652 + polyA 20——**polyA 源在 891 生成池中只有 20 个，n=20 即该口径的全量，不是抽样**。CI 不可算的原因 = n=20 而非漏算。扩样上限：891 池内扩样空间为 0；**换池扩样**（更大 polyA 生成池）= 新生成实验（非「重收割既有产物」）。891 全量既有产物（D891_main/seed16/seed17 guided 产物树 + comb_tier2_* JSON）已全部收割完毕，无未收割余量。 |
| **状态** | **GAP-不可补（池内）/ GAP-可补跑（换池，P3）**——正确处置 = limitation 声明（「n=20 = 891 生成池内 polyA 源全量；扩样需新生成池，超出本文范围」）。若审稿人强烈要求：换池 = 新 512/1024 polyA 源生成 + COMB D 臂重跑（GPU 数卡日级 + 周级墙钟——与 M1 臂排队冲突，不建议现在排）。 |
| **补跑方案与成本** | 不补跑（默认 limitation 声明）；备选：新生成池实验（GPU ~4-6 卡日 + 3 seed 墙钟 ~2 周）——仅当审稿意见强制。 |
| **优先级** | P3（声明优先于补跑） |

### A8. oracle 臂 4 行 3 seeds 充分性 + 协议发现 Methods 表述完备性（§4.2-4/§10）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | ① oracle 结论（MRL NO_SIGNAL / polyA EQUIVALENT）只测了 RNA-FM/UTR-LM 两族 3 seeds——「其他表征也 NO_SIGNAL 才能下一般结论」；② supervision_form 协议发现（docstring 与代码不一致、两监督形式数学等价）是否在 Methods 完备披露 |
| **现有证据（实查）** | ① `results_oracle_probe.json` 4 行 × 3 seeds（per-seed 全列 + seed 敏感伪信号发现本身入档——MRL 0.1370→0.0640/0.0597 塌缩演示）；**一般化边界**：骨架 §4.2 措辞已限定「MRL 两族 NO_SIGNAL」非「所有表征」（B10 条款合规）；跨族一般化靠 §3.4 矩阵 20 格近零 + §7.4 分类互证（非 oracle 单证）。残余：mRNABERT 等 3 族未入 oracle 臂（属扩展非缺陷——oracle 臂预注册范围即 2 族）。② 协议发现：journal 批次 108 原文「已入 protocol note（results_oracle_probe.json → supervision_form_audit_note）」+ 骨架 §4.3/§10.7 均已披露——**Methods 表述完备**（骨架级审计通过）。 |
| **状态** | **SEALED**（① 措辞已自限边界 + 矩阵/分类互证链闭合；② 披露链闭合）。可选扩展：oracle 臂加测 1-2 新族（mRNABERT frozen 表征 probe，GPU 0.5 卡日）——增强非必需。 |
| **补跑方案与成本** | 可选：mRNABERT/HydraRNA 嵌入 oracle probe 各 3 seeds（复用 oracle 协议，GPU MIG 即可，~2h/族）。预期 polyA 侧 EQUIVALENT 复现、MRL 侧 NO_SIGNAL 复现 → 一般化升格。 |
| **优先级** | P3（可选增强） |

### A9. M1/M6/S1 新行 n=800/2,805/5,572 的 power 声明（§2.2/§3.4 新行集体近零）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | 「新行上 5 新族集体 <0.09」——审稿人主张：这些行样本量/源数下近零读数可能是检验力不足（|ρ| 小 ≠ 0），近零结论需要 MDE 声明；尤其 M6 n=800/509 源、S1 双臂 |
| **现有证据（实查）** | `analysis_baseline_power_20260909/power_statement_v1.md` = Fisher z MDE 框架已成文，但**覆盖对象是 bottom-line 对位族**（MRL/polyA/MPRAU/TE/149487/186455 8 对位），**M1/M6/S1 新行的 15 格（5 族×3 行）无逐格 MDE**。近零读数的对称面：矩阵 cells 有 bootstrap CI 吗？实查 `matrix_v2_summary` 声明口径 = frozen-Δ 点估计（CI 未逐格给——65 格点值 + 复核一致，但无 per-cell CI）。反证链在：STCNet M1 0.0672 vs GEMORNA 0.0889 等 15 格全 <0.09 的「集体性」本身就是一致性证据，且 M1 5 格中 4 个 MRL 域内族「托底高于带」读数（§5.3）说明检验力可分辨 0.1 级差异。 |
| **状态** | **GAP-可补跑（P1）** |
| **补跑方案与成本** | 新行 15 格 per-cell bootstrap CI + MDE 声明：对 predictions.jsonl（137,350 行既有存档）做 per-cell source-group bootstrap 2,000 iters（纯 CPU，~1h 15 格），产出 `new_rows_power_v1.json`（逐格 CI/MDE/power 标记）。预期 CI 均含但宽度 <0.1 → 「近零带读数与 CI 一致」SEALED；个别格 CI 宽 → 加 power-limited 标注。**零新前向**（纯重算）。 |
| **优先级** | P1 |

### A10. 天花板 ICC 表 locator 正确性（HL 纠错核对）

| 维度 | 审计结果 |
|---|---|
| **攻击面** | v1 骨架 §7.1 claim 表曾用 `ceiling_icc_20260907` 为 ICC 出处——journal 批次 108 已纠错「HL 真实源 = e0_e1_e3_b2/mechanism_results_v2.json，非 ceiling_icc_20260907」。若 v2 仍有错引，审稿人复算对不上即信任崩塌 |
| **现有证据（实查）** | 本审计逐项打开两 JSON 比对：`mechanism_results_v2.json → label_icc_reference` = 9 任务全表（HL5 0.0014 / HL3 0.013 / TE149487 −0.274 等）；`ceiling_icc_results.json` = 仅 3 任务（TE149487 n=48 k=3 ICC −0.2742 / RNA149487 0.3641 / TE200304 0.5857 meta-analytic）。**交叉一致**：两源在重叠 3 任务上数值一致（−0.274 / 0.364 / 0.586≈0.59 表值）→ label_icc_reference 的 9 任务表可信。**v2 骨架核对结果**：v2 已全文改用 label_icc_reference 为统一源（§2.4/§8.1 素材映射行 + §7.1 claim 表双引两源并注「佐证」）——**v2 引用正确**。注意点：`label_icc_reference` 的 TE200304 = 0.586 而 `ceiling_reference`（同文件另一块）= 0.59——骨架 §2.4 用 0.586、§2.4 大纲老文写 0.59 的混用风险已在 v2 统一为 0.586。polyA 0.90/MRL 0.83/MPRAU 0.683 三个关键上界的**计算过程与 n 仍未落盘**（A1 补跑 2 覆盖 polyA；MRL 0.83/MPRAU 0.683 同缺——MRL 0.83 出处疑为 730-record 标签侧 ICC（k 重复），MPRAU 0.683 疑为 3+3 split-half（first_order summary 表注「3+3 split-half」）——写作期需把这两个上界的方法句补进 §10.7）。 |
| **状态** | **SEALED（locator 层面）/ GAP-可补跑（方法句层面，并入 A1 补跑 2 扩展）** |
| **补跑方案与成本** | 并入 A1 补跑 2：ICC 复算脚本一次跑 9 任务（polyA 2,628 / MRL 730 / MPRAU split-half + 其余引用任务），产出 `label_icc_recompute_v1.json`（每任务 n/k/方法/ICC + 与 label_icc_reference 对照行）。CPU 分钟-小时级。 |
| **优先级** | P1（随 A1） |

### A11.（汇总项）V5 polyA 主行外的 §5/§6 继承数字

| 维度 | 审计结果 |
|---|---|
| **攻击面** | v1 已自查的 76 行数字（批次 113 曾逐字符核对）；v2 新引数字 = CMS −0.2261 / W0 +47% / constructive CI / APARENT2 0.6810 / comb fingerprint 等 |
| **现有证据（实查）** | v2 新引数字全部实查通过：CMS polya −0.2261（`option3_cmsaug_adjudication.json → per_task_side_effects.note` 原文「polya -0.2261 / mrl 0.1809」）；W0 +47%（journal 行 66「vs V5 多任务 0.1354：单任务训练 +47% 相对提升」= 0.1987/0.1354−1）；constructive CI（A6 已核）；APARENT2 0.6810（`result.json → task_macro_spearman`）；B+C 0.02974/D 0.02972（`comb_tier2_harvest.json → verdict.additivity_fingerprint`）；top-1 0.5007 vs 0.6011（journal 行 446/560 + task1_alignment ranking 块 −0.1004 CI 与 v1 的 −0.100 一致）。 |
| **状态** | **SEALED** |
| **优先级** | — |

---

## 二、状态统计

| 状态 | 数量 | 项 |
|---|---|---|
| **SEALED** | 4 | A6（带分层义务）、A8、A10（locator 层）、A11 |
| **GAP-可补跑** | 6 | A1（P0）、A4（P1）、A5（P1）、A9（P1）、A2+A3（P2 合并）、A6 可选补强（P3） |
| **GAP-不可补** | 1 | A7（池内 n=20 上限——声明 limitation；换池补跑为 P3 备选不入默认清单） |

- 语义说明：A2/A3 计一项合并（determinism_check 单实验覆盖两项）；A6 主状态 SEALED、可选补强另计 P3。**十项必查中最大攻击面确认为 A1（V5 polyA 单 seed）**，与用户预判一致。

---

## 三、补跑实验清单（按优先级）

> 执行纪律：每项先落 mini prereg note（判定口径/预期带/禁改判条款）再执行；全部 VALIDATION-only；GPU 排队需与 M1 干预臂（在途 seed 20260920）协调——**P0/P1 项 GPU 需求低，可穿插 MIG 空闲卡；不与 M1 抢整卡**。

| 优先级 | 实验 | 覆盖审计项 | GPU 需求 | 预计时长 | 产出 |
|---|---|---|---|---|---|
| **P0-1** | V5 polyA 训练 3-seed（seed 追加 ×2，frozen-Δ 评测 + CI） | A1 | GPU 1 卡 × ~9h × 2（可两卡并行 1 天内完） | ~1-2 天（含评测） | `v5_polya_3seed_v1.json`：3-seed 均值/range/逐 seed CI；0.8219 升格或改区间表述 |
| **P0-2** | label ICC 9 任务复算块（polyA/MRL/MPRAU 等，n/k/方法全落盘） | A1 + A10 | CPU | 分钟-小时级 | `label_icc_recompute_v1.json`（含与 label_icc_reference 对照） |
| **P1-1** | M1/M6/S1 密度双口径敏感性重算（先 mini prereg：判定以原口径为准） | A4 | CPU | <1 天 | `density_caliber_sensitivity_v1.json`（双口径 in-band 率区间） |
| **P1-2** | 一阶分解 TRAIN-only α 版本（polyA/MPRAU/TE） | A5 | CPU | ~小时级 | `first_order_train_alpha_sensitivity_v1.json`（ρ₁ 区间） |
| **P1-3** | 新行 15 格 per-cell bootstrap CI + MDE（纯重算 predictions.jsonl） | A9 | CPU | ~1h | `new_rows_power_v1.json`（逐格 CI/MDE/power） |
| **P2-1** | 5 族前向 determinism check（STCNet paddle 重点 + HydraRNA Triton autotune cache 固定） | A2 + A3 | GPU 0.5 卡日（可 CPU/MIG） | ~半天 | `determinism_check_v1.json`（max_abs_diff per family） |
| **P3-1（可选）** | 委员会路由 bootstrap seed 扰动 ×3 | A6 | CPU | 分钟级 | CI 稳定性四联表 |
| **P3-2（可选）** | oracle 臂扩展 2 新族 × 3 seeds | A8 | GPU MIG ~2h/族 | ~1 天 | oracle 一般化增强 |
| **不补跑** | COMB polyA n=20 扩样 | A7 | —（换池 = 4-6 卡日 + 2 周墙钟，与 M1 冲突） | — | 论文 limitation 声明（n=20 = 891 池 polyA 源全量） |

- GPU 总需求（默认清单 P0-P2）：约 **3-4 卡日**（P0-1 占 2 卡日、P2-1 半卡日、其余 CPU）；墙钟 ~3-5 天（穿插 M1 臂排队空隙）。
- 汇报纪律：每项产物落 /mnt 新目录 + 逐格对照本审计表更新状态（GAP→SEALED/limitation），骨架 v2 不回改数字（补跑结果为敏感性/稳健性层，进 §9.3 与全文撰写期）。

（审计 v1 完——对象骨架 v2；下一动作 = P0-1/P0-2 补跑或全文撰写，二者可并行）
