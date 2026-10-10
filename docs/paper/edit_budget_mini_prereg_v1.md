# MEF 编辑预算测试 mini-prereg v1（2026-10-10 · 计算前冻结）

- **change-id**: mef-edit-budget-mini-prereg-v1-20261010
- **状态**: FROZEN（本文件落盘 commit 后，任何结果计算方可开始）
- **上游**: 用户 2026-10-10 拍板 D1-D9（`mRNA_EditFlow_科研项目交接审计_20260826/MEF编辑预算测试交接执行文档_v1.md` §3，逐条照搬，不得更改）
- **执行人**: TRAE agent（交接审计会话）

## 0. 科学问题

被测模型从 source 序列出发做编辑、直到性质达到目标值，统计用了多少步编辑（编辑预算）。步数越少 = 设计效率越高。与数据集真实实验步数对照。补 DeltaBench 判别面之外的「生成/决策效率」维度。

## 1. 口径（D1-D9 拍板，逐条冻结）

| # | 决策点 | 冻结内容 |
|---|---|---|
| D1 | 被测对象 | 打分类（矩阵 5 移植族，本轮执行；14 族其余为存档族无同口径冻结前向，登记）+ 生成式（3 外部组 + 1 我方 SetFlow；**本轮先执行打分类臂**，生成式臂分两步走：GEMORNA 条件生成 + SetFlow rollout，通用 LM proposal 组若接口成本超预期按降级条款处理） |
| D2 | 目标口径 | 真实 candidate 目标：target = 该 source 的真实 measured candidate 的 `direction_normalized_delta`；模型用自己的预测器判定「达到」 |
| D3 | 任务范围 | polyA 主战场（2,628 行）+ MRL 参照行（730 行） |
| D4 | 判定门 | 达到 = 模型预测的 direction-normalized delta ≥ target delta；**B_max = 实测 95 分位 + 1**：polyA = 22+1 = **23**；MRL = 2+1 = **3**（实测 p95：polyA 22.0 / MRL 2.0，2026-10-10 服务器实测）；超限记 FAIL_budget 照登 |
| D5 | 打分类协议 | 全枚举贪心：每步枚举所有单点突变（4L−1 候选；polyA L=164 → 655 候选/轮；MRL L=50 → 199 候选/轮），模型 frozen 打分取最优，逐步迭代；预算步数 = 迭代轮数；零调参 |
| D6 | 生成式协议 | 模型原生输出编辑序列 → 与 source diff 得步数（不做贪边包装）；采样性环节 3 seeds 取中位 |
| D7 | 生成式清单 | GEMORNA（设计打分头，官方权重已在 port ledger）+ 通用 LM frozen proposal + 我方 SetFlow（v5 产物线） |
| D8 | 产出路径 | mini-prereg（本文件）→ 全量跑批 → 判定表 → 论文 §8.5 → PPT 新页，全部 append-only |
| D9 | 规模 | 全量：polyA 2,628 + MRL 730；生成式 3 seeds |

## 2. 指标定义（冻结）

- **median steps / mean steps**：达标所需贪心轮数的中位/均值（超 B_max 记哨兵值并单列 FAIL_budget 率，不并入 steps 统计）
- **FAIL_budget 率**：未在 B_max 内达标的 (source, target) 测试占比
- **budget ratio** = k_model / k_true（同源同目标配对；k_true = 该 candidate 的真实 `source_relative_edits` 长度）；报告 median 与 IQR
- 达标判定每步执行：`pred_delta(cur) >= target_delta`，其中 pred_delta(x) = scorer(x) − scorer(source)，与 frozen-Δ 口径完全同源（scorer = 官方权重零调参，复用 benchmark_v2_family_adapters_v1.build_scorer）

## 3. 冻结参数（数值今日实测）

- polyA：n=2,628 行（902 源组）；k_true median 7 / mean 8.52 / max 36；p95=22 → B_max=23；L=164（655 候选/轮）
- MRL：n=730 行；k_true median 1 / mean 1.28 / max 3；p95=2 → B_max=3；L=50（199 候选/轮）
- 参照冻结值（模型侧 sanity 锚，来自 matrix_v2_results.json）：GEMORNA polyA 0.222513 / MRL 0.219562；LAMAR polyA 0.1483 / MRL 0.0737；UTR-Insight polyA 0.3136 / MRL 0.2739；UTR-STCNet polyA −0.0217 / MRL 0.8144；HydraRNA polyA −0.3719 / MRL −0.0156
- 打分类被测 = 矩阵 5 移植族（gemorna / lamar_utr5te / utr_insight / utr_stcnet / hydrarna）；每条 (source, target-candidate) 记录独立测试（一源多 candidate 逐条分组），budget ratio 对 k_true 逐条配对
- tie-break = 字典序最小（冻结）；打分类确定性无采样

## 4. 降级条款（预登记）

1. 若全部打分类 FAIL_budget → 结论 = 「贪心引导不足以在预算内达标」（与 Δ 可学性互洽的如实正结果，照登）
2. 若步数普遍 < k_true → 报告时注明「模型预测器与真实测定的偏差可能贡献」
3. 通用 LM proposal 组若接口成本超预期 → 降级为 GEMORNA + SetFlow 双生成器，结果表如实标注少一组
4. 若 polyA 全量贪心计算超预算 → 按族分批落盘中间产物（断点续跑），不改协议
5. 生成式臂（GEMORNA 条件生成 / SetFlow rollout）在打分类臂终态后单独执行（GPU 排队不阻塞主线）

## 5. 禁止事项

- 禁 peak-picking（B_max 与判定门冻结后不得调整）
- 只用 VALIDATION（本测试不触 TEST）
- 结果不得以 smoke / 单例 sanity 充当全量结论
- 产物 /mnt，代码 /home + push；CUDA 硬门（cpu_fallback=false 留证）

## 6. 验收（对齐 MEF 文档 §5）

1. 本文件在任何计算前 commit（哈希入 journal）
2. 结果 JSON 逐记录可回源；判定表数字与 JSON 一致
3. 论文/图/PPT 数字与判定表逐 token 一致（独立复核条款 8）
4. 既有冻结数字零改动
