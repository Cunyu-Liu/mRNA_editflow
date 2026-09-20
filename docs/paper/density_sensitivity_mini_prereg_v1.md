# 密度双口径敏感性 mini prereg（P1-1，审计 A4）

- **状态**: FROZEN（计算前落盘，2026-09-20）
- **触发**: 严谨性审计 A4——delta-density v2 held-out FAIL 判定（in-band 33/65 = 50.77% < 70% 门）使用 M1 密度 1.0014 / M6 1.5717 / S1 3.6682，均为 row_manifest `cand_per_source_density.mean`（评测行构建口径），而非 25 行拟合集的 TRAIN within-source 配对口径。

## 1. 定位声明（本分析不改判定）

- **原口径判定为准，不动**：delta_vs_density_v2 的 verdict = FAIL（in-band 50.77%）与降级条款（密度单因子主张降级为任务特异性现象）**均维持不变**。本补跑是**敏感性报告**，不是改判重跑。
- **预注册框架条款重申**（delta_density_prereg_framework_v1.md）：禁挑行、禁改带（tolerance 0.10）、禁改门（70%）、禁回灌拟合（冻结斜率 0.3663/decade、截距 −0.0857 不动）。本敏感性仅替换 M1/M6/S1 三行的密度输入值，其余 62 格密度与全部 65 格 rho_observed 不动。
- **结果无论方向都如实报告**：若换口径后 in-band 率升高甚至过门，不得据此翻转 FAIL→PASS（口径不一致本身已触发降级条款）；若进一步降低，如实登记。

## 2. 替换口径定义（计算前冻结）

- **M1_MRL_EVAL_ROW**: 训练侧配对口径密度 = M1 全量 kept 1,128,562 行的配对池密度。定义：mother-variant 家族结构（prefix-15 组 + 全局 Hamming-1 兜底）下的候选数/源，与 25 行拟合集 TRAIN cand/source 同构口径。具体 = 6 文件配对池（prefix 组 + Hamming-1 兜底，9,578 对）的 candidate-per-source 密度（对数/源数，逐文件与汇总两粒度报告）。预期 ≈1.0-1.3（随机库天然近 1）。
- **M6_NDD5UTR_EVAL_ROW**: 替换为全量 1,507 variant-key（QC 后）的 Family 组密度（763 families 全量口径），而非抽样 800 行 / 509 源组口径（1.5717）。预期 ≈1.9-2.2。
- **S1_STABILITY_EVAL_ROW（3UTR/5UTR 同值）**: 替换为 eligible 3,742 行的变异 ID 级分组密度（同一变异族全量，行 × assay sub-row 展开前的变异 ID 去重分组），而非 5,572 sub-row / 1,519 源组口径（3.6682）。预期 ≈2-3。
- **其余 9 既有任务 12 格**：密度值一律不动（25 行拟合集同款 within-source 口径）。

## 3. 判定读出（预声明）

- 报双口径 in-band 率：原口径 33/65 = 50.77% vs 敏感性口径 X/65。仅替换 M1 5 格 + M6 5 格 + S1 10 格 = 20 格的密度（其余 45 格 in-band 标记不动）。
- 结论三档表述（先冻结）：(a) FAIL 判定对口径切换稳健（in-band 变化 <5pp 且不过门）；(b) 量级敏感（≥5pp 但不过门，降级条款仍生效）；(c) 过门（≥70%）仍不改判（§1），登记为口径依赖边界情形交主会话裁决。
- 预测带（先写后算）：M1 ≈1.0-1.3 / M6 ≈1.9-2.2 / S1 ≈2-3；20 格翻转预计 0-4 格，in-band 率预计 50-57%（FAIL 维持）。

## 4. 纪律

- 只读既有产物；protected TEST reads = 0；CPU only；不改 verdict 与已冻结判定。
- 产物 `experiments/analysis_delta_vs_density_v2/sensitivity_density_caliber/`（results.json + 对比表 md）。
- 本文件与补跑脚本随批次 commit（W0 worktree）。
