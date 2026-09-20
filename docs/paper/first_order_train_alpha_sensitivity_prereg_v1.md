# 一阶分解 TRAIN-only α 敏感性 mini prereg（P1-2，审计 A5）

- **状态**: FROZEN（计算前落盘，2026-09-20）
- **触发**: 严谨性审计 A5——一阶分解（analysis_first_order_decomposition_v1）的 α 网格 (0.3, 1, 3, 10, 30, 100) 按 **VALIDATION Spearman** 选优，用评测 split 选超参 → 一阶上限读数（polyA 0.4555 / MPRAU 0.0825 / TE 0.0458）存在乐观偏置风险。

## 1. 敏感性定位声明

- **本分析不改任何已冻结读数与结论**：原 VALIDATION 选 α 版本（results_first_order.json 入档值）保持为主读数；本补跑只报告 TRAIN-only α 选择版本的对照读数，作为 §8.3「一阶解释 55.4%」主张的稳健性脚注素材。
- **零公式改动**：完全复用 run_route2_first_order_decomposition_v1.py 的特征构建（E7-a v2 ctx_feats 一阶项）、源编码（per-source mean + grand-mean prior 5）、Ridge 闭式拟合、评测口径（polyA/TE = VALIDATION task Spearman；MPRAU = variant pair-mean rho）；唯一变更 = α 选择不再看 VALIDATION。
- **预期带（先写后算）**：α 网格粗（6 点）+ Ridge 闭式平滑 → TRAIN-only 选择的 α 与 VALIDATION 选择的 α 大概率相同或相邻，ρ₁ 变动预计 |Δ| < 0.02（审计 A5 预估同量级）。若变动大（≥0.05），如实报告并建议 §8.3 一阶占比改区间表述（该建议本身不在本批执行）。

## 2. TRAIN-only α 选择协议（计算前冻结）

- **方法 = TRAIN 内 5-fold CV**：在 TRAIN 上按 source_id 分组做 5-fold（GroupKFold 语义：同一 source 的行不跨 fold，防止源编码泄漏放大选择信号）；每 fold 内重估源编码（该 fold TRAIN 部分的 per-source mean）后拟合 Ridge，预测 held-out fold；α 取 5-fold 平均 Spearman（对 MPRAU 用 pair-mean 口径在 fold 内近似：按 canonical_record_id 的 variant 前缀聚合）最优者。CV seed = 20260920（sklearn GroupKFold 无随机性 + 指定 permutation seed 固定 fold 分配）。
- **对照读数**：α_train 与 α_val（原版选择的 α）并报；ρ₁(train-α) 在同一 VALIDATION 面上评测（评测面不变——敏感性只针对超参选择过程，不针对评测面）。
- 三任务：polyA / MPRAU / TE（与原版一致）。

## 3. 纪律

- 只读既有投影（projections/xedit_v3/development_train_validation_v1）；protected TEST reads = 0；CPU only。
- 产物 `experiments/analysis_first_order_decomposition_v1/sensitivity_train_alpha/`（results.json：三任务 TRAIN-only α 一阶 ρ vs 原值对比 + 本 prereg 一段内嵌）。
- 本文件与补跑脚本随批次 commit（W0 worktree）。
