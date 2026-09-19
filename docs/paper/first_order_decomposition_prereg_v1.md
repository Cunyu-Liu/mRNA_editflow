# 预注册：一阶/结构信号分解分析（Task 1.1 机理链 B）— first_order_decomposition_prereg_v1

日期：2026-09-19
写定时间：任何结果计算开始**之前**（本文件先行落盘并 commit，再跑拟合脚本）。
执行 worktree：route_a_v3_w0_diagnosis_20260902；产物根：`experiments/analysis_first_order_decomposition_v1/`。

## 1. 目的

把 E7-a v2 式**一阶上下文能量表闭式拟合**（MRL 上 val 0.2069 ≈ ERK 1.2K 学习模型 0.2015 → "MRL 可学部分 ≈ 一阶信号"）
扩展到三个代表任务：polyA（GSE269595）、MPRAU（ENCSR854RUF）、TE（GSE200304），
产出各任务的**一阶可学上限**，并与对应外部行 / 我方 critic V5 / 天花板 ICC 对照，
按 spec R3 判定框架逐任务判定分解叙事"成立 / 部分成立 / 不成立"。

## 2. 模型（精确复刻 E7-a v2 probe，run_erk_e7_v2.py，零改动公式）

一阶上下文能量表（闭式 Ridge 拟合）：

```
Delta_hat(candidate | source) = sum_{e in edits(candidate|source)} W[block(e), ctx(e)] + b[source]
block(e) = pos(e) // 8                # 8nt 位置块
ctx(e)   = idx(left_nt)*16 + idx(right_nt)*4 + idx(alt_base)   # 64 组合
```

- left/right nt 取自 **原始 source 序列**（U→T，pos-1 / pos+1；边界或非 ACGT 记 'N'：
  双 N 跳过该 edit；单 N 用另一侧回填——与 run_erk_e7_v2.py 完全一致）。
- `b[source]` = source background，TRAIN 内 target encoding（per-source 均值，向全局均值按 5 先验平滑；
  验证期未见过的 source 用全局均值）——与 E7 完全一致。
- 拟合 = `sklearn.linear_model.Ridge`，对 `y - b` 拟合，预测 `X @ w + b`；
  alpha 网格 (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)，按 VALIDATION Spearman 取最优（与 E7 完全一致）。
- L（序列长上限）与 n_blocks 按各任务 TRAIN+VALIDATION 的 max(序列长, 最大 edit pos + 1) 定，
  n_blocks = ceil(L/8)，特征数 = n_blocks x 64。
- 无二阶项（本分析只评一阶可学上限；E7b 的二阶增量另行引用，不在此判定内）。
- 只换数据源任务，公式/正则/特征构造/闭式解/口径**零改动**。

## 3. 数据与口径

- 数据：`projections/xedit_v3/development_train_validation_v1/train.jsonl + validation.jsonl`（canonical 记录，
  标签 direction_normalized_delta）。
- 任务与 task_id：
  - polyA = `PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1`（GSE269595）
  - MPRAU = `MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1`（ENCSR854RUF）
  - TE = `TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1`（GSE200304）
- 拟合 = TRAIN split；评测 = VALIDATION split。
- 各任务主口径（与外部行 / V5 行同口径）：
  - polyA：任务 Spearman（VALIDATION 2,628 行）
  - MPRAU：**variant pair-mean ρ**（rid 去掉 `:context:` 后缀分组为 variant，≥2 context 的 variant，
    per-variant 对 target 与 prediction 分别取均值后，在 2,008 个 variant 上做 Spearman）——
    与 W-ladder / V6-H3 / Saluki MPRAU 行、`analyze_route2_v6_swa_offline_v1.py` 逐字同口径；
    记录级（record-level）ρ 作为附加报告行（V5 record-level 0.0732 同口径对照）。
  - TE：任务 Spearman（VALIDATION 1,614 行）
- 纪律：protected TEST reads = 0；纯 CPU；不修改任何既有产物。

## 4. 对照行（数值在写定时刻从既有档案冻结）

| 任务 | 一阶闭式 ρ | 无稠密监督外部行带（frozen-Δ） | 稠密监督行 | critic V5 | 天花板 ICC |
|---|---|---|---|---|---|
| polyA | 本实验产出 | UTR-LM 0.7490 / RNA-FM 0.7114 | APARENT 2019 0.7343 | 0.8219 | 0.90（ICC） |
| MPRAU (pair-mean) | 本实验产出 | UTR-LM 0.0147 / RNA-FM 0.0180（CI 跨零） | Saluki 0.1205（弱对照，非该任务稠密监督） | 0.1025 | 0.683（ICC） |
| TE | 本实验产出 | UTR-LM 0.0113 / RNA-FM 0.0009 | （无该任务稠密监督行） | 0.0579 | 0.5857（meta-analytic ICC） |

注：
- polyA 的外部行普遍偏高（APARENT 为稠密监督对照，UTR-LM/RNA-FM 为无稠密监督带）→ 判定时**如实报告**，
  不因外部行高而调宽阈值。
- MPRAU 外部带无稠密监督模型均为 CI 跨零的零信号行；Saluki 0.1205 为弱对照（非 MPRAU 专属稠密监督），
  在 MPRAU 判定中充当"外部最强行"用于差距比较，但稠密监督差条件按"不适用"处理（见 §5）。
- TE 无稠密监督外部行带只有 frozen-Δ 通用 LM 两行（≈0）；无该任务专属稠密监督外部行。
- 我方行：critic V5（polyA 0.8219 / MPRAU pair-mean 0.1025 / TE 0.0579）。
- MPRAU V5 record-level 0.0732 与本实验 record-level 附加行同口径对照。
- 天花板 ICC：polyA 0.90 / MPRAU 0.683 / TE 0.5857（TE 为 meta-analytic ICC single estimate；
  ICC 档案为标签信噪比天花板，非模型可达上限，报告用）。

## 5. 预注册三分支判定规则（量化阈值，先于结果写定）

记：ρ1 = 本实验一阶闭式 ρ；U = 该任务无稠密监督外部行带的**中位**（polyA: median(0.7490, 0.7114)=0.7302；
MPRAU: median(0.0147, 0.0180)=0.0164；TE: median(0.0113, 0.0009)=0.0061）；
D = 稠密监督行（polyA: APARENT 0.7343；MPRAU: Saluki 0.1205 但记为弱对照——稠密差条件**不适用**；
TE: 无稠密监督行，稠密差条件**不适用**）。

条件 A（≈ 外部带）：|ρ1 − U| ≤ 0.05
条件 B（显著低于稠密监督）：D − ρ1 > 0.05

- **成立**：A 且 B 均满足。
- **部分成立**：A、B 恰好其一满足。
- **不成立**：A、B 均不满足。
- 条件 B 不适用的任务（MPRAU、TE）：按"部分成立"为上限的退化判定——
  - 若 A 满足（ρ1 落入无监督带 ±0.05）→ 判"部分成立（B 不适用）"；
  - 若 ρ1 > U + 0.05（一阶超过无监督带）→ 判"不成立（一阶超出无监督带，超出部分归非一阶/结构信号）"；
  - 若 ρ1 < U − 0.05（一阶低于无监督带）→ 判"不成立（一阶不足以解释外部行信号）"。
- polyA（B 适用）：按完整三分支判定；如实报告外部行偏高的事实（APARENT/UTR-LM/RNA-FM 均 0.71–0.75 带，
  V5 0.8219 仍为最强行）。
- 附带报告（不参与判定，只进叙事）：一阶 ρ vs critic V5 差、vs 天花板 ICC 差、
  一阶 ρ 占 V5 的比例、MPRAU record-level ρ。

## 6. 交付物

- `experiments/analysis_first_order_decomposition_v1/results_first_order.json`（逐任务：一阶 ρ + alpha +
  特征数 + n_train/n_val + 对照行 + 判定）
- 图：`first_order_vs_external_vs_ceiling.{png,pdf}`（条形图：一阶 vs 外部行 vs V5 vs 天花板，逐任务面板）
- `summary.md`（逐任务判定 + 章节底稿素材）
- 拟合脚本（worktree scripts/route_a_v3/，复刻 E7 公式）。

## 7. 诚实性声明

- 阈值 0.05 为写定时刻先验选定（对应"带内≈"与"显著低于"两个直觉量级），未看任何本实验结果。
- E7 既有 polyA/MPRAU probe 行（0.504 / 0.063，含二阶项 + alpha VALIDATION 选择）为既有档案，
  本实验**只评一阶**（无二阶），口径差异（是否含二阶特征、MPRAU pair-mean vs 记录级）在 summary 中如实标注。
- 本实验为纯 CPU 只读分析，不触碰 protected TEST，不修改既有产物。
