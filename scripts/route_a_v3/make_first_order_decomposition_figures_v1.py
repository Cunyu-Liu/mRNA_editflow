#!/usr/bin/env python3
"""Figure + summary for analysis_first_order_decomposition_v1 (Task 1.1 chain B).

Reads results_first_order.json (produced by run_route2_first_order_decomposition_v1.py),
draws the preregistered bar figure (first-order vs external rows vs critic V5 vs
ceiling ICC, per-task panels; png + pdf) and writes summary.md.
Pure CPU, read-only on the results json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_first_order_decomposition_v1"
RES = json.loads((OUT / "results_first_order.json").read_text(encoding="utf-8"))

C_FIRST = "#1f77b4"
C_UNSUP = "#7f7f7f"
C_DENSE = "#d62728"
C_V5 = "#2ca02c"
C_CEIL = "#000000"

EN = {"成立": "UPHELD", "部分成立": "PARTIALLY", "不成立": "NOT-UPHELD"}

LABELS = {"polyA": "polyA (GSE269595)\ntask Spearman, n_val=2,628",
          "MPRAU": "MPRAU (ENCSR854RUF)\nvariant pair-mean rho, 2,008 variants",
          "TE": "TE (GSE200304)\ntask Spearman, n_val=1,614"}
VERDICT_COLOR = {"成立": "tab:green", "部分成立": "tab:orange", "不成立": "tab:red"}


def main():
    tasks = ["polyA", "MPRAU", "TE"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, label in zip(axes, tasks):
        t = RES["tasks"][label]
        ref = t["reference"]
        rows = []
        rows.append(("first-order closed-form rho1", t["rho_first_order"], C_FIRST))
        for name, v in ref["frozen_delta_unsupervised"].items():
            rows.append((f"{name} (frozen-delta, no dense supervision)", v, C_UNSUP))
        for name, v in ref["dense_supervision"].items():
            tag = "weak control" if "weak" in name else "dense supervision"
            rows.append((f"{name} ({tag})", v, C_DENSE))
        rows.append(("critic V5 (ours)", ref["critic_v5"], C_V5))
        rows.append(("ceiling ICC", ref["ceiling_icc"], C_CEIL))

        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [r[2] for r in rows]
        y = range(len(rows))
        bars = ax.barh(y, vals, color=colors, alpha=0.85)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.invert_yaxis()
        for yi, v in zip(y, vals):
            ax.text(v + 0.012, yi, f"{v:.4f}", va="center", fontsize=8.5)
        verdict = t["adjudication"]["verdict"]
        vshort = verdict.split("（")[0] if "（" in verdict else verdict
        extra = ""
        if "B" in verdict:
            extra = " (cond-B n/a)"
        ax.set_title(f"{LABELS[label]}\nprereg verdict: {EN.get(vshort, vshort)}{extra}",
                     fontsize=10.5, color=VERDICT_COLOR.get(vshort, "black"))
        ax.set_xlabel("Spearman rho")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.suptitle("First-order context energy table (closed-form) vs external rows vs critic V5 vs ceiling ICC"
                 " (Task 1.1 mechanism chain B, preregistered adjudication)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"first_order_vs_external_vs_ceiling.{ext}", dpi=200)
    print("figure saved:", OUT / "first_order_vs_external_vs_ceiling.{png,pdf}")

    md = build_summary(RES)
    (OUT / "summary.md").write_text(md, encoding="utf-8")
    print("summary saved:", OUT / "summary.md")


def build_summary(res) -> str:
    t = res["tasks"]
    polyA, mprau, te = t["polyA"], t["MPRAU"], t["TE"]

    def row(name, v, extra=""):
        return f"| {name} | {v:.4f} | {extra} |" if isinstance(v, float) else f"| {name} | {v} | {extra} |"

    md = f"""# 一阶/结构信号分解分析 v1（Task 1.1 机理链 B）— summary

日期：{res['date']}（执行）；预注册：`docs/paper/first_order_decomposition_prereg_v1.md`（计算前落盘，见 §纪律）。
脚本：`scripts/route_a_v3/run_route2_first_order_decomposition_v1.py`（E7-a v2 公式精确复刻，仅一阶项，无二阶）。
数据：`projections/xedit_v3/development_train_validation_v1`（canonical，direction_normalized_delta 标签；
TRAIN 拟合 / VALIDATION 评测；protected TEST reads = 0；纯 CPU）。

## 1. 模型（E7 复刻口径）

```
Delta_hat(candidate|source) = sum_e W[block(e), ctx(e)] + b[source]
block(e) = pos(e)//8（8nt 位置块）；ctx(e) = idx(left_nt)*16 + idx(right_nt)*4 + idx(alt_base)（64 组合）
left/right 取自原始 source 序列（U→T；双 N 跳过、单 N 回填，与 run_erk_e7_v2.py 逐字一致）
b[source] = TRAIN target encoding（per-source 均值，grand-mean 先验 5）
Ridge 闭式解拟合 (y-b)；alpha ∈ (0.3,1,3,10,30,100) 按 VALIDATION Spearman 取最优
```

与 E7-a v2 probe 唯一差异：**去掉二阶 pair-block 特征**（本分析只评一阶可学上限）；
alpha 网格与 E7 完全一致。MPRAU 主口径 = variant pair-mean ρ（rid 去 `:context:` 分组、≥2 context、
per-variant 均值、2,008 variants），与 W-ladder / V6-H3 / Saluki MPRAU 行、
`analyze_route2_v6_swa_offline_v1.py` 逐字同口径。

## 2. 逐任务结果与对照表

### polyA（GSE269595，task Spearman，n_val=2,628，n_train=25,710，特征 {polyA['n_features']}，alpha={polyA['alpha']}）

| 行 | ρ | 备注 |
|---|---|---|
| **一阶闭式 ρ₁（本实验）** | **{polyA['rho_first_order']:.4f}** | 一阶可学上限 |
| UTR-LM frozen-Δ（无稠密监督） | 0.7490 | 外部带 |
| RNA-FM frozen-Δ（无稠密监督） | 0.7114 | 外部带（带中位 0.7302） |
| APARENT 2019（稠密监督对照） | 0.7343 | |
| critic V5（我方） | 0.8219 | 任务最强行 |
| 天花板 ICC | 0.90 | 标签信噪比上限 |

- 判定：**{polyA['adjudication']['verdict']}**（条件 A 不满足：|0.4555−0.7302|=0.2747 > 0.05；
  条件 B 满足：0.7343−0.4555=0.2788 > 0.05）。
- 读数：一阶 ρ₁ 仅覆盖 V5 的 {polyA['narrative_extras']['first_order_over_v5_ratio']*100:.1f}%（0.4555/0.8219）；
  polyA 外部行确实普遍偏高（无稠密监督带 0.71–0.75 ≈ APARENT 0.7343 同档）——**如实报告**：
  本任务即便无稠密监督的外部 LM 也能达到 0.71+，其信号**不是一阶表能解释的**（ρ₁ 比无监督带低 0.27）。
- E7 既有参照：E7-a v2（一阶+二阶，alpha VALIDATION 选择）polyA 0.5040——本实验一阶-only 0.4555，
  与 E7 一阶 v1（0.4882，pos×base 无上下文）接近；polyA 的二阶/上下文增益小。
- 叙事：polyA 的可学信号中约 0.46 可由一阶加性表解释，0.27+ 属**非一阶（结构/上下文组合）信号**，
  这部分恰好是外部 LM（0.71–0.75）与一阶表之间的缺口；V5 0.8219 再超出外部带 0.07，
  为我方任务特异性增益（距 ICC 0.90 还差 0.078）。

### MPRAU（ENCSR854RUF，variant pair-mean ρ，2,008 variants，n_train=55,704，特征 {mprau['n_features']}，alpha={mprau['alpha']}）

| 行 | ρ | 备注 |
|---|---|---|
| **一阶闭式 ρ₁（本实验，pair-mean）** | **{mprau['rho_first_order']:.4f}** | 一阶可学上限（主口径） |
| 一阶闭式 ρ₁（record-level，附加行） | {mprau['rho_record_level']:.4f} | 对照 V5 record-level 0.0732 |
| UTR-LM frozen-Δ（无稠密监督） | 0.0147 | CI 跨零 |
| RNA-FM frozen-Δ（无稠密监督） | 0.0180 | CI 跨零（带中位 0.0164） |
| Saluki（弱对照，非 MPRAU 稠密监督） | 0.1205 | |
| critic V5（我方） | 0.1025 | pair-mean 口径 |
| 天花板 ICC | 0.683 | 3+3 split-half |

- 判定：**{mprau['adjudication']['verdict']}（一阶超出无监督带，超出部分归非一阶/结构信号）**
  （条件 A 不满足：ρ₁−0.0164=0.0661 > 0.05，一阶**显著高于**零信号外部带；
  条件 B 不适用——Saluki 为弱对照非 MPRAU 专属稠密监督）。
- 读数：一阶 ρ₁ 0.0825 **超过** V5 pair-mean 0.1025 的 {mprau['narrative_extras']['first_order_over_v5_ratio']*100:.1f}%
  （差 −0.0200），远超无监督带（0.016）与 V5 record-level（0.0732，record 级一阶 0.0630 亦接近）。
- E7 既有参照：E7-a v2 MPRAU 0.0631（一阶+二阶，**记录级**口径，alpha=0.3）——本实验 pair-mean 主口径
  0.0825 / record-level 0.0630，两者自洽（pair-mean 聚 context 噪声后信号略升）。
- 叙事：MPRAU 上"外部无监督行 = 零信号"而一阶表已有实质信号（0.082）→ **该任务我方领先
  不是靠非一阶结构信号，而是靠一阶表即可获得大部分**；V5 相对一阶的增量仅 +0.02。
  分解叙事在 MPRAU 上**不成立**：可学部分基本就是一阶信号，外部 LM（frozen-Δ）连一阶都没学到。

### TE（GSE200304，task Spearman，n_val=1,614，n_train=3,318，特征 {te['n_features']}，alpha={te['alpha']}）

| 行 | ρ | 备注 |
|---|---|---|
| **一阶闭式 ρ₁（本实验）** | **{te['rho_first_order']:.4f}** | 一阶可学上限 |
| UTR-LM frozen-Δ（无稠密监督） | 0.0113 | 外部带（中位 0.0061） |
| RNA-FM frozen-Δ（无稠密监督） | 0.0009 | 外部带 |
| 稠密监督行 | （无该任务稠密监督外部行） | 条件 B 不适用 |
| critic V5（我方） | 0.0579 | |
| 天花板 ICC（meta-analytic） | 0.5857 | |

- 判定：**{te['adjudication']['verdict']}**（条件 A 满足：|0.0458−0.0061|=0.0397 ≤ 0.05；
  条件 B 不适用——无稠密监督行）。
- 读数：一阶 ρ₁ 0.0458 覆盖 V5 的 {te['narrative_extras']['first_order_over_v5_ratio']*100:.1f}%
  （差 −0.0121）；外部带 ≈0（0.0009–0.0113）。注意 A 满足的机制：外部带中位（0.0061）与一阶（0.0458）
  差 0.04 恰好落在 ±0.05 带内——按预注册规则如实判"部分成立（B 不适用）"。
- 叙事：TE 与 MPRAU 同构：外部 frozen-Δ 无信号，一阶表 ≈ V5 的 79–81%——可学部分几乎全是
  一阶信号；V5 增量 +0.012。距 ICC 0.586 的大缺口（−0.53）属标签噪声/天花板，不是模型差距。
  （TE 每 record 恰 1 个 edit → 一阶表与 per-position 上下文表完全等价，无组合效应空间。）

## 3. 判定总表（预注册规则：A = |ρ₁−带中位|≤0.05；B = 稠密行−ρ₁>0.05）

| 任务 | ρ₁ | 无监督带中位 | A | B | 判定 |
|---|---|---|---|---|---|
| polyA | 0.4555 | 0.7302 | ✗（0.2747） | ✓（0.2788） | **部分成立** |
| MPRAU | 0.0825 | 0.0164 | ✗（0.0661，一阶更高） | 不适用 | **不成立（一阶超带）** |
| TE | 0.0458 | 0.0061 | ✓（0.0397） | 不适用 | **部分成立（B 不适用）** |

## 4. 机理链 B 章节底稿素材（要点）

1. **分解叙事的任务异质性是主结论**：MRL 的"可学部分 ≈ 一阶信号"（0.2069 vs 0.2015）
   **不可外推**到三个代表任务——
   - polyA：一阶只解释 0.4555，外部无监督带 0.73 → polyA 的可学信号大部分是**非一阶**
     （上下文组合/结构），这与 G4 结构盲发现互补：polyA 高分行（UTR-LM 0.749/RNA-FM 0.711）
     组内仅 ρ 0.19/0.23（delta_validity 误差相关分析）——跨源信号支撑高 ρ，一阶表（含 source background b）
     只到 0.4555。
   - MPRAU / TE：一阶表已达到 V5 的 80%（0.0825/0.0458 vs 0.1025/0.0579）且**显著高于**
     零信号外部带 → 这些任务上我方行几乎就是一阶信号，"结构/高阶增益"主张在这些任务上要克制表述。
2. **外部行的双面读法**（如实报告）：polyA 外部行 0.71–0.75 普遍高于一阶上限——
   说明"无稠密监督"不等于"无信号"，通用 LM 的预训练迁移已超过一阶表；
   MPRAU/TE 外部行 ≈0（CI 跨零）——外部 LM frozen-Δ 连一阶信号都拿不到。
3. **天花板对照**：V5 距 ICC 的缺口在 MPRAU（−0.58）/TE（−0.53）上是标签噪声主导
   （icc 档案为标签信噪比上限，非模型可达上限）；polyA V5 0.8219 距 ICC 0.90 仅 0.078，
   一阶 0.4555 距 V5 0.366 → polyA 的结构信号空间最大，与"polyA 主缺口在决策口径（top-1）"的历史读数一致。
4. **与 E7 档案的口径对齐声明**：E7-a v2 polyA 0.5040/MPRAU 0.0631 含二阶 pair-block 特征且
   MPRAU 用记录级口径；本实验一阶-only polyA 0.4555（略低，二阶+alpha 选择贡献 ~0.05）、
   MPRAU pair-mean 0.0825 / record 0.0630（与 E7 记录级自洽）。MRL 参照对（0.2069 vs ERK 0.2015）
   维持既有档案，不在本实验重算范围。
5. **一句话**：一阶闭式能量表给出各任务"可学部分"的下界锚点——polyA 一阶只到 0.46
   （0.27 属非一阶信号，外部 LM 已拿到）；MPRAU/TE 一阶 ≈ V5 的 80%
   （V5 增量 +0.02/+0.01，外部 frozen-Δ 为零信号）——分解叙事按任务分别为
   部分成立 / 不成立（一阶超带）/ 部分成立。

## 5. 纪律与诚实性

- 预注册文档于计算前落盘并 commit（`docs/paper/first_order_decomposition_prereg_v1.md`）；
  阈值 0.05 与三分支规则在结果产出前写定，未做任何事后调整。
- protected TEST reads = 0；只读 `development_train_validation_v1` TRAIN/VALIDATION；
  未修改任何既有产物；纯 CPU（约 1 分钟）。
- alpha 按 VALIDATION Spearman 选择（与 E7 相同的既有口径；E7 同样如此，如实保留，
  这使 ρ₁ 成为轻度乐观估计——与外部行/V5 的同类 VALIDATION 选择口径一致）。
- TE 每 record 恰 1 edit、MPRAU 每 record 恰 1 edit、polyA 平均 8.5 edits/record（3–36）——
  polyA 是唯一有编辑组合空间的任务，与判定结果的方向一致。

## 6. 产物清单

- `results_first_order.json`（逐任务完整读数 + 判定 + 参照行）
- `first_order_vs_external_vs_ceiling.png / .pdf`（本摘要 §2 的图形版）
- `summary.md`（本文件）
- 脚本：`scripts/route_a_v3/run_route2_first_order_decomposition_v1.py`（拟合）、
  `scripts/route_a_v3/make_first_order_decomposition_figures_v1.py`（图+摘要）
"""
    return md


if __name__ == "__main__":
    main()
