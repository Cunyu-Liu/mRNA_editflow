#!/usr/bin/env python3
"""Baseline Task 10.1: headline adjudication H1 / H2 / H-polyA / H3 with
numbers (Gate P preparation; Task 8 terminal dependency met).

10.2 (route/venue) and 10.3 (TEST opening) require user decisions — the
numbers here are the adjudication basis for those 拍板 items. All values from
on-file terminal products (no new computation).

Output: experiments/analysis_task8_bottomline_20260909/headline_adjudication_v1.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_task8_bottomline_20260909")

H1 = {
    "claim": "critic+flow 超过全部 baseline（底线兑现，W 阶梯收敛）",
    "verdict": "PARTIAL — met on 1 task family-wise, parity on the flagship, honest negatives elsewhere",
    "evidence": [
        "polyA: V5 0.8219 vs APARENT 0.7343, Δ+0.088 CI [0.069,0.120], Holm p=0.004 — the ONLY family-wise significant win (91% of 0.90 ceiling)",
        "MRL: V9-1a 3-seed ensemble 0.3217 / Route A 0.3158 vs frozen-Optimus 0.3132 — exact-bootstrap tie (p=0.767), point estimate ahead; top-1 0.4416 vs 0.4069 (3/3 seeds)",
        "MPRAU: s_mprau_in 5-seed 0.1351 (first in-house > V5 0.1025) vs Saluki 0.1205 — tie (p=0.895); external four-mode closure (frozen/FT/CMS/LLR) all ≈0 or negative = no external baseline beats us either",
        "GSE149487-RNA: registered loss to RNA-FM frozen (0.050 vs 0.296, n=48 power-limited) — the single per-task loss row",
        "HALF_LIFE: unlearnable (label ICC 0.001-0.013) — no baseline family learns it either (7-way external evidence)",
        "Generation: unguided 0.12046 / V5-guided 0.12626 B2 FAIL (Δ+0.0058 CI crossing) — flow+critic does not yet beat unguided at the original gate; β sweep terminal pending",
    ],
    "bottom_line": (
        "底线在 polyA 上以家族级显著性兑现；MRL 打到统计平局点估计领先（先验假说链闭合：280K 外部库监督 = 差距主因）；"
        "其余任务的对位结果由数据体制约束主导（power/标签/覆盖），W 阶梯执行至终态且负结果如实入档。"
        "H1 不成立为全任务宣言，成立为'1 显著 + 1 平局 + 体制约束定论'的诚实形态。"
    ),
}

H2 = {
    "claim": "绝对精度 ≠ 编辑优先级（frozen SOTA 退化；编辑优先级是独立任务）",
    "verdict": "SUPPORTED — five independent evidence lines",
    "evidence": [
        "matched-FT 对称退化（P1-1）：Optimus 0.2977 < frozen 0.3132 / FramePool 0.2300 < frozen 0.2956 — 薄任务数据微调损害强先验，跨阵营对称（与我方 Route A Step-2 同构）",
        "MPRAU matched-FT 坍塌（P1-5 3-seed）：RNA-FM −0.0747×3（逐位一致）/ UTR-LM −0.078~−0.107 — 通用 LM 监督微调系统性坍塌到负带",
        "LLR 零信号（P0-1）：NT-v2 0.0183 / HyenaDNA 0.0450 << V5 0.1025 — 似然比族无变体差分信号",
        "决策 vs 关联口径分叉（Task 4.2）：MRL NDCG 序（Optimus 居首）≠ Spearman 序（V9-1a 居首）",
        "HALF_LIFE 分离：绝对端点回归可学（Saluki monitor 0.0165）≠ 变体差分可学（全部行 ≈0）",
    ],
    "bottom_line": "绝对预测精度与编辑优先级（源相对差分排序）是可分离的任务性质；'拿绝对模型当编辑打分器'在四个模型族上系统性失效——方法学主张成立。",
}

H_POLYA = {
    "claim": "polyA 对位胜利（critic 多任务 vs APARENT 专才）",
    "verdict": "UPHELD — the single family-wise significant win",
    "evidence": [
        "Spearman: V5 0.8219 vs APARENT 0.7343 (Δ+0.088 CI 排零, Holm p=0.004) — 决策口径混合（top-1 −0.053~−0.100 显著负）如实并报（Task 1.3 口径）",
        "天花板归一 91%（0.8219/0.90）；V9-1a 3-seed 0.8404 均值再证稳定（天花板 93%）",
        "附加外部行不翻案：APARENT2 0.6810（2024 模型 frozen-Δ 未超 2019）/ UTR-LM 0.7490 / RNA-FM 0.7114",
    ],
    "bottom_line": "polyA 行是榜单上唯一可在 Holm 校正下宣称'统计稳健超越最强外部行'的任务；决策口径（top-1/NDCG）混合结果按预注册口径如实并报，不隐瞒。",
}

H3 = {
    "claim": "9 endpoint 可学性地图 + 跨 context 共享效应",
    "verdict": "DELIVERED — learnability map complete with per-task attribution",
    "evidence": [
        "可学性谱（Table 6 素材）：polyA 饱和带（91-93% 天花板）/ MRL 先验带（280K 外部库监督 = 差距主因，W0/W1'/W2 链闭合）/ MPRAU 数据体制带（s_mprau_in 0.1351 为 ~6K 对规模经验上界，远带学习侧阻塞）/ TE 小样本带（n=48/274 power-limited）/ HL 不可学带（标签 ICC 主导）",
        "跨 context 共享：V9-1a 跷跷板打破（任务专属容量 + per-task 头，MRL 0.322 + polyA 0.84-0.86 三 seed 稳定并存）但离流形崩塌 3/3 seed 一致（探针 0.027-0.036 vs V5 0.0614）——'骨干靠先验、任务靠隔离、判别力靠数据分布'三位一体归因链（V6 loss 层 / V8 先验层 / V9 架构层三层修复不解分布约束的完整证据链）",
        "guidance 侧映证（M3）：零训练任务路由专家委员会 0.0715 > 最强单模 V5 0.0614 > 全部统一权重行——'任务靠隔离'在 potentials 侧独立成立",
        "生成线覆盖约束：76% 零命中源主导 closed NDCG（joint 0.0273）——覆盖约束第四独立证据",
    ],
    "bottom_line": (
        "可学性地图交付：五带分类 + 每带归因（先验/数据体制/样本量/标签/覆盖）；"
        "跨 context 共享效应的答案是结构性的——统一容量与跨任务判别力在当前数据体制下不可兼得，"
        "任务路由（隔离）是已达成的工程最优。β sweep 终态将补生成线侧最后一块。"
    ),
}

GATE_P_ITEMS = {
    "10.2 路线/venue 拍板": "amendment 呈报已备（批次六十四 + 六十六扩充）：① 离流形探索臂 ② 零训练任务路由委员会臂（M3 新证据，零成本）③ CMS-as-training-augmentation 臂（15.6 新发现）④ V9-2 ⑤ 收官转论文——等待用户",
    "10.3 TEST 开启决策": "当前证据形态（1 显著 + 1 平局 + 体制约束定论）支持在 Gate P 开启 TEST 前先完成路线拍板；TEST 开启本身需用户预注册拍板（R11），protected reads=0 维持",
    "10.4 论文骨架": "Table 1-6 素材全部就位（Table 3 = Task 8.1 终判表 / Table 4 = closed NDCG + β sweep 终态 / Table 6 = 可学性地图）；R1-R11 自查待骨架重写时逐条",
}


def main() -> int:
    payload = {
        "schema_version": "route_a_v3_task10_1_headline_adjudication_v1",
        "H1": H1, "H2": H2, "H_polyA": H_POLYA, "H3": H3,
        "gate_p_pending_user": GATE_P_ITEMS,
        "note": "10.2/10.3 are user-decision items; this document is their numerical basis. β sweep terminal will finalize the generation-line entries.",
    }
    (OUT / "headline_adjudication_v1.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    md = ["# Task 10.1 headline adjudication（H1/H2/H-polyA/H3 附数字）", ""]
    for name, h in (("H1", H1), ("H2", H2), ("H-polyA", H_POLYA), ("H3", H3)):
        md += [f"## {name}: {h['claim']}", "", f"**判定：{h['verdict']}**", ""]
        md += [f"- {e}" for e in h["evidence"]]
        md += ["", f"**底线**：{h['bottom_line']}", ""]
    md += ["## Gate P 待拍板项（10.2/10.3/10.4）", ""]
    for k, v in GATE_P_ITEMS.items():
        md += [f"- **{k}**：{v}", ""]
    (OUT / "headline_adjudication_v1.md").write_text("\n".join(md))
    print("H1:", H1["verdict"])
    print("H2:", H2["verdict"])
    print("H-polyA:", H_POLYA["verdict"])
    print("H3:", H3["verdict"])
    print(f"wrote {OUT}/headline_adjudication_v1.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
