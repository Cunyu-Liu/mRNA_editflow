#!/usr/bin/env python3
"""Baseline Task 10.4 (partial): R1-R11 point-by-point closure self-audit.

The paper-skeleton rewrite itself is user-facing authorship work; the
requirement's auditable core — "R1-R11 逐条闭环自查" — compiles the evidence
map per reviewer attack from on-file terminal products. Table skeleton
material pointers included per table.

Output: experiments/analysis_task8_bottomline_20260909/r1_r11_closure_audit_v1.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_task8_bottomline_20260909")

AUDIT = [
    {
        "id": "R1", "attack": "baseline 调参不足",
        "closure": "CLOSED",
        "evidence": "frozen zero-shot/delta + matched-FT 双模式全表：MRL 双臂对称退化（Optimus 0.2977 < frozen 0.3132 / FramePool 0.2300 < 0.2956，批次六十）+ MPRAU matched-FT 3-seed 坍塌带（RNA-FM −0.0747×3 / UTR-LM −0.08~−0.11，批次六十二）——外部获得对等微调机会且退化；超参来源逐行声明（protocol v1 commit 7dc3dd98 §R1 条款）",
    },
    {
        "id": "R2", "attack": "多任务 vs 单任务不公平",
        "closure": "CLOSED (LOSO-lite 表 pending user 拍板 A/B 裁决)",
        "evidence": "per-task 对位为主表（Task 8.1 终判表 9 任务逐行）；公平预算协议四条入协议（对位预算/预训练等价性/≥3seeds+Holm/closed-form 主指标）；多任务增益单列 = Table 5 LOSO-lite（11.1 登记：协议解释 A/B 待用户裁决，素材已备）",
    },
    {
        "id": "R3", "attack": "划分泄漏",
        "closure": "CLOSED（制度化）",
        "evidence": "3-block 鸽笼审计全数据集执行 + 两个外部模型被审计排除的先例（UTR-STCNet 08ff6c2f 三臂 INVALID / UTR-Insight b18d686e INVALID——均训练于 GSE114002 同研究源）；LAMAR R3 PASS 零重叠先例；S1/M6 五研究 flagged=0；UTR-STCNet/CMS 事件制度化为新候选标准流程（spec Requirement）",
    },
    {
        "id": "R4", "attack": "指标/天花板归一化自创",
        "closure": "CLOSED",
        "evidence": "Spearman+决策口径（top-1/NDCG）双报（polyA 混合结果如实：Spearman 胜 top-1 负）；ICC/Spearman-Brown 天花板推导入档（analysis_ceiling_icc_20260907）；NDCG K 敏感性 29 行全稳定 + V5 行与榜单精确对齐（P0-4，R4 闭合批次五十三）；sc-hit@1 新口径走 amendment v1 流程留痕（5074f8c2）",
    },
    {
        "id": "R5", "attack": "MPRAU 无外部对照",
        "closure": "CLOSED（结构性空白主张实证化）",
        "evidence": "四模式闭环：frozen ≈0.01 / matched-FT 负带 3-seed / CMS 先验三臂 FAIL / LLR 双模型 ≈0——外部通用模型无 MPRAU 域信号从'推断'升级为'多模式实证'；Saluki 弱对照 0.1205 如实标注口径差异；s_mprau_in 0.1351 in-house 行 CI 跨零边际如实",
    },
    {
        "id": "R6", "attack": "复现可信度",
        "closure": "CLOSED",
        "evidence": "Track A 五模型 native 对齐五态登记（全 Partial 如实——native 绝对活性 vs frozen-Δ 承接对位）；Saluki 移植数值 parity spearman=1.0（8e871062）；RNA-FM matched-FT 三 seed 逐位一致（复现性发现，批次六十二）；λ=1.0 与首训精确一致（V6 复现性交叉验证）",
    },
    {
        "id": "R7", "attack": "统计",
        "closure": "CLOSED",
        "evidence": "source-group paired bootstrap 95% CI 全表 + Holm step-down（Task 12.1：polyA 唯一显著 p=0.004，exact bootstrap MRL/MPRAU 双平局如实）+ 小样本 power 声明成文（Task 12.2：8 对位 MDE 表 + directional-only 标准条款）+ ≥3 seeds：MRL 双 3-seed / MPRAU 5-seed / matched-FT 3-seed / V9-1a 3-seed 全达标；V5 冻结终态单次训练如实标注",
    },
    {
        "id": "R8", "attack": "增量在哪",
        "closure": "CLOSED（素材就位）",
        "evidence": "Table 1 四轴定位差异表完成（11.4：14 竞品谱系 + source-relative 形式化/泄漏控制划分/closed-form 设计评测/天花板归一化四轴）——vs Sample 2019 / UTailoR / FunUV / RNAGenScape 定位差异全部数字化",
    },
    {
        "id": "R9", "attack": "可用性",
        "closure": "PARTIAL（rights review 为治理动作）",
        "evidence": "converter + accession 指针 + 环境 lock 就位（Task 11.2 DATA_INVENTORY 增补备好）；14-study rights review 启动清单已备（11.3）但具名 owner 为项目侧治理动作待指定——如实登记为唯一开放项",
    },
    {
        "id": "R10", "attack": "mRNABERT 独立性",
        "closure": "CLOSED",
        "evidence": "冻结 bottom encoder 地位明示：V8/V9 全系冻结 Stage 1 backbone（33.0M 可训练 = LoRA+heads，backbone 参数零更新）；W0 从头训对照 0.1987（架构可提取但天花板在先验）入档——mRNABERT 贡献与任务适配贡献可分离",
    },
    {
        "id": "R11", "attack": "无 test 结果",
        "closure": "CLOSED（纪律维持）",
        "evidence": "protected reads=0 全程执行（每批次 journal 纪律条款复核）；TEST 开启 = Gate P 显式决策点（10.3 待用户拍板——判定基础 headline_adjudication_v1 已就位）；无任何 TEST 接触记录",
    },
]

TABLE_MAP = {
    "Table 1": "task11_deliverables_local.md §二（四轴定位差异表，14 竞品）",
    "Table 2": "benchmark 统计 = 14 研究身份核验（spec 已核验事实基础节）+ 划分/泄漏审计（R3 证据链）",
    "Table 3": "analysis_task8_bottomline_20260909/bottomline_adjudication_v1（9 任务 + 宏观 + CI + Holm 标注）",
    "Table 4": "closed_ndcg_ranking_v1 + table4_generation_closed_form_v1 + C2 β sweep（891 全量确认终态后回填 guided 行）",
    "Table 5": "LOSO-lite（协议 A/B 裁决待用户；lite 拼表素材半小时可出）",
    "Table 6": "headline_adjudication_v1 H3 节（五带可学性地图 + 三位一体归因链）",
    "图": "数据谱系图（Task 2 协议）/ 失败几何 P1–P4（spec N 段）/ H2 退化分析图（matched-FT 对称退化 + 口径分叉）",
}


def main() -> int:
    closed = sum(1 for a in AUDIT if a["closure"].startswith("CLOSED"))
    payload = {
        "schema_version": "route_a_v3_r1_r11_closure_audit_v1",
        "date": "2026-09-09",
        "summary": f"{closed}/11 CLOSED；R2 的 LOSO-lite 表与 R9 rights-review owner 为仅存开放点（均为用户/治理侧动作，非方法学缺口）",
        "audit": AUDIT,
        "table_material_map": TABLE_MAP,
    }
    (OUT / "r1_r11_closure_audit_v1.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    md = ["# R1-R11 逐条闭环自查（Task 10.4 审计核心）", "",
          f"**{closed}/11 CLOSED**；开放点均为用户/治理侧动作。", "",
          "| # | 攻击 | 状态 | 闭环证据 |", "|---|---|---|---|"]
    for a in AUDIT:
        md.append(f"| {a['id']} | {a['attack']} | {a['closure']} | {a['evidence']} |")
    md += ["", "## 论文表格素材映射", ""]
    for t, src in TABLE_MAP.items():
        md.append(f"- **{t}**: {src}")
    (OUT / "r1_r11_closure_audit_v1.md").write_text("\n".join(md))
    print(f"{closed}/11 CLOSED")
    for a in AUDIT:
        print(f"{a['id']}: {a['closure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
