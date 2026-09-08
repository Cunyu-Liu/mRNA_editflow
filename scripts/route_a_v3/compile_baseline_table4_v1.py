#!/usr/bin/env python3
"""Baseline P0-3: generation-line closed-form leaderboard (Table 4 v1).

Compiles the frozen terminal adjudication products of the generation line
into one table (SPECS_BASELINE_LEADERBOARD spec 2026-09-08 增补 §V.4 P0-3):

  rows: unguided base (B2 2026-09-03 run) / V5-critic guided / V8-S specialist
        guided / V8-S joint guided / TreeG sample-then-select
  metrics (closed-form, frozen evaluator): candidate_recovery_rate@K=10,
        measured top-K recovery@K=10, hit@1, legality, budget violations;
        evaluator uplift NOT included (STOP-05: auxiliary only, and not
        present in these products).

Known gap (registered): closed NDCG@K / regret require an offline scoring
pass over generated_candidates.private.jsonl (generation_score per candidate
exists) + measured-neighbourhood mapping -- registered as P0-3b follow-up;
v1 ships the adjudicated closed-form fields.

pool256 (Phase C option 4a) is in flight at compile time -> excluded from v1
(not terminal; will be appended as a B=256 coverage row by Phase C).
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments")
OUT = MNT / "analysis_baseline_table4_20260908"

SOURCES = {
    "b2_20260903": MNT / "xeditsetflow_v5/guided_b2_20260903/b2_full_891_adjudication.json",
    "b2_v8_specialist_20260908": MNT / "xeditsetflow_v5/guided_b2_v8_20260908/b2_full_891_adjudication.json",
    "b2_v8_joint_20260908": MNT / "xeditsetflow_v5/guided_b2_v8joint_20260908/b2_full_891_adjudication.json",
    "treeg_20260905": MNT / "treeg_sample_select_20260905/full/result_full891.json",
}


def arm_row(arm: dict) -> dict:
    return {
        "recovery_at_k10": arm.get("source_macro_candidate_recovery_rate"),
        "measured_top_k_recovery_at_k10": arm.get("source_macro_measured_top_k_recovery_at_k"),
        "hit_at_1": arm.get("hit_at_1"),
        "top1_generated_is_measured": arm.get("top1_generated_is_measured"),
        "hard_legality_rate": arm.get("hard_legality_rate"),
        "edit_budget_violations": arm.get("edit_budget_violation_count"),
        "candidate_budget_violations": arm.get("candidate_budget_violation_count"),
        "method_id": arm.get("method_id"),
    }


def with_legality(row: dict, legality_arm: dict) -> dict:
    row = dict(row)
    for k in ("hard_legality_rate", "edit_budget_violation_count", "candidate_budget_violation_count"):
        if k in legality_arm:
            row[k] = legality_arm[k]
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = {"schema_version": "route_a_v3_baseline_table4_generation_v1",
             "split": "VALIDATION", "sources": 891, "candidate_cap_per_source": 32,
             "k": 10, "bootstrap": "source-group paired cluster, 2000 iters, seed 20260816/20260817",
             "notes": [
                 "closed-form adjudicated fields only; evaluator uplift excluded (STOP-05)",
                 "closed NDCG@K / regret offline scoring pass registered as P0-3b follow-up "
                 "(generation_score present in candidates files)",
                 "pool256 (Phase C 4a) in flight at compile time -> excluded from v1",
             ],
             "rows": {}, "gates": {}}

    # B2 2026-09-03: unguided + V5-guided (metrics dict LAST so real values win the union)
    r = json.load(open(SOURCES["b2_20260903"]))
    table["rows"]["unguided_base_b_fix2_pass2"] = with_legality(arm_row(r["unguided"]), r["legality"]["unguided"])
    table["rows"]["v5_critic_guided"] = with_legality(arm_row(r["guided"]), r["legality"]["guided"])
    table["gates"]["b2_20260903"] = {
        "gate_b2_passed": r.get("gate_b2_passed"),
        "gate_b3_passed": r.get("gate_b3_passed"),
        "delta_recovery_ci95": [r["delta_recovery"]["ci_low"], r["delta_recovery"]["ci_high"]],
        "delta_hit_at_1_ci95": [r["delta_hit_at_1"]["ci_low"], r["delta_hit_at_1"]["ci_high"]],
    }
    # V8 arms
    for key, name in (("b2_v8_specialist_20260908", "v8_specialist_guided"),
                      ("b2_v8_joint_20260908", "v8_joint_guided")):
        r = json.load(open(SOURCES[key]))
        table["rows"][name] = with_legality(arm_row(r["guided"]), r["legality"]["guided"])
        table["gates"][key] = {
            "gate_b2_passed": r.get("gate_b2_passed"),
            "gate_b3_passed": r.get("gate_b3_passed"),
            "delta_recovery_ci95": [r["delta_recovery"]["ci_low"], r["delta_recovery"]["ci_high"]],
            "delta_hit_at_1_ci95": [r["delta_hit_at_1"]["ci_low"], r["delta_hit_at_1"]["ci_high"]],
        }
    # TreeG
    r = json.load(open(SOURCES["treeg_20260905"]))
    table["rows"]["treeg_sample_then_select"] = {
        "hit_at_1": r.get("treeg", {}).get("hit_at_1") if isinstance(r.get("treeg"), dict) else None,
        "select_k": r.get("select_k"),
        "pool_candidate_count": r.get("pool_candidate_count"),
        "delta_hit_at_1_vs_unguided": r.get("delta_hit_at_1_vs_unguided"),
        "delta_recovery_vs_unguided": r.get("delta_recovery_vs_unguided"),
        "caveat": r.get("caveat"),
    }
    table["gates"]["treeg_20260905"] = {"status": r.get("status"),
                                        "verdict": "备选证伪 (delta_hit_at_1 CI 全负, 见 spec)"}

    # markdown rendering (paper skeleton Table 4 draft)
    md = ["# Table 4 (v1): Generation-line closed-form leaderboard (VALIDATION, 891 sources, K=10)",
          "",
          "| method | recovery@10 | top-K rec@10 | hit@1 | legality | budget viol | Gate B2 | Gate B3 |",
          "|---|---|---|---|---|---|---|---|"]
    gate_by_row = {
        "unguided_base_b_fix2_pass2": ("—(基线)", "—"),
        "v5_critic_guided": ("FAIL", "FAIL"),
        "v8_specialist_guided": ("FAIL", "FAIL"),
        "v8_joint_guided": ("FAIL", "FAIL"),
        "treeg_sample_then_select": ("证伪", "—"),
    }
    for name, row in table["rows"].items():
        if name == "treeg_sample_then_select":
            md.append("| %s | — | — | %s | 1.0* | 0* | %s | %s |" % (
                name, row.get("hit_at_1"), gate_by_row[name][0], gate_by_row[name][1]))
            continue
        md.append("| %s | %.5f | %.5f | %.5f | %s | %s | %s | %s |" % (
            name, row["recovery_at_k10"] or 0, row["measured_top_k_recovery_at_k10"] or 0,
            row["hit_at_1"] or 0, row["hard_legality_rate"],
            int(row["edit_budget_violations"] or 0) + int(row["candidate_budget_violations"] or 0),
            gate_by_row[name][0], gate_by_row[name][1]))
    md += ["", "* TreeG legality/budget 继承候选池（legal edits only）；hit@1 = critic top-1 命中率（0.0045）。",
           "", "缺口登记：closed NDCG@K / regret 需离线打分通道（P0-3b）；pool256 在途。"]
    (OUT / "table4_generation_closed_form_v1.json").write_text(json.dumps(table, indent=1))
    (OUT / "table4_generation_closed_form_v1.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
