#!/usr/bin/env python3
"""MEF edit-budget adjudication v1 (2026-10-10). Frozen per edit_budget_mini_prereg_v1.md.

Reads the 10 per-(family,task) result JSONs and emits the prereg-defined
verdict table: model x {polyA, MRL} with median steps (pass set), FAIL_budget
rate, budget ratio median + IQR, k_true contrast. CSV + JSON dual archive.
Every number is copied verbatim from the per-run summaries (no recomputation).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

D = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_edit_budget_v1")
FAMILIES = ["gemorna", "lamar_utr5te", "utr_insight", "utr_stcnet", "hydrarna"]
TASKS = ["polya", "mrl"]

EXPECTED = {
    ("polya"): 2628,
    ("mrl"): 730,
}


def main() -> int:
    rows = []
    per_file = {}
    missing = []
    for fam in FAMILIES:
        for task in TASKS:
            p = D / f"edit_budget_{fam}_{task}_v1.json"
            if not p.exists():
                missing.append(str(p.name))
                continue
            payload = json.load(open(p))
            s = payload["summary"]
            per_file[p.name] = s
            assert s["n"] == EXPECTED[task], f"{p.name}: n={s['n']}"
            assert s["b_max"] == (23 if task == "polya" else 3), f"{p.name}: b_max"
            assert s["cpu_fallback_used"] is False, f"{p.name}: cpu fallback!"
            rows.append({
                "family": fam,
                "task": task,
                "n": s["n"],
                "n_pass": s["n_pass"],
                "fail_budget_rate": s["fail_budget_rate"],
                "median_steps_pass": s["median_steps_pass"],
                "mean_steps_pass": s["mean_steps_pass"],
                "budget_ratio_median": s["median_budget_ratio"],
                "budget_ratio_q1": s["budget_ratio_q1"],
                "budget_ratio_q3": s["budget_ratio_q3"],
                "elapsed_s": s["elapsed_s"],
                "b_max": s["b_max"],
            })
    if missing:
        print("MISSING (incomplete run set):", missing)
        verdict = {"status": "INCOMPLETE", "missing": missing, "rows": rows}
        json.dump(verdict, open(D / "edit_budget_adjudication_v1.json", "w"),
                  ensure_ascii=False, indent=1)
        print(f"wrote INCOMPLETE adjudication with {len(rows)} rows")
        return 1

    # k_true references (prereg-frozen): polya median 7 / mrl median 1
    k_ref = {"polya": 7, "mrl": 1}
    for r in rows:
        r["k_true_median_ref"] = k_ref[r["task"]]

    csv_path = D / "edit_budget_adjudication_v1.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdict = {
        "status": "COMPLETE",
        "prereg": "docs/paper/edit_budget_mini_prereg_v1.md (commit e26ac88a)",
        "protocol": "greedy full-enumeration, frozen scorers, tie-break lexicographic; B_max polya=23/mrl=3",
        "n_runs": len(rows),
        "rows": rows,
        "notes": [
            "median_steps_pass computed over passing records only (prereg §2)",
            "budget_ratio = k_model/k_true per record, median + IQR over passing records with k_true>0",
            "cpu_fallback_used=false asserted per run (CUDA hard gate)",
        ],
    }
    json.dump(verdict, open(D / "edit_budget_adjudication_v1.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"COMPLETE: {len(rows)} runs adjudicated -> {csv_path}")
    for r in rows:
        print("%-14s %-6s fail=%-7s med_steps=%-5s ratio_med=%s" % (
            r["family"], r["task"], r["fail_budget_rate"],
            r["median_steps_pass"], r["budget_ratio_median"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
