#!/usr/bin/env python3
"""V8 Stage 2 adjudicator (frozen prereg route2_v8_stage2_prereg_v1.md §3).

Reads terminal run_report.json (FINAL-EPOCH-FIXED only; smoke/proxy/train-set
results are hard-rejected) for the Stage 2 arms and applies the frozen gates:

- MPRAU (primary): variant pair-mean rho > 0.1025 AND bootstrap CI not
  crossing zero (vs V5 0.1025, paired bootstrap 2,000 iters seed 20260816).
- MRL >= 0.28 (GSE114002 VALIDATION task_macro_spearman, frozen-delta K=10).
- polyA >= 0.80 (GSE269595 VALIDATION task_macro_spearman).
- TE family >= 0.1317 (GSE217518 + GSE256185 VALIDATION macro).
- task macro significant improvement vs 0.167 (V5 reference, paired bootstrap
  not recomputed here; the point estimate comparison is recorded with a
  caveat that the formal paired bootstrap runs in the harvest step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/v8_stage2_adapt_20260907")

ARM_ORDER = ["h_cms_full", "s_cms_full", "h_cms_lora", "h_bench9", "h_mprau_in",
             "s_mprau_in", "h_mprau_lora", "s_mprau_in_s2", "s_mprau_in_s3",
             "h_mprau_lora_s2", "h_mprau_in_s2"]
GATES = {
    "mprau": {"ref": 0.1025, "op": "gt_ci", "label": "MPRAU pair-mean > 0.1025 CI not cross zero"},
    "mrl": {"ref": 0.28, "op": "ge", "label": "MRL >= 0.28"},
    "polya": {"ref": 0.80, "op": "ge", "label": "polyA >= 0.80"},
    "te": {"ref": 0.1317, "op": "ge", "label": "TE family >= 0.1317"},
    "macro": {"ref": 0.167, "op": "ge", "label": "task macro >= 0.167 (V5 ref; formal bootstrap at harvest)"},
}


def load_arm_report(arm: str) -> dict | None:
    report = OUT_ROOT / arm / "run_report.json"
    if not report.is_file():
        return None
    with report.open() as handle:
        return json.load(handle)


def arm_extract(report: dict) -> dict:
    """Flatten per-arm judgment numbers from the terminal eval_final."""
    eval_final = report.get("eval_final") or {}
    mprau = eval_final.get("mprau") or {}
    vs_v5 = mprau.get("vs_v5") or {}
    per_task = eval_final.get("per_task") or []
    by_study = {r["study"]: r for r in per_task}
    mrl = (by_study.get("GSE114002") or {}).get("task_macro_spearman")
    polya = (by_study.get("GSE269595") or {}).get("task_macro_spearman")
    te_vals = []
    for study in ("GSE217518", "GSE256185"):
        v = (by_study.get(study) or {}).get("task_macro_spearman")
        if v is not None:
            te_vals.append(float(v))
    te = float(sum(te_vals) / len(te_vals)) if te_vals else None
    return {
        "mode": report.get("mode"),
        "adapt_mode": report.get("adapt_mode"),
        "arch": report.get("arch"),
        "selection_rule": report.get("selection_rule"),
        "steps_done": report.get("steps_done"),
        "epochs_completed": report.get("epochs_completed"),
        "smoke": report.get("smoke"),
        "cpu_fallback_used": report.get("cpu_fallback_used"),
        "mprau_pair_mean": mprau.get("pair_mean_spearman"),
        "mprau_n_variants": mprau.get("n_variants"),
        "mprau_delta_ci95": (vs_v5.get("delta_ci95") if vs_v5 else None),
        "mprau_delta_crosses_zero": (vs_v5.get("crosses_zero") if vs_v5 else None),
        "mrl_spearman": mrl,
        "polya_spearman": polya,
        "te_spearman": te,
        "task_macro": eval_final.get("task_macro_spearman"),
    }


def judge_mprau_only(ext: dict) -> dict:
    """MPRAU primary gate only (specialist arms)."""
    m = ext.get("mprau_pair_mean")
    ci = ext.get("mprau_delta_ci95")
    return {
        "pass": m is not None and float(m) > GATES["mprau"]["ref"]
                and not bool(ext.get("mprau_delta_crosses_zero")),
        "value": m, "ci95": ci, "criterion": GATES["mprau"]["label"],
    }


def judge(ext: dict) -> dict:
    gates = {}
    m = ext.get("mprau_pair_mean")
    ci = ext.get("mprau_delta_ci95")
    gates["mprau"] = {
        "pass": m is not None and float(m) > GATES["mprau"]["ref"] and not bool(ext.get("mprau_delta_crosses_zero")),
        "value": m, "ci95": ci, "criterion": GATES["mprau"]["label"],
    }
    for key in ("mrl", "polya", "te", "macro"):
        v = ext.get("mrl_spearman" if key == "mrl" else "polya_spearman" if key == "polya"
                    else "te_spearman" if key == "te" else "task_macro")
        gates[key] = {
            "pass": v is not None and float(v) >= GATES[key]["ref"],
            "value": v, "criterion": GATES[key]["label"],
        }
    return {"all_gates_pass": all(g["pass"] for g in gates.values()), "gates": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    root = Path(args.out_root)
    arms = {}
    for arm in ARM_ORDER:
        report = load_arm_report(arm)
        if report is None:
            arms[arm] = {"status": "NOT_TERMINAL"}
            continue
        if report.get("smoke"):
            arms[arm] = {"status": "REJECTED_SMOKE", "reason": "smoke run cannot be adjudicated"}
            continue
        if report.get("selection_rule") != "FINAL_EPOCH_FIXED":
            arms[arm] = {"status": "REJECTED_SELECTION", "reason": report.get("selection_rule")}
            continue
        ext = arm_extract(report)
        # Specialist arms (MPRAU adaptation, single-study) are judged on the
        # MPRAU primary gate only - they intentionally sacrifice other tasks.
        # The multi-task arm (h_bench9) is judged on all gates.
        if arm == "h_bench9":
            verdict = judge(ext)
        else:
            verdict = {"all_gates_pass": None, "mprau_primary": judge_mprau_only(ext),
                       "gates": {"mprau": judge(ext)["gates"]["mprau"]}}
        arms[arm] = {"status": "TERMINAL", "numbers": ext, **verdict}
    result = {
        "schema_version": "route_a_v3_route2_v8_stage2_adjudication.v1",
        "adjudicated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "prereg": "docs/paper/route2_v8_stage2_prereg_v1.md",
        "gates": GATES,
        "arms": arms,
    }
    out = root / "adjudication_v8_stage2.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({"arms": {k: v.get("status") for k, v in arms.items()},
                      "gate_pass": {k: (v.get("all_gates_pass") if v.get("status") == "TERMINAL" else None)
                                    for k, v in arms.items()}}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
