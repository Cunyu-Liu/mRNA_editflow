#!/usr/bin/env python3
"""V8 Stage 1 terminal adjudication (prereg route2_v8_stage1_prereg_v1.md §8).

Judgment gates (registered, no peak-picking; FINAL-EPOCH-FIXED primary records only):
1. Non-destruction per domain: joint arm zero-shot >= 0.9 * single-domain arm.
   - MRL:  S/H >= 0.9 * M-arm 3-seed mean (0.3076) = 0.2768.
   - polyA: S/H >= 0.9 * polyA-only baseline (s_polya run_report primary).
2. S vs H adjudication (MRL): choose S if S >= H - 0.02, else H.
3. Stage 1 success: >=1 arm passes non-destruction (both domains) + adjudication done.
4. Failure: neither arm passes -> "joint prior injection failed", Stage 2 not launched.

Outputs: adjudication_v8_stage1.json (verdict + per-arm per-domain table).
smoke runs (run_report.smoke=true) are rejected as non-terminal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V8_OUT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/v8_stage1_joint_prefinetune_20260904")
SEED_RESULTS = {
    20260903: Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/frozen_delta_results.json"),
    20260904: Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260904/frozen_delta_results.json"),
    20260905: Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260905/frozen_delta_results.json"),
}
M_ARM_REGISTERED_MEAN = 0.3076  # prereg §2 (3-seed mean of fullft_v2 ep6)
NON_DESTRUCTION_FACTOR = 0.9
S_VS_H_TOLERANCE = 0.02


def load_report(arch: str, libs: str) -> dict | None:
    path = V8_OUT / f"{arch}_{libs}" / "run_report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("smoke"):
        raise SystemExit(f"FATAL: {path} is a SMOKE run (non-terminal) - refuse to adjudicate")
    return report


def primary_metrics(report: dict, domain: str) -> dict | None:
    for rec in report.get("zeroshot_final", []):
        if rec.get("primary") and rec.get("domain") == domain:
            return rec
    return None


def m_arm_mean() -> float:
    vals = []
    for path in SEED_RESULTS.values():
        if path.exists():
            d = json.loads(path.read_text())
            m = (d.get("metrics") or {}).get("task_macro_spearman")
            if m is not None:
                vals.append(m)
    if len(vals) == 3:
        return sum(vals) / len(vals)
    return M_ARM_REGISTERED_MEAN  # fallback to registered constant


def main() -> int:
    s = load_report("s", "mrl-polya")
    h = load_report("h", "mrl-polya")
    p = load_report("s", "polya")  # polyA-only baseline (may be pending)

    m_mean = m_arm_mean()
    mrl_threshold = NON_DESTRUCTION_FACTOR * m_mean

    verdict = {
        "schema_version": "route_a_v3_route2_v8_stage1_adjudication.v1",
        "prereg": "docs/paper/route2_v8_stage1_prereg_v1.md §8",
        "m_arm_mean": m_mean,
        "mrl_non_destruction_threshold": mrl_threshold,
        "arms": {},
        "s_vs_h": None,
        "stage1_success": None,
        "polyA_gate_status": "PENDING" if p is None else "RESOLVED",
    }

    for arch, rep in (("s", s), ("h", h)):
        if rep is None:
            verdict["arms"][arch] = {"terminal": False}
            continue
        mrl_m = primary_metrics(rep, "mrl")
        polya_m = primary_metrics(rep, "polya")
        arm = {"terminal": True, "mrl": mrl_m, "polya": polya_m}
        arm["mrl_non_destruction_pass"] = (
            mrl_m is not None and mrl_m["task_macro_spearman"] >= mrl_threshold
        )
        if p is not None:
            p_m = primary_metrics(p, "polya")
            p_thr = NON_DESTRUCTION_FACTOR * (p_m["task_macro_spearman"] if p_m else 0.0)
            arm["polya_baseline_spearman"] = p_m["task_macro_spearman"] if p_m else None
            arm["polya_non_destruction_threshold"] = p_thr
            arm["polya_non_destruction_pass"] = (
                polya_m is not None and p_m is not None
                and polya_m["task_macro_spearman"] >= p_thr
            )
        else:
            arm["polya_non_destruction_pass"] = None
        verdict["arms"][arch] = arm

    if s is not None and h is not None:
        s_mrl = verdict["arms"]["s"].get("mrl")
        h_mrl = verdict["arms"]["h"].get("mrl")
        if s_mrl and h_mrl:
            delta = s_mrl["task_macro_spearman"] - h_mrl["task_macro_spearman"]
            verdict["s_vs_h"] = {
                "rule": "choose S if S >= H - 0.02 on MRL task_macro_spearman",
                "delta_s_minus_h": delta,
                "winner": "S" if delta >= -S_VS_H_TOLERANCE else "H",
            }
        passed = [a for a, v in verdict["arms"].items()
                  if v.get("terminal") and v.get("mrl_non_destruction_pass")]
        if p is None:
            # polyA gate unresolved; mark provisional success only if >=1 arm
            # already clears MRL; polyA gate completes after baseline lands.
            verdict["stage1_success"] = "PROVISIONAL_MRL_ONLY" if passed else False
        else:
            full_pass = [a for a, v in verdict["arms"].items()
                         if v.get("mrl_non_destruction_pass") and v.get("polya_non_destruction_pass")]
            verdict["stage1_success"] = bool(full_pass)

    out = V8_OUT / "adjudication_v8_stage1.json"
    out.write_text(json.dumps(verdict, indent=1, sort_keys=True))
    print(json.dumps(verdict, indent=1, sort_keys=True))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
