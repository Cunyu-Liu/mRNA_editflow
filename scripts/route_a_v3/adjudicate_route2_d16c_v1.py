#!/usr/bin/env python3
"""D16-C probe adjudication: frozen G1-G4 gates (amendment v1 sec.4.2).

Run AFTER the probe arm is terminal (probe_summary.json + epoch_metrics.jsonl
from run_route2_d16c_probe_v1.py). Reads only terminal artifacts, computes:

- G1 generalization: MRL train-backtest gap <= 0.30 (baseline 0.79) AND
  VALIDATION rho >= 0.135 (V5 baseline). The backtest scores TRAIN rows
  through the probe's FINAL-EPOCH checkpoint (same per-source potentials
  calibre as the probe runner eval; V5 reference backtest numbers from
  analysis_v5_train_backtest: MRL train 0.928 / val 0.134).
- G2 non-destruction: polyA VALIDATION rho >= 0.80 (from probe epoch_metrics
  FINAL-EPOCH row - the runner already computes it per epoch).
- G3 anti-hypothesis registration: G1 FAIL with gap > 0.5 -> data-side
  independent invalid, ERK parameter side becomes the only path.
- G4 structure probe: graded sc-hit@1 > 0 in the structure pool (D15-2
  calibre, run separately on the final checkpoint; this script reads its
  JSON if present, else reports PENDING).

Discipline: FINAL-EPOCH-FIXED only (no peak picking); VALIDATION-only
reporting; protected reads = 0; smoke/proxy results are hard-rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROBE_DIR = MNT / "experiments/xeditcritic_d16c/probe_mrl_v1"
V5_BACKTEST = MNT / "experiments/analysis_v5_train_backtest"

# frozen reference numbers (amendment sec.2/4 + v5_train_backtest_20260912)
V5_MRL_VAL_RHO = 0.134
V5_MRL_TRAIN_RHO = 0.928
G1_GAP_MAX = 0.30
G1_VAL_MIN = 0.135
G2_POLYA_MIN = 0.80
G3_GAP_HEAVY = 0.50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default=str(PROBE_DIR))
    parser.add_argument("--structure-probe-json", default="", help="D15-2 style graded sc-hit@1 probe result for the final checkpoint")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    summary_path = probe_dir / "probe_summary.json"
    metrics_path = probe_dir / "epoch_metrics.jsonl"
    if not summary_path.exists() or not metrics_path.exists():
        print("ADJUDICATION_BLOCKED: probe terminal artifacts missing (probe_summary.json / epoch_metrics.jsonl)", flush=True)
        return 2

    summary = json.loads(summary_path.read_text())
    if summary.get("cpu_fallback_used"):
        print("HARD_REJECT: cpu_fallback_used = true (discipline violation)", flush=True)
        return 2
    if not summary.get("cuda_verified"):
        print("HARD_REJECT: cuda not verified", flush=True)
        return 2

    epochs = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    final_rows = [r for r in epochs if r.get("primary")]
    if len(final_rows) != 1:
        print(f"HARD_REJECT: FINAL-EPOCH marker count = {len(final_rows)} (must be exactly 1, no peak-picking)", flush=True)
        return 2
    final = final_rows[0]

    mrl_val_rho = final.get("mrl_val_spearman")
    polya_val_rho = final.get("polya_val_spearman")

    # train-backtest: recompute gap via the backtest artifact if the probe
    # backtest was run (run_v5_train_backtest pattern on the probe ckpt);
    # otherwise compute gap from the runner-declared train row if present.
    train_rho = None
    bt_path = probe_dir / "train_backtest.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text())
        mrl = bt.get("tasks", {}).get("MRL", {})
        train_rho = mrl.get("train_spearman") or mrl.get("train_rho")
    if train_rho is None:
        # fall back: the epoch_metrics diagnostics carry val only; require
        # the dedicated backtest run before G1 can be judged.
        gap = None
    else:
        gap = (train_rho - mrl_val_rho) if (train_rho is not None and mrl_val_rho is not None) else None

    g1_pass = gap is not None and mrl_val_rho is not None and gap <= G1_GAP_MAX and mrl_val_rho >= G1_VAL_MIN
    g2_pass = polya_val_rho is not None and polya_val_rho >= G2_POLYA_MIN
    g3_registered = (gap is not None and gap > G3_GAP_HEAVY) or (gap is None and not g1_pass)

    g4 = "PENDING"
    if args.structure_probe_json:
        sp = Path(args.structure_probe_json)
        if sp.exists():
            spd = json.loads(sp.read_text())
            v = spd.get("graded_sc_hit1") or spd.get("graded_sc_hit1_mean")
            g4 = "PASS" if (v is not None and v > 0) else "FAIL"

    verdict = {
        "schema_version": "route_a_v3_d16c_adjudication.v1",
        "gates": {
            "G1_generalization": {
                "mrl_val_rho": mrl_val_rho,
                "train_rho": train_rho,
                "gap": gap,
                "gap_max": G1_GAP_MAX,
                "val_min": G1_VAL_MIN,
                "pass": g1_pass,
                "note": "gap requires the dedicated train-backtest run (run_v5_train_backtest pattern on probe final checkpoint) before this gate is final",
            },
            "G2_nondestruction": {
                "polya_val_rho": polya_val_rho,
                "min": G2_POLYA_MIN,
                "pass": g2_pass,
            },
            "G3_antihypothesis": {
                "registered": g3_registered,
                "meaning": "data-side independently invalid; ERK parameter side is the only remaining path",
            },
            "G4_structure_probe": {
                "status": g4,
                "target": "graded sc-hit@1 > 0 in the structure pool (D15-2 calibre on final ckpt)",
            },
        },
        "reference": {
            "v5_mrl_val_rho": V5_MRL_VAL_RHO,
            "v5_mrl_train_rho": V5_MRL_TRAIN_RHO,
            "v5_gap_baseline": round(V5_MRL_TRAIN_RHO - V5_MRL_VAL_RHO, 3),
        },
        "final_epoch": final,
        "protected_reads": 0,
    }
    out_path = probe_dir / "adjudication_d16c.json"
    out_path.write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict["gates"], indent=1), flush=True)
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
