#!/usr/bin/env python3
"""V7 formal adjudication on the preregistered MPRAU main criterion.

V7 = gradient-norm scaling mechanism (W4 line). The run_summary's
extended pair_mean_spearman is the ALL-TASK POOLED value and is NOT the
preregistered criterion (V6-H3 lesson, recorded 2026-09-03). This script
uses the identical caliber as the V6-H3/W-ladder adjudication: MPRAU variant
pair-mean rho (variants = rid before ":context:", >=2 contexts, per-variant
means of direction_normalized_delta vs prediction), paired bootstrap over
variants vs V5 (2,000 iters, seed 20260816), gate = CI excluding zero above
V5 0.1025. Pure prediction-file statistics.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
MAN = MNT + "/manifests/route2_development_frozen_v1/development_manifest.jsonl"
OUT = Path(MNT + "/experiments/analysis_v7_adjudication_20260903/results.json")
BOOT_ITERS = 2000
BOOT_SEED = 20260816


def manifest_ids(study, split="VALIDATION"):
    ids = set()
    with open(MAN) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split and r["study_unit_id"] == study:
                ids.add(str(r["canonical_record_id"]))
    return ids


def load_critic_predictions(path):
    preds, rows = {}, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            preds[str(r["record_id"])] = float(r["prediction"])
            rows[str(r["record_id"])] = r
    return preds, rows


def variant_pair_table(rows_by_id, study="ENCSR854RUF"):
    ids = manifest_ids(study)
    by_variant = defaultdict(list)
    for rid, row in rows_by_id.items():
        if rid in ids and rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(row)
    variants = {}
    for variant, rs in by_variant.items():
        if len(rs) >= 2:
            variants[variant] = (
                float(np.mean([r["target"] for r in rs])),
                float(np.mean([r["prediction"] for r in rs])),
            )
    return variants


def paired_bootstrap(v5_variants, arm_variants):
    shared = sorted(set(v5_variants) & set(arm_variants))
    t = np.array([v5_variants[v][0] for v in shared])
    p5 = np.array([v5_variants[v][1] for v in shared])
    pa = np.array([arm_variants[v][1] for v in shared])
    rho5 = spearmanr(t, p5).statistic
    rhoa = spearmanr(t, pa).statistic
    rng = np.random.default_rng(BOOT_SEED)
    n = len(shared)
    deltas = []
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, n)
        try:
            r5 = spearmanr(t[idx], p5[idx]).statistic
            ra = spearmanr(t[idx], pa[idx]).statistic
            if np.isfinite(r5) and np.isfinite(ra):
                deltas.append(ra - r5)
        except Exception:
            continue
    deltas = np.asarray(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    return {
        "shared_variant_count": n,
        "v5_pair_mean_spearman": float(rho5),
        "arm_pair_mean_spearman": float(rhoa),
        "delta_pair_mean_spearman": float(rhoa) - float(rho5),
        "bootstrap_ci_95": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "gate_pass_beats_v5_ci_excludes_zero": bool(ci[0] > 0),
        "ceiling_ratio_0_683": float(rhoa) / 0.683,
    }


def main() -> int:
    v5_path = glob.glob(MNT + "/experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")[0]
    v7_path = MNT + "/experiments/xeditcritic_v7/v7_screen_seed_20260907_runner_10965037/v7_full/final_validation_predictions.jsonl"
    _, v5rows = load_critic_predictions(v5_path)
    _, v7rows = load_critic_predictions(v7_path)
    v5_variants = variant_pair_table(v5rows)
    v7_variants = variant_pair_table(v7rows)
    result = {
        "schema_version": "route_a_v3_v7_mprau_adjudication.v1",
        "caliber": "MPRAU variant pair-mean rho (identical to W-ladder/V6-H3 adjudication)",
        "pooled_pair_mean_warning": "run_summary pair_mean_spearman is all-task pooled and NOT the criterion",
        "v5_reference_path": v5_path,
        "v7_path": v7_path,
        "bootstrap_iterations": BOOT_ITERS,
        "bootstrap_seed": BOOT_SEED,
        "v7_adjudication": paired_bootstrap(v5_variants, v7_variants),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
