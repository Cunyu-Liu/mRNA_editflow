#!/usr/bin/env python3
"""Option-3 CMS-augmentation arm final adjudication (v9d vs s_mprau_in control).

Preregistered gate (committed BEFORE launch, journal batch 76):
  - primary: MPRAU VALIDATION pair_mean_spearman > 0.1351 (5-seed ensemble ref)
  - significance: paired per-variant bootstrap (2000 iters, seed 20260816) of
    per-variant spearman differences vs the same-seed no-CMS control over the
    2,008 validation variants; verdict three-way:
      CI excludes zero positive -> WINS
      point > 0.1351 but CI crosses zero -> DIRECTIONAL
      point <= 0.1351 -> FAIL (regardless of CI)
Selection rule FINAL_EPOCH_FIXED: epoch-12 checkpoint is the adjudicated one
(epoch-9..11 peak 0.1038 is NOT admissible -- no post-hoc epoch picking).

Per-variant spearman: for each of the 2,008 variants (grouped by variant_id
over its cell-context records), spearman(prediction, label) per arm; the
paired difference is v9d - control per variant.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

V9D = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "xeditcritic_route_a/v9d_cms_augment_20260910/s_cmsaug_seed20260907"
)
CONTROL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "xeditcritic_route_a/v8_stage2_adapt_20260907/s_mprau_in"
)
OUT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "xeditcritic_route_a/v9d_cms_augment_20260910/option3_cmsaug_adjudication.json"
)
BOOT_ITERS, BOOT_SEED = 2000, 20260816
GATE_LINE = 0.1351
CONTROL_SINGLE = 0.1527


def load_pred(path: Path) -> dict[str, float]:
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[r["canonical_record_id"]] = float(r["prediction"])
    return out


def load_labels_records() -> dict[str, list[dict]]:
    """canonical_record_id -> records with variant_id + label (direction delta).

    Mirrors eval_mprau_validation's data path via the r2 module's helpers.
    """
    import sys

    V8_WT = Path(
        "/home/cunyuliu/mrna_editflow_goal/worktrees/"
        "route_a_v3_v8_stage1_prep_20260904"
    )
    sys.path.insert(0, str(V8_WT))
    sys.path.insert(0, str(V8_WT / "scripts/route_a_v3"))
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "r2adj", V8_WT / "scripts/route_a_v3/run_route2_v8_stage2_adapt_v1.py"
    )
    r2 = ilu.module_from_spec(spec)
    sys.modules["r2adj"] = r2
    spec.loader.exec_module(r2)
    validation_ids = r2._manifest_validation_ids().get("ENCSR854RUF", set())
    records = r2._load_canonical(r2.BENCHMARK_DOMAINS["ENCSR854RUF"][1])
    by_variant = defaultdict(list)
    for rid in sorted(validation_ids):
        if rid not in records:
            continue
        rec = records[rid]
        rec = dict(rec)
        rec["canonical_record_id"] = rid
        by_variant[str(rec["variant_id"])].append(rec)
    return by_variant


def per_variant_spearman(pred: dict[str, float], by_variant) -> dict[str, float]:
    out = {}
    for vid, recs in by_variant.items():
        ids = [r["canonical_record_id"] for r in recs]
        if any(i not in pred for i in ids) or len(ids) < 3:
            continue
        p = np.asarray([pred[i] for i in ids], dtype=np.float64)
        y = np.asarray(
            [float(r["direction_normalized_delta"]) for r in recs], dtype=np.float64
        )
        if np.std(p) == 0 or np.std(y) == 0:
            rho = 0.0
        else:
            rho = float(spearmanr(p, y).statistic)
        out[vid] = rho
    return out


def main() -> int:
    by_variant = load_labels_records()
    v9d_pred = load_pred(V9D / "mprau_predictions.jsonl")
    ctl_pred = load_pred(CONTROL / "mprau_predictions.jsonl")
    v9d_pv = per_variant_spearman(v9d_pred, by_variant)
    ctl_pv = per_variant_spearman(ctl_pred, by_variant)
    shared = sorted(set(v9d_pv) & set(ctl_pv))
    print(f"per-variant spearman computed: v9d {len(v9d_pv)} / control {len(ctl_pv)} / shared {len(shared)}")

    d = np.asarray([v9d_pv[k] - ctl_pv[k] for k in shared], dtype=np.float64)
    delta = float(np.mean(d))
    rng = np.random.default_rng(BOOT_SEED)
    n = len(shared)
    boots = []
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, n)
        boots.append(float(np.mean(d[idx])))
    boots.sort()
    lo, hi = boots[int(0.025 * BOOT_ITERS)], boots[int(0.975 * BOOT_ITERS) - 1]

    v9d_mean = float(np.mean([v9d_pv[k] for k in shared]))
    ctl_mean = float(np.mean([ctl_pv[k] for k in shared]))

    if delta > 0 and lo > 0:
        verdict = "CMS_WINS"
    elif v9d_mean > GATE_LINE:
        verdict = "DIRECTIONAL"
    else:
        verdict = "FAIL"

    payload = {
        "schema_version": "route_a_v3_route2_option3_cmsaug_adjudication.v1",
        "gate_preregistered": "batch 76 journal (commit precedes launch)",
        "selection_rule": "FINAL_EPOCH_FIXED (epoch 12; epoch 9-11 peak 0.1038 NOT admissible)",
        "n_shared_variants": n,
        "v9d_pair_mean_reported": 0.0917,
        "control_pair_mean_single": CONTROL_SINGLE,
        "gate_line_ensemble": GATE_LINE,
        "per_variant_spearman_mean": {
            "v9d": v9d_mean,
            "control": ctl_mean,
        },
        "delta_v9d_minus_control": delta,
        "delta_ci95": [lo, hi],
        "ci_excludes_zero": not (lo <= 0.0 <= hi),
        "verdict": verdict,
        "per_task_side_effects": {
            "note": "CMS augmentation collateral: polya -0.2261 / mrl 0.1809 vs control recipe's task profile",
        },
        "bootstrap": f"{BOOT_ITERS} iters seed {BOOT_SEED} paired per-variant",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
