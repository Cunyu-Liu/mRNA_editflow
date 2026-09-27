#!/usr/bin/env python3
"""M1 intervention arm 3-seed ensemble harvest v1 (activates with amendment v2).

Protocol (verbatim reuse of the frozen Route A V2 ensemble caliber,
`adjudicate_route2_fullft_v2_3seed_ensemble_v1.py`):
  - per-seed predictions -> per-seed z-score -> mean (PER_SEED_ZSCORE_MEAN);
  - frozen Task-1 evaluator (K=10) on GSE114002 VALIDATION (n=730);
  - paired source-group bootstrap vs the V2 3-seed ensemble predictions
    (2,000 iters, seed 20260816).

Arm seeds per amendment v2 (ACTIVE/FROZEN 2026-09-27T19:06Z; training pre-committed,
analysis activated only when G1 direction is positive):
  {20260920, 20260904, 20260905}; V2 baseline seeds: {20260903, 20260904, 20260905}.

Verification mode: `--standin-v2` uses the three V2 seed dirs as the "arm", which
must reproduce the archived ensemble value 0.3158289984824722 exactly (delta 0.0).

Exit codes: 0 = rendered; 3 = arm seeds not all terminal yet.
Read-only: writes only under the arm adjudication directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL_GSE114002 = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
ARM_ROOT = MNT / "experiments/xeditcritic_m1_intervention"
DEFAULT_OUT = ARM_ROOT / "adjudication_v1"
V2_DIRS = {
    20260903: MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903",
    20260904: MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260904",
    20260905: MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260905",
}
ARM_SEEDS = (20260920, 20260904, 20260905)
V2_ENSEMBLE_ARCHIVED = 0.3158289984824722
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_preds(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("canonical_record_id") or row.get("record_id"))
            out[key] = float(row.get("predicted_direction_normalized_delta", row.get("prediction")))
    return out


def validation_ids() -> set[str]:
    ids: set[str] = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION" and row["study_unit_id"] == "GSE114002":
                ids.add(str(row["canonical_record_id"]))
    return ids


def zscored_mean(dir_by_seed: dict[int, Path], ids: set[str]) -> dict[str, float]:
    seed_preds: dict[int, dict[str, float]] = {}
    for seed, directory in dir_by_seed.items():
        preds = {k: v for k, v in load_preds(directory / "predictions.jsonl").items() if k in ids}
        if len(preds) != len(ids):
            raise FileNotFoundError(f"seed {seed}: coverage {len(preds)}/{len(ids)} (not terminal?)")
        values = np.array(list(preds.values()))
        z = (values - values.mean()) / values.std()
        seed_preds[seed] = dict(zip(preds.keys(), z))
    return {rid: float(np.mean([seed_preds[s][rid] for s in dir_by_seed])) for rid in ids}


def per_seed_readings(dir_by_seed: dict[int, Path], ids: set[str]) -> dict[str, dict]:
    obs = ev.load_observations([CANONICAL_GSE114002], ids)
    out = {}
    for seed, directory in dir_by_seed.items():
        preds = {k: v for k, v in load_preds(directory / "predictions.jsonl").items() if k in ids}
        if len(preds) != len(ids):
            raise FileNotFoundError(f"seed {seed}: coverage {len(preds)}/{len(ids)} (not terminal?)")
        m = ev.evaluate(obs, preds, K)
        out[f"seed{seed}"] = {
            "task_macro_spearman": float(m["task_macro_spearman"]),
            "top_1": m["source_macro_top_1_accuracy"],
            "ndcg_at_10": m["source_macro_ndcg_at_k"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--standin-v2", action="store_true",
                        help="verification mode: use the V2 seed dirs as the arm (must reproduce 0.3158289984824722)")
    args = parser.parse_args()

    ids = validation_ids()
    if len(ids) != 730:
        raise SystemExit(f"GSE114002 VALIDATION id count {len(ids)} != 730")

    arm_dirs = (V2_DIRS if args.standin_v2
                else {seed: ARM_ROOT / f"seed_{seed}" for seed in ARM_SEEDS})
    try:
        arm_ensemble = zscored_mean(arm_dirs, ids)
        per_seed = per_seed_readings(arm_dirs, ids)
    except FileNotFoundError as exc:
        print(f"NOT_TERMINAL: {exc}")
        return 3
    v2_ensemble = zscored_mean(V2_DIRS, ids)

    obs = ev.load_observations([CANONICAL_GSE114002], ids)
    m_arm = ev.evaluate(obs, arm_ensemble, K)
    m_v2 = ev.evaluate(obs, v2_ensemble, K)
    pb = ev.paired_group_bootstrap(obs, arm_ensemble, v2_ensemble, BOOT_ITERS, BOOT_SEED, K)
    ts = pb["task_macro_spearman"]
    rk = pb.get("ranking") or {}

    def ci(entry):
        c = entry.get("bootstrap_ci_95") if isinstance(entry, dict) else None
        return None if not c else [float(c[0]), float(c[1])]

    arm_value = float(m_arm["task_macro_spearman"])
    delta_lock = abs(arm_value - V2_ENSEMBLE_ARCHIVED) if args.standin_v2 else None
    if args.standin_v2 and (delta_lock or 0.0) > 1e-12:
        raise SystemExit(f"stand-in verification FAILED: {arm_value} vs archived {V2_ENSEMBLE_ARCHIVED}")

    result = {
        "schema_version": "route_a_v3_m1_intervention_3seed_ensemble.v1",
        "utc": utcnow(),
        "mode": "STANDIN_V2_VERIFICATION" if args.standin_v2 else "M1_ARM_3SEED",
        "amendment": "docs/paper/m1_intervention_arm_amendment_v2_seed_extension.md",
        "arm_seeds": sorted(arm_dirs),
        "arm_dirs": {str(k): str(v) for k, v in arm_dirs.items()},
        "v2_seeds": sorted(V2_DIRS),
        "averaging": "PER_SEED_ZSCORE_MEAN",
        "bootstrap_iterations": BOOT_ITERS,
        "bootstrap_seed": BOOT_SEED,
        "per_seed_readings": per_seed,
        "arm_ensemble": {
            "task_macro_spearman": arm_value,
            "top_1": m_arm["source_macro_top_1_accuracy"],
            "ndcg_at_10": m_arm["source_macro_ndcg_at_k"],
        },
        "v2_ensemble_recomputed": {
            "task_macro_spearman": float(m_v2["task_macro_spearman"]),
            "archived_value": V2_ENSEMBLE_ARCHIVED,
            "match": bool(abs(float(m_v2["task_macro_spearman"]) - V2_ENSEMBLE_ARCHIVED) <= 1e-12),
        },
        "delta_vs_v2_ensemble": {
            "point": ts["improvement"],
            "ci_95": ci(ts),
            "ci_excludes_zero_positive": bool((ci(ts) or [0.0, 0.0])[0] > 0.0),
        },
        "delta_top_1": {"point": rk.get("top_1", {}).get("mean_improvement"), "ci_95": ci(rk.get("top_1", {}))},
        "delta_ndcg": {"point": rk.get("ndcg", {}).get("mean_improvement"), "ci_95": ci(rk.get("ndcg", {}))},
    }
    if args.standin_v2:
        result["verification"] = {"archived_match_abs_diff": delta_lock,
                                  "note": "stand-in arm == V2 dirs; ensemble value must equal the archived value"}
    result["verdict"] = (
        "STANDIN_VERIFICATION_PASS" if args.standin_v2 else
        ("SIGNIFICANT_IMPROVEMENT" if result["delta_vs_v2_ensemble"]["ci_excludes_zero_positive"]
         else "NOT_SIGNIFICANT (CI crosses zero or negative)")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "m1_3seed_ensemble_standin_v2.json" if args.standin_v2 else "m1_3seed_ensemble_v1.json"
    out_path = args.out_dir / name
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: result[k] for k in ("mode", "arm_ensemble", "v2_ensemble_recomputed",
                                             "delta_vs_v2_ensemble", "verdict")}, indent=1))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())