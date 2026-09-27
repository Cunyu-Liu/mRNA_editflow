#!/usr/bin/env python3
"""V5 polyA 3-seed robustness harvest v1 (P0-1 / rigor audit A1, 2026-09-28).

Implements the frozen harvest protocol of
docs/paper/polya_3seed_mini_prereg_v1.md sec.2:

  * per-seed polyA VALIDATION overall Spearman (n=2,628, frozen evaluator),
    full triple reported (20260907 main row + 20260921 + 20260922), no picking;
  * 3-seed mean +/- range;
  * source-group paired bootstrap (2,000 iters, seed 20260920) for the CI of
    the 3-seed mean and of the delta vs APARENT 0.7343 (frozen-delta);
  * Holm family direction recompute for the polyA row (other 3 raw p's kept
    frozen); SIGNIFICANT / not-significant reported as direction only;
  * dual caliber top-1 / NDCG@10 per seed (K=10).

Exit codes: 0 = harvest written; 3 = seeds not terminal yet (caller should skip).

Discipline: zero training; zero GPU; VALIDATION only; protected TEST reads = 0;
main row 0.8219 never replaced (new row = polyA-V5-3seed-mean, append-only).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL_GSE269595 = MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl"
APARENT_PREDS = MNT / "runs/development_hpo/aparent_v1/validation_predictions.jsonl"
HOLM_TABLE = MNT / "experiments/analysis_baseline_holm_20260909/holm_bottom_line.json"
DEFAULT_OUT = MNT / "experiments/xeditcritic_v5_polya_3seed/harvest_v1"

SEEDS_MAIN = {"seed20260907": None}  # resolved via glob
SEEDS_NEW = ("seed20260921", "seed20260922")
POLYA_MAIN = 0.8218686245779881
APARENT_FROZEN = 0.7343  # frozen-delta row point value (2,628 rows)
APARENT_BOOTSTRAP_SEED = 20260816  # frozen Task-1 protocol seed (cross-check)
BOOT_ITERS = 2000
BOOT_SEED = 20260920  # frozen by the mini prereg
K = 10


class NotTerminal(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_predictions(path: Path) -> dict[str, float]:
    preds: dict[str, float] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("canonical_record_id") or row.get("record_id"))
            preds[key] = float(row.get("predicted_direction_normalized_delta", row.get("prediction")))
    return preds


def polyA_ids() -> tuple[set[str], dict[str, str]]:
    """VALIDATION ids for GSE269595 (bootstrap groups are taken from canonical
    `source_id` inside main(), mirroring the frozen evaluator's cluster key
    (study_unit_id, source_id, biological_context_id, endpoint_id))."""
    ids: set[str] = set()
    groups: dict[str, str] = {}
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION" and row["study_unit_id"] == "GSE269595":
                rid = str(row["canonical_record_id"])
                ids.add(rid)
                groups[rid] = str(
                    row.get("source_group_key")
                    or row.get("source_group_id")
                    or row.get("connected_source_component_id")
                    or rid
                )
    return ids, groups


def seed_predictions() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    main_glob = glob.glob(str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl"))
    if main_glob:
        paths["seed20260907"] = Path(sorted(main_glob)[0])
    for name in SEEDS_NEW:
        p = MNT / "experiments/xeditcritic_v5_polya_3seed" / name / "final_validation_predictions.jsonl"
        if p.exists():
            paths[name] = p
    return paths


def evaluate_seed(preds_all: dict[str, float], ids: set[str], obs) -> dict:
    sub = {k: v for k, v in preds_all.items() if k in ids}
    if len(sub) != len(ids):
        raise NotTerminal(f"prediction coverage {len(sub)}/{len(ids)} - seed not terminal")
    metrics = ev.evaluate(obs, sub, K)
    return {
        "n": len(sub),
        "spearman": float(metrics["task_macro_spearman"]),
        "top_1": metrics["source_macro_top_1_accuracy"],
        "ndcg_at_10": metrics["source_macro_ndcg_at_k"],
        "predictions": sub,
    }


def bootstrap_mean_ci(
    seed_preds: list[dict[str, float]],
    aparent: dict[str, float],
    groups: dict[str, str],
    ids_sorted: list[str],
    obs: dict[str, dict],
    iters: int,
    seed: int,
) -> dict:
    """Source-group paired bootstrap: percentile CI of the 3-seed mean and of
    mean - APARENT, plus a two-sided bootstrap p for the paired delta."""
    group_index: dict[str, list[int]] = {}
    for idx, rid in enumerate(ids_sorted):
        group_index.setdefault(groups[rid], []).append(idx)
    group_keys = sorted(group_index)
    group_rows = [np.asarray(group_index[g], dtype=int) for g in group_keys]
    observed = np.asarray([float(obs[rid]["observed"]) for rid in ids_sorted], dtype=float)
    pred_arrays = [np.asarray([p[rid] for rid in ids_sorted], dtype=float) for p in seed_preds]
    aparent_array = np.asarray([aparent[rid] for rid in ids_sorted], dtype=float)

    rng = np.random.default_rng(seed)
    means: list[float] = []
    deltas: list[float] = []
    for _ in range(iters):
        pick = rng.integers(0, len(group_keys), len(group_keys))
        rows = np.concatenate([group_rows[i] for i in pick])
        if len(rows) < 3:
            continue
        y = observed[rows]
        rho_seeds = []
        for arr in pred_arrays:
            value = spearmanr(y, arr[rows]).statistic
            if np.isfinite(value):
                rho_seeds.append(value)
        rho_aparent = spearmanr(y, aparent_array[rows]).statistic
        if len(rho_seeds) != len(pred_arrays) or not np.isfinite(rho_aparent):
            continue
        mean3 = float(np.mean(rho_seeds))
        means.append(mean3)
        deltas.append(mean3 - float(rho_aparent))

    means_arr = np.asarray(means)
    deltas_arr = np.asarray(deltas)
    frac_nonpos = float(np.mean(deltas_arr <= 0.0))
    frac_nonneg = float(np.mean(deltas_arr >= 0.0))
    p_raw = min(1.0, max(2.0 * min(frac_nonpos, frac_nonneg), 1.0 / iters))
    return {
        "iterations": iters,
        "seed": seed,
        "unit": "source-group paired cluster",
        "mean_ci95": [float(np.percentile(means_arr, 2.5)), float(np.percentile(means_arr, 97.5))],
        "delta_ci95": [float(np.percentile(deltas_arr, 2.5)), float(np.percentile(deltas_arr, 97.5))],
        "delta_point": float(np.mean(deltas_arr)),
        "raw_p": p_raw,
        "ci_excludes_zero": bool(np.percentile(deltas_arr, 2.5) > 0.0),
    }


def holm_direction(polyA_raw_p: float) -> dict:
    if not HOLM_TABLE.exists():
        return {"available": False}
    table = json.loads(HOLM_TABLE.read_text())
    others = {k: v for k, v in table["pvals_raw"].items() if not k.startswith("polyA")}
    entries = sorted([("polyA: V5-3seed-mean vs APARENT", polyA_raw_p)] + list(others.items()),
                     key=lambda kv: kv[1])
    m = len(entries)
    adjusted = {}
    running = 0.0
    for i, (key, p) in enumerate(entries):
        value = min(1.0, (m - i) * p)
        running = max(running, value)
        adjusted[key] = {"raw_p": p, "holm_p": running, "significant_holm_0.05": running < 0.05}
    return {
        "available": True,
        "family_size": m,
        "frozen_other_rows": others,
        "frozen_polyA_row": table["pvals_raw"].get("polyA: V5 vs APARENT"),
        "recomputed": adjusted,
        "polyA_direction_recomputed": adjusted["polyA: V5-3seed-mean vs APARENT"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ids, _manifest_groups = polyA_ids()
    if len(ids) != 2628:
        raise SystemExit(f"GSE269595 VALIDATION id count {len(ids)} != 2628")
    obs = ev.load_observations([CANONICAL_GSE269595], ids)
    obs = {str(o["canonical_record_id"]): o for o in obs}
    groups = {
        rid: (str(entry["study_unit_id"]), str(entry["source_id"]),
              str(entry["biological_context_id"]), str(entry["endpoint_id"]))
        for rid, entry in obs.items()
    }
    print(f"[groups] cluster key = evaluator 4-tuple; n_groups={len(set(groups.values()))}")

    paths = seed_predictions()
    missing = [name for name in ("seed20260907",) + SEEDS_NEW if name not in paths]
    if missing:
        print(f"NOT_TERMINAL: missing prediction files for {missing}")
        return 3

    per_seed = {}
    for name in ("seed20260907",) + SEEDS_NEW:
        per_seed[name] = evaluate_seed(load_predictions(paths[name]), ids, list(obs.values()))

    main_value = per_seed["seed20260907"]["spearman"]
    if abs(main_value - POLYA_MAIN) > 1e-9:
        raise SystemExit(f"main-row reproduce check failed: {main_value} != {POLYA_MAIN}")

    aparent_all = load_predictions(APARENT_PREDS)
    aparent = {k: v for k, v in aparent_all.items() if k in ids}
    if len(aparent) != len(ids):
        raise SystemExit(f"APARENT coverage {len(aparent)}/{len(ids)}")
    aparent_metrics = ev.evaluate(list(obs.values()), aparent, K)

    ids_sorted = sorted(ids)
    seed_pred_list = [per_seed[name]["predictions"] for name in ("seed20260907",) + SEEDS_NEW]
    boot = bootstrap_mean_ci(seed_pred_list, aparent, groups, ids_sorted, obs, BOOT_ITERS, BOOT_SEED)
    boot_cross = bootstrap_mean_ci(seed_pred_list, aparent, groups, ids_sorted, obs,
                                   BOOT_ITERS, APARENT_BOOTSTRAP_SEED)

    values = [per_seed[name]["spearman"] for name in ("seed20260907",) + SEEDS_NEW]
    mean3 = float(np.mean(values))
    result = {
        "schema_version": "route_a_v3_v5_polya_3seed_harvest.v1",
        "utc": utcnow(),
        "prereg": "docs/paper/polya_3seed_mini_prereg_v1.md",
        "coverage": {"n_records": len(ids), "n_source_groups": len(set(groups.values()))},
        "per_seed": {name: {k: v for k, v in entry.items() if k != "predictions"}
                     for name, entry in per_seed.items()},
        "three_seed": {
            "values_in_order": {"seed20260907": values[0], "seed20260921": values[1], "seed20260922": values[2]},
            "mean": mean3,
            "range": [float(min(values)), float(max(values))],
            "main_row_unchanged": POLYA_MAIN,
            "row_label": "polyA-V5-3seed-mean",
        },
        "aparent_reference": {
            "frozen_point": APARENT_FROZEN,
            "recomputed_spearman": float(aparent_metrics["task_macro_spearman"]),
            "top_1": aparent_metrics["source_macro_top_1_accuracy"],
            "ndcg_at_10": aparent_metrics["source_macro_ndcg_at_k"],
        },
        "delta_vs_aparent": {
            "point": mean3 - float(aparent_metrics["task_macro_spearman"]),
            "ci95": boot["delta_ci95"],
            "ci_excludes_zero": boot["ci_excludes_zero"],
            "freeze_note": "prereg: main row 0.8219 never replaced; new row = mean +/- CI",
        },
        "bootstrap": boot,
        "bootstrap_crosscheck_seed20260816": boot_cross,
        "holm_direction": holm_direction(boot["raw_p"]),
        "dual_caliber": {
            "top_1": {name: per_seed[name]["top_1"] for name in per_seed},
            "ndcg_at_10": {name: per_seed[name]["ndcg_at_10"] for name in per_seed},
        },
    }
    direction = result["holm_direction"].get("polyA_direction_recomputed") if result["holm_direction"].get("available") else None
    if direction is not None:
        result["verdict"] = ("SIGNIFICANT_MAINTAINED_3SEED_ROBUST" if direction["significant_holm_0.05"]
                             else "INTERVAL_CLAIM_REQUIRED (bootstrap CI crosses zero - rewrite as interval)")
    else:
        result["verdict"] = "SIGNIFICANT_MAINTAINED" if boot["ci_excludes_zero"] else "INTERVAL_CLAIM_REQUIRED"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "polya_3seed_harvest_v1.json"
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: result[k] for k in ("three_seed", "delta_vs_aparent", "verdict")}, indent=1))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())