#!/usr/bin/env python3
"""V9-1a 2-seed ensemble vs Route A 3-seed ensemble (MRL, paired bootstrap).

Route A per-record predictions: 280k_fullft_v2_6ep_{20260903,seed20260904,
seed20260905}/predictions.jsonl (field: predicted_direction_normalized_delta).
Both ensembles = per-seed z-score then mean (V8 convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
V9_DIR = MNT / "experiments/xeditcritic_route_a/v9_adapter_zoo_20260908"
ROUTE_A = [
    MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_20260903/predictions.jsonl",
    MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260904/predictions.jsonl",
    MNT / "experiments/xeditcritic_route_a/280k_fullft_v2_6ep_seed20260905/predictions.jsonl",
]
OUT = MNT / "experiments/analysis_v9_stage0_20260908/mrl_ensemble_vs_route_a.json"


def load(path: Path) -> dict[str, float]:
    preds = {}
    for line in path.open():
        row = json.loads(line)
        preds[row["canonical_record_id"]] = float(row["prediction"] if "prediction" in row
                                                  else row["predicted_direction_normalized_delta"])
    return preds


def zmean(per_seed: list[dict[str, float]], shared: list[str]) -> dict[str, float]:
    zs = []
    for p in per_seed:
        vals = np.array([p[r] for r in shared])
        mu, sd = vals.mean(), vals.std()
        zs.append({r: (p[r] - mu) / (sd if sd > 0 else 1.0) for r in shared})
    return {r: float(np.mean([z[r] for z in zs])) for r in shared}


def main() -> int:
    v9_seeds = [load(V9_DIR / f"seed{s}" / "mrl_predictions.jsonl") for s in ("20260907", "20260915")]
    ra_seeds = [load(p) for p in ROUTE_A]
    shared = set(v9_seeds[0]) & set(v9_seeds[1])
    for p in ra_seeds:
        shared &= set(p)
    shared = sorted(shared)
    # targets
    import sys
    sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904")
    sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904/scripts/route_a_v3")
    from run_route2_v8_stage2_adapt_v1 import _manifest_validation_ids, _load_canonical, BENCHMARK_DOMAINS
    vids = _manifest_validation_ids().get("GSE114002", set())
    recs = _load_canonical(BENCHMARK_DOMAINS["GSE114002"][1])
    shared = [r for r in shared if r in vids and r in recs]
    target = {r: float(recs[r]["direction_normalized_delta"]) for r in shared}

    v9_ens = zmean(v9_seeds, shared)
    ra_ens = zmean(ra_seeds, shared)

    def rho(pred):
        return float(spearmanr([target[r] for r in shared], [pred[r] for r in shared]).statistic)

    r_v9, r_ra = rho(v9_ens), rho(ra_ens)
    print("V9-1a 2-seed ensemble: %.4f | Route A 3-seed ensemble: %.4f | n=%d" % (r_v9, r_ra, len(shared)))

    rng = np.random.default_rng(20260816)
    n = len(shared)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        rs = [shared[i] for i in idx]
        t = np.array([target[r] for r in rs])
        e = np.array([v9_ens[r] for r in rs])
        v = np.array([ra_ens[r] for r in rs])
        deltas.append(spearmanr(t, e).statistic - spearmanr(t, v).statistic)
    ci = np.percentile(deltas, [2.5, 97.5])
    report = {"schema_version": "route_a_v3_v9_mrl_vs_route_a.v1", "n": n,
              "v9_ensemble_rho": r_v9, "route_a_ensemble_rho": r_ra,
              "delta": r_v9 - r_ra, "ci95": [float(ci[0]), float(ci[1])],
              "crosses_zero": bool(ci[0] < 0 < ci[1]), "iters": 2000, "seed": 20260816}
    print("Δ=%.4f CI [%.4f, %.4f] crosses_zero=%s" % (r_v9 - r_ra, ci[0], ci[1], report["crosses_zero"]))
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
