#!/usr/bin/env python3
"""ERK E7 probes v2 (bug fixes: pos cap, source encoding effect isolation)."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
OUT = MNT / "experiments/analysis_erk_e7_probes"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "MRL": "MEAN_RIBOSOME_LOAD::region=0",
    "polyA": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "MPRAU": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
}
GATE = {"MRL": 0.5 * 0.1354, "polyA": 0.5 * 0.8219, "MPRAU": 0.5 * 0.1025}

def load(split):
    by_task = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            by_task[d["task_id"]].append(d)
    return by_task

def edits_of(r):
    return [(e.get("position", e.get("source_position")), e.get("candidate_base", e.get("to", e.get("new_base")))) for e in r.get("source_relative_edits", [])]

def main():
    train = load("train")
    val = load("validation")
    results = {"schema": "erk_e7_probes.v2", "date": "2026-09-12", "probes": {}}

    for label, task_id in TASKS.items():
        tr, va = train[task_id], val[task_id]
        Ls = [len(r["source_sequence"]) for r in tr + va]
        L = int(max(Ls))
        print(f"\n== {label} (L={L}, train={len(tr)}, val={len(va)}) ==")

        # position indices can exceed L? clip check
        all_pos = [p for r in tr + va for p, _ in edits_of(r) if p is not None]
        max_pos = max(all_pos) if all_pos else 0
        L = max(L, max_pos + 1)

        def first_feats(r):
            f = np.zeros((L, 4))
            for pos, alt in edits_of(r):
                if pos is None or pos >= L or alt is None:
                    continue
                b = "ACGT".find(alt)
                if b >= 0:
                    f[pos, b] = 1.0
            return f.ravel()

        Xtr = np.stack([first_feats(r) for r in tr])
        ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
        Xva = np.stack([first_feats(r) for r in va])
        yva = np.array([r["direction_normalized_delta"] for r in va], float)

        # source background (target encoding smoothed)
        src_mean = defaultdict(list)
        for r, y in zip(tr, ytr):
            src_mean[r["source_id"]].append(y)
        grand = float(np.mean(ytr))
        src_enc = {s: (np.mean(v) * len(v) + grand * 5) / (len(v) + 5) for s, v in src_mean.items()}
        btr = np.array([src_enc.get(r["source_id"], grand) for r in tr])
        bva = np.array([src_enc.get(r["source_id"], grand) for r in va])

        best = None
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
            ridge = Ridge(alpha=alpha).fit(Xtr, ytr - btr)
            pred_va = ridge.predict(Xva) + bva
            if np.std(pred_va) == 0:
                continue
            rho = float(spearmanr(pred_va, yva).statistic)
            if best is None or rho > best[0]:
                best = (rho, alpha)
        rho1, alpha1 = best if best else (float("nan"), None)
        gate1 = GATE[label]
        # NOTE: background alone baseline (is first-order adding anything?)
        rho_bg = float(spearmanr(bva, yva).statistic) if np.std(bva) > 0 else float("nan")
        passed_a = (not np.isnan(rho1)) and rho1 >= gate1
        print(f"E7-a first-order: rho={rho1:.4f} (alpha={alpha1}) | background-only rho={rho_bg:.4f} | gate {gate1:.4f} -> {'PASS' if passed_a else 'FAIL'}")
        results["probes"][f"{label}__E7a"] = {"rho_first_order": rho1, "rho_background_only": rho_bg, "alpha": alpha1, "gate": gate1, "passed": bool(passed_a)}

        # E7-b: second-order co-occurrence block features
        n_blocks = (L + 7) // 8
        def pair_feats(r):
            F = np.zeros((n_blocks, n_blocks))
            es = [p for p, _ in edits_of(r) if p is not None and p < L]
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    a, b = es[i] // 8, es[j] // 8
                    F[a, b] += 1.0
                    F[b, a] += 1.0
            return F.ravel()

        Xtr2 = np.stack([pair_feats(r) for r in tr])
        Xva2 = np.stack([pair_feats(r) for r in va])
        Xtr12 = np.hstack([Xtr, Xtr2])
        Xva12 = np.hstack([Xva, Xva2])
        best2 = None
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
            ridge2 = Ridge(alpha=alpha).fit(Xtr12, ytr - btr)
            pred2 = ridge2.predict(Xva12) + bva
            if np.std(pred2) == 0:
                continue
            rho2 = float(spearmanr(pred2, yva).statistic)
            if best2 is None or rho2 > best2[0]:
                best2 = (rho2, alpha)
        rho2, alpha2 = best2 if best2 else (float("nan"), None)
        delta = (rho2 - rho1) if (not np.isnan(rho2) and not np.isnan(rho1)) else float("nan")
        passed_b = (not np.isnan(delta)) and delta >= 0.02
        print(f"E7-b +second-order: rho={rho2:.4f}  delta={delta:+.4f}  gate +0.02 -> {'PASS' if passed_b else 'FAIL'}")
        results["probes"][f"{label}__E7b"] = {"rho_second_order": rho2, "delta_vs_first": delta, "passed": bool(passed_b)}

    with open(OUT / "e7_probe_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved:", OUT / "e7_probe_results.json")

if __name__ == "__main__":
    main()
