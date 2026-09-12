#!/usr/bin/env python3
"""ERK v2 E7 probes: k-mer CONTEXT energy table (pos-block x flank-left x flank-right x new-base).

v1 diagnosis: position+base features lack sequence context -> mixed-pool failure.
v2: each edit contributes to feature (block, left_nt, right_nt, new_base) using the
ORIGINAL source sequence flanks. Params: n_blocks x 64. Plus second-order co-occurrence.
Gates unchanged: E7a rho >= 0.5xV5; E7b delta >= +0.02; E7c mixed-pool >= 0.10.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
OUT = MNT / "experiments/analysis_erk_e7_probes"

TASKS = {
    "MRL": "MEAN_RIBOSOME_LOAD::region=0",
    "polyA": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "MPRAU": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
}
GATE = {"MRL": 0.5 * 0.1354, "polyA": 0.5 * 0.8219, "MPRAU": 0.5 * 0.1025}
BASES = "ACGT"

def load(split):
    bt = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bt[d["task_id"]].append(d)
    return bt

def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]

def main():
    train, val = load("train"), load("validation")
    results = {"schema": "erk_e7_v2_context_probes.v1", "date": "2026-09-12", "probes": {}}

    for label, task_id in TASKS.items():
        tr, va = train[task_id], val[task_id]
        L = max(len(r["source_sequence"]) for r in tr + va)
        all_pos = [p for r in tr + va for p, _ in edits_of(r) if p is not None]
        L = max(L, (max(all_pos) if all_pos else 0) + 1)
        n_blocks = (L + 7) // 8
        n_ctx = 4 * 4 * 4  # left x right x new
        print(f"\n== {label} (L={L}, blocks={n_blocks}, ctx_feats={n_blocks * n_ctx}) ==")

        def ctx_feats(r):
            f = np.zeros((n_blocks, n_ctx))
            s = r["source_sequence"].upper().replace("U", "T")
            es = []
            for pos, alt in edits_of(r):
                if pos is None or pos >= L or alt is None or alt not in BASES:
                    continue
                left = s[pos - 1] if pos > 0 and s[pos - 1] in BASES else "N"
                right = s[pos + 1] if pos + 1 < len(s) and s[pos + 1] in BASES else "N"
                if left == "N" and right == "N":
                    continue
                if left == "N":
                    left = right
                if right == "N":
                    right = left
                ci = BASES.index(left) * 16 + BASES.index(right) * 4 + BASES.index(alt)
                f[pos // 8, ci] += 1.0
                es.append(pos)
            return f.ravel(), es

        def pair_feats(es):
            F = np.zeros((n_blocks, n_blocks))
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    a, b = es[i] // 8, es[j] // 8
                    F[a, b] += 1.0
                    F[b, a] += 1.0
            return F.ravel()

        def build(rows):
            X1, X2, srcs = [], [], []
            for r in rows:
                f1, es = ctx_feats(r)
                X1.append(f1)
                X2.append(pair_feats(es))
                srcs.append(r["source_id"])
            return np.hstack([np.stack(X1), np.stack(X2)]), srcs

        Xtr, srcs_tr = build(tr)
        Xva, srcs_va = build(va)
        ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
        yva = np.array([r["direction_normalized_delta"] for r in va], float)

        src_mean = defaultdict(list)
        for r, y in zip(tr, ytr):
            src_mean[r["source_id"]].append(y)
        grand = float(np.mean(ytr))
        src_enc = {s: (np.mean(v) * len(v) + grand * 5) / (len(v) + 5) for s, v in src_mean.items()}
        btr = np.array([src_enc.get(s, grand) for s in srcs_tr])
        bva = np.array([src_enc.get(s, grand) for s in srcs_va])

        best = None
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            ridge = Ridge(alpha=alpha).fit(Xtr, ytr - btr)
            pred = ridge.predict(Xva) + bva
            if np.std(pred) == 0:
                continue
            rho = float(spearmanr(pred, yva).statistic)
            if best is None or rho > best[0]:
                best = (rho, alpha)
        rho_v2, alpha_v2 = best if best else (float("nan"), None)
        gate = GATE[label]
        passed = (not np.isnan(rho_v2)) and rho_v2 >= gate
        print(f"E7-a v2 (context): rho={rho_v2:.4f} (alpha={alpha_v2})  gate {gate:.4f} -> {'PASS' if passed else 'FAIL'}")
        results["probes"][f"{label}__E7a_v2"] = {"rho_context": rho_v2, "alpha": alpha_v2, "gate": gate, "passed": bool(passed)}

    with open(OUT / "e7_v2_probe_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved:", OUT / "e7_v2_probe_results.json")

if __name__ == "__main__":
    main()
