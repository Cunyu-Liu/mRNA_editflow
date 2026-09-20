#!/usr/bin/env python3
"""P1-2 (audit A5): first-order decomposition TRAIN-only alpha sensitivity.

Exact replication of run_route2_first_order_decomposition_v1.py (features,
source encoding, Ridge closed form, per-task evaluation caliber), with the
ONLY change: alpha grid (0.3,1,3,10,30,100) selected by TRAIN-internal
5-fold GroupKFold CV (group = source_id; fold-internal source-encoding
re-estimation; CV permutation seed 20260920) instead of VALIDATION Spearman.

Evaluation face unchanged (VALIDATION) - sensitivity targets only the
hyperparameter selection process. Read-only, CPU only, TEST reads = 0.
Outputs -> experiments/analysis_first_order_decomposition_v1/sensitivity_train_alpha/
Prereg: docs/paper/first_order_train_alpha_sensitivity_prereg_v1.md (frozen
before computation). Original VALIDATION-alpha values stay as primary reads.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
OUT = MNT / "experiments/analysis_first_order_decomposition_v1/sensitivity_train_alpha"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "polyA": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "MPRAU": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "TE": "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",
}
BASES = "ACGT"
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
MPRAU_STUDY = "ENCSR854RUF"
CV_SEED = 20260920
ORIG = {
    "polyA": {"rho1": 0.45554942055906655, "alpha": 100.0, "n_train": 25710, "n_val": 2628},
    "MPRAU": {"rho1": 0.08253518190864256, "alpha": 10.0, "n_train": 55704, "n_val": 12048},
    "TE": {"rho1": 0.045771970425483255, "alpha": 10.0, "n_train": 3318, "n_val": 1614},
}


def load(split: str) -> dict[str, list[dict]]:
    bt = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bt[d["task_id"]].append(d)
    return bt


def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]


def build_ctx_feats(rows, n_blocks):
    n_ctx = 64
    X = np.zeros((len(rows), n_blocks * n_ctx))
    srcs = []
    for i, r in enumerate(rows):
        s = r["source_sequence"].upper().replace("U", "T")
        for pos, alt in edits_of(r):
            if pos is None or pos >= n_blocks * 8 or alt is None or alt not in BASES:
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
            X[i, (pos // 8) * n_ctx + ci] += 1.0
        srcs.append(r["source_id"])
    return X, srcs


def source_encoding(ytr, srcs_tr):
    src_mean = defaultdict(list)
    for s, y in zip(srcs_tr, ytr):
        src_mean[s].append(y)
    grand = float(np.mean(ytr))
    return {s: (np.mean(v) * len(v) + grand * 5) / (len(v) + 5) for s, v in src_mean.items()}, grand


def mprau_pair_mean(rows, preds):
    by_var = defaultdict(list)
    for r, p in zip(rows, preds):
        rid = str(r["canonical_record_id"])
        if rid.startswith(f"{MPRAU_STUDY}:"):
            by_var[rid.split(":context:")[0]].append((float(r["direction_normalized_delta"]), float(p)))
    tgt, prd = [], []
    for members in by_var.values():
        if len(members) >= 2:
            tgt.append(float(np.mean([m[0] for m in members])))
            prd.append(float(np.mean([m[1] for m in members])))
    return {"n_variants": len(tgt), "pair_mean_rho": float(spearmanr(prd, tgt).statistic)}


def eval_rho(label, rows, pred, yva):
    if label == "MPRAU":
        return mprau_pair_mean(rows, pred)["pair_mean_rho"]
    return float(spearmanr(pred, yva).statistic)


def train_cv_alpha(label, Xtr, ytr, srcs_tr, rows_tr):
    """GroupKFold-style 5-fold CV on TRAIN (group = source_id), fold-internal
    source-encoding re-estimation; returns per-alpha mean CV Spearman."""
    groups = np.array(srcs_tr)
    uniq = sorted(set(srcs_tr))
    rng = random.Random(CV_SEED)
    rng.shuffle(uniq)
    fold_of = {s: i % 5 for i, s in enumerate(uniq)}
    fold_ids = np.array([fold_of[s] for s in srcs_tr])
    scores = {a: [] for a in ALPHAS}
    for k in range(5):
        te = fold_ids == k
        tr = ~te
        if te.sum() < 10 or tr.sum() < 10:
            continue
        Xtr_f, ytr_f = Xtr[tr], ytr[tr]
        src_enc_f, grand_f = source_encoding(ytr_f, [s for s, keep in zip(srcs_tr, tr) if keep])
        btr_f = np.array([src_enc_f.get(s, grand_f) for s, keep in zip(srcs_tr, tr) if keep])
        bte_f = np.array([src_enc_f.get(s, grand_f) for s, keep in zip(srcs_tr, te) if keep])
        rows_te_f = [r for r, keep in zip(rows_tr, te) if keep]
        yte_f = ytr[te]
        for a in ALPHAS:
            ridge = Ridge(alpha=a).fit(Xtr_f, ytr_f - btr_f)
            pred = ridge.predict(Xtr[te]) + bte_f
            if np.std(pred) == 0:
                continue
            scores[a].append(eval_rho(label, rows_te_f, pred, yte_f))
    mean_scores = {a: float(np.mean(v)) if v else float("nan") for a, v in scores.items()}
    best_alpha = max(mean_scores, key=lambda a: (mean_scores[a] if not np.isnan(mean_scores[a]) else -1e9))
    return best_alpha, mean_scores


def main():
    train, val = load("train"), load("validation")
    results = {
        "schema": "route_a_v3_first_order_train_alpha_sensitivity.v1",
        "date": str(_date.today()),
        "prereg": "docs/paper/first_order_train_alpha_sensitivity_prereg_v1.md (frozen before computation; sensitivity-only, original VALIDATION-alpha reads stay primary)",
        "sensitivity_positioning": "alpha grid selected by TRAIN-internal 5-fold GroupKFold CV (group=source_id, fold-internal source-encoding re-estimation, CV seed 20260920); evaluation face unchanged (VALIDATION); zero formula changes vs run_route2_first_order_decomposition_v1.py",
        "tasks": {},
    }
    for label, task_id in TASKS.items():
        tr, va = train[task_id], val[task_id]
        L = max(len(r["source_sequence"]) for r in tr + va)
        all_pos = [p for r in tr + va for p, _ in edits_of(r) if p is not None]
        L = max(L, (max(all_pos) if all_pos else 0) + 1)
        n_blocks = (L + 7) // 8
        Xtr, srcs_tr = build_ctx_feats(tr, n_blocks)
        Xva, srcs_va = build_ctx_feats(va, n_blocks)
        ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
        yva = np.array([r["direction_normalized_delta"] for r in va], float)
        src_enc, grand = source_encoding(ytr, srcs_tr)
        btr = np.array([src_enc.get(s, grand) for s in srcs_tr])
        bva = np.array([src_enc.get(s, grand) for s in srcs_va])

        alpha_tr, cv_scores = train_cv_alpha(label, Xtr, ytr, srcs_tr, tr)
        ridge = Ridge(alpha=alpha_tr).fit(Xtr, ytr - btr)
        pred = ridge.predict(Xva) + bva
        rho_train_alpha = eval_rho(label, va, pred, yva)

        o = ORIG[label]
        rec = {
            "task_id": task_id,
            "n_train": len(tr),
            "n_val": len(va),
            "alpha_train_cv": float(alpha_tr),
            "cv_mean_spearman_by_alpha": {str(a): round(v, 6) for a, v in cv_scores.items()},
            "rho_first_order_train_alpha": float(rho_train_alpha),
            "original_alpha_validation": o["alpha"],
            "original_rho_first_order": o["rho1"],
            "delta_rho": float(rho_train_alpha - o["rho1"]),
            "caliber": ("variant pair-mean rho (rid before ':context:', >=2 contexts, per-variant means)" if label == "MPRAU" else "task Spearman on VALIDATION"),
            "over354_note": "sensitivity-only: original VALIDATION-alpha value remains the primary archived read; this row is the TRAIN-only-alpha robustness footnote material for the 55.4% first-order claim",
        }
        results["tasks"][label] = rec
        print(f"== {label}: rho1(train-alpha)={rho_train_alpha:.4f} (alpha={alpha_tr}, orig alpha={o['alpha']}, orig rho1={o['rho1']:.4f}, delta={rho_train_alpha-o['rho1']:+.4f})")

    results["summary"] = {
        "max_abs_delta_rho": max(abs(r["delta_rho"]) for r in results["tasks"].values()),
        "expected_band_from_prereg": "|delta| < 0.02 (coarse 6-point grid + closed-form Ridge smoothness)",
        "interpretation_rule": "if max |delta| < 0.02 -> 55.4% first-order share robust to alpha selection caliber; if >= 0.05 -> recommend range wording in section 8.3 (recommendation only, not executed in this batch)",
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("saved:", OUT / "results.json")


if __name__ == "__main__":
    main()
