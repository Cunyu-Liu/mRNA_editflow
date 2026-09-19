#!/usr/bin/env python3
"""Task 1.1 (mechanism chain B): first-order context energy table decomposition.

Exact replication of the E7-a v2 probe (run_erk_e7_v2.py) first-order-only
form, extended to three tasks: polyA (GSE269595), MPRAU (ENCSR854RUF, variant
pair-mean caliber), TE (GSE200304).

Model (zero formula changes vs run_erk_e7_v2.py, second-order terms REMOVED):
    Delta_hat(candidate|source) = sum_e W[block(e), ctx(e)] + b[source]
    block(e) = pos(e)//8 ; ctx(e) = idx(left)*16 + idx(right)*4 + idx(alt)
    left/right from ORIGINAL source sequence (N handling identical to E7).
    b[source] = TRAIN target encoding (per-source mean, grand-mean prior 5).
    Closed-form Ridge on (y - b); alpha grid (0.3,1,3,10,30,100) selected by
    VALIDATION Spearman (identical to E7).

Calibers per task (main criterion):
    polyA: task Spearman on VALIDATION (2,628)
    MPRAU: variant pair-mean rho (rid before ':context:', >=2 contexts,
           per-variant means of target/pred, Spearman over 2,008 variants)
           -- identical to W-ladder/V6-H3/Saluki caliber; record-level rho
           reported as an additional row.
    TE: task Spearman on VALIDATION (1,614)

Discipline: TRAIN fit / VALIDATION eval only, protected TEST reads = 0,
pure CPU, read-only on existing artifacts.

Prereg: docs/paper/first_order_decomposition_prereg_v1.md (written BEFORE
this run). Outputs -> experiments/analysis_first_order_decomposition_v1/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
OUT = MNT / "experiments/analysis_first_order_decomposition_v1"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "polyA": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "MPRAU": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "TE": "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY::region=1",
}
BASES = "ACGT"
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
MPRAU_STUDY = "ENCSR854RUF"

REFERENCE = {
    "polyA": {
        "frozen_delta_unsupervised": {"UTR-LM": 0.7490, "RNA-FM": 0.7114},
        "dense_supervision": {"APARENT_2019": 0.7343},
        "critic_v5": 0.8219,
        "ceiling_icc": 0.90,
    },
    "MPRAU": {
        "frozen_delta_unsupervised": {"UTR-LM": 0.0147, "RNA-FM": 0.0180},
        "dense_supervision": {"Saluki_weak_control": 0.1205},
        "critic_v5": 0.1025,
        "critic_v5_record_level": 0.0732,
        "ceiling_icc": 0.683,
    },
    "TE": {
        "frozen_delta_unsupervised": {"UTR-LM": 0.0113, "RNA-FM": 0.0009},
        "dense_supervision": {},
        "critic_v5": 0.0579,
        "ceiling_icc": 0.5857,
    },
}
THRESH = 0.05


def load(split: str) -> dict[str, list[dict]]:
    bt = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bt[d["task_id"]].append(d)
    return bt


def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]


def build_ctx_feats(rows: list[dict], n_blocks: int) -> tuple[np.ndarray, list[str]]:
    """Exact E7-a v2 ctx_feats (first-order only)."""
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


def source_encoding(ytr: np.ndarray, srcs_tr: list[str]):
    src_mean = defaultdict(list)
    for s, y in zip(srcs_tr, ytr):
        src_mean[s].append(y)
    grand = float(np.mean(ytr))
    return {s: (np.mean(v) * len(v) + grand * 5) / (len(v) + 5) for s, v in src_mean.items()}, grand


def mprau_pair_mean(rows, preds) -> dict:
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
    return {
        "n_variants": len(tgt),
        "pair_mean_rho": float(spearmanr(prd, tgt).statistic),
    }


def adjudicate(task: str, rho1: float) -> dict:
    ref = REFERENCE[task]
    band = list(ref["frozen_delta_unsupervised"].values())
    U = float(np.median(band))
    dense = ref["dense_supervision"]
    A = abs(rho1 - U) <= THRESH
    if dense:
        D = max(dense.values())
        B = (D - rho1) > THRESH
        if A and B:
            verdict = "成立"
        elif A or B:
            verdict = "部分成立"
        else:
            verdict = "不成立"
        return {"unsupervised_band_median": U, "A_within_band": bool(A),
                "B_below_dense": bool(B), "dense_row": D, "verdict": verdict}
    # B not applicable (MPRAU: Saluki = weak control, not MPRAU dense supervision; TE: none)
    if A:
        verdict = "部分成立（B 不适用）"
    elif rho1 > U + THRESH:
        verdict = "不成立（一阶超出无监督带，超出部分归非一阶/结构信号）"
    else:
        verdict = "不成立（一阶不足以解释外部行信号）"
    return {"unsupervised_band_median": U, "A_within_band": bool(A),
            "B_below_dense": None, "verdict": verdict}


def main():
    train, val = load("train"), load("validation")
    results = {
        "schema": "route_a_v3_first_order_decomposition.v1",
        "date": str(_date.today()),
        "prereg": "docs/paper/first_order_decomposition_prereg_v1.md (committed before computation)",
        "e7_replication": "run_erk_e7_v2.py ctx_feats, first-order only (no second-order), Ridge closed form, alpha by VALIDATION Spearman",
        "tasks": {},
    }

    for label, task_id in TASKS.items():
        tr, va = train[task_id], val[task_id]
        L = max(len(r["source_sequence"]) for r in tr + va)
        all_pos = [p for r in tr + va for p, _ in edits_of(r) if p is not None]
        L = max(L, (max(all_pos) if all_pos else 0) + 1)
        n_blocks = (L + 7) // 8
        n_feat = n_blocks * 64

        Xtr, srcs_tr = build_ctx_feats(tr, n_blocks)
        Xva, srcs_va = build_ctx_feats(va, n_blocks)
        ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
        yva = np.array([r["direction_normalized_delta"] for r in va], float)

        src_enc, grand = source_encoding(ytr, srcs_tr)
        btr = np.array([src_enc.get(s, grand) for s in srcs_tr])
        bva = np.array([src_enc.get(s, grand) for s in srcs_va])

        best = None
        for alpha in ALPHAS:
            ridge = Ridge(alpha=alpha).fit(Xtr, ytr - btr)
            pred = ridge.predict(Xva) + bva
            if np.std(pred) == 0:
                continue
            if label == "MPRAU":
                rho = mprau_pair_mean(va, pred)["pair_mean_rho"]
            else:
                rho = float(spearmanr(pred, yva).statistic)
            if best is None or rho > best[0]:
                best = (rho, alpha, pred)
        rho1, alpha1, pred1 = best

        rec = {
            "task_id": task_id,
            "n_train": len(tr),
            "n_val": len(va),
            "L": L,
            "n_blocks": n_blocks,
            "n_features": n_feat,
            "alpha": alpha1,
            "rho_first_order": float(rho1),
            "caliber": ("variant pair-mean rho (rid before ':context:', >=2 contexts, per-variant means)"
                        if label == "MPRAU" else "task Spearman on VALIDATION"),
            "reference": REFERENCE[label],
        }
        if label == "MPRAU":
            rec["rho_record_level"] = float(spearmanr(pred1, yva).statistic)
            rec["pair_mean_detail"] = mprau_pair_mean(va, pred1)
        rec["adjudication"] = adjudicate(label, float(rho1))
        ref = REFERENCE[label]
        v5 = ref["critic_v5"]
        rec["narrative_extras"] = {
            "first_order_vs_v5": float(rho1) - v5,
            "first_order_over_v5_ratio": float(rho1) / v5 if v5 else None,
            "v5_vs_ceiling_gap": v5 - ref["ceiling_icc"],
            "first_order_vs_ceiling_gap": float(rho1) - ref["ceiling_icc"],
        }
        results["tasks"][label] = rec
        print(f"== {label}: rho1={rho1:.4f} (alpha={alpha1}, feats={n_feat}, "
              f"n_train={len(tr)}, n_val={len(va)}) -> {rec['adjudication']['verdict']}")

    with open(OUT / "results_first_order.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("saved:", OUT / "results_first_order.json")


if __name__ == "__main__":
    main()
