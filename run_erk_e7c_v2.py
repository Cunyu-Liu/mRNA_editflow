#!/usr/bin/env python3
"""E7-c v2: mixed-pool off-manifold transfer with CONTEXT energy table.
Same protocol as E7-c v1 (clean per-source pool: 32 generated + measured),
model = context first-order (+pair blocks), gate 0.10, V5 ref 0.0614."""
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
BASES = "ACGT"

def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]

def load(split):
    bt = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bt[d["task_id"]].append(d)
    return bt

def main():
    train, val = load("train"), load("validation")
    task_id = "MEAN_RIBOSOME_LOAD::region=0"
    tr, va = train[task_id], val[task_id]
    L = 50
    n_blocks = (L + 7) // 8
    n_ctx = 64

    def feats_from_seqs(src, cand):
        f = np.zeros((n_blocks, n_ctx))
        es = []
        for p in range(min(len(src), L)):
            if src[p] != cand[p]:
                alt = cand[p]
                if alt not in BASES:
                    continue
                left = src[p - 1] if p > 0 and src[p - 1] in BASES else "N"
                right = src[p + 1] if p + 1 < len(src) and src[p + 1] in BASES else "N"
                if left == "N":
                    left = right if right != "N" else "A"
                if right == "N":
                    right = left if left != "N" else "A"
                ci = BASES.index(left) * 16 + BASES.index(right) * 4 + BASES.index(alt)
                f[p // 8, ci] += 1.0
                es.append(p)
        F = np.zeros((n_blocks, n_blocks))
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = es[i] // 8, es[j] // 8
                F[a, b] += 1.0
                F[b, a] += 1.0
        return np.concatenate([f.ravel(), F.ravel()])

    def feats_from_rows(r):
        s = r["source_sequence"].upper().replace("U", "T")
        c = list(s)
        for pos, alt in edits_of(r):
            if pos is not None and pos < len(c) and alt:
                c[pos] = alt
        return feats_from_seqs(s, "".join(c))

    Xtr = np.stack([feats_from_rows(r) for r in tr])
    ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
    ridge = Ridge(alpha=30.0).fit(Xtr, ytr)

    src_seq = {r["source_id"]: r["source_sequence"] for r in va + tr}
    measured = defaultdict(dict)
    for r in va:
        measured[r["source_id"]][r["candidate_sequence"]] = r["direction_normalized_delta"]

    gen_file = MNT / "experiments/xeditsetflow_v5/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl"
    pool_by_sk = defaultdict(list)
    with open(gen_file) as f:
        for line in f:
            d = json.loads(line)
            if "MEAN_RIBOSOME" not in d["source_key"]:
                continue
            pool_by_sk[d["source_key"]].append(d["candidate_sequence"])

    n_eval = n_correct = 0
    for sk, gen_cands in pool_by_sk.items():
        best_sid = None
        for sid, md in measured.items():
            if set(gen_cands) & set(md.keys()) and len(md) >= 2:
                best_sid = sid
                break
        if best_sid is None:
            continue
        md = measured[best_sid]
        s = src_seq[best_sid].upper().replace("T", "U").replace("U", "T")
        pool = list(dict.fromkeys(gen_cands))[:32] + list(md.keys())
        if len(pool) < 4:
            continue
        scores = [float(ridge.predict(feats_from_seqs(s, c).reshape(1, -1))[0]) for c in pool]
        top = pool[int(np.argmax(scores))]
        n_eval += 1
        if top == max(md.items(), key=lambda x: x[1])[0]:
            n_correct += 1
        if n_eval >= 300:
            break

    acc = n_correct / max(n_eval, 1)
    print(f"E7-c v2 (context model, clean pool): cond_acc@1 = {acc:.4f} (n={n_eval})")
    print(f"V5 ref 0.0614 | gate 0.10 -> {'PASS' if acc >= 0.10 else 'FAIL'}  (v1 contextless was 0.0267)")

    results = json.load(open(MNT / "experiments/analysis_erk_e7_probes/e7_v2_probe_results.json"))
    results["probes"]["E7c_v2_context_MRL"] = {"cond_acc1": acc, "n_eval": n_eval, "v5_ref": 0.0614, "gate": 0.10, "passed": bool(acc >= 0.10)}
    with open(MNT / "experiments/analysis_erk_e7_probes/e7_v2_probe_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved")

if __name__ == "__main__":
    main()
