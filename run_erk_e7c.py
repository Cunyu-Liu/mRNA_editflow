#!/usr/bin/env python3
"""E7-c: off-manifold transfer probe. ERK first+second-order vs V5 mixed-pool cond_acc@1.

Mixed pool = V6.5 B256 unguided generated candidates (off-manifold) + measured
candidates (on-manifold) per MRL source; rank by ERK score; cond_acc@1 = P(top-1
by model == top-1 by measured value), conditional on pool containing >=1 measured.
V5 reference = 0.0614 (A3 probe). Gate = 0.10.
"""
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

def edits_of(r):
    return [(e.get("position", e.get("source_position")), e.get("candidate_base", e.get("to", e.get("new_base")))) for e in r.get("source_relative_edits", [])]

def load(split):
    by_task = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            by_task[d["task_id"]].append(d)
    return by_task

def main():
    train = load("train")
    val = load("validation")
    task_id = "MEAN_RIBOSOME_LOAD::region=0"
    tr, va = train[task_id], val[task_id]
    L = 50

    def first_feats(r):
        f = np.zeros((L, 4))
        for pos, alt in edits_of(r):
            if pos is None or pos >= L or alt is None:
                continue
            b = "ACGT".find(alt)
            if b >= 0:
                f[pos, b] = 1.0
        return f.ravel()

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

    Xtr = np.hstack([np.stack([first_feats(r) for r in tr]), np.stack([pair_feats(r) for r in tr])])
    ytr = np.array([r["direction_normalized_delta"] for r in tr], float)
    ridge = Ridge(alpha=30.0).fit(Xtr, ytr)

    src_seq_by_id = {r["source_id"]: r["source_sequence"] for r in va + tr}
    measured_by_src = defaultdict(dict)
    for r in va:
        measured_by_src[r["source_id"]][r["candidate_sequence"]] = r["direction_normalized_delta"]

    def erk_score(src_seq, cand_seq):
        if len(cand_seq) != len(src_seq):
            return 0.0
        f1 = np.zeros((L, 4))
        es = []
        for p in range(min(len(src_seq), L)):
            if src_seq[p] != cand_seq[p]:
                b = "ACGT".find(cand_seq[p])
                if b >= 0:
                    f1[p, b] = 1.0
                es.append(p)
        F = np.zeros((n_blocks, n_blocks))
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = es[i] // 8, es[j] // 8
                F[a, b] += 1.0
                F[b, a] += 1.0
        x = np.concatenate([f1.ravel(), F.ravel()])
        return float(ridge.predict(x.reshape(1, -1))[0])

    gen_file = MNT / "experiments/xeditsetflow_v5/pool256_unguided_20260908/b2_full_891_B256/unguided/generated_candidates.private.jsonl"
    pool_by_seq = defaultdict(list)
    with open(gen_file) as f:
        for line in f:
            d = json.loads(line)
            if "MEAN_RIBOSOME" not in d["source_key"]:
                continue
            pool_by_seq[d["candidate_sequence"]].append(d["source_key"])

    # match generated candidates to validation sources by sequence membership
    n_eval = n_correct = 0
    used_srcs = set()
    for cand_seq, sks in pool_by_seq.items():
        for sid, md in measured_by_src.items():
            if cand_seq in md and sid not in used_srcs and len(md) >= 2:
                # build mixed pool: this + other generated cands for this source
                pool = [cand_seq]
                # find other generated candidates overlapping this source's measured keys or random extras
                others = [c for c in pool_by_seq if c != cand_seq and c in md]
                pool += others
                # add generated-only (off-manifold) candidates from the same run (random subsample)
                gen_only = [c for c in list(pool_by_seq)[:40] if c not in md]
                pool += gen_only[:16]
                if len(pool) < 4:
                    continue
                scores = [erk_score(src_seq_by_id[sid], c) for c in pool]
                best_pred = pool[int(np.argmax(scores))]
                best_true = max(md.items(), key=lambda x: x[1])[0]
                n_eval += 1
                used_srcs.add(sid)
                if best_pred == best_true:
                    n_correct += 1
                break
        if n_eval >= 400:
            break

    acc = n_correct / max(n_eval, 1)
    print(f"E7-c mixed-pool cond_acc@1 = {acc:.4f} (n={n_eval}) | V5 ref 0.0614 | gate 0.10 -> {'PASS' if acc >= 0.10 else 'FAIL'}")

    results = json.load(open(OUT / "e7_probe_results.json"))
    results["probes"]["E7c_mixed_pool_MRL"] = {"cond_acc1": acc, "n_eval": n_eval, "v5_ref": 0.0614, "gate": 0.10, "passed": bool(acc >= 0.10)}
    with open(OUT / "e7_probe_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved")

if __name__ == "__main__":
    main()
