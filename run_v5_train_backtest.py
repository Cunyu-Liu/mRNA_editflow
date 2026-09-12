#!/usr/bin/env python3
"""V5 critic TRAIN-split backtest: did the model learn (memorize) the training data?

Scores TRAIN rows through the official frozen V5 critic (same checkpoint as B2),
computes per-task Spearman on TRAIN vs VALIDATION. High TRAIN rho + low VAL rho
=> memorization without generalization. Low TRAIN rho => underfitting (never
learned even the training signal).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import torch

sys.path.insert(0, "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
OUT = MNT / "experiments/analysis_v5_train_backtest"
OUT.mkdir(parents=True, exist_ok=True)

V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
MRNABERT = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
WORKTREE = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")
sys.path.insert(0, str(WORKTREE))

from scripts.route_a_v3.route2_xeditcritic_v5_frozen_guidance_v1 import FrozenXEditCriticV5  # noqa: E402

TASKS = {
    "MRL": "MEAN_RIBOSOME_LOAD::region=0",
    "polyA": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1",
    "MPRAU": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE::region=1",
    "HL5": "RNA_HALF_LIFE_MINUTES::region=0",
    "HL3": "RNA_HALF_LIFE_MINUTES::region=1",
}

def load(split):
    bt = defaultdict(list)
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bt[d["task_id"]].append(d)
    return bt

class FakeState:
    def __init__(self, src, cand, assay, ctx):
        self.source_sequence = src
        self.current_sequence = cand
        self.assay_id = assay
        self.context_id = ctx

def main():
    device = torch.device("cuda:6")
    critic = FrozenXEditCriticV5(V5_CKPT, MRNABERT, device)

    train, val = load("train"), load("validation")
    results = {"schema": "v5_train_backtest.v1", "date": "2026-09-12", "tasks": {}}

    # subsample TRAIN for compute (2000 rows/task max, grouped by source)
    rng = np.random.RandomState(20260912)
    for label, task_id in TASKS.items():
        tr = train[task_id]
        if len(tr) > 2000:
            by_src = defaultdict(list)
            for r in tr:
                by_src[r["source_id"]].append(r)
            srcs = sorted(by_src)
            picked = []
            for s in srcs:
                picked.extend(by_src[s])
                if len(picked) >= 2000:
                    break
            tr = picked
        va = val[task_id]

        # group by source for potentials batch
        def score_rows(rows):
            by_src = defaultdict(list)
            for r in rows:
                by_src[r["source_id"]].append(r)
            ys, yh = [], []
            for sid, rs in by_src.items():
                src_row = rs[0]
                assay = src_row["assay_id"]
                ctx = src_row["biological_context_id"]
                src = src_row["source_sequence"].upper().replace("T", "U")
                endpoint = task_id.split("::")[0]
                region = "5UTR" if "region=0" in task_id else "3UTR"
                # limit batch per source to 32
                rs_batch = rs[:32]
                states = [FakeState(src, r["candidate_sequence"].upper().replace("T", "U"), assay, ctx) for r in rs_batch]
                try:
                    pots = critic.potentials(states, endpoint_id=endpoint, region=region, source_row=src_row)
                except Exception as e:
                    if len(ys) == 0:
                        print(f"  [debug] {label} first source failed: {type(e).__name__}: {e}")
                    continue
                for r, p in zip(rs_batch, pots):
                    ys.append(r["direction_normalized_delta"])
                    yh.append(p)
            return np.array(ys), np.array(yh)

        ys_tr, yh_tr = score_rows(tr)
        ys_va, yh_va = score_rows(va)
        rho_tr = float(spearmanr(yh_tr, ys_tr).statistic) if len(ys_tr) > 10 and np.std(yh_tr) > 0 else None
        rho_va = float(spearmanr(yh_va, ys_va).statistic) if len(ys_va) > 10 and np.std(yh_va) > 0 else None
        gap = (rho_tr - rho_va) if (rho_tr is not None and rho_va is not None) else None
        results["tasks"][label] = {
            "n_train": len(ys_tr), "n_val": len(ys_va),
            "rho_train": rho_tr, "rho_val": rho_va, "gap": gap,
        }
        print(f"{label}: TRAIN rho={rho_tr} (n={len(ys_tr)})  VAL rho={rho_va} (n={len(ys_va)})  gap={gap}")

    with open(OUT / "v5_train_backtest.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved:", OUT / "v5_train_backtest.json")

if __name__ == "__main__":
    main()
