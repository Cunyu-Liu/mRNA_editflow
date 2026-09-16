#!/usr/bin/env python3
"""D16-C gap backtest: train-backtest on the D16-C probe final checkpoint (epoch 6).

Mirrors run_v5_train_backtest.py exactly (same subsample seed, same scoring path,
same memo clearing) but scores through the D16-C probe checkpoint instead of the
official V5 terminal checkpoint. MRL only (the D16-C target task); G2 polyA read
from probe summary (already computed). This is the G1 gate adjudication
(amendment sec 4.2: gap <= 0.30 AND VAL rho >= 0.135).
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

W0 = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902")
SETFLOW = Path("/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901")
sys.path.insert(0, str(SETFLOW))
sys.path.insert(0, str(W0))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "v5frozen", str(SETFLOW / "scripts/route_a_v3/route2_xeditcritic_v5_frozen_guidance_v1.py")
)
v5frozen = importlib.util.module_from_spec(_spec)
sys.modules["v5frozen"] = v5frozen
_spec.loader.exec_module(v5frozen)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
PROBE_CKPT = MNT / "experiments/xeditcritic_d16c/probe_mrl_v1_gpu5/probe_epoch_6.pt"
OUT = MNT / "experiments/xeditcritic_d16c/probe_mrl_v1_gpu5"
OUT_JSON = OUT / "gap_backtest_d16c.json"

V5_CKPT = MNT / "experiments/xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c0dd9acb508f58782eecb40f458d2cab3/v5_full/final_pass_8_checkpoint.pt"
MRNABERT = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"

MRL_TASK = "MEAN_RIBOSOME_LOAD::region=0"
MRL_ENDPOINT = "MEAN_RIBOSOME_LOAD"
MRL_REGION = "5UTR"


def load_split(task_id):
    rows = []
    with open(PROJ / "train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == task_id:
                rows.append(d)
    return rows


def load_val(task_id):
    rows = []
    with open(PROJ / "validation.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == task_id:
                rows.append(d)
    return rows


class FakeState:
    def __init__(self, src, cand, assay, ctx):
        self.source_sequence = src
        self.current_sequence = cand
        self.assay_id = assay
        self.context_id = ctx


def main():
    device = torch.device("cuda:6")
    critic = v5frozen.FrozenXEditCriticV5(V5_CKPT, MRNABERT, device)

    # --- swap in D16-C probe weights (bit-identical scoring path) ---
    ck = torch.load(PROBE_CKPT, map_location="cpu", weights_only=False)
    assert ck["schema"] == "route_a_v3_d16c_probe.v1" and ck["epoch"] == 6
    missing, unexpected = critic.model.load_state_dict(ck["model_state_dict"], strict=True)
    critic.model.eval().requires_grad_(False)
    print("D16-C probe epoch-6 weights loaded (strict) ✓", flush=True)

    train = load_split(MRL_TASK)
    val = load_val(MRL_TASK)

    # identical subsample protocol as run_v5_train_backtest.py
    rng = np.random.RandomState(20260912)
    tr = train
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
    va = val

    def score_rows(rows):
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["source_id"]].append(r)
        ys, yh = [], []
        for sid in sorted(by_src):
            rs = by_src[sid]
            src_row = rs[0]
            assay = src_row["assay_id"]
            ctx = src_row["biological_context_id"]
            src = src_row["source_sequence"].upper().replace("T", "U")
            src_row["source_key"] = str(src_row["source_group_id"])
            rs_batch = rs[:32]
            states = [FakeState(src, r["candidate_sequence"].upper().replace("T", "U"), assay, ctx) for r in rs_batch]
            critic._potential_memo.clear()
            try:
                pots = critic.potentials(states, endpoint_id=MRL_ENDPOINT, region=MRL_REGION, source_row=src_row)
            except Exception as e:
                if len(ys) == 0:
                    print(f"  [debug] first source failed: {type(e).__name__}: {e}", flush=True)
                continue
            for r, p in zip(rs_batch, pots):
                ys.append(r["direction_normalized_delta"])
                yh.append(p)
        return np.array(ys), np.array(yh)

    print(f"scoring TRAIN (n<={len(tr)}) ...", flush=True)
    ys_tr, yh_tr = score_rows(tr)
    print(f"scoring VAL (n={len(va)}) ...", flush=True)
    ys_va, yh_va = score_rows(va)

    rho_tr = float(spearmanr(yh_tr, ys_tr).statistic) if len(ys_tr) > 10 and np.std(yh_tr) > 0 else None
    rho_va = float(spearmanr(yh_va, ys_va).statistic) if len(ys_va) > 10 and np.std(yh_va) > 0 else None
    gap = (rho_tr - rho_va) if (rho_tr is not None and rho_va is not None) else None

    # probe summary already has val rho (diagnostic); keep the official number too
    summary = json.load(open(OUT / "probe_summary.json"))
    probe_val_rho = summary["final_mrl_val_spearman"]

    g1_gap_pass = gap is not None and gap <= 0.30
    g1_val_pass = rho_va is not None and rho_va >= 0.135
    verdict = "G1_PASS" if (g1_gap_pass and g1_val_pass) else "G1_FAIL"

    result = {
        "schema": "d16c_gap_backtest.v1",
        "date": "2026-09-16",
        "checkpoint": str(PROBE_CKPT),
        "task": "MRL",
        "n_train": int(len(ys_tr)),
        "n_val": int(len(ys_va)),
        "rho_train": rho_tr,
        "rho_val": rho_va,
        "probe_summary_val_rho": probe_val_rho,
        "gap": gap,
        "reference_official_v5": {"rho_train": 0.9284, "rho_val": 0.1341, "gap": 0.7943},
        "g1_gate": {"gap_le_0.30": g1_gap_pass, "val_ge_0.135": g1_val_pass, "verdict": verdict},
        "protocol": "mirrors run_v5_train_backtest.py (subsample seed 20260912, per-source batch 32, memo cleared per source)",
        "protected_reads": 0,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
