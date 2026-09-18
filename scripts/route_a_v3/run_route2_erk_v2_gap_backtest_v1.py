#!/usr/bin/env python3
"""ERK v2 gap backtest: train-backtest on ERK FINAL-EPOCH checkpoint.

Mirrors run_d16c_gap_backtest_v1.py protocol (same deterministic 2000-row
source-ordered truncation, same per-source first-32 scoring calibre) but scores
through the ERK linear head (no memo, no encoder). MRL only.
This is the G1 gate adjudication (amendment ERK-G1: gap <= 0.30 AND VAL >= 0.135).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
CKPT = MNT / "experiments/xeditcritic_erk_v2/erk_train_seed2026091901/erk_epoch_6.pt"
OUT = MNT / "experiments/xeditcritic_erk_v2/erk_train_seed2026091901"
OUT_JSON = OUT / "gap_backtest_erk_v2.json"

MRL_TASK = "MEAN_RIBOSOME_LOAD::region=0"
BASES = "ACGT"


def load_split(path, task_id):
    rows = []
    with open(PROJ / path) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] == task_id:
                rows.append(d)
    return rows


def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]


def features(rows, n_blocks):
    n_ctx = 64
    X = np.zeros((len(rows), n_blocks * n_ctx), dtype=np.float32)
    for i, r in enumerate(rows):
        s = r["source_sequence"].upper().replace("U", "T")
        for pos, alt in edits_of(r):
            if pos is None or alt is None or alt not in BASES:
                continue
            if pos >= n_blocks * 8:
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
    return X


def main():
    # CUDA hard gate
    if not torch.cuda.is_available():
        OUT_JSON.write_text(json.dumps({"status": "STOPPED_WITH_EVIDENCE",
                                        "error": "CUDA unavailable", "cpu_fallback_used": True}))
        print("STOPPED_WITH_EVIDENCE: CUDA unavailable", flush=True)
        return 2
    # reuse GPU3 (ERK matrix is tiny)
    torch.cuda.set_device(3)
    dev = torch.device("cuda:3")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck["schema"] == "route_a_v3_erk_v2_train.v1" and ck["epoch"] == 6
    W = ck["model_state_dict"]["W"].to(dev, dtype=torch.float32)
    b = ck["model_state_dict"]["b"].to(dev, dtype=torch.float32)
    src_index = ck["source_index"]
    n_blocks = ck["n_blocks"]

    train = load_split("train.jsonl", MRL_TASK)
    val = load_split("validation.jsonl", MRL_TASK)

    # identical deterministic truncation as run_v5_train_backtest / D16-C backtest
    tr = train
    if len(tr) > 2000:
        by_src = defaultdict(list)
        for r in tr:
            by_src[r["source_id"]].append(r)
        picked = []
        for s in sorted(by_src):
            picked.extend(by_src[s])
            if len(picked) >= 2000:
                break
        tr = picked

    def build_batched(rows):
        out_rows = []
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["source_id"]].append(r)
        for sid in sorted(by_src):
            out_rows.extend(by_src[sid][:32])
        return out_rows

    tr_b = build_batched(tr)
    va_b = build_batched(val)
    Xtr = torch.from_numpy(features(tr_b, n_blocks)).to(dev)
    Xva = torch.from_numpy(features(va_b, n_blocks)).to(dev)
    itr = torch.tensor([src_index.get(str(r["source_id"]), 0) for r in tr_b], device=dev)
    iva = torch.tensor([src_index.get(str(r["source_id"]), 0) for r in va_b], device=dev)
    ytr = np.array([float(r["direction_normalized_delta"]) for r in tr_b])
    yva = np.array([float(r["direction_normalized_delta"]) for r in va_b])

    with torch.inference_mode():
        ptr = (Xtr @ W + b[itr]).cpu().numpy()
        pva = (Xva @ W + b[iva]).cpu().numpy()

    rho_tr = float(spearmanr(ptr, ytr).statistic) if len(ytr) > 10 and np.std(ptr) > 0 else None
    rho_va = float(spearmanr(pva, yva).statistic) if len(yva) > 10 and np.std(pva) > 0 else None
    gap = (rho_tr - rho_va) if (rho_tr is not None and rho_va is not None) else None

    summary = json.load(open(OUT / "erk_train_summary.json"))
    probe_val_rho = summary["final_mrl_val_spearman"]

    g1_gap_pass = gap is not None and gap <= 0.30
    g1_val_pass = rho_va is not None and rho_va >= 0.135
    verdict = "G1_PASS" if (g1_gap_pass and g1_val_pass) else "G1_FAIL"

    result = {
        "schema": "erk_v2_gap_backtest.v1",
        "date": "2026-09-19",
        "checkpoint": str(CKPT),
        "task": "MRL",
        "n_train": int(len(ptr)),
        "n_val": int(len(pva)),
        "rho_train": rho_tr,
        "rho_val": rho_va,
        "summary_val_rho": probe_val_rho,
        "gap": gap,
        "reference_official_v5": {"rho_train": 0.9284, "rho_val": 0.1341, "gap": 0.7943},
        "reference_d16c_probe": {"rho_train": 0.9558, "rho_val": 0.1324, "gap": 0.8235},
        "g1_gate": {"gap_le_0.30": g1_gap_pass, "val_ge_0.135": g1_val_pass, "verdict": verdict},
        "protocol": "mirrors run_v5_train_backtest.py (source-ordered 2000 truncation, per-source first-32, no-memo by construction)",
        "cuda_device": "cuda:3",
        "cpu_fallback_used": False,
        "protected_reads": 0,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())