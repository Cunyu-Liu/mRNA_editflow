#!/usr/bin/env python3
"""ERK v2 training arm (amendment v1, APPROVED 2026-09-19): explicit first-order
context energy table trained end-to-end on MRL, replacing the 170M free critic head.

Model (exact inheritance of E7-a v2 probe form, run_erk_e7_v2.py):
    Delta(candidate|source) = sum_e W[block(e), ctx(e)] + b[source]
  - block(e) = pos // 8 (8nt position blocks)
  - ctx(e)  = index(left_nt)*16 + index(right_nt)*4 + index(alt_base)
              left/right come from the ORIGINAL source sequence (N handled as in probe)
  - b = per-source residual head (one scalar per source id)
  - NO frozen encoder, NO pretrained features, NO second-order kernel (disabled by prereg)

Protocol alignment: Huber loss with V5 target scale (0.5232), FINAL-EPOCH-FIXED,
CUDA hard gate (no CPU, evidence recorded), VALIDATION-only reporting.

Discipline: no synthetic rows (D16-C data excluded), memory-free model by design.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
PROJ = MNT / "projections/xedit_v3/development_train_validation_v1"
MRL_TASK = "MEAN_RIBOSOME_LOAD::region=0"
POLYA_TASK = "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS::region=1"
BASES = "ACGT"
V5_TARGET_SCALE = 0.5232
V5_POLYA_VAL = 0.8219          # frozen V5 main-row reference (not touched by this arm)
PROBE_POLYA_CONTEXT = 0.5040   # E7-a v2 probe polyA context rho (report-only reference)


def load_rows(split: str) -> list[dict]:
    rows = []
    with open(PROJ / f"{split}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id") == MRL_TASK:
                rows.append(d)
    return rows


def edits_of(r):
    return [(e.get("position"), e.get("candidate_base")) for e in r.get("source_relative_edits", [])]


def build_features(rows, n_blocks):
    """Exact probe ctx_feats form (run_erk_e7_v2.py)."""
    n_ctx = 64
    X = np.zeros((len(rows), n_blocks * n_ctx), dtype=np.float32)
    skipped = 0
    for i, r in enumerate(rows):
        s = r["source_sequence"].upper().replace("U", "T")
        hits = 0
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
            hits += 1
        if hits == 0:
            skipped += 1
    return X, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physical-gpu-index", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out-dir", default=str(MNT / "experiments/xeditcritic_erk_v2/erk_train_seed2026091901"))
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026091901)
    args = ap.parse_args()

    # CUDA hard gate (no CPU fallback; evidence recorded)
    if not torch.cuda.is_available():
        off = Path(args.out_dir) / "erk_failed.json"
        off.parent.mkdir(parents=True, exist_ok=True)
        off.write_text(json.dumps({"status": "STOPPED_WITH_EVIDENCE", "error": "CUDA unavailable",
                                   "cpu_fallback_used": True}))
        print("STOPPED_WITH_EVIDENCE: CUDA unavailable", flush=True)
        return 2
    torch.cuda.set_device(args.physical_gpu_index)
    dev = torch.device(f"cuda:{args.physical_gpu_index}")
    cuda_name = torch.cuda.get_device_name(args.physical_gpu_index)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_rows("train")
    val_rows = load_rows("validation")
    L = max(len(r["source_sequence"]) for r in train_rows + val_rows)
    all_pos = [p for r in train_rows + val_rows for p, _ in edits_of(r) if p is not None]
    L = max(L, (max(all_pos) if all_pos else 0) + 1)
    n_blocks = (L + 7) // 8
    n_ctx = 64
    F = n_blocks * n_ctx
    print(f"MRL train={len(train_rows)} val={len(val_rows)} L={L} blocks={n_blocks} F={F}", flush=True)

    Xtr, skip_tr = build_features(train_rows, n_blocks)
    Xva, skip_va = build_features(val_rows, n_blocks)
    ytr = np.array([float(r["direction_normalized_delta"]) for r in train_rows], dtype=np.float32)
    yva = np.array([float(r["direction_normalized_delta"]) for r in val_rows], dtype=np.float32)
    print(f"feature rows: train {Xtr.shape}, val {Xva.shape}; zero-feature rows train={skip_tr} val={skip_va}", flush=True)

    src_ids_tr = [str(r["source_id"]) for r in train_rows]
    src_ids_va = [str(r["source_id"]) for r in val_rows]
    all_srcs = sorted(set(src_ids_tr) | set(src_ids_va))
    src_index = {s: i for i, s in enumerate(all_srcs)}
    n_src = len(all_srcs)

    # warm start for F: ridgeless-ish init not needed; init W=0, b=shrunk source mean (probe-style)
    src_mean = defaultdict(list)
    for s, y in zip(src_ids_tr, ytr):
        src_mean[s].append(float(y))
    grand = float(np.mean(ytr))
    b_init = np.array([(np.mean(src_mean.get(s, [])) * len(src_mean.get(s, [])) + grand * 5) /
                       (len(src_mean.get(s, [])) + 5) if src_mean.get(s) else grand for s in all_srcs],
                      dtype=np.float32)

    W = torch.zeros(F, dtype=torch.float32, device=dev)
    b = torch.from_numpy(b_init).to(dev)
    W.requires_grad_(True)
    b.requires_grad_(True)
    n_params = F + n_src
    print(f"model params: W={F} + b={n_src} = {n_params} (band {n_params/1e3:.1f}K)", flush=True)

    Xtr_t = torch.from_numpy(Xtr).to(dev, dtype=torch.float32)
    Xva_t = torch.from_numpy(Xva).to(dev, dtype=torch.float32)
    ytr_t = torch.from_numpy(ytr).to(dev, dtype=torch.float32)
    yva_t = torch.from_numpy(yva).to(dev, dtype=torch.float32)
    idx_tr = torch.tensor([src_index[s] for s in src_ids_tr], device=dev)
    idx_va = torch.tensor([src_index[s] for s in src_ids_va], device=dev)

    opt = torch.optim.Adam([W, b], lr=args.lr)
    huber = torch.nn.HuberLoss(delta=1.0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_events = open(out_dir / "training_losses.jsonl", "w")
    epoch_metrics_file = open(out_dir / "epoch_metrics.jsonl", "w")

    def predict(Xt, bi):
        return Xt @ W + b[bi]

    def spearman(pred, y):
        pr = pred.detach().cpu().numpy()
        yy = y.detach().cpu().numpy()
        from scipy.stats import spearmanr
        return float(spearmanr(pr, yy).statistic)

    for epoch in range(1, args.epochs + 1):
        n_steps = args.smoke_steps or 1
        for _ in range(n_steps):
            opt.zero_grad()
            pred = predict(Xtr_t, idx_tr)
            loss = huber(pred / V5_TARGET_SCALE, ytr_t / V5_TARGET_SCALE)
            loss.backward()
            opt.step()
        with torch.inference_mode():
            ptr = predict(Xtr_t, idx_tr)
            pva = predict(Xva_t, idx_va)
            rho_tr = spearman(ptr, ytr_t)
            rho_va = spearman(pva, yva_t)
        primary = epoch == args.epochs
        rec = {"epoch": epoch, "primary": primary, "loss_mean": float(loss.item()),
               "mrl_train_spearman": rho_tr, "mrl_val_spearman": rho_va,
               "polyA_reference_untouched_v5": V5_POLYA_VAL,
               "probe_polya_context_ref": PROBE_POLYA_CONTEXT}
        epoch_metrics_file.write(json.dumps(rec) + "\n")
        epoch_metrics_file.flush()
        train_events.write(json.dumps({"epoch": epoch, "loss": float(loss.item()), "step": n_steps}) + "\n")
        train_events.flush()
        mrk = " FINAL" if primary else ""
        print("epoch {}{}: loss {:.4f} train_rho {:.4f} val_rho {:.4f}".format(epoch, mrk, float(loss), rho_tr, rho_va), flush=True)
        torch.save({"model_state_dict": {"W": W.detach().cpu(), "b": b.detach().cpu()},
                    "epoch": epoch, "seed": args.seed,
                    "schema": "route_a_v3_erk_v2_train.v1",
                    "n_blocks": n_blocks, "n_ctx": n_ctx,
                    "source_index": {s: i for i, s in enumerate(all_srcs)},
                    "target_scale": V5_TARGET_SCALE}, out_dir / f"erk_epoch_{epoch}.pt")

    summary = {
        "schema_version": "route_a_v3_erk_v2_train.v1",
        "status": "ERK_V2_TRAIN_COMPLETE",
        "seed": args.seed,
        "epochs": args.epochs,
        "n_params": n_params,
        "n_blocks": n_blocks, "n_ctx": n_ctx,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "zero_feature_train_rows": skip_tr, "zero_feature_val_rows": skip_va,
        "cpu_fallback_used": False,
        "cuda_verified": True,
        "cuda_device_index": args.physical_gpu_index,
        "cuda_device_name": cuda_name,
        "final_epoch": args.epochs,
        "final_mrl_val_spearman": None,
        "target_scale": V5_TARGET_SCALE,
        "polyA_reference_untouched_v5": V5_POLYA_VAL,
        "protected_reads": 0,
    }
    final_rows = []
    epoch_metrics_file.close()
    for line in Path(out_dir / "epoch_metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("primary"):
            final_rows.append(r)
    if len(final_rows) != 1:
        print(f"HARD_REJECT: FINAL-EPOCH marker count = {len(final_rows)}", flush=True)
        return 2
    summary["final_mrl_val_spearman"] = final_rows[0]["mrl_val_spearman"]
    summary["final_mrl_train_spearman"] = final_rows[0]["mrl_train_spearman"]
    (out_dir / "erk_train_summary.json").write_text(json.dumps(summary, indent=1))
    print("summary:", json.dumps(summary, indent=1)[:600], flush=True)
    print("wrote", str(out_dir / "erk_train_summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
