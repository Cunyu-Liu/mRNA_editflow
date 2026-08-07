"""E0-X development: selective/coverage-conditional sign-accuracy frontier.

Phase E0-X — PURE DEVELOPMENT VALIDATION (pre-unblinding amendment exploration).

This script does NOT access GSE246381 sealed labels, does NOT modify the frozen
M4 critic, and does NOT change any frozen threshold.  It answers one question:

  * Under the contract's T5 Selective-prediction mechanism (abstain + report
    coverage-risk instead of forcing a sign on every OOD edit), can the frozen
    delta critic achieve macro sign accuracy >= 0.60 on a coverage-conditional
    high-confidence subset of the ordinary (non-sealed) 5U-A1 held-out folds?

It reuses the SAME frozen delta critic checkpoints, SAME S4 leave-one-study-out
folds, and SAME test rows as the E0-X ordinary internal test.  Confidence is the
heteroscedastic predictive variance (logvar head); a pre-registered coverage
floor selects the highest-confidence fraction per study.  We report the full
sign-accuracy-vs-coverage frontier so the honest trade-off is visible BEFORE any
amendment is frozen.

No GO/NO-GO verdict is produced here; this is evidence gathering for a possible
pre-unblinding contract amendment (decision-logged, not a post-hoc threshold
change).  This mirrors the contract's T5: abstention + coverage-risk, not
forcing high-confidence answers on all OOD edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys
_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.m4_sparse import config as C
from scripts.m4_sparse.dataset import EffectDataset, build_vocab
from scripts.m4_sparse.model import SparseEditFormer
from scripts.m4_sparse.train import build_folds

DATASET = "artifacts/b0x/effect_dataset.jsonl"
BENCH = "5U-A1"
CKPT_DIR = "artifacts/m4_sparse/delta"   # frozen delta critic
SEED = 42
BATCH_SIZE = 256


def select_device(cfg, override):
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA unavailable"
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            return None, "forbidden " + override
        return torch.device(override), None
    for dev in cfg.CUDA_DEVICES:
        if dev.split(":")[-1] in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev), None
    if "0" not in cfg.FORBIDDEN_DEVICES:
        return torch.device("cuda:0"), None
    return None, "no permitted CUDA device"


def load_critic(ckpt_path, device):
    import torch
    sd = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    ck = sd["cfg"]
    model = SparseEditFormer(ck).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict_delta_and_logvar(model, test_rows, vocab, cfg, device):
    """Return per-row (delta_pred, predictive_variance=exp(logvar), true_delta)."""
    import torch
    from torch.utils.data import DataLoader
    ds = EffectDataset(test_rows, vocab, target=cfg.TARGET)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds, vars_, true_ = [], [], []
    for b in loader:
        bb = {k: v.to(device) for k, v in b.items()}
        out = model(bb["src"], bb["cand"], bb["edit"], bb["study"],
                    bb["endpoint"], bb["bench"])
        delta = out["mean"].cpu().numpy()
        if cfg.TARGET == "candidate_value" and cfg.ANCHOR_AT_TEST:
            delta = delta - bb["source_value"].cpu().numpy()
        preds.append(delta)
        vars_.append(np.exp(out["logvar"].cpu().numpy()))
        true_.append(bb["y"].cpu().numpy())
    return (np.concatenate(preds), np.concatenate(vars_),
            np.concatenate(true_))


def selective_sign_accuracy(pred, var, true, coverage: float):
    """Sign accuracy on the highest-confidence coverage fraction.

    Confidence = lower predictive variance (exp(logvar)).  We select the
    `coverage`-fraction of rows with the SMALLEST variance.  Only rows with
    nonzero true delta are scored (mirror context_metrics).
    """
    nz = true != 0
    if nz.sum() == 0:
        return None, 0.0, 0
    pred_nz, var_nz, true_nz = pred[nz], var[nz], true[nz]
    k = max(1, int(round(coverage * len(true_nz))))
    k = min(k, len(true_nz))
    # lowest variance = highest confidence
    keep = np.argsort(var_nz, kind="mergesort")[:k]
    acc = float(np.mean(np.sign(pred_nz[keep]) == np.sign(true_nz[keep])))
    return acc, coverage, int(k)


def run_frontier(args, cfg, device):
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    rows = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if r.get("delta") is not None:
                rows.append(r)
    bench_rows = [r for r in rows if r["benchmark"] == BENCH]
    vocab = build_vocab(bench_rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])

    folds = build_folds(bench_rows, BENCH, split=cfg.PRIMARY_SPLIT)

    # coverage grid (development exploration; nothing frozen here)
    coverages = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.0]

    per_fold = {}
    macro = {c: [] for c in coverages}
    for fold in folds:
        held = fold["held_out_study"]
        ckpt = Path(args.ckpt_dir) / ("model_%s__%s.pt" % (BENCH, held))
        if not ckpt.exists():
            print("[selective-dev] MISSING %s -> skip" % ckpt)
            continue
        model, ck = load_critic(ckpt, device)
        # honor the frozen critic's eval contract (TARGET/ANCHOR from checkpoint)
        cfg.TARGET = getattr(ck, "TARGET", cfg.TARGET)
        cfg.ANCHOR_AT_TEST = getattr(ck, "ANCHOR_AT_TEST", cfg.ANCHOR_AT_TEST)
        pred, var, true = predict_delta_and_logvar(model, fold["test"], vocab, cfg, device)

        pf = {}
        for c in coverages:
            acc, cov, k = selective_sign_accuracy(pred, var, true, c)
            pf[c] = {"sign_accuracy": acc, "coverage": cov, "n_scored": k,
                     "n_total": int((true != 0).sum())}
            if acc is not None:
                macro[c].append(acc)
        per_fold[held] = pf
        print("[selective-dev] %s n_test=%d" % (held, len(fold["test"])), flush=True)

    summary = {
        "phase": "E0X_SELECTIVE_SIGN_DEV",
        "prereg": "E0X_PREREG_20260807",
        "benchmark": BENCH,
        "split": cfg.PRIMARY_SPLIT,
        "model": "frozen delta critic (artifacts/m4_sparse/delta)",
        "coverage_grid": coverages,
        "macro_selective_sign_accuracy_by_coverage": {
            str(c): float(np.mean(macro[c])) if macro[c] else None for c in coverages
        },
        "per_fold": per_fold,
        "note": (
            "DEV ONLY: no sealed access, no threshold change, no verdict. "
            "Evidence for a possible pre-unblinding T5 selective-prediction "
            "amendment.  Sign accuracy on the lowest-variance coverage subset.")
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "e0x_selective_sign_frontier.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote %s" % out)
    print("macro selective sign_acc by coverage:")
    for c in coverages:
        v = summary["macro_selective_sign_accuracy_by_coverage"][str(c)]
        print("  coverage=%.2f -> sign_acc=%s" % (c, ("%.4f" % v) if v is not None else "None"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default="artifacts/e0x/selective_dev")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args()
    args.out_dir = Path(args.out)

    cfg = C.get_config()
    device, err = select_device(cfg, args.gpu)
    if device is None:
        print("GPU policy fail-closed: %s" % err, file=sys.stderr)
        return 3
    run_frontier(args, cfg, device)
    return 0


if __name__ == "__main__":
    import torch
    raise SystemExit(main())
