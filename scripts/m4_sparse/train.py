"""S4 leave-one-study-out training for SparseEditFormer.

For each benchmark (5U-A1, 3U-A1) the held-out study is left out; the model is
trained on the remaining studies (A1-NATURAL; no A2 dense pretraining exists
because EditBench-5U-A2-Dense is DORMANT by governance).  A seeded dev split of
the train fold is used for early stopping and sign-head calibration.  The test
fold (held-out study) is never used for training or early stopping.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import EffectDataset
from .model import SparseEditFormer


def build_folds(rows: List[dict], benchmark: str, split: str = "S4") -> List[Dict]:
    """S4 leave-one-study-out folds for a benchmark.  Train=all other studies,
    test=held-out study (no source/study overlap)."""
    bench_rows = [r for r in rows if r["benchmark"] == benchmark]
    studies = sorted({r["study"] for r in bench_rows})
    folds = []
    for held in studies:
        train = [r for r in bench_rows if r["study"] != held]
        test = [r for r in bench_rows if r["study"] == held]
        folds.append({"held_out_study": held, "train": train, "test": test,
                      "split": split})
    return folds


def _pairwise_rank_loss(score: torch.Tensor, target: torch.Tensor,
                        margin: float, device) -> torch.Tensor:
    n = score.size(0)
    if n < 2:
        return torch.tensor(0.0, device=device)
    perm = torch.randperm(n, device=device)
    diff_s = score - score[perm]
    diff_y = target - target[perm]
    m = diff_y != 0
    if m.sum() == 0:
        return torch.tensor(0.0, device=device)
    sign = torch.sign(diff_y[m])
    err = torch.nn.functional.relu(margin - sign * diff_s[m])
    return err.mean()


def compute_loss(model, batch, cfg, device) -> Dict[str, torch.Tensor]:
    out = model(batch["src"], batch["cand"], batch["edit"],
                batch["study"], batch["endpoint"], batch["bench"])
    y = batch["y"]
    # heteroscedastic NLL (mean/variance head)
    var = torch.nn.functional.softplus(out["logvar"]) + 1e-6
    nll = 0.5 * ((y - out["mean"]) ** 2 / var + torch.log(var))
    nll = nll.mean()
    # sign head (BCE) on nonzero deltas
    mask = y != 0
    if mask.sum() > 0:
        bce = F.binary_cross_entropy_with_logits(out["sign"][mask], (y[mask] > 0).float())
    else:
        bce = torch.tensor(0.0, device=device)
    # inverse consistency: f(cand->src) should be ~ -f(src->cand)
    out_inv = model(batch["cand"], batch["src"], batch["inv_edit"],
                    batch["study"], batch["endpoint"], batch["bench"])
    consis = torch.mean((out["mean"] + out_inv["mean"]) ** 2)
    # pairwise ranking head
    rank_loss = _pairwise_rank_loss(out["rank"], y, cfg.MARGIN, device)

    loss = (cfg.LAMBDA_MEAN * nll + cfg.LAMBDA_VAR * nll
            + cfg.LAMBDA_SIGN * bce + cfg.LAMBDA_CONSISTENCY * consis
            + cfg.LAMBDA_RANK * rank_loss)
    return {"loss": loss, "nll": nll, "bce": bce, "consis": consis,
            "rank": rank_loss}


def _train_epoch(model, loader, opt, cfg, device) -> float:
    model.train()
    tot, nb = 0.0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        losses = compute_loss(model, b, cfg, device)
        losses["loss"].backward()
        opt.step()
        tot += losses["loss"].item()
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def _eval_loss(model, loader, cfg, device, target_delta: bool) -> float:
    model.eval()
    tot, nb = 0.0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b["src"], b["cand"], b["edit"], b["study"], b["endpoint"],
                    b["bench"])
        y = b["y"]
        mse = F.mse_loss(out["mean"], y)
        tot += mse.item()
        nb += 1
    return tot / max(nb, 1)


def train_fold(fold: Dict, cfg, vocab: Dict, device, out_path: Optional[Path] = None,
               seed: Optional[int] = None) -> SparseEditFormer:
    torch.manual_seed(int(seed if seed is not None else cfg.SEED))
    np.random.seed(int(seed if seed is not None else cfg.SEED))
    train_rows = fold["train"]
    if cfg.MAX_TRAIN_CAP and len(train_rows) > cfg.MAX_TRAIN_CAP:
        rng = np.random.RandomState(cfg.SEED)
        keep = rng.choice(len(train_rows), cfg.MAX_TRAIN_CAP, replace=False)
        train_rows = [train_rows[i] for i in keep]

    rng = np.random.RandomState(cfg.SEED)
    idx = rng.permutation(len(train_rows))
    n_dev = max(int(len(train_rows) * cfg.DEV_FRAC), 1)
    dev_idx, tt_idx = idx[:n_dev], idx[n_dev:]
    tt = [train_rows[i] for i in tt_idx]
    dev = [train_rows[i] for i in dev_idx]

    train_ds = EffectDataset(tt, vocab, target=cfg.TARGET)
    dev_ds = EffectDataset(dev, vocab, target=cfg.TARGET)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=0)

    model = SparseEditFormer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_dev, best_state = float("inf"), None
    for epoch in range(cfg.EPOCHS):
        tr = _train_epoch(model, train_loader, opt, cfg, device)
        dv = _eval_loss(model, dev_loader, cfg, device, target_delta=(cfg.TARGET == "delta"))
        if dv < best_dev:
            best_dev = dv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "cfg": cfg, "held_out_study": fold["held_out_study"]},
                   out_path)
    return model


def calibrate_sign_temperature(model, dev_rows, vocab, cfg, device) -> float:
    """Fit a single temperature on the sign logits (dev fold) so the sign
    probabilities are calibrated (Platt/temperature).  Returns temperature T."""
    model.eval()
    ds = EffectDataset(dev_rows, vocab, target=cfg.TARGET)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for b in loader:
            bb = {k: v.to(device) for k, v in b.items()}
            out = model(bb["src"], bb["cand"], bb["edit"], bb["study"],
                        bb["endpoint"], bb["bench"])
            mask = bb["y"] != 0
            if mask.sum() > 0:
                logits.append(out["sign"][mask].cpu())
                labels.append((bb["y"][mask] > 0).float().cpu())
    if not logits:
        return 1.0
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    T = torch.tensor(1.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=30)
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits.to(device) / T, labels.to(device))
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(min=0.1, max=10.0).item())
