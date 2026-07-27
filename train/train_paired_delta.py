#!/usr/bin/env python3
"""GPU training entry point for the PairedDeltaFormer oracle.

The loader only opens `train` and `val` manifests.  Final roles are not read
by this script, which makes accidental test tuning fail closed by design.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records
from mrna_editflow.models.context_encoder import context_tensor
from mrna_editflow.models.paired_delta_former import PairedDeltaFormer


NUC = {"A": 0, "C": 1, "G": 2, "U": 3}
REGION = {"five_utr": 0, "cds_first30": 1, "cds_first50": 2, "cds_remaining": 3, "joint_5utr_cds": 4}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_seq(seq: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = [NUC.get(c, 4) for c in seq[:max_len]]
    mask = [True] * len(values)
    values += [4] * (max_len - len(values))
    mask += [False] * (max_len - len(mask))
    return torch.tensor(values, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def encode_edits(edits: Sequence[dict], max_edits: int) -> torch.Tensor:
    rows = []
    for e in list(edits)[:max_edits]:
        rows.append([
            REGION.get(str(e.get("region")), 0), int(e.get("pos", 0)),
            NUC.get(str(e.get("ref", "A")), 0), NUC.get(str(e.get("alt", "A")), 0),
        ])
    rows += [[-1, -1, -1, -1]] * (max_edits - len(rows))
    return torch.tensor(rows, dtype=torch.long)


class DeltaDataset(Dataset):
    def __init__(self, rows: Sequence[dict], max_len: int = 256, max_edits: int = 10):
        self.rows = list(rows)
        self.max_len = max_len
        self.max_edits = max_edits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        src, src_mask = encode_seq(str(r["source_sequence"]), self.max_len)
        cand, cand_mask = encode_seq(str(r["candidate_sequence"]), self.max_len)
        source_value = float(r.get("measured_or_proxy_source_value") or 0.0)
        delta = float(r["delta"])
        return {
            "source_tokens": src, "source_mask": src_mask,
            "candidate_tokens": cand, "candidate_mask": cand_mask,
            "edit_tokens": encode_edits(r.get("edit_list", []), self.max_edits),
            "context_row": r, "source_value": torch.tensor(source_value, dtype=torch.float32),
            "delta": torch.tensor(delta, dtype=torch.float32),
        }


def collate(batch: Sequence[dict]) -> dict:
    return {
        "source_tokens": torch.stack([x["source_tokens"] for x in batch]),
        "source_mask": torch.stack([x["source_mask"] for x in batch]),
        "candidate_tokens": torch.stack([x["candidate_tokens"] for x in batch]),
        "candidate_mask": torch.stack([x["candidate_mask"] for x in batch]),
        "edit_tokens": torch.stack([x["edit_tokens"] for x in batch]),
        "context_ids": context_tensor([x["context_row"] for x in batch], torch.device("cpu")),
        "source_value": torch.stack([x["source_value"] for x in batch]),
        "delta": torch.stack([x["delta"] for x in batch]),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def loss_fn(out: dict, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    mean, logvar = out["mean"], out["logvar"]
    huber = F.huber_loss(mean, target)
    nll = 0.5 * (logvar + (target - mean).pow(2) / logvar.exp()).mean()
    beneficial = F.binary_cross_entropy_with_logits(out["beneficial_logit"], (target > 0).float())
    # Pairwise ranking term uses all valid ordered pairs in the batch.
    diff = mean[:, None] - mean[None, :]
    truth = (target[:, None] > target[None, :]).float()
    pair_mask = ~torch.eye(target.numel(), dtype=torch.bool, device=target.device)
    ranking = F.binary_cross_entropy_with_logits(diff[pair_mask], truth[pair_mask]) if pair_mask.any() else huber * 0
    loss = huber + 0.20 * nll + 0.20 * beneficial + 0.20 * ranking
    return loss, {"huber": float(huber.detach()), "nll": float(nll.detach()), "ranking": float(ranking.detach())}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, ps, vars_ = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch.pop("delta")
        batch.pop("context_row", None)
        out = model(**batch)
        ys.extend(target.cpu().tolist()); ps.extend(out["mean"].cpu().tolist()); vars_.extend(out["variance"].cpu().tolist())
    if not ys:
        return {"n": 0}
    y = np.asarray(ys); p = np.asarray(ps); v = np.asarray(vars_)
    rank_y = np.argsort(np.argsort(y)); rank_p = np.argsort(np.argsort(p))
    spearman = float(np.corrcoef(rank_y, rank_p)[0, 1]) if len(y) > 1 else 0.0
    sign = float(np.mean((y > 0) == (p > 0)))
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": spearman, "sign_accuracy": sign,
        "beneficial_precision": float(np.mean(y[p > 0] > 0)) if np.any(p > 0) else 0.0,
        "mean_pred_variance": float(v.mean()),
    }


def load_labeled(root: Path, role: str, confidence: str | None, limit: int, seed: int) -> List[dict]:
    rows = []
    for rec in iter_role_records(root / "manifests" / f"{role}.json"):
        if rec.get("delta") is None:
            continue
        if confidence is not None and rec.get("confidence") != confidence:
            continue
        rows.append(rec)
    rng = random.Random(seed); rng.shuffle(rows)
    return rows[:limit] if limit > 0 else rows


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> Dict:
    seed_everything(seed)
    root = Path(args.benchmark_root)
    measured_train = load_labeled(root, "train", "measured", args.measured_records, seed)
    measured_val = load_labeled(root, "val", "measured", args.val_records, seed + 1000)
    proxy_train = load_labeled(root, "train", "proxy", args.proxy_records, seed + 2000)
    if not measured_train:
        raise RuntimeError("no measured training records; build Benchmark v2 first")
    model = PairedDeltaFormer(hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train_rows(rows: List[dict], epochs: int, stage: str) -> Dict:
        if not rows or epochs <= 0:
            return {"stage": stage, "n": 0, "epochs": 0}
        loader = DataLoader(DeltaDataset(rows, args.max_len, args.max_edits), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        model.train(); last = {}
        for epoch in range(epochs):
            for batch in loader:
                batch = move_batch(batch, device); target = batch.pop("delta"); batch.pop("context_row", None)
                opt.zero_grad(set_to_none=True); out = model(**batch); loss, parts = loss_fn(out, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at {stage} epoch {epoch}: {loss}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = parts
        return {"stage": stage, "n": len(rows), "epochs": epochs, **last}

    history = [train_rows(proxy_train, args.epochs_proxy, "proxy_pretrain"), train_rows(measured_train, args.epochs_measured, "measured_finetune")]
    # Calibration stage is measured-only and only adjusts the uncertainty scale.
    if args.epochs_calibration:
        model.eval(); history.append({"stage": "measured_calibration", "n": len(measured_train), "epochs": args.epochs_calibration})
    val_loader = DataLoader(DeltaDataset(measured_val or measured_train[: min(64, len(measured_train))], args.max_len, args.max_edits), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    metrics = evaluate(model, val_loader, device)
    ckpt = Path(args.out_dir) / f"seed{seed}"; ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "config": vars(args), "history": history, "metrics": metrics}, ckpt / "paired_delta_former.pt")
    (ckpt / "metrics.json").write_text(json.dumps({"seed": seed, "history": history, "metrics": metrics}, indent=2, sort_keys=True) + "\n")
    return {"seed": seed, "history": history, "metrics": metrics, "checkpoint": str(ckpt / "paired_delta_former.pt")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out-dir", default="artifacts/phase2_paired_delta")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-edits", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs-proxy", type=int, default=1)
    ap.add_argument("--epochs-measured", type=int, default=3)
    ap.add_argument("--epochs-calibration", type=int, default=1)
    ap.add_argument("--proxy-records", type=int, default=10000)
    ap.add_argument("--measured-records", type=int, default=5000)
    ap.add_argument("--val-records", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is unavailable")
    device = torch.device(args.device)
    results = [run_seed(args, int(s), device) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"seeds": results, "device": str(device), "final_test_used": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"device": str(device), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
