"""X0-X development: 3'UTR RegionAdapter training + evaluation on 3U-A1.

Phase X0-X (3'UTR & CDS transfer) — PURE DEVELOPMENT TRAINING.  This script:
  * does NOT touch the frozen 5' primary model (F0-X base flow / M4 critic),
  * does NOT access GSE246381 sealed labels,
  * does NOT trigger the formal X0-X transfer gate,
  * claims NO measured cross-region transfer result.

It trains a small source-relative backbone + the X0-X `RegionAdapter`
(scripts/x0x/region.py) on the 3U-A1 benchmark only, with S4 leave-one-study-out
over the 7 3'UTR studies and independent 3' endpoint heads.  Because every 3U-A1
source has exactly one measured candidate (no measured search neighborhood), no
ranking headline is computed — the honest evaluation is 3'UTR delta-effect
transfer (Spearman / sign-accuracy / top-decile enrichment) per held-out study,
with the 5' vs 3' heads kept structurally independent.

The encoding conventions mirror M4 (NUC_ORDER="ACGU", MAX_SEQ_LEN=100, 12-dim
edit features) so results are comparable to B0-X / M4.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset

# ---- reuse M4 encoding + dataset primitives ----
import sys
_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts.m4_sparse.dataset import (EffectDataset, edit_features, one_hot,
                                        build_vocab)
from scripts.x0x import region

# ---- encoding conventions (mirror scripts/m4_sparse/config.py) ----
NUC_ORDER = "ACGU"
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
MAX_SEQ_LEN = 100
EDIT_FEAT_DIM = 12

# ---- split / seed ----
PRIMARY_SPLIT = "S4"     # leave-one-study-out (3'UTR studies)
SEED = 42

# ---- model ----
HIDDEN_DIM = 64
NHEAD = 4
N_LAYERS = 2
DIM_FF = 128
CONV_KS = 5
DROP = 0.1

# ---- training ----
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 4
DEV_FRAC = 0.15
WEIGHT_DECAY = 1e-5

DATASET = "artifacts/b0x/effect_dataset.jsonl"


class _Cfg:
    """Minimal attribute container mirroring the M4 config surface used by
    SparseEditFormer's encoder (hidden/nhead/layers/ff/ks/drop/emb counts)."""
    def __init__(self, n_studies, n_endpoints, n_benchmarks):
        self.HIDDEN_DIM = HIDDEN_DIM
        self.NHEAD = NHEAD
        self.N_LAYERS = N_LAYERS
        self.DIM_FF = DIM_FF
        self.CONV_KS = CONV_KS
        self.DROP = DROP
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.EDIT_FEAT_DIM = EDIT_FEAT_DIM
        self.N_STUDIES = n_studies
        self.N_ENDPOINTS = n_endpoints
        self.N_BENCHMARKS = n_benchmarks


class SparseBackbone(nn.Module):
    """Source-relative backbone producing a context vector z (B,d).

    Reuses the M4 encoder structure (stem + source/candidate self-attn +
    cross-attn + edit MLP + context conditioning) but returns the raw context
    vector `z` instead of applying frozen effect heads.  The X0-X RegionAdapter
    then routes z to independent per-region effect heads.
    """
    def __init__(self, cfg: _Cfg):
        super().__init__()
        d = cfg.HIDDEN_DIM
        self.stem = nn.Sequential(
            nn.Conv1d(4, d, cfg.CONV_KS, padding=cfg.CONV_KS // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d, d, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, cfg.MAX_SEQ_LEN, d))
        nn.init.normal_(self.pos, std=0.02)
        src_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.src_encoder = nn.TransformerEncoder(src_layer, num_layers=cfg.N_LAYERS)
        cand_layer = nn.TransformerEncoderLayer(
            d, cfg.NHEAD, cfg.DIM_FF, dropout=cfg.DROP, batch_first=True,
            activation="gelu")
        self.cand_encoder = nn.TransformerEncoder(cand_layer, num_layers=1)
        self.cross_attn = nn.MultiheadAttention(d, cfg.NHEAD, dropout=cfg.DROP,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pool_proj = nn.Linear(d * 2, d)
        self.edit_mlp = nn.Sequential(
            nn.Linear(cfg.EDIT_FEAT_DIM, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.study_emb = nn.Embedding(max(cfg.N_STUDIES, 1), d)
        self.endpoint_emb = nn.Embedding(max(cfg.N_ENDPOINTS, 1), d)
        self.bench_emb = nn.Embedding(max(cfg.N_BENCHMARKS, 1), d)

    def forward(self, src, cand, edit_feat, study_id, endpoint_id, bench_id):
        hs = self.stem(src.permute(0, 2, 1)).permute(0, 2, 1) + self.pos
        hs = self.src_encoder(hs)
        hc = self.stem(cand.permute(0, 2, 1)).permute(0, 2, 1) + self.pos
        hc = self.cand_encoder(hc)
        cross, _ = self.cross_attn(hc, hs, hs)
        hc = self.norm(hc + cross)
        pooled = torch.cat([hc.mean(dim=1), hc.max(dim=1).values], dim=-1)
        vec = self.pool_proj(pooled)
        ev = self.edit_mlp(edit_feat)
        ctx = (self.study_emb(study_id) + self.endpoint_emb(endpoint_id)
               + self.bench_emb(bench_id))
        return vec + ev + ctx


class RegionModel(nn.Module):
    """SparseBackbone + RegionAdapter composition.

    forward() returns the backbone context vector z; the RegionAdapter is kept
    as a named submodule `region_adapter` so effect heads stay independent.
    """
    def __init__(self, backbone: SparseBackbone, adapter: "region.RegionAdapter"):
        super().__init__()
        self.backbone = backbone
        self.region_adapter = adapter

    def forward(self, src, cand, edit_feat, study_id, endpoint_id, bench_id):
        return self.backbone(src, cand, edit_feat, study_id, endpoint_id, bench_id)


def build_3u_folds(rows: List[dict], split: str = "S4") -> List[Dict]:
    """S4 leave-one-study-out folds over 3U-A1 records only."""
    bench = [r for r in rows if r["benchmark"] == "3U-A1"]
    studies = sorted({r["study"] for r in bench})
    folds = []
    for held in studies:
        train = [r for r in bench if r["study"] != held]
        test = [r for r in bench if r["study"] == held]
        folds.append({"held_out_study": held, "train": train, "test": test,
                      "split": split})
    return folds


@torch.no_grad()
def evaluate_3u(model, rows, vocab, device) -> Dict:
    """Evaluate 3'UTR delta-effect on rows: Spearman / sign-acc / top-decile
    enrichment using the RegionAdapter's 3' mean head (source-anchored delta)."""
    model.eval()
    ds = EffectDataset(rows, vocab, target="delta")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # 3UTR region id = 1
    region_id = region.RegionAdapter._region_idx(region.REGION_3UTR)
    ys, preds, signs = [], [], []
    with torch.no_grad():
        for b in loader:
            src, cand = b["src"].to(device), b["cand"].to(device)
            edit = b["edit"].to(device)
            study, ep, bench = (b["study"].to(device), b["endpoint"].to(device),
                                b["bench"].to(device))
            z = model(src, cand, edit, study, ep, bench)
            reg = torch.full((src.size(0),), region_id, dtype=torch.long,
                             device=device)
            out = model.region_adapter(z, reg, ep)
            y = b["y"].numpy()
            mean = out["mean"].cpu().numpy()
            rank = out["rank"].cpu().numpy()
            mask = y != 0
            ys.append(y[mask])
            preds.append(mean[mask])
            signs.append(rank[mask])
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    r = np.concatenate(signs)
    if y.size < 8:
        return {"n": int(y.size), "spearman": float("nan"), "sign_acc": float("nan"),
                "top10_enrichment": float("nan")}
    spearman = float(stats.spearmanr(p, y).statistic if y.size >= 8 else float("nan"))
    sign_acc = float(np.mean(np.sign(p) == np.sign(y)))
    # top-decile enrichment: mean predicted-top-10% true delta / overall mean
    k = max(int(round(0.10 * y.size)), 1)
    order = np.argsort(-r)[:k]
    top_mean = float(np.mean(y[order]))
    overall = float(np.mean(y))
    enrich = (top_mean / overall) if overall != 0 else float("nan")
    return {"n": int(y.size), "spearman": spearman, "sign_acc": sign_acc,
            "top10_enrichment": enrich}


def train_fold(fold: Dict, vocab: Dict, device, out_dir: Path) -> Dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rng = np.random.RandomState(SEED)
    train_rows = fold["train"]
    idx = rng.permutation(len(train_rows))
    n_dev = max(int(len(train_rows) * DEV_FRAC), 1)
    dev_idx, tt_idx = idx[:n_dev], idx[n_dev:]
    tt, dev = [train_rows[i] for i in tt_idx], [train_rows[i] for i in dev_idx]

    vocab_cfg = _Cfg(len(vocab["study"]), len(vocab["endpoint"]),
                     len(vocab["benchmark"]))
    backbone = SparseBackbone(vocab_cfg).to(device)
    # region adapter: 3' endpoint head space sized to 3U endpoints
    n_3u_eps = len({r["endpoint"] for r in tt + dev + fold["test"]})
    adapter = region.RegionAdapter(
        region.build_region_config(n_3u_endpoints=n_3u_eps, region=region.REGION_3UTR,
                                   hidden=HIDDEN_DIM)).to(device)
    model = RegionModel(backbone, adapter).to(device)

    opt = torch.optim.AdamW(list(backbone.parameters())
                            + list(adapter.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)

    train_ds = EffectDataset(tt, vocab, target="delta")
    dev_ds = EffectDataset(dev, vocab, target="delta")
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                    drop_last=True)
    dl = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    region_id = region.RegionAdapter._region_idx(region.REGION_3UTR)
    best_dev, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for b in tl:
            src, cand = b["src"].to(device), b["cand"].to(device)
            edit, ep = b["edit"].to(device), b["endpoint"].to(device)
            study, bench = b["study"].to(device), b["bench"].to(device)
            y = b["y"].to(device)
            reg = torch.full((src.size(0),), region_id, dtype=torch.long,
                             device=device)
            opt.zero_grad()
            z = backbone(src, cand, edit, study, ep, bench)
            out = adapter(z, reg, ep)
            var = F.softplus(out["logvar"]) + 1e-6
            nll = (0.5 * ((y - out["mean"]) ** 2 / var + torch.log(var))).mean()
            nll.backward()
            opt.step()
            tot += nll.item()
            nb += 1
        # dev MSE for early stopping
        model.eval()
        dms = 0.0
        with torch.no_grad():
            for b in dl:
                src, cand = b["src"].to(device), b["cand"].to(device)
                edit, ep = b["edit"].to(device), b["endpoint"].to(device)
                study, bench = b["study"].to(device), b["bench"].to(device)
                y = b["y"].to(device)
                reg = torch.full((src.size(0),), region_id, dtype=torch.long,
                                 device=device)
                z = backbone(src, cand, edit, study, ep, bench)
                out = adapter(z, reg, ep)
                dms += F.mse_loss(out["mean"], y).item()
        dms /= max(nb, 1)
        if dms < best_dev:
            best_dev = dms
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # independent-head structural guard
    independent = adapter.has_independent_heads()

    ev = evaluate_3u(model, fold["test"], vocab, device)
    dev_ev = evaluate_3u(model, dev, vocab, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "held_out_study": fold["held_out_study"]},
               out_dir / f"region_3u_fold_{fold['held_out_study']}.pt")
    return {"held_out_study": fold["held_out_study"],
            "n_train": len(tt), "n_dev": len(dev), "n_test": len(fold["test"]),
            "independent_heads": independent,
            "dev_mse": float(best_dev), "test": ev, "dev": dev_ev}


def _macro(folds: List[Dict], key: str) -> float:
    vals = [f["test"][key] for f in folds if not np.isnan(f["test"][key])]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="cuda:3", help="GPU device (default cuda:3)")
    ap.add_argument("--out", default="artifacts/x0x/region_3u_dev",
                    help="output dir under repo root")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with open(args.dataset) as f:
        for line in f:
            r = json.loads(line)
            if r["benchmark"] != "3U-A1":
                continue
            # delta-defined only (mirror B0-X / M4): rows with source-anchor
            # unavailable (delta is None) must not enter the regression target.
            if r.get("delta") is None:
                continue
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[x0x-3u-dev] 3U-A1 records: {len(rows)}")
    if not rows:
        raise SystemExit("no 3U-A1 rows")
    vocab = build_vocab(rows)
    folds = build_3u_folds(rows)
    print(f"[x0x-3u-dev] studies: {len(folds)}  split={PRIMARY_SPLIT}")

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[x0x-3u-dev] WARNING: falling back to CPU")
    out_dir = Path(args.out)
    results = []
    for i, fold in enumerate(folds):
        print(f"[x0x-3u-dev] fold {i+1}/{len(folds)} held={fold['held_out_study']} "
              f"train={len(fold['train'])} test={len(fold['test'])}", flush=True)
        res = train_fold(fold, vocab, device, out_dir)
        results.append(res)
        print(f"    -> spearman={res['test']['spearman']:.4f} "
              f"sign_acc={res['test']['sign_acc']:.4f} "
              f"top10_enrich={res['test']['top10_enrichment']:.4f} "
              f"indep_heads={res['independent_heads']}", flush=True)

    summary = {
        "phase": "X0-X_3UTR_REGION_DEV",
        "split": PRIMARY_SPLIT,
        "n_records": len(rows),
        "n_studies": len(folds),
        "independent_heads_all": all(r["independent_heads"] for r in results),
        "macro_delta_spearman": _macro(results, "spearman"),
        "macro_sign_accuracy": _macro(results, "sign_acc"),
        "macro_top10_enrichment": _macro(results, "top10_enrichment"),
        "per_fold": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "region_3u_dev_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
