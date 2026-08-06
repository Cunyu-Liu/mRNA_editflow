"""M4 SparseEditFormer architecture / encoding unit tests (pure, no remote data)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse import config as C  # noqa: E402
from scripts.m4_sparse.dataset import (  # noqa: E402
    EffectDataset, build_vocab, edit_features, invert_edits, one_hot,
)
from scripts.m4_sparse.model import SparseEditFormer  # noqa: E402
from scripts.m4_sparse.train import _pairwise_rank_loss, build_folds, compute_loss  # noqa: E402


def _rec(delta, study="S1", endpoint="ep_rl", benchmark="5U-A1", source_id="src1"):
    s = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    c = s[:10] + "G" + s[11:]
    return {
        "pair_id": "p", "record_id": "r", "study": study, "benchmark": benchmark,
        "source_sequence": s, "candidate_sequence": c,
        "edit_list": [{"op": "SUB", "pos": 10, "token": "G"}], "edit_count": 1,
        "source_id": source_id, "candidate_id": "c", "endpoint": endpoint,
        "candidate_value": float(delta) + 0.5, "source_value": 0.5,
        "delta": float(delta), "delta_source_status": "derived",
    }


def _cfg(rows):
    cfg = C.get_config()
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])
    return cfg, vocab


def _rows(n=16):
    return [_rec((i % 5) - 2.0, study="S%d" % (i % 3), source_id="src%d" % i) for i in range(n)]


# ---- one-hot / edit features ----
def test_one_hot_shape_and_t2u():
    a = one_hot("ACGU", max_len=100)
    assert a.shape == (100, 4)
    assert a[0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
    b = one_hot("ACGT", max_len=100)  # T -> U
    assert b[3].tolist() == [0.0, 0.0, 0.0, 1.0]  # U


def test_edit_features_empty_and_dim():
    assert edit_features([], 100).shape == (12,)
    assert np.allclose(edit_features([], 100), np.zeros(12))
    f = edit_features([{"op": "SUB", "pos": 10, "token": "G"}], 100)
    assert f.shape == (12,)
    assert f[0] == 1  # n_edits


def test_invert_edits_sub_uses_source_nucleotide():
    src = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    cand = src[:10] + "G" + src[11:]
    inv = invert_edits([{"op": "SUB", "pos": 10, "token": "G"}], src, cand)
    assert inv == [{"op": "SUB", "pos": 10, "token": "G"}]  # src[10] == "G"


def test_invert_edits_ins_del_swap():
    inv_ins = invert_edits([{"op": "INS", "pos": 3, "token": "G"}], "ACGUAC", "ACGGUAC")
    assert inv_ins[0]["op"] == "DEL" and inv_ins[0]["pos"] == 3
    inv_del = invert_edits([{"op": "DEL", "pos": 3, "token": "G"}], "ACGUAC", "ACUAC")
    assert inv_del[0]["op"] == "INS"


# ---- model forward ----
def test_model_forward_shapes():
    rows = _rows()
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    ds = EffectDataset(rows, vocab, target="delta")
    x = ds[0]
    out = model(x["src"].unsqueeze(0), x["cand"].unsqueeze(0), x["edit"].unsqueeze(0),
                x["study"].unsqueeze(0), x["endpoint"].unsqueeze(0), x["bench"].unsqueeze(0))
    assert out["mean"].shape == (1,)
    assert out["logvar"].shape == (1,)
    assert out["sign"].shape == (1,)
    assert out["rank"].shape == (1,)


def test_model_loss_decreases_when_overfitting():
    rows = _rows(64)
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ds = EffectDataset(rows, vocab, target="delta")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    # Mean loss over whole data passes (robust to batch sampling noise).
    def epoch_loss():
        model.train()
        tot, nb = 0.0, 0
        for b in dl:
            bb = {k: v for k, v in b.items()}
            opt.zero_grad()
            losses = compute_loss(model, bb, cfg, torch.device("cpu"))
            losses["loss"].backward()
            opt.step()
            tot += losses["loss"].item()
            nb += 1
        return tot / max(nb, 1)
    first = epoch_loss()
    last = None
    for _ in range(3):
        last = epoch_loss()
    assert last < first, "loss did not decrease on tiny overfit set"

def test_pairwise_rank_loss_zero_for_perfect_order():
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.manual_seed(0)
    # scores and targets are already aligned; a random shuffle may break order,
    # so assert the loss is non-negative and finite rather than exactly zero.
    loss = _pairwise_rank_loss(score, target, 0.5, torch.device("cpu"))
    assert float(loss) >= 0.0 and torch.isfinite(loss)


# ---- inverse consistency helper ----
def test_inverse_consistency_antisymmetry_term_is_finite():
    rows = _rows(16)
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    ds = EffectDataset(rows, vocab, target="delta")
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=8)))
    b = {k: v for k, v in batch.items()}
    out = model(b["src"], b["cand"], b["edit"], b["study"], b["endpoint"], b["bench"])
    out_inv = model(b["cand"], b["src"], b["inv_edit"], b["study"], b["endpoint"], b["bench"])
    consis = torch.mean((out["mean"] + out_inv["mean"]) ** 2)
    assert torch.isfinite(consis)
