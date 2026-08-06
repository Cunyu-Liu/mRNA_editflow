"""PyTorch Dataset for the M4 effect dataset (artifacts/b0x/effect_dataset.jsonl).

Encoding conventions mirror scripts/b0x/features.py (NUC_ORDER="ACGU",
MAX_SEQ_LEN=100, 12-dim edit features) so M4 is comparable to B0-X.
One-hot encodes source+candidate; encodes edit tokens (op/pos/token) into the
12-dim edit feature vector; provides study/endpoint/benchmark conditioning ids
and (for inverse consistency) the inverted edit features.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EDIT_FEAT_DIM, MAX_SEQ_LEN, NUC_ORDER

NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
GUESS = "N"


def normalize(seq: Optional[str]) -> str:
    return (seq or "").upper().replace("T", "U")


def one_hot(seq: Optional[str], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """(max_len, 4) ACGU one-hot of the first max_len nucleotides."""
    s = normalize(seq)
    arr = np.zeros((max_len, len(NUC_ORDER)), dtype=np.float32)
    for i, ch in enumerate(s[:max_len]):
        idx = NUC_TO_IDX.get(ch, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def edit_features(edit_list: Optional[List[Dict]], seq_len: int) -> np.ndarray:
    """12-dim edit feature vector (mirrors scripts/b0x/features.py edit_features).

    [n_edits, pos_mean, pos_min, pos_max, n_sub, n_ins, n_del,
     token_freqs(ACGU + other)].
    """
    edits = edit_list or []
    if not edits:
        return np.zeros(EDIT_FEAT_DIM, dtype=np.float32)
    posns = [e.get("pos", 0) for e in edits]
    denom = max(seq_len, 1)
    pos_mean = float(np.mean(posns)) / denom
    pos_min = float(np.min(posns)) / denom
    pos_max = float(np.max(posns)) / denom
    nsub = sum(1 for e in edits if e.get("op") == "SUB")
    nins = sum(1 for e in edits if e.get("op") == "INS")
    ndel = sum(1 for e in edits if e.get("op") == "DEL")
    tokens = [(str(e.get("token", GUESS)) or GUESS)[0].upper() for e in edits]
    tok = np.zeros(len(NUC_ORDER) + 1, dtype=np.float32)
    for t in tokens:
        idx = NUC_TO_IDX.get(t, -1)
        if idx >= 0:
            tok[idx] += 1.0
        else:
            tok[-1] += 1.0
    tok /= max(len(tokens), 1)
    return np.concatenate([
        np.array([len(edits), pos_mean, pos_min, pos_max, nsub, nins, ndel],
                 dtype=np.float32), tok]).astype(np.float32)


def invert_edits(edit_list: Optional[List[Dict]], source_seq: Optional[str],
                 candidate_seq: Optional[str]) -> List[Dict]:
    """Invert a source->candidate edit list into candidate->source edits.

    Assumes edit positions are in source coordinates (as in the effect dataset).
    SUB token is inverted to the source nucleotide at that position; INS<->DEL
    are swapped.  Exact for single-position point edits; a faithful approximation
    for larger edit scripts.
    """
    src = normalize(source_seq)
    cand = normalize(candidate_seq)
    inv: List[Dict] = []
    for e in (edit_list or []):
        op = e.get("op")
        pos = int(e.get("pos", 0))
        tok = str(e.get("token", "") or "")
        if op == "SUB":
            orig = (src[pos] if pos < len(src) else
                    (cand[pos] if pos < len(cand) else GUESS))
            inv.append({"op": "SUB", "pos": pos, "token": orig})
        elif op == "INS":
            inv.append({"op": "DEL", "pos": pos, "token": tok})
        elif op == "DEL":
            orig = src[pos] if pos < len(src) else GUESS
            inv.append({"op": "INS", "pos": pos, "token": orig})
        else:
            inv.append(dict(e))
    return inv


def build_vocab(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Global id maps for study / endpoint / benchmark across all rows so that
    held-out studies still have a valid (if untrained) conditioning embedding."""
    studies = sorted({r["study"] for r in rows})
    endpoints = sorted({r["endpoint"] for r in rows})
    benchmarks = sorted({r["benchmark"] for r in rows})
    return {
        "study": {s: i for i, s in enumerate(studies)},
        "endpoint": {e: i for i, e in enumerate(endpoints)},
        "benchmark": {b: i for i, b in enumerate(benchmarks)},
    }


class EffectDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Dict[str, Dict[str, int]],
                 target: str = "delta", max_len: int = MAX_SEQ_LEN):
        if not rows:
            raise ValueError("empty EffectDataset")
        self.rows = rows
        self.vocab = vocab
        self.target = target
        self.max_len = max_len
        self.src = np.stack([one_hot(r["source_sequence"], max_len) for r in rows])
        self.cand = np.stack([one_hot(r["candidate_sequence"], max_len) for r in rows])
        self.edit = np.stack([
            edit_features(r["edit_list"], len(r.get("source_sequence") or ""))
            for r in rows]).astype(np.float32)
        self.inv_edit = np.stack([
            edit_features(invert_edits(r["edit_list"], r.get("source_sequence"),
                                       r.get("candidate_sequence")),
                          len(r.get("candidate_sequence") or ""))
            for r in rows]).astype(np.float32)
        self.study = np.array([vocab["study"][r["study"]] for r in rows], dtype=np.int64)
        self.endpoint = np.array([vocab["endpoint"][r["endpoint"]] for r in rows],
                                 dtype=np.int64)
        self.bench = np.array([vocab["benchmark"][r["benchmark"]] for r in rows],
                              dtype=np.int64)
        if target == "delta":
            self.y = np.array([r["delta"] for r in rows], dtype=np.float32)
            self.anchor = np.zeros(len(rows), dtype=np.float32)
        elif target == "candidate_value":
            self.y = np.array([r["candidate_value"] if r["candidate_value"] is not None
                               else 0.0 for r in rows], dtype=np.float32)
            self.anchor = np.array([r["source_value"] if r["source_value"] is not None
                                    else 0.0 for r in rows], dtype=np.float32)
        else:
            raise ValueError(f"unknown target {target}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "src": torch.from_numpy(self.src[i]),
            "cand": torch.from_numpy(self.cand[i]),
            "edit": torch.from_numpy(self.edit[i]),
            "inv_edit": torch.from_numpy(self.inv_edit[i]),
            "study": torch.tensor(self.study[i], dtype=torch.long),
            "endpoint": torch.tensor(self.endpoint[i], dtype=torch.long),
            "bench": torch.tensor(self.bench[i], dtype=torch.long),
            "source_value": torch.tensor(self.anchor[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
        }
