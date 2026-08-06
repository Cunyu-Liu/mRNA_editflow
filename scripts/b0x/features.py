"""Lightweight sequence/delta features for B0-X baselines.

Self-contained implemention mirroring the feature families in
core/p3_02_delta_oracle.py (source_feat/candidate_feat/diff_feat/edit_feat and
one-hot encodings).  Kept local to the b0x package to avoid import coupling.
No learned features; all deterministic.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MAX_SEQ_LEN, NUC_ORDER  # noqa: E402

# RNA alphabet (T -> U). Nucleotide index map.
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
START_CODON = "AUG"
TOP_DINUC = ["AA", "UU", "GC", "CG", "AU", "UA"]
GUESS = "N"


def normalize(seq: str) -> str:
    return (seq or "").upper().replace("T", "U")


def _nuc_frequencies(seq: str) -> np.ndarray:
    n = len(seq)
    if n == 0:
        return np.zeros(len(NUC_ORDER), dtype=np.float32)
    c = Counter(seq)
    return np.array([c.get(ch, 0) / n for ch in NUC_ORDER], dtype=np.float32)


def _max_run(seq: str, ch: str) -> int:
    best = cur = 0
    for c in seq:
        if c == ch:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _entropy(freqs: np.ndarray) -> float:
    e = 0.0
    for f in freqs:
        if f > 0:
            e -= f * math.log2(f)
    return e


def sequence_features(seq: str) -> np.ndarray:
    """20-dim deterministic features: len, GC, gc_first10, AUG-pos, nt freqs,
    top dinuc freqs, poly-run max, entropy."""
    seq = normalize(seq)
    n = len(seq)
    if n == 0:
        return np.zeros(20, dtype=np.float32)
    freqs = _nuc_frequencies(seq)
    gc = float(freqs[NUC_TO_IDX["G"]] + freqs[NUC_TO_IDX["C"]])
    first = seq[:10]
    gc_first10 = (first.count("G") + first.count("C")) / max(len(first), 1)
    aug = seq.find(START_CODON)
    aug_norm = aug / n if aug >= 0 else -1.0
    din_c = Counter(seq[i:i + 2] for i in range(n - 1))
    tot = max(sum(din_c.values()), 1)
    din_freqs = np.array([din_c.get(d, 0) / tot for d in TOP_DINUC], dtype=np.float32)
    runs = np.array([_max_run(seq, ch) / n for ch in NUC_ORDER], dtype=np.float32)
    ent = np.array([_entropy(freqs)], dtype=np.float32)
    diversity = np.array([len(set(seq)) / len(NUC_ORDER)], dtype=np.float32)
    return np.concatenate([
        np.array([n / 100.0, gc, gc_first10, aug_norm], dtype=np.float32),
        freqs, din_freqs, runs, ent, diversity,
    ]).astype(np.float32)


def one_hot(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    seq = normalize(seq)
    arr = np.zeros((max_len, len(NUC_ORDER)), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        idx = NUC_TO_IDX.get(ch, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def edit_features(edit_list: List[Dict[str, Any]], seq_len: int) -> np.ndarray:
    """12-dim edit features (mirrors core edit_feat)."""
    edits = edit_list or []
    if not edits:
        return np.zeros(12, dtype=np.float32)
    posns = [e.get("pos", 0) for e in edits]
    pos_mean = float(np.mean(posns)) / max(seq_len, 1)
    pos_min = float(np.min(posns)) / max(seq_len, 1)
    pos_max = float(np.max(posns)) / max(seq_len, 1)
    nsub = sum(1 for e in edits if e.get("op") == "SUB")
    nins = sum(1 for e in edits if e.get("op") == "INS")
    ndel = sum(1 for e in edits if e.get("op") == "DEL")
    tokens = [(str(e.get("token", GUESS)) or GUESS)[0].upper() for e in edits]
    tok_freqs = np.zeros(len(NUC_ORDER) + 1, dtype=np.float32)  # +1 for non-ACGU
    for t in tokens:
        idx = NUC_TO_IDX.get(t, -1)
        if idx >= 0:
            tok_freqs[idx] += 1
        else:
            tok_freqs[-1] += 1
    tok_freqs /= max(len(tokens), 1)
    return np.concatenate([
        np.array([len(edits), pos_mean, pos_min, pos_max,
                  nsub, nins, ndel], dtype=np.float32),
        tok_freqs,
    ]).astype(np.float32)


def extract_features(source_seq: str, candidate_seq: str,
                     edit_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Return dict of feature arrays (mirrors core.extract_features)."""
    src = sequence_features(source_seq)
    cand = sequence_features(candidate_seq)
    return {
        "source_onehot": one_hot(source_seq),
        "candidate_onehot": one_hot(candidate_seq),
        "source_feat": src,
        "candidate_feat": cand,
        "diff_feat": cand - src,
        "edit_feat": edit_features(edit_list, len(source_seq or "")),
    }


def kmers(seq: str, k: int = 3) -> Counter:
    seq = normalize(seq)
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def kmers_vector(source: str, candidate: str, max_k: int = 3) -> np.ndarray:
    """Concatenated 1..max_k-mer count vectors for source and candidate."""
    parts = []
    for k in range(1, max_k + 1):
        n = 4 ** k
        src_vec = np.zeros(n, dtype=np.float32)
        cand_vec = np.zeros(n, dtype=np.float32)
        for seq, vec in ((source, src_vec), (candidate, cand_vec)):
            s = normalize(seq)
            total = max(len(s) - k + 1, 1)
            for i in range(len(s) - k + 1):
                mer = s[i:i + k]
                idx = 0
                ok = True
                for ch in mer:
                    j = NUC_TO_IDX.get(ch, -1)
                    if j < 0:
                        ok = False
                        break
                    idx = idx * 4 + j
                if ok:
                    vec[idx] += 1.0
            vec /= total
        parts.append(src_vec)
        parts.append(cand_vec)
    return np.concatenate(parts).astype(np.float32)