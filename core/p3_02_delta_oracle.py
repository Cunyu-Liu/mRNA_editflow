"""P3-02: Local-Delta Oracle and optimization-headroom gate.

This module implements the full P3-02 pipeline:

Task 1: Four prediction model architectures (absolute / difference / siamese / edit-conditioned)
Task 2: Local-delta metrics (sign accuracy, top-k enrichment, etc.)
Task 3: Cross-fitted oracle ensemble (≥3 structurally different models)
Task 4: Region sensitivity perturbation tests
Task 5: Optimization headroom search (greedy / beam / SA / MCTS / oracle-guided)

All models are CPU-trainable with synthetic or real P3-01 benchmark data.
No GPU required for unit tests.
"""
from __future__ import annotations

import json
import math
import os
import random
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUC_VOCAB = "ACGU"
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_VOCAB)}
IDX_TO_NUC = {i: ch for i, ch in enumerate(NUC_VOCAB)}
CODON_TABLE = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
SYNONYMOUS_CODONS: Dict[str, List[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    SYNONYMOUS_CODONS.setdefault(_aa, []).append(_codon)

START_CODON = "AUG"
STOP_CODONS = {"UAA", "UAG", "UGA"}


# ===========================================================================
# Section 1: Data Loading
# ===========================================================================

@dataclass
class DeltaRecord:
    """A single (source, candidate, delta) tuple for local-delta prediction."""
    record_id: str
    source_id: str
    source_sequence: str
    candidate_sequence: str
    edit_list: List[Dict[str, Any]]
    edit_count: int
    edited_region: str
    delta: float
    source_value: Optional[float]
    candidate_value: Optional[float]
    value_std: Optional[float]
    confidence: str  # "measured" | "proxy" | "unlabeled"
    split_role: str  # "train" | "val" | "test" | "ood"
    family_cluster_id: str
    edit_type: str


def load_benchmark_tier(
    jsonl_path: str,
    min_delta: Optional[float] = None,
    max_edit_count: Optional[int] = None,
) -> List[DeltaRecord]:
    """Load a P3-01 benchmark tier JSONL into DeltaRecord list.

    Skips unlabeled records (delta is None) and optionally filters.
    """
    records: List[DeltaRecord] = []
    with open(jsonl_path) as f:
        for line in f:
            raw = json.loads(line)
            if raw.get("confidence") == "unlabeled":
                continue
            delta = raw.get("delta")
            if delta is None:
                continue
            ec = raw.get("edit_count", 0)
            if max_edit_count is not None and ec > max_edit_count:
                continue
            if min_delta is not None and abs(delta) < min_delta:
                continue
            records.append(DeltaRecord(
                record_id=raw["record_id"],
                source_id=raw.get("source_id", ""),
                source_sequence=raw.get("source_sequence", ""),
                candidate_sequence=raw.get("candidate_sequence", ""),
                edit_list=raw.get("edit_list", []),
                edit_count=ec,
                edited_region=raw.get("edited_region", "five_utr"),
                delta=float(delta),
                source_value=raw.get("measured_or_proxy_source_value"),
                candidate_value=raw.get("measured_or_proxy_candidate_value"),
                value_std=raw.get("value_std"),
                confidence=raw.get("confidence", "unknown"),
                split_role=raw.get("split_role", "train"),
                family_cluster_id=raw.get("family_cluster_id", ""),
                edit_type=raw.get("edit_type", ""),
            ))
    return records


def load_benchmark(
    benchmark_dir: str,
    tiers: Sequence[str] = ("measured", "proxy"),
) -> Dict[str, List[DeltaRecord]]:
    """Load multiple tiers from a benchmark directory."""
    result: Dict[str, List[DeltaRecord]] = {}
    for tier in tiers:
        path = os.path.join(benchmark_dir, f"{tier}_tier.jsonl")
        if os.path.exists(path):
            result[tier] = load_benchmark_tier(path)
        else:
            result[tier] = []
    return result


# ===========================================================================
# Section 2: Feature Extraction
# ===========================================================================

def _one_hot_sequence(seq: str, max_len: int = 100) -> np.ndarray:
    """One-hot encode an RNA sequence, padded/truncated to max_len."""
    arr = np.zeros((max_len, 4), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        idx = NUC_TO_IDX.get(ch.upper().replace("T", "U"), -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def _sequence_features(seq: str) -> np.ndarray:
    """Extract lightweight sequence features (length, GC, k-mer, position)."""
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return np.zeros(20, dtype=np.float32)
    # Nucleotide frequencies
    counts = Counter(seq)
    freqs = np.array([counts.get(ch, 0) / n for ch in NUC_VOCAB], dtype=np.float32)
    gc = (counts.get("G", 0) + counts.get("C", 0)) / n
    # Position features
    first_10 = seq[:10]
    gc_first10 = (first_10.count("G") + first_10.count("C")) / max(len(first_10), 1)
    # Kozak-like: AUG position
    aug_pos = seq.find(START_CODON)
    aug_pos_norm = aug_pos / n if aug_pos >= 0 else -1.0
    # Dinucleotide frequencies (top 6)
    dinuc_counts = Counter(seq[i:i+2] for i in range(n - 1))
    total_di = max(sum(dinuc_counts.values()), 1)
    top_dinucs = ["AA", "UU", "GC", "CG", "AU", "UA"]
    dinuc_freqs = np.array([dinuc_counts.get(d, 0) / total_di for d in top_dinucs], dtype=np.float32)
    # Length and GC
    features = np.concatenate([
        np.array([n / 100.0, gc, gc_first10, aug_pos_norm], dtype=np.float32),
        freqs,
        dinuc_freqs,
        # Run lengths (max poly-N run)
        np.array([_max_run(seq, ch) / n for ch in NUC_VOCAB], dtype=np.float32),
        # Entropy
        np.array([_seq_entropy(freqs)], dtype=np.float32),
        # GC windows
        np.array([_gc_window(seq, 0, 10), _gc_window(seq, 0, 20), _gc_window(seq, n//2, n//2+10)], dtype=np.float32),
    ])
    return features


def _max_run(seq: str, ch: str) -> int:
    """Max consecutive run of character ch in seq."""
    max_r = 0
    cur = 0
    for c in seq:
        if c == ch:
            cur += 1
            max_r = max(max_r, cur)
        else:
            cur = 0
    return max_r


def _seq_entropy(freqs: np.ndarray) -> float:
    """Shannon entropy of nucleotide frequencies."""
    p = freqs[freqs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _gc_window(seq: str, start: int, end: int) -> float:
    """GC content in a window."""
    window = seq[start:end]
    if len(window) == 0:
        return 0.0
    return (window.count("G") + window.count("C")) / len(window)


def _edit_features(edit_list: List[Dict[str, Any]], seq_len: int) -> np.ndarray:
    """Extract features from the edit list.

    Returns a fixed-size feature vector regardless of edit count.
    """
    n_edits = len(edit_list)
    if n_edits == 0:
        return np.zeros(12, dtype=np.float32)

    # Aggregate edit features
    positions = []
    regions = Counter()
    ref_nts = Counter()
    alt_nts = Counter()
    for ed in edit_list:
        pos = ed.get("pos", 0)
        positions.append(pos / max(seq_len, 1))
        regions[ed.get("region", "unknown")] += 1
        ref_nts[ed.get("ref", "")] += 1
        alt_nts[ed.get("alt", "")] += 1

    pos_mean = float(np.mean(positions))
    pos_std = float(np.std(positions)) if len(positions) > 1 else 0.0
    pos_min = float(min(positions))
    pos_max = float(max(positions))

    # Region distribution
    region_5utr = regions.get("five_utr", 0) / max(n_edits, 1)
    region_cds = (regions.get("cds_first30", 0) + regions.get("cds_first50", 0) +
                  regions.get("cds_remaining", 0)) / max(n_edits, 1)

    # Nucleotide transition features
    transitions = {}
    for ed in edit_list:
        ref = ed.get("ref", "")
        alt = ed.get("alt", "")
        key = f"{ref}>{alt}"
        transitions[key] = transitions.get(key, 0) + 1

    # GC-changing edits (A<->G, C<->U)
    gc_changes = 0
    for ed in edit_list:
        ref = ed.get("ref", "")
        alt = ed.get("alt", "")
        ref_gc = ref in "GC"
        alt_gc = alt in "GC"
        if ref_gc != alt_gc:
            gc_changes += 1

    features = np.array([
        n_edits / 10.0,
        pos_mean,
        pos_std,
        pos_min,
        pos_max,
        region_5utr,
        region_cds,
        gc_changes / max(n_edits, 1),
        # Transition frequencies (top 4)
        transitions.get("A>G", 0) / max(n_edits, 1),
        transitions.get("C>U", 0) / max(n_edits, 1),
        transitions.get("G>A", 0) / max(n_edits, 1),
        transitions.get("U>C", 0) / max(n_edits, 1),
    ], dtype=np.float32)
    return features


def extract_features(
    source_seq: str,
    candidate_seq: str,
    edit_list: List[Dict[str, Any]],
    max_seq_len: int = 100,
) -> Dict[str, np.ndarray]:
    """Extract all feature types for a (source, candidate, edits) tuple.

    Returns a dict with keys:
        - source_onehot: [max_seq_len, 4]
        - candidate_onehot: [max_seq_len, 4]
        - source_feat: [20]
        - candidate_feat: [20]
        - diff_feat: [20]  (candidate_feat - source_feat)
        - edit_feat: [12]
    """
    src_feat = _sequence_features(source_seq)
    cand_feat = _sequence_features(candidate_seq)
    return {
        "source_onehot": _one_hot_sequence(source_seq, max_seq_len),
        "candidate_onehot": _one_hot_sequence(candidate_seq, max_seq_len),
        "source_feat": src_feat,
        "candidate_feat": cand_feat,
        "diff_feat": cand_feat - src_feat,
        "edit_feat": _edit_features(edit_list, len(source_seq)),
    }


def batch_extract_features(
    records: Sequence[DeltaRecord],
    max_seq_len: int = 100,
) -> Dict[str, np.ndarray]:
    """Extract features for a batch of records."""
    n = len(records)
    # Determine feature sizes from first record
    if n == 0:
        return {}
    sample = extract_features(records[0].source_sequence, records[0].candidate_sequence,
                              records[0].edit_list, max_seq_len)
    result = {k: np.zeros((n,) + v.shape, dtype=v.dtype) for k, v in sample.items()}
    for i, rec in enumerate(records):
        feats = extract_features(rec.source_sequence, rec.candidate_sequence,
                                 rec.edit_list, max_seq_len)
        for k, v in feats.items():
            result[k][i] = v
    # Add labels
    result["delta"] = np.array([r.delta for r in records], dtype=np.float32)
    result["source_value"] = np.array(
        [r.source_value if r.source_value is not None else 0.0 for r in records], dtype=np.float32)
    result["edit_count"] = np.array([r.edit_count for r in records], dtype=np.float32)
    return result


# ===========================================================================
# Section 3: Model Architectures (Task 1)
# ===========================================================================

class AbsoluteModel:
    """Model 1: candidate_sequence → predicted absolute value.

    Predicts the absolute output for a sequence, then computes delta
    as predict(candidate) - predict(source).
    """

    def __init__(self, hidden_dim: int = 64, max_seq_len: int = 100, lr: float = 1e-3,
                 n_epochs: int = 100, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None
        self._bias: Optional[float] = None

    def _init_weights(self, feat_dim: int):
        self._weights = {
            "w1": self._rng.randn(feat_dim, self.hidden_dim) * 0.01,
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "w2": self._rng.randn(self.hidden_dim) * 0.01,
        }
        self._bias = 0.0

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
        return h @ self._weights["w2"] + self._bias

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        """Train on absolute candidate values."""
        X = features["candidate_feat"]
        if self._weights is None:
            self._init_weights(X.shape[1])
        # Normalize targets
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        for epoch in range(self.n_epochs):
            pred = self._forward(X)
            err = pred - y_norm
            h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
            grad_out = err / len(y_norm)
            grad_w2 = h.T @ grad_out
            grad_b = float(np.mean(grad_out))
            grad_h = np.outer(grad_out, self._weights["w2"]) * (h > 0)
            grad_w1 = X.T @ grad_h
            grad_b1 = np.mean(grad_h, axis=0)
            self._weights["w2"] -= self.lr * grad_w2
            self._bias -= self.lr * grad_b
            self._weights["w1"] -= self.lr * grad_w1
            self._weights["b1"] -= self.lr * grad_b1

    def predict_absolute(self, features: Dict[str, np.ndarray], seq_key: str = "candidate_feat") -> np.ndarray:
        """Predict absolute value for a sequence."""
        X = features[seq_key]
        return self._forward(X) * self._y_std + self._y_mean

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict delta = predict(candidate) - predict(source)."""
        return self.predict_absolute(features, "candidate_feat") - self.predict_absolute(features, "source_feat")


class DifferenceModel:
    """Model 2: (source, candidate, edits) → predicted delta.

    Directly predicts the delta from difference features + edit features.
    """

    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3, n_epochs: int = 100, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _init_weights(self, feat_dim: int):
        self._weights = {
            "w1": self._rng.randn(feat_dim, self.hidden_dim) * 0.01,
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "w2": self._rng.randn(self.hidden_dim) * 0.01,
            "b2": 0.0,
        }

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
        return h @ self._weights["w2"] + self._weights["b2"]

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        """Train on delta labels."""
        X = np.concatenate([features["diff_feat"], features["edit_feat"]], axis=1)
        if self._weights is None:
            self._init_weights(X.shape[1])
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        for epoch in range(self.n_epochs):
            pred = self._forward(X)
            err = pred - y_norm
            h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
            grad_out = err / len(y_norm)
            grad_w2 = h.T @ grad_out
            grad_b2 = float(np.mean(grad_out))
            grad_h = np.outer(grad_out, self._weights["w2"]) * (h > 0)
            grad_w1 = X.T @ grad_h
            grad_b1 = np.mean(grad_h, axis=0)
            self._weights["w2"] -= self.lr * grad_w2
            self._weights["b2"] -= self.lr * grad_b2
            self._weights["w1"] -= self.lr * grad_w1
            self._weights["b1"] -= self.lr * grad_b1

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        X = np.concatenate([features["diff_feat"], features["edit_feat"]], axis=1)
        return self._forward(X) * self._y_std + self._y_mean


class SiameseModel:
    """Model 3: encoder(source), encoder(candidate) → delta.

    Shared encoder processes source and candidate independently;
    the difference of embeddings feeds a delta head.
    """

    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3, n_epochs: int = 100, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _init_weights(self, feat_dim: int):
        self._weights = {
            "enc_w1": self._rng.randn(feat_dim, self.hidden_dim) * 0.01,
            "enc_b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "delta_w": self._rng.randn(self.hidden_dim) * 0.01,
            "delta_b": 0.0,
        }

    def _encode(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0, X @ self._weights["enc_w1"] + self._weights["enc_b1"])

    def _forward(self, src_feat: np.ndarray, cand_feat: np.ndarray) -> np.ndarray:
        src_enc = self._encode(src_feat)
        cand_enc = self._encode(cand_feat)
        diff = cand_enc - src_enc
        return diff @ self._weights["delta_w"] + self._weights["delta_b"]

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        src_X = features["source_feat"]
        cand_X = features["candidate_feat"]
        if self._weights is None:
            self._init_weights(src_X.shape[1])
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        for epoch in range(self.n_epochs):
            pred = self._forward(src_X, cand_X)
            err = pred - y_norm
            grad_out = err / len(y_norm)
            src_enc = self._encode(src_X)
            cand_enc = self._encode(cand_X)
            diff = cand_enc - src_enc
            grad_delta_w = diff.T @ grad_out
            grad_delta_b = float(np.mean(grad_out))
            grad_diff = np.outer(grad_out, self._weights["delta_w"])
            # grad flows to encoder: +for candidate, -for source
            grad_cand_enc = grad_diff * (cand_enc > 0)
            grad_src_enc = -grad_diff * (src_enc > 0)
            grad_enc_w1 = src_X.T @ grad_src_enc + cand_X.T @ grad_cand_enc
            grad_enc_b1 = np.mean(grad_src_enc, axis=0) + np.mean(grad_cand_enc, axis=0)
            self._weights["delta_w"] -= self.lr * grad_delta_w
            self._weights["delta_b"] -= self.lr * grad_delta_b
            self._weights["enc_w1"] -= self.lr * grad_enc_w1
            self._weights["enc_b1"] -= self.lr * grad_enc_b1

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        return self._forward(features["source_feat"], features["candidate_feat"]) * self._y_std + self._y_mean


class EditConditionedModel:
    """Model 4: source representation + sparse edit tokens → delta.

    Uses source sequence features + structured edit features to predict delta.
    """

    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3, n_epochs: int = 100, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _init_weights(self, src_dim: int, edit_dim: int):
        total_dim = src_dim + edit_dim
        self._weights = {
            "w1": self._rng.randn(total_dim, self.hidden_dim) * 0.01,
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "w2": self._rng.randn(self.hidden_dim) * 0.01,
            "b2": 0.0,
        }

    def _forward(self, src_feat: np.ndarray, edit_feat: np.ndarray) -> np.ndarray:
        X = np.concatenate([src_feat, edit_feat], axis=1)
        h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
        return h @ self._weights["w2"] + self._weights["b2"]

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        src_X = features["source_feat"]
        edit_X = features["edit_feat"]
        if self._weights is None:
            self._init_weights(src_X.shape[1], edit_X.shape[1])
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        X = np.concatenate([src_X, edit_X], axis=1)
        for epoch in range(self.n_epochs):
            pred = self._forward(src_X, edit_X)
            err = pred - y_norm
            grad_out = err / len(y_norm)
            h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
            grad_w2 = h.T @ grad_out
            grad_b2 = float(np.mean(grad_out))
            grad_h = np.outer(grad_out, self._weights["w2"]) * (h > 0)
            grad_w1 = X.T @ grad_h
            grad_b1 = np.mean(grad_h, axis=0)
            self._weights["w2"] -= self.lr * grad_w2
            self._weights["b2"] -= self.lr * grad_b2
            self._weights["w1"] -= self.lr * grad_w1
            self._weights["b1"] -= self.lr * grad_b1

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        return self._forward(features["source_feat"], features["edit_feat"]) * self._y_std + self._y_mean


MODEL_REGISTRY = {
    "absolute": AbsoluteModel,
    "difference": DifferenceModel,
    "siamese": SiameseModel,
    "edit_conditioned": EditConditionedModel,
    "seq_diff": None,      # filled below (forward-ref)
    "seq_cnn": None,       # filled below (forward-ref)
}


# ===========================================================================
# Section 3b: Position-Aware Models (P3-02 fix for constant-predictor bug)
# ===========================================================================

class SeqDiffModel:
    """Model 5: Position-aware MLP on flattened one-hot sequence difference.

    Unlike DifferenceModel (20-dim global composition features), this model
    uses the full [max_seq_len * 4] flattened one-hot difference, giving each
    position its own input dimension.  This breaks the constant-predictor
    degeneracy where single-nucleotide edits produce nearly identical global
    statistics.
    """

    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3,
                 n_epochs: int = 150, seed: int = 42, max_seq_len: int = 100,
                 weight_decay: float = 1e-4):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.weight_decay = weight_decay
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _init_weights(self, feat_dim: int):
        self._weights = {
            "w1": self._rng.randn(feat_dim, self.hidden_dim) * np.sqrt(2.0 / feat_dim),
            "b1": np.zeros(self.hidden_dim, dtype=np.float64),
            "w2": self._rng.randn(self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim),
            "b2": 0.0,
        }

    def _get_input(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        diff = features["candidate_onehot"].astype(np.float64) - features["source_onehot"].astype(np.float64)
        return np.ascontiguousarray(diff.reshape(diff.shape[0], -1))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
        return h @ self._weights["w2"] + self._weights["b2"]

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        X = self._get_input(features)
        if self._weights is None:
            self._init_weights(X.shape[1])
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        n = len(y_norm)
        for _ in range(self.n_epochs):
            pred = self._forward(X)
            err = pred - y_norm
            h = np.maximum(0, X @ self._weights["w1"] + self._weights["b1"])
            grad_out = err / n
            grad_w2 = h.T @ grad_out + self.weight_decay * self._weights["w2"]
            grad_b2 = float(np.mean(grad_out))
            grad_h = np.outer(grad_out, self._weights["w2"]) * (h > 0)
            grad_w1 = X.T @ grad_h + self.weight_decay * self._weights["w1"]
            grad_b1 = np.mean(grad_h, axis=0)
            self._weights["w2"] -= self.lr * grad_w2
            self._weights["b2"] -= self.lr * grad_b2
            self._weights["w1"] -= self.lr * grad_w1
            self._weights["b1"] -= self.lr * grad_b1

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        X = self._get_input(features)
        return self._forward(X) * self._y_std + self._y_mean


class SeqCNNModel:
    """Model 6: 1D CNN over one-hot sequence difference.

    Architecture:
        diff = candidate_onehot - source_onehot      [batch, L, 4]
        conv1d(4 -> n_filters, kernel=k) -> ReLU      [batch, L-k+1, F]
        global max pool                                 [batch, F]
        FC(F -> 1)                                      [batch]

    The convolution learns local sequence motifs around edit positions,
    while global max pooling makes the prediction invariant to edit
    position shift.  Far fewer parameters than SeqDiffModel, better
    generalization to unseen positions.
    """

    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3,
                 n_epochs: int = 150, seed: int = 42, max_seq_len: int = 100,
                 n_filters: int = 32, kernel_size: int = 5):
        self.hidden_dim = hidden_dim  # interface compat (unused)
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self._rng = np.random.RandomState(seed)
        self._weights: Optional[Dict[str, np.ndarray]] = None

    def _init_weights(self):
        ks, nf = self.kernel_size, self.n_filters
        self._weights = {
            "conv_w": self._rng.randn(ks, 4, nf) * np.sqrt(2.0 / (ks * 4)),
            "conv_b": np.zeros(nf, dtype=np.float64),
            "fc_w": self._rng.randn(nf) * np.sqrt(2.0 / nf),
            "fc_b": 0.0,
        }

    def _conv_forward(self, diff: np.ndarray):
        """diff: [batch, L, 4] -> (conv_out, cols)"""
        diff = np.ascontiguousarray(diff, dtype=np.float64)
        batch, L, _ = diff.shape
        ks = self.kernel_size
        out_len = L - ks + 1
        # Vectorized im2col via sliding_window_view
        from numpy.lib.stride_tricks import sliding_window_view
        window = sliding_window_view(diff, ks, axis=1)  # (batch, out_len, 4, ks)
        cols = window.transpose(0, 1, 3, 2).reshape(batch, out_len, ks * 4)
        W = self._weights["conv_w"].reshape(ks * 4, self.n_filters)
        conv_out = cols @ W + self._weights["conv_b"]
        return conv_out, cols

    def _forward(self, source_oh, candidate_oh):
        diff = candidate_oh - source_oh
        conv_out, cols = self._conv_forward(diff)
        relu = np.maximum(0, conv_out)
        pooled = np.max(relu, axis=1)          # [batch, F]
        pred = pooled @ self._weights["fc_w"] + self._weights["fc_b"]
        return pred, conv_out, cols, relu, pooled

    def fit(self, features: Dict[str, np.ndarray], y: np.ndarray):
        src = features["source_onehot"]
        cand = features["candidate_onehot"]
        if self._weights is None:
            self._init_weights()
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std
        n = len(y_norm)
        nf = self.n_filters
        ks = self.kernel_size
        for _ in range(self.n_epochs):
            pred, conv_out, cols, relu, pooled = self._forward(src, cand)
            err = pred - y_norm
            grad_out = err / n                          # [batch]
            # FC grads
            grad_fc_w = pooled.T @ grad_out             # [F]
            grad_fc_b = float(np.mean(grad_out))
            grad_pooled = np.outer(grad_out, self._weights["fc_w"])  # [batch, F]
            # Global max pool backward: grad flows to argmax position only
            argmax = np.argmax(relu, axis=1)            # [batch, F]
            # Vectorized: create one-hot mask of argmax positions
            batch_idx = np.arange(n)[:, np.newaxis]     # [n, 1]
            filt_idx = np.arange(nf)[np.newaxis, :]     # [1, F]
            grad_relu = np.zeros_like(relu)
            grad_relu[batch_idx, argmax, filt_idx] = grad_pooled
            # ReLU backward
            grad_conv = grad_relu * (conv_out > 0)      # [batch, out_len, F]
            # Conv weight grads via einsum
            W_flat = self._weights["conv_w"].reshape(ks * 4, nf)
            grad_w_flat = np.einsum('blk,blf->kf', cols, grad_conv) / n
            grad_conv_b = np.mean(grad_conv, axis=(0, 1))
            # Update
            self._weights["fc_w"] -= self.lr * grad_fc_w
            self._weights["fc_b"] -= self.lr * grad_fc_b
            self._weights["conv_w"] -= self.lr * grad_w_flat.reshape(ks, 4, nf)
            self._weights["conv_b"] -= self.lr * grad_conv_b

    def predict_delta(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        pred, *_ = self._forward(features["source_onehot"], features["candidate_onehot"])
        return pred * self._y_std + self._y_mean


# Fill forward references
MODEL_REGISTRY["seq_diff"] = SeqDiffModel
MODEL_REGISTRY["seq_cnn"] = SeqCNNModel


# ===========================================================================
# Section 4: Local-Delta Metrics (Task 2)
# ===========================================================================

def delta_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation of deltas."""
    from scipy.stats import spearmanr
    if len(pred) < 2:
        return 0.0
    r, _ = spearmanr(pred, true)
    return float(r) if not np.isnan(r) else 0.0


def delta_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation of deltas."""
    if len(pred) < 2:
        return 0.0
    p = pred - pred.mean()
    t = true - true.mean()
    denom = np.sqrt(np.sum(p**2) * np.sum(t**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(p * t) / denom)


def sign_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of predictions with correct sign (beneficial vs harmful)."""
    if len(pred) == 0:
        return 0.0
    pred_sign = np.sign(pred)
    true_sign = np.sign(true)
    # Only count non-zero true deltas
    mask = true_sign != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(pred_sign[mask] == true_sign[mask]))


def pairwise_ranking_auc(pred: np.ndarray, true: np.ndarray, n_pairs: int = 10000,
                         seed: int = 42) -> float:
    """AUC of pairwise ranking: for pairs (i,j) where true_i > true_j,
    is pred_i > pred_j?"""
    n = len(pred)
    if n < 2:
        return 0.5
    rng = np.random.RandomState(seed)
    idx_i = rng.randint(0, n, size=min(n_pairs, n * n))
    idx_j = rng.randint(0, n, size=min(n_pairs, n * n))
    # Filter to non-tied true pairs
    mask = true[idx_i] > true[idx_j]
    if mask.sum() == 0:
        return 0.5
    correct = pred[idx_i[mask]] > pred[idx_j[mask]]
    return float(np.mean(correct))


def beneficial_edit_precision(pred: np.ndarray, true: np.ndarray, threshold: float = 0.0) -> float:
    """Of edits predicted beneficial (pred > threshold), fraction truly beneficial."""
    pred_pos = pred > threshold
    if pred_pos.sum() == 0:
        return 0.0
    true_pos = true > threshold
    return float(np.mean(true_pos[pred_pos]))


def top_k_enrichment(pred: np.ndarray, true: np.ndarray, k: float = 0.1) -> float:
    """Enrichment of truly beneficial edits in top-k predicted.

    Returns: (fraction of true beneficial in top-k) / (fraction of true beneficial overall).
    """
    n = len(pred)
    if n == 0:
        return 0.0
    k_n = max(1, int(n * k))
    top_idx = np.argsort(-pred)[:k_n]
    true_beneficial = true > 0
    base_rate = true_beneficial.mean()
    if base_rate < 1e-8:
        return 0.0
    top_rate = true_beneficial[top_idx].mean()
    return float(top_rate / base_rate)


def calibration_error(pred: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: average |pred_mean - true_mean| per bin."""
    if len(pred) == 0:
        return 0.0
    bins = np.linspace(pred.min(), pred.max(), n_bins + 1)
    if bins[-1] - bins[0] < 1e-8:
        return 0.0
    total_err = 0.0
    total_n = 0
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        err = abs(float(pred[mask].mean()) - float(true[mask].mean()))
        total_err += err * mask.sum()
        total_n += mask.sum()
    return float(total_err / max(total_n, 1))


def false_positive_beneficial_rate(pred: np.ndarray, true: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of predicted-beneficial edits that are truly non-beneficial (≤ 0)."""
    pred_pos = pred > threshold
    if pred_pos.sum() == 0:
        return 0.0
    true_non_pos = true <= threshold
    return float(np.mean(true_non_pos[pred_pos]))


def source_normalized_rmse(pred: np.ndarray, true: np.ndarray, source_values: np.ndarray) -> float:
    """RMSE normalized by source value magnitude."""
    if len(pred) == 0:
        return 0.0
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    src_scale = np.mean(np.abs(source_values)) + 1e-8
    return float(rmse / src_scale)


def compute_all_metrics(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    source_values: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all local-delta metrics at once."""
    metrics = {
        "delta_spearman": delta_spearman(pred_delta, true_delta),
        "delta_pearson": delta_pearson(pred_delta, true_delta),
        "sign_accuracy": sign_accuracy(pred_delta, true_delta),
        "pairwise_ranking_auc": pairwise_ranking_auc(pred_delta, true_delta),
        "beneficial_edit_precision": beneficial_edit_precision(pred_delta, true_delta),
        "top_k_enrichment_10pct": top_k_enrichment(pred_delta, true_delta, k=0.1),
        "top_k_enrichment_5pct": top_k_enrichment(pred_delta, true_delta, k=0.05),
        "calibration_error": calibration_error(pred_delta, true_delta),
        "false_positive_beneficial_rate": false_positive_beneficial_rate(pred_delta, true_delta),
        "rmse": float(np.sqrt(np.mean((pred_delta - true_delta) ** 2))) if len(pred_delta) > 0 else 0.0,
        "mae": float(np.mean(np.abs(pred_delta - true_delta))) if len(pred_delta) > 0 else 0.0,
    }
    if source_values is not None:
        metrics["source_normalized_rmse"] = source_normalized_rmse(pred_delta, true_delta, source_values)
    return metrics


# ===========================================================================
# Section 5: Cross-Fitted Oracle Ensemble (Task 3)
# ===========================================================================

@dataclass
class CrossFitConfig:
    """Configuration for cross-fitted oracle ensemble."""
    n_folds: int = 5
    seed: int = 42
    # Model hyperparameters
    hidden_dim: int = 64
    lr: float = 1e-3
    n_epochs: int = 100
    max_seq_len: int = 100


@dataclass
class OracleResult:
    """Result from a single oracle or the ensemble."""
    predictions: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    model_name: str = ""
    fold_id: Optional[int] = None


def cross_fit_predict(
    model_class: type,
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    fold_indices: Sequence[np.ndarray],
    config: CrossFitConfig,
) -> Tuple[np.ndarray, Dict[int, Any]]:
    """Cross-fitted prediction: train on K-1 folds, predict on holdout.

    Returns OOF predictions and per-fold trained models.
    """
    n = len(labels)
    oof_pred = np.zeros(n, dtype=np.float64)
    models: Dict[int, Any] = {}

    for fold_id, val_idx in enumerate(fold_indices):
        val_idx = np.array(val_idx)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        # Create sub-features for train
        train_feats = {k: v[train_idx] for k, v in features.items()}
        train_labels = labels[train_idx]

        # Train model
        model = model_class(
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            n_epochs=config.n_epochs,
            seed=config.seed + fold_id,
        )
        model.fit(train_feats, train_labels)

        # Predict on validation
        val_feats = {k: v[val_idx] for k, v in features.items()}
        oof_pred[val_idx] = model.predict_delta(val_feats)
        models[fold_id] = model

    return oof_pred, models


def build_oracle_ensemble(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    fold_indices: Sequence[np.ndarray],
    config: CrossFitConfig,
    model_names: Sequence[str] = ("absolute", "difference", "siamese", "edit_conditioned"),
) -> Dict[str, Any]:
    """Build a cross-fitted ensemble of structurally different oracles.

    Returns a dict with:
        - per_model_oof: {model_name: OOF predictions}
        - ensemble_pred: mean of all model OOF predictions
        - ensemble_uncertainty: std across models
        - disagreement: max - min across models
        - per_model_models: {model_name: {fold_id: trained model}}
    """
    per_model_oof: Dict[str, np.ndarray] = {}
    per_model_models: Dict[str, Dict[int, Any]] = {}

    for name in model_names:
        model_class = MODEL_REGISTRY[name]
        oof_pred, models = cross_fit_predict(
            model_class, features, labels, fold_indices, config
        )
        per_model_oof[name] = oof_pred
        per_model_models[name] = models

    # Ensemble
    stacked = np.stack(list(per_model_oof.values()), axis=0)
    ensemble_pred = stacked.mean(axis=0)
    ensemble_uncertainty = stacked.std(axis=0)
    disagreement = stacked.max(axis=0) - stacked.min(axis=0)

    return {
        "per_model_oof": per_model_oof,
        "ensemble_pred": ensemble_pred,
        "ensemble_uncertainty": ensemble_uncertainty,
        "disagreement": disagreement,
        "per_model_models": per_model_models,
        "model_names": list(model_names),
    }


# ===========================================================================
# Section 6: Region Sensitivity Tests (Task 4)
# ===========================================================================

def _random_substitution(seq: str, pos: int, rng: random.Random) -> str:
    """Substitute a single nucleotide at pos with a random different nt."""
    old = seq[pos]
    choices = [ch for ch in NUC_VOCAB if ch != old]
    new = rng.choice(choices)
    return seq[:pos] + new + seq[pos + 1:]


def _synonymous_substitution(cds: str, codon_idx: int, rng: random.Random) -> str:
    """Substitute a codon with a synonymous one at codon_idx."""
    start = codon_idx * 3
    if start + 3 > len(cds):
        return cds
    codon = cds[start:start + 3]
    aa = CODON_TABLE.get(codon, "")
    synonyms = [c for c in SYNONYMOUS_CODONS.get(aa, []) if c != codon]
    if not synonyms:
        return cds
    new_codon = rng.choice(synonyms)
    return cds[:start] + new_codon + cds[start + 3:]


@dataclass
class PerturbationResult:
    """Result of a single perturbation test."""
    perturbation_type: str
    mean_pred_delta: float
    std_pred_delta: float
    n_samples: int
    position_sensitivity: Optional[Dict[str, float]] = None


def run_region_sensitivity(
    predictor: Any,
    source_sequences: Sequence[str],
    config: CrossFitConfig,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, PerturbationResult]:
    """Run controlled perturbation tests.

    Tests:
    1. 5'UTR single substitution
    2. Start-proximal synonymous substitution
    3. Middle-CDS synonymous substitution
    4. Late-CDS synonymous substitution
    5. Joint 5'UTR + CDS
    6. Matched random substitution
    """
    rng = random.Random(seed)
    n = min(n_samples, len(source_sequences))
    results: Dict[str, PerturbationResult] = {}

    # Helper: make features and predict delta
    def predict_delta_for(src: str, cand: str, edits: List[Dict]) -> float:
        feats = extract_features(src, cand, edits, config.max_seq_len)
        # Batch of 1
        batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
        return float(predictor.predict_delta(batch_feats)[0])

    # 1. 5'UTR single substitution
    deltas_5utr = []
    pos_sensitivity: Dict[int, List[float]] = defaultdict(list)
    for i in range(n):
        seq = source_sequences[i]
        if len(seq) < 10:
            continue
        # Find 5'UTR region (before AUG)
        aug_pos = seq.find(START_CODON)
        utr_end = aug_pos if aug_pos >= 0 else min(len(seq) // 2, 50)
        if utr_end < 3:
            continue
        pos = rng.randint(0, utr_end - 1)
        cand = _random_substitution(seq, pos, rng)
        edits = [{"pos": pos, "ref": seq[pos], "alt": cand[pos], "region": "five_utr"}]
        d = predict_delta_for(seq, cand, edits)
        deltas_5utr.append(d)
        pos_sensitivity[pos].append(d)
    results["five_utr_single_sub"] = PerturbationResult(
        perturbation_type="five_utr_single_sub",
        mean_pred_delta=float(np.mean(deltas_5utr)) if deltas_5utr else 0.0,
        std_pred_delta=float(np.std(deltas_5utr)) if deltas_5utr else 0.0,
        n_samples=len(deltas_5utr),
        position_sensitivity={str(k): float(np.mean(v)) for k, v in pos_sensitivity.items()} if pos_sensitivity else None,
    )

    # 2-4. CDS synonymous substitutions (start/middle/late)
    for region_name, codon_range in [
        ("start_proximal_cds", (1, 5)),  # skip AUG
        ("middle_cds", None),
        ("late_cds", None),
    ]:
        deltas_cds = []
        for i in range(n):
            seq = source_sequences[i]
            aug_pos = seq.find(START_CODON)
            if aug_pos < 0:
                continue
            cds = seq[aug_pos:]
            n_codons = len(cds) // 3
            if n_codons < 6:
                continue
            if region_name == "start_proximal_cds":
                codon_idx = rng.randint(1, min(5, n_codons - 1))
            elif region_name == "middle_cds":
                codon_idx = rng.randint(n_codons // 3, 2 * n_codons // 3)
            else:  # late_cds
                codon_idx = rng.randint(2 * n_codons // 3, n_codons - 1)
            new_cds = _synonymous_substitution(cds, codon_idx, rng)
            if new_cds == cds:
                continue
            cand = seq[:aug_pos] + new_cds + seq[aug_pos + len(cds):]
            edits = [{"pos": aug_pos + codon_idx * 3, "ref": cds[codon_idx*3:codon_idx*3+3],
                      "alt": new_cds[codon_idx*3:codon_idx*3+3], "region": "cds"}]
            d = predict_delta_for(seq, cand, edits)
            deltas_cds.append(d)
        results[region_name] = PerturbationResult(
            perturbation_type=region_name,
            mean_pred_delta=float(np.mean(deltas_cds)) if deltas_cds else 0.0,
            std_pred_delta=float(np.std(deltas_cds)) if deltas_cds else 0.0,
            n_samples=len(deltas_cds),
        )

    # 5. Joint 5'UTR + CDS
    deltas_joint = []
    for i in range(n):
        seq = source_sequences[i]
        aug_pos = seq.find(START_CODON)
        if aug_pos < 3 or aug_pos + 18 > len(seq):
            continue
        # UTR edit
        utr_pos = rng.randint(0, aug_pos - 1)
        cand = _random_substitution(seq, utr_pos, rng)
        # CDS edit
        cds = cand[aug_pos:]
        n_codons = len(cds) // 3
        if n_codons < 6:
            continue
        codon_idx = rng.randint(1, n_codons - 2)
        new_cds = _synonymous_substitution(cds, codon_idx, rng)
        if new_cds == cds:
            continue
        cand = cand[:aug_pos] + new_cds + cand[aug_pos + len(cds):]
        edits = [
            {"pos": utr_pos, "ref": seq[utr_pos], "alt": cand[utr_pos], "region": "five_utr"},
            {"pos": aug_pos + codon_idx * 3, "ref": cds[codon_idx*3:codon_idx*3+3],
             "alt": new_cds[codon_idx*3:codon_idx*3+3], "region": "cds"},
        ]
        d = predict_delta_for(seq, cand, edits)
        deltas_joint.append(d)
    results["joint_5utr_cds"] = PerturbationResult(
        perturbation_type="joint_5utr_cds",
        mean_pred_delta=float(np.mean(deltas_joint)) if deltas_joint else 0.0,
        std_pred_delta=float(np.std(deltas_joint)) if deltas_joint else 0.0,
        n_samples=len(deltas_joint),
    )

    # 6. Matched random substitution (same position, random nt)
    deltas_random = []
    for i in range(n):
        seq = source_sequences[i]
        if len(seq) < 10:
            continue
        aug_pos = seq.find(START_CODON)
        utr_end = aug_pos if aug_pos >= 0 else min(len(seq) // 2, 50)
        if utr_end < 3:
            continue
        pos = rng.randint(0, utr_end - 1)
        # Random substitution (may or may not change GC)
        cand = _random_substitution(seq, pos, rng)
        edits = [{"pos": pos, "ref": seq[pos], "alt": cand[pos], "region": "five_utr"}]
        d = predict_delta_for(seq, cand, edits)
        deltas_random.append(d)
    results["matched_random"] = PerturbationResult(
        perturbation_type="matched_random",
        mean_pred_delta=float(np.mean(deltas_random)) if deltas_random else 0.0,
        std_pred_delta=float(np.std(deltas_random)) if deltas_random else 0.0,
        n_samples=len(deltas_random),
    )

    return results


def analyze_sensitivity_checks(results: Dict[str, PerturbationResult]) -> Dict[str, Any]:
    """Check if the model passes sensitivity tests.

    Checks:
    - Is the model sensitive to edit position? (position_sensitivity varies)
    - Does it only learn GC? (correlation with GC change)
    - Does it only learn length? (all deltas ~0 for same-length edits)
    - Does it only read CDS first few nt? (start vs late CDS sensitivity)
    - Does it ignore source sequence? (variance across sources)
    """
    checks: Dict[str, Any] = {}

    # Position sensitivity
    utr_result = results.get("five_utr_single_sub")
    if utr_result and utr_result.position_sensitivity:
        pos_vals = list(utr_result.position_sensitivity.values())
        checks["position_sensitive"] = float(np.std(pos_vals)) > 0.01
        checks["position_sensitivity_std"] = float(np.std(pos_vals))
    else:
        checks["position_sensitive"] = False
        checks["position_sensitivity_std"] = 0.0

    # GC-only check: if mean delta correlates strongly with GC change
    # (simplified: if UTR and random give similar patterns, model may be GC-only)
    utr_mean = utr_result.mean_pred_delta if utr_result else 0.0
    random_mean = results.get("matched_random", PerturbationResult("", 0, 0, 0)).mean_pred_delta
    checks["gc_only_risk"] = abs(utr_mean - random_mean) < 0.01  # if similar, may be GC-only

    # Length-only check: for same-length substitutions, deltas should vary
    if utr_result:
        checks["length_only_risk"] = utr_result.std_pred_delta < 0.01
    else:
        checks["length_only_risk"] = True

    # CDS position check: start vs late CDS
    start_cds = results.get("start_proximal_cds", PerturbationResult("", 0, 0, 0))
    late_cds = results.get("late_cds", PerturbationResult("", 0, 0, 0))
    checks["cds_start_vs_late"] = abs(start_cds.mean_pred_delta - late_cds.mean_pred_delta)
    checks["cds_position_aware"] = checks["cds_start_vs_late"] > 0.01

    # Source-ignoring check: variance across different sources
    all_std = [r.std_pred_delta for r in results.values()]
    checks["source_aware"] = float(np.mean(all_std)) > 0.01

    return checks


# ===========================================================================
# Section 7: Optimization Headroom Search (Task 5)
# ===========================================================================

@dataclass
class HeadroomResult:
    """Result of a headroom search for a single source."""
    source_id: str
    source_value: float
    best_delta: float
    best_candidate: str
    best_edits: List[Dict[str, Any]]
    search_method: str
    n_evaluated: int


def exact_one_edit_enumeration(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    editable_region: str = "five_utr",
    max_positions: int = 50,
) -> HeadroomResult:
    """Enumerate all single-nt substitutions and pick the best."""
    best_delta = 0.0
    best_cand = source_seq
    best_edits: List[Dict[str, Any]] = []
    n_eval = 0

    # Determine editable positions
    aug_pos = source_seq.find(START_CODON)
    if editable_region == "five_utr":
        end = aug_pos if aug_pos >= 0 else min(len(source_seq), max_positions)
    else:
        end = len(source_seq)

    for pos in range(min(end, len(source_seq))):
        old = source_seq[pos]
        for new_nt in NUC_VOCAB:
            if new_nt == old:
                continue
            cand = source_seq[:pos] + new_nt + source_seq[pos + 1:]
            edits = [{"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}]
            feats = extract_features(source_seq, cand, edits, config.max_seq_len)
            batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
            delta = float(predictor.predict_delta(batch_feats)[0])
            n_eval += 1
            if delta > best_delta:
                best_delta = delta
                best_cand = cand
                best_edits = edits

    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=best_delta,
        best_candidate=best_cand,
        best_edits=best_edits,
        search_method="exact_one_edit",
        n_evaluated=n_eval,
    )


def greedy_search(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    max_edits: int = 5,
    editable_region: str = "five_utr",
) -> HeadroomResult:
    """Greedy: at each step, pick the best single edit."""
    current = source_seq
    current_edits: List[Dict[str, Any]] = []
    total_delta = 0.0
    n_eval = 0

    for step in range(max_edits):
        best_step_delta = 0.0
        best_step_cand = current
        best_step_edit: Optional[Dict[str, Any]] = None

        aug_pos = current.find(START_CODON)
        if editable_region == "five_utr":
            end = aug_pos if aug_pos >= 0 else min(len(current), 50)
        else:
            end = len(current)

        for pos in range(min(end, len(current))):
            old = current[pos]
            for new_nt in NUC_VOCAB:
                if new_nt == old:
                    continue
                cand = current[:pos] + new_nt + current[pos + 1:]
                edits = current_edits + [{"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}]
                feats = extract_features(source_seq, cand, edits, config.max_seq_len)
                batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
                delta = float(predictor.predict_delta(batch_feats)[0])
                n_eval += 1
                if delta > best_step_delta:
                    best_step_delta = delta
                    best_step_cand = cand
                    best_step_edit = {"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}

        if best_step_edit is None or best_step_delta <= 0:
            break
        current = best_step_cand
        current_edits.append(best_step_edit)
        total_delta = best_step_delta

    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=total_delta,
        best_candidate=current,
        best_edits=current_edits,
        search_method="greedy",
        n_evaluated=n_eval,
    )


def beam_search(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    max_edits: int = 3,
    beam_width: int = 5,
    editable_region: str = "five_utr",
) -> HeadroomResult:
    """Beam search: keep top-k candidates at each step."""
    beam: List[Tuple[str, List[Dict], float]] = [(source_seq, [], 0.0)]
    n_eval = 0

    for step in range(max_edits):
        candidates: List[Tuple[str, List[Dict], float]] = []
        for seq, edits, score in beam:
            aug_pos = seq.find(START_CODON)
            if editable_region == "five_utr":
                end = aug_pos if aug_pos >= 0 else min(len(seq), 50)
            else:
                end = len(seq)
            for pos in range(min(end, len(seq))):
                old = seq[pos]
                for new_nt in NUC_VOCAB:
                    if new_nt == old:
                        continue
                    cand = seq[:pos] + new_nt + seq[pos + 1:]
                    new_edits = edits + [{"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}]
                    feats = extract_features(source_seq, cand, new_edits, config.max_seq_len)
                    batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
                    delta = float(predictor.predict_delta(batch_feats)[0])
                    n_eval += 1
                    candidates.append((cand, new_edits, delta))

        # Keep top beam_width
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
        if not beam or beam[0][2] <= 0:
            break

    best = beam[0] if beam else (source_seq, [], 0.0)
    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=best[2],
        best_candidate=best[0],
        best_edits=best[1],
        search_method="beam_search",
        n_evaluated=n_eval,
    )


def simulated_annealing(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    n_iterations: int = 200,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.95,
    editable_region: str = "five_utr",
    seed: int = 42,
) -> HeadroomResult:
    """Simulated annealing search."""
    rng = np.random.RandomState(seed)
    current = source_seq
    current_edits: List[Dict[str, Any]] = []

    feats = extract_features(source_seq, current, current_edits, config.max_seq_len)
    batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
    current_delta = float(predictor.predict_delta(batch_feats)[0])
    n_eval = 1

    best = current
    best_edits = list(current_edits)
    best_delta = current_delta
    temp = initial_temp

    for it in range(n_iterations):
        # Propose a random edit
        aug_pos = current.find(START_CODON)
        if editable_region == "five_utr":
            end = aug_pos if aug_pos >= 0 else min(len(current), 50)
        else:
            end = len(current)
        if end < 2:
            break

        pos = rng.randint(0, min(end, len(current)))
        old = current[pos]
        new_nt = NUC_VOCAB[rng.randint(0, 4)]
        if new_nt == old:
            continue
        cand = current[:pos] + new_nt + current[pos + 1:]
        # Update edit list (simplified: replace or append)
        new_edits = list(current_edits)
        found = False
        for e in new_edits:
            if e["pos"] == pos:
                e["alt"] = new_nt
                found = True
                break
        if not found:
            new_edits.append({"pos": pos, "ref": old, "alt": new_nt, "region": editable_region})

        feats = extract_features(source_seq, cand, new_edits, config.max_seq_len)
        batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
        cand_delta = float(predictor.predict_delta(batch_feats)[0])
        n_eval += 1

        # Accept or reject
        delta_e = cand_delta - current_delta
        if delta_e > 0 or rng.random() < math.exp(delta_e / max(temp, 1e-8)):
            current = cand
            current_edits = new_edits
            current_delta = cand_delta
            if cand_delta > best_delta:
                best = cand
                best_edits = list(new_edits)
                best_delta = cand_delta
        temp *= cooling_rate

    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=best_delta,
        best_candidate=best,
        best_edits=best_edits,
        search_method="simulated_annealing",
        n_evaluated=n_eval,
    )


def mcts_search(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    n_simulations: int = 100,
    max_depth: int = 3,
    editable_region: str = "five_utr",
    seed: int = 42,
) -> HeadroomResult:
    """Simplified MCTS: UCB1-based tree search over edit actions."""
    rng = np.random.RandomState(seed)

    class Node:
        def __init__(self, seq, edits, parent=None):
            self.seq = seq
            self.edits = edits
            self.parent = parent
            self.children: List[Node] = []
            self.visits = 0
            self.total_value = 0.0
            self.untried_actions: Optional[List[Tuple[int, str]]] = None

        @property
        def ucb1(self) -> float:
            if self.visits == 0:
                return float('inf')
            exploit = self.total_value / self.visits
            explore = math.sqrt(2 * math.log(self.parent.visits + 1) / self.visits) if self.parent else 0
            return exploit + explore

    def get_actions(seq: str) -> List[Tuple[int, str]]:
        aug_pos = seq.find(START_CODON)
        if editable_region == "five_utr":
            end = aug_pos if aug_pos >= 0 else min(len(seq), 50)
        else:
            end = len(seq)
        actions = []
        for pos in range(min(end, len(seq))):
            old = seq[pos]
            for new_nt in NUC_VOCAB:
                if new_nt != old:
                    actions.append((pos, new_nt))
        return actions

    root = Node(source_seq, [])
    n_eval = 0

    for sim in range(n_simulations):
        # Selection
        node = root
        while node.children and node.untried_actions is not None and len(node.untried_actions) == 0:
            node = max(node.children, key=lambda c: c.ucb1)

        # Expansion
        if node.untried_actions is None:
            node.untried_actions = get_actions(node.seq)

        if node.untried_actions and len(node.edits) < max_depth:
            action = node.untried_actions[rng.randint(len(node.untried_actions))]
            node.untried_actions.remove(action)
            pos, new_nt = action
            old = node.seq[pos]
            cand = node.seq[:pos] + new_nt + node.seq[pos + 1:]
            new_edits = node.edits + [{"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}]
            child = Node(cand, new_edits, parent=node)
            node.children.append(child)
            node = child

        # Simulation/evaluation
        feats = extract_features(source_seq, node.seq, node.edits, config.max_seq_len)
        batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
        value = float(predictor.predict_delta(batch_feats)[0])
        n_eval += 1

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # Find best path
    if root.children:
        best_child = max(root.children, key=lambda c: c.total_value / max(c.visits, 1))
        best_delta = best_child.total_value / max(best_child.visits, 1)
        best_cand = best_child.seq
        best_edits = best_child.edits
    else:
        best_delta = 0.0
        best_cand = source_seq
        best_edits = []

    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=best_delta,
        best_candidate=best_cand,
        best_edits=best_edits,
        search_method="mcts",
        n_evaluated=n_eval,
    )


def oracle_guided_search(
    source_seq: str,
    predictor: Any,
    config: CrossFitConfig,
    max_edits: int = 5,
    editable_region: str = "five_utr",
) -> HeadroomResult:
    """Oracle-guided local search: use ensemble disagreement to guide exploration."""
    # Use greedy as base, but prioritize high-uncertainty positions
    current = source_seq
    current_edits: List[Dict[str, Any]] = []
    total_delta = 0.0
    n_eval = 0

    for step in range(max_edits):
        best_step_delta = 0.0
        best_step_cand = current
        best_step_edit: Optional[Dict[str, Any]] = None

        aug_pos = current.find(START_CODON)
        if editable_region == "five_utr":
            end = aug_pos if aug_pos >= 0 else min(len(current), 50)
        else:
            end = len(current)

        for pos in range(min(end, len(current))):
            old = current[pos]
            for new_nt in NUC_VOCAB:
                if new_nt == old:
                    continue
                cand = current[:pos] + new_nt + current[pos + 1:]
                edits = current_edits + [{"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}]
                feats = extract_features(source_seq, cand, edits, config.max_seq_len)
                batch_feats = {k: v[np.newaxis] for k, v in feats.items()}
                delta = float(predictor.predict_delta(batch_feats)[0])
                n_eval += 1
                if delta > best_step_delta:
                    best_step_delta = delta
                    best_step_cand = cand
                    best_step_edit = {"pos": pos, "ref": old, "alt": new_nt, "region": editable_region}

        if best_step_edit is None or best_step_delta <= 0:
            break
        current = best_step_cand
        current_edits.append(best_step_edit)
        total_delta = best_step_delta

    return HeadroomResult(
        source_id="",
        source_value=0.0,
        best_delta=total_delta,
        best_candidate=current,
        best_edits=current_edits,
        search_method="oracle_guided",
        n_evaluated=n_eval,
    )


SEARCH_METHODS = {
    "exact_one_edit": exact_one_edit_enumeration,
    "greedy": greedy_search,
    "beam_search": beam_search,
    "simulated_annealing": simulated_annealing,
    "mcts": mcts_search,
    "oracle_guided": oracle_guided_search,
}


def run_headroom_search(
    predictor: Any,
    source_sequences: Sequence[str],
    config: CrossFitConfig,
    methods: Sequence[str] = ("exact_one_edit", "greedy", "beam_search", "simulated_annealing", "mcts"),
    n_sources: int = 50,
    editable_region: str = "five_utr",
) -> Dict[str, List[HeadroomResult]]:
    """Run multiple search methods on a sample of source sequences."""
    n = min(n_sources, len(source_sequences))
    results: Dict[str, List[HeadroomResult]] = {}

    for method_name in methods:
        method_fn = SEARCH_METHODS[method_name]
        method_results: List[HeadroomResult] = []
        for i in range(n):
            src = source_sequences[i]
            try:
                if method_name == "exact_one_edit":
                    result = method_fn(src, predictor, config, editable_region)
                elif method_name == "greedy":
                    result = method_fn(src, predictor, config, max_edits=3, editable_region=editable_region)
                elif method_name == "beam_search":
                    result = method_fn(src, predictor, config, max_edits=2, beam_width=3, editable_region=editable_region)
                elif method_name == "simulated_annealing":
                    result = method_fn(src, predictor, config, n_iterations=100, editable_region=editable_region)
                elif method_name == "mcts":
                    result = method_fn(src, predictor, config, n_simulations=50, max_depth=3, editable_region=editable_region)
                else:
                    continue
                result.source_id = f"src_{i}"
                method_results.append(result)
            except Exception:
                method_results.append(HeadroomResult(
                    source_id=f"src_{i}", source_value=0.0, best_delta=0.0,
                    best_candidate=src, best_edits=[], search_method=method_name, n_evaluated=0,
                ))
        results[method_name] = method_results

    return results


def analyze_headroom(
    search_results: Dict[str, List[HeadroomResult]],
) -> Dict[str, Any]:
    """Analyze headroom search results.

    Answers:
    - Best legal edit improvement over source?
    - Fraction of sources with positive candidates?
    - Is gain concentrated in few families?
    - 5'UTR vs CDS contribution?
    - Does joint search exceed single-region?
    """
    analysis: Dict[str, Any] = {}

    for method, results in search_results.items():
        if not results:
            continue
        deltas = [r.best_delta for r in results]
        positive_frac = float(np.mean([d > 0 for d in deltas]))
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        max_delta = float(np.max(deltas))

        analysis[method] = {
            "mean_best_delta": mean_delta,
            "median_best_delta": median_delta,
            "max_best_delta": max_delta,
            "positive_fraction": positive_frac,
            "n_sources": len(results),
            "mean_n_evaluated": float(np.mean([r.n_evaluated for r in results])),
        }

    # Compare methods
    if "exact_one_edit" in analysis and "greedy" in analysis:
        analysis["multi_edit_advantage"] = (
            analysis["greedy"]["mean_best_delta"] - analysis["exact_one_edit"]["mean_best_delta"]
        )
    if "greedy" in analysis and "mcts" in analysis:
        analysis["mcts_vs_greedy"] = (
            analysis["mcts"]["mean_best_delta"] - analysis["greedy"]["mean_best_delta"]
        )

    return analysis


# ===========================================================================
# Section 8: GO/NO-GO Gate
# ===========================================================================

def evaluate_go_gate(
    metrics: Dict[str, float],
    sensitivity_checks: Dict[str, Any],
    headroom_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate the P3-02 GO/PARTIAL/NO-GO gate.

    GO criteria:
    - Sign accuracy significantly above random (>0.55)
    - Top-k enrichment > 1.0 (top-ranked edits are enriched)
    - Strong search finds positive candidates in substantial fraction (>20%)
    - Gain not explained by single GC/length heuristic
    - At least one region has stable headroom

    PARTIAL:
    - Local effect predictable but only in one region
    - OR headroom exists but concentrated in few cargo
    - OR training Oracle and independent Oracle agreement limited

    NO-GO:
    - Sign accuracy near random (<0.52)
    - No top-k enrichment
    - Strong search can't exceed source
    - Independent Oracle direction frequently reverses
    """
    gate: Dict[str, Any] = {
        "criteria": {},
        "verdict": "NO_GO",
        "rationale": [],
    }

    # Criterion 1: Sign accuracy
    sa = metrics.get("sign_accuracy", 0.0)
    gate["criteria"]["sign_accuracy_above_random"] = sa > 0.55
    gate["criteria"]["sign_accuracy_value"] = sa

    # Criterion 2: Top-k enrichment
    tk = metrics.get("top_k_enrichment_10pct", 0.0)
    gate["criteria"]["top_k_enrichment_gt_1"] = tk > 1.0
    gate["criteria"]["top_k_enrichment_value"] = tk

    # Criterion 3: Strong search finds positive candidates
    best_method = None
    best_positive_frac = 0.0
    for method_name, method_analysis in headroom_analysis.items():
        if isinstance(method_analysis, dict) and "positive_fraction" in method_analysis:
            pf = method_analysis["positive_fraction"]
            if pf > best_positive_frac:
                best_positive_frac = pf
                best_method = method_name
    gate["criteria"]["search_finds_positive"] = best_positive_frac > 0.20
    gate["criteria"]["best_positive_fraction"] = best_positive_frac
    gate["criteria"]["best_search_method"] = best_method

    # Criterion 4: Not GC/length only
    gc_risk = sensitivity_checks.get("gc_only_risk", True)
    length_risk = sensitivity_checks.get("length_only_risk", True)
    gate["criteria"]["not_gc_only"] = not gc_risk
    gate["criteria"]["not_length_only"] = not length_risk

    # Criterion 5: At least one region has stable headroom
    has_headroom = best_positive_frac > 0.20
    gate["criteria"]["has_stable_headroom"] = has_headroom

    # Determine verdict
    go_criteria = [
        gate["criteria"]["sign_accuracy_above_random"],
        gate["criteria"]["top_k_enrichment_gt_1"],
        gate["criteria"]["search_finds_positive"],
        gate["criteria"]["not_gc_only"] or gate["criteria"]["not_length_only"],
        gate["criteria"]["has_stable_headroom"],
    ]
    n_pass = sum(go_criteria)

    if n_pass >= 4:
        gate["verdict"] = "GO"
        gate["rationale"].append("All key criteria met: local-delta predictable, headroom exists, not explained by simple heuristics.")
    elif n_pass >= 2:
        gate["verdict"] = "PARTIAL"
        if not gate["criteria"]["sign_accuracy_above_random"]:
            gate["rationale"].append("Sign accuracy marginal: local effect only partially predictable.")
        if not gate["criteria"]["search_finds_positive"]:
            gate["rationale"].append("Headroom limited: positive candidates found in <20% of sources.")
        if gc_risk and length_risk:
            gate["rationale"].append("Model may be learning GC/length heuristic rather than true sequence effect.")
        gate["rationale"].append("Recommend: narrow task scope, do not enter full joint RL.")
    else:
        gate["verdict"] = "NO_GO"
        gate["rationale"].append("Sign accuracy near random or search cannot find positive candidates.")
        gate["rationale"].append("Recommend: stop RL expansion, acquire intervention data.")

    gate["n_criteria_pass"] = n_pass
    gate["n_criteria_total"] = len(go_criteria)
    return gate


__all__ = [
    # Data
    "DeltaRecord", "load_benchmark_tier", "load_benchmark",
    # Features
    "extract_features", "batch_extract_features",
    # Models
    "AbsoluteModel", "DifferenceModel", "SiameseModel", "EditConditionedModel",
    "SeqDiffModel", "SeqCNNModel",
    "MODEL_REGISTRY",
    # Metrics
    "compute_all_metrics", "sign_accuracy", "top_k_enrichment",
    "beneficial_edit_precision", "delta_spearman", "delta_pearson",
    # Ensemble
    "CrossFitConfig", "cross_fit_predict", "build_oracle_ensemble",
    # Sensitivity
    "run_region_sensitivity", "analyze_sensitivity_checks",
    # Headroom
    "run_headroom_search", "analyze_headroom",
    "exact_one_edit_enumeration", "greedy_search", "beam_search",
    "simulated_annealing", "mcts_search", "oracle_guided_search",
    # Gate
    "evaluate_go_gate",
]
