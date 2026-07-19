"""Independent local MRL/TE oracle for mRNA-EditFlow evaluation.

The oracle intentionally depends only on lightweight sequence features and an
optional tiny 5'UTR CNN. It does not import or share parameters with the
generation model, so evaluation remains independent from training-time heads.
The deterministic feature-regressor fallback is always available offline.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

RNA_ALPHABET = ("A", "C", "G", "U")
RNA_SET = set(RNA_ALPHABET)
PAD_ID = len(RNA_ALPHABET)
NT_TO_ID = {nt: i for i, nt in enumerate(RNA_ALPHABET)}


def _normalise_rna(seq: str) -> str:
    """Upper-case RNA string with DNA T converted to U; whitespace removed."""
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _valid_only(seq: str) -> str:
    return "".join(ch for ch in _normalise_rna(seq) if ch in RNA_SET)


def _gc_fraction(seq: str) -> float:
    seq = _valid_only(seq)
    if not seq:
        return 0.0
    return float(seq.count("G") + seq.count("C")) / float(len(seq))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _count_overlapping(seq: str, motif: str) -> int:
    if not motif:
        return 0
    count = 0
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def _kozak_score(five_utr: str, cds_start_context: str = "") -> float:
    """Score the -3/+4 Kozak positions around the annotated start codon.

    The canonical mammalian signal has A/G at -3 and G at +4 relative to AUG.
    If +4 is unavailable, the score gracefully uses the available -3 evidence.
    """
    utr = _valid_only(five_utr)
    cds = _valid_only(cds_start_context)
    upstream = utr[-6:]
    context = upstream + cds
    aug = context.find("AUG", len(upstream) - 1 if len(upstream) else 0)
    if aug < 0:
        aug = len(upstream)
    minus3 = context[aug - 3] if aug >= 3 else ""
    plus4 = context[aug + 3] if aug + 3 < len(context) else ""
    score = 0.0
    denom = 0.0
    if minus3:
        score += 1.0 if minus3 in ("A", "G") else 0.0
        denom += 1.0
    if plus4:
        score += 1.0 if plus4 == "G" else 0.0
        denom += 1.0
    return float(score / denom) if denom else 0.0


def _start_accessibility_proxy(five_utr: str) -> float:
    """A deterministic local accessibility proxy near the start codon.

    High GC and self-complementarity in the last 30 nt of 5'UTR reduce the
    score. This is not a folding engine; it is a stable offline proxy for
    ranking and unit-testable benchmark plumbing.
    """
    seq = _valid_only(five_utr)
    if not seq:
        return 1.0
    window = seq[-30:]
    gc = _gc_fraction(window)
    pairs = 0
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}
    for i in range(len(window) // 2):
        if complement.get(window[i]) == window[-1 - i]:
            pairs += 1
    pair_frac = pairs / max(1.0, len(window) / 2.0)
    return float(max(0.0, min(1.0, 1.0 - 0.55 * gc - 0.35 * pair_frac)))


def extract_utr_features(five_utr: str, cds_start_context: str = "") -> dict:
    """Extract deterministic 5'UTR features used by both fallback predictors."""
    raw = _normalise_rna(five_utr)
    seq = _valid_only(raw)
    invalid = max(0, len(raw) - len(seq))
    length = len(seq)
    uaug = _count_overlapping(seq, "AUG")
    stop_like = sum(_count_overlapping(seq, motif) for motif in ("UAA", "UAG", "UGA"))
    pyrimidine_start = len(seq) > 0 and all(ch in "CU" for ch in seq[: min(8, len(seq))])
    gc = _gc_fraction(seq)
    gc_opt = max(0.0, 1.0 - abs(gc - 0.52) / 0.52)
    len_opt = math.exp(-abs(length - 70.0) / 90.0) if length else 0.25
    access = _start_accessibility_proxy(seq)
    kozak = _kozak_score(seq, cds_start_context)
    top_motif = 1.0 if pyrimidine_start else 0.0
    poly_gc_run = 1.0 if ("GGGG" in seq or "CCCC" in seq) else 0.0
    return {
        "length": float(length),
        "gc": float(gc),
        "gc_opt": float(gc_opt),
        "length_opt": float(len_opt),
        "kozak": float(kozak),
        "uaug_count": float(uaug),
        "stop_like_count": float(stop_like),
        "start_accessibility": float(access),
        "top_motif": float(top_motif),
        "poly_gc_run": float(poly_gc_run),
        "invalid_fraction": float(invalid / max(1, len(raw))),
    }


class FivePrimeCNN(nn.Module):
    """Tiny Optimus-5Prime-style CNN for optional train/load oracle weights."""

    def __init__(self, embed_dim: int = 8, hidden_dim: int = 24) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(RNA_ALPHABET) + 1, embed_dim, padding_idx=PAD_ID)
        self.conv3 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, token_ids: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        emb = self.embed(token_ids).transpose(1, 2)
        x3 = self.act(self.conv3(emb)).amax(dim=-1)
        x5 = self.act(self.conv5(emb)).amax(dim=-1)
        return self.head(torch.cat([x3, x5, aux], dim=-1))


def _encode_utrs(utrs: Sequence[str], max_len: int) -> torch.Tensor:
    arr = torch.full((len(utrs), max_len), PAD_ID, dtype=torch.long)
    for row, seq in enumerate(utrs):
        clean = _valid_only(seq)[:max_len]
        if clean:
            arr[row, : len(clean)] = torch.tensor(
                [NT_TO_ID[ch] for ch in clean], dtype=torch.long
            )
    return arr


def _aux_tensor(utrs: Sequence[str]) -> torch.Tensor:
    rows = []
    for utr in utrs:
        f = extract_utr_features(utr)
        rows.append([f["gc"], f["length"] / 200.0, f["start_accessibility"]])
    return torch.tensor(rows, dtype=torch.float32)


@dataclass
class OracleScore:
    """Serializable oracle output for one 5'UTR or transcript record."""

    mrl: float
    te: float
    predictor2_mrl: float
    predictor2_te: float
    ensemble_mrl: float
    ensemble_te: float
    agreement: float
    uncertainty: float
    features: Mapping[str, float]

    def to_dict(self) -> dict:
        return {
            "mrl": float(self.mrl),
            "te": float(self.te),
            "predictor2_mrl": float(self.predictor2_mrl),
            "predictor2_te": float(self.predictor2_te),
            "ensemble_mrl": float(self.ensemble_mrl),
            "ensemble_te": float(self.ensemble_te),
            "agreement": float(self.agreement),
            "uncertainty": float(self.uncertainty),
            "features": {k: float(v) for k, v in self.features.items()},
        }


class LocalTranslationOracle:
    """Independent local MRL/TE oracle with deterministic offline fallback."""

    def __init__(
        self,
        max_len: int = 160,
        seed: int = 1729,
        use_cnn_when_available: bool = True,
    ) -> None:
        self.max_len = int(max_len)
        self.seed = int(seed)
        self.use_cnn_when_available = bool(use_cnn_when_available)
        torch.manual_seed(self.seed)
        self.cnn = FivePrimeCNN()
        self._cnn_available = False

    @staticmethod
    def _primary_regressor(features: Mapping[str, float]) -> tuple[float, float]:
        raw = (
            1.20 * features["gc_opt"]
            + 1.00 * features["kozak"]
            + 0.90 * features["start_accessibility"]
            + 0.35 * features["length_opt"]
            + 0.10 * features["top_motif"]
            - 0.42 * features["uaug_count"]
            - 0.16 * features["stop_like_count"]
            - 0.30 * features["poly_gc_run"]
            - 1.20 * features["invalid_fraction"]
            - 0.15
        )
        te = _sigmoid(raw)
        mrl = 2.0 + 8.0 * te
        return float(mrl), float(te)

    @staticmethod
    def _secondary_regressor(features: Mapping[str, float]) -> tuple[float, float]:
        """Different deterministic predictor for cross-checking oracle signal."""
        raw = (
            1.05 * features["start_accessibility"]
            + 0.82 * features["gc_opt"]
            + 0.74 * features["kozak"]
            + 0.45 * features["length_opt"]
            - 0.35 * math.log1p(features["uaug_count"])
            - 0.24 * features["poly_gc_run"]
            - 0.85 * features["invalid_fraction"]
            + 0.06 * features["top_motif"]
            - 0.05
        )
        te = _sigmoid(raw)
        mrl = 1.8 + 8.4 * te
        return float(mrl), float(te)

    def fit(
        self,
        utrs: Sequence[str],
        targets: Sequence[Mapping[str, float] | Sequence[float]],
        epochs: int = 120,
        lr: float = 2e-3,
        seed: Optional[int] = None,
    ) -> "LocalTranslationOracle":
        """Train the optional CNN predictor on local MRL/TE targets.

        ``targets`` may contain mappings with ``mrl``/``te`` keys or two-value
        sequences ``(mrl, te)``. The fallback feature regressors are unchanged.
        """
        if len(utrs) != len(targets):
            raise ValueError("utrs and targets must have the same length")
        if not utrs:
            raise ValueError("at least one training example is required")
        if seed is not None:
            torch.manual_seed(int(seed))
        y_rows = []
        for target in targets:
            if isinstance(target, Mapping):
                y_rows.append([float(target["mrl"]), float(target["te"])])
            else:
                if len(target) != 2:  # type: ignore[arg-type]
                    raise ValueError("target sequences must be (mrl, te)")
                y_rows.append([float(target[0]), float(target[1])])  # type: ignore[index]
        x = _encode_utrs(utrs, self.max_len)
        aux = _aux_tensor(utrs)
        y = torch.tensor(y_rows, dtype=torch.float32)
        opt = torch.optim.Adam(self.cnn.parameters(), lr=float(lr))
        self.cnn.train()
        for _ in range(int(epochs)):
            opt.zero_grad(set_to_none=True)
            pred = self.cnn(x, aux)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
        self.cnn.eval()
        self._cnn_available = True
        return self

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.cnn.state_dict(),
                "max_len": self.max_len,
                "seed": self.seed,
            },
            path,
        )

    def load_weights(self, path: str) -> "LocalTranslationOracle":
        payload = torch.load(path, map_location="cpu")
        self.max_len = int(payload.get("max_len", self.max_len))
        self.seed = int(payload.get("seed", self.seed))
        self.cnn.load_state_dict(payload["state_dict"])
        self.cnn.eval()
        self._cnn_available = True
        return self

    def _cnn_score(self, five_utr: str) -> Optional[tuple[float, float]]:
        if not (self._cnn_available and self.use_cnn_when_available):
            return None
        with torch.no_grad():
            out = self.cnn(_encode_utrs([five_utr], self.max_len), _aux_tensor([five_utr]))
        mrl = float(out[0, 0].detach().cpu())
        te = float(out[0, 1].detach().cpu())
        if not (math.isfinite(mrl) and math.isfinite(te)):
            return None
        return mrl, max(0.0, min(1.0, te))

    def score_utr(self, five_utr: str, cds_start_context: str = "") -> dict:
        features = extract_utr_features(five_utr, cds_start_context)
        p1 = self._cnn_score(five_utr) or self._primary_regressor(features)
        p2 = self._secondary_regressor(features)
        mrl_delta = abs(p1[0] - p2[0]) / 10.0
        te_delta = abs(p1[1] - p2[1])
        uncertainty = float(max(0.0, min(1.0, 0.5 * mrl_delta + 0.5 * te_delta)))
        agreement = float(1.0 - uncertainty)
        return OracleScore(
            mrl=p1[0],
            te=p1[1],
            predictor2_mrl=p2[0],
            predictor2_te=p2[1],
            ensemble_mrl=0.5 * (p1[0] + p2[0]),
            ensemble_te=0.5 * (p1[1] + p2[1]),
            agreement=agreement,
            uncertainty=uncertainty,
            features=features,
        ).to_dict()

    def score_record(self, record: object) -> dict:
        five_utr = _field(record, "five_utr", "")
        cds = _field(record, "cds", "")
        score = self.score_utr(five_utr, cds[:12])
        tid = _field(record, "transcript_id", None)
        if tid is not None:
            score["transcript_id"] = tid
        return score

    def batch_score(self, records_or_utrs: Iterable[object]) -> List[dict]:
        scores = []
        for item in records_or_utrs:
            if isinstance(item, str):
                scores.append(self.score_utr(item))
            else:
                scores.append(self.score_record(item))
        return scores

    def cross_validate_predictors(self, records_or_utrs: Iterable[object]) -> dict:
        scores = self.batch_score(records_or_utrs)
        if not scores:
            return {
                "n": 0,
                "mrl_mae_between_predictors": 0.0,
                "te_mae_between_predictors": 0.0,
                "agreement_mean": 1.0,
            }
        mrl_mae = float(np.mean([abs(s["mrl"] - s["predictor2_mrl"]) for s in scores]))
        te_mae = float(np.mean([abs(s["te"] - s["predictor2_te"]) for s in scores]))
        agreement = float(np.mean([s["agreement"] for s in scores]))
        return {
            "n": len(scores),
            "mrl_mae_between_predictors": mrl_mae,
            "te_mae_between_predictors": te_mae,
            "agreement_mean": agreement,
        }


def _field(record: object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


_DEFAULT_ORACLE = LocalTranslationOracle()


def score_utr(five_utr: str, cds_start_context: str = "") -> dict:
    """Score one 5'UTR with the default deterministic oracle."""
    return _DEFAULT_ORACLE.score_utr(five_utr, cds_start_context)


def score_record(record: object) -> dict:
    """Score one transcript-like object or mapping."""
    return _DEFAULT_ORACLE.score_record(record)


def batch_score(records_or_utrs: Iterable[object]) -> List[dict]:
    """Score a batch of UTR strings or transcript-like records."""
    return _DEFAULT_ORACLE.batch_score(records_or_utrs)


def save_scores_json(scores: Sequence[Mapping[str, object]], path: str) -> None:
    """Small convenience writer used by offline benchmark scripts."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(list(scores), fh, indent=2, sort_keys=True)


__all__ = [
    "FivePrimeCNN",
    "LocalTranslationOracle",
    "OracleScore",
    "extract_utr_features",
    "score_utr",
    "score_record",
    "batch_score",
    "save_scores_json",
]
