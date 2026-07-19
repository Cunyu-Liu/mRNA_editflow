"""mRNABench-style linear-probe scaffold with offline synthetic fallback.

mRNABench comparisons usually probe pretrained mRNA representations on labelled
downstream tasks. Local smoke tests do not ship the benchmark data or external
encoders, so this module provides the minimal honest scaffold:

* deterministic handcrafted transcript features,
* a real trainable linear probe,
* synthetic labels when benchmark labels are absent,
* explicit trainable-parameter accounting.

The API is intentionally compatible with later replacement of
``record_feature_vector`` by frozen encoder embeddings.

Complexity: feature extraction is ``O(total_sequence_length)``; linear-probe
training is ``O(steps * N * feature_dim * num_labels)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mrna_editflow.core.constants import STOP_CODONS
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import synthesize_corpus


FEATURE_NAMES = (
    "length_scaled",
    "five_utr_fraction",
    "cds_fraction",
    "three_utr_fraction",
    "gc_fraction",
    "a_fraction",
    "c_fraction",
    "g_fraction",
    "u_fraction",
    "kozak_gccacc",
    "uaug_per_100nt",
    "poly_a_signal",
    "stop_uaa",
    "stop_uag",
    "stop_uga",
)
"""Feature order used by :func:`record_feature_vector`."""


@dataclass
class MRNABenchProbeConfig:
    """Configuration for the offline mRNABench linear probe.

    Complexity: construction is ``O(1)``.
    """

    num_labels: int = 2
    lr: float = 5e-2
    weight_decay: float = 0.0
    steps: int = 20
    seed: int = 0
    synthetic_size: int = 8


@dataclass
class MRNABenchProbeResult:
    """Return object from :func:`run_mrnabench_probe`.

    ``mode`` is ``"synthetic_fallback"`` when no records are supplied. The
    parameter count covers only trainable probe weights, matching linear-probe
    reporting practice.

    Complexity: stores ``O(steps)`` scalar losses.
    """

    model: "MRNABenchLinearProbe"
    mode: str
    n_records: int
    feature_dim: int
    num_labels: int
    trainable_params: int
    losses: List[float]
    final_loss: float


class MRNABenchLinearProbe(nn.Module):
    """Single linear classifier/regressor head for mRNA benchmark probing.

    The current scaffold implements classification via cross-entropy. Later
    benchmark-specific heads can keep this parameter-count API.

    Complexity: ``O(B * feature_dim * num_labels)`` per forward pass.
    """

    def __init__(self, feature_dim: int, num_labels: int):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_labels <= 1:
            raise ValueError("num_labels must be at least 2")
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.linear = nn.Linear(self.feature_dim, self.num_labels)

    def forward(self, features: Tensor) -> Tensor:
        """Return class logits. Complexity: one linear projection."""
        return self.linear(features.float())


def record_feature_vector(record: MRNARecord) -> List[float]:
    """Extract deterministic transcript-level features for one record.

    Features are deliberately simple and auditable: length/region fractions,
    nucleotide composition and common regulatory motif indicators.

    Complexity: ``O(len(record.seq))``.
    """
    seq = record.seq
    length = max(len(seq), 1)
    counts = {nt: seq.count(nt) for nt in "ACGU"}
    gc = (counts["G"] + counts["C"]) / length
    stop = record.cds[-3:] if len(record.cds) >= 3 else ""
    return [
        min(length, 4096) / 4096.0,
        len(record.five_utr) / length,
        len(record.cds) / length,
        len(record.three_utr) / length,
        gc,
        counts["A"] / length,
        counts["C"] / length,
        counts["G"] / length,
        counts["U"] / length,
        1.0 if record.five_utr.endswith("GCCACC") else 0.0,
        100.0 * record.five_utr.count("AUG") / max(len(record.five_utr), 1),
        1.0 if "AAUAAA" in record.three_utr or "AUUAAA" in record.three_utr else 0.0,
        1.0 if stop == "UAA" else 0.0,
        1.0 if stop == "UAG" else 0.0,
        1.0 if stop == "UGA" else 0.0,
    ]


def feature_matrix(records: Sequence[MRNARecord]) -> Tensor:
    """Return ``[N, feature_dim]`` handcrafted features. Complexity: ``O(total L)``."""
    if not records:
        raise ValueError("records must be non-empty")
    return torch.tensor([record_feature_vector(r) for r in records], dtype=torch.float32)


def synthetic_mrnabench_labels(records: Sequence[MRNARecord], num_labels: int = 2) -> Tensor:
    """Create deterministic offline labels when mRNABench labels are absent.

    The binary default marks transcripts with a strong Kozak motif or balanced
    GC content as positives. For ``num_labels > 2`` the score is bucketed.

    Complexity: ``O(total L)``.
    """
    if num_labels <= 1:
        raise ValueError("num_labels must be at least 2")
    labels: List[int] = []
    for rec in records:
        seq = rec.seq
        length = max(len(seq), 1)
        gc = (seq.count("G") + seq.count("C")) / length
        kozak = 1.0 if rec.five_utr.endswith("GCCACC") else 0.0
        poly_a = 1.0 if "AAUAAA" in rec.three_utr or "AUUAAA" in rec.three_utr else 0.0
        valid_stop = 1.0 if rec.cds[-3:] in STOP_CODONS else 0.0
        score = 0.45 * (1.0 - min(abs(gc - 0.52) / 0.52, 1.0)) + 0.25 * kozak
        score += 0.15 * poly_a + 0.15 * valid_stop
        bucket = min(num_labels - 1, int(score * num_labels))
        labels.append(bucket)
    return torch.tensor(labels, dtype=torch.long)


def count_trainable_params(module: nn.Module) -> int:
    """Count parameters with ``requires_grad=True``. Complexity: ``O(parameters)``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def run_mrnabench_probe(
    records: Optional[Sequence[MRNARecord]] = None,
    labels: Optional[Sequence[int]] = None,
    config: Optional[MRNABenchProbeConfig] = None,
    device: str = "cpu",
) -> MRNABenchProbeResult:
    """Train the offline mRNABench-style linear probe.

    When ``records`` is ``None`` a deterministic synthetic corpus is generated.
    When ``labels`` is ``None`` deterministic synthetic labels are used. This
    keeps the benchmark scaffold runnable without external files while preserving
    a real trainable optimisation path.

    Complexity: ``O(feature_extraction + steps * N * feature_dim * labels)``.
    """
    cfg = config or MRNABenchProbeConfig()
    torch.manual_seed(cfg.seed)
    mode = "provided_records"
    if records is None:
        records = synthesize_corpus(cfg.synthetic_size, seed=cfg.seed)
        mode = "synthetic_fallback"
    records = list(records)
    if not records:
        raise ValueError("records must be non-empty")

    x = feature_matrix(records)
    if labels is None:
        y = synthetic_mrnabench_labels(records, num_labels=cfg.num_labels)
        if mode != "synthetic_fallback":
            mode = "synthetic_labels"
    else:
        if len(labels) != len(records):
            raise ValueError("labels length must match records length")
        y = torch.tensor(list(labels), dtype=torch.long)
    if int(y.min().item()) < 0 or int(y.max().item()) >= cfg.num_labels:
        raise ValueError("labels must be in [0, num_labels)")

    dev = torch.device(device)
    x = x.to(dev)
    y = y.to(dev)
    model = MRNABenchLinearProbe(feature_dim=x.shape[1], num_labels=cfg.num_labels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses: List[float] = []

    model.train()
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite mRNABench probe loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    return MRNABenchProbeResult(
        model=model,
        mode=mode,
        n_records=len(records),
        feature_dim=int(x.shape[1]),
        num_labels=cfg.num_labels,
        trainable_params=count_trainable_params(model),
        losses=losses,
        final_loss=losses[-1],
    )


__all__ = [
    "FEATURE_NAMES",
    "MRNABenchProbeConfig",
    "MRNABenchProbeResult",
    "MRNABenchLinearProbe",
    "record_feature_vector",
    "feature_matrix",
    "synthetic_mrnabench_labels",
    "count_trainable_params",
    "run_mrnabench_probe",
]
